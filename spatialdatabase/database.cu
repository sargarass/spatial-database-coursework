#include "database.h"
#include <cub/cub/cub.cuh>


__device__
void gpuRowGpuToGpu(gpudb::GpuRow * dst, gpudb::GpuRow const * src, uint64_t const memsize) {
    memcpy(dst, src, memsize);
    dst->spatialPart.key = newAddress(dst->spatialPart.key, src, dst);
    switch (dst->spatialPart.type) {
        case SpatialType::POINT:
        break;
        case SpatialType::LINE:
        {
            gpudb::GpuLine *line = ((gpudb::GpuLine*)(dst->spatialPart.key));
            line->points = newAddress(line->points, src, dst);
        }
        break;
        case SpatialType::POLYGON:
        {
            gpudb::GpuPolygon *polygon = ((gpudb::GpuPolygon*)(dst->spatialPart.key));
            polygon->points = newAddress(polygon->points, src, dst);
        }
        break;
    }

    dst->value = newAddress(dst->value, src, dst);

    for (uint i = 0; i < dst->valueSize; i++) {
        dst->value[i].value = newAddress(dst->value[i].value, src, dst);
    }
}

__global__
void gpuRowsCopy(gpudb::GpuRow **dst, gpudb::GpuRow * const *src, uint64_t *sizes, uint count) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= count) { return; }

    gpuRowGpuToGpu(dst[idx], src[idx], sizes[idx]);
}

__global__
void gpuRowsCopyOnlySelected(gpudb::GpuRow **dst, gpudb::GpuRow * const *src, uint *selectors, uint64_t *sizes, uint count) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= count) { return; }
    gpuRowGpuToGpu(dst[idx], src[selectors[idx]], sizes[idx]);
}

bool DataBase::copyTempTable(TableDescription const &description, gpudb::GpuTable const *gpuTable, TempTable &table) {
    if (gpuTable->rows.size() == 0) {
        return false;
    }

    TempTable result;
    result.description = description;
    result.table = new gpudb::GpuTable;

    if (result.table == nullptr) {
        return false;
    }

    result.table->columns.reserve(gpuTable->columns.size());
    result.table->columns = gpuTable->columns;
    memcpy(result.table->name, gpuTable->name, NAME_MAX_LEN * sizeof(char));
    result.table->rows.reserve(gpuTable->rows.size());
    thrust::host_vector<gpudb::GpuRow *> hostRows(gpuTable->rows.size());
    bool success = true;
    StackAllocator::getInstance().pushPosition();
    gpudb::GpuStackAllocator::getInstance().pushPosition();

    do {
        for (size_t i = 0; i < hostRows.size(); i++) {
            hostRows[i] = (gpudb::GpuRow*) gpudb::gpuAllocator::getInstance().alloc<uint8_t>(gpuTable->rowsSize[i]);
        }
        uint64_t * sizes = gpudb::GpuStackAllocator::getInstance().alloc<uint64_t>(hostRows.size());
        if (success == false || sizes == nullptr) {
            break;
        }
        cudaMemcpy(sizes, gpuTable->rowsSize.data(), sizeof(uint64_t) * hostRows.size(), cudaMemcpyHostToDevice);
        result.table->rows = hostRows;
        result.table->rowsSize = gpuTable->rowsSize;

        dim3 grid = gridConfigure(hostRows.size(), BLOCK_SIZE);
        dim3 block = dim3(BLOCK_SIZE);

        gpuRowsCopy<<<grid, block>>>(thrust::raw_pointer_cast(result.table->rows.data()),
                                     thrust::raw_pointer_cast(gpuTable->rows.data()),
                                     sizes,
                                     hostRows.size());
    }
    while(0);
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();

    if (success == false) {
        for (size_t i = 0; i < hostRows.size(); i++) {
            if (hostRows[i] != nullptr) {
                gpudb::gpuAllocator::getInstance().free(hostRows[i]);
            }
        }
        result.table->rows.clear();
        delete result.table;
        return false;
    }
    result.valid = true;
    table = result;

    result.valid = false;
    result.parents.clear();
    result.needToBeFree.clear();
    result.table = nullptr;
    return true;
}

bool DataBase::selectTable(std::string tableName, TempTable &table) {
    auto descriptionIt = tablesType.find(tableName);
    auto tableIt = tables.find(tableName);

    if (descriptionIt == tablesType.end() || tableIt == tables.end()) {
        return false;
    }


    TableDescription &refTableDescription = descriptionIt->second;
    gpudb::GpuTable *pointerGpuTable = tableIt->second;

    return copyTempTable(refTableDescription, pointerGpuTable, table);
}

__global__
void buildKeysAABB(gpudb::GpuRow** rows, gpudb::AABB *boxes, uint size) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= size) {
        return;
    }

    rows[idx]->spatialPart.boundingBox(&boxes[idx]);
    AABBmin(boxes[idx].z) = AABBmax(boxes[idx].z) = 0.0f;
    AABBmin(boxes[idx].w) = AABBmax(boxes[idx].w) = 0.0f;
    //rows[idx]->temporalPart.boundingBox(&boxes[idx]);
}

__device__
AABBRelation boxIntersection2D(float4 aMin, float4 aMax, float4 bMin, float4 bMax) {
    bool a1 = aMax.x < bMin.x;
    bool a2 = aMin.x > bMax.x;

    bool a3 = aMax.y < bMin.y;
    bool a4 = aMin.y > bMax.y;

   /* bool a5 = aMax.z < bMin.z;
    bool a6 = aMin.z > bMax.z;

    bool a7 = aMax.w < bMin.w;
    bool a8 = aMin.w > bMax.w;*/

    if (a1 || a2 || a3 || a4 /*&& a5 &&a6 && a7 && a8*/) {
        return AABBRelation::DISJOINT;
    }

    if (aMin.x <= bMin.x && bMin.x <= aMax.x &&
        aMin.x <= bMax.x && bMax.x <= aMax.x &&
        aMin.y <= bMin.y && bMin.y <= aMax.y &&
        aMin.y <= bMax.y && bMax.y <= aMax.y /*&&
        aMin.z <= bMin.z && bMin.z <= aMax.z &&
        aMin.z <= bMax.z && bMax.z <= aMax.z &&
        aMin.w <= bMin.w && bMin.w <= aMax.w &&
        aMin.w <= bMax.w && bMax.w <= aMax.w*/) {
        return AABBRelation::BINSIDEA;
    }

    return AABBRelation::OVERLAP;
}


__device__
void computeBoundingBoxLineBuffer(gpudb::GpuLine *line, float radius, float2 &min2, float2 &max2) {
    min2 = make_float2(INFINITY, INFINITY);
    max2 = make_float2(-INFINITY, -INFINITY);

    float2 s = line->points[0];
    float2 e = line->points[1]; // предпологаем что линия не из одной точки

    float2 dir = e - s;
    dir = norma(dir);
    float2 n = make_float2(dir.y, -dir.x);
    float2 resizedN = n * radius;
    float2 resizedDir = dir * radius;

    float2 p1 = s - resizedDir;

    min2 = fmin(p1, min2);
    max2 = fmax(p1, max2);

    float2 p1_1 = p1 + resizedN;
    float2 p1_2 = p1 - resizedN;

    min2 = fmin(p1_1, min2);
    min2 = fmin(p1_2, min2);
    max2 = fmax(p1_1, max2);
    max2 = fmax(p1_2, max2);

    for (uint i = 0; i < line->size - 1; i++) {
        s = line->points[i];
        e = line->points[i + 1];
        dir = e - s;
        dir = norma(dir);
        n = make_float2(dir.y, -dir.x);
        resizedN = n * radius;
        resizedDir = dir * radius;
        p1 = e + resizedDir;
        p1_1 = p1 + resizedN;
        p1_2 = p1 - resizedN;
        min2 = fmin(p1, min2);
        max2 = fmax(p1, max2);
        min2 = fmin(p1_1, min2);
        min2 = fmin(p1_2, min2);
        max2 = fmax(p1_1, max2);
        max2 = fmax(p1_2, max2);
    }
}

__device__
void boxInsideBoxSubKernel(float4 aabbMin, float4 aabbMax, uint &numBoxBinside, gpudb::HLBVH &bvh, uint *stack, uint stackSize) {
    GpuStack<uint> st(stack, stackSize);
    st.push(0);
    uint sum = 0;
    while(!st.empty()) {
        uint pos = st.top(); st.pop();
        float4 aMax = bvh.aabbMax[pos];
        float4 aMin = bvh.aabbMin[pos];
        float4 bMax = bvh.aabbMax[pos + 1];
        float4 bMin = bvh.aabbMin[pos + 1];
        int link1 = bvh.links[pos];
        int link2 = bvh.links[pos + 1];
        AABBRelation r1 = boxIntersection2D(aabbMin, aabbMax, aMin, aMax);
        AABBRelation r2 = boxIntersection2D(aabbMin, aabbMax, bMin, bMax);

        if (r1 == AABBRelation::OVERLAP && link1 != LEAF) {
            st.push(link1);
        }

        if (r2 == AABBRelation::OVERLAP && link2 != LEAF) {
            st.push(link2);
        }

        if (r1 == AABBRelation::BINSIDEA || (r1 == AABBRelation::OVERLAP && link1 == LEAF)) {
            sum += getRangeSize(bvh.ranges[pos]);
        }

        if (r2 == AABBRelation::BINSIDEA || (r2 == AABBRelation::OVERLAP && link2 == LEAF)) {
            sum += getRangeSize(bvh.ranges[pos + 1]);
        }
    }
    numBoxBinside = sum;
}

__device__
void boxInsideBoxSubKernel(uint idx, float4 aabbMin, float4 aabbMax, uint numBoxBInside, uint offsets, uint *testedBoxANum,
                           uint *testedBoxBNum, gpudb::HLBVH &bvh, uint *stack, uint stackSize) {
    GpuStack<uint> st(stack, stackSize);
    uint num = numBoxBInside;
    uint pointCounter = 0;
    st.push(0);
    while(!st.empty() && pointCounter < num) {
        uint pos = st.top(); st.pop();
        float4 aMax = bvh.aabbMax[pos];
        float4 aMin = bvh.aabbMin[pos];
        float4 bMax = bvh.aabbMax[pos + 1];
        float4 bMin = bvh.aabbMin[pos + 1];
        int link1 = bvh.links[pos];
        int link2 = bvh.links[pos + 1];
        AABBRelation r1 = boxIntersection2D(aabbMin, aabbMax, aMin, aMax);
        AABBRelation r2 = boxIntersection2D(aabbMin, aabbMax, bMin, bMax);

        if (r1 == AABBRelation::OVERLAP && link1 != LEAF) {
            st.push(link1);
        }

        if (r2 == AABBRelation::OVERLAP && link2 != LEAF) {
            st.push(link2);
        }

        if ((r1 == AABBRelation::BINSIDEA) || (r1 == AABBRelation::OVERLAP && link1 == LEAF)) {
            for (uint i = getLeftBound(bvh.ranges[pos]); i < getRightBound(bvh.ranges[pos]); i++) {
                testedBoxBNum[offsets + pointCounter] = bvh.references[i];
                testedBoxANum[offsets + pointCounter] = idx;
                pointCounter++;
            }
        }

        if ((r2 == AABBRelation::BINSIDEA) || (r2 == AABBRelation::OVERLAP && link2 == LEAF)) {
            for (uint i = getLeftBound(bvh.ranges[pos + 1]); i < getRightBound(bvh.ranges[pos + 1]); i++) {
                testedBoxBNum[offsets + pointCounter] = bvh.references[i];
                testedBoxANum[offsets + pointCounter] = idx;
                pointCounter++;
            }
        }
    }
}

__global__
void boxInsideBoxKernel(gpudb::GpuRow **rowsBoxA, uint *numBoxBinside, gpudb::HLBVH bvh, uint *stack, uint stackSize, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }
    gpudb::AABB box;
    rowsBoxA[idx]->spatialPart.boundingBox(&box);
    rowsBoxA[idx]->temporalPart.boundingBox(&box);

    float4 min = make_float4(box.x.x, box.y.x, box.z.x, box.w.x);
    float4 max = make_float4(box.x.y, box.y.y, box.z.y, box.w.y);

    boxInsideBoxSubKernel(min, max, numBoxBinside[idx], bvh, &stack[idx * stackSize], stackSize);
}

__global__
void boxInsideBoxKernel2(gpudb::GpuRow **rowsBoxA, uint *numBoxBInside, uint *offsets, uint *testedBoxANum, uint *testedBoxBNum,
                         gpudb::HLBVH bvh, uint *stack, uint stackSize, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }
    gpudb::AABB box;
    rowsBoxA[idx]->spatialPart.boundingBox(&box);
    rowsBoxA[idx]->temporalPart.boundingBox(&box);

    float4 min = make_float4(box.x.x, box.y.x, box.z.x, box.w.x);
    float4 max = make_float4(box.x.y, box.y.y, box.z.y, box.w.y);

    boxInsideBoxSubKernel(idx, min, max, numBoxBInside[idx], offsets[idx], testedBoxANum, testedBoxBNum, bvh, &stack[idx * stackSize], stackSize);
}


__global__
void boxInsideBoxLineKernel(gpudb::GpuRow **rowsBoxA, uint *numBoxBinside, gpudb::HLBVH bvh, uint *stack, uint stackSize, uint workSize, float radius) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }
    gpudb::AABB box;
    gpudb::GpuLine *line = (gpudb::GpuLine *)rowsBoxA[idx]->spatialPart.key;
    float2 min2, max2;

    computeBoundingBoxLineBuffer(line, radius, min2, max2);
    rowsBoxA[idx]->temporalPart.boundingBox(&box);

    float4 min = make_float4(min2.x, min2.y, box.z.x, box.w.x);
    float4 max = make_float4(max2.x, max2.y, box.z.y, box.w.y);

    boxInsideBoxSubKernel(min, max, numBoxBinside[idx], bvh, &stack[idx * stackSize], stackSize);
}

__global__
void boxInsideBoxLineKernel2(gpudb::GpuRow **rowsBoxA, uint *numBoxBInside, uint *offsets, uint *testedBoxANum, uint *testedBoxBNum,
                             gpudb::HLBVH bvh, uint *stack, uint stackSize, uint workSize, float radius) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }
    gpudb::AABB box;
    gpudb::GpuLine *line = (gpudb::GpuLine *)rowsBoxA[idx]->spatialPart.key;
    float2 min2, max2;

    computeBoundingBoxLineBuffer(line, radius, min2, max2);
    rowsBoxA[idx]->temporalPart.boundingBox(&box);

    float4 min = make_float4(min2.x, min2.y, box.z.x, box.w.x);
    float4 max = make_float4(max2.x, max2.y, box.z.y, box.w.y);

    boxInsideBoxSubKernel(idx, min, max, numBoxBInside[idx], offsets[idx], testedBoxANum, testedBoxBNum, bvh, &stack[idx * stackSize], stackSize);
}


__device__
float isLeft(float2 s, float2 e, float2 p) {
    return (e.x - s.x) * (p.y - s.y) - (p.x - s.x) * (e.y - s.y);
}

__global__
void pointInsidePolygonKernel(gpudb::GpuRow **polygons, uint *numPoints, uint *testedPolygonNum, uint *testedPointNum,
                              uint *testResult, gpudb::GpuRow **points, uint sizeWork) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= sizeWork) {
        return;
    }

    gpudb::GpuPolygon *polygon = (gpudb::GpuPolygon*)polygons[testedPolygonNum[idx]]->spatialPart.key;
    gpudb::GpuPoint *point = (gpudb::GpuPoint*)points[testedPointNum[idx]]->spatialPart.key;
    float2 p = point->p;
    int wn = 0;
    for (uint i = 0; i < polygon->size; i++) {
        float2 s = polygon->points[i];
        float2 e = polygon->points[((i + 1) < polygon->size)? i + 1 : 0];

        float left = isLeft(s, e, p);
        if (s.y <= p.y && e.y > p.y && left > 0) {
            wn++;
        } else {
            if (s.y > p.y && e.y < p.y && left < 0) {
                wn--;
            }
        }
    }
    testResult[idx] = wn != 0;
    /*if (wn != 0) {
        printf("{%f %f}\n", p.x, p.y);
    }*/
}

__device__
bool insideRect(float2 A, float2 B, float2 C, float2 D, float2 point) {
    float2 dir = B - A;
    float2 dirToP = point - A;
    float dot1 = dot(dir, dirToP);

    float2 dir2 = C - B;
    float2 dirToP2 = point - B;
    float dot2 = dot(dir2, dirToP2);

    float2 dir3 = D - C;
    float2 dirToP3 = point - C;
    float dot3 = dot(dir3, dirToP3);

    float2 dir4 = A - D;
    float2 dirToP4 = point - D;
    float dot4 = dot(dir4, dirToP4);

    return ((dot1 > 0) && (dot2 > 0) && (dot3 > 0) && (dot4 > 0)) ||
           ((dot1 < 0) && (dot2 < 0) && (dot3 < 0) && (dot4 < 0));
}

__global__
void pointInsideLineBufferKernel(gpudb::GpuRow **lines, uint *numPoints, uint *testedLineNum, uint *testedPointNum, uint *testResult,
                                 gpudb::GpuRow **points, uint sizeWork, float radius) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= sizeWork) {
        return;
    }

    gpudb::GpuLine *line = (gpudb::GpuLine *)lines[testedLineNum[idx]]->spatialPart.key;
    gpudb::GpuPoint *point = (gpudb::GpuPoint *)points[testedPointNum[idx]]->spatialPart.key;
    float2 p = point->p;
    float2 s = line->points[0];
    float2 e;

    bool result = (lenSqr(s, p) < radius * radius);
    for (uint i = 0; i < line->size - 1; i++) {
        s = line->points[i];
        e = line->points[i + 1];
        result = result || (lenSqr(e, p) < radius * radius);

        float2 dir = e - s;
        dir = norma(dir);
        float2 n = make_float2(dir.y, -dir.x);

        float2 A, B, C, D;
        float2 resizedN = n * radius;
        A = resizedN + s;
        B = resizedN + e;
        C = e - resizedN;
        D = s - resizedN;
        // это даёт ориентацию либо по часовой, либо против часовой стрелки
        result = result || insideRect(A, B, C, D, p);
        if (result) { break; }
    }

    /*if (result) {
        printf("%d %f %f\n", testedPointNum[idx], p.x, p.y);
    }*/

    testResult[idx] = result;
}

__global__
void computeResultSize(uint *result, uint *resultPrefixSum, uint *testedPrefixSum, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    uint offset = resultPrefixSum[idx + 1]; // сумма a0 + ... + a_idx
    result[idx] = testedPrefixSum[offset];
}

__global__
void computeResultSize2(uint *result, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    uint sub = 0;
    if (idx > 0) {
        sub = result[idx - 1];
    }

    result[idx] = result[idx] - sub;
}


__global__
void selector(uint *dsc, uint *pointNum, uint *tests, uint *offset, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    if (tests[idx]) {
        dsc[offset[idx]] = pointNum[idx];
    }
}

bool DataBase::resultToTempTable2(TempTable const &sourceA, TempTable &sourceB, std::string nameForNewTempTebles,
                                 TempTable **newTempTables, std::string nameForResultTempTable, TempTable &resultTempTable) {
    gpudb::GpuTable *resultTable = new gpudb::GpuTable;

    if (resultTable == nullptr) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought memory");
        return false;
    }

    StackAllocator::getInstance().pushPosition();
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    do {
        gpudb::GpuColumnAttribute atr;
        std::snprintf(atr.name, NAME_MAX_LEN, nameForNewTempTebles.c_str());
        atr.type = Type::SET;

        AttributeDescription desc;
        desc.name.resize(NAME_MAX_LEN);
        std::snprintf(&desc.name[0], NAME_MAX_LEN, nameForNewTempTebles.c_str());
        desc.type = Type::SET;

        resultTable->bvh.builded = false;
        std::snprintf(resultTable->name, NAME_MAX_LEN, nameForResultTempTable.c_str());
        resultTable->rows.reserve(sourceA.table->rows.size());
        resultTable->rowsSize.resize(sourceA.table->rows.size());
        resultTable->columns.reserve(sourceA.table->columns.size() + 1);
        resultTable->columns = sourceA.table->columns;
        resultTable->columns.push_back(atr);

        TableDescription tdescription;
        tdescription = sourceA.description;
        tdescription.columnDescription.push_back(desc);
        tdescription.name = nameForResultTempTable;
        uint8_t **resultRows = StackAllocator::getInstance().alloc<uint8_t*>(sourceA.table->rows.size());

        if (resultRows == nullptr) {
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought cpu memory");
            break;
        }

        thrust::host_vector<gpudb::GpuRow*> hostRowsResult;
        hostRowsResult.resize(sourceA.table->rows.size());

        uint64_t maxSize = 0;
        for (uint i = 0; i < sourceA.table->rows.size(); i++) {
            uint64_t memsize = sourceA.table->rowsSize[i] + sizeof(gpudb::Value) + typeSize(Type::SET);
            maxSize = std::max(memsize, maxSize);

            resultTable->rowsSize[i] = memsize;
            resultRows[i] = gpudb::gpuAllocator::getInstance().alloc<uint8_t>(memsize);
            hostRowsResult[i] = ((gpudb::GpuRow*)(resultRows[i]));

            if (resultRows[i] == nullptr) {
                gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought memory");
                for (int j = 0; j < i; j++) {
                    gpudb::gpuAllocator::getInstance().free(resultRows[j]);
                }
                goto error;
            }

        }

        uint8_t *cpuRow = StackAllocator::getInstance().alloc<uint8_t>(maxSize);
        if (cpuRow == nullptr) {
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought cpu memory");
            for (int j = 0; j < sourceA.table->rows.size(); j++) {
                gpudb::gpuAllocator::getInstance().free(resultRows[j]);
            }
            break;
        }

        resultTable->rows = hostRowsResult;

        thrust::host_vector<gpudb::GpuRow*> hostRows = sourceA.table->rows;
        for (uint i = 0; i < sourceA.table->rows.size(); i++) {
            uint8_t *aRow = StackAllocator::getInstance().alloc<uint8_t>(sourceA.table->rowsSize[i]);
            DataBase::getInstance().loadCPU((gpudb::GpuRow*)aRow, hostRows[i], sourceA.table->rowsSize[i]);
            gpudb::GpuRow* cpuRowPointer = ((gpudb::GpuRow*)cpuRow);
            gpudb::GpuRow* aCpuRowPointer = ((gpudb::GpuRow*)aRow);
            uintptr_t cpuRawPointer = ((uintptr_t)cpuRow);

            strncpy(cpuRowPointer->spatialPart.name, aCpuRowPointer->spatialPart.name, typeSize(Type::STRING));
            cpuRowPointer->spatialPart.name[typeSize(Type::STRING) - 1] = 0;
            strncpy(cpuRowPointer->temporalPart.name, aCpuRowPointer->temporalPart.name, typeSize(Type::STRING));
            cpuRowPointer->temporalPart.name[typeSize(Type::STRING) - 1] = 0;

            cpuRowPointer->spatialPart.type = aCpuRowPointer->spatialPart.type;
            cpuRowPointer->temporalPart.type = sourceA.description.temporalKeyType;
            cpuRowPointer->valueSize = tdescription.columnDescription.size();
            cpuRowPointer->value = (gpudb::Value*)(cpuRawPointer + sizeof(gpudb::GpuRow));

            uintptr_t memoryValues = cpuRawPointer + sizeof(gpudb::GpuRow) + sizeof(gpudb::Value) * cpuRowPointer->valueSize;
            for (uint j = 0; j < tdescription.columnDescription.size(); j++) {
                cpuRowPointer->value[j].value = (void*)memoryValues;
                if (j < sourceA.description.columnDescription.size()) {
                    cpuRowPointer->value[j].isNull = aCpuRowPointer->value[j].isNull;
                } else {
                    cpuRowPointer->value[j].isNull = false;
                }

                uint64_t attrSize = typeSize(tdescription.columnDescription[j].type);
                if (j < sourceA.description.columnDescription.size()) {
                    memcpy(cpuRowPointer->value[j].value, aCpuRowPointer->value[j].value, attrSize);
                } else {
                    gpudb::GpuSet set;
                    set.temptable = newTempTables[i];
                    set.columns = thrust::raw_pointer_cast(newTempTables[i]->table->columns.data());
                    set.rows = thrust::raw_pointer_cast(newTempTables[i]->table->rows.data());
                    set.rowsSize = newTempTables[i]->table->rows.size();
                    set.columnsSize = newTempTables[i]->table->columns.size();

                    memcpy(cpuRowPointer->value[j].value, &set, attrSize);
                }
                memoryValues += attrSize;
            }

            cpuRowPointer->spatialPart.key = (void*)memoryValues;
            switch (tdescription.spatialKeyType) {
                case SpatialType::POINT:
                {
                    gpudb::GpuPoint *p1 = ((gpudb::GpuPoint*)(cpuRowPointer->spatialPart.key));
                    gpudb::GpuPoint *p2 = ((gpudb::GpuPoint*)(aCpuRowPointer->spatialPart.key));
                    p1->p = p2->p;
                }
                break;
                case SpatialType::LINE:
                {
                    gpudb::GpuLine *l1 = ((gpudb::GpuLine*)(cpuRowPointer->spatialPart.key));
                    gpudb::GpuLine *l2 = ((gpudb::GpuLine*)(aCpuRowPointer->spatialPart.key));
                    l1->size = l2->size;
                    l1->points = (float2*)(memoryValues + sizeof(gpudb::GpuLine));
                    for (int i = 0; i < l2->size; i++) {
                        l1->points[i] = l2->points[i];
                    }
                }
                break;
                case SpatialType::POLYGON:
                {
                    gpudb::GpuPolygon *p1 = ((gpudb::GpuPolygon*)(cpuRowPointer->spatialPart.key));
                    gpudb::GpuPolygon *p2 = ((gpudb::GpuPolygon*)(aCpuRowPointer->spatialPart.key));
                    p1->size = p2->size;
                    p1->points = (float2*)(memoryValues + sizeof(gpudb::GpuPolygon));
                    for (int i = 0; i < p2->size; i++) {
                        p1->points[i] = p2->points[i];
                    }
                }
                break;
            }

            cpuRowPointer->temporalPart.transactionTimeCode = aCpuRowPointer->temporalPart.transactionTimeCode;
            cpuRowPointer->temporalPart.validTimeECode = aCpuRowPointer->temporalPart.validTimeECode;
            cpuRowPointer->temporalPart.validTimeSCode = aCpuRowPointer->temporalPart.validTimeSCode;

            DataBase::getInstance().storeGPU((gpudb::GpuRow*)resultRows[i], cpuRowPointer, resultTable->rowsSize[i]);
            StackAllocator::getInstance().free(aRow);
        }
        sourceB.references.push_back(&resultTempTable);
        resultTempTable.parents.push_back(&sourceB);

        resultTempTable.table = resultTable;
        resultTempTable.valid = true;
        resultTempTable.description = tdescription;

        StackAllocator::getInstance().popPosition();
        gpudb::GpuStackAllocator::getInstance().popPosition();
        return true;
    } while(0);

    error:
    delete resultTable;
    StackAllocator::getInstance().popPosition();
    gpudb::GpuStackAllocator::getInstance().popPosition();
    return false;
}

bool DataBase::resultToTempTable1(TempTable const &a, TempTable &b, std::string opname, uint *selectedRowsFromB, uint *selectedRowsSize, TempTable &resultTempTable) {
    StackAllocator::getInstance().pushPosition();
    gpudb::GpuStackAllocator::getInstance().pushPosition();

    do {
        gpudb::GpuTable **tables = StackAllocator::getInstance().alloc<gpudb::GpuTable*>(a.table->rows.size());
        TempTable **newTempTables = StackAllocator::getInstance().alloc<TempTable*>(a.table->rows.size());

        for (uint i = 0; i < a.table->rows.size(); i++) {
            tables[i] = new gpudb::GpuTable;
            newTempTables[i] = new TempTable;

            resultTempTable.needToBeFree.push_back((uintptr_t)newTempTables[i]);

            if (tables[i] == nullptr || newTempTables == nullptr) {
                for (int j = 0; j <= i; j++) {
                    if (tables[j]) {
                        delete tables[j];
                    }
                    if (newTempTables[j]) {
                        delete newTempTables[j];
                    }
                }

                resultTempTable.needToBeFree.clear();
                goto error;
            }
        }

        thrust::host_vector<gpudb::GpuRow*> rows;
        thrust::host_vector<gpudb::GpuRow*> brows = b.table->rows;

        size_t j = 0;
        for (size_t i = 0; i < a.table->rows.size(); i++) {

            tables[i]->columns.reserve(a.table->columns.size());
            tables[i]->bvh.builded = false;
            tables[i]->columns = a.table->columns;

            memcpy(tables[i]->name, a.table->name, NAME_MAX_LEN * sizeof(char));
            tables[i]->rowReferenses = true;

            if (selectedRowsSize[i] > 0) {
                rows.resize(selectedRowsSize[i]);
                tables[i]->rowsSize.resize(selectedRowsSize[i]);
                tables[i]->rows.reserve(selectedRowsSize[i]);

                for (size_t p = 0; p < selectedRowsSize[i]; p++, j++) {
                    tables[i]->rowsSize[p] = b.table->rowsSize[selectedRowsFromB[j]];
                    rows[p] = brows[selectedRowsFromB[j]];
                }

                tables[i]->rows = rows;
            }
        }

        for (uint i = 0; i < a.table->rows.size(); i++) {
            newTempTables[i]->description = b.description;
            newTempTables[i]->valid = true;
            newTempTables[i]->table = tables[i];
        }

        if (!resultToTempTable2(a, b, opname.c_str(), newTempTables, opname.c_str(), resultTempTable)) {
            for (int j = 0; j < a.table->rows.size(); j++) {
                if (tables[j]) {
                    delete tables[j];
                }
            }
            break;
        }

        StackAllocator::getInstance().popPosition();
        gpudb::GpuStackAllocator::getInstance().popPosition();
        return true;
    } while (0);
    error:
    StackAllocator::getInstance().popPosition();
    gpudb::GpuStackAllocator::getInstance().popPosition();
    return false;
}

TempTable DataBase::linexpointPointsInBufferLine(TempTable const &a, TempTable &b, float radius) {
    TempTable resultTempTable;

    if (!a.isValid() || !b.isValid()) {
        return resultTempTable;
    }

    if (a.table == nullptr ||
        b.table == nullptr ||
        a.getSpatialKeyType() != SpatialType::LINE ||
        b.getSpatialKeyType() != SpatialType::POINT ||
        radius <= 0.0001f ||
        a.table->rows.size() == 0 || b.table->rows.size() == 0) {
        return resultTempTable;
    }

    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();

    do {
        if (!b.table->bvh.isBuilded()) {

            gpudb::AABB * boxes = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::AABB> (b.table->rows.size());
            if (boxes == nullptr) {
                break;
            }

            dim3 block(BLOCK_SIZE);
            dim3 grid(gridConfigure(b.table->rows.size(), block));
            buildKeysAABB<<<grid, block>>>(thrust::raw_pointer_cast(b.table->rows.data()), boxes, b.table->rows.size());
            if (b.table->bvh.build(boxes, b.table->rows.size()) == false) {
                break;
            }
            gpudb::GpuStackAllocator::getInstance().free(boxes);
        }

        uint stackSize = b.table->bvh.numBVHLevels * 2 + 1;
        uint *stack = gpudb::GpuStackAllocator::getInstance().alloc<uint>(stackSize * a.table->rows.size());
        uint *pointsInsideLineBuffer = gpudb::GpuStackAllocator::getInstance().alloc<uint>(a.table->rows.size() + 1);
        uint *prefixSumPointsInsideLineBuffer = gpudb::GpuStackAllocator::getInstance().alloc<uint>(a.table->rows.size() + 1);
        uint8_t *cub_tmp_mem = nullptr;
        uint64_t cub_tmp_memsize = 0;
        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, pointsInsideLineBuffer, prefixSumPointsInsideLineBuffer, a.table->rows.size() + 1);
        cub_tmp_mem = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_memsize);

        uint *cpuPointsInsidePolygon = StackAllocator::getInstance().alloc<uint>(a.table->rows.size());

        if (pointsInsideLineBuffer == nullptr ||
            stack == nullptr ||
            cpuPointsInsidePolygon == nullptr ||
            cub_tmp_mem == nullptr ||
            prefixSumPointsInsideLineBuffer == nullptr) {
            break;
        }

        dim3 block(BLOCK_SIZE);
        dim3 grid(gridConfigure(a.table->rows.size(), block));
        boxInsideBoxLineKernel<<<grid, block>>>(thrust::raw_pointer_cast(a.table->rows.data()),
                                                        pointsInsideLineBuffer,
                                                        b.table->bvh,
                                                        stack,
                                                        stackSize,
                                                        a.table->rows.size(),
                                                        radius);

        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, pointsInsideLineBuffer, prefixSumPointsInsideLineBuffer, a.table->rows.size() + 1);
        uint allsize = 0;
        cudaMemcpy(&allsize, prefixSumPointsInsideLineBuffer + a.table->rows.size(), sizeof(uint), cudaMemcpyDeviceToHost);

        gpudb::GpuStackAllocator::getInstance().free(cub_tmp_mem);
        cub_tmp_mem = nullptr;
        uint *testedLineNum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allsize);
        uint *testedResult = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allsize + 1);
        uint *testedPointNum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allsize);
        uint *testedResultPrefixSum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allsize + 1);

        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, testedResult, testedResultPrefixSum, allsize + 1);
        cub_tmp_mem = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_memsize);

        if (testedLineNum == nullptr || testedResult == nullptr ||
            testedPointNum == nullptr || testedResultPrefixSum == nullptr || cub_tmp_mem == nullptr) {
            break;
        }

        boxInsideBoxLineKernel2<<<grid, block>>>(thrust::raw_pointer_cast(a.table->rows.data()),
                                                        pointsInsideLineBuffer,
                                                        prefixSumPointsInsideLineBuffer,
                                                        testedLineNum,
                                                        testedPointNum,
                                                        b.table->bvh,
                                                        stack,
                                                        stackSize,
                                                        a.table->rows.size(),
                                                        radius);
        grid = gridConfigure(allsize, block);
        pointInsideLineBufferKernel<<<grid, block>>>(thrust::raw_pointer_cast(a.table->rows.data()),
                                                   pointsInsideLineBuffer,
                                                   testedLineNum,
                                                   testedPointNum,
                                                   testedResult,
                                                   thrust::raw_pointer_cast(b.table->rows.data()),
                                                   allsize, radius);


        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, testedResult, testedResultPrefixSum, allsize + 1);

        uint totalRowsSelected = 0;
        cudaMemcpy(&totalRowsSelected, testedResultPrefixSum + allsize, sizeof(uint), cudaMemcpyDeviceToHost);

        uint *cpuselectedRows = nullptr;

        if (totalRowsSelected > 0) {
            uint *selectedRows = gpudb::GpuStackAllocator::getInstance().alloc<uint>(totalRowsSelected);
            cpuselectedRows = StackAllocator::getInstance().alloc<uint>(totalRowsSelected);

            if (selectedRows == nullptr || cpuselectedRows == nullptr) {
                break;
            }

            dim3 grid = gridConfigure(allsize, block);
            selector<<<grid, block>>>(selectedRows, testedPointNum, testedResult, testedResultPrefixSum, allsize);
            cudaMemcpy(cpuselectedRows, selectedRows, sizeof(uint) * totalRowsSelected, cudaMemcpyDeviceToHost);
            gpudb::GpuStackAllocator::getInstance().free(selectedRows);
        }

        grid = gridConfigure(a.table->rows.size(), block);
        computeResultSize<<<grid, block>>>(pointsInsideLineBuffer, prefixSumPointsInsideLineBuffer, testedResultPrefixSum, a.table->rows.size());
        computeResultSize2<<<grid, block>>>(pointsInsideLineBuffer, a.table->rows.size());

        gpudb::GpuStackAllocator::getInstance().free(cub_tmp_mem);
        gpudb::GpuStackAllocator::getInstance().free(testedResultPrefixSum);

        uint *cputestedResultSize = StackAllocator::getInstance().alloc<uint>(a.table->rows.size());
        cudaMemcpy(cputestedResultSize, pointsInsideLineBuffer, sizeof(uint) * a.table->rows.size(), cudaMemcpyDeviceToHost);

        /*uint *cputestedLineNum = StackAllocator::getInstance().alloc<uint>(allsize);
        uint *cputestedResult = StackAllocator::getInstance().alloc<uint>(allsize);
        uint *cputestedPointNum = StackAllocator::getInstance().alloc<uint>(allsize);
        cudaMemcpy(cputestedLineNum, testedLineNum, allsize * sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(cputestedResult, testedResult, allsize * sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(cputestedPointNum, testedPointNum, allsize * sizeof(uint), cudaMemcpyDeviceToHost);

        for (int i = 0; i < a.table->rows.size(); i++) {
            printf("%d ", cputestedResultSize[i]);
        }
        printf("\n");

        for (uint i = 0; i < allsize; i++) {
            printf("{ point : %d result : %d line : %d} \n", cputestedPointNum[i], cputestedResult[i], cputestedLineNum[i]);
        }*/

        resultToTempTable1(a, b, "Linebuffer operation result", cpuselectedRows, cputestedResultSize, resultTempTable);
    } while(0);
    error:
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();
    return resultTempTable;
}

TempTable DataBase::polygonxpointPointsInPolygon(TempTable const &a, TempTable &b) {
    TempTable resultTempTable;

    if (!a.isValid() || !b.isValid()) {
        return resultTempTable;
    }
    if (a.table == nullptr ||
        b.table == nullptr ||
        a.getSpatialKeyType() != SpatialType::POLYGON ||
        b.getSpatialKeyType() != SpatialType::POINT ||
        a.table->rows.size() == 0 || b.table->rows.size() == 0) {
        return resultTempTable;
    }

    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();

    do {
        if (!b.table->bvh.isBuilded()) {

            gpudb::AABB * boxes = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::AABB> (b.table->rows.size());
            if (boxes == nullptr) {
                break;
            }

            dim3 block(BLOCK_SIZE);
            dim3 grid(gridConfigure(b.table->rows.size(), block));
            buildKeysAABB<<<grid, block>>>(thrust::raw_pointer_cast(b.table->rows.data()), boxes, b.table->rows.size());
            if (b.table->bvh.build(boxes, b.table->rows.size()) == false) {
                break;
            }
            gpudb::GpuStackAllocator::getInstance().free(boxes);
        }

        uint stackSize = b.table->bvh.numBVHLevels * 2 + 1;
        uint *stack = gpudb::GpuStackAllocator::getInstance().alloc<uint>(stackSize * a.table->rows.size());
        uint *pointsInsidePolygon = gpudb::GpuStackAllocator::getInstance().alloc<uint>(a.table->rows.size() + 1);
        uint *prefixSumPointsInsidePolygon = gpudb::GpuStackAllocator::getInstance().alloc<uint>(a.table->rows.size() + 1);
        uint8_t *cub_tmp_mem =nullptr;
        uint64_t cub_tmp_memsize = 0;
        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, pointsInsidePolygon, prefixSumPointsInsidePolygon, a.table->rows.size() + 1);
        cub_tmp_mem = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_memsize);
        uint *cpuPointsInsidePolygon = StackAllocator::getInstance().alloc<uint>(a.table->rows.size());

        if (pointsInsidePolygon == nullptr ||
            stack == nullptr ||
            cpuPointsInsidePolygon == nullptr ||
            cub_tmp_mem == nullptr ||
            prefixSumPointsInsidePolygon == nullptr) {
            break;
        }

        dim3 block(BLOCK_SIZE);
        dim3 grid(gridConfigure(a.table->rows.size(), block));
        boxInsideBoxKernel<<<grid, block>>>(thrust::raw_pointer_cast(a.table->rows.data()),
                                                        pointsInsidePolygon,
                                                        b.table->bvh,
                                                        stack,
                                                        stackSize,
                                                        a.table->rows.size());

        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, pointsInsidePolygon, prefixSumPointsInsidePolygon, a.table->rows.size() + 1);
        uint allsize = 0;
        cudaMemcpy(&allsize, prefixSumPointsInsidePolygon + a.table->rows.size(), sizeof(uint), cudaMemcpyDeviceToHost);

        gpudb::GpuStackAllocator::getInstance().free(cub_tmp_mem);
        cub_tmp_mem = nullptr;

        // номер точки /результат тестирвования/номер полигона
        uint *testedPolygonNum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allsize);
        uint *testedResult = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allsize + 1);
        uint *testedPointNum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allsize);
        uint *testedResultPrefixSum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allsize + 1);


        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, testedResult, testedResultPrefixSum, allsize + 1);
        cub_tmp_mem = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_memsize);

        if (testedPolygonNum == nullptr || testedResult == nullptr || testedPointNum == nullptr || testedResultPrefixSum == nullptr || cub_tmp_mem == nullptr) {
            break;
        }

        boxInsideBoxKernel2<<<grid, block>>>(thrust::raw_pointer_cast(a.table->rows.data()),
                                                        pointsInsidePolygon,
                                                        prefixSumPointsInsidePolygon,
                                                        testedPolygonNum,
                                                        testedPointNum,
                                                        b.table->bvh,
                                                        stack,
                                                        stackSize,
                                                        a.table->rows.size());
        grid = gridConfigure(allsize, block);
        pointInsidePolygonKernel<<<grid, block>>>(thrust::raw_pointer_cast(a.table->rows.data()),
                                                   pointsInsidePolygon,
                                                   testedPolygonNum,
                                                   testedPointNum,
                                                   testedResult,
                                                   thrust::raw_pointer_cast(b.table->rows.data()),
                                                   allsize);

//        uint *cputestedPolygonNum = StackAllocator::getInstance().alloc<uint>(allsize);
//        uint *cputestedResult = StackAllocator::getInstance().alloc<uint>(allsize);
//        uint *cputestedPointNum = StackAllocator::getInstance().alloc<uint>(allsize);
//        cudaMemcpy(cputestedPolygonNum, testedPolygonNum, allsize * sizeof(uint), cudaMemcpyDeviceToHost);
//        cudaMemcpy(cputestedResult, testedResult, allsize * sizeof(uint), cudaMemcpyDeviceToHost);
//        cudaMemcpy(cputestedPointNum, testedPointNum, allsize * sizeof(uint), cudaMemcpyDeviceToHost);
//        for (uint i = 0; i < allsize; i++) {
//            printf("{ point : %d result : %d polygon : %d} \n", cputestedPointNum[i], cputestedResult[i], cputestedPolygonNum[i]);
//        }

        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, testedResult, testedResultPrefixSum, allsize + 1);

        uint totalRowsSelected = 0;
        cudaMemcpy(&totalRowsSelected, testedResultPrefixSum + allsize, sizeof(uint), cudaMemcpyDeviceToHost);

        uint *cpuselectedRows = nullptr;

        if (totalRowsSelected > 0) {
            uint *selectedRows = gpudb::GpuStackAllocator::getInstance().alloc<uint>(totalRowsSelected);
            cpuselectedRows = StackAllocator::getInstance().alloc<uint>(totalRowsSelected);

            if (selectedRows == nullptr || cpuselectedRows == nullptr) {
                break;
            }

            dim3 grid = gridConfigure(allsize, block);
            selector<<<grid, block>>>(selectedRows, testedPointNum, testedResult, testedResultPrefixSum, allsize);
            cudaMemcpy(cpuselectedRows, selectedRows, sizeof(uint) * totalRowsSelected, cudaMemcpyDeviceToHost);
            gpudb::GpuStackAllocator::getInstance().free(selectedRows);
        }

        grid = gridConfigure(a.table->rows.size(), block);
        computeResultSize<<<grid, block>>>(pointsInsidePolygon, prefixSumPointsInsidePolygon, testedResultPrefixSum, a.table->rows.size());
        computeResultSize2<<<grid, block>>>(pointsInsidePolygon, a.table->rows.size());

        gpudb::GpuStackAllocator::getInstance().free(cub_tmp_mem);
        gpudb::GpuStackAllocator::getInstance().free(testedResultPrefixSum);

        uint *cputestedResultSize = StackAllocator::getInstance().alloc<uint>(a.table->rows.size());
        cudaMemcpy(cputestedResultSize, pointsInsidePolygon, sizeof(uint) * a.table->rows.size(), cudaMemcpyDeviceToHost);

        resultToTempTable1(a, b, "Points inside Polygon operation result", cpuselectedRows, cputestedResultSize, resultTempTable);
    } while(0);
    error:
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();
    return resultTempTable;
}

__device__
static float mindist1D(float p, float s, float t) {
    if (p < s) {
        return s;
    }
    if (p > t) {
        return t;
    }
    return p;
}

__device__
static float mindist(float2 p, float4 s, float4 t) {
    float2 r;
    r.x = mindist1D(p.x, s.x, t.x);
    r.y = mindist1D(p.y, s.y, t.y);
    return sqr(p.x - r.x) + sqr(p.y - r.y);
}

__device__
static  float minmaxdist(float2 p, float4 s, float4 t) {
    float2 rM, rm;
    rM.x = (2.0f * p.x >= (s.x + t.x))? s.x : t.x;
    rM.y = (2.0f * p.y >= (s.y + t.y))? s.y : t.y;

    rm.x = (2.0f * p.x <= (s.x + t.x))? s.x : t.x;
    rm.y = (2.0f * p.y <= (s.y + t.y))? s.y : t.y;

    float d1 = sqr(p.x - rm.x) + sqr(p.y - rM.y);
    float d2 = sqr(p.y - rm.y) + sqr(p.x - rM.x);

    return min(d1 + 2 * d1 * FLT_EPSILON, d2 + 2 * d2 * FLT_EPSILON);
}

#define NOT_USED 0xFFFFFFFF

__device__ void visitOrder(uint pos,
                           gpudb::HLBVH &bvh,
                           float2 point,
                           Heap<float, uint, uint> &heap,
                           GpuStack<uint2> &st)
{
    float4 bmin1 = bvh.aabbMin[pos];
    float4 bmax1 = bvh.aabbMax[pos];
    float4 bmin2 = bvh.aabbMin[pos + 1];
    float4 bmax2 = bvh.aabbMax[pos + 1];
    float min1 = mindist(point, bmin1, bmax1);
    float min2 = mindist(point, bmin2, bmax2);
    float minmax1 = minmaxdist(point, bmin1, bmax1);
    float minmax2 = minmaxdist(point, bmin2, bmax2);
    uint *memoryRef1 = 0;
    uint *memoryRef2 = 0;

    if (min1 < min2) {
        st.push(make_uint2(pos + 1, NOT_USED));
        memoryRef2 = &st.topRef().y;

        st.push(make_uint2(pos, NOT_USED));
        memoryRef1 = &st.topRef().y;
    } else {
        st.push(make_uint2(pos, NOT_USED));
        memoryRef1 = &st.topRef().y;

        st.push(make_uint2(pos + 1, NOT_USED));
        memoryRef2 = &st.topRef().y;
    }

    if (minmax1 < heap.maxKey()) {
        if (heap.count == heap.cap) {
            heap.extractMax();
        }
        heap.insert(minmax1, -1, memoryRef1);
    }

    if (minmax2 < heap.maxKey()) {
        if (heap.count == heap.cap) {
            heap.extractMax();
        }
        heap.insert(minmax2, -1, memoryRef2);
    }
}

__global__
void knearestNeighbor(gpudb::HLBVH bvh,
                      gpudb::GpuRow **search,
                      gpudb::GpuRow **data,
                      float *heapKeys,
                      uint *heapValues,
                      uint **heapIndexes,
                      uint2 *stack,
                      uint stackSize,
                      uint k,
                      uint workSize)
{
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    float2 point;
    point.x = ((gpudb::GpuPoint*)search[idx]->spatialPart.key)->p.x;
    point.y = ((gpudb::GpuPoint*)search[idx]->spatialPart.key)->p.y;

    GpuStack<uint2> st(stack + idx * stackSize, stackSize);

    Heap<float, uint, uint> heap(heapKeys + idx *  k, heapValues + idx * k, heapIndexes + idx * k, k);
    heap.count = heap.cap;

    for (int i = 0; i < k; i++) {
        heap.keys[i] = INFINITY;
        heap.values[i] = -1;
        heap.indexes[i] = nullptr;
    }

    visitOrder(0, bvh, point, heap, st);
    while(!st.empty()) {
        uint2 posSt = st.top(); st.pop();
        uint pos = posSt.x;
        uint ref = posSt.y;
        int link = bvh.links[pos];

        float4 bmin1 = bvh.aabbMin[pos];
        float4 bmax1 = bvh.aabbMax[pos];

        if (heap.maxKey() < mindist(point, bmin1, bmax1)) {
            continue;
        }

        if (ref != NOT_USED) {
            heap.deleteKey(ref);
            heap.insert(INFINITY, -1, nullptr);
        }

        if (link == LEAF) {
            for (int i = bvh.ranges[pos].x; i < bvh.ranges[pos].y; i++) {
                uint bvhref = bvh.references[i];
                float2 p = ((gpudb::GpuPoint*)data[bvhref]->spatialPart.key)->p;

                float dist = lenSqr(p, point);
                if (dist < heap.maxKey()) {
                    if (heap.cap == heap.count) {
                        heap.extractMax();
                    }
                    heap.insert(dist, bvhref, nullptr);
                }
            }
        } else {
            visitOrder(link, bvh, point, heap, st);
        }
    }

    while(!heap.empty()) {
        uint heapV = heap.maxValue();
        float heapK = heap.maxKey();
        heap.extractMax();
        heap.values[heap.count] = heapV;
        heap.keys[heap.count] = heapK;
    }
}

TempTable DataBase::pointxpointKnearestNeighbor(TempTable const &a, TempTable &b, uint k) {
    TempTable resultTempTable;

    if (!a.isValid() || !b.isValid()) {

        return resultTempTable;
    }

    if (a.table == nullptr ||
        b.table == nullptr ||
        a.getSpatialKeyType() != SpatialType::POINT ||
        b.getSpatialKeyType() != SpatialType::POINT) {
        std::cout << typeToString(a.getSpatialKeyType()) << " " << typeToString(b.getSpatialKeyType()) << std::endl;
        return resultTempTable;
    }

    if (k == 0) {
        return resultTempTable;
    }

    if (k > b.table->rows.size()) {
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "k = %d is more than %d", k, b.table->rowsSize.size());
        return resultTempTable;
    }

    if (a.table->rows.size() == 0 || b.table->rows.size() == 0) {
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "table a or b is empty");
        return resultTempTable;
    }

    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();

    do {
        if (!b.table->bvh.isBuilded()) {

            gpudb::AABB * boxes = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::AABB> (b.table->rows.size());
            if (boxes == nullptr) {
                break;
            }

            dim3 block(BLOCK_SIZE);
            dim3 grid(gridConfigure(b.table->rows.size(), block));
            buildKeysAABB<<<grid, block>>>(thrust::raw_pointer_cast(b.table->rows.data()), boxes, b.table->rows.size());
            if (b.table->bvh.build(boxes, b.table->rows.size()) == false) {
                break;
            }
            gpudb::GpuStackAllocator::getInstance().free(boxes);
        }

        uint stackSize = b.table->bvh.numBVHLevels * 2 + 1;
        float *heapKeys = gpudb::GpuStackAllocator::getInstance().alloc<float>(k * a.table->rows.size());
        uint *heapValues = gpudb::GpuStackAllocator::getInstance().alloc<uint>(k * a.table->rows.size());
        uint **heapIndexes = gpudb::GpuStackAllocator::getInstance().alloc<uint*>(k * a.table->rows.size());
        uint2 *stack = gpudb::GpuStackAllocator::getInstance().alloc<uint2>(stackSize * a.table->rows.size());
        uint *result = StackAllocator::getInstance().alloc<uint>(k * a.table->rows.size());
        if (heapIndexes == nullptr || stack == nullptr || heapKeys == nullptr || heapValues == nullptr || result == nullptr) {
            break;
        }

        dim3 block(BLOCK_SIZE);
        dim3 grid(gridConfigure(a.table->rows.size(), block));
        Timer t;
        t.start();
        knearestNeighbor<<<grid, block>>>(b.table->bvh,
                                          thrust::raw_pointer_cast(a.table->rows.data()),
                                          thrust::raw_pointer_cast(b.table->rows.data()),
                                          heapKeys,
                                          heapValues,
                                          heapIndexes,
                                          stack,
                                          stackSize,
                                          k,
                                          a.table->rows.size());
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "k nearest neighbor in %d ms", t.elapsedMillisecondsU64());
        cudaMemcpy(result, heapValues, sizeof(uint) * k * a.table->rows.size(), cudaMemcpyDeviceToHost);
        uint *sizes = StackAllocator::getInstance().alloc<uint>(a.table->rows.size());

        if (sizes == nullptr) {
            break;
        }

        for (int i = 0; i < a.table->rows.size(); i++) {
            sizes[i] = k;
        }

        resultToTempTable1(a, b, "k nearest neighbor",result, sizes, resultTempTable);
    } while(0);

    error:
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();

    return resultTempTable;
}
