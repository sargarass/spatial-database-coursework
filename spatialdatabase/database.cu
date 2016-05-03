#include "database.h"

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

    result.table->spatialKey = gpuTable->spatialKey;
    result.table->temporalKey = gpuTable->temporalKey;

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
    rows[idx]->temporalPart.boundingBox(&boxes[idx]);
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

    return min(d1 + 5 * d1 * FLT_EPSILON, d2 + 5 * d2 * FLT_EPSILON);
}

#define NOT_USED 0xFFFFFFFF

__device__ void visitOrder(uint pos, gpudb::HLBVH &bvh, float2 point, Heap<float, uint, uint> &heap, GpuStack<uint2> &st) {
    float4 bmin1 = bvh.aabbMin[pos];
    float4 bmax1 = bvh.aabbMax[pos];
    float4 bmin2 = bvh.aabbMin[pos + 1];
    float4 bmax2 = bvh.aabbMax[pos + 1];
    int link1 = bvh.links[pos];
    int link2 = bvh.links[pos + 1];
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
void knearestNeighbor(gpudb::HLBVH bvh, gpudb::GpuRow **search, gpudb::GpuRow **data, float *heapKeys, uint *heapValues, uint **heapIndexes, uint2 *stack, uint stackSize, uint k, uint workSize) {
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

    st.push(make_uint2(0, NOT_USED));
    st.push(make_uint2(1, NOT_USED));

    int len = 0;
    while(!st.empty()) {
        len++;
        if (len >= 2000) {
            break;
        }

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

//__device__ uint2 visitOrder(uint pos, gpudb::HLBVH &bvh, float2 point, uint k, float actualDist) {
//    float4 bmin1 = bvh.aabbMin[pos];
//    float4 bmax1 = bvh.aabbMax[pos];
//    float4 bmin2 = bvh.aabbMin[pos + 1];
//    float4 bmax2 = bvh.aabbMax[pos + 1];
//    float min1 = mindist(point, bmin1, bmax1);
//    float min2 = mindist(point, bmin2, bmax2);
//    float minmax1 = minmaxdist(point, bmin1, bmax1);
//    float minmax2 = minmaxdist(point, bmin2, bmax2);

//    bool discard1 = false;
//    bool discard2 = false;

//    if (min1 > minmax2 && actualdist < min1 && k == 0) {
//        discard1 = true;
//    }

//    if (min2 > minmax1 && actualdist < min2 &&  k == 0) {
//        discard2 = true;
//    }

//    if (discard1) {
//        return make_uint2(pos + 1, USED);
//    }

//    if (discard2) {
//        return make_uint2(pos, USED);
//    }

//    if (min1 < min2) {
//        return make_uint2(pos, pos + 1);
//    }

//    return make_uint2(pos + 1, pos);
//}

//__device__
//bool strategy3(uint pos, float2 point, gpudb::HLBVH &bvh, float actualDist, uint k) {
//    float4 bmin1 = bvh.aabbMin[pos];
//    float4 bmax1 = bvh.aabbMax[pos];

//    if (actualDist < mindist(point, bmin1, bmax1) && k == 0) {
//        return true;
//    } else {
//        return false;
//    }
//}

//__device__
//void strategy2(uint pos, gpudb::HLBVH &bvh, float2 point, Heap<float, uint> &heap, float &actualDist, uint &k) {
//    float4 bmin1 = bvh.aabbMin[pos];
//    float4 bmax1 = bvh.aabbMax[pos];
//    float minmax = minmaxdist(point, bmin1, bmax1);

//    if (minmax < heap.maxKey() && k == 0) {
//        heap.extractMax();
//        if (!heap.empty()) {
//            actualDist = heap.maxKey();
//        }
//        k++;
//    }
//}

//__global__
//void knearestNeighbor(gpudb::HLBVH bvh, gpudb::GpuRow **search, gpudb::GpuRow **data, float *heapKeys, uint *heapValues, float *heapKeys2, uint *heapValues2, uint2 *stack, uint stackSize, uint k, uint workSize, uint dataSize) {
//    uint idx = getGlobalIdx3DZXY();
//    if (idx >= workSize) {
//        return;
//    }

//    float2 point;
//    point.x = ((gpudb::GpuPoint*)search[idx]->spatialPart.key)->p.x;
//    point.y = ((gpudb::GpuPoint*)search[idx]->spatialPart.key)->p.y;

//    float actualDist = INFINITY;
//    GpuStack<uint2> st(stack + idx * stackSize, stackSize);

//    st.push(visitOrder(0, bvh, point, k, actualDist));
//    Heap<float, uint> heap(heapKeys + idx *  k, heapValues + idx * k, k);
//    Heap<float, uint> heap2(heapKeys2 + idx *  k, heapValues2 + idx * k, k);

//    for (int i = 0; i < dataSize; i++) {
//        float2 p = ((gpudb::GpuPoint*)data[i]->spatialPart.key)->p;
//        float dist = lenSqr(p, point);

//        if (heap2.count < heap2.cap) {
//            heap2.insert(dist, dist);
//        } else {
//            if (heap2.maxKey() > dist) {
//                heap2.extractMax();
//                heap2.insert(dist, dist);
//            }
//        }
//    }

//    int len = 0;
//    while(!st.empty()) {
//        len++;
//        if (len == 4000) {
//            printf("TIME ERROR");
//            break;
//        }

//        uint2 pos = st.top();
//        if (st.empty()) {
//            break;
//        }

//        if (pos.x == USED) {
//            // upward puning
//            myswap(pos.x, pos.y);
//            if (strategy3(pos.x, point, bvh, actualDist, k)) {
//                st.pop();
//            } else {
//              //  strategy2(pos.x, bvh, point, heap, actualDist, k);
//                st.topRef() = pos;
//            }
//        } else {
//            st.topRef().x = USED;
//            if (st.topRef().y == USED) {
//                st.pop();
//            }

//            int link = bvh.links[pos.x];
//            if (link == LEAF) {
//                // actualdistance
//                for (int i = bvh.ranges[pos.x].x; i < bvh.ranges[pos.x].y; i++) {
//                    uint ref = bvh.references[i];
//                    float2 p = ((gpudb::GpuPoint*)data[ref]->spatialPart.key)->p;

//                    float dist = lenSqr(p, point);
//                    if (heap.cap > heap.count) {
//                        heap.insert(dist, ref);
//                        k = heap.cap - heap.count;
//                    } else {
//                        if (dist < heap.maxKey()) {
//                            heap.extractMax();
//                            heap.insert(dist, ref);
//                        }
//                    }
//                }
//                actualDist = heap.maxKey();
//            } else {
//                st.push(visitOrder(link, bvh, point, k, actualDist));
//            }
//        }
//    }
//    printf("iters %d\n", len);

//    printf("%d %d\n", heap.count, heap2.count);
//    while (!heap.empty() && !heap2.empty()) {
//        printf("%f\n", heap.maxKey() - heap2.maxKey());
//        heap.extractMax();
//        heap2.extractMax();
//    }
//}

bool pointxpointKnearestNeighbor(TempTable const &a, TempTable const &b, uint k, TempTable &resultTempTable) {
    if (a.table == nullptr ||
        b.table == nullptr ||
        a.getSpatialKeyType() != SpatialType::POINT ||
        b.getSpatialKeyType() != SpatialType::POINT) {
        return false;
    }

    if (a.getSpatialKeyType() != b.getSpatialKeyType()) {
        return false;
    }

    if (k == 0) {
        return false;
    }

    if (k > b.table->rows.size()) {
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "k = %d is more than %d", k, b.table->rowsSize.size());
        return false;
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
        fflush(stdout);
        cudaMemcpy(result, heapValues, sizeof(uint) * k * a.table->rows.size(), cudaMemcpyDeviceToHost);
        gLogWrite(LOG_MESSAGE_TYPE::INFO, "k nearest neighbor in %d ms", t.elapsedMillisecondsU64());

        gpudb::GpuTable *tables[a.table->rows.size()];
        TempTable *newTempTables[a.table->rows.size()];

        gpudb::GpuTable *resultTable = new gpudb::GpuTable;

        if (!resultTable) {
            break;
        }

        for (int i = 0; i < a.table->rows.size(); i++) {
            tables[i] = new gpudb::GpuTable;
            newTempTables[i] = new TempTable;

            if (tables[i] == nullptr) {
                for (int j = 0; j < i; j++) {
                    delete tables[j];
                    delete newTempTables[j];
                }
                goto error;
            }
        }

        gpudb::GpuRow **gpuRows = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::GpuRow*>(k * a.table->rows.size());
        uint64_t *sizes = gpudb::GpuStackAllocator::getInstance().alloc<uint64_t>(k * a.table->rows.size());
        uint64_t *cpuSizes = StackAllocator::getInstance().alloc<uint64_t>(k * a.table->rows.size());


        gpudb::GpuColumnAttribute atr;
        std::snprintf(atr.name, NAME_MAX_LEN, "%d nearest neighbor set", k);
        atr.type = Type::SET;

        AttributeDescription desc;
        desc.name.resize(NAME_MAX_LEN);
        std::snprintf(&desc.name[0], NAME_MAX_LEN, "%d nearest neighbor set", k);
        desc.type = Type::SET;

        uint8_t *gpuRowsMemory[k * a.table->rows.size()];
        for (size_t i = 0; i < k * a.table->rows.size(); i++) {
            //printf("%d %d %d\n", result[i], i, i / k);
            gpuRowsMemory[i] = gpudb::gpuAllocator::getInstance().alloc<uint8_t>(b.table->rowsSize[result[i]]);

            if (gpuRowsMemory[i] == nullptr) {
                for (size_t j = 0; j < i; j++) {
                    gpudb::gpuAllocator::getInstance().free(gpuRowsMemory[j]);
                }
                goto error;
            }
        }

        thrust::host_vector<gpudb::GpuRow*> rows;
        rows.resize(k);
        for (size_t i = 0; i < a.table->rows.size(); i++) {
            tables[i]->spatialKey = a.table->spatialKey;
            tables[i]->temporalKey = a.table->temporalKey;
            tables[i]->columns.reserve(a.table->columns.size());
            tables[i]->bvh.builded = false;
            tables[i]->columns = a.table->columns;

            memcpy(tables[i]->name, a.table->name, NAME_MAX_LEN * sizeof(char));
            tables[i]->rowsSize.resize(k);
            tables[i]->rows.reserve(k);
            for (size_t j = i * k, p = 0; j < (i + 1) * k; j++, p++) {
                rows[p] = (gpudb::GpuRow*) gpuRowsMemory[j];
                tables[i]->rowsSize[p] = b.table->rowsSize[result[j]];
                cpuSizes[j] = b.table->rowsSize[result[j]];
            }
            tables[i]->rows = rows;
        }

        grid = gridConfigure(k * a.table->rows.size(), block);
        cudaMemcpy(gpuRows, gpuRowsMemory, sizeof(uint8_t*) * k * a.table->rows.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(sizes, cpuSizes, sizeof(uint64_t) * k * a.table->rows.size(), cudaMemcpyHostToDevice);
        gpuRowsCopyOnlySelected<<<grid, block>>>(gpuRows, thrust::raw_pointer_cast(b.table->rows.data()), heapValues, sizes, k * a.table->rows.size());

        for (int i = 0; i < a.table->rows.size(); i++) {
            newTempTables[i]->description = b.description;
            newTempTables[i]->valid = true;
            newTempTables[i]->table = tables[i];
        }

        resultTable->bvh.builded = false;
        std::snprintf(resultTable->name, NAME_MAX_LEN, "%d nearest neighbor", k);
        resultTable->rows.reserve(a.table->rows.size());

        resultTable->rowsSize.resize(a.table->rows.size());

        resultTable->spatialKey = a.table->spatialKey;
        resultTable->temporalKey = a.table->temporalKey;
        resultTable->columns.reserve(a.table->columns.size() + 1);
        resultTable->columns = a.table->columns;
        resultTable->columns.push_back(atr);

        TableDescription tdescription;
        tdescription = a.description;
        tdescription.columnDescription.push_back(desc);

        uint8_t *resultRows[a.table->rows.size()];
        uint8_t *cpuRows[a.table->rows.size()];

        thrust::host_vector<gpudb::GpuRow*> hostRowsResult;
        hostRowsResult.resize(a.table->rows.size());

        for (int i = 0; i < a.table->rows.size(); i++) {
            uint64_t memsize = a.table->rowsSize[i] + sizeof(gpudb::Value) + typeSize(Type::SET);
            resultTable->rowsSize[i] = memsize;
            resultRows[i] = gpudb::gpuAllocator::getInstance().alloc<uint8_t>(memsize);
            cpuRows[i] = StackAllocator::getInstance().alloc<uint8_t>(memsize);

            hostRowsResult[i] = ((gpudb::GpuRow*)(resultRows[i]));
            if (resultRows[i] == nullptr) {
                for (int j = 0; j < i; j++) {
                    gpudb::gpuAllocator::getInstance().free(resultRows[j]);
                }
                goto error;
            }
        }
        resultTable->rows = hostRowsResult;

        thrust::host_vector<gpudb::GpuRow*> hostRows = a.table->rows;
        for (int i = 0; i < a.table->rows.size(); i++) {
            uint8_t *aRow = StackAllocator::getInstance().alloc<uint8_t>(a.table->rowsSize[i]);
            DataBase::getInstance().loadCPU((gpudb::GpuRow*)aRow, hostRows[i], a.table->rowsSize[i]);
            gpudb::GpuRow* cpuRowPointer = ((gpudb::GpuRow*)cpuRows[i]);
            gpudb::GpuRow* aCpuRowPointer = ((gpudb::GpuRow*)aRow);
            uintptr_t cpuRawPointer = (uintptr_t)cpuRows[i];
            cpuRowPointer->spatialPart.type = aCpuRowPointer->spatialPart.type;
            cpuRowPointer->temporalPart.type = a.description.temporalKeyType;
            cpuRowPointer->valueSize = tdescription.columnDescription.size();
            cpuRowPointer->value = (gpudb::Value*)(cpuRawPointer + sizeof(gpudb::GpuRow));
            uintptr_t memoryValues = cpuRawPointer + sizeof(gpudb::GpuRow) + sizeof(gpudb::Value) * cpuRowPointer->valueSize;
            for (int j = 0; j < tdescription.columnDescription.size(); j++) {
                cpuRowPointer->value[j].value = (void*)memoryValues;
                if (j < a.description.columnDescription.size()) {
                    cpuRowPointer->value[j].isNull = aCpuRowPointer->value[j].isNull;
                } else {
                    cpuRowPointer->value[j].isNull = false;
                }

                uint64_t attrSize = typeSize(tdescription.columnDescription[j].type);
                if (j < a.description.columnDescription.size()) {
                    memcpy(cpuRowPointer->value[j].value, aCpuRowPointer->value[j].value, attrSize);
                } else {
                    uintptr_t pointer = (uintptr_t)newTempTables[i];
                    memcpy(cpuRowPointer->value[j].value, &pointer, attrSize);
                }

                memoryValues += attrSize;
            }
            cpuRowPointer->spatialPart.key = (void*)memoryValues;
            ((gpudb::GpuPoint*) (cpuRowPointer->spatialPart.key))->p.x = ((gpudb::GpuPoint*)(aCpuRowPointer->spatialPart.key))->p.x;
            ((gpudb::GpuPoint*) (cpuRowPointer->spatialPart.key))->p.y = ((gpudb::GpuPoint*)(aCpuRowPointer->spatialPart.key))->p.y;

            cpuRowPointer->temporalPart.transactionTimeCode = aCpuRowPointer->temporalPart.transactionTimeCode;
            cpuRowPointer->temporalPart.validTimeECode = aCpuRowPointer->temporalPart.validTimeECode;
            cpuRowPointer->temporalPart.validTimeSCode = aCpuRowPointer->temporalPart.validTimeSCode;

            DataBase::getInstance().storeGPU((gpudb::GpuRow*)resultRows[i], cpuRowPointer, resultTable->rowsSize[i]);
            StackAllocator::getInstance().free(aRow);
        }

        resultTempTable.table = resultTable;
        resultTempTable.valid = true;
        resultTempTable.description = tdescription;

        gpudb::GpuStackAllocator::getInstance().popPosition();
        StackAllocator::getInstance().popPosition();

        return true;
    } while(0);

    error:
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();

    return false;
}
