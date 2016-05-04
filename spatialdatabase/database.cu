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
    error:
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

    return min(d1 + 2 * d1 * FLT_EPSILON, d2 + 2 * d2 * FLT_EPSILON);
}

#define NOT_USED 0xFFFFFFFF

__device__ void visitOrder(uint pos, gpudb::HLBVH &bvh, float2 point, Heap<float, uint, uint> &heap, GpuStack<uint2> &st) {
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

bool pointxpointKnearestNeighbor(TempTable const &a, TempTable &b, uint k, TempTable &resultTempTable) {
    if (!a.isValid() || !b.isValid()) {
        return false;
    }

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
        cudaMemcpy(result, heapValues, sizeof(uint) * k * a.table->rows.size(), cudaMemcpyDeviceToHost);
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "k nearest neighbor in %d ms", t.elapsedMillisecondsU64());

        gpudb::GpuTable *tables[a.table->rows.size()];
        TempTable *newTempTables[a.table->rows.size()];

        gpudb::GpuTable *resultTable = new gpudb::GpuTable;

        if (!resultTable) {
            break;
        }

        for (int i = 0; i < a.table->rows.size(); i++) {
            tables[i] = new gpudb::GpuTable;
            newTempTables[i] = new TempTable;

            resultTempTable.needToBeFree.push_back((uintptr_t)newTempTables[i]);

            if (tables[i] == nullptr) {
                for (int j = 0; j < i; j++) {
                    delete tables[j];
                    delete newTempTables[j];
                }
                resultTempTable.needToBeFree.clear();
                goto error;
            }
        }

        gpudb::GpuColumnAttribute atr;
        std::snprintf(atr.name, NAME_MAX_LEN, "%d nearest neighbor set", k);
        atr.type = Type::SET;

        AttributeDescription desc;
        desc.name.resize(NAME_MAX_LEN);
        std::snprintf(&desc.name[0], NAME_MAX_LEN, "%d nearest neighbor set", k);
        desc.type = Type::SET;

        thrust::host_vector<gpudb::GpuRow*> rows;
        thrust::host_vector<gpudb::GpuRow*> brows = b.table->rows;

        rows.resize(k);
        for (size_t i = 0; i < a.table->rows.size(); i++) {
            tables[i]->columns.reserve(a.table->columns.size());
            tables[i]->bvh.builded = false;
            tables[i]->columns = a.table->columns;

            memcpy(tables[i]->name, a.table->name, NAME_MAX_LEN * sizeof(char));
            tables[i]->rowsSize.resize(k);
            tables[i]->rows.reserve(k);
            tables[i]->rowReferenses = true;
            for (size_t j = i * k, p = 0; j < (i + 1) * k; j++, p++) {
                tables[i]->rowsSize[p] = b.table->rowsSize[result[j]];
                rows[p] = brows[result[j]];
            }
            tables[i]->rows = rows;
        }

        for (int i = 0; i < a.table->rows.size(); i++) {
            newTempTables[i]->description = b.description;
            newTempTables[i]->valid = true;
            newTempTables[i]->table = tables[i];
        }

        resultTable->bvh.builded = false;
        std::snprintf(resultTable->name, NAME_MAX_LEN, "%d nearest neighbor", k);
        resultTable->rows.reserve(a.table->rows.size());
        resultTable->rowsSize.resize(a.table->rows.size());
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
            strncpy(cpuRowPointer->spatialPart.name, aCpuRowPointer->spatialPart.name, typeSize(Type::STRING));
            strncpy(cpuRowPointer->temporalPart.name, aCpuRowPointer->temporalPart.name, typeSize(Type::STRING));

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
            ((gpudb::GpuPoint*) (cpuRowPointer->spatialPart.key))->p.x = ((gpudb::GpuPoint*)(aCpuRowPointer->spatialPart.key))->p.x;
            ((gpudb::GpuPoint*) (cpuRowPointer->spatialPart.key))->p.y = ((gpudb::GpuPoint*)(aCpuRowPointer->spatialPart.key))->p.y;

            cpuRowPointer->temporalPart.transactionTimeCode = aCpuRowPointer->temporalPart.transactionTimeCode;
            cpuRowPointer->temporalPart.validTimeECode = aCpuRowPointer->temporalPart.validTimeECode;
            cpuRowPointer->temporalPart.validTimeSCode = aCpuRowPointer->temporalPart.validTimeSCode;

            DataBase::getInstance().storeGPU((gpudb::GpuRow*)resultRows[i], cpuRowPointer, resultTable->rowsSize[i]);
            StackAllocator::getInstance().free(aRow);
        }

        b.references.push_back(&resultTempTable);
        resultTempTable.parents.push_back(&b);

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

////////////////////////////////////////////

__device__
int mystrncmp(const char *__s1, const char *__s2, size_t __n) {
    while (__n > 0 && ((*__s1) - (*__s2)) == 0 && (*__s1) && (*__s2)) {
        --__n;
        ++__s1;
        ++__s2;
    }

    if (__n == 0) {
        return 0;
    }
    return *__s1 - *__s2;
}

__global__
void updaterKernel(gpudb::GpuColumnAttribute *columns, gpudb::GpuRow **rows, gpudb::Value *atrValues, char **atrNames, uint atrSize, Predicate p, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }
    gpudb::CRow crow(rows[idx], columns, rows[idx]->valueSize);
    if (p(crow)) {
        int j = 0;
        for (int i = 0; i < rows[idx]->valueSize; i++) {
            if (mystrncmp(columns[i].name, atrNames[j], typeSize(Type::STRING)) == 0) {
                rows[idx]->value[i].isNull = atrValues[j].isNull;
                memcpy(rows[idx]->value[i].value, atrValues[j].value, typeSize(columns[i].type));

                j++;
                if (j == atrSize) {
                    break;
                }
            }
        }
    }
}


__global__
void testRowsOnPredicate(gpudb::GpuRow **rows, gpudb::GpuColumnAttribute *columns, uint columnsSize, uint64_t *result, uint64_t *resultInverse, Predicate p, uint size) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= size) {
        return;
    }

    gpudb::CRow crow(rows[idx], columns, columnsSize);
    if (p(crow)) {
        result[idx] = 1;
        resultInverse[idx] = 0;
    } else {
        result[idx] = 0;
        resultInverse[idx] = 1;
    }
}

__global__
void distributor(gpudb::GpuRow **rows, uint64_t *decision, uint64_t *offset, gpudb::GpuRow **result, uint64_t *toDeleteOffeset, gpudb::GpuRow **toDelete, uint size) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= size) {
        return;
    }

    if (decision[idx]) {
        result[offset[idx]] = rows[idx];
    } else {
        toDelete[toDeleteOffeset[idx]] = rows[idx];
    }
}

bool DataBase::dropRow(std::string tableName, Predicate p) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);
    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return false;
    }

    TableDescription &desc = (*tableDesc).second;
    gpudb::GpuTable *table = (*tableIt).second;
    if (table->rows.size() == 0) {
        return false;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid = gridConfigure(table->rows.size(), block);
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();
    do {
        uint64_t *result = gpudb::GpuStackAllocator::getInstance().alloc<uint64_t>(table->rows.size() + 1);
        uint64_t *prefixSum = gpudb::GpuStackAllocator::getInstance().alloc<uint64_t>(table->rows.size() + 1);
        uint64_t *resultInverse = gpudb::GpuStackAllocator::getInstance().alloc<uint64_t>(table->rows.size() + 1);
        uint64_t *prefixSumInverse = gpudb::GpuStackAllocator::getInstance().alloc<uint64_t>(table->rows.size() + 1);

        uint64_t *cpuPrefixSum = StackAllocator::getInstance().alloc<uint64_t>(table->rows.size() + 1);
        uint64_t *cpuPrefixSumInverse = StackAllocator::getInstance().alloc<uint64_t>(table->rows.size() + 1);
        size_t cub_tmp_memsize = 0;
        uint8_t *cub_tmp_mem = nullptr;
        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, result, prefixSum, table->rows.size() + 1);
        cub_tmp_mem = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_memsize);

        if (prefixSum == nullptr || result == nullptr || cub_tmp_mem == nullptr ||
            cpuPrefixSum == nullptr || cpuPrefixSumInverse == nullptr || prefixSumInverse == nullptr || resultInverse == nullptr) {
            return false;
        }

        testRowsOnPredicate<<<grid, block>>>(thrust::raw_pointer_cast(table->rows.data()),
                                             thrust::raw_pointer_cast(table->columns.data()),
                                             table->columns.size(),
                                             result,
                                             resultInverse,
                                             p,
                                             table->rows.size());

        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, result, prefixSum, table->rows.size() + 1);
        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_memsize, resultInverse, prefixSumInverse, table->rows.size() + 1);

        cudaMemcpy(cpuPrefixSum, prefixSum, sizeof(uint64_t) * (table->rows.size() + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuPrefixSumInverse, prefixSumInverse, sizeof(uint64_t) * (table->rows.size() + 1), cudaMemcpyDeviceToHost);

        uint64_t toSaveSize = cpuPrefixSumInverse[table->rows.size()];
        uint64_t toDeleteSize = cpuPrefixSum[table->rows.size()];

        if (toDeleteSize == 0) {
            break;
        }

        gpudb::GpuRow **toDelete = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::GpuRow *>(toDeleteSize);
        gpudb::GpuRow **toDeleteCpu = StackAllocator::getInstance().alloc<gpudb::GpuRow *>(toDeleteSize);

        if (toDelete == nullptr || toDeleteCpu == nullptr) {
            break;
        }

        thrust::device_vector<gpudb::GpuRow *> resultRows;
        std::vector<uint64_t> sizes;
        if (toSaveSize > 0) {
            resultRows.resize(toSaveSize);
            sizes.resize(toSaveSize);
        }

        distributor<<<grid, block>>>(thrust::raw_pointer_cast(table->rows.data()),
                                resultInverse,
                                prefixSumInverse,
                                thrust::raw_pointer_cast(resultRows.data()),
                                prefixSum,
                                toDelete,
                                table->rows.size());


        if (toSaveSize > 0) {
            for (int i = 0; i < table->rowsSize.size(); i++) {
                sizes[cpuPrefixSumInverse[i]] = table->rowsSize[i];
            }
        }

        swap(table->rows, resultRows);
        swap(table->rowsSize, sizes);

        cudaMemcpy(toDeleteCpu, toDelete, sizeof(gpudb::GpuRow *) * (toDeleteSize), cudaMemcpyDeviceToHost);
        for (int i = 0; i < toDeleteSize; i++) {
            gpudb::gpuAllocator::getInstance().free(toDeleteCpu[i]);
        }

        gpudb::GpuStackAllocator::getInstance().popPosition();
        StackAllocator::getInstance().popPosition();
        return true;
    } while(0);
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();
    return false;
}

bool DataBase::dropTable(std::string tableName) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);
    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return false;
    }

    gpudb::GpuTable *table = (*tableIt).second;

    delete table;
    this->tables.erase(tableIt);
    this->tablesType.erase(tableDesc);
    return true;
}

bool DataBase::update(std::string tableName, std::set<Attribute> const &atrSet, Predicate p) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);
    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return false;
    }

    TableDescription &desc = (*tableDesc).second;
    gpudb::GpuTable *table = (*tableIt).second;
    if (table->rows.size() == 0) {
        return false;
    }

    std::vector<Attribute> atrVec(atrSet.begin(), atrSet.end());

    bool canBeUsed = false;
    uint64_t memsize = 0;
    int j = 0;
    for (int i = 0; i < desc.columnDescription.size(); i++) {
        if (desc.columnDescription[i].name == atrVec[j].name && desc.columnDescription[i].type == atrVec[j].type) {
            memsize += typeSize(atrVec[j].type);
            j++;

            if (j == atrVec.size()) {
                canBeUsed = true;
                break;
            }
        }
    }

    if (canBeUsed == false) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "this attribute set cannot be used");
        return false;
    }
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();

    uint8_t *gpuValuesMemory = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(memsize);
    gpudb::Value * gpuValues = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::Value>(atrVec.size());
    char *gpuStringsMemory = gpudb::GpuStackAllocator::getInstance().alloc<char>(typeSize(Type::STRING) * atrVec.size());
    char **gpuStringsPointers = gpudb::GpuStackAllocator::getInstance().alloc<char*>(atrVec.size());
    if (gpuValuesMemory == nullptr || gpuValues == nullptr || gpuStringsMemory == nullptr || gpuStringsPointers == nullptr) {
        gpudb::GpuStackAllocator::getInstance().popPosition();
        StackAllocator::getInstance().popPosition();
        return false;
    }

    uint8_t *cpuValuesMemory = StackAllocator::getInstance().alloc<uint8_t>(memsize);
    gpudb::Value * cpuValues = StackAllocator::getInstance().alloc<gpudb::Value>(atrVec.size());
    char *cpuStringsMemory = StackAllocator::getInstance().alloc<char>(typeSize(Type::STRING) * atrVec.size());
    char **cpuStringsPointers = StackAllocator::getInstance().alloc<char*>(atrVec.size());

    // копируем значения на cpu
    uint8_t *cpuValuesMemoryPointer = cpuValuesMemory;
    char *cpuStringsMemoryPointer = cpuStringsMemory;

    for (int i = 0; i < atrVec.size(); i++) {
        cpuValues[i].isNull = atrVec[i].isNull;
        cpuValues[i].value = (void*)cpuValuesMemoryPointer;
        cpuStringsPointers[i] = cpuStringsMemoryPointer;

        if (cpuValues[i].isNull == false) {
            memcpy(cpuValues[i].value, atrVec[i].value, typeSize(atrVec[i].type));
        }
        strncpy(cpuStringsPointers[i], atrVec[i].name.c_str(), typeSize(atrVec[i].type));

        cpuValuesMemoryPointer += typeSize(atrVec[i].type);
        cpuStringsMemoryPointer += typeSize(Type::STRING);
    }

    for (int i = 0; i < atrVec.size(); i++) {
        cpuValues[i].value = newAddress(cpuValues[i].value, cpuValuesMemory, gpuValuesMemory);
        cpuStringsPointers[i] = newAddress(cpuStringsPointers[i], cpuStringsMemory, gpuStringsMemory);
    }

    cudaMemcpy(gpuValues, cpuValues, atrVec.size() * sizeof(gpudb::Value), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuStringsPointers, cpuStringsPointers, atrVec.size() * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuValuesMemory, cpuValuesMemory, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuStringsMemory, cpuStringsMemory, typeSize(Type::STRING) * atrVec.size(), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid = gridConfigure(table->rows.size(), block);
    updaterKernel<<<block, grid>>>(thrust::raw_pointer_cast(table->columns.data()),
                                   thrust::raw_pointer_cast(table->rows.data()),
                                   gpuValues,
                                   gpuStringsPointers,
                                   atrVec.size(),
                                   p,
                                   table->rows.size());

    StackAllocator::getInstance().popPosition();
    gpudb::GpuStackAllocator::getInstance().popPosition();
    return false;

}
