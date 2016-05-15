#include "database.h"
#include <cub/cub/cub.cuh>
#include "moderngpu/src/moderngpu/context.hxx"
#include "moderngpu/src/moderngpu/kernel_mergesort.hxx"

bool DataBase::selectRow(std::string tableName, Predicate p, std::vector<Row> &result) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);

    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return false;
    }

    TableDescription &desc = (*tableDesc).second;
    gpudb::GpuTable *table = (*tableIt).second;
    return selectRowImp(desc, table, p, result);
}

bool DataBase::selectRow(TempTable &table, Predicate p, std::vector<Row> &result) {
    if (!table.isValid()) {
        return false;
    }

    return selectRowImp(table.description, table.table, p, result);
}

__global__
void runTestsPredicate(gpudb::GpuRow **rows, gpudb::GpuColumnAttribute *columns, uint columnsSize, uint *testsResult, Predicate p, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }
    gpudb::CRow c(rows[idx], columns, columnsSize);
    testsResult[idx] = p(c);
}

__global__
void fillGpuPassedRows(gpudb::GpuRow **rows, uint *testsResult, uint *offset, gpudb::GpuRow **passedRows, uint *passedRowsIndex, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    if (!testsResult[idx]) {
        return;
    }

    passedRowsIndex[offset[idx]] = idx;
    passedRows[offset[idx]] = rows[idx];
}

bool DataBase::selectRowImp(TableDescription &desc, gpudb::GpuTable *gputable, Predicate p, std::vector<Row> &result) {
    uint testSize = gputable->rows.size();
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();
    do {
        uint *gpuTests = gpudb::GpuStackAllocator::getInstance().alloc<uint>(testSize + 1);
        uint *gpuTestsExclusiveSum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(testSize + 1);

        uint8_t *cub_tmp_mem = nullptr;
        uint64_t cub_tmp_mem_size = 0;

        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_mem_size, gpuTests, gpuTestsExclusiveSum, testSize + 1);
        cub_tmp_mem = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_mem_size);

        if (gpuTests == nullptr || gpuTestsExclusiveSum == nullptr || cub_tmp_mem == nullptr) {
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought stack memory");
            break;
        }

        dim3 block(BLOCK_SIZE);
        dim3 grid = gridConfigure(gputable->rows.size(), block);
        runTestsPredicate<<<grid, block>>>(thrust::raw_pointer_cast(gputable->rows.data()),
                                           thrust::raw_pointer_cast(gputable->columns.data()),
                                           gputable->columns.size(),
                                           gpuTests,
                                           p,
                                           gputable->rows.size());
        cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_mem_size, gpuTests, gpuTestsExclusiveSum, testSize + 1);
        uint passedRowsNum;
        cudaMemcpy(&passedRowsNum, gpuTestsExclusiveSum + testSize, sizeof(uint), cudaMemcpyDeviceToHost);
        result.clear();
        if (passedRowsNum != 0) {
            uint *gpuPassedRowsIndexes = gpudb::GpuStackAllocator::getInstance().alloc<uint>(passedRowsNum);
            gpudb::GpuRow **gpuRows = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::GpuRow *>(passedRowsNum);
            uint *cpuPassedRowsIndexes = StackAllocator::getInstance().alloc<uint>(passedRowsNum);
            gpudb::GpuRow ** cpuGpuRows = StackAllocator::getInstance().alloc<gpudb::GpuRow *>(passedRowsNum);

            if (gpuPassedRowsIndexes == nullptr || gpuRows == nullptr || cpuGpuRows == nullptr || cpuPassedRowsIndexes == nullptr) {
                gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought stack memory");
                break;
            }

            fillGpuPassedRows<<<grid, block>>>(thrust::raw_pointer_cast(gputable->rows.data()),
                                               gpuTests,
                                               gpuTestsExclusiveSum,
                                               gpuRows,
                                               gpuPassedRowsIndexes,
                                               testSize);
            cudaMemcpy(cpuPassedRowsIndexes, gpuPassedRowsIndexes, sizeof(uint) * passedRowsNum, cudaMemcpyDeviceToHost);
            cudaMemcpy(cpuGpuRows, gpuRows, sizeof(gpudb::GpuRow *) * passedRowsNum, cudaMemcpyDeviceToHost);

            for (uint i = 0; i < passedRowsNum; i++) {
                uint memsize = gputable->rowsSize[cpuPassedRowsIndexes[i]];
                uint8_t *memory = StackAllocator::getInstance().alloc<uint8_t>(memsize);
                gpudb::GpuRow *cpuRow = (gpudb::GpuRow *) memory;
                loadCPU(cpuRow, cpuGpuRows[i], memsize);

                Row newRow;

                newRow.temporalKey.name = std::string(cpuRow->temporalPart.name);
                Date date;
                date.setFromCode(cpuRow->temporalPart.transactionTimeCode);
                newRow.temporalKey.transactionTime = date;

                date.setFromCode(cpuRow->temporalPart.validTimeSCode);
                newRow.temporalKey.validTimeS = date;

                date.setFromCode(cpuRow->temporalPart.validTimeECode);
                newRow.temporalKey.validTimeE = date;

                newRow.temporalKey.type = cpuRow->temporalPart.type;

                newRow.spatialKey.type = cpuRow->spatialPart.type;

                switch(cpuRow->spatialPart.type) {
                    case SpatialType::POINT:
                    {
                        gpudb::GpuPoint *p = static_cast<gpudb::GpuPoint *>(cpuRow->spatialPart.key);
                        newRow.spatialKey.points.push_back(p->p);
                    }
                    break;
                    case SpatialType::LINE:
                    {
                        gpudb::GpuLine *p = static_cast<gpudb::GpuLine *>(cpuRow->spatialPart.key);
                        newRow.spatialKey.points.resize(p->size);

                        for (uint j = 0; j < p->size; j++) {
                            newRow.spatialKey.points[j] = p->points[j];
                        }
                    }
                    break;
                    case SpatialType::POLYGON:
                    {
                        gpudb::GpuPolygon *p = static_cast<gpudb::GpuPolygon *>(cpuRow->spatialPart.key);
                        newRow.spatialKey.points.resize(p->size);

                        for (uint j = 0; j < p->size; j++) {
                            newRow.spatialKey.points[j] = p->points[j];
                        }
                    }
                    break;
                }

                newRow.values.resize(cpuRow->valueSize);
                for (uint j = 0; j < cpuRow->valueSize; j++) {
                    Attribute &atr = newRow.values[j];

                    atr.setName(desc.columnDescription[j].name);
                    atr.isNullVal = cpuRow->value[j].isNull;
                    if (cpuRow->value[j].isNull) {
                        atr.setNullValue(desc.columnDescription[j].type);
                    } else {
                        atr.type = desc.columnDescription[j].type;
                        atr.setValueImp(desc.columnDescription[j].type, cpuRow->value[j].value);
                    }
                }
                result.push_back(std::move(newRow));
                StackAllocator::getInstance().free(memory);
            }
        }
        gpudb::GpuStackAllocator::getInstance().popPosition();
        StackAllocator::getInstance().popPosition();
        return true;
    } while(0);
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();
    return false;
}


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
void updaterKernel(gpudb::GpuColumnAttribute *columns,
                   gpudb::GpuRow **rows,
                   gpudb::Value *atrValues,
                   char **atrNames,
                   uint atrSize,
                   Predicate p,
                   uint workSize)
{
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
void testRowsOnPredicate(gpudb::GpuRow **rows,
                         gpudb::GpuColumnAttribute *columns,
                         uint columnsSize,
                         uint64_t *result,
                         uint64_t *resultInverse,
                         Predicate p,
                         uint size)
{
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
void distributor(gpudb::GpuRow **rows,
                 uint64_t *decision,
                 uint64_t *offset,
                 gpudb::GpuRow **result,
                 uint64_t *toDeleteOffeset,
                 gpudb::GpuRow **toDelete,
                 uint size)
{
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

bool DataBase::dropRowImp(gpudb::GpuTable *table, Predicate p, bool freeRowMemory) {
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
            cpuPrefixSum == nullptr || cpuPrefixSumInverse == nullptr ||
            prefixSumInverse == nullptr || resultInverse == nullptr) {
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought stack memory");
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

        if (toDeleteSize == 0) { // хотели удалить 0 строк? Мы удалили
            return true;
        }

        gpudb::GpuRow **toDelete = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::GpuRow *>(toDeleteSize);
        gpudb::GpuRow **toDeleteCpu = StackAllocator::getInstance().alloc<gpudb::GpuRow *>(toDeleteSize);

        if (toDelete == nullptr || toDeleteCpu == nullptr) {
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought stack memory");
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


        if (toSaveSize > 0 && freeRowMemory == true) {
            for (uint i = 0; i < table->rowsSize.size(); i++) {
                sizes[cpuPrefixSumInverse[i]] = table->rowsSize[i];
            }
        }

        swap(table->rows, resultRows);
        swap(table->rowsSize, sizes);

        if (freeRowMemory == true) {
            cudaMemcpy(toDeleteCpu, toDelete, sizeof(gpudb::GpuRow *) * (toDeleteSize), cudaMemcpyDeviceToHost);
            for (uint i = 0; i < toDeleteSize; i++) {
                gpudb::gpuAllocator::getInstance().free(toDeleteCpu[i]);
            }
        }
        gpudb::GpuStackAllocator::getInstance().popPosition();
        StackAllocator::getInstance().popPosition();
        return true;
    } while(0);
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();
    return false;
}

bool DataBase::dropRow(TempTable &t, Predicate p) {
    if (!t.isValid()) {
        return false;
    }
    bool freeM = !t.table->rowReferenses;
    return dropRowImp(t.table, p, freeM);
}

bool DataBase::dropRow(std::string tableName, Predicate p) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);
    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return false;
    }

    gpudb::GpuTable *table = (*tableIt).second;
    return dropRowImp(table, p, true);

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

bool DataBase::update(TempTable &t, std::set<Attribute> const &atrSet, Predicate p) {
    return updateImp(t.description, t.table, atrSet, p);
}

bool DataBase::update(std::string tableName, std::set<Attribute> const &atrSet, Predicate p) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);
    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return false;
    }

    TableDescription &desc = (*tableDesc).second;
    gpudb::GpuTable *table = (*tableIt).second;
    return updateImp(desc, table, atrSet, p);
}

bool DataBase::updateImp(TableDescription &desc, gpudb::GpuTable *table, std::set<Attribute> const &atrSet, Predicate p) {
    if (table->rows.size() == 0) {
        return false;
    }

    std::vector<Attribute> atrVec(atrSet.begin(), atrSet.end());

    bool canBeUsed = false;
    uint64_t memsize = 0;
    uint j = 0;
    for (uint i = 0; i < desc.columnDescription.size(); i++) {
        if (desc.columnDescription[i].name == atrVec[j].name && desc.columnDescription[i].type == atrVec[j].type) {

            if (atrVec[j].type == Type::SET) {
                gLogWrite(LOG_MESSAGE_TYPE::ERROR, "this attribute set cannot be used");
                return false;
            }

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

    for (uint i = 0; i < atrVec.size(); i++) {
        cpuValues[i].isNull = atrVec[i].isNullVal;
        cpuValues[i].value = (void*)cpuValuesMemoryPointer;
        cpuStringsPointers[i] = cpuStringsMemoryPointer;

        if (cpuValues[i].isNull == false) {
            memcpy(cpuValues[i].value, atrVec[i].value, typeSize(atrVec[i].type));
        }
        strncpy(cpuStringsPointers[i], atrVec[i].name.c_str(), typeSize(atrVec[i].type));

        cpuValuesMemoryPointer += typeSize(atrVec[i].type);
        cpuStringsMemoryPointer += typeSize(Type::STRING);
    }

    for (uint i = 0; i < atrVec.size(); i++) {
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
    return true;
}

////// INSERT

__global__
void computeStandartBoundingBox(gpudb::GpuRow **rows, gpudb::AABB *boxes, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    gpudb::AABB box;
    rows[idx]->spatialPart.boundingBox(&box);
    rows[idx]->temporalPart.boundingBox(&box);
    boxes[idx] = box;
}

__device__
AABBRelation boxIntersection4D(float4 aMin, float4 aMax, float4 bMin, float4 bMax) {
    bool a1 = aMax.x < bMin.x;
    bool a2 = aMin.x > bMax.x;

    bool a3 = aMax.y < bMin.y;
    bool a4 = aMin.y > bMax.y;

    bool a5 = aMax.z < bMin.z;
    bool a6 = aMin.z > bMax.z;

    bool a7 = aMax.w < bMin.w;
    bool a8 = aMin.w > bMax.w;

    if (a1 || a2 || a3 || a4 || a5 || a6 || a7 || a8) {
        return AABBRelation::DISJOINT;
    }

    if (aMin.x == bMin.x && aMin.y == bMin.y && aMin.z == bMin.z && aMin.w == bMin.w &&
        aMax.x == bMax.x && aMax.y == bMax.y && aMax.z == bMax.z && aMax.w == bMax.w) {
        return AABBRelation::EQUAL;
    }

    return AABBRelation::OVERLAP;
}

__global__
void testEqualBoxes(gpudb::AABB *newBoxes, gpudb::HLBVH bvh, uint *testResults, uint *stack, uint stackSize, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    float4 min, max;
    gpudb::AABB box = newBoxes[idx];
    min = make_float4(AABBmin(box.x), AABBmin(box.y), AABBmin(box.z), AABBmin(box.w));
    max = make_float4(AABBmax(box.x), AABBmax(box.y), AABBmax(box.z), AABBmax(box.w));

    GpuStack<uint> st(stack + idx * stackSize, stackSize);
    st.push(0);

    while(!st.empty()) {
        uint pos = st.top(); st.pop();
        int link1 = bvh.links[pos];
        int link2 = bvh.links[pos + 1];
        float4 box1Min = bvh.aabbMin[pos];
        float4 box1Max = bvh.aabbMax[pos];
        float4 box2Min = bvh.aabbMin[pos + 1];
        float4 box2Max = bvh.aabbMax[pos + 1];

        AABBRelation r1 = boxIntersection4D(min, max, box1Min, box1Max);
        AABBRelation r2 = boxIntersection4D(min, max, box2Min, box2Max);

        if (link1 == LEAF && r1 == AABBRelation::EQUAL) {
            for (uint i = getLeftBound(bvh.ranges[pos]); i < getRightBound(bvh.ranges[pos]); i++) {
                testResults[bvh.references[i]] = 1;
            }
        }

        if (link1 != LEAF && r1 != AABBRelation::DISJOINT) {
                st.push(link1);
        }

        if (link2 == LEAF && r2 == AABBRelation::EQUAL) {
            for (uint i = getLeftBound(bvh.ranges[pos + 1]); i < getRightBound(bvh.ranges[pos + 1]); i++) {
                testResults[bvh.references[i]] = 1;
            }
        }

        if (link2 != LEAF && r2 != AABBRelation::DISJOINT) {
                st.push(link2);
        }
    }
}

template <SpatialType st, TemporalType tt>
struct GpuRowComparer {
    FUNC_PREFIX
    bool operator()(gpudb::GpuRow *row1, gpudb::GpuRow *row2) const {
         switch(tt) {
             case TemporalType::TRANSACTION_TIME:
             {
                 if (row1->temporalPart.transactionTimeCode < row2->temporalPart.transactionTimeCode) {
                     return true;
                 }

                 if (row1->temporalPart.transactionTimeCode > row2->temporalPart.transactionTimeCode) {
                     return false;
                 }
             }
             break;
             case TemporalType::VALID_TIME:
             {
                 if (row1->temporalPart.validTimeSCode < row2->temporalPart.validTimeSCode) {
                     return true;
                 }

                 if (row1->temporalPart.validTimeSCode > row2->temporalPart.validTimeSCode) {
                     return false;
                 }

                 if (row1->temporalPart.validTimeECode < row2->temporalPart.validTimeECode) {
                     return true;
                 }

                 if (row1->temporalPart.validTimeECode > row2->temporalPart.validTimeECode) {
                     return false;
                 }
             }
             break;
             case TemporalType::BITEMPORAL_TIME:
             {
                 if (row1->temporalPart.transactionTimeCode < row2->temporalPart.transactionTimeCode) {
                     return true;
                 }

                 if (row1->temporalPart.transactionTimeCode > row2->temporalPart.transactionTimeCode) {
                     return false;
                 }

                 if (row1->temporalPart.validTimeSCode < row2->temporalPart.validTimeSCode) {
                     return true;
                 }

                 if (row1->temporalPart.validTimeSCode > row2->temporalPart.validTimeSCode) {
                     return false;
                 }

                 if (row1->temporalPart.validTimeECode < row2->temporalPart.validTimeECode) {
                     return true;
                 }

                 if (row1->temporalPart.validTimeECode > row2->temporalPart.validTimeECode) {
                     return false;
                 }
             }
             break;
         }

         switch(st) {
             case SpatialType::POINT:
             {
                 float2 p1 = ((gpudb::GpuPoint *)row1->spatialPart.key)->p;
                 float2 p2 = ((gpudb::GpuPoint *)row2->spatialPart.key)->p;

                 if (p1.x < p2.x) {
                     return true;
                 }

                 if (p1.x > p2.x) {
                     return false;
                 }

                 if (p1.y < p2.y) {
                     return true;
                 }

                 if (p1.y > p2.y) {
                     return false;
                 }
             }
             break;
             case SpatialType::LINE:
             {
                 gpudb::GpuLine *l1 = (gpudb::GpuLine *)row1->spatialPart.key;
                 gpudb::GpuLine *l2 = (gpudb::GpuLine *)row2->spatialPart.key;

                 if (l1->size < l2->size) {
                     return true;
                 }

                 if (l1->size > l2->size) {
                     return false;
                 }

                 for (int i = 0; i < l1->size; i++) {
                     float2 p1 = l1->points[i];
                     float2 p2 = l2->points[i];

                     if (p1.x < p2.x) {
                         return true;
                     }

                     if (p1.x > p2.x) {
                         return false;
                     }

                     if (p1.y < p2.y) {
                         return true;
                     }

                     if (p1.y > p2.y) {
                         return false;
                     }
                 }
             }
             break;
             case SpatialType::POLYGON:
             {
                 gpudb::GpuPolygon *pol1 = (gpudb::GpuPolygon *)row1->spatialPart.key;
                 gpudb::GpuPolygon *pol2 = (gpudb::GpuPolygon *)row2->spatialPart.key;

                 if (pol1->size < pol2->size) {
                     return true;
                 }

                 if (pol1->size > pol2->size) {
                     return false;
                 }

                 for (int i = 0; i < pol1->size; i++) {
                     float2 p1 = pol1->points[i];
                     float2 p2 = pol2->points[i];

                     if (p1.x < p2.x) {
                         return true;
                     }

                     if (p1.x > p2.x) {
                         return false;
                     }

                     if (p1.y < p2.y) {
                         return true;
                     }

                     if (p1.y > p2.y) {
                         return false;
                     }
                 }
             }
             break;
         }
         return false;
    }
};

__global__
void copySelectedRows(gpudb::GpuRow **rows, gpudb::GpuRow **selected, uint *tests, uint *offset, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    if (tests[idx]) {
        selected[offset[idx]] = rows[idx];
    }
}

template<SpatialType sp, TemporalType tt> __global__
void testEqualSortedRows(gpudb::GpuRow **rows, uint *testResult, uint workSize) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= workSize) {
        return;
    }

    uint result = 0;
    if (idx < workSize - 1) {
        result = testIdenticalRowKeys<sp, tt>(rows[idx], rows[idx + 1]);
    }
    testResult[idx] = result;
}

#define SWITCH_mergesort(spatialtype, temporaltype, array, size, context) \
switch(spatialtype) { \
    case SpatialType::POINT: { \
        switch(temporaltype) { \
            case TemporalType::BITEMPORAL_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::POINT, TemporalType::BITEMPORAL_TIME>(), context); \
                break; \
            case TemporalType::VALID_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::POINT, TemporalType::VALID_TIME>(), context);\
                break; \
            case TemporalType::TRANSACTION_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::POINT, TemporalType::TRANSACTION_TIME>(), context);\
                break; \
        } \
    } \
    break; \
    case SpatialType::POLYGON: { \
        switch(temporaltype) { \
            case TemporalType::BITEMPORAL_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::POLYGON, TemporalType::BITEMPORAL_TIME>(), context); \
                break; \
            case TemporalType::VALID_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::POLYGON, TemporalType::VALID_TIME>(), context); \
                break; \
            case TemporalType::TRANSACTION_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::POLYGON, TemporalType::TRANSACTION_TIME>(), context); \
                break; \
        } \
    }\
    break; \
    case SpatialType::LINE: { \
        switch(temporaltype) { \
            case TemporalType::BITEMPORAL_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::LINE, TemporalType::BITEMPORAL_TIME>(), context); \
                break; \
            case TemporalType::VALID_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::LINE, TemporalType::VALID_TIME>(), context); \
                break; \
            case TemporalType::TRANSACTION_TIME: \
                mgpu::mergesort(array, size, GpuRowComparer<SpatialType::LINE, TemporalType::TRANSACTION_TIME>(), context); \
                break; \
        } \
    } \
    break; \
}

bool DataBase::insertRow(std::string tableName, std::vector<Row> &rows) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);
    if (tableIt == tables.end() || tableDesc == tablesType.end() || rows.size() == 0) {
        return false;
    }

    TableDescription &desc = (*tableDesc).second;
    gpudb::GpuTable *table = (*tableIt).second;

    for (uint j = 0; j < rows.size(); j++) {
        if (!validateRow(rows[j], desc)) {
            return false;
        }
    }

    std::vector<uint64_t> memorySize(rows.size());
    std::vector<gpudb::GpuRow*> gpuRows(rows.size());

    for (uint j = 0; j < rows.size(); j++) {
        gpuRows[j] = allocateRow(rows[j], desc, memorySize[j]);
        if (gpuRows[j] == nullptr) {
            for (int i = 0; i < j; i++) {
                gpudb::gpuAllocator::getInstance().free(gpuRows[i]);
            }
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enought memory for rows");
            return false;
        }
    }

    gpudb::GpuStackAllocator::getInstance().pushPosition();
    do {
        if (!table->bvh.isBuilded() && table->rows.size() > 0) {
            gpudb::AABB *boxes = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::AABB>(table->rows.size());

            if (boxes == nullptr) {
                break;
            }

            dim3 block(BLOCK_SIZE);
            dim3 grid = gridConfigure(table->rows.size(), block);
            computeStandartBoundingBox<<<grid, block>>>(thrust::raw_pointer_cast(table->rows.data()), boxes, table->rows.size());

            if (!table->bvh.build(boxes, table->rows.size())) {
                break;
            }
            gpudb::GpuStackAllocator::getInstance().free(boxes);

            uint stackSize = table->bvh.numBVHLevels * 2 + 1;

            uint *testResult = gpudb::GpuStackAllocator::getInstance().alloc<uint>(table->rows.size() + 1);
            uint *testResultPrefixSum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(table->rows.size() + 1);
            gpudb::GpuRow **newRowsPointers = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::GpuRow *>(rows.size());
            gpudb::AABB *newRowAABB = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::AABB>(rows.size());
            uint *stack = gpudb::GpuStackAllocator::getInstance().alloc<uint>(rows.size() * stackSize);

            uint8_t *cub_tmp_mem = nullptr;
            uint64_t cub_tmp_mem_size = 0;
            cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_mem_size, testResult, testResultPrefixSum, table->rows.size() + 1);
            cub_tmp_mem = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_mem_size);

            if (cub_tmp_mem == nullptr ||
                testResult == nullptr ||
                testResultPrefixSum == nullptr ||
                newRowsPointers == nullptr ||
                newRowAABB == nullptr ||
                stack == nullptr) {
               break;
            }

            cudaMemcpy(newRowsPointers, gpuRows.data(), sizeof(gpudb::GpuRow *) * rows.size(), cudaMemcpyHostToDevice);
            cudaMemset(testResult, 0, sizeof(uint) * table->rows.size() + 1);

            grid = gridConfigure(rows.size(), block);

            computeStandartBoundingBox<<<grid, block>>>(newRowsPointers, newRowAABB, rows.size());
            testEqualBoxes<<<block, grid>>>(newRowAABB, table->bvh, testResult, stack, stackSize, rows.size());

            cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_mem_size, testResult, testResultPrefixSum, table->rows.size() + 1);

            gpudb::GpuStackAllocator::getInstance().free(cub_tmp_mem);
            gpudb::GpuStackAllocator::getInstance().free(stack);
            gpudb::GpuStackAllocator::getInstance().free(newRowAABB);
            gpudb::GpuStackAllocator::getInstance().free(newRowsPointers);
            uint allSize = 0;
            cudaMemcpy(&allSize, testResultPrefixSum + table->rows.size(), sizeof(uint), cudaMemcpyDeviceToHost);
            // в стеке у нас последний массив это newRowsPointers. К нему добавим выбранные элементы и посортируем
            if (allSize > 0) {
                uint offsetInNewRowsPointers = rows.size();
                allSize += rows.size();
                newRowsPointers = gpudb::GpuStackAllocator::getInstance().alloc<gpudb::GpuRow *>(allSize);
                gpudb::GpuRow **selectedRows = newRowsPointers + offsetInNewRowsPointers;

                if (newRowsPointers == nullptr) {
                    break;
                }

                dim3 block(BLOCK_SIZE);
                dim3 grid = gridConfigure(table->rows.size(), block);
                copySelectedRows<<<grid, block>>>(thrust::raw_pointer_cast(table->rows.data()), selectedRows, testResult, testResultPrefixSum, table->rows.size());
                mgpu::standard_context_t context;
                SWITCH_mergesort(desc.spatialKeyType, desc.temporalKeyType, newRowsPointers, allSize, context);


                /*gpudb::GpuRow **cpuRows = StackAllocator::getInstance().alloc<gpudb::GpuRow *>(allSize);
                cudaMemcpy(cpuRows, newRowsPointers, allSize * sizeof(gpudb::GpuRow *), cudaMemcpyDeviceToHost);
                uint8_t *mem = StackAllocator::getInstance().alloc<uint8_t>(memorySize[0]);
                gpudb::GpuRow *cpurow = (gpudb::GpuRow *)mem;
                for (int i = 0; i < allSize; i++) {
                    loadCPU(cpurow, cpuRows[i], memorySize[0]);
                    printf("%d %zu %zu %zu %f %f\n", i, cpurow->temporalPart.transactionTimeCode,
                           cpurow->temporalPart.validTimeSCode,
                           cpurow->temporalPart.validTimeECode,
                           ((gpudb::GpuPoint *)(cpurow->spatialPart.key))->p.x,
                           ((gpudb::GpuPoint *)(cpurow->spatialPart.key))->p.y);
                }
*/


                uint *testEqualResult = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allSize + 1);
                uint *testEqualResultPrefixSum = gpudb::GpuStackAllocator::getInstance().alloc<uint>(allSize + 1);

                cub_tmp_mem = nullptr;
                cub_tmp_mem_size = 0;
                cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_mem_size, testEqualResult, testEqualResultPrefixSum, allSize + 1);
                cub_tmp_mem = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_mem_size);

                if (testEqualResult == nullptr || testEqualResultPrefixSum == nullptr || cub_tmp_mem == nullptr) {
                    break;
                }

                grid = gridConfigure(allSize, block);
                SWITCH_RUN(desc.spatialKeyType, desc.temporalKeyType, testEqualSortedRows, grid, block, newRowsPointers, testEqualResult, allSize);
                cub::DeviceScan::ExclusiveSum(cub_tmp_mem, cub_tmp_mem_size, testEqualResult, testEqualResultPrefixSum, allSize + 1);
                uint numNotUniqueRows = 0;
                cudaMemcpy(&numNotUniqueRows, testEqualResultPrefixSum + allSize, sizeof(uint), cudaMemcpyDeviceToHost);
                if (numNotUniqueRows > 0) {
                    break;
                }
            }
        }
        uint offset = table->rows.size();
        table->rows.resize(table->rows.size() + rows.size());
        table->rowsSize.resize(table->rowsSize.size() + rows.size());

        cudaMemcpy(thrust::raw_pointer_cast(table->rows.data()) + offset, gpuRows.data(), sizeof(gpudb::GpuRow *) * gpuRows.size(), cudaMemcpyHostToDevice);

        for (uint i = offset, j = 0; i < table->rowsSize.size(); i++, j++) {
            table->rowsSize[i] = memorySize[j];
        }
        gpudb::GpuStackAllocator::getInstance().popPosition();
        return true;
    } while(0);
    gpudb::GpuStackAllocator::getInstance().popPosition();
    for (uint j = 0; j < rows.size(); j++) {
        gpudb::gpuAllocator::getInstance().free(gpuRows[j]);
    }
    return false;
}
