#include "database.h"
#include <cub/cub/cub.cuh>
#include "moderngpu/src/moderngpu/context.hxx"
#include "moderngpu/src/moderngpu/kernel_mergesort.hxx"

Result<std::vector<Row>, Error<std::string>> DataBase::selectRow(std::string tableName, Predicate p) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);

    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return MYERR_STRING(string_format("Table \"%s\" is not exist", tableName.c_str()));
    }

    TableDescription &desc = (*tableDesc).second;
    gpudb::GpuTable *table = (*tableIt).second;
    return selectRowImp(desc, table, p);
}

Result<std::vector<Row>, Error<std::string>>
DataBase::selectRow(std::unique_ptr<TempTable> &table, Predicate p) {
    if (table == nullptr) {
        return MYERR_STRING("table is nullptr");
    }

    if (!table->isValid()) {
        return MYERR_STRING("TempTable is invalid");
    }

    return selectRowImp(table->description, table->table, p);
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

Result<std::vector<Row>, Error<std::string>>
DataBase::selectRowImp(TableDescription &desc, gpudb::GpuTable *gputable, Predicate p) {
    uint testSize = gputable->rows.size();
    auto gpuTests = gpudb::GpuStackAllocatorAdditions::allocUnique<uint>(testSize + 1);
    auto gpuTestsExclusiveSum = gpudb::GpuStackAllocatorAdditions::allocUnique<uint>(testSize + 1);

    uint64_t cub_tmp_mem_size = 0;

    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_mem_size, gpuTests.get(), gpuTestsExclusiveSum.get(), testSize + 1);
    auto cub_tmp_mem = gpudb::GpuStackAllocatorAdditions::allocUnique<uint8_t>(cub_tmp_mem_size);

    if (gpuTests == nullptr || gpuTestsExclusiveSum == nullptr || cub_tmp_mem == nullptr) {
        return MYERR_STRING("not enought stack memory");
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid = gridConfigure(gputable->rows.size(), block);
    runTestsPredicate<<<grid, block>>>(thrust::raw_pointer_cast(gputable->rows.data()),
                                       thrust::raw_pointer_cast(gputable->columns.data()),
                                       gputable->columns.size(),
                                       gpuTests.get(),
                                       p,
                                       gputable->rows.size());
    cub::DeviceScan::ExclusiveSum(cub_tmp_mem.get(), cub_tmp_mem_size, gpuTests.get(), gpuTestsExclusiveSum.get(), testSize + 1);
    uint passedRowsNum;
    cudaMemcpy(&passedRowsNum, gpuTestsExclusiveSum.get() + testSize, sizeof(uint), cudaMemcpyDeviceToHost);
    std::vector<Row> result;
    if (passedRowsNum != 0) {
        auto gpuPassedRowsIndexes = gpudb::GpuStackAllocatorAdditions::allocUnique<uint>(passedRowsNum);
        auto gpuRows = gpudb::GpuStackAllocatorAdditions::allocUnique<gpudb::GpuRow *>(passedRowsNum);
        auto cpuPassedRowsIndexes = StackAllocatorAdditions::allocUnique<uint>(passedRowsNum);
        auto cpuGpuRows = StackAllocatorAdditions::allocUnique<gpudb::GpuRow *>(passedRowsNum);

        if (gpuPassedRowsIndexes == nullptr
            || gpuRows == nullptr
            || cpuGpuRows == nullptr
            || cpuPassedRowsIndexes == nullptr) {
            if (gpuTests == nullptr || gpuTestsExclusiveSum == nullptr || cub_tmp_mem == nullptr) {
                return MYERR_STRING("not enought stack memory");
            }
        }

        fillGpuPassedRows<<<grid, block>>>(thrust::raw_pointer_cast(gputable->rows.data()),
                                           gpuTests.get(),
                                           gpuTestsExclusiveSum.get(),
                                           gpuRows.get(),
                                           gpuPassedRowsIndexes.get(),
                                           testSize);
        cudaMemcpy(cpuPassedRowsIndexes.get(), gpuPassedRowsIndexes.get(), sizeof(uint) * passedRowsNum, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuGpuRows.get(), gpuRows.get(), sizeof(gpudb::GpuRow *) * passedRowsNum, cudaMemcpyDeviceToHost);

        for (uint i = 0; i < passedRowsNum; i++) {
            uint memsize = gputable->rowsSize[cpuPassedRowsIndexes.get()[i]];
            auto memory = StackAllocatorAdditions::allocUnique<uint8_t>(memsize);
            gpudb::GpuRow *cpuRow = (gpudb::GpuRow *) (memory.get());
            loadCPU(cpuRow, cpuGpuRows.get()[i], memsize);

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
        }
    }
    return Ok(std::move(result));
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


Result<void, Error<std::string>>
DataBase::dropTable(std::string tableName) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);

    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return MYERR_STRING(string_format("table with \"%s\" name is not exist", tableName.c_str()));
    }

    gpudb::GpuTable *table = (*tableIt).second;

    delete table;
    this->tables.erase(tableIt);
    this->tablesType.erase(tableDesc);
    return Ok();
}

Result<void, Error<std::string>>
DataBase::update(std::unique_ptr<TempTable> &t, std::set<Attribute> const &atrSet, Predicate p) {
    if (t == nullptr) {
        return MYERR_STRING("t is nullptr");
    }

    if (!t->isValid()) {
        return MYERR_STRING("t is invalid");
    }

    return updateImp(t->description, t->table, atrSet, p);
}

Result<void, Error<std::string>>
DataBase::update(std::string tableName, std::set<Attribute> const &atrSet, Predicate p) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);
    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return MYERR_STRING(string_format("table with \"%s\" name is not exist", tableName.c_str()));
    }

    TableDescription &desc = (*tableDesc).second;
    gpudb::GpuTable *table = (*tableIt).second;
    return updateImp(desc, table, atrSet, p);
}

Result<void, Error<std::string>>
DataBase::updateImp(TableDescription &desc, gpudb::GpuTable *table, std::set<Attribute> const &atrSet, Predicate p) {
    if (table->rows.size() == 0) {
        return MYERR_STRING("table has no rows");
    }

    std::vector<Attribute> atrVec(atrSet.begin(), atrSet.end());

    bool canBeUsed = false;
    uint64_t memsize = 0;
    uint j = 0;
    for (uint i = 0; i < desc.columnDescription.size(); i++) {
        if (desc.columnDescription[i].name == atrVec[j].name && desc.columnDescription[i].type == atrVec[j].type) {

            if (atrVec[j].type == Type::SET) {
                return MYERR_STRING("this set of attributes cannot be used");
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
        return MYERR_STRING("this set of attributes cannot be used");
    }

    auto gpuValuesMemory = gpudb::GpuStackAllocatorAdditions::allocUnique<uint8_t>(memsize);
    auto gpuValues = gpudb::GpuStackAllocatorAdditions::allocUnique<gpudb::Value>(atrVec.size());
    auto gpuStringsMemory = gpudb::GpuStackAllocatorAdditions::allocUnique<char>(typeSize(Type::STRING) * atrVec.size());
    auto gpuStringsPointers = gpudb::GpuStackAllocatorAdditions::allocUnique<char*>(atrVec.size());

    if (gpuValuesMemory == nullptr
        || gpuValues == nullptr
        || gpuStringsMemory == nullptr
        || gpuStringsPointers == nullptr) {
        return MYERR_STRING("not enough gpu stack memory");
    }

    auto cpuValuesMemory = StackAllocatorAdditions::allocUnique<uint8_t>(memsize);
    auto cpuValues = StackAllocatorAdditions::allocUnique<gpudb::Value>(atrVec.size());
    auto cpuStringsMemory = StackAllocatorAdditions::allocUnique<char>(typeSize(Type::STRING) * atrVec.size());
    auto cpuStringsPointers = StackAllocatorAdditions::allocUnique<char*>(atrVec.size());

    // копируем значения на cpu
    uint8_t *cpuValuesMemoryPointer = cpuValuesMemory.get();
    char *cpuStringsMemoryPointer = cpuStringsMemory.get();

    for (uint i = 0; i < atrVec.size(); i++) {
        cpuValues.get()[i].isNull = atrVec[i].isNullVal;
        cpuValues.get()[i].value = (void*)cpuValuesMemoryPointer;
        cpuStringsPointers.get()[i] = cpuStringsMemoryPointer;

        if (cpuValues.get()[i].isNull == false) {
            memcpy(cpuValues.get()[i].value, atrVec[i].value, typeSize(atrVec[i].type));
        }
        strncpy(cpuStringsPointers.get()[i], atrVec[i].name.c_str(), typeSize(atrVec[i].type));

        cpuValuesMemoryPointer += typeSize(atrVec[i].type);
        cpuStringsMemoryPointer += typeSize(Type::STRING);
    }

    for (uint i = 0; i < atrVec.size(); i++) {
        cpuValues.get()[i].value = newAddress(cpuValues.get()[i].value, cpuValuesMemory.get(), gpuValuesMemory.get());
        cpuStringsPointers.get()[i] = newAddress(cpuStringsPointers.get()[i], cpuStringsMemory.get(), gpuStringsMemory.get());
    }

    cudaMemcpy(gpuValues.get(), cpuValues.get(), atrVec.size() * sizeof(gpudb::Value), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuStringsPointers.get(), cpuStringsPointers.get(), atrVec.size() * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuValuesMemory.get(), cpuValuesMemory.get(), memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuStringsMemory.get(), cpuStringsMemory.get(), typeSize(Type::STRING) * atrVec.size(), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid = gridConfigure(table->rows.size(), block);
    updaterKernel<<<block, grid>>>(thrust::raw_pointer_cast(table->columns.data()),
                                   thrust::raw_pointer_cast(table->rows.data()),
                                   gpuValues.get(),
                                   gpuStringsPointers.get(),
                                   atrVec.size(),
                                   p,
                                   table->rows.size());
    return Ok();
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

Result<void, Error<std::string>>
DataBase::insertRow(std::string tableName, std::vector<Row> &rows) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);
    if (tableIt == tables.end() || tableDesc == tablesType.end() || rows.size() == 0) {
        return MYERR_STRING("table with \"%s\" name is not exist");
    }

    TableDescription &desc = (*tableDesc).second;
    gpudb::GpuTable *table = (*tableIt).second;

    for (uint j = 0; j < rows.size(); j++) {
        TRY(validateRow(rows[j], desc));
    }

    std::vector<uint64_t> memorySize(rows.size());
    std::vector<gpudb::GpuRow*> gpuRows(rows.size());

    RAII_GC<gpudb::GpuRow> gc;
    for (uint j = 0; j < rows.size(); j++) {
        gpuRows[j] = TRY(allocateRow(rows[j], desc, memorySize[j]));
        gc.registrGPU(gpuRows[j]);
        if (gpuRows[j] == nullptr) {
            return MYERR_STRING("not enough memory for rows");
        }
    }

    dim3 block(BLOCK_SIZE);
    if (!table->bvh.isBuilded() && table->rows.size() > 0) {
        auto boxes = gpudb::GpuStackAllocatorAdditions::allocUnique<gpudb::AABB>(table->rows.size());

        if (boxes == nullptr) {
            return MYERR_STRING("not enough gpu stack memory");
        }

        dim3 grid = gridConfigure(table->rows.size(), block);
        computeStandartBoundingBox<<<grid, block>>>(thrust::raw_pointer_cast(table->rows.data()), boxes.get(), table->rows.size());
        TRY(table->bvh.build(boxes.get(), table->rows.size()));
    }

    uint stackSize = table->bvh.numBVHLevels * 2 + 1;

    auto testResult = gpudb::GpuStackAllocatorAdditions::allocUnique<uint>(table->rows.size() + 1);
    auto testResultPrefixSum = gpudb::GpuStackAllocatorAdditions::allocUnique<uint>(table->rows.size() + 1);
    auto newRowsPointers = gpudb::GpuStackAllocatorAdditions::allocUnique<gpudb::GpuRow *>(rows.size());
    auto newRowAABB = gpudb::GpuStackAllocatorAdditions::allocUnique<gpudb::AABB>(rows.size());
    auto stack = gpudb::GpuStackAllocatorAdditions::allocUnique<uint>(rows.size() * stackSize);

    uint64_t cub_tmp_mem_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_mem_size, testResult.get(), testResultPrefixSum.get(), table->rows.size() + 1);
    auto cub_tmp_mem = gpudb::GpuStackAllocatorAdditions::allocUnique<uint8_t>(cub_tmp_mem_size);

    if (cub_tmp_mem == nullptr ||
        testResult == nullptr ||
        testResultPrefixSum == nullptr ||
        newRowsPointers == nullptr ||
        newRowAABB == nullptr ||
        stack == nullptr) {
       return MYERR_STRING("not enough stack memory");
    }

    cudaMemcpy(newRowsPointers.get(), gpuRows.data(), sizeof(gpudb::GpuRow *) * rows.size(), cudaMemcpyHostToDevice);
    cudaMemset(testResult.get(), 0, sizeof(uint) * table->rows.size() + 1);

    dim3 grid = gridConfigure(rows.size(), block);

    computeStandartBoundingBox<<<grid, block>>>(newRowsPointers.get(), newRowAABB.get(), rows.size());
    testEqualBoxes<<<block, grid>>>(newRowAABB.get(), table->bvh, testResult.get(), stack.get(), stackSize, rows.size());

    cub::DeviceScan::ExclusiveSum(cub_tmp_mem.get(), cub_tmp_mem_size, testResult.get(), testResultPrefixSum.get(), table->rows.size() + 1);

    cub_tmp_mem.reset();
    stack.reset();
    newRowAABB.reset();
    newRowsPointers.reset();

    if (table->bvh.isBuilded()) {
        table->bvh.free();
    }

    uint allSize = 0;
    cudaMemcpy(&allSize, testResultPrefixSum.get() + table->rows.size(), sizeof(uint), cudaMemcpyDeviceToHost);
    // в стеке у нас последний массив это newRowsPointers. К нему добавим выбранные элементы и посортируем
    if (allSize > 0) {
        uint offsetInNewRowsPointers = rows.size();
        allSize += rows.size();
        newRowsPointers = gpudb::GpuStackAllocatorAdditions::allocUnique<gpudb::GpuRow *>(allSize);
        gpudb::GpuRow **selectedRows = newRowsPointers.get() + offsetInNewRowsPointers;

        if (newRowsPointers == nullptr) {
            return MYERR_STRING("not enough gpu stack memory");
        }

        dim3 block(BLOCK_SIZE);
        dim3 grid = gridConfigure(table->rows.size(), block);
        copySelectedRows<<<grid, block>>>(thrust::raw_pointer_cast(table->rows.data()), selectedRows, testResult.get(), testResultPrefixSum.get(), table->rows.size());
        mgpu::standard_context_t context;
        SWITCH_mergesort(desc.spatialKeyType, desc.temporalKeyType, newRowsPointers.get(), allSize, context);


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


        auto testEqualResult = gpudb::GpuStackAllocatorAdditions::allocUnique<uint>(allSize + 1);
        auto testEqualResultPrefixSum = gpudb::GpuStackAllocatorAdditions::allocUnique<uint>(allSize + 1);

        cub_tmp_mem_size = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_mem_size, testEqualResult.get(), testEqualResultPrefixSum.get(), allSize + 1);
        auto cub_tmp_mem = gpudb::GpuStackAllocatorAdditions::allocUnique<uint8_t>(cub_tmp_mem_size);

        if (testEqualResult == nullptr || testEqualResultPrefixSum == nullptr || cub_tmp_mem == nullptr) {
            return MYERR_STRING("not enough stack memory");
        }

        grid = gridConfigure(allSize, block);
        SWITCH_RUN(desc.spatialKeyType, desc.temporalKeyType, testEqualSortedRows, grid, block, newRowsPointers.get(), testEqualResult.get(), allSize);
        cub::DeviceScan::ExclusiveSum(cub_tmp_mem.get(), cub_tmp_mem_size, testEqualResult.get(), testEqualResultPrefixSum.get(), allSize + 1);
        uint numNotUniqueRows = 0;
        cudaMemcpy(&numNotUniqueRows, testEqualResultPrefixSum.get() + allSize, sizeof(uint), cudaMemcpyDeviceToHost);
        if (numNotUniqueRows > 0) {
            return MYERR_STRING("some of keys are already inside table");
        }
    }
    uint offset = table->rows.size();
    table->rows.resize(table->rows.size() + rows.size());
    table->rowsSize.resize(table->rowsSize.size() + rows.size());

    cudaMemcpy(thrust::raw_pointer_cast(table->rows.data()) + offset, gpuRows.data(), sizeof(gpudb::GpuRow *) * gpuRows.size(), cudaMemcpyHostToDevice);

    for (uint i = offset, j = 0; i < table->rowsSize.size(); i++, j++) {
        table->rowsSize[i] = memorySize[j];
    }

    gc.takeCPU();
    gc.takeGPU();
    return Ok();
}
