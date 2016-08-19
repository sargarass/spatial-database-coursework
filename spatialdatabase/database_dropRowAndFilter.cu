#include "database.h"
#include <cub/cub/cub.cuh>

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

Result<void, Error<std::string>>
DataBase::dropRowImp(gpudb::GpuTable *table, Predicate p) {
    if (table->rows.size() == 0) {
        return MYERR_STRING("table has no rows");
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid = gridConfigure(table->rows.size(), block);

    auto result = gpudb::GpuStackAllocatorAdditions::allocUnique<uint64_t>(table->rows.size() + 1);
    auto prefixSum = gpudb::GpuStackAllocatorAdditions::allocUnique<uint64_t>(table->rows.size() + 1);
    auto resultInverse = gpudb::GpuStackAllocatorAdditions::allocUnique<uint64_t>(table->rows.size() + 1);
    auto prefixSumInverse = gpudb::GpuStackAllocatorAdditions::allocUnique<uint64_t>(table->rows.size() + 1);

    auto cpuPrefixSum = StackAllocatorAdditions::allocUnique<uint64_t>(table->rows.size() + 1);
    auto cpuPrefixSumInverse = StackAllocatorAdditions::allocUnique<uint64_t>(table->rows.size() + 1);
    size_t cub_tmp_memsize = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_memsize, result.get(), prefixSum.get(), table->rows.size() + 1);
    auto cub_tmp_mem = gpudb::GpuStackAllocatorAdditions::allocUnique<uint8_t>(cub_tmp_memsize);

    if (prefixSum == nullptr || result == nullptr || cub_tmp_mem == nullptr ||
        cpuPrefixSum == nullptr || cpuPrefixSumInverse == nullptr ||
        prefixSumInverse == nullptr || resultInverse == nullptr) {
        return MYERR_STRING("not enought stack memory");
    }

    testRowsOnPredicate<<<grid, block>>>(thrust::raw_pointer_cast(table->rows.data()),
                                         thrust::raw_pointer_cast(table->columns.data()),
                                         table->columns.size(),
                                         result.get(),
                                         resultInverse.get(),
                                         p,
                                         table->rows.size());

    cub::DeviceScan::ExclusiveSum(cub_tmp_mem.get(), cub_tmp_memsize, result.get(), prefixSum.get(), table->rows.size() + 1);
    cub::DeviceScan::ExclusiveSum(cub_tmp_mem.get(), cub_tmp_memsize, resultInverse.get(), prefixSumInverse.get(), table->rows.size() + 1);

    cudaMemcpy(cpuPrefixSum.get(), prefixSum.get(), sizeof(uint64_t) * (table->rows.size() + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuPrefixSumInverse.get(), prefixSumInverse.get(), sizeof(uint64_t) * (table->rows.size() + 1), cudaMemcpyDeviceToHost);

    uint64_t toSaveSize = cpuPrefixSumInverse.get()[table->rows.size()];
    uint64_t toDeleteSize = cpuPrefixSum.get()[table->rows.size()];

    if (toDeleteSize == 0) { // хотели удалить 0 строк? Мы удалили
        return Ok();
    }

    auto toDelete = gpudb::GpuStackAllocatorAdditions::allocUnique<gpudb::GpuRow *>(toDeleteSize);
    auto toDeleteCpu = StackAllocatorAdditions::allocUnique<gpudb::GpuRow *>(toDeleteSize);

    if (toDelete == nullptr || toDeleteCpu == nullptr) {
        return MYERR_STRING("not enought stack memory");
    }

    thrust::device_vector<gpudb::GpuRow *> resultRows;
    std::vector<uint64_t> sizes;
    if (toSaveSize > 0) {
        resultRows.resize(toSaveSize);
        sizes.resize(toSaveSize);
    }

    distributor<<<grid, block>>>(thrust::raw_pointer_cast(table->rows.data()),
                            resultInverse.get(),
                            prefixSumInverse.get(),
                            thrust::raw_pointer_cast(resultRows.data()),
                            prefixSum.get(),
                            toDelete.get(),
                            table->rows.size());


    if (toSaveSize > 0) {
        for (uint i = 0; i < table->rowsSize.size(); i++) {
            sizes[cpuPrefixSumInverse.get()[i]] = table->rowsSize[i];
        }
    }

    swap(table->rows, resultRows);
    swap(table->rowsSize, sizes);

    cudaMemcpy(toDeleteCpu.get(), toDelete.get(), sizeof(gpudb::GpuRow *) * (toDeleteSize), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < toDeleteSize; i++) {
        gpudb::gpuAllocator::getInstance().free(toDeleteCpu.get()[i]);
    }


    table->bvh.free();
    return Ok();
}



Result<void, Error<std::string>>
DataBase::dropRow(std::string tableName, Predicate p) {
    auto tableIt = this->tables.find(tableName);
    auto tableDesc = this->tablesType.find(tableName);

    if (tableIt == tables.end() || tableDesc == tablesType.end()) {
        return MYERR_STRING(string_format("table with \"%s\" name is not exist", tableName.c_str()));
    }

    gpudb::GpuTable *table = (*tableIt).second;
    return dropRowImp(table, p);

}

Result<std::unique_ptr<TempTable>, Error<std::string>> DataBase::filter(std::unique_ptr<TempTable> &t, Predicate p) {
    if (t == nullptr) {
        return MYERR_STRING("t is nullptr");
    }

    if (!t->isValid()) {
        return MYERR_STRING("t is invalid");
    }

    if (t->table->rows.size() == 0) {
        return MYERR_STRING("table has no rows");
    }

    std::unique_ptr<TempTable> resTT = std::make_unique<TempTable>();
    if (resTT == nullptr) {
        return MYERR_STRING("not enough memory");
    }
    resTT->parents.push_back(t.get());
    t->references.push_back(resTT.get());
    resTT->description = t->description;

    resTT->table = new (std::nothrow) gpudb::GpuTable;
    if (resTT->table == nullptr) {
        return MYERR_STRING("not enough memory");
    }
    memcpy(resTT->table->name, t->table->name, NAME_MAX_LEN);
    resTT->table->rowReferenses = true;
    resTT->table->columns = t->table->columns;

    dim3 block(BLOCK_SIZE);
    dim3 grid = gridConfigure(t->table->rows.size(), block);

    auto result = gpudb::GpuStackAllocatorAdditions::allocUnique<uint64_t>(t->table->rows.size() + 1);
    auto prefixSum = gpudb::GpuStackAllocatorAdditions::allocUnique<uint64_t>(t->table->rows.size() + 1);
    auto resultInverse = gpudb::GpuStackAllocatorAdditions::allocUnique<uint64_t>(t->table->rows.size() + 1);
    auto prefixSumInverse = gpudb::GpuStackAllocatorAdditions::allocUnique<uint64_t>(t->table->rows.size() + 1);

    auto cpuPrefixSum = StackAllocatorAdditions::allocUnique<uint64_t>(t->table->rows.size() + 1);
    auto cpuPrefixSumInverse = StackAllocatorAdditions::allocUnique<uint64_t>(t->table->rows.size() + 1);
    size_t cub_tmp_memsize = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_memsize, result.get(), prefixSum.get(), t->table->rows.size() + 1);
    auto cub_tmp_mem = gpudb::GpuStackAllocatorAdditions::allocUnique<uint8_t>(cub_tmp_memsize);

    if (prefixSum == nullptr || result == nullptr || cub_tmp_mem == nullptr ||
        cpuPrefixSum == nullptr || cpuPrefixSumInverse == nullptr ||
        prefixSumInverse == nullptr || resultInverse == nullptr) {
        return MYERR_STRING("not enought stack memory");
    }

    testRowsOnPredicate<<<grid, block>>>(thrust::raw_pointer_cast(t->table->rows.data()),
                                         thrust::raw_pointer_cast(t->table->columns.data()),
                                         t->table->columns.size(),
                                         result.get(),
                                         resultInverse.get(),
                                         p,
                                         t->table->rows.size());

    cub::DeviceScan::ExclusiveSum(cub_tmp_mem.get(), cub_tmp_memsize, result.get(), prefixSum.get(), t->table->rows.size() + 1);
    cub::DeviceScan::ExclusiveSum(cub_tmp_mem.get(), cub_tmp_memsize, resultInverse.get(), prefixSumInverse.get(), t->table->rows.size() + 1);

    cudaMemcpy(cpuPrefixSum.get(), prefixSum.get(), sizeof(uint64_t) * (t->table->rows.size() + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpuPrefixSumInverse.get(), prefixSumInverse.get(), sizeof(uint64_t) * (t->table->rows.size() + 1), cudaMemcpyDeviceToHost);

    uint64_t toSaveSize = cpuPrefixSumInverse.get()[t->table->rows.size()];
    uint64_t toDeleteSize = cpuPrefixSum.get()[t->table->rows.size()];

    if (toDeleteSize == 0) { // хотели удалить 0 строк? Мы удалили
        resTT->valid = true;
        return Ok(std::move(resTT));
    }

    auto toDelete = gpudb::GpuStackAllocatorAdditions::allocUnique<gpudb::GpuRow *>(toDeleteSize);
    auto toDeleteCpu = StackAllocatorAdditions::allocUnique<gpudb::GpuRow *>(toDeleteSize);

    if (toDelete == nullptr || toDeleteCpu == nullptr) {
        return MYERR_STRING("not enought stack memory");
    }

    thrust::device_vector<gpudb::GpuRow *> resultRows;
    std::vector<uint64_t> sizes;
    if (toSaveSize > 0) {
        resultRows.resize(toSaveSize);
        sizes.resize(toSaveSize);
    }

    distributor<<<grid, block>>>(thrust::raw_pointer_cast(t->table->rows.data()),
                            resultInverse.get(),
                            prefixSumInverse.get(),
                            thrust::raw_pointer_cast(resultRows.data()),
                            prefixSum.get(),
                            toDelete.get(),
                            t->table->rows.size());


    if (toSaveSize > 0) {
        for (uint i = 0; i < t->table->rowsSize.size(); i++) {
            sizes[cpuPrefixSumInverse.get()[i]] = t->table->rowsSize[i];
        }
    }

    swap(resTT->table->rows, resultRows);
    swap(resTT->table->rowsSize, sizes);

    resTT->table->bvh.free();
    resTT->valid = true;
    return Ok(std::move(resTT));
}
