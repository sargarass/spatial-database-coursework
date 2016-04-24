#include "hlbvh.h"
#include "../externlibs/cub/cub/cub.cuh"
bool gpudb::HLBVH::alloc(uint32_t size) {
    do {
        links = gpuAllocator::getInstance().alloc<int>( 2 * size );
        if (links == nullptr) { break; }
        parents = gpuAllocator::getInstance().alloc<int>( 2 * size );
        if (parents == nullptr) { break; }
        ranges = gpuAllocator::getInstance().alloc<uint2>( 2 * size );
        if (ranges == nullptr) { break; }

        this->numBVHLevels = 0;
        this->numNodes = 0;
        this->numReferences = size;
        return true;
    } while(0);

    if (links) {
        gpuAllocator::getInstance().free(links);
    }

    if (parents) {
        gpuAllocator::getInstance().free(parents);
    }

    if (ranges) {
        gpuAllocator::getInstance().free(ranges);
    }
    return false;
}

void gpudb::HLBVH::free() {
    if (links) {
        gpuAllocator::getInstance().free(links);
    }

    if (parents) {
        gpuAllocator::getInstance().free(parents);
    }

    if (ranges) {
        gpuAllocator::getInstance().free(ranges);
    }
    this->numBVHLevels = 0;
    this->numNodes = 0;
    this->numReferences = 0;
}

static __device__ __inline__
uint getGlobalIdx3DZ() {
    uint blockId = blockIdx.x
                 + blockIdx.y * gridDim.x
                 + gridDim.x * gridDim.y * blockIdx.z;
    return blockId * blockDim.z + threadIdx.z;
}

static __device__ __inline__
uint getGlobalIdx3DZXY()
{
    uint blockId = blockIdx.x
             + blockIdx.y * gridDim.x
             + gridDim.x * gridDim.y * blockIdx.z;
    return blockId * (blockDim.x * blockDim.y * blockDim.z)
              + (threadIdx.z * (blockDim.x * blockDim.y))
              + (threadIdx.y * blockDim.x)
              + threadIdx.x;
}

static dim3 gridConfigure(uint64_t problemSize, dim3 block) {
    /// TODO
    /*dim3 MaxGridDim = {(uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[0],
                       (uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[1],
                       (uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[2]};
    dim3 gridDim = {1, 1, 1};

    uint64_t blockSize = block.x * block.y * block.z;

    if (problemSize > MaxGridDim.y * MaxGridDim.x * blockSize) {
        gridDim.z = problemSize / MaxGridDim.x * MaxGridDim.y * blockSize;
        problemSize = problemSize % MaxGridDim.x * MaxGridDim.y * blockSize;
    }

    if (problemSize > MaxGridDim.x * blockSize) {
        gridDim.y = problemSize / MaxGridDim.x * blockSize;
        problemSize = problemSize % MaxGridDim.x * blockSize;
    }

    gridDim.x = (problemSize + blockSize - 1) / blockSize;*/

    return dim3((problemSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

static __global__
void computeMortonCodesAndReferenceKernel(gpudb::MortonCode *keys, int *values, gpudb::AABB *aabb, uint32_t size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    keys[thread] = aabb[thread].getMortonCode();
    values[thread] = thread;
}

void computeMortonCodesAndReference(gpudb::MortonCode *keys, int *values, gpudb::AABB *aabb, uint size) {
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = gridConfigure(size, block);
    computeMortonCodesAndReferenceKernel<<<grid,block>>>(keys, values, aabb, size);
}

__global__
void initKeys(uint64_t *keys, int *values, gpudb::MortonCode *codes, uint size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }
    keys[thread] = codes[thread].high;
    values[thread] = thread;
}

__global__
void computeDiff(uint64_t *keys, int *array, uint size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    if ((thread + 1) < size && keys[thread] != keys[thread + 1])  {
        array[thread] = 1;
    } else {
        array[thread] = 0;
    }
}


template<bool high> __global__
void writeNewKeys(uint64_t *keys, int *values, gpudb::MortonCode *codes, int *prefixSum, uint size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }
    keys[thread] = 0;
    if (high) {
        keys[thread] = ((uint64_t)prefixSum[thread]) << 32ULL | ((codes[values[thread]].low & 0xFFFFFFFF00000000ULL) >> 32ULL);
    } else {
        keys[thread] = ((uint64_t)prefixSum[thread]) << 32ULL | ((codes[values[thread]].low & 0x00000000FFFFFFFFULL));
    }
}

__global__
void copyKeys(gpudb::MortonCode *new_keys, gpudb::MortonCode *old_keys, int *new_values, int *old_values, uint size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    new_keys[thread].bits = old_keys[old_values[thread]].bits;
    new_keys[thread].low = old_keys[old_values[thread]].low;
    new_keys[thread].high = old_keys[old_values[thread]].high;
    new_values[thread] = old_values[thread];
}

bool sortMortonCodes(gpudb::MortonCode *keys, int *values, uint size) {
    uint64_t *cub_keys[2];
    int *cub_values[2];
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = gridConfigure(size, block);

    int switcher = 0;
    for (int i = 0; i < 2; i++) {
        cub_keys[i] = gpudb::GpuStackAllocator::getInstance().alloc<uint64_t>(size);
        cub_values[i]  = gpudb::GpuStackAllocator::getInstance().alloc<int>(size);
    }

    size_t cub_tmp_memory_size = 0;
    size_t cub_tmp_memory_size2 = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, cub_tmp_memory_size, cub_keys[0], cub_keys[0], cub_keys[0], cub_keys[0], size);
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_memory_size2, cub_values[0], cub_values[0], size);
    cub_tmp_memory_size = std::max(cub_tmp_memory_size, cub_tmp_memory_size2);
    void *cub_tmp_memory  = (void *)gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_memory_size);

    if (cub_keys[0] == nullptr || cub_values[0] == nullptr ||
        cub_keys[1] == nullptr || cub_values[1] == nullptr ||
        cub_tmp_memory == nullptr) {
        return false;
    }
    // закончили с выделением памяти, фух

    // первая сортировка
    initKeys<<<grid, block>>>(cub_keys[switcher], cub_values[switcher], keys, size);
    cub::DeviceRadixSort::SortPairs(cub_tmp_memory, cub_tmp_memory_size,
                                    cub_keys[switcher], cub_keys[1 - switcher],
                                    cub_values[switcher], cub_values[1 - switcher], size);
    switcher = 1 - switcher;
    // теперь нам надо ещё 2. Чтобы сохранить порядок, посчитаем префиксную сумму
    // и в качестве ключа будем использовать
    // prefixsum << 32 | keypart

    // отсортированное в cub_keys/cub_values[switcher], эти два массива теперь не нужны

    int *arrayPrefixSum = reinterpret_cast<int*>(cub_keys[1 - switcher]);
    int *array = reinterpret_cast<int*>(cub_values[1 - switcher]);
    computeDiff<<<grid, block>>>(cub_keys[switcher], array, size);
    cub::DeviceScan::ExclusiveSum(cub_tmp_memory, cub_tmp_memory_size, array, arrayPrefixSum, size);
    writeNewKeys<true><<<grid, block>>>(cub_keys[switcher], cub_values[switcher], keys, arrayPrefixSum, size);
    cub::DeviceRadixSort::SortPairs(cub_tmp_memory, cub_tmp_memory_size,
                                    cub_keys[switcher], cub_keys[1 - switcher],
                                    cub_values[switcher], cub_values[1 - switcher], size);

    switcher = 1 - switcher;
    arrayPrefixSum = reinterpret_cast<int*>(cub_keys[1 - switcher]);
    array = reinterpret_cast<int*>(cub_values[1 - switcher]);
    computeDiff<<<grid, block>>>(cub_keys[switcher], array, size);
    cub::DeviceScan::ExclusiveSum(cub_tmp_memory, cub_tmp_memory_size, array, arrayPrefixSum, size);
    writeNewKeys<false><<<grid, block>>>(cub_keys[switcher], cub_values[switcher], keys, arrayPrefixSum, size);
    cub::DeviceRadixSort::SortPairs(cub_tmp_memory, cub_tmp_memory_size,
                                    cub_keys[switcher], cub_keys[1 - switcher],
                                    cub_values[switcher], cub_values[1 - switcher], size);

    // памяти от cub_keys[0] до cub_values[0] должно хватить, чтобы вместить копию мортон кодов...
  //  cudaMemcpy(cub_values[switcher], cub_values[1 - switcher], sizeof(int) * size, cudaMemcpyDeviceToDevice);
    gpudb::MortonCode *old = reinterpret_cast<gpudb::MortonCode *> (cub_keys[0]);
    cudaMemcpy(old, keys, sizeof(gpudb::MortonCode) * size, cudaMemcpyDeviceToDevice);
    copyKeys<<<grid, block>>>(keys, old, values, cub_values[1], size);
    return true;
}

bool gpudb::HLBVH::build(AABB *aabb, uint32_t size) {
    if (size == 0) { return false; }
    gpudb::GpuStackAllocator &gpuStackAlloc = gpudb::GpuStackAllocator::getInstance();
    if (alloc(size) == false) {
        return false;
    }

    do {
        MortonCode *keys = gpuStackAlloc.alloc<MortonCode>(size);
        int *values = gpuStackAlloc.alloc<int>(size);

        if (keys == nullptr || values == nullptr) {
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enough memory");
            break;
        }
        computeMortonCodesAndReference(keys, values, aabb, size);
   /*     MortonCode *cpuKeys = StackAllocator::getInstance().alloc<MortonCode>(size);
        int *cpuValues = StackAllocator::getInstance().alloc<int>(size);
        MortonCode *cpuKeys2 = StackAllocator::getInstance().alloc<MortonCode>(size);
        int *cpuValues2 = StackAllocator::getInstance().alloc<int>(size);

        cudaMemcpy(cpuKeys2, keys, sizeof(MortonCode) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuValues2, values, sizeof(int) * size, cudaMemcpyDeviceToHost);

        cudaMemcpy(cpuKeys, keys, sizeof(MortonCode) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuValues, values, sizeof(int) * size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++) {
            printf("{ (%s), %d }\n", cpuKeys[i].toString().c_str(), cpuValues[i]);
        }
*/
        if (!sortMortonCodes(keys, values, size)) {
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enough memory");
            break;
        }

       /* cudaMemcpy(cpuKeys, keys, sizeof(MortonCode) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuValues, values, sizeof(int) * size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++) {
            printf("{ (%s), %d }\n", cpuKeys[i].toString().c_str(), cpuValues[i]);
            cpuKeys2[i].bits = cpuValues2[i];
        }
        printf("\n\n\n");
        std::sort(cpuKeys2, cpuKeys2 + size);

        for (int i = 0; i < size; i++) {
            if (cpuValues[i] != cpuKeys2[i].bits) {
                printf("{ %d }\n", cpuValues[i] == cpuKeys2[i].bits);
            }
        }
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "end hlbvh build");*/
    } while(0);
    gpuStackAlloc.popPosition();
    return false;
}
