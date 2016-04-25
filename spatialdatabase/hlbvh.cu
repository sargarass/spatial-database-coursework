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
        aabbMin = gpuAllocator::getInstance().alloc<float4>( 2 * size );
        if (aabbMin == nullptr) { break; }
        aabbMax = gpuAllocator::getInstance().alloc<float4>( 2 * size );
        if (aabbMax == nullptr) { break; }

        this->numBVHLevels = 0;
        this->numNodes = 0;
        this->numReferences = size;
        return true;
    } while(0);

    free();
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

    if (aabbMin) {
        gpuAllocator::getInstance().free(aabbMin);
    }

    if (aabbMax) {
        gpuAllocator::getInstance().free(aabbMax);
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

///////////////////////////////////////
/// Morton code part {
static __global__
void computeMortonCodesAndReferenceKernel(gpudb::MortonCode *keys, int *values, gpudb::AABB *aabb, gpudb::AABB globalAABB, uint32_t size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    keys[thread] = aabb[thread].getMortonCode(globalAABB);
    values[thread] = thread;
}

void computeMortonCodesAndReference(gpudb::MortonCode *keys, int *values, gpudb::AABB *aabb, gpudb::AABB globalAABB, uint size) {
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = gridConfigure(size, block);
    computeMortonCodesAndReferenceKernel<<<grid,block>>>(keys, values, aabb, globalAABB, size);
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
    gpudb::GpuStackAllocator::getInstance().pushPosition();
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
        gpudb::GpuStackAllocator::getInstance().popPosition();
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
    cudaDeviceSynchronize();
    gpudb::GpuStackAllocator::getInstance().popPosition();
    return true;
}
/// } Morton code part
///////////////////////////////////////
/// Global AABB {
template<char comp, bool min>
__global__
void copyAABBComponent(float *dst, gpudb::AABB *aabb, uint32_t size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    switch(comp) {
        case 'x':
        {
            if (min) {
                dst[thread] = AABBmin(aabb[thread].x);
            } else {
                dst[thread] = AABBmax(aabb[thread].x);
            }
        }
        break;
        case 'y':
        {
            if (min) {
                dst[thread] = AABBmin(aabb[thread].y);
            } else {
                dst[thread] = AABBmax(aabb[thread].y);
            }
        }
        break;
        case 'z':
        {
            if (min) {
                dst[thread] = AABBmin(aabb[thread].z);
            } else {
                dst[thread] = AABBmax(aabb[thread].z);
            }
        }
        break;
        case 'w':
        {
            if (min) {
                dst[thread] = AABBmin(aabb[thread].w);
            } else {
                dst[thread] = AABBmax(aabb[thread].w);
            }
        }
        break;
    }
}

bool computeGlobalAABB(gpudb::AABB *aabb, uint32_t size, gpudb::AABB &result) {
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();
    do {
        float *array = gpudb::GpuStackAllocator::getInstance().alloc<float>(size);
        float *minmax = gpudb::GpuStackAllocator::getInstance().alloc<float>(8);
        float *cpuMinMax = StackAllocator::getInstance().alloc<float>(8);
        size_t cub_tmp_memory_size = 0;


        cub::DeviceReduce::Min(nullptr, cub_tmp_memory_size, array, minmax, size);
        uint8_t *cub_tmp_memory = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_memory_size);

        if (array == nullptr || minmax == nullptr || cub_tmp_memory == nullptr || cpuMinMax == nullptr) { break; }

        dim3 block = dim3(BLOCK_SIZE);
        dim3 grid = gridConfigure(size, block);
        copyAABBComponent<'x', true> <<<grid, block>>> (array, aabb, size);
        cub::DeviceReduce::Min(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 0, size);
        copyAABBComponent<'y', true> <<<grid, block>>> (array, aabb, size);
        cub::DeviceReduce::Min(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 1, size);
        copyAABBComponent<'z', true> <<<grid, block>>> (array, aabb, size);
        cub::DeviceReduce::Min(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 2, size);
        copyAABBComponent<'w', true> <<<grid, block>>> (array, aabb, size);
        cub::DeviceReduce::Min(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 3, size);

        copyAABBComponent<'x', false> <<<grid, block>>> (array, aabb, size);
        cub::DeviceReduce::Max(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 4, size);
        copyAABBComponent<'y', false> <<<grid, block>>> (array, aabb, size);
        cub::DeviceReduce::Max(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 5, size);
        copyAABBComponent<'z', false> <<<grid, block>>> (array, aabb, size);
        cub::DeviceReduce::Max(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 6, size);
        copyAABBComponent<'w', false> <<<grid, block>>> (array, aabb, size);
        cub::DeviceReduce::Max(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 7, size);

        cudaMemcpy(cpuMinMax, minmax, sizeof(float) * 8, cudaMemcpyDeviceToHost);

        result.x.x = cpuMinMax[0];
        result.y.x = cpuMinMax[1];
        result.z.x = cpuMinMax[2];
        result.w.x = cpuMinMax[3];

        result.x.y = cpuMinMax[4];
        result.y.y = cpuMinMax[5];
        result.z.y = cpuMinMax[6];
        result.w.y = cpuMinMax[7];

        gpudb::GpuStackAllocator::getInstance().popPosition();
        StackAllocator::getInstance().popPosition();
        return true;
    } while(0);

    gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enough memory");
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();
    return false;
}
/// } Global AABB
//////////////////////////////////////////
/// Build Tree topology {

struct WorkQueue {
    int *nodeId;
    uint2 *range;
};

void initQueue(WorkQueue &queue, uint32_t size) {
    int nodeId = 0;
    uint2 range;
    range.x = 0;
    range.y = size;
    cudaMemcpy(queue.nodeId, &nodeId, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(queue.range, &range, sizeof(uint2), cudaMemcpyHostToDevice);
}

#define getLeftBound(p) p.x
#define getRightBound(p) p.y
#define getRangeSize(p) (p.y - p.x)

#define clzll(x) __clzll((x))
#define clzllHost(x) ((x) == 0)? 64 : __builtin_clzll((x))

__global__
void split(gpudb::HLBVH hlbvh, gpudb::MortonCode *keys, uint32_t queueSize, WorkQueue qIn, WorkQueue qOut, uint *counter) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= queueSize) {
        return;
    }
    uint2 rangeLeft, rangeRight, range;
    range = qIn.range[thread];
    rangeRight = range;
    rangeLeft = range;

    bool isLeaf = true;

    int parent = qIn.nodeId[thread];
    if (getRangeSize(range) > 1) {
        isLeaf = false;
        gpudb::MortonCode keyA = keys[getLeftBound(range)];
        gpudb::MortonCode keyB = keys[getRightBound(range) - 1];
        uint64_t ha = 64;
        uint64_t mask;

        bool high = false;
        if (keyA.high != keyB.high) {
            ha = clzll(keyA.high ^ keyB.high);
            high = true;
        } else if (keyA.low != keyB.low) {
            ha = clzll(keyA.low ^ keyB.low);
        }

        if (ha == 64) {
            uint mid = getLeftBound(range) + (getRightBound(range) - getLeftBound(range)) / 2;
            getRightBound(rangeLeft) = getLeftBound(rangeRight) = mid;
        } else {
            mask = 1ULL << (64 - ha - 1);
            uint left, right;
            left = getLeftBound(range);
            right = getRightBound(range);
            bool test;
            uint mid;

            while (left < right) {
                mid = left + (right - left) / 2;
                if (high) {
                    test = (keys[mid].high & mask) > 0;
                } else {
                    test = (keys[mid].low & mask) > 0;
                }
                /* key[mid] > key[left] */
                if (test) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }

            getRightBound(rangeLeft) = left;
            getLeftBound(rangeRight) = left;
        }
    }

    hlbvh.ranges[parent] = range;
    if (isLeaf) {
        hlbvh.links[parent] = 0xFFFFFFFFU; // лист
    } else {
        uint offset = atomicAdd(counter, 2);
        uint left = hlbvh.numNodes + offset;

        hlbvh.links[parent] = left;

        qOut.nodeId[offset] = left;
        qOut.nodeId[offset + 1] = left + 1;
        qOut.range[offset] = rangeLeft;
        qOut.range[offset + 1] = rangeRight;
    }
}

bool buildTreeStructure(gpudb::HLBVH &hlbvh, gpudb::MortonCode *keys, int *values, uint32_t size) {
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();

    do {
        WorkQueue work[2];
        work[0].nodeId = gpudb::GpuStackAllocator::getInstance().alloc<int>(size);
        work[0].range = gpudb::GpuStackAllocator::getInstance().alloc<uint2>(size);
        work[1].nodeId = gpudb::GpuStackAllocator::getInstance().alloc<int>(size);
        work[1].range = gpudb::GpuStackAllocator::getInstance().alloc<uint2>(size);

        uint* counter = gpudb::gpuAllocator::getInstance().alloc<uint>(400);
        if (work[0].nodeId == nullptr || work[0].range == nullptr
            || work[1].nodeId == nullptr || work[1].range == nullptr
            || counter == nullptr)
        {
            break;
        }

        int switcher = 0;
        cudaMemset(counter, 0, sizeof(uint) * 400);
        initQueue(work[switcher], size);
        uint queueSize = 1;
        dim3 block = dim3(BLOCK_SIZE);
        dim3 grid;
        while(queueSize > 0) {
            grid = gridConfigure(queueSize, block);
            split<<<grid, block>>>(hlbvh, keys, queueSize, work[switcher], work[1 - switcher], counter);
            cudaMemcpy(&queueSize, counter, sizeof(uint), cudaMemcpyDeviceToHost);
            hlbvh.numNodes += queueSize;
            switcher = 1 - switcher;
            counter++;
        }
        gpudb::GpuStackAllocator::getInstance().popPosition();
        StackAllocator::getInstance().popPosition();
        return true;
    } while(0);

    gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enough memory");
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();
    return false;
}
/// } Build Tree topology
/////////////////////////////////////////////
void test(gpudb::MortonCode *keys, uint32_t bitshift, uint2 range, uint2 &rangeLeft, uint2 &rangeRight) {
    rangeRight = rangeLeft = range;

    gpudb::MortonCode keyA = keys[getLeftBound(range)];
    gpudb::MortonCode keyB = keys[getRightBound(range) - 1];
    uint64_t ha = 64;

    uint64_t hb = 64;
    uint64_t mask;
    bool high = false;
    if (bitshift > 64) {
        if (keyA.high != keyB.high) {
            mask = (1ULL << (bitshift - 64)) - 1;
            ha = keyA.high & mask;
            hb = keyB.high & mask;
            ha = clzllHost(ha ^ hb);
            high = true;
        } else {
            ha = clzllHost(keyA.low ^ keyB.low);
        }
    } else {
        mask = (1ULL << (bitshift)) - 1;
        ha = keyA.low & mask;
        hb = keyB.low & mask;
        ha = clzllHost(ha ^ hb);
    }
/*
     bool high = false;
     if (keyA.high != keyB.high) {
         ha = clzllHost(keyA.high ^ keyB.high);
         high = true;
     } else {
         ha = clzllHost(keyA.low ^ keyB.low);
     }*/

    if (ha == 64) {
        uint mid = getLeftBound(range) + (getRightBound(range) - getLeftBound(range)) / 2;
        getRightBound(rangeLeft) = getLeftBound(rangeRight) = mid;
    } else {
        mask = 1ULL << (64 - ha - 1);
        uint left, right;
        left = getLeftBound(range);
        right = getRightBound(range);
        while (left < right) {
            uint mid = left + (right - left) / 2;
            bool test;
            if (high) {
                test = (keys[mid].high & mask) > 0;
            } else {
                test = (keys[mid].low & mask) > 0;
            }
            /* key[mid] > key[left] */
            if (test) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        getRightBound(rangeLeft) = left;
        getLeftBound(rangeRight) = left;
    }
    printf("%s \n", keys[rangeLeft.x].toString().c_str());
    printf("%s \n", keys[rangeLeft.y - 1].toString().c_str());
    printf("%s \n", keys[rangeRight.x].toString().c_str());
    printf("%s \n", keys[rangeRight.y - 1].toString().c_str());
}

bool gpudb::HLBVH::build(AABB *aabb, uint32_t size) {
    if (size == 0) { return false; }
    gpudb::GpuStackAllocator &gpuStackAlloc = gpudb::GpuStackAllocator::getInstance();
    if (alloc(size) == false) {
        return false;
    }

    do {
        AABB globalAABB;
        if (!computeGlobalAABB(aabb, size, globalAABB)) { break; }

        int *values = gpuStackAlloc.alloc<int>(size);
        MortonCode *keys = gpuStackAlloc.alloc<MortonCode>(size);
        if (keys == nullptr || values == nullptr) { break; }
        computeMortonCodesAndReference(keys, values, aabb, globalAABB, size);
        MortonCode *cpuKeys = StackAllocator::getInstance().alloc<MortonCode>(size);
        int *cpuValues = StackAllocator::getInstance().alloc<int>(size);
        MortonCode *cpuKeys2 = StackAllocator::getInstance().alloc<MortonCode>(size);
        int *cpuValues2 = StackAllocator::getInstance().alloc<int>(size);

        cudaMemcpy(cpuKeys2, keys, sizeof(MortonCode) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuValues2, values, sizeof(int) * size, cudaMemcpyDeviceToHost);

        cudaMemcpy(cpuKeys, keys, sizeof(MortonCode) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuValues, values, sizeof(int) * size, cudaMemcpyDeviceToHost);
        //for (int i = 0; i < size; i++) {
        //    printf("{ (%s), %d }\n", cpuKeys[i].toString().c_str(), cpuValues[i]);
       // }

        if (!sortMortonCodes(keys, values, size)) { break; }
        if (!buildTreeStructure(*this, keys, values, size)) { break; }


        cudaMemcpy(cpuKeys, keys, sizeof(MortonCode) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuValues, values, sizeof(int) * size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++) {
            //printf("{ (%s), %d }\n", cpuKeys[i].toString().c_str(), cpuValues[i]);
            cpuKeys2[i].bits = cpuValues2[i];
        }
        //printf("\n\n\n");
        std::sort(cpuKeys2, cpuKeys2 + size);

        /*for (int i = 0; i < size; i++) {
            if (cpuValues[i] != cpuKeys2[i].bits) {
                printf("{ %d }\n", cpuValues[i] == cpuKeys2[i].bits);
            }
        }*/
        /*uint2 range;
        range.x = 0;
        range.y = size;
        uint2 rangeL;
        uint2 rangeR;
        test(cpuKeys, 96, range, rangeL, rangeR);
        printf("{%d %d} {%d %d}\n", rangeL.x, rangeL.y, rangeR.x, rangeR.y);
        test(cpuKeys, 95, rangeR, rangeL, rangeR);
        printf("{%d %d} {%d %d}\n", rangeL.x, rangeL.y, rangeR.x, rangeR.y);
        test(cpuKeys, 94, rangeR, rangeL, rangeR);
        printf("{%d %d} {%d %d}\n", rangeL.x, rangeL.y, rangeR.x, rangeR.y);
        test(cpuKeys, 93, rangeR, rangeL, rangeR);
        printf("{%d %d} {%d %d}\n", rangeL.x, rangeL.y, rangeR.x, rangeR.y);
        test(cpuKeys, 93, rangeR, rangeL, rangeR);
        printf("{%d %d} {%d %d}\n", rangeL.x, rangeL.y, rangeR.x, rangeR.y);
        test(cpuKeys, 93, rangeR, rangeL, rangeR);
        printf("{%d %d} {%d %d}\n", rangeL.x, rangeL.y, rangeR.x, rangeR.y);*/
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "end hlbvh build");
        gpuStackAlloc.popPosition();
        return true;
    } while(0);
    gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enough memory");
    gpuStackAlloc.popPosition();
    return false;
}
