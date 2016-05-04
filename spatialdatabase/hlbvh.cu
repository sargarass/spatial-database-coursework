#include "hlbvh.h"
#include "utils.h"
#include <cub/cub/cub.cuh>



gpudb::HLBVH::HLBVH() {
    this->memorySize = 0;
    this->numBVHLevels = 0;
    this->numNodes = 0;
    this->numReferences = 0;
    this->aabbMax = nullptr;
    this->aabbMin = nullptr;
    this->links = nullptr;
    this->memory = nullptr;
    this->parents = nullptr;
    this->references = nullptr;
    this->ranges = nullptr;
    builded = false;
}

bool gpudb::HLBVH::alloc(uint32_t size) {
    memorySize = 2 * size * (2 * sizeof(int) + sizeof(uint2) + 2 * sizeof(float4)) + 1 * size * ( sizeof(uint) );
    memory = gpuAllocator::getInstance().alloc<uint8_t>(memorySize);
    if (memory == nullptr) {
        memorySize = 0;
        return false;
    }
    links =      reinterpret_cast<int*>    (memory);
    parents =    reinterpret_cast<int*>    (memory + 2 * size * sizeof(int));
    ranges =     reinterpret_cast<uint2*>  (memory + 2 * size * (2 * sizeof(int)));
    aabbMin =    reinterpret_cast<float4*> (memory + 2 * size * (2 * sizeof(int) + sizeof(uint2)));
    aabbMax =    reinterpret_cast<float4*> (memory + 2 * size * (2 * sizeof(int) + sizeof(uint2) + 1 * sizeof(float4)));
    references = reinterpret_cast<uint*>   (memory + 2 * size * (2 * sizeof(int) + sizeof(uint2) + 2 * sizeof(float4)));

    numBVHLevels = 0;
    numNodes = 0;
    numReferences = size;
    return true;
}

void gpudb::HLBVH::free() {
    if (memory) {
        gpuAllocator::getInstance().free(memory);
    }
    builded = false;
    memory = nullptr;
    links = nullptr;
    parents = nullptr;
    ranges = nullptr;
    aabbMin = nullptr;
    aabbMax = nullptr;
    references = nullptr;
    memorySize = 0;
    numBVHLevels = 0;
    numNodes = 0;
    numReferences = 0;
}

///////////////////////////////////////
/// Morton code part {
static __global__
void computeMortonCodesAndReferenceKernel(gpudb::MortonCode *keys, uint *values, gpudb::AABB *aabb, gpudb::AABB globalAABB, uint32_t size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    keys[thread] = aabb[thread].getMortonCode(globalAABB);
    values[thread] = thread;
}

void computeMortonCodesAndReference(gpudb::MortonCode *keys, uint *values, gpudb::AABB *aabb, gpudb::AABB globalAABB, uint size) {
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = gridConfigure(size, block);
    computeMortonCodesAndReferenceKernel<<<grid,block>>>(keys, values, aabb, globalAABB, size);
}

__global__
void initKeys(uint64_t *keys, uint *values, gpudb::MortonCode *codes, uint size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }
    keys[thread] = (codes[thread].high << 32ULL) | ((codes[thread].low >> 32ULL) & 0x00000000FFFFFFFFULL);
    values[thread] = thread;
}

__global__
void computeDiff(uint64_t *keys, uint *array, uint size) {
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

__global__
void writeNewKeys(uint64_t *keys, uint *values, gpudb::MortonCode *codes, uint *prefixSum, uint size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }
    keys[thread] = ((uint64_t)prefixSum[thread]) << 32ULL | ((codes[values[thread]].low & 0x00000000FFFFFFFFULL));
}

__global__
void copyKeys(gpudb::MortonCode *new_keys, gpudb::MortonCode *old_keys, uint *new_values, uint *old_values, uint size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    new_keys[thread].bits = old_keys[old_values[thread]].bits;
    new_keys[thread].low = old_keys[old_values[thread]].low;
    new_keys[thread].high = old_keys[old_values[thread]].high;
    new_values[thread] = old_values[thread];
}

bool sortMortonCodes(gpudb::MortonCode *keys, uint *values, uint size) {
    // 96 bit
    Timer t;
    t.start();
    uint64_t *cub_keys[2];
    uint *cub_values[2];
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = gridConfigure(size, block);
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    int switcher = 1;
    for (int i = 0; i < 2; i++) {
        cub_keys[i] = gpudb::GpuStackAllocator::getInstance().alloc<uint64_t>(size);
        cub_values[i]  = gpudb::GpuStackAllocator::getInstance().alloc<uint>(size);
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
    // Чтобы сохранить порядок, посчитаем префиксную сумму
    // и в качестве ключа будем использовать
    // prefixsum << 32 | keypart
    // отсортированное в cub_keys/cub_values[switcher], эти два массива теперь не нужны
    uint *arrayPrefixSum = reinterpret_cast<uint*>(cub_keys[1 - switcher]);
    uint *array = reinterpret_cast<uint*>(cub_values[1 - switcher]);
    computeDiff<<<grid, block>>>(cub_keys[switcher], array, size);
    cub::DeviceScan::ExclusiveSum(cub_tmp_memory, cub_tmp_memory_size, array, arrayPrefixSum, size);
    writeNewKeys<<<grid, block>>>(cub_keys[switcher], cub_values[switcher], keys, arrayPrefixSum, size);
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
    gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "radixsort %d ms", t.elapsedMillisecondsU64());
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
        Timer t;
        t.start();
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
        cudaDeviceSynchronize();
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "globalAABB %d ms", t.elapsedMillisecondsU64());
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



#define clzll(x) __clzll((x))
#define clzllHost(x) ((x) == 0)? 64 : __builtin_clzll((x))
#define maxComp(x , y) ((x) > (y))? (x) : (y)
__global__
void split(gpudb::HLBVH hlbvh, gpudb::MortonCode *keys, uint queueSize, uint64_t bitshift, uint nodeSize, WorkQueue qIn, WorkQueue qOut, uint *counter) {
    uint thread = getGlobalIdx3DZXY();
    bool isWorking = true;
    if (thread >= queueSize) {
        isWorking = false;
    }

    int warpId = threadIdx.x / 32;
    __shared__  cub::WarpReduce<int>::TempStorage warpReduceTemp[BLOCK_SIZE / 32];
    __shared__  cub::WarpScan<int>::TempStorage warpScanTemp[BLOCK_SIZE / 32];
    uint2 rangeLeft, rangeRight, range;
    bool isLeaf = true;
    int node = 0;
    if (isWorking) {
        range = qIn.range[thread];
        rangeRight = range;
        rangeLeft = range;
        node = qIn.nodeId[thread];
        if (getRangeSize(range) > nodeSize) {
            isLeaf = false;
            gpudb::MortonCode keyA = keys[getLeftBound(range)];
            gpudb::MortonCode keyB = keys[getRightBound(range) - 1];
            uint64_t ha = 64;
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
                uint64_t mask = 1ULL << (64 - ha - 1);
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
    }

    int thread_data = 2;
    if (isLeaf) {
        thread_data = 0;
    }

    int sum = cub::WarpReduce<int>(warpReduceTemp[warpId]).Sum(thread_data);
    uint offset = 0;

    if ((threadIdx.x & 31) == 0 && isWorking) {
        offset = atomicAdd(counter, sum);
        thread_data += offset;
    }

    cub::WarpScan<int>(warpScanTemp[warpId]).ExclusiveSum(thread_data, thread_data);
    offset += thread_data;

    if (isWorking) {
        hlbvh.ranges[node] = range;
        if (isLeaf) {
            hlbvh.links[node] = LEAF; // лист
        } else {
            uint left = hlbvh.numNodes + offset;
            hlbvh.links[node] = left;

            qOut.nodeId[offset] = left;
            qOut.nodeId[offset + 1] = left + 1;
            qOut.range[offset] = rangeLeft;
            qOut.range[offset + 1] = rangeRight;
        }
    }
}

bool buildTreeStructure(gpudb::HLBVH &hlbvh, uint nodeSize, gpudb::MortonCode *keys, uint *offset, uint32_t size) {
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    StackAllocator::getInstance().pushPosition();

    do {
        Timer t;
        t.start();
        WorkQueue work[2];
        work[0].nodeId = gpudb::GpuStackAllocator::getInstance().alloc<int>(size);
        work[0].range = gpudb::GpuStackAllocator::getInstance().alloc<uint2>(size);
        work[1].nodeId = gpudb::GpuStackAllocator::getInstance().alloc<int>(size);
        work[1].range = gpudb::GpuStackAllocator::getInstance().alloc<uint2>(size);
        offset[hlbvh.numBVHLevels++] = 0;
        uint* counter = gpudb::GpuStackAllocator::getInstance().alloc<uint>(400);
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
        uint64_t bitshift = 95;

        while(queueSize > 0) {
            grid = gridConfigure(queueSize, block);
            split<<<grid, block>>>(hlbvh, keys, queueSize, bitshift, nodeSize, work[switcher], work[1 - switcher], counter);
            cudaMemcpy(&queueSize, counter, sizeof(uint), cudaMemcpyDeviceToHost);
            counter++;
            bitshift--;
            hlbvh.numNodes += queueSize;
            switcher = 1 - switcher;
            if (queueSize > 0) {
                offset[hlbvh.numBVHLevels++] = hlbvh.numNodes;
            }
        }
        gpudb::GpuStackAllocator::getInstance().popPosition();
        StackAllocator::getInstance().popPosition();
        cudaDeviceSynchronize();
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "buildTreeStructure %d ms", t.elapsedMillisecondsU64());
        return true;
    } while(0);

    gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enough memory");
    gpudb::GpuStackAllocator::getInstance().popPosition();
    StackAllocator::getInstance().popPosition();
    return false;
}
/// } Build Tree topology
/////////////////////////////////////////////
/// Refit boxes {
__host__ __device__
void readBoxFromAABB(gpudb::AABB *aabb, uint id, float4 &min, float4 &max) {
    gpudb::AABB read = aabb[id];
    min.x = read.x.x;
    min.y = read.y.x;
    min.z = read.z.x;
    min.w = read.w.x;

    max.x = read.x.y;
    max.y = read.y.y;
    max.z = read.z.y;
    max.w = read.w.y;
}

__global__
void refitBoxesKernel(gpudb::HLBVH hlbvh, gpudb::AABB *aabb, uint2 range, bool isRoot) {
    uint node = getGlobalIdx3DZXY();
    if (node >= getRangeSize(range)) {
        return;
    }

    node += getLeftBound(range);
    int link = hlbvh.links[node];
    if (link == LEAF) {
        uint2 nodeRange = hlbvh.ranges[node];
        float4 aabbMin;
        float4 aabbMax;
        readBoxFromAABB(aabb, hlbvh.references[getLeftBound(nodeRange)], aabbMin, aabbMax);
        for (uint j = getLeftBound(nodeRange) + 1; j < getRightBound(nodeRange); j++) {
            float4 readAABBMin;
            float4 readAABBMax;
            readBoxFromAABB(aabb, hlbvh.references[j], readAABBMin, readAABBMax);
            aabbMin = fmin(aabbMin, readAABBMin);
            aabbMax = fmax(aabbMax, readAABBMax);
        }
        hlbvh.aabbMin[node] = aabbMin;
        hlbvh.aabbMax[node] = aabbMax;
    } else {
        float4 boxAaabbMin = hlbvh.aabbMin[link + 0];
        float4 boxAaabbMax = hlbvh.aabbMax[link + 0];

        float4 boxBaabbMin = hlbvh.aabbMin[link + 1];
        float4 boxBaabbMax = hlbvh.aabbMax[link + 1];

        hlbvh.parents[link + 0] = node;
        hlbvh.parents[link + 1] = node;

        hlbvh.aabbMin[node] = fmin(boxAaabbMin, boxBaabbMin);
        hlbvh.aabbMax[node] = fmax(boxAaabbMax, boxBaabbMax);
    }

    if (isRoot) {
        hlbvh.parents[node] = -1;
    }
}

void refitBoxes(gpudb::HLBVH &hlbvh, gpudb::AABB *aabb, uint *offset) {
    Timer t;
    t.start();
    bool isRoot;
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid;
    for (int i = hlbvh.numBVHLevels - 2; i >= 0; --i)  {
        uint2 range;
        getLeftBound(range) = offset[i];
        getRightBound(range) = offset[i + 1];
        if (i == 0) {
            isRoot = true;
        }
        grid = gridConfigure(getRangeSize(range), block);
        refitBoxesKernel<<<grid, block>>>(hlbvh, aabb, range, isRoot);
    }
    cudaDeviceSynchronize();
    gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "refitBoxes %d ms", t.elapsedMillisecondsU64());
}
/// } Refit boxes
/////////////////////////////////////////////

bool gpudb::HLBVH::build(AABB *aabb, uint32_t size) {
    if (size == 0) { return false; }
    gpudb::GpuStackAllocator &gpuStackAlloc = gpudb::GpuStackAllocator::getInstance();
    if (alloc(size) == false) {
        return false;
    }

    StackAllocator::getInstance().pushPosition();
    gpuStackAlloc.pushPosition();
    do {
        AABB globalAABB;
        if (!computeGlobalAABB(aabb, size, globalAABB)) { break; }
        MortonCode *keys = gpuStackAlloc.alloc<MortonCode>(size);
        uint *offset = StackAllocator::getInstance().alloc<uint>(400);

        if (keys == nullptr || offset == nullptr) { break; }
        computeMortonCodesAndReference(keys, references, aabb, globalAABB, size);

        if (!sortMortonCodes(keys, references, size)) { break; }
        if (!buildTreeStructure(*this, 1, keys, offset, size)) { break; }

        refitBoxes(*this, aabb, offset);
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "end hlbvh build");

        StackAllocator::getInstance().popPosition();
        gpuStackAlloc.popPosition();
        builded = true;
        return true;
    } while(0);
    gLogWrite(LOG_MESSAGE_TYPE::ERROR, "not enough memory");
    StackAllocator::getInstance().popPosition();
    gpuStackAlloc.popPosition();
    return false;
}

__device__
bool boxIntersection(float4 aMin, float4 aMax, float4 bMin, float4 bMax) {
    if (aMin.x < bMin.x) return false;
    if (aMax.x > bMax.x) return false;
    if (aMin.y < bMin.y) return false;
    if (aMax.y > bMax.y) return false;
    if (aMin.z < bMin.z) return false;
    if (aMax.z > bMax.z) return false;
    if (aMin.w < bMin.w) return false;
    if (aMax.w > bMax.w) return false;
    return true;
}

__global__
void searchAABBkernel(gpudb::HLBVH hlbvh, float4 aabbMin, float4 aabbMax, uint size, bool *result, int *stack, int StackSize) {
    int id = getGlobalIdx3DZXY();
    if (id >= size) {
        return;
    }

    GpuStack<int> st(stack, StackSize);
    st.push(0);
    st.push(1);
    while(!st.empty()) {
        int nodeId = st.top();
        st.pop();
        float4 boxAmin = hlbvh.aabbMin[nodeId];
        float4 boxAmax = hlbvh.aabbMax[nodeId];
        if (hlbvh.links[nodeId] == LEAF) {
            if (aabbMin.x == boxAmin.x
                && aabbMin.y == boxAmin.y
                && aabbMin.z == boxAmin.z
                && aabbMin.w == boxAmin.w
                && aabbMax.x == boxAmax.x
                && aabbMax.y == boxAmax.y
                && aabbMax.z == boxAmax.z
                && aabbMax.w == boxAmax.w)
            {
                *result = true;
                return;
            }
        } else {
            if (boxIntersection(aabbMin, aabbMax, boxAmin, boxAmax)) {
                st.push(hlbvh.links[nodeId] + 0);
                st.push(hlbvh.links[nodeId] + 1);
            }
        }
    }
    *result = false;
}
