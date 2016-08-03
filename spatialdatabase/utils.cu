#include "utils.h"

__device__
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


dim3 gridConfigure(uint64_t problemSize, dim3 block) {
    UNUSED_PARAM_HANDLER(block);
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
