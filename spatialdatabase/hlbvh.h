#pragma once
#include "types.h"
#include "gpuallocator.h"
#include "stackallocator.h"
#include "gpustackallocator.h"
#include "aabb.h"

namespace gpudb {
    class HLBVH {
    public:
        bool build(AABB *aabb, uint32_t size);
        bool search(AABB aabb);
    public:
        bool alloc(uint32_t size);
        void free();

        uint32_t numNodes;
        uint32_t numReferences;
        uint32_t numBVHLevels;

        float4 *aabbMin;
        float4 *aabbMax;

        uint *references;
        int *parents;
        uint2 *ranges;
        int *links;
    };
}
