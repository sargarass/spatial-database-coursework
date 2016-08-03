#pragma once
#include "types.h"
#include "utils.h"
#include "gpuallocator.h"
#include "stackallocator.h"
#include "gpustackallocator.h"
#include "aabb.h"
#include "gpustack.h"
#include "timer.h"
#define LEAF 0xFFFFFFFF
#define getLeftBound(p) p.x
#define getRightBound(p) p.y
#define getRangeSize(p) (p.y - p.x)

namespace gpudb {
    class HLBVH {
    public:
        Result<void, Error<std::string>> build(AABB *aabb, uint32_t size);
        bool search(AABB aabb);
        bool isBuilded() { return builded; }
        HLBVH();
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
        uint8_t *memory;
        uint64_t memorySize;
        bool builded;
    };
}
