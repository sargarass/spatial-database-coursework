#pragma once
#include "types.h"
#include "singletonmanager.h"
#include "log.h"

namespace gpudb {
    class gpuAllocator : public Singleton {
    public:
        static gpuAllocator &getInstance();
        template <typename T>
        T *alloc(uint64_t count) {
            void* memory;
            cudaMalloc(&memory, count * sizeof(T));
            memoryPtrs.insert(reinterpret_cast<uintptr_t>(memory));
            return reinterpret_cast<T*>(memory);
        }

        bool free(void *ptr);
        void freeAll();
    protected:
        virtual ~gpuAllocator(){ gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "delete gpuAllocator"); freeAll(); }
        std::set<uintptr_t> memoryPtrs;
    };
}

