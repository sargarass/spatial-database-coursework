#include "gpuallocator.h"

gpudb::gpuAllocator &gpudb::gpuAllocator::getInstance() {
    static gpuAllocator &allocator = SingletonFactory::getInstance().create<gpuAllocator>();
    static bool init = false;
    if (!init) {
        dynamic_cast<Singleton*>(&allocator)->dependOn(Log::getInstance());
        init = true;
    }
    return allocator;
}

bool gpudb::gpuAllocator::free(void *ptr) {
    if (!memoryPtrs.erase(reinterpret_cast<uintptr_t>(ptr))) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "gpuAllocator", "free",
                                 "memory was not alloced");
        return false;
    }
    cudaFree(ptr);
    return true;
}

void gpudb::gpuAllocator::freeAll() {
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO, "gpuAllocator", "freeAll",
                             "freeing all memory");
    for (auto& ptr : memoryPtrs) {
        cudaFree((void*)ptr);
    }
    memoryPtrs.clear();
}
