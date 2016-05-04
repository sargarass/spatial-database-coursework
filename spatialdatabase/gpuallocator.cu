#include "gpuallocator.h"

gpudb::gpuAllocator &gpudb::gpuAllocator::getInstance() {
    static gpuAllocator *allocator = new gpuAllocator();
    static bool init = false;
    if (init == false) {
        init = true;
        SingletonFactory::getInstance().registration<gpuAllocator>(allocator);
        dynamic_cast<Singleton*>(allocator)->dependOn(Log::getInstance());
    }
    return *allocator;
}

bool gpudb::gpuAllocator::free(void *ptr) {
    if (ptr == nullptr) {
        return false;
    }

    if (!memoryPtrs.erase(reinterpret_cast<uintptr_t>(ptr))) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "memory was not alloced");
        return false;
    }
    cudaFree(ptr);
    return true;
}

void gpudb::gpuAllocator::freeAll() {
    gLogWrite(LOG_MESSAGE_TYPE::INFO, "freeing all memory");
    for (auto& ptr : memoryPtrs) {
        gLogWrite(LOG_MESSAGE_TYPE::INFO, "%p", ptr);
        cudaFree((void*)ptr);
    }
    memoryPtrs.clear();
    cudaDeviceReset();
}
