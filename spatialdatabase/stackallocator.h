#pragma once
#include "types.h"
#include <deque>
#include "singletonmanager.h"

class StackAllocator : public Singleton {
public:
    static const uint64_t CpuAlignSize = 256;

    StackAllocator(){
        m_size = 0;
        m_memory = 0;
        m_top = 0;
    }

    static StackAllocator &getInstance();

    virtual ~StackAllocator();

    template<typename T>
    T* alloc(uint64_t count = 1) {
        return reinterpret_cast<T*>( helpAlloc(sizeof(T) * count) );
    }

    void clear() {
        m_top = m_memory;
        m_position.clear();
    }

    bool free(void *pointer);

    void resize(uint64_t size);
    uint64_t availableMemory() {
        return m_memory + m_size - m_top;
    }

    void pushPosition();
    bool popPosition();
private:
    void *helpAlloc(uint64_t size);

    uint64_t m_size;
    uintptr_t m_memory;
    uintptr_t m_top;
    std::deque<uintptr_t> m_position;
};

namespace StackAllocatorAdditions {
    template<typename T>
    void free(T *handle) {
        StackAllocator::getInstance().free(handle);
    }

    template<typename T>
    T* alloc(uint64_t count) {
        return StackAllocator::getInstance().alloc<T>(count);
    }

    template<typename T>
    std::unique_ptr<T, void(*)(T *)> allocUnique(uint64_t count = 1) {
        return std::move(std::unique_ptr<T, void(*)(T *)>(alloc<T>(count), free<T>));
    }

    template<typename T>
    std::shared_ptr<T> allocShared(uint64_t count = 1) {
        return std::move(std::shared_ptr<T>(alloc<T>(count), free<T>));
    }
}
