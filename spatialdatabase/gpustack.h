#pragma once
#include "types.h"

template<typename T>
class GpuStack {
public:
    __host__ __device__
    GpuStack(T *memory, uint size);

    __host__ __device__
    T top();

    __host__ __device__
    void pop();

    __host__ __device__
    bool push(T elem);

    __host__ __device__
    bool empty();
private:
    T *m_gpuMemory;
    uint m_top;
    uint m_size;
};

template<typename T> __host__ __device__
GpuStack<T>::GpuStack(T *memory, uint size) {
    if (memory == nullptr) {
        m_gpuMemory = nullptr;
        m_size = 0;
        m_top = 0;
    } else {
        m_gpuMemory = memory;
        m_size = size;
        m_top = 0;
    }
}

template<typename T> __host__ __device__
bool GpuStack<T>::push(T elem) {
    if (m_top == m_size || m_gpuMemory == nullptr) {
        return false;
    }
    m_gpuMemory[m_top] = elem;
    ++m_top;
    return true;
}

template<typename T> __host__ __device__
T GpuStack<T>::top() {
    return m_gpuMemory[m_top - 1];
}

template<typename T> __host__ __device__
void GpuStack<T>::pop() {
    if (m_top > 0) {
        --m_top;
    }
}

template<typename T> __host__ __device__
bool GpuStack<T>::empty() {
    return m_top == 0;
}
