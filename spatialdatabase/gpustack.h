#pragma once
#include "types.h"

template<typename T> FUNC_PREFIX
void myswap(T &a, T&b) {
    T tmp;
    tmp = a;
    a = b;
    b = tmp;
}

template <typename K, typename V, typename I>
struct Heap{
    uint cap;
    uint count;
    K *keys;
    V *values;
    I **indexes;

    FUNC_PREFIX
    Heap(K *memKeys, V *memValues, I **memIndexes, uint size) {
        cap = size;
        count = 0;
        keys = memKeys;
        values = memValues;
        indexes = memIndexes;
    }

    FUNC_PREFIX
    K maxKey() {
        return keys[0];
    }

    FUNC_PREFIX
    V maxValue() {
        return values[0];
    }

    FUNC_PREFIX
    bool empty() {
        return count == 0;
    }

    FUNC_PREFIX
    bool insert(K key, V val, I *index) {
        if (cap == count) {
            return false;
        }
        uint i = count;
        ++count;
        keys[i] = key;
        values[i] = val;
        indexes[i] = index;
        while (i > 0 && keys[(i - 1) / 2] < keys[i])  {
            myswap(keys[(i - 1) / 2], keys[i]);
            myswap(values[(i - 1) / 2], values[i]);
            myswap(indexes[(i - 1) / 2], indexes[i]);
            if (indexes[i] != nullptr) { *indexes[i] = i; }
            i = (i - 1) / 2;
        }
        if (indexes[i] != nullptr) { *indexes[i] = i; }
        return true;
    }

    FUNC_PREFIX
    void heapify(int i, int size) {
        while (1) {
            int left = 2 * i + 1;
            int right = left + 1;
            int j = i;
            if (left < size && keys[i] < keys[left]) {
                i = left;
            }
            if (right < size && keys[i] < keys[right]) {
                i = right;
            }
            if (i == j) {
                break;
            }

            myswap(values[i], values[j]);
            myswap(keys[i], keys[j]);
            myswap(indexes[i], indexes[j]);

            if (indexes[i] != nullptr) { *indexes[i] = i; }
            if (indexes[j] != nullptr) { *indexes[j] = j; }
        }
    }

    FUNC_PREFIX
    bool extractMax() {
        if (empty()) {return false;}
        count--;
        if (count > 0) {
            keys[0] = keys[count];
            values[0] = values[count];
            indexes[0] = indexes[count];
            if (indexes[0] != nullptr) { *indexes[0] = 0; }
            heapify(0, count);
        }
        return true;
    }

    FUNC_PREFIX
    void increaseKey(int i, float k) {
        keys[i] = k;
        while (i > 0 && keys[(i - 1) / 2] < k) {
            myswap(keys[(i - 1) / 2], keys[i]);
            myswap(values[(i - 1) / 2], values[i]);
            myswap(indexes[(i - 1) / 2], indexes[i]);
            if (indexes[i] != nullptr) { *indexes[i] = i; }
            i = (i - 1) / 2;
        }
        if (indexes[i] != nullptr) { *indexes[i] = i; }
    }

    FUNC_PREFIX
    void deleteKey(int i) {
        while (i > 0) {
            myswap(keys[(i - 1) / 2], keys[i]);
            myswap(values[(i - 1) / 2], values[i]);
            myswap(indexes[(i - 1) / 2], indexes[i]);
            if (indexes[i] != nullptr) { *indexes[i] = i; }
            i = (i - 1) / 2;
        }
        if (indexes[i] != nullptr) { *indexes[i] = i; }
        extractMax();
    }
};

template<typename T>
class GpuStack {
public:
    __host__ __device__
    GpuStack(T *memory, uint size);

    __host__ __device__
    T top();
    T &topRef();
    __host__ __device__
    void pop();

    __host__ __device__
    bool push(T elem);

    __host__ __device__
    bool empty();
public:
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
T &GpuStack<T>::topRef() {
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
