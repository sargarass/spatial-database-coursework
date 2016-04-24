#include "gpustackallocator.h"

gpudb::GpuStackAllocator::GpuStackAllocator() {
    this->m_memory = 0;
    this->m_size = 0;
    this->m_top = 0;
}

gpudb::GpuStackAllocator::~GpuStackAllocator(){
     gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "delete GpuStackAllocator");
     deinit();
}

void gpudb::GpuStackAllocator::resize(uint64_t size) {
    if (m_memory != 0) {
        gpudb::gpuAllocator::getInstance().free((void *)m_memory);
        m_memory = 0;
    }
    m_memory = reinterpret_cast<uintptr_t>(gpudb::gpuAllocator::getInstance().alloc<char>(size));
    m_top = m_memory;
    m_size = size;
    m_alloced.clear();
}

void gpudb::GpuStackAllocator::deinit() {
    if (m_memory) {
        gpudb::gpuAllocator::getInstance().free((void *)m_memory);
        m_alloced.clear();
        m_memory = 0;
    }
}

void gpudb::GpuStackAllocator::pushPosition() {
    m_alloced.push_back(GpuStackBoarder);
}

bool gpudb::GpuStackAllocator::popPosition() {
    if (m_alloced.empty()) {
        return false;
    }

    while (!m_alloced.empty() && m_alloced.back() != GpuStackBoarder) {
        m_top -= m_alloced.back();
        m_alloced.pop_back();
    }

    if (!m_alloced.empty()) {
        m_alloced.pop_back();
    }
    return true;
}

void gpudb::GpuStackAllocator::clear() {
    m_top = m_memory;
    m_alloced.clear();
}
