#include "stackallocator.h"
#include "log.h"

StackAllocator::~StackAllocator(){
    gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "stackallocator delete");
    if (m_memory) {
        delete [] reinterpret_cast<char*>( m_memory );
        m_memory = 0;
    }
}

StackAllocator &StackAllocator::getInstance() {
    static StackAllocator* ptr = new StackAllocator();
    static bool init = false;
    if (init == false) {
        init = true;
        if (SingletonFactory::getInstance().registration<StackAllocator>(ptr) == false) {
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "stackallocator was not registrated");
            exit(-1);
        }
        dynamic_cast<Singleton *>(ptr)->dependOn(Log::getInstance());
    }
    return *ptr;
}


void StackAllocator::resize(uint64_t size) {
    if (m_memory != 0) {
        delete [] reinterpret_cast<char*>( m_memory );
        m_memory = 0;
    }

    m_memory = reinterpret_cast<uintptr_t>( new char[size] );
    m_top = m_memory;
    m_size = size;
}

void* StackAllocator::helpAlloc(uint64_t objSize) {
    uint64_t offset = CpuAlignSize - 1 + sizeof( uint64_t );
    if (m_top + offset + objSize <= m_memory + m_size) {
        uintptr_t aligned = (m_top + offset) & ~(CpuAlignSize - 1); // выравняли указатель
        *(reinterpret_cast<uint64_t *>( aligned - sizeof( uint64_t ) )) = objSize + offset; // записали размер
        m_top += offset + objSize; // сдвинули указатель на top стека
        return reinterpret_cast<void*>(aligned);
    }
    return nullptr;
}

bool StackAllocator::free(void *pointer) {
    uint64_t size = *reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>( pointer ) - sizeof( uint64_t ));
    uint64_t offset = CpuAlignSize - 1 + sizeof( uint64_t );
    uintptr_t aligned = (m_top - size + offset) & ~(CpuAlignSize - 1);
    if (aligned == reinterpret_cast<uintptr_t>( pointer )) {
        m_top -= size;
        return true;
    }
    return false;
}

void StackAllocator::pushPosition() {
    m_position.push_front(m_top);
}

bool StackAllocator::popPosition() {
    if (!m_position.empty()) {
        m_top = m_position.front();
        m_position.pop_front();
        return true;
    }
    return false;
}
