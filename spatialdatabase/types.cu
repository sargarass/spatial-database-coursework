#include "types.h"
#include "temptable.h"

FUNC_PREFIX
uint64_t typeSize(Type t) {
    switch (t) {
        case Type::SET:
            return sizeof(gpudb::GpuSet);
        case Type::DATE_TYPE:
            return sizeof(Date);
        case Type::STRING:
            return 256;
        case Type::INT:
            return 8;
        case Type::REAL:
            return 8;
        default:
            return 0;
    }
    return 0;
}
