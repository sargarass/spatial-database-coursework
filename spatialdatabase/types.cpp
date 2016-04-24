#include "types.h"

std::string typeToString(Type t) {
    switch (t) {
        case Type::STRING:
            return "STRING";
        case Type::INT:
            return "INT";
        case Type::REAL:
            return "REAL";
        default:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

uint64_t typeSize(Type t) {
    switch (t) {
        case Type::STRING:
            return 255;
        case Type::INT:
            return 8;
        case Type::REAL:
            return 8;
        default:
            return 0;
    }
    return 0;
}
