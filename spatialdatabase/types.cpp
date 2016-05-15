#include "types.h"
#include "temptable.h"

std::string typeToString(Type t) {
    switch (t) {
        case Type::DATE_TYPE:
            return "DATE_TYPE";
        case Type::STRING:
            return "STRING";
        case Type::INT:
            return "INT";
        case Type::REAL:
            return "REAL";
        case Type::SET:
            return "SET";
        default:
            return "UNKNOWN";
    }
}

std::string typeToString(SpatialType t) {
    switch (t) {
        case SpatialType::POINT:
            return "POINT";
        case SpatialType::POLYGON:
            return "POLYGON";
        case SpatialType::LINE:
            return "LINE";
        default:
            return "UNKNOWN";
    }
}



