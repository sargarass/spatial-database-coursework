#pragma once
#include <string>
#include <set>
#include <map>
#include <cuda.h>
#include <cstring>
#include "date.h"
#include <inttypes.h>
#include <cuda_runtime.h>
#include <memory>
#include "error.h"
#include "result.h"
#include "make_unique.h"
#include <openssl/sha.h>
#define NAME_MAX_LEN 256
#define BLOCK_SIZE 512
#define FUNC_PREFIX __host__ __device__

#define MYERR_STRING(x) Err(Error<std::string>(ERROR_ARGS(x)))

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    auto buf = std::make_unique<char[]>(size);
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

static FUNC_PREFIX
float clamp(float in, float min, float max) {
    float res = (in > max)? max : in;
    res = (res < min)? min : res;
    return res;
}

inline void UNUSED_PARAM_HANDLER(){}
template <typename Head, typename ...Tail>
inline void UNUSED_PARAM_HANDLER(Head car, Tail ...cdr) { ((void) car); UNUSED_PARAM_HANDLER(cdr...);}

enum class Type {
    STRING, //char 256
    INT, // int64_t
    REAL, // double
    DATE_TYPE,
    SET,
    UNKNOWN
};

enum class SpatialType {
    POLYGON,
    LINE,
    POINT,
    UNKNOWN
};

enum class TemporalType {
    TRANSACTION_TIME,
    VALID_TIME,
    BITEMPORAL_TIME,
    UNKNOWN
};


enum class AABBRelation {
    OVERLAP,
    BINSIDEA,
    DISJOINT,
    EQUAL
};

std::string typeToString(Type t);
std::string typeToString(SpatialType t);
FUNC_PREFIX
uint64_t typeSize(Type t);
