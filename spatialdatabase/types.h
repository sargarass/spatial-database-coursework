#pragma once
#include <string>
#include <set>
#include <map>
#include <cuda.h>
#include <cstring>
#include "date.h"
#include <inttypes.h>
#include <cuda_runtime.h>
#define NAME_MAX_LEN 256
#define BLOCK_SIZE 512
#define FUNC_PREFIX __host__ __device__

static FUNC_PREFIX
float clamp(float in, float min, float max) {
    float res = (in > max)? max : in;
    res = (res < min)? min : res;
    return res;
}

inline void UNUSED_PARAM_HANDLER(){}
template <typename Head, typename ...Tail>
inline void UNUSED_PARAM_HANDLER(Head car, Tail ...cdr) { ((void) car); UNUSED_PARAM_HANDLER(cdr...);}

enum Type {
    STRING, //char 255
    INT, // int64_t
    REAL, // double
    DATE_TYPE,
    SET,
    UNKNOWN
};

enum SpatialType {
    POLYGON,
    LINE,
    POINT
};

enum TemporalType {
    TRANSACTION_TIME,
    VALID_TIME,
    BITEMPORAL_TIME,
};

std::string typeToString(Type t);
uint64_t typeSize(Type t);
