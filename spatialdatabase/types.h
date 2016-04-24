#pragma once
#include <string>
#include <set>
#include <map>
#include <cuda.h>
#include <cstring>
#include <inttypes.h>
#include <cuda_runtime.h>
#define NAME_MAX_LEN 255
#define BLOCK_SIZE 512

enum Type {
    STRING, //char 255
    INT, // int64_t
    REAL, // double
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
