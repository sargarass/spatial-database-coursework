#pragma once
#include <cinttypes>
#include <string>
#include <set>
#include <map>
#include <cuda.h>
#include <cstring>
#include <cuda_runtime.h>
#define NAME_MAX_LEN 255

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
