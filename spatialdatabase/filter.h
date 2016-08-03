#pragma once
#include "constobjects.h"
typedef bool(*Predicate)(gpudb::CRow const &);

#define FILTER_H(name) \
Predicate name()

#define FILTER_CU(name) \
__device__ bool name##_tester(gpudb::CRow const &row); \
__device__ Predicate h_##name##_predicate = name##_tester; \
Predicate name() { \
    static Predicate p;\
    static bool init = false;\
    if (!init) { \
        init = true; \
        cudaMemcpyFromSymbol(&p, h_##name##_predicate, sizeof(Predicate));\
    } \
    return p;\
} \
__device__ bool name##_tester(gpudb::CRow const &row)

FILTER_H(SELECT_ALL_ROWS);
