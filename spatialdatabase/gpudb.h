#pragma once
#include "types.h"
#include "aabb.h"
#include "log.h"
#include "tabledescription.h"
#include "gpuallocator.h"
#include "gpustackallocator.h"
#include "stackallocator.h"
#include <thrust/thrust/device_vector.h>
//Miller cylindrical projection
#include "utils.h"
#include "hlbvh.h"

#define SWITCH_RUN(spatialtype, temporaltype, kernel, grid, block, parameters, ...) \
switch(spatialtype) { \
    case SpatialType::POINT: { \
        switch(temporaltype) { \
            case TemporalType::BITEMPORAL_TIME: \
                kernel<SpatialType::POINT, TemporalType::BITEMPORAL_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
            case TemporalType::VALID_TIME: \
                kernel<SpatialType::POINT, TemporalType::VALID_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
            case TemporalType::TRANSACTION_TIME: \
                kernel<SpatialType::POINT, TemporalType::TRANSACTION_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
        } \
    } \
    break; \
    case SpatialType::POLYGON: { \
        switch(temporaltype) { \
            case TemporalType::BITEMPORAL_TIME: \
                kernel<SpatialType::POLYGON, TemporalType::BITEMPORAL_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
            case TemporalType::VALID_TIME: \
                kernel<SpatialType::POLYGON, TemporalType::VALID_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
            case TemporalType::TRANSACTION_TIME: \
                kernel<SpatialType::POLYGON, TemporalType::TRANSACTION_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
        } \
    }\
    break; \
    case SpatialType::LINE: { \
        switch(temporaltype) { \
            case TemporalType::BITEMPORAL_TIME: \
                kernel<SpatialType::LINE, TemporalType::BITEMPORAL_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
            case TemporalType::VALID_TIME: \
                kernel<SpatialType::LINE, TemporalType::VALID_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
            case TemporalType::TRANSACTION_TIME: \
                kernel<SpatialType::LINE, TemporalType::TRANSACTION_TIME><<<grid, block>>>(parameters, ##__VA_ARGS__); \
                break; \
        } \
    } \
    break; \
}

namespace gpudb {
    struct GpuColumnAttribute {
        char name[NAME_MAX_LEN];
        Type type;

        FUNC_PREFIX

        GpuColumnAttribute() {}
        FUNC_PREFIX

        GpuColumnAttribute(GpuColumnAttribute const &atr) {
            *this = atr;
        }

        FUNC_PREFIX
        void operator=(GpuColumnAttribute const &b) {
            type = b.type;
            memcpy(name, b.name, NAME_MAX_LEN);
            name[NAME_MAX_LEN - 1] = 0;
        }
    };

    struct  __align__(16) GpuLine {
        float2 *points;
        uint size;
        FUNC_PREFIX
        void operator=(GpuLine const &b) {
            points = b.points;
            size = b.size;
        }
    };

    struct  __align__(16) GpuPolygon {
        float2 *points;
        uint size;
        FUNC_PREFIX
        void operator=(GpuPolygon const &b) {
            points = b.points;
            size = b.size;
        }
    };


    struct  __align__(16) GpuPoint {
        float2 p;
        FUNC_PREFIX
        void operator=(GpuPoint const &b) {
            p = b.p;
        }
    };

    struct  __align__(16) SpatialKey {
        char name[NAME_MAX_LEN];
        SpatialType type;
        void *key;

        FUNC_PREFIX
        void operator=(SpatialKey const &b) {
            type = b.type;
            key = b.key;
        }

        FUNC_PREFIX
        void boundingBox(AABB *box);
    };

    struct  __align__(16) TemporalKey {
        char name[NAME_MAX_LEN];
        TemporalType type;
        uint64_t validTimeSCode;
        uint64_t validTimeECode;
        uint64_t transactionTimeCode;

        FUNC_PREFIX
        void boundingBox(AABB *box);
    };

    struct  __align__(16) Value {
        bool isNull;
        void *value;
    };

    struct  __align__(16) GpuRow {
        SpatialKey spatialPart;
        TemporalKey temporalPart;
        Value *value;
        uint64_t valueSize;

        void operator=(GpuRow const &b) {
            spatialPart = b.spatialPart;
            temporalPart = b.temporalPart;
            value = b.value;
            valueSize = b.valueSize;
        }
    };

    struct __align__(16) GpuSet {
        GpuRow **rows;
        GpuColumnAttribute *columns;
        uint rowsSize;
        uint columnsSize;
        TempTable *temptable;
    };

    struct GpuTable {
        GpuTable();
        ~GpuTable();

        char name[NAME_MAX_LEN];
        bool rowReferenses;
        HLBVH bvh;
        thrust::device_vector<GpuColumnAttribute> columns;
        thrust::device_vector<GpuRow*> rows;
        std::vector<uint64_t> rowsSize;
        bool set(TableDescription table);
        bool setName(char *dst, std::string const &src);
        bool insertRow(TableDescription &descriptor, gpudb::GpuRow*  row, uint64_t memsize);
    };
}

template<SpatialType spatialtype, TemporalType temporaltype> __device__
bool testIdenticalRowKeys(gpudb::GpuRow *a, gpudb::GpuRow *b) {
    bool spatialEx = false;
    bool temporalEx = false;

    switch(temporaltype) {
        case TemporalType::BITEMPORAL_TIME:
        {
            if (a->temporalPart.validTimeSCode      == b->temporalPart.validTimeSCode &&
                a->temporalPart.validTimeECode      == b->temporalPart.validTimeECode &&
                a->temporalPart.transactionTimeCode == b->temporalPart.transactionTimeCode) {
                temporalEx = true;
            }
        }
        break;
        case TemporalType::VALID_TIME:
        {
            if (a->temporalPart.validTimeSCode == b->temporalPart.validTimeSCode &&
                a->temporalPart.validTimeECode == b->temporalPart.validTimeECode) {
                temporalEx = true;
            }
        }
        break;
        case TemporalType::TRANSACTION_TIME:
        {
            if (a->temporalPart.transactionTimeCode == b->temporalPart.transactionTimeCode) {
                temporalEx = true;
            }
        }
        break;
    }

    if (temporalEx) {
        switch (spatialtype) {
            case SpatialType::POINT:
            {
                float2 pa = ((gpudb::GpuPoint*)a->spatialPart.key)->p;
                float2 pb = ((gpudb::GpuPoint*)b->spatialPart.key)->p;
                if (pa == pb) {
                    spatialEx = true;
                }
            }
            break;
            case SpatialType::LINE:
            {
                gpudb::GpuLine *lineA = (gpudb::GpuLine*)a->spatialPart.key;
                gpudb::GpuLine *lineB = (gpudb::GpuLine*)b->spatialPart.key;
                if (lineA->size == lineB->size) {
                    bool identical = true;
                    for (int i = 0; i < lineA->size; i++) {
                        float2 pa = lineA->points[i];
                        float2 pb = lineB->points[i];
                        if (pa != pb) {
                            identical = false;
                            break;
                        }
                    }
                    if (identical) {
                        spatialEx = true;
                    }
                }
            }
            break;
            case SpatialType::POLYGON:
            {
                gpudb::GpuPolygon *polygonA = reinterpret_cast<gpudb::GpuPolygon*>(a->spatialPart.key);
                gpudb::GpuPolygon *polygonB = reinterpret_cast<gpudb::GpuPolygon*>(b->spatialPart.key);
                if (polygonA->size == polygonB->size) {
                    bool identical = true;
                    for (int i = 0; i < polygonA->size; i++) {
                        float2 pa = polygonA->points[i];
                        float2 pb = polygonB->points[i];
                        if (pa != pb) {
                            identical = false;
                            break;
                        }
                    }
                    if (identical) {
                        spatialEx = true;
                    }
                }
            }
            break;
        }
    }
    return temporalEx && spatialEx;
}
