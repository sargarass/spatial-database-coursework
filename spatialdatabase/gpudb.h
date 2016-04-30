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

namespace gpudb {

    struct GpuColumnSpatialKey {
        char name[NAME_MAX_LEN];
        SpatialType type;
    };

    struct GpuColumnTemoralKey {
        char name[NAME_MAX_LEN];
        TemporalType type;
    };

    struct GpuColumnAttribute {
        char name[NAME_MAX_LEN];
        Type type;
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
        SpatialType type;
        void *key;

        FUNC_PREFIX
        void operator=(SpatialKey const &b) {
            type = b.type;
            key = b.key;
        }

        __device__ __host__
        void boundingBox(AABB *box);
    };

    struct  __align__(16) TemporalKey {
        TemporalType type;
        uint64_t validTimeSCode;
        uint64_t validTimeECode;
        uint64_t transactionTimeCode;

        __device__ __host__
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
        uint64_t rowSize;
        void operator=(GpuRow const &b) {
            spatialPart = b.spatialPart;
            temporalPart = b.temporalPart;
            value = b.value;
            valueSize = b.valueSize;
            rowSize = b.rowSize;
        }
    };

    struct GpuTable {
        GpuTable();
        ~GpuTable();
        char name[NAME_MAX_LEN];
        GpuColumnSpatialKey spatialKey;
        GpuColumnTemoralKey temporalKey;
        thrust::device_vector<GpuColumnAttribute> columns;
        thrust::host_vector<GpuColumnAttribute> columnsCPU;
        thrust::device_vector<GpuRow*> rows;
        bool set(TableDescription table);
        bool setName(std::string const &string);
        bool insertRow(gpudb::GpuRow*  row);
    };

}
