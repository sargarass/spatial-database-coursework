#pragma once
#include "types.h"
#include "aabb.h"
#include "log.h"
#include "tabledescription.h"
#include "gpuallocator.h"
#include "gpustackallocator.h"
#include "stackallocator.h"
#include "thrust/device_vector.h"
//Miller cylindrical projection

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

    struct GpuLine {
        float2 *points;
        size_t size;
    };

    struct GpuPolygon {

    };

    struct GpuPoint {
        float2 p;
    };

    struct SpatialKey {
        SpatialType type;
        void *key;
        AABB boundingBox();
    };

    struct TemporalKey {
        TemporalType type;
        uint64_t validTimeS;
        uint64_t validTimeE;

        uint64_t transactionTypeS;
        uint64_t transactionTypeE;

        uint2 centroid();
    };

    struct Value {
        bool isNull;
        void *value;
    };

    struct GpuRow {
        SpatialKey spatialPart;
        TemporalKey temporalPart;
        Value *value;
        uint64_t valueSize;
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
        uint64_t rowMemSize;
        bool set(TableDescription table);
        bool setName(std::string const &string);
        bool insertRow(gpudb::GpuRow*  row);
    };
}
