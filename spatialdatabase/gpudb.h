#pragma once
#include "types.h"
#include "aabb.h"
#include "log.h"
//Miller cylindrical projection

namespace gpudb {

    struct SpatialKey {
        SpatialType spatialType;

        AABB boundingBox();
    };

    struct TemporalKey {
        TemporalType temporalType;
        uint64_t validTimeS;
        uint64_t validTimeE;

        uint64_t transactionTypeS;
        uint64_t transactionTypeE;

        uint2 centroid();
    };

    struct GpuColumnSpatialKey {
        SpatialType temporalType;
    };

    struct GpuColumnTemoralKey {
        TemporalType temporalType;
    };

    struct GpuColumnAttribute {
        Type type;
        char name[NAME_MAX_LEN];
    };

    struct Value {
        bool isNull;
        void *value;
    };

    struct GpuRow {
        SpatialKey spatialPart;
        TemporalKey temporalPart;

        Value *value;
    };

    struct GpuTable {
        char name[NAME_MAX_LEN];
        GpuColumnSpatialKey spatialKey;
        GpuColumnTemoralKey temporalKey;
        GpuColumnAttribute *columns;
        uint64_t columnsSize;

        GpuRow *rows;
        uint64_t rowsSize;
    };
}
