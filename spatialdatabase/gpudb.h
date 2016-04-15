#pragma once
#include "types.h"
#include "aabb.h"
#include "log.h"
#include "tabledescription.h"
#include "gpuallocator.h"
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
        GpuTable();
        char name[NAME_MAX_LEN];
        GpuColumnSpatialKey spatialKey;
        GpuColumnTemoralKey temporalKey;
        GpuColumnAttribute *columns;
        uint64_t columnsSize;
        GpuRow *rows;
        uint64_t rowsSize;

        bool set(TableDescription table);
        bool setName(std::string const &string);
    };
}
