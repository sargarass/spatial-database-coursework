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
        bool setName(std::string const &string);
        bool insertRow(TableDescription &descriptor, gpudb::GpuRow*  row, uint64_t memsize);
    };

}
