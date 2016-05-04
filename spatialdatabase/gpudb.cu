#include "gpudb.h"
#include "exception"
#include "utils.h"
#include <cub/cub/cub.cuh>
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

gpudb::GpuTable::GpuTable() {
    this->name[0] = 0;
    this->rowReferenses = false;
}

gpudb::GpuTable::~GpuTable() {
    gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "destructor");
    if (!rowReferenses) {
        for (GpuRow *row : rows) {
            gpudb::gpuAllocator::getInstance().free(row);
        }
    }
    if (bvh.isBuilded()) {
        bvh.free();
    }
}

void gpudb::SpatialKey::boundingBox(AABB *box) {
    switch(type) {
        case SpatialType::POINT:
        {
            GpuPoint k = *((GpuPoint*)key);
            AABBmin(box->x) = k.p.x;
            AABBmax(box->x) = k.p.x;

            AABBmin(box->y) = k.p.y;
            AABBmax(box->y) = k.p.y;
        }
        break;
        case SpatialType::LINE:
        {
            GpuLine line = *((GpuLine*)key);
            float2 min = line.points[0];
            float2 max = min;
            for (uint i = 1; i < line.size; i++) {
                min = fmin(min, line.points[i]);
                max = fmax(max, line.points[i]);
            }
            AABBmin(box->x) = min.x;
            AABBmin(box->y) = min.y;
            AABBmax(box->x) = max.x;
            AABBmax(box->y) = max.y;
        }
        break;
        case SpatialType::POLYGON:
        {
            GpuPolygon polygon = *((GpuPolygon*)key);
            float2 min = polygon.points[0];
            float2 max = min;
            for (uint i = 1; i < polygon.size; i++) {
                min = fmin(min, polygon.points[i]);
                max = fmax(max, polygon.points[i]);
            }
            AABBmin(box->x) = min.x;
            AABBmin(box->y) = min.y;
            AABBmax(box->x) = max.x;
            AABBmax(box->y) = max.y;
        }
        break;
    }
    box->numComp = 2;
}

void gpudb::TemporalKey::boundingBox(AABB *box) {
    switch(type) {
        case TemporalType::VALID_TIME:
            box->numComp = 3;
            AABBmin(box->z) = this->validTimeSCode * CODE_NORMALIZE;
            AABBmax(box->z) = this->validTimeECode * CODE_NORMALIZE;
            break;
        case TemporalType::TRANSACTION_TIME:
            box->numComp = 3;
            AABBmin(box->z) = this->transactionTimeCode * CODE_NORMALIZE;
            AABBmax(box->z) = AABBmin(box->z);
            break;
        case TemporalType::BITEMPORAL_TIME:
            box->numComp = 4;
            AABBmin(box->z) = this->validTimeSCode * CODE_NORMALIZE;
            AABBmax(box->z) = this->validTimeECode * CODE_NORMALIZE;
            AABBmin(box->w) = this->transactionTimeCode * CODE_NORMALIZE;
            AABBmax(box->w) = AABBmin(box->w);
            break;
    }
}

bool gpudb::GpuTable::setName(std::string const &string) {
    if (string.length() > NAME_MAX_LEN) {
       gLogWrite(LOG_MESSAGE_TYPE::ERROR, "input string length is greater than NAME_MAX_LEN");
       return false;
    }

    std::memcpy(name, string.c_str(), string.length());
    return true;
}

bool gpudb::GpuTable::set(TableDescription table) {
    if (!setName(table.name)) {
        return false;
    }

    thrust::host_vector<GpuColumnAttribute> vec;
    for (auto& col : table.columnDescription) {
        GpuColumnAttribute att;
        if (col.name.length() == 0) {
            return false;
        }

        if (col.type == Type::UNKNOWN) {
            return false;
        }

        memcpy(att.name, col.name.c_str(), col.name.length());
        att.type = col.type;
        vec.push_back(att);
    }

    columns.reserve(vec.size());
    thrust::copy(vec.begin(), vec.end(), columns.begin());
    return true;

}

template<SpatialType spatialtype, TemporalType temporaltype>
__global__
void testIdenticalKeys(gpudb::GpuRow** rows, uint size, gpudb::GpuRow *nRow, int *result) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= size) {
        return;
    }
    gpudb::GpuRow row = *rows[idx];
    gpudb::GpuRow newRow = *nRow;
    bool spatialEx = false;
    bool temporalEx = false;

    switch(temporaltype) {
        case TemporalType::BITEMPORAL_TIME:
        {
            if (row.temporalPart.validTimeSCode == newRow.temporalPart.validTimeSCode &&
                row.temporalPart.validTimeECode == newRow.temporalPart.validTimeECode &&
                row.temporalPart.transactionTimeCode == newRow.temporalPart.transactionTimeCode) {
                temporalEx = true;
            }
        }
        break;
        case TemporalType::VALID_TIME:
        {
            if (row.temporalPart.validTimeSCode == newRow.temporalPart.validTimeSCode &&
                row.temporalPart.validTimeECode == newRow.temporalPart.validTimeECode) {
                temporalEx = true;
            }
        }
        break;
        case TemporalType::TRANSACTION_TIME:
        {
            if (row.temporalPart.transactionTimeCode == newRow.temporalPart.transactionTimeCode) {
                temporalEx = true;
            }
        }
        break;
    }

    if (temporalEx) {
        switch (spatialtype) {
            case SpatialType::POINT:
            {
                gpudb::GpuPoint *point = (gpudb::GpuPoint*)row.spatialPart.key;
                gpudb::GpuPoint *pointNewRow = (gpudb::GpuPoint*)newRow.spatialPart.key;
                if (point->p == pointNewRow->p) {
                    spatialEx = true;
                }
            }
            break;
            case SpatialType::LINE:
            {
                gpudb::GpuLine *line = (gpudb::GpuLine*)row.spatialPart.key;
                gpudb::GpuLine *lineNewRow = (gpudb::GpuLine*)newRow.spatialPart.key;
                if (line->size == lineNewRow->size) {
                    bool identical = true;
                    for (int i = 0; i < line->size; i++) {
                        if (line->points[i] != lineNewRow->points[i]) {
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
                gpudb::GpuPolygon *polygon = reinterpret_cast<gpudb::GpuPolygon*>(row.spatialPart.key);
                gpudb::GpuPolygon *polygonNewRow = reinterpret_cast<gpudb::GpuPolygon*>(newRow.spatialPart.key);
                if (polygon->size == polygonNewRow->size) {
                    bool identical = true;
                    for (int i = 0; i < polygon->size; i++) {
                        if (polygon->points[i] != polygon->points[i]) {
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


    result[idx] = temporalEx && spatialEx;
}

bool gpudb::GpuTable::insertRow(TableDescription &descriptor,gpudb::GpuRow* row, uint64_t memsize) {
    gpudb::GpuStackAllocator::getInstance().pushPosition();
    do {
        int cpuRes = 0;
        if (rows.size() > 0) {
            int *result = gpudb::GpuStackAllocator::getInstance().alloc<int>(rows.size());
            size_t temp_size = 0;
            cub::DeviceReduce::Sum(nullptr, temp_size, result, &cpuRes, rows.size());
            uint8_t *tmp = gpudb::GpuStackAllocator::getInstance().alloc<uint8_t>(temp_size);
            int *resgpu = gpudb::GpuStackAllocator::getInstance().alloc<int>();
            if (result == nullptr || tmp == nullptr || resgpu == nullptr) {
                break;
            }

            dim3 block(BLOCK_SIZE);
            dim3 grid = gridConfigure(rows.size(), block);

            SWITCH_RUN(descriptor.spatialKeyType, descriptor.temporalKeyType, testIdenticalKeys, grid, block, thrust::raw_pointer_cast(rows.data()), rows.size(), row, result);
            cub::DeviceReduce::Sum(tmp, temp_size, result, resgpu, rows.size());
            cudaMemcpy(&cpuRes, resgpu, sizeof(int), cudaMemcpyDeviceToHost);
            gpudb::GpuStackAllocator::getInstance().free(result);
        }

        gpudb::GpuStackAllocator::getInstance().popPosition();

        if (cpuRes) {
            return false;
        }

        this->rows.push_back(row);
        this->rowsSize.push_back(memsize);
        return true;
    } while(0);
    gpudb::GpuStackAllocator::getInstance().popPosition();
    return false;
}
