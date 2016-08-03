#include "gpudb.h"
#include "exception"
#include "utils.h"
#include "cub/cub/cub.cuh"

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
}

void gpudb::TemporalKey::boundingBox(AABB *box) {
    switch(type) {
        case TemporalType::VALID_TIME:
            AABBmin(box->z) = this->validTimeSCode * CODE_NORMALIZE;
            AABBmax(box->z) = this->validTimeECode * CODE_NORMALIZE;
            AABBmin(box->w) = 0;
            AABBmax(box->w) = 0;
            break;
        case TemporalType::TRANSACTION_TIME:
            AABBmin(box->z) = 0;
            AABBmax(box->z) = 0;
            AABBmin(box->w) = this->transactionTimeCode * CODE_NORMALIZE;
            AABBmax(box->w) = AABBmin(box->w);
            break;
        case TemporalType::BITEMPORAL_TIME:
            AABBmin(box->z) = this->validTimeSCode * CODE_NORMALIZE;
            AABBmax(box->z) = this->validTimeECode * CODE_NORMALIZE;
            AABBmin(box->w) = this->transactionTimeCode * CODE_NORMALIZE;
            AABBmax(box->w) = AABBmin(box->w);
            break;
    }
}

Result<void, Error<std::string>> gpudb::GpuTable::setName(char *dst, std::string const &src) {
    if (src.length() > NAME_MAX_LEN) {
       return MYERR_STRING("input string length is greater than NAME_MAX_LEN");
    }

    std::memcpy(dst, src.c_str(), typeSize(Type::STRING));
    name[typeSize(Type::STRING) - 1] = 0;

    return Ok();
}

Result<void, Error<std::string>> gpudb::GpuTable::set(TableDescription table) {
    TRY(setName(this->name, table.name));

    std::vector<GpuColumnAttribute> vec;
    for (auto& col : table.columnDescription) {
        GpuColumnAttribute att;

        if (col.type == Type::UNKNOWN || col.type == Type::SET) {
            return MYERR_STRING("Table has column with unknown or set type!");
        }

        TRY(setName(att.name, col.name));
        att.type = col.type;
        vec.push_back(att);
    }

    columns.resize(vec.size());
    thrust::copy(vec.begin(), vec.end(), columns.begin());
    return Ok();
}

template<SpatialType spatialtype, TemporalType temporaltype>
__global__
void testIdenticalKeys(gpudb::GpuRow** rows, uint size, gpudb::GpuRow *nRow, int *result) {
    uint idx = getGlobalIdx3DZXY();
    if (idx >= size) {
        return;
    }
    result[idx] = testIdenticalRowKeys<spatialtype, temporaltype>(rows[idx], nRow);
}

Result<void, Error<std::string>>
gpudb::GpuTable::insertRow(TableDescription &descriptor, gpudb::GpuRow* row, uint64_t memsize) {
        int cpuRes = 0;

        if (rows.size() > 0) {
            auto result = gpudb::GpuStackAllocatorAdditions::allocUnique<int>(rows.size());

            size_t temp_size = 0;
            cub::DeviceReduce::Sum(nullptr, temp_size, result.get(), &cpuRes, rows.size());

            auto tmp = gpudb::GpuStackAllocatorAdditions::allocUnique<uint8_t>(temp_size);
            auto resgpu = gpudb::GpuStackAllocatorAdditions::allocUnique<int>();

            if (result == nullptr || tmp == nullptr || resgpu == nullptr) {
                return MYERR_STRING("nullptr was returned");
            }

            dim3 block(BLOCK_SIZE);
            dim3 grid = gridConfigure(rows.size(), block);

            SWITCH_RUN(descriptor.spatialKeyType, descriptor.temporalKeyType, testIdenticalKeys, grid, block, thrust::raw_pointer_cast(rows.data()), rows.size(), row, result.get());
            cub::DeviceReduce::Sum(tmp.get(), temp_size, result.get(), resgpu.get(), rows.size());
            cudaMemcpy(&cpuRes, resgpu.get(), sizeof(int), cudaMemcpyDeviceToHost);
        }

        if (cpuRes) {
            return MYERR_STRING("row already exist in table");
        }

        this->rows.push_back(row);
        this->rowsSize.push_back(memsize);
        return Ok();
}
