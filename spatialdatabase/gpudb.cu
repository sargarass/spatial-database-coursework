#include "gpudb.h"
#include "exception"

gpudb::GpuTable::GpuTable() {
    this->name[0] = 0;
    this->rowMemSize = 0;


}

gpudb::GpuTable::~GpuTable() {
    gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "destructor");
    for (GpuRow *row : rows) {
        gpudb::gpuAllocator::getInstance().free(row);
    }
}

uint2 gpudb::TemporalKey::centroid() {
    uint2 res;
    switch(type) {
        case TemporalType::VALID_TIME:
            res.x = (validTimeS + ((validTimeE - validTimeS) / 2ULL)) >> 32ULL;
            res.y = 0xFFFFFFFF; // Зарезервированное значение -- значит у нас только 3-мерные координаты
            break;
        case TemporalType::TRANSACTION_TIME:
            res.x = (transactionTypeS + ((transactionTypeE - transactionTypeS) / 2ULL)) >> 32ULL;
            res.y = 0xFFFFFFFF;
            break;
        case TemporalType::BITEMPORAL_TIME:
            res.x = (validTimeS + ((validTimeE - validTimeS) / 2ULL)) >> 32ULL;
            res.y = (transactionTypeS + ((transactionTypeE - transactionTypeS) / 2ULL)) >> 32ULL;
            break;
        default:
            gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Unexpected temporalType Value");

            break;
    }
    return res;
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

    columns.resize(vec.size());
    columnsCPU.resize(vec.size());
    thrust::copy(vec.begin(), vec.end(), columns.begin());
    thrust::copy(vec.begin(), vec.end(), columnsCPU.begin());
    std::memcpy(this->spatialKey.name, table.spatialKeyName.c_str(), table.spatialKeyName.length());
    std::memcpy(this->temporalKey.name, table.temporalKeyName.c_str(), table.temporalKeyName.length());
    this->spatialKey.type = table.spatialKeyType;
    this->temporalKey.type = table.temporalKeyType;
    return true;
}

bool gpudb::GpuTable::insertRow(gpudb::GpuRow*  row) {
    this->rows.push_back(row);
    return true;
}
