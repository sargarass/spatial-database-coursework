#include "gpudb.h"

uint2 gpudb::TemporalKey::centroid() {
    uint2 res;
    switch(temporalType) {
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
            Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR,
                       "gpudb::TemporalKey",
                       "centroid",
                       "Unexpected temporalType Value");
            break;
    }
    return res;
}

bool gpudb::GpuTable::setName(std::string const &string) {
    if (string.length() > NAME_MAX_LEN) {
       Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "GpuTable", "setName",
                  "input string length is greater than NAME_MAX_LEN");
       return false;
    }

    std::memcpy(name, string.c_str(), string.length());
    return true;
}

gpudb::GpuTable::GpuTable() {
    this->columnsSize = 0;
    this->rowsSize = 0;
    this->rows = nullptr;
    this->columns = nullptr;
    this->name[0] = 0;
}

bool gpudb::GpuTable::set(TableDescription table) {
    if (!setName(table.name)) {
        return false;
    }
    this->columnsSize = table.columnDescription.size();
    this->columns = gpuAllocator::getInstance().alloc<GpuColumnAttribute>(columnsSize);
    std::memcpy(this->spatialKey.name, table.spatialKeyName.c_str(), table.spatialKeyName.length());
    std::memcpy(this->temporalKey.name, table.temporalKeyName.c_str(), table.temporalKeyName.length());
    this->spatialKey.type = table.spatialKeyType;
    this->temporalKey.type = table.temporalKeyType;
    this->rowsSize = 0;
    return true;
}

