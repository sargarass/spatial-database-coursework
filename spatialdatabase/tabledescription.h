#pragma once
#include "types.h"
namespace gpudb {
    class GpuTable;
}
class TempTable;
class DataBase;

class AttributeDescription {
    friend class gpudb::GpuTable;
public:
    std::string name;
    Type type;

    bool operator==(AttributeDescription const &b) const {
        return name == b.name;
    }

    bool operator<(AttributeDescription const &b) const {
        return name < b.name;
    }
};

class TableDescription {
public:
    Result<void, Error<std::string>> setName(std::string newName);
    Result<void, Error<std::string>> setSpatialKey(std::string const keyName, SpatialType keyType);
    Result<void, Error<std::string>> setTemporalKey(std::string const keyName, TemporalType keyType);
    Result<void, Error<std::string>> addColumn(AttributeDescription col);
    Result<void, Error<std::string>> delColumn(std::string colName);
private:
    friend class DataBase;
    friend class gpudb::GpuTable;
    friend class TempTable;
    friend class std::map<std::string, TableDescription>;
    friend class std::set<std::string, TableDescription>;
    std::string name;
    std::string spatialKeyName;
    SpatialType spatialKeyType;
    TemporalType temporalKeyType;
    std::string temporalKeyName;
    std::vector<AttributeDescription> columnDescription;
    uint64_t getRowMemoryValuesSize();
    bool operator<(TableDescription const &b) const;
};
