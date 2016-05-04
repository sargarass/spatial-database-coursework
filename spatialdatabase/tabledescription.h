#pragma once
#include "types.h"
namespace gpudb {
    class GpuTable;
}
class TempTable;


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

class DataBase;

class TableDescription {
    friend bool pointxpointKnearestNeighbor(TempTable const &a, TempTable &b, uint k, TempTable &result);
    friend class DataBase;
    friend class gpudb::GpuTable;
    friend class TempTable;
    std::string name;
    std::string spatialKeyName;
    SpatialType spatialKeyType;
    TemporalType temporalKeyType;
    std::string temporalKeyName;
    std::vector<AttributeDescription> columnDescription;
    uint64_t getRowMemoryValuesSize();
public:
    bool setName(std::string newName);
    bool setSpatialKey(std::string const keyName, SpatialType keyType);
    bool setTemporalKey(std::string const keyName, TemporalType keyType);
    bool addColumn(AttributeDescription col);
    bool delColumn(AttributeDescription col);
    bool operator<(TableDescription const &b) const;
};
