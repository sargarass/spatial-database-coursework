#pragma once
#include "types.h"
namespace gpudb {
    class GpuTable;
}

class AttributeDescription {
public:
    std::string name;
    Type type;

    bool operator<(AttributeDescription const &b) const {
        return name < b.name;
    }
};

class DataBase;

class TableDescription {
    friend class DataBase;
    friend class gpudb::GpuTable;
    std::string name;
    std::string spatialKeyName;
    SpatialType spatialKeyType;
    TemporalType temporalKeyType;
    std::string temporalKeyName;
    std::set<AttributeDescription> columnDescription;

public:
    bool setName(std::string newName);
    bool setSpatialKey(std::string const keyName, SpatialType keyType);
    bool setTemporalKey(std::string const keyName, TemporalType keyType);
    bool addColumn(AttributeDescription col);
    bool delColumn(AttributeDescription col);
    bool operator<(TableDescription const &b) const;
};
