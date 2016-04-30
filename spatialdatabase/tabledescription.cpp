#include "tabledescription.h"

bool TableDescription::setName(std::string newName) {
    if (newName.length() > NAME_MAX_LEN) {
        return false;
    }

    name = newName;
    return true;
}

bool TableDescription::setSpatialKey(std::string const keyName, SpatialType keyType) {
    if (keyName.length() > NAME_MAX_LEN) {
        return false;
    }

    spatialKeyName = keyName;
    spatialKeyType = keyType;
    return true;
}

bool TableDescription::setTemporalKey(std::string const keyName, TemporalType keyType) {
    if (keyName.length() > NAME_MAX_LEN) {
        return false;
    }

    temporalKeyName = keyName;
    temporalKeyType = keyType;
    return true;
}

bool TableDescription::addColumn(AttributeDescription col) {
    auto res = columnDescription.insert(col);
    return res.second;
}

bool TableDescription::delColumn(AttributeDescription col) {
    return columnDescription.erase(col);
}

bool TableDescription::operator<(TableDescription const &b) const {
    return name < b.name;
}

uint64_t TableDescription::getRowMemoryValuesSize() {
    uint64_t res = 0;
    for (auto& v : this->columnDescription) {
        res+=typeSize(v.type);
    }
    return res;
}
