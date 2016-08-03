#include "tabledescription.h"

Result<void, Error<std::string>> TableDescription::setName(std::string newName) {
    if (newName.length() > NAME_MAX_LEN) {
        return MYERR_STRING(string_format("newName len %d is great than %d", newName.length(), NAME_MAX_LEN));
    }

    name = newName;
    return Ok();
}

Result<void, Error<std::string>> TableDescription::setSpatialKey(std::string const keyName, SpatialType keyType) {
    if (keyName.length() > NAME_MAX_LEN) {
        return MYERR_STRING((string_format("KeyName len %d is great than %d", keyName.length(), NAME_MAX_LEN)));
    }

    spatialKeyName = keyName;
    spatialKeyType = keyType;
    return Ok();
}

Result<void, Error<std::string>> TableDescription::setTemporalKey(std::string const keyName, TemporalType keyType) {
    if (keyName.length() > NAME_MAX_LEN) {
        return MYERR_STRING(string_format("KeyName len %d is great than %d", keyName.length(), NAME_MAX_LEN));
    }

    temporalKeyName = keyName;
    temporalKeyType = keyType;
    return Ok();
}

Result<void, Error<std::string>> TableDescription::addColumn(AttributeDescription col) {
    auto it = std::find(columnDescription.begin(), columnDescription.end(), col);

    if (it != columnDescription.end()) {
        return MYERR_STRING("Col is already exist");
    }

    columnDescription.push_back(col);
    return Ok();
}

Result<void, Error<std::string>> TableDescription::delColumn(AttributeDescription col) {
    auto it = std::find(columnDescription.begin(), columnDescription.end(), col);

    if (it == columnDescription.end()) {
        return MYERR_STRING("Col was not found");
    }
    columnDescription.erase(it);
    return Ok();
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
