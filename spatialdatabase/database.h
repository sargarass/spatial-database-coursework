#pragma once
#include "gpudb.h"
#include "types.h"
#include "tabledescription.h"

class Attribute {
public:
    bool setName(std::string name) {
        if (name.length() > NAME_MAX_LEN) {
            return false;
        }

        this->name = name;
        return true;
    }

    template<typename T>
    bool setValue(bool isNull, T val = T()) {
        return false;
    }

    bool operator<(Attribute const &b) {
        return name < b.name;
    }

private:
    template<typename T>
    bool setValueImp(T val) {
        if (value) {
            delete [] (char*)(value);
        }

        T* value_ptr = new T;
        *value_ptr = val;
        value = reinterpret_cast<void*>(value_ptr);
        return true;
    }

    std::string name;
    Type type;
    bool isNull;
    void *value;
};

class Row {
    SpatialType spatialKey;
    TemporalType temporalKey;
    uint64_t keyValidTime;
    uint64_t keyTemporalTIme;
    std::set<Attribute> values;
};


class DataBase {
    typedef std::pair<std::string, TableDescription> tablesTypePair;
    typedef std::pair<std::string, gpudb::GpuTable> tablesPair;
public:
    bool insertRow(std::string tableName, Row row);
    bool createTable(TableDescription table);

private:
    std::map<std::string, gpudb::GpuTable> tables;
    std::map<std::string, TableDescription> tablesType;

};
