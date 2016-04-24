#pragma once
#include "gpudb.h"
#include "stackallocator.h"
#include "types.h"
#include "tabledescription.h"

class Attribute {
    friend class gpudb::GpuTable;
    friend class DataBase;
public:
    Attribute() {
        type = Type::UNKNOWN;
        isNull = true;
        value = nullptr;
    }

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

    bool operator<(Attribute const &b) const {
        return name < b.name;
    }

private:
    bool setValueImp(Type t, void const *val) {
        if (value) {
            delete [] (char*)(value);
        }
        uint64_t type_size = typeSize(t);
        uint8_t* value_ptr = new uint8_t[type_size];
        switch(t) {
            case Type::STRING:
                {
                    memset(value_ptr, 0, type_size);
                    uint64_t copysize = std::min(type_size, (uint64_t)(strlen((char*) val)));
                    memcpy(value_ptr, val, copysize);
                    value_ptr[copysize] = 0;
                }
                break;
            default:
                memcpy(value_ptr, val, type_size);
                break;
        }

        value = reinterpret_cast<void*>(value_ptr);
        return true;
    }

    std::string name;
    Type type;
    bool isNull;
    void *value;
};

class SpatialKey {
public:
    SpatialType type;
    std::string name;
    std::list<float2> points;
};

class TemportalKey {
public:
    std::string name;
    TemporalType type;
    uint64_t validTimeS;
    uint64_t validTimeE;

    uint64_t transactionTypeS;
    uint64_t transactionTypeE;
};

class Row {
public:
    SpatialKey spatialKey;
    TemportalKey temporalKey;
    std::set<Attribute> values;
};


class DataBase : public Singleton {
    typedef std::pair<std::string, TableDescription> tablesTypePair;
    typedef std::pair<std::string, gpudb::GpuTable*> tablesPair;
public:
    bool insertRow(std::string tableName, Row row);
    bool createTable(TableDescription table);
    bool showTable(std::string tableName);

    static DataBase &getInstance() {
        static DataBase *db = new DataBase;
        static bool init = false;
        if (init == false) {
            init = true;
            SingletonFactory::getInstance().registration<DataBase>(db);
            dynamic_cast<Singleton*>(db)->dependOn(Log::getInstance());
            dynamic_cast<Singleton*>(db)->dependOn(gpudb::gpuAllocator::getInstance());
            dynamic_cast<Singleton*>(db)->dependOn(gpudb::GpuStackAllocator::getInstance());
        }
        return *db;
    }

    void deinit() {
        for (auto& v : tables) {
            delete (v.second);
        }
        tables.clear();
        tablesType.clear();
    }

    virtual ~DataBase() {
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "delete DataBase");
        deinit();
    }
private:
    DataBase(){
        gpudb::GpuStackAllocator::getInstance().resize(512 * 1024 * 1024);
        StackAllocator::getInstance().resize(512 * 1024 * 1024);
    }
    std::map<std::string, gpudb::GpuTable*> tables;
    std::map<std::string, TableDescription> tablesType;
};
