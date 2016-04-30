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
        UNUSED_PARAM_HANDLER(isNull, val);
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
    std::vector<float2> points;

    bool isValid();
    bool addVertex(float2 v) {UNUSED_PARAM_HANDLER(v); return false; }
    bool delVertex(float2 v) {UNUSED_PARAM_HANDLER(v); return false; }
};

class TemportalKey {
    friend class DataBase;
public:
    std::string name;
    TemporalType type;

    Date validTimeS;
    Date validTimeE;

    bool isValid() {
        bool validT = validTimeE.isValid() && validTimeS.isValid() && (validTimeS.codeDate() <= validTimeE.codeDate());
        bool transactT = transactionTime.isValid();
        if (type == BITEMPORAL_TIME) {
            return validT && transactT;
        }
        if (type == TRANSACTION_TIME) {
            return transactT;
        }
        if (type == VALID_TIME) {
            return validT;
        }
        return false;
    }
protected:
    Date transactionTime;
};

class Row {
    friend class DataBase;
public:
    SpatialKey spatialKey;
    TemportalKey temporalKey;

    bool addAttribute(Attribute const& atr) {
        return values.insert(atr).second;
    }

    bool delAttribute(std::string const &atr) {
        Attribute tmp;
        if (!tmp.setName(atr)) {
            return false;
        }

        return values.erase(tmp);
    }

    void clearAttributes() {
        values.clear();
    }

protected:
    std::set<Attribute> values;
};


class DataBase : public Singleton {
    typedef std::pair<std::string, TableDescription> tablesTypePair;
    typedef std::pair<std::string, gpudb::GpuTable*> tablesPair;
public:
    bool insertRow(std::string tableName, Row row);
    bool createTable(TableDescription table);
    bool showTable(std::string tableName);
    bool showTable(gpudb::GpuTable const &table, TableDescription &description);
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
    void storeGPU(gpudb::GpuRow *dst, gpudb::GpuRow *src, uint64_t memsize);
    void loadCPU(gpudb::GpuRow *dstCPU, gpudb::GpuRow *srcGPU, uint64_t &memsize);

    DataBase(){
        gpudb::GpuStackAllocator::getInstance().resize(512ULL * 1024ULL * 1024ULL);
        StackAllocator::getInstance().resize(1024ULL * 1024ULL * 1024ULL);
    }
    std::map<std::string, gpudb::GpuTable*> tables;
    std::map<std::string, TableDescription> tablesType;
};

template<>
bool Attribute::setValue(bool isNull, int64_t const val);
template<>
bool Attribute::setValue(bool isNull, int32_t const val);
template<>
bool Attribute::setValue(bool isNull, uint32_t const val);
template<>
bool Attribute::setValue(bool isNull, uint64_t const val);
template<>
bool Attribute::setValue(bool isNull, float const val);
template<>
bool Attribute::setValue(bool isNull, double const val);
template<>
bool Attribute::setValue(bool isNull, std::string const &val);
template<>
bool Attribute::setValue(bool isNull, std::string val);
template<>
bool Attribute::setValue(bool isNull, char *val);
template<>
bool Attribute::setValue(bool isNull, Date const &date);
template<>
bool Attribute::setValue(bool isNull, Date const date);
