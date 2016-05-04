#pragma once
#include "types.h"
#include "gpudb.h"

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

    bool operator==(Attribute const &b) const {
        return name == b.name;
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
        if (type == TemporalType::BITEMPORAL_TIME) {
            return validT && transactT;
        }
        if (type == TemporalType::TRANSACTION_TIME) {
            return transactT;
        }
        if (type == TemporalType::VALID_TIME) {
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
        if (std::find(values.begin(), values.end(), atr) != values.end()) {
            return false;
        }

        values.push_back(atr);
        return true;
    }

    bool delAttribute(std::string const &atr) {
        Attribute tmp;
        if (!tmp.setName(atr)) {
            return false;
        }
        auto found = std::find(values.begin(), values.end(), tmp);
        if (found == values.end()) {
            return false;
        }

        values.erase(found);
        return true;
    }

    void clearAttributes() {
        values.clear();
    }

protected:
    std::vector<Attribute> values;
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
bool Attribute::setValue(bool isNull, char const *val);
template<>
bool Attribute::setValue(bool isNull, Date const &date);
template<>
bool Attribute::setValue(bool isNull, Date const date);
