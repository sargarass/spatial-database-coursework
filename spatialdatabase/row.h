#pragma once
#include "types.h"
#include "gpudb.h"
class TempTable;

class Attribute {
    friend class gpudb::GpuTable;
    friend class DataBase;
public:
    Attribute();
    ~Attribute();
    void operator=(Attribute const &c);
    Attribute(Attribute const &c);
    bool setName(std::string name);
    std::string getName() const;
    bool isNull() const;
    Type getType() const;
    std::string getString() const;
    int64_t getInt() const;
    double getReal() const;
    TempTable *getSet() const;

    template<typename T>
    bool setValue(T val = T()) {
        UNUSED_PARAM_HANDLER(val);
        return false;
    }

    bool setNullValue(Type t);

    bool operator<(Attribute const &b) const;
    bool operator==(Attribute const &b) const;
private:
    bool setValueImp(Type t, void const *val);

    std::string name;
    Type type;
    bool isNullVal;
    void *value;
};

class SpatialKey {
public:
    SpatialType type;
    std::string name;
    std::vector<float2> points;
    bool isValid();
};

class TemportalKey {
    friend class DataBase;
public:
    std::string name;
    TemporalType type;

    Date validTimeS;
    Date validTimeE;
    Date transactionTime;

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

    uint getAttributeSize() {
        return values.size();
    }

    Attribute const &getAttribute(uint id) {
        return values[id];
    }

    void clearAttributes() {
        values.clear();
    }

protected:
    std::vector<Attribute> values;
};

template<>
bool Attribute::setValue(int64_t const val);
template<>
bool Attribute::setValue(int32_t const val);
template<>
bool Attribute::setValue(uint32_t const val);
template<>
bool Attribute::setValue(uint64_t const val);
template<>
bool Attribute::setValue(float const val);
template<>
bool Attribute::setValue(double const val);
template<>
bool Attribute::setValue(std::string const &val);
template<>
bool Attribute::setValue(std::string val);
template<>
bool Attribute::setValue(char *val);
template<>
bool Attribute::setValue(char const *val);
template<>
bool Attribute::setValue(Date const &date);
template<>
bool Attribute::setValue(Date const date);
