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
    Result<void, Error<std::string>> setName(std::string name);

    std::string getName() const;
    bool isNull() const;
    Type getType() const;
    Result<std::string, Error<std::string>> getString() const;
    Result<int64_t, Error<std::string>> getInt() const;
    Result<double, Error<std::string>> getReal() const;
    Result<std::unique_ptr<TempTable>, Error<std::string>> getSet() const;

    template<typename T>
    Result<void, Error<std::string>> setValue(T val = T()) {
        UNUSED_PARAM_HANDLER(val);
        return MYERR_STRING("unsupported T type");
    }

    Result<void, Error<std::string>> setNullValue(Type t);

    bool operator<(Attribute const &b) const;
    bool operator==(Attribute const &b) const;
private:
    Result<void, Error<std::string>> setValueImp(Type t, void const *val);

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

    Result<void, Error<std::string>> addAttribute(Attribute const& atr) {
        if (std::find(values.begin(), values.end(), atr) != values.end()) {
            return MYERR_STRING(std::string("atr with same name already inside"));
        }

        values.push_back(atr);
        return Ok();
    }

    Result<void, Error<std::string>> delAttribute(std::string const &atr) {
        Attribute tmp;
        TRY(tmp.setName(atr));

        auto found = std::find(values.begin(), values.end(), tmp);

        if (found == values.end()) {
            return MYERR_STRING(string_format("atr with name %s is not exist", atr.c_str()));
        }

        values.erase(found);
        return Ok();
    }

    uint getAttributeSize() {
        return values.size();
    }

    Result<Attribute, Error<std::string>> getAttribute(uint id) {
        if (id >= values.size()) {
            return MYERR_STRING(string_format("id is to big: wait 0 <= id <= %zu", values.size()));
        }
        return Ok(values[id]);
    }

    void clearAttributes() {
        values.clear();
    }

protected:
    std::vector<Attribute> values;
};

template<>
Result<void, Error<std::string>> Attribute::setValue(int64_t const val);
template<>
Result<void, Error<std::string>> Attribute::setValue(int32_t const val);
template<>
Result<void, Error<std::string>> Attribute::setValue(uint32_t const val);
template<>
Result<void, Error<std::string>> Attribute::setValue(uint64_t const val);
template<>
Result<void, Error<std::string>> Attribute::setValue(float const val);
template<>
Result<void, Error<std::string>> Attribute::setValue(double const val);
template<>
Result<void, Error<std::string>> Attribute::setValue(std::string const &val);
template<>
Result<void, Error<std::string>> Attribute::setValue(std::string val);
template<>
Result<void, Error<std::string>> Attribute::setValue(char const *val);
template<>
Result<void, Error<std::string>> Attribute::setValue(Date const &date);
