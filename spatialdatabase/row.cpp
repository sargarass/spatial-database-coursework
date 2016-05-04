#include "row.h"

template<>
bool Attribute::setValue(bool isNull, int64_t const val) {
    this->type = Type::INT;
    this->isNull = isNull;
    return setValueImp(Type::INT, &val);
}

template<>
bool Attribute::setValue(bool isNull, int32_t const val) {
    this->type = Type::INT;
    this->isNull = isNull;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
bool Attribute::setValue(bool isNull, uint32_t const val) {
    this->type = Type::INT;
    this->isNull = isNull;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
bool Attribute::setValue(bool isNull, uint64_t const val) {
    this->type = Type::INT;
    this->isNull = isNull;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
bool Attribute::setValue(bool isNull, float const val) {
    this->type = Type::REAL;
    this->isNull = isNull;
    double dval = val;
    return setValueImp(Type::REAL, &dval);
}

template<>
bool Attribute::setValue(bool isNull, double const val) {
    this->type = Type::REAL;
    this->isNull = isNull;
    return setValueImp(Type::REAL, &val);
}

template<>
bool Attribute::setValue(bool isNull, std::string const &val) {
    this->type = Type::STRING;
    this->isNull = isNull;
    return setValueImp(Type::STRING, val.c_str());
}

template<>
bool Attribute::setValue(bool isNull, std::string val) {
    this->type = Type::STRING;
    this->isNull = isNull;
    return setValueImp(Type::STRING, val.c_str());
}

template<>
bool Attribute::setValue(bool isNull, char *val) {
    this->type = Type::STRING;
    this->isNull = isNull;
    return setValueImp(Type::STRING, val);
}

template<>
bool Attribute::setValue(bool isNull, char const *val) {
    this->type = Type::STRING;
    this->isNull = isNull;
    return setValueImp(Type::STRING, val);
}

template<>
bool Attribute::setValue(bool isNull, Date const &date) {
    if (date.isValid()) {
        this->type = Type::DATE_TYPE;
        this->isNull = isNull;
        return setValueImp(Type::DATE_TYPE, &date);
    }
    return false;
}

template<>
bool Attribute::setValue(bool isNull, Date const date) {
    if (date.isValid()) {
        this->type = Type::DATE_TYPE;
        this->isNull = isNull;
        return setValueImp(Type::DATE_TYPE, &date);
    }
    return false;
}

