#include "row.h"
#include "temptable.h"

Result<void, Error<std::string>> Attribute::setNullValue(Type t) {
    this->type = t;
    this->isNullVal = true;
    return setValueImp(t, nullptr);
}

template<>
Result<void, Error<std::string>> Attribute::setValue(int64_t const val) {
    this->type = Type::INT;
    this->isNullVal = false;
    return setValueImp(Type::INT, &val);
}

template<>
Result<void, Error<std::string>> Attribute::setValue(int32_t const val) {
    this->type = Type::INT;
    this->isNullVal = false;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
Result<void, Error<std::string>> Attribute::setValue(uint32_t const val) {
    this->type = Type::INT;
    this->isNullVal = false;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
Result<void, Error<std::string>> Attribute::setValue(uint64_t const val) {
    this->type = Type::INT;
    this->isNullVal = false;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
Result<void, Error<std::string>> Attribute::setValue(float const val) {
    this->type = Type::REAL;
    this->isNullVal = false;
    double dval = val;
    return setValueImp(Type::REAL, &dval);
}

template<>
Result<void, Error<std::string>> Attribute::setValue(double const val) {
    this->type = Type::REAL;
    this->isNullVal = false;
    return setValueImp(Type::REAL, &val);
}

template<>
Result<void, Error<std::string>> Attribute::setValue(std::string const &val) {
    this->type = Type::STRING;
    this->isNullVal = false;
    return setValueImp(Type::STRING, val.c_str());
}

template<>
Result<void, Error<std::string>> Attribute::setValue(std::string val) {
    this->type = Type::STRING;
    this->isNullVal = false;
    return setValueImp(Type::STRING, val.c_str());
}

template<>
Result<void, Error<std::string>> Attribute::setValue(char const *val) {
    this->type = Type::STRING;
    this->isNullVal = false;
    return setValueImp(Type::STRING, val);
}

template<>
Result<void, Error<std::string>> Attribute::setValue(Date const &date) {
    if (date.isValid()) {
        this->type = Type::DATE_TYPE;
        this->isNullVal = false;
        return setValueImp(Type::DATE_TYPE, &date);
    }
    return MYERR_STRING("date is invalid");
}

bool Attribute::operator<(Attribute const &b) const {
    return name < b.name;
}

bool Attribute::operator==(Attribute const &b) const {
    return name == b.name;
}

Result<void, Error<std::string>> Attribute::setValueImp(Type t, void const *val) {
    if (value) {
        delete [] value;
        value = nullptr;
    }

    uint64_t type_size = typeSize(t);
    uint8_t* value_ptr = new (std::nothrow) uint8_t[type_size];

    if (value_ptr == nullptr) {
        return MYERR_STRING("not enough memory (value_ptr == nullptr)");
    }

    memset(value_ptr, 0, type_size);
    switch(t) {
        case Type::STRING:
            {
                if (val != nullptr) {
                    strncpy((char*)value_ptr, (char const *)val, type_size);
                    value_ptr[type_size - 1] = 0;
                }
            }
            break;
        default:
            if (val != nullptr) {
                memcpy(value_ptr, val, type_size);
            }
            break;
    }

    value = reinterpret_cast<void*>(value_ptr);
    return Ok();
}

Attribute::Attribute() {
    type = Type::UNKNOWN;
    isNullVal = true;
    value = nullptr;
}

Attribute::~Attribute() {
    if (value != nullptr) {
        delete [] value;
    }
    value = nullptr;
    isNullVal = true;
    type = Type::UNKNOWN;
}

void Attribute::operator=(Attribute const &c) {
    type = c.type;
    isNullVal = c.isNullVal;
    name = c.name;

    void *old = value;
    value = nullptr;

    setValueImp(c.type, c.value);

    if (old != nullptr) {
        delete [] old;
    }
}

Attribute::Attribute(Attribute const &c)
    : Attribute() {
    *this = c;
}

Result<void, Error<std::string>> Attribute::setName(std::string name) {
    if (name.length() > NAME_MAX_LEN) {
        return MYERR_STRING(string_format("name len %d is great than max len %d", name.length(), NAME_MAX_LEN));
    }

    this->name = name;
    return Ok();
}

std::string Attribute::getName() const {
    return name;
}

bool Attribute::isNull() const {
    return this->isNullVal;
}

Type Attribute::getType() const {
    return type;
}

Result<std::string, Error<std::string>> Attribute::getString() const {
    if (this->type == Type::STRING) {
        return Ok(std::string((char*)value));
    }
    return MYERR_STRING("Attribute has different type!");
}

Result<int64_t, Error<std::string>> Attribute::getInt() const {
    if (this->type == Type::INT) {
        return Ok(*(int64_t*)value);
    }
    return MYERR_STRING("Attribute has different type!");
}

Result<double, Error<std::string>> Attribute::getReal() const {
    if (this->type == Type::REAL) {
        return Ok(*(double*)value);
    }
    return MYERR_STRING("Attribute has different type!");
}

Result<std::unique_ptr<TempTable>, Error<std::string>> Attribute::getSet() const {
    if (this->type == Type::SET) {
        std::unique_ptr<TempTable> tmp { ((gpudb::GpuSet*)value)->temptable };
        std::default_delete<TempTable> deleter;
        deleter.bdelete = false;
        tmp.get_deleter() = deleter;
        return Ok(std::move(tmp));
    }
    return MYERR_STRING("Attribute has different type!");
}

template<typename T>
bool setValue(T val = T()) {
    UNUSED_PARAM_HANDLER(val);
    return false;
}

