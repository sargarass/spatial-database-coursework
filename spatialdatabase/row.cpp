#include "row.h"
#include "temptable.h"

bool Attribute::setNullValue(Type t) {
    this->type = t;
    this->isNullVal = true;
    return setValueImp(t, nullptr);
}

template<>
bool Attribute::setValue(int64_t const val) {
    this->type = Type::INT;
    this->isNullVal = false;
    return setValueImp(Type::INT, &val);
}

template<>
bool Attribute::setValue(int32_t const val) {
    this->type = Type::INT;
    this->isNullVal = false;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
bool Attribute::setValue(uint32_t const val) {
    this->type = Type::INT;
    this->isNullVal = false;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
bool Attribute::setValue(uint64_t const val) {
    this->type = Type::INT;
    this->isNullVal = false;
    int64_t ival = val;
    return setValueImp(Type::INT, &ival);
}

template<>
bool Attribute::setValue(float const val) {
    this->type = Type::REAL;
    this->isNullVal = false;
    double dval = val;
    return setValueImp(Type::REAL, &dval);
}

template<>
bool Attribute::setValue(double const val) {
    this->type = Type::REAL;
    this->isNullVal = false;
    return setValueImp(Type::REAL, &val);
}

template<>
bool Attribute::setValue(std::string const &val) {
    this->type = Type::STRING;
    this->isNullVal = false;
    return setValueImp(Type::STRING, val.c_str());
}

template<>
bool Attribute::setValue(std::string val) {
    this->type = Type::STRING;
    this->isNullVal = false;
    return setValueImp(Type::STRING, val.c_str());
}

template<>
bool Attribute::setValue(char *val) {
    this->type = Type::STRING;
    this->isNullVal = false;
    return setValueImp(Type::STRING, val);
}

template<>
bool Attribute::setValue(char const *val) {
    this->type = Type::STRING;
    this->isNullVal = false;
    return setValueImp(Type::STRING, val);
}

template<>
bool Attribute::setValue(Date const &date) {
    if (date.isValid()) {
        this->type = Type::DATE_TYPE;
        this->isNullVal = false;
        return setValueImp(Type::DATE_TYPE, &date);
    }
    return false;
}

template<>
bool Attribute::setValue(Date const date) {
    if (date.isValid()) {
        this->type = Type::DATE_TYPE;
        this->isNullVal = false;
        return setValueImp(Type::DATE_TYPE, &date);
    }
    return false;
}

bool Attribute::operator<(Attribute const &b) const {
    return name < b.name;
}

bool Attribute::operator==(Attribute const &b) const {
    return name == b.name;
}

bool Attribute::setValueImp(Type t, void const *val) {
    if (value) {
        delete [] value;
        value = nullptr;
    }

    uint64_t type_size = typeSize(t);
    uint8_t* value_ptr = new uint8_t[type_size];

    if (value_ptr == nullptr) {
        return false;
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
    return true;
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

bool Attribute::setName(std::string name) {
    if (name.length() > NAME_MAX_LEN) {
        return false;
    }

    this->name = name;
    return true;
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

std::string Attribute::getString() const {
    return std::string((char*)value);
}

int64_t Attribute::getInt() const {
    return *(int64_t*)value;
}

double Attribute::getReal() const {
    return *(double*)value;
}

TempTable *Attribute::getSet() const {
    return (((gpudb::GpuSet*)value)->temptable);
}

template<typename T>
bool setValue(T val = T()) {
    UNUSED_PARAM_HANDLER(val);
    return false;
}

