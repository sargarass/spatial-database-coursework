#include "database.h"

template<>
bool Attribute::setValue(bool isNull, int64_t val) {
    this->type = Type::INT;
    this->isNull = isNull;
    return setValueImp(val);
}

template<>
bool Attribute::setValue(bool isNull, float val) {
    this->type = Type::REAL;
    this->isNull = isNull;
    return setValueImp((double)(val));
}

template<>
bool Attribute::setValue(bool isNull, double val) {
    this->type = Type::REAL;
    this->isNull = isNull;
    return setValueImp(val);
}

bool DataBase::createTable(TableDescription table) {
    if (tablesType.find(table.name) != tablesType.end()
        || (table.name.length() > NAME_MAX_LEN)
        || (table.columnDescription.size() == 0)) {
        return false;
    }

    tablesType.insert(tablesTypePair(table.name, table));
    gpudb::GpuTable gputable;
/// тут мы копируем заголовок таблицы
    /*thrust::host_vector<gpudb::GpuColumnAttribute> toGPU;
    for (auto &elem : table.columnDescription) {
        gpudb::GpuColumnAttribute col;
        memcpy(col.name, elem.name.c_str(), NAME_MAX_LEN);
        col.type = elem.type;
        toGPU.push_back(col);
    }
    gputable.columns = toGPU;*/

    std::memcpy(gputable.name, table.name.c_str(), table.name.length());
    tables.insert(tablesPair(table.name, gputable));
    return true;
}
