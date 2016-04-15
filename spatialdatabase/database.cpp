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
    if (table.name.length() > NAME_MAX_LEN) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "DataBase", "createTable",
                                 "Cannot create table with name = \"%s\" length %d: max %d length",
                                 table.name.c_str(),
                                 table.name.length(),
                                 NAME_MAX_LEN);
        return false;
    }

    if (table.columnDescription.size() == 0) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "DataBase", "createTable",
                                 "Cannot create table with zero columns");
        return false;
    }

    if (tablesType.find(table.name) != tablesType.end()) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "DataBase", "createTable",
                                 "Cannot create table: table with this name is exist");
        return false;
    }

    tablesType.insert(tablesTypePair(table.name, table));
    gpudb::GpuTable gputable;
    gputable.set(table);
    tables.insert(tablesPair(table.name, gputable));
    return true;
}


bool DataBase::insertRow(std::string tableName, Row row) {
    return false;
}
