#include "database.h"
#include "gpuallocator.h"
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
    return setValueImp(Type::INT, &val);
}

template<>
bool Attribute::setValue(bool isNull, uint32_t const val) {
    this->type = Type::INT;
    this->isNull = isNull;
    return setValueImp(Type::INT, &val);
}

template<>
bool Attribute::setValue(bool isNull, uint64_t const val) {
    this->type = Type::INT;
    this->isNull = isNull;
    return setValueImp(Type::INT, &val);
}

template<>
bool Attribute::setValue(bool isNull, float const val) {
    this->type = Type::REAL;
    this->isNull = isNull;
    return setValueImp(Type::REAL, &val);
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
bool Attribute::setValue(bool isNull, char const *val) {
    this->type = Type::STRING;
    this->isNull = isNull;
    return setValueImp(Type::STRING, val);
}

bool DataBase::createTable(TableDescription table) {
    if (table.name.length() > NAME_MAX_LEN) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Cannot create table with name = \"%s\" length %d: max %d length",
                                 table.name.c_str(),
                                 table.name.length(),
                                 NAME_MAX_LEN);
        return false;
    }

    if (table.columnDescription.size() == 0) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Cannot create table with zero columns");
        return false;
    }

    if (tablesType.find(table.name) != tablesType.end()) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Cannot create table: table with this name is exist");
        return false;
    }

    tablesType.insert(tablesTypePair(table.name, table));
    gpudb::GpuTable *gputable = new gpudb::GpuTable();

    if (gputable == nullptr) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Cannot create table: allocation falled");
        return false;
    }

    if (gputable->set(table)) {
        tables.insert(tablesPair(table.name, gputable));
        return true;
    }
    delete gputable;
    return false;
}


bool DataBase::insertRow(std::string tableName, Row row) {
    auto tableType = tablesType.find(tableName);
    if (tableType == tablesType.end()) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Table \"%s\" is not exist", tableName.c_str());
        return false;
    }

    if (row.spatialKey.type != (*tableType).second.spatialKeyType) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "1");
        return false;
    }

    if (row.spatialKey.name != (*tableType).second.spatialKeyName) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "DataBase", "insertRow", "2");
        return false;
    }

    if (row.temporalKey.type != (*tableType).second.temporalKeyType) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "3");
        return false;
    }

    if (row.temporalKey.name != (*tableType).second.temporalKeyName) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "4");
        return false;
    }

    auto it = (*tableType).second.columnDescription.begin();
    uint64_t iter = 0;
    for (Attribute const &v : row.values) {
        if (v.name != (*it).name || v.type != (*it).type) {
            gLogWrite(LOG_MESSAGE_TYPE::WARNING,
                                     "Cannot insert row:\n"
                                     "attribute name:\n"
                                     " wait: \"%s\"\n"
                                     " get: \"%s\"\n"
                                     "type \n"
                                     " wait: \"%s\"\n"
                                     " get: \"%s\"\n",
                                     (*it).name.c_str(), v.name.c_str(), typeToString((*it).type).c_str(), typeToString(v.type).c_str());
            return false;
        }
        iter++;
        it++;
    }
    if (it != (*tableType).second.columnDescription.end()) {
        return false;
    }
    uint64_t memsize =  sizeof(gpudb::GpuRow) +
                        sizeof(gpudb::Value) * (*tableType).second.columnDescription.size()  +
                       (*tableType).second.getRowMemoryValuesSize();
    uint8_t *memory = new uint8_t[memsize];
    if (!memory) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough cpu memory ");
        return false;
    }

    gpudb::GpuRow *gpuRow = reinterpret_cast<gpudb::GpuRow *>(memory);
    gpuRow->spatialPart.type = row.spatialKey.type;
    //gpuRow->spatialPart.points = nullptr;
    //gpuRow->spatialPart.pointsSize = 0;
    gpuRow->value = reinterpret_cast<gpudb::Value*>(memory + sizeof(gpudb::GpuRow));
    gpuRow->valueSize = (*tableType).second.columnDescription.size();
    gpuRow->temporalPart.type = row.temporalKey.type;
    gpuRow->temporalPart.validTimeS = row.temporalKey.validTimeS;
    gpuRow->temporalPart.validTimeE = row.temporalKey.validTimeE;
    gpuRow->temporalPart.transactionTypeS = row.temporalKey.transactionTypeS;
    gpuRow->temporalPart.transactionTypeE = row.temporalKey.transactionTypeE;

    uint8_t *gpuMemory = gpudb::gpuAllocator::getInstance().alloc<uint8_t>(memsize);
    uint8_t *memoryValue = memory + sizeof(gpudb::GpuRow) + sizeof(gpudb::Value) * gpuRow->valueSize;
    uint8_t *gpuValue = gpuMemory + sizeof(gpudb::GpuRow) + sizeof(gpudb::Value) * gpuRow->valueSize;
    if (!gpuMemory) {
        delete [] memory;
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough gpu memory ");
        return false;
    }

    iter = 0;
    for (Attribute const &v : row.values) {
        gpuRow->value[iter].isNull = v.isNull;
        gpuRow->value[iter].value = gpuValue;

        uint64_t attrSize = typeSize(v.type);
        memcpy(memoryValue, v.value, attrSize);
        memoryValue += attrSize;
        gpuValue += attrSize;
        iter++;
        it++;
    }

    gpuRow->value = reinterpret_cast<gpudb::Value*>(gpuMemory + sizeof(gpudb::GpuRow)); // фиксим указатель на value на gpu
    cudaMemcpy(gpuMemory, memory, memsize, cudaMemcpyHostToDevice);
    delete [] memory;
    gpudb::GpuTable *table = (*tables.find(tableName)).second;
    table->rowMemSize = memsize; // TODO: убрать пересчёт

    if ( table->insertRow(reinterpret_cast<gpudb::GpuRow*>(gpuMemory)) ) {
        return true;
    }

    gpudb::gpuAllocator::getInstance().free(gpuMemory);
    return false;
}

bool DataBase::showTable(std::string tableName) {
    auto it2 = tables.find(tableName);
    if (it2 == tables.end()) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Table \"%s\" is not exist", tableName.c_str());
        return false;
    }

    gpudb::GpuTable const *gputable = (*it2).second;
    uint8_t *memory = new uint8_t[gputable->rowMemSize];
    if (!memory) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough gpu memory ");
        return false;
    }

    uint64_t it = 0;
    for (gpudb::GpuRow *v : gputable->rows) {
        cudaMemcpy(memory, v, gputable->rowMemSize, cudaMemcpyDeviceToHost);
        gpudb::GpuRow *cpu = reinterpret_cast<gpudb::GpuRow *>(memory);
        cpu->value  = reinterpret_cast<gpudb::Value *>(memory + sizeof(gpudb::GpuRow));
        uint8_t *valuePtr = memory + sizeof(gpudb::GpuRow) + sizeof(gpudb::Value) * cpu->valueSize;
        printf("%-5zu: ", it);
        for (int i = 0; i < cpu->valueSize; i++) {
            cpu->value[i].value = valuePtr;
            printf("%-10s: ", typeToString(gputable->columnsCPU[i].type).c_str());
            if (cpu->value[i].isNull == false) {
                switch (gputable->columnsCPU[i].type) {
                    case Type::STRING:
                        printf("\"%s \" ", (char*)cpu->value[i].value);
                        break;
                    case Type::REAL:
                        printf("%f ", *(double*)cpu->value[i].value);
                        break;
                    case Type::INT:
                        printf("%d ", *(int*)cpu->value[i].value);
                        break;
                    default:
                        break;
                }
            } else {
                printf("%s ", "NULL");
            }
            valuePtr += typeSize(gputable->columnsCPU[i].type);
        }
        printf("\n");
        it++;
    }


    delete [] memory;
    return true;
}

