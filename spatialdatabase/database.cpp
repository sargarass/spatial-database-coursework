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

bool validPoint(float2 p) {
    if (-180.0f <= p.x && p.x <= 180.0f
        && -90.0f <= p.y && p.y <= 90.0f) {
        return true;
    }
    return false;
}

bool SpatialKey::isValid() {
    switch (type) {
        case POINT:
            return (points.size() == 1 && validPoint(points[0]));
            break;
        case LINE:
        {
            if (points.size() < 2) {
                return false;
            }
            bool b = true;
            for (auto &p : points) {
                b = b && validPoint(p);
            }
            return b;
        }
        break;
        case POLYGON:
        {
            if (points.size() < 3) {
                return false;
            }
            bool b = true;
            // conterclockwise
            double sum = 0.0f;
            for (int i = 0; i < points.size(); i++) {
                float2 x = points[i];
                float2 y = points[(i + 1 < points.size())? i + 1 : 0];
                b = b && validPoint(x) && validPoint(y);
                sum += (y.x - x.x) * (y.y + x.x);
            }
            return b && sum < 0;
        }
        break;
    }
    return false;
}

template<typename T1, typename T2, typename T3>
T1 *newAddress(T1 *old, T2 *oldMemory, T3 *newMemory) {
    uintptr_t step1 =  reinterpret_cast<uintptr_t>(old);
    step1 -= reinterpret_cast<uintptr_t>(oldMemory);
    step1 += reinterpret_cast<uintptr_t>(newMemory);
    return reinterpret_cast<T1*>(step1);
}

void DataBase::storeGPU(gpudb::GpuRow * const dst, gpudb::GpuRow * const src, uint64_t const memsize) {
    switch (src->spatialPart.type) {
        case POINT:
        break;
        case LINE:
        {
            gpudb::GpuLine *line = ((gpudb::GpuLine*)(src->spatialPart.key));
            line->points = newAddress(line->points, src, dst);
        }
        break;
        case POLYGON:
        {
            gpudb::GpuPolygon *polygon = ((gpudb::GpuPolygon*)(src->spatialPart.key));
            polygon->points = newAddress(polygon->points, src, dst);
        }
        break;
    }
    src->spatialPart.key = newAddress(src->spatialPart.key, src, dst);

    for (uint i = 0; i < src->valueSize; i++) {
        src->value[i].value = newAddress(src->value[i].value, src, dst);
    }
    src->value = newAddress(src->value, src, dst);
    cudaMemcpy(dst, src, memsize, cudaMemcpyHostToDevice);
}

void DataBase::loadCPU(gpudb::GpuRow *dstCPU, gpudb::GpuRow *srcGPU, uint64_t &memsize) {
    if (memsize == 0 || dstCPU == nullptr) {
        cudaMemcpy(&memsize, reinterpret_cast<uint8_t*>(srcGPU) + offsetof(gpudb::GpuRow, rowSize), sizeof(uint64_t), cudaMemcpyDeviceToHost);
        return;
    }
    cudaMemcpy(dstCPU, srcGPU, memsize, cudaMemcpyDeviceToHost);

    dstCPU->spatialPart.key = newAddress(dstCPU->spatialPart.key, srcGPU, dstCPU);
    switch (dstCPU->spatialPart.type) {
        case POINT:
        break;
        case LINE:
        {
            gpudb::GpuLine *line = ((gpudb::GpuLine*)(dstCPU->spatialPart.key));
            line->points = newAddress(line->points, srcGPU, dstCPU);
        }
        break;
        case POLYGON:
        {
            gpudb::GpuPolygon *polygon = ((gpudb::GpuPolygon*)(dstCPU->spatialPart.key));
            polygon->points = newAddress(polygon->points, srcGPU, dstCPU);
        }
        break;
    }

    dstCPU->value = newAddress(dstCPU->value, srcGPU, dstCPU);

    for (uint i = 0; i < dstCPU->valueSize; i++) {
        dstCPU->value[i].value = newAddress(dstCPU->value[i].value, srcGPU, dstCPU);
    }
}


bool DataBase::insertRow(std::string tableName, Row row) {
    auto tableType = tablesType.find(tableName);
    if (tableType == tablesType.end()) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Table \"%s\" is not exist", tableName.c_str());
        return false;
    }

    if (row.spatialKey.type != (*tableType).second.spatialKeyType) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "spatialkey type mismatch");
        return false;
    }

    if (row.spatialKey.name != (*tableType).second.spatialKeyName) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "DataBase", "insertRow", "spatialkey name mismatch");
        return false;
    }

    if (row.temporalKey.type != (*tableType).second.temporalKeyType) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "temporalkey type mismatch");
        return false;
    }

    if (row.temporalKey.name != (*tableType).second.temporalKeyName) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "temporalkey name mismatch");
        return false;
    }

    if (row.values.size() != (*tableType).second.columnDescription.size()) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING,
                                 "Cannot insert row:"
                                 " columns size mismatch");
        return false;
    }

    if (row.temporalKey.type == BITEMPORAL_TIME || row.temporalKey.type == TRANSACTION_TIME) {
        row.temporalKey.transactionTime = Date::getDateFromEpoch();
    }

    if (!row.spatialKey.isValid()) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING,
                                 "Cannot insert row:"
                                 " spatialkey invalid");
        return false;
    }

    if (!row.temporalKey.isValid()) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING,
                                 "Cannot insert row:"
                                 " temporalkey invalid");
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

    uint64_t withoutKey =  sizeof(gpudb::GpuRow) +
                           sizeof(gpudb::Value) * (*tableType).second.columnDescription.size()  +
                           (*tableType).second.getRowMemoryValuesSize();
    uint64_t memsize = withoutKey;

    /// Spatial part size
    switch (tableType->second.spatialKeyType) {
        case POINT:
            memsize += sizeof(gpudb::GpuPoint);
            break;
        case LINE:
            memsize += sizeof(gpudb::GpuLine) + sizeof(float2) * row.spatialKey.points.size();
            break;
        case POLYGON:
            memsize += sizeof(gpudb::GpuPolygon) + sizeof(float2) * row.spatialKey.points.size();
            break;
        default:
            return false;
            break;
    }
    ///

    StackAllocator::getInstance().pushPosition();
    uint8_t *memory = StackAllocator::getInstance().alloc<uint8_t>(memsize);
    if (memory == nullptr) {
        StackAllocator::getInstance().popPosition();
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough cpu memory ");
        return false;
    }

    gpudb::GpuRow *gpuRow = reinterpret_cast<gpudb::GpuRow *>(memory);
    gpuRow->value = reinterpret_cast<gpudb::Value*>(memory + sizeof(gpudb::GpuRow));
    gpuRow->valueSize = (*tableType).second.columnDescription.size();
    gpuRow->temporalPart.type = row.temporalKey.type;

    if (row.temporalKey.type == BITEMPORAL_TIME || row.temporalKey.type == VALID_TIME) {
        gpuRow->temporalPart.validTimeSCode = row.temporalKey.validTimeS.codeDate();
        gpuRow->temporalPart.validTimeECode = row.temporalKey.validTimeE.codeDate();
    }

    if (row.temporalKey.type == BITEMPORAL_TIME || row.temporalKey.type == TRANSACTION_TIME) {
        gpuRow->temporalPart.transactionTimeCode = row.temporalKey.transactionTime.codeDate();
    }

    uint8_t *gpuMemory = gpudb::gpuAllocator::getInstance().alloc<uint8_t>(memsize);
    uint8_t *memoryValue = memory + sizeof(gpudb::GpuRow) + sizeof(gpudb::Value) * gpuRow->valueSize;
    if (gpuMemory  == nullptr) {
        StackAllocator::getInstance().popPosition();
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough gpu memory ");
        return false;
    }

    gpuRow->spatialPart.type = row.spatialKey.type;
    gpuRow->spatialPart.key = memory + withoutKey;

    switch (tableType->second.spatialKeyType) {
        case POINT:
        {
            gpudb::GpuPoint *point = ((gpudb::GpuPoint*)(gpuRow->spatialPart.key));
            point->p = row.spatialKey.points[0];
        }
        break;
        case LINE:
        {
            gpudb::GpuLine *line = ((gpudb::GpuLine*)(gpuRow->spatialPart.key));
            line->points = reinterpret_cast<float2*>(memory + withoutKey + sizeof(gpudb::GpuLine));
            line->size = row.spatialKey.points.size();
            for (int i = 0; i < row.spatialKey.points.size(); i++) {
                line->points[i] = row.spatialKey.points[i];
            }
        }
        break;
        case POLYGON:
        {
            gpudb::GpuPolygon *polygon = ((gpudb::GpuPolygon*)(gpuRow->spatialPart.key));
            polygon->points = reinterpret_cast<float2*>(memory + withoutKey + sizeof(gpudb::GpuPolygon));
            polygon->size = row.spatialKey.points.size();
            for (int i = 0; i < row.spatialKey.points.size(); i++) {
                polygon->points[i] = row.spatialKey.points[i];
            }
        }
        break;
    }

    iter = 0;
    for (Attribute const &v : row.values) {
        gpuRow->value[iter].isNull = v.isNull;
        gpuRow->value[iter].value = memoryValue;
        uint64_t attrSize = typeSize(v.type);
        memcpy(memoryValue, v.value, attrSize);
        memoryValue += attrSize;
        iter++;
        it++;
    }
    gpuRow->rowSize = memsize;
    storeGPU(reinterpret_cast<gpudb::GpuRow*>(gpuMemory), gpuRow, memsize);
    StackAllocator::getInstance().popPosition();
    gpudb::GpuTable *table = (*tables.find(tableName)).second;
    if ( table->insertRow(reinterpret_cast<gpudb::GpuRow*>(gpuMemory)) ) {
        return true;
    }

    gpudb::gpuAllocator::getInstance().free(gpuMemory);
    return false;
}

bool DataBase::showTable(gpudb::GpuTable const &table, TableDescription &description) {
    uint64_t rowNum = 0;
    for (gpudb::GpuRow *v : table.rows) {
        uint64_t memsize = 0;
        loadCPU(nullptr, v, memsize);
        uint8_t *memory = StackAllocator::getInstance().alloc<uint8_t>(memsize);
        if (!memory) {
            gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough memory ");
            return false;
        }
        loadCPU(reinterpret_cast<gpudb::GpuRow*>(memory), v, memsize);
        printf("%zu: ", rowNum);
        printf(" Key {");
        gpudb::GpuRow *cpu = reinterpret_cast<gpudb::GpuRow *>(memory);
        switch (cpu->spatialPart.type) {
            case POINT:
            {
                gpudb::GpuPoint *point = reinterpret_cast<gpudb::GpuPoint*>(cpu->spatialPart.key);
                printf("[%s : Point : {%f, %f}], ", description.spatialKeyName.c_str(), point->p.x, point->p.y);
            }
            break;
            case LINE:
            {
                gpudb::GpuLine *line = reinterpret_cast<gpudb::GpuLine*>(cpu->spatialPart.key);
                printf("[%s : Line : { ", description.spatialKeyName.c_str());
                for (int i = 0; i < line->size; i++) {
                    printf("{%f, %f} ", line->points[i].x, line->points[i].y);
                }
                printf("}], ");
            }
            break;
            case POLYGON:
            {
                gpudb::GpuPolygon *polygon = reinterpret_cast<gpudb::GpuPolygon*>(cpu->spatialPart.key);
                printf("[%s : Polygon : { ", description.spatialKeyName.c_str());
                for (int i = 0; i < polygon->size; i++) {
                    printf("{%f, %f} ", polygon->points[i].x, polygon->points[i].y);
                }
                printf("}], ");
            }
            break;
        }

        if (cpu->temporalPart.type == BITEMPORAL_TIME || cpu->temporalPart.type == VALID_TIME) {
            Date validTimeS, validTimeE;
            validTimeS.setFromCode(cpu->temporalPart.validTimeSCode);
            validTimeE.setFromCode(cpu->temporalPart.validTimeECode);
            printf("[%s : Valid Time {%s - %s}]", description.temporalKeyName.c_str(), validTimeS.toString().c_str(), validTimeE.toString().c_str());

        }

        if (cpu->temporalPart.type == BITEMPORAL_TIME) {
            printf(", ");
        }

        if (cpu->temporalPart.type == BITEMPORAL_TIME || cpu->temporalPart.type == TRANSACTION_TIME) {
            Date transactionTime;
            transactionTime.setFromCode(cpu->temporalPart.transactionTimeCode);
            printf("[Transaction Time : {%s}]", transactionTime.toString().c_str());
        }
        printf("} ");
        printf("Value {");
        auto iter = description.columnDescription.begin();
        for (size_t i = 0; i < cpu->valueSize; i++) {
            printf("[%s : %s : ", iter->name.c_str(), typeToString(table.columnsCPU[i].type).c_str());
            if (cpu->value[i].isNull == false) {
                switch (table.columnsCPU[i].type) {
                    case Type::STRING:
                        printf("\"%s\"", (char*)cpu->value[i].value);
                        break;
                    case Type::REAL:
                        printf("%f", *(double*)cpu->value[i].value);
                        break;
                    case Type::INT:
                        printf("%d", *(int*)cpu->value[i].value);
                        break;
                    case Type::DATE_TYPE:
                    {
                        Date *ptr= (Date*)cpu->value[i].value;
                        printf("%s", ptr->toString().c_str());
                    }
                    break;
                    default:
                        break;
                }
            } else {
                printf("%s", "NULL");
            }
            printf("] ");
            if (i + 1 < cpu->valueSize) {
                printf(", ");
            }
            iter++;
        }
        printf("}\n");
        rowNum++;
        StackAllocator::getInstance().free(memory);
    }
    return true;
}

bool DataBase::showTable(std::string tableName) {
    auto it2 = tables.find(tableName);
    auto it = tablesType.find(tableName);
    if (it2 == tables.end()) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Table \"%s\" is not exist", tableName.c_str());
        return false;
    }

    gpudb::GpuTable const *gputable = (*it2).second;

    return showTable(*gputable, it->second);
}

