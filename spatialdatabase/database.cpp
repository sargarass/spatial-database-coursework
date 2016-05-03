#include "database.h"
#include "gpuallocator.h"
#include <stdarg.h>

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

    std::sort(table.columnDescription.begin(),table.columnDescription.end());
    gpudb::GpuTable *gputable = new gpudb::GpuTable();

    if (gputable == nullptr) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Cannot create table: allocation falled");
        return false;
    }

    if (gputable->set(table)) {
        tablesType.insert(tablesTypePair(table.name, table));
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
        case SpatialType::POINT:
            return (points.size() == 1 && validPoint(points[0]));
            break;
        case SpatialType::LINE:
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
        case SpatialType::POLYGON:
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

void DataBase::storeGPU(gpudb::GpuRow * const dst, gpudb::GpuRow * const src, uint64_t const memsize) {
    switch (src->spatialPart.type) {
        case SpatialType::POINT:
        break;
        case SpatialType::LINE:
        {
            gpudb::GpuLine *line = ((gpudb::GpuLine*)(src->spatialPart.key));
            line->points = newAddress(line->points, src, dst);
        }
        break;
        case SpatialType::POLYGON:
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

void DataBase::loadCPU(gpudb::GpuRow *dstCPU, gpudb::GpuRow *srcGPU, uint64_t memsize) {
    cudaMemcpy(dstCPU, srcGPU, memsize, cudaMemcpyDeviceToHost);

    dstCPU->spatialPart.key = newAddress(dstCPU->spatialPart.key, srcGPU, dstCPU);
    switch (dstCPU->spatialPart.type) {
        case SpatialType::POINT:
        break;
        case SpatialType::LINE:
        {
            gpudb::GpuLine *line = ((gpudb::GpuLine*)(dstCPU->spatialPart.key));
            line->points = newAddress(line->points, srcGPU, dstCPU);
        }
        break;
        case SpatialType::POLYGON:
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
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "spatialkey name mismatch");
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

    if (row.temporalKey.type == TemporalType::BITEMPORAL_TIME || row.temporalKey.type == TemporalType::TRANSACTION_TIME) {
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
    std::sort(row.values.begin(), row.values.end());

    TableDescription &description = (*tableType).second;
    for (size_t i = 0; i < row.values.size(); i++) {
        if (row.values[i].name != description.columnDescription[i].name
            || row.values[i].type != description.columnDescription[i].type) {
            gLogWrite(LOG_MESSAGE_TYPE::WARNING,
                                     "Cannot insert row:\n"
                                     "attribute name:\n"
                                     " wait: \"%s\"\n"
                                     " get: \"%s\"\n"
                                     "type \n"
                                     " wait: \"%s\"\n"
                                     " get: \"%s\"\n",
                                     description.columnDescription[i].name.c_str(),
                                     row.values[i].name.c_str(),
                                     typeToString(description.columnDescription[i].type).c_str(),
                                     typeToString(row.values[i].type).c_str());
            return false;
        }
    }

    uint64_t withoutKey =  sizeof(gpudb::GpuRow) +
                           sizeof(gpudb::Value) * (*tableType).second.columnDescription.size()  +
                           (*tableType).second.getRowMemoryValuesSize();
    uint64_t memsize = withoutKey;

    /// Spatial part size
    switch (tableType->second.spatialKeyType) {
        case SpatialType::POINT:
            memsize += sizeof(gpudb::GpuPoint);
            break;
        case SpatialType::LINE:
            memsize += sizeof(gpudb::GpuLine) + sizeof(float2) * row.spatialKey.points.size();
            break;
        case SpatialType::POLYGON:
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

    if (row.temporalKey.type == TemporalType::BITEMPORAL_TIME || row.temporalKey.type == TemporalType::VALID_TIME) {
        gpuRow->temporalPart.validTimeSCode = row.temporalKey.validTimeS.codeDate();
        gpuRow->temporalPart.validTimeECode = row.temporalKey.validTimeE.codeDate();
    }

    if (row.temporalKey.type == TemporalType::BITEMPORAL_TIME || row.temporalKey.type == TemporalType::TRANSACTION_TIME) {
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
        case SpatialType::POINT:
        {
            gpudb::GpuPoint *point = ((gpudb::GpuPoint*)(gpuRow->spatialPart.key));
            point->p = row.spatialKey.points[0];
        }
        break;
        case SpatialType::LINE:
        {
            gpudb::GpuLine *line = ((gpudb::GpuLine*)(gpuRow->spatialPart.key));
            line->points = reinterpret_cast<float2*>(memory + withoutKey + sizeof(gpudb::GpuLine));
            line->size = row.spatialKey.points.size();
            for (int i = 0; i < row.spatialKey.points.size(); i++) {
                line->points[i] = row.spatialKey.points[i];
            }
        }
        break;
        case SpatialType::POLYGON:
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

    for (size_t i = 0; i < row.values.size(); i++) {
        gpuRow->value[i].isNull = row.values[i].isNull;
        gpuRow->value[i].value = memoryValue;
        uint64_t attrSize = typeSize(row.values[i].type);
        memcpy(memoryValue, row.values[i].value, attrSize);
        memoryValue += attrSize;
    }

    storeGPU(reinterpret_cast<gpudb::GpuRow*>(gpuMemory), gpuRow, memsize);
    StackAllocator::getInstance().popPosition();
    gpudb::GpuTable *table = (*tables.find(tableName)).second;
    if ( table->insertRow(reinterpret_cast<gpudb::GpuRow*>(gpuMemory), memsize) ) {
        return true;
    }

    gpudb::gpuAllocator::getInstance().free(gpuMemory);
    return false;
}

void DataBase::myprintf(uint tabs, char *format, ...) {
    va_list arglist;
    char str[256];
    int i = 0;
    for (; i < std::min(tabs, 255U); i++) {
        str[i] = ' ';
    }
    str[i] = '\0';
    printf(str);
    va_start( arglist, format );
    vprintf(format, arglist);
    va_end( arglist );
}

bool DataBase::showTable(gpudb::GpuTable const &table, TableDescription const &description, uint tabs) {
    uint64_t rowNum = 0;
    thrust::host_vector<gpudb::GpuRow*> rows = table.rows;
    for (size_t i = 0; i < table.rowsSize.size(); i++) {
        uint64_t memsize = table.rowsSize[i];
        uint8_t *memory = StackAllocator::getInstance().alloc<uint8_t>(memsize);
        if (!memory) {
            gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough memory ");
            return false;
        }

        loadCPU(reinterpret_cast<gpudb::GpuRow*>(memory), rows[i], memsize);
        myprintf(tabs, "%zu: ", rowNum);
        myprintf(tabs, " Key {");
        gpudb::GpuRow *cpu = reinterpret_cast<gpudb::GpuRow *>(memory);
        switch (cpu->spatialPart.type) {
            case SpatialType::POINT:
            {
                gpudb::GpuPoint *point = reinterpret_cast<gpudb::GpuPoint*>(cpu->spatialPart.key);
                myprintf(tabs, "[%s : Point : {%f, %f}], ", description.spatialKeyName.c_str(), point->p.x, point->p.y);
            }
            break;
            case SpatialType::LINE:
            {
                gpudb::GpuLine *line = reinterpret_cast<gpudb::GpuLine*>(cpu->spatialPart.key);
                myprintf(tabs, "[%s : Line : { ", description.spatialKeyName.c_str());
                for (int i = 0; i < line->size; i++) {
                    myprintf(tabs, "{%f, %f} ", line->points[i].x, line->points[i].y);
                }
                myprintf(tabs, "}], ");
            }
            break;
            case SpatialType::POLYGON:
            {
                gpudb::GpuPolygon *polygon = reinterpret_cast<gpudb::GpuPolygon*>(cpu->spatialPart.key);
                myprintf(tabs, "[%s : Polygon : { ", description.spatialKeyName.c_str());
                for (int i = 0; i < polygon->size; i++) {
                    myprintf(tabs, "{%f, %f} ", polygon->points[i].x, polygon->points[i].y);
                }
                myprintf(tabs, "}], ");
            }
            break;
        }

        if (cpu->temporalPart.type == TemporalType::BITEMPORAL_TIME || cpu->temporalPart.type == TemporalType::VALID_TIME) {
            Date validTimeS, validTimeE;
            validTimeS.setFromCode(cpu->temporalPart.validTimeSCode);
            validTimeE.setFromCode(cpu->temporalPart.validTimeECode);
            myprintf(tabs, "[%s : Valid Time {%s - %s}]", description.temporalKeyName.c_str(), validTimeS.toString().c_str(), validTimeE.toString().c_str());

        }

        if (cpu->temporalPart.type == TemporalType::BITEMPORAL_TIME) {
            myprintf(tabs, ", ");
        }

        if (cpu->temporalPart.type == TemporalType::BITEMPORAL_TIME || cpu->temporalPart.type == TemporalType::TRANSACTION_TIME) {
            Date transactionTime;
            transactionTime.setFromCode(cpu->temporalPart.transactionTimeCode);
            myprintf(tabs, "[Transaction Time : {%s}]", transactionTime.toString().c_str());
        }
        myprintf(tabs, "} ");
        myprintf(tabs, "Value {");
        fflush(stdout);
        auto iter = description.columnDescription.begin();
        for (size_t i = 0; i < cpu->valueSize; i++) {
            myprintf(tabs, "[%s : %s : ", iter->name.c_str(), typeToString(description.columnDescription[i].type).c_str());
            if (cpu->value[i].isNull == false) {
                switch (description.columnDescription[i].type) {
                    case Type::STRING:
                        myprintf(tabs, "\"%s\"", (char*)cpu->value[i].value);
                        break;
                    case Type::REAL:
                        myprintf(tabs, "%f", *(double*)cpu->value[i].value);
                        break;
                    case Type::INT:
                        myprintf(tabs, "%d", *(int*)cpu->value[i].value);
                        break;
                    case Type::DATE_TYPE:
                    {
                        Date *ptr= (Date*)cpu->value[i].value;
                        myprintf(tabs, "%s", ptr->toString().c_str());
                    }
                    case Type::SET:
                    {
                        myprintf(tabs, "\n");
                        TempTable *ptr= *(TempTable**)cpu->value[i].value;
                        showTable(*ptr->table, ptr->description, tabs + 2);
                    }
                    break;
                    default:
                        break;
                }
            } else {
                myprintf(tabs, "%s", "NULL");
            }
            myprintf(tabs, "] ");
            if (i + 1 < cpu->valueSize) {
                myprintf(tabs, ", ");
            }
            iter++;
        }
        myprintf(tabs, "}\n");
        rowNum++;
        StackAllocator::getInstance().free(memory);
    }
    return true;
}

bool DataBase::showTable(std::string tableName) {
    auto it2 = tables.find(tableName);
    auto it = tablesType.find(tableName);
    if (it2 == tables.end() || it == tablesType.end()) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Table \"%s\" is not exist", tableName.c_str());
        return false;
    }

    gpudb::GpuTable const *gputable = (*it2).second;

    return showTable(*gputable, it->second);
}
