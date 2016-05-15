#include "database.h"
#include "gpuallocator.h"
#include <stdarg.h>

DataBase::DataBase() {
    gpudb::GpuStackAllocator::getInstance().resize(512ULL * 1024ULL * 1024ULL);
    StackAllocator::getInstance().resize(1024ULL * 1024ULL * 1024ULL);
}

DataBase &DataBase::getInstance() {
    static DataBase *db = new DataBase;
    static bool init = false;
    if (init == false) {
        init = true;
        SingletonFactory::getInstance().registration<DataBase>(db);
        dynamic_cast<Singleton*>(db)->dependOn(Log::getInstance());
        dynamic_cast<Singleton*>(db)->dependOn(gpudb::gpuAllocator::getInstance());
        dynamic_cast<Singleton*>(db)->dependOn(gpudb::GpuStackAllocator::getInstance());
    }
    return *db;
}

DataBase::~DataBase() {
    gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "delete DataBase");
    deinit();
}

void DataBase::deinit() {
    for (auto& v : tables) {
        delete (v.second);
    }
    tables.clear();
    tablesType.clear();
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
            for (uint i = 0; i < points.size(); i++) {
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

void DataBase::store(gpudb::GpuRow * const dst, gpudb::GpuRow * const src) {
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
}

void DataBase::storeGPU(gpudb::GpuRow * const dst, gpudb::GpuRow * const src, uint64_t const memsize) {
    store(dst, src);
    cudaMemcpy(dst, src, memsize, cudaMemcpyHostToDevice);
}

void DataBase::load(gpudb::GpuRow *dst, gpudb::GpuRow *src) {
    dst->spatialPart.key = newAddress(dst->spatialPart.key, src, dst);
    switch (dst->spatialPart.type) {
        case SpatialType::POINT:
        break;
        case SpatialType::LINE:
        {
            gpudb::GpuLine *line = ((gpudb::GpuLine*)(dst->spatialPart.key));
            line->points = newAddress(line->points, src, dst);
        }
        break;
        case SpatialType::POLYGON:
        {
            gpudb::GpuPolygon *polygon = ((gpudb::GpuPolygon*)(dst->spatialPart.key));
            polygon->points = newAddress(polygon->points, src, dst);
        }
        break;
    }

    dst->value = newAddress(dst->value, src, dst);

    for (uint i = 0; i < dst->valueSize; i++) {
        dst->value[i].value = newAddress(dst->value[i].value, src, dst);
    }
}

void DataBase::loadCPU(gpudb::GpuRow *dstCPU, gpudb::GpuRow *srcGPU, uint64_t memsize) {
    cudaMemcpy(dstCPU, srcGPU, memsize, cudaMemcpyDeviceToHost);
    load(dstCPU, srcGPU);
}

bool DataBase::validateRow(Row &row, TableDescription &desc) {
    if (row.spatialKey.type != desc.spatialKeyType) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "spatialkey type mismatch: \n"
                                             " wait: %s \n"
                                             " get: %s \n",
                                             typeToString(desc.spatialKeyType).c_str(),
                                             typeToString(row.spatialKey.type).c_str());
        return false;
    }

    if (row.spatialKey.name != desc.spatialKeyName) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "spatialkey name mismatch");
        return false;
    }

    if (row.temporalKey.type != desc.temporalKeyType) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "temporalkey type mismatch");
        return false;
    }

    if (row.temporalKey.name != desc.temporalKeyName) {
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "temporalkey name mismatch");
        return false;
    }

    if (row.values.size() != desc.columnDescription.size()) {
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
                                 " temporalkey invalid \"%s\" \"%s\"", row.temporalKey.validTimeS.toString().c_str(), row.temporalKey.validTimeE.toString().c_str());
        return false;
    }

    std::sort(row.values.begin(), row.values.end());

    for (size_t i = 0; i < row.values.size(); i++) {
        if (row.values[i].name != desc.columnDescription[i].name
            || row.values[i].type != desc.columnDescription[i].type) {
            gLogWrite(LOG_MESSAGE_TYPE::WARNING,
                                     "Cannot insert row:\n"
                                     "attribute name:\n"
                                     " wait: \"%s\"\n"
                                     " get: \"%s\"\n"
                                     "type \n"
                                     " wait: \"%s\"\n"
                                     " get: \"%s\"\n",
                                     desc.columnDescription[i].name.c_str(),
                                     row.values[i].name.c_str(),
                                     typeToString(desc.columnDescription[i].type).c_str(),
                                     typeToString(row.values[i].type).c_str());
            return false;
        }
    }
    return true;
}

gpudb::GpuRow *DataBase::allocateRow(Row &row, TableDescription &desc, uint64_t &growMemSize) {
    uint64_t withoutKey =  sizeof(gpudb::GpuRow) +
                           sizeof(gpudb::Value) * desc.columnDescription.size()  +
                           desc.getRowMemoryValuesSize();
    uint64_t memsize = withoutKey;

    /// Spatial part size
    switch (desc.spatialKeyType) {
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
            return nullptr;
            break;
    }
    ///

    StackAllocator::getInstance().pushPosition();
    uint8_t *memory = StackAllocator::getInstance().alloc<uint8_t>(memsize);
    if (memory == nullptr) {
        StackAllocator::getInstance().popPosition();
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough cpu memory ");
        return nullptr;
    }

    gpudb::GpuRow *gpuRow = reinterpret_cast<gpudb::GpuRow *>(memory);
    gpuRow->value = reinterpret_cast<gpudb::Value*>(memory + sizeof(gpudb::GpuRow));
    gpuRow->valueSize = desc.columnDescription.size();
    gpuRow->temporalPart.type = row.temporalKey.type;

    if (row.temporalKey.type == TemporalType::BITEMPORAL_TIME || row.temporalKey.type == TemporalType::VALID_TIME) {
        gpuRow->temporalPart.validTimeSCode = row.temporalKey.validTimeS.codeDate();
        gpuRow->temporalPart.validTimeECode = row.temporalKey.validTimeE.codeDate();
    }

    if (row.temporalKey.type == TemporalType::BITEMPORAL_TIME || row.temporalKey.type == TemporalType::TRANSACTION_TIME) {
        gpuRow->temporalPart.transactionTimeCode = row.temporalKey.transactionTime.codeDate();
    }

    strncpy(gpuRow->temporalPart.name, desc.temporalKeyName.c_str(), typeSize(Type::STRING));
    gpuRow->temporalPart.name[typeSize(Type::STRING) - 1] = 0;

    strncpy(gpuRow->spatialPart.name, desc.spatialKeyName.c_str(), typeSize(Type::STRING));
    gpuRow->spatialPart.name[typeSize(Type::STRING) - 1] = 0;

    uint8_t *gpuMemory = gpudb::gpuAllocator::getInstance().alloc<uint8_t>(memsize);
    uint8_t *memoryValue = memory + sizeof(gpudb::GpuRow) + sizeof(gpudb::Value) * gpuRow->valueSize;
    if (gpuMemory  == nullptr) {
        StackAllocator::getInstance().popPosition();
        gLogWrite(LOG_MESSAGE_TYPE::WARNING, "Not enough gpu memory ");
        growMemSize = 0;
        return nullptr;
    }

    gpuRow->spatialPart.type = row.spatialKey.type;
    gpuRow->spatialPart.key = memory + withoutKey;

    switch (desc.spatialKeyType) {
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
        gpuRow->value[i].isNull = row.values[i].isNullVal;
        gpuRow->value[i].value = memoryValue;
        uint64_t attrSize = typeSize(row.values[i].type);
        memcpy(memoryValue, row.values[i].value, attrSize);
        memoryValue += attrSize;
    }

    storeGPU(reinterpret_cast<gpudb::GpuRow*>(gpuMemory), gpuRow, memsize);
    StackAllocator::getInstance().popPosition();
    growMemSize = memsize;
    return reinterpret_cast<gpudb::GpuRow*>(gpuMemory);
}

bool DataBase::insertRow(std::string tableName, Row &row) {
    auto tableType = tablesType.find(tableName);
    if (tableType == tablesType.end()) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Table \"%s\" is not exist", tableName.c_str());
        return false;
    }

    TableDescription &desc = tableType->second;

    if (!validateRow(row, desc)) {
        return false;
    }
    uint64_t grow_mem_size = 0;
    gpudb::GpuRow *grow = allocateRow(row, desc, grow_mem_size);
    if (grow == nullptr) {
        return false;
    }

    gpudb::GpuTable *table = (*tables.find(tableName)).second;
    if ( table->insertRow(desc, grow, grow_mem_size) ) {
        return true;
    }

    gpudb::gpuAllocator::getInstance().free(grow);
    return false;
}

void DataBase::myprintf(uint tabs, char *format, ...) {
    va_list arglist;
    char str[256];
    int i = 0;
    for (; i < std::min(tabs, 255U); i++) {
        str[i] = ' ';
    }
    str[i] = 0;
    printf(str);
    va_start( arglist, format );
    vprintf(format, arglist);
    va_end( arglist );
}

bool DataBase::showTableImp(gpudb::GpuTable const &table, TableDescription const &description, uint tabs) {
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
                myprintf(tabs, "[%s : Point : {%f, %f}], ", cpu->spatialPart.name, point->p.x, point->p.y);
            }
            break;
            case SpatialType::LINE:
            {
                gpudb::GpuLine *line = reinterpret_cast<gpudb::GpuLine*>(cpu->spatialPart.key);
                myprintf(tabs, "[%s : Line : { ", cpu->spatialPart.name);
                for (int i = 0; i < line->size; i++) {
                    myprintf(tabs, "{%f, %f} ", line->points[i].x, line->points[i].y);
                }
                myprintf(tabs, "}], ");
            }
            break;
            case SpatialType::POLYGON:
            {
                gpudb::GpuPolygon *polygon = reinterpret_cast<gpudb::GpuPolygon*>(cpu->spatialPart.key);
                myprintf(tabs, "[%s : Polygon : { ", cpu->spatialPart.name);
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
            myprintf(tabs, "[%s : Valid Time {%s - %s}]", cpu->temporalPart.name, validTimeS.toString().c_str(), validTimeE.toString().c_str());

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
                        TempTable *ptr= ((gpudb::GpuSet*)cpu->value[i].value)->temptable;
                        showTableImp(*ptr->table, ptr->description, tabs + 2);
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

    printf("Table \"%s\" [{\n", it->second.name.c_str());
    gpudb::GpuTable const *gputable = (*it2).second;
    bool result = showTableImp(*gputable, it->second);
    printf("}]\n_________________________________________\n");
    return result;
}

bool DataBase::showTableHeaderImp(gpudb::GpuTable const &table, TableDescription const &description) {
    for (int i = 0; i < description.columnDescription.size(); i++) {
        printf("col %d : name = \"%s\" : type = %s\n", i + 1, description.columnDescription[i].name.c_str(), typeToString(description.columnDescription[i].type).c_str());
    }
}

bool DataBase::showTableHeader(std::string tableName) {
    auto it2 = tables.find(tableName);
    auto it = tablesType.find(tableName);

    if (it2 == tables.end() || it == tablesType.end()) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "Table \"%s\" is not exist", tableName.c_str());
        return false;
    }

    printf("Table \"%s\" [{\n", it->second.name.c_str());
    gpudb::GpuTable const *gputable = (*it2).second;
    bool result = showTableHeaderImp(*gputable, it->second);
    printf("}]\n_________________________________________\n");
    return result;
}

bool DataBase::showTableHeader(TempTable &t) {
    if (t.table == nullptr || t.isValid() == false) {
        return false;
    }
    printf("TempTable \"%s\" [{\n", t.description.name.c_str());
    bool result = DataBase::getInstance().showTableImp(*t.table, t.description);
    printf("}]\n_________________________________________\n");
    return result;
}

bool DataBase::showTable(TempTable &t) {
    if (t.table == nullptr || t.isValid() == false) {
        return false;
    }
    printf("TempTable \"%s\" [{\n", t.description.name.c_str());
    bool result = DataBase::getInstance().showTableImp(*t.table, t.description);
    printf("}]\n_________________________________________\n");
    return result;
}

