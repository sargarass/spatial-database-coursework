#include "constobjects.h"

gpudb::CRow::CRow(gpudb::GpuRow *row, GpuColumnAttribute *columns, uint numColumns) {
    this->row = row;
    this->columns = columns;
    this->numColumns = numColumns;
}

SpatialType  gpudb::CRow::getSpatialKeyType() const {
    return row->spatialPart.type;
}

TemporalType gpudb::CRow::getTemporalType() const {
    return row->temporalPart.type;
}

char const  *gpudb::CRow::getSpatialKeyName() const {
    return row->spatialPart.name;
}

char const  *gpudb::CRow::getTransactionName() const {
    return row->temporalPart.name;
}

bool gpudb::CRow::getColumnType(uint id, Type &t) const {
    if (id >= numColumns) {
        return false;
    }

    t = columns[id].type;
    return true;
}

bool gpudb::CRow::getColumnIsNull(uint id, bool &isNull) const {
    if (id >= numColumns) {
        return false;
    }

    isNull = this->row->value[id].isNull;
    return true;
}

bool gpudb::CRow::getColumnName(uint id, char const * &name) const {
    if (id >= numColumns) {
        return false;
    }
    name = columns[id].name;
    return true;
}

bool gpudb::CRow::getColumnINT(uint id, int64_t &val) const {
    if (id >= numColumns) {
        return false;
    }
    if (columns[id].type != Type::INT) {
        return false;
    }
    val = *(int64_t*)(this->row->value[id].value);

    return true;
}

bool gpudb::CRow::getColumnSTRING(uint id, char const * &val) const {
    if (id >= numColumns) {
        return false;
    }
    if (columns[id].type != Type::STRING) {
        return false;
    }
    val = (char*)(this->row->value[id].value);
    return true;
}

bool gpudb::CRow::getColumnREAL(uint id, double &val) const {
    if (id >= numColumns) {
        return false;
    }
    if (columns[id].type != Type::REAL) {
        return false;
    }
    val = *(double*)(this->row->value[id].value);
    return true;
}

bool gpudb::CRow::getColumnSET(uint id, CGpuSET &set) const {
    if (id >= numColumns) {
        return false;
    }
    if (columns[id].type != Type::SET) {
        return false;
    }

    //set = CGpuSET()
    return false;
}

bool gpudb::CRow::getKeyValidTime(Date &start, Date &end) const {
    if (this->row->temporalPart.type == TemporalType::VALID_TIME || this->row->temporalPart.type == TemporalType::BITEMPORAL_TIME) {
        start.setFromCode(this->row->temporalPart.validTimeSCode);
        end.setFromCode(this->row->temporalPart.validTimeECode);
        return true;
    }
    return false;
}

bool gpudb::CRow::getKeyTransationType(Date &d) const {
    if (this->row->temporalPart.type == TemporalType::TRANSACTION_TIME || this->row->temporalPart.type == TemporalType::BITEMPORAL_TIME) {
        d.setFromCode(this->row->temporalPart.transactionTimeCode);
        return true;
    }
    return false;
}

bool gpudb::CRow::getKeyGpuPoint(CGpuPoint &point) const {
    if (this->row->spatialPart.type == SpatialType::POINT) {

    }
    return false;
}

uint gpudb::CRow::getColumnSize() const {
    return numColumns;
}

bool gpudb::CRow::getKeyGpuPolygon(CGpuPolygon &polygon) const {
    if (this->row->spatialPart.type == SpatialType::POLYGON) {

    }
    return false;
}

bool gpudb::CRow::getKeyGpuLine(CGpuLine &line) const {
    if (this->row->spatialPart.type == SpatialType::LINE) {

    }
    return false;
}
