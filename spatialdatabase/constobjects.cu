#include "constobjects.h"

gpudb::CRow::CRow() {
    row = nullptr;
    columns = nullptr;
    numColumns = 0;
}

gpudb::CRow::CRow(gpudb::GpuRow *row, GpuColumnAttribute *columns, uint numColumns) {
    this->row = row;
    this->columns = columns;
    this->numColumns = numColumns;
}

SpatialType gpudb::CRow::getSpatialKeyType() const {
    if (row == nullptr) {
        return SpatialType::UNKNOWN;
    }
    return row->spatialPart.type;
}

TemporalType gpudb::CRow::getTemporalKeyType() const {
    if (row == nullptr) {
         return TemporalType::UNKNOWN;
    }
    return row->temporalPart.type;
}

bool gpudb::CRow::getSpatialKeyName(char const  * &name) const {
    if (row == nullptr) {
        return false;
    }
    name = row->spatialPart.name;
    return true;
}

bool gpudb::CRow::getTransactionKeyName(char const  * &name) const {
    if (row == nullptr) {
        return false;
    }
    name = row->temporalPart.name;
    return true;
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
    if (id >= numColumns || columns[id].type != Type::REAL) {
        return false;
    }
    val = *(double*)(this->row->value[id].value);
    return true;
}

bool gpudb::CRow::getColumnSET(uint id, CGpuSet &set) const {
    if (this->row == nullptr || id >= numColumns || columns[id].type != Type::SET) {
        return false;
    }
    CGpuSet s((GpuSet*)this->row->value[id].value);
    set = s;
    return true;
}

bool gpudb::CRow::getKeyValidTime(Date &start, Date &end) const {
    if (this->row == nullptr) {
        return false;
    }

    if (this->row->temporalPart.type == TemporalType::VALID_TIME || this->row->temporalPart.type == TemporalType::BITEMPORAL_TIME) {
        start.setFromCode(this->row->temporalPart.validTimeSCode);
        end.setFromCode(this->row->temporalPart.validTimeECode);
        return true;
    }
    return false;
}

bool gpudb::CRow::getKeyTransationType(Date &d) const {
    if (this->row == nullptr) {
        return false;
    }

    if (this->row->temporalPart.type == TemporalType::TRANSACTION_TIME || this->row->temporalPart.type == TemporalType::BITEMPORAL_TIME) {
        d.setFromCode(this->row->temporalPart.transactionTimeCode);
        return true;
    }

    return false;
}

bool gpudb::CRow::getKeyGpuPoint(CGpuPoint &point) const {
    if (this->row->spatialPart.type != SpatialType::POINT || this->row == nullptr) {
        return false;
    }
    CGpuPoint p((GpuPoint*)this->row->spatialPart.key);
    point = p;
    return true;
}

uint gpudb::CRow::getColumnSize() const {
    return numColumns;
}

bool gpudb::CRow::getKeyGpuPolygon(CGpuPolygon &polygon) const {
    if (this->row->spatialPart.type != SpatialType::POLYGON || this->row == nullptr) {
        return false;
    }

    CGpuPolygon p((GpuPolygon*)this->row->spatialPart.key);
    polygon = p;
    return true;
}

bool gpudb::CRow::getKeyGpuLine(CGpuLine &line) const {
    if (this->row->spatialPart.type != SpatialType::LINE || this->row == nullptr) {
        return false;
    }
    CGpuLine l((GpuLine*)this->row->spatialPart.key);
    line = l;
    return true;
}


bool gpudb::CGpuPolygon::getPoint(uint id, float2 &point) const {
    if (id >= this->polygon->size || this->polygon == nullptr) {
        return false;
    }

    point = this->polygon->points[id];
    return true;
}

uint gpudb::CGpuPolygon::getPointNum() const {
    if (polygon == nullptr) {
        return 0;
    }

    return this->polygon->size;
}

float2 gpudb::CGpuPoint::getPoint() const {
    if (point == nullptr) {
        return make_float2(NAN, NAN);
    }
    return this->point->p;
}

bool gpudb::CGpuLine::getPoint(uint id, float2 &point) const {
    if (id >= this->line->size || line == nullptr) {
        return false;
    }
    point = this->line->points[id];
    return true;
}

uint gpudb::CGpuLine::getPointNum() const {
    if (line == nullptr) {
        return 0;
    }
    return this->line->size;
}

bool gpudb::CGpuSet::getRow(uint id, CRow& row) const {
    if (id >= this->set->rowsSize || set == nullptr) {
        return false;
    }

    CRow nrow(this->set->rows[id], this->set->columns, this->set->columnsSize);
    row = nrow;
    return true;
}

uint gpudb::CGpuSet::getRowNum() const {
    if (set == nullptr) {
        return 0;
    }
    return this->set->rowsSize;
}

gpudb::CGpuSet::CGpuSet(GpuSet *set) {
    this->set = set;
}

gpudb::CGpuLine::CGpuLine(GpuLine *l) {
    this->line = l;
}

gpudb::CGpuPoint::CGpuPoint(GpuPoint *p) {
    this->point = p;
}

gpudb::CGpuPolygon::CGpuPolygon(GpuPolygon *p) {
    this->polygon = p;
}

gpudb::CGpuSet::CGpuSet() {
    this->set = nullptr;
}

gpudb::CGpuLine::CGpuLine() {
    this->line = nullptr;
}

gpudb::CGpuPoint::CGpuPoint() {
    this->point = nullptr;
}

gpudb::CGpuPolygon::CGpuPolygon() {
    this->polygon = nullptr;
}
