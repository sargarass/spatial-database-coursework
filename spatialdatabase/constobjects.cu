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
    return row->spatialPart.type;
}

TemporalType gpudb::CRow::getTemporalKeyType() const {
    return row->temporalPart.type;
}

const char *gpudb::CRow::getSpatialKeyName() const {
    if (row == nullptr) {
        return "(null)";
    }
    return row->spatialPart.name;
}

const char *gpudb::CRow::getTransactionKeyName() const {
    if (row == nullptr) {
        return "(null)";
    }
    return row->temporalPart.name;
}

Type gpudb::CRow::getColumnType(uint id) const {
    return columns[id].type;
}

bool gpudb::CRow::getColumnIsNull(uint id) const {
    return this->row->value[id].isNull;
}

const char *gpudb::CRow::getColumnName(uint id) const {
    if (id >= numColumns) {
        return "(null)";
    }
    return columns[id].name;
}

int64_t gpudb::CRow::getColumnINT(uint id) const {
    return *(int64_t*)(this->row->value[id].value);
}

char const *gpudb::CRow::getColumnSTRING(uint id) const {
    return (char*)(this->row->value[id].value);
}

double gpudb::CRow::getColumnREAL(uint id) const {
    return *(double*)(this->row->value[id].value);
}

gpudb::CGpuSet gpudb::CRow::getColumnSET(uint id) const {
    CGpuSet s((GpuSet*)this->row->value[id].value);
    return s;
}

Date gpudb::CRow::getKeyValidTimeStart() const {
    Date start;
    start.setFromCode(this->row->temporalPart.validTimeSCode);
    return start;
}

Date gpudb::CRow::getKeyValidTimeEnd() const {
    Date end;
    end.setFromCode(this->row->temporalPart.validTimeECode);
    return end;
}

Date gpudb::CRow::getKeyTransationType() const {
    Date t;
    t.setFromCode(row->temporalPart.transactionTimeCode);
    return t;
}

gpudb::CGpuPoint gpudb::CRow::getKeyGpuPoint() const {
    CGpuPoint p((GpuPoint*)this->row->spatialPart.key);
    return p;
}

uint gpudb::CRow::getColumnSize() const {
    return numColumns;
}

gpudb::CGpuPolygon gpudb::CRow::getKeyGpuPolygon() const {
    CGpuPolygon p((GpuPolygon*)this->row->spatialPart.key);
    return p;
}

gpudb::CGpuLine gpudb::CRow::getKeyGpuLine() const {
    CGpuLine l((GpuLine*)this->row->spatialPart.key);
    return l;
}


float2 gpudb::CGpuPolygon::getPoint(uint id) const {
    return this->polygon->points[id];
}

uint gpudb::CGpuPolygon::getPointNum() const {
    if (polygon == nullptr) {
        return 0;
    }
    return this->polygon->size;
}

float2 gpudb::CGpuPoint::getPoint() const {
    return this->point->p;
}

float2 gpudb::CGpuLine::getPoint(uint id) const {
    return this->line->points[id];
}

uint gpudb::CGpuLine::getPointNum() const {
    return this->line->size;
}

gpudb::CRow gpudb::CGpuSet::getRow(uint id) const {
    CRow nrow(this->set->rows[id], this->set->columns, this->set->columnsSize);
    return nrow;
}

uint gpudb::CGpuSet::getRowNum() const {
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
