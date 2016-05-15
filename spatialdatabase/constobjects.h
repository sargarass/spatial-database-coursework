#pragma once
#include "gpudb.h"
namespace gpudb {
    class CRow;

    class CGpuPolygon {
    public:
        FUNC_PREFIX
        CGpuPolygon();
        FUNC_PREFIX
        CGpuPolygon(GpuPolygon *p);
        FUNC_PREFIX
        bool getPoint(uint id, float2 &point) const;
        FUNC_PREFIX
        uint getPointNum() const;
    private:
        GpuPolygon *polygon;
    };

    class CGpuPoint {
    public:
        FUNC_PREFIX
        CGpuPoint();
        FUNC_PREFIX
        CGpuPoint(GpuPoint *p);
        FUNC_PREFIX
        float2 getPoint() const;
    private:
        GpuPoint *point;
    };

    class CGpuLine {
    public:
        FUNC_PREFIX
        CGpuLine();
        FUNC_PREFIX
        CGpuLine (GpuLine *l);
        FUNC_PREFIX
        bool getPoint(uint id, float2 &point) const;
        FUNC_PREFIX
        uint getPointNum() const;
    private:
        GpuLine *line;
    };

    class CGpuSet {
    public:
        FUNC_PREFIX
        CGpuSet();
        FUNC_PREFIX
        CGpuSet(GpuSet *set);
        FUNC_PREFIX
        bool getRow(uint id, CRow &row) const;
        FUNC_PREFIX
        uint getRowNum() const;
    private:
        GpuSet *set;
    };

    class CRow {
    public:
        FUNC_PREFIX
        CRow();
        FUNC_PREFIX
        CRow(gpudb::GpuRow *row, GpuColumnAttribute *columns, uint numColumns);
        FUNC_PREFIX
        SpatialType getSpatialKeyType() const ;
        FUNC_PREFIX
        TemporalType getTemporalKeyType() const ;
        FUNC_PREFIX
        bool getSpatialKeyName(char const  * &name) const ;
        FUNC_PREFIX
        bool getTransactionKeyName(char const  * &name) const ;
        FUNC_PREFIX
        uint getColumnSize() const;
        FUNC_PREFIX
        bool getColumnType(uint id, Type &t) const ;
        FUNC_PREFIX
        bool getColumnIsNull(uint id, bool &isNull) const ;
        FUNC_PREFIX
        bool getColumnName(uint id, char const * &name) const ;
        FUNC_PREFIX
        bool getColumnINT(uint id, int64_t &val) const ;
        FUNC_PREFIX
        bool getColumnSTRING(uint id, char const * &val) const ;
        FUNC_PREFIX
        bool getColumnREAL(uint id, double &val) const ;
        FUNC_PREFIX
        bool getColumnSET(uint id, CGpuSet &set) const ;
        FUNC_PREFIX
        bool getKeyValidTime(Date &start, Date &end) const ;
        FUNC_PREFIX
        bool getKeyTransationType(Date &d) const ;
        FUNC_PREFIX
        bool getKeyGpuPoint(CGpuPoint &point) const ;
        FUNC_PREFIX
        bool getKeyGpuPolygon(CGpuPolygon &polygon) const ;
        FUNC_PREFIX
        bool getKeyGpuLine(CGpuLine &line) const ;
    private:
        gpudb::GpuRow *row;
        GpuColumnAttribute *columns;
        uint numColumns;
    };

}
