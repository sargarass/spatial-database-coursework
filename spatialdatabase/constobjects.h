#pragma once
#include "gpudb.h"
namespace gpudb {

    class  CGpuPolygon {

    private:
        GpuPolygon *polygon;
    };

    class CGpuPoint {
    private:
        GpuPoint *point;
    };

    class CGpuLine {

    private:
        GpuLine *line;
    };

    class CGpuSET {

    private:
        GpuSet *set;
    };

    class CRow {
    public:
        FUNC_PREFIX
        CRow(gpudb::GpuRow *row, GpuColumnAttribute *columns, uint numColumns);
        FUNC_PREFIX
        SpatialType  getSpatialKeyType() const ;
        FUNC_PREFIX
        TemporalType getTemporalType() const ;
        FUNC_PREFIX
        char const  *getSpatialKeyName() const ;
        FUNC_PREFIX
        char const  *getTransactionName() const ;
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
        bool getColumnSET(uint id, CGpuSET &set) const ;
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
