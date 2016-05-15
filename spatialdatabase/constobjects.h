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
        float2 getPoint(uint id) const;

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
        float2 getPoint(uint id) const;

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
        CRow getRow(uint id) const;

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
        SpatialType getSpatialKeyType() const;

        FUNC_PREFIX
        TemporalType getTemporalKeyType() const;

        FUNC_PREFIX
        char const *getSpatialKeyName() const;

        FUNC_PREFIX
        char const  *getTransactionKeyName() const;

        FUNC_PREFIX
        uint getColumnSize() const;

        FUNC_PREFIX
        Type getColumnType(uint id) const;

        FUNC_PREFIX
        bool getColumnIsNull(uint id) const;

        FUNC_PREFIX
        char const *getColumnName(uint id) const;

        FUNC_PREFIX
        int64_t getColumnINT(uint id) const;

        FUNC_PREFIX
        char const *getColumnSTRING(uint id) const;

        FUNC_PREFIX
        double getColumnREAL(uint id) const;

        FUNC_PREFIX
        CGpuSet getColumnSET(uint id) const;

        FUNC_PREFIX
        Date getKeyValidTimeStart() const;

        FUNC_PREFIX
        Date getKeyValidTimeEnd() const;

        FUNC_PREFIX
        Date getKeyTransationType() const;

        FUNC_PREFIX
        CGpuPoint getKeyGpuPoint() const;

        FUNC_PREFIX
        CGpuPolygon getKeyGpuPolygon() const;

        FUNC_PREFIX
        CGpuLine getKeyGpuLine() const;
    private:
        gpudb::GpuRow *row;
        GpuColumnAttribute *columns;
        uint numColumns;
    };

}
