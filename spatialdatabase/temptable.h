#pragma once
#include "gpudb.h"
#include "row.h"

struct  __align__(16) TempTable {
    friend class DataBase;
    friend bool pointxpointKnearestNeighbor(TempTable const &a, TempTable &b, uint k, TempTable &result);
public:
    SpatialType getSpatialKeyType() const;
    TemporalType getTemporalType() const;
    std::vector<Row> selectByKey(/* INTERSECTION / FULL INSIDE / EXACTLY */) const;

    ~TempTable();
    TempTable();
    bool isValid() const;
    bool showTable();
public:
    std::list<TempTable *> references;
    std::list<TempTable *> parents;
    std::list<uintptr_t> needToBeFree;
    TempTable(TempTable const &table);
    gpudb::GpuTable *table;
    TableDescription description;
    bool valid;
};
