#pragma once
#include "gpudb.h"
#include "row.h"

struct  __align__(16) TempTable {
    friend class DataBase;
public:
    SpatialType getSpatialKeyType() const;
    TemporalType getTemporalType() const;

    ~TempTable();
    TempTable();

    TempTable(TempTable &table);
    void operator=(TempTable &t);
    bool isValid() const;
    void deinit();
private:
    TempTable(TempTable const &table){UNUSED_PARAM_HANDLER(table);}
    void operator=(TempTable const &t){ UNUSED_PARAM_HANDLER(t);}
    std::list<TempTable *> references;
    std::list<TempTable *> parents;
    std::list<uintptr_t> needToBeFree;
    gpudb::GpuTable *table;
    TableDescription description;
    bool valid;
};
