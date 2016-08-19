#pragma once
#include "gpudb.h"
#include "row.h"

class __align__(16) TempTable {
    friend class DataBase;
public:
    SpatialType getSpatialKeyType() const;
    TemporalType getTemporalType() const;

    ~TempTable();
    TempTable();

    bool isValid() const;
    TempTable(TempTable const &table) = delete;
    void operator=(TempTable const &t) = delete;
    void operator=(TempTable &&t) = delete;
    TempTable(TempTable &&table) = delete;
protected:
    void deinit();
    std::list<TempTable *> references;
    std::list<TempTable *> parents;
    std::list<TempTable *> insideAllocations;
    gpudb::GpuTable *table;
    TableDescription description;
    bool valid;
};

namespace std {
    template<>
    class default_delete<TempTable> {
    public:
        default_delete() { bdelete = true;}

        void operator()(TempTable *table) noexcept {
            if (bdelete) {
                delete table;
            }
        }
        bool bdelete;
    };
}
