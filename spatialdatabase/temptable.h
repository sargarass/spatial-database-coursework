#pragma once
#include "gpudb.h"
#include "row.h"

struct __align__(16) TempTable {
    friend class DataBase;
public:
    SpatialType getSpatialKeyType() const;
    TemporalType getTemporalType() const;

    ~TempTable();
    TempTable();

    bool isValid() const;
    void deinit();

    TempTable(TempTable const &table) = delete;
    void operator=(TempTable const &t) = delete;
    void operator=(TempTable &&t) = delete;
    TempTable(TempTable &&table) = delete;
protected:
    std::list<TempTable *> references;
    std::list<TempTable *> parents;
    std::list<uintptr_t> needToBeFree;
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
