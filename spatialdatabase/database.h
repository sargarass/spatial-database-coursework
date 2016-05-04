#pragma once
#include "gpudb.h"
#include "stackallocator.h"
#include "types.h"
#include "tabledescription.h"
#include "row.h"
#include "temptable.h"
#include "constobjects.h"

typedef bool(*Predicate)(gpudb::CRow const &);

class DataBase : public Singleton {
    typedef std::pair<std::string, TableDescription> tablesTypePair;
    typedef std::pair<std::string, gpudb::GpuTable*> tablesPair;
public:
    bool insertRow(std::string tableName, Row row);
    bool createTable(TableDescription table);
    bool showTable(std::string tableName);
    bool showTable(gpudb::GpuTable const &table, const TableDescription &description, uint tabs =0);
    bool selectTable(std::string tableName, TempTable &table);

    bool update(std::string tableName, std::set<Attribute> const &atrSet, Predicate p);
    bool dropRow(std::string tableName, Predicate p);
    bool dropTable(std::string tableName);

    static DataBase &getInstance() {
        static DataBase *db = new DataBase;
        static bool init = false;
        if (init == false) {
            init = true;
            SingletonFactory::getInstance().registration<DataBase>(db);
            dynamic_cast<Singleton*>(db)->dependOn(Log::getInstance());
            dynamic_cast<Singleton*>(db)->dependOn(gpudb::gpuAllocator::getInstance());
            dynamic_cast<Singleton*>(db)->dependOn(gpudb::GpuStackAllocator::getInstance());
        }
        return *db;
    }

    void deinit() {
        for (auto& v : tables) {
            delete (v.second);
        }
        tables.clear();
        tablesType.clear();
    }

    virtual ~DataBase() {
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "delete DataBase");
        deinit();
    }
public:
    static void storeGPU(gpudb::GpuRow *dst, gpudb::GpuRow *src, uint64_t memsize);
    static void loadCPU(gpudb::GpuRow *dstCPU, gpudb::GpuRow *srcGPU, uint64_t memsize);
    static bool copyTempTable(TableDescription const &description, gpudb::GpuTable const *gpuTable, TempTable &table);
    static void myprintf(uint tabs, char *format ...);
    DataBase(){
        gpudb::GpuStackAllocator::getInstance().resize(512ULL * 1024ULL * 1024ULL);
        StackAllocator::getInstance().resize(1024ULL * 1024ULL * 1024ULL);
    }
    std::map<std::string, gpudb::GpuTable*> tables;
    std::map<std::string, TableDescription> tablesType;
    friend bool pointxpointKnearestNeighbor(TempTable const &a, TempTable const &b, uint k, TempTable &result);
    friend class TempTable;
};

template<typename T1, typename T2, typename T3> FUNC_PREFIX
T1 *newAddress(T1 *old, T2 *oldMemory, T3 *newMemory) {
    uintptr_t step1 =  reinterpret_cast<uintptr_t>(old);
    step1 -= reinterpret_cast<uintptr_t>(oldMemory);
    step1 += reinterpret_cast<uintptr_t>(newMemory);
    return reinterpret_cast<T1*>(step1);
}

bool pointxpointKnearestNeighbor(TempTable const &a, TempTable const &b, uint k, TempTable &result);
