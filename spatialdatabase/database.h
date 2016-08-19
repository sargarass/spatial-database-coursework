#pragma once
#include "gpudb.h"
#include "stackallocator.h"
#include "types.h"
#include "tabledescription.h"
#include "row.h"
#include "temptable.h"
#include "constobjects.h"
#include "filter.h"
#pragma pack(push, 1)
/*
 * FileDescriptor
 * TableChunk
 * ColumnsChunk[numColumns]
 * RowChunk[numRows]
 * Rows[numRows]
 * TableChunk
 * .......
 */
struct FileDescriptor {
    unsigned char sha512[SHA512_DIGEST_LENGTH];
    const uint32_t databaseId = 0x88855535;
    uint32_t tablesNum;
};

struct TableChunk {
    char tableName[NAME_MAX_LEN];
    char temporalKeyName[NAME_MAX_LEN];
    char spatialKeyName[NAME_MAX_LEN];
    SpatialType spatialKeyType;
    TemporalType temporalKeyType;
    uint32_t numRows;
    uint32_t numColumns;
};

struct ColumnsChunk {
    gpudb::GpuColumnAttribute atr;
};

struct RowChunk {
    uint32_t rowSize;
};


template<typename CPUClass>
struct RAII_GC {
    void registrCPU(CPUClass *ptr) {
        cpuMemory.push_back(ptr);
    }

    void registrGPU(void *ptr) {
        gpuMemory.push_back(ptr);
    }

    void takeGPU() {
        cpuMemory.clear();
    }

    void takeCPU() {
        gpuMemory.clear();
    }

    ~RAII_GC() {
        for (CPUClass *v : cpuMemory) {
            delete v;
        }

        for (void *v : gpuMemory) {
            gpudb::gpuAllocator::getInstance().free(v);
        }
    }

    std::list<CPUClass *> cpuMemory;
    std::list<void*> gpuMemory;
};

#pragma pack(pop)


class DataBase : public Singleton {
    typedef std::pair<std::string, TableDescription> tablesTypePair;
    typedef std::pair<std::string, gpudb::GpuTable*> tablesPair;
public:
    Result<void, Error<std::string>> loadFromDisk(std::string path);
    Result<void, Error<std::string>> saveOnDisk(std::string path);
    Result<std::vector<Row>, Error<std::string>> selectRow(std::string tableName, Predicate p);
    Result<std::vector<Row>, Error<std::string>> selectRow(std::unique_ptr<TempTable> &table, Predicate p);
    Result<std::unique_ptr<TempTable>, Error<std::string>> selectTable(std::string tableName);


    Result<void, Error<std::string>> createTable(TableDescription table);
    Result<void, Error<std::string>> dropTable(std::string tableName);

    Result<void, Error<std::string>> showTable(std::string tableName);
    Result<void, Error<std::string>> showTable(std::unique_ptr<TempTable> const &t);
    Result<void, Error<std::string>> showTableHeader(std::string tableName);
    Result<void, Error<std::string>> showTableHeader(std::unique_ptr<TempTable> const &t);

    Result<void, Error<std::string>> insertRow(std::string tableName, std::vector<Row> &rows);
    Result<void, Error<std::string>> insertRow(std::string tableName, Row &row);
    Result<void, Error<std::string>> update(std::string tableName, std::set<Attribute> const &atrSet, Predicate p);
    Result<void, Error<std::string>> update(std::unique_ptr<TempTable> &t, std::set<Attribute> const &atrSet, Predicate p);
    Result<void, Error<std::string>> dropRow(std::string tableName, Predicate p);
    Result<std::unique_ptr<TempTable>, Error<std::string>> filter(std::unique_ptr<TempTable> &t, Predicate p);

    static DataBase &getInstance();
    void deinit();
    virtual ~DataBase();
    Result<std::unique_ptr<TempTable>, Error<std::string>> pointxpointKnearestNeighbor(std::unique_ptr<TempTable> const &a, std::unique_ptr<TempTable> &b, uint k);
    Result<std::unique_ptr<TempTable>, Error<std::string>> polygonxpointPointsInPolygon(std::unique_ptr<TempTable> const &a, std::unique_ptr<TempTable> &b);
    Result<std::unique_ptr<TempTable>, Error<std::string>> linexpointPointsInBufferLine(std::unique_ptr<TempTable> const &a, std::unique_ptr<TempTable> &b, float radius);

private:
    Result<std::unique_ptr<TempTable>, Error<std::string>> resultToTempTable2(std::unique_ptr<TempTable> const &sourceA, std::unique_ptr<TempTable> &sourceB, std::string nameForNewTempTebles,
                           TempTable **newTempTables, std::string nameForResultTempTable);
    Result<std::unique_ptr<TempTable>, Error<std::string>> resultToTempTable1(std::unique_ptr<TempTable> const &a, std::unique_ptr<TempTable> &b, std::string opname, uint *selectedRowsFromB, uint *selectedRowsSize);
    Result<gpudb::GpuRow *, Error<std::string>> allocateRow(Row &row, TableDescription &desc, uint64_t &growMemSize);
    Result<void, Error<std::string>> validateRow(Row &row, TableDescription &desc);
    Result<void, Error<std::string>> hashDataBaseFile(FILE *file, FileDescriptor &desc);
    Result<std::vector<Row>, Error<std::string>> selectRowImp(TableDescription &desc, gpudb::GpuTable *table, Predicate p);
    Result<void, Error<std::string>> updateImp(TableDescription &desc, gpudb::GpuTable *table, std::set<Attribute> const &atrSet, Predicate p);
    void showTableImp(gpudb::GpuTable const &table, const TableDescription &description, uint tabs =0);
    void showTableHeaderImp(gpudb::GpuTable const &table, const TableDescription &description);
    Result<void, Error<std::string>> dropRowImp(gpudb::GpuTable *table, Predicate p);
    static void store(gpudb::GpuRow * const dst, gpudb::GpuRow * const src);
    static void load(gpudb::GpuRow * const dst, gpudb::GpuRow * const src);
    static void storeGPU(gpudb::GpuRow *dst, gpudb::GpuRow *src, uint64_t memsize);
    static void loadCPU(gpudb::GpuRow *dstCPU, gpudb::GpuRow *srcGPU, uint64_t memsize);
    static Result<std::unique_ptr<TempTable>, Error<std::string>> copyTempTable(TableDescription const &description, gpudb::GpuTable const *gpuTable);
    static void myprintf(uint tabs, const char *format ...);
    DataBase();
    std::map<std::string, gpudb::GpuTable*> tables;
    std::map<std::string, TableDescription> tablesType;
    friend class TempTable;
};

template<typename T1, typename T2, typename T3> FUNC_PREFIX
T1 *newAddress(T1 *old, T2 *oldMemory, T3 *newMemory) {
    uintptr_t step1 =  reinterpret_cast<uintptr_t>(old);
    step1 -= reinterpret_cast<uintptr_t>(oldMemory);
    step1 += reinterpret_cast<uintptr_t>(newMemory);
    return reinterpret_cast<T1*>(step1);
}
