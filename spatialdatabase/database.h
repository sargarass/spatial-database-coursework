#pragma once
#include "gpudb.h"
#include "stackallocator.h"
#include "types.h"
#include "tabledescription.h"
#include "row.h"
#include "temptable.h"
#include "constobjects.h"

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

#pragma pack(pop)

typedef bool(*Predicate)(gpudb::CRow const &);

class DataBase : public Singleton {
    typedef std::pair<std::string, TableDescription> tablesTypePair;
    typedef std::pair<std::string, gpudb::GpuTable*> tablesPair;
public:
    bool loadFromDisk(std::string path);
    bool saveOnDisk(std::string path);
    bool selectRow(std::string tableName, Predicate p, std::vector<Row> &result);
    bool selectRow(TempTable &table, Predicate p, std::vector<Row> &result);
    bool insertTable(TempTable &table, std::string tableName);
    bool insertRow(std::string tableName, std::vector<Row> &rows);
    bool insertRow(std::string tableName, Row &row);
    bool createTable(TableDescription table);
    bool showTable(std::string tableName);
    bool showTable(TempTable &t);
    bool showTableHeader(std::string tableName);
    bool showTableHeader(TempTable &t);
    bool selectTable(std::string tableName, TempTable &table);
    bool update(std::string tableName, std::set<Attribute> const &atrSet, Predicate p);
    bool update(TempTable &t, std::set<Attribute> const &atrSet, Predicate p);
    bool dropRow(std::string tableName, Predicate p);
    bool dropRow(TempTable &t, Predicate p);
    bool dropTable(std::string tableName);

    static DataBase &getInstance();
    void deinit();
    virtual ~DataBase();
    TempTable pointxpointKnearestNeighbor(TempTable const &a, TempTable &b, uint k);
    TempTable polygonxpointPointsInPolygon(TempTable const &a, TempTable &b);
    TempTable linexpointPointsInBufferLine(TempTable const &a, TempTable &b, float radius);

private:
    gpudb::GpuRow *allocateRow(Row &row, TableDescription &desc, uint64_t &growMemSize);
    bool validateRow(Row &row, TableDescription &desc);
    bool hashDataBaseFile(FILE *file, FileDescriptor &desc);
    bool selectRowImp(TableDescription &desc, gpudb::GpuTable *table, Predicate p, std::vector<Row> &result);
    bool updateImp(TableDescription &desc, gpudb::GpuTable *table, std::set<Attribute> const &atrSet, Predicate p);
    bool showTableImp(gpudb::GpuTable const &table, const TableDescription &description, uint tabs =0);
    bool showTableHeaderImp(gpudb::GpuTable const &table, const TableDescription &description);
    bool dropRowImp(gpudb::GpuTable *table, Predicate p, bool freeRowMemory);

    static void store(gpudb::GpuRow * const dst, gpudb::GpuRow * const src);
    static void load(gpudb::GpuRow * const dst, gpudb::GpuRow * const src);
    static void storeGPU(gpudb::GpuRow *dst, gpudb::GpuRow *src, uint64_t memsize);
    static void loadCPU(gpudb::GpuRow *dstCPU, gpudb::GpuRow *srcGPU, uint64_t memsize);
    static bool copyTempTable(TableDescription const &description, gpudb::GpuTable const *gpuTable, TempTable &table);
    static void myprintf(uint tabs, char *format ...);
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
