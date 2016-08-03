#include "database.h"

struct RAII_file {
    RAII_file(FILE *file) : f {file}
    {}

    ~RAII_file() {
        if (f != nullptr) {
            fclose(f);
        }
    }

    operator FILE*() {
        return f;
    }

    FILE *f;
};


Result<void, Error<std::string>> DataBase::loadFromDisk(std::string path) {
    deinit();

    RAII_file file { fopen(path.c_str(), "rb") };

    if (file == nullptr) {
        return MYERR_STRING("file was not opened");
    }

    FileDescriptor fdesc;
    fread(&fdesc, sizeof(FileDescriptor), 1, file);
    FileDescriptor hash;

    TRY(hashDataBaseFile(file, hash));

    for (int i = 0; i < SHA512_DIGEST_LENGTH; i++) {
        if (hash.sha512[i] != fdesc.sha512[i]) {
            return MYERR_STRING("file was corrupted");
        }
    }

    RAII_GC<gpudb::GpuTable> gc;
    for (uint64_t tableIter = 0; tableIter < fdesc.tablesNum; tableIter++) {
        gpudb::GpuTable * gputable = new (std::nothrow) gpudb::GpuTable;

        if (gputable == nullptr) {
            return MYERR_STRING("not enought memory");
        }
        gc.registrCPU(gputable);


        TableChunk tchunk;

        if (fread(&tchunk, sizeof(TableChunk), 1, file) != 1) {
            return MYERR_STRING("I/O error");
        }

        memcpy(gputable->name, tchunk.tableName, NAME_MAX_LEN);
        gputable->name[NAME_MAX_LEN - 1] = 0;

        TableDescription desc;
        desc.name = std::string(gputable->name);
        desc.spatialKeyName = std::string(tchunk.spatialKeyName);
        desc.temporalKeyName = std::string(tchunk.temporalKeyName);
        desc.spatialKeyType = tchunk.spatialKeyType;
        desc.temporalKeyType = tchunk.temporalKeyType;

        std::vector<gpudb::GpuColumnAttribute> columns;
        for (uint32_t colIter = 0; colIter < tchunk.numColumns; colIter++) {
            AttributeDescription atr;
            ColumnsChunk readCol;

            if (fread(&readCol, sizeof(ColumnsChunk), 1, file) != 1) {
                return MYERR_STRING("I/O error");
            }

            atr.name = std::string(readCol.atr.name);
            atr.type = readCol.atr.type;

            desc.addColumn(atr);
            columns.push_back(readCol.atr);
        }

        gputable->columns.reserve(columns.size());
        gputable->columns = columns;

        for (uint32_t sizeIter = 0; sizeIter < tchunk.numRows; sizeIter++) {
            RowChunk sizechunk;
            if (fread(&sizechunk, sizeof(RowChunk), 1, file) != 1) {
                return MYERR_STRING("I/O error");
            }
            gputable->rowsSize.push_back(sizechunk.rowSize);
        }

        std::vector<gpudb::GpuRow*> hostRowMemory;
        for (uint32_t rowIter = 0; rowIter < tchunk.numRows; rowIter++) {
            uint8_t *memory = gpudb::gpuAllocator::getInstance().alloc<uint8_t>(gputable->rowsSize[rowIter]);

            if (memory != nullptr) {
                gc.registrGPU(memory);
                hostRowMemory.push_back((gpudb::GpuRow*)(memory));
            } else {
                return MYERR_STRING("not enought gpumemory");
            }
        }

        for (uint32_t rowIter = 0; rowIter < tchunk.numRows; rowIter++) {
            auto memory = StackAllocatorAdditions::allocUnique<uint8_t>(gputable->rowsSize[rowIter]);
            uint32_t writeSize = gputable->rowsSize[rowIter];
            uint32_t chunkSize = 1024;
            uint32_t numfwrite = writeSize / chunkSize;
            uint32_t tail = writeSize % chunkSize;

            uint64_t offset = 0;
            if (fread(memory.get() + offset, chunkSize, numfwrite, file) != numfwrite) {
                return MYERR_STRING("I/O error");
            }

            offset += chunkSize * numfwrite;

            if (tail > 0) {
                if (fread(memory.get() + offset, tail, 1, file) != 1) {
                    return MYERR_STRING("I/O error");
                }
            }

            load((gpudb::GpuRow *)(memory.get()), NULL);
            storeGPU(hostRowMemory[rowIter], (gpudb::GpuRow *)memory.get(), writeSize);
        }

        gputable->rows.resize(hostRowMemory.size());
        thrust::copy(hostRowMemory.begin(), hostRowMemory.end(), gputable->rows.begin());

        tables.insert(tablesPair(desc.name, gputable));
        tablesType.insert(tablesTypePair(desc.name, desc));

        gc.takeGPU();
        gc.takeCPU();
    }
    return Ok();
}


Result<void, Error<std::string>> DataBase::hashDataBaseFile(FILE *file, FileDescriptor &desc) {
    if (!file) {
        return MYERR_STRING("file is nullptr");
    }

    uint64_t position = ftell(file);
    SHA512_CTX ctx;
    SHA512_Init(&ctx);

    fseek(file, 0L, SEEK_END);
    uint64_t sz = ftell(file) - offsetof(FileDescriptor, databaseId);
    fseek(file, offsetof(FileDescriptor, databaseId), SEEK_SET);

    uint32_t readSize = sz;
    uint32_t chunkSize = 1024;
    uint32_t numfread = readSize / chunkSize;
    uint32_t tail = readSize % chunkSize;

    char chunk[chunkSize];
    for (uint32_t part = 0; part < numfread; part++) {
        if (fread(chunk, chunkSize, 1, file) != 1) {
            return MYERR_STRING("IO error");
        }
        SHA512_Update(&ctx, chunk, chunkSize);
    }

    if (tail > 0) {
        if (fread(chunk, tail, 1, file) != 1) {
            return MYERR_STRING("IO error");
        }

        SHA512_Update(&ctx, chunk, tail);
    }

    SHA512_Final(desc.sha512, &ctx);
    fseek(file, position, SEEK_SET);
    return Ok();
}

Result<void, Error<std::string>> DataBase::saveOnDisk(std::string path) {
    FileDescriptor fdesc;
    fdesc.tablesNum = this->tables.size();
    RAII_file file = { fopen(path.c_str(), "wb+") };

    if (!file) {
        return MYERR_STRING("file is nullptr");
    }

    if (fwrite(&fdesc, sizeof(FileDescriptor), 1, file) != 1) {
        return MYERR_STRING("IO error");
    }

    auto it = this->tables.begin();
    auto it2 = this->tablesType.begin();
    for (uint64_t i = 0; i < this->tables.size(); i++) {
        gpudb::GpuTable *table = it->second;
        TableDescription &desc = it2->second;

        std::vector<gpudb::GpuColumnAttribute> atrVec(table->columns.size());
        std::vector<gpudb::GpuRow*> gpuRows(table->rows.size());
        thrust::copy(table->columns.begin(), table->columns.end(), atrVec.begin());
        thrust::copy(table->rows.begin(), table->rows.end(), gpuRows.begin());

        TableChunk curtable;
        memcpy(curtable.tableName, table->name, NAME_MAX_LEN);
        memcpy(curtable.spatialKeyName, desc.spatialKeyName.data(), NAME_MAX_LEN);
        memcpy(curtable.temporalKeyName, desc.temporalKeyName.data(), NAME_MAX_LEN);

        curtable.spatialKeyType = desc.spatialKeyType;
        curtable.temporalKeyType = desc.temporalKeyType;

        curtable.tableName[NAME_MAX_LEN - 1] = 0;
        curtable.temporalKeyName[NAME_MAX_LEN - 1] = 0;
        curtable.spatialKeyName[NAME_MAX_LEN - 1] = 0;

        curtable.numColumns = table->columns.size();
        curtable.numRows = table->rows.size();

        if (fwrite(&curtable, sizeof(TableChunk), 1, file) != 1) {
            return MYERR_STRING("IO error");
        }

        for (gpudb::GpuColumnAttribute &atr : atrVec) {
            ColumnsChunk chunk;
            chunk.atr = atr;
            if (fwrite(&chunk, sizeof(ColumnsChunk), 1, file) != 1) {
                return MYERR_STRING("IO error");
            }
        }

        for (uint64_t size : table->rowsSize) {
            RowChunk chunk;
            chunk.rowSize = size;
            if (fwrite(&chunk, sizeof(RowChunk), 1, file) != 1) {
                return MYERR_STRING("IO error");
            }
        }

        for (uint64_t iter = 0; iter < table->rows.size(); iter++) {
            auto memory = StackAllocatorAdditions::allocUnique<uint8_t>(table->rowsSize[iter]);
            gpudb::GpuRow *cpuRow = reinterpret_cast<gpudb::GpuRow*>(memory.get());

            loadCPU(cpuRow, gpuRows[iter], table->rowsSize[iter]);
            store((gpudb::GpuRow *)NULL, cpuRow); // там нужно чтобы указатели обозначали только offset от начала структуры
            uint32_t writeSize = table->rowsSize[iter];
            uint32_t chunkSize = 1024;
            uint32_t numfwrite = writeSize / chunkSize;
            uint32_t tail = writeSize % chunkSize;

            uint64_t offset = 0;
            if (fwrite(memory.get() + offset, chunkSize, numfwrite, file) != numfwrite) {
                return MYERR_STRING("IO error");
            }

            offset += chunkSize * numfwrite;


            if (tail > 0) {
                if (fwrite(memory.get() + offset, tail, 1, file) != 1) {
                    return MYERR_STRING("IO error");
                }
            }
        }
        it++;
        it2++;
    }

    TRY(hashDataBaseFile(file, fdesc));

    fseek(file, 0, SEEK_SET);

    if (fwrite(&fdesc, sizeof(FileDescriptor), 1, file) != 1) {
        return MYERR_STRING("IO error");
    }

    return Ok();
}
