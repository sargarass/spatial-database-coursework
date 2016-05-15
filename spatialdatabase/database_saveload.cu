#include "database.h"

bool DataBase::loadFromDisk(std::string path) {
    deinit();
    FILE *file = fopen(path.c_str(), "rb");

    if (file == nullptr) {
        gLogWrite(LOG_MESSAGE_TYPE::ERROR, "file %s was not opened", path.c_str());
        return false;
    }

    FileDescriptor fdesc;
    fread(&fdesc, sizeof(FileDescriptor), 1, file);
    FileDescriptor hash;

    if (!hashDataBaseFile(file, hash)) {
        fclose(file);
        return false;
    }

    for (int i = 0; i < SHA512_DIGEST_LENGTH; i++) {
        if (hash.sha512[i] != fdesc.sha512[i]) {
            fclose(file);
            return false;
        }
    }

    for (uint64_t tableIter = 0; tableIter < fdesc.tablesNum; tableIter++) {
        gpudb::GpuTable * gputable = new gpudb::GpuTable;
        TableChunk tchunk;
        fread(&tchunk, sizeof(TableChunk), 1, file);

        memcpy(gputable->name, tchunk.tableName, NAME_MAX_LEN);
        gputable->name[NAME_MAX_LEN - 1] = 0;

        TableDescription desc;
        desc.name = std::string(gputable->name);
        desc.spatialKeyName = std::string(tchunk.spatialKeyName);
        desc.temporalKeyName = std::string(tchunk.temporalKeyName);
        desc.spatialKeyType = tchunk.spatialKeyType;
        desc.temporalKeyType = tchunk.temporalKeyType;

        thrust::host_vector<gpudb::GpuColumnAttribute> columns;
        for (uint32_t colIter = 0; colIter < tchunk.numColumns; colIter++) {
            AttributeDescription atr;
            ColumnsChunk readCol;

            fread(&readCol, sizeof(ColumnsChunk), 1, file);

            atr.name = std::string(readCol.atr.name);
            atr.type = readCol.atr.type;

            desc.addColumn(atr);
            columns.push_back(readCol.atr);
        }

        gputable->columns.reserve(columns.size());
        gputable->columns = columns;

        for (uint32_t sizeIter = 0; sizeIter < tchunk.numRows; sizeIter++) {
            RowChunk sizechunk;
            fread(&sizechunk, sizeof(RowChunk), 1, file);
            gputable->rowsSize.push_back(sizechunk.rowSize);
        }

        thrust::host_vector<gpudb::GpuRow*> hostRowMemory;
        for (uint32_t rowIter = 0; rowIter < tchunk.numRows; rowIter++) {
            uint8_t *memory = gpudb::gpuAllocator::getInstance().alloc<uint8_t>(gputable->rowsSize[rowIter]);

            if (memory != nullptr) {
                hostRowMemory.push_back((gpudb::GpuRow*)(memory));
            } else {
                for (uint j = 0; j < rowIter; j++) {
                    gpudb::gpuAllocator::getInstance().free(hostRowMemory[j]);
                }
                return false;
            }
        }

        for (uint32_t rowIter = 0; rowIter < tchunk.numRows; rowIter++) {
            uint8_t *memory = StackAllocator::getInstance().alloc<uint8_t>(gputable->rowsSize[rowIter]);
            uint32_t writeSize = gputable->rowsSize[rowIter];
            uint32_t chunkSize = 1024;
            uint32_t numfwrite = writeSize / chunkSize;
            uint32_t tail = writeSize % chunkSize;

            uint64_t offset = 0;
            for (uint32_t part = 0; part < numfwrite; part++) {
                fread(memory + offset, chunkSize, 1, file);
                offset += chunkSize;
            }

            if (tail > 0) {
                fread(memory + offset, tail, 1, file);
            }
            load((gpudb::GpuRow *)(memory), NULL);
            storeGPU(hostRowMemory[rowIter], (gpudb::GpuRow *)memory, writeSize);
            StackAllocator::getInstance().free(memory);
        }

        gputable->rows.reserve(hostRowMemory.size());
        gputable->rows = hostRowMemory;
        tables.insert(tablesPair(desc.name, gputable));
        tablesType.insert(tablesTypePair(desc.name, desc));
    }
    fclose(file);
    return true;
}


bool DataBase::hashDataBaseFile(FILE *file, FileDescriptor &desc) {
    if (!file) {
        return false;
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
        fread(chunk, chunkSize, 1, file);
        SHA512_Update(&ctx, chunk, chunkSize);
    }

    if (tail > 0) {
        fread(chunk, tail, 1, file);
        SHA512_Update(&ctx, chunk, tail);
    }

    SHA512_Final(desc.sha512, &ctx);
    fseek(file, position, SEEK_SET);
/*
    for (int i = 0; i < SHA512_DIGEST_LENGTH; i++) {
        printf("%d", (uint)(desc.sha512[i]));
    }
    printf("\n");*/

    return true;
}

bool DataBase::saveOnDisk(std::string path) {
    FileDescriptor fdesc;
    fdesc.tablesNum = this->tables.size();
    FILE *file = fopen(path.c_str(), "wb+");

    if (!file) {
        return false;
    }

    fwrite(&fdesc, sizeof(FileDescriptor), 1, file);

    auto it = this->tables.begin();
    auto it2 = this->tablesType.begin();
    for (uint64_t i = 0; i < this->tables.size(); i++) {
        gpudb::GpuTable *table = it->second;
        TableDescription &desc = it2->second;

        thrust::host_vector<gpudb::GpuColumnAttribute> atrVec = table->columns;
        thrust::host_vector<gpudb::GpuRow*> gpuRows = table->rows;

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

        fwrite(&curtable, sizeof(TableChunk), 1, file);

        for (gpudb::GpuColumnAttribute &atr : atrVec) {
            ColumnsChunk chunk;
            chunk.atr = atr;
            fwrite(&chunk, sizeof(ColumnsChunk), 1, file);
        }

        for (uint64_t size : table->rowsSize) {
            RowChunk chunk;
            chunk.rowSize = size;
            fwrite(&chunk, sizeof(RowChunk), 1, file);
        }

        for (uint64_t iter = 0; iter < table->rows.size(); iter++) {
            uint8_t *memory = StackAllocator::getInstance().alloc<uint8_t>(table->rowsSize[iter]);
            gpudb::GpuRow *cpuRow = reinterpret_cast<gpudb::GpuRow*>(memory);

            loadCPU(cpuRow, gpuRows[iter], table->rowsSize[iter]);
            store((gpudb::GpuRow *)NULL, cpuRow); // там нужно чтобы указатели обозначали только offset от начала структуры
            uint32_t writeSize = table->rowsSize[iter];
            uint32_t chunkSize = 1024;
            uint32_t numfwrite = writeSize / chunkSize;
            uint32_t tail = writeSize % chunkSize;

            uint64_t offset = 0;
            for (uint32_t part = 0; part < numfwrite; part++) {
                fwrite(memory + offset, chunkSize, 1, file);
                offset += chunkSize;
            }

            if (tail > 0) {
                fwrite(memory + offset, tail, 1, file);
            }

            StackAllocator::getInstance().free(memory);
        }
        it++;
        it2++;
    }

    if (!hashDataBaseFile(file, fdesc)) {
        fclose(file);
        return false;
    }

    fseek(file, 0, SEEK_SET);
    fwrite(&fdesc, sizeof(FileDescriptor), 1, file);
    fclose(file);
    return true;
}
