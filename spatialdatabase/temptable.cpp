#include "temptable.h"
#include "database.h"

TempTable::~TempTable() {
    if (this->table) {
        bool set = false;
        for (int i = 0; i < description.columnDescription.size(); i++) {
            if (description.columnDescription[i].type == Type::SET) {
                set = true;
                break;
            }
        }
        if (set == true) {
            thrust::host_vector<gpudb::GpuRow*> rows = table->rows;
            for (int i = 0; i < table->rows.size(); i++) {
                uint8_t* memory = StackAllocator::getInstance().alloc<uint8_t>(table->rowsSize[i]);
                gpudb::GpuRow* pointer = (gpudb::GpuRow*)memory;

                DataBase::loadCPU(pointer, rows[i], table->rowsSize[i]);
                for (int j = 0; j < pointer->valueSize; j++) {
                    if (description.columnDescription[j].type == Type::SET) {
                        delete (*((TempTable**)(pointer->value[j].value)));
                    }
                }
            }
        }
    }
    delete this->table;
    this->table = nullptr;
    this->valid = false;
}

TempTable::TempTable() {
    this->table = nullptr;
    this->valid = false;
}

bool TempTable::showTable() {
    if (this->table == nullptr) {
        return false;
    }
    return DataBase::getInstance().showTable(*this->table, this->description);
}

SpatialType TempTable::getSpatialKeyType() const {
    if (this->table == nullptr) {
        return SpatialType::UNKNOWN;
    }
    return this->table->spatialKey.type;
}

TemporalType TempTable::getTemporalType() const {
    if (this->table == nullptr) {
        return TemporalType::UNKNOWN;
    }
    return this->table->temporalKey.type;
}

bool TempTable::isValid() const {
    return this->valid;
}
