#include "temptable.h"
#include "database.h"
#include <stack>
TempTable::~TempTable() {
    for (auto& v : needToBeFree) {
        delete (void*) v;
    }

    // удалили ссылки вверх
    for (TempTable * p: parents) {
        auto it = std::find(p->references.begin(), p->references.end(), this);
        p->references.erase(it);
    }

    std::stack<TempTable *> st;
    for (TempTable *ref : references) {
        st.push(ref);
    }
    // почистили ссылки вниз
    while(!st.empty()) {
        TempTable *t = st.top(); st.pop();
        t->valid = false;
        t->parents.clear();
        for (TempTable* t : t->references) {
            st.push(t);
        }
        t->references.clear();
    }

    if (table) {
        delete this->table;
    }
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
    return this->description.spatialKeyType;
}

TemporalType TempTable::getTemporalType() const {
    if (this->table == nullptr) {
        return TemporalType::UNKNOWN;
    }
    return this->description.temporalKeyType;
}

bool TempTable::isValid() const {
    return this->valid;
}
