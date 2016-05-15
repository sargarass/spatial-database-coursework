#include "temptable.h"
#include "database.h"
#include <stack>
TempTable::~TempTable() {
    deinit();
}

void TempTable::deinit() {
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
    this->description = TableDescription();
    this->needToBeFree.clear();
    this->parents.clear();
    this->references.clear();
}

TempTable::TempTable() {
    this->table = nullptr;
    this->valid = false;
}

void TempTable::operator=(TempTable &t) {
    if (this->table) {
        deinit();
    }

    std::swap(this->description, t.description);
    std::swap(this->table, t.table);
    std::swap(this->needToBeFree, t.needToBeFree);
    std::swap(this->parents, t.parents);
    std::swap(this->references, t.references);
    std::swap(this->valid, t.valid);
}

TempTable::TempTable(TempTable &table) {
    *this = table;
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
