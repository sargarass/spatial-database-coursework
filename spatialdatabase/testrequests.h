#pragma once
#include "database.h"

void test();

__device__
bool tester(const gpudb::CRow &row);

Predicate getTesterPointer();
