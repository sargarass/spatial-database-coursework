#include "testrequests.h"
#include "constobjects.h"

__device__
bool tester(gpudb::CRow const &row) {
    Date validS, validE;
    row.getKeyValidTime(validS, validE);
    return (validS.getYear() >= 2016 && validE.getYear() <= 2017);
}

__device__ Predicate h_tester = tester;

Predicate getTesterPointer() {
    Predicate p;
    cudaMemcpyFromSymbol(&p, h_tester, sizeof(Predicate));
    return p;
}
