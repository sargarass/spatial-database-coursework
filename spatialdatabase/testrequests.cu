#include "testrequests.h"
#include "constobjects.h"
using namespace gpudb;

__device__
bool tester(gpudb::CRow const &row) {
    Date validS, validE;
    row.getKeyValidTime(validS, validE);
    return (validS.getYear() >= -21669);
}


__device__ Predicate h_tester = tester;
Predicate getTesterPointer() {
    Predicate p;
    cudaMemcpyFromSymbol(&p, h_tester, sizeof(Predicate));
    return p;
}
