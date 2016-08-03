#include "testrequests.h"
#include "constobjects.h"
using namespace gpudb;

FILTER_CU(tester){
    Date validS = row.getKeyValidTimeStart();
    Date validE = row.getKeyValidTimeEnd();
    return (validS.getYear() < 2013 && validE.getYear() < 2013);
}

