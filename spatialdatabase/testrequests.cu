#include "testrequests.h"
#include "constobjects.h"
#include "types.h"
using namespace gpudb;

bool FILTER_CU_FUNC(filter_strcmp)(char const *str1, char const *str2) {
    for (; *str1 && *str2 && *str1 == *str2; ++str1, ++str2) {}
    return *str2 == *str1;
}

FILTER_CU(roadFilter) {
    return !(filter_strcmp(row.getColumnSTRING(1), "Кафе")
             || filter_strcmp(row.getColumnSTRING(1), "Ресторан")); // Фильтруем все результаты, для которых не выполнено условие
}

FILTER_CU(tester) {
    Date validS = row.getKeyValidTimeStart();
    Date validE = row.getKeyValidTimeEnd();
    return (validS.getYear() < 2013 && validE.getYear() < 2013);
}

