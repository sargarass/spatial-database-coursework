#include "date.h"
#include <chrono>
#include "gpuallocator.h"
#define DATE_SIZE (2 * 50000 + 1)
static uint64_t dateYearToDay[DATE_SIZE];
static uint64_t dateYearToDayPrefixScan[DATE_SIZE];
static uint32_t  yearMonth[2][12];
static uint32_t  yearMonthPrefixScan[2][12];

__constant__ uint64_t *cudaDateYearToDay;
__constant__ uint64_t *cudaDateYearToDayPrefixScan;
__constant__ uint32_t  cudaYearMonth[2][12];
__constant__ uint32_t  cudaYearMonthPrefixScan[2][12];

static bool dateYearToDayInit = false;

static FUNC_PREFIX
bool testIfLeapYear(uint32_t year) {
    if (((year % 4) == 0 && (year % 100) != 0) || (year % 400 == 0)) {
        return true;
    }
    return false;
}

static void dateYearToDayInitFunc() {
#ifndef __CUDA_ARCH__
    if (dateYearToDayInit == false) {
        dateYearToDayInit = true;
        uint64_t numDaysFromBegining = 0;
        for (int i = -50000; i <= 50000; i++) {
            dateYearToDayPrefixScan[i + 50000] = numDaysFromBegining;
            if (testIfLeapYear(i)) {
                numDaysFromBegining += 366;
                dateYearToDay[i + 50000] = 366;
            } else {
                numDaysFromBegining += 365;
                dateYearToDay[i + 50000] = 365;
            }
        }
        for (int i = 0; i < 12; i++) {
            if (i % 2 == 0 && i <= 6) {
                yearMonth[0][i] = yearMonth[1][i] = 31;
            } else {
                if (i % 2 == 1 && i >= 7) {
                    yearMonth[0][i] = yearMonth[1][i] = 31;
                } else {
                    yearMonth[0][i] = yearMonth[1][i] = 30;
                }
            }
        }
        yearMonth[0][1] = 28;
        yearMonth[1][1] = 29;
        int m1 = 0;
        int m2 = 0;
        for (int i = 0; i < 12; i++) {
            yearMonthPrefixScan[0][i] = m1;
            m1 += yearMonth[0][i];

            yearMonthPrefixScan[1][i] = m2;
            m2 += yearMonth[1][i];
        }

        uint64_t *cudaDateYearToDayPrefixScanLocal = gpudb::gpuAllocator::getInstance().alloc<uint64_t>(2 * 50000 + 1);
        uint64_t *cudaDateYearToDayLocal = gpudb::gpuAllocator::getInstance().alloc<uint64_t>(2 * 50000 + 1);

        cudaMemcpy(cudaDateYearToDayPrefixScanLocal, dateYearToDayPrefixScan, sizeof(uint64_t) * DATE_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(cudaDateYearToDayLocal, dateYearToDay, sizeof(uint64_t) * DATE_SIZE, cudaMemcpyHostToDevice);

        cudaMemcpyToSymbol(cudaDateYearToDay, &cudaDateYearToDayLocal, sizeof(uint64_t *));
        cudaMemcpyToSymbol(cudaDateYearToDayPrefixScan, &cudaDateYearToDayPrefixScanLocal, sizeof(uint64_t *));
        cudaMemcpyToSymbol(cudaYearMonth, yearMonth, sizeof(yearMonth));
        cudaMemcpyToSymbol(cudaYearMonthPrefixScan, yearMonthPrefixScan, sizeof(yearMonthPrefixScan));
    }
#endif
}

FUNC_PREFIX
static bool isValidDate(int32_t year, uint32_t month, uint32_t day) {
    if (year < 0 || year > 100000) {
        return false;
    }

    if (month >= 12) {
        return false;
    }
    #ifdef __CUDA_ARCH__

    if (day >= cudaYearMonth[testIfLeapYear(year)][month]) {
        return false;
    }

    #else
    if (day >= yearMonth[testIfLeapYear(year)][month])  {
        return false;
    }
    #endif
    return true;
}

FUNC_PREFIX
static int findYear(uint32_t days) {
    int left = 0;
    int right = 50000 * 2 + 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        #ifdef __CUDA_ARCH__
        if (cudaDateYearToDayPrefixScan[mid] > days) {
        #else
        if (dateYearToDayPrefixScan[mid] > days) {
        #endif
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return (left - 1);
}

FUNC_PREFIX
static int findMonth(bool isLeap, uint32_t days) {
    int left = 0;
    int right = 12;
    while (left < right) {
        int mid = left + (right - left) / 2;
        #ifdef __CUDA_ARCH__
        if (cudaYearMonthPrefixScan[isLeap][mid] > days)
        #else
        if (yearMonthPrefixScan[isLeap][mid] > days)
        #endif
        {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return (left - 1);
}

FUNC_PREFIX
static void daysToYearMonthDays(uint64_t days, uint32_t &year, uint32_t &month, uint32_t &day) {
    year = findYear(days);
    bool leap = testIfLeapYear(year);
    #ifdef __CUDA_ARCH__
    month = findMonth(leap, days - cudaDateYearToDayPrefixScan[year]);
    day = days - cudaDateYearToDayPrefixScan[year] - cudaYearMonthPrefixScan[leap][month];
    #else
    month = findMonth(leap, days - dateYearToDayPrefixScan[year]);
    day = days - dateYearToDayPrefixScan[year] - yearMonthPrefixScan[leap][month];
    #endif
}

FUNC_PREFIX
static bool dateToDays(int32_t year, uint32_t month, uint32_t day, uint64_t &res) {
    if (!isValidDate(year, month, day)) {
        return false;
    }
    #ifdef __CUDA_ARCH__
    res = cudaDateYearToDayPrefixScan[year];
    res += cudaYearMonthPrefixScan[testIfLeapYear(year)][month];
    #else
    res = dateYearToDayPrefixScan[year];
    res += yearMonthPrefixScan[testIfLeapYear(year)][month];
    #endif
    res += day;
    return true;
}

Date Date::getDateFromEpoch() {
    std::chrono::microseconds ms = std::chrono::duration_cast< std::chrono::microseconds >(
        std::chrono::system_clock::now().time_since_epoch()
    );

    Date sinceEpoch;
    sinceEpoch.setFromEpoch(ms.count());
    return sinceEpoch;
}

FUNC_PREFIX
bool Date::setDate(int32_t year, uint64_t month, uint64_t day) {
    if (!isValidDate(year + 50000, month - 1, day - 1)) {
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "invalid date");
        return false;
    }

    this->year = (year + 50000);
    init |= 1 << DateType::YEAR;

    this->month = month - 1;
    init |= 1 << DateType::MONTH;

    this->day = day - 1;
    init |= 1 << DateType::DAY;
    return true;
}
bool Date::setFromEpoch(uint64_t microseconds) {
    uint64_t code = dateYearToDayPrefixScan[50000 + 1970] * 24ULL * 60ULL * 60ULL * 1000000ULL + microseconds;
    return setFromCode(code);
}

FUNC_PREFIX
bool Date::setFromCode(uint64_t code) {
    if (code >= MAX_CODE) {
        return false;
    }

    microseconds = code % 1000000ULL;
    code /= 1000000ULL;
    seconds = code % 60ULL;
    code /= 60ULL;
    minutes = code % 60ULL;
    code /= 60ULL;
    hour = code % 24ULL;
    code /= 24ULL;

    uint32_t t_year, t_month, t_day;
    daysToYearMonthDays(code, t_year, t_month, t_day);

    year = t_year;
    month = t_month;
    day = t_day;
    init = 0b01111111;
    return true;
}

FUNC_PREFIX
Date::Date() {
    dateYearToDayInitFunc();
    month = 0;
    day = 0;
    year = 0;
    minutes = 0;
    hour = 0;
    seconds = 0;
    microseconds = 0;
    init = 0;
}

Date Date::getRandomDate() {
    Date res;
    uint64_t u = ((double)rand()) / (double)(RAND_MAX) * MAX_CODE;
    res.setFromCode(u);
    return res;
}

FUNC_PREFIX
uint64_t Date::codeDate() {
    if (!isValid()) {
        return 0xFFFFFFFFFFFFFFFFFFULL;
    }
    uint64_t res;
    dateToDays(year, month, day, res);
    return (((res * 24ULL + hour) * 60ULL + minutes) * 60ULL + seconds) * 1000000ULL + microseconds;
}
