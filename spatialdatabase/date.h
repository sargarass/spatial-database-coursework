#pragma once
#include "log.h"
// log2(100000*12*31*24*60*60*1000000) = 61.4791115468
// и того:
// лет (-50000 до 50000)
// точность до микросекунд
#include <cuda_runtime.h>

#define FUNC_PREFIX __host__ __device__
#define MAX_CODE 3155726736000000000ULL
#define MAX_CODE_BITS 62ULL
#define MAX_CODE_AFTER_SHIFT 2938999549.0
#define CODE_NORMALIZE 1e-16
#define CODE_SHIFT_TO_UINT (32ULL - (64ULL - MAX_CODE_BITS))

enum DateType {
    YEAR,
    MONTH,
    DAY,
    HOUR,
    MINUTE,
    SECOND,
    MICROSECOND,
    DATA_TYPE_BOUND
};

#pragma pack(push, 1)
struct Date {
public:
    FUNC_PREFIX
    bool setFromCode(uint64_t code);

    FUNC_PREFIX
    Date();

    FUNC_PREFIX
    Date(uint32_t year, uint32_t month, uint32_t day, uint32_t hour, uint32_t minutes, uint32_t seconds, uint32_t microseconds)
        : Date() {
        set(year, month, day, hour, minutes, seconds, microseconds);
    }

    FUNC_PREFIX
    bool setHour(uint64_t hour) {
        if (hour <= 23) {
            this->hour = hour;
            init |= 1  << DateType::HOUR;
            return true;
        }
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "invalid hours");
        return false;
    }

    FUNC_PREFIX
    bool setMinutes(uint64_t minutes) {
        if (minutes <= 59) {
            this->minutes = minutes;
            init |= 1 << DateType::MINUTE;
            return true;
        }
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "invalid minutes");
        return false;
    }

    FUNC_PREFIX
    bool setSeconds(uint64_t seconds) {
        if (seconds <= 59) {
            this->seconds = seconds;
            init |= 1 << DateType::SECOND;
            return true;
        }
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "invalid seconds");
        return false;
    }

    FUNC_PREFIX
    bool setMicroseconds(uint64_t microseconds) {
        if (microseconds < 1000000) {
            this->microseconds = microseconds;
            init |= 1 << DateType::MICROSECOND;
            return true;
        }
        gLogWrite(LOG_MESSAGE_TYPE::DEBUG, "invalid microseconds");
        return false;
    }

    FUNC_PREFIX
    bool set(uint32_t year, uint32_t month, uint32_t day, uint32_t hour, uint32_t minutes, uint32_t seconds, uint32_t microseconds) {
        Date tmp;
        tmp.setDate(year, month, day);
        tmp.setHour(hour);
        tmp.setMinutes(minutes);
        tmp.setSeconds(seconds);
        tmp.setMicroseconds(microseconds);
        if (tmp.isValid()) {
            *this = tmp;
            return true;
        }
        return false;
    }

    static Date getRandomDate();
    static Date getRandomDate(Date const &start, Date const &end);
    FUNC_PREFIX
    int32_t getYear() const {
        return year - 50000;
    }

    FUNC_PREFIX
    uint32_t getMonth() const {
        return month + 1;
    }

    FUNC_PREFIX
    uint32_t getDay() const {
        return day + 1;
    }

    FUNC_PREFIX
    uint32_t getHour() const {
        return hour;
    }

    FUNC_PREFIX
    uint32_t getSeconds() const {
        return seconds;
    }

    FUNC_PREFIX
    uint32_t getMinutes() const {
        return minutes;
    }

    FUNC_PREFIX
    uint32_t getMicroseconds() const {
        return microseconds;
    }

    FUNC_PREFIX
    bool isValid() const {
        return ((init ^ 0b01111111) == 0);
    }

    FUNC_PREFIX
    bool setDate(int32_t year, uint64_t month, uint64_t day);
    static Date getDateFromEpoch();

    FUNC_PREFIX
    uint64_t codeDate() const;

    std::string toString() const {
        std::string str;
        str.resize(256);
        std::snprintf(&str[0], 256, "%d/%02d/%02d %02d:%02d:%02d:%05d", getYear(), getMonth(), getDay(), getHour(), getMinutes(), getSeconds(), getMicroseconds());
        return str;
    }

    bool setFromEpoch(uint64_t microseconds);
private:
    uint8_t init;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint32_t year;
    uint8_t minutes;
    uint8_t seconds;
    uint32_t microseconds;
};

#pragma pack(pop)
