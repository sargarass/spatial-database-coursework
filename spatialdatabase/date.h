#pragma once

struct Date {
public:
    Date() {
        month = 0;
        day = 0;
        year = 0;
        minutes = 0;
        hour = 0;
    }

    bool setDay(uint64_t day) {
        if (day >= 31 && day <= 0) {
            return false;
        }
        return false;
    }

    bool setYear(uint64_t year) {
        return false;
    }

    bool setMonth(uint64_t month) {
        return false;
    }
    bool setHour(uint64_t hour) {
        return false;
    }
    bool setMinutes(uint64_t minutes) {
        return false;
    }
    bool isValid() {
        return false;
    }

    uint64_t codeDate() { return 0; }
private:
    uint64_t month;
    uint64_t day;
    uint64_t hour;
    uint64_t year;
    uint64_t minutes;
};

