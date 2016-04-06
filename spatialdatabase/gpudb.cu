#include "gpudb.h"

uint2 gpudb::TemporalKey::centroid() {
    uint2 res;
    switch(temporalType) {
        case TemporalType::VALID_TIME:
            res.x = (validTimeS + ((validTimeE - validTimeS) / 2ULL)) >> 32ULL;
            res.y = 0xFFFFFFFF; // Зарезервированное значение -- значит у нас только 3-мерные координаты
            break;
        case TemporalType::TRANSACTION_TIME:
            res.x = (transactionTypeS + ((transactionTypeE - transactionTypeS) / 2ULL)) >> 32ULL;
            res.y = 0xFFFFFFFF;
            break;
        case TemporalType::BITEMPORAL_TIME:
            res.x = (validTimeS + ((validTimeE - validTimeS) / 2ULL)) >> 32ULL;
            res.y = (transactionTypeS + ((transactionTypeE - transactionTypeS) / 2ULL)) >> 32ULL;
            break;
        default:
            gLog.getInstance().write(LOG_MESSAGE_TYPE::ERROR,
                       "gpudb::TemporalKey",
                       "centroid",
                       "Unexpected temporalType Value");
            break;
    }
    return res;
}
