#pragma once
#include "types.h"
#include <cfloat>
#include <cmath>

namespace gpudb {
#define AABBmin(p) p.x
#define AABBmax(p) p.y
#pragma pack(push, 1)
    struct MortonCode {
        uint64_t high;
        uint64_t low;
        uint bits;

        std::string toString() {
            std::string s;
            uint64_t llow = low;
            uint64_t lhigh = high;

            for (int i = 63; i >= 0; i--) {
                s += ((lhigh & 0x8000000000000000ULL) >> 63ULL) + '0';
                lhigh <<= 1;
            }
            s += " ";
            for (int i = 63; i >= 0; i--) {
                s += ((llow & 0x8000000000000000ULL) >> 63ULL) + '0';
                llow <<= 1;
            }
            return s;
        }

        bool operator<(MortonCode const &m) const {
            if (high < m.high) {
                return true;
            }

            if ((high == m.high) && (low < m.low)){
                return true;
            }
            return false;
        }
    };
#pragma pack(pop)

    class AABB {
    public:
        float2 x; // x
        float2 y; // y
        float2 z; // valid time
        float2 w; // transaction time
        int numComp;
        int64_t clamp(int64_t in, int64_t min, int64_t max) {
            int64_t res = (in > max)? max : in;
            res = (res < min)? min : res;
            return res;
        }

        __device__ __host__
        MortonCode getMortonCode(AABB const &global) {
            float4 centroid = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            MortonCode code;
            switch (numComp) {
                case 4:
                    centroid.w = AABBmin(w) + (AABBmax(w) - AABBmin(w)) / 2.0f;
                case 3:
                    centroid.z = AABBmin(z) + (AABBmax(z) - AABBmin(z)) / 2.0f;
                case 2:
                    centroid.x = AABBmin(x) + (AABBmax(x) - AABBmin(x)) / 2.0f;
                    centroid.y = AABBmin(y) + (AABBmax(y) - AABBmin(y)) / 2.0f;

                    centroid.x -= AABBmin(global.x);
                    centroid.y -= AABBmin(global.y);
                    centroid.z -= AABBmin(global.z);
                    centroid.w -= AABBmin(global.w);

                    centroid.x /= (AABBmax(global.x) - AABBmin(global.x));
                    centroid.y /= (AABBmax(global.y) - AABBmin(global.y));
                    centroid.z /= (AABBmax(global.z) - AABBmin(global.z));
                    centroid.w /= (AABBmax(global.w) - AABBmin(global.w));
                break;
                default:
                    break;
            }

            uint4 centroidInt;
            const double shift = (double)(UINT_MAX);
            centroidInt.x = clamp(centroid.x * shift, 0, UINT_MAX);
            centroidInt.y = clamp(centroid.y * shift, 0, UINT_MAX);
            centroidInt.z = clamp(centroid.z * shift, 0, UINT_MAX);
            centroidInt.w = clamp(centroid.w * shift, 0, UINT_MAX);

            code.bits = 96;
            code.high = 0;
            code.low = 0;
            // заполняем верхную часть
            uint p = 23;
            for (int i = 0; i < 6; i++) {
                code.high |= ((centroidInt.x & 0x80000000) >> 31) << (p    );
                code.high |= ((centroidInt.y & 0x80000000) >> 31) << (p - 1);
                code.high |= ((centroidInt.z & 0x80000000) >> 31) << (p - 2);
                code.high |= ((centroidInt.w & 0x80000000) >> 31) << (p - 3);
                p -= 4;
                centroidInt.x <<= 1;
                centroidInt.y <<= 1;
                centroidInt.z <<= 1;
                centroidInt.w <<= 1;
            }

            p = 63;
            for (int i = 0; i < 24; i++) {
                code.low |= ((uint64_t) ((centroidInt.x & 0x80000000) >> 31)) << (p    );
                code.low |= ((uint64_t) ((centroidInt.y & 0x80000000) >> 31)) << (p - 1);
                code.low |= ((uint64_t) ((centroidInt.z & 0x80000000) >> 31)) << (p - 2);
                code.low |= ((uint64_t) ((centroidInt.w & 0x80000000) >> 31)) << (p - 3);
                p -= 4;
                centroidInt.x <<= 1;
                centroidInt.y <<= 1;
                centroidInt.z <<= 1;
                centroidInt.w <<= 1;
            }

            return code;
        }

        __device__ __host__
        MortonCode getMortonCode() {
            AABB global;
            AABBmin(global.x) = -180.0f;
            AABBmin(global.y) = -180.0f;
            AABBmin(global.z) = 0.0f;
            AABBmin(global.w) = 0.0f;

            AABBmax(global.x) = 180.0f;
            AABBmax(global.y) = 180.0f;
            AABBmax(global.z) = 1.0f;
            AABBmax(global.w) = 1.0f;
            return getMortonCode(global);
        }
    };
}
