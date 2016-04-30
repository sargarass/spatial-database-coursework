#include "types.h"
#include "math.h"

static FUNC_PREFIX
float3 make_float3(float4 f4)
{
    float3 a;
    a.x = f4.x;
    a.y = f4.y;
    a.z = f4.z;
    return a;
}

static FUNC_PREFIX
float3 make_float3(float2 xy, float z)
{
    float3 a;
    a.x = xy.x;
    a.y = xy.y;
    a.z = z;
    return a;
}

static FUNC_PREFIX
float3 make_float3(float x)
{
    float3 a;
    a.x = x;
    a.y = x;
    a.z = x;
    return a;
}

static FUNC_PREFIX
float3 & operator += (float3 & a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

static FUNC_PREFIX
float3 & operator -= (float3 & a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

static FUNC_PREFIX
float3 & operator *= (float3 & a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

static FUNC_PREFIX
float3 & operator *= (float3 & a, float c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}

static FUNC_PREFIX
float3 & operator /= (float3 & a, float3 & b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

static FUNC_PREFIX
float3 & operator /= (float3 & a, float c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    return a;
}

static FUNC_PREFIX
float3 operator - (float3 a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

static FUNC_PREFIX
float3 operator + (float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static FUNC_PREFIX
float3 operator + (float3 a, float c)
{
    return make_float3(a.x + c, a.y + c, a.z + c);
}

static FUNC_PREFIX
float3 operator - (float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static FUNC_PREFIX
float3 operator - (float3 a, float c)
{
    return make_float3(a.x - c, a.y - c, a.z - c);
}

static FUNC_PREFIX
float3 operator * (float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static FUNC_PREFIX
float3 operator * (float3 a, float c)
{
    return make_float3(a.x * c, a.y * c, a.z * c);
}

static FUNC_PREFIX
float3 operator * (float c, float3 a)
{
    return make_float3(a.x * c, a.y * c, a.z * c);
}

static FUNC_PREFIX
float3 operator / (float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

static FUNC_PREFIX
float3 operator / (float3 a, float c)
{
    float rc = 1.0f / c;
    return make_float3(a.x * rc, a.y * rc, a.z * rc);
}

static FUNC_PREFIX
float3 operator / (float c, float3 a)
{
    return make_float3(c / a.x, c / a.y, c / a.z);
}

static FUNC_PREFIX
float3 Cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

static FUNC_PREFIX
float Dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static FUNC_PREFIX
float len(float3 a)
{
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

static FUNC_PREFIX
float lenSqr(float3 a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

static FUNC_PREFIX
float3 norma(float3 a)
{
    float c = 1.0f / (len(a) + 1e-15f);
    return make_float3(a.x * c, a.y * c, a.z * c);
}

static FUNC_PREFIX
float3 absf3(float3 a)
{
    return make_float3(abs(a.x), abs(a.y), abs(a.z));
}

static FUNC_PREFIX
float3 sign(float3 a)
{
    return make_float3(a.x > 0.0f ? 1.0f : -1.0f,
                       a.y > 0.0f ? 1.0f : -1.0f,
                       a.z > 0.0f ? 1.0f : -1.0f);
}

static FUNC_PREFIX
float3 fmax(float3 a, float b)
{
    return make_float3(fmax(a.x, b), fmax(a.y, b), fmax(a.z, b));
}

static FUNC_PREFIX
float3 fmin(float3 a, float b)
{
    return make_float3(fmin(a.x, b), fmin(a.y, b), fmin(a.z, b));
}

static FUNC_PREFIX
float3 fmax(float3 a, float3 b)
{
    return make_float3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

static FUNC_PREFIX
float3 fmin(float3 a, float3 b)
{
    return make_float3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}

static FUNC_PREFIX
float3 lerpf(float3 a, float3 b, float c)
{
    return a + (b - a) * c;
}

static FUNC_PREFIX
bool isEqual(float3 & A, float3 & B, float eps)
{
    return (B.x - eps <= A.x && A.x <= B.x + eps &&
            B.y - eps <= A.y && A.y <= B.y + eps &&
            B.z - eps <= A.z && A.z <= B.z + eps);
}

static FUNC_PREFIX
bool isZero(float3 & A)
{
    return (A.x == 0.0f) && (A.y == 0.0f) && (A.z == 0.0f);
}

static FUNC_PREFIX
float3 clampf3(float3 a, float minx, float maxx)
{
    return make_float3(clamp(a.x, minx, maxx),
                       clamp(a.y, minx, maxx),
                       clamp(a.z, minx, maxx));
}

static FUNC_PREFIX
float3 clampf3(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x),
                       clamp(v.y, a.y, b.y),
                       clamp(v.z, a.z, b.z));
}

static FUNC_PREFIX
float3 ceilf3(float3 a)
{
    return make_float3(ceil(a.x), ceil(a.y), ceil(a.z));
}

static FUNC_PREFIX
float3 floorf3(float3 a)
{
    return make_float3(floor(a.x), floor(a.y), floor(a.z));
}

static FUNC_PREFIX
bool non_zero(float3 a)
{
    return a.x != 0.0f || a.y != 0.0f || a.z != 0.0f;
}

static FUNC_PREFIX
bool non_negative(float3 a)
{
    return a.x >= 0.0f && a.y >= 0.0f && a.z >= 0.0f;
}

static FUNC_PREFIX
float max_cmpnt(float3 a)
{
    return fmax(a.x, fmax(a.y, a.z));
}

static FUNC_PREFIX
float min_cmpnt(float3 a)
{
    return fmin(a.x, fmin(a.y, a.z));
}

static FUNC_PREFIX
float sum_cmpnt(float3 a)
{
    return a.x + a.y + a.z;
}

static FUNC_PREFIX
float3 rcp(float3 a)
{
    return 1.0f / a;
}

static FUNC_PREFIX
float3 logf3(float3 a)
{
    return make_float3(log(a.x), log(a.y), log(a.z));
}

static FUNC_PREFIX
float3 expf3(float3 a)
{
    return make_float3(expf(a.x), expf(a.y), expf(a.z));
}

static FUNC_PREFIX
float3 powf(float3 a, float e)
{
    return make_float3(pow(a.x, e), pow(a.y, e), pow(a.z, e));
}

static FUNC_PREFIX
float3 powf(float3 a, float3 e)
{
    return make_float3(pow(a.x, e.x), pow(a.y, e.y), pow(a.z, e.z));
}
