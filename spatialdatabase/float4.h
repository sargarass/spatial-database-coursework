#include "types.h"

static FUNC_PREFIX
float4 make_float4(float x)
{
    float4 a;
    a.x = x;
    a.y = x;
    a.z = x;
    a.w = x;
    return a;
}

static FUNC_PREFIX
float4 make_float4(float3 a)
{
    float4 r;
    r.x = a.x;
    r.y = a.y;
    r.z = a.z;
    r.w = 0.0f;
    return r;
}

static FUNC_PREFIX
float4 make_float4(float3 a, float w)
{
    float4 r;
    r.x = a.x;
    r.y = a.y;
    r.z = a.z;
    r.w = w;
    return r;
}

static FUNC_PREFIX
float4 make_float4(float3 v, int i)
{
    float4 a;
    a.x = v.x;
    a.y = v.y;
    a.z = v.z;
    a.w = *((float *)&i);
    return a;
}

static FUNC_PREFIX
float4 clampf4(float4 a, float minx, float maxx)
{
    return make_float4(clamp(a.x, minx, maxx),
                       clamp(a.y, minx, maxx),
                       clamp(a.z, minx, maxx),
                       clamp(a.w, minx, maxx));
}

static FUNC_PREFIX
float4 fmax(float4 a, float b)
{
    return make_float4(fmax(a.x, b), fmax(a.y, b), fmax(a.z, b), fmax(a.w, b));
}

static FUNC_PREFIX
float4 fmin(float4 a, float b)
{
    return make_float4(fmin(a.x, b), fmin(a.y, b), fmin(a.z, b), fmin(a.w, b));
}

static FUNC_PREFIX
float4 fmax(float4 a, float4 b)
{
    return make_float4(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z), fmax(a.w, b.w));
}

static FUNC_PREFIX
float4 fmin(float4 a, float4 b)
{
    return make_float4(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z), fmin(a.w, b.w));
}

static FUNC_PREFIX
float4 powf(float4 a, float e)
{
    return make_float4(pow(a.x, e), pow(a.y, e), pow(a.z, e), pow(a.w, e));
}

static FUNC_PREFIX
float4 operator + (float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static FUNC_PREFIX
float4 operator - (float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

static FUNC_PREFIX
float4 operator * (float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

static FUNC_PREFIX
float4 operator * (float4 a, float c)
{
    return make_float4(a.x * c, a.y * c, a.z * c, a.w * c);
}

static FUNC_PREFIX
float4 operator / (float4 a, float c)
{
    return make_float4(a.x / c, a.y / c, a.z / c, a.w / c);
}

static FUNC_PREFIX
float4 operator * (float c, float4 a)
{
    return make_float4(a.x * c, a.y * c, a.z * c, a.w * c);
}

static FUNC_PREFIX
float4 & operator += (float4 & a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

static FUNC_PREFIX
float4 & operator *= (float4 & a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

static FUNC_PREFIX
float4 & operator *= (float4 & a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}
