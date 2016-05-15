#include "types.h"

static FUNC_PREFIX
float2 make_float2(float x)
{
    float2 a;
    a.x = x;
    a.y = x;
    return a;
}

static FUNC_PREFIX
float sqr(float x) {
    return x * x;
}

static FUNC_PREFIX
float2 & operator += (float2 & a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

static FUNC_PREFIX
bool operator == (float2 &a, float2 &b) {
    return a.x == b.x && a.y == b.y;
}

static FUNC_PREFIX
bool operator != (float2 &a, float2 &b) {
    return a.x != b.x || a.y != b.y;
}

static FUNC_PREFIX
float2 & operator *= (float2 & a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

static FUNC_PREFIX
float2 operator - (float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

static FUNC_PREFIX
float2 operator + (float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

static FUNC_PREFIX
float2 operator * (float2 a, float c)
{
    return make_float2(a.x * c, a.y * c);
}

static FUNC_PREFIX
float2 operator * (float c, float2 a)
{
    return make_float2(a.x * c, a.y * c);
}

static FUNC_PREFIX
float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}


static FUNC_PREFIX
float len(float2 a)
{
    return sqrt(a.x * a.x + a.y * a.y);
}

static FUNC_PREFIX
float lenSqr(float2 a, float2 b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

static FUNC_PREFIX
float2 norma(float2 a)
{
    float c = 1.0f / (len(a) + 1e-15f);
    return make_float2(a.x * c, a.y * c);
}


static FUNC_PREFIX
float2 fmin(float2 a, float2 b) {
    float2 res;
    res.x = (a.x > b.x)? b.x : a.x;
    res.y = (a.y > b.y)? b.y : a.y;
    return res;
}

static FUNC_PREFIX
float2 fmax(float2 a, float2 b) {
    float2 res;
    res.x = (a.x < b.x)? b.x : a.x;
    res.y = (a.y < b.y)? b.y : a.y;
    return res;
}
