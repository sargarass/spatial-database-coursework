#pragma once
#include "types.h"
#include "float2.h"
#include "float3.h"
#include "float4.h"

__device__
uint getGlobalIdx3DZXY();
FUNC_PREFIX
dim3 gridConfigure(uint64_t problemSize, dim3 block);
