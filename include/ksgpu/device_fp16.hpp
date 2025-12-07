#ifndef _KSGPU_DEVICE_FP16_HPP
#define _KSGPU_DEVICE_FP16_HPP

#include <cuda_fp16.h>

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// half4_load, half4_store: 64-bit (4 × float16) load/store using uint2
//
// These functions provide a clean interface for loading/storing 4 half values
// using 64-bit transactions. The uint2 type is reinterpreted as 2 × __half2.


__device__ __forceinline__ void half4_load(const uint2 *p, __half2 &x, __half2 &y)
{
    uint2 u = *p;
    x = *reinterpret_cast<__half2*>(&u.x);
    y = *reinterpret_cast<__half2*>(&u.y);
}


__device__ __forceinline__ void half4_store(uint2 *p, __half2 x, __half2 y)
{
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&x);
    u.y = *reinterpret_cast<uint32_t*>(&y);
    *p = u;
}


// -------------------------------------------------------------------------------------------------
//
// half8_load, half8_store: 128-bit (8 × float16) load/store using uint4
//
// These functions provide a clean interface for loading/storing 8 half values
// using 128-bit transactions. The uint4 type is reinterpreted as 4 × __half2.


__device__ __forceinline__ void half8_load(const uint4 *p, __half2 &x, __half2 &y, __half2 &z, __half2 &w)
{
    uint4 u = *p;
    x = *reinterpret_cast<__half2*>(&u.x);
    y = *reinterpret_cast<__half2*>(&u.y);
    z = *reinterpret_cast<__half2*>(&u.z);
    w = *reinterpret_cast<__half2*>(&u.w);
}


__device__ __forceinline__ void half8_store(uint4 *p, __half2 x, __half2 y, __half2 z, __half2 w)
{
    uint4 u;
    u.x = *reinterpret_cast<uint32_t*>(&x);
    u.y = *reinterpret_cast<uint32_t*>(&y);
    u.z = *reinterpret_cast<uint32_t*>(&z);
    u.w = *reinterpret_cast<uint32_t*>(&w);
    *p = u;
}


} // namespace ksgpu

#endif // _KSGPU_DEVICE_FP16_HPP

