#ifndef _KSGPU_DEVICE_TRANSPOSES_HPP
#define _KSGPU_DEVICE_TRANSPOSES_HPP

// constexpr_is_divisible(), 
#include "constexpr_functions.hpp"

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// Decided to put this here, for lack of a better place.
// Usage: __shfl_sync(FULL_MASK, val, src_lane)

static constexpr uint FULL_MASK = 0xffffffffU;


// Usage: to transpose thread bit t_3 with a register bit, do
//   warp_transpose(x,y,8);  // not warp_transpose(x,y,3)!
//
// Warning: the 'thread_stride' arg must be a power of 2, but this
// isn't checked!

template<typename T>
__device__ inline void warp_transpose(T &x, T &y, uint thread_stride)
{
    bool upper = (threadIdx.x & thread_stride);
    T src = upper ? x : y;
    T z = __shfl_sync(FULL_MASK, src, threadIdx.x ^ thread_stride);
    x = upper ? z : x;
    y = upper ? y : z;
}


// Usage: suppose we have 16 registers (4 register bits), and we want
// to transpose thread bit t_3 with register bit r_2. Then:
//
//   float x[16];
//   multi_warp_transpose<4,16> (x,8);   // not (x,3)!
//
// Warning: the 'thread_stride' arg must be a power of 2, but this
// isn't checked!

template<int RegisterStride, int NumRegisters, typename T>
__device__ inline void multi_warp_transpose(T x[NumRegisters], uint thread_stride)
{
    static_assert(NumRegisters > 0);
    static_assert(RegisterStride > 0);
    static_assert(constexpr_is_divisible(NumRegisters, 2*RegisterStride));

    constexpr int R = RegisterStride;
    constexpr int S = NumRegisters / (2*R);

    #pragma unroll
    for (int s = 0; s < S; s++) {
	#pragma unroll
	for (int r = 0; r < R; r++)
	    warp_transpose(x[(2*s)*R + r], x[(2*s+1)*R + r], thread_stride);
    }
}


} // namespace ksgpu

#endif // _KSGPU_DEVICE_TRANSPOSES_HPP
