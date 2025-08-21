#ifndef _KSGPU_DEVICE_TRANSPOSES_HPP
#define _KSGPU_DEVICE_TRANSPOSES_HPP

#include <cuda_fp16.h>              // __half2
#include "device_basics.hpp"        // FULL_MASK
#include "constexpr_functions.hpp"  // constexpr_is_divisible

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Warp transpose


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


// -------------------------------------------------------------------------------------------------
//
// Half2 "local" transpose
//
// Given __half2 variables a = [a0,a1] and b = [b0,b1]:
//
//    __lows2half2() returns [a0,b0]
//    __highs2half2() returns [a1,b1]
//    f16_align() returns [a1,b0]
//    f16_blend() returns [a0,b1]
//    half2_transpose() replaces (a,b) by ([a0,b0],[a1,b1]).


__device__ __forceinline__
__half2 f16_align(__half2 a, __half2 b)
{
    __half2 d;
    
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
    // Note: I chose to use prmt.b32.f4e(d,a,b,2) but I think prmt.b32(d,a,b,0x5432) is equivalent.
    
    asm("prmt.b32.f4e %0, %1, %2, %3;" :
	"=r" (*(unsigned int *) &d) :
	"r" (*(const unsigned int *) &a),
	"r" (*(const unsigned int *) &b),
	"n"(2)
    );

    return d;
}

__device__ __forceinline__
__half2 f16_blend(__half2 a, __half2 b)
{
    __half2 d;
        
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
    asm("prmt.b32 %0, %1, %2, %3;" :
	"=r" (*(unsigned int *) &d) :
	"r" (*(const unsigned int *) &a),
	"r" (*(const unsigned int *) &b),
	"n"(0x7610)
    );

    return d;
}


__device__ inline void half2_transpose(__half2 &x, __half2 &y)
{
    __half2 xnew = __lows2half2(x,y);
    __half2 ynew = __highs2half2(x,y);
    
    x = xnew;
    y = ynew;
}


template<int RegisterStride, int NumRegisters>
__device__ inline void multi_half2_transpose(__half2 x[NumRegisters])
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
	    half2_transpose(x[(2*s)*R + r], x[(2*s+1)*R + r]);
    }    
}


// -------------------------------------------------------------------------------------------------
//
// "Warp and half2" transpose


__device__ __forceinline__
__half2 f16_perm(__half2 a, __half2 b, unsigned int s)
{
    // This is just __byte_perm(), but hacked up with casts so that the inputs/outputs have dtype __half2.
    //
    // The "selector" s should have the form 0xMMNN, where:
    //   - NN corresponds to the low 16 bits of the output __half2.
    //   - MM corresponds to the high 16 bits of the output __half2.
    //   - Each of MM, NN is {10, 32, 54, 76} for {a0, a1, b0, b1} respectively.
    //
    // Examples:
    //   - [a0,b0] can be returned with either __lows2half2() or f16_perm(0x5410).
    //   - [a1,b1] can be returned with either __highs2half2() or f16_perm(0x7632).
    //   - [a1,b0] can be returned with either f16_align() or f16_perm(0x5432).
    //   - [a0,b1] can be returned with either f16_blend() or f16_perm(0x7610).
    
    __half2 d;

    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
    asm("prmt.b32 %0, %1, %2, %3;" :
	"=r" (*(unsigned int *) &d) :
	"r" (*(const unsigned int *) &a),
	"r" (*(const unsigned int *) &b),
	"r" (s)
    );

    return d;
}


__device__ inline void warp_and_half2_transpose(__half2 &x, uint thread_stride)
{
    bool upper = (threadIdx.x & thread_stride);
    uint s = upper ? 0x3276 : 0x5410;
    __half2 y = __shfl_sync(FULL_MASK, x, threadIdx.x ^ thread_stride);
    x = f16_perm(x,y,s);   // upper ? [y1,x1] : [x0,y0]
}


// Equivalent to:
//
//   _warp_and_half2_transpose(x, thread_stride);
//   _warp_and_half2_transpose(y, thread_stride);
//
// but uses one call to __shfl_sync() instead of two.

__device__ inline void warp_and_half2_transpose(__half2 &x, __half2 &y, uint thread_stride)
{
    bool upper = (threadIdx.x & thread_stride);
    uint s1 = upper ? 0x5410 : 0x7632;
    uint s2 = upper ? 0x3254 : 0x5410;
    uint s3 = upper ? 0x3276 : 0x7610;
    
    __half2 z = f16_perm(x,y,s1);  // upper ? [x0,y0] : [x1,y1]
    z = __shfl_sync(FULL_MASK, z, threadIdx.x ^ thread_stride);
    x = f16_perm(x,z,s2);          // upper ? [z0,x1] : [x0,z0]
    y = f16_perm(y,z,s3);          // upper ? [z1,y1] : [y0,z1]
}


} // namespace ksgpu

#endif // _KSGPU_DEVICE_TRANSPOSES_HPP
