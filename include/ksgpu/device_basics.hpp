#ifndef _KSGPU_DEVICE_DTYPE_OPS_HPP
#define _KSGPU_DEVICE_DTYPE_OPS_HPP

#include <cuda_fp16.h>

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif

// Usage: __shfl_sync(FULL_MASK, val, src_lane)
static constexpr uint FULL_MASK = 0xffffffffU;


// -------------------------------------------------------------------------------------------------
//
// device_array: workaround for the nvcc error "zero-sized variable is not allowed in device code".
//
//   float x[N];                       // valid cuda for N > 0
//   ksgpu::device_array<float,N> x;   // valid cuda for N >= 0


template<typename T, int N_>
struct device_array
{
    static constexpr int N = N_;
    static_assert(N >= 0);
    
    T data[N];
    
    __device__ __forceinline__ T &operator[](int i) { return data[i]; }
    __device__ __forceinline__ const T &operator[](int i) const { return data[i]; }
};


template<typename T>
struct device_array<T,0>
{
    static constexpr int N = 0;
};


// -------------------------------------------------------------------------------------------------
//
// dtype_ops: placeholder for future expansion

// More to come!
template<typename T> struct dtype_ops { };


template<> struct dtype_ops<float>
{
    static constexpr int simd_lanes = 1;
};


template<> struct dtype_ops<__half2>
{
    static constexpr int simd_lanes = 2;
};


} // namespace ksgpu

#endif // _KSGPU_DEVICE_DTYPE_OPS_HPP
