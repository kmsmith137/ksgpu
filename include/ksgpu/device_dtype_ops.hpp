#ifndef _KSGPU_DEVICE_DTYPE_OPS_HPP
#define _KSGPU_DEVICE_DTYPE_OPS_HPP

#include <cuda_fp16.h>

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// More to come!
template<typename T> struct dtype_ops { };


template<> struct dtype_ops<float>
{
    static constexpr int simd_width = 1;
};


template struct dtype_ops<__half2>
{
    static constexpr int simd_width = 2;
};


} // namespace ksgpu

#endif // _KSGPU_DEVICE_DTYPE_OPS_HPP
