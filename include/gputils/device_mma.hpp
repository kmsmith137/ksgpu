#ifndef _KSGPU_DEVICE_MMA_HPP
#define _KSGPU_DEVICE_MMA_HPP

// Autogenerated by generate_device_mma_hpp.py
//
// Reference for matrix shapes:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape
//
// Reference for PTX instruction syntax:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma

#include <cuda_fp16.h>

namespace ksgpu {


// D = A*B + C
__device__ __forceinline__
void mma_f16_m16_n8_k8(__half2 d[2], const __half2 a[2], const __half2 b[1], const __half2 c[2])
{
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3}, "
        "{%4}, "
        "{%5, %6};" :
        "=r" (*(uint *) &d[0]), "=r" (*(uint *) &d[1]) :
        "r" (*(const uint *) &a[0]), "r" (*(const uint *) &a[1]),
        "r" (*(const uint *) &b[0]),
        "r" (*(const uint *) &c[0]), "r" (*(const uint *) &c[1])
    );
}


// D = A*B + C
__device__ __forceinline__
void mma_f16_m16_n8_k16(__half2 d[2], const __half2 a[4], const __half2 b[2], const __half2 c[2])
{
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};" :
        "=r" (*(uint *) &d[0]), "=r" (*(uint *) &d[1]) :
        "r" (*(const uint *) &a[0]), "r" (*(const uint *) &a[1]), "r" (*(const uint *) &a[2]), "r" (*(const uint *) &a[3]),
        "r" (*(const uint *) &b[0]), "r" (*(const uint *) &b[1]),
        "r" (*(const uint *) &c[0]), "r" (*(const uint *) &c[1])
    );
}


// D = A*B + C
__device__ __forceinline__
void mma_s4_m8_n8_k32(int d[2], const int a[1], const int b[1], const int c[2])
{
    asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};" :
        "=r" (d[0]), "=r" (d[1]) :
        "r" (a[0]),
        "r" (b[0]),
        "r" (c[0]), "r" (c[1])
    );
}


// D = A*B + C
__device__ __forceinline__
void mma_s4_m16_n8_k32(int d[4], const int a[2], const int b[1], const int c[4])
{
    asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};" :
        "=r" (d[0]), "=r" (d[1]), "=r" (d[2]), "=r" (d[3]) :
        "r" (a[0]), "r" (a[1]),
        "r" (b[0]),
        "r" (c[0]), "r" (c[1]), "r" (c[2]), "r" (c[3])
    );
}


// D = A*B + C
__device__ __forceinline__
void mma_s4_m16_n8_k64(int d[4], const int a[4], const int b[2], const int c[4])
{
    asm("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};" :
        "=r" (d[0]), "=r" (d[1]), "=r" (d[2]), "=r" (d[3]) :
        "r" (a[0]), "r" (a[1]), "r" (a[2]), "r" (a[3]),
        "r" (b[0]), "r" (b[1]),
        "r" (c[0]), "r" (c[1]), "r" (c[2]), "r" (c[3])
    );
}


// D = A*B + C
__device__ __forceinline__
void mma_s8_m8_n8_k16(int d[2], const int a[1], const int b[1], const int c[2])
{
    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};" :
        "=r" (d[0]), "=r" (d[1]) :
        "r" (a[0]),
        "r" (b[0]),
        "r" (c[0]), "r" (c[1])
    );
}


// D = A*B + C
__device__ __forceinline__
void mma_s8_m16_n8_k16(int d[4], const int a[2], const int b[1], const int c[4])
{
    asm("mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};" :
        "=r" (d[0]), "=r" (d[1]), "=r" (d[2]), "=r" (d[3]) :
        "r" (a[0]), "r" (a[1]),
        "r" (b[0]),
        "r" (c[0]), "r" (c[1]), "r" (c[2]), "r" (c[3])
    );
}


// D = A*B + C
__device__ __forceinline__
void mma_s8_m16_n8_k32(int d[4], const int a[4], const int b[2], const int c[4])
{
    asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};" :
        "=r" (d[0]), "=r" (d[1]), "=r" (d[2]), "=r" (d[3]) :
        "r" (a[0]), "r" (a[1]), "r" (a[2]), "r" (a[3]),
        "r" (b[0]), "r" (b[1]),
        "r" (c[0]), "r" (c[1]), "r" (c[2]), "r" (c[3])
    );
}


// D = A*B + C
__device__ __forceinline__
void mma_b1_m8_n8_k128(int d[2], const int a[1], const int b[1], const int c[2])
{
    asm("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};" :
        "=r" (d[0]), "=r" (d[1]) :
        "r" (a[0]),
        "r" (b[0]),
        "r" (c[0]), "r" (c[1])
    );
}


// D = A*B + C
template<uint F>
__device__ __forceinline__
void mma_sp_f16_m16_n8_k16(__half2 d[2], const __half2 a[2], const __half2 b[2], const __half2 c[2], uint e)
{
    asm(
#if CUDART_VERSION >= 12050
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
#else
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
#endif
        "{%0, %1}, "
        "{%2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "%8, "
        "%9;" :
        "=r" (*(uint *) &d[0]), "=r" (*(uint *) &d[1]) :
        "r" (*(const uint *) &a[0]), "r" (*(const uint *) &a[1]),
        "r" (*(const uint *) &b[0]), "r" (*(const uint *) &b[1]),
        "r" (*(const uint *) &c[0]), "r" (*(const uint *) &c[1]),
        "r" (e),
        "n" (F)
    );
}


// D = A*B + C
template<uint F>
__device__ __forceinline__
void mma_sp_f16_m16_n8_k32(__half2 d[2], const __half2 a[4], const __half2 b[4], const __half2 c[2], uint e)
{
    asm(
#if CUDART_VERSION >= 12050
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
#else
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
#endif
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7, %8, %9}, "
        "{%10, %11}, "
        "%12, "
        "%13;" :
        "=r" (*(uint *) &d[0]), "=r" (*(uint *) &d[1]) :
        "r" (*(const uint *) &a[0]), "r" (*(const uint *) &a[1]), "r" (*(const uint *) &a[2]), "r" (*(const uint *) &a[3]),
        "r" (*(const uint *) &b[0]), "r" (*(const uint *) &b[1]), "r" (*(const uint *) &b[2]), "r" (*(const uint *) &b[3]),
        "r" (*(const uint *) &c[0]), "r" (*(const uint *) &c[1]),
        "r" (e),
        "n" (F)
    );
}


} // namespace ksgpu

#endif // _KSGPU_DEVICE_MMA_HPP
