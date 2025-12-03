#include <iostream>
#include <cuda_fp16.h>

#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/KernelTimer.hpp"
#include "../include/ksgpu/cuda_utils.hpp"

using namespace std;
using namespace ksgpu;


// -------------------------------------------------------------------------------------------------
//
// Some boilerplate, to use the same launchable kernel for f16 and i16.


struct local_transpose_f16
{
    using Dtype16 = __half;
    using Dtype32 = __half2;
    
    static __device__ __forceinline__ void do_transpose(__half2 &x, __half2 &y)
    {
        __half2 xnew = __lows2half2(x, y);
        __half2 ynew = __highs2half2(x, y);
        x = xnew;
        y = ynew;
    }
};


struct local_transpose_i16
{
    using Dtype16 = ushort;
    using Dtype32 = uint;
    
    static __device__ __forceinline__ void do_transpose(uint &x, uint &y)
    {
        uint xnew = __byte_perm(x, y, 0x5410);
        uint ynew = __byte_perm(x, y, 0x7632);
        x = xnew;
        y = ynew;
    }
};


// -------------------------------------------------------------------------------------------------


template<class T, typename D = typename T::Dtype32>
__global__ void local_transpose_kernel(D *dst, const D *src, int niter)
{
    int it = threadIdx.x;
    int nt = blockDim.x;

    src += blockIdx.x * 8*nt;
    dst += blockIdx.x * 8*nt;
    
    D x0 = src[it];
    D x1 = src[it + nt];
    D x2 = src[it + 2*nt];
    D x3 = src[it + 3*nt];
    D x4 = src[it + 4*nt];
    D x5 = src[it + 5*nt];
    D x6 = src[it + 6*nt];
    D x7 = src[it + 7*nt];
        
    for (int i = 0; i < niter; i++) {
        T::do_transpose(x0, x1);
        T::do_transpose(x2, x3);
        T::do_transpose(x4, x5);
        T::do_transpose(x6, x7);
        
        T::do_transpose(x0, x2);
        T::do_transpose(x1, x3);
        T::do_transpose(x4, x6);
        T::do_transpose(x5, x7);
        
        T::do_transpose(x0, x4);
        T::do_transpose(x1, x5);
        T::do_transpose(x2, x6);
        T::do_transpose(x3, x7);
    }

    dst[it] = x0;
    dst[it + nt] = x1;
    dst[it + 2*nt] = x2;
    dst[it + 3*nt] = x3;
    dst[it + 4*nt] = x4;
    dst[it + 5*nt] = x5;
    dst[it + 6*nt] = x6;
    dst[it + 7*nt] = x7;
}


template<typename T>
void time_local_transpose_kernel(const char *name)
{
    using D16 = typename T::Dtype16;
    using D32 = typename T::Dtype32;
    
    static_assert(sizeof(D16) == 2);
    static_assert(sizeof(D32) == 4);
    
    const int nblocks = 82 * 84;
    const int nthreads_per_block = 1024;
    const int nouter = 10;
    const int nstreams = 2;
    const int ninner = 256 * 1024;
    const double tera_transposes_per_kernel = double(ninner) * nblocks * nthreads_per_block * 12 / pow(2.,40.);

    const int arr_size = nblocks * nthreads_per_block * 16;
    Array<D16> dst({nstreams,arr_size}, af_zero | af_gpu);
    Array<D16> src({nstreams,arr_size}, af_zero | af_gpu);

    KernelTimer kt(nstreams);

    for (int i = 0; i < nouter; i++) {
        D16 *d = dst.data + kt.istream*arr_size;
        D16 *s = src.data + kt.istream*arr_size;

        local_transpose_kernel<T> <<< nblocks, nthreads_per_block, 0, kt.stream >>> ((D32 *)d, (D32 *)s, ninner);
        CUDA_PEEK(name);

        if (kt.advance()) {
            double tera_per_sec = tera_transposes_per_kernel / kt.dt;
            cout << name << " teratransposes / sec: " << tera_per_sec << endl;
        }
    }
}


int main(int argc, char **argv)
{
    cout << "** FIXME: local_transpose_f16() timings are misleadingly optimistic! **" << endl;
    time_local_transpose_kernel<local_transpose_f16> ("local_transpose_f16");
    time_local_transpose_kernel<local_transpose_i16> ("local_transpose_i16");
    return 0;
}
