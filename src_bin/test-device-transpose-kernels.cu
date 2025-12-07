#include "../include/ksgpu/device_fp16.hpp"
#include "../include/ksgpu/device_transposes.hpp"

#include <cassert>
#include <iostream>
#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/cuda_utils.hpp"

using namespace std;
using namespace ksgpu;


// Operates on an array of shape __half[n][32][2];
__global__ void single_warp_half2_kernel(__half *p, int n, uint thread_stride)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    __half2 *p2 = reinterpret_cast<__half2 *> (p);
    __half2 x = p2[i];
    warp_and_half2_transpose(x, thread_stride);
    p2[i] = x;
}


// Operates on an array of shape __half[n][32][2];
__global__ void double_warp_half2_kernel(__half *p, int n, uint thread_stride)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    i = ((i >> 5) << 6) + (i & 0x1f);
    __half2 *p2 = reinterpret_cast<__half2 *> (p);
    __half2 x = p2[i];
    __half2 y = p2[i+32];
    warp_and_half2_transpose(x, y, thread_stride);
    p2[i] = x;
    p2[i+32] = y;
}


static void test_warp_half2_kernels(int n, uint thread_stride)
{
    cout << "test_warp_half2_kernels: n=" << n << ", thread_stride=" << thread_stride << endl;
    assert((n % 8) == 0);
    
    Array<float> src({n,32,2}, af_rhost | af_random);
    Array<float> dst({n,32,2}, af_rhost | af_random);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 2; k++) {
                int ix_dst = 64*i + 2*j + k;

                int j0 = j & ~thread_stride;
                int j1 = (j & thread_stride) ? 1 : 0;
                int ix_src = 64*i + 2*(j0 + thread_stride*k) + j1;

                dst.data[ix_dst] = src.data[ix_src];
            }
        }
    }

    Array<__half> garr1 = src.template convert<__half>().to_gpu();
    Array<__half> garr2 = garr1.clone();

    int nblocks = n/8;
    single_warp_half2_kernel<<<nblocks,256>>> (garr1.data, n, thread_stride);
    CUDA_PEEK("single_warp_half2_kernel launch");
    
    double_warp_half2_kernel<<<nblocks,128>>> (garr2.data, n, thread_stride);
    CUDA_PEEK("double_warp_half2_kernel launch");

    assert_arrays_equal(dst, garr1, "host", "gpu_single", {"i","t","l"});
    assert_arrays_equal(dst, garr2, "host", "gpu_double", {"i","t","l"});
}


int main(int argc, char **argv)
{
    for (int thread_stride = 1; thread_stride < 32; thread_stride *= 2)
        test_warp_half2_kernels(1024, thread_stride);

    cout << "TODO: test kernels other than warp_half2" << endl;
    return 0;
}
