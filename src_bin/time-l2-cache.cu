#include <iostream>
#include "../include/ksgpu.hpp"

using namespace std;
using namespace ksgpu;


constexpr int nblocks = 64 * 1024;
constexpr int nthreads_per_block = 1024;
constexpr int num_inner_iterations = 10;
constexpr int l2_footprint_nbytes = 1024 * 1024;
constexpr int nstreams = 2;

constexpr int nsrc = l2_footprint_nbytes / sizeof(int);
constexpr int ndst = nblocks * 32;


// -------------------------------------------------------------------------------------------------


__global__ void
l2_bandwidth_kernel(int *dst, const int *src)
{
    int subtotal = 0;
    for (int i = 0; i < num_inner_iterations; i++)
        for (int j = threadIdx.x; j < nsrc; j += blockDim.x)
            subtotal += src[j];

    static_assert(nthreads_per_block == 1024);

    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 0x31;
    int dstId = (32 * blockIdx.x) + laneId;
    
    __shared__ int warp_total[32];
    warp_total[warpId] = __reduce_add_sync(0xffffffff, subtotal);

    __syncthreads();
    
    if (warpId == 0)
        dst[dstId] = warp_total[laneId];
}


int main(int argc, char **argv)
{
    double gb = nblocks * num_inner_iterations * (l2_footprint_nbytes / double(1<<30));

    cout << "L2 cache bandwidth test\n"
         << "    nblocks = " << nblocks << "\n"
         << "    nthreads_per_block = " << nthreads_per_block << "\n"
         << "    num_inner_iterations = " << num_inner_iterations << "\n"
         << "    L2 footprint (bytes) = " << l2_footprint_nbytes << "\n"
         << "    Total data read (GB) = " << gb << endl;
    
    Array<int> dst({nstreams,ndst}, af_gpu | af_zero);
    Array<int> src({nstreams,nsrc}, af_gpu | af_zero);    
    KernelTimer kt(nstreams);

    for (int i = 0; i < 20; i++) {
        l2_bandwidth_kernel<<<nblocks, nthreads_per_block, 0, kt.stream>>> 
            (dst.data + kt.istream * ndst, src.data + kt.istream * nsrc);

        if (kt.advance()) {
            double gb_per_sec = gb / kt.dt;
            cout << "    Bandwidth (GB/s) = " << gb_per_sec << endl;
        }
    }

    return 0;
}
