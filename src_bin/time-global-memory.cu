#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/KernelTimer.hpp"
#include "../include/ksgpu/cuda_utils.hpp"

#include <iostream>

using namespace std;
using namespace ksgpu;


// -------------------------------------------------------------------------------------------------


__device__ inline void set_zero(uint &x) { x = 0.0f; }
__device__ inline void set_zero(uint2 &x) { x.x = 0.0f; x.y = 0.0f; }
__device__ inline void set_zero(uint4 &x) { x.x = 0.0f; x.y = 0.0f; x.z = 0.0f; x.w = 0.0f; }

__device__ inline void add(uint &x, uint y) { x += y; }
__device__ inline void add(uint2 &x, uint2 y) { x.x += y.x; x.y += y.y; }
__device__ inline void add(uint4 &x, uint4 y) { x.x += y.x; x.y += y.y; x.z += y.z; x.w += y.w; }

// Launch with {32,W} warps and {B} blocks.
// The 'nelts_per_warp' and 'nelts_tot' args must be powers of two.
template<typename T>
__global__ void read_gmem(T *p, uint nelts_per_warp, long nelts_tot, uint niter)
{
    T accum;
    set_zero(accum);

    uint w = blockIdx.x * blockDim.y + threadIdx.y;
    uint s0 = (ulong(w) * ulong(nelts_per_warp)) & (nelts_tot-1);
    p += s0;

    for (uint i = 0; i < niter; i++) {
        uint s = (32*i + threadIdx.x) & (nelts_per_warp-1);
        add(accum, p[s]);
    }

    // prevent compiler from optimizing out the loop.
    p[threadIdx.x] = accum;
}


static void time_gmem(long dtype_bits)
{
    // Everything must be a power of two, including nstreams.
    uint warps_per_block = 8;
    long blocks_per_launch = 4096;
    long nbytes_per_warp = 1024L * 1024L;
    long nbytes_tot = 8 * 1024L * 1024L * 1024L;
    long bw_per_launch = 128 * 1024L * 1024L * 1024L;
    int nstreams = 2;

    long dtype_bytes = dtype_bits / 8;
    long nelts_per_warp = nbytes_per_warp / dtype_bytes;
    long nbytes_per_stream = nbytes_tot / nstreams;
    long nelts_per_stream = nbytes_per_stream / dtype_bytes;
    long niter = bw_per_launch / (blocks_per_launch * warps_per_block * 32 * dtype_bytes);

    cout << "time_gmem(dtype_bits=" << dtype_bits << ")" << endl
         << "    nelts_per_warp = " << nelts_per_warp << endl
         << "    nelts_per_stream = " << nelts_per_stream << endl
         << "    niter = " << niter << endl;

    Array<unsigned char> arr({nstreams, nbytes_per_stream}, af_gpu | af_zero);

    KernelTimer kt(nstreams);

    for (int i = 0; i < 20; i++) {
        unsigned char *p = arr.data + kt.istream * nbytes_per_stream;

        if (dtype_bits == 32)
            read_gmem<uint> <<< blocks_per_launch, {32,warps_per_block}, 0, kt.stream >>> 
                ((uint *)p, nelts_per_warp, nelts_per_stream, niter);
        else if (dtype_bits == 64)
            read_gmem<uint2> <<< blocks_per_launch, {32,warps_per_block}, 0, kt.stream >>> 
                ((uint2 *)p, nelts_per_warp, nelts_per_stream, niter);
        else if (dtype_bits == 128)
            read_gmem<uint4> <<< blocks_per_launch, {32,warps_per_block}, 0, kt.stream >>> 
                ((uint4 *)p, nelts_per_warp, nelts_per_stream, niter);
        else
            throw runtime_error("Unsupported dtype_bits: " + to_string(dtype_bits));

        CUDA_PEEK("read_gmem");

        if (kt.advance()) {
            double gb_per_sec = bw_per_launch / kt.dt / 1.0e9;
            cout << "Global memory BW (GB/s): " << gb_per_sec << endl;
        }
    }
}


int main(int argc, char **argv)
{
    time_gmem(32);
    time_gmem(64);
    time_gmem(128);

    return 0;
}