#include <iostream>
#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/KernelTimer.hpp"

using namespace std;
using namespace ksgpu;


// -------------------------------------------------------------------------------------------------


__global__ void shfl_int_kernel(int4 *buf, const uint4 *lane, int niter)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int4 a = buf[s];
    uint4 l = lane[s];
    
    for (int i = 0; i < niter; i++) {
	int b, c, d, e;

	// Eight 32-bit shuffles per iteration.
	// Probably overkill, but I wanted to make sure that dependencies
	// and register bank conflicts aren't issues.
	
	b = __shfl_sync(~0u, a.x, l.x);
	c = __shfl_sync(~0u, a.y, l.y);
	d = __shfl_sync(~0u, a.z, l.z);
	e = __shfl_sync(~0u, a.w, l.w);
	
	a.x = __shfl_sync(~0u, b, l.x);
	a.y = __shfl_sync(~0u, c, l.y);
	a.z = __shfl_sync(~0u, d, l.z);
	a.w = __shfl_sync(~0u, e, l.w);
    }

    buf[s] = a;
}


__global__ void shfl_long_kernel(long2 *buf, const uint2 *lane, int niter)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    long2 a = buf[s];
    uint2 l = lane[s];
    
    for (int i = 0; i < niter; i++) {
	long b, c;

	// Four 64-bit shuffles per iteration.
	
	b = __shfl_sync(~0u, a.x, l.x);
	c = __shfl_sync(~0u, a.y, l.y);
	
	a.x = __shfl_sync(~0u, b, l.x);
	a.y = __shfl_sync(~0u, c, l.y);
    }

    buf[s] = a;
}


static void time_shfl_int(int nblocks, int nthreads, int nstreams, int nouter, int ninner)
{
    int s = nblocks * nthreads;
    Array<int> bufs({nstreams,s*4}, af_zero | af_gpu);
    Array<uint> lanes({nstreams,s*4}, af_zero | af_gpu);

    double thread_cycles_per_second = 32. * get_sm_cycles_per_second();
    double shuffles_per_kernel = 8. * double(nblocks) * double(nthreads) * double(ninner);  // 8 shuffles per iteration (see above)

    KernelTimer kt(nouter, nstreams);

    while (kt.next()) {
        int4 *buf = (int4*)bufs.data + kt.istream * s;
        uint4 *lane = (uint4*)lanes.data + kt.istream * s;
        
        shfl_int_kernel <<< nblocks, nthreads, 0, kt.stream >>>
            (buf, lane, ninner);

        CUDA_PEEK("shfl_kernel launch");

        if (kt.warmed_up) {
            double cycles = kt.dt / (shuffles_per_kernel / thread_cycles_per_second);
            cout << "32-bit warp shuffle Clock cycles: " << cycles << endl;
        }
    }
}


static void time_shfl_long(int nblocks, int nthreads, int nstreams, int nouter, int ninner)
{
    int s = nblocks * nthreads;
    Array<long> bufs({nstreams,s*2}, af_zero | af_gpu);
    Array<uint> lanes({nstreams,s*2}, af_zero | af_gpu);

    double thread_cycles_per_second = 32. * get_sm_cycles_per_second();
    double shuffles_per_kernel = 4. * double(nblocks) * double(nthreads) * double(ninner);  // 4 shuffles per iteration (see above)

    KernelTimer kt(nouter, nstreams);

    while (kt.next()) {
        long2 *buf = (long2*)bufs.data + kt.istream * s;
        uint2 *lane = (uint2*)lanes.data + kt.istream * s;
        
        shfl_long_kernel <<< nblocks, nthreads, 0, kt.stream >>>
            (buf, lane, ninner);

        CUDA_PEEK("shfl_kernel launch");

        if (kt.warmed_up) {
            double cycles = kt.dt / (shuffles_per_kernel / thread_cycles_per_second);
            cout << "64-bit warp shuffle Clock cycles: " << cycles << endl;
        }
    }
}


// -------------------------------------------------------------------------------------------------


__global__ void reduce_add_kernel(int *dst, const int *src, int niter)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int x = src[s];
    
    for (int i = 0; i < niter; i++)
        x ^= __reduce_add_sync(0xffffffff, x);

    dst[s] = x;
}


static void time_reduce_add(int nblocks, int nthreads, int nstreams, int nouter, int ninner)
{
    int s = nblocks * nthreads;
    Array<int> dst_arr({nstreams,s}, af_zero | af_gpu);
    Array<int> src_arr({nstreams,s}, af_zero | af_gpu);

    // gigareduces per callback
    double thread_cycles_per_second = 32. * get_sm_cycles_per_second();
    double reduces_per_kernel = double(s) * double(ninner);

    KernelTimer kt(nouter, nstreams);

    while (kt.next()) {
        int *dst = dst_arr.data + kt.istream * s;
        int *src = src_arr.data + kt.istream * s;
        
        reduce_add_kernel <<< nblocks, nthreads, 0, kt.stream >>>
            (dst, src, ninner);

        CUDA_PEEK("reduce_add_kernel");

        if (kt.warmed_up) {
            double cycles = kt.dt / (reduces_per_kernel / thread_cycles_per_second);
            cout << "reduce_add<int> Clock cycles: " << cycles << endl;
        }
    }
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    // (nblocks, nthreads, nstreams, ncallbacks, niter)
    time_shfl_int(1000, 128, 2, 10, 1000000); 
    time_shfl_long(1000, 128, 2, 10, 1000000); 
    time_reduce_add(1000, 128, 2, 10, 3000000);
    return 0;
}
