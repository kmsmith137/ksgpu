#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/KernelTimer.hpp"
#include "../include/ksgpu/cuda_utils.hpp"

#include <iostream>

using namespace std;
using namespace ksgpu;


__device__ inline int _reduce(int x) { return x; }
__device__ inline int _reduce(int2 x) { return x.x ^ x.y; }
__device__ inline int _reduce(int4 x) { return x.x ^ x.y ^ x.z ^ x.w; }


// 'p' should be a zeroed array of length (threads/block) * (blocks/kernel).
// Kernels should be launched with shmem_nbytes = 4 * (threads/block).
template<typename T, bool Read, bool Write>
__global__ void shmem_read_write_kernel(T *p, int niter)
{
    constexpr int N = (Read && Write) ? 2 : 1;
    extern __shared__ int shmem_[];
    T *shmem = (T *) shmem_;

    p += threadIdx.x + (blockIdx.x * blockDim.x);
    T x = *p;

    int s = threadIdx.x;
    shmem[s] = x;
    
    for (int i = 2; i < niter; i += N) {
	T y = Read ? shmem[s] : x;
	
	if constexpr (Write)
	    shmem[s] = x;

	// Doesn't actually change the value of s, but the compiler doesn't know that.
	s = s ^ (_reduce(y) & i);
    }

    *p = shmem[s];
}


template<typename T, bool Read, bool Write>
static void time_kernel(const char *name)
{
    const int ninner = 8 * 1024;
    const int threads_per_block = 16 * 32;
    const int blocks_per_kernel = 32 * 1024;
    const int nstreams = 1;
    const int nouter = 10;

    const double sm_cycles_per_second = get_sm_cycles_per_second();
    const double instructions_per_kernel = double(ninner) * double(threads_per_block/32) * double(blocks_per_kernel);
    const double shmem_tb_per_kernel = 1.0e-12 * 32.0 * sizeof(T) * instructions_per_kernel;
    const int shmem_nbytes = threads_per_block * sizeof(T);

    Array<uint8_t> arr({nstreams, blocks_per_kernel * threads_per_block * sizeof(T) }, af_gpu | af_zero);
    
    KernelTimer kt(nstreams);

    for (int i = 0; i < nouter; i++) {
	T *gmem = (T*)arr.data + (kt.istream * arr.shape[1]);
	
	shmem_read_write_kernel<T, Read, Write>
	    <<< blocks_per_kernel, threads_per_block, shmem_nbytes, kt.stream >>>
	    (gmem, ninner);
	
	CUDA_PEEK(name);

        if (kt.advance()) {
            double tb_per_sec = shmem_tb_per_kernel / kt.dt;
            double cycles = kt.dt / (instructions_per_kernel / sm_cycles_per_second);
            cout << name << " Shared memory BW (TB/s): " << tb_per_sec
                 << ", Clock cycles: " << cycles << endl;
        }
    }
}

    
int main(int argc, char **argv)
{
    time_kernel<int,true,false> ("Read shared memory (int)");
    time_kernel<int,false,true> ("Write shared memory (int)");
    time_kernel<int,true,true> ("Read/write shared memory (int)");
    
    time_kernel<int2,true,false> ("Read shared memory (int2)");
    time_kernel<int2,false,true> ("Write shared memory (int2)");
    time_kernel<int2,true,true> ("Read/write shared memory (int2)");
    
    time_kernel<int4,true,false> ("Read shared memory (int4)");
    time_kernel<int4,false,true> ("Write shared memory (int4)");
    time_kernel<int4,true,true> ("Read/write shared memory (int4)");
    
    return 0;
}
