#include <sstream>
#include <iostream>

#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/KernelTimer.hpp"
#include "../include/ksgpu/cuda_utils.hpp"
#include "../include/ksgpu/memcpy_kernels.hpp"

using namespace std;
using namespace ksgpu;


static void time_memcpy(long nbytes, int ninner, int nouter, int nstreams=1)
{
    Array<char> adst({nstreams,nbytes}, af_zero | af_gpu);
    Array<char> asrc({nstreams,nbytes}, af_zero | af_gpu);

    stringstream ss;
    ss << "memcpy(nbytes=" << nbytes << ")";
    string name = ss.str();

    double gb_per_kernel = 2.0e-9 * ninner * nbytes;

    KernelTimer kt(nouter, nstreams);

    while (kt.next()) {
        char *d = adst.data + kt.istream*nbytes;
        char *s = asrc.data + kt.istream*nbytes;

        for (int j = 0; j < ninner; j++)
            launch_memcpy_kernel(d, s, nbytes, kt.stream);
        
        CUDA_PEEK("launch_memcpy_kernel");

        if (kt.warmed_up) {
            double gb_per_sec = gb_per_kernel / kt.dt;
            cout << name << " GB/s: " << gb_per_sec << endl;
        }
    }
}


static void time_memcpy_2d(long dpitch, long spitch, long width, long height, int ninner, int nouter, int nstreams=1)
{
    long dst_nbytes = height * dpitch;
    long src_nbytes = height * spitch;
    
    Array<char> adst({nstreams,dst_nbytes}, af_zero | af_gpu);
    Array<char> asrc({nstreams,src_nbytes}, af_zero | af_gpu);

    stringstream ss;
    ss << "memcpy_2d(dpitch=" << dpitch << ", spitch=" << spitch << ", width=" << width << ", height=" << height << ")";
    string name = ss.str();

    double gb_per_kernel = 2.0e-9 * ninner * width * height;

    KernelTimer kt(nouter, nstreams);

    while (kt.next()) {
        char *d = adst.data + kt.istream * dst_nbytes;
        char *s = asrc.data + kt.istream * src_nbytes;

        for (int j = 0; j < ninner; j++)
            launch_memcpy_2d_kernel(d, dpitch, s, spitch, width, height, kt.stream);
        
        CUDA_PEEK("launch_memcpy_2d_kernel");

        if (kt.warmed_up) {
            double gb_per_sec = gb_per_kernel / kt.dt;
            cout << name << " GB/s: " << gb_per_sec << endl;
        }
    }
}


int main(int argc, char **argv)
{
    long gb4 = 4L * 1024L * 1024L * 1024L;
    
    time_memcpy(gb4, 20, 10);

    // (dpitch, spitch, width, height)
    time_memcpy_2d(65536+128, 65536+1024, 65536, 65536, 20, 10);
    time_memcpy_2d(128, 128, 128, 32L * 1024L * 1024L, 20, 10);
    time_memcpy_2d(256, 256, 128, 32L * 1024L * 1024L, 20, 10);
    time_memcpy_2d(gb4, gb4, gb4, 1, 20, 10);
    
    return 0;
}
