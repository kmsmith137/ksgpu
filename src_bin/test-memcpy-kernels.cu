#include "../include/ksgpu/memcpy_kernels.hpp"

#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/cuda_utils.hpp"
#include "../include/ksgpu/test_utils.hpp"
#include "../include/ksgpu/string_utils.hpp"

#include <iostream>


using namespace std;
using namespace ksgpu;


static void test_memcpy_kernel(long nbytes)
{
    cout << "test_memcpy_kernel(nbytes=" << nbytes << ")" << endl;

    long nelts = nbytes >> 2;
    long nguard = 128;
    
    xassert((nbytes % 128) == 0);
    
    Array<int> hsrc({nelts}, af_rhost | af_random);
    Array<int> hdst({nelts + 2*nguard}, af_rhost | af_random);

    Array<int> gsrc = hsrc.to_gpu();
    Array<int> gdst = hdst.to_gpu();
    
    launch_memcpy_kernel(gdst.data + nguard, gsrc.data, nbytes);
    CUDA_PEEK("launch_memcpy_kernel");

    memcpy(hdst.data + nguard, hsrc.data, nbytes);
    assert_arrays_equal(hdst, gdst.to_host(), "hdst", "gdst", {"i"});
}


static void test_memcpy_kernel_2d(long dpitch, long spitch, long width, long height)
{
    cout << "test_memcpy_kernel_2d(dpitch=" << dpitch << ", spitch=" << spitch
         << ", width=" << width << ", height=" << height << ")" << endl;
        
    // dpitch/spitch/width are in bytes (matching launch_memcpy_2d_kernel),
    // but Array<int> lengths and pointer offsets are in ints -- careful with
    // units below.
    xassert_divisible(dpitch, 128);
    xassert_divisible(spitch, 128);
    xassert_divisible(width, 128);
    xassert_le(width, dpitch);
    xassert_le(width, spitch);

    // Guard regions (in ints) before and after the 2-d destination region,
    // checked by assert_arrays_equal() to catch out-of-bounds writes:
    // 4 rows' worth of each pitch, plus 128 bytes.
    long nguard = dpitch + spitch + 32;
    Array<int> hsrc({height * (spitch >> 2)}, af_rhost | af_random);
    Array<int> hdst({height * (dpitch >> 2) + 2*nguard}, af_rhost | af_random);

    Array<int> gsrc = hsrc.to_gpu();
    Array<int> gdst = hdst.to_gpu();

    launch_memcpy_2d_kernel(gdst.data + nguard, dpitch, gsrc.data, spitch, width, height);
    CUDA_PEEK("launch_memcpy_kernel_2d");

    for (long r = 0; r < height; r++) {
        memcpy(hdst.data + nguard + r * (dpitch >> 2),
               hsrc.data + r * (spitch >> 2),
               width);
    }
    
    assert_arrays_equal(hdst, gdst.to_host(), "hdst", "gdst", {"i"});
}


static void test_random_memcpy_kernel(long nb_max)
{
    long nb = 128 * rand_int(nb_max/256, nb_max/128+1);
    test_memcpy_kernel(nb);
}


static void test_random_memcpy_kernel_2d(long nb_max)
{
    long wh_target = rand_int(nb_max/2, nb_max+1);
    long width = 128 * long(exp(rand_uniform() * log(wh_target/128.)));
    long height = wh_target / width;
    
    long maxpitch = long(nb_max/height);
    long pmax = (maxpitch - width) / 128;
    long dpitch = width + 128 * rand_int(0,pmax+1);
    long spitch = width + 128 * rand_int(0,pmax+1);
    
    test_memcpy_kernel_2d(dpitch, spitch, width, height);    
}


int main(int argc, char **argv)
{
    ksgpu::seed_default_rng(137);   // reproducible run; remove for full randomness

    for (long nb = 128; nb < 64*1024; nb += 128)
        test_memcpy_kernel(nb);

    for (long nb_max = 128*1024; nb_max <= 4L * 1024L * 1024L * 1024L; nb_max *= 2) {
        cout << "\nnb_max = " << nb_max << " (" << nbytes_to_str(nb_max) << ")" << endl;
        
        test_random_memcpy_kernel(nb_max);
        for (int i = 0; i < 4; i++)
            test_random_memcpy_kernel_2d(nb_max);
    }
    
    return 0;
}
