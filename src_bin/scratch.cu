// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

#include <complex>
#include <iostream>
#include "../include/ksgpu.hpp"
// #include "../include/ksgpu/Array.hpp"

using namespace std;
using namespace ksgpu;

int main(int argc, char **argv)
{
    vector<long> dst_shape, src_shape, src_strides;
    
    for (int i = 0; i < 50; i++) {
	make_random_reshape_compatible_shapes(dst_shape, src_shape, src_strides);
	cout << "dst_shape = " << tuple_str(dst_shape)
	     << ", src_shape = " << tuple_str(src_shape)
	     << ", src_strides = " << tuple_str(src_strides)
	     << endl;
    }
    
    return 0;
}
