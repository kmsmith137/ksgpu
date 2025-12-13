// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

#include <complex>
#include <iostream>
#include "../include/ksgpu.hpp"
#include "../include/ksgpu/Array.hpp"

using namespace std;
using namespace ksgpu;


int main(int argc, char **argv)
{
    Array<float> arr({2,3,4}, {20,4,1}, af_uhost | af_zero);
    arr.randomize(true);
    print_array(arr);
    return 0;
}
