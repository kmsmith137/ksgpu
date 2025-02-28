// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

//#include <iostream>
// #include "../include/ksgpu.hpp"
#include "../include/ksgpu/test_utils.hpp"

using namespace std;
using namespace ksgpu;


int main(int argc, char **argv)
{
    Array<__half> a({2,3,4}, af_uhost | af_random);
    print_array(a, {"ax0","ax1","ax2"});
    return 0;
}
