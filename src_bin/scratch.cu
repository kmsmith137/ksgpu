// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

//#include <iostream>
// #include "../include/ksgpu.hpp"
#include "../include/ksgpu/xassert.hpp"
#include "../include/ksgpu/Array.hpp"

using namespace std;
using namespace ksgpu;


int main(int argc, char **argv)
{
    int m = 2;
    int n = 3;
    int p = 5;
    Array<int> arr({2,3,4}, af_uhost);
    xassert_shape_eq(arr, ({m,n,p}));
    return 0;
}
