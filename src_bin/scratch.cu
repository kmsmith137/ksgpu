// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

#include <complex>
#include <iostream>
// #include "../include/ksgpu.hpp"
#include "../include/ksgpu/Array.hpp"

using namespace std;
using namespace ksgpu;

template<typename T> void f()
{
    Array<T> arr({2,3}, af_uhost | af_random);
    cout << "XXX: " << arr.dtype << endl;
    print_array(arr);
}

int main(int argc, char **argv)
{
    f<__half> ();
    f<float> ();
    f<complex<double>> ();
    f<int> ();
    f<complex<long>> ();
    return 0;
}
