// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

#include <complex>
#include <iostream>
#include "../include/ksgpu.hpp"
// #include "../include/ksgpu/Array.hpp"

using namespace std;
using namespace ksgpu;

template<typename T>
static void f()
{
    Dtype d = Dtype::native<T>();
    Dtype d2 = Dtype::from_str(d.str(), false);
    cout << d << " -> " << d2 << endl;
}

template<typename T>
static void g()
{
    f<T> ();
    f<complex<T>> ();
}

int main(int argc, char **argv)
{
    g<int> ();
    g<uint> ();
    g<long> ();
    g<ulong> ();
    g<short> ();
    g<ushort> ();
    g<char> ();
    g<unsigned char> ();
    g<float> ();
    g<double> ();
    g<__half> ();
    
    return 0;
}
