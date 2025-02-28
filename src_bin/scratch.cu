// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

#include <complex>
#include <iostream>
// #include "../include/ksgpu.hpp"
#include "../include/ksgpu/Array.hpp"

using namespace std;
using namespace ksgpu;

template<typename T1, typename T2>
static void f()
{
    cout << "\nf(): T1=" << Dtype::native<T1>() << ", T2=" << Dtype::native<T2>() << endl;

    Array<T1> arr1({2,3}, af_uhost | af_random);
    Array<T2> arr2 = arr1.template convert<T2> ();

    cout << "arr1, dtype=" << arr1.dtype << "\n";
    print_array(arr1);
    cout << "arr2, dtype=" << arr2.dtype << "\n";
    print_array(arr2);

    cout << "first call\n";
    assert_arrays_equal(arr1, arr2, "arr1", "arr2", {"ax0","ax1"});

    cout << "second call\n";
    arr2.at({1,2}) += 1;
    try {
	assert_arrays_equal(arr1, arr2, "arr1", "arr2", {"ax0","ax1"});
    } catch (...) {
	cout << "failed as intended\n";
    }
}


int main(int argc, char **argv)
{
    f<double,__half>();
    f<complex<float>,complex<double>>();
    f<complex<float>,complex<__half>>();
    f<int,int>();
    return 0;
}
