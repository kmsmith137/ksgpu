// A place for one-off hacks that won't be committed to git.
// (Note that the Makefile is set up to compile it.)

#include <complex>
#include <iostream>
// #include "../include/ksgpu.hpp"
#include "../include/ksgpu/Array.hpp"

using namespace std;
using namespace ksgpu;

int main(int argc, char **argv)
{
    float x = 1.0;
    double y = 2.0;
    bool b = (x < y);
    return 0;
}
