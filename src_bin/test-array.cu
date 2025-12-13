#include <iostream>
#include <algorithm>

#include "../include/ksgpu.hpp"

using namespace std;
using namespace ksgpu;


// -------------------------------------------------------------------------------------------------
//
// FIXME -- all tests are on the host! GPU arrays are not tested. This is a real shame, but I'm not
// sure how to prioritize fixing it.


struct TestArrayAxis {
    long index = 0;
    long length = 0;
    long stride = 0;

    bool operator<(const TestArrayAxis &s) const
    {
        return stride < s.stride;
    }
};


template<typename T>
struct RandomlyStridedArray {
    Array<T> arr;
    int ndim = 0;

    shared_ptr<T> cbase;  // contiguous base array
    shared_ptr<T> cbase_copy;
    long cbase_len = 0;

    std::vector<TestArrayAxis> axes;
    bool noisy = false;

    
    RandomlyStridedArray(const vector<long> &shape, const vector<long> &strides, bool noisy_=false) :
        noisy(noisy_)
    {
        if (noisy) {
            cout << "RandomlyStridedArray" // << "<" << type_name<T>() << ">"
                 << ": shape=" << tuple_str(shape)
                 << ", strides=" << tuple_str(strides) << endl;
        }
                
        ndim = shape.size();
        xassert(ndim > 0);
        xassert(ndim <= ArrayMaxDim);
        xassert(shape.size() == strides.size());

        cbase_len = 1;
        for (int d = 0; d < ndim; d++) 
            cbase_len += (shape[d]-1) * strides[d];

        cbase = af_alloc<T> (cbase_len, af_rhost | af_guard | af_random);
        cbase_copy = af_alloc<T> (cbase_len, af_rhost);
        memcpy(cbase_copy.get(), cbase.get(), cbase_len * sizeof(T));

        arr.data = cbase.get();
        arr.dtype = Dtype::native<T>();
        arr.ndim = ndim;
        arr.base = cbase;
        arr.aflags = af_rhost;
        arr.size = 1;
        
        for (int d = 0; d < ArrayMaxDim; d++) {
            arr.shape[d] = (d < ndim) ? shape[d] : 0;
            arr.strides[d] = (d < ndim) ? strides[d] : 0;
            arr.size *= ((d < ndim) ? shape[d] : 1);
        }

        arr.check_invariants("RandomlyStridedArray::arr");
        // check_for_buffer_overflows();

        axes.resize(ndim);
        for (int d = 0; d < ndim; d++) {
            axes[d].index = d;
            axes[d].length = shape[d];
            axes[d].stride = strides[d];
        }

        std::sort(axes.begin(), axes.end());
    }

    RandomlyStridedArray(const vector<long> &shape_, bool noisy_=false)
        : RandomlyStridedArray(shape_, make_random_strides(shape_), noisy_)
    { }
    
    RandomlyStridedArray(bool noisy_=false)
        : RandomlyStridedArray(make_random_shape(), noisy_)
    { }


    // Given a cbase index 0 <= pos < cbase_len, return true (and
    // initialize the length-ndim array ix_out) if in 'arr'.
    
    bool find(long pos, long *ix_out) const
    {
        xassert(pos >= 0 && pos < cbase_len);

        // Process axes from largest stride to smallest
        for (int d = ndim-1; d >= 0; d--) {
            const TestArrayAxis &axis = axes[d];
            int i = axis.index;

            if (axis.length == 1) {
                ix_out[i] = 0;
                continue;
            }

            long j = pos / axis.stride;
            if (j >= axis.length)
                return false;
            
            pos -= j * axis.stride;             
            ix_out[i] = j;
        }

        return (pos == 0);
    }


    void check_for_buffer_overflows() const
    {
        T *p1 = cbase.get();
        T *p2 = cbase_copy.get();
        long ix_out[ndim];
        
        for (long i = 0; i < cbase_len; i++)
            if (!find(i, ix_out))
                xassert(p1[i] == p2[i]);
    }

    // test_basics(): tests consistency of find(), Array::at(), and
    // Array index enumeration (ix_start, ix_valid, ix_next).
    
    void test_basics() const
    {
        vector<bool> flag(cbase_len, false);
        long ix_out[ndim];
        long s = 0;
        
        if (noisy)
            cout << "    test_basics()" << endl;

        for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix)) {
            long j = &arr.at(ix) - arr.data;
            xassert(j >= 0 && j < cbase_len);
            xassert(find(j, ix_out));
            xassert(!flag[j]);
            flag[j] = true;
            s++;

            for (int d = 0; d < ndim; d++)
                xassert(ix_out[d] == ix[d]);
        }

        xassert(s == arr.size);

        for (long j = 0; j < cbase_len; j++)
            if (!flag[j])
                xassert(!find(j, ix_out));
    }

    
    void test_thin_slice(int axis, int pos) const
    {
        long ix_out[ndim];

        // Thin-slicing 1-D arrays isn't implemented (see comment in Array.hpp)
        xassert(ndim > 1);
        
        if (noisy)
            cout << "    test_thin_slice(axis=" << axis << ",pos=" << pos << ")" << endl;

        Array<T> s = arr.slice(axis, pos);

        xassert(s.ndim == ndim-1);
        for (int d = 0; d < axis; d++)
            xassert(s.shape[d] == arr.shape[d]);
        for (int d = axis+1; d < ndim; d++)
            xassert(s.shape[d-1] == arr.shape[d]);

        for (auto ix = s.ix_start(); s.ix_valid(ix); s.ix_next(ix)) {
            long j = &s.at(ix) - arr.data;
            xassert(j >= 0 && j < cbase_len);
            xassert(find(j, ix_out));

            xassert(ix_out[axis] == pos);
            for (int d = 0; d < axis; d++)
                xassert(ix_out[d] == ix[d]);
            for (int d = axis+1; d < ndim; d++)
                xassert(ix_out[d] == ix[d-1]);
        }
    }

    void test_thin_slice() const
    {
        int axis = rand_int(0, ndim);
        int pos = rand_int(0, arr.shape[axis]);
        test_thin_slice(axis, pos);
    }


    void test_thick_slice(int axis, int start, int stop) const
    {
        long ix_out[ndim];
        
        if (noisy) {
            cout << "    test_thick_slice(axis=" << axis
                 << ", start=" << start
                 << ", stop=" << stop << ")" << endl;
        }

        Array<T> s = arr.slice(axis, start, stop);

        xassert(s.ndim == ndim);
        for (int d = 0; d < ndim; d++) {
            long t = (d == axis) ? (stop-start) : arr.shape[d];
            xassert(s.shape[d] == t);
        }

        for (auto ix = s.ix_start(); s.ix_valid(ix); s.ix_next(ix)) {
            long j = &s.at(ix) - arr.data;
            xassert(j >= 0 && j < cbase_len);
            xassert(find(j, ix_out));

            for (int d = 0; d < ndim; d++) {
                long t = (d == axis) ? start : 0;
                xassert(ix_out[d] == ix[d] + t);
            }
        }
    }

    void test_thick_slice() const
    {
        int axis = rand_int(0, ndim);
        int slen = rand_int(0, arr.shape[axis] + 1);
        int start = rand_int(0, arr.shape[axis] - slen + 1);
        int stop = start + slen;
        test_thick_slice(axis, start, stop);
    }

    
    void run_simple_tests() const
    {
        test_basics();
        test_thick_slice();

        if (ndim > 1)
            test_thin_slice();
    }
};


// -------------------------------------------------------------------------------------------------


template<typename T>
void test_set_zero(bool noisy)
{
    vector<long> shape = make_random_shape();
    vector<long> strides = make_random_strides(shape);

    RandomlyStridedArray<T> rs(shape, strides);
    rs.arr.set_zero(noisy);

    for (auto ix = rs.arr.ix_start(); rs.arr.ix_valid(ix); rs.arr.ix_next(ix))
        xassert(rs.arr.at(ix) == T(0));

    // FIXME: I don't have a unit test yet for Array::randomize(), but in the meantime
    // this thrown-in call to Array::randomize() does check that it's not overflowing
    // its buffers.
    rs.arr.randomize(noisy);

    rs.check_for_buffer_overflows();
}


// -------------------------------------------------------------------------------------------------


template<typename T>
struct FillTestInstance {
    vector<long> shape;
    vector<long> strides1;
    vector<long> strides2;
    
    RandomlyStridedArray<T> rs1;
    RandomlyStridedArray<T> rs2;

    bool noisy = false;
    
    
    FillTestInstance(const vector<long> &shape_,
                     const vector<long> &strides1_,
                     const vector<long> &strides2_) :
        shape(shape_),
        strides1(strides1_),
        strides2(strides2_),
        rs1(shape_, strides1_),
        rs2(shape_, strides2_)
    { }
    
    FillTestInstance(const vector<long> &shape_) :
        FillTestInstance(shape_, make_random_strides(shape_), make_random_strides(shape_))
    { }

    FillTestInstance() :
        FillTestInstance(make_random_shape())
    { }

    
    void run()
    {
        if (noisy) {
            cout << "test_fill: shape=" << tuple_str(shape)
                 << ", strides1=" << tuple_str(strides1)
                 << ", strides2=" << tuple_str(strides2)
                 << ")" << endl;
        }

        Array<T> &arr1 = rs1.arr;
        Array<T> &arr2 = rs2.arr;

        array_fill(arr1, arr2, noisy);
        
        for (auto ix = arr1.ix_start(); arr1.ix_valid(ix); arr1.ix_next(ix))
            xassert(arr1.at(ix) == arr2.at(ix));
        
        rs1.check_for_buffer_overflows();
        rs2.check_for_buffer_overflows();
    }
};
    

// -------------------------------------------------------------------------------------------------
//
// test_reshape()
//
// FIXME it would be nice to test that reshape() correctly throws an exception,
// in cases where it should fail.


template<typename T>
static void test_reshape(const vector<long> &dst_shape,
                         const vector<long> &src_shape,
                         const vector<long> &src_strides,
                         bool noisy=false)
{
    if (noisy) {
        cout << "test_reshape: dst_shape=" << tuple_str(dst_shape)
             << ", src_shape=" << tuple_str(src_shape)
             << ", src_strides=" << tuple_str(src_strides)
             << endl;
    }

    // Note: src array is uninitialized, but that's okay since we compare array addresses (not contents) below.
    Array<T> src(src_shape, src_strides, af_uhost);
    Array<T> dst = src.reshape(dst_shape);

    auto src_ix = src.ix_start();
    auto dst_ix = dst.ix_start();

    for (;;) {
        bool src_valid = src.ix_valid(src_ix);
        bool dst_valid = dst.ix_valid(dst_ix);
        xassert(src_valid == dst_valid);

        if (!src_valid)
            return;

        // Compare array addresses (not contents).
        const T *srcp = &src.at(src_ix);
        const T *dstp = &dst.at(dst_ix);
        xassert(srcp == dstp);
        
        src.ix_next(src_ix);
        dst.ix_next(dst_ix);
    }
}


template<typename T>
static void test_reshape(bool noisy=false)
{
    vector<long> dshape, sshape, sstrides;
    make_random_reshape_compatible_shapes(dshape, sshape, sstrides);
    test_reshape<T> (dshape, sshape, sstrides, noisy);
}


// -------------------------------------------------------------------------------------------------
//
// test_convert()
//
// I wrote this unit test after implementing Array<void> and ksgpu::Dtypes.
//
// Previous unit tests were written before these features were implemented, and use
// template<typename T> instead. The previous tests do have some nice features that
// are missing here, such as RandomlyStridedArray::check_for_buffer_overflows().
//
// FIXME: develop a best-of-both-worlds framework. (More generally, should systematically
// revisit the tests in this file, which are sort of ad-hoc and not 100% complete.)


// DstSrcPair: pair of arrays 'dst', 'src' with the same shape and aflags.
// (Dtypes and strides can be different.)

struct DstSrcPair
{
    Array<void> dst;
    Array<void> src;
    bool noisy = false;

    // Note that we always add (af_random) to src.aflags!
    DstSrcPair(Dtype dst_dtype, const vector<long> &dst_strides,
               Dtype src_dtype, const vector<long> &src_strides,
               const vector<long> &shape, int aflags, bool noisy_)
        : dst(dst_dtype, shape, dst_strides, aflags),
          src(src_dtype, shape, src_strides, aflags | af_random),
          noisy(noisy_)
    { }

    void assert_equal(const string &where)
    {
        vector<string> axis_names(dst.ndim);
        
        for (int d = 0; d < dst.ndim; d++) {
            stringstream ss;
            ss << "ax" << d;
            axis_names[d] = ss.str();
        }

        // FIXME the biggest weakness in this unit test is that the test assumes
        // that ksgpu::assert_arrays_equal() works. One way to address this would
        // be to add a standalone unit test for assert_arrays_equal().
        
        ksgpu::assert_arrays_equal(dst, src, where + " dst", "src", axis_names);
    }
    
    static DstSrcPair make_random(bool noisy)
    {
        vector<long> shape = make_random_shape();
        vector<long> dst_strides = make_random_strides(shape);
        vector<long> src_strides = make_random_strides(shape);
        int aflags = af_uhost;  // for now
    
        Dtype dst_dtype;
        Dtype src_dtype;
        
        if (rand_uniform() < 0.2) {
            ushort flags = (rand_uniform() < 0.5) ? df_int : df_uint;   
            ushort nbits = 1 << rand_int(3,7);
            dst_dtype = src_dtype = Dtype(flags, nbits);
        }
        else {
            ushort nbits1 = 1 << rand_int(4,7);
            ushort nbits2 = 1 << rand_int(4,7);
            dst_dtype = Dtype(df_float, nbits1);
            src_dtype = Dtype(df_float, nbits2);
        }
        
        if (rand_uniform() < 0.5) {
            dst_dtype = dst_dtype.complex();
            src_dtype = src_dtype.complex();
        }

        return DstSrcPair(dst_dtype, dst_strides, src_dtype, src_strides, shape, aflags, noisy);
    }
};


void test_convert(DstSrcPair &ds)
{
    Array<void> &dst = ds.dst;
    Array<void> &src = ds.src;
    
    if (ds.noisy) {
        cout << "test_convert: shape=" << dst.shape_str()
             << ", dst_dtype=" << dst.dtype
             << ", src_dtype=" << src.dtype
             << ", dst_strides=" << dst.stride_str()
             << ", src_strides=" << src.stride_str()
             << ", aflags=" << aflag_str(dst.aflags)
             << endl;
    }

    array_convert(dst, src, ds.noisy);
    ds.assert_equal("test convert");
}


void test_convert(bool noisy)
{
    DstSrcPair ds = DstSrcPair::make_random(noisy);
    test_convert(ds);
}


// -------------------------------------------------------------------------------------------------


template<typename T>
static void run_all_tests(bool noisy)
{
    // Note: test_convert() is not included in run_all_tests().
    
    RandomlyStridedArray<T> rs(noisy);  
    rs.run_simple_tests();

    test_set_zero<T> (noisy);

    FillTestInstance<T> ft;
    ft.noisy = noisy;
    ft.run();

    test_reshape<T> (noisy);
}


int main(int argc, char **argv)
{
    bool noisy = false;
    int niter = 1000;
        
    for (int i = 0; i < niter; i++) {
        if (i % 100 == 0)
            cout << "test-array: iteration " << i << "/" << niter << endl;

        run_all_tests<float> (noisy);
        run_all_tests<char> (noisy);
        test_convert(noisy);
    }

    cout << "test-array passed!" << endl;
    return 0;
}
