#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/cuda_utils.hpp"    // CUDA_CALL()
#include "../include/ksgpu/string_utils.hpp"  // tuple_str()

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

using namespace std;


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


bool _tuples_equal(int ndim1, const long *shape1, int ndim2, const long *shape2)
{
    if (ndim1 != ndim2)
	return false;
    
    for (int i = 0; i < ndim1; i++)
	if (shape1[i] != shape2[i])
	    return false;

    return true;
}


string _tuple_str(int ndim, const long *shape)
{
    // Defined in string_utils.hpp.
    // (Defining this one-line wrapper helps with compile time.)
    return ksgpu::tuple_str(ndim, shape);
}


// -------------------------------------------------------------------------------------------------


int array_get_ncontig(const Array<void> &arr)
{
    int ndim = arr.ndim;
    const long *shape = arr.shape;
    const long *strides = arr.strides;
    
    for (int d = 0; d < ndim; d++)
	if (shape[d] == 0)
	    return ndim;

    long s = 1;
    for (int d = ndim-1; d >= 0; d--) {
	if ((shape[d] > 1) && (strides[d] != s))
	    return ndim-1-d;
	s *= shape[d];
    }

    return ndim;
}


// -------------------------------------------------------------------------------------------------
//
// _check_array_invariants()


// Helper for check_array_invariants()
struct stride_checker {
    long axis_length;
    long axis_stride;

    bool operator<(const stride_checker &x)
    {
	return this->axis_stride < x.axis_stride;
    }
};


// Checks all array invariants except dtype.
void _check_array_invariants(const Array<void> &arr)
{
    int ndim = arr.ndim;
    xassert((ndim >= 0) && (ndim <= ArrayMaxDim));

    long expected_size = ndim ? 1 : 0;
    
    for (int d = 0; d < ndim; d++) {
	xassert(arr.shape[d] >= 0);
	xassert(arr.strides[d] >= 0);
	expected_size *= arr.shape[d];
    }

    xassert_eq(arr.size, expected_size);
    xassert_eq((arr.data != nullptr), (arr.size != 0));
    check_aflags(arr.aflags, "Array::check_invariants");

    if (arr.size <= 1)
	return;

    // Stride checks follow
    
    stride_checker sc[ndim];
    int n = 0;
    
    for (int d = 0; d < ndim; d++) {
	// Length-1 axes can have arbitrary strides
	if (arr.shape[d] == 1)
	    continue;
	
	sc[n].axis_length = arr.shape[d];
	sc[n].axis_stride = arr.strides[d];
	n++;
    }

    xassert(n > 0);  // should never fail
    std::sort(sc, sc+n);

    long min_stride = 1;
    for (int i = 0; i < n; i++) {
	xassert(sc[i].axis_stride >= min_stride);
	min_stride += (sc[i].axis_length - 1) * sc[i].axis_stride;
    }
}


// -------------------------------------------------------------------------------------------------
//
// array_reshape()


// reshape_helper()
//
//   - assumes (dst.ndim, dst.shape, dst.size) have been initialized by caller
//      and error-checked, but not checked for consistency with 'src'.
//
//   - initializes dst.strides and returns:
//       0 = success
//       1 = dst.shape is incompatible with src_shape (or dst.shape is invalid)
//       2 = src and dst shapes are compatible, but src_strides don't allow axes to be combined


static int reshape_helper(Array<void> &dst, const Array<void> &src)
{
    // If we detect shape incompatbility, we "return 1" immediately.
    // If we detect bad src_strides, we set ret=2, rather than "return 2" immediately.
    // This is so shape incompatibility takes precedence over bad strides.
    int ret = 0;
    
    if (dst.size != src.size)
	return 1;      // catches empty-array corner cases

    if (src.size == 0) {
	// Both arrays are empty
	for (int d = 0; d < ArrayMaxDim; d++)
	    dst.strides[d] = 0;  // arbitrary
	return 0;
    }

    for (int d = dst.ndim; d < ArrayMaxDim; d++)
	dst.strides[d] = 0;
    
    int is = 0;
    int id = 0;
    
    for (;;) {
	// At top of loop, src indices < is and dst indices < id
	// have been "consumed", and verified to be compatible.
	
	// Advance until non-1 is reached
	while ((is < src.ndim) && (src.shape[is] == 1))
	    is++;
	
	// Advance until non-1 is reached
	while ((id < dst.ndim) && (dst.shape[id] == 1))
	    dst.strides[id++] = 0;  // arbitrary

	if ((is == src.ndim) && (id == dst.ndim))
	    return ret;  // shapes are compatible (return value may be 0 or 2)

	if ((is == src.ndim) || (id == dst.ndim))
	    return 1;    // should never happen, thanks to "if (dst.size != src.size) ..." above

	long ss = src.shape[is];
	long sd = dst.shape[id];
	xassert((ss >= 2) && (sd >= 2));  // should never fail

	if ((ss % sd) == 0) {
	    // Split source axis across one or more destination axes.
	    // In the loop below, source axis parameters (is,ss) are fixed.
	    // At top, dst axes <= id have "consumed" sd elements, where sd | ss.
	    
	    for (;;) {
		dst.strides[id] = (ss/sd) * src.strides[is];
		id++;
		if (ss == sd)
		    break;
		if (id == dst.ndim)
		    return 1;
		sd *= dst.shape[id];
		if ((ss % sd) != 0)
		    return 1;
	    }
	    
	    is++;
	}
	else if ((sd % ss) == 0) {
	    // Combine multiple source axes into single destination axis.
	    // In the loop below, destination axis parameters (id,sd) are fixed.
	    // At top, src axes <= is have "consumed" ss elements, where ss | sd.
	    
	    long tot_stride = (src.strides[is] * ss);
	    if ((tot_stride % sd) != 0)
		ret = 2;   // not "return 2"
	    
	    dst.strides[id] = tot_stride / sd;

	    for (;;) {
		is++;
		if (ss == sd)
		    break;
		if (is == src.ndim)
		    return 1;
		ss *= src.shape[is];
		if ((sd % ss) != 0)
		    return 1;
		if ((src.strides[is] * ss) != tot_stride)
		    ret = 2;  // not "return 2"
	    }

	    id++;
	}
	else
	    return 1;
    }
}
	    

void array_reshape(Array<void> &dst, const Array<void> &src, int dst_ndim, const long *dst_shape)
{
    xassert(dst_ndim >= 0);
    xassert(dst_ndim <= ArrayMaxDim);
    xassert((dst_ndim == 0) || (dst_shape != nullptr));

    dst.ndim = dst_ndim;
    dst.data = src.data;
    dst.base = src.base;
    dst.dtype = src.dtype;
    dst.aflags = src.aflags;
    dst.size = dst_ndim ? 1 : 0;
    
    for (int d = 0; d < dst_ndim; d++) {
	xassert(dst_shape[d] >= 0);
	dst.shape[d] = dst_shape[d];
	dst.size *= dst_shape[d];
    }

    for (int d = dst_ndim; d < ArrayMaxDim; d++)
	dst.shape[d] = 0;

    // reshape_helper() initializes dst.strides
    int status = reshape_helper(dst, src);

    if (status == 1) {
	stringstream ss;
	ss << "Array::reshape_ref(): src_shape=" << src.shape_str()
	   << " is incompatible with dst_shape=" << dst.shape_str();
	throw runtime_error(ss.str());
	
    }
    else if (status == 2) {
	stringstream ss;
	ss << "Array::reshape_ref(): src_shape=" << src.shape_str()
	   << " and dst_shape=" << dst.shape_str()
	   << " are compatible, but src_strides=" << src.stride_str()
	   << " don't allow axes to be combined";
	throw runtime_error(ss.str());
    }

    dst.check_invariants();
}


// -------------------------------------------------------------------------------------------------
//
// array_fill()


struct fill_axis {
    long length;
    long dstride_nbits;
    long sstride_nbits;

    inline bool operator<(const fill_axis &a) const
    {
	// FIXME put more thought into this
	return dstride_nbits < a.dstride_nbits;
    }
};


// Recursive helper function called by array_fill().
static void array_fill2(char *dst, const char *src, int ndim, const fill_axis *axes)
{
    // Caller guarantees the following (these are asserts in array_fill()).
    // Note that axis 0 corresponds to single bits.
    //
    //   ndim > 0
    //   axes[0].dstride_nbits == 1
    //   axes[0].sstride_nbits == 1
    //   (axes[0].length % 8) == 0
    //   (axes[d].dstride_nbits % 8) == 0    for d > 0
    //   (axes[d].sstride_nbits % 8) == 0    for d > 0
    
    if (ndim <= 1)
	CUDA_CALL(cudaMemcpy(dst, src, axes[0].length >> 3, cudaMemcpyDefault));

    else if (ndim == 2) {
	// These two conditions are required by cudaMemcpy2D().
	// In particular, strides must be positive.
	xassert(axes[1].dstride_nbits >= axes[0].length);
	xassert(axes[1].sstride_nbits >= axes[0].length);
	
	CUDA_CALL(cudaMemcpy2D(dst, axes[1].dstride_nbits >> 3,
			       src, axes[1].sstride_nbits >> 3,
			       axes[0].length >> 3,
			       axes[1].length,
			       cudaMemcpyDefault));
    }

    else {
	// Note: there is a cudaMemcpy3D(), but it requires that either the
	// data be in a cudaArray, or that strides are contiguous (so that
	// it's a cudaMemcpy2D in disguise), so we can't use it here.

	long ds = axes[ndim-1].dstride_nbits >> 3;
	long ss = axes[ndim-1].sstride_nbits >> 3;
	
	for (int i = 0; i < axes[ndim-1].length; i++)
	    array_fill2(dst + i*ds, src + i*ss, ndim-1, axes);
    }
}


// Helper function called by array_fill().
static void show_array_fill(ostream &os, const Array<void> &dst, const Array<void> &src, int naxes, const fill_axis *axes)
{
    os << "ksgpu::Array::fill: shape=" << dst.shape_str()
       << ", dst.strides=" << dst.stride_str()
       << ", src.strides=" << src.stride_str()
       << ", elt_nbits=" << dst.dtype.nbits
       << "\n";
    
    for (int d = 0; d < naxes; d++)
	os << "    coalesced axis: len=" << axes[d].length
	   << ", dstride_nbits=" << axes[d].dstride_nbits
	   << ", sstride_nbits=" << axes[d].sstride_nbits
	   << "\n";
}


void array_fill(Array<void> &dst, const Array<void> &src, bool noisy)
{
    // Check that dst/src shapes match.
    if (!dst.shape_equals(src)) {
	stringstream ss;
	ss << "ksgpu::Array::fill(): dst.shape=" << dst.shape_str()
	   << " and src.shape=" << src.shape_str() << " are unequal";
	throw runtime_error(ss.str());
    }

    // Check that dst/src dtypes match.
    // Currently require dtypes to match exactly (e.g. can't fill signed int from unsigned int).
    if (dst.dtype != src.dtype) {
	stringstream ss;
	ss << "ksgpu::Array::fill(): dst.dtype=" << dst.dtype
	   << " and src.dtype=" << src.dtype << " are unequal";
	throw runtime_error(ss.str());
    }

    // If empty array, then return early.
    if (dst.size == 0)
	return;

    // Uncoalesced axes
    int ndim = dst.ndim;
    long elt_nbits = dst.dtype.nbits;
    fill_axis axes_u[ndim];
    int nax_u = 0;
    
    for (int d = 0; d < ndim; d++) {
	if (dst.shape[d] <= 1)
	    continue;

	axes_u[nax_u].length = dst.shape[d];
	axes_u[nax_u].dstride_nbits = dst.strides[d] * elt_nbits;
	axes_u[nax_u].sstride_nbits = src.strides[d] * elt_nbits;
	nax_u++;
    }

    // Sort by increasing dstride
    std::sort(axes_u, axes_u + nax_u);

    // Coalece axes, and represent itemsize by a new length-nbits axis.
    fill_axis axes_c[ndim+1];
    int nax_c = 1;

    axes_c[0].length = elt_nbits;
    axes_c[0].dstride_nbits = 1;
    axes_c[0].sstride_nbits = 1;

    // (Length * stride_nbits) of last coalesced axis.
    long dlen = elt_nbits;
    long slen = elt_nbits;
    
    for (int d = 0; d < nax_u; d++) {
	if ((axes_u[d].dstride_nbits == dlen) && (axes_u[d].sstride_nbits == slen)) {
	    // Can coalesce.
	    long s = axes_u[d].length;
	    axes_c[nax_c-1].length *= s;
	    dlen *= s;
	    slen *= s;
	}
	else {
	    // Can't coalesce.
	    axes_c[nax_c] = axes_u[d];
	    dlen = axes_u[d].length * axes_u[d].dstride_nbits;
	    slen = axes_u[d].length * axes_u[d].sstride_nbits;
	    nax_c++;
	}
    }

    if (noisy)
	show_array_fill(cout, dst, src, nax_c, axes_c);    
    
    xassert(nax_c > 0);
    xassert(axes_c[0].dstride_nbits == 1);
    xassert(axes_c[0].sstride_nbits == 1);

    bool invalid = (axes_c[0].length & 7) != 0;

    for (int d = 1; d < nax_c; d++)
	if ((axes_c[d].dstride_nbits & 7) || (axes_c[d].sstride_nbits & 7))
	    invalid = true;

    if (invalid) {
	stringstream ss;
	ss << "Array::fill() operation can't be decomposed into byte-contiguous copies.\n"
	   << "This is currently treated as an error, even in tame cases such as a contiguous 1-d array with (total_nbits % 8) != 0.\n"
	   << "Addressing this is nontrivial (e.g. consider case where 'tame' array is subarray of an ambient array).\n"
	   << "I may revisit this in the future, but it's not a high priority right now.\n";
	
	show_array_fill(ss, dst, src, nax_c, axes_c);
	throw runtime_error(ss.str());
    }

    array_fill2((char *) dst.data, (const char *) src.data, nax_c, axes_c);
}


// -------------------------------------------------------------------------------------------------
//
// array_slice()


void array_slice(Array<void> &dst, const Array<void> &src, int axis, long ix)
{
    int ndim = src.ndim;
    
    xassert((axis >= 0) && (axis < ndim));
    xassert((ix >= 0) && (ix < src.shape[axis]));

    // Slicing (1-dim -> 0-dim) doesn't make sense,
    // since our zero-dimensional Arrays are empty.
    xassert(ndim > 1);
    
    dst.ndim = ndim-1;
    dst.dtype = src.dtype;
    dst.aflags = src.aflags & af_location_flags;
    dst.base = src.base;
    dst.size = 1;  // will be updated in loop below

// Suppress spurious GCC warning in loop that follows.
// FIXME is this really necessary?
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
    
    for (int i = 0; i < ndim-1; i++) {
	int j = (i < axis) ? i : (i+1);
	dst.shape[i] = src.shape[j];
	dst.strides[i] = src.strides[j];
	dst.size *= src.shape[j];
    }
    
#pragma GCC diagnostic pop

    for (int i = ndim-1; i < ArrayMaxDim; i++) {
	dst.shape[i] = 0;
	dst.strides[i] = 0;
    }

    long nbits = ix * src.strides[axis] * long(src.dtype.nbits);
    
    if (dst.size && (nbits & 7))
	throw runtime_error("Array::slice(): slicing operation is not byte-aligned");

    const char *p = dst.size ? ((const char *)src.data + (nbits >> 3)) : nullptr;
    dst.data = (void *) p;
    dst.check_invariants();
}


void array_slice(Array<void> &dst, const Array<void> &src, int axis, long start, long stop)
{
    int ndim = src.ndim;
    
    // Currently we allow slices like arr[2:10] but not arr[2:-10] or arr[2:10:2].
    // This would be easy to generalize!
    xassert((axis >= 0) && (axis < ndim));
    xassert((start >= 0) && (start <= stop) && (stop <= src.shape[axis]));
    
    dst.ndim = ndim;
    dst.dtype = src.dtype;
    dst.aflags = src.aflags & af_location_flags;
    dst.base = src.base;
    dst.size = 1;  // will be updated in loop below

    for (int i = 0; i < ndim; i++) {
	dst.shape[i] = (i == axis) ? (stop-start) : src.shape[i];
	dst.strides[i] = src.strides[i];
	dst.size *= src.shape[i];
    }

    for (int i = ndim; i < ArrayMaxDim; i++)
	dst.shape[i] = dst.strides[i] = 0;
    
    long nbits = start * src.strides[axis] * long(src.dtype.nbits);
    
    if (dst.size && (nbits & 7))
	throw runtime_error("Array::slice(): slicing operation is not byte-aligned");

    const char *p = dst.size ? ((const char *)src.data + (nbits >> 3)) : nullptr;
    dst.data = (void *)p;
    dst.check_invariants();
}


// -------------------------------------------------------------------------------------------------
//
// array_transpose()


void array_transpose(Array<void> &dst, const Array<void> &src, const int *perm)
{
    xassert(perm != nullptr);
    int ndim = src.ndim;
    
    dst.ndim = ndim;
    dst.data = src.data;
    dst.size = src.size;
    dst.base = src.base;
    dst.dtype = src.dtype;
    dst.aflags = src.aflags;

    bool flags[ArrayMaxDim];

    for (int d = 0; d < ArrayMaxDim; d++)
        flags[d] = false;

    for (int d = 0; d < ndim; d++) {
        int p = perm[d];
        xassert((p >= 0) && (p < ndim));
        xassert(!flags[p]);   // if fails, then 'perm' is not a permutation

        dst.shape[d] = src.shape[p];
        dst.strides[d] = src.strides[p];
        flags[p] = true;
    }

    for (int d = ndim; d < ArrayMaxDim; d++)
	dst.shape[d] = dst.strides[d] = 0;
    
    dst.check_invariants();
}


}  // namespace ksgpu
