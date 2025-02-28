#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/cuda_utils.hpp"    // CUDA_CALL()
#include "../include/ksgpu/string_utils.hpp"  // tuple_str()

#include <sstream>
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


// Just for readability.
inline bool iff(bool x, bool y) { return x==y; }


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
void _check_array_invariants(const Array<void> &arr, const char *where)
{
    int ndim = arr.ndim;
    xassert_where((ndim >= 0) && (ndim <= ArrayMaxDim), where);

    long expected_size = ndim ? 1 : 0;
    
    for (int d = 0; d < ndim; d++) {
	xassert_where(arr.shape[d] >= 0, where);
	xassert_where(arr.strides[d] >= 0, where);
	expected_size *= arr.shape[d];
    }

    xassert_where((arr.size == expected_size), where);
    xassert_where(iff(arr.data != nullptr, arr.size != 0), where);
    check_aflags(arr.aflags, where);

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

    xassert_where(n > 0, where);  // should never fail
    std::sort(sc, sc+n);

    long min_stride = 1;
    for (int i = 0; i < n; i++) {
	xassert_where(sc[i].axis_stride >= min_stride, where);
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

    dst.check_invariants("array_reshape()");
}


// -------------------------------------------------------------------------------------------------
//
// fill_axis: a helper class for array_fill() and array_convert().


struct fill_axis {
    // Meaning of 'dstride' and 'sstride' is context-dependent:
    //  - in array_fill(), strides are bit counts.
    //  - in array_convert(), strides are element counts.
    
    long length = 0;
    long dstride = 0;
    long sstride = 0;

    inline bool operator<(const fill_axis &a) const
    {
	// FIXME put more thought into this
	return dstride < a.dstride;
    }

    fill_axis() { }
    fill_axis(const fill_axis &) = default;
};


static void init_uncoalesced_axes(int &naxes, fill_axis *axes, const Array<void> &dst, const Array<void> &src, const char *where)
{	
    // Check that dst/src shapes match.
    // Note: we don't check that dst/src dtypes match.
    
    if (!dst.shape_equals(src)) {
	stringstream ss;
	ss << where << ": dst.shape=" << dst.shape_str()
	   << " and src.shape=" << src.shape_str() << " are unequal";
	throw runtime_error(ss.str());
    }

    if (dst.size == 0) {
	naxes = 0;
	return;
    }

    if (dst.size == 1) {
	naxes = 1;
	axes[0].length = 1;
	axes[0].dstride = 1;
	axes[0].sstride = 1;
	return;
    }

    // Uncoalesced axes.
    int ndim = dst.ndim;
    naxes = 0;
    
    for (int d = 0; d < ndim; d++) {
	xassert(dst.shape[d] > 0);
	
	// Skip length-1 axes.
	if (dst.shape[d] == 1)
	    continue;
	
	axes[naxes].length = dst.shape[d];
	axes[naxes].dstride = dst.strides[d];
	axes[naxes].sstride = src.strides[d];
	naxes++;
    }

    xassert(naxes > 0);
}


// 'noinline' suppresses gcc compiler warning when inlining std::sort() on few-element array.
static __attribute__((noinline)) void
init_coalesced_axes(int &nax_c, fill_axis *axes_c, int nax_u, fill_axis *axes_u)
{
    // Sort 'axes_u' by increasing dstride.
    xassert(nax_u > 0);
    std::sort(axes_u, axes_u + nax_u);
    
    axes_c[0] = axes_u[0];
    nax_c = 1;
    
    for (int i = 1; i < nax_u; i++) {
	long ds = axes_c[nax_c-1].length * axes_c[nax_c-1].dstride;
	long ss = axes_c[nax_c-1].length * axes_c[nax_c-1].sstride;
	bool can_coalesce = (axes_u[i].dstride == ds) && (axes_u[i].sstride == ss);
	
	if (can_coalesce)
	    axes_c[nax_c-1].length *= axes_u[i].length;
	else
	    axes_c[nax_c++] = axes_u[i];
    }
}


static void show_axes(ostream &os, const Array<void> &dst, const Array<void> &src, int nax_c, const fill_axis *axes_c)
{
    os << "    shape=" << dst.shape_str()
       << ", dst.strides=" << dst.stride_str()
       << ", src.strides=" << src.stride_str()
       << ", dst.dtype=" << dst.dtype
       << ", src.dtype=" << src.dtype
       << "\n";
    
    for (int d = 0; d < nax_c; d++)
	os << "    coalesced axis: len=" << axes_c[d].length
	   << ", dstride=" << axes_c[d].dstride
	   << ", sstride=" << axes_c[d].sstride
	   << "\n";
};


// -------------------------------------------------------------------------------------------------
//
// array_fill()


// Recursive helper function called by array_fill().
static void array_fill2(char *dst, const char *src, int naxes, const fill_axis *axes)
{
    // Caller guarantees the following (these are asserts in array_fill()).
    // Note that fill_axis::dstride and fill_axis::sstride correspond to bits.
    //
    //   naxes > 0
    //   axes[0].dstride == 1
    //   axes[0].sstride == 1
    //   (axes[0].length % 8) == 0
    //   (axes[d].dstride % 8) == 0    for d > 0
    //   (axes[d].sstride % 8) == 0    for d > 0
    
    if (naxes == 1)
	CUDA_CALL(cudaMemcpy(dst, src, axes[0].length >> 3, cudaMemcpyDefault));

    else if (naxes == 2) {
	// These two conditions are required by cudaMemcpy2D().
	// In particular, strides must be positive.
	xassert(axes[1].dstride >= axes[0].length);
	xassert(axes[1].sstride >= axes[0].length);
	
	CUDA_CALL(cudaMemcpy2D(dst, axes[1].dstride >> 3,
			       src, axes[1].sstride >> 3,
			       axes[0].length >> 3,
			       axes[1].length,
			       cudaMemcpyDefault));
    }

    else {
	// Note: there is a cudaMemcpy3D(), but it requires that either the
	// data be in a cudaArray, or that strides are contiguous (so that
	// it's a cudaMemcpy2D in disguise), so we can't use it here.

	xassert(naxes >= 3);
	long ds = axes[naxes-1].dstride >> 3;
	long ss = axes[naxes-1].sstride >> 3;
	
	for (long i = 0; i < axes[naxes-1].length; i++)
	    array_fill2(dst + i*ds, src + i*ss, naxes-1, axes);
    }
}


void array_fill(Array<void> &dst, const Array<void> &src, bool noisy)
{
    // Uncoalesced axes.
    int nax_u = 0;
    fill_axis axes_u[ArrayMaxDim+1];
    init_uncoalesced_axes(nax_u, axes_u, dst, src, "Array::fill()");

    // Check that dst/src dtypes match.
    // Currently require dtypes to match exactly (e.g. can't fill signed int from unsigned int).
    if (dst.dtype != src.dtype) {
	stringstream ss;
	ss << "ksgpu::Array::fill(): dst.dtype=" << dst.dtype
	   << " and src.dtype=" << src.dtype << " are unequal";
	throw runtime_error(ss.str());
    }

    // Pure paranoia.
    xassert(dst.size == src.size);
    xassert(dst.dtype.nbits == src.dtype.nbits);
    int nbits = dst.dtype.nbits;
    
    // Return early if there is no data to copy.
    if (nax_u == 0)
	return;

    // Convert strides to bits.
    for (int i = 0; i < nax_u; i++) {
	axes_u[i].dstride *= nbits;
	axes_u[i].sstride *= nbits;
    }

    // Add dummy "single-bit" axis -- this allows us to subsequently ignore dtypes, and work with bits.
    axes_u[nax_u].length = nbits;
    axes_u[nax_u].dstride = 1;
    axes_u[nax_u].sstride = 1;
    nax_u++;

    // Coalesced axes.
    int nax_c = 0;
    fill_axis axes_c[ArrayMaxDim+1];
    init_coalesced_axes(nax_c, axes_c, nax_u, axes_u);   // note: permutes 'axes_u' in place.

    if (noisy) {
	cout << "Array::fill(): note that 'dst/src' strides are elements, and 'coalesced axis' strides are bits\n";
	show_axes(cout, dst, src, nax_c, axes_c);
    }
    
    // Now a bunch of asserts/checks, in preparation for calling array_fill2().
    
    xassert(nax_c > 0);
    xassert(axes_c[0].dstride == 1);
    xassert(axes_c[0].sstride == 1);

    bool invalid = (axes_c[0].length & 7) != 0;

    for (int d = 1; d < nax_c; d++)
	if ((axes_c[d].dstride & 7) || (axes_c[d].sstride & 7))
	    invalid = true;

    if (invalid) {
	stringstream ss;
	
	ss << "Array::fill() operation can't be decomposed into byte-contiguous copies.\n"
	   << "This is currently treated as an error, even in tame cases such as a contiguous 1-d array with (total_nbits % 8) != 0.\n"
	   << "Addressing this is nontrivial (e.g. consider case where 'tame' array is subarray of an ambient array).\n"
	   << "I may revisit this in the future, but it's not a high priority right now.\n"
	   << "In the diagnostic output below, note that 'dst/src' strides are elements, and 'coalesced axis' strides are bits\n";
	
	show_axes(ss, dst, src, nax_c, axes_c);
	throw runtime_error(ss.str());
    }

    // Checks pass, now we can call array_fill2().
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
    dst.check_invariants("\"thin\" array_slice()");
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
	dst.size *= dst.shape[i];
    }

    for (int i = ndim; i < ArrayMaxDim; i++)
	dst.shape[i] = dst.strides[i] = 0;
    
    long nbits = start * src.strides[axis] * long(src.dtype.nbits);
    
    if (dst.size && (nbits & 7))
	throw runtime_error("Array::slice(): slicing operation is not byte-aligned");

    const char *p = dst.size ? ((const char *)src.data + (nbits >> 3)) : nullptr;
    dst.data = (void *)p;
    dst.check_invariants("\"thick\" array_slice()");
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
    
    dst.check_invariants("array_transpose()");
}


// -------------------------------------------------------------------------------------------------
//
// array_convert()
//
// Currently, we only implement conversions between { __half, float, double }.
//
// Adding more (dst_dtype, src_dtype) pairs should be strightforward:
//  - make sure convert_0d<Tdst,Tsrc> is defined
//  - add if-statements in get_1d_converter() and partially_templated_1d_converter().
//  - add unit test coverage


// --- scalar (0-dimensional) converters start here ---

template<typename Tdst, typename Tsrc>
struct convert_0d { static inline Tdst conv(Tsrc x) { return x; } };

// Conversions involving __half use CUDA intrinsics.
// Note that CUDA defines __half2float() but not __half2double().
template<> struct convert_0d<__half,float>  { static inline __half conv(float x)  { return __float2half(x); } };
template<> struct convert_0d<__half,double> { static inline __half conv(double x) { return __double2half(x); } };
template<> struct convert_0d<float,__half>  { static inline float conv(__half x)  { return __half2float(x); } };
template<> struct convert_0d<double,__half> { static inline double conv(__half x) { return __half2float(x); } };

// Conversions involving complex<...>
template<typename Tdst, typename Tsrc>
struct convert_0d<complex<Tdst>, complex<Tsrc>>
{
    static inline complex<Tdst> conv(complex<Tsrc> x)
    {
	using C = convert_0d<Tdst,Tsrc>;
	return { C::conv(x.real()), C::conv(x.imag()) };
    }
};


// --- 1-d converters start here ---

using converter_1d = void (*)(void *, const void *, const fill_axis &);

template<typename Tdst, typename Tsrc>
static void fully_templated_1d_converter(void *dst_, const void *src_, const fill_axis &axis)
{
    Tdst *dst = (Tdst *) dst_;
    const Tsrc *src = (const Tsrc *) src_;
    
    for (long i = 0; i < axis.length; i++)
	dst[i * axis.dstride] = convert_0d<Tdst,Tsrc>::conv(src[i * axis.sstride]);
}

template<typename Tdst, typename Tsrc>
static converter_1d doubly_templated_1d_converter(Dtype dst_dtype, Dtype src_dtype)
{
    if ((dst_dtype == Dtype::native<Tdst>()) && (src_dtype == Dtype::native<Tsrc>()))
	return fully_templated_1d_converter<Tdst, Tsrc>;
    
    if ((dst_dtype == Dtype::native<complex<Tdst>>()) && (src_dtype == Dtype::native<complex<Tsrc>>()))
	return fully_templated_1d_converter<complex<Tdst>, complex<Tsrc>>;

    return nullptr;
}

template<typename Tdst>
static converter_1d singly_templated_1d_converter(Dtype dst_dtype, Dtype src_dtype)
{
    Dtype dt = src_dtype.real();
    
    if (dt == Dtype::native<__half>())
	return doubly_templated_1d_converter<Tdst, __half> (dst_dtype, src_dtype);
    
    if (dt == Dtype::native<float>())
	return doubly_templated_1d_converter<Tdst, float> (dst_dtype, src_dtype);
    
    if (dt == Dtype::native<double>())
	return doubly_templated_1d_converter<Tdst, double> (dst_dtype, src_dtype);

    return nullptr;
}

static converter_1d get_1d_converter(Dtype dst_dtype, Dtype src_dtype)
{
    Dtype dt = dst_dtype.real();
    
    if (dt == Dtype::native<__half>())
	return singly_templated_1d_converter<__half> (dst_dtype, src_dtype);
    
    if (dt == Dtype::native<float>())
	return singly_templated_1d_converter<float> (dst_dtype, src_dtype);
    
    if (dt == Dtype::native<double>())
	return singly_templated_1d_converter<double> (dst_dtype, src_dtype);

    return nullptr;
}

// --- generic (N-dimensional) conversion starts here

static void convert_Nd(char *dst, const char *src, int naxes, const fill_axis *axes, converter_1d conv_1d, int dst_nbytes, int src_nbytes)
{
    if (naxes == 1) {
	conv_1d(dst, src, *axes);
	return;
    }

    xassert(naxes >= 2);
    long ds = axes[naxes-1].dstride * dst_nbytes;
    long ss = axes[naxes-1].sstride * src_nbytes;
    
    for (long i = 0; i < axes[naxes-1].length; i++)
	convert_Nd(dst + i*ds, src + i*ss, naxes-1, axes, conv_1d, dst_nbytes, src_nbytes);
}


void array_convert(Array<void> &dst, const Array<void> &src, bool noisy)
{
    if (!src.on_host())
	throw runtime_error("Array::convert(): source array must be on host");
    if (!dst.on_host())
	throw runtime_error("Array::convert(): destination array must be on host");
    
    if (dst.dtype == src.dtype) {
	array_fill(dst, src, noisy);
	return;
    }

    // Uncoalesced axes.
    int nax_u = 0;
    fill_axis axes_u[ArrayMaxDim];
    init_uncoalesced_axes(nax_u, axes_u, dst, src, "Array::convert()");
    
    // Return early if there is no data to convert.
    if (nax_u == 0)
	return;

    // Coalesced axes.
    int nax_c = 0;
    fill_axis axes_c[ArrayMaxDim];
    init_coalesced_axes(nax_c, axes_c, nax_u, axes_u);   // note: permutes 'axes_u' in place.

    converter_1d conv_1d = get_1d_converter(dst.dtype, src.dtype);

    if (!conv_1d) {
	stringstream ss;
	ss << "Array::convert(): (dst_dtype, src_dtype) = (" << dst.dtype << ", " << src.dtype
	   << ") is currently unimplemented (implementing new dtypes should be straightforward,"
	   << " see comments in ksgpu/src_lib/Array.cu)";
	throw runtime_error(ss.str());
    }

    if (noisy) {
	cout << "Array::convert(): note that all strides are elements (not bits or bytes)\n";
	show_axes(cout, dst, src, nax_c, axes_c);
    }

    xassert((dst.dtype.nbits & 7) == 0);
    xassert((src.dtype.nbits & 7) == 0);
    
    int dst_nbytes = dst.dtype.nbits >> 3;
    int src_nbytes = src.dtype.nbits >> 3;
    
    convert_Nd((char *) dst.data, (const char *) src.data, nax_c, axes_c, conv_1d, dst_nbytes, src_nbytes);
}


// -------------------------------------------------------------------------------------------------
//
// print_array()


template<typename T>
static void _print_array(const Array<void> &arr_, const vector<string> &axis_names, std::ostream &os)
{
    Array<T> arr = arr_.cast<T> ("print_array");
    int nd = arr_.ndim;
    
    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix)) {
	if (axis_names.size() == 0) {
	    os << "    (";
	    for (int d = 0; d < nd; d++)
		os << (d ? "," : "") << ix[d];
	    os << ((nd <= 1) ? ",)" : ")");
	}
	else {
	    os << "   ";
	    for (int d = 0; d < nd; d++)
		os << " " << axis_names[d] << "=" << ix[d];
	}

	os << ": " << arr.at(ix) << "\n";
    }
}


template<typename T>
static void _print_array2(const Array<void> &arr, const vector<string> &axis_names, std::ostream &os)
{
    if (arr.dtype == Dtype::native<T>())
	_print_array<T> (arr, axis_names, os);
    else if (arr.dtype == Dtype::native<complex<T>>())
	_print_array<complex<T>> (arr, axis_names, os);
    else
	throw runtime_error("internal error in ksgpu::print_array()");
}


void print_array(const Array<void> &arr_, const vector<string> &axis_names, std::ostream &os)
{
    xassert((axis_names.size() == 0) || (axis_names.size() == uint(arr_.ndim)));
    
    Array<void> arr = arr_.to_host(false);  // page_locked=true
    Dtype dt = arr.dtype.real();

    if (dt == Dtype::native<float>())
	_print_array2<float> (arr, axis_names, os);
    
    else if (dt == Dtype::native<double>())
	_print_array2<double> (arr, axis_names, os);
    
    else if (dt == Dtype::native<__half>())
	_print_array2<__half> (arr, axis_names, os);
    
    else if (dt == Dtype::native<int>())
	_print_array2<int> (arr, axis_names, os);
    
    else if (dt == Dtype::native<uint>())
	_print_array2<uint> (arr, axis_names, os);
    
    else if (dt == Dtype::native<long>())
	_print_array2<long> (arr, axis_names, os);
    
    else if (dt == Dtype::native<ulong>())
	_print_array2<ulong> (arr, axis_names, os);
    
    else if (dt == Dtype::native<short>())
	_print_array2<short> (arr, axis_names, os);
    
    else if (dt == Dtype::native<ushort>())
	_print_array2<ushort> (arr, axis_names, os);
    
    else if (dt == Dtype::native<char>())
	_print_array2<char> (arr, axis_names, os);
    
    else if (dt == Dtype::native<unsigned char>())
	_print_array2<unsigned char> (arr, axis_names, os);

    else
	throw runtime_error("ksgpu::print_array() is not implemented for dtype " + arr.dtype.str());
    
    os.flush();
}


}  // namespace ksgpu
