#ifndef _KSGPU_ARRAY_HPP
#define _KSGPU_ARRAY_HPP

#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_fp16.h>    // __half

#include "Dtype.hpp"
#include "xassert.hpp"
#include "mem_utils.hpp"  // af_alloc() and flags


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


static constexpr int ArrayMaxDim = 8;


// Used to disable some Array<T> methods for (T == void), see below.
template<typename T>
using enable_if_non_void = std::enable_if_t<!std::is_void_v<T>>;


// Array<T>: generic N-dimensional array with strides.
//
// If (T == void), then array type is determined at runtime by 'dtype' member.
// If (T != void), then 'dtype' member must equal Dtype::native<T>(), and is redundant.
//
// Scope: Array<T> implements member functions for things like:
//
//   - moving/copying data (e.g. between CPU and GPU)
//   - axis manipulation (e.g. reshape/transpose/slice)
//   - type casting/conversion
//   - debugging/printing/comparing arrays
//
// but doesn't implement any "math" kernels (e.g. tensor dot product). 


template<typename T>
struct Array {
    T *data = nullptr;
    
    int ndim = 0;
    long shape[ArrayMaxDim];
    long size = 0;

    long strides[ArrayMaxDim];  // in multiples of dtype size (not bytes or bits)

    Dtype dtype;
    std::shared_ptr<void> base;
    int aflags = 0;

    // "Empty" arrays are size-zero objects containing null pointers.
    // All Arrays obey the rules:
    //
    //  (data == nullptr)
    //      iff (size == 0)
    //      iff ((ndim==0) || (shape[i]==0 for some 0 <= i < ndim))
    //
    // Note that zero-dimensional arrays are empty (unlike numpy,
    // where zero-dimensional arrays have size 1).

    // Default constructor: caller is responsible for initializing all members,
    // then calling Array::check_invariants(). Useful in non-standard situations
    // that can't be covered by any of the "stock" constructors.
    Array();
    
    // Allocator flags ('aflags') are defined in mem_utils.hpp
    // Flags can be used to allocate memory on CPU/GPU, zero memory after allocation, etc.
    Array(int ndim, const long *shape, int aflags);
    Array(const std::vector<long> &shape, int aflags);
    
    // Syntactic sugar for constructing array with "inline" dimensions, e.g.
    //    Array<float> arr({m,n}, af_gpu);
    Array(std::initializer_list<long> shape, int aflags);

    // These constructors allow explicit strides.
    // Often used in unit tests, with make_random_strides() in test_utils.hpp.
    Array(int ndim, const long *shape, const long *strides, int aflags);
    Array(const std::vector<long> &shape, const std::vector<long> &strides, int aflags);
    Array(std::initializer_list<long> shape, std::initializer_list<long> strides, int aflags);

    // These constructors allow an explicit dtype.
    // If (T != void), then these constructors check that the dtype is consistent with T.
    // If (T == void), then constructors _without_ an explicit dtype throw excptions.
    Array(const Dtype &dtype, int ndim, const long *shape, int aflags);
    Array(const Dtype &dtype, const std::vector<long> &shape, int aflags);
    Array(const Dtype &dtype, std::initializer_list<long> shape, int aflags);
    
    // Is array addressable on GPU? On host?
    inline bool on_gpu() const { return !data || af_on_gpu(aflags); }
    inline bool on_host() const { return !data || af_on_host(aflags); }

    // Copies data from 'src' to 'this'.
    // Arrays must have the same shape, but need not have the same strides.
    // FIXME currently require dtypes to match exactly (e.g. can't fill signed int from unsigned int).
    // FIXME currently using cudaMemcpy() even if both arrays are on host -- is this slow?
    template<typename T2> inline void fill(const Array<T2> &src, bool noisy=false);
    
    inline Array<T> clone(int aflags) const;
    inline Array<T> clone() const;  // retains location_flags of source array
    
    // Returns an Array addressable on GPU (or host), making a copy if necessary.
    inline Array<T> to_gpu() const;
    inline Array<T> to_host(bool registered=true) const;

    // Returns number of contiguous dimensions, assuming indices are ordered
    // from slowest to fastest varying. Returns 'ndim' for an empty array.
    inline int get_ncontig() const;
    inline bool is_fully_contiguous() const { return get_ncontig() == ndim; }
    
    // The new Arrays returned by slice() contain references
    // (not copies) to the data in the original Array.
    inline Array<T> slice(int axis, long start, long stop) const;
    inline Array<T> slice(int axis, long ix) const;   // returns array of dimension (ndim-1)

    // The new Arrays returned by transpose() contain references (not copies) to data in
    // the original Array. (FIXME transpose() hasn't been systematically tested!)

    inline Array<T> transpose(const int *perm) const;
    inline Array<T> transpose(const std::vector<int> &perm) const;
    inline Array<T> transpose(std::initializer_list<int> ix) const;

    // Reshape-by-reference. Throws an exception if either (1) requested shape is
    // incompatible with the current shape; (2) current strides don't permit axes to
    // be combined without copying. (FIXME reshape() hasn't been systematically tested!)
    
    inline Array<T> reshape(int ndim, const long *shape) const;
    inline Array<T> reshape(const std::vector<long> &shape) const;
    inline Array<T> reshape(std::initializer_list<long> shape) const;

    // Type casting/conversion:
    //
    //   - (Array<T> &) can be implicitly converted to (Array<void> &).
    //
    //   - cast(): zero-copy conversion of Array types, throws exception if datatypes are incompatible.
    //     (Morally similar to std::dynamic_pointer_cast()).
    //
    //   - convert_dtype(): returns copy of Array, with new datatype.
    //     (Example: convert Array<int> -> Array<float>).
    //
    // FIXME current implementation of convert_dtype() is primitive:
    //   - src/dst datatypes must be known at compile time (no Array<void>).
    //   - both arrays must be on host
    //   - slow!
    
    // Implicit conversions. The weird templating avoids instantiating for (T==void).    
    template<typename U=T, typename=enable_if_non_void<U>> inline operator Array<void>& ();
    template<typename U=T, typename=enable_if_non_void<U>> inline operator const Array<void>& () const;

    // cast(): zero-copy conversion of (Array<T> &), throws exception if datatypes are incompatible.
    template<typename U> inline Array<U>& cast();
    template<typename U> inline const Array<U>& cast() const;

    // convert_dtype(): returns copy of Array, with new datatype.
    template<typename Tdst> inline Array<Tdst> convert_dtype() const;
    template<typename Tdst> inline Array<Tdst> convert_dtype(int aflags) const;

    //
    // Remaining methods are intended for debugging/testing.
    //
    
    inline bool shape_equals(int ndim, const long *shape) const;
    inline bool shape_equals(const std::vector<long> &shape) const;
    inline bool shape_equals(std::initializer_list<long> shape) const;
    template<typename T2> inline bool shape_equals(const Array<T2> &a) const;    

    inline std::string shape_str() const;
    inline std::string stride_str() const;

    // Throws exception on failure.
    inline void check_invariants() const;
        
    // at(): range-checked accessor.
    // (I'm reserving operator[] for an unchecked accessor.)
    // The weird templating avoids instantiating for (T==void).
    // Slow, and intended only for debugging!

    template<typename U=T, typename=enable_if_non_void<U>> inline U& at(int ndim, const long *ix);
    template<typename U=T, typename=enable_if_non_void<U>> inline U& at(const std::vector<long> &ix);
    template<typename U=T, typename=enable_if_non_void<U>> inline U& at(std::initializer_list<long> ix);
    
    template<typename U=T, typename=enable_if_non_void<U>> inline const U& at(int ndim, const long *ix) const;
    template<typename U=T, typename=enable_if_non_void<U>> inline const U& at(const std::vector<long> &ix) const;
    template<typename U=T, typename=enable_if_non_void<U>> inline const U& at(std::initializer_list<long> ix) const;
    
    // ix_start(): for looping over array indices (not high-performance):
    //
    //    Array<T> arr;
    //    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix)) {
    //        T x = arr.at(ix);
    //        // ...
    //    }
    //    
    // Warning: ix_valid() is not a general-purpose index validator!
    // It only works on the output of ix_start() -> ix_next() -> ...
    // Slow, and intended only for debugging!
    
    inline std::vector<long> ix_start() const;
    inline bool ix_valid(const std::vector<long> &ix) const;
    inline void ix_next(std::vector<long> &ix) const;

    // Helper function for constructors.
    inline void _construct(const Dtype &dtype_, int ndim_, const long *shape_, const long *strides_, int aflags_);
    
    // "Cheat" accessor, which gives a non-const reference to a const Array.
    template<typename U=T, typename=enable_if_non_void<U>> inline U& _at(int ndim, const long *ix) const;
};


// Alternate interfaces to some of the Array methods above:
//   array_get_ncontig()   alternate interface to Array<T>::get_ncontig()
//   array_fill()          alternate interface to Array<T>::fill()
//   array_slice()         alternate interface to Array<T>::slice()
//   array_transpose()     alternate interface to Array<T>::transpose()
//   array_reshape()       alternate interface to Array<T>::reshape()

extern int array_get_ncontig(const Array<void> &arr);
extern void array_fill(Array<void> &dst, const Array<void> &src, bool noisy=false);
extern void array_slice(Array<void> &dst, const Array<void> &src, int axis, long ix);
extern void array_slice(Array<void> &dst, const Array<void> &src, int axis, long start, long stop);
extern void array_transpose(Array<void> &dst, const Array<void> &src, const int *perm);
extern void array_reshape(Array<void> &dst, const Array<void> &src, int dst_ndim, const long *dst_shape);

// Checks all Array invariants except dtype.
extern void _check_array_invariants(const Array<void> &arr);

// Misc helpers.
extern bool _tuples_equal(int ndim1, const long *shape1, int ndim2, const long *shape2);
extern std::string _tuple_str(int ndim, const long *shape);


// -------------------------------------------------------------------------------------------------
//
// The rest of this source file contains implementations of inline member functions.
// First, constructors.


// Default constructor: caller is responsible for initializing all members,
// then calling Array::check_invariants(). Useful in non-standard situations
// that can't be covered by any of the "stock" constructors.

template<typename T>
inline Array<T>::Array()
{
    if constexpr (!std::is_void_v<T>)
	dtype = Dtype::native<T>();
    
    for (int i = ndim; i < ArrayMaxDim; i++)
	shape[i] = strides[i] = 0;

    // No call to this->check_invariants() here.
}


// The next six Array constructors:
//   - do not have a runtime 'dtype' arg
//   - throw exceptions if called with T=void.
//   - may have a 'strides' argument.


template<typename T>
inline Array<T>::Array(const std::vector<long> &shape_, int aflags_)
    : Array(shape_.size(), &shape_[0], nullptr, aflags_) { }

template<typename T>
inline Array<T>::Array(std::initializer_list<long> shape_, int aflags_)
    : Array(shape_.size(), shape_.begin(), nullptr, aflags_) { }

template<typename T>
inline Array<T>::Array(int ndim_, const long *shape_, int aflags_)
{
    static_assert(!std::is_void_v<T>, "Array<T> constructor: If T=void, then you must call a constructor with a runtime dtype");
    this->_construct(Dtype::native<T>(), ndim_, shape_, nullptr, aflags_);
}


// This little device is useful in chaining Array constructors with shape + strides.
template<class C> inline int ndim_ss(const C &shape, const C &strides)
{
    if (shape.size() != strides.size())
	throw std::runtime_error("shape/strides length mismatch in Array constructor");
    return shape.size();
}

template<typename T>
inline Array<T>::Array(const std::vector<long> &shape_, const std::vector<long> &strides_, int aflags_)
    : Array(ndim_ss(shape_,strides_), &shape_[0], &strides_[0], aflags_) { }

template<typename T>
inline Array<T>::Array(std::initializer_list<long> shape_, std::initializer_list<long> strides_, int aflags_)
    : Array(ndim_ss(shape_,strides_), shape_.begin(), strides_.begin(), aflags_) { }

template<typename T>
inline Array<T>::Array(int ndim_, const long *shape_, const long *strides_, int aflags_)
{
    static_assert(!std::is_void_v<T>, "Array<T> constructor: If T=void, then you must call a constructor with a runtime dtype");
    xassert(strides_ != nullptr);    
    this->_construct(Dtype::native<T>(), ndim_, shape_, strides_, aflags_);
}


// The next three Array constructors:
//   - have a runtime 'dtype' arg
//   - are callable with (T == void).
//   - if called with (T != void), check consistency between runtime 'dtype' arg, and compile-time type T.
//   - do not have a 'strides' argument.


template<typename T>
inline Array<T>::Array(const Dtype &dtype_, const std::vector<long> &shape_, int aflags_)
    : Array(dtype_, shape_.size(), &shape_[0], aflags_) { }

template<typename T>
inline Array<T>:: Array(const Dtype &dtype_, std::initializer_list<long> shape_, int aflags_)
    : Array(dtype_, shape_.size(), shape_.begin(), aflags_) { }


template<typename T>
inline Array<T>::Array(const Dtype &dtype_, int ndim_, const long *shape_, int aflags_)
{
    this->_construct(dtype_, ndim_, shape_, nullptr, aflags_);
}


// Helper function for constructors.
template<typename T>
inline void Array<T>::_construct(const Dtype &dtype_, int ndim_, const long *shape_, const long *strides_, int aflags_)
{
    _check_dtype<T> (dtype_, "Array constructor");
    
    xassert(ndim_ >= 0);
    xassert(ndim_ <= ArrayMaxDim);
    xassert((ndim_ == 0) || (shape_ != nullptr));  // if strides_ is null, then array is contiguous.
    // check_aflags() will be called in _af_alloc() below.

    this->ndim = ndim_;
    this->dtype = dtype_;
    this->aflags = aflags_;
    this->size = ndim_ ? 1 : 0;  // updated in loop below
    long nalloc = size;
	
    for (int d = 0; d < ndim; d++) {
	shape[d] = shape_[d];
	strides[d] = strides_ ? strides_[d] : size;
	
	xassert(shape[d] >= 0);
	xassert(strides[d] >= 0);
	
	size *= shape_[d];
	nalloc += (shape_[d]-1) * strides[d];
	// Note that if array is contiguous, then nalloc==size.
    }

    for (int d = ndim; d < ArrayMaxDim; d++)
	shape[d] = strides[d] = 0;

    // Note: _af_alloc() calls check_aflags().
    // Note: if nalloc==0, then _af_alloc() returns an empty pointer.
    this->base = _af_alloc(dtype, nalloc, aflags);
    this->data = reinterpret_cast<T *> (base.get());
    this->check_invariants();    
}


// -------------------------------------------------------------------------------------------------
//
// Member functions which move data around: fill(), clone(), to_gpu(), to_host().


template<typename T> template<typename T2>
inline void Array<T>::fill(const Array<T2> &src, bool noisy)
{
    // Implicit conversion (Array<T> &) -> (Array<void> &).
    array_fill(*this, src, noisy);
}


template<typename T>
inline Array<T> Array<T>::clone(int aflags_) const
{
    Array<T> ret(this->ndim, this->shape, aflags_ & ~af_initialization_flags);
    array_fill(ret, *this);
    return ret;
}

template<typename T>
inline Array<T> Array<T>::clone() const
{
    return this->clone(this->aflags & af_location_flags);
}

template<typename T>
inline Array<T> Array<T>::to_gpu() const
{
    return this->on_gpu() ? (*this) : this->clone(af_gpu);
}

template<typename T>
inline Array<T> Array<T>::to_host(bool registered) const
{
    int dst_flags = registered ? af_rhost : af_uhost;
    return this->on_host() ? (*this) : this->clone(dst_flags);
}


// -------------------------------------------------------------------------------------------------
//
// Member functions for axis manipulation: get_ncontig(), slice(), transpose(), reshape().


template<typename T>
inline int Array<T>::get_ncontig() const
{
    // Note implicit conversion (Array<T> &) -> (Array<void> &).
    return array_get_ncontig(*this);
}


template<typename T>
inline Array<T> Array<T>::slice(int axis, long ix) const
{
    Array<T> ret;
    array_slice(ret, *this, axis, ix);
    return ret;
}


template<typename T>
inline Array<T> Array<T>::slice(int axis, long start, long stop) const
{
    Array<T> ret;
    array_slice(ret, *this, axis, start, stop);
    return ret;
}


template<typename T>
inline Array<T> Array<T>::transpose(const int *perm) const
{
    Array<T> ret;
    array_transpose(ret, *this, perm);
    return ret;
}


template<typename T>
inline Array<T> Array<T>::transpose(const std::vector<int> &perm) const
{
    xassert(long(perm.size()) == ndim);
    return this->transpose(&perm[0]);
}


template<typename T>
inline Array<T> Array<T>::transpose(std::initializer_list<int> perm) const
{
    xassert(long(perm.size()) == ndim);
    return this->transpose(perm.begin());
}


template<typename T>
Array<T> Array<T>::reshape(int ndim_, const long *shape_) const
{
    Array<T> ret;
    array_reshape(ret, *this, ndim_, shape_);
    return ret;
}

template<typename T>
inline Array<T> Array<T>::reshape(const std::vector<long> &shape) const
{
    return this->reshape(shape.size(), &shape[0]);
}

template<typename T>
inline Array<T> Array<T>::reshape(std::initializer_list<long> shape) const
{
    return this->reshape(shape.size(), shape.begin());
}


// -------------------------------------------------------------------------------------------------
//
// Member functions intended for debugging/testing:
//  - ix_start(), ix_valid(), ix_next()
//  - shape_equals(), shape_str(), stride_str()
//  - check_invariants()
//  - at().


template<typename T>
inline std::vector<long> Array<T>::ix_start() const
{
    return std::vector<long> (ndim, 0);
}

template<typename T>
inline bool Array<T>::ix_valid(const std::vector<long> &ix) const
{
    // Warning: ix_valid() is not a general-purpose index validator!
    // Only works on the output of ix_start() -> ix_next() -> ...
    return (size > 0) && (ix[0] < shape[0]);
}

template<typename T>
inline void Array<T>::ix_next(std::vector<long> &ix) const
{
    for (int d = ndim-1; d >= 1; d--) {
	if (ix[d] < shape[d]-1) {
	    ix[d]++;
	    return;
	}
	ix[d] = 0;
    }
    
    if (ndim > 0)
	ix[0]++;
}


template<typename T>
inline bool Array<T>::shape_equals(int ndim_, const long *shape_) const
{
    return _tuples_equal(this->ndim, this->shape, ndim_, shape_);
}

template<typename T>
inline bool Array<T>::shape_equals(const std::vector<long> &shape) const
{
    return _tuples_equal(this->ndim, this->shape, shape.size(), &shape[0]);
}

template<typename T>
inline bool Array<T>::shape_equals(std::initializer_list<long> shape_) const
{
    return _tuples_equal(this->ndim, this->shape, shape_.size(), shape_.begin());
}

template<typename T> template<typename T2>
inline bool Array<T>::shape_equals(const Array<T2> &a) const
{
    return _tuples_equal(this->ndim, this->shape, a.ndim, a.shape);
}

template<typename T>
inline std::string Array<T>::shape_str() const
{
    return _tuple_str(ndim, shape);
}

template<typename T>
inline std::string Array<T>::stride_str() const
{
    return _tuple_str(ndim, strides);
}

template<typename T>
inline void Array<T>::check_invariants() const
{
    _check_dtype<T> (dtype, "check_invariants");
    _check_array_invariants(*this);
}


// Array<T>::at(): range-checked accessor.
// (I'm reserving operator[] for an unchecked accessor.)
// The weird double template avoids instantiating for (T==void).


template<typename T> template<typename U, typename X>
inline U& Array<T>::at(int nd, const long *ix) { return _at(nd, ix); }

template<typename T> template<typename U, typename X>
inline U& Array<T>::at(const std::vector<long> &ix) { return _at(ix.size(), &ix[0]); }

template<typename T> template<typename U, typename X>
inline U& Array<T>::at(std::initializer_list<long> ix) { return _at(ix.size(), ix.begin()); }

template<typename T> template<typename U, typename X>
inline const U& Array<T>::at(int nd, const long *ix) const { return _at(nd, ix); }

template<typename T> template<typename U, typename X>
inline const U& Array<T>::at(const std::vector<long> &ix) const { return _at(ix.size(), &ix[0]); }

template<typename T> template<typename U, typename X>
inline const U& Array<T>::at(std::initializer_list<long> ix) const { return _at(ix.size(), ix.begin()); }


template<typename T> template<typename U, typename X>
inline U& Array<T>::_at(int nd, const long *ix) const
{
    xassert(on_host());
    xassert(this->ndim == nd);
    
    long pos = 0;
    for (int d = 0; d < nd; d++) {
	xassert(ix[d] >= 0 && ix[d] < shape[d]);
	pos += ix[d] * strides[d];
    }
    
    return data[pos];
}


// -------------------------------------------------------------------------------------------------
//
// Type casting/conversion:
//
//   - Array<T> can be implicitly converted to Array<void>.
//
//   - cast(): zero-copy conversion of Array types, throws exception if datatypes are incompatible.
//     (Morally similar to std::dynamic_pointer_cast()).
//
//   - convert_dtype(): returns copy of Array, with new datatype.
//     (Example: convert Array<int> -> Array<float>).


// Implicit conversions (Array<T> &) -> (Array<void> &) and (const Array<T> &) -> (const Array<void> &).
// (The weird double-templating avoids instantiating for (T==void), see declaration above.)

template<typename T> template<typename U, typename X>
inline Array<T>::operator Array<void>& ()
{
	return reinterpret_cast<Array<void> &> (*this);
}

template<typename T> template<typename U, typename X>
inline Array<T>::operator const Array<void>& () const
{
    return reinterpret_cast<const Array<void> &> (*this);
}


// cast(): explicit pointer conversion with runtime type-checking.
// (Morally similar to std::dynamic_pointer_cast()).

template<typename T> template<typename U>
inline Array<U>& Array<T>::cast()
{
    _check_dtype<U> (dtype, "Array::cast()");
    return reinterpret_cast<Array<U> &> (*this);
}

template<typename T> template<typename U>
inline const Array<U>& Array<T>::cast() const
{
    _check_dtype<U> (dtype, "Array::cast()");
    return reinterpret_cast<const Array<U> &> (*this);
}


// convert_dtype(): returns copy of Array, with new datatype.
// (Example: convert Array<int> -> Array<float>).
//
// Current implementation of convert_dtype() is primitive:
//   - src/dst datatypes must be known at compile time (no Array<void>).
//   - both arrays must be on host
//   - slow!
//
// Element conversion is done using C++ type conversion, except in cases
// (__half) <-> (float or double), when we call cuda intrinsics such as
// __float2half().

// Default dtype converter (just use C++ type conversion)
template<typename Tdst, typename Tsrc>
struct dtype_converter
{
    static inline Tdst convert(Tsrc x) { return x; }
};

// Dtype conversion float -> __half
template<> struct dtype_converter<__half, float>
{
    static inline __half convert(float x) { return __float2half(x); }
};

// Dtype conversion double -> __half
template<> struct dtype_converter<__half, double>
{
    static inline __half convert(double x) { return __double2half(x); }
};


// Dtype conversion __half -> float
template<> struct dtype_converter<float, __half>
{
    static inline float convert(__half x) { return __half2float(x); }
};

// Dtype conversion __half -> double
template<> struct dtype_converter<double, __half>
{
    // CUDA doesn't define __half2double()
    static inline double convert(__half x) { return __half2float(x); }
};


template<typename Tdst, typename Tsrc>
inline void convert_dtype_helper(Tdst *dst,
				 const Tsrc *src,
				 int nouter_dims,
				 const long *outer_shape,			    
				 const long *outer_dst_strides,
				 const long *outer_src_strides,
				 long nelts_contig)
{
    if (nouter_dims > 0) {
	for (int i = 0; i < outer_shape[0]; i++) {
	    convert_dtype_helper(dst + i * outer_dst_strides[0],
				 src + i * outer_src_strides[0],
				 nouter_dims - 1,
				 outer_shape + 1,
				 outer_dst_strides + 1,
				 outer_src_strides + 1,
				 nelts_contig);
	}
    }
    else {
	for (int i = 0; i < nelts_contig; i++)
	    dst[i] = dtype_converter<Tdst,Tsrc>::convert(src[i]);
    }
}


template<typename Tsrc> template<typename Tdst>
inline Array<Tdst> Array<Tsrc>::convert_dtype(int aflags) const
{
    xassert(on_host());           // src array must be on host
    xassert(af_on_host(aflags));  // dst array must be on host
    Array<Tdst> dst(ndim, shape, aflags);

    int ncontig = get_ncontig();
    int nouter = ndim - ncontig;

    long nelts_contig = 1;
    for (int i = nouter; i < ndim; i++)
	nelts_contig *= shape[i];

    convert_dtype_helper(dst.data, this->data, nouter, shape, dst.strides, strides, nelts_contig);
    return dst;
}


template<typename Tsrc> template<typename Tdst>
inline Array<Tdst> Array<Tsrc>::convert_dtype() const
{
    return this->convert_dtype<Tdst> (aflags & af_location_flags);
}

    
} // namespace ksgpu

#endif // _KSGPU_ARRAY_HPP
