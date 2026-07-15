#ifndef _KSGPU_PYBIND11_HPP
#define _KSGPU_PYBIND11_HPP

#include <pybind11/pybind11.h>

#include "Dtype.hpp"
#include "Array.hpp"
#include "xassert.hpp"

// convert_array_from_python(), convert_array_to_python(), array_type_name<T>
#include "pybind11_utils.hpp"


// type_caster<> converters: these must be available at compile time,
// to any pybind11 extension module which uses ksgpu::Array<T>.
//
// Note: conversion is zero-copy and does NO cuda stream synchronization; all
// ordering with asynchronous GPU work is the caller's/binding's responsibility.
// See the "SYNCHRONIZATION WARNING" comment in src_pybind11/pybind11_utils.cpp.

namespace PYBIND11_NAMESPACE { namespace detail {
#if 0
}} // editor
#endif


template<typename T>
struct type_caster<ksgpu::Array<T>>
{
    // This macro establishes the name 'Array' in in function signatures,
    // and declares a local variable 'value' of type ksgpu::Array<T>>.

    PYBIND11_TYPE_CASTER(ksgpu::Array<T>, ksgpu::array_type_name<T>::value);
    
    // load(): convert python -> C++.
    // FIXME for now, we ignore the 'convert' argument.

    bool load(handle src, bool convert)
    {
        ksgpu::Dtype dt_expected;
        
        if constexpr (!std::is_void_v<T>)
            dt_expected = ksgpu::Dtype::native<T>();
        
        // Throws a C++ exception on failure. (I tried a few ways of reporting
        // failure, including calling PyErr_SetString() and returning false,
        // but I liked throwing a C++ exception best.)
        //
        // If 'dt_expected' is an empty type (i.e. flags==nbits==0) then no
        // type-checking is performed.
        
        ksgpu::convert_array_from_python(this->value, src.ptr(), dt_expected, convert);
        return true;
    }
    
    // cast(): convert C++ -> python
    // FIXME for now, we ignore the 'policy' and 'parent' args.

    static handle cast(ksgpu::Array<T> src, return_value_policy policy, handle parent)
    {
        // On failure, ksgpu::convert_array_to_python() calls PyErr_SetString()
        // and returns NULL. (I tried a few ways of reporting failure, and I liked
        // this way best.)

        return ksgpu::convert_array_to_python(src, policy, parent);
    }
};


// type_caster<ksgpu::Dtype>: python code sees numpy.dtype objects, C++ code sees
// ksgpu::Dtype, with conversion at the language boundary in both directions.
//
//   - python -> C++: accepts None (-> empty Dtype), numpy dtypes, numpy scalar
//     types (e.g. np.float32), cupy dtypes, and strings. Strings are parsed with
//     Dtype::from_str() first, which accepts some names that numpy doesn't have
//     (e.g. "complex16+16", "int7"); numpy-only names (e.g. "complex64") fall
//     back to numpy parsing. Python ints are rejected: numpy would interpret
//     them as "type numbers" (np.dtype(1) is int8), which seems like a footgun.
//
//   - C++ -> python: an empty Dtype converts to None; anything else converts to
//     the corresponding numpy.dtype. A valid ksgpu::Dtype with no numpy
//     equivalent (e.g. "int7") raises an exception -- if a python-visible
//     member/return value can hold such a dtype, expose it as a string instead.

template<>
struct type_caster<ksgpu::Dtype>
{
    // This macro establishes the name 'numpy.dtype' in function signatures,
    // and declares a local variable 'value' of type ksgpu::Dtype.

    PYBIND11_TYPE_CASTER(ksgpu::Dtype, const_name("numpy.dtype"));

    // load(): convert python -> C++.
    // FIXME for now, we ignore the 'convert' argument (same as Array<T> above).

    bool load(handle src, bool convert)
    {
        // Throws a C++ exception on failure (same error-reporting convention
        // as Array<T> above).
        ksgpu::convert_dtype_from_python(this->value, src.ptr());
        return true;
    }

    // cast(): convert C++ -> python
    // FIXME for now, we ignore the 'policy' and 'parent' args.

    static handle cast(ksgpu::Dtype src, return_value_policy policy, handle parent)
    {
        // Throws a C++ exception on failure (unlike Array<T> above: if cast()
        // returns NULL, pybind11 raises a generic "Unable to convert function
        // return value" TypeError that would clobber the error message).
        return ksgpu::convert_dtype_to_python(src);
    }
};


}} // namespace PYBIND11_NAMESPACE::detail

#endif  // _KSGPU_PYBIND11_HPP
