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


}} // namespace PYBIND11_NAMESPACE::detail

#endif  // _KSGPU_PYBIND11_HPP
