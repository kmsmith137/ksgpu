#ifndef _KSGPU_PYBIND11_UTILS_HPP
#define _KSGPU_PYBIND11_UTILS_HPP

#include <pybind11/pybind11.h>
#include "dlpack.h"


namespace ksgpu {
#if 0
}  // editor
#endif


// Convert python -> C++.
// On failure, throws a C++ exception.
// If 'dt_expected' is an empty type (i.e. flags==nbits==0) then no type-checking is performed.
//
// If the 'debug_prefix' argument is specified, then some debug info will be printed to stdout.
// This feature is wrapped by ksgpu.convert_array_from_python(). It is intended as a mechanism
// for tracing/debugging array conversion.

extern void convert_array_from_python(
    Array<void> &dst, PyObject *src, const Dtype &dt_expected,
    bool convert, const char *debug_prefix = nullptr);


// Convert C++ -> python.
// On failure, calls PyErr_SetString() and returns NULL.

extern PyObject *convert_array_to_python(
    const Array<void> &src,
    pybind11::return_value_policy policy,
    pybind11::handle parent);


// PybindBasePtr: hack for array C++ -> python conversion.
// See comments in ksgpu/src_pybind11/ksgpu_pybind11_utils.cu.
// Must be visible at compile time in ksgpu/src_pybind11/ksgpu_pybind11.cu.

struct PybindBasePtr
{
    std::shared_ptr<void> p;
    PybindBasePtr(const std::shared_ptr<void> &p_) : p(p_) { }
};


// These array type names will appear in docstrings and error messages.
// (Via type_caster<Array<T>>, which is declared in pybind11.hpp)

template<typename> struct array_type_name { };
template<> struct array_type_name<float>                   { static constexpr auto value = pybind11::detail::const_name("Array<float>"); };
template<> struct array_type_name<double>                  { static constexpr auto value = pybind11::detail::const_name("Array<double>"); };
template<> struct array_type_name<std::complex<float>>     { static constexpr auto value = pybind11::detail::const_name("Array<complex<float>>"); };
template<> struct array_type_name<std::complex<double>>    { static constexpr auto value = pybind11::detail::const_name("Array<complex<double>>"); };
template<> struct array_type_name<long>                    { static constexpr auto value = pybind11::detail::const_name("Array<long>"); };
template<> struct array_type_name<int>                     { static constexpr auto value = pybind11::detail::const_name("Array<int>"); };
template<> struct array_type_name<short>                   { static constexpr auto value = pybind11::detail::const_name("Array<short>"); };
template<> struct array_type_name<char>                    { static constexpr auto value = pybind11::detail::const_name("Array<char>"); };
template<> struct array_type_name<ulong>                   { static constexpr auto value = pybind11::detail::const_name("Array<ulong>"); };
template<> struct array_type_name<uint>                    { static constexpr auto value = pybind11::detail::const_name("Array<uint>"); };
template<> struct array_type_name<ushort>                  { static constexpr auto value = pybind11::detail::const_name("Array<ushort>"); };
template<> struct array_type_name<unsigned char>           { static constexpr auto value = pybind11::detail::const_name("Array<uchar>"); };
template<> struct array_type_name<void>                    { static constexpr auto value = pybind11::detail::const_name("Array<void>"); };


}   // namespace ksgpu


#endif  // _KSGPU_PYBIND11_UTILS_HPP
