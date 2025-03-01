// For an explanation of NO_IMPORT_ARRAY + PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu_pybind11.cu.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_ksgpu
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <complex>
#include <iostream>

#include "../include/ksgpu/Dtype.hpp"
#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/cuda_utils.hpp"    // CUDA_CALL()
#include "../include/ksgpu/string_utils.hpp"  // tuple_str()
#include "../include/ksgpu/pybind11_utils.hpp"

using namespace std;

namespace ksgpu {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// String helper functions


static string py_str(PyObject *x)
{
    // FIXME using pybind11 as a clutch here.
    // I'd prefer to use the python C-api directly, to guarantee that no exception is thrown.
    return string(pybind11::str(x));
}

static string py_type_str(PyObject *x)
{
    PyObject *t = (PyObject *) Py_TYPE(x);
    
    if (!t) {
	PyErr_Clear();
	return "unknown";
    }

    return py_str(t);
}


// -------------------------------------------------------------------------------------------------
//
// Helper functions for DLDataType and ksgpu::Dtype.
//
// struct DLDataType {
//   uint8_t code;    // kDLInt, kDLUInt, kDLFloat, kDLBfloat, kDLComplex, kDLBool
//   uint8_t bits;
//   uint16_t lanes;  // 1 for scalar types, >1 for simd dtypes
// };


static const char *dl_type_code_to_str(int code)
{
    switch (code) {
    case kDLInt:
	return "int";
    case kDLUInt:
	return "uint";
    case kDLFloat:
	return "float";
    case kDLOpaqueHandle:
	return "opaque";
    case kDLBfloat:
	return "bfloat";
    case kDLComplex:
	return "complex";
    case kDLBool:
	return "bool";
    default:
	return "unrecognized";
    }
}


static string dl_type_to_str(DLDataType d)
{
    stringstream ss;
    ss << dl_type_code_to_str(d.code);

    if (d.code == kDLComplex) {
	int n2 = d.bits >> 1;
	ss << n2 << "+" << int(d.bits - n2);
    }
    else if ((d.code != kDLBool) || (d.bits != 1))
	ss << int(d.bits);

    if (d.lanes != 1)
	ss << "x" << int(d.lanes);

    return ss.str();

}


// Converts DLDataType to ksgpu::Dtype object.
// If conversion fails, returns an invalid Dtype (with flags == nbits == 0).
static ksgpu::Dtype dl_type_to_ksgpu_dtype(DLDataType d)
{
    ksgpu::Dtype ret;

    if ((d.bits == 0) || (d.lanes != 1))
	return ret;  // failure (note that ksgpu::Dtype doesn't support simd types).

    if (d.code == kDLInt)
	ret.flags = df_int;
    else if (d.code == kDLUInt)
	ret.flags = df_uint;
    else if (d.code == kDLFloat)
	ret.flags = df_float;
    else if (d.code == kDLComplex)
	ret.flags = (df_complex | df_float);
    else
	return ret;

    ret.nbits = d.bits;
    return ret;
}


// Returns (-1) on failure.
static int ksgpu_dtype_to_npy_type_code(const ksgpu::Dtype &dtype)
{
    // Reference: https://numpy.org/doc/stable/reference/c-api/dtype.html
    
    if (dtype.flags == df_int) {
	if (dtype.nbits == 8) return NPY_INT8;
	if (dtype.nbits == 16) return NPY_INT16;
	if (dtype.nbits == 32) return NPY_INT32;
	if (dtype.nbits == 64) return NPY_INT64;
    }
    else if (dtype.flags == df_uint) {
	if (dtype.nbits == 8) return NPY_UINT8;
	if (dtype.nbits == 16) return NPY_UINT16;
	if (dtype.nbits == 32) return NPY_UINT32;
	if (dtype.nbits == 64) return NPY_UINT64;
    }
    else if (dtype.flags == df_float) {
	if (dtype.nbits == 16) return NPY_FLOAT16;
	if (dtype.nbits == 32) return NPY_FLOAT32;
	if (dtype.nbits == 64) return NPY_FLOAT64;
    }
    else if (dtype.flags == (df_complex | df_float)) {
	// Numpy doesn't support complex16+16 (as of numpy 2.2.3).
	// if (dtype.nbits == 32) return NPY_COMPLEX32;
	if (dtype.nbits == 64) return NPY_COMPLEX64;
	if (dtype.nbits == 128) return NPY_COMPLEX128;
    }

    return -1;
}


// -------------------------------------------------------------------------------------------------
//
// Helper functions for DLDevice and DLDeviceType.
//
// enum DLDeviceType { kDLCUP, kDLCUDA, kDLCUDAHost, kDLCUDAManaged, ... };
//
// struct DLDevice {
//   DLDeviceType device_type;
//   int32_t device_id;  // 0 for vanilla CPU memory, pinned memory, or managed memory
// };


static const char *dl_device_type_to_str(DLDeviceType d)
{
    // Reference: https://dmlc.github.io/dlpack/latest/c_api.html#c.DLDeviceType
    
    switch (d) {
    case kDLCPU:
	return "kDLCPU";
    case kDLCUDA:
	return "kDLCUDA";
    case kDLCUDAHost:
	return "kDLCUDAHost";
    case kDLOpenCL:
	return "kDLOpenCL";
    case kDLVulkan:
	return "DLVulkan";
    case kDLMetal:
	return "kDLMetal";
    case kDLVPI:
	return "kDLVPI";
    case kDLROCM:
	return "kDLROCM";
    case kDLROCMHost:
	return "kDLROCMHost";
    case kDLExtDev:
	return "kDLExtDev";
    case kDLCUDAManaged:
	return "kDLCUDAManaged";
    case kDLOneAPI:
	return "kDLOneAPI";
    case kDLWebGPU:
	return "kDLWebGPU";
    case kDLHexagon:
	return "kDLHexagon";
    case kDLMAIA:
	return "kDLMAIA";
    default:
	return "unrecognized";
    }
}


// Returns 0 on failure.
static int dl_device_type_to_aflags(DLDeviceType d)
{
    switch (d) {
    case kDLCPU:
	return af_uhost;
	
    case kDLCUDAHost:  // Pinned CUDA CPU memory by cudaMallocHost
    case kDLROCMHost:  // Pinned ROCm CPU memory allocated by hipMallocHost
	return af_rhost;
	
    case kDLCUDA:
    // case kDLROCM:
	return af_gpu;

    case kDLCUDAManaged:
	return af_unified;

    default:
	return 0;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Array conversion part 1: python -> C++
//
// On failure, convert_array_from_python() throws a C++ exception.
// If 'dt_expected' is an empty type (i.e. flags==nbits==0) then no type-checking is performed.
//
// If the 'debug_prefix' argument is specified, then some debug info will be printed to stdout.
// This feature is wrapped by ksgpu.convert_array_from_python(). It is intended as a mechanism
// for tracing/debugging array conversion.
//
// FIXME: at some point I should try to implement "base compression", here and/or in the
// python -> C++ conversion.
//
// FIXME implement 'convert' argument.
//
// FIXME we currently use __dlpack__ for all array conversions. If the array is a numpy array,
// then the conversion can be done more efficiently by calling functions in the numpy C-API.


__attribute__ ((visibility ("default")))
void convert_array_from_python(Array<void> &dst, PyObject *src, Dtype dt_expected, bool convert, const char *debug_prefix)
{
    DLManagedTensor *mt = nullptr;    
    pybind11::object capsule;   // must hold reference for entire function
    
    if (debug_prefix != nullptr)
	cout << debug_prefix << ": testing for presence of __dlpack__ attr\n";
    
    PyObject *dlp = PyObject_GetAttrString(src, "__dlpack__");
    
    if (dlp) {
	if (debug_prefix != nullptr)
	    cout << debug_prefix << ": __dlpack__ attr found, now calling it with no arguments\n";

	// FIXME the dlpack documentation specifies that __dlpack__() should be called
	// with a keyword argument 'max_version'. However, this didn't seem to be implemented
	// in numpy/cupy (in June 2024). For now, we call __dlpack__() with no args, but
	// I might revisit this in the future.

	PyObject *rp = PyObject_CallNoArgs(dlp);
	capsule = pybind11::reinterpret_steal<pybind11::object> (rp);
	Py_DECREF(dlp);
    }

    if (capsule.ptr()) {
	if (debug_prefix != nullptr)
	    cout << debug_prefix << ": __dlpack__() returned, now testing whether return value is a capsule with name \"dltensor\"\n";
    
	// Okay if this pointer is NULL.
	mt = reinterpret_cast<DLManagedTensor *> (PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    
	if (debug_prefix && mt)
	    cout << debug_prefix << ": successfully extracted pointer from capsule" << endl;
    }
    
    
    PyErr_Clear();  // no-ops if PyErr is not set
    
    if (!mt) {
	stringstream ss;
	bool vflag = capsule.ptr() && PyCapsule_GetPointer(capsule.ptr(), "dltensor_versioned");
	PyErr_Clear();

	if (vflag) {
	    ss << "ksgpu::convert_array_from_python() received 'dltensor_versioned' object."
	       << " This is a planned dlpack feature which isn't implemented yet (in June 2024) in numpy/cupy."
	       << " Unfortunately some (minor) code changes will be needed in ksgpu to support it!";
	}
	else {
	    ss << "Couldn't convert python argument(s) to a C++ array."
	       << " You might need to wrap the argument in numpy.asarray(...) or cupy.asarray(...).";
	}
	
	ss << " The offending argument is: " << py_str(src)
	   << " and its type is " << py_type_str(src) << ".";

	throw pybind11::type_error(ss.str());
    }
    
    DLTensor &t = mt->dl_tensor;
    ksgpu::Dtype dtype = dl_type_to_ksgpu_dtype(t.dtype);
    int aflags = dl_device_type_to_aflags(t.device.device_type);
    int ndim = t.ndim;

    if (debug_prefix != nullptr) {
	cout << debug_prefix << ": dereferencing DLManagedTensor\n"
	     << "   data: " << t.data << "\n"
	     << "   device_type: " << t.device.device_type
	     << " (" << dl_device_type_to_str(t.device.device_type) << ")\n"
	     << "   device_id: " << t.device.device_id << "\n"
	     << "   ndim: " << t.ndim << "\n"
	     << "   dtype_code: " << int(t.dtype.code)
	     << " (" << dl_type_code_to_str(t.dtype.code) << ")\n"
	     << "   dtype_bits: " << int(t.dtype.bits) << "\n"
	     << "   dtype_lanes: " << t.dtype.lanes << "\n"
	     << "   byte_offset: " << t.byte_offset << "\n"
	     << "C++ dtype: " << dtype << "\n"
	     << "C++ aflags: " << aflag_str(aflags) << "\n";
	
	cout << debug_prefix << ": dereferencing shape" << endl;
	cout << "   shape: " << ksgpu::tuple_str(t.ndim, t.shape, " ") << "\n";
	
	if (t.strides == nullptr)
	    cout << debug_prefix << ": strides pointer is null" << endl;
	else {
	    cout << debug_prefix << ": dereferencing strides" << endl;
	    cout << "   strides: " << ksgpu::tuple_str(t.ndim, t.strides, " ") << "\n";
	}
    }

    if (ndim <= 0) {
	// Zero-dimensional arrays: these are awkward. In python a zero-dimensional array
	// is a scalar, whereas in ksgpu a zero-dimensional array is empty. I'm currently
	// making the most conservative choice, by throwing an exception if a zero-dimensional
	// array is encountered.

	const char *msg = "Converting zero-dimensional python arrays to C++ is currently not allowed";
	throw pybind11::type_error(msg);
    }
    
    if (ndim > ksgpu::ArrayMaxDim) {
	stringstream ss;
	ss << "Couldn't convert python argument to a C++ array."
	   << " The python argument is an array of dimension " << ndim
	   << ", and ksgpu::ArrayMaxDim=" << ksgpu::ArrayMaxDim;

	throw pybind11::type_error(ss.str());
    }

    if (!dtype.is_valid()) {
	stringstream ss;
	ss << "Couldn't convert python array to a C++ array."
	   << " The python array has dtype " << dl_type_to_str(t.dtype) << ","
	   << " which we don't (currently?) support."
	   << " The offending argument is: " << py_str(src)
	   << " and its type is " << py_type_str(src) << ".";

	throw pybind11::type_error(ss.str());
    }

    if ((dt_expected.flags || dt_expected.nbits) && (dtype != dt_expected)) {
	stringstream ss;
	ss << "Couldn't convert python argument to a C++ array: type mismatch."
	   << " The python array has dtype " << dl_type_to_str(t.dtype) << ","
	   << " and the C++ code expects dtype " << dt_expected << "."
	   << " The offending argument is: " << py_str(src)
	   << " and its type is " << py_type_str(src) << ".";

	throw pybind11::type_error(ss.str());
    }
    
    if (aflags == 0) {
	stringstream ss;
	ss << "Couldn't convert python array to a C++ array."
	   << " The python array has DLDeviceType=" << t.device.device_type
	   << " [" << dl_device_type_to_str(t.device.device_type) << "],"
	   << " which we don't currently support."
	   << " The offending argument is: " << py_str(src)
	   << " and its type is " << py_type_str(src) << ".";
	
	throw pybind11::type_error(ss.str());
    }

    if (aflags & af_gpu) {
	// FIXME this kludge is necessary since we don't currently have aflags
	// to keep track of which GPU the memory is located on!
    
	int current_device = -1;
	CUDA_CALL(cudaGetDevice(&current_device));

	if (t.device.device_id != current_device) {
	    stringstream ss;
	    ss << "Couldn't convert python array to a C++ array."
	       << " The python array has device_id=" << t.device.device_id
	       << " and the active cuda device is id=" << current_device << "."
	       << " The offending argument is: " << py_str(src)
	       << " and its type is " << py_type_str(src) << ".";
	    
	    throw pybind11::type_error(ss.str());
	}
    }

    dst.ndim = ndim;
    dst.dtype = dtype;
    dst.aflags = aflags;
    dst.size = ndim ? 1 : 0;  // updated in subsequent loop
    dst.data = (void *) ((char *)t.data + t.byte_offset);
    
    for (int i = ndim-1; i >= 0; i--) {
	// Note: if t.strides==NULL, then array is contiguous.
	dst.shape[i] = t.shape[i];
	dst.strides[i] = t.strides ? t.strides[i] : dst.size;
	dst.size *= t.shape[i];
    }
    
    for (int i = ndim; i < ArrayMaxDim; i++)
	dst.shape[i] = dst.strides[i] = 0;

    // C++ array holds reference to python object!
    // FIXME could be improved by pointer-chasing to base object.
    dst.base = shared_ptr<void> (src, Py_DecRef);
    Py_INCREF(src);

    dst.check_invariants();
    
    if (debug_prefix != nullptr)
	cout << debug_prefix << ": array converted succesfully" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Array conversion part 2: C++ -> python
// On failure, convert_array_to_python() returns NULL and sets PyErr.
//
// FIXME: at some point I should try to implement "base compression", here and/or in the
// python -> C++ conversion.
//
// FIXME implement return value policies (and the 'parent' argument).
//   https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies
//   https://github.com/pybind/pybind11/blob/master/include/pybind11/detail/common.h


__attribute__ ((visibility ("default")))
PyObject *convert_array_to_python(const Array<void> &src, pybind11::return_value_policy policy, pybind11::handle parent)
{
    if (!af_on_host(src.aflags)) {
	// FIXME currently C++ -> python array conversion is not implemented for GPU arrays.
	// However, I'm skeptical that this is a good idea (in code 

	const char *msg = "Currently C++ -> python array conversion is not implemented for GPU arrays";
	PyErr_SetString(PyExc_TypeError, msg);
	return NULL;
    }

    int ndim = src.ndim;
    int type_num = ksgpu_dtype_to_npy_type_code(src.dtype);

    if (ndim <= 0) {
	// Zero-dimensional arrays: these are awkward. In python a zero-dimensional array
	// is a scalar, whereas in ksgpu a zero-dimensional array is empty. I'm currently
	// making the most conservative choice, by throwing an exception if a zero-dimensional
	// array is encountered.

	const char *msg = "Converting zero-dimensional C++ arrays to python is currently not allowed";
	PyErr_SetString(PyExc_TypeError, msg);
	return NULL;
    }

    if (type_num < 0) {
	stringstream ss;
	ss << "Couldn't convert C++ array dtype " << src.dtype << " to python";

	// FIXME memory leak here (exception text, unlikely to be an issue in practice)
	string s = ss.str();
	const char *msg = strdup(s.c_str());
	
	if (!msg)
	    msg = "internal error: strdup() returned NULL";
	
	PyErr_SetString(PyExc_TypeError, msg);
	return NULL;
    }

    int nbits = src.dtype.nbits;
    int itemsize = (nbits >> 3);
    xassert((nbits > 0) && ((nbits & 7) == 0));
    xassert(ndim <= ksgpu::ArrayMaxDim);  // paranoid
    
    npy_intp npy_shape[ksgpu::ArrayMaxDim];
    for (int i = 0; i < ndim; i++)
	npy_shape[i] = src.shape[i];
	
    npy_intp npy_strides[ksgpu::ArrayMaxDim];
    for (int i = 0; i < ndim; i++)
	npy_strides[i] = src.strides[i] * itemsize;
    
    // Array creation: https://numpy.org/doc/stable/reference/c-api/array.html#creating-arrays
    // Array flags: https://numpy.org/doc/stable/reference/c-api/array.html#array-flags
    // PyArray_New() is defined in this source file: src/multiarray/ctors.c
    // Flags are defined in this source file: include/numpy/ndarraytypes.h
    
    PyObject *ret = PyArray_New(
	&PyArray_Type,   // PyTypeObject *subtype
	ndim,            // int nd
	npy_shape,       // npy_intp const *dims
	type_num,        // int type_num
	npy_strides,     // npy_intp const *strides
	src.data,        // void *data
	itemsize,        // int itemsize
        NPY_ARRAY_WRITEABLE,   // int flags
	NULL             // PyObject *obj (extra constructor arg, only used if subtype != &PyArrayType)
    );
    
    if (!ret)
	return NULL;

    // Paranoid!
    int flags = PyArray_FLAGS((PyArrayObject *) ret);
    xassert((flags & NPY_ARRAY_OWNDATA) == 0);
    
    // We need a mechanism for keeping a reference to 'base' (a shared_ptr<void>) in the
    // newly constructed numpy array.
    //
    // For now we use a hack (class PybindBasePtr).
    // FIXME defining a PyArray subclass would be cleaner and more efficient.
    //
    // The PybindBasePtr class is a trivial wrapper around shared_ptr<void>, but the
    // wrapper class is exported to python (via a pybind11::class_<> in the top-level
    // ksgpu extension module). This allows us to set the numpy 'base' member (which
    // must be a python object) to a PybindBasePtr instance.
    //
    // One nuisance issue: here in the C++ code, we need to convert a PybindBasePtr
    // instance to a (PyObject *). Surprisingly, pybind11::cast() doesn't work here,
    // and we need the following mysterious code.
    // (This reference was useful: https://github.com/pybind/pybind11/issues/1176)
    
    PybindBasePtr p(src.base);
    using caster = pybind11::detail::type_caster_base<PybindBasePtr>;
    pybind11::handle base_ptr = caster::cast(p, pybind11::return_value_policy::copy, pybind11::handle()); 
    
    // PyArray_SetBaseObject() checks whether 'base_ptr' is NULL.
    // PyArray_SetBaseObject() steals the reference to 'base' (even on error).
    // PyArray_SetBaseObject() is defined in this file: numpy/_core/src/multiarray/arrayobject.c
    
    int err = PyArray_SetBaseObject((PyArrayObject *) ret, base_ptr.ptr());

    if (err < 0) {
	Py_XDECREF(ret);
	return NULL;
    }

    return ret;
}


}  // namespace ksgpu
