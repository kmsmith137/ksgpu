// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments later in this source filE.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_ksgpu
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "../include/ksgpu/pybind11.hpp"
#include "../include/ksgpu/cuda_utils.hpp"
#include "../include/ksgpu/mem_utils.hpp"
#include "../include/ksgpu/test_utils.hpp"
#include <pybind11/stl.h>
#include <iostream>


using namespace std;
using namespace ksgpu;

namespace py = pybind11;


// -----------------------------------  ksgpu toplevel -----------------------------------------


int get_cuda_num_devices()
{
    int count = 0;
    CUDA_CALL(cudaGetDeviceCount(&count));
    return count;
}


int get_cuda_device()
{
    int device = -1;
    CUDA_CALL(cudaGetDevice(&device));
    return device;
}


void set_cuda_device(int device)
{
    CUDA_CALL(cudaSetDevice(device));
}


string get_cuda_pcie_bus_id(int cuda_device)
{
    // CUDA guarantees the buffer size is at least 16 bytes
    constexpr int buflen = 16;
    
    char pcieBusId[buflen+1];
    memset(pcieBusId, 0, buflen+1);

    CUDA_CALL(cudaDeviceGetPCIBusId(pcieBusId, buflen, cuda_device));
    return string(pcieBusId);
}


// -------------------------------------------------------------------------------------------------
//
// Some ad hoc functions from when I was testing C++ <-> python array conversion.
// I decided to put these in the 'ksgpu.tests' submodule.
// These are not particularly well thought out!


static double _sum(Array<double> &arr)
{
    if (!arr.on_host()) {
        cout << "sum() copying array from GPU to host" << endl;
        arr = arr.to_host();
    }
    
    double sum = 0;
    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix))
        sum += arr.at(ix);
    
    return sum;
}


static void _double(Array<double> &arr)
{
    Array<double> asave;
    bool copy = !arr.on_host();

    if (copy) {
        cout << "double() copying array from GPU to host" << endl;
        asave = arr;
        arr = arr.to_host();
    }
    
    for (auto ix = arr.ix_start(); arr.ix_valid(ix); arr.ix_next(ix))
        arr.at({ix}) *= 2;

    if (copy) {
        cout << "double() copying array from host to GPU" << endl;
        asave.fill(arr);
    }
}    


static Array<int> _arange(int n)
{
    // Note af_verbose here.
    Array<int> ret({n}, af_uhost | af_verbose);
    
    for (int i = 0; i < n; i++)
        ret.at({i}) = i;
    
    return ret;
}


static void _convert_array_from_python(py::object &obj)
{
    ksgpu::Array<void> arr;

    ksgpu::convert_array_from_python(
        arr,                          // Array<void> &dst
        obj.ptr(),                    // PyObject *src
        ksgpu::Dtype(),               // Dtype dt_expected
        false,                        // bool convert
        "convert_array_from_python"   // const char *debug_prefix
    );
}


void _launch_busy_wait_kernel(Array<uint> &arr, double a40_sec, long stream_ptr)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t> (stream_ptr);
    ksgpu::launch_busy_wait_kernel(arr, a40_sec, s);
}


// -------------------------------------------------------------------------------------------------
//
// ArrayInfo: struct for returning array metadata to Python for verification in unit tests.


struct ArrayInfo
{
    vector<long> shape;
    vector<long> strides;
    string dtype_str;
    string location;      // "gpu", "rhost", "uhost", "unified"
    uintptr_t data_ptr;
    
    ArrayInfo() : data_ptr(0) { }
    
    ArrayInfo(const Array<void> &arr)
    {
        shape.assign(arr.shape, arr.shape + arr.ndim);
        strides.assign(arr.strides, arr.strides + arr.ndim);
        dtype_str = arr.dtype.str();
        data_ptr = reinterpret_cast<uintptr_t>(arr.data);
        
        if (arr.aflags & af_gpu)
            location = "gpu";
        else if (arr.aflags & af_rhost)
            location = "rhost";
        else if (arr.aflags & af_unified)
            location = "unified";
        else
            location = "uhost";
    }
};


// get_array_info(): Convert a Python array to C++ and return its metadata.
// This allows Python tests to verify that C++ sees the correct shape/strides/dtype/location.

static ArrayInfo get_array_info(const Array<void> &arr)
{
    return ArrayInfo(arr);
}


// make_strided_array(): Create a C++ array with specified shape/strides, then return to Python.
// This is useful for testing C++ -> Python conversion with non-contiguous arrays.
// The array is filled with sequential values 0, 1, 2, ... for easy verification.

static Array<void> make_strided_array(
    const vector<long> &shape,
    const vector<long> &strides,
    const string &dtype_str,
    bool on_gpu)
{
    Dtype dtype = Dtype::from_str(dtype_str);
    int ndim = shape.size();
    
    if ((int)strides.size() != ndim)
        throw runtime_error("make_strided_array: shape and strides must have same length");
    
    // Calculate the total buffer size needed for these strides
    long max_offset = 0;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0)
            throw runtime_error("make_strided_array: shape elements must be positive");
        if (strides[i] <= 0)
            throw runtime_error("make_strided_array: strides must be positive (ksgpu doesn't support negative strides)");
        max_offset += (shape[i] - 1) * strides[i];
    }
    long buffer_nelts = max_offset + 1;
    
    // Allocate buffer on host first, fill with sequential values
    int aflags_host = af_rhost;
    Array<void> buffer(dtype, {buffer_nelts}, aflags_host);
    
    // Fill buffer with 0, 1, 2, ... (as float64 for simplicity, then we rely on the fact
    // that we're using this for testing and the values will be recognizable)
    if (dtype.flags == df_float && dtype.nbits == 32) {
        float *p = static_cast<float*>(buffer.data);
        for (long i = 0; i < buffer_nelts; i++)
            p[i] = static_cast<float>(i);
    }
    else if (dtype.flags == df_float && dtype.nbits == 64) {
        double *p = static_cast<double*>(buffer.data);
        for (long i = 0; i < buffer_nelts; i++)
            p[i] = static_cast<double>(i);
    }
    else if (dtype.flags == df_int && dtype.nbits == 32) {
        int32_t *p = static_cast<int32_t*>(buffer.data);
        for (long i = 0; i < buffer_nelts; i++)
            p[i] = static_cast<int32_t>(i);
    }
    else if (dtype.flags == df_int && dtype.nbits == 64) {
        int64_t *p = static_cast<int64_t*>(buffer.data);
        for (long i = 0; i < buffer_nelts; i++)
            p[i] = static_cast<int64_t>(i);
    }
    else if (dtype.flags == df_uint && dtype.nbits == 32) {
        uint32_t *p = static_cast<uint32_t*>(buffer.data);
        for (long i = 0; i < buffer_nelts; i++)
            p[i] = static_cast<uint32_t>(i);
    }
    else if (dtype.flags == (df_complex | df_float) && dtype.nbits == 64) {
        complex<float> *p = static_cast<complex<float>*>(buffer.data);
        for (long i = 0; i < buffer_nelts; i++)
            p[i] = complex<float>(static_cast<float>(i), 0.0f);
    }
    else if (dtype.flags == (df_complex | df_float) && dtype.nbits == 128) {
        complex<double> *p = static_cast<complex<double>*>(buffer.data);
        for (long i = 0; i < buffer_nelts; i++)
            p[i] = complex<double>(static_cast<double>(i), 0.0);
    }
    else {
        throw runtime_error("make_strided_array: unsupported dtype " + dtype_str);
    }
    
    // Copy to GPU if needed
    if (on_gpu)
        buffer = buffer.to_gpu();
    
    // Create the strided view
    Array<void> ret;
    ret.data = buffer.data;
    ret.ndim = ndim;
    ret.dtype = dtype;
    ret.aflags = buffer.aflags;
    ret.base = buffer.base;
    ret.size = 1;
    
    for (int i = 0; i < ndim; i++) {
        ret.shape[i] = shape[i];
        ret.strides[i] = strides[i];
        ret.size *= shape[i];
    }
    for (int i = ndim; i < ArrayMaxDim; i++) {
        ret.shape[i] = 0;
        ret.strides[i] = 0;
    }
    
    ret.check_invariants();
    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// _CudaStreamWrapperBase: Internal pybind11 binding for CudaStreamWrapper.
//
// This class is NOT intended to be used directly from Python. Instead, users should use
// the Python class `ksgpu.CudaStreamWrapper`, which:
//
//   1. Inherits from cupy.cuda.ExternalStream (providing context manager support and
//      seamless cupy interop via isinstance() checks)
//
//   2. Holds a reference to this _CudaStreamWrapperBase object, which ensures proper
//      reference counting via std::shared_ptr<CUstream_st>
//
// The reference counting works as follows:
//
//   - CudaStreamWrapper (C++) contains std::shared_ptr<CUstream_st>
//   - When _CudaStreamWrapperBase is returned to Python, pybind11 creates a COPY,
//     incrementing the shared_ptr refcount
//   - The Python CudaStreamWrapper stores this _CudaStreamWrapperBase in self._cpp_wrapper
//   - When Python's GC collects the CudaStreamWrapper, _cpp_wrapper is destroyed,
//     decrementing the refcount
//   - When refcount reaches 0, cudaStreamDestroy() is called
//
// This design allows C++ code to return CudaStreamWrapper objects to Python while
// ensuring the underlying CUDA stream stays alive as long as Python holds a reference.


// _StreamHolderBase: Test helper class for verifying stream reference counting.
// Used in unit tests to create a stream, pass it to Python, then verify the stream
// stays alive after the C++ holder is destroyed.

struct _StreamHolderBase {
    CudaStreamWrapper stream;
    
    _StreamHolderBase() : stream(CudaStreamWrapper::create()) {
        std::cout << "_StreamHolderBase created, stream ptr=" 
                  << reinterpret_cast<uintptr_t>(stream.p.get()) << std::endl;
    }
    
    ~_StreamHolderBase() {
        std::cout << "_StreamHolderBase destroyed" << std::endl;
    }
    
    CudaStreamWrapper get_stream() { return stream; }
};


// -------------------------------------------------------------------------------------------------


struct Stash
{
    Array<void> x;
    
    Stash(const Array<void> &x_) : x(x_) { }
    ~Stash() { cout << "Stash destructor called!" << endl; }
    
    Array<void> get() { return x; }
    void clear() { x = Array<void> (); }
    
    // Return metadata about the stashed array (for testing)
    ArrayInfo info() const { return ArrayInfo(x); }
    
    // Check if stash is empty
    bool is_empty() const { return x.size == 0; }
};


// -------------------------------------------------------------------------------------------------


PYBIND11_MODULE(ksgpu_pybind11, m)  // extension module gets compiled to ksgpu_pybind11.so
{
    m.doc() = "ksgpu: a library of low-level utilities for cuda/cupy.";

    // Here is a quick summary of what you should do for import_array():
    //
    //   - In the main pybind11 source file (i.e. this file), before including
    //     any source files, put:
    //
    //       #define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_{modname}
    //
    //     where "modname" is e.g. 'ksgpu' (a per-extension-module string)
    //
    //   - If there are any other source files in the python extension module
    //     (e.g. ksgpu/src_pybind11/ksgpu_pybind11_utils.cu), then those
    //     files should contain (before including any source files):
    //
    //       #define NO_IMPORT_ARRAY
    //       #define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_{modname}
    // 
    // To understand the reasoning behind this, it's easiest to read the source
    // file core/include/numpy/__multiarray_api.h (search for PyArray_API). However,
    // note that this file is procedurally generated during the numpy build process,
    // so it will be in your python site-packages, but not in the numpy source code
    // in the github repo.
    //
    // Note: looks like _import_array() will fail if different numpy versions are
    // found at compile-time versus runtime.

    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "ksgpu: numpy.core.multiarray failed to import");
        return;
    }

    // -----------------------------------  ksgpu toplevel -----------------------------------------
    
    const char *baseptr_doc =
        "This helper class is a hack, used interally for converting C++ arrays to python\n"
        "It can't be instantiated or used from python.";

    // Reminder: struct PybindBasePtr is in pybind11_utils.{hpp,cu}.
    py::class_<PybindBasePtr>(m, "BasePtr", baseptr_doc);

    m.def("get_cuda_num_devices", &get_cuda_num_devices, "Returns number of cuda devices");
    
    m.def("get_cuda_device", &get_cuda_device, "Returns current cuda device");
    
    m.def("set_cuda_device", &set_cuda_device, "Sets current cuda device");
    
    m.def("get_cuda_pcie_bus_id", &get_cuda_pcie_bus_id,
          "Returns PCIe bus ID of specified cuda device, as string e.g. '0000:E1:00.0'",
          py::arg("device"));
    
    // -----------------------------------  aflags and aflag_str  -----------------------------------
    
    // Location flags
    m.attr("af_gpu") = af_gpu;
    m.attr("af_uhost") = af_uhost;
    m.attr("af_rhost") = af_rhost;
    m.attr("af_unified") = af_unified;
    
    // Initialization flags
    m.attr("af_zero") = af_zero;
    m.attr("af_random") = af_random;
    
    // Mmap flags
    m.attr("af_mmap_small") = af_mmap_small;
    m.attr("af_mmap_huge") = af_mmap_huge;
    m.attr("af_mmap_try_huge") = af_mmap_try_huge;
    
    // Debug flags
    m.attr("af_guard") = af_guard;
    m.attr("af_verbose") = af_verbose;
    
    // aflag_str function
    m.def("aflag_str", &aflag_str,
          "Convert allocation flags to human-readable string (e.g. 'af_gpu | af_zero')",
          py::arg("flags"));
    
    // ------------------------------  ksgpu.tests submodule  --------------------------------------

    // ArrayInfo: struct for returning array metadata to Python
    py::class_<ArrayInfo>(m, "ArrayInfo",
        "Metadata about a C++ array, returned by get_array_info() and Stash.info()")
        .def_readonly("shape", &ArrayInfo::shape)
        .def_readonly("strides", &ArrayInfo::strides)
        .def_readonly("dtype_str", &ArrayInfo::dtype_str)
        .def_readonly("location", &ArrayInfo::location)
        .def_readonly("data_ptr", &ArrayInfo::data_ptr)
        .def("__repr__", [](const ArrayInfo &info) {
            stringstream ss;
            ss << "ArrayInfo(shape=[";
            for (size_t i = 0; i < info.shape.size(); i++)
                ss << (i ? "," : "") << info.shape[i];
            ss << "], strides=[";
            for (size_t i = 0; i < info.strides.size(); i++)
                ss << (i ? "," : "") << info.strides[i];
            ss << "], dtype='" << info.dtype_str << "', location='" << info.location << "')";
            return ss.str();
        })
    ;
    
    m.def("get_array_info", &get_array_info,
          "Convert a Python array to C++ and return its metadata (shape, strides, dtype, location).\n"
          "Useful for testing that C++ sees the correct array properties.",
          py::arg("arr"));
    
    m.def("make_strided_array", &make_strided_array,
          "Create a C++ array with specified shape/strides, filled with sequential values 0,1,2,...\n"
          "Returns the array converted to Python (numpy or cupy depending on on_gpu).\n"
          "Useful for testing C++ -> Python conversion with non-contiguous arrays.",
          py::arg("shape"), py::arg("strides"), py::arg("dtype"), py::arg("on_gpu"));
     
    const char *stash_doc =
        "Helper class intended for testing C++ <-> python array conversion.\n"
        "   s = Stash(numpy_or_cupy_array)     # converts array to C++ and saves it\n"
        "   arr = Stash.get()                  # converts array to python and returns it\n"
        "   info = Stash.info()                # returns ArrayInfo about the stashed array";


    py::class_<Stash>(m, "Stash", stash_doc)
        .def(py::init<const Array<void> &>(), py::arg("arr"))
        .def("get", &Stash::get)
        .def("clear", &Stash::clear)
        .def("info", &Stash::info)
        .def("is_empty", &Stash::is_empty)
    ;
    
    m.def("sum", &_sum,   
          "Equivalent to {numpy,cupy}.sum(arr), but uses the python -> C++ converter",
          py::arg("arr"));
          
    m.def("double", &_double,
          "Equivalent to 'arr *= 2', but uses the python -> C++ converter",
          py::arg("arr"));

    m.def("arange", &_arange,
          "Equivalent to numpy.arange(n), but uses the C++ -> python converter."
          " (FIXME CPU-only for now.)",
          py::arg("n"));

    m.def("convert_array_from_python", &_convert_array_from_python,
          "Converts array from python to C++, but with a lot of debug output."
          " This is intended as a mechanism for tracing/debugging array conversion.",
          py::arg("arr"));

    m.def("_launch_busy_wait_kernel", &_launch_busy_wait_kernel,
          py::arg("arr"), py::arg("a40_sec"), py::arg("stream_ptr"));
    
    // --------------------------  CudaStreamWrapper bindings  ---------------------------------
    //
    // See comments above _StreamHolderBase for detailed explanation of the design.
    
    py::class_<CudaStreamWrapper>(m, "_CudaStreamWrapperBase",
        "Internal: C++ CUDA stream wrapper with reference counting.\n\n"
        "DO NOT USE DIRECTLY. Use ksgpu.CudaStreamWrapper instead, which inherits from\n"
        "cupy.cuda.ExternalStream and holds a reference to this object for proper\n"
        "reference counting.\n\n"
        "The underlying CUDA stream is destroyed when all Python and C++ references\n"
        "to this object are released.")
        .def(py::init<>(), "Create wrapper for default stream (ptr=0)")
        .def_static("create", &CudaStreamWrapper::create,
            "Create a new CUDA stream with optional priority (default=0)", 
            py::arg("priority") = 0)
        .def_property_readonly("ptr", [](const CudaStreamWrapper &s) {
            return reinterpret_cast<uintptr_t>(s.p.get());
        }, "Raw cudaStream_t pointer as integer")
        .def_property_readonly("is_default", [](const CudaStreamWrapper &s) {
            return s.p.get() == nullptr;
        }, "True if this wraps the default stream (ptr=0)")
        .def("__repr__", [](const CudaStreamWrapper &s) {
            std::stringstream ss;
            ss << "_CudaStreamWrapperBase(ptr=0x" << std::hex 
               << reinterpret_cast<uintptr_t>(s.p.get()) << ")";
            return ss.str();
        })
    ;
    
    py::class_<_StreamHolderBase>(m, "_StreamHolderBase",
        "Internal: Test helper for stream reference counting.\n\n"
        "Creates a CUDA stream on construction, allows retrieving it via get_stream().\n"
        "Used in unit tests to verify that streams stay alive when Python holds\n"
        "a reference even after the C++ holder is destroyed.")
        .def(py::init<>(), "Create holder with a new CUDA stream")
        .def("get_stream", &_StreamHolderBase::get_stream,
            "Return the stream (as _CudaStreamWrapperBase)")
    ;
}
