# ksgpu - GPU C++/CUDA Core Utilities

This file was originally written for LLMs, but isn't a bad reference for human
collaborators. (Humans can ignore content toward the end of the file, starting
with "Code Conventions".)


## Project Overview

`ksgpu` is a foundational GPU C++/CUDA utilities library providing:
- `Array<T>` - N-dimensional arrays with strides (CPU/GPU).
- `Dtype` - Runtime datatype specification.
- Memory management utilities.
- CUDA wrappers and RAII helpers.
- Assertion macros (xassert).
- Pybind11 integration for Python.

Used as a dependency by downstream projects like `pirate` or `gpu_mm`.

## Build System

Uses `pipmake`, a tiny build system which is pip-compatible, but just forwards
pip commands to a Makefile (e.g. `pip install` forwards to `make wheel`).

The Makefile runs the script `makefile_helper.py`, which contains miscellaneous
logic that's more convenient to write in python than in Makefile language. (For
example, figuring out which directory the numpy `*.h` files are in.) The output
of this script is a file `makefile_helper.out`, containing variable declarations
in Makefile language.

You can freely mix invokations of `pip` and `make`, which is convenient but can
be confusing. See README.md for a recommendation.

## File Structure

```
include/ksgpu/    # Header files (.hpp)
src_lib/          # Library implementation (.cu)
src_bin/          # Standalone test/timing binaries
src_pybind11/     # Python bindings
ksgpu/            # Python package
```

## Overview

### Array<T>: N-dimensional array with strides, in either host or GPU memory.
- Key members: `data`, `ndim`, `shape[]`, `size`, `strides[]`, `dtype`, `aflags`.
- Low-level class with no protected members -- caller is allowed to manually
  manipulate members and bare pointers, when this makes sense.
- The type `T` can either be known at compile time (e.g. `Array<float>`), or
  dynamic at runtime (`Array<void>`) via the `dtype` member.
- The `aflags` member keeps track of whether the array resides on the host or GPU,
  and how the memory is freed (e.g. hugepages are freed with `munmap()` not `free()`).
- Memory ownership is refcounted with a member `shared_ptr<void> base`.
  This usually points to the same memory as the `data` pointer, but there are exceptions.
  For example, when an array is converted from python, the `base` pointer is a
  `(PyObject *)`, and the `shared_ptr` deleter calls `Py_DECREF()`. This allows
  refcounts to be shared between python and C++.
- Automatic conversion to/from python, whenever a C++ function with `Array` arguments
  is pybind11-wrapped. Arrays in host/GPU memory are converted to numpy/cupy arrays.
- `to_gpu()`, `to_host()` - one-liners to copy between host and GPU.
- `on_gpu()`, `on_host()` - check location.
- `fill(src)` - copy data (handles different strides).
- `clone(aflags)` - copy with optional location change.
- `slice(axis, ix)`, `slice(axis, start, stop)` - slicing.
- `reshape(shape)`, `transpose(perm)` - axis manipulation.
- `cast<U>()`, `convert<U>()` - type conversion.
- `at(ix)` - range-checked accessor (slow).

Examples:
```cpp
// Compile-time type. The af_* flags control how memory is allocated.
Array<float> arr({m,n}, af_gpu);             // shape as initializer_list, allocate on GPU.
Array<float> arr(shape, strides, af_rhost);  // explicit strides, allocate cuda-registered host memory.

// Runtime type.
Dtype dtype = ...;                        // See below.
Array<void> arr(dtype, {m,n}, af_uhost);  // allocate host memory which is not cuda-registered.

// Default constructor (must initialize manually, then check_invariants())
Array<float> arr;
```

### Memory Allocation Flags (mem_utils.hpp)

These flags `af_*` specify how memory should be allocated, and also arise as class members
structures (e.g. `int Array<T>::aflags`) to keep track of whether memory resides on the
host or GPU.

Location flags (exactly one required):
- `af_gpu` - GPU memory
- `af_uhost` - unregistered host memory
- `af_rhost` - registered/page-locked host memory
- `af_unified` - unified memory

Initialization flags:
- `af_zero` - zero memory after allocation
- `af_random` - randomize memory after allocation

Hugepages:
- `af_mmap_huge` - allocate on hugepages
- `af_mmap_try_huge` - try hugepages, but fall back to normal `malloc()` with warning

Debug flags:
- `af_guard` - guard regions for overflow detection
- `af_verbose` - print alloc/free messages

### Dtypes

Runtime datatype with flags (`df_*`) + nbits.
Used in `Array<void>` to keep track of array datatypes dynamically at runtime.

```cpp
Dtype dt(df_float, 32);                    // float32
Dtype dt(df_int, 64);                      // int64
Dtype dt(df_complex | df_float, 64);       // complex64 (float32+32)
Dtype dt = Dtype::native<float>();         // from C++ type
Dtype dt = Dtype::from_str("float32");     // from string
```

### Assertions (xassert.hpp)

Use `xassert` macros instead of `assert()` - they throw exceptions with useful messages:

```cpp
xassert(cond);                    // basic assertion
xassert_eq(lhs, rhs);             // shows values on failure
xassert_ne(lhs, rhs);
xassert_lt(lhs, rhs);
xassert_le(lhs, rhs);
xassert_gt(lhs, rhs);
xassert_ge(lhs, rhs);
xassert_divisible(lhs, rhs);      // check (lhs % rhs == 0)
xassert_shape_eq(arr, ({m,n}));   // array shape check (note parens + braces)
```

### CUDA utils and RAII wrappers

```cpp
CudaStreamWrapper stream;           // auto-destroyed stream
CudaEventWrapper event;             // auto-destroyed event
CudaSetDevice setter(device);       // restores original device in destructor
```

### KernelTimer: multi-stream cuda kernel timing.

```cpp
KernelTimer kt(niterations, nstreams);

while (kt.next()) {}
    // Note use of kt.stream, kt.istream
    mykernel <<< B, W, 0, kt.stream >>> (base_ptr + kt.istream * offset);
    CUDA_PEEK("mykernel");
    if (kt.warmed_up)
        cout << "average time/kernel = " << kt.dt << ", sec" << endl;
}
```

### String Utilities

```cpp
tuple_str({1, 2, 3});           // "(1,2,3)"
type_name<float>();             // "float"
from_str<int>("42");            // 42
nbytes_to_str(1048576);         // "1.0 MB"
```

## Code Conventions

### Namespace

All code lives in the `ksgpu` namespace with the auto-indent trick:

```cpp
namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif

// ... code here ...

} // namespace ksgpu
```

### Integer Types

Use `long` for sizes and indices (not `size_t` or `int`):

```cpp
long nelts = arr.size;
for (long i = 0; i < nelts; i++) { ... }
```

### CUDA Error Handling

Always wrap CUDA API calls:

```cpp
CUDA_CALL(cudaMalloc(&ptr, size));      // throws on error
CUDA_PEEK("kernel_name");               // check after kernel launch
CUDA_CALL_ABORT(cudaSetDevice(dev));    // for contexts where exceptions aren't allowed
```

### Include Paths

Use relative includes from source files:

```cpp
#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/cuda_utils.hpp"
```

## Device Code Patterns

```cpp
static constexpr uint FULL_MASK = ~0u;  // for warp shuffles

__device__ __forceinline__ void helper() { ... }

#pragma unroll
for (int i = 0; i < N; i++) { ... }
```

## Pybind11 Integration

Main module in `src_pybind11/ksgpu_pybind11.cu`:

```cpp
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_ksgpu
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// In PYBIND11_MODULE:
if (_import_array() < 0) { ... }
```

Python `__init__.py` uses "ctypes trick" to load libraries with `RTLD_GLOBAL`.

## What NOT to Do

- Don't use `size_t` or `int` for sizes - use `long`
- Don't use raw `assert()` - use `xassert()` macros
- Don't use raw CUDA API calls - use `CUDA_CALL()` wrapper
- Don't forget `CUDA_PEEK()` after kernel launches
- Don't use `cudaMemcpy` when `Array::fill()` handles strides correctly
- Don't allocate without proper `af_*` flags

