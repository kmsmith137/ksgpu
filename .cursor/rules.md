# ksgpu - GPU C++/CUDA Core Utilities

## Project Overview

`ksgpu` is a foundational GPU C++/CUDA utilities library providing:
- `Array<T>` - N-dimensional arrays with strides (CPU/GPU)
- `Dtype` - Runtime datatype specification
- Memory management utilities
- CUDA wrappers and RAII helpers
- Assertion macros
- Pybind11 integration for Python

Used as a dependency by downstream projects like `pirate`.

## Build System

- **Compiler**: `nvcc -std=c++17 -m64 -O3`
- **Build**: Makefile-based with automatic dependency generation (`-MMD -MP`)
- **Outputs**:
  - `lib/libksgpu.so` - C++ shared library
  - `ksgpu/ksgpu_pybind11*.so` - Python extension
- **NVIDIA architectures**: compute capability 80/86/89

## File Structure

```
include/ksgpu/    # Header files (.hpp)
src_lib/          # Library implementation (.cu)
src_bin/          # Standalone test/timing binaries
src_pybind11/     # Python bindings
ksgpu/            # Python package
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

### CUDA Error Handling

Always wrap CUDA API calls:

```cpp
CUDA_CALL(cudaMalloc(&ptr, size));      // throws on error
CUDA_PEEK("kernel_name");                // check after kernel launch
CUDA_CALL_ABORT(cudaSetDevice(dev));    // for contexts where exceptions aren't allowed
```

### Memory Allocation Flags (mem_utils.hpp)

Location flags (exactly one required):
- `af_gpu` - GPU memory
- `af_uhost` - unregistered host memory
- `af_rhost` - registered/page-locked host memory
- `af_unified` - unified memory

Initialization flags:
- `af_zero` - zero memory after allocation
- `af_random` - randomize memory after allocation

Debug flags:
- `af_guard` - guard regions for overflow detection
- `af_verbose` - print alloc/free messages

### Array Construction

```cpp
// Compile-time type
Array<float> arr({m, n}, af_gpu);                    // shape as initializer_list
Array<float> arr(shape, strides, af_rhost);          // explicit strides

// Runtime type
Array<void> arr(dtype, {m, n}, af_gpu);

// Default constructor (must initialize manually, then check_invariants())
Array<float> arr;
```

### Include Paths

Use relative includes from source files:

```cpp
#include "../include/ksgpu/Array.hpp"
#include "../include/ksgpu/cuda_utils.hpp"
```

## Key Classes

### Array<T>

N-dimensional array with strides. Key members:
- `data`, `ndim`, `shape[]`, `size`, `strides[]`, `dtype`, `aflags`
- `on_gpu()`, `on_host()` - check location
- `fill(src)` - copy data (handles different strides)
- `clone(aflags)` - copy with optional location change
- `to_gpu()`, `to_host()` - location conversion
- `slice(axis, ix)`, `slice(axis, start, stop)` - slicing
- `reshape(shape)`, `transpose(perm)` - axis manipulation
- `cast<U>()`, `convert<U>()` - type conversion
- `at(ix)` - range-checked accessor (debug only)

### Dtype

Runtime datatype with flags + nbits:

```cpp
Dtype dt(df_float, 32);                    // float32
Dtype dt(df_int, 64);                      // int64
Dtype dt(df_complex | df_float, 64);       // complex64 (float32+32)
Dtype::native<float>();                    // from C++ type
```

### RAII Wrappers

```cpp
CudaStreamWrapper stream;           // auto-destroyed stream
CudaEventWrapper event;             // auto-destroyed event
CudaTimer timer;                    // timer.stop() returns elapsed seconds
CudaSetDevice setter(device);       // restores original device in destructor
```

### Thread/Stream Pools

For timing with load-balanced callbacks:

```cpp
CudaStreamPool pool(callback, max_callbacks, nstreams, "name");
pool.monitor_throughput("GB/s", gbytes_per_callback);
pool.run();
```

## Device Code Patterns

```cpp
static constexpr uint FULL_MASK = 0xffffffffU;  // for warp shuffles

__device__ __forceinline__ void helper() { ... }

#pragma unroll
for (int i = 0; i < N; i++) { ... }

// Workaround for zero-sized arrays in device code
ksgpu::device_array<float, N> arr;  // works for N >= 0
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

## String Utilities

```cpp
tuple_str({1, 2, 3});           // "(1,2,3)"
type_name<float>();             // "float"
from_str<int>("42");            // 42
nbytes_to_str(1048576);         // "1.0 MB"
```

## What NOT to Do

- Don't use `size_t` or `int` for sizes - use `long`
- Don't use raw `assert()` - use `xassert()` macros
- Don't use raw CUDA API calls - use `CUDA_CALL()` wrapper
- Don't forget `CUDA_PEEK()` after kernel launches
- Don't use `cudaMemcpy` when `Array::fill()` handles strides correctly
- Don't allocate without proper `af_*` flags

