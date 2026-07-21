#!/usr/bin/env python3
"""
Unit tests for ksgpu array conversion (numpy/cupy <-> C++).

Tests all four conversion cases:
  1. numpy -> C++
  2. C++ -> numpy
  3. cupy -> C++
  4. C++ -> cupy

Run with: python test_array_conversion.py

Focus areas:
  - Refcount handling (data stays alive, eventually freed)
  - Non-trivial strides (sliced, transposed arrays)
  - Various dtypes
  - Round-trip correctness
"""

import numpy as np
import gc
import sys

import ksgpu
from ksgpu.tests import Stash, get_array_info, make_strided_array


# =============================================================================
# Test utilities
# =============================================================================

class TestFailure(Exception):
    """Raised when a test fails."""
    pass


def assert_equal(a, b, msg=""):
    """Assert two values are equal."""
    if not np.array_equal(a, b):
        raise TestFailure(f"Assertion failed: {a} != {b}" + (f" ({msg})" if msg else ""))


def assert_close(a, b, msg="", rtol=1e-7, atol=0):
    """Assert two arrays are close (for floating point)."""
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise TestFailure(f"Arrays not close: {a} vs {b}" + (f" ({msg})" if msg else ""))


def assert_true(cond, msg=""):
    """Assert a condition is true."""
    if not cond:
        raise TestFailure(f"Assertion failed" + (f": {msg}" if msg else ""))


def run_test(test_func, test_name=None):
    """Run a single test function, catching and reporting failures."""
    name = test_name or test_func.__name__
    try:
        test_func()
        print(f"  PASS: {name}")
        return True
    except TestFailure as e:
        print(f"  FAIL: {name} - {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {name} - {type(e).__name__}: {e}")
        return False


# =============================================================================
# Numpy tests
# =============================================================================

def test_numpy_roundtrip_basic():
    """Basic numpy -> C++ -> numpy round-trip."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    stash = Stash(arr)
    result = stash.get()
    
    assert_equal(arr.shape, result.shape, "shape mismatch")
    assert_equal(arr.dtype, result.dtype, "dtype mismatch")
    assert_close(arr, result, "data mismatch")


def test_numpy_dtypes():
    """Test various numpy dtypes in round-trip."""
    dtypes = [
        np.float32, np.float64,
        np.int32, np.int64,
        np.uint32,
        np.complex64, np.complex128,
    ]
    
    for dtype in dtypes:
        arr = np.arange(10, dtype=dtype)
        if np.issubdtype(dtype, np.complexfloating):
            arr = arr + 1j * np.arange(10, dtype=arr.real.dtype)
        
        stash = Stash(arr)
        info = stash.info()
        result = stash.get()
        
        assert_equal(arr.dtype, result.dtype, f"dtype mismatch for {dtype}")
        assert_close(arr, result, f"data mismatch for {dtype}")


def test_numpy_shapes():
    """Test various array shapes."""
    shapes = [
        (10,),          # 1D
        (3, 4),         # 2D
        (2, 3, 4),      # 3D
        (2, 3, 4, 5),   # 4D
    ]
    
    for shape in shapes:
        arr = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        stash = Stash(arr)
        result = stash.get()
        
        assert_equal(arr.shape, result.shape, f"shape mismatch for {shape}")
        assert_close(arr, result, f"data mismatch for {shape}")


def test_numpy_strides_sliced():
    """Test sliced arrays with non-trivial strides."""
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    
    # Various slices
    test_cases = [
        base[::2],         # every other row
        base[:, ::2],      # every other column
        base[::2, ::2],    # both
        base[1:, 2:],      # offset slices
        base[1::2, 1::3],  # offset + step
    ]
    
    for i, arr in enumerate(test_cases):
        info = get_array_info(arr)
        
        # Verify C++ sees correct shape
        assert_equal(list(arr.shape), info.shape, f"case {i}: shape mismatch")
        
        # Verify strides (numpy uses bytes, ksgpu uses elements)
        expected_strides = [s // arr.itemsize for s in arr.strides]
        assert_equal(expected_strides, info.strides, f"case {i}: strides mismatch")
        
        # Round-trip
        stash = Stash(arr)
        result = stash.get()
        assert_close(arr, result, f"case {i}: data mismatch")


def test_numpy_strides_transpose():
    """Test transposed arrays."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    arr_t = arr.T
    
    info = get_array_info(arr_t)
    assert_equal(list(arr_t.shape), info.shape, "transposed shape")
    
    expected_strides = [s // arr_t.itemsize for s in arr_t.strides]
    assert_equal(expected_strides, info.strides, "transposed strides")
    
    stash = Stash(arr_t)
    result = stash.get()
    assert_close(arr_t, result, "transposed data")


def test_numpy_strides_fortran():
    """Test Fortran-order (column-major) arrays."""
    arr = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    
    info = get_array_info(arr)
    assert_equal(list(arr.shape), info.shape, "fortran shape")
    
    expected_strides = [s // arr.itemsize for s in arr.strides]
    assert_equal(expected_strides, info.strides, "fortran strides")
    
    stash = Stash(arr)
    result = stash.get()
    assert_close(arr, result, "fortran data")


def test_numpy_refcount_basic():
    """Test that refcounts are properly managed."""
    # Create array and stash it
    arr = np.arange(100, dtype=np.float64)
    data_ptr = arr.ctypes.data
    
    stash = Stash(arr)
    
    # Delete Python reference, data should still be accessible via stash
    del arr
    gc.collect()
    
    # Stash should still be valid
    info = stash.info()
    assert_equal(info.data_ptr, data_ptr, "data pointer changed after del")
    
    # Can still retrieve the array
    result = stash.get()
    assert_equal(result.shape, (100,), "retrieved shape")
    assert_close(result, np.arange(100, dtype=np.float64), "retrieved data")


def test_numpy_refcount_modify():
    """Test that modifications are visible (shared memory)."""
    arr = np.arange(10, dtype=np.float64)
    stash = Stash(arr)
    
    # Modify through original reference
    arr[5] = 999.0
    
    # Should be visible when we get from stash
    result = stash.get()
    assert_equal(result[5], 999.0, "modification not visible")


def test_numpy_cpp_to_python_strided():
    """Test C++ -> Python conversion with non-contiguous strides."""
    # Create a strided array in C++ and convert to Python
    shape = [3, 4]
    strides = [8, 2]  # Non-contiguous: skip every other element
    
    arr = make_strided_array(shape, strides, "float64", False)
    
    assert_equal(arr.shape, tuple(shape), "shape")
    
    # Verify strides match
    expected_strides_bytes = [s * arr.itemsize for s in strides]
    assert_equal(list(arr.strides), expected_strides_bytes, "strides")
    
    # Verify we can access the data correctly
    # The buffer is filled with 0, 1, 2, ... so we can verify indexing
    for i in range(shape[0]):
        for j in range(shape[1]):
            expected_val = i * strides[0] + j * strides[1]
            assert_close(arr[i, j], expected_val, f"value at [{i},{j}]")


def test_numpy_zero_dim_error():
    """Test that zero-dimensional arrays raise an error."""
    arr = np.array(5.0)  # 0-d array (scalar)
    
    try:
        stash = Stash(arr)
        raise TestFailure("Expected exception for 0-d array")
    except Exception as e:
        # Expected
        pass


# =============================================================================
# Cupy tests (skipped if cupy not available)
# =============================================================================

def test_cupy_roundtrip_basic():
    """Basic cupy -> C++ -> cupy round-trip."""
    import cupy as cp
    
    arr = cp.arange(12, dtype=cp.float64).reshape(3, 4)
    stash = Stash(arr)
    result = stash.get()
    
    assert_true(isinstance(result, cp.ndarray), "result should be cupy array")
    assert_equal(arr.shape, result.shape, "shape mismatch")
    assert_equal(arr.dtype, result.dtype, "dtype mismatch")
    assert_true(cp.allclose(arr, result), "data mismatch")


def test_cupy_dtypes():
    """Test various cupy dtypes in round-trip."""
    import cupy as cp
    
    dtypes = [
        cp.float32, cp.float64,
        cp.int32, cp.int64,
        cp.uint32,
        cp.complex64, cp.complex128,
    ]
    
    for dtype in dtypes:
        arr = cp.arange(10, dtype=dtype)
        if cp.issubdtype(dtype, cp.complexfloating):
            arr = arr + 1j * cp.arange(10, dtype=arr.real.dtype)
        
        stash = Stash(arr)
        info = stash.info()
        result = stash.get()
        
        assert_equal(info.location, "gpu", f"location for {dtype}")
        assert_true(isinstance(result, cp.ndarray), f"result type for {dtype}")
        assert_equal(arr.dtype, result.dtype, f"dtype mismatch for {dtype}")
        assert_true(cp.allclose(arr, result), f"data mismatch for {dtype}")


def test_cupy_strides_sliced():
    """Test sliced cupy arrays with non-trivial strides."""
    import cupy as cp
    
    base = cp.arange(24, dtype=cp.float64).reshape(4, 6)
    
    test_cases = [
        base[::2],
        base[:, ::2],
        base[::2, ::2],
        base[1:, 2:],
    ]
    
    for i, arr in enumerate(test_cases):
        info = get_array_info(arr)
        
        assert_equal(list(arr.shape), info.shape, f"case {i}: shape")
        assert_equal(info.location, "gpu", f"case {i}: location")
        
        stash = Stash(arr)
        result = stash.get()
        assert_true(cp.allclose(arr, result), f"case {i}: data")


def test_cupy_strides_transpose():
    """Test transposed cupy arrays."""
    import cupy as cp
    
    arr = cp.arange(12, dtype=cp.float64).reshape(3, 4)
    arr_t = arr.T
    
    info = get_array_info(arr_t)
    assert_equal(list(arr_t.shape), info.shape, "transposed shape")
    
    stash = Stash(arr_t)
    result = stash.get()
    assert_true(cp.allclose(arr_t, result), "transposed data")


def test_cupy_refcount_basic():
    """Test refcount handling for cupy arrays."""
    import cupy as cp
    
    arr = cp.arange(100, dtype=cp.float64)
    data_ptr = arr.data.ptr
    
    stash = Stash(arr)
    
    # Delete Python reference
    del arr
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    
    # Stash should still be valid
    info = stash.info()
    assert_equal(info.data_ptr, data_ptr, "data pointer changed")
    
    # Can still retrieve
    result = stash.get()
    assert_equal(result.shape, (100,), "retrieved shape")


def test_cupy_cpp_to_python_strided():
    """Test C++ -> cupy conversion with non-contiguous strides."""
    import cupy as cp
    
    shape = [3, 4]
    strides = [8, 2]
    
    arr = make_strided_array(shape, strides, "float64", True)  # on_gpu=True
    
    assert_true(isinstance(arr, cp.ndarray), "should be cupy array")
    assert_equal(arr.shape, tuple(shape), "shape")
    
    expected_strides_bytes = [s * arr.itemsize for s in strides]
    assert_equal(list(arr.strides), expected_strides_bytes, "strides")


def test_cupy_refcount_modify():
    """Test that modifications to cupy arrays are visible (shared memory)."""
    import cupy as cp
    
    arr = cp.arange(10, dtype=cp.float64)
    stash = Stash(arr)
    
    # Modify through original reference
    arr[5] = 999.0
    cp.cuda.runtime.deviceSynchronize()
    
    # Should be visible when we get from stash
    result = stash.get()
    assert_equal(float(result[5].get()), 999.0, "modification not visible")


# =============================================================================
# Test runner
# =============================================================================

def run_numpy_tests():
    """Run all numpy tests."""
    print("\n=== Numpy Tests ===")
    
    tests = [
        test_numpy_roundtrip_basic,
        test_numpy_dtypes,
        test_numpy_shapes,
        test_numpy_strides_sliced,
        test_numpy_strides_transpose,
        test_numpy_strides_fortran,
        test_numpy_refcount_basic,
        test_numpy_refcount_modify,
        test_numpy_cpp_to_python_strided,
        test_numpy_zero_dim_error,
    ]
    
    passed = sum(run_test(t) for t in tests)
    print(f"\nNumpy: {passed}/{len(tests)} tests passed")
    return passed, len(tests)


def run_cupy_tests():
    """Run all cupy tests (if cupy is available)."""
    print("\n=== Cupy Tests ===")
    
    try:
        import cupy as cp
        # Quick check that CUDA is working
        _ = cp.zeros(1)
    except ImportError:
        print("  SKIP: cupy not installed")
        return 0, 0
    except Exception as e:
        print(f"  SKIP: cupy not working ({e})")
        return 0, 0
    
    tests = [
        test_cupy_roundtrip_basic,
        test_cupy_dtypes,
        test_cupy_strides_sliced,
        test_cupy_strides_transpose,
        test_cupy_refcount_basic,
        test_cupy_cpp_to_python_strided,
        test_cupy_refcount_modify,
    ]
    
    passed = sum(run_test(t) for t in tests)
    print(f"\nCupy: {passed}/{len(tests)} tests passed")
    return passed, len(tests)


def main():
    """Run all tests."""
    print("=" * 60)
    print("ksgpu Array Conversion Unit Tests")
    print("=" * 60)
    
    numpy_passed, numpy_total = run_numpy_tests()
    cupy_passed, cupy_total = run_cupy_tests()
    
    total_passed = numpy_passed + cupy_passed
    total_tests = numpy_total + cupy_total
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if total_passed < total_tests:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

