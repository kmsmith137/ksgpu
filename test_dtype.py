#!/usr/bin/env python3
"""
Unit tests for ksgpu.Dtype class and dtype conversion functionality.

Tests:
  - Dtype construction (empty, flags+nbits, from string, from numpy/cupy)
  - Dtype class attributes (INT, UINT, FLOAT, COMPLEX constants)
  - Dtype properties (is_valid, is_empty, precision, real, complex)
  - Dtype operators (equality, string representation)
  - Dtype conversion from various numpy/cupy dtypes
  - Error handling for invalid dtypes

Run with: python test_dtype.py
"""

import numpy as np
import sys

import ksgpu
from ksgpu import Dtype


# =============================================================================
# Test utilities
# =============================================================================

class TestFailure(Exception):
    """Raised when a test fails."""
    pass


def assert_equal(a, b, msg=""):
    """Assert two values are equal."""
    if a != b:
        raise TestFailure(f"Assertion failed: {a} != {b}" + (f" ({msg})" if msg else ""))


def assert_true(cond, msg=""):
    """Assert a condition is true."""
    if not cond:
        raise TestFailure(f"Assertion failed" + (f": {msg}" if msg else ""))


def assert_false(cond, msg=""):
    """Assert a condition is false."""
    if cond:
        raise TestFailure(f"Assertion failed (expected False)" + (f": {msg}" if msg else ""))


def assert_in(substring, string, msg=""):
    """Assert substring is in string."""
    if substring not in string:
        raise TestFailure(f"'{substring}' not in '{string}'" + (f" ({msg})" if msg else ""))


def assert_raises(exception_type, func, msg=""):
    """Assert that calling func raises the specified exception."""
    try:
        func()
        raise TestFailure(f"Expected {exception_type.__name__} but no exception raised" + (f" ({msg})" if msg else ""))
    except exception_type:
        pass  # Expected
    except Exception as e:
        raise TestFailure(f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}" + (f" ({msg})" if msg else ""))


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
# Basic construction tests
# =============================================================================

def test_dtype_empty_construction():
    """Test empty Dtype construction."""
    dt = Dtype()
    assert_true(dt.is_empty, "empty dtype should have is_empty=True")
    assert_false(dt.is_valid, "empty dtype should have is_valid=False")


def test_dtype_flags_nbits_construction():
    """Test Dtype construction with flags and nbits."""
    dt = Dtype(Dtype.FLOAT, 32)
    assert_equal(dt.flags, Dtype.FLOAT, "flags mismatch")
    assert_equal(dt.nbits, 32, "nbits mismatch")
    assert_true(dt.is_valid, "dtype should be valid")
    assert_false(dt.is_empty, "dtype should not be empty")


def test_dtype_copy_constructor():
    """Test Dtype copy constructor."""
    dt1 = Dtype(Dtype.FLOAT, 32)
    dt2 = Dtype(dt1)
    
    # Should be equal
    assert_equal(dt2.flags, dt1.flags, "copied flags should match")
    assert_equal(dt2.nbits, dt1.nbits, "copied nbits should match")
    assert_true(dt1 == dt2, "copied dtype should equal original")
    
    # Modifying one should not affect the other
    dt2.nbits = 64
    assert_equal(dt1.nbits, 32, "original should be unchanged")
    assert_equal(dt2.nbits, 64, "copy should be modified")


def test_dtype_class_attributes():
    """Test that df_* constants are class attributes."""
    assert_true(hasattr(Dtype, 'INT'), "Dtype.INT should exist")
    assert_true(hasattr(Dtype, 'UINT'), "Dtype.UINT should exist")
    assert_true(hasattr(Dtype, 'FLOAT'), "Dtype.FLOAT should exist")
    assert_true(hasattr(Dtype, 'COMPLEX'), "Dtype.COMPLEX should exist")
    assert_true(isinstance(Dtype.INT, int), "Dtype.INT should be int")
    assert_true(isinstance(Dtype.UINT, int), "Dtype.UINT should be int")
    assert_true(isinstance(Dtype.FLOAT, int), "Dtype.FLOAT should be int")
    assert_true(isinstance(Dtype.COMPLEX, int), "Dtype.COMPLEX should be int")


# =============================================================================
# String parsing tests
# =============================================================================

def test_dtype_from_string():
    """Test Dtype construction from string."""
    dt = Dtype("float32")
    assert_equal(dt.flags, Dtype.FLOAT, "float32 should have FLOAT flag")
    assert_equal(dt.nbits, 32, "float32 should have 32 bits")
    assert_true(dt.is_valid, "dtype should be valid")
    
    dt = Dtype("int64")
    assert_equal(dt.flags, Dtype.INT, "int64 should have INT flag")
    assert_equal(dt.nbits, 64, "int64 should have 64 bits")
    
    dt = Dtype("uint16")
    assert_equal(dt.flags, Dtype.UINT, "uint16 should have UINT flag")
    assert_equal(dt.nbits, 16, "uint16 should have 16 bits")


def test_dtype_from_str_static():
    """Test Dtype.from_str() static method."""
    dt = Dtype.from_str("float32")
    assert_equal(dt.flags, Dtype.FLOAT, "flags mismatch")
    assert_equal(dt.nbits, 32, "nbits mismatch")
    
    dt = Dtype.from_str("int64")
    assert_equal(dt.flags, Dtype.INT, "flags mismatch")
    assert_equal(dt.nbits, 64, "nbits mismatch")
    
    # Test with throw_exception_on_failure=False
    dt = Dtype.from_str("invalid_dtype_xyz", throw_exception_on_failure=False)
    assert_true(dt.is_empty, "invalid dtype should return empty")


def test_dtype_from_str_static_exception():
    """Test Dtype.from_str() with invalid input and exceptions enabled."""
    assert_raises(Exception, lambda: Dtype.from_str("invalid_dtype_xyz", throw_exception_on_failure=True))


# =============================================================================
# Numpy dtype tests
# =============================================================================

def test_dtype_from_numpy_float():
    """Test Dtype construction from numpy float dtypes."""
    dt = Dtype(np.float32)
    assert_equal(dt.flags, Dtype.FLOAT, "np.float32 flags mismatch")
    assert_equal(dt.nbits, 32, "np.float32 nbits mismatch")
    
    dt = Dtype(np.float64)
    assert_equal(dt.flags, Dtype.FLOAT, "np.float64 flags mismatch")
    assert_equal(dt.nbits, 64, "np.float64 nbits mismatch")
    
    dt = Dtype(np.float16)
    assert_equal(dt.flags, Dtype.FLOAT, "np.float16 flags mismatch")
    assert_equal(dt.nbits, 16, "np.float16 nbits mismatch")


def test_dtype_from_numpy_int():
    """Test Dtype construction from numpy int dtypes."""
    dt = Dtype(np.int8)
    assert_equal(dt.flags, Dtype.INT, "np.int8 flags mismatch")
    assert_equal(dt.nbits, 8, "np.int8 nbits mismatch")
    
    dt = Dtype(np.int16)
    assert_equal(dt.flags, Dtype.INT, "np.int16 flags mismatch")
    assert_equal(dt.nbits, 16, "np.int16 nbits mismatch")
    
    dt = Dtype(np.int32)
    assert_equal(dt.flags, Dtype.INT, "np.int32 flags mismatch")
    assert_equal(dt.nbits, 32, "np.int32 nbits mismatch")
    
    dt = Dtype(np.int64)
    assert_equal(dt.flags, Dtype.INT, "np.int64 flags mismatch")
    assert_equal(dt.nbits, 64, "np.int64 nbits mismatch")


def test_dtype_from_numpy_uint():
    """Test Dtype construction from numpy uint dtypes."""
    dt = Dtype(np.uint8)
    assert_equal(dt.flags, Dtype.UINT, "np.uint8 flags mismatch")
    assert_equal(dt.nbits, 8, "np.uint8 nbits mismatch")
    
    dt = Dtype(np.uint16)
    assert_equal(dt.flags, Dtype.UINT, "np.uint16 flags mismatch")
    assert_equal(dt.nbits, 16, "np.uint16 nbits mismatch")
    
    dt = Dtype(np.uint32)
    assert_equal(dt.flags, Dtype.UINT, "np.uint32 flags mismatch")
    assert_equal(dt.nbits, 32, "np.uint32 nbits mismatch")
    
    dt = Dtype(np.uint64)
    assert_equal(dt.flags, Dtype.UINT, "np.uint64 flags mismatch")
    assert_equal(dt.nbits, 64, "np.uint64 nbits mismatch")


def test_dtype_from_numpy_complex():
    """Test Dtype construction from numpy complex dtypes."""
    dt = Dtype(np.complex64)
    assert_equal(dt.flags, Dtype.COMPLEX | Dtype.FLOAT, "np.complex64 flags mismatch")
    assert_equal(dt.nbits, 64, "np.complex64 nbits mismatch")
    
    dt = Dtype(np.complex128)
    assert_equal(dt.flags, Dtype.COMPLEX | Dtype.FLOAT, "np.complex128 flags mismatch")
    assert_equal(dt.nbits, 128, "np.complex128 nbits mismatch")


def test_dtype_numpy_array_dtype():
    """Test Dtype construction from numpy array's dtype."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dt = Dtype(arr.dtype)
    assert_equal(dt.flags, Dtype.FLOAT, "flags mismatch")
    assert_equal(dt.nbits, 32, "nbits mismatch")


# =============================================================================
# Dtype properties tests
# =============================================================================

def test_dtype_properties():
    """Test Dtype properties."""
    dt = Dtype(Dtype.COMPLEX | Dtype.FLOAT, 64)
    assert_true(dt.is_valid, "dtype should be valid")
    assert_false(dt.is_empty, "dtype should not be empty")
    
    # Test real property
    real_dt = dt.real
    assert_equal(real_dt.flags, Dtype.FLOAT, "real dtype flags mismatch")
    assert_equal(real_dt.nbits, 32, "real dtype nbits mismatch")
    
    # Test complex property
    dt2 = Dtype(Dtype.FLOAT, 32)
    complex_dt = dt2.complex
    assert_equal(complex_dt.flags, Dtype.COMPLEX | Dtype.FLOAT, "complex dtype flags mismatch")
    assert_equal(complex_dt.nbits, 64, "complex dtype nbits mismatch")


def test_dtype_precision():
    """Test Dtype precision property."""
    # Integer types should have precision 0
    dt = Dtype(Dtype.INT, 32)
    assert_equal(dt.precision, 0.0, "int precision should be 0")
    
    dt = Dtype(Dtype.UINT, 64)
    assert_equal(dt.precision, 0.0, "uint precision should be 0")
    
    # Float types should have non-zero precision
    dt = Dtype(Dtype.FLOAT, 64)
    assert_true(dt.precision > 0.0, "float64 precision should be > 0")
    assert_true(dt.precision < 1e-10, "float64 precision should be < 1e-10")
    
    dt = Dtype(Dtype.FLOAT, 32)
    assert_true(dt.precision > 0.0, "float32 precision should be > 0")
    assert_true(dt.precision < 1e-5, "float32 precision should be < 1e-5")


# =============================================================================
# Operator tests
# =============================================================================

def test_dtype_equality():
    """Test Dtype equality operators."""
    dt1 = Dtype(Dtype.FLOAT, 32)
    dt2 = Dtype(np.float32)
    dt3 = Dtype("float32")
    
    assert_true(dt1 == dt2, "dtypes should be equal")
    assert_true(dt2 == dt3, "dtypes should be equal")
    assert_true(dt1 == dt3, "dtypes should be equal")
    
    dt4 = Dtype(Dtype.FLOAT, 64)
    assert_true(dt1 != dt4, "dtypes should not be equal")


def test_dtype_repr():
    """Test Dtype string representation."""
    dt = Dtype(Dtype.FLOAT, 32)
    repr_str = repr(dt)
    assert_in("float", repr_str.lower(), "repr should contain 'float'")
    assert_in("32", repr_str, "repr should contain '32'")
    
    dt = Dtype(Dtype.INT, 64)
    repr_str = repr(dt)
    assert_in("int", repr_str.lower(), "repr should contain 'int'")
    assert_in("64", repr_str, "repr should contain '64'")


# =============================================================================
# Error handling tests
# =============================================================================

def test_dtype_invalid():
    """Test invalid Dtype handling."""
    # Invalid: nbits=0
    assert_raises(Exception, lambda: Dtype(Dtype.FLOAT, 0))
    
    # Invalid: odd nbits for complex
    assert_raises(Exception, lambda: Dtype(Dtype.COMPLEX | Dtype.FLOAT, 33))


def test_dtype_unsupported_numpy():
    """Test that unsupported numpy dtypes raise proper errors."""
    # Unicode string dtype
    assert_raises(Exception, lambda: Dtype(np.dtype('U10')))
    
    # Object dtype
    assert_raises(Exception, lambda: Dtype(np.dtype('O')))


# =============================================================================
# Advanced tests
# =============================================================================

def test_dtype_conversion_consistency():
    """Test that multiple paths to the same dtype produce identical results."""
    # All of these should produce the same dtype
    dt1 = Dtype(Dtype.FLOAT, 32)
    dt2 = Dtype(np.float32)
    dt3 = Dtype("float32")
    dt4 = Dtype.from_str("float32")
    
    assert_true(dt1 == dt2 == dt3 == dt4, "all dtypes should be equal")
    assert_equal(dt1.flags, dt2.flags, "flags should match")
    assert_equal(dt2.flags, dt3.flags, "flags should match")
    assert_equal(dt3.flags, dt4.flags, "flags should match")
    assert_equal(dt1.nbits, dt2.nbits, "nbits should match")
    assert_equal(dt2.nbits, dt3.nbits, "nbits should match")
    assert_equal(dt3.nbits, dt4.nbits, "nbits should match")


def test_dtype_complex_int_uint():
    """Test complex dtypes with int and uint (though uncommon)."""
    # Complex int is a valid ksgpu dtype concept
    dt = Dtype(Dtype.COMPLEX | Dtype.INT, 64)
    assert_true(dt.is_valid, "complex int should be valid")
    assert_equal(dt.flags, Dtype.COMPLEX | Dtype.INT, "flags mismatch")
    assert_equal(dt.nbits, 64, "nbits mismatch")
    
    # Complex uint is also valid
    dt = Dtype(Dtype.COMPLEX | Dtype.UINT, 32)
    assert_true(dt.is_valid, "complex uint should be valid")
    assert_equal(dt.flags, Dtype.COMPLEX | Dtype.UINT, "flags mismatch")
    assert_equal(dt.nbits, 32, "nbits mismatch")


def test_dtype_readwrite_fields():
    """Test that flags and nbits fields are read-write."""
    dt = Dtype()
    assert_true(dt.is_empty, "dtype should start empty")
    
    # Manually set fields (note: this bypasses validation)
    dt.flags = Dtype.FLOAT
    dt.nbits = 32
    
    assert_equal(dt.flags, Dtype.FLOAT, "flags should be settable")
    assert_equal(dt.nbits, 32, "nbits should be settable")
    assert_true(dt.is_valid, "dtype should be valid after manual setting")


# =============================================================================
# Cupy tests (optional, skipped if cupy not available)
# =============================================================================

def test_dtype_from_cupy():
    """Test Dtype construction from cupy dtypes."""
    try:
        import cupy as cp
    except ImportError:
        raise TestFailure("SKIP: cupy not available")
    
    dt = Dtype(cp.float32)
    assert_equal(dt.flags, Dtype.FLOAT, "cp.float32 flags mismatch")
    assert_equal(dt.nbits, 32, "cp.float32 nbits mismatch")
    
    dt = Dtype(cp.int64)
    assert_equal(dt.flags, Dtype.INT, "cp.int64 flags mismatch")
    assert_equal(dt.nbits, 64, "cp.int64 nbits mismatch")
    
    dt = Dtype(cp.uint32)
    assert_equal(dt.flags, Dtype.UINT, "cp.uint32 flags mismatch")
    assert_equal(dt.nbits, 32, "cp.uint32 nbits mismatch")
    
    dt = Dtype(cp.complex128)
    assert_equal(dt.flags, Dtype.COMPLEX | Dtype.FLOAT, "cp.complex128 flags mismatch")
    assert_equal(dt.nbits, 128, "cp.complex128 nbits mismatch")


# =============================================================================
# Test runner
# =============================================================================

def run_basic_tests():
    """Run basic dtype tests."""
    print("\nBasic Construction Tests:")
    tests = [
        test_dtype_empty_construction,
        test_dtype_flags_nbits_construction,
        test_dtype_copy_constructor,
        test_dtype_class_attributes,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_string_tests():
    """Run string parsing tests."""
    print("\nString Parsing Tests:")
    tests = [
        test_dtype_from_string,
        test_dtype_from_str_static,
        test_dtype_from_str_static_exception,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_numpy_tests():
    """Run numpy dtype conversion tests."""
    print("\nNumpy Dtype Tests:")
    tests = [
        test_dtype_from_numpy_float,
        test_dtype_from_numpy_int,
        test_dtype_from_numpy_uint,
        test_dtype_from_numpy_complex,
        test_dtype_numpy_array_dtype,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_property_tests():
    """Run property tests."""
    print("\nProperty Tests:")
    tests = [
        test_dtype_properties,
        test_dtype_precision,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_operator_tests():
    """Run operator tests."""
    print("\nOperator Tests:")
    tests = [
        test_dtype_equality,
        test_dtype_repr,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_error_tests():
    """Run error handling tests."""
    print("\nError Handling Tests:")
    tests = [
        test_dtype_invalid,
        test_dtype_unsupported_numpy,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_advanced_tests():
    """Run advanced tests."""
    print("\nAdvanced Tests:")
    tests = [
        test_dtype_conversion_consistency,
        test_dtype_complex_int_uint,
        test_dtype_readwrite_fields,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_cupy_tests():
    """Run cupy tests (optional)."""
    print("\nCupy Tests:")
    
    # Check if cupy is available
    try:
        import cupy as cp
        # Try a simple operation to make sure cupy works
        _ = cp.array([1, 2, 3])
    except Exception as e:
        print(f"  SKIP: cupy not working ({e})")
        return 0, 0
    
    tests = [
        test_dtype_from_cupy,
    ]
    
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def main():
    """Run all tests."""
    print("=" * 60)
    print("ksgpu Dtype Unit Tests")
    print("=" * 60)
    
    results = []
    results.append(run_basic_tests())
    results.append(run_string_tests())
    results.append(run_numpy_tests())
    results.append(run_property_tests())
    results.append(run_operator_tests())
    results.append(run_error_tests())
    results.append(run_advanced_tests())
    results.append(run_cupy_tests())
    
    total_passed = sum(p for p, _ in results)
    total_tests = sum(t for _, t in results)
    
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
