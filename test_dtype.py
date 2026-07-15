#!/usr/bin/env python3
"""
Unit tests for the numpy.dtype <-> ksgpu::Dtype boundary conversion.

There is no python-visible Dtype class: ksgpu::Dtype arguments and return
values convert automatically at the C++/python boundary, via the
type_caster<ksgpu::Dtype> in include/ksgpu/pybind11.hpp. Python code only
ever sees numpy.dtype objects. These tests exercise both directions through
ksgpu.tests.dtype_roundtrip() (python -> C++ -> python) and
ksgpu.tests.make_strided_array() (dtype argument -> returned array).

Tests:
  - Roundtrip of all numpy-representable dtypes (dtype objects, scalar types, strings)
  - ksgpu-specific dtype string names (e.g. "complex32+32")
  - None <-> empty Dtype convention
  - Rejection of python ints (numpy "type number" footgun)
  - Rejection of unsupported numpy kinds (e.g. bytes, bool)
  - Errors when returning a Dtype with no numpy equivalent (e.g. "int7")
  - Conversion from cupy dtypes (optional, if cupy available)

Run with: python test_dtype.py
"""

import numpy as np
import sys

import ksgpu
from ksgpu.tests import dtype_roundtrip, make_strided_array


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


def assert_raises_with_message(substring, func, msg=""):
    """Assert that calling func raises, with 'substring' in the exception text."""
    try:
        func()
    except Exception as e:
        if substring not in str(e):
            raise TestFailure(f"Exception raised, but '{substring}' not in message '{e}'"
                              + (f" ({msg})" if msg else ""))
        return
    raise TestFailure(f"Expected an exception but none was raised" + (f" ({msg})" if msg else ""))


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


# All dtypes representable on both sides of the boundary.
ROUNDTRIP_NAMES = [
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'float16', 'float32', 'float64',
    'complex64', 'complex128',
]


# =============================================================================
# Roundtrip tests (python -> ksgpu::Dtype -> python)
# =============================================================================

def test_roundtrip_dtype_objects():
    """Roundtrip numpy.dtype objects; result is an equal numpy.dtype."""
    for name in ROUNDTRIP_NAMES:
        dt_in = np.dtype(name)
        dt_out = dtype_roundtrip(dt_in)
        assert_true(isinstance(dt_out, np.dtype), f"{name}: expected numpy.dtype, got {type(dt_out)}")
        assert_equal(dt_out, dt_in, f"{name}: roundtrip mismatch")


def test_roundtrip_scalar_types():
    """Roundtrip numpy scalar types (np.float32 etc.)."""
    for name in ROUNDTRIP_NAMES:
        scalar_type = getattr(np, name)
        dt_out = dtype_roundtrip(scalar_type)
        assert_equal(dt_out, np.dtype(name), f"np.{name}: roundtrip mismatch")


def test_roundtrip_numpy_strings():
    """Roundtrip numpy dtype names passed as strings."""
    for name in ROUNDTRIP_NAMES:
        dt_out = dtype_roundtrip(name)
        assert_equal(dt_out, np.dtype(name), f"'{name}': roundtrip mismatch")


def test_roundtrip_array_dtype():
    """Roundtrip the .dtype attribute of a numpy array."""
    arr = np.zeros(3, dtype=np.float32)
    assert_equal(dtype_roundtrip(arr.dtype), np.dtype('float32'))


def test_roundtrip_python_builtins():
    """Python builtin types convert via numpy's rules (float -> float64, etc.)."""
    assert_equal(dtype_roundtrip(float), np.dtype(np.float64))
    assert_equal(dtype_roundtrip(int), np.dtype(int))       # int64 on linux
    assert_equal(dtype_roundtrip(complex), np.dtype(np.complex128))


def test_ksgpu_string_names():
    """ksgpu-specific dtype names (Dtype::from_str syntax) are also accepted."""
    # ksgpu writes complex dtypes as "complex<real bits>+<imag bits>".
    assert_equal(dtype_roundtrip("complex32+32"), np.dtype('complex64'))
    assert_equal(dtype_roundtrip("complex64+64"), np.dtype('complex128'))


# =============================================================================
# None <-> empty Dtype
# =============================================================================

def test_none_roundtrip():
    """None converts to an empty ksgpu::Dtype, which converts back to None."""
    # This also verifies that the caster does NOT inherit numpy's behavior
    # np.dtype(None) == float64.
    assert_true(dtype_roundtrip(None) is None, "expected None -> None")


# =============================================================================
# Error handling
# =============================================================================

def test_int_rejected():
    """Python ints are rejected (numpy would treat them as "type numbers")."""
    # np.dtype(1) is int8 -- make sure we don't inherit that footgun.
    assert_raises_with_message("couldn't convert python int", lambda: dtype_roundtrip(1))
    assert_raises_with_message("couldn't convert python int", lambda: dtype_roundtrip(0))


def test_unsupported_numpy_kinds():
    """Numpy dtypes with no ksgpu equivalent (kinds other than i/u/f/c) are rejected."""
    for bad in [np.dtype('S10'), np.dtype('U5'), np.bool_, np.dtype('datetime64[s]')]:
        assert_raises_with_message("not supported by ksgpu", lambda: dtype_roundtrip(bad))


def test_garbage_rejected():
    """Objects that numpy can't parse as a dtype are rejected."""
    assert_raises_with_message("couldn't convert", lambda: dtype_roundtrip("not_a_dtype_xyz"))
    assert_raises_with_message("couldn't convert", lambda: dtype_roundtrip([1, 2, 3]))


def test_unrepresentable_return():
    """A valid ksgpu::Dtype with no numpy equivalent raises when returned to python."""
    # These names are accepted by Dtype::from_str() (so the python -> C++
    # direction succeeds), but the C++ -> python direction must raise.
    for name in ["int7", "float8", "complex16+16", "complex_int4+4"]:
        assert_raises_with_message("no numpy equivalent", lambda: dtype_roundtrip(name))


# =============================================================================
# Dtype arguments of ordinary bound functions
# =============================================================================

def test_make_strided_array_dtype_arg():
    """make_strided_array()'s ksgpu::Dtype argument accepts numpy dtypes and strings."""
    for dt in [np.dtype('float32'), np.float64, 'int32', np.dtype('complex64')]:
        arr = make_strided_array([3, 4], [4, 1], dt, False)
        assert_equal(arr.dtype, np.dtype(dt), f"returned array has wrong dtype for {dt!r}")
        assert_equal(arr.shape, (3, 4))


# =============================================================================
# Cupy tests (optional)
# =============================================================================

def test_dtype_from_cupy():
    """Cupy dtypes and scalar types convert like their numpy counterparts."""
    import cupy as cp
    assert_equal(dtype_roundtrip(cp.float32), np.dtype('float32'))
    assert_equal(dtype_roundtrip(cp.int64), np.dtype('int64'))
    arr = cp.zeros(3, dtype=cp.float16)
    assert_equal(dtype_roundtrip(arr.dtype), np.dtype('float16'))


# =============================================================================
# Test runners
# =============================================================================

def run_roundtrip_tests():
    print("\nRoundtrip Tests:")
    tests = [
        test_roundtrip_dtype_objects,
        test_roundtrip_scalar_types,
        test_roundtrip_numpy_strings,
        test_roundtrip_array_dtype,
        test_roundtrip_python_builtins,
        test_ksgpu_string_names,
        test_none_roundtrip,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_error_tests():
    print("\nError Handling Tests:")
    tests = [
        test_int_rejected,
        test_unsupported_numpy_kinds,
        test_garbage_rejected,
        test_unrepresentable_return,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_argument_tests():
    print("\nDtype Argument Tests:")
    tests = [
        test_make_strided_array_dtype_arg,
    ]
    passed = sum(run_test(t) for t in tests)
    return passed, len(tests)


def run_cupy_tests():
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
    print("ksgpu Dtype Conversion Unit Tests")
    print("=" * 60)

    results = []
    results.append(run_roundtrip_tests())
    results.append(run_error_tests())
    results.append(run_argument_tests())
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
