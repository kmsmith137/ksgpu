#!/usr/bin/env python3
"""
Unit tests for ksgpu CUDA stream wrapper with cupy interop.

Tests:
  1. Basic CudaStreamWrapper creation and properties
  2. Context manager protocol (inherited from cupy.cuda.ExternalStream)
  3. Reference counting (stream stays alive while Python holds reference)
  4. Integration with C++ objects that return streams

Run with: python test_stream_conversion.py
"""

import gc
import sys

import ksgpu
from ksgpu import CudaStreamWrapper, StreamHolder


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


def assert_isinstance(obj, cls, msg=""):
    """Assert object is instance of class."""
    if not isinstance(obj, cls):
        raise TestFailure(f"{obj} is not instance of {cls}" + (f" ({msg})" if msg else ""))


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
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Basic tests
# =============================================================================

def test_stream_wrapper_basic():
    """Test basic CudaStreamWrapper creation and properties."""
    # Default stream
    default_stream = CudaStreamWrapper.default_stream()
    assert_equal(default_stream.ptr, 0, "default stream ptr should be 0")
    assert_true(default_stream.is_default, "is_default should be True")
    
    # New stream
    stream = CudaStreamWrapper.create()
    assert_true(stream.ptr != 0, "new stream ptr should be non-zero")
    assert_true(not stream.is_default, "is_default should be False")


def test_stream_wrapper_priority():
    """Test stream creation with priority."""
    # Default priority
    stream_default = CudaStreamWrapper.create()
    assert_true(stream_default.ptr != 0)
    
    # High priority (lower number = higher priority)
    stream_high = CudaStreamWrapper.create(priority=-1)
    assert_true(stream_high.ptr != 0)
    
    # Different streams should have different pointers
    assert_true(stream_default.ptr != stream_high.ptr, 
                "different streams should have different pointers")


def test_stream_wrapper_is_cupy_stream():
    """Test that CudaStreamWrapper is a cupy stream subclass."""
    import cupy.cuda
    
    stream = CudaStreamWrapper.create()
    
    # Should be instance of ExternalStream
    assert_isinstance(stream, cupy.cuda.ExternalStream,
                     "should be instance of ExternalStream")
    
    # ExternalStream inherits from BaseStream, which some code checks for
    # (The exact hierarchy may vary by cupy version, but ExternalStream check is key)


def test_stream_wrapper_repr():
    """Test string representation."""
    stream = CudaStreamWrapper.create()
    repr_str = repr(stream)
    
    assert_true("CudaStreamWrapper" in repr_str, "repr should contain class name")
    assert_true("ptr=0x" in repr_str, "repr should contain ptr")
    
    default_stream = CudaStreamWrapper.default_stream()
    default_repr = repr(default_stream)
    assert_true("is_default=True" in default_repr, "default stream repr should show is_default=True")


# =============================================================================
# Context manager tests
# =============================================================================

def test_stream_wrapper_context_manager():
    """Test that CudaStreamWrapper works as context manager."""
    import cupy as cp
    
    stream = CudaStreamWrapper.create()
    
    # Context manager usage (inherited from ExternalStream)
    with stream:
        a = cp.arange(1000)
        b = a * 2
    
    # Verify computation worked
    assert_equal(int(b[500].get()), 1000, "computation result")


def test_stream_wrapper_context_manager_nested():
    """Test nested context managers with different streams."""
    import cupy as cp
    
    stream1 = CudaStreamWrapper.create()
    stream2 = CudaStreamWrapper.create()
    
    with stream1:
        a = cp.arange(100)
        with stream2:
            b = cp.arange(100) + 1
        c = a + 1
    
    # Both computations should complete correctly
    assert_equal(int(a[50].get()), 50)
    assert_equal(int(b[50].get()), 51)
    assert_equal(int(c[50].get()), 51)


def test_stream_from_cpp_object():
    """Test the target use case: getting stream from C++ object."""
    import cupy as cp
    
    holder = StreamHolder()
    
    # This is the desired syntax!
    with holder.get_stream() as s:
        a = cp.arange(100)
        b = a + 1
    
    assert_equal(int(b[50].get()), 51)


# =============================================================================
# Reference counting tests
# =============================================================================

def test_stream_wrapper_refcount():
    """Test that Python properly owns a reference to the stream."""
    import cupy as cp
    
    # Create holder, get stream, delete holder
    holder = StreamHolder()
    stream = holder.get_stream()
    original_ptr = stream.ptr
    
    # Delete the C++ holder
    del holder
    gc.collect()
    
    # Stream should still be valid (Python holds reference via _cpp_wrapper)
    assert_equal(stream.ptr, original_ptr, "ptr should be unchanged")
    
    # Use stream with context manager to verify it's actually valid
    with stream:
        arr = cp.zeros(10)
        cp.cuda.runtime.deviceSynchronize()
    
    # Now delete Python reference
    del stream
    gc.collect()
    # Stream is now destroyed (no crash = success)


def test_stream_wrapper_multiple_refs():
    """Test that multiple Python references work correctly."""
    stream1 = CudaStreamWrapper.create()
    stream2 = stream1  # Both point to same wrapper
    
    assert_equal(stream1.ptr, stream2.ptr, "same ptr")
    
    del stream1
    gc.collect()
    
    # stream2 should still be valid
    assert_true(stream2.ptr != 0, "stream2 should still be valid")


def test_stream_wrapper_multiple_get_stream():
    """Test calling get_stream() multiple times."""
    holder = StreamHolder()
    
    stream1 = holder.get_stream()
    stream2 = holder.get_stream()
    
    # Both should have the same underlying stream pointer
    assert_equal(stream1.ptr, stream2.ptr, "same underlying stream")
    
    # But they are different Python wrapper objects
    assert_true(stream1 is not stream2, "different wrapper objects")
    
    # Delete one, the other should still work
    ptr = stream1.ptr
    del stream1
    gc.collect()
    
    assert_equal(stream2.ptr, ptr, "stream2 still valid")


def test_stream_holder_lifecycle():
    """Test the full lifecycle of StreamHolder and streams."""
    import cupy as cp
    
    # Phase 1: Create holder and use stream
    holder = StreamHolder()
    stream = holder.get_stream()
    ptr = stream.ptr
    
    with stream:
        arr = cp.arange(10)
    
    # Phase 2: Delete holder, stream should still work
    del holder
    gc.collect()
    
    assert_equal(stream.ptr, ptr, "ptr unchanged after holder deletion")
    
    with stream:
        arr2 = cp.arange(10) * 2
    
    # Phase 3: Delete stream
    del stream
    gc.collect()
    # No crash = success


# =============================================================================
# Edge case tests
# =============================================================================

def test_default_stream_context_manager():
    """Test context manager with default stream."""
    import cupy as cp
    
    stream = CudaStreamWrapper.default_stream()
    
    with stream:
        arr = cp.arange(100)
        result = int(arr.sum().get())
    
    assert_equal(result, 4950, "sum of 0..99")


def test_stream_synchronize():
    """Test stream synchronization."""
    import cupy as cp
    
    stream = CudaStreamWrapper.create()
    
    with stream:
        arr = cp.arange(1000000)
        arr = arr * 2
    
    # ExternalStream should have synchronize method
    stream.synchronize()
    
    # Computation should be complete
    assert_equal(int(arr[500].get()), 1000)


# =============================================================================
# Test runner
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n=== CudaStreamWrapper Tests ===")
    
    tests = [
        # Basic tests
        test_stream_wrapper_basic,
        test_stream_wrapper_priority,
        test_stream_wrapper_is_cupy_stream,
        test_stream_wrapper_repr,
        
        # Context manager tests
        test_stream_wrapper_context_manager,
        test_stream_wrapper_context_manager_nested,
        test_stream_from_cpp_object,
        
        # Reference counting tests
        test_stream_wrapper_refcount,
        test_stream_wrapper_multiple_refs,
        test_stream_wrapper_multiple_get_stream,
        test_stream_holder_lifecycle,
        
        # Edge cases
        test_default_stream_context_manager,
        test_stream_synchronize,
    ]
    
    passed = sum(run_test(t) for t in tests)
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    return passed, len(tests)


def main():
    """Run all tests."""
    print("=" * 60)
    print("ksgpu CudaStreamWrapper Unit Tests")
    print("=" * 60)
    
    try:
        import cupy as cp
        # Quick check that CUDA is working
        _ = cp.zeros(1)
    except ImportError:
        print("\nERROR: cupy not installed")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: cupy not working ({e})")
        sys.exit(1)
    
    passed, total = run_all_tests()
    
    print("\n" + "=" * 60)
    if passed < total:
        print(f"FAILED: {passed}/{total} tests passed")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {total} tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

