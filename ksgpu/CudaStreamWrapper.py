"""
CudaStreamWrapper: Python wrapper for CUDA streams with reference counting.

This module provides CudaStreamWrapper, a Python class that:
  1. Inherits from cupy.cuda.ExternalStream for seamless cupy interop
  2. Maintains proper reference counting so streams stay alive while Python holds references

Architecture
------------
The implementation uses two classes:

  _CudaStreamWrapperBase (C++, via pybind11):
      - Wraps the C++ ksgpu::CudaStreamWrapper class
      - Contains std::shared_ptr<CUstream_st> for RAII-based stream lifetime management
      - When copied (e.g., returned to Python), shared_ptr refcount is incremented
      - Stream is destroyed (cudaStreamDestroy) when refcount reaches 0
  
  CudaStreamWrapper (Python, this class):
      - Inherits from cupy.cuda.ExternalStream
      - Holds a reference to _CudaStreamWrapperBase in self._cpp_wrapper
      - This reference keeps the C++ shared_ptr alive as long as Python holds the object
      - Inherits context manager (__enter__/__exit__) from ExternalStream

Reference Counting Flow
-----------------------
When a C++ object returns a stream to Python:

  1. C++ method returns CudaStreamWrapper (contains shared_ptr, refcount=N)
  2. Pybind11 creates _CudaStreamWrapperBase with COPY of CudaStreamWrapper (refcount=N+1)
  3. Python code wraps in CudaStreamWrapper(cpp_wrapper) which stores _cpp_wrapper
  4. C++ object may be destroyed (refcount=N, stream still alive)
  5. Python CudaStreamWrapper is garbage collected, _cpp_wrapper destroyed (refcount=N-1)
  6. When refcount reaches 0, cudaStreamDestroy() is called

Example Usage
-------------
    import cupy as cp
    from ksgpu import CudaStreamWrapper
    
    # Create a new stream
    stream = CudaStreamWrapper.create()
    
    # Use as context manager (inherited from ExternalStream)
    with stream:
        arr = cp.zeros(100)  # Uses this stream
        result = cp.sum(arr)
    
    # Works with C++ objects that return streams
    # (assuming some_cpp_object.get_stream() returns _CudaStreamWrapperBase)
    with CudaStreamWrapper(some_cpp_object.get_stream()):
        arr = cp.arange(1000)
"""

import cupy.cuda
from .ksgpu_pybind11 import _CudaStreamWrapperBase


class CudaStreamWrapper(cupy.cuda.ExternalStream):
    """
    RAII wrapper for cudaStream_t that inherits from cupy.cuda.ExternalStream.
    
    This class provides:
      - Seamless cupy interop (isinstance(stream, cupy.cuda.Stream) is True)
      - Context manager protocol for setting the current CUDA stream
      - Proper reference counting via an internal C++ wrapper object
    
    The stream is automatically destroyed when all Python and C++ references
    are released.
    
    Parameters
    ----------
    cpp_wrapper : _CudaStreamWrapperBase
        Internal C++ wrapper object. Users should not pass this directly;
        use the create() or default_stream() class methods instead.
    
    Attributes
    ----------
    ptr : int
        Raw cudaStream_t pointer as integer.
    is_default : bool
        True if this wraps the default stream (ptr=0).
    
    Examples
    --------
    Create a new stream and use with cupy:
    
        >>> import cupy as cp
        >>> from ksgpu import CudaStreamWrapper
        >>> stream = CudaStreamWrapper.create()
        >>> with stream:
        ...     arr = cp.zeros(100)
        ...     result = cp.sum(arr)
    
    Verify it's a cupy stream:
    
        >>> isinstance(stream, cupy.cuda.Stream)
        True
    
    Notes
    -----
    The reference counting is handled by storing the C++ wrapper object
    (_CudaStreamWrapperBase) in self._cpp_wrapper. This object contains a
    std::shared_ptr<CUstream_st> which ensures the CUDA stream stays alive
    as long as any Python or C++ code holds a reference.
    """
    
    def __init__(self, cpp_wrapper):
        """
        Initialize CudaStreamWrapper from a C++ wrapper object.
        
        Users should typically use create() or default_stream() instead
        of calling this constructor directly.
        
        Parameters
        ----------
        cpp_wrapper : _CudaStreamWrapperBase
            Internal C++ wrapper object that manages the stream lifetime.
        """
        # Store the C++ wrapper to prevent premature destruction of the stream.
        # The _CudaStreamWrapperBase contains a std::shared_ptr<CUstream_st>,
        # so as long as we hold this reference, the stream stays alive.
        self._cpp_wrapper = cpp_wrapper
        
        # Initialize the ExternalStream base class with the raw pointer.
        # ExternalStream expects an integer pointer value.
        super().__init__(cpp_wrapper.ptr)
    
    @classmethod
    def create(cls, priority=0):
        """
        Create a new CUDA stream.
        
        Parameters
        ----------
        priority : int, optional
            Stream priority. CUDA priorities follow a convention where lower
            numbers represent higher priorities. 0 is default priority.
            The range of meaningful priorities can be queried using
            cudaDeviceGetStreamPriorityRange(). On an A40, the range is [-5, 0].
        
        Returns
        -------
        CudaStreamWrapper
            A new stream wrapper.
        
        Examples
        --------
            >>> stream = CudaStreamWrapper.create()
            >>> stream.ptr != 0
            True
            >>> high_priority_stream = CudaStreamWrapper.create(priority=-5)
        """
        return cls(_CudaStreamWrapperBase.create(priority))
    
    @classmethod
    def default_stream(cls):
        """
        Create a wrapper for the default CUDA stream.
        
        Returns
        -------
        CudaStreamWrapper
            A wrapper for the default stream (ptr=0).
        
        Examples
        --------
            >>> stream = CudaStreamWrapper.default_stream()
            >>> stream.ptr
            0
            >>> stream.is_default
            True
        """
        return cls(_CudaStreamWrapperBase())
    
    @property
    def is_default(self):
        """
        True if this wraps the default stream (ptr=0).
        
        Returns
        -------
        bool
        """
        return self._cpp_wrapper.is_default
    
    def __repr__(self):
        return f"CudaStreamWrapper(ptr=0x{self.ptr:x}, is_default={self.is_default})"


class StreamHolder:
    """
    Test helper class that holds a CUDA stream.
    
    This class is used in unit tests to verify reference counting behavior.
    It creates a stream on construction, and get_stream() returns a
    CudaStreamWrapper that shares ownership of the stream.
    
    The test pattern is:
      1. Create StreamHolder (creates stream)
      2. Call get_stream() to get a CudaStreamWrapper
      3. Delete the StreamHolder
      4. Verify the stream is still valid (because Python holds a reference)
    
    Examples
    --------
        >>> holder = StreamHolder()
        >>> stream = holder.get_stream()
        >>> ptr = stream.ptr
        >>> del holder  # Stream still alive via 'stream'
        >>> stream.ptr == ptr  # Still valid
        True
    """
    
    def __init__(self):
        """Create a StreamHolder with a new CUDA stream."""
        from .ksgpu_pybind11 import _StreamHolderBase
        self._holder = _StreamHolderBase()
    
    def get_stream(self):
        """
        Get the stream as a CudaStreamWrapper.
        
        Returns
        -------
        CudaStreamWrapper
            A wrapper that shares ownership of the stream.
        """
        return CudaStreamWrapper(self._holder.get_stream())

