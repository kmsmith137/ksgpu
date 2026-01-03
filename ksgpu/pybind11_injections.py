"""
Utilities for extending pybind11-wrapped C++ classes with Python functionality.

This module provides decorators and extensions for adding Python-side methods,
properties, and constructor overrides to C++ classes exposed via pybind11.
"""


def inject_methods(target_class):
    """
    A class decorator that injects methods from the decorated class into the 
    target class. 
    
    This is useful for adding Python logic (like nice __repr__ methods or 
    helper functions) to C++ classes wrapped with Pybind11, without creating 
    subclasses or shadow classes.

    Args:
        target_class: The class to be modified (e.g., the Pybind11 class).

    Returns:
        The modified target_class.

    Example:
    
        >>> # Assume MyClass is a C++ class bound via Pybind11
        >>> from my_module import MyClass 
        >>> 
        >>> @inject_methods(MyClass)
        >>> class MyClassInjections:
        ...     
        ...     # 1. Injecting a standard method
        ...     def nice_print(self):
        ...         print(f"I am a C++ object! Value: {self.get_value()}")
        ...
        ...     # 2. Injecting/Overriding the Constructor (__init__)
        ...     # We save the original C++ init so we can call it.
        ...     _cpp_init = MyClass.__init__
        ...
        ...     def __init__(self, value, label="Default"):
        ...         # Call the C++ constructor
        ...         self._cpp_init(value)
        ...         # Add pure-Python attributes (requires py::dynamic_attr() in C++)
        ...         self.label = label
        ...
        >>> # Usage
        >>> obj = MyClass(10, label="Test")  # Uses the new Python __init__
        >>> obj.nice_print()                 # Uses the injected method
    """

    def decorator(extension_class):
        # Dunder methods that can be safely injected/overridden
        ALLOWED_DUNDERS = {
            '__init__', '__repr__', '__str__', '__hash__', 
            '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
            '__len__', '__bool__', '__iter__', '__contains__',
            '__call__', '__getitem__', '__setitem__', '__delitem__'
        }
        
        # Iterate over all attributes in the extension definition
        for name, value in extension_class.__dict__.items():
            # Skip internal Python attributes (like __module__, __doc__, __weakref__)
            # Allow specific dunders that users commonly want to override
            if name.startswith("__") and name.endswith("__") and name not in ALLOWED_DUNDERS:
                continue
            
            # Inject the method, property, or attribute into the target class
            setattr(target_class, name, value)
            
        # Return the target class so the name 'MyClassInjections' 
        # becomes an alias for 'MyClass' (or can be ignored)
        return target_class
    
    return decorator


# Import the pybind11-wrapped Dtype class
from . import ksgpu_pybind11


@inject_methods(ksgpu_pybind11.Dtype)
class DtypeExtensions:
    """
    Extensions to the C++ Dtype class, including a flexible constructor
    that accepts strings, numpy dtypes, or cupy dtypes.
    """
    
    # Save original C++ constructor
    _cpp_init = ksgpu_pybind11.Dtype.__init__
    
    def __init__(self, x=None, nbits=None):
        """
        Create a ksgpu Dtype object.
        
        Parameters
        ----------
        x : str, numpy.dtype, cupy.dtype, ksgpu.Dtype, or int, optional
            - If ksgpu.Dtype: create a copy
            - If str: parse as dtype string (e.g., "float32", "int64")
            - If numpy/cupy dtype: extract flags and nbits automatically
            - If int: interpreted as flags (requires nbits parameter)
            - If None: create empty/invalid Dtype
        nbits : int, optional
            Number of bits (only used when x is an int representing flags)
        
        Examples
        --------
        >>> import ksgpu
        >>> import numpy as np
        >>> 
        >>> # From string
        >>> dt = ksgpu.Dtype("float32")
        >>> 
        >>> # From numpy dtype
        >>> dt = ksgpu.Dtype(np.float32)
        >>> 
        >>> # From cupy dtype
        >>> import cupy as cp
        >>> dt = ksgpu.Dtype(cp.int64)
        >>> 
        >>> # Copy constructor
        >>> dt2 = ksgpu.Dtype(dt)
        >>> 
        >>> # From flags + nbits (low-level)
        >>> dt = ksgpu.Dtype(ksgpu.Dtype.FLOAT, 32)
        >>> 
        >>> # Empty dtype
        >>> dt = ksgpu.Dtype()
        """
        # Case 1: No arguments → empty Dtype
        if x is None and nbits is None:
            self._cpp_init()
            return
        
        # Case 2: Two arguments (flags, nbits) → direct C++ constructor
        if nbits is not None:
            if not isinstance(x, int):
                raise TypeError(f"When nbits is specified, first argument must be int (flags), got {type(x).__name__}")
            self._cpp_init(x, nbits)
            return
        
        # Case 3: Copy constructor (x is already a ksgpu.Dtype)
        if isinstance(x, ksgpu_pybind11.Dtype):
            self._cpp_init(x.flags, x.nbits)
            return
        
        # Case 4: String → try from_str()
        if isinstance(x, str):
            try:
                parsed = ksgpu_pybind11.Dtype.from_str(x, throw_exception_on_failure=True)
                self._cpp_init(parsed.flags, parsed.nbits)
                return
            except Exception:
                # If from_str() fails, try numpy dtype conversion below
                pass
        
        # Case 5: Try numpy/cupy dtype conversion
        try:
            import numpy as np
            dt = np.dtype(x)  # This handles numpy dtypes, cupy dtypes, and strings
            
            # Map numpy kind to ksgpu flags
            if dt.kind == 'i':  # signed int
                flags = ksgpu_pybind11.Dtype.INT
            elif dt.kind == 'u':  # unsigned int
                flags = ksgpu_pybind11.Dtype.UINT
            elif dt.kind == 'f':  # float
                flags = ksgpu_pybind11.Dtype.FLOAT
            elif dt.kind == 'c':  # complex
                flags = ksgpu_pybind11.Dtype.COMPLEX | ksgpu_pybind11.Dtype.FLOAT
            else:
                raise ValueError(f"Unsupported dtype kind '{dt.kind}': {dt}. "
                               f"Supported kinds: 'i' (int), 'u' (uint), 'f' (float), 'c' (complex)")
            
            nbits_computed = dt.itemsize * 8
            self._cpp_init(flags, nbits_computed)
            return
            
        except Exception as e:
            raise TypeError(f"Could not convert {x!r} (type {type(x).__name__}) to ksgpu.Dtype: {e}")

