"""Utility functions for ksgpu"""

from . import ksgpu_pybind11


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


def parse_aflags(x):
    """
    Parse allocation flags from various input types.
    
    Parameters
    ----------
    x : int, str, or other
        The input to parse:
        - If int: returns int(x)
        - If str: parses the string as flag expressions like "af_gpu | af_zero"
                  and returns the computed flag value
        - Other types: raises an exception
    
    Returns
    -------
    int
        The parsed flag value
        
    Raises
    ------
    TypeError
        If x is not an integer or string type
    ValueError
        If a parse error occurs when processing the string
        
    Examples
    --------
    >>> parse_aflags(0x01)
    1
    >>> parse_aflags("af_gpu")
    1
    >>> parse_aflags("af_gpu | af_zero")
    17
    """
    # Handle integer types
    if isinstance(x, (int, bool)):
        return int(x)
    
    # Handle string types
    if isinstance(x, str):
        # Strip whitespace
        x = x.strip()
        
        if not x:
            raise ValueError("parse_aflags: empty string is not a valid flag expression")
        
        # Create namespace with all available flags
        namespace = {
            'af_gpu': ksgpu_pybind11.af_gpu,
            'af_uhost': ksgpu_pybind11.af_uhost,
            'af_rhost': ksgpu_pybind11.af_rhost,
            'af_unified': ksgpu_pybind11.af_unified,
            'af_zero': ksgpu_pybind11.af_zero,
            'af_random': ksgpu_pybind11.af_random,
            'af_mmap_small': ksgpu_pybind11.af_mmap_small,
            'af_mmap_huge': ksgpu_pybind11.af_mmap_huge,
            'af_mmap_try_huge': ksgpu_pybind11.af_mmap_try_huge,
            'af_guard': ksgpu_pybind11.af_guard,
            'af_verbose': ksgpu_pybind11.af_verbose,
        }
        
        try:
            # Evaluate the expression (supports | operator for bitwise OR)
            result = eval(x, {"__builtins__": {}}, namespace)
            
            # Ensure result is an integer
            if not isinstance(result, int):
                raise ValueError(f"parse_aflags: expression '{x}' did not evaluate to an integer (got {type(result).__name__})")
            
            return result
            
        except NameError as e:
            raise ValueError(f"parse_aflags: unrecognized flag name in '{x}': {str(e)}")
        except SyntaxError as e:
            raise ValueError(f"parse_aflags: syntax error in '{x}': {str(e)}")
        except Exception as e:
            raise ValueError(f"parse_aflags: error parsing '{x}': {str(e)}")
    
    # Handle unsupported types
    raise TypeError(f"parse_aflags: expected int or str, got {type(x).__name__}")

