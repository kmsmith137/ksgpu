"""Utility functions for ksgpu"""

from . import ksgpu_pybind11


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

