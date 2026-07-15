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

    If the decorated class has a docstring, it overrides the target class's
    docstring (e.g. the one set in the pybind11 binding); if it has none, the
    target's docstring is left unchanged. So each class with injections should
    carry its docstring on exactly one side -- see notes/pybind11.md.

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
            '__call__', '__getitem__', '__setitem__', '__delitem__',
            '__enter__', '__exit__'
        }
        
        # Iterate over all attributes in the extension definition
        for name, value in extension_class.__dict__.items():
            # The class docstring: a Python class always carries __doc__ in its
            # __dict__ (None when it has no docstring). Copy a non-None docstring
            # onto the target, so an injector can supply the class docstring and
            # have it OVERRIDE the pybind11 one; when the injector omits a
            # docstring, leave the pybind11 docstring intact. (Each injected class
            # should put its docstring on exactly one side -- see the policy in
            # notes/pybind11.md.)
            if name == "__doc__":
                if value is not None:
                    setattr(target_class, "__doc__", value)
                continue

            # Skip internal Python attributes (like __module__, __weakref__).
            # Allow specific dunders that users commonly want to override
            if name.startswith("__") and name.endswith("__") and name not in ALLOWED_DUNDERS:
                continue

            # Inject the method, property, or attribute into the target class
            setattr(target_class, name, value)
            
        # Return the target class so the name 'MyClassInjections' 
        # becomes an alias for 'MyClass' (or can be ignored)
        return target_class
    
    return decorator

