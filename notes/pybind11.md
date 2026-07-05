# Pybind11

## Method injections

Sometimes, we want to write methods in python, and add them to a pybind11-wrapped
C++ class. This can be done with the class decorator `inject_methods`:

```
    # Assume MyClass is a C++ class bound via Pybind11
    from .ksgpu_pybind11 import MyClass
    from .pybind11_injections import inject_methods

    @inject_methods(MyClass)
    class MyClassInjections:
        # 1. Injecting a standard method
        def nice_print(self):
            print(f"Value: {self.get_value()}")

        # 2. Injecting/Overriding the Constructor (__init__)
        # We save the original C++ init so we can call it.
        _cpp_init = MyClass.__init__

        def __init__(self, value, label="Default"):
            # Call the C++ constructor
            self._cpp_init(value)
            # Add pure-Python attributes 
            # (Requires py::dynamic_attr() in C++)
            self.label = label
```

If the injector class has a docstring, `inject_methods` copies it onto the target
class -- overriding the docstring set in the pybind11 binding; if the injector has no
docstring, the pybind11 one stands. So a class's docstring may live on either side.
Keep it on exactly ONE side and leave a `#` comment on the other pointing to it:

- Option 1: docstring in the pybind11 `py::class_<...>(m, "Name", "docstring...")`;
  the injector class then has no docstring (just a comment saying so).
- Option 2: docstring on the injector class; the pybind11 `py::class_<...>(m, "Name")`
  then sets none (a comment there points to the injector).

Use option 2 when the class's primary Python interface is the injection itself (e.g. a
context-manager usage pattern); otherwise prefer option 1.

## Source file organization

Pybind11 code is in the following source files:
```
   src_pybind11/ksgpu_pybind11.cpp
   src_pybind11/pybind11_utils.cpp
```
These get compiled into a single extension module `ksgpu_pybind11.so`.

Note that each pybind11 class will appear with two names -- for example `ksgpu.ksgpu_pybind11.Dtype`
is the same as `ksgpu.Dtype`. In python code, always use the latter "non-pybind11" name if possible.

If you are asked to python-bind a new class, please make sure that it is also imported into a python subpackage,
and documented (with `autoclass`) in the sphinx docs.

## General notes

- Please write docstrings in the pybind11 code, but keep them concise and avoid superficial comments. If the meaning of a member/method is self-evident, then don't write a docstring.

- When writing docstrings, comments for the corresponding C++ function/class may be useful.

- If it's technically challenging (or awkward) to python-bind a C++ class member/method, or if the member/method seems unlikely to be useful from python, then skip it. Please list in the chat all "skipped" members/methods.

- If a class has method injections, then add a C++ comment to the pybind11 code with a concise description of the injections.

- A class's docstring may live either in the pybind11 binding or on the `inject_methods` injector class (an injector docstring overrides the pybind11 one). Keep it on exactly one side and put a pointer comment on the other (see "Method injections" above).

- Don't use lambda-functions in cases where a named function (or constructor) would be equivalent.

- Some C++ classes have protected constructors and a public static method `shared_ptr<X> X::create(...)`. In such cases, the python syntax should be `x = X(...)`, not `x = X.create(...)`.

## Specific argument types

- The rules below usually require method injections to implement. They apply to both constructors and non-constructor methods.

- If a C++ function takes an `aflags` argument (from `ksgpu/mem_utils.hpp`), then call `ksgpu.parse_aflags(aflags)` before passing it to the C++ function.

- If a C++ function takes a `ksgpu::Dtype` argument, then call the `ksgpu.Dtype` constructor on `x` before passing it to the C++ function.

- If a C++ function returns a bare pointer or `shared_ptr<void>`, then don't python-wrap it unless specifically requested.

- If a C++ member has type `dim3`, then don't python-wrap it unless specifically requested.

- If a C++ function has an argument of type `ostream &`, `YAML::Emitter &`, or `YamlFile &`, then don't python-wrap it unless specifically requested.

- C++ atomics must be converted to non-atomic types before converting to python.

- If a C++ function has a `cudaStream_t` argument, it should appear in python as `stream=None`, where `stream` is a `cupy.cuda.stream`, and the default is the current cupy stream.

- If a C++ function returns a `ksgpu::CudaStreamWrapper`, then the pybind11 binding will return type `ksgpu_pybind11._CudaStreamWrapperBase`. It should appear to a python caller as returning type `ksgpu.CudaStreamWrapper` instead. Define a python wrapper which does the conversion. (This is a one-liner: the `ksgpu.CudaStreamWrapper` constructor takes a `_CudaStreamWrapperBase` argument. See `ksgpu/CudaStreamWrapper.py` for more context.)
