# Build system

- Uses pipmake, a tiny build system which is pip-compatible, but forwards pip commands to a Makefile (e.g. `pip install` forwards to `make wheel`).
- The Makefile runs the script `makefile_helper.py,` which contains miscellaneous logic that's more convenient to write in python than in Makefile language. The output of this script is a file `makefile_helper.out`, containing variable declarations in Makefile language.
- Build the whole project with `make -j 32`. Compile time is high, so `make src_lib/FILE.o` may be useful when working on `src_lib/FILE.cu`.
- If you add a new source file, see comments near the top of `Makefile` for instructions on how to modify the makefile.

## File Structure

```
include/ksgpu/*.hpp                 - Public C++ headers
src_lib/*.cpp                       - Pure C++ implementations
src_lib/*.cu                        - CUDA implementations (kernels)
src_pybind11/*.cpp                  - Pybind11 bindings
ksgpu/*.py                          - High-level python interface
ksgpu/__main__.py                   - Command-line driver ('python -m ksgpu <command>')
lib/libksgpu.so                     - Shared library
```

There are no standalone binaries: tests, timings, and utilities are C++
functions in src_lib/ (declared in include/ksgpu/command_line_interface.hpp), which
are python-bound and dispatched by the command-line driver, e.g.
`python -m ksgpu test --arr` or `python -m ksgpu time --mcpy`.