# This Makefile will be invoked by the python build system (e.g. via 'pip install'),
# but you can also build individual targets by invoking 'make' directly.

# Disable built-in rules and variables (must be first).
MAKEFLAGS += --no-builtin-rules 
MAKEFLAGS += --no-builtin-variables

# Default target 'all' must be first target in Makefile.
# The 'bin' target builds a bunch of binaries in bin/...
# The 'lib' target builds the C++ library lib/libksgpu.so, and the python extension ksgpu/ksgpu_pybind11...so.
# The 'build_wheel' and 'build_sdist' targets are invoked by 'pip' (or 'make all').
all: bin lib build_wheel build_sdist

.PHONY: all bin lib build_wheel build_sdist clean


####################################################################################################
#
# Variables encoding configuration: PYTHON, NVCC, NVCC_ARCH, NVCC_DEPFLAGS.
#
# FIXME some day I'll define a configure-script mechanism for setting these variables.
# For now, if you want to change the defaults, just edit the Makfile.


PYTHON ?= python3
NVCC ?= nvcc -std=c++17 -m64 -O3 --compiler-options -Wall,-fPIC

# Extra nvcc flags needed to build Makefile dependencies
#   -MMD create dep file, omitting "system" headers
#   -MP add phony target for each header in dep file (makes error reporting less confusing)
# Note: we don't need "-MT $@", since we use in-tree object filenames (x.cu -> x.o).
# Note: we don't need "-MT $*.d", since we use in-tree depfile names (x.cu -> x.d).
NVCC_DEPFLAGS ?= -MMD -MP

# NVIDIA archictecture.
DEFAULT_NVCC_ARCH = -gencode arch=compute_80,code=sm_80
DEFAULT_NVCC_ARCH += -gencode arch=compute_86,code=sm_86
DEFAULT_NVCC_ARCH += -gencode arch=compute_89,code=sm_89
# DEFAULT_ARCH += -gencode arch=compute_90,code=sm_90
NVCC_ARCH ?= $(DEFAULT_NVCC_ARCH)


####################################################################################################
#
# "Derived" config variables: PYTHON_INCDIR, NUMPY_INCDIR, PYBIND11_INCDIR, PYEXT_SUFFIX.
#
# These are autogenerated by makefile_helper.py, and cached in makefile_helper.out.
# PYEXT_SUFFIX is something like .cpython-312-x86_64-linux-gnu.so.


ifneq ($(MAKECMDGOALS),clean)
  include makefile_helper.out
endif

makefile_helper.out: makefile_helper.py Makefile
	$(PYTHON) makefile_helper.py


####################################################################################################


# The main output of the build process is these two libraries.
# Reminder: PYEXT_SUFFIX is something like .cpython-312-x86_64-linux-gnu.so.
KSGPU_LIB := lib/libksgpu.so
KSGPU_PYEXT = ksgpu/ksgpu_pybind11$(PYEXT_SUFFIX)

# These get compiled into lib/libksgpu.so
LIB_SRCFILES := \
  src_lib/Array.cu \
  src_lib/Barrier.cu \
  src_lib/CpuThreadPool.cu \
  src_lib/CudaStreamPool.cu \
  src_lib/cuda_utils.cu \
  src_lib/mem_utils.cu \
  src_lib/memcpy_kernels.cu \
  src_lib/rand_utils.cu \
  src_lib/string_utils.cu \
  src_lib/test_utils.cu

# These get compiled into ksgpu/ksgpu_pybind11....so
PYEXT_SRCFILES := \
  src_pybind11/ksgpu_pybind11.cu \
  src_pybind11/pybind11_utils.cu

# These are in 1-1 corresponding with executables in bin/
# For example, 'src_bin/time-atomic-add.cu' gets compiled to 'bin/time-atomic-add'.
BIN_SRCFILES := \
  src_bin/time-atomic-add.cu \
  src_bin/time-fma.cu \
  src_bin/time-l2-cache.cu \
  src_bin/time-local-transpose.cu \
  src_bin/time-memcpy-kernels.cu \
  src_bin/time-shared-memory.cu \
  src_bin/time-tensor-cores.cu \
  src_bin/time-warp-shuffle.cu \
  src_bin/scratch.cu \
  src_bin/reverse-engineer-mma.cu \
  src_bin/test-array.cu \
  src_bin/test-memcpy-kernels.cu \
  src_bin/test-sparse-mma.cu \
  src_bin/show-devices.cu

# Must list all python source files here.
# (Otherwise they won't show up in 'pip install' or pypi.)
PYFILES := \
  ksgpu/__init__.py

# Must list all header files here.
# (Otherwise they won't show up in 'pip install' or pypi.)
HFILES := \
  include/ksgpu/Array.hpp \
  include/ksgpu/Barrier.hpp \
  include/ksgpu/CpuThreadPool.hpp \
  include/ksgpu/CudaStreamPool.hpp \
  include/ksgpu/ThreadSafeRingBuffer.hpp \
  include/ksgpu/complex_type_traits.hpp \
  include/ksgpu/constexpr_functions.hpp \
  include/ksgpu/cuda_utils.hpp \
  include/ksgpu/device_mma.hpp \
  include/ksgpu/mem_utils.hpp \
  include/ksgpu/memcpy_kernels.hpp \
  include/ksgpu/rand_utils.hpp \
  include/ksgpu/string_utils.hpp \
  include/ksgpu/test_utils.hpp \
  include/ksgpu/time_utils.hpp \
  include/ksgpu/xassert.hpp \
  include/ksgpu/dlpack.h \
  include/ksgpu/pybind11.hpp \
  include/ksgpu/pybind11_utils.hpp

# 'make clean' deletes {*~, *.o, *.d, *.so, *.pyc} from these dirs.
CLEAN_DIRS := . include include/ksgpu lib src_bin src_lib src_pybind11 ksgpu ksgpu/__pycache__

# Extra files to be deleted by 'make clean'.
# Note that 'ksgpu/include' and 'ksgpu/lib' are symlinks, so we put them in CLEAN_FILES, not CLEAN_RMDIRS
CLEAN_FILES := sdist_files.txt wheel_files.txt makefile_helper.out ksgpu/include ksgpu/lib

# Directories that should be empty at the end of 'make clean', and can be deleted.
CLEAN_RMDIRS := bin lib ksgpu/__pycache__


####################################################################################################


LIB_OFILES := $(LIB_SRCFILES:%.cu=%.o)
PYEXT_OFILES := $(PYEXT_SRCFILES:%.cu=%.o)
BIN_XFILES := $(BIN_SRCFILES:src_bin/%.cu=bin/%)

# Must include all .d files, or build will break!
ALL_SRCFILES := $(LIB_SRCFILES) $(PYEXT_SRCFILES) $(BIN_SRCFILES)
DEPFILES := $(ALL_SRCFILES:%.cu=%.d)

SDIST_FILES := pyproject.toml Makefile makefile_helper.py
SDIST_FILES += $(PYFILES) $(ALL_SRCFILES) $(HFILES)

# Some symlinks for the wheel:
#  - header file include/%.hpp gets symlinked to ksgpu/include/%.hpp
#  - library lib/libksgpu.so gets symlinked to ksgpu/lib/libksgpu.so
#  - python extension ksgpu/ksgpu_pybind11...so does not need to be symlinked/renamed.
WHEEL_FILES := $(PYFILES) $(KSGPU_PYEXT) ksgpu/$(KSGPU_LIB)
WHEEL_FILES += $(HFILES:%=ksgpu/%)

# Phony targets. The special targets 'build_wheel' and 'build_sdist' are needed by pip/pipmake.
lib: $(KSGPU_LIB) $(KSGPU_PYEXT)
bin: $(BIN_XFILES)
build_wheel: wheel_files.txt $(KSGPU_LIB) $(KSGPU_PYEXT)
build_sdist: sdist_files.txt

# Symlink {include,lib} into python directory 'ksgpu'.
ksgpu/include:
	ln -s ../include $@
ksgpu/lib:
	ln -s ../lib $@

# Build object files in src_lib/ or src_bin/
%.o: %.cu %.d
	$(NVCC) $(NVCC_ARCH) $(NVCC_DEPFLAGS) -c -o $@ $<

# Build object files in src_pybind11/ with special flags.
src_pybind11/%.o: src_pybind11/%.cu src_pybind11/%.d
	$(NVCC) $(NVCC_ARCH) $(NVCC_DEPFLAGS) -I$(PYTHON_INCDIR) -I$(NUMPY_INCDIR) -I$(PYBIND11_INCDIR) -c -o $@ $<

# Build the C++ library (lib/libksgpu.so)
$(KSGPU_LIB): $(LIB_OFILES)
	@mkdir -p lib
	$(NVCC) $(NVCC_ARCH) -shared -o $@ $^

# Build binaries (bin/*)
bin/%: src_bin/%.o $(KSGPU_LIB)
	@mkdir -p bin/
	$(NVCC) $(NVCC_ARCH) -o $@ $^

# Build the python extension (ksgpu/ksgpu_pybind11...so)
# We want it to automatically pull in the C++ library ksgpu/lib/libksgpu.so.
#
# The python extension has been built correctly if 'objdump -x' shows the following:
#   NEEDED   libksgpu.so
#   RUNPATH  $ORIGIN/lib
#
# The quoting can be understood by working backwards as follows:
#  - g++ command line should look like:   g++ -Wl,-rpath="\$ORIGIN/lib"
#  - nvcc command line should look like:  nvcc -Xcompiler '"-Wl,-rpath=\\$ORIGIN/lib"'
#  - Makefile line should look like:      nvcc -Xcompiler '"-Wl,-rpath=\\$$ORIGIN/lib"'

$(KSGPU_PYEXT): $(PYEXT_OFILES) $(KSGPU_LIB) ksgpu/lib
	$(NVCC) $(NVCC_ARCH) -shared -o $@ $(PYEXT_OFILES) -lksgpu -Lksgpu/lib -Xcompiler '"-Wl,-rpath=\\$$ORIGIN/lib"'

# Needed by pip/pipmake: list of all files that go into the (non-editable) wheel.
wheel_files.txt: Makefile ksgpu/include ksgpu/lib
	rm -f $@
	for f in $(WHEEL_FILES); do echo $$f; done >>$@

# Needed by pip/pipmake: list of all files that go into the sdist.
sdist_files.txt: Makefile
	rm -f $@
	for f in $(SDIST_FILES); do echo $$f; done >>$@

clean:
	@for f in $(foreach d,$(CLEAN_DIRS),$(wildcard $d/*~ $d/*.o $d/*.d $d/*.so $d/*.pyc)); do echo rm $$f; rm $$f; done
	@for f in $(wildcard $(CLEAN_FILES) $(BIN_XFILES)); do echo rm $$f; rm $$f; done
	@for d in $(wildcard $(CLEAN_RMDIRS)); do echo rmdir $$d; rmdir $$d; done

# Specifying .SECONDARY with no prerequisites disables auto-deletion of intermediate files.
.SECONDARY:

# If a depfile is absent, build can still proceed.
$(DEPFILES):

# Include any depfiles which are present.
include $(wildcard $(DEPFILES))
