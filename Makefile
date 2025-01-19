SHELL := /bin/bash
PYTHON := python3
PYTHON_CONFIG := python3-config
NVCC = nvcc -std=c++17 $(ARCH) -m64 -O3 --compiler-options -Wall,-fPIC
NVCC_PYEXT = $(NVCC) $(PYBIND11_INC) $(NUMPY_INC)

# The 'lib' target includes the python extension.
.PHONY: all bin lib build_wheel clean .FORCE

all: bin build_wheel

####################################################################################################

ARCH = -gencode arch=compute_80,code=sm_80
ARCH += -gencode arch=compute_86,code=sm_86
ARCH += -gencode arch=compute_89,code=sm_89
# ARCH += -gencode arch=compute_90,code=sm_90

# Note: 'pybind11 --includes' includes the base python dir (e.g. -I/xxx/include/python3.12) in addition to pybind11
PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)
ifneq ($(.SHELLSTATUS),0)
  $(error 'pybind11 --includes' failed. Maybe pybind11 is not installed?)
endif

NUMPY_INC := -I$(shell $(PYTHON) -c 'import numpy; print(numpy.get_include())')
ifneq ($(.SHELLSTATUS),0)
  $(error 'numpy.get_include() failed'. Maybe numpy is not installed?)
endif

PYEXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)
ifneq ($(.SHELLSTATUS),0)
  $(error $(PYTHON_CONFIG) --extension-suffix failed)
endif

####################################################################################################

HFILES = \
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

OFILES = \
  src_lib/Array.o \
  src_lib/Barrier.o \
  src_lib/CpuThreadPool.o \
  src_lib/CudaStreamPool.o \
  src_lib/cuda_utils.o \
  src_lib/mem_utils.o \
  src_lib/memcpy_kernels.o \
  src_lib/rand_utils.o \
  src_lib/string_utils.o \
  src_lib/test_utils.o

PYEXT_OFILES = \
  src_pybind11/ksgpu_pybind11.o \
  src_pybind11/pybind11_utils.o

XFILES = \
  bin/time-atomic-add \
  bin/time-fma \
  bin/time-l2-cache \
  bin/time-local-transpose \
  bin/time-memcpy-kernels \
  bin/time-shared-memory \
  bin/time-tensor-cores \
  bin/time-warp-shuffle \
  bin/scratch \
  bin/reverse-engineer-mma \
  bin/test-array \
  bin/test-memcpy-kernels \
  bin/test-sparse-mma \
  bin/show-devices

KSGPU_LIB := ksgpu/lib/libksgpu.so
KSGPU_PYEXT := ksgpu/ksgpu_pybind11$(PYEXT_SUFFIX)

PYFILES = \
  ksgpu/__init__.py

# 'make clean' deletes {*~, *.o, *.so, *.pyc} from these dirs.
CLEAN_DIRS = \
  . \
  include \
  include/ksgpu \
  src_bin \
  src_lib \
  src_pybind11 \
  ksgpu \
  ksgpu/lib \
  ksgpu/__pycache__

# Extra files to be deleted by 'make clean'.
CLEAN_FILES = \
  $(XFILES) \
  ksgpu/include \
  wheel_files.txt

# Directories that should be empty at the end of 'make clean', and can be deleted.
CLEAN_RMDIRS = \
  bin \
  ksgpu/lib \
  ksgpu/__pycache__

####################################################################################################


# FIXME dependencies
%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

src_pybind11/%.o: src_pybind11/%.cu $(HFILES)
	$(NVCC_PYEXT) -c -o $@ $<

$(KSGPU_LIB): $(OFILES)
	@mkdir -p ksgpu/lib
	rm -f $@
	$(NVCC) -shared -o $@ $^

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

$(KSGPU_PYEXT): $(PYEXT_OFILES) $(KSGPU_LIB)
	$(NVCC) -shared -o $@ $(PYEXT_OFILES) -lksgpu -Lksgpu/lib -Xcompiler '"-Wl,-rpath=\\$$ORIGIN/lib"'

bin/%: src_bin/%.o $(KSGPU_LIB)
	mkdir -p bin && $(NVCC) -o $@ $^

# This symlink is needed by pip/pipmake.
ksgpu/include:
	ln -s ../include $@

# Needed by pip/pipmake: list of all files that go into the (non-editable) wheel.
wheel_files.txt: Makefile ksgpu/include
	for f in $(PYFILES) $(KSGPU_PYEXT) $(KSGPU_LIB); do echo $$f; done >$@
	for f in $(HFILES); do echo ksgpu/$$f; done >>$@


####################################################################################################

bin: $(XFILES)

lib: $(KSGPU_LIB) $(KSGPU_PYEXT)

# Target 'build_wheel' is needed by pip/pipmake.
build_wheel: lib wheel_files.txt

# FIXME add CLEAN_RMDIRS
clean:
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o $$d/*.so $$d/*.pyc; done	
	rm -f $(CLEAN_FILES)
