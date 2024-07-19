PYTHON := python3
SHELL := /bin/bash
NVCC = nvcc -std=c++17 $(ARCH) -m64 -O3 --compiler-options -Wall,-fPIC
NVCC_PYEXT = $(NVCC) $(PYBIND11_INC) $(NUMPY_INC)

.DEFAULT_GOAL: all
.PHONY: all clean install .FORCE


ARCH = -gencode arch=compute_80,code=sm_80
ARCH += -gencode arch=compute_86,code=sm_86
ARCH += -gencode arch=compute_89,code=sm_89
# ARCH += -gencode arch=compute_90,code=sm_90

PY_INSTALL_DIR := $(shell $(PYTHON) choose_python_install_dir.py)
ifneq ($(.SHELLSTATUS),0)
  $(error choose_python_install_dir.py failed!)
endif

# Note: 'pybind11 --includes' includes the base python dir (e.g. -I/xxx/include/python3.12) in addition to pybind11
PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)
ifneq ($(.SHELLSTATUS),0)
  $(error 'pybind11 --includes' failed. Maybe pybind11 is not installed?)
endif

NUMPY_INC := -I$(shell $(PYTHON) -c 'import numpy; print(numpy.get_include())')
ifneq ($(.SHELLSTATUS),0)
  $(error 'numpy.get_include() failed'. Maybe numpy is not installed?)
endif


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

# FIXME instead of ksgpu_pybind11.o, should I be using a name like ksgpu_pybind11.cpython-312-x86_64-linux-gnu.so?
PYOFILES = \
  src_pybind11/ksgpu_pybind11.o \
  src_pybind11/pybind11_utils.o

LIBFILES = \
  lib/libksgpu.so \
  lib/libksgpu.a

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

PYEXTFILES = \
  src_python/ksgpu/ksgpu_pybind11.so \
  src_python/ksgpu/libksgpu.so   # just a copy of lib/libksgpu.so

SRCDIRS = \
  include \
  include/ksgpu \
  src_bin \
  src_lib \
  src_pybind11 \
  src_python \
  src_python/ksgpu \

all: $(LIBFILES) $(XFILES) $(PYOFILES) $(PYEXTFILES)

%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

src_pybind11/%.o: src_pybind11/%.cu $(HFILES)
	$(NVCC_PYEXT) -c -o $@ $<

bin/%: src_bin/%.o lib/libksgpu.so
	mkdir -p bin && $(NVCC) -o $@ $^

lib/libksgpu.so: $(OFILES)
	@mkdir -p lib
	rm -f $@
	$(NVCC) -shared -o $@ $^

lib/libksgpu.a: $(OFILES)
	@mkdir -p lib
	rm -f $@
	ar rcs $@ $^

src_python/ksgpu/libksgpu.so: lib/libksgpu.so
	cp -f $< $@

# Check out the obnoxious level of quoting needed around $ORIGIN!
src_python/ksgpu/ksgpu_pybind11.so: $(PYOFILES) src_python/ksgpu/libksgpu.so
	cd src_python/ksgpu && $(NVCC) '-Xcompiler=-Wl\,-rpath\,'"'"'$$ORIGIN/'"'"'' -shared -o ksgpu_pybind11.so $(addprefix ../../,$(PYOFILES)) libksgpu.so

install: src_python/ksgpu/ksgpu_pybind11.so src_python/ksgpu/libksgpu.so
	@mkdir -p $(PY_INSTALL_DIR)/ksgpu
	@mkdir -p $(PY_INSTALL_DIR)/ksgpu/include/ksgpu
	cp -f src_python/ksgpu/__init__.py $^ $(PY_INSTALL_DIR)/ksgpu/
	cp -f include/ksgpu.hpp $(PY_INSTALL_DIR)/ksgpu/include
	cp -f $(HFILES) $(PY_INSTALL_DIR)/ksgpu/include/ksgpu


# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh; do echo $$f; done; done >$@

clean:
	rm -f $(XFILES) $(LIBFILES) $(PYEXTFILES) source_files.txt *~
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o; done

# INSTALL_DIR ?= /usr/local
#
# install: $(LIBFILES)
# 	mkdir -p $(INSTALL_DIR)/include
#	mkdir -p $(INSTALL_DIR)/lib
#	cp -rv lib $(INSTALL_DIR)/
#	cp -rv include $(INSTALL_DIR)/
