project(
    'ksgpu',
    'cuda',
     default_options: ['python.install_env=auto','optimization=3','debug=false','b_ndebug=false','cpp_std=c++17']
)

header_files_toplevel = [
    'include/ksgpu.hpp'
]

header_files_ksgpu_subdir = [
    'include/ksgpu/Array.hpp',
    'include/ksgpu/Barrier.hpp',
    'include/ksgpu/CpuThreadPool.hpp',
    'include/ksgpu/CudaStreamPool.hpp',
    'include/ksgpu/ThreadSafeRingBuffer.hpp',
    'include/ksgpu/complex_type_traits.hpp',
    'include/ksgpu/constexpr_functions.hpp',
    'include/ksgpu/cuda_utils.hpp',
    'include/ksgpu/device_mma.hpp',
    'include/ksgpu/mem_utils.hpp',
    'include/ksgpu/memcpy_kernels.hpp',
    'include/ksgpu/rand_utils.hpp',
    'include/ksgpu/string_utils.hpp',
    'include/ksgpu/test_utils.hpp',
    'include/ksgpu/time_utils.hpp',
    'include/ksgpu/xassert.hpp',
    'include/ksgpu/dlpack.h',
    'include/ksgpu/pybind11.hpp',
    'include/ksgpu/pybind11_utils.hpp'
]

libksgpu_source_files = [
    'src_lib/Array.cu',
    'src_lib/Barrier.cu',
    'src_lib/CpuThreadPool.cu',
    'src_lib/CudaStreamPool.cu',
    'src_lib/cuda_utils.cu',
    'src_lib/mem_utils.cu',
    'src_lib/memcpy_kernels.cu',
    'src_lib/rand_utils.cu',
    'src_lib/string_utils.cu',
    'src_lib/test_utils.cu'
]

pyext_source_files = [
    'src_pybind11/ksgpu_pybind11.cu',
    'src_pybind11/pybind11_utils.cu'
]

python_files_toplevel = [
    'src_python/ksgpu/__init__.py'
]    

local_executables = [
    'reverse-engineer-mma',
    'scratch',
    'show-devices',
    'test-array',
    'test-memcpy-kernels',
    'test-sparse-mma',
    'time-atomic-add',
    'time-fma',
    'time-l2-cache',
    'time-local-transpose',
    'time-memcpy-kernels',
    'time-shared-memory',
    'time-tensor-cores',
    'time-warp-shuffle'
]


################################################################################


cuda_args = ['-std=c++17', '-m64']
cuda_args += ['-gencode','arch=compute_80,code=sm_80']
cuda_args += ['-gencode','arch=compute_86,code=sm_86']
cuda_args += ['-gencode','arch=compute_89,code=sm_89']
link_args = ['-m64']

py3 = import('python').find_installation('python3', pure: false)
# message('py3.path = ', py3.path())
# message('py3.install_dir = ', py3.get_install_dir())

# Show sys.path for debugging
# ret = run_command(py3.path(), '-c', 'import sys; print(sys.path)', check: true)
# message('Debug: py3.sys.path = ', ret.stdout().strip())


################################################################################


pybind11_ret = run_command(py3.path(), '-m', 'pybind11', '--includes', check: true)
pybind11_ret = pybind11_ret.stdout().strip()
# message('Debug: Output of python -m pybind11 --includes: ', pybind11_ret)

pybind11_include_dirs = [ ]
foreach include_dir: pybind11_ret.split()
    include_dir = include_dir.strip().strip('-I')
    pybind11_include_dirs += include_dir
endforeach

# message('Debug: pybind11_include_dirs = ', pybind11_include_dirs)
pybind11_dep = declare_dependency(include_directories: pybind11_include_dirs)


################################################################################


numpy_incdir = run_command(py3.path(), '-c', 'import numpy; print(numpy.get_include())', check: true)
numpy_incdir = numpy_incdir.stdout().strip()

message('Debug: numpy.get_include() = ', numpy_incdir)
numpy_dep = declare_dependency(include_directories: numpy_incdir)


################################################################################


ksgpu_lib = library(
    'ksgpu',
    libksgpu_source_files,
    cuda_args: cuda_args,
    link_args: link_args,
    install: true,
    install_dir: py3.get_install_dir() / 'ksgpu'
)

py3.extension_module(
     'ksgpu_pybind11',   # compiled extension will be named ksgpu_pybind11.so
     sources: pyext_source_files,
     dependencies : [pybind11_dep, numpy_dep],
     cuda_args: cuda_args,
     link_args: link_args,
     link_language : 'cuda',
     link_with: ksgpu_lib,  # link libksgpu.so to ksgpu_pybind11.so
     override_options: ['cpp_rtti=true'],
     subdir: 'ksgpu',
     install: true
)

py3.install_sources(
    python_files_toplevel,
    subdir: 'ksgpu'
)

py3.install_sources(
    header_files_toplevel,
    subdir: 'ksgpu/include'
)

py3.install_sources(
    header_files_ksgpu_subdir,
    subdir: 'ksgpu/include/ksgpu'
)

foreach x: local_executables
    executable(
        x,
        'src_bin/' + x + '.cu',
        cuda_args: cuda_args,
	link_args: link_args,
	link_with: ksgpu_lib
    )
endforeach
