import os
import ctypes
import sysconfig
import numpy


# The "ctypes trick": before doing anything else, we load the libraries
#
#  lib/libksgpu.so
#  ksgpu_pybind11...so
#
# using ctypes.CDLL(..., ctypes.RTLD_GLOBAL). This way of loading the libraries
# ensures that other packages which depend on ksgpu can see all symbols in these
# libraries, without needing to link to them explicitly.


# Equivalent to 'python3-config --extension-suffix'
ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

libksgpu_filename = os.path.join(os.path.dirname(__file__), 'lib', 'libksgpu.so')
ctypes.CDLL(libksgpu_filename, mode = ctypes.RTLD_GLOBAL)

libksgpu_pybind11_filename = os.path.join(os.path.dirname(__file__), 'ksgpu_pybind11' + ext_suffix)
ctypes.CDLL(libksgpu_pybind11_filename, mode = ctypes.RTLD_GLOBAL)


####################################################################################################


# FIXME this makes dir(ksgpu) look weird, since it consists entirely of ad hoc functions
# for testing. I'll probably clean this up when there's more python functionality in ksgpu.
from .ksgpu_pybind11 import *

def launch_busy_wait_kernel(arr, a40_seconds):
    """
    Launches a "busy wait" kernel with one threadblock and 32 threads.
    Useful for testing stream/device synchronization.

    The 'arr' argument is a caller-allocated length-32 uint32 array.
    The 'a40_seconds' arg determines the amount of work done by the kernel,
    normalized to "seconds on an NVIDIA A40".
    """

    # We import cupy here, since putting 'import cupy' at the top of the file would lead
    # to the following tragic sequence of events:
    #
    #   - "Downstream" modules (e.g. gpu_mm) must declare cupy as a build-time
    #     dependency, since they 'import ksgpu' in order to get the location
    #     of the ksgpu .hpp files.
    #
    #   - When a downstream module is installed with 'pip install', pip creates
    #     an "isolated" build environment, without cupy installed (even if cupy
    #     is already installed in the "main" environment).
    #
    #   - This triggers 'pip install cupy' in the build env (not the main env).
    #
    #   - Since pypi cupy is a source distributionb, not a precompiled distribution,
    #     this takes forever and is unlikely to work.
    #
    # (Note: launch_busy_wait_kernel() is the only function in ksgpu which uses cupy.)
    
    import cupy
    
    ksgpu_pybind11._launch_busy_wait_kernel(arr, a40_seconds, cupy.cuda.get_current_stream().ptr)
