import numpy

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
