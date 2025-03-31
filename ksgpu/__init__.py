import numpy

def _import_pybind11():
    """This code is in its own function, to avoid polluting the ksgpu namespace with global variables."""
    
    import os
    import ctypes
    import sysconfig

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

_import_pybind11()

####################################################################################################

from .ksgpu_pybind11 import \
    get_cuda_num_devices, \
    get_cuda_device, \
    get_cuda_pcie_bus_id

from . import tests
