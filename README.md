### GPU C++/CUDA core utils


1. Make sure you have cuda, cupy, cufft, cublas, curand, pybind11
installed. This conda environment ("heavycuda") works for me:
```
conda create -c conda-forge -n heavycuda \
         cupy scipy matplotlib pybind11 \
         cuda-nvcc libcublas-dev libcufft-dev libcurand-dev
```

2. Install with::

    make -j install

to install in python site-packages.

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
