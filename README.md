### GPU C++/CUDA core utils


1. Make sure you have cuda, cupy, cufft, cublas, curand, pybind11
installed. This conda environment ("heavycuda") works for me:
```
    conda create -c conda-forge -n heavycuda \
         cupy scipy matplotlib pybind11 \
         cuda-nvcc libcublas-dev libcufft-dev libcurand-dev
```

If you have the cuda toolkit installed outside conda, then you can
omit some of these conda packages.


2. Install with either:
```
      # Clone github repo and make 'editable' install
      git clone https://github.com/kmsmith137/ksgpu
      cd ksgpu
      pip install -v -e .
```
or:
```
     # Install from pypi
     pip install -v ksgpu
```

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
