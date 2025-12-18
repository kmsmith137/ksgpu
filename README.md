### GPU C++/CUDA core utils

**Warning.** If you're building `pirate`, then you need the `chord` branch of `ksgpu`,
not the `main` branch. (The chord branch is ~100 commits ahead of the main branch -- I
hope to merge soon!)

1. Make sure you have cuda, cupy, cufft, cublas, curand, pybind11 installed.
If you're starting from scratch on a minimal system, this conda environment works for me:
```
    # Case 1: starting from scratch on a minimal system.
    conda create -c conda-forge -n ENVNAME \
         cupy scipy matplotlib pybind11 \
         cuda-nvcc libcublas-dev libcufft-dev libcurand-dev
```
If you have the cuda toolkit installed outside conda, then you can omit some
of these conda packages. In particular, on the CHIME/CHORD machines you can do:
```
    # Case 2: installing on a CHIME/CHORD machine.
    # Note: I've also included some packages that you need for 'pirate'.
    conda create -c conda-forge -n ENVNAME \
         cupy scipy matplotlib pybind11 yaml-cpp
```
Note: I recommend the `miniforge` fork of conda, not the original conda.

2. The build system supports either python builds with `pip`, or C++ builds
with `make`. Here's what I recommend:
```
    # Step 1. Clone the repo and build with 'make', so that you can read
    # the error messages if anything goes wrong. (pip either generates too
    # little output or too much output, depending on whether you use -v).

    git clone https://github.com/kmsmith137/ksgpu
    cd ksgpu
    # You may need to switch to the chord branch here -- see above!
    # git checkout chord
    make -j 32

    # Step 2: Run a test program, just to verify that it worked.
    
    ./bin/test-array

    # Step 3: If everything looks good, build an editable pip install.
    # This only needs to be done once per conda env (or virtualenv).
    # The pip install is necessary for downstream dependencies (pirate
    # or gpu_mm) that import the 'ksgpu' python module.
    
    pip install -v -e .    # -e for "editable" install

    # Step 4: In the future, if you want to rebuild ksgpu (e.g. after a
    # git pull), you can ignore pip and build with 'make'. (This is only
    # true for editable installs -- for a non-editable install you need
    # to do 'pip install' again.)

    git pull
    make -j 32   # no pip install needed, if existing install is editable
```

3. For an overview for developers, `.cursor/rules.md` is not a bad place
   to start (originally written for LLMs, but not a bad reference for
   humans). For now, this file is on the chord branch only.

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
