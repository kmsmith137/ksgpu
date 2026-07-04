### GPU C++/CUDA core utils

1. Set up a conda environment. `ksgpu` relies on the **system** CUDA toolkit
(`nvcc`, CUDA headers) and the **system** host compiler (`gcc`/`g++`); everything
else is conda-installed. The repo ships a minimal `environment.yml`:
```
    # WARNING: this is probably not the conda env you want -- see below!
    conda env create -n ENVNAME -f environment.yml
    conda activate ENVNAME
```
This is a barebones, `ksgpu`-only environment. If you're building `ksgpu` as
part of a larger project -- for example `pirate` for CHIME/CHORD -- then use
that project's environment file instead (e.g. `pirate/environment_minimal.yml`
or `pirate/environment_dev.yml`), which is a superset of this one.

Note: I recommend the `miniforge` fork of conda, not the original conda.

2. The build system supports either python builds with `pip`, or C++ builds
with `make`. Here's what I recommend:
```
    # Step 1. Clone the repo and build with 'make', so that you can read
    # the error messages if anything goes wrong. (pip either generates too
    # little output or too much output, depending on whether you use -v).

    git clone https://github.com/kmsmith137/ksgpu
    cd ksgpu
    make -j 32

    # Step 2: Run a test program, just to verify that it worked.
    
    ./bin/test-array

    # Step 3: If everything looks good, build an editable pip install.
    # This only needs to be done once per conda env (or virtualenv).
    # The pip install is necessary for downstream dependencies (pirate
    # or gpu_mm) that import the 'ksgpu' python module.
    #
    # We install the 'pipmake' build backend from PyPI, then build with
    # --no-build-isolation. The flag isn't strictly necessary here (ksgpu's
    # build deps are all on PyPI) -- it's cautious: it forces the build to use
    # the conda pybind11/numpy instead of pip-fetched copies.
    
    pip install pipmake
    pip install --no-build-isolation -v -e .    # -e for "editable" install

    # Step 4: In the future, if you want to rebuild ksgpu (e.g. after a
    # git pull), you can ignore pip and build with 'make'. (This is only
    # true for editable installs -- for a non-editable install you need
    # to do 'pip install' again.)

    git pull
    make -j 32   # no pip install needed, if existing install is editable
```

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
