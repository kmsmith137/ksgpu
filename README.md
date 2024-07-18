### GPU C++/CUDA core utils

Installation:

1. First, install cupy and numpy. Here's a convenient conda env:
```
conda create -c conda-forge -n ksgpu python cupy meson-python pybind11
conda activate ksgpu
```
Then, to install ksgpu, you have two options: "pip install" (option 2a),
or "developer install" (option 2b).

2a. pip install: two slightly different versions
```
pip install -v git+https://github.com/kmsmith137/ksgpu     # current 'main' branch
pip install -v ksgpu       # latest pypi version (https://pypi.org/project/ksgpu/)
```
Note: the pypi package is a source distribution (not a precompiled distribution) so in both
cases, `pip' will attempt to compile the source files.

If `pip install` doesn't work, then I recommend trying the developer install (option 2b) instead.
(Debugging a failed pip install can be challenging, since pip sets up a temporary build
environment and deletes it afterwards.

2b. Developer install:
```
git clone https://github.com/kmsmith137/ksgpu
cd ksgpu
meson setup build   # creates directory build/
cd build
ninja install -v
```
Note: the exotic build toolchain (meson + ninja) seems to be the only combination that's
compatible with both pip and cuda.

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
