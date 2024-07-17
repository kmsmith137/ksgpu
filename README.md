### GPU C++/CUDA core utils

Installation: first, install cupy and numpy. Here's a convenient conda env:
```
conda create -c conda-forge -n ksgpu python cupy meson-python pybind11
```
Then you should be able to install with pip, e.g.
```
pip install -v git+https://github.com/kmsmith137/ksgpu.git    # current 'main' branch
pip install -v ksgpu          # latest pypi version (https://pypi.org/project/ksgpu/)
```
Note: the pypi package is a source distribution (not a precompiled distribution) so in both
cases, `pip' will attempt to compile the source files.

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
