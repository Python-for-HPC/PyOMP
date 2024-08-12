[![Documentation Status](https://readthedocs.org/projects/pyomp/badge/?version=latest)](https://pyomp.readthedocs.io/en/latest/?badge=latest)

# PyOMP
OpenMP for Python in Numba

Currently, PyOMP is distributed as a full version of Numba which is based on a Numba version a few versions behind mainline.  Since Numba is available for every combination of the past few Python versions and the past few NumPy versions and various operating systems and architectures, there is quite an extensive build infrastructure required to get all these combinations and recently we have sorted out some of these combinations.  The architecture and operating system combinations that currently work are linux-ppc6le, linux-64 (x86_64), and osx-arm64 (mac).  These distributions are available with the conda command in the next section.  Due to PyOMP using the LLVM OpenMP infrastructure, we also inherit its limitations which means that GPU support is only available on Linux.  It is possible to build from sources for other environments but this process is manual and generally difficult.

In the future, we plan on converting PyOMP to a Numba extension which should eliminate the Python and NumPy versioning issues.  We continue to work to make it easier to get a working version of PyOMP for other environments.

## Installation

### Conda
The easiest and recommended way to install PyOMP is through conda, currently
supporting linux-ppc6le, linux-64 (x86_64), and osx-arm64 (mac) architectures

```
conda install -c python-for-hpc -c conda-forge --override-channels pyomp
```

### Building from source

Building from source is possible but not recommended
```
git clone --recursive https://github.com/Python-for-HPC/PyOMP.git
cd PyOMP/buildscripts/local/
./build-all.sh
```

After building, it is necessary to source the built environment before using PyOMP
```
. setup-env.sh
```

## Trying it out

### Binder
You can try it out for free on a multi-core CPU in JupyterLab at the following link:
https://mybinder.org/v2/gh/ggeorgakoudis/my-binder.git/HEAD

### Docker

We also provide pre-built containers for arm64 and amd64 architectures in two
flavours: (1) with Jupyter pre-installed exporting a web interface to the host,
and (2) without Jupyter assuming usage through the terminal in the container.

#### arm64 
##### Jupyter
```
docker pull ghcr.io/ggeorgakoudis/pyomp-jupyter-arm64:latest
docker run -t -p 8888:8888 pyomp-jupyter
```

##### terminal
```
docker pull ghcr.io/ggeorgakoudis/pyomp-arm64:latest
docker run -it pyomp
```

#### amd64
##### Jupyter
```
docker pull ghcr.io/ggeorgakoudis/pyomp-jupyter-amd64:latest
docker run -t -p 8888:8888 pyomp-jupyter
```

##### terminal
```
docker pull ghcr.io/ggeorgakoudis/pyomp-amd64:latest
docker run -it pyomp
```

## Usage

Import Numba and add the `@njit` decorator to the function in which you want to use OpenMP.
Add `with` contexts for each OpenMP region you want to have, importing the
context `openmp_context` from the `numba.openmp` module.

For a list of supported OpenMP directives and more detailed information, check
out the [Documentation](https://pyomp.readthedocs.io).
PyOMP supports both CPU and GPU programming for NVIDIA GPUs through the `target`
directive for offloading.
For GPU programming, PyOMP supports the `device` clause and by convention the
default without using the clause or providing `device(0)` always refers to the
accelerator GPU device.
It is also possible to use the host as a multi-core CPU target device setting `device(1)`.

### Example

This is an example of calculating $\pi$ with PyOMP with a `parallel for` loop.

```python
from numba import njit
from numba.openmp import openmp_context as openmp

@njit
def calc_pi():
    num_steps = 100000
    step = 1.0 / num_steps

    the_sum = 0.0
    with openmp("parallel for reduction(+:the_sum) schedule(static)"):
        for j in range(num_steps):
            c = step
            x = ((j-1) - 0.5) * step
            the_sum += 4.0 / (1.0 + x * x)

    pi = step * the_sum
    return pi

print("pi =", calc_pi())
```
