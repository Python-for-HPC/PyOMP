[![Documentation Status](https://readthedocs.org/projects/pyomp/badge/?version=latest)](https://pyomp.readthedocs.io/en/latest/?badge=latest)
[![Deploy conda pkgs (main)](https://github.com/Python-for-HPC/PyOMP/actions/workflows/build-upload-conda.yml/badge.svg?event=release)](https://github.com/Python-for-HPC/PyOMP/actions/workflows/build-upload-conda.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Python-for-HPC/binder/HEAD)

# PyOMP
OpenMP for Python in Numba for CPU/GPU parallel programming.

Currently, PyOMP is distributed as a full version of Numba which is based on a
Numba version a few versions behind mainline.
Since Numba is available for every combination of the past few Python versions
and the past few NumPy versions and various operating systems and architectures,
there is quite an extensive build infrastructure required to get all these
combinations and recently we have sorted out some of these combinations.
The architecture and operating system combinations that currently work are:
linux-64 (x86_64), osx-arm64 (mac), and linux-ppc64le.
These distributions are available with the `conda` command in the next section.

Due to PyOMP using the LLVM OpenMP infrastructure, we also inherit its
limitations which means that GPU support is only available on Linux.

In the future, we plan on converting PyOMP to a Numba extension which should eliminate the Python and NumPy versioning issues.

## Installation

### Conda
PyOMP is distributed as a package through Conda, currently supporting linux-64
(x86_64), osx-arm64 (mac), and linux-ppc64le architectures.

```bash
conda install -c python-for-hpc -c conda-forge --override-channels pyomp
```

## Trying it out

### Binder
You can try it out for free on a multi-core CPU in JupyterLab at the following link:

https://mybinder.org/v2/gh/Python-for-HPC/binder/HEAD

### Docker

We also provide pre-built containers for arm64 and amd64 architectures with
PyOMP and Jupyter pre-installed.
The following show how to access the container through the terminal or using
jupyter.

First pull the container
```
docker pull ghcr.io/python-for-hpc/pyomp:latest
```

To use the terminal, run a shell on the container
```
docker run -it ghcr.io/python-for-hpc/pyomp:latest /bin/bash
```

To use Jupyter, run without arguments and forward port 8888.
```
podman run -it -p 8888:8888 ghcr.io/python-for-hpc/pyomp:latest
```
Jupyter will start as a service on localhost with token authentication by default.
Grep the url with the token from the output and copy it to the browser.
```
...
[I 2024-09-15 17:24:47.912 ServerApp]     http://127.0.0.1:8888/tree?token=<token>
...
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
