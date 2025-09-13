[![Documentation Status](https://readthedocs.org/projects/pyomp/badge/?version=latest)](https://pyomp.readthedocs.io/en/latest/?badge=latest)
[![pypi](https://github.com/Python-for-HPC/PyOMP/actions/workflows/build-upload-wheels.yml/badge.svg?branch=main&event=release)](https://github.com/Python-for-HPC/PyOMP/actions/workflows/build-upload-wheels.yml)
[![conda](https://github.com/Python-for-HPC/PyOMP/actions/workflows/build-upload-conda.yml/badge.svg?branch=main&event=release)](https://github.com/Python-for-HPC/PyOMP/actions/workflows/build-upload-conda.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Python-for-HPC/binder/HEAD)

# PyOMP
OpenMP for Python CPU/GPU parallel programming, powered by Numba.

PyOMP provides a familiar interface for CPU/GPU programming using OpenMP
abstractions adapted for Python.
Besides effortless programmability, PyOMP generates fast code using Numba's JIT
compiler based on LLVM, which is competitive with equivalent C/C++ implementations.

PyOMP is developed and distributed as an *extension* to Numba, so it uses
Numba as a dependency.
It is currently tested with Numba versions 0.57.x, 0.58.x, 0.59.x, 0.60.x on the
following architecture and operating system combinations: linux-64 (x86_64),
osx-arm64 (mac), linux-arm64, and linux-ppc64le.
Installation is possible through `pip` or `conda`, detailed in the next section.

As PyOMP builds on top of the LLVM OpenMP infrastructure, it also inherits its
limitations: GPU support is only available on Linux.
Also, PyOMP currently supports only NVIDIA GPUs with AMD GPU support planned for.

## Installation

### Pip
PyOMP is distributed through PyPI, installable using the following command:

```bash
pip install pyomp
```

### Conda
PyOMP is also distributed through Conda, installable using the following command:

```bash
conda install -c python-for-hpc -c conda-forge pyomp
```

Besides a standard installation, we also provide the following options to
quickly try out PyOMP online or through a container.

### Trying it out

#### Binder
You can try it out for free on a multi-core CPU in JupyterLab at the following link:

https://mybinder.org/v2/gh/Python-for-HPC/binder/HEAD

#### Docker

We also provide pre-built containers for arm64 and amd64 architectures with
PyOMP and Jupyter pre-installed.
The following show how to access the container through the terminal or using
Jupyter.

First pull the container
```bash
docker pull ghcr.io/python-for-hpc/pyomp:latest
```

To use the terminal, run a shell on the container
```bash
docker run -it ghcr.io/python-for-hpc/pyomp:latest /bin/bash
```

To use Jupyter, run without arguments and forward port 8888.
```bash
docker run -it -p 8888:8888 ghcr.io/python-for-hpc/pyomp:latest
```
Jupyter will start as a service on localhost with token authentication by default.
Grep the url with the token from the output and copy it to the browser.
```bash
...
[I 2024-09-15 17:24:47.912 ServerApp]     http://127.0.0.1:8888/tree?token=<token>
...
```

## Usage

From `numba.openmp` import the `@njit` decorator and the `openmp_context`.
Decorate with `njit` the function you want to parallelize with OpenMP and
describe parallelism in OpenMP directives using `with` contexts.
Enjoy the simplicity of OpenMP with Python syntax and parallel performance.

For a list of supported OpenMP directives and more detailed information, check
out the [Documentation](https://pyomp.readthedocs.io).

PyOMP supports both CPU and GPU programming.
For GPU programming, PyOMP implements OpenMP's `target` directive for offloading
and supports the `device` clause, with `device(0)` by convention offloading to a
GPU device.
It is also possible to use the host as a multi-core CPU target device (mainly
for testing purposes) by setting `device(1)`.

### Example

This is an example of calculating $\pi$ with PyOMP with a `parallel for` loop
using CPU parallelism:

```python
from numba.openmp import njit
from numba.openmp import openmp_context as openmp

@njit
def calc_pi(num_steps):
    step = 1.0 / num_steps
    red_sum = 0.0
    with openmp("parallel for reduction(+:red_sum) schedule(static)"):
        for j in range(num_steps):
            x = ((j-1) - 0.5) * step
            red_sum += 4.0 / (1.0 + x * x)

    pi = step * red_sum
    return pi

print("pi =", calc_pi(1000000))
```

and this is the same example using GPU offloading:

```python
from numba.openmp import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_thread_num

@njit
def calc_pi(num_steps):
    step = 1.0/num_steps
    red_sum = 0.0
    with openmp("target map(tofrom: red_sum)"):
        with openmp("loop private(x) reduction(+:red_sum)"):
               for i in range(num_steps):
                   tid = omp_get_thread_num()
                   x = (i+0.5)*step
                   red_sum += 4.0 / (1.0 + x*x)

    pi = step * red_sum
    print("pi=", pi)

print("pi =", calc_pi(1000000))
```

## Support

We welcome any feedback, bug reports, or feature requests.
Please open an [Issue](https://github.com/Python-for-HPC/PyOMP/issues) or post
in [Discussions](https://github.com/Python-for-HPC/PyOMP/discussions).

## License

PyOMP is licensed under the BSD-2-Clause license (see [LICENSE](LICENSE)).

The package includes the LLVM OpenMP runtime library, which is distributed under
the Apache License v2.0 with LLVM Exceptions. See
[LICENSE-OPENMP.txt](LICENSE-OPENMP.txt) for details.
