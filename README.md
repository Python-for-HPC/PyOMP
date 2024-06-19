# PyOMP
OpenMP for Python in Numba

Currently, PyOMP is distributed as a full version of Numba which is based on a Numba version a few versions behind mainline.  Since Numba is available for every combination of the past few Python versions and the past few NumPy versions and various operating systems and architectures, there is quite an extensive build infrastructure required to get all these combinations and we haven't sorted that out yet.  As a result, the current PyOMP is distributed only for Linux x86-64 with one specific Python and NumPy version.  That distribution is available with the conda command in the next section.  Moreover, there is a glibc dependency problem such that this distirbution won't work if your machine has lower than glibc version 2.29.  It is possible to build from sources for other environments but this process is manual and generally difficult.

In the future, we plan on converting PyOMP to a Numba extension which should eliminate the Python and NumPy versioning issues.  We are working on making it easier to get a working version of PyOMP for your given environment.

Installing with Conda
---------------------

conda install Python-for-HPC::numba Python-for-HPC::llvmlite -c conda-forge -–override-channels

Installing with Docker
----------------------

# arm64 with Jupyter
docker pull ghcr.io/ggeorgakoudis/pyomp-jupyter-arm64:latest

docker run -t -p 8888:8888 pyomp-jupyter

# arm64 terminal
docker pull ghcr.io/ggeorgakoudis/pyomp-arm64:latest

docker run -it pyomp 

# amd64 with Jupyter
docker pull ghcr.io/ggeorgakoudis/pyomp-jupyter-amd64:latest

docker run -t -p 8888:8888 pyomp-jupyter

# amd64 terminal
docker pull ghcr.io/ggeorgakoudis/pyomp-amd64:latest

docker run -it pyomp 

Building
--------

Run build.sh.

Using
-----

Import Numba and add the njit decorator to the function in which you want to use OpenMP.
Add with contexts for each OpenMP region you want to have where the with context is
openmp_context from the numba.openmp module.

The most common target directive (target teams distribute parallel for) should now work.
Some other variations of target directives and nested directives also work but not all
combinations are currently supported.  Target directives support the device clause and
for PyOMP, device(0) always refers to a multi-core CPU target device where as device(1)
always refers to an Nvidia target device.

Example
-------

    from numba import njit
    from numba.openmp import openmp_context as openmp

    @njit
    def test_pi_loop():
        num_steps = 100000
        step = 1.0 / num_steps

        the_sum = 0.0
        omp_set_num_threads(4)

        with openmp("parallel"):
            with openmp("for reduction(+:the_sum) schedule(static)"):
                for j in range(num_steps):
                    c = step
                    x = ((j-1) - 0.5) * step
                    the_sum += 4.0 / (1.0 + x * x)

        pi = step * the_sum
        return pi

