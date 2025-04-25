Usage
=====

OpenMP for Python builds on `Numba <https://numba.pydata.org/>`_ Just-In-Time
(JIT) compilation extensions to implement portable parallel execution using
LLVM's OpenMP implementation.
OpenMP regions are specified using a ``with`` statement for the ``openmp``
context, passing the OpenMP syntax specification as a string.
The OpenMP specification syntax in PyOMP is identical to the C/C++ syntax and
section :doc:`openmp` provides information on what are the currently supported
OpenMP directives.

Diving right in, this is a minimal, parallel `hello world` example:

.. code-block:: python
   :linenos:

   from numba.openmp import njit
   from numba.openmp import openmp_context as openmp
   from numba.openmp import omp_get_thread_num

   @njit
   def hello():
      with openmp("parallel"):
         print("Hello from thread", omp_get_thread_num())

   hello()

The important things to notice here are:

* the required numba imports in lines 1--3

* the ``@njit`` decorator to the ``hello()`` function in line 5 to JIT compile the function in `nopython` mode, which generates native binary code using LLVM bypassing the Python interpreter

* the specification of a parallel region in lines 7--8 that executes the code inside the ``with`` block multithreaded, in parallel, using OpenMP.

OpenMP runtime functions, such as ``omp_get_thread_num()`` returning the unique
thread numerical identifier, are also available in PyOMP, explicitly imported
from the ``numba.openmp`` implementation (see line 3).

Executing this example, the expected output on an 8-core CPU machine is:

.. note::

   The print order may differ

.. code-block:: bash

   Hello from thread 4
   Hello from thread 5
   Hello from thread 7
   Hello from thread 0
   Hello from thread 2
   Hello from thread 3
   Hello from thread 1
   Hello from thread 6

PyOMP supports GPU programming through the ``target`` directive in OpenMP
offloading.
The current implementations supports only NVIDIA GPUs with AMD and Intel support
under way.

This is a very basic example of an OpenMP offloading program for GPU
execution, using the common idiom of ``target teams distribute parallel for``.
It parallelizes a vector addition loop by distributing its iterations (1M
elements) over all available parallelism -- teams of threads mapping to
thread-blocks on the GPU device:

.. code-block:: python
   :linenos:

   from numba.openmp import njit
   from numba.openmp import openmp_context as openmp
   from numba.openmp import omp_get_thread_num
   import numpy as np

   @njit
   def vecadd(a, b, n):
     c = np.empty(n)
     with openmp("target teams distribute parallel for"):
       for i in range(n):
        c[i] = a[i] + b[i]

     return c

   n = 1000000
   a = np.full(n, 1)
   b = np.full(n, 2)
   c = vecadd(a, b, n)
   print("c = ", c)

The expected output is as follows::

   c = [3. 3. 3. ... 3. 3. 3.]
