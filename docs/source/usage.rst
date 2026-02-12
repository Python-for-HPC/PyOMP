Usage
=====

Syntax
------

Overview
~~~~~~~~
PyOMP is an extension to `Numba <https://numba.pydata.org/>`_ that brings
OpenMP parallel programming capabilities to Python. All PyOMP functionality
is implemented in the ``numba.openmp`` module.

To use PyOMP, you **must** import from the ``numba.openmp`` module. Key imports
include:

* ``njit`` - The JIT decorator for compiling functions with OpenMP support
* ``openmp_context`` (typically aliased as ``openmp``) - The context manager for specifying OpenMP directives
* OpenMP runtime functions - Functions for querying and controlling parallel execution (e.g., ``omp_get_thread_num()``, ``omp_get_num_threads()``)

OpenMP directives
~~~~~~~~~~~~~~~~~
OpenMP parallel regions are specified using a ``with`` statement for the
``openmp`` context, passing the OpenMP syntax specification as a string.
The ``with`` statement for OpenMP regions **must** always be placed
within a function decorated with the ``@njit`` decorator from ``numba.openmp``.
The OpenMP directive syntax in PyOMP is identical to the C/C++ OpenMP syntax.
For a complete list of supported OpenMP directives with detailed information,
see section :doc:`openmp`.

.. important::
   OpenMP regions **must** be placed within functions decorated with the
   ``@njit`` decorator from ``numba.openmp``. Failure to do so will result in
   undefined behavior, including potential runtime errors or incorrect
   execution. Always ensure that any function containing OpenMP directives is
   properly decorated to avoid such issues.


OpenMP runtime functions
~~~~~~~~~~~~~~~~~~~~~~~~
Beyond directives, PyOMP exposes OpenMP runtime functions that allow you to
query and control parallel execution behavior. These functions are imported
directly from ``numba.openmp``. Commonly used runtime functions include:

* ``omp_get_thread_num()`` - Returns the unique identifier of the calling thread
* ``omp_get_num_threads()`` - Returns the total number of threads in the current parallel region
* ``omp_set_num_threads(n)`` - Sets the number of threads for subsequent parallel regions
* ``omp_get_wtime()`` - Returns elapsed wall-clock time (useful for performance profiling)
* ``omp_get_max_threads()`` - Returns the maximum number of threads available

For a comprehensive list of all available runtime functions, refer to the
:doc:`openmp` documentation.

Examples
--------

CPU parallelism example
~~~~~~~~~~~~~~~~~~~~~~~

Here is a minimal parallel "hello world" example for CPU execution:

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

Key aspects of this example:

* **Imports** (lines 1--3): Import the ``njit`` decorator, ``openmp_context`` context manager, and runtime function ``omp_get_thread_num()`` from ``numba.openmp``.

* **@njit decorator** (line 5): Required to compile the function with OpenMP support using Numba's JIT compiler in nopython mode.

* **Parallel region** (lines 7--8): The ``with openmp("parallel")`` statement creates a parallel region that executes the enclosed code block across multiple threads.

* **Runtime function** (line 8): ``omp_get_thread_num()`` returns the unique thread identifier, demonstrating how to use OpenMP runtime functions within a parallel region.

On an 8-core machine, the output will display one line per thread. Note that thread execution order is non-deterministic:

.. code-block:: bash

   Hello from thread 4
   Hello from thread 5
   Hello from thread 7
   Hello from thread 0
   Hello from thread 2
   Hello from thread 3
   Hello from thread 1
   Hello from thread 6

GPU offloading example
~~~~~~~~~~~~~~~~~~~~~~

PyOMP supports GPU programming through OpenMP's ``target`` directive for device offloading.
Currently, NVIDIA GPUs are supported (AMD and Intel support are in development).

This example parallelizes a vector addition operation using the GPU:

.. code-block:: python
   :linenos:

   from numba.openmp import njit
   from numba.openmp import openmp_context as openmp
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

The ``target teams distribute parallel for`` directive offloads the loop to the GPU.
The directive automatically distributes loop iterations across GPU teams (thread-blocks)
and threads to maximize available parallelism.

Expected output:

.. code-block:: bash

   c = [3. 3. 3. ... 3. 3. 3.]
