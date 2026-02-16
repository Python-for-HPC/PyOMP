OpenMP support
==============

OpenMP directives and clauses
-----------------------------

The following section shows supported OpenMP directives and the support status of
their clauses.

.. note::
   ✅ = supported;
   ❌ = unsupported;
   🔶 = partial support


barrier
~~~~~~~

No clauses.

critical
~~~~~~~~

No clauses.

for
~~~

* ✅ collapse, firstprivate, lastprivate, private
* 🔶 reduction
* ❌ allocate, linear, nowait, order, ordered, schedule

parallel
~~~~~~~~

* ✅ default, firstprivate, if, num_threads, private, shared
* 🔶 reduction
* ❌ allocate, copyin, proc_bind

parallel for
~~~~~~~~~~~~

Combines ``parallel`` and ``for`` directives. See clauses for `for`_ and `parallel`_ above.

single
~~~~~~

* ❌ allocate, copyprivate, firstprivate, nowait, private

task
~~~~

* ✅ default, firstprivate, private, shared
* ❌ affinity, allocate, detach, if, in_reduction, final, mergeable, priority, untied

taskwait
~~~~~~~~

* ❌ depend, nowait

target
~~~~~~

* ✅ device, firstprivate, map, private, thread_limit
* ❌ allocate, defaultmap, depend, has_device_addr, if, in_reduction, is_device_ptr, nowait, uses_allocators

teams
~~~~~

* ✅ default, firstprivate, num_teams, private, shared, thread_limit
* 🔶 reduction

distribute
~~~~~~~~~~

* ✅ firstprivate, lastprivate, private
* ❌ allocate, collapse, dist_schedule, order

teams distribute
~~~~~~~~~~~~~~~~

Combines ``teams`` and ``distribute`` directives. See clauses for `teams`_ and `distribute`_ above.

target teams
~~~~~~~~~~~~

Combines ``target`` and ``teams`` directives. See clauses for `target`_ and `teams`_ above.

target data
~~~~~~~~~~~

* ✅ device, map
* ❌ if, use_device_ptr, use_device_addr

target enter data
~~~~~~~~~~~~~~~~~

* ✅ device, map
* ❌ depend, if, nowait

target exit data
~~~~~~~~~~~~~~~~

Same clauses as `target enter data`_. See above.

target update
~~~~~~~~~~~~~

* ✅ device, from, to
* ❌ depend, if, nowait

target teams distribute
~~~~~~~~~~~~~~~~~~~~~~~

Combines ``target``, ``teams``, and ``distribute`` directives. See clauses for `target`_, `teams`_, and `distribute`_ above.

distribute parallel for
~~~~~~~~~~~~~~~~~~~~~~~

Combines ``distribute`` and ``parallel for`` directives. See clauses for `distribute`_, `parallel`_, and `for`_ above.

target teams distribute parallel for
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combines ``target``, ``teams``, ``distribute``, and ``parallel for`` directives. See clauses for `target`_, `teams`_, `parallel`_, and `for`_ above.


OpenMP runtime functions
-------------------------
**Thread and team information:**

* ``omp_get_thread_num()`` - Returns the unique identifier of the calling thread
* ``omp_get_num_threads()`` - Returns the total number of threads in the current parallel region
* ``omp_set_num_threads(n)`` - Sets the number of threads for subsequent parallel regions
* ``omp_get_max_threads()`` - Returns the maximum number of threads available
* ``omp_get_num_procs()`` - Returns the number of processors in the system
* ``omp_get_thread_limit()`` - Returns the thread limit for the parallel region
* ``omp_in_parallel()`` - Returns 1 if called within a parallel region, 0 otherwise
* ``omp_get_team_num()`` - Returns the team number in a target region
* ``omp_get_num_teams()`` - Returns the number of teams in a target region

**Timing:**

* ``omp_get_wtime()`` - Returns elapsed wall-clock time (useful for performance profiling)

**Nested and hierarchical parallelism:**

* ``omp_set_nested(flag)`` - Enables or disables nested parallelism
* ``omp_set_dynamic(flag)`` - Enables or disables dynamic thread adjustment
* ``omp_set_max_active_levels(n)`` - Sets the maximum number of nested parallel levels
* ``omp_get_max_active_levels()`` - Returns the maximum number of nested parallel levels
* ``omp_get_level()`` - Returns the current nesting level
* ``omp_get_active_level()`` - Returns the current active nesting level
* ``omp_get_ancestor_thread_num(level)`` - Returns the thread number at a given nesting level
* ``omp_get_team_size(level)`` - Returns the team size at a given nesting level
* ``omp_get_supported_active_levels()`` - Returns the supported number of nested active levels

**Advanced features:**

* ``omp_get_proc_bind()`` - Returns the processor binding policy
* ``omp_get_num_places()`` - Returns the number of available places
* ``omp_get_place_num_procs(place)`` - Returns the number of processors in a place
* ``omp_get_place_num()`` - Returns the current place number

Supported features and platforms
---------------------------------

OpenMP and GPU Offloading Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyOMP builds on `Numba <https://numba.pydata.org/>`_ Just-In-Time (JIT)
compilation extensions and leverages LLVM's OpenMP implementation to provide
portable parallel execution. The supported OpenMP features depend on your versions of
LLVM and Numba. For compatibility details, see the `Numba support info
<https://numba.readthedocs.io/en/stable/user/installing.html#numba-support-info>`_
in the Numba documentation.

PyOMP also supports GPU offloading for NVIDIA GPUs. The supported GPU
architectures depend on the LLVM version and its OpenMP runtime. Consult the
LLVM OpenMP documentation for details on your specific version.

Version and platform support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table shows tested combinations of PyOMP, Numba, Python, LLVM, and supported platforms:

.. table::
   :widths: auto

   ===================== ==================== ==================== ============ ================================
   PyOMP                 Numba                Python               LLVM         Supported Platforms
   ===================== ==================== ==================== ============ ================================
   0.5.x                 0.62.x - 0.63.x      3.10 - 3.14          20.x         linux-64, osx-arm64, linux-arm64
   0.4.x                 0.61.x               3.10 - 3.13          15.x         linux-64, osx-arm64, linux-arm64
   0.3.x                 0.57.x - 0.60.x      3.9 - 3.12           14.x         linux-64, osx-arm64, linux-arm64
   ===================== ==================== ==================== ============ ================================

Platform Details
^^^^^^^^^^^^^^^^

* **linux-64**: Linux x86_64 architecture
* **osx-arm64**: macOS ARM64 (Apple Silicon)
* **linux-arm64**: Linux ARM64 architecture
* **GPU offloading**: Available on Linux platforms only (linux-64 and linux-arm64)

Notes
^^^^^

* Python 3.14 free-threaded build (cp314t) is not supported with the current Numba/llvmlite version.
* LLVM version 20.1.8 is used for the current PyOMP 0.5.x releases.
* For GPU offloading support, NVIDIA GPU and NVIDIA driver are required on supported Linux platforms.
* AMD GPU support is in active development.
