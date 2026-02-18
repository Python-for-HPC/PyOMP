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

Thread and team information
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 65

   * - **omp_get_thread_num()**
     - Returns the unique identifier of the calling thread
   * - **omp_get_num_threads()**
     - Returns the total number of threads in the current parallel region
   * - **omp_set_num_threads(n)**
     - Sets the number of threads for subsequent parallel regions
   * - **omp_get_max_threads()**
     - Returns the maximum number of threads available
   * - **omp_get_num_procs()**
     - Returns the number of processors in the system
   * - **omp_get_thread_limit()**
     - Returns the thread limit for the parallel region
   * - **omp_in_parallel()**
     - Returns 1 if called within a parallel region, 0 otherwise
   * - **omp_get_team_num()**
     - Returns the team number in a target region
   * - **omp_get_num_teams()**
     - Returns the number of teams in a target region

Timing
~~~~~~

.. list-table::
   :widths: 35 65

   * - **omp_get_wtime()**
     - Returns elapsed wall-clock time (useful for performance profiling)

Nested and hierarchical parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 65

   * - **omp_set_nested(flag)**
     - Enables or disables nested parallelism
   * - **omp_set_dynamic(flag)**
     - Enables or disables dynamic thread adjustment
   * - **omp_set_max_active_levels(n)**
     - Sets the maximum number of nested parallel levels
   * - **omp_get_max_active_levels()**
     - Returns the maximum number of nested parallel levels
   * - **omp_get_level()**
     - Returns the current nesting level
   * - **omp_get_active_level()**
     - Returns the current active nesting level
   * - **omp_get_ancestor_thread_num(level)**
     - Returns the thread number at a given nesting level
   * - **omp_get_team_size(level)**
     - Returns the team size at a given nesting level
   * - **omp_get_supported_active_levels()**
     - Returns the supported number of nested active levels

Advanced features
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 65

   * - **omp_get_proc_bind()**
     - Returns the processor binding policy
   * - **omp_get_num_places()**
     - Returns the number of available places
   * - **omp_get_place_num_procs(place)**
     - Returns the number of processors in a place
   * - **omp_get_place_num()**
     - Returns the current place number
   * - **omp_in_final()**
     - Returns 1 if called in a final task, 0 otherwise

Device and target offloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 65

   * - **omp_get_num_devices()**
     - Returns the number of available target devices
   * - **omp_get_device_num()**
     - Returns the device number of the current target device
   * - **omp_set_default_device(device_id)**
     - Sets the default device for subsequent target regions
   * - **omp_get_default_device()**
     - Returns the default device ID for target regions
   * - **omp_is_initial_device()**
     - Returns 1 if executing on the initial device (host), 0 otherwise
   * - **omp_get_initial_device()**
     - Returns the device ID of the initial device (host)

Supported features and platforms
---------------------------------

OpenMP and GPU offloading support
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

Device selection and querying
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyOMP provides utilities in the ``offloading`` module to query available OpenMP target
devices and select specific devices for offloading based on device type, vendor, and
architecture. This enables fine-grained control over where target regions execute.

Discovering Available Devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see all available devices and their properties, use ``print_offloading_info()``:

.. code-block:: python

   from numba.openmp.offloading import print_offloading_info

   print_offloading_info()

This prints information about all devices, including device counts and default device settings.

Finding devices by criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To programmatically find device IDs matching specific criteria, use ``find_device_ids()``:

.. code-block:: python

   from numba.openmp.offloading import find_device_ids

   # Find all GPU devices
   gpu_devices = find_device_ids(type="gpu")

   # Find all NVIDIA GPUs
   nvidia_gpus = find_device_ids(vendor="nvidia")

   # Find NVIDIA GPUs with specific architecture (e.g., sm_80)
   sm80_gpus = find_device_ids(vendor="nvidia", arch="sm_80")

   # Find all AMD GPUs
   amd_gpus = find_device_ids(vendor="amd")

   # Find host/CPU device
   host_devices = find_device_ids(type="host")

The function returns a list of device IDs (integers) matching the criteria. Any parameter
can be ``None`` to act as a wildcard and match all values.

Querying device properties
^^^^^^^^^^^^^^^^^^^^^^^^^^

To determine the type, vendor, or architecture of a specific device ID, use the property
getter functions:

.. code-block:: python

   from numba.openmp.offloading import (
       get_device_type,
       get_device_vendor,
       get_device_arch,
   )

   # Check device type
   dev_type = get_device_type(device_id)  # Returns "gpu", "host", or None

   # Check vendor
   vendor = get_device_vendor(device_id)  # Returns "nvidia", "amd", "host", or None

   # Check architecture
   arch = get_device_arch(device_id)  # Returns architecture string or None

Using device ids in target regions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have identified a device ID, you can use it in OpenMP target directives via the
``device`` clause:

.. code-block:: python

   from numba.openmp import njit, openmp_context as openmp
   from numba.openmp.offloading import find_device_ids
   import numpy as np

   # Find first available NVIDIA GPU
   nvidia_devices = find_device_ids(vendor="nvidia")
   if nvidia_devices:
       device_id = nvidia_devices[0]
   else:
       # Fall back to host if no NVIDIA GPU found
       device_id = find_device_ids(type="host")[0]


   @njit
   def inc(x):
       with openmp(f"target loop device({device_id}) map(tofrom: x)"):
           # Computation runs on specified device
           for i in range(len(x)):
               x[i] = x[i] + 1

       return x


   x = inc(np.ones(10))
   print(f"Result on device {device_id}: {x}")


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

OpenMP parallelism support by platform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

=========== ================ ================= ===================
Platform    CPU              NVIDIA GPU        AMD GPU
=========== ================ ================= ===================
linux-64    ✅ Supported     ✅ Supported      🔶 Work in progress
linux-arm64 ✅ Supported     ✅ Supported      🔶 Work in progress
osx-arm64   ✅ Supported     ❌ Unsupported    ❌ Unsupported
=========== ================ ================= ===================


Platform details
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
