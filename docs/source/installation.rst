Installation
============

You can install PyOMP from PyPI using `pip`:

.. code-block:: console

   $ pip install pyomp

It is also possible to install PyOMP through `conda`:

.. code-block:: console

   $ conda install -c python-for-hpc -c conda-forge pyomp

Compatibility
-------------

PyOMP releases are compatible with a specific range of Numba versions. The table
below summarizes the supported Numba versions for each PyOMP release series.
The `x` in versions indicates that all patch levels within that version are
supported.

+--------+---------------------+
| PyOMP  | Numba               |
+========+=====================+
| 0.5.x  | 0.62.x - 0.63.x     |
+--------+---------------------+
| 0.4.x  | 0.61.x              |
+--------+---------------------+
| 0.3.x  | 0.57.x - 0.60.x     |
+--------+---------------------+

Additional options
------------------

Binder and Docker images are provided to try PyOMP without installing locally.

Binder (free hosted JupyterLab):
`Binder <https://mybinder.org/v2/gh/Python-for-HPC/binder/HEAD>`_

Docker (pre-built images):

.. code-block:: console

   $ docker pull ghcr.io/python-for-hpc/pyomp:latest
