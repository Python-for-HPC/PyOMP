import numba
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_set_num_threads, omp_get_thread_num, omp_get_num_threads, omp_get_wtime
import numpy as np

@numba.njit
def simple(n, a, b):
    with openmp("parallel for"):
        for i in range(1, n):
            b[i] = (a[i] + a[i-1]) / 2.0

a = np.ones(100)
b = np.empty(len(a))
b[0] = 0
simple(len(a), a, b)
print(b)
