from numba.openmp import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_num_threads, omp_get_thread_num, omp_set_num_threads
import numpy as np


@njit
def test_impl(nt):
    omp_set_num_threads(nt)
    a = np.zeros(4, dtype=np.int64)
    with openmp("teams"):
        with openmp("single"):
            a[0] += 1
        return a


a = test_impl(4)
print("a =", a)
