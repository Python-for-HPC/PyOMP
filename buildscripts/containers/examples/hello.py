from numba.openmp import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_thread_num, omp_get_num_threads


@njit
def hello():
    with openmp("parallel"):
        print("Hello thread", omp_get_thread_num(), " of ", omp_get_num_threads())


hello()
