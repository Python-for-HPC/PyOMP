from numba.openmp import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_num_threads, omp_get_thread_num

@njit
def hello():
    with openmp("target device(1)"):
        print("hello thread", omp_get_thread_num(),"of", omp_get_num_threads())

hello()
