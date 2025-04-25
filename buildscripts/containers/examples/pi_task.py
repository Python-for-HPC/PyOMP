from numba.openmp import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import (
    omp_set_num_threads,
    omp_get_thread_num,
    omp_get_num_threads,
    omp_get_wtime,
)
import numpy as np


@njit
def pi_comp(Nstart, Nfinish, step):
    MIN_BLK = 256
    # MIN_BLK = 1024*1024*256
    pi_sum = 0.0
    if Nfinish - Nstart < MIN_BLK:
        for i in range(Nstart, Nfinish):
            x = (i + 0.5) * step
            pi_sum += 4.0 / (1.0 + x * x)
    else:
        iblk = Nfinish - Nstart
        pi_sum1 = 0.0
        pi_sum2 = 0.0
        cut = Nfinish - (iblk // 2)
        with openmp("task shared(pi_sum1)"):
            pi_sum1 = pi_comp(Nstart, cut, step)
        with openmp("task shared(pi_sum2)"):
            pi_sum2 = pi_comp(cut, Nfinish, step)
        with openmp("taskwait"):
            pi_sum = pi_sum1 + pi_sum2
    #        pi_sum1 = pi_comp(Nstart, cut, step)
    #        pi_sum2 = pi_comp(cut, Nfinish, step)
    #        pi_sum = pi_sum1 + pi_sum2
    return pi_sum


@njit
def f1(lb, num_steps):
    step = 1.0 / num_steps
    MAX_THREADS = 4
    tsum = np.zeros(MAX_THREADS)

    for j in range(1, MAX_THREADS + 1):
        omp_set_num_threads(j)
        full_sum = 0.0
        start_time = omp_get_wtime()

        with openmp("parallel"):
            with openmp("single"):
                print("num_threads = ", omp_get_num_threads())
                full_sum = pi_comp(lb, num_steps, step)

        pi = step * full_sum
        runtime = omp_get_wtime() - start_time
        print("pi = ", pi, "runtime = ", runtime, j)


lb = 0
num_steps = 1024
# num_steps = 1024*1024*1024
# num_steps = 1000000000
f1(lb, num_steps)
print("DONE")
