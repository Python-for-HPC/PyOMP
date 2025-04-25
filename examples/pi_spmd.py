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
def f1():
    num_steps = 100000000
    step = 1.0 / num_steps
    MAX_THREADS = 8
    for j in range(1, MAX_THREADS + 1):
        tsum = np.zeros(j)

        omp_set_num_threads(j)
        start_time = omp_get_wtime()

        with openmp("parallel private(local_sum, tid, numthreads, x)"):
            tid = omp_get_thread_num()
            numthreads = omp_get_num_threads()
            local_sum = 0.0
            if tid == 0:
                print("num_threads = ", numthreads)

            for i in range(tid, num_steps, numthreads):
                x = (i + 0.5) * step
                local_sum += 4.0 / (1.0 + x * x)

            #            print("foo:", j, tid, local_sum)
            tsum[tid] = local_sum

        #        print("tsum:", tsum)
        full_sum = np.sum(tsum)

        pi = step * full_sum
        runtime = omp_get_wtime() - start_time
        print("pi = ", pi, "runtime = ", runtime, j)


f1()
print("DONE")
