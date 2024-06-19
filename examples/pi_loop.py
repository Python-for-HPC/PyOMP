from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_set_num_threads, omp_get_thread_num, omp_get_num_threads, omp_get_wtime
import numpy as np

@njit
def f1():
    num_steps = 100000
    #num_steps = 1000000000
    step = 1.0 / num_steps
    MAX_THREADS=8

    for i in range(1, MAX_THREADS+1):
        the_sum = 0.0
        omp_set_num_threads(i)

        start_time = omp_get_wtime()

        with openmp("parallel private(x)"):
#            with openmp("single"):
#                print("num_threads =", omp_get_num_threads())
#            with openmp("single"):
#                print("got here")

#            with openmp("for reduction(+:the_sum)"):
            with openmp("for reduction(+:the_sum) schedule(static) private(x)"):
                for j in range(num_steps):
#                    o1 = omp_get_thread_num()
#                    o2 = omp_get_num_threads()
#                    print("id and number of threads",j,o1,o2)
#                    print("in range:")
#                    c = step
                    x = ((j-1) - 0.5) * step
                    the_sum += 4.0 / (1.0 + x * x)

        print("the_sum:", the_sum)
        pi = step * the_sum
        runtime = omp_get_wtime() - start_time
        print("pi = ", pi, "runtime = ", runtime, i)

f1()
print("DONE")
