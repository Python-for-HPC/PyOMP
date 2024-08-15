from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_num_threads
from numba.openmp import omp_get_thread_num

@njit
def piFunc(NumSteps):
    step = 1.0/NumSteps
    sum  = 0.0
    start_time = omp_get_wtime()
    with openmp("target "):
         with openmp("loop private(x) reduction(+:sum)"):
               for i in range(NumSteps):
                    x = (i+0.5)*step
                    sum += 4.0/(1.0 + x*x)

    pi = step * sum
    print("pi = ", pi, "runtime = ", runtime = omp_get_wtime() - start_time)
    return pi

pi = piFunc(10000000)
print(pi)
