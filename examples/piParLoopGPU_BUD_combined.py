from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime

@njit
def piFunc(NumSteps):
    step = 1.0/NumSteps
    sum  = 0.0
    start_time = omp_get_wtime()  
    with openmp("target teams distribute parallel for private(x) reduction(+:sum)"):
               for i in range(NumSteps):
                    x = (i+0.5)*step
                    sum += 4.0/(1.0 + x*x)

    pi = step * sum
    runtime = omp_get_wtime() - start_time
    print("pi = ", pi, "runtime = ", runtime)
    return pi

pi = piFunc(10000000)
print(pi)
