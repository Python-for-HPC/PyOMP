from numba.openmp import njit
from numba.openmp import openmp_context as openmp


@njit
def calc_pi():
    num_steps = 100000
    step = 1.0 / num_steps

    the_sum = 0.0
    with openmp("parallel for reduction(+:the_sum) schedule(static)"):
        for j in range(num_steps):
            x = ((j - 1) - 0.5) * step
            the_sum += 4.0 / (1.0 + x * x)

    pi = step * the_sum
    return pi


print("pi =", calc_pi())
