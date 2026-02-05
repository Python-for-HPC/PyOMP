from numba.openmp import njit
from numba.openmp import openmp_context as openmp
import numpy as np


@njit
def simple(n, a, b):
    with openmp("parallel for"):
        for i in range(1, n):
            b[i] = (a[i] + a[i - 1]) / 2.0


a = np.ones(100)
b = np.empty(len(a))
b[0] = 0
simple(len(a), a, b)
print(b)
