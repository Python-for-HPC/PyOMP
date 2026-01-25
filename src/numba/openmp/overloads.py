from numba.extending import overload
from numba.core import types
from numba import cuda as numba_cuda
import numpy as np


from .config import DEBUG_OPENMP


def openmp_copy(a):
    pass  # should always be called through overload


@overload(openmp_copy)
def openmp_copy_overload(a):
    if DEBUG_OPENMP >= 1:
        print("openmp_copy:", a, type(a))
    if isinstance(a, types.npytypes.Array):

        def cimpl(a):
            return np.copy(a)

        return cimpl
    else:

        def cimpl(a):
            return a

        return cimpl


def omp_shared_array(size, dtype):
    return np.empty(size, dtype=dtype)


@overload(omp_shared_array, target="cpu", inline="always", prefer_literal=True)
def omp_shared_array_overload_cpu(size, dtype):
    assert isinstance(size, types.IntegerLiteral)

    def impl(size, dtype):
        return np.empty(size, dtype=dtype)

    return impl


@overload(omp_shared_array, target="cuda", inline="always", prefer_literal=True)
def omp_shared_array_overload_cuda(size, dtype):
    assert isinstance(size, types.IntegerLiteral)

    def impl(size, dtype):
        return numba_cuda.shared.array(size, dtype)

    return impl
