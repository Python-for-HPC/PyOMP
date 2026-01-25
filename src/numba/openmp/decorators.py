import warnings
import numba

from .compiler import (
    CustomCompiler,
    CustomFunctionCompiler,
)


def jit(*args, **kws):
    """
    Equivalent to jit(nopython=True, nogil=True)
    """
    if "nopython" in kws:
        warnings.warn("nopython is set for njit and is ignored", RuntimeWarning)
    if "forceobj" in kws:
        warnings.warn("forceobj is set for njit and is ignored", RuntimeWarning)
        del kws["forceobj"]
    kws.update({"nopython": True, "nogil": True})
    dispatcher = numba.jit(*args, **kws)
    dispatcher._compiler.__class__ = CustomFunctionCompiler
    dispatcher._compiler.pipeline_class = CustomCompiler
    return dispatcher


def njit(*args, **kws):
    return jit(*args, **kws)
