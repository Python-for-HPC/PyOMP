import warnings
import numba

from .compiler import CustomCompiler


def jit(*args, **kws):
    """
    Equivalent to jit(nopython=True, nogil=True)
    """
    if "nopython" in kws:
        warnings.warn("nopython is set for njit and is ignored", RuntimeWarning)
    if "forceobj" in kws:
        warnings.warn("forceobj is set for njit and is ignored", RuntimeWarning)
        del kws["forceobj"]
    kws.update({"nopython": True, "nogil": True, "pipeline_class": CustomCompiler})
    dispatcher = numba.jit(*args, **kws)
    return dispatcher


def njit(*args, **kws):
    return jit(*args, **kws)
