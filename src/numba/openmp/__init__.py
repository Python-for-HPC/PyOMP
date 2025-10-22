import warnings

import llvmlite.binding as ll
import sys
import numba
from ._version import version as __version__  # noqa: F401

from .config import (
    libpath,
    DEBUG_OPENMP,
)
from .omp_runtime import (  # noqa F401
    omp_set_num_threads,
    omp_get_thread_num,
    omp_get_num_threads,
    omp_get_wtime,
    omp_set_dynamic,
    omp_set_nested,
    omp_set_max_active_levels,
    omp_get_max_active_levels,
    omp_get_max_threads,
    omp_get_num_procs,
    omp_in_parallel,
    omp_get_thread_limit,
    omp_get_supported_active_levels,
    omp_get_level,
    omp_get_active_level,
    omp_get_ancestor_thread_num,
    omp_get_team_size,
    omp_in_final,
    omp_get_proc_bind,
    omp_get_num_places,
    omp_get_place_num_procs,
    omp_get_place_num,
    omp_set_default_device,
    omp_get_default_device,
    omp_get_num_devices,
    omp_get_device_num,
    omp_get_team_num,
    omp_get_num_teams,
    omp_is_initial_device,
    omp_get_initial_device,
)

from .compiler import (
    CustomCompiler,
    CustomFunctionCompiler,
)

from .exceptions import (  # noqa: F401
    UnspecifiedVarInDefaultNone,
    ParallelForExtraCode,
    ParallelForWrongLoopCount,
    ParallelForInvalidCollapseCount,
    NonconstantOpenmpSpecification,
    NonStringOpenmpSpecification,
    MultipleNumThreadsClauses,
)
from .overloads import omp_shared_array  # noqa: F401
from .omp_context import _OpenmpContextType


### Decorators.
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


def _init():
    sys_platform = sys.platform

    llvm_major, llvm_minor, llvm_patch = ll.llvm_version_info
    if llvm_major != 14:
        raise RuntimeError(
            f"Incompatible LLVM version {llvm_major}.{llvm_minor}.{llvm_patch}, PyOMP expects LLVM 14.x"
        )

    omplib = (
        libpath
        / "libomp"
        / "lib"
        / f"libomp{'.dylib' if sys_platform == 'darwin' else '.so'}"
    )
    if DEBUG_OPENMP >= 1:
        print("Found OpenMP runtime library at", omplib)
    ll.load_library_permanently(str(omplib))

    # libomptarget is unavailable on apple, windows, so return.
    if sys_platform.startswith("darwin") or sys_platform.startswith("win32"):
        return

    omptargetlib = libpath / "libomp" / "lib" / "libomptarget.so"
    if DEBUG_OPENMP >= 1:
        print("Found OpenMP target runtime library at", omptargetlib)
    ll.load_library_permanently(str(omptargetlib))


_init()

openmp_context = _OpenmpContextType()
