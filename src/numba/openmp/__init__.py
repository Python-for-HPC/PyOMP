import llvmlite.binding as ll
import sys
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
from .decorators import jit, njit  # noqa: F401
from .offloading import (  # noqa: F401
    find_device_ids,
    get_device_type,
    get_device_vendor,
    get_device_arch,
    print_device_info,
    print_offloading_info,
)


def _init_runtimes():
    """
    Initialize the OpenMP runtimes by loading the wrapper library that links
    with libomp and libomptarget, and calling the libomptarget initialization
    functions.
    """
    sys_platform = sys.platform

    # Find the wrapper library, which links with libomp and libomptarget, and
    # loads them in the correct order with global visibility. Ensures RTLD_NEXT
    # works correctly for libomp to find libomptarget symbols.
    lib_ext = "dylib" if sys_platform == "darwin" else "so"
    wrapper_lib = libpath / "openmp" / "lib" / f"libpyomp_loader.{lib_ext}"

    if not wrapper_lib.exists():
        raise RuntimeError(
            f"OpenMP loader wrapper not found at {wrapper_lib}. "
            "Ensure the package was built correctly."
        )

    if DEBUG_OPENMP >= 1:
        print("Loading OpenMP runtimes via wrapper at", wrapper_lib)

    # Load the wrapper.
    ll.load_library_permanently(str(wrapper_lib))

    # Initialize the OpenMP target runtime.
    from ctypes import CFUNCTYPE

    try:
        addr = ll.address_of_symbol("__tgt_rtl_init")
        if addr:
            cfunctype = CFUNCTYPE(None)
            cfunc = cfunctype(addr)
            cfunc()

        addr = ll.address_of_symbol("__tgt_init_all_rtls")
        if addr:
            cfunc = cfunctype(addr)
            cfunc()

    except Exception as e:
        if DEBUG_OPENMP >= 1:
            print(f"Warning: Failed to initialize OpenMP target runtime: {e}")


def _init_offloading_info():
    """
    Iterate over all OpenMP devices and query their info using the __tgt_get_device_info.
    """
    from .offloading import add_device_info

    num_devices = omp_get_num_devices()

    try:
        addr = ll.address_of_symbol("__tgt_get_device_info")
        if not addr:
            raise RuntimeError(
                "Symbol __tgt_get_device_info not found in OpenMP runtime"
            )
        from ctypes import (
            CFUNCTYPE,
            c_void_p,
            c_size_t,
            c_int,
            string_at,
        )

        copy_callback_ctype = CFUNCTYPE(None, c_void_p, c_size_t)
        cfunctype = CFUNCTYPE(c_int, c_int, copy_callback_ctype)
        __tgt_get_device_info = cfunctype(addr)

        for i in range(num_devices):
            out = bytearray()

            def _copy_cb(ptr, size):
                out.extend(string_at(ptr, size))

            copy_cb = copy_callback_ctype(_copy_cb)
            __tgt_get_device_info(i, copy_cb)

            info_str = out.decode()
            add_device_info(i, info_str)

    except Exception as e:
        print(f"Warning: Failed to initialize offloading info: {e}")


def _init():
    _init_runtimes()
    _init_offloading_info()


_init()

openmp_context = _OpenmpContextType()
