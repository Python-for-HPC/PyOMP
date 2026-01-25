from numba.core import types
from numba.core.types.functions import ExternalFunction
from numba.core.datamodel.registry import register_default as model_register
from numba.core.datamodel.models import OpaqueModel


class _OpenmpExternalFunction(types.ExternalFunction):
    def __call__(self, *args):
        import inspect

        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        if mod.__name__.startswith("numba") and not mod.__name__.startswith(
            "numba.openmp.tests"
        ):
            return super(ExternalFunction, self).__call__(*args)

        # Resolve the function address via llvmlite's symbol table so we
        # call the same LLVM-registered symbol the JIT uses. Then wrap
        # it with ctypes CFUNCTYPE to call from Python. This avoids
        # dlopen/dlsym namespace mismatches.
        import llvmlite.binding as ll
        import ctypes

        fname = self.symbol

        addr = ll.address_of_symbol(fname)
        if not addr:
            raise RuntimeError(
                f"symbol {fname} not found via llvmlite.address_of_symbol"
            )

        def numba_to_ctype(tstr):
            if tstr == "int32":
                return ctypes.c_int
            elif tstr == "none":
                return None
            elif tstr == "float64":
                return ctypes.c_double
            else:
                raise RuntimeError(f"unsupported type: {tstr}")

        restype = numba_to_ctype(str(self.sig.return_type))
        argtypes = [numba_to_ctype(str(a)) for a in self.sig.args]

        # CFUNCTYPE requires a valid ctypes restype; None maps to None (void)
        cfunctype = (
            ctypes.CFUNCTYPE(restype, *argtypes)
            if argtypes
            else ctypes.CFUNCTYPE(restype)
        )
        cfunc = cfunctype(addr)
        return cfunc(*args)


model_register(_OpenmpExternalFunction)(OpaqueModel)

omp_set_num_threads = _OpenmpExternalFunction(
    "omp_set_num_threads", types.void(types.int32)
)
omp_get_thread_num = _OpenmpExternalFunction("omp_get_thread_num", types.int32())
omp_get_num_threads = _OpenmpExternalFunction("omp_get_num_threads", types.int32())
omp_get_wtime = _OpenmpExternalFunction("omp_get_wtime", types.float64())
omp_set_dynamic = _OpenmpExternalFunction("omp_set_dynamic", types.void(types.int32))
omp_set_nested = _OpenmpExternalFunction("omp_set_nested", types.void(types.int32))
omp_set_max_active_levels = _OpenmpExternalFunction(
    "omp_set_max_active_levels", types.void(types.int32)
)
omp_get_max_active_levels = _OpenmpExternalFunction(
    "omp_get_max_active_levels", types.int32()
)
omp_get_max_threads = _OpenmpExternalFunction("omp_get_max_threads", types.int32())
omp_get_num_procs = _OpenmpExternalFunction("omp_get_num_procs", types.int32())
omp_in_parallel = _OpenmpExternalFunction("omp_in_parallel", types.int32())
omp_get_thread_limit = _OpenmpExternalFunction("omp_get_thread_limit", types.int32())
omp_get_supported_active_levels = _OpenmpExternalFunction(
    "omp_get_supported_active_levels", types.int32()
)
omp_get_level = _OpenmpExternalFunction("omp_get_level", types.int32())
omp_get_active_level = _OpenmpExternalFunction("omp_get_active_level", types.int32())
omp_get_ancestor_thread_num = _OpenmpExternalFunction(
    "omp_get_ancestor_thread_num", types.int32(types.int32)
)
omp_get_team_size = _OpenmpExternalFunction(
    "omp_get_team_size", types.int32(types.int32)
)
omp_in_final = _OpenmpExternalFunction("omp_in_finale", types.int32())
omp_get_proc_bind = _OpenmpExternalFunction("omp_get_proc_bind", types.int32())
omp_get_num_places = _OpenmpExternalFunction("omp_get_num_places", types.int32())
omp_get_place_num_procs = _OpenmpExternalFunction(
    "omp_get_place_num_procs", types.int32(types.int32)
)
omp_get_place_num = _OpenmpExternalFunction("omp_get_place_num", types.int32())
omp_set_default_device = _OpenmpExternalFunction(
    "omp_set_default_device", types.int32(types.int32)
)
omp_get_default_device = _OpenmpExternalFunction(
    "omp_get_default_device", types.int32()
)
omp_get_num_devices = _OpenmpExternalFunction("omp_get_num_devices", types.int32())
omp_get_device_num = _OpenmpExternalFunction("omp_get_device_num", types.int32())
omp_get_team_num = _OpenmpExternalFunction("omp_get_team_num", types.int32())
omp_get_num_teams = _OpenmpExternalFunction("omp_get_num_teams", types.int32())
omp_is_initial_device = _OpenmpExternalFunction("omp_is_initial_device", types.int32())
omp_get_initial_device = _OpenmpExternalFunction(
    "omp_get_initial_device", types.int32()
)
