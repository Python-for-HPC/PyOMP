import os
import warnings
from numba.core import config
from pathlib import Path

libpath = Path(__file__).absolute().parent / "libs"


def _safe_readenv(name, ctor, default):
    value = os.environ.get(name, default)
    try:
        return ctor(value)
    except Exception:
        warnings.warn(
            "environ %s defined but failed to parse '%s'" % (name, value),
            RuntimeWarning,
        )
        return default


DEBUG_OPENMP = _safe_readenv("NUMBA_DEBUG_OPENMP", int, 0)
if DEBUG_OPENMP > 0 and config.DEBUG_ARRAY_OPT == 0:
    config.DEBUG_ARRAY_OPT = 1
DEBUG_OPENMP_LLVM_PASS = _safe_readenv("NUMBA_DEBUG_OPENMP_LLVM_PASS", int, 0)
OPENMP_DISABLED = _safe_readenv("NUMBA_OPENMP_DISABLED", int, 0)
OPENMP_DEVICE_TOOLCHAIN = _safe_readenv("NUMBA_OPENMP_DEVICE_TOOLCHAIN", int, 0)
