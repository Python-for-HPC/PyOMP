import os
import tempfile
from functools import lru_cache
from pathlib import Path

# Python 3.12+ removed distutils; use the shim in setuptools.
try:
    from setuptools._distutils import ccompiler, sysconfig
except Exception:  # Python <3.12, or older setuptools
    from distutils import ccompiler, sysconfig  # type: ignore


def link_shared_library(obj_path, out_path):
    # Generate trampolines for numba/NRT symbols. We use trampolines to link the
    # absolute symbol addresses from numba to the self-contained shared library
    # for the OpenMP target CPU module.
    # TODO: ask numba upstream to provide a static library with these symbols.
    @lru_cache
    def generate_trampolines():
        from numba import _helperlib
        from numba.core.runtime import _nrt_python as _nrt

        # Signature mapping for numba/NRT functions. Add more as needed.
        SIGNATURES = {
            # GIL management
            "numba_gil_ensure": ("void", []),
            "numba_gil_release": ("void", []),
            # Memory allocation
            "NRT_MemInfo_alloc": ("void*", ["size_t"]),
            "NRT_MemInfo_alloc_safe": ("void*", ["size_t"]),
            "NRT_MemInfo_alloc_aligned": ("void*", ["size_t", "size_t"]),
            "NRT_MemInfo_alloc_safe_aligned": ("void*", ["size_t", "size_t"]),
            "NRT_MemInfo_free": ("void", ["void*"]),
            # Helperlib
            "numba_unpickle": ("void*", ["void*", "int", "void*"]),
        }

        trampoline_c = """#include <stddef.h>"""

        symbols = []
        # Process _helperlib symbols
        for py_name in _helperlib.c_helpers:
            c_name = "numba_" + py_name
            c_address = _helperlib.c_helpers[py_name]

            if c_name in SIGNATURES:
                ret_type, params = SIGNATURES[c_name]
                symbols.append((c_name, c_address, ret_type, params))

        # Process _nrt symbols
        for py_name in _nrt.c_helpers:
            if py_name.startswith("_"):
                c_name = py_name
            else:
                c_name = "NRT_" + py_name
            c_address = _nrt.c_helpers[py_name]

            if c_name in SIGNATURES:
                ret_type, params = SIGNATURES[c_name]
                symbols.append((c_name, c_address, ret_type, params))

        # Generate trampolines
        for c_name, c_address, ret_type, params in sorted(symbols):
            # Build parameter list
            if not params:
                param_list = "void"
                arg_list = ""
            else:
                param_list = ", ".join(
                    f"{ptype} arg{i}" for i, ptype in enumerate(params)
                )
                arg_list = ", ".join(f"arg{i}" for i in range(len(params)))

            # Build function pointer type
            func_ptr_type = f"{ret_type} (*)({', '.join(params) if params else 'void'})"

            # Generate the trampoline
            trampoline_c += f"""
    __attribute__((visibility("default")))
    {ret_type} {c_name}({param_list}) {{
        {"" if ret_type == "void" else "return "}(({func_ptr_type})0x{c_address:x})({arg_list});
    }}
    """

        return trampoline_c

    """
    Produce a shared library from a single object file and link numba C symbols.
    Uses distutils' compiler.
    """
    obj_path = str(Path(obj_path))
    out_path = str(Path(out_path))

    trampoline_code = generate_trampolines()
    fd, trampoline_c = tempfile.mkstemp(".c")
    os.close(fd)
    with open(trampoline_c, "w") as f:
        f.write(trampoline_code)

    cc = ccompiler.new_compiler()
    sysconfig.customize_compiler(cc)
    extra_pre = []
    extra_post = []

    try:
        trampoline_o = cc.compile([trampoline_c])
    except Exception as e:
        raise RuntimeError(
            f"Compilation failed for trampolines in {trampoline_c}"
        ) from e
    finally:
        os.remove(trampoline_c)

    objs = [obj_path] + trampoline_o
    try:
        cc.link_shared_object(
            objects=objs,
            output_filename=out_path,
            extra_preargs=extra_pre,
            extra_postargs=extra_post,
        )
    except Exception as e:
        raise RuntimeError(f"Link failed for {out_path}") from e
    finally:
        for file_o in trampoline_o:
            os.remove(file_o)
