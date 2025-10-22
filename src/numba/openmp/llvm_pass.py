import ctypes
import sys
import llvmlite.binding as ll


from .config import libpath, DEBUG_OPENMP_LLVM_PASS


def run_intrinsics_openmp_pass(ll_module):
    libpass = (
        libpath
        / "pass"
        / f"libIntrinsicsOpenMP.{'dylib' if sys.platform == 'darwin' else 'so'}"
    )

    # Roundtrip the LLVM module through the intrinsics OpenMP pass.
    WRITE_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t)

    out = bytearray()

    def _writer_cb(ptr, size):
        out.extend(ctypes.string_at(ptr, size))

    writer_cb = WRITE_CB(_writer_cb)

    lib = ctypes.CDLL(str(libpass))
    lib.runIntrinsicsOpenMPPass.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        WRITE_CB,
    ]
    lib.runIntrinsicsOpenMPPass.restype = ctypes.c_int

    bc = ll_module.as_bitcode()
    buf = ctypes.create_string_buffer(bc)
    ptr = ctypes.cast(buf, ctypes.c_void_p)
    rc = lib.runIntrinsicsOpenMPPass(ptr, len(bc), writer_cb)
    if rc != 0:
        raise RuntimeError(f"Running IntrinsicsOpenMPPass failed with return code {rc}")

    bc_out = bytes(out)
    lowered_module = ll.parse_bitcode(bc_out)
    if DEBUG_OPENMP_LLVM_PASS >= 1:
        print(lowered_module)

    return lowered_module
