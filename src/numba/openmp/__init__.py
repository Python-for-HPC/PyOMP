import warnings
from numba.core.withcontexts import WithContext, _CallContextType
from lark import Lark, Transformer
from lark.exceptions import VisitError
from numba.core.ir_utils import (
    get_call_table,
    dump_blocks,
    dprint_func_ir,
    replace_vars,
    apply_copy_propagate_extensions,
    visit_vars_extensions,
    remove_dels,
    visit_vars_inner,
    visit_vars,
    get_name_var_table,
    replace_var_names,
    get_definition,
    build_definitions,
    dead_code_elimination,
    mk_unique_var,
    find_topo_order,
    flatten_labels,
)
from numba.core.analysis import (
    compute_cfg_from_blocks,
    compute_use_defs,
    compute_live_map,
    _fix_loop_exit,
)
from numba.core import (
    ir,
    config,
    types,
    typeinfer,
    cgutils,
    compiler,
    transforms,
    bytecode,
    typed_passes,
    imputils,
    typing,
    cpu,
    compiler_machinery,
)
from numba.core.compiler_machinery import PassManager
from numba.core.compiler import DefaultPassBuilder
from numba.core.untyped_passes import (
    TranslateByteCode,
    FixupArgs,
    IRProcessing,
    InlineClosureLikes,
    RewriteSemanticConstants,
    DeadBranchPrune,
    GenericRewrites,
    RewriteDynamicRaises,
    MakeFunctionToJitFunction,
    InlineInlinables,
    FindLiterallyCalls,
    LiteralUnroll,
    LiteralPropagationSubPipelinePass,
    WithLifting,
)
from numba import np as numba_np
from numba import cuda as numba_cuda
from numba.core.controlflow import CFGraph
from numba.core.ssa import _run_ssa
from numba.extending import overload, intrinsic
from numba.core.callconv import (
    BaseCallConv,
    MinimalCallConv,
    errcode_t,
    RETCODE_OK,
    Status,
    excinfo_t,
    CPUCallConv,
)
from functools import cached_property, lru_cache
from numba.core.datamodel.registry import register_default as model_register
from numba.core.datamodel.registry import default_manager as model_manager
from numba.core.datamodel.models import OpaqueModel
from numba.core.types.functions import Dispatcher, ExternalFunction
from numba.core.dispatcher import _FunctionCompiler
from numba.np.ufunc import array_exprs
from cffi import FFI
import llvmlite.binding as ll
import llvmlite.ir as lir
import operator
import sys
import copy
import os
import numpy as np
from numba.core.analysis import ir_extension_usedefs, _use_defs_result
from numba.core.lowering import Lower
from numba.core.codegen import AOTCodeLibrary, JITCodeLibrary
from numba.cuda import descriptor as cuda_descriptor, compiler as cuda_compiler
from numba.cuda.target import CUDACallConv
import subprocess
import tempfile
import types as python_types
import numba
import ctypes
from pathlib import Path
from ._version import version as __version__

libpath = Path(__file__).absolute().parent / "libs"

### START OF EXTENSIONS TO AVOID SUBPROCESS TOOLS ###
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


###


###### START OF NUMBA EXTENSIONS ######


### ir_utils.py
def dump_block(label, block):
    print(label, ":")
    for stmt in block.body:
        print("    ", stmt)


###


### analysis.py
def filter_nested_loops(cfg, loops):
    blocks_in_loop = set()
    # get loop bodies
    for loop in loops.values():
        insiders = set(loop.body) | set(loop.entries) | set(loop.exits)
        insiders.discard(loop.header)
        blocks_in_loop |= insiders
    # find loop that is not part of other loops
    for loop in loops.values():
        if loop.header not in blocks_in_loop:
            yield _fix_loop_exit(cfg, loop)


###


### config.py
def safe_readenv(name, ctor, default):
    value = os.environ.get(name, default)
    try:
        return ctor(value)
    except Exception:
        warnings.warn(
            "environ %s defined but failed to parse '%s'" % (name, value),
            RuntimeWarning,
        )
        return default


DEBUG_OPENMP = safe_readenv("NUMBA_DEBUG_OPENMP", int, 0)
if DEBUG_OPENMP > 0 and config.DEBUG_ARRAY_OPT == 0:
    config.DEBUG_ARRAY_OPT = 1
DEBUG_OPENMP_LLVM_PASS = safe_readenv("NUMBA_DEBUG_OPENMP_LLVM_PASS", int, 0)
OPENMP_DISABLED = safe_readenv("NUMBA_OPENMP_DISABLED", int, 0)
OPENMP_DEVICE_TOOLCHAIN = safe_readenv("NUMBA_OPENMP_DEVICE_TOOLCHAIN", int, 0)
###


class LowerNoSROA(Lower):
    @property
    def _disable_sroa_like_opt(self):
        # Always return True for this instance
        return True

    def lower_assign_inst(self, orig, inst):
        # This fixes assignments for Arg instructions when the target is a
        # CPointer. It sets the backing storage to the pointer of the argument
        # itself.
        if isinstance(self.context, OpenmpCPUTargetContext) or isinstance(
            self.context, OpenmpCUDATargetContext
        ):
            value = inst.value
            if isinstance(value, ir.Arg):
                argname = value.name
                argty = self.typeof("arg." + argname)
                if isinstance(argty, types.CPointer):
                    llty = self.context.get_value_type(argty)
                    ptr = lir.values.Argument(self.module, llty, "arg." + argname)
                    self.varmap[value.name] = ptr
                    return

        return orig(self, inst)

    def lower_return_inst(self, orig, inst):
        if isinstance(self.context, OpenmpCUDATargetContext):
            # This fixes Return instructions for CUDA device functions in an
            # OpenMP target region. It avoids setting a value to the return
            # value pointer argument, which otherwise breaks OpenMP code
            # generation (looks like an upstream miscompilation) by DCE any
            # memory effects (e.g., to other pointer arguments from a tofrom
            # mapping.)
            if self.fndesc.qualname == self.context.device_func_name:
                self.call_conv._return_errcode_raw(self.builder, RETCODE_OK)
                return
        return orig(self, inst)


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


class CustomCPUCodeLibrary(JITCodeLibrary):
    def add_llvm_module(self, ll_module):
        lowered_module = run_intrinsics_openmp_pass(ll_module)
        super().add_llvm_module(lowered_module)

    def _finalize_specific(self):
        super()._finalize_specific()
        ll.ExecutionEngine.run_static_constructors(self._codegen._engine._ee)


class CustomAOTCPUCodeLibrary(AOTCodeLibrary):
    def add_llvm_module(self, ll_module):
        lowered_module = run_intrinsics_openmp_pass(ll_module)
        super().add_llvm_module(lowered_module)


class CustomFunctionCompiler(_FunctionCompiler):
    def _customize_flags(self, flags):
        # We need to disable SSA form for OpenMP analysis to detect variables
        # used within regions.
        flags.enable_ssa = False
        return flags


class CustomCompiler(compiler.CompilerBase):
    @staticmethod
    def custom_untyped_pipeline(state, name="untyped-openmp"):
        """Returns an untyped part of the nopython OpenMP pipeline"""
        pm = PassManager(name)
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")

        # inline closures early in case they are using nonlocal's
        # see issue #6585.
        pm.add_pass(InlineClosureLikes, "inline calls to locally defined closures")

        # pre typing
        if not state.flags.no_rewrites:
            pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
            pm.add_pass(DeadBranchPrune, "dead branch pruning")
            pm.add_pass(GenericRewrites, "nopython rewrites")

        pm.add_pass(RewriteDynamicRaises, "rewrite dynamic raises")

        # convert any remaining closures into functions
        pm.add_pass(
            MakeFunctionToJitFunction, "convert make_function into JIT functions"
        )
        # inline functions that have been determined as inlinable and rerun
        # branch pruning, this needs to be run after closures are inlined as
        # the IR repr of a closure masks call sites if an inlinable is called
        # inside a closure
        pm.add_pass(InlineInlinables, "inline inlinable functions")
        if not state.flags.no_rewrites:
            pm.add_pass(DeadBranchPrune, "dead branch pruning")

        pm.add_pass(FindLiterallyCalls, "find literally calls")
        pm.add_pass(LiteralUnroll, "handles literal_unroll")

        if state.flags.enable_ssa:
            assert False, "SSA form is not supported in OpenMP"

        pm.add_pass(LiteralPropagationSubPipelinePass, "Literal propagation")
        # Run WithLifting late to for make_implicit_explicit to work.  TODO: We
        # should create a pass that does this instead of replicating and hacking
        # the untyped pipeline. This handling may also negatively affect
        # optimizations.
        pm.add_pass(WithLifting, "Handle with contexts")

        pm.finalize()
        return pm

    def define_pipelines(self):
        # compose pipeline from untyped, typed and lowering parts
        dpb = DefaultPassBuilder
        pm = PassManager("omp")
        untyped_passes = self.custom_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = dpb.define_nopython_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return [pm]


class CustomContext(cpu.CPUContext):
    def post_lowering(self, mod, library):
        if hasattr(library, "openmp") and library.openmp:
            post_lowering_openmp(mod)
            super().post_lowering(mod, library)


### decorators


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


class OpenmpCUDATargetContext(cuda_descriptor.CUDATargetContext):
    def __init__(self, name, typingctx, target="cuda"):
        super().__init__(typingctx, target)
        self.device_func_name = name

    def post_lowering(self, mod, library):
        if hasattr(library, "openmp") and library.openmp:
            post_lowering_openmp(mod)
            super().post_lowering(mod, library)

    @cached_property
    def call_conv(self):
        return CUDACallConv(self)


class OpenmpCPUTargetContext(CustomContext):
    def __init__(self, name, typingctx, target="cpu"):
        super().__init__(typingctx, target)
        self.device_func_name = name


##### END OF NUMBA EXTENSIONS ######


###### START OF LLVMLITE EXTENSIONS ######
def get_decl(alloca):
    if not isinstance(alloca, lir.instructions.AllocaInstr):
        raise TypeError("Expected AllocaInstr, got %s" % type(alloca))
    return '{0} %"{1}"'.format(alloca.type, alloca._get_name())


# TODO: Upstream to llvmlite, it's part of the langref.
class TokenType(lir.Type):
    """
    The type for tokens.  From the LLVM Language Reference.

      'The token type is used when a value is associated with an
       instruction but all uses of the value must not attempt to
       introspect or obscure it. As such, it is not appropriate
       to have a phi or select of type token.'
    """

    def _to_string(self):
        return "token"

    def __eq__(self, other):
        return isinstance(other, TokenType)

    def __hash__(self):
        return hash(TokenType)


class CallInstrWithOperandBundle(lir.instructions.CallInstr):
    def set_tags(self, tags):
        self.tags = tags

    # TODO: This is ugly duplication, we should upstream to llvmlite.
    def descr(self, buf, add_metadata=True):
        def descr_arg(i, a):
            if i in self.arg_attributes:
                attrs = " ".join(self.arg_attributes[i]._to_list()) + " "
            else:
                attrs = ""
            return "{0} {1}{2}".format(a.type, attrs, a.get_reference())

        args = ", ".join([descr_arg(i, a) for i, a in enumerate(self.args)])

        fnty = self.callee.function_type
        # Only print function type if variable-argument
        if fnty.var_arg:
            ty = fnty
        # Otherwise, just print the return type.
        else:
            # Fastmath flag work only in this case
            ty = fnty.return_type
        callee_ref = "{0} {1}".format(ty, self.callee.get_reference())
        if self.cconv:
            callee_ref = "{0} {1}".format(self.cconv, callee_ref)

        tail_marker = ""
        if self.tail:
            tail_marker = "{0} ".format(self.tail)

        buf.append(
            "{tail}{op}{fastmath} {callee}({args}){attr}{tags}{meta}\n".format(
                tail=tail_marker,
                op=self.opname,
                fastmath="".join([" " + attr for attr in self.fastmath]),
                callee=callee_ref,
                args=args,
                attr="".join([" " + attr for attr in self.attributes]),
                tags=(" " + self.tags if self.tags is not None else ""),
                meta=(
                    self._stringify_metadata(leading_comma=True) if add_metadata else ""
                ),
            )
        )


###### END OF LLVMLITE EXTENSIONS ######


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


# ----------------------------------------------------------------------------------------------


class NameSlice:
    def __init__(self, name, the_slice):
        self.name = name
        self.the_slice = the_slice

    def __str__(self):
        return "NameSlice(" + str(self.name) + "," + str(self.the_slice) + ")"


class StringLiteral:
    def __init__(self, x):
        self.x = x


@intrinsic
def get_itercount(typingctx, it):
    if isinstance(it, types.RangeIteratorType):
        sig = typing.signature(it.yield_type, it)

        def codegen(context, builder, signature, args):
            assert len(args) == 1
            val = args[0]
            pair = context.make_helper(builder, it, val)
            return builder.load(pair.count)

        return sig, codegen


def remove_privatized(x):
    if isinstance(x, ir.Var):
        x = x.name

    if isinstance(x, str) and x.endswith("%privatized"):
        return x[: len(x) - len("%privatized")]
    else:
        return x


def remove_all_privatized(x):
    new_x = None
    while new_x != x:
        new_x = x
        x = remove_privatized(new_x)

    return new_x


def typemap_lookup(typemap, x):
    orig_x = x
    if isinstance(x, ir.Var):
        x = x.name

    while True:
        if x in typemap:
            return typemap[x]
        new_x = remove_privatized(x)
        if new_x == x:
            break
        else:
            x = new_x

    tkeys = typemap.keys()

    # Get basename (without privatized)
    x = remove_all_privatized(x)

    potential_keys = list(filter(lambda y: y.startswith(x), tkeys))

    for pkey in potential_keys:
        pkey_base = remove_all_privatized(pkey)
        if pkey_base == x:
            return typemap[pkey]

    raise KeyError(f"{orig_x} and all of its non-privatized names not found in typemap")


class openmp_tag(object):
    def __init__(self, name, arg=None, load=False, non_arg=False, omp_slice=None):
        self.name = name
        self.arg = arg
        self.load = load
        self.loaded_arg = None
        self.xarginfo = []
        self.non_arg = non_arg
        self.omp_slice = omp_slice

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.arg, lir.instructions.AllocaInstr):
            del state["arg"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "arg"):
            self.arg = None

    def var_in(self, var):
        assert isinstance(var, str)

        if isinstance(self.arg, ir.Var):
            return self.arg.name == var

        if isinstance(self.arg, str):
            return self.arg == var

        return False

    def arg_size(self, x, lowerer):
        if DEBUG_OPENMP >= 2:
            print("arg_size:", x, type(x))
        if isinstance(x, NameSlice):
            x = x.name
        if isinstance(x, ir.Var):
            # Make sure the var referred to has been alloc'ed already.
            lowerer._alloca_var(x.name, lowerer.fndesc.typemap[x.name])
            if self.load:
                assert False
            else:
                arg_str = lowerer.getvar(x.name)
                return lowerer.context.get_abi_sizeof(arg_str.type.pointee)
        elif isinstance(x, lir.instructions.AllocaInstr):
            return lowerer.context.get_abi_sizeof(x.type.pointee)
        elif isinstance(x, str):
            xtyp = lowerer.fndesc.typemap[x]
            if DEBUG_OPENMP >= 1:
                print("xtyp:", xtyp, type(xtyp))
            lowerer._alloca_var(x, xtyp)
            if self.load:
                assert False
            else:
                arg_str = lowerer.getvar(x)
                return lowerer.context.get_abi_sizeof(arg_str.type.pointee)
        elif isinstance(x, int):
            assert False
        else:
            print("unknown arg type:", x, type(x))
            assert False

    def arg_to_str(
        self, x, lowerer, struct_lower=False, var_table=None, gen_copy=False
    ):
        if DEBUG_OPENMP >= 1:
            print("arg_to_str:", x, type(x), self.load, type(self.load))
        if struct_lower:
            assert isinstance(x, str)
            assert var_table is not None

        typemap = lowerer.fndesc.typemap

        if isinstance(x, NameSlice):
            if DEBUG_OPENMP >= 2:
                print("nameslice found:", x)
            x = x.name
        if isinstance(x, ir.Var):
            # Make sure the var referred to has been alloc'ed already.
            lowerer._alloca_var(x.name, typemap_lookup(typemap, x))
            if self.load:
                if not self.loaded_arg:
                    self.loaded_arg = lowerer.loadvar(x.name)
                lop = self.loaded_arg.operands[0]
                loptype = lop.type
                pointee = loptype.pointee
                ref = self.loaded_arg._get_reference()
                decl = str(pointee) + " " + ref
            else:
                arg_str = lowerer.getvar(x.name)
                if isinstance(arg_str, lir.values.Argument):
                    decl = str(arg_str)
                else:
                    decl = get_decl(arg_str)
        elif isinstance(x, lir.instructions.AllocaInstr):
            decl = get_decl(x)
        elif isinstance(x, str):
            if "*" in x:
                xsplit = x.split("*")
                assert len(xsplit) == 2
                # xtyp = get_dotted_type(x, typemap, lowerer)
                xtyp = typemap_lookup(typemap, xsplit[0])
                if DEBUG_OPENMP >= 1:
                    print("xtyp:", xtyp, type(xtyp))
                lowerer._alloca_var(x, xtyp)
                if self.load:
                    if not self.loaded_arg:
                        self.loaded_arg = lowerer.loadvar(x)
                    lop = self.loaded_arg.operands[0]
                    loptype = lop.type
                    pointee = loptype.pointee
                    ref = self.loaded_arg._get_reference()
                    decl = str(pointee) + " " + ref
                    assert len(xsplit) == 1
                else:
                    arg_str = lowerer.getvar(xsplit[0])
                    # arg_str = lowerer.getvar(x)
                    if isinstance(arg_str, lir.Argument):
                        decl = str(arg_str)
                    else:
                        decl = get_decl(arg_str)
                    if len(xsplit) > 1:
                        cur_typ = xtyp
                        field_indices = []
                        for field in xsplit[1:]:
                            dm = lowerer.context.data_model_manager.lookup(cur_typ)
                            findex = dm._fields.index(field)
                            field_indices.append("i32 " + str(findex))
                            cur_typ = dm._members[findex]
                        fi_str = ",".join(field_indices)
                        decl += f", {fi_str}"
                        # decl = f"SCOPE({decl}, {fi_str})"
            else:
                xtyp = typemap_lookup(typemap, x)
                if DEBUG_OPENMP >= 1:
                    print("xtyp:", xtyp, type(xtyp))
                lowerer._alloca_var(x, xtyp)
                if self.load:
                    if not self.loaded_arg:
                        self.loaded_arg = lowerer.loadvar(x)
                    lop = self.loaded_arg.operands[0]
                    loptype = lop.type
                    pointee = loptype.pointee
                    ref = self.loaded_arg._get_reference()
                    decl = str(pointee) + " " + ref
                else:
                    arg_str = lowerer.getvar(x)
                    if isinstance(arg_str, lir.values.Argument):
                        decl = str(arg_str)
                    elif isinstance(arg_str, lir.instructions.AllocaInstr):
                        decl = get_decl(arg_str)
                    else:
                        assert False, (
                            f"Don't know how to get decl string for variable {arg_str} of type {type(arg_str)}"
                        )

                if struct_lower and isinstance(xtyp, types.npytypes.Array):
                    dm = lowerer.context.data_model_manager.lookup(xtyp)
                    cur_tag_ndim = xtyp.ndim
                    stride_typ = lowerer.context.get_value_type(
                        types.intp
                    )  # lir.Type.int(64)
                    stride_abi_size = lowerer.context.get_abi_sizeof(stride_typ)
                    array_var = var_table[self.arg]
                    if DEBUG_OPENMP >= 1:
                        print(
                            "Found array mapped:",
                            self.name,
                            self.arg,
                            xtyp,
                            type(xtyp),
                            stride_typ,
                            type(stride_typ),
                            stride_abi_size,
                            array_var,
                            type(array_var),
                        )
                    size_var = ir.Var(None, self.arg + "_size_var", array_var.loc)
                    # size_var = array_var.scope.redefine("size_var", array_var.loc)
                    size_getattr = ir.Expr.getattr(array_var, "size", array_var.loc)
                    size_assign = ir.Assign(size_getattr, size_var, array_var.loc)
                    typemap[size_var.name] = types.int64
                    lowerer._alloca_var(size_var.name, typemap[size_var.name])
                    lowerer.lower_inst(size_assign)
                    data_field = dm._fields.index("data")
                    shape_field = dm._fields.index("shape")
                    strides_field = dm._fields.index("strides")
                    size_lowered = get_decl(lowerer.getvar(size_var.name))
                    fixed_size = cur_tag_ndim
                    # fixed_size = stride_abi_size * cur_tag_ndim
                    decl += f", i32 {data_field}, i64 0, {size_lowered}"
                    decl += f", i32 {shape_field}, i64 0, i64 {fixed_size}"
                    decl += f", i32 {strides_field}, i64 0, i64 {fixed_size}"

                    # see core/datamodel/models.py
                    # struct_tags.append(openmp_tag(cur_tag.name, cur_tag.arg + "*data", non_arg=True, omp_slice=(0,lowerer.loadvar(size_var.name))))
                    # struct_tags.append(openmp_tag(cur_tag.name, cur_tag.arg + "*shape", non_arg=True, omp_slice=(0,stride_abi_size * cur_tag_ndim)))
                    # struct_tags.append(openmp_tag(cur_tag.name, cur_tag.arg + "*strides", non_arg=True, omp_slice=(0,stride_abi_size * cur_tag_ndim)))

                if gen_copy and isinstance(xtyp, types.npytypes.Array):
                    native_np_copy, copy_cres = create_native_np_copy(xtyp)
                    lowerer.library.add_llvm_module(copy_cres.library._final_module)
                    nnclen = len(native_np_copy)
                    decl += f', [{nnclen} x i8] c"{native_np_copy}"'
        elif isinstance(x, StringLiteral):
            decl = str(cgutils.make_bytearray(x.x))
        elif isinstance(x, int):
            decl = "i32 " + str(x)
        else:
            print("unknown arg type:", x, type(x))

        if self.omp_slice is not None:

            def handle_var(x):
                if isinstance(x, ir.Var):
                    loaded_size = lowerer.loadvar(x.name)
                    loaded_op = loaded_size.operands[0]
                    loaded_pointee = loaded_op.type.pointee
                    ret = str(loaded_pointee) + " " + loaded_size._get_reference()
                else:
                    ret = "i64 " + str(x)
                return ret

            start_slice = handle_var(self.omp_slice[0])
            end_slice = handle_var(self.omp_slice[1])
            decl += f", {start_slice}, {end_slice}"
            # decl = f"SLICE({decl}, {self.omp_slice[0]}, {self.omp_slice[1]})"

        return decl

    def post_entry(self, lowerer):
        for xarginfo, xarginfo_args, x, alloca_tuple_list in self.xarginfo:
            loaded_args = [
                lowerer.builder.load(alloca_tuple[2])
                for alloca_tuple in alloca_tuple_list
            ]
            fa_res = xarginfo.from_arguments(lowerer.builder, tuple(loaded_args))
            # fa_res = xarginfo.from_arguments(lowerer.builder,tuple([xarg for xarg in xarginfo_args]))
            assert len(fa_res) == 1
            lowerer.storevar(fa_res[0], x)

    def add_length_firstprivate(self, x, lowerer):
        if self.name == "QUAL.OMP.FIRSTPRIVATE":
            return [x]
            # return [x, self.arg_size(x, lowerer)]
            # return [x, lowerer.context.get_constant(types.uintp, self.arg_size(x, lowerer))]
        else:
            return [x]

    def unpack_arg(self, x, lowerer, xarginfo_list):
        if isinstance(x, ir.Var):
            return self.add_length_firstprivate(x, lowerer), None
        elif isinstance(x, lir.instructions.AllocaInstr):
            return self.add_length_firstprivate(x, lowerer), None
        elif isinstance(x, str):
            xtyp = lowerer.fndesc.typemap[x]
            if DEBUG_OPENMP >= 2:
                print("xtyp:", xtyp, type(xtyp))
            if self.load:
                return self.add_length_firstprivate(x, lowerer), None
            else:
                names_to_unpack = []
                # names_to_unpack = ["QUAL.OMP.FIRSTPRIVATE"]
                # names_to_unpack = ["QUAL.OMP.PRIVATE", "QUAL.OMP.FIRSTPRIVATE"]
                if (
                    isinstance(xtyp, types.npytypes.Array)
                    and self.name in names_to_unpack
                ):
                    # from core/datamodel/packer.py
                    xarginfo = lowerer.context.get_arg_packer((xtyp,))
                    xloaded = lowerer.loadvar(x)
                    xarginfo_args = list(
                        xarginfo.as_arguments(lowerer.builder, [xloaded])
                    )
                    xarg_alloca_vars = []
                    for xarg in xarginfo_args:
                        if DEBUG_OPENMP >= 2:
                            print(
                                "xarg:",
                                type(xarg),
                                xarg,
                                "agg:",
                                xarg.aggregate,
                                type(xarg.aggregate),
                                "ind:",
                                xarg.indices,
                            )
                            print(xarg.aggregate.type.elements[xarg.indices[0]])
                        alloca_name = "$alloca_" + xarg.name
                        alloca_typ = xarg.aggregate.type.elements[xarg.indices[0]]
                        alloca_res = lowerer.alloca_lltype(alloca_name, alloca_typ)
                        if DEBUG_OPENMP >= 2:
                            print(
                                "alloca:",
                                alloca_name,
                                alloca_typ,
                                alloca_res,
                                alloca_res.get_reference(),
                            )
                        xarg_alloca_vars.append((alloca_name, alloca_typ, alloca_res))
                        lowerer.builder.store(xarg, alloca_res)
                    xarginfo_list.append((xarginfo, xarginfo_args, x, xarg_alloca_vars))
                    rets = []
                    for i, xarg in enumerate(xarg_alloca_vars):
                        rets.append(xarg[2])
                        if i == 4:
                            alloca_name = "$alloca_total_size_" + str(x)
                            if DEBUG_OPENMP >= 2:
                                print("alloca_name:", alloca_name)
                            alloca_typ = lowerer.context.get_value_type(
                                types.intp
                            )  # lir.Type.int(64)
                            alloca_res = lowerer.alloca_lltype(alloca_name, alloca_typ)
                            if DEBUG_OPENMP >= 2:
                                print(
                                    "alloca:",
                                    alloca_name,
                                    alloca_typ,
                                    alloca_res,
                                    alloca_res.get_reference(),
                                )
                            mul_res = lowerer.builder.mul(
                                lowerer.builder.load(xarg_alloca_vars[2][2]),
                                lowerer.builder.load(xarg_alloca_vars[3][2]),
                            )
                            lowerer.builder.store(mul_res, alloca_res)
                            rets.append(alloca_res)
                        else:
                            rets.append(self.arg_size(xarg[2], lowerer))
                    return rets, [x]
                else:
                    return self.add_length_firstprivate(x, lowerer), None
        elif isinstance(x, int):
            return self.add_length_firstprivate(x, lowerer), None
        else:
            print("unknown arg type:", x, type(x))

        return self.add_length_firstprivate(x, lowerer), None

    def unpack_arrays(self, lowerer):
        if isinstance(self.arg, list):
            arg_list = self.arg
        elif self.arg is not None:
            arg_list = [self.arg]
        else:
            return [self]
        new_xarginfo = []
        unpack_res = [self.unpack_arg(arg, lowerer, new_xarginfo) for arg in arg_list]
        new_args = [x[0] for x in unpack_res]
        arrays_to_private = []
        for x in unpack_res:
            if x[1]:
                arrays_to_private.append(x[1])
        ot_res = openmp_tag(self.name, sum(new_args, []), self.load)
        ot_res.xarginfo = new_xarginfo
        return [ot_res] + (
            []
            if len(arrays_to_private) == 0
            else [openmp_tag("QUAL.OMP.PRIVATE", sum(arrays_to_private, []), self.load)]
        )

    def lower(self, lowerer, debug):
        decl = ""
        if debug and DEBUG_OPENMP >= 1:
            print("openmp_tag::lower", self.name, self.arg, type(self.arg))

        if isinstance(self.arg, list):
            arg_list = self.arg
        elif self.arg is not None:
            arg_list = [self.arg]
        else:
            arg_list = []
        typemap = lowerer.fndesc.typemap
        assert len(arg_list) <= 1

        if self.name == "QUAL.OMP.TARGET.IMPLICIT":
            assert False  # shouldn't get here anymore

        name_to_use = self.name

        is_array = self.arg in typemap and isinstance(
            typemap[self.arg], types.npytypes.Array
        )

        gen_copy = name_to_use in ["QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.LASTPRIVATE"]

        if (
            name_to_use
            in [
                "QUAL.OMP.MAP.TOFROM",
                "QUAL.OMP.MAP.TO",
                "QUAL.OMP.MAP.FROM",
                "QUAL.OMP.MAP.ALLOC",
            ]
            and is_array
        ):
            # name_to_use += ".STRUCT"
            # var_table = get_name_var_table(lowerer.func_ir.blocks)
            # decl = ",".join([self.arg_to_str(x, lowerer, struct_lower=True, var_table=var_table) for x in arg_list])
            decl = ",".join(
                [
                    self.arg_to_str(x, lowerer, struct_lower=False, gen_copy=gen_copy)
                    for x in arg_list
                ]
            )
        else:
            decl = ",".join(
                [
                    self.arg_to_str(x, lowerer, struct_lower=False, gen_copy=gen_copy)
                    for x in arg_list
                ]
            )

        return '"' + name_to_use + '"(' + decl + ")"

    def replace_vars_inner(self, var_dict):
        if isinstance(self.arg, ir.Var):
            self.arg = replace_vars_inner(self.arg, var_dict)

    def add_to_usedef_set(self, use_set, def_set, start):
        assert start == True or start == False
        if DEBUG_OPENMP >= 3:
            print("add_to_usedef_set", start, self.name, "is_dsa=", is_dsa(self.name))

        def add_arg(arg, the_set):
            if isinstance(self.arg, ir.Var):
                the_set.add(self.arg.name)
            elif isinstance(self.arg, str):
                the_set.add(self.arg)
            elif isinstance(self.arg, NameSlice):
                assert isinstance(self.arg.name, str), "Expected str in NameSlice arg"
                the_set.add(self.arg.name)
            # TODO: Create a good error check mechanism.
            # else: ?

        if self.name.startswith("DIR.OMP"):
            assert not isinstance(self.arg, (ir.Var, str))
            return

        if self.name in [
            "QUAL.OMP.MAP.TO",
            "QUAL.OMP.IF",
            "QUAL.OMP.NUM_THREADS",
            "QUAL.OMP.NUM_TEAMS",
            "QUAL.OMP.THREAD_LIMIT",
            "QUAL.OMP.SCHEDULE.STATIC",
            "QUAL.OMP.SCHEDULE.RUNTIME",
            "QUAL.OMP.SCHEDULE.GUIDED",
            "QUAL.OMP.SCHEDULE_DYNAMIC",
            "QUAL.OMP.FIRSTPRIVATE",
            "QUAL.OMP.COPYIN",
            "QUAL.OMP.COPYPRIVATE",
            "QUAL.OMP.NORMALIZED.LB",
            "QUAL.OMP.NORMALIZED.START",
            "QUAL.OMP.NORMALIZED.UB",
            "QUAL.OMP.MAP.TO.STRUCT",
        ]:
            if start:
                add_arg(self.arg, use_set)
        elif self.name in [
            "QUAL.OMP.PRIVATE",
            "QUAL.OMP.LINEAR",
            "QUAL.OMP.NORMALIZED.IV",
            "QUAL.OMP.MAP.ALLOC",
            "QUAL.OMP.MAP.ALLOC.STRUCT",
        ]:
            # Intentionally do nothing.
            pass
        elif self.name in ["QUAL.OMP.SHARED"]:
            add_arg(self.arg, use_set)
        elif self.name in [
            "QUAL.OMP.MAP.TOFROM",
            "QUAL.OMP.TARGET.IMPLICIT",
            "QUAL.OMP.MAP.TOFROM.STRUCT",
        ]:
            if start:
                add_arg(self.arg, use_set)
            else:
                add_arg(self.arg, use_set)
                add_arg(self.arg, def_set)
        elif self.name in [
            "QUAL.OMP.MAP.FROM",
            "QUAL.OMP.LASTPRIVATE",
            "QUAL.OMP.MAP.FROM.STRUCT",
        ] or self.name.startswith("QUAL.OMP.REDUCTION"):
            if not start:
                add_arg(self.arg, use_set)
                add_arg(self.arg, def_set)
        else:
            # All other clauses should not have a variable argument.
            if isinstance(self.arg, (ir.Var, str)):
                print("Bad usedef tag:", self.name, self.arg)
            assert not isinstance(self.arg, (ir.Var, str))

    def __str__(self):
        return (
            "openmp_tag("
            + str(self.name)
            + ","
            + str(self.arg)
            + (
                ""
                if self.omp_slice is None
                else f", omp_slice({self.omp_slice[0]},{self.omp_slice[1]})"
            )
            + ")"
        )

    def __repr__(self):
        return self.__str__()


def openmp_tag_list_to_str(tag_list, lowerer, debug):
    tag_strs = [x.lower(lowerer, debug) for x in tag_list]
    return "[ " + ", ".join(tag_strs) + " ]"


def list_vars_from_tags(tags):
    used_vars = []
    for t in tags:
        if isinstance(t.arg, ir.Var):
            used_vars.append(t.arg)
    return used_vars


def openmp_region_alloca(obj, alloca_instr, typ):
    obj.alloca(alloca_instr, typ)


def push_alloca_callback(lowerer, callback, data, builder):
    # cgutils.push_alloca_callbacks(callback, data)
    if not hasattr(builder, "_lowerer_push_alloca_callbacks"):
        builder._lowerer_push_alloca_callbacks = 0
    builder._lowerer_push_alloca_callbacks += 1


def pop_alloca_callback(lowerer, builder):
    # cgutils.pop_alloca_callbacks()
    builder._lowerer_push_alloca_callbacks -= 1


def in_openmp_region(builder):
    if hasattr(builder, "_lowerer_push_alloca_callbacks"):
        return builder._lowerer_push_alloca_callbacks > 0
    else:
        return False


def find_target_start_end(func_ir, target_num):
    start_block = None
    end_block = None

    for label, block in func_ir.blocks.items():
        if isinstance(block.body[0], openmp_region_start):
            block_target_num = block.body[0].has_target()
            if target_num == block_target_num:
                start_block = label
                if start_block is not None and end_block is not None:
                    return start_block, end_block
        elif isinstance(block.body[0], openmp_region_end):
            block_target_num = block.body[0].start_region.has_target()
            if target_num == block_target_num:
                end_block = label
                if start_block is not None and end_block is not None:
                    return start_block, end_block

    dprint_func_ir(func_ir, "find_target_start_end")
    print("target_num:", target_num)
    assert False


def get_tags_of_type(clauses, ctype):
    ret = []
    for c in clauses:
        if c.name == ctype:
            ret.append(c)
    return ret


def copy_one(x, calltypes):
    if DEBUG_OPENMP >= 2:
        print("copy_one:", x, type(x))
    if isinstance(x, ir.Loc):
        return copy.copy(x)
    elif isinstance(x, ir.Expr):
        if x in calltypes:
            ctyp = calltypes[x]
        else:
            ctyp = None
        ret = ir.Expr(
            copy_one(x.op, calltypes),
            copy_one(x.loc, calltypes),
            **copy_one(x._kws, calltypes),
        )
        if ctyp and ret not in calltypes:
            calltypes[ret] = ctyp
        return ret
    elif isinstance(x, dict):
        return {k: copy_one(v, calltypes) for k, v in x.items()}
    elif isinstance(x, list):
        return [copy_one(v, calltypes) for v in x]
    elif isinstance(x, tuple):
        return tuple([copy_one(v, calltypes) for v in x])
    elif isinstance(x, ir.Const):
        return ir.Const(
            copy_one(x.value, calltypes), copy_one(x.loc, calltypes), x.use_literal_type
        )
    elif isinstance(
        x,
        (
            int,
            float,
            str,
            ir.Global,
            python_types.BuiltinFunctionType,
            ir.UndefinedType,
            type(None),
            types.functions.ExternalFunction,
        ),
    ):
        return x
    elif isinstance(x, ir.Var):
        return ir.Var(x.scope, copy_one(x.name, calltypes), copy_one(x.loc, calltypes))
    elif isinstance(x, ir.Del):
        return ir.Del(copy_one(x.value, calltypes), copy_one(x.loc, calltypes))
    elif isinstance(x, ir.Jump):
        return ir.Jump(copy_one(x.target, calltypes), copy_one(x.loc, calltypes))
    elif isinstance(x, ir.Return):
        return ir.Return(copy_one(x.value, calltypes), copy_one(x.loc, calltypes))
    elif isinstance(x, ir.Branch):
        return ir.Branch(
            copy_one(x.cond, calltypes),
            copy_one(x.truebr, calltypes),
            copy_one(x.falsebr, calltypes),
            copy_one(x.loc, calltypes),
        )
    elif isinstance(x, ir.Print):
        ctyp = calltypes[x]
        ret = copy.copy(x)
        calltypes[ret] = ctyp
        return ret
    elif isinstance(x, ir.Assign):
        return ir.Assign(
            copy_one(x.value, calltypes),
            copy_one(x.target, calltypes),
            copy_one(x.loc, calltypes),
        )
    elif isinstance(x, ir.Arg):
        return ir.Arg(
            copy_one(x.name, calltypes),
            copy_one(x.index, calltypes),
            copy_one(x.loc, calltypes),
        )
    elif isinstance(x, ir.SetItem):
        ctyp = calltypes[x]
        ret = ir.SetItem(
            copy_one(x.target, calltypes),
            copy_one(x.index, calltypes),
            copy_one(x.value, calltypes),
            copy_one(x.loc, calltypes),
        )
        calltypes[ret] = ctyp
        return ret
    elif isinstance(x, ir.StaticSetItem):
        ctyp = calltypes[x]
        ret = ir.StaticSetItem(
            copy_one(x.target, calltypes),
            copy_one(x.index, calltypes),
            copy_one(x.index_var, calltypes),
            copy_one(x.value, calltypes),
            copy_one(x.loc, calltypes),
        )
        calltypes[ret] = ctyp
        return ret
    elif isinstance(x, ir.FreeVar):
        return ir.FreeVar(
            copy_one(x.index, calltypes),
            copy_one(x.name, calltypes),
            copy_one(x.value, calltypes),
            copy_one(x.loc, calltypes),
        )
    elif isinstance(x, slice):
        return slice(
            copy_one(x.start, calltypes),
            copy_one(x.stop, calltypes),
            copy_one(x.step, calltypes),
        )
    elif isinstance(x, ir.PopBlock):
        return ir.PopBlock(copy_one(x.loc, calltypes))
    elif isinstance(x, ir.SetAttr):
        ctyp = calltypes[x]
        ret = ir.SetAttr(
            copy_one(x.target, calltypes),
            copy_one(x.attr, calltypes),
            copy_one(x.value, calltypes),
            copy_one(x.loc, calltypes),
        )
        calltypes[ret] = ctyp
        return ret
    elif isinstance(x, ir.DelAttr):
        return ir.DelAttr(
            copy_one(x.target, calltypes),
            copy_one(x.attr, calltypes),
            copy_one(x.loc, calltypes),
        )
    elif isinstance(x, types.Type):
        return x  # Don't copy types.
    print("Failed to handle the following type when copying target IR.", type(x), x)
    assert False


def copy_ir(input_ir, calltypes, depth=1):
    assert depth >= 0 and depth <= 1

    # This is a depth 0 copy.
    cur_ir = input_ir.copy()
    if depth == 1:
        for blk in cur_ir.blocks.values():
            for i in range(len(blk.body)):
                if not isinstance(
                    blk.body[i], (openmp_region_start, openmp_region_end)
                ):
                    blk.body[i] = copy_one(blk.body[i], calltypes)

    return cur_ir


def is_target_tag(x):
    ret = x.startswith("DIR.OMP.TARGET") and x not in [
        "DIR.OMP.TARGET.DATA",
        "DIR.OMP.TARGET.ENTER.DATA",
        "DIR.OMP.TARGET.EXIT.DATA",
    ]
    return ret


def replace_np_empty_with_cuda_shared(
    outlined_ir, typemap, calltypes, prefix, typingctx
):
    if DEBUG_OPENMP >= 2:
        print("starting replace_np_empty_with_cuda_shared")
    outlined_ir = outlined_ir.blocks
    converted_arrays = []
    consts = {}
    topo_order = find_topo_order(outlined_ir)
    mode = 0  # 0 = non-target region, 1 = target region, 2 = teams region, 3 = teams parallel region
    # For each block in topological order...
    for label in topo_order:
        block = outlined_ir[label]
        new_block_body = []
        blen = len(block.body)
        index = 0
        # For each statement in the block.
        while index < blen:
            stmt = block.body[index]
            # Adjust mode based on the start of an openmp region.
            if isinstance(stmt, openmp_region_start):
                if "TARGET" in stmt.tags[0].name:
                    assert mode == 0
                    mode = 1
                if "TEAMS" in stmt.tags[0].name and mode == 1:
                    mode = 2
                if "PARALLEL" in stmt.tags[0].name and mode == 2:
                    mode = 3
                new_block_body.append(stmt)
            # Adjust mode based on the end of an openmp region.
            elif isinstance(stmt, openmp_region_end):
                if mode == 3 and "PARALLEL" in stmt.tags[0].name:
                    mode = 2
                if mode == 2 and "TEAMS" in stmt.tags[0].name:
                    mode = 1
                if mode == 1 and "TARGET" in stmt.tags[0].name:
                    mode = 0
                new_block_body.append(stmt)
            # Fix calltype for the np.empty call to have literal as first
            # arg and include explicit dtype.
            elif (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Expr)
                and stmt.value.op == "call"
                and stmt.value.func in converted_arrays
            ):
                size = consts[stmt.value.args[0].name]
                # The 1D case where the dimension size is directly a const.
                if isinstance(size, ir.Const):
                    size = size.value
                    signature = calltypes[stmt.value]
                    signature_args = (
                        types.scalars.IntegerLiteral(size),
                        types.functions.NumberClass(signature.return_type.dtype),
                    )
                    del calltypes[stmt.value]
                    calltypes[stmt.value] = typing.templates.Signature(
                        signature.return_type, signature_args, signature.recvr
                    )
                # The 2D+ case where the dimension sizes are in a tuple.
                elif isinstance(size, ir.Expr):
                    signature = calltypes[stmt.value]
                    signature_args = (
                        types.Tuple(
                            [
                                types.scalars.IntegerLiteral(consts[x.name].value)
                                for x in size.items
                            ]
                        ),
                        types.functions.NumberClass(signature.return_type.dtype),
                    )
                    del calltypes[stmt.value]
                    calltypes[stmt.value] = typing.templates.Signature(
                        signature.return_type, signature_args, signature.recvr
                    )

                # These lines will force the function to be in the data structures that lowering uses.
                afnty = typemap[stmt.value.func.name]
                afnty.get_call_type(typingctx, signature_args, {})
                if len(stmt.value.args) == 1:
                    dtype_to_use = signature.return_type.dtype
                    # If dtype in kwargs then remove it.
                    if len(stmt.value.kws) > 0:
                        for kwarg in stmt.value.kws:
                            if kwarg[0] == "dtype":
                                stmt.value.kws = list(
                                    filter(lambda x: x[0] != "dtype", stmt.value.kws)
                                )
                                break
                    new_block_body.append(
                        ir.Assign(
                            ir.Global("np", np, lhs.loc),
                            ir.Var(lhs.scope, mk_unique_var(".np_global"), lhs.loc),
                            lhs.loc,
                        )
                    )
                    typemap[new_block_body[-1].target.name] = types.Module(np)
                    new_block_body.append(
                        ir.Assign(
                            ir.Expr.getattr(
                                new_block_body[-1].target, str(dtype_to_use), lhs.loc
                            ),
                            ir.Var(lhs.scope, mk_unique_var(".np_dtype"), lhs.loc),
                            lhs.loc,
                        )
                    )
                    typemap[new_block_body[-1].target.name] = (
                        types.functions.NumberClass(signature.return_type.dtype)
                    )
                    stmt.value.args.append(new_block_body[-1].target)
                else:
                    raise NotImplementedError(
                        "np.empty having more than shape and dtype arguments not yet supported."
                    )
                new_block_body.append(stmt)
            # Keep track of variables assigned from consts or from build_tuples make up exclusively of
            # variables assigned from consts.
            elif isinstance(stmt, ir.Assign) and (
                isinstance(stmt.value, ir.Const)
                or (
                    isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "build_tuple"
                    and all([x.name in consts for x in stmt.value.items])
                )
            ):
                consts[stmt.target.name] = stmt.value
                new_block_body.append(stmt)
            # If we see a global for the numpy module.
            elif (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Global)
                and isinstance(stmt.value.value, python_types.ModuleType)
                and stmt.value.value.__name__ == "numpy"
            ):
                lhs = stmt.target
                index += 1
                next_stmt = block.body[index]
                # And the next statement is a getattr for the name "empty" on the numpy module
                # and we are in a target region.
                if (
                    isinstance(next_stmt, ir.Assign)
                    and isinstance(next_stmt.value, ir.Expr)
                    and next_stmt.value.value == lhs
                    and next_stmt.value.op == "getattr"
                    and next_stmt.value.attr == "empty"
                    and mode > 0
                ):
                    # Remember that we are converting this np.empty into a CUDA call.
                    converted_arrays.append(next_stmt.target)

                    # Create numba.cuda module variable.
                    new_block_body.append(
                        ir.Assign(
                            ir.Global("numba", numba, lhs.loc),
                            ir.Var(
                                lhs.scope, mk_unique_var(".cuda_shared_global"), lhs.loc
                            ),
                            lhs.loc,
                        )
                    )
                    typemap[new_block_body[-1].target.name] = types.Module(numba)
                    new_block_body.append(
                        ir.Assign(
                            ir.Expr.getattr(new_block_body[-1].target, "cuda", lhs.loc),
                            ir.Var(
                                lhs.scope,
                                mk_unique_var(".cuda_shared_getattr"),
                                lhs.loc,
                            ),
                            lhs.loc,
                        )
                    )
                    typemap[new_block_body[-1].target.name] = types.Module(numba.cuda)

                    if mode == 1:
                        raise NotImplementedError(
                            "np.empty used in non-teams or parallel target region"
                        )
                        pass
                    elif mode == 2:
                        # Create numba.cuda.shared module variable.
                        new_block_body.append(
                            ir.Assign(
                                ir.Expr.getattr(
                                    new_block_body[-1].target, "shared", lhs.loc
                                ),
                                ir.Var(
                                    lhs.scope,
                                    mk_unique_var(".cuda_shared_getattr"),
                                    lhs.loc,
                                ),
                                lhs.loc,
                            )
                        )
                        typemap[new_block_body[-1].target.name] = types.Module(
                            numba.cuda.stubs.shared
                        )
                    elif mode == 3:
                        # Create numba.cuda.local module variable.
                        new_block_body.append(
                            ir.Assign(
                                ir.Expr.getattr(
                                    new_block_body[-1].target, "local", lhs.loc
                                ),
                                ir.Var(
                                    lhs.scope,
                                    mk_unique_var(".cuda_local_getattr"),
                                    lhs.loc,
                                ),
                                lhs.loc,
                            )
                        )
                        typemap[new_block_body[-1].target.name] = types.Module(
                            numba.cuda.stubs.local
                        )

                    # Change the typemap for the original function variable for np.empty.
                    afnty = typingctx.resolve_getattr(
                        typemap[new_block_body[-1].target.name], "array"
                    )
                    del typemap[next_stmt.target.name]
                    typemap[next_stmt.target.name] = afnty
                    # Change the variable that previously was assigned np.empty to now be one of
                    # the CUDA array allocators.
                    new_block_body.append(
                        ir.Assign(
                            ir.Expr.getattr(
                                new_block_body[-1].target, "array", lhs.loc
                            ),
                            next_stmt.target,
                            lhs.loc,
                        )
                    )
                else:
                    new_block_body.append(stmt)
                    new_block_body.append(next_stmt)
            else:
                new_block_body.append(stmt)
            index += 1
        block.body = new_block_body


class openmp_region_start(ir.Stmt):
    def __init__(self, tags, region_number, loc, firstprivate_dead_after=None):
        if DEBUG_OPENMP >= 2:
            print("region ids openmp_region_start::__init__", id(self))
        self.tags = tags
        self.region_number = region_number
        self.loc = loc
        self.omp_region_var = None
        self.omp_metadata = None
        self.tag_vars = set()
        self.normal_iv = None
        self.target_copy = False
        self.firstprivate_dead_after = (
            [] if firstprivate_dead_after is None else firstprivate_dead_after
        )
        for tag in self.tags:
            if isinstance(tag.arg, ir.Var):
                self.tag_vars.add(tag.arg.name)
            elif isinstance(tag.arg, str):
                self.tag_vars.add(tag.arg)
            elif isinstance(tag.arg, NameSlice):
                self.tag_vars.add(tag.arg.name)

            if tag.name == "QUAL.OMP.NORMALIZED.IV":
                self.normal_iv = tag.arg
        if DEBUG_OPENMP >= 1:
            print("tags:", self.tags)
            print("tag_vars:", sorted(self.tag_vars))
        self.acq_res = False
        self.acq_rel = False
        self.alloca_queue = []
        self.end_region = None

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def replace_var_names(self, namedict):
        for i in range(len(self.tags)):
            if isinstance(self.tags[i].arg, ir.Var):
                if self.tags[i].arg.name in namedict:
                    var = self.tags[i].arg
                    self.tags[i].arg = ir.Var(var.scope, namedict[var.name], var.log)
            elif isinstance(self.tags[i].arg, str):
                if "*" in self.tags[i].arg:
                    xsplit = self.tags[i].arg.split("*")
                    assert len(xsplit) == 2
                    if xsplit[0] in namedict:
                        self.tags[i].arg = namedict[xsplit[0]] + "*" + xsplit[1]
                else:
                    if self.tags[i].arg in namedict:
                        self.tags[i].arg = namedict[self.tags[i].arg]

    def add_tag(self, tag):
        tag_arg_str = None
        if isinstance(tag.arg, ir.Var):
            tag_arg_str = tag.arg.name
        elif isinstance(tag.arg, str):
            tag_arg_str = tag.arg
        elif isinstance(tag.arg, lir.instructions.AllocaInstr):
            tag_arg_str = tag.arg._get_name()
        else:
            assert False
        if isinstance(tag_arg_str, str):
            self.tag_vars.add(tag_arg_str)
        self.tags.append(tag)

    def get_var_dsa(self, var):
        assert isinstance(var, str)
        for tag in self.tags:
            if is_dsa(tag.name) and tag.var_in(var):
                return tag.name
        return None

    def requires_acquire_release(self):
        pass
        # self.acq_res = True

    def requires_combined_acquire_release(self):
        pass
        # self.acq_rel = True

    def has_target(self):
        for t in self.tags:
            if is_target_tag(t.name):
                return t.arg
        return None

    def list_vars(self):
        return list_vars_from_tags(self.tags)

    def update_tags(self):
        with self.builder.goto_block(self.block):
            cur_instr = -1

            while True:
                last_instr = self.builder.block.instructions[cur_instr]
                if (
                    isinstance(last_instr, lir.instructions.CallInstr)
                    and last_instr.tags is not None
                    and len(last_instr.tags) > 0
                ):
                    break
                cur_instr -= 1

            last_instr.tags = openmp_tag_list_to_str(self.tags, self.lowerer, False)
            if DEBUG_OPENMP >= 1:
                print("last_tags:", last_instr.tags, type(last_instr.tags))

    def alloca(self, alloca_instr, typ):
        # We can't process these right away since the processing required can
        # lead to infinite recursion.  So, we just accumulate them in a queue
        # and then process them later at the end_region marker so that the
        # variables are guaranteed to exist in their full form so that when we
        # process them then they won't lead to infinite recursion.
        self.alloca_queue.append((alloca_instr, typ))

    def process_alloca_queue(self):
        # This should be old code...making sure with the assertion.
        assert len(self.alloca_queue) == 0
        has_update = False
        for alloca_instr, typ in self.alloca_queue:
            has_update = self.process_one_alloca(alloca_instr, typ) or has_update
        if has_update:
            self.update_tags()
        self.alloca_queue = []

    def post_lowering_process_alloca_queue(self, enter_directive):
        has_update = False
        if DEBUG_OPENMP >= 1:
            print("starting post_lowering_process_alloca_queue")
        for alloca_instr, typ in self.alloca_queue:
            has_update = self.process_one_alloca(alloca_instr, typ) or has_update
        if has_update:
            if DEBUG_OPENMP >= 1:
                print(
                    "post_lowering_process_alloca_queue has update:",
                    enter_directive.tags,
                )
            enter_directive.tags = openmp_tag_list_to_str(
                self.tags, self.lowerer, False
            )
            # LLVM IR is doing some string caching and the following line is necessary to
            # reset that caching so that the original tag text can be overwritten above.
            enter_directive._clear_string_cache()
            if DEBUG_OPENMP >= 1:
                print(
                    "post_lowering_process_alloca_queue updated tags:",
                    enter_directive.tags,
                )
        self.alloca_queue = []

    def process_one_alloca(self, alloca_instr, typ):
        avar = alloca_instr.name
        if DEBUG_OPENMP >= 1:
            print(
                "openmp_region_start process_one_alloca:",
                id(self),
                alloca_instr,
                avar,
                typ,
                type(alloca_instr),
                self.tag_vars,
            )

        has_update = False
        if (
            self.normal_iv is not None
            and avar != self.normal_iv
            and avar.startswith(self.normal_iv)
        ):
            for i in range(len(self.tags)):
                if DEBUG_OPENMP >= 1:
                    print("Replacing normalized iv with", avar)
                self.tags[i].arg = avar
                has_update = True
                break

        if not self.needs_implicit_vars():
            return has_update
        if avar not in self.tag_vars:
            if DEBUG_OPENMP >= 1:
                print(
                    f"LLVM variable {avar} didn't previously exist in the list of vars so adding as private."
                )
            self.add_tag(
                openmp_tag("QUAL.OMP.PRIVATE", alloca_instr)
            )  # is FIRSTPRIVATE right here?
            has_update = True
        return has_update

    def needs_implicit_vars(self):
        first_tag = self.tags[0]
        if (
            first_tag.name == "DIR.OMP.PARALLEL"
            or first_tag.name == "DIR.OMP.PARALLEL.LOOP"
            or first_tag.name == "DIR.OMP.TASK"
        ):
            return True
        return False

    def update_context(self, context, builder):
        cctyp = type(context.call_conv)
        # print("start update_context id(context)", id(context), "id(const.call_conv)", id(context.call_conv), "cctyp", cctyp, "id(cctyp)", id(cctyp))

        if (
            not hasattr(cctyp, "pyomp_patch_installed")
            or cctyp.pyomp_patch_installed == False
        ):
            cctyp.pyomp_patch_installed = True
            # print("update_context", "id(cctyp.return_user_exec)", id(cctyp.return_user_exc), "id(context)", id(context))
            setattr(cctyp, "orig_return_user_exc", cctyp.return_user_exc)

            def pyomp_return_user_exc(self, builder, *args, **kwargs):
                # print("pyomp_return_user_exc")
                # Handle exceptions in OpenMP regions by emitting a trap and an
                # unreachable terminator.
                if in_openmp_region(builder):
                    fnty = lir.types.FunctionType(lir.types.VoidType(), [])
                    fn = builder.module.declare_intrinsic("llvm.trap", (), fnty)
                    builder.call(fn, [])
                    builder.unreachable()
                    return
                self.orig_return_user_exc(builder, *args, **kwargs)

            setattr(cctyp, "return_user_exc", pyomp_return_user_exc)
            # print("after", id(pyomp_return_user_exc), id(cctyp.return_user_exc))

            setattr(
                cctyp, "orig_return_status_propagate", cctyp.return_status_propagate
            )

            def pyomp_return_status_propagate(self, builder, *args, **kwargs):
                if in_openmp_region(builder):
                    return
                self.orig_return_status_propagate(builder, *args, **kwargs)

            setattr(cctyp, "return_status_propagate", pyomp_return_status_propagate)

        cemtyp = type(context.error_model)
        # print("start update_context id(context)", id(context), "id(const.error_model)", id(context.error_model), "cemtyp", cemtyp, "id(cemtyp)", id(cemtyp))

        if (
            not hasattr(cemtyp, "pyomp_patch_installed")
            or cemtyp.pyomp_patch_installed == False
        ):
            cemtyp.pyomp_patch_installed = True
            # print("update_context", "id(cemtyp.return_user_exec)", id(cemtyp.fp_zero_division), "id(context)", id(context))
            setattr(cemtyp, "orig_fp_zero_division", cemtyp.fp_zero_division)

            def pyomp_fp_zero_division(self, builder, *args, **kwargs):
                # print("pyomp_fp_zero_division")
                if in_openmp_region(builder):
                    return False
                return self.orig_fp_zero_division(builder, *args, **kwargs)

            setattr(cemtyp, "fp_zero_division", pyomp_fp_zero_division)
            # print("after", id(pyomp_fp_zero_division), id(cemtyp.fp_zero_division))

        pyapi = context.get_python_api(builder)
        ptyp = type(pyapi)

        if (
            not hasattr(ptyp, "pyomp_patch_installed")
            or ptyp.pyomp_patch_installed == False
        ):
            ptyp.pyomp_patch_installed = True
            # print("update_context", "id(ptyp.emit_environment_sentry)", id(ptyp.emit_environment_sentry), "id(context)", id(context))
            setattr(ptyp, "orig_emit_environment_sentry", ptyp.emit_environment_sentry)

            def pyomp_emit_environment_sentry(self, *args, **kwargs):
                builder = self.builder
                # print("pyomp_emit_environment_sentry")
                if in_openmp_region(builder):
                    return False
                return self.orig_emit_environment_sentry(*args, **kwargs)

            setattr(ptyp, "emit_environment_sentry", pyomp_emit_environment_sentry)
            # print("after", id(pyomp_emit_environment_sentry), id(ptyp.emit_environment_sentry))

    def fix_dispatchers(self, typemap, typingctx, cuda_target):
        fixup_dict = {}
        for k, v in typemap.items():
            if isinstance(v, Dispatcher) and not isinstance(
                v, numba_cuda.types.CUDADispatcher
            ):
                # targetoptions = v.targetoptions.copy()
                # targetoptions['device'] = True
                # targetoptions['debug'] = targetoptions.get('debug', False)
                # targetoptions['opt'] = targetoptions.get('opt', True)
                vdispatcher = v.dispatcher
                vdispatcher.targetoptions.pop("nopython", None)
                vdispatcher.targetoptions.pop("boundscheck", None)
                disp = typingctx.resolve_value_type(vdispatcher)
                fixup_dict[k] = disp
                for sig in vdispatcher.overloads.keys():
                    disp.dispatcher.compile_device(sig, cuda_target=cuda_target)

        for k, v in fixup_dict.items():
            del typemap[k]
            typemap[k] = v

    def lower(self, lowerer):
        typingctx = lowerer.context.typing_context
        targetctx = lowerer.context
        typemap = lowerer.fndesc.typemap
        calltypes = lowerer.fndesc.calltypes
        context = lowerer.context
        builder = lowerer.builder
        mod = builder.module
        library = lowerer.library
        library.openmp = True
        self.block = builder.block
        self.builder = builder
        self.lowerer = lowerer
        self.update_context(context, builder)
        if DEBUG_OPENMP >= 1:
            print(
                "region ids lower:block",
                id(self),
                self,
                id(self.block),
                self.block,
                type(self.block),
                self.tags,
                len(self.tags),
                "builder_id:",
                id(self.builder),
                "block_id:",
                id(self.block),
            )
            for k, v in lowerer.func_ir.blocks.items():
                print("block post copy:", k, id(v), id(v.body))

        # Convert implicit tags to explicit form now that we have typing info.
        for i in range(len(self.tags)):
            cur_tag = self.tags[i]
            if cur_tag.name == "QUAL.OMP.TARGET.IMPLICIT":
                if isinstance(
                    typemap_lookup(typemap, cur_tag.arg), types.npytypes.Array
                ):
                    cur_tag.name = "QUAL.OMP.MAP.TOFROM"
                else:
                    cur_tag.name = "QUAL.OMP.FIRSTPRIVATE"

        if DEBUG_OPENMP >= 1:
            for otag in self.tags:
                print("otag:", otag, type(otag.arg))

        # Remove LLVM vars that might have been added if this is an OpenMP
        # region inside a target region.
        count_alloca_instr = len(
            list(
                filter(
                    lambda x: isinstance(x.arg, lir.instructions.AllocaInstr), self.tags
                )
            )
        )
        assert count_alloca_instr == 0
        # self.tags = list(filter(lambda x: not isinstance(x.arg, lir.instructions.AllocaInstr), self.tags))
        if DEBUG_OPENMP >= 1:
            print("after LLVM tag filter", self.tags, len(self.tags))
            for otag in self.tags:
                print("otag:", otag, type(otag.arg))

        host_side_target_tags = []
        target_num = self.has_target()

        def add_struct_tags(self, var_table):
            extras_before = []
            struct_tags = []
            for i in range(len(self.tags)):
                cur_tag = self.tags[i]
                if cur_tag.name in [
                    "QUAL.OMP.MAP.TOFROM",
                    "QUAL.OMP.MAP.TO",
                    "QUAL.OMP.MAP.FROM",
                    "QUAL.OMP.MAP.ALLOC",
                ]:
                    cur_tag_var = cur_tag.arg
                    if isinstance(cur_tag_var, NameSlice):
                        cur_tag_var = cur_tag_var.name
                    assert isinstance(cur_tag_var, str)
                    cur_tag_typ = typemap_lookup(typemap, cur_tag_var)
                    if isinstance(cur_tag_typ, types.npytypes.Array):
                        cur_tag_ndim = cur_tag_typ.ndim
                        stride_typ = lowerer.context.get_value_type(
                            types.intp
                        )  # lir.Type.int(64)
                        stride_abi_size = context.get_abi_sizeof(stride_typ)
                        array_var = var_table[cur_tag_var]
                        if DEBUG_OPENMP >= 1:
                            print(
                                "Found array mapped:",
                                cur_tag.name,
                                cur_tag.arg,
                                cur_tag_typ,
                                type(cur_tag_typ),
                                stride_typ,
                                type(stride_typ),
                                stride_abi_size,
                                array_var,
                                type(array_var),
                            )
                        uniqueness = get_unique()
                        if isinstance(cur_tag.arg, NameSlice):
                            the_slice = cur_tag.arg.the_slice[0][0]
                            assert the_slice.step is None
                            if isinstance(the_slice.start, int):
                                start_index_var = ir.Var(
                                    None,
                                    f"{cur_tag_var}_start_index_var{target_num}{uniqueness}",
                                    array_var.loc,
                                )
                                start_assign = ir.Assign(
                                    ir.Const(the_slice.start, array_var.loc),
                                    start_index_var,
                                    array_var.loc,
                                )

                                typemap[start_index_var.name] = types.int64
                                lowerer.lower_inst(start_assign)
                                extras_before.append(start_assign)
                                lowerer._alloca_var(
                                    start_index_var.name, typemap[start_index_var.name]
                                )
                                lowerer.loadvar(start_index_var.name)
                            else:
                                start_index_var = the_slice.start
                                assert isinstance(start_index_var, str)
                                start_index_var = ir.Var(
                                    None, start_index_var, array_var.loc
                                )
                            if isinstance(the_slice.stop, int):
                                end_index_var = ir.Var(
                                    None,
                                    f"{cur_tag_var}_end_index_var{target_num}{uniqueness}",
                                    array_var.loc,
                                )
                                end_assign = ir.Assign(
                                    ir.Const(the_slice.stop, array_var.loc),
                                    end_index_var,
                                    array_var.loc,
                                )
                                typemap[end_index_var.name] = types.int64
                                lowerer.lower_inst(end_assign)
                                extras_before.append(end_assign)
                                lowerer._alloca_var(
                                    end_index_var.name, typemap[end_index_var.name]
                                )
                                lowerer.loadvar(end_index_var.name)
                            else:
                                end_index_var = the_slice.stop
                                assert isinstance(end_index_var, str)
                                end_index_var = ir.Var(
                                    None, end_index_var, array_var.loc
                                )

                            num_elements_var = ir.Var(
                                None,
                                f"{cur_tag_var}_num_elements_var{target_num}{uniqueness}",
                                array_var.loc,
                            )
                            size_binop = ir.Expr.binop(
                                operator.sub,
                                end_index_var,
                                start_index_var,
                                array_var.loc,
                            )
                            size_assign = ir.Assign(
                                size_binop, num_elements_var, array_var.loc
                            )
                            calltypes[size_binop] = typing.signature(
                                types.int64, types.int64, types.int64
                            )
                        else:
                            start_index_var = 0
                            num_elements_var = ir.Var(
                                None,
                                f"{cur_tag_var}_num_elements_var{target_num}{uniqueness}",
                                array_var.loc,
                            )
                            size_getattr = ir.Expr.getattr(
                                array_var, "size", array_var.loc
                            )
                            size_assign = ir.Assign(
                                size_getattr, num_elements_var, array_var.loc
                            )

                        typemap[num_elements_var.name] = types.int64
                        lowerer.lower_inst(size_assign)
                        extras_before.append(size_assign)
                        lowerer._alloca_var(
                            num_elements_var.name, typemap[num_elements_var.name]
                        )

                        # see core/datamodel/models.py
                        lowerer.loadvar(num_elements_var.name)  # alloca the var

                        # see core/datamodel/models.py
                        if isinstance(start_index_var, ir.Var):
                            lowerer.loadvar(start_index_var.name)  # alloca the var
                        if isinstance(num_elements_var, ir.Var):
                            lowerer.loadvar(num_elements_var.name)  # alloca the var
                        struct_tags.append(
                            openmp_tag(
                                cur_tag.name + ".STRUCT",
                                cur_tag_var + "*data",
                                non_arg=True,
                                omp_slice=(start_index_var, num_elements_var),
                            )
                        )
                        struct_tags.append(
                            openmp_tag(
                                "QUAL.OMP.MAP.TO.STRUCT",
                                cur_tag_var + "*shape",
                                non_arg=True,
                                omp_slice=(0, 1),
                            )
                        )
                        struct_tags.append(
                            openmp_tag(
                                "QUAL.OMP.MAP.TO.STRUCT",
                                cur_tag_var + "*strides",
                                non_arg=True,
                                omp_slice=(0, 1),
                            )
                        )
                        # Peel off NameSlice, it served its purpose and is not
                        # needed by the rest of compilation.
                        if isinstance(cur_tag.arg, NameSlice):
                            cur_tag.arg = cur_tag.arg.name

            return struct_tags, extras_before

        if self.tags[0].name in [
            "DIR.OMP.TARGET.DATA",
            "DIR.OMP.TARGET.ENTER.DATA",
            "DIR.OMP.TARGET.EXIT.DATA",
            "DIR.OMP.TARGET.UPDATE",
        ]:
            var_table = get_name_var_table(lowerer.func_ir.blocks)
            struct_tags, extras_before = add_struct_tags(self, var_table)
            self.tags.extend(struct_tags)
            for extra in extras_before:
                lowerer.lower_inst(extra)

        elif target_num is not None and self.target_copy != True:
            var_table = get_name_var_table(lowerer.func_ir.blocks)

            ompx_attrs = list(
                filter(lambda x: x.name == "QUAL.OMP.OMPX_ATTRIBUTE", self.tags)
            )
            self.tags = list(
                filter(lambda x: x.name != "QUAL.OMP.OMPX_ATTRIBUTE", self.tags)
            )
            selected_device = 0
            device_tags = get_tags_of_type(self.tags, "QUAL.OMP.DEVICE")
            if len(device_tags) > 0:
                device_tag = device_tags[-1]
                if isinstance(device_tag.arg, int):
                    selected_device = device_tag.arg
                else:
                    assert False
                if DEBUG_OPENMP >= 1:
                    print("new selected device:", selected_device)

            struct_tags, extras_before = add_struct_tags(self, var_table)
            self.tags.extend(struct_tags)
            if DEBUG_OPENMP >= 1:
                for otag in self.tags:
                    print("tag in target:", otag, type(otag.arg))

            from numba.core.compiler import Compiler, Flags

            if DEBUG_OPENMP >= 1:
                print("openmp start region lower has target", type(lowerer.func_ir))
            # Make a copy of the host IR being lowered.
            dprint_func_ir(lowerer.func_ir, "original func_ir")
            func_ir = copy_ir(lowerer.func_ir, calltypes)
            dprint_func_ir(func_ir, "copied func_ir")
            if DEBUG_OPENMP >= 1:
                for k, v in lowerer.func_ir.blocks.items():
                    print(
                        "region ids block post copy:",
                        k,
                        id(v),
                        id(func_ir.blocks[k]),
                        id(v.body),
                        id(func_ir.blocks[k].body),
                    )

            remove_dels(func_ir.blocks)

            dprint_func_ir(func_ir, "func_ir after remove_dels")

            def fixup_openmp_pairs(blocks):
                """The Numba IR nodes for the start and end of an OpenMP region
                contain references to each other.  When a target region is
                outlined that contains these pairs of IR nodes then if we
                simply shallow copy them then they'll point to their original
                matching pair in the original IR.  In this function, we go
                through and find what should be matching pairs in the
                outlined (target) IR and make those copies point to each
                other.
                """
                start_dict = {}
                end_dict = {}

                # Go through the blocks in the original IR and create a mapping
                # between the id of the start nodes with their block label and
                # position in the block.  Likewise, do the same for end nodes.
                for label, block in func_ir.blocks.items():
                    for bindex, bstmt in enumerate(block.body):
                        if isinstance(bstmt, openmp_region_start):
                            if DEBUG_OPENMP >= 2:
                                print("region ids found region start", id(bstmt))
                            start_dict[id(bstmt)] = (label, bindex)
                        elif isinstance(bstmt, openmp_region_end):
                            if DEBUG_OPENMP >= 2:
                                print(
                                    "region ids found region end",
                                    id(bstmt.start_region),
                                    id(bstmt),
                                )
                            end_dict[id(bstmt.start_region)] = (label, bindex)
                assert len(start_dict) == len(end_dict)

                # For each start node that we found above, create a copy in the target IR
                # and fixup the references of the copies to point at each other.
                for start_id, blockindex in start_dict.items():
                    start_block, sbindex = blockindex

                    end_block_index = end_dict[start_id]
                    end_block, ebindex = end_block_index

                    if DEBUG_OPENMP >= 2:
                        start_pre_copy = blocks[start_block].body[sbindex]
                        end_pre_copy = blocks[end_block].body[ebindex]

                    # Create copy of the OpenMP start and end nodes in the target outlined IR.
                    blocks[start_block].body[sbindex] = copy.copy(
                        blocks[start_block].body[sbindex]
                    )
                    blocks[end_block].body[ebindex] = copy.copy(
                        blocks[end_block].body[ebindex]
                    )
                    # Reset some fields in the start OpenMP region because the target IR
                    # has not been lowered yet.
                    start_region = blocks[start_block].body[sbindex]
                    start_region.builder = None
                    start_region.block = None
                    start_region.lowerer = None
                    start_region.target_copy = True
                    start_region.tags = copy.deepcopy(start_region.tags)
                    # Remove unnecessary num_teams, thread_limit tags when
                    # emitting a target directive within a kernel to avoid
                    # extraneous arguments in the kernel function.
                    if start_region.has_target() == target_num:
                        start_region.tags.append(openmp_tag("OMP.DEVICE"))
                    end_region = blocks[end_block].body[ebindex]
                    # assert(start_region.omp_region_var is None)
                    assert len(start_region.alloca_queue) == 0
                    # Make start and end copies point at each other.
                    end_region.start_region = start_region
                    start_region.end_region = end_region
                    if DEBUG_OPENMP >= 2:
                        print(
                            f"region ids fixup start: {id(start_pre_copy)}->{id(start_region)} end: {id(end_pre_copy)}->{id(end_region)}"
                        )

            fixup_openmp_pairs(func_ir.blocks)
            state = compiler.StateDict()
            fndesc = lowerer.fndesc
            state.typemap = fndesc.typemap
            state.calltypes = fndesc.calltypes
            state.argtypes = fndesc.argtypes
            state.return_type = fndesc.restype
            if DEBUG_OPENMP >= 1:
                print("context:", context, type(context))
                print("targetctx:", targetctx, type(targetctx))
                print("state:", state, dir(state))
                print("fndesc:", fndesc, type(fndesc))
                print("func_ir type:", type(func_ir))
            dprint_func_ir(func_ir, "target func_ir")
            internal_codegen = targetctx._internal_codegen

            # Find the start and end IR blocks for this offloaded region.
            start_block, end_block = find_target_start_end(func_ir, target_num)
            end_target_node = func_ir.blocks[end_block].body[0]

            if DEBUG_OPENMP >= 1:
                print("start_block:", start_block)
                print("end_block:", end_block)

            blocks_in_region = get_blocks_between_start_end(
                func_ir.blocks, start_block, end_block
            )
            if DEBUG_OPENMP >= 1:
                print("lower blocks_in_region:", blocks_in_region)

            # Find the variables that cross the boundary between the target
            # region and the non-target host-side code.
            ins, outs = transforms.find_region_inout_vars(
                blocks=func_ir.blocks,
                livemap=func_ir.variable_lifetime.livemap,
                callfrom=start_block,
                returnto=end_block,
                body_block_ids=blocks_in_region,
            )

            def add_mapped_to_ins(ins, tags):
                for tag in tags:
                    if tag.arg in ins:
                        continue

                    if tag.name in ["QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.MAP.FROM"]:
                        ins.append(tag.arg)

            add_mapped_to_ins(ins, self.tags)

            normalized_ivs = get_tags_of_type(self.tags, "QUAL.OMP.NORMALIZED.IV")
            if DEBUG_OPENMP >= 1:
                print("ivs ins", normalized_ivs, ins, outs)
            for niv in normalized_ivs:
                if DEBUG_OPENMP >= 1:
                    print("Removing normalized iv from ins", niv.arg)
                if niv.arg in ins:
                    ins.remove(niv.arg)
            # Get the types of the variables live-in to the target region.
            target_args_unordered = ins + list(set(outs) - set(ins))
            if DEBUG_OPENMP >= 1:
                print("ins:", ins, type(ins))
                print("outs:", outs, type(outs))
                # print("args:", state.args)
                print("rettype:", state.return_type, type(state.return_type))
                print("target_args_unordered:", target_args_unordered)
            # Re-use Numba loop lifting code to extract the target region as
            # its own function.
            region_info = transforms._loop_lift_info(
                loop=None,
                inputs=ins,
                # outputs=outs,
                outputs=(),
                callfrom=start_block,
                returnto=end_block,
            )

            region_blocks = dict((k, func_ir.blocks[k]) for k in blocks_in_region)

            if DEBUG_OPENMP >= 1:
                print("region_info:", region_info)
            transforms._loop_lift_prepare_loop_func(region_info, region_blocks)
            # exit_block_label = max(region_blocks.keys())
            # region_blocks[exit_block_label].body = []
            # exit_scope = region_blocks[exit_block_label].scope
            # tmp = exit_scope.make_temp(loc=func_ir.loc)
            # region_blocks[exit_block_label].append(ir.Assign(value=ir.Const(0, func_ir.loc), target=tmp, loc=func_ir.loc))
            # region_blocks[exit_block_label].append(ir.Return(value=tmp, loc=func_ir.loc))

            target_args = []
            outline_arg_typs = []
            # outline_arg_typs = [None] * len(target_args_unordered)
            for tag in self.tags:
                if DEBUG_OPENMP >= 1:
                    print(1, "target_arg?", tag, tag.non_arg, is_target_arg(tag.name))
                if (
                    tag.arg in target_args_unordered
                    and not tag.non_arg
                    and is_target_arg(tag.name)
                ):
                    target_args.append(tag.arg)
                    # target_arg_index = target_args.index(tag.arg)
                    atyp = get_dotted_type(tag.arg, typemap, lowerer)
                    if is_pointer_target_arg(tag.name, atyp):
                        # outline_arg_typs[target_arg_index] = types.CPointer(atyp)
                        outline_arg_typs.append(types.CPointer(atyp))
                        if DEBUG_OPENMP >= 1:
                            print(1, "found cpointer target_arg", tag, atyp, id(atyp))
                    else:
                        # outline_arg_typs[target_arg_index] = atyp
                        outline_arg_typs.append(atyp)
                        if DEBUG_OPENMP >= 1:
                            print(1, "found target_arg", tag, atyp, id(atyp))

            if DEBUG_OPENMP >= 1:
                print("target_args:", target_args)
                print("target_args_unordered:", target_args_unordered)
                print("outline_arg_typs:", outline_arg_typs)
                print("extras_before:", extras_before, start_block)
                for eb in extras_before:
                    print(eb)

            assert len(target_args) == len(target_args_unordered)
            assert len(target_args) == len(outline_arg_typs)

            # Create the outlined IR from the blocks in the region, making the
            # variables crossing into the regions argument.
            outlined_ir = func_ir.derive(
                blocks=region_blocks,
                arg_names=tuple(target_args),
                arg_count=len(target_args),
                force_non_generator=True,
            )
            outlined_ir.blocks[start_block].body = (
                extras_before + outlined_ir.blocks[start_block].body
            )
            for stmt in outlined_ir.blocks[min(outlined_ir.blocks.keys())].body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                    stmt.value.index = target_args.index(stmt.value.name)

            def prepend_device_to_func_name(outlined_ir):
                # Change the name of the outlined function to prepend the
                # word "device" to the function name.
                fparts = outlined_ir.func_id.func_qualname.split(".")
                fparts[-1] = "device" + str(target_num) + fparts[-1]
                outlined_ir.func_id.func_qualname = ".".join(fparts)
                outlined_ir.func_id.func_name = fparts[-1]
                uid = next(bytecode.FunctionIdentity._unique_ids)
                outlined_ir.func_id.unique_name = "{}${}".format(
                    outlined_ir.func_id.func_qualname, uid
                )

            prepend_device_to_func_name(outlined_ir)
            device_func_name = outlined_ir.func_id.func_qualname
            if DEBUG_OPENMP >= 1:
                print(
                    "outlined_ir:",
                    type(outlined_ir),
                    type(outlined_ir.func_id),
                    outlined_ir.arg_names,
                    device_func_name,
                )
                dprint_func_ir(outlined_ir, "outlined_ir")

            # Create a copy of the state and the typemap inside of it so that changes
            # for compiling the outlined IR don't effect the original compilation state
            # of the host.
            state_copy = copy.copy(state)
            state_copy.typemap = copy.copy(typemap)

            entry_block_num = min(outlined_ir.blocks.keys())
            entry_block = outlined_ir.blocks[entry_block_num]
            if DEBUG_OPENMP >= 1:
                print("entry_block:", entry_block)
                for x in entry_block.body:
                    print(x)
            rev_arg_assigns = []
            # Add entries in the copied typemap for the arguments to the outlined IR.
            for idx, zipvar in enumerate(zip(target_args, outline_arg_typs)):
                var_in, vtyp = zipvar
                arg_name = "arg." + var_in
                state_copy.typemap.pop(arg_name, None)
                state_copy.typemap[arg_name] = vtyp

            last_block = outlined_ir.blocks[end_block]
            last_block.body = (
                [end_target_node]
                + last_block.body[:-1]
                + rev_arg_assigns
                + last_block.body[-1:]
            )

            assert isinstance(last_block.body[-1], ir.Return)
            # Add typemap entry for the empty tuple return type.
            state_copy.typemap[last_block.body[-1].value.name] = types.none
            # end test

            if DEBUG_OPENMP >= 1:
                print("selected_device:", selected_device)

            if selected_device == 1:
                flags = Flags()
                flags.enable_ssa = False
                device_lowerer_pipeline = OnlyLower

                subtarget = OpenmpCPUTargetContext(
                    device_func_name, targetctx.typing_context
                )
                # Copy everything (like registries) from cpu context into the new OpenMPCPUTargetContext subtarget
                # except call_conv which is the whole point of that class so that the minimal call convention is used.
                subtarget.__dict__.update(
                    {
                        k: targetctx.__dict__[k]
                        for k in targetctx.__dict__.keys() - {"call_conv"}
                    }
                )
                # subtarget.install_registry(imputils.builtin_registry)
                # Turn off the Numba runtime (incref and decref mostly) for the target compilation.
                subtarget.enable_nrt = False
                typingctx_outlined = targetctx.typing_context

                import numba.core.codegen as codegen

                subtarget._internal_codegen = codegen.AOTCPUCodegen(
                    mod.name + f"$device{selected_device}"
                )
                subtarget._internal_codegen._library_class = CustomAOTCPUCodeLibrary
                subtarget._internal_codegen._engine.set_object_cache(None, None)
                device_target = subtarget
            elif selected_device == 0:
                from numba.core import target_extension

                orig_target = getattr(
                    target_extension._active_context,
                    "target",
                    target_extension._active_context_default,
                )
                target_extension._active_context.target = "cuda"

                flags = cuda_compiler.CUDAFlags()

                typingctx_outlined = cuda_descriptor.cuda_target.typing_context
                device_target = OpenmpCUDATargetContext(
                    device_func_name, typingctx_outlined
                )
                device_target.fndesc = fndesc
                # device_target = cuda_descriptor.cuda_target.target_context

                device_lowerer_pipeline = OnlyLowerCUDA
                openmp_cuda_target = numba_cuda.descriptor.CUDATarget("openmp_cuda")
                openmp_cuda_target._typingctx = typingctx_outlined
                openmp_cuda_target._targetctx = device_target
                self.fix_dispatchers(
                    state_copy.typemap, typingctx_outlined, openmp_cuda_target
                )

                typingctx_outlined.refresh()
                device_target.refresh()
                dprint_func_ir(outlined_ir, "outlined_ir before replace np.empty")
                replace_np_empty_with_cuda_shared(
                    outlined_ir,
                    state_copy.typemap,
                    calltypes,
                    device_func_name,
                    typingctx_outlined,
                )
                dprint_func_ir(outlined_ir, "outlined_ir after replace np.empty")
            else:
                raise NotImplementedError("Unsupported OpenMP device number")

            device_target.state_copy = state_copy
            # Do not compile (generate native code), just lower (to LLVM)
            flags.no_compile = True
            flags.no_cpython_wrapper = True
            flags.no_cfunc_wrapper = True
            # What to do here?
            flags.forceinline = True
            # Propagate fastmath flag on the outer function to the inner outlined compile.
            # TODO: find a good way to handle fastmath. Clang has
            # fp-contractions on by default for GPU code.
            # flags.fastmath = True#state_copy.flags.fastmath
            flags.release_gil = True
            flags.nogil = True
            flags.inline = "always"
            # Create a pipeline that only lowers the outlined target code.  No need to
            # compile because it has already gone through those passes.
            if DEBUG_OPENMP >= 1:
                print(
                    "outlined_ir:",
                    outlined_ir,
                    type(outlined_ir),
                    outlined_ir.arg_names,
                )
                dprint_func_ir(outlined_ir, "outlined_ir")
                dprint_func_ir(func_ir, "target after outline func_ir")
                dprint_func_ir(lowerer.func_ir, "original func_ir")
                print("state_copy.typemap:", state_copy.typemap)
                print("region ids before compile_ir")
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )

            cres = compiler.compile_ir(
                typingctx_outlined,
                device_target,
                outlined_ir,
                outline_arg_typs,
                types.none,
                flags,
                {},
                pipeline_class=device_lowerer_pipeline,
                is_lifted_loop=False,
            )  # tried this as True since code derived from loop lifting code but it goes through the pipeline twice and messes things up

            if DEBUG_OPENMP >= 2:
                print("cres:", type(cres))
                print("fndesc:", cres.fndesc, cres.fndesc.mangled_name)
                print("metadata:", cres.metadata)
            cres_library = cres.library
            if DEBUG_OPENMP >= 2:
                print("cres_library:", type(cres_library))
                sys.stdout.flush()
            cres_library._ensure_finalized()
            if DEBUG_OPENMP >= 2:
                print("ensure_finalized:")
                sys.stdout.flush()

            if DEBUG_OPENMP >= 1:
                print("region ids compile_ir")
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )
                print(
                    "==================================================================================="
                )

                for k, v in lowerer.func_ir.blocks.items():
                    print(
                        "block post copy:",
                        k,
                        id(v),
                        id(func_ir.blocks[k]),
                        id(v.body),
                        id(func_ir.blocks[k].body),
                    )

            shared_ext = ".so"
            if sys.platform.startswith("win"):
                shared_ext = ".dll"

            # TODO: move device pipelines in numba proper.
            if selected_device == 1:
                if DEBUG_OPENMP >= 1:
                    with open(cres_library.name + ".ll", "w") as f:
                        f.write(cres_library.get_llvm_str())

                fd_o, filename_o = tempfile.mkstemp(".o")
                os.close(fd_o)
                filename_so = Path(filename_o).with_suffix(".so")

                target_elf = cres_library.emit_native_object()
                with open(filename_o, "wb") as f:
                    f.write(target_elf)

                # Create shared library as required by the libomptarget host
                # plugin.

                link_shared_library(obj_path=filename_o, out_path=filename_so)

                with open(filename_so, "rb") as f:
                    target_elf = f.read()
                if DEBUG_OPENMP >= 1:
                    print("filename_o", filename_o, "filename_so", filename_so)

                # Remove the temporary files.
                os.remove(filename_o)
                os.remove(filename_so)

                if DEBUG_OPENMP >= 1:
                    print("target_elf:", type(target_elf), len(target_elf))
                    sys.stdout.flush()
            elif selected_device == 0:
                import numba.cuda.api as cudaapi
                import numba.cuda.cudadrv.libs as cudalibs
                from numba.cuda.cudadrv import driver
                from numba.core.llvm_bindings import create_pass_manager_builder
                from numba.cuda.codegen import CUDA_TRIPLE

                class OpenMPCUDACodegen:
                    def __init__(self):
                        self.cc = cudaapi.get_current_device().compute_capability
                        self.sm = "sm_" + str(self.cc[0]) + str(self.cc[1])
                        self.libdevice_path = cudalibs.get_libdevice()
                        with open(self.libdevice_path, "rb") as f:
                            self.libs_mod = ll.parse_bitcode(f.read())
                        self.libomptarget_arch = (
                            libpath
                            / "libomp"
                            / "lib"
                            / f"libomptarget-new-nvptx-{self.sm}.bc"
                        )
                        with open(self.libomptarget_arch, "rb") as f:
                            libomptarget_mod = ll.parse_bitcode(f.read())
                        ## Link in device, openmp libraries.
                        self.libs_mod.link_in(libomptarget_mod)
                        # Initialize asm printers to codegen ptx.
                        ll.initialize_all_targets()
                        ll.initialize_all_asmprinters()
                        target = ll.Target.from_triple(CUDA_TRIPLE)
                        self.tm = target.create_target_machine(cpu=self.sm, opt=3)

                    def _get_target_image(
                        self, mod, filename_prefix, use_toolchain=False
                    ):
                        if DEBUG_OPENMP_LLVM_PASS >= 1:
                            with open(filename_prefix + ".ll", "w") as f:
                                f.write(str(mod))

                        # Lower openmp intrinsics.
                        mod = run_intrinsics_openmp_pass(mod)
                        with ll.create_module_pass_manager() as pm:
                            pm.add_cfg_simplification_pass()
                            pm.run(mod)

                        if DEBUG_OPENMP_LLVM_PASS >= 1:
                            with open(filename_prefix + "-intrinsics_omp.ll", "w") as f:
                                f.write(str(mod))

                        mod.link_in(self.libs_mod, preserve=True)
                        # Internalize non-kernel function definitions.
                        for func in mod.functions:
                            if func.is_declaration:
                                continue
                            if func.linkage != ll.Linkage.external:
                                continue
                            if "__omp_offload_numba" in func.name:
                                continue
                            func.linkage = "internal"

                        with ll.create_module_pass_manager() as pm:
                            self.tm.add_analysis_passes(pm)
                            pm.add_global_dce_pass()
                            pm.run(mod)

                        if DEBUG_OPENMP_LLVM_PASS >= 1:
                            with open(
                                filename_prefix + "-intrinsics_omp-linked.ll", "w"
                            ) as f:
                                f.write(str(mod))

                        # Run passes for optimization, including target-specific passes.
                        # Run function passes.
                        with ll.create_function_pass_manager(mod) as pm:
                            self.tm.add_analysis_passes(pm)
                            with create_pass_manager_builder(
                                opt=3, slp_vectorize=True, loop_vectorize=True
                            ) as pmb:
                                # TODO: upstream adjust_pass_manager to llvmlite?
                                # self.tm.adjust_pass_manager(pmb)
                                pmb.populate(pm)
                            for func in mod.functions:
                                pm.initialize()
                                pm.run(func)
                                pm.finalize()

                        # Run module passes.
                        with ll.create_module_pass_manager() as pm:
                            self.tm.add_analysis_passes(pm)
                            with create_pass_manager_builder(
                                opt=3, slp_vectorize=True, loop_vectorize=True
                            ) as pmb:
                                # TODO: upstream adjust_pass_manager to llvmlite?
                                # self.tm.adjust_pass_manager(pmb)
                                pmb.populate(pm)
                            pm.run(mod)

                        if DEBUG_OPENMP_LLVM_PASS >= 1:
                            mod.verify()
                            with open(
                                filename_prefix + "-intrinsics_omp-linked-opt.ll", "w"
                            ) as f:
                                f.write(str(mod))

                        # Generate ptx assemlby.
                        ptx = self.tm.emit_assembly(mod)
                        if use_toolchain:
                            # ptxas does file I/O, so output the assembly and ingest the generated cubin.
                            with open(
                                filename_prefix + "-intrinsics_omp-linked-opt.s", "w"
                            ) as f:
                                f.write(ptx)

                            subprocess.run(
                                [
                                    "ptxas",
                                    "-m64",
                                    "--gpu-name",
                                    self.sm,
                                    filename_prefix + "-intrinsics_omp-linked-opt.s",
                                    "-o",
                                    filename_prefix + "-intrinsics_omp-linked-opt.o",
                                ],
                                check=True,
                            )

                            with open(
                                filename_prefix + "-intrinsics_omp-linked-opt.o", "rb"
                            ) as f:
                                cubin = f.read()
                        else:
                            if DEBUG_OPENMP_LLVM_PASS >= 1:
                                with open(
                                    filename_prefix + "-intrinsics_omp-linked-opt.s",
                                    "w",
                                ) as f:
                                    f.write(ptx)

                            linker_kwargs = {}
                            for x in ompx_attrs:
                                linker_kwargs[x.arg[0]] = (
                                    tuple(x.arg[1])
                                    if len(x.arg[1]) > 1
                                    else x.arg[1][0]
                                )
                            # NOTE: DO NOT set cc, since the linker will always
                            # compile for the existing GPU context and it is
                            # incompatible with the launch_bounds ompx_attribute.
                            linker = driver.Linker.new(**linker_kwargs)
                            linker.add_ptx(ptx.encode())
                            cubin = linker.complete()

                            if DEBUG_OPENMP_LLVM_PASS >= 1:
                                with open(
                                    filename_prefix + "-intrinsics_omp-linked-opt.o",
                                    "wb",
                                ) as f:
                                    f.write(cubin)

                        return cubin

                    def get_target_image(self, cres):
                        filename_prefix = cres_library.name
                        allmods = cres_library.modules
                        linked_mod = ll.parse_assembly(str(allmods[0]))
                        for mod in allmods[1:]:
                            linked_mod.link_in(ll.parse_assembly(str(mod)))
                        if OPENMP_DEVICE_TOOLCHAIN >= 1:
                            return self._get_target_image(
                                linked_mod, filename_prefix, use_toolchain=True
                            )
                        else:
                            return self._get_target_image(linked_mod, filename_prefix)

                target_extension._active_context.target = orig_target
                omp_cuda_cg = OpenMPCUDACodegen()
                target_elf = omp_cuda_cg.get_target_image(cres)
            else:
                raise NotImplementedError("Unsupported OpenMP device number")

            # if cuda then run ptxas on the cres and pass that

            # bytes_array_typ = lir.ArrayType(cgutils.voidptr_t, len(target_elf))
            # bytes_array_typ = lir.ArrayType(cgutils.int8_t, len(target_elf))
            # dev_image = cgutils.add_global_variable(mod, bytes_array_typ, ".omp_offloading.device_image")
            # dev_image.initializer = lir.Constant.array(cgutils.int8_t, target_elf)
            # dev_image.initializer = lir.Constant.array(cgutils.int8_t, target_elf)
            add_target_globals_in_numba = int(
                os.environ.get("NUMBA_OPENMP_ADD_TARGET_GLOBALS", 0)
            )
            if add_target_globals_in_numba != 0:
                elftext = cgutils.make_bytearray(target_elf)
                dev_image = targetctx.insert_unique_const(
                    mod, ".omp_offloading.device_image", elftext
                )
                mangled_name = cgutils.make_bytearray(
                    cres.fndesc.mangled_name.encode("utf-8") + b"\x00"
                )
                mangled_var = targetctx.insert_unique_const(
                    mod, ".omp_offloading.entry_name", mangled_name
                )

                llvmused_typ = lir.ArrayType(cgutils.voidptr_t, 2)
                llvmused_gv = cgutils.add_global_variable(
                    mod, llvmused_typ, "llvm.used"
                )
                llvmused_syms = [
                    lir.Constant.bitcast(dev_image, cgutils.voidptr_t),
                    lir.Constant.bitcast(mangled_var, cgutils.voidptr_t),
                ]
                llvmused_gv.initializer = lir.Constant.array(
                    cgutils.voidptr_t, llvmused_syms
                )
                llvmused_gv.linkage = "appending"
            else:
                host_side_target_tags.append(
                    openmp_tag(
                        "QUAL.OMP.TARGET.DEV_FUNC",
                        StringLiteral(cres.fndesc.mangled_name.encode("utf-8")),
                    )
                )
                host_side_target_tags.append(
                    openmp_tag("QUAL.OMP.TARGET.ELF", StringLiteral(target_elf))
                )

            if DEBUG_OPENMP >= 1:
                dprint_func_ir(func_ir, "target after outline compiled func_ir")

        llvm_token_t = TokenType()
        fnty = lir.FunctionType(llvm_token_t, [])
        tags_to_include = self.tags + host_side_target_tags
        # tags_to_include = list(filter(lambda x: x.name != "DIR.OMP.TARGET", tags_to_include))
        self.filtered_tag_length = len(tags_to_include)
        if DEBUG_OPENMP >= 1:
            print("filtered_tag_length:", self.filtered_tag_length)

        if len(tags_to_include) > 0:
            if DEBUG_OPENMP >= 1:
                print("push_alloca_callbacks")

            push_alloca_callback(lowerer, openmp_region_alloca, self, builder)
            tag_str = openmp_tag_list_to_str(tags_to_include, lowerer, True)
            pre_fn = builder.module.declare_intrinsic(
                "llvm.directive.region.entry", (), fnty
            )
            assert self.omp_region_var is None
            self.omp_region_var = builder.call(pre_fn, [], tail=False)
            self.omp_region_var.__class__ = CallInstrWithOperandBundle
            self.omp_region_var.set_tags(tag_str)
            # This is used by the post-lowering pass over LLVM to add LLVM alloca
            # vars to the Numba IR openmp node and then when the exit of the region
            # is detected then the tags in the enter directive are updated.
            self.omp_region_var.save_orig_numba_openmp = self
            if DEBUG_OPENMP >= 2:
                print("setting omp_region_var", self.omp_region_var._get_name())
        if self.acq_res:
            builder.fence("acquire")
        if self.acq_rel:
            builder.fence("acq_rel")

        for otag in self.tags:  # should be tags_to_include?
            otag.post_entry(lowerer)

        if DEBUG_OPENMP >= 1:
            sys.stdout.flush()

    def __str__(self):
        return (
            "openmp_region_start "
            + ", ".join([str(x) for x in self.tags])
            + " target="
            + str(self.target_copy)
        )


class OnlyLower(compiler.CompilerBase):
    def __init__(self, typingctx, targetctx, library, args, restype, flags, locals):
        super().__init__(typingctx, targetctx, library, args, restype, flags, locals)
        self.state.typemap = targetctx.state_copy.typemap
        self.state.calltypes = targetctx.state_copy.calltypes

    def define_pipelines(self):
        pms = []
        if not self.state.flags.force_pyobject:
            pms.append(
                compiler.DefaultPassBuilder.define_nopython_lowering_pipeline(
                    self.state
                )
            )
        return pms


class OnlyLowerCUDA(numba_cuda.compiler.CUDACompiler):
    def __init__(self, typingctx, targetctx, library, args, restype, flags, locals):
        super().__init__(typingctx, targetctx, library, args, restype, flags, locals)
        self.state.typemap = targetctx.state_copy.typemap
        self.state.calltypes = targetctx.state_copy.calltypes

    def define_pipelines(self):
        pm = compiler_machinery.PassManager("cuda")
        # Numba <=0.57 implements CUDALegalization to support CUDA <11.2
        # versions.  Numba >0.58 drops this support. We enclose in a try-except
        # block to avoid errors, delegating to Numba support.
        try:
            pm.add_pass(numba_cuda.compiler.CUDALegalization, "CUDA legalization")
        except AttributeError:
            pass
        lowering_passes = self.define_cuda_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)
        pm.finalize()
        return [pm]


class openmp_region_end(ir.Stmt):
    def __init__(self, start_region, tags, loc):
        if DEBUG_OPENMP >= 1:
            print("region ids openmp_region_end::__init__", id(self), id(start_region))
        self.start_region = start_region
        self.tags = tags
        self.loc = loc
        self.start_region.end_region = self

    def __new__(cls, *args, **kwargs):
        instance = super(openmp_region_end, cls).__new__(cls)
        # print("openmp_region_end::__new__", id(instance))
        return instance

    def list_vars(self):
        return list_vars_from_tags(self.tags)

    def lower(self, lowerer):
        typingctx = lowerer.context.typing_context
        targetctx = lowerer.context
        typemap = lowerer.fndesc.typemap
        context = lowerer.context
        builder = lowerer.builder
        library = lowerer.library

        if DEBUG_OPENMP >= 2:
            print("openmp_region_end::lower", id(self), id(self.start_region))
            sys.stdout.flush()

        if self.start_region.acq_res:
            builder.fence("release")

        if DEBUG_OPENMP >= 1:
            print("pop_alloca_callbacks")

        if DEBUG_OPENMP >= 2:
            print("start_region tag length:", self.start_region.filtered_tag_length)

        if self.start_region.filtered_tag_length > 0:
            llvm_token_t = TokenType()
            fnty = lir.FunctionType(lir.VoidType(), [llvm_token_t])
            # The callback is only needed if llvm.directive.region.entry was added
            # which only happens if tag length > 0.
            pop_alloca_callback(lowerer, builder)

            # Process the accumulated allocas in the start region.
            self.start_region.process_alloca_queue()

            assert self.start_region.omp_region_var != None
            if DEBUG_OPENMP >= 2:
                print(
                    "before adding exit", self.start_region.omp_region_var._get_name()
                )

            for fp in filter(
                lambda x: x.name == "QUAL.OMP.FIRSTPRIVATE", self.start_region.tags
            ):
                new_del = ir.Del(fp.arg, self.loc)
                lowerer.lower_inst(new_del)

            pre_fn = builder.module.declare_intrinsic(
                "llvm.directive.region.exit", (), fnty
            )
            or_end_call = builder.call(
                pre_fn, [self.start_region.omp_region_var], tail=True
            )
            or_end_call.__class__ = CallInstrWithOperandBundle
            or_end_call.set_tags(openmp_tag_list_to_str(self.tags, lowerer, True))

            if DEBUG_OPENMP >= 1:
                print(
                    "OpenMP end lowering firstprivate_dead_after len:",
                    len(self.start_region.firstprivate_dead_after),
                )

            for fp in self.start_region.firstprivate_dead_after:
                new_del = ir.Del(fp.arg, self.loc)
                lowerer.lower_inst(new_del)

    def __str__(self):
        return "openmp_region_end " + ", ".join([str(x) for x in self.tags])

    def has_target(self):
        for t in self.tags:
            if is_target_tag(t.name):
                return t.arg
        return None


def compute_cfg_from_llvm_blocks(blocks):
    cfg = CFGraph()
    name_to_index = {}
    for b in blocks:
        # print("b:", b.name, type(b.name))
        cfg.add_node(b.name)

    for bindex, b in enumerate(blocks):
        term = b.terminator
        # print("term:", b.name, term, type(term))
        if isinstance(term, lir.instructions.Branch):
            cfg.add_edge(b.name, term.operands[0].name)
            name_to_index[b.name] = (bindex, [term.operands[0].name])
        elif isinstance(term, lir.instructions.ConditionalBranch):
            cfg.add_edge(b.name, term.operands[1].name)
            cfg.add_edge(b.name, term.operands[2].name)
            name_to_index[b.name] = (
                bindex,
                [term.operands[1].name, term.operands[2].name],
            )
        elif isinstance(term, lir.instructions.Ret):
            name_to_index[b.name] = (bindex, [])
        elif isinstance(term, lir.instructions.SwitchInstr):
            cfg.add_edge(b.name, term.default.name)
            for _, blk in term.cases:
                cfg.add_edge(b.name, blk.name)
            out_blks = [x[1].name for x in term.cases]
            out_blks.append(term.default.name)
            name_to_index[b.name] = (bindex, out_blks)
        elif isinstance(term, lir.instructions.Unreachable):
            pass
        else:
            print("Unknown term:", term, type(term))
            assert False  # Should never get here.

    cfg.set_entry_point("entry")
    cfg.process()
    return cfg, name_to_index


def compute_llvm_topo_order(blocks):
    cfg, name_to_index = compute_cfg_from_llvm_blocks(blocks)
    post_order = []
    seen = set()

    def _dfs_rec(node):
        if node not in seen:
            seen.add(node)
            succs = cfg._succs[node]

            # If there are no successors then we are done.
            # This is the case for an unreachable.
            if not succs:
                return

            # This is needed so that the inside of loops are
            # handled first before their exits.
            nexts = name_to_index[node][1]
            if len(nexts) == 2:
                succs = [nexts[1], nexts[0]]

            for dest in succs:
                if (node, dest) not in cfg._back_edges:
                    _dfs_rec(dest)
            post_order.append(node)

    _dfs_rec(cfg.entry_point())
    post_order.reverse()
    return post_order, name_to_index


class CollectUnknownLLVMVarsPrivate(lir.transforms.Visitor):
    def __init__(self):
        self.active_openmp_directives = []
        self.start_num = 0

    # Override the default function visitor to go in topo order
    def visit_Function(self, func):
        self._function = func
        if len(func.blocks) == 0:
            return None
        if DEBUG_OPENMP >= 1:
            print("Collect visit_Function:", func.blocks, type(func.blocks))
        topo_order, name_to_index = compute_llvm_topo_order(func.blocks)
        topo_order = list(topo_order)
        if DEBUG_OPENMP >= 1:
            print("topo_order:", topo_order)

        for bbname in topo_order:
            if DEBUG_OPENMP >= 1:
                print("Visiting block:", bbname)
            self.visit_BasicBlock(func.blocks[name_to_index[bbname][0]])

        if DEBUG_OPENMP >= 1:
            print("Collect visit_Function done")

    def visit_Instruction(self, instr):
        if len(self.active_openmp_directives) > 0:
            if DEBUG_OPENMP >= 1:
                print("Collect instr:", instr, type(instr))
            for op in instr.operands:
                if isinstance(op, lir.AllocaInstr):
                    if DEBUG_OPENMP >= 1:
                        print("Collect AllocaInstr operand:", op, op.name)
                    for directive in self.active_openmp_directives:
                        directive.save_orig_numba_openmp.alloca(op, None)
                else:
                    if DEBUG_OPENMP >= 2:
                        print("non-alloca:", op, type(op))
                    pass

        if isinstance(instr, lir.CallInstr):
            if instr.callee.name == "llvm.directive.region.entry":
                if DEBUG_OPENMP >= 1:
                    print(
                        "Collect Found openmp region entry:",
                        instr,
                        type(instr),
                        "\n",
                        instr.tags,
                        type(instr.tags),
                        id(self),
                        len(self.active_openmp_directives),
                    )
                self.active_openmp_directives.append(instr)
                if DEBUG_OPENMP >= 1:
                    print("post append:", len(self.active_openmp_directives))
                assert hasattr(instr, "save_orig_numba_openmp")
            if instr.callee.name == "llvm.directive.region.exit":
                if DEBUG_OPENMP >= 1:
                    print(
                        "Collect Found openmp region exit:",
                        instr,
                        type(instr),
                        "\n",
                        instr.tags,
                        type(instr.tags),
                        id(self),
                        len(self.active_openmp_directives),
                    )
                enter_directive = self.active_openmp_directives.pop()
                enter_directive.save_orig_numba_openmp.post_lowering_process_alloca_queue(
                    enter_directive
                )


def post_lowering_openmp(mod):
    if DEBUG_OPENMP >= 1:
        print("post_lowering_openmp")

    # This will gather the information.
    collect_fixup = CollectUnknownLLVMVarsPrivate()
    collect_fixup.visit(mod)

    if DEBUG_OPENMP >= 1:
        print("post_lowering_openmp done")


# Callback for ir_extension_usedefs
def openmp_region_start_defs(region, use_set=None, def_set=None):
    assert isinstance(region, openmp_region_start)
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()
    for tag in region.tags:
        tag.add_to_usedef_set(use_set, def_set, start=True)
    return _use_defs_result(usemap=use_set, defmap=def_set)


def openmp_region_end_defs(region, use_set=None, def_set=None):
    assert isinstance(region, openmp_region_end)
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()
    # We refer to the clauses from the corresponding start of the region.
    start_region = region.start_region
    for tag in start_region.tags:
        tag.add_to_usedef_set(use_set, def_set, start=False)
    return _use_defs_result(usemap=use_set, defmap=def_set)


# Extend usedef analysis to support openmp_region_start/end nodes.
ir_extension_usedefs[openmp_region_start] = openmp_region_start_defs
ir_extension_usedefs[openmp_region_end] = openmp_region_end_defs


def openmp_region_start_infer(prs, typeinferer):
    pass


def openmp_region_end_infer(pre, typeinferer):
    pass


typeinfer.typeinfer_extensions[openmp_region_start] = openmp_region_start_infer
typeinfer.typeinfer_extensions[openmp_region_end] = openmp_region_end_infer


def _lower_openmp_region_start(lowerer, prs):
    # TODO: if we set it always in numba_fixups we can remove from here
    if isinstance(lowerer.context, OpenmpCPUTargetContext) or isinstance(
        lowerer.context, OpenmpCUDATargetContext
    ):
        pass
    else:
        lowerer.library.__class__ = CustomCPUCodeLibrary
        lowerer.context.__class__ = CustomContext
    prs.lower(lowerer)


def _lower_openmp_region_end(lowerer, pre):
    # TODO: if we set it always in numba_fixups we can remove from here
    if isinstance(lowerer.context, OpenmpCPUTargetContext) or isinstance(
        lowerer.context, OpenmpCUDATargetContext
    ):
        pass
    else:
        lowerer.library.__class__ = CustomCPUCodeLibrary
        lowerer.context.__class__ = CustomContext
    pre.lower(lowerer)


def apply_copies_openmp_region(
    region, var_dict, name_var_table, typemap, calltypes, save_copies
):
    for i in range(len(region.tags)):
        region.tags[i].replace_vars_inner(var_dict)


apply_copy_propagate_extensions[openmp_region_start] = apply_copies_openmp_region
apply_copy_propagate_extensions[openmp_region_end] = apply_copies_openmp_region


def visit_vars_openmp_region(region, callback, cbdata):
    for i in range(len(region.tags)):
        if DEBUG_OPENMP >= 1:
            print("visit_vars before", region.tags[i], type(region.tags[i].arg))
        region.tags[i].arg = visit_vars_inner(region.tags[i].arg, callback, cbdata)
        if DEBUG_OPENMP >= 1:
            print("visit_vars after", region.tags[i])


visit_vars_extensions[openmp_region_start] = visit_vars_openmp_region
visit_vars_extensions[openmp_region_end] = visit_vars_openmp_region

# ----------------------------------------------------------------------------------------------


class PythonOpenmp:
    def __init__(self, *args):
        self.args = args

    def __enter__(self):
        pass

    def __exit__(self, typ, val, tb):
        pass


def iscall(x):
    if isinstance(x, ir.Assign):
        return isinstance(x.value, ir.Expr) and x.value.op == "call"
    elif isinstance(x, ir.Expr):
        return x.op == "call"
    else:
        return False


def extract_args_from_openmp(func_ir):
    """Find all the openmp context calls in the function and then
    use the VarCollector transformer to find all the Python variables
    referenced in the openmp clauses.  We then add those variables as
    regular arguments to the openmp context call just so Numba's
    usedef analysis is able to keep variables alive that are only
    referenced in openmp clauses.
    """
    func_ir._definitions = build_definitions(func_ir.blocks)
    var_table = get_name_var_table(func_ir.blocks)
    for block in func_ir.blocks.values():
        for inst in block.body:
            if iscall(inst):
                func_def = get_definition(func_ir, inst.value.func)
                if isinstance(func_def, ir.Global) and isinstance(
                    func_def.value, _OpenmpContextType
                ):
                    str_def = get_definition(func_ir, inst.value.args[0])
                    if not isinstance(str_def, ir.Const) or not isinstance(
                        str_def.value, str
                    ):
                        # The non-const openmp string error is handled later.
                        continue
                    assert isinstance(str_def, ir.Const) and isinstance(
                        str_def.value, str
                    )
                    parse_res = var_collector_parser.parse(str_def.value)
                    visitor = VarCollector()
                    try:
                        visit_res = visitor.transform(parse_res)
                        inst.value.args.extend([var_table[x] for x in visit_res])
                    except Exception as f:
                        print("generic transform exception")
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        # print("Internal error for OpenMp pragma '{}'".format(arg.value))
                        sys.exit(-2)
                    except:
                        print("fallthrough exception")
                        # print("Internal error for OpenMp pragma '{}'".format(arg.value))
                        sys.exit(-3)


def remove_empty_blocks(blocks):
    found = True
    while found:
        found = False
        empty_block = None
        for label, block in blocks.items():
            if len(block.body) == 1:
                assert isinstance(block.body[-1], ir.Jump)
                empty_block = label
                next_block = block.body[-1].target
                break

        if empty_block is not None:
            del blocks[empty_block]

            found = True
            for block in blocks.values():
                last_stmt = block.body[-1]
                if isinstance(last_stmt, ir.Jump):
                    if last_stmt.target == empty_block:
                        block.body[-1] = ir.Jump(next_block, last_stmt.loc)
                elif isinstance(last_stmt, ir.Branch):
                    if last_stmt.truebr == empty_block:
                        block.body[-1] = ir.Branch(
                            last_stmt.cond, next_block, last_stmt.falsebr, last_stmt.loc
                        )
                    elif block.body[-1].falsebr == empty_block:
                        block.body[-1] = ir.Branch(
                            last_stmt.cond, last_stmt.truebr, next_block, last_stmt.loc
                        )
                elif isinstance(last_stmt, ir.Return):
                    # Intentionally do nothing.
                    pass
                else:
                    print(type(last_stmt))
                    assert False


class _OpenmpContextType(WithContext):
    is_callable = True
    first_time = True
    blk_end_live_map = set()

    def do_numba_fixups(self):
        from numba import core

        orig_lower_inst = core.lowering.Lower.lower_inst
        core.lowering.Lower.orig_lower_inst = orig_lower_inst

        orig_lower = core.lowering.Lower.lower
        core.lowering.Lower.orig_lower = orig_lower

        # Use method to retrieve the outside region live map, which is updated
        # during the with-context mutation.
        def get_blk_end_live_map():
            return self.blk_end_live_map

        def new_lower(self, inst):
            if not isinstance(self, LowerNoSROA):
                self.__class__ = LowerNoSROA
            if isinstance(inst, openmp_region_start):
                return _lower_openmp_region_start(self, inst)
            elif isinstance(inst, openmp_region_end):
                return _lower_openmp_region_end(self, inst)
            # TODO: instead of monkey patching for Del instructions outside the
            # openmp region do: (1) either outline to create a function scope
            # that will decouple the lifetime of variables inside the OpenMP
            # region, (2) or subclass the PostProcessor to extend use-def
            # analysis with OpenMP lifetime information.
            elif isinstance(inst, ir.Del):
                # Lower Del normally in the openmp region.
                if in_openmp_region(self.builder):
                    return self.orig_lower_inst(inst)

                # Lower the Del instruction ONLY if the variable is not live
                # after the openmp region.
                if inst.value not in get_blk_end_live_map():
                    return self.orig_lower_inst(inst)
            elif isinstance(inst, ir.Assign):
                return self.lower_assign_inst(orig_lower_inst, inst)
            elif isinstance(inst, ir.Return):
                return self.lower_return_inst(orig_lower_inst, inst)
            else:
                return self.orig_lower_inst(inst)

        core.lowering.Lower.lower_inst = new_lower

    def mutate_with_body(
        self,
        func_ir,
        blocks,
        blk_start,
        blk_end,
        body_blocks,
        dispatcher_factory,
        extra,
    ):
        if _OpenmpContextType.first_time == True:
            _OpenmpContextType.first_time = False
            self.do_numba_fixups()

        if DEBUG_OPENMP >= 1:
            print("pre-dead-code")
            dump_blocks(blocks)
        if not OPENMP_DISABLED and not hasattr(func_ir, "has_openmp_region"):
            # We can't do dead code elimination at this point because if an argument
            # is used only in an openmp clause then it is detected as dead and is
            # eliminated.  We'd have to run through the IR and find all the
            # openmp regions and extract the vars used there and then modify the
            # IR with something fake just to take the var alive.  The other approach
            # would be to modify dead code elimination to find the vars referenced
            # in openmp context strings.
            extract_args_from_openmp(func_ir)
            # dead_code_elimination(func_ir)
            remove_ssa_from_func_ir(func_ir)
            # remove_empty_blocks(blocks)
            func_ir.has_openmp_region = True
        if DEBUG_OPENMP >= 1:
            print("pre-with-removal")
            dump_blocks(blocks)
        if OPENMP_DISABLED:
            # If OpenMP disabled, do nothing except remove the enter_with marker.
            sblk = blocks[blk_start]
            sblk.body = sblk.body[1:]
        else:
            if DEBUG_OPENMP >= 1:
                print("openmp:mutate_with_body")
                dprint_func_ir(func_ir, "func_ir")
                print("blocks:", blocks, type(blocks))
                print("blk_start:", blk_start, type(blk_start))
                print("blk_end:", blk_end, type(blk_end))
                print("body_blocks:", body_blocks, type(body_blocks))
                print("extra:", extra, type(extra))
            assert extra is not None
            _add_openmp_ir_nodes(
                func_ir, blocks, blk_start, blk_end, body_blocks, extra
            )
            func_ir._definitions = build_definitions(blocks)
            if DEBUG_OPENMP >= 1:
                print("post-with-removal")
                dump_blocks(blocks)
            dispatcher = dispatcher_factory(func_ir)
            dispatcher.can_cache = True

            # Find live variables after the region to make sure we don't Del
            # them if they are defined in the openmp region.
            cfg = compute_cfg_from_blocks(blocks)
            usedefs = compute_use_defs(blocks)
            live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
            self.blk_end_live_map = live_map[blk_end]
            return dispatcher

    def __call__(self, args):
        return PythonOpenmp(args)


def remove_indirections(clause):
    try:
        while len(clause) == 1 and isinstance(clause[0], list):
            clause = clause[0]
    except:
        pass
    return clause


class default_shared_val:
    def __init__(self, val):
        self.val = val


class UnspecifiedVarInDefaultNone(Exception):
    pass


class ParallelForExtraCode(Exception):
    pass


class ParallelForWrongLoopCount(Exception):
    pass


class ParallelForInvalidCollapseCount(Exception):
    pass


class NonconstantOpenmpSpecification(Exception):
    pass


class NonStringOpenmpSpecification(Exception):
    pass


class MultipleNumThreadsClauses(Exception):
    pass


openmp_context = _OpenmpContextType()


def is_dsa(name):
    return (
        name
        in [
            "QUAL.OMP.FIRSTPRIVATE",
            "QUAL.OMP.PRIVATE",
            "QUAL.OMP.SHARED",
            "QUAL.OMP.LASTPRIVATE",
            "QUAL.OMP.TARGET.IMPLICIT",
        ]
        or name.startswith("QUAL.OMP.REDUCTION")
        or name.startswith("QUAL.OMP.MAP")
    )


def get_dotted_type(x, typemap, lowerer):
    xsplit = x.split("*")
    cur_typ = typemap_lookup(typemap, xsplit[0])
    # print("xsplit:", xsplit, cur_typ, type(cur_typ))
    for field in xsplit[1:]:
        dm = lowerer.context.data_model_manager.lookup(cur_typ)
        findex = dm._fields.index(field)
        cur_typ = dm._members[findex]
        # print("dm:", dm, type(dm), dm._members, type(dm._members), dm._fields, type(dm._fields), findex, cur_typ, type(cur_typ))
    return cur_typ


def is_target_arg(name):
    return (
        name in ["QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.TARGET.IMPLICIT"]
        or name.startswith("QUAL.OMP.MAP")
        or name.startswith("QUAL.OMP.REDUCTION")
    )


def is_pointer_target_arg(name, typ):
    if name.startswith("QUAL.OMP.REDUCTION"):
        return True
    if name.startswith("QUAL.OMP.MAP"):
        return True
    if name in ["QUAL.OMP.TARGET.IMPLICIT"]:
        if isinstance(typ, types.npytypes.Array):
            return True

    return False


def is_internal_var(var):
    # Determine if a var is a Python var or an internal Numba var.
    if var.is_temp:
        return True
    return var.unversioned_name != var.name


def remove_ssa(var_name, scope, loc):
    # Get the base name of a variable, removing the SSA extension.
    var = ir.Var(scope, var_name, loc)
    return var.unversioned_name


def user_defined_var(var):
    if not isinstance(var, str):
        return False
    return not var.startswith("$")


def has_user_defined_var(the_set):
    for x in the_set:
        if user_defined_var(x):
            return True
    return False


def get_user_defined_var(the_set):
    ret = set()
    for x in the_set:
        if user_defined_var(x):
            ret.add(x)
    return ret


unique = 0


def get_unique():
    global unique
    ret = unique
    unique += 1
    return ret


def is_private(x):
    return x in [
        "QUAL.OMP.PRIVATE",
        "QUAL.OMP.FIRSTPRIVATE",
        "QUAL.OMP.LASTPRIVATE",
        "QUAL.OMP.TARGET.IMPLICIT",
    ]


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


def replace_ssa_var_callback(var, vardict):
    assert isinstance(var, ir.Var)
    while var.unversioned_name in vardict.keys():
        assert vardict[var.unversioned_name].name != var.unversioned_name
        new_var = vardict[var.unversioned_name]
        var = ir.Var(new_var.scope, new_var.name, new_var.loc)
    return var


def replace_ssa_vars(blocks, vardict):
    """replace variables (ir.Var to ir.Var) from dictionary (name -> ir.Var)"""
    # remove identity values to avoid infinite loop
    new_vardict = {}
    for l, r in vardict.items():
        if l != r.name:
            new_vardict[l] = r
    visit_vars(blocks, replace_ssa_var_callback, new_vardict)


def get_blocks_between_start_end(blocks, start_block, end_block):
    cfg = compute_cfg_from_blocks(blocks)
    blocks_in_region = [start_block]

    def add_in_region(cfg, blk, blocks_in_region, end_block):
        """For each successor in the CFG of the block we're currently
        adding to blocks_in_region, add that successor to
        blocks_in_region if it isn't the end_block.  Then,
        recursively call this routine for the added block to add
        its successors.
        """
        for out_blk, _ in cfg.successors(blk):
            if out_blk != end_block and out_blk not in blocks_in_region:
                blocks_in_region.append(out_blk)
                add_in_region(cfg, out_blk, blocks_in_region, end_block)

    # Calculate all the Numba IR blocks in the target region.
    add_in_region(cfg, start_block, blocks_in_region, end_block)
    return blocks_in_region


class VarName(str):
    pass


class OnlyClauseVar(VarName):
    pass


# This Transformer visitor class just finds the referenced python names
# and puts them in a list of VarName.  The default visitor function
# looks for list of VarNames in the args to that tree node and then
# concatenates them all together.  The final return value is a list of
# VarName that are variables used in the openmp clauses.
class VarCollector(Transformer):
    def __init__(self):
        super(VarCollector, self).__init__()

    def PYTHON_NAME(self, args):
        return [VarName(args)]

    def const_num_or_var(self, args):
        return args[0]

    def num_threads_clause(self, args):
        (_, num_threads) = args
        if isinstance(num_threads, list):
            assert len(num_threads) == 1
            return [OnlyClauseVar(num_threads[0])]
        else:
            return None

    def __default__(self, data, children, meta):
        ret = []
        for c in children:
            if isinstance(c, list) and len(c) > 0:
                if isinstance(c[0], OnlyClauseVar):
                    ret.extend(c)
        return ret


def add_tags_to_enclosing(func_ir, cur_block, tags):
    enclosing_region = get_enclosing_region(func_ir, cur_block)
    if enclosing_region:
        for region in enclosing_region:
            for tag in tags:
                region.add_tag(tag)


def add_enclosing_region(func_ir, blocks, openmp_node):
    if not hasattr(func_ir, "openmp_enclosing"):
        func_ir.openmp_enclosing = {}
    if not hasattr(func_ir, "openmp_regions"):
        func_ir.openmp_regions = {}
    func_ir.openmp_regions[openmp_node] = sorted(blocks)
    for b in blocks:
        if b not in func_ir.openmp_enclosing:
            func_ir.openmp_enclosing[b] = []
        func_ir.openmp_enclosing[b].append(openmp_node)


def get_enclosing_region(func_ir, cur_block):
    if not hasattr(func_ir, "openmp_enclosing"):
        func_ir.openmp_enclosing = {}
    if cur_block in func_ir.openmp_enclosing:
        return func_ir.openmp_enclosing[cur_block]
    else:
        return None


def get_var_from_enclosing(enclosing_regions, var):
    if not enclosing_regions:
        return None
    if len(enclosing_regions) == 0:
        return None
    return enclosing_regions[-1].get_var_dsa(var)


class OpenmpVisitor(Transformer):
    target_num = 0

    def __init__(self, func_ir, blocks, blk_start, blk_end, body_blocks, loc):
        self.func_ir = func_ir
        self.blocks = blocks
        self.blk_start = blk_start
        self.blk_end = blk_end
        self.body_blocks = body_blocks
        self.loc = loc
        super(OpenmpVisitor, self).__init__()

    # --------- Non-parser functions --------------------

    def remove_explicit_from_one(
        self, varset, vars_in_explicit_clauses, clauses, scope, loc
    ):
        """Go through a set of variables and see if their non-SSA form is in an explicitly
        provided data clause.  If so, remove it from the set and add a clause so that the
        SSA form gets the same data clause.
        """
        if DEBUG_OPENMP >= 1:
            print(
                "remove_explicit start:",
                sorted(varset),
                sorted(vars_in_explicit_clauses),
            )
        diff = set()
        # For each variable in the set.
        for v in sorted(varset):
            # Get the non-SSA form.
            flat = remove_ssa(v, scope, loc)
            # Skip non-SSA introduced variables (i.e., Python vars).
            if flat == v:
                continue
            if DEBUG_OPENMP >= 1:
                print("remove_explicit:", v, flat, flat in vars_in_explicit_clauses)
            # If we have the non-SSA form in an explicit data clause.
            if flat in vars_in_explicit_clauses:
                # We will remove it from the set.
                diff.add(v)
                # Copy the non-SSA variables data clause.
                ccopy = copy.copy(vars_in_explicit_clauses[flat])
                # Change the name in the clause to the SSA form.
                ccopy.arg = ir.Var(scope, v, loc)
                # Add to the clause set.
                clauses.append(ccopy)
        # Remove the vars from the set that we added a clause for.
        varset.difference_update(diff)
        if DEBUG_OPENMP >= 1:
            print("remove_explicit end:", sorted(varset))

    def remove_explicit_from_io_vars(
        self,
        inputs_to_region,
        def_but_live_out,
        private_to_region,
        vars_in_explicit_clauses,
        clauses,
        non_user_explicits,
        scope,
        loc,
    ):
        """Remove vars in explicit data clauses from the auto-determined vars.
        Then call remove_explicit_from_one to take SSA variants out of the auto-determined sets
        and to create clauses so that SSA versions get the same clause as the explicit Python non-SSA var.
        """
        inputs_to_region.difference_update(vars_in_explicit_clauses.keys())
        def_but_live_out.difference_update(vars_in_explicit_clauses.keys())
        private_to_region.difference_update(vars_in_explicit_clauses.keys())
        inputs_to_region.difference_update(non_user_explicits.keys())
        def_but_live_out.difference_update(non_user_explicits.keys())
        private_to_region.difference_update(non_user_explicits.keys())
        self.remove_explicit_from_one(
            inputs_to_region, vars_in_explicit_clauses, clauses, scope, loc
        )
        self.remove_explicit_from_one(
            def_but_live_out, vars_in_explicit_clauses, clauses, scope, loc
        )
        self.remove_explicit_from_one(
            private_to_region, vars_in_explicit_clauses, clauses, scope, loc
        )

    def find_io_vars(self, selected_blocks):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        cfg = compute_cfg_from_blocks(self.blocks)
        usedefs = compute_use_defs(self.blocks)
        if DEBUG_OPENMP >= 1:
            print("usedefs:", usedefs)
        live_map = compute_live_map(cfg, self.blocks, usedefs.usemap, usedefs.defmap)
        # Assumes enter_with is first statement in block.
        inputs_to_region = live_map[self.blk_start]
        if DEBUG_OPENMP >= 1:
            print("live_map:", live_map)
            print("inputs_to_region:", sorted(inputs_to_region), type(inputs_to_region))
            print("selected blocks:", sorted(selected_blocks))
        all_uses = set()
        all_defs = set()
        for label in selected_blocks:
            all_uses = all_uses.union(usedefs.usemap[label])
            all_defs = all_defs.union(usedefs.defmap[label])
        # Filter out those vars live to the region but not used within it.
        inputs_to_region = inputs_to_region.intersection(all_uses)
        def_but_live_out = all_defs.difference(inputs_to_region).intersection(
            live_map[self.blk_end]
        )
        private_to_region = all_defs.difference(inputs_to_region).difference(
            live_map[self.blk_end]
        )

        if DEBUG_OPENMP >= 1:
            print("all_uses:", sorted(all_uses))
            print("inputs_to_region:", sorted(inputs_to_region))
            print("private_to_region:", sorted(private_to_region))
            print("def_but_live_out:", sorted(def_but_live_out))
        return inputs_to_region, def_but_live_out, private_to_region, live_map

    def get_explicit_vars(self, clauses):
        user_vars = {}
        non_user_vars = {}
        privates = []
        for c in clauses:
            if DEBUG_OPENMP >= 1:
                print("get_explicit_vars:", c, type(c))
            if isinstance(c, openmp_tag):
                if DEBUG_OPENMP >= 1:
                    print("arg:", c.arg, type(c.arg))
                if isinstance(c.arg, list):
                    carglist = c.arg
                else:
                    carglist = [c.arg]
                # carglist = c.arg if isinstance(c.arg, list) else [c.arg]
                for carg in carglist:
                    if DEBUG_OPENMP >= 1:
                        print(
                            "carg:",
                            carg,
                            type(carg),
                            user_defined_var(carg),
                            is_dsa(c.name),
                        )
                    # Extract the var name from the NameSlice.
                    if isinstance(carg, NameSlice):
                        carg = carg.name
                    if isinstance(carg, str) and is_dsa(c.name):
                        if user_defined_var(carg):
                            user_vars[carg] = c
                            if is_private(c.name):
                                privates.append(carg)
                        else:
                            non_user_vars[carg] = c
        return user_vars, privates, non_user_vars

    def filter_unused_vars(self, clauses, used_vars):
        new_clauses = []
        for c in clauses:
            if DEBUG_OPENMP >= 1:
                print("filter_unused_vars:", c, type(c))
            if isinstance(c, openmp_tag):
                if DEBUG_OPENMP >= 1:
                    print("arg:", c.arg, type(c.arg))
                assert not isinstance(c.arg, list)
                if DEBUG_OPENMP >= 1:
                    print(
                        "c.arg:",
                        c.arg,
                        type(c.arg),
                        user_defined_var(c.arg),
                        is_dsa(c.name),
                    )

                if (
                    isinstance(c.arg, str)
                    and user_defined_var(c.arg)
                    and is_dsa(c.name)
                ):
                    if c.arg in used_vars:
                        new_clauses.append(c)
                else:
                    new_clauses.append(c)
        return new_clauses

    def get_clause_privates(self, clauses, def_but_live_out, scope, loc):
        # Get all the private clauses from the whole set of clauses.
        private_clauses_vars = [
            remove_privatized(x.arg)
            for x in clauses
            if x.name in ["QUAL.OMP.PRIVATE", "QUAL.OMP.FIRSTPRIVATE"]
        ]
        # private_clauses_vars = [remove_privatized(x.arg) for x in clauses if x.name in ["QUAL.OMP.PRIVATE", "QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.LASTPRIVATE"]]
        ret = {}
        # Get a mapping of vars in private clauses to the SSA version of variable exiting the region.
        for lo in def_but_live_out:
            without_ssa = remove_ssa(lo, scope, loc)
            if without_ssa in private_clauses_vars:
                ret[without_ssa] = lo
        return ret

    def make_implicit_explicit(
        self,
        scope,
        vars_in_explicit,
        explicit_clauses,
        gen_shared,
        inputs_to_region,
        def_but_live_out,
        private_to_region,
        for_task=False,
    ):
        if for_task is None:
            for_task = []
        if gen_shared:
            for var_name in sorted(inputs_to_region):
                if (
                    for_task != False
                    and get_var_from_enclosing(for_task, var_name) != "QUAL.OMP.SHARED"
                ):
                    explicit_clauses.append(
                        openmp_tag("QUAL.OMP.FIRSTPRIVATE", var_name)
                    )
                else:
                    explicit_clauses.append(openmp_tag("QUAL.OMP.SHARED", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]

            for var_name in sorted(def_but_live_out):
                if (
                    for_task != False
                    and get_var_from_enclosing(for_task, var_name) != "QUAL.OMP.SHARED"
                ):
                    explicit_clauses.append(
                        openmp_tag("QUAL.OMP.FIRSTPRIVATE", var_name)
                    )
                else:
                    explicit_clauses.append(openmp_tag("QUAL.OMP.SHARED", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]

            # What to do below for task regions?
            for var_name in sorted(private_to_region):
                temp_var = ir.Var(scope, var_name, self.loc)
                if not is_internal_var(temp_var):
                    explicit_clauses.append(openmp_tag("QUAL.OMP.PRIVATE", var_name))
                    vars_in_explicit[var_name] = explicit_clauses[-1]

        for var_name in sorted(private_to_region):
            temp_var = ir.Var(scope, var_name, self.loc)
            if is_internal_var(temp_var):
                explicit_clauses.append(openmp_tag("QUAL.OMP.PRIVATE", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]

    def make_implicit_explicit_target(
        self,
        scope,
        vars_in_explicit,
        explicit_clauses,
        gen_shared,
        inputs_to_region,
        def_but_live_out,
        private_to_region,
    ):
        # unversioned_privates = set() # we get rid of SSA on the first openmp region so no SSA forms should be here
        if gen_shared:
            for var_name in sorted(inputs_to_region):
                explicit_clauses.append(
                    openmp_tag(
                        "QUAL.OMP.TARGET.IMPLICIT"
                        if user_defined_var(var_name)
                        else "QUAL.OMP.PRIVATE",
                        var_name,
                    )
                )
                vars_in_explicit[var_name] = explicit_clauses[-1]
            for var_name in sorted(def_but_live_out):
                explicit_clauses.append(
                    openmp_tag(
                        "QUAL.OMP.TARGET.IMPLICIT"
                        if user_defined_var(var_name)
                        else "QUAL.OMP.PRIVATE",
                        var_name,
                    )
                )
                vars_in_explicit[var_name] = explicit_clauses[-1]
            for var_name in sorted(private_to_region):
                temp_var = ir.Var(scope, var_name, self.loc)
                if not is_internal_var(temp_var):
                    explicit_clauses.append(openmp_tag("QUAL.OMP.PRIVATE", var_name))
                    # explicit_clauses.append(openmp_tag("QUAL.OMP.TARGET.IMPLICIT" if user_defined_var(var_name) else "QUAL.OMP.PRIVATE", var_name))
                    vars_in_explicit[var_name] = explicit_clauses[-1]

        for var_name in sorted(private_to_region):
            temp_var = ir.Var(scope, var_name, self.loc)
            if is_internal_var(temp_var):
                explicit_clauses.append(
                    openmp_tag(
                        "QUAL.OMP.TARGET.IMPLICIT"
                        if user_defined_var(var_name)
                        else "QUAL.OMP.PRIVATE",
                        var_name,
                    )
                )
                vars_in_explicit[var_name] = explicit_clauses[-1]

    def add_explicits_to_start(
        self,
        scope,
        vars_in_explicit,
        explicit_clauses,
        gen_shared,
        start_tags,
        keep_alive,
    ):
        start_tags.extend(explicit_clauses)
        return []
        # tags_for_enclosing = []
        # for var in vars_in_explicit:
        #    if not is_private(vars_in_explicit[var].name):
        #        print("EVAR_COPY FOR", var)
        #        evar = ir.Var(scope, var, self.loc)
        #        evar_copy = scope.redefine("evar_copy_aets", self.loc)
        #        keep_alive.append(ir.Assign(evar, evar_copy, self.loc))
        #        #keep_alive.append(ir.Assign(evar, evar, self.loc))
        #        tags_for_enclosing.append(openmp_tag("QUAL.OMP.PRIVATE", evar_copy))
        # return tags_for_enclosing

    def flatten(self, all_clauses, start_block):
        if DEBUG_OPENMP >= 1:
            print("flatten", id(start_block))
        incoming_clauses = [remove_indirections(x) for x in all_clauses]
        clauses = []
        default_shared = True
        for clause in incoming_clauses:
            if DEBUG_OPENMP >= 1:
                print("clause:", clause, type(clause))
            if isinstance(clause, openmp_tag):
                clauses.append(clause)
            elif isinstance(clause, list):
                clauses.extend(remove_indirections(clause))
            elif clause == "nowait":
                clauses.append(openmp_tag("QUAL.OMP.NOWAIT"))
            elif isinstance(clause, default_shared_val):
                default_shared = clause.val
                if DEBUG_OPENMP >= 1:
                    print("got new default_shared:", clause.val)
            else:
                if DEBUG_OPENMP >= 1:
                    print(
                        "Unknown clause type in incoming_clauses", clause, type(clause)
                    )
                assert 0

        if hasattr(start_block, "openmp_replace_vardict"):
            for clause in clauses:
                # print("flatten out clause:", clause, clause.arg, type(clause.arg))
                for vardict in start_block.openmp_replace_vardict:
                    if clause.arg in vardict:
                        # print("clause.arg in vardict:", clause.arg, type(clause.arg), vardict[clause.arg], type(vardict[clause.arg]))
                        clause.arg = vardict[clause.arg].name

        return clauses, default_shared

    def add_replacement(self, blocks, replace_vardict):
        for b in blocks.values():
            if not hasattr(b, "openmp_replace_vardict"):
                b.openmp_replace_vardict = []
            b.openmp_replace_vardict.append(replace_vardict)

    def make_consts_unliteral_for_privates(self, privates, blocks):
        for blk in blocks.values():
            for stmt in blk.body:
                if (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Const)
                    and stmt.target.name in privates
                ):
                    stmt.value.use_literal_type = False

    def fix_empty_header(self, block, label):
        if len(block.body) == 1:
            assert isinstance(block.body[0], ir.Jump)
            return self.blocks[block.body[0].target], block.body[0].target
        return block, label

    def prepare_for_directive(
        self,
        clauses,
        vars_in_explicit_clauses,
        before_start,
        after_start,
        start_tags,
        end_tags,
        scope,
    ):
        start_tags = clauses
        call_table, _ = get_call_table(self.blocks)
        cfg = compute_cfg_from_blocks(self.blocks)
        usedefs = compute_use_defs(self.blocks)
        live_map = compute_live_map(cfg, self.blocks, usedefs.usemap, usedefs.defmap)

        def get_loops_in_region(all_loops):
            loops = {}
            for k, v in all_loops.items():
                if v.header >= self.blk_start and v.header <= self.blk_end:
                    loops[k] = v
            return loops

        all_loops = cfg.loops()
        if DEBUG_OPENMP >= 1:
            print("all_loops:", all_loops)
            print("live_map:", live_map)
            print("body_blocks:", self.body_blocks)

        loops = get_loops_in_region(all_loops)
        # Find the outer-most loop in this OpenMP region.
        loops = list(filter_nested_loops(cfg, loops))

        if DEBUG_OPENMP >= 1:
            print("loops:", loops)
        if len(loops) != 1:
            raise ParallelForWrongLoopCount(
                f"OpenMP parallel for regions must contain exactly one range based loop.  The parallel for at line {self.loc} contains {len(loops)} loops."
            )

        collapse_tags = get_tags_of_type(clauses, "QUAL.OMP.COLLAPSE")
        new_stmts_for_iterspace = []
        collapse_iterspace_block = set()
        iterspace_vars = []
        if len(collapse_tags) > 0:
            # Limit all_loops to just loops within the openmp region.
            all_loops = get_loops_in_region(all_loops)
            # In case of multiple collapse tags, use the last one.
            collapse_tag = collapse_tags[-1]
            # Remove collapse tags from clauses so they don't go to LLVM pass.
            clauses[:] = [x for x in clauses if x not in collapse_tags]
            # Add top level loop to loop_order list.
            loop_order = list(filter_nested_loops(cfg, all_loops))
            if len(loop_order) != 1:
                raise ParallelForWrongLoopCount(
                    f"OpenMP parallel for region must have only one top-level loop at line {self.loc}."
                )
            # Determine how many nested loops we need to process.
            collapse_value = collapse_tag.arg - 1
            # Make sure initial collapse value was >= 2.
            if collapse_value <= 0:
                raise ParallelForInvalidCollapseCount(
                    f"OpenMP parallel for regions with collapse clauses must be greather than or equal to 2 at line {self.loc}."
                )

            # Delete top-level loop from all_loops.
            del all_loops[loop_order[-1].header]
            # For remaining nested loops...
            for _ in range(collapse_value):
                # Get the next most top-level loop.
                loops = list(filter_nested_loops(cfg, all_loops))
                # Make sure there is only one.
                if len(loops) != 1:
                    raise ParallelForWrongLoopCount(
                        f"OpenMP parallel for collapse regions must be perfectly nested for the parallel for at line {self.loc}."
                    )
                # Add this loop to the loops to process in order.
                loop_order.append(loops[0])
                # Delete this loop from all_loops.
                del all_loops[loop_order[-1].header]

            if DEBUG_OPENMP >= 2:
                print("loop_order:", loop_order)
            stmts_to_retain = []
            loop_bounds = []
            for loop in loop_order:
                loop_entry = list(loop.entries)[0]
                loop_exit = list(loop.exits)[0]
                loop_header = loop.header
                loop_entry_block = self.blocks[loop_entry]
                loop_exit_block = self.blocks[loop_exit]
                loop_header_block, _ = self.fix_empty_header(
                    self.blocks[loop_header], loop_header
                )

                # Copy all stmts from the loop entry block up to the ir.Global
                # for range.
                call_offset = None
                for entry_block_index, stmt in enumerate(loop_entry_block.body):
                    found_range = False
                    if (
                        isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Global)
                        and stmt.value.name == "range"
                    ):
                        found_range = True
                        range_target = stmt.target
                        found_call = False
                        for call_index in range(
                            entry_block_index + 1, len(loop_entry_block.body)
                        ):
                            call_stmt = loop_entry_block.body[call_index]
                            if (
                                isinstance(call_stmt, ir.Assign)
                                and isinstance(call_stmt.value, ir.Expr)
                                and call_stmt.value.op == "call"
                                and call_stmt.value.func == range_target
                            ):
                                found_call = True
                                # Remove stmts that were retained.
                                loop_entry_block.body = loop_entry_block.body[
                                    entry_block_index:
                                ]
                                call_offset = call_index - entry_block_index
                                break
                        assert found_call
                        break
                    stmts_to_retain.append(stmt)
                assert found_range
                for header_block_index, stmt in enumerate(loop_header_block.body):
                    if (
                        isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == "iternext"
                    ):
                        iternext_inst = loop_header_block.body[header_block_index]
                        pair_first_inst = loop_header_block.body[header_block_index + 1]
                        pair_second_inst = loop_header_block.body[
                            header_block_index + 2
                        ]

                        assert (
                            isinstance(iternext_inst, ir.Assign)
                            and isinstance(iternext_inst.value, ir.Expr)
                            and iternext_inst.value.op == "iternext"
                        )
                        assert (
                            isinstance(pair_first_inst, ir.Assign)
                            and isinstance(pair_first_inst.value, ir.Expr)
                            and pair_first_inst.value.op == "pair_first"
                        )
                        assert (
                            isinstance(pair_second_inst, ir.Assign)
                            and isinstance(pair_second_inst.value, ir.Expr)
                            and pair_second_inst.value.op == "pair_second"
                        )
                        stmts_to_retain.extend(
                            loop_header_block.body[header_block_index + 3 : -1]
                        )
                        loop_index = pair_first_inst.target
                        break
                    stmts_to_retain.append(stmt)
                loop_bounds.append((call_stmt.value.args[0], loop_index))
            if DEBUG_OPENMP >= 1:
                print("collapse 1")
                dump_blocks(self.blocks)
            # For all the loops except the last...
            for loop in loop_order[:-1]:
                # Change the unneeded headers to just jump to the next block.
                loop_header = loop.header
                loop_header_block, real_loop_header = self.fix_empty_header(
                    self.blocks[loop_header], loop_header
                )
                collapse_iterspace_block.add(real_loop_header)
                loop_header_block.body[-1] = ir.Jump(
                    loop_header_block.body[-1].truebr, loop_header_block.body[-1].loc
                )
                last_eliminated_loop_header_block = loop_header_block
                self.body_blocks = [
                    x for x in self.body_blocks if x not in loop.entries
                ]
                self.body_blocks.remove(loop.header)
            if DEBUG_OPENMP >= 1:
                print("loop order:", loop_order)
                print("loop bounds:", loop_bounds)
                print("collapse 2")
                dump_blocks(self.blocks)
            last_loop = loop_order[-1]
            last_loop_entry = list(last_loop.entries)[0]
            last_loop_exit = list(last_loop.exits)[0]
            last_loop_header = last_loop.header
            last_loop_entry_block = self.blocks[last_loop_entry]
            last_loop_exit_block = self.blocks[last_loop_exit]
            last_loop_header_block, _ = self.fix_empty_header(
                self.blocks[last_loop_header], loop_header
            )
            last_loop_first_body_block = last_loop_header_block.body[-1].truebr
            self.blocks[last_loop_first_body_block].body = (
                stmts_to_retain + self.blocks[last_loop_first_body_block].body
            )
            last_loop_header_block.body[-1].falsebr = list(loop_order[0].exits)[0]
            new_var_scope = last_loop_entry_block.body[0].target.scope

            # -------- Add vars to remember cumulative product of iteration space sizes.
            new_iterspace_var = new_var_scope.redefine("new_iterspace0", self.loc)
            start_tags.append(
                openmp_tag("QUAL.OMP.FIRSTPRIVATE", new_iterspace_var.name)
            )
            iterspace_vars.append(new_iterspace_var)
            new_stmts_for_iterspace.append(
                ir.Assign(loop_bounds[0][0], new_iterspace_var, self.loc)
            )
            for lb_num, loop_bound in enumerate(loop_bounds[1:]):
                mul_op = ir.Expr.binop(
                    operator.mul, new_iterspace_var, loop_bound[0], self.loc
                )
                new_iterspace_var = new_var_scope.redefine(
                    "new_iterspace" + str(lb_num + 1), self.loc
                )
                start_tags.append(
                    openmp_tag("QUAL.OMP.FIRSTPRIVATE", new_iterspace_var.name)
                )
                iterspace_vars.append(new_iterspace_var)
                new_stmts_for_iterspace.append(
                    ir.Assign(mul_op, new_iterspace_var, self.loc)
                )
            # Change iteration space of innermost loop to the product of all the
            # loops' iteration spaces.
            last_loop_entry_block.body[call_offset].value.args[0] = new_iterspace_var

            last_eliminated_loop_header_block.body = (
                new_stmts_for_iterspace + last_eliminated_loop_header_block.body
            )

            deconstruct_indices = []
            new_deconstruct_var = new_var_scope.redefine("deconstruct", self.loc)
            deconstruct_indices.append(
                ir.Assign(loop_bounds[-1][1], new_deconstruct_var, self.loc)
            )
            for deconstruct_index in range(len(loop_bounds) - 1):
                cur_iterspace_var = iterspace_vars[
                    len(loop_bounds) - 2 - deconstruct_index
                ]
                cur_loop_bound = loop_bounds[deconstruct_index][1]
                # if DEBUG_OPENMP >= 1:
                #    print("deconstructing", cur_iterspace_var)
                #    deconstruct_indices.append(ir.Print([new_deconstruct_var, cur_iterspace_var], None, self.loc))
                deconstruct_div = ir.Expr.binop(
                    operator.floordiv, new_deconstruct_var, cur_iterspace_var, self.loc
                )
                new_deconstruct_var_loop = new_var_scope.redefine(
                    "deconstruct" + str(deconstruct_index), self.loc
                )
                deconstruct_indices.append(
                    ir.Assign(deconstruct_div, cur_loop_bound, self.loc)
                )
                # if DEBUG_OPENMP >= 1:
                #    deconstruct_indices.append(ir.Print([cur_loop_bound], None, self.loc))
                new_deconstruct_var_mul = new_var_scope.redefine(
                    "deconstruct_mul" + str(deconstruct_index), self.loc
                )
                deconstruct_indices.append(
                    ir.Assign(
                        ir.Expr.binop(
                            operator.mul, cur_loop_bound, cur_iterspace_var, self.loc
                        ),
                        new_deconstruct_var_mul,
                        self.loc,
                    )
                )
                # if DEBUG_OPENMP >= 1:
                #    deconstruct_indices.append(ir.Print([new_deconstruct_var_mul], None, self.loc))
                deconstruct_indices.append(
                    ir.Assign(
                        ir.Expr.binop(
                            operator.sub,
                            new_deconstruct_var,
                            new_deconstruct_var_mul,
                            self.loc,
                        ),
                        new_deconstruct_var_loop,
                        self.loc,
                    )
                )
                # if DEBUG_OPENMP >= 1:
                #    deconstruct_indices.append(ir.Print([new_deconstruct_var_loop], None, self.loc))
                new_deconstruct_var = new_deconstruct_var_loop
            deconstruct_indices.append(
                ir.Assign(new_deconstruct_var, loop_bounds[-1][1], self.loc)
            )

            self.blocks[last_loop_first_body_block].body = (
                deconstruct_indices + self.blocks[last_loop_first_body_block].body
            )

            if DEBUG_OPENMP >= 1:
                print("collapse 3", self.blk_start, self.blk_end)
                dump_blocks(self.blocks)

            cfg = compute_cfg_from_blocks(self.blocks)
            live_map = compute_live_map(
                cfg, self.blocks, usedefs.usemap, usedefs.defmap
            )
            all_loops = cfg.loops()
            loops = get_loops_in_region(all_loops)
            loops = list(filter_nested_loops(cfg, loops))
            if DEBUG_OPENMP >= 2:
                print("loops after collapse:", loops)
            if DEBUG_OPENMP >= 1:
                print("blocks after collapse", self.blk_start, self.blk_end)
                dump_blocks(self.blocks)

        def _get_loop_kind(func_var, call_table):
            if func_var not in call_table:
                return False
            call = call_table[func_var]
            if len(call) == 0:
                return False

            return call[0]

        loop = loops[0]
        entry = list(loop.entries)[0]
        header = loop.header
        exit = list(loop.exits)[0]

        loop_blocks_for_io = loop.entries.union(loop.body)
        loop_blocks_for_io_minus_entry = loop_blocks_for_io - {entry}
        non_loop_blocks = set(self.body_blocks)
        non_loop_blocks.difference_update(loop_blocks_for_io)
        non_loop_blocks.difference_update(collapse_iterspace_block)
        # non_loop_blocks.difference_update({exit})

        if DEBUG_OPENMP >= 1:
            print("non_loop_blocks:", non_loop_blocks)
            print("entry:", entry)
            print("header:", header)
            print("exit:", exit)
            print("body_blocks:", self.body_blocks)
            print("loop:", loop)

        # Find the first statement after any iterspace calculation ones for collapse.
        first_stmt = self.blocks[entry].body[0]
        # first_stmt = self.blocks[entry].body[len(new_stmts_for_iterspace)]
        if (
            not isinstance(first_stmt, ir.Assign)
            or not isinstance(first_stmt.value, ir.Global)
            or first_stmt.value.name != "range"
        ):
            raise ParallelForExtraCode(
                f"Extra code near line {self.loc} is not allowed before or after the loop in an OpenMP parallel for region."
            )

        live_end = live_map[self.blk_end]
        for non_loop_block in non_loop_blocks:
            nlb = self.blocks[non_loop_block]
            if isinstance(nlb.body[0], ir.Jump):
                # Non-loop empty blocks are fine.
                continue
            if (
                isinstance(nlb.body[-1], ir.Jump)
                and nlb.body[-1].target == self.blk_end
            ):
                # Loop through all statements in block that jumps to the end of the region.
                # If those are all assignments where the LHS is dead then they are safe.
                for nlb_stmt in nlb.body[:-1]:
                    if isinstance(nlb_stmt, ir.PopBlock):
                        continue

                    break
                    # if not isinstance(nlb_stmt, ir.Assign):
                    #    break  # Non-assignment is not known to be safe...will fallthrough to raise exception.
                    # if nlb_stmt.target.name in live_end:
                    #    break  # Non-dead variables in assignment is not safe...will fallthrough to raise exception.
                else:
                    continue
            raise ParallelForExtraCode(
                f"Extra code near line {self.loc} is not allowed before or after the loop in an OpenMP parallel for region."
            )

        if DEBUG_OPENMP >= 1:
            print("loop_blocks_for_io:", loop_blocks_for_io, entry, exit)
            print("non_loop_blocks:", non_loop_blocks)
            print("header:", header)

        entry_block = self.blocks[entry]
        assert isinstance(entry_block.body[-1], ir.Jump)
        assert entry_block.body[-1].target == header
        exit_block = self.blocks[exit]
        header_block = self.blocks[header]
        extra_block = (
            None if len(header_block.body) > 1 else header_block.body[-1].target
        )

        latch_block_num = max(self.blocks.keys()) + 1

        # We have to reformat the Numba style of loop to the form that the LLVM
        # openmp pass supports.
        header_preds = [x[0] for x in cfg.predecessors(header)]
        entry_preds = list(set(header_preds).difference(loop.body))
        back_blocks = list(set(header_preds).intersection(loop.body))
        if DEBUG_OPENMP >= 1:
            print("header_preds:", header_preds)
            print("entry_preds:", entry_preds)
            print("back_blocks:", back_blocks)
        assert len(entry_preds) == 1
        entry_pred_label = entry_preds[0]
        entry_pred = self.blocks[entry_pred_label]
        if extra_block is not None:
            header_block = self.blocks[extra_block]
            header = extra_block
        header_branch = header_block.body[-1]
        post_header = {header_branch.truebr, header_branch.falsebr}
        post_header.remove(exit)
        if DEBUG_OPENMP >= 1:
            print("post_header:", post_header)
        post_header = self.blocks[list(post_header)[0]]
        if DEBUG_OPENMP >= 1:
            print("post_header:", post_header)

        normalized = True

        for inst_num, inst in enumerate(entry_block.body):
            if (
                isinstance(inst, ir.Assign)
                and isinstance(inst.value, ir.Expr)
                and inst.value.op == "call"
            ):
                loop_kind = _get_loop_kind(inst.value.func.name, call_table)
                if DEBUG_OPENMP >= 1:
                    print("loop_kind:", loop_kind)
                if loop_kind != False and loop_kind == range:
                    range_inst = inst
                    range_args = inst.value.args
                    if DEBUG_OPENMP >= 1:
                        print("found one", loop_kind, inst, range_args)

                    # ----------------------------------------------
                    # Find getiter instruction for this range.
                    for entry_inst in entry_block.body[inst_num + 1 :]:
                        if (
                            isinstance(entry_inst, ir.Assign)
                            and isinstance(entry_inst.value, ir.Expr)
                            and entry_inst.value.op == "getiter"
                            and entry_inst.value.value == range_inst.target
                        ):
                            getiter_inst = entry_inst
                            break
                    assert getiter_inst
                    if DEBUG_OPENMP >= 1:
                        print("getiter_inst:", getiter_inst)
                    # ----------------------------------------------

                    assert len(header_block.body) > 3
                    if DEBUG_OPENMP >= 1:
                        print("header block before removing Numba range vars:")
                        dump_block(header, header_block)

                    for ii in range(len(header_block.body)):
                        ii_inst = header_block.body[ii]
                        if (
                            isinstance(ii_inst, ir.Assign)
                            and isinstance(ii_inst.value, ir.Expr)
                            and ii_inst.value.op == "iternext"
                        ):
                            iter_num = ii
                            break

                    iternext_inst = header_block.body[iter_num]
                    pair_first_inst = header_block.body[iter_num + 1]
                    pair_second_inst = header_block.body[iter_num + 2]

                    assert (
                        isinstance(iternext_inst, ir.Assign)
                        and isinstance(iternext_inst.value, ir.Expr)
                        and iternext_inst.value.op == "iternext"
                    )
                    assert (
                        isinstance(pair_first_inst, ir.Assign)
                        and isinstance(pair_first_inst.value, ir.Expr)
                        and pair_first_inst.value.op == "pair_first"
                    )
                    assert (
                        isinstance(pair_second_inst, ir.Assign)
                        and isinstance(pair_second_inst.value, ir.Expr)
                        and pair_second_inst.value.op == "pair_second"
                    )
                    # Remove those nodes from the IR.
                    header_block.body = (
                        header_block.body[:iter_num] + header_block.body[iter_num + 3 :]
                    )
                    if DEBUG_OPENMP >= 1:
                        print("header block after removing Numba range vars:")
                        dump_block(header, header_block)

                    loop_index = pair_first_inst.target
                    if DEBUG_OPENMP >= 1:
                        print("loop_index:", loop_index, type(loop_index))
                    # The loop_index from Numba's perspective is not what it is from the
                    # programmer's perspective.  The OpenMP loop index is always private so
                    # we need to start from Numba's loop index (e.g., $48for_iter.3) and
                    # trace assignments from that through the header block and then find
                    # the first such assignment in the first loop block that the header
                    # branches to.
                    latest_index = loop_index
                    for hinst in header_block.body:
                        if isinstance(hinst, ir.Assign) and isinstance(
                            hinst.value, ir.Var
                        ):
                            if hinst.value.name == latest_index.name:
                                latest_index = hinst.target
                    for phinst in post_header.body:
                        if isinstance(phinst, ir.Assign) and isinstance(
                            phinst.value, ir.Var
                        ):
                            if phinst.value.name == latest_index.name:
                                latest_index = phinst.target
                                break
                    if DEBUG_OPENMP >= 1:
                        print("latest_index:", latest_index, type(latest_index))

                    if latest_index.name not in vars_in_explicit_clauses:
                        new_index_clause = openmp_tag(
                            "QUAL.OMP.PRIVATE",
                            ir.Var(loop_index.scope, latest_index.name, inst.loc),
                        )
                        clauses.append(new_index_clause)
                        vars_in_explicit_clauses[latest_index.name] = new_index_clause
                    else:
                        if (
                            vars_in_explicit_clauses[latest_index.name].name
                            != "QUAL.OMP.PRIVATE"
                        ):
                            pass
                            # throw error?  FIX ME

                    if DEBUG_OPENMP >= 1:
                        for clause in clauses:
                            print("post-latest_index clauses:", clause)

                    start = 0
                    step = 1
                    size_var = range_args[0]
                    if len(range_args) == 2:
                        start = range_args[0]
                        size_var = range_args[1]
                    if len(range_args) == 3:
                        start = range_args[0]
                        size_var = range_args[1]
                        try:
                            step = self.func_ir.get_definition(range_args[2])
                            # Only use get_definition to get a const if
                            # available.  Otherwise use the variable.
                            if not isinstance(step, (int, ir.Const)):
                                step = range_args[2]
                        except KeyError:
                            # If there is more than one definition possible for the
                            # step variable then just use the variable and don't try
                            # to convert to a const.
                            step = range_args[2]
                        if isinstance(step, ir.Const):
                            step = step.value

                    if DEBUG_OPENMP >= 1:
                        print("size_var:", size_var, type(size_var))

                    omp_lb_var = loop_index.scope.redefine("$omp_lb", inst.loc)
                    before_start.append(
                        ir.Assign(ir.Const(0, inst.loc), omp_lb_var, inst.loc)
                    )

                    omp_iv_var = loop_index.scope.redefine("$omp_iv", inst.loc)
                    # before_start.append(ir.Assign(omp_lb_var, omp_iv_var, inst.loc))
                    # Don't use omp_lb here because that makes a live-in to the region that
                    # becomes a parameter to an outlined target region.
                    after_start.append(
                        ir.Assign(ir.Const(0, inst.loc), omp_iv_var, inst.loc)
                    )
                    # after_start.append(ir.Assign(omp_lb_var, omp_iv_var, inst.loc))

                    types_mod_var = loop_index.scope.redefine(
                        "$numba_types_mod", inst.loc
                    )
                    types_mod = ir.Global("types", types, inst.loc)
                    types_mod_assign = ir.Assign(types_mod, types_mod_var, inst.loc)
                    before_start.append(types_mod_assign)

                    int64_var = loop_index.scope.redefine("$int64_var", inst.loc)
                    int64_getattr = ir.Expr.getattr(types_mod_var, "int64", inst.loc)
                    int64_assign = ir.Assign(int64_getattr, int64_var, inst.loc)
                    before_start.append(int64_assign)

                    get_itercount_var = loop_index.scope.redefine(
                        "$get_itercount", inst.loc
                    )
                    get_itercount_global = ir.Global(
                        "get_itercount", get_itercount, inst.loc
                    )
                    get_itercount_assign = ir.Assign(
                        get_itercount_global, get_itercount_var, inst.loc
                    )
                    before_start.append(get_itercount_assign)

                    itercount_var = loop_index.scope.redefine("$itercount", inst.loc)
                    itercount_expr = ir.Expr.call(
                        get_itercount_var, [getiter_inst.target], (), inst.loc
                    )
                    # itercount_expr = ir.Expr.itercount(getiter_inst.target, inst.loc)
                    before_start.append(
                        ir.Assign(itercount_expr, itercount_var, inst.loc)
                    )

                    omp_ub_var = loop_index.scope.redefine("$omp_ub", inst.loc)
                    omp_ub_expr = ir.Expr.call(int64_var, [itercount_var], (), inst.loc)
                    before_start.append(ir.Assign(omp_ub_expr, omp_ub_var, inst.loc))

                    const1_var = loop_index.scope.redefine("$const1", inst.loc)
                    start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", const1_var))
                    const1_assign = ir.Assign(
                        ir.Const(1, inst.loc), const1_var, inst.loc
                    )
                    before_start.append(const1_assign)
                    count_add_1 = ir.Expr.binop(
                        operator.sub, omp_ub_var, const1_var, inst.loc
                    )
                    before_start.append(ir.Assign(count_add_1, omp_ub_var, inst.loc))

                    #                    before_start.append(ir.Print([omp_ub_var], None, inst.loc))

                    omp_start_var = loop_index.scope.redefine("$omp_start", inst.loc)
                    if start == 0:
                        start = ir.Const(start, inst.loc)
                    before_start.append(ir.Assign(start, omp_start_var, inst.loc))

                    # ---------- Create latch block -------------------------------
                    latch_iv = omp_iv_var

                    latch_block = ir.Block(scope, inst.loc)
                    const1_latch_var = loop_index.scope.redefine(
                        "$const1_latch", inst.loc
                    )
                    start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", const1_latch_var))
                    const1_assign = ir.Assign(
                        ir.Const(1, inst.loc), const1_latch_var, inst.loc
                    )
                    latch_block.body.append(const1_assign)
                    latch_assign = ir.Assign(
                        ir.Expr.binop(
                            operator.add, omp_iv_var, const1_latch_var, inst.loc
                        ),
                        latch_iv,
                        inst.loc,
                    )
                    latch_block.body.append(latch_assign)
                    latch_block.body.append(ir.Jump(header, inst.loc))

                    self.blocks[latch_block_num] = latch_block
                    for bb in back_blocks:
                        if False:
                            str_var = scope.redefine("$str_var", inst.loc)
                            str_const = ir.Const("mid start:", inst.loc)
                            str_assign = ir.Assign(str_const, str_var, inst.loc)
                            str_print = ir.Print([str_var, size_var], None, inst.loc)
                            # before_start.append(str_assign)
                            # before_start.append(str_print)
                            self.blocks[bb].body = self.blocks[bb].body[:-1] + [
                                str_assign,
                                str_print,
                                ir.Jump(latch_block_num, inst.loc),
                            ]
                        else:
                            self.blocks[bb].body[-1] = ir.Jump(
                                latch_block_num, inst.loc
                            )
                    # -------------------------------------------------------------

                    # ---------- Header Manipulation ------------------------------
                    step_var = loop_index.scope.redefine("$step_var", inst.loc)
                    detect_step_assign = ir.Assign(
                        ir.Const(0, inst.loc), step_var, inst.loc
                    )
                    after_start.append(detect_step_assign)

                    if isinstance(step, int):
                        step_assign = ir.Assign(
                            ir.Const(step, inst.loc), step_var, inst.loc
                        )
                    elif isinstance(step, ir.Var):
                        step_assign = ir.Assign(step, step_var, inst.loc)
                        start_tags.append(
                            openmp_tag("QUAL.OMP.FIRSTPRIVATE", step.name)
                        )
                    else:
                        print("Unsupported step:", step, type(step))
                        raise NotImplementedError(
                            f"Unknown step type that isn't a constant or variable but {type(step)} instead."
                        )
                    scale_var = loop_index.scope.redefine("$scale", inst.loc)
                    fake_iternext = ir.Assign(
                        ir.Const(0, inst.loc), iternext_inst.target, inst.loc
                    )
                    fake_second = ir.Assign(
                        ir.Const(0, inst.loc), pair_second_inst.target, inst.loc
                    )
                    scale_assign = ir.Assign(
                        ir.Expr.binop(operator.mul, step_var, omp_iv_var, inst.loc),
                        scale_var,
                        inst.loc,
                    )
                    unnormalize_iv = ir.Assign(
                        ir.Expr.binop(operator.add, omp_start_var, scale_var, inst.loc),
                        loop_index,
                        inst.loc,
                    )
                    cmp_var = loop_index.scope.redefine("$cmp", inst.loc)
                    iv_lte_ub = ir.Assign(
                        ir.Expr.binop(operator.le, omp_iv_var, omp_ub_var, inst.loc),
                        cmp_var,
                        inst.loc,
                    )
                    old_branch = header_block.body[-1]
                    new_branch = ir.Branch(
                        cmp_var, old_branch.truebr, old_branch.falsebr, old_branch.loc
                    )
                    body_label = old_branch.truebr
                    first_body_block = self.blocks[body_label]
                    new_end = [iv_lte_ub, new_branch]
                    # Turn this on to add printing to help debug at runtime.
                    if False:
                        str_var = loop_index.scope.redefine("$str_var", inst.loc)
                        str_const = ir.Const("header1:", inst.loc)
                        str_assign = ir.Assign(str_const, str_var, inst.loc)
                        new_end.append(str_assign)
                        str_print = ir.Print(
                            [str_var, omp_start_var, omp_iv_var], None, inst.loc
                        )
                        new_end.append(str_print)

                    # Prepend original contents of header into the first body block minus the comparison
                    first_body_block.body = (
                        [
                            fake_iternext,
                            fake_second,
                            step_assign,
                            scale_assign,
                            unnormalize_iv,
                        ]
                        + header_block.body[:-1]
                        + first_body_block.body
                    )

                    header_block.body = new_end
                    # header_block.body = [fake_iternext, fake_second, unnormalize_iv] + header_block.body[:-1] + new_end

                    # -------------------------------------------------------------

                    # const_start_var = loop_index.scope.redefine("$const_start", inst.loc)
                    # before_start.append(ir.Assign(ir.Const(0, inst.loc), const_start_var, inst.loc))
                    # start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", const_start_var.name))
                    start_tags.append(
                        openmp_tag("QUAL.OMP.NORMALIZED.IV", omp_iv_var.name)
                    )
                    start_tags.append(
                        openmp_tag("QUAL.OMP.NORMALIZED.START", omp_start_var.name)
                    )
                    start_tags.append(
                        openmp_tag("QUAL.OMP.NORMALIZED.LB", omp_lb_var.name)
                    )
                    start_tags.append(
                        openmp_tag("QUAL.OMP.NORMALIZED.UB", omp_ub_var.name)
                    )
                    start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", omp_iv_var.name))
                    start_tags.append(
                        openmp_tag("QUAL.OMP.FIRSTPRIVATE", omp_start_var.name)
                    )
                    start_tags.append(
                        openmp_tag("QUAL.OMP.FIRSTPRIVATE", omp_lb_var.name)
                    )
                    start_tags.append(
                        openmp_tag("QUAL.OMP.FIRSTPRIVATE", omp_ub_var.name)
                    )
                    tags_for_enclosing = [
                        cmp_var.name,
                        omp_lb_var.name,
                        omp_start_var.name,
                        omp_iv_var.name,
                        types_mod_var.name,
                        int64_var.name,
                        itercount_var.name,
                        omp_ub_var.name,
                        const1_var.name,
                        const1_latch_var.name,
                        get_itercount_var.name,
                    ] + [x.name for x in iterspace_vars]
                    tags_for_enclosing = [
                        openmp_tag("QUAL.OMP.PRIVATE", x) for x in tags_for_enclosing
                    ]
                    # Don't blindly copy code here...this isn't doing what the other spots are doing with privatization.
                    add_tags_to_enclosing(
                        self.func_ir, self.blk_start, tags_for_enclosing
                    )
                    # start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.IV", loop_index.name))
                    # start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.UB", size_var.name))
                    return (
                        True,
                        loop_blocks_for_io,
                        loop_blocks_for_io_minus_entry,
                        entry_pred,
                        exit_block,
                        inst,
                        size_var,
                        step_var,
                        latest_index,
                        loop_index,
                    )

        return False, None, None, None, None, None, None, None, None, None

    def some_for_directive(
        self, args, main_start_tag, main_end_tag, first_clause, gen_shared
    ):
        if DEBUG_OPENMP >= 1:
            print("some_for_directive", self.body_blocks)
        start_tags = [openmp_tag(main_start_tag)]
        end_tags = [openmp_tag(main_end_tag)]
        clauses = self.some_data_clause_directive(
            args, start_tags, end_tags, first_clause, has_loop=True
        )

        if "PARALLEL" in main_start_tag:
            # ---- Back propagate THREAD_LIMIT to enclosed target region. ----
            self.parallel_back_prop(clauses)

        if len(list(filter(lambda x: x.name == "QUAL.OMP.NUM_THREADS", clauses))) > 1:
            raise MultipleNumThreadsClauses(
                f"Multiple num_threads clauses near line {self.loc} is not allowed in an OpenMP parallel region."
            )

    # --------- Parser functions ------------------------

    def barrier_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit barrier_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.BARRIER")], 0, self.loc)
        or_start.requires_combined_acquire_release()
        or_end = openmp_region_end(
            or_start, [openmp_tag("DIR.OMP.END.BARRIER")], self.loc
        )
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def taskwait_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit taskwait_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.TASKWAIT")], 0, self.loc)
        or_start.requires_combined_acquire_release()
        or_end = openmp_region_end(
            or_start, [openmp_tag("DIR.OMP.END.TASKWAIT")], self.loc
        )
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def taskyield_directive(self, args):
        raise NotImplementedError("Taskyield currently unsupported.")

    # Don't need a rule for BARRIER.
    # Don't need a rule for TASKWAIT.
    # Don't need a rule for TASKYIELD.

    def taskgroup_directive(self, args):
        raise NotImplementedError("Taskgroup currently unsupported.")

    # Don't need a rule for taskgroup_construct.
    # Don't need a rule for TASKGROUP.

    # Don't need a rule for openmp_construct.

    # def teams_distribute_parallel_for_simd_clause(self, args):
    #    raise NotImplementedError("""Simd clause for target teams
    #                             distribute parallel loop currently unsupported.""")
    #    if DEBUG_OPENMP >= 1:
    #        print("visit device_clause", args, type(args))

    # Don't need a rule for for_simd_construct.

    def for_simd_directive(self, args):
        raise NotImplementedError("For simd currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit for_simd_directive", args, type(args))

    def for_simd_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit for_simd_clause", args, type(args), args[0])
        return args[0]

    def schedule_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit schedule_clause", args, type(args), args[0])
        return args[0]

    def dist_schedule_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit dist_schedule_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for parallel_for_simd_construct.

    def parallel_for_simd_directive(self, args):
        raise NotImplementedError("Parallel for simd currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit parallel_for_simd_directive", args, type(args))

    def parallel_for_simd_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit parallel_for_simd_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for target_data_construct.

    def target_data_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit target_data_directive", args, type(args))

        before_start = []
        after_start = []

        clauses, default_shared = self.flatten(args[2:], sblk)

        if DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("final clause:", clause)

        inputs_to_region, def_but_live_out, private_to_region, live_map = (
            self.find_io_vars(self.body_blocks)
        )
        used_in_region = inputs_to_region | def_but_live_out | private_to_region
        clauses = self.filter_unused_vars(clauses, used_in_region)

        start_tags = [openmp_tag("DIR.OMP.TARGET.DATA")] + clauses
        end_tags = [openmp_tag("DIR.OMP.END.TARGET.DATA")]

        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = before_start + [or_start] + after_start + sblk.body[:]
        eblk.body = [or_end] + eblk.body[:]

        add_enclosing_region(self.func_ir, self.body_blocks, or_start)

    # Don't need a rule for DATA.

    def target_data_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit target_data_clause", args, type(args), args[0])
        (val,) = args
        if isinstance(val, openmp_tag):
            return [val]
        elif isinstance(val, list):
            return val
        elif val == "nowait":
            return openmp_tag("QUAL.OMP.NOWAIT")
        else:
            return val

    def target_enter_data_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit target_enter_data_clause", args, type(args), args[0])
        (val,) = args
        if isinstance(val, openmp_tag):
            return [val]
        elif isinstance(val, list):
            return val
        elif val == "nowait":
            return openmp_tag("QUAL.OMP.NOWAIT")
        else:
            return val

    def target_exit_data_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit target_exit_data_clause", args, type(args), args[0])
        (val,) = args
        if isinstance(val, openmp_tag):
            return [val]
        elif isinstance(val, list):
            return val
        elif val == "nowait":
            return openmp_tag("QUAL.OMP.NOWAIT")
        else:
            return val

    def device_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit device_clause", args, type(args))
        return [openmp_tag("QUAL.OMP.DEVICE", args[0])]

    def map_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit map_clause", args, type(args), args[0])
        if args[0] in ["to", "from", "alloc", "tofrom"]:
            map_type = args[0].upper()
            var_list = args[1]
            assert len(args) == 2
        else:
            # TODO: is this default right?
            map_type = "TOFROM"
            var_list = args[1]
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.MAP." + map_type, var))
        return ret

    def map_enter_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit map_enter_clause", args, type(args), args[0])
        assert args[0] in ["to", "alloc"]
        map_type = args[0].upper()
        var_list = args[1]
        assert len(args) == 2
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.MAP." + map_type, var))
        return ret

    def map_exit_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit map_exit_clause", args, type(args), args[0])
        assert args[0] in ["from", "release", "delete"]
        map_type = args[0].upper()
        var_list = args[1]
        assert len(args) == 2
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.MAP." + map_type, var))
        return ret

    def depend_with_modifier_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit depend_with_modifier_clause", args, type(args), args[0])
        dep_type = args[1].upper()
        var_list = args[2]
        assert len(args) == 3
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.DEPEND." + dep_type, var))
        return ret

    def map_type(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit map_type", args, type(args), args[0])
        return str(args[0])

    def map_enter_type(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit map_enter_type", args, type(args), args[0])
        return str(args[0])

    def map_exit_type(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit map_exit_type", args, type(args), args[0])
        return str(args[0])

    def update_motion_type(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit update_motion_type", args, type(args), args[0])
        return str(args[0])

    # Don't need a rule for TO.
    # Don't need a rule for FROM.
    # Don't need a rule for ALLOC.
    # Don't need a rule for TOFROM.
    # Don't need a rule for parallel_sections_construct.

    def parallel_sections_directive(self, args):
        raise NotImplementedError("Parallel sections currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit parallel_sections_directive", args, type(args))

    def parallel_sections_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit parallel_sections_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for sections_construct.

    def sections_directive(self, args):
        raise NotImplementedError("Sections directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit sections_directive", args, type(args))

    # Don't need a rule for SECTIONS.

    def sections_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit sections_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for section_construct.

    def section_directive(self, args):
        raise NotImplementedError("Section directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit section_directive", args, type(args))

    # Don't need a rule for SECTION.
    # Don't need a rule for atomic_construct.

    def atomic_directive(self, args):
        raise NotImplementedError("Atomic currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit atomic_directive", args, type(args))

    # Don't need a rule for ATOMIC.

    def atomic_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit atomic_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for READ.
    # Don't need a rule for WRITE.
    # Don't need a rule for UPDATE.
    # Don't need a rule for CAPTURE.
    # Don't need a rule for seq_cst_clause.
    # Don't need a rule for critical_construct.

    def critical_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        if DEBUG_OPENMP >= 1:
            print("visit critical_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.CRITICAL")], 0, self.loc)
        or_start.requires_acquire_release()
        or_end = openmp_region_end(
            or_start, [openmp_tag("DIR.OMP.END.CRITICAL")], self.loc
        )

        inputs_to_region, def_but_live_out, private_to_region, live_map = (
            self.find_io_vars(self.body_blocks)
        )
        inputs_to_region = {remove_ssa(x, scope, self.loc): x for x in inputs_to_region}
        def_but_live_out = {remove_ssa(x, scope, self.loc): x for x in def_but_live_out}
        common_keys = inputs_to_region.keys() & def_but_live_out.keys()
        in_def_live_out = {
            inputs_to_region[k]: def_but_live_out[k] for k in common_keys
        }
        if DEBUG_OPENMP >= 1:
            print("inputs_to_region:", sorted(inputs_to_region))
            print("def_but_live_out:", sorted(def_but_live_out))
            print("in_def_live_out:", sorted(in_def_live_out))

        reset = []
        for k, v in in_def_live_out.items():
            reset.append(
                ir.Assign(
                    ir.Var(scope, v, self.loc), ir.Var(scope, k, self.loc), self.loc
                )
            )

        sblk.body = [or_start] + sblk.body[:]
        eblk.body = reset + [or_end] + eblk.body[:]

    # Don't need a rule for CRITICAL.
    # Don't need a rule for target_construct.
    # Don't need a rule for target_teams_distribute_parallel_for_simd_construct.

    def teams_back_prop(self, clauses):
        enclosing_regions = get_enclosing_region(self.func_ir, self.blk_start)
        if DEBUG_OPENMP >= 1:
            print("teams enclosing_regions:", enclosing_regions)
        if not enclosing_regions:
            return

        for enclosing_region in enclosing_regions[::-1]:
            if not self.get_directive_match(enclosing_region.tags, "DIR.OMP.TARGET"):
                continue

            nt_tag = self.get_clauses_by_name(
                enclosing_region.tags, "QUAL.OMP.NUM_TEAMS"
            )
            assert len(nt_tag) == 1
            cur_num_team_clauses = self.get_clauses_by_name(
                clauses, "QUAL.OMP.NUM_TEAMS", remove_from_orig=True
            )
            if len(cur_num_team_clauses) >= 1:
                nt_tag[-1].arg = cur_num_team_clauses[-1].arg
            else:
                nt_tag[-1].arg = 0

            nt_tag = self.get_clauses_by_name(
                enclosing_region.tags, "QUAL.OMP.THREAD_LIMIT"
            )
            assert len(nt_tag) == 1
            cur_num_team_clauses = self.get_clauses_by_name(
                clauses, "QUAL.OMP.THREAD_LIMIT", remove_from_orig=True
            )
            if len(cur_num_team_clauses) >= 1:
                nt_tag[-1].arg = cur_num_team_clauses[-1].arg
            else:
                nt_tag[-1].arg = 0

            return

    def check_distribute_nesting(self, dir_tag):
        if "DISTRIBUTE" in dir_tag and "TEAMS" not in dir_tag:
            enclosing_regions = get_enclosing_region(self.func_ir, self.blk_start)
            if (
                len(enclosing_regions) < 1
                or "TEAMS" not in enclosing_regions[-1].tags[0].name
            ):
                raise NotImplementedError(
                    "DISTRIBUTE must be nested under or combined with TEAMS."
                )

    def teams_directive(self, args):
        if DEBUG_OPENMP >= 1:
            print(
                "visit teams_directive", args, type(args), self.blk_start, self.blk_end
            )
        start_tags = [openmp_tag("DIR.OMP.TEAMS")]
        end_tags = [openmp_tag("DIR.OMP.END.TEAMS")]
        clauses = self.some_data_clause_directive(args, start_tags, end_tags, 1)

        self.teams_back_prop(clauses)

    def target_directive(self, args):
        if sys.platform.startswith("darwin"):
            print("ERROR: OpenMP target offloading is unavailable on Darwin")
            sys.exit(-1)
        self.some_target_directive(args, "TARGET", 1)

    def target_teams_directive(self, args):
        self.some_target_directive(args, "TARGET.TEAMS", 2)

    def target_teams_distribute_directive(self, args):
        self.some_target_directive(args, "TARGET.TEAMS.DISTRIBUTE", 3, has_loop=True)

    def target_loop_directive(self, args):
        self.some_target_directive(
            args, "TARGET.TEAMS.DISTRIBUTE.PARALLEL.LOOP", 2, has_loop=True
        )

    def target_teams_loop_directive(self, args):
        self.some_target_directive(
            args, "TARGET.TEAMS.DISTRIBUTE.PARALLEL.LOOP", 3, has_loop=True
        )

    def target_teams_distribute_parallel_for_directive(self, args):
        self.some_target_directive(
            args, "TARGET.TEAMS.DISTRIBUTE.PARALLEL.LOOP", 5, has_loop=True
        )

    def target_teams_distribute_parallel_for_simd_directive(self, args):
        # Intentionally dropping "SIMD" from string as that typically isn't implemented on GPU.
        self.some_target_directive(
            args, "TARGET.TEAMS.DISTRIBUTE.PARALLEL.LOOP", 6, has_loop=True
        )

    def get_clauses_by_name(self, clauses, names, remove_from_orig=False):
        if not isinstance(names, list):
            names = [names]

        ret = list(filter(lambda x: x.name in names, clauses))
        if remove_from_orig:
            clauses[:] = list(filter(lambda x: x.name not in names, clauses))
        return ret

    def get_clauses_by_start(self, clauses, names, remove_from_orig=False):
        if not isinstance(names, list):
            names = [names]
        ret = list(
            filter(lambda x: any([x.name.startswith(y) for y in names]), clauses)
        )
        if remove_from_orig:
            clauses[:] = list(
                filter(
                    lambda x: any([not x.name.startswith(y) for y in names]), clauses
                )
            )
        return ret

    def get_clauses_if_contains(self, clauses, names, remove_from_orig=False):
        if not isinstance(names, list):
            names = [names]
        ret = list(filter(lambda x: any([y in x.name for y in names]), clauses))
        if remove_from_orig:
            clauses[:] = list(
                filter(lambda x: any([not y in x.name for y in names]), clauses)
            )
        return ret

    def get_directive_if_contains(self, tags, name):
        dir = [x for x in tags if x.name.startswith("DIR")]
        assert len(dir) == 1, "Expected one directive tag"
        ret = [x for x in dir if name in x.name]
        return ret

    def get_directive_match(self, tags, name):
        dir = [x for x in tags if x.name.startswith("DIR")]
        assert len(dir) == 1, "Expected one directive tag"
        ret = [x for x in dir if name == x.name]
        return ret

    def target_enter_data_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit target_enter_data_directive", args, type(args))

        clauses, _ = self.flatten(args[3:], sblk)
        or_start = openmp_region_start(
            [openmp_tag("DIR.OMP.TARGET.ENTER.DATA")] + clauses, 0, self.loc
        )
        or_end = openmp_region_end(
            or_start, [openmp_tag("DIR.OMP.END.TARGET.ENTER.DATA")], self.loc
        )
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def target_exit_data_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit target_exit_data_directive", args, type(args))

        clauses, _ = self.flatten(args[3:], sblk)
        or_start = openmp_region_start(
            [openmp_tag("DIR.OMP.TARGET.EXIT.DATA")] + clauses, 0, self.loc
        )
        or_end = openmp_region_end(
            or_start, [openmp_tag("DIR.OMP.END.TARGET.EXIT.DATA")], self.loc
        )
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def teams_distribute_parallel_for_simd_directive(self, args):
        self.some_distribute_directive(
            args, "TEAMS.DISTRIBUTE.PARALLEL.LOOP.SIMD", 5, has_loop=True
        )

    def teams_distribute_parallel_for_directive(self, args):
        self.some_distribute_directive(
            args, "TEAMS.DISTRIBUTE.PARALLEL.LOOP", 4, has_loop=True
        )

    def teams_distribute_directive(self, args):
        self.some_distribute_directive(args, "TEAMS.DISTRIBUTE", 2, has_loop=True)

    def teams_distribute_simd_directive(self, args):
        self.some_distribute_directive(args, "TEAMS.DISTRIBUTE.SIMD", 3, has_loop=True)

    def teams_loop_directive(self, args):
        self.some_distribute_directive(
            args, "TEAMS.DISTRIBUTE.PARALLEL.LOOP", 2, has_loop=True
        )

    def loop_directive(self, args):
        # TODO Add error checking that a clause that the parser accepts if we find that
        # loop can even take clauses, which we're not sure that it can.
        enclosing_regions = get_enclosing_region(self.func_ir, self.blk_start)
        if not enclosing_regions or len(enclosing_regions) < 1:
            self.some_for_directive(
                args, "DIR.OMP.PARALLEL.LOOP", "DIR.OMP.END.PARALLEL.LOOP", 1, True
            )
        else:
            if "DISTRIBUTE" in enclosing_regions[-1].tags[0].name:
                self.some_distribute_directive(args, "PARALLEL.LOOP", 1, has_loop=True)
            elif "TEAMS" in enclosing_regions[-1].tags[0].name:
                self.some_distribute_directive(
                    args, "DISTRIBUTE.PARALLEL.LOOP", 1, has_loop=True
                )
            else:
                if "TARGET" in enclosing_regions[-1].tags[0].name:
                    self.some_distribute_directive(
                        args, "TEAMS.DISTRIBUTE.PARALLEL.LOOP", 1, has_loop=True
                    )
                else:
                    self.some_for_directive(
                        args,
                        "DIR.OMP.PARALLEL.LOOP",
                        "DIR.OMP.END.PARALLEL.LOOP",
                        1,
                        True,
                    )

    def distribute_directive(self, args):
        self.some_distribute_directive(args, "DISTRIBUTE", 1, has_loop=True)

    def distribute_simd_directive(self, args):
        self.some_distribute_directive(args, "DISTRIBUTE.SIMD", 2, has_loop=True)

    def distribute_parallel_for_directive(self, args):
        self.some_distribute_directive(
            args, "DISTRIBUTE.PARALLEL.LOOP", 3, has_loop=True
        )

    def distribute_parallel_for_simd_directive(self, args):
        self.some_distribute_directive(
            args, "DISTRIBUTE.PARALLEL.LOOP.SIMD", 4, has_loop=True
        )

    def some_distribute_directive(self, args, dir_tag, lexer_count, has_loop=False):
        if DEBUG_OPENMP >= 1:
            print(
                "visit some_distribute_directive",
                args,
                type(args),
                self.blk_start,
                self.blk_end,
            )

        self.check_distribute_nesting(dir_tag)

        target_num = OpenmpVisitor.target_num
        OpenmpVisitor.target_num += 1

        dir_start_tag = "DIR.OMP." + dir_tag
        dir_end_tag = "DIR.OMP.END." + dir_tag
        start_tags = [openmp_tag(dir_start_tag, target_num)]
        end_tags = [openmp_tag(dir_end_tag, target_num)]

        sblk = self.blocks[self.blk_start]
        clauses, _ = self.flatten(args[lexer_count:], sblk)

        if "TEAMS" in dir_tag:
            # NUM_TEAMS, THREAD_LIMIT are not in clauses, set them to 0 to
            # use runtime defaults in teams, thread launching.
            if len(self.get_clauses_by_name(clauses, "QUAL.OMP.NUM_TEAMS")) == 0:
                start_tags.append(openmp_tag("QUAL.OMP.NUM_TEAMS", 0))
            if len(self.get_clauses_by_name(clauses, "QUAL.OMP.THREAD_LIMIT")) == 0:
                start_tags.append(openmp_tag("QUAL.OMP.THREAD_LIMIT", 0))
            self.teams_back_prop(clauses)
        elif "PARALLEL" in dir_tag:
            self.parallel_back_prop(clauses)

        if DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("target clause:", clause)

        self.some_data_clause_directive(
            clauses, start_tags, end_tags, 0, has_loop=has_loop, for_target=False
        )

    def some_target_directive(self, args, dir_tag, lexer_count, has_loop=False):
        if DEBUG_OPENMP >= 1:
            print(
                "visit some_target_directive",
                args,
                type(args),
                self.blk_start,
                self.blk_end,
            )

        self.check_distribute_nesting(dir_tag)

        target_num = OpenmpVisitor.target_num
        OpenmpVisitor.target_num += 1

        dir_start_tag = "DIR.OMP." + dir_tag
        dir_end_tag = "DIR.OMP.END." + dir_tag
        start_tags = [openmp_tag(dir_start_tag, target_num)]
        end_tags = [openmp_tag(dir_end_tag, target_num)]

        sblk = self.blocks[self.blk_start]
        clauses, _ = self.flatten(args[lexer_count:], sblk)

        if "TEAMS" in dir_tag:
            # When NUM_TEAMS, THREAD_LIMIT are not in clauses, set them to 0 to
            # use runtime defaults in teams, thread launching, otherwise use
            # existing clauses.
            clause_num_teams = self.get_clauses_by_name(clauses, "QUAL.OMP.NUM_TEAMS")
            if not clause_num_teams:
                start_tags.append(openmp_tag("QUAL.OMP.NUM_TEAMS", 0))

            # Use the THREAD_LIMIT clause value if it exists, regardless of a
            # combined PARALLEL (see
            # https://www.openmp.org/spec-html/5.0/openmpse15.html) since
            # THREAD_LIMIT takes precedence.  If clause does not exist, set to 0
            # or to NUM_THREADS of the combined PARALLEL (if this exists).
            clause_thread_limit = self.get_clauses_by_name(
                clauses, "QUAL.OMP.THREAD_LIMIT"
            )
            if not clause_thread_limit:
                thread_limit = 0
                if "PARALLEL" in dir_tag:
                    clause_num_threads = self.get_clauses_by_name(
                        clauses, "QUAL.OMP.NUM_THREADS"
                    )
                    if clause_num_threads:
                        assert len(clause_num_threads) == 1, (
                            "Expected single NUM_THREADS clause"
                        )
                        thread_limit = clause_num_threads[0].arg
                start_tags.append(openmp_tag("QUAL.OMP.THREAD_LIMIT", thread_limit))
        elif "PARALLEL" in dir_tag:
            # PARALLEL in the directive (without TEAMS), set THREAD_LIMIT to NUM_THREADS clause
            # (if NUM_THREADS exists), or 0 (if NUM_THREADS does not exist)
            num_threads = 0
            clause_num_threads = self.get_clauses_by_name(
                clauses, "QUAL.OMP.NUM_THREADS"
            )
            if clause_num_threads:
                assert len(clause_num_threads) == 1, (
                    "Expected single NUM_THREADS clause"
                )
                num_threads = clause_num_threads[0].arg

            # Replace existing THREAD_LIMIT clause.
            clause_thread_limit = self.get_clauses_by_name(
                clauses, "QUAL.OMP.THREAD_LIMIT", remove_from_orig=True
            )
            clauses.append(openmp_tag("QUAL.OMP.THREAD_LIMIT", num_threads))
        else:
            # Neither TEAMS or PARALLEL in directive, set teams, threads to 1.
            start_tags.append(openmp_tag("QUAL.OMP.NUM_TEAMS", 1))
            start_tags.append(openmp_tag("QUAL.OMP.THREAD_LIMIT", 1))

        if DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("target clause:", clause)

        self.some_data_clause_directive(
            clauses, start_tags, end_tags, 0, has_loop=has_loop, for_target=True
        )
        # self.some_data_clause_directive(args, start_tags, end_tags, lexer_count, has_loop=has_loop)

    def add_to_returns(self, stmts):
        for blk in self.blocks.values():
            if isinstance(blk.body[-1], ir.Return):
                blk.body = blk.body[:-1] + stmts + [blk.body[-1]]

    def add_block_in_order(self, new_block, insert_after_block):
        """Insert a new block after the specified block while maintaining topological order"""
        new_blocks = {}
        # Copy blocks up to and including insert_after_block
        for label, block in self.blocks.items():
            new_blocks[label] = block
            if label == insert_after_block:
                # Insert new block right after
                # We add a fractional to make sure the block is sorted right
                # after the insert_after_block and before its successor.
                # TODO: Avoid this fractional addition.
                new_block_num = label + 0.1
                new_blocks[new_block_num] = new_block
        # Copy remaining blocks
        for label, block in self.blocks.items():
            if label > insert_after_block:
                new_blocks[label] = block
        # new_blocks = flatten_labels(new_blocks)
        self.blocks.clear()
        self.blocks.update(new_blocks)
        return new_block_num

    def some_data_clause_directive(
        self,
        args,
        start_tags,
        end_tags,
        lexer_count,
        has_loop=False,
        for_target=False,
        for_task=False,
    ):
        if DEBUG_OPENMP >= 1:
            print(
                "visit some_data_clause_directive",
                args,
                type(args),
                self.blk_start,
                self.blk_end,
            )
        assert not (for_target and for_task)

        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        if DEBUG_OPENMP >= 1:
            for clause in args[lexer_count:]:
                print("pre clause:", clause)
        clauses, default_shared = self.flatten(args[lexer_count:], sblk)
        if DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("final clause:", clause)

        before_start = []
        after_start = []
        for_before_start = []
        for_after_start = []

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses, explicit_privates, non_user_explicits = (
            self.get_explicit_vars(clauses)
        )
        if DEBUG_OPENMP >= 1:
            print(
                "vars_in_explicit_clauses:",
                sorted(vars_in_explicit_clauses),
                type(vars_in_explicit_clauses),
            )
            for v in clauses:
                print("vars_in_explicit clauses first:", v)

        if has_loop:
            prepare_out = self.prepare_for_directive(
                clauses,
                vars_in_explicit_clauses,
                for_before_start,
                for_after_start,
                start_tags,
                end_tags,
                scope,
            )
            vars_in_explicit_clauses, explicit_privates, non_user_explicits = (
                self.get_explicit_vars(clauses)
            )
            (
                found_loop,
                blocks_for_io,
                blocks_in_region,
                entry_pred,
                exit_block,
                inst,
                size_var,
                step_var,
                latest_index,
                loop_index,
            ) = prepare_out
            assert found_loop
        else:
            blocks_for_io = self.body_blocks
            blocks_in_region = get_blocks_between_start_end(
                self.blocks, self.blk_start, self.blk_end
            )
            entry_pred = sblk
            exit_block = eblk

        # Do an analysis to get variable use information coming into and out of the region.
        inputs_to_region, def_but_live_out, private_to_region, live_map = (
            self.find_io_vars(blocks_for_io)
        )
        live_out_copy = copy.copy(def_but_live_out)

        if DEBUG_OPENMP >= 1:
            print("inputs_to_region:", sorted(inputs_to_region))
            print("def_but_live_out:", sorted(def_but_live_out))
            print("private_to_region:", sorted(private_to_region))
            for v in clauses:
                print("clause after find_io_vars:", v)

        # Remove variables the user explicitly added to a clause from the auto-determined variables.
        # This will also treat SSA forms of vars the same as their explicit Python var clauses.
        self.remove_explicit_from_io_vars(
            inputs_to_region,
            def_but_live_out,
            private_to_region,
            vars_in_explicit_clauses,
            clauses,
            non_user_explicits,
            scope,
            self.loc,
        )

        if DEBUG_OPENMP >= 1:
            for v in clauses:
                print("clause after remove_explicit_from_io_vars:", v)

        if DEBUG_OPENMP >= 1:
            for k, v in vars_in_explicit_clauses.items():
                print("vars_in_explicit before:", k, v)
            for v in clauses:
                print("vars_in_explicit clauses before:", v)
            for k, v in non_user_explicits.items():
                print("non_user_explicits before:", k, v)

        if DEBUG_OPENMP >= 1:
            print("inputs_to_region after remove_explicit:", sorted(inputs_to_region))
            print("def_but_live_out after remove_explicit:", sorted(def_but_live_out))
            print("private_to_region after remove_explicit:", sorted(private_to_region))

        if not default_shared and (
            has_user_defined_var(inputs_to_region)
            or has_user_defined_var(def_but_live_out)
            or has_user_defined_var(private_to_region)
        ):
            user_defined_inputs = get_user_defined_var(inputs_to_region)
            user_defined_def_live = get_user_defined_var(def_but_live_out)
            user_defined_private = get_user_defined_var(private_to_region)
            if DEBUG_OPENMP >= 1:
                print("inputs users:", sorted(user_defined_inputs))
                print("def users:", sorted(user_defined_def_live))
                print("private users:", sorted(user_defined_private))
            raise UnspecifiedVarInDefaultNone(
                "Variables with no data env clause in OpenMP region: "
                + str(
                    user_defined_inputs.union(user_defined_def_live).union(
                        user_defined_private
                    )
                )
            )

        if for_target:
            self.make_implicit_explicit_target(
                scope,
                vars_in_explicit_clauses,
                clauses,
                True,
                inputs_to_region,
                def_but_live_out,
                private_to_region,
            )
        elif for_task:
            self.make_implicit_explicit(
                scope,
                vars_in_explicit_clauses,
                clauses,
                True,
                inputs_to_region,
                def_but_live_out,
                private_to_region,
                for_task=get_enclosing_region(self.func_ir, self.blk_start),
            )
        else:
            self.make_implicit_explicit(
                scope,
                vars_in_explicit_clauses,
                clauses,
                True,
                inputs_to_region,
                def_but_live_out,
                private_to_region,
            )
        if DEBUG_OPENMP >= 1:
            for k, v in vars_in_explicit_clauses.items():
                print("vars_in_explicit after:", k, v)
            for v in clauses:
                print("vars_in_explicit clauses after:", v)
        vars_in_explicit_clauses, explicit_privates, non_user_explicits = (
            self.get_explicit_vars(clauses)
        )
        if DEBUG_OPENMP >= 1:
            print("post get_explicit_vars:", explicit_privates)
            for k, v in vars_in_explicit_clauses.items():
                print("vars_in_explicit post:", k, v)
        if DEBUG_OPENMP >= 1:
            print("blocks_in_region:", blocks_in_region)

        self.make_consts_unliteral_for_privates(explicit_privates, self.blocks)

        # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
        clause_privates = self.get_clause_privates(
            clauses, live_out_copy, scope, self.loc
        )

        if DEBUG_OPENMP >= 1:
            print("clause_privates:", sorted(clause_privates), type(clause_privates))
            print("inputs_to_region:", sorted(inputs_to_region))
            print("def_but_live_out:", sorted(def_but_live_out))
            print("live_out_copy:", sorted(live_out_copy))
            print("private_to_region:", sorted(private_to_region))

        keep_alive = []
        tags_for_enclosing = self.add_explicits_to_start(
            scope, vars_in_explicit_clauses, clauses, True, start_tags, keep_alive
        )
        add_tags_to_enclosing(self.func_ir, self.blk_start, tags_for_enclosing)

        # or_start = openmp_region_start([openmp_tag("DIR.OMP.TARGET", target_num)] + clauses, 0, self.loc)
        # or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.TARGET", target_num)], self.loc)
        # new_header_block_num = max(self.blocks.keys()) + 1

        firstprivate_dead_after = list(
            filter(
                lambda x: x.name == "QUAL.OMP.FIRSTPRIVATE"
                and x.arg not in live_map[self.blk_end],
                start_tags,
            )
        )

        or_start = openmp_region_start(
            start_tags, 0, self.loc, firstprivate_dead_after=firstprivate_dead_after
        )
        or_end = openmp_region_end(or_start, end_tags, self.loc)

        if DEBUG_OPENMP >= 1:
            for x in keep_alive:
                print("keep_alive:", x)
            for x in firstprivate_dead_after:
                print("firstprivate_dead_after:", x)

        # Adding the openmp tags in topo order to avoid problems with code
        # generation and with_lifting legalization.
        # TODO: we should remove the requirement to process in topo order. There
        # is state depending on topo order processing.
        if has_loop:
            new_header_block = ir.Block(scope, self.loc)
            new_header_block.body = (
                [or_start] + after_start + for_after_start + [entry_pred.body[-1]]
            )
            new_block_num = self.add_block_in_order(new_header_block, self.blk_start)
            entry_pred.body = (
                entry_pred.body[:-1]
                + before_start
                + for_before_start
                + [ir.Jump(new_block_num, self.loc)]
            )

            if for_task:
                exit_block.body = [or_end] + exit_block.body
                self.add_to_returns(keep_alive)
            else:
                exit_block.body = [or_end] + keep_alive + exit_block.body
        else:
            new_header_block = ir.Block(scope, self.loc)
            new_header_block.body = [or_start] + after_start + sblk.body[:]
            new_header_block_num = self.add_block_in_order(
                new_header_block, self.blk_start
            )
            sblk.body = before_start + [ir.Jump(new_header_block_num, self.loc)]

            # NOTE: or_start could also be inlined for correct codegen as
            # follows. Favoring the add_block_in_order method for consistency.
            # sblk.body = before_start + [or_start] + after_start + sblk.body[:]

            if for_task:
                eblk.body = [or_end] + eblk.body[:]
                self.add_to_returns(keep_alive)
            else:
                eblk.body = [or_end] + keep_alive + eblk.body[:]

        add_enclosing_region(self.func_ir, self.body_blocks, or_start)
        return clauses

    def target_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit target_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        (val,) = args
        if isinstance(val, openmp_tag):
            return [val]
        elif isinstance(val, list):
            return val
        elif val == "nowait":
            return openmp_tag("QUAL.OMP.NOWAIT")
        else:
            return val
        # return args[0]

    def target_teams_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit target_teams_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def target_teams_distribute_parallel_for_simd_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print(
                "visit target_teams_distribute_parallel_for_simd_clause",
                args,
                type(args),
                args[0],
            )
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def teams_distribute_parallel_for_simd_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print(
                "visit teams_distribute_parallel_for_simd_clause",
                args,
                type(args),
                args[0],
            )
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def teams_distribute_parallel_for_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print(
                "visit teams_distribute_parallel_for_clause", args, type(args), args[0]
            )
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def distribute_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit distribute_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def teams_distribute_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit teams_distribute_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def teams_distribute_simd_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit teams_distribute_simd_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def distribute_parallel_for_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit distribute_parallel_for_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def target_teams_distribute_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit target_teams_distribute_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def target_teams_distribute_parallel_for_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print(
                "visit target_teams_distribute_parallel_for_clause",
                args,
                type(args),
                args[0],
            )
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    # Don't need a rule for target_update_construct.

    def target_update_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit target_update_directive", args, type(args))
        clauses, _ = self.flatten(args[2:], sblk)
        or_start = openmp_region_start(
            [openmp_tag("DIR.OMP.TARGET.UPDATE")] + clauses, 0, self.loc
        )
        or_end = openmp_region_end(
            or_start, [openmp_tag("DIR.OMP.END.TARGET.UPDATE")], self.loc
        )
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def target_update_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit target_update_clause", args, type(args), args[0])
        # return args[0]
        (val,) = args
        if isinstance(val, openmp_tag):
            return [val]
        elif isinstance(val, list):
            return val
        else:
            return val

    def motion_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit motion_clause", args, type(args))
        assert args[0] in ["to", "from"]
        map_type = args[0].upper()
        var_list = args[1]
        assert len(args) == 2
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.MAP." + map_type, var))
        return ret

    def variable_array_section_list(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit variable_array_section_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

    """
    def array_section(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit array_section", args, type(args))
        return args

    def array_section_subscript(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit array_section_subscript", args, type(args))
        return args
    """

    # Don't need a rule for TARGET.
    # Don't need a rule for single_construct.

    def single_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit single_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.SINGLE")], 0, self.loc)
        or_start.requires_acquire_release()
        or_end = openmp_region_end(
            or_start, [openmp_tag("DIR.OMP.END.SINGLE")], self.loc
        )
        sblk.body = [or_start] + sblk.body[:]
        eblk.body = [or_end] + eblk.body[:]

    def single_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit single_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for unique_single_clause.
    # def NOWAIT(self, args):
    #    return "nowait"
    # Don't need a rule for NOWAIT.
    # Don't need a rule for master_construct.

    def master_directive(self, args):
        raise NotImplementedError("Master directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit master_directive", args, type(args))

    # Don't need a rule for simd_construct.

    def simd_directive(self, args):
        raise NotImplementedError("Simd directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit simd_directive", args, type(args))

    # Don't need a rule for SIMD.

    def simd_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit simd_clause", args, type(args), args[0])
        return args[0]

    def aligned_clause(self, args):
        raise NotImplementedError("Aligned clause currently unsupported.")
        if DEBUG_OPENMP >= 1:
            print("visit aligned_clause", args, type(args))

    # Don't need a rule for declare_simd_construct.

    def declare_simd_directive_seq(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit declare_simd_directive_seq", args, type(args), args[0])
        return args[0]

    def declare_simd_directive(self, args):
        raise NotImplementedError("Declare simd directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit declare_simd_directive", args, type(args))

    def declare_simd_clause(self, args):
        raise NotImplementedError("Declare simd clauses currently unsupported.")
        if DEBUG_OPENMP >= 1:
            print("visit declare_simd_clause", args, type(args))

    # Don't need a rule for ALIGNED.

    def inbranch_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit inbranch_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for INBRANCH.
    # Don't need a rule for NOTINBRANCH.

    def uniform_clause(self, args):
        raise NotImplementedError("Uniform clause currently unsupported.")
        if DEBUG_OPENMP >= 1:
            print("visit uniform_clause", args, type(args))

    # Don't need a rule for UNIFORM.

    def collapse_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit collapse_clause", args, type(args))
        return openmp_tag("QUAL.OMP.COLLAPSE", args[1])

    # Don't need a rule for COLLAPSE.
    # Don't need a rule for task_construct.
    # Don't need a rule for TASK.

    def task_directive(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit task_directive", args, type(args))

        start_tags = [openmp_tag("DIR.OMP.TASK")]
        end_tags = [openmp_tag("DIR.OMP.END.TASK")]
        self.some_data_clause_directive(args, start_tags, end_tags, 1, for_task=True)

    def task_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit task_clause", args, type(args), args[0])
        return args[0]

    def unique_task_clause(self, args):
        raise NotImplementedError("Task-related clauses currently unsupported.")
        if DEBUG_OPENMP >= 1:
            print("visit unique_task_clause", args, type(args))

    # Don't need a rule for DEPEND.
    # Don't need a rule for FINAL.
    # Don't need a rule for UNTIED.
    # Don't need a rule for MERGEABLE.

    def dependence_type(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit dependence_type", args, type(args), args[0])
        return args[0]

    # Don't need a rule for IN.
    # Don't need a rule for OUT.
    # Don't need a rule for INOUT.

    def data_default_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit data_default_clause", args, type(args), args[0])
        return args[0]

    def data_sharing_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit data_sharing_clause", args, type(args), args[0])
        return args[0]

    def data_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit data_clause", args, type(args), args[0])
        return args[0]

    def private_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit private_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.PRIVATE", var))
        return ret

    # Don't need a rule for PRIVATE.

    def copyprivate_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit copyprivate_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.COPYPRIVATE", var))
        return ret

    # Don't need a rule for COPYPRIVATE.

    def firstprivate_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit firstprivate_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", var))
        return ret

    # Don't need a rule for FIRSTPRIVATE.

    def lastprivate_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit lastprivate_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.LASTPRIVATE", var))
        return ret

    # Don't need a rule for LASTPRIVATE.

    def shared_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit shared_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.SHARED", var))
        return ret

    # Don't need a rule for SHARED.

    def copyin_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit copyin_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.COPYIN", var))
        return ret

    # Don't need a rule for COPYIN.
    # Don't need a rule for REDUCTION.

    def reduction_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit reduction_clause", args, type(args), args[0])

        (_, red_op, red_list) = args
        ret = []
        for shared in red_list:
            ret.append(openmp_tag("QUAL.OMP.REDUCTION." + red_op, shared))
        return ret

    def default_shared_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit default_shared_clause", args, type(args))
        return default_shared_val(True)

    def default_none_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit default_none", args, type(args))
        return default_shared_val(False)

    def const_num_or_var(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit const_num_or_var", args, type(args))
        return args[0]

    # Don't need a rule for parallel_construct.

    def parallel_back_prop(self, clauses):
        enclosing_regions = get_enclosing_region(self.func_ir, self.blk_start)
        if DEBUG_OPENMP >= 1:
            print("parallel enclosing_regions:", enclosing_regions)
        if not enclosing_regions:
            return

        for enclosing_region in enclosing_regions[::-1]:
            # If there is TEAMS in the enclosing region then THREAD_LIMIT is
            # already set, do nothing.
            if self.get_directive_if_contains(enclosing_region.tags, "TEAMS"):
                return
            if not self.get_directive_if_contains(enclosing_region.tags, "TARGET"):
                continue

            # Set to 0 means "don't care", use implementation specific number of threads.
            num_threads = 0
            num_threads_clause = self.get_clauses_by_name(
                clauses, "QUAL.OMP.NUM_THREADS"
            )
            if num_threads_clause:
                assert len(num_threads_clause) == 1, (
                    "Expected num_threads clause defined once"
                )
                num_threads = num_threads_clause[0].arg
            nt_tag = self.get_clauses_by_name(
                enclosing_region.tags, "QUAL.OMP.THREAD_LIMIT"
            )
            assert len(nt_tag) > 0

            # If THREAD_LIMIT is less than requested NUM_THREADS or 1,
            # increase it.  This is still valid if THREAD_LIMIT is 0, since this
            # means there was a parallel region before that did not specify
            # NUM_THREADS so we can set to the concrete value of the sibling
            # parallel region with the max value of NUM_THREADS.
            if num_threads > nt_tag[-1].arg or nt_tag[-1].arg == 1:
                nt_tag[-1].arg = num_threads
            return

    def parallel_directive(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit parallel_directive", args, type(args))

        start_tags = [openmp_tag("DIR.OMP.PARALLEL")]
        end_tags = [openmp_tag("DIR.OMP.END.PARALLEL")]
        clauses = self.some_data_clause_directive(args, start_tags, end_tags, 1)

        # sblk = self.blocks[self.blk_start]
        # eblk = self.blocks[self.blk_end]
        # scope = sblk.scope

        # before_start = []
        # after_start = []
        # clauses, default_shared = self.flatten(args[1:], sblk)

        if len(list(filter(lambda x: x.name == "QUAL.OMP.NUM_THREADS", clauses))) > 1:
            raise MultipleNumThreadsClauses(
                f"Multiple num_threads clauses near line {self.loc} is not allowed in an OpenMP parallel region."
            )

        if DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("final clause:", clause)

        # ---- Back propagate THREAD_LIMIT to enclosed target region. ----
        self.parallel_back_prop(clauses)

    def parallel_clause(self, args):
        (val,) = args
        if DEBUG_OPENMP >= 1:
            print("visit parallel_clause", args, type(args), args[0])
        return val

    def unique_parallel_clause(self, args):
        (val,) = args
        if DEBUG_OPENMP >= 1:
            print("visit unique_parallel_clause", args, type(args), args[0])
        assert isinstance(val, openmp_tag)
        return val

    def teams_clause(self, args):
        (val,) = args
        if DEBUG_OPENMP >= 1:
            print("visit teams_clause", args, type(args), args[0])
        return val

    def num_teams_clause(self, args):
        (_, num_teams) = args
        if DEBUG_OPENMP >= 1:
            print("visit num_teams_clause", args, type(args))

        return openmp_tag("QUAL.OMP.NUM_TEAMS", num_teams, load=True)

    def thread_limit_clause(self, args):
        (_, thread_limit) = args
        if DEBUG_OPENMP >= 1:
            print("visit thread_limit_clause", args, type(args))

        return openmp_tag("QUAL.OMP.THREAD_LIMIT", thread_limit, load=True)

    def if_clause(self, args):
        (_, if_val) = args
        if DEBUG_OPENMP >= 1:
            print("visit if_clause", args, type(args))

        return openmp_tag("QUAL.OMP.IF", if_val, load=True)

    # Don't need a rule for IF.

    def num_threads_clause(self, args):
        (_, num_threads) = args
        if DEBUG_OPENMP >= 1:
            print("visit num_threads_clause", args, type(args))

        return openmp_tag("QUAL.OMP.NUM_THREADS", num_threads, load=True)

    # Don't need a rule for NUM_THREADS.
    # Don't need a rule for PARALLEL.
    # Don't need a rule for FOR.
    # Don't need a rule for parallel_for_construct.

    def parallel_for_directive(self, args):
        return self.some_for_directive(
            args, "DIR.OMP.PARALLEL.LOOP", "DIR.OMP.END.PARALLEL.LOOP", 2, True
        )

    def parallel_for_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit parallel_for_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for for_construct.

    def for_directive(self, args):
        return self.some_for_directive(
            args, "DIR.OMP.LOOP", "DIR.OMP.END.LOOP", 1, False
        )

    def for_clause(self, args):
        (val,) = args
        if DEBUG_OPENMP >= 1:
            print("visit for_clause", args, type(args))
        if isinstance(val, openmp_tag):
            return [val]
        elif isinstance(val, list):
            return val
        elif val == "nowait":
            return openmp_tag("QUAL.OMP.NOWAIT")

    def unique_for_clause(self, args):
        (val,) = args
        if DEBUG_OPENMP >= 1:
            print("visit unique_for_clause", args, type(args))
        if isinstance(val, openmp_tag):
            return val
        elif val == "ordered":
            return openmp_tag("QUAL.OMP.ORDERED", 0)

    # Don't need a rule for LINEAR.

    def linear_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit linear_clause", args, type(args), args[0])
        return args[0]

    """
    Linear_expr not in grammar
    def linear_expr(self, args):
        (_, var, step) = args
        if DEBUG_OPENMP >= 1:
            print("visit linear_expr", args, type(args))
        return openmp_tag("QUAL.OMP.LINEAR", [var, step])
    """

    """
    def ORDERED(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit ordered", args, type(args))
        return "ordered"
    """

    def sched_no_expr(self, args):
        (_, kind) = args
        if DEBUG_OPENMP >= 1:
            print("visit sched_no_expr", args, type(args))
        if kind == "static":
            return openmp_tag("QUAL.OMP.SCHEDULE.STATIC", 0)
        elif kind == "dynamic":
            return openmp_tag("QUAL.OMP.SCHEDULE.DYNAMIC", 0)
        elif kind == "guided":
            return openmp_tag("QUAL.OMP.SCHEDULE.GUIDED", 0)
        elif kind == "runtime":
            return openmp_tag("QUAL.OMP.SCHEDULE.RUNTIME", 0)

    def sched_expr(self, args):
        (_, kind, num_or_var) = args
        if DEBUG_OPENMP >= 1:
            print("visit sched_expr", args, type(args), num_or_var, type(num_or_var))
        if kind == "static":
            return openmp_tag("QUAL.OMP.SCHEDULE.STATIC", num_or_var, load=True)
        elif kind == "dynamic":
            return openmp_tag("QUAL.OMP.SCHEDULE.DYNAMIC", num_or_var, load=True)
        elif kind == "guided":
            return openmp_tag("QUAL.OMP.SCHEDULE.GUIDED", num_or_var, load=True)
        elif kind == "runtime":
            return openmp_tag("QUAL.OMP.SCHEDULE.RUNTIME", num_or_var, load=True)

    def SCHEDULE(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit SCHEDULE", args, type(args))
        return "schedule"

    def schedule_kind(self, args):
        (kind,) = args
        if DEBUG_OPENMP >= 1:
            print("visit schedule_kind", args, type(args))
        return kind

    # Don't need a rule for STATIC.
    # Don't need a rule for DYNAMIC.
    # Don't need a rule for GUIDED.
    # Don't need a rule for RUNTIME.

    """
    def STATIC(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit STATIC", args, type(args))
        return "static"

    def DYNAMIC(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit DYNAMIC", args, type(args))
        return "dynamic"

    def GUIDED(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit GUIDED", args, type(args))
        return "guided"

    def RUNTIME(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit RUNTIME", args, type(args))
        return "runtime"
    """

    def COLON(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit COLON", args, type(args))
        return ":"

    def oslice(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit oslice", args, type(args))
        start = None
        end = None
        if args[0] != ":":
            start = args[0]
            args = args[2:]
        else:
            args = args[1:]

        if len(args) > 0:
            end = args[0]
        return slice(start, end)

    def slice_list(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit slice_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

    def name_slice(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit name_slice", args, type(args))
        if len(args) == 1 or args[1] is None:
            return args[0]
        else:
            return NameSlice(args[0], args[1:])

    def var_list(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit var_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

    def number_list(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit number_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

    def ompx_attribute(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit ompx_attribute", args, type(args), args[0])
        (_, attr, number_list) = args
        return openmp_tag("QUAL.OMP.OMPX_ATTRIBUTE", (attr, number_list))

    def PLUS(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit PLUS", args, type(args))
        return "+"

    def MINUS(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit MINUS", args, type(args))
        return "-"

    def STAR(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit STAR", args, type(args))
        return "*"

    def reduction_operator(self, args):
        arg = args[0]
        if DEBUG_OPENMP >= 1:
            print("visit reduction_operator", args, type(args), arg, type(arg))
        if arg == "+":
            return "ADD"
        elif arg == "-":
            return "SUB"
        elif arg == "*":
            return "MUL"
        assert 0

    def threadprivate_directive(self, args):
        raise NotImplementedError("Threadprivate currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit threadprivate_directive", args, type(args))

    def cancellation_point_directive(self, args):
        raise NotImplementedError("""Explicit cancellation points
                                 currently unsupported.""")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit cancellation_point_directive", args, type(args))

    def construct_type_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit construct_type_clause", args, type(args), args[0])
        return args[0]

    def cancel_directive(self, args):
        raise NotImplementedError("Cancel directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit cancel_directive", args, type(args))

    # Don't need a rule for ORDERED.

    def flush_directive(self, args):
        raise NotImplementedError("Flush directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if DEBUG_OPENMP >= 1:
            print("visit flush_directive", args, type(args))

    def region_phrase(self, args):
        raise NotImplementedError("No implementation for region phrase.")
        if DEBUG_OPENMP >= 1:
            print("visit region_phrase", args, type(args))

    def PYTHON_NAME(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit PYTHON_NAME", args, type(args), str(args))
        return str(args)

    def NUMBER(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit NUMBER", args, type(args), str(args))
        return int(args)


openmp_grammar = r"""
    openmp_statement: openmp_construct
                    | openmp_directive
    openmp_directive: barrier_directive
                    | taskwait_directive
                    | taskyield_directive
                    | flush_directive
    barrier_directive: BARRIER
    taskwait_directive: TASKWAIT
    taskyield_directive: TASKYIELD
    BARRIER: "barrier"
    TASKWAIT: "taskwait"
    TASKYIELD: "taskyield"
    taskgroup_directive: TASKGROUP
    taskgroup_construct: taskgroup_directive
    TASKGROUP: "taskgroup"
    openmp_construct: parallel_construct
                    | parallel_for_construct
                    | for_construct
                    | single_construct
                    | task_construct
                    | teams_construct
                    | teams_distribute_construct
                    | teams_distribute_simd_construct
                    | teams_distribute_parallel_for_construct
                    | teams_distribute_parallel_for_simd_construct
                    | loop_construct
                    | teams_loop_construct
                    | target_construct
                    | target_teams_construct
                    | target_teams_distribute_construct
                    | target_teams_distribute_simd_construct
                    | target_teams_distribute_parallel_for_simd_construct
                    | target_teams_distribute_parallel_for_construct
                    | target_loop_construct
                    | target_teams_loop_construct
                    | target_enter_data_construct
                    | target_exit_data_construct
                    | distribute_construct
                    | distribute_simd_construct
                    | distribute_parallel_for_construct
                    | distribute_parallel_for_simd_construct
                    | critical_construct
                    | atomic_construct
                    | sections_construct
                    | section_construct
                    | simd_construct
                    | for_simd_construct
                    | parallel_for_simd_construct
                    | target_data_construct
                    | target_update_construct
                    | parallel_sections_construct
                    | master_construct
                    | ordered_construct
    for_simd_construct: for_simd_directive
    for_simd_directive: FOR SIMD [for_simd_clause*]
    for_simd_clause: for_clause
                   | simd_clause
    parallel_for_simd_construct: parallel_for_simd_directive
    parallel_for_simd_directive: PARALLEL FOR SIMD [parallel_for_simd_clause*]
    parallel_for_simd_clause: parallel_for_clause
                            | simd_clause
    distribute_construct: distribute_directive
    distribute_simd_construct: distribute_simd_directive
    distribute_directive: DISTRIBUTE [distribute_clause*]
    distribute_simd_directive: DISTRIBUTE SIMD [distribute_simd_clause*]
    distribute_clause: private_clause
                     | firstprivate_clause
              //     | lastprivate_distribute_clause
                     | collapse_clause
                     | dist_schedule_clause
                     | allocate_clause
    distribute_simd_clause: private_clause
                          | firstprivate_clause
                   //     | lastprivate_distribute_clause
                          | collapse_clause
                          | dist_schedule_clause
                          | allocate_clause
                          | if_clause
                   //     | safelen_clause
                   //     | simdlen_clause
                          | linear_clause
                          | aligned_clause
                   //     | nontemporal_clause
                          | reduction_clause
                   //     | order_clause

    teams_distribute_clause: num_teams_clause
                           | thread_limit_clause
                           | data_default_clause
                           | private_clause
                           | firstprivate_clause
                           | data_sharing_clause
                           | reduction_clause
                           | allocate_clause
                    //     | lastprivate_distribute_clause
                           | collapse_clause
                           | dist_schedule_clause
                           | ompx_attribute

    teams_distribute_simd_clause: num_teams_clause
                                | thread_limit_clause
                                | data_default_clause
                                | private_clause
                                | firstprivate_clause
                                | data_sharing_clause
                                | reduction_clause
                                | allocate_clause
                         //     | lastprivate_distribute_clause
                                | collapse_clause
                                | dist_schedule_clause
                                | if_clause
                         //     | safelen_clause
                         //     | simdlen_clause
                                | linear_clause
                                | aligned_clause
                         //     | nontemporal_clause
                         //     | order_clause
                                | ompx_attribute

    distribute_parallel_for_construct: distribute_parallel_for_directive
    distribute_parallel_for_directive: DISTRIBUTE PARALLEL FOR [distribute_parallel_for_clause*]
    distribute_parallel_for_clause: if_clause
                                  | num_threads_clause
                                  | data_default_clause
                                  | private_clause
                                  | firstprivate_clause
                                  | data_sharing_clause
                                  | reduction_clause
                                  | copyin_clause
                           //     | proc_bind_clause
                                  | allocate_clause
                                  | lastprivate_clause
                                  | linear_clause
                                  | schedule_clause
                                  | collapse_clause
                                  | ORDERED
                                  | NOWAIT
                           //     | order_clause
                                  | dist_schedule_clause

    distribute_parallel_for_simd_construct: distribute_parallel_for_simd_directive
    distribute_parallel_for_simd_directive: DISTRIBUTE PARALLEL FOR SIMD [distribute_parallel_for_simd_clause*]
    distribute_parallel_for_simd_clause: if_clause
                                  | num_threads_clause
                                  | data_default_clause
                                  | private_clause
                                  | firstprivate_clause
                                  | data_sharing_clause
                                  | reduction_clause
                                  | copyin_clause
                           //     | proc_bind_clause
                                  | allocate_clause
                                  | lastprivate_clause
                                  | linear_clause
                                  | schedule_clause
                                  | collapse_clause
                                  | ORDERED
                                  | NOWAIT
                           //     | order_clause
                                  | dist_schedule_clause
                           //     | safelen_clause
                           //     | simdlen_clause
                                  | aligned_clause
                           //     | nontemporal_clause

    target_data_construct: target_data_directive
    target_data_directive: TARGET DATA [target_data_clause*]
    DATA: "data"
    ENTER: "enter"
    EXIT: "exit"
    target_enter_data_construct: target_enter_data_directive
    target_enter_data_directive: TARGET ENTER DATA [target_enter_data_clause*]
    target_exit_data_construct: target_exit_data_directive
    target_exit_data_directive: TARGET EXIT DATA [target_exit_data_clause*]
    target_data_clause: device_clause
                      | map_clause
                      | if_clause
                      | NOWAIT
                      | depend_with_modifier_clause
    target_enter_data_clause: device_clause
                            | map_enter_clause
                            | if_clause
                            | NOWAIT
                            | depend_with_modifier_clause
    target_exit_data_clause: device_clause
                           | map_exit_clause
                           | if_clause
                           | NOWAIT
                           | depend_with_modifier_clause
    device_clause: "device" "(" const_num_or_var ")"
    map_clause: "map" "(" [map_type ":"] var_list ")"
    map_type: ALLOC | TO | FROM | TOFROM
    map_enter_clause: "map" "(" map_enter_type ":" var_list ")"
    map_enter_type: ALLOC | TO
    map_exit_clause: "map" "(" map_exit_type ":" var_list ")"
    map_exit_type: FROM | RELEASE | DELETE
    update_motion_type: TO | FROM
    TO: "to"
    FROM: "from"
    ALLOC: "alloc"
    TOFROM: "tofrom"
    RELEASE: "release"
    DELETE: "delete"
    parallel_sections_construct: parallel_sections_directive
    parallel_sections_directive: PARALLEL SECTIONS [parallel_sections_clause*]
    parallel_sections_clause: unique_parallel_clause
                            | data_default_clause
                            | private_clause
                            | firstprivate_clause
                            | lastprivate_clause
                            | data_sharing_clause
                            | reduction_clause
    sections_construct: sections_directive
    sections_directive: SECTIONS [sections_clause*]
    SECTIONS: "sections"
    sections_clause: private_clause
                   | firstprivate_clause
                   | lastprivate_clause
                   | reduction_clause
                   | NOWAIT
    section_construct: section_directive
    section_directive: SECTION
    SECTION: "section"
    atomic_construct: atomic_directive
    atomic_directive: ATOMIC [atomic_clause] [seq_cst_clause]
    ATOMIC: "atomic"
    atomic_clause: READ
                 | WRITE
                 | UPDATE
                 | CAPTURE
    READ: "read"
    WRITE: "write"
    UPDATE: "update"
    CAPTURE: "capture"
    seq_cst_clause: "seq_cst"
    critical_construct: critical_directive
    critical_directive: CRITICAL
    CRITICAL: "critical"
    teams_construct: teams_directive
    teams_directive: TEAMS [teams_clause*]
    teams_distribute_directive: TEAMS DISTRIBUTE [teams_distribute_clause*]
    teams_distribute_simd_directive: TEAMS DISTRIBUTE SIMD [teams_distribute_simd_clause*]
    target_construct: target_directive
    target_teams_distribute_parallel_for_simd_construct: target_teams_distribute_parallel_for_simd_directive
    target_teams_distribute_parallel_for_construct: target_teams_distribute_parallel_for_directive
    teams_distribute_parallel_for_construct: teams_distribute_parallel_for_directive
    teams_distribute_parallel_for_simd_construct: teams_distribute_parallel_for_simd_directive
    loop_construct: loop_directive
    teams_loop_construct: teams_loop_directive
    target_loop_construct: target_loop_directive
    target_teams_loop_construct: target_teams_loop_directive
    target_teams_construct: target_teams_directive
    target_teams_distribute_construct: target_teams_distribute_directive
    target_teams_distribute_simd_construct: target_teams_distribute_simd_directive
    teams_distribute_construct: teams_distribute_directive
    teams_distribute_simd_construct: teams_distribute_simd_directive
    target_directive: TARGET [target_clause*]
    HAS_DEVICE_ADDR: "has_device_addr"
    has_device_addr_clause: HAS_DEVICE_ADDR "(" var_list ")"
    target_clause: if_clause
                 | device_clause
                 | thread_limit_clause
                 | private_clause
                 | firstprivate_clause
          //     | in_reduction_clause
                 | map_clause
                 | is_device_ptr_clause
                 | has_device_addr_clause
          //     | defaultmap_clause
                 | NOWAIT
                 | allocate_clause
                 | depend_with_modifier_clause
          //     | uses_allocators_clause
                 | ompx_attribute
    teams_clause: num_teams_clause
                | thread_limit_clause
                | data_default_clause
                | private_clause
                | firstprivate_clause
                | data_sharing_clause
                | reduction_clause
                | allocate_clause
    num_teams_clause: NUM_TEAMS "(" const_num_or_var ")"
    NUM_TEAMS: "num_teams"
    thread_limit_clause: THREAD_LIMIT "(" const_num_or_var ")"
    THREAD_LIMIT: "thread_limit"

    dist_schedule_expr: DIST_SCHEDULE "(" STATIC ")"
    dist_schedule_no_expr: DIST_SCHEDULE "(" STATIC "," const_num_or_var ")"
    dist_schedule_clause: dist_schedule_expr
                        | dist_schedule_no_expr
    DIST_SCHEDULE: "dist_schedule"

    target_teams_distribute_parallel_for_simd_directive: TARGET TEAMS DISTRIBUTE PARALLEL FOR SIMD [target_teams_distribute_parallel_for_simd_clause*]
    target_teams_distribute_parallel_for_simd_clause: if_clause
                                                    | device_clause
                                                    | private_clause
                                                    | firstprivate_clause
                                             //     | in_reduction_clause
                                                    | map_clause
                                                    | is_device_ptr_clause
                                             //     | defaultmap_clause
                                                    | NOWAIT
                                                    | allocate_clause
                                                    | depend_with_modifier_clause
                                             //     | uses_allocators_clause
                                                    | num_teams_clause
                                                    | thread_limit_clause
                                                    | data_default_clause
                                                    | data_sharing_clause
                                                    | reduction_clause
                                                    | num_threads_clause
                                                    | copyin_clause
                                             //     | proc_bind_clause
                                                    | lastprivate_clause
                                                    | linear_clause
                                                    | schedule_clause
                                                    | collapse_clause
                                                    | ORDERED
                                             //     | order_clause
                                                    | dist_schedule_clause
                                             //     | safelen_clause
                                             //     | simdlen_clause
                                                    | aligned_clause
                                             //     | nontemporal_clause
                                                    | ompx_attribute

    teams_distribute_parallel_for_simd_directive: TEAMS DISTRIBUTE PARALLEL FOR SIMD [teams_distribute_parallel_for_simd_clause*]
    teams_distribute_parallel_for_simd_clause: num_teams_clause
                                             | thread_limit_clause
                                      //     | default_clause
                                             | private_clause
                                             | firstprivate_clause
                                             | data_sharing_clause
                                             | reduction_clause
                                             | if_clause
                                             | num_threads_clause
                                             | copyin_clause
                                      //     | proc_bind_clause
                                             | lastprivate_clause
                                             | linear_clause
                                             | schedule_clause
                                             | collapse_clause
                                             | ORDERED
                                             | NOWAIT
                                      //     | order_clause
                                             | dist_schedule_clause
                                      //     | safelen_clause
                                      //     | simdlen_clause
                                             | aligned_clause
                                      //     | nontemporal_clause
                                      //     | in_reduction_clause
                                             | map_clause
                                             | is_device_ptr_clause
                                      //     | defaultmap_clause
                                             | allocate_clause
                                             | depend_with_modifier_clause
                                      //     | uses_allocators_clause
                                             | data_default_clause
                                             | ompx_attribute

    target_teams_distribute_parallel_for_directive: TARGET TEAMS DISTRIBUTE PARALLEL FOR [target_teams_distribute_parallel_for_clause*]
    target_teams_distribute_parallel_for_clause: if_clause
                                               | device_clause
                                               | private_clause
                                               | firstprivate_clause
                                        //     | in_reduction_clause
                                               | map_clause
                                               | is_device_ptr_clause
                                        //     | defaultmap_clause
                                               | NOWAIT
                                               | allocate_clause
                                               | depend_with_modifier_clause
                                        //     | uses_allocators_clause
                                               | num_teams_clause
                                               | thread_limit_clause
                                               | data_default_clause
                                               | data_sharing_clause
                                               | reduction_clause
                                               | num_threads_clause
                                               | copyin_clause
                                        //     | proc_bind_clause
                                               | lastprivate_clause
                                               | linear_clause
                                               | schedule_clause
                                               | collapse_clause
                                               | ORDERED
                                        //     | order_clause
                                               | dist_schedule_clause
                                               | ompx_attribute

    teams_distribute_parallel_for_directive: TEAMS DISTRIBUTE PARALLEL FOR [teams_distribute_parallel_for_clause*]
    teams_distribute_parallel_for_clause: num_teams_clause
                                        | thread_limit_clause
                                        | data_default_clause
                                        | private_clause
                                        | firstprivate_clause
                                        | data_sharing_clause
                                        | reduction_clause
                                        | allocate_clause
                                        | if_clause
                                        | num_threads_clause
                                        | copyin_clause
                                 //     | proc_bind_clause
                                        | lastprivate_clause
                                        | linear_clause
                                        | schedule_clause
                                        | collapse_clause
                                        | ORDERED
                                        | NOWAIT
                                 //     | order_clause
                                        | dist_schedule_clause
                                        | ompx_attribute

    LOOP: "loop"

    ompx_attribute: OMPX_ATTRIBUTE "(" PYTHON_NAME "(" number_list ")" ")"
    OMPX_ATTRIBUTE: "ompx_attribute"
    loop_directive: LOOP [teams_distribute_parallel_for_clause*]
    teams_loop_directive: TEAMS LOOP [teams_distribute_parallel_for_clause*]
    target_loop_directive: TARGET LOOP [target_teams_distribute_parallel_for_clause*]
    target_teams_loop_directive: TARGET TEAMS LOOP [target_teams_distribute_parallel_for_clause*]

    target_teams_directive: TARGET TEAMS [target_teams_clause*]
    target_teams_clause: if_clause
                       | device_clause
                       | private_clause
                       | firstprivate_clause
                //     | in_reduction_clause
                       | map_clause
                       | is_device_ptr_clause
                //     | defaultmap_clause
                       | NOWAIT
                       | allocate_clause
                       | depend_with_modifier_clause
                //     | uses_allocators_clause
                       | num_teams_clause
                       | thread_limit_clause
                       | data_default_clause
                       | data_sharing_clause
                //     | reduction_default_only_clause
                       | reduction_clause
                       | ompx_attribute

    target_teams_distribute_simd_directive: TARGET TEAMS DISTRIBUTE SIMD [target_teams_distribute_simd_clause*]
    target_teams_distribute_simd_clause: if_clause
                                       | device_clause
                                       | private_clause
                                       | firstprivate_clause
                                //     | in_reduction_clause
                                       | map_clause
                                       | is_device_ptr_clause
                                //     | defaultmap_clause
                                       | NOWAIT
                                       | allocate_clause
                                       | depend_with_modifier_clause
                                //     | uses_allocators_clause
                                       | num_teams_clause
                                       | thread_limit_clause
                                       | data_default_clause
                                       | data_sharing_clause
                                       | reduction_clause
                                //     | reduction_default_only_clause
                                       | lastprivate_clause
                                       | collapse_clause
                                       | dist_schedule_clause
                                //     | safelen_clause
                                //     | simdlen_clause
                                       | linear_clause
                                       | aligned_clause
                                //     | nontemporal_clause
                                //     | order_clause
                                       | ompx_attribute

    target_teams_distribute_directive: TARGET TEAMS DISTRIBUTE [target_teams_distribute_clause*]
    target_teams_distribute_clause: if_clause
                                  | device_clause
                                  | private_clause
                                  | firstprivate_clause
                           //     | in_reduction_clause
                                  | map_clause
                                  | is_device_ptr_clause
                           //     | defaultmap_clause
                                  | NOWAIT
                                  | allocate_clause
                                  | depend_with_modifier_clause
                           //     | uses_allocators_clause
                                  | num_teams_clause
                                  | thread_limit_clause
                                  | data_default_clause
                                  | data_sharing_clause
                                  | reduction_clause
                                  | lastprivate_clause
                                  | collapse_clause
                                  | dist_schedule_clause
                                  | ompx_attribute

    IS_DEVICE_PTR: "is_device_ptr"
    is_device_ptr_clause: IS_DEVICE_PTR "(" var_list ")"
    allocate_clause: ALLOCATE "(" allocate_parameter ")"
    ALLOCATE: "allocate"
    allocate_parameter: [const_num_or_var] var_list

    target_update_construct: target_update_directive
    target_update_directive: TARGET UPDATE target_update_clause*
    target_update_clause: motion_clause
                        | device_clause
                        | if_clause
    motion_clause: update_motion_type "(" variable_array_section_list ")"
    variable_array_section_list: PYTHON_NAME
                           //    | array_section
                               | name_slice
                               | variable_array_section_list "," PYTHON_NAME
                               | variable_array_section_list "," name_slice
                           //    | variable_array_section_list "," array_section
    //array_section: PYTHON_NAME array_section_subscript
    //array_section_subscript: array_section_subscript "[" [const_num_or_var] ":" [const_num_or_var] "]"
    //                       | array_section_subscript "[" const_num_or_var "]"
    //                       | "[" [const_num_or_var] ":" [const_num_or_var] "]"
    //                       | "[" const_num_or_var "]"
    TARGET: "target"
    TEAMS: "teams"
    DISTRIBUTE: "distribute"
    single_construct: single_directive
    single_directive: SINGLE [single_clause*]
    SINGLE: "single"
    single_clause: unique_single_clause
                 | private_clause
                 | firstprivate_clause
                 | NOWAIT
    unique_single_clause: copyprivate_clause
    NOWAIT: "nowait"
    master_construct: master_directive
    master_directive: "master"
    simd_construct: simd_directive
    simd_directive: SIMD [simd_clause*]
    SIMD: "simd"
    simd_clause: collapse_clause
               | aligned_clause
               | linear_clause
               | uniform_clause
               | reduction_clause
               | inbranch_clause
    aligned_clause: ALIGNED "(" var_list ")"
                  | ALIGNED "(" var_list ":" const_num_or_var ")"
    declare_simd_construct: declare_simd_directive_seq
    declare_simd_directive_seq: declare_simd_directive
                              | declare_simd_directive_seq declare_simd_directive
    declare_simd_directive: SIMD [declare_simd_clause*]
    declare_simd_clause: "simdlen" "(" const_num_or_var ")"
                       | aligned_clause
                       | linear_clause
                       | uniform_clause
                       | reduction_clause
                       | inbranch_clause
    ALIGNED: "aligned"
    inbranch_clause: INBRANCH | NOTINBRANCH
    INBRANCH: "inbranch"
    NOTINBRANCH: "notinbranch"
    uniform_clause: UNIFORM "(" var_list ")"
    UNIFORM: "uniform"
    collapse_clause: COLLAPSE "(" const_num_or_var ")"
    COLLAPSE: "collapse"
    task_construct: task_directive
    TASK: "task"
    task_directive: TASK [task_clause*]
    task_clause: unique_task_clause
               | data_sharing_clause
               | private_clause
               | firstprivate_clause
               | data_default_clause
    unique_task_clause: if_clause
                      | UNTIED
                      | MERGEABLE
                      | FINAL "(" const_num_or_var ")"
                      | depend_with_modifier_clause
    DEPEND: "depend"
    FINAL: "final"
    UNTIED: "untied"
    MERGEABLE: "mergeable"
    dependence_type: IN
                   | OUT
                   | INOUT
    depend_with_modifier_clause: DEPEND "(" dependence_type ":" variable_array_section_list ")"
    IN: "in"
    OUT: "out"
    INOUT: "inout"
    data_default_clause: default_shared_clause
                       | default_none_clause
    data_sharing_clause: shared_clause
    data_clause: private_clause
               | copyprivate_clause
               | firstprivate_clause
               | lastprivate_clause
               | data_sharing_clause
               | data_default_clause
               | copyin_clause
               | reduction_clause
    private_clause: PRIVATE "(" var_list ")"
    PRIVATE: "private"
    copyprivate_clause: COPYPRIVATE "(" var_list ")"
    COPYPRIVATE: "copyprivate"
    firstprivate_clause: FIRSTPRIVATE "(" var_list ")"
    FIRSTPRIVATE: "firstprivate"
    lastprivate_clause: LASTPRIVATE "(" var_list ")"
    LASTPRIVATE: "lastprivate"
    shared_clause: SHARED "(" var_list ")"
    SHARED: "shared"
    copyin_clause: COPYIN "(" var_list ")"
    COPYIN: "copyin"
    REDUCTION: "reduction"
    DEFAULT: "default"
    reduction_clause: REDUCTION "(" reduction_operator ":" var_list ")"
    default_shared_clause: DEFAULT "(" "shared" ")"
    default_none_clause: DEFAULT "(" "none" ")"
    const_num_or_var: NUMBER | PYTHON_NAME
    parallel_construct: parallel_directive
    parallel_directive: PARALLEL [parallel_clause*]
    parallel_clause: unique_parallel_clause
                   | data_default_clause
                   | private_clause
                   | firstprivate_clause
                   | data_sharing_clause
                   | reduction_clause
    unique_parallel_clause: if_clause | num_threads_clause
    if_clause: IF "(" const_num_or_var ")"
    IF: "if"
    num_threads_clause: NUM_THREADS "(" const_num_or_var ")"
    NUM_THREADS: "num_threads"
    PARALLEL: "parallel"
    FOR: "for"
    parallel_for_construct: parallel_for_directive
    parallel_for_directive: PARALLEL FOR [parallel_for_clause*]
    parallel_for_clause: unique_parallel_clause
                       | unique_for_clause
                       | data_default_clause
                       | private_clause
                       | firstprivate_clause
                       | lastprivate_clause
                       | data_sharing_clause
                       | reduction_clause
    for_construct: for_directive
    for_directive: FOR [for_clause*]
    for_clause: unique_for_clause | data_clause | NOWAIT
    unique_for_clause: ORDERED
                     | schedule_clause
                     | collapse_clause
    LINEAR: "linear"
    linear_clause: LINEAR "(" var_list ":" const_num_or_var ")"
                 | LINEAR "(" var_list ")"
    sched_no_expr: SCHEDULE "(" schedule_kind ")"
    sched_expr: SCHEDULE "(" schedule_kind "," const_num_or_var ")"
    schedule_clause: sched_no_expr
                   | sched_expr
    SCHEDULE: "schedule"
    schedule_kind: STATIC | DYNAMIC | GUIDED | RUNTIME | AUTO
    STATIC: "static"
    DYNAMIC: "dynamic"
    GUIDED: "guided"
    RUNTIME: "runtime"
    AUTO: "auto"
    COLON: ":"
    oslice: [const_num_or_var] COLON [const_num_or_var]
    slice_list: oslice | slice_list "," oslice
    name_slice: PYTHON_NAME [ "[" slice_list "]" ]
    var_list: name_slice | var_list "," name_slice
    number_list: NUMBER | number_list "," NUMBER
    PLUS: "+"
    MINUS: "-"
    STAR: "*"
    reduction_operator: PLUS | "\\" | STAR | MINUS | "&" | "^" | "|" | "&&" | "||"
    threadprivate_directive: "threadprivate" "(" var_list ")"
    cancellation_point_directive: "cancellation point" construct_type_clause
    construct_type_clause: PARALLEL
                         | SECTIONS
                         | FOR
                         | TASKGROUP
    cancel_directive: "cancel" construct_type_clause [if_clause]
    ordered_directive: ORDERED
    ordered_construct: ordered_directive
    ORDERED: "ordered"
    flush_directive: "flush" "(" var_list ")"

    region_phrase: "(" PYTHON_NAME ")"
    PYTHON_NAME: /[a-zA-Z_]\w*/

    %import common.NUMBER
    %import common.WS
    %ignore WS
    """

"""
    name_slice: PYTHON_NAME [ "[" slice ["," slice]* "]" ]
"""

openmp_parser = Lark(openmp_grammar, start="openmp_statement")
var_collector_parser = Lark(openmp_grammar, start="openmp_statement")


def remove_ssa_callback(var, unused):
    assert isinstance(var, ir.Var)
    new_var = ir.Var(var.scope, var.unversioned_name, var.loc)
    return new_var


def remove_ssa_from_func_ir(func_ir):
    typed_passes.PreLowerStripPhis()._strip_phi_nodes(func_ir)
    #    new_func_ir = typed_passes.PreLowerStripPhis()._strip_phi_nodes(func_ir)
    #    func_ir.blocks = new_func_ir.blocks
    visit_vars(func_ir.blocks, remove_ssa_callback, None)
    func_ir._definitions = build_definitions(func_ir.blocks)


def _add_openmp_ir_nodes(func_ir, blocks, blk_start, blk_end, body_blocks, extra):
    """Given the starting and ending block of the with-context,
    replaces the head block with a new block that has the starting
    openmp ir nodes in it and adds the ending openmp ir nodes to
    the end block.
    """
    sblk = blocks[blk_start]
    loc = sblk.loc
    sblk.body = sblk.body[1:]

    args = extra["args"]
    arg = args[0]
    # If OpenMP argument is not a constant or not a string then raise exception
    if not isinstance(arg, (ir.Const, ir.FreeVar)):
        raise NonconstantOpenmpSpecification(
            f"Non-constant OpenMP specification at line {arg.loc}"
        )
    if not isinstance(arg.value, str):
        raise NonStringOpenmpSpecification(
            f"Non-string OpenMP specification at line {arg.loc}"
        )

    if DEBUG_OPENMP >= 1:
        print("args:", args, type(args))
        print("arg:", arg, type(arg), arg.value, type(arg.value))
    parse_res = openmp_parser.parse(arg.value)
    if DEBUG_OPENMP >= 1:
        print(parse_res.pretty())
    visitor = OpenmpVisitor(func_ir, blocks, blk_start, blk_end, body_blocks, loc)
    try:
        visitor.transform(parse_res)
    except VisitError as e:
        raise e.__context__
        if isinstance(e.__context__, UnspecifiedVarInDefaultNone):
            print(str(e.__context__))
            raise e.__context__
        else:
            print(
                "Internal error for OpenMp pragma '{}'".format(arg.value),
                e.__context__,
                type(e.__context__),
            )
        sys.exit(-1)
    except Exception as f:
        print("generic transform exception")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("Internal error for OpenMp pragma '{}'".format(arg.value))
        sys.exit(-2)
    except:
        print("fallthrough exception")
        print("Internal error for OpenMP pragma '{}'".format(arg.value))
        sys.exit(-3)
    assert blocks is visitor.blocks


class OpenmpExternalFunction(types.ExternalFunction):
    def __call__(self, *args):
        import inspect

        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        if mod.__name__.startswith("numba") and not mod.__name__.startswith(
            "numba.openmp.tests"
        ):
            return super(ExternalFunction, self).__call__(*args)

        ffi = FFI()
        fname = self.symbol
        ret_typ = str(self.sig.return_type)

        def numba_to_c(ret_typ):
            if ret_typ == "int32":
                return "int"
            elif ret_typ == "none":
                return "void"
            elif ret_typ == "float64":
                return "double"
            else:
                assert False

        ret_typ = numba_to_c(ret_typ)
        arg_str = ",".join([numba_to_c(str(x)) for x in self.sig.args])
        proto = f"{ret_typ} {fname}({arg_str});"
        ffi.cdef(proto)
        # Should be loaded into the process by the load_library_permanently
        # at the top of this file.
        C = ffi.dlopen(None)
        return getattr(C, fname)(*args)


model_register(OpenmpExternalFunction)(OpaqueModel)

omp_set_num_threads = OpenmpExternalFunction(
    "omp_set_num_threads", types.void(types.int32)
)
omp_get_thread_num = OpenmpExternalFunction("omp_get_thread_num", types.int32())
omp_get_num_threads = OpenmpExternalFunction("omp_get_num_threads", types.int32())
omp_get_wtime = OpenmpExternalFunction("omp_get_wtime", types.float64())
omp_set_dynamic = OpenmpExternalFunction("omp_set_dynamic", types.void(types.int32))
omp_set_nested = OpenmpExternalFunction("omp_set_nested", types.void(types.int32))
omp_set_max_active_levels = OpenmpExternalFunction(
    "omp_set_max_active_levels", types.void(types.int32)
)
omp_get_max_active_levels = OpenmpExternalFunction(
    "omp_get_max_active_levels", types.int32()
)
omp_get_max_threads = OpenmpExternalFunction("omp_get_max_threads", types.int32())
omp_get_num_procs = OpenmpExternalFunction("omp_get_num_procs", types.int32())
omp_in_parallel = OpenmpExternalFunction("omp_in_parallel", types.int32())
omp_get_thread_limit = OpenmpExternalFunction("omp_get_thread_limit", types.int32())
omp_get_supported_active_levels = OpenmpExternalFunction(
    "omp_get_supported_active_levels", types.int32()
)
omp_get_level = OpenmpExternalFunction("omp_get_level", types.int32())
omp_get_active_level = OpenmpExternalFunction("omp_get_active_level", types.int32())
omp_get_ancestor_thread_num = OpenmpExternalFunction(
    "omp_get_ancestor_thread_num", types.int32(types.int32)
)
omp_get_team_size = OpenmpExternalFunction(
    "omp_get_team_size", types.int32(types.int32)
)
omp_in_final = OpenmpExternalFunction("omp_in_finale", types.int32())
omp_get_proc_bind = OpenmpExternalFunction("omp_get_proc_bind", types.int32())
omp_get_num_places = OpenmpExternalFunction("omp_get_num_places", types.int32())
omp_get_place_num_procs = OpenmpExternalFunction(
    "omp_get_place_num_procs", types.int32(types.int32)
)
omp_get_place_num = OpenmpExternalFunction("omp_get_place_num", types.int32())
omp_set_default_device = OpenmpExternalFunction(
    "omp_set_default_device", types.int32(types.int32)
)
omp_get_default_device = OpenmpExternalFunction("omp_get_default_device", types.int32())
omp_get_num_devices = OpenmpExternalFunction("omp_get_num_devices", types.int32())
omp_get_device_num = OpenmpExternalFunction("omp_get_device_num", types.int32())
omp_get_team_num = OpenmpExternalFunction("omp_get_team_num", types.int32())
omp_get_num_teams = OpenmpExternalFunction("omp_get_num_teams", types.int32())
omp_is_initial_device = OpenmpExternalFunction("omp_is_initial_device", types.int32())
omp_get_initial_device = OpenmpExternalFunction("omp_get_initial_device", types.int32())


def copy_np_array(x):
    return np.copy(x)


# {meminfo, parent, ...} copy_np_array({meminfo,  parent, ...})


def create_native_np_copy(arg_typ):
    # The cfunc wrapper of this function is what we need.
    copy_cres = compiler.compile_isolated(copy_np_array, (arg_typ,), arg_typ)
    copy_name = getattr(copy_cres.fndesc, "llvm_cfunc_wrapper_name")
    return (copy_name, copy_cres)


def omp_shared_array(size, dtype):
    return np.empty(size, dtype=dtype)


@overload(omp_shared_array, target="cpu", inline="always", prefer_literal=True)
def omp_shared_array_overload(size, dtype):
    assert isinstance(size, types.IntegerLiteral)

    def impl(size, dtype):
        return np.empty(size, dtype=dtype)

    return impl


@overload(omp_shared_array, target="cuda", inline="always", prefer_literal=True)
def omp_shared_array_overload(size, dtype):
    assert isinstance(size, types.IntegerLiteral)

    def impl(size, dtype):
        return numba_cuda.shared.array(size, dtype)

    return impl
