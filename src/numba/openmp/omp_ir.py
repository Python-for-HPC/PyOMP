from numba.core import (
    ir,
    types,
    cgutils,
    typing,
    transforms,
    bytecode,
    compiler,
    typeinfer,
)
from numba.core.ir_utils import (
    dprint_func_ir,
    find_topo_order,
    mk_unique_var,
    apply_copy_propagate_extensions,
    visit_vars_extensions,
    visit_vars_inner,
)
from numba import cuda as numba_cuda
from numba.cuda import descriptor as cuda_descriptor, compiler as cuda_compiler
from numba.core.types.functions import Dispatcher
from numba.core.analysis import ir_extension_usedefs, _use_defs_result
import numba
import llvmlite.ir as lir
import llvmlite.binding as ll
import sys
import os
import copy
import tempfile
import subprocess
import operator
import numpy as np
from pathlib import Path
import types as python_types

from .analysis import (
    is_dsa,
    typemap_lookup,
    is_target_tag,
    is_target_arg,
    in_openmp_region,
    get_blocks_between_start_end,
    get_name_var_table,
    is_pointer_target_arg,
)
from .tags import (
    openmp_tag_list_to_str,
    list_vars_from_tags,
    get_tags_of_type,
    StringLiteral,
    openmp_tag,
    NameSlice,
)
from .llvmlite_extensions import TokenType, CallInstrWithOperandBundle
from .config import (
    libpath,
    DEBUG_OPENMP,
    DEBUG_OPENMP_LLVM_PASS,
    OPENMP_DEVICE_TOOLCHAIN,
)
from .link_utils import link_shared_library
from .llvm_pass import run_intrinsics_openmp_pass
from .compiler import (
    OnlyLower,
    OnlyLowerCUDA,
    OpenmpCPUTargetContext,
    OpenmpCUDATargetContext,
    CustomAOTCPUCodeLibrary,
    CustomCPUCodeLibrary,
    CustomContext,
)

unique = 0


def get_unique():
    global unique
    ret = unique
    unique += 1
    return ret


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


class OpenMPCUDACodegen:
    def __init__(self):
        import numba.cuda.api as cudaapi
        import numba.cuda.cudadrv.libs as cudalibs
        from numba.cuda.codegen import CUDA_TRIPLE

        self.cc = cudaapi.get_current_device().compute_capability
        self.sm = "sm_" + str(self.cc[0]) + str(self.cc[1])

        # Read the libdevice bitcode for the architecture to link with the module.
        self.libdevice_path = cudalibs.get_libdevice()
        with open(self.libdevice_path, "rb") as f:
            self.libdevice_mod = ll.parse_bitcode(f.read())

        # Read the OpenMP device RTL for the architecture to link with the module.
        self.libomptarget_arch = (
            libpath / "libomp" / "lib" / f"libomptarget-nvptx-{self.sm}.bc"
        )
        try:
            with open(self.libomptarget_arch, "rb") as f:
                self.libomptarget_mod = ll.parse_bitcode(f.read())
        except FileNotFoundError:
            raise RuntimeError(
                f"Device RTL for architecture {self.sm} not found. Check compute capability with LLVM version {'.'.join(map(str, ll.llvm_version_info))}."
            )

        # Initialize asm printers to codegen ptx.
        ll.initialize_all_targets()
        ll.initialize_all_asmprinters()
        target = ll.Target.from_triple(CUDA_TRIPLE)
        # We pick opt=2 as a reasonable optimization level for codegen.
        self.tm = target.create_target_machine(cpu=self.sm, opt=2)

    def _get_target_image(self, mod, filename_prefix, ompx_attrs, use_toolchain=False):
        from numba.cuda.cudadrv import driver
        from numba.core.llvm_bindings import create_pass_manager_builder

        if DEBUG_OPENMP_LLVM_PASS >= 1:
            with open(filename_prefix + ".ll", "w") as f:
                f.write(str(mod))

        # Lower openmp intrinsics.
        mod = run_intrinsics_openmp_pass(mod)
        if DEBUG_OPENMP_LLVM_PASS >= 1:
            with open(filename_prefix + "-intr.ll", "w") as f:
                f.write(str(mod))

        def _internalize():
            # Internalize non-kernel function definitions.
            for func in mod.functions:
                if func.is_declaration:
                    continue
                if func.linkage != ll.Linkage.external:
                    continue
                if "__omp_offload_numba" in func.name:
                    continue
                func.linkage = "internal"

        # Link first libdevice and optimize aggressively with opt=2 as a
        # reasonable optimization default.
        mod.link_in(self.libdevice_mod, preserve=True)
        # Internalize non-kernel function definitions.
        _internalize()
        # Run passes for optimization, including target-specific passes.
        # Run function passes.
        with ll.create_function_pass_manager(mod) as pm:
            self.tm.add_analysis_passes(pm)
            with create_pass_manager_builder(
                opt=2, slp_vectorize=True, loop_vectorize=True
            ) as pmb:
                pmb.populate(pm)
            pm.initialize()
            for func in mod.functions:
                pm.run(func)
            pm.finalize()

        # Run module passes.
        with ll.create_module_pass_manager() as pm:
            self.tm.add_analysis_passes(pm)
            with create_pass_manager_builder(
                opt=2, slp_vectorize=True, loop_vectorize=True
            ) as pmb:
                pmb.populate(pm)
            pm.run(mod)

        if DEBUG_OPENMP_LLVM_PASS >= 1:
            mod.verify()
            with open(filename_prefix + "-intr-dev.ll", "w") as f:
                f.write(str(mod))

        # Link in OpenMP device RTL and optimize lightly, with opt=1 to avoid
        # aggressive optimization can break openmp execution synchronization for
        # target regions.
        mod.link_in(self.libomptarget_mod, preserve=True)
        # Internalize non-kernel function definitions.
        _internalize()
        # Run module passes.
        with ll.create_module_pass_manager() as pm:
            self.tm.add_analysis_passes(pm)
            with create_pass_manager_builder(
                opt=1, slp_vectorize=True, loop_vectorize=True
            ) as pmb:
                pmb.populate(pm)
            pm.run(mod)

        if DEBUG_OPENMP_LLVM_PASS >= 1:
            mod.verify()
            with open(filename_prefix + "-intr-dev-rtl.ll", "w") as f:
                f.write(str(mod))

        # Generate ptx assemlby.
        ptx = self.tm.emit_assembly(mod)
        if use_toolchain:
            # ptxas does file I/O, so output the assembly and ingest the generated cubin.
            with open(filename_prefix + "-intr-dev-rtl.s", "w") as f:
                f.write(ptx)

            subprocess.run(
                [
                    "ptxas",
                    "-m64",
                    "--gpu-name",
                    self.sm,
                    filename_prefix + "-intr-dev-rtl.s",
                    "-o",
                    filename_prefix + "-intr-dev-rtl.o",
                ],
                check=True,
            )

            with open(filename_prefix + "-intr-dev-rtl.o", "rb") as f:
                cubin = f.read()
        else:
            if DEBUG_OPENMP_LLVM_PASS >= 1:
                with open(
                    filename_prefix + "-intr-dev-rtl.s",
                    "w",
                ) as f:
                    f.write(ptx)

            linker_kwargs = {}
            for x in ompx_attrs:
                linker_kwargs[x.arg[0]] = (
                    tuple(x.arg[1]) if len(x.arg[1]) > 1 else x.arg[1][0]
                )
            # NOTE: DO NOT set cc, since the linker will always
            # compile for the existing GPU context and it is
            # incompatible with the launch_bounds ompx_attribute.
            linker = driver.Linker.new(**linker_kwargs)
            linker.add_ptx(ptx.encode())
            cubin = linker.complete()

            if DEBUG_OPENMP_LLVM_PASS >= 1:
                with open(
                    filename_prefix + "-intr-dev-rtl.o",
                    "wb",
                ) as f:
                    f.write(cubin)

        return cubin

    def get_target_image(self, cres, ompx_attrs):
        filename_prefix = cres.library.name
        allmods = cres.library.modules
        linked_mod = ll.parse_assembly(str(allmods[0]))
        for mod in allmods[1:]:
            linked_mod.link_in(ll.parse_assembly(str(mod)))
        if OPENMP_DEVICE_TOOLCHAIN >= 1:
            return self._get_target_image(
                linked_mod, filename_prefix, ompx_attrs, use_toolchain=True
            )
        else:
            return self._get_target_image(linked_mod, filename_prefix, ompx_attrs)


_omp_cuda_codegen = None


# Accessor for the singleton OpenMPCUDACodegen instance. Initializes the
# instance on first use to ensure a single CUDA context and codegen setup
# per process.
def get_omp_cuda_codegen():
    global _omp_cuda_codegen
    if _omp_cuda_codegen is None:
        _omp_cuda_codegen = OpenMPCUDACodegen()
    return _omp_cuda_codegen


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
                            ir.Global("np", np, stmt.loc),
                            ir.Var(
                                stmt.target.scope, mk_unique_var(".np_global"), stmt.loc
                            ),
                            stmt.loc,
                        )
                    )
                    typemap[new_block_body[-1].target.name] = types.Module(np)
                    new_block_body.append(
                        ir.Assign(
                            ir.Expr.getattr(
                                new_block_body[-1].target, str(dtype_to_use), stmt.loc
                            ),
                            ir.Var(
                                stmt.target.scope, mk_unique_var(".np_dtype"), stmt.loc
                            ),
                            stmt.loc,
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


def remove_dels(blocks):
    """remove ir.Del nodes"""
    for block in blocks.values():
        new_body = []
        for stmt in block.body:
            if not isinstance(stmt, ir.Del):
                new_body.append(stmt)
        block.body = new_body
    return


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
            or not cctyp.pyomp_patch_installed
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
            or not cemtyp.pyomp_patch_installed
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

        if not hasattr(ptyp, "pyomp_patch_installed") or not ptyp.pyomp_patch_installed:
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

        elif target_num is not None and not self.target_copy:
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

            from numba.core.compiler import Flags

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

            # NOTE: workaround for python 3.10 lowering in numba that may
            # include a branch converging variable $cp. Remove it to avoid the
            # assert since the openmp region must be single-entry, single-exit.
            if sys.version_info >= (3, 10) and sys.version_info < (3, 11):
                assert len(target_args) == len(
                    [x for x in target_args_unordered if x != "$cp"]
                )
            else:
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
                target_extension._active_context.target = orig_target
                omp_cuda_cg = get_omp_cuda_codegen()
                target_elf = omp_cuda_cg.get_target_image(cres, ompx_attrs)
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
        builder = lowerer.builder

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

            assert self.start_region.omp_region_var is not None
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


class default_shared_val:
    def __init__(self, val):
        self.val = val


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
