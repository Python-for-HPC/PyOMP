from numba.core import compiler, compiler_machinery, cpu, ir, types
from numba import cuda as numba_cuda
from numba.core.controlflow import CFGraph
from numba.cuda import descriptor as cuda_descriptor
from numba.cuda.target import CUDACallConv
from numba.core.lowering import Lower
from functools import cached_property
from numba.core.callconv import (
    RETCODE_OK,
)

from numba.core.codegen import AOTCodeLibrary, JITCodeLibrary
from numba.core.dispatcher import _FunctionCompiler
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
import llvmlite.binding as ll
import llvmlite.ir as lir

from .config import DEBUG_OPENMP
from .llvm_pass import run_intrinsics_openmp_pass


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


class CustomContext(cpu.CPUContext):
    def post_lowering(self, mod, library):
        if hasattr(library, "openmp") and library.openmp:
            post_lowering_openmp(mod)
            super().post_lowering(mod, library)


class OpenmpCPUTargetContext(CustomContext):
    def __init__(self, name, typingctx, target="cpu"):
        super().__init__(typingctx, target)
        self.device_func_name = name


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


class CustomCPUCodeLibrary(JITCodeLibrary):
    def add_llvm_module(self, ll_module):
        lowered_module = run_intrinsics_openmp_pass(ll_module)
        super().add_llvm_module(lowered_module)

    def _finalize_specific(self):
        super()._finalize_specific()
        ll.ExecutionEngine.run_static_constructors(self._codegen._engine._ee)


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


class CustomAOTCPUCodeLibrary(AOTCodeLibrary):
    def add_llvm_module(self, ll_module):
        lowered_module = run_intrinsics_openmp_pass(ll_module)
        super().add_llvm_module(lowered_module)
