from numba.core import ir
from numba.core.ir_utils import (
    build_definitions,
    get_definition,
    dump_blocks,
    dprint_func_ir,
    compute_cfg_from_blocks,
    compute_use_defs,
    compute_live_map,
)
from numba.core.withcontexts import WithContext
import sys
import os


from .parser import var_collector_parser
from .analysis import get_name_var_table
from .config import DEBUG_OPENMP, OPENMP_DISABLED
from .compiler import LowerNoSROA
from .omp_ir import (
    openmp_region_start,
    openmp_region_end,
    _lower_openmp_region_start,
    _lower_openmp_region_end,
)
from .analysis import in_openmp_region
from .omp_lower import VarCollector, remove_ssa_from_func_ir, _add_openmp_ir_nodes


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
                    except Exception as e:
                        print(f"generic transform exception: {e}")
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        sys.exit(-2)


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
        if _OpenmpContextType.first_time:
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
