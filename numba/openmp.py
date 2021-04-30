from numba.core.withcontexts import WithContext
from lark import Lark, Transformer
from lark.exceptions import VisitError
from numba.parfors.parfor_lowering import openmp_tag, openmp_region_start, openmp_region_end
from numba.core.ir_utils import get_call_table, mk_unique_var, dump_block, dump_blocks, dprint_func_ir
from numba.core.analysis import compute_cfg_from_blocks, compute_use_defs, compute_live_map, find_top_level_loops
from numba.core import ir, config
from numba.core.ssa import _run_ssa
from cffi import FFI
import llvmlite.binding as ll
import operator
import sys
import copy

iomplib = '/opt/intel/compilers_and_libraries_2018.0.128/linux/compiler/lib/intel64_lin/libiomp5.so'
ll.load_library_permanently(iomplib)

irclib = '/opt/intel/compilers_and_libraries_2018.0.128/linux/compiler/lib/intel64_lin/libirc.so'
ll.load_library_permanently(irclib)

class PythonOpenmp:
    def __init__(self, *args):
        self.args = args

    def __enter__(self):
        pass

    def __exit__(self, typ, val, tb):
        pass

class _OpenmpContextType(WithContext):
    is_callable = True

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory, extra, state=None, flags=None):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("openmp:mutate_with_body")
            dprint_func_ir(func_ir, "func_ir")
            print("blocks:", blocks, type(blocks))
            print("blk_start:", blk_start, type(blk_start))
            print("blk_end:", blk_end, type(blk_end))
            print("body_blocks:", body_blocks, type(body_blocks))
            print("extra:", extra, type(extra))
            print("flags:", flags, type(flags))
        assert extra is not None
        assert flags is not None
        flags.enable_ssa = False
        flags.release_gil = True
        _add_openmp_ir_nodes(func_ir, blocks, blk_start, blk_end, body_blocks, extra, state)
        if config.DEBUG_ARRAY_OPT >= 1:
            print("post-with-removal")
            dump_blocks(blocks)
        dispatcher = dispatcher_factory(func_ir)
        dispatcher.can_cache = True
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

openmp_context = _OpenmpContextType()

def is_internal_var(var):
    # Determine if a var is a Python var or an internal Numba var.
    if var.is_temp:
        return True
    return var.unversioned_name != var.name

def remove_ssa(var_name, scope, loc):
    # Get the base name of a variable, removing the SSA extension.
    var = ir.Var(scope, var_name, loc)
    return var.unversioned_name

def has_user_defined_var(the_set):
    for x in the_set:
        if not x.startswith("$"):
            return True
    return False

def get_user_defined_var(the_set):
    ret = set()
    for x in the_set:
        if not x.startswith("$"):
            ret.add(x)
    return ret

class OpenmpVisitor(Transformer):
    def __init__(self, func_ir, blocks, blk_start, blk_end, body_blocks, loc, state):
        self.func_ir = func_ir
        self.blocks = blocks
        self.blk_start = blk_start
        self.blk_end = blk_end
        self.body_blocks = body_blocks
        self.loc = loc
        self.state = state
        super(OpenmpVisitor, self).__init__()

    # --------- Non-parser functions --------------------

    def remove_explicit_from_one(self, varset, vars_in_explicit_clauses, clauses, scope, loc):
        """Go through a set of variables and see if their non-SSA form is in an explicitly
        provided data clause.  If so, remove it from the set and add a clause so that the
        SSA form gets the same data clause.
        """
        if config.DEBUG_ARRAY_OPT >= 2:
            print("remove_explicit:", varset, vars_in_explicit_clauses)
        diff = set()
        # For each variable inthe set.
        for v in varset:
            # Get the non-SSA form.
            flat = remove_ssa(v, scope, loc)
            # Skip non-SSA introduced variables (i.e., Python vars).
            if flat == v:
                continue
            if config.DEBUG_ARRAY_OPT >= 2:
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
        if config.DEBUG_ARRAY_OPT >= 2:
            print("remove_explicit:", varset)

    def remove_explicit_from_io_vars(self, inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, loc):
        """Remove vars in explicit data clauses from the auto-determined vars.
        Then call remove_explicit_from_one to take SSA variants out of the auto-determined sets
        and to create clauses so that SSA versions get the same clause as the explicit Python non-SSA var.
        """
        inputs_to_region.difference_update(vars_in_explicit_clauses.keys())
        def_but_live_out.difference_update(vars_in_explicit_clauses.keys())
        private_to_region.difference_update(vars_in_explicit_clauses.keys())
        self.remove_explicit_from_one(inputs_to_region, vars_in_explicit_clauses, clauses, scope, loc)
        self.remove_explicit_from_one(def_but_live_out, vars_in_explicit_clauses, clauses, scope, loc)
        self.remove_explicit_from_one(private_to_region, vars_in_explicit_clauses, clauses, scope, loc)

    def find_io_vars(self, selected_blocks):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        cfg = compute_cfg_from_blocks(self.blocks)
        usedefs = compute_use_defs(self.blocks)
        live_map = compute_live_map(cfg, self.blocks, usedefs.usemap, usedefs.defmap)
        # Assumes enter_with is first statement in block.
        inputs_to_region = live_map[self.blk_start]
        if config.DEBUG_ARRAY_OPT >= 1:
            print("usedefs:", usedefs)
            print("live_map:", live_map)
            print("inputs_to_region:", inputs_to_region, type(inputs_to_region))
        all_uses = set()
        all_defs = set()
        for label in selected_blocks:
            all_uses = all_uses.union(usedefs.usemap[label])
            all_defs = all_defs.union(usedefs.defmap[label])
        # Filter out those vars live to the region but not used within it.
        inputs_to_region = inputs_to_region.intersection(all_uses)
        def_but_live_out = all_defs.difference(inputs_to_region).intersection(live_map[self.blk_end])
        private_to_region = all_defs.difference(inputs_to_region).difference(live_map[self.blk_end])

        if config.DEBUG_ARRAY_OPT >= 1:
            print("inputs_to_region:", inputs_to_region)
            print("private_to_region:", private_to_region)
            print("def_but_live_out:", def_but_live_out)
        return inputs_to_region, def_but_live_out, private_to_region

    def get_explicit_vars(self, clauses):
        ret = {}
        for c in clauses:
            if config.DEBUG_ARRAY_OPT >= 1:
                print("get_explicit_vars:", c, type(c))
            if isinstance(c, openmp_tag):
                if config.DEBUG_ARRAY_OPT >= 1:
                    print("arg:", c.arg, type(c.arg))
                if isinstance(c.arg, str):
                    ret[c.arg] = c
        return ret

    def get_clause_privates(self, clauses, def_but_live_out, scope, loc):
        # Get all the private clauses from the whole set of clauses.
        private_clauses_vars = [x.arg for x in clauses if x.name == "QUAL.OMP.PRIVATE" or x.name == "QUAL.OMP.FIRSTPRIVATE"]
        ret = {}
        # Get a mapping of vars in private clauses to the SSA version of variable exiting the region.
        for lo in def_but_live_out:
            without_ssa = remove_ssa(lo, scope, loc)
            if without_ssa in private_clauses_vars:
                ret[without_ssa] = lo
        return ret

    def add_variables_to_start(self, scope, vars_in_explicit, explicit_clauses, gen_shared, start_tags, keep_alive, inputs_to_region, def_but_live_out, private_to_region):
        start_tags.extend(explicit_clauses)
        for var in vars_in_explicit:
            evar = ir.Var(scope, var, self.loc)
            keep_alive.append(ir.Assign(evar, evar, self.loc))

        if gen_shared:
            for itr in inputs_to_region:
                itr_var = ir.Var(scope, itr, self.loc)
                start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
                keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
            for itr in def_but_live_out:
                itr_var = ir.Var(scope, itr, self.loc)
                start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
                keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
            for ptr in private_to_region:
                itr_var = ir.Var(scope, ptr, self.loc)
                if not is_internal_var(itr_var):
                    start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
                    keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
        for ptr in private_to_region:
            itr_var = ir.Var(scope, ptr, self.loc)
            if is_internal_var(itr_var):
                start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", itr_var))

    def some_for_directive(self, args, main_start_tag, main_end_tag, first_clause, gen_shared):
        sblk = self.blocks[self.blk_start]
        scope = sblk.scope
        eblk = self.blocks[self.blk_end]

        clauses = []
        default_shared = True
        if config.DEBUG_ARRAY_OPT >= 1:
            print("some_for_directive")
        incoming_clauses = [remove_indirections(x) for x in args[first_clause:]]
        # Process all the incoming clauses which can be in singular or list form
        # and flatten them to a list of openmp_tags.
        for clause in incoming_clauses:
            if config.DEBUG_ARRAY_OPT >= 1:
                print("clause:", clause, type(clause))
            if isinstance(clause, openmp_tag):
                clauses.append(clause)
            elif isinstance(clause, list):
                clauses.extend(remove_indirections(clause))
            elif isinstance(clause, default_shared_val):
                default_shared = clause.val
                if config.DEBUG_ARRAY_OPT >= 1:
                    print("got new default_shared:", clause.val)
            else:
                assert(0)
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit", main_start_tag, args, type(args), clauses, default_shared)

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses = self.get_explicit_vars(clauses)
        if config.DEBUG_ARRAY_OPT >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses))

        before_start = []
        after_start = []
        start_tags = [ openmp_tag(main_start_tag) ]
        end_tags   = [ openmp_tag(main_end_tag) ]

        call_table, _ = get_call_table(self.blocks)
        cfg = compute_cfg_from_blocks(self.blocks)
        usedefs = compute_use_defs(self.blocks)
        live_map = compute_live_map(cfg, self.blocks, usedefs.usemap, usedefs.defmap)
        all_loops = cfg.loops()
        if config.DEBUG_ARRAY_OPT >= 1:
            print("all_loops:", all_loops)
        loops = {}
        # Find the outer-most loop in this OpenMP region.
        for k, v in all_loops.items():
            if v.header >= self.blk_start and v.header <= self.blk_end:
                loops[k] = v
        loops = list(find_top_level_loops(cfg, loops=loops))

        if config.DEBUG_ARRAY_OPT >= 1:
            print("loops:", loops)
        assert(len(loops) == 1)

        def _get_loop_kind(func_var, call_table):
            if func_var not in call_table:
                return False
            call = call_table[func_var]
            if len(call) == 0:
                return False

            return call[0] # or call[0] == prange
                    #or call[0] == 'internal_prange' or call[0] == internal_prange
                    #$or call[0] == 'pndindex' or call[0] == pndindex)

        loop = loops[0]
        entry = list(loop.entries)[0]
        header = loop.header
        exit = list(loop.exits)[0]

        loop_blocks_for_io = loop.entries.union(loop.body)
        if config.DEBUG_ARRAY_OPT >= 1:
            print("loop_blocks_for_io:", loop_blocks_for_io, entry, exit)

        found_loop = False
        entry_block = self.blocks[entry]
        exit_block = self.blocks[exit]
        header_block = self.blocks[header]

        latch_block_num = max(self.blocks.keys()) + 1

        # We have to reformat the Numba style of loop to the only form that xmain openmp supports.
        header_preds = [x[0] for x in cfg.predecessors(header)]
        entry_preds = list(set(header_preds).difference(loop.body))
        back_blocks = list(set(header_preds).intersection(loop.body))
        if config.DEBUG_ARRAY_OPT >= 1:
            print("header_preds:", header_preds)
            print("entry_preds:", entry_preds)
            print("back_blocks:", back_blocks)
        assert(len(entry_preds) == 1)
        entry_pred_label = entry_preds[0]
        entry_pred = self.blocks[entry_pred_label]
        header_branch = header_block.body[-1]
        post_header = {header_branch.truebr, header_branch.falsebr}
        post_header.remove(exit)
        if config.DEBUG_ARRAY_OPT >= 1:
            print("post_header:", post_header)
        post_header = self.blocks[list(post_header)[0]]
        if config.DEBUG_ARRAY_OPT >= 1:
            print("post_header:", post_header)

        normalized = True

        for inst_num, inst in enumerate(entry_block.body):
            if (isinstance(inst, ir.Assign)
                    and isinstance(inst.value, ir.Expr)
                    and inst.value.op == 'call'):
                loop_kind = _get_loop_kind(inst.value.func.name, call_table) 
                if config.DEBUG_ARRAY_OPT >= 1:
                    print("loop_kind:", loop_kind)
                if loop_kind != False and loop_kind == range:
                    found_loop = True
                    range_inst = inst
                    range_args = inst.value.args
                    if config.DEBUG_ARRAY_OPT >= 1:
                        print("found one", loop_kind, inst, range_args)
                    #----------------------------------------------
                    # Find getiter instruction for this range.
                    for entry_inst in entry_block.body[inst_num+1:]:
                        if (isinstance(entry_inst, ir.Assign) and
                            isinstance(entry_inst.value, ir.Expr) and
                            entry_inst.value.op == 'getiter' and
                            entry_inst.value.value == range_inst.target):
                            getiter_inst = entry_inst
                            break
                    assert(getiter_inst)
                    if config.DEBUG_ARRAY_OPT >= 1:
                        print("getiter_inst:", getiter_inst)
                    #----------------------------------------------
                    assert(len(header_block.body) > 3)
                    if config.DEBUG_ARRAY_OPT >= 1:
                        print("header block before removing Numba range vars:")
                        dump_block(header, header_block)

                    for ii in range(len(header_block.body)):
                        ii_inst = header_block.body[ii]
                        if (isinstance(ii_inst, ir.Assign) and
                            isinstance(ii_inst.value, ir.Expr) and
                            ii_inst.value.op == 'iternext'):
                            iter_num = ii
                            break

                    iternext_inst = header_block.body[iter_num]
                    pair_first_inst = header_block.body[iter_num + 1]
                    pair_second_inst = header_block.body[iter_num + 2]

                    assert(isinstance(iternext_inst, ir.Assign) and isinstance(iternext_inst.value, ir.Expr) and iternext_inst.value.op == 'iternext')
                    assert(isinstance(pair_first_inst, ir.Assign) and isinstance(pair_first_inst.value, ir.Expr) and pair_first_inst.value.op == 'pair_first')
                    assert(isinstance(pair_second_inst, ir.Assign) and isinstance(pair_second_inst.value, ir.Expr) and pair_second_inst.value.op == 'pair_second')
                    # Remove those nodes from the IR.
                    header_block.body = header_block.body[:iter_num] + header_block.body[iter_num+3:]
                    if config.DEBUG_ARRAY_OPT >= 1:
                        print("header block after removing Numba range vars:")
                        dump_block(header, header_block)

                    loop_index = pair_first_inst.target
                    if config.DEBUG_ARRAY_OPT >= 1:
                        print("loop_index:", loop_index)
                    # The loop_index from Numba's perspective is not what it is from the
                    # programmer's perspective.  The OpenMP loop index is always private so
                    # we need to start from Numba's loop index (e.g., $48for_iter.3) and
                    # trace assignments from that through the header block and then find
                    # the first such assignment in the first loop block that the header
                    # branches to.
                    latest_index = loop_index
                    for hinst in header_block.body:
                        if isinstance(hinst, ir.Assign) and isinstance(hinst.value, ir.Var):
                            if hinst.value.name == latest_index.name:
                                latest_index = hinst.target
                    for phinst in post_header.body:
                        if isinstance(phinst, ir.Assign) and isinstance(phinst.value, ir.Var):
                            if phinst.value.name == latest_index.name:
                                latest_index = phinst.target.name
                                break
                    if config.DEBUG_ARRAY_OPT >= 1:
                        print("latest_index:", latest_index)
                    new_index_clause = openmp_tag("QUAL.OMP.PRIVATE", ir.Var(loop_index.scope, latest_index, inst.loc))
                    clauses.append(new_index_clause)
                    vars_in_explicit_clauses[latest_index] = new_index_clause

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
                        except KeyError:
                            raise NotImplementedError(
                                "Only known step size is supported for prange")
                        if not isinstance(step, ir.Const):
                            raise NotImplementedError(
                                "Only constant step size is supported for prange")
                        step = step.value
#                        if step != 1:
#                            print("unsupported step:", step, type(step))
#                            raise NotImplementedError(
#                                "Only constant step size of 1 is supported for prange")

                    #assert(start == 0 or (isinstance(start, ir.Const) and start.value == 0))

                    omp_lb_var = ir.Var(loop_index.scope, mk_unique_var("$omp_lb"), inst.loc)
                    before_start.append(ir.Assign(ir.Const(0, inst.loc), omp_lb_var, inst.loc))

                    omp_iv_var = ir.Var(loop_index.scope, mk_unique_var("$omp_iv"), inst.loc)
                    #before_start.append(ir.Assign(omp_lb_var, omp_iv_var, inst.loc))
                    after_start.append(ir.Assign(omp_lb_var, omp_iv_var, inst.loc))

                    omp_ub_var = ir.Var(loop_index.scope, mk_unique_var("$omp_ub"), inst.loc)
                    omp_ub_expr = ir.Expr.itercount(getiter_inst.target, inst.loc)
                    before_start.append(ir.Assign(omp_ub_expr, omp_ub_var, inst.loc))

                    const1_var = ir.Var(loop_index.scope, mk_unique_var("$const1"), inst.loc)
                    const1_assign = ir.Assign(ir.Const(1, inst.loc), const1_var, inst.loc)
                    before_start.append(const1_assign)
                    count_add_1 = ir.Expr.binop(operator.sub, omp_ub_var, const1_var, inst.loc)
                    before_start.append(ir.Assign(count_add_1, omp_ub_var, inst.loc))

#                    before_start.append(ir.Print([omp_ub_var], None, inst.loc))

                    omp_start_var = ir.Var(loop_index.scope, mk_unique_var("$omp_start"), inst.loc)
                    if start == 0:
                        start = ir.Const(start, inst.loc)
                    before_start.append(ir.Assign(start, omp_start_var, inst.loc))

                    gen_ssa = False

                    # ---------- Create latch block -------------------------------
                    if gen_ssa:
                        latch_iv = ir.Var(loop_index.scope, mk_unique_var("$omp_iv"), inst.loc)
                    else:
                        latch_iv = omp_iv_var

                    latch_block = ir.Block(scope, inst.loc)
                    const1_var = ir.Var(loop_index.scope, mk_unique_var("$const1"), inst.loc)
                    const1_assign = ir.Assign(ir.Const(1, inst.loc), const1_var, inst.loc)
                    latch_block.body.append(const1_assign)
                    latch_assign = ir.Assign(
                        ir.Expr.binop(
                            operator.add,
                            omp_iv_var,
                            const1_var,
                            inst.loc
                        ),
                        latch_iv,
                        inst.loc
                    )
                    latch_block.body.append(latch_assign)
                    latch_block.body.append(ir.Jump(header, inst.loc))

                    self.blocks[latch_block_num] = latch_block
                    for bb in back_blocks:
                        self.blocks[bb].body[-1] = ir.Jump(latch_block_num, inst.loc)
                    # -------------------------------------------------------------

                    # ---------- Header Manipulation ------------------------------
                    ssa_code = []
                    if gen_ssa:
                        ssa_var = ir.Var(loop_index.scope, mk_unique_var("$omp_iv"), inst.loc)
                        ssa_phi = ir.Expr.phi(inst.loc)
                        ssa_phi.incoming_values = [latch_iv, omp_iv_var]
                        ssa_phi.incoming_blocks = [latch_block_num, entry_pred_label]
                        ssa_inst = ir.Assign(ssa_phi, ssa_var, inst.loc)
                        ssa_code.append(ssa_inst)

                    step_var = ir.Var(loop_index.scope, mk_unique_var("$step_var"), inst.loc)
                    step_assign = ir.Assign(ir.Const(step, inst.loc), step_var, inst.loc)
                    scale_var = ir.Var(loop_index.scope, mk_unique_var("$scale"), inst.loc)
                    fake_iternext = ir.Assign(ir.Const(0, inst.loc), iternext_inst.target, inst.loc)
                    fake_second = ir.Assign(ir.Const(0, inst.loc), pair_second_inst.target, inst.loc)
                    scale_assign = ir.Assign(ir.Expr.binop(operator.mul, step_var, omp_iv_var, inst.loc), scale_var, inst.loc)
                    unnormalize_iv = ir.Assign(ir.Expr.binop(operator.add, omp_start_var, scale_var, inst.loc), loop_index, inst.loc)
                    cmp_var = ir.Var(loop_index.scope, mk_unique_var("$cmp"), inst.loc)
                    if gen_ssa:
                        iv_lte_ub = ir.Assign(ir.Expr.binop(operator.le, ssa_var, omp_ub_var, inst.loc), cmp_var, inst.loc)
                    else:
                        iv_lte_ub = ir.Assign(ir.Expr.binop(operator.le, omp_iv_var, omp_ub_var, inst.loc), cmp_var, inst.loc)
                    old_branch = header_block.body[-1]
                    new_branch = ir.Branch(cmp_var, old_branch.truebr, old_branch.falsebr, old_branch.loc)
                    body_label = old_branch.truebr
                    first_body_block = self.blocks[body_label]
                    new_end = [iv_lte_ub, new_branch]
                    if False:
                        str_var = ir.Var(loop_index.scope, mk_unique_var("$str_var"), inst.loc)
                        str_const = ir.Const("header1:", inst.loc)
                        str_assign = ir.Assign(str_const, str_var, inst.loc)
                        new_end.append(str_assign)
                        str_print = ir.Print([str_var, omp_start_var, omp_iv_var], None, inst.loc)
                        new_end.append(str_print)

                    # Prepend original contents of header into the first body block minus the comparison
                    first_body_block.body = [fake_iternext, fake_second, step_assign, scale_assign, unnormalize_iv] + header_block.body[:-1] + first_body_block.body

                    header_block.body = new_end
                    #header_block.body = ssa_code + [fake_iternext, fake_second, unnormalize_iv] + header_block.body[:-1] + new_end

                    # -------------------------------------------------------------

                    #const_start_var = ir.Var(loop_index.scope, mk_unique_var("$const_start"), inst.loc)
                    #before_start.append(ir.Assign(ir.Const(0, inst.loc), const_start_var, inst.loc))
                    #start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", const_start_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.IV", omp_iv_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.UB", omp_ub_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", omp_lb_var.name))
                    #start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.IV", loop_index.name))
                    #start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.UB", size_var.name))

                    keep_alive = []
                    # Do an analysis to get variable use information coming into and out of the region.
                    inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(loop_blocks_for_io)

                    # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
                    clause_privates = self.get_clause_privates(clauses, def_but_live_out, scope, self.loc)
                    priv_saves = []
                    priv_restores = []
                    # Numba typing is not aware of OpenMP semantics, so for private variables we save the value
                    # before entering the region and then restore it afterwards but we have to restore it to the SSA
                    # version of the variable at that point.
                    for cp in clause_privates:
                        cpvar = ir.Var(scope, cp, self.loc)
                        cplovar = ir.Var(scope, clause_privates[cp], self.loc)
                        save_var = ir.Var(scope, mk_unique_var("$"+cp), self.loc)
                        priv_saves.append(ir.Assign(cpvar, save_var, self.loc))
                        priv_restores.append(ir.Assign(save_var, cplovar, self.loc))

                    # Remove variables the user explicitly added to a clause from the auto-determined variables.
                    # This will also treat SSA forms of vars the same as their explicit Python var clauses.
                    self.remove_explicit_from_io_vars(inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, self.loc)

                    if not default_shared and (
                        has_user_defined_var(inputs_to_region) or
                        has_user_defined_var(def_but_live_out) or
                        has_user_defined_var(private_to_region)):
                        user_defined_inputs = get_user_defined_var(inputs_to_region)
                        user_defined_def_live = get_user_defined_var(def_but_live_out)
                        user_defined_private = get_user_defined_var(private_to_region)
                        if config.DEBUG_ARRAY_OPT >= 1:
                            print("inputs users:", user_defined_inputs)
                            print("def users:", user_defined_def_live)
                            print("private users:", user_defined_private)
                        raise UnspecifiedVarInDefaultNone("Variables with no data env clause in OpenMP region: " + str(user_defined_inputs.union(user_defined_def_live).union(user_defined_private)))

                    self.add_variables_to_start(scope, vars_in_explicit_clauses, clauses, gen_shared, start_tags, keep_alive, inputs_to_region, def_but_live_out, private_to_region)

                    or_start = openmp_region_start(start_tags, 0, self.loc)
                    or_end   = openmp_region_end(or_start, end_tags, self.loc)

                    #new_header_block.body = [or_start] + before_start + new_header_block.body[:]
                    entry_pred.body = entry_pred.body[:-1] + priv_saves + before_start + [or_start] + after_start + [entry_pred.body[-1]]
                    #entry_block.body = [or_start] + before_start + entry_block.body[:]
                    #entry_block.body = entry_block.body[:inst_num] + before_start + [or_start] + entry_block.body[inst_num:]
                    exit_block.body = [or_end] + priv_restores + keep_alive + exit_block.body
                    #exit_block.body = [or_end] + exit_block.body

                    break

        assert(found_loop)

        return None

    # --------- Parser functions ------------------------

    def barrier_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit barrier_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.BARRIER")], 0, self.loc)
        or_start.requires_combined_acquire_release()
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.BARRIER")], self.loc)
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def taskwait_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit taskwait_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.TASKWAIT")], 0, self.loc)
        or_start.requires_combined_acquire_release()
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.TASKWAIT")], self.loc)
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def target_directive(self, args):
        pass

    def critical_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit critical_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.CRITICAL")], 0, self.loc)
        or_start.requires_acquire_release()
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.CRITICAL")], self.loc)

        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)
        inputs_to_region = {remove_ssa(x, scope, self.loc):x for x in inputs_to_region}
        def_but_live_out = {remove_ssa(x, scope, self.loc):x for x in def_but_live_out}
        common_keys = inputs_to_region.keys() & def_but_live_out.keys()
        in_def_live_out = {inputs_to_region[k]:def_but_live_out[k] for k in common_keys}
        if config.DEBUG_ARRAY_OPT >= 1:
            print("inputs_to_region:", inputs_to_region)
            print("def_but_live_out:", def_but_live_out)
            print("in_def_live_out:", in_def_live_out)

        reset = []
        for k,v in in_def_live_out.items():
            reset.append(ir.Assign(ir.Var(scope, v, self.loc), ir.Var(scope, k, self.loc), self.loc))

        sblk.body = [or_start] + sblk.body[:]
        eblk.body = reset + [or_end] + eblk.body[:]

    def single_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit single_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.SINGLE")], 0, self.loc)
        or_start.requires_acquire_release()
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.SINGLE")], self.loc)
        sblk.body = [or_start] + sblk.body[:]
        eblk.body = [or_end]   + eblk.body[:]

    def NOWAIT(self, args):
        return "nowait"

    # Don't need a rule for TASK.

    def task_directive(self, args):
        clauses = args[1:]
        tag_clauses = []
        for clause in clauses:
            tag_clauses.extend(clause)

        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        start_tags = [openmp_tag("DIR.OMP.TASK")] + tag_clauses
        end_tags   = [openmp_tag("DIR.OMP.END.TASK")]
        keep_alive = []

        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)
        for itr in inputs_to_region:
            if config.DEBUG_ARRAY_OPT >= 1:
                print("input_to_region:", itr)
            start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", ir.Var(scope, itr, self.loc)))
        for itr in def_but_live_out:
            if config.DEBUG_ARRAY_OPT >= 1:
                print("def_but_live_out:", itr)
            itr_var = ir.Var(scope, itr, self.loc)
            start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
            keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
        for ptr in private_to_region:
            if config.DEBUG_ARRAY_OPT >= 1:
                print("private_to_region:", ptr)
            start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", ir.Var(scope, ptr, self.loc)))

        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit task_directive", args, type(args), tag_clauses)
        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = [or_start] + sblk.body[:]
        #eblk.body = [or_end]   + eblk.body[:]
        eblk.body = [or_end]   + keep_alive + eblk.body[:]

    def task_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit task_clause", args, type(args), args[0])
        return args[0]

    def data_default_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit data_default_clause", args, type(args), args[0])
        return args[0]

    def data_sharing_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit data_sharing_clause", args, type(args), args[0])
        return args[0]

    def data_privatization_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit data_privatization_clause", args, type(args), args[0])
        return args[0]

    def data_privatization_in_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit data_privatization_in_clause", args, type(args), args[0])
        return args[0]

    def data_privatization_out_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit data_privatization_out_clause", args, type(args), args[0])
        return args[0]

    def data_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit data_clause", args, type(args), args[0])
        return args[0]

    def private_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit private_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.PRIVATE", var))
        return ret

    # Don't need a rule for PRIVATE.

    def copyprivate_clause(self, args):
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.COPYPRIVATE", var))
        return ret

    # Don't need a rule for COPYPRIVATE.

    def firstprivate_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit firstprivate_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", var))
        return ret

    # Don't need a rule for FIRSTPRIVATE.

    def lastprivate_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit lastprivate_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.LASTPRIVATE", var))
        return ret

    # Don't need a rule for LASTPRIVATE.

    def shared_clause(self, args):
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.SHARED", var))
        return ret

    # Don't need a rule for SHARED.

    def copyin_clause(self, args):
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.COPYIN", var))
        return ret

    # Don't need a rule for COPYIN.

    # Don't need a rule for REDUCTION.

    def data_reduction_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit data_reduction_clause", args, type(args), args[0])

        (_, red_op, red_list) = args
        ret = []
        for shared in red_list:
            ret.append(openmp_tag("QUAL.OMP.REDUCTION." + red_op, shared))
        return ret

    def default_shared_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit default_shared_clause", args, type(args))
        return default_shared_val(True)

    def default_none_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit default_none", args, type(args))
        return default_shared_val(False)

    def const_num_or_var(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit const_num_or_var", args, type(args))
        return args[0]

    # Don't need a rule for PARALLEL_CONSTRUCT.

    def parallel_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        clauses = []
        default_shared = True
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit parallel_directive", args, type(args))
        incoming_clauses = [remove_indirections(x) for x in args[1:]]
        # Process all the incoming clauses which can be in singular or list form
        # and flatten them to a list of openmp_tags.
        for clause in incoming_clauses:
            if config.DEBUG_ARRAY_OPT >= 1:
                print("clause:", clause, type(clause))
            if isinstance(clause, openmp_tag):
                clauses.append(clause)
            elif isinstance(clause, list):
                clauses.extend(remove_indirections(clause))
            elif isinstance(clause, default_shared_val):
                default_shared = clause.val
                if config.DEBUG_ARRAY_OPT >= 1:
                    print("got new default_shared:", clause.val)
            else:
                if config.DEBUG_ARRAY_OPT >= 1:
                    print("Unknown clause type in incoming_clauses", clause, type(clause))
                assert(0)

        if config.DEBUG_ARRAY_OPT >= 1:
            for clause in clauses:
                print("final clause:", clause)

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses = self.get_explicit_vars(clauses)
        if config.DEBUG_ARRAY_OPT >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses))

        # Do an analysis to get variable use information coming into and out of the region.
        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)

        # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
        clause_privates = self.get_clause_privates(clauses, def_but_live_out, scope, self.loc)
        priv_saves = []
        priv_restores = []
        # Numba typing is not aware of OpenMP semantics, so for private variables we save the value
        # before entering the region and then restore it afterwards but we have to restore it to the SSA
        # version of the variable at that point.
        for cp in clause_privates:
            cpvar = ir.Var(scope, cp, self.loc)
            cplovar = ir.Var(scope, clause_privates[cp], self.loc)
            save_var = ir.Var(scope, mk_unique_var("$"+cp), self.loc)
            priv_saves.append(ir.Assign(cpvar, save_var, self.loc))
            priv_restores.append(ir.Assign(save_var, cplovar, self.loc))

        # Remove variables the user explicitly added to a clause from the auto-determined variables.
        # This will also treat SSA forms of vars the same as their explicit Python var clauses.
        self.remove_explicit_from_io_vars(inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, self.loc)

        if not default_shared and (
            has_user_defined_var(inputs_to_region) or
            has_user_defined_var(def_but_live_out) or
            has_user_defined_var(private_to_region)):
            user_defined_inputs = get_user_defined_var(inputs_to_region)
            user_defined_def_live = get_user_defined_var(def_but_live_out)
            user_defined_private = get_user_defined_var(private_to_region)
            if config.DEBUG_ARRAY_OPT >= 1:
                print("inputs users:", user_defined_inputs)
                print("def users:", user_defined_def_live)
                print("private users:", user_defined_private)
            raise UnspecifiedVarInDefaultNone("Variables with no data env clause in OpenMP region: " + str(user_defined_inputs.union(user_defined_def_live).union(user_defined_private)))

        start_tags = [openmp_tag("DIR.OMP.PARALLEL")]
        end_tags = [openmp_tag("DIR.OMP.END.PARALLEL")]
        keep_alive = []
        self.add_variables_to_start(scope, vars_in_explicit_clauses, clauses, True, start_tags, keep_alive, inputs_to_region, def_but_live_out, private_to_region)

        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = priv_saves + [or_start] + sblk.body[:]
        #eblk.body = [or_end]   + eblk.body[:]
        eblk.body = [or_end] + priv_restores + keep_alive + eblk.body[:]

    def parallel_clause(self, args):
        (val,) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit parallel_clause", args, type(args))
        return val

    def unique_parallel_clause(self, args):
        (val,) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit unique_parallel_clause", args, type(args))
        assert(isinstance(val, openmp_tag))
        return val

    def if_clause(self, args):
        (_, if_val) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit if_clause", args, type(args))

        return openmp_tag("QUAL.OMP.IF", if_val, load=True)

    # Don't need a rule for IF.

    def num_threads_clause(self, args):
        (_, num_threads) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit num_threads_clause", args, type(args))

        return openmp_tag("QUAL.OMP.NUM_THREADS", num_threads, load=True)

    # Don't need a rule for NUM_THREADS.
    # Don't need a rule for PARALLEL.
    # Don't need a rule for FOR.
    # Don't need a rule for parallel_for_construct.

    def parallel_for_directive(self, args):
        return self.some_for_directive(args, "DIR.OMP.PARALLEL.LOOP", "DIR.OMP.END.PARALLEL.LOOP", 2, True)

    def parallel_for_clause(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit parallel_for_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for for_construct.

    def for_directive(self, args):
        return self.some_for_directive(args, "DIR.OMP.LOOP", "DIR.OMP.END.LOOP", 1, False)

    def for_clause(self, args):
        (val,) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit for_clause", args, type(args))
        if isinstance(val, openmp_tag):
            return [val]
        elif isinstance(val, list):
            return val
        elif val == 'nowait':
            return openmp_tag("QUAL.OMP.NOWAIT")

    def unique_for_clause(self, args):
        (val,) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit unique_for_clause", args, type(args))
        if isinstance(val, openmp_tag):
            return val
        elif val == 'ordered':
            return openmp_tag("QUAL.OMP.ORDERED", 0)

    def linear_expr(self, args):
        (_, var, step) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit linear_expr", args, type(args))
        return openmp_tag("QUAL.OMP.LINEAR", [var, step])

    def ORDERED(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit ordered", args, type(args))
        return "ordered"

    def sched_no_expr(self, args):
        (_, kind) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit sched_no_expr", args, type(args))
        if kind == 'static':
            return openmp_tag("QUAL.OMP.SCHEDULE.STATIC", 0)
        elif kind == 'dynamic':
            return openmp_tag("QUAL.OMP.SCHEDULE.DYNAMIC", 0)
        elif kind == 'guided':
            return openmp_tag("QUAL.OMP.SCHEDULE.GUIDED", 0)
        elif kind == 'runtime':
            return openmp_tag("QUAL.OMP.SCHEDULE.RUNTIME", 0)

    def sched_expr(self, args):
        (_, kind, num_or_var) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit sched_expr", args, type(args), num_or_var, type(num_or_var))
        if kind == 'static':
            return openmp_tag("QUAL.OMP.SCHEDULE.STATIC", num_or_var, load=True)
        elif kind == 'dynamic':
            return openmp_tag("QUAL.OMP.SCHEDULE.DYNAMIC", num_or_var, load=True)
        elif kind == 'guided':
            return openmp_tag("QUAL.OMP.SCHEDULE.GUIDED", num_or_var, load=True)
        elif kind == 'runtime':
            return openmp_tag("QUAL.OMP.SCHEDULE.RUNTIME", num_or_var, load=True)

    def SCHEDULE(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit SCHEDULE", args, type(args))
        return "schedule"

    def schedule_kind(self, args):
        (kind,) = args
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit schedule_kind", args, type(args))
        return kind

    def STATIC(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit STATIC", args, type(args))
        return "static"

    def DYNAMIC(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit DYNAMIC", args, type(args))
        return "dynamic"

    def GUIDED(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit GUIDED", args, type(args))
        return "guided"

    def RUNTIME(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit RUNTIME", args, type(args))
        return "RUNTIME"

    def var_list(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit var_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

    def PLUS(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit PLUS", args, type(args))
        return "+"

    def reduction_operator(self, args):
        arg = args[0]
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit reduction_operator", args, type(args), arg, type(arg))
        if arg == "+":
            return "ADD"
        assert(0)

    def PYTHON_NAME(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
            print("visit PYTHON_NAME", args, type(args), str(args))
        return str(args)

    def NUMBER(self, args):
        if config.DEBUG_ARRAY_OPT >= 1:
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
    openmp_construct: parallel_construct
                    | parallel_for_construct
                    | for_construct
                    | single_construct
                    | task_construct
                    | target_construct
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
    teams_distribute_parallel_for_simd_clause: target_clause
                                             | teams_distribute_parallel_for_simd_clause
    for_simd_construct: for_simd_directive
    for_simd_directive: FOR SIMD [for_simd_clause*]
    for_simd_clause: for_clause
                   | simd_clause
    parallel_for_simd_construct: parallel_for_simd_directive
    parallel_for_simd_directive: PARALLEL FOR SIMD [parallel_for_simd_clause*]
    parallel_for_simd_clause: parallel_for_clause
                            | simd_clause
    target_data_construct: target_data_directive
    target_data_directive: TARGET DATA [target_data_clause*]
    DATA: "data"
    target_data_clause: device_clause
                      | map_clause
                      | if_clause
    device_clause: "device" "(" const_num_or_var ")"
    map_clause: "map" "(" [map_type] variable_array_section_list ")"
    map_type: "alloc:" | "to:" | "from:" | "tofrom:"
    parallel_sections_construct: parallel_sections_directive
    parallel_sections_directive: PARALLEL SECTIONS [parallel_sections_clause*]
    parallel_sections_clause: unique_parallel_clause
                            | data_default_clause
                            | data_privatization_clause
                            | data_privatization_in_clause
                            | data_privatization_out_clause
                            | data_sharing_clause
                            | data_reduction_clause
    sections_construct: sections_directive
    sections_directive: SECTIONS [sections_clause*]
    SECTIONS: "sections"
    sections_clause: data_privatization_clause
                   | data_privatization_in_clause
                   | data_privatization_out_clause
                   | data_reduction_clause
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
    target_construct: target_directive
    target_directive: TARGET [target_clause*]
    target_clause: device_clause
                 | map_clause
                 | if_clause
    target_update_construct: target_update_directive
    target_update_directive: TARGET UPDATE target_update_clause*
    target_update_clause: motion_clause
                        | device_clause
                        | if_clause
    motion_clause: "to" "(" variable_array_section_list ")"
                 | "from" "(" variable_array_section_list ")"
    variable_array_section_list: PYTHON_NAME
                               | array_section
                               | variable_array_section_list "," PYTHON_NAME
                               | variable_array_section_list "," array_section
    array_section: PYTHON_NAME array_section_subscript
    array_section_subscript: array_section_subscript "[" [const_num_or_var] ":" [const_num_or_var] "]"
                           | array_section_subscript "[" const_num_or_var "]"
                           | "[" [const_num_or_var] ":" [const_num_or_var] "]"
                           | "[" const_num_or_var "]"
    TARGET: "target"
    single_construct: single_directive
    single_directive: SINGLE [single_clause*]
    SINGLE: "single"
    single_clause: unique_single_clause
                 | data_privatization_clause
                 | data_privatization_in_clause
                 | NOWAIT
    unique_single_clause: copyprivate_clause
    NOWAIT: "nowait"
    simd_construct: simd_directive
    simd_directive: SIMD [simd_clause*]
    SIMD: "simd"
    simd_clause: collapse_expr
               | aligned_clause
               | linear_clause
               | uniform_clause
               | data_reduction_clause
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
                       | data_reduction_clause
                       | inbranch_clause
    ALIGNED: "aligned"
    inbranch_clause: INBRANCH | NOTINBRANCH
    INBRANCH: "inbranch"
    NOTINBRANCH: "notinbranch"
    uniform_clause: UNIFORM "(" var_list ")"
    UNIFORM: "uniform"
    ligned_clause: ALIGNED "(" var_list ")"
                  | ALIGNED "(" var_list ":" const_num_or_var ")" 
    collapse_expr: COLLAPSE "(" const_num_or_var ")"
    COLLAPSE: "collapse"
    task_construct: task_directive
    TASK: "task"
    task_directive: TASK [task_clause*]
    task_clause: unique_task_clause
               | data_sharing_clause
               | data_privatization_clause
               | data_privatization_in_clause
               | data_default_clause
    unique_task_clause: if_clause
                      | UNTIED
                      | MERGEABLE
                      | FINAL "(" const_num_or_var ")"
                      | DEPEND "(" dependence_type ":" variable_array_section_list ")"
    DEPEND: "depend"
    FINAL: "final"
    UNTIED: "untied"
    MERGEABLE: "MERGEABLE"
    data_default_clause: default_shared_clause
                       | default_none_clause
    data_sharing_clause: shared_clause
    data_privatization_clause: private_clause
    data_privatization_in_clause: firstprivate_clause
    data_privatization_out_clause: lastprivate_clause
    data_clause: data_privatization_clause
               | copyprivate_clause
               | data_privatization_in_clause
               | data_privatization_out_clause
               | data_sharing_clause
               | data_default_clause
               | copyin_clause
               | data_reduction_clause
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
    data_reduction_clause: REDUCTION "(" reduction_operator ":" var_list ")"
    default_shared_clause: "default" "(" "shared" ")"
    default_none_clause: "default" "(" "none" ")"
    const_num_or_var: NUMBER | PYTHON_NAME
    parallel_construct: parallel_directive
    parallel_directive: PARALLEL [parallel_clause*]
    parallel_clause: unique_parallel_clause
                   | data_default_clause
                   | data_privatization_clause
                   | data_privatization_in_clause
                   | data_sharing_clause
                   | data_reduction_clause
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
                       | data_privatization_clause
                       | data_privatization_in_clause
                       | data_privatization_out_clause
                       | data_sharing_clause
                       | data_reduction_clause
    for_construct: for_directive
    for_directive: FOR [for_clause*]
    for_clause: unique_for_clause | data_clause | NOWAIT
    unique_for_clause: ORDERED
                     | sched_no_expr
                     | sched_expr
                     | collapse_expr
    LINEAR: "linear"
    linear_clause: LINEAR "(" var_list ":" const_num_or_var ")"
                 | LINEAR "(" var_list ")"
    ORDERED: "ordered"
    sched_no_expr: SCHEDULE "(" schedule_kind ")"
    sched_expr: SCHEDULE "(" schedule_kind "," const_num_or_var ")"
    SCHEDULE: "schedule"
    schedule_kind: STATIC | DYNAMIC | GUIDED | RUNTIME | AUTO
    STATIC: "static"
    DYNAMIC: "dynamic"
    GUIDED: "guided"
    RUNTIME: "runtime"
    AUTO: "auto"
    var_list: PYTHON_NAME | var_list "," PYTHON_NAME
    PLUS: "+"
    reduction_operator: PLUS | "\\" | "*" | "-" | "&" | "^" | "|" | "&&" | "||"
    threadprivate_directive: "threadprivate" "(" var_list ")"
    cancellation_point_directive: "cancellation point" construct_type_clause
    construct_type_clause: PARALLEL
                         | SECTIONS
                         | FOR
                         | TASKGROUP
    TASKGROUP: "taskgroup"
    cancel_directive: "cancel" construct_type_clause [if_clause]
    ordered_directive: ORDERED
    ordered_construct: ordered_directive
    flush_vars: "(" var_list ")"
    flush_directive: "flush" [flush_vars]
    region_phrase: "(" PYTHON_NAME ")"
    master_construct: master_directive
    master_directive: "master"
    dependence_type: IN
                   | OUT
                   | INOUT
    IN: "in"
    OUT: "out"
    INOUT: "inout"

    PYTHON_NAME: /[a-zA-Z_]\w*/

    %import common.NUMBER
    %import common.WS
    %ignore WS
    """

"""
    openmp_construct: parallel_construct
                    | target_teams_distribute_construct
                    | teams_distribute_parallel_for_simd_construct
                    | target_teams_distribute_parallel_for_construct
                    | target_teams_distribute_parallel_for_simd_construct
                    | target_teams_construct
                    | target_teams_distribute_simd_construct
                    | teams_distribute_parallel_for_construct
                    | teams_construct
                    | distribute_parallel_for_construct
                    | distribute_parallel_for_simd_construct
                    | distribute_construct
                    | distribute_simd_construct
                    | teams_distribute_construct
                    | teams_distribute_simd_construct
"""

openmp_parser = Lark(openmp_grammar, start='openmp_statement')

def _add_openmp_ir_nodes(func_ir, blocks, blk_start, blk_end, body_blocks, extra, state):
    """Given the starting and ending block of the with-context,
    replaces the head block with a new block that has the starting
    openmp ir nodes in it and adds the ending openmp ir nodes to
    the end block.
    """
    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc
    sblk.body = sblk.body[1:]

    args = extra["args"]
    arg = args[0]
    parse_res = openmp_parser.parse(arg.value)
    if config.DEBUG_ARRAY_OPT >= 1:
        print("args:", args, type(args))
        print("arg:", arg, type(arg), arg.value, type(arg.value))
        print(parse_res.pretty())
    visitor = OpenmpVisitor(func_ir, blocks, blk_start, blk_end, body_blocks, loc, state)
    try:
        visitor.transform(parse_res)
    except VisitError as e:
        if isinstance(e.__context__, UnspecifiedVarInDefaultNone):
            print(str(e.__context__))
            raise e.__context__
        else:
            print("Internal error for OpenMp pragma '{}'".format(arg.value), e.__context__, type(e.__context__))
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
        print("Internal error for OpenMp pragma '{}'".format(arg.value))
        sys.exit(-3)
    assert(blocks is visitor.blocks)

ffi = FFI()
ffi.cdef('void omp_set_num_threads(int num_threads);')
ffi.cdef('int omp_get_thread_num(void);')
ffi.cdef('int omp_get_num_threads(void);')
ffi.cdef('double omp_get_wtime(void);')
ffi.cdef('void omp_set_dynamic(int num_threads);')
ffi.cdef('void omp_set_nested(int nested);')
ffi.cdef('void omp_set_max_active_levels(int levels);')
ffi.cdef('int omp_get_max_active_levels(void);')
ffi.cdef('int omp_get_max_threads(void);')

C = ffi.dlopen(None)
#C = ffi.dlopen(iomplib)
omp_set_num_threads = C.omp_set_num_threads
omp_get_thread_num = C.omp_get_thread_num
omp_get_num_threads = C.omp_get_num_threads
omp_get_wtime = C.omp_get_wtime
omp_set_dynamic = C.omp_set_dynamic
omp_set_nested = C.omp_set_nested
omp_set_max_active_levels = C.omp_set_max_active_levels
omp_get_max_active_levels = C.omp_get_max_active_levels
omp_get_max_threads = C.omp_get_max_threads
