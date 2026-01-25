from lark import Transformer
from lark.exceptions import VisitError

from numba.core import ir, types, typed_passes
from numba.core.analysis import (
    compute_cfg_from_blocks,
    compute_use_defs,
    compute_live_map,
)
from numba.core.ir_utils import (
    dump_blocks,
    get_call_table,
    visit_vars,
    build_definitions,
)
import copy
import operator
import sys
import os

from .config import DEBUG_OPENMP
from .parser import openmp_parser
from .analysis import (
    remove_ssa,
    user_defined_var,
    is_dsa,
    is_private,
    is_internal_var,
    has_user_defined_var,
    get_user_defined_var,
    get_enclosing_region,
    add_enclosing_region,
    filter_nested_loops,
    remove_privatized,
    get_var_from_enclosing,
    remove_indirections,
    add_tags_to_enclosing,
    get_blocks_between_start_end,
    get_itercount,
)
from .exceptions import (
    UnspecifiedVarInDefaultNone,
    ParallelForExtraCode,
    ParallelForWrongLoopCount,
    ParallelForInvalidCollapseCount,
    MultipleNumThreadsClauses,
    NonconstantOpenmpSpecification,
    NonStringOpenmpSpecification,
)
from .ir_utils import dump_block
from .tags import openmp_tag, NameSlice, get_tags_of_type
from .omp_ir import (
    openmp_region_start,
    openmp_region_end,
    default_shared_val,
)


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

        for inst_num, inst in enumerate(entry_block.body):
            if (
                isinstance(inst, ir.Assign)
                and isinstance(inst.value, ir.Expr)
                and inst.value.op == "call"
            ):
                loop_kind = _get_loop_kind(inst.value.func.name, call_table)
                if DEBUG_OPENMP >= 1:
                    print("loop_kind:", loop_kind)
                if loop_kind and loop_kind is range:
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

    def parallel_sections_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit parallel_sections_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for sections_construct.

    def sections_directive(self, args):
        raise NotImplementedError("Sections directive currently unsupported.")

    # Don't need a rule for SECTIONS.

    def sections_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit sections_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for section_construct.

    def section_directive(self, args):
        raise NotImplementedError("Section directive currently unsupported.")

    # Don't need a rule for SECTION.
    # Don't need a rule for atomic_construct.

    def atomic_directive(self, args):
        raise NotImplementedError("Atomic currently unsupported.")

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
                filter(lambda x: any([y not in x.name for y in names]), clauses)
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

    # Don't need a rule for simd_construct.

    def simd_directive(self, args):
        raise NotImplementedError("Simd directive currently unsupported.")

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

    def cancellation_point_directive(self, args):
        raise NotImplementedError("""Explicit cancellation points
                                 currently unsupported.""")

    def construct_type_clause(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit construct_type_clause", args, type(args), args[0])
        return args[0]

    def cancel_directive(self, args):
        raise NotImplementedError("Cancel directive currently unsupported.")

    # Don't need a rule for ORDERED.

    def flush_directive(self, args):
        raise NotImplementedError("Flush directive currently unsupported.")

    def region_phrase(self, args):
        raise NotImplementedError("No implementation for region phrase.")

    def PYTHON_NAME(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit PYTHON_NAME", args, type(args), str(args))
        return str(args)

    def NUMBER(self, args):
        if DEBUG_OPENMP >= 1:
            print("visit NUMBER", args, type(args), str(args))
        return int(args)


# This Transformer visitor class just finds the referenced python names
# and puts them in a list of VarName.  The default visitor function
# looks for list of VarNames in the args to that tree node and then
# concatenates them all together.  The final return value is a list of
# VarName that are variables used in the openmp clauses.


class VarName(str):
    pass


class OnlyClauseVar(VarName):
    pass


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
    for n, r in vardict.items():
        if n != r.name:
            new_vardict[n] = r
    visit_vars(blocks, replace_ssa_var_callback, new_vardict)


def remove_ssa_callback(var, unused):
    assert isinstance(var, ir.Var)
    new_var = ir.Var(var.scope, var.unversioned_name, var.loc)
    return new_var


def remove_ssa_from_func_ir(func_ir):
    typed_passes.PreLowerStripPhis()._strip_phi_nodes(func_ir)
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
    except Exception:
        print("generic transform exception")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("Internal error for OpenMp pragma '{}'".format(arg.value))
        sys.exit(-2)
    assert blocks is visitor.blocks
