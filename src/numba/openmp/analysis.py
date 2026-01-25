from numba.core import ir, types, typing
from numba.core.analysis import _fix_loop_exit, compute_cfg_from_blocks
from numba.core.ir_utils import visit_vars
from numba.extending import intrinsic


def remove_ssa(var_name, scope, loc):
    # Get the base name of a variable, removing the SSA extension.
    var = ir.Var(scope, var_name, loc)
    return var.unversioned_name


def user_defined_var(var):
    if not isinstance(var, str):
        return False
    return not var.startswith("$")


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


def is_private(x):
    return x in [
        "QUAL.OMP.PRIVATE",
        "QUAL.OMP.FIRSTPRIVATE",
        "QUAL.OMP.LASTPRIVATE",
        "QUAL.OMP.TARGET.IMPLICIT",
    ]


def is_internal_var(var):
    # Determine if a var is a Python var or an internal Numba var.
    if var.is_temp:
        return True
    return var.unversioned_name != var.name


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


def get_enclosing_region(func_ir, cur_block):
    if not hasattr(func_ir, "openmp_enclosing"):
        func_ir.openmp_enclosing = {}
    if cur_block in func_ir.openmp_enclosing:
        return func_ir.openmp_enclosing[cur_block]
    else:
        return None


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


def remove_privatized(x):
    if isinstance(x, ir.Var):
        x = x.name

    if isinstance(x, str) and x.endswith("%privatized"):
        return x[: len(x) - len("%privatized")]
    else:
        return x


def get_var_from_enclosing(enclosing_regions, var):
    if not enclosing_regions:
        return None
    if len(enclosing_regions) == 0:
        return None
    return enclosing_regions[-1].get_var_dsa(var)


def remove_indirections(clause):
    if not isinstance(clause, list):
        return clause
    while len(clause) == 1 and isinstance(clause[0], list):
        clause = clause[0]
    return clause


def add_tags_to_enclosing(func_ir, cur_block, tags):
    enclosing_region = get_enclosing_region(func_ir, cur_block)
    if enclosing_region:
        for region in enclosing_region:
            for tag in tags:
                region.add_tag(tag)


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


def is_target_tag(x):
    ret = x.startswith("DIR.OMP.TARGET") and x not in [
        "DIR.OMP.TARGET.DATA",
        "DIR.OMP.TARGET.ENTER.DATA",
        "DIR.OMP.TARGET.EXIT.DATA",
    ]
    return ret


def is_target_arg(name):
    return (
        name in ["QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.TARGET.IMPLICIT"]
        or name.startswith("QUAL.OMP.MAP")
        or name.startswith("QUAL.OMP.REDUCTION")
    )


def in_openmp_region(builder):
    if hasattr(builder, "_lowerer_push_alloca_callbacks"):
        return builder._lowerer_push_alloca_callbacks > 0
    else:
        return False


def get_name_var_table(blocks):
    """create a mapping from variable names to their ir.Var objects"""

    def get_name_var_visit(var, namevar):
        namevar[var.name] = var
        return var

    namevar = {}
    visit_vars(blocks, get_name_var_visit, namevar)
    return namevar


def is_pointer_target_arg(name, typ):
    if name.startswith("QUAL.OMP.REDUCTION"):
        return True
    if name.startswith("QUAL.OMP.MAP"):
        return True
    if name in ["QUAL.OMP.TARGET.IMPLICIT"]:
        if isinstance(typ, types.npytypes.Array):
            return True

    return False
