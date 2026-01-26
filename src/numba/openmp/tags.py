from numba.core import ir, types, cgutils
from numba import njit
from numba.core.ir_utils import replace_vars_inner
import llvmlite.ir as lir
import numpy as np

from .config import DEBUG_OPENMP
from .llvmlite_extensions import get_decl
from .analysis import typemap_lookup, is_dsa


def copy_np_array(x):
    return np.copy(x)


class StringLiteral:
    def __init__(self, x):
        self.x = x


class NameSlice:
    def __init__(self, name, the_slice):
        self.name = name
        self.the_slice = the_slice

    def __str__(self):
        return "NameSlice(" + str(self.name) + "," + str(self.the_slice) + ")"


def create_native_np_copy(arg_typ):
    # Use the high-level dispatcher API (`njit`) instead of the
    # removed/legacy `compile_isolated` helper.
    dispatcher = njit(copy_np_array)
    dispatcher.get_function_type()
    atypes = (arg_typ,)
    # copy_cres = dispatcher.get_compile_result(sig)
    dispatcher.compile(atypes)
    copy_cres = dispatcher.overloads[atypes]
    assert copy_cres is not None
    fndesc = getattr(copy_cres, "fndesc", None)
    assert fndesc is not None
    copy_name = getattr(fndesc, "llvm_cfunc_wrapper_name", None)
    assert copy_name is not None

    return (copy_name, copy_cres)


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

    def arg_to_str(self, x, lowerer, gen_copy=False):
        if DEBUG_OPENMP >= 1:
            print("arg_to_str:", x, type(x), self.load, type(self.load))

        typemap = lowerer.fndesc.typemap
        xtyp = None

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
                        field_info = []
                        for field in xsplit[1:]:
                            dm = lowerer.context.data_model_manager.lookup(cur_typ)
                            findex = dm._fields.index(field)
                            cur_typ = dm._members[findex]
                            llvm_type = lowerer.context.get_value_type(cur_typ)
                            if isinstance(cur_typ, types.CPointer):
                                llvm_type = llvm_type.pointee
                            field_info.append(f"{llvm_type} poison")
                            field_info.append("i32 " + str(findex))
                        fi_str = ", ".join(field_info)
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

                if gen_copy and isinstance(xtyp, types.npytypes.Array):
                    native_np_copy, copy_cres = create_native_np_copy(xtyp)
                    lowerer.library.add_llvm_module(copy_cres.library._final_module)
                    nnclen = len(native_np_copy)
                    decl += f', [{nnclen} x i8] c"{native_np_copy}"'

            # Add type information using a poison value operand for non-alloca pointers.
            if not isinstance(lowerer.getvar(x), lir.instructions.AllocaInstr):
                llvm_type = lowerer.context.get_value_type(xtyp)
                decl += f", {llvm_type} poison"
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
            decl = ",".join(
                [self.arg_to_str(x, lowerer, gen_copy=gen_copy) for x in arg_list]
            )
        else:
            decl = ",".join(
                [self.arg_to_str(x, lowerer, gen_copy=gen_copy) for x in arg_list]
            )

        return '"' + name_to_use + '"(' + decl + ")"

    def replace_vars_inner(self, var_dict):
        if isinstance(self.arg, ir.Var):
            self.arg = replace_vars_inner(self.arg, var_dict)

    def add_to_usedef_set(self, use_set, def_set, start):
        assert start in (True, False)
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


def get_tags_of_type(clauses, ctype):
    ret = []
    for c in clauses:
        if c.name == ctype:
            ret.append(c)
    return ret
