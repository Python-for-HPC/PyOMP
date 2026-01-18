import llvmlite.ir as lir


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
