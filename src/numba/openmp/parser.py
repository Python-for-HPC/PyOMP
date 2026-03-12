from lark import Lark

from .omp_grammar import openmp_grammar

# Use Lark's contextual lexer with LALR to resolve keyword-vs-identifier
# overlaps (e.g. `TO` vs `PYTHON_NAME`) without reserving words.
_LARK_KWARGS = dict(parser="lalr", lexer="contextual")

openmp_parser = Lark(openmp_grammar, start="openmp_statement", **_LARK_KWARGS)
var_collector_parser = Lark(openmp_grammar, start="openmp_statement", **_LARK_KWARGS)
