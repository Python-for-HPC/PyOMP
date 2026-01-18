from lark import Lark, Transformer

from .omp_grammar import openmp_grammar

openmp_parser = Lark(openmp_grammar, start="openmp_statement")
var_collector_parser = Lark(openmp_grammar, start="openmp_statement")
