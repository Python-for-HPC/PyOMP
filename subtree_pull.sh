#!/bin/sh

git subtree pull --prefix llvm-project https://github.com/ggeorgakoudis/llvm-project.git numba --squash
git subtree pull --prefix llvmlite https://github.com/ggeorgakoudis/llvmliteWithOpenmp.git op2_llvm12 --squash
git subtree pull --prefix numba https://github.com/ggeorgakoudis/NumbaWithOpenmp.git with_openmp --squash

