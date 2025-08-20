#!/usr/bin/env bash

set -exo pipefail

export LLVM_DIR=/root/miniconda3/envs/llvmdev
export CLANG_TOOL=/root/miniconda3/envs/clang14/bin/clang
# Set CXX11 ABI to 0 for compatibility with numba manylinux llvmdev builds.
export USE_CXX11_ABI=0
