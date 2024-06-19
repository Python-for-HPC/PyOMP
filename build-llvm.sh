#!/bin/bash

cmake -G Ninja \
    -S llvm-project/llvm \
    -B ${LLVM_BUILD_PATH} \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS='clang' \
    -DLLVM_ENABLE_RUNTIMES='openmp' \
    -DCMAKE_INSTALL_PREFIX=${LLVM_BUILD_PATH}/install \
    -DBUILD_SHARED_LIBS=off \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_PARALLEL_COMPILE_JOBS=16 \
    -DLLVM_PARALLEL_LINK_JOBS=1

pushd ${LLVM_BUILD_PATH}
ninja install
popd
