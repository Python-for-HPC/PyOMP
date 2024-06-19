#!/bin/bash

source pyenv/bin/activate
export LLVM_BUILD_PATH=${PWD}/build-llvm
export PATH=${LLVM_BUILD_PATH}/install/bin:$PATH
export NUMBA_OMP_LIB=${LLVM_BUILD_PATH}/install/lib/libomp.so
export NUMBA_OMPTARGET_LIB=${LLVM_BUILD_PATH}/install/lib/libomptarget.so
export OMP_TARGET_OFFLOAD=mandatory
export PYTHONHASHSEED=0
export NUMBA_OPENMP_DEVICE_TOOLCHAIN=1
