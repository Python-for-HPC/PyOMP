#!/bin/bash
 
export PYTHONNOUSERSITE=1
export LLVMLITE_CXX_STATIC_LINK=1
export LLVMLITE_SKIP_LLVM_VERSION_CHECK=1
export CC=gcc
export CXX=g++

pushd ../../llvmliteWithOpenmp
rm -rf build
config=$LLVM_BUILD_PATH/bin/llvm-config
LLVM_CONFIG=$config python setup.py clean
EXTRA_LLVM_LIBS="-g -fno-lto" LDFLAGS=-fPIC LLVM_CONFIG=$config python setup.py build --force
EXTRA_LLVM_LIBS="-g -fno-lto" LLVM_CONFIG=$config python setup.py install
popd
