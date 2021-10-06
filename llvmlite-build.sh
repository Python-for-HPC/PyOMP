#!/bin/bash

config=$PWD/llvm-project-install/bin/llvm-config
cd llvmliteWithOpenmp
export PYTHONNOUSERSITE=1
export LLVMLITE_CXX_STATIC_LINK=1
export LLVMLITE_SKIP_LLVM_VERSION_CHECK=1
export CC=gcc-9
export CXX=g++-9
#find . -name "*.so" -exec rm {} \;
rm -rf build
LLVM_CONFIG=$config python setup.py clean
EXTRA_LLVM_LIBS="-L /opt/intel/intelpython3/lib -fno-lto" LDFLAGS=-fPIC LLVM_CONFIG=$config python setup.py build --force
EXTRA_LLVM_LIBS="-L /opt/intel/intelpython3/lib -fno-lto" LLVM_CONFIG=$config python setup.py install
cd ..
