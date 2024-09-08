#!/bin/bash

set -x

export PYTHONNOUSERSITE=1

# Enables static linking of stdlibc++
export LLVMLITE_CXX_STATIC_LINK=1
# cmake is broken for osx builds.
#export LLVMLITE_USE_CMAKE=1
export LLVMLITE_SHARED=1

$PYTHON setup.py build --force
$PYTHON setup.py install
