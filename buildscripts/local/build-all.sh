#!/bin/bash

# Exit if any command fails.
set -e

echo "=> Create python venv..."
python -m venv pyenv
echo "=> Activate python venv..."
source pyenv/bin/activate
echo "=> Install python requirements..."
python -m pip install -r requirements.txt

echo "=> Build LLVM..."
export LLVM_BUILD_PATH=${PWD}/build-llvm
./build-llvm.sh
echo "=> Build llvmlite..."
./build-llvmlite.sh
echo "=> Build numba..."
./build-numba.sh

# Setup numba paths to LLVM.
pushd ../../NumbaWithOpenmp/numba
rm -f bin lib
ln -s ${LLVM_BUILD_PATH}/install/lib
ln -s ${LLVM_BUILD_PATH}/install/bin
popd

echo "DONE!"
