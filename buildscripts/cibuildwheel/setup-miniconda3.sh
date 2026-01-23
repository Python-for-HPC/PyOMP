#!/usr/bin/env bash

set -euxo pipefail

LLVM_VERSION="14.0.6"

if [ "$(uname)" = "Darwin" ]; then
    OS_NAME="MacOSX"
else
    OS_NAME="Linux"
fi

echo "Installing miniconda3..."
mkdir -p _downloads
curl -L https://repo.anaconda.com/miniconda/Miniconda3-py311_25.5.1-1-${OS_NAME}-$(uname -m).sh -o _downloads/mini3.sh
mkdir -p _stage
bash _downloads/mini3.sh -b -f -p "_stage/miniconda3"
echo "Miniconda installed"
source "_stage/miniconda3/bin/activate" base
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

# Create llvmdev environment and install llvmdev ${LLVM_VERSION}.
echo "Installing manylinux llvmdev ${LLVM_VERSION}..."
conda create -n llvmdev -c conda-forge -y llvmdev=${LLVM_VERSION}

# Create clang14 environment and install clang ${LLVM_VERSION}.
echo "Installing clang ${LLVM_VERSION}..."
conda create -n clang${LLVM_VERSION} -c conda-forge -y clang=${LLVM_VERSION}
