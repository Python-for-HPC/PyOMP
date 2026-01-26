#!/usr/bin/env bash

set -euxo pipefail

# Read LLVM_VERSION from environment and error if not set
if [ -z "${LLVM_VERSION:-}" ]; then
    echo "Error: LLVM_VERSION environment variable is not set." >&2
    exit 1
fi

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

# Create clangdev ${LLVM_VERSION}
echo "Installing manylinux llvmdev ${LLVM_VERSION}..."
conda create -n llvmdev-${LLVM_VERSION} -c conda-forge -q -y clang=${LLVM_VERSION} clang-tools=${LLVM_VERSION} llvmdev=${LLVM_VERSION}
