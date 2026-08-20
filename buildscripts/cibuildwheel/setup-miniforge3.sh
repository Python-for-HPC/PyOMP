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

echo "Installing Miniforge..."
mkdir -p _downloads
curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${OS_NAME}-$(uname -m).sh" -o _downloads/miniforge3.sh
mkdir -p _stage
bash _downloads/miniforge3.sh -b -f -p "_stage/miniforge3"
echo "Miniforge installed"
source "_stage/miniforge3/bin/activate" base

# Create conda environment with tools and libraries for the LLVM_VERSION.
echo "Installing llvmdev ${LLVM_VERSION}..."
conda create -n llvmdev-${LLVM_VERSION} --override-channels -c conda-forge -q -y clang=${LLVM_VERSION} clangxx=${LLVM_VERSION} clang-tools=${LLVM_VERSION} llvmdev=${LLVM_VERSION} zstd
