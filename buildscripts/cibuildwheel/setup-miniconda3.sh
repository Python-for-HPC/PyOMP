#!/usr/bin/env bash

set -euxo pipefail

if [ "$(uname)" = "Darwin" ]; then
    OS_NAME="MacOSX"
else
    OS_NAME="Linux"
fi

echo "Installing miniconda3..."
curl -L https://repo.anaconda.com/miniconda/Miniconda3-py311_25.5.1-1-${OS_NAME}-$(uname -m).sh -o mini3.sh
bash mini3.sh -b -f -p "${HOME}/miniconda3"
echo "Miniconda installed"
source "${HOME}/miniconda3/bin/activate" base
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

conda create -n llvmdev -y

# Create llvmdev environment and install manylinux llvmdev 14.0.6 from numba channel.
echo "Installing manylinux llvmdev 14.0.6..."
conda activate llvmdev
if [ "${OS_NAME}" = "MacOSX" ]; then
    conda install -y numba/label/ci_old_llvm14::llvmdev=14.0.6 --no-deps
else
    conda install -y -c conda-forge numba/label/manylinux2014::llvmdev=14.0.6 --no-deps
fi
conda deactivate

echo "Installing clang 14.0.6..."
# Create clang14 environment and install clang 14.0.6.
conda create -n clang14 -y clang=14.0.6
