#!/usr/bin/env bash

set -euxo pipefail

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

conda create -n llvmdev -y

# Create llvmdev environment and install manylinux llvmdev 14.0.6 from numba channel.
echo "Installing manylinux llvmdev 14.0.6..."
conda activate llvmdev
if [ "${OS_NAME}" = "MacOSX" ]; then
    conda install -y -c conda-forge llvmdev=14.0.6
else
    conda install -y -c conda-forge llvmdev=14.0.6
fi
conda deactivate

echo "Installing clang 14.0.6..."
# Create clang14 environment and install clang 14.0.6.
conda create -n clang14 -c conda-forge -y clang=14.0.6
