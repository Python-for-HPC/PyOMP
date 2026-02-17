#!/usr/bin/env bash

set -e

# Create a unique temporary directory for this job.
TMPDIR=/tmp/pyomp/${CI_JOB_ID}
rm -rf ${TMPDIR}
mkdir -p ${TMPDIR}
pushd ${TMPDIR}

# Set the LLVM_VERSION to use.
export LLVM_VERSION="20.1.8"

# Set the envs directory under the temporary directory.
export CONDA_ENVS_DIRS="${TMPDIR}/_stage/miniconda3/envs"

# Install miniconda and llvmdev environment.
source ${CI_PROJECT_DIR}/buildscripts/cibuildwheel/setup-miniconda3.sh

# Export environment variables for building and testing.
export ENABLE_BUNDLED_LIBOMP="1"
export ENABLE_BUNDLED_LIBOMPTARGET="1"
export LLVM_DIR="${CONDA_ENVS_DIRS}/llvmdev-${LLVM_VERSION}"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"
export USE_CXX11_ABI="1"
export PIP_NO_INPUT="1"

# Create and activate a conda environment with the desired Python version.
conda create -n py-${PYOMP_CI_PYTHON_VERSION} -c conda-forge -y python=${PYOMP_CI_PYTHON_VERSION}
conda activate py-${PYOMP_CI_PYTHON_VERSION}
# Add extra packages needed to build openmp libraries.
conda install -c conda-forge -y zstd libffi

# Clone and fetch the commit with history for package versioning.
git clone https://github.com/${GITHUB_PROJECT_ORG}/${GITHUB_PROJECT_NAME}.git --single-branch
cd ${GITHUB_PROJECT_NAME}
git fetch origin ${CI_COMMIT_SHA}
git checkout ${CI_COMMIT_SHA}

# Install pyomp.
CC=clang CXX=clang++ python -m pip install -v .

# Run host OpenMP tests.
RUN_TARGET=0 python -m numba.runtests -v -- numba.openmp.tests.test_openmp
# Run device (cpu target) OpenMP tests.
OMP_TARGET_OFFLOAD=mandatory TEST_DEVICE=host RUN_TARGET=1 python -m numba.runtests -v -- numba.openmp.tests.test_openmp.TestOpenmpTarget

popd
