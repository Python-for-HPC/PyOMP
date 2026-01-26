#!/usr/bin/env bash

set -e

# Create a unique temporary directory for this job.
TMPDIR=/tmp/pyomp/${CI_JOB_ID}
mkdir -p ${TMPDIR}
pushd ${TMPDIR}

# Set the LLVM_VERSION to use.
export LLVM_VERSION="15.0.7"

# Set the envs directory under the temporary directory.
export CONDA_ENVS_DIRS="${TMPDIR}/_stage/miniconda3/envs"

# Install miniconda and llvmdev environment.
source ${CI_PROJECT_DIR}/buildscripts/cibuildwheel/setup-miniconda3.sh

# Export environment variables for building and testing.
export LLVM_DIR="${CONDA_ENVS_DIRS}/llvmdev-${LLVM_VERSION}"
export PATH="${CONDA_ENVS_DIRS}/llvmdev-${LLVM_VERSION}/bin:${PATH}"
export USE_CXX11_ABI="1"
export PIP_NO_INPUT="1"

# Create and activate a conda environment with the desired Python version.
conda create -n py-${PYOMP_CI_PYTHON_VERSION} -c conda-forge -y python=${PYOMP_CI_PYTHON_VERSION}
conda activate py-${PYOMP_CI_PYTHON_VERSION}

# Clone and fetch the commit with history for package versioning.
git clone https://github.com/${GITHUB_PROJECT_ORG}/${GITHUB_PROJECT_NAME}.git --single-branch
cd ${GITHUB_PROJECT_NAME}
git fetch origin ${CI_COMMIT_SHA}
git checkout ${CI_COMMIT_SHA}

# Install pyomp.
CC=gcc CXX=g++ python -m pip install -v .

# Run host OpenMP tests.
TEST_DEVICES=0 RUN_TARGET=0 python -m numba.runtests -v -- numba.openmp.tests.test_openmp
# Run device (cpu target) OpenMP tests.
OMP_TARGET_OFFLOAD=mandatory TEST_DEVICES=1 RUN_TARGET=1 python -m numba.runtests -v -- numba.openmp.tests.test_openmp.TestOpenmpTarget

popd
