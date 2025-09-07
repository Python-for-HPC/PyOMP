#!/usr/bin/env bash

set -e

if [ -n "${CI_COMMIT_TAG}" ]; then
  LABEL="main"
else
  LABEL="test"
fi

# Create a temporary directory for the build to clone the full repo for package
# versioning.
TMPDIR=/tmp/ggeorgak/${CI_JOB_ID}
mkdir -p ${TMPDIR}
pushd ${TMPDIR}

# Clone and fetch the commit with history for package versioning.
git clone https://github.com/${GITHUB_PROJECT_ORG}/${GITHUB_PROJECT_NAME}.git --single-branch
cd ${GITHUB_PROJECT_NAME}
git fetch origin ${CI_COMMIT_SHA}
git checkout ${CI_COMMIT_SHA}

# Set pkg dir per job to avoid conflicts.
export CONDA_PKGS_DIRS=/tmp/ggeorgak/conda-pkgs-${CI_JOB_ID}
mkdir -p "$CONDA_PKGS_DIRS"

function deploy_conda() {
  PKG="${1}"

  echo "=> Conda deploy ${PKG}"

  set -x

  export CONDA_BLD_PATH="/tmp/ggeorgak/conda-build-${PYOMP_CI_BUILD_PKG}-${PYOMP_CI_PYTHON_VERSION}"
  conda build --no-lock --no-locking --user python-for-hpc --label ${LABEL} \
    -c python-for-hpc/label/${LABEL} -c conda-forge \
    --python ${PYOMP_CI_PYTHON_VERSION} \
    buildscripts/conda-recipes/${PKG}

  rm -rf ${CONDA_BLD_PATH}
  set +x
}

echo "=> Building ${PYOMP_CI_BUILD_PKG} Python version ${PYOMP_CI_PYTHON_VERSION} Label ${LABEL}"


case ${PYOMP_CI_BUILD_PKG} in

  "pyomp")
    deploy_conda "pyomp"
    ;;

  *)
    echo "Unknown package to build ${PYOMP_CI_BUILD_PKG}"
    exit 1
    ;;

esac

popd
