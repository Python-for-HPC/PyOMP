#!/bin/bash
set -e

if [ -n "${CI_COMMIT_TAG}" ]; then
  LABEL="main"
else
  LABEL="test"
fi

function deploy_conda() {
  PKG="${1}"

  echo "=> Conda deploy ${PKG}"

  set -x

  if [ -z "${PYOMP_CI_PYTHON_VERSION}" ]; then
    export CONDA_BLD_PATH="/tmp/ggeorgak/conda-build-${PYOMP_CI_BUILD_PKG}-noarch"
    conda build --no-lock --no-locking --user python-for-hpc --label ${LABEL} \
      -c python-for-hpc/label/${LABEL} -c conda-forge \
      ${CI_PROJECT_DIR}/buildscripts/conda-recipes/${PKG}
  else
    export CONDA_BLD_PATH="/tmp/ggeorgak/conda-build-${PYOMP_CI_BUILD_PKG}-${PYOMP_CI_PYTHON_VERSION}"
    conda build --no-lock --no-locking --user python-for-hpc --label ${LABEL} \
      -c python-for-hpc/label/${LABEL} -c conda-forge \
      --python ${PYOMP_CI_PYTHON_VERSION} \
      ${CI_PROJECT_DIR}/buildscripts/conda-recipes/${PKG}
  fi

  rm -rf ${CONDA_BLD_PATH}
  set +x
}

echo "=> Building ${PYOMP_CI_BUILD_PKG} Python version ${PYOMP_CI_PYTHON_VERSION} Label ${LABEL}"


case ${PYOMP_CI_BUILD_PKG} in

  "llvm-openmp-dev")
    deploy_conda "llvm-openmp-dev"
    ;;

  "llvmlite")
    deploy_conda "llvmlite"
    ;;

  "numba")
    deploy_conda "numba"
    ;;

  "pyomp")
    deploy_conda "pyomp"
    ;;

  *)
    echo "Unknown package to build ${PYOMP_CI_BUILD_PKG}"
    exit 1
    ;;

esac

