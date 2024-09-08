#!/bin/bash

rm -rf build

PACKAGE_VERSION=$(${CONDA_PREFIX}/bin/llvm-config --version)
if [[ "${target_platform}" == osx-* ]]; then
  # See https://github.com/AnacondaRecipes/aggregate/issues/107
  export CPPFLAGS="-mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET} -isystem ${CONDA_PREFIX}/include -D_FORTIFY_SOURCE=2"
elif [[ "${target_platform}" == linux-* ]]; then
  DIR1=${CONDA_PREFIX}/lib/gcc/${CONDA_TOOLCHAIN_HOST}/*/include/c++
  DIR2=${CONDA_PREFIX}/lib/gcc/${CONDA_TOOLCHAIN_HOST}/*/include/c++/${CONDA_TOOLCHAIN_HOST}
  CONDA_TOOLCHAIN_CXX_INCLUDES="-cxx-isystem ${DIR1} -cxx-isystem ${DIR2}"
fi

cmake -G'Unix Makefiles' \
  -B build \
  -S openmp-14.0.6.src \
  -DCMAKE_C_COMPILER=${CONDA_PREFIX}/bin/clang \
  -DCMAKE_CXX_COMPILER=${CONDA_PREFIX}/bin/clang++ \
  -DCMAKE_CXX_FLAGS="${CONDA_TOOLCHAIN_CXX_INCLUDES}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  -DPACKAGE_VERSION="${PACKAGE_VERSION}" \
  -DENABLE_CHECK_TARGETS=OFF

pushd build
make -j${CPU_COUNT} VERBOSE=1
make -j${CPU_COUNT} install || exit $?
popd

