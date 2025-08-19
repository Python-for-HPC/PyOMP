#!/usr/bin/env bash

if [ "${BUILD_STANDALONE}" = "1" ]; then
  rm -rf openmp-14.0.6.src*
  curl -L https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.6/openmp-14.0.6.src.tar.xz -o openmp-14.0.6.src.tar.xz
  EXPECTED_HASH="4f731ff202add030d9d68d4c6daabd91d3aeed9812e6a5b4968815cfdff0eb1f"
  # Compute the actual hash
  ACTUAL_HASH=$(sha256sum openmp-14.0.6.src.tar.xz | awk '{print $1}')
  if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
    echo "SHA256 checksum does not match! Exiting."
    exit 1
  fi

  tar xf openmp-14.0.6.src.tar.xz
  patch -p1 < patches/*

  PREFIX=$(pwd)/build
fi

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
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DCMAKE_CXX_FLAGS="${CONDA_TOOLCHAIN_CXX_INCLUDES}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  -DPACKAGE_VERSION="${PACKAGE_VERSION}" \
  -DENABLE_CHECK_TARGETS=OFF

exit 1

pushd build
make -j16 VERBOSE=1
make -j16 install || exit $?
popd
