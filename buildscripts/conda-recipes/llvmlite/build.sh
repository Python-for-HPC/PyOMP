#!/bin/bash

set -x

if [[ $(uname) == Darwin ]]; then
  if [[ $build_platform == osx-arm64 ]]; then
      CLANG_PKG_SELECTOR=clangxx_osx-arm64=12
  else
      CLANG_PKG_SELECTOR=clangxx_osx-64=10
  fi
  ${SYS_PREFIX}/bin/conda create -y -p ${SRC_DIR}/bootstrap ${CLANG_PKG_SELECTOR}
  export PATH=${SRC_DIR}/bootstrap/bin:${PATH}
  CONDA_PREFIX=${SRC_DIR}/bootstrap \
    . ${SRC_DIR}/bootstrap/etc/conda/activate.d/*
  export CONDA_BUILD_SYSROOT=${CONDA_BUILD_SYSROOT:-/opt/MacOSX${MACOSX_DEPLOYMENT_TARGET}.sdk}
  export CXXFLAGS=${CFLAGS}" -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
  export CFLAGS=${CFLAGS}" -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
  SYSROOT_DIR=${CONDA_BUILD_SYSROOT}
  CFLAG_SYSROOT="--sysroot ${SYSROOT_DIR}"
  # export LLVM_CONFIG explicitly as the one installed from llvmdev
  # in the build root env, the one in the bootstrap location needs to be ignored.
  export LLVM_CONFIG="${PREFIX}/bin/llvm-config"
  ${LLVM_CONFIG} --version
fi

if [ -n "$MACOSX_DEPLOYMENT_TARGET" ]; then
    if [[ $build_platform == osx-arm64 ]]; then
        export MACOSX_DEPLOYMENT_TARGET=11.0
    else
        # OSX needs 10.7 or above with libc++ enabled
        export MACOSX_DEPLOYMENT_TARGET=10.10
    fi
fi


# This is the clang compiler prefix
if [[ $build_platform == osx-arm64 ]]; then
    DARWIN_TARGET=arm64-apple-darwin20.0.0
else
    DARWIN_TARGET=x86_64-apple-darwin13.4.0
fi


export PYTHONNOUSERSITE=1
# Enables static linking of stdlibc++
export LLVMLITE_CXX_STATIC_LINK=1
export LLVMLITE_SKIP_LLVM_VERSION_CHECK=1

#EXTRA_LLVM_LIBS="-L /opt/intel/intelpython3/lib -fno-lto" LDFLAGS=-fPIC LLVM_CONFIG=$config $PYTHON setup.py build --force
#EXTRA_LLVM_LIBS="-L /opt/intel/intelpython3/lib -fno-lto" LLVM_CONFIG=$config $PYTHON setup.py install

#EXTRA_LLVM_LIBS="-L /opt/intel/intelpython3/lib -fno-lto" LDFLAGS=-fPIC $PYTHON setup.py build --force
#EXTRA_LLVM_LIBS="-L /opt/intel/intelpython3/lib -fno-lto" $PYTHON setup.py install

# THOSE WERE ENABLED
#EXTRA_LLVM_LIBS="-fno-lto" LDFLAGS=-fPIC $PYTHON setup.py build --force
#EXTRA_LLVM_LIBS="-fno-lto" $PYTHON setup.py install

$PYTHON setup.py build --force
$PYTHON setup.py install

# Move below to clang RT package.

#cp "$BUILD_PREFIX/lib/libomptarget*.bc" "$PREFIX/lib"

#declare -a tools=( \
#"opt"              \
#"llc"              \
#"llvm-link"        \
#"clang"            \
#"lld"              \
#)

#for tool in "${tools[@]}"; do
#    cp "$BUILD_PREFIX/bin/$tool" "$PREFIX/bin/$tool"
#done
