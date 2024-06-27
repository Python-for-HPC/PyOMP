#!/bin/bash

# based on https://github.com/AnacondaRecipes/llvmdev-feedstock/blob/master/recipe/build.sh

set -x

# allow setting the targets to build as an environment variable
# default is LLVM 11 default architectures + RISCV.  Can remove this entire option in LLVM 13
LLVM_TARGETS_TO_BUILD=${LLVM_TARGETS_TO_BUILD:-"host;AMDGPU;NVPTX"}

declare -a _cmake_config
_cmake_config+=(-DCMAKE_INSTALL_PREFIX:PATH=${PREFIX})
_cmake_config+=(-DCMAKE_BUILD_TYPE:STRING=Release)
_cmake_config+=(-DLLVM_ENABLE_PROJECTS:STRING="clang")
#_cmake_config+=(-DLLVM_ENABLE_RUNTIMES:STRING="openmp")
# The bootstrap clang I use was built with a static libLLVMObject.a and I trying to get the same here
# _cmake_config+=(-DBUILD_SHARED_LIBS:BOOL=ON)
_cmake_config+=(-DLLVM_ENABLE_ASSERTIONS:BOOL=ON)
#_cmake_config+=(-DLINK_POLLY_INTO_TOOLS:BOOL=ON)
# Don't really require libxml2. Turn it off explicitly to avoid accidentally linking to system libs
_cmake_config+=(-DLLVM_ENABLE_LIBXML2:BOOL=OFF)
# Urgh, llvm *really* wants to link to ncurses / terminfo and we *really* do not want it to.
_cmake_config+=(-DHAVE_TERMINFO_CURSES=OFF)
_cmake_config+=(-DLLVM_ENABLE_TERMINFO=OFF)
# Sometimes these are reported as unused. Whatever.
_cmake_config+=(-DHAVE_TERMINFO_NCURSES=OFF)
_cmake_config+=(-DHAVE_TERMINFO_NCURSESW=OFF)
_cmake_config+=(-DHAVE_TERMINFO_TERMINFO=OFF)
_cmake_config+=(-DHAVE_TERMINFO_TINFO=OFF)
_cmake_config+=(-DHAVE_TERMIOS_H=OFF)
_cmake_config+=(-DCLANG_ENABLE_LIBXML=OFF)
_cmake_config+=(-DLIBOMP_INSTALL_ALIASES=OFF)
_cmake_config+=(-DLLVM_ENABLE_RTTI=OFF)
_cmake_config+=(-DLLVM_TARGETS_TO_BUILD=${LLVM_TARGETS_TO_BUILD})
#_cmake_config+=(-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly)
_cmake_config+=(-DLLVM_INCLUDE_UTILS=ON) # for llvm-lit
_cmake_config+=(-DLLVM_INCLUDE_TESTS=OFF)
_cmake_config+=(-DLLVM_INCLUDE_BENCHMARKS:BOOL=OFF) # doesn't build without the rest of LLVM project
#_cmake_config+=(-DLLVM_HOST_TRIPLE:STRING=${HOST})
#_cmake_config+=(-DLLVM_DEFAULT_TARGET_TRIPLE:STRING=${HOST})
_cmake_config+=(-DCLANG_INCLUDE_TESTS=OFF)
_cmake_config+=(-DCLANG_INCLUDE_DOCS=OFF)
_cmake_config+=(-DLLVM_INCLUDE_EXAMPLES=OFF)
# do not build/install unused tools to save time and store.
_cmake_config+=(-DLLVM_TOOL_LLVM_LTO_BUILD=OFF)
_cmake_config+=(-DLLVM_TOOL_LLVM_LTO2_BUILD=OFF)
_cmake_config+=(-DLLVM_TOOL_BUGPOINT_BUILD=OFF)
_cmake_config+=(-DLLVM_TOOL_BUGPOINT_PASSES_BUILD=OFF)
_cmake_config+=(-DLLVM_TOOL_SANCOV_BUILD=OFF)
_cmake_config+=(-DLLVM_TOOL_LLVM_EXEGESIS_BUILD=OFF)
_cmake_config+=(-DLLVM_TOOL_LLVM_REDUCE_BUILD=OFF)
_cmake_config+=(-DLLVM_TOOL_LLVM_DWP_BUILD=OFF)
_cmake_config+=(-DCLANG_TOOL_CLANG_REPL_BUILD=OFF)
_cmake_config+=(-DCLANG_TOOL_CLANG_SCAN_DEPS_BUILD=OFF)
_cmake_config+=(-DCLANG_TOOL_CLANG_CHECK_BUILD=OFF)
_cmake_config+=(-DCLANG_TOOL_CLANG_RENAME_BUILD=OFF)
_cmake_config+=(-DCLANG_TOOL_CLANG_REFACTOR_BUILD=OFF)
_cmake_config+=(-DCLANG_TOOL_CLANG_EXTDEF_MAPPING_BUILD=OFF)

if [[ $(uname) == Linux ]]; then
  #_cmake_config+=(-DLLVM_USE_INTEL_JITEVENTS=ON)
  _cmake_config+=(-DLLVM_USE_LINKER=gold)
fi

if [[ "$target_platform" == "linux-ppc64le" ]]; then
  # avoid problematic flags when compiling OpenMP with built clang
  CFLAGS="$(echo ${CFLAGS} | sed 's/-mpower8-fusion//g')"
  CXXFLAGS="$(echo ${CXXFLAGS} | sed 's/-mpower8-fusion//g')"
fi

# For when the going gets tough:
#_cmake_config+=(-Wdev)
#_cmake_config+=(--debug-output)
#_cmake_config+=(--trace-expand)
#CPU_COUNT=1

mkdir -p build
cd build

cmake -G'Unix Makefiles'     \
      "${_cmake_config[@]}"  \
      ../llvm

ARCH=`uname -m`
if [ $ARCH == 'armv7l' ]; then # RPi need thread count throttling
    make -j2 VERBOSE=1
else
    make -j${CPU_COUNT} VERBOSE=1
fi

#make check-llvm-unit || exit $?

# From: https://github.com/conda-forge/llvmdev-feedstock/pull/53
make install || exit $?

# SVML tests on x86_64 arch only
#if [[ $ARCH == 'x86_64' ]]; then
#   bin/opt -S -vector-library=SVML -mcpu=haswell -O3 $RECIPE_DIR/numba-3016.ll | bin/FileCheck $RECIPE_DIR/numba-3016.ll || exit $?
#fi

cd ..
mkdir -p build-openmp
cd build-openmp

PACKAGE_VERSION=$(${PREFIX}/bin/llvm-config --version)
if [[ "${target_platform}" == osx-* ]]; then
  # See https://github.com/AnacondaRecipes/aggregate/issues/107
  export CPPFLAGS="-mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET} -isystem ${PREFIX}/include -D_FORTIFY_SOURCE=2"
  if [[ $ARCH == "arm64" ]]; then
    # Link with libclang_rt.builtins
    export LDFLAGS="${BUILD_PREFIX}/lib/clang/${PACKAGE_VERSION}/lib/libclang_rt.builtins_arm64_osx.a"
  fi
fi

cmake -G'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=${PREFIX}/bin/clang \
  -DCMAKE_CXX_COMPILER=${PREFIX}/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  -DPACKAGE_VERSION="${PACKAGE_VERSION}" \
  -DLLVM_DIR=${PREFIX}/lib/cmake/llvm \
  -DLLVM_ENABLE_RUNTIMES=openmp \
  -DLLVM_INCLUDE_TESTS=OFF \
  ../runtimes

make -j${CPU_COUNT} VERBOSE=1

make install || exit $?
