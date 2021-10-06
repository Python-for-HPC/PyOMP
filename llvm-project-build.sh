#!/bin/bash

cd llvm-project/llvm

PREFIX=../../../llvm-project-install

set -x

export ENABLE_SPATIAL_ADVISOR=1

declare -a _cmake_config
_cmake_config+=(-DCMAKE_INSTALL_PREFIX:PATH=${PREFIX})
_cmake_config+=(-DCMAKE_BUILD_TYPE:STRING=Debug)
_cmake_config+=(-DLLVM_ENABLE_ASSERTIONS:BOOL=ON)
_cmake_config+=(-DLINK_POLLY_INTO_TOOLS:BOOL=ON)
# Urgh, llvm *really* wants to link to ncurses / terminfo and we *really* do not want it to.
_cmake_config+=(-DHAVE_TERMINFO_CURSES=OFF)
# Sometimes these are reported as unused. Whatever.
_cmake_config+=(-DHAVE_TERMINFO_NCURSES=OFF)
_cmake_config+=(-DHAVE_TERMINFO_NCURSESW=OFF)
_cmake_config+=(-DHAVE_TERMINFO_TERMINFO=OFF)
_cmake_config+=(-DHAVE_TERMINFO_TINFO=OFF)
_cmake_config+=(-DHAVE_TERMIOS_H=OFF)
#_cmake_config+=(-DCLANG_ENABLE_LIBXML=OFF)
#_cmake_config+=(-DLIBOMP_INSTALL_ALIASES=OFF)
_cmake_config+=(-DLLVM_ENABLE_RTTI=OFF)
_cmake_config+=(-DLLVM_TARGETS_TO_BUILD=X86)
_cmake_config+=(-DCMAKE_COLOR_MAKEFILE=0)
#_cmake_config+=(-DLLVM_USE_INTEL_JITEVENTS:BOOL=ON)
_cmake_config+=(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
#_cmake_config+=(-DLLVM_ENABLE_WERROR=ON)
_cmake_config+=(-DINTEL_CUSTOMIZATION=1)
_cmake_config+=(-DINTEL_SPECIFIC_CILKPLUS=1)
_cmake_config+=(-DINTEL_SPECIFIC_OPENMP=1)
_cmake_config+=(-DLLVM_BUILD_RUNTIME=OFF)

rm -rf build
mkdir build
cd build

cmake -G'Unix Makefiles'     \
      "${_cmake_config[@]}"  \
      ..

make -j4 VERBOSE=1
make install

cd ../..
