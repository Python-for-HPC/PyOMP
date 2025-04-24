cmake_minimum_required(VERSION 3.20)
project(pyomp-pass)

# Set this to a valid LLVM installation dir
set(LT_LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")

# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM CONFIG)
if("${LLVM_VERSION_MAJOR}" VERSION_LESS 14)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 14 or above")
endif()

# HelloWorld includes headers from LLVM - update the include paths accordingly
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

# Use the same C++ standard as LLVM does
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

# LLVM is normally built without RTTI. Be consistent with that.
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

add_library(IntrinsicsOpenMP SHARED 
  CGIntrinsicsOpenMP.cpp 
  DebugOpenMP.cpp 
  IntrinsicsOpenMP.cpp)

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(IntrinsicsOpenMP
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

install(TARGETS IntrinsicsOpenMP
  EXPORT IntrinsicsOpenMP
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
