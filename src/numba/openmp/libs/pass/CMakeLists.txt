cmake_minimum_required(VERSION 3.20)
project(pyomp-pass)

if(NOT DEFINED LLVM_DIR OR LLVM_DIR STREQUAL "")
  message(FATAL_ERROR
    "LLVM_DIR is required but not defined. "
    "Please specify it with: cmake -DLLVM_DIR=/path/to/llvm/install ...")
endif()

message(STATUS "LLVM_DIR ${LLVM_DIR}")

find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH PATHS
  ${LLVM_DIR}
)

message(STATUS "Found LLVM version ${LLVM_VERSION}")

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

if(DEFINED ENV{USE_CXX11_ABI})
  target_compile_definitions(IntrinsicsOpenMP PRIVATE _GLIBCXX_USE_CXX11_ABI=$ENV{USE_CXX11_ABI})
endif()

# Use static library components to avoid issues with shared library dependencies.
set(llvm_libs LLVMPasses)

if(NOT APPLE)
  target_link_options(IntrinsicsOpenMP PRIVATE "-Wl,--no-undefined")
endif()

target_link_libraries(IntrinsicsOpenMP
  PRIVATE ${llvm_libs}
)

if(APPLE)
  set_property(TARGET IntrinsicsOpenMP APPEND_STRING PROPERTY LINK_FLAGS "-flto -Wl,-exported_symbol,_runIntrinsicsOpenMPPass")
else()
  set_property(TARGET IntrinsicsOpenMP APPEND_STRING PROPERTY LINK_FLAGS "-flto -Wl,--exclude-libs,ALL")
endif()

install(TARGETS IntrinsicsOpenMP
  EXPORT IntrinsicsOpenMP
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
