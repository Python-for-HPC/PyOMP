cmake_minimum_required(VERSION 3.20)
project(pyomp-pass)

find_package(LLVM REQUIRED CONFIG)

if(NOT LLVM_VERSION_MAJOR EQUAL 14)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 14.x")
endif()

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

target_link_libraries(IntrinsicsOpenMP
  PRIVATE ${LLVM_AVAILABLE_LIBS}
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

set_property(TARGET IntrinsicsOpenMP APPEND_STRING PROPERTY LINK_FLAGS "-flto -Wl,--exclude-libs,ALL")

install(TARGETS IntrinsicsOpenMP
  EXPORT IntrinsicsOpenMP
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
