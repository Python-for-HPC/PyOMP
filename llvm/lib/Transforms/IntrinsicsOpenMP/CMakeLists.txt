add_llvm_component_library(LLVMIntrinsicsOpenMP
  IntrinsicsOpenMP.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/IntrinsicsOpenMP

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  FrontendOpenMP
  IPO
  Support
  TransformUtils
  )
