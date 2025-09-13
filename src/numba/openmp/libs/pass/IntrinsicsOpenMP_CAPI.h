#ifndef LLVM_C_TRANSFORMS_INTRINSICS_OPENMP_H
#define LLVM_C_TRANSFORMS_INTRINSICS_OPENMP_H

#include "llvm-c/ExternC.h"
#include "llvm-c/Types.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCTransformsIntrinsicsOpenMP IntrinsicsOpenMP transformations
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::createIntrinsicsOpenMPPass function. */
void LLVMAddIntrinsicsOpenMPPass(LLVMPassManagerRef PM);

/**
 * @}
 */
LLVM_C_EXTERN_C_END
#endif
