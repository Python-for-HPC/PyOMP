#ifndef LLVM_TRANSFORMS_INTRINSICS_OPENMP_H
#define LLVM_TRANSFORMS_INTRINSICS_OPENMP_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {


ModulePass *createIntrinsicsOpenMPPass();

} // namespace llvm

#endif // LLVM_TRANSFORMS_INTRINSICS_OPENMP_H