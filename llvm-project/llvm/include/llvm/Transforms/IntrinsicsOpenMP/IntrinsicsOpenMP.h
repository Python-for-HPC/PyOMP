#ifndef LLVM_TRANSFORMS_INTRINSICS_OPENMP_H
#define LLVM_TRANSFORMS_INTRINSICS_OPENMP_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class IntrinsicsOpenMPPass : public PassInfoMixin<IntrinsicsOpenMPPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  // Run always to lower OpenMP intrinsics.
  static bool isRequired() { return true; }
};

ModulePass *createIntrinsicsOpenMPPass();

} // namespace llvm

#endif // LLVM_TRANSFORMS_INTRINSICS_OPENMP_H