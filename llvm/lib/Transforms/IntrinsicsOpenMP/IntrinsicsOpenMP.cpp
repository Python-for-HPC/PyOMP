//===- IntrinsicsOpenMP.cpp - Codegen OpenMP from IR intrinsics ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code generation for OpenMP from intrinsics embedded in
// the IR, using the OpenMPIRBuilder
//
//===-------------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IntrinsicsOpenMP/IntrinsicsOpenMP.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace omp;

#define DEBUG_TYPE "intrinsics-openmp"

// TODO: Increment.
STATISTIC(NumOpenMPRegions, "Counts number of OpenMP regions created");

namespace {

  // TODO: explose clauses through namespace omp?
  enum DSAType {
    DSA_PRIVATE,
    DSA_FIRSTPRIVATE,
    DSA_SHARED
  };

  static const DenseMap<StringRef, Directive> StringToDir = {
      {"DIR.OMP.PARALLEL", OMPD_parallel},
      {"DIR.OMP.SINGLE", OMPD_single},
  };

  static const DenseMap<StringRef, DSAType> StringToDSA = {
      {"QUAL.OMP.PRIVATE", DSA_PRIVATE},
      {"QUAL.OMP.FIRSTPRIVATE", DSA_FIRSTPRIVATE},
  };

  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;

  LoopInfo *LI = nullptr;
  DominatorTree *DT = nullptr;

  struct IntrinsicsOpenMP: public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    IntrinsicsOpenMP() : ModulePass(ID) {}

    bool runOnModule(Module &M) override {
      dbgs() << "=== Start IntrinsicsOpenMPPass\n";

      Function *RegionEntryF = M.getFunction("llvm.directive.region.entry");

      // Return early for lack of directive intrinsics.
      if (!RegionEntryF)
        return false;

      OpenMPIRBuilder OMPBuilder(M);
      OMPBuilder.initialize();

      bool Changed = false;

      SmallVector<User *, 4> RegionEntryUsers(RegionEntryF->users());
      for(User *Usr : RegionEntryUsers) {
        Changed = true;

        dbgs() << "Found Usr " << *Usr << "\n";
        CallBase *CBEntry = dyn_cast<CallBase>(Usr);
        assert(CBEntry && "Expected call to region entry intrinsic");

        SmallVector<OperandBundleDef, 16> OpBundles;
        Directive Dir;
        DenseMap<Value *, DSAType>  DSAValueMap;
        CBEntry->getOperandBundlesAsDefs(OpBundles);
        for(OperandBundleDef &O : OpBundles) {
          dbgs() << "OP " << O.getTag() << "\n";

          // Store directive kind from the operand bundle.
          auto It = StringToDir.find(O.getTag());
          if (It != StringToDir.end()) {
            dbgs() << "Found Str " << It->first << "\n";
            Dir = It->second;
            continue;
          }

          for(auto I = O.input_begin(), E = O.input_end(); I != E; ++I) {
            Value *V = dyn_cast<Value>(*I);
            assert(V && "Expected Value");
            auto It = StringToDSA.find(O.getTag());
            assert(It != StringToDSA.end() && "Expected DSA type found in map");
            DSAValueMap[V] = It->second;
          }
        }

        assert(CBEntry->getNumUses() == 1 &&
               "Expected single use of the directive entry CB");
        Use &U = *CBEntry->use_begin();
        CallBase *CBExit = dyn_cast<CallBase>(U.getUser());
        assert(CBExit && "Expected call to region exit intrinsic");
        dbgs() << "Found Use of " << *CBEntry << "\n->AT->\n" << *CBExit << "\n";

        if(Dir == OMPD_parallel) {
          BasicBlock *BB = CBEntry->getParent();
          Function *Fn = BB->getParent();
          const DebugLoc DL = BB->getTerminator()->getDebugLoc();

          //BasicBlock *StartBB = SplitBlock(BB, CBEntry, DT);
          BasicBlock *StartBB = BB->getUniqueSuccessor();
          assert(StartBB && "Expected unique successor at region start BB");
          BB->getTerminator()->eraseFromParent();
          // TODO: CBExit may not be the first instruction in EndBB, check and fix.
          BasicBlock *EndBB = SplitBlock(CBExit->getParent(), CBExit->getNextNode(), DT);
          BasicBlock *AfterBB = SplitBlock(EndBB, &*EndBB->getFirstInsertionPt(), DT);
          // Remove intrinsics of OpenMP tags, first CBExit to also remove use
          // of CBEntry, then CBEntry.
          CBExit->eraseFromParent();
          CBEntry->eraseFromParent();

          auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                               BasicBlock &ContinuationIP) {
            BasicBlock *CGStartBB = CodeGenIP.getBlock();
            BasicBlock *CGEndBB =
                SplitBlock(CGStartBB, &*CodeGenIP.getPoint(), DT, LI);
            assert(StartBB != nullptr && "StartBB should not be null");
            CGStartBB->getTerminator()->setSuccessor(0, StartBB);
            assert(EndBB != nullptr && "EndBB should not be null");
            EndBB->getTerminator()->setSuccessor(0, CGEndBB);
          };

          auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                            Value &Orig, Value &Inner,
                            Value *&ReplacementValue) -> InsertPointTy {
            auto It = DSAValueMap.find(&Inner);
            if (It->second == DSA_PRIVATE) {
              OMPBuilder.Builder.restoreIP(AllocaIP);
              // TODO: looks like all values come in as pointer, may need to
              // adjust the type.
              Type *VTy = Inner.getType()->getPointerElementType();
              ReplacementValue = OMPBuilder.Builder.CreateAlloca(
                  VTy, /*ArraySize */ nullptr, Inner.getName());
              dbgs() << "Privatizing Inner " << Inner << " -> to -> " << *ReplacementValue
                     << "\n";
            } else if (It->second == DSA_FIRSTPRIVATE) {
              OMPBuilder.Builder.restoreIP(AllocaIP);
              Type *VTy = Inner.getType()->getPointerElementType();
              Value *V = OMPBuilder.Builder.CreateLoad(VTy, &Inner, Orig.getName() + ".reload");
              ReplacementValue = OMPBuilder.Builder.CreateAlloca(
                  VTy, /*ArraySize */ nullptr, Inner.getName());
              OMPBuilder.Builder.restoreIP(CodeGenIP);
              OMPBuilder.Builder.CreateStore(V, ReplacementValue);
              dbgs() << "Firstprivatizing Inner " << Inner << " -> to -> " << *ReplacementValue
                     << "\n";
            } else {
              ReplacementValue = &Inner;
            }

            return CodeGenIP;
          };

          auto FiniCB = [&](InsertPointTy CodeGenIP) {};

          OpenMPIRBuilder::LocationDescription Loc(InsertPointTy(BB, BB->end()),
                                               DL);

          IRBuilder<>::InsertPoint AllocaIP(
              &Fn->getEntryBlock(),
              Fn->getEntryBlock().getFirstInsertionPt());

          // TODO: amend for parameters.
          InsertPointTy AfterIP = OMPBuilder.createParallel(
              Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
              /* IfCondition*/ nullptr, /* NumThreads */ nullptr,
              OMP_PROC_BIND_default, /* IsCancellable */ false);
          BranchInst::Create(AfterBB, AfterIP.getBlock());

          dbgs() << "=== Before Fn\n";
          Fn->dump();
          dbgs() << "=== End of Before Fn\n";
          OMPBuilder.finalize(Fn, /* AllowExtractorSinking */ true);
          dbgs() << "=== Finalize Fn\n";
          Fn->dump();
          dbgs() << "=== End of Finalize Fn\n";
        } else if (Dir == OMPD_single) {
          // TODO: handle code generation.
          CBExit->eraseFromParent();
          CBEntry->eraseFromParent();
        } else {
          dbgs() << "Unknown directive " << *CBEntry << "\n";
          assert(false && "Unknown directive");
        }
      }

      errs() << "IntrinscsOpenMP pass\n";
      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      ModulePass::getAnalysisUsage(AU);
    }
  };

}

PreservedAnalyses IntrinsicsOpenMPPass::run(Module &M, ModuleAnalysisManager &AM) {
    IntrinsicsOpenMP IOMP;
    bool Changed = IOMP.runOnModule(M);

    if(Changed)
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
}

char IntrinsicsOpenMP::ID = 0;
static RegisterPass<IntrinsicsOpenMP> X("intrinsics-openmp", "IntrinsicsOpenMP Pass");

static RegisterStandardPasses Y(PassManagerBuilder::EP_ModuleOptimizerEarly,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new IntrinsicsOpenMP());
                                });

static RegisterStandardPasses Z(PassManagerBuilder::EP_EnabledOnOptLevel0,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new IntrinsicsOpenMP());
                                });
ModulePass *llvm::createIntrinsicsOpenMPPass() { return new IntrinsicsOpenMP(); }
