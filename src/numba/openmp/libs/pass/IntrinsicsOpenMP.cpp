//===- IntrinsicsOpenMP.cpp - Codegen OpenMP from IR intrinsics
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code generation for OpenMP from intrinsics embedded in
// the IR.
//
//===-------------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cstddef>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Passes/PassPlugin.h>

#include "CGIntrinsicsOpenMP.h"
#include "DebugOpenMP.h"
#include "IntrinsicsOpenMP.h"
#include "IntrinsicsOpenMP_CAPI.h"

#include <algorithm>
#include <memory>

using namespace llvm;
using namespace omp;
using namespace iomp;

#define DEBUG_TYPE "intrinsics-openmp"

// TODO: Increment.
STATISTIC(NumOpenMPRegions, "Counts number of OpenMP regions created");

namespace {

class DirectiveRegionAnalysis;

class DirectiveRegion;
SmallVector<std::unique_ptr<DirectiveRegion>, 8> DirectiveRegionStorage;

class DirectiveRegion {
public:
  DirectiveRegion() = delete;

  void addNested(DirectiveRegionAnalysis &DRA, DirectiveRegion *DR);

  const SmallVector<DirectiveRegion *, 4> &getNested() const { return Nested; }

  CallBase *getEntry() const { return CBEntry; }

  CallBase *getExit() const { return CBExit; }

  void setParent(DirectiveRegion *P) { Parent = P; }

  DirectiveRegion *getParent() const { return Parent; }

  StringRef getTag() const {
    return getEntry()->getOperandBundleAt(0).getTagName();
  }

  static DirectiveRegion *create(CallBase *CBEntry, CallBase *CBExit) {
    // Use global storage of unique_ptr for auto-cleanup.
    DirectiveRegionStorage.push_back(
        std::unique_ptr<DirectiveRegion>(new DirectiveRegion{CBEntry, CBExit}));
    return DirectiveRegionStorage.back().get();
  }

private:
  CallBase *CBEntry;
  CallBase *CBExit;
  DirectiveRegion *Parent;
  SmallVector<DirectiveRegion *, 4> Nested;

  DirectiveRegion(CallBase *CBEntry, CallBase *CBExit)
      : CBEntry(CBEntry), CBExit(CBExit), Parent(nullptr) {}
};

class DirectiveRegionAnalysis {
public:
  explicit DirectiveRegionAnalysis(Function &F) : DT(F), PDT(F) {}

  bool directiveEncloses(DirectiveRegion *DR, DirectiveRegion *OtherDR) {
    // Use DominatorTree for Entry and PostDominatorTree for Exit.
    // PostDominator is effective for checking Exit when there are loops in
    // the CFG, since dominance does not hold for graphs with cycles, but
    // post-dominance does.
    if (DT.dominates(DR->getEntry(), OtherDR->getEntry()) &&
        PDT.dominates(DR->getExit(), OtherDR->getExit()))
      return true;

    return false;
  };

  bool directiveEntryDominates(DirectiveRegion *DR, DirectiveRegion *OtherDR) {
    if (DT.dominates(DR->getEntry(), OtherDR->getEntry()))
      return true;

    return false;
  }

private:
  DominatorTree DT;
  PostDominatorTree PDT;
};

void DirectiveRegion::addNested(DirectiveRegionAnalysis &DRA,
                                DirectiveRegion *DR) {
  // Insert in topological order.
  auto Compare = [&DRA](DirectiveRegion *DR, DirectiveRegion *OtherDR) {
    return DRA.directiveEntryDominates(DR, OtherDR);
  };

  Nested.insert(std::upper_bound(Nested.begin(), Nested.end(), DR, Compare),
                DR);
}

static SmallVector<Value *>
collectGlobalizedValues(DirectiveRegion &Directive) {

  SmallVector<Value *> GlobalizedValues;

  SmallVector<OperandBundleDef, 16> OpBundles;
  Directive.getEntry()->getOperandBundlesAsDefs(OpBundles);
  for (OperandBundleDef &O : OpBundles) {
    StringRef Tag = O.getTag();
    auto It = StringToDSA.find(Tag);
    if (It == StringToDSA.end())
      continue;

    const ArrayRef<Value *> &TagInputs = O.inputs();

    DSAType DSATy = It->second;

    switch (DSATy) {
    case iomp::DSA_FIRSTPRIVATE:
    case iomp::DSA_PRIVATE:
      continue;
    default:
      GlobalizedValues.push_back(TagInputs[0]);
    }
  }

  return GlobalizedValues;
}

struct IntrinsicsOpenMP {

  IntrinsicsOpenMP() { DebugOpenMPInit(); }

  bool runOnModule(Module &M) {
    // Codegen for nested or combined constructs assumes code is generated
    // bottom-up, that is from the innermost directive to the outermost. This
    // simplifies handling of DSA attributes by avoiding renaming values (tags
    // contain pre-lowered values when defining the data sharing environment)
    // when an outlined function privatizes them in the DSAValueMap.
    DEBUG_ENABLE(dbgs() << "=== Start IntrinsicsOpenMPPass v4\n");

    Function *RegionEntryF = M.getFunction("llvm.directive.region.entry");

    // Return early for lack of directive intrinsics.
    if (!RegionEntryF) {
      DEBUG_ENABLE(dbgs() << "No intrinsics directives, exiting...\n");
      return false;
    }

    DEBUG_ENABLE(dbgs() << "=== Dump Module\n"
                        << M << "=== End of Dump Module\n");

    CGIntrinsicsOpenMP CGIOMP(M);
    // Find all calls to directive intrinsics.
    SmallMapVector<Function *, SmallVector<DirectiveRegion *, 4>, 8>
        FunctionToDirectives;

    for (User *Usr : RegionEntryF->users()) {
      CallBase *CBEntry = dyn_cast<CallBase>(Usr);
      assert(CBEntry && "Expected call to directive entry");
      assert(CBEntry->getNumUses() == 1 &&
             "Expected single use of the directive entry");
      Use &U = *CBEntry->use_begin();
      CallBase *CBExit = dyn_cast<CallBase>(U.getUser());
      assert(CBExit && "Expected call to region exit intrinsic");
      Function *F = CBEntry->getFunction();
      assert(F == CBExit->getFunction() &&
             "Expected directive entry/exit in the same function");

      DirectiveRegion *DM = DirectiveRegion::create(CBEntry, CBExit);
      FunctionToDirectives[F].push_back(DM);
    }

    SmallVector<SmallVector<DirectiveRegion *, 4>, 4> DirectiveListVector;
    // Create directive lists per function, building trees of directive nests.
    // Each list stores directives outermost to innermost (pre-order).
    for (auto &FTD : FunctionToDirectives) {
      // Find the dominator tree for the function to find directive lists.
      Function &F = *FTD.first;
      auto &DirectiveRegions = FTD.second;
      DirectiveRegionAnalysis DRA{F};

      // Construct directive tree nests. First, find immediate parents, then add
      // nested children to parents.

      // Find immediate parents.
      for (auto *DR : DirectiveRegions) {
        for (auto *OtherDR : DirectiveRegions) {
          if (DR == OtherDR)
            continue;

          if (!DRA.directiveEncloses(OtherDR, DR))
            continue;

          DirectiveRegion *Parent = DR->getParent();
          if (!Parent) {
            DR->setParent(OtherDR);
            continue;
          }

          // If OtherDR is nested under Parent and encloses DR, then OtherDR is
          // the immediate parent of DR.
          if (DRA.directiveEncloses(Parent, OtherDR)) {
            DR->setParent(OtherDR);
            continue;
          }

          // Else, OtherDR must be enclosing Parent. It is not OtherDR's
          // immediate parent, hence no change to OtherDR.
          assert(DRA.directiveEncloses(OtherDR, Parent));
        }
      }
      // Gather all root directives, add nested children.
      SmallVector<DirectiveRegion *, 4> Roots;
      for (auto *DR : DirectiveRegions) {
        DirectiveRegion *Parent = DR->getParent();
        if (!Parent) {
          Roots.push_back(DR);
          continue;
        }

        Parent->addNested(DRA, DR);
      }

      // Travese the tree and add directives (outermost to innermost)
      // in a list.
      for (auto *Root : Roots) {
        SmallVector<DirectiveRegion *, 4> DirectiveList;

        auto VisitNode = [&DirectiveList](DirectiveRegion *Node, int Depth,
                                          auto &&VisitNode) -> void {
          DirectiveList.push_back(Node);
          for (auto *Nested : Node->getNested())
            VisitNode(Nested, Depth + 1, VisitNode);
        };

        VisitNode(Root, 0, VisitNode);

        DirectiveListVector.push_back(DirectiveList);

        auto PrintTree = [&]() {
          dbgs() << " === TREE\n";
          auto PrintNode = [](DirectiveRegion *Node, int Depth,
                              auto &&PrintNode) -> void {
            if (Depth) {
              for (int I = 0; I < Depth; ++I)
                dbgs() << "  ";
              dbgs() << "|_ ";
            }
            dbgs() << Node->getTag() << "\n";

            for (auto *Nested : Node->getNested())
              PrintNode(Nested, Depth + 1, PrintNode);
          };
          PrintNode(Root, 0, PrintNode);
          dbgs() << " === END OF TREE\n";
        };
        DEBUG_ENABLE(PrintTree());

        auto PrintList = [&]() {
          dbgs() << " === List\n";
          for (auto *DR : DirectiveList)
            dbgs() << DR->getTag() << " -> ";
          dbgs() << "EOL\n";
          dbgs() << " === End of List\n";
        };
        DEBUG_ENABLE(PrintList());
      }
    }

    // Iterate all directive lists and codegen.
    for (auto &DirectiveList : DirectiveListVector) {
      // If the outermost directive is a TARGET directive, collect globalized
      // values to set for codegen.
      // TODO: implement Directives as a class, parse each directive before
      // codegen, optimize privatization.
      auto *Outer = DirectiveList.front();
      if (Outer->getEntry()->getOperandBundleAt(0).getTagName().contains(
              "TARGET")) {
        auto GlobalizedValues = collectGlobalizedValues(*Outer);
        CGIOMP.setDeviceGlobalizedValues(GlobalizedValues);
      }
      // Iterate post-order, from innermost to outermost to avoid renaming
      // values in codegen.
      for (auto It = DirectiveList.rbegin(), E = DirectiveList.rend(); It != E;
           ++It) {
        DirectiveRegion *DR = *It;
        DEBUG_ENABLE(dbgs() << "Found Directive " << *DR->getEntry() << "\n");
        // Extract the directive kind and data sharing attributes of values
        // from the operand bundles of the intrinsic call.
        Directive Dir = OMPD_unknown;
        SmallVector<OperandBundleDef, 16> OpBundles;
        DSAValueMapTy DSAValueMap;

        // RAII for directive metainfo structs.
        OMPLoopInfoStruct OMPLoopInfo;
        ParRegionInfoStruct ParRegionInfo;
        TargetInfoStruct TargetInfo;
        TeamsInfoStruct TeamsInfo;

        MapVector<Value *, SmallVector<FieldMappingInfo, 4>>
            StructMappingInfoMap;

        bool IsDeviceTargetRegion = false;

        DR->getEntry()->getOperandBundlesAsDefs(OpBundles);
        // TODO: parse clauses.
        for (OperandBundleDef &O : OpBundles) {
          StringRef Tag = O.getTag();
          DEBUG_ENABLE(dbgs() << "OPB " << Tag << "\n");

          // TODO: check for conflicting DSA, for example reduction variables
          // cannot be set private. Should be done in Numba.
          if (Tag.startswith("DIR")) {
            auto It = StringToDir.find(Tag);
            assert(It != StringToDir.end() && "Directive is not supported!");
            Dir = It->second;
          } else if (Tag.startswith("QUAL")) {
            const ArrayRef<Value *> &TagInputs = O.inputs();
            if (Tag.startswith("QUAL.OMP.NORMALIZED.IV")) {
              assert(O.input_size() == 1 && "Expected single IV value");
              OMPLoopInfo.IV = TagInputs[0];
            } else if (Tag.startswith("QUAL.OMP.NORMALIZED.START")) {
              assert(O.input_size() == 1 && "Expected single START value");
              OMPLoopInfo.Start = TagInputs[0];
            } else if (Tag.startswith("QUAL.OMP.NORMALIZED.LB")) {
              assert(O.input_size() == 1 && "Expected single LB value");
              OMPLoopInfo.LB = TagInputs[0];
            } else if (Tag.startswith("QUAL.OMP.NORMALIZED.UB")) {
              assert(O.input_size() == 1 && "Expected single UB value");
              OMPLoopInfo.UB = TagInputs[0];
            } else if (Tag.startswith("QUAL.OMP.NUM_THREADS")) {
              assert(O.input_size() == 1 && "Expected single NumThreads value");
              ParRegionInfo.NumThreads = TagInputs[0];
            } else if (Tag.startswith("QUAL.OMP.SCHEDULE")) {
              // TODO: Add DIST_SCHEDULE for distribute loops.
              assert(O.input_size() == 1 &&
                     "Expected single chunking scheduling value");
              Constant *Zero = ConstantInt::get(TagInputs[0]->getType(), 0);
              OMPLoopInfo.Chunk = TagInputs[0];

              if (Tag == "QUAL.OMP.SCHEDULE.STATIC") {
                if (TagInputs[0] == Zero)
                  OMPLoopInfo.Sched = OMPScheduleType::Static;
                else {
                  OMPLoopInfo.Sched = OMPScheduleType::StaticChunked;
                  OMPLoopInfo.Chunk = TagInputs[0];
                }
              } else
                FATAL_ERROR("Unsupported scheduling type");
            } else if (Tag.startswith("QUAL.OMP.IF")) {
              assert(O.input_size() == 1 &&
                     "Expected single if condition value");
              ParRegionInfo.IfCondition = TagInputs[0];
            } else if (Tag.startswith("QUAL.OMP.TARGET.DEV_FUNC")) {
              assert(O.input_size() == 1 &&
                     "Expected a single device function name");
              ConstantDataArray *DevFuncArray =
                  dyn_cast<ConstantDataArray>(TagInputs[0]);
              assert(DevFuncArray &&
                     "Expected constant string for the device function");
              TargetInfo.DevFuncName = DevFuncArray->getAsString();
            } else if (Tag.startswith("QUAL.OMP.TARGET.ELF")) {
              assert(O.input_size() == 1 &&
                     "Expected a single elf image string");
              ConstantDataArray *ELF =
                  dyn_cast<ConstantDataArray>(TagInputs[0]);
              assert(ELF && "Expected constant string for ELF");
              TargetInfo.ELF = ELF;
            } else if (Tag.startswith("QUAL.OMP.DEVICE")) {
              // TODO: Handle device selection for target regions.
            } else if (Tag.startswith("QUAL.OMP.NUM_TEAMS")) {
              assert(O.input_size() == 1 && "Expected single NumTeams value");
              switch (Dir) {
              case OMPD_target:
                TargetInfo.NumTeams = TagInputs[0];
                break;
              case OMPD_teams:
              case OMPD_teams_distribute:
              case OMPD_teams_distribute_parallel_for:
                TeamsInfo.NumTeams = TagInputs[0];
                break;
              case OMPD_target_teams:
              case OMPD_target_teams_distribute:
                TargetInfo.NumTeams = TagInputs[0];
                TeamsInfo.NumTeams = TagInputs[0];
                break;
              case OMPD_target_teams_distribute_parallel_for:
                TargetInfo.NumTeams = TagInputs[0];
                TeamsInfo.NumTeams = TagInputs[0];
                break;
              default:
                FATAL_ERROR("Unsupported qualifier in directive");
              }
            } else if (Tag.startswith("QUAL.OMP.THREAD_LIMIT")) {
              assert(O.input_size() == 1 &&
                     "Expected single ThreadLimit value");
              switch (Dir) {
              case OMPD_target:
                TargetInfo.ThreadLimit = TagInputs[0];
                break;
              case OMPD_teams:
              case OMPD_teams_distribute:
              case OMPD_teams_distribute_parallel_for:
                TeamsInfo.ThreadLimit = TagInputs[0];
                break;
              case OMPD_target_teams:
              case OMPD_target_teams_distribute:
              case OMPD_target_teams_distribute_parallel_for:
                TargetInfo.ThreadLimit = TagInputs[0];
                TeamsInfo.ThreadLimit = TagInputs[0];
                break;
              default:
                FATAL_ERROR("Unsupported qualifier in directive");
              }
            } else if (Tag.startswith("QUAL.OMP.NOWAIT")) {
              switch (Dir) {
              case OMPD_target:
              case OMPD_target_teams:
              case OMPD_target_teams_distribute:
              case OMPD_target_teams_distribute_parallel_for:
                TargetInfo.NoWait = true;
                break;
              default:
                FATAL_ERROR("Unsupported nowait qualifier in directive");
              }
            } else /* DSA Qualifiers */ {
              auto It = StringToDSA.find(Tag);
              assert(It != StringToDSA.end() && "DSA type not found in map");
              if (It->second == DSA_MAP_ALLOC_STRUCT ||
                  It->second == DSA_MAP_TO_STRUCT ||
                  It->second == DSA_MAP_FROM_STRUCT ||
                  It->second == DSA_MAP_TOFROM_STRUCT) {
                assert((TagInputs.size() - 1) == 3 &&
                       "Expected input triple for struct mapping");
                Value *Index = TagInputs[1];
                Value *Offset = TagInputs[2];
                Value *NumElements = TagInputs[3];
                StructMappingInfoMap[TagInputs[0]].push_back(
                    {Index, Offset, NumElements, It->second});

                DSAValueMap[TagInputs[0]] = DSATypeInfo(DSA_MAP_STRUCT);
              } else {
                // This firstprivate includes a copy-constructor operand.
                if ((It->second == DSA_FIRSTPRIVATE ||
                     It->second == DSA_LASTPRIVATE) &&
                    TagInputs.size() == 2) {
                  Value *V = TagInputs[0];
                  ConstantDataArray *CopyFnNameArray =
                      dyn_cast<ConstantDataArray>(TagInputs[1]);
                  assert(CopyFnNameArray && "Expected constant string for the "
                                            "copy-constructor function");
                  StringRef CopyFnName = CopyFnNameArray->getAsString();
                  FunctionCallee CopyConstructor = M.getOrInsertFunction(
                      CopyFnName, V->getType()->getPointerElementType(),
                      V->getType()->getPointerElementType());
                  DSAValueMap[TagInputs[0]] =
                      DSATypeInfo(It->second, CopyConstructor);
                } else
                  // Sink for DSA qualifiers that do not require special
                  // handling.
                  DSAValueMap[TagInputs[0]] = DSATypeInfo(It->second);
              }
            }
          } else if (Tag == "OMP.DEVICE")
            IsDeviceTargetRegion = true;
          else
            FATAL_ERROR(("Unknown tag " + Tag).str().c_str());
        }

        assert(Dir != OMPD_unknown && "Expected valid OMP directive");

        // Gather info.
        BasicBlock *BBEntry = DR->getEntry()->getParent();
        Function *Fn = BBEntry->getParent();
        const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();

        // Create the basic block structure to isolate the outlined region.
        // Structure: BBEntry -> StartBB -> BBExit -> EndBB -> AfterBB
        // TODO: Reverse naming on BBExit and EndBB?
        BasicBlock *StartBB = SplitBlock(BBEntry, DR->getEntry());
        assert(BBEntry->getUniqueSuccessor() == StartBB &&
               "Expected unique successor at region start BB");

        BasicBlock *BBExit = DR->getExit()->getParent();
        BasicBlock *EndBB = SplitBlock(BBExit, DR->getExit()->getNextNode());
        assert(BBExit->getUniqueSuccessor() == EndBB &&
               "Expected unique successor at region end BB");
        BasicBlock *AfterBB = SplitBlock(EndBB, &*EndBB->getFirstInsertionPt());

        DEBUG_ENABLE(dbgs() << "BBEntry " << BBEntry->getName() << "\n");
        DEBUG_ENABLE(dbgs() << "StartBB " << StartBB->getName() << "\n");
        DEBUG_ENABLE(dbgs() << "BBExit " << BBExit->getName() << "\n");
        DEBUG_ENABLE(dbgs() << "EndBB " << EndBB->getName() << "\n");
        DEBUG_ENABLE(dbgs() << "AfterBB " << AfterBB->getName() << "\n");

        // Define the default BodyGenCB lambda.
        auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                             BasicBlock &ContinuationIP) {
          BasicBlock *CGStartBB = CodeGenIP.getBlock();
          BasicBlock *CGEndBB = SplitBlock(CGStartBB, &*CodeGenIP.getPoint());
          assert(StartBB != nullptr && "StartBB should not be null");
          CGStartBB->getTerminator()->setSuccessor(0, StartBB);
          assert(EndBB != nullptr && "EndBB should not be null");
          EndBB->getTerminator()->setSuccessor(0, CGEndBB);
        };

        // Define the default FiniCB lambda.
        auto FiniCB = [&](InsertPointTy CodeGenIP) {};

        // Remove intrinsics of OpenMP tags, first CBExit to also remove use
        // of CBEntry, then CBEntry.
        DR->getExit()->eraseFromParent();
        DR->getEntry()->eraseFromParent();

        if (Dir == OMPD_parallel) {
          CGIOMP.emitOMPParallel(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                                 EndBB, AfterBB, FiniCB, ParRegionInfo);
        } else if (Dir == OMPD_single) {
          CGIOMP.emitOMPSingle(Fn, BBEntry, AfterBB, BodyGenCB, FiniCB);
        } else if (Dir == OMPD_critical) {
          CGIOMP.emitOMPCritical(Fn, BBEntry, AfterBB, BodyGenCB, FiniCB);
        } else if (Dir == OMPD_barrier) {
          CGIOMP.emitOMPBarrier(Fn, BBEntry, OMPD_barrier);
        } else if (Dir == OMPD_for) {
          CGIOMP.emitOMPFor(DSAValueMap, OMPLoopInfo, StartBB, BBExit,
                            /* IsStandalone */ true, false);
        } else if (Dir == OMPD_parallel_for) {
          CGIOMP.emitOMPFor(DSAValueMap, OMPLoopInfo, StartBB, BBExit,
                            /* IsStandalone */ false, false);

          CGIOMP.emitOMPParallel(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                                 EndBB, AfterBB, FiniCB, ParRegionInfo);
        } else if (Dir == OMPD_task) {
          CGIOMP.emitOMPTask(DSAValueMap, Fn, BBEntry, StartBB, EndBB, AfterBB);
        } else if (Dir == OMPD_taskwait) {
          CGIOMP.emitOMPTaskwait(BBEntry);
        } else if (Dir == OMPD_target) {
          TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
          CGIOMP.emitOMPTarget(Fn, BBEntry, StartBB, EndBB, DSAValueMap,
                               StructMappingInfoMap, TargetInfo,
                               /* OMPLoopInfo */ nullptr, IsDeviceTargetRegion);
        } else if (Dir == OMPD_teams) {
          CGIOMP.emitOMPTeams(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                              EndBB, AfterBB, TeamsInfo);
        } else if (Dir == OMPD_distribute) {
          CGIOMP.emitOMPDistribute(DSAValueMap, OMPLoopInfo, StartBB, BBExit,
                                   /* IsStandalone */ true, false);
        } else if (Dir == OMPD_teams_distribute) {
          CGIOMP.emitOMPDistribute(DSAValueMap, OMPLoopInfo, StartBB, BBExit,
                                   /* IsStandalone */ false, false);
          CGIOMP.emitOMPTeams(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                              EndBB, AfterBB, TeamsInfo);
        } else if (Dir == OMPD_teams_distribute_parallel_for) {
          CGIOMP.emitOMPDistributeParallelFor(DSAValueMap, StartBB, BBExit,
                                              OMPLoopInfo, ParRegionInfo,
                                              /* IsStandalone */ false);

          CGIOMP.emitOMPTeams(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                              EndBB, AfterBB, TeamsInfo);
        } else if (Dir == OMPD_target_teams) {
          TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;

          // Construct teams info from target info.
          TeamsInfoStruct TeamsInfo;
          TeamsInfo.NumTeams = TargetInfo.NumTeams;
          TeamsInfo.ThreadLimit = TargetInfo.ThreadLimit;
          CGIOMP.emitOMPTargetTeams(DSAValueMap, nullptr, DL, Fn, BBEntry,
                                    StartBB, EndBB, AfterBB, TargetInfo,
                                    TeamsInfo,
                                    /* OMPLoopInfo */ nullptr,
                                    StructMappingInfoMap, IsDeviceTargetRegion);
        } else if (Dir == OMPD_target_data) {
          if (IsDeviceTargetRegion)
            FATAL_ERROR("Target enter data should never appear inside a "
                        "device target region");
          CGIOMP.emitOMPTargetData(Fn, BBEntry, BBExit, DSAValueMap,
                                   StructMappingInfoMap);
        } else if (Dir == OMPD_target_enter_data) {
          if (IsDeviceTargetRegion)
            FATAL_ERROR("Target enter data should never appear inside a "
                        "device target region");

          CGIOMP.emitOMPTargetEnterData(Fn, BBEntry, DSAValueMap,
                                        StructMappingInfoMap);
        } else if (Dir == OMPD_target_exit_data) {
          if (IsDeviceTargetRegion)
            FATAL_ERROR("Target exit data should never appear inside a "
                        "device target region");

          CGIOMP.emitOMPTargetExitData(Fn, BBEntry, DSAValueMap,
                                       StructMappingInfoMap);
        } else if (Dir == OMPD_target_update) {
          if (IsDeviceTargetRegion)
            FATAL_ERROR("Target exit data should never appear inside a "
                        "device target region");

          CGIOMP.emitOMPTargetUpdate(Fn, BBEntry, DSAValueMap,
                                     StructMappingInfoMap);
        } else if (Dir == OMPD_target_teams_distribute) {
          TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
          CGIOMP.emitOMPDistribute(DSAValueMap, OMPLoopInfo, StartBB, BBExit,
                                   /* IsStandalone */ false, false);
          TeamsInfoStruct TeamsInfo;
          TeamsInfo.NumTeams = TargetInfo.NumTeams;
          TeamsInfo.ThreadLimit = TargetInfo.ThreadLimit;
          CGIOMP.emitOMPTargetTeams(DSAValueMap, nullptr, DL, Fn, BBEntry,
                                    StartBB, EndBB, AfterBB, TargetInfo,
                                    TeamsInfo, &OMPLoopInfo,
                                    StructMappingInfoMap, IsDeviceTargetRegion);
        } else if (Dir == OMPD_distribute_parallel_for) {
          CGIOMP.emitOMPDistributeParallelFor(DSAValueMap, StartBB, BBExit,
                                              OMPLoopInfo, ParRegionInfo,
                                              /* isStandalone */ false);
        } else if (Dir == OMPD_target_teams_distribute_parallel_for) {
          CGIOMP.emitOMPDistributeParallelFor(DSAValueMap, StartBB, BBExit,
                                              OMPLoopInfo, ParRegionInfo,
                                              /* isStandalone */ false);

          TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD;
          //  Construct teams info from target info.
          TeamsInfoStruct TeamsInfo;
          TeamsInfo.NumTeams = TargetInfo.NumTeams;
          TeamsInfo.ThreadLimit = TargetInfo.ThreadLimit;
          TeamsInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD;
          CGIOMP.emitOMPTargetTeams(DSAValueMap, nullptr, DL, Fn, BBEntry,
                                    StartBB, EndBB, AfterBB, TargetInfo,
                                    TeamsInfo, &OMPLoopInfo,
                                    StructMappingInfoMap, IsDeviceTargetRegion);
        } else {
          FATAL_ERROR("Unknown directive");
        }

        if (verifyFunction(*Fn, &errs()))
          FATAL_ERROR("Verification of IntrinsicsOpenMP lowering failed!");
      }
    }

    DEBUG_ENABLE(dbgs() << "=== Dump Lowered Module\n"
                        << M << "=== End of Dump Lowered Module\n");

    DEBUG_ENABLE(dbgs() << "=== End of IntrinsicsOpenMP pass\n");

    return true;
  }
};
} // namespace

// Legacy PM registration.
struct LegacyIntrinsicsOpenmMPPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  LegacyIntrinsicsOpenmMPPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    IntrinsicsOpenMP IOMP;
    return IOMP.runOnModule(M);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
};

char LegacyIntrinsicsOpenmMPPass::ID = 0;
static RegisterPass<LegacyIntrinsicsOpenmMPPass>
    X("intrinsics-openmp", "Legacy IntrinsicsOpenMP Pass");

ModulePass *llvm::createIntrinsicsOpenMPPass() {
  return new LegacyIntrinsicsOpenmMPPass();
}

extern "C" __attribute__((visibility("default"))) void
LLVMAddIntrinsicsOpenMPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIntrinsicsOpenMPPass());
}

// New PM registration.

class IntrinsicsOpenMPPass : public PassInfoMixin<IntrinsicsOpenMPPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    IntrinsicsOpenMP IOMP;
    bool Changed = IOMP.runOnModule(M);

    if (Changed)
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
  }

  // Run always to lower OpenMP intrinsics.
  static bool isRequired() { return true; }
};

llvm::PassPluginLibraryInfo getIntrinsicsOpenMPPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "IntrinsicsOpenMP", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "intrinsics-openmp") {
                    MPM.addPass(IntrinsicsOpenMPPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getIntrinsicsOpenMPPluginInfo();
}

typedef void (*WriteCallback)(const void *data, size_t size);

extern "C" int runIntrinsicsOpenMPPass(const char *BitcodePtr,
                                       size_t BitcodeSize,
                                       WriteCallback WriteCB) {
  if (BitcodePtr == nullptr || BitcodeSize == 0 || WriteCB == nullptr) {
    errs() << "Invalid arguments to runIntrinsicsOpenMPPass\n";
    return 1;
  }

  MemoryBufferRef BufferRef{StringRef{BitcodePtr, BitcodeSize}, "module"};

  llvm::LLVMContext Ctx;
  auto ModOrErr = llvm::parseBitcodeFile(BufferRef, Ctx);
  if (!ModOrErr) {
    errs() << "Bitcode parse failed\n";
    return 2;
  }
  std::unique_ptr<llvm::Module> M = std::move(*ModOrErr);

  PassBuilder PB;

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  MPM.addPass(IntrinsicsOpenMPPass());
  MPM.run(*M, MAM);

  SmallVector<char, 0> Buf;
  raw_svector_ostream OS(Buf);
  WriteBitcodeToFile(*M, OS);

  WriteCB(Buf.data(), Buf.size());
  return 0;
}
