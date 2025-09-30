#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <llvm/Frontend/OpenMP/OMP.h.inc>
#include <llvm/Frontend/OpenMP/OMPIRBuilder.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <stdexcept>

#include "CGIntrinsicsOpenMP.h"
#include "DebugOpenMP.h"

#define DEBUG_TYPE "intrinsics-openmp"

using namespace llvm;
using namespace omp;
using namespace iomp;

namespace {

static CallInst *checkCreateCall(IRBuilderBase &Builder, FunctionCallee &Fn,
                                 ArrayRef<Value *> Args) {
  auto PrintDebugOutput = [&]() {
    dbgs() << "=== CGOpenMP checkCreateCall\n";
    dbgs() << "FunctionCallee: " << Fn.getCallee()->getName() << "\n";
    dbgs() << "FunctionCalee Type: " << *Fn.getFunctionType() << "\n";
    size_t ArgNo = 0;
    for (Value *Arg : Args) {
      dbgs() << "Arg " << ArgNo << ": " << *Arg << "\n";
      ArgNo++;
    }
    dbgs() << "=== End of CGOpenMP checkCreateCall\n";
  };
  DEBUG_ENABLE(PrintDebugOutput());

  // Check number of parameters only for non-vararg functions.
  if (!Fn.getFunctionType()->isVarArg())
    if (Args.size() != Fn.getFunctionType()->getNumParams()) {
      DEBUG_ENABLE(dbgs() << "Mismatch argument size " << Args.size() << " != "
                          << Fn.getFunctionType()->getNumParams() << "\n");
      return nullptr;
    }

  // Check argument types up to number params in the callee type to avoid
  // checking varargs unknow types.
  for (size_t I = 0; I < Fn.getFunctionType()->getNumParams(); ++I)
    if (Args[I]->getType() != Fn.getFunctionType()->getParamType(I)) {
      DEBUG_ENABLE(dbgs() << "Mismatch type at " << I << "\n";
                   dbgs() << "Arg " << *Args[I] << "\n";
                   dbgs() << "Expected type "
                          << *Fn.getFunctionType()->getParamType(I) << "\n";);
      return nullptr;
    }

  return Builder.CreateCall(Fn, Args);
}

} // namespace

InsertPointTy CGIntrinsicsOpenMP::emitReductionsHost(
    const OpenMPIRBuilder::LocationDescription &Loc, InsertPointTy AllocaIP,
    ArrayRef<OpenMPIRBuilder::ReductionInfo> ReductionInfos) {
  // If targeting the host runtime, use the OpenMP IR builder.
  return OMPBuilder.createReductions(Loc, AllocaIP, ReductionInfos);
}

InsertPointTy CGIntrinsicsOpenMP::emitReductionsDevice(
    const OpenMPIRBuilder::LocationDescription &Loc, InsertPointTy AllocaIP,
    ArrayRef<OpenMPIRBuilder::ReductionInfo> ReductionInfos, bool IsTeamSPMD) {
  // GPU reductions use atomics in a two-level hierarchy:
  // - Within each team, all threads update a shared-memory reduction variable.
  // - Across teams, only the team leaders update a global reduction variable in
  // global memory to work with the SPMD execution model.
  // TODO: optimize with warp reductions.

  BasicBlock *InsertBlock = Loc.IP.getBlock();
  BasicBlock *ReductionBlock =
      InsertBlock->splitBasicBlock(Loc.IP.getPoint(), "reduce");
  BasicBlock *ContinuationBlock = ReductionBlock->splitBasicBlock(
      ReductionBlock->begin(), "reduce.finalize");

  if (IsTeamSPMD) {
    // Make sure only team leads participate in the team reduction to correctly
    // execute in SPMD mode.
    InsertBlock->getTerminator()->eraseFromParent();
    OMPBuilder.Builder.SetInsertPoint(InsertBlock, InsertBlock->end());

    FunctionCallee GetHwThreadId = OMPBuilder.getOrCreateRuntimeFunction(
        M, llvm::omp::RuntimeFunction::
               OMPRTL___kmpc_get_hardware_thread_id_in_block);
    Value *ThreadId = OMPBuilder.Builder.CreateCall(GetHwThreadId, {});
    Value *Cond = OMPBuilder.Builder.CreateICmpEQ(
        ThreadId, OMPBuilder.Builder.getInt32(0));
    // Branch control flow to the reduction block for team leads, i.e., threads
    // with thread id 0 in the block, or to the continuation blocks for others.
    Value *Branch = OMPBuilder.Builder.CreateCondBr(Cond, ReductionBlock,
                                                    ContinuationBlock);
  }

  ReductionBlock->getTerminator()->eraseFromParent();
  OMPBuilder.Builder.SetInsertPoint(ReductionBlock, ReductionBlock->end());

  // Emit the reduction atomics.
  for (auto &RI : ReductionInfos) {
    assert(RI.Variable && "Expected non-null variable");
    assert(RI.PrivateVariable && "Expected non-null private variable");
    assert(RI.AtomicReductionGen &&
           "Expected non-null atomic reduction generator callback");
    assert(RI.Variable->getType() == RI.PrivateVariable->getType() &&
           "Expected variables and their private equivalents to have the same "
           "type");
    assert(RI.Variable->getType()->isPointerTy() &&
           "Expected variables to be pointers");

    OMPBuilder.Builder.restoreIP(
        RI.AtomicReductionGen(OMPBuilder.Builder.saveIP(), RI.ElementType,
                              RI.Variable, RI.PrivateVariable));
  }

  // Add terminator branch to the continuation block.
  OMPBuilder.Builder.CreateBr(ContinuationBlock);

  OMPBuilder.Builder.SetInsertPoint(ContinuationBlock);
  return OMPBuilder.Builder.saveIP();
}

void CGIntrinsicsOpenMP::setDeviceGlobalizedValues(
    const ArrayRef<Value *> GlobalizedValues) {
  DeviceGlobalizedValues.clear();
  DeviceGlobalizedValues.insert(GlobalizedValues.begin(),
                                GlobalizedValues.end());
}

Value *CGIntrinsicsOpenMP::createScalarCast(Value *V, Type *DestTy) {
  Value *Scalar = nullptr;
  assert(V && "Expected non-null value");
  if (V->getType()->isPointerTy()) {
    Value *Load =
        OMPBuilder.Builder.CreateLoad(V->getType()->getPointerElementType(), V);
    Scalar = OMPBuilder.Builder.CreateTruncOrBitCast(Load, DestTy);
  } else {
    Scalar = OMPBuilder.Builder.CreateTruncOrBitCast(V, DestTy);
  }

  return Scalar;
}

OutlinedInfoStruct CGIntrinsicsOpenMP::createOutlinedFunction(
    DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap, Function *OuterFn,
    BasicBlock *StartBB, BasicBlock *EndBB,
    SmallVectorImpl<Value *> &CapturedVars, StringRef Suffix,
    omp::Directive Kind) {
  SmallVector<Value *, 16> Privates;
  SmallVector<Value *, 16> CapturedShared;
  SmallVector<Value *, 16> CapturedFirstprivate;
  SmallVector<Value *, 16> Reductions;

  InsertPointTy SavedIP = OMPBuilder.Builder.saveIP();

  OpenMPIRBuilder::OutlineInfo OI;
  OI.EntryBB = StartBB;
  OI.ExitBB = EndBB;
  SmallPtrSet<BasicBlock *, 8> BlockSet;
  SmallVector<BasicBlock *, 8> BlockVector;
  OI.collectBlocks(BlockSet, BlockVector);

  CodeExtractorAnalysisCache CEAC(*OuterFn);
  CodeExtractor Extractor(BlockVector, /* DominatorTree */ nullptr,
                          /* AggregateArgs */ false,
                          /* BlockFrequencyInfo */ nullptr,
                          /* BranchProbabilityInfo */ nullptr,
                          /* AssumptionCache */ nullptr,
                          /* AllowVarArgs */ true,
                          /* AllowAlloca */ true,
                          /* Suffix */ ".");

  // Find inputs to, outputs from the code region.
  BasicBlock *CommonExit = nullptr;
  SetVector<Value *> Inputs, Outputs, SinkingCands, HoistingCands;
  Extractor.findAllocas(CEAC, SinkingCands, HoistingCands, CommonExit);
  Extractor.findInputsOutputs(Inputs, Outputs, SinkingCands);

  assert(Outputs.empty() && "Expected empty outputs from outlined region");
  assert(SinkingCands.empty() && "Expected empty alloca sinking candidates");

  auto IsTempOrDefaultPrivate = [](Value *V) {
    if (V->getName().startswith("."))
      return true;

    if (V->getName().startswith("excinfo"))
      return true;

    if (V->getName() == "quot")
      return true;

    if (V->getName() == "rem")
      return true;

    return false;
  };

  // Scan Inputs and define any missing values as Privates. Those values must
  // correspond to Numba-generated temporaries that should be privatized.
  for (auto *V : Inputs) {
    if (!DSAValueMap.count(V)) {
      DEBUG_ENABLE(dbgs() << "Missing V " << *V
                          << " from DSAValueMap, will privatize\n");
      if (!IsTempOrDefaultPrivate(V))
        FATAL_ERROR(
            "Expected Numba temporary value or default private, named starting "
            "with . but got " +
            V->getName().str());
      Privates.push_back(V);
      continue;
    }

    DSAType DSA = DSAValueMap[V].Type;

    DEBUG_ENABLE(dbgs() << "V " << *V << " from DSAValueMap Type " << DSA
                        << "\n");
    switch (DSA) {
    case DSA_PRIVATE:
      Privates.push_back(V);
      break;
    case DSA_FIRSTPRIVATE:
      CapturedFirstprivate.push_back(V);
      break;
    case DSA_SHARED:
    // Treat as shared to capture the pointer.
    case DSA_LASTPRIVATE:
    case DSA_MAP_TO:
    case DSA_MAP_FROM:
    case DSA_MAP_TOFROM:
    case DSA_MAP_STRUCT:
      CapturedShared.push_back(V);
      break;
    case DSA_REDUCTION_ADD:
    case DSA_REDUCTION_SUB:
    case DSA_REDUCTION_MUL:
      Reductions.push_back(V);
      break;
    default:
      FATAL_ERROR("Unexpected DSA type");
    }
  }

  SmallVector<Type *, 16> Params;
  // tid
  Params.push_back(OMPBuilder.Int32Ptr);
  // bound_tid
  Params.push_back(OMPBuilder.Int32Ptr);
  for (auto *V : CapturedShared)
    Params.push_back(V->getType());
  for (auto *V : CapturedFirstprivate) {
    Type *VPtrElemTy = V->getType()->getPointerElementType();
    if (VPtrElemTy->isSingleValueType())
      // TODO: The OpenMP runtime expects and propagates arguments
      // typed as Int64, thus we cast byval firstprivates to Int64. Using an
      // aggregate to store arguments would avoid this peculiarity.
      // Params.push_back(VPtrElemTy);
      Params.push_back(OMPBuilder.Int64);
    else
      Params.push_back(V->getType());
  }
  for (auto *V : Reductions)
    Params.push_back(V->getType());

  FunctionType *OutlinedFnTy =
      FunctionType::get(OMPBuilder.Void, Params, /* isVarArgs */ false);
  Function *OutlinedFn =
      Function::Create(OutlinedFnTy, GlobalValue::InternalLinkage,
                       OuterFn->getName() + Suffix, M);

  // Name the parameters and add attributes. Shared are ordered before
  // firstprivate in the parameter list.
  OutlinedFn->arg_begin()->setName("global_tid");
  std::next(OutlinedFn->arg_begin())->setName("bound_tid");
  Function::arg_iterator AI = std::next(OutlinedFn->arg_begin(), 2);
  int arg_no = 2;
  for (auto *V : CapturedShared) {
    AI->setName(V->getName() + ".shared");
    // Insert pointers in device globalized if they correspond to a device
    // globalized pointer.
    if (DeviceGlobalizedValues.contains(V))
      DeviceGlobalizedValues.insert(AI);

    OutlinedFn->addParamAttr(arg_no, Attribute::NonNull);
    OutlinedFn->addParamAttr(
        arg_no, Attribute::get(M.getContext(), Attribute::Dereferenceable, 8));
    ++AI;
    ++arg_no;
  }
  for (auto *V : CapturedFirstprivate) {
    Type *VPtrElemTy = V->getType()->getPointerElementType();
    if (VPtrElemTy->isSingleValueType()) {
      AI->setName(V->getName() + ".firstprivate.byval");
    } else {
      AI->setName(V->getName() + ".firstprivate");
      OutlinedFn->addParamAttr(arg_no, Attribute::NonNull);
      OutlinedFn->addParamAttr(
          arg_no,
          Attribute::get(M.getContext(), Attribute::Dereferenceable, 8));
    }
    ++AI;
    ++arg_no;
  }
  for (auto *V : Reductions) {
    AI->setName(V->getName() + ".red");
    OutlinedFn->addParamAttr(arg_no, Attribute::NonNull);
    OutlinedFn->addParamAttr(
        arg_no, Attribute::get(M.getContext(), Attribute::Dereferenceable, 8));
    ++AI;
    ++arg_no;
  }

  BasicBlock *OutlinedEntryBB =
      BasicBlock::Create(M.getContext(), ".outlined.entry", OutlinedFn);
  BasicBlock *OutlinedExitBB =
      BasicBlock::Create(M.getContext(), ".outlined.exit", OutlinedFn);

  auto CreateAllocaAtEntry = [&](Type *Ty, Value *ArraySize = nullptr,
                                 const Twine &Name = "") {
    auto CurIP = OMPBuilder.Builder.saveIP();
    OMPBuilder.Builder.SetInsertPoint(OutlinedEntryBB,
                                      OutlinedEntryBB->getFirstInsertionPt());
    Value *Alloca = OMPBuilder.Builder.CreateAlloca(Ty, nullptr, Name);
    OMPBuilder.Builder.restoreIP(CurIP);
    return Alloca;
  };

  OMPBuilder.Builder.SetInsertPoint(OutlinedEntryBB);

  OutlinedFn->addParamAttr(0, Attribute::NoAlias);
  OutlinedFn->addParamAttr(1, Attribute::NoAlias);
  OutlinedFn->addFnAttr(Attribute::NoUnwind);
  OutlinedFn->addFnAttr(Attribute::NoRecurse);

  auto CollectUses = [&BlockSet](Value *V, SetVector<Use *> &Uses) {
    for (Use &U : V->uses())
      if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
        if (BlockSet.count(UserI->getParent()))
          Uses.insert(&U);
  };

  auto ReplaceUses = [](SetVector<Use *> &Uses, Value *ReplacementValue) {
    for (Use *UPtr : Uses)
      UPtr->set(ReplacementValue);
  };

  for (auto *V : Privates) {
    SetVector<Use *> Uses;
    CollectUses(V, Uses);

    Type *VTy = V->getType()->getPointerElementType();
    Value *ReplacementValue =
        CreateAllocaAtEntry(VTy, nullptr, V->getName() + ".private");
    // NOTE: We need to zero initialize privates because Numba reference
    // counting breaks when those privates correspond to memory-managed
    // data structures.
    OMPBuilder.Builder.CreateStore(Constant::getNullValue(VTy),
                                   ReplacementValue);

    if (VMap)
      (*VMap)[V] = ReplacementValue;

    ReplaceUses(Uses, ReplacementValue);
  }

  AI = std::next(OutlinedFn->arg_begin(), 2);
  for (auto *V : CapturedShared) {
    SetVector<Use *> Uses;
    CollectUses(V, Uses);

    Value *ReplacementValue = AI;

    if (VMap)
      (*VMap)[V] = ReplacementValue;

    ReplaceUses(Uses, ReplacementValue);
    ++AI;
  }

  for (auto *V : CapturedFirstprivate) {
    SetVector<Use *> Uses;
    CollectUses(V, Uses);

    Type *VPtrElemTy = V->getType()->getPointerElementType();
    Value *ReplacementValue =
        CreateAllocaAtEntry(VPtrElemTy, nullptr, V->getName() + ".copy");
    if (VPtrElemTy->isSingleValueType()) {
      // TODO: The OpenMP runtime expects and propagates arguments
      // typed as Int64, thus we cast byval firstprivates to Int64. Using an
      // aggregate to store arguments would avoid this peculiarity.
      // OMPBuilder.Builder.CreateStore(AI, ReplacementValue);
      Value *Alloca = CreateAllocaAtEntry(OMPBuilder.Int64);

      OMPBuilder.Builder.CreateStore(AI, Alloca);
      Value *BitCast = OMPBuilder.Builder.CreateBitCast(Alloca, V->getType());
      Value *Load = OMPBuilder.Builder.CreateLoad(VPtrElemTy, BitCast);
      OMPBuilder.Builder.CreateStore(Load, ReplacementValue);
    } else {
      Value *Load = OMPBuilder.Builder.CreateLoad(VPtrElemTy, AI,
                                                  V->getName() + ".reload");
      FunctionCallee CopyConstructor = DSAValueMap[V].CopyConstructor;
      if (CopyConstructor) {
        Value *Copy = OMPBuilder.Builder.CreateCall(CopyConstructor, {Load});
        OMPBuilder.Builder.CreateStore(Copy, ReplacementValue);
      } else
        OMPBuilder.Builder.CreateStore(Load, ReplacementValue);
    }

    if (VMap)
      (*VMap)[V] = ReplacementValue;

    ReplaceUses(Uses, ReplacementValue);

    ++AI;
  }

  SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;
  for (auto *V : Reductions) {
    SetVector<Use *> Uses;
    CollectUses(V, Uses);

    Value *ReplacementValue = nullptr;

    // Privatize the reduction variable and initialize it.
    InsertPointTy AllocaIP(OutlinedEntryBB,
                           OutlinedEntryBB->getFirstInsertionPt());

    // Detect a GPU team reduction to configure emitting the private reduction
    // variable.
    bool IsGPUTeamsReduction =
        ((Kind == omp::Directive::OMPD_teams) && isOpenMPDeviceRuntime());

    Value *Priv = nullptr;
    switch (DSAValueMap[V].Type) {
    case DSA_REDUCTION_ADD:
      Priv = CGReduction::emitInitAndAppendInfo<DSA_REDUCTION_ADD>(
          OMPBuilder.Builder, AllocaIP, AI, ReductionInfos,
          IsGPUTeamsReduction);
      break;
    case DSA_REDUCTION_SUB:
      Priv = CGReduction::emitInitAndAppendInfo<DSA_REDUCTION_SUB>(
          OMPBuilder.Builder, AllocaIP, AI, ReductionInfos,
          IsGPUTeamsReduction);
      break;
    case DSA_REDUCTION_MUL:
      Priv = CGReduction::emitInitAndAppendInfo<DSA_REDUCTION_MUL>(
          OMPBuilder.Builder, AllocaIP, AI, ReductionInfos,
          IsGPUTeamsReduction);
      break;
    default:
      FATAL_ERROR("Unsupported reduction");
    }

    assert(Priv && "Expected non-null private reduction variable");
    ReplacementValue = Priv;

    assert(ReplacementValue && "Expected non-null replacement value");
    if (VMap)
      (*VMap)[V] = ReplacementValue;

    ReplaceUses(Uses, ReplacementValue);

    ++AI;
  }

  OMPBuilder.Builder.CreateBr(StartBB);

  EndBB->getTerminator()->setSuccessor(0, OutlinedExitBB);
  OMPBuilder.Builder.SetInsertPoint(OutlinedExitBB);
  OMPBuilder.Builder.CreateRetVoid();

  // Deterministic insertion of BBs, BlockVector needs ExitBB to move to the
  // outlined function.
  BlockVector.push_back(OI.ExitBB);
  for (auto *BB : BlockVector)
    BB->moveBefore(OutlinedExitBB);

  DEBUG_ENABLE(dbgs() << "=== Dump OutlinedFn\n"
                      << *OutlinedFn << "=== End of Dump OutlinedFn\n");

  if (verifyFunction(*OutlinedFn, &errs()))
    FATAL_ERROR("Verification of OutlinedFn failed!");

  CapturedVars.append(CapturedShared);
  CapturedVars.append(CapturedFirstprivate);
  CapturedVars.append(Reductions);

  if (SavedIP.isSet())
    OMPBuilder.Builder.restoreIP(SavedIP);

  return OutlinedInfoStruct{OutlinedFn, OutlinedEntryBB, OutlinedExitBB,
                            ReductionInfos};
}

CGIntrinsicsOpenMP::CGIntrinsicsOpenMP(Module &M) : OMPBuilder(M), M(M) {
  OMPBuilder.initialize();

  TgtOffloadEntryTy = StructType::create({OMPBuilder.Int8Ptr,
                                          OMPBuilder.Int8Ptr, OMPBuilder.SizeTy,
                                          OMPBuilder.Int32, OMPBuilder.Int32},
                                         "struct.__tgt_offload_entry");
  // OpenMP device runtime expects this global that controls debugging, default
  // to 0 (no debugging enabled).
  if (isOpenMPDeviceRuntime())
    OMPBuilder.createGlobalFlag(0, "__omp_rtl_debug_kind");
}

void CGIntrinsicsOpenMP::emitOMPParallel(
    DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap, const DebugLoc &DL,
    Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
    BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
    ParRegionInfoStruct &ParRegionInfo) {
  if (isOpenMPDeviceRuntime())
    emitOMPParallelDeviceRuntime(DSAValueMap, VMap, DL, Fn, BBEntry, StartBB,
                                 EndBB, AfterBB, FiniCB, ParRegionInfo);
  else
    emitOMPParallelHostRuntime(DSAValueMap, VMap, DL, Fn, BBEntry, StartBB,
                               EndBB, AfterBB, FiniCB, ParRegionInfo);
}

void CGIntrinsicsOpenMP::emitOMPParallelHostRuntime(
    DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap, const DebugLoc &DL,
    Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
    BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
    ParRegionInfoStruct &ParRegionInfo) {

  SmallVector<Value *, 16> CapturedVars;
  OutlinedInfoStruct OI = createOutlinedFunction(
      DSAValueMap, VMap, Fn, StartBB, EndBB, CapturedVars,
      ".omp_outlined_parallel", omp::Directive::OMPD_parallel);

  if (!OI.ReductionInfos.empty())
    emitReductionsHost(InsertPointTy(OI.ExitBB, OI.ExitBB->begin()),
                       InsertPointTy(OI.EntryBB, OI.EntryBB->begin()),
                       OI.ReductionInfos);

  Function *OutlinedFn = OI.Fn;

  // Set the insertion location at the end of the BBEntry.
  BBEntry->getTerminator()->eraseFromParent();
  OMPBuilder.Builder.SetInsertPoint(BBEntry);
  OMPBuilder.Builder.CreateBr(AfterBB);

  OMPBuilder.Builder.SetInsertPoint(BBEntry->getTerminator());
  OpenMPIRBuilder::LocationDescription Loc(OMPBuilder.Builder.saveIP(), DL);
  OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

  auto EmitForkCall = [&](InsertPointTy InsertIP) {
    OMPBuilder.Builder.restoreIP(InsertIP);

    auto *OutlinedFnCast = OMPBuilder.Builder.CreateBitCast(
        OutlinedFn, OMPBuilder.ParallelTaskPtr);
    FunctionCallee ForkCall =
        OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_fork_call);
    SmallVector<Value *, 16> ForkArgs;
    ForkArgs.append({Ident, OMPBuilder.Builder.getInt32(CapturedVars.size()),
                     OutlinedFnCast});

    for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
      // Pass firstprivate scalar by value.
      if (DSAValueMap[CapturedVars[Idx]].Type == DSA_FIRSTPRIVATE &&
          CapturedVars[Idx]
              ->getType()
              ->getPointerElementType()
              ->isSingleValueType()) {
        // TODO: check type conversions.
        Value *Alloca = OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int64);
        Type *VPtrElemTy =
            CapturedVars[Idx]->getType()->getPointerElementType();
        Value *LoadV =
            OMPBuilder.Builder.CreateLoad(VPtrElemTy, CapturedVars[Idx]);
        Value *BitCast = OMPBuilder.Builder.CreateBitCast(
            Alloca, CapturedVars[Idx]->getType());
        OMPBuilder.Builder.CreateStore(LoadV, BitCast);
        Value *Load = OMPBuilder.Builder.CreateLoad(OMPBuilder.Int64, Alloca);
        ForkArgs.push_back(Load);
        continue;
      }

      ForkArgs.push_back(CapturedVars[Idx]);
    }

    OMPBuilder.Builder.CreateCall(ForkCall, ForkArgs);
  };

  auto EmitSerializedParallel = [&](InsertPointTy InsertIP) {
    OMPBuilder.Builder.restoreIP(InsertIP);

    // Build calls __kmpc_serialized_parallel(&Ident, GTid);
    Value *Args[] = {Ident, ThreadID};
    OMPBuilder.Builder.CreateCall(OMPBuilder.getOrCreateRuntimeFunctionPtr(
                                      OMPRTL___kmpc_serialized_parallel),
                                  Args);

    Value *ZeroAddr = OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr,
                                                      ".zero.addr");
    OMPBuilder.Builder.CreateStore(Constant::getNullValue(OMPBuilder.Int32),
                                   ZeroAddr);
    // Zero for thread id, bound tid.
    SmallVector<Value *, 16> OutlinedArgs = {ZeroAddr, ZeroAddr};
    for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
      // Pass firstprivate scalar by value.
      if (DSAValueMap[CapturedVars[Idx]].Type == DSA_FIRSTPRIVATE &&
          CapturedVars[Idx]
              ->getType()
              ->getPointerElementType()
              ->isSingleValueType()) {
        // TODO: check type conversions.
        Type *VPtrElemTy =
            CapturedVars[Idx]->getType()->getPointerElementType();
        Value *Load =
            OMPBuilder.Builder.CreateLoad(VPtrElemTy, CapturedVars[Idx]);
        OutlinedArgs.push_back(Load);
        continue;
      }

      OutlinedArgs.push_back(CapturedVars[Idx]);
    }

    OMPBuilder.Builder.CreateCall(OutlinedFn, OutlinedArgs);

    // __kmpc_end_serialized_parallel(&Ident, GTid);
    OMPBuilder.Builder.CreateCall(OMPBuilder.getOrCreateRuntimeFunctionPtr(
                                      OMPRTL___kmpc_end_serialized_parallel),
                                  Args);
  };

  if (ParRegionInfo.NumThreads) {
    Value *NumThreads =
        createScalarCast(ParRegionInfo.NumThreads, OMPBuilder.Int32);
    assert(NumThreads && "Expected non-null num threads");
    Value *Args[] = {Ident, ThreadID, NumThreads};
    OMPBuilder.Builder.CreateCall(OMPBuilder.getOrCreateRuntimeFunctionPtr(
                                      OMPRTL___kmpc_push_num_threads),
                                  Args);
  }

  if (ParRegionInfo.IfCondition) {
    Instruction *ThenTI = nullptr, *ElseTI = nullptr;
    Value *IfConditionEval = nullptr;

    if (ParRegionInfo.IfCondition->getType()->isFloatingPointTy())
      IfConditionEval = OMPBuilder.Builder.CreateFCmpUNE(
          ParRegionInfo.IfCondition,
          ConstantFP::get(ParRegionInfo.IfCondition->getType(), 0));
    else
      IfConditionEval = OMPBuilder.Builder.CreateICmpNE(
          ParRegionInfo.IfCondition,
          ConstantInt::get(ParRegionInfo.IfCondition->getType(), 0));

    assert(IfConditionEval && "Expected non-null condition");
    SplitBlockAndInsertIfThenElse(IfConditionEval, BBEntry->getTerminator(),
                                  &ThenTI, &ElseTI);

    assert(ThenTI && "Expected non-null ThenTI");
    assert(ElseTI && "Expected non-null ElseTI");
    EmitForkCall(InsertPointTy(ThenTI->getParent(), ThenTI->getIterator()));
    EmitSerializedParallel(
        InsertPointTy(ElseTI->getParent(), ElseTI->getIterator()));
  } else {
    EmitForkCall(
        InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()));
  }

  DEBUG_ENABLE(dbgs() << "=== Dump OuterFn\n"
                      << *Fn << "=== End of Dump OuterFn\n");

  if (verifyFunction(*Fn, &errs()))
    FATAL_ERROR("Verification of OuterFn failed!");
}

void CGIntrinsicsOpenMP::emitOMPParallelDeviceRuntime(
    DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap, const DebugLoc &DL,
    Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
    BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
    ParRegionInfoStruct &ParRegionInfo) {
  // Extract parallel region
  SmallVector<Value *, 16> CapturedVars;
  OutlinedInfoStruct OI = createOutlinedFunction(
      DSAValueMap, VMap, Fn, StartBB, EndBB, CapturedVars,
      ".omp_outlined_parallel", omp::Directive::OMPD_parallel);

  if (!OI.ReductionInfos.empty())
    emitReductionsDevice(InsertPointTy(OI.ExitBB, OI.ExitBB->begin()),
                         InsertPointTy(OI.EntryBB, OI.EntryBB->begin()),
                         OI.ReductionInfos, false);

  Function *OutlinedFn = OI.Fn;

  // Create wrapper for worker threads
  SmallVector<Type *, 2> Params;
  // parallelism level, unused?
  Params.push_back(OMPBuilder.Int16);
  // tid
  Params.push_back(OMPBuilder.Int32);

  FunctionType *OutlinedWrapperFnTy =
      FunctionType::get(OMPBuilder.Void, Params, /* isVarArgs */ false);
  Function *OutlinedWrapperFn =
      Function::Create(OutlinedWrapperFnTy, GlobalValue::InternalLinkage,
                       OutlinedFn->getName() + ".wrapper", M);
  BasicBlock *OutlinedWrapperEntryBB =
      BasicBlock::Create(M.getContext(), "entry", OutlinedWrapperFn);

  // Code generation for the outlined wrapper function.
  OMPBuilder.Builder.SetInsertPoint(OutlinedWrapperEntryBB);

  constexpr const int TIDArgNo = 1;
  AllocaInst *TIDAddr =
      OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, ".tid.addr");
  AllocaInst *ZeroAddr =
      OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, ".zero.addr");
  AllocaInst *GlobalArgs = OMPBuilder.Builder.CreateAlloca(
      OMPBuilder.Int8PtrPtr, nullptr, "global_args");

  OMPBuilder.Builder.CreateStore(OutlinedWrapperFn->getArg(TIDArgNo), TIDAddr);
  OMPBuilder.Builder.CreateStore(Constant::getNullValue(OMPBuilder.Int32),
                                 ZeroAddr);
  FunctionCallee KmpcGetSharedVariables = OMPBuilder.getOrCreateRuntimeFunction(
      M, OMPRTL___kmpc_get_shared_variables);
  OMPBuilder.Builder.CreateCall(KmpcGetSharedVariables, {GlobalArgs});

  SmallVector<Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(TIDAddr);
  OutlinedFnArgs.push_back(ZeroAddr);

  for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
    Value *LoadGlobalArgs =
        OMPBuilder.Builder.CreateLoad(OMPBuilder.Int8PtrPtr, GlobalArgs);
    Value *GEP = OMPBuilder.Builder.CreateConstInBoundsGEP1_64(
        OMPBuilder.Int8Ptr, LoadGlobalArgs, Idx);

    // Pass firstprivate scalar by value.
    if (DSAValueMap[CapturedVars[Idx]].Type == DSA_FIRSTPRIVATE &&
        CapturedVars[Idx]
            ->getType()
            ->getPointerElementType()
            ->isSingleValueType()) {
      Type *VPtrElemTy = CapturedVars[Idx]->getType()->getPointerElementType();
      Value *Bitcast =
          OMPBuilder.Builder.CreateBitCast(GEP, CapturedVars[Idx]->getType());
      Value *Load = OMPBuilder.Builder.CreateLoad(VPtrElemTy, Bitcast);
      // TODO: Runtime expects values in Int64 type, fix with arguments in
      // struct.
      AllocaInst *TmpInt64 = OMPBuilder.Builder.CreateAlloca(
          OMPBuilder.Int64, nullptr,
          CapturedVars[Idx]->getName() + "fpriv.byval");
      Value *Cast = OMPBuilder.Builder.CreateBitCast(
          TmpInt64, CapturedVars[Idx]->getType());
      OMPBuilder.Builder.CreateStore(Load, Cast);
      Value *ConvLoad =
          OMPBuilder.Builder.CreateLoad(OMPBuilder.Int64, TmpInt64);
      OutlinedFnArgs.push_back(ConvLoad);

      continue;
    }

    Value *Bitcast = OMPBuilder.Builder.CreateBitCast(
        GEP, CapturedVars[Idx]->getType()->getPointerTo());
    Value *Load =
        OMPBuilder.Builder.CreateLoad(CapturedVars[Idx]->getType(), Bitcast);
    OutlinedFnArgs.push_back(Load);
  }

  FunctionCallee OutlinedFnCallee(OutlinedFn->getFunctionType(), OutlinedFn);

  auto *OutlinedCI =
      checkCreateCall(OMPBuilder.Builder, OutlinedFnCallee, OutlinedFnArgs);
  assert(OutlinedCI && "Expected valid call");
  OMPBuilder.Builder.CreateRetVoid();

  if (verifyFunction(*OutlinedWrapperFn, &errs()))
    FATAL_ERROR("Verification of OutlinedWrapperFn failed!");

  DEBUG_ENABLE(dbgs() << "=== Dump OutlinedWrapper\n"
                      << *OutlinedWrapperFn
                      << "=== End of Dump OutlinedWrapper\n");

  // Setup the call to kmpc_parallel_51
  BBEntry->getTerminator()->eraseFromParent();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  OMPBuilder.Builder.restoreIP(Loc.IP);
  OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);

  // Create the address table of the global data.
  // The number of outlined arguments without global_tid, bound_tid.
  Value *NumCapturedArgs =
      ConstantInt::get(OMPBuilder.SizeTy, CapturedVars.size());
  Type *CapturedVarsAddrsTy =
      ArrayType::get(OMPBuilder.Int8Ptr, CapturedVars.size());

  // TODO: Re-think allocas, move to start of caller. If the caller is outlined
  // in an outer OpenMP region, dot naming ensures captured_var_addrs is a
  // private value, since it's only used for setting up the call to
  // kmpc_parallel_51.
  auto PrevIP = OMPBuilder.Builder.saveIP();
  InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                         Fn->getEntryBlock().getFirstInsertionPt());
  OMPBuilder.Builder.restoreIP(AllocaIP);
  Value *CapturedVarsAddrs = OMPBuilder.Builder.CreateAlloca(
      CapturedVarsAddrsTy, nullptr, ".captured_var_addrs");
  OMPBuilder.Builder.restoreIP(PrevIP);

  SmallVector<Value *> GlobalAllocas;
  for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
    DEBUG_ENABLE(dbgs() << "CapturedVar " << Idx << " " << *CapturedVars[Idx]
                        << "\n");
    Value *GEP = OMPBuilder.Builder.CreateConstInBoundsGEP2_64(
        CapturedVarsAddrsTy, CapturedVarsAddrs, 0, Idx);

    // Pass firstprivate scalar by value.
    if (DSAValueMap[CapturedVars[Idx]].Type == DSA_FIRSTPRIVATE &&
        CapturedVars[Idx]
            ->getType()
            ->getPointerElementType()
            ->isSingleValueType()) {
      // TODO: check type conversions.
      Value *BitCast = OMPBuilder.Builder.CreateBitCast(CapturedVars[Idx],
                                                        OMPBuilder.Int64Ptr);
      Value *Load = OMPBuilder.Builder.CreateLoad(OMPBuilder.Int64, BitCast);
      Value *IntToPtr =
          OMPBuilder.Builder.CreateIntToPtr(Load, OMPBuilder.Int8Ptr);
      OMPBuilder.Builder.CreateStore(IntToPtr, GEP);

      continue;
    }

    // Allocate from global memory if the pointer is not globalized (not in the
    // global address space).
    FunctionCallee KmpcAllocShared =
        OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_alloc_shared);
    assert(CapturedVars[Idx]->getType()->isPointerTy() &&
           "Expected pointer type");

    if (DeviceGlobalizedValues.contains(CapturedVars[Idx])) {
      Value *Bitcast = OMPBuilder.Builder.CreateBitCast(CapturedVars[Idx],
                                                        OMPBuilder.Int8Ptr);
      OMPBuilder.Builder.CreateStore(Bitcast, GEP);
    } else {
      Type *AllocTy = CapturedVars[Idx]->getType()->getPointerElementType();
      Value *Size = ConstantInt::get(
          OMPBuilder.SizeTy, M.getDataLayout().getTypeAllocSize(AllocTy));
      CallBase *GlobalAlloc =
          OMPBuilder.Builder.CreateCall(KmpcAllocShared, {Size});
      GlobalAlloc->addRetAttr(
          llvm::Attribute::get(M.getContext(), llvm::Attribute::Alignment, 16));
      GlobalAllocas.push_back(GlobalAlloc);
      // TODO: this assumes the type is trivally copyable, use the copy
      // constructor for more complex types.
      OMPBuilder.Builder.CreateMemCpy(
          GlobalAlloc, GlobalAlloc->getPointerAlignment(M.getDataLayout()),
          CapturedVars[Idx],
          CapturedVars[Idx]->getPointerAlignment(M.getDataLayout()), Size);

      OMPBuilder.Builder.CreateStore(GlobalAlloc, GEP);
    }
  }

  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

  Value *IfCondition = ParRegionInfo.IfCondition;
  Value *NumThreads = ParRegionInfo.NumThreads;
  if (!IfCondition)
    // Set condition to 1 (execute in parallel) if not set.
    IfCondition = ConstantInt::get(OMPBuilder.Int32, 1);

  if (!NumThreads)
    NumThreads = ConstantInt::get(OMPBuilder.Int32, -1);
  else
    NumThreads =
        OMPBuilder.Builder.CreateTruncOrBitCast(NumThreads, OMPBuilder.Int32);

  assert(NumThreads && "Expected non-null NumThreads");

  FunctionCallee KmpcParallel51 =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_parallel_51);

  // Set proc_bind to -1 by default as it is unused.
  assert(Ident && "Expected non-null Ident");
  assert(ThreadID && "Expected non-null ThreadID");
  assert(IfCondition && "Expected non-null IfCondition");
  assert(NumThreads && "Expected non-null NumThreads");
  assert(OutlinedWrapperFn && "Expected non-null OutlinedWrapperFn");
  assert(CapturedVarsAddrs && "Expected non-null CapturedVarsAddrs");
  assert(NumCapturedArgs && "Expected non-null NumCapturedArgs");

  Value *ProcBind = OMPBuilder.Builder.getInt32(-1);
  Value *OutlinedFnBitcast =
      OMPBuilder.Builder.CreateBitCast(OutlinedFn, OMPBuilder.VoidPtr);
  Value *OutlinedWrapperFnBitcast =
      OMPBuilder.Builder.CreateBitCast(OutlinedWrapperFn, OMPBuilder.VoidPtr);
  Value *CapturedVarAddrsBitcast = OMPBuilder.Builder.CreateBitCast(
      CapturedVarsAddrs, OMPBuilder.VoidPtrPtr);

  SmallVector<Value *, 10> Args = {Ident,
                                   ThreadID,
                                   IfCondition,
                                   NumThreads,
                                   ProcBind,
                                   OutlinedFnBitcast,
                                   OutlinedWrapperFnBitcast,
                                   CapturedVarAddrsBitcast,
                                   NumCapturedArgs};

  auto *CallKmpcParallel51 =
      checkCreateCall(OMPBuilder.Builder, KmpcParallel51, Args);
  assert(CallKmpcParallel51 &&
         "Expected non-null call instr from code generation");

  FunctionCallee KmpcFreeShared =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_free_shared);
  for (Value *GA : GlobalAllocas) {
    Type *AllocTy = GA->getType()->getPointerElementType();
    Value *Size = ConstantInt::get(OMPBuilder.SizeTy,
                                   M.getDataLayout().getTypeAllocSize(AllocTy));
    auto *CI = checkCreateCall(OMPBuilder.Builder, KmpcFreeShared, {GA, Size});
    assert(CI && "Expected valid call");
  }

  OMPBuilder.Builder.CreateBr(AfterBB);

  DEBUG_ENABLE(dbgs() << "=== Dump OuterFn\n"
                      << *Fn << "=== End of Dump OuterFn\n");

  if (verifyFunction(*Fn, &errs()))
    FATAL_ERROR("Verification of OuterFn failed!");
}

FunctionCallee CGIntrinsicsOpenMP::getKmpcForStaticInit(Type *Ty) {
  DEBUG_ENABLE(dbgs() << "Type " << *Ty << "\n");
  unsigned Bitwidth = Ty->getIntegerBitWidth();
  DEBUG_ENABLE(dbgs() << "Bitwidth " << Bitwidth << "\n");
  if (Bitwidth == 32)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___kmpc_for_static_init_4u);
  if (Bitwidth == 64)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___kmpc_for_static_init_8u);

  FATAL_ERROR("unknown OpenMP loop iterator bitwidth");
}

FunctionCallee CGIntrinsicsOpenMP::getKmpcDistributeStaticInit(Type *Ty) {
  DEBUG_ENABLE(dbgs() << "Type " << *Ty << "\n");
  unsigned Bitwidth = Ty->getIntegerBitWidth();
  DEBUG_ENABLE(dbgs() << "Bitwidth " << Bitwidth << "\n");
  if (Bitwidth == 32)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___kmpc_distribute_static_init_4u);
  if (Bitwidth == 64)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___kmpc_distribute_static_init_8u);

  FATAL_ERROR("unknown OpenMP loop iterator bitwidth");
}

void CGIntrinsicsOpenMP::emitLoop(DSAValueMapTy &DSAValueMap,
                                  OMPLoopInfoStruct &OMPLoopInfo,
                                  BasicBlock *StartBB, BasicBlock *ExitBB,
                                  bool IsStandalone, bool IsDistribute,
                                  bool IsDistributeParallelFor,
                                  OMPDistributeInfoStruct *OMPDistributeInfo) {
  DEBUG_ENABLE(dbgs() << "OMPLoopInfo.IV " << *OMPLoopInfo.IV << "\n");
  DEBUG_ENABLE(dbgs() << "OMPLoopInfo.UB " << *OMPLoopInfo.UB << "\n");
  assert(OMPLoopInfo.IV && "Expected non-null IV");
  assert(OMPLoopInfo.UB && "Expected non-null UB");

  assert(static_cast<int>(OMPLoopInfo.Sched) &&
         "Expected non-zero loop schedule");

  BasicBlock *PreHeader = StartBB;
  PreHeader->setName("omp.for.preheader");
  BasicBlock *Header = PreHeader->getUniqueSuccessor();
  assert(Header && "Expected unique successor header");
  Header->setName("omp.for.cond");
  BasicBlock *Exit = ExitBB;
  Exit->setName("omp.for.exit");
  assert(Header && "Expected unique successor from PreHeader to Header");
  DEBUG_ENABLE(dbgs() << "=== PreHeader\n"
                      << *PreHeader << "=== End of PreHeader\n");
  DEBUG_ENABLE(dbgs() << "=== Header\n" << *Header << "=== End of Header\n");
  assert(Header->getTerminator()->getNumSuccessors() == 2 &&
         "Expected 2 successors (loopbody, exit)");
  BasicBlock *HeaderSuccBBs[2] = {Header->getTerminator()->getSuccessor(0),
                                  Header->getTerminator()->getSuccessor(1)};
  BasicBlock *LoopBody =
      (HeaderSuccBBs[0] == Exit ? HeaderSuccBBs[1] : HeaderSuccBBs[0]);
  assert(LoopBody && "Expected non-null loop body basic block\n");

  assert(Header->hasNPredecessors(2) &&
         "Expected exactly 2 predecessors to loop header (preheader, latch)");
  BasicBlock *HeaderPredBBs[2] = {*predecessors(Header).begin(),
                                  *std::next(predecessors(Header).begin(), 1)};
  BasicBlock *Latch =
      (HeaderPredBBs[0] == PreHeader ? HeaderPredBBs[1] : HeaderPredBBs[0]);
  Latch->setName("omp.for.inc");
  assert(Latch && "Expected latch basicblock");

  auto ClearBlockInstructions = [](BasicBlock *BB) {
    // Remove all instructions in the BB, iterate backwards to avoid
    // dangling uses for safe deletion. The BB becomes malformed and
    // requires a terminator added.
    while (!BB->empty()) {
      Instruction &I = BB->back();
      assert(I.getNumUses() == 0 && "Expected no uses to delete");
      I.eraseFromParent();
    }
  };
  // Clear Latch, Header.
  ClearBlockInstructions(Latch);
  ClearBlockInstructions(Header);

  DEBUG_ENABLE(dbgs() << "=== Exit\n" << *Exit << "=== End of Exit\n");

  Type *IVTy = OMPLoopInfo.IV->getType()->getPointerElementType();
  SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;

  FunctionCallee LoopStaticInit = ((IsDistribute && isOpenMPDeviceRuntime())
                                       ? getKmpcDistributeStaticInit(IVTy)
                                       : getKmpcForStaticInit(IVTy));
  FunctionCallee LoopStaticFini =
      ((IsDistribute && isOpenMPDeviceRuntime())
           ? OMPBuilder.getOrCreateRuntimeFunction(
                 M, OMPRTL___kmpc_distribute_static_fini)
           : OMPBuilder.getOrCreateRuntimeFunction(
                 M, OMPRTL___kmpc_for_static_fini));

  const DebugLoc DL = PreHeader->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(PreHeader, PreHeader->getTerminator()->getIterator()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadNum = nullptr;

  // Create allocas for static init values.
  // TODO: Move the AllocaIP to the start of the containing function.
  InsertPointTy AllocaIP(PreHeader, PreHeader->getFirstInsertionPt());
  Type *I32Type = Type::getInt32Ty(M.getContext());
  OMPBuilder.Builder.restoreIP(AllocaIP);
  Value *PLastIter =
      OMPBuilder.Builder.CreateAlloca(I32Type, nullptr, "omp.for.is_last");
  // Value *PStart = OMPBuilder.Builder.CreateAlloca(IVTy, nullptr,
  // "omp.for.start");
  Value *PLowerBound =
      OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.for.lb");
  Value *PStride =
      OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.for.stride");
  Value *PUpperBound =
      OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.for.ub");

  // Store distribute LB, UB to be used by combined loop constructs.
  if (IsDistribute)
    if (OMPDistributeInfo) {
      OMPDistributeInfo->LB = PLowerBound;
      OMPDistributeInfo->UB = PUpperBound;
    }

  // Create BasicBlock structure.
  BasicBlock *MinUBBlock =
      PreHeader->splitBasicBlock(PreHeader->getTerminator(), "omp.for.min.ub");
  BasicBlock *CapUBBlock = MinUBBlock->splitBasicBlock(
      MinUBBlock->getTerminator(), "omp.for.cap.ub");
  BasicBlock *SetupLoopBlock =
      CapUBBlock->splitBasicBlock(CapUBBlock->getTerminator(), "omp.for.setup");
  BasicBlock *ForEndBB =
      ExitBB->splitBasicBlockBefore(ExitBB->getFirstInsertionPt());
  ForEndBB->setName("omp.for.end");

  BasicBlock *DispatchCondBB = nullptr;
  BasicBlock *DispatchIncBB = nullptr;
  BasicBlock *DispatchEndBB = nullptr;
  if (OMPLoopInfo.Sched == OMPScheduleType::StaticChunked ||
      OMPLoopInfo.Sched == OMPScheduleType::DistributeChunked) {
    DispatchCondBB = SetupLoopBlock->splitBasicBlock(
        SetupLoopBlock->getTerminator(), "omp.dispatch.cond");
    DispatchIncBB = ExitBB->splitBasicBlockBefore(ExitBB->getFirstInsertionPt(),
                                                  "omp.dispatch.inc");
    DispatchEndBB = ExitBB->splitBasicBlockBefore(ExitBB->getFirstInsertionPt(),
                                                  "omp.dispatch.end");
  }

  Constant *Zero_I32 = ConstantInt::get(I32Type, 0);
  Constant *One = ConstantInt::get(IVTy, 1);

  // Extend PreHeader
  {
    OMPBuilder.Builder.SetInsertPoint(PreHeader->getTerminator());
    // Store the initial normalized upper bound to PUpperBound.
    Value *LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.UB);
    OMPBuilder.Builder.CreateStore(LoadUB, PUpperBound);

    Value *LoadLB = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.LB);
    OMPBuilder.Builder.CreateStore(LoadLB, PLowerBound);
    OMPBuilder.Builder.CreateStore(One, PStride);
    OMPBuilder.Builder.CreateStore(Zero_I32, PLastIter);

    // If Chunk is not specified (nullptr), default to one, complying with
    // the OpenMP specification.
    if (!OMPLoopInfo.Chunk)
      OMPLoopInfo.Chunk = One;
    Value *ChunkCast = OMPBuilder.Builder.CreateIntCast(OMPLoopInfo.Chunk, IVTy,
                                                        /*isSigned*/ false);

    Constant *SchedulingType =
        ConstantInt::get(I32Type, static_cast<int>(OMPLoopInfo.Sched));

    ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);
    DEBUG_ENABLE(dbgs() << "=== SchedulingType " << *SchedulingType << "\n");
    DEBUG_ENABLE(dbgs() << "=== PLowerBound " << *PLowerBound << "\n");
    DEBUG_ENABLE(dbgs() << "=== PUpperBound " << *PUpperBound << "\n");
    DEBUG_ENABLE(dbgs() << "=== PStride " << *PStride << "\n");
    DEBUG_ENABLE(dbgs() << "=== Incr " << *One << "\n");
    DEBUG_ENABLE(dbgs() << "=== Schedule "
                        << static_cast<int>(OMPLoopInfo.Sched) << "\n");
    DEBUG_ENABLE(dbgs() << "=== Chunk " << *ChunkCast << "\n");
    OMPBuilder.Builder.CreateCall(
        LoopStaticInit, {SrcLoc, ThreadNum, SchedulingType, PLastIter,
                         PLowerBound, PUpperBound, PStride, One, ChunkCast});
  }

  // Create MinUBBlock.
  {
    OMPBuilder.Builder.SetInsertPoint(MinUBBlock,
                                      MinUBBlock->getFirstInsertionPt());
    auto *LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, PUpperBound);
    auto *LoadGlobalUB = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.UB);
    auto *Cond = OMPBuilder.Builder.CreateICmpUGT(LoadUB, LoadGlobalUB);
    OMPBuilder.Builder.CreateCondBr(Cond, CapUBBlock, SetupLoopBlock);
    MinUBBlock->getTerminator()->eraseFromParent();
  }

  // Create CapUBBlock
  {
    OMPBuilder.Builder.SetInsertPoint(CapUBBlock,
                                      CapUBBlock->getFirstInsertionPt());
    auto *LoadGlobalUB = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.UB);
    OMPBuilder.Builder.CreateStore(LoadGlobalUB, PUpperBound);
  }

  // Create SetupLoopBlock
  {
    OMPBuilder.Builder.SetInsertPoint(SetupLoopBlock,
                                      SetupLoopBlock->getFirstInsertionPt());
    Value *LoadLB = OMPBuilder.Builder.CreateLoad(IVTy, PLowerBound);
    OMPBuilder.Builder.CreateStore(LoadLB, OMPLoopInfo.IV);
  }

  // Create Header
  {
    auto SaveIP = OMPBuilder.Builder.saveIP();
    OMPBuilder.Builder.SetInsertPoint(Header);
    auto *LoadIV = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.IV);
    auto *LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, PUpperBound);
    auto *Cond = OMPBuilder.Builder.CreateICmpSLE(LoadIV, LoadUB);
    OMPBuilder.Builder.CreateCondBr(Cond, LoopBody, ForEndBB);
    OMPBuilder.Builder.restoreIP(SaveIP);
  }

  // Create Latch.
  {
    auto SaveIP = OMPBuilder.Builder.saveIP();
    OMPBuilder.Builder.SetInsertPoint(Latch);
    Value *LoadIV = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.IV);
    if (IsDistribute && IsDistributeParallelFor) {
      Value *LoadStride = OMPBuilder.Builder.CreateLoad(IVTy, PStride);
      Value *Inc = OMPBuilder.Builder.CreateAdd(LoadIV, LoadStride);
      OMPBuilder.Builder.CreateStore(Inc, OMPLoopInfo.IV);
    } else {
      Value *Inc = OMPBuilder.Builder.CreateAdd(LoadIV, One);
      OMPBuilder.Builder.CreateStore(Inc, OMPLoopInfo.IV);
    }

    // If it's a combined "distribute parallel for" with static/distribute
    // chunked then fall through to the strided dispatch increment.
    if (IsDistributeParallelFor &&
        ((OMPLoopInfo.Sched == OMPScheduleType::StaticChunked) ||
         (OMPLoopInfo.Sched == OMPScheduleType::DistributeChunked)))
      OMPBuilder.Builder.CreateBr(DispatchIncBB);
    else
      OMPBuilder.Builder.CreateBr(Header);

    OMPBuilder.Builder.restoreIP(SaveIP);
  }

  assert(ThreadNum && "Expected non-null threadnum");
  if (OMPLoopInfo.Sched == OMPScheduleType::Static ||
      OMPLoopInfo.Sched == OMPScheduleType::Distribute) {
    OMPBuilder.Builder.SetInsertPoint(ForEndBB,
                                      ForEndBB->getFirstInsertionPt());
    OMPBuilder.Builder.CreateCall(LoopStaticFini, {SrcLoc, ThreadNum});
  } else if (OMPLoopInfo.Sched == OMPScheduleType::StaticChunked ||
             OMPLoopInfo.Sched == OMPScheduleType::DistributeChunked) {
    assert(DispatchCondBB && "Expected non-null dispatch cond bb");
    assert(DispatchIncBB && "Expected non-null dispatch inc bb");
    assert(DispatchEndBB && "Expected non-null dispatch end bb");
    // Create DispatchCond
    {
      auto SaveIP = OMPBuilder.Builder.saveIP();
      DispatchCondBB->getTerminator()->eraseFromParent();
      OMPBuilder.Builder.SetInsertPoint(DispatchCondBB);
      auto *LoadLB = OMPBuilder.Builder.CreateLoad(IVTy, PLowerBound);
      OMPBuilder.Builder.CreateStore(LoadLB, OMPLoopInfo.IV);
      auto *LoadIV = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.IV);
      auto *LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, PUpperBound);
      auto *Cond = OMPBuilder.Builder.CreateICmpSLE(LoadIV, LoadUB);
      OMPBuilder.Builder.CreateCondBr(Cond, Header, DispatchEndBB);
      OMPBuilder.Builder.restoreIP(SaveIP);
    }
    // Create DispatchIncBB.
    {
      auto SaveIP = OMPBuilder.Builder.saveIP();
      DispatchIncBB->getTerminator()->eraseFromParent();
      OMPBuilder.Builder.SetInsertPoint(DispatchIncBB);
      auto *LoadLB = OMPBuilder.Builder.CreateLoad(IVTy, PLowerBound);
      auto *LoadStride = OMPBuilder.Builder.CreateLoad(IVTy, PStride);
      auto *LBPlusStride = OMPBuilder.Builder.CreateAdd(LoadLB, LoadStride);
      OMPBuilder.Builder.CreateStore(LBPlusStride, PLowerBound);

      auto *LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, PUpperBound);
      auto *UBPlusStride = OMPBuilder.Builder.CreateAdd(LoadUB, LoadStride);
      OMPBuilder.Builder.CreateStore(UBPlusStride, PUpperBound);

      // OMPBuilder.Builder.CreateBr(DispatchCondBB);
      OMPBuilder.Builder.CreateBr(MinUBBlock);
      OMPBuilder.Builder.restoreIP(SaveIP);
    }
    // Create ForEndBB
    {
      ForEndBB->getTerminator()->eraseFromParent();
      OMPBuilder.Builder.SetInsertPoint(ForEndBB);
      OMPBuilder.Builder.CreateBr(DispatchIncBB);
    }

    // Create DispatchEndBB
    {
      OMPBuilder.Builder.SetInsertPoint(DispatchEndBB,
                                        DispatchEndBB->getFirstInsertionPt());
      OMPBuilder.Builder.CreateCall(LoopStaticFini, {SrcLoc, ThreadNum});
    }
  } else {
    FATAL_ERROR("Unknown loop schedule type");
  }

  OpenMPIRBuilder::OutlineInfo OI;
  OI.EntryBB = PreHeader;
  OI.ExitBB = Exit;
  SmallPtrSet<BasicBlock *, 8> BlockSet;
  SmallVector<BasicBlock *, 8> BlockVector;
  OI.collectBlocks(BlockSet, BlockVector);

  // TODO: De-duplicate privatization code.
  auto PrivatizeWithReductions = [&]() {
    auto CurrentIP = OMPBuilder.Builder.saveIP();
    for (auto &It : DSAValueMap) {
      Value *Orig = It.first;
      DSAType DSA = It.second.Type;
      FunctionCallee CopyConstructor = It.second.CopyConstructor;
      Value *ReplacementValue = nullptr;
      Type *VTy = Orig->getType()->getPointerElementType();

      if (DSA == DSA_SHARED)
        continue;

      // Lastprivates are handled later, need elaborate codegen.
      if (DSA == DSA_LASTPRIVATE)
        continue;

      // Store previous uses to set them to the ReplacementValue after
      // privatization codegen.
      SetVector<Use *> Uses;
      for (Use &U : Orig->uses())
        if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
          if (BlockSet.count(UserI->getParent()))
            Uses.insert(&U);

      OMPBuilder.Builder.restoreIP(AllocaIP);
      if (DSA == DSA_PRIVATE) {
        ReplacementValue = OMPBuilder.Builder.CreateAlloca(
            VTy, /*ArraySize */ nullptr, Orig->getName() + ".for.priv");
        OMPBuilder.Builder.CreateStore(Constant::getNullValue(VTy),
                                       ReplacementValue);
      } else if (DSA == DSA_FIRSTPRIVATE) {
        Value *V = OMPBuilder.Builder.CreateLoad(
            VTy, Orig, Orig->getName() + ".for.firstpriv.reload");
        ReplacementValue = OMPBuilder.Builder.CreateAlloca(
            VTy, /*ArraySize */ nullptr,
            Orig->getName() + ".for.firstpriv.copy");
        if (CopyConstructor) {
          Value *Copy = OMPBuilder.Builder.CreateCall(CopyConstructor, {V});
          OMPBuilder.Builder.CreateStore(Copy, ReplacementValue);
        } else
          OMPBuilder.Builder.CreateStore(V, ReplacementValue);
      } else if (DSA == DSA_REDUCTION_ADD) {
        ReplacementValue =
            CGReduction::emitInitAndAppendInfo<DSA_REDUCTION_ADD>(
                OMPBuilder.Builder, OMPBuilder.Builder.saveIP(), Orig,
                ReductionInfos, false);
      } else if (DSA == DSA_REDUCTION_SUB) {
        ReplacementValue =
            CGReduction::emitInitAndAppendInfo<DSA_REDUCTION_SUB>(
                OMPBuilder.Builder, OMPBuilder.Builder.saveIP(), Orig,
                ReductionInfos, false);
      } else if (DSA == DSA_REDUCTION_MUL) {
        ReplacementValue =
            CGReduction::emitInitAndAppendInfo<DSA_REDUCTION_MUL>(
                OMPBuilder.Builder, OMPBuilder.Builder.saveIP(), Orig,
                ReductionInfos, false);
      } else
        FATAL_ERROR("Unsupported privatization");

      assert(ReplacementValue && "Expected non-null ReplacementValue");

      for (Use *UPtr : Uses)
        UPtr->set(ReplacementValue);
    }

    OMPBuilder.Builder.restoreIP(CurrentIP);
  };

  auto EmitLastPrivate = [&](InsertPointTy CodeGenIP) {
    auto ShouldReplace = [&BlockSet](Use &U) {
      if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
        if (BlockSet.count(UserI->getParent()))
          return true;

      return false;
    };

    for (auto &It : DSAValueMap) {
      Value *Orig = It.first;
      DSAType DSA = It.second.Type;

      if (DSA != DSA_LASTPRIVATE)
        continue;

      FunctionCallee CopyConstructor = It.second.CopyConstructor;
      Value *ReplacementValue = nullptr;
      Type *VTy = Orig->getType()->getPointerElementType();

      OMPBuilder.Builder.restoreIP(AllocaIP);
      ReplacementValue = OMPBuilder.Builder.CreateAlloca(
          VTy, /*ArraySize */ nullptr, Orig->getName() + ".for.lastpriv");
      OMPBuilder.Builder.CreateStore(Constant::getNullValue(VTy),
                                     ReplacementValue);
      Orig->replaceUsesWithIf(ReplacementValue, ShouldReplace);

      BasicBlock *InsertBB = CodeGenIP.getBlock();

      BasicBlock *LastPrivCond =
          SplitBlock(InsertBB, InsertBB->getTerminator());
      LastPrivCond->setName("omp.for.lastpriv.cond");
      BasicBlock *LastPrivThen =
          SplitBlock(LastPrivCond, LastPrivCond->getTerminator());
      LastPrivThen->setName("omp.for.lastpriv.then");
      BasicBlock *LastPrivEnd =
          SplitBlock(LastPrivThen, LastPrivThen->getTerminator());
      LastPrivEnd->setName("omp.for.lastpriv.end");
      OMPBuilder.Builder.SetInsertPoint(LastPrivThen->getTerminator());
      Value *Load = OMPBuilder.Builder.CreateLoad(VTy, ReplacementValue);
      if (CopyConstructor) {
        Value *Copy = OMPBuilder.Builder.CreateCall(CopyConstructor, {Load});
        OMPBuilder.Builder.CreateStore(Copy, Orig);
      } else
        OMPBuilder.Builder.CreateStore(Load, Orig);

      LastPrivCond->getTerminator()->eraseFromParent();
      OMPBuilder.Builder.SetInsertPoint(LastPrivCond);
      Value *PLastIterLoad =
          OMPBuilder.Builder.CreateLoad(OMPBuilder.Int32, PLastIter);
      Value *Cond = OMPBuilder.Builder.CreateICmpNE(
          PLastIterLoad, ConstantInt::get(OMPBuilder.Int32, 0));
      OMPBuilder.Builder.CreateCondBr(Cond, LastPrivThen, LastPrivEnd);
    }
  };

  BasicBlock *FiniBB =
      (OMPLoopInfo.Sched == OMPScheduleType::Static) ? ForEndBB : DispatchEndBB;
  EmitLastPrivate(InsertPointTy(FiniBB, FiniBB->end()));

  // Emit reductions, barrier, privatize if standalone.
  if (IsStandalone) {
    PrivatizeWithReductions();
    if (!ReductionInfos.empty()) {
      OMPBuilder.Builder.SetInsertPoint(ForEndBB->getTerminator());
      if (isOpenMPDeviceRuntime())
        emitReductionsDevice(OpenMPIRBuilder::LocationDescription(
                                 OMPBuilder.Builder.saveIP(), Loc.DL),
                             AllocaIP, ReductionInfos, false);
      else
        emitReductionsHost(OpenMPIRBuilder::LocationDescription(
                               OMPBuilder.Builder.saveIP(), Loc.DL),
                           AllocaIP, ReductionInfos);
    }

    OMPBuilder.Builder.SetInsertPoint(ExitBB->getTerminator());
    OMPBuilder.createBarrier(OpenMPIRBuilder::LocationDescription(
                                 OMPBuilder.Builder.saveIP(), Loc.DL),
                             omp::Directive::OMPD_for,
                             /* ForceSimpleCall */ false,
                             /* CheckCancelFlag */ false);
  }

  if (verifyFunction(*PreHeader->getParent(), &errs()))
    FATAL_ERROR("Verification of omp for lowering failed!");
}

void CGIntrinsicsOpenMP::emitOMPFor(DSAValueMapTy &DSAValueMap,
                                    OMPLoopInfoStruct &OMPLoopInfo,
                                    BasicBlock *StartBB, BasicBlock *ExitBB,
                                    bool IsStandalone,
                                    bool IsDistributeParallelFor) {
  // Set default loop schedule.
  if (static_cast<int>(OMPLoopInfo.Sched) == 0)
    OMPLoopInfo.Sched =
        (isOpenMPDeviceRuntime() ? OMPScheduleType::StaticChunked
                                 : OMPScheduleType::Static);

  emitLoop(DSAValueMap, OMPLoopInfo, StartBB, ExitBB, IsStandalone, false,
           IsDistributeParallelFor);
}

void CGIntrinsicsOpenMP::emitOMPTask(DSAValueMapTy &DSAValueMap, Function *Fn,
                                     BasicBlock *BBEntry, BasicBlock *StartBB,
                                     BasicBlock *EndBB, BasicBlock *AfterBB) {
  // Define types.
  // ************** START TYPE DEFINITION ************** //
  enum {
    TiedFlag = 0x1,
    FinalFlag = 0x2,
    DestructorsFlag = 0x8,
    PriorityFlag = 0x20,
    DetachableFlag = 0x40,
  };

  // This is a union for priority/firstprivate destructors, use the
  // routine entry pointer to allocate space since it is larger than
  // Int32Ty for priority, see kmp.h. Unused for now.
  StructType *KmpCmplrdataTy =
      StructType::create({OMPBuilder.TaskRoutineEntryPtr});
  StructType *KmpTaskTTy =
      StructType::create({OMPBuilder.VoidPtr, OMPBuilder.TaskRoutineEntryPtr,
                          OMPBuilder.Int32, KmpCmplrdataTy, KmpCmplrdataTy},
                         "struct.kmp_task_t");
  Type *KmpTaskTPtrTy = KmpTaskTTy->getPointerTo();

  FunctionCallee KmpcOmpTaskAlloc =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_task_alloc);
  SmallVector<Type *, 8> SharedsTy;
  SmallVector<Type *, 8> PrivatesTy;
  for (auto &It : DSAValueMap) {
    Value *OriginalValue = It.first;
    if (It.second.Type == DSA_SHARED)
      SharedsTy.push_back(OriginalValue->getType());
    else if (It.second.Type == DSA_PRIVATE ||
             It.second.Type == DSA_FIRSTPRIVATE) {
      assert(isa<PointerType>(OriginalValue->getType()) &&
             "Expected private, firstprivate value with pointer type");
      // Store a copy of the value, thus get the pointer element type.
      PrivatesTy.push_back(OriginalValue->getType()->getPointerElementType());
    } else
      FATAL_ERROR("Unknown DSA type");
  }

  StructType *KmpSharedsTTy = nullptr;
  if (SharedsTy.empty())
    KmpSharedsTTy = StructType::create(M.getContext(), "struct.kmp_shareds");
  else
    KmpSharedsTTy = StructType::create(SharedsTy, "struct.kmp_shareds");
  assert(KmpSharedsTTy && "Expected non-null KmpSharedsTTy");
  Type *KmpSharedsTPtrTy = KmpSharedsTTy->getPointerTo();
  StructType *KmpPrivatesTTy =
      StructType::create(PrivatesTy, "struct.kmp_privates");
  Type *KmpPrivatesTPtrTy = KmpPrivatesTTy->getPointerTo();
  StructType *KmpTaskTWithPrivatesTy = StructType::create(
      {KmpTaskTTy, KmpPrivatesTTy}, "struct.kmp_task_t_with_privates");
  Type *KmpTaskTWithPrivatesPtrTy = KmpTaskTWithPrivatesTy->getPointerTo();

  // Declare the task entry function.
  Function *TaskEntryFn = Function::Create(
      OMPBuilder.TaskRoutineEntry, GlobalValue::InternalLinkage,
      Fn->getAddressSpace(), Fn->getName() + ".omp_task_entry", &M);
  // Name arguments.
  TaskEntryFn->getArg(0)->setName(".global_tid");
  TaskEntryFn->getArg(1)->setName(".task_t_with_privates");

  // Declare the task outlined function.
  FunctionType *TaskOutlinedFnTy =
      FunctionType::get(OMPBuilder.Void,
                        {OMPBuilder.Int32, OMPBuilder.Int32Ptr,
                         OMPBuilder.VoidPtr, KmpTaskTPtrTy, KmpSharedsTPtrTy},
                        /*isVarArg=*/false);
  Function *TaskOutlinedFn = Function::Create(
      TaskOutlinedFnTy, GlobalValue::InternalLinkage, Fn->getAddressSpace(),
      Fn->getName() + ".omp_task_outlined", &M);
  TaskOutlinedFn->getArg(0)->setName(".global_tid");
  TaskOutlinedFn->getArg(1)->setName(".part_id");
  TaskOutlinedFn->getArg(2)->setName(".privates");
  TaskOutlinedFn->getArg(3)->setName(".task.data");
  TaskOutlinedFn->getArg(4)->setName(".shareds");

  // ************** END TYPE DEFINITION ************** //

  // Emit kmpc_omp_task_alloc, kmpc_omp_task
  {
    const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
    OpenMPIRBuilder::LocationDescription Loc(
        InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);
    uint32_t SrcLocStrSize;
    Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
    Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
    // TODO: parse clauses, for now fix flags to tied
    unsigned TaskFlags = TiedFlag;
    Value *SizeofShareds = nullptr;
    if (KmpSharedsTTy->isEmptyTy())
      SizeofShareds = OMPBuilder.Builder.getInt64(0);
    else
      SizeofShareds = OMPBuilder.Builder.getInt64(
          M.getDataLayout().getTypeAllocSize(KmpSharedsTTy));
    Value *SizeofKmpTaskTWithPrivates = OMPBuilder.Builder.getInt64(
        M.getDataLayout().getTypeAllocSize(KmpTaskTWithPrivatesTy));
    OMPBuilder.Builder.SetInsertPoint(BBEntry, BBEntry->getFirstInsertionPt());
    Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);
    Value *KmpTaskTWithPrivatesVoidPtr = OMPBuilder.Builder.CreateCall(
        KmpcOmpTaskAlloc,
        {SrcLoc, ThreadNum, OMPBuilder.Builder.getInt32(TaskFlags),
         SizeofKmpTaskTWithPrivates, SizeofShareds, TaskEntryFn},
        ".task.data");
    Value *KmpTaskTWithPrivates = OMPBuilder.Builder.CreateBitCast(
        KmpTaskTWithPrivatesVoidPtr, KmpTaskTWithPrivatesPtrTy);

    const unsigned KmpTaskTIdx = 0;
    const unsigned KmpSharedsIdx = 0;
    Value *KmpTaskT = OMPBuilder.Builder.CreateStructGEP(
        KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, KmpTaskTIdx);
    Value *KmpSharedsGEP =
        OMPBuilder.Builder.CreateStructGEP(KmpTaskTTy, KmpTaskT, KmpSharedsIdx);
    Value *KmpSharedsVoidPtr =
        OMPBuilder.Builder.CreateLoad(OMPBuilder.VoidPtr, KmpSharedsGEP);
    Value *KmpShareds =
        OMPBuilder.Builder.CreateBitCast(KmpSharedsVoidPtr, KmpSharedsTPtrTy);
    const unsigned KmpPrivatesIdx = 1;
    Value *KmpPrivates = OMPBuilder.Builder.CreateStructGEP(
        KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, KmpPrivatesIdx);

    // Store shareds by reference, firstprivates by value, in task data
    // storage.
    unsigned SharedsGEPIdx = 0;
    unsigned PrivatesGEPIdx = 0;
    for (auto &It : DSAValueMap) {
      Value *OriginalValue = It.first;
      DSAType DSA = It.second.Type;
      FunctionCallee CopyConstructor = It.second.CopyConstructor;
      if (DSA == DSA_SHARED) {
        Value *SharedGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpSharedsTTy, KmpShareds, SharedsGEPIdx,
            OriginalValue->getName() + ".task.shared");
        OMPBuilder.Builder.CreateStore(OriginalValue, SharedGEP);
        ++SharedsGEPIdx;
      } else if (DSA == DSA_FIRSTPRIVATE) {
        Value *FirstprivateGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpPrivatesTTy, KmpPrivates, PrivatesGEPIdx,
            OriginalValue->getName() + ".task.firstprivate");
        Value *Load = OMPBuilder.Builder.CreateLoad(
            OriginalValue->getType()->getPointerElementType(), OriginalValue);
        if (CopyConstructor) {
          Value *Copy = OMPBuilder.Builder.CreateCall(CopyConstructor, {Load});
          OMPBuilder.Builder.CreateStore(Copy, FirstprivateGEP);
        } else
          OMPBuilder.Builder.CreateStore(Load, FirstprivateGEP);
        ++PrivatesGEPIdx;
      } else if (DSA == DSA_PRIVATE)
        ++PrivatesGEPIdx;
    }

    FunctionCallee KmpcOmpTask =
        OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_task);
    OMPBuilder.Builder.CreateCall(
        KmpcOmpTask, {SrcLoc, ThreadNum, KmpTaskTWithPrivatesVoidPtr});
  }

  // Emit task entry function.
  {
    BasicBlock *TaskEntryBB =
        BasicBlock::Create(M.getContext(), "entry", TaskEntryFn);
    OMPBuilder.Builder.SetInsertPoint(TaskEntryBB);
    const unsigned TaskTIdx = 0;
    const unsigned PrivatesIdx = 1;
    const unsigned SharedsIdx = 0;
    Value *GTId = TaskEntryFn->getArg(0);
    Value *KmpTaskTWithPrivates = OMPBuilder.Builder.CreateBitCast(
        TaskEntryFn->getArg(1), KmpTaskTWithPrivatesPtrTy);
    Value *KmpTaskT = OMPBuilder.Builder.CreateStructGEP(
        KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, TaskTIdx, ".task.data");
    Value *SharedsGEP = OMPBuilder.Builder.CreateStructGEP(
        KmpTaskTTy, KmpTaskT, SharedsIdx, ".shareds.gep");
    Value *SharedsVoidPtr = OMPBuilder.Builder.CreateLoad(
        OMPBuilder.VoidPtr, SharedsGEP, ".shareds.void.ptr");
    Value *Shareds = OMPBuilder.Builder.CreateBitCast(
        SharedsVoidPtr, KmpSharedsTPtrTy, ".shareds");

    Value *Privates = nullptr;
    if (PrivatesTy.empty()) {
      Privates = Constant::getNullValue(OMPBuilder.VoidPtr);
    } else {
      Value *PrivatesTyped = OMPBuilder.Builder.CreateStructGEP(
          KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, PrivatesIdx,
          ".privates");
      Privates = OMPBuilder.Builder.CreateBitCast(
          PrivatesTyped, OMPBuilder.VoidPtr, ".privates.void.ptr");
    }
    assert(Privates && "Expected non-null privates");

    const unsigned PartIdIdx = 2;
    Value *PartId = OMPBuilder.Builder.CreateStructGEP(KmpTaskTTy, KmpTaskT,
                                                       PartIdIdx, ".part_id");
    OMPBuilder.Builder.CreateCall(TaskOutlinedFnTy, TaskOutlinedFn,
                                  {GTId, PartId, Privates, KmpTaskT, Shareds});
    OMPBuilder.Builder.CreateRet(OMPBuilder.Builder.getInt32(0));
  }

  // Emit TaskOutlinedFn code.
  {
    OpenMPIRBuilder::OutlineInfo OI;
    OI.EntryBB = StartBB;
    OI.ExitBB = EndBB;
    SmallPtrSet<BasicBlock *, 8> OutlinedBlockSet;
    SmallVector<BasicBlock *, 8> OutlinedBlockVector;
    OI.collectBlocks(OutlinedBlockSet, OutlinedBlockVector);
    BasicBlock *TaskOutlinedEntryBB =
        BasicBlock::Create(M.getContext(), "entry", TaskOutlinedFn);
    BasicBlock *TaskOutlinedExitBB =
        BasicBlock::Create(M.getContext(), "exit", TaskOutlinedFn);
    for (BasicBlock *BB : OutlinedBlockVector)
      BB->moveBefore(TaskOutlinedExitBB);
    // Explicitly move EndBB to the outlined functions, since OutlineInfo
    // does not contain it in the OutlinedBlockVector.
    EndBB->moveBefore(TaskOutlinedExitBB);
    EndBB->getTerminator()->setSuccessor(0, TaskOutlinedExitBB);

    OMPBuilder.Builder.SetInsertPoint(TaskOutlinedEntryBB);
    const unsigned KmpPrivatesArgNo = 2;
    const unsigned KmpSharedsArgNo = 4;
    Value *KmpPrivatesArgVoidPtr = TaskOutlinedFn->getArg(KmpPrivatesArgNo);
    Value *KmpPrivatesArg = OMPBuilder.Builder.CreateBitCast(
        KmpPrivatesArgVoidPtr, KmpPrivatesTPtrTy);
    Value *KmpSharedsArg = TaskOutlinedFn->getArg(KmpSharedsArgNo);

    // Replace shareds, privates, firstprivates to refer to task data
    // storage.
    unsigned SharedsGEPIdx = 0;
    unsigned PrivatesGEPIdx = 0;
    for (auto &It : DSAValueMap) {
      Value *OriginalValue = It.first;
      Value *ReplacementValue = nullptr;
      if (It.second.Type == DSA_SHARED) {
        Value *SharedGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpSharedsTTy, KmpSharedsArg, SharedsGEPIdx,
            OriginalValue->getName() + ".task.shared.gep");
        ReplacementValue = OMPBuilder.Builder.CreateLoad(
            OriginalValue->getType(), SharedGEP,
            OriginalValue->getName() + ".task.shared");
        ++SharedsGEPIdx;
      } else if (It.second.Type == DSA_PRIVATE) {
        Value *PrivateGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpPrivatesTTy, KmpPrivatesArg, PrivatesGEPIdx,
            OriginalValue->getName() + ".task.private.gep");
        ReplacementValue = PrivateGEP;
        // NOTE: Zero initialize private to avoid issue with Numba ref counting.
        OMPBuilder.Builder.CreateStore(
            Constant::getNullValue(
                OriginalValue->getType()->getPointerElementType()),
            ReplacementValue);
        ++PrivatesGEPIdx;
      } else if (It.second.Type == DSA_FIRSTPRIVATE) {
        Value *FirstprivateGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpPrivatesTTy, KmpPrivatesArg, PrivatesGEPIdx,
            OriginalValue->getName() + ".task.firstprivate.gep");
        ReplacementValue = FirstprivateGEP;
        ++PrivatesGEPIdx;
      } else
        FATAL_ERROR("Unknown DSA type");

      assert(ReplacementValue && "Expected non-null ReplacementValue");
      SmallVector<User *, 8> Users(OriginalValue->users());
      for (User *U : Users)
        if (Instruction *I = dyn_cast<Instruction>(U))
          if (OutlinedBlockSet.contains(I->getParent()))
            I->replaceUsesOfWith(OriginalValue, ReplacementValue);
    }

    OMPBuilder.Builder.CreateBr(StartBB);
    OMPBuilder.Builder.SetInsertPoint(TaskOutlinedExitBB);
    OMPBuilder.Builder.CreateRetVoid();
    BBEntry->getTerminator()->setSuccessor(0, AfterBB);
  }
}

void CGIntrinsicsOpenMP::emitOMPOffloadingEntry(const Twine &DevFuncName,
                                                Value *EntryPtr,
                                                Constant *&OMPOffloadEntry) {

  Constant *DevFuncNameConstant =
      ConstantDataArray::getString(M.getContext(), DevFuncName.str());
  auto *GV = new GlobalVariable(
      M, DevFuncNameConstant->getType(),
      /* isConstant */ true, GlobalValue::InternalLinkage, DevFuncNameConstant,
      ".omp_offloading.entry_name", nullptr, GlobalVariable::NotThreadLocal,
      /* AddressSpace */ 0);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  Constant *EntryConst = dyn_cast<Constant>(EntryPtr);
  assert(EntryConst && "Expected constant entry pointer");
  OMPOffloadEntry = ConstantStruct::get(
      TgtOffloadEntryTy,
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(EntryConst,
                                                     OMPBuilder.VoidPtr),
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, OMPBuilder.Int8Ptr),
      ConstantInt::get(OMPBuilder.SizeTy, 0),
      ConstantInt::get(OMPBuilder.Int32, 0),
      ConstantInt::get(OMPBuilder.Int32, 0));
  auto *OMPOffloadEntryGV = new GlobalVariable(
      M, TgtOffloadEntryTy,
      /* isConstant */ true, GlobalValue::WeakAnyLinkage, OMPOffloadEntry,
      ".omp_offloading.entry." + DevFuncName);
  OMPOffloadEntryGV->setSection("omp_offloading_entries");
  OMPOffloadEntryGV->setAlignment(Align(1));
}

void CGIntrinsicsOpenMP::emitOMPOffloadingMappings(
    InsertPointTy AllocaIP, DSAValueMapTy &DSAValueMap,
    StructMapTy &StructMappingInfoMap,
    OffloadingMappingArgsTy &OffloadingMappingArgs, bool IsTargetRegion) {

  struct MapperInfo {
    Value *BasePtr;
    Value *Ptr;
    Value *Size;
  };

  SmallVector<MapperInfo, 8> MapperInfos;
  // SmallVector<Constant *, 8> OffloadSizes;
  SmallVector<Constant *, 8> OffloadMapTypes;
  SmallVector<Constant *, 8> OffloadMapNames;

  if (DSAValueMap.empty()) {
    OffloadingMappingArgs.Size = 0;
    OffloadingMappingArgs.BasePtrs =
        Constant::getNullValue(OMPBuilder.VoidPtrPtr);
    OffloadingMappingArgs.Ptrs = Constant::getNullValue(OMPBuilder.VoidPtrPtr);
    OffloadingMappingArgs.Sizes = Constant::getNullValue(OMPBuilder.Int64Ptr);
    OffloadingMappingArgs.MapTypes =
        Constant::getNullValue(OMPBuilder.Int64Ptr);
    OffloadingMappingArgs.MapNames =
        Constant::getNullValue(OMPBuilder.VoidPtrPtr);

    return;
  }

  auto EmitMappingEntry = [&](Value *Size, uint64_t MapType, Value *BasePtr,
                              Value *Ptr) {
    OffloadMapTypes.push_back(ConstantInt::get(OMPBuilder.SizeTy, MapType));
    // TODO: maybe add debug info.
    uint32_t SrcLocStrSize;
    OffloadMapNames.push_back(OMPBuilder.getOrCreateSrcLocStr(
        BasePtr->getName(), "", 0, 0, SrcLocStrSize));
    DEBUG_ENABLE(dbgs() << "Emit mapping entry BasePtr " << *BasePtr << " Ptr "
                        << *Ptr << " Size " << *Size << " MapType " << MapType
                        << "\n");
    MapperInfos.push_back({BasePtr, Ptr, Size});
  };

  auto GetMapType = [IsTargetRegion](DSAType DSA) {
    uint64_t MapType;
    // Determine the map type, completely or partly (structs).
    switch (DSA) {
    case DSA_FIRSTPRIVATE:
      MapType = OMP_TGT_MAPTYPE_LITERAL;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_ALLOC:
      // Allocation is the default in the OpenMP runtime, no extra flags.
      MapType = OMP_TGT_MAPTYPE_NONE;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_TO:
      MapType = OMP_TGT_MAPTYPE_TO;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_FROM:
      MapType = OMP_TGT_MAPTYPE_FROM;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_REDUCTION_ADD:
    case DSA_REDUCTION_SUB:
    case DSA_REDUCTION_MUL:
    case DSA_MAP_TOFROM:
      MapType = OMP_TGT_MAPTYPE_TO | OMP_TGT_MAPTYPE_FROM;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_STRUCT:
      MapType = OMP_TGT_MAPTYPE_NONE;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_ALLOC_STRUCT:
      // Allocation is the default in the OpenMP runtime, no extra flags.
      MapType = OMP_TGT_MAPTYPE_NONE;
      break;
    case DSA_MAP_TO_STRUCT:
      MapType = OMP_TGT_MAPTYPE_TO;
      break;
    case DSA_MAP_FROM_STRUCT:
      MapType = OMP_TGT_MAPTYPE_FROM;
      break;
    case DSA_MAP_TOFROM_STRUCT:
      MapType = OMP_TGT_MAPTYPE_TO | OMP_TGT_MAPTYPE_FROM;
      break;
    case DSA_PRIVATE:
      // do nothing
      break;
    default:
      FATAL_ERROR("Unknown mapping type");
    }

    return MapType;
  };

  // Keep track of argument position, needed for struct mappings.
  for (auto &It : DSAValueMap) {
    Value *V = It.first;
    DSAType DSA = It.second.Type;

    // Emit the mapping entry.
    Value *Size;
    switch (DSA) {
    case DSA_MAP_ALLOC:
    case DSA_MAP_TO:
    case DSA_MAP_FROM:
    case DSA_MAP_TOFROM:
    case DSA_REDUCTION_ADD:
    case DSA_REDUCTION_SUB:
    case DSA_REDUCTION_MUL:
      Size = ConstantInt::get(OMPBuilder.SizeTy,
                              M.getDataLayout().getTypeAllocSize(V->getType()));
      EmitMappingEntry(Size, GetMapType(DSA), V, V);
      break;
    case DSA_FIRSTPRIVATE: {
      auto *Load = OMPBuilder.Builder.CreateLoad(
          V->getType()->getPointerElementType(), V);
      // TODO: Runtime expects values in Int64 type, fix with arguments in
      // struct.
      AllocaInst *TmpInt64 = OMPBuilder.Builder.CreateAlloca(
          OMPBuilder.Int64, nullptr, V->getName() + ".casted");
      Value *Cast = OMPBuilder.Builder.CreateBitCast(TmpInt64, V->getType());
      auto *Store = OMPBuilder.Builder.CreateStore(Load, Cast);
      Value *ScalarV =
          OMPBuilder.Builder.CreateLoad(OMPBuilder.Int64, TmpInt64);
      Size = ConstantInt::get(OMPBuilder.SizeTy,
                              M.getDataLayout().getTypeAllocSize(
                                  V->getType()->getPointerElementType()));
      EmitMappingEntry(Size, GetMapType(DSA), ScalarV, ScalarV);
      break;
    }
    case DSA_MAP_STRUCT: {
      Size = ConstantInt::get(OMPBuilder.SizeTy,
                              M.getDataLayout().getTypeAllocSize(
                                  V->getType()->getPointerElementType()));
      EmitMappingEntry(Size, GetMapType(DSA), V, V);
      // Stores the argument position (starting from 1) of the parent
      // struct, to be used to set MEMBER_OF in the map type.
      size_t ArgPos = MapperInfos.size();

      for (auto &FieldInfo : StructMappingInfoMap[V]) {
        // MEMBER_OF(Argument Position)
        const size_t MemberOfOffset = 48;
        uint64_t MemberOfBits = ArgPos << MemberOfOffset;
        uint64_t FieldMapType = GetMapType(FieldInfo.MapType) | MemberOfBits;
        auto *FieldGEP = OMPBuilder.Builder.CreateInBoundsGEP(
            V->getType()->getPointerElementType(), V,
            {OMPBuilder.Builder.getInt32(0), FieldInfo.Index});

        Value *BasePtr = nullptr;
        Value *Ptr = nullptr;

        if (FieldGEP->getType()->getPointerElementType()->isPointerTy()) {
          FieldMapType |= OMP_TGT_MAPTYPE_PTR_AND_OBJ;
          BasePtr = FieldGEP;
          auto *Load = OMPBuilder.Builder.CreateLoad(
              BasePtr->getType()->getPointerElementType(), BasePtr);
          Ptr = OMPBuilder.Builder.CreateInBoundsGEP(
              Load->getType()->getPointerElementType(), Load, FieldInfo.Offset);
        } else {
          BasePtr = V;
          Ptr = OMPBuilder.Builder.CreateInBoundsGEP(
              FieldGEP->getType()->getPointerElementType(), FieldGEP,
              FieldInfo.Offset);
        }

        assert(BasePtr && "Expected non-null base pointer");
        assert(Ptr && "Expected non-null pointer");

        auto ElementSize = ConstantInt::get(
            OMPBuilder.SizeTy, M.getDataLayout().getTypeAllocSize(
                                   Ptr->getType()->getPointerElementType()));
        Value *NumElements = nullptr;

        // Load the value of NumElements if it is a pointer.
        if (FieldInfo.NumElements->getType()->isPointerTy())
          NumElements = OMPBuilder.Builder.CreateLoad(OMPBuilder.SizeTy,
                                                      FieldInfo.NumElements);
        else
          NumElements = FieldInfo.NumElements;

        auto *Size = OMPBuilder.Builder.CreateMul(ElementSize, NumElements);
        EmitMappingEntry(Size, FieldMapType, BasePtr, Ptr);
      }
      break;
    }
    case DSA_PRIVATE: {
      // do nothing
      break;
    }
    default:
      FATAL_ERROR("Unsupported mapping type " + toString(DSA));
    }
  }

  auto EmitConstantArrayGlobalBitCast = [&](SmallVectorImpl<Constant *> &Vector,
                                            Type *Ty, Type *DestTy,
                                            StringRef Name) {
    auto *Init = ConstantArray::get(ArrayType::get(Ty, Vector.size()), Vector);
    auto *GV = new GlobalVariable(M, ArrayType::get(Ty, Vector.size()),
                                  /* isConstant */ true,
                                  GlobalVariable::PrivateLinkage, Init, Name);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    return OMPBuilder.Builder.CreateBitCast(GV, DestTy);
  };

  // TODO: offload_sizes can be a global of constants for optimization if all
  // sizes are constants.
  // OffloadingMappingArgs.Sizes =
  //    EmitConstantArrayGlobalBitCast(OffloadSizes, OMPBuilder.SizeTy,
  //                            OMPBuilder.Int64Ptr, ".offload_sizes");
  OffloadingMappingArgs.MapTypes =
      EmitConstantArrayGlobalBitCast(OffloadMapTypes, OMPBuilder.SizeTy,
                                     OMPBuilder.Int64Ptr, ".offload_maptypes");
  OffloadingMappingArgs.MapNames = EmitConstantArrayGlobalBitCast(
      OffloadMapNames, OMPBuilder.Int8Ptr, OMPBuilder.VoidPtrPtr,
      ".offload_mapnames");

  auto EmitArrayAlloca = [&](size_t Size, Type *Ty, StringRef Name) {
    InsertPointTy CodeGenIP = OMPBuilder.Builder.saveIP();

    OMPBuilder.Builder.restoreIP(AllocaIP);
    auto *Alloca = OMPBuilder.Builder.CreateAlloca(ArrayType::get(Ty, Size),
                                                   nullptr, Name);

    OMPBuilder.Builder.restoreIP(CodeGenIP);

    return Alloca;
  };

  auto *BasePtrsAlloca = EmitArrayAlloca(MapperInfos.size(), OMPBuilder.VoidPtr,
                                         ".offload_baseptrs");
  auto *PtrsAlloca =
      EmitArrayAlloca(MapperInfos.size(), OMPBuilder.VoidPtr, ".offload_ptrs");
  auto *SizesAlloca =
      EmitArrayAlloca(MapperInfos.size(), OMPBuilder.SizeTy, ".offload_sizes");

  size_t Idx = 0;
  for (auto &MI : MapperInfos) {
    // Store in the base pointers alloca.
    auto *GEP = OMPBuilder.Builder.CreateInBoundsGEP(
        BasePtrsAlloca->getType()->getPointerElementType(), BasePtrsAlloca,
        {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
    auto *Bitcast = OMPBuilder.Builder.CreateBitCast(
        GEP, MI.BasePtr->getType()->getPointerTo());
    OMPBuilder.Builder.CreateStore(MI.BasePtr, Bitcast);

    // Store in the pointers alloca.
    GEP = OMPBuilder.Builder.CreateInBoundsGEP(
        PtrsAlloca->getType()->getPointerElementType(), PtrsAlloca,
        {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
    Bitcast = OMPBuilder.Builder.CreateBitCast(
        GEP, MI.Ptr->getType()->getPointerTo());
    OMPBuilder.Builder.CreateStore(MI.Ptr, Bitcast);

    // Store in the sizes alloca.
    GEP = OMPBuilder.Builder.CreateInBoundsGEP(
        SizesAlloca->getType()->getPointerElementType(), SizesAlloca,
        {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
    Bitcast = OMPBuilder.Builder.CreateBitCast(
        GEP, MI.Size->getType()->getPointerTo());
    OMPBuilder.Builder.CreateStore(MI.Size, Bitcast);

    Idx++;
  }

  OffloadingMappingArgs.Size = MapperInfos.size();
  OffloadingMappingArgs.BasePtrs =
      OMPBuilder.Builder.CreateBitCast(BasePtrsAlloca, OMPBuilder.VoidPtrPtr);
  OffloadingMappingArgs.Ptrs =
      OMPBuilder.Builder.CreateBitCast(PtrsAlloca, OMPBuilder.VoidPtrPtr);
  OffloadingMappingArgs.Sizes = OMPBuilder.Builder.CreateBitCast(
      SizesAlloca, OMPBuilder.SizeTy->getPointerTo());

  // OffloadingMappingArgs.BasePtrs = OMPBuilder.Builder.CreateInBoundsGEP(
  //     BasePtrsAlloca->getType()->getPointerElementType(), BasePtrsAlloca,
  //     {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
  // OffloadingMappingArgs.Ptrs = OMPBuilder.Builder.CreateInBoundsGEP(
  //     PtrsAlloca->getType()->getPointerElementType(), PtrsAlloca,
  //     {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
  // OffloadingMappingArgs.Sizes = OMPBuilder.Builder.CreateInBoundsGEP(
  //     SizesAlloca->getType()->getPointerElementType(), SizesAlloca,
  //     {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
}

void CGIntrinsicsOpenMP::emitOMPSingle(Function *Fn, BasicBlock *BBEntry,
                                       BasicBlock *AfterBB,
                                       BodyGenCallbackTy BodyGenCB,
                                       FinalizeCallbackTy FiniCB) {
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  BBEntry->getTerminator()->eraseFromParent();
  // Set the insertion location at the end of the BBEntry.
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  InsertPointTy AfterIP =
      OMPBuilder.createSingle(Loc, BodyGenCB, FiniCB, /*DidIt*/ nullptr);
  BranchInst::Create(AfterBB, AfterIP.getBlock());
  DEBUG_ENABLE(dbgs() << "=== Single Fn\n" << *Fn << "=== End of Single Fn\n");
}

void CGIntrinsicsOpenMP::emitOMPCritical(Function *Fn, BasicBlock *BBEntry,
                                         BasicBlock *AfterBB,
                                         BodyGenCallbackTy BodyGenCB,
                                         FinalizeCallbackTy FiniCB) {
  if (isOpenMPDeviceRuntime())
    FATAL_ERROR("Critical regions are not (yet) implemented on device");

  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  BBEntry->getTerminator()->eraseFromParent();
  // Set the insertion location at the end of the BBEntry.
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  InsertPointTy AfterIP = OMPBuilder.createCritical(Loc, BodyGenCB, FiniCB, "",
                                                    /*HintInst*/ nullptr);
  BranchInst::Create(AfterBB, AfterIP.getBlock());
  DEBUG_ENABLE(dbgs() << "=== Critical Fn\n"
                      << *Fn << "=== End of Critical Fn\n");
}

void CGIntrinsicsOpenMP::emitOMPBarrier(Function *Fn, BasicBlock *BBEntry,
                                        Directive DK) {
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  // Set the insertion location at the end of the BBEntry.
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);

  // TODO: check ForceSimpleCall usage.
  OMPBuilder.createBarrier(Loc, DK,
                           /*ForceSimpleCall*/ false,
                           /*CheckCancelFlag*/ true);
  DEBUG_ENABLE(dbgs() << "=== Barrier Fn\n"
                      << *Fn << "=== End of Barrier Fn\n");
}

void CGIntrinsicsOpenMP::emitOMPTaskwait(BasicBlock *BBEntry) {
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  // Set the insertion location at the end of the BBEntry.
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);

  OMPBuilder.createTaskwait(Loc);
}

GlobalVariable *
CGIntrinsicsOpenMP::emitOffloadingGlobals(StringRef DevWrapperFuncName,
                                          ConstantDataArray *ELF) {
  GlobalVariable *OMPRegionId = nullptr;
  GlobalVariable *OMPOffloadEntries = nullptr;

  // TODO: assumes 1 target region, can we call tgt_register_lib
  // multiple times?
  OMPRegionId = new GlobalVariable(
      M, OMPBuilder.Int8, /* isConstant */ true, GlobalValue::WeakAnyLinkage,
      ConstantInt::get(OMPBuilder.Int8, 0), DevWrapperFuncName + ".region_id",
      nullptr, GlobalVariable::NotThreadLocal,
      /* AddressSpace */ 0);

  Constant *OMPOffloadEntry;
  CGIntrinsicsOpenMP::emitOMPOffloadingEntry(DevWrapperFuncName, OMPRegionId,
                                             OMPOffloadEntry);

  // TODO: do this at finalization when all entries have been
  // found.
  // TODO: assumes 1 device image, can we call tgt_register_lib
  // multiple times?
  auto *ArrayTy = ArrayType::get(TgtOffloadEntryTy, 1);
  OMPOffloadEntries =
      new GlobalVariable(M, ArrayTy,
                         /* isConstant */ true, GlobalValue::InternalLinkage,
                         ConstantArray::get(ArrayTy, {OMPOffloadEntry}),
                         ".omp_offloading.entries");

  assert(OMPRegionId && "Expected non-null omp region id global");
  assert(OMPOffloadEntries &&
         "Expected non-null omp offloading entries constant");

  auto EmitOffloadingBinaryGlobals = [&]() {
    auto *GV = new GlobalVariable(M, ELF->getType(), /* isConstant */ true,
                                  GlobalValue::InternalLinkage, ELF,
                                  ".omp_offloading.device_image");
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    StructType *TgtDeviceImageTy = StructType::create(
        {OMPBuilder.Int8Ptr, OMPBuilder.Int8Ptr,
         TgtOffloadEntryTy->getPointerTo(), TgtOffloadEntryTy->getPointerTo()},
        "struct.__tgt_device_image");

    StructType *TgtBinDescTy = StructType::create(
        {OMPBuilder.Int32, TgtDeviceImageTy->getPointerTo(),
         TgtOffloadEntryTy->getPointerTo(), TgtOffloadEntryTy->getPointerTo()},
        "struct.__tgt_bin_desc");

    auto *ArrayTy = ArrayType::get(TgtDeviceImageTy, 1);
    auto *Zero = ConstantInt::get(OMPBuilder.SizeTy, 0);
    auto *One = ConstantInt::get(OMPBuilder.SizeTy, 1);
    auto *Size = ConstantInt::get(OMPBuilder.SizeTy, ELF->getNumElements());
    Constant *ZeroZero[] = {Zero, Zero};
    Constant *ZeroOne[] = {Zero, One};
    Constant *ZeroSize[] = {Zero, Size};

    auto *ImageB =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, ZeroZero);
    auto *ImageE =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, ZeroSize);
    auto *EntriesB = ConstantExpr::getGetElementPtr(
        OMPOffloadEntries->getValueType(), OMPOffloadEntries, ZeroZero);
    auto *EntriesE = ConstantExpr::getGetElementPtr(
        OMPOffloadEntries->getValueType(), OMPOffloadEntries, ZeroOne);

    auto *DeviceImageEntry = ConstantStruct::get(TgtDeviceImageTy, ImageB,
                                                 ImageE, EntriesB, EntriesE);
    auto *DeviceImages =
        new GlobalVariable(M, ArrayTy,
                           /* isConstant */ true, GlobalValue::InternalLinkage,
                           ConstantArray::get(ArrayTy, {DeviceImageEntry}),
                           ".omp_offloading.device_images");

    auto *ImagesB = ConstantExpr::getGetElementPtr(DeviceImages->getValueType(),
                                                   DeviceImages, ZeroZero);
    auto *DescInit =
        ConstantStruct::get(TgtBinDescTy,
                            ConstantInt::get(OMPBuilder.Int32,
                                             /* number of images */ 1),
                            ImagesB, EntriesB, EntriesE);
    auto *BinDesc =
        new GlobalVariable(M, DescInit->getType(),
                           /* isConstant */ true, GlobalValue::InternalLinkage,
                           DescInit, ".omp_offloading.descriptor");

    // Add tgt_register_requires, tgt_register_lib,
    // tgt_unregister_lib.
    {
      // tgt_register_requires.
      auto *FuncTy = FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
      auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                    ".omp_offloading.requires_reg", &M);
      Func->setSection(".text.startup");

      // Get __tgt_register_lib function declaration.
      auto *RegFuncTy = FunctionType::get(OMPBuilder.Void, OMPBuilder.Int64,
                                          /*isVarArg*/ false);
      FunctionCallee RegFuncC =
          M.getOrInsertFunction("__tgt_register_requires", RegFuncTy);

      // Construct function body
      IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", Func));
      // TODO: fix to pass the requirements enum value.
      Builder.CreateCall(RegFuncC, ConstantInt::get(OMPBuilder.Int64, 1));
      Builder.CreateRetVoid();

      // Add this function to constructors.
      // Set priority to 1 so that __tgt_register_lib is executed
      // AFTER
      // __tgt_register_requires (we want to know what requirements
      // have been asked for before we load a libomptarget plugin so
      // that by the time the plugin is loaded it can report how
      // many devices there are which can satisfy these
      // requirements).
      appendToGlobalCtors(M, Func, /*Priority*/ 0);
    }
    {
      // ctor
      auto *FuncTy = FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
      auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                    ".omp_offloading.descriptor_reg", &M);
      Func->setSection(".text.startup");

      // Get __tgt_register_lib function declaration.
      auto *RegFuncTy =
          FunctionType::get(OMPBuilder.Void, TgtBinDescTy->getPointerTo(),
                            /*isVarArg*/ false);
      FunctionCallee RegFuncC =
          M.getOrInsertFunction("__tgt_register_lib", RegFuncTy);

      // Construct function body
      IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", Func));
      Builder.CreateCall(RegFuncC, BinDesc);
      Builder.CreateRetVoid();

      // Add this function to constructors.
      // Set priority to 1 so that __tgt_register_lib is executed
      // AFTER
      // __tgt_register_requires (we want to know what requirements
      // have been asked for before we load a libomptarget plugin so
      // that by the time the plugin is loaded it can report how
      // many devices there are which can satisfy these
      // requirements).
      appendToGlobalCtors(M, Func, /*Priority*/ 1);
    }
    {
      auto *FuncTy = FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
      auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                    ".omp_offloading.descriptor_unreg", &M);
      Func->setSection(".text.startup");

      // Get __tgt_unregister_lib function declaration.
      auto *UnRegFuncTy =
          FunctionType::get(OMPBuilder.Void, TgtBinDescTy->getPointerTo(),
                            /*isVarArg*/ false);
      FunctionCallee UnRegFuncC =
          M.getOrInsertFunction("__tgt_unregister_lib", UnRegFuncTy);

      // Construct function body
      IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", Func));
      Builder.CreateCall(UnRegFuncC, BinDesc);
      Builder.CreateRetVoid();

      // Add this function to global destructors.
      // Match priority of __tgt_register_lib
      appendToGlobalDtors(M, Func, /*Priority*/ 1);
    }
  };

  EmitOffloadingBinaryGlobals();

  return OMPRegionId;
}

void CGIntrinsicsOpenMP::emitOMPTarget(Function *Fn, BasicBlock *EntryBB,
                                       BasicBlock *StartBB, BasicBlock *EndBB,
                                       DSAValueMapTy &DSAValueMap,
                                       StructMapTy &StructMappingInfoMap,
                                       TargetInfoStruct &TargetInfo,
                                       OMPLoopInfoStruct *OMPLoopInfo,
                                       bool IsDeviceTargetRegion) {
  if (IsDeviceTargetRegion)
    emitOMPTargetDevice(Fn, EntryBB, StartBB, EndBB, DSAValueMap,
                        StructMappingInfoMap, TargetInfo);
  else
    emitOMPTargetHost(Fn, EntryBB, StartBB, EndBB, DSAValueMap,
                      StructMappingInfoMap, TargetInfo, OMPLoopInfo);
}

void CGIntrinsicsOpenMP::emitOMPTargetHost(
    Function *Fn, BasicBlock *EntryBB, BasicBlock *StartBB, BasicBlock *EndBB,
    DSAValueMapTy &DSAValueMap, StructMapTy &StructMappingInfoMap,
    TargetInfoStruct &TargetInfo, OMPLoopInfoStruct *OMPLoopInfo) {

  Twine DevWrapperFuncName = getDevWrapperFuncPrefix() + TargetInfo.DevFuncName;

  GlobalVariable *OMPRegionId =
      emitOffloadingGlobals(DevWrapperFuncName.str(), TargetInfo.ELF);

  const DebugLoc DL = EntryBB->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(EntryBB, EntryBB->getTerminator()->getIterator()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  // TODO: should we use target_mapper without teams or the more general
  // target_teams_mapper. Does the former buy us anything (less overhead?)
  // FunctionCallee TargetMapper =
  //    OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___tgt_target_mapper);
  // TODO: For nowait we need to enclose the host code in a task for async
  // execution.
  FunctionCallee TargetMapper =
      (TargetInfo.NoWait ? OMPBuilder.getOrCreateRuntimeFunction(
                               M, OMPRTL___tgt_target_teams_nowait_mapper)
                         : OMPBuilder.getOrCreateRuntimeFunction(
                               M, OMPRTL___tgt_target_teams_mapper));
  OMPBuilder.Builder.SetInsertPoint(EntryBB->getTerminator());

  // Emit mappings.
  OffloadingMappingArgsTy OffloadingMappingArgs;
  InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                         Fn->getEntryBlock().getFirstInsertionPt());
  emitOMPOffloadingMappings(AllocaIP, DSAValueMap, StructMappingInfoMap,
                            OffloadingMappingArgs, /* isTargetRegion */ true);

  // Push the tripcount.
  if (OMPLoopInfo) {
    FunctionCallee TripcountMapper = OMPBuilder.getOrCreateRuntimeFunction(
        M,
        llvm::omp::RuntimeFunction::OMPRTL___kmpc_push_target_tripcount_mapper);
    Value *Load =
        OMPBuilder.Builder.CreateLoad(OMPBuilder.Int64, OMPLoopInfo->UB);
    Value *Tripcount = OMPBuilder.Builder.CreateAdd(
        Load, ConstantInt::get(OMPBuilder.Int64, 1));
    auto *CI = checkCreateCall(
        OMPBuilder.Builder, TripcountMapper,
        {Ident, ConstantInt::get(OMPBuilder.Int64, -1), Tripcount});
    assert(CI && "Expected valid call");
  }

  Value *NumTeams = createScalarCast(TargetInfo.NumTeams, OMPBuilder.Int32);
  Value *ThreadLimit =
      createScalarCast(TargetInfo.ThreadLimit, OMPBuilder.Int32);

  assert(NumTeams && "Expected non-null NumTeams");
  assert(ThreadLimit && "Expected non-null ThreadLimit");

  SmallVector<Value *, 16> Args = {
      Ident, ConstantInt::get(OMPBuilder.Int64, -1),
      ConstantExpr::getBitCast(OMPRegionId, OMPBuilder.VoidPtr),
      ConstantInt::get(OMPBuilder.Int32, OffloadingMappingArgs.Size),
      OffloadingMappingArgs.BasePtrs, OffloadingMappingArgs.Ptrs,
      OffloadingMappingArgs.Sizes, OffloadingMappingArgs.MapTypes,
      OffloadingMappingArgs.MapNames,
      // TODO: offload_mappers is null for now.
      Constant::getNullValue(OMPBuilder.VoidPtrPtr), NumTeams, ThreadLimit};

  if (TargetInfo.NoWait) {
    // Add extra dependency information (unused for now).
    Args.push_back(Constant::getNullValue(OMPBuilder.Int32));
    Args.push_back(Constant::getNullValue(OMPBuilder.Int8Ptr));
    Args.push_back(Constant::getNullValue(OMPBuilder.Int32));
    Args.push_back(Constant::getNullValue(OMPBuilder.Int8Ptr));
  }

  auto *OffloadResult = checkCreateCall(OMPBuilder.Builder, TargetMapper, Args);
  assert(OffloadResult && "Expected non-null call inst from code generation");
  auto *Failed = OMPBuilder.Builder.CreateIsNotNull(OffloadResult);
  OMPBuilder.Builder.CreateCondBr(Failed, StartBB, EndBB);
  EntryBB->getTerminator()->eraseFromParent();
}

void CGIntrinsicsOpenMP::emitOMPTargetDevice(Function *Fn, BasicBlock *EntryBB,
                                             BasicBlock *StartBB,
                                             BasicBlock *EndBB,
                                             DSAValueMapTy &DSAValueMap,
                                             StructMapTy &StructMappingInfoMap,
                                             TargetInfoStruct &TargetInfo) {
  // Emit the Numba wrapper offloading function.
  SmallVector<Type *, 8> WrapperArgsTypes;
  SmallVector<StringRef, 8> WrapperArgsNames;
  for (auto &It : DSAValueMap) {
    Value *V = It.first;
    DSAType DSA = It.second.Type;

    DEBUG_ENABLE(dbgs() << "V " << *V << " DSA " << DSA << "\n");
    switch (DSA) {
    case DSA_FIRSTPRIVATE:
      // TODO: Runtime expects firstprivate (scalars) typed as Int64.
      WrapperArgsTypes.push_back(OMPBuilder.Int64);
      WrapperArgsNames.push_back(V->getName());
      break;
    case DSA_PRIVATE:
      // do nothing
      break;
    default:
      WrapperArgsTypes.push_back(V->getType());
      WrapperArgsNames.push_back(V->getName());
    }
  }

  Twine DevWrapperFuncName = getDevWrapperFuncPrefix() + Fn->getName();
  FunctionType *NumbaWrapperFnTy =
      FunctionType::get(OMPBuilder.Void, WrapperArgsTypes,
                        /* isVarArg */ false);
  Function *NumbaWrapperFunc = Function::Create(
      NumbaWrapperFnTy, GlobalValue::ExternalLinkage, DevWrapperFuncName, M);

  // Name the wrapper arguments for readability.
  for (size_t I = 0; I < NumbaWrapperFunc->arg_size(); ++I)
    NumbaWrapperFunc->getArg(I)->setName(WrapperArgsNames[I]);

  IRBuilder<> Builder(
      BasicBlock::Create(M.getContext(), "entry", NumbaWrapperFunc));
  // Set up default arguments. Depends on the target architecture.
  FunctionCallee DevFuncCallee(Fn);
  // Set the callee device function with internal linkage to enable
  // optimization.
  Fn->setLinkage(GlobalValue::InternalLinkage);
  SmallVector<Value *, 8> DevFuncArgs;
  Triple TargetTriple(M.getTargetTriple());

  // Adapt arguments to the Numba calling convention depending on target. First
  // two arguments are Numba-generated pointers for return value and exceptions
  // (if targeting the CPU), which are unused. Init to nullptr.
  size_t ArgOffset;
  DevFuncArgs.push_back(Constant::getNullValue(Fn->getArg(0)->getType()));
  if (!isOpenMPDeviceRuntime()) {
    DevFuncArgs.push_back(Constant::getNullValue(Fn->getArg(1)->getType()));
    ArgOffset = 2;
  } else {
    ArgOffset = 1;
  }
  for (auto &Arg : NumbaWrapperFunc->args()) {
    // TODO: Runtime expects all scalars typed as Int64.
    if (!Arg.getType()->isPointerTy()) {
      auto *ParamType = DevFuncCallee.getFunctionType()->getParamType(
          ArgOffset + Arg.getArgNo());
      AllocaInst *TmpInt64 = Builder.CreateAlloca(OMPBuilder.Int64, nullptr,
                                                  Arg.getName() + ".casted");
      Builder.CreateStore(&Arg, TmpInt64);
      Value *Cast = Builder.CreateBitCast(TmpInt64, ParamType->getPointerTo());
      Value *ConvLoad = Builder.CreateLoad(ParamType, Cast);
      DevFuncArgs.push_back(ConvLoad);
    } else
      DevFuncArgs.push_back(&Arg);
  }

  bool IsSPMD = (TargetInfo.ExecMode == omp::OMP_TGT_EXEC_MODE_SPMD);
  if (isOpenMPDeviceRuntime()) {
    OpenMPIRBuilder::LocationDescription Loc(Builder);
    auto IP = OMPBuilder.createTargetInit(Loc, /* IsSPMD */ IsSPMD,
                                          /* RequiresFullRuntime */ false);
    Builder.restoreIP(IP);
  }

  auto *CI = checkCreateCall(Builder, DevFuncCallee, DevFuncArgs);
  assert(CI && "Expected valid call");

  if (isOpenMPDeviceRuntime()) {
    OpenMPIRBuilder::LocationDescription Loc(Builder);
    OMPBuilder.createTargetDeinit(Loc, /* IsSPMD */ IsSPMD,
                                  /* RequiresFullRuntime */ false);
  }

  Builder.CreateRetVoid();

  if (isOpenMPDeviceRuntime()) {
    assert(TargetInfo.ExecMode && "Expected non-zero ExecMode");
    // Emit OMP device globals and metadata.
    // TODO: Make the exec_mode a parameter and use SPMD when possible.
    auto *ExecModeGV = new GlobalVariable(
        M, OMPBuilder.Int8, /* isConstant */ false, GlobalValue::WeakAnyLinkage,
        Builder.getInt8(TargetInfo.ExecMode),
        DevWrapperFuncName + "_exec_mode");
    appendToCompilerUsed(M, {ExecModeGV});

    // Get "nvvm.annotations" metadata node.
    // TODO: may need to adjust for AMD gpus.
    NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

    Metadata *MDVals[] = {
        ConstantAsMetadata::get(NumbaWrapperFunc),
        MDString::get(M.getContext(), "kernel"),
        ConstantAsMetadata::get(ConstantInt::get(OMPBuilder.Int32, 1))};
    // Append metadata to nvvm.annotations.
    MD->addOperand(MDNode::get(M.getContext(), MDVals));

    // Add a function attribute for the kernel.
    NumbaWrapperFunc->addFnAttr(Attribute::get(M.getContext(), "kernel"));

  } else {
    // Generating an offloading entry is required by the x86_64 plugin.
    Constant *OMPOffloadEntry;
    emitOMPOffloadingEntry(DevWrapperFuncName, NumbaWrapperFunc,
                           OMPOffloadEntry);
  }
  // Add llvm.module.flags for "openmp", "openmp-device" to enable
  // OpenMPOpt.
  M.addModuleFlag(llvm::Module::Max, "openmp", 50);
  M.addModuleFlag(llvm::Module::Max, "openmp-device", 50);
}

void CGIntrinsicsOpenMP::emitOMPTeamsDeviceRuntime(
    DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap, const DebugLoc &DL,
    Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
    BasicBlock *AfterBB, TeamsInfoStruct &TeamsInfo) {
  SmallVector<Value *, 16> CapturedVars;
  OutlinedInfoStruct OI = createOutlinedFunction(
      DSAValueMap, VMap, Fn, StartBB, EndBB, CapturedVars,
      ".omp_outlined_teams", omp::Directive::OMPD_teams);

  if (!OI.ReductionInfos.empty())
    emitReductionsDevice(
        InsertPointTy(OI.ExitBB, OI.ExitBB->begin()),
        InsertPointTy(OI.EntryBB, OI.EntryBB->begin()), OI.ReductionInfos,
        (TeamsInfo.ExecMode == OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD));

  Function *OutlinedFn = OI.Fn;

  // Set up the call to the teams outlined function.
  BBEntry->getTerminator()->eraseFromParent();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  OMPBuilder.Builder.restoreIP(Loc.IP);
  OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);
  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

  assert(Ident && "Expected non-null Ident");
  assert(ThreadID && "Expected non-null ThreadID");

  // Create global_tid, bound_tid (zero) to pass to the teams outlined function.
  AllocaInst *ThreadIDAddr = OMPBuilder.Builder.CreateAlloca(
      OMPBuilder.Int32, nullptr, ".threadid.addr");
  AllocaInst *ZeroAddr =
      OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, ".zero.addr");
  OMPBuilder.Builder.CreateStore(ThreadID, ThreadIDAddr);
  OMPBuilder.Builder.CreateStore(Constant::getNullValue(OMPBuilder.Int32),
                                 ZeroAddr);

  FunctionCallee TeamsOutlinedFn(OutlinedFn);
  SmallVector<Value *, 8> Args;
  Args.append({ThreadIDAddr, ZeroAddr});

  for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
    // Pass firstprivate scalar by value.
    if (DSAValueMap[CapturedVars[Idx]].Type == DSA_FIRSTPRIVATE &&
        CapturedVars[Idx]
            ->getType()
            ->getPointerElementType()
            ->isSingleValueType()) {
      Type *VPtrElemTy = CapturedVars[Idx]->getType()->getPointerElementType();
      Value *Load =
          OMPBuilder.Builder.CreateLoad(VPtrElemTy, CapturedVars[Idx]);
      // TODO: Runtime expects values in Int64 type, fix with arguments in
      // struct.
      AllocaInst *TmpInt64 = OMPBuilder.Builder.CreateAlloca(
          OMPBuilder.Int64, nullptr,
          CapturedVars[Idx]->getName() + "fpriv.byval");
      Value *Cast = OMPBuilder.Builder.CreateBitCast(
          TmpInt64, CapturedVars[Idx]->getType());
      OMPBuilder.Builder.CreateStore(Load, Cast);
      Value *ConvLoad =
          OMPBuilder.Builder.CreateLoad(OMPBuilder.Int64, TmpInt64);
      Args.push_back(ConvLoad);

      continue;
    }
    Args.push_back(CapturedVars[Idx]);
  }

  auto *CI = checkCreateCall(OMPBuilder.Builder, TeamsOutlinedFn, Args);
  assert(CI && "Expected valid call");

  OMPBuilder.Builder.CreateBr(AfterBB);

  DEBUG_ENABLE(dbgs() << "=== Dump OuterFn\n"
                      << *Fn << "=== End of Dump OuterFn\n");

  if (verifyFunction(*Fn, &errs()))
    FATAL_ERROR("Verification of OuterFn failed!");
}

void CGIntrinsicsOpenMP::emitOMPTeams(DSAValueMapTy &DSAValueMap,
                                      ValueToValueMapTy *VMap,
                                      const DebugLoc &DL, Function *Fn,
                                      BasicBlock *BBEntry, BasicBlock *StartBB,
                                      BasicBlock *EndBB, BasicBlock *AfterBB,
                                      TeamsInfoStruct &TeamsInfo) {
  if (isOpenMPDeviceRuntime())
    emitOMPTeamsDeviceRuntime(DSAValueMap, VMap, DL, Fn, BBEntry, StartBB,
                              EndBB, AfterBB, TeamsInfo);
  else
    emitOMPTeamsHostRuntime(DSAValueMap, VMap, DL, Fn, BBEntry, StartBB, EndBB,
                            AfterBB, TeamsInfo);
}

void CGIntrinsicsOpenMP::emitOMPTeamsHostRuntime(
    DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap, const DebugLoc &DL,
    Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
    BasicBlock *AfterBB, TeamsInfoStruct &TeamsInfo) {
  SmallVector<Value *, 16> CapturedVars;
  OutlinedInfoStruct OI = createOutlinedFunction(
      DSAValueMap, /*ValueToValueMapTy */ VMap, Fn, StartBB, EndBB,
      CapturedVars, ".omp_outlined_teams", omp::Directive::OMPD_teams);

  if (!OI.ReductionInfos.empty())
    emitReductionsHost(InsertPointTy(OI.ExitBB, OI.ExitBB->begin()),
                       InsertPointTy(OI.EntryBB, OI.EntryBB->begin()),
                       OI.ReductionInfos);

  Function *OutlinedFn = OI.Fn;

  // Set up the call to the teams outlined function.
  BBEntry->getTerminator()->eraseFromParent();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  OMPBuilder.Builder.restoreIP(Loc.IP);
  OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);
  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

  assert(Ident && "Expected non-null Ident");
  // Emit call to set the number of teams and thread limit.
  if (TeamsInfo.NumTeams || TeamsInfo.ThreadLimit) {
    Value *NumTeams =
        (TeamsInfo.NumTeams
             ? createScalarCast(TeamsInfo.NumTeams, OMPBuilder.Int32)
             : Constant::getNullValue(OMPBuilder.Int32));
    Value *ThreadLimit =
        (TeamsInfo.ThreadLimit
             ? createScalarCast(TeamsInfo.ThreadLimit, OMPBuilder.Int32)
             : Constant::getNullValue(OMPBuilder.Int32));
    FunctionCallee KmpcPushNumTeams =
        OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_push_num_teams);
    auto *CI = checkCreateCall(OMPBuilder.Builder, KmpcPushNumTeams,
                               {Ident, ThreadID, NumTeams, ThreadLimit});
    assert(CI && "Expected valid call");
  }

  FunctionCallee ForkTeams =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_fork_teams);

  SmallVector<Value *, 8> Args;
  Value *NumCapturedVars = OMPBuilder.Builder.getInt32(CapturedVars.size());
  Args.append({Ident, NumCapturedVars,
               OMPBuilder.Builder.CreateBitCast(OutlinedFn,
                                                OMPBuilder.ParallelTaskPtr)});

  for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
    // Pass firstprivate scalar by value.
    if (DSAValueMap[CapturedVars[Idx]].Type == DSA_FIRSTPRIVATE &&
        CapturedVars[Idx]
            ->getType()
            ->getPointerElementType()
            ->isSingleValueType()) {
      Type *VPtrElemTy = CapturedVars[Idx]->getType()->getPointerElementType();
      Value *Load =
          OMPBuilder.Builder.CreateLoad(VPtrElemTy, CapturedVars[Idx]);
      // TODO: Runtime expects values in Int64 type, fix with arguments in
      // struct.
      AllocaInst *TmpInt64 = OMPBuilder.Builder.CreateAlloca(
          OMPBuilder.Int64, nullptr,
          CapturedVars[Idx]->getName() + ".fpriv.byval");
      Value *Cast = OMPBuilder.Builder.CreateBitCast(
          TmpInt64, CapturedVars[Idx]->getType());
      OMPBuilder.Builder.CreateStore(Load, Cast);
      Value *ConvLoad =
          OMPBuilder.Builder.CreateLoad(OMPBuilder.Int64, TmpInt64);
      Args.push_back(ConvLoad);

      continue;
    }
    Args.push_back(CapturedVars[Idx]);
  }

  auto *CI = checkCreateCall(OMPBuilder.Builder, ForkTeams, Args);
  assert(CI && "Expected valid call");

  OMPBuilder.Builder.CreateBr(AfterBB);

  DEBUG_ENABLE(dbgs() << "=== Dump OuterFn\n"
                      << *Fn << "=== End of Dump OuterFn\n");

  if (verifyFunction(*Fn, &errs()))
    FATAL_ERROR("Verification of OuterFn failed!");
}

void CGIntrinsicsOpenMP::emitOMPTargetEnterData(
    Function *Fn, BasicBlock *BBEntry, DSAValueMapTy &DSAValueMap,
    StructMapTy &StructMappingInfoMap) {

  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  FunctionCallee TargetDataBeginMapper = OMPBuilder.getOrCreateRuntimeFunction(
      M, OMPRTL___tgt_target_data_begin_mapper);
  OMPBuilder.Builder.SetInsertPoint(BBEntry->getTerminator());

  // Emit mappings.
  OffloadingMappingArgsTy OffloadingMappingArgs;
  InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                         Fn->getEntryBlock().getFirstInsertionPt());
  emitOMPOffloadingMappings(AllocaIP, DSAValueMap, StructMappingInfoMap,
                            OffloadingMappingArgs, /* IsTargetRegion */ false);

  OMPBuilder.Builder.CreateCall(
      TargetDataBeginMapper,
      {SrcLoc, ConstantInt::get(OMPBuilder.Int64, -1),
       ConstantInt::get(OMPBuilder.Int32, OffloadingMappingArgs.Size),
       OffloadingMappingArgs.BasePtrs, OffloadingMappingArgs.Ptrs,
       OffloadingMappingArgs.Sizes, OffloadingMappingArgs.MapTypes,
       OffloadingMappingArgs.MapNames,
       // TODO: offload_mappers is null for now.
       Constant::getNullValue(OMPBuilder.VoidPtrPtr)});
}

void CGIntrinsicsOpenMP::emitOMPTargetExitData(
    Function *Fn, BasicBlock *BBEntry, DSAValueMapTy &DSAValueMap,
    StructMapTy &StructMappingInfoMap) {

  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  FunctionCallee TargetDataEndMapper = OMPBuilder.getOrCreateRuntimeFunction(
      M, OMPRTL___tgt_target_data_end_mapper);
  OMPBuilder.Builder.SetInsertPoint(BBEntry->getTerminator());

  // Emit mappings.
  OffloadingMappingArgsTy OffloadingMappingArgs;
  InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                         Fn->getEntryBlock().getFirstInsertionPt());
  emitOMPOffloadingMappings(AllocaIP, DSAValueMap, StructMappingInfoMap,
                            OffloadingMappingArgs, /* IsTargetRegion */ false);

  OMPBuilder.Builder.CreateCall(
      TargetDataEndMapper,
      {SrcLoc, ConstantInt::get(OMPBuilder.Int64, -1),
       ConstantInt::get(OMPBuilder.Int32, OffloadingMappingArgs.Size),
       OffloadingMappingArgs.BasePtrs, OffloadingMappingArgs.Ptrs,
       OffloadingMappingArgs.Sizes, OffloadingMappingArgs.MapTypes,
       OffloadingMappingArgs.MapNames,
       // TODO: offload_mappers is null for now.
       Constant::getNullValue(OMPBuilder.VoidPtrPtr)});
}

void CGIntrinsicsOpenMP::emitOMPTargetData(Function *Fn, BasicBlock *BBEntry,
                                           BasicBlock *BBExit,
                                           DSAValueMapTy &DSAValueMap,
                                           StructMapTy &StructMappingInfoMap) {
  // Re-use codegen from TARGET ENTER/EXIT DATA.
  emitOMPTargetEnterData(Fn, BBEntry, DSAValueMap, StructMappingInfoMap);
  emitOMPTargetExitData(Fn, BBExit, DSAValueMap, StructMappingInfoMap);
}

void CGIntrinsicsOpenMP::emitOMPTargetUpdate(
    Function *Fn, BasicBlock *BBEntry, DSAValueMapTy &DSAValueMap,
    StructMapTy &StructMappingInfoMap) {
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  FunctionCallee TargetDataUpdateMapper = OMPBuilder.getOrCreateRuntimeFunction(
      M, OMPRTL___tgt_target_data_update_mapper);
  OMPBuilder.Builder.SetInsertPoint(BBEntry->getTerminator());

  // Emit mappings.
  OffloadingMappingArgsTy OffloadingMappingArgs;
  InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                         Fn->getEntryBlock().getFirstInsertionPt());
  emitOMPOffloadingMappings(AllocaIP, DSAValueMap, StructMappingInfoMap,
                            OffloadingMappingArgs, /* IsTargetRegion */ false);

  OMPBuilder.Builder.CreateCall(
      TargetDataUpdateMapper,
      {SrcLoc, ConstantInt::get(OMPBuilder.Int64, -1),
       ConstantInt::get(OMPBuilder.Int32, OffloadingMappingArgs.Size),
       OffloadingMappingArgs.BasePtrs, OffloadingMappingArgs.Ptrs,
       OffloadingMappingArgs.Sizes, OffloadingMappingArgs.MapTypes,
       OffloadingMappingArgs.MapNames,
       // TODO: offload_mappers is null for now.
       Constant::getNullValue(OMPBuilder.VoidPtrPtr)});
}

void CGIntrinsicsOpenMP::emitOMPDistribute(
    DSAValueMapTy &DSAValueMap, OMPLoopInfoStruct &OMPLoopInfo,
    BasicBlock *StartBB, BasicBlock *ExitBB, bool IsStandalone,
    bool IsDistributeParallelFor, OMPDistributeInfoStruct *DistributeInfo) {
  if (static_cast<int>(OMPLoopInfo.Sched) == 0)
    OMPLoopInfo.Sched = OMPScheduleType::Distribute;

  emitLoop(DSAValueMap, OMPLoopInfo, StartBB, ExitBB, IsStandalone, true,
           IsDistributeParallelFor, DistributeInfo);
}

void CGIntrinsicsOpenMP::emitOMPDistributeParallelFor(
    DSAValueMapTy &DSAValueMap, BasicBlock *StartBB, BasicBlock *ExitBB,
    OMPLoopInfoStruct &OMPLoopInfo, ParRegionInfoStruct &ParRegionInfo,
    bool IsStandalone) {

  Function *Fn = StartBB->getParent();
  const DebugLoc DL = StartBB->getTerminator()->getDebugLoc();

  BasicBlock *DistPreheader =
      StartBB->splitBasicBlock(StartBB->begin(), "omp.distribute.preheader");
  BasicBlock *DistHeader = DistPreheader->splitBasicBlock(
      DistPreheader->begin(), "omp.distribute.header");
  BasicBlock *ForEntry =
      DistHeader->splitBasicBlock(DistHeader->begin(), "omp.inner.for.entry");
  BasicBlock *ForBegin =
      ForEntry->splitBasicBlock(ForEntry->begin(), "omp.inner.for.begin");
  BasicBlock *ForEnd = splitBlockBefore(
      ExitBB, &*ExitBB->getFirstInsertionPt(), /*DomTreeUpdater*/ nullptr,
      /*LoopInfo*/ nullptr, /*MemorySSAUpdater*/ nullptr);
  ForEnd->setName("omp.inner.for.end");
  BasicBlock *ForExit = SplitBlock(ForEnd, ForEnd->getTerminator());
  ForExit->setName("omp.inner.for.exit");
  BasicBlock *ForExitAfter = SplitBlock(ForExit, ForExit->getTerminator());
  ForExitAfter->setName("omp.inner.for.exit.after");
  BasicBlock *DistInc = ForExitAfter->splitBasicBlock(
      ForExitAfter->getTerminator(), "omp.distribute.inc");
  BasicBlock *DistExit =
      DistInc->splitBasicBlock(DistInc->getTerminator(), "omp.distribute.exit");

  // Create skeleton DistHeader
  {
    // Dummy condition to create the expected structure.
    DistHeader->getTerminator()->eraseFromParent();
    OMPBuilder.Builder.SetInsertPoint(DistHeader);
    auto *Cond =
        OMPBuilder.Builder.CreateICmpSLE(OMPLoopInfo.IV, OMPLoopInfo.UB);
    OMPBuilder.Builder.CreateCondBr(Cond, ForEntry, DistExit);
  }
  // Create skeleton DistInc
  {
    DistInc->getTerminator()->eraseFromParent();
    OMPBuilder.Builder.SetInsertPoint(DistInc);
    OMPBuilder.Builder.CreateBr(DistHeader);
  }

  OMPLoopInfo.Sched = (isOpenMPDeviceRuntime() ? OMPScheduleType::StaticChunked
                                               : OMPScheduleType::Static);
  emitOMPFor(DSAValueMap, OMPLoopInfo, ForBegin, ForEnd, IsStandalone, true);
  BasicBlock *ParEntryBB = ForEntry;
  DEBUG_ENABLE(dbgs() << "ParEntryBB " << ParEntryBB->getName() << "\n");
  BasicBlock *ParStartBB = ForBegin;
  DEBUG_ENABLE(dbgs() << "ParStartBB " << ParStartBB->getName() << "\n");
  BasicBlock *ParEndBB = ForExit;
  DEBUG_ENABLE(dbgs() << "ParEndBB " << ParEndBB->getName() << "\n");
  BasicBlock *ParAfterBB = ForExitAfter;
  DEBUG_ENABLE(dbgs() << "ParAfterBB " << ParAfterBB->getName() << "\n");

  emitOMPParallel(
      DSAValueMap, nullptr, DL, Fn, ParEntryBB, ParStartBB, ParEndBB,
      ParAfterBB, [](auto) {}, ParRegionInfo);

  // By default, to maximize performance on GPUs, we do static chunked with a
  // chunk size equal to the block size when targeting the device runtime.
  if (isOpenMPDeviceRuntime()) {
    OMPLoopInfo.Sched = OMPScheduleType::DistributeChunked;
    // Extend DistPreheader
    {
      OMPBuilder.Builder.SetInsertPoint(DistPreheader,
                                        DistPreheader->getFirstInsertionPt());

      FunctionCallee NumTeamThreadsFn = OMPBuilder.getOrCreateRuntimeFunction(
          M, llvm::omp::RuntimeFunction::
                 OMPRTL___kmpc_get_hardware_num_threads_in_block);
      Value *NumTeamThreads =
          OMPBuilder.Builder.CreateCall(NumTeamThreadsFn, {});
      OMPLoopInfo.Chunk = NumTeamThreads;
    }
  } else {
    OMPLoopInfo.Sched = OMPScheduleType::Distribute;
  }

  OMPDistributeInfoStruct DistributeInfo;
  emitOMPDistribute(DSAValueMap, OMPLoopInfo, DistPreheader, DistExit,
                    IsStandalone, true, &DistributeInfo);

  // Replace upper bound, lower bound to the "parallel for" with distribute
  // bounds.
  {
    assert(DistributeInfo.LB && "Expected non-null distribute lower bound");
    assert(DistributeInfo.UB && "Expected non-null distribute upper bound");
    auto ShouldReplace = [&](Use &U) {
      if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
        if (UserI->getParent() == ForEntry)
          return true;

      return false;
    };

    // Replace the inner, parallel for loop LB, UB.
    OMPLoopInfo.LB->replaceUsesWithIf(DistributeInfo.LB, ShouldReplace);
    OMPLoopInfo.UB->replaceUsesWithIf(DistributeInfo.UB, ShouldReplace);
  }
}

void CGIntrinsicsOpenMP::emitOMPTargetTeams(
    DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap, const DebugLoc &DL,
    Function *Fn, BasicBlock *EntryBB, BasicBlock *StartBB, BasicBlock *EndBB,
    BasicBlock *AfterBB, TargetInfoStruct &TargetInfo,
    TeamsInfoStruct &TeamsInfo, OMPLoopInfoStruct *OMPLoopInfo,
    StructMapTy &StructMappingInfoMap, bool IsDeviceTargetRegion) {

  BasicBlock *TeamsEntryBB = SplitBlock(EntryBB, EntryBB->getTerminator());
  TeamsEntryBB->setName("omp.teams.entry");
  BasicBlock *TeamsStartBB =
      splitBlockBefore(StartBB, &*StartBB->getFirstInsertionPt(), nullptr,
                       nullptr, nullptr, "omp.teams.start");
  BasicBlock *TeamsEndBB =
      splitBlockBefore(EndBB, &*EndBB->getFirstInsertionPt(), nullptr, nullptr,
                       nullptr, "omp.teams.end");

  emitOMPTeams(DSAValueMap, VMap, DL, Fn, TeamsEntryBB, TeamsStartBB,
               TeamsEndBB, EndBB, TeamsInfo);

  emitOMPTarget(Fn, EntryBB, TeamsEntryBB, EndBB, DSAValueMap,
                StructMappingInfoMap, TargetInfo, OMPLoopInfo,
                IsDeviceTargetRegion);
}

bool CGIntrinsicsOpenMP::isOpenMPDeviceRuntime() {
  Triple TargetTriple(M.getTargetTriple());

  if (TargetTriple.isNVPTX())
    return true;

  return false;
}

template <>
Value *CGReduction::emitOperation<DSA_REDUCTION_ADD>(IRBuilderBase &IRB,
                                                     Value *LHS, Value *RHS) {
  Type *VTy = RHS->getType();
  if (VTy->isIntegerTy())
    return IRB.CreateAdd(LHS, RHS, "red.add");
  else if (VTy->isFloatTy() || VTy->isDoubleTy())
    return IRB.CreateFAdd(LHS, RHS, "red.add");
  else
    FATAL_ERROR("Unsupported type for reduction operation");
}

// OpenMP 5.1, 2.21.5, sub is the same as add.
template <>
Value *CGReduction::emitOperation<DSA_REDUCTION_SUB>(IRBuilderBase &IRB,
                                                     Value *LHS, Value *RHS) {
  return emitOperation<DSA_REDUCTION_ADD>(IRB, LHS, RHS);
}

template <>
Value *CGReduction::emitOperation<DSA_REDUCTION_MUL>(IRBuilderBase &IRB,
                                                     Value *LHS, Value *RHS) {
  Type *VTy = RHS->getType();
  if (VTy->isIntegerTy())
    return IRB.CreateMul(LHS, RHS, "red.mul");
  else if (VTy->isFloatTy() || VTy->isDoubleTy())
    return IRB.CreateFMul(LHS, RHS, "red.mul");
  else
    FATAL_ERROR("Unsupported type for reduction operation");
}

template <>
InsertPointTy CGReduction::emitAtomicOperationRMW<DSA_REDUCTION_ADD>(
    IRBuilderBase &IRB, Value *LHS, Value *Partial) {
  IRB.CreateAtomicRMW(AtomicRMWInst::Add, LHS, Partial, None,
                      AtomicOrdering::Monotonic);
  return IRB.saveIP();
}

// OpenMP 5.1, 2.21.5, sub is the same as add.
template <>
InsertPointTy CGReduction::emitAtomicOperationRMW<DSA_REDUCTION_SUB>(
    IRBuilderBase &IRB, Value *LHS, Value *Partial) {
  return emitAtomicOperationRMW<DSA_REDUCTION_ADD>(IRB, LHS, Partial);
}
