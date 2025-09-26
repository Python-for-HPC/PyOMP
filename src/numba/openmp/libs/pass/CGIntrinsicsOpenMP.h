#ifndef LLVM_TRANSFORMS_INTRINSICS_OPENMP_CODEGEN_H
#define LLVM_TRANSFORMS_INTRINSICS_OPENMP_CODEGEN_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/AtomicOrdering.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "DebugOpenMP.h"

using namespace llvm;
using namespace omp;

using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
using BodyGenCallbackTy = OpenMPIRBuilder::BodyGenCallbackTy;
using FinalizeCallbackTy = OpenMPIRBuilder::FinalizeCallbackTy;

namespace iomp {
// TODO: expose clauses through namespace omp?
enum DSAType {
  DSA_NONE,
  DSA_PRIVATE,
  DSA_FIRSTPRIVATE,
  DSA_LASTPRIVATE,
  DSA_SHARED,
  DSA_REDUCTION_ADD,
  DSA_REDUCTION_SUB,
  DSA_REDUCTION_MUL,
  DSA_MAP_ALLOC,
  DSA_MAP_TO,
  DSA_MAP_FROM,
  DSA_MAP_TOFROM,
  DSA_MAP_ALLOC_STRUCT,
  DSA_MAP_TO_STRUCT,
  DSA_MAP_FROM_STRUCT,
  DSA_MAP_TOFROM_STRUCT,
  DSA_MAP_STRUCT
};

struct DSATypeInfo {
  DSAType Type;
  FunctionCallee CopyConstructor;

  DSATypeInfo() : Type(DSA_NONE), CopyConstructor(nullptr) {}
  DSATypeInfo(DSAType InType) : Type(InType), CopyConstructor(nullptr) {}
  DSATypeInfo(DSAType InType, FunctionCallee InCopyConstructor)
      : Type(InType), CopyConstructor(InCopyConstructor) {}
  DSATypeInfo(const DSATypeInfo &DTI) {
    Type = DTI.Type;
    CopyConstructor = DTI.CopyConstructor;
  }
  DSATypeInfo &operator=(const DSATypeInfo &DTI) = default;
};

using DSAValueMapTy = MapVector<Value *, DSATypeInfo>;

// using DSAValueMapTy = MapVector<Value *, DSAType>;

static const DenseMap<StringRef, Directive> StringToDir = {
    {"DIR.OMP.PARALLEL", OMPD_parallel},
    {"DIR.OMP.SINGLE", OMPD_single},
    {"DIR.OMP.CRITICAL", OMPD_critical},
    {"DIR.OMP.BARRIER", OMPD_barrier},
    {"DIR.OMP.LOOP", OMPD_for},
    {"DIR.OMP.PARALLEL.LOOP", OMPD_parallel_for},
    {"DIR.OMP.TASK", OMPD_task},
    {"DIR.OMP.TASKWAIT", OMPD_taskwait},
    {"DIR.OMP.TARGET", OMPD_target},
    {"DIR.OMP.TEAMS", OMPD_teams},
    {"DIR.OMP.DISTRIBUTE", OMPD_distribute},
    {"DIR.OMP.TEAMS.DISTRIBUTE", OMPD_teams_distribute},
    {"DIR.OMP.TEAMS.DISTRIBUTE.PARALLEL.LOOP",
     OMPD_teams_distribute_parallel_for},
    {"DIR.OMP.TARGET.TEAMS", OMPD_target_teams},
    {"DIR.OMP.TARGET.DATA", OMPD_target_data},
    {"DIR.OMP.TARGET.ENTER.DATA", OMPD_target_enter_data},
    {"DIR.OMP.TARGET.EXIT.DATA", OMPD_target_exit_data},
    {"DIR.OMP.TARGET.UPDATE", OMPD_target_update},
    {"DIR.OMP.TARGET.TEAMS.DISTRIBUTE", OMPD_target_teams_distribute},
    {"DIR.OMP.DISTRIBUTE.PARALLEL.LOOP", OMPD_distribute_parallel_for},
    {"DIR.OMP.TARGET.TEAMS.DISTRIBUTE.PARALLEL.LOOP",
     OMPD_target_teams_distribute_parallel_for}};

// TODO: add more reduction operators.
static const DenseMap<StringRef, DSAType> StringToDSA = {
    {"QUAL.OMP.PRIVATE", DSA_PRIVATE},
    {"QUAL.OMP.FIRSTPRIVATE", DSA_FIRSTPRIVATE},
    {"QUAL.OMP.LASTPRIVATE", DSA_LASTPRIVATE},
    {"QUAL.OMP.SHARED", DSA_SHARED},
    {"QUAL.OMP.REDUCTION.ADD", DSA_REDUCTION_ADD},
    {"QUAL.OMP.REDUCTION.SUB", DSA_REDUCTION_SUB},
    {"QUAL.OMP.REDUCTION.MUL", DSA_REDUCTION_MUL},
    {"QUAL.OMP.MAP.ALLOC", DSA_MAP_ALLOC},
    {"QUAL.OMP.MAP.TO", DSA_MAP_TO},
    {"QUAL.OMP.MAP.FROM", DSA_MAP_FROM},
    {"QUAL.OMP.MAP.TOFROM", DSA_MAP_TOFROM},
    {"QUAL.OMP.MAP.ALLOC.STRUCT", DSA_MAP_ALLOC_STRUCT},
    {"QUAL.OMP.MAP.TO.STRUCT", DSA_MAP_TO_STRUCT},
    {"QUAL.OMP.MAP.FROM.STRUCT", DSA_MAP_FROM_STRUCT},
    {"QUAL.OMP.MAP.TOFROM.STRUCT", DSA_MAP_TOFROM_STRUCT}};

inline std::string toString(DSAType DSA) {
  switch (DSA) {
  case DSA_NONE:
    return "DSA_NONE";
  case DSA_PRIVATE:
    return "DSA_PRIVATE";
  case DSA_FIRSTPRIVATE:
    return "DSA_FIRSTPRIVATE";
  case DSA_LASTPRIVATE:
    return "DSA_LASTPRIVATE";
  case DSA_SHARED:
    return "DSA_SHARED";
  case DSA_REDUCTION_ADD:
    return "DSA_REDUCTION_ADD";
  case DSA_REDUCTION_SUB:
    return "DSA_REDUCTION_SUB";
  case DSA_REDUCTION_MUL:
    return "DSA_REDUCTION_MUL";
  case DSA_MAP_ALLOC:
    return "DSA_MAP_ALLOC";
  case DSA_MAP_TO:
    return "DSA_MAP_TO";
  case DSA_MAP_FROM:
    return "DSA_MAP_FROM";
  case DSA_MAP_TOFROM:
    return "DSA_MAP_TOFROM";
  case DSA_MAP_ALLOC_STRUCT:
    return "DSA_MAP_ALLOC_STRUCT";
  case DSA_MAP_TO_STRUCT:
    return "DSA_MAP_TO_STRUCT";
  case DSA_MAP_FROM_STRUCT:
    return "DSA_MAP_FROM_STRUCT";
  case DSA_MAP_TOFROM_STRUCT:
    return "DSA_MAP_TOFROM_STRUCT";
  case DSA_MAP_STRUCT:
    return "DSA_MAP_STRUCT";
  default:
    FATAL_ERROR("Unknown DSA: " + std::to_string(DSA));
  }
}

/// Data attributes for each data reference used in an OpenMP target region.
enum tgt_map_type {
  // No flags
  OMP_TGT_MAPTYPE_NONE = 0x000,
  // copy data from host to device
  OMP_TGT_MAPTYPE_TO = 0x001,
  // copy data from device to host
  OMP_TGT_MAPTYPE_FROM = 0x002,
  // copy regardless of the reference count
  OMP_TGT_MAPTYPE_ALWAYS = 0x004,
  // force unmapping of data
  OMP_TGT_MAPTYPE_DELETE = 0x008,
  // map the pointer as well as the pointee
  OMP_TGT_MAPTYPE_PTR_AND_OBJ = 0x010,
  // pass device base address to kernel
  OMP_TGT_MAPTYPE_TARGET_PARAM = 0x020,
  // return base device address of mapped data
  OMP_TGT_MAPTYPE_RETURN_PARAM = 0x040,
  // private variable - not mapped
  OMP_TGT_MAPTYPE_PRIVATE = 0x080,
  // copy by value - not mapped
  OMP_TGT_MAPTYPE_LITERAL = 0x100,
  // mapping is implicit
  OMP_TGT_MAPTYPE_IMPLICIT = 0x200,
  // copy data to device
  OMP_TGT_MAPTYPE_CLOSE = 0x400,
  // runtime error if not already allocated
  OMP_TGT_MAPTYPE_PRESENT = 0x1000,
  // descriptor for non-contiguous target-update
  OMP_TGT_MAPTYPE_NON_CONTIG = 0x100000000000,
  // member of struct, member given by [16 MSBs] - 1
  OMP_TGT_MAPTYPE_MEMBER_OF = 0xffff000000000000
};

struct OffloadingMappingArgsTy {
  Value *Sizes;
  Value *MapTypes;
  Value *MapNames;
  Value *BasePtrs;
  Value *Ptrs;
  size_t Size;
};

struct FieldMappingInfo {
  Value *Index;
  Value *Offset;
  Value *NumElements;
  DSAType MapType;
};

using StructMapTy = MapVector<Value *, SmallVector<FieldMappingInfo, 4>>;

struct OMPLoopInfoStruct {
  Value *IV = nullptr;
  Value *Start = nullptr;
  Value *LB = nullptr;
  Value *UB = nullptr;
  // 0 is invalid, schedule will be set by the user or to reasonable defaults
  // by the pass.
  OMPScheduleType DistSched = static_cast<OMPScheduleType>(0);
  OMPScheduleType Sched = static_cast<OMPScheduleType>(0);
  Value *Chunk = nullptr;
};

struct OMPDistributeInfoStruct {
  Value *UB = nullptr;
  Value *LB = nullptr;
};

struct TargetInfoStruct {
  StringRef DevFuncName;
  ConstantDataArray *ELF = nullptr;
  Value *NumTeams = nullptr;
  Value *ThreadLimit = nullptr;
  OMPTgtExecModeFlags ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
  bool NoWait = false;
};

struct ParRegionInfoStruct {
  Value *NumThreads = nullptr;
  Value *IfCondition = nullptr;
};

struct TeamsInfoStruct {
  Value *NumTeams = nullptr;
  Value *ThreadLimit = nullptr;
  OMPTgtExecModeFlags ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
};

struct OutlinedInfoStruct {
  Function *Fn;
  BasicBlock *EntryBB;
  BasicBlock *ExitBB;
  SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;
};

struct CGReduction {
  template <DSAType ReductionOperator>
  static Value *emitOperation(IRBuilderBase &IRB, Value *LHS, Value *RHS);

  template <DSAType ReductionOperator>
  static OpenMPIRBuilder::InsertPointTy
  reductionNonAtomic(OpenMPIRBuilder::InsertPointTy IP, Value *LHS, Value *RHS,
                     Value *&Result) {
    IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
    Result = emitOperation<ReductionOperator>(Builder, LHS, RHS);
    return Builder.saveIP();
  }

  template <DSAType ReductionOperator>
  static InsertPointTy emitAtomicOperationRMW(IRBuilderBase &IRB, Value *LHS,
                                              Value *Partial);

  template <DSAType ReductionOperator>
  static InsertPointTy emitAtomicOperationCmpxchg(IRBuilderBase &IRB,
                                                  InsertPointTy IP, Type *VTy,
                                                  Value *LHS, Value *Partial) {
    LLVMContext &Ctx = IRB.getContext();
    unsigned int Bitwidth = VTy->getScalarSizeInBits();
    auto *IntTy =
        (Bitwidth == 64 ? Type::getInt64Ty(Ctx) : Type::getInt32Ty(Ctx));
    auto *IntPtrTy =
        (Bitwidth == 64 ? Type::getInt64PtrTy(Ctx) : Type::getInt32PtrTy(Ctx));

    auto SaveIP = IRB.saveIP();
    // TODO: move alloca to function entry point, may be outlined later, e.g.,
    // for nested under parallel.
    Value *AllocaTemp = IRB.CreateAlloca(IntTy, nullptr, "atomic.alloca.tmp");
    IRB.restoreIP(SaveIP);

    Value *CastLHS =
        IRB.CreateBitCast(LHS, IntPtrTy, LHS->getName() + ".cast.int");
    auto *LoadAtomic =
        IRB.CreateLoad(IntTy, CastLHS, LHS->getName() + ".load.atomic");
    LoadAtomic->setAtomic(AtomicOrdering::Monotonic);

    Value *CastFP = IRB.CreateBitCast(LoadAtomic, VTy, "cast.fp");
    Value *RedOp = emitOperation<ReductionOperator>(IRB, CastFP, Partial);
    Value *CastFAdd =
        IRB.CreateBitCast(RedOp, IntTy, RedOp->getName() + ".cast.int");

    auto *CmpXchg = IRB.CreateAtomicCmpXchg(CastLHS, LoadAtomic, CastFAdd, None,
                                            AtomicOrdering::Monotonic,
                                            AtomicOrdering::Monotonic);

    auto *Returned = IRB.CreateExtractValue(CmpXchg, 0);
    auto *StoreTemp = IRB.CreateStore(Returned, AllocaTemp);
    auto *Cond = IRB.CreateExtractValue(CmpXchg, 1);
    // Add unreachable as placholder for splitting.
    auto *Unreachable = IRB.CreateUnreachable();
    auto *IfTrueTerm = SplitBlockAndInsertIfThen(Cond, Unreachable, false);
    auto *ExitBlock = IfTrueTerm->getParent();
    auto *Retry = ExitBlock->getSingleSuccessor();
    assert(Retry && "Expected single successor tail block");
    // Erase the fall-through branch.
    IfTrueTerm->eraseFromParent();

    SaveIP = IRB.saveIP();
    IRB.SetInsertPoint(Retry, Retry->getFirstInsertionPt());
    auto *LoadReturned = IRB.CreateLoad(IntTy, AllocaTemp);
    auto *CastLoad = IRB.CreateBitCast(LoadReturned, VTy);
    // FAdd = IRB.CreateFAdd(CastLoad, Partial, "retry.add");
    RedOp = emitOperation<ReductionOperator>(IRB, CastLoad, Partial);
    CastFAdd = IRB.CreateBitCast(RedOp, IntTy, RedOp->getName() + ".cast.int");
    CmpXchg = IRB.CreateAtomicCmpXchg(CastLHS, LoadReturned, CastFAdd, None,
                                      AtomicOrdering::Monotonic,
                                      AtomicOrdering::Monotonic);
    Returned = IRB.CreateExtractValue(CmpXchg, 0);
    StoreTemp = IRB.CreateStore(Returned, AllocaTemp);
    Cond = IRB.CreateExtractValue(CmpXchg, 1);
    IRB.CreateCondBr(Cond, ExitBlock, Retry);
    // Remove unreachable placeholder.
    Unreachable->eraseFromParent();
    IRB.restoreIP(SaveIP);

    return InsertPointTy(ExitBlock, ExitBlock->getFirstInsertionPt());
  }

  template <DSAType ReductionOperator>
  static OpenMPIRBuilder::InsertPointTy
  reductionAtomic(OpenMPIRBuilder::InsertPointTy IP, Type *VTy, Value *LHS,
                  Value *RHS) {
    IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
    Value *Partial = Builder.CreateLoad(VTy, RHS, "red.partial");
    if (VTy->isIntegerTy())
      switch (ReductionOperator) {
      case DSA_REDUCTION_ADD:
      case DSA_REDUCTION_SUB:
        return emitAtomicOperationRMW<ReductionOperator>(Builder, LHS, Partial);
        break;
      case DSA_REDUCTION_MUL:
        // RMW does not support mul.
        return emitAtomicOperationCmpxchg<ReductionOperator>(Builder, IP, VTy,
                                                             LHS, Partial);
      default:
        FATAL_ERROR("Unsupported reduction operation");
      }
    else if (VTy->isFloatTy() || VTy->isDoubleTy()) {
      // NOTE: Using atomicrmw for floats is buggy for aarch64, fallback to
      // cmpxchg codegen for now similarly to Clang. Revisit with newer LLVM
      // versions.
      // Builder.CreateAtomicRMW(AtomicRMWInst::FAdd, LHS, Partial, None,
      //                        AtomicOrdering::Monotonic);
      return emitAtomicOperationCmpxchg<ReductionOperator>(Builder, IP, VTy,
                                                           LHS, Partial);
    } else
      FATAL_ERROR("Unsupported type for reductionAtomic");
  }

  template <DSAType ReductionOperator>
  static Value *emitInitAndAppendInfo(
      IRBuilderBase &IRB, InsertPointTy AllocaIP, Value *Orig,
      SmallVectorImpl<OpenMPIRBuilder::ReductionInfo> &ReductionInfos,
      bool IsGPUTeamsReduction) {
    auto GetIdentityValue = []() {
      switch (ReductionOperator) {
      case DSA_REDUCTION_ADD:
      case DSA_REDUCTION_SUB:
        return 0;
      case DSA_REDUCTION_MUL:
        return 1;
      default:
        FATAL_ERROR("Unknown reduction type");
      }
    };

    Type *VTy = Orig->getType()->getPointerElementType();
    auto SaveIP = IRB.saveIP();
    IRB.restoreIP(AllocaIP);
    Value *Priv = nullptr;

    if (IsGPUTeamsReduction) {
      Module *M = IRB.GetInsertBlock()->getModule();
      GlobalVariable *ShmemGV = new GlobalVariable(
          *M, VTy, false, GlobalValue::InternalLinkage, UndefValue::get(VTy),
          Orig->getName() + ".red.priv.shmem", nullptr,
          llvm::GlobalValue::NotThreadLocal, 3, false);
      Value *AddrCast = IRB.CreateAddrSpaceCast(ShmemGV, Orig->getType());
      Priv = AddrCast;
    } else {
      Priv = IRB.CreateAlloca(VTy, /* ArraySize */ nullptr,
                              Orig->getName() + ".red.priv");
    }
    IRB.restoreIP(SaveIP);

    // Store identity value based on operation and type.
    if (VTy->isIntegerTy()) {
      IRB.CreateStore(ConstantInt::get(VTy, GetIdentityValue()), Priv);
    } else if (VTy->isFloatTy() || VTy->isDoubleTy()) {
      IRB.CreateStore(ConstantFP::get(VTy, GetIdentityValue()), Priv);
    } else
      FATAL_ERROR("Unsupported type to init with identity reduction value");

    ReductionInfos.push_back(
        {VTy, Orig, Priv, CGReduction::reductionNonAtomic<ReductionOperator>,
         CGReduction::reductionAtomic<ReductionOperator>});

    return Priv;
  }
};

class CGIntrinsicsOpenMP {
public:
  CGIntrinsicsOpenMP(Module &M);

  OpenMPIRBuilder OMPBuilder;
  Module &M;
  StructType *TgtOffloadEntryTy;

  StructType *getTgtOffloadEntryTy() { return TgtOffloadEntryTy; }

  void emitOMPParallel(DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap,
                       const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry,
                       BasicBlock *StartBB, BasicBlock *EndBB,
                       BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
                       ParRegionInfoStruct &ParRegionInfo);

  void emitOMPFor(DSAValueMapTy &DSAValueMap, OMPLoopInfoStruct &OMPLoopInfo,
                  BasicBlock *StartBB, BasicBlock *ExitBB, bool IsStandalone,
                  bool IsDistributeParallelFor);

  void emitOMPTask(DSAValueMapTy &DSAValueMap, Function *Fn,
                   BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
                   BasicBlock *AfterBB);

  void emitOMPOffloadingEntry(const Twine &DevFuncName, Value *EntryPtr,
                              Constant *&OMPOffloadEntry);

  void emitOMPOffloadingMappings(InsertPointTy AllocaIP,
                                 DSAValueMapTy &DSAValueMap,
                                 StructMapTy &StructMappingInfoMap,
                                 OffloadingMappingArgsTy &OffloadingMappingArgs,
                                 bool IsTargetRegion);

  void emitOMPSingle(Function *Fn, BasicBlock *BBEntry, BasicBlock *AfterBB,
                     BodyGenCallbackTy BodyGenCB, FinalizeCallbackTy FiniCB);

  void emitOMPCritical(Function *Fn, BasicBlock *BBEntry, BasicBlock *AfterBB,
                       BodyGenCallbackTy BodyGenCB, FinalizeCallbackTy FiniCB);

  void emitOMPBarrier(Function *Fn, BasicBlock *BBEntry, Directive DK);

  void emitOMPTaskwait(BasicBlock *BBEntry);

  void emitOMPTarget(Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB,
                     BasicBlock *EndBB, DSAValueMapTy &DSAValueMap,
                     StructMapTy &StructMappingInfoMap,
                     TargetInfoStruct &TargetInfo,
                     OMPLoopInfoStruct *OMPLoopInfo, bool IsDeviceTargetRegion);

  void emitOMPTeams(DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap,
                    const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry,
                    BasicBlock *StartBB, BasicBlock *EndBB, BasicBlock *AfterBB,
                    TeamsInfoStruct &TeamsInfo);

  void emitOMPTargetData(Function *Fn, BasicBlock *BBEntry, BasicBlock *BBExit,
                         DSAValueMapTy &DSAValueMap,
                         StructMapTy &StructMappingInfoMap);

  void emitOMPTargetEnterData(Function *Fn, BasicBlock *BBEntry,
                              DSAValueMapTy &DSAValueMap,
                              StructMapTy &StructMappingInfoMap);

  void emitOMPTargetExitData(Function *Fn, BasicBlock *BBEntry,
                             DSAValueMapTy &DSAValueMap,
                             StructMapTy &StructMappingInfoMap);

  void emitOMPTargetUpdate(Function *Fn, BasicBlock *BBEntry,
                           DSAValueMapTy &DSAValueMap,
                           StructMapTy &StructMappingInfoMap);

  void emitOMPDistribute(DSAValueMapTy &DSAValueMap,
                         OMPLoopInfoStruct &OMPLoopInfo, BasicBlock *StartBB,
                         BasicBlock *ExitBB, bool IsStandalone,
                         bool IsDistributeParallelFor,
                         OMPDistributeInfoStruct *DistributeInfo = nullptr);

  void emitOMPDistributeParallelFor(DSAValueMapTy &DSAValueMap,
                                    BasicBlock *StartBB, BasicBlock *ExitBB,
                                    OMPLoopInfoStruct &OMPLoopInfo,
                                    ParRegionInfoStruct &ParRegionInfo,
                                    bool IsStandalone);

  void emitOMPTargetTeams(DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap,
                          const DebugLoc &DL, Function *Fn, BasicBlock *EntryBB,
                          BasicBlock *StartBB, BasicBlock *EndBB,
                          BasicBlock *AfterBB, TargetInfoStruct &TargetInfo,
                          TeamsInfoStruct &TeamsInfo,
                          OMPLoopInfoStruct *OMPLoopInfo,
                          StructMapTy &StructMappingInfoMap,
                          bool IsDeviceTargetRegion);

  GlobalVariable *emitOffloadingGlobals(StringRef DevWrapperFuncName,
                                        ConstantDataArray *ELF);

  Twine getDevWrapperFuncPrefix() { return "__omp_offload_numba_"; }

  OutlinedInfoStruct
  createOutlinedFunction(DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap,
                         Function *OuterFn, BasicBlock *StartBB,
                         BasicBlock *EndBB,
                         SmallVectorImpl<llvm::Value *> &CapturedVars,
                         StringRef Suffix, omp::Directive Kind);

  void setDeviceGlobalizedValues(const ArrayRef<Value *> GlobalizedValues);

private:
  void emitOMPParallelDeviceRuntime(DSAValueMapTy &DSAValueMap,
                                    ValueToValueMapTy *VMap, const DebugLoc &DL,
                                    Function *Fn, BasicBlock *BBEntry,
                                    BasicBlock *StartBB, BasicBlock *EndBB,
                                    BasicBlock *AfterBB,
                                    FinalizeCallbackTy FiniCB,
                                    ParRegionInfoStruct &ParRegionInfo);

  void emitOMPParallelHostRuntime(DSAValueMapTy &DSAValueMap,
                                  ValueToValueMapTy *VMap, const DebugLoc &DL,
                                  Function *Fn, BasicBlock *BBEntry,
                                  BasicBlock *StartBB, BasicBlock *EndBB,
                                  BasicBlock *AfterBB,
                                  FinalizeCallbackTy FiniCB,
                                  ParRegionInfoStruct &ParRegionInfo);
  void emitOMPParallelHostRuntimeOMPIRBuilder(
      DSAValueMapTy &DSAValueMap, ValueToValueMapTy *VMap, const DebugLoc &DL,
      Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
      BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
      ParRegionInfoStruct &ParRegionInfo);

  void emitOMPTeamsDeviceRuntime(DSAValueMapTy &DSAValueMap,
                                 ValueToValueMapTy *VMap, const DebugLoc &DL,
                                 Function *Fn, BasicBlock *BBEntry,
                                 BasicBlock *StartBB, BasicBlock *EndBB,
                                 BasicBlock *AfterBB,
                                 TeamsInfoStruct &TeamsInfo);
  void emitOMPTeamsHostRuntime(DSAValueMapTy &DSAValueMap,
                               ValueToValueMapTy *VMap, const DebugLoc &DL,
                               Function *Fn, BasicBlock *BBEntry,
                               BasicBlock *StartBB, BasicBlock *EndBB,
                               BasicBlock *AfterBB, TeamsInfoStruct &TeamsInfo);

  void emitOMPTargetHost(Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB,
                         BasicBlock *EndBB, DSAValueMapTy &DSAValueMap,
                         StructMapTy &StructMappingInfoMap,
                         TargetInfoStruct &TargetInfo,
                         OMPLoopInfoStruct *OMPLoopInfo);

  void emitOMPTargetDevice(Function *Fn, BasicBlock *BBEntry,
                           BasicBlock *StartBB, BasicBlock *EndBB,
                           DSAValueMapTy &DSAValueMap,
                           StructMapTy &StructMappingInfoMap,
                           TargetInfoStruct &TargetInfo);

  void emitLoop(DSAValueMapTy &DSAValueMap, OMPLoopInfoStruct &OMPLoopInfo,
                BasicBlock *StartBB, BasicBlock *ExitBB, bool IsStandalone,
                bool IsDistribute, bool IsDistributeParallelFor,
                OMPDistributeInfoStruct *OMPDistributeInfo = nullptr);

  InsertPointTy
  emitReductionsHost(const OpenMPIRBuilder::LocationDescription &Loc,
                     InsertPointTy AllocaIP,
                     ArrayRef<OpenMPIRBuilder::ReductionInfo> ReductionInfos);

  InsertPointTy emitReductionsDevice(
      const OpenMPIRBuilder::LocationDescription &Loc, InsertPointTy AllocaIP,
      ArrayRef<OpenMPIRBuilder::ReductionInfo> ReductionInfos, bool IsTeamSPMD);

  FunctionCallee getKmpcForStaticInit(Type *Ty);
  FunctionCallee getKmpcDistributeStaticInit(Type *Ty);
  Value *createScalarCast(Value *V, Type *DestTy);
  bool isOpenMPDeviceRuntime();

  SmallPtrSet<Value *, 32> DeviceGlobalizedValues;
};

} // namespace iomp

#endif
