#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tritonamdgpu-prera-llir-schedule"

using namespace llvm;

namespace {

/// Pick the loop with the largest number of instructions.
static Loop *findMainLoop(LoopInfo &LI) {
  Loop *mainLoop = nullptr;
  size_t maxInsts = 0;

  for (Loop *L : LI) {
    size_t instCount = 0;
    for (BasicBlock *BB : L->blocks())
      instCount += BB->size();

    if (instCount > maxInsts) {
      maxInsts = instCount;
      mainLoop = L;
    }
  }

  return mainLoop;
}

enum class InstClass {
  MFMA,
  WMMA,
  schedBarrier,
  sBarrier,
  sWaitCnt,
  schedGroupBarrier,
  bufferLoadLDS,
  OtherIntrinsic,
  Load,
  Store,
  FAdd,
  FMul,
  FMA,
  Branch,
  Other
};

enum class MFMAInputSource { SameIterationLoad, PrefetchedFromPHI, Unknown };

struct MFMAStats {
  unsigned Total = 0;
  unsigned SameIter = 0;
  unsigned Prefetched = 0;
  unsigned Mixed = 0;
};

enum class SchedKind { MFMA, BufferLoadLDS, LDSLoad, Other };

struct AnchorInst {
  Instruction *I;
  SchedKind Kind;
};

static InstClass classifyInstruction(const Instruction &I) {
  if (auto *CI = dyn_cast<CallInst>(&I)) {
    if (!CI->isInlineAsm()) {
      if (Function *Callee = CI->getCalledFunction()) {
        if (Callee->isIntrinsic()) {
          StringRef Name = Callee->getName();
          if (Name.contains("mfma"))
            return InstClass::MFMA;
          if (Name.contains("wmma"))
            return InstClass::WMMA;
          if (Name.contains("sched.barrier"))
            return InstClass::schedBarrier;
          if (Name.contains("sched.group.barrier"))
            return InstClass::schedGroupBarrier;
          if (Name.contains("s.waitcnt"))
            return InstClass::sWaitCnt;
          if (Name.contains("s.barrier"))
            return InstClass::sBarrier;
          if (Name.contains("llvm.amdgcn.raw.ptr.buffer.load.lds"))
            return InstClass::bufferLoadLDS;
          return InstClass::OtherIntrinsic;
        }
      }
    }
    return InstClass::Other;
  }

  if (isa<LoadInst>(I))
    return InstClass::Load;
  if (isa<StoreInst>(I))
    return InstClass::Store;

  switch (I.getOpcode()) {
  case Instruction::FAdd:
    return InstClass::FAdd;
  case Instruction::FMul:
    return InstClass::FMul;
  case Instruction::Br:
    return InstClass::Branch;
  default:
    return InstClass::Other;
  }
}

static StringRef instClassName(InstClass C) {
  switch (C) {
  case InstClass::MFMA:
    return "MFMA";
  case InstClass::WMMA:
    return "WMMA";
  case InstClass::sBarrier:
    return "s.barrier";
  case InstClass::schedBarrier:
    return "sched.barrier";
  case InstClass::schedGroupBarrier:
    return "sched.group.barrier";
  case InstClass::sWaitCnt:
    return "waitcnt";
  case InstClass::bufferLoadLDS:
    return "buffer_load_lds";
  case InstClass::OtherIntrinsic:
    return "OtherIntrinsic";
  case InstClass::Load:
    return "Load";
  case InstClass::Store:
    return "Store";
  case InstClass::FAdd:
    return "FAdd";
  case InstClass::FMul:
    return "FMul";
  case InstClass::Branch:
    return "Branch";
  case InstClass::Other:
    return "Other";
  }
  llvm_unreachable("unknown InstClass");
}

/// Emit a histogram of instruction opcodes inside the loop.
static void dumpInstructionHistogram(Loop *L) {
  DenseMap<InstClass, unsigned> hist;

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      InstClass C = classifyInstruction(I);
      hist[C]++;
    }
  }

  LLVM_DEBUG({
    dbgs() << "=== Pre-RA Main Loop Instruction Histogram ===\n";
    for (auto &KV : hist) {
      dbgs() << "  " << instClassName(KV.first) << ": " << KV.second << "\n";
    }
    dbgs() << "============================================\n";
  });
}

static bool isLDSLoadInst(const Instruction *I) {
  auto *LI = dyn_cast<LoadInst>(I);
  if (!LI)
    return false;

  return LI->getPointerAddressSpace() == 3;
}

static bool isTransparentInst(const Instruction *I) {
  return isa<ShuffleVectorInst>(I) || isa<BitCastInst>(I) ||
         isa<ExtractElementInst>(I) || isa<InsertElementInst>(I) ||
         isa<AddrSpaceCastInst>(I);
}

static MFMAInputSource traceMFMAOperandSource(Value *StartV, Loop *L) {
  SmallVector<Value *, 8> Worklist;
  SmallPtrSet<Value *, 16> Visited;

  Worklist.push_back(StartV);

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();

    if (!Visited.insert(V).second)
      continue;

    // PHI → prefetched from previous iteration
    if (auto *PN = dyn_cast<PHINode>(V)) {
      if (L->contains(PN->getParent()))
        return MFMAInputSource::PrefetchedFromPHI;
    }

    if (auto *I = dyn_cast<Instruction>(V)) {

      // ONLY real LDS loads count
      if (isLDSLoadInst(I))
        return MFMAInputSource::SameIterationLoad;

      // Walk through data-movement instructions
      if (isTransparentInst(I)) {
        for (Value *Op : I->operands())
          Worklist.push_back(Op);
        continue;
      }

      // Any other instruction stops the walk
      continue;
    }
  }

  return MFMAInputSource::Unknown;
}

static void analyzeMFMA(CallInst &CI, Loop *L, MFMAStats &Stats) {
  MFMAInputSource Src0 = traceMFMAOperandSource(CI.getArgOperand(0), L);
  MFMAInputSource Src1 = traceMFMAOperandSource(CI.getArgOperand(1), L);

  Stats.Total++;

  if (Src0 == MFMAInputSource::SameIterationLoad ||
      Src1 == MFMAInputSource::SameIterationLoad) {
    Stats.SameIter++;
  } else if (Src0 == MFMAInputSource::PrefetchedFromPHI &&
             Src1 == MFMAInputSource::PrefetchedFromPHI) {
    Stats.Prefetched++;
  } else {
    Stats.Mixed++;
  }
}

static bool isMFMAorWMMA(const Instruction &I) {
  auto *CI = dyn_cast<CallInst>(&I);
  if (!CI || CI->isInlineAsm())
    return false;

  Function *Callee = CI->getCalledFunction();
  if (!Callee || !Callee->isIntrinsic())
    return false;

  StringRef Name = Callee->getName();
  return Name.contains("mfma") || Name.contains("wmma");
}

static SchedKind classifySchedInst(Instruction &I) {
  if (isMFMAorWMMA(I))
    return SchedKind::MFMA;

  if (auto *CI = dyn_cast<CallInst>(&I)) {
    if (Function *F = CI->getCalledFunction()) {
      if (F->isIntrinsic() && F->getName().contains("buffer.load.lds"))
        return SchedKind::BufferLoadLDS;
    }
  }

  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    if (LI->getPointerAddressSpace() == 3)
      return SchedKind::LDSLoad;
  }

  return SchedKind::Other;
}

static void analyzeMainLoopMFMA(Loop *L) {
  MFMAStats Stats;

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (isMFMAorWMMA(I))
          analyzeMFMA(*CI, L, Stats);
      }
    }
  }

  LLVM_DEBUG({
    dbgs() << "=== MFMA Operand Source Analysis ===\n";
    dbgs() << "Total MFMA: " << Stats.Total << "\n";
    dbgs() << "MFMA with same-iteration loads: " << Stats.SameIter << "\n";
    dbgs() << "MFMA fully prefetched (PHI): " << Stats.Prefetched << "\n";
    dbgs() << "MFMA mixed inputs: " << Stats.Mixed << "\n";
    dbgs() << "===================================\n";
  });
}

static bool isHoistTransparentInst(const Instruction &I) {
  return isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I);
}

static bool isSinkTransparentInst(const Instruction &I) {
  return isa<ExtractElementInst>(I);
}

static bool feedsMFMA(Instruction *I) {
  SmallVector<Value *, 8> Worklist;
  SmallPtrSet<Value *, 16> Visited;

  Worklist.push_back(I);

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    for (User *U : V->users()) {
      if (auto *UI = dyn_cast<Instruction>(U)) {
        if (isMFMAorWMMA(*UI))
          return true;

        if (isHoistTransparentInst(*UI)) {
          Worklist.push_back(UI);
        }
      }
    }
  }
  return false;
}

static bool definedByMFMA(Instruction *I) {
  SmallVector<Value *, 8> Worklist;
  SmallPtrSet<Value *, 16> Visited;

  Worklist.push_back(I);

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    if (auto *DefI = dyn_cast<Instruction>(V)) {
      if (isMFMAorWMMA(*DefI))
        return true;

      if (isSinkTransparentInst(*DefI)) {
        for (Value *Op : DefI->operands())
          Worklist.push_back(Op);
      }
    }
  }
  return false;
}

static void collectMFMAAndTransparentInsts(
    Loop *L, SmallVectorImpl<Instruction *> &HoistInsts,
    SmallVectorImpl<Instruction *> &SinkInsts, Instruction *&LastAnchor) {

  LastAnchor = nullptr;

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {

      SchedKind K = classifySchedInst(I);
      if (K == SchedKind::BufferLoadLDS || K == SchedKind::LDSLoad)
        LastAnchor = &I;

      if (isHoistTransparentInst(I)) {
        if (feedsMFMA(&I))
          HoistInsts.push_back(&I);
        continue;
      }

      if (isSinkTransparentInst(I)) {
        if (definedByMFMA(&I))
          SinkInsts.push_back(&I);
        continue;
      }
    }
  }
}

static Instruction *getLoopHoistInsertPoint(Loop *L) {
  BasicBlock *Header = L->getHeader();
  return &*Header->getFirstInsertionPt();
}

static bool preprocessMFMAInsts(Loop *L) {
  SmallVector<Instruction *, 32> HoistInsts;
  SmallVector<Instruction *, 32> SinkInsts;
  Instruction *LastAnchor = nullptr;

  collectMFMAAndTransparentInsts(L, HoistInsts, SinkInsts, LastAnchor);

  if (!LastAnchor && HoistInsts.empty() && SinkInsts.empty())
    return false;

  Instruction *HoistIP = getLoopHoistInsertPoint(L);

  for (Instruction *I : llvm::reverse(HoistInsts))
    I->moveAfter(HoistIP);
  for (Instruction *I : llvm::reverse(SinkInsts))
    I->moveAfter(LastAnchor);

  return true;
}

static void collectAnchorsInOrder(Loop *L, SmallVectorImpl<AnchorInst> &Anchors,
                                  SmallVectorImpl<Instruction *> &MFMAInsts) {

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      SchedKind K = classifySchedInst(I);
      if (K == SchedKind::MFMA) {
        MFMAInsts.push_back(&I);
      } else if (K == SchedKind::BufferLoadLDS || K == SchedKind::LDSLoad) {
        Anchors.push_back({&I, K});
      }
    }
  }
}

static void insertSchedBarrier(Instruction *IP) {
  Function *F = IP->getFunction();
  Module *M = F->getParent();

  Function *BarrierFn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);

  IRBuilder<> Builder(F->getContext());

  // Insert BEFORE IP
  Builder.SetInsertPoint(IP);

  Value *Zero = Builder.getInt32(0);
  CallInst *CI = Builder.CreateCall(BarrierFn, {Zero});

  // Mark as tail call (matches your example)
  CI->setTailCallKind(CallInst::TCK_Tail);
}

static void scheduleMFMAWithSpacing(ArrayRef<AnchorInst> Anchors,
                                    SmallVectorImpl<Instruction *> &MFMAInsts,
                                    unsigned X, unsigned Y) {

  unsigned MFMAIdx = MFMAInsts.size() - 1;

  auto spacingFor = [&](SchedKind K) {
    return (K == SchedKind::BufferLoadLDS) ? X : Y;
  };

  for (int i = Anchors.size() - 1; i >= 0; --i) {
    Instruction *InsertPt = Anchors[i].I;
    unsigned Count = spacingFor(Anchors[i].Kind);

    for (int j = 0; j < Count && MFMAIdx >= 0; ++j) {
      MFMAInsts[MFMAIdx]->moveAfter(InsertPt);
      MFMAIdx--;
    }
    // insertSchedBarrier(InsertPt);
  }
}

struct PreRALLIRSchedulePass : FunctionPass {
  static char ID;

  PreRALLIRSchedulePass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesAll();
  }

  bool runOnFunction(Function &F) override {
    if (F.isDeclaration())
      return false;

    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    if (LI.empty())
      return false;

    Loop *mainLoop = findMainLoop(LI);
    if (!mainLoop)
      return false;

    LLVM_DEBUG(dbgs() << "Pre-RA scheduler analyzing function: " << F.getName()
                      << "\n");

    dumpInstructionHistogram(mainLoop);

    analyzeMainLoopMFMA(mainLoop);

    if (!preprocessMFMAInsts(mainLoop))
      llvm::outs() << "preprocessMFMAInsts failed!!\n";

    unsigned X = /* tunable: mfma between buffer.load.lds */ 4;
    unsigned Y = /* tunable: mfma between lds load */ 1;
    SmallVector<AnchorInst, 32> Anchors;
    SmallVector<Instruction *, 32> MFMAInsts;
    collectAnchorsInOrder(mainLoop, Anchors, MFMAInsts);
    scheduleMFMAWithSpacing(Anchors, MFMAInsts, X, Y);

    // Analysis-only pass
    return false;
  }
};

} // anonymous namespace

char PreRALLIRSchedulePass::ID = 0;

namespace mlir::triton::AMD {

void runPreRALLIRSchedulePass(Function &F) {
  // PreRALLIRSchedulePass pass;
  // pass.runOnFunction(F);

  // assert(!llvm::verifyFunction(F) &&
  //        "expected function to verify successfully");
  llvm::legacy::FunctionPassManager FPM(F.getParent());

  FPM.add(new PreRALLIRSchedulePass());
  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();

  assert(!llvm::verifyFunction(F) &&
         "expected function to verify successfully");
}

} // namespace mlir::triton::AMD
