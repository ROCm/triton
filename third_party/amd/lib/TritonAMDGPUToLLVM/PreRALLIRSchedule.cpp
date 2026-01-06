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
  bufferStore,
  OtherIntrinsic,
  Load,
  Store,
  FAdd,
  FMul,
  FMA,
  Branch,
  Other
};

enum class MFMAInputSource { FullyPrefetched, SameRegionLoad, Unknown };

struct MFMAStats {
  unsigned Total = 0;
  unsigned SameIter = 0;
  unsigned Prefetched = 0;
  unsigned Mixed = 0;
};

enum class SchedKind { MFMA, BufferLoadLDS, BufferStore, LDSLoad, Other };

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
          if (Name.contains("llvm.amdgcn.raw.ptr.buffer.store"))
            return InstClass::bufferStore;
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
  case InstClass::bufferStore:
    return "buffer_store";
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

struct MFMARegionInfo {
  Instruction *Barrier = nullptr; // sched.barrier starting this region
  unsigned TotalMFMA = 0;
  unsigned FullyPrefetchedMFMA = 0;

  bool hasOnlyPrefetchedMFMA() const {
    return (TotalMFMA != 0) && (TotalMFMA == FullyPrefetchedMFMA);
  }
};

using MFMARegionList = SmallVector<MFMARegionInfo, 8>;

using BBMFMAAnalysisMap = DenseMap<const BasicBlock *, MFMARegionList>;

struct BBRegion {
  BasicBlock *BB;
  Instruction *Begin; // sched.barrier starting the region (exclusive)
  Instruction *End;   // next sched.barrier or nullptr (exclusive)
};

static iterator_range<BasicBlock::iterator>
instructionsInRegion(const BBRegion &R) {
  BasicBlock *BB = R.BB;

  auto ItBegin = R.Begin ? std::next(R.Begin->getIterator()) : BB->begin();

  auto ItEnd = R.End ? R.End->getIterator() : BB->end();

  return make_range(ItBegin, ItEnd);
}

struct MFMARegionCollectResult {
  SmallVector<Instruction *, 16> Hoist;
  SmallVector<Instruction *, 16> Sink;
  Instruction *LastAnchor = nullptr;
  SmallVector<AnchorInst, 32> Anchors;
  SmallVector<Instruction *, 32> MFMAInsts;
};

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

static bool isSchedBarrier(const Instruction &I) {
  return InstClass::schedBarrier == classifyInstruction(I);
}

using InstRegionMap = DenseMap<const Instruction *, unsigned>;

static unsigned assignRegions(BasicBlock *BB, InstRegionMap &RegionMap) {
  unsigned CurRegion = 0;
  bool SeenFirstBarrier = false;

  for (Instruction &I : *BB) {
    if (isSchedBarrier(I)) {
      SeenFirstBarrier = true;
      CurRegion++;
      continue;
    }

    if (!SeenFirstBarrier)
      continue; // ignore prologue

    RegionMap[&I] = CurRegion;
  }

  return CurRegion; // number of regions
}

static MFMAInputSource traceMFMAOperandSource(Value *StartV,
                                              unsigned MFMARegion,
                                              const InstRegionMap &RegionMap) {
  SmallVector<Value *, 8> Worklist;
  SmallPtrSet<Value *, 16> Visited;

  Worklist.push_back(StartV);

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    // PHI => prefetched
    if (isa<PHINode>(V))
      return MFMAInputSource::FullyPrefetched;

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      continue;

    // LDS load
    if (isLDSLoadInst(I)) {
      auto It = RegionMap.find(I);
      if (It == RegionMap.end())
        return MFMAInputSource::FullyPrefetched;

      unsigned LoadRegion = It->second;
      if (LoadRegion == MFMARegion)
        return MFMAInputSource::SameRegionLoad;
      else
        return MFMAInputSource::FullyPrefetched;
    }

    // Transparent ops
    if (isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I) ||
        isa<ExtractElementInst>(I)) {
      for (Value *Op : I->operands())
        Worklist.push_back(Op);
      continue;
    }
  }

  return MFMAInputSource::Unknown;
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

void analyzeBBMFMA(BasicBlock *BB, BBMFMAAnalysisMap &Out) {
  InstRegionMap RegionMap;
  unsigned NumRegions = assignRegions(BB, RegionMap);
  if (NumRegions == 0)
    return;

  MFMARegionList Regions;
  Regions.resize(NumRegions + 1);

  // Identify region barriers
  unsigned RegionID = 0;
  for (Instruction &I : *BB) {
    if (!isSchedBarrier(I))
      continue;

    // Barrier starts *next* region
    if (RegionID + 1 < Regions.size()) {
      Regions[RegionID + 1].Barrier = &I;
      RegionID++;
    }
  }

  // Count MFMA
  for (Instruction &I : *BB) {
    if (!isMFMAorWMMA(I))
      continue;

    auto It = RegionMap.find(&I);
    if (It == RegionMap.end())
      continue;

    unsigned R = It->second;
    Regions[R].TotalMFMA++;

    CallInst *CI = cast<CallInst>(&I);
    Value *Op0 = CI->getArgOperand(0);
    Value *Op1 = CI->getArgOperand(1);

    auto S0 = traceMFMAOperandSource(Op0, R, RegionMap);
    auto S1 = traceMFMAOperandSource(Op1, R, RegionMap);

    if (S0 != MFMAInputSource::SameRegionLoad &&
        S1 != MFMAInputSource::SameRegionLoad) {
      Regions[R].FullyPrefetchedMFMA++;
    }
  }

  Out[BB] = std::move(Regions);
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

MFMARegionCollectResult
collectMFMAAndTransparentInstsInRegion(const BBRegion &R) {
  MFMARegionCollectResult Res;

  for (Instruction &I : instructionsInRegion(R)) {
    SchedKind K = classifySchedInst(I);
    if (K == SchedKind::BufferLoadLDS || K == SchedKind::LDSLoad ||
        K == SchedKind::BufferStore) {
      Res.LastAnchor = &I;
      Res.Anchors.push_back({&I, K});
      continue;
    }

    if (K == SchedKind::MFMA) {
      Res.MFMAInsts.push_back(&I);
      continue;
    }

    // Hoist candidates:
    //  shufflevector / insertelement
    if (isa<ShuffleVectorInst>(&I) || isa<InsertElementInst>(&I)) {
      if (feedsMFMA(&I))
        Res.Hoist.push_back(&I);
      continue;
    }

    // Sink candidates:
    //  extractelement fed by mfma
    if (auto *EI = dyn_cast<ExtractElementInst>(&I)) {
      if (definedByMFMA(&I))
        Res.Sink.push_back(&I);
    }
  }

  return Res;
}

static Instruction *getLoopHoistInsertPoint(Loop *L) {
  BasicBlock *Header = L->getHeader();
  return &*Header->getFirstInsertionPt();
}

MFMARegionCollectResult preprocessMFMAInstsInRegion(const BBRegion &R) {
  auto Res = collectMFMAAndTransparentInstsInRegion(R);

  if (Res.Hoist.empty() && Res.Sink.empty())
    return Res;

  // llvm::outs() << "hoist has " << Res.Hoist.size() << " insts \n";
  // llvm::outs() << "sink has " << Res.Sink.size() << " insts \n";

  // Hoist target: the first instruction in the region, i.e. sched.barrier
  Instruction *HoistPos = R.Begin;

  // Sink target: last anchor instruction in the region
  Instruction *SinkPos = Res.LastAnchor;

  // Hoist
  for (Instruction *I : llvm::reverse(Res.Hoist))
    I->moveAfter(HoistPos);

  // Sink
  for (Instruction *I : llvm::reverse(Res.Sink))
    I->moveAfter(SinkPos);

  return Res;
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

  // Put 2 extra mfma after the last anchor inst
  Instruction *InsertPt = Anchors[Anchors.size() - 1].I;
  MFMAInsts[MFMAIdx--]->moveAfter(InsertPt);
  MFMAInsts[MFMAIdx--]->moveAfter(InsertPt);

  SchedKind kind = Anchors[Anchors.size() - 1].Kind;

  for (int i = Anchors.size() - 1; i >= 0; --i) {
    Instruction *InsertPt = Anchors[i].I;
    unsigned Count = spacingFor(Anchors[i].Kind);

    // If changing Kind, put 2 extra mfma here
    if (kind != Anchors[i].Kind) {
      MFMAInsts[MFMAIdx--]->moveAfter(InsertPt);
      MFMAInsts[MFMAIdx--]->moveAfter(InsertPt);
    }

    kind = Anchors[i].Kind;

    for (int j = 0; j < Count && MFMAIdx >= 0; ++j) {
      MFMAInsts[MFMAIdx--]->moveAfter(InsertPt);
    }
    // insertSchedBarrier(InsertPt);
  }
}

static bool containsMFMAAndBufferStore(const BasicBlock *BB) {
  bool HasMFMA = false;
  bool HasStore = false;

  for (const Instruction &I : *BB) {
    HasMFMA |= (InstClass::MFMA == classifyInstruction(I));
    HasStore |= (InstClass::bufferStore == classifyInstruction(I));

    if (HasMFMA && HasStore)
      return true;
  }
  return false;
}

static BasicBlock *findEpilogueBlock(Loop *MainLoop, LoopInfo &LI) {
  SmallVector<BasicBlock *, 4> ExitBlocks;
  MainLoop->getExitBlocks(ExitBlocks);

  for (BasicBlock *ExitBB : ExitBlocks) {
    // Must not be part of the loop
    if (MainLoop->contains(ExitBB))
      continue;

    if (containsMFMAAndBufferStore(ExitBB))
      return ExitBB;
  }

  return nullptr;
}

void scheduleBB(BasicBlock *BB, const BBMFMAAnalysisMap &Analysis) {
  auto It = Analysis.find(BB);
  if (It == Analysis.end())
    return;

  const MFMARegionList &Regions = It->second;
  llvm::outs() << "total regions: " << Regions.size() << "\n";

  unsigned X = /* tunable: mfma between buffer.load.lds */ 4;
  unsigned Y = /* tunable: mfma between lds load */ 1;

  for (unsigned i = 1; i < Regions.size(); ++i) {
    const MFMARegionInfo &R = Regions[i];

    if (!R.Barrier)
      continue;

    LLVM_DEBUG(dbgs() << i << ": total MFMA: " << R.TotalMFMA
                      << ", fully prefetch: " << R.FullyPrefetchedMFMA << "\n");

    if (R.hasOnlyPrefetchedMFMA()) {
      BBRegion bbR;
      bbR.BB = BB;
      bbR.Begin = Regions[i].Barrier;
      bbR.End = (i + 1 < Regions.size()) ? Regions[i + 1].Barrier : nullptr;

      auto Res = preprocessMFMAInstsInRegion(bbR);
      scheduleMFMAWithSpacing(Res.Anchors, Res.MFMAInsts, X, Y);
    }
  }
}

void scheduleEp(BasicBlock *BB, const BBMFMAAnalysisMap &Analysis) {
  auto It = Analysis.find(BB);
  if (It == Analysis.end())
    return;

  const MFMARegionList &Regions = It->second;
  llvm::outs() << "total regions: " << Regions.size() << "\n";

  for (unsigned i = 1; i < Regions.size(); ++i) {
    const MFMARegionInfo &R = Regions[i];

    if (!R.Barrier)
      continue;

    LLVM_DEBUG(dbgs() << i << ": total MFMA: " << R.TotalMFMA
                      << ", fully prefetch: " << R.FullyPrefetchedMFMA << "\n");
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

    BBMFMAAnalysisMap BBMFMAMap;

    for (BasicBlock *BB : mainLoop->blocks()) {
      LLVM_DEBUG(dbgs() << "BB: " << BB->getName() << "\n");
      analyzeBBMFMA(BB, BBMFMAMap);
      scheduleBB(BB, BBMFMAMap);
    }

    LLVM_DEBUG(dbgs() << "============================================\n");

    BasicBlock *epilogue = findEpilogueBlock(mainLoop, LI);
    if (epilogue) {
      LLVM_DEBUG(dbgs() << "Found epilogue block: " << epilogue->getName()
                        << "\n");
      analyzeBBMFMA(epilogue, BBMFMAMap);
      scheduleEp(epilogue, BBMFMAMap);
    }

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
