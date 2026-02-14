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

#define DEBUG_TYPE "tritonamdgpu-llir-schedule"

namespace {

using namespace llvm;

// Classifications used by the pass

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

enum class SchedKind { MFMA, BufferLoadLDS, BufferStore, LDSLoad, Other };

// Structures used for region analysis/scheduling

struct MFMAStats {
  unsigned Total = 0;
  unsigned SameIter = 0;
  unsigned Prefetched = 0;
  unsigned Mixed = 0;
};

struct AnchorInst {
  Instruction *I = nullptr;
  SchedKind Kind = SchedKind::Other;
};

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
using InstRegionMap = DenseMap<const Instruction *, unsigned>;

struct BBRegion {
  BasicBlock *BB = nullptr;
  Instruction *Begin = nullptr; // First instruction in region (inclusive)
  Instruction *End =
      nullptr; // First instruction of next region or nullptr (exclusive)
};

struct MFMARegionCollectResult {
  SmallVector<Instruction *, 16> Hoist;
  SmallVector<Instruction *, 16> Sink;
  Instruction *LastAnchor = nullptr;
  SmallVector<AnchorInst, 32> Anchors;
  SmallVector<Instruction *, 32> MFMAInsts;
};

// Utilities grouped for clarity

struct Utils {
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

  static void dumpInstructionHistogram(Loop *L) {
    DenseMap<InstClass, unsigned> hist;

    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        hist[classifyInstruction(I)]++;
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

  static bool isSchedBarrier(const Instruction &I) {
    return InstClass::schedBarrier == classifyInstruction(I);
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

  static bool isLDSLoadInst(const Instruction *I) {
    auto *LI = dyn_cast<LoadInst>(I);
    return LI && (LI->getPointerAddressSpace() == 3);
  }

  static bool isHoistTransparentInst(const Instruction &I) {
    return isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I);
  }

  static bool isSinkTransparentInst(const Instruction &I) {
    return isa<ExtractElementInst>(I);
  }

  static SchedKind classifySchedInst(Instruction &I) {
    if (isMFMAorWMMA(I))
      return SchedKind::MFMA;

    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->isIntrinsic() && F->getName().contains("buffer.load.lds"))
          return SchedKind::BufferLoadLDS;
        if (F->isIntrinsic() && F->getName().contains("buffer.store"))
          return SchedKind::BufferStore;
      }
    }

    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->getPointerAddressSpace() == 3)
        return SchedKind::LDSLoad;
    }

    return SchedKind::Other;
  }

  static iterator_range<BasicBlock::iterator>
  instructionsInRegion(const BBRegion &R) {
    BasicBlock *BB = R.BB;
    // Begin is now inclusive (region starts at this instruction)
    auto ItBegin = R.Begin ? R.Begin->getIterator() : BB->begin();
    auto ItEnd = R.End ? R.End->getIterator() : BB->end();
    return make_range(ItBegin, ItEnd);
  }

  static Instruction *getLoopHoistInsertPoint(Loop *L) {
    BasicBlock *Header = L->getHeader();
    return &*Header->getFirstInsertionPt();
  }

  static void insertSchedBarrierBefore(Instruction *IP) {
    Function *F = IP->getFunction();
    Module *M = F->getParent();
    Function *BarrierFn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);
    IRBuilder<> Builder(F->getContext());
    Builder.SetInsertPoint(IP);
    Value *Zero = Builder.getInt32(0);
    CallInst *CI = Builder.CreateCall(BarrierFn, {Zero});
    CI->setTailCallKind(CallInst::TCK_Tail);
  }

  static void insertSWaitCntBefore(Instruction *IP, int /*cnt*/) {
    // NOTE: LLVM AMDGPU s_waitcnt encoding operand packs several fields.
    // 49279 is taken from the original code. Keep as-is for behavior parity.
    Function *F = IP->getFunction();
    Module *M = F->getParent();
    Function *WaitFn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_waitcnt);
    IRBuilder<> Builder(F->getContext());
    Builder.SetInsertPoint(IP);
    Value *Cnt = Builder.getInt32(49279);
    CallInst *CI = Builder.CreateCall(WaitFn, {Cnt});
    CI->setTailCallKind(CallInst::TCK_Tail);
  }

  static bool containsMFMAAndBufferStore(const BasicBlock *BB) {
    bool HasMFMA = false;
    bool HasStore = false;

    for (const Instruction &I : *BB) {
      auto C = classifyInstruction(I);
      HasMFMA |= (C == InstClass::MFMA);
      HasStore |= (C == InstClass::bufferStore);
      if (HasMFMA && HasStore)
        return true;
    }
    return false;
  }

  static BasicBlock *findEpilogueBlock(Loop *MainLoop, LoopInfo &LI) {
    SmallVector<BasicBlock *, 4> ExitBlocks;
    MainLoop->getExitBlocks(ExitBlocks);

    for (BasicBlock *ExitBB : ExitBlocks) {
      if (MainLoop->contains(ExitBB))
        continue;
      if (containsMFMAAndBufferStore(ExitBB))
        return ExitBB;
    }
    return nullptr;
  }

  static unsigned getMFMACyclesFromName(StringRef Name) {
    // Extend this table as needed.
    if (Name.contains("mfma.scale.f32.16x16x128.f8f6f4"))
      return 32;
    if (Name.contains("mfma.f32.16x16x32.f16"))
      return 16;
    return 0; // Unknown
  }

  static unsigned getMFMACycles(const Instruction &I) {
    if (!isMFMAorWMMA(I))
      return 0;
    const auto *CI = cast<CallInst>(&I);
    const Function *Callee = CI->getCalledFunction();
    if (!Callee)
      return 0;
    return getMFMACyclesFromName(Callee->getName());
  }
};

// Region analysis and scheduling logic grouped into a helper class

class PreRAScheduler {
public:
  explicit PreRAScheduler() = default;

  void runOnLoop(Function &F, Loop &MainLoop, LoopInfo &LI) {
    LLVM_DEBUG(dbgs() << "Pre-RA scheduler analyzing function: " << F.getName()
                      << "\n");
    Utils::dumpInstructionHistogram(&MainLoop);

    BBMFMAAnalysisMap BBMFMAMap;

    for (BasicBlock *BB : MainLoop.blocks()) {
      LLVM_DEBUG(dbgs() << "BB: " << BB->getName() << "\n");
      analyzeBBMFMA(*BB, BBMFMAMap);
      scheduleBB(*BB, BBMFMAMap);
    }

    LLVM_DEBUG(dbgs() << "============================================\n");

    if (BasicBlock *epilogue = Utils::findEpilogueBlock(&MainLoop, LI)) {
      LLVM_DEBUG(dbgs() << "Found epilogue block: " << epilogue->getName()
                        << "\n");
      analyzeBBMFMA(*epilogue, BBMFMAMap);
      scheduleEpilogue(*epilogue, BBMFMAMap);
    }
  }

private:
  // Helper: trace back from MFMA operands (0 and 1) to find the first
  // shuffle or insert instruction that prepares the input
  static Instruction *findMFMAInputPrep(CallInst *MFMA) {
    if (!MFMA || MFMA->arg_size() < 2)
      return nullptr;

    SmallPtrSet<Value *, 16> Visited;
    SmallVector<Value *, 8> Worklist;

    // Only check operands 0 and 1 as instructed
    Worklist.push_back(MFMA->getArgOperand(0));
    Worklist.push_back(MFMA->getArgOperand(1));

    Instruction *FirstPrep = nullptr;

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      auto *I = dyn_cast<Instruction>(V);
      if (!I)
        continue;

      // Found a shuffle or insert - candidate for region start
      if (isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I)) {
        // Keep the earliest one (in program order)
        if (!FirstPrep || I->comesBefore(FirstPrep))
          FirstPrep = I;
        // Continue tracing to find earlier ones
        for (Value *Op : I->operands())
          Worklist.push_back(Op);
      } else if (isa<ExtractElementInst>(I)) {
        // Trace through extract as well
        for (Value *Op : I->operands())
          Worklist.push_back(Op);
      }
    }

    return FirstPrep;
  }

  // Automatic region detection based on MFMA + memory operation patterns
  static unsigned assignRegions(BasicBlock &BB, InstRegionMap &RegionMap) {
    unsigned CurRegion = 0;
    bool SeenMemoryOps = false;
    Instruction *RegionStart = nullptr;

    for (Instruction &I : BB) {
      // Check if this is a memory operation (buffer.load.lds or ds_read)
      auto SK = Utils::classifySchedInst(I);
      if (SK == SchedKind::BufferLoadLDS || SK == SchedKind::LDSLoad) {
        SeenMemoryOps = true;
      }

      // Check if this is an MFMA/WMMA
      if (Utils::isMFMAorWMMA(I)) {
        // If we've seen memory ops and already have a region, start a new one
        if (SeenMemoryOps && RegionStart != nullptr) {
          CurRegion++;
          SeenMemoryOps = false;
          RegionStart = nullptr;
        }

        // If this is the first MFMA in the current region, find its prep
        // instructions
        if (RegionStart == nullptr) {
          RegionStart = findMFMAInputPrep(cast<CallInst>(&I));
          if (!RegionStart)
            RegionStart = &I; // Fallback: use MFMA itself as region start
        }
      }

      // Assign current instruction to the current region
      if (RegionStart != nullptr) {
        RegionMap[&I] = CurRegion;
      }
    }

    return CurRegion; // number of regions (0-indexed, so actual count is
                      // CurRegion + 1 if any)
  }

  static MFMAInputSource
  traceMFMAOperandSource(Value *StartV, unsigned MFMARegion,
                         const InstRegionMap &RegionMap) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;
    Worklist.push_back(StartV);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      if (isa<PHINode>(V))
        return MFMAInputSource::FullyPrefetched;

      auto *I = dyn_cast<Instruction>(V);
      if (!I)
        continue;

      if (Utils::isLDSLoadInst(I)) {
        auto It = RegionMap.find(I);
        if (It == RegionMap.end())
          return MFMAInputSource::FullyPrefetched;
        unsigned LoadRegion = It->second;
        return (LoadRegion == MFMARegion) ? MFMAInputSource::SameRegionLoad
                                          : MFMAInputSource::FullyPrefetched;
      }

      if (isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I) ||
          isa<ExtractElementInst>(I)) {
        for (Value *Op : I->operands())
          Worklist.push_back(Op);
      }
    }
    return MFMAInputSource::Unknown;
  }

  static void analyzeBBMFMA(BasicBlock &BB, BBMFMAAnalysisMap &Out) {
    InstRegionMap RegionMap;
    unsigned MaxRegion = assignRegions(BB, RegionMap);

    // If no regions detected, return early
    if (RegionMap.empty())
      return;

    MFMARegionList Regions;
    Regions.resize(MaxRegion + 1);

    // Identify region start instructions (barriers)
    // Track which instruction starts each region
    DenseMap<unsigned, Instruction *> RegionStarts;

    for (auto &Entry : RegionMap) {
      const Instruction *I = Entry.first;
      unsigned RegionID = Entry.second;

      // Find the first instruction in each region to use as the "barrier"
      if (RegionStarts.find(RegionID) == RegionStarts.end()) {
        RegionStarts[RegionID] = const_cast<Instruction *>(I);
      } else {
        // Keep the earliest instruction
        if (I->comesBefore(RegionStarts[RegionID]))
          RegionStarts[RegionID] = const_cast<Instruction *>(I);
      }
    }

    // Set the barriers in the region list
    for (auto &Entry : RegionStarts) {
      unsigned RegionID = Entry.first;
      Instruction *StartInst = Entry.second;
      if (RegionID < Regions.size()) {
        Regions[RegionID].Barrier = StartInst;
      }
    }

    // Count MFMA and analyze their inputs
    for (Instruction &I : BB) {
      if (!Utils::isMFMAorWMMA(I))
        continue;

      auto It = RegionMap.find(&I);
      if (It == RegionMap.end())
        continue;

      unsigned R = It->second;
      Regions[R].TotalMFMA++;

      auto *CI = cast<CallInst>(&I);
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);

      auto S0 = traceMFMAOperandSource(Op0, R, RegionMap);
      auto S1 = traceMFMAOperandSource(Op1, R, RegionMap);

      if (S0 != MFMAInputSource::SameRegionLoad &&
          S1 != MFMAInputSource::SameRegionLoad) {
        Regions[R].FullyPrefetchedMFMA++;
      }
    }

    Out[&BB] = std::move(Regions);

    // print info
    for (unsigned i = 0; i < Out[&BB].size(); ++i) {
      const MFMARegionInfo &R = Out[&BB][i];
      if (!R.Barrier)
        continue;

      LLVM_DEBUG(dbgs() << "Region " << i << ": total MFMA: " << R.TotalMFMA
                        << ", fully prefetch: " << R.FullyPrefetchedMFMA
                        << "\n");
    }
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
          if (Utils::isMFMAorWMMA(*UI))
            return true;
          if (Utils::isHoistTransparentInst(*UI))
            Worklist.push_back(UI);
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
        if (Utils::isMFMAorWMMA(*DefI))
          return true;

        if (Utils::isSinkTransparentInst(*DefI)) {
          for (Value *Op : DefI->operands())
            Worklist.push_back(Op);
        }
      }
    }
    return false;
  }

  static MFMARegionCollectResult
  collectMFMAAndTransparentInstsInRegion(const BBRegion &R) {
    MFMARegionCollectResult Res;

    for (Instruction &I : Utils::instructionsInRegion(R)) {
      SchedKind K = Utils::classifySchedInst(I);
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

      if (Utils::isHoistTransparentInst(I)) {
        if (feedsMFMA(&I))
          Res.Hoist.push_back(&I);
        continue;
      }

      if (auto *EI = dyn_cast<ExtractElementInst>(&I)) {
        if (definedByMFMA(&I))
          Res.Sink.push_back(&I);
        (void)EI; // silence unused in release builds
      }
    }

    return Res;
  }

  static MFMARegionCollectResult
  preprocessMFMAInstsInRegion(const BBRegion &R) {
    auto Res = collectMFMAAndTransparentInstsInRegion(R);

    if (Res.Hoist.empty() && Res.Sink.empty())
      return Res;

    Instruction *HoistPos =
        R.Begin; // Region start (shuffle/insert feeding MFMA)
    Instruction *SinkPos = Res.LastAnchor; // last anchor in region

    for (Instruction *I : llvm::reverse(Res.Hoist)) {
      // Don't hoist R.Begin after itself
      if (I != HoistPos)
        I->moveAfter(HoistPos);
    }

    for (Instruction *I : llvm::reverse(Res.Sink))
      I->moveAfter(SinkPos);

    return Res;
  }

  static void scheduleMFMAWithSpacing(ArrayRef<AnchorInst> Anchors,
                                      SmallVectorImpl<Instruction *> &MFMAInsts,
                                      unsigned X, unsigned Y) {
    if (MFMAInsts.empty() || Anchors.empty())
      return;

    unsigned MFMAIdx = MFMAInsts.size();
    // Insert s.waitcnt before the first MFMA in the region
    Utils::insertSWaitCntBefore(MFMAInsts.front(), 0);

    auto spacingFor = [&](SchedKind K) {
      return (K == SchedKind::BufferLoadLDS) ? X : Y;
    };

    // Put 2 extra mfma after the last anchor inst
    {
      Instruction *InsertPt = Anchors.back().I;
      if (MFMAIdx >= 1)
        MFMAInsts[--MFMAIdx]->moveAfter(InsertPt);
      if (MFMAIdx >= 1)
        MFMAInsts[--MFMAIdx]->moveAfter(InsertPt);
    }

    SchedKind PrevKind = Anchors.back().Kind;

    for (int i = static_cast<int>(Anchors.size()) - 1; i >= 0; --i) {
      Instruction *InsertPt = Anchors[static_cast<size_t>(i)].I;
      SchedKind ThisKind = Anchors[static_cast<size_t>(i)].Kind;

      // If changing kind, put 2 extra mfma here

      if (PrevKind != ThisKind) {
        if (MFMAIdx >= 1)
          MFMAInsts[--MFMAIdx]->moveAfter(InsertPt);
        if (MFMAIdx >= 1)
          MFMAInsts[--MFMAIdx]->moveAfter(InsertPt);
      }

      PrevKind = ThisKind;
      unsigned Count = spacingFor(ThisKind);

      for (unsigned j = 0; j < Count && MFMAIdx > 0; ++j)
        MFMAInsts[--MFMAIdx]->moveAfter(InsertPt);
    }
  }

  static void scheduleBB(BasicBlock &BB, const BBMFMAAnalysisMap &Analysis) {
    auto It = Analysis.find(&BB);
    if (It == Analysis.end())
      return;

    const MFMARegionList &Regions = It->second;

    // Tunables
    unsigned X = 4;       // mfma between buffer.load.lds
    const unsigned Y = 1; // mfma between lds load

    for (unsigned i = 0; i < Regions.size(); ++i) {
      const MFMARegionInfo &R = Regions[i];
      if (!R.Barrier)
        continue;

      if (R.hasOnlyPrefetchedMFMA()) {
        BBRegion bbR;
        bbR.BB = &BB;
        bbR.Begin = Regions[i].Barrier;
        bbR.End = (i + 1 < Regions.size()) ? Regions[i + 1].Barrier : nullptr;

        MFMARegionCollectResult Res = preprocessMFMAInstsInRegion(bbR);

        if (!Res.MFMAInsts.empty()) {
          unsigned cycles = Utils::getMFMACycles(*Res.MFMAInsts.front());
          if (cycles == 32)
            X = 2;
          else if (cycles == 16)
            X = 4;
          // Unknown cycles -> keep fallback X = 4
        }

        scheduleMFMAWithSpacing(Res.Anchors, Res.MFMAInsts, X, Y);
      }
    }
  }

  static void scheduleEpilogue(BasicBlock &BB,
                               const BBMFMAAnalysisMap &Analysis) {
    auto It = Analysis.find(&BB);
    if (It == Analysis.end())
      return;

    const MFMARegionList &Regions = It->second;

    for (unsigned i = 0; i < Regions.size(); ++i) {
      const MFMARegionInfo &R = Regions[i];
      if (!R.Barrier)
        continue;

      LLVM_DEBUG(
          dbgs() << "Epilogue region " << i << ": total MFMA: " << R.TotalMFMA
                 << ", fully prefetch: " << R.FullyPrefetchedMFMA << "\n");
    }
  }
};

// Pass wrapper

struct LLIRSchedulePass : FunctionPass {
  static char ID;
  PreRAScheduler Scheduler;

  LLIRSchedulePass() : FunctionPass(ID) {}

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

    Loop *mainLoop = Utils::findMainLoop(LI);
    if (!mainLoop)
      return false;

    Scheduler.runOnLoop(F, *mainLoop, LI);
    // Analysis-only pass
    return false;
  }
};

} // end anonymous namespace

char LLIRSchedulePass::ID = 0;

namespace mlir::triton::AMD {

void runLLIRSchedulePass(llvm::Function &F) {
  llvm::legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(new LLIRSchedulePass());
  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();

  assert(!llvm::verifyFunction(F) &&
         "expected function to verify successfully");
}

} // namespace mlir::triton::AMD
