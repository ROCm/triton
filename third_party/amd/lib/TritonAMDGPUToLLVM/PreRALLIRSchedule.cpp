#include "TritonAMDGPUToLLVM/Passes.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
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
