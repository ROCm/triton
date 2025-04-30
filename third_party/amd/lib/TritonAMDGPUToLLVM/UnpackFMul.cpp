#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"

#define DEBUG_TYPE "unpack-fmul"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

bool isMFMAorWMMA(Instruction *inst) {
  if (auto *callInst = llvm::dyn_cast<CallInst>(inst)) {
    StringRef intrinName = callInst->getCalledFunction()->getName();
    if (intrinName.contains("mfma") || intrinName.contains("wmma"))
      return true;
  }
  return false;
}

bool containsMFMA(BasicBlock &BB) {
  for (Instruction &inst : BB) {
    if (isMFMAorWMMA(&inst))
      return true;
  }
  return false;
}

bool unpackFMul(Instruction *inst, IRBuilder<> &builder) {
  Value *lhs, *rhs;
  if (!match(inst, m_BinOp(m_Value(lhs), m_Value(rhs))))
    return false;
  auto *VecLhs = dyn_cast<VectorType>(lhs->getType());
  if (!VecLhs)
    return false;
  assert(!VecLhs->isScalableTy() && "expected fixed-len vector");
  builder.SetInsertPoint(inst);
  Value *newVec = llvm::UndefValue::get(VecLhs);
  for (int i = 0; i < VecLhs->getElementCount().getFixedValue(); ++i) {
    Value *newLhs = builder.CreateExtractElement(lhs, i);
    Value *newRhs = builder.CreateExtractElement(rhs, i);
    Value *mul = builder.CreateFMul(newLhs, newRhs);
    newVec = builder.CreateInsertElement(newVec, mul, i);
  }
  inst->replaceAllUsesWith(newVec);
  return true;
}

// originates from VectorCombine::foldPermuteOfBinops
struct UnpackVectorFMul : FunctionPass {
  UnpackVectorFMul() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    IRBuilder<> builder(F.getContext());
    bool Changed = false;
    SmallVector<Instruction *> instsToErase(F.size());
    for (BasicBlock &BB : F) {
      if (!containsMFMA(BB))
        continue;
      for (Instruction &inst : BB) {
        if (inst.getOpcode() != Instruction::FMul)
          continue;
        if (unpackFMul(&inst, builder)) {
          instsToErase.push_back(&inst);
          Changed = true;
        }
      }
    }
    for (Instruction *inst : instsToErase) {
      if (inst)
        inst->eraseFromParent();
    }
    return Changed;
  }

  static char ID;
};

} // end anonymous namespace

char UnpackVectorFMul::ID = 0;

void runUnpackVectorFMulPassOnFunction(Function &F) {
  UnpackVectorFMul pass;
  pass.runOnFunction(F);
  // If there are no errors, the function returns false.
  assert(!llvm::verifyFunction(F) &&
         "expected function to verify successfully");
}
