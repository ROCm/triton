#include "third_party/amd/include/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {

// Traverses the def-chain including control flow of the token and returns true
// if all defining operations are an AsyncWait
bool comesFromAsyncWait(mlir::Value token) {
  if (auto defOp = token.getDefiningOp()) {
    if (isa<triton::gpu::AsyncWaitOp>(defOp))
      return true;
    else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(defOp))
      return comesFromAsyncWait(castOp.getInputs()[0]);
    else
      return false;
  }

  auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(token);
  // If the token has no defining op and is not an BlockArgument bail out
  if (!blockArg) {
    return false;
  }

  auto block = blockArg.getOwner();
  auto argId = blockArg.getArgNumber();

  auto destOperandFromAsyncWait = [argId](auto &&operands) {
    assert(argId < operands.size());
    return comesFromAsyncWait(operands[argId]);
  };

  // Check all predecessor block's terminator and follow the passed value at
  // argId to see if they are immediately an AsyncWait.
  for (auto *pred : block->getPredecessors()) {
    auto terminator = pred->getTerminator();
    if (auto br = llvm::dyn_cast<cf::BranchOp>(terminator)) {
      if (!destOperandFromAsyncWait(br.getDestOperands()))
        return false;
    } else if (auto condBr = llvm::dyn_cast<cf::CondBranchOp>(terminator)) {
      if (condBr.getTrueDest() == block) {
        if (!destOperandFromAsyncWait(condBr.getTrueDestOperands()))
          return false;
      }
      if (condBr.getFalseDest() == block) {
        if (!destOperandFromAsyncWait(condBr.getFalseDestOperands()))
          return false;
      }
    } else if (auto br = llvm::dyn_cast<LLVM::BrOp>(terminator)) {
      if (!destOperandFromAsyncWait(br.getDestOperands()))
        return false;
    } else {
      llvm::dbgs() << "no terminator!" << *terminator << "\n";
      return false;
    }
  }
  return true;
}

} // namespace mlir::triton::AMD
