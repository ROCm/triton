#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPUNROLL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Check whether a dominating llvm.intr.assume tells us that `value` is
// divisible by `divisor`.  We pattern-match:
//   %rem  = arith.remsi %value, %divisor_cst
//   %cmp  = arith.cmpi eq, %rem, %c0
//   llvm.intr.assume %cmp
// (or the commuted cmpi eq, %c0, %rem form).
static bool hasDivisibilityAssumption(Value value, int64_t divisor,
                                      Operation *beforeOp) {
  Block *block = beforeOp->getBlock();
  for (auto &op : *block) {
    if (&op == beforeOp)
      break;

    auto assumeOp = dyn_cast<LLVM::AssumeOp>(&op);
    if (!assumeOp)
      continue;

    auto cmpi = assumeOp.getCond().getDefiningOp<arith::CmpIOp>();
    if (!cmpi || cmpi.getPredicate() != arith::CmpIPredicate::eq)
      continue;

    // Match cmpi eq(%rem, 0) or cmpi eq(0, %rem)
    Value lhs = cmpi.getLhs(), rhs = cmpi.getRhs();
    Operation *remDef = nullptr;
    Value zeroSide;
    if (auto rem = lhs.getDefiningOp<arith::RemSIOp>()) {
      remDef = rem;
      zeroSide = rhs;
    } else if (auto rem = lhs.getDefiningOp<arith::RemUIOp>()) {
      remDef = rem;
      zeroSide = rhs;
    } else if (auto rem = rhs.getDefiningOp<arith::RemSIOp>()) {
      remDef = rem;
      zeroSide = lhs;
    } else if (auto rem = rhs.getDefiningOp<arith::RemUIOp>()) {
      remDef = rem;
      zeroSide = lhs;
    }
    if (!remDef)
      continue;

    APInt zeroVal;
    if (!matchPattern(zeroSide, m_ConstantInt(&zeroVal)) || !zeroVal.isZero())
      continue;

    // Match rem(%value, %divisor_cst)
    if (remDef->getOperand(0) != value)
      continue;
    APInt factorVal;
    if (!matchPattern(remDef->getOperand(1), m_ConstantInt(&factorVal)) ||
        factorVal.getSExtValue() != divisor)
      continue;

    LDBG("Found divisibility assumption for unroll factor " << divisor);
    return true;
  }
  return false;
}

class LoopUnrollPass : public impl::TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    if (auto factor =
            forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName))
      return factor.getInt();
    return 1;
  }

  // Check if the trip count of `forOp` is assumed divisible by `unrollFactor`
  // via a preceding tl.assume / llvm.intr.assume.
  // Restricted to the common case: lb == 0, step == 1 (trip count == ub).
  bool isAssumedDivisible(scf::ForOp forOp, int64_t unrollFactor) {
    APInt lbVal, stepVal;
    if (!matchPattern(forOp.getLowerBound(), m_ConstantInt(&lbVal)) ||
        !lbVal.isZero())
      return false;
    if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
        stepVal != 1)
      return false;
    return hasDivisibilityAssumption(forOp.getUpperBound(), unrollFactor,
                                     forOp);
  }

  const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
  const char *pipelineStagesAttrName = "tt.num_stages";

public:
  void runOnOperation() override {
    LDBG("Loop unroll pass");
    SmallVector<scf::ForOp, 4> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with unroll factor <= 1.
      if (getUnrollFactorOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    auto ctx = getOperation()->getContext();
    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      bool assumeDivisible = isAssumedDivisible(loop, unrollFactor);
      Value originalUB = loop.getUpperBound();

      loop->removeAttr(loopUnrollFactorAttrName);
      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      auto resultLoops = loopUnrollByFactor(loop, unrollFactor);
      if (failed(resultLoops))
        continue;

      if (assumeDivisible && resultLoops->epilogueLoopOp) {
        // The user asserted (via tl.assume) that the trip count is divisible
        // by the unroll factor, so the epilogue is dead.  Replace its results
        // with the main loop's results and erase it.
        auto epilogue = *resultLoops->epilogueLoopOp;
        auto mainLoop = *resultLoops->mainLoopOp;
        LDBG("Erasing epilogue (trip count assumed divisible)");
        for (auto [epiRes, mainRes] :
             llvm::zip(epilogue.getResults(), mainLoop.getResults()))
          epiRes.replaceAllUsesWith(mainRes);
        // Point the main loop at the original upper bound; under the
        // divisibility assumption  ub == ub - (ub % factor).
        mainLoop.setUpperBound(originalUB);
        epilogue.erase();
      } else if (resultLoops->epilogueLoopOp) {
        // Do not pipeline the epilog loop.
        (*resultLoops->epilogueLoopOp)
            ->setAttr(pipelineStagesAttrName,
                      mlir::IntegerAttr::get(IntegerType::get(ctx, 32), 1));
      }
    }
  }
};

} // namespace mlir::triton
