#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <list>

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create stream operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

#define DEBUG_TYPE "tritonamdgpu-unmask"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;


namespace {
struct UnmaskPass : public TritonAMDGPUUnmaskBase<UnmaskPass> {
  UnmaskPass() = default;

  bool hasMaskedLoads(scf::ForOp forOp) {
    bool has = false;
    auto iv = forOp.getInductionVar();
    
    forOp.walk([&](tt::LoadOp load) {
        if (load.getMask())
          has = true; // check if dependent on IV
      });
    return has;
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (hasMaskedLoads(forOp))
        loops.push_back(forOp);
    });

    for (auto loop : loops) {
      // reduce upper bound by 1 -- compute distance with
      auto loc = loop.getLoc();
      IRRewriter b(loop->getContext());
      b.setInsertionPoint(loop);
      auto ub = loop.getUpperBound();
      auto ubMinusOne = b.create<arith::SubIOp>(loc, ub, loop.getStep());
      loop.setUpperBound(ubMinusOne);
      // body for last iteration
      b.setInsertionPointAfter(loop);
      auto body = loop.getBody();
      SmallVector<Value> lresults, castResults, lastResults;
      lresults.push_back(ubMinusOne);
      //llvm::copy(loop.getResults(), lresults.end());
      for (Value res : loop.getResults()) {
        lresults.push_back(res);
        // make dummy cast
        auto cast = b.create<arith::ExtUIOp>(loc, res.getType(), res);
        res.replaceAllUsesWith(cast);
        castResults.push_back(cast);
      }
      assert(lresults.size() == body->getArguments().size());
      IRMapping map;
      for (auto [arg, res] : llvm::zip(body->getArguments(), lresults))
        map.map(arg, res);
      for (auto &op : *body) {
        if (auto yield = dyn_cast<scf::YieldOp>(op)) {
          for (Value yop : yield.getOperands())
            lastResults.push_back(map.lookup(yop));
        } else
          b.clone(op, map);
      }
      b.replaceAllUsesWith(castResults, lastResults);
      for (auto cast : castResults)
        b.eraseOp(cast.getDefiningOp());
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUUnmaskPass() {
  return std::make_unique<UnmaskPass>();
}
