#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.h"
#include <memory>

#define DEBUG_TYPE "tritonamdgpu-optimize-dot-operands"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

namespace {

// This pattern creates LocalAllocOp and LocalLoadOp with unswizzled shared
// layout for the scale operand used in ScaledUpcastFp4Op/ScaledUpcastFp8Op.
// The AMD pipeliner will respect the layout created here and pipeline ops
// according to the need.
//
// It matches
// tt.load -> ... -> amdg.scaled_upcast_x
//
// And rewrites it to
// tt.load -> ttg.local_alloc -> ttg.local_load -> ... -> amdg.scaled_upcast_x
template <typename OpTy>
class AllocSharedMemForUpcastedScales : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  AllocSharedMemForUpcastedScales(MLIRContext *context, bool targetIsCDNA4)
      : OpRewritePattern<OpTy>(context), targetIsCDNA4(targetIsCDNA4) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!targetIsCDNA4)
      return rewriter.notifyMatchFailure(op, "NYI: Only supported on CDNA4");

    auto forOp = op->template getParentOfType<scf::ForOp>();
    if (!forOp)
      return rewriter.notifyMatchFailure(op,
                                         "Don't alloc lds outside for loop");

    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    SetVector<Operation *> slice;
    (void)getBackwardSlice(op.getOperand(1), &slice, options);
    tt::LoadOp loadOp;
    unsigned cnt = 0;
    bool hasAllocatedLDS = false;
    for (auto &op : slice) {
      if (isa<tt::LoadOp>(op)) {
        loadOp = dyn_cast<tt::LoadOp>(op);
        cnt++;
      } else if (isa<ttg::LocalLoadOp>(op)) {
        hasAllocatedLDS = true;
        break;
      }
    }

    if (hasAllocatedLDS)
      return rewriter.notifyMatchFailure(
          op, "There's already lds allocation in the def chain.");

    if (!loadOp || cnt != 1)
      return rewriter.notifyMatchFailure(
          op, "Require exactly 1 load in the def chain.");

    LDBG("Found load of scale: " << loadOp << " for ScaleUpcast: " << op);
    auto srcTy = dyn_cast<RankedTensorType>(loadOp.getType());
    auto sharedOrder = ttg::getOrderForMemory(srcTy);
    auto cgaLayout = ttg::getCGALayout(srcTy.getEncoding());

    auto ctx = loadOp.getContext();
    auto attr = ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, sharedOrder,
                                                     cgaLayout);
    Location loc = loadOp.getLoc();
    auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
    rewriter.setInsertionPointAfter(loadOp);
    auto alloc = ttg::LocalAllocOp::create(
        rewriter, loc,
        ttg::MemDescType::get(srcTy.getShape(), srcTy.getElementType(), attr,
                              sharedMemorySpace),
        loadOp.getResult());
    LDBG("Create alloc: " << alloc);

    auto localLoad = ttg::LocalLoadOp::create(rewriter, loc, srcTy, alloc);
    LDBG("Create localload: " << localLoad);

    rewriter.replaceAllUsesExcept(loadOp.getResult(), localLoad, alloc);
    return success();
  }

private:
  bool targetIsCDNA4;
};

// A fusion between a dot operand's load and the dot keeps that operand in
// registers. The pipeliner only stages a load whose users all convert to a dot
// operand layout, and the fused op is neither, so it multi-buffers the global
// tiles in registers. On the scalar-FMA path those tiles are what spills.
// Route the fused tile through LDS instead. The buffer is operand-major and
// unswizzled, so the dot reads it coalesced and a K segment is a contiguous
// range of the buffer.
//
// It matches
//   tt.load -> <elementwise> -> ttg.convert_layout -> tt.dot
// and rewrites it to
//   tt.load -> <elementwise> -> ttg.local_alloc -> ttg.local_load
//            -> ttg.convert_layout -> tt.dot
//
// Off unless TRITON_AMD_STAGE_FUSED_DOT_OPERANDS is set.
class AllocSharedMemForFusedFMADotOperand : public OpRewritePattern<tt::DotOp> {
public:
  using OpRewritePattern<tt::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    if (!tt::tools::getBoolEnv("TRITON_AMD_STAGE_FUSED_DOT_OPERANDS"))
      return rewriter.notifyMatchFailure(dot, "not enabled");

    // Outside a loop the tile is live once, so staging it only adds traffic.
    if (!dot->getParentOfType<scf::ForOp>())
      return rewriter.notifyMatchFailure(dot, "dot is not in a loop");

    // The same test the AMD backend uses to route a dot to the scalar-FMA
    // lowering. WMMA and MFMA dots cost one instruction per tile and do not
    // have the operand register pressure this addresses.
    if (!isa<ttg::BlockedEncodingAttr>(dot.getD().getType().getEncoding()))
      return rewriter.notifyMatchFailure(dot, "does not lower to scalar FMAs");

    bool staged = false;
    for (unsigned opIdx : {0u, 1u}) {
      Value operand = opIdx == 0 ? dot.getA() : dot.getB();
      auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>();
      if (!cvt)
        continue;
      Value tile = cvt.getSrc();
      // An operand already coming out of shared memory is not what this
      // targets, and re-staging it would loop.
      Operation *def = tile.getDefiningOp();
      if (!def || !def->hasTrait<OpTrait::Elementwise>())
        continue;
      auto tileTy = dyn_cast<RankedTensorType>(tile.getType());
      if (!tileTy || tileTy.getRank() != 2)
        continue;

      MLIRContext *ctx = dot.getContext();
      // kContig=false puts K on the slow axis.
      SmallVector<unsigned> sharedOrder =
          ttg::getOrderForDotOperand(opIdx, /*rank=*/2, /*kContig=*/false);
      auto attr = ttg::SwizzledSharedEncodingAttr::get(
          ctx, 1, 1, 1, sharedOrder, ttg::getCGALayout(tileTy.getEncoding()));
      auto memTy = ttg::MemDescType::get(tileTy.getShape(),
                                         tileTy.getElementType(), attr,
                                         ttg::SharedMemorySpaceAttr::get(ctx));

      Location loc = def->getLoc();
      rewriter.setInsertionPointAfter(def);
      auto alloc = ttg::LocalAllocOp::create(rewriter, loc, memTy, tile);
      auto localLoad = ttg::LocalLoadOp::create(rewriter, loc, tileTy, alloc);
      LDBG("Staged fused operand " << opIdx << " of " << dot << " via "
                                   << alloc);
      rewriter.modifyOpInPlace(
          cvt, [&]() { cvt.getSrcMutable().assign(localLoad); });
      staged = true;
    }

    if (!staged)
      return rewriter.notifyMatchFailure(dot, "no fused operand to stage");
    return success();
  }
};
} // namespace

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEDOTOPERANDS
#include "TritonAMDGPUTransforms/Passes.h.inc"

struct TritonAMDGPUOptimizeDotOperandsPass
    : impl::TritonAMDGPUOptimizeDotOperandsBase<
          TritonAMDGPUOptimizeDotOperandsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    triton::amdgpu::TargetFeatures targetFeatures{llvm::StringRef(gfxArch)};
    patterns
        .add<AllocSharedMemForUpcastedScales<tt::amdgpu::ScaledUpcastFp8Op>,
             AllocSharedMemForUpcastedScales<tt::amdgpu::ScaledUpcastFp4Op>>(
            context, targetFeatures.isCDNA4());
    patterns.add<AllocSharedMemForFusedFMADotOperand>(context);
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir
