#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>
#include <optional>

#include <memory>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTOTENSOROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"
class TensorLoadLowering : public OpRewritePattern<DescriptorLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getResult().getType();

    // Very important. For a shared layout to "work" the main thing we need is
    // the order (all the rest is about swizzling). We need to get the order
    // somewhere.
    SmallVector<unsigned> order = getOrder(tensorType);
    if (auto blockedLayout =
            dyn_cast<BlockedEncodingAttr>(tensorType.getEncoding())) {
      order = llvm::to_vector(blockedLayout.getOrder());
    }

    auto ctaLayout = getCTALayout(tensorType.getEncoding());
    // At this point, we don't have any information about how this load is used.
    // Hence, we cannot set padding information
    Attribute encoding = SwizzledSharedEncodingAttr::get(
        tensorType.getContext(), 1, 1, 1, order, ctaLayout);

    // given this descriptor and the encoding, the framework should be able to
    // compute the LDS size.
    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType);
    Value waveId = rewriter.create<amdgpu::GetWaveIdOp>(loc);
    Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    // rewriter.create<amdgpu::GlobalTDMPrefetch>(
    //     loc, op.getDesc(), op.getIndices(), pred, waveId, ctaLayout);
    rewriter.create<amdgpu::AsyncTDMCopyGlobalToLocalOp>(
        loc, op.getDesc(), op.getIndices(), alloc, pred, waveId);
    rewriter.create<amdgpu::AsyncTDMWait>(loc, ArrayRef<Value>{}, 0);
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(), alloc);
    return success();
  }
};

struct TritonAMDGPUConvertToTensorOps
    : impl::TritonAMDGPUConvertToTensorOpsBase<TritonAMDGPUConvertToTensorOps> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    // TODO: add the conversion passes
    patterns.add<TensorLoadLowering>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
