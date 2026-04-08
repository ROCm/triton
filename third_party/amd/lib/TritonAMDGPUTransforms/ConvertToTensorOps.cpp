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
#include "third_party/amd/lib/TritonAMDGPUTransforms/Utility.h"
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
    auto encoding = getEncodingFromDescriptor(op, tensorType, op.getDesc());
    if (!encoding) {
      op.emitError() << "Could not create encoding for descriptor load";
      return failure();
    }

    // given this descriptor and the encoding, the framework should be able to
    // compute the LDS size.
    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = LocalAllocOp::create(rewriter, loc, memDescType);
    Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 32);

    amdgpu::AsyncTDMCopyGlobalToLocalOp::create(rewriter, loc, op.getDesc(),
                                                op.getIndices(), alloc, pred);
    amdgpu::AsyncTDMWait::create(rewriter, loc, ArrayRef<Value>{}, 0);
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(), alloc);
    return success();
  }
};

// Build the index encoding for TDM gather/scatter.
//
// Layout: BlockedLayout([1, M], [threadsPerWarp, 1], [1, numWarps], [0, 1])
// sliced along dim 0 to produce a 1D encoding. M is the max number of row
// indices per TDM instruction (256 bits / index element bitwidth). The
// freeVarMasks mechanism in the LLVM lowering adapts the number of active
// warps and gathers per warp to the actual problem size.
static SliceEncodingAttr getTDMGatherIndexEncoding(Operation *op) {
  MLIRContext *ctx = op->getContext();
  auto indicesType = cast<RankedTensorType>(
      cast<DescriptorGatherOp>(op).getXOffsets().getType());
  unsigned idxBitWidth = indicesType.getElementType().getIntOrFloatBitWidth();
  assert((idxBitWidth == 16 || idxBitWidth == 32) &&
         "TDM gather/scatter indices must be i16 or i32");
  unsigned maxIndicesPerInstr = 256 / idxBitWidth;

  unsigned numWarps = triton::gpu::lookupNumWarps(op);
  unsigned threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
      op->getParentOfType<ModuleOp>());
  auto cgaLayout = CGAEncodingAttr::get1CTALayout(ctx, /*rank=*/2);

  std::array<unsigned, 2> sizePerThread = {1, maxIndicesPerInstr};
  std::array<unsigned, 2> tPerWarp = {threadsPerWarp, 1};
  std::array<unsigned, 2> warpsPerCTA = {1, numWarps};
  std::array<unsigned, 2> order = {0, 1};
  auto parentEnc = BlockedEncodingAttr::get(ctx, sizePerThread, tPerWarp,
                                            warpsPerCTA, order, cgaLayout);
  return SliceEncodingAttr::get(ctx, /*dim=*/0, parentEnc);
}

struct TensorGatherLowering : public OpRewritePattern<DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorGatherOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getResult().getType();

    auto encoding = getEncodingFromDescriptor(op, tensorType, op.getDesc());
    if (!encoding) {
      op.emitError() << "Could not create encoding for descriptor gather";
      return failure();
    }

    auto idxEnc = getTDMGatherIndexEncoding(op);
    auto indices = op.getXOffsets();
    auto indicesType = cast<RankedTensorType>(indices.getType());

    // NOTE: The shared TritonToTritonGPU conversion (GatherScatterOpPattern)
    // unconditionally applies an NVIDIA-oriented index layout. Because of
    // this, the indices arriving here already carry that layout, making default
    // index encoding never matches most desirable AMD index encoding, and
    // therefore an additional ConvertLayoutOp emitted.
    if (indicesType.getEncoding() != idxEnc) {
      auto newIdxType = RankedTensorType::get(
          indicesType.getShape(), indicesType.getElementType(), idxEnc);
      indices = ConvertLayoutOp::create(rewriter, loc, newIdxType, indices);
    }

    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = LocalAllocOp::create(rewriter, loc, memDescType);
    Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 32);

    amdgpu::AsyncTDMGatherOp::create(rewriter, loc, op.getDesc(), indices,
                                     op.getYOffset(), alloc, pred);
    amdgpu::AsyncTDMWait::create(rewriter, loc, ArrayRef<Value>{}, 0);
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(), alloc);
    return success();
  }
};

class TensorStoreLowering : public OpRewritePattern<DescriptorStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    Value desc = op.getDesc();
    mlir::TypedValue<RankedTensorType> src = op.getSrc();
    auto tensorType = src.getType();
    auto encoding = getEncodingFromDescriptor(op, tensorType, desc);
    if (!encoding) {
      op.emitError() << "Could not create encoding for descriptor store";
      return failure();
    }

    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = LocalAllocOp::create(rewriter, loc, memDescType, op.getSrc());
    amdgpu::AsyncTDMCopyLocalToGlobalOp::create(rewriter, loc, op.getDesc(),
                                                op.getIndices(), alloc,
                                                /*barrier=*/Value{});
    amdgpu::AsyncTDMWait::create(rewriter, loc, ArrayRef<Value>{}, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TritonAMDGPUConvertToTensorOps
    : impl::TritonAMDGPUConvertToTensorOpsBase<TritonAMDGPUConvertToTensorOps> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TensorLoadLowering, TensorGatherLowering, TensorStoreLowering>(
        context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
