#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Get the num_program from the implicitarg ptr which might be lowered into
    // directly loading it from user sgprs
    // The first 12 bytes represent the hidden block count x,y,z
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ptrTy = ptr_ty(rewriter.getContext(), 4);
    auto implicitArgPtr =
        LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, "llvm.amdgcn.implicitarg.ptr", {ptrTy}, {})
            .getResult(0);

    auto offset = b.i32_val(op.getAxisAsInt());
    auto gridDimAxisPtr = b.gep(ptrTy, i32_ty, implicitArgPtr, offset);
    rewriter.replaceOp(op, b.load(i32_ty, gridDimAxisPtr));

    // TODO(alex) I think that is not true anymore, the same should work on gfx9
    // TODO: this needs to stay for arch < MI400. Check the target info and
    // assert(op.getAxisAsInt() < 3);
    // Value blockId =
    //     ::mlir::gpu::GridDimOp::create(rewriter, loc,
    //     dims[op.getAxisAsInt()]);
    // rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

struct CondBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::CondBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::CondBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterCondBarBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterCondBarBlock);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, adaptor.getPred(), trueBlock,
                           afterCondBarBlock);

    // conditional barrier
    rewriter.setInsertionPointToStart(trueBlock);
    ROCDL::SBarrierOp::create(rewriter, loc);
    LLVM::BrOp::create(rewriter, loc, afterCondBarBlock);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<CondBarrierOpConversion>(typeConverter, benefit);
}
