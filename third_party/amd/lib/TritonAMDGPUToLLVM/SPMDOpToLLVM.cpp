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
    // branch Value blockId =
    //     rewriter.create<::mlir::gpu::GridDimOp>(loc,
    //     dims[op.getAxisAsInt()]);
    // rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    //
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
    rewriter.create<LLVM::CondBrOp>(loc, adaptor.getPred(), trueBlock,
                                    afterCondBarBlock);

    // conditional barrier
    rewriter.setInsertionPointToStart(trueBlock);
    rewriter.create<ROCDL::SBarrierOp>(loc);
    rewriter.create<LLVM::BrOp>(loc, afterCondBarBlock);
    rewriter.eraseOp(op);
    return success();
  }
};

struct GetWaveIdOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::GetWaveIdOp> {
  using ConvertOpToLLVMPattern<
      triton::amdgpu::GetWaveIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::GetWaveIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value id = getThreadId(rewriter, loc);
    int waveSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value waveId = LLVM::createLLVMIntrinsicCallOp(
                       rewriter, loc, "llvm.amdgcn.readfirstlane", {i32_ty},
                       {b.udiv(id, b.i32_val(waveSize))})
                       ->getResult(0);
    rewriter.replaceOp(op, waveId);
    return success();
  }
};

} // namespace

void mlir::triton::AMD::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<CondBarrierOpConversion>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpConversion, GetWaveIdOpConversion>(typeConverter,
                                                                  benefit);
}
