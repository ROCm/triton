#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TDMUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
Attribute findEncodingFromUsers(Operation *op) {
  Attribute sharedEnc;

  for (auto use : op->getUsers()) {
    Attribute userEnc;
    if (auto load = llvm::dyn_cast<amdgpu::AsyncTDMCopyGlobalToLocalOp>(use)) {
      userEnc = load.getResult().getType().getEncoding();
    } else if (auto store =
                   llvm::dyn_cast<amdgpu::AsyncTDMCopyLocalToGlobalOp>(use)) {
      userEnc = store.getSrc().getType().getEncoding();
    }
    if (!userEnc)
      continue;

    // Assign first encoding found; or error out if different encoding is found
    if (!sharedEnc)
      sharedEnc = userEnc;
    else if (sharedEnc != userEnc) {
      op->emitError("Descriptor is used with different shared encodings.");
      return {};
    }
  }
  return sharedEnc;
}

struct MakeTensorPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorPtrOp> {
  using ConvertOpToLLVMPattern<triton::MakeTensorPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto tensorOffset = adaptor.getOffsets();
    auto tensorShape = adaptor.getShape();
    auto tensorStride = adaptor.getStrides();
    auto basePtr = adaptor.getBase();
    auto result = op.getResult();

    SmallVector<Value> elems;
    for (auto offset : tensorOffset)
      elems.push_back(offset);
    for (auto s : tensorShape)
      elems.push_back(s);
    for (auto stride : tensorStride)
      elems.push_back(stride);

    elems.push_back(basePtr);

    auto newValue = packLLElements(op.getLoc(), getTypeConverter(), elems,
                                   rewriter, result.getType());
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

struct MakeTensorDescOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorDescOp> {
  MakeTensorDescOpConversion(LLVMTypeConverter &typeConverter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::MakeTensorDescOp>(typeConverter,
                                                         benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto basePtr = adaptor.getBase();
    auto tensorShape = adaptor.getShape();
    auto tensorStride = adaptor.getStrides();
    auto result = op.getResult();

    auto tensorDescTy = result.getType();
    auto blockTy = tensorDescTy.getBlockType();
    auto sharedEnc = blockTy.getEncoding();
    if (!sharedEnc) {
      // TODO: add an extra pass to assign layout to descriptors
      sharedEnc = findEncodingFromUsers(op);
      if (!sharedEnc)
        return op.emitError("Descriptor requires a shared encoding during "
                            "lowering");
    }
    auto paddedEnc = llvm::dyn_cast<PaddedSharedEncodingAttr>(sharedEnc);

    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (paddedEnc) {
      if (paddedEnc.getIntervals().size() != 1 ||
          paddedEnc.getPaddings().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "NYI: Multiple interval-padding pairs in TDM.");
      padInterval = paddedEnc.getIntervals()[0];
      padAmount = paddedEnc.getPaddings()[0];
    }

    Type elementType =
        getTypeConverter()->convertType(blockTy.getElementType());
    SmallVector<int64_t> blockShape = to_vector(blockTy.getShape());
    int numWarps = lookupNumWarps(op);

    triton::LinearLayout sharedLayout;
    if (paddedEnc) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(blockTy.getShape(), sharedEnc);
    }
    auto shapePerCTA = triton::gpu::getShapePerCTA(sharedEnc, blockShape);
    auto kBlock = rewriter.getStringAttr("block");
    auto ctaLayout = sharedLayout.sublayout(
        {kBlock}, to_vector(sharedLayout.getOutDimNames()));

    // Create TDM descriptor for 2D-5D tensors
    auto tdmDesc = LLVM::AMD::createTDMDescriptor(
        rewriter, loc, getTypeConverter(), elementType, shapePerCTA, numWarps,
        padInterval, padAmount, tensorShape, tensorStride, basePtr,
        targetInfo.getClusterCTAId(rewriter, loc), ctaLayout);

    SmallVector<Value> groups = tdmDesc.getAllGroups();

    auto desc =
        packLLElements(loc, getTypeConverter(), groups, rewriter, tensorDescTy);

    rewriter.replaceOp(op, desc);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

struct AdvanceOpConversion : public ConvertOpToLLVMPattern<triton::AdvanceOp> {
  using ConvertOpToLLVMPattern<triton::AdvanceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ptrType = op.getPtr().getType();
    auto tensorPtr = adaptor.getPtr();

    TensorType tensorType = cast<TensorType>(ptrType.getPointeeType());
    Type llvmElemTy = typeConverter->convertType(tensorType.getElementType());

    auto offsets = adaptor.getOffsets();
    auto elems = unpackLLElements(loc, tensorPtr, rewriter);
    // TODO: this can be easily generalized to muliple dimensions
    SmallVector<Value, 2> tensorOffset{elems[0], elems[1]};
    SmallVector<Value, 2> tensorShape{elems[2], elems[3]};
    SmallVector<Value, 2> tensorStride{elems[4], elems[5]};
    Value basePtr = elems.back();
    int nDims = offsets.size();

    // a) Compute the new global ptr (this works only for 2d)
    Value ptrNewOffset = b.null(i64_ty);
    for (int i = 0; i < nDims; i++) {
      ptrNewOffset = b.add(ptrNewOffset,
                           b.mul(b.sext(i64_ty, offsets[i]), tensorStride[i]));
    }

    Type elemPtrTy = ptr_ty(rewriter.getContext(), 1);
    Value newPtr = b.gep(elemPtrTy, llvmElemTy, basePtr, ptrNewOffset);

    // b) Compute the new tensor shape
    SmallVector<Value, 2> newTensorShape(nDims);
    for (int i = 0; i < nDims; i++) {
      newTensorShape[i] = b.sub(tensorShape[i], b.sext(i64_ty, offsets[i]));
    }

    // c) Compute new offsets
    SmallVector<Value, 2> newOffsets(nDims);
    for (int i = 0; i < nDims; i++) {
      newOffsets[i] = b.add(tensorOffset[i], offsets[i]);
    }

    // Update the different fields
    size_t i = 0;
    for (i = 0; i < nDims; ++i) {
      elems[i] = newOffsets[i];
    }
    for (i = 0; i < nDims; i++) {
      elems[i + nDims] = newTensorShape[i];
    }
    elems.back() = newPtr;

    auto newValue = packLLElements(op.getLoc(), getTypeConverter(), elems,
                                   rewriter, ptrType);
    rewriter.replaceOp(op, newValue);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const AMD::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  patterns.add<AdvanceOpConversion>(typeConverter, benefit);
  patterns.add<MakeTensorDescOpConversion>(typeConverter, targetInfo, benefit);
  return;
}
