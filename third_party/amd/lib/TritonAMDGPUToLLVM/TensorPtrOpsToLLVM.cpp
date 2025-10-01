
/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
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
  using ConvertOpToLLVMPattern<
      triton::MakeTensorDescOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto tensorShape = adaptor.getShape();
    auto tensorStride = adaptor.getStrides();
    auto basePtr = adaptor.getBase();
    auto result = op.getResult();

    SmallVector<Value> elems;
    elems.push_back(basePtr);
    llvm::append_range(elems, tensorShape);
    llvm::append_range(elems, tensorStride);

    auto newValue = packLLElements(op.getLoc(), getTypeConverter(), elems,
                                   rewriter, result.getType());
    rewriter.replaceOp(op, newValue);
    return success();
  }
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
    PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  patterns.add<AdvanceOpConversion>(typeConverter, benefit);
  patterns.add<MakeTensorDescOpConversion>(typeConverter, benefit);
  return;
}
