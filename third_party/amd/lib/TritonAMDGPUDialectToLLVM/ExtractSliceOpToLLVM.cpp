#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

// clang-format off
//===--------------------------------------------------------------------------------===//
//   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
//   # WO   #  W1 #                                     |                                #
//   #      #     #                                     |                                #
//   #  #   #  #  #                                     |                                #
//   # W2   # W3  #   ....                              |                                #
//   #      #     #                                     |  SkipElems                     #
//   #  #   #  #  #                                     |                                #
//   #                                                  |                                #
//   #                                        Slice     |                                #
//   #    .                                 /        \  |                                #
//   #    .                                /          \ |                                #
//   #    .                               /            \|                                #
//   #                                    #   #  #  #  #                                 #
//   #                                    #  W0  #  W1 #                                 #
//   #                                    #      #     #                                 #
//   #                                    #  #   #  #  #    tensorStride                 #
//   #                                    #  W2  #  W3 # --------------------------------#
//   #                                    #      #     #                                 #
//   #                                    #  #   #  #  #                                 #
//   #          tensorStride              #  W0  #  W1 #                                 #
//   # ---------------------------------- #      #     #                                 #
//   #                                    #  #   #  #  #                                 #
//   #                                    #  W2  #  W3 #                                 #
//   #                                    #      #     #                                 #
//   #                                    #  #   #  #  # ---> lastIdx                    #
//   #                                         .                                         #
//   #                                         .                                         #
//   #                                         .                                         #
//   #                                                                                   #
//   #                                                                                   #
//   #                                                                                   #
//   #                                                                                   #
//   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
//===--------------------------------------------------------------------------------===//
// clang-format on

namespace {
struct ExtractSliceOpConversion
    : public ConvertOpToLLVMPattern<amdgpu::ExtractSliceOp> {
  explicit ExtractSliceOpConversion(LLVMTypeConverter &typeConverter,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<amdgpu::ExtractSliceOp>(typeConverter, benefit) {
  }

  LogicalResult processLayout(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSource().getType());
    auto srcLayout = srcTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto vals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto elemsPerThread = triton::gpu::getElemsPerThread(srcTy);
    auto contigPerThread =
        triton::gpu::toLinearEncoding(srcTy).getSizePerThread();

    auto totalContigPerThread = product<unsigned>(contigPerThread);
    auto order = triton::gpu::getOrder(srcTy);

    // Calculate valid total number of workers in each dimension
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(srcTy);
    for (auto i = 0; i < shapePerCTATile.size(); ++i) {
      shapePerCTATile[i] =
          std::min(static_cast<unsigned>(srcShape[i]), shapePerCTATile[i]);
    }

    auto offsets = op.getStaticOffsets();

    // Calculate offsets and sizes in terms of CTA units.
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> CTAOffsets;
    SmallVector<int64_t> CTASizes;
    SmallVector<int64_t> CTAPerShape;
    for (auto i = 0; i < resultTy.getRank(); ++i) {
      sizes.push_back(resultTy.getDimSize(i));
      CTAOffsets.push_back(offsets[i] / shapePerCTATile[i]);
      CTASizes.push_back(sizes[i] / shapePerCTATile[i]);
      CTAPerShape.push_back(srcShape[i] / shapePerCTATile[i]);
    }

    // SliceLayout uses 1d offsets.
    auto skipElems = CTAOffsets[0] * totalContigPerThread;
    auto tensorStride = (CTAPerShape[0] - CTASizes[0]) * totalContigPerThread;
    auto lastIdx = (CTAOffsets[0] + CTASizes[0]) * totalContigPerThread;
    auto numElemsPerVec = totalContigPerThread * CTASizes[0];

    auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(srcLayout);
    if (!sliceLayout) {
      // The diagram above illustrates the graphical representation of the
      // skipElems, tensorStride, and lastIdx variables.
      skipElems = CTAOffsets[order[1]] *
                      (elemsPerThread[order[0]] * contigPerThread[order[1]]) +
                  CTAOffsets[order[0]] * totalContigPerThread;
      tensorStride =
          (CTAPerShape[order[0]] - CTASizes[order[0]]) * totalContigPerThread;
      lastIdx =
          (CTAOffsets[order[1]] + CTASizes[order[1]] - 1) *
              elemsPerThread[order[0]] * contigPerThread[order[1]] +
          (CTAOffsets[order[0]] + CTASizes[order[0]]) * totalContigPerThread;
      numElemsPerVec = totalContigPerThread * CTASizes[order[0]];
    }

    assert(lastIdx <= vals.size());

    SmallVector<Value> resultVals;
    for (int i = skipElems; i < lastIdx; i += tensorStride) {
      for (int j = 0; j < numElemsPerVec; ++j, ++i) {
        assert(i < lastIdx);
        resultVals.push_back(vals[i]);
      }
    }
    Value ret = packLLElements(loc, this->getTypeConverter(), resultVals,
                               rewriter, resultTy);

    rewriter.replaceOp(op, ret);
    return success();
  }

  LogicalResult
  matchAndRewrite(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSource().getType();
    auto encoding = srcTy.getEncoding();
    if (isa<BlockedEncodingAttr, AMDMfmaEncodingAttr, DotOperandEncodingAttr>(
            encoding)) {
      return processLayout(op, adaptor, rewriter);
    } else if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(encoding)) {
      auto parent = sliceLayout.getParent();
      if (isa<BlockedEncodingAttr, AMDMfmaEncodingAttr, DotOperandEncodingAttr>(
              parent)) {
        return processLayout(op, adaptor, rewriter);
      }
    }
    return failure();
  }
};
} // namespace

namespace mlir::triton::AMD {

void populateExtractSliceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit) {
  patterns.add<ExtractSliceOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
