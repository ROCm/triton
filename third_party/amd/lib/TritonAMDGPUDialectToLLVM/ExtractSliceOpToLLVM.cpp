#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-extract-slice-to-llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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

  LogicalResult processLayout2d(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSource().getType());
    auto srcLayout = srcTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto vals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto elemsPerThread = triton::gpu::getElemsPerThread(srcTy);
    auto contigPerThread = triton::gpu::getContigPerThread(srcTy);
    auto totalContigPerThread = product<unsigned>(contigPerThread);
    auto order = triton::gpu::getOrder(srcTy);

    // Calculate valid total number of workers in each dimension
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(srcTy);
    shapePerCTATile[0] =
        std::min(static_cast<unsigned>(srcShape[0]), shapePerCTATile[0]);
    shapePerCTATile[1] =
        std::min(static_cast<unsigned>(srcShape[1]), shapePerCTATile[1]);

    // Rank == 2 checked in the verifier
    SmallVector<int64_t, 2> sizes;
    for (auto i = 0; i < 2; ++i) {
      sizes.push_back(resultTy.getDimSize(i));
    }

    auto offsets = op.getStaticOffsets();

    // Calculate offsets and sizes in terms of CTA units.
    std::array<int64_t, 2> CTAOffsets{offsets[0] / shapePerCTATile[0],
                                      offsets[1] / shapePerCTATile[1]};
    std::array<int64_t, 2> CTASizes{sizes[0] / shapePerCTATile[0],
                                    sizes[1] / shapePerCTATile[1]};
    std::array<int64_t, 2> CTAPerShape{srcShape[0] / shapePerCTATile[0],
                                       srcShape[1] / shapePerCTATile[1]};

    // The diagram above illustrates the graphical representation of the
    // skipElems, tensorStride, and lastIdx variables.
    auto skipElems = CTAOffsets[order[1]] * (elemsPerThread[order[0]] *
                                             contigPerThread[order[1]]) +
                     CTAOffsets[order[0]] * totalContigPerThread;
    auto tensorStride =
        (CTAPerShape[order[0]] - CTASizes[order[0]]) * totalContigPerThread;
    auto lastIdx =
        (CTAOffsets[order[1]] + CTASizes[order[1]] - 1) *
            elemsPerThread[order[0]] * contigPerThread[order[1]] +
        (CTAOffsets[order[0]] + CTASizes[order[0]]) * totalContigPerThread;

    assert(lastIdx <= vals.size());

    SmallVector<Value> resultVals;
    for (int i = skipElems; i < lastIdx; i += tensorStride) {
      for (int j = 0; j < totalContigPerThread * CTASizes[order[0]]; ++j, ++i) {
        assert(i < lastIdx);
        resultVals.push_back(vals[i]);
      }
    }
    Value ret = packLLElements(loc, this->getTypeConverter(), resultVals,
                               rewriter, resultTy);

    rewriter.replaceOp(op, ret);
    return success();
  }

  // This handles extract_slice when the layout is ttg.slice.
  // For this case, the offsets, strides and sizes are only 1d,
  // and the sizePerThread is takes from the parentLayout.
  // ShapePerCTATile already examines the parentLayout for sliceLayout.
  LogicalResult processLayout1d(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSource().getType());
    auto srcLayout = srcTy.getEncoding();
    auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(srcLayout);
    auto dim = sliceLayout.getDim();
    auto parent = sliceLayout.getParent();
    auto srcShape = srcTy.getShape();
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto vals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto elemsPerThread = triton::gpu::getElemsPerThread(srcTy);
    auto sizePerThread = triton::gpu::getSizePerThread(parent);
    auto totalSizePerThread = product<unsigned>(sizePerThread);
    auto offsets = op.getStaticOffsets();

    LDBG("dim: " << dim);
    LDBG("srcShape: " << srcShape[0] << "x" << "!");
    LDBG("elemsPerThread: " << elemsPerThread[0] << "x" << "!");
    LDBG("sizePerThread: " << sizePerThread[0] << "x" << "!");
    LDBG("totalSizePerThread: " << totalSizePerThread);
    LDBG("offsets: " << offsets[0] << "x" << "!");

    // Calculate valid total number of workers in each dimension
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(srcLayout);
    for (auto i = 0; i < shapePerCTATile.size(); ++i) {
      shapePerCTATile[i] =
          std::min(static_cast<unsigned>(srcShape[i]), shapePerCTATile[i]);
    }
    LDBG("shapePerCTATile: " << shapePerCTATile[0] << "x" << "!");

    SmallVector<int64_t> sizes;
    SmallVector<int64_t> CTAOffsets;
    SmallVector<int64_t> CTASizes;
    SmallVector<int64_t> CTAPerShape;

    // Calculate offsets and sizes in terms of CTA units.
    for (auto i = 0; i < resultTy.getRank(); ++i) {
      sizes.push_back(resultTy.getDimSize(i));
      CTAOffsets.push_back(offsets[i] / shapePerCTATile[i]);
      CTASizes.push_back(sizes[i] / shapePerCTATile[i]);
      CTAPerShape.push_back(srcShape[i] / shapePerCTATile[i]);
    }
    LDBG("sizes: " << sizes[0] << "x" << "!");
    LDBG("CTAOffsets: " << CTAOffsets[0] << "x" << "!");
    LDBG("CTASizes: " << CTASizes[0] << "x" << "!");
    LDBG("CTAPerShape: " << CTAPerShape[0] << "x" << "!");

    auto skipElems = CTAOffsets[0] * totalSizePerThread;
    auto tensorStride = (CTAPerShape[0] - CTASizes[0]) * totalSizePerThread;
    auto lastIdx = (CTAOffsets[0] + CTASizes[0]) * totalSizePerThread;
    LDBG("skipElems: " << skipElems);
    LDBG("tensorStride: " << tensorStride);
    LDBG("lastIdx: " << lastIdx);

    assert(lastIdx <= vals.size());

    SmallVector<Value> resultVals;
    for (int i = skipElems; i < lastIdx; i += tensorStride) {
      for (int j = 0; j < totalSizePerThread * CTASizes[0]; ++j, ++i) {
        assert(i < lastIdx);
        LDBG("i: " << i);
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
      return processLayout2d(op, adaptor, rewriter);
    } else if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(encoding)) {
      auto parent = sliceLayout.getParent();
      if (isa<BlockedEncodingAttr, AMDMfmaEncodingAttr, DotOperandEncodingAttr>(
              parent)) {
        return processLayout1d(op, adaptor, rewriter);
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
