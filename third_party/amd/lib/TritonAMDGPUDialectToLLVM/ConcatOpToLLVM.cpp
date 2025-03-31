#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

inline size_t getSourceSize(Value source) {
  ArrayRef<Type> types = cast<LLVM::LLVMStructType>(source.getType()).getBody();
  return types.size();
}

struct ConcatOpConversion : public ConvertOpToLLVMPattern<amdgpu::ConcatOp> {
  explicit ConcatOpConversion(LLVMTypeConverter &typeConverter,
                              PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<amdgpu::ConcatOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(amdgpu::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    auto dims = op.getCoords();
    auto order = op.getLoweringOrder();

    auto sources = adaptor.getSources();
    auto elemPerSource = getSourceSize(sources.front());
    llvm::SmallVector<Value> resultVals;

    auto coords =
        mlir::triton::AMD::CoordinateMapper::cartesian(dims.vec(), order.vec());
    std::vector<int> strides(dims.size(), 1);
    std::exclusive_scan(dims.rbegin(), dims.rend(), strides.rbegin(), 1,
                        std::multiplies<>());

    for (const auto &vec : coords) {
      int linearIndex = 0;
      for (size_t i = 0; i < vec.size(); ++i) {
        linearIndex += vec[i] * strides[i];
      }

      auto elements = unpackLLElements(loc, sources[linearIndex], rewriter);
      for (auto elem : elements) {
        resultVals.push_back(elem);
      }
    }

    Value ret = packLLElements(loc, this->getTypeConverter(), resultVals,
                               rewriter, resultTy);

    rewriter.replaceOp(op, ret);
    return llvm::success();
  }
};
} // namespace

namespace mlir::triton::AMD {
void populateConcatOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns,
                                    mlir::PatternBenefit benefit) {
  patterns.add<ConcatOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
