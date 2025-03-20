#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#define PRINT_SMALL_VECTOR(X) \
  { \
    std::string dbgStr; \
    llvm::raw_string_ostream dbgStream(dbgStr); \
    dbgStream << #X << ": "; \
    for (auto i = 0; i < X.size(); ++i) { \
      dbgStream << X[i]; \
      if (i < X.size()-1) { \
        dbgStream << "x"; \
      } \
    } \
    llvm::dbgs() << dbgStream.str() << "\n"; \
  }

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

inline size_t getSourceSize(Value &source) {
  ArrayRef<Type> types = cast<LLVM::LLVMStructType>(source.getType()).getBody();
  return types.size();
}

template<typename T>
SmallVector<int64_t> convertSerialToND(int64_t serial, ArrayRef<T> dims) {
  unsigned rank = dims.size();
  SmallVector<int64_t> coords(dims.size(), 0);
  for (int d = rank-1; d >= 0; --d) {
    coords[d] = serial % dims[d];
    serial /= dims[d];
  }
  return coords;
}

template<typename T1, typename T2>
int64_t convertNDToSerial(ArrayRef<T1> coord, ArrayRef<T2> dims) {
  unsigned rank = dims.size();
  int64_t serial = 0;
  int64_t stride = 1;
  for (int d = rank-1; d >= 0; --d) {
    serial += coord[d] * stride;
    stride *= dims[d];
  }
  return serial;
}

struct ConcatOpConversion : public ConvertOpToLLVMPattern<amdgpu::ConcatOp> {
  explicit ConcatOpConversion(LLVMTypeConverter &typeConverter,
                              PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<amdgpu::ConcatOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(amdgpu::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::dbgs() << "ConcatOpToLLVM\n";
                    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    auto resultShape = resultTy.getShape();
    PRINT_SMALL_VECTOR(resultShape)
    auto sources = adaptor.getSources();

    size_t totalNumElements = 0;
    for (auto source : sources) {
      totalNumElements += getSourceSize(source);
    }
    llvm::dbgs() << "totalNumElements: " << totalNumElements << "\n";

    auto coords = op.getCoords();
    // there's 1 coord for each rank.

    auto srcTy = cast<RankedTensorType>(op.getOperand(0).getType());
    auto srcShape = srcTy.getShape();
    PRINT_SMALL_VECTOR(srcShape)
    unsigned rank = srcShape.size();
    auto srcEncoding = srcTy.getEncoding();


    // don't want full tile shape, total output only has 32 elements
    // each source has 8 inputs.
    // I probably want elemsPerThread for a src shape
    auto srcElemsPerThread = ttg::getElemsPerThread(srcTy);
    PRINT_SMALL_VECTOR(srcElemsPerThread)
    auto resElemsPerThread = ttg::getElemsPerThread(resultTy);
    PRINT_SMALL_VECTOR(resElemsPerThread)

    llvm::SmallVector<Value> resultVals(totalNumElements);
    for (int srcIdx = 0; srcIdx < sources.size(); ++srcIdx) {
      auto src = sources[srcIdx];
      // Coordinate of source within result shape.
      auto srcCoord = convertSerialToND(srcIdx, coords);
      
      PRINT_SMALL_VECTOR(srcCoord)
      SmallVector<int64_t> srcOffset = srcCoord;
      for (int i = 0; i < rank; ++i) {
        srcOffset[i] *= srcElemsPerThread[i]; // srcShape[i];
      }
      PRINT_SMALL_VECTOR(srcOffset)
      auto elements = unpackLLElements(loc, src, rewriter);
      for (auto [elemIdx, element] : llvm::enumerate(elements)) {
        // Element offset within source shape.
        SmallVector<int64_t> elemOffset = convertSerialToND<unsigned>(elemIdx, srcElemsPerThread /*srcShape*/);
        PRINT_SMALL_VECTOR(elemOffset)
        SmallVector<int64_t> netCoord(rank, 0);
        for (int i = 0; i < rank; ++i) {
         netCoord[i] = srcOffset[i] + elemOffset[i];
        }
        PRINT_SMALL_VECTOR(netCoord)
        // Index in result shape.
        int64_t serial = convertNDToSerial<int64_t, unsigned>(netCoord, resElemsPerThread /*resultShape*/);
        llvm::dbgs() << "serial: " << serial << "\n";
        assert(serial < totalNumElements);
        resultVals[serial] = element;
      }
    }

#if 0
    size_t currNumElements = 0;
    for (auto source : sources) {
      auto elements = unpackLLElements(loc, source, rewriter);
      for (auto [idx, element] : llvm::enumerate(elements)) {
        resultVals[currNumElements + idx] = element;
      }
      currNumElements += getSourceSize(source);
    }
#endif
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
