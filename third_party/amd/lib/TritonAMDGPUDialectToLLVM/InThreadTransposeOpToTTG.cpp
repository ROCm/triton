#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::RankedTensorType;
namespace ttg = mlir::triton::gpu;

namespace {

struct InThreadTransposeOpConversion
    : public OpConversionPattern<triton::amdgpu::InThreadTransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::InThreadTransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, op.getType(),
                                                      op.getSrc());
    return success();
  }
};

struct InThreadTranspose8bitOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::InThreadTransposeOp> {
public:
  using ConvertOpToLLVMPattern<triton::amdgpu::InThreadTransposeOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::amdgpu::InThreadTransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    StringRef intrinsicName = "llvm.amdgcn.perm";
    Operation *permFuncOperation = SymbolTable::lookupSymbolIn(mod, intrinsicName);
    LLVM::LLVMFuncOp permFuncOp;
    if (!permFuncOperation) {
      auto insertionPoint = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(mod.getBody());
      LLVM::LLVMFunctionType permFuncType = LLVM::LLVMFunctionType::get(i32_ty, SmallVector<Type>{i32_ty, i32_ty, i32_ty});
      permFuncOp = rewriter.create<LLVM::LLVMFuncOp>(loc, intrinsicName, permFuncType);
      LLVM::LLVMDialect *llvmDialect = ctx->getLoadedDialect<LLVM::LLVMDialect>();
      permFuncOp->setAttr("llvm.readnone", UnitAttr::get(ctx));
      permFuncOp->setAttr("llvm.nounwind", UnitAttr::get(ctx));
      rewriter.restoreInsertionPoint(insertionPoint);
    } else {
      permFuncOp = cast<LLVM::LLVMFuncOp>(*permFuncOperation);
    }

    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    Value regVal = op.getSrc();
    Value memDescVal = op.getResult();

    auto regTy = cast<mlir::RankedTensorType>(regVal.getType());
    auto memDescTy = cast<mlir::RankedTensorType>(memDescVal.getType());

    auto regElemTy = regTy.getElementType();
    if (regElemTy.getIntOrFloatBitWidth() != 8) return failure();

    auto regEnc = dyn_cast<ttg::BlockedEncodingAttr>(regTy.getEncoding());
    auto memEnc = dyn_cast<ttg::LinearEncodingAttr>(memDescTy.getEncoding());
    if (!regEnc || !memEnc) return failure();

    auto regOrder = regEnc.getOrder();
    auto memOrder = memEnc.getOrder();
    if (regOrder.size()!=2 || memOrder.size()!=2) return failure();
    if (regOrder[0] != memOrder[1]) return failure();

    auto regSizePerThread = regEnc.getSizePerThread();
    if (regSizePerThread[0] < 4 || regSizePerThread[1] < 4) return failure();

    auto regVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto regOffsets = emitOffsetForLayout(regEnc, regTy);
    std::map<SmallVector<unsigned>, Value> regOffValues;
    for (size_t i = 0; i < regOffsets.size(); i++) {
      regOffValues[regOffsets[i]] = regVals[i];
    }
    llvm::dbgs() << "@@@@@ construct regOffValues end.\n";
    auto llvmRegElemTy = typeConverter->convertType(regElemTy);
    for (unsigned i=0; i<regSizePerThread[regOrder[1]]; i+=4) {
      for (unsigned j=0; j<regSizePerThread[regOrder[0]]; j+=4) {
        Type ty = vec_ty(llvmRegElemTy, 4);
        SmallVector<Value> raw{tb.undef(ty), tb.undef(ty), tb.undef(ty), tb.undef(ty)};
        for (int l = 0; l < 4; ++l) 
          for (int k = 0; k < 4; ++k) {
            raw[l] = tb.insert_element(raw[l], regOffValues[{i + l, j + k}], tb.i32_val(k));
          }

        Value raw0 = rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{tb.bitcast(raw[1], i32_ty), tb.bitcast(raw[0], i32_ty), tb.i32_val(0x05010400)}).getResult();  // 00, 01, 02, 03, 10, 11, 12, 13
        Value raw1 = rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{tb.bitcast(raw[1], i32_ty), tb.bitcast(raw[0], i32_ty), tb.i32_val(0x07030602)}).getResult();  // 00, 10, 01, 11, 02, 12, 03, 13 
        Value raw2 = rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{tb.bitcast(raw[3], i32_ty), tb.bitcast(raw[2], i32_ty), tb.i32_val(0x05010400)}).getResult();  // 20, 21, 22, 23, 30, 31, 32, 33
        Value raw3 = rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{tb.bitcast(raw[3], i32_ty), tb.bitcast(raw[2], i32_ty), tb.i32_val(0x07030602)}).getResult();  // 20, 30, 21, 31, 22, 32, 23, 33

        raw[0] = tb.bitcast(rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{raw2, raw0, tb.i32_val(0x05040100)}).getResult(), ty);  // 00, 10, 01, 11, 20, 30, 21, 31
        raw[1] = tb.bitcast(rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{raw2, raw0, tb.i32_val(0x07060302)}).getResult(), ty);  // 00, 10, 20, 30, 01, 11, 21, 31
        raw[2] = tb.bitcast(rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{raw3, raw1, tb.i32_val(0x05040100)}).getResult(), ty);  // 02, 12, 03, 13, 22, 32, 23, 33
        raw[3] = tb.bitcast(rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{raw3, raw1, tb.i32_val(0x07060302)}).getResult(), ty);  // 02, 12, 22, 32, 03, 13, 23, 33

        for (int l = 0; l < 4; ++l) 
          for (int k = 0; k < 4; ++k) {
            regOffValues[{i + k, j + l}] = tb.extract_element(llvmRegElemTy, raw[l], tb.i32_val(k));
          }
      }
    }
    llvm::dbgs() << "@@@@@ shuffle regOffValues end.\n";
    SmallVector<Value> resultVals;
    auto resOffsets = emitOffsetForLayout(regEnc, memDescTy);
    for (size_t i = 0; i < resOffsets.size(); i++) {
      resultVals.push_back(regOffValues[resOffsets[i]]);
    }
    llvm::dbgs() << "@@@@@ construct resultVals end.\n";

    Value result = packLLElements(loc, typeConverter, resultVals, rewriter,
                                  op.getType());

    rewriter.replaceOp(op, result);
    return success();
  }
};


} // namespace

namespace mlir::triton::AMD {

void populateInThreadTransposeOpToTTGPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
                                              PatternBenefit benefit) {
  patterns.add<InThreadTransposeOpConversion>(patterns.getContext(), benefit);
  patterns.add<InThreadTranspose8bitOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::AMD
