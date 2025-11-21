#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::RankedTensorType;

namespace mlir::triton::gpu {
LogicalResult lowerLocalStore(Location loc, MLIRContext *ctx, Value regVal,
                              MemDescType memDescTy, SharedMemoryObject smemObj,
                              ArrayRef<Value> inVals,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo);
}

namespace {
template <typename LocalLoadOpType>
class TransLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<LocalLoadOpType> {
public:
  TransLocalLoadOpConversion(const LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<LocalLoadOpType>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename LocalLoadOpType::Adaptor;

  static constexpr bool isPackedLoad =
      std::is_same_v<triton::amdgpu::LocalLoadPackedTransposedOp,
                     LocalLoadOpType>;

  LogicalResult
  matchAndRewrite(LocalLoadOpType op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (isPackedLoad || canUseTransLoad(op, srcTy, dstTy)) {
      return lowerSharedToDotOperandTransLL(op, adaptor,
                                            this->getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  bool checkLayoutProperties(MemDescType srcTy, RankedTensorType dstTy) const {
    // Verify the layout properties required for using the ds_read_tr
    // instruction. This instruction is used to load non-k contiguous tensors
    // from shared memory into a dot layout with an MFMA layout parent.
    auto dotEnc = llvm::dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    if (!dotEnc) {
      return false;
    }

    auto mfmaEnc = llvm::dyn_cast<AMDMfmaEncodingAttr>(dotEnc.getParent());
    if (!mfmaEnc) {
      return false;
    }

    auto tilesPerWarp = mfmaEnc.getTilesPerWarp();
    if (!mfmaEnc.hasUnitTilesPerWarp()) {
      return false;
    }

    auto sharedEnc =
        dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());
    if (!sharedEnc)
      return false;

    int rank = dstTy.getRank();
    const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
    return kDim != sharedEnc.getOrder()[0];
  }

  bool checkKWidth(MemDescType srcTy, RankedTensorType dstTy) const {
    // Single rate MFMA insts:
    // fp16, bf16: mfma32x32x8, mfma16x16x16
    // fp8, bf8: mfma32x32x16, mfma16x16x32
    // int8: mfma32x32x16, mfma16x16x32
    //
    // Double rate MFMA insts:
    // fp16, bf16: mfma32x32x16, mfma16x16x32
    // fp8, bf8: mfma32x32x64, mfma16x16x128
    // int8: mfma32x32x32, mfma16x16x64
    //
    // Check that kWidth of the dst dotOp layout is large enough to
    // work with the transposed lds load instructions.
    auto dotEnc = llvm::cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mfmaEnc = llvm::cast<AMDMfmaEncodingAttr>(dotEnc.getParent());

    int rank = dstTy.getRank();
    auto bitwidth = this->typeConverter->convertType(dstTy.getElementType())
                        .getIntOrFloatBitWidth();
    int32_t kWidth = dotEnc.getKWidth();
    const int32_t mDim = mfmaEnc.getMDim();
    if (mDim != 32 && mDim != 16)
      return false;

    const int kFactor = 16 / bitwidth;
    const int kSizeDoubleRateMfma32 = 16 * kFactor;
    const int kSizeDoubleRateMfma16 = 32 * kFactor;
    int largeTileThreshold =
        (mDim == 32) ? kSizeDoubleRateMfma32 : kSizeDoubleRateMfma16;

    // For FP8, wider MFMA instructions (scaled MFMA) have a k-dimension
    // that is four times of regular MFMA instructions.
    if (dstTy.getElementType().isFloat() && bitwidth == 8) {
      largeTileThreshold *= 2;
    }

    const auto shape = dstTy.getShape();
    const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
    const bool isLargeTile = shape[kDim] >= largeTileThreshold;

    const int kWidthLargeTile = 8 * kFactor;
    const int kWidthSmallTile = 4 * kFactor;
    // For largeTile, i.e. double rated mfma is an option, it's accepted to
    // have kWidth set for both double and single rated mfma
    // For smallTile, it's only accepted to have kWidth set to single rate
    // mfma. Smaller kWidth is not allowed to use transposed lds load.
    return (isLargeTile &&
            llvm::is_contained({kWidthLargeTile, kWidthSmallTile}, kWidth)) ||
           (kWidth == kWidthSmallTile);
  }

  bool checkCurrentLimitation(Operation *localLoad,
                              RankedTensorType dstTy) const {

    auto bitwidth = this->typeConverter->convertType(dstTy.getElementType())
                        .getIntOrFloatBitWidth();

    // FP4 is represented as i8 and, when packed along K, can be
    // transposed using ds_read_tr8 which doesn't change packing.
    if (bitwidth != 16 && bitwidth != 8) {
      return false;
    }

    return true;
  }

  bool canUseTransLoad(Operation *localLoad, MemDescType srcTy,
                       RankedTensorType dstTy) const {
    auto bitwidth = this->typeConverter->convertType(dstTy.getElementType())
                        .getIntOrFloatBitWidth();

    // 1. Check GPU arch properties.
    if (!targetInfo.canUseLDSTransLoad(bitwidth)) {
      return false;
    }

    // 2. Check layout properties.
    if (!checkLayoutProperties(srcTy, dstTy)) {
      return false;
    }

    // 3. Check current limitations.
    if (!checkCurrentLimitation(localLoad, dstTy)) {
      return false;
    }

    // 4. Check kWidth
    if (!checkKWidth(srcTy, dstTy)) {
      return false;
    }

    return true;
  }

  LogicalResult
  lowerSharedToDotOperandTransLL(LocalLoadOpType op, OpAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dotEnc = cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto shape = isPackedLoad ? srcTy.getShape() : dstTy.getShape();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto llBitwidth = isPackedLoad ? 4 : llvmElemTy.getIntOrFloatBitWidth();
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto ldsTransLayout = chooseDsReadB64TrLayout(dotEnc, shape, llBitwidth);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    SmallVector<Value> outVals;
    SmallVector<Value> elemsI32;
    mlir::Type retTy = dstTy;
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    bool valid = emitTransferBetweenRegistersAndShared(
        ldsTransLayout, srcTy, llvmElemTy,
        /*maxVecElems=*/std::nullopt, smemObj, loc, rewriter, targetInfo,
        laneId, warpId, [&](VectorType vecTy, Value vecAddr) {
          if constexpr (isPackedLoad) {
            assert(bitwidth == 8);
            auto numElems = vecTy.getNumElements();
            auto numElemsI32 = (numElems * bitwidth / 32);
            auto i32VecTy = VectorType::get(numElemsI32, i32_ty);
            auto dsReadOp =
                rewriter.create<ROCDL::ds_read_tr4_b64>(loc, i32VecTy, vecAddr);
            auto res = b.bitcast(dsReadOp.getResult(), vecTy);
            Value vecVal = res.getResult();
            for (int v = 0; v < vecTy.getNumElements(); v++) {
              outVals.push_back(
                  b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
            }
          } else if (bitwidth == 16) {
            auto dsReadOp =
                rewriter.create<ROCDL::ds_read_tr16_b64>(loc, vecTy, vecAddr);
            if constexpr (!isPackedLoad) {
              AMD::addLocalLoadNoAliasScope(op, dsReadOp);
            }
            Value vecVal = dsReadOp.getResult();
            for (int v = 0; v < vecTy.getNumElements(); v++) {
              outVals.push_back(
                  b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
            }
          } else {
            // pack elements in i32 vectors
            auto numElems = vecTy.getNumElements();
            auto numElemsI32 = (numElems * bitwidth / 32);
            auto i32VecTy = VectorType::get(numElemsI32, i32_ty);

            auto dsReadOp =
                rewriter.create<ROCDL::ds_read_tr8_b64>(loc, i32VecTy, vecAddr);
            if constexpr (!isPackedLoad) {
              AMD::addLocalLoadNoAliasScope(op, dsReadOp);
            }
            Value vecVal = dsReadOp.getResult();
            for (auto i = 0; i < numElemsI32; ++i) {
              elemsI32.push_back(
                  b.extract_element(i32_ty, vecVal, b.i32_val(i)));
            }
          }
        });

    // unpack i32 vectors and cast to native type
    if (bitwidth != 16) {
      auto numElemsPerVec = 32 / bitwidth;
      auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
      for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
        auto vec = b.bitcast(elemsI32[v], vecTy);
        for (int i = 0; i < numElemsPerVec; ++i)
          outVals.push_back(b.extract_element(llvmElemTy, vec, b.i32_val(i)));
      }

      retTy = LLVM::LLVMStructType::getLiteral(
          ctx, SmallVector<Type>(outVals.size(), llvmElemTy));
    }
    assert(valid && "Failed to emit LDS transpose load operations");
    Value result = packLLElements(loc, typeConverter, outVals, rewriter, retTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

struct TransStore8BitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  TransStore8BitOpConversion(const LLVMTypeConverter &converter,
                         const AMD::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
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
      rewriter.restoreInsertionPoint(insertionPoint);
    } else {
      permFuncOp = cast<LLVM::LLVMFuncOp>(*permFuncOperation);
    }

    if (targetInfo.getISAFamily() == AMD::ISAFamily::CDNA4) return failure();
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    Value regVal = op.getSrc();
    Value memDescVal = op.getDst();

    auto regTy = cast<RankedTensorType>(regVal.getType());
    auto memDescTy = cast<MemDescType>(memDescVal.getType());

    auto regElemTy = regTy.getElementType();
    if (regElemTy.getIntOrFloatBitWidth() != 8) return failure();

    auto regEnc = dyn_cast<BlockedEncodingAttr>(regTy.getEncoding());
    auto memEnc = dyn_cast<SwizzledSharedEncodingAttr>(memDescTy.getEncoding());
    if (!regEnc || !memEnc) return failure();

    auto regOrder = regEnc.getOrder();
    auto memOrder = memEnc.getOrder();
    if (regOrder.size()!=2 || memOrder.size()!=2) return failure();
    if (regOrder[0] != memOrder[1]) return failure();

    auto regSizePerThread = regEnc.getSizePerThread();
    auto vec = memEnc.getVec();
    if (regSizePerThread[0] < 4 || regSizePerThread[1] < 4 || vec < 4) return failure();

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

        Value raw0 = rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{tb.bitcast(raw[1], i32_ty), tb.bitcast(raw[0], i32_ty), tb.i32_val(0x05010400)}).getResult();  // 10, 11, 12, 13, 00, 01, 02, 03
        Value raw1 = rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{tb.bitcast(raw[1], i32_ty), tb.bitcast(raw[0], i32_ty), tb.i32_val(0x07030602)}).getResult();  // 01, 11, 00, 10, 03, 13, 02, 12 
        Value raw2 = rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{tb.bitcast(raw[3], i32_ty), tb.bitcast(raw[2], i32_ty), tb.i32_val(0x05010400)}).getResult();  // 30, 31, 32, 33, 20, 21, 22, 23
        Value raw3 = rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{tb.bitcast(raw[3], i32_ty), tb.bitcast(raw[2], i32_ty), tb.i32_val(0x07030602)}).getResult();  // 21, 31, 20, 30, 23, 33, 22, 32 

        raw[0] = tb.bitcast(rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{raw2, raw0, tb.i32_val(0x05040100)}).getResult(), ty);  // 01, 11, 00, 10, 21, 31, 20, 30
        raw[1] = tb.bitcast(rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{raw2, raw0, tb.i32_val(0x07060302)}).getResult(), ty);  // 31, 21, 
        raw[2] = tb.bitcast(rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{raw3, raw1, tb.i32_val(0x05040100)}).getResult(), ty);  // 02, 03, 12, 13, 22, 23, 32, 33
        raw[3] = tb.bitcast(rewriter.create<LLVM::CallOp>(loc, permFuncOp, SmallVector<Value>{raw3, raw1, tb.i32_val(0x07060302)}).getResult(), ty);

        for (int l = 0; l < 4; ++l) 
          for (int k = 0; k < 4; ++k) {
            regOffValues[{i + k, j + l}] = tb.extract_element(llvmRegElemTy, raw[l], tb.i32_val(k));
          }
      }
    }
    llvm::dbgs() << "@@@@@ shuffle regOffValues end.\n";
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < regOffsets.size(); i++) {
      resultVals.push_back(regOffValues[regOffsets[i]]);
    }
    llvm::dbgs() << "@@@@@ construct resultVals end.\n";
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);

    if (failed(mlir::triton::gpu::lowerLocalStore(loc, ctx, regVal, memDescTy, smemObj, resultVals,
                               typeConverter, rewriter, targetInfo))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  patterns.add<TransStore8BitOpConversion>(
      typeConverter, targetInfo, PatternBenefit(benefit.getBenefit() + 2));
  patterns.add<TransLocalLoadOpConversion<triton::gpu::LocalLoadOp>>(
      typeConverter, targetInfo, transBenefit);
  patterns.add<
      TransLocalLoadOpConversion<triton::amdgpu::LocalLoadPackedTransposedOp>>(
      typeConverter, targetInfo, benefit);
}
