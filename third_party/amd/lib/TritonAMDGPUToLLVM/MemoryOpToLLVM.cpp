#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/TargetParser/TargetParser.h"

using ::mlir::LLVM::AMD::isUsedByDotScaledOp;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MemDescType;

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

    auto mfmaEnc =
        llvm::dyn_cast_or_null<AMDMfmaEncodingAttr>(dotEnc.getParent());
    auto wmmaEnc =
        llvm::dyn_cast_or_null<AMDWmmaEncodingAttr>(dotEnc.getParent());
    if ((!wmmaEnc) && (!mfmaEnc)) {
      return false;
    }

    if (mfmaEnc && !mfmaEnc.hasUnitTilesPerWarp()) {
      return false;
    }

    int rank = dstTy.getRank();
    const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;

    auto swizzledEnc =
        dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcTy.getEncoding());
    if (!swizzledEnc && !paddedEnc)
      return false;

    bool out = kDim != (swizzledEnc ? swizzledEnc.getOrder()[0]
                                    : paddedEnc.getOrder()[0]);
    return out;
  }

  // bool checkPerformanceProperties(MemDescType srcTy,
  //                                 RankedTensorType dstTy) const {
  //   // The transposed load lowering logic assumes that double-rate MFMA (
  //   // mfma32x32x16 and mfma16x16x32) instructions are used whenever
  //   possible.
  //   // This code verifies whether double-rate MFMA instructions are being
  //   used
  //   // and falls back to the default path if they are not. (Note: The
  //   lowering
  //   // logic for double-rate MFMA is the same as for single-rate (mfma32x32x8
  //   // and mfma16x16x16) with kpack=2). This check should be removed once
  //   // double-rate MFMA support is fully implemented in the compiler, leaving
  //   // only an assertion. Currently, single-rate configurations with kpack=1
  //   are
  //   // still in use, so in such cases, we revert to the default lowering
  //   logic
  //   // without LDS transpose read instructions.
  //   auto dotEnc =
  //   llvm::dyn_cast_or_null<DotOperandEncodingAttr>(dstTy.getEncoding()); if
  //   (!dotEnc) {
  //     return false;
  //   }

  //   auto wmmaEnc =
  //   llvm::dyn_cast_or_null<AMDWmmaEncodingAttr>(dotEnc.getParent()); auto
  //   mfmaEnc =
  //   llvm::dyn_cast_or_null<AMDMfmaEncodingAttr>(dotEnc.getParent()); int32_t
  //   mDim = -1; if (wmmaEnc) {
  //     return true;
  //     //mDim = wmmaEnc.getMNKDimPerInstr()[0];
  //   } else if (mfmaEnc) {
  //     mDim = mfmaEnc.getMDim();
  //   } else {
  //     return false;
  //   }
  //   assert((mDim == 32 || mDim == 16) && "Invalid MFMA or WMMA instruction
  //   dimension");

  //   int rank = dstTy.getRank();
  //   auto bitwidth = this->typeConverter->convertType(dstTy.getElementType())
  //                       .getIntOrFloatBitWidth();
  //   int32_t kWidth = dotEnc.getKWidth();

  //   const auto shape = dstTy.getShape();
  //   const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
  //   const bool isLargeTile = shape[kDim] >= largeTileThreshold;

  //   const int kWidthLargeTile = 8 * kFactor;
  //   const int kWidthSmallTile = 4 * kFactor;
  //   // For largeTile, i.e. double rated mfma is an option, it's accepted to
  //   // have kWidth set for both double and single rated mfma
  //   // For smallTile, it's only accepted to have kWidth set to single rate
  //   // mfma. Smaller kWidth is not allowed to use transposed lds load.
  //   return (isLargeTile &&
  //           llvm::is_contained({kWidthLargeTile, kWidthSmallTile}, kWidth))
  //           ||
  //          (kWidth == kWidthSmallTile);
  // }

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
    // if (!checkKWidth(srcTy, dstTy)) {
    //   return false;
    // }

    // 4. Check current limitations.
    // if (bitwidth != 16) {
    //   return false;
    // }

    // We cannot use transpose linear layouts for wmma mx data types, because
    // the layouts don't match
    if (isUsedByDotScaledOp(localLoad)) {
      return false;
    }

    return true;
  }

  Value transLoadGfx1250(ConversionPatternRewriter &rewriter, Location loc,
                         unsigned bitwidth, Type vecTy, Value vecAddr) const {

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (bitwidth == 16) {
      return LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                             "llvm.amdgcn.ds.load.tr16.b128",
                                             {vecTy}, {vecAddr})
          .getResult(0);
    }
    // This works for fp8 and fp4, because fp4 in reality does not exist.
    auto l = LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                             "llvm.amdgcn.ds.load.tr8.b64",
                                             {vec_ty(i32_ty, 2)}, {vecAddr})
                 .getResult(0);
    l = b.bitcast(l, vec_ty(i8_ty, 8));
    return l;
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
    auto ldsTransLayout = chooseDsReadB64Tr16Layout(dotEnc, shape, llBitwidth);
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
          if (targetInfo.getISAFamily() == AMD::ISAFamily::GFX1250) {
            auto vecVal =
                transLoadGfx1250(rewriter, loc, bitwidth, vecTy, vecAddr);
            for (int v = 0; v < vecTy.getNumElements(); v++) {
              outVals.push_back(
                  b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
            }
          } else if constexpr (isPackedLoad) {
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
              if (targetInfo.requiresAliasInfoForAsyncOps()) {
                AMD::addLocalLoadNoAliasScope(op, dsReadOp);
              }
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
              if (targetInfo.requiresAliasInfoForAsyncOps()) {
                AMD::addLocalLoadNoAliasScope(op, dsReadOp);
              }
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

class LocalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp> {
public:
  LocalBarrierOpConversion(const LLVMTypeConverter &converter,
                           const AMD::TargetInfo &targetInfo,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::LocalBarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCDNA(targetInfo.getISAFamily()) ||
        targetInfo.getISAFamily() == AMD::ISAFamily::GFX1250)
      return failure();
    // In CDNA we can lower local_barrier to s_waitcnt + s_barrier
    // - s_waitcnt specifies how many operations to VMEM/LDS can be outstanding
    //   when the instruction completes.
    //   In this case we require 0 outstanding LDS operations
    // - s_barrier syncronizes the execution for the CTA
    constexpr int32_t ldsOnlyBits = ~(0x1f << 8);
    Location loc = op->getLoc();
    ROCDL::SWaitcntOp::create(rewriter, loc, ldsOnlyBits);
    rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);

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
  PatternBenefit barrierBenefit = PatternBenefit(benefit.getBenefit() + 1);

  patterns.add<TransLocalLoadOpConversion<triton::gpu::LocalLoadOp>>(
      typeConverter, targetInfo, transBenefit);
  /*
  patterns.add<
      TransLocalLoadOpConversion<triton::amdgpu::LocalLoadPackedTransposedOp>>(
      typeConverter, targetInfo, benefit);
  */
  patterns.add<LocalBarrierOpConversion>(typeConverter, targetInfo,
                                         barrierBenefit);
}
