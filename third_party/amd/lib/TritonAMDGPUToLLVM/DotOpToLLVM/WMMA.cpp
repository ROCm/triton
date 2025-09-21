/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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

#include "../PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
// #include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::AMD {
namespace {

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;

enum class WMMAInstrType : uint8_t {
  // D = AB + C;
  // typeof(D) == typeof(C)
  // typeof(A) == typeof(B)
  // typeof(D), typeof(A):
  FP32_FP16,
  FP32_BF16,
  FP16_FP16,
  BF16_BF16,
  I32_I8,
  I32_I4,
  NOT_APPLICABLE,
};

using ValueTable = std::map<std::tuple<unsigned, unsigned, unsigned>, Value>;

ValueTable getValuesFromDotOperandLayoutStruct(
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter *typeConverter,
    Value value, int batch, int n0, int n1, int kWidth, Type type, bool isFp6,
    Location loc) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = unpackLLElements(loc, value, rewriter);

  int vecSize = kWidth;
  if (isFp6)
    vecSize = vecSize * 3 / 4;

  ValueTable vals;
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        const auto actualVecSize = kWidth == 2 ? vecSize * 2 : vecSize;
        Type elemTy = typeConverter->convertType(type);
        Type ty = vec_ty(elemTy, actualVecSize);
        Value rawElems = tb.undef(ty);
        for (size_t i = 0; i < actualVecSize; ++i) {
          Value zero;
          if (auto t = dyn_cast<mlir::IntegerType>(elemTy)) {
            if (t.getWidth() == 8) {
              const int i8bias = 127;
              zero = rewriter.create<LLVM::ConstantOp>(
                  loc, t, rewriter.getIntegerAttr(t, i8bias));
            } else
              zero = rewriter.create<LLVM::ConstantOp>(
                  loc, t, rewriter.getIntegerAttr(t, 0));
          } else {
            zero = rewriter.create<LLVM::ConstantOp>(
                loc, elemTy, rewriter.getFloatAttr(elemTy, 0.0));
          }
          tb.insert_element(ty, rawElems, zero, tb.i32_val(i));
        }

        for (int k = 0, ki = 0; k < kWidth; ++k) {
          if (isFp6 && ((k + 1) % 4 == 0))
            continue;
          rawElems = tb.insert_element(
              ty, rawElems,
              elems[n0 * n1 * kWidth * b + kWidth * (n1 * i + j) + k],
              tb.i32_val(ki++));
        }

        Value convertedElems;
        if (type.isF16() || type.isBF16() && kWidth == 16) {
          convertedElems = rawElems;
        } else if (type.isBF16()) {
          convertedElems = tb.bitcast(rawElems, vec_ty(i16_ty, kWidth));
        } else if (kWidth == 1) {
          convertedElems = tb.zext(i32_ty, tb.bitcast(rawElems, i8_ty));
        } else if (((kWidth == 2) || (kWidth == 4)) &&
                   type.getIntOrFloatBitWidth() == 8) {
          convertedElems = tb.bitcast(rawElems, i32_ty);
        } else if (kWidth == 8 && type.getIntOrFloatBitWidth() == 8) {
          convertedElems = tb.bitcast(rawElems, i64_ty);
        } else {
          convertedElems = tb.bitcast(
              rawElems, vec_ty(i32_ty, vecSize * type.getIntOrFloatBitWidth() /
                                           i32_ty.getIntOrFloatBitWidth()));
        }
        vals[{b, i, j}] = convertedElems;
      }
    }
  }
  return vals;
}

WMMAInstrType getWMMAInstrTypeFromDot(DotOp op) {
  auto aOperandTy = op.getA().getType();
  auto aTensorTy = cast<RankedTensorType>(aOperandTy);
  auto aElemTy = aTensorTy.getElementType();
  auto bOperandTy = op.getB().getType();
  auto bTensorTy = cast<RankedTensorType>(bOperandTy);
  auto bElemTy = bTensorTy.getElementType();
  assert(aElemTy == bElemTy);
  auto cOperandTy = op.getC().getType();
  auto cTensorTy = cast<RankedTensorType>(cOperandTy);
  auto cElemTy = cTensorTy.getElementType();
  auto dOperandTy = op.getD().getType();
  auto dTensorTy = cast<RankedTensorType>(dOperandTy);
  auto dElemTy = dTensorTy.getElementType();
  assert(cElemTy == dElemTy);

  if (dElemTy.isF32() && aElemTy.isF16())
    return WMMAInstrType::FP32_FP16;
  if (dElemTy.isF32() && aElemTy.isBF16())
    return WMMAInstrType::FP32_BF16;
  if (dElemTy.isF16() && aElemTy.isF16())
    return WMMAInstrType::FP16_FP16;
  if (dElemTy.isBF16() && aElemTy.isBF16())
    return WMMAInstrType::BF16_BF16;
  if (dElemTy.isInteger(32) && aElemTy.isInteger(8))
    return WMMAInstrType::I32_I8;
  if (dElemTy.isInteger(32) && aElemTy.isInteger(4))
    return WMMAInstrType::I32_I4;

  return WMMAInstrType::NOT_APPLICABLE;
}

Value generateROCDLOp(ConversionPatternRewriter &rewriter, Location loc,
                      WMMAInstrType wmmaType, Value valA, Value valB,
                      Value valC, Type aElType, Type bElType) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto resType = valC.getType();
  Value falseFlag = b.int_val(1, false);
  switch (wmmaType) {
  case WMMAInstrType::FP32_FP16:
    return rewriter.create<ROCDL::wmma_f32_16x16x16_f16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC});
  case WMMAInstrType::FP32_BF16:
    return rewriter.create<ROCDL::wmma_f32_16x16x16_bf16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC});
  case WMMAInstrType::FP16_FP16:
    return rewriter.create<ROCDL::wmma_f16_16x16x16_f16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
  case WMMAInstrType::BF16_BF16:
    return rewriter.create<ROCDL::wmma_bf16_16x16x16_bf16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
  case WMMAInstrType::I32_I8:
    return rewriter.create<ROCDL::wmma_i32_16x16x16_iu8>(
        loc, TypeRange{resType},
        ValueRange{b.int_val(1, !aElType.isUnsignedInteger()), valA,
                   b.int_val(1, !bElType.isUnsignedInteger()), valB, valC,
                   falseFlag});
  case WMMAInstrType::I32_I4:
    return rewriter.create<ROCDL::wmma_i32_16x16x16_iu4>(
        loc, TypeRange{resType},
        ValueRange{b.int_val(1, !aElType.isUnsignedInteger()), valA,
                   b.int_val(1, !bElType.isUnsignedInteger()), valB, valC,
                   falseFlag});
  default:
    llvm::report_fatal_error("WMMA data type not supported");
  }
  return Value();
}

std::string getTypeStr(Type ty) {
  std::string scalarName;
  if (ty.isF32()) {
    scalarName = "f32";
  } else if (ty.isF16()) {
    scalarName = "f16";
  } else if (llvm::isa<Float8E4M3FNType>(ty)) {
    scalarName = "fp8";
  } else if (llvm::isa<Float8E5M2Type>(ty)) {
    scalarName = "bf8";
  } else if (ty.isBF16()) {
    scalarName = "bf16";
  } else if (ty.isInteger(32)) {
    scalarName = "i32";
  } else if (ty.isInteger(16)) {
    scalarName = "i16";
  } else if (ty.isInteger(8)) {
    scalarName = "iu8";
  } else if (ty.isInteger(4)) {
    scalarName = "iu4";
  } else if (llvm::isa<Float8E4M3FNType>(ty)) {
    scalarName = "fp8";
  } else if (llvm::isa<Float8E5M2Type>(ty)) {
    scalarName = "bf8";
  } else if (auto vecTy = dyn_cast<VectorType>(ty)) {
    auto elemType = vecTy.getElementType();
    auto numElems = vecTy.getNumElements();
    scalarName = "v" + std::to_string(numElems) + getTypeStr(elemType);
  } else {
    llvm::report_fatal_error("WMMA data type not supported");
  }
  return scalarName;
}

StringRef getWmmaIntrinsicName(Type aElTy, Type bElTy, Type dElTy, Type valATy,
                               Type valCTy, bool tied) {
  static llvm::SmallDenseMap<llvm::hash_code, std::string> intrinsics;
  using MapInfo = llvm::DenseMapInfo<Type>;
  llvm::hash_code h = llvm::hash_combine(
      MapInfo::getHashValue(aElTy), MapInfo::getHashValue(bElTy),
      MapInfo::getHashValue(dElTy), MapInfo::getHashValue(valATy),
      MapInfo::getHashValue(valCTy), llvm::hash_value(tied));
  int64_t kWidth = cast<VectorType>(valATy).getNumElements();
  int bitness = aElTy.getIntOrFloatBitWidth();
  if (!intrinsics.contains(h)) {
    std::string name = "llvm.amdgcn.wmma.";
    name += getTypeStr(dElTy);
    if (bitness == 8)
      name += ".16x16x64."; // TODO support 16x16x32 for i4 operands
    else if (bitness == 16)
      name += ".16x16x32."; // TODO support 16x16x32 for i4 operands
    else
      llvm::report_fatal_error("invalid kWidth during intrinsic selection");

    name += getTypeStr(aElTy);
    if (tied) {
      name += ".tied";
    } else {
      if (isa<FloatType>(aElTy) && aElTy.getIntOrFloatBitWidth() == 8)
        name += '.' + getTypeStr(bElTy);
      name += '.' + getTypeStr(valCTy) + "." + getTypeStr(valATy);
    }
    intrinsics[h] = name;
  }
  return intrinsics[h];
}

std::string addInstructionSuffix(std::string intrinsicName, unsigned kWidth,
                                 Type aElTy, Type bElTy, Type dElTy,
                                 bool tied) {
  if (tied) {
    intrinsicName += ".tied";
  } else {
    if (isa<FloatType>(aElTy) && aElTy.getIntOrFloatBitWidth() == 8)
      intrinsicName += "." + getTypeStr(bElTy);
    intrinsicName += ".v" + std::to_string(kWidth) + getTypeStr(dElTy);
    intrinsicName += ".v" + std::to_string(kWidth) + getTypeStr(aElTy);
  }

  return intrinsicName;
}

static inline int32_t getWmmaF8F6F4MatrixFormat(Type t) {
  return llvm::TypeSwitch<Type, int32_t>(t)
      .Case<Float8E4M3FNType>([](Type) { return 0; })
      .Case<Float8E5M2Type>([](Type) { return 1; })
      .Case<Float6E2M3FNType>([](Type) { return 2; })
      .Case<Float6E3M2FNType>([](Type) { return 3; })
      .Case<Float4E2M1FNType>([](Type) { return 4; })
      .Default([](Type) { return -1; });
}

Value generateWMMAIntrinsic(ConversionPatternRewriter &rewriter, Location loc,
                            Value valA, Value valB, Value valC, Type aElType,
                            Type bElType, Type dElType, StringRef name,
                            std::optional<bool> tiedLower) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  LLVM::FastmathFlagsAttr defaultFlags{};
  SmallVector<Value> operands;
  int64_t kWidth = cast<VectorType>(valA.getType()).getNumElements();
  if (aElType.isInteger())
    operands.push_back(b.int_val(1, !aElType.isUnsignedInteger()));

  if (kWidth == 16 && (aElType.isBF16() || aElType.isF16()))
    operands.push_back(b.int_val(1, 0));
  operands.push_back(valA);

  if (kWidth == 16 && (bElType.isBF16() || bElType.isF16()))
    operands.push_back(b.int_val(1, 0));

  if (bElType.isInteger())
    operands.push_back(b.int_val(1, !bElType.isUnsignedInteger()));
  operands.push_back(valB);

  if (kWidth == 16 || (kWidth == 8 && aElType.getIntOrFloatBitWidth() == 8))
    operands.push_back(b.int_val(16, 0));
  operands.push_back(valC);

  // Flag for using low bits in registers. Result could be already packed to
  // int32. Set low bits by default for now.
  if (tiedLower.has_value() || 32 / dElType.getIntOrFloatBitWidth() > 1 ||
      dElType.isInteger(32)) {
    operands.push_back(b.int_val(1, tiedLower.value_or(false)));
  }

  // add two addtional operands perf llvm changes
  operands.push_back(b.i1_val(0));
  operands.push_back(b.i1_val(0));

  auto wmmaIntrinsic = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, name, valC.getType(), operands);
  return wmmaIntrinsic.getResult(0);
}

Value generateScaledWMMAIntrinsic(ConversionPatternRewriter &rewriter,
                                  Location loc, Value valA, Value valScaleA,
                                  Value valB, Value valScaleB, Value valC,
                                  Type aElType, Type bElType, Type dElType,
                                  int scaleKWidth) {
  assert(scaleKWidth == 2 || scaleKWidth == 4 || scaleKWidth == 8);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  std::string name = "llvm.amdgcn.wmma.scale";
  if (scaleKWidth == 8) {
    name += "16";
  }
  name += ".f32.16x16x128.f8f6f4";

  LLVM::FastmathFlagsAttr defaultFlags{};
  SmallVector<Value> operands;

  Value fmtA = b.i32_val(getWmmaF8F6F4MatrixFormat(aElType));
  operands.push_back(fmtA);
  operands.push_back(valA);
  Value fmtB = b.i32_val(getWmmaF8F6F4MatrixFormat(bElType));
  operands.push_back(fmtB);
  operands.push_back(valB);
  Value modC = b.i16_val(0);
  operands.push_back(modC);
  operands.push_back(valC);
  operands.push_back(b.i32_val(0));
  operands.push_back(b.i32_val(0));
  operands.push_back(valScaleA);
  operands.push_back(b.i32_val(0));
  operands.push_back(b.i32_val(0));
  operands.push_back(valScaleB);
  operands.push_back(b.i1_val(0));
  operands.push_back(b.i1_val(0));
  auto wmmaIntrinsic = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, name, valC.getType(), operands);
  return wmmaIntrinsic.getResult(0);
}

Value generateWMMAOp(ConversionPatternRewriter &rewriter, Location loc,
                     Value valA, Value valB, Value valC, Type aElType,
                     Type bElType, Type dElType, StringRef intrinsicName,
                     std::optional<bool> tiedLower) {
  // Independent of wmma version because builtin functions are backward
  // compatible
  return generateWMMAIntrinsic(rewriter, loc, valA, valB, valC, aElType,
                               bElType, dElType, intrinsicName, tiedLower);
}

// Conduct the Dot conversion.
LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         const LLVMTypeConverter *typeConverter) {
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  int wmmaVer = wmmaLayout.getVersion();
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto mnkDim = wmmaLayout.getMNKDimPerInstr();

  auto loc = op.getLoc();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  Value a = op.getA();
  Value b = op.getB();
  Value d = op.getD();
  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  auto aElemTy = aTensorTy.getElementType();
  auto bElemTy = bTensorTy.getElementType();
  auto dElemTy = dTensorTy.getElementType();

  const auto kDimOperandSize = aTensorTy.getShape().back();

  std::string intrinsicName;
  FailureOr<WmmaIntrinsic> maybeWmmaIntrinsic =
      WmmaIntrinsic::selectFor(wmmaVer, mnkDim[0], mnkDim[1], kDimOperandSize,
                               aElemTy, bElemTy, dElemTy);
  if (failed(maybeWmmaIntrinsic)) {

    return op.emitError(
        "no matching matrix core intrinsic due to unsupported element type");
  }

  unsigned kDim = maybeWmmaIntrinsic->kDim;

  auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  int kWidth = aEncoding.getKWidth();
  intrinsicName = maybeWmmaIntrinsic->name;
  auto repA = wmmaLayout.getRepForOperand(
      gpu::getShapePerCTA(aEncoding, aTensorTy.getShape()), aTensorTy, kWidth,
      kDim, 0);
  auto repB = wmmaLayout.getRepForOperand(
      gpu::getShapePerCTA(bEncoding, bTensorTy.getShape()), bTensorTy, kWidth,
      kDim, 1);

  assert(repA[2] == repB[1]);

  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedC = adaptor.getC();
  auto numRepM = repA[1];
  auto numRepN = repB[2];
  auto numRepK = repA[2];
  auto numRepB = repA[0];

  ValueTable ha = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedA, numRepB, numRepM, numRepK, kWidth,
      aTensorTy.getElementType(), false, loc);
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedB, numRepB, numRepN, numRepK, kWidth,
      aTensorTy.getElementType(), false, loc);
  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  unsigned warpSize = gpu::lookupThreadsPerWarp(rewriter);
  constexpr unsigned vgprElemBitWidth = 32;
  unsigned paddedOutputElemSize =
      wmmaVer == 1 ? vgprElemBitWidth / dstElemTy.getIntOrFloatBitWidth() : 1;
  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] * paddedOutputElemSize / warpSize;
  auto dElemsToStorePerThread = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);
  bool tied = numRepM % 2 == 0 && paddedOutputElemSize == 2;
  int tiedGroup = tied ? 2 : 1;

  intrinsicName = addInstructionSuffix(intrinsicName, kWidth, aElemTy, bElemTy,
                                       dElemTy, tied);
  for (int b = 0; b < numRepB; ++b) {
    for (int m = 0; m < numRepM / tiedGroup; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        auto batchOffIdx = b * numRepM * numRepN * dElemsToStorePerThread;
        auto nRepOffId = n * dElemsToStorePerThread;
        auto nBatchOffSum = nRepOffId + batchOffIdx;

        Value acc = tb.undef(vecTy);
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          for (int subTied = 0; subTied < tiedGroup; ++subTied) {
            auto mRepOffId =
                (m * tiedGroup + subTied) * numRepN * dElemsToStorePerThread;
            auto fcThreadOffIdx = nBatchOffSum + mRepOffId;
            acc = tb.insert_element(
                vecTy, acc, fc[fcThreadOffIdx + v],
                tb.i32_val(v * paddedOutputElemSize + subTied));
          }
        }
        for (size_t k = 0; k < numRepK; ++k) {
          for (int subTied = 0; subTied < tiedGroup; ++subTied) {
            auto optTied =
                tied ? std::optional<bool>(subTied != 0) : std::nullopt;
            acc = wmmaLayout.getIsTransposed()
                      ? generateWMMAOp(rewriter, loc, hb[{b, n, k}],
                                       ha[{b, m * tiedGroup + subTied, k}], acc,
                                       bTensorTy.getElementType(),
                                       aTensorTy.getElementType(), dstElemTy,
                                       intrinsicName, optTied)
                      : generateWMMAOp(
                            rewriter, loc, ha[{b, m * tiedGroup + subTied, k}],
                            hb[{b, n, k}], acc, aTensorTy.getElementType(),
                            bTensorTy.getElementType(), dstElemTy,
                            intrinsicName, optTied);
          }
        }
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          for (int subTied = 0; subTied < tiedGroup; ++subTied) {
            auto mRepOffId =
                (m * tiedGroup + subTied) * numRepN * dElemsToStorePerThread;
            auto fcThreadOffIdx = nBatchOffSum + mRepOffId;
            fc[fcThreadOffIdx + v] = tb.extract_element(
                dstElemTy, acc, tb.i32_val(v * paddedOutputElemSize + subTied));
          }
        }
      }
    }
  }

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      wmmaLayout.getContext(), SmallVector<Type>(fc.size(), dstElemTy));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult convertScaledDot(triton::DotScaledOp op,
                               triton::DotScaledOp::Adaptor adaptor,
                               ConversionPatternRewriter &rewriter,
                               const LLVMTypeConverter *typeConverter) {
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  int wmmaVer = wmmaLayout.getVersion();
  assert(wmmaVer == 3 && "Scaled dot not supported for wmma1/wmma2");
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto mnkDim = wmmaLayout.getMNKDimPerInstr();

  auto loc = op.getLoc();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  Value a = op.getA();
  Value b = op.getB();
  Value aScale = op.getAScale();
  Value bScale = op.getBScale();
  Value d = op.getD();
  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto aScaleTensorTy = cast<RankedTensorType>(aScale.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto bScaleTensorTy = cast<RankedTensorType>(bScale.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  auto elemTy = aTensorTy.getElementType();

  int kDim = 128;
  int kWidth = 64;

  bool isFp4A = op.getAElemType() == triton::ScaleDotElemType::E2M1;
  int kWidthA = isFp4A ? kWidth / 2 : kWidth;
  int kDimA = isFp4A ? kDim / 2 : kDim;

  bool isFp4B = op.getBElemType() == triton::ScaleDotElemType::E2M1;
  int kWidthB = isFp4B ? kWidth / 2 : kWidth;
  int kDimB = isFp4B ? kDim / 2 : kDim;

  bool isFp6A = (op.getAElemType() == triton::ScaleDotElemType::E2M3) ||
                (op.getAElemType() == triton::ScaleDotElemType::E3M2);
  bool isFp6B = (op.getBElemType() == triton::ScaleDotElemType::E2M3) ||
                (op.getBElemType() == triton::ScaleDotElemType::E3M2);

  auto repA = wmmaLayout.getRepForOperand(aTensorTy.getShape(), elemTy, kWidthA,
                                          kDimA, 0);
  auto repB = wmmaLayout.getRepForOperand(bTensorTy.getShape(), elemTy, kWidthB,
                                          kDimB, 1);

  assert(repA[2] == repB[1]);

  Value loadedA = adaptor.getA();
  Value loadedAScale = adaptor.getAScale();
  Value loadedB = adaptor.getB();
  Value loadedBScale = adaptor.getBScale();
  Value loadedC = adaptor.getC();
  auto numRepM = repA[1];
  auto numRepN = repB[2];
  auto numRepK = repA[2];
  auto numRepB = repA[0];

  const auto kDimTensorA = aTensorTy.getShape().back();
  const auto kWWMADim = 128;
  int paddingFactor = 1;
  if (kWWMADim > kDimTensorA) {
    paddingFactor = kWWMADim / kDimTensorA;
  }

  auto scaleShapeA = aScaleTensorTy.getShape();
  int scaleKWidthA = 4 / paddingFactor;
  auto scaleShapeB = bScaleTensorTy.getShape();
  int scaleKWidthB = 4 / paddingFactor;
  constexpr int scaleKBase = 1;

  ValueTable ha = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedA, numRepB, numRepM, numRepK, kWidthA,
      aTensorTy.getElementType(), isFp6A, loc);
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedB, numRepB, numRepN, numRepK, kWidthB,
      bTensorTy.getElementType(), isFp6B, loc);
  ValueTable sa = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedAScale, numRepB, numRepM, numRepK,
      scaleKWidthA, aScaleTensorTy.getElementType(), false, loc);
  ValueTable sb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedBScale, numRepB, numRepN, numRepK,
      scaleKWidthB, bScaleTensorTy.getElementType(), false, loc);
  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  Type scaledAElemType =
      LLVM::AMD::scaleDotElemTypeToMLIRType(op.getContext(), op.getAElemType());
  Type scaledBElemType =
      LLVM::AMD::scaleDotElemTypeToMLIRType(op.getContext(), op.getBElemType());

  unsigned warpSize = gpu::lookupThreadsPerWarp(rewriter);
  constexpr unsigned vgprElemBitWidth = 32;
  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] / warpSize;
  auto dElemsToStorePerThread = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);
  for (int b = 0; b < numRepB; ++b) {
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        auto batchOffIdx = b * numRepM * numRepN * dElemsToStorePerThread;
        auto mRepOffId = m * numRepN * dElemsToStorePerThread;
        auto nRepOffId = n * dElemsToStorePerThread;
        auto fcThreadOffIdx = batchOffIdx + mRepOffId + nRepOffId;

        Value acc = tb.undef(vecTy);
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          acc = tb.insert_element(vecTy, acc, fc[fcThreadOffIdx + v],
                                  tb.i32_val(v));
        }
        for (size_t k = 0; k < numRepK; k++) {
          acc = wmmaLayout.getIsTransposed()
                    ? generateScaledWMMAIntrinsic(
                          rewriter, loc, hb[{b, n, k}], sb[{b, n, k}],
                          ha[{b, m, k}], sa[{b, m, k}], acc, scaledBElemType,
                          scaledAElemType, dstElemTy, scaleKWidthA)
                    : generateScaledWMMAIntrinsic(
                          rewriter, loc, ha[{b, m, k}], sa[{b, m, k}],
                          hb[{b, n, k}], sb[{b, n, k}], acc, scaledAElemType,
                          scaledBElemType, dstElemTy, scaleKWidthB);
        }
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          fc[fcThreadOffIdx + v] =
              tb.extract_element(dstElemTy, acc, tb.i32_val(v));
        }
      }
    }
  }

  Type structTy = LLVM::LLVMStructType::getLiteral(
      wmmaLayout.getContext(), SmallVector<Type>(fc.size(), dstElemTy));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

  // const size_t mmaCount = numRepB * numRepM * numRepN * numRepK;
  // setNumGeneratedMMAs(op, mmaCount, mnkDim[0], mnkDim[1], mnkDim[2], elemTy);

  rewriter.replaceOp(op, res);
  return success();
}

} // namespace

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDWmmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support $c with a wmma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  return convertDot(op, adaptor, rewriter, typeConverter);
}

LogicalResult convertScaledWMMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  assert(isa<LinearEncodingAttr>(op.getAScale().getType().getEncoding()) &&
         isa<LinearEncodingAttr>(op.getBScale().getType().getEncoding()) &&
         "Both LhsScale and RhsScale should be linear layout.");

  auto cTensorTy = op.getC().getType();
  auto dTensorTy = op.getD().getType();
  assert(isa<AMDWmmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support C with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's C operand should pass the same number of values as D.");

  auto loc = op.getLoc();
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  return convertScaledDot(op, adaptor, rewriter, typeConverter);
}
} // namespace mlir::triton::AMD
