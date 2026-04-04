#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_TYPECONVERTER_H

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonAMDGPUToLLVMTypeConverter : public TritonGPUToLLVMTypeConverter {
public:
  TritonAMDGPUToLLVMTypeConverter(MLIRContext *ctx,
                                  const LowerToLLVMOptions &options,
                                  const TargetInfoBase &targetInfo,
                                  const DataLayoutAnalysis *analysis = nullptr)
      : TritonGPUToLLVMTypeConverter(ctx, options, targetInfo, analysis) {
    addConversion([&](TensorDescType type) -> std::optional<Type> {
      return convertTensorDescType(type);
    });
  }

  Type convertTensorDescType(triton::TensorDescType type) {
    auto ctx = type.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto v4i32Ty = VectorType::get(4, i32Ty);
    auto v8i32Ty = VectorType::get(8, i32Ty);

    int rank = type.getBlockType().getRank();
    SmallVector<Type> fields;
    fields.push_back(v4i32Ty);  // group0: <4 x i32>
    fields.push_back(v8i32Ty);  // group1: <8 x i32>
    if (rank > 2) {
      fields.push_back(v4i32Ty);  // group2: <4 x i32>
      fields.push_back(v4i32Ty);  // group3: <4 x i32>
    }
    return LLVM::LLVMStructType::getLiteral(ctx, fields);
  }
};

#endif
