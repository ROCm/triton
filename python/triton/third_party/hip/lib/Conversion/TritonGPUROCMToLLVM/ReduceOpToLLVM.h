#ifndef TRITON_CONVERSION_TRITONGPUROCM_TO_LLVM_REDUCE_OP_H
#define TRITON_CONVERSION_TRITONGPUROCM_TO_LLVM_REDUCE_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateReduceOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    int computeCapability, PatternBenefit benefit);

#endif
