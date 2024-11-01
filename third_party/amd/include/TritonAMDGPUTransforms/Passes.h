#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace amdgpu {
struct ClusterInfo {
  int clusterDimX{1};
  int clusterDimY{1};
  int clusterDimZ{1};
};
} // namespace amdgpu
} // namespace triton

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "TritonAMDGPUTransforms/Passes.h.inc"

} // namespace mlir

namespace mlir::triton::amdgpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL_TRITONAMDGPUOPTIMIZEDOTOPERANDS
#include "TritonAMDGPUTransforms/Passes.h.inc"

void registerTritonAMDGPUOptimizeDotOperands();
} // namespace mlir::triton::amdgpu

namespace mlir {
std::unique_ptr<Pass> createTritonAMDGPUPlanCTAPass(
    triton::amdgpu::ClusterInfo *clusterInfo = nullptr);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUTransforms/Passes.h.inc"
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
