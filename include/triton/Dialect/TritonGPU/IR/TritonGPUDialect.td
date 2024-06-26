#ifndef TRITONGPU_DIALECT
#define TRITONGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPU_Dialect : Dialect {
  let name = "triton_gpu";

  let cppNamespace = "::mlir::triton::gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "triton_gpu.num-warps"; }
    static int getNumWarps(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-warps attribute");
      return mod->getAttr("triton_gpu.num-warps").cast<IntegerAttr>().getInt();
    }
    static int getNumCTAs(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.num-ctas"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-ctas attribute");
      return mod->getAttr("triton_gpu.num-ctas").cast<IntegerAttr>().getInt();
    }
    static int getComputeCapability(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.compute-capability"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.compute-capability attribute");
      return mod->getAttrOfType<IntegerAttr>("triton_gpu.compute-capability").getInt();
    }
    void registerTypes();

    static std::string getThreadsPerWarpAttrName() { return "triton_gpu.threads-per-warp"; }

    static int getThreadsPerWarp(ModuleOp mod) {
      Attribute threadsPerWarp = mod->getDiscardableAttr("triton_gpu.threads-per-warp");
      if(!threadsPerWarp) {
        return TRITONGPU_DEFAULT_WARPSIZE;
      }
      return threadsPerWarp.cast<IntegerAttr>().getInt();
    }
    static int getSharedSize(ModuleOp mod) {
      Attribute sharedAttr = mod->getDiscardableAttr("triton_gpu.shared");
      if(!sharedAttr) {
        return 0;
      }
      return sharedAttr.cast<IntegerAttr>().getInt();
    }

  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
