// RUN: triton-translate --target=llvmir --sm=80 %s | FileCheck %s

// == LLVM IR check begin ==
// CHECK-LABEL: ; ModuleID = 'LLVMDialectModule'
// CHECK: define amdgpu_kernel void @test_empty_kernel
// XHECK: !nvvm.annotations
// XHECK: !{ptr @test_empty_kernel, !"maxntidx", i32 128}

module attributes {"triton_gpu.nvidia-target" = #triton_gpu.targetNvidiaInfo<computeCapability = 80>, "triton_gpu.common-target" = #triton_gpu.targetCommonInfo<triple = "nvptx64-nvidia-cuda",warpSize = 32>, "triton_gpu.num-warps" = 4 : i32} {

func.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  return
}

}
