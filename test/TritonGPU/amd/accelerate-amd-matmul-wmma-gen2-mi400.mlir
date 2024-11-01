// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx1210 matrix-instruction-size=0' | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_cf16(
   %0: tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
   %1: tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
   %2: tensor<32x32x!tt.ptr<f16>, #blocked>) {
    %3 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    %4 = tt.dot %0, %1, %3 : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf16, #blocked>
    tt.store %2, %4 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
