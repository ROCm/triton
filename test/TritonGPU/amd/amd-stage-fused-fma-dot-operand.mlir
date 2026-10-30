// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="gfx-arch=gfx1151" | FileCheck %s --check-prefix=OFF
// RUN: env TRITON_AMD_STAGE_FUSED_DOT_OPERANDS=1 triton-opt %s -split-input-file -tritonamdgpu-optimize-dot-operands="gfx-arch=gfx1151" | FileCheck %s --check-prefix=ON

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 2], order = [1, 0]}>

// OFF-LABEL: fused_fma_dot_operand
// OFF-NOT: ttg.local_alloc
// ON-LABEL: fused_fma_dot_operand
// ON: %[[ADD:.+]] = arith.addf
// ON: %[[ALLOC:.+]] = ttg.local_alloc %[[ADD]] : (tensor<64x16xf32, #blocked>) -> !ttg.memdesc<64x16xf32, #{{.*}}swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]
// ON: %[[LOAD:.+]] = ttg.local_load %[[ALLOC]]
// ON: %[[A:.+]] = ttg.convert_layout %[[LOAD]]
// ON: tt.dot %[[A]]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1151", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fused_fma_dot_operand(
      %a0: tensor<64x16xf32, #blocked>,
      %a1: tensor<64x16xf32, #blocked>,
      %b: tensor<16x64xf32, #blocked>,
      %acc: tensor<64x64xf32, #blocked>,
      %ub: i32) {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %0 = scf.for %k = %c0 to %ub step %c16 iter_args(%arg = %acc) -> (tensor<64x64xf32, #blocked>) : i32 {
      %add = arith.addf %a0, %a1 : tensor<64x16xf32, #blocked>
      %cvt_a = ttg.convert_layout %add : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %cvt_b = ttg.convert_layout %b : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %dot = tt.dot %cvt_a, %cvt_b, %arg : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      scf.yield %dot : tensor<64x64xf32, #blocked>
    }
    tt.return
  }
}
