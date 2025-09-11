// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx942 matrix-instruction-size=0" | FileCheck %s --check-prefixes MFMA0,CHECK,CHECK-GFX942
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx942 matrix-instruction-size=16" | FileCheck %s --check-prefixes MFMA16,CHECK,CHECK-GFX942
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx950 matrix-instruction-size=16" | FileCheck %s --check-prefixes MFMA16,CHECK,CHECK-GFX950

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_fp8e5m2
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_fp8e5m2(
      %arg0: tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x256x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // CHECK-GFX942: %[[A0:.+]] = ttg.convert_layout %arg0 : {{.*}} -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    // CHECK-GFX942: %[[A1:.+]] = tt.fp_to_fp %[[A0]] : {{.*}} -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    // CHECK-GFX942: %[[B0:.+]] = ttg.convert_layout %arg1 : {{.*}} -> tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK-GFX942: %[[B1:.+]] = tt.fp_to_fp %[[B0]] : tensor<64x256xf8E5M2, {{.*}} -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK-GFX942: tt.dot %[[A1]], %[[B1]]
    // CHECK-GFX950: %[[A:.+]] = ttg.convert_layout %arg0 : {{.*}} -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK-GFX950: %[[B:.+]] = ttg.convert_layout %arg1 : {{.*}} -> tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK-GFX950: tt.dot_scaled %[[A]], %[[B]]
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Verify that we use FMA when the N dimension is too small for any mma.
// MFMA0-NOT: #ttg.amd_mfma
// MFMA16: #ttg.amd_mfma
// CHECK-LABEL: small_n_size
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @small_n_size(
    %a: tensor<4x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>)
    -> tensor<4x128xf32, #blocked> {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<4x128xf32, #blocked>
    %result = tt.dot %a, %b, %zero_f32 : tensor<4x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x128xf32, #blocked>
    tt.return %result : tensor<4x128xf32, #blocked>
  }
}

// -----

// MFMA0-NOT: amd_mfma
// MFMA16: amd_mfma
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_small_k
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_small_k(
      %arg0: tensor<128x4xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<4x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x256x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x4xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<4x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: mfma_dot_scale_warp1
// CHECK-GFX950: dot_scaled
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scale_warp1(
      %A: tensor<16x128xi8, #blocked>,
      %B: tensor<128x64xi8, #blocked>,
      %AS: tensor<16x8xi8, #blocked1>,
      %BS: tensor<64x8xi8, #blocked1>,
      %ADDR: tensor<16x64x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %C = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked>
    %D = tt.dot_scaled %A scale %AS, %B scale %BS, %C lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x128xi8, #blocked>, tensor<16x8xi8, #blocked1> * tensor<128x64xi8, #blocked>, tensor<64x8xi8, #blocked1> -> tensor<16x64xf32, #blocked>
    tt.store %ADDR, %D : tensor<16x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
