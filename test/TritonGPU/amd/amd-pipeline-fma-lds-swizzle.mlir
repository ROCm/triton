// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline | FileCheck %s

// Bank-conflict swizzling for FMA dot operands. Upstream leaves a blocked-parent
// dot operand unswizzled (vec = 1, perPhase = 1, maxPhase = 1), so lanes that
// share a bank on the LDS write serialize.

// -----

// FMA dot, B operand (opIdx = 1) arriving K-contiguous, so the shared buffer is
// flipped to operand-major and K becomes the strided dim. threadsPerWarp[K] = 4
// lanes land on one bank, so 4 phases separate them.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#fma = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_operand_b_is_swizzled
  tt.func @fma_operand_b_is_swizzled(
                %argB: tensor<64x64x!tt.ptr<f32>, #blocked>,
                %argA: tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<64x64xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<64x64xf32, #fma>) : i32 {
      %b = tt.load %argB : tensor<64x64x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<64x64xf32, #fma>
      scf.yield %c : tensor<64x64xf32, #fma>
    }
    tt.return %result : tensor<64x64xf32, #fma>
  }
}

// -----

// Only 2 lanes collide, so the swizzle stays at 2 phases even though the tile
// is wide enough for 8: surplus phases only cost address arithmetic.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#fma = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 2, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_swizzle_is_capped_by_colliding_lanes
  tt.func @fma_swizzle_is_capped_by_colliding_lanes(
                %argB: tensor<32x32x!tt.ptr<f32>, #blocked>,
                %argA: tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<32x32xf32, #fma>) : i32 {
      %b = tt.load %argB : tensor<32x32x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<32x32xf32, #fma>
      scf.yield %c : tensor<32x32xf32, #fma>
    }
    tt.return %result : tensor<32x32xf32, #fma>
  }
}

// -----

// Each lane holds 2 contiguous elements along the strided dim, so one phase
// spans 2 rows.

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#fma = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_swizzle_per_phase_follows_size_per_thread
  tt.func @fma_swizzle_per_phase_follows_size_per_thread(
                %argB: tensor<64x64x!tt.ptr<f32>, #blocked>,
                %argA: tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<64x64xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<64x64xf32, #fma>) : i32 {
      %b = tt.load %argB : tensor<64x64x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<64x64xf32, #fma>
      scf.yield %c : tensor<64x64xf32, #fma>
    }
    tt.return %result : tensor<64x64xf32, #fma>
  }
}

// -----

// A 16-element f32 row occupies half of the 32 LDS banks, so adjacent rows
// start 16 banks apart. Change phase every two rows, when the row stride wraps.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#fma = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 2, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_swizzle_accounts_for_narrow_rows
  tt.func @fma_swizzle_accounts_for_narrow_rows(
                %argB: tensor<16x16x!tt.ptr<f32>, #blocked>,
                %argA: tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<16x16xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x16xf32, #fma>) : i32 {
      %b = tt.load %argB : tensor<16x16x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<16x16xf32, #fma>
      scf.yield %c : tensor<16x16xf32, #fma>
    }
    tt.return %result : tensor<16x16xf32, #fma>
  }
}

// -----

// Negative case: other FMA element types use different LDS access widths and
// packed patterns, so leave them unswizzled until those cases are handled.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#fma = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_non_f32_operand_is_not_swizzled
  tt.func @fma_non_f32_operand_is_not_swizzled(
                %argB: tensor<64x64x!tt.ptr<f16>, #blocked>,
                %argA: tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<64x64xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<64x64xf32, #fma>) : i32 {
      %b = tt.load %argB : tensor<64x64x!tt.ptr<f16>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #fma}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<64x64xf32, #fma>
      scf.yield %c : tensor<64x64xf32, #fma>
    }
    tt.return %result : tensor<64x64xf32, #fma>
  }
}

// -----

// Negative case: a single lane spans the strided dim, so no two lanes share a
// bank and the buffer keeps the unswizzled layout.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
#fma = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_no_bank_conflict_is_left_unswizzled
  tt.func @fma_no_bank_conflict_is_left_unswizzled(
                %argB: tensor<32x32x!tt.ptr<f32>, #blocked>,
                %argA: tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<32x32xf32, #fma>) : i32 {
      %b = tt.load %argB : tensor<32x32x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<32x32xf32, #fma>
      scf.yield %c : tensor<32x32xf32, #fma>
    }
    tt.return %result : tensor<32x32xf32, #fma>
  }
}

// -----

// Negative case: MFMA dot. Only blocked-parent (FMA) operands take this path;
// accelerator operands keep the layout the dot-operand constructor picked.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 2, maxPhase = 8, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_operand_is_not_swizzled_by_fma_path
  tt.func @mfma_operand_is_not_swizzled_by_fma_path(
                %argB: tensor<32x32x!tt.ptr<f32>, #blocked>,
                %argA: tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<16x32xf32, #mma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x32xf32, #mma>) : i32 {
      %b = tt.load %argB : tensor<32x32x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x32xf32, #mma>
      scf.yield %c : tensor<16x32xf32, #mma>
    }
    tt.return %result : tensor<16x32xf32, #mma>
  }
}
