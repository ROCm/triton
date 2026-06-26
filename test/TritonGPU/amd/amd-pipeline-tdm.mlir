// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-descriptor-encoding -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline="use_async_copy=1" -canonicalize | FileCheck %s

// A/B descriptor loads are converted to TDM copies, but a descriptor load used
// as the dot accumulator operand remains a raw tt.descriptor_load. Dynamic loop
// predication must guard that raw load directly.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 4]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_load_accumulator_predicated(
      %a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %acc_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c15_i32 = arith.constant 15 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %m = arith.muli %pid_m, %c256_i32 : i32
    %n = arith.muli %pid_n, %c64_i32 : i32
    %k_stride = arith.extsi %K : i32 to i64
    %a_desc = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%k_stride, %c1_i64] : <f32>, <256x16xf32>
    %n_stride = arith.extsi %N : i32 to i64
    %b_desc = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%n_stride, %c1_i64] : <f32>, <16x64xf32>
    %acc_desc = tt.make_tensor_descriptor %acc_ptr, [%M, %N], [%n_stride, %c1_i64] : <f32>, <256x64xf32>
    %c_desc = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%n_stride, %c1_i64] : <f32>, <256x64xf32>
    %k_plus = arith.addi %K, %c15_i32 : i32
    %num_k = arith.divsi %k_plus, %c16_i32 : i32
    %accumulator:2 = scf.for %iv = %c0_i32 to %num_k step %c1_i32 iter_args(%k_off = %c0_i32, %acc = %zero) -> (i32, tensor<256x64xf32, #mma>)  : i32 {
      %a = tt.descriptor_load %a_desc[%m, %k_off] : !tt.tensordesc<256x16xf32> -> tensor<256x16xf32, #blocked>
      %b = tt.descriptor_load %b_desc[%k_off, %n] : !tt.tensordesc<16x64xf32> -> tensor<16x64xf32, #blocked1>
      %acc_load = tt.descriptor_load %acc_desc[%m, %n] : !tt.tensordesc<256x64xf32> -> tensor<256x64xf32, #mma>
      %a_dot = ttg.convert_layout %a : tensor<256x16xf32, #blocked> -> tensor<256x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_dot = ttg.convert_layout %b : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_dot, %b_dot, %acc_load, inputPrecision = tf32 : tensor<256x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x64xf32, #mma>
      %next_k = arith.addi %k_off, %c16_i32 : i32
      scf.yield %next_k, %d : i32, tensor<256x64xf32, #mma>
    }
    %out = ttg.convert_layout %accumulator#1 : tensor<256x64xf32, #mma> -> tensor<256x64xf32, #blocked1>
    tt.descriptor_store %c_desc[%m, %n], %out : !tt.tensordesc<256x64xf32>, tensor<256x64xf32, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: tt.func @descriptor_load_accumulator_predicated
// CHECK: scf.if {{.*}} -> (tensor<256x64xf32
// CHECK-NEXT: tt.descriptor_load
// CHECK-NEXT: tt.dot
// CHECK-NEXT: scf.yield
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield
