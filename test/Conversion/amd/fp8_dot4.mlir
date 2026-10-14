// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck %s
// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 | FileCheck %s

// gfx1170 and RDNA4 ship the Dot11Insts fp8 dot4 instructions
// (v_dot4_f32_{fp8,bf8}_{fp8,bf8}). A blocked (non-WMMA) fp8 dot with
// K % 4 == 0 lowers to those intrinsics instead of upcasting each element to
// f32 and issuing scalar fmuladds.
//
// Note the operand form: Triton's type converter maps every fp8 type to i8, so
// the four K-consecutive elements arrive as <4 x i8> and are bitcast to a
// single i32. Unlike fdot2 and sdot4, these intrinsics take no trailing clamp
// operand, so the call must have exactly three arguments.
//
// chooseIntrinsic() does not check the target; isLegalFMAForm() is what keeps
// fp8 operands away from targets without the instruction, matching how the
// fdot2 and sdot4 branches are handled.

// CHECK-LABEL: v_dot_fp8_fp8
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @v_dot_fp8_fp8(%arg0: tensor<16x16xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xf32, #blocked>) {
    // CHECK: llvm.bitcast %{{.*}} : vector<4xi8> to i32
    // CHECK-COUNT-8: llvm.call_intrinsic "llvm.amdgcn.dot4.f32.fp8.fp8"({{.*}}) : (i32, i32, f32) -> f32
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: v_dot_bf8_bf8
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @v_dot_bf8_bf8(%arg0: tensor<16x16xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xf32, #blocked>) {
    // CHECK: llvm.bitcast %{{.*}} : vector<4xi8> to i32
    // CHECK-COUNT-8: llvm.call_intrinsic "llvm.amdgcn.dot4.f32.bf8.bf8"({{.*}}) : (i32, i32, f32) -> f32
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// A dot may mix the two fp8 types: DotOp::verify only requires A and B to share
// a bit width. Each ordering has its own instruction, so the two cases below
// must select different intrinsics.

// CHECK-LABEL: v_dot_fp8_bf8
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @v_dot_fp8_bf8(%arg0: tensor<16x16xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xf32, #blocked>) {
    // CHECK: llvm.bitcast %{{.*}} : vector<4xi8> to i32
    // CHECK-COUNT-8: llvm.call_intrinsic "llvm.amdgcn.dot4.f32.fp8.bf8"({{.*}}) : (i32, i32, f32) -> f32
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: v_dot_bf8_fp8
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @v_dot_bf8_fp8(%arg0: tensor<16x16xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg1: tensor<16x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg2: tensor<16x16xf32, #blocked>) {
    // CHECK: llvm.bitcast %{{.*}} : vector<4xi8> to i32
    // CHECK-COUNT-8: llvm.call_intrinsic "llvm.amdgcn.dot4.f32.bf8.fp8"({{.*}}) : (i32, i32, f32) -> f32
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}
