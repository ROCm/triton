// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck %s

// gfx1170 floating-point conversions.
//
// Standard fp16/bf16 <-> fp32 casts use the scalar LLVM path, identical to the
// other AMD targets covered by fp_to_fp.mlir.
//
// The OCP fp8 conversions (f8E5M2 / f8E4M3FN) are the interesting part: gfx1170
// does NOT use hardware fp8 cvt instructions for them. Unlike gfx950
// (rocdl.cvt.scalef32.pk.*) or gfx942 (rocdl.cvt.pk.{fp8,bf8}.*), the gfx1170
// lowering falls back to *software* emulation (bit manipulation: llvm.trunc to
// i8, llvm.select, llvm.lshr, ...). These tests lock that fallback in so a
// future switch to hardware fp8 cvt on gfx1170 is a deliberate, reviewed change.

// CHECK-LABEL: f16_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f16_to_f32(%arg0: tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // CHECK-COUNT-8: llvm.fpext %{{.+}} : f16 to f32
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: f32_to_f16
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f32_to_f16(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // rtne: scalar fptrunc (same as gfx942, not gfx950's packed vector fptrunc).
    // CHECK-COUNT-8: llvm.fptrunc %{{.+}} : f32 to f16
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // rtz: packed round-to-zero conversion.
    // CHECK-COUNT-4: rocdl.cvt.pkrtz
    %1 = tt.fp_to_fp %arg0, rounding = rtz : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: downcast_f32_to_ocp_f8
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @downcast_f32_to_ocp_f8(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // f32 -> f8E4M3FN: software emulation, no hardware fp8 cvt.
    // CHECK-NOT: rocdl.cvt
    // CHECK: llvm.trunc %{{.+}} : i32 to i8
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: downcast_f32_to_ocp_bf8
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @downcast_f32_to_ocp_bf8(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // f32 -> f8E5M2: software emulation, no hardware fp8 cvt.
    // CHECK-NOT: rocdl.cvt
    // CHECK: llvm.trunc %{{.+}} : i32 to i8
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: upcast_ocp_f8_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @upcast_ocp_f8_to_f32(%arg0: tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // f8E4M3FN -> f32: software emulation, no hardware fp8 cvt.
    // CHECK-NOT: rocdl.cvt
    // CHECK: llvm.select
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: upcast_ocp_bf8_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @upcast_ocp_bf8_to_f32(%arg0: tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // f8E5M2 -> f32: software emulation, no hardware fp8 cvt. E5M2 shares f16's
    // exponent layout, so the upcast is a bit-shift into f16 then fpext (no
    // llvm.select, unlike the E4M3FN path).
    // CHECK-NOT: rocdl.cvt
    // CHECK: llvm.fpext
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}
