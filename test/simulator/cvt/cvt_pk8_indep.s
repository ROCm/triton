// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:-:n:--   v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0
// CHECK: 0004:-:-:n:--   ld_scale
// CHECK: 0005:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK: 0008:-:3:n:--   v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0
// CHECK: 0012:-:7:n:--   s_lshr_b32 s22, s53, 4

v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0
v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0
s_lshr_b32 s22, s53, 4

// Notes:
// 1. v_cvt_scale*_pk8 takes 4 cycles to issue
