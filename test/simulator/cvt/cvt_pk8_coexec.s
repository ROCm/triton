// RUN: simulator.py %s | FileCheck %s

// CHECK: 0000:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
// CHECK: 0008:-:-:n:--   v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0
// CHECK: 0012:-:-:n:--   ld_scale
// CHECK: 0013:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK: 0021:-:-:n:--   v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0

v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0
v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0

// Notes:
// 1. cvt_pk8 instructions do not co-exec with wmma


// CHECK: 0025:-:-:n:--   v_exp_f32_e32 v241, v132
// CHECK: 0027:-:-:n:16   v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0

v_exp_f32_e32 v241, v132
v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0

// Notes:
// 2. cvt_pk8 instructions do not co-exec with exp
