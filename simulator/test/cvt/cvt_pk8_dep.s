// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:-:n:--   v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0
// CHECK: 0008:-:-:n: 4   v_pk_add_f32 v[106:107], v[108:109], v[228:229]

v_cvt_scalef32_pk8_fp8_f32 v[108:109], v[242:249], 1.0
v_pk_add_f32 v[106:107], v[108:109], v[228:229]

// Notes:
// 1. v_cvt_scale*_pk8 takes 8 cycles before dependent instructions
