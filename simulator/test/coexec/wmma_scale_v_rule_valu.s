// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:-:n:--   ld_scale
// CHECK-NEXT: 0001:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK-NEXT: 0002:-:1:n:--   ds_load_tr8_b64 v[242:243], v99 offset:43008
// CHECK-NEXT: 0003:-:2:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0004:-:3:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK-NEXT: 0005:-:4:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0006:-:5:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0007:-:6:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK-NEXT: 0008:-:7:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK-NEXT: 0011:-:-:n: 2   v_pk_add_f32 v[106:107], v[226:227], v[228:229]

v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
ds_load_tr8_b64 v[242:243], v99 offset:43008
s_lshr_b32 s22, s53, 4
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
s_lshr_b32 s22, s53, 4
s_lshr_b32 s22, s53, 4
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
v_pk_add_f32 v[106:107], v[226:227], v[228:229]

// Notes:
// 1. The last v_pk_add has to wait to be issued at cycle 11 due to the V rule.
