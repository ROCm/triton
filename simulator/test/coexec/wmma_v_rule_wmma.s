// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
// CHECK: 0001:-:1:n:--   ds_load_tr8_b64 v[242:243], v99 offset:43008
// CHECK: 0002:-:2:n:--   s_lshr_b32 s22, s53, 4
// CHECK: 0003:-:3:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK: 0004:-:4:n:--   s_lshr_b32 s22, s53, 4
// CHECK: 0005:-:5:n:--   s_lshr_b32 s22, s53, 4
// CHECK: 0006:-:6:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK: 0007:-:7:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK: 0008:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0

v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
ds_load_tr8_b64 v[242:243], v99 offset:43008
s_lshr_b32 s22, s53, 4
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
s_lshr_b32 s22, s53, 4
s_lshr_b32 s22, s53, 4
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0


// Notes:
// 1. There is no delay for the 2nd wmma since the V rule does not apply to it.
