// RUN: simulator.py %s | FileCheck %s

// CHECK: 0000:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
// CHECK: 0001:-:1:n:--   s_lshl_b32 s22, s14, 16
// CHECK: 0002:-:2:n:--   v_exp_f32_e32 v242, v132
// CHECK: 0006:-:6:n: 1   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK: 0007:-:7:n:--   s_lshl_b32 s22, s14, 16

v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
s_lshl_b32 s22, s14, 16
v_exp_f32_e32 v242, v132
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
s_lshl_b32 s22, s14, 16

// Notes:
// 1. The v_pk_add cannot be issued at cycle 3 since tri-exec of wmma+exp+valu
//    is not allowed.


// CHECK: 0008:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
// CHECK: 0009:-:1:n:--   s_lshl_b32 s22, s14, 16
// CHECK: 0010:-:2:n:--   s_lshl_b32 s22, s14, 16
// CHECK: 0011:-:3:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK: 0012:-:4:n:--   s_lshl_b32 s22, s14, 16
// CHECK: 0014:-:6:n:--   v_exp_f32_e32 v242, v132
// CHECK: 0017:-:-:n: 3   v_pk_add_f32 v[106:107], v[226:227], v[228:229]

v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
s_lshl_b32 s22, s14, 16
s_lshl_b32 s22, s14, 16
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
s_lshl_b32 s22, s14, 16
v_exp_f32_e32 v242, v132
v_pk_add_f32 v[106:107], v[226:227], v[228:229]

// Notes:
// 2. The last v_pk_add is delay by 2 reasons (reason sum = 3)
//    - It cannot be issued at cycle 15 since wmma+exp+valu cannot tri-execute ==> reason value 1
//    - It cannot be issued at cycle 16 due to the V rule ==> reason value 2
