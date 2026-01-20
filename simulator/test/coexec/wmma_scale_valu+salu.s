// RUN: simulator.py %s | FileCheck %s


// ---------------------------------------------
// 1st wmma at cycle 0
// salu takes E slots while valu takes I slots

// CHECK: 0000:-:-:n:--   ld_scale
// CHECK-NEXT: 0001:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK-NEXT: 0002:-:1:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0003:-:2:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0004:-:3:n:--   v_pk_add_f32 v[106:107], v[106:107], v[236:237]
// CHECK-NEXT: 0005:-:4:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0006:-:5:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0007:-:6:n:--   v_pk_add_f32 v[108:109], v[108:109], v[236:237]
// CHECK-NEXT: 0008:-:7:n:--   v_pk_add_f32 v[110:111], v[110:111], v[236:237]

# 1st wmma at 0
v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
s_lshr_b32 s22, s53, 4
s_lshr_b32 s22, s53, 4
v_pk_add_f32 v[106:107], v[106:107], v[236:237]
s_lshr_b32 s22, s53, 4
s_lshr_b32 s22, s53, 4
v_pk_add_f32 v[108:109], v[108:109], v[236:237]
v_pk_add_f32 v[110:111], v[110:111], v[236:237]

// CHECK-NEXT: 0009:-:-:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0010:-:-:n:--   s_lshr_b32 s22, s53, 4

s_lshr_b32 s22, s53, 4
s_lshr_b32 s22, s53, 4

// ---------------------------------------------
// 1st wmma at cycle 0
// salu takes I slot so valu is delayed

// CHECK: 0011:-:-:n:--   ld_scale
// CHECK-NEXT: 0012:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK-NEXT: 0013:-:1:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0014:-:2:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0015:-:3:n:--   s_lshr_b32 s22, s53, 4
// CHECK-NEXT: 0018:-:6:n:--   v_pk_add_f32 v[106:107], v[106:107], v[236:237]

# 2nd wmma at 11
v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
s_lshr_b32 s22, s53, 4
s_lshr_b32 s22, s53, 4
s_lshr_b32 s22, s53, 4
v_pk_add_f32 v[106:107], v[106:107], v[236:237]

// Note:
// 1. The 3rd salu takes the I slot so v_pk_add is pushed to coExec slot 6
