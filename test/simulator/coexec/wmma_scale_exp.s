// RUN: simulator.py %s | FileCheck %s


// CHECK:      0000:-:-:n:--   ld_scale
// CHECK-NEXT: 0001:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK-NEXT: 0002:-:1:n:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0003:-:2:n:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0004:-:3:n:--   v_exp_f32_e32 v242, v132
// CHECK-NEXT: 0005:-:4:y:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0007:-:6:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK-NEXT: 0008:-:7:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]

v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
s_lshl_b32 s22, s14, 16
s_lshl_b32 s22, s14, 16
v_exp_f32_e32 v242, v132
s_lshl_b32 s22, s14, 16
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
v_pk_add_f32 v[106:107], v[226:227], v[228:229]

// Notes:
// 1. when exp is issued at coExec slot 3, its latency can be hidden
//    with whatever instruction is issued at coExec slot 4.



// CHECK-NEXT: 0011:-:-:n: 2   ld_scale
// CHECK-NEXT: 0012:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK-NEXT: 0013:-:1:n:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0014:-:2:n:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0015:-:3:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK-NEXT: 0016:-:4:n:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0018:-:6:n:--   v_exp_f32_e32 v242, v132
// CHECK-NEXT: 0022:-:-:n: 3   v_pk_add_f32 v[106:107], v[226:227], v[228:229]

v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
s_lshl_b32 s22, s14, 16
s_lshl_b32 s22, s14, 16
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
s_lshl_b32 s22, s14, 16
v_exp_f32_e32 v242, v132
v_pk_add_f32 v[106:107], v[226:227], v[228:229]

// Notes:
// 2. The last v_pk_add is delay by 2 reasons (reason sum = 3)
//    - It cannot be issued at cycle 19 since wmma+exp+valu cannot tri-execute ==> reason value 1
//    - It cannot be issued at cycle 20 and 21 due to the V rule ==> reason value 2


// CHECK-NEXT: 0023:-:-:n:--   ld_scale
// CHECK-NEXT: 0024:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK-NEXT: 0025:-:1:n:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0026:-:2:n:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0027:-:3:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK-NEXT: 0028:-:4:n:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0030:-:6:n:--   v_exp_f32_e32 v242, v132
// CHECK-NEXT: 0031:-:7:y:--   s_lshl_b32 s22, s14, 16
// CHECK-NEXT: 0034:-:-:n: 2   v_pk_add_f32 v[106:107], v[226:227], v[228:229]

v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
s_lshl_b32 s22, s14, 16
s_lshl_b32 s22, s14, 16
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
s_lshl_b32 s22, s14, 16
v_exp_f32_e32 v242, v132
s_lshl_b32 s22, s14, 16
v_pk_add_f32 v[106:107], v[226:227], v[228:229]

// Notes:
// 3. It's ok to tri-execute wmma+exp+salu, so we can issue s_lshl at cycle 31.
//    Now the last v_pk_add is only delayed by the V law.
