// RUN: simulator.py %s | FileCheck %s


// valu can only appear at slot 3, 6, and 7

// CHECK: 0000:-:-:n:--   ld_scale
// CHECK: 0001:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[218:225], v[66:81], v[50:65], 0, s87, s86
// CHECK: 0004:-:3:n:--   v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
// CHECK: 0007:-:6:n:--   v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
// CHECK: 0008:-:7:n:--   v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/

v_wmma_scale_f32_16x16x128_f8f6f4 v[218:225], v[66:81], v[50:65], 0, s87, s86
v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
