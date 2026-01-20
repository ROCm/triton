// RUN: simulator.py %s | FileCheck %s


// valu can only appear at slot 2, 3, 6, and 7

// CHECK: 0000:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
// CHECK: 0002:-:2:n:--   v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
// CHECK: 0003:-:3:n:--   v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
// CHECK: 0006:-:6:n:--   v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
// CHECK: 0007:-:7:n:--   v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/

v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
v_fma_f32 v47, s34, v151 /*v919*/, -v136 /*v904*/
