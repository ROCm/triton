// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
// CHECK: 0001:-:1:n:--   s_lshl_b32 s22, s14, 16
// CHECK: 0002:-:2:n:--   v_exp_f32_e32 v242, v132
// CHECK: 0003:-:3:y:--   s_lshl_b32 s22, s14, 16
// CHECK: 0004:-:4:n:--   s_lshl_b32 s22, s14, 16
// CHECK: 0005:-:5:n:--   s_lshl_b32 s22, s14, 16
// CHECK: 0006:-:6:n:--   v_exp_f32_e32 v242, v132
// CHECK: 0007:-:7:y:--   s_lshl_b32 s22, s14, 16

v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
s_lshl_b32 s22, s14, 16
v_exp_f32_e32 v242, v132
s_lshl_b32 s22, s14, 16
s_lshl_b32 s22, s14, 16
s_lshl_b32 s22, s14, 16
v_exp_f32_e32 v242, v132
s_lshl_b32 s22, s14, 16

// Notes:
// 1. when exp is issued at coExec slot 3, its latency can be hidden
//    with whatever instruction is issued at coExec slot 4.
// 2. It's ok to tri-execute wmma+exp+salu, so we can issue s_lshl at cycle 3 and 7.
