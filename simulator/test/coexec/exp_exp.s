// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:-:n:--   v_exp_f32_e32 v242, v132
// CHECK: 0002:-:-:n:16   v_exp_f32_e32 v242, v132
// CHECK: 0003:-:-:y:--   v_nop
// CHECK: 0004:-:-:n:--   v_exp_f32_e32 v242, v132

v_exp_f32_e32 v242, v132
v_exp_f32_e32 v242, v132
v_nop
v_exp_f32_e32 v242, v132

// Notes:
// 1. The 2nd exp is delayed due to the latency of trans instruction
// 2. The 3rd exp is not delayed since there is an independent valu
//    taking the coExec slot of the 2nd exp.
