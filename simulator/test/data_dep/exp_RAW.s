// RUN: simulator.py %s | FileCheck %s

// CHECK: 0000:-:-:n:--   v_exp_f32_e32 v99 /*v867*/, v137
// CHECK: 0008:-:-:n: 4   v_exp_f32_e32 v100 /*v868*/, v99 /*v867*/

v_exp_f32_e32 v99 /*v867*/, v137
v_exp_f32_e32 v100 /*v868*/, v99 /*v867*/
