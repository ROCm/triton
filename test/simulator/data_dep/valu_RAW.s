// RUN: simulator.py %s | FileCheck %s

// CHECK: 0000:-:-:n:--   v_add_f32_e32 v0, v1, v2
// CHECK: 0005:-:-:n: 4   v_add_f32_e32 v4, v0, v3

v_add_f32_e32 v0, v1, v2
v_add_f32_e32 v4, v0, v3
