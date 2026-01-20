// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:-:n:--   v_exp_f32_e32 v242, v132
// CHECK: 0001:-:-:y:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]

v_exp_f32_e32 v242, v132
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
