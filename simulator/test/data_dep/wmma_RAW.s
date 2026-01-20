// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
// CHECK: 0008:-:0:n:--   v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]

v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], 0
v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]
