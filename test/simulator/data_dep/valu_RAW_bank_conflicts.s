// RUN: simulator.py %s | FileCheck %s

// CHECK: 0000:-:-:n:--   v_add_f32_e32 v0, v1, v9
// CHECK: 0006:-:-:n: 4   v_add_f32_e32 v4, v0, v3
// CHECK: 0007:-:-:n:--   v_fma_f32 v0, v1, v9, v17
// CHECK: 0014:-:-:n: 4   v_fma_f32 v2, v0, v3, v4

v_add_f32_e32 v0, v1, v9
v_add_f32_e32 v4, v0, v3
v_fma_f32 v0, v1, v9, v17
v_fma_f32 v2, v0, v3, v4

// Notes
// 1. There is 1 way reg bank conflicts for v_add, since v1 and v9
//    both go to bank 1.
//    Therefore, the 1st v_add instruction takes 5 + 1 = 6 cycles to finish.
// 2. There are 2 way reg bank conflicts for v_fma, since v1, v9, and v17
//    all go to bank 1.
//    Therefore, the 1st v_fma takes 5 + 2 = 7 cycles to finish.
// 3. Even the 2nd v_add and v_fma are delayed by data dependency
//    including reg bank conflict, the delay reason for the instruction
//    only show 4, which is "data dep".
