// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:-:n:--   ld_scale
// CHECK: 0001:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[218:225] /*v[730:737]*/, v[66:81], v[50:65], 0, s87, s86
// CHECK: 0004:-:3:n:--   ld_scale
// CHECK: 0009:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[218:225] /*v[730:737]*/, v[66:81], v[50:65], v[218:225] /*v[730:737]*/, s87, s86

v_wmma_scale_f32_16x16x128_f8f6f4 v[218:225] /*v[730:737]*/, v[66:81], v[50:65], 0, s87, s86
v_wmma_scale_f32_16x16x128_f8f6f4 v[218:225] /*v[730:737]*/, v[66:81], v[50:65], v[218:225] /*v[730:737]*/, s87, s86

// Notes
// 1. wmma instruction needs to read 40 registers as inputs.
//    Those vgprs has contiguous ids so it takes 5 cycles to
//    read all inputs.
//    These cycles will not delay any following instructions
//    that reads wmma's results. This is because in our model,
//    wmma takes 8 cycles to issue.
