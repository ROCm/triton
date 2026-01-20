// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:-:n:--   v_exp_f32_e32 v242, v132
// CHECK: 0000:y:-:n:--   s_set_vgpr_msb 0x8c0                    ;  msbs: dst=3 src0=0 src1=0 src2=0
// CHECK: 0001:-:0:y:--   v_wmma_f32_16x16x32_f16 v[34:41] /*v[802:809]*/, v[8:15], v[16:23], 0

v_exp_f32_e32 v242, v132
s_set_vgpr_msb 0x8c0                    ;  msbs: dst=3 src0=0 src1=0 src2=0
v_wmma_f32_16x16x32_f16 v[34:41] /*v[802:809]*/, v[8:15], v[16:23], 0

// Notes:
// 1. v_set_vgpr_msb is co-issued with exp at cycle 0
