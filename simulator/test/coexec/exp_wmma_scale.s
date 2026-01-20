// RUN: simulator.py %s | FileCheck %s


// CHECK: 0000:-:-:n:--   v_exp_f32_e32 v242, v132
// CHECK: 0000:y:-:n:--   s_set_vgpr_msb 0x8c0                    ;  msbs: dst=3 src0=0 src1=0 src2=0
// CHECK: 0001:-:-:y:--   ld_scale
// CHECK: 0002:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[34:41] /*v[802:809]*/, v[178:193], v[34:49], 0, s87, s86

v_exp_f32_e32 v242, v132
s_set_vgpr_msb 0x8c0                    ;  msbs: dst=3 src0=0 src1=0 src2=0
v_wmma_scale_f32_16x16x128_f8f6f4 v[34:41] /*v[802:809]*/, v[178:193], v[34:49], 0, s87, s86

// Notes:
// 1. v_set_vgpr_msb is co-issued with exp at cycle 0
// 2. ld_scale is co-executed with exp and is issued at cycle 1
