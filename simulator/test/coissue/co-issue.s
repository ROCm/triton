// RUN: simulator.py %s | FileCheck %s

// CHECK: 0000:-:-:n:--   ds_load_b128 v[130:133], v190 offset:8192
// CHECK: 0001:n:-:n:--   s_set_vgpr_msb 0xc00c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
// CHECK: 0002:-:-:n:--   v_add_nc_u32_e32 v190, s6, v152 /*v920*/
// CHECK: 0002:y:-:n:--   s_set_vgpr_msb 0xc080                   ;  msbs: dst=2 src0=0 src1=0 src2=0
// CHECK: 0003:-:-:n:--   ld_scale
// CHECK: 0004:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[194:201] /*v[706:713]*/, v[114:129], v[50:65], 0, s87, s86
// CHECK: 0004:y:0:n:--   s_set_vgpr_msb 0x80c0                   ;  msbs: dst=3 src0=0 src1=0 src2=0
// CHECK: 0007:-:3:n:--   v_exp_f32_e32 v104 /*v872*/, v134
// CHECK: 0008:-:4:y:--   s_wait_dscnt 0x1
// CHECK: 0009:n:5:n:--   s_set_vgpr_msb 0xc00a                   ;  msbs: dst=0 src0=2 src1=2 src2=0
// CHECK: 0010:n:6:n:--   s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
// CHECK: 0011:-:7:n:--   v_pk_add_f32 v[68:69], v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/

ds_load_b128 v[130:133], v190 offset:8192
s_set_vgpr_msb 0xc00c                   ;  msbs: dst=0 src0=0 src1=3 src2=0
v_add_nc_u32_e32 v190, s6, v152 /*v920*/
s_set_vgpr_msb 0xc080                   ;  msbs: dst=2 src0=0 src1=0 src2=0
v_wmma_scale_f32_16x16x128_f8f6f4 v[194:201] /*v[706:713]*/, v[114:129], v[50:65], 0, s87, s86
s_set_vgpr_msb 0x80c0                   ;  msbs: dst=3 src0=0 src1=0 src2=0
v_exp_f32_e32 v104 /*v872*/, v134
s_wait_dscnt 0x1
s_set_vgpr_msb 0xc00a                   ;  msbs: dst=0 src0=2 src1=2 src2=0
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
v_pk_add_f32 v[68:69], v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/

// Note
// 1. control cannot co-issue with ds_load
// 2. control can co-issue with valu
// 3. control can co-issue with wmma
// 4. control cannot co-issue with control or internal instructions, such as
//    s_wait*, s_barrier*. etc
