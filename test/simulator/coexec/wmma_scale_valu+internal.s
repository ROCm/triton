// RUN: simulator.py %s | FileCheck %s

// CHECK: 0000:-:-:n:--   ld_scale
// CHECK: 0001:-:0:n:--   v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
// CHECK: 0002:-:1:n:--   s_wait_dscnt 0x0
// CHECK: 0003:n:2:n:--   s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
// CHECK: 0004:-:3:n:--   v_pk_add_f32 v[106:107], v[226:227], v[228:229]
// CHECK: 0005:-:4:n:--   s_barrier_signal -1
// CHECK: 0006:-:5:n:--   s_barrier_wait -1
// CHECK: 0007:n:6:n:--   s_set_vgpr_msb 8                        ;  msbs: dst=0 src0=0 src1=2 src2=0
// CHECK: 0008:-:7:n:--   v_pk_add_f32 v[102:103], v[102:103], v[38:39] /*v[550:551]*/

v_wmma_scale_f32_16x16x128_f8f6f4 v[2:9], v[178:193], v[18:33], 0, s87, s86
s_wait_dscnt 0x0
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_pk_add_f32 v[106:107], v[226:227], v[228:229]
s_barrier_signal -1
s_barrier_wait -1
s_set_vgpr_msb 8                        ;  msbs: dst=0 src0=0 src1=2 src2=0
v_pk_add_f32 v[102:103], v[102:103], v[38:39] /*v[550:551]*/
