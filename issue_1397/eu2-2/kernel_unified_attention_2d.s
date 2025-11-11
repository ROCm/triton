	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	kernel_unified_attention_2d     ; -- Begin function kernel_unified_attention_2d
	.p2align	8
	.type	kernel_unified_attention_2d,@function
kernel_unified_attention_2d:            ; @kernel_unified_attention_2d
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.10:
	.file	1 "/var/lib/jenkins/aiter/aiter/ops/triton/_triton_kernels" "unified_attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.11:
.LBB0_0:
	s_mov_b64 s[18:19], s[6:7]
	s_mov_b64 s[80:81], s[2:3]
	s_load_dwordx2 s[2:3], s[0:1], 0xa0
	s_load_dwordx4 s[24:27], s[0:1], 0x90
	s_load_dwordx8 s[72:79], s[0:1], 0x70
	s_load_dword s6, s[0:1], 0xa8
	s_load_dwordx8 s[64:71], s[0:1], 0x40
	s_load_dwordx2 s[82:83], s[0:1], 0x60
	s_mov_b32 s28, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s6, 1
	s_cbranch_scc1 .LBB0_2
.LBB0_1:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	s_add_i32 s0, s6, s28
	s_lshr_b32 s1, s0, 31
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[20:21], s[0:1], 2
	s_add_u32 s20, s2, s20
	s_addc_u32 s21, s3, s21
	s_load_dword s1, s[20:21], 0x0
	s_add_i32 s7, s0, 1
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s20, s1, 31
	s_lshr_b32 s20, s20, 28
	s_add_i32 s1, s1, s20
	s_ashr_i32 s1, s1, 4
	s_add_i32 s1, s1, s0
	s_cmp_gt_i32 s1, s17
	s_cselect_b32 s6, s0, s6
	s_cselect_b32 s28, s28, s7
	s_cmp_lt_i32 s28, s6
	s_cbranch_scc1 .LBB0_1
.LBB0_2:                                ; %._crit_edge
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[30:31], s[28:29], 2
	s_add_u32 s0, s2, s30
	s_addc_u32 s1, s3, s31
	s_add_u32 s0, s0, -4
	s_addc_u32 s1, s1, -1
	s_load_dwordx2 s[84:85], s[0:1], 0x0
	s_sub_i32 s0, s17, s28
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s1, s84, 31
	s_lshr_b32 s1, s1, 28
	s_add_i32 s1, s84, s1
	s_lshr_b32 s1, s1, 4
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s29, s0, 4
	s_sub_i32 s17, s85, s84
	s_add_i32 s29, s29, 16
	s_cmp_lt_i32 s29, s17
	s_cbranch_scc0 .LBB0_9
; %bb.3:
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshrrev_b32_e32 v34, 3, v0
	v_or_b32_e32 v2, s29, v1
	s_lshl_b32 s33, s16, 3
	v_or_b32_e32 v3, 4, v2
	v_or_b32_e32 v4, 8, v2
	v_or_b32_e32 v5, 12, v2
	v_and_or_b32 v10, v34, 7, s33
	v_add_u32_e32 v6, s84, v2
	v_add_u32_e32 v7, s84, v3
	v_add_u32_e32 v8, s84, v4
	v_add_u32_e32 v9, s84, v5
	v_lshlrev_b32_e32 v1, 3, v0
	v_cmp_gt_i32_e32 vcc, s17, v2
	v_mul_lo_u32 v2, s68, v10
	v_and_b32_e32 v36, 56, v1
	v_cmp_gt_i32_e64 s[0:1], s17, v3
	v_cmp_gt_i32_e64 s[2:3], s17, v4
	v_cmp_gt_i32_e64 s[20:21], s17, v5
	v_cmp_gt_i32_e64 s[22:23], 64, v10
	v_mad_u64_u32 v[4:5], s[6:7], s66, v6, v[2:3]
	v_mad_u64_u32 v[6:7], s[6:7], s66, v7, v[2:3]
	v_mad_u64_u32 v[12:13], s[6:7], s66, v8, v[2:3]
	v_mad_u64_u32 v[2:3], s[6:7], s66, v9, v[2:3]
	v_add_lshl_u32 v3, v36, v4, 1
	v_bfrev_b32_e32 v13, 1
	s_and_b64 vcc, s[22:23], vcc
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e32 v3, v13, v3, vcc
	v_add_lshl_u32 v4, v36, v6, 1
	s_and_b64 vcc, s[22:23], s[0:1]
	v_cndmask_b32_e32 v14, v13, v4, vcc
	buffer_load_dwordx4 v[4:7], v3, s[4:7], 0 offen sc0 nt
	buffer_load_dwordx4 v[8:11], v14, s[4:7], 0 offen sc0 nt
	v_add_lshl_u32 v3, v36, v12, 1
	s_and_b64 vcc, s[22:23], s[2:3]
	v_cndmask_b32_e32 v3, v13, v3, vcc
	v_add_lshl_u32 v2, v36, v2, 1
	s_and_b64 vcc, s[22:23], s[20:21]
	v_cndmask_b32_e32 v2, v13, v2, vcc
	buffer_load_dwordx4 v[12:15], v3, s[4:7], 0 offen sc0 nt
	buffer_load_dwordx4 v[16:19], v2, s[4:7], 0 offen sc0 nt
	v_and_b32_e32 v3, 0xc0, v0
	v_and_b32_e32 v102, 32, v0
	v_lshlrev_b32_e32 v37, 4, v0
	s_movk_i32 s0, 0x70
	v_and_b32_e32 v2, 31, v0
	v_bitop3_b32 v20, v37, v0, s0 bitop3:0x78
	v_lshlrev_b32_e32 v21, 6, v3
	v_and_b32_e32 v39, 0x70, v1
	v_lshrrev_b32_e32 v40, 1, v102
	v_lshlrev_b32_e32 v38, 7, v2
	v_add_u32_e32 v103, 0, v20
	v_bitop3_b32 v20, v39, v21, v40 bitop3:0xde
	s_movk_i32 s1, 0x60
	v_or_b32_e32 v21, v20, v38
	v_bitop3_b32 v20, v20, s1, v38 bitop3:0x36
	v_add_u32_e32 v22, 0, v21
	v_xad_u32 v23, v21, 32, 0
	v_xad_u32 v21, v21, 64, 0
	v_add_u32_e32 v20, 0, v20
	v_and_or_b32 v90, v0, 7, s33
	v_ashrrev_i32_e32 v91, 31, v90
	v_cmp_gt_i32_e32 vcc, 64, v90
	v_mov_b32_e32 v41, 0xff800000
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[4:7]
	s_waitcnt vmcnt(2)
	ds_write_b128 v103, v[8:11] offset:4096
	s_waitcnt vmcnt(1)
	ds_write_b128 v103, v[12:15] offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b128 v103, v[16:19] offset:12288
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[66:69], v22
	ds_read_b128 v[70:73], v23
	ds_read_b128 v[74:77], v21
	ds_read_b128 v[78:81], v20
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:
	v_lshl_add_u64 v[4:5], v[90:91], 2, s[10:11]
	global_load_dword v41, v[4:5], off
.LBB0_5:
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v3, 4, v3
	v_lshrrev_b32_e32 v2, 3, v2
	v_or3_b32 v91, v3, v2, s29
	v_cmp_gt_i32_e64 s[0:1], s17, v91
	s_and_b64 s[0:1], vcc, s[0:1]
	s_add_u32 s2, s14, s30
	s_addc_u32 s3, s15, s31
	v_mov_b32_e32 v19, 0
	global_load_dword v16, v19, s[2:3] offset:-4
	v_mov_b32_e32 v35, 1.0
	v_mov_b32_e32 v18, v19
	v_mov_b32_e32 v21, v19
	v_mov_b32_e32 v20, v19
	v_mov_b32_e32 v23, v19
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v25, v19
	v_mov_b32_e32 v24, v19
	v_mov_b32_e32 v27, v19
	v_mov_b32_e32 v26, v19
	v_mov_b32_e32 v29, v19
	v_mov_b32_e32 v28, v19
	v_mov_b32_e32 v31, v19
	v_mov_b32_e32 v30, v19
	v_mov_b32_e32 v33, v19
	v_mov_b32_e32 v32, v19
	v_mov_b32_e32 v3, v19
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v5, v19
	v_mov_b32_e32 v4, v19
	v_mov_b32_e32 v7, v19
	v_mov_b32_e32 v6, v19
	v_mov_b32_e32 v9, v19
	v_mov_b32_e32 v8, v19
	v_mov_b32_e32 v11, v19
	v_mov_b32_e32 v10, v19
	v_mov_b32_e32 v13, v19
	v_mov_b32_e32 v12, v19
	v_mov_b32_e32 v15, v19
	v_mov_b32_e32 v14, v19
	v_mov_b32_e32 v17, v19
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v16
	s_sub_i32 s2, s3, s17
	s_add_i32 s4, s29, s2
	s_add_i32 s4, s4, 16
	s_min_i32 s3, s4, s3
	s_add_i32 s3, s3, 63
	s_cmp_lt_i32 s3, 64
	v_mov_b32_e32 v16, v19
	s_cbranch_scc1 .LBB0_8
; %bb.6:                                ; %.lr.ph18
	s_ashr_i32 s4, s3, 31
	s_lshr_b32 s4, s4, 26
	s_add_i32 s3, s3, s4
	s_ashr_i32 s33, s3, 6
	s_ashr_i32 s3, s16, 31
	s_mul_i32 s5, s26, s3
	s_mul_hi_u32 s6, s76, s16
	s_mul_i32 s3, s76, s3
	s_mul_hi_u32 s4, s26, s16
	s_add_i32 s3, s6, s3
	s_mul_i32 s6, s77, s16
	s_add_i32 s4, s4, s5
	s_mul_i32 s5, s27, s16
	s_add_i32 s7, s3, s6
	s_mul_i32 s6, s76, s16
	s_add_i32 s71, s28, -1
	s_add_i32 s5, s4, s5
	v_add3_u32 v104, v91, s2, 1
	s_lshl_b64 s[2:3], s[6:7], 1
	s_add_u32 s2, s18, s2
	s_mul_i32 s4, s26, s16
	s_addc_u32 s3, s19, s3
	v_lshlrev_b32_e32 v2, 1, v36
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[92:93], s[2:3], 0, v[2:3]
	s_lshl_b64 s[2:3], s[4:5], 1
	s_add_u32 s2, s8, s2
	v_and_b32_e32 v4, 0x70, v0
	s_addc_u32 s3, s9, s3
	v_lshl_add_u64 v[94:95], s[2:3], 0, v[2:3]
	v_lshrrev_b32_e32 v2, 1, v4
	v_xor_b32_e32 v107, v37, v2
	v_lshlrev_b32_e32 v2, 5, v0
	v_bfe_i32 v4, v0, 3, 1
	v_lshlrev_b32_e32 v0, 1, v0
	v_and_b32_e32 v0, 32, v0
	s_movk_i32 s2, 0x80
	v_and_b32_e32 v1, 24, v1
	v_mov_b32_e32 v5, 0x210
	v_cmp_eq_u32_e32 vcc, 0, v102
	v_and_or_b32 v0, v2, s2, v0
	s_movk_i32 s2, 0x108
	v_cndmask_b32_e64 v5, v5, 0, vcc
	v_bitop3_b32 v1, v4, v1, s2 bitop3:0x6c
	v_xor_b32_e32 v1, v1, v5
	v_or_b32_e32 v11, v0, v1
	v_bitop3_b32 v12, v0, 32, v1 bitop3:0x36
	v_or_b32_e32 v13, 32, v34
	v_mad_u64_u32 v[0:1], s[2:3], s24, v34, 0
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[96:97], s[2:3], s24, v13, 0
	v_mad_u64_u32 v[4:5], s[2:3], s25, v34, v[2:3]
	v_mov_b32_e32 v2, v97
	v_mad_u64_u32 v[98:99], s[2:3], s74, v34, 0
	v_mov_b32_e32 v1, v4
	v_mad_u64_u32 v[4:5], s[2:3], s25, v13, v[2:3]
	v_mov_b32_e32 v2, v99
	v_mad_u64_u32 v[100:101], s[2:3], s74, v13, 0
	v_bitop3_b32 v6, v38, v40, v39 bitop3:0x36
	v_mov_b32_e32 v97, v4
	v_mad_u64_u32 v[4:5], s[2:3], s75, v34, v[2:3]
	v_mov_b32_e32 v2, v101
	v_xor_b32_e32 v7, 32, v6
	v_xor_b32_e32 v8, 64, v6
	v_xor_b32_e32 v9, 0x60, v6
	v_xor_b32_e32 v10, 8, v107
	v_mov_b32_e32 v99, v4
	v_mad_u64_u32 v[4:5], s[2:3], s75, v13, v[2:3]
	s_mul_i32 s71, s71, s64
	s_mov_b32 s76, 0
	v_lshrrev_b32_e32 v105, 3, v102
	v_mul_f32_e32 v106, 0x3fb8aa3b, v41
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_and_b32 s13, s13, 0xffff
	v_mov_b32_e32 v101, v4
	v_mov_b32_e32 v35, 1.0
	v_add_u32_e32 v108, 0, v7
	v_add_u32_e32 v109, 0, v8
	v_add_u32_e32 v110, 0, v9
	s_mov_b32 s74, 0xff800000
	v_add_u32_e32 v111, 0, v10
	v_add_u32_e32 v112, 0, v11
	v_add_u32_e32 v113, 0, v12
	v_add_u32_e32 v114, 0, v6
	v_mov_b32_e32 v115, 0xff800000
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v11, v3
	v_mov_b32_e32 v12, v3
	v_mov_b32_e32 v13, v3
	v_mov_b32_e32 v14, v3
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v16, v3
	v_mov_b32_e32 v17, v3
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v20, v3
	v_mov_b32_e32 v21, v3
	v_mov_b32_e32 v22, v3
	v_mov_b32_e32 v23, v3
	v_mov_b32_e32 v24, v3
	v_mov_b32_e32 v25, v3
	v_mov_b32_e32 v26, v3
	v_mov_b32_e32 v27, v3
	v_mov_b32_e32 v28, v3
	v_mov_b32_e32 v29, v3
	v_mov_b32_e32 v30, v3
	v_mov_b32_e32 v31, v3
	v_mov_b32_e32 v32, v3
	v_mov_b32_e32 v33, v3
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s2, s76, 0x3ffffff
	s_add_i32 s42, s2, s71
	v_add_u32_e32 v41, 1, v105
	s_lshl_b32 s42, s42, 2
	v_cmp_lt_i32_e64 s[2:3], v41, v104
	v_mov_b32_e32 v41, s42
	buffer_load_dword v41, v41, s[12:15], 0 offen
	v_add_u32_e32 v43, 3, v105
	v_add_u32_e32 v42, 2, v105
	v_add_u32_e32 v44, 8, v105
	v_add_u32_e32 v45, 9, v105
	v_add_u32_e32 v46, 10, v105
	v_add_u32_e32 v47, 11, v105
	v_add_u32_e32 v48, 16, v105
	v_cmp_lt_i32_e64 s[6:7], v43, v104
	v_cmp_lt_i32_e64 s[4:5], v42, v104
	v_cmp_lt_i32_e64 s[8:9], v44, v104
	v_cmp_lt_i32_e64 s[10:11], v45, v104
	v_cmp_lt_i32_e64 s[16:17], v46, v104
	v_cmp_lt_i32_e64 s[18:19], v47, v104
	v_cmp_lt_i32_e64 s[20:21], v48, v104
	v_add_u32_e32 v49, 17, v105
	v_add_u32_e32 v50, 18, v105
	v_add_u32_e32 v51, 19, v105
	v_add_u32_e32 v52, 24, v105
	v_add_u32_e32 v53, 25, v105
	v_cmp_lt_i32_e64 s[22:23], v49, v104
	v_cmp_lt_i32_e64 s[24:25], v50, v104
	v_cmp_lt_i32_e64 s[26:27], v51, v104
	v_cmp_lt_i32_e64 s[28:29], v52, v104
	v_cmp_lt_i32_e64 s[30:31], v53, v104
	v_add_u32_e32 v54, 26, v105
	v_add_u32_e32 v55, 27, v105
	v_add_u32_e32 v56, 32, v105
	v_add_u32_e32 v57, 33, v105
	v_cmp_lt_i32_e64 s[34:35], v54, v104
	v_cmp_lt_i32_e64 s[36:37], v55, v104
	v_cmp_lt_i32_e64 s[38:39], v56, v104
	v_cmp_lt_i32_e64 s[40:41], v57, v104
	v_mov_b32_e32 v116, v35
	v_add_u32_e32 v34, 34, v105
	v_add_u32_e32 v35, 35, v105
	v_add_u32_e32 v36, 40, v105
	v_add_u32_e32 v37, 41, v105
	v_add_u32_e32 v38, 42, v105
	v_add_u32_e32 v39, 43, v105
	v_add_u32_e32 v40, 48, v105
	v_cmp_lt_i32_e64 s[42:43], v34, v104
	v_cmp_lt_i32_e64 s[44:45], v35, v104
	v_cmp_lt_i32_e64 s[46:47], v36, v104
	v_cmp_lt_i32_e64 s[48:49], v37, v104
	v_cmp_lt_i32_e64 s[50:51], v38, v104
	v_cmp_lt_i32_e64 s[52:53], v39, v104
	v_cmp_lt_i32_e64 s[54:55], v40, v104
	v_add_u32_e32 v82, 49, v105
	v_add_u32_e32 v83, 50, v105
	v_add_u32_e32 v84, 51, v105
	v_add_u32_e32 v85, 56, v105
	v_add_u32_e32 v86, 57, v105
	v_add_u32_e32 v87, 58, v105
	v_add_u32_e32 v88, 59, v105
	v_cmp_lt_i32_e64 s[56:57], v82, v104
	v_cmp_lt_i32_e64 s[58:59], v83, v104
	v_cmp_lt_i32_e64 s[60:61], v84, v104
	v_cmp_lt_i32_e64 s[62:63], v85, v104
	v_cmp_lt_i32_e64 s[64:65], v86, v104
	v_cmp_lt_i32_e64 s[66:67], v87, v104
	v_cmp_lt_i32_e64 s[68:69], v88, v104
	v_cmp_lt_i32_e32 vcc, v105, v104
	s_and_b64 s[2:3], s[0:1], s[2:3]
	s_and_b64 vcc, s[0:1], vcc
	v_add_u32_e32 v117, 0, v107
	s_and_b64 s[4:5], s[0:1], s[4:5]
	s_and_b64 s[6:7], s[0:1], s[6:7]
	s_and_b64 s[8:9], s[0:1], s[8:9]
	s_and_b64 s[10:11], s[0:1], s[10:11]
	s_and_b64 s[16:17], s[0:1], s[16:17]
	s_and_b64 s[18:19], s[0:1], s[18:19]
	s_and_b64 s[20:21], s[0:1], s[20:21]
	s_and_b64 s[22:23], s[0:1], s[22:23]
	s_and_b64 s[24:25], s[0:1], s[24:25]
	s_and_b64 s[26:27], s[0:1], s[26:27]
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_and_b64 s[30:31], s[0:1], s[30:31]
	s_and_b64 s[34:35], s[0:1], s[34:35]
	s_and_b64 s[36:37], s[0:1], s[36:37]
	s_waitcnt vmcnt(0)
	v_ashrrev_i32_e32 v43, 31, v41
	v_mul_hi_u32 v45, s78, v41
	v_mul_lo_u32 v46, s79, v41
	v_mul_lo_u32 v42, s78, v41
	v_mul_hi_u32 v47, s72, v41
	v_mul_lo_u32 v48, s73, v41
	v_mul_lo_u32 v44, s72, v41
	v_mul_lo_u32 v41, s78, v43
	v_mul_lo_u32 v43, s72, v43
	v_add_u32_e32 v41, v45, v41
	v_add_u32_e32 v45, v47, v43
	v_add_u32_e32 v45, v45, v48
	v_add_u32_e32 v43, v41, v46
	v_lshl_add_u64 v[44:45], v[44:45], 1, v[92:93]
	v_lshl_add_u64 v[42:43], v[42:43], 1, v[94:95]
	v_lshl_add_u64 v[46:47], v[98:99], 1, v[44:45]
	v_lshl_add_u64 v[48:49], v[100:101], 1, v[44:45]
	v_lshl_add_u64 v[50:51], v[0:1], 1, v[42:43]
	v_lshl_add_u64 v[52:53], v[96:97], 1, v[42:43]
	global_load_dwordx4 v[42:45], v[46:47], off
	s_nop 0
	global_load_dwordx4 v[46:49], v[48:49], off
	s_nop 0
	global_load_dwordx4 v[118:121], v[50:51], off
	global_load_dwordx4 v[122:125], v[52:53], off
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b64 s[38:39], s[0:1], s[38:39]
	s_and_b64 s[40:41], s[0:1], s[40:41]
	s_and_b64 s[42:43], s[0:1], s[42:43]
	s_and_b64 s[44:45], s[0:1], s[44:45]
	s_and_b64 s[46:47], s[0:1], s[46:47]
	s_and_b64 s[48:49], s[0:1], s[48:49]
	s_and_b64 s[50:51], s[0:1], s[50:51]
	s_and_b64 s[52:53], s[0:1], s[52:53]
	s_and_b64 s[54:55], s[0:1], s[54:55]
	s_and_b64 s[56:57], s[0:1], s[56:57]
	s_and_b64 s[58:59], s[0:1], s[58:59]
	s_and_b64 s[60:61], s[0:1], s[60:61]
	s_and_b64 s[62:63], s[0:1], s[62:63]
	s_and_b64 s[64:65], s[0:1], s[64:65]
	s_and_b64 s[66:67], s[0:1], s[66:67]
	s_and_b64 s[68:69], s[0:1], s[68:69]
	s_add_i32 s76, s76, 1
	v_add_u32_e32 v105, 64, v105
	s_cmp_lg_u32 s33, s76
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[42:45]
	s_waitcnt vmcnt(2)
	ds_write_b128 v103, v[46:49] offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[42:45], v114
	ds_read_b128 v[46:49], v114 offset:4096
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[50:65], v[42:45], v[66:69], 0
	ds_read_b128 v[126:129], v108
	ds_read_b128 v[130:133], v108 offset:4096
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[46:49], v[66:69], 0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[50:65], v[126:129], v[70:73], v[50:65]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[130:133], v[70:73], v[34:49]
	ds_read_b128 v[82:85], v109
	ds_read_b128 v[86:89], v109 offset:4096
	ds_read_b128 v[126:129], v110
	ds_read_b128 v[130:133], v110 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write2st64_b64 v117, v[118:119], v[122:123] offset1:8
	ds_write2st64_b64 v111, v[120:121], v[124:125] offset1:8
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[118:119], v112
	ds_read_b64_tr_b16 v[122:123], v112 offset:2048
	v_mfma_f32_32x32x16_bf16 v[50:65], v[82:85], v[74:77], v[50:65]
	v_mfma_f32_32x32x16_bf16 v[34:49], v[86:89], v[74:77], v[34:49]
	ds_read_b64_tr_b16 v[134:135], v112 offset:2112
	ds_read_b64_tr_b16 v[138:139], v112 offset:64
	ds_read_b64_tr_b16 v[142:143], v112 offset:4096
	ds_read_b64_tr_b16 v[86:87], v112 offset:6144
	ds_read_b64_tr_b16 v[82:83], v112 offset:6208
	ds_read_b64_tr_b16 v[146:147], v112 offset:4160
	ds_read_b64_tr_b16 v[120:121], v113 offset:1024
	ds_read_b64_tr_b16 v[124:125], v113 offset:3072
	ds_read_b64_tr_b16 v[136:137], v113 offset:3136
	ds_read_b64_tr_b16 v[140:141], v113 offset:1088
	ds_read_b64_tr_b16 v[144:145], v113 offset:5120
	ds_read_b64_tr_b16 v[88:89], v113 offset:7168
	ds_read_b64_tr_b16 v[84:85], v113 offset:7232
	ds_read_b64_tr_b16 v[148:149], v113 offset:5184
	v_mfma_f32_32x32x16_bf16 v[50:65], v[126:129], v[78:81], v[50:65]
	v_mfma_f32_32x32x16_bf16 v[34:49], v[130:133], v[78:81], v[34:49]
	s_nop 10
	v_mul_f32_e32 v50, 0x3e38aa3b, v50
	v_mul_f32_e32 v51, 0x3e38aa3b, v51
	v_mul_f32_e32 v52, 0x3e38aa3b, v52
	v_mul_f32_e32 v53, 0x3e38aa3b, v53
	v_cndmask_b32_e32 v50, v115, v50, vcc
	v_cndmask_b32_e64 v51, v115, v51, s[2:3]
	v_mul_f32_e32 v54, 0x3e38aa3b, v54
	v_mul_f32_e32 v55, 0x3e38aa3b, v55
	v_cndmask_b32_e64 v52, v115, v52, s[4:5]
	v_cndmask_b32_e64 v53, v115, v53, s[6:7]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v117, v50, v51
	v_mul_f32_e32 v56, 0x3e38aa3b, v56
	v_mul_f32_e32 v57, 0x3e38aa3b, v57
	v_cndmask_b32_e64 v54, v115, v54, s[8:9]
	v_cndmask_b32_e64 v55, v115, v55, s[10:11]
	v_max3_f32 v117, v117, v52, v53
	v_mul_f32_e32 v58, 0x3e38aa3b, v58
	v_mul_f32_e32 v59, 0x3e38aa3b, v59
	v_cndmask_b32_e64 v56, v115, v56, s[16:17]
	v_cndmask_b32_e64 v57, v115, v57, s[18:19]
	v_max3_f32 v117, v117, v54, v55
	v_mul_f32_e32 v60, 0x3e38aa3b, v60
	v_mul_f32_e32 v61, 0x3e38aa3b, v61
	v_cndmask_b32_e64 v58, v115, v58, s[20:21]
	v_cndmask_b32_e64 v59, v115, v59, s[22:23]
	v_max3_f32 v117, v117, v56, v57
	v_mul_f32_e32 v62, 0x3e38aa3b, v62
	v_mul_f32_e32 v63, 0x3e38aa3b, v63
	v_cndmask_b32_e64 v60, v115, v60, s[24:25]
	v_cndmask_b32_e64 v61, v115, v61, s[26:27]
	v_max3_f32 v117, v117, v58, v59
	v_mul_f32_e32 v64, 0x3e38aa3b, v64
	v_mul_f32_e32 v65, 0x3e38aa3b, v65
	v_cndmask_b32_e64 v62, v115, v62, s[28:29]
	v_cndmask_b32_e64 v63, v115, v63, s[30:31]
	v_max3_f32 v117, v117, v60, v61
	v_mul_f32_e32 v34, 0x3e38aa3b, v34
	v_mul_f32_e32 v35, 0x3e38aa3b, v35
	v_cndmask_b32_e64 v64, v115, v64, s[34:35]
	v_cndmask_b32_e64 v65, v115, v65, s[36:37]
	v_max3_f32 v117, v117, v62, v63
	v_mul_f32_e32 v36, 0x3e38aa3b, v36
	v_mul_f32_e32 v37, 0x3e38aa3b, v37
	v_cndmask_b32_e64 v34, v115, v34, s[38:39]
	v_cndmask_b32_e64 v35, v115, v35, s[40:41]
	v_max3_f32 v117, v117, v64, v65
	v_mul_f32_e32 v38, 0x3e38aa3b, v38
	v_mul_f32_e32 v39, 0x3e38aa3b, v39
	v_cndmask_b32_e64 v36, v115, v36, s[42:43]
	v_cndmask_b32_e64 v37, v115, v37, s[44:45]
	v_max3_f32 v117, v117, v34, v35
	v_mul_f32_e32 v40, 0x3e38aa3b, v40
	v_mul_f32_e32 v41, 0x3e38aa3b, v41
	v_cndmask_b32_e64 v38, v115, v38, s[46:47]
	v_cndmask_b32_e64 v39, v115, v39, s[48:49]
	v_max3_f32 v117, v117, v36, v37
	v_mul_f32_e32 v42, 0x3e38aa3b, v42
	v_mul_f32_e32 v43, 0x3e38aa3b, v43
	v_cndmask_b32_e64 v40, v115, v40, s[50:51]
	v_cndmask_b32_e64 v41, v115, v41, s[52:53]
	v_max3_f32 v117, v117, v38, v39
	v_mul_f32_e32 v44, 0x3e38aa3b, v44
	v_mul_f32_e32 v45, 0x3e38aa3b, v45
	v_cndmask_b32_e64 v42, v115, v42, s[54:55]
	v_cndmask_b32_e64 v43, v115, v43, s[56:57]
	v_max3_f32 v117, v117, v40, v41
	v_mul_f32_e32 v46, 0x3e38aa3b, v46
	v_mul_f32_e32 v47, 0x3e38aa3b, v47
	v_cndmask_b32_e64 v44, v115, v44, s[58:59]
	v_cndmask_b32_e64 v45, v115, v45, s[60:61]
	v_max3_f32 v117, v117, v42, v43
	v_mul_f32_e32 v48, 0x3e38aa3b, v48
	v_mul_f32_e32 v49, 0x3e38aa3b, v49
	v_cndmask_b32_e64 v46, v115, v46, s[62:63]
	v_cndmask_b32_e64 v47, v115, v47, s[64:65]
	v_max3_f32 v117, v117, v44, v45
	v_cndmask_b32_e64 v48, v115, v48, s[66:67]
	v_cndmask_b32_e64 v49, v115, v49, s[68:69]
	v_max3_f32 v117, v117, v46, v47
	v_max3_f32 v117, v117, v48, v49
	v_mov_b32_e32 v126, v117
	s_nop 1
	v_permlane32_swap_b32_e32 v117, v126
	v_max3_f32 v117, v106, v117, v126
	v_cmp_lg_f32_e32 vcc, s74, v117
	s_nop 1
	v_cndmask_b32_e32 v117, 0, v117, vcc
	v_sub_f32_e32 v50, v50, v117
	v_sub_f32_e32 v51, v51, v117
	v_sub_f32_e32 v52, v52, v117
	v_sub_f32_e32 v53, v53, v117
	v_sub_f32_e32 v54, v54, v117
	v_sub_f32_e32 v55, v55, v117
	v_sub_f32_e32 v56, v56, v117
	v_sub_f32_e32 v57, v57, v117
	v_sub_f32_e32 v126, v46, v117
	v_sub_f32_e32 v46, v106, v117
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_exp_f32_e32 v46, v46
	v_sub_f32_e32 v34, v34, v117
	v_sub_f32_e32 v35, v35, v117
	v_sub_f32_e32 v36, v36, v117
	v_sub_f32_e32 v37, v37, v117
	v_sub_f32_e32 v47, v47, v117
	v_sub_f32_e32 v58, v58, v117
	v_sub_f32_e32 v59, v59, v117
	v_sub_f32_e32 v60, v60, v117
	v_sub_f32_e32 v61, v61, v117
	v_sub_f32_e32 v62, v62, v117
	v_sub_f32_e32 v63, v63, v117
	v_sub_f32_e32 v64, v64, v117
	v_sub_f32_e32 v65, v65, v117
	v_sub_f32_e32 v38, v38, v117
	v_sub_f32_e32 v39, v39, v117
	v_sub_f32_e32 v40, v40, v117
	v_sub_f32_e32 v41, v41, v117
	v_sub_f32_e32 v42, v42, v117
	v_sub_f32_e32 v43, v43, v117
	v_sub_f32_e32 v44, v44, v117
	v_sub_f32_e32 v45, v45, v117
	v_sub_f32_e32 v48, v48, v117
	v_sub_f32_e32 v49, v49, v117
	v_mov_b32_e32 v106, v117
	v_exp_f32_e32 v117, v34
	v_exp_f32_e32 v127, v35
	v_exp_f32_e32 v128, v36
	v_exp_f32_e32 v129, v37
	v_cvt_pk_bf16_f32 v34, v50, v51
	v_cvt_pk_bf16_f32 v35, v52, v53
	v_cvt_pk_bf16_f32 v36, v54, v55
	v_cvt_pk_bf16_f32 v37, v56, v57
	v_pk_mul_f32 v[32:33], v[32:33], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[46:47] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_bf16 v[18:33], v[118:121], v[34:37], v[18:33]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_exp_f32_e32 v62, v62
	v_exp_f32_e32 v63, v63
	v_exp_f32_e32 v64, v64
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[138:141], v[34:37], v[2:17]
	v_exp_f32_e32 v65, v65
	v_cvt_pk_bf16_f32 v34, v58, v59
	v_cvt_pk_bf16_f32 v35, v60, v61
	v_cvt_pk_bf16_f32 v36, v62, v63
	v_cvt_pk_bf16_f32 v37, v64, v65
	v_add_f32_e32 v50, v50, v51
	v_exp_f32_e32 v130, v38
	v_mfma_f32_32x32x16_bf16 v[18:33], v[122:125], v[34:37], v[18:33]
	v_exp_f32_e32 v131, v39
	v_exp_f32_e32 v132, v40
	v_exp_f32_e32 v133, v41
	v_cvt_pk_bf16_f32 v38, v117, v127
	v_cvt_pk_bf16_f32 v39, v128, v129
	v_cvt_pk_bf16_f32 v40, v130, v131
	v_cvt_pk_bf16_f32 v41, v132, v133
	v_mfma_f32_32x32x16_bf16 v[2:17], v[134:137], v[34:37], v[2:17]
	v_add_f32_e32 v34, v52, v50
	v_add_f32_e32 v34, v53, v34
	v_add_f32_e32 v34, v54, v34
	v_add_f32_e32 v34, v55, v34
	v_add_f32_e32 v34, v56, v34
	v_add_f32_e32 v34, v57, v34
	v_add_f32_e32 v34, v58, v34
	v_add_f32_e32 v34, v59, v34
	v_add_f32_e32 v34, v60, v34
	v_add_f32_e32 v34, v61, v34
	v_add_f32_e32 v34, v62, v34
	v_add_f32_e32 v34, v63, v34
	v_add_f32_e32 v34, v64, v34
	v_add_f32_e32 v34, v65, v34
	v_add_f32_e32 v34, v117, v34
	v_add_f32_e32 v34, v127, v34
	v_add_f32_e32 v34, v128, v34
	v_add_f32_e32 v34, v129, v34
	v_exp_f32_e32 v150, v42
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_bf16 v[18:33], v[142:145], v[38:41], v[18:33]
	v_add_f32_e32 v34, v130, v34
	v_exp_f32_e32 v151, v43
	v_add_f32_e32 v34, v131, v34
	v_exp_f32_e32 v152, v44
	v_add_f32_e32 v34, v132, v34
	v_exp_f32_e32 v153, v45
	v_exp_f32_e32 v118, v126
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[146:149], v[38:41], v[2:17]
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_add_f32_e32 v34, v133, v34
	v_add_f32_e32 v34, v150, v34
	v_add_f32_e32 v34, v151, v34
	v_add_f32_e32 v34, v152, v34
	v_cvt_pk_bf16_f32 v42, v150, v151
	v_cvt_pk_bf16_f32 v43, v152, v153
	v_cvt_pk_bf16_f32 v44, v118, v47
	v_cvt_pk_bf16_f32 v45, v48, v49
	v_add_f32_e32 v34, v153, v34
	v_add_f32_e32 v34, v118, v34
	v_mfma_f32_32x32x16_bf16 v[18:33], v[86:89], v[42:45], v[18:33]
	v_add_f32_e32 v34, v47, v34
	v_add_f32_e32 v34, v48, v34
	v_add_f32_e32 v34, v49, v34
	v_mov_b32_e32 v35, v34
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v35
	v_add_f32_e32 v35, v34, v35
	v_mfma_f32_32x32x16_bf16 v[2:17], v[82:85], v[42:45], v[2:17]
	v_fmac_f32_e32 v35, v116, v46
	s_cbranch_scc1 .LBB0_7
.LBB0_8:                                ; %._crit_edge19
	v_div_scale_f32 v0, s[2:3], v35, v35, 1.0
	v_rcp_f32_e32 v1, v0
	v_add_u32_e32 v46, s84, v91
	v_lshrrev_b32_e32 v48, 2, v102
	s_and_b32 s81, s81, 0xffff
	v_fma_f32 v34, -v0, v1, 1.0
	v_fmac_f32_e32 v1, v34, v1
	v_div_scale_f32 v34, vcc, 1.0, v35, 1.0
	v_mul_f32_e32 v36, v34, v1
	v_fma_f32 v37, -v0, v36, v34
	v_fmac_f32_e32 v36, v37, v1
	v_fma_f32 v0, -v0, v36, v34
	v_div_fmas_f32 v0, v0, v1, v36
	v_div_fixup_f32 v0, v0, v35, 1.0
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[0:1], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[0:1], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[0:1], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[0:1], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[0:1], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[0:1], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[0:1], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[0:1], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[0:1], v[16:17] op_sel_hi:[0,1]
	v_mul_lo_u32 v0, s82, v90
	v_mad_u64_u32 v[46:47], s[2:3], s70, v46, v[0:1]
	v_cvt_pk_bf16_f32 v0, v18, v19
	v_cvt_pk_bf16_f32 v1, v20, v21
	v_cvt_pk_bf16_f32 v2, v22, v23
	v_cvt_pk_bf16_f32 v3, v24, v25
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_cvt_pk_bf16_f32 v15, v16, v17
	v_add_lshl_u32 v16, v48, v46, 1
	v_bfrev_b32_e32 v17, 1
	v_permlane32_swap_b32_e32 v0, v2
	v_permlane32_swap_b32_e32 v1, v3
	s_mov_b32 s83, 0x27000
	s_mov_b32 s82, 0x7ffffffe
	v_cndmask_b32_e64 v18, v17, v16, s[0:1]
	v_cvt_pk_bf16_f32 v4, v26, v27
	v_cvt_pk_bf16_f32 v5, v28, v29
	v_cvt_pk_bf16_f32 v6, v30, v31
	v_cvt_pk_bf16_f32 v7, v32, v33
	buffer_store_dwordx4 v[0:3], v18, s[80:83], 0 offen
	v_permlane32_swap_b32_e32 v4, v6
	s_nop 0
	v_add_u32_e32 v0, 32, v16
	v_permlane32_swap_b32_e32 v5, v7
	v_cndmask_b32_e64 v0, v17, v0, s[0:1]
	v_cvt_pk_bf16_f32 v8, v34, v35
	v_cvt_pk_bf16_f32 v9, v36, v37
	v_cvt_pk_bf16_f32 v10, v38, v39
	v_cvt_pk_bf16_f32 v11, v40, v41
	buffer_store_dwordx4 v[4:7], v0, s[80:83], 0 offen
	v_add_u32_e32 v0, 64, v16
	v_permlane32_swap_b32_e32 v8, v10
	v_permlane32_swap_b32_e32 v9, v11
	v_cndmask_b32_e64 v0, v17, v0, s[0:1]
	v_cvt_pk_bf16_f32 v12, v42, v43
	v_cvt_pk_bf16_f32 v13, v44, v45
	buffer_store_dwordx4 v[8:11], v0, s[80:83], 0 offen
	v_add_u32_e32 v0, 0x60, v16
	v_permlane32_swap_b32_e32 v12, v14
	v_permlane32_swap_b32_e32 v13, v15
	v_cndmask_b32_e64 v0, v17, v0, s[0:1]
	buffer_store_dwordx4 v[12:15], v0, s[80:83], 0 offen
.LBB0_9:                                ; %common.ret
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernel_unified_attention_2d
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 192
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 169
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 156
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	kernel_unified_attention_2d, .Lfunc_end0-kernel_unified_attention_2d
	.cfi_endproc
                                        ; -- End function
	.set kernel_unified_attention_2d.num_vgpr, 154
	.set kernel_unified_attention_2d.num_agpr, 0
	.set kernel_unified_attention_2d.numbered_sgpr, 86
	.set kernel_unified_attention_2d.num_named_barrier, 0
	.set kernel_unified_attention_2d.private_seg_size, 0
	.set kernel_unified_attention_2d.uses_vcc, 1
	.set kernel_unified_attention_2d.uses_flat_scratch, 0
	.set kernel_unified_attention_2d.has_dyn_sized_stack, 0
	.set kernel_unified_attention_2d.has_recursion, 0
	.set kernel_unified_attention_2d.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4788
; TotalNumSgprs: 92
; NumVgprs: 154
; NumAgprs: 0
; TotalNumVgprs: 154
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 169
; AccumOffset: 156
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 38
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x85 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x5f DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	112                             ; DW_AT_call_line
	.byte	68                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4d:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	211                             ; DW_AT_call_line
	.byte	44                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x59:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	335                             ; DW_AT_call_line
	.byte	35                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x66:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	189                             ; DW_AT_call_line
	.byte	40                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x73:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	345                             ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x80:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	291                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"unified_attention.py"          ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/aiter/aiter/ops/triton/_triton_kernels" ; string offset=28
.Linfo_string3:
	.asciz	"kernel_unified_attention_2d"   ; string offset=84
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           8
        .value_kind:     by_value
      - .offset:         72
        .size:           8
        .value_kind:     by_value
      - .offset:         80
        .size:           8
        .value_kind:     by_value
      - .offset:         88
        .size:           8
        .value_kind:     by_value
      - .offset:         96
        .size:           8
        .value_kind:     by_value
      - .offset:         104
        .size:           8
        .value_kind:     by_value
      - .offset:         112
        .size:           8
        .value_kind:     by_value
      - .offset:         120
        .size:           8
        .value_kind:     by_value
      - .offset:         128
        .size:           8
        .value_kind:     by_value
      - .offset:         136
        .size:           8
        .value_kind:     by_value
      - .offset:         144
        .size:           8
        .value_kind:     by_value
      - .offset:         152
        .size:           8
        .value_kind:     by_value
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     global_buffer
      - .offset:         168
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         176
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         184
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 192
    .max_flat_workgroup_size: 256
    .name:           kernel_unified_attention_2d
    .private_segment_fixed_size: 0
    .sgpr_count:     92
    .sgpr_spill_count: 0
    .symbol:         kernel_unified_attention_2d.kd
    .uses_dynamic_stack: false
    .vgpr_count:     154
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
