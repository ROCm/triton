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
	v_and_b32_e32 v35, 56, v1
	v_cmp_gt_i32_e64 s[0:1], s17, v3
	v_cmp_gt_i32_e64 s[2:3], s17, v4
	v_cmp_gt_i32_e64 s[20:21], s17, v5
	v_cmp_gt_i32_e64 s[22:23], 64, v10
	v_mad_u64_u32 v[4:5], s[6:7], s66, v6, v[2:3]
	v_mad_u64_u32 v[6:7], s[6:7], s66, v7, v[2:3]
	v_mad_u64_u32 v[12:13], s[6:7], s66, v8, v[2:3]
	v_mad_u64_u32 v[2:3], s[6:7], s66, v9, v[2:3]
	v_add_lshl_u32 v3, v35, v4, 1
	v_bfrev_b32_e32 v13, 1
	s_and_b64 vcc, s[22:23], vcc
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e32 v3, v13, v3, vcc
	v_add_lshl_u32 v4, v35, v6, 1
	s_and_b64 vcc, s[22:23], s[0:1]
	v_cndmask_b32_e32 v14, v13, v4, vcc
	buffer_load_dwordx4 v[4:7], v3, s[4:7], 0 offen sc0 nt
	buffer_load_dwordx4 v[8:11], v14, s[4:7], 0 offen sc0 nt
	v_add_lshl_u32 v3, v35, v12, 1
	s_and_b64 vcc, s[22:23], s[2:3]
	v_cndmask_b32_e32 v3, v13, v3, vcc
	v_add_lshl_u32 v2, v35, v2, 1
	s_and_b64 vcc, s[22:23], s[20:21]
	v_cndmask_b32_e32 v2, v13, v2, vcc
	buffer_load_dwordx4 v[12:15], v3, s[4:7], 0 offen sc0 nt
	buffer_load_dwordx4 v[16:19], v2, s[4:7], 0 offen sc0 nt
	v_and_b32_e32 v3, 0xc0, v0
	v_and_b32_e32 v86, 32, v0
	v_lshlrev_b32_e32 v36, 4, v0
	s_movk_i32 s0, 0x70
	v_and_b32_e32 v2, 31, v0
	v_bitop3_b32 v20, v36, v0, s0 bitop3:0x78
	v_lshlrev_b32_e32 v21, 6, v3
	v_and_b32_e32 v38, 0x70, v1
	v_lshrrev_b32_e32 v39, 1, v86
	v_lshlrev_b32_e32 v37, 7, v2
	v_add_u32_e32 v87, 0, v20
	v_bitop3_b32 v20, v38, v21, v39 bitop3:0xde
	s_movk_i32 s1, 0x60
	v_or_b32_e32 v21, v20, v37
	v_bitop3_b32 v20, v20, s1, v37 bitop3:0x36
	v_add_u32_e32 v22, 0, v21
	v_xad_u32 v23, v21, 32, 0
	v_xad_u32 v21, v21, 64, 0
	v_add_u32_e32 v20, 0, v20
	v_and_or_b32 v74, v0, 7, s33
	v_ashrrev_i32_e32 v75, 31, v74
	v_cmp_gt_i32_e32 vcc, 64, v74
	v_mov_b32_e32 v40, 0xff800000
	s_waitcnt vmcnt(3)
	ds_write_b128 v87, v[4:7]
	s_waitcnt vmcnt(2)
	ds_write_b128 v87, v[8:11] offset:4096
	s_waitcnt vmcnt(1)
	ds_write_b128 v87, v[12:15] offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b128 v87, v[16:19] offset:12288
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[50:53], v22
	ds_read_b128 v[54:57], v23
	ds_read_b128 v[58:61], v21
	ds_read_b128 v[62:65], v20
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:
	v_lshl_add_u64 v[4:5], v[74:75], 2, s[10:11]
	global_load_dword v40, v[4:5], off
.LBB0_5:
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v3, 4, v3
	v_lshrrev_b32_e32 v2, 3, v2
	v_or3_b32 v75, v3, v2, s29
	v_cmp_gt_i32_e64 s[0:1], s17, v75
	s_and_b64 s[0:1], vcc, s[0:1]
	s_add_u32 s2, s14, s30
	s_addc_u32 s3, s15, s31
	v_mov_b32_e32 v19, 0
	global_load_dword v16, v19, s[2:3] offset:-4
	v_mov_b32_e32 v90, 1.0
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
	v_add3_u32 v88, v75, s2, 1
	s_lshl_b64 s[2:3], s[6:7], 1
	s_add_u32 s2, s18, s2
	s_mul_i32 s4, s26, s16
	s_addc_u32 s3, s19, s3
	v_lshlrev_b32_e32 v2, 1, v35
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[76:77], s[2:3], 0, v[2:3]
	s_lshl_b64 s[2:3], s[4:5], 1
	s_add_u32 s2, s8, s2
	v_and_b32_e32 v4, 0x70, v0
	s_addc_u32 s3, s9, s3
	v_lshl_add_u64 v[78:79], s[2:3], 0, v[2:3]
	v_lshrrev_b32_e32 v2, 1, v4
	v_xor_b32_e32 v91, v36, v2
	v_lshlrev_b32_e32 v2, 5, v0
	v_bfe_i32 v4, v0, 3, 1
	v_lshlrev_b32_e32 v0, 1, v0
	v_and_b32_e32 v0, 32, v0
	s_movk_i32 s2, 0x80
	v_and_b32_e32 v1, 24, v1
	v_mov_b32_e32 v5, 0x210
	v_cmp_eq_u32_e32 vcc, 0, v86
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
	v_mad_u64_u32 v[80:81], s[2:3], s24, v13, 0
	v_mad_u64_u32 v[4:5], s[2:3], s25, v34, v[2:3]
	v_mov_b32_e32 v2, v81
	v_mad_u64_u32 v[82:83], s[2:3], s74, v34, 0
	v_mov_b32_e32 v1, v4
	v_mad_u64_u32 v[4:5], s[2:3], s25, v13, v[2:3]
	v_mov_b32_e32 v2, v83
	v_mad_u64_u32 v[84:85], s[2:3], s74, v13, 0
	v_bitop3_b32 v6, v37, v39, v38 bitop3:0x36
	v_mov_b32_e32 v81, v4
	v_mad_u64_u32 v[4:5], s[2:3], s75, v34, v[2:3]
	v_mov_b32_e32 v2, v85
	v_xor_b32_e32 v7, 32, v6
	v_xor_b32_e32 v8, 64, v6
	v_xor_b32_e32 v9, 0x60, v6
	v_xor_b32_e32 v10, 8, v91
	v_mov_b32_e32 v83, v4
	v_mad_u64_u32 v[4:5], s[2:3], s75, v13, v[2:3]
	s_mul_i32 s71, s71, s64
	s_mov_b32 s76, 0
	v_lshrrev_b32_e32 v89, 3, v86
	v_mul_f32_e32 v100, 0x3fb8aa3b, v40
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_and_b32 s13, s13, 0xffff
	v_mov_b32_e32 v85, v4
	v_mov_b32_e32 v90, 1.0
	v_add_u32_e32 v92, 0, v7
	v_add_u32_e32 v93, 0, v8
	v_add_u32_e32 v94, 0, v9
	s_mov_b32 s74, 0xff800000
	v_add_u32_e32 v95, 0, v10
	v_add_u32_e32 v96, 0, v11
	v_add_u32_e32 v97, 0, v12
	v_add_u32_e32 v98, 0, v6
	v_mov_b32_e32 v99, 0xff800000
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
	s_add_i32 s2, s2, s71
	s_lshl_b32 s2, s2, 2
	v_mov_b32_e32 v34, s2
	buffer_load_dword v35, v34, s[12:15], 0 offen
	v_add_u32_e32 v117, 0, v91
	v_cmp_lt_i32_e64 s[68:69], v89, v88
	s_and_b64 s[68:69], s[0:1], s[68:69]
	s_add_i32 s76, s76, 1
	s_waitcnt vmcnt(0)
	v_ashrrev_i32_e32 v37, 31, v35
	v_mul_hi_u32 v38, s78, v35
	v_mul_hi_u32 v39, s72, v35
	v_mul_lo_u32 v40, s79, v35
	v_mul_lo_u32 v34, s78, v35
	v_mul_lo_u32 v41, s73, v35
	v_mul_lo_u32 v36, s72, v35
	v_mul_lo_u32 v35, s78, v37
	v_mul_lo_u32 v37, s72, v37
	v_add_u32_e32 v37, v39, v37
	v_add_u32_e32 v35, v38, v35
	v_add_u32_e32 v37, v37, v41
	v_add_u32_e32 v35, v35, v40
	v_lshl_add_u64 v[36:37], v[36:37], 1, v[76:77]
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[78:79]
	v_lshl_add_u64 v[42:43], v[82:83], 1, v[36:37]
	v_lshl_add_u64 v[44:45], v[84:85], 1, v[36:37]
	v_lshl_add_u64 v[46:47], v[0:1], 1, v[34:35]
	v_lshl_add_u64 v[48:49], v[80:81], 1, v[34:35]
	global_load_dwordx4 v[34:37], v[42:43], off
	global_load_dwordx4 v[38:41], v[44:45], off
	global_load_dwordx4 v[66:69], v[46:47], off
	global_load_dwordx4 v[70:73], v[48:49], off
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(3)
	ds_write_b128 v87, v[34:37]
	s_waitcnt vmcnt(2)
	ds_write_b128 v87, v[38:41] offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[34:37], v98
	ds_read_b128 v[102:105], v92
	ds_read_b128 v[118:121], v92 offset:4096
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[34:37], v[50:53], 0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[102:105], v[54:57], v[34:49]
	ds_read_b128 v[102:105], v93
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[102:105], v[58:61], v[34:49]
	ds_read_b128 v[102:105], v94
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[102:105], v[62:65], v[34:49]
	s_nop 11
	v_mul_f32_e32 v101, 0x3e38aa3b, v34
	v_mul_f32_e32 v102, 0x3e38aa3b, v35
	v_mul_f32_e32 v103, 0x3e38aa3b, v36
	v_mul_f32_e32 v104, 0x3e38aa3b, v37
	ds_read_b128 v[34:37], v98 offset:4096
	v_mul_f32_e32 v105, 0x3e38aa3b, v38
	v_mul_f32_e32 v106, 0x3e38aa3b, v39
	v_mul_f32_e32 v107, 0x3e38aa3b, v40
	v_mul_f32_e32 v108, 0x3e38aa3b, v41
	v_mul_f32_e32 v109, 0x3e38aa3b, v42
	v_mul_f32_e32 v110, 0x3e38aa3b, v43
	v_mul_f32_e32 v111, 0x3e38aa3b, v44
	v_mul_f32_e32 v112, 0x3e38aa3b, v45
	v_mul_f32_e32 v113, 0x3e38aa3b, v46
	v_mul_f32_e32 v114, 0x3e38aa3b, v47
	v_mul_f32_e32 v115, 0x3e38aa3b, v48
	v_mul_f32_e32 v116, 0x3e38aa3b, v49
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[34:37], v[50:53], 0
	v_mfma_f32_32x32x16_bf16 v[34:49], v[118:121], v[54:57], v[34:49]
	ds_read_b128 v[118:121], v93 offset:4096
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[118:121], v[58:61], v[34:49]
	ds_read_b128 v[118:121], v94 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write2st64_b64 v117, v[66:67], v[70:71] offset1:8
	ds_write2st64_b64 v95, v[68:69], v[72:73] offset1:8
	v_add_u32_e32 v66, 1, v89
	v_cmp_lt_i32_e32 vcc, v66, v88
	v_add_u32_e32 v66, 2, v89
	v_cmp_lt_i32_e64 s[2:3], v66, v88
	v_add_u32_e32 v66, 3, v89
	v_cmp_lt_i32_e64 s[4:5], v66, v88
	v_add_u32_e32 v66, 8, v89
	v_cmp_lt_i32_e64 s[6:7], v66, v88
	v_add_u32_e32 v66, 9, v89
	v_cmp_lt_i32_e64 s[8:9], v66, v88
	v_add_u32_e32 v66, 10, v89
	v_cmp_lt_i32_e64 s[10:11], v66, v88
	v_add_u32_e32 v66, 11, v89
	v_cmp_lt_i32_e64 s[16:17], v66, v88
	v_add_u32_e32 v66, 16, v89
	v_cmp_lt_i32_e64 s[18:19], v66, v88
	v_add_u32_e32 v66, 17, v89
	v_cmp_lt_i32_e64 s[20:21], v66, v88
	v_add_u32_e32 v66, 18, v89
	v_cmp_lt_i32_e64 s[22:23], v66, v88
	v_add_u32_e32 v66, 19, v89
	v_cmp_lt_i32_e64 s[24:25], v66, v88
	v_add_u32_e32 v66, 24, v89
	v_cmp_lt_i32_e64 s[26:27], v66, v88
	v_add_u32_e32 v66, 25, v89
	v_cmp_lt_i32_e64 s[28:29], v66, v88
	v_add_u32_e32 v66, 26, v89
	v_cmp_lt_i32_e64 s[30:31], v66, v88
	v_add_u32_e32 v66, 27, v89
	v_cmp_lt_i32_e64 s[34:35], v66, v88
	v_add_u32_e32 v66, 32, v89
	v_cmp_lt_i32_e64 s[36:37], v66, v88
	v_add_u32_e32 v66, 33, v89
	v_cmp_lt_i32_e64 s[38:39], v66, v88
	v_add_u32_e32 v66, 34, v89
	v_cmp_lt_i32_e64 s[40:41], v66, v88
	v_add_u32_e32 v66, 35, v89
	v_cmp_lt_i32_e64 s[42:43], v66, v88
	v_add_u32_e32 v66, 40, v89
	v_cmp_lt_i32_e64 s[44:45], v66, v88
	v_add_u32_e32 v66, 41, v89
	v_cmp_lt_i32_e64 s[46:47], v66, v88
	v_add_u32_e32 v66, 42, v89
	v_cmp_lt_i32_e64 s[48:49], v66, v88
	v_add_u32_e32 v66, 43, v89
	v_mfma_f32_32x32x16_bf16 v[34:49], v[118:121], v[62:65], v[34:49]
	v_cmp_lt_i32_e64 s[50:51], v66, v88
	v_add_u32_e32 v66, 48, v89
	v_cmp_lt_i32_e64 s[52:53], v66, v88
	v_add_u32_e32 v66, 49, v89
	v_cmp_lt_i32_e64 s[54:55], v66, v88
	v_add_u32_e32 v66, 50, v89
	v_cmp_lt_i32_e64 s[56:57], v66, v88
	v_add_u32_e32 v66, 51, v89
	v_cmp_lt_i32_e64 s[58:59], v66, v88
	v_add_u32_e32 v66, 56, v89
	v_cmp_lt_i32_e64 s[60:61], v66, v88
	v_add_u32_e32 v66, 57, v89
	v_cmp_lt_i32_e64 s[62:63], v66, v88
	v_add_u32_e32 v66, 58, v89
	v_cmp_lt_i32_e64 s[64:65], v66, v88
	v_add_u32_e32 v66, 59, v89
	s_and_b64 vcc, s[0:1], vcc
	v_mul_f32_e32 v34, 0x3e38aa3b, v34
	v_cmp_lt_i32_e64 s[66:67], v66, v88
	s_and_b64 s[2:3], s[0:1], s[2:3]
	s_and_b64 s[4:5], s[0:1], s[4:5]
	s_and_b64 s[18:19], s[0:1], s[18:19]
	s_and_b64 s[36:37], s[0:1], s[36:37]
	v_cndmask_b32_e64 v66, v99, v101, s[68:69]
	v_cndmask_b32_e32 v67, v99, v102, vcc
	s_and_b64 s[6:7], s[0:1], s[6:7]
	s_and_b64 s[8:9], s[0:1], s[8:9]
	v_cndmask_b32_e64 v68, v99, v103, s[2:3]
	v_cndmask_b32_e64 v69, v99, v104, s[4:5]
	v_cndmask_b32_e64 v101, v99, v109, s[18:19]
	v_cndmask_b32_e64 v109, v99, v34, s[36:37]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v34, v66, v67
	s_and_b64 s[10:11], s[0:1], s[10:11]
	s_and_b64 s[16:17], s[0:1], s[16:17]
	v_cndmask_b32_e64 v70, v99, v105, s[6:7]
	v_cndmask_b32_e64 v71, v99, v106, s[8:9]
	v_max3_f32 v34, v34, v68, v69
	s_and_b64 s[20:21], s[0:1], s[20:21]
	v_cndmask_b32_e64 v72, v99, v107, s[10:11]
	v_cndmask_b32_e64 v73, v99, v108, s[16:17]
	v_max3_f32 v34, v34, v70, v71
	s_and_b64 s[22:23], s[0:1], s[22:23]
	s_and_b64 s[24:25], s[0:1], s[24:25]
	v_cndmask_b32_e64 v102, v99, v110, s[20:21]
	v_max3_f32 v34, v34, v72, v73
	s_and_b64 s[26:27], s[0:1], s[26:27]
	s_and_b64 s[28:29], s[0:1], s[28:29]
	v_cndmask_b32_e64 v103, v99, v111, s[22:23]
	v_cndmask_b32_e64 v104, v99, v112, s[24:25]
	v_max3_f32 v34, v34, v101, v102
	s_and_b64 s[30:31], s[0:1], s[30:31]
	s_and_b64 s[34:35], s[0:1], s[34:35]
	v_cndmask_b32_e64 v105, v99, v113, s[26:27]
	v_cndmask_b32_e64 v106, v99, v114, s[28:29]
	v_max3_f32 v34, v34, v103, v104
	v_mul_f32_e32 v35, 0x3e38aa3b, v35
	s_and_b64 s[38:39], s[0:1], s[38:39]
	v_cndmask_b32_e64 v107, v99, v115, s[30:31]
	v_cndmask_b32_e64 v108, v99, v116, s[34:35]
	v_max3_f32 v34, v34, v105, v106
	v_mul_f32_e32 v36, 0x3e38aa3b, v36
	v_mul_f32_e32 v37, 0x3e38aa3b, v37
	s_and_b64 s[40:41], s[0:1], s[40:41]
	s_and_b64 s[42:43], s[0:1], s[42:43]
	v_cndmask_b32_e64 v110, v99, v35, s[38:39]
	v_max3_f32 v34, v34, v107, v108
	v_mul_f32_e32 v38, 0x3e38aa3b, v38
	v_mul_f32_e32 v39, 0x3e38aa3b, v39
	s_and_b64 s[44:45], s[0:1], s[44:45]
	s_and_b64 s[46:47], s[0:1], s[46:47]
	v_cndmask_b32_e64 v111, v99, v36, s[40:41]
	v_cndmask_b32_e64 v112, v99, v37, s[42:43]
	v_max3_f32 v34, v34, v109, v110
	v_mul_f32_e32 v40, 0x3e38aa3b, v40
	v_mul_f32_e32 v41, 0x3e38aa3b, v41
	s_and_b64 s[48:49], s[0:1], s[48:49]
	s_and_b64 s[50:51], s[0:1], s[50:51]
	v_cndmask_b32_e64 v113, v99, v38, s[44:45]
	v_cndmask_b32_e64 v114, v99, v39, s[46:47]
	v_max3_f32 v34, v34, v111, v112
	v_mul_f32_e32 v42, 0x3e38aa3b, v42
	v_mul_f32_e32 v43, 0x3e38aa3b, v43
	s_and_b64 s[52:53], s[0:1], s[52:53]
	s_and_b64 s[54:55], s[0:1], s[54:55]
	v_cndmask_b32_e64 v115, v99, v40, s[48:49]
	v_cndmask_b32_e64 v116, v99, v41, s[50:51]
	v_max3_f32 v34, v34, v113, v114
	v_mul_f32_e32 v44, 0x3e38aa3b, v44
	v_mul_f32_e32 v45, 0x3e38aa3b, v45
	s_and_b64 s[56:57], s[0:1], s[56:57]
	s_and_b64 s[58:59], s[0:1], s[58:59]
	v_cndmask_b32_e64 v36, v99, v42, s[52:53]
	v_cndmask_b32_e64 v37, v99, v43, s[54:55]
	v_max3_f32 v34, v34, v115, v116
	v_mul_f32_e32 v46, 0x3e38aa3b, v46
	v_mul_f32_e32 v47, 0x3e38aa3b, v47
	s_and_b64 s[60:61], s[0:1], s[60:61]
	s_and_b64 s[62:63], s[0:1], s[62:63]
	v_cndmask_b32_e64 v38, v99, v44, s[56:57]
	v_cndmask_b32_e64 v39, v99, v45, s[58:59]
	v_max3_f32 v34, v34, v36, v37
	v_mul_f32_e32 v48, 0x3e38aa3b, v48
	v_mul_f32_e32 v49, 0x3e38aa3b, v49
	s_and_b64 s[64:65], s[0:1], s[64:65]
	s_and_b64 s[66:67], s[0:1], s[66:67]
	v_cndmask_b32_e64 v40, v99, v46, s[60:61]
	v_cndmask_b32_e64 v41, v99, v47, s[62:63]
	v_max3_f32 v34, v34, v38, v39
	v_cndmask_b32_e64 v42, v99, v48, s[64:65]
	v_cndmask_b32_e64 v43, v99, v49, s[66:67]
	v_max3_f32 v34, v34, v40, v41
	v_max3_f32 v34, v34, v42, v43
	v_mov_b32_e32 v35, v34
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v35
	v_max3_f32 v34, v100, v34, v35
	v_cmp_lg_f32_e32 vcc, s74, v34
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cndmask_b32_e32 v35, 0, v34, vcc
	v_sub_f32_e32 v34, v66, v35
	v_sub_f32_e32 v48, v67, v35
	v_sub_f32_e32 v49, v68, v35
	v_sub_f32_e32 v66, v69, v35
	v_sub_f32_e32 v67, v70, v35
	v_sub_f32_e32 v68, v71, v35
	v_sub_f32_e32 v69, v72, v35
	v_sub_f32_e32 v70, v73, v35
	v_sub_f32_e32 v71, v100, v35
	ds_read_b64_tr_b16 v[44:45], v96
	ds_read_b64_tr_b16 v[46:47], v97 offset:1024
	v_exp_f32_e32 v72, v34
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v73, v66
	v_exp_f32_e32 v100, v67
	v_exp_f32_e32 v117, v68
	v_exp_f32_e32 v118, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v34, v71
	v_cvt_pk_bf16_f32 v66, v72, v48
	v_cvt_pk_bf16_f32 v67, v49, v73
	v_cvt_pk_bf16_f32 v68, v100, v117
	v_cvt_pk_bf16_f32 v69, v118, v70
	v_pk_mul_f32 v[32:33], v[32:33], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[34:35] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[18:33], v[44:47], v[66:69], v[18:33]
	ds_read_b64_tr_b16 v[44:45], v96 offset:64
	ds_read_b64_tr_b16 v[46:47], v97 offset:1088
	v_mul_f32_e64 v12, v12, v34
	v_mul_f32_e64 v13, v13, v34
	v_mul_f32_e64 v10, v10, v34
	v_mul_f32_e64 v11, v11, v34
	v_pk_mul_f32 v[8:9], v[8:9], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[34:35] op_sel_hi:[1,0]
	v_sub_f32_e32 v71, v105, v35
	v_exp_f32_e32 v71, v71
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[44:47], v[66:69], v[2:17]
	v_sub_f32_e32 v66, v101, v35
	v_sub_f32_e32 v67, v102, v35
	v_sub_f32_e32 v68, v103, v35
	v_sub_f32_e32 v69, v104, v35
	v_sub_f32_e32 v101, v106, v35
	v_sub_f32_e32 v102, v107, v35
	v_sub_f32_e32 v103, v108, v35
	ds_read_b64_tr_b16 v[44:45], v96 offset:2048
	ds_read_b64_tr_b16 v[46:47], v97 offset:3072
	v_exp_f32_e32 v104, v66
	v_exp_f32_e32 v105, v67
	v_exp_f32_e32 v106, v68
	v_exp_f32_e32 v107, v69
	v_exp_f32_e32 v101, v101
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	v_cvt_pk_bf16_f32 v66, v104, v105
	v_cvt_pk_bf16_f32 v67, v106, v107
	v_cvt_pk_bf16_f32 v68, v71, v101
	v_cvt_pk_bf16_f32 v69, v102, v103
	v_sub_f32_e32 v108, v113, v35
	v_exp_f32_e32 v108, v108
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[18:33], v[44:47], v[66:69], v[18:33]
	ds_read_b64_tr_b16 v[46:47], v97 offset:3136
	ds_read_b64_tr_b16 v[44:45], v96 offset:2112
	v_sub_f32_e32 v36, v36, v35
	v_sub_f32_e32 v37, v37, v35
	v_sub_f32_e32 v38, v38, v35
	v_sub_f32_e32 v39, v39, v35
	v_sub_f32_e32 v40, v40, v35
	v_sub_f32_e32 v41, v41, v35
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[44:47], v[66:69], v[2:17]
	v_sub_f32_e32 v66, v109, v35
	v_sub_f32_e32 v67, v110, v35
	v_sub_f32_e32 v68, v111, v35
	v_sub_f32_e32 v69, v112, v35
	v_sub_f32_e32 v109, v114, v35
	v_sub_f32_e32 v110, v115, v35
	v_sub_f32_e32 v111, v116, v35
	ds_read_b64_tr_b16 v[44:45], v96 offset:4096
	ds_read_b64_tr_b16 v[46:47], v97 offset:5120
	v_exp_f32_e32 v112, v66
	v_exp_f32_e32 v113, v67
	v_exp_f32_e32 v114, v68
	v_exp_f32_e32 v115, v69
	v_exp_f32_e32 v109, v109
	v_exp_f32_e32 v110, v110
	v_exp_f32_e32 v111, v111
	v_cvt_pk_bf16_f32 v66, v112, v113
	v_cvt_pk_bf16_f32 v67, v114, v115
	v_cvt_pk_bf16_f32 v68, v108, v109
	v_cvt_pk_bf16_f32 v69, v110, v111
	v_sub_f32_e32 v42, v42, v35
	v_sub_f32_e32 v43, v43, v35
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[18:33], v[44:47], v[66:69], v[18:33]
	ds_read_b64_tr_b16 v[44:45], v96 offset:4160
	ds_read_b64_tr_b16 v[46:47], v97 offset:5184
	v_exp_f32_e32 v116, v40
	v_exp_f32_e32 v119, v41
	v_exp_f32_e32 v120, v42
	v_exp_f32_e32 v121, v43
	v_add_u32_e32 v89, 64, v89
	s_cmp_lg_u32 s33, s76
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[44:47], v[66:69], v[2:17]
	ds_read_b64_tr_b16 v[44:45], v96 offset:6144
	ds_read_b64_tr_b16 v[46:47], v97 offset:7168
	v_exp_f32_e32 v66, v36
	v_exp_f32_e32 v67, v37
	v_exp_f32_e32 v68, v38
	v_exp_f32_e32 v69, v39
	ds_read_b64_tr_b16 v[42:43], v97 offset:7232
	ds_read_b64_tr_b16 v[40:41], v96 offset:6208
	v_cvt_pk_bf16_f32 v36, v66, v67
	v_cvt_pk_bf16_f32 v38, v116, v119
	v_cvt_pk_bf16_f32 v37, v68, v69
	v_cvt_pk_bf16_f32 v39, v120, v121
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_bf16 v[18:33], v[44:47], v[36:39], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[2:17], v[40:43], v[36:39], v[2:17]
	v_add_f32_e32 v36, v72, v48
	v_add_f32_e32 v36, v49, v36
	v_add_f32_e32 v36, v73, v36
	v_add_f32_e32 v36, v100, v36
	v_add_f32_e32 v36, v117, v36
	v_add_f32_e32 v36, v118, v36
	v_add_f32_e32 v36, v70, v36
	v_add_f32_e32 v36, v104, v36
	v_add_f32_e32 v36, v105, v36
	v_add_f32_e32 v36, v106, v36
	v_add_f32_e32 v36, v107, v36
	v_add_f32_e32 v36, v71, v36
	v_add_f32_e32 v36, v101, v36
	v_add_f32_e32 v36, v102, v36
	v_add_f32_e32 v36, v103, v36
	v_add_f32_e32 v36, v112, v36
	v_add_f32_e32 v36, v113, v36
	v_add_f32_e32 v36, v114, v36
	v_add_f32_e32 v36, v115, v36
	v_add_f32_e32 v36, v108, v36
	v_add_f32_e32 v36, v109, v36
	v_add_f32_e32 v36, v110, v36
	v_add_f32_e32 v36, v111, v36
	v_mov_b32_e32 v100, v35
	v_add_f32_e32 v35, v66, v36
	v_add_f32_e32 v35, v67, v35
	v_add_f32_e32 v35, v68, v35
	v_add_f32_e32 v35, v69, v35
	v_add_f32_e32 v35, v116, v35
	v_add_f32_e32 v35, v119, v35
	v_add_f32_e32 v35, v120, v35
	v_add_f32_e32 v35, v121, v35
	v_mov_b32_e32 v36, v35
	s_nop 1
	v_permlane32_swap_b32_e32 v35, v36
	v_mov_b32_e32 v37, v90
	v_add_f32_e32 v90, v35, v36
	v_fmac_f32_e32 v90, v37, v34
	s_cbranch_scc1 .LBB0_7
.LBB0_8:                                ; %._crit_edge19
	v_div_scale_f32 v0, s[2:3], v90, v90, 1.0
	v_rcp_f32_e32 v1, v0
	v_add_u32_e32 v46, s84, v75
	v_lshrrev_b32_e32 v48, 2, v86
	s_and_b32 s81, s81, 0xffff
	v_fma_f32 v34, -v0, v1, 1.0
	v_fmac_f32_e32 v1, v34, v1
	v_div_scale_f32 v34, vcc, 1.0, v90, 1.0
	v_mul_f32_e32 v35, v34, v1
	v_fma_f32 v36, -v0, v35, v34
	v_fmac_f32_e32 v35, v36, v1
	v_fma_f32 v0, -v0, v35, v34
	v_div_fmas_f32 v0, v0, v1, v35
	v_div_fixup_f32 v0, v0, v90, 1.0
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
	v_mul_lo_u32 v0, s82, v74
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
		.amdhsa_next_free_vgpr 122
		.amdhsa_next_free_sgpr 86
		.amdhsa_accum_offset 124
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
	.set kernel_unified_attention_2d.num_vgpr, 122
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
; codeLenInByte = 4820
; TotalNumSgprs: 92
; NumVgprs: 122
; NumAgprs: 0
; TotalNumVgprs: 122
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 11
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 92
; NumVGPRsForWavesPerEU: 122
; AccumOffset: 124
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 30
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
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
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
	.byte	1                               ; Abbrev [1] 0xb:0x8d DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x67 DW_TAG_subprogram
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
	.byte	6                               ; Abbrev [6] 0x73:0x23 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp40                         ; DW_AT_low_pc
	.long	.Ltmp43-.Ltmp40                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	345                             ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x88:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
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
    .vgpr_count:     122
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
