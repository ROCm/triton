	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	attn_fwd                        ; -- Begin function attn_fwd
	.p2align	8
	.type	attn_fwd,@function
attn_fwd:                               ; @attn_fwd
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.11:
	.file	1 "/var/lib/jenkins/OAI-triton/python/../fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.12:
.LBB0_0:
	s_load_dwordx8 s[20:27], s[0:1], 0x38
	s_mul_i32 s0, s12, s18
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s28, s16, 8
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s13, s17
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s14, s28
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s19, s14, 5
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v35, 3, v0
	v_lshrrev_b32_e32 v1, 4, v0
	s_add_u32 s0, s2, s0
	v_and_b32_e32 v34, 0x78, v35
	s_mul_i32 s30, s15, s18
	s_addc_u32 s1, s3, s1
	v_mad_u64_u32 v[2:3], s[2:3], s14, v1, v[34:35]
	s_ashr_i32 s31, s30, 31
	s_lshl_b64 s[2:3], s[30:31], 1
	s_add_u32 s12, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s34, s20, s17
	s_addc_u32 s13, s5, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 1
	s_add_u32 s20, s12, s2
	s_mul_i32 s2, s22, s18
	s_addc_u32 s13, s13, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s23, s17
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	v_or_b32_e32 v4, s28, v1
	s_add_u32 s12, s6, s2
	s_movk_i32 s6, 0x4000
	v_or_b32_e32 v5, 32, v4
	v_add_u32_e32 v3, s19, v2
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v45, 1
	v_cmp_gt_i32_e32 vcc, s6, v4
	v_or_b32_e32 v10, 64, v4
	v_add_u32_e32 v12, s19, v3
	s_addc_u32 s16, s7, s3
	s_and_b32 s2, s14, 0x3fff
	v_cndmask_b32_e32 v14, v45, v2, vcc
	v_lshlrev_b32_e32 v2, 1, v3
	v_cmp_gt_i32_e32 vcc, s6, v5
	v_or_b32_e32 v11, 0x60, v4
	v_add_u32_e32 v13, s19, v12
	s_bitset1_b32 s2, 14
	v_cndmask_b32_e32 v15, v45, v2, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	v_cmp_gt_i32_e32 vcc, s6, v10
	v_or_b32_e32 v18, 0x80, v4
	v_add_u32_e32 v20, s19, v13
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_cndmask_b32_e32 v22, v45, v12, vcc
	v_lshlrev_b32_e32 v10, 1, v13
	v_cmp_gt_i32_e32 vcc, s6, v11
	v_or_b32_e32 v19, 0xa0, v4
	v_add_u32_e32 v21, s19, v20
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v23, v45, v10, vcc
	v_lshlrev_b32_e32 v20, 1, v20
	v_cmp_gt_i32_e32 vcc, s6, v18
	v_or_b32_e32 v26, 0xc0, v4
	v_or_b32_e32 v27, 0xe0, v4
	v_add_u32_e32 v28, s19, v21
	buffer_load_dwordx4 v[2:5], v14, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v15, s[0:3], 0 offen
	v_cndmask_b32_e32 v29, v45, v20, vcc
	v_lshlrev_b32_e32 v18, 1, v21
	v_cmp_gt_i32_e32 vcc, s6, v19
	buffer_load_dwordx4 v[10:13], v22, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v23, s[0:3], 0 offen
	v_cndmask_b32_e32 v30, v45, v18, vcc
	buffer_load_dwordx4 v[18:21], v29, s[0:3], 0 offen
	buffer_load_dwordx4 v[22:25], v30, s[0:3], 0 offen
	v_lshlrev_b32_e32 v29, 1, v28
	v_cmp_gt_i32_e32 vcc, s6, v26
	v_add_lshl_u32 v26, v28, s19, 1
	v_lshrrev_b32_e32 v204, 1, v0
	v_cndmask_b32_e32 v36, v45, v29, vcc
	v_cmp_gt_i32_e32 vcc, s6, v27
	v_lshlrev_b32_e32 v47, 7, v1
	s_movk_i32 s33, 0x78
	v_cndmask_b32_e32 v37, v45, v26, vcc
	buffer_load_dwordx4 v[26:29], v36, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v37, s[0:3], 0 offen
	v_and_b32_e32 v36, 0x78, v204
	v_bitop3_b32 v36, v36, v47, v34 bitop3:0xde
	v_lshlrev_b32_e32 v48, 1, v36
	v_add_u32_e32 v36, 0, v48
	s_barrier
	v_bitop3_b32 v46, v204, v35, s33 bitop3:0x28
	v_and_b32_e32 v203, 31, v0
	s_movk_i32 s6, 0xe0
	v_lshrrev_b32_e32 v44, 2, v0
	v_and_b32_e32 v74, 63, v0
	v_mad_u64_u32 v[200:201], s[0:1], s21, v1, v[34:35]
	v_lshl_add_u32 v206, s21, 5, v200
	s_and_b32 s0, s21, 0x3fff
	s_lshl_b32 s36, s21, 6
	s_bitset1_b32 s0, 14
	s_and_b32 s1, s13, 0xffff
	s_lshl_b32 s19, s0, 16
	s_ashr_i32 s37, s36, 31
	s_lshl_b32 s38, s24, 6
	s_or_b32 s21, s1, s19
	s_mov_b32 s22, s2
	s_mov_b32 s23, s3
	v_lshlrev_b32_e32 v217, 7, v203
	s_waitcnt vmcnt(7)
	ds_write_b128 v36, v[2:5]
	v_or_b32_e32 v2, 0x1000, v47
	v_or_b32_e32 v3, v2, v46
	v_lshlrev_b32_e32 v4, 1, v3
	v_add_u32_e32 v3, 0, v4
	s_waitcnt vmcnt(6)
	ds_write_b128 v3, v[6:9]
	s_waitcnt vmcnt(5)
	ds_write_b128 v36, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v36, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v36, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v36, v[22:25] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v36, v[26:29] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v36, v[30:33] offset:57344
	v_and_b32_e32 v36, 8, v44
	v_and_or_b32 v3, v204, s6, v203
	v_or_b32_e32 v37, 16, v36
	v_lshlrev_b32_e32 v3, 7, v3
	v_or_b32_e32 v38, 32, v36
	v_or_b32_e32 v39, 48, v36
	v_bitop3_b32 v5, v3, v36, v34 bitop3:0xf6
	v_bitop3_b32 v6, v3, v37, v34 bitop3:0xf6
	v_or_b32_e32 v40, 64, v36
	v_or_b32_e32 v41, 0x50, v36
	v_bitop3_b32 v7, v3, v38, v34 bitop3:0xf6
	v_bitop3_b32 v8, v3, v39, v34 bitop3:0xf6
	v_lshl_add_u32 v5, v5, 1, 0
	v_lshl_add_u32 v6, v6, 1, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_or_b32_e32 v42, 0x60, v36
	v_or_b32_e32 v43, 0x70, v36
	v_bitop3_b32 v9, v3, v40, v34 bitop3:0xf6
	v_bitop3_b32 v10, v3, v41, v34 bitop3:0xf6
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[146:149], v6
	v_lshl_add_u32 v5, v7, 1, 0
	v_lshl_add_u32 v6, v8, 1, 0
	v_bitop3_b32 v11, v3, v42, v34 bitop3:0xf6
	v_bitop3_b32 v3, v3, v43, v34 bitop3:0xf6
	ds_read_b128 v[142:145], v5
	ds_read_b128 v[138:141], v6
	v_lshl_add_u32 v5, v9, 1, 0
	v_lshl_add_u32 v6, v10, 1, 0
	ds_read_b128 v[134:137], v5
	ds_read_b128 v[130:133], v6
	v_lshl_add_u32 v6, v3, 1, 0
	v_or_b32_e32 v3, v47, v34
	v_lshlrev_b32_e32 v207, 1, v3
	v_or_b32_e32 v2, v2, v34
	v_lshlrev_b32_e32 v208, 1, v2
	v_sub_u32_e32 v2, v48, v207
	v_ashrrev_i16_e32 v2, 4, v2
	v_add_u32_sdwa v2, v74, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshlrev_b32_e32 v3, 2, v2
	ds_bpermute_b32 v7, v3, v200
	v_lshrrev_b64 v[2:3], v2, exec
	v_and_b32_e32 v8, 1, v2
	v_sub_u32_e32 v2, v4, v208
	v_ashrrev_i16_e32 v2, 4, v2
	v_add_u32_sdwa v2, v74, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshlrev_b32_e32 v3, 2, v2
	ds_bpermute_b32 v4, v3, v206
	v_lshrrev_b64 v[2:3], v2, exec
	v_and_b32_e32 v9, 1, v2
	v_sub_u32_e32 v2, v46, v34
	v_ashrrev_i32_e32 v2, 3, v2
	v_add_u32_e32 v2, v2, v74
	v_lshlrev_b32_e32 v209, 2, v2
	ds_bpermute_b32 v10, v209, v200
	v_add_u32_e32 v75, 0, v207
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v7, 1, v7
	v_lshrrev_b64 v[2:3], v2, exec
	v_cmp_eq_u32_e32 vcc, 1, v8
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v4, 1, v4
	v_and_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v76, v45, v7, vcc
	v_readfirstlane_b32 s29, v75
	v_cmp_eq_u32_e32 vcc, 1, v9
	v_add_u32_e32 v78, 0, v208
	v_lshl_add_u32 v5, v11, 1, 0
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v10
	s_mov_b32 m0, s29
	v_cndmask_b32_e32 v77, v45, v4, vcc
	v_readfirstlane_b32 s40, v78
	s_lshl_b64 s[6:7], s[36:37], 1
	v_cmp_eq_u32_e32 vcc, 1, v2
	ds_read_b128 v[154:157], v5
	ds_read_b128 v[150:153], v6
	buffer_load_dwordx4 v76, s[20:23], 0 offen lds
	s_mov_b32 m0, s40
	s_add_u32 s0, s20, s6
	v_cndmask_b32_e32 v2, v45, v3, vcc
	ds_bpermute_b32 v3, v209, v206
	v_add_u32_e32 v11, 0x4000, v75
	buffer_load_dwordx4 v77, s[20:23], 0 offen lds
	s_addc_u32 s20, s13, s7
	v_readfirstlane_b32 s14, v11
	s_and_b32 s1, s20, 0xffff
	s_or_b32 s1, s1, s19
	s_mov_b32 m0, s14
	v_bitop3_b32 v6, v37, v217, v34 bitop3:0xde
	buffer_load_dwordx4 v2, s[0:3], 0 offen lds
	v_add_u32_e32 v2, 0x4000, v78
	v_lshlrev_b32_e32 v81, 1, v6
	v_readfirstlane_b32 s13, v2
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v3
	v_cndmask_b32_e32 v2, v45, v2, vcc
	s_mov_b32 m0, s13
	v_add_u32_e32 v82, 0, v81
	buffer_load_dwordx4 v2, s[0:3], 0 offen lds
	v_bitop3_b32 v2, v36, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v79, 1, v2
	v_add_u32_e32 v80, 0, v79
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b128 v[2:5], v80
	ds_read_b128 v[18:21], v80 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[114:117], 0
	ds_read_b128 v[22:25], v82
	ds_read_b128 v[46:49], v82 offset:8192
	v_mad_u64_u32 v[250:251], s[14:15], s24, v1, v[34:35]
	v_lshl_add_u32 v253, s24, 5, v250
	s_and_b32 s13, s24, 0x3fff
	s_bitset1_b32 s13, 14
	s_and_b32 s14, s16, 0xffff
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[146:149], v[2:17]
	v_bitop3_b32 v22, v38, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v83, 1, v22
	v_add_u32_e32 v84, 0, v83
	ds_read_b128 v[22:25], v84
	ds_read_b128 v[50:53], v84 offset:8192
	s_lshl_b32 s22, s13, 16
	s_or_b32 s13, s14, s22
	s_add_u32 s0, s0, s6
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[142:145], v[2:17]
	v_bitop3_b32 v22, v39, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v85, 1, v22
	v_add_u32_e32 v86, 0, v85
	ds_read_b128 v[22:25], v86
	ds_read_b128 v[54:57], v86 offset:8192
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_movk_i32 s37, 0x50
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[138:141], v[2:17]
	v_bitop3_b32 v22, v40, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v87, 1, v22
	v_add_u32_e32 v88, 0, v87
	ds_read_b128 v[22:25], v88
	ds_read_b128 v[58:61], v88 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[134:137], v[2:17]
	v_bitop3_b32 v22, v41, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v89, 1, v22
	v_add_u32_e32 v90, 0, v89
	ds_read_b128 v[22:25], v90
	ds_read_b128 v[62:65], v90 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[130:133], v[2:17]
	v_bitop3_b32 v22, v42, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v91, 1, v22
	v_add_u32_e32 v92, 0, v91
	ds_read_b128 v[22:25], v92
	ds_read_b128 v[66:69], v92 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[154:157], v[2:17]
	v_bitop3_b32 v22, v43, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v93, 1, v22
	v_add_u32_e32 v118, 0, v93
	ds_read_b128 v[22:25], v118
	ds_read_b128 v[70:73], v118 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[150:153], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[114:117], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[46:49], v[146:149], v[18:33]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_nop 7
	s_nop 1
	v_max_f32_e32 v46, v3, v3
	v_max_f32_e32 v47, v2, v2
	v_max_f32_e32 v46, v47, v46
	v_max3_f32 v46, v46, v4, v5
	v_max3_f32 v46, v46, v6, v7
	v_max3_f32 v46, v46, v8, v9
	v_max3_f32 v46, v46, v10, v11
	v_mfma_f32_32x32x16_f16 v[18:33], v[50:53], v[142:145], v[18:33]
	v_max3_f32 v46, v46, v12, v13
	v_max3_f32 v46, v46, v14, v15
	v_max3_f32 v46, v46, v16, v17
	v_mfma_f32_32x32x16_f16 v[18:33], v[54:57], v[138:141], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[58:61], v[134:137], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[62:65], v[130:133], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[66:69], v[154:157], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[70:73], v[150:153], v[18:33]
	s_nop 7
	s_nop 3
	v_max3_f32 v46, v46, v18, v19
	v_max3_f32 v46, v46, v20, v21
	v_max3_f32 v46, v46, v22, v23
	v_max3_f32 v46, v46, v24, v25
	v_max3_f32 v46, v46, v26, v27
	v_max3_f32 v46, v46, v28, v29
	v_max3_f32 v48, v46, v30, v31
	v_lshlrev_b32_e32 v46, 1, v0
	v_and_b32_e32 v46, 0x60, v46
	v_bitop3_b32 v46, v35, v46, s33 bitop3:0x6c
	v_sub_u32_e32 v46, v46, v34
	v_ashrrev_i32_e32 v46, 3, v46
	v_add_u32_e32 v252, v46, v74
	v_lshlrev_b32_e32 v251, 2, v252
	ds_bpermute_b32 v1, v251, v250
	ds_bpermute_b32 v49, v251, v253
	v_lshrrev_b64 v[46:47], v252, exec
	v_and_b32_e32 v46, 1, v46
	v_add_u32_e32 v47, 0x8000, v75
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v1, 1, v1
	v_cmp_eq_u32_e32 vcc, 1, v46
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v46, 1, v49
	v_readfirstlane_b32 s1, v47
	v_cndmask_b32_e32 v1, v45, v1, vcc
	v_cndmask_b32_e32 v45, v45, v46, vcc
	v_add_u32_e32 v46, 0x8000, v78
	s_mov_b32 m0, s1
	v_readfirstlane_b32 s21, v46
	s_addc_u32 s1, s20, s7
	s_ashr_i32 s39, s38, 31
	buffer_load_dwordx4 v1, s[12:15], 0 offen lds
	s_mov_b32 m0, s21
	s_lshl_b64 s[20:21], s[38:39], 1
	s_add_u32 s23, s12, s20
	s_addc_u32 s24, s16, s21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v45, s[12:15], 0 offen lds
	s_or_b32 s1, s1, s19
	s_mov_b32 m0, s29
	v_add_u32_e32 v46, 0xc000, v75
	buffer_load_dwordx4 v76, s[0:3], 0 offen lds
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s12, v46
	buffer_load_dwordx4 v77, s[0:3], 0 offen lds
	s_and_b32 s0, s24, 0xffff
	s_or_b32 s1, s0, s22
	s_mov_b32 s0, s23
	s_mov_b32 m0, s12
	s_waitcnt vmcnt(4)
	s_barrier
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_add_u32_e32 v1, 0xc000, v78
	s_movk_i32 s14, 0x60
	v_readfirstlane_b32 s12, v1
	s_mov_b32 m0, s12
	v_mov_b32_e32 v1, 0xff800000
	buffer_load_dwordx4 v45, s[0:3], 0 offen lds
	v_max3_f32 v45, v48, v32, v33
	v_mov_b32_e32 v46, v45
	s_nop 1
	v_permlane32_swap_b32_e32 v45, v46
	v_max3_f32 v186, v45, v46, v1
	v_mul_f32_e32 v45, 0xbe0293ee, v186
	v_fmamk_f32 v46, v18, 0x3e0293ee, v45
	s_movk_i32 s0, 0x1ff
	v_add_u32_e32 v18, 0xff, v0
	v_cmp_gt_u32_e32 vcc, s0, v18
	s_movk_i32 s0, 0x1fe
	s_add_i32 s12, 0, 0x4000
	v_fmamk_f32 v2, v2, 0x3e0293ee, v45
	v_fmamk_f32 v3, v3, 0x3e0293ee, v45
	v_fmamk_f32 v4, v4, 0x3e0293ee, v45
	v_fmamk_f32 v5, v5, 0x3e0293ee, v45
	v_fmamk_f32 v6, v6, 0x3e0293ee, v45
	v_fmamk_f32 v7, v7, 0x3e0293ee, v45
	v_fmamk_f32 v8, v8, 0x3e0293ee, v45
	v_fmamk_f32 v9, v9, 0x3e0293ee, v45
	v_fmamk_f32 v10, v10, 0x3e0293ee, v45
	v_fmamk_f32 v11, v11, 0x3e0293ee, v45
	v_fmamk_f32 v12, v12, 0x3e0293ee, v45
	v_fmamk_f32 v13, v13, 0x3e0293ee, v45
	v_fmamk_f32 v14, v14, 0x3e0293ee, v45
	v_fmamk_f32 v15, v15, 0x3e0293ee, v45
	v_fmamk_f32 v16, v16, 0x3e0293ee, v45
	v_fmamk_f32 v17, v17, 0x3e0293ee, v45
	v_fmamk_f32 v19, v19, 0x3e0293ee, v45
	v_fmamk_f32 v20, v20, 0x3e0293ee, v45
	v_fmamk_f32 v21, v21, 0x3e0293ee, v45
	v_fmamk_f32 v22, v22, 0x3e0293ee, v45
	v_fmamk_f32 v23, v23, 0x3e0293ee, v45
	v_fmamk_f32 v24, v24, 0x3e0293ee, v45
	v_fmamk_f32 v25, v25, 0x3e0293ee, v45
	v_fmamk_f32 v26, v26, 0x3e0293ee, v45
	v_fmamk_f32 v27, v27, 0x3e0293ee, v45
	v_fmamk_f32 v28, v28, 0x3e0293ee, v45
	v_fmamk_f32 v29, v29, 0x3e0293ee, v45
	v_fmamk_f32 v30, v30, 0x3e0293ee, v45
	v_fmamk_f32 v31, v31, 0x3e0293ee, v45
	v_fmamk_f32 v32, v32, 0x3e0293ee, v45
	v_fmac_f32_e32 v45, 0x3e0293ee, v33
	v_cmp_lt_u32_e64 s[0:1], s0, v18
	v_add_u32_e32 v18, s12, v79
	v_add_u32_e32 v33, s12, v81
	v_add_u32_e32 v47, s12, v83
	v_add_u32_e32 v48, s12, v85
	v_add_u32_e32 v49, s12, v87
	v_add_u32_e32 v50, s12, v89
	v_add_u32_e32 v51, s12, v91
	v_add_u32_e32 v52, s12, v93
	ds_read_b128 v[66:69], v80 offset:16384
	ds_read_b128 v[110:113], v82 offset:16384
	ds_read_b128 v[106:109], v84 offset:16384
	ds_read_b128 v[102:105], v86 offset:16384
	ds_read_b128 v[98:101], v88 offset:16384
	ds_read_b128 v[94:97], v90 offset:16384
	ds_read_b128 v[90:93], v92 offset:16384
	ds_read_b128 v[86:89], v118 offset:16384
	ds_read_b128 v[82:85], v18 offset:8192
	ds_read_b128 v[170:173], v33 offset:8192
	ds_read_b128 v[166:169], v47 offset:8192
	ds_read_b128 v[162:165], v48 offset:8192
	ds_read_b128 v[158:161], v49 offset:8192
	ds_read_b128 v[126:129], v50 offset:8192
	ds_read_b128 v[122:125], v51 offset:8192
	ds_read_b128 v[118:121], v52 offset:8192
	s_mov_b32 s16, 0
	s_movk_i32 s15, 0x70
	s_mov_b32 s29, 0x3e0293ee
	v_fmac_f32_e32 v1, 0xbe0293ee, v186
	v_mov_b32_e32 v18, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[12:13]
	v_exp_f32_e32 v175, v2
	v_lshrrev_b32_e32 v2, 3, v0
	v_exp_f32_e32 v177, v3
	v_exp_f32_e32 v178, v6
	v_and_b32_e32 v6, 4, v2
	v_lshlrev_b32_e32 v3, 2, v0
	v_exp_f32_e32 v174, v1
	v_and_b32_e32 v1, 16, v0
	v_and_b32_e32 v2, 64, v35
	v_and_b32_e32 v3, 12, v3
	scratch_store_dword off, v6, off        ; 4-byte Folded Spill
	v_and_or_b32 v6, v44, 3, v6
	v_exp_f32_e32 v176, v4
	v_exp_f32_e32 v179, v5
	v_and_or_b32 v4, v35, 32, v3
	v_or_b32_e32 v5, v2, v1
	v_lshlrev_b32_e32 v218, 7, v6
	s_add_u32 s0, s30, s34
	v_or3_b32 v222, v5, v4, v218
	v_bitop3_b32 v5, v3, v35, 32 bitop3:0x72
	s_addc_u32 s1, s31, s35
	v_or3_b32 v221, v1, v5, v2
	v_bitop3_b32 v2, v4, v35, 64 bitop3:0x72
	s_mul_i32 s13, s36, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_exp_f32_e32 v181, v7
	v_exp_f32_e32 v180, v8
	v_exp_f32_e32 v183, v9
	v_exp_f32_e32 v182, v10
	v_exp_f32_e32 v185, v11
	v_exp_f32_e32 v184, v12
	v_exp_f32_e32 v187, v13
	v_exp_f32_e32 v188, v14
	v_exp_f32_e32 v190, v15
	v_exp_f32_e32 v189, v16
	v_exp_f32_e32 v192, v17
	v_exp_f32_e32 v191, v46
	v_exp_f32_e32 v194, v19
	v_exp_f32_e32 v193, v20
	v_exp_f32_e32 v196, v21
	v_exp_f32_e32 v195, v22
	v_exp_f32_e32 v232, v23
	v_exp_f32_e32 v197, v24
	v_exp_f32_e32 v234, v25
	v_exp_f32_e32 v233, v26
	v_exp_f32_e32 v235, v27
	v_exp_f32_e32 v236, v28
	v_exp_f32_e32 v238, v29
	v_exp_f32_e32 v237, v30
	v_exp_f32_e32 v240, v31
	v_exp_f32_e32 v239, v32
	v_exp_f32_e32 v241, v45
	v_or_b32_e32 v220, v2, v1
	v_bitop3_b32 v2, v35, v3, s14 bitop3:0x4e
	s_mul_hi_i32 s12, s36, 6
	s_add_u32 s0, s13, s0
	v_or_b32_e32 v219, v2, v1
	v_lshlrev_b32_e32 v1, 3, v203
	s_addc_u32 s1, s12, s1
	v_bitop3_b32 v33, v36, v34, s15 bitop3:0x36
	v_bitop3_b32 v2, v1, v36, s33 bitop3:0x6c
	v_bitop3_b32 v3, v1, v37, s33 bitop3:0x6c
	v_bitop3_b32 v4, v1, v38, s33 bitop3:0x6c
	v_bitop3_b32 v5, v1, v39, s33 bitop3:0x6c
	v_bitop3_b32 v6, v1, v40, s33 bitop3:0x6c
	v_bitop3_b32 v7, v1, v41, s33 bitop3:0x6c
	v_bitop3_b32 v8, v1, v42, s33 bitop3:0x6c
	v_bitop3_b32 v1, v1, v43, s33 bitop3:0x6c
	s_add_u32 s4, s4, s0
	v_and_b32_e32 v223, 64, v204
	v_bitop3_b32 v216, v44, v34, 8 bitop3:0x6c
	v_bitop3_b32 v215, v36, v34, 16 bitop3:0x36
	v_bitop3_b32 v214, v36, v34, 32 bitop3:0x36
	v_bitop3_b32 v213, v36, v34, 48 bitop3:0x36
	v_bitop3_b32 v212, v36, v34, 64 bitop3:0x36
	v_bitop3_b32 v211, v36, v34, s37 bitop3:0x36
	v_bitop3_b32 v210, v36, v34, s14 bitop3:0x36
	scratch_store_dword off, v33, off offset:4 ; 4-byte Folded Spill
	s_addc_u32 s31, s5, s1
	s_add_i32 s5, 0, 0x8000
	s_add_i32 s30, 0, 0xc000
	v_mov_b32_e32 v242, 1.0
	s_movk_i32 s33, 0xffc0
	v_lshlrev_b32_e32 v224, 1, v2
	v_lshlrev_b32_e32 v225, 1, v3
	v_lshlrev_b32_e32 v226, 1, v4
	v_lshlrev_b32_e32 v227, 1, v5
	v_lshlrev_b32_e32 v228, 1, v6
	v_lshlrev_b32_e32 v229, 1, v7
	v_lshlrev_b32_e32 v230, 1, v8
	v_lshlrev_b32_e32 v231, 1, v1
	s_mov_b32 s13, 0
	v_mov_b32_e32 v19, v18
	v_mov_b32_e32 v20, v18
	v_mov_b32_e32 v21, v18
	v_mov_b32_e32 v22, v18
	v_mov_b32_e32 v23, v18
	v_mov_b32_e32 v24, v18
	v_mov_b32_e32 v25, v18
	v_mov_b32_e32 v26, v18
	v_mov_b32_e32 v27, v18
	v_mov_b32_e32 v28, v18
	v_mov_b32_e32 v29, v18
	v_mov_b32_e32 v30, v18
	v_mov_b32_e32 v31, v18
	v_mov_b32_e32 v32, v18
	v_mov_b32_e32 v33, v18
	v_mov_b32_e32 v2, v18
	v_mov_b32_e32 v3, v18
	v_mov_b32_e32 v4, v18
	v_mov_b32_e32 v5, v18
	v_mov_b32_e32 v6, v18
	v_mov_b32_e32 v7, v18
	v_mov_b32_e32 v8, v18
	v_mov_b32_e32 v9, v18
	v_mov_b32_e32 v10, v18
	v_mov_b32_e32 v11, v18
	v_mov_b32_e32 v12, v18
	v_mov_b32_e32 v13, v18
	v_mov_b32_e32 v14, v18
	v_mov_b32_e32 v15, v18
	v_mov_b32_e32 v16, v18
	v_mov_b32_e32 v17, v18
	v_mov_b32_e32 v34, v18
	v_mov_b32_e32 v35, v18
	v_mov_b32_e32 v36, v18
	v_mov_b32_e32 v37, v18
	v_mov_b32_e32 v38, v18
	v_mov_b32_e32 v39, v18
	v_mov_b32_e32 v40, v18
	v_mov_b32_e32 v41, v18
	v_mov_b32_e32 v42, v18
	v_mov_b32_e32 v43, v18
	v_mov_b32_e32 v44, v18
	v_mov_b32_e32 v45, v18
	v_mov_b32_e32 v46, v18
	v_mov_b32_e32 v47, v18
	v_mov_b32_e32 v48, v18
	v_mov_b32_e32 v49, v18
	v_mov_b32_e32 v50, v18
	v_mov_b32_e32 v51, v18
	v_mov_b32_e32 v52, v18
	v_mov_b32_e32 v53, v18
	v_mov_b32_e32 v54, v18
	v_mov_b32_e32 v55, v18
	v_mov_b32_e32 v56, v18
	v_mov_b32_e32 v57, v18
	v_mov_b32_e32 v58, v18
	v_mov_b32_e32 v59, v18
	v_mov_b32_e32 v60, v18
	v_mov_b32_e32 v61, v18
	v_mov_b32_e32 v62, v18
	v_mov_b32_e32 v63, v18
	v_mov_b32_e32 v64, v18
	v_mov_b32_e32 v65, v18
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[114:117], 0
	s_mov_b32 s14, s5
	s_mov_b32 s5, s30
	v_mov_b32_e32 v1, v242
	v_mov_b32_e32 v244, v186
	s_mov_b32 s36, s16
	s_setprio 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[146:149], v[66:81]
	v_pk_mul_f32 v[18:19], v[18:19], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[174:175] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[142:145], v[66:81]
	v_pk_mul_f32 v[32:33], v[32:33], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[174:175] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[138:141], v[66:81]
	v_add_f32_e32 v102, v175, v177
	v_add_f32_e32 v102, v102, v176
	v_add_f32_e32 v102, v102, v179
	v_add_f32_e32 v102, v102, v178
	v_add_f32_e32 v102, v102, v181
	v_add_f32_e32 v102, v102, v180
	v_pk_mul_f32 v[62:63], v[62:63], v[174:175] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[98:101], v[134:137], v[66:81]
	v_add_f32_e32 v98, v102, v183
	v_add_f32_e32 v98, v98, v182
	v_add_f32_e32 v98, v98, v185
	v_add_f32_e32 v98, v98, v184
	v_add_f32_e32 v98, v98, v187
	v_add_f32_e32 v98, v98, v188
	v_add_f32_e32 v98, v98, v190
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[130:133], v[66:81]
	v_add_f32_e32 v94, v98, v189
	v_add_f32_e32 v94, v94, v192
	v_add_f32_e32 v94, v94, v191
	v_add_f32_e32 v94, v94, v194
	v_add_f32_e32 v94, v94, v193
	v_add_f32_e32 v94, v94, v196
	v_add_f32_e32 v94, v94, v195
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[154:157], v[66:81]
	v_add_f32_e32 v90, v94, v232
	v_add_f32_e32 v90, v90, v197
	v_add_f32_e32 v90, v90, v234
	v_add_f32_e32 v90, v90, v233
	v_add_f32_e32 v90, v90, v235
	v_add_f32_e32 v90, v90, v236
	v_add_f32_e32 v90, v90, v238
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[150:153], v[66:81]
	v_add_f32_e32 v86, v90, v237
	v_add_f32_e32 v86, v86, v240
	v_add_f32_e32 v86, v86, v239
	v_add_f32_e32 v98, v86, v241
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[114:117], 0
	v_add_f32_e32 v242, v98, v99
	v_pk_mul_f32 v[64:65], v[64:65], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[174:175] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[146:149], v[82:97]
	v_pk_mul_f32 v[44:45], v[44:45], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[174:175] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[142:145], v[82:97]
	v_pk_mul_f32 v[10:11], v[10:11], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[174:175] op_sel_hi:[1,0]
	v_fmac_f32_e32 v242, v1, v174
	v_cvt_pk_f16_f32 v110, v175, v177
	v_cvt_pk_f16_f32 v111, v176, v179
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[138:141], v[82:97]
	v_cvt_pk_f16_f32 v112, v178, v181
	v_cvt_pk_f16_f32 v113, v180, v183
	v_cvt_pk_f16_f32 v106, v182, v185
	v_cvt_pk_f16_f32 v107, v184, v187
	v_cvt_pk_f16_f32 v108, v188, v190
	v_cvt_pk_f16_f32 v109, v189, v192
	v_cvt_pk_f16_f32 v102, v191, v194
	v_mfma_f32_32x32x16_f16 v[82:97], v[158:161], v[134:137], v[82:97]
	v_cvt_pk_f16_f32 v103, v193, v196
	v_cvt_pk_f16_f32 v104, v195, v232
	v_cvt_pk_f16_f32 v105, v197, v234
	v_cvt_pk_f16_f32 v98, v233, v235
	v_cvt_pk_f16_f32 v99, v236, v238
	v_cvt_pk_f16_f32 v100, v237, v240
	v_cvt_pk_f16_f32 v101, v239, v241
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[130:133], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[154:157], v[82:97]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[150:153], v[82:97]
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s23, s20
	s_addc_u32 s35, s24, s21
	s_add_i32 s0, s13, 1
	s_cmp_lt_i32 s0, 2
	s_cselect_b32 s37, s0, 0
	ds_bpermute_b32 v1, v209, v200
	s_lshl_b32 s0, s37, 14
	ds_bpermute_b32 v118, v209, v206
	s_add_i32 s16, s0, 0
	v_add_u32_e32 v243, s16, v207
	v_add_u32_e32 v245, s16, v208
	s_and_b32 s0, s31, 0xffff
	v_readfirstlane_b32 s15, v243
	s_or_b32 s1, s0, s19
	s_mov_b32 s0, s4
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v1, 1, v1
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v245
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v118
	s_mov_b32 m0, s15
	v_lshlrev_b32_e32 v118, 1, v218
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	v_lshl_add_u32 v1, v222, 1, s14
	ds_read_b64_tr_b16 v[232:233], v1
	ds_read_b64_tr_b16 v[234:235], v1 offset:2048
	ds_read_b64_tr_b16 v[236:237], v1 offset:4096
	ds_read_b64_tr_b16 v[238:239], v1 offset:6144
	ds_read_b64_tr_b16 v[246:247], v1 offset:8192
	ds_read_b64_tr_b16 v[248:249], v1 offset:10240
	ds_read_b64_tr_b16 v[194:195], v1 offset:12288
	ds_read_b64_tr_b16 v[196:197], v1 offset:14336
	v_lshlrev_b32_e32 v1, 1, v221
	v_add3_u32 v1, s14, v1, v118
	ds_read_b64_tr_b16 v[190:191], v1
	ds_read_b64_tr_b16 v[192:193], v1 offset:2048
	ds_read_b64_tr_b16 v[186:187], v1 offset:4096
	ds_read_b64_tr_b16 v[188:189], v1 offset:6144
	ds_read_b64_tr_b16 v[182:183], v1 offset:8192
	ds_read_b64_tr_b16 v[184:185], v1 offset:10240
	ds_read_b64_tr_b16 v[178:179], v1 offset:12288
	ds_read_b64_tr_b16 v[180:181], v1 offset:14336
	v_lshlrev_b32_e32 v1, 1, v220
	v_add3_u32 v1, s14, v1, v118
	ds_read_b64_tr_b16 v[174:175], v1
	ds_read_b64_tr_b16 v[176:177], v1 offset:2048
	ds_read_b64_tr_b16 v[170:171], v1 offset:4096
	ds_read_b64_tr_b16 v[172:173], v1 offset:6144
	ds_read_b64_tr_b16 v[166:167], v1 offset:8192
	ds_read_b64_tr_b16 v[168:169], v1 offset:10240
	ds_read_b64_tr_b16 v[162:163], v1 offset:12288
	ds_read_b64_tr_b16 v[164:165], v1 offset:14336
	v_lshlrev_b32_e32 v1, 1, v219
	v_add3_u32 v1, s14, v1, v118
	ds_read_b64_tr_b16 v[158:159], v1
	ds_read_b64_tr_b16 v[160:161], v1 offset:2048
	ds_read_b64_tr_b16 v[126:127], v1 offset:4096
	ds_read_b64_tr_b16 v[128:129], v1 offset:6144
	ds_read_b64_tr_b16 v[122:123], v1 offset:8192
	ds_read_b64_tr_b16 v[124:125], v1 offset:10240
	ds_read_b64_tr_b16 v[118:119], v1 offset:12288
	ds_read_b64_tr_b16 v[120:121], v1 offset:14336
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[18:33], v[232:235], v[110:113], v[18:33]
	s_setprio 0
	v_mfma_f32_32x32x16_f16 v[50:65], v[190:193], v[110:113], v[50:65]
	v_max_f32_e32 v1, v67, v67
	v_max_f32_e32 v198, v66, v66
	v_max_f32_e32 v1, v198, v1
	v_max3_f32 v1, v1, v68, v69
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_max3_f32 v1, v1, v74, v75
	v_mfma_f32_32x32x16_f16 v[34:49], v[174:177], v[110:113], v[34:49]
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[110:113], v[2:17]
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_mfma_f32_32x32x16_f16 v[18:33], v[236:239], v[106:109], v[18:33]
	v_mfma_f32_32x32x16_f16 v[50:65], v[186:189], v[106:109], v[50:65]
	v_mov_b32_e32 v186, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v186
	v_max3_f32 v186, v244, v1, v186
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[106:109], v[34:49]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[126:129], v[106:109], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[246:249], v[102:105], v[18:33]
	v_mul_f32_e32 v249, 0x3e0293ee, v186
	v_fma_f32 v1, v66, s29, -v249
	v_fma_f32 v66, v67, s29, -v249
	v_fma_f32 v67, v68, s29, -v249
	v_fma_f32 v68, v69, s29, -v249
	v_fma_f32 v69, v70, s29, -v249
	v_fma_f32 v70, v71, s29, -v249
	v_mfma_f32_32x32x16_f16 v[50:65], v[182:185], v[102:105], v[50:65]
	v_fma_f32 v71, v72, s29, -v249
	v_fma_f32 v72, v73, s29, -v249
	v_fma_f32 v73, v74, s29, -v249
	v_fma_f32 v74, v75, s29, -v249
	v_fma_f32 v75, v76, s29, -v249
	v_fma_f32 v76, v77, s29, -v249
	v_fma_f32 v77, v78, s29, -v249
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[102:105], v[34:49]
	v_fma_f32 v78, v79, s29, -v249
	v_fma_f32 v79, v80, s29, -v249
	v_fma_f32 v80, v81, s29, -v249
	v_fma_f32 v81, v82, s29, -v249
	v_fma_f32 v82, v83, s29, -v249
	v_fma_f32 v83, v84, s29, -v249
	v_fma_f32 v84, v85, s29, -v249
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[122:125], v[102:105], v[2:17]
	v_fma_f32 v85, v86, s29, -v249
	v_fma_f32 v86, v87, s29, -v249
	v_fma_f32 v87, v88, s29, -v249
	v_fma_f32 v88, v89, s29, -v249
	v_fma_f32 v89, v90, s29, -v249
	v_fma_f32 v90, v91, s29, -v249
	v_fma_f32 v91, v92, s29, -v249
	v_mfma_f32_32x32x16_f16 v[18:33], v[194:197], v[98:101], v[18:33]
	v_fma_f32 v92, v93, s29, -v249
	v_fma_f32 v93, v94, s29, -v249
	v_fma_f32 v94, v95, s29, -v249
	v_fma_f32 v95, v96, s29, -v249
	v_fma_f32 v96, v97, s29, -v249
	v_exp_f32_e32 v175, v1
	v_fma_f32 v1, v244, s29, -v249
	v_mfma_f32_32x32x16_f16 v[50:65], v[178:181], v[98:101], v[50:65]
	v_exp_f32_e32 v177, v66
	v_exp_f32_e32 v176, v67
	v_exp_f32_e32 v179, v68
	v_exp_f32_e32 v178, v69
	v_exp_f32_e32 v181, v70
	v_exp_f32_e32 v180, v71
	v_exp_f32_e32 v183, v72
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[98:101], v[34:49]
	v_exp_f32_e32 v182, v73
	v_exp_f32_e32 v185, v74
	v_exp_f32_e32 v184, v75
	v_exp_f32_e32 v187, v76
	v_exp_f32_e32 v188, v77
	v_exp_f32_e32 v190, v78
	v_exp_f32_e32 v189, v79
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[118:121], v[98:101], v[2:17]
	v_exp_f32_e32 v192, v80
	v_exp_f32_e32 v191, v81
	v_exp_f32_e32 v194, v82
	v_exp_f32_e32 v193, v83
	v_exp_f32_e32 v196, v84
	v_exp_f32_e32 v195, v85
	v_exp_f32_e32 v232, v86
	v_exp_f32_e32 v197, v87
	v_exp_f32_e32 v234, v88
	v_exp_f32_e32 v233, v89
	v_exp_f32_e32 v235, v90
	v_exp_f32_e32 v236, v91
	v_exp_f32_e32 v238, v92
	v_exp_f32_e32 v237, v93
	v_exp_f32_e32 v240, v94
	v_exp_f32_e32 v239, v95
	v_exp_f32_e32 v241, v96
	v_exp_f32_e32 v174, v1
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s0, s13, 14
	ds_bpermute_b32 v67, v251, v250
	s_add_i32 s34, s0, 0
	ds_bpermute_b32 v68, v251, v253
	s_add_i32 s30, s34, 0x8000
	v_add_u32_e32 v1, s30, v207
	s_and_b32 s0, s35, 0xffff
	v_add_u32_e32 v66, s30, v208
	s_or_b32 s13, s0, s22
	v_readfirstlane_b32 s0, v1
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v67, 1, v67
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v66
	buffer_load_dwordx4 v67, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v68
	s_mov_b32 m0, s0
	v_lshlrev_b32_e32 v255, 1, v217
	buffer_load_dwordx4 v1, s[12:15], 0 offen lds
	v_add3_u32 v1, s36, v224, v255
	v_add3_u32 v70, s36, v225, v255
	v_add3_u32 v71, s36, v226, v255
	v_add3_u32 v72, s36, v227, v255
	v_add3_u32 v73, s36, v228, v255
	v_add3_u32 v74, s36, v229, v255
	v_add3_u32 v75, s36, v230, v255
	v_add3_u32 v76, s36, v231, v255
	ds_read_b128 v[66:69], v1
	ds_read_b128 v[82:85], v1 offset:8192
	ds_read_b128 v[110:113], v70
	ds_read_b128 v[170:173], v70 offset:8192
	ds_read_b128 v[106:109], v71
	ds_read_b128 v[166:169], v71 offset:8192
	ds_read_b128 v[102:105], v72
	ds_read_b128 v[162:165], v72 offset:8192
	ds_read_b128 v[98:101], v73
	ds_read_b128 v[158:161], v73 offset:8192
	ds_read_b128 v[94:97], v74
	ds_read_b128 v[126:129], v74 offset:8192
	ds_read_b128 v[90:93], v75
	ds_read_b128 v[122:125], v75 offset:8192
	ds_read_b128 v[86:89], v76
	ds_read_b128 v[118:121], v76 offset:8192
	; sched_barrier mask(0x00000000)
	s_add_u32 s23, s23, s20
	s_addc_u32 s24, s24, s21
	s_add_u32 s4, s4, s6
	s_addc_u32 s31, s31, s7
	s_add_i32 s33, s33, 64
	s_cmpk_lt_u32 s33, 0x1f00
	s_mov_b32 s13, s37
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[114:117], 0
	v_and_b32_e32 v1, 0x100, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v1
	v_and_b32_e32 v1, 0xa0, v204
	v_or3_b32 v1, v1, v203, v223
	scratch_store_dword off, v1, off offset:8 ; 4-byte Folded Spill
	v_add_f32_e32 v1, v175, v177
	v_add_f32_e32 v1, v1, v176
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[146:149], v[66:81]
	v_add_f32_e32 v1, v1, v179
	v_add_f32_e32 v1, v1, v178
	v_add_f32_e32 v1, v1, v181
	v_add_f32_e32 v1, v1, v180
	v_add_f32_e32 v1, v1, v183
	v_add_f32_e32 v1, v1, v182
	v_add_f32_e32 v1, v1, v185
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[142:145], v[66:81]
	v_add_f32_e32 v1, v1, v184
	v_add_f32_e32 v1, v1, v187
	v_add_f32_e32 v1, v1, v188
	v_add_f32_e32 v1, v1, v190
	v_add_f32_e32 v1, v1, v189
	v_add_f32_e32 v1, v1, v192
	v_add_f32_e32 v1, v1, v191
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[138:141], v[66:81]
	v_add_f32_e32 v1, v1, v194
	v_add_f32_e32 v1, v1, v193
	v_add_f32_e32 v1, v1, v196
	v_add_f32_e32 v1, v1, v195
	v_add_f32_e32 v1, v1, v232
	v_add_f32_e32 v1, v1, v197
	v_add_f32_e32 v1, v1, v234
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[98:101], v[134:137], v[66:81]
	v_add_f32_e32 v1, v1, v233
	v_add_f32_e32 v1, v1, v235
	v_add_f32_e32 v1, v1, v236
	v_add_f32_e32 v1, v1, v238
	v_add_f32_e32 v1, v1, v237
	v_add_f32_e32 v1, v1, v240
	v_add_f32_e32 v1, v1, v239
	v_mfma_f32_32x32x16_f16 v[98:113], v[82:85], v[114:117], 0
	v_cvt_pk_f16_f32 v199, v184, v187
	v_lshlrev_b32_e32 v187, 1, v222
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_pk_mul_f32 v[84:85], v[20:21], v[174:175] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[146:149], v[98:113]
	v_cvt_pk_f16_f32 v170, v175, v177
	v_cvt_pk_f16_f32 v171, v176, v179
	v_cvt_pk_f16_f32 v172, v178, v181
	v_cvt_pk_f16_f32 v173, v180, v183
	v_pk_mul_f32 v[82:83], v[18:19], v[174:175] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v198, v182, v185
	v_add_lshl_u32 v18, v221, v218, 1
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[142:145], v[98:113]
	v_add_f32_e32 v166, v1, v241
	v_add_u32_e32 v1, s5, v187
	v_mov_b32_e32 v246, v18
	v_cvt_pk_f16_f32 v200, v188, v190
	v_cvt_pk_f16_f32 v201, v189, v192
	v_pk_mul_f32 v[64:65], v[64:65], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[174:175] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[138:141], v[98:113]
	v_pk_mul_f32 v[60:61], v[60:61], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[174:175] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v202, v191, v194
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[130:133], v[66:81]
	v_pk_mul_f32 v[96:97], v[32:33], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[30:31], v[174:175] op_sel_hi:[1,0]
	v_add_u32_e32 v30, s5, v18
	v_cvt_pk_f16_f32 v203, v193, v196
	v_cvt_pk_f16_f32 v204, v195, v232
	v_cvt_pk_f16_f32 v205, v197, v234
	v_add_lshl_u32 v248, v220, v218, 1
	v_mfma_f32_32x32x16_f16 v[98:113], v[158:161], v[134:137], v[98:113]
	v_cvt_pk_f16_f32 v206, v233, v235
	v_cvt_pk_f16_f32 v207, v236, v238
	v_cvt_pk_f16_f32 v208, v237, v240
	v_cvt_pk_f16_f32 v209, v239, v241
	v_pk_mul_f32 v[32:33], v[48:49], v[174:175] op_sel_hi:[1,0]
	v_mov_b32_e32 v167, v166
	s_nop 1
	v_permlane32_swap_b32_e32 v166, v167
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[154:157], v[66:81]
	v_pk_mul_f32 v[92:93], v[28:29], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[26:27], v[174:175] op_sel_hi:[1,0]
	s_mul_i32 s2, s18, 0xc0000
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s8, s2
	s_addc_u32 s6, s9, s3
	v_mfma_f32_32x32x16_f16 v[98:113], v[126:129], v[130:133], v[98:113]
	ds_read_b64_tr_b16 v[126:127], v1
	ds_read_b64_tr_b16 v[128:129], v1 offset:2048
	ds_read_b64_tr_b16 v[176:177], v1 offset:4096
	ds_read_b64_tr_b16 v[178:179], v1 offset:6144
	s_lshl_b32 s2, s17, 14
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s4, s2
	s_addc_u32 s6, s6, s3
	s_ashr_i32 s29, s28, 31
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[150:153], v[66:81]
	v_pk_mul_f32 v[88:89], v[24:25], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[22:23], v[174:175] op_sel_hi:[1,0]
	s_lshl_b64 s[2:3], s[28:29], 2
	s_add_u32 s4, s4, s2
	s_addc_u32 s19, s6, s3
	s_add_i32 s2, s28, 0xffffc100
	s_add_u32 s12, s12, s20
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[170:173], v[82:97]
	s_nop 2
	v_max_f32_e32 v159, v66, v66
	s_addc_u32 s6, s35, s21
	s_and_b32 s6, s6, 0xffff
	v_pk_mul_f32 v[16:17], v[16:17], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[174:175] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[98:113], v[122:125], v[154:157], v[98:113]
	ds_read_b64_tr_b16 v[122:123], v1 offset:8192
	ds_read_b64_tr_b16 v[124:125], v1 offset:10240
	ds_read_b64_tr_b16 v[180:181], v1 offset:12288
	ds_read_b64_tr_b16 v[182:183], v1 offset:14336
	v_or_b32_e32 v1, v221, v218
	v_lshlrev_b32_e32 v158, 1, v1
	v_mov_b32_e32 v244, v158
	v_add_u32_e32 v1, s5, v158
	scratch_store_dword off, v244, off offset:20 ; 4-byte Folded Spill
	scratch_store_dword off, v246, off offset:24 ; 4-byte Folded Spill
	ds_read_b64_tr_b16 v[18:19], v1
	ds_read_b64_tr_b16 v[20:21], v30 offset:2048
	ds_read_b64_tr_b16 v[22:23], v30 offset:4096
	ds_read_b64_tr_b16 v[24:25], v30 offset:6144
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[82:97], v[176:179], v[198:201], v[82:97]
	v_or_b32_e32 v1, v220, v218
	v_lshlrev_b32_e32 v254, 1, v1
	v_add_u32_e32 v1, s5, v254
	v_max_f32_e32 v158, v67, v67
	v_pk_mul_f32 v[8:9], v[8:9], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[174:175] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[18:21], v[170:173], v[50:65]
	v_add_lshl_u32 v20, v219, v218, 1
	v_add_u32_e32 v19, s5, v20
	v_pk_mul_f32 v[2:3], v[2:3], v[174:175] op_sel_hi:[1,0]
	s_add_i32 s3, s16, 0x8000
	s_or_b32 s13, s6, s22
	v_add_u32_e32 v184, 0x8000, v243
	s_cmp_lt_i32 s2, 1
	v_mfma_f32_32x32x16_f16 v[98:113], v[118:121], v[150:153], v[98:113]
	ds_read_b64_tr_b16 v[26:27], v30 offset:8192
	ds_read_b64_tr_b16 v[28:29], v30 offset:10240
	ds_read_b64_tr_b16 v[118:119], v30 offset:12288
	ds_read_b64_tr_b16 v[120:121], v30 offset:14336
	v_add_u32_e32 v30, s5, v248
	v_add_u32_e32 v185, 0x8000, v245
	v_readfirstlane_b32 s2, v184
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_mov_b32 m0, s2
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[202:205], v[82:97]
	ds_read_b64_tr_b16 v[122:123], v1
	ds_read_b64_tr_b16 v[124:125], v30 offset:2048
	ds_read_b64_tr_b16 v[126:127], v30 offset:4096
	ds_read_b64_tr_b16 v[128:129], v30 offset:6144
	v_or_b32_e32 v1, v219, v218
	v_lshlrev_b32_e32 v1, 1, v1
	v_add_u32_e32 v18, s5, v1
	v_readfirstlane_b32 s2, v185
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[126:129], off offset:28 ; 16-byte Folded Spill
	ds_read_b64_tr_b16 v[238:239], v30 offset:8192
	ds_read_b64_tr_b16 v[240:241], v30 offset:10240
	ds_read_b64_tr_b16 v[234:235], v30 offset:12288
	ds_read_b64_tr_b16 v[236:237], v30 offset:14336
	v_mfma_f32_32x32x16_f16 v[50:65], v[22:25], v[198:201], v[50:65]
	scratch_store_dword off, v0, off offset:12 ; 4-byte Folded Spill
	ds_read_b64_tr_b16 v[230:231], v18
	ds_read_b64_tr_b16 v[232:233], v19 offset:2048
	ds_read_b64_tr_b16 v[226:227], v19 offset:4096
	ds_read_b64_tr_b16 v[228:229], v19 offset:6144
	ds_read_b64_tr_b16 v[222:223], v19 offset:8192
	ds_read_b64_tr_b16 v[224:225], v19 offset:10240
	ds_read_b64_tr_b16 v[218:219], v19 offset:12288
	ds_read_b64_tr_b16 v[220:221], v19 offset:14336
	v_pk_mul_f32 v[18:19], v[34:35], v[174:175] op_sel_hi:[1,0]
	v_lshlrev_b32_e32 v34, 1, v216
	v_pk_mul_f32 v[22:23], v[38:39], v[174:175] op_sel_hi:[1,0]
	v_add3_u32 v38, s16, v34, v255
	v_mov_b32_e32 v0, v20
	v_pk_mul_f32 v[20:21], v[36:37], v[174:175] op_sel_hi:[1,0]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[34:37], v38
	v_mfma_f32_32x32x16_f16 v[50:65], v[26:29], v[202:205], v[50:65]
	v_pk_mul_f32 v[30:31], v[46:47], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[44:45], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[42:43], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[40:41], v[174:175] op_sel_hi:[1,0]
	v_lshlrev_b32_e32 v39, 1, v215
	v_add3_u32 v126, s16, v39, v255
	v_add_f32_e32 v127, v166, v167
	v_mfma_f32_32x32x16_f16 v[50:65], v[118:121], v[206:209], v[50:65]
	v_fmac_f32_e32 v127, v242, v174
	v_lshrrev_b64 v[174:175], v252, exec
	v_and_b32_e32 v174, 1, v174
	v_cmp_eq_u32_e32 vcc, 1, v174
	v_mfma_f32_32x32x16_f16 v[18:33], v[122:125], v[170:173], v[18:33]
	ds_read_b128 v[118:121], v38 offset:8192
	ds_read_b128 v[122:125], v126
	scratch_store_dword off, v127, off offset:16 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v127, 1, v214
	v_add3_u32 v160, s16, v127, v255
	ds_read_b128 v[176:179], v126 offset:8192
	ds_read_b128 v[126:129], v160
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[34:49], v[34:37], v[114:117], 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[146:149], v[34:49]
	v_max_f32_e32 v122, v159, v158
	v_max3_f32 v122, v122, v68, v69
	v_max3_f32 v158, v122, v70, v71
	v_lshlrev_b32_e32 v122, 1, v213
	v_add3_u32 v159, s16, v122, v255
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[142:145], v[34:49]
	v_lshlrev_b32_e32 v126, 1, v212
	v_add3_u32 v126, s16, v126, v255
	v_lshlrev_b32_e32 v127, 1, v211
	v_mfma_f32_32x32x16_f16 v[82:97], v[180:183], v[206:209], v[82:97]
	ds_read_b128 v[180:183], v160 offset:8192
	ds_read_b128 v[122:125], v159
	ds_read_b128 v[188:191], v159 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[138:141], v[34:49]
	ds_read_b128 v[122:125], v126
	v_add3_u32 v159, s16, v127, v255
	ds_read_b128 v[192:195], v126 offset:8192
	ds_read_b128 v[126:129], v159
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[134:137], v[34:49]
	v_max3_f32 v122, v158, v72, v73
	v_max3_f32 v122, v122, v74, v75
	v_max3_f32 v162, v122, v76, v77
	v_lshlrev_b32_e32 v122, 1, v210
	v_add3_u32 v163, s16, v122, v255
	ds_read_b128 v[158:161], v159 offset:8192
	ds_read_b128 v[122:125], v163
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[130:133], v[34:49]
	v_max3_f32 v126, v162, v78, v79
	v_max3_f32 v126, v126, v80, v81
	v_max3_f32 v166, v126, v98, v99
	scratch_load_dword v126, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v126, 1, v126
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[154:157], v[34:49]
	v_max3_f32 v122, v166, v100, v101
	v_max3_f32 v122, v122, v102, v103
	v_max3_f32 v122, v122, v104, v105
	v_max3_f32 v122, v122, v106, v107
	v_add3_u32 v167, s16, v126, v255
	ds_read_b128 v[162:165], v163 offset:8192
	ds_read_b128 v[126:129], v167
	v_max3_f32 v122, v122, v108, v109
	v_max3_f32 v122, v122, v110, v111
	v_max3_f32 v122, v122, v112, v113
	v_mov_b32_e32 v123, v122
	s_nop 1
	v_permlane32_swap_b32_e32 v122, v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[150:153], v[34:49]
	v_max3_f32 v242, v186, v122, v123
	ds_bpermute_b32 v186, v251, v250
	v_mov_b32_e32 v255, v187
	ds_read_b128 v[166:169], v167 offset:8192
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v175, 1, v186
	v_mfma_f32_32x32x16_f16 v[114:129], v[118:121], v[114:117], 0
	v_bfrev_b32_e32 v186, 1
	v_mfma_f32_32x32x16_f16 v[114:129], v[176:179], v[146:149], v[114:129]
	ds_bpermute_b32 v146, v251, v253
	v_cndmask_b32_e32 v147, v186, v175, vcc
	buffer_load_dwordx4 v147, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
	s_mov_b32 s2, 0x3e0293ee
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v146, 1, v146
	v_mfma_f32_32x32x16_f16 v[114:129], v[180:183], v[142:145], v[114:129]
	v_cndmask_b32_e32 v142, v186, v146, vcc
	buffer_load_dwordx4 v142, s[12:15], 0 offen lds
	s_waitcnt vmcnt(2)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[114:129], v[188:191], v[138:141], v[114:129]
	v_add_u32_e32 v138, s34, v255
	v_add_u32_e32 v139, s30, v255
	ds_read_b64_tr_b16 v[214:215], v138 offset:32768
	ds_read_b64_tr_b16 v[216:217], v139 offset:2048
	ds_read_b64_tr_b16 v[210:211], v139 offset:4096
	ds_read_b64_tr_b16 v[212:213], v139 offset:6144
	v_mfma_f32_32x32x16_f16 v[114:129], v[192:195], v[134:137], v[114:129]
	v_add_u32_e32 v134, s34, v244
	v_add_u32_e32 v135, s30, v246
	ds_read_b64_tr_b16 v[194:195], v139 offset:8192
	ds_read_b64_tr_b16 v[196:197], v139 offset:10240
	ds_read_b64_tr_b16 v[250:251], v139 offset:12288
	ds_read_b64_tr_b16 v[252:253], v139 offset:14336
	ds_read_b64_tr_b16 v[244:245], v134 offset:32768
	ds_read_b64_tr_b16 v[246:247], v135 offset:2048
	ds_read_b64_tr_b16 v[190:191], v135 offset:4096
	ds_read_b64_tr_b16 v[192:193], v135 offset:6144
	ds_read_b64_tr_b16 v[186:187], v135 offset:8192
	ds_read_b64_tr_b16 v[188:189], v135 offset:10240
	ds_read_b64_tr_b16 v[182:183], v135 offset:12288
	ds_read_b64_tr_b16 v[184:185], v135 offset:14336
	v_mfma_f32_32x32x16_f16 v[114:129], v[158:161], v[130:133], v[114:129]
	v_add_u32_e32 v130, s34, v254
	v_add_u32_e32 v131, s30, v248
	ds_read_b64_tr_b16 v[178:179], v130 offset:32768
	ds_read_b64_tr_b16 v[180:181], v131 offset:2048
	ds_read_b64_tr_b16 v[174:175], v131 offset:4096
	ds_read_b64_tr_b16 v[176:177], v131 offset:6144
	v_add_u32_e32 v130, s34, v1
	v_add_u32_e32 v132, s30, v0
	v_add_u32_e32 v1, s16, v1
	v_mfma_f32_32x32x16_f16 v[114:129], v[162:165], v[154:157], v[114:129]
	ds_read_b64_tr_b16 v[154:155], v131 offset:8192
	ds_read_b64_tr_b16 v[156:157], v131 offset:10240
	ds_read_b64_tr_b16 v[146:147], v131 offset:12288
	ds_read_b64_tr_b16 v[148:149], v131 offset:14336
	ds_read_b64_tr_b16 v[142:143], v130 offset:32768
	ds_read_b64_tr_b16 v[144:145], v132 offset:2048
	ds_read_b64_tr_b16 v[138:139], v132 offset:4096
	ds_read_b64_tr_b16 v[140:141], v132 offset:6144
	ds_read_b64_tr_b16 v[134:135], v132 offset:8192
	ds_read_b64_tr_b16 v[136:137], v132 offset:10240
	ds_read_b64_tr_b16 v[130:131], v132 offset:12288
	ds_read_b64_tr_b16 v[132:133], v132 offset:14336
	scratch_load_dwordx4 v[158:161], off, off offset:28 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[114:129], v[166:169], v[150:153], v[114:129]
	v_max_f32_e32 v150, v35, v35
	v_max_f32_e32 v151, v34, v34
	v_max_f32_e32 v150, v151, v150
	v_max3_f32 v150, v150, v36, v37
	v_max3_f32 v150, v150, v38, v39
	v_max3_f32 v150, v150, v40, v41
	v_max3_f32 v150, v150, v42, v43
	v_max3_f32 v150, v150, v44, v45
	v_max3_f32 v150, v150, v46, v47
	v_max3_f32 v150, v150, v48, v49
	s_nop 1
	v_max3_f32 v150, v150, v114, v115
	v_max3_f32 v150, v150, v116, v117
	v_max3_f32 v150, v150, v118, v119
	v_max3_f32 v150, v150, v120, v121
	v_max3_f32 v150, v150, v122, v123
	v_max3_f32 v150, v150, v124, v125
	v_max3_f32 v150, v150, v126, v127
	v_max3_f32 v150, v150, v128, v129
	v_mov_b32_e32 v151, v150
	s_nop 1
	v_permlane32_swap_b32_e32 v150, v151
	v_max3_f32 v243, v242, v150, v151
	v_pk_mul_f32 v[150:151], v[242:243], s[2:3] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[2:17], v[230:233], v[170:173], v[2:17]
	v_fma_f32 v66, v66, s2, -v150
	v_fma_f32 v67, v67, s2, -v150
	v_fma_f32 v68, v68, s2, -v150
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_fma_f32 v69, v69, s2, -v150
	v_exp_f32_e32 v68, v68
	v_fma_f32 v70, v70, s2, -v150
	v_fma_f32 v74, v74, s2, -v150
	v_exp_f32_e32 v69, v69
	v_fma_f32 v71, v71, s2, -v150
	v_fma_f32 v102, v102, s2, -v150
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v152, v74
	v_sub_f32_e32 v74, v249, v150
	v_fma_f32 v72, v72, s2, -v150
	v_exp_f32_e32 v71, v71
	v_exp_f32_e32 v168, v102
	v_exp_f32_e32 v102, v74
	v_add_f32_e32 v74, v66, v67
	v_fma_f32 v73, v73, s2, -v150
	v_exp_f32_e32 v72, v72
	v_add_f32_e32 v74, v68, v74
	v_exp_f32_e32 v73, v73
	v_add_f32_e32 v74, v69, v74
	v_fma_f32 v75, v75, s2, -v150
	v_add_f32_e32 v74, v70, v74
	v_fma_f32 v76, v76, s2, -v150
	v_exp_f32_e32 v153, v75
	v_add_f32_e32 v74, v71, v74
	v_mfma_f32_32x32x16_f16 v[18:33], v[158:161], v[198:201], v[18:33]
	v_fma_f32 v77, v77, s2, -v150
	v_exp_f32_e32 v158, v76
	v_add_f32_e32 v74, v72, v74
	v_fma_f32 v78, v78, s2, -v150
	v_exp_f32_e32 v159, v77
	v_add_f32_e32 v74, v73, v74
	v_fma_f32 v79, v79, s2, -v150
	v_exp_f32_e32 v160, v78
	v_add_f32_e32 v74, v152, v74
	v_fma_f32 v80, v80, s2, -v150
	v_exp_f32_e32 v161, v79
	v_add_f32_e32 v74, v153, v74
	v_fma_f32 v81, v81, s2, -v150
	v_exp_f32_e32 v162, v80
	v_add_f32_e32 v74, v158, v74
	v_mfma_f32_32x32x16_f16 v[2:17], v[226:229], v[198:201], v[2:17]
	v_fma_f32 v98, v98, s2, -v150
	v_exp_f32_e32 v163, v81
	v_add_f32_e32 v74, v159, v74
	v_fma_f32 v99, v99, s2, -v150
	v_exp_f32_e32 v164, v98
	v_add_f32_e32 v74, v160, v74
	v_fma_f32 v100, v100, s2, -v150
	v_exp_f32_e32 v165, v99
	v_add_f32_e32 v74, v161, v74
	v_fma_f32 v101, v101, s2, -v150
	v_exp_f32_e32 v166, v100
	v_add_f32_e32 v74, v162, v74
	v_mfma_f32_32x32x16_f16 v[18:33], v[238:241], v[202:205], v[18:33]
	v_exp_f32_e32 v167, v101
	v_add_f32_e32 v74, v163, v74
	v_fma_f32 v103, v103, s2, -v150
	v_add_f32_e32 v74, v164, v74
	v_fma_f32 v104, v104, s2, -v150
	v_exp_f32_e32 v103, v103
	v_add_f32_e32 v74, v165, v74
	v_fma_f32 v105, v105, s2, -v150
	v_exp_f32_e32 v104, v104
	v_add_f32_e32 v74, v166, v74
	v_mfma_f32_32x32x16_f16 v[2:17], v[222:225], v[202:205], v[2:17]
	v_fma_f32 v106, v106, s2, -v150
	v_exp_f32_e32 v105, v105
	v_add_f32_e32 v74, v167, v74
	v_fma_f32 v107, v107, s2, -v150
	v_exp_f32_e32 v106, v106
	v_add_f32_e32 v74, v168, v74
	v_fma_f32 v108, v108, s2, -v150
	v_exp_f32_e32 v107, v107
	v_add_f32_e32 v74, v103, v74
	v_fma_f32 v109, v109, s2, -v150
	v_exp_f32_e32 v108, v108
	v_add_f32_e32 v74, v104, v74
	v_mfma_f32_32x32x16_f16 v[18:33], v[234:237], v[206:209], v[18:33]
	v_fma_f32 v110, v110, s2, -v150
	v_exp_f32_e32 v109, v109
	v_add_f32_e32 v74, v105, v74
	v_fma_f32 v111, v111, s2, -v150
	v_exp_f32_e32 v110, v110
	v_add_f32_e32 v74, v106, v74
	v_fma_f32 v112, v112, s2, -v150
	v_exp_f32_e32 v111, v111
	v_add_f32_e32 v74, v107, v74
	v_fma_f32 v113, v113, s2, -v150
	v_exp_f32_e32 v112, v112
	v_add_f32_e32 v169, v108, v74
	v_mfma_f32_32x32x16_f16 v[2:17], v[218:221], v[206:209], v[2:17]
	v_exp_f32_e32 v113, v113
	v_cvt_pk_f16_f32 v98, v66, v67
	v_pk_mul_f32 v[66:67], v[82:83], v[102:103] op_sel_hi:[1,0]
	v_add_f32_e32 v82, v109, v169
	v_add_f32_e32 v82, v110, v82
	v_cvt_pk_f16_f32 v99, v68, v69
	v_cvt_pk_f16_f32 v100, v70, v71
	v_cvt_pk_f16_f32 v101, v72, v73
	v_pk_mul_f32 v[80:81], v[96:97], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[94:95], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[92:93], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[90:91], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[88:89], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[86:87], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[84:85], v[102:103] op_sel_hi:[1,0]
	v_add_f32_e32 v86, v111, v82
	v_add_f32_e32 v86, v112, v86
	v_mfma_f32_32x32x16_f16 v[66:81], v[214:217], v[98:101], v[66:81]
	v_cvt_pk_f16_f32 v88, v168, v103
	v_pk_mul_f32 v[64:65], v[102:103], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[102:103], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[102:103], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[102:103], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[102:103], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[102:103], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[102:103], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[102:103], v[50:51] op_sel_hi:[0,1]
	v_fma_f32 v103, v37, s2, -v151
	v_cvt_pk_f16_f32 v82, v152, v153
	v_cvt_pk_f16_f32 v83, v158, v159
	v_add_f32_e32 v91, v113, v86
	v_cvt_pk_f16_f32 v89, v104, v105
	v_cvt_pk_f16_f32 v92, v106, v107
	v_cvt_pk_f16_f32 v93, v108, v109
	v_cvt_pk_f16_f32 v94, v110, v111
	v_cvt_pk_f16_f32 v95, v112, v113
	v_fma_f32 v90, v34, s2, -v151
	v_fma_f32 v96, v35, s2, -v151
	v_fma_f32 v97, v36, s2, -v151
	v_fma_f32 v104, v38, s2, -v151
	v_fma_f32 v105, v39, s2, -v151
	v_fma_f32 v106, v40, s2, -v151
	v_fma_f32 v107, v41, s2, -v151
	v_fma_f32 v108, v42, s2, -v151
	v_fma_f32 v109, v43, s2, -v151
	v_fma_f32 v110, v44, s2, -v151
	v_fma_f32 v111, v45, s2, -v151
	v_fma_f32 v112, v46, s2, -v151
	v_fma_f32 v113, v47, s2, -v151
	v_fma_f32 v153, v48, s2, -v151
	v_fma_f32 v158, v49, s2, -v151
	v_pk_mul_f32 v[48:49], v[102:103], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[102:103], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[102:103], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[102:103], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[102:103], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[102:103], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[102:103], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[102:103], v[18:19] op_sel_hi:[0,1]
	v_exp_f32_e32 v103, v103
	v_mfma_f32_32x32x16_f16 v[50:65], v[244:247], v[98:101], v[50:65]
	v_cvt_pk_f16_f32 v84, v160, v161
	v_cvt_pk_f16_f32 v85, v162, v163
	v_pk_mul_f32 v[32:33], v[102:103], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[102:103], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[102:103], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[102:103], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[102:103], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[102:103], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[102:103], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[102:103], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_f16 v[34:49], v[178:181], v[98:101], v[34:49]
	v_cvt_pk_f16_f32 v86, v164, v165
	v_cvt_pk_f16_f32 v87, v166, v167
	v_sub_f32_e32 v2, v150, v151
	v_exp_f32_e32 v159, v90
	v_exp_f32_e32 v160, v96
	v_exp_f32_e32 v161, v97
	v_exp_f32_e32 v104, v104
	v_mfma_f32_32x32x16_f16 v[18:33], v[142:145], v[98:101], v[18:33]
	v_exp_f32_e32 v105, v105
	v_exp_f32_e32 v106, v106
	v_exp_f32_e32 v107, v107
	v_exp_f32_e32 v90, v2
	v_add_u32_e32 v2, s16, v255
	v_exp_f32_e32 v108, v108
	v_mfma_f32_32x32x16_f16 v[66:81], v[210:213], v[82:85], v[66:81]
	v_exp_f32_e32 v109, v109
	v_exp_f32_e32 v110, v110
	v_exp_f32_e32 v111, v111
	v_exp_f32_e32 v112, v112
	v_exp_f32_e32 v100, v113
	v_exp_f32_e32 v101, v153
	v_exp_f32_e32 v113, v158
	v_mfma_f32_32x32x16_f16 v[50:65], v[190:193], v[82:85], v[50:65]
	v_fma_f32 v114, v114, s2, -v151
	v_fma_f32 v115, v115, s2, -v151
	v_fma_f32 v116, v116, s2, -v151
	v_fma_f32 v117, v117, s2, -v151
	v_fma_f32 v118, v118, s2, -v151
	v_fma_f32 v119, v119, s2, -v151
	v_fma_f32 v120, v120, s2, -v151
	v_mfma_f32_32x32x16_f16 v[34:49], v[174:177], v[82:85], v[34:49]
	v_fma_f32 v121, v121, s2, -v151
	v_exp_f32_e32 v114, v114
	v_exp_f32_e32 v115, v115
	v_exp_f32_e32 v116, v116
	v_exp_f32_e32 v117, v117
	v_exp_f32_e32 v118, v118
	v_exp_f32_e32 v119, v119
	v_mfma_f32_32x32x16_f16 v[18:33], v[138:141], v[82:85], v[18:33]
	v_cvt_pk_f16_f32 v82, v108, v109
	v_cvt_pk_f16_f32 v83, v110, v111
	v_cvt_pk_f16_f32 v84, v112, v100
	v_cvt_pk_f16_f32 v85, v101, v113
	v_exp_f32_e32 v120, v120
	v_exp_f32_e32 v121, v121
	v_fma_f32 v122, v122, s2, -v151
	v_mfma_f32_32x32x16_f16 v[66:81], v[194:197], v[86:89], v[66:81]
	v_fma_f32 v123, v123, s2, -v151
	v_fma_f32 v124, v124, s2, -v151
	v_fma_f32 v125, v125, s2, -v151
	v_fma_f32 v126, v126, s2, -v151
	v_fma_f32 v127, v127, s2, -v151
	v_fma_f32 v128, v128, s2, -v151
	v_fma_f32 v129, v129, s2, -v151
	v_mfma_f32_32x32x16_f16 v[50:65], v[186:189], v[86:89], v[50:65]
	v_exp_f32_e32 v122, v122
	v_exp_f32_e32 v123, v123
	v_exp_f32_e32 v124, v124
	v_exp_f32_e32 v125, v125
	v_exp_f32_e32 v126, v126
	v_exp_f32_e32 v127, v127
	v_exp_f32_e32 v128, v128
	v_mfma_f32_32x32x16_f16 v[34:49], v[154:157], v[86:89], v[34:49]
	v_exp_f32_e32 v129, v129
	v_mov_b32_e32 v152, v91
	v_mfma_f32_32x32x16_f16 v[18:33], v[134:137], v[86:89], v[18:33]
	v_cvt_pk_f16_f32 v86, v159, v160
	v_cvt_pk_f16_f32 v87, v161, v103
	v_cvt_pk_f16_f32 v88, v104, v105
	v_cvt_pk_f16_f32 v89, v106, v107
	v_mfma_f32_32x32x16_f16 v[66:81], v[250:253], v[92:95], v[66:81]
	v_mfma_f32_32x32x16_f16 v[50:65], v[182:185], v[92:95], v[50:65]
	s_nop 7
	s_nop 2
	v_pk_mul_f32 v[16:17], v[80:81], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[78:79], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[76:77], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[74:75], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[72:73], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[70:71], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[68:69], v[90:91] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[34:49], v[146:149], v[92:95], v[34:49]
	v_cvt_pk_f16_f32 v68, v118, v119
	v_cvt_pk_f16_f32 v69, v120, v121
	v_pk_mul_f32 v[64:65], v[90:91], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[90:91], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[90:91], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[90:91], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[90:91], v[56:57] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_f16 v[18:33], v[130:133], v[92:95], v[18:33]
	v_add_u32_e32 v130, s3, v255
	ds_read_b64_tr_b16 v[92:93], v2 offset:32768
	ds_read_b64_tr_b16 v[94:95], v130 offset:2048
	ds_read_b64_tr_b16 v[96:97], v130 offset:4096
	ds_read_b64_tr_b16 v[98:99], v130 offset:6144
	v_pk_mul_f32 v[2:3], v[66:67], v[90:91] op_sel_hi:[1,0]
	ds_read_b64_tr_b16 v[70:71], v130 offset:8192
	ds_read_b64_tr_b16 v[72:73], v130 offset:10240
	v_cvt_pk_f16_f32 v66, v114, v115
	v_cvt_pk_f16_f32 v67, v116, v117
	ds_read_b64_tr_b16 v[78:79], v130 offset:12288
	ds_read_b64_tr_b16 v[80:81], v130 offset:14336
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[92:95], v[86:89], v[2:17]
	v_pk_mul_f32 v[54:55], v[90:91], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[90:91], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[90:91], v[50:51] op_sel_hi:[0,1]
	v_permlane32_swap_b32_e32 v91, v152
	v_add_f32_e32 v91, v91, v152
	v_cvt_pk_f16_f32 v74, v122, v123
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[96:99], v[82:85], v[2:17]
	v_cvt_pk_f16_f32 v75, v124, v125
	v_cvt_pk_f16_f32 v76, v126, v127
	v_cvt_pk_f16_f32 v77, v128, v129
	v_add_u32_e32 v97, s3, v248
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[70:73], v[66:69], v[2:17]
	scratch_load_dword v70, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v71, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v70, s16, v70
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v96, s3, v71
	ds_read_b64_tr_b16 v[70:71], v70 offset:32768
	ds_read_b64_tr_b16 v[72:73], v96 offset:2048
	ds_read_b64_tr_b16 v[92:93], v96 offset:4096
	ds_read_b64_tr_b16 v[94:95], v96 offset:6144
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[86:89], v[50:65]
	scratch_load_dword v70, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v91, v70, v102
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[92:95], v[82:85], v[50:65]
	v_add_f32_e32 v70, v159, v160
	v_pk_mul_f32 v[48:49], v[90:91], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[90:91], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[90:91], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[90:91], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[90:91], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[90:91], v[38:39] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_f16 v[2:17], v[78:81], v[74:77], v[2:17]
	v_add_f32_e32 v78, v161, v70
	ds_read_b64_tr_b16 v[70:71], v96 offset:8192
	ds_read_b64_tr_b16 v[72:73], v96 offset:10240
	v_add_f32_e32 v78, v103, v78
	v_add_f32_e32 v78, v104, v78
	v_add_f32_e32 v78, v105, v78
	v_add_f32_e32 v78, v106, v78
	v_add_f32_e32 v92, v107, v78
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[66:69], v[50:65]
	v_add_u32_e32 v70, s16, v254
	ds_read_b64_tr_b16 v[78:79], v96 offset:12288
	ds_read_b64_tr_b16 v[80:81], v96 offset:14336
	v_add_f32_e32 v96, v108, v92
	ds_read_b64_tr_b16 v[70:71], v70 offset:32768
	ds_read_b64_tr_b16 v[72:73], v97 offset:2048
	ds_read_b64_tr_b16 v[92:93], v97 offset:4096
	ds_read_b64_tr_b16 v[94:95], v97 offset:6144
	v_pk_mul_f32 v[36:37], v[90:91], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[90:91], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[90:91], v[32:33] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[78:81], v[74:77], v[50:65]
	v_pk_mul_f32 v[30:31], v[90:91], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[90:91], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[90:91], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[90:91], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[90:91], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[90:91], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[90:91], v[18:19] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[70:73], v[86:89], v[34:49]
	v_add_f32_e32 v70, v109, v96
	v_add_f32_e32 v70, v110, v70
	v_add_f32_e32 v70, v111, v70
	v_add_f32_e32 v70, v112, v70
	v_add_f32_e32 v78, v100, v70
	ds_read_b64_tr_b16 v[70:71], v97 offset:8192
	ds_read_b64_tr_b16 v[72:73], v97 offset:10240
	v_add_f32_e32 v78, v101, v78
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[92:95], v[82:85], v[34:49]
	v_add_f32_e32 v78, v113, v78
	v_add_f32_e32 v78, v114, v78
	v_add_f32_e32 v78, v115, v78
	v_add_f32_e32 v92, v116, v78
	ds_read_b64_tr_b16 v[78:79], v97 offset:12288
	ds_read_b64_tr_b16 v[80:81], v97 offset:14336
	v_add_f32_e32 v96, v117, v92
	v_add_u32_e32 v97, s3, v0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[70:73], v[66:69], v[34:49]
	ds_read_b64_tr_b16 v[70:71], v1 offset:32768
	ds_read_b64_tr_b16 v[72:73], v97 offset:2048
	ds_read_b64_tr_b16 v[92:93], v97 offset:4096
	ds_read_b64_tr_b16 v[94:95], v97 offset:6144
	v_add_f32_e32 v1, v118, v96
	v_add_f32_e32 v1, v119, v1
	v_add_f32_e32 v1, v120, v1
	v_add_f32_e32 v1, v121, v1
	v_add_f32_e32 v1, v122, v1
	v_add_f32_e32 v1, v123, v1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[70:73], v[86:89], v[18:33]
	ds_read_b64_tr_b16 v[70:71], v97 offset:8192
	ds_read_b64_tr_b16 v[72:73], v97 offset:10240
	v_add_f32_e32 v1, v124, v1
	v_add_f32_e32 v1, v125, v1
	v_add_f32_e32 v1, v126, v1
	v_add_f32_e32 v1, v127, v1
	v_add_f32_e32 v1, v128, v1
	v_add_f32_e32 v1, v129, v1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[92:95], v[82:85], v[18:33]
	v_mfma_f32_32x32x16_f16 v[34:49], v[78:81], v[74:77], v[34:49]
	ds_read_b64_tr_b16 v[78:79], v97 offset:12288
	ds_read_b64_tr_b16 v[80:81], v97 offset:14336
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[70:73], v[66:69], v[18:33]
	scratch_load_dword v70, off, off offset:8 ; 4-byte Folded Reload
	v_mov_b32_e32 v66, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v66
	v_add_f32_e32 v66, v1, v66
	v_fmac_f32_e32 v66, v91, v90
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[78:81], v[74:77], v[18:33]
	s_barrier
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v67, v70, 2, 0
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	scratch_load_dword v248, off, off offset:12 ; 4-byte Folded Reload
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v66
	v_mov_b32_e32 v68, 0x42000000
	v_or_b32_e32 v1, s28, v70
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v66, v69
	v_log_f32_e32 v69, v69
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v68, 0, v68, vcc
	v_cmp_gt_i32_e64 s[8:9], s2, v1
	v_sub_f32_e32 v1, v69, v68
	v_add_f32_e32 v1, v243, v1
	ds_write_b32 v67, v1
	v_mov_b32_e32 v1, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_sub_i32 s2, 0x4000, s28
	v_bfrev_b32_e32 v0, 1
	s_and_b32 s5, s19, 0xffff
	s_mov_b32 s6, s14
	s_mov_b32 s7, s15
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v1, v1, v248 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v68, 0, v1
	ds_read_b32 v68, v68
	v_cmp_lt_i32_sdwa s[2:3], v248, s2 src0_sel:BYTE_0 src1_sel:DWORD
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v1, v0, v1, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v68, v1, s[4:7], 0 offen
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
	scratch_load_dword v248, off, off offset:12 ; 4-byte Folded Reload
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_9:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v66
	v_mov_b32_e32 v1, 0x42000000
	s_and_b32 s5, s19, 0xffff
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v66, v68
	v_log_f32_e32 v68, v68
	v_cndmask_b32_e32 v1, 0, v1, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_sub_f32_e32 v1, v68, v1
	v_add_f32_e32 v1, v243, v1
	ds_write_b32 v67, v1
	v_mov_b32_e32 v1, 2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v0, v1, v248 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v1, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v1, v1
	v_bfrev_b32_e32 v67, 1
	v_cndmask_b32_e64 v0, v67, v0, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v1, v0, s[4:7], 0 offen
.LBB0_10:                               ; %.critedge
	v_div_scale_f32 v0, s[0:1], v66, v66, 1.0
	v_rcp_f32_e32 v0, v0
	v_div_scale_f32 v1, vcc, 1.0, v66, 1.0
	s_mul_i32 s0, s25, s18
	v_mul_f32_e32 v0, v1, v0
	s_ashr_i32 s1, s0, 31
	s_nop 0
	v_div_fmas_f32 v0, 0, 0, v0
	v_div_fixup_f32 v0, v0, v66, 1.0
	v_pk_mul_f32 v[24:25], v[0:1], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[0:1], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[0:1], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[0:1], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v25, v24, v25
	v_cvt_pk_f16_f32 v24, v22, v23
	v_cvt_pk_f16_f32 v21, v20, v21
	v_cvt_pk_f16_f32 v20, v18, v19
	v_pk_mul_f32 v[18:19], v[0:1], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[0:1], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[0:1], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v29, v28, v29
	v_cvt_pk_f16_f32 v28, v26, v27
	v_cvt_pk_f16_f32 v19, v18, v19
	v_cvt_pk_f16_f32 v18, v22, v23
	v_pk_mul_f32 v[22:23], v[0:1], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[0:1], v[42:43] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v33, v32, v33
	v_cvt_pk_f16_f32 v32, v30, v31
	v_cvt_pk_f16_f32 v23, v22, v23
	v_cvt_pk_f16_f32 v22, v26, v27
	v_pk_mul_f32 v[26:27], v[0:1], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[0:1], v[38:39] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v27, v26, v27
	v_cvt_pk_f16_f32 v26, v30, v31
	v_pk_mul_f32 v[30:31], v[0:1], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v31, v30, v31
	v_cvt_pk_f16_f32 v30, v34, v35
	v_pk_mul_f32 v[34:35], v[0:1], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[0:1], v[62:63] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v35, v34, v35
	v_cvt_pk_f16_f32 v34, v36, v37
	v_pk_mul_f32 v[36:37], v[0:1], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[0:1], v[58:59] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v37, v36, v37
	v_cvt_pk_f16_f32 v36, v38, v39
	v_pk_mul_f32 v[38:39], v[0:1], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[0:1], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[0:1], v[4:5] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v39, v38, v39
	v_cvt_pk_f16_f32 v38, v40, v41
	v_pk_mul_f32 v[40:41], v[0:1], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[0:1], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[0:1], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[0:1], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[0:1], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v1, v4, v5
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v0, v2, v3
	scratch_load_dword v3, off, off         ; 4-byte Folded Reload
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s26, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s27, s28
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s27, 0x3fff
	v_mul_lo_u32 v2, s27, v70
	s_bitset1_b32 s2, 14
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cvt_pk_f16_f32 v9, v8, v9
	v_cvt_pk_f16_f32 v8, v6, v7
	v_cvt_pk_f16_f32 v13, v12, v13
	v_cvt_pk_f16_f32 v12, v10, v11
	v_cvt_pk_f16_f32 v17, v16, v17
	v_cvt_pk_f16_f32 v16, v14, v15
	v_cvt_pk_f16_f32 v41, v40, v41
	v_cvt_pk_f16_f32 v40, v42, v43
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v2, v2, v3, 1
	v_bfrev_b32_e32 v3, 1
	v_cndmask_b32_e64 v4, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v0, 16, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[8:9], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 32, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[12:13], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 48, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[16:17], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 64, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[40:41], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x50, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[38:39], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x60, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[36:37], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x70, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[34:35], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x80, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[30:31], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x90, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[26:27], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xa0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[22:23], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xb0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[18:19], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xc0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[20:21], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xd0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[24:25], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xe0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[28:29], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xf0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[8:9]
	buffer_store_dwordx2 v[32:33], v0, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 48
		.amdhsa_kernarg_size 144
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 41
		.amdhsa_accum_offset 256
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
	.size	attn_fwd, .Lfunc_end0-attn_fwd
	.cfi_endproc
                                        ; -- End function
	.set attn_fwd.num_vgpr, 256
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 41
	.set attn_fwd.private_seg_size, 48
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12384
; TotalNumSgprs: 47
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 48
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 47
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
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
	.byte	1                               ; Abbrev [1] 0xb:0x45 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x1f DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	677                             ; DW_AT_call_line
	.byte	52                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
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
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"flash-attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton/python/../fa" ; string offset=26
.Linfo_string3:
	.asciz	"attn_fwd"                      ; string offset=67
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
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .offset:         108
        .size:           4
        .value_kind:     by_value
      - .offset:         112
        .size:           4
        .value_kind:     by_value
      - .offset:         116
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .offset:         128
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .max_flat_workgroup_size: 512
    .name:           attn_fwd
    .private_segment_fixed_size: 48
    .sgpr_count:     47
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 11
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
