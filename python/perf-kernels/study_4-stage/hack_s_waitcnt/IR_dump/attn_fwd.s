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
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
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
	s_movk_i32 s40, 0x78
	v_cndmask_b32_e32 v37, v45, v26, vcc
	buffer_load_dwordx4 v[26:29], v36, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v37, s[0:3], 0 offen
	v_and_b32_e32 v36, 0x78, v204
	v_bitop3_b32 v36, v36, v47, v34 bitop3:0xde
	v_lshlrev_b32_e32 v48, 1, v36
	v_add_u32_e32 v36, 0, v48
	s_barrier
	v_bitop3_b32 v46, v204, v35, s40 bitop3:0x28
	v_and_b32_e32 v203, 31, v0
	s_movk_i32 s6, 0xe0
	v_lshrrev_b32_e32 v44, 2, v0
	v_and_b32_e32 v74, 63, v0
	v_mad_u64_u32 v[200:201], s[0:1], s21, v1, v[34:35]
	v_lshl_add_u32 v209, s21, 5, v200
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
	ds_read_b128 v[154:157], v6
	v_lshl_add_u32 v5, v7, 1, 0
	v_lshl_add_u32 v6, v8, 1, 0
	v_bitop3_b32 v11, v3, v42, v34 bitop3:0xf6
	v_bitop3_b32 v3, v3, v43, v34 bitop3:0xf6
	ds_read_b128 v[150:153], v5
	ds_read_b128 v[138:141], v6
	v_lshl_add_u32 v5, v9, 1, 0
	v_lshl_add_u32 v6, v10, 1, 0
	ds_read_b128 v[134:137], v5
	ds_read_b128 v[130:133], v6
	v_lshl_add_u32 v6, v3, 1, 0
	v_or_b32_e32 v3, v47, v34
	v_lshlrev_b32_e32 v213, 1, v3
	v_or_b32_e32 v2, v2, v34
	v_lshlrev_b32_e32 v214, 1, v2
	v_sub_u32_e32 v2, v48, v213
	v_ashrrev_i16_e32 v2, 4, v2
	v_add_u32_sdwa v2, v74, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshlrev_b32_e32 v3, 2, v2
	ds_bpermute_b32 v7, v3, v200
	v_lshrrev_b64 v[2:3], v2, exec
	v_and_b32_e32 v8, 1, v2
	v_sub_u32_e32 v2, v4, v214
	v_ashrrev_i16_e32 v2, 4, v2
	v_add_u32_sdwa v2, v74, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshlrev_b32_e32 v3, 2, v2
	ds_bpermute_b32 v4, v3, v209
	v_lshrrev_b64 v[2:3], v2, exec
	v_and_b32_e32 v9, 1, v2
	v_sub_u32_e32 v2, v46, v34
	v_ashrrev_i32_e32 v2, 3, v2
	v_add_u32_e32 v2, v2, v74
	v_lshlrev_b32_e32 v216, 2, v2
	ds_bpermute_b32 v10, v216, v200
	v_add_u32_e32 v75, 0, v213
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v7, 1, v7
	v_lshrrev_b64 v[2:3], v2, exec
	v_cmp_eq_u32_e32 vcc, 1, v8
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v4, 1, v4
	v_and_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v76, v45, v7, vcc
	v_readfirstlane_b32 s33, v75
	v_cmp_eq_u32_e32 vcc, 1, v9
	v_add_u32_e32 v78, 0, v214
	v_lshl_add_u32 v5, v11, 1, 0
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v10
	s_mov_b32 m0, s33
	v_cndmask_b32_e32 v77, v45, v4, vcc
	v_readfirstlane_b32 s41, v78
	s_lshl_b64 s[6:7], s[36:37], 1
	v_cmp_eq_u32_e32 vcc, 1, v2
	ds_read_b128 v[146:149], v5
	ds_read_b128 v[142:145], v6
	buffer_load_dwordx4 v76, s[20:23], 0 offen lds
	s_mov_b32 m0, s41
	s_add_u32 s0, s20, s6
	v_cndmask_b32_e32 v2, v45, v3, vcc
	ds_bpermute_b32 v3, v216, v209
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
	v_mad_u64_u32 v[198:199], s[14:15], s24, v1, v[34:35]
	v_lshl_add_u32 v202, s24, 5, v198
	s_and_b32 s13, s24, 0x3fff
	s_bitset1_b32 s13, 14
	s_and_b32 s14, s16, 0xffff
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[154:157], v[2:17]
	v_bitop3_b32 v22, v38, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v83, 1, v22
	v_add_u32_e32 v84, 0, v83
	ds_read_b128 v[22:25], v84
	ds_read_b128 v[50:53], v84 offset:8192
	s_lshl_b32 s22, s13, 16
	s_or_b32 s13, s14, s22
	s_add_u32 s0, s0, s6
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[150:153], v[2:17]
	v_bitop3_b32 v22, v39, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v85, 1, v22
	v_add_u32_e32 v86, 0, v85
	ds_read_b128 v[22:25], v86
	ds_read_b128 v[54:57], v86 offset:8192
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_movk_i32 s23, 0x50
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
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[146:149], v[2:17]
	v_bitop3_b32 v22, v43, v217, v34 bitop3:0xde
	v_lshlrev_b32_e32 v93, 1, v22
	v_add_u32_e32 v98, 0, v93
	ds_read_b128 v[22:25], v98
	ds_read_b128 v[70:73], v98 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[142:145], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[114:117], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[46:49], v[154:157], v[18:33]
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
	v_mfma_f32_32x32x16_f16 v[18:33], v[50:53], v[150:153], v[18:33]
	v_max3_f32 v46, v46, v12, v13
	v_max3_f32 v46, v46, v14, v15
	v_max3_f32 v46, v46, v16, v17
	v_mfma_f32_32x32x16_f16 v[18:33], v[54:57], v[138:141], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[58:61], v[134:137], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[62:65], v[130:133], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[66:69], v[146:149], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[70:73], v[142:145], v[18:33]
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
	v_bitop3_b32 v46, v35, v46, s40 bitop3:0x6c
	v_sub_u32_e32 v46, v46, v34
	v_ashrrev_i32_e32 v46, 3, v46
	v_add_u32_e32 v201, v46, v74
	v_lshlrev_b32_e32 v199, 2, v201
	ds_bpermute_b32 v1, v199, v198
	ds_bpermute_b32 v49, v199, v202
	v_lshrrev_b64 v[46:47], v201, exec
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
	s_add_u32 s24, s12, s20
	s_addc_u32 s29, s16, s21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v45, s[12:15], 0 offen lds
	s_or_b32 s1, s1, s19
	s_mov_b32 m0, s33
	v_add_u32_e32 v46, 0xc000, v75
	buffer_load_dwordx4 v76, s[0:3], 0 offen lds
	s_mov_b32 m0, s41
	v_readfirstlane_b32 s12, v46
	buffer_load_dwordx4 v77, s[0:3], 0 offen lds
	s_and_b32 s0, s29, 0xffff
	s_or_b32 s1, s0, s22
	s_mov_b32 s0, s24
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
	v_fmamk_f32 v46, v2, 0x3e0293ee, v45
	s_movk_i32 s0, 0x1ff
	v_add_u32_e32 v2, 0xff, v0
	v_cmp_gt_u32_e32 vcc, s0, v2
	s_movk_i32 s0, 0x1fe
	s_add_i32 s12, 0, 0x4000
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
	v_fmamk_f32 v18, v18, 0x3e0293ee, v45
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
	v_cmp_lt_u32_e64 s[0:1], s0, v2
	v_add_u32_e32 v2, s12, v79
	v_add_u32_e32 v33, s12, v81
	v_add_u32_e32 v47, s12, v83
	v_add_u32_e32 v48, s12, v85
	v_add_u32_e32 v49, s12, v87
	v_add_u32_e32 v50, s12, v89
	v_add_u32_e32 v51, s12, v91
	v_add_u32_e32 v52, s12, v93
	ds_read_b128 v[66:69], v80 offset:16384
	ds_read_b128 v[170:173], v82 offset:16384
	ds_read_b128 v[166:169], v84 offset:16384
	ds_read_b128 v[162:165], v86 offset:16384
	ds_read_b128 v[158:161], v88 offset:16384
	ds_read_b128 v[94:97], v90 offset:16384
	ds_read_b128 v[90:93], v92 offset:16384
	ds_read_b128 v[86:89], v98 offset:16384
	ds_read_b128 v[82:85], v2 offset:8192
	ds_read_b128 v[126:129], v33 offset:8192
	ds_read_b128 v[122:125], v47 offset:8192
	ds_read_b128 v[118:121], v48 offset:8192
	ds_read_b128 v[110:113], v49 offset:8192
	ds_read_b128 v[106:109], v50 offset:8192
	ds_read_b128 v[102:105], v51 offset:8192
	ds_read_b128 v[98:101], v52 offset:8192
	s_mov_b32 s16, 0
	s_movk_i32 s15, 0x70
	s_mov_b32 s33, 0x3e0293ee
	v_fmac_f32_e32 v1, 0xbe0293ee, v186
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[12:13]
	v_exp_f32_e32 v174, v1
	v_lshrrev_b32_e32 v1, 3, v0
	v_exp_f32_e32 v179, v5
	v_and_b32_e32 v1, 4, v1
	v_lshlrev_b32_e32 v5, 2, v0
	v_exp_f32_e32 v177, v3
	v_exp_f32_e32 v176, v4
	v_exp_f32_e32 v180, v8
	v_and_b32_e32 v3, 16, v0
	v_and_b32_e32 v4, 64, v35
	v_and_b32_e32 v5, 12, v5
	v_and_or_b32 v8, v44, 3, v1
	v_exp_f32_e32 v178, v6
	v_exp_f32_e32 v181, v7
	v_and_or_b32 v6, v35, 32, v5
	v_or_b32_e32 v7, v4, v3
	v_lshlrev_b32_e32 v218, 7, v8
	s_add_u32 s0, s30, s34
	v_or3_b32 v222, v7, v6, v218
	v_bitop3_b32 v7, v5, v35, 32 bitop3:0x72
	s_addc_u32 s1, s31, s35
	v_or3_b32 v221, v3, v7, v4
	v_bitop3_b32 v4, v6, v35, 64 bitop3:0x72
	s_mul_i32 s13, s36, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_exp_f32_e32 v175, v46
	v_exp_f32_e32 v183, v9
	v_exp_f32_e32 v182, v10
	v_exp_f32_e32 v185, v11
	v_exp_f32_e32 v184, v12
	v_exp_f32_e32 v187, v13
	v_exp_f32_e32 v188, v14
	v_exp_f32_e32 v193, v15
	v_exp_f32_e32 v189, v16
	v_exp_f32_e32 v195, v17
	v_exp_f32_e32 v194, v18
	v_exp_f32_e32 v197, v19
	v_exp_f32_e32 v196, v20
	v_exp_f32_e32 v236, v21
	v_exp_f32_e32 v235, v22
	v_exp_f32_e32 v238, v23
	v_exp_f32_e32 v237, v24
	v_exp_f32_e32 v240, v25
	v_exp_f32_e32 v239, v26
	v_exp_f32_e32 v241, v27
	v_exp_f32_e32 v242, v28
	v_exp_f32_e32 v244, v29
	v_exp_f32_e32 v243, v30
	v_exp_f32_e32 v246, v31
	v_exp_f32_e32 v245, v32
	v_exp_f32_e32 v247, v45
	v_or_b32_e32 v220, v4, v3
	v_bitop3_b32 v4, v35, v5, s14 bitop3:0x4e
	s_mul_hi_i32 s12, s36, 6
	s_add_u32 s0, s13, s0
	v_or_b32_e32 v219, v4, v3
	v_lshlrev_b32_e32 v3, 3, v203
	s_addc_u32 s1, s12, s1
	v_bitop3_b32 v4, v3, v36, s40 bitop3:0x6c
	v_bitop3_b32 v5, v3, v37, s40 bitop3:0x6c
	v_bitop3_b32 v6, v3, v38, s40 bitop3:0x6c
	v_bitop3_b32 v7, v3, v39, s40 bitop3:0x6c
	v_bitop3_b32 v8, v3, v40, s40 bitop3:0x6c
	v_bitop3_b32 v9, v3, v41, s40 bitop3:0x6c
	v_bitop3_b32 v10, v3, v42, s40 bitop3:0x6c
	v_bitop3_b32 v3, v3, v43, s40 bitop3:0x6c
	s_add_u32 s4, s4, s0
	v_and_b32_e32 v223, 64, v204
	v_bitop3_b32 v215, v44, v34, 8 bitop3:0x6c
	v_bitop3_b32 v212, v36, v34, 16 bitop3:0x36
	v_bitop3_b32 v211, v36, v34, 32 bitop3:0x36
	v_bitop3_b32 v210, v36, v34, 48 bitop3:0x36
	v_bitop3_b32 v208, v36, v34, 64 bitop3:0x36
	v_bitop3_b32 v207, v36, v34, s23 bitop3:0x36
	v_bitop3_b32 v206, v36, v34, s14 bitop3:0x36
	v_bitop3_b32 v205, v36, v34, s15 bitop3:0x36
	s_addc_u32 s30, s5, s1
	s_add_i32 s5, 0, 0x8000
	s_add_i32 s23, 0, 0xc000
	v_mov_b32_e32 v224, 1.0
	s_movk_i32 s31, 0xffc0
	v_lshlrev_b32_e32 v225, 1, v4
	v_lshlrev_b32_e32 v226, 1, v5
	v_lshlrev_b32_e32 v229, 1, v6
	v_lshlrev_b32_e32 v230, 1, v7
	v_lshlrev_b32_e32 v231, 1, v8
	v_lshlrev_b32_e32 v232, 1, v9
	v_lshlrev_b32_e32 v233, 1, v10
	v_lshlrev_b32_e32 v234, 1, v3
	s_mov_b32 s13, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[114:117], 0
	v_mov_b32_e32 v248, v186
	s_mov_b32 s14, s5
	s_mov_b32 s5, s23
	v_mov_b32_e32 v190, v224
	s_mov_b32 s36, s16
	s_setprio 0
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[154:157], v[66:81]
	v_add_f32_e32 v186, v175, v177
	v_add_f32_e32 v186, v186, v176
	v_add_f32_e32 v186, v186, v179
	v_add_f32_e32 v186, v186, v178
	v_add_f32_e32 v186, v186, v181
	v_add_f32_e32 v186, v186, v180
	v_add_f32_e32 v170, v186, v183
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[150:153], v[66:81]
	v_add_f32_e32 v170, v170, v182
	v_add_f32_e32 v170, v170, v185
	v_add_f32_e32 v170, v170, v184
	v_add_f32_e32 v170, v170, v187
	v_add_f32_e32 v170, v170, v188
	v_add_f32_e32 v170, v170, v193
	v_add_f32_e32 v166, v170, v189
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[138:141], v[66:81]
	v_add_f32_e32 v166, v166, v195
	v_add_f32_e32 v166, v166, v194
	v_add_f32_e32 v166, v166, v197
	v_add_f32_e32 v166, v166, v196
	v_add_f32_e32 v166, v166, v236
	v_add_f32_e32 v166, v166, v235
	v_add_f32_e32 v162, v166, v238
	v_mfma_f32_32x32x16_f16 v[66:81], v[158:161], v[134:137], v[66:81]
	v_add_f32_e32 v162, v162, v237
	v_add_f32_e32 v162, v162, v240
	v_add_f32_e32 v162, v162, v239
	v_add_f32_e32 v162, v162, v241
	v_add_f32_e32 v162, v162, v242
	v_add_f32_e32 v162, v162, v244
	v_add_f32_e32 v158, v162, v243
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[130:133], v[66:81]
	v_add_f32_e32 v158, v158, v246
	v_add_f32_e32 v158, v158, v245
	v_add_f32_e32 v158, v158, v247
	v_mov_b32_e32 v159, v158
	s_nop 1
	v_permlane32_swap_b32_e32 v158, v159
	v_add_f32_e32 v224, v158, v159
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[146:149], v[66:81]
	v_mul_f32_e32 v50, v50, v174
	v_mul_f32_e32 v51, v51, v174
	v_mul_f32_e32 v52, v52, v174
	v_mul_f32_e32 v53, v53, v174
	v_mul_f32_e32 v54, v54, v174
	v_mul_f32_e32 v55, v55, v174
	v_mul_f32_e32 v56, v56, v174
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[142:145], v[66:81]
	v_mul_f32_e32 v57, v57, v174
	v_mul_f32_e32 v58, v58, v174
	v_mul_f32_e32 v59, v59, v174
	v_mul_f32_e32 v60, v60, v174
	v_mul_f32_e32 v61, v61, v174
	v_mul_f32_e32 v62, v62, v174
	v_mul_f32_e32 v63, v63, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[114:117], 0
	v_mul_f32_e32 v64, v64, v174
	v_mul_f32_e32 v65, v65, v174
	v_mul_f32_e32 v34, v34, v174
	v_mul_f32_e32 v35, v35, v174
	v_mul_f32_e32 v36, v36, v174
	v_mul_f32_e32 v37, v37, v174
	v_mul_f32_e32 v38, v38, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[154:157], v[82:97]
	v_mul_f32_e32 v39, v39, v174
	v_mul_f32_e32 v40, v40, v174
	v_mul_f32_e32 v41, v41, v174
	v_mul_f32_e32 v42, v42, v174
	v_mul_f32_e32 v43, v43, v174
	v_mul_f32_e32 v44, v44, v174
	v_mul_f32_e32 v45, v45, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[150:153], v[82:97]
	v_mul_f32_e32 v46, v46, v174
	v_mul_f32_e32 v47, v47, v174
	v_mul_f32_e32 v48, v48, v174
	v_mul_f32_e32 v49, v49, v174
	v_mul_f32_e32 v18, v18, v174
	v_mul_f32_e32 v19, v19, v174
	v_mul_f32_e32 v20, v20, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[138:141], v[82:97]
	v_mul_f32_e32 v21, v21, v174
	v_mul_f32_e32 v22, v22, v174
	v_mul_f32_e32 v23, v23, v174
	v_mul_f32_e32 v24, v24, v174
	v_mul_f32_e32 v25, v25, v174
	v_mul_f32_e32 v26, v26, v174
	v_mul_f32_e32 v27, v27, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[134:137], v[82:97]
	v_mul_f32_e32 v28, v28, v174
	v_mul_f32_e32 v29, v29, v174
	v_mul_f32_e32 v30, v30, v174
	v_mul_f32_e32 v31, v31, v174
	v_mul_f32_e32 v32, v32, v174
	v_mul_f32_e32 v33, v33, v174
	v_mul_f32_e32 v2, v2, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[130:133], v[82:97]
	v_mul_f32_e32 v3, v3, v174
	v_mul_f32_e32 v4, v4, v174
	v_mul_f32_e32 v5, v5, v174
	v_mul_f32_e32 v6, v6, v174
	v_mul_f32_e32 v7, v7, v174
	v_mul_f32_e32 v8, v8, v174
	v_mul_f32_e32 v9, v9, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[146:149], v[82:97]
	v_mul_f32_e32 v10, v10, v174
	v_mul_f32_e32 v11, v11, v174
	v_mul_f32_e32 v12, v12, v174
	v_mul_f32_e32 v13, v13, v174
	v_mul_f32_e32 v14, v14, v174
	v_mul_f32_e32 v15, v15, v174
	v_mul_f32_e32 v16, v16, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[142:145], v[82:97]
	v_mul_f32_e32 v17, v17, v174
	v_fmac_f32_e32 v224, v190, v174
	v_cvt_pk_f16_f32 v110, v175, v177
	v_cvt_pk_f16_f32 v111, v176, v179
	v_cvt_pk_f16_f32 v112, v178, v181
	v_cvt_pk_f16_f32 v113, v180, v183
	v_cvt_pk_f16_f32 v106, v182, v185
	v_cvt_pk_f16_f32 v107, v184, v187
	v_cvt_pk_f16_f32 v108, v188, v193
	v_cvt_pk_f16_f32 v109, v189, v195
	v_cvt_pk_f16_f32 v102, v194, v197
	v_cvt_pk_f16_f32 v103, v196, v236
	v_cvt_pk_f16_f32 v104, v235, v238
	v_cvt_pk_f16_f32 v105, v237, v240
	v_cvt_pk_f16_f32 v98, v239, v241
	v_cvt_pk_f16_f32 v99, v242, v244
	v_cvt_pk_f16_f32 v100, v243, v246
	v_cvt_pk_f16_f32 v101, v245, v247
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s24, s20
	s_addc_u32 s35, s29, s21
	s_add_i32 s0, s13, 1
	s_cmp_lt_i32 s0, 2
	s_cselect_b32 s37, s0, 0
	ds_bpermute_b32 v118, v216, v200
	s_lshl_b32 s0, s37, 14
	ds_bpermute_b32 v119, v216, v209
	s_add_i32 s16, s0, 0
	v_add_u32_e32 v227, s16, v213
	v_add_u32_e32 v228, s16, v214
	s_and_b32 s0, s30, 0xffff
	v_readfirstlane_b32 s15, v227
	s_or_b32 s1, s0, s19
	s_mov_b32 s0, s4
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v118, 1, v118
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v228
	buffer_load_dwordx4 v118, s[0:3], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v118, 1, v119
	s_mov_b32 m0, s15
	v_lshlrev_b32_e32 v119, 1, v218
	buffer_load_dwordx4 v118, s[0:3], 0 offen lds
	v_lshl_add_u32 v118, v222, 1, s14
	ds_read_b64_tr_b16 v[236:237], v118
	ds_read_b64_tr_b16 v[238:239], v118 offset:2048
	ds_read_b64_tr_b16 v[240:241], v118 offset:4096
	ds_read_b64_tr_b16 v[242:243], v118 offset:6144
	ds_read_b64_tr_b16 v[244:245], v118 offset:8192
	ds_read_b64_tr_b16 v[246:247], v118 offset:10240
	ds_read_b64_tr_b16 v[194:195], v118 offset:12288
	ds_read_b64_tr_b16 v[196:197], v118 offset:14336
	v_lshlrev_b32_e32 v118, 1, v221
	v_add3_u32 v118, s14, v118, v119
	ds_read_b64_tr_b16 v[190:191], v118
	ds_read_b64_tr_b16 v[192:193], v118 offset:2048
	ds_read_b64_tr_b16 v[186:187], v118 offset:4096
	ds_read_b64_tr_b16 v[188:189], v118 offset:6144
	ds_read_b64_tr_b16 v[182:183], v118 offset:8192
	ds_read_b64_tr_b16 v[184:185], v118 offset:10240
	ds_read_b64_tr_b16 v[178:179], v118 offset:12288
	ds_read_b64_tr_b16 v[180:181], v118 offset:14336
	v_lshlrev_b32_e32 v118, 1, v220
	v_add3_u32 v118, s14, v118, v119
	ds_read_b64_tr_b16 v[174:175], v118
	ds_read_b64_tr_b16 v[176:177], v118 offset:2048
	ds_read_b64_tr_b16 v[170:171], v118 offset:4096
	ds_read_b64_tr_b16 v[172:173], v118 offset:6144
	ds_read_b64_tr_b16 v[166:167], v118 offset:8192
	ds_read_b64_tr_b16 v[168:169], v118 offset:10240
	ds_read_b64_tr_b16 v[162:163], v118 offset:12288
	ds_read_b64_tr_b16 v[164:165], v118 offset:14336
	v_lshlrev_b32_e32 v118, 1, v219
	v_add3_u32 v120, s14, v118, v119
	ds_read_b64_tr_b16 v[158:159], v120
	ds_read_b64_tr_b16 v[160:161], v120 offset:2048
	ds_read_b64_tr_b16 v[126:127], v120 offset:4096
	ds_read_b64_tr_b16 v[128:129], v120 offset:6144
	ds_read_b64_tr_b16 v[122:123], v120 offset:8192
	ds_read_b64_tr_b16 v[124:125], v120 offset:10240
	ds_read_b64_tr_b16 v[118:119], v120 offset:12288
	ds_read_b64_tr_b16 v[120:121], v120 offset:14336
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[236:239], v[110:113], v[50:65]
	v_max_f32_e32 v235, v67, v67
	v_max_f32_e32 v236, v66, v66
	v_max_f32_e32 v235, v236, v235
	v_max3_f32 v235, v235, v68, v69
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[50:65], v[240:243], v[106:109], v[50:65]
	v_mfma_f32_32x32x16_f16 v[50:65], v[244:247], v[102:105], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[190:193], v[110:113], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[110:113], v[18:33]
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[110:113], v[2:17]
	v_mfma_f32_32x32x16_f16 v[50:65], v[194:197], v[98:101], v[50:65]
	v_max3_f32 v194, v235, v70, v71
	v_max3_f32 v194, v194, v72, v73
	v_max3_f32 v194, v194, v74, v75
	v_max3_f32 v194, v194, v76, v77
	v_max3_f32 v194, v194, v78, v79
	v_max3_f32 v194, v194, v80, v81
	v_max3_f32 v194, v194, v82, v83
	v_mfma_f32_32x32x16_f16 v[34:49], v[186:189], v[106:109], v[34:49]
	v_max3_f32 v190, v194, v84, v85
	v_max3_f32 v190, v190, v86, v87
	v_max3_f32 v190, v190, v88, v89
	v_max3_f32 v190, v190, v90, v91
	v_max3_f32 v190, v190, v92, v93
	v_max3_f32 v190, v190, v94, v95
	v_max3_f32 v190, v190, v96, v97
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[106:109], v[18:33]
	v_mov_b32_e32 v186, v190
	s_nop 1
	v_permlane32_swap_b32_e32 v190, v186
	v_max3_f32 v186, v248, v190, v186
	v_mul_f32_e32 v192, 0x3e0293ee, v186
	v_fma_f32 v66, v66, s33, -v192
	v_fma_f32 v67, v67, s33, -v192
	v_mfma_f32_32x32x16_f16 v[2:17], v[126:129], v[106:109], v[2:17]
	v_fma_f32 v68, v68, s33, -v192
	v_fma_f32 v69, v69, s33, -v192
	v_fma_f32 v70, v70, s33, -v192
	v_fma_f32 v71, v71, s33, -v192
	v_fma_f32 v72, v72, s33, -v192
	v_fma_f32 v73, v73, s33, -v192
	v_fma_f32 v74, v74, s33, -v192
	v_mfma_f32_32x32x16_f16 v[34:49], v[182:185], v[102:105], v[34:49]
	v_fma_f32 v75, v75, s33, -v192
	v_fma_f32 v76, v76, s33, -v192
	v_fma_f32 v77, v77, s33, -v192
	v_fma_f32 v78, v78, s33, -v192
	v_fma_f32 v79, v79, s33, -v192
	v_fma_f32 v80, v80, s33, -v192
	v_fma_f32 v81, v81, s33, -v192
	v_mfma_f32_32x32x16_f16 v[18:33], v[166:169], v[102:105], v[18:33]
	v_fma_f32 v82, v82, s33, -v192
	v_fma_f32 v83, v83, s33, -v192
	v_fma_f32 v84, v84, s33, -v192
	v_fma_f32 v85, v85, s33, -v192
	v_fma_f32 v86, v86, s33, -v192
	v_fma_f32 v87, v87, s33, -v192
	v_fma_f32 v88, v88, s33, -v192
	v_mfma_f32_32x32x16_f16 v[2:17], v[122:125], v[102:105], v[2:17]
	v_fma_f32 v89, v89, s33, -v192
	v_fma_f32 v90, v90, s33, -v192
	v_fma_f32 v91, v91, s33, -v192
	v_fma_f32 v92, v92, s33, -v192
	v_fma_f32 v93, v93, s33, -v192
	v_fma_f32 v94, v94, s33, -v192
	v_fma_f32 v95, v95, s33, -v192
	v_mfma_f32_32x32x16_f16 v[34:49], v[178:181], v[98:101], v[34:49]
	v_fma_f32 v96, v96, s33, -v192
	v_fma_f32 v97, v97, s33, -v192
	v_exp_f32_e32 v175, v66
	v_fma_f32 v66, v248, s33, -v192
	v_exp_f32_e32 v177, v67
	v_exp_f32_e32 v176, v68
	v_exp_f32_e32 v179, v69
	v_mfma_f32_32x32x16_f16 v[18:33], v[162:165], v[98:101], v[18:33]
	v_exp_f32_e32 v178, v70
	v_exp_f32_e32 v181, v71
	v_exp_f32_e32 v180, v72
	v_exp_f32_e32 v183, v73
	v_exp_f32_e32 v182, v74
	v_exp_f32_e32 v185, v75
	v_exp_f32_e32 v184, v76
	v_mfma_f32_32x32x16_f16 v[2:17], v[118:121], v[98:101], v[2:17]
	v_exp_f32_e32 v187, v77
	v_exp_f32_e32 v188, v78
	v_exp_f32_e32 v193, v79
	v_exp_f32_e32 v189, v80
	v_exp_f32_e32 v195, v81
	v_exp_f32_e32 v194, v82
	v_exp_f32_e32 v197, v83
	v_exp_f32_e32 v196, v84
	v_exp_f32_e32 v236, v85
	v_exp_f32_e32 v235, v86
	v_exp_f32_e32 v238, v87
	v_exp_f32_e32 v237, v88
	v_exp_f32_e32 v240, v89
	v_exp_f32_e32 v239, v90
	v_exp_f32_e32 v241, v91
	v_exp_f32_e32 v242, v92
	v_exp_f32_e32 v244, v93
	v_exp_f32_e32 v243, v94
	v_exp_f32_e32 v246, v95
	v_exp_f32_e32 v245, v96
	v_exp_f32_e32 v247, v97
	v_exp_f32_e32 v174, v66
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s0, s13, 14
	ds_bpermute_b32 v68, v199, v198
	s_add_i32 s34, s0, 0
	ds_bpermute_b32 v69, v199, v202
	s_add_i32 s23, s34, 0x8000
	v_add_u32_e32 v66, s23, v213
	s_and_b32 s0, s35, 0xffff
	v_add_u32_e32 v67, s23, v214
	s_or_b32 s13, s0, s22
	v_readfirstlane_b32 s0, v66
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v68, 1, v68
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v67
	buffer_load_dwordx4 v68, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v69
	s_mov_b32 m0, s0
	v_lshlrev_b32_e32 v248, 1, v217
	buffer_load_dwordx4 v66, s[12:15], 0 offen lds
	v_add3_u32 v70, s36, v225, v248
	v_add3_u32 v71, s36, v226, v248
	v_add3_u32 v72, s36, v229, v248
	v_add3_u32 v73, s36, v230, v248
	v_add3_u32 v74, s36, v231, v248
	v_add3_u32 v75, s36, v232, v248
	v_add3_u32 v76, s36, v233, v248
	v_add3_u32 v77, s36, v234, v248
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[82:85], v70 offset:8192
	ds_read_b128 v[170:173], v71
	ds_read_b128 v[126:129], v71 offset:8192
	ds_read_b128 v[166:169], v72
	ds_read_b128 v[122:125], v72 offset:8192
	ds_read_b128 v[162:165], v73
	ds_read_b128 v[118:121], v73 offset:8192
	ds_read_b128 v[158:161], v74
	ds_read_b128 v[110:113], v74 offset:8192
	ds_read_b128 v[94:97], v75
	ds_read_b128 v[106:109], v75 offset:8192
	ds_read_b128 v[90:93], v76
	ds_read_b128 v[102:105], v76 offset:8192
	ds_read_b128 v[86:89], v77
	ds_read_b128 v[98:101], v77 offset:8192
	; sched_barrier mask(0x00000000)
	s_add_u32 s24, s24, s20
	s_addc_u32 s29, s29, s21
	s_add_u32 s4, s4, s6
	s_addc_u32 s30, s30, s7
	s_add_i32 s31, s31, 64
	s_cmpk_lt_u32 s31, 0x1f00
	s_mov_b32 s13, s37
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[114:117], 0
	v_lshlrev_b32_e32 v200, 1, v222
	s_waitcnt vmcnt(0)
	s_barrier
	v_mul_f32_e32 v50, v50, v174
	v_mul_f32_e32 v51, v51, v174
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[154:157], v[66:81]
	v_mul_f32_e32 v52, v52, v174
	v_mul_f32_e32 v53, v53, v174
	v_mul_f32_e32 v54, v54, v174
	v_mul_f32_e32 v55, v55, v174
	v_mul_f32_e32 v56, v56, v174
	v_mul_f32_e32 v57, v57, v174
	v_mul_f32_e32 v58, v58, v174
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[150:153], v[66:81]
	v_mul_f32_e32 v59, v59, v174
	v_mul_f32_e32 v60, v60, v174
	v_mul_f32_e32 v61, v61, v174
	v_mul_f32_e32 v62, v62, v174
	v_mul_f32_e32 v63, v63, v174
	v_mul_f32_e32 v64, v64, v174
	v_mul_f32_e32 v65, v65, v174
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[138:141], v[66:81]
	v_add_u32_e32 v162, s5, v200
	v_and_b32_e32 v170, 0x100, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v170
	v_and_b32_e32 v170, 0xa0, v204
	v_or3_b32 v190, v170, v203, v223
	v_add_lshl_u32 v203, v221, v218, 1
	v_mul_f32_e32 v34, v34, v174
	v_mfma_f32_32x32x16_f16 v[66:81], v[158:161], v[134:137], v[66:81]
	v_add_f32_e32 v158, v175, v177
	v_add_f32_e32 v158, v158, v176
	v_add_f32_e32 v158, v158, v179
	v_cvt_pk_f16_f32 v159, v176, v179
	v_cvt_pk_f16_f32 v160, v178, v181
	v_cvt_pk_f16_f32 v161, v180, v183
	v_mul_f32_e32 v35, v35, v174
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[130:133], v[66:81]
	v_add_f32_e32 v94, v158, v178
	v_add_f32_e32 v94, v94, v181
	v_add_f32_e32 v94, v94, v180
	v_add_f32_e32 v94, v94, v183
	v_add_f32_e32 v94, v94, v182
	v_add_f32_e32 v94, v94, v185
	v_mul_f32_e32 v36, v36, v174
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[146:149], v[66:81]
	v_add_f32_e32 v90, v94, v184
	v_add_f32_e32 v90, v90, v187
	v_add_f32_e32 v90, v90, v188
	v_add_f32_e32 v90, v90, v193
	v_add_f32_e32 v90, v90, v189
	v_add_f32_e32 v90, v90, v195
	v_mul_f32_e32 v37, v37, v174
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[142:145], v[66:81]
	v_add_f32_e32 v86, v90, v194
	v_add_f32_e32 v86, v86, v197
	v_add_f32_e32 v86, v86, v196
	v_add_f32_e32 v86, v86, v236
	v_add_f32_e32 v86, v86, v235
	v_add_f32_e32 v86, v86, v238
	v_add_f32_e32 v158, v86, v237
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[114:117], 0
	v_add_f32_e32 v158, v158, v240
	v_add_f32_e32 v158, v158, v239
	v_add_f32_e32 v158, v158, v241
	v_add_f32_e32 v158, v158, v242
	v_add_f32_e32 v158, v158, v244
	v_add_f32_e32 v158, v158, v243
	v_add_f32_e32 v158, v158, v246
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[154:157], v[82:97]
	v_add_f32_e32 v126, v158, v245
	v_cvt_pk_f16_f32 v158, v175, v177
	v_add_f32_e32 v126, v126, v247
	v_mov_b32_e32 v127, v126
	s_nop 1
	v_permlane32_swap_b32_e32 v126, v127
	v_add_f32_e32 v191, v126, v127
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[150:153], v[82:97]
	v_cvt_pk_f16_f32 v122, v194, v197
	v_cvt_pk_f16_f32 v126, v182, v185
	v_cvt_pk_f16_f32 v127, v184, v187
	v_cvt_pk_f16_f32 v128, v188, v193
	v_cvt_pk_f16_f32 v129, v189, v195
	v_mul_f32_e32 v38, v38, v174
	v_mul_f32_e32 v39, v39, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[138:141], v[82:97]
	v_mul_f32_e32 v40, v40, v174
	v_mul_f32_e32 v41, v41, v174
	v_mul_f32_e32 v42, v42, v174
	v_mul_f32_e32 v43, v43, v174
	v_mul_f32_e32 v44, v44, v174
	v_mul_f32_e32 v45, v45, v174
	v_mul_f32_e32 v46, v46, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[134:137], v[82:97]
	v_mul_f32_e32 v47, v47, v174
	v_mul_f32_e32 v48, v48, v174
	v_mul_f32_e32 v49, v49, v174
	v_cvt_pk_f16_f32 v123, v196, v236
	v_cvt_pk_f16_f32 v124, v235, v238
	v_cvt_pk_f16_f32 v125, v237, v240
	v_cvt_pk_f16_f32 v118, v239, v241
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[130:133], v[82:97]
	ds_read_b64_tr_b16 v[106:107], v162
	ds_read_b64_tr_b16 v[108:109], v162 offset:2048
	ds_read_b64_tr_b16 v[110:111], v162 offset:4096
	ds_read_b64_tr_b16 v[112:113], v162 offset:6144
	v_cvt_pk_f16_f32 v119, v242, v244
	v_cvt_pk_f16_f32 v120, v243, v246
	v_cvt_pk_f16_f32 v121, v245, v247
	v_add_lshl_u32 v196, v220, v218, 1
	v_mul_f32_e32 v18, v18, v174
	v_mul_f32_e32 v19, v19, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[146:149], v[82:97]
	v_mul_f32_e32 v20, v20, v174
	v_mul_f32_e32 v21, v21, v174
	v_mul_f32_e32 v22, v22, v174
	v_mul_f32_e32 v23, v23, v174
	v_mul_f32_e32 v24, v24, v174
	v_mul_f32_e32 v25, v25, v174
	v_mul_f32_e32 v26, v26, v174
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[106:109], v[158:161], v[50:65]
	v_or_b32_e32 v106, v221, v218
	v_lshlrev_b32_e32 v197, 1, v106
	v_add_u32_e32 v106, s5, v197
	v_mul_f32_e32 v27, v27, v174
	v_mul_f32_e32 v28, v28, v174
	v_mul_f32_e32 v29, v29, v174
	v_mul_f32_e32 v30, v30, v174
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[142:145], v[82:97]
	ds_read_b64_tr_b16 v[98:99], v162 offset:8192
	ds_read_b64_tr_b16 v[100:101], v162 offset:10240
	ds_read_b64_tr_b16 v[102:103], v162 offset:12288
	ds_read_b64_tr_b16 v[104:105], v162 offset:14336
	v_add_u32_e32 v162, s5, v203
	v_mul_f32_e32 v31, v31, v174
	v_mul_f32_e32 v32, v32, v174
	v_mul_f32_e32 v33, v33, v174
	v_add_lshl_u32 v194, v219, v218, 1
	v_mul_f32_e32 v2, v2, v174
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[110:113], v[126:129], v[50:65]
	ds_read_b64_tr_b16 v[106:107], v106
	ds_read_b64_tr_b16 v[108:109], v162 offset:2048
	ds_read_b64_tr_b16 v[110:111], v162 offset:4096
	ds_read_b64_tr_b16 v[112:113], v162 offset:6144
	v_mul_f32_e32 v3, v3, v174
	v_mul_f32_e32 v4, v4, v174
	v_mul_f32_e32 v5, v5, v174
	v_mul_f32_e32 v6, v6, v174
	v_mul_f32_e32 v7, v7, v174
	v_mul_f32_e32 v8, v8, v174
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[106:109], v[158:161], v[34:49]
	v_or_b32_e32 v106, v220, v218
	v_lshlrev_b32_e32 v195, 1, v106
	v_add_u32_e32 v106, s5, v195
	v_mul_f32_e32 v9, v9, v174
	v_mul_f32_e32 v10, v10, v174
	v_mul_f32_e32 v11, v11, v174
	v_mul_f32_e32 v12, v12, v174
	v_mfma_f32_32x32x16_f16 v[50:65], v[98:101], v[122:125], v[50:65]
	v_mul_f32_e32 v13, v13, v174
	v_mul_f32_e32 v14, v14, v174
	v_mul_f32_e32 v15, v15, v174
	v_mul_f32_e32 v16, v16, v174
	v_mul_f32_e32 v17, v17, v174
	s_mul_i32 s2, s18, 0xc0000
	s_ashr_i32 s3, s2, 31
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[110:113], v[126:129], v[34:49]
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s8, s2
	s_addc_u32 s6, s9, s3
	s_lshl_b32 s2, s17, 14
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s4, s2
	v_mfma_f32_32x32x16_f16 v[50:65], v[102:105], v[118:121], v[50:65]
	ds_read_b64_tr_b16 v[98:99], v162 offset:8192
	ds_read_b64_tr_b16 v[100:101], v162 offset:10240
	ds_read_b64_tr_b16 v[102:103], v162 offset:12288
	ds_read_b64_tr_b16 v[104:105], v162 offset:14336
	v_add_u32_e32 v162, s5, v196
	ds_read_b64_tr_b16 v[106:107], v106
	ds_read_b64_tr_b16 v[108:109], v162 offset:2048
	ds_read_b64_tr_b16 v[110:111], v162 offset:4096
	ds_read_b64_tr_b16 v[112:113], v162 offset:6144
	s_addc_u32 s6, s6, s3
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[2:3], s[28:29], 2
	s_add_u32 s4, s4, s2
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[34:49], v[98:101], v[122:125], v[34:49]
	s_addc_u32 s19, s6, s3
	s_add_i32 s3, s28, 0xffffc100
	ds_bpermute_b32 v168, v199, v198
	s_add_u32 s12, s12, s20
	s_addc_u32 s6, s35, s21
	s_and_b32 s6, s6, 0xffff
	s_add_i32 s2, s16, 0x8000
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[18:33], v[106:109], v[158:161], v[18:33]
	v_or_b32_e32 v106, v219, v218
	v_lshlrev_b32_e32 v193, 1, v106
	v_add_u32_e32 v106, s5, v193
	s_or_b32 s13, s6, s22
	v_fmac_f32_e32 v191, v224, v174
	s_cmp_lt_i32 s3, 1
	v_bfrev_b32_e32 v198, 1
	v_mfma_f32_32x32x16_f16 v[34:49], v[102:105], v[118:121], v[34:49]
	ds_read_b64_tr_b16 v[98:99], v162 offset:8192
	ds_read_b64_tr_b16 v[100:101], v162 offset:10240
	ds_read_b64_tr_b16 v[102:103], v162 offset:12288
	ds_read_b64_tr_b16 v[104:105], v162 offset:14336
	v_add_u32_e32 v162, s5, v194
	v_add_u32_e32 v167, 0x8000, v228
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[18:33], v[110:113], v[126:129], v[18:33]
	ds_read_b64_tr_b16 v[106:107], v106
	ds_read_b64_tr_b16 v[108:109], v162 offset:2048
	ds_read_b64_tr_b16 v[110:111], v162 offset:4096
	ds_read_b64_tr_b16 v[112:113], v162 offset:6144
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[106:109], v[158:161], v[2:17]
	v_max_f32_e32 v106, v67, v67
	v_max_f32_e32 v107, v66, v66
	v_max_f32_e32 v106, v107, v106
	v_max3_f32 v106, v106, v68, v69
	v_max3_f32 v106, v106, v70, v71
	v_max3_f32 v106, v106, v72, v73
	v_max3_f32 v106, v106, v74, v75
	v_mfma_f32_32x32x16_f16 v[18:33], v[98:101], v[122:125], v[18:33]
	v_max3_f32 v106, v106, v76, v77
	v_max3_f32 v106, v106, v78, v79
	v_max3_f32 v106, v106, v80, v81
	v_max3_f32 v106, v106, v82, v83
	v_max3_f32 v106, v106, v84, v85
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[110:113], v[126:129], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[102:105], v[118:121], v[18:33]
	ds_read_b64_tr_b16 v[98:99], v162 offset:8192
	ds_read_b64_tr_b16 v[100:101], v162 offset:10240
	ds_read_b64_tr_b16 v[102:103], v162 offset:12288
	ds_read_b64_tr_b16 v[104:105], v162 offset:14336
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[2:17], v[98:101], v[122:125], v[2:17]
	v_lshlrev_b32_e32 v98, 1, v215
	v_add3_u32 v107, s16, v98, v248
	ds_read_b128 v[98:101], v107
	v_mfma_f32_32x32x16_f16 v[2:17], v[102:105], v[118:121], v[2:17]
	v_max3_f32 v102, v106, v86, v87
	v_max3_f32 v102, v102, v88, v89
	v_max3_f32 v126, v102, v90, v91
	v_lshlrev_b32_e32 v102, 1, v212
	v_add3_u32 v127, s16, v102, v248
	ds_read_b128 v[118:121], v107 offset:8192
	ds_read_b128 v[122:125], v127
	v_max3_f32 v126, v126, v92, v93
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[98:113], v[98:101], v[114:117], 0
	v_max3_f32 v126, v126, v94, v95
	v_max3_f32 v162, v126, v96, v97
	v_lshlrev_b32_e32 v126, 1, v211
	v_add3_u32 v163, s16, v126, v248
	ds_read_b128 v[158:161], v127 offset:8192
	ds_read_b128 v[126:129], v163
	v_mov_b32_e32 v164, v162
	s_nop 1
	v_permlane32_swap_b32_e32 v162, v164
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[98:113], v[122:125], v[154:157], v[98:113]
	v_lshlrev_b32_e32 v122, 1, v210
	v_add3_u32 v165, s16, v122, v248
	ds_read_b128 v[210:213], v163 offset:8192
	ds_read_b128 v[122:125], v165
	v_max3_f32 v204, v186, v162, v164
	v_fmac_f32_e32 v192, 0xbe0293ee, v204
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[98:113], v[126:129], v[150:153], v[98:113]
	v_lshlrev_b32_e32 v126, 1, v208
	v_add3_u32 v162, s16, v126, v248
	v_lshlrev_b32_e32 v126, 1, v207
	v_add3_u32 v163, s16, v126, v248
	ds_read_b128 v[214:217], v165 offset:8192
	ds_read_b128 v[126:129], v162
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[98:113], v[122:125], v[138:141], v[98:113]
	v_lshlrev_b32_e32 v122, 1, v206
	v_add3_u32 v164, s16, v122, v248
	v_lshlrev_b32_e32 v122, 1, v205
	v_add3_u32 v166, s16, v122, v248
	ds_read_b128 v[206:209], v162 offset:8192
	ds_read_b128 v[122:125], v163
	ds_read_b128 v[218:221], v163 offset:8192
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[98:113], v[126:129], v[134:137], v[98:113]
	ds_read_b128 v[126:129], v164
	ds_read_b128 v[222:225], v164 offset:8192
	ds_read_b128 v[162:165], v166
	ds_read_b128 v[230:233], v166 offset:8192
	v_add_u32_e32 v166, 0x8000, v227
	s_nop 0
	v_readfirstlane_b32 s3, v166
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v167
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[98:113], v[122:125], v[130:133], v[98:113]
	ds_bpermute_b32 v124, v199, v202
	v_lshrrev_b64 v[122:123], v201, exec
	v_and_b32_e32 v122, 1, v122
	v_lshlrev_b32_e32 v123, 1, v168
	v_cmp_eq_u32_e32 vcc, 1, v122
	v_mul_f32_e32 v201, 0xbe0293ee, v204
	v_fmamk_f32 v66, v66, 0x3e0293ee, v201
	v_cndmask_b32_e32 v122, v198, v123, vcc
	buffer_load_dwordx4 v122, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v122, 1, v124
	v_cndmask_b32_e32 v122, v198, v122, vcc
	s_mov_b32 m0, s3
	v_mfma_f32_32x32x16_f16 v[98:113], v[126:129], v[146:149], v[98:113]
	buffer_load_dwordx4 v122, s[12:15], 0 offen lds
	v_fmamk_f32 v67, v67, 0x3e0293ee, v201
	v_fmamk_f32 v68, v68, 0x3e0293ee, v201
	v_fmamk_f32 v69, v69, 0x3e0293ee, v201
	v_fmamk_f32 v70, v70, 0x3e0293ee, v201
	v_fmamk_f32 v71, v71, 0x3e0293ee, v201
	v_fmamk_f32 v72, v72, 0x3e0293ee, v201
	v_mfma_f32_32x32x16_f16 v[114:129], v[118:121], v[114:117], 0
	v_fmamk_f32 v73, v73, 0x3e0293ee, v201
	v_fmamk_f32 v74, v74, 0x3e0293ee, v201
	v_fmamk_f32 v75, v75, 0x3e0293ee, v201
	v_fmamk_f32 v76, v76, 0x3e0293ee, v201
	v_fmamk_f32 v77, v77, 0x3e0293ee, v201
	v_fmamk_f32 v78, v78, 0x3e0293ee, v201
	v_fmamk_f32 v79, v79, 0x3e0293ee, v201
	v_mfma_f32_32x32x16_f16 v[114:129], v[158:161], v[154:157], v[114:129]
	v_fmamk_f32 v80, v80, 0x3e0293ee, v201
	v_fmamk_f32 v81, v81, 0x3e0293ee, v201
	v_fmamk_f32 v82, v82, 0x3e0293ee, v201
	v_fmamk_f32 v83, v83, 0x3e0293ee, v201
	v_fmamk_f32 v84, v84, 0x3e0293ee, v201
	v_fmamk_f32 v85, v85, 0x3e0293ee, v201
	v_fmamk_f32 v86, v86, 0x3e0293ee, v201
	v_mfma_f32_32x32x16_f16 v[114:129], v[210:213], v[150:153], v[114:129]
	v_fmamk_f32 v87, v87, 0x3e0293ee, v201
	v_fmamk_f32 v88, v88, 0x3e0293ee, v201
	v_fmamk_f32 v89, v89, 0x3e0293ee, v201
	v_fmamk_f32 v90, v90, 0x3e0293ee, v201
	v_fmamk_f32 v91, v91, 0x3e0293ee, v201
	v_fmamk_f32 v92, v92, 0x3e0293ee, v201
	v_fmamk_f32 v93, v93, 0x3e0293ee, v201
	v_mfma_f32_32x32x16_f16 v[114:129], v[214:217], v[138:141], v[114:129]
	v_fmamk_f32 v94, v94, 0x3e0293ee, v201
	v_fmamk_f32 v95, v95, 0x3e0293ee, v201
	v_fmamk_f32 v96, v96, 0x3e0293ee, v201
	v_fmac_f32_e32 v201, 0x3e0293ee, v97
	v_exp_f32_e32 v97, v66
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x16_f16 v[114:129], v[206:209], v[134:137], v[114:129]
	s_barrier
	v_add_u32_e32 v199, s23, v203
	v_add_u32_e32 v154, s34, v195
	v_exp_f32_e32 v202, v75
	v_exp_f32_e32 v205, v76
	v_exp_f32_e32 v206, v77
	v_mfma_f32_32x32x16_f16 v[114:129], v[218:221], v[130:133], v[114:129]
	v_exp_f32_e32 v207, v78
	v_exp_f32_e32 v208, v79
	v_exp_f32_e32 v209, v80
	v_exp_f32_e32 v210, v81
	v_exp_f32_e32 v211, v82
	v_exp_f32_e32 v212, v83
	v_exp_f32_e32 v213, v84
	v_mfma_f32_32x32x16_f16 v[114:129], v[222:225], v[146:149], v[114:129]
	v_exp_f32_e32 v146, v71
	v_exp_f32_e32 v147, v72
	v_exp_f32_e32 v148, v73
	v_exp_f32_e32 v149, v74
	v_exp_f32_e32 v74, v192
	v_cvt_pk_f16_f32 v71, v205, v206
	v_cvt_pk_f16_f32 v72, v207, v208
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[142:145], v[98:113]
	v_add_u32_e32 v162, s34, v200
	v_add_u32_e32 v163, s23, v200
	ds_read_b64_tr_b16 v[186:187], v162 offset:32768
	ds_read_b64_tr_b16 v[188:189], v163 offset:2048
	ds_read_b64_tr_b16 v[182:183], v163 offset:4096
	ds_read_b64_tr_b16 v[184:185], v163 offset:6144
	ds_read_b64_tr_b16 v[178:179], v163 offset:8192
	ds_read_b64_tr_b16 v[180:181], v163 offset:10240
	ds_read_b64_tr_b16 v[174:175], v163 offset:12288
	ds_read_b64_tr_b16 v[176:177], v163 offset:14336
	v_add_u32_e32 v162, s34, v197
	ds_read_b64_tr_b16 v[170:171], v162 offset:32768
	ds_read_b64_tr_b16 v[172:173], v199 offset:2048
	ds_read_b64_tr_b16 v[166:167], v199 offset:4096
	ds_read_b64_tr_b16 v[168:169], v199 offset:6144
	ds_read_b64_tr_b16 v[162:163], v199 offset:8192
	ds_read_b64_tr_b16 v[164:165], v199 offset:10240
	ds_read_b64_tr_b16 v[158:159], v199 offset:12288
	ds_read_b64_tr_b16 v[160:161], v199 offset:14336
	v_add_u32_e32 v199, s23, v196
	ds_read_b64_tr_b16 v[154:155], v154 offset:32768
	ds_read_b64_tr_b16 v[156:157], v199 offset:2048
	ds_read_b64_tr_b16 v[150:151], v199 offset:4096
	ds_read_b64_tr_b16 v[152:153], v199 offset:6144
	v_mfma_f32_32x32x16_f16 v[114:129], v[230:233], v[142:145], v[114:129]
	v_exp_f32_e32 v142, v67
	v_exp_f32_e32 v143, v68
	v_exp_f32_e32 v144, v69
	v_exp_f32_e32 v145, v70
	v_cvt_pk_f16_f32 v66, v97, v142
	v_cvt_pk_f16_f32 v69, v147, v148
	v_cvt_pk_f16_f32 v67, v143, v144
	v_cvt_pk_f16_f32 v68, v145, v146
	v_mul_f32_e32 v50, v50, v74
	v_mul_f32_e32 v51, v51, v74
	v_mul_f32_e32 v52, v52, v74
	v_mul_f32_e32 v53, v53, v74
	v_mul_f32_e32 v54, v54, v74
	v_mul_f32_e32 v55, v55, v74
	v_mul_f32_e32 v56, v56, v74
	v_mul_f32_e32 v57, v57, v74
	v_mul_f32_e32 v58, v58, v74
	v_mul_f32_e32 v59, v59, v74
	v_mul_f32_e32 v60, v60, v74
	v_mul_f32_e32 v61, v61, v74
	v_mul_f32_e32 v62, v62, v74
	v_mul_f32_e32 v63, v63, v74
	v_mul_f32_e32 v64, v64, v74
	v_mul_f32_e32 v65, v65, v74
	v_mul_f32_e32 v18, v18, v74
	v_mul_f32_e32 v19, v19, v74
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[186:189], v[66:69], v[50:65]
	v_mul_f32_e32 v20, v20, v74
	v_mul_f32_e32 v21, v21, v74
	v_mul_f32_e32 v22, v22, v74
	v_mul_f32_e32 v23, v23, v74
	v_mul_f32_e32 v24, v24, v74
	v_mul_f32_e32 v25, v25, v74
	v_mul_f32_e32 v26, v26, v74
	v_mul_f32_e32 v27, v27, v74
	v_mul_f32_e32 v28, v28, v74
	v_mul_f32_e32 v29, v29, v74
	v_mul_f32_e32 v30, v30, v74
	v_mul_f32_e32 v31, v31, v74
	v_mul_f32_e32 v32, v32, v74
	v_mul_f32_e32 v33, v33, v74
	v_cvt_pk_f16_f32 v70, v149, v202
	v_cvt_pk_f16_f32 v73, v209, v210
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[154:157], v[66:69], v[18:33]
	v_exp_f32_e32 v214, v85
	v_exp_f32_e32 v215, v86
	v_exp_f32_e32 v216, v87
	v_exp_f32_e32 v217, v88
	v_exp_f32_e32 v186, v89
	v_mul_f32_e32 v34, v34, v74
	v_mul_f32_e32 v35, v35, v74
	v_mfma_f32_32x32x16_f16 v[50:65], v[182:185], v[70:73], v[50:65]
	v_mul_f32_e32 v36, v36, v74
	v_mul_f32_e32 v37, v37, v74
	v_mul_f32_e32 v38, v38, v74
	v_mul_f32_e32 v39, v39, v74
	v_mul_f32_e32 v40, v40, v74
	v_mul_f32_e32 v41, v41, v74
	v_mul_f32_e32 v42, v42, v74
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[150:153], v[70:73], v[18:33]
	v_mul_f32_e32 v43, v43, v74
	v_mul_f32_e32 v44, v44, v74
	v_mul_f32_e32 v45, v45, v74
	v_mul_f32_e32 v46, v46, v74
	v_mul_f32_e32 v47, v47, v74
	v_mul_f32_e32 v48, v48, v74
	v_mul_f32_e32 v49, v49, v74
	v_add_f32_e32 v97, v97, v142
	v_exp_f32_e32 v187, v90
	v_exp_f32_e32 v188, v91
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[66:69], v[34:49]
	ds_read_b64_tr_b16 v[84:85], v199 offset:8192
	ds_read_b64_tr_b16 v[86:87], v199 offset:10240
	ds_read_b64_tr_b16 v[88:89], v199 offset:12288
	ds_read_b64_tr_b16 v[90:91], v199 offset:14336
	v_add_f32_e32 v97, v143, v97
	v_cvt_pk_f16_f32 v76, v211, v212
	v_cvt_pk_f16_f32 v77, v213, v214
	v_cvt_pk_f16_f32 v78, v215, v216
	v_cvt_pk_f16_f32 v79, v217, v186
	v_add_u32_e32 v75, s34, v193
	v_add_f32_e32 v97, v144, v97
	v_exp_f32_e32 v182, v92
	v_exp_f32_e32 v183, v93
	v_exp_f32_e32 v184, v94
	v_mfma_f32_32x32x16_f16 v[50:65], v[178:181], v[76:79], v[50:65]
	v_exp_f32_e32 v178, v95
	v_add_u32_e32 v140, s23, v194
	ds_read_b64_tr_b16 v[92:93], v75 offset:32768
	ds_read_b64_tr_b16 v[94:95], v140 offset:2048
	ds_read_b64_tr_b16 v[130:131], v140 offset:4096
	ds_read_b64_tr_b16 v[132:133], v140 offset:6144
	v_add_f32_e32 v97, v145, v97
	v_mul_f32_e32 v2, v2, v74
	v_mul_f32_e32 v3, v3, v74
	v_mul_f32_e32 v4, v4, v74
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[84:87], v[76:79], v[18:33]
	v_add_f32_e32 v84, v146, v97
	v_add_f32_e32 v84, v147, v84
	v_mul_f32_e32 v5, v5, v74
	v_mul_f32_e32 v6, v6, v74
	v_mul_f32_e32 v7, v7, v74
	v_mul_f32_e32 v8, v8, v74
	v_mul_f32_e32 v9, v9, v74
	v_mul_f32_e32 v10, v10, v74
	v_mul_f32_e32 v11, v11, v74
	v_mul_f32_e32 v12, v12, v74
	v_mul_f32_e32 v13, v13, v74
	v_mul_f32_e32 v14, v14, v74
	v_mul_f32_e32 v15, v15, v74
	v_mul_f32_e32 v16, v16, v74
	v_mul_f32_e32 v17, v17, v74
	v_add_f32_e32 v84, v148, v84
	v_add_f32_e32 v84, v149, v84
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[92:95], v[66:69], v[2:17]
	v_add_f32_e32 v84, v202, v84
	v_max_f32_e32 v75, v99, v99
	v_add_f32_e32 v84, v205, v84
	v_add_f32_e32 v84, v206, v84
	v_add_f32_e32 v66, v207, v84
	v_add_f32_e32 v66, v208, v66
	v_add_f32_e32 v66, v209, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[70:73], v[34:49]
	v_max_f32_e32 v166, v98, v98
	v_max_f32_e32 v75, v166, v75
	v_max3_f32 v75, v75, v100, v101
	v_max3_f32 v75, v75, v102, v103
	v_max3_f32 v75, v75, v104, v105
	v_max3_f32 v75, v75, v106, v107
	v_add_f32_e32 v66, v210, v66
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[130:133], v[70:73], v[2:17]
	v_max3_f32 v75, v75, v108, v109
	v_add_f32_e32 v66, v211, v66
	v_max3_f32 v75, v75, v110, v111
	v_add_f32_e32 v66, v212, v66
	v_max3_f32 v75, v75, v112, v113
	v_add_f32_e32 v66, v213, v66
	v_max3_f32 v75, v75, v114, v115
	v_add_f32_e32 v66, v214, v66
	ds_read_b64_tr_b16 v[134:135], v140 offset:8192
	ds_read_b64_tr_b16 v[136:137], v140 offset:10240
	ds_read_b64_tr_b16 v[138:139], v140 offset:12288
	ds_read_b64_tr_b16 v[140:141], v140 offset:14336
	v_max3_f32 v75, v75, v116, v117
	v_add_f32_e32 v66, v215, v66
	v_max3_f32 v75, v75, v118, v119
	v_add_f32_e32 v66, v216, v66
	v_max3_f32 v75, v75, v120, v121
	v_add_f32_e32 v66, v217, v66
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v179, v201
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[76:79], v[34:49]
	v_max3_f32 v75, v75, v122, v123
	v_add_f32_e32 v66, v186, v66
	v_max3_f32 v75, v75, v124, v125
	v_add_f32_e32 v66, v187, v66
	v_max3_f32 v75, v75, v126, v127
	v_add_f32_e32 v66, v188, v66
	v_max3_f32 v75, v75, v128, v129
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[134:137], v[76:79], v[2:17]
	v_add_f32_e32 v66, v182, v66
	v_cvt_pk_f16_f32 v80, v187, v188
	v_cvt_pk_f16_f32 v81, v182, v183
	v_cvt_pk_f16_f32 v82, v184, v178
	v_cvt_pk_f16_f32 v83, v96, v179
	v_mov_b32_e32 v154, v75
	v_add_f32_e32 v66, v183, v66
	v_mfma_f32_32x32x16_f16 v[50:65], v[174:177], v[80:83], v[50:65]
	v_permlane32_swap_b32_e32 v75, v154
	v_add_f32_e32 v66, v184, v66
	v_max3_f32 v75, v204, v75, v154
	v_add_f32_e32 v66, v178, v66
	v_mul_f32_e32 v150, 0xbe0293ee, v75
	v_add_f32_e32 v66, v96, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[158:161], v[80:83], v[34:49]
	v_add_f32_e32 v96, v179, v66
	v_fmamk_f32 v66, v98, 0x3e0293ee, v150
	v_fmamk_f32 v67, v99, 0x3e0293ee, v150
	v_fmamk_f32 v68, v100, 0x3e0293ee, v150
	v_fmamk_f32 v69, v101, 0x3e0293ee, v150
	v_fmamk_f32 v70, v102, 0x3e0293ee, v150
	v_fmamk_f32 v71, v103, 0x3e0293ee, v150
	v_mfma_f32_32x32x16_f16 v[18:33], v[88:91], v[80:83], v[18:33]
	v_fmamk_f32 v72, v104, 0x3e0293ee, v150
	v_fmamk_f32 v73, v105, 0x3e0293ee, v150
	v_fmamk_f32 v76, v106, 0x3e0293ee, v150
	v_fmamk_f32 v77, v107, 0x3e0293ee, v150
	v_fmamk_f32 v78, v108, 0x3e0293ee, v150
	v_fmamk_f32 v79, v109, 0x3e0293ee, v150
	v_fmamk_f32 v84, v114, 0x3e0293ee, v150
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[138:141], v[80:83], v[2:17]
	v_fmamk_f32 v80, v110, 0x3e0293ee, v150
	v_fmamk_f32 v81, v111, 0x3e0293ee, v150
	v_fmamk_f32 v82, v112, 0x3e0293ee, v150
	v_fmamk_f32 v83, v113, 0x3e0293ee, v150
	v_fmamk_f32 v85, v115, 0x3e0293ee, v150
	v_fmamk_f32 v86, v116, 0x3e0293ee, v150
	v_fmamk_f32 v87, v117, 0x3e0293ee, v150
	v_fmamk_f32 v88, v118, 0x3e0293ee, v150
	v_fmamk_f32 v89, v119, 0x3e0293ee, v150
	v_fmamk_f32 v90, v120, 0x3e0293ee, v150
	v_fmamk_f32 v91, v121, 0x3e0293ee, v150
	v_fmamk_f32 v92, v122, 0x3e0293ee, v150
	v_fmamk_f32 v93, v123, 0x3e0293ee, v150
	v_fmamk_f32 v94, v124, 0x3e0293ee, v150
	v_fmamk_f32 v95, v125, 0x3e0293ee, v150
	v_fmamk_f32 v98, v126, 0x3e0293ee, v150
	v_fmamk_f32 v99, v127, 0x3e0293ee, v150
	v_fmamk_f32 v100, v128, 0x3e0293ee, v150
	v_fmamk_f32 v101, v129, 0x3e0293ee, v150
	v_fmac_f32_e32 v150, 0x3e0293ee, v204
	v_exp_f32_e32 v102, v66
	v_exp_f32_e32 v103, v67
	v_exp_f32_e32 v104, v68
	v_exp_f32_e32 v105, v69
	v_exp_f32_e32 v106, v70
	v_exp_f32_e32 v107, v71
	v_exp_f32_e32 v108, v72
	v_exp_f32_e32 v109, v73
	v_exp_f32_e32 v130, v150
	v_add_u32_e32 v67, s16, v200
	v_exp_f32_e32 v110, v76
	v_exp_f32_e32 v111, v77
	v_exp_f32_e32 v112, v78
	v_exp_f32_e32 v113, v79
	v_exp_f32_e32 v114, v80
	v_exp_f32_e32 v115, v81
	v_exp_f32_e32 v116, v82
	v_exp_f32_e32 v117, v83
	v_exp_f32_e32 v124, v90
	s_waitcnt vmcnt(0)
	s_barrier
	v_add_u32_e32 v90, s2, v200
	ds_read_b64_tr_b16 v[76:77], v67 offset:32768
	ds_read_b64_tr_b16 v[78:79], v90 offset:2048
	ds_read_b64_tr_b16 v[80:81], v90 offset:4096
	ds_read_b64_tr_b16 v[82:83], v90 offset:6144
	v_cvt_pk_f16_f32 v70, v102, v103
	v_cvt_pk_f16_f32 v71, v104, v105
	v_cvt_pk_f16_f32 v72, v106, v107
	v_cvt_pk_f16_f32 v73, v108, v109
	v_mul_f32_e32 v50, v50, v130
	v_mul_f32_e32 v51, v51, v130
	v_mul_f32_e32 v52, v52, v130
	v_mul_f32_e32 v53, v53, v130
	v_mul_f32_e32 v54, v54, v130
	v_mul_f32_e32 v55, v55, v130
	v_mul_f32_e32 v56, v56, v130
	v_mul_f32_e32 v57, v57, v130
	v_mul_f32_e32 v58, v58, v130
	v_mul_f32_e32 v59, v59, v130
	v_mul_f32_e32 v60, v60, v130
	v_mul_f32_e32 v61, v61, v130
	v_mul_f32_e32 v62, v62, v130
	v_mul_f32_e32 v63, v63, v130
	v_mul_f32_e32 v64, v64, v130
	v_mul_f32_e32 v65, v65, v130
	v_cvt_pk_f16_f32 v66, v110, v111
	v_cvt_pk_f16_f32 v67, v112, v113
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[76:79], v[70:73], v[50:65]
	v_cvt_pk_f16_f32 v68, v114, v115
	v_cvt_pk_f16_f32 v69, v116, v117
	v_exp_f32_e32 v118, v84
	v_exp_f32_e32 v119, v85
	v_exp_f32_e32 v120, v86
	v_exp_f32_e32 v121, v87
	v_exp_f32_e32 v122, v88
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[80:83], v[66:69], v[50:65]
	v_exp_f32_e32 v123, v89
	v_exp_f32_e32 v125, v91
	ds_read_b64_tr_b16 v[84:85], v90 offset:8192
	ds_read_b64_tr_b16 v[86:87], v90 offset:10240
	v_cvt_pk_f16_f32 v76, v118, v119
	v_cvt_pk_f16_f32 v77, v120, v121
	v_cvt_pk_f16_f32 v78, v122, v123
	v_cvt_pk_f16_f32 v79, v124, v125
	v_exp_f32_e32 v126, v92
	v_exp_f32_e32 v127, v93
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[84:87], v[76:79], v[50:65]
	v_add_u32_e32 v84, s16, v197
	v_exp_f32_e32 v128, v94
	v_exp_f32_e32 v129, v95
	ds_read_b64_tr_b16 v[88:89], v90 offset:12288
	ds_read_b64_tr_b16 v[90:91], v90 offset:14336
	v_add_u32_e32 v131, s2, v203
	ds_read_b64_tr_b16 v[84:85], v84 offset:32768
	ds_read_b64_tr_b16 v[86:87], v131 offset:2048
	ds_read_b64_tr_b16 v[92:93], v131 offset:4096
	ds_read_b64_tr_b16 v[94:95], v131 offset:6144
	v_mul_f32_e32 v34, v34, v130
	v_mul_f32_e32 v35, v35, v130
	v_mul_f32_e32 v36, v36, v130
	v_mul_f32_e32 v37, v37, v130
	v_mul_f32_e32 v38, v38, v130
	v_mul_f32_e32 v39, v39, v130
	v_mul_f32_e32 v40, v40, v130
	v_mul_f32_e32 v41, v41, v130
	v_mul_f32_e32 v42, v42, v130
	v_mul_f32_e32 v43, v43, v130
	v_mul_f32_e32 v44, v44, v130
	v_mul_f32_e32 v45, v45, v130
	v_mul_f32_e32 v46, v46, v130
	v_mul_f32_e32 v47, v47, v130
	v_mul_f32_e32 v48, v48, v130
	v_mul_f32_e32 v49, v49, v130
	v_exp_f32_e32 v98, v98
	v_exp_f32_e32 v99, v99
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[84:87], v[70:73], v[34:49]
	ds_read_b64_tr_b16 v[84:85], v131 offset:8192
	ds_read_b64_tr_b16 v[86:87], v131 offset:10240
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v101, v101
	v_mov_b32_e32 v97, v96
	v_cvt_pk_f16_f32 v80, v126, v127
	v_cvt_pk_f16_f32 v81, v128, v129
	v_cvt_pk_f16_f32 v82, v98, v99
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[92:95], v[66:69], v[34:49]
	v_cvt_pk_f16_f32 v83, v100, v101
	v_permlane32_swap_b32_e32 v96, v97
	v_add_f32_e32 v96, v96, v97
	v_add_u32_e32 v97, s2, v196
	v_mul_f32_e32 v18, v18, v130
	v_mul_f32_e32 v19, v19, v130
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[84:87], v[76:79], v[34:49]
	v_add_u32_e32 v84, s16, v195
	v_mul_f32_e32 v20, v20, v130
	v_mul_f32_e32 v21, v21, v130
	v_mul_f32_e32 v22, v22, v130
	v_mul_f32_e32 v23, v23, v130
	v_mul_f32_e32 v24, v24, v130
	v_mul_f32_e32 v25, v25, v130
	v_mfma_f32_32x32x16_f16 v[50:65], v[88:91], v[80:83], v[50:65]
	ds_read_b64_tr_b16 v[88:89], v131 offset:12288
	ds_read_b64_tr_b16 v[90:91], v131 offset:14336
	ds_read_b64_tr_b16 v[84:85], v84 offset:32768
	ds_read_b64_tr_b16 v[86:87], v97 offset:2048
	ds_read_b64_tr_b16 v[92:93], v97 offset:4096
	ds_read_b64_tr_b16 v[94:95], v97 offset:6144
	v_mul_f32_e32 v26, v26, v130
	v_mul_f32_e32 v27, v27, v130
	v_mul_f32_e32 v28, v28, v130
	v_mul_f32_e32 v29, v29, v130
	v_mul_f32_e32 v30, v30, v130
	v_mul_f32_e32 v31, v31, v130
	v_mul_f32_e32 v32, v32, v130
	v_mul_f32_e32 v33, v33, v130
	v_fmac_f32_e32 v96, v191, v74
	v_add_f32_e32 v74, v102, v103
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[84:87], v[70:73], v[18:33]
	v_add_f32_e32 v74, v104, v74
	v_add_f32_e32 v74, v105, v74
	v_add_f32_e32 v74, v106, v74
	ds_read_b64_tr_b16 v[84:85], v97 offset:8192
	ds_read_b64_tr_b16 v[86:87], v97 offset:10240
	v_add_f32_e32 v74, v107, v74
	v_add_f32_e32 v74, v108, v74
	v_add_f32_e32 v74, v109, v74
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[92:95], v[66:69], v[18:33]
	v_add_f32_e32 v74, v110, v74
	v_add_f32_e32 v74, v111, v74
	v_add_f32_e32 v74, v112, v74
	v_add_f32_e32 v74, v113, v74
	v_add_f32_e32 v74, v114, v74
	v_mul_f32_e32 v2, v2, v130
	v_mul_f32_e32 v3, v3, v130
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[84:87], v[76:79], v[18:33]
	v_add_u32_e32 v84, s16, v193
	v_mul_f32_e32 v4, v4, v130
	v_mul_f32_e32 v5, v5, v130
	v_mul_f32_e32 v6, v6, v130
	v_mul_f32_e32 v7, v7, v130
	v_mul_f32_e32 v8, v8, v130
	v_mul_f32_e32 v9, v9, v130
	v_mfma_f32_32x32x16_f16 v[34:49], v[88:91], v[80:83], v[34:49]
	ds_read_b64_tr_b16 v[88:89], v97 offset:12288
	ds_read_b64_tr_b16 v[90:91], v97 offset:14336
	v_add_u32_e32 v97, s2, v194
	ds_read_b64_tr_b16 v[84:85], v84 offset:32768
	ds_read_b64_tr_b16 v[86:87], v97 offset:2048
	ds_read_b64_tr_b16 v[92:93], v97 offset:4096
	ds_read_b64_tr_b16 v[94:95], v97 offset:6144
	v_mul_f32_e32 v10, v10, v130
	v_mul_f32_e32 v11, v11, v130
	v_mul_f32_e32 v12, v12, v130
	v_mul_f32_e32 v13, v13, v130
	v_mul_f32_e32 v14, v14, v130
	v_mul_f32_e32 v15, v15, v130
	v_mul_f32_e32 v16, v16, v130
	v_mul_f32_e32 v17, v17, v130
	v_add_f32_e32 v74, v115, v74
	v_add_f32_e32 v74, v116, v74
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[84:87], v[70:73], v[2:17]
	v_add_f32_e32 v74, v117, v74
	v_add_f32_e32 v74, v118, v74
	v_add_f32_e32 v74, v119, v74
	v_add_f32_e32 v74, v120, v74
	v_add_f32_e32 v74, v121, v74
	v_add_f32_e32 v70, v122, v74
	v_add_f32_e32 v70, v123, v70
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[92:95], v[66:69], v[2:17]
	v_add_f32_e32 v70, v124, v70
	v_add_f32_e32 v70, v125, v70
	v_add_f32_e32 v74, v126, v70
	ds_read_b64_tr_b16 v[70:71], v97 offset:8192
	ds_read_b64_tr_b16 v[72:73], v97 offset:10240
	ds_read_b64_tr_b16 v[84:85], v97 offset:12288
	ds_read_b64_tr_b16 v[86:87], v97 offset:14336
	v_add_f32_e32 v66, v127, v74
	v_add_f32_e32 v66, v128, v66
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[70:73], v[76:79], v[2:17]
	v_add_f32_e32 v66, v129, v66
	v_add_f32_e32 v66, v98, v66
	v_add_f32_e32 v66, v99, v66
	v_add_f32_e32 v66, v100, v66
	v_add_f32_e32 v66, v101, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_mfma_f32_32x32x16_f16 v[18:33], v[88:91], v[80:83], v[18:33]
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v96, v130
	v_lshl_add_u32 v67, v190, 2, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[2:17], v[84:87], v[80:83], v[2:17]
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v66
	v_mov_b32_e32 v69, 0x42000000
	v_or_b32_e32 v68, s28, v190
	v_cndmask_b32_e64 v70, 0, 32, vcc
	v_ldexp_f32 v70, v66, v70
	v_log_f32_e32 v70, v70
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v69, 0, v69, vcc
	v_cmp_gt_i32_e64 s[8:9], s2, v68
	v_sub_f32_e32 v68, v70, v69
	v_add_f32_e32 v68, v75, v68
	ds_write_b32 v67, v68
	v_mov_b32_e32 v68, 2
	v_lshlrev_b32_sdwa v68, v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v69, 0, v68
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v69, v69
	s_sub_i32 s2, 0x4000, s28
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	s_and_b64 vcc, s[0:1], s[2:3]
	s_and_b32 s5, s19, 0xffff
	s_mov_b32 s6, s14
	s_mov_b32 s7, s15
	v_cndmask_b32_e32 v68, v198, v68, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v69, v68, s[4:7], 0 offen
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_9:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v66
	v_mov_b32_e32 v68, 0x42000000
	s_and_b32 s5, s19, 0xffff
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v66, v69
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v68, 0, v68, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_sub_f32_e32 v68, v69, v68
	v_add_f32_e32 v68, v75, v68
	ds_write_b32 v67, v68
	v_mov_b32_e32 v67, 2
	v_lshlrev_b32_sdwa v0, v67, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v67, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v67, v67
	v_bfrev_b32_e32 v68, 1
	v_cndmask_b32_e64 v0, v68, v0, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v0, s[4:7], 0 offen
.LBB0_10:                               ; %.critedge
	v_div_scale_f32 v0, s[0:1], v66, v66, 1.0
	v_rcp_f32_e32 v0, v0
	v_div_scale_f32 v67, vcc, 1.0, v66, 1.0
	s_mul_i32 s0, s25, s18
	v_mul_f32_e32 v0, v67, v0
	s_ashr_i32 s1, s0, 31
	s_nop 0
	v_div_fmas_f32 v0, 0, 0, v0
	v_div_fixup_f32 v0, v0, v66, 1.0
	v_pk_mul_f32 v[8:9], v[0:1], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[0:1], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[0:1], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v9, v8, v9
	v_cvt_pk_f16_f32 v8, v6, v7
	v_cvt_pk_f16_f32 v5, v4, v5
	v_cvt_pk_f16_f32 v4, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[0:1], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v13, v12, v13
	v_cvt_pk_f16_f32 v12, v10, v11
	v_cvt_pk_f16_f32 v3, v2, v3
	v_cvt_pk_f16_f32 v2, v6, v7
	v_pk_mul_f32 v[6:7], v[0:1], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v17, v16, v17
	v_cvt_pk_f16_f32 v16, v14, v15
	v_cvt_pk_f16_f32 v7, v6, v7
	v_cvt_pk_f16_f32 v6, v10, v11
	v_pk_mul_f32 v[10:11], v[0:1], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[0:1], v[22:23] op_sel_hi:[0,1]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v11, v10, v11
	v_cvt_pk_f16_f32 v10, v14, v15
	v_pk_mul_f32 v[14:15], v[0:1], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s26, s17
	v_cvt_pk_f16_f32 v15, v14, v15
	v_cvt_pk_f16_f32 v14, v18, v19
	v_pk_mul_f32 v[18:19], v[0:1], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[0:1], v[46:47] op_sel_hi:[0,1]
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	v_cvt_pk_f16_f32 v19, v18, v19
	v_cvt_pk_f16_f32 v18, v20, v21
	v_pk_mul_f32 v[20:21], v[0:1], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[0:1], v[42:43] op_sel_hi:[0,1]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v21, v20, v21
	v_cvt_pk_f16_f32 v20, v22, v23
	v_pk_mul_f32 v[22:23], v[0:1], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[0:1], v[38:39] op_sel_hi:[0,1]
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s27, s28
	v_cvt_pk_f16_f32 v23, v22, v23
	v_cvt_pk_f16_f32 v22, v24, v25
	v_pk_mul_f32 v[24:25], v[0:1], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[0:1], v[34:35] op_sel_hi:[0,1]
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_cvt_pk_f16_f32 v25, v24, v25
	v_cvt_pk_f16_f32 v24, v26, v27
	v_pk_mul_f32 v[26:27], v[0:1], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[0:1], v[62:63] op_sel_hi:[0,1]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v27, v26, v27
	v_cvt_pk_f16_f32 v26, v28, v29
	v_pk_mul_f32 v[28:29], v[0:1], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[0:1], v[58:59] op_sel_hi:[0,1]
	s_add_u32 s0, s2, s0
	v_cvt_pk_f16_f32 v29, v28, v29
	v_cvt_pk_f16_f32 v28, v30, v31
	v_pk_mul_f32 v[30:31], v[0:1], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[0:1], v[54:55] op_sel_hi:[0,1]
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s27, 0x3fff
	v_cvt_pk_f16_f32 v31, v30, v31
	v_cvt_pk_f16_f32 v30, v32, v33
	v_pk_mul_f32 v[32:33], v[0:1], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[0:1], v[50:51] op_sel_hi:[0,1]
	v_mul_lo_u32 v0, s27, v190
	s_bitset1_b32 s2, 14
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_add_lshl_u32 v0, v0, v1, 1
	v_bfrev_b32_e32 v1, 1
	v_cvt_pk_f16_f32 v33, v32, v33
	v_cvt_pk_f16_f32 v32, v34, v35
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e64 v34, v1, v0, s[8:9]
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_add_u32_e32 v32, 16, v0
	v_cndmask_b32_e64 v32, v1, v32, s[8:9]
	buffer_store_dwordx2 v[30:31], v32, s[0:3], 0 offen
	v_add_u32_e32 v30, 32, v0
	v_cndmask_b32_e64 v30, v1, v30, s[8:9]
	buffer_store_dwordx2 v[28:29], v30, s[0:3], 0 offen
	v_add_u32_e32 v28, 48, v0
	v_cndmask_b32_e64 v28, v1, v28, s[8:9]
	buffer_store_dwordx2 v[26:27], v28, s[0:3], 0 offen
	v_add_u32_e32 v26, 64, v0
	v_cndmask_b32_e64 v26, v1, v26, s[8:9]
	buffer_store_dwordx2 v[24:25], v26, s[0:3], 0 offen
	v_add_u32_e32 v24, 0x50, v0
	v_cndmask_b32_e64 v24, v1, v24, s[8:9]
	buffer_store_dwordx2 v[22:23], v24, s[0:3], 0 offen
	v_add_u32_e32 v22, 0x60, v0
	v_cndmask_b32_e64 v22, v1, v22, s[8:9]
	buffer_store_dwordx2 v[20:21], v22, s[0:3], 0 offen
	v_add_u32_e32 v20, 0x70, v0
	v_cndmask_b32_e64 v20, v1, v20, s[8:9]
	buffer_store_dwordx2 v[18:19], v20, s[0:3], 0 offen
	v_add_u32_e32 v18, 0x80, v0
	v_cndmask_b32_e64 v18, v1, v18, s[8:9]
	buffer_store_dwordx2 v[14:15], v18, s[0:3], 0 offen
	v_add_u32_e32 v14, 0x90, v0
	v_cndmask_b32_e64 v14, v1, v14, s[8:9]
	buffer_store_dwordx2 v[10:11], v14, s[0:3], 0 offen
	v_add_u32_e32 v10, 0xa0, v0
	v_cndmask_b32_e64 v10, v1, v10, s[8:9]
	buffer_store_dwordx2 v[6:7], v10, s[0:3], 0 offen
	v_add_u32_e32 v6, 0xb0, v0
	v_cndmask_b32_e64 v6, v1, v6, s[8:9]
	buffer_store_dwordx2 v[2:3], v6, s[0:3], 0 offen
	v_add_u32_e32 v2, 0xc0, v0
	v_cndmask_b32_e64 v2, v1, v2, s[8:9]
	buffer_store_dwordx2 v[4:5], v2, s[0:3], 0 offen
	v_add_u32_e32 v2, 0xd0, v0
	v_cndmask_b32_e64 v2, v1, v2, s[8:9]
	buffer_store_dwordx2 v[8:9], v2, s[0:3], 0 offen
	v_add_u32_e32 v2, 0xe0, v0
	v_add_u32_e32 v0, 0xf0, v0
	v_cndmask_b32_e64 v2, v1, v2, s[8:9]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[12:13], v2, s[0:3], 0 offen
	buffer_store_dwordx2 v[16:17], v0, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
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
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 249
		.amdhsa_next_free_sgpr 42
		.amdhsa_accum_offset 252
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
	.set attn_fwd.num_vgpr, 249
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 42
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12168
; TotalNumSgprs: 48
; NumVgprs: 249
; NumAgprs: 0
; TotalNumVgprs: 249
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 48
; NumVGPRsForWavesPerEU: 249
; AccumOffset: 252
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 62
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
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"flash-attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton/fa" ; string offset=26
.Linfo_string3:
	.asciz	"attn_fwd"                      ; string offset=57
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
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     249
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
