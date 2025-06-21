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
	.file	1 "/app/OAI-triton/fa" "flash-attention.py"
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
	s_lshl_b32 s13, s14, 5
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v35, 3, v0
	v_lshrrev_b32_e32 v38, 4, v0
	s_add_u32 s0, s2, s0
	v_and_b32_e32 v34, 0x78, v35
	s_mul_i32 s30, s15, s18
	s_addc_u32 s1, s3, s1
	v_mad_u64_u32 v[2:3], s[2:3], s14, v38, v[34:35]
	s_ashr_i32 s31, s30, 31
	s_lshl_b64 s[2:3], s[30:31], 1
	s_add_u32 s12, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s34, s20, s17
	s_addc_u32 s15, s5, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 1
	s_add_u32 s20, s12, s2
	s_mul_i32 s2, s22, s18
	s_addc_u32 s15, s15, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s23, s17
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s6, s2
	s_addc_u32 s16, s7, s3
	s_and_b32 s2, s14, 0x3fff
	v_or_b32_e32 v1, s28, v38
	s_movk_i32 s6, 0x4000
	s_bitset1_b32 s2, 14
	v_or_b32_e32 v4, 32, v1
	v_add_u32_e32 v3, s13, v2
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v39, 1
	v_cmp_gt_i32_e32 vcc, s6, v1
	v_or_b32_e32 v10, 64, v1
	v_or_b32_e32 v11, 0x60, v1
	v_or_b32_e32 v18, 0x80, v1
	v_or_b32_e32 v19, 0xa0, v1
	v_or_b32_e32 v26, 0xc0, v1
	v_or_b32_e32 v27, 0xe0, v1
	v_add_u32_e32 v12, s13, v3
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v1, v39, v2, vcc
	v_lshlrev_b32_e32 v2, 1, v3
	v_cmp_gt_i32_e32 vcc, s6, v4
	v_add_u32_e32 v13, s13, v12
	v_add_u32_e32 v20, s13, v13
	v_cndmask_b32_e32 v14, v39, v2, vcc
	buffer_load_dwordx4 v[2:5], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v14, s[0:3], 0 offen
	v_lshlrev_b32_e32 v1, 1, v12
	v_cmp_gt_i32_e32 vcc, s6, v10
	v_lshlrev_b32_e32 v10, 1, v13
	v_add_u32_e32 v21, s13, v20
	v_cndmask_b32_e32 v1, v39, v1, vcc
	v_cmp_gt_i32_e32 vcc, s6, v11
	v_add_u32_e32 v28, s13, v21
	v_lshlrev_b32_e32 v188, 7, v38
	v_cndmask_b32_e32 v22, v39, v10, vcc
	buffer_load_dwordx4 v[10:13], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v22, s[0:3], 0 offen
	v_lshlrev_b32_e32 v1, 1, v20
	v_cmp_gt_i32_e32 vcc, s6, v18
	v_lshlrev_b32_e32 v18, 1, v21
	v_and_b32_e32 v37, 31, v0
	v_cndmask_b32_e32 v1, v39, v1, vcc
	v_cmp_gt_i32_e32 vcc, s6, v19
	v_or_b32_e32 v191, v188, v34
	v_lshlrev_b32_e32 v192, 1, v191
	v_cndmask_b32_e32 v29, v39, v18, vcc
	buffer_load_dwordx4 v[18:21], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[22:25], v29, s[0:3], 0 offen
	v_lshlrev_b32_e32 v1, 1, v28
	v_cmp_gt_i32_e32 vcc, s6, v26
	v_add_lshl_u32 v26, v28, s13, 1
	s_movk_i32 s13, 0x78
	v_cndmask_b32_e32 v1, v39, v1, vcc
	v_cmp_gt_i32_e32 vcc, s6, v27
	s_movk_i32 s6, 0xe0
	v_and_b32_e32 v186, 63, v0
	v_cndmask_b32_e32 v36, v39, v26, vcc
	buffer_load_dwordx4 v[26:29], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v36, s[0:3], 0 offen
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v36, 0x78, v1
	v_bitop3_b32 v36, v36, v188, v34 bitop3:0xde
	v_lshlrev_b32_e32 v41, 1, v36
	v_add_u32_e32 v36, 0, v41
	s_barrier
	v_bitop3_b32 v40, v1, v35, s13 bitop3:0x28
	v_and_or_b32 v1, v1, s6, v37
	v_mad_u64_u32 v[176:177], s[0:1], s21, v38, v[34:35]
	v_lshl_add_u32 v187, s21, 5, v176
	s_and_b32 s0, s21, 0x3fff
	s_lshl_b32 s36, s21, 6
	v_add_u32_e32 v68, 0, v192
	s_bitset1_b32 s0, 14
	s_and_b32 s1, s15, 0xffff
	s_lshl_b32 s19, s0, 16
	v_readfirstlane_b32 s29, v68
	s_ashr_i32 s37, s36, 31
	s_lshl_b32 s38, s24, 6
	s_or_b32 s21, s1, s19
	s_mov_b32 s22, s2
	s_mov_b32 s23, s3
	s_mov_b32 m0, s29
	s_lshl_b64 s[6:7], s[36:37], 1
	s_add_u32 s0, s20, s6
	s_waitcnt vmcnt(7)
	ds_write_b128 v36, v[2:5]
	v_or_b32_e32 v2, 0x1000, v188
	v_or_b32_e32 v3, v2, v40
	v_lshlrev_b32_e32 v4, 1, v3
	v_add_u32_e32 v3, 0, v4
	s_waitcnt vmcnt(6)
	ds_write_b128 v3, v[6:9]
	v_lshlrev_b32_e32 v3, 1, v40
	v_lshlrev_b32_e32 v5, 8, v38
	v_add3_u32 v3, 0, v3, v5
	v_lshrrev_b32_e32 v36, 2, v0
	s_waitcnt vmcnt(5)
	ds_write_b128 v3, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v3, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v3, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v3, v[22:25] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v3, v[26:29] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v3, v[30:33] offset:57344
	v_and_b32_e32 v3, 8, v36
	v_or_b32_e32 v5, 16, v3
	v_or_b32_e32 v6, 32, v3
	v_and_b32_e32 v11, 15, v0
	v_lshrrev_b32_e32 v12, 5, v0
	v_or_b32_e32 v7, 48, v3
	v_or_b32_e32 v8, 64, v3
	v_bitop3_b32 v11, v12, v11, 1 bitop3:0x6c
	v_lshrrev_b32_e32 v5, 3, v5
	v_lshrrev_b32_e32 v6, 3, v6
	v_or_b32_e32 v9, 0x50, v3
	v_bitop3_b32 v5, v5, v0, 15 bitop3:0x78
	v_bitop3_b32 v6, v6, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v7, 3, v7
	v_lshrrev_b32_e32 v8, 3, v8
	v_lshl_add_u32 v12, v1, 8, 0
	v_lshlrev_b32_e32 v183, 4, v11
	v_or_b32_e32 v2, v2, v34
	v_or_b32_e32 v10, 0x60, v3
	v_or_b32_e32 v3, 0x70, v3
	v_bitop3_b32 v7, v7, v0, 15 bitop3:0x78
	v_bitop3_b32 v8, v8, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v9
	v_add_u32_e32 v1, v12, v183
	v_lshlrev_b32_e32 v182, 4, v5
	v_lshlrev_b32_e32 v181, 4, v6
	v_lshlrev_b32_e32 v193, 1, v2
	v_sub_u32_e32 v2, v41, v192
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_bitop3_b32 v9, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v3, 3, v3
	v_add_u32_e32 v5, v12, v182
	ds_read_b128 v[138:141], v1
	ds_read_b128 v[134:137], v5
	v_add_u32_e32 v1, v12, v181
	v_lshlrev_b32_e32 v180, 4, v7
	v_lshlrev_b32_e32 v179, 4, v8
	v_ashrrev_i16_e32 v2, 4, v2
	v_bitop3_b32 v3, v3, v0, 15 bitop3:0x78
	v_add_u32_e32 v5, v12, v180
	ds_read_b128 v[126:129], v1
	ds_read_b128 v[122:125], v5
	v_add_u32_e32 v1, v12, v179
	v_lshlrev_b32_e32 v178, 4, v9
	v_add_u32_sdwa v2, v186, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v5, v12, v178
	ds_read_b128 v[118:121], v1
	ds_read_b128 v[114:117], v5
	v_lshlrev_b32_e32 v1, 4, v3
	v_lshlrev_b32_e32 v3, 2, v2
	ds_bpermute_b32 v7, v3, v176
	v_lshrrev_b64 v[2:3], v2, exec
	v_and_b32_e32 v8, 1, v2
	v_sub_u32_e32 v2, v4, v193
	v_ashrrev_i16_e32 v2, 4, v2
	v_add_u32_sdwa v2, v186, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshlrev_b32_e32 v3, 2, v2
	ds_bpermute_b32 v4, v3, v187
	v_lshrrev_b64 v[2:3], v2, exec
	v_and_b32_e32 v9, 1, v2
	v_sub_u32_e32 v2, v40, v34
	v_ashrrev_i32_e32 v2, 3, v2
	v_lshrrev_b32_e32 v10, 3, v10
	v_add_u32_e32 v2, v2, v186
	v_bitop3_b32 v10, v10, v0, 15 bitop3:0x78
	v_lshlrev_b32_e32 v194, 2, v2
	v_lshlrev_b32_e32 v177, 4, v10
	ds_bpermute_b32 v10, v194, v176
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v7, 1, v7
	v_lshrrev_b64 v[2:3], v2, exec
	v_cmp_eq_u32_e32 vcc, 1, v8
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v4, 1, v4
	v_and_b32_e32 v2, 1, v2
	v_cndmask_b32_e32 v69, v39, v7, vcc
	v_cmp_eq_u32_e32 vcc, 1, v9
	v_add_u32_e32 v71, 0, v193
	v_add_u32_e32 v5, v12, v177
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v10
	v_cndmask_b32_e32 v70, v39, v4, vcc
	v_readfirstlane_b32 s33, v71
	v_cmp_eq_u32_e32 vcc, 1, v2
	v_add_u32_e32 v6, v12, v1
	ds_read_b128 v[142:145], v5
	ds_read_b128 v[130:133], v6
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v69, s[20:23], 0 offen lds
	s_mov_b32 m0, s33
	v_cndmask_b32_e32 v2, v39, v3, vcc
	ds_bpermute_b32 v3, v194, v187
	v_add_u32_e32 v11, 0x4000, v68
	buffer_load_dwordx4 v70, s[20:23], 0 offen lds
	s_addc_u32 s20, s15, s7
	v_readfirstlane_b32 s14, v11
	s_and_b32 s1, s20, 0xffff
	s_or_b32 s1, s1, s19
	s_mov_b32 m0, s14
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v2, s[0:3], 0 offen lds
	v_add_u32_e32 v2, 0x4000, v71
	v_lshlrev_b32_e32 v26, 8, v37
	v_readfirstlane_b32 s14, v2
	v_lshlrev_b32_e32 v2, 1, v3
	v_or_b32_e32 v72, v183, v26
	v_cndmask_b32_e32 v2, v39, v2, vcc
	s_mov_b32 m0, s14
	v_add_u32_e32 v73, 0, v72
	buffer_load_dwordx4 v2, s[0:3], 0 offen lds
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b128 v[2:5], v73
	ds_read_b128 v[18:21], v73 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[138:141], 0
	v_or_b32_e32 v74, v182, v26
	v_add_u32_e32 v75, 0, v74
	ds_read_b128 v[22:25], v75
	ds_read_b128 v[40:43], v75 offset:8192
	v_or_b32_e32 v76, v181, v26
	v_add_u32_e32 v77, 0, v76
	v_or_b32_e32 v78, v180, v26
	v_add_u32_e32 v79, 0, v78
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[134:137], v[2:17]
	ds_read_b128 v[22:25], v77
	ds_read_b128 v[44:47], v77 offset:8192
	v_or_b32_e32 v80, v179, v26
	v_add_u32_e32 v81, 0, v80
	v_or_b32_e32 v82, v178, v26
	v_add_u32_e32 v83, 0, v82
	v_or_b32_e32 v84, v177, v26
	v_add_u32_e32 v85, 0, v84
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[126:129], v[2:17]
	ds_read_b128 v[22:25], v79
	ds_read_b128 v[48:51], v79 offset:8192
	v_or_b32_e32 v86, v1, v26
	v_add_u32_e32 v98, 0, v86
	v_mad_u64_u32 v[174:175], s[14:15], s24, v38, v[34:35]
	v_lshl_add_u32 v175, s24, 5, v174
	s_and_b32 s14, s16, 0xffff
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[122:125], v[2:17]
	ds_read_b128 v[22:25], v81
	ds_read_b128 v[52:55], v81 offset:8192
	s_mov_b32 s15, s3
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[118:121], v[2:17]
	ds_read_b128 v[22:25], v83
	ds_read_b128 v[56:59], v83 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[114:117], v[2:17]
	ds_read_b128 v[22:25], v85
	ds_read_b128 v[60:63], v85 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[142:145], v[2:17]
	ds_read_b128 v[22:25], v98
	ds_read_b128 v[64:67], v98 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[130:133], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[138:141], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[134:137], v[18:33]
	.file	2 "/app/OAI-triton/python/triton/language" "standard.py"
	s_nop 7
	s_nop 1
	v_max_f32_e32 v40, v3, v3
	v_max_f32_e32 v41, v2, v2
	v_max_f32_e32 v40, v41, v40
	v_max3_f32 v40, v40, v4, v5
	v_max3_f32 v40, v40, v6, v7
	v_max3_f32 v40, v40, v8, v9
	v_max3_f32 v40, v40, v10, v11
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[126:129], v[18:33]
	v_max3_f32 v40, v40, v12, v13
	v_max3_f32 v40, v40, v14, v15
	v_max3_f32 v40, v40, v16, v17
	v_mfma_f32_32x32x16_f16 v[18:33], v[48:51], v[122:125], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[52:55], v[118:121], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[56:59], v[114:117], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[60:63], v[142:145], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[64:67], v[130:133], v[18:33]
	s_nop 7
	s_nop 3
	v_max3_f32 v40, v40, v18, v19
	v_max3_f32 v40, v40, v20, v21
	v_max3_f32 v40, v40, v22, v23
	v_max3_f32 v40, v40, v24, v25
	v_max3_f32 v42, v40, v26, v27
	v_lshlrev_b32_e32 v40, 1, v0
	v_and_b32_e32 v40, 0x60, v40
	v_bitop3_b32 v199, v35, v40, s13 bitop3:0x6c
	v_sub_u32_e32 v40, v199, v34
	v_ashrrev_i32_e32 v40, 3, v40
	v_add_u32_e32 v43, v40, v186
	v_lshrrev_b64 v[40:41], v43, exec
	v_lshlrev_b32_e32 v38, 2, v43
	ds_bpermute_b32 v41, v38, v174
	ds_bpermute_b32 v38, v38, v175
	s_and_b32 s13, s24, 0x3fff
	s_bitset1_b32 s13, 14
	v_and_b32_e32 v40, 1, v40
	s_lshl_b32 s22, s13, 16
	v_add_u32_e32 v43, 0x8000, v68
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v41, 1, v41
	v_cmp_eq_u32_e32 vcc, 1, v40
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v38, 1, v38
	s_or_b32 s13, s14, s22
	v_readfirstlane_b32 s1, v43
	v_cndmask_b32_e32 v40, v39, v41, vcc
	v_cndmask_b32_e32 v38, v39, v38, vcc
	v_add_u32_e32 v39, 0x8000, v71
	s_add_u32 s0, s0, s6
	v_readfirstlane_b32 s21, v39
	s_mov_b32 s14, s2
	s_mov_b32 m0, s1
	s_addc_u32 s1, s20, s7
	s_ashr_i32 s39, s38, 31
	buffer_load_dwordx4 v40, s[12:15], 0 offen lds
	s_mov_b32 m0, s21
	s_lshl_b64 s[20:21], s[38:39], 1
	s_add_u32 s23, s12, s20
	s_addc_u32 s24, s16, s21
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v38, s[12:15], 0 offen lds
	s_or_b32 s1, s1, s19
	s_mov_b32 m0, s29
	v_add_u32_e32 v39, 0xc000, v68
	buffer_load_dwordx4 v69, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	v_readfirstlane_b32 s12, v39
	buffer_load_dwordx4 v70, s[0:3], 0 offen lds
	s_and_b32 s0, s24, 0xffff
	v_add_u32_e32 v39, 0xc000, v71
	s_or_b32 s1, s0, s22
	s_mov_b32 s0, s23
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v39
	s_waitcnt vmcnt(4)
	s_barrier
	buffer_load_dwordx4 v40, s[0:3], 0 offen lds
	s_mov_b32 m0, s12
	s_add_i32 s12, 0, 0x4000
	buffer_load_dwordx4 v38, s[0:3], 0 offen lds
	v_max3_f32 v38, v42, v28, v29
	v_max3_f32 v38, v38, v30, v31
	v_max3_f32 v39, v38, v32, v33
	v_mov_b32_e32 v40, v39
	s_nop 1
	v_permlane32_swap_b32_e32 v39, v40
	v_mov_b32_e32 v38, 0xff800000
	v_max3_f32 v203, v39, v40, v38
	v_mul_f32_e32 v39, 0xbe0293ee, v203
	v_fmamk_f32 v40, v2, 0x3e0293ee, v39
	s_movk_i32 s0, 0x1ff
	v_add_u32_e32 v2, 0xff, v0
	v_cmp_gt_u32_e32 vcc, s0, v2
	s_movk_i32 s0, 0x1fe
	v_fmamk_f32 v3, v3, 0x3e0293ee, v39
	v_fmamk_f32 v4, v4, 0x3e0293ee, v39
	v_fmamk_f32 v5, v5, 0x3e0293ee, v39
	v_fmamk_f32 v6, v6, 0x3e0293ee, v39
	v_fmamk_f32 v7, v7, 0x3e0293ee, v39
	v_fmamk_f32 v8, v8, 0x3e0293ee, v39
	v_fmamk_f32 v9, v9, 0x3e0293ee, v39
	v_fmamk_f32 v10, v10, 0x3e0293ee, v39
	v_fmamk_f32 v11, v11, 0x3e0293ee, v39
	v_fmamk_f32 v12, v12, 0x3e0293ee, v39
	v_fmamk_f32 v13, v13, 0x3e0293ee, v39
	v_fmamk_f32 v14, v14, 0x3e0293ee, v39
	v_fmamk_f32 v15, v15, 0x3e0293ee, v39
	v_fmamk_f32 v16, v16, 0x3e0293ee, v39
	v_fmamk_f32 v17, v17, 0x3e0293ee, v39
	v_fmamk_f32 v18, v18, 0x3e0293ee, v39
	v_fmamk_f32 v19, v19, 0x3e0293ee, v39
	v_fmamk_f32 v20, v20, 0x3e0293ee, v39
	v_fmamk_f32 v21, v21, 0x3e0293ee, v39
	v_fmamk_f32 v22, v22, 0x3e0293ee, v39
	v_fmamk_f32 v23, v23, 0x3e0293ee, v39
	v_fmamk_f32 v24, v24, 0x3e0293ee, v39
	v_fmamk_f32 v25, v25, 0x3e0293ee, v39
	v_fmamk_f32 v26, v26, 0x3e0293ee, v39
	v_fmamk_f32 v27, v27, 0x3e0293ee, v39
	v_fmamk_f32 v28, v28, 0x3e0293ee, v39
	v_fmamk_f32 v29, v29, 0x3e0293ee, v39
	v_fmamk_f32 v30, v30, 0x3e0293ee, v39
	v_fmamk_f32 v31, v31, 0x3e0293ee, v39
	v_fmamk_f32 v32, v32, 0x3e0293ee, v39
	v_fmac_f32_e32 v39, 0x3e0293ee, v33
	v_cmp_lt_u32_e64 s[0:1], s0, v2
	v_add_u32_e32 v2, s12, v72
	v_add_u32_e32 v33, s12, v74
	v_add_u32_e32 v41, s12, v76
	v_add_u32_e32 v42, s12, v78
	v_add_u32_e32 v43, s12, v80
	v_add_u32_e32 v44, s12, v82
	v_add_u32_e32 v45, s12, v84
	v_add_u32_e32 v46, s12, v86
	ds_read_b128 v[66:69], v73 offset:16384
	ds_read_b128 v[110:113], v75 offset:16384
	ds_read_b128 v[106:109], v77 offset:16384
	ds_read_b128 v[102:105], v79 offset:16384
	ds_read_b128 v[94:97], v81 offset:16384
	ds_read_b128 v[90:93], v83 offset:16384
	ds_read_b128 v[86:89], v85 offset:16384
	ds_read_b128 v[98:101], v98 offset:16384
	ds_read_b128 v[82:85], v2 offset:8192
	ds_read_b128 v[170:173], v33 offset:8192
	ds_read_b128 v[166:169], v41 offset:8192
	ds_read_b128 v[162:165], v42 offset:8192
	ds_read_b128 v[158:161], v43 offset:8192
	ds_read_b128 v[154:157], v44 offset:8192
	ds_read_b128 v[150:153], v45 offset:8192
	ds_read_b128 v[146:149], v46 offset:8192
	s_movk_i32 s14, 0x60
	s_mov_b32 s16, 0
	s_mov_b32 s29, 0x3e0293ee
	v_fmac_f32_e32 v38, 0xbe0293ee, v203
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[12:13]
	s_add_u32 s0, s30, s34
	s_addc_u32 s1, s31, s35
	s_mul_i32 s13, s36, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_exp_f32_e32 v227, v3
	v_exp_f32_e32 v232, v4
	v_and_b32_e32 v3, 32, v0
	v_lshlrev_b32_e32 v4, 2, v0
	s_mul_hi_i32 s12, s36, 6
	s_add_u32 s0, s13, s0
	v_and_b32_e32 v4, 12, v4
	v_lshrrev_b32_e32 v3, 3, v3
	s_addc_u32 s1, s12, s1
	v_exp_f32_e32 v208, v40
	v_exp_f32_e32 v229, v5
	v_exp_f32_e32 v228, v6
	v_exp_f32_e32 v230, v7
	v_exp_f32_e32 v231, v8
	v_exp_f32_e32 v209, v9
	v_exp_f32_e32 v237, v10
	v_exp_f32_e32 v221, v11
	v_exp_f32_e32 v220, v12
	v_exp_f32_e32 v233, v13
	v_exp_f32_e32 v224, v14
	v_exp_f32_e32 v213, v15
	v_exp_f32_e32 v223, v16
	v_exp_f32_e32 v234, v17
	v_exp_f32_e32 v219, v18
	v_exp_f32_e32 v214, v19
	v_exp_f32_e32 v218, v20
	v_exp_f32_e32 v235, v21
	v_exp_f32_e32 v222, v22
	v_exp_f32_e32 v217, v23
	v_exp_f32_e32 v216, v24
	v_exp_f32_e32 v215, v25
	v_exp_f32_e32 v236, v26
	v_exp_f32_e32 v238, v27
	v_exp_f32_e32 v226, v28
	v_exp_f32_e32 v225, v29
	v_exp_f32_e32 v212, v30
	v_exp_f32_e32 v211, v31
	v_exp_f32_e32 v189, v32
	v_exp_f32_e32 v190, v39
	v_exp_f32_e32 v185, v38
	v_and_or_b32 v202, v35, 32, v4
	v_and_or_b32 v3, v36, 3, v3
	s_add_u32 s4, s4, s0
	v_lshlrev_b32_e32 v204, 7, v37
	v_and_b32_e32 v195, 16, v0
	v_and_b32_e32 v200, 64, v35
	v_lshlrev_b32_e32 v196, 7, v3
	v_bitop3_b32 v201, v4, v35, 32 bitop3:0x72
	v_bitop3_b32 v198, v202, v35, 64 bitop3:0x72
	v_bitop3_b32 v197, v35, v4, s14 bitop3:0x4e
	v_add_u32_e32 v205, v188, v34
	s_addc_u32 s31, s5, s1
	s_add_i32 s5, 0, 0x8000
	s_add_i32 s30, 0, 0xc000
	v_mov_b32_e32 v184, 1.0
	s_movk_i32 s33, 0xffc0
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
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[138:141], 0
	v_mov_b32_e32 v206, v184
	s_mov_b32 s14, s5
	s_mov_b32 s5, s30
	v_mov_b32_e32 v252, v203
	s_mov_b32 s35, s16
	s_setprio 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[134:137], v[66:81]
	v_mul_f32_e32 v22, v22, v185
	v_cvt_pk_f16_f32 v111, v223, v234
	v_add_f32_e32 v110, v208, v227
	v_mul_f32_e32 v46, v46, v185
	v_add_f32_e32 v110, v110, v232
	v_mul_f32_e32 v27, v27, v185
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[126:129], v[66:81]
	v_mul_f32_e32 v4, v4, v185
	v_cvt_pk_f16_f32 v109, v220, v233
	v_cvt_pk_f16_f32 v108, v237, v221
	v_cvt_pk_f16_f32 v107, v216, v215
	v_add_f32_e32 v110, v110, v229
	v_mul_f32_e32 v10, v10, v185
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[122:125], v[66:81]
	v_mul_f32_e32 v11, v11, v185
	v_cvt_pk_f16_f32 v105, v218, v235
	v_cvt_pk_f16_f32 v104, v219, v214
	v_add_f32_e32 v110, v110, v228
	v_mul_f32_e32 v12, v12, v185
	v_add_f32_e32 v110, v110, v230
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[118:121], v[66:81]
	v_add_f32_e32 v110, v110, v231
	v_add_f32_e32 v106, v110, v209
	v_mul_f32_e32 v14, v14, v185
	v_cvt_pk_f16_f32 v110, v224, v213
	v_add_f32_e32 v106, v106, v237
	v_mul_f32_e32 v13, v13, v185
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[114:117], v[66:81]
	v_mul_f32_e32 v15, v15, v185
	v_add_f32_e32 v106, v106, v221
	v_mul_f32_e32 v16, v16, v185
	v_add_f32_e32 v106, v106, v220
	v_mul_f32_e32 v17, v17, v185
	v_add_f32_e32 v106, v106, v233
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[142:145], v[66:81]
	v_add_f32_e32 v106, v106, v224
	v_add_f32_e32 v102, v106, v213
	v_mul_f32_e32 v9, v9, v185
	v_cvt_pk_f16_f32 v106, v222, v217
	v_add_f32_e32 v102, v102, v223
	v_mul_f32_e32 v7, v7, v185
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[98:101], v[130:133], v[66:81]
	v_mul_f32_e32 v8, v8, v185
	v_cvt_pk_f16_f32 v101, v226, v225
	v_cvt_pk_f16_f32 v100, v236, v238
	v_mul_f32_e32 v6, v6, v185
	v_mul_f32_e32 v5, v5, v185
	v_mul_f32_e32 v3, v3, v185
	v_add_f32_e32 v102, v102, v234
	v_add_f32_e32 v102, v102, v219
	v_add_f32_e32 v102, v102, v214
	v_add_f32_e32 v102, v102, v218
	v_add_f32_e32 v94, v102, v235
	v_add_f32_e32 v94, v94, v222
	v_add_f32_e32 v94, v94, v217
	v_add_f32_e32 v94, v94, v216
	v_add_f32_e32 v94, v94, v215
	v_add_f32_e32 v94, v94, v236
	v_add_f32_e32 v90, v94, v238
	v_add_f32_e32 v90, v90, v226
	v_add_f32_e32 v90, v90, v225
	v_add_f32_e32 v90, v90, v212
	v_add_f32_e32 v90, v90, v211
	v_add_f32_e32 v90, v90, v189
	v_add_f32_e32 v102, v90, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[138:141], 0
	v_mul_f32_e32 v33, v33, v185
	v_mul_f32_e32 v2, v2, v185
	v_mov_b32_e32 v103, v102
	v_mul_f32_e32 v32, v32, v185
	s_nop 0
	v_permlane32_swap_b32_e32 v102, v103
	v_add_f32_e32 v184, v102, v103
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[134:137], v[82:97]
	v_cvt_pk_f16_f32 v103, v189, v190
	v_mul_f32_e32 v31, v31, v185
	v_cvt_pk_f16_f32 v102, v212, v211
	v_fmac_f32_e32 v184, v206, v185
	v_mul_f32_e32 v30, v30, v185
	v_mul_f32_e32 v29, v29, v185
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[126:129], v[82:97]
	v_mul_f32_e32 v26, v26, v185
	v_mul_f32_e32 v28, v28, v185
	v_mul_f32_e32 v25, v25, v185
	v_mul_f32_e32 v24, v24, v185
	v_mul_f32_e32 v21, v21, v185
	v_mul_f32_e32 v23, v23, v185
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[122:125], v[82:97]
	v_mul_f32_e32 v19, v19, v185
	v_mul_f32_e32 v20, v20, v185
	v_mul_f32_e32 v18, v18, v185
	v_mul_f32_e32 v49, v49, v185
	v_mul_f32_e32 v47, v47, v185
	v_mul_f32_e32 v48, v48, v185
	v_mfma_f32_32x32x16_f16 v[82:97], v[158:161], v[118:121], v[82:97]
	v_mul_f32_e32 v44, v44, v185
	v_mul_f32_e32 v45, v45, v185
	v_mul_f32_e32 v43, v43, v185
	v_mul_f32_e32 v42, v42, v185
	v_mul_f32_e32 v41, v41, v185
	v_mul_f32_e32 v50, v50, v185
	v_mfma_f32_32x32x16_f16 v[82:97], v[154:157], v[114:117], v[82:97]
	v_mul_f32_e32 v52, v52, v185
	v_mul_f32_e32 v51, v51, v185
	v_mul_f32_e32 v53, v53, v185
	v_mul_f32_e32 v55, v55, v185
	v_mul_f32_e32 v54, v54, v185
	v_mul_f32_e32 v56, v56, v185
	v_mfma_f32_32x32x16_f16 v[82:97], v[150:153], v[142:145], v[82:97]
	v_mul_f32_e32 v58, v58, v185
	v_mul_f32_e32 v57, v57, v185
	v_mul_f32_e32 v59, v59, v185
	v_mul_f32_e32 v61, v61, v185
	v_mul_f32_e32 v60, v60, v185
	v_mul_f32_e32 v62, v62, v185
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[146:149], v[130:133], v[82:97]
	v_mul_f32_e32 v64, v64, v185
	v_cvt_pk_f16_f32 v149, v231, v209
	v_cvt_pk_f16_f32 v148, v228, v230
	v_cvt_pk_f16_f32 v147, v232, v229
	v_cvt_pk_f16_f32 v146, v208, v227
	v_mul_f32_e32 v65, v65, v185
	v_mul_f32_e32 v40, v40, v185
	v_mul_f32_e32 v39, v39, v185
	v_mul_f32_e32 v38, v38, v185
	v_mul_f32_e32 v37, v37, v185
	v_mul_f32_e32 v36, v36, v185
	v_mul_f32_e32 v34, v34, v185
	v_mul_f32_e32 v35, v35, v185
	v_mul_f32_e32 v63, v63, v185
	; iglp_opt mask(0x0000000A)
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s23, s20
	s_addc_u32 s36, s24, s21
	s_add_i32 s0, s13, 1
	s_cmp_lt_i32 s0, 2
	s_cselect_b32 s37, s0, 0
	ds_bpermute_b32 v98, v194, v176
	s_lshl_b32 s0, s37, 14
	ds_bpermute_b32 v99, v194, v187
	s_add_i32 s16, s0, 0
	v_add_u32_e32 v206, s16, v192
	v_add_u32_e32 v207, s16, v193
	s_and_b32 s0, s31, 0xffff
	v_readfirstlane_b32 s15, v206
	s_or_b32 s1, s0, s19
	s_mov_b32 s0, s4
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v98, 1, v98
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v207
	buffer_load_dwordx4 v98, s[0:3], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v98, 1, v99
	s_mov_b32 m0, s15
	s_nop 0
	buffer_load_dwordx4 v98, s[0:3], 0 offen lds
	v_lshlrev_b32_e32 v98, 1, v202
	v_lshlrev_b32_e32 v99, 1, v200
	v_add3_u32 v98, s14, v98, v99
	v_lshlrev_b32_e32 v112, 1, v195
	v_lshlrev_b32_e32 v113, 1, v196
	v_add3_u32 v98, v98, v112, v113
	ds_read_b64_tr_b16 v[150:151], v98
	ds_read_b64_tr_b16 v[152:153], v98 offset:2048
	ds_read_b64_tr_b16 v[154:155], v98 offset:4096
	ds_read_b64_tr_b16 v[156:157], v98 offset:6144
	ds_read_b64_tr_b16 v[158:159], v98 offset:8192
	ds_read_b64_tr_b16 v[160:161], v98 offset:10240
	ds_read_b64_tr_b16 v[162:163], v98 offset:12288
	ds_read_b64_tr_b16 v[164:165], v98 offset:14336
	v_lshlrev_b32_e32 v98, 1, v201
	v_add3_u32 v98, s14, v98, v99
	v_add3_u32 v98, v98, v112, v113
	ds_read_b64_tr_b16 v[166:167], v98
	ds_read_b64_tr_b16 v[168:169], v98 offset:2048
	ds_read_b64_tr_b16 v[170:171], v98 offset:4096
	ds_read_b64_tr_b16 v[172:173], v98 offset:6144
	ds_read_b64_tr_b16 v[214:215], v98 offset:8192
	ds_read_b64_tr_b16 v[216:217], v98 offset:10240
	ds_read_b64_tr_b16 v[228:229], v98 offset:12288
	ds_read_b64_tr_b16 v[230:231], v98 offset:14336
	v_lshl_add_u32 v98, v198, 1, s14
	v_add3_u32 v98, v98, v112, v113
	ds_read_b64_tr_b16 v[218:219], v98
	ds_read_b64_tr_b16 v[220:221], v98 offset:2048
	ds_read_b64_tr_b16 v[222:223], v98 offset:4096
	ds_read_b64_tr_b16 v[224:225], v98 offset:6144
	ds_read_b64_tr_b16 v[232:233], v98 offset:8192
	ds_read_b64_tr_b16 v[234:235], v98 offset:10240
	ds_read_b64_tr_b16 v[236:237], v98 offset:12288
	ds_read_b64_tr_b16 v[238:239], v98 offset:14336
	v_lshl_add_u32 v98, v197, 1, s14
	v_add3_u32 v98, v98, v112, v113
	ds_read_b64_tr_b16 v[208:209], v98
	ds_read_b64_tr_b16 v[210:211], v98 offset:2048
	ds_read_b64_tr_b16 v[240:241], v98 offset:4096
	ds_read_b64_tr_b16 v[242:243], v98 offset:6144
	ds_read_b64_tr_b16 v[244:245], v98 offset:8192
	ds_read_b64_tr_b16 v[246:247], v98 offset:10240
	ds_read_b64_tr_b16 v[248:249], v98 offset:12288
	ds_read_b64_tr_b16 v[250:251], v98 offset:14336
	; sched_barrier mask(0x00000000)
	s_barrier
	s_setprio 0
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[208:211], v[146:149], v[2:17]
	v_max_f32_e32 v98, v66, v66
	v_max_f32_e32 v99, v67, v67
	v_max_f32_e32 v98, v98, v99
	v_max3_f32 v98, v98, v68, v69
	v_max3_f32 v98, v98, v70, v71
	v_max3_f32 v98, v98, v72, v73
	v_mfma_f32_32x32x16_f16 v[18:33], v[218:221], v[146:149], v[18:33]
	v_max3_f32 v98, v98, v74, v75
	v_max3_f32 v98, v98, v76, v77
	v_max3_f32 v98, v98, v78, v79
	v_max3_f32 v98, v98, v80, v81
	v_max3_f32 v98, v98, v82, v83
	v_max3_f32 v98, v98, v84, v85
	v_mfma_f32_32x32x16_f16 v[18:33], v[222:225], v[108:111], v[18:33]
	v_max3_f32 v98, v98, v86, v87
	v_max3_f32 v98, v98, v88, v89
	v_max3_f32 v98, v98, v90, v91
	v_max3_f32 v98, v98, v92, v93
	v_max3_f32 v98, v98, v94, v95
	v_max3_f32 v98, v98, v96, v97
	v_mfma_f32_32x32x16_f16 v[18:33], v[232:235], v[104:107], v[18:33]
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_max3_f32 v203, v252, v98, v99
	v_mul_f32_e32 v210, 0x3e0293ee, v203
	v_fma_f32 v66, v66, s29, -v210
	v_fma_f32 v67, v67, s29, -v210
	v_mfma_f32_32x32x16_f16 v[18:33], v[236:239], v[100:103], v[18:33]
	v_fma_f32 v97, v97, s29, -v210
	v_fma_f32 v98, v252, s29, -v210
	v_fma_f32 v96, v96, s29, -v210
	v_fma_f32 v95, v95, s29, -v210
	v_fma_f32 v94, v94, s29, -v210
	v_fma_f32 v93, v93, s29, -v210
	v_mfma_f32_32x32x16_f16 v[50:65], v[150:153], v[146:149], v[50:65]
	v_fma_f32 v92, v92, s29, -v210
	v_fma_f32 v91, v91, s29, -v210
	v_fma_f32 v90, v90, s29, -v210
	v_fma_f32 v86, v86, s29, -v210
	v_fma_f32 v85, v85, s29, -v210
	v_fma_f32 v84, v84, s29, -v210
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[240:243], v[108:111], v[2:17]
	v_fma_f32 v82, v82, s29, -v210
	v_fma_f32 v81, v81, s29, -v210
	v_fma_f32 v80, v80, s29, -v210
	v_fma_f32 v79, v79, s29, -v210
	v_fma_f32 v78, v78, s29, -v210
	v_fma_f32 v77, v77, s29, -v210
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[146:149], v[34:49]
	v_fma_f32 v76, v76, s29, -v210
	v_fma_f32 v75, v75, s29, -v210
	v_fma_f32 v74, v74, s29, -v210
	v_fma_f32 v73, v73, s29, -v210
	v_fma_f32 v68, v68, s29, -v210
	v_exp_f32_e32 v225, v93
	v_mfma_f32_32x32x16_f16 v[50:65], v[154:157], v[108:111], v[50:65]
	v_exp_f32_e32 v232, v68
	v_exp_f32_e32 v218, v84
	v_exp_f32_e32 v227, v67
	v_mfma_f32_32x32x16_f16 v[50:65], v[158:161], v[104:107], v[50:65]
	v_exp_f32_e32 v233, v77
	v_exp_f32_e32 v208, v66
	v_exp_f32_e32 v190, v97
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[244:247], v[104:107], v[2:17]
	v_exp_f32_e32 v237, v74
	v_exp_f32_e32 v185, v98
	v_exp_f32_e32 v220, v76
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[108:111], v[34:49]
	v_exp_f32_e32 v224, v78
	v_exp_f32_e32 v189, v96
	v_exp_f32_e32 v213, v79
	v_mfma_f32_32x32x16_f16 v[34:49], v[214:217], v[104:107], v[34:49]
	v_fma_f32 v89, v89, s29, -v210
	v_fma_f32 v88, v88, s29, -v210
	v_fma_f32 v87, v87, s29, -v210
	v_fma_f32 v83, v83, s29, -v210
	v_exp_f32_e32 v234, v81
	v_mfma_f32_32x32x16_f16 v[34:49], v[228:231], v[100:103], v[34:49]
	v_fma_f32 v71, v71, s29, -v210
	v_fma_f32 v69, v69, s29, -v210
	v_fma_f32 v70, v70, s29, -v210
	v_fma_f32 v72, v72, s29, -v210
	v_exp_f32_e32 v221, v75
	v_mfma_f32_32x32x16_f16 v[50:65], v[162:165], v[100:103], v[50:65]
	v_exp_f32_e32 v231, v72
	v_exp_f32_e32 v235, v85
	v_exp_f32_e32 v222, v86
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[248:251], v[100:103], v[2:17]
	v_exp_f32_e32 v236, v90
	v_exp_f32_e32 v215, v89
	v_exp_f32_e32 v219, v82
	v_exp_f32_e32 v209, v73
	v_exp_f32_e32 v211, v95
	v_exp_f32_e32 v212, v94
	v_exp_f32_e32 v226, v92
	v_exp_f32_e32 v216, v88
	v_exp_f32_e32 v217, v87
	v_exp_f32_e32 v214, v83
	v_exp_f32_e32 v223, v80
	v_exp_f32_e32 v228, v70
	v_exp_f32_e32 v229, v69
	v_exp_f32_e32 v230, v71
	v_exp_f32_e32 v238, v91
	; iglp_opt mask(0x0000000A)
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s0, s13, 14
	s_add_i32 s34, s0, 0
	s_add_i32 s30, s34, 0x8000
	v_lshlrev_b32_e32 v66, 1, v199
	v_lshlrev_b32_e32 v67, 1, v188
	v_add3_u32 v66, s34, v66, v67
	v_lshl_add_u32 v67, v191, 1, s30
	v_lshl_add_u32 v68, v205, 1, s30
	v_sub_u32_e32 v69, v66, v67
	v_add_u32_e32 v68, 0x2000, v68
	v_add_u32_e32 v69, 0x8000, v69
	v_ashrrev_i32_e32 v70, 31, v69
	v_sub_u32_e32 v66, v66, v68
	v_lshrrev_b32_e32 v70, 28, v70
	v_add_u32_e32 v66, 0xa000, v66
	v_add_u32_e32 v69, v69, v70
	v_ashrrev_i32_e32 v70, 31, v66
	v_lshrrev_b32_e32 v70, 28, v70
	v_ashrrev_i32_e32 v69, 4, v69
	v_add_u32_e32 v66, v66, v70
	v_add_lshl_u32 v69, v69, v186, 2
	v_ashrrev_i32_e32 v66, 4, v66
	ds_bpermute_b32 v69, v69, v174
	v_add_lshl_u32 v66, v66, v186, 2
	ds_bpermute_b32 v66, v66, v175
	s_and_b32 s0, s36, 0xffff
	s_or_b32 s13, s0, s22
	v_readfirstlane_b32 s0, v67
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v69, 1, v69
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v68
	buffer_load_dwordx4 v69, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	s_mov_b32 m0, s0
	v_lshlrev_b32_e32 v239, 1, v204
	buffer_load_dwordx4 v66, s[12:15], 0 offen lds
	v_add3_u32 v70, s35, v183, v239
	v_add3_u32 v71, s35, v182, v239
	v_add3_u32 v72, s35, v181, v239
	v_add3_u32 v73, s35, v180, v239
	v_add3_u32 v74, s35, v179, v239
	v_add3_u32 v75, s35, v178, v239
	v_add3_u32 v76, s35, v177, v239
	v_add3_u32 v77, s35, v1, v239
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[82:85], v70 offset:8192
	ds_read_b128 v[110:113], v71
	ds_read_b128 v[170:173], v71 offset:8192
	ds_read_b128 v[106:109], v72
	ds_read_b128 v[166:169], v72 offset:8192
	ds_read_b128 v[102:105], v73
	ds_read_b128 v[162:165], v73 offset:8192
	ds_read_b128 v[94:97], v74
	ds_read_b128 v[158:161], v74 offset:8192
	ds_read_b128 v[90:93], v75
	ds_read_b128 v[154:157], v75 offset:8192
	ds_read_b128 v[86:89], v76
	ds_read_b128 v[150:153], v76 offset:8192
	ds_read_b128 v[98:101], v77
	ds_read_b128 v[146:149], v77 offset:8192
	; sched_barrier mask(0x00000000)
	s_add_u32 s23, s23, s20
	s_addc_u32 s24, s24, s21
	s_add_u32 s4, s4, s6
	s_addc_u32 s31, s31, s7
	s_add_i32 s33, s33, 64
	s_cmpk_lt_u32 s33, 0x1f00
	s_mov_b32 s13, s37
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	v_lshlrev_b32_e32 v70, 3, v0
	v_and_b32_e32 v70, 0x78, v70
	v_sub_u32_e32 v70, v199, v70
	v_ashrrev_i32_e32 v70, 3, v70
	v_and_b32_e32 v176, 31, v0
	v_lshrrev_b32_e32 v187, 1, v0
	v_add_u32_e32 v191, v70, v186
	v_and_b32_e32 v192, 64, v187
	v_lshlrev_b32_e32 v188, 2, v191
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[138:141], 0
	v_and_b32_e32 v186, 0x100, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v186
	v_and_b32_e32 v186, 0xa0, v187
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[134:137], v[66:81]
	v_mul_f32_e32 v50, v50, v185
	v_mul_f32_e32 v51, v51, v185
	v_mul_f32_e32 v52, v52, v185
	v_mul_f32_e32 v53, v53, v185
	v_mul_f32_e32 v54, v54, v185
	v_mul_f32_e32 v55, v55, v185
	v_mul_f32_e32 v56, v56, v185
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[126:129], v[66:81]
	v_mul_f32_e32 v57, v57, v185
	v_mul_f32_e32 v58, v58, v185
	v_mul_f32_e32 v59, v59, v185
	v_mul_f32_e32 v60, v60, v185
	v_mul_f32_e32 v61, v61, v185
	v_mul_f32_e32 v62, v62, v185
	v_mul_f32_e32 v63, v63, v185
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[122:125], v[66:81]
	v_add_f32_e32 v102, v208, v227
	v_add_f32_e32 v102, v102, v232
	v_add_f32_e32 v102, v102, v229
	v_add_f32_e32 v102, v102, v228
	v_mul_f32_e32 v64, v64, v185
	v_mul_f32_e32 v65, v65, v185
	v_mul_f32_e32 v34, v34, v185
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[118:121], v[66:81]
	v_add_f32_e32 v94, v102, v230
	v_add_f32_e32 v94, v94, v231
	v_add_f32_e32 v94, v94, v209
	v_add_f32_e32 v94, v94, v237
	v_add_f32_e32 v94, v94, v221
	v_add_f32_e32 v94, v94, v220
	v_cvt_pk_f16_f32 v95, v232, v229
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[114:117], v[66:81]
	v_add_f32_e32 v90, v94, v233
	v_add_f32_e32 v90, v90, v224
	v_add_f32_e32 v90, v90, v213
	v_add_f32_e32 v90, v90, v223
	v_add_f32_e32 v90, v90, v234
	v_add_f32_e32 v90, v90, v219
	v_add_f32_e32 v90, v90, v214
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[142:145], v[66:81]
	v_add_f32_e32 v86, v90, v218
	v_add_f32_e32 v86, v86, v235
	v_add_f32_e32 v86, v86, v222
	v_add_f32_e32 v86, v86, v217
	v_add_f32_e32 v86, v86, v216
	v_add_f32_e32 v86, v86, v215
	v_add_f32_e32 v86, v86, v236
	v_mfma_f32_32x32x16_f16 v[66:81], v[98:101], v[130:133], v[66:81]
	v_add_f32_e32 v86, v86, v238
	v_add_f32_e32 v86, v86, v226
	v_add_f32_e32 v86, v86, v225
	v_add_f32_e32 v86, v86, v212
	v_add_f32_e32 v187, v86, v211
	v_cvt_pk_f16_f32 v92, v224, v213
	v_cvt_pk_f16_f32 v86, v219, v214
	v_mfma_f32_32x32x16_f16 v[98:113], v[82:85], v[138:141], 0
	v_cvt_pk_f16_f32 v89, v216, v215
	v_cvt_pk_f16_f32 v84, v212, v211
	v_cvt_pk_f16_f32 v94, v208, v227
	v_cvt_pk_f16_f32 v96, v228, v230
	v_cvt_pk_f16_f32 v97, v231, v209
	v_cvt_pk_f16_f32 v90, v237, v221
	v_cvt_pk_f16_f32 v91, v220, v233
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[134:137], v[98:113]
	v_cvt_pk_f16_f32 v93, v223, v234
	v_cvt_pk_f16_f32 v87, v218, v235
	v_cvt_pk_f16_f32 v88, v222, v217
	v_cvt_pk_f16_f32 v83, v226, v225
	v_mul_f32_e32 v35, v35, v185
	v_mul_f32_e32 v36, v36, v185
	v_mul_f32_e32 v37, v37, v185
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[126:129], v[98:113]
	v_or3_b32 v167, v200, v195, v202
	v_or_b32_e32 v166, v167, v196
	v_lshlrev_b32_e32 v166, 1, v166
	v_mul_f32_e32 v38, v38, v185
	v_mul_f32_e32 v39, v39, v185
	v_mul_f32_e32 v40, v40, v185
	v_mul_f32_e32 v41, v41, v185
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[122:125], v[98:113]
	v_add_u32_e32 v163, s5, v166
	v_add_lshl_u32 v162, v167, v196, 1
	v_add_u32_e32 v164, s5, v162
	ds_read_b64_tr_b16 v[168:169], v163
	ds_read_b64_tr_b16 v[170:171], v164 offset:2048
	ds_read_b64_tr_b16 v[212:213], v164 offset:4096
	ds_read_b64_tr_b16 v[214:215], v164 offset:6144
	ds_read_b64_tr_b16 v[216:217], v164 offset:8192
	ds_read_b64_tr_b16 v[218:219], v164 offset:10240
	ds_read_b64_tr_b16 v[220:221], v164 offset:12288
	ds_read_b64_tr_b16 v[222:223], v164 offset:14336
	v_mul_f32_e32 v42, v42, v185
	v_mul_f32_e32 v43, v43, v185
	v_mfma_f32_32x32x16_f16 v[98:113], v[158:161], v[118:121], v[98:113]
	v_or3_b32 v159, v195, v201, v200
	v_or_b32_e32 v158, v159, v196
	v_lshlrev_b32_e32 v158, 1, v158
	v_add_u32_e32 v160, s5, v158
	v_add_lshl_u32 v159, v159, v196, 1
	v_add_u32_e32 v161, s5, v159
	v_mul_f32_e32 v44, v44, v185
	v_mfma_f32_32x32x16_f16 v[98:113], v[154:157], v[114:117], v[98:113]
	ds_read_b64_tr_b16 v[154:155], v160
	ds_read_b64_tr_b16 v[156:157], v161 offset:2048
	ds_read_b64_tr_b16 v[224:225], v161 offset:4096
	ds_read_b64_tr_b16 v[226:227], v161 offset:6144
	v_mul_f32_e32 v45, v45, v185
	v_mul_f32_e32 v46, v46, v185
	v_mul_f32_e32 v47, v47, v185
	v_mul_f32_e32 v48, v48, v185
	v_mul_f32_e32 v49, v49, v185
	v_cvt_pk_f16_f32 v82, v236, v238
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[50:65], v[168:171], v[94:97], v[50:65]
	v_cvt_pk_f16_f32 v85, v189, v190
	v_or3_b32 v186, v186, v176, v192
	s_mul_i32 s2, s18, 0xc0000
	v_mul_f32_e32 v18, v18, v185
	v_mul_f32_e32 v19, v19, v185
	v_mul_f32_e32 v20, v20, v185
	v_mul_f32_e32 v21, v21, v185
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[154:157], v[94:97], v[34:49]
	v_mul_f32_e32 v22, v22, v185
	v_mul_f32_e32 v23, v23, v185
	v_mul_f32_e32 v24, v24, v185
	v_mul_f32_e32 v25, v25, v185
	v_mul_f32_e32 v26, v26, v185
	v_mul_f32_e32 v27, v27, v185
	v_mul_f32_e32 v28, v28, v185
	v_mfma_f32_32x32x16_f16 v[98:113], v[150:153], v[142:145], v[98:113]
	ds_read_b64_tr_b16 v[150:151], v161 offset:8192
	ds_read_b64_tr_b16 v[152:153], v161 offset:10240
	ds_read_b64_tr_b16 v[168:169], v161 offset:12288
	ds_read_b64_tr_b16 v[170:171], v161 offset:14336
	v_mul_f32_e32 v29, v29, v185
	v_mul_f32_e32 v30, v30, v185
	v_mul_f32_e32 v31, v31, v185
	v_mul_f32_e32 v32, v32, v185
	v_mul_f32_e32 v33, v33, v185
	s_ashr_i32 s3, s2, 31
	v_mfma_f32_32x32x16_f16 v[50:65], v[212:215], v[90:93], v[50:65]
	v_mul_f32_e32 v2, v2, v185
	v_mul_f32_e32 v3, v3, v185
	v_mul_f32_e32 v4, v4, v185
	v_mul_f32_e32 v5, v5, v185
	v_mul_f32_e32 v6, v6, v185
	v_mul_f32_e32 v7, v7, v185
	v_mul_f32_e32 v8, v8, v185
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[34:49], v[224:227], v[90:93], v[34:49]
	v_mul_f32_e32 v9, v9, v185
	v_mul_f32_e32 v10, v10, v185
	v_mul_f32_e32 v11, v11, v185
	v_mul_f32_e32 v12, v12, v185
	v_mul_f32_e32 v13, v13, v185
	v_mul_f32_e32 v14, v14, v185
	v_mul_f32_e32 v15, v15, v185
	v_mfma_f32_32x32x16_f16 v[98:113], v[146:149], v[130:133], v[98:113]
	v_or_b32_e32 v146, v198, v195
	v_or_b32_e32 v147, v146, v196
	v_lshlrev_b32_e32 v148, 1, v147
	v_add_u32_e32 v147, s5, v148
	v_add_lshl_u32 v149, v146, v196, 1
	v_add_u32_e32 v146, s5, v149
	ds_read_b64_tr_b16 v[198:199], v147
	ds_read_b64_tr_b16 v[200:201], v146 offset:2048
	ds_read_b64_tr_b16 v[212:213], v146 offset:4096
	ds_read_b64_tr_b16 v[214:215], v146 offset:6144
	v_mfma_f32_32x32x16_f16 v[50:65], v[216:219], v[86:89], v[50:65]
	v_or_b32_e32 v147, v197, v195
	ds_read_b64_tr_b16 v[154:155], v146 offset:8192
	ds_read_b64_tr_b16 v[156:157], v146 offset:10240
	ds_read_b64_tr_b16 v[216:217], v146 offset:12288
	ds_read_b64_tr_b16 v[218:219], v146 offset:14336
	v_or_b32_e32 v146, v147, v196
	v_lshlrev_b32_e32 v146, 1, v146
	v_add_u32_e32 v160, s5, v146
	v_add_lshl_u32 v147, v147, v196, 1
	v_add_u32_e32 v161, s5, v147
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[34:49], v[150:153], v[86:89], v[34:49]
	v_max_f32_e32 v150, v67, v67
	v_max_f32_e32 v151, v66, v66
	v_max_f32_e32 v150, v151, v150
	v_max3_f32 v150, v150, v68, v69
	v_max3_f32 v150, v150, v70, v71
	v_max3_f32 v150, v150, v72, v73
	v_mul_f32_e32 v16, v16, v185
	v_mfma_f32_32x32x16_f16 v[50:65], v[220:223], v[82:85], v[50:65]
	ds_read_b64_tr_b16 v[192:193], v160
	ds_read_b64_tr_b16 v[194:195], v161 offset:2048
	ds_read_b64_tr_b16 v[220:221], v161 offset:4096
	ds_read_b64_tr_b16 v[222:223], v161 offset:6144
	v_mul_f32_e32 v17, v17, v185
	v_max3_f32 v150, v150, v74, v75
	s_lshl_b64 s[2:3], s[2:3], 2
	v_max3_f32 v150, v150, v76, v77
	s_add_u32 s4, s8, s2
	s_addc_u32 s6, s9, s3
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[18:33], v[198:201], v[94:97], v[18:33]
	s_lshl_b32 s2, s17, 14
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s4, s2
	s_addc_u32 s6, s6, s3
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[2:3], s[28:29], 2
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[192:195], v[94:97], v[2:17]
	v_max3_f32 v94, v150, v78, v79
	v_max3_f32 v94, v94, v80, v81
	v_max3_f32 v94, v94, v98, v99
	v_max3_f32 v94, v94, v100, v101
	v_max3_f32 v94, v94, v102, v103
	v_max3_f32 v94, v94, v104, v105
	v_max3_f32 v94, v94, v106, v107
	v_mfma_f32_32x32x16_f16 v[18:33], v[212:215], v[90:93], v[18:33]
	s_add_u32 s4, s4, s2
	s_addc_u32 s19, s6, s3
	s_add_i32 s3, s28, 0xffffc100
	s_add_u32 s12, s12, s20
	s_addc_u32 s6, s36, s21
	s_and_b32 s6, s6, 0xffff
	s_add_i32 s2, s16, 0x8000
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[220:223], v[90:93], v[2:17]
	v_max3_f32 v90, v94, v108, v109
	v_max3_f32 v90, v90, v110, v111
	v_max3_f32 v90, v90, v112, v113
	v_mov_b32_e32 v91, v90
	s_nop 1
	v_permlane32_swap_b32_e32 v90, v91
	s_or_b32 s13, s6, s22
	v_mfma_f32_32x32x16_f16 v[34:49], v[168:171], v[82:85], v[34:49]
	ds_read_b64_tr_b16 v[168:169], v161 offset:8192
	ds_read_b64_tr_b16 v[170:171], v161 offset:10240
	ds_read_b64_tr_b16 v[196:197], v161 offset:12288
	ds_read_b64_tr_b16 v[198:199], v161 offset:14336
	v_max3_f32 v151, v203, v90, v91
	s_cmp_lt_i32 s3, 1
	s_mov_b32 s3, 0x3e0293ee
	v_mul_f32_e32 v150, 0x3e0293ee, v151
	v_fma_f32 v160, v68, s3, -v150
	v_fma_f32 v161, v69, s3, -v150
	v_mfma_f32_32x32x16_f16 v[18:33], v[154:157], v[86:89], v[18:33]
	v_fma_f32 v156, v66, s3, -v150
	v_add3_u32 v66, s16, v183, v239
	v_fma_f32 v157, v67, s3, -v150
	v_fma_f32 v163, v70, s3, -v150
	v_add3_u32 v70, s16, v182, v239
	v_fma_f32 v164, v71, s3, -v150
	v_fma_f32 v165, v72, s3, -v150
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[168:171], v[86:89], v[2:17]
	ds_read_b128 v[86:89], v66
	ds_read_b128 v[66:69], v66 offset:8192
	ds_read_b128 v[152:155], v70
	v_fma_f32 v167, v73, s3, -v150
	v_fma_f32 v172, v74, s3, -v150
	v_add3_u32 v74, s16, v181, v239
	ds_read_b128 v[168:171], v70 offset:8192
	ds_read_b128 v[70:73], v74
	v_fma_f32 v173, v75, s3, -v150
	v_mfma_f32_32x32x16_f16 v[18:33], v[216:219], v[82:85], v[18:33]
	v_fma_f32 v202, v98, s3, -v150
	v_fma_f32 v203, v99, s3, -v150
	v_fma_f32 v204, v100, s3, -v150
	v_fma_f32 v205, v102, s3, -v150
	v_fma_f32 v208, v103, s3, -v150
	v_fma_f32 v209, v104, s3, -v150
	v_fma_f32 v211, v108, s3, -v150
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[2:17], v[196:199], v[82:85], v[2:17]
	v_fma_f32 v196, v76, s3, -v150
	v_fma_f32 v197, v77, s3, -v150
	v_fma_f32 v198, v78, s3, -v150
	v_add3_u32 v78, s16, v180, v239
	v_fma_f32 v199, v79, s3, -v150
	v_add3_u32 v79, s16, v179, v239
	v_fma_f32 v179, v101, s3, -v150
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[86:89], v[138:141], 0
	v_fma_f32 v212, v110, s3, -v150
	v_fma_f32 v213, v111, s3, -v150
	v_add3_u32 v1, s16, v1, v239
	v_fma_f32 v200, v80, s3, -v150
	v_fma_f32 v201, v81, s3, -v150
	v_fma_f32 v112, v112, s3, -v150
	v_fma_f32 v113, v113, s3, -v150
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[152:155], v[134:137], v[82:97]
	ds_read_b128 v[152:155], v74 offset:8192
	ds_read_b128 v[74:77], v78
	v_fma_f32 v106, v106, s3, -v150
	v_fma_f32 v107, v107, s3, -v150
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_fmac_f32_e32 v210, 0xbe0293ee, v151
	v_exp_f32_e32 v156, v156
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[70:73], v[126:129], v[82:97]
	ds_read_b128 v[180:183], v78 offset:8192
	ds_read_b128 v[70:73], v79
	v_add3_u32 v78, s16, v178, v239
	v_fma_f32 v178, v105, s3, -v150
	v_exp_f32_e32 v157, v157
	v_exp_f32_e32 v160, v160
	v_exp_f32_e32 v161, v161
	v_exp_f32_e32 v163, v163
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[74:77], v[122:125], v[82:97]
	ds_read_b128 v[98:101], v79 offset:8192
	ds_read_b128 v[74:77], v78
	v_add3_u32 v79, s16, v177, v239
	v_fma_f32 v177, v109, s3, -v150
	v_exp_f32_e32 v164, v164
	v_exp_f32_e32 v165, v165
	v_exp_f32_e32 v167, v167
	v_exp_f32_e32 v172, v172
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[70:73], v[118:121], v[82:97]
	ds_read_b128 v[102:105], v78 offset:8192
	ds_read_b128 v[70:73], v79
	v_exp_f32_e32 v173, v173
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v197, v197
	v_exp_f32_e32 v107, v107
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[74:77], v[114:117], v[82:97]
	ds_read_b128 v[108:111], v79 offset:8192
	ds_read_b128 v[74:77], v1
	ds_read_b128 v[192:195], v1 offset:8192
	ds_bpermute_b32 v1, v188, v174
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[70:73], v[142:145], v[82:97]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[74:77], v[130:133], v[82:97]
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[138:141], 0
	v_exp_f32_e32 v138, v198
	v_exp_f32_e32 v139, v199
	v_exp_f32_e32 v140, v200
	v_exp_f32_e32 v141, v201
	v_exp_f32_e32 v198, v202
	v_exp_f32_e32 v199, v203
	v_exp_f32_e32 v200, v204
	v_mfma_f32_32x32x16_f16 v[66:81], v[168:171], v[134:137], v[66:81]
	v_exp_f32_e32 v169, v106
	v_exp_f32_e32 v106, v210
	v_exp_f32_e32 v134, v179
	v_mul_f32_e32 v50, v50, v106
	v_mul_f32_e32 v51, v51, v106
	v_mul_f32_e32 v52, v52, v106
	v_mul_f32_e32 v53, v53, v106
	v_mfma_f32_32x32x16_f16 v[66:81], v[152:155], v[126:129], v[66:81]
	v_exp_f32_e32 v154, v112
	v_exp_f32_e32 v155, v113
	v_lshrrev_b64 v[112:113], v191, exec
	v_mul_f32_e32 v54, v54, v106
	v_mul_f32_e32 v55, v55, v106
	v_mul_f32_e32 v56, v56, v106
	v_mul_f32_e32 v57, v57, v106
	v_mfma_f32_32x32x16_f16 v[66:81], v[180:183], v[122:125], v[66:81]
	v_add_u32_e32 v122, 0x8000, v206
	s_nop 0
	v_readfirstlane_b32 s3, v122
	v_add_u32_e32 v123, 0x8000, v207
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v123
	v_add_u32_e32 v122, s30, v162
	v_mul_f32_e32 v58, v58, v106
	v_mfma_f32_32x32x16_f16 v[66:81], v[98:101], v[118:121], v[66:81]
	v_and_b32_e32 v98, 1, v112
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v99, 1, v1
	v_bfrev_b32_e32 v1, 1
	v_cmp_eq_u32_e32 vcc, 1, v98
	s_nop 1
	v_cndmask_b32_e32 v98, v1, v99, vcc
	ds_bpermute_b32 v99, v188, v175
	buffer_load_dwordx4 v98, s[12:15], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[114:117], v[66:81]
	s_mov_b32 m0, s3
	v_cvt_pk_f16_f32 v102, v156, v157
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v98, 1, v99
	v_cndmask_b32_e32 v98, v1, v98, vcc
	v_add_u32_e32 v99, s34, v166
	buffer_load_dwordx4 v98, s[12:15], 0 offen lds
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[108:111], v[142:145], v[66:81]
	s_barrier
	ds_read_b64_tr_b16 v[108:109], v99 offset:32768
	ds_read_b64_tr_b16 v[110:111], v122 offset:2048
	ds_read_b64_tr_b16 v[112:113], v122 offset:4096
	ds_read_b64_tr_b16 v[114:115], v122 offset:6144
	v_cvt_pk_f16_f32 v103, v160, v161
	v_cvt_pk_f16_f32 v104, v163, v164
	v_cvt_pk_f16_f32 v105, v165, v167
	v_mul_f32_e32 v59, v59, v106
	v_mul_f32_e32 v60, v60, v106
	v_mul_f32_e32 v61, v61, v106
	v_mul_f32_e32 v62, v62, v106
	v_mul_f32_e32 v63, v63, v106
	v_mul_f32_e32 v64, v64, v106
	v_mul_f32_e32 v65, v65, v106
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[50:65], v[108:111], v[102:105], v[50:65]
	v_cvt_pk_f16_f32 v98, v172, v173
	v_cvt_pk_f16_f32 v99, v196, v197
	v_cvt_pk_f16_f32 v100, v138, v139
	v_cvt_pk_f16_f32 v101, v140, v141
	ds_read_b64_tr_b16 v[116:117], v122 offset:8192
	ds_read_b64_tr_b16 v[118:119], v122 offset:10240
	v_exp_f32_e32 v135, v205
	v_exp_f32_e32 v136, v208
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[112:115], v[98:101], v[50:65]
	v_exp_f32_e32 v137, v209
	v_exp_f32_e32 v168, v178
	v_cvt_pk_f16_f32 v108, v198, v199
	v_cvt_pk_f16_f32 v109, v200, v134
	v_cvt_pk_f16_f32 v110, v135, v136
	v_cvt_pk_f16_f32 v111, v137, v168
	ds_read_b64_tr_b16 v[120:121], v122 offset:12288
	ds_read_b64_tr_b16 v[122:123], v122 offset:14336
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[116:119], v[108:111], v[50:65]
	v_add_u32_e32 v116, s34, v158
	v_mul_f32_e32 v34, v34, v106
	v_mul_f32_e32 v35, v35, v106
	v_mul_f32_e32 v36, v36, v106
	v_mul_f32_e32 v37, v37, v106
	v_mul_f32_e32 v38, v38, v106
	v_mul_f32_e32 v39, v39, v106
	v_mfma_f32_32x32x16_f16 v[66:81], v[192:195], v[130:133], v[66:81]
	v_add_u32_e32 v130, s30, v159
	ds_read_b64_tr_b16 v[116:117], v116 offset:32768
	ds_read_b64_tr_b16 v[118:119], v130 offset:2048
	ds_read_b64_tr_b16 v[124:125], v130 offset:4096
	ds_read_b64_tr_b16 v[126:127], v130 offset:6144
	v_mul_f32_e32 v40, v40, v106
	v_mul_f32_e32 v41, v41, v106
	v_mul_f32_e32 v42, v42, v106
	v_mul_f32_e32 v43, v43, v106
	v_mul_f32_e32 v44, v44, v106
	v_mul_f32_e32 v45, v45, v106
	v_mul_f32_e32 v46, v46, v106
	v_mul_f32_e32 v47, v47, v106
	v_mul_f32_e32 v48, v48, v106
	v_mul_f32_e32 v49, v49, v106
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[34:49], v[116:119], v[102:105], v[34:49]
	v_exp_f32_e32 v128, v211
	v_exp_f32_e32 v129, v177
	v_exp_f32_e32 v152, v212
	v_exp_f32_e32 v153, v213
	v_cvt_pk_f16_f32 v112, v169, v107
	v_cvt_pk_f16_f32 v113, v128, v129
	v_cvt_pk_f16_f32 v114, v152, v153
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[124:127], v[98:101], v[34:49]
	v_cvt_pk_f16_f32 v115, v154, v155
	v_add_f32_e32 v116, v187, v189
	v_add_u32_e32 v132, s30, v149
	v_mul_f32_e32 v18, v18, v106
	v_mul_f32_e32 v19, v19, v106
	v_mul_f32_e32 v20, v20, v106
	v_mul_f32_e32 v21, v21, v106
	v_mfma_f32_32x32x16_f16 v[50:65], v[120:123], v[112:115], v[50:65]
	v_add_f32_e32 v120, v116, v190
	ds_read_b64_tr_b16 v[116:117], v130 offset:8192
	ds_read_b64_tr_b16 v[118:119], v130 offset:10240
	v_mov_b32_e32 v121, v120
	s_nop 1
	v_permlane32_swap_b32_e32 v120, v121
	v_add_f32_e32 v131, v120, v121
	v_add_f32_e32 v120, v156, v157
	v_add_f32_e32 v120, v160, v120
	v_add_f32_e32 v124, v161, v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[116:119], v[108:111], v[34:49]
	v_add_u32_e32 v116, s34, v148
	ds_read_b64_tr_b16 v[120:121], v130 offset:12288
	ds_read_b64_tr_b16 v[122:123], v130 offset:14336
	v_add_f32_e32 v130, v163, v124
	ds_read_b64_tr_b16 v[116:117], v116 offset:32768
	ds_read_b64_tr_b16 v[118:119], v132 offset:2048
	ds_read_b64_tr_b16 v[124:125], v132 offset:4096
	ds_read_b64_tr_b16 v[126:127], v132 offset:6144
	v_mul_f32_e32 v22, v22, v106
	v_mul_f32_e32 v23, v23, v106
	v_mul_f32_e32 v24, v24, v106
	v_mul_f32_e32 v25, v25, v106
	v_mul_f32_e32 v26, v26, v106
	v_mul_f32_e32 v27, v27, v106
	v_mul_f32_e32 v28, v28, v106
	v_mul_f32_e32 v29, v29, v106
	v_mul_f32_e32 v30, v30, v106
	v_mul_f32_e32 v31, v31, v106
	v_mul_f32_e32 v32, v32, v106
	v_mul_f32_e32 v33, v33, v106
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[116:119], v[102:105], v[18:33]
	v_add_f32_e32 v116, v164, v130
	v_add_f32_e32 v116, v165, v116
	v_add_f32_e32 v116, v167, v116
	v_add_f32_e32 v116, v172, v116
	v_mul_f32_e32 v2, v2, v106
	v_mul_f32_e32 v3, v3, v106
	v_mul_f32_e32 v4, v4, v106
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[124:127], v[98:101], v[18:33]
	v_mul_f32_e32 v5, v5, v106
	v_mul_f32_e32 v6, v6, v106
	v_mul_f32_e32 v7, v7, v106
	v_mul_f32_e32 v8, v8, v106
	v_mul_f32_e32 v9, v9, v106
	v_mul_f32_e32 v10, v10, v106
	v_mul_f32_e32 v11, v11, v106
	v_mfma_f32_32x32x16_f16 v[34:49], v[120:123], v[112:115], v[34:49]
	v_add_f32_e32 v120, v173, v116
	ds_read_b64_tr_b16 v[116:117], v132 offset:8192
	ds_read_b64_tr_b16 v[118:119], v132 offset:10240
	v_add_f32_e32 v120, v196, v120
	v_add_f32_e32 v120, v197, v120
	v_add_f32_e32 v120, v138, v120
	v_add_f32_e32 v120, v139, v120
	v_add_f32_e32 v124, v140, v120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[116:119], v[108:111], v[18:33]
	v_add_u32_e32 v116, s34, v146
	ds_read_b64_tr_b16 v[120:121], v132 offset:12288
	ds_read_b64_tr_b16 v[122:123], v132 offset:14336
	v_add_f32_e32 v130, v141, v124
	v_add_u32_e32 v132, s30, v147
	ds_read_b64_tr_b16 v[116:117], v116 offset:32768
	ds_read_b64_tr_b16 v[118:119], v132 offset:2048
	ds_read_b64_tr_b16 v[124:125], v132 offset:4096
	ds_read_b64_tr_b16 v[126:127], v132 offset:6144
	v_mul_f32_e32 v12, v12, v106
	v_mul_f32_e32 v13, v13, v106
	v_mul_f32_e32 v14, v14, v106
	v_mul_f32_e32 v15, v15, v106
	v_mul_f32_e32 v16, v16, v106
	v_mul_f32_e32 v17, v17, v106
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[2:17], v[116:119], v[102:105], v[2:17]
	v_add_f32_e32 v102, v198, v130
	v_add_f32_e32 v102, v199, v102
	v_add_f32_e32 v102, v200, v102
	v_add_f32_e32 v102, v134, v102
	v_add_f32_e32 v116, v135, v102
	ds_read_b64_tr_b16 v[102:103], v132 offset:8192
	ds_read_b64_tr_b16 v[104:105], v132 offset:10240
	v_fmac_f32_e32 v131, v184, v185
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[124:127], v[98:101], v[2:17]
	v_add_f32_e32 v98, v136, v116
	v_add_f32_e32 v98, v137, v98
	v_add_f32_e32 v98, v168, v98
	v_add_f32_e32 v98, v169, v98
	v_add_f32_e32 v107, v107, v98
	ds_read_b64_tr_b16 v[98:99], v132 offset:12288
	ds_read_b64_tr_b16 v[100:101], v132 offset:14336
	v_add_u32_e32 v132, s2, v159
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[102:105], v[108:111], v[2:17]
	v_add_f32_e32 v102, v128, v107
	v_add_f32_e32 v102, v129, v102
	v_add_f32_e32 v102, v152, v102
	v_add_f32_e32 v102, v153, v102
	v_add_f32_e32 v102, v154, v102
	v_add_f32_e32 v102, v155, v102
	v_mov_b32_e32 v103, v102
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[98:101], v[112:115], v[2:17]
	v_max_f32_e32 v98, v83, v83
	v_max_f32_e32 v99, v82, v82
	v_max_f32_e32 v98, v99, v98
	v_max3_f32 v98, v98, v84, v85
	v_max3_f32 v98, v98, v86, v87
	v_max3_f32 v98, v98, v88, v89
	v_max3_f32 v98, v98, v90, v91
	v_max3_f32 v98, v98, v92, v93
	v_max3_f32 v98, v98, v94, v95
	v_max3_f32 v98, v98, v96, v97
	v_max3_f32 v98, v98, v66, v67
	v_max3_f32 v98, v98, v68, v69
	v_max3_f32 v98, v98, v70, v71
	v_max3_f32 v98, v98, v72, v73
	v_max3_f32 v98, v98, v74, v75
	v_max3_f32 v98, v98, v76, v77
	v_max3_f32 v98, v98, v78, v79
	v_max3_f32 v98, v98, v80, v81
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_max3_f32 v98, v151, v98, v99
	v_mul_f32_e32 v99, 0xbe0293ee, v98
	v_fmamk_f32 v67, v67, 0x3e0293ee, v99
	v_fmamk_f32 v82, v82, 0x3e0293ee, v99
	v_fmamk_f32 v83, v83, 0x3e0293ee, v99
	v_fmamk_f32 v84, v84, 0x3e0293ee, v99
	v_fmamk_f32 v85, v85, 0x3e0293ee, v99
	v_fmamk_f32 v86, v86, 0x3e0293ee, v99
	v_fmamk_f32 v87, v87, 0x3e0293ee, v99
	v_fmamk_f32 v88, v88, 0x3e0293ee, v99
	v_fmamk_f32 v89, v89, 0x3e0293ee, v99
	v_fmamk_f32 v74, v74, 0x3e0293ee, v99
	v_fmamk_f32 v75, v75, 0x3e0293ee, v99
	v_fmamk_f32 v76, v76, 0x3e0293ee, v99
	v_fmamk_f32 v77, v77, 0x3e0293ee, v99
	v_fmamk_f32 v78, v78, 0x3e0293ee, v99
	v_fmamk_f32 v79, v79, 0x3e0293ee, v99
	v_fmamk_f32 v80, v80, 0x3e0293ee, v99
	v_exp_f32_e32 v116, v67
	v_fmac_f32_e32 v150, 0xbe0293ee, v98
	v_add_u32_e32 v67, s16, v166
	v_mfma_f32_32x32x16_f16 v[18:33], v[120:123], v[112:115], v[18:33]
	v_fmamk_f32 v90, v90, 0x3e0293ee, v99
	v_fmamk_f32 v91, v91, 0x3e0293ee, v99
	v_fmamk_f32 v92, v92, 0x3e0293ee, v99
	v_fmamk_f32 v93, v93, 0x3e0293ee, v99
	v_fmamk_f32 v94, v94, 0x3e0293ee, v99
	v_fmamk_f32 v95, v95, 0x3e0293ee, v99
	v_fmamk_f32 v96, v96, 0x3e0293ee, v99
	v_fmamk_f32 v97, v97, 0x3e0293ee, v99
	v_fmamk_f32 v66, v66, 0x3e0293ee, v99
	v_fmamk_f32 v68, v68, 0x3e0293ee, v99
	v_fmamk_f32 v69, v69, 0x3e0293ee, v99
	v_fmamk_f32 v70, v70, 0x3e0293ee, v99
	v_fmamk_f32 v71, v71, 0x3e0293ee, v99
	v_fmamk_f32 v72, v72, 0x3e0293ee, v99
	v_fmamk_f32 v73, v73, 0x3e0293ee, v99
	v_fmac_f32_e32 v99, 0x3e0293ee, v81
	v_exp_f32_e32 v100, v82
	v_exp_f32_e32 v101, v83
	v_exp_f32_e32 v104, v84
	v_exp_f32_e32 v105, v85
	v_exp_f32_e32 v107, v86
	v_exp_f32_e32 v108, v87
	v_exp_f32_e32 v109, v88
	v_exp_f32_e32 v110, v89
	v_exp_f32_e32 v123, v74
	v_exp_f32_e32 v124, v75
	v_exp_f32_e32 v125, v76
	v_exp_f32_e32 v126, v77
	v_exp_f32_e32 v127, v78
	v_exp_f32_e32 v128, v79
	v_exp_f32_e32 v129, v80
	v_exp_f32_e32 v130, v150
	v_add_u32_e32 v88, s2, v162
	ds_read_b64_tr_b16 v[74:75], v67 offset:32768
	ds_read_b64_tr_b16 v[76:77], v88 offset:2048
	ds_read_b64_tr_b16 v[78:79], v88 offset:4096
	ds_read_b64_tr_b16 v[80:81], v88 offset:6144
	v_exp_f32_e32 v119, v70
	v_exp_f32_e32 v120, v71
	v_exp_f32_e32 v121, v72
	v_exp_f32_e32 v122, v73
	v_cvt_pk_f16_f32 v70, v100, v101
	v_cvt_pk_f16_f32 v71, v104, v105
	v_cvt_pk_f16_f32 v72, v107, v108
	v_cvt_pk_f16_f32 v73, v109, v110
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
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[50:65], v[74:77], v[70:73], v[50:65]
	v_exp_f32_e32 v111, v90
	v_exp_f32_e32 v112, v91
	v_exp_f32_e32 v113, v92
	v_exp_f32_e32 v114, v93
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_exp_f32_e32 v115, v66
	v_exp_f32_e32 v117, v68
	v_exp_f32_e32 v118, v69
	v_cvt_pk_f16_f32 v66, v111, v112
	v_cvt_pk_f16_f32 v67, v113, v114
	v_cvt_pk_f16_f32 v68, v94, v95
	v_cvt_pk_f16_f32 v69, v96, v97
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[50:65], v[78:81], v[66:69], v[50:65]
	ds_read_b64_tr_b16 v[82:83], v88 offset:8192
	ds_read_b64_tr_b16 v[84:85], v88 offset:10240
	v_cvt_pk_f16_f32 v74, v115, v116
	v_cvt_pk_f16_f32 v75, v117, v118
	v_cvt_pk_f16_f32 v76, v119, v120
	v_cvt_pk_f16_f32 v77, v121, v122
	ds_read_b64_tr_b16 v[86:87], v88 offset:12288
	ds_read_b64_tr_b16 v[88:89], v88 offset:14336
	v_mul_f32_e32 v34, v34, v130
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[82:85], v[74:77], v[50:65]
	v_add_u32_e32 v82, s16, v158
	ds_read_b64_tr_b16 v[82:83], v82 offset:32768
	ds_read_b64_tr_b16 v[84:85], v132 offset:2048
	ds_read_b64_tr_b16 v[90:91], v132 offset:4096
	ds_read_b64_tr_b16 v[92:93], v132 offset:6144
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
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[70:73], v[34:49]
	v_exp_f32_e32 v99, v99
	v_cvt_pk_f16_f32 v78, v123, v124
	v_cvt_pk_f16_f32 v79, v125, v126
	v_cvt_pk_f16_f32 v80, v127, v128
	v_cvt_pk_f16_f32 v81, v129, v99
	v_add_f32_e32 v82, v100, v101
	v_add_u32_e32 v101, s2, v149
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[90:93], v[66:69], v[34:49]
	v_mul_f32_e32 v18, v18, v130
	v_mul_f32_e32 v19, v19, v130
	v_mul_f32_e32 v20, v20, v130
	v_mul_f32_e32 v21, v21, v130
	v_mul_f32_e32 v22, v22, v130
	v_mul_f32_e32 v23, v23, v130
	v_mul_f32_e32 v24, v24, v130
	v_mfma_f32_32x32x16_f16 v[50:65], v[86:89], v[78:81], v[50:65]
	v_add_f32_e32 v86, v104, v82
	ds_read_b64_tr_b16 v[82:83], v132 offset:8192
	ds_read_b64_tr_b16 v[84:85], v132 offset:10240
	v_add_f32_e32 v86, v105, v86
	v_add_f32_e32 v86, v107, v86
	v_add_f32_e32 v86, v108, v86
	v_add_f32_e32 v86, v109, v86
	v_add_f32_e32 v90, v110, v86
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[74:77], v[34:49]
	v_add_u32_e32 v82, s16, v148
	ds_read_b64_tr_b16 v[86:87], v132 offset:12288
	ds_read_b64_tr_b16 v[88:89], v132 offset:14336
	v_add_f32_e32 v100, v111, v90
	ds_read_b64_tr_b16 v[82:83], v82 offset:32768
	ds_read_b64_tr_b16 v[84:85], v101 offset:2048
	ds_read_b64_tr_b16 v[90:91], v101 offset:4096
	ds_read_b64_tr_b16 v[92:93], v101 offset:6144
	v_mul_f32_e32 v25, v25, v130
	v_mul_f32_e32 v26, v26, v130
	v_mul_f32_e32 v27, v27, v130
	v_mul_f32_e32 v28, v28, v130
	v_mul_f32_e32 v29, v29, v130
	v_mul_f32_e32 v30, v30, v130
	v_mul_f32_e32 v31, v31, v130
	v_mul_f32_e32 v32, v32, v130
	v_mul_f32_e32 v33, v33, v130
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[70:73], v[18:33]
	v_add_f32_e32 v82, v112, v100
	v_add_f32_e32 v82, v113, v82
	v_add_f32_e32 v82, v114, v82
	v_add_f32_e32 v82, v94, v82
	v_mul_f32_e32 v2, v2, v130
	v_mul_f32_e32 v3, v3, v130
	v_mul_f32_e32 v4, v4, v130
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[90:93], v[66:69], v[18:33]
	v_mul_f32_e32 v5, v5, v130
	v_mul_f32_e32 v6, v6, v130
	v_mul_f32_e32 v7, v7, v130
	v_mul_f32_e32 v8, v8, v130
	v_mul_f32_e32 v9, v9, v130
	v_mul_f32_e32 v10, v10, v130
	v_mul_f32_e32 v11, v11, v130
	v_mfma_f32_32x32x16_f16 v[34:49], v[86:89], v[78:81], v[34:49]
	v_add_f32_e32 v86, v95, v82
	ds_read_b64_tr_b16 v[82:83], v101 offset:8192
	ds_read_b64_tr_b16 v[84:85], v101 offset:10240
	v_add_f32_e32 v86, v96, v86
	v_add_f32_e32 v86, v97, v86
	v_add_f32_e32 v86, v115, v86
	v_add_f32_e32 v86, v116, v86
	v_add_f32_e32 v90, v117, v86
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[74:77], v[18:33]
	v_add_u32_e32 v82, s16, v146
	ds_read_b64_tr_b16 v[86:87], v101 offset:12288
	ds_read_b64_tr_b16 v[88:89], v101 offset:14336
	v_add_f32_e32 v94, v118, v90
	v_add_u32_e32 v95, s2, v147
	ds_read_b64_tr_b16 v[82:83], v82 offset:32768
	ds_read_b64_tr_b16 v[84:85], v95 offset:2048
	ds_read_b64_tr_b16 v[90:91], v95 offset:4096
	ds_read_b64_tr_b16 v[92:93], v95 offset:6144
	v_mul_f32_e32 v12, v12, v130
	v_mul_f32_e32 v13, v13, v130
	v_mul_f32_e32 v14, v14, v130
	v_mul_f32_e32 v15, v15, v130
	v_mul_f32_e32 v16, v16, v130
	v_mul_f32_e32 v17, v17, v130
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[70:73], v[2:17]
	v_add_f32_e32 v70, v119, v94
	v_add_f32_e32 v70, v120, v70
	v_add_f32_e32 v70, v121, v70
	v_add_f32_e32 v70, v122, v70
	v_add_f32_e32 v82, v123, v70
	ds_read_b64_tr_b16 v[70:71], v95 offset:8192
	ds_read_b64_tr_b16 v[72:73], v95 offset:10240
	v_permlane32_swap_b32_e32 v102, v103
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[90:93], v[66:69], v[2:17]
	v_add_f32_e32 v66, v124, v82
	ds_read_b64_tr_b16 v[82:83], v95 offset:12288
	ds_read_b64_tr_b16 v[84:85], v95 offset:14336
	v_add_f32_e32 v66, v125, v66
	v_add_f32_e32 v66, v126, v66
	v_add_f32_e32 v66, v127, v66
	v_add_f32_e32 v66, v128, v66
	v_add_f32_e32 v66, v129, v66
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[70:73], v[74:77], v[2:17]
	v_add_f32_e32 v66, v99, v66
	v_mov_b32_e32 v67, v66
	v_add_f32_e32 v102, v102, v103
	s_nop 0
	v_permlane32_swap_b32_e32 v66, v67
	v_fmac_f32_e32 v102, v131, v106
	v_add_f32_e32 v66, v66, v67
	v_mfma_f32_32x32x16_f16 v[18:33], v[86:89], v[78:81], v[18:33]
	v_fmac_f32_e32 v66, v102, v130
	v_lshl_add_u32 v67, v186, 2, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[78:81], v[2:17]
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v66
	s_nop 1
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_or_b32_e32 v68, s28, v186
	s_movk_i32 s2, 0x4000
	v_ldexp_f32 v69, v66, v69
	v_cmp_gt_i32_e64 s[8:9], s2, v68
	v_mov_b32_e32 v68, 0x42000000
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v68, 0, v68, vcc
	v_sub_f32_e32 v68, v69, v68
	v_add_f32_e32 v68, v98, v68
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
	v_cndmask_b32_e32 v1, v1, v68, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v69, v1, s[4:7], 0 offen
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_9:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v66
	s_nop 1
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v66, v68
	v_mov_b32_e32 v1, 0x42000000
	v_log_f32_e32 v68, v68
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_sub_f32_e32 v1, v68, v1
	v_add_f32_e32 v1, v98, v1
	ds_write_b32 v67, v1
	v_mov_b32_e32 v1, 2
	v_lshlrev_b32_sdwa v1, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v67, 0, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v67, v67
	v_bfrev_b32_e32 v68, 1
	s_and_b32 s5, s19, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e64 v1, v68, v1, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v1, s[4:7], 0 offen
.LBB0_10:                               ; %.critedge
	v_and_b32_e32 v67, 32, v0
	v_lshlrev_b32_e32 v68, 2, v0
	v_div_scale_f32 v0, s[0:1], v66, v66, 1.0
	v_rcp_f32_e32 v0, v0
	v_div_scale_f32 v1, vcc, 1.0, v66, 1.0
	v_mul_f32_e32 v0, v1, v0
	s_nop 2
	v_div_fmas_f32 v0, 0, 0, v0
	v_div_fixup_f32 v0, v0, v66, 1.0
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v66, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[32:33] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v32, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v30, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[28:29] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v28, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v26, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[24:25] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v24, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[22:23] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v22, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[20:21] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v20, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v18, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[48:49] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v48, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[46:47] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v46, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[44:45] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v44, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[42:43] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v42, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[40:41] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v40, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[38:39] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v38, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[36:37] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v36, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v34, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[64:65] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v64, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[62:63] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v62, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[60:61] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v60, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[58:59] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v58, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[56:57] op_sel_hi:[0,1]
	s_mul_i32 s0, s25, s18
	v_cvt_pk_f16_f32 v56, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[54:55] op_sel_hi:[0,1]
	s_ashr_i32 s1, s0, 31
	v_pk_mul_f32 v[16:17], v[0:1], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[0:1], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[0:1], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[0:1], v[4:5] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v54, v2, v3
	v_pk_mul_f32 v[2:3], v[0:1], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[0:1], v[50:51] op_sel_hi:[0,1]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v16, v16, v17
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v12, v12, v13
	v_cvt_pk_f16_f32 v10, v10, v11
	v_cvt_pk_f16_f32 v8, v8, v9
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v4, v4, v5
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v0, v0, v1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s26, s17
	v_lshrrev_b32_e32 v17, 16, v16
	v_lshrrev_b32_e32 v15, 16, v14
	v_lshrrev_b32_e32 v13, 16, v12
	v_lshrrev_b32_e32 v11, 16, v10
	v_lshrrev_b32_e32 v9, 16, v8
	v_lshrrev_b32_e32 v7, 16, v6
	v_lshrrev_b32_e32 v5, 16, v4
	v_lshrrev_b32_e32 v69, 16, v66
	v_lshrrev_b32_e32 v33, 16, v32
	v_lshrrev_b32_e32 v31, 16, v30
	v_lshrrev_b32_e32 v29, 16, v28
	v_lshrrev_b32_e32 v27, 16, v26
	v_lshrrev_b32_e32 v25, 16, v24
	v_lshrrev_b32_e32 v23, 16, v22
	v_lshrrev_b32_e32 v21, 16, v20
	v_lshrrev_b32_e32 v19, 16, v18
	v_lshrrev_b32_e32 v49, 16, v48
	v_lshrrev_b32_e32 v47, 16, v46
	v_lshrrev_b32_e32 v45, 16, v44
	v_lshrrev_b32_e32 v43, 16, v42
	v_lshrrev_b32_e32 v41, 16, v40
	v_lshrrev_b32_e32 v39, 16, v38
	v_lshrrev_b32_e32 v37, 16, v36
	v_lshrrev_b32_e32 v35, 16, v34
	v_lshrrev_b32_e32 v65, 16, v64
	v_lshrrev_b32_e32 v63, 16, v62
	v_lshrrev_b32_e32 v61, 16, v60
	v_lshrrev_b32_e32 v59, 16, v58
	v_lshrrev_b32_e32 v57, 16, v56
	v_lshrrev_b32_e32 v55, 16, v54
	v_lshrrev_b32_e32 v3, 16, v2
	v_lshrrev_b32_e32 v1, 16, v0
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	v_cmp_eq_u32_e32 vcc, 0, v67
	v_lshrrev_b32_e32 v50, 2, v67
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cndmask_b32_e32 v52, v54, v0, vcc
	v_and_b32_e32 v53, 0xfc, v68
	v_cndmask_b32_e32 v68, v55, v1, vcc
	v_cndmask_b32_e32 v70, v56, v2, vcc
	v_cndmask_b32_e32 v71, v57, v3, vcc
	v_cndmask_b32_e32 v0, v0, v54, vcc
	v_bitop3_b32 v54, v176, v67, 32 bitop3:0x36
	v_cndmask_b32_e32 v1, v1, v55, vcc
	v_cndmask_b32_e32 v2, v2, v56, vcc
	v_cndmask_b32_e32 v3, v3, v57, vcc
	v_cndmask_b32_e32 v55, v62, v58, vcc
	v_cndmask_b32_e32 v56, v63, v59, vcc
	v_cndmask_b32_e32 v57, v64, v60, vcc
	v_cndmask_b32_e32 v67, v65, v61, vcc
	v_cndmask_b32_e32 v58, v58, v62, vcc
	v_cndmask_b32_e32 v59, v59, v63, vcc
	v_cndmask_b32_e32 v60, v60, v64, vcc
	v_cndmask_b32_e32 v61, v61, v65, vcc
	v_cndmask_b32_e32 v62, v38, v34, vcc
	v_cndmask_b32_e32 v63, v39, v35, vcc
	v_cndmask_b32_e32 v64, v40, v36, vcc
	v_cndmask_b32_e32 v65, v41, v37, vcc
	v_cndmask_b32_e32 v34, v34, v38, vcc
	v_cndmask_b32_e32 v35, v35, v39, vcc
	v_cndmask_b32_e32 v36, v36, v40, vcc
	v_cndmask_b32_e32 v37, v37, v41, vcc
	v_cndmask_b32_e32 v38, v46, v42, vcc
	v_cndmask_b32_e32 v39, v47, v43, vcc
	v_cndmask_b32_e32 v40, v48, v44, vcc
	v_cndmask_b32_e32 v41, v49, v45, vcc
	v_cndmask_b32_e32 v42, v42, v46, vcc
	v_cndmask_b32_e32 v43, v43, v47, vcc
	v_cndmask_b32_e32 v44, v44, v48, vcc
	v_cndmask_b32_e32 v45, v45, v49, vcc
	v_cndmask_b32_e32 v46, v22, v18, vcc
	v_cndmask_b32_e32 v47, v23, v19, vcc
	v_cndmask_b32_e32 v48, v24, v20, vcc
	v_cndmask_b32_e32 v49, v25, v21, vcc
	v_cndmask_b32_e32 v18, v18, v22, vcc
	v_cndmask_b32_e32 v19, v19, v23, vcc
	v_cndmask_b32_e32 v20, v20, v24, vcc
	v_cndmask_b32_e32 v21, v21, v25, vcc
	v_cndmask_b32_e32 v22, v30, v26, vcc
	v_cndmask_b32_e32 v23, v31, v27, vcc
	v_cndmask_b32_e32 v24, v32, v28, vcc
	v_cndmask_b32_e32 v25, v33, v29, vcc
	v_cndmask_b32_e32 v26, v26, v30, vcc
	v_cndmask_b32_e32 v27, v27, v31, vcc
	v_cndmask_b32_e32 v28, v28, v32, vcc
	v_cndmask_b32_e32 v29, v29, v33, vcc
	v_cndmask_b32_e32 v30, v6, v66, vcc
	v_cndmask_b32_e32 v31, v7, v69, vcc
	v_cndmask_b32_e32 v32, v8, v4, vcc
	v_cndmask_b32_e32 v33, v9, v5, vcc
	v_cndmask_b32_e32 v6, v66, v6, vcc
	v_cndmask_b32_e32 v7, v69, v7, vcc
	v_cndmask_b32_e32 v4, v4, v8, vcc
	v_cndmask_b32_e32 v5, v5, v9, vcc
	v_cndmask_b32_e32 v8, v14, v10, vcc
	v_cndmask_b32_e32 v9, v15, v11, vcc
	v_cndmask_b32_e32 v66, v16, v12, vcc
	v_cndmask_b32_e32 v69, v17, v13, vcc
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_cndmask_b32_e32 v12, v12, v16, vcc
	v_cndmask_b32_e32 v13, v13, v17, vcc
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s27, s28
	v_bfe_i32 v52, v52, 0, 16
	v_bfe_i32 v68, v68, 0, 16
	v_bfe_i32 v70, v70, 0, 16
	v_bfe_i32 v71, v71, 0, 16
	v_bfe_i32 v0, v0, 0, 16
	v_lshlrev_b32_e32 v54, 2, v54
	v_bfe_i32 v1, v1, 0, 16
	v_bfe_i32 v2, v2, 0, 16
	v_bfe_i32 v3, v3, 0, 16
	v_bfe_i32 v55, v55, 0, 16
	v_bfe_i32 v56, v56, 0, 16
	v_bfe_i32 v57, v57, 0, 16
	v_bfe_i32 v67, v67, 0, 16
	v_bfe_i32 v58, v58, 0, 16
	v_bfe_i32 v59, v59, 0, 16
	v_bfe_i32 v60, v60, 0, 16
	v_bfe_i32 v61, v61, 0, 16
	v_bfe_i32 v62, v62, 0, 16
	v_bfe_i32 v63, v63, 0, 16
	v_bfe_i32 v64, v64, 0, 16
	v_bfe_i32 v65, v65, 0, 16
	v_bfe_i32 v34, v34, 0, 16
	v_bfe_i32 v35, v35, 0, 16
	v_bfe_i32 v36, v36, 0, 16
	v_bfe_i32 v37, v37, 0, 16
	v_bfe_i32 v38, v38, 0, 16
	v_bfe_i32 v39, v39, 0, 16
	v_bfe_i32 v40, v40, 0, 16
	v_bfe_i32 v41, v41, 0, 16
	v_bfe_i32 v42, v42, 0, 16
	v_bfe_i32 v43, v43, 0, 16
	v_bfe_i32 v44, v44, 0, 16
	v_bfe_i32 v45, v45, 0, 16
	v_bfe_i32 v46, v46, 0, 16
	v_bfe_i32 v47, v47, 0, 16
	v_bfe_i32 v48, v48, 0, 16
	v_bfe_i32 v49, v49, 0, 16
	v_bfe_i32 v18, v18, 0, 16
	v_bfe_i32 v19, v19, 0, 16
	v_bfe_i32 v20, v20, 0, 16
	v_bfe_i32 v21, v21, 0, 16
	v_bfe_i32 v22, v22, 0, 16
	v_bfe_i32 v23, v23, 0, 16
	v_bfe_i32 v24, v24, 0, 16
	v_bfe_i32 v25, v25, 0, 16
	v_bfe_i32 v26, v26, 0, 16
	v_bfe_i32 v27, v27, 0, 16
	v_bfe_i32 v28, v28, 0, 16
	v_bfe_i32 v29, v29, 0, 16
	v_bfe_i32 v30, v30, 0, 16
	v_bfe_i32 v31, v31, 0, 16
	v_bfe_i32 v32, v32, 0, 16
	v_bfe_i32 v33, v33, 0, 16
	v_bfe_i32 v6, v6, 0, 16
	v_bfe_i32 v7, v7, 0, 16
	v_bfe_i32 v4, v4, 0, 16
	v_bfe_i32 v5, v5, 0, 16
	v_bfe_i32 v8, v8, 0, 16
	v_bfe_i32 v9, v9, 0, 16
	v_bfe_i32 v66, v66, 0, 16
	v_bfe_i32 v69, v69, 0, 16
	v_bfe_i32 v10, v10, 0, 16
	v_bfe_i32 v11, v11, 0, 16
	v_bfe_i32 v12, v12, 0, 16
	v_bfe_i32 v13, v13, 0, 16
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	ds_bpermute_b32 v52, v53, v52
	ds_bpermute_b32 v68, v53, v68
	ds_bpermute_b32 v70, v53, v70
	ds_bpermute_b32 v71, v53, v71
	ds_bpermute_b32 v0, v54, v0
	ds_bpermute_b32 v1, v54, v1
	ds_bpermute_b32 v2, v54, v2
	ds_bpermute_b32 v3, v54, v3
	ds_bpermute_b32 v55, v53, v55
	ds_bpermute_b32 v56, v53, v56
	ds_bpermute_b32 v57, v53, v57
	ds_bpermute_b32 v67, v53, v67
	ds_bpermute_b32 v58, v54, v58
	ds_bpermute_b32 v59, v54, v59
	ds_bpermute_b32 v60, v54, v60
	ds_bpermute_b32 v61, v54, v61
	ds_bpermute_b32 v62, v53, v62
	ds_bpermute_b32 v63, v53, v63
	ds_bpermute_b32 v64, v53, v64
	ds_bpermute_b32 v65, v53, v65
	ds_bpermute_b32 v34, v54, v34
	ds_bpermute_b32 v35, v54, v35
	ds_bpermute_b32 v36, v54, v36
	ds_bpermute_b32 v37, v54, v37
	ds_bpermute_b32 v38, v53, v38
	ds_bpermute_b32 v39, v53, v39
	ds_bpermute_b32 v40, v53, v40
	ds_bpermute_b32 v41, v53, v41
	ds_bpermute_b32 v42, v54, v42
	ds_bpermute_b32 v43, v54, v43
	ds_bpermute_b32 v44, v54, v44
	ds_bpermute_b32 v45, v54, v45
	ds_bpermute_b32 v46, v53, v46
	ds_bpermute_b32 v47, v53, v47
	ds_bpermute_b32 v48, v53, v48
	ds_bpermute_b32 v49, v53, v49
	ds_bpermute_b32 v18, v54, v18
	ds_bpermute_b32 v19, v54, v19
	ds_bpermute_b32 v20, v54, v20
	ds_bpermute_b32 v21, v54, v21
	ds_bpermute_b32 v22, v53, v22
	ds_bpermute_b32 v23, v53, v23
	ds_bpermute_b32 v24, v53, v24
	ds_bpermute_b32 v25, v53, v25
	ds_bpermute_b32 v26, v54, v26
	ds_bpermute_b32 v27, v54, v27
	ds_bpermute_b32 v28, v54, v28
	ds_bpermute_b32 v29, v54, v29
	ds_bpermute_b32 v30, v53, v30
	ds_bpermute_b32 v31, v53, v31
	ds_bpermute_b32 v32, v53, v32
	ds_bpermute_b32 v33, v53, v33
	ds_bpermute_b32 v6, v54, v6
	ds_bpermute_b32 v7, v54, v7
	ds_bpermute_b32 v4, v54, v4
	ds_bpermute_b32 v5, v54, v5
	ds_bpermute_b32 v8, v53, v8
	ds_bpermute_b32 v9, v53, v9
	ds_bpermute_b32 v66, v53, v66
	ds_bpermute_b32 v53, v53, v69
	ds_bpermute_b32 v10, v54, v10
	ds_bpermute_b32 v11, v54, v11
	ds_bpermute_b32 v12, v54, v12
	ds_bpermute_b32 v13, v54, v13
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s27, 0x3fff
	v_mul_lo_u32 v51, s27, v186
	s_bitset1_b32 s2, 14
	s_waitcnt lgkmcnt(14)
	v_cndmask_b32_e32 v14, v0, v52, vcc
	v_cndmask_b32_e32 v15, v1, v68, vcc
	v_cndmask_b32_e32 v16, v2, v70, vcc
	v_cndmask_b32_e32 v17, v3, v71, vcc
	v_cndmask_b32_e32 v0, v52, v0, vcc
	v_cndmask_b32_e32 v1, v68, v1, vcc
	v_cndmask_b32_e32 v2, v70, v2, vcc
	v_cndmask_b32_e32 v3, v71, v3, vcc
	v_cndmask_b32_e32 v52, v58, v55, vcc
	v_cndmask_b32_e32 v54, v59, v56, vcc
	v_cndmask_b32_e32 v68, v60, v57, vcc
	v_cndmask_b32_e32 v69, v61, v67, vcc
	v_cndmask_b32_e32 v55, v55, v58, vcc
	v_cndmask_b32_e32 v56, v56, v59, vcc
	v_cndmask_b32_e32 v57, v57, v60, vcc
	v_cndmask_b32_e32 v58, v67, v61, vcc
	v_cndmask_b32_e32 v59, v34, v62, vcc
	v_cndmask_b32_e32 v60, v35, v63, vcc
	v_cndmask_b32_e32 v61, v36, v64, vcc
	v_cndmask_b32_e32 v67, v37, v65, vcc
	v_cndmask_b32_e32 v34, v62, v34, vcc
	v_cndmask_b32_e32 v35, v63, v35, vcc
	v_cndmask_b32_e32 v36, v64, v36, vcc
	v_cndmask_b32_e32 v37, v65, v37, vcc
	v_cndmask_b32_e32 v62, v42, v38, vcc
	v_cndmask_b32_e32 v63, v43, v39, vcc
	v_cndmask_b32_e32 v64, v44, v40, vcc
	v_cndmask_b32_e32 v65, v45, v41, vcc
	v_cndmask_b32_e32 v38, v38, v42, vcc
	v_cndmask_b32_e32 v39, v39, v43, vcc
	v_cndmask_b32_e32 v40, v40, v44, vcc
	v_cndmask_b32_e32 v41, v41, v45, vcc
	v_cndmask_b32_e32 v42, v18, v46, vcc
	v_cndmask_b32_e32 v43, v19, v47, vcc
	v_cndmask_b32_e32 v44, v20, v48, vcc
	v_cndmask_b32_e32 v45, v21, v49, vcc
	v_cndmask_b32_e32 v18, v46, v18, vcc
	v_cndmask_b32_e32 v19, v47, v19, vcc
	v_cndmask_b32_e32 v20, v48, v20, vcc
	v_cndmask_b32_e32 v21, v49, v21, vcc
	v_cndmask_b32_e32 v46, v26, v22, vcc
	v_cndmask_b32_e32 v47, v27, v23, vcc
	v_cndmask_b32_e32 v48, v28, v24, vcc
	v_cndmask_b32_e32 v49, v29, v25, vcc
	v_cndmask_b32_e32 v22, v22, v26, vcc
	v_cndmask_b32_e32 v23, v23, v27, vcc
	v_cndmask_b32_e32 v24, v24, v28, vcc
	v_cndmask_b32_e32 v25, v25, v29, vcc
	s_waitcnt lgkmcnt(11)
	v_cndmask_b32_e32 v26, v6, v30, vcc
	s_waitcnt lgkmcnt(10)
	v_cndmask_b32_e32 v27, v7, v31, vcc
	s_waitcnt lgkmcnt(9)
	v_cndmask_b32_e32 v28, v4, v32, vcc
	s_waitcnt lgkmcnt(8)
	v_cndmask_b32_e32 v29, v5, v33, vcc
	v_cndmask_b32_e32 v6, v30, v6, vcc
	v_cndmask_b32_e32 v7, v31, v7, vcc
	v_cndmask_b32_e32 v4, v32, v4, vcc
	v_cndmask_b32_e32 v5, v33, v5, vcc
	s_waitcnt lgkmcnt(3)
	v_cndmask_b32_e32 v30, v10, v8, vcc
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e32 v31, v11, v9, vcc
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e32 v32, v12, v66, vcc
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e32 v33, v13, v53, vcc
	v_cndmask_b32_e32 v8, v8, v10, vcc
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_cndmask_b32_e32 v10, v66, v12, vcc
	v_cndmask_b32_e32 v11, v53, v13, vcc
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	s_mov_b32 s4, 0x5040100
	v_add_lshl_u32 v12, v51, v50, 1
	v_bfrev_b32_e32 v13, 1
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_perm_b32 v3, v3, v2, s4
	v_perm_b32 v2, v1, v0, s4
	v_perm_b32 v1, v17, v16, s4
	v_perm_b32 v0, v15, v14, s4
	v_cndmask_b32_e64 v14, v13, v12, s[8:9]
	buffer_store_dwordx4 v[0:3], v14, s[0:3], 0 offen
	v_add_u32_e32 v14, 32, v12
	s_nop 0
	v_perm_b32 v3, v58, v57, s4
	v_perm_b32 v2, v56, v55, s4
	v_perm_b32 v1, v69, v68, s4
	v_perm_b32 v0, v54, v52, s4
	v_cndmask_b32_e64 v14, v13, v14, s[8:9]
	buffer_store_dwordx4 v[0:3], v14, s[0:3], 0 offen
	v_add_u32_e32 v14, 64, v12
	s_nop 0
	v_perm_b32 v3, v37, v36, s4
	v_perm_b32 v2, v35, v34, s4
	v_perm_b32 v1, v67, v61, s4
	v_perm_b32 v0, v60, v59, s4
	v_cndmask_b32_e64 v14, v13, v14, s[8:9]
	buffer_store_dwordx4 v[0:3], v14, s[0:3], 0 offen
	v_add_u32_e32 v14, 0x60, v12
	s_nop 0
	v_perm_b32 v3, v41, v40, s4
	v_perm_b32 v2, v39, v38, s4
	v_perm_b32 v1, v65, v64, s4
	v_perm_b32 v0, v63, v62, s4
	v_cndmask_b32_e64 v14, v13, v14, s[8:9]
	buffer_store_dwordx4 v[0:3], v14, s[0:3], 0 offen
	v_add_u32_e32 v14, 0x80, v12
	s_nop 0
	v_perm_b32 v3, v21, v20, s4
	v_perm_b32 v2, v19, v18, s4
	v_perm_b32 v1, v45, v44, s4
	v_perm_b32 v0, v43, v42, s4
	v_cndmask_b32_e64 v14, v13, v14, s[8:9]
	buffer_store_dwordx4 v[0:3], v14, s[0:3], 0 offen
	v_add_u32_e32 v14, 0xa0, v12
	s_nop 0
	v_perm_b32 v3, v25, v24, s4
	v_perm_b32 v2, v23, v22, s4
	v_perm_b32 v1, v49, v48, s4
	v_perm_b32 v0, v47, v46, s4
	v_cndmask_b32_e64 v14, v13, v14, s[8:9]
	buffer_store_dwordx4 v[0:3], v14, s[0:3], 0 offen
	s_nop 1
	v_perm_b32 v3, v5, v4, s4
	v_add_u32_e32 v4, 0xc0, v12
	v_perm_b32 v2, v7, v6, s4
	v_perm_b32 v1, v29, v28, s4
	v_perm_b32 v0, v27, v26, s4
	v_cndmask_b32_e64 v4, v13, v4, s[8:9]
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xe0, v12
	s_nop 0
	v_perm_b32 v3, v11, v10, s4
	v_perm_b32 v2, v9, v8, s4
	v_perm_b32 v1, v33, v32, s4
	v_perm_b32 v0, v31, v30, s4
	v_cndmask_b32_e64 v4, v13, v4, s[8:9]
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
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
		.amdhsa_next_free_vgpr 253
		.amdhsa_next_free_sgpr 40
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
	.set attn_fwd.num_vgpr, 253
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 40
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14132
; TotalNumSgprs: 46
; NumVgprs: 253
; NumAgprs: 0
; TotalNumVgprs: 253
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 253
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
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
	.asciz	"flash-attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/app/OAI-triton/fa"            ; string offset=26
.Linfo_string3:
	.asciz	"attn_fwd"                      ; string offset=45
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
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     253
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
