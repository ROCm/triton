	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
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
	s_mul_i32 s20, s12, s18
	s_ashr_i32 s21, s20, 31
	s_lshl_b32 s28, s16, 8
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s12, s2, s20
	s_mul_i32 s2, s13, s17
	s_addc_u32 s16, s3, s21
	s_ashr_i32 s3, s2, 31
	v_mov_b32_e32 v60, v0
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshrrev_b32_e32 v34, 4, v60
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s28
	v_or_b32_e32 v0, 0x60, v34
	s_addc_u32 s13, s16, s3
	s_ashr_i32 s3, s2, 31
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	s_load_dword s19, s[0:1], 0x48
	v_or_b32_e32 v11, s28, v0
	s_lshl_b32 s29, s14, 6
	v_mul_lo_u32 v12, s14, v0
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v0, 3, v60
	v_or_b32_e32 v1, 0xa0, v34
	s_add_u32 s20, s12, s2
	v_and_b32_e32 v42, 0x78, v0
	s_mul_i32 s34, s15, s18
	v_or_b32_e32 v19, s28, v1
	v_mul_lo_u32 v20, s14, v1
	s_addc_u32 s13, s13, s3
	v_mad_u64_u32 v[0:1], s[2:3], s14, v34, v[42:43]
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 1
	s_add_u32 s12, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s36, s24, s17
	s_addc_u32 s15, s5, s3
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[2:3], s[36:37], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s22, s26, s18
	s_addc_u32 s2, s15, s3
	s_ashr_i32 s23, s22, 31
	s_lshl_b64 s[22:23], s[22:23], 1
	s_add_u32 s3, s6, s22
	s_mul_i32 s6, s27, s17
	s_addc_u32 s15, s7, s23
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_u32 s24, s3, s6
	s_addc_u32 s16, s15, s7
	s_and_b32 s6, s14, 0x3fff
	v_or_b32_e32 v35, 32, v34
	v_or_b32_e32 v3, s28, v34
	s_movk_i32 s3, 0x4000
	s_bitset1_b32 s6, 14
	v_or_b32_e32 v2, 0xe0, v34
	v_or_b32_e32 v4, s28, v35
	v_mul_lo_u32 v5, s14, v35
	v_add_u32_e32 v1, s29, v0
	s_and_b32 s7, s13, 0xffff
	s_lshl_b32 s6, s6, 16
	v_lshlrev_b32_e32 v0, 1, v0
	v_bfrev_b32_e32 v30, 1
	v_cmp_gt_i32_e32 vcc, s3, v3
	v_or_b32_e32 v10, 64, v3
	v_or_b32_e32 v27, s28, v2
	v_mul_lo_u32 v28, s14, v2
	s_or_b32 s21, s7, s6
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_add_lshl_u32 v2, v5, v42, 1
	v_cmp_gt_i32_e32 vcc, s3, v4
	v_or_b32_e32 v18, 0x80, v3
	v_or_b32_e32 v26, 0xc0, v3
	v_cndmask_b32_e32 v13, v30, v2, vcc
	buffer_load_dwordx4 v[2:5], v0, s[20:23], 0 offen
	buffer_load_dwordx4 v[6:9], v13, s[20:23], 0 offen
	v_lshlrev_b32_e32 v0, 1, v1
	v_cmp_gt_i32_e32 vcc, s3, v10
	v_add_u32_e32 v29, s29, v1
	v_add_lshl_u32 v1, v12, v42, 1
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s3, v11
	v_lshrrev_b32_e32 v38, 1, v60
	v_and_b32_e32 v36, 56, v38
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[10:13], v0, s[20:23], 0 offen
	buffer_load_dwordx4 v[14:17], v1, s[20:23], 0 offen
	v_lshlrev_b32_e32 v0, 1, v29
	v_cmp_gt_i32_e32 vcc, s3, v18
	v_add_lshl_u32 v1, v20, v42, 1
	v_xor_b32_e32 v36, v36, v42
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s3, v19
	s_and_b32 s6, s2, 0xffff
	v_mul_lo_u32 v35, s25, v35
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[18:21], v0, s[20:23], 0 offen
	buffer_load_dwordx4 v[22:25], v1, s[20:23], 0 offen
	v_add_lshl_u32 v0, v29, s29, 1
	v_cmp_gt_i32_e32 vcc, s3, v26
	v_add_lshl_u32 v1, v28, v42, 1
	s_mov_b32 s14, s22
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s3, v27
	s_and_b32 s3, s25, 0x3fff
	s_bitset1_b32 s3, 14
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[26:29], v0, s[20:23], 0 offen
	buffer_load_dwordx4 v[30:33], v1, s[20:23], 0 offen
	v_and_b32_e32 v0, 0x80, v60
	v_lshrrev_b32_e32 v37, 1, v0
	v_xor_b32_e32 v36, v36, v37
	v_mul_lo_u32 v1, s25, v34
	v_lshl_add_u32 v36, v36, 1, 0
	v_lshlrev_b32_e32 v34, 8, v34
	s_lshl_b32 s29, s3, 16
	v_add_u32_e32 v198, v36, v34
	s_or_b32 s13, s6, s29
	s_mov_b32 s15, s23
	v_add_lshl_u32 v61, v1, v42, 1
	s_barrier
	scratch_store_dword off, v37, off offset:140 ; 4-byte Folded Spill
	s_waitcnt vmcnt(8)
	ds_write_b128 v198, v[2:5]
	s_waitcnt vmcnt(7)
	ds_write_b128 v198, v[6:9] offset:8192
	s_waitcnt vmcnt(6)
	ds_write_b128 v198, v[10:13] offset:16384
	s_waitcnt vmcnt(5)
	ds_write_b128 v198, v[14:17] offset:24576
	s_waitcnt vmcnt(4)
	ds_write_b128 v198, v[18:21] offset:32768
	s_waitcnt vmcnt(3)
	ds_write_b128 v198, v[22:25] offset:40960
	s_waitcnt vmcnt(2)
	ds_write_b128 v198, v[26:29] offset:49152
	s_waitcnt vmcnt(1)
	ds_write_b128 v198, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_lshl_u32 v62, v35, v42, 1
	buffer_load_dwordx4 v[2:5], v61, s[12:15], 0 offen
	buffer_load_dwordx4 v[6:9], v62, s[12:15], 0 offen
	v_and_b32_e32 v19, 31, v60
	s_movk_i32 s3, 0xe0
	v_bfe_u32 v1, v60, 5, 1
	v_and_b32_e32 v11, 15, v60
	s_lshl_b32 s38, s25, 6
	v_and_or_b32 v10, v38, s3, v19
	v_xor_b32_e32 v12, v1, v11
	v_or_b32_e32 v13, 2, v1
	v_or_b32_e32 v14, 4, v1
	v_or_b32_e32 v15, 6, v1
	v_or_b32_e32 v16, 8, v1
	v_or_b32_e32 v17, 10, v1
	v_or_b32_e32 v18, 12, v1
	v_or_b32_e32 v1, 14, v1
	v_xor_b32_e32 v13, v13, v11
	v_xor_b32_e32 v14, v14, v11
	v_xor_b32_e32 v15, v15, v11
	v_xor_b32_e32 v16, v16, v11
	v_xor_b32_e32 v17, v17, v11
	v_xor_b32_e32 v18, v18, v11
	v_xor_b32_e32 v1, v1, v11
	v_lshl_add_u32 v10, v10, 8, 0
	v_lshlrev_b32_e32 v11, 4, v12
	s_ashr_i32 s39, s38, 31
	s_lshl_b32 s40, s19, 6
	scratch_store_dword off, v38, off offset:144 ; 4-byte Folded Spill
	v_add_u32_e32 v12, v10, v11
	v_lshlrev_b32_e32 v20, 4, v13
	v_lshlrev_b32_e32 v43, 4, v14
	s_lshl_b64 s[30:31], s[38:39], 1
	v_add_u32_e32 v13, v10, v20
	ds_read_b128 v[158:161], v12
	ds_read_b128 v[154:157], v13
	v_add_u32_e32 v12, v10, v43
	v_lshlrev_b32_e32 v52, 4, v15
	v_lshlrev_b32_e32 v53, 4, v16
	s_add_u32 s20, s12, s30
	v_add_u32_e32 v13, v10, v52
	ds_read_b128 v[150:153], v12
	ds_read_b128 v[146:149], v13
	v_add_u32_e32 v12, v10, v53
	v_lshlrev_b32_e32 v54, 4, v17
	v_lshlrev_b32_e32 v55, 4, v18
	s_addc_u32 s41, s2, s31
	v_add_u32_e32 v13, v10, v54
	ds_read_b128 v[142:145], v12
	ds_read_b128 v[138:141], v13
	v_add_u32_e32 v12, v10, v55
	v_lshlrev_b32_e32 v1, 4, v1
	s_and_b32 s2, s41, 0xffff
	v_add_u32_e32 v10, v10, v1
	ds_read_b128 v[134:137], v12
	ds_read_b128 v[130:133], v10
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(2)
	ds_write_b128 v198, v[2:5]
	s_waitcnt vmcnt(1)
	ds_write_b128 v198, v[6:9] offset:8192
	s_or_b32 s21, s2, s29
	v_lshlrev_b32_e32 v56, 8, v19
	buffer_load_dwordx4 v[34:37], v61, s[20:23], 0 offen
	buffer_load_dwordx4 v[38:41], v62, s[20:23], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v19, off offset:148 ; 4-byte Folded Spill
	v_add3_u32 v197, 0, v11, v56
	ds_read_b128 v[16:19], v197
	ds_read_b128 v[44:47], v197 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[16:17], v[158:159], 0
	v_add3_u32 v203, 0, v20, v56
	v_add3_u32 v206, 0, v43, v56
	v_add3_u32 v207, 0, v52, v56
	v_add3_u32 v249, 0, v53, v56
	v_add3_u32 v250, 0, v54, v56
	v_add3_u32 v251, 0, v55, v56
	v_add3_u32 v252, 0, v1, v56
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[160:161], v[2:17]
	ds_read_b128 v[18:21], v203
	ds_read_b128 v[48:51], v203 offset:8192
	v_and_b32_e32 v1, 0x100, v60
	v_or_b32_e32 v0, v0, v1
	v_and_b32_e32 v43, 16, v60
	v_lshrrev_b32_e32 v0, 3, v0
	v_lshrrev_b32_e32 v58, 3, v43
	s_mov_b32 s26, s22
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[154:155], v[2:17]
	s_mov_b32 s27, s23
	s_mov_b32 s39, 0x5040100
	s_mov_b32 s42, 0x7060302
	v_mov_b32_e32 v202, v61
	v_mov_b32_e32 v246, v62
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[156:157], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[44:45], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[46:47], v[160:161], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[48:49], v[154:155], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[50:51], v[156:157], v[18:33]
	ds_read_b128 v[44:47], v206
	ds_read_b128 v[48:51], v206 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[44:45], v[150:151], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[48:49], v[150:151], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[46:47], v[152:153], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[50:51], v[152:153], v[18:33]
	ds_read_b128 v[44:47], v207
	ds_read_b128 v[48:51], v207 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[44:45], v[146:147], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[48:49], v[146:147], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[46:47], v[148:149], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[50:51], v[148:149], v[18:33]
	ds_read_b128 v[44:47], v249
	ds_read_b128 v[48:51], v249 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[44:45], v[142:143], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[48:49], v[142:143], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[46:47], v[144:145], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[50:51], v[144:145], v[18:33]
	ds_read_b128 v[44:47], v250
	ds_read_b128 v[48:51], v250 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[44:45], v[138:139], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[48:49], v[138:139], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[46:47], v[140:141], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[50:51], v[140:141], v[18:33]
	ds_read_b128 v[44:47], v251
	ds_read_b128 v[48:51], v251 offset:8192
	scratch_store_dword off, v1, off offset:152 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[44:45], v[134:135], v[2:17]
	v_lshrrev_b32_e32 v44, 3, v60
	v_and_or_b32 v59, v44, 12, v0
	v_or_b32_e32 v0, v59, v58
	v_mad_u64_u32 v[0:1], s[2:3], s19, v0, v[42:43]
	s_and_b32 s2, s19, 0x3fff
	s_bitset1_b32 s2, 14
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[48:49], v[134:135], v[18:33]
	s_and_b32 s3, s16, 0xffff
	s_lshl_b32 s33, s2, 16
	v_lshlrev_b32_e32 v74, 1, v0
	s_or_b32 s25, s3, s33
	v_add_lshl_u32 v75, v0, s19, 1
	s_add_u32 s20, s20, s30
	s_addc_u32 s19, s41, s31
	v_mfma_f32_32x32x8_f16 v[2:17], v[46:47], v[136:137], v[2:17]
	ds_read_b128 v[46:49], v252
	s_ashr_i32 s41, s40, 31
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_lshlrev_b32_e32 v0, 2, v60
	v_xor_b32_e32 v42, 0x80, v0
	v_mfma_f32_32x32x8_f16 v[18:33], v[50:51], v[136:137], v[18:33]
	ds_read_b128 v[50:53], v252 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[46:47], v[130:131], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[48:49], v[132:133], v[2:17]
	buffer_load_dwordx4 v[46:49], v74, s[24:27], 0 offen
	buffer_load_dwordx4 v[54:57], v75, s[24:27], 0 offen
	s_lshl_b64 s[26:27], s[40:41], 1
	s_add_u32 s40, s24, s26
	s_addc_u32 s41, s16, s27
	s_and_b32 s16, s19, 0xffff
	scratch_store_dword off, v42, off offset:116 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(6)
	ds_write_b128 v198, v[34:37]
	s_waitcnt vmcnt(5)
	ds_write_b128 v198, v[38:41] offset:8192
	s_or_b32 s21, s16, s29
	s_and_b32 s16, s41, 0xffff
	buffer_load_dwordx4 v[110:113], v61, s[20:23], 0 offen
	buffer_load_dwordx4 v[106:109], v62, s[20:23], 0 offen
                                        ; kill: killed $sgpr20_sgpr21
	s_or_b32 s21, s16, s33
	s_mov_b32 s20, s40
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[98:101], v74, s[20:23], 0 offen
	buffer_load_dwordx4 v[102:105], v75, s[20:23], 0 offen
	v_mfma_f32_32x32x8_f16 v[18:33], v[50:51], v[130:131], v[18:33]
	v_max_f32_e32 v0, v3, v3
	v_max_f32_e32 v1, v2, v2
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v4, v5
	v_max3_f32 v0, v0, v6, v7
	v_max3_f32 v0, v0, v8, v9
	v_max3_f32 v0, v0, v10, v11
	v_mfma_f32_32x32x8_f16 v[18:33], v[52:53], v[132:133], v[18:33]
	v_max3_f32 v0, v0, v12, v13
	v_max3_f32 v0, v0, v14, v15
	v_max3_f32 v0, v0, v16, v17
	v_and_b32_e32 v34, 1, v60
	v_cmp_eq_u32_e64 s[12:13], 0, v34
	v_and_b32_e32 v34, 2, v60
	v_cmp_eq_u32_e64 s[14:15], 0, v34
	s_nop 3
	v_max3_f32 v0, v0, v18, v19
	v_max3_f32 v0, v0, v20, v21
	v_max3_f32 v0, v0, v22, v23
	v_max3_f32 v0, v0, v24, v25
	v_max3_f32 v0, v0, v26, v27
	v_max3_f32 v0, v0, v28, v29
	v_max3_f32 v0, v0, v30, v31
	v_max3_f32 v0, v0, v32, v33
	v_and_b32_e32 v34, 4, v60
	ds_bpermute_b32 v1, v42, v0
	v_cmp_eq_u32_e64 s[6:7], 0, v34
	v_and_b32_e32 v34, 8, v60
	v_cmp_eq_u32_e64 s[2:3], 0, v34
	v_bfe_i32 v34, v60, 0, 1
	v_bfe_i32 v35, v60, 1, 1
	v_and_b32_e32 v34, 0x220, v34
	v_and_b32_e32 v35, 0x404, v35
	v_bfe_i32 v36, v60, 2, 1
	v_and_b32_e32 v36, 0x808, v36
	v_bfe_i32 v37, v60, 3, 1
	v_or_b32_e32 v38, v34, v35
	v_mov_b32_e32 v42, 0xff800000
	v_and_b32_e32 v37, 0x1010, v37
	v_or_b32_e32 v39, v38, v36
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v195, v0, v1, v42
	v_or3_b32 v40, v37, v58, v39
	v_mul_f32_e32 v45, 0xbe0293ee, v195
	v_xor_b32_e32 v40, v59, v40
	v_fmamk_f32 v1, v3, 0x3e0293ee, v45
	v_fmamk_f32 v3, v5, 0x3e0293ee, v45
	v_fmamk_f32 v5, v7, 0x3e0293ee, v45
	v_fmamk_f32 v7, v9, 0x3e0293ee, v45
	v_fmamk_f32 v9, v11, 0x3e0293ee, v45
	v_fmamk_f32 v11, v13, 0x3e0293ee, v45
	v_fmamk_f32 v13, v15, 0x3e0293ee, v45
	v_fmamk_f32 v15, v17, 0x3e0293ee, v45
	v_fmamk_f32 v17, v19, 0x3e0293ee, v45
	v_fmamk_f32 v19, v21, 0x3e0293ee, v45
	v_fmamk_f32 v21, v23, 0x3e0293ee, v45
	v_fmamk_f32 v23, v25, 0x3e0293ee, v45
	v_fmamk_f32 v25, v27, 0x3e0293ee, v45
	v_fmamk_f32 v27, v29, 0x3e0293ee, v45
	v_fmamk_f32 v29, v31, 0x3e0293ee, v45
	v_lshl_add_u32 v67, v40, 1, 0
	s_waitcnt vmcnt(5)
	v_perm_b32 v31, v54, v46, s39
	ds_write_b32 v67, v31 offset:16384
	v_or_b32_e32 v31, 0x44, v34
	v_xor_b32_e32 v31, v31, v35
	v_or_b32_e32 v40, v37, v36
	v_or3_b32 v31, v58, v31, v40
	v_xor_b32_e32 v31, v59, v31
	v_lshl_add_u32 v68, v31, 1, 0
	v_or_b32_e32 v31, 0x88, v38
	v_xor_b32_e32 v31, v31, v36
	v_or3_b32 v31, v58, v31, v37
	v_fmamk_f32 v0, v2, 0x3e0293ee, v45
	v_fmamk_f32 v2, v4, 0x3e0293ee, v45
	v_fmamk_f32 v4, v6, 0x3e0293ee, v45
	v_fmamk_f32 v6, v8, 0x3e0293ee, v45
	v_fmamk_f32 v8, v10, 0x3e0293ee, v45
	v_fmamk_f32 v10, v12, 0x3e0293ee, v45
	v_fmamk_f32 v12, v14, 0x3e0293ee, v45
	v_fmamk_f32 v14, v16, 0x3e0293ee, v45
	v_fmamk_f32 v16, v18, 0x3e0293ee, v45
	v_fmamk_f32 v18, v20, 0x3e0293ee, v45
	v_fmamk_f32 v20, v22, 0x3e0293ee, v45
	v_fmamk_f32 v22, v24, 0x3e0293ee, v45
	v_fmamk_f32 v24, v26, 0x3e0293ee, v45
	v_fmamk_f32 v26, v28, 0x3e0293ee, v45
	v_fmamk_f32 v28, v30, 0x3e0293ee, v45
	v_fmamk_f32 v30, v32, 0x3e0293ee, v45
	v_perm_b32 v32, v54, v46, s42
	v_xor_b32_e32 v31, v59, v31
	ds_read_b128 v[190:193], v197
	ds_read_b128 v[174:177], v197 offset:8192
	ds_read_b128 v[186:189], v203
	ds_read_b128 v[170:173], v203 offset:8192
	ds_read_b128 v[182:185], v206
	ds_read_b128 v[166:169], v206 offset:8192
	ds_read_b128 v[178:181], v207
	ds_read_b128 v[162:165], v207 offset:8192
	ds_read_b128 v[94:97], v249
	ds_read_b128 v[126:129], v249 offset:8192
	ds_read_b128 v[90:93], v250
	ds_read_b128 v[122:125], v250 offset:8192
	ds_read_b128 v[86:89], v251
	ds_read_b128 v[118:121], v251 offset:8192
	ds_read_b128 v[82:85], v252
	ds_read_b128 v[114:117], v252 offset:8192
	ds_write_b32 v68, v32 offset:16384
	v_lshl_add_u32 v69, v31, 1, 0
	v_or_b32_e32 v31, 0xcc, v34
	v_or_b32_e32 v32, v36, v35
	v_xor_b32_e32 v31, v32, v31
	v_or3_b32 v31, v58, v31, v37
	v_xor_b32_e32 v31, v59, v31
	v_lshl_add_u32 v70, v31, 1, 0
	v_or_b32_e32 v31, 0x110, v39
	v_xor_b32_e32 v31, v31, v37
	v_or_b32_e32 v31, v31, v58
	v_xor_b32_e32 v31, v59, v31
	v_lshl_add_u32 v71, v31, 1, 0
	v_or_b32_e32 v31, 0x154, v34
	v_xor_b32_e32 v31, v31, v35
	v_or_b32_e32 v31, v31, v36
	v_xor_b32_e32 v31, v31, v37
	v_or_b32_e32 v31, v31, v58
	v_xor_b32_e32 v31, v59, v31
	v_lshl_add_u32 v72, v31, 1, 0
	v_or_b32_e32 v31, 0x198, v38
	v_xor_b32_e32 v31, v40, v31
	v_or_b32_e32 v31, v31, v58
	v_xor_b32_e32 v31, v59, v31
	v_lshl_add_u32 v73, v31, 1, 0
	v_or_b32_e32 v31, v32, v37
	v_or_b32_e32 v32, 0x1dc, v34
	v_xor_b32_e32 v31, v31, v32
	v_or_b32_e32 v31, v31, v58
	s_movk_i32 s19, 0x100
	v_xor_b32_e32 v31, v59, v31
	v_cmp_gt_u32_e32 vcc, s19, v60
	s_movk_i32 s19, 0xff
	v_fmac_f32_e32 v45, 0x3e0293ee, v33
	v_perm_b32 v33, v55, v47, s39
	v_perm_b32 v46, v55, v47, s42
	v_perm_b32 v47, v56, v48, s39
	v_perm_b32 v48, v56, v48, s42
	v_perm_b32 v50, v57, v49, s39
	v_perm_b32 v49, v57, v49, s42
	v_lshl_add_u32 v66, v31, 1, 0
	s_mov_b32 s16, 0x3e0293ee
	v_fmac_f32_e32 v42, 0xbe0293ee, v195
	v_cmp_lt_u32_e64 s[20:21], s19, v60
	ds_write_b32 v69, v33 offset:16384
	ds_write_b32 v70, v46 offset:16384
	ds_write_b32 v71, v47 offset:16384
	ds_write_b32 v72, v48 offset:16384
	ds_write_b32 v73, v50 offset:16384
	ds_write_b32 v66, v49 offset:16384
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v60, off offset:128 ; 4-byte Folded Spill
	s_and_saveexec_b64 s[24:25], s[20:21]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[24:25]
	s_load_dwordx2 s[24:25], s[0:1], 0x4c
	s_load_dword s19, s[0:1], 0x54
	scratch_load_dword v255, off, off offset:116 ; 4-byte Folded Reload
	v_exp_f32_e32 v209, v1
	v_exp_f32_e32 v1, v2
	v_exp_f32_e32 v210, v3
	v_exp_f32_e32 v212, v5
	v_mov_b32_e32 v2, 0x44
	v_mov_b32_e32 v3, 0x88
	v_mov_b32_e32 v5, 0x110
	v_cndmask_b32_e64 v2, v2, 0, s[12:13]
	v_cndmask_b32_e64 v3, v3, 0, s[14:15]
	v_cndmask_b32_e64 v5, v5, 0, s[6:7]
	v_exp_f32_e32 v201, v0
	v_exp_f32_e32 v0, v4
	v_exp_f32_e32 v214, v7
	v_exp_f32_e32 v216, v9
	v_exp_f32_e32 v219, v14
	v_exp_f32_e32 v222, v15
	v_exp_f32_e32 v229, v24
	v_or_b32_e32 v4, v2, v3
	v_mov_b32_e32 v7, 0x220
	v_mov_b32_e32 v9, 0x404
	v_cmp_eq_u32_e64 s[0:1], 0, v43
	v_or_b32_e32 v14, 24, v2
	v_or_b32_e32 v15, v5, v3
	v_or_b32_e32 v24, 0x818, v2
	v_or_b32_e32 v31, 0x1018, v2
	v_or_b32_e32 v38, 0x1818, v2
	v_exp_f32_e32 v211, v6
	v_exp_f32_e32 v217, v12
	v_exp_f32_e32 v220, v13
	v_exp_f32_e32 v227, v22
	v_exp_f32_e32 v232, v25
	v_exp_f32_e32 v235, v30
	v_and_b32_e32 v40, 4, v44
	v_or_b32_e32 v6, v4, v5
	v_cndmask_b32_e64 v7, v7, 0, s[2:3]
	v_cndmask_b32_e64 v9, v9, 0, s[0:1]
	v_or_b32_e32 v12, 8, v2
	v_or_b32_e32 v13, 16, v4
	v_xor_b32_e32 v14, v15, v14
	v_or_b32_e32 v22, 0x808, v2
	v_xor_b32_e32 v24, v15, v24
	v_or_b32_e32 v25, 0x810, v4
	v_or_b32_e32 v30, 0x1010, v4
	v_xor_b32_e32 v31, v15, v31
	v_or_b32_e32 v33, 0x1008, v2
	v_xor_b32_e32 v15, v15, v38
	v_or_b32_e32 v38, 0x1810, v4
	v_or_b32_e32 v39, 0x1808, v2
	v_exp_f32_e32 v213, v8
	v_exp_f32_e32 v215, v10
	v_or_b32_e32 v8, v6, v7
	v_xor_b32_e32 v10, v9, v40
	v_xor_b32_e32 v12, v12, v3
	v_xor_b32_e32 v13, v13, v5
	v_xor_b32_e32 v22, v22, v3
	v_xor_b32_e32 v25, v25, v5
	v_xor_b32_e32 v30, v30, v5
	v_xor_b32_e32 v33, v33, v3
	v_xor_b32_e32 v38, v38, v5
	v_xor_b32_e32 v39, v39, v3
	v_exp_f32_e32 v218, v11
	v_exp_f32_e32 v230, v23
	v_xor_b32_e32 v11, v10, v8
	v_or3_b32 v12, v5, v12, v7
	v_or_b32_e32 v13, v13, v7
	v_or_b32_e32 v14, v14, v7
	v_or_b32_e32 v10, v10, v7
	v_or3_b32 v22, v5, v22, v7
	v_or_b32_e32 v23, 0x800, v8
	v_or_b32_e32 v24, v24, v7
	v_or_b32_e32 v25, v25, v7
	v_or_b32_e32 v30, v30, v7
	v_or_b32_e32 v31, v31, v7
	v_or_b32_e32 v32, 0x1000, v8
	v_or3_b32 v33, v5, v33, v7
	v_or_b32_e32 v15, v15, v7
	v_or_b32_e32 v38, v38, v7
	v_or3_b32 v7, v5, v39, v7
	v_or_b32_e32 v8, 0x1800, v8
	v_xor_b32_e32 v12, v40, v12
	v_xor_b32_e32 v13, v40, v13
	v_xor_b32_e32 v14, v40, v14
	v_xor_b32_e32 v22, v40, v22
	v_xor_b32_e32 v23, v40, v23
	v_xor_b32_e32 v24, v40, v24
	v_xor_b32_e32 v25, v40, v25
	v_xor_b32_e32 v30, v40, v30
	v_xor_b32_e32 v31, v40, v31
	v_xor_b32_e32 v32, v40, v32
	v_xor_b32_e32 v33, v40, v33
	v_xor_b32_e32 v15, v40, v15
	v_xor_b32_e32 v38, v40, v38
	v_xor_b32_e32 v7, v40, v7
	v_xor_b32_e32 v8, v40, v8
	v_exp_f32_e32 v224, v17
	v_exp_f32_e32 v225, v20
	v_exp_f32_e32 v231, v26
	v_exp_f32_e32 v233, v28
	v_xor_b32_e32 v12, v12, v9
	v_xor_b32_e32 v13, v13, v9
	v_xor_b32_e32 v14, v14, v9
	v_or_b32_e32 v17, 40, v2
	v_or_b32_e32 v20, 56, v2
	v_xor_b32_e32 v22, v22, v9
	v_xor_b32_e32 v23, v23, v9
	v_xor_b32_e32 v24, v24, v9
	v_xor_b32_e32 v25, v25, v9
	v_or_b32_e32 v26, 0x828, v2
	v_or_b32_e32 v28, 0x838, v2
	v_xor_b32_e32 v30, v30, v9
	v_xor_b32_e32 v31, v31, v9
	v_xor_b32_e32 v32, v32, v9
	v_xor_b32_e32 v33, v33, v9
	v_or_b32_e32 v35, 0x1038, v2
	v_or_b32_e32 v37, 0x1028, v2
	v_xor_b32_e32 v15, v15, v9
	v_xor_b32_e32 v38, v38, v9
	v_xor_b32_e32 v7, v7, v9
	v_xor_b32_e32 v8, v8, v9
	v_or_b32_e32 v9, 0x1838, v2
	v_or_b32_e32 v2, 0x1828, v2
	v_xor_b32_e32 v17, v17, v3
	v_xor_b32_e32 v26, v26, v3
	v_xor_b32_e32 v37, v37, v3
	v_xor_b32_e32 v2, v2, v3
	v_exp_f32_e32 v226, v19
	v_or_b32_e32 v17, v17, v5
	v_or_b32_e32 v19, v10, v5
	v_or_b32_e32 v26, v26, v5
	v_or_b32_e32 v37, v37, v5
	v_or_b32_e32 v2, v2, v5
	v_lshl_add_u32 v5, v11, 1, 0
	scratch_store_dword off, v5, off offset:52 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v12, 1, 0
	v_exp_f32_e32 v221, v16
	v_or_b32_e32 v16, 32, v6
	scratch_store_dword off, v5, off offset:56 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v13, 1, 0
	v_xor_b32_e32 v16, v10, v16
	scratch_store_dword off, v5, off offset:60 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v14, 1, 0
	v_exp_f32_e32 v223, v18
	v_xor_b32_e32 v17, v10, v17
	v_or_b32_e32 v18, 48, v4
	scratch_store_dword off, v5, off offset:64 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v16, 1, 0
	v_exp_f32_e32 v228, v21
	v_xor_b32_e32 v18, v19, v18
	v_or_b32_e32 v21, v19, v3
	scratch_store_dword off, v5, off offset:68 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v17, 1, 0
	v_xor_b32_e32 v20, v21, v20
	scratch_store_dword off, v5, off offset:72 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v18, 1, 0
	scratch_store_dword off, v5, off offset:76 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v20, 1, 0
	scratch_store_dword off, v5, off offset:80 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v22, 1, 0
	scratch_store_dword off, v5, off offset:84 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v23, 1, 0
	scratch_store_dword off, v5, off offset:48 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v24, 1, 0
	v_exp_f32_e32 v234, v27
	v_xor_b32_e32 v26, v10, v26
	v_or_b32_e32 v27, 0x820, v6
	scratch_store_dword off, v5, off offset:44 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v25, 1, 0
	v_xor_b32_e32 v27, v10, v27
	scratch_store_dword off, v5, off        ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v26, 1, 0
	v_exp_f32_e32 v236, v29
	v_xor_b32_e32 v28, v21, v28
	v_or_b32_e32 v29, 0x830, v4
	scratch_store_dword off, v5, off offset:88 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v27, 1, 0
	v_xor_b32_e32 v29, v19, v29
	scratch_store_dword off, v5, off offset:92 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v28, 1, 0
	scratch_store_dword off, v5, off offset:96 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v29, 1, 0
	scratch_store_dword off, v5, off offset:4 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v30, 1, 0
	scratch_store_dword off, v5, off offset:8 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v31, 1, 0
	v_or_b32_e32 v34, 0x1030, v4
	scratch_store_dword off, v5, off offset:12 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v32, 1, 0
	v_xor_b32_e32 v34, v19, v34
	scratch_store_dword off, v5, off offset:16 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v33, 1, 0
	v_xor_b32_e32 v35, v21, v35
	v_or_b32_e32 v36, 0x1020, v6
	scratch_store_dword off, v5, off offset:20 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v34, 1, 0
	v_xor_b32_e32 v36, v10, v36
	scratch_store_dword off, v5, off offset:100 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v35, 1, 0
	v_xor_b32_e32 v37, v10, v37
	scratch_store_dword off, v5, off offset:104 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v36, 1, 0
	scratch_store_dword off, v5, off offset:24 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v37, 1, 0
	s_add_u32 s0, s34, s36
	scratch_store_dword off, v5, off offset:28 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v15, 1, 0
	s_addc_u32 s1, s35, s37
	scratch_store_dword off, v5, off offset:32 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v38, 1, 0
	s_mul_i32 s3, s38, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_exp_f32_e32 v237, v45
	v_exp_f32_e32 v196, v42
	scratch_store_dword off, v5, off offset:108 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v7, 1, 0
	s_mul_hi_i32 s2, s38, 6
	s_add_u32 s0, s3, s0
	v_xor_b32_e32 v9, v21, v9
	v_or_b32_e32 v4, 0x1830, v4
	v_xor_b32_e32 v2, v10, v2
	v_or_b32_e32 v3, 0x1820, v6
	scratch_store_dword off, v5, off offset:112 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v8, 1, 0
	s_addc_u32 s1, s2, s1
	v_xor_b32_e32 v4, v19, v4
	v_xor_b32_e32 v3, v10, v3
	scratch_store_dword off, v5, off offset:36 ; 4-byte Folded Spill
	v_lshl_add_u32 v5, v9, 1, 0
	v_lshl_add_u32 v247, v2, 1, 0
	s_add_u32 s0, s4, s0
	v_mov_b32_e32 v2, 0
	s_waitcnt vmcnt(30)
	v_lshrrev_b32_e32 v199, 16, v102
	scratch_store_dword off, v40, off offset:132 ; 4-byte Folded Spill
	scratch_store_dword off, v5, off offset:40 ; 4-byte Folded Spill
	v_lshl_add_u32 v208, v4, 1, 0
	v_lshl_add_u32 v248, v3, 1, 0
	s_addc_u32 s1, s5, s1
	v_mov_b32_e32 v200, 1.0
	s_movk_i32 s2, 0xffc0
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
	v_mov_b32_e32 v238, v67
	v_mov_b32_e32 v239, v68
	v_mov_b32_e32 v240, v69
	v_mov_b32_e32 v241, v70
	v_mov_b32_e32 v242, v71
	v_mov_b32_e32 v243, v72
	v_mov_b32_e32 v244, v73
	v_mov_b32_e32 v245, v66
	v_mov_b32_e32 v253, v74
	v_mov_b32_e32 v254, v75
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[158:159], 0
	v_mov_b32_e32 v204, v200
	v_mov_b32_e32 v194, v195
	s_setprio 0
	v_pk_mul_f32 v[50:51], v[50:51], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[196:197] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[192:193], v[160:161], v[66:81]
	v_pk_mul_f32 v[4:5], v[4:5], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[196:197] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[154:155], v[66:81]
	v_pk_mul_f32 v[24:25], v[24:25], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[196:197] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[156:157], v[66:81]
	v_pk_mul_f32 v[44:45], v[44:45], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[196:197] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[150:151], v[66:81]
	v_pk_mul_f32 v[64:65], v[64:65], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[196:197] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[184:185], v[152:153], v[66:81]
	v_pk_mul_f32 v[52:53], v[52:53], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[196:197] op_sel_hi:[1,0]
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[180:181], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[96:97], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[86:87], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[88:89], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[82:83], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[84:85], v[132:133], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[160:161], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[154:155], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[156:157], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[150:151], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[152:153], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[146:147], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[126:127], v[142:143], v[82:97]
	v_cvt_pkrtz_f16_f32 v126, v229, v232
	v_cvt_pkrtz_f16_f32 v127, v231, v234
	v_mfma_f32_32x32x8_f16 v[82:97], v[128:129], v[144:145], v[82:97]
	v_cvt_pkrtz_f16_f32 v128, v233, v236
	v_cvt_pkrtz_f16_f32 v129, v235, v237
	v_mfma_f32_32x32x8_f16 v[82:97], v[122:123], v[138:139], v[82:97]
	v_cvt_pkrtz_f16_f32 v122, v221, v224
	v_cvt_pkrtz_f16_f32 v123, v223, v226
	v_mfma_f32_32x32x8_f16 v[82:97], v[124:125], v[140:141], v[82:97]
	v_cvt_pkrtz_f16_f32 v124, v225, v228
	v_cvt_pkrtz_f16_f32 v125, v227, v230
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[134:135], v[82:97]
	v_cvt_pkrtz_f16_f32 v118, v213, v216
	v_cvt_pkrtz_f16_f32 v119, v215, v218
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[136:137], v[82:97]
	v_cvt_pkrtz_f16_f32 v120, v217, v220
	v_cvt_pkrtz_f16_f32 v121, v219, v222
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[130:131], v[82:97]
	v_add_f32_e32 v114, v201, v209
	v_add_f32_e32 v114, v114, v1
	v_add_f32_e32 v114, v114, v210
	v_add_f32_e32 v114, v114, v0
	v_add_f32_e32 v114, v114, v212
	v_add_f32_e32 v114, v114, v211
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[132:133], v[82:97]
	v_add_f32_e32 v114, v114, v214
	v_cvt_pkrtz_f16_f32 v117, v211, v214
	v_add_f32_e32 v114, v114, v213
	v_cvt_pkrtz_f16_f32 v116, v0, v212
	v_add_f32_e32 v114, v114, v216
	v_add_f32_e32 v114, v114, v215
	v_add_f32_e32 v114, v114, v218
	v_add_f32_e32 v114, v114, v217
	v_add_f32_e32 v114, v114, v220
	v_add_f32_e32 v114, v114, v219
	v_add_f32_e32 v114, v114, v222
	v_add_f32_e32 v114, v114, v221
	v_add_f32_e32 v114, v114, v224
	v_add_f32_e32 v114, v114, v223
	v_add_f32_e32 v114, v114, v226
	v_add_f32_e32 v114, v114, v225
	v_add_f32_e32 v114, v114, v228
	v_add_f32_e32 v114, v114, v227
	v_add_f32_e32 v114, v114, v230
	v_add_f32_e32 v114, v114, v229
	v_add_f32_e32 v114, v114, v232
	v_add_f32_e32 v114, v114, v231
	v_add_f32_e32 v114, v114, v234
	v_add_f32_e32 v114, v114, v233
	v_add_f32_e32 v114, v114, v236
	v_add_f32_e32 v114, v114, v235
	v_add_f32_e32 v114, v114, v237
	s_waitcnt vmcnt(30)
	ds_bpermute_b32 v115, v255, v114
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v200, v114, v115
	v_fmac_f32_e32 v200, v204, v196
	v_cvt_pkrtz_f16_f32 v114, v201, v209
	v_cvt_pkrtz_f16_f32 v115, v1, v210
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_barrier
	s_barrier
	scratch_load_dword v195, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v162, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v164, off, off offset:60 ; 4-byte Folded Reload
	scratch_load_dword v176, off, off offset:84 ; 4-byte Folded Reload
	scratch_load_dword v178, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v182, off, off       ; 4-byte Folded Reload
	scratch_load_dword v192, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v166, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v168, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v170, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v172, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v174, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v180, off, off offset:44 ; 4-byte Folded Reload
	scratch_load_dword v184, off, off offset:88 ; 4-byte Folded Reload
	scratch_load_dword v186, off, off offset:92 ; 4-byte Folded Reload
	scratch_load_dword v188, off, off offset:96 ; 4-byte Folded Reload
	scratch_load_dword v190, off, off offset:4 ; 4-byte Folded Reload
	s_add_u32 s12, s40, s26
	s_addc_u32 s3, s41, s27
	s_and_b32 s4, s1, 0xffff
	s_or_b32 s21, s4, s29
	s_mov_b32 s20, s0
	ds_read_b64 v[232:233], v208 offset:16384
	ds_read_b64 v[234:235], v247 offset:16384
	ds_read_b64 v[236:237], v248 offset:16384
	s_waitcnt vmcnt(17)
	ds_read_b64 v[204:205], v195 offset:16384
	scratch_load_dword v195, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(17)
	ds_read_b64 v[162:163], v162 offset:16384
	s_waitcnt vmcnt(16)
	ds_read_b64 v[164:165], v164 offset:16384
	s_waitcnt vmcnt(15)
	ds_read_b64 v[176:177], v176 offset:16384
	s_waitcnt vmcnt(14)
	ds_read_b64 v[178:179], v178 offset:16384
	s_waitcnt vmcnt(13)
	ds_read_b64 v[182:183], v182 offset:16384
	s_waitcnt vmcnt(12)
	ds_read_b64 v[192:193], v192 offset:16384
	s_waitcnt vmcnt(11)
	ds_read_b64 v[0:1], v0 offset:16384
	s_waitcnt vmcnt(10)
	ds_read_b64 v[166:167], v166 offset:16384
	s_waitcnt vmcnt(9)
	ds_read_b64 v[168:169], v168 offset:16384
	s_waitcnt vmcnt(8)
	ds_read_b64 v[170:171], v170 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[172:173], v172 offset:16384
	s_waitcnt vmcnt(6)
	ds_read_b64 v[174:175], v174 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[180:181], v180 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[184:185], v184 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[186:187], v186 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[188:189], v188 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[190:191], v190 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[210:211], v195 offset:16384
	scratch_load_dword v195, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v195 offset:16384
	scratch_load_dword v195, off, off offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v195 offset:16384
	scratch_load_dword v195, off, off offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[216:217], v195 offset:16384
	scratch_load_dword v195, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[218:219], v195 offset:16384
	scratch_load_dword v195, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[220:221], v195 offset:16384
	scratch_load_dword v195, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[222:223], v195 offset:16384
	scratch_load_dword v195, off, off offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[224:225], v195 offset:16384
	scratch_load_dword v195, off, off offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[226:227], v195 offset:16384
	scratch_load_dword v195, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[228:229], v195 offset:16384
	scratch_load_dword v195, off, off offset:40 ; 4-byte Folded Reload
	ds_write_b128 v198, v[110:113]
	ds_write_b128 v198, v[106:109] offset:8192
	buffer_load_dwordx4 v[110:113], v202, s[20:23], 0 offen
	buffer_load_dwordx4 v[106:109], v246, s[20:23], 0 offen
	s_waitcnt vmcnt(2)
	ds_read_b64 v[230:231], v195 offset:16384
	; sched_barrier mask(0x00000000)
	s_barrier
	s_setprio 0
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[50:65], v[0:1], v[114:115], v[50:65]
	v_max_f32_e32 v0, v67, v67
	v_max_f32_e32 v1, v66, v66
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v68, v69
	v_max3_f32 v0, v0, v70, v71
	v_max3_f32 v0, v0, v72, v73
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[2:17], v[222:223], v[114:115], v[2:17]
	v_max3_f32 v0, v0, v74, v75
	v_max3_f32 v0, v0, v76, v77
	v_max3_f32 v0, v0, v78, v79
	v_max3_f32 v0, v0, v80, v81
	v_max3_f32 v0, v0, v82, v83
	v_max3_f32 v0, v0, v84, v85
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[2:17], v[224:225], v[116:117], v[2:17]
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[2:17], v[226:227], v[118:119], v[2:17]
	ds_bpermute_b32 v1, v255, v0
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v195, v194, v0, v1
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[34:49], v[176:177], v[114:115], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[192:193], v[114:115], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[204:205], v[116:117], v[18:33]
	v_pk_mul_f32 v[204:205], v[194:195], s[16:17] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v1, v67, s16, -v205
	v_fma_f32 v0, v66, s16, -v205
	v_fma_f32 v67, v69, s16, -v205
	v_fma_f32 v66, v68, s16, -v205
	v_fma_f32 v69, v71, s16, -v205
	v_mfma_f32_32x32x8_f16 v[18:33], v[210:211], v[118:119], v[18:33]
	v_fma_f32 v71, v73, s16, -v205
	v_exp_f32_e32 v210, v67
	v_fma_f32 v73, v75, s16, -v205
	v_mfma_f32_32x32x8_f16 v[18:33], v[212:213], v[120:121], v[18:33]
	v_fma_f32 v68, v70, s16, -v205
	v_fma_f32 v70, v72, s16, -v205
	v_fma_f32 v72, v74, s16, -v205
	v_exp_f32_e32 v212, v69
	v_mfma_f32_32x32x8_f16 v[18:33], v[214:215], v[122:123], v[18:33]
	v_fma_f32 v74, v76, s16, -v205
	v_exp_f32_e32 v214, v71
	v_fma_f32 v76, v78, s16, -v205
	v_mfma_f32_32x32x8_f16 v[18:33], v[216:217], v[124:125], v[18:33]
	v_exp_f32_e32 v209, v1
	v_exp_f32_e32 v201, v0
	v_mfma_f32_32x32x8_f16 v[2:17], v[228:229], v[120:121], v[2:17]
	v_exp_f32_e32 v0, v68
	v_exp_f32_e32 v1, v66
	v_mfma_f32_32x32x8_f16 v[2:17], v[230:231], v[122:123], v[2:17]
	v_sub_f32_e32 v66, v204, v205
	v_fma_f32 v75, v77, s16, -v205
	v_exp_f32_e32 v211, v70
	v_mfma_f32_32x32x8_f16 v[18:33], v[218:219], v[126:127], v[18:33]
	v_fma_f32 v78, v80, s16, -v205
	v_fma_f32 v77, v79, s16, -v205
	v_fma_f32 v79, v81, s16, -v205
	v_fma_f32 v81, v83, s16, -v205
	v_fma_f32 v83, v85, s16, -v205
	v_fma_f32 v85, v87, s16, -v205
	v_mfma_f32_32x32x8_f16 v[18:33], v[220:221], v[128:129], v[18:33]
	v_fma_f32 v80, v82, s16, -v205
	v_fma_f32 v87, v89, s16, -v205
	v_fma_f32 v82, v84, s16, -v205
	v_fma_f32 v84, v86, s16, -v205
	v_fma_f32 v86, v88, s16, -v205
	v_fma_f32 v88, v90, s16, -v205
	v_mfma_f32_32x32x8_f16 v[2:17], v[232:233], v[124:125], v[2:17]
	v_fma_f32 v90, v92, s16, -v205
	v_fma_f32 v89, v91, s16, -v205
	v_fma_f32 v92, v94, s16, -v205
	v_exp_f32_e32 v213, v72
	v_mfma_f32_32x32x8_f16 v[2:17], v[234:235], v[126:127], v[2:17]
	v_fma_f32 v91, v93, s16, -v205
	v_fma_f32 v94, v96, s16, -v205
	v_fma_f32 v93, v95, s16, -v205
	v_exp_f32_e32 v218, v75
	v_mfma_f32_32x32x8_f16 v[2:17], v[236:237], v[128:129], v[2:17]
	v_fma_f32 v95, v97, s16, -v205
	v_exp_f32_e32 v215, v74
	v_exp_f32_e32 v216, v73
	v_mfma_f32_32x32x8_f16 v[50:65], v[162:163], v[116:117], v[50:65]
	v_exp_f32_e32 v217, v76
	v_exp_f32_e32 v196, v66
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[116:117], v[34:49]
	v_exp_f32_e32 v233, v92
	v_exp_f32_e32 v234, v91
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[118:119], v[34:49]
	v_exp_f32_e32 v231, v90
	v_exp_f32_e32 v232, v89
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[120:121], v[34:49]
	v_exp_f32_e32 v227, v86
	v_exp_f32_e32 v229, v88
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[122:123], v[34:49]
	v_exp_f32_e32 v223, v82
	v_exp_f32_e32 v225, v84
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[124:125], v[34:49]
	v_exp_f32_e32 v221, v80
	v_exp_f32_e32 v230, v87
	v_mfma_f32_32x32x8_f16 v[34:49], v[188:189], v[126:127], v[34:49]
	v_exp_f32_e32 v226, v83
	v_exp_f32_e32 v228, v85
	v_mfma_f32_32x32x8_f16 v[50:65], v[164:165], v[118:119], v[50:65]
	v_exp_f32_e32 v222, v79
	v_exp_f32_e32 v224, v81
	v_mfma_f32_32x32x8_f16 v[50:65], v[166:167], v[120:121], v[50:65]
	v_exp_f32_e32 v219, v78
	v_exp_f32_e32 v220, v77
	v_mfma_f32_32x32x8_f16 v[50:65], v[168:169], v[122:123], v[50:65]
	v_exp_f32_e32 v235, v94
	v_exp_f32_e32 v236, v93
	v_mfma_f32_32x32x8_f16 v[50:65], v[170:171], v[124:125], v[50:65]
	v_exp_f32_e32 v237, v95
	v_mov_b32_e32 v73, v244
	v_mov_b32_e32 v72, v243
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[126:127], v[50:65]
	v_mov_b32_e32 v69, v240
	v_mov_b32_e32 v74, v253
	v_mov_b32_e32 v70, v241
	v_mov_b32_e32 v68, v239
	v_mov_b32_e32 v75, v254
	v_mov_b32_e32 v71, v242
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[128:129], v[50:65]
	v_mov_b32_e32 v67, v238
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[128:129], v[34:49]
	s_setprio 1
	; sched_barrier mask(0x00000000)
	v_perm_b32 v66, v102, v98, s39
	s_barrier
	ds_write_b32 v238, v66 offset:16384
	v_alignbit_b32 v66, v199, v98, 16
	ds_read_b128 v[190:193], v197
	ds_read_b128 v[174:177], v197 offset:8192
	ds_read_b128 v[186:189], v203
	ds_read_b128 v[170:173], v203 offset:8192
	ds_read_b128 v[182:185], v206
	ds_read_b128 v[166:169], v206 offset:8192
	ds_read_b128 v[178:181], v207
	ds_read_b128 v[162:165], v207 offset:8192
	ds_read_b128 v[94:97], v249
	ds_read_b128 v[126:129], v249 offset:8192
	ds_read_b128 v[90:93], v250
	ds_read_b128 v[122:125], v250 offset:8192
	ds_read_b128 v[86:89], v251
	ds_read_b128 v[118:121], v251 offset:8192
	ds_read_b128 v[82:85], v252
	ds_read_b128 v[114:117], v252 offset:8192
	ds_write_b32 v239, v66 offset:16384
	v_perm_b32 v66, v103, v99, s39
	ds_write_b32 v240, v66 offset:16384
	v_perm_b32 v66, v103, v99, s42
	ds_write_b32 v241, v66 offset:16384
	v_perm_b32 v66, v104, v100, s39
	ds_write_b32 v242, v66 offset:16384
	v_perm_b32 v66, v104, v100, s42
	s_and_b32 s4, s3, 0xffff
	ds_write_b32 v243, v66 offset:16384
	v_perm_b32 v66, v105, v101, s39
	s_or_b32 s13, s4, s33
	s_mov_b32 s14, s22
	s_mov_b32 s15, s23
	ds_write_b32 v244, v66 offset:16384
	v_perm_b32 v66, v105, v101, s42
	buffer_load_dwordx4 v[98:101], v253, s[12:15], 0 offen
	buffer_load_dwordx4 v[102:105], v254, s[12:15], 0 offen
	ds_write_b32 v245, v66 offset:16384
	v_mov_b32_e32 v66, v245
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v199, 16, v102
	; sched_barrier mask(0x00000000)
	s_add_u32 s40, s40, s26
	s_addc_u32 s41, s41, s27
	s_add_u32 s0, s0, s30
	s_addc_u32 s1, s1, s31
	s_add_i32 s2, s2, 64
	s_cmpk_lt_u32 s2, 0x1f00
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	scratch_store_dword off, v248, off offset:136 ; 4-byte Folded Spill
	scratch_store_dword off, v247, off offset:124 ; 4-byte Folded Spill
	scratch_store_dword off, v208, off offset:120 ; 4-byte Folded Spill
	scratch_load_dword v194, off, off offset:84 ; 4-byte Folded Reload
	s_nop 0
	scratch_load_dword v208, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v204, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v202, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v199, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v255, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v248, off, off offset:60 ; 4-byte Folded Reload
	scratch_load_dword v247, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v246, off, off offset:52 ; 4-byte Folded Reload
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dword v66, off, off offset:152 ; 4-byte Folded Reload
	v_pk_mul_f32 v[16:17], v[16:17], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[196:197] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[196:197] op_sel_hi:[1,0]
	s_mul_i32 s4, s18, 0xc0000
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s2, s8, s4
	s_addc_u32 s6, s9, s5
	s_lshl_b32 s4, s17, 14
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s2, s2, s4
	s_addc_u32 s6, s6, s5
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[4:5], s[28:29], 2
	s_add_u32 s4, s2, s4
	s_addc_u32 s16, s6, s5
	s_add_i32 s2, s28, 0xffffc100
	s_add_u32 s12, s12, s26
	s_addc_u32 s3, s3, s27
	s_and_b32 s13, s3, 0xffff
	s_mov_b32 s3, 0x5040100
	s_mov_b32 s5, 0x7060302
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_cmp_lt_i32 s2, 1
	s_mov_b32 s2, 0x3e0293ee
	s_waitcnt vmcnt(0)
	v_cmp_eq_u32_e64 s[0:1], 0, v66
	scratch_load_dword v66, off, off offset:144 ; 4-byte Folded Reload
	scratch_load_dword v67, off, off offset:140 ; 4-byte Folded Reload
	scratch_load_dword v68, off, off offset:148 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v66, 0xa0, v66
	s_waitcnt vmcnt(0)
	v_or3_b32 v66, v66, v68, v67
	scratch_store_dword off, v66, off offset:140 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[66:81], v[192:193], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[184:185], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[180:181], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[96:97], v[144:145], v[66:81]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[66:81], v[86:87], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[88:89], v[136:137], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_f16 v[66:81], v[82:83], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[84:85], v[132:133], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[160:161], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[154:155], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[156:157], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[150:151], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[152:153], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[146:147], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[126:127], v[142:143], v[82:97]
	v_cvt_pkrtz_f16_f32 v126, v229, v232
	v_cvt_pkrtz_f16_f32 v127, v231, v234
	v_mfma_f32_32x32x8_f16 v[82:97], v[128:129], v[144:145], v[82:97]
	v_cvt_pkrtz_f16_f32 v128, v233, v236
	v_cvt_pkrtz_f16_f32 v129, v235, v237
	v_mfma_f32_32x32x8_f16 v[82:97], v[122:123], v[138:139], v[82:97]
	v_cvt_pkrtz_f16_f32 v122, v221, v224
	v_cvt_pkrtz_f16_f32 v123, v223, v226
	v_mfma_f32_32x32x8_f16 v[82:97], v[124:125], v[140:141], v[82:97]
	v_cvt_pkrtz_f16_f32 v124, v225, v228
	v_cvt_pkrtz_f16_f32 v125, v227, v230
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[134:135], v[82:97]
	v_cvt_pkrtz_f16_f32 v118, v213, v216
	v_cvt_pkrtz_f16_f32 v119, v215, v218
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[136:137], v[82:97]
	v_cvt_pkrtz_f16_f32 v121, v219, v222
	v_cvt_pkrtz_f16_f32 v120, v217, v220
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[130:131], v[82:97]
	scratch_load_dword v115, off, off offset:116 ; 4-byte Folded Reload
	v_add_f32_e32 v114, v201, v209
	v_add_f32_e32 v114, v114, v1
	v_add_f32_e32 v114, v114, v210
	v_add_f32_e32 v114, v114, v0
	v_add_f32_e32 v114, v114, v212
	v_add_f32_e32 v114, v114, v211
	v_add_f32_e32 v114, v114, v214
	v_add_f32_e32 v114, v114, v213
	v_add_f32_e32 v114, v114, v216
	v_add_f32_e32 v114, v114, v215
	v_add_f32_e32 v114, v114, v218
	v_add_f32_e32 v114, v114, v217
	v_add_f32_e32 v114, v114, v220
	v_add_f32_e32 v114, v114, v219
	v_add_f32_e32 v114, v114, v222
	v_add_f32_e32 v114, v114, v221
	v_add_f32_e32 v114, v114, v224
	v_add_f32_e32 v114, v114, v223
	v_add_f32_e32 v114, v114, v226
	v_add_f32_e32 v114, v114, v225
	v_add_f32_e32 v114, v114, v228
	v_add_f32_e32 v114, v114, v227
	v_add_f32_e32 v114, v114, v230
	v_add_f32_e32 v114, v114, v229
	v_add_f32_e32 v114, v114, v232
	v_add_f32_e32 v114, v114, v231
	v_add_f32_e32 v114, v114, v234
	v_add_f32_e32 v114, v114, v233
	v_add_f32_e32 v114, v114, v236
	v_add_f32_e32 v114, v114, v235
	v_add_f32_e32 v114, v114, v237
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[132:133], v[82:97]
	v_cvt_pkrtz_f16_f32 v116, v0, v212
	v_cvt_pkrtz_f16_f32 v117, v211, v214
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v115, v115, v114
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v114, v114, v115
	v_fmac_f32_e32 v114, v200, v196
	scratch_store_dword off, v114, off offset:144 ; 4-byte Folded Spill
	v_cvt_pkrtz_f16_f32 v115, v1, v210
	s_barrier
	ds_read_b64 v[0:1], v246 offset:16384
	ds_read_b64 v[162:163], v247 offset:16384
	ds_read_b64 v[164:165], v248 offset:16384
	ds_read_b64 v[166:167], v255 offset:16384
	ds_read_b64 v[168:169], v199 offset:16384
	ds_read_b64 v[170:171], v202 offset:16384
	ds_read_b64 v[172:173], v204 offset:16384
	ds_read_b64 v[174:175], v208 offset:16384
	ds_read_b64 v[176:177], v194 offset:16384
	scratch_load_dword v194, off, off offset:12 ; 4-byte Folded Reload
	v_cvt_pkrtz_f16_f32 v114, v201, v209
	scratch_load_dword v202, off, off offset:108 ; 4-byte Folded Reload
	scratch_load_dword v209, off, off offset:112 ; 4-byte Folded Reload
	scratch_load_dword v178, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v180, off, off offset:44 ; 4-byte Folded Reload
	scratch_load_dword v182, off, off       ; 4-byte Folded Reload
	scratch_load_dword v208, off, off offset:88 ; 4-byte Folded Reload
	scratch_load_dword v246, off, off offset:92 ; 4-byte Folded Reload
	scratch_load_dword v247, off, off offset:96 ; 4-byte Folded Reload
	scratch_load_dword v190, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v192, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v248, off, off offset:100 ; 4-byte Folded Reload
	scratch_load_dword v255, off, off offset:104 ; 4-byte Folded Reload
	scratch_load_dword v204, off, off offset:136 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_f16 v[50:65], v[0:1], v[114:115], v[50:65]
	v_max_f32_e32 v0, v67, v67
	v_max_f32_e32 v1, v66, v66
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v68, v69
	v_max3_f32 v0, v0, v70, v71
	v_max3_f32 v0, v0, v72, v73
	v_max3_f32 v0, v0, v74, v75
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[176:177], v[114:115], v[34:49]
	v_max3_f32 v0, v0, v76, v77
	v_max3_f32 v0, v0, v78, v79
	v_max3_f32 v0, v0, v80, v81
	v_max3_f32 v0, v0, v82, v83
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_mfma_f32_32x32x8_f16 v[50:65], v[162:163], v[116:117], v[50:65]
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	s_waitcnt vmcnt(13)
	ds_read_b64 v[200:201], v194 offset:16384
	scratch_load_dword v194, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(13)
	ds_read_b64 v[224:225], v202 offset:16384
	s_waitcnt vmcnt(12)
	ds_read_b64 v[226:227], v209 offset:16384
	s_waitcnt vmcnt(11)
	ds_read_b64 v[178:179], v178 offset:16384
	s_waitcnt vmcnt(10)
	ds_read_b64 v[180:181], v180 offset:16384
	s_waitcnt vmcnt(9)
	ds_read_b64 v[182:183], v182 offset:16384
	s_waitcnt vmcnt(8)
	ds_read_b64 v[184:185], v208 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[186:187], v246 offset:16384
	s_waitcnt vmcnt(6)
	ds_read_b64 v[188:189], v247 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[190:191], v190 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[192:193], v192 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[214:215], v248 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[216:217], v255 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[236:237], v204 offset:16384
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[18:33], v[192:193], v[114:115], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[210:211], v194 offset:16384
	scratch_load_dword v194, off, off offset:20 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[116:117], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v194 offset:16384
	scratch_load_dword v194, off, off offset:24 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[18:33], v[200:201], v[116:117], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[218:219], v194 offset:16384
	scratch_load_dword v194, off, off offset:28 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[50:65], v[164:165], v[118:119], v[50:65]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[220:221], v194 offset:16384
	scratch_load_dword v194, off, off offset:32 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[118:119], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[222:223], v194 offset:16384
	scratch_load_dword v194, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[222:223], v[114:115], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[228:229], v194 offset:16384
	scratch_load_dword v194, off, off offset:40 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[2:17], v[224:225], v[116:117], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[230:231], v194 offset:16384
	scratch_load_dword v194, off, off offset:120 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[2:17], v[226:227], v[118:119], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[232:233], v194 offset:16384
	scratch_load_dword v194, off, off offset:124 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[228:229], v[120:121], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[234:235], v194 offset:16384
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[230:231], v[122:123], v[2:17]
	ds_write_b128 v198, v[110:113]
	ds_write_b128 v198, v[106:109] offset:8192
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[2:17], v[232:233], v[124:125], v[2:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[234:235], v[126:127], v[2:17]
	scratch_load_dword v235, off, off offset:116 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v1, v235, v0
	v_mfma_f32_32x32x8_f16 v[18:33], v[210:211], v[118:119], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v234, v195, v0, v1
	v_perm_b32 v0, v102, v98, s3
	v_mfma_f32_32x32x8_f16 v[50:65], v[166:167], v[120:121], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[120:121], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[212:213], v[120:121], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[168:169], v[122:123], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[122:123], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[214:215], v[122:123], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[170:171], v[124:125], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[124:125], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[216:217], v[124:125], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[126:127], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[188:189], v[126:127], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[218:219], v[126:127], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[128:129], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[128:129], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[220:221], v[128:129], v[18:33]
	ds_read_b128 v[106:109], v197
	ds_read_b128 v[110:113], v203
	ds_read_b128 v[210:213], v206
	ds_read_b128 v[214:217], v207
	ds_read_b128 v[218:221], v249
	ds_read_b128 v[222:225], v250
	ds_read_b128 v[226:229], v251
	ds_read_b128 v[230:233], v252
	ds_read_b128 v[198:201], v197 offset:8192
	ds_read_b128 v[194:197], v203 offset:8192
	ds_read_b128 v[190:193], v206 offset:8192
	ds_read_b128 v[186:189], v207 offset:8192
	ds_read_b128 v[182:185], v249 offset:8192
	ds_read_b128 v[178:181], v250 offset:8192
	ds_read_b128 v[174:177], v251 offset:8192
	ds_read_b128 v[170:173], v252 offset:8192
	ds_write_b32 v238, v0 offset:16384
	v_perm_b32 v0, v102, v98, s5
	ds_write_b32 v239, v0 offset:16384
	v_perm_b32 v0, v103, v99, s3
	ds_write_b32 v240, v0 offset:16384
	v_perm_b32 v0, v103, v99, s5
	v_mfma_f32_32x32x8_f16 v[2:17], v[236:237], v[128:129], v[2:17]
	ds_write_b32 v241, v0 offset:16384
	v_perm_b32 v0, v104, v100, s3
	ds_write_b32 v242, v0 offset:16384
	v_perm_b32 v0, v104, v100, s5
	ds_write_b32 v243, v0 offset:16384
	v_perm_b32 v0, v105, v101, s3
	ds_write_b32 v244, v0 offset:16384
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[114:129], v[106:107], v[158:159], 0
	v_perm_b32 v0, v105, v101, s5
	ds_write_b32 v245, v0 offset:16384
	buffer_load_dwordx4 v[162:165], v253, s[12:15], 0 offen
	buffer_load_dwordx4 v[166:169], v254, s[12:15], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v203, off, off offset:84 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[114:129], v[108:109], v[160:161], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[110:111], v[154:155], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[112:113], v[156:157], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[210:211], v[150:151], v[114:129]
	v_mov_b32_e32 v211, v235
	v_mfma_f32_32x32x8_f16 v[114:129], v[212:213], v[152:153], v[114:129]
	scratch_load_dword v212, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v213, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	ds_read_b64 v[206:207], v212 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[198:199], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[114:129], v[214:215], v[146:147], v[114:129]
	scratch_load_dword v214, off, off offset:60 ; 4-byte Folded Reload
	scratch_load_dword v215, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	ds_read_b64 v[198:199], v214 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[200:201], v[160:161], v[98:113]
	ds_read_b64 v[200:201], v213 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[216:217], v[148:149], v[114:129]
	scratch_load_dword v216, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v217, off, off offset:72 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[194:195], v[154:155], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[218:219], v[142:143], v[114:129]
	scratch_load_dword v218, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v219, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	ds_read_b64 v[194:195], v218 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[196:197], v[156:157], v[98:113]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[196:197], v219 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[220:221], v[144:145], v[114:129]
	scratch_load_dword v220, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v221, off, off offset:44 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[190:191], v[150:151], v[98:113]
	ds_read_b64 v[190:191], v216 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[222:223], v[138:139], v[114:129]
	scratch_load_dword v222, off, off       ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[192:193], v[152:153], v[98:113]
	ds_read_b64 v[192:193], v217 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[186:187], v[146:147], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[188:189], v[148:149], v[98:113]
	ds_read_b64 v[188:189], v215 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[182:183], v[142:143], v[98:113]
	s_waitcnt vmcnt(2)
	ds_read_b64 v[182:183], v220 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[184:185], v[144:145], v[98:113]
	s_waitcnt vmcnt(1)
	ds_read_b64 v[184:185], v221 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[178:179], v[138:139], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[180:181], v[140:141], v[98:113]
	ds_read_b64 v[180:181], v203 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[174:175], v[134:135], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[176:177], v[136:137], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[224:225], v[140:141], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[170:171], v[130:131], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[226:227], v[134:135], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[172:173], v[132:133], v[98:113]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[186:187], v222 offset:16384
	ds_read_b64 v[172:173], v208 offset:16384
	ds_read_b64 v[174:175], v246 offset:16384
	ds_read_b64 v[176:177], v247 offset:16384
	scratch_load_dword v223, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v224, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v225, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v226, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v227, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(4)
	ds_read_b64 v[178:179], v223 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[228:229], v[136:137], v[114:129]
	s_waitcnt vmcnt(3)
	ds_read_b64 v[156:157], v224 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[158:159], v225 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[160:161], v226 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v227 offset:16384
	ds_read_b64 v[148:149], v248 offset:16384
	ds_read_b64 v[150:151], v255 offset:16384
	scratch_load_dword v228, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v229, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v236, off, off offset:40 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[114:129], v[230:231], v[130:131], v[114:129]
	scratch_load_dword v230, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v237, off, off offset:120 ; 4-byte Folded Reload
	scratch_load_dword v249, off, off offset:124 ; 4-byte Folded Reload
	v_mov_b32_e32 v231, v202
	ds_read_b64 v[134:135], v202 offset:16384
	ds_read_b64 v[136:137], v209 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[152:153], v228 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[232:233], v[132:133], v[114:129]
	scratch_load_dword v233, off, off offset:36 ; 4-byte Folded Reload
	v_mov_b32_e32 v232, v209
	s_waitcnt vmcnt(5)
	ds_read_b64 v[154:155], v229 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[132:133], v230 offset:16384
	ds_read_b64 v[140:141], v236 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[142:143], v237 offset:16384
	s_nop 1
	v_max_f32_e32 v0, v115, v115
	v_max_f32_e32 v1, v114, v114
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v116, v117
	v_max3_f32 v0, v0, v118, v119
	v_max3_f32 v0, v0, v120, v121
	v_max3_f32 v0, v0, v122, v123
	v_max3_f32 v0, v0, v124, v125
	v_max3_f32 v0, v0, v126, v127
	v_max3_f32 v0, v0, v128, v129
	v_max3_f32 v0, v0, v98, v99
	v_max3_f32 v0, v0, v100, v101
	v_max3_f32 v0, v0, v102, v103
	v_max3_f32 v0, v0, v104, v105
	v_max3_f32 v0, v0, v106, v107
	v_max3_f32 v0, v0, v108, v109
	v_max3_f32 v0, v0, v110, v111
	v_max3_f32 v0, v0, v112, v113
	ds_bpermute_b32 v1, v235, v0
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v235, v234, v0, v1
	v_pk_mul_f32 v[130:131], v[234:235], s[2:3] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v0, v66, s2, -v130
	v_fma_f32 v1, v67, s2, -v130
	v_fma_f32 v66, v68, s2, -v130
	v_fma_f32 v68, v70, s2, -v130
	v_fma_f32 v70, v72, s2, -v130
	v_fma_f32 v72, v74, s2, -v130
	v_fma_f32 v74, v76, s2, -v130
	v_fma_f32 v76, v78, s2, -v130
	v_fma_f32 v78, v80, s2, -v130
	v_fma_f32 v80, v82, s2, -v130
	v_fma_f32 v82, v84, s2, -v130
	v_fma_f32 v84, v86, s2, -v130
	v_fma_f32 v86, v88, s2, -v130
	v_fma_f32 v88, v90, s2, -v130
	v_fma_f32 v90, v92, s2, -v130
	v_fma_f32 v92, v94, s2, -v130
	v_fma_f32 v94, v96, s2, -v130
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v96, v1
	v_fma_f32 v67, v69, s2, -v130
	v_fma_f32 v69, v71, s2, -v130
	v_fma_f32 v71, v73, s2, -v130
	v_fma_f32 v73, v75, s2, -v130
	v_fma_f32 v75, v77, s2, -v130
	v_fma_f32 v77, v79, s2, -v130
	v_fma_f32 v79, v81, s2, -v130
	v_fma_f32 v81, v83, s2, -v130
	v_fma_f32 v83, v85, s2, -v130
	v_fma_f32 v85, v87, s2, -v130
	v_fma_f32 v87, v89, s2, -v130
	v_fma_f32 v89, v91, s2, -v130
	v_fma_f32 v91, v93, s2, -v130
	v_fma_f32 v93, v95, s2, -v130
	v_fma_f32 v95, v97, s2, -v130
	v_exp_f32_e32 v97, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v202, v68
	v_sub_f32_e32 v1, v205, v130
	v_exp_f32_e32 v209, v69
	v_exp_f32_e32 v66, v1
	v_add_f32_e32 v1, v0, v96
	v_exp_f32_e32 v210, v70
	v_add_f32_e32 v1, v97, v1
	v_exp_f32_e32 v71, v71
	v_add_f32_e32 v1, v67, v1
	v_exp_f32_e32 v72, v72
	v_add_f32_e32 v1, v202, v1
	v_exp_f32_e32 v73, v73
	v_add_f32_e32 v1, v209, v1
	v_exp_f32_e32 v74, v74
	v_add_f32_e32 v1, v210, v1
	v_exp_f32_e32 v75, v75
	v_add_f32_e32 v1, v71, v1
	v_exp_f32_e32 v76, v76
	v_add_f32_e32 v1, v72, v1
	v_exp_f32_e32 v77, v77
	v_add_f32_e32 v1, v73, v1
	v_exp_f32_e32 v78, v78
	v_add_f32_e32 v1, v74, v1
	v_exp_f32_e32 v79, v79
	v_add_f32_e32 v1, v75, v1
	v_exp_f32_e32 v80, v80
	v_add_f32_e32 v1, v76, v1
	v_exp_f32_e32 v81, v81
	v_add_f32_e32 v1, v77, v1
	v_exp_f32_e32 v82, v82
	v_add_f32_e32 v1, v78, v1
	v_exp_f32_e32 v83, v83
	v_add_f32_e32 v1, v79, v1
	v_exp_f32_e32 v84, v84
	v_add_f32_e32 v1, v80, v1
	v_exp_f32_e32 v85, v85
	v_add_f32_e32 v1, v81, v1
	v_exp_f32_e32 v86, v86
	v_add_f32_e32 v1, v82, v1
	v_exp_f32_e32 v87, v87
	v_add_f32_e32 v1, v83, v1
	v_exp_f32_e32 v88, v88
	v_add_f32_e32 v1, v84, v1
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v1, v85, v1
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v1, v86, v1
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v1, v87, v1
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v1, v88, v1
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v1, v89, v1
	v_exp_f32_e32 v94, v94
	v_add_f32_e32 v1, v90, v1
	v_exp_f32_e32 v95, v95
	v_add_f32_e32 v1, v91, v1
	v_add_f32_e32 v1, v92, v1
	v_add_f32_e32 v1, v93, v1
	v_add_f32_e32 v1, v94, v1
	v_add_f32_e32 v1, v95, v1
	ds_bpermute_b32 v68, v211, v1
	s_waitcnt vmcnt(0)
	ds_read_b64 v[138:139], v233 offset:16384
	ds_read_b64 v[144:145], v249 offset:16384
	ds_read_b64 v[146:147], v204 offset:16384
	v_cvt_pkrtz_f16_f32 v69, v97, v67
	v_pk_mul_f32 v[64:65], v[64:65], v[66:67] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v1, v1, v68
	scratch_load_dword v68, off, off offset:144 ; 4-byte Folded Reload
	v_pk_mul_f32 v[62:63], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[66:67] op_sel_hi:[1,0]
	v_cvt_pkrtz_f16_f32 v70, v202, v209
	v_cvt_pkrtz_f16_f32 v71, v210, v71
	v_cvt_pkrtz_f16_f32 v72, v72, v73
	v_cvt_pkrtz_f16_f32 v73, v74, v75
	v_cvt_pkrtz_f16_f32 v74, v76, v77
	v_cvt_pkrtz_f16_f32 v75, v78, v79
	v_cvt_pkrtz_f16_f32 v76, v80, v81
	v_cvt_pkrtz_f16_f32 v77, v82, v83
	v_cvt_pkrtz_f16_f32 v78, v84, v85
	v_cvt_pkrtz_f16_f32 v79, v86, v87
	v_cvt_pkrtz_f16_f32 v80, v88, v89
	v_cvt_pkrtz_f16_f32 v81, v90, v91
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_fma_f32 v67, v116, s2, -v131
	v_exp_f32_e32 v67, v67
	v_cvt_pkrtz_f16_f32 v82, v92, v93
	v_cvt_pkrtz_f16_f32 v83, v94, v95
	v_fma_f32 v84, v101, s2, -v131
	v_fma_f32 v85, v102, s2, -v131
	v_exp_f32_e32 v84, v84
	v_fma_f32 v86, v103, s2, -v131
	v_exp_f32_e32 v85, v85
	v_fma_f32 v87, v104, s2, -v131
	v_exp_f32_e32 v101, v86
	v_fma_f32 v88, v105, s2, -v131
	v_exp_f32_e32 v102, v87
	v_fma_f32 v89, v106, s2, -v131
	v_exp_f32_e32 v88, v88
	v_fma_f32 v90, v107, s2, -v131
	v_exp_f32_e32 v89, v89
	v_fma_f32 v91, v108, s2, -v131
	v_exp_f32_e32 v103, v90
	v_fma_f32 v92, v109, s2, -v131
	v_exp_f32_e32 v104, v91
	v_fma_f32 v93, v110, s2, -v131
	v_exp_f32_e32 v92, v92
	v_fma_f32 v94, v111, s2, -v131
	v_exp_f32_e32 v93, v93
	v_fma_f32 v95, v112, s2, -v131
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v1, v68, v66
	v_cvt_pkrtz_f16_f32 v68, v0, v96
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[206:207], v[68:69], v[50:65]
	v_fma_f32 v0, v114, s2, -v131
	v_fma_f32 v66, v115, s2, -v131
	v_exp_f32_e32 v97, v0
	v_sub_f32_e32 v0, v130, v131
	v_fma_f32 v96, v113, s2, -v131
	v_exp_f32_e32 v96, v96
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[156:157], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[132:133], v[68:69], v[2:17]
	v_fma_f32 v68, v117, s2, -v131
	v_fma_f32 v69, v118, s2, -v131
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_cvt_pkrtz_f16_f32 v91, v67, v68
	v_mfma_f32_32x32x8_f16 v[50:65], v[200:201], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[158:159], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[134:135], v[70:71], v[2:17]
	v_fma_f32 v70, v119, s2, -v131
	v_fma_f32 v71, v120, s2, -v131
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_cvt_pkrtz_f16_f32 v86, v69, v70
	v_mfma_f32_32x32x8_f16 v[50:65], v[198:199], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[160:161], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[136:137], v[72:73], v[2:17]
	v_fma_f32 v72, v121, s2, -v131
	v_fma_f32 v73, v122, s2, -v131
	v_exp_f32_e32 v72, v72
	v_exp_f32_e32 v73, v73
	v_cvt_pkrtz_f16_f32 v87, v71, v72
	v_mfma_f32_32x32x8_f16 v[50:65], v[188:189], v[74:75], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[74:75], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[170:171], v[74:75], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[138:139], v[74:75], v[2:17]
	v_fma_f32 v74, v123, s2, -v131
	v_fma_f32 v75, v124, s2, -v131
	v_exp_f32_e32 v74, v74
	v_exp_f32_e32 v75, v75
	v_mfma_f32_32x32x8_f16 v[50:65], v[190:191], v[76:77], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[76:77], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[148:149], v[76:77], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[140:141], v[76:77], v[2:17]
	v_fma_f32 v76, v125, s2, -v131
	v_fma_f32 v77, v126, s2, -v131
	v_exp_f32_e32 v76, v76
	v_exp_f32_e32 v77, v77
	v_mfma_f32_32x32x8_f16 v[50:65], v[192:193], v[78:79], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[174:175], v[78:79], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[150:151], v[78:79], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[142:143], v[78:79], v[2:17]
	v_fma_f32 v78, v127, s2, -v131
	v_fma_f32 v79, v128, s2, -v131
	v_exp_f32_e32 v78, v78
	v_exp_f32_e32 v79, v79
	v_mfma_f32_32x32x8_f16 v[50:65], v[194:195], v[80:81], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[176:177], v[80:81], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[152:153], v[80:81], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[144:145], v[80:81], v[2:17]
	v_fma_f32 v81, v98, s2, -v131
	v_exp_f32_e32 v98, v66
	v_exp_f32_e32 v66, v0
	v_perm_b32 v0, v166, v162, s3
	ds_write_b32 v238, v0 offset:16384
	v_perm_b32 v0, v166, v162, s5
	ds_write_b32 v239, v0 offset:16384
	v_perm_b32 v0, v167, v163, s3
	ds_write_b32 v240, v0 offset:16384
	v_perm_b32 v0, v167, v163, s5
	ds_write_b32 v241, v0 offset:16384
	v_perm_b32 v0, v168, v164, s3
	ds_write_b32 v242, v0 offset:16384
	v_perm_b32 v0, v168, v164, s5
	ds_write_b32 v243, v0 offset:16384
	v_perm_b32 v0, v169, v165, s3
	ds_write_b32 v244, v0 offset:16384
	v_perm_b32 v0, v169, v165, s5
	ds_write_b32 v245, v0 offset:16384
	v_add_f32_e32 v0, v97, v98
	v_add_f32_e32 v0, v67, v0
	v_add_f32_e32 v0, v68, v0
	v_add_f32_e32 v0, v69, v0
	v_add_f32_e32 v0, v70, v0
	v_add_f32_e32 v0, v71, v0
	v_add_f32_e32 v0, v72, v0
	v_add_f32_e32 v0, v73, v0
	v_add_f32_e32 v0, v74, v0
	v_fma_f32 v80, v129, s2, -v131
	v_add_f32_e32 v0, v75, v0
	v_exp_f32_e32 v80, v80
	v_add_f32_e32 v0, v76, v0
	v_mfma_f32_32x32x8_f16 v[50:65], v[196:197], v[82:83], v[50:65]
	v_exp_f32_e32 v81, v81
	v_add_f32_e32 v0, v77, v0
	v_add_f32_e32 v0, v78, v0
	v_add_f32_e32 v0, v79, v0
	v_add_f32_e32 v0, v80, v0
	v_add_f32_e32 v0, v81, v0
	v_cvt_pkrtz_f16_f32 v90, v97, v98
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[82:83], v[34:49]
	s_nop 2
	v_pk_mul_f32 v[64:65], v[64:65], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[18:33], v[154:155], v[82:83], v[18:33]
	v_pk_mul_f32 v[50:51], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[146:147], v[82:83], v[2:17]
	v_fma_f32 v82, v99, s2, -v131
	v_fma_f32 v83, v100, s2, -v131
	v_exp_f32_e32 v99, v82
	v_exp_f32_e32 v100, v83
	v_pk_mul_f32 v[36:37], v[36:37], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[66:67] op_sel_hi:[1,0]
	v_add_f32_e32 v0, v99, v0
	v_add_f32_e32 v0, v100, v0
	v_add_f32_e32 v0, v84, v0
	v_add_f32_e32 v0, v85, v0
	v_add_f32_e32 v0, v101, v0
	v_add_f32_e32 v0, v102, v0
	v_add_f32_e32 v0, v88, v0
	v_add_f32_e32 v0, v89, v0
	v_add_f32_e32 v0, v103, v0
	v_add_f32_e32 v0, v104, v0
	v_add_f32_e32 v0, v92, v0
	v_add_f32_e32 v0, v93, v0
	v_add_f32_e32 v0, v94, v0
	v_add_f32_e32 v0, v95, v0
	v_add_f32_e32 v0, v96, v0
	ds_bpermute_b32 v82, v211, v0
	v_pk_mul_f32 v[32:33], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[66:67] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v82
	v_cvt_pkrtz_f16_f32 v82, v73, v74
	v_cvt_pkrtz_f16_f32 v83, v75, v76
	v_cvt_pkrtz_f16_f32 v76, v77, v78
	v_cvt_pkrtz_f16_f32 v77, v79, v80
	v_cvt_pkrtz_f16_f32 v74, v81, v99
	v_cvt_pkrtz_f16_f32 v75, v100, v84
	v_cvt_pkrtz_f16_f32 v72, v85, v101
	v_cvt_pkrtz_f16_f32 v73, v102, v88
	v_cvt_pkrtz_f16_f32 v68, v89, v103
	v_cvt_pkrtz_f16_f32 v69, v104, v92
	v_cvt_pkrtz_f16_f32 v70, v93, v94
	v_cvt_pkrtz_f16_f32 v71, v95, v96
	s_barrier
	ds_read_b64 v[140:141], v212 offset:16384
	ds_read_b64 v[142:143], v213 offset:16384
	ds_read_b64 v[144:145], v214 offset:16384
	ds_read_b64 v[146:147], v215 offset:16384
	ds_read_b64 v[138:139], v216 offset:16384
	ds_read_b64 v[136:137], v217 offset:16384
	ds_read_b64 v[134:135], v218 offset:16384
	ds_read_b64 v[132:133], v219 offset:16384
	ds_read_b64 v[130:131], v203 offset:16384
	ds_read_b64 v[128:129], v220 offset:16384
	ds_read_b64 v[126:127], v221 offset:16384
	ds_read_b64 v[124:125], v222 offset:16384
	ds_read_b64 v[122:123], v208 offset:16384
	ds_read_b64 v[120:121], v246 offset:16384
	ds_read_b64 v[116:117], v247 offset:16384
	ds_read_b64 v[118:119], v223 offset:16384
	ds_read_b64 v[114:115], v224 offset:16384
	ds_read_b64 v[112:113], v225 offset:16384
	ds_read_b64 v[110:111], v226 offset:16384
	ds_read_b64 v[108:109], v227 offset:16384
	ds_read_b64 v[106:107], v248 offset:16384
	ds_read_b64 v[104:105], v255 offset:16384
	ds_read_b64 v[102:103], v228 offset:16384
	ds_read_b64 v[100:101], v229 offset:16384
	ds_read_b64 v[98:99], v230 offset:16384
	ds_read_b64 v[96:97], v231 offset:16384
	ds_read_b64 v[94:95], v232 offset:16384
	ds_read_b64 v[92:93], v233 offset:16384
	ds_read_b64 v[88:89], v236 offset:16384
	ds_read_b64 v[84:85], v237 offset:16384
	ds_read_b64 v[78:79], v249 offset:16384
	ds_read_b64 v[80:81], v204 offset:16384
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[50:65], v[140:141], v[90:91], v[50:65]
	v_fmac_f32_e32 v0, v1, v66
	v_mfma_f32_32x32x8_f16 v[34:49], v[130:131], v[90:91], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[114:115], v[90:91], v[18:33]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 v[2:17], v[98:99], v[90:91], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[142:143], v[86:87], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[128:129], v[86:87], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[112:113], v[86:87], v[18:33]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[2:17], v[96:97], v[86:87], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[144:145], v[82:83], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[126:127], v[82:83], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[110:111], v[82:83], v[18:33]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[2:17], v[94:95], v[82:83], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[146:147], v[76:77], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[124:125], v[76:77], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[76:77], v[18:33]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[2:17], v[92:93], v[76:77], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[138:139], v[74:75], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[122:123], v[74:75], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[106:107], v[74:75], v[18:33]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[2:17], v[88:89], v[74:75], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[136:137], v[72:73], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[120:121], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[104:105], v[72:73], v[18:33]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[84:85], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[134:135], v[68:69], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[116:117], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[102:103], v[68:69], v[18:33]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[78:79], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[132:133], v[70:71], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[118:119], v[70:71], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[100:101], v[70:71], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[80:81], v[70:71], v[2:17]
	scratch_load_dword v70, off, off offset:140 ; 4-byte Folded Reload
	s_barrier
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v70, 2, 0
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	scratch_load_dword v69, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v0
	s_nop 1
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v0, v68
	v_log_f32_e32 v68, v68
	v_mov_b32_e32 v67, 0x42000000
	v_or_b32_e32 v66, s28, v70
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v67, 0, v67, vcc
	v_cmp_gt_i32_e64 s[8:9], s2, v66
	v_sub_f32_e32 v66, v68, v67
	v_add_f32_e32 v66, v235, v66
	ds_write_b32 v1, v66
	v_mov_b32_e32 v66, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_sub_i32 s2, 0x4000, s28
	v_bfrev_b32_e32 v68, 1
	s_and_b32 s5, s16, 0xffff
	s_mov_b32 s6, s14
	s_mov_b32 s7, s15
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v66, v66, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v67, 0, v66
	ds_read_b32 v67, v67
	v_cmp_lt_i32_sdwa s[2:3], v69, s2 src0_sel:BYTE_0 src1_sel:DWORD
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v66, v68, v66, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v66, s[4:7], 0 offen
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_9:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v0
	s_nop 1
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v0, v67
	v_log_f32_e32 v67, v67
	v_mov_b32_e32 v66, 0x42000000
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_and_b32 s5, s16, 0xffff
	v_sub_f32_e32 v66, v67, v66
	v_add_f32_e32 v66, v235, v66
	ds_write_b32 v1, v66
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v66, off, off offset:128 ; 4-byte Folded Reload
	v_mov_b32_e32 v1, 2
	v_bfrev_b32_e32 v67, 1
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v1, v1, v66 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v1
	ds_read_b32 v66, v66
	v_cndmask_b32_e64 v1, v67, v1, s[0:1]
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v1, s[4:7], 0 offen
.LBB0_10:                               ; %.critedge
	v_div_scale_f32 v1, s[0:1], v0, v0, 1.0
	v_rcp_f32_e32 v1, v1
	v_div_scale_f32 v66, vcc, 1.0, v0, 1.0
	v_mov_b32_e32 v67, v16
	v_mul_f32_e32 v1, v66, v1
	s_nop 1
	v_div_fmas_f32 v1, 0, 0, v1
	v_div_fixup_f32 v0, v1, v0, 1.0
	v_mov_b32_e32 v66, v15
	v_fma_mixlo_f16 v68, v0, v17, 0
	v_pk_mul_f32 v[16:17], v[0:1], v[66:67] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v66, v0, v14, 0
	v_mov_b32_e32 v14, v11
	v_mov_b32_e32 v15, v12
	v_fma_mixlo_f16 v67, v0, v13, 0
	v_pk_mul_f32 v[12:13], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v14, v0, v10, 0
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v8
	v_fma_mixlo_f16 v15, v0, v9, 0
	v_pk_mul_f32 v[8:9], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v10, v0, v6, 0
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v4
	v_fma_mixlo_f16 v11, v0, v5, 0
	v_pk_mul_f32 v[4:5], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v6, v0, v2, 0
	v_mov_b32_e32 v2, v31
	v_mov_b32_e32 v3, v32
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v31, v3
	v_cvt_f16_f32_e32 v32, v2
	v_mov_b32_e32 v2, v27
	v_mov_b32_e32 v3, v28
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v27, v3
	v_cvt_f16_f32_e32 v28, v2
	v_mov_b32_e32 v2, v23
	v_mov_b32_e32 v3, v24
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v23, v3
	v_cvt_f16_f32_e32 v24, v2
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v3, v20
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v19, v3
	v_cvt_f16_f32_e32 v20, v2
	v_mov_b32_e32 v2, v47
	v_mov_b32_e32 v3, v48
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v47, v3
	v_cvt_f16_f32_e32 v48, v2
	v_mov_b32_e32 v2, v43
	v_mov_b32_e32 v3, v44
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v43, v3
	v_cvt_f16_f32_e32 v44, v2
	v_mov_b32_e32 v2, v39
	v_mov_b32_e32 v3, v40
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v39, v3
	v_cvt_f16_f32_e32 v40, v2
	v_mov_b32_e32 v2, v35
	v_mov_b32_e32 v3, v36
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v35, v3
	v_cvt_f16_f32_e32 v36, v2
	v_mov_b32_e32 v2, v63
	v_mov_b32_e32 v3, v64
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v63, v3
	v_cvt_f16_f32_e32 v64, v2
	v_mov_b32_e32 v2, v59
	v_mov_b32_e32 v3, v60
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v59, v3
	v_cvt_f16_f32_e32 v60, v2
	v_mov_b32_e32 v2, v55
	v_mov_b32_e32 v3, v56
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v55, v3
	v_cvt_f16_f32_e32 v56, v2
	v_fma_mixlo_f16 v1, v0, v53, 0
	v_mov_b32_e32 v2, v51
	v_mov_b32_e32 v3, v52
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v2, v2
	v_fma_mixlo_f16 v7, v0, v33, 0
	v_fma_mixlo_f16 v30, v0, v30, 0
	v_fma_mixlo_f16 v29, v0, v29, 0
	v_fma_mixlo_f16 v26, v0, v26, 0
	v_fma_mixlo_f16 v25, v0, v25, 0
	v_fma_mixlo_f16 v22, v0, v22, 0
	v_fma_mixlo_f16 v21, v0, v21, 0
	v_fma_mixlo_f16 v18, v0, v18, 0
	v_fma_mixlo_f16 v33, v0, v49, 0
	v_fma_mixlo_f16 v46, v0, v46, 0
	v_fma_mixlo_f16 v45, v0, v45, 0
	v_fma_mixlo_f16 v42, v0, v42, 0
	v_fma_mixlo_f16 v41, v0, v41, 0
	v_fma_mixlo_f16 v38, v0, v38, 0
	v_fma_mixlo_f16 v37, v0, v37, 0
	v_fma_mixlo_f16 v34, v0, v34, 0
	v_fma_mixlo_f16 v49, v0, v65, 0
	v_fma_mixlo_f16 v62, v0, v62, 0
	v_fma_mixlo_f16 v61, v0, v61, 0
	v_fma_mixlo_f16 v58, v0, v58, 0
	v_fma_mixlo_f16 v57, v0, v57, 0
	v_fma_mixlo_f16 v54, v0, v54, 0
	v_fma_mixlo_f16 v0, v0, v50, 0
	v_pack_b32_f16 v0, v0, v2
	scratch_load_dword v2, off, off offset:132 ; 4-byte Folded Reload
	s_mul_i32 s0, s24, s18
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s25, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s19, s28
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_cvt_f16_f32_e32 v3, v3
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s19, 0x3fff
	v_mul_lo_u32 v50, s19, v70
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s4, 0x5040100
	s_or_b32 s1, s2, s1
	v_perm_b32 v1, v1, v3, s4
	v_bfrev_b32_e32 v3, 1
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v5, v5
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v12, v12
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v16, v16
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v2, v50, v2, 1
	v_cndmask_b32_e64 v50, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_add_u32_e32 v50, 16, v2
	v_pack_b32_f16 v0, v54, v56
	v_perm_b32 v1, v57, v55, s4
	v_cndmask_b32_e64 v50, v3, v50, s[8:9]
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_add_u32_e32 v50, 32, v2
	v_pack_b32_f16 v0, v58, v60
	v_perm_b32 v1, v61, v59, s4
	v_cndmask_b32_e64 v50, v3, v50, s[8:9]
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_perm_b32 v1, v49, v63, s4
	v_add_u32_e32 v49, 48, v2
	v_pack_b32_f16 v0, v62, v64
	v_cndmask_b32_e64 v49, v3, v49, s[8:9]
	buffer_store_dwordx2 v[0:1], v49, s[0:3], 0 offen
	v_pack_b32_f16 v0, v34, v36
	v_add_u32_e32 v34, 64, v2
	v_perm_b32 v1, v37, v35, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_add_u32_e32 v34, 0x50, v2
	v_pack_b32_f16 v0, v38, v40
	v_perm_b32 v1, v41, v39, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_add_u32_e32 v34, 0x60, v2
	v_pack_b32_f16 v0, v42, v44
	v_perm_b32 v1, v45, v43, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_perm_b32 v1, v33, v47, s4
	v_add_u32_e32 v33, 0x70, v2
	v_pack_b32_f16 v0, v46, v48
	v_cndmask_b32_e64 v33, v3, v33, s[8:9]
	buffer_store_dwordx2 v[0:1], v33, s[0:3], 0 offen
	v_pack_b32_f16 v0, v18, v20
	v_add_u32_e32 v18, 0x80, v2
	v_perm_b32 v1, v21, v19, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_add_u32_e32 v18, 0x90, v2
	v_pack_b32_f16 v0, v22, v24
	v_perm_b32 v1, v25, v23, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_add_u32_e32 v18, 0xa0, v2
	v_pack_b32_f16 v0, v26, v28
	v_perm_b32 v1, v29, v27, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_perm_b32 v1, v7, v31, s4
	v_add_u32_e32 v7, 0xb0, v2
	v_pack_b32_f16 v0, v30, v32
	v_cndmask_b32_e64 v7, v3, v7, s[8:9]
	buffer_store_dwordx2 v[0:1], v7, s[0:3], 0 offen
	v_pack_b32_f16 v0, v6, v4
	v_add_u32_e32 v4, 0xc0, v2
	v_perm_b32 v1, v11, v5, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xd0, v2
	v_pack_b32_f16 v0, v10, v8
	v_perm_b32 v1, v15, v9, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xe0, v2
	v_pack_b32_f16 v0, v14, v12
	v_perm_b32 v1, v67, v13, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	v_add_u32_e32 v2, 0xf0, v2
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_pack_b32_f16 v0, v66, v16
	v_perm_b32 v1, v68, v17, s4
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 160
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
		.amdhsa_next_free_sgpr 43
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
	.set attn_fwd.numbered_sgpr, 43
	.set attn_fwd.private_seg_size, 160
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15676
; TotalNumSgprs: 49
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 160
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 49
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
	.short	679                             ; DW_AT_call_line
	.byte	61                              ; DW_AT_call_column
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
    .private_segment_fixed_size: 160
    .sgpr_count:     49
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 41
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
