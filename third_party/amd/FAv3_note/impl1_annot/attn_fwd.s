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
; %bb.7:
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.8:
.LBB0_0:
	s_load_dwordx8 s[20:27], s[0:1], 0x38
	s_mul_i32 s0, s12, s18
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s34, s16, 8
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s13, s17
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_mov_b32_e32 v79, v0
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v34, 4, v79
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s14, s34
	v_or_b32_e32 v0, 0x60, v34
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_or_b32_e32 v11, s34, v0
	s_lshl_b32 s16, s14, 6
	v_mul_lo_u32 v12, s14, v0
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v0, 3, v79
	v_or_b32_e32 v1, 0xa0, v34
	s_add_u32 s0, s2, s0
	v_and_b32_e32 v58, 0x78, v0
	s_mul_i32 s52, s15, s18
	v_or_b32_e32 v19, s34, v1
	v_mul_lo_u32 v20, s14, v1
	s_addc_u32 s1, s3, s1
	v_mad_u64_u32 v[0:1], s[2:3], s14, v34, v[58:59]
	s_ashr_i32 s53, s52, 31
	s_lshl_b64 s[2:3], s[52:53], 1
	s_add_u32 s12, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s54, s20, s17
	s_addc_u32 s13, s5, s3
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[2:3], s[54:55], 1
	s_add_u32 s28, s12, s2
	s_mul_i32 s2, s22, s18
	s_addc_u32 s13, s13, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s23, s17
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s6, s2
	s_addc_u32 s33, s7, s3
	s_and_b32 s2, s14, 0x3fff
	v_or_b32_e32 v35, 32, v34
	v_or_b32_e32 v3, s34, v34
	s_movk_i32 s6, 0x4000
	s_bitset1_b32 s2, 14
	v_or_b32_e32 v2, 0xe0, v34
	v_or_b32_e32 v4, s34, v35
	v_mul_lo_u32 v5, s14, v35
	v_add_u32_e32 v1, s16, v0
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_lshlrev_b32_e32 v0, 1, v0
	v_bfrev_b32_e32 v30, 1
	v_cmp_gt_i32_e32 vcc, s6, v3
	v_or_b32_e32 v10, 64, v3
	v_or_b32_e32 v27, s34, v2
	v_mul_lo_u32 v28, s14, v2
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_add_lshl_u32 v2, v5, v58, 1
	v_cmp_gt_i32_e32 vcc, s6, v4
	v_or_b32_e32 v18, 0x80, v3
	v_or_b32_e32 v26, 0xc0, v3
	v_cndmask_b32_e32 v13, v30, v2, vcc
	buffer_load_dwordx4 v[2:5], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v13, s[0:3], 0 offen
	v_lshlrev_b32_e32 v0, 1, v1
	v_cmp_gt_i32_e32 vcc, s6, v10
	v_add_u32_e32 v29, s16, v1
	v_add_lshl_u32 v1, v12, v58, 1
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s6, v11
	v_lshrrev_b32_e32 v38, 1, v79
	v_and_b32_e32 v36, 56, v38
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[10:13], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v1, s[0:3], 0 offen
	v_lshlrev_b32_e32 v0, 1, v29
	v_cmp_gt_i32_e32 vcc, s6, v18
	v_add_lshl_u32 v1, v20, v58, 1
	v_xor_b32_e32 v36, v36, v58
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s6, v19
	v_mul_lo_u32 v35, s21, v35
	s_mov_b32 s30, s2
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[18:21], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[22:25], v1, s[0:3], 0 offen
	v_add_lshl_u32 v0, v29, s16, 1
	v_cmp_gt_i32_e32 vcc, s6, v26
	v_add_lshl_u32 v1, v28, v58, 1
	s_mov_b32 s31, s3
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s6, v27
	v_add_lshl_u32 v98, v35, v58, 1
	s_lshl_b32 s22, s24, 6
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[26:29], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v1, s[0:3], 0 offen
	v_and_b32_e32 v0, 0x80, v79
	v_lshrrev_b32_e32 v37, 1, v0
	s_and_b32 s0, s21, 0x3fff
	v_xor_b32_e32 v36, v36, v37
	s_bitset1_b32 s0, 14
	v_mul_lo_u32 v1, s21, v34
	v_lshl_add_u32 v36, v36, 1, 0
	v_lshlrev_b32_e32 v34, 8, v34
	s_and_b32 s1, s13, 0xffff
	s_lshl_b32 s19, s0, 16
	v_add_u32_e32 v96, v36, v34
	s_or_b32 s29, s1, s19
	v_add_lshl_u32 v97, v1, v58, 1
	s_barrier
	scratch_store_dword off, v37, off offset:260 ; 4-byte Folded Spill
	s_waitcnt vmcnt(8)
	ds_write_b128 v96, v[2:5]
	s_waitcnt vmcnt(7)
	ds_write_b128 v96, v[6:9] offset:8192
	s_waitcnt vmcnt(6)
	ds_write_b128 v96, v[10:13] offset:16384
	s_waitcnt vmcnt(5)
	ds_write_b128 v96, v[14:17] offset:24576
	s_waitcnt vmcnt(4)
	ds_write_b128 v96, v[18:21] offset:32768
	s_waitcnt vmcnt(3)
	ds_write_b128 v96, v[22:25] offset:40960
	s_waitcnt vmcnt(2)
	ds_write_b128 v96, v[26:29] offset:49152
	s_waitcnt vmcnt(1)
	ds_write_b128 v96, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[2:5], v97, s[28:31], 0 offen
	buffer_load_dwordx4 v[6:9], v98, s[28:31], 0 offen
	v_and_b32_e32 v19, 31, v79
	s_movk_i32 s0, 0xe0
	v_bfe_u32 v1, v79, 5, 1
	v_and_b32_e32 v11, 15, v79
	s_lshl_b32 s30, s21, 6
	v_and_or_b32 v10, v38, s0, v19
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
	s_ashr_i32 s31, s30, 31
	scratch_store_dword off, v38, off offset:264 ; 4-byte Folded Spill
	v_add_u32_e32 v12, v10, v11
	v_lshlrev_b32_e32 v22, 4, v13
	v_lshlrev_b32_e32 v23, 4, v14
	s_lshl_b64 s[6:7], s[30:31], 1
	v_add_u32_e32 v13, v10, v22
	ds_read_b128 v[142:145], v12
	ds_read_b128 v[138:141], v13
	v_add_u32_e32 v12, v10, v23
	v_lshlrev_b32_e32 v24, 4, v15
	v_lshlrev_b32_e32 v59, 4, v16
	s_add_u32 s0, s28, s6
	v_add_u32_e32 v13, v10, v24
	ds_read_b128 v[134:137], v12
	ds_read_b128 v[130:133], v13
	v_add_u32_e32 v12, v10, v59
	v_lshlrev_b32_e32 v76, 4, v17
	v_lshlrev_b32_e32 v77, 4, v18
	s_addc_u32 s23, s13, s7
	v_add_u32_e32 v13, v10, v76
	ds_read_b128 v[126:129], v12
	ds_read_b128 v[88:91], v13
	v_add_u32_e32 v12, v10, v77
	v_lshlrev_b32_e32 v1, 4, v1
	s_and_b32 s1, s23, 0xffff
	v_lshlrev_b32_e32 v78, 8, v19
	v_add_u32_e32 v10, v10, v1
	ds_read_b128 v[84:87], v12
	ds_read_b128 v[80:83], v10
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(2)
	ds_write_b128 v96, v[2:5]
	s_waitcnt vmcnt(1)
	ds_write_b128 v96, v[6:9] offset:8192
	s_or_b32 s1, s1, s19
	v_add3_u32 v92, 0, v11, v78
	buffer_load_dwordx4 v[50:53], v97, s[0:3], 0 offen
	buffer_load_dwordx4 v[54:57], v98, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v19, off offset:268 ; 4-byte Folded Spill
	ds_read_b128 v[18:21], v92
	ds_read_b128 v[60:63], v92 offset:8192
	s_mov_b32 s36, 0
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	v_mov_b64_e32 v[34:35], s[36:37]
	v_mov_b64_e32 v[36:37], s[38:39]
	v_mov_b64_e32 v[38:39], s[40:41]
	v_mov_b64_e32 v[40:41], s[42:43]
	v_mov_b64_e32 v[42:43], s[44:45]
	v_mov_b64_e32 v[44:45], s[46:47]
	v_mov_b64_e32 v[46:47], s[48:49]
	v_mov_b64_e32 v[48:49], s[50:51]
	v_add3_u32 v93, 0, v22, v78
	v_add3_u32 v94, 0, v23, v78
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[142:143], v[34:49]
	v_add3_u32 v95, 0, v24, v78
	v_add3_u32 v59, 0, v59, v78
	s_and_b32 s1, s24, 0x3fff
	s_bitset1_b32 s1, 14
	s_and_b32 s13, s33, 0xffff
	s_lshl_b32 s20, s1, 16
	s_mov_b32 s14, s2
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[144:145], v[2:17]
	ds_read_b128 v[18:21], v93
	ds_read_b128 v[64:67], v93 offset:8192
	s_mov_b32 s15, s3
	s_or_b32 s13, s13, s20
	s_mov_b32 s1, 0xff800000
	s_mov_b32 s16, 0x3e0293ee
	s_add_u32 s0, s0, s6
	v_mov_b32_e32 v123, v97
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[138:139], v[2:17]
	v_mov_b32_e32 v124, v98
	v_mov_b32_e32 v119, 1.0
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[140:141], v[2:17]
	ds_read_b128 v[18:21], v94
	ds_read_b128 v[68:71], v94 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[134:135], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[136:137], v[2:17]
	ds_read_b128 v[18:21], v95
	ds_read_b128 v[72:75], v95 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[130:131], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[132:133], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[60:61], v[142:143], v[34:49]
	s_nop 6
	ds_read_b128 v[34:37], v59
	ds_read_b128 v[38:41], v59 offset:8192
	v_add3_u32 v60, 0, v76, v78
	v_add3_u32 v61, 0, v77, v78
	v_and_b32_e32 v42, 0x100, v79
	v_or_b32_e32 v0, v0, v42
	v_lshrrev_b32_e32 v46, 3, v79
	v_lshrrev_b32_e32 v0, 3, v0
	v_mfma_f32_32x32x8_f16 v[18:33], v[62:63], v[144:145], v[18:33]
	v_add3_u32 v62, 0, v1, v78
	v_and_b32_e32 v1, 16, v79
	v_lshrrev_b32_e32 v47, 3, v1
	v_and_or_b32 v48, v46, 12, v0
	v_or_b32_e32 v0, v48, v47
	v_mad_u64_u32 v[0:1], s[28:29], s24, v0, v[58:59]
	v_mfma_f32_32x32x8_f16 v[18:33], v[64:65], v[138:139], v[18:33]
	s_mov_b32 s28, 0x7060302
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[140:141], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[68:69], v[134:135], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[70:71], v[136:137], v[18:33]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[130:131], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[132:133], v[18:33]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[126:127], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[126:127], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[128:129], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[128:129], v[18:33]
	ds_read_b128 v[34:37], v60
	ds_read_b128 v[38:41], v60 offset:8192
	scratch_store_dwordx4 off, v[88:91], off offset:132 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[88:89], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[88:89], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[90:91], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[90:91], v[18:33]
	ds_read_b128 v[34:37], v61
	ds_read_b128 v[38:41], v61 offset:8192
	scratch_store_dwordx4 off, v[84:87], off offset:116 ; 16-byte Folded Spill
	scratch_store_dword off, v42, off offset:272 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[84:85], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[86:87], v[2:17]
	ds_read_b128 v[34:37], v62
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[84:85], v[18:33]
	v_lshlrev_b32_e32 v84, 1, v0
	v_add_lshl_u32 v85, v0, s24, 1
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_lshlrev_b32_e32 v0, 2, v79
	s_mov_b32 s24, 0x5040100
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[86:87], v[18:33]
	ds_read_b128 v[38:41], v62 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[80:81], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[82:83], v[2:17]
	buffer_load_dwordx4 v[34:37], v84, s[12:15], 0 offen
	buffer_load_dwordx4 v[42:45], v85, s[12:15], 0 offen
	s_mul_i32 s14, s21, 0x180
	scratch_store_dwordx4 off, v[80:83], off offset:100 ; 16-byte Folded Spill
	s_mul_hi_i32 s15, s30, 6
	s_movk_i32 s21, 0xffc0
	s_nop 4
	v_max_f32_e32 v1, v2, v2
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[80:81], v[18:33]
	v_xor_b32_e32 v38, 0x80, v0
	v_max_f32_e32 v0, v3, v3
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v4, v5
	v_max3_f32 v0, v0, v6, v7
	v_max3_f32 v0, v0, v8, v9
	v_max3_f32 v0, v0, v10, v11
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[82:83], v[18:33]
	v_max3_f32 v0, v0, v12, v13
	v_max3_f32 v0, v0, v14, v15
	v_max3_f32 v0, v0, v16, v17
	scratch_store_dword off, v38, off offset:156 ; 4-byte Folded Spill
	s_barrier
	s_nop 5
	v_max3_f32 v0, v0, v18, v19
	v_max3_f32 v0, v0, v20, v21
	v_max3_f32 v0, v0, v22, v23
	v_max3_f32 v0, v0, v24, v25
	v_max3_f32 v0, v0, v26, v27
	v_max3_f32 v0, v0, v28, v29
	v_max3_f32 v0, v0, v30, v31
	v_max3_f32 v0, v0, v32, v33
	ds_bpermute_b32 v1, v38, v0
	v_mov_b32_e32 v226, v33
	s_waitcnt vmcnt(9)
	ds_write_b128 v96, v[50:53]
	s_waitcnt vmcnt(8)
	ds_write_b128 v96, v[54:57] offset:8192
	scratch_store_dword off, v96, off offset:248 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(2)
	v_max3_f32 v227, v0, v1, s1
	v_pk_mul_f32 v[0:1], v[226:227], s[16:17] op_sel_hi:[1,0]
	s_addc_u32 s1, s23, s7
	v_fma_f32 v33, v2, s16, -v1
	v_fma_f32 v3, v3, s16, -v1
	v_fma_f32 v4, v4, s16, -v1
	v_fma_f32 v5, v5, s16, -v1
	v_fma_f32 v6, v6, s16, -v1
	v_fma_f32 v7, v7, s16, -v1
	v_fma_f32 v8, v8, s16, -v1
	v_fma_f32 v9, v9, s16, -v1
	v_fma_f32 v10, v10, s16, -v1
	v_fma_f32 v11, v11, s16, -v1
	v_fma_f32 v12, v12, s16, -v1
	v_fma_f32 v13, v13, s16, -v1
	v_fma_f32 v14, v14, s16, -v1
	v_fma_f32 v15, v15, s16, -v1
	v_fma_f32 v16, v16, s16, -v1
	v_fma_f32 v17, v17, s16, -v1
	v_fma_f32 v18, v18, s16, -v1
	v_fma_f32 v19, v19, s16, -v1
	v_fma_f32 v20, v20, s16, -v1
	v_fma_f32 v21, v21, s16, -v1
	v_fma_f32 v22, v22, s16, -v1
	v_fma_f32 v23, v23, s16, -v1
	v_fma_f32 v24, v24, s16, -v1
	v_fma_f32 v25, v25, s16, -v1
	v_fma_f32 v26, v26, s16, -v1
	v_fma_f32 v27, v27, s16, -v1
	v_fma_f32 v28, v28, s16, -v1
	v_fma_f32 v29, v29, s16, -v1
	v_fma_f32 v30, v30, s16, -v1
	v_fma_f32 v31, v31, s16, -v1
	v_fma_f32 v32, v32, s16, -v1
	v_sub_f32_e32 v2, 0xff800000, v1
	s_ashr_i32 s23, s22, 31
	s_lshl_b64 s[22:23], s[22:23], 1
	s_add_u32 s29, s12, s22
	s_addc_u32 s31, s33, s23
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s1, s19
	buffer_load_dwordx4 v[158:161], v97, s[0:3], 0 offen
	buffer_load_dwordx4 v[154:157], v98, s[0:3], 0 offen
	s_waitcnt vmcnt(5)
	v_perm_b32 v38, v42, v34, s24
	v_perm_b32 v34, v42, v34, s28
	v_sub_f32_e32 v42, v0, v1
	v_bfe_i32 v0, v79, 0, 1
	v_bfe_i32 v1, v79, 1, 1
	v_perm_b32 v39, v43, v35, s24
	v_perm_b32 v35, v43, v35, s28
	v_perm_b32 v41, v45, v37, s24
	v_perm_b32 v37, v45, v37, s28
	v_bfe_i32 v43, v79, 2, 1
	v_and_b32_e32 v45, 0x220, v0
	v_and_b32_e32 v49, 0x404, v1
	v_perm_b32 v40, v44, v36, s24
	v_perm_b32 v36, v44, v36, s28
	v_bfe_i32 v44, v79, 3, 1
	v_or_b32_e32 v50, v45, v49
	v_and_b32_e32 v51, 0x808, v43
	v_or_b32_e32 v52, v50, v51
	v_and_b32_e32 v53, 0x1010, v44
	v_or3_b32 v54, v53, v47, v52
	v_xor_b32_e32 v54, v48, v54
	v_lshl_add_u32 v54, v54, 1, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v54, off offset:160 ; 4-byte Folded Spill
	ds_write_b32 v54, v38 offset:16384
	scratch_store_dword off, v92, off offset:208 ; 4-byte Folded Spill
	ds_read_b128 v[222:225], v92
	ds_read_b128 v[190:193], v92 offset:8192
	ds_read_b128 v[218:221], v93
	scratch_store_dword off, v93, off offset:212 ; 4-byte Folded Spill
	ds_read_b128 v[186:189], v93 offset:8192
	ds_read_b128 v[214:217], v94
	scratch_store_dword off, v94, off offset:216 ; 4-byte Folded Spill
	ds_read_b128 v[182:185], v94 offset:8192
	ds_read_b128 v[210:213], v95
	scratch_store_dword off, v95, off offset:220 ; 4-byte Folded Spill
	ds_read_b128 v[178:181], v95 offset:8192
	ds_read_b128 v[206:209], v59
	scratch_store_dword off, v59, off offset:224 ; 4-byte Folded Spill
	ds_read_b128 v[174:177], v59 offset:8192
	ds_read_b128 v[202:205], v60
	scratch_store_dword off, v60, off offset:228 ; 4-byte Folded Spill
	s_and_b32 s0, s31, 0xffff
	ds_read_b128 v[170:173], v60 offset:8192
	ds_read_b128 v[198:201], v61
	scratch_store_dword off, v61, off offset:232 ; 4-byte Folded Spill
	s_or_b32 s1, s0, s20
	s_mov_b32 s0, s29
	ds_read_b128 v[166:169], v61 offset:8192
	ds_read_b128 v[194:197], v62
	buffer_load_dwordx4 v[150:153], v85, s[0:3], 0 offen
	buffer_load_dwordx4 v[146:149], v84, s[0:3], 0 offen
	scratch_load_dword v122, off, off offset:156 ; 4-byte Folded Reload
	v_or_b32_e32 v38, 0x44, v45
	v_xor_b32_e32 v38, v38, v49
	v_or_b32_e32 v54, v53, v51
	v_or3_b32 v38, v47, v38, v54
	v_xor_b32_e32 v38, v48, v38
	v_lshl_add_u32 v38, v38, 1, 0
	ds_read_b128 v[162:165], v62 offset:8192
	ds_write_b32 v38, v34 offset:16384
	v_or_b32_e32 v34, 0x88, v50
	v_xor_b32_e32 v34, v34, v51
	v_or3_b32 v34, v47, v34, v53
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v38, off offset:164 ; 4-byte Folded Spill
	scratch_store_dword off, v34, off offset:168 ; 4-byte Folded Spill
	ds_write_b32 v34, v39 offset:16384
	v_or_b32_e32 v34, 0xcc, v45
	v_or_b32_e32 v38, v51, v49
	v_xor_b32_e32 v34, v38, v34
	v_or3_b32 v34, v47, v34, v53
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:172 ; 4-byte Folded Spill
	ds_write_b32 v34, v35 offset:16384
	v_or_b32_e32 v34, 0x110, v52
	v_xor_b32_e32 v34, v34, v53
	v_or_b32_e32 v34, v34, v47
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:176 ; 4-byte Folded Spill
	ds_write_b32 v34, v40 offset:16384
	v_or_b32_e32 v34, 0x154, v45
	v_xor_b32_e32 v34, v34, v49
	v_or_b32_e32 v34, v34, v51
	v_xor_b32_e32 v34, v34, v53
	v_or_b32_e32 v34, v34, v47
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:180 ; 4-byte Folded Spill
	ds_write_b32 v34, v36 offset:16384
	v_or_b32_e32 v34, 0x198, v50
	v_xor_b32_e32 v34, v54, v34
	v_or_b32_e32 v34, v34, v47
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:184 ; 4-byte Folded Spill
	ds_write_b32 v34, v41 offset:16384
	v_or_b32_e32 v34, v38, v53
	v_or_b32_e32 v35, 0x1dc, v45
	v_xor_b32_e32 v34, v34, v35
	v_or_b32_e32 v34, v34, v47
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:188 ; 4-byte Folded Spill
	ds_write_b32 v34, v37 offset:16384
	v_and_b32_e32 v0, 0x44, v0
	v_and_b32_e32 v1, 0x88, v1
	v_and_b32_e32 v34, 0x110, v43
	v_or_b32_e32 v35, 24, v0
	v_or_b32_e32 v36, v34, v1
	v_or_b32_e32 v37, 0x818, v0
	v_or_b32_e32 v38, 0x1018, v0
	v_or_b32_e32 v39, 0x1818, v0
	v_bfe_i32 v40, v79, 4, 1
	v_or_b32_e32 v43, v0, v1
	v_xor_b32_e32 v35, v36, v35
	v_xor_b32_e32 v37, v36, v37
	v_xor_b32_e32 v38, v36, v38
	v_xor_b32_e32 v36, v36, v39
	v_and_b32_e32 v56, 4, v46
	v_and_b32_e32 v39, 0x220, v44
	v_and_b32_e32 v40, 0x404, v40
	v_or_b32_e32 v41, 8, v0
	v_or_b32_e32 v44, 16, v43
	v_or_b32_e32 v45, 0x808, v0
	v_or_b32_e32 v46, 0x810, v43
	v_or_b32_e32 v47, 0x1010, v43
	v_or_b32_e32 v48, 0x1008, v0
	v_or_b32_e32 v49, 0x1810, v43
	v_or_b32_e32 v50, 0x1808, v0
	v_or_b32_e32 v51, v43, v34
	v_xor_b32_e32 v41, v41, v1
	v_xor_b32_e32 v44, v44, v34
	v_xor_b32_e32 v45, v45, v1
	v_xor_b32_e32 v46, v46, v34
	v_xor_b32_e32 v47, v47, v34
	v_xor_b32_e32 v48, v48, v1
	v_xor_b32_e32 v49, v49, v34
	v_xor_b32_e32 v50, v50, v1
	v_or_b32_e32 v52, v51, v39
	v_xor_b32_e32 v53, v40, v56
	v_or3_b32 v41, v34, v41, v39
	v_or_b32_e32 v44, v44, v39
	v_or_b32_e32 v35, v35, v39
	v_or3_b32 v45, v34, v45, v39
	v_or_b32_e32 v37, v37, v39
	v_or_b32_e32 v46, v46, v39
	v_or_b32_e32 v47, v47, v39
	v_or_b32_e32 v38, v38, v39
	v_or3_b32 v48, v34, v48, v39
	v_or_b32_e32 v36, v36, v39
	v_or_b32_e32 v49, v49, v39
	v_or3_b32 v50, v34, v50, v39
	v_or_b32_e32 v39, v53, v39
	v_xor_b32_e32 v53, v53, v52
	v_or_b32_e32 v54, 0x800, v52
	v_or_b32_e32 v55, 0x1000, v52
	v_or_b32_e32 v52, 0x1800, v52
	v_xor_b32_e32 v41, v56, v41
	v_xor_b32_e32 v44, v56, v44
	v_xor_b32_e32 v35, v56, v35
	v_xor_b32_e32 v45, v56, v45
	v_xor_b32_e32 v54, v56, v54
	v_xor_b32_e32 v37, v56, v37
	v_xor_b32_e32 v46, v56, v46
	v_xor_b32_e32 v47, v56, v47
	v_xor_b32_e32 v38, v56, v38
	v_xor_b32_e32 v55, v56, v55
	v_xor_b32_e32 v48, v56, v48
	v_xor_b32_e32 v36, v56, v36
	v_xor_b32_e32 v49, v56, v49
	v_xor_b32_e32 v50, v56, v50
	v_xor_b32_e32 v52, v56, v52
	v_xor_b32_e32 v41, v41, v40
	v_xor_b32_e32 v44, v44, v40
	v_xor_b32_e32 v35, v35, v40
	v_xor_b32_e32 v45, v45, v40
	v_xor_b32_e32 v54, v54, v40
	v_xor_b32_e32 v37, v37, v40
	v_xor_b32_e32 v46, v46, v40
	v_xor_b32_e32 v47, v47, v40
	v_xor_b32_e32 v38, v38, v40
	v_xor_b32_e32 v55, v55, v40
	v_xor_b32_e32 v48, v48, v40
	v_xor_b32_e32 v36, v36, v40
	v_xor_b32_e32 v49, v49, v40
	v_xor_b32_e32 v50, v50, v40
	v_xor_b32_e32 v40, v52, v40
	v_or_b32_e32 v52, v39, v34
	scratch_store_dword off, v56, off offset:256 ; 4-byte Folded Spill
	v_or_b32_e32 v56, 56, v0
	v_or_b32_e32 v57, v52, v1
	v_or_b32_e32 v58, 0x838, v0
	v_or_b32_e32 v59, 0x1038, v0
	v_or_b32_e32 v60, 0x1838, v0
	scratch_store_dword off, v62, off offset:236 ; 4-byte Folded Spill
	v_xor_b32_e32 v56, v57, v56
	v_xor_b32_e32 v58, v57, v58
	v_xor_b32_e32 v59, v57, v59
	v_xor_b32_e32 v57, v57, v60
	v_or_b32_e32 v60, 48, v43
	v_or_b32_e32 v61, 0x830, v43
	v_or_b32_e32 v62, 0x1030, v43
	v_or_b32_e32 v43, 0x1830, v43
	v_xor_b32_e32 v60, v52, v60
	v_xor_b32_e32 v61, v52, v61
	v_xor_b32_e32 v62, v52, v62
	v_xor_b32_e32 v43, v52, v43
	v_or_b32_e32 v52, 40, v0
	v_or_b32_e32 v63, 0x828, v0
	v_or_b32_e32 v64, 0x1028, v0
	v_or_b32_e32 v0, 0x1828, v0
	v_xor_b32_e32 v52, v52, v1
	v_xor_b32_e32 v63, v63, v1
	v_xor_b32_e32 v64, v64, v1
	v_xor_b32_e32 v0, v0, v1
	v_exp_f32_e32 v116, v3
	v_lshl_add_u32 v3, v41, 1, 0
	v_or_b32_e32 v1, v52, v34
	v_or_b32_e32 v52, v63, v34
	v_or_b32_e32 v63, v64, v34
	v_or_b32_e32 v0, v0, v34
	v_or_b32_e32 v34, 32, v51
	scratch_store_dword off, v3, off offset:148 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v44, 1, 0
	v_xor_b32_e32 v34, v39, v34
	scratch_store_dword off, v3, off offset:152 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v35, 1, 0
	v_xor_b32_e32 v64, v39, v1
	scratch_store_dword off, v3, off        ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v34, 1, 0
	scratch_store_dword off, v3, off offset:4 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v64, 1, 0
	scratch_store_dword off, v3, off offset:8 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v60, 1, 0
	scratch_store_dword off, v3, off offset:12 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v56, 1, 0
	scratch_store_dword off, v3, off offset:16 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v45, 1, 0
	scratch_store_dword off, v3, off offset:20 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v54, 1, 0
	scratch_store_dword off, v3, off offset:24 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v37, 1, 0
	v_xor_b32_e32 v52, v39, v52
	v_or_b32_e32 v1, 0x820, v51
	scratch_store_dword off, v3, off offset:28 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v46, 1, 0
	v_xor_b32_e32 v65, v39, v1
	scratch_store_dword off, v3, off offset:32 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v52, 1, 0
	scratch_store_dword off, v3, off offset:36 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v65, 1, 0
	scratch_store_dword off, v3, off offset:40 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v58, 1, 0
	scratch_store_dword off, v3, off offset:44 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v61, 1, 0
	scratch_store_dword off, v3, off offset:48 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v47, 1, 0
	scratch_store_dword off, v3, off offset:52 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v38, 1, 0
	scratch_store_dword off, v3, off offset:56 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v55, 1, 0
	scratch_store_dword off, v3, off offset:60 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v48, 1, 0
	v_or_b32_e32 v1, 0x1020, v51
	scratch_store_dword off, v3, off offset:64 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v62, 1, 0
	v_xor_b32_e32 v66, v39, v1
	scratch_store_dword off, v3, off offset:68 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v59, 1, 0
	v_xor_b32_e32 v63, v39, v63
	scratch_store_dword off, v3, off offset:72 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v66, 1, 0
	scratch_store_dword off, v3, off offset:76 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v63, 1, 0
	scratch_store_dword off, v3, off offset:80 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v36, 1, 0
	scratch_store_dword off, v3, off offset:84 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v49, 1, 0
	s_add_u32 s12, s52, s54
	scratch_store_dword off, v3, off offset:88 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v50, 1, 0
	s_addc_u32 s13, s53, s55
	v_xor_b32_e32 v67, v39, v0
	v_or_b32_e32 v0, 0x1820, v51
	scratch_store_dword off, v3, off offset:192 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v40, 1, 0
	s_lshl_b64 s[12:13], s[12:13], 1
	v_xor_b32_e32 v39, v39, v0
	v_exp_f32_e32 v117, v33
	v_exp_f32_e32 v115, v4
	v_exp_f32_e32 v233, v5
	v_exp_f32_e32 v232, v6
	v_exp_f32_e32 v114, v7
	v_exp_f32_e32 v239, v8
	v_exp_f32_e32 v236, v9
	v_exp_f32_e32 v235, v10
	v_exp_f32_e32 v234, v11
	v_exp_f32_e32 v230, v12
	v_exp_f32_e32 v238, v13
	v_exp_f32_e32 v231, v14
	v_exp_f32_e32 v1, v15
	v_exp_f32_e32 v0, v16
	v_exp_f32_e32 v241, v17
	v_exp_f32_e32 v240, v18
	v_exp_f32_e32 v251, v19
	v_exp_f32_e32 v250, v20
	v_exp_f32_e32 v237, v21
	v_exp_f32_e32 v255, v22
	v_exp_f32_e32 v254, v23
	v_exp_f32_e32 v253, v24
	v_exp_f32_e32 v252, v25
	v_exp_f32_e32 v246, v26
	v_exp_f32_e32 v245, v27
	v_exp_f32_e32 v249, v28
	v_exp_f32_e32 v248, v29
	v_exp_f32_e32 v247, v30
	v_exp_f32_e32 v244, v31
	v_exp_f32_e32 v243, v32
	v_exp_f32_e32 v242, v42
	scratch_store_dword off, v3, off offset:196 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v57, 1, 0
	s_add_u32 s12, s14, s12
	v_exp_f32_e32 v228, v2
	scratch_store_dword off, v3, off offset:200 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v43, 1, 0
	s_addc_u32 s13, s15, s13
	scratch_store_dword off, v3, off offset:92 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v67, 1, 0
	s_add_u32 s4, s4, s12
	v_mov_b32_e32 v18, 0
	scratch_store_dword off, v79, off offset:252 ; 4-byte Folded Spill
	v_lshl_add_u32 v125, v53, 1, 0
	scratch_store_dword off, v3, off offset:96 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v39, 1, 0
	s_addc_u32 s5, s5, s13
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
	v_mov_b32_e32 v66, v18
	v_mov_b32_e32 v67, v18
	v_mov_b32_e32 v68, v18
	v_mov_b32_e32 v69, v18
	v_mov_b32_e32 v70, v18
	v_mov_b32_e32 v71, v18
	v_mov_b32_e32 v72, v18
	v_mov_b32_e32 v73, v18
	v_mov_b32_e32 v74, v18
	v_mov_b32_e32 v75, v18
	v_mov_b32_e32 v76, v18
	v_mov_b32_e32 v77, v18
	v_mov_b32_e32 v78, v18
	v_mov_b32_e32 v79, v18
	v_mov_b32_e32 v80, v18
	v_mov_b32_e32 v81, v18
	s_waitcnt vmcnt(42)
	v_lshrrev_b32_e32 v118, 16, v150
	scratch_store_dword off, v3, off offset:204 ; 4-byte Folded Spill
	scratch_store_dword off, v85, off offset:244 ; 4-byte Folded Spill
	scratch_store_dword off, v84, off offset:240 ; 4-byte Folded Spill
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b64_e32 v[112:113], s[50:51]
	v_mov_b64_e32 v[110:111], s[48:49]
	v_mov_b64_e32 v[108:109], s[46:47]
	v_mov_b64_e32 v[106:107], s[44:45]
	v_mov_b64_e32 v[104:105], s[42:43]
	v_mov_b64_e32 v[102:103], s[40:41]
	v_mov_b64_e32 v[100:101], s[38:39]
	v_mov_b64_e32 v[98:99], s[36:37]
	v_mov_b32_e32 v120, v119
	v_mov_b32_e32 v226, v227
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[2:17], v[222:223], v[142:143], v[98:113]
	v_pk_mul_f32 v[66:67], v[66:67], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[224:225], v[144:145], v[2:17]
	v_pk_mul_f32 v[80:81], v[80:81], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[218:219], v[138:139], v[2:17]
	v_pk_mul_f32 v[62:63], v[62:63], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[220:221], v[140:141], v[2:17]
	v_pk_mul_f32 v[44:45], v[44:45], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[214:215], v[134:135], v[2:17]
	v_pk_mul_f32 v[26:27], v[26:27], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[228:229] op_sel_hi:[1,0]
	v_cvt_f16_f32_e32 v214, v231
	v_cvt_f16_f32_e32 v215, v1
	v_cvt_f16_f32_e32 v218, v240
	v_mfma_f32_32x32x8_f16 v[2:17], v[216:217], v[136:137], v[2:17]
	v_cvt_f16_f32_e32 v216, v0
	v_cvt_f16_f32_e32 v217, v241
	v_cvt_f16_f32_e32 v219, v251
	v_cvt_f16_f32_e32 v220, v250
	v_cvt_f16_f32_e32 v221, v237
	v_cvt_f16_f32_e32 v222, v255
	v_cvt_f16_f32_e32 v223, v254
	v_mfma_f32_32x32x8_f16 v[2:17], v[210:211], v[130:131], v[2:17]
	v_cvt_f16_f32_e32 v210, v235
	v_cvt_f16_f32_e32 v211, v234
	v_cvt_f16_f32_e32 v224, v253
	v_cvt_f16_f32_e32 v225, v252
	v_cvt_f16_f32_e32 v227, v246
	v_cvt_f16_f32_e32 v229, v249
	v_mfma_f32_32x32x8_f16 v[2:17], v[212:213], v[132:133], v[2:17]
	v_cvt_f16_f32_e32 v212, v230
	v_cvt_f16_f32_e32 v213, v238
	v_mfma_f32_32x32x8_f16 v[2:17], v[206:207], v[126:127], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[208:209], v[128:129], v[2:17]
	scratch_load_dwordx4 v[206:209], off, off offset:132 ; 16-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[190:191], v[142:143], v[98:113]
	v_mfma_f32_32x32x8_f16 v[82:97], v[192:193], v[144:145], v[82:97]
	s_nop 5
	v_add_f32_e32 v98, v117, v116
	v_add_f32_e32 v98, v98, v115
	v_add_f32_e32 v98, v98, v233
	v_add_f32_e32 v98, v98, v232
	v_add_f32_e32 v98, v98, v114
	v_add_f32_e32 v98, v98, v239
	v_add_f32_e32 v98, v98, v236
	v_mfma_f32_32x32x8_f16 v[82:97], v[186:187], v[138:139], v[82:97]
	v_add_f32_e32 v98, v98, v235
	v_add_f32_e32 v98, v98, v234
	v_add_f32_e32 v98, v98, v230
	v_add_f32_e32 v98, v98, v238
	v_add_f32_e32 v98, v98, v231
	v_add_f32_e32 v98, v98, v1
	v_add_f32_e32 v98, v98, v0
	v_mfma_f32_32x32x8_f16 v[82:97], v[188:189], v[140:141], v[82:97]
	v_add_f32_e32 v98, v98, v241
	v_add_f32_e32 v98, v98, v240
	v_add_f32_e32 v98, v98, v251
	v_add_f32_e32 v98, v98, v250
	v_add_f32_e32 v98, v98, v237
	v_add_f32_e32 v98, v98, v255
	v_add_f32_e32 v98, v98, v254
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[82:97], v[182:183], v[134:135], v[82:97]
	v_add_f32_e32 v98, v98, v253
	v_add_f32_e32 v98, v98, v252
	v_add_f32_e32 v98, v98, v246
	v_add_f32_e32 v98, v98, v245
	v_add_f32_e32 v98, v98, v249
	v_add_f32_e32 v98, v98, v248
	v_add_f32_e32 v98, v98, v247
	v_mfma_f32_32x32x8_f16 v[82:97], v[184:185], v[136:137], v[82:97]
	v_add_f32_e32 v98, v98, v244
	v_add_f32_e32 v98, v98, v243
	v_add_f32_e32 v98, v98, v242
	s_waitcnt vmcnt(44)
	ds_bpermute_b32 v99, v122, v98
	v_cvt_f16_f32_e32 v230, v248
	v_cvt_f16_f32_e32 v231, v247
	v_cvt_f16_f32_e32 v234, v242
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[82:97], v[178:179], v[130:131], v[82:97]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v119, v98, v99
	v_fmac_f32_e32 v119, v120, v228
	v_cvt_f16_f32_e32 v178, v117
	v_cvt_f16_f32_e32 v179, v115
	v_cvt_f16_f32_e32 v228, v245
	v_mfma_f32_32x32x8_f16 v[82:97], v[180:181], v[132:133], v[82:97]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[202:203], v[206:207], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[204:205], v[208:209], v[2:17]
	scratch_load_dwordx4 v[202:205], off, off offset:116 ; 16-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[126:127], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[128:129], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[206:207], v[82:97]
	v_cvt_f16_f32_e32 v206, v232
	v_cvt_f16_f32_e32 v207, v114
	v_cvt_f16_f32_e32 v232, v244
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[208:209], v[82:97]
	v_cvt_f16_f32_e32 v208, v239
	v_cvt_f16_f32_e32 v209, v236
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[198:199], v[202:203], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[200:201], v[204:205], v[2:17]
	scratch_load_dwordx4 v[198:201], off, off offset:100 ; 16-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[202:203], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[204:205], v[82:97]
	v_cvt_f16_f32_e32 v204, v116
	v_cvt_f16_f32_e32 v205, v233
	v_cvt_f16_f32_e32 v233, v243
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[194:195], v[198:199], v[2:17]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[198:199], v[82:97]
	v_mfma_f32_32x32x8_f16 v[2:17], v[196:197], v[200:201], v[2:17]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[200:201], v[82:97]
	; sched_barrier mask(0x00000000)
	s_barrier
	scratch_load_dword v98, off, off offset:148 ; 4-byte Folded Reload
	ds_read_b64 v[0:1], v125 offset:16384
	s_add_u32 s12, s29, s22
	s_addc_u32 s30, s31, s23
	s_and_b32 s0, s5, 0xffff
	s_or_b32 s1, s0, s19
	s_mov_b32 s0, s4
	scratch_load_dword v235, off, off offset:248 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	ds_read_b64 v[114:115], v98 offset:16384
	scratch_load_dword v98, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[116:117], v98 offset:16384
	scratch_load_dword v98, off, off        ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[120:121], v98 offset:16384
	scratch_load_dword v98, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[196:197], v98 offset:16384
	scratch_load_dword v98, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[198:199], v98 offset:16384
	scratch_load_dword v98, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[200:201], v98 offset:16384
	scratch_load_dword v98, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[202:203], v98 offset:16384
	scratch_load_dword v98, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[194:195], v98 offset:16384
	scratch_load_dword v98, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[192:193], v98 offset:16384
	scratch_load_dword v98, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[188:189], v98 offset:16384
	scratch_load_dword v98, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[190:191], v98 offset:16384
	scratch_load_dword v98, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[186:187], v98 offset:16384
	scratch_load_dword v98, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[184:185], v98 offset:16384
	scratch_load_dword v98, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[182:183], v98 offset:16384
	scratch_load_dword v98, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[180:181], v98 offset:16384
	scratch_load_dword v98, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[176:177], v98 offset:16384
	scratch_load_dword v98, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[172:173], v98 offset:16384
	scratch_load_dword v98, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[174:175], v98 offset:16384
	scratch_load_dword v98, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v98 offset:16384
	scratch_load_dword v98, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[168:169], v98 offset:16384
	scratch_load_dword v98, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[166:167], v98 offset:16384
	scratch_load_dword v98, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[164:165], v98 offset:16384
	scratch_load_dword v98, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[162:163], v98 offset:16384
	scratch_load_dword v98, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[110:111], v98 offset:16384
	scratch_load_dword v98, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[112:113], v98 offset:16384
	scratch_load_dword v98, off, off offset:192 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[108:109], v98 offset:16384
	scratch_load_dword v98, off, off offset:196 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[106:107], v98 offset:16384
	scratch_load_dword v98, off, off offset:200 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[104:105], v98 offset:16384
	scratch_load_dword v98, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[102:103], v98 offset:16384
	scratch_load_dword v98, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[100:101], v98 offset:16384
	scratch_load_dword v98, off, off offset:204 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[98:99], v98 offset:16384
	ds_write_b128 v235, v[158:161]
	ds_write_b128 v235, v[154:157] offset:8192
	buffer_load_dwordx4 v[158:161], v123, s[0:3], 0 offen
	buffer_load_dwordx4 v[154:157], v124, s[0:3], 0 offen
	; sched_barrier mask(0x00000000)
	v_pack_b32_f16 v179, v179, v205
	v_pack_b32_f16 v178, v178, v204
	v_pack_b32_f16 v205, v216, v217
	v_pack_b32_f16 v204, v214, v215
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[66:81], v[0:1], v[178:179], v[66:81]
	v_pack_b32_f16 v1, v208, v209
	v_pack_b32_f16 v0, v206, v207
	v_pack_b32_f16 v207, v224, v225
	v_pack_b32_f16 v206, v222, v223
	v_mfma_f32_32x32x8_f16 v[50:65], v[194:195], v[178:179], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[176:177], v[178:179], v[34:49]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_f16 v[18:33], v[110:111], v[178:179], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[0:1], v[66:81]
	v_pack_b32_f16 v115, v212, v213
	v_pack_b32_f16 v114, v210, v211
	v_mfma_f32_32x32x8_f16 v[50:65], v[192:193], v[0:1], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[0:1], v[34:49]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_f16 v[18:33], v[112:113], v[0:1], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[114:115], v[66:81]
	v_max_f32_e32 v116, v3, v3
	v_max_f32_e32 v117, v2, v2
	v_max_f32_e32 v116, v117, v116
	v_max3_f32 v116, v116, v4, v5
	v_max3_f32 v116, v116, v6, v7
	v_max3_f32 v116, v116, v8, v9
	v_max3_f32 v116, v116, v10, v11
	v_mfma_f32_32x32x8_f16 v[50:65], v[188:189], v[114:115], v[50:65]
	v_max3_f32 v116, v116, v12, v13
	v_max3_f32 v116, v116, v14, v15
	v_max3_f32 v116, v116, v16, v17
	v_max3_f32 v116, v116, v82, v83
	v_max3_f32 v116, v116, v84, v85
	v_max3_f32 v116, v116, v86, v87
	v_max3_f32 v116, v116, v88, v89
	v_mfma_f32_32x32x8_f16 v[34:49], v[174:175], v[114:115], v[34:49]
	v_max3_f32 v116, v116, v90, v91
	v_max3_f32 v116, v116, v92, v93
	v_max3_f32 v116, v116, v94, v95
	v_max3_f32 v116, v116, v96, v97
	ds_bpermute_b32 v117, v122, v116
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[114:115], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[204:205], v[66:81]
	v_pack_b32_f16 v121, v220, v221
	v_pack_b32_f16 v120, v218, v219
	v_mfma_f32_32x32x8_f16 v[50:65], v[190:191], v[204:205], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[204:205], v[34:49]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 v[18:33], v[106:107], v[204:205], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[196:197], v[120:121], v[66:81]
	v_pack_b32_f16 v197, v233, v234
	v_pack_b32_f16 v196, v231, v232
	v_mfma_f32_32x32x8_f16 v[50:65], v[186:187], v[120:121], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[168:169], v[120:121], v[34:49]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[18:33], v[104:105], v[120:121], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[198:199], v[206:207], v[66:81]
	v_pack_b32_f16 v199, v229, v230
	v_pack_b32_f16 v198, v227, v228
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v227, v226, v116, v117
	v_pk_mul_f32 v[228:229], v[226:227], s[16:17] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v2, v2, s16, -v229
	v_fma_f32 v3, v3, s16, -v229
	v_mfma_f32_32x32x8_f16 v[50:65], v[184:185], v[206:207], v[50:65]
	v_fma_f32 v4, v4, s16, -v229
	v_fma_f32 v5, v5, s16, -v229
	v_fma_f32 v6, v6, s16, -v229
	v_fma_f32 v7, v7, s16, -v229
	v_fma_f32 v8, v8, s16, -v229
	v_fma_f32 v9, v9, s16, -v229
	v_fma_f32 v10, v10, s16, -v229
	v_mfma_f32_32x32x8_f16 v[34:49], v[166:167], v[206:207], v[34:49]
	v_fma_f32 v11, v11, s16, -v229
	v_fma_f32 v12, v12, s16, -v229
	v_fma_f32 v13, v13, s16, -v229
	v_fma_f32 v14, v14, s16, -v229
	v_fma_f32 v15, v15, s16, -v229
	v_fma_f32 v16, v16, s16, -v229
	v_fma_f32 v17, v17, s16, -v229
	v_mfma_f32_32x32x8_f16 v[18:33], v[102:103], v[206:207], v[18:33]
	v_fma_f32 v82, v82, s16, -v229
	v_fma_f32 v83, v83, s16, -v229
	v_fma_f32 v84, v84, s16, -v229
	v_fma_f32 v85, v85, s16, -v229
	v_fma_f32 v86, v86, s16, -v229
	v_fma_f32 v87, v87, s16, -v229
	v_fma_f32 v88, v88, s16, -v229
	v_mfma_f32_32x32x8_f16 v[66:81], v[200:201], v[198:199], v[66:81]
	v_fma_f32 v89, v89, s16, -v229
	v_fma_f32 v90, v90, s16, -v229
	v_fma_f32 v91, v91, s16, -v229
	v_fma_f32 v92, v92, s16, -v229
	v_fma_f32 v93, v93, s16, -v229
	v_fma_f32 v94, v94, s16, -v229
	v_fma_f32 v95, v95, s16, -v229
	v_mfma_f32_32x32x8_f16 v[50:65], v[182:183], v[198:199], v[50:65]
	v_fma_f32 v96, v96, s16, -v229
	v_fma_f32 v97, v97, s16, -v229
	v_exp_f32_e32 v117, v2
	v_sub_f32_e32 v2, v228, v229
	v_exp_f32_e32 v116, v3
	v_exp_f32_e32 v115, v4
	v_exp_f32_e32 v233, v5
	v_mfma_f32_32x32x8_f16 v[34:49], v[164:165], v[198:199], v[34:49]
	v_exp_f32_e32 v232, v6
	v_exp_f32_e32 v114, v7
	v_exp_f32_e32 v239, v8
	v_exp_f32_e32 v236, v9
	v_exp_f32_e32 v235, v10
	v_exp_f32_e32 v234, v11
	v_exp_f32_e32 v230, v12
	v_mfma_f32_32x32x8_f16 v[18:33], v[100:101], v[198:199], v[18:33]
	v_exp_f32_e32 v238, v13
	v_exp_f32_e32 v231, v14
	v_exp_f32_e32 v1, v15
	v_exp_f32_e32 v0, v16
	v_exp_f32_e32 v241, v17
	v_exp_f32_e32 v240, v82
	v_exp_f32_e32 v251, v83
	v_mfma_f32_32x32x8_f16 v[66:81], v[202:203], v[196:197], v[66:81]
	v_exp_f32_e32 v250, v84
	v_exp_f32_e32 v237, v85
	v_exp_f32_e32 v255, v86
	v_exp_f32_e32 v254, v87
	v_exp_f32_e32 v253, v88
	v_exp_f32_e32 v252, v89
	v_exp_f32_e32 v246, v90
	v_mfma_f32_32x32x8_f16 v[50:65], v[180:181], v[196:197], v[50:65]
	v_exp_f32_e32 v245, v91
	v_exp_f32_e32 v249, v92
	v_exp_f32_e32 v248, v93
	v_exp_f32_e32 v247, v94
	v_exp_f32_e32 v244, v95
	v_exp_f32_e32 v243, v96
	v_exp_f32_e32 v242, v97
	v_mfma_f32_32x32x8_f16 v[34:49], v[162:163], v[196:197], v[34:49]
	v_exp_f32_e32 v228, v2
	v_mfma_f32_32x32x8_f16 v[18:33], v[98:99], v[196:197], v[18:33]
	; sched_barrier mask(0x00000000)
	s_barrier
	scratch_load_dword v2, off, off offset:208 ; 4-byte Folded Reload
	scratch_load_dword v3, off, off offset:212 ; 4-byte Folded Reload
	scratch_load_dword v4, off, off offset:216 ; 4-byte Folded Reload
	scratch_load_dword v5, off, off offset:220 ; 4-byte Folded Reload
	scratch_load_dword v6, off, off offset:224 ; 4-byte Folded Reload
	scratch_load_dword v7, off, off offset:228 ; 4-byte Folded Reload
	scratch_load_dword v8, off, off offset:232 ; 4-byte Folded Reload
	scratch_load_dword v9, off, off offset:236 ; 4-byte Folded Reload
	s_and_b32 s0, s30, 0xffff
	s_or_b32 s13, s0, s20
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_add_u32 s29, s29, s22
	s_addc_u32 s31, s31, s23
	s_add_u32 s4, s4, s6
	s_addc_u32 s5, s5, s7
	s_add_i32 s21, s21, 64
	s_cmpk_lt_u32 s21, 0x1f00
	s_waitcnt vmcnt(7)
	ds_read_b128 v[222:225], v2
	s_waitcnt vmcnt(6)
	ds_read_b128 v[218:221], v3
	s_waitcnt vmcnt(5)
	ds_read_b128 v[214:217], v4
	s_waitcnt vmcnt(4)
	ds_read_b128 v[210:213], v5
	s_waitcnt vmcnt(3)
	ds_read_b128 v[206:209], v6
	s_waitcnt vmcnt(2)
	ds_read_b128 v[202:205], v7
	s_waitcnt vmcnt(1)
	ds_read_b128 v[198:201], v8
	s_waitcnt vmcnt(0)
	ds_read_b128 v[194:197], v9
	ds_read_b128 v[190:193], v2 offset:8192
	ds_read_b128 v[186:189], v3 offset:8192
	ds_read_b128 v[182:185], v4 offset:8192
	ds_read_b128 v[178:181], v5 offset:8192
	ds_read_b128 v[174:177], v6 offset:8192
	ds_read_b128 v[170:173], v7 offset:8192
	ds_read_b128 v[166:169], v8 offset:8192
	ds_read_b128 v[162:165], v9 offset:8192
	scratch_load_dword v3, off, off offset:160 ; 4-byte Folded Reload
	v_perm_b32 v2, v150, v146, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:164 ; 4-byte Folded Reload
	v_alignbit_b32 v2, v118, v146, 16
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:168 ; 4-byte Folded Reload
	v_perm_b32 v2, v151, v147, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:172 ; 4-byte Folded Reload
	v_perm_b32 v2, v151, v147, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:176 ; 4-byte Folded Reload
	v_perm_b32 v2, v152, v148, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:180 ; 4-byte Folded Reload
	v_perm_b32 v2, v152, v148, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:184 ; 4-byte Folded Reload
	v_perm_b32 v2, v153, v149, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:188 ; 4-byte Folded Reload
	v_perm_b32 v2, v153, v149, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v2, off, off offset:240 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[146:149], v2, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v2, off, off offset:244 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[150:153], v2, s[12:15], 0 offen
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v118, 16, v150
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	scratch_load_dword v2, off, off offset:272 ; 4-byte Folded Reload
	s_mov_b32 s36, 0
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	v_add_f32_e32 v118, v117, v116
	v_add_f32_e32 v118, v118, v115
	v_add_f32_e32 v118, v118, v233
	v_add_f32_e32 v118, v118, v232
	v_add_f32_e32 v118, v118, v114
	v_add_f32_e32 v118, v118, v239
	v_add_f32_e32 v118, v118, v236
	v_add_f32_e32 v118, v118, v235
	v_add_f32_e32 v118, v118, v234
	v_add_f32_e32 v118, v118, v230
	v_add_f32_e32 v118, v118, v238
	v_add_f32_e32 v118, v118, v231
	v_add_f32_e32 v118, v118, v1
	v_add_f32_e32 v118, v118, v0
	v_add_f32_e32 v118, v118, v241
	v_add_f32_e32 v118, v118, v240
	v_add_f32_e32 v118, v118, v251
	v_add_f32_e32 v118, v118, v250
	v_add_f32_e32 v118, v118, v237
	v_add_f32_e32 v118, v118, v255
	v_add_f32_e32 v118, v118, v254
	v_add_f32_e32 v118, v118, v253
	v_add_f32_e32 v118, v118, v252
	v_add_f32_e32 v118, v118, v246
	v_add_f32_e32 v118, v118, v245
	v_add_f32_e32 v118, v118, v249
	v_add_f32_e32 v118, v118, v248
	v_add_f32_e32 v118, v118, v247
	v_add_f32_e32 v118, v118, v244
	v_add_f32_e32 v118, v118, v243
	v_add_f32_e32 v118, v118, v242
	v_pk_mul_f32 v[64:65], v[64:65], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[228:229] op_sel_hi:[1,0]
	v_cvt_f16_f32_e32 v230, v230
	v_cvt_f16_f32_e32 v231, v231
	v_cvt_f16_f32_e32 v237, v237
	v_pk_mul_f32 v[80:81], v[80:81], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[228:229] op_sel_hi:[1,0]
	v_cvt_f16_f32_e32 v252, v252
	s_mul_i32 s2, s18, 0xc0000
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s8, s2
	s_addc_u32 s5, s9, s3
	s_lshl_b32 s2, s17, 14
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	v_cvt_f16_f32_e32 v246, v246
	v_cvt_f16_f32_e32 v245, v245
	v_cvt_f16_f32_e32 v249, v249
	v_cvt_f16_f32_e32 v248, v248
	s_waitcnt vmcnt(0)
	v_cmp_eq_u32_e64 s[0:1], 0, v2
	scratch_load_dword v2, off, off offset:264 ; 4-byte Folded Reload
	scratch_load_dword v3, off, off offset:260 ; 4-byte Folded Reload
	scratch_load_dword v4, off, off offset:268 ; 4-byte Folded Reload
	scratch_load_dwordx4 v[98:101], off, off offset:132 ; 16-byte Folded Reload
	s_add_u32 s4, s4, s2
	s_addc_u32 s5, s5, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 2
	s_add_u32 s4, s4, s2
	v_pk_mul_f32 v[48:49], v[48:49], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[228:229] op_sel_hi:[1,0]
	s_addc_u32 s16, s5, s3
	s_add_i32 s2, s34, 0xffffc100
	s_add_u32 s12, s12, s22
	s_addc_u32 s3, s30, s23
	s_and_b32 s13, s3, 0xffff
	v_cvt_f16_f32_e32 v247, v247
	v_cvt_f16_f32_e32 v244, v244
	v_cvt_f16_f32_e32 v243, v243
	v_cvt_f16_f32_e32 v242, v242
	s_cmp_lt_i32 s2, 1
	s_mov_b32 s2, 0x3e0293ee
	v_pk_mul_f32 v[32:33], v[32:33], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[228:229] op_sel_hi:[1,0]
	s_mov_b32 s5, 0x7060302
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v2, 0xa0, v2
	s_waitcnt vmcnt(1)
	v_or3_b32 v2, v2, v4, v3
	scratch_store_dword off, v2, off offset:260 ; 4-byte Folded Spill
	v_mov_b64_e32 v[2:3], s[36:37]
	v_mov_b64_e32 v[4:5], s[38:39]
	v_mov_b64_e32 v[6:7], s[40:41]
	v_mov_b64_e32 v[8:9], s[42:43]
	v_mov_b64_e32 v[10:11], s[44:45]
	v_mov_b64_e32 v[12:13], s[46:47]
	v_mov_b64_e32 v[14:15], s[48:49]
	v_mov_b64_e32 v[16:17], s[50:51]
	s_waitcnt lgkmcnt(14)
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[82:97], v[222:223], v[142:143], v[2:17]
	v_cvt_f16_f32_e32 v222, v114
	v_cvt_f16_f32_e32 v223, v239
	v_cvt_f16_f32_e32 v239, v251
	v_cvt_f16_f32_e32 v251, v253
	v_mfma_f32_32x32x8_f16 v[82:97], v[224:225], v[144:145], v[82:97]
	v_cvt_f16_f32_e32 v225, v235
	v_cvt_f16_f32_e32 v235, v0
	v_cvt_f16_f32_e32 v224, v236
	v_cvt_f16_f32_e32 v236, v241
	v_cvt_f16_f32_e32 v241, v255
	v_mfma_f32_32x32x8_f16 v[82:97], v[218:219], v[138:139], v[82:97]
	v_cvt_f16_f32_e32 v218, v116
	v_cvt_f16_f32_e32 v219, v115
	v_mfma_f32_32x32x8_f16 v[82:97], v[220:221], v[140:141], v[82:97]
	v_cvt_f16_f32_e32 v221, v232
	v_cvt_f16_f32_e32 v232, v234
	v_cvt_f16_f32_e32 v234, v1
	v_cvt_f16_f32_e32 v220, v233
	v_cvt_f16_f32_e32 v233, v238
	v_cvt_f16_f32_e32 v238, v240
	v_cvt_f16_f32_e32 v240, v250
	v_mfma_f32_32x32x8_f16 v[82:97], v[214:215], v[134:135], v[82:97]
	v_cvt_f16_f32_e32 v250, v254
	v_mfma_f32_32x32x8_f16 v[82:97], v[216:217], v[136:137], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[210:211], v[130:131], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[212:213], v[132:133], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[206:207], v[126:127], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[208:209], v[128:129], v[82:97]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_f16 v[82:97], v[202:203], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[204:205], v[100:101], v[82:97]
	v_mov_b64_e32 v[204:205], v[100:101]
	v_mov_b64_e32 v[202:203], v[98:99]
	scratch_load_dwordx4 v[98:101], off, off offset:116 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[122:123], v[100:101]
	v_mfma_f32_32x32x8_f16 v[82:97], v[198:199], v[98:99], v[82:97]
	v_mov_b64_e32 v[120:121], v[98:99]
	v_mfma_f32_32x32x8_f16 v[82:97], v[200:201], v[100:101], v[82:97]
	scratch_load_dwordx4 v[98:101], off, off offset:100 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_f16 v[82:97], v[194:195], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[196:197], v[100:101], v[82:97]
	v_mov_b64_e32 v[196:197], v[100:101]
	v_mov_b64_e32 v[194:195], v[98:99]
	v_mfma_f32_32x32x8_f16 v[98:113], v[190:191], v[142:143], v[2:17]
	v_mfma_f32_32x32x8_f16 v[98:113], v[192:193], v[144:145], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[186:187], v[138:139], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[188:189], v[140:141], v[98:113]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[98:113], v[182:183], v[134:135], v[98:113]
	scratch_load_dword v182, off, off offset:156 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v183, v117
	v_mfma_f32_32x32x8_f16 v[98:113], v[184:185], v[136:137], v[98:113]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[98:113], v[178:179], v[130:131], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[180:181], v[132:133], v[98:113]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_f16 v[98:113], v[174:175], v[126:127], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[176:177], v[128:129], v[98:113]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[98:113], v[170:171], v[202:203], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[172:173], v[204:205], v[98:113]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_f16 v[98:113], v[166:167], v[120:121], v[98:113]
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v120, v182, v118
	v_mfma_f32_32x32x8_f16 v[98:113], v[168:169], v[122:123], v[98:113]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v118, v118, v120
	v_fmac_f32_e32 v118, v119, v228
	scratch_store_dword off, v118, off offset:264 ; 4-byte Folded Spill
	s_barrier
	scratch_load_dword v114, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v122, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v118, off, off       ; 4-byte Folded Reload
	scratch_load_dword v120, off, off offset:4 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[162:163], v[194:195], v[98:113]
	ds_read_b64 v[0:1], v125 offset:16384
	scratch_load_dword v116, off, off offset:152 ; 4-byte Folded Reload
	scratch_load_dword v253, off, off offset:192 ; 4-byte Folded Reload
	scratch_load_dword v254, off, off offset:196 ; 4-byte Folded Reload
	scratch_load_dword v255, off, off offset:200 ; 4-byte Folded Reload
	scratch_load_dword v226, off, off offset:204 ; 4-byte Folded Reload
	s_waitcnt vmcnt(8)
	ds_read_b64 v[114:115], v114 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[162:163], v122 offset:16384
	scratch_load_dword v122, off, off offset:12 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[164:165], v[196:197], v[98:113]
	scratch_store_dword off, v125, off offset:276 ; 4-byte Folded Spill
	s_waitcnt vmcnt(8)
	ds_read_b64 v[118:119], v118 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[120:121], v120 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[206:207], v253 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[208:209], v254 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[210:211], v255 offset:16384
	ds_read_b64 v[116:117], v116 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[216:217], v226 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[164:165], v122 offset:16384
	scratch_load_dword v122, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[166:167], v122 offset:16384
	scratch_load_dword v122, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[168:169], v122 offset:16384
	scratch_load_dword v122, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v122 offset:16384
	scratch_load_dword v122, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[172:173], v122 offset:16384
	scratch_load_dword v122, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[174:175], v122 offset:16384
	scratch_load_dword v122, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[176:177], v122 offset:16384
	scratch_load_dword v122, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[178:179], v122 offset:16384
	scratch_load_dword v122, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[180:181], v122 offset:16384
	scratch_load_dword v122, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[184:185], v122 offset:16384
	scratch_load_dword v122, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[186:187], v122 offset:16384
	scratch_load_dword v122, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[188:189], v122 offset:16384
	scratch_load_dword v122, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[190:191], v122 offset:16384
	scratch_load_dword v122, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[192:193], v122 offset:16384
	scratch_load_dword v122, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[194:195], v122 offset:16384
	scratch_load_dword v122, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[196:197], v122 offset:16384
	scratch_load_dword v122, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[198:199], v122 offset:16384
	scratch_load_dword v122, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[200:201], v122 offset:16384
	scratch_load_dword v122, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[202:203], v122 offset:16384
	scratch_load_dword v122, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[204:205], v122 offset:16384
	scratch_load_dword v122, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v122 offset:16384
	scratch_load_dword v122, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v122 offset:16384
	scratch_load_dword v122, off, off offset:248 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_write_b128 v122, v[158:161]
	ds_write_b128 v122, v[154:157] offset:8192
	v_pack_b32_f16 v155, v219, v220
	v_pack_b32_f16 v154, v183, v218
	v_pack_b32_f16 v157, v223, v224
	v_pack_b32_f16 v156, v221, v222
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[50:65], v[168:169], v[154:155], v[50:65]
	v_pack_b32_f16 v159, v230, v233
	v_pack_b32_f16 v158, v225, v232
	v_pack_b32_f16 v161, v235, v236
	v_pack_b32_f16 v160, v231, v234
	v_pack_b32_f16 v219, v240, v237
	v_pack_b32_f16 v218, v238, v239
	v_pack_b32_f16 v221, v251, v252
	v_mfma_f32_32x32x8_f16 v[50:65], v[170:171], v[156:157], v[50:65]
	v_pack_b32_f16 v220, v241, v250
	v_pack_b32_f16 v223, v249, v248
	v_pack_b32_f16 v222, v246, v245
	v_pack_b32_f16 v225, v243, v242
	v_pack_b32_f16 v224, v247, v244
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[158:159], v[50:65]
	v_mov_b32_e32 v245, v182
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[160:161], v[50:65]
	v_mfma_f32_32x32x8_f16 v[66:81], v[0:1], v[154:155], v[66:81]
	v_max_f32_e32 v0, v83, v83
	v_max_f32_e32 v1, v82, v82
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[218:219], v[50:65]
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_max3_f32 v0, v0, v98, v99
	v_max3_f32 v0, v0, v100, v101
	v_max3_f32 v0, v0, v102, v103
	v_max3_f32 v0, v0, v104, v105
	v_mfma_f32_32x32x8_f16 v[50:65], v[178:179], v[220:221], v[50:65]
	v_max3_f32 v0, v0, v106, v107
	v_max3_f32 v0, v0, v108, v109
	v_max3_f32 v0, v0, v110, v111
	v_max3_f32 v0, v0, v112, v113
	ds_bpermute_b32 v1, v182, v0
	v_mfma_f32_32x32x8_f16 v[50:65], v[180:181], v[222:223], v[50:65]
	v_mov_b32_e32 v180, v113
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v181, v227, v0, v1
	v_pk_mul_f32 v[178:179], v[180:181], s[2:3] op_sel_hi:[1,0]
	s_mov_b32 s3, 0x5040100
	v_fma_f32 v0, v82, s2, -v179
	v_fma_f32 v1, v83, s2, -v179
	v_fma_f32 v82, v84, s2, -v179
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[154:155], v[34:49]
	v_fma_f32 v83, v85, s2, -v179
	v_fma_f32 v84, v86, s2, -v179
	v_fma_f32 v85, v87, s2, -v179
	v_fma_f32 v86, v88, s2, -v179
	v_fma_f32 v87, v89, s2, -v179
	v_exp_f32_e32 v180, v0
	v_sub_f32_e32 v0, v229, v179
	v_mfma_f32_32x32x8_f16 v[50:65], v[184:185], v[224:225], v[50:65]
	v_fma_f32 v89, v91, s2, -v179
	v_fma_f32 v91, v93, s2, -v179
	v_fma_f32 v93, v95, s2, -v179
	v_fma_f32 v95, v97, s2, -v179
	v_fma_f32 v97, v99, s2, -v179
	v_fma_f32 v99, v101, s2, -v179
	v_fma_f32 v101, v103, s2, -v179
	v_mfma_f32_32x32x8_f16 v[34:49], v[188:189], v[156:157], v[34:49]
	v_fma_f32 v103, v105, s2, -v179
	v_fma_f32 v105, v107, s2, -v179
	v_fma_f32 v107, v109, s2, -v179
	v_fma_f32 v109, v111, s2, -v179
	v_sub_f32_e32 v111, v178, v179
	v_exp_f32_e32 v183, v1
	v_exp_f32_e32 v184, v82
	v_exp_f32_e32 v185, v83
	v_exp_f32_e32 v186, v84
	v_exp_f32_e32 v187, v85
	v_exp_f32_e32 v188, v86
	v_exp_f32_e32 v189, v87
	v_exp_f32_e32 v178, v0
	scratch_load_dword v0, off, off offset:208 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:212 ; 4-byte Folded Reload
	scratch_load_dword v82, off, off offset:216 ; 4-byte Folded Reload
	scratch_load_dword v83, off, off offset:220 ; 4-byte Folded Reload
	scratch_load_dword v84, off, off offset:224 ; 4-byte Folded Reload
	scratch_load_dword v85, off, off offset:228 ; 4-byte Folded Reload
	scratch_load_dword v86, off, off offset:232 ; 4-byte Folded Reload
	scratch_load_dword v87, off, off offset:236 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[18:33], v[202:203], v[154:155], v[18:33]
	v_fma_f32 v88, v90, s2, -v179
	v_fma_f32 v90, v92, s2, -v179
	v_fma_f32 v92, v94, s2, -v179
	v_fma_f32 v94, v96, s2, -v179
	v_fma_f32 v96, v98, s2, -v179
	v_fma_f32 v98, v100, s2, -v179
	v_fma_f32 v100, v102, s2, -v179
	v_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[156:157], v[66:81]
	v_fma_f32 v102, v104, s2, -v179
	v_fma_f32 v104, v106, s2, -v179
	v_fma_f32 v106, v108, s2, -v179
	v_fma_f32 v108, v110, s2, -v179
	v_fma_f32 v110, v112, s2, -v179
	v_exp_f32_e32 v202, v100
	v_exp_f32_e32 v203, v101
	v_mfma_f32_32x32x8_f16 v[18:33], v[204:205], v[156:157], v[18:33]
	v_exp_f32_e32 v204, v102
	v_exp_f32_e32 v205, v103
	v_pk_mul_f32 v[64:65], v[64:65], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[178:179] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[158:159], v[66:81]
	v_pk_mul_f32 v[54:55], v[54:55], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[178:179] op_sel_hi:[1,0]
	s_waitcnt vmcnt(7)
	ds_read_b128 v[114:117], v0
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[158:159], v[34:49]
	v_exp_f32_e32 v190, v88
	v_exp_f32_e32 v191, v89
	s_waitcnt vmcnt(2)
	ds_read_b128 v[228:231], v85
	s_waitcnt vmcnt(1)
	ds_read_b128 v[232:235], v86
	v_mfma_f32_32x32x8_f16 v[18:33], v[206:207], v[158:159], v[18:33]
	v_exp_f32_e32 v206, v104
	v_exp_f32_e32 v207, v105
	v_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[192:193], v[160:161], v[34:49]
	v_exp_f32_e32 v192, v90
	v_exp_f32_e32 v193, v91
	v_mfma_f32_32x32x8_f16 v[18:33], v[208:209], v[160:161], v[18:33]
	v_exp_f32_e32 v208, v106
	v_exp_f32_e32 v209, v107
	v_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[218:219], v[66:81]
	ds_read_b128 v[118:121], v1
	v_mfma_f32_32x32x8_f16 v[34:49], v[194:195], v[218:219], v[34:49]
	v_exp_f32_e32 v194, v92
	v_exp_f32_e32 v195, v93
	v_mfma_f32_32x32x8_f16 v[18:33], v[210:211], v[218:219], v[18:33]
	v_exp_f32_e32 v210, v108
	v_exp_f32_e32 v211, v109
	v_mfma_f32_32x32x8_f16 v[66:81], v[162:163], v[220:221], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[196:197], v[220:221], v[34:49]
	v_exp_f32_e32 v196, v94
	v_exp_f32_e32 v197, v95
	v_mfma_f32_32x32x8_f16 v[18:33], v[212:213], v[220:221], v[18:33]
	v_exp_f32_e32 v212, v110
	v_exp_f32_e32 v213, v111
	ds_read_b128 v[218:221], v83
	v_mfma_f32_32x32x8_f16 v[66:81], v[164:165], v[222:223], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[198:199], v[222:223], v[34:49]
	v_exp_f32_e32 v198, v96
	v_exp_f32_e32 v199, v97
	v_mfma_f32_32x32x8_f16 v[18:33], v[214:215], v[222:223], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[166:167], v[224:225], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[200:201], v[224:225], v[34:49]
	v_exp_f32_e32 v200, v98
	v_exp_f32_e32 v201, v99
	s_nop 7
	v_pk_mul_f32 v[80:81], v[80:81], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[178:179] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[18:33], v[216:217], v[224:225], v[18:33]
	ds_read_b128 v[214:217], v82
	ds_read_b128 v[222:225], v84
	s_waitcnt vmcnt(0)
	ds_read_b128 v[236:239], v87
	ds_read_b128 v[174:177], v0 offset:8192
	ds_read_b128 v[170:173], v1 offset:8192
	ds_read_b128 v[166:169], v82 offset:8192
	ds_read_b128 v[162:165], v83 offset:8192
	ds_read_b128 v[158:161], v84 offset:8192
	ds_read_b128 v[154:157], v85 offset:8192
	ds_read_b128 v[110:113], v86 offset:8192
	ds_read_b128 v[106:109], v87 offset:8192
	scratch_load_dword v227, off, off offset:160 ; 4-byte Folded Reload
	scratch_load_dword v240, off, off offset:172 ; 4-byte Folded Reload
	scratch_load_dword v242, off, off offset:180 ; 4-byte Folded Reload
	v_perm_b32 v0, v150, v146, s3
	scratch_load_dwordx4 v[122:125], off, off offset:132 ; 16-byte Folded Reload
	scratch_load_dword v241, off, off offset:176 ; 4-byte Folded Reload
	scratch_load_dword v243, off, off offset:184 ; 4-byte Folded Reload
	scratch_load_dword v244, off, off offset:188 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[142:143], v[2:17]
	v_pk_mul_f32 v[32:33], v[32:33], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[178:179] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[144:145], v[82:97]
	scratch_load_dwordx4 v[114:117], off, off offset:100 ; 16-byte Folded Reload
	v_pk_mul_f32 v[18:19], v[18:19], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[178:179] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[138:139], v[82:97]
	v_pk_mul_f32 v[44:45], v[44:45], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[178:179] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[178:179] op_sel_hi:[1,0]
	s_waitcnt vmcnt(7)
	ds_write_b32 v227, v0 offset:16384
	v_perm_b32 v0, v150, v146, s5
	scratch_load_dword v146, off, off offset:164 ; 4-byte Folded Reload
	scratch_load_dword v150, off, off offset:168 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[140:141], v[82:97]
	scratch_load_dwordx4 v[118:121], off, off offset:116 ; 16-byte Folded Reload
	s_waitcnt vmcnt(2)
	ds_write_b32 v146, v0 offset:16384
	v_perm_b32 v0, v151, v147, s3
	s_waitcnt vmcnt(1)
	ds_write_b32 v150, v0 offset:16384
	v_perm_b32 v0, v151, v147, s5
	ds_write_b32 v240, v0 offset:16384
	v_perm_b32 v0, v152, v148, s3
	ds_write_b32 v241, v0 offset:16384
	v_perm_b32 v0, v152, v148, s5
	ds_write_b32 v242, v0 offset:16384
	v_perm_b32 v0, v153, v149, s3
	ds_write_b32 v243, v0 offset:16384
	v_perm_b32 v0, v153, v149, s5
	ds_write_b32 v244, v0 offset:16384
	scratch_load_dword v0, off, off offset:240 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[2:17], v[174:175], v[142:143], v[2:17]
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[98:101], v0, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v0, off, off offset:244 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[214:215], v[134:135], v[82:97]
	v_mov_b32_e32 v214, v227
	v_mov_b32_e32 v215, v146
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[102:105], v0, s[12:15], 0 offen
	v_add_f32_e32 v0, v180, v183
	v_add_f32_e32 v0, v184, v0
	v_add_f32_e32 v0, v185, v0
	v_add_f32_e32 v0, v186, v0
	v_add_f32_e32 v0, v187, v0
	v_add_f32_e32 v0, v188, v0
	v_add_f32_e32 v0, v189, v0
	v_add_f32_e32 v0, v190, v0
	v_add_f32_e32 v0, v191, v0
	v_add_f32_e32 v0, v192, v0
	v_add_f32_e32 v0, v193, v0
	v_add_f32_e32 v0, v194, v0
	v_add_f32_e32 v0, v195, v0
	v_add_f32_e32 v0, v196, v0
	v_add_f32_e32 v0, v197, v0
	v_add_f32_e32 v0, v198, v0
	v_add_f32_e32 v0, v199, v0
	v_add_f32_e32 v0, v200, v0
	v_add_f32_e32 v0, v201, v0
	v_add_f32_e32 v0, v202, v0
	v_add_f32_e32 v0, v203, v0
	v_add_f32_e32 v0, v204, v0
	v_add_f32_e32 v0, v205, v0
	v_add_f32_e32 v0, v206, v0
	v_add_f32_e32 v0, v207, v0
	v_add_f32_e32 v0, v208, v0
	v_add_f32_e32 v0, v209, v0
	v_add_f32_e32 v0, v210, v0
	v_add_f32_e32 v0, v211, v0
	v_add_f32_e32 v0, v212, v0
	v_add_f32_e32 v0, v213, v0
	ds_bpermute_b32 v1, v182, v0
	v_mfma_f32_32x32x8_f16 v[2:17], v[176:177], v[144:145], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v0, v1
	scratch_load_dword v0, off, off offset:264 ; 4-byte Folded Reload
	s_barrier
	v_mfma_f32_32x32x8_f16 v[82:97], v[216:217], v[136:137], v[82:97]
	scratch_load_dword v217, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v227, off, off offset:76 ; 4-byte Folded Reload
	v_mov_b32_e32 v216, v150
	s_waitcnt vmcnt(2)
	v_fmac_f32_e32 v1, v0, v178
	v_mfma_f32_32x32x8_f16 v[2:17], v[170:171], v[138:139], v[2:17]
	v_cvt_f16_f32_e32 v0, v180
	v_cvt_f16_f32_e32 v180, v199
	v_cvt_f16_f32_e32 v199, v209
	scratch_load_dword v209, off, off offset:8 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[218:219], v[130:131], v[82:97]
	scratch_load_dword v218, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v219, off, off offset:36 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[2:17], v[172:173], v[140:141], v[2:17]
	v_mfma_f32_32x32x8_f16 v[82:97], v[220:221], v[132:133], v[82:97]
	scratch_load_dword v220, off, off offset:40 ; 4-byte Folded Reload
	scratch_load_dword v221, off, off offset:44 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[2:17], v[166:167], v[134:135], v[2:17]
	v_cvt_f16_f32_e32 v134, v195
	v_cvt_f16_f32_e32 v135, v196
	v_cvt_f16_f32_e32 v195, v205
	v_cvt_f16_f32_e32 v196, v206
	scratch_load_dword v205, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v206, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(8)
	ds_read_b64 v[166:167], v217 offset:16384
	v_mfma_f32_32x32x8_f16 v[82:97], v[222:223], v[126:127], v[82:97]
	scratch_load_dword v222, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v223, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	ds_read_b64 v[174:175], v206 offset:16384
	v_mfma_f32_32x32x8_f16 v[2:17], v[168:169], v[136:137], v[2:17]
	v_cvt_f16_f32_e32 v136, v197
	v_cvt_f16_f32_e32 v137, v198
	v_cvt_f16_f32_e32 v197, v207
	v_cvt_f16_f32_e32 v198, v208
	scratch_load_dword v207, off, off       ; 4-byte Folded Reload
	scratch_load_dword v208, off, off offset:4 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[224:225], v[128:129], v[82:97]
	scratch_load_dword v224, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v225, off, off offset:60 ; 4-byte Folded Reload
	ds_read_b64 v[168:169], v218 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[138:139], v223 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[140:141], v224 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[142:143], v225 offset:16384
	v_mfma_f32_32x32x8_f16 v[2:17], v[162:163], v[130:131], v[2:17]
	v_cvt_f16_f32_e32 v130, v191
	v_cvt_f16_f32_e32 v131, v192
	v_cvt_f16_f32_e32 v191, v201
	v_cvt_f16_f32_e32 v192, v202
	v_cvt_f16_f32_e32 v201, v211
	v_cvt_f16_f32_e32 v202, v212
	scratch_load_dword v211, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v212, off, off offset:20 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[228:229], v[122:123], v[82:97]
	scratch_load_dword v229, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v228, off, off offset:80 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[2:17], v[164:165], v[132:133], v[2:17]
	v_cvt_f16_f32_e32 v132, v193
	v_cvt_f16_f32_e32 v133, v194
	v_cvt_f16_f32_e32 v193, v203
	v_cvt_f16_f32_e32 v194, v204
	v_cvt_f16_f32_e32 v203, v213
	scratch_load_dword v204, off, off offset:276 ; 4-byte Folded Reload
	scratch_load_dword v213, off, off offset:24 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[230:231], v[124:125], v[82:97]
	scratch_load_dword v230, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v231, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(6)
	ds_read_b64 v[162:163], v212 offset:16384
	v_mfma_f32_32x32x8_f16 v[2:17], v[158:159], v[126:127], v[2:17]
	v_cvt_f16_f32_e32 v126, v186
	v_cvt_f16_f32_e32 v127, v187
	ds_read_b64 v[158:159], v221 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[170:171], v204 offset:16384
	v_mfma_f32_32x32x8_f16 v[82:97], v[232:233], v[118:119], v[82:97]
	scratch_load_dword v232, off, off offset:84 ; 4-byte Folded Reload
	scratch_load_dword v233, off, off offset:88 ; 4-byte Folded Reload
	ds_read_b64 v[172:173], v205 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[164:165], v213 offset:16384
	ds_read_b64 v[144:145], v229 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[146:147], v230 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[148:149], v231 offset:16384
	v_mfma_f32_32x32x8_f16 v[2:17], v[160:161], v[128:129], v[2:17]
	v_cvt_f16_f32_e32 v129, v190
	v_cvt_f16_f32_e32 v190, v200
	v_cvt_f16_f32_e32 v200, v210
	scratch_load_dword v210, off, off offset:12 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v128, v189
	ds_read_b64 v[176:177], v207 offset:16384
	ds_read_b64 v[160:161], v222 offset:16384
	v_mfma_f32_32x32x8_f16 v[2:17], v[154:155], v[122:123], v[2:17]
	v_cvt_f16_f32_e32 v122, v183
	v_cvt_f16_f32_e32 v123, v184
	ds_read_b64 v[182:183], v208 offset:16384
	ds_read_b64 v[154:155], v219 offset:16384
	ds_read_b64 v[150:151], v227 offset:16384
	ds_read_b64 v[152:153], v228 offset:16384
	v_pack_b32_f16 v122, v0, v122
	v_mfma_f32_32x32x8_f16 v[2:17], v[156:157], v[124:125], v[2:17]
	ds_read_b64 v[156:157], v220 offset:16384
	v_cvt_f16_f32_e32 v124, v185
	v_cvt_f16_f32_e32 v125, v188
	ds_read_b64 v[184:185], v209 offset:16384
	ds_read_b64 v[188:189], v211 offset:16384
	v_pack_b32_f16 v123, v123, v124
	v_pack_b32_f16 v125, v125, v128
	v_mfma_f32_32x32x8_f16 v[2:17], v[110:111], v[118:119], v[2:17]
	v_pack_b32_f16 v124, v126, v127
	v_pack_b32_f16 v127, v131, v132
	v_pack_b32_f16 v126, v129, v130
	v_pack_b32_f16 v129, v135, v136
	v_pack_b32_f16 v128, v133, v134
	v_pack_b32_f16 v131, v190, v191
	v_pack_b32_f16 v130, v137, v180
	v_mfma_f32_32x32x8_f16 v[2:17], v[112:113], v[120:121], v[2:17]
	v_pack_b32_f16 v133, v194, v195
	v_pack_b32_f16 v132, v192, v193
	v_pack_b32_f16 v135, v198, v199
	v_pack_b32_f16 v134, v196, v197
	v_pack_b32_f16 v137, v202, v203
	v_pack_b32_f16 v136, v200, v201
	s_waitcnt vmcnt(0)
	ds_read_b64 v[186:187], v210 offset:16384
	v_mfma_f32_32x32x8_f16 v[82:97], v[234:235], v[120:121], v[82:97]
	v_mfma_f32_32x32x8_f16 v[2:17], v[106:107], v[114:115], v[2:17]
	ds_read_b64 v[106:107], v232 offset:16384
	v_mfma_f32_32x32x8_f16 v[82:97], v[236:237], v[114:115], v[82:97]
	v_mfma_f32_32x32x8_f16 v[2:17], v[108:109], v[116:117], v[2:17]
	ds_read_b64 v[108:109], v233 offset:16384
	ds_read_b64 v[110:111], v253 offset:16384
	ds_read_b64 v[112:113], v254 offset:16384
	ds_read_b64 v[114:115], v255 offset:16384
	scratch_load_dword v234, off, off offset:92 ; 4-byte Folded Reload
	scratch_load_dword v235, off, off offset:96 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[238:239], v[116:117], v[82:97]
	s_waitcnt vmcnt(1)
	ds_read_b64 v[116:117], v234 offset:16384
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[18:33], v[106:107], v[122:123], v[18:33]
	s_nop 6
	v_max_f32_e32 v0, v83, v83
	v_max_f32_e32 v106, v82, v82
	v_max_f32_e32 v0, v106, v0
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_mfma_f32_32x32x8_f16 v[66:81], v[170:171], v[122:123], v[66:81]
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_max3_f32 v0, v0, v2, v3
	v_max3_f32 v0, v0, v4, v5
	v_max3_f32 v0, v0, v6, v7
	v_max3_f32 v0, v0, v8, v9
	v_max3_f32 v0, v0, v10, v11
	v_max3_f32 v0, v0, v12, v13
	v_mfma_f32_32x32x8_f16 v[66:81], v[172:173], v[124:125], v[66:81]
	v_max3_f32 v0, v0, v14, v15
	v_max3_f32 v0, v0, v16, v17
	ds_bpermute_b32 v106, v245, v0
	s_waitcnt vmcnt(0)
	ds_read_b64 v[118:119], v235 offset:16384
	ds_read_b64 v[120:121], v226 offset:16384
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[50:65], v[162:163], v[122:123], v[50:65]
	v_max3_f32 v107, v181, v0, v106
	v_mov_b32_e32 v106, v17
	v_mfma_f32_32x32x8_f16 v[34:49], v[138:139], v[122:123], v[34:49]
	v_mfma_f32_32x32x8_f16 v[66:81], v[174:175], v[126:127], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[164:165], v[124:125], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[140:141], v[124:125], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[124:125], v[18:33]
	v_pk_mul_f32 v[108:109], v[106:107], s[2:3] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v0, v82, s2, -v109
	v_fma_f32 v17, v83, s2, -v109
	v_fma_f32 v82, v84, s2, -v109
	v_fma_f32 v83, v85, s2, -v109
	v_fma_f32 v85, v87, s2, -v109
	v_fma_f32 v87, v89, s2, -v109
	v_fma_f32 v89, v91, s2, -v109
	v_fma_f32 v91, v93, s2, -v109
	v_fma_f32 v93, v95, s2, -v109
	v_fma_f32 v95, v97, s2, -v109
	v_exp_f32_e32 v97, v0
	v_sub_f32_e32 v0, v179, v109
	v_exp_f32_e32 v106, v82
	v_exp_f32_e32 v82, v0
	v_perm_b32 v0, v102, v98, s3
	ds_write_b32 v214, v0 offset:16384
	v_perm_b32 v0, v102, v98, s5
	ds_write_b32 v215, v0 offset:16384
	v_perm_b32 v0, v103, v99, s3
	ds_write_b32 v216, v0 offset:16384
	v_perm_b32 v0, v103, v99, s5
	v_exp_f32_e32 v17, v17
	ds_write_b32 v240, v0 offset:16384
	v_perm_b32 v0, v104, v100, s3
	v_mfma_f32_32x32x8_f16 v[66:81], v[176:177], v[128:129], v[66:81]
	ds_write_b32 v241, v0 offset:16384
	v_perm_b32 v0, v104, v100, s5
	v_fma_f32 v84, v86, s2, -v109
	v_exp_f32_e32 v83, v83
	ds_write_b32 v242, v0 offset:16384
	v_perm_b32 v0, v105, v101, s3
	v_exp_f32_e32 v84, v84
	v_mfma_f32_32x32x8_f16 v[50:65], v[166:167], v[126:127], v[50:65]
	ds_write_b32 v243, v0 offset:16384
	v_perm_b32 v0, v105, v101, s5
	v_fma_f32 v86, v88, s2, -v109
	v_exp_f32_e32 v85, v85
	ds_write_b32 v244, v0 offset:16384
	v_add_f32_e32 v0, v97, v17
	v_exp_f32_e32 v86, v86
	v_mfma_f32_32x32x8_f16 v[34:49], v[142:143], v[126:127], v[34:49]
	v_add_f32_e32 v0, v106, v0
	v_fma_f32 v88, v90, s2, -v109
	v_exp_f32_e32 v87, v87
	v_add_f32_e32 v0, v83, v0
	v_exp_f32_e32 v88, v88
	v_add_f32_e32 v0, v84, v0
	v_fma_f32 v90, v92, s2, -v109
	v_mfma_f32_32x32x8_f16 v[18:33], v[110:111], v[126:127], v[18:33]
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v0, v85, v0
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v0, v86, v0
	v_fma_f32 v92, v94, s2, -v109
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v0, v87, v0
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[130:131], v[66:81]
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v0, v88, v0
	v_fma_f32 v94, v96, s2, -v109
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v0, v89, v0
	v_exp_f32_e32 v94, v94
	v_add_f32_e32 v0, v90, v0
	v_mfma_f32_32x32x8_f16 v[50:65], v[168:169], v[128:129], v[50:65]
	v_fma_f32 v2, v2, s2, -v109
	v_exp_f32_e32 v95, v95
	v_add_f32_e32 v0, v91, v0
	v_fma_f32 v3, v3, s2, -v109
	v_exp_f32_e32 v2, v2
	v_add_f32_e32 v0, v92, v0
	v_fma_f32 v4, v4, s2, -v109
	v_mfma_f32_32x32x8_f16 v[34:49], v[144:145], v[128:129], v[34:49]
	v_exp_f32_e32 v3, v3
	v_add_f32_e32 v0, v93, v0
	v_fma_f32 v5, v5, s2, -v109
	v_exp_f32_e32 v4, v4
	v_add_f32_e32 v0, v94, v0
	v_fma_f32 v6, v6, s2, -v109
	v_exp_f32_e32 v5, v5
	v_mfma_f32_32x32x8_f16 v[18:33], v[112:113], v[128:129], v[18:33]
	v_add_f32_e32 v0, v95, v0
	v_fma_f32 v7, v7, s2, -v109
	v_exp_f32_e32 v6, v6
	v_add_f32_e32 v0, v2, v0
	v_fma_f32 v8, v8, s2, -v109
	v_exp_f32_e32 v7, v7
	v_add_f32_e32 v0, v3, v0
	v_mfma_f32_32x32x8_f16 v[66:81], v[184:185], v[132:133], v[66:81]
	v_fma_f32 v9, v9, s2, -v109
	v_exp_f32_e32 v8, v8
	v_add_f32_e32 v0, v4, v0
	v_fma_f32 v10, v10, s2, -v109
	v_exp_f32_e32 v9, v9
	v_add_f32_e32 v0, v5, v0
	v_fma_f32 v11, v11, s2, -v109
	v_mfma_f32_32x32x8_f16 v[50:65], v[154:155], v[130:131], v[50:65]
	v_exp_f32_e32 v10, v10
	v_add_f32_e32 v0, v6, v0
	v_fma_f32 v12, v12, s2, -v109
	v_exp_f32_e32 v11, v11
	v_add_f32_e32 v0, v7, v0
	v_fma_f32 v13, v13, s2, -v109
	v_exp_f32_e32 v12, v12
	v_mfma_f32_32x32x8_f16 v[34:49], v[146:147], v[130:131], v[34:49]
	v_add_f32_e32 v0, v8, v0
	v_fma_f32 v14, v14, s2, -v109
	v_exp_f32_e32 v13, v13
	v_add_f32_e32 v0, v9, v0
	v_fma_f32 v15, v15, s2, -v109
	v_exp_f32_e32 v14, v14
	v_add_f32_e32 v0, v10, v0
	v_mfma_f32_32x32x8_f16 v[18:33], v[114:115], v[130:131], v[18:33]
	v_fma_f32 v16, v16, s2, -v109
	v_exp_f32_e32 v15, v15
	v_add_f32_e32 v0, v11, v0
	v_sub_f32_e32 v96, v108, v109
	v_exp_f32_e32 v16, v16
	v_add_f32_e32 v0, v12, v0
	v_exp_f32_e32 v96, v96
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[134:135], v[66:81]
	v_add_f32_e32 v0, v13, v0
	v_add_f32_e32 v0, v14, v0
	v_add_f32_e32 v0, v15, v0
	v_add_f32_e32 v0, v16, v0
	v_add_f32_e32 v0, v96, v0
	ds_bpermute_b32 v98, v245, v0
	v_cvt_f16_f32_e32 v83, v83
	v_mfma_f32_32x32x8_f16 v[50:65], v[156:157], v[132:133], v[50:65]
	v_cvt_f16_f32_e32 v102, v86
	v_cvt_f16_f32_e32 v103, v87
	v_cvt_f16_f32_e32 v6, v6
	v_cvt_f16_f32_e32 v7, v7
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v98
	v_fmac_f32_e32 v0, v1, v82
	v_cvt_f16_f32_e32 v1, v97
	v_mfma_f32_32x32x8_f16 v[34:49], v[148:149], v[132:133], v[34:49]
	v_cvt_f16_f32_e32 v97, v106
	v_cvt_f16_f32_e32 v98, v84
	v_cvt_f16_f32_e32 v99, v85
	v_cvt_f16_f32_e32 v106, v88
	v_cvt_f16_f32_e32 v108, v89
	v_cvt_f16_f32_e32 v109, v90
	v_cvt_f16_f32_e32 v110, v91
	v_mfma_f32_32x32x8_f16 v[18:33], v[116:117], v[132:133], v[18:33]
	v_cvt_f16_f32_e32 v166, v92
	v_cvt_f16_f32_e32 v167, v93
	v_cvt_f16_f32_e32 v111, v94
	v_cvt_f16_f32_e32 v168, v95
	s_barrier
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[136:137], v[66:81]
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v5, v5
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v9, v9
	v_mfma_f32_32x32x8_f16 v[50:65], v[158:159], v[134:135], v[50:65]
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v11, v11
	v_cvt_f16_f32_e32 v12, v12
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v16, v16
	v_mfma_f32_32x32x8_f16 v[34:49], v[150:151], v[134:135], v[34:49]
	v_cvt_f16_f32_e32 v169, v96
	v_pack_b32_f16 v123, v97, v83
	v_pack_b32_f16 v122, v1, v17
	v_pack_b32_f16 v115, v109, v110
	v_pack_b32_f16 v114, v106, v108
	v_pack_b32_f16 v109, v4, v5
	v_pack_b32_f16 v108, v2, v3
	v_mfma_f32_32x32x8_f16 v[18:33], v[118:119], v[134:135], v[18:33]
	v_pack_b32_f16 v119, v102, v103
	v_pack_b32_f16 v102, v6, v7
	v_pk_mul_f32 v[6:7], v[70:71], v[82:83] op_sel_hi:[1,0]
	v_pack_b32_f16 v118, v98, v99
	v_pack_b32_f16 v103, v8, v9
	v_pack_b32_f16 v97, v12, v13
	v_pack_b32_f16 v96, v10, v11
	v_mfma_f32_32x32x8_f16 v[50:65], v[160:161], v[136:137], v[50:65]
	v_pack_b32_f16 v99, v16, v169
	v_pack_b32_f16 v98, v14, v15
	v_pk_mul_f32 v[16:17], v[80:81], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[78:79], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[76:77], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[74:75], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[72:73], v[82:83] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[34:49], v[152:153], v[136:137], v[34:49]
	v_pk_mul_f32 v[4:5], v[68:69], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[66:67], v[82:83] op_sel_hi:[1,0]
	s_nop 0
	v_pk_mul_f32 v[64:65], v[64:65], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[82:83] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[18:33], v[120:121], v[136:137], v[18:33]
	ds_read_b64 v[162:163], v204 offset:16384
	ds_read_b64 v[164:165], v205 offset:16384
	ds_read_b64 v[160:161], v206 offset:16384
	ds_read_b64 v[158:159], v207 offset:16384
	ds_read_b64 v[156:157], v208 offset:16384
	ds_read_b64 v[154:155], v209 offset:16384
	ds_read_b64 v[152:153], v210 offset:16384
	ds_read_b64 v[150:151], v211 offset:16384
	ds_read_b64 v[148:149], v212 offset:16384
	ds_read_b64 v[146:147], v213 offset:16384
	ds_read_b64 v[144:145], v217 offset:16384
	ds_read_b64 v[142:143], v218 offset:16384
	ds_read_b64 v[140:141], v219 offset:16384
	ds_read_b64 v[138:139], v220 offset:16384
	ds_read_b64 v[134:135], v221 offset:16384
	ds_read_b64 v[136:137], v222 offset:16384
	ds_read_b64 v[132:133], v223 offset:16384
	ds_read_b64 v[130:131], v224 offset:16384
	ds_read_b64 v[128:129], v225 offset:16384
	ds_read_b64 v[126:127], v229 offset:16384
	ds_read_b64 v[124:125], v230 offset:16384
	ds_read_b64 v[120:121], v231 offset:16384
	ds_read_b64 v[116:117], v227 offset:16384
	ds_read_b64 v[112:113], v228 offset:16384
	ds_read_b64 v[104:105], v232 offset:16384
	ds_read_b64 v[100:101], v233 offset:16384
	ds_read_b64 v[94:95], v253 offset:16384
	ds_read_b64 v[92:93], v254 offset:16384
	ds_read_b64 v[90:91], v255 offset:16384
	ds_read_b64 v[88:89], v234 offset:16384
	ds_read_b64 v[84:85], v235 offset:16384
	ds_read_b64 v[86:87], v226 offset:16384
	scratch_load_dword v70, off, off offset:260 ; 4-byte Folded Reload
	v_pk_mul_f32 v[54:55], v[54:55], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[82:83] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[2:17], v[162:163], v[122:123], v[2:17]
	v_pack_b32_f16 v111, v111, v168
	v_pack_b32_f16 v110, v166, v167
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v70, 2, 0
	v_mfma_f32_32x32x8_f16 v[50:65], v[148:149], v[122:123], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[132:133], v[122:123], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[104:105], v[122:123], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[164:165], v[118:119], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[146:147], v[118:119], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[130:131], v[118:119], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[100:101], v[118:119], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[160:161], v[114:115], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[144:145], v[114:115], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[128:129], v[114:115], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[94:95], v[114:115], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[158:159], v[110:111], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[142:143], v[110:111], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[126:127], v[110:111], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[92:93], v[110:111], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[156:157], v[108:109], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[140:141], v[108:109], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[124:125], v[108:109], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[90:91], v[108:109], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[154:155], v[102:103], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[138:139], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[120:121], v[102:103], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[88:89], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[152:153], v[96:97], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[134:135], v[96:97], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[116:117], v[96:97], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[84:85], v[96:97], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[150:151], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[136:137], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[112:113], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[86:87], v[98:99], v[18:33]
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	scratch_load_dword v69, off, off offset:252 ; 4-byte Folded Reload
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v0
	v_mov_b32_e32 v67, 0x42000000
	v_or_b32_e32 v66, s34, v70
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v0, v68
	v_log_f32_e32 v68, v68
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v67, 0, v67, vcc
	v_cmp_gt_i32_e64 s[8:9], s2, v66
	v_sub_f32_e32 v66, v68, v67
	v_add_f32_e32 v66, v107, v66
	ds_write_b32 v1, v66
	v_mov_b32_e32 v66, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_sub_i32 s2, 0x4000, s34
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
	s_cbranch_execz .LBB0_5
	s_branch .LBB0_6
.LBB0_4:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_5:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v0
	v_mov_b32_e32 v66, 0x42000000
	s_and_b32 s5, s16, 0xffff
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v0, v67
	v_log_f32_e32 v67, v67
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_sub_f32_e32 v66, v67, v66
	v_add_f32_e32 v66, v107, v66
	ds_write_b32 v1, v66
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v66, off, off offset:252 ; 4-byte Folded Reload
	v_mov_b32_e32 v1, 2
	v_bfrev_b32_e32 v67, 1
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v1, v1, v66 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v1
	ds_read_b32 v66, v66
	v_cndmask_b32_e64 v1, v67, v1, s[0:1]
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v1, s[4:7], 0 offen
.LBB0_6:                                ; %.critedge
	v_div_scale_f32 v1, s[0:1], v0, v0, 1.0
	v_rcp_f32_e32 v1, v1
	v_div_scale_f32 v66, vcc, 1.0, v0, 1.0
	v_mov_b32_e32 v67, v32
	v_mul_f32_e32 v1, v66, v1
	v_mov_b32_e32 v66, v31
	s_nop 0
	v_div_fmas_f32 v1, 0, 0, v1
	v_div_fixup_f32 v0, v1, v0, 1.0
	v_fma_mixlo_f16 v68, v0, v33, 0
	v_pk_mul_f32 v[32:33], v[0:1], v[66:67] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v66, v0, v30, 0
	v_mov_b32_e32 v30, v27
	v_mov_b32_e32 v31, v28
	v_fma_mixlo_f16 v67, v0, v29, 0
	v_pk_mul_f32 v[28:29], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v30, v0, v26, 0
	v_mov_b32_e32 v26, v23
	v_mov_b32_e32 v27, v24
	v_fma_mixlo_f16 v31, v0, v25, 0
	v_pk_mul_f32 v[24:25], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v26, v0, v22, 0
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v20
	v_fma_mixlo_f16 v27, v0, v21, 0
	v_pk_mul_f32 v[20:21], v[0:1], v[22:23] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v22, v0, v18, 0
	v_mov_b32_e32 v18, v47
	v_mov_b32_e32 v19, v48
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v47, v19
	v_cvt_f16_f32_e32 v48, v18
	v_mov_b32_e32 v18, v43
	v_mov_b32_e32 v19, v44
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v43, v19
	v_cvt_f16_f32_e32 v44, v18
	v_mov_b32_e32 v18, v39
	v_mov_b32_e32 v19, v40
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v39, v19
	v_cvt_f16_f32_e32 v40, v18
	v_mov_b32_e32 v18, v35
	v_mov_b32_e32 v19, v36
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v35, v19
	v_cvt_f16_f32_e32 v36, v18
	v_mov_b32_e32 v18, v63
	v_mov_b32_e32 v19, v64
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v63, v19
	v_cvt_f16_f32_e32 v64, v18
	v_mov_b32_e32 v18, v59
	v_mov_b32_e32 v19, v60
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v59, v19
	v_cvt_f16_f32_e32 v60, v18
	v_mov_b32_e32 v18, v55
	v_mov_b32_e32 v19, v56
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v55, v19
	v_cvt_f16_f32_e32 v56, v18
	v_mov_b32_e32 v18, v51
	v_mov_b32_e32 v19, v52
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v51, v19
	v_cvt_f16_f32_e32 v52, v18
	v_mov_b32_e32 v18, v15
	v_mov_b32_e32 v19, v16
	v_fma_mixlo_f16 v23, v0, v49, 0
	v_fma_mixlo_f16 v49, v0, v65, 0
	v_fma_mixlo_f16 v65, v0, v17, 0
	v_pk_mul_f32 v[16:17], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v18, v0, v14, 0
	v_mov_b32_e32 v14, v11
	v_mov_b32_e32 v15, v12
	v_fma_mixlo_f16 v19, v0, v13, 0
	v_pk_mul_f32 v[12:13], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v14, v0, v10, 0
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v8
	v_fma_mixlo_f16 v15, v0, v9, 0
	v_pk_mul_f32 v[8:9], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v10, v0, v6, 0
	v_fma_mixlo_f16 v1, v0, v5, 0
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v4
	v_pk_mul_f32 v[4:5], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v3, v5
	s_mov_b32 s4, 0x5040100
	s_mul_i32 s0, s25, s18
	s_ashr_i32 s1, s0, 31
	v_perm_b32 v1, v1, v3, s4
	scratch_load_dword v3, off, off offset:256 ; 4-byte Folded Reload
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s26, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s27, s34
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_f16_f32_e32 v4, v4
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s27, 0x3fff
	v_fma_mixlo_f16 v46, v0, v46, 0
	v_fma_mixlo_f16 v45, v0, v45, 0
	v_fma_mixlo_f16 v42, v0, v42, 0
	v_fma_mixlo_f16 v41, v0, v41, 0
	v_fma_mixlo_f16 v38, v0, v38, 0
	v_fma_mixlo_f16 v37, v0, v37, 0
	v_fma_mixlo_f16 v34, v0, v34, 0
	v_fma_mixlo_f16 v62, v0, v62, 0
	v_fma_mixlo_f16 v61, v0, v61, 0
	v_fma_mixlo_f16 v58, v0, v58, 0
	v_fma_mixlo_f16 v57, v0, v57, 0
	v_fma_mixlo_f16 v54, v0, v54, 0
	v_fma_mixlo_f16 v53, v0, v53, 0
	v_fma_mixlo_f16 v50, v0, v50, 0
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v8, v8
	v_fma_mixlo_f16 v0, v0, v2, 0
	v_mul_lo_u32 v2, s27, v70
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s2, s1
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v12, v12
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_pack_b32_f16 v0, v0, v4
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v29, v29
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v33, v33
	v_cvt_f16_f32_e32 v32, v32
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v2, v2, v3, 1
	v_bfrev_b32_e32 v3, 1
	v_cndmask_b32_e64 v4, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 16, v2
	v_pack_b32_f16 v0, v10, v8
	v_perm_b32 v1, v15, v9, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 32, v2
	v_pack_b32_f16 v0, v14, v12
	v_perm_b32 v1, v19, v13, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 48, v2
	v_pack_b32_f16 v0, v18, v16
	v_perm_b32 v1, v65, v17, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 64, v2
	v_pack_b32_f16 v0, v50, v52
	v_perm_b32 v1, v53, v51, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x50, v2
	v_pack_b32_f16 v0, v54, v56
	v_perm_b32 v1, v57, v55, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x60, v2
	v_pack_b32_f16 v0, v58, v60
	v_perm_b32 v1, v61, v59, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x70, v2
	v_pack_b32_f16 v0, v62, v64
	v_perm_b32 v1, v49, v63, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x80, v2
	v_pack_b32_f16 v0, v34, v36
	v_perm_b32 v1, v37, v35, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x90, v2
	v_pack_b32_f16 v0, v38, v40
	v_perm_b32 v1, v41, v39, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xa0, v2
	v_pack_b32_f16 v0, v42, v44
	v_perm_b32 v1, v45, v43, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xb0, v2
	v_pack_b32_f16 v0, v46, v48
	v_perm_b32 v1, v23, v47, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xc0, v2
	v_pack_b32_f16 v0, v22, v20
	v_perm_b32 v1, v27, v21, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xd0, v2
	v_pack_b32_f16 v0, v26, v24
	v_perm_b32 v1, v31, v25, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xe0, v2
	v_pack_b32_f16 v0, v30, v28
	v_perm_b32 v1, v67, v29, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	v_add_u32_e32 v2, 0xf0, v2
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_pack_b32_f16 v0, v66, v32
	v_perm_b32 v1, v68, v33, s4
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 284
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
		.amdhsa_next_free_sgpr 56
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
	.set attn_fwd.numbered_sgpr, 56
	.set attn_fwd.private_seg_size, 284
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16880
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 284
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 62
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
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
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
    .private_segment_fixed_size: 284
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 72
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
