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
	v_add_lshl_u32 v94, v35, v58, 1
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
	v_add_u32_e32 v92, v36, v34
	s_or_b32 s29, s1, s19
	v_add_lshl_u32 v93, v1, v58, 1
	s_barrier
	scratch_store_dword off, v37, off offset:256 ; 4-byte Folded Spill
	s_waitcnt vmcnt(8)
	ds_write_b128 v92, v[2:5]
	s_waitcnt vmcnt(7)
	ds_write_b128 v92, v[6:9] offset:8192
	s_waitcnt vmcnt(6)
	ds_write_b128 v92, v[10:13] offset:16384
	s_waitcnt vmcnt(5)
	ds_write_b128 v92, v[14:17] offset:24576
	s_waitcnt vmcnt(4)
	ds_write_b128 v92, v[18:21] offset:32768
	s_waitcnt vmcnt(3)
	ds_write_b128 v92, v[22:25] offset:40960
	s_waitcnt vmcnt(2)
	ds_write_b128 v92, v[26:29] offset:49152
	s_waitcnt vmcnt(1)
	ds_write_b128 v92, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[2:5], v93, s[28:31], 0 offen
	buffer_load_dwordx4 v[6:9], v94, s[28:31], 0 offen
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
	scratch_store_dword off, v38, off offset:260 ; 4-byte Folded Spill
	v_add_u32_e32 v12, v10, v11
	v_lshlrev_b32_e32 v22, 4, v13
	v_lshlrev_b32_e32 v23, 4, v14
	s_lshl_b64 s[6:7], s[30:31], 1
	v_add_u32_e32 v13, v10, v22
	ds_read_b128 v[158:161], v12
	ds_read_b128 v[154:157], v13
	v_add_u32_e32 v12, v10, v23
	v_lshlrev_b32_e32 v24, 4, v15
	v_lshlrev_b32_e32 v59, 4, v16
	s_add_u32 s0, s28, s6
	v_add_u32_e32 v13, v10, v24
	ds_read_b128 v[150:153], v12
	ds_read_b128 v[146:149], v13
	v_add_u32_e32 v12, v10, v59
	v_lshlrev_b32_e32 v76, 4, v17
	v_lshlrev_b32_e32 v77, 4, v18
	s_addc_u32 s23, s13, s7
	v_add_u32_e32 v13, v10, v76
	ds_read_b128 v[142:145], v12
	ds_read_b128 v[138:141], v13
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
	ds_write_b128 v92, v[2:5]
	s_waitcnt vmcnt(1)
	ds_write_b128 v92, v[6:9] offset:8192
	s_or_b32 s1, s1, s19
	v_add3_u32 v88, 0, v11, v78
	buffer_load_dwordx4 v[50:53], v93, s[0:3], 0 offen
	buffer_load_dwordx4 v[54:57], v94, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v19, off offset:264 ; 4-byte Folded Spill
	ds_read_b128 v[18:21], v88
	ds_read_b128 v[60:63], v88 offset:8192
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
	v_add3_u32 v89, 0, v22, v78
	v_add3_u32 v90, 0, v23, v78
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[158:159], v[34:49]
	v_add3_u32 v91, 0, v24, v78
	v_add3_u32 v59, 0, v59, v78
	s_and_b32 s1, s24, 0x3fff
	s_bitset1_b32 s1, 14
	s_and_b32 s13, s33, 0xffff
	s_lshl_b32 s20, s1, 16
	s_mov_b32 s14, s2
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[160:161], v[2:17]
	ds_read_b128 v[18:21], v89
	ds_read_b128 v[64:67], v89 offset:8192
	s_mov_b32 s15, s3
	s_or_b32 s13, s13, s20
	s_mov_b32 s1, 0xff800000
	s_mov_b32 s16, 0x3e0293ee
	s_add_u32 s0, s0, s6
	v_mov_b32_e32 v135, 1.0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[154:155], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[156:157], v[2:17]
	ds_read_b128 v[18:21], v90
	ds_read_b128 v[68:71], v90 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[150:151], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[152:153], v[2:17]
	ds_read_b128 v[18:21], v91
	ds_read_b128 v[72:75], v91 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[146:147], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[148:149], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[60:61], v[158:159], v[34:49]
	s_nop 6
	ds_read_b128 v[34:37], v59
	ds_read_b128 v[38:41], v59 offset:8192
	v_add3_u32 v60, 0, v76, v78
	v_add3_u32 v61, 0, v77, v78
	v_and_b32_e32 v42, 0x100, v79
	v_or_b32_e32 v0, v0, v42
	v_lshrrev_b32_e32 v46, 3, v79
	v_lshrrev_b32_e32 v0, 3, v0
	v_mfma_f32_32x32x8_f16 v[18:33], v[62:63], v[160:161], v[18:33]
	v_add3_u32 v62, 0, v1, v78
	v_and_b32_e32 v1, 16, v79
	v_lshrrev_b32_e32 v47, 3, v1
	v_and_or_b32 v48, v46, 12, v0
	v_or_b32_e32 v0, v48, v47
	v_mad_u64_u32 v[0:1], s[28:29], s24, v0, v[58:59]
	v_mfma_f32_32x32x8_f16 v[18:33], v[64:65], v[154:155], v[18:33]
	s_mov_b32 s28, 0x7060302
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[156:157], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[68:69], v[150:151], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[70:71], v[152:153], v[18:33]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[146:147], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[148:149], v[18:33]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[142:143], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[142:143], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[144:145], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[144:145], v[18:33]
	ds_read_b128 v[34:37], v60
	ds_read_b128 v[38:41], v60 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[138:139], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[138:139], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[140:141], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[140:141], v[18:33]
	ds_read_b128 v[34:37], v61
	ds_read_b128 v[38:41], v61 offset:8192
	scratch_store_dwordx4 off, v[84:87], off offset:80 ; 16-byte Folded Spill
	scratch_store_dword off, v42, off offset:268 ; 4-byte Folded Spill
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
	scratch_store_dwordx4 off, v[80:83], off offset:64 ; 16-byte Folded Spill
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
	scratch_store_dword off, v38, off offset:96 ; 4-byte Folded Spill
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
	s_waitcnt vmcnt(8)
	ds_write_b128 v92, v[50:53]
	s_waitcnt vmcnt(7)
	ds_write_b128 v92, v[54:57] offset:8192
	scratch_store_dword off, v92, off offset:228 ; 4-byte Folded Spill
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
	scratch_store_dword off, v93, off offset:240 ; 4-byte Folded Spill
	buffer_load_dwordx4 v[118:121], v93, s[0:3], 0 offen
	buffer_load_dwordx4 v[114:117], v94, s[0:3], 0 offen
	s_waitcnt vmcnt(6)
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
	scratch_store_dword off, v94, off offset:244 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v54, off       ; 4-byte Folded Spill
	ds_write_b32 v54, v38 offset:16384
	scratch_store_dword off, v88, off offset:196 ; 4-byte Folded Spill
	ds_read_b128 v[222:225], v88
	ds_read_b128 v[190:193], v88 offset:8192
	ds_read_b128 v[218:221], v89
	scratch_store_dword off, v89, off offset:200 ; 4-byte Folded Spill
	ds_read_b128 v[186:189], v89 offset:8192
	ds_read_b128 v[214:217], v90
	scratch_store_dword off, v90, off offset:204 ; 4-byte Folded Spill
	ds_read_b128 v[182:185], v90 offset:8192
	ds_read_b128 v[210:213], v91
	scratch_store_dword off, v91, off offset:208 ; 4-byte Folded Spill
	ds_read_b128 v[178:181], v91 offset:8192
	ds_read_b128 v[206:209], v59
	scratch_store_dword off, v59, off offset:212 ; 4-byte Folded Spill
	ds_read_b128 v[174:177], v59 offset:8192
	ds_read_b128 v[202:205], v60
	scratch_store_dword off, v60, off offset:216 ; 4-byte Folded Spill
	s_and_b32 s0, s31, 0xffff
	ds_read_b128 v[170:173], v60 offset:8192
	ds_read_b128 v[198:201], v61
	scratch_store_dword off, v61, off offset:220 ; 4-byte Folded Spill
	s_or_b32 s1, s0, s20
	s_mov_b32 s0, s29
	ds_read_b128 v[166:169], v61 offset:8192
	ds_read_b128 v[194:197], v62
	buffer_load_dwordx4 v[126:129], v85, s[0:3], 0 offen
	buffer_load_dwordx4 v[122:125], v84, s[0:3], 0 offen
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
	scratch_store_dword off, v38, off offset:4 ; 4-byte Folded Spill
	scratch_store_dword off, v34, off offset:100 ; 4-byte Folded Spill
	ds_write_b32 v34, v39 offset:16384
	v_or_b32_e32 v34, 0xcc, v45
	v_or_b32_e32 v38, v51, v49
	v_xor_b32_e32 v34, v38, v34
	v_or3_b32 v34, v47, v34, v53
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:104 ; 4-byte Folded Spill
	ds_write_b32 v34, v35 offset:16384
	v_or_b32_e32 v34, 0x110, v52
	v_xor_b32_e32 v34, v34, v53
	v_or_b32_e32 v34, v34, v47
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:8 ; 4-byte Folded Spill
	ds_write_b32 v34, v40 offset:16384
	v_or_b32_e32 v34, 0x154, v45
	v_xor_b32_e32 v34, v34, v49
	v_or_b32_e32 v34, v34, v51
	v_xor_b32_e32 v34, v34, v53
	v_or_b32_e32 v34, v34, v47
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:108 ; 4-byte Folded Spill
	ds_write_b32 v34, v36 offset:16384
	v_or_b32_e32 v34, 0x198, v50
	v_xor_b32_e32 v34, v54, v34
	v_or_b32_e32 v34, v34, v47
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:112 ; 4-byte Folded Spill
	ds_write_b32 v34, v41 offset:16384
	v_or_b32_e32 v34, v38, v53
	v_or_b32_e32 v35, 0x1dc, v45
	v_xor_b32_e32 v34, v34, v35
	v_or_b32_e32 v34, v34, v47
	v_xor_b32_e32 v34, v48, v34
	v_lshl_add_u32 v34, v34, 1, 0
	scratch_store_dword off, v34, off offset:116 ; 4-byte Folded Spill
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
	scratch_store_dword off, v56, off offset:252 ; 4-byte Folded Spill
	v_or_b32_e32 v56, 56, v0
	v_or_b32_e32 v57, v52, v1
	v_or_b32_e32 v58, 0x838, v0
	v_or_b32_e32 v59, 0x1038, v0
	v_or_b32_e32 v60, 0x1838, v0
	scratch_store_dword off, v62, off offset:224 ; 4-byte Folded Spill
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
	v_exp_f32_e32 v252, v3
	v_lshl_add_u32 v3, v53, 1, 0
	v_xor_b32_e32 v52, v52, v1
	v_xor_b32_e32 v63, v63, v1
	v_xor_b32_e32 v64, v64, v1
	v_xor_b32_e32 v0, v0, v1
	scratch_store_dword off, v3, off offset:120 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v41, 1, 0
	v_or_b32_e32 v1, v52, v34
	v_or_b32_e32 v52, v63, v34
	v_or_b32_e32 v63, v64, v34
	v_or_b32_e32 v0, v0, v34
	v_or_b32_e32 v34, 32, v51
	scratch_store_dword off, v3, off offset:124 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v44, 1, 0
	v_xor_b32_e32 v34, v39, v34
	scratch_store_dword off, v3, off offset:128 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v35, 1, 0
	v_xor_b32_e32 v64, v39, v1
	scratch_store_dword off, v3, off offset:132 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v34, 1, 0
	scratch_store_dword off, v3, off offset:136 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v64, 1, 0
	scratch_store_dword off, v3, off offset:140 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v60, 1, 0
	scratch_store_dword off, v3, off offset:144 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v56, 1, 0
	scratch_store_dword off, v3, off offset:148 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v45, 1, 0
	scratch_store_dword off, v3, off offset:152 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v54, 1, 0
	scratch_store_dword off, v3, off offset:156 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v37, 1, 0
	v_xor_b32_e32 v52, v39, v52
	v_or_b32_e32 v1, 0x820, v51
	scratch_store_dword off, v3, off offset:160 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v46, 1, 0
	v_xor_b32_e32 v65, v39, v1
	scratch_store_dword off, v3, off offset:164 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v52, 1, 0
	scratch_store_dword off, v3, off offset:168 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v65, 1, 0
	scratch_store_dword off, v3, off offset:172 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v58, 1, 0
	scratch_store_dword off, v3, off offset:176 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v61, 1, 0
	scratch_store_dword off, v3, off offset:180 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v47, 1, 0
	scratch_store_dword off, v3, off offset:184 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v38, 1, 0
	scratch_store_dword off, v3, off offset:12 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v55, 1, 0
	scratch_store_dword off, v3, off offset:16 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v48, 1, 0
	v_or_b32_e32 v1, 0x1020, v51
	scratch_store_dword off, v3, off offset:20 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v62, 1, 0
	v_xor_b32_e32 v66, v39, v1
	scratch_store_dword off, v3, off offset:188 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v59, 1, 0
	v_xor_b32_e32 v63, v39, v63
	scratch_store_dword off, v3, off offset:24 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v66, 1, 0
	scratch_store_dword off, v3, off offset:28 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v63, 1, 0
	scratch_store_dword off, v3, off offset:32 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v36, 1, 0
	scratch_store_dword off, v3, off offset:36 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v49, 1, 0
	s_add_u32 s12, s52, s54
	scratch_store_dword off, v3, off offset:40 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v50, 1, 0
	s_addc_u32 s13, s53, s55
	v_xor_b32_e32 v67, v39, v0
	v_or_b32_e32 v0, 0x1820, v51
	scratch_store_dword off, v3, off offset:44 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v40, 1, 0
	s_lshl_b64 s[12:13], s[12:13], 1
	v_xor_b32_e32 v39, v39, v0
	v_exp_f32_e32 v250, v33
	v_exp_f32_e32 v251, v4
	v_exp_f32_e32 v254, v5
	v_exp_f32_e32 v253, v6
	v_exp_f32_e32 v238, v7
	v_exp_f32_e32 v255, v8
	v_exp_f32_e32 v240, v9
	v_exp_f32_e32 v239, v10
	v_exp_f32_e32 v230, v11
	v_exp_f32_e32 v241, v12
	v_exp_f32_e32 v1, v13
	v_exp_f32_e32 v231, v14
	v_exp_f32_e32 v234, v15
	v_exp_f32_e32 v0, v16
	v_exp_f32_e32 v233, v17
	v_exp_f32_e32 v232, v18
	v_exp_f32_e32 v236, v19
	v_exp_f32_e32 v235, v20
	v_exp_f32_e32 v130, v21
	v_exp_f32_e32 v237, v22
	v_exp_f32_e32 v132, v23
	v_exp_f32_e32 v131, v24
	v_exp_f32_e32 v133, v25
	v_exp_f32_e32 v242, v26
	v_exp_f32_e32 v243, v27
	v_exp_f32_e32 v244, v28
	v_exp_f32_e32 v246, v29
	v_exp_f32_e32 v245, v30
	v_exp_f32_e32 v248, v31
	v_exp_f32_e32 v247, v32
	v_exp_f32_e32 v249, v42
	scratch_store_dword off, v3, off offset:48 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v57, 1, 0
	s_add_u32 s12, s14, s12
	v_exp_f32_e32 v228, v2
	scratch_store_dword off, v3, off offset:52 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v43, 1, 0
	s_addc_u32 s13, s15, s13
	scratch_store_dword off, v3, off offset:56 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v67, 1, 0
	s_add_u32 s4, s4, s12
	v_mov_b32_e32 v18, 0
	scratch_store_dword off, v79, off offset:248 ; 4-byte Folded Spill
	scratch_store_dword off, v3, off offset:60 ; 4-byte Folded Spill
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
	v_lshrrev_b32_e32 v134, 16, v126
	scratch_store_dword off, v3, off offset:192 ; 4-byte Folded Spill
	scratch_store_dword off, v85, off offset:236 ; 4-byte Folded Spill
	scratch_store_dword off, v84, off offset:232 ; 4-byte Folded Spill
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b64_e32 v[112:113], s[50:51]
	v_mov_b64_e32 v[110:111], s[48:49]
	v_mov_b64_e32 v[108:109], s[46:47]
	v_mov_b64_e32 v[106:107], s[44:45]
	v_mov_b64_e32 v[104:105], s[42:43]
	v_mov_b64_e32 v[102:103], s[40:41]
	v_mov_b64_e32 v[100:101], s[38:39]
	v_mov_b64_e32 v[98:99], s[36:37]
	v_mov_b32_e32 v136, v135
	v_pk_mul_f32 v[66:67], v[66:67], v[228:229] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[2:17], v[222:223], v[158:159], v[98:113]
	v_pk_mul_f32 v[68:69], v[68:69], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[224:225], v[160:161], v[2:17]
	v_pk_mul_f32 v[50:51], v[50:51], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[218:219], v[154:155], v[2:17]
	v_pk_mul_f32 v[64:65], v[64:65], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[220:221], v[156:157], v[2:17]
	v_pk_mul_f32 v[46:47], v[46:47], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[214:215], v[150:151], v[2:17]
	v_pk_mul_f32 v[28:29], v[28:29], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[228:229] op_sel_hi:[1,0]
	v_mov_b32_e32 v226, v227
	scratch_load_dword v218, off, off offset:96 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[2:17], v[216:217], v[152:153], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[210:211], v[146:147], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[212:213], v[148:149], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[206:207], v[142:143], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[208:209], v[144:145], v[2:17]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[2:17], v[202:203], v[138:139], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[204:205], v[140:141], v[2:17]
	scratch_load_dwordx4 v[202:205], off, off offset:80 ; 16-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[190:191], v[158:159], v[98:113]
	v_mfma_f32_32x32x8_f16 v[82:97], v[192:193], v[160:161], v[82:97]
	s_nop 5
	v_add_f32_e32 v98, v250, v252
	v_add_f32_e32 v98, v98, v251
	v_add_f32_e32 v98, v98, v254
	v_add_f32_e32 v98, v98, v253
	v_add_f32_e32 v98, v98, v238
	v_add_f32_e32 v98, v98, v255
	v_add_f32_e32 v98, v98, v240
	v_mfma_f32_32x32x8_f16 v[82:97], v[186:187], v[154:155], v[82:97]
	v_add_f32_e32 v98, v98, v239
	v_add_f32_e32 v98, v98, v230
	v_add_f32_e32 v98, v98, v241
	v_add_f32_e32 v98, v98, v1
	v_add_f32_e32 v98, v98, v231
	v_add_f32_e32 v98, v98, v234
	v_add_f32_e32 v98, v98, v0
	s_waitcnt vmcnt(0) lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[2:17], v[198:199], v[202:203], v[2:17]
	v_add_f32_e32 v98, v98, v233
	v_add_f32_e32 v98, v98, v232
	v_add_f32_e32 v98, v98, v236
	v_add_f32_e32 v98, v98, v235
	v_add_f32_e32 v98, v98, v130
	v_add_f32_e32 v98, v98, v237
	v_add_f32_e32 v98, v98, v132
	v_mfma_f32_32x32x8_f16 v[82:97], v[188:189], v[156:157], v[82:97]
	v_add_f32_e32 v98, v98, v131
	v_add_f32_e32 v98, v98, v133
	v_add_f32_e32 v98, v98, v242
	v_add_f32_e32 v98, v98, v243
	v_add_f32_e32 v98, v98, v244
	v_add_f32_e32 v98, v98, v246
	v_add_f32_e32 v98, v98, v245
	v_mfma_f32_32x32x8_f16 v[2:17], v[200:201], v[204:205], v[2:17]
	scratch_load_dwordx4 v[198:201], off, off offset:64 ; 16-byte Folded Reload
	v_add_f32_e32 v98, v98, v248
	v_add_f32_e32 v98, v98, v247
	v_add_f32_e32 v98, v98, v249
	v_cvt_pkrtz_f16_f32 v108, v250, v252
	v_cvt_pkrtz_f16_f32 v109, v251, v254
	v_cvt_pkrtz_f16_f32 v106, v253, v238
	v_mfma_f32_32x32x8_f16 v[82:97], v[182:183], v[150:151], v[82:97]
	v_cvt_pkrtz_f16_f32 v107, v255, v240
	v_cvt_pkrtz_f16_f32 v104, v239, v230
	v_cvt_pkrtz_f16_f32 v105, v241, v1
	v_cvt_pkrtz_f16_f32 v102, v231, v234
	v_cvt_pkrtz_f16_f32 v103, v0, v233
	v_cvt_pkrtz_f16_f32 v100, v232, v236
	v_cvt_pkrtz_f16_f32 v101, v235, v130
	v_mfma_f32_32x32x8_f16 v[82:97], v[184:185], v[152:153], v[82:97]
	v_cvt_pkrtz_f16_f32 v112, v242, v243
	v_cvt_pkrtz_f16_f32 v113, v244, v246
	v_cvt_pkrtz_f16_f32 v110, v245, v248
	v_cvt_pkrtz_f16_f32 v111, v247, v249
	ds_bpermute_b32 v99, v218, v98
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v135, v98, v99
	v_mfma_f32_32x32x8_f16 v[82:97], v[178:179], v[146:147], v[82:97]
	v_cvt_pkrtz_f16_f32 v98, v237, v132
	v_cvt_pkrtz_f16_f32 v99, v131, v133
	v_fmac_f32_e32 v135, v136, v228
	v_mfma_f32_32x32x8_f16 v[82:97], v[180:181], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[142:143], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[144:145], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[138:139], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[140:141], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[202:203], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[204:205], v[82:97]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[194:195], v[198:199], v[2:17]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[198:199], v[82:97]
	v_mfma_f32_32x32x8_f16 v[2:17], v[196:197], v[200:201], v[2:17]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[200:201], v[82:97]
	; sched_barrier mask(0x00000000)
	s_barrier
	scratch_load_dword v0, off, off offset:120 ; 4-byte Folded Reload
	scratch_load_dword v162, off, off offset:136 ; 4-byte Folded Reload
	scratch_load_dword v130, off, off offset:124 ; 4-byte Folded Reload
	scratch_load_dword v132, off, off offset:128 ; 4-byte Folded Reload
	scratch_load_dword v136, off, off offset:132 ; 4-byte Folded Reload
	s_add_u32 s12, s29, s22
	s_addc_u32 s30, s31, s23
	s_and_b32 s0, s5, 0xffff
	s_or_b32 s1, s0, s19
	s_mov_b32 s0, s4
	scratch_load_dword v219, off, off offset:228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(4)
	ds_read_b64 v[194:195], v162 offset:16384
	scratch_load_dword v162, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[196:197], v162 offset:16384
	scratch_load_dword v162, off, off offset:144 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[198:199], v162 offset:16384
	scratch_load_dword v162, off, off offset:148 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[200:201], v162 offset:16384
	scratch_load_dword v162, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[202:203], v162 offset:16384
	scratch_load_dword v162, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[204:205], v162 offset:16384
	scratch_load_dword v162, off, off offset:160 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[206:207], v162 offset:16384
	scratch_load_dword v162, off, off offset:164 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[208:209], v162 offset:16384
	scratch_load_dword v162, off, off offset:168 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[210:211], v162 offset:16384
	scratch_load_dword v162, off, off offset:172 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v162 offset:16384
	scratch_load_dword v162, off, off offset:176 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v162 offset:16384
	scratch_load_dword v162, off, off offset:180 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[216:217], v162 offset:16384
	scratch_load_dword v162, off, off offset:184 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[192:193], v162 offset:16384
	scratch_load_dword v162, off, off offset:12 ; 4-byte Folded Reload
	ds_read_b64 v[0:1], v0 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[188:189], v162 offset:16384
	scratch_load_dword v162, off, off offset:16 ; 4-byte Folded Reload
	ds_read_b64 v[130:131], v130 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[190:191], v162 offset:16384
	scratch_load_dword v162, off, off offset:20 ; 4-byte Folded Reload
	ds_read_b64 v[132:133], v132 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[186:187], v162 offset:16384
	scratch_load_dword v162, off, off offset:188 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[184:185], v162 offset:16384
	scratch_load_dword v162, off, off offset:24 ; 4-byte Folded Reload
	ds_read_b64 v[136:137], v136 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[182:183], v162 offset:16384
	scratch_load_dword v162, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[180:181], v162 offset:16384
	scratch_load_dword v162, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[178:179], v162 offset:16384
	scratch_load_dword v162, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[174:175], v162 offset:16384
	scratch_load_dword v162, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[176:177], v162 offset:16384
	scratch_load_dword v162, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[172:173], v162 offset:16384
	scratch_load_dword v162, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v162 offset:16384
	scratch_load_dword v162, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[168:169], v162 offset:16384
	scratch_load_dword v162, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[166:167], v162 offset:16384
	scratch_load_dword v162, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[164:165], v162 offset:16384
	scratch_load_dword v162, off, off offset:192 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[162:163], v162 offset:16384
	ds_write_b128 v219, v[118:121]
	ds_write_b128 v219, v[114:117] offset:8192
	scratch_load_dword v114, off, off offset:240 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[118:121], v114, s[0:3], 0 offen
	s_nop 0
	scratch_load_dword v114, off, off offset:244 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[114:117], v114, s[0:3], 0 offen
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[66:81], v[0:1], v[108:109], v[66:81]
	v_max_f32_e32 v0, v3, v3
	v_max_f32_e32 v1, v2, v2
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v4, v5
	v_max3_f32 v0, v0, v6, v7
	v_max3_f32 v0, v0, v8, v9
	v_max3_f32 v0, v0, v10, v11
	v_mfma_f32_32x32x8_f16 v[50:65], v[202:203], v[108:109], v[50:65]
	v_max3_f32 v0, v0, v12, v13
	v_max3_f32 v0, v0, v14, v15
	v_max3_f32 v0, v0, v16, v17
	v_max3_f32 v0, v0, v82, v83
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_mfma_f32_32x32x8_f16 v[34:49], v[192:193], v[108:109], v[34:49]
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	ds_bpermute_b32 v1, v218, v0
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v227, v226, v0, v1
	v_mfma_f32_32x32x8_f16 v[18:33], v[174:175], v[108:109], v[18:33]
	v_pk_mul_f32 v[228:229], v[226:227], s[16:17] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v0, v2, s16, -v229
	v_fma_f32 v2, v4, s16, -v229
	v_fma_f32 v1, v3, s16, -v229
	v_fma_f32 v3, v5, s16, -v229
	v_fma_f32 v4, v6, s16, -v229
	v_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[106:107], v[66:81]
	v_fma_f32 v5, v7, s16, -v229
	v_fma_f32 v6, v8, s16, -v229
	v_fma_f32 v7, v9, s16, -v229
	v_fma_f32 v8, v10, s16, -v229
	v_fma_f32 v9, v11, s16, -v229
	v_fma_f32 v10, v12, s16, -v229
	v_fma_f32 v11, v13, s16, -v229
	v_mfma_f32_32x32x8_f16 v[50:65], v[204:205], v[106:107], v[50:65]
	v_fma_f32 v12, v14, s16, -v229
	v_fma_f32 v13, v15, s16, -v229
	v_fma_f32 v14, v16, s16, -v229
	v_fma_f32 v15, v17, s16, -v229
	v_fma_f32 v16, v82, s16, -v229
	v_fma_f32 v17, v83, s16, -v229
	v_fma_f32 v82, v84, s16, -v229
	v_mfma_f32_32x32x8_f16 v[34:49], v[188:189], v[106:107], v[34:49]
	v_fma_f32 v83, v85, s16, -v229
	v_fma_f32 v84, v86, s16, -v229
	v_fma_f32 v85, v87, s16, -v229
	v_fma_f32 v86, v88, s16, -v229
	v_fma_f32 v87, v89, s16, -v229
	v_fma_f32 v88, v90, s16, -v229
	v_fma_f32 v89, v91, s16, -v229
	v_mfma_f32_32x32x8_f16 v[18:33], v[176:177], v[106:107], v[18:33]
	v_fma_f32 v90, v92, s16, -v229
	v_fma_f32 v91, v93, s16, -v229
	v_fma_f32 v92, v94, s16, -v229
	v_fma_f32 v93, v95, s16, -v229
	v_fma_f32 v94, v96, s16, -v229
	v_fma_f32 v95, v97, s16, -v229
	v_exp_f32_e32 v251, v2
	v_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[104:105], v[66:81]
	v_sub_f32_e32 v2, v228, v229
	v_exp_f32_e32 v250, v0
	v_exp_f32_e32 v252, v1
	v_exp_f32_e32 v254, v3
	v_exp_f32_e32 v253, v4
	v_exp_f32_e32 v238, v5
	v_exp_f32_e32 v255, v6
	v_mfma_f32_32x32x8_f16 v[50:65], v[206:207], v[104:105], v[50:65]
	v_exp_f32_e32 v240, v7
	v_exp_f32_e32 v239, v8
	v_exp_f32_e32 v230, v9
	v_exp_f32_e32 v241, v10
	v_exp_f32_e32 v1, v11
	v_exp_f32_e32 v231, v12
	v_exp_f32_e32 v234, v13
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[104:105], v[34:49]
	v_exp_f32_e32 v0, v14
	v_exp_f32_e32 v233, v15
	v_exp_f32_e32 v232, v16
	v_exp_f32_e32 v236, v17
	v_exp_f32_e32 v235, v82
	v_exp_f32_e32 v130, v83
	v_exp_f32_e32 v237, v84
	v_mfma_f32_32x32x8_f16 v[18:33], v[172:173], v[104:105], v[18:33]
	v_exp_f32_e32 v132, v85
	v_exp_f32_e32 v131, v86
	v_exp_f32_e32 v133, v87
	v_exp_f32_e32 v242, v88
	v_exp_f32_e32 v243, v89
	v_exp_f32_e32 v244, v90
	v_exp_f32_e32 v246, v91
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[102:103], v[66:81]
	v_exp_f32_e32 v245, v92
	v_exp_f32_e32 v248, v93
	v_exp_f32_e32 v247, v94
	v_exp_f32_e32 v249, v95
	v_exp_f32_e32 v228, v2
	v_mfma_f32_32x32x8_f16 v[50:65], v[208:209], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[102:103], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[170:171], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[194:195], v[100:101], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[210:211], v[100:101], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[100:101], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[168:169], v[100:101], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[196:197], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[212:213], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[166:167], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[198:199], v[112:113], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[214:215], v[112:113], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[112:113], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[164:165], v[112:113], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[200:201], v[110:111], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[216:217], v[110:111], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[110:111], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[162:163], v[110:111], v[18:33]
	; sched_barrier mask(0x00000000)
	s_barrier
	scratch_load_dword v2, off, off offset:196 ; 4-byte Folded Reload
	scratch_load_dword v3, off, off offset:200 ; 4-byte Folded Reload
	scratch_load_dword v4, off, off offset:204 ; 4-byte Folded Reload
	scratch_load_dword v5, off, off offset:208 ; 4-byte Folded Reload
	scratch_load_dword v6, off, off offset:212 ; 4-byte Folded Reload
	scratch_load_dword v7, off, off offset:216 ; 4-byte Folded Reload
	scratch_load_dword v8, off, off offset:220 ; 4-byte Folded Reload
	scratch_load_dword v9, off, off offset:224 ; 4-byte Folded Reload
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
	scratch_load_dword v3, off, off         ; 4-byte Folded Reload
	v_perm_b32 v2, v126, v122, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:4 ; 4-byte Folded Reload
	v_alignbit_b32 v2, v134, v122, 16
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:100 ; 4-byte Folded Reload
	v_perm_b32 v2, v127, v123, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:104 ; 4-byte Folded Reload
	v_perm_b32 v2, v127, v123, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:8 ; 4-byte Folded Reload
	v_perm_b32 v2, v128, v124, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:108 ; 4-byte Folded Reload
	v_perm_b32 v2, v128, v124, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:112 ; 4-byte Folded Reload
	v_perm_b32 v2, v129, v125, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v3, off, off offset:116 ; 4-byte Folded Reload
	v_perm_b32 v2, v129, v125, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2 offset:16384
	scratch_load_dword v2, off, off offset:232 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[122:125], v2, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v2, off, off offset:236 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[126:129], v2, s[12:15], 0 offen
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v134, 16, v126
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	scratch_load_dword v2, off, off offset:268 ; 4-byte Folded Reload
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
	v_add_f32_e32 v134, v250, v252
	v_add_f32_e32 v134, v134, v251
	v_add_f32_e32 v134, v134, v254
	v_add_f32_e32 v134, v134, v253
	v_add_f32_e32 v134, v134, v238
	v_add_f32_e32 v134, v134, v255
	v_add_f32_e32 v134, v134, v240
	v_add_f32_e32 v134, v134, v239
	v_add_f32_e32 v134, v134, v230
	v_add_f32_e32 v134, v134, v241
	v_add_f32_e32 v134, v134, v1
	v_add_f32_e32 v134, v134, v231
	v_add_f32_e32 v134, v134, v234
	v_add_f32_e32 v134, v134, v0
	v_add_f32_e32 v134, v134, v233
	v_add_f32_e32 v134, v134, v232
	v_add_f32_e32 v134, v134, v236
	v_add_f32_e32 v134, v134, v235
	v_add_f32_e32 v134, v134, v130
	v_add_f32_e32 v134, v134, v237
	v_add_f32_e32 v134, v134, v132
	v_add_f32_e32 v134, v134, v131
	v_add_f32_e32 v134, v134, v133
	v_add_f32_e32 v134, v134, v242
	v_add_f32_e32 v134, v134, v243
	v_add_f32_e32 v134, v134, v244
	v_add_f32_e32 v134, v134, v246
	v_add_f32_e32 v134, v134, v245
	v_add_f32_e32 v134, v134, v248
	v_add_f32_e32 v134, v134, v247
	v_add_f32_e32 v134, v134, v249
	v_pk_mul_f32 v[80:81], v[80:81], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[228:229] op_sel_hi:[1,0]
	s_waitcnt vmcnt(0)
	v_cmp_eq_u32_e64 s[0:1], 0, v2
	scratch_load_dword v2, off, off offset:260 ; 4-byte Folded Reload
	scratch_load_dword v3, off, off offset:256 ; 4-byte Folded Reload
	scratch_load_dword v4, off, off offset:264 ; 4-byte Folded Reload
	scratch_load_dwordx4 v[98:101], off, off offset:80 ; 16-byte Folded Reload
	s_mul_i32 s2, s18, 0xc0000
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s8, s2
	s_addc_u32 s5, s9, s3
	s_lshl_b32 s2, s17, 14
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s4, s2
	s_addc_u32 s5, s5, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 2
	s_add_u32 s4, s4, s2
	s_addc_u32 s16, s5, s3
	s_add_i32 s2, s34, 0xffffc100
	s_add_u32 s12, s12, s22
	s_addc_u32 s3, s30, s23
	s_and_b32 s13, s3, 0xffff
	s_mov_b32 s3, 0x5040100
	s_mov_b32 s5, 0x7060302
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_cmp_lt_i32 s2, 1
	s_mov_b32 s2, 0x3e0293ee
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v2, 0xa0, v2
	s_waitcnt vmcnt(1)
	v_or3_b32 v2, v2, v4, v3
	scratch_store_dword off, v2, off offset:240 ; 4-byte Folded Spill
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
	v_mfma_f32_32x32x8_f16 v[82:97], v[222:223], v[158:159], v[2:17]
	v_mfma_f32_32x32x8_f16 v[82:97], v[224:225], v[160:161], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[218:219], v[154:155], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[220:221], v[156:157], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[214:215], v[150:151], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[216:217], v[152:153], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[210:211], v[146:147], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[212:213], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[206:207], v[142:143], v[82:97]
	scratch_load_dword v206, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v136, v206, v134
	v_mfma_f32_32x32x8_f16 v[82:97], v[208:209], v[144:145], v[82:97]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v134, v134, v136
	v_fmac_f32_e32 v134, v135, v228
	scratch_store_dword off, v134, off offset:244 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8_f16 v[82:97], v[202:203], v[138:139], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[204:205], v[140:141], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[198:199], v[98:99], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[200:201], v[100:101], v[82:97]
	v_mov_b64_e32 v[200:201], v[100:101]
	v_mov_b64_e32 v[198:199], v[98:99]
	scratch_load_dwordx4 v[98:101], off, off offset:64 ; 16-byte Folded Reload
	s_barrier
	scratch_load_dword v202, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v204, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_f16 v[82:97], v[194:195], v[98:99], v[82:97]
	scratch_load_dword v226, off, off offset:192 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[196:197], v[100:101], v[82:97]
	v_mov_b64_e32 v[196:197], v[100:101]
	v_mov_b64_e32 v[194:195], v[98:99]
	v_mfma_f32_32x32x8_f16 v[98:113], v[190:191], v[158:159], v[2:17]
	v_mfma_f32_32x32x8_f16 v[98:113], v[192:193], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[186:187], v[154:155], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[188:189], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[182:183], v[150:151], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[184:185], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[178:179], v[146:147], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[180:181], v[148:149], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[174:175], v[142:143], v[98:113]
	v_cvt_pkrtz_f16_f32 v174, v242, v243
	scratch_load_dword v243, off, off offset:120 ; 4-byte Folded Reload
	scratch_load_dword v242, off, off offset:144 ; 4-byte Folded Reload
	v_cvt_pkrtz_f16_f32 v175, v244, v246
	scratch_load_dword v244, off, off offset:128 ; 4-byte Folded Reload
	scratch_load_dword v246, off, off offset:136 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[176:177], v[144:145], v[98:113]
	v_cvt_pkrtz_f16_f32 v176, v245, v248
	scratch_load_dword v248, off, off offset:148 ; 4-byte Folded Reload
	v_cvt_pkrtz_f16_f32 v177, v247, v249
	scratch_load_dword v247, off, off offset:124 ; 4-byte Folded Reload
	scratch_load_dword v245, off, off offset:132 ; 4-byte Folded Reload
	scratch_load_dword v249, off, off offset:156 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[170:171], v[138:139], v[98:113]
	v_cvt_pkrtz_f16_f32 v170, v232, v236
	v_cvt_pkrtz_f16_f32 v171, v235, v130
	v_mfma_f32_32x32x8_f16 v[98:113], v[172:173], v[140:141], v[98:113]
	v_cvt_pkrtz_f16_f32 v172, v237, v132
	v_cvt_pkrtz_f16_f32 v173, v131, v133
	s_waitcnt vmcnt(8)
	ds_read_b64 v[236:237], v226 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[166:167], v[198:199], v[98:113]
	v_cvt_pkrtz_f16_f32 v166, v239, v230
	scratch_load_dword v239, off, off offset:172 ; 4-byte Folded Reload
	v_cvt_pkrtz_f16_f32 v167, v241, v1
	scratch_load_dword v241, off, off offset:140 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[168:169], v[200:201], v[98:113]
	v_cvt_pkrtz_f16_f32 v169, v0, v233
	v_cvt_pkrtz_f16_f32 v168, v231, v234
	v_mfma_f32_32x32x8_f16 v[98:113], v[162:163], v[194:195], v[98:113]
	v_cvt_pkrtz_f16_f32 v162, v250, v252
	scratch_load_dword v252, off, off offset:164 ; 4-byte Folded Reload
	scratch_load_dword v207, off, off offset:20 ; 4-byte Folded Reload
	ds_read_b64 v[202:203], v202 offset:16384
	ds_read_b64 v[204:205], v204 offset:16384
	v_cvt_pkrtz_f16_f32 v163, v251, v254
	scratch_load_dword v254, off, off offset:180 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[164:165], v[196:197], v[98:113]
	v_cvt_pkrtz_f16_f32 v164, v253, v238
	scratch_load_dword v253, off, off offset:168 ; 4-byte Folded Reload
	scratch_load_dword v238, off, off offset:188 ; 4-byte Folded Reload
	v_cvt_pkrtz_f16_f32 v165, v255, v240
	scratch_load_dword v240, off, off offset:176 ; 4-byte Folded Reload
	scratch_load_dword v255, off, off offset:184 ; 4-byte Folded Reload
	scratch_load_dword v250, off, off offset:152 ; 4-byte Folded Reload
	scratch_load_dword v251, off, off offset:160 ; 4-byte Folded Reload
	s_waitcnt vmcnt(18)
	ds_read_b64 v[0:1], v243 offset:16384
	s_waitcnt vmcnt(10)
	ds_read_b64 v[194:195], v239 offset:16384
	s_waitcnt vmcnt(8)
	ds_read_b64 v[190:191], v252 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[208:209], v207 offset:16384
	scratch_load_dword v207, off, off offset:24 ; 4-byte Folded Reload
	ds_read_b64 v[130:131], v247 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[198:199], v254 offset:16384
	s_waitcnt vmcnt(6)
	ds_read_b64 v[192:193], v253 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[210:211], v238 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[196:197], v240 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[200:201], v255 offset:16384
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[200:201], v[162:163], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v207 offset:16384
	scratch_load_dword v207, off, off offset:28 ; 4-byte Folded Reload
	ds_read_b64 v[132:133], v244 offset:16384
	v_mfma_f32_32x32x8_f16 v[34:49], v[202:203], v[164:165], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v207 offset:16384
	scratch_load_dword v207, off, off offset:32 ; 4-byte Folded Reload
	ds_read_b64 v[134:135], v245 offset:16384
	v_mfma_f32_32x32x8_f16 v[34:49], v[204:205], v[166:167], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[216:217], v207 offset:16384
	scratch_load_dword v207, off, off offset:36 ; 4-byte Folded Reload
	ds_read_b64 v[136:137], v246 offset:16384
	v_mfma_f32_32x32x8_f16 v[34:49], v[208:209], v[168:169], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[218:219], v207 offset:16384
	scratch_load_dword v207, off, off offset:40 ; 4-byte Folded Reload
	ds_read_b64 v[178:179], v241 offset:16384
	v_mfma_f32_32x32x8_f16 v[66:81], v[0:1], v[162:163], v[66:81]
	v_max_f32_e32 v0, v83, v83
	v_max_f32_e32 v1, v82, v82
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_max3_f32 v0, v0, v98, v99
	v_max3_f32 v0, v0, v100, v101
	v_max3_f32 v0, v0, v102, v103
	v_max3_f32 v0, v0, v104, v105
	v_max3_f32 v0, v0, v106, v107
	v_max3_f32 v0, v0, v108, v109
	v_max3_f32 v0, v0, v110, v111
	v_max3_f32 v0, v0, v112, v113
	ds_bpermute_b32 v1, v206, v0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[218:219], v[162:163], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v202, v227, v0, v1
	s_waitcnt vmcnt(0)
	ds_read_b64 v[220:221], v207 offset:16384
	scratch_load_dword v207, off, off offset:44 ; 4-byte Folded Reload
	ds_read_b64 v[180:181], v242 offset:16384
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[18:33], v[220:221], v[164:165], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[222:223], v207 offset:16384
	scratch_load_dword v207, off, off offset:48 ; 4-byte Folded Reload
	ds_read_b64 v[182:183], v248 offset:16384
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[18:33], v[222:223], v[166:167], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[224:225], v207 offset:16384
	scratch_load_dword v207, off, off offset:52 ; 4-byte Folded Reload
	ds_read_b64 v[184:185], v250 offset:16384
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[184:185], v[162:163], v[50:65]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[230:231], v207 offset:16384
	scratch_load_dword v207, off, off offset:56 ; 4-byte Folded Reload
	ds_read_b64 v[186:187], v249 offset:16384
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[186:187], v[164:165], v[50:65]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[232:233], v207 offset:16384
	scratch_load_dword v207, off, off offset:60 ; 4-byte Folded Reload
	ds_read_b64 v[188:189], v251 offset:16384
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[188:189], v[166:167], v[50:65]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[234:235], v207 offset:16384
	scratch_load_dword v207, off, off offset:228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_write_b128 v207, v[118:121]
	ds_write_b128 v207, v[114:117] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v0, off, off offset:196 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:200 ; 4-byte Folded Reload
	scratch_load_dword v114, off, off offset:204 ; 4-byte Folded Reload
	scratch_load_dword v115, off, off offset:208 ; 4-byte Folded Reload
	scratch_load_dword v116, off, off offset:212 ; 4-byte Folded Reload
	scratch_load_dword v117, off, off offset:216 ; 4-byte Folded Reload
	scratch_load_dword v118, off, off offset:220 ; 4-byte Folded Reload
	scratch_load_dword v119, off, off offset:224 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[164:165], v[66:81]
	s_waitcnt vmcnt(2)
	ds_read_b128 v[220:223], v117
	v_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[166:167], v[66:81]
	ds_read_b128 v[130:133], v0
	v_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[168:169], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[190:191], v[168:169], v[50:65]
	v_mfma_f32_32x32x8_f16 v[18:33], v[224:225], v[168:169], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[170:171], v[66:81]
	ds_read_b128 v[134:137], v1
	v_mfma_f32_32x32x8_f16 v[50:65], v[192:193], v[170:171], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[210:211], v[170:171], v[34:49]
	ds_read_b128 v[208:211], v114
	v_mfma_f32_32x32x8_f16 v[18:33], v[230:231], v[170:171], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[172:173], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[194:195], v[172:173], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[212:213], v[172:173], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[232:233], v[172:173], v[18:33]
	s_waitcnt vmcnt(1)
	ds_read_b128 v[230:233], v118
	v_mfma_f32_32x32x8_f16 v[66:81], v[180:181], v[174:175], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[196:197], v[174:175], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[214:215], v[174:175], v[34:49]
	ds_read_b128 v[212:215], v115
	v_mfma_f32_32x32x8_f16 v[18:33], v[234:235], v[174:175], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[176:177], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[198:199], v[176:177], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[216:217], v[176:177], v[34:49]
	ds_read_b128 v[216:219], v116
	v_mfma_f32_32x32x8_f16 v[18:33], v[236:237], v[176:177], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b128 v[234:237], v119
	ds_read_b128 v[198:201], v0 offset:8192
	ds_read_b128 v[194:197], v1 offset:8192
	ds_read_b128 v[190:193], v114 offset:8192
	ds_read_b128 v[186:189], v115 offset:8192
	ds_read_b128 v[182:185], v116 offset:8192
	ds_read_b128 v[178:181], v117 offset:8192
	ds_read_b128 v[174:177], v118 offset:8192
	ds_read_b128 v[170:173], v119 offset:8192
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	scratch_load_dword v227, off, off offset:100 ; 4-byte Folded Reload
	scratch_load_dword v224, off, off offset:112 ; 4-byte Folded Reload
	v_perm_b32 v0, v126, v122, s3
	scratch_load_dword v228, off, off offset:104 ; 4-byte Folded Reload
	scratch_load_dword v225, off, off offset:116 ; 4-byte Folded Reload
	scratch_load_dword v203, off, off offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(5)
	ds_write_b32 v1, v0 offset:16384
	scratch_load_dword v1, off, off offset:4 ; 4-byte Folded Reload
	v_perm_b32 v0, v126, v122, s5
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v0 offset:16384
	scratch_load_dword v1, off, off offset:8 ; 4-byte Folded Reload
	v_perm_b32 v0, v127, v123, s3
	ds_write_b32 v227, v0 offset:16384
	v_perm_b32 v0, v127, v123, s5
	ds_write_b32 v228, v0 offset:16384
	v_perm_b32 v0, v128, v124, s3
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v0 offset:16384
	v_perm_b32 v0, v128, v124, s5
	ds_write_b32 v203, v0 offset:16384
	v_perm_b32 v0, v129, v125, s3
	ds_write_b32 v224, v0 offset:16384
	v_perm_b32 v0, v129, v125, s5
	ds_write_b32 v225, v0 offset:16384
	scratch_load_dword v0, off, off offset:232 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[114:129], v[130:131], v[158:159], v[2:17]
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[162:165], v0, s[12:15], 0 offen
	v_mfma_f32_32x32x8_f16 v[114:129], v[132:133], v[160:161], v[114:129]
	scratch_load_dwordx4 v[130:133], off, off offset:64 ; 16-byte Folded Reload
	scratch_load_dword v0, off, off offset:236 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[166:169], v0, s[12:15], 0 offen
	v_mfma_f32_32x32x8_f16 v[2:17], v[198:199], v[158:159], v[2:17]
	v_mfma_f32_32x32x8_f16 v[114:129], v[134:135], v[154:155], v[114:129]
	v_mfma_f32_32x32x8_f16 v[2:17], v[200:201], v[160:161], v[2:17]
	v_mfma_f32_32x32x8_f16 v[114:129], v[136:137], v[156:157], v[114:129]
	scratch_load_dwordx4 v[134:137], off, off offset:80 ; 16-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[2:17], v[194:195], v[154:155], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[196:197], v[156:157], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[190:191], v[150:151], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[192:193], v[152:153], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[186:187], v[146:147], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[188:189], v[148:149], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[182:183], v[142:143], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[184:185], v[144:145], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[178:179], v[138:139], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[180:181], v[140:141], v[2:17]
	v_mfma_f32_32x32x8_f16 v[114:129], v[208:209], v[150:151], v[114:129]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[174:175], v[134:135], v[2:17]
	v_mfma_f32_32x32x8_f16 v[114:129], v[210:211], v[152:153], v[114:129]
	v_mov_b32_e32 v210, v202
	v_mfma_f32_32x32x8_f16 v[2:17], v[176:177], v[136:137], v[2:17]
	v_mfma_f32_32x32x8_f16 v[114:129], v[212:213], v[146:147], v[114:129]
	v_mfma_f32_32x32x8_f16 v[2:17], v[170:171], v[130:131], v[2:17]
	v_mfma_f32_32x32x8_f16 v[114:129], v[214:215], v[148:149], v[114:129]
	v_mfma_f32_32x32x8_f16 v[2:17], v[172:173], v[132:133], v[2:17]
	ds_read_b64 v[204:205], v243 offset:16384
	ds_read_b64 v[200:201], v247 offset:16384
	ds_read_b64 v[198:199], v244 offset:16384
	ds_read_b64 v[188:189], v245 offset:16384
	ds_read_b64 v[190:191], v246 offset:16384
	ds_read_b64 v[192:193], v241 offset:16384
	ds_read_b64 v[194:195], v242 offset:16384
	ds_read_b64 v[196:197], v248 offset:16384
	ds_read_b64 v[180:181], v250 offset:16384
	ds_read_b64 v[182:183], v249 offset:16384
	ds_read_b64 v[184:185], v251 offset:16384
	ds_read_b64 v[186:187], v252 offset:16384
	ds_read_b64 v[172:173], v253 offset:16384
	ds_read_b64 v[174:175], v239 offset:16384
	ds_read_b64 v[176:177], v240 offset:16384
	ds_read_b64 v[178:179], v254 offset:16384
	ds_read_b64 v[156:157], v255 offset:16384
	scratch_load_dword v212, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v213, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v214, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	ds_read_b64 v[158:159], v212 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[160:161], v213 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[216:217], v[142:143], v[114:129]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v214 offset:16384
	ds_read_b64 v[148:149], v238 offset:16384
	scratch_load_dword v215, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v216, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v217, off, off offset:32 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[114:129], v[218:219], v[144:145], v[114:129]
	scratch_load_dword v218, off, off offset:36 ; 4-byte Folded Reload
	scratch_load_dword v219, off, off offset:40 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[114:129], v[220:221], v[138:139], v[114:129]
	scratch_load_dword v220, off, off offset:44 ; 4-byte Folded Reload
	scratch_load_dword v221, off, off offset:48 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[114:129], v[222:223], v[140:141], v[114:129]
	scratch_load_dword v222, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v223, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(8)
	ds_read_b64 v[150:151], v215 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[152:153], v216 offset:16384
	s_waitcnt vmcnt(6)
	ds_read_b64 v[154:155], v217 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[230:231], v[134:135], v[114:129]
	scratch_load_dword v230, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(5)
	ds_read_b64 v[134:135], v219 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[138:139], v221 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[232:233], v[136:137], v[114:129]
	ds_read_b64 v[136:137], v220 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[140:141], v222 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[142:143], v223 offset:16384
	v_mfma_f32_32x32x8_f16 v[114:129], v[234:235], v[130:131], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[236:237], v[132:133], v[114:129]
	ds_read_b64 v[132:133], v218 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[144:145], v230 offset:16384
	ds_read_b64 v[146:147], v226 offset:16384
	s_nop 6
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
	v_max3_f32 v0, v0, v2, v3
	v_max3_f32 v0, v0, v4, v5
	v_max3_f32 v0, v0, v6, v7
	v_max3_f32 v0, v0, v8, v9
	v_max3_f32 v0, v0, v10, v11
	v_max3_f32 v0, v0, v12, v13
	v_max3_f32 v0, v0, v14, v15
	v_max3_f32 v0, v0, v16, v17
	ds_bpermute_b32 v1, v206, v0
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v211, v210, v0, v1
	v_pk_mul_f32 v[130:131], v[210:211], s[2:3] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v0, v82, s2, -v130
	v_fma_f32 v1, v83, s2, -v130
	v_fma_f32 v82, v84, s2, -v130
	v_fma_f32 v84, v86, s2, -v130
	v_fma_f32 v86, v88, s2, -v130
	v_fma_f32 v88, v90, s2, -v130
	v_fma_f32 v90, v92, s2, -v130
	v_fma_f32 v92, v94, s2, -v130
	v_fma_f32 v94, v96, s2, -v130
	v_fma_f32 v96, v98, s2, -v130
	v_fma_f32 v98, v100, s2, -v130
	v_fma_f32 v100, v102, s2, -v130
	v_fma_f32 v102, v104, s2, -v130
	v_fma_f32 v104, v106, s2, -v130
	v_fma_f32 v106, v108, s2, -v130
	v_fma_f32 v108, v110, s2, -v130
	v_fma_f32 v110, v112, s2, -v130
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v112, v1
	v_fma_f32 v83, v85, s2, -v130
	v_fma_f32 v85, v87, s2, -v130
	v_fma_f32 v87, v89, s2, -v130
	v_fma_f32 v89, v91, s2, -v130
	v_fma_f32 v91, v93, s2, -v130
	v_fma_f32 v93, v95, s2, -v130
	v_fma_f32 v95, v97, s2, -v130
	v_fma_f32 v97, v99, s2, -v130
	v_fma_f32 v99, v101, s2, -v130
	v_fma_f32 v101, v103, s2, -v130
	v_fma_f32 v103, v105, s2, -v130
	v_fma_f32 v105, v107, s2, -v130
	v_fma_f32 v107, v109, s2, -v130
	v_fma_f32 v109, v111, s2, -v130
	v_fma_f32 v111, v113, s2, -v130
	v_exp_f32_e32 v113, v82
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v202, v84
	v_sub_f32_e32 v1, v229, v130
	v_exp_f32_e32 v207, v85
	v_exp_f32_e32 v82, v1
	v_add_f32_e32 v1, v0, v112
	v_exp_f32_e32 v208, v86
	v_add_f32_e32 v1, v113, v1
	v_exp_f32_e32 v87, v87
	v_add_f32_e32 v1, v83, v1
	v_exp_f32_e32 v88, v88
	v_add_f32_e32 v1, v202, v1
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v1, v207, v1
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v1, v208, v1
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
	v_exp_f32_e32 v96, v96
	v_add_f32_e32 v1, v92, v1
	v_exp_f32_e32 v97, v97
	v_add_f32_e32 v1, v93, v1
	v_exp_f32_e32 v98, v98
	v_add_f32_e32 v1, v94, v1
	v_exp_f32_e32 v99, v99
	v_add_f32_e32 v1, v95, v1
	v_exp_f32_e32 v100, v100
	v_add_f32_e32 v1, v96, v1
	v_exp_f32_e32 v101, v101
	v_add_f32_e32 v1, v97, v1
	v_exp_f32_e32 v102, v102
	v_add_f32_e32 v1, v98, v1
	v_exp_f32_e32 v103, v103
	v_add_f32_e32 v1, v99, v1
	v_exp_f32_e32 v104, v104
	v_add_f32_e32 v1, v100, v1
	v_exp_f32_e32 v105, v105
	v_add_f32_e32 v1, v101, v1
	v_exp_f32_e32 v106, v106
	v_add_f32_e32 v1, v102, v1
	v_exp_f32_e32 v107, v107
	v_add_f32_e32 v1, v103, v1
	v_exp_f32_e32 v108, v108
	v_add_f32_e32 v1, v104, v1
	v_exp_f32_e32 v109, v109
	v_add_f32_e32 v1, v105, v1
	v_exp_f32_e32 v110, v110
	v_add_f32_e32 v1, v106, v1
	v_exp_f32_e32 v111, v111
	v_add_f32_e32 v1, v107, v1
	v_add_f32_e32 v1, v108, v1
	v_add_f32_e32 v1, v109, v1
	v_add_f32_e32 v1, v110, v1
	v_add_f32_e32 v1, v111, v1
	ds_bpermute_b32 v84, v206, v1
	v_cvt_pkrtz_f16_f32 v85, v113, v83
	v_pk_mul_f32 v[80:81], v[80:81], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[82:83] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v84
	scratch_load_dword v84, off, off offset:244 ; 4-byte Folded Reload
	v_pk_mul_f32 v[74:75], v[74:75], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[82:83] op_sel_hi:[1,0]
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
	v_cvt_pkrtz_f16_f32 v86, v202, v207
	v_cvt_pkrtz_f16_f32 v87, v208, v87
	v_cvt_pkrtz_f16_f32 v88, v88, v89
	v_cvt_pkrtz_f16_f32 v89, v90, v91
	v_cvt_pkrtz_f16_f32 v90, v92, v93
	v_cvt_pkrtz_f16_f32 v91, v94, v95
	v_cvt_pkrtz_f16_f32 v92, v96, v97
	v_cvt_pkrtz_f16_f32 v93, v98, v99
	v_cvt_pkrtz_f16_f32 v94, v100, v101
	v_cvt_pkrtz_f16_f32 v95, v102, v103
	v_cvt_pkrtz_f16_f32 v96, v104, v105
	v_cvt_pkrtz_f16_f32 v97, v106, v107
	v_cvt_pkrtz_f16_f32 v98, v108, v109
	v_cvt_pkrtz_f16_f32 v99, v110, v111
	s_barrier
	v_fma_f32 v83, v116, s2, -v131
	v_exp_f32_e32 v83, v83
	v_fma_f32 v2, v2, s2, -v131
	v_fma_f32 v3, v3, s2, -v131
	v_exp_f32_e32 v2, v2
	v_fma_f32 v4, v4, s2, -v131
	v_exp_f32_e32 v3, v3
	v_fma_f32 v5, v5, s2, -v131
	v_exp_f32_e32 v4, v4
	v_fma_f32 v6, v6, s2, -v131
	v_exp_f32_e32 v5, v5
	v_fma_f32 v7, v7, s2, -v131
	v_exp_f32_e32 v6, v6
	v_fma_f32 v8, v8, s2, -v131
	v_exp_f32_e32 v7, v7
	v_fma_f32 v9, v9, s2, -v131
	v_exp_f32_e32 v8, v8
	v_fma_f32 v10, v10, s2, -v131
	v_exp_f32_e32 v9, v9
	v_fma_f32 v11, v11, s2, -v131
	v_exp_f32_e32 v10, v10
	v_fma_f32 v12, v12, s2, -v131
	v_exp_f32_e32 v11, v11
	v_fma_f32 v13, v13, s2, -v131
	v_exp_f32_e32 v12, v12
	v_fma_f32 v14, v14, s2, -v131
	v_exp_f32_e32 v13, v13
	v_fma_f32 v15, v15, s2, -v131
	v_exp_f32_e32 v14, v14
	v_fma_f32 v16, v16, s2, -v131
	v_exp_f32_e32 v15, v15
	v_fma_f32 v17, v17, s2, -v131
	v_exp_f32_e32 v16, v16
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v1, v84, v82
	v_cvt_pkrtz_f16_f32 v84, v0, v112
	v_fma_f32 v0, v114, s2, -v131
	v_fma_f32 v82, v115, s2, -v131
	v_mfma_f32_32x32x8_f16 v[66:81], v[204:205], v[84:85], v[66:81]
	v_exp_f32_e32 v17, v17
	v_mfma_f32_32x32x8_f16 v[50:65], v[180:181], v[84:85], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[156:157], v[84:85], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[132:133], v[84:85], v[18:33]
	v_fma_f32 v84, v117, s2, -v131
	v_fma_f32 v85, v118, s2, -v131
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v85, v85
	v_cvt_pkrtz_f16_f32 v107, v83, v84
	v_mfma_f32_32x32x8_f16 v[66:81], v[200:201], v[86:87], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[182:183], v[86:87], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[158:159], v[86:87], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[134:135], v[86:87], v[18:33]
	v_fma_f32 v86, v119, s2, -v131
	v_fma_f32 v87, v120, s2, -v131
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_cvt_pkrtz_f16_f32 v102, v85, v86
	v_mfma_f32_32x32x8_f16 v[66:81], v[198:199], v[88:89], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[184:185], v[88:89], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[160:161], v[88:89], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[136:137], v[88:89], v[18:33]
	v_fma_f32 v88, v121, s2, -v131
	v_fma_f32 v89, v122, s2, -v131
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	v_cvt_pkrtz_f16_f32 v103, v87, v88
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[90:91], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[186:187], v[90:91], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[90:91], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[138:139], v[90:91], v[18:33]
	v_fma_f32 v90, v123, s2, -v131
	v_fma_f32 v91, v124, s2, -v131
	v_exp_f32_e32 v90, v90
	v_exp_f32_e32 v91, v91
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[92:93], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[92:93], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[148:149], v[92:93], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[140:141], v[92:93], v[18:33]
	v_fma_f32 v92, v125, s2, -v131
	v_fma_f32 v93, v126, s2, -v131
	v_exp_f32_e32 v92, v92
	v_exp_f32_e32 v93, v93
	v_mfma_f32_32x32x8_f16 v[66:81], v[192:193], v[94:95], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[94:95], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[150:151], v[94:95], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[142:143], v[94:95], v[18:33]
	v_fma_f32 v94, v127, s2, -v131
	v_fma_f32 v95, v128, s2, -v131
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_mfma_f32_32x32x8_f16 v[66:81], v[194:195], v[96:97], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[96:97], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[152:153], v[96:97], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[144:145], v[96:97], v[18:33]
	v_exp_f32_e32 v97, v0
	v_sub_f32_e32 v0, v130, v131
	v_fma_f32 v96, v129, s2, -v131
	v_exp_f32_e32 v96, v96
	v_mfma_f32_32x32x8_f16 v[66:81], v[196:197], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[178:179], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[154:155], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[146:147], v[98:99], v[18:33]
	scratch_load_dword v99, off, off        ; 4-byte Folded Reload
	v_exp_f32_e32 v98, v82
	v_exp_f32_e32 v82, v0
	v_perm_b32 v0, v166, v162, s3
	v_cvt_pkrtz_f16_f32 v106, v97, v98
	s_nop 3
	v_pk_mul_f32 v[64:65], v[64:65], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[82:83] op_sel_hi:[1,0]
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
	s_waitcnt vmcnt(0)
	ds_write_b32 v99, v0 offset:16384
	scratch_load_dword v99, off, off offset:4 ; 4-byte Folded Reload
	v_perm_b32 v0, v166, v162, s5
	s_waitcnt vmcnt(0)
	ds_write_b32 v99, v0 offset:16384
	scratch_load_dword v99, off, off offset:8 ; 4-byte Folded Reload
	v_perm_b32 v0, v167, v163, s3
	ds_write_b32 v227, v0 offset:16384
	v_perm_b32 v0, v167, v163, s5
	ds_write_b32 v228, v0 offset:16384
	v_perm_b32 v0, v168, v164, s3
	s_waitcnt vmcnt(0)
	ds_write_b32 v99, v0 offset:16384
	v_perm_b32 v0, v168, v164, s5
	ds_write_b32 v203, v0 offset:16384
	v_perm_b32 v0, v169, v165, s3
	ds_write_b32 v224, v0 offset:16384
	v_perm_b32 v0, v169, v165, s5
	ds_write_b32 v225, v0 offset:16384
	v_add_f32_e32 v0, v97, v98
	v_add_f32_e32 v0, v83, v0
	v_add_f32_e32 v0, v84, v0
	v_add_f32_e32 v0, v85, v0
	v_add_f32_e32 v0, v86, v0
	v_add_f32_e32 v0, v87, v0
	v_add_f32_e32 v0, v88, v0
	v_add_f32_e32 v0, v89, v0
	v_add_f32_e32 v0, v90, v0
	v_add_f32_e32 v0, v91, v0
	v_add_f32_e32 v0, v92, v0
	v_add_f32_e32 v0, v93, v0
	v_add_f32_e32 v0, v94, v0
	v_add_f32_e32 v0, v95, v0
	v_add_f32_e32 v0, v96, v0
	v_add_f32_e32 v0, v2, v0
	v_add_f32_e32 v0, v3, v0
	v_add_f32_e32 v0, v4, v0
	v_add_f32_e32 v0, v5, v0
	v_add_f32_e32 v0, v6, v0
	v_add_f32_e32 v0, v7, v0
	v_add_f32_e32 v0, v8, v0
	v_add_f32_e32 v0, v9, v0
	v_add_f32_e32 v0, v10, v0
	v_add_f32_e32 v0, v11, v0
	v_add_f32_e32 v0, v12, v0
	v_add_f32_e32 v0, v13, v0
	v_add_f32_e32 v0, v14, v0
	v_add_f32_e32 v0, v15, v0
	v_add_f32_e32 v0, v16, v0
	v_add_f32_e32 v0, v17, v0
	ds_bpermute_b32 v99, v206, v0
	v_cvt_pkrtz_f16_f32 v88, v6, v7
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_f32_e32 v0, v0, v99
	v_cvt_pkrtz_f16_f32 v99, v91, v92
	v_cvt_pkrtz_f16_f32 v92, v93, v94
	v_cvt_pkrtz_f16_f32 v93, v95, v96
	ds_read_b64 v[156:157], v243 offset:16384
	ds_read_b64 v[158:159], v247 offset:16384
	ds_read_b64 v[160:161], v244 offset:16384
	ds_read_b64 v[162:163], v245 offset:16384
	ds_read_b64 v[154:155], v246 offset:16384
	ds_read_b64 v[152:153], v241 offset:16384
	ds_read_b64 v[150:151], v242 offset:16384
	ds_read_b64 v[148:149], v248 offset:16384
	ds_read_b64 v[146:147], v250 offset:16384
	ds_read_b64 v[144:145], v249 offset:16384
	ds_read_b64 v[142:143], v251 offset:16384
	ds_read_b64 v[140:141], v252 offset:16384
	ds_read_b64 v[138:139], v253 offset:16384
	ds_read_b64 v[136:137], v239 offset:16384
	ds_read_b64 v[132:133], v240 offset:16384
	ds_read_b64 v[134:135], v254 offset:16384
	ds_read_b64 v[130:131], v255 offset:16384
	ds_read_b64 v[128:129], v212 offset:16384
	ds_read_b64 v[126:127], v213 offset:16384
	ds_read_b64 v[124:125], v214 offset:16384
	ds_read_b64 v[122:123], v238 offset:16384
	ds_read_b64 v[120:121], v215 offset:16384
	ds_read_b64 v[118:119], v216 offset:16384
	ds_read_b64 v[116:117], v217 offset:16384
	ds_read_b64 v[114:115], v218 offset:16384
	ds_read_b64 v[112:113], v219 offset:16384
	ds_read_b64 v[110:111], v220 offset:16384
	ds_read_b64 v[108:109], v221 offset:16384
	ds_read_b64 v[104:105], v222 offset:16384
	ds_read_b64 v[100:101], v223 offset:16384
	ds_read_b64 v[94:95], v230 offset:16384
	ds_read_b64 v[96:97], v226 offset:16384
	v_pk_mul_f32 v[6:7], v[70:71], v[82:83] op_sel_hi:[1,0]
	scratch_load_dword v70, off, off offset:240 ; 4-byte Folded Reload
	v_cvt_pkrtz_f16_f32 v98, v89, v90
	v_cvt_pkrtz_f16_f32 v90, v2, v3
	v_cvt_pkrtz_f16_f32 v91, v4, v5
	v_cvt_pkrtz_f16_f32 v89, v8, v9
	v_cvt_pkrtz_f16_f32 v84, v10, v11
	v_cvt_pkrtz_f16_f32 v85, v12, v13
	v_cvt_pkrtz_f16_f32 v86, v14, v15
	v_cvt_pkrtz_f16_f32 v87, v16, v17
	v_pk_mul_f32 v[16:17], v[80:81], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[78:79], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[76:77], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[74:75], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[72:73], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[68:69], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[66:67], v[82:83] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[50:65], v[146:147], v[106:107], v[50:65]
	v_fmac_f32_e32 v0, v1, v82
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v70, 2, 0
	v_mfma_f32_32x32x8_f16 v[2:17], v[156:157], v[106:107], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[130:131], v[106:107], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[114:115], v[106:107], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[158:159], v[102:103], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[144:145], v[102:103], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[128:129], v[102:103], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[112:113], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[160:161], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[142:143], v[98:99], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[126:127], v[98:99], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[110:111], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[162:163], v[92:93], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[140:141], v[92:93], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[124:125], v[92:93], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[92:93], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[154:155], v[90:91], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[138:139], v[90:91], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[122:123], v[90:91], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[104:105], v[90:91], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[152:153], v[88:89], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[136:137], v[88:89], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[120:121], v[88:89], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[100:101], v[88:89], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[150:151], v[84:85], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[132:133], v[84:85], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[118:119], v[84:85], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[94:95], v[84:85], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[148:149], v[86:87], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[134:135], v[86:87], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[116:117], v[86:87], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[96:97], v[86:87], v[18:33]
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	scratch_load_dword v69, off, off offset:248 ; 4-byte Folded Reload
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
	v_add_f32_e32 v66, v211, v66
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
	v_add_f32_e32 v66, v211, v66
	ds_write_b32 v1, v66
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v66, off, off offset:248 ; 4-byte Folded Reload
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
	scratch_load_dword v3, off, off offset:252 ; 4-byte Folded Reload
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
		.amdhsa_private_segment_fixed_size 276
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
	.set attn_fwd.private_seg_size, 276
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16164
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 276
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
    .private_segment_fixed_size: 276
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 70
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
