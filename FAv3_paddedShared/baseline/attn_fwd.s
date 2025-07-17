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
	s_mul_i32 s24, s12, s18
	s_ashr_i32 s25, s24, 31
	s_lshl_b32 s28, s16, 8
	s_lshl_b64 s[24:25], s[24:25], 1
	s_add_u32 s12, s2, s24
	s_mul_i32 s2, s13, s17
	s_addc_u32 s16, s3, s25
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s28
	s_addc_u32 s13, s16, s3
	s_ashr_i32 s3, s2, 31
	s_load_dwordx4 s[20:23], s[0:1], 0x38
	s_load_dword s29, s[0:1], 0x48
	s_lshl_b32 s19, s14, 5
	s_lshl_b64 s[2:3], s[2:3], 1
	v_and_b32_e32 v69, 15, v0
	v_lshrrev_b32_e32 v1, 4, v0
	s_add_u32 s12, s12, s2
	v_lshlrev_b32_e32 v58, 3, v69
	s_mul_i32 s30, s15, s18
	s_addc_u32 s13, s13, s3
	v_mad_u64_u32 v[2:3], s[2:3], s14, v1, v[58:59]
	s_ashr_i32 s31, s30, 31
	s_lshl_b64 s[2:3], s[30:31], 1
	s_add_u32 s15, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s34, s20, s17
	s_addc_u32 s16, s5, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 1
	s_add_u32 s24, s15, s2
	s_mul_i32 s2, s22, s18
	s_addc_u32 s33, s16, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s36, s21, 5
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s23, s17
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	v_or_b32_e32 v4, s28, v1
	s_add_u32 s20, s6, s2
	s_movk_i32 s2, 0x4000
	v_or_b32_e32 v5, 32, v4
	v_add_u32_e32 v3, s19, v2
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v29, 1
	v_cmp_gt_i32_e32 vcc, s2, v4
	v_or_b32_e32 v10, 64, v4
	v_add_u32_e32 v12, s19, v3
	v_cndmask_b32_e32 v14, v29, v2, vcc
	v_lshlrev_b32_e32 v2, 1, v3
	v_cmp_gt_i32_e32 vcc, s2, v5
	v_or_b32_e32 v11, 0x60, v4
	v_add_u32_e32 v13, s19, v12
	s_addc_u32 s16, s7, s3
	s_and_b32 s3, s14, 0x3fff
	v_cndmask_b32_e32 v15, v29, v2, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	v_cmp_gt_i32_e32 vcc, s2, v10
	v_or_b32_e32 v18, 0x80, v4
	v_add_u32_e32 v20, s19, v13
	s_bitset1_b32 s3, 14
	v_cndmask_b32_e32 v22, v29, v12, vcc
	v_lshlrev_b32_e32 v10, 1, v13
	v_cmp_gt_i32_e32 vcc, s2, v11
	v_or_b32_e32 v19, 0xa0, v4
	v_add_u32_e32 v21, s19, v20
	s_and_b32 s6, s13, 0xffff
	s_lshl_b32 s3, s3, 16
	v_cndmask_b32_e32 v23, v29, v10, vcc
	v_lshlrev_b32_e32 v20, 1, v20
	v_cmp_gt_i32_e32 vcc, s2, v18
	v_or_b32_e32 v26, 0xc0, v4
	v_add_u32_e32 v28, s19, v21
	s_or_b32 s13, s6, s3
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_cndmask_b32_e32 v30, v29, v20, vcc
	v_lshlrev_b32_e32 v18, 1, v21
	v_cmp_gt_i32_e32 vcc, s2, v19
	v_or_b32_e32 v27, 0xe0, v4
	buffer_load_dwordx4 v[2:5], v14, s[12:15], 0 offen
	buffer_load_dwordx4 v[6:9], v15, s[12:15], 0 offen
	buffer_load_dwordx4 v[10:13], v22, s[12:15], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[14:17], v23, s[12:15], 0 offen
	v_cndmask_b32_e32 v31, v29, v18, vcc
	buffer_load_dwordx4 v[18:21], v30, s[12:15], 0 offen
	buffer_load_dwordx4 v[22:25], v31, s[12:15], 0 offen
	v_lshlrev_b32_e32 v30, 1, v28
	v_cmp_gt_i32_e32 vcc, s2, v26
	v_add_lshl_u32 v26, v28, s19, 1
	s_mov_b32 s26, s14
	v_cndmask_b32_e32 v34, v29, v30, vcc
	v_cmp_gt_i32_e32 vcc, s2, v27
	s_mov_b32 s27, s15
	s_lshl_b32 s52, s21, 6
	v_cndmask_b32_e32 v35, v29, v26, vcc
	buffer_load_dwordx4 v[26:29], v34, s[12:15], 0 offen
	buffer_load_dwordx4 v[30:33], v35, s[12:15], 0 offen
	v_mad_u64_u32 v[34:35], s[2:3], s21, v1, v[58:59]
	s_and_b32 s2, s21, 0x3fff
	v_lshlrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v35, 0xf0, v0
	s_bitset1_b32 s2, 14
	v_xor_b32_e32 v1, v1, v35
	s_and_b32 s3, s33, 0xffff
	s_lshl_b32 s19, s2, 16
	v_add_u32_e32 v71, 0, v1
	s_or_b32 s25, s3, s19
	v_lshlrev_b32_e32 v170, 1, v34
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v71, v[2:5]
	s_waitcnt vmcnt(6)
	ds_write_b128 v71, v[6:9] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v71, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v71, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v71, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v71, v[22:25] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v71, v[26:29] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v71, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_lshl_u32 v220, v34, s36, 1
	buffer_load_dwordx4 v[2:5], v170, s[24:27], 0 offen
	buffer_load_dwordx4 v[6:9], v220, s[24:27], 0 offen
	v_and_b32_e32 v70, 31, v0
	v_lshlrev_b32_e32 v10, 4, v69
	s_ashr_i32 s53, s52, 31
	v_and_b32_e32 v12, 0x1c0, v0
	v_and_b32_e32 v11, 32, v0
	v_lshl_or_b32 v10, v70, 8, v10
	s_lshl_b32 s2, s29, 6
	s_add_i32 s3, 0, 0x4440
	s_lshl_b64 s[26:27], s[52:53], 1
	scratch_store_dword off, v11, off offset:24 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v11, 1, v11
	scratch_store_dword off, v12, off offset:160 ; 4-byte Folded Spill
	v_lshl_or_b32 v12, v12, 7, v10
	s_add_u32 s12, s24, s26
	v_xor_b32_e32 v12, v12, v11
	v_add_u32_e32 v59, s3, v1
	s_addc_u32 s3, s33, s27
	v_add_u32_e32 v13, 0, v12
	v_xad_u32 v14, v12, 32, 0
	v_xad_u32 v15, v12, 64, 0
	v_xor_b32_e32 v16, 0x60, v12
	v_xor_b32_e32 v17, 0x80, v12
	v_xor_b32_e32 v18, 0xa0, v12
	v_xor_b32_e32 v19, 0xc0, v12
	v_xor_b32_e32 v12, 0xe0, v12
	s_and_b32 s6, s3, 0xffff
	v_xor_b32_e32 v68, v10, v11
	v_add_u32_e32 v16, 0, v16
	v_add_u32_e32 v17, 0, v17
	v_add_u32_e32 v18, 0, v18
	v_add_u32_e32 v19, 0, v19
	v_add_u32_e32 v12, 0, v12
	ds_read_b128 v[106:109], v13
	ds_read_b128 v[102:105], v14
	ds_read_b128 v[98:101], v15
	ds_read_b128 v[78:81], v16
	ds_read_b128 v[94:97], v17
	ds_read_b128 v[90:93], v18
	ds_read_b128 v[86:89], v19
	ds_read_b128 v[82:85], v12
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(3)
	ds_write_b128 v71, v[2:5] offset:17472
	s_waitcnt vmcnt(2)
	ds_write_b128 v59, v[6:9] offset:8192
	s_or_b32 s13, s6, s19
	v_add_u32_e32 v73, 0, v68
	buffer_load_dwordx4 v[50:53], v170, s[12:15], 0 offen
	buffer_load_dwordx4 v[54:57], v220, s[12:15], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v73 offset:17472
	ds_read_b128 v[60:63], v73 offset:25664
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
	v_xor_b32_e32 v1, 32, v68
	v_add_u32_e32 v74, 0, v1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[106:107], v[34:49]
	v_xor_b32_e32 v1, 64, v68
	v_add_u32_e32 v75, 0, v1
	v_xor_b32_e32 v1, 0x60, v68
	v_add_u32_e32 v76, 0, v1
	v_xor_b32_e32 v1, 0x80, v68
	v_add_u32_e32 v77, 0, v1
	v_xor_b32_e32 v1, 0xa0, v68
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[108:109], v[2:17]
	ds_read_b128 v[18:21], v74 offset:17472
	ds_read_b128 v[64:67], v74 offset:25664
	scratch_store_dwordx4 off, v[106:109], off offset:140 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[102:105], off offset:124 ; 16-byte Folded Spill
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s25, 0x5040100
	s_mov_b32 s33, 0x3e0293ee
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[102:103], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[60:61], v[106:107], v[34:49]
	s_nop 6
	ds_read_b128 v[34:37], v75 offset:17472
	ds_read_b128 v[38:41], v75 offset:25664
	scratch_store_dwordx4 off, v[98:101], off offset:108 ; 16-byte Folded Spill
	v_and_b32_e32 v61, 0x1f0, v0
	v_lshlrev_b32_e32 v60, 2, v0
	v_mfma_f32_32x32x8_f16 v[18:33], v[62:63], v[108:109], v[18:33]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_xor_b32_e32 v62, 0x80, v60
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[64:65], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[104:105], v[18:33]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[98:99], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[98:99], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[100:101], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[100:101], v[18:33]
	ds_read_b128 v[34:37], v76 offset:17472
	ds_read_b128 v[38:41], v76 offset:25664
	scratch_store_dwordx4 off, v[78:81], off offset:92 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[78:79], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[78:79], v[18:33]
	v_add_u32_e32 v78, 0, v1
	v_xor_b32_e32 v1, 0xc0, v68
	v_add_u32_e32 v79, 0, v1
	v_mov_b32_e32 v1, 0xff800000
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[80:81], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[80:81], v[18:33]
	ds_read_b128 v[34:37], v77 offset:17472
	ds_read_b128 v[38:41], v77 offset:25664
	scratch_store_dwordx4 off, v[94:97], off offset:76 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[94:95], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[94:95], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[96:97], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[96:97], v[18:33]
	ds_read_b128 v[34:37], v78 offset:17472
	ds_read_b128 v[38:41], v78 offset:25664
	scratch_store_dwordx4 off, v[90:93], off offset:60 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[90:91], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[90:91], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[92:93], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[92:93], v[18:33]
	ds_read_b128 v[34:37], v79 offset:17472
	ds_read_b128 v[38:41], v79 offset:25664
	scratch_store_dwordx4 off, v[86:89], off offset:44 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[86:87], v[2:17]
	v_lshrrev_b32_e32 v34, 3, v61
	v_xor_b32_e32 v35, 0xe0, v68
	v_add_u32_e32 v99, 0, v35
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[86:87], v[18:33]
	v_mad_u64_u32 v[38:39], s[6:7], s29, v34, v[58:59]
	s_and_b32 s6, s29, 0x3fff
	s_bitset1_b32 s6, 14
	s_and_b32 s7, s16, 0xffff
	s_lshl_b32 s24, s6, 16
	v_lshlrev_b32_e32 v80, 1, v38
	s_or_b32 s21, s7, s24
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[88:89], v[2:17]
	ds_read_b128 v[34:37], v99 offset:17472
	v_add_lshl_u32 v81, v38, s29, 1
	s_add_u32 s12, s12, s26
	s_addc_u32 s6, s3, s27
	s_ashr_i32 s3, s2, 31
	s_and_b32 s13, s6, 0xffff
	s_lshl_b64 s[6:7], s[2:3], 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[88:89], v[18:33]
	ds_read_b128 v[38:41], v99 offset:25664
	buffer_load_dwordx4 v[42:45], v80, s[20:23], 0 offen
	buffer_load_dwordx4 v[46:49], v81, s[20:23], 0 offen
	s_or_b32 s13, s13, s19
	s_add_u32 s22, s20, s6
	s_addc_u32 s23, s16, s7
	s_and_b32 s2, s23, 0xffff
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(9)
	ds_write_b128 v59, v[54:57] offset:8192
	scratch_store_dword off, v71, off offset:4 ; 4-byte Folded Spill
	ds_write_b128 v71, v[50:53] offset:17472
	buffer_load_dwordx4 v[252:255], v170, s[12:15], 0 offen
	buffer_load_dwordx4 v[172:175], v220, s[12:15], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dwordx4 off, v[82:85], off offset:28 ; 16-byte Folded Spill
	scratch_store_dword off, v62, off       ; 4-byte Folded Spill
	s_or_b32 s13, s2, s24
	s_mov_b32 s12, s22
	buffer_load_dwordx4 v[162:165], v80, s[12:15], 0 offen
	buffer_load_dwordx4 v[166:169], v81, s[12:15], 0 offen
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[82:83], v[2:17]
	s_mov_b32 s29, 0x7060302
	s_movk_i32 s2, 0x100
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_movk_i32 s2, 0xff
	v_cmp_lt_u32_e64 s[2:3], s2, v0
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[84:85], v[2:17]
	s_waitcnt vmcnt(7)
	v_perm_b32 v36, v47, v43, s25
	v_mfma_f32_32x32x8_f16 v[18:33], v[38:39], v[82:83], v[18:33]
	s_nop 7
	v_max_f32_e32 v34, v3, v3
	v_max_f32_e32 v35, v2, v2
	v_max_f32_e32 v34, v35, v34
	v_max3_f32 v34, v34, v4, v5
	v_max3_f32 v34, v34, v6, v7
	v_max3_f32 v34, v34, v8, v9
	v_max3_f32 v34, v34, v10, v11
	v_mfma_f32_32x32x8_f16 v[18:33], v[40:41], v[84:85], v[18:33]
	v_max3_f32 v34, v34, v12, v13
	v_max3_f32 v34, v34, v14, v15
	v_max3_f32 v34, v34, v16, v17
	v_perm_b32 v37, v47, v43, s29
	v_perm_b32 v38, v48, v44, s25
	v_perm_b32 v39, v48, v44, s29
	v_and_b32_e32 v43, 56, v60
	s_nop 3
	v_max3_f32 v34, v34, v18, v19
	v_max3_f32 v34, v34, v20, v21
	v_max3_f32 v34, v34, v22, v23
	v_max3_f32 v34, v34, v24, v25
	v_max3_f32 v34, v34, v26, v27
	v_max3_f32 v34, v34, v28, v29
	v_max3_f32 v34, v34, v30, v31
	v_max3_f32 v34, v34, v32, v33
	ds_bpermute_b32 v35, v62, v34
	v_lshl_add_u32 v44, v69, 10, 0
	v_perm_b32 v40, v49, v45, s25
	v_perm_b32 v41, v49, v45, s29
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v249, v34, v35, v1
	v_mul_f32_e32 v34, 0xbe0293ee, v249
	v_fmamk_f32 v2, v2, 0x3e0293ee, v34
	v_fmamk_f32 v3, v3, 0x3e0293ee, v34
	v_fmamk_f32 v4, v4, 0x3e0293ee, v34
	v_fmamk_f32 v5, v5, 0x3e0293ee, v34
	v_fmamk_f32 v6, v6, 0x3e0293ee, v34
	v_fmamk_f32 v7, v7, 0x3e0293ee, v34
	v_fmamk_f32 v8, v8, 0x3e0293ee, v34
	v_fmamk_f32 v9, v9, 0x3e0293ee, v34
	v_fmamk_f32 v10, v10, 0x3e0293ee, v34
	v_fmamk_f32 v11, v11, 0x3e0293ee, v34
	v_fmamk_f32 v12, v12, 0x3e0293ee, v34
	v_fmamk_f32 v13, v13, 0x3e0293ee, v34
	v_fmamk_f32 v14, v14, 0x3e0293ee, v34
	v_fmamk_f32 v15, v15, 0x3e0293ee, v34
	v_fmamk_f32 v16, v16, 0x3e0293ee, v34
	v_fmamk_f32 v17, v17, 0x3e0293ee, v34
	v_fmamk_f32 v18, v18, 0x3e0293ee, v34
	v_fmamk_f32 v19, v19, 0x3e0293ee, v34
	v_fmamk_f32 v20, v20, 0x3e0293ee, v34
	v_fmamk_f32 v21, v21, 0x3e0293ee, v34
	v_fmamk_f32 v22, v22, 0x3e0293ee, v34
	v_fmamk_f32 v23, v23, 0x3e0293ee, v34
	v_fmamk_f32 v24, v24, 0x3e0293ee, v34
	v_fmamk_f32 v25, v25, 0x3e0293ee, v34
	v_fmamk_f32 v26, v26, 0x3e0293ee, v34
	v_fmamk_f32 v27, v27, 0x3e0293ee, v34
	v_fmamk_f32 v28, v28, 0x3e0293ee, v34
	v_fmamk_f32 v29, v29, 0x3e0293ee, v34
	v_fmamk_f32 v30, v30, 0x3e0293ee, v34
	v_fmamk_f32 v31, v31, 0x3e0293ee, v34
	v_fmamk_f32 v32, v32, 0x3e0293ee, v34
	v_fmac_f32_e32 v34, 0x3e0293ee, v33
	v_perm_b32 v33, v46, v42, s25
	v_perm_b32 v35, v46, v42, s29
	v_lshrrev_b32_e32 v42, 2, v61
	v_add3_u32 v42, v44, v42, v43
	v_lshlrev_b32_e32 v43, 6, v69
	v_add_u32_e32 v66, v42, v43
	ds_write2_b32 v66, v33, v35 offset1:34
	ds_write2_b32 v66, v36, v37 offset0:68 offset1:102
	ds_write2_b32 v66, v38, v39 offset0:136 offset1:170
	ds_write2_b32 v66, v40, v41 offset0:204 offset1:238
	ds_read_b128 v[206:209], v73 offset:17472
	ds_read_b128 v[190:193], v73 offset:25664
	ds_read_b128 v[202:205], v74 offset:17472
	ds_read_b128 v[186:189], v74 offset:25664
	ds_read_b128 v[198:201], v75 offset:17472
	ds_read_b128 v[182:185], v75 offset:25664
	ds_read_b128 v[194:197], v76 offset:17472
	ds_read_b128 v[178:181], v76 offset:25664
	ds_read_b128 v[94:97], v77 offset:17472
	ds_read_b128 v[126:129], v77 offset:25664
	ds_read_b128 v[90:93], v78 offset:17472
	ds_read_b128 v[122:125], v78 offset:25664
	ds_read_b128 v[86:89], v79 offset:17472
	ds_read_b128 v[118:121], v79 offset:25664
	ds_read_b128 v[82:85], v99 offset:17472
	ds_read_b128 v[114:117], v99 offset:25664
	v_fmac_f32_e32 v1, 0xbe0293ee, v249
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v0, off offset:20 ; 4-byte Folded Spill
	s_and_saveexec_b64 s[12:13], s[2:3]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[12:13]
	s_load_dwordx2 s[20:21], s[0:1], 0x4c
	s_load_dword s16, s[0:1], 0x54
	v_exp_f32_e32 v223, v2
	v_exp_f32_e32 v215, v1
	scratch_load_dword v1, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v2, off, off offset:20 ; 4-byte Folded Reload
	v_exp_f32_e32 v230, v3
	v_lshlrev_b32_e32 v0, 7, v70
	v_lshlrev_b32_e32 v3, 3, v70
	scratch_store_dword off, v70, off offset:156 ; 4-byte Folded Spill
	s_add_u32 s0, s30, s34
	s_addc_u32 s1, s31, s35
	s_mul_i32 s3, s52, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_exp_f32_e32 v225, v4
	v_exp_f32_e32 v229, v5
	v_exp_f32_e32 v222, v6
	v_exp_f32_e32 v228, v7
	v_exp_f32_e32 v227, v8
	v_exp_f32_e32 v235, v9
	v_exp_f32_e32 v234, v10
	v_exp_f32_e32 v216, v11
	v_exp_f32_e32 v224, v12
	v_exp_f32_e32 v226, v13
	v_exp_f32_e32 v233, v14
	v_exp_f32_e32 v161, v15
	v_exp_f32_e32 v160, v16
	v_exp_f32_e32 v245, v17
	v_exp_f32_e32 v131, v18
	v_exp_f32_e32 v232, v19
	v_exp_f32_e32 v231, v20
	v_exp_f32_e32 v244, v21
	v_exp_f32_e32 v243, v22
	v_exp_f32_e32 v248, v23
	v_exp_f32_e32 v247, v24
	v_exp_f32_e32 v237, v25
	v_exp_f32_e32 v236, v26
	v_exp_f32_e32 v242, v27
	v_exp_f32_e32 v246, v28
	v_exp_f32_e32 v102, v29
	v_exp_f32_e32 v98, v30
	v_exp_f32_e32 v219, v31
	v_exp_f32_e32 v218, v32
	v_exp_f32_e32 v217, v34
	s_mul_hi_i32 s2, s52, 6
	s_add_u32 s0, s3, s0
	s_addc_u32 s1, s2, s1
	s_add_u32 s0, s4, s0
	v_mov_b32_e32 v18, 0
	s_waitcnt vmcnt(4)
	v_lshrrev_b32_e32 v211, 16, v166
	s_addc_u32 s1, s5, s1
	v_mov_b32_e32 v213, 1.0
	s_movk_i32 s2, 0xffc0
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
	s_waitcnt vmcnt(2)
	v_lshrrev_b32_e32 v1, 2, v1
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v2, 1, v2
	v_and_b32_e32 v2, 8, v2
	v_add3_u32 v0, 0, v0, v1
	scratch_store_dword off, v3, off offset:164 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:172 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off offset:168 ; 4-byte Folded Spill
	scratch_load_dwordx4 v[132:135], off, off offset:28 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[136:139], off, off offset:44 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[140:143], off, off offset:60 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[144:147], off, off offset:76 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[148:151], off, off offset:92 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[152:155], off, off offset:108 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[156:159], off, off offset:124 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[238:241], off, off offset:140 ; 16-byte Folded Reload
	v_add_u32_e32 v0, v0, v3
	v_add_u32_e32 v0, v0, v2
	v_mov_b32_e32 v2, v18
	v_mov_b32_e32 v3, v18
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
	scratch_store_dword off, v81, off offset:8 ; 4-byte Folded Spill
	scratch_store_dword off, v80, off offset:12 ; 4-byte Folded Spill
	scratch_store_dword off, v99, off offset:16 ; 4-byte Folded Spill
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v1, v249
	v_mov_b32_e32 v249, v98
	v_mov_b32_e32 v210, v170
	v_mov_b32_e32 v170, v218
	v_mov_b32_e32 v218, v102
	v_mov_b64_e32 v[112:113], s[50:51]
	v_mov_b64_e32 v[110:111], s[48:49]
	v_mov_b64_e32 v[108:109], s[46:47]
	v_mov_b64_e32 v[106:107], s[44:45]
	v_mov_b64_e32 v[104:105], s[42:43]
	v_mov_b64_e32 v[102:103], s[40:41]
	v_mov_b64_e32 v[100:101], s[38:39]
	v_mov_b64_e32 v[98:99], s[36:37]
	v_mov_b32_e32 v251, v73
	v_mov_b32_e32 v171, v74
	v_mov_b32_e32 v130, v75
	v_mov_b32_e32 v177, v76
	v_mov_b32_e32 v176, v77
	v_mov_b32_e32 v221, v78
	v_mov_b32_e32 v214, v79
	v_mov_b32_e32 v250, v66
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[206:207], v[238:239], v[98:113]
	v_mov_b32_e32 v212, v213
	s_setprio 0
	v_mfma_f32_32x32x8_f16 v[66:81], v[208:209], v[240:241], v[66:81]
	scratch_load_dword v208, off, off       ; 4-byte Folded Reload
	v_mul_f32_e32 v18, v18, v215
	v_mul_f32_e32 v19, v19, v215
	v_mul_f32_e32 v20, v20, v215
	v_mul_f32_e32 v21, v21, v215
	v_mul_f32_e32 v22, v22, v215
	v_mul_f32_e32 v23, v23, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[202:203], v[156:157], v[66:81]
	v_mul_f32_e32 v24, v24, v215
	v_mul_f32_e32 v25, v25, v215
	v_mul_f32_e32 v26, v26, v215
	v_mul_f32_e32 v27, v27, v215
	v_mul_f32_e32 v28, v28, v215
	v_mul_f32_e32 v29, v29, v215
	v_mul_f32_e32 v30, v30, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[204:205], v[158:159], v[66:81]
	v_mul_f32_e32 v31, v31, v215
	v_mul_f32_e32 v32, v32, v215
	v_mul_f32_e32 v33, v33, v215
	v_mul_f32_e32 v50, v50, v215
	v_mul_f32_e32 v51, v51, v215
	v_mul_f32_e32 v52, v52, v215
	v_mul_f32_e32 v53, v53, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[198:199], v[152:153], v[66:81]
	v_mul_f32_e32 v54, v54, v215
	v_mul_f32_e32 v55, v55, v215
	v_mul_f32_e32 v56, v56, v215
	v_mul_f32_e32 v57, v57, v215
	v_mul_f32_e32 v58, v58, v215
	v_mul_f32_e32 v59, v59, v215
	v_mul_f32_e32 v60, v60, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[200:201], v[154:155], v[66:81]
	v_mul_f32_e32 v61, v61, v215
	v_mul_f32_e32 v62, v62, v215
	v_mul_f32_e32 v63, v63, v215
	v_mul_f32_e32 v64, v64, v215
	v_mul_f32_e32 v65, v65, v215
	v_mul_f32_e32 v34, v34, v215
	v_mul_f32_e32 v35, v35, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[194:195], v[148:149], v[66:81]
	v_mul_f32_e32 v36, v36, v215
	v_mul_f32_e32 v37, v37, v215
	v_mul_f32_e32 v38, v38, v215
	v_mul_f32_e32 v39, v39, v215
	v_mul_f32_e32 v40, v40, v215
	v_mul_f32_e32 v41, v41, v215
	v_mul_f32_e32 v42, v42, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[196:197], v[150:151], v[66:81]
	v_mul_f32_e32 v43, v43, v215
	v_mul_f32_e32 v44, v44, v215
	v_mul_f32_e32 v45, v45, v215
	v_mul_f32_e32 v46, v46, v215
	v_mul_f32_e32 v47, v47, v215
	v_mul_f32_e32 v48, v48, v215
	v_mul_f32_e32 v49, v49, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[144:145], v[66:81]
	v_mul_f32_e32 v2, v2, v215
	v_mul_f32_e32 v3, v3, v215
	v_mul_f32_e32 v4, v4, v215
	v_mul_f32_e32 v5, v5, v215
	v_mul_f32_e32 v6, v6, v215
	v_mul_f32_e32 v7, v7, v215
	v_mul_f32_e32 v8, v8, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[96:97], v[146:147], v[66:81]
	v_mul_f32_e32 v9, v9, v215
	v_mul_f32_e32 v10, v10, v215
	v_mul_f32_e32 v11, v11, v215
	v_mul_f32_e32 v12, v12, v215
	v_mul_f32_e32 v13, v13, v215
	v_mul_f32_e32 v14, v14, v215
	v_mul_f32_e32 v15, v15, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[140:141], v[66:81]
	v_mul_f32_e32 v16, v16, v215
	v_mul_f32_e32 v17, v17, v215
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[86:87], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[88:89], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[82:83], v[132:133], v[66:81]
	v_add_f32_e32 v82, v223, v230
	v_add_f32_e32 v82, v82, v225
	v_add_f32_e32 v82, v82, v229
	v_add_f32_e32 v82, v82, v222
	v_add_f32_e32 v82, v82, v228
	v_add_f32_e32 v82, v82, v227
	v_add_f32_e32 v82, v82, v235
	v_mfma_f32_32x32x8_f16 v[66:81], v[84:85], v[134:135], v[66:81]
	v_add_f32_e32 v194, v82, v234
	v_mfma_f32_32x32x8_f16 v[82:97], v[190:191], v[238:239], v[98:113]
	v_mfma_f32_32x32x8_f16 v[82:97], v[192:193], v[240:241], v[82:97]
	s_nop 5
	v_add_f32_e32 v98, v194, v216
	v_add_f32_e32 v98, v98, v224
	v_add_f32_e32 v98, v98, v226
	v_add_f32_e32 v98, v98, v233
	v_add_f32_e32 v98, v98, v161
	v_add_f32_e32 v98, v98, v160
	v_add_f32_e32 v98, v98, v245
	v_mfma_f32_32x32x8_f16 v[82:97], v[186:187], v[156:157], v[82:97]
	v_add_f32_e32 v98, v98, v131
	v_add_f32_e32 v98, v98, v232
	v_add_f32_e32 v98, v98, v231
	v_add_f32_e32 v98, v98, v244
	v_add_f32_e32 v98, v98, v243
	v_add_f32_e32 v98, v98, v248
	v_add_f32_e32 v98, v98, v247
	v_mfma_f32_32x32x8_f16 v[82:97], v[188:189], v[158:159], v[82:97]
	v_add_f32_e32 v98, v98, v237
	v_add_f32_e32 v98, v98, v236
	v_add_f32_e32 v98, v98, v242
	v_add_f32_e32 v98, v98, v246
	v_add_f32_e32 v98, v98, v218
	v_add_f32_e32 v98, v98, v249
	v_add_f32_e32 v98, v98, v219
	v_mfma_f32_32x32x8_f16 v[82:97], v[182:183], v[152:153], v[82:97]
	v_add_f32_e32 v98, v98, v170
	v_add_f32_e32 v98, v98, v217
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v99, v208, v98
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v213, v98, v99
	v_mfma_f32_32x32x8_f16 v[82:97], v[184:185], v[154:155], v[82:97]
	v_fmac_f32_e32 v213, v212, v215
	v_mfma_f32_32x32x8_f16 v[82:97], v[178:179], v[148:149], v[82:97]
	v_cvt_pkrtz_f16_f32 v178, v223, v230
	v_cvt_pkrtz_f16_f32 v179, v225, v229
	v_mfma_f32_32x32x8_f16 v[82:97], v[180:181], v[150:151], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[126:127], v[144:145], v[82:97]
	v_cvt_pkrtz_f16_f32 v126, v234, v216
	v_cvt_pkrtz_f16_f32 v127, v224, v226
	v_mfma_f32_32x32x8_f16 v[82:97], v[128:129], v[146:147], v[82:97]
	v_cvt_pkrtz_f16_f32 v128, v222, v228
	v_cvt_pkrtz_f16_f32 v129, v227, v235
	v_mfma_f32_32x32x8_f16 v[82:97], v[122:123], v[140:141], v[82:97]
	v_cvt_pkrtz_f16_f32 v122, v131, v232
	v_cvt_pkrtz_f16_f32 v123, v231, v244
	v_mfma_f32_32x32x8_f16 v[82:97], v[124:125], v[142:143], v[82:97]
	v_cvt_pkrtz_f16_f32 v124, v233, v161
	v_cvt_pkrtz_f16_f32 v125, v160, v245
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[136:137], v[82:97]
	v_cvt_pkrtz_f16_f32 v118, v236, v242
	v_cvt_pkrtz_f16_f32 v119, v246, v218
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[138:139], v[82:97]
	v_cvt_pkrtz_f16_f32 v120, v243, v248
	v_cvt_pkrtz_f16_f32 v121, v247, v237
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[132:133], v[82:97]
	v_cvt_pkrtz_f16_f32 v114, v249, v219
	v_cvt_pkrtz_f16_f32 v115, v170, v217
	v_mov_b32_e32 v170, v210
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[134:135], v[82:97]
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_barrier
	s_barrier
	scratch_load_dword v98, off, off offset:4 ; 4-byte Folded Reload
	s_and_b32 s3, s1, 0xffff
	s_mov_b32 s12, s0
	s_or_b32 s13, s3, s19
	v_add_u32_e32 v99, 0x2000, v0
	v_add_u32_e32 v100, 0x3000, v0
	s_waitcnt vmcnt(0)
	ds_write_b128 v98, v[252:255] offset:17472
	ds_write_b128 v98, v[172:175] offset:25664
	buffer_load_dwordx4 v[252:255], v210, s[12:15], 0 offen
	buffer_load_dwordx4 v[172:175], v220, s[12:15], 0 offen
	v_add_u32_e32 v98, 0x1000, v0
	ds_read2_b64 v[180:183], v0 offset1:2
	ds_read2_b64 v[184:187], v0 offset0:4 offset1:6
	ds_read2_b64 v[188:191], v0 offset0:8 offset1:10
	ds_read2_b64 v[192:195], v0 offset0:12 offset1:14
	ds_read2_b64 v[196:199], v98 offset0:34 offset1:36
	ds_read2_b64 v[200:203], v98 offset0:38 offset1:40
	ds_read2_b64 v[204:207], v98 offset0:42 offset1:44
	ds_read2_b64 v[216:219], v98 offset0:46 offset1:48
	ds_read2_b64 v[222:225], v99 offset0:68 offset1:70
	ds_read2_b64 v[226:229], v99 offset0:72 offset1:74
	ds_read2_b64 v[230:233], v99 offset0:76 offset1:78
	ds_read2_b64 v[234:237], v99 offset0:80 offset1:82
	ds_read2_b64 v[110:113], v100 offset0:102 offset1:104
	ds_read2_b64 v[106:109], v100 offset0:106 offset1:108
	ds_read2_b64 v[102:105], v100 offset0:110 offset1:112
	ds_read2_b64 v[98:101], v100 offset0:114 offset1:116
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[18:33], v[180:181], v[178:179], v[18:33]
	s_barrier
	s_setprio 0
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[2:17], v[110:111], v[178:179], v[2:17]
	v_max_f32_e32 v116, v67, v67
	v_max_f32_e32 v117, v66, v66
	v_max_f32_e32 v116, v117, v116
	v_max3_f32 v116, v116, v68, v69
	v_max3_f32 v116, v116, v70, v71
	v_max3_f32 v116, v116, v72, v73
	v_max3_f32 v116, v116, v74, v75
	v_mfma_f32_32x32x8_f16 v[2:17], v[112:113], v[128:129], v[2:17]
	v_max3_f32 v116, v116, v76, v77
	v_max3_f32 v116, v116, v78, v79
	v_max3_f32 v116, v116, v80, v81
	v_max3_f32 v116, v116, v82, v83
	v_max3_f32 v116, v116, v84, v85
	v_max3_f32 v116, v116, v86, v87
	v_max3_f32 v116, v116, v88, v89
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[106:107], v[126:127], v[2:17]
	v_max3_f32 v116, v116, v90, v91
	v_max3_f32 v116, v116, v92, v93
	v_max3_f32 v116, v116, v94, v95
	v_max3_f32 v116, v116, v96, v97
	ds_bpermute_b32 v117, v208, v116
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v249, v1, v116, v117
	v_mfma_f32_32x32x8_f16 v[2:17], v[108:109], v[124:125], v[2:17]
	v_mul_f32_e32 v212, 0x3e0293ee, v249
	v_fma_f32 v80, v80, s33, -v212
	v_fma_f32 v81, v81, s33, -v212
	v_exp_f32_e32 v160, v80
	scratch_load_dword v80, off, off offset:12 ; 4-byte Folded Reload
	v_exp_f32_e32 v245, v81
	v_fma_f32 v66, v66, s33, -v212
	v_mfma_f32_32x32x8_f16 v[2:17], v[102:103], v[122:123], v[2:17]
	v_fma_f32 v67, v67, s33, -v212
	v_fma_f32 v68, v68, s33, -v212
	v_fma_f32 v69, v69, s33, -v212
	v_fma_f32 v70, v70, s33, -v212
	v_fma_f32 v71, v71, s33, -v212
	v_fma_f32 v72, v72, s33, -v212
	v_fma_f32 v73, v73, s33, -v212
	v_mfma_f32_32x32x8_f16 v[2:17], v[104:105], v[120:121], v[2:17]
	v_fma_f32 v74, v74, s33, -v212
	v_fma_f32 v75, v75, s33, -v212
	v_fma_f32 v76, v76, s33, -v212
	v_fma_f32 v77, v77, s33, -v212
	v_fma_f32 v78, v78, s33, -v212
	v_fma_f32 v79, v79, s33, -v212
	v_fma_f32 v82, v82, s33, -v212
	v_mfma_f32_32x32x8_f16 v[2:17], v[98:99], v[118:119], v[2:17]
	scratch_load_dword v99, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v81, off, off offset:8 ; 4-byte Folded Reload
	v_fma_f32 v83, v83, s33, -v212
	v_fma_f32 v84, v84, s33, -v212
	v_fma_f32 v85, v85, s33, -v212
	v_fma_f32 v86, v86, s33, -v212
	v_fma_f32 v87, v87, s33, -v212
	v_mfma_f32_32x32x8_f16 v[50:65], v[196:197], v[178:179], v[50:65]
	v_fma_f32 v88, v88, s33, -v212
	v_fma_f32 v89, v89, s33, -v212
	v_fma_f32 v90, v90, s33, -v212
	v_fma_f32 v91, v91, s33, -v212
	v_fma_f32 v92, v92, s33, -v212
	v_fma_f32 v93, v93, s33, -v212
	v_fma_f32 v94, v94, s33, -v212
	v_mfma_f32_32x32x8_f16 v[34:49], v[222:223], v[178:179], v[34:49]
	v_fma_f32 v95, v95, s33, -v212
	v_fma_f32 v96, v96, s33, -v212
	v_fma_f32 v97, v97, s33, -v212
	v_fma_f32 v1, v1, s33, -v212
	v_exp_f32_e32 v223, v66
	v_exp_f32_e32 v222, v70
	v_exp_f32_e32 v161, v79
	v_mfma_f32_32x32x8_f16 v[18:33], v[182:183], v[128:129], v[18:33]
	v_exp_f32_e32 v131, v82
	v_exp_f32_e32 v244, v85
	v_exp_f32_e32 v243, v86
	v_exp_f32_e32 v248, v87
	v_exp_f32_e32 v247, v88
	v_exp_f32_e32 v242, v91
	v_exp_f32_e32 v246, v92
	v_mfma_f32_32x32x8_f16 v[50:65], v[198:199], v[128:129], v[50:65]
	v_exp_f32_e32 v102, v93
	v_exp_f32_e32 v98, v94
	v_exp_f32_e32 v215, v1
	v_mov_b32_e32 v79, v214
	v_mfma_f32_32x32x8_f16 v[34:49], v[224:225], v[128:129], v[34:49]
	v_exp_f32_e32 v225, v68
	v_exp_f32_e32 v224, v76
	v_mov_b32_e32 v76, v177
	v_mfma_f32_32x32x8_f16 v[18:33], v[184:185], v[126:127], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[200:201], v[126:127], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[226:227], v[126:127], v[34:49]
	v_exp_f32_e32 v227, v72
	v_exp_f32_e32 v226, v77
	v_mov_b32_e32 v77, v176
	v_mfma_f32_32x32x8_f16 v[18:33], v[186:187], v[124:125], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[202:203], v[124:125], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[228:229], v[124:125], v[34:49]
	v_exp_f32_e32 v229, v69
	v_exp_f32_e32 v228, v71
	v_mfma_f32_32x32x8_f16 v[18:33], v[188:189], v[122:123], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[204:205], v[122:123], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[230:231], v[122:123], v[34:49]
	v_exp_f32_e32 v230, v67
	v_exp_f32_e32 v231, v84
	v_mfma_f32_32x32x8_f16 v[18:33], v[190:191], v[120:121], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[206:207], v[120:121], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[232:233], v[120:121], v[34:49]
	v_exp_f32_e32 v233, v78
	v_exp_f32_e32 v232, v83
	v_mov_b32_e32 v78, v221
	v_mfma_f32_32x32x8_f16 v[18:33], v[192:193], v[118:119], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[216:217], v[118:119], v[50:65]
	v_exp_f32_e32 v216, v75
	v_exp_f32_e32 v217, v97
	v_mov_b32_e32 v75, v130
	v_mfma_f32_32x32x8_f16 v[34:49], v[234:235], v[118:119], v[34:49]
	v_exp_f32_e32 v235, v73
	v_exp_f32_e32 v234, v74
	v_mov_b32_e32 v73, v251
	v_mov_b32_e32 v74, v171
	v_mfma_f32_32x32x8_f16 v[18:33], v[194:195], v[114:115], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[218:219], v[114:115], v[50:65]
	v_exp_f32_e32 v219, v95
	v_exp_f32_e32 v218, v96
	v_mfma_f32_32x32x8_f16 v[34:49], v[236:237], v[114:115], v[34:49]
	v_exp_f32_e32 v237, v89
	v_exp_f32_e32 v236, v90
	v_mfma_f32_32x32x8_f16 v[2:17], v[100:101], v[114:115], v[2:17]
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s22, s6
	s_addc_u32 s3, s23, s7
	s_and_b32 s4, s3, 0xffff
	s_or_b32 s13, s4, s24
	s_barrier
	v_perm_b32 v1, v166, v162, s25
	v_alignbit_b32 v66, v211, v162, 16
	v_perm_b32 v67, v167, v163, s25
	v_perm_b32 v68, v167, v163, s29
	v_perm_b32 v69, v168, v164, s25
	v_perm_b32 v70, v168, v164, s29
	v_perm_b32 v71, v169, v165, s25
	v_perm_b32 v72, v169, v165, s29
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[166:169], v81, s[12:15], 0 offen
	buffer_load_dwordx4 v[162:165], v80, s[12:15], 0 offen
	ds_read_b128 v[206:209], v251 offset:17472
	ds_read_b128 v[190:193], v251 offset:25664
	ds_read_b128 v[202:205], v171 offset:17472
	ds_read_b128 v[186:189], v171 offset:25664
	ds_read_b128 v[198:201], v130 offset:17472
	ds_read_b128 v[182:185], v130 offset:25664
	ds_read_b128 v[194:197], v177 offset:17472
	ds_read_b128 v[178:181], v177 offset:25664
	ds_read_b128 v[94:97], v176 offset:17472
	ds_read_b128 v[126:129], v176 offset:25664
	ds_read_b128 v[90:93], v221 offset:17472
	ds_read_b128 v[122:125], v221 offset:25664
	ds_read_b128 v[86:89], v214 offset:17472
	ds_read_b128 v[118:121], v214 offset:25664
	ds_read_b128 v[82:85], v99 offset:17472
	ds_read_b128 v[114:117], v99 offset:25664
	ds_write2_b32 v250, v1, v66 offset1:34
	v_mov_b32_e32 v66, v250
	ds_write2_b32 v250, v67, v68 offset0:68 offset1:102
	ds_write2_b32 v250, v69, v70 offset0:136 offset1:170
	ds_write2_b32 v250, v71, v72 offset0:204 offset1:238
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v211, 16, v166
	; sched_barrier mask(0x00000000)
	s_add_u32 s22, s22, s6
	s_addc_u32 s23, s23, s7
	s_add_u32 s0, s0, s26
	s_addc_u32 s1, s1, s27
	s_add_i32 s2, s2, 64
	s_cmpk_lt_u32 s2, 0x1f00
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	v_mov_b32_e32 v1, v102
	v_mov_b32_e32 v0, v98
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx4 v[238:241], off, off offset:140 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[154:157], off, off offset:124 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[150:153], off, off offset:108 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[146:149], off, off offset:92 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[142:145], off, off offset:76 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[138:141], off, off offset:60 ; 16-byte Folded Reload
	s_mov_b32 s36, 0
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
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
	v_mov_b64_e32 v[80:81], s[50:51]
	v_mov_b64_e32 v[78:79], s[48:49]
	v_mov_b64_e32 v[76:77], s[46:47]
	v_mov_b64_e32 v[74:75], s[44:45]
	v_mov_b64_e32 v[72:73], s[42:43]
	v_mov_b64_e32 v[70:71], s[40:41]
	v_mov_b64_e32 v[68:69], s[38:39]
	v_mov_b64_e32 v[66:67], s[36:37]
	scratch_load_dwordx4 v[132:135], off, off offset:44 ; 16-byte Folded Reload
	v_mul_f32_e32 v50, v50, v215
	v_mul_f32_e32 v51, v51, v215
	v_mul_f32_e32 v52, v52, v215
	v_mul_f32_e32 v53, v53, v215
	v_mul_f32_e32 v54, v54, v215
	v_mul_f32_e32 v55, v55, v215
	v_mul_f32_e32 v56, v56, v215
	v_mul_f32_e32 v57, v57, v215
	v_mul_f32_e32 v58, v58, v215
	v_mul_f32_e32 v59, v59, v215
	v_mul_f32_e32 v60, v60, v215
	v_mul_f32_e32 v61, v61, v215
	v_mul_f32_e32 v62, v62, v215
	v_mul_f32_e32 v63, v63, v215
	v_mul_f32_e32 v64, v64, v215
	v_mul_f32_e32 v65, v65, v215
	v_mul_f32_e32 v2, v2, v215
	v_mul_f32_e32 v3, v3, v215
	v_mul_f32_e32 v4, v4, v215
	v_mul_f32_e32 v5, v5, v215
	v_mul_f32_e32 v6, v6, v215
	v_mul_f32_e32 v7, v7, v215
	v_mul_f32_e32 v8, v8, v215
	v_mul_f32_e32 v9, v9, v215
	v_mul_f32_e32 v10, v10, v215
	v_mul_f32_e32 v11, v11, v215
	v_mul_f32_e32 v12, v12, v215
	v_mul_f32_e32 v13, v13, v215
	v_mul_f32_e32 v14, v14, v215
	v_mul_f32_e32 v15, v15, v215
	v_mul_f32_e32 v16, v16, v215
	v_mul_f32_e32 v17, v17, v215
	s_mul_i32 s0, s18, 0xc0000
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s2, s8, s0
	s_addc_u32 s4, s9, s1
	s_lshl_b32 s0, s17, 14
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s2, s2, s0
	s_addc_u32 s4, s4, s1
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[0:1], s[28:29], 2
	s_add_u32 s0, s2, s0
	s_addc_u32 s8, s4, s1
	s_add_i32 s1, s28, 0xffffc100
	s_add_u32 s4, s12, s6
	s_addc_u32 s2, s3, s7
	s_waitcnt vmcnt(6) lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[98:113], v[206:207], v[238:239], v[66:81]
	s_and_b32 s5, s2, 0xffff
	s_mov_b32 s2, 0x5040100
	s_mov_b32 s3, 0x7060302
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	s_cmp_lt_i32 s1, 1
	v_mfma_f32_32x32x8_f16 v[98:113], v[208:209], v[240:241], v[98:113]
	scratch_load_dwordx4 v[208:211], off, off offset:28 ; 16-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(6)
	v_mfma_f32_32x32x8_f16 v[98:113], v[202:203], v[154:155], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[204:205], v[156:157], v[98:113]
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x8_f16 v[98:113], v[198:199], v[150:151], v[98:113]
	v_cvt_pkrtz_f16_f32 v198, v236, v242
	v_cvt_pkrtz_f16_f32 v199, v246, v1
	v_mfma_f32_32x32x8_f16 v[98:113], v[200:201], v[152:153], v[98:113]
	v_cvt_pkrtz_f16_f32 v200, v0, v219
	v_cvt_pkrtz_f16_f32 v201, v218, v217
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x8_f16 v[98:113], v[194:195], v[146:147], v[98:113]
	v_cvt_pkrtz_f16_f32 v194, v131, v232
	v_cvt_pkrtz_f16_f32 v195, v231, v244
	v_mfma_f32_32x32x8_f16 v[98:113], v[196:197], v[148:149], v[98:113]
	v_cvt_pkrtz_f16_f32 v196, v243, v248
	v_cvt_pkrtz_f16_f32 v197, v247, v237
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_f16 v[98:113], v[94:95], v[142:143], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[96:97], v[144:145], v[98:113]
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x8_f16 v[98:113], v[90:91], v[138:139], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[92:93], v[140:141], v[98:113]
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x8_f16 v[98:113], v[86:87], v[132:133], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[88:89], v[134:135], v[98:113]
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8_f16 v[98:113], v[82:83], v[208:209], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[84:85], v[210:211], v[98:113]
	v_mfma_f32_32x32x8_f16 v[82:97], v[190:191], v[238:239], v[66:81]
	v_cvt_pkrtz_f16_f32 v190, v234, v216
	v_cvt_pkrtz_f16_f32 v191, v224, v226
	v_mfma_f32_32x32x8_f16 v[82:97], v[192:193], v[240:241], v[82:97]
	v_cvt_pkrtz_f16_f32 v192, v233, v161
	v_cvt_pkrtz_f16_f32 v193, v160, v245
	v_mfma_f32_32x32x8_f16 v[82:97], v[186:187], v[154:155], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[188:189], v[156:157], v[82:97]
	v_cvt_pkrtz_f16_f32 v188, v222, v228
	v_cvt_pkrtz_f16_f32 v189, v227, v235
	v_mfma_f32_32x32x8_f16 v[82:97], v[182:183], v[150:151], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[184:185], v[152:153], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[178:179], v[146:147], v[82:97]
	v_cvt_pkrtz_f16_f32 v178, v223, v230
	v_cvt_pkrtz_f16_f32 v179, v225, v229
	v_mfma_f32_32x32x8_f16 v[82:97], v[180:181], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[126:127], v[142:143], v[82:97]
	v_mul_f32_e32 v126, v30, v215
	v_mul_f32_e32 v127, v31, v215
	v_mul_f32_e32 v30, v46, v215
	v_mul_f32_e32 v31, v47, v215
	v_mfma_f32_32x32x8_f16 v[82:97], v[128:129], v[144:145], v[82:97]
	v_mul_f32_e32 v128, v32, v215
	v_mul_f32_e32 v129, v33, v215
	v_mul_f32_e32 v32, v48, v215
	v_mul_f32_e32 v33, v49, v215
	v_mfma_f32_32x32x8_f16 v[82:97], v[122:123], v[138:139], v[82:97]
	v_mul_f32_e32 v122, v26, v215
	v_mul_f32_e32 v123, v27, v215
	v_mul_f32_e32 v26, v42, v215
	v_mul_f32_e32 v27, v43, v215
	v_mfma_f32_32x32x8_f16 v[82:97], v[124:125], v[140:141], v[82:97]
	v_mul_f32_e32 v124, v28, v215
	v_mul_f32_e32 v125, v29, v215
	v_mul_f32_e32 v28, v44, v215
	v_mul_f32_e32 v29, v45, v215
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[132:133], v[82:97]
	v_mul_f32_e32 v118, v22, v215
	v_mul_f32_e32 v119, v23, v215
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[134:135], v[82:97]
	v_mul_f32_e32 v120, v24, v215
	v_mul_f32_e32 v121, v25, v215
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[208:209], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[210:211], v[82:97]
	scratch_load_dword v114, off, off offset:164 ; 4-byte Folded Reload
	scratch_load_dword v115, off, off offset:168 ; 4-byte Folded Reload
	scratch_load_dword v116, off, off offset:172 ; 4-byte Folded Reload
	v_mul_f32_e32 v117, v21, v215
	scratch_store_dword off, v1, off offset:176 ; 4-byte Folded Spill
	scratch_load_dword v1, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_add3_u32 v136, v116, v115, v114
	ds_read2_b64 v[182:185], v136 offset1:2
	v_mul_f32_e32 v114, v18, v215
	v_mul_f32_e32 v115, v19, v215
	v_mul_f32_e32 v116, v20, v215
	ds_read2_b64 v[18:21], v136 offset0:4 offset1:6
	scratch_store_dword off, v0, off offset:180 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[114:129], v[182:183], v[178:179], v[114:129]
	v_add_u32_e32 v182, 0x1000, v136
	ds_read2_b64 v[22:25], v182 offset0:34 offset1:36
	v_add_u32_e32 v183, 0x2000, v136
	v_add_u32_e32 v0, 0x3000, v136
	v_mfma_f32_32x32x8_f16 v[114:129], v[184:185], v[188:189], v[114:129]
	ds_read2_b64 v[184:187], v183 offset0:68 offset1:70
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[114:129], v[18:19], v[190:191], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[20:21], v[192:193], v[114:129]
	ds_read2_b64 v[18:21], v136 offset0:8 offset1:10
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[114:129], v[18:19], v[194:195], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[20:21], v[196:197], v[114:129]
	ds_read2_b64 v[18:21], v136 offset0:12 offset1:14
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[114:129], v[18:19], v[198:199], v[114:129]
	v_mfma_f32_32x32x8_f16 v[50:65], v[22:23], v[178:179], v[50:65]
	v_mul_f32_e32 v22, v38, v215
	v_mul_f32_e32 v23, v39, v215
	v_mfma_f32_32x32x8_f16 v[114:129], v[20:21], v[200:201], v[114:129]
	ds_read2_b64 v[18:21], v182 offset0:38 offset1:40
	v_mfma_f32_32x32x8_f16 v[50:65], v[24:25], v[188:189], v[50:65]
	v_mul_f32_e32 v24, v40, v215
	v_mul_f32_e32 v25, v41, v215
	ds_read2_b64 v[38:41], v0 offset0:102 offset1:104
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[50:65], v[18:19], v[190:191], v[50:65]
	v_mfma_f32_32x32x8_f16 v[50:65], v[20:21], v[192:193], v[50:65]
	ds_read2_b64 v[18:21], v182 offset0:42 offset1:44
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[18:19], v[194:195], v[50:65]
	v_mfma_f32_32x32x8_f16 v[50:65], v[20:21], v[196:197], v[50:65]
	ds_read2_b64 v[18:21], v182 offset0:46 offset1:48
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[18:19], v[198:199], v[50:65]
	v_mul_f32_e32 v18, v34, v215
	v_mul_f32_e32 v19, v35, v215
	v_mfma_f32_32x32x8_f16 v[50:65], v[20:21], v[200:201], v[50:65]
	v_mul_f32_e32 v20, v36, v215
	v_mul_f32_e32 v21, v37, v215
	ds_read2_b64 v[34:37], v183 offset0:72 offset1:74
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[184:185], v[178:179], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[186:187], v[188:189], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[34:35], v[190:191], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[192:193], v[18:33]
	ds_read2_b64 v[34:37], v183 offset0:76 offset1:78
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[34:35], v[194:195], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[196:197], v[18:33]
	ds_read2_b64 v[34:37], v183 offset0:80 offset1:82
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[34:35], v[198:199], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[38:39], v[178:179], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[200:201], v[18:33]
	ds_read2_b64 v[34:37], v0 offset0:106 offset1:108
	v_mfma_f32_32x32x8_f16 v[2:17], v[40:41], v[188:189], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[190:191], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[192:193], v[2:17]
	ds_read2_b64 v[34:37], v0 offset0:110 offset1:112
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[194:195], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[196:197], v[2:17]
	ds_read2_b64 v[34:37], v0 offset0:114 offset1:116
	s_waitcnt vmcnt(1)
	ds_write_b128 v1, v[252:255] offset:17472
	ds_write_b128 v1, v[172:175] offset:25664
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[172:175], v251 offset:17472
	ds_read_b128 v[178:181], v251 offset:25664
	v_mov_b32_e32 v252, v171
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[198:199], v[2:17]
	v_max_f32_e32 v1, v99, v99
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[200:201], v[2:17]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[238:239], v[66:81]
	ds_read_b128 v[170:173], v171 offset:17472
	ds_read_b128 v[184:187], v252 offset:25664
	v_mfma_f32_32x32x8_f16 v[34:49], v[174:175], v[240:241], v[34:49]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[154:155], v[34:49]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[156:157], v[34:49]
	ds_read_b128 v[170:173], v130 offset:17472
	ds_read_b128 v[188:191], v130 offset:25664
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[150:151], v[34:49]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[152:153], v[34:49]
	ds_read_b128 v[170:173], v177 offset:17472
	ds_read_b128 v[192:195], v177 offset:25664
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[146:147], v[34:49]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[148:149], v[34:49]
	ds_read_b128 v[170:173], v176 offset:17472
	ds_read_b128 v[196:199], v176 offset:25664
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[142:143], v[34:49]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[144:145], v[34:49]
	ds_read_b128 v[170:173], v221 offset:17472
	ds_read_b128 v[200:203], v221 offset:25664
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[138:139], v[34:49]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[140:141], v[34:49]
	ds_read_b128 v[170:173], v214 offset:17472
	ds_read_b128 v[204:207], v214 offset:25664
	scratch_load_dword v220, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[132:133], v[34:49]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[134:135], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b128 v[170:173], v220 offset:17472
	ds_read_b128 v[252:255], v220 offset:25664
	scratch_load_dword v221, off, off       ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[238:239], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[180:181], v[240:241], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[184:185], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[150:151], v[66:81]
	v_max_f32_e32 v150, v98, v98
	v_max_f32_e32 v1, v150, v1
	v_max3_f32 v1, v1, v100, v101
	v_max3_f32 v1, v1, v102, v103
	v_max3_f32 v1, v1, v104, v105
	v_max3_f32 v1, v1, v106, v107
	v_max3_f32 v1, v1, v108, v109
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[152:153], v[66:81]
	v_max3_f32 v1, v1, v110, v111
	v_max3_f32 v1, v1, v112, v113
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_mfma_f32_32x32x8_f16 v[66:81], v[192:193], v[146:147], v[66:81]
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v146, v221, v1
	v_mfma_f32_32x32x8_f16 v[66:81], v[194:195], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[196:197], v[142:143], v[66:81]
	v_perm_b32 v142, v166, v162, s3
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[208:209], v[34:49]
	v_mfma_f32_32x32x8_f16 v[66:81], v[198:199], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[210:211], v[34:49]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v173, v249, v1, v146
	v_perm_b32 v1, v166, v162, s2
	ds_write2_b32 v250, v1, v142 offset1:34
	v_perm_b32 v1, v167, v163, s2
	v_perm_b32 v142, v167, v163, s3
	ds_write2_b32 v250, v1, v142 offset0:68 offset1:102
	v_perm_b32 v1, v168, v164, s2
	v_perm_b32 v142, v168, v164, s3
	ds_write2_b32 v250, v1, v142 offset0:136 offset1:170
	v_mfma_f32_32x32x8_f16 v[66:81], v[200:201], v[138:139], v[66:81]
	v_perm_b32 v1, v169, v165, s2
	v_perm_b32 v138, v169, v165, s3
	ds_write2_b32 v250, v1, v138 offset0:204 offset1:238
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	v_mul_f32_e32 v178, 0x3e0293ee, v173
	v_fmac_f32_e32 v212, 0xbe0293ee, v173
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[142:145], v1, s[4:7], 0 offen
	s_nop 0
	scratch_load_dword v1, off, off offset:8 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[66:81], v[202:203], v[140:141], v[66:81]
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[146:149], v1, s[4:7], 0 offen
	v_mfma_f32_32x32x8_f16 v[66:81], v[204:205], v[132:133], v[66:81]
	s_mov_b32 s4, 0x3e0293ee
	v_fma_f32 v1, v98, s4, -v178
	v_fma_f32 v98, v99, s4, -v178
	v_fma_f32 v99, v100, s4, -v178
	v_fma_f32 v100, v101, s4, -v178
	v_exp_f32_e32 v132, v1
	v_exp_f32_e32 v133, v98
	v_mfma_f32_32x32x8_f16 v[66:81], v[206:207], v[134:135], v[66:81]
	v_exp_f32_e32 v134, v99
	v_exp_f32_e32 v135, v100
	v_exp_f32_e32 v1, v212
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[152:155], v136 offset1:2
	ds_read2_b64 v[174:177], v136 offset0:4 offset1:6
	ds_read2_b64 v[168:171], v136 offset0:8 offset1:10
	ds_read2_b64 v[138:141], v136 offset0:12 offset1:14
	v_fma_f32 v101, v102, s4, -v178
	v_fma_f32 v137, v103, s4, -v178
	v_fma_f32 v151, v104, s4, -v178
	v_fma_f32 v156, v105, s4, -v178
	v_fma_f32 v157, v106, s4, -v178
	v_fma_f32 v158, v107, s4, -v178
	v_fma_f32 v159, v108, s4, -v178
	v_fma_f32 v164, v109, s4, -v178
	v_fma_f32 v165, v110, s4, -v178
	v_fma_f32 v166, v111, s4, -v178
	v_fma_f32 v167, v112, s4, -v178
	v_fma_f32 v172, v113, s4, -v178
	v_exp_f32_e32 v180, v101
	v_cvt_pkrtz_f16_f32 v204, v132, v133
	v_cvt_pkrtz_f16_f32 v205, v134, v135
	v_mul_f32_e32 v98, v114, v1
	v_mul_f32_e32 v99, v115, v1
	v_mul_f32_e32 v100, v116, v1
	v_mul_f32_e32 v101, v117, v1
	v_mul_f32_e32 v102, v118, v1
	v_mul_f32_e32 v103, v119, v1
	v_mul_f32_e32 v104, v120, v1
	v_mul_f32_e32 v105, v121, v1
	v_mul_f32_e32 v106, v122, v1
	v_mul_f32_e32 v107, v123, v1
	v_mul_f32_e32 v108, v124, v1
	v_mul_f32_e32 v109, v125, v1
	v_mul_f32_e32 v110, v126, v1
	v_mul_f32_e32 v111, v127, v1
	v_mul_f32_e32 v112, v128, v1
	v_mul_f32_e32 v113, v129, v1
	v_exp_f32_e32 v150, v137
	v_exp_f32_e32 v151, v151
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[98:113], v[152:153], v[204:205], v[98:113]
	v_exp_f32_e32 v152, v156
	v_cvt_pkrtz_f16_f32 v124, v180, v150
	v_exp_f32_e32 v153, v157
	v_exp_f32_e32 v157, v158
	v_cvt_pkrtz_f16_f32 v125, v151, v152
	v_exp_f32_e32 v156, v165
	v_exp_f32_e32 v158, v166
	v_mfma_f32_32x32x8_f16 v[98:113], v[154:155], v[124:125], v[98:113]
	v_exp_f32_e32 v154, v159
	v_exp_f32_e32 v155, v164
	v_cvt_pkrtz_f16_f32 v126, v153, v157
	v_exp_f32_e32 v159, v167
	v_exp_f32_e32 v206, v172
	v_cvt_pkrtz_f16_f32 v127, v154, v155
	v_cvt_pkrtz_f16_f32 v118, v156, v158
	v_fma_f32 v82, v82, s4, -v178
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[98:113], v[174:175], v[126:127], v[98:113]
	v_cvt_pkrtz_f16_f32 v119, v159, v206
	v_fma_f32 v83, v83, s4, -v178
	v_fma_f32 v84, v84, s4, -v178
	v_fma_f32 v85, v85, s4, -v178
	v_exp_f32_e32 v207, v82
	v_exp_f32_e32 v162, v83
	v_exp_f32_e32 v163, v84
	v_mfma_f32_32x32x8_f16 v[98:113], v[176:177], v[118:119], v[98:113]
	v_exp_f32_e32 v164, v85
	v_cvt_pkrtz_f16_f32 v120, v207, v162
	v_fma_f32 v86, v86, s4, -v178
	v_fma_f32 v87, v87, s4, -v178
	v_cvt_pkrtz_f16_f32 v121, v163, v164
	v_fma_f32 v88, v88, s4, -v178
	v_fma_f32 v89, v89, s4, -v178
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[98:113], v[168:169], v[120:121], v[98:113]
	v_exp_f32_e32 v165, v86
	v_exp_f32_e32 v168, v87
	v_exp_f32_e32 v169, v88
	v_exp_f32_e32 v166, v89
	ds_read2_b64 v[174:177], v182 offset0:34 offset1:36
	v_cvt_pkrtz_f16_f32 v122, v165, v168
	v_fma_f32 v90, v90, s4, -v178
	v_cvt_pkrtz_f16_f32 v123, v169, v166
	v_fma_f32 v91, v91, s4, -v178
	v_fma_f32 v92, v92, s4, -v178
	v_mfma_f32_32x32x8_f16 v[98:113], v[170:171], v[122:123], v[98:113]
	v_fma_f32 v93, v93, s4, -v178
	v_exp_f32_e32 v167, v90
	v_exp_f32_e32 v170, v91
	v_exp_f32_e32 v171, v92
	v_exp_f32_e32 v172, v93
	v_fma_f32 v94, v94, s4, -v178
	v_fma_f32 v82, v95, s4, -v178
	v_fma_f32 v83, v96, s4, -v178
	v_fma_f32 v84, v97, s4, -v178
	v_cvt_pkrtz_f16_f32 v116, v167, v170
	v_cvt_pkrtz_f16_f32 v117, v171, v172
	v_exp_f32_e32 v128, v94
	v_exp_f32_e32 v137, v82
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[98:113], v[138:139], v[116:117], v[98:113]
	v_exp_f32_e32 v138, v83
	v_exp_f32_e32 v139, v84
	v_mul_f32_e32 v82, v50, v1
	v_mul_f32_e32 v83, v51, v1
	v_mul_f32_e32 v84, v52, v1
	v_mul_f32_e32 v85, v53, v1
	v_mul_f32_e32 v86, v54, v1
	v_mul_f32_e32 v87, v55, v1
	v_mul_f32_e32 v88, v56, v1
	v_mul_f32_e32 v89, v57, v1
	v_mul_f32_e32 v90, v58, v1
	v_mul_f32_e32 v91, v59, v1
	v_mul_f32_e32 v92, v60, v1
	v_mul_f32_e32 v93, v61, v1
	v_mul_f32_e32 v94, v62, v1
	v_mul_f32_e32 v95, v63, v1
	v_mul_f32_e32 v96, v64, v1
	v_mul_f32_e32 v97, v65, v1
	ds_read2_b64 v[50:53], v182 offset0:38 offset1:40
	v_mfma_f32_32x32x8_f16 v[66:81], v[252:253], v[208:209], v[66:81]
	v_cvt_pkrtz_f16_f32 v114, v128, v137
	v_cvt_pkrtz_f16_f32 v115, v138, v139
	v_mul_f32_e32 v54, v22, v1
	v_mul_f32_e32 v55, v23, v1
	v_mul_f32_e32 v56, v24, v1
	v_mul_f32_e32 v57, v25, v1
	v_mul_f32_e32 v62, v30, v1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[204:205], v[82:97]
	v_mul_f32_e32 v63, v31, v1
	v_mul_f32_e32 v64, v32, v1
	v_mul_f32_e32 v65, v33, v1
	v_mul_f32_e32 v22, v6, v1
	v_mul_f32_e32 v23, v7, v1
	v_mul_f32_e32 v24, v8, v1
	v_mul_f32_e32 v25, v9, v1
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[124:125], v[82:97]
	v_mul_f32_e32 v30, v14, v1
	v_mul_f32_e32 v31, v15, v1
	v_mul_f32_e32 v32, v16, v1
	v_mul_f32_e32 v33, v17, v1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[82:97], v[50:51], v[126:127], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[52:53], v[118:119], v[82:97]
	ds_read2_b64 v[50:53], v182 offset0:42 offset1:44
	ds_read2_b64 v[58:61], v182 offset0:46 offset1:48
	ds_read2_b64 v[174:177], v183 offset0:68 offset1:70
	ds_read2_b64 v[184:187], v183 offset0:72 offset1:74
	ds_read2_b64 v[188:191], v183 offset0:76 offset1:78
	ds_read2_b64 v[192:195], v183 offset0:80 offset1:82
	ds_read2_b64 v[196:199], v0 offset0:102 offset1:104
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[82:97], v[50:51], v[120:121], v[82:97]
	v_mul_f32_e32 v50, v18, v1
	v_mul_f32_e32 v51, v19, v1
	v_max_f32_e32 v18, v35, v35
	v_max_f32_e32 v19, v34, v34
	v_max_f32_e32 v18, v19, v18
	v_max3_f32 v18, v18, v36, v37
	v_max3_f32 v18, v18, v38, v39
	v_mfma_f32_32x32x8_f16 v[82:97], v[52:53], v[122:123], v[82:97]
	v_max3_f32 v18, v18, v40, v41
	v_max3_f32 v18, v18, v42, v43
	v_max3_f32 v18, v18, v44, v45
	v_max3_f32 v18, v18, v46, v47
	v_max3_f32 v18, v18, v48, v49
	v_mul_f32_e32 v52, v20, v1
	v_mul_f32_e32 v53, v21, v1
	v_mfma_f32_32x32x8_f16 v[66:81], v[254:255], v[210:211], v[66:81]
	v_mul_f32_e32 v20, v4, v1
	v_mul_f32_e32 v21, v5, v1
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[82:97], v[58:59], v[116:117], v[82:97]
	s_nop 6
	v_max3_f32 v18, v18, v66, v67
	v_max3_f32 v18, v18, v68, v69
	v_max3_f32 v18, v18, v70, v71
	v_max3_f32 v18, v18, v72, v73
	v_max3_f32 v18, v18, v74, v75
	v_max3_f32 v18, v18, v76, v77
	v_mul_f32_e32 v58, v26, v1
	v_mfma_f32_32x32x8_f16 v[82:97], v[60:61], v[114:115], v[82:97]
	v_mul_f32_e32 v59, v27, v1
	v_mul_f32_e32 v60, v28, v1
	v_mul_f32_e32 v61, v29, v1
	v_max3_f32 v18, v18, v78, v79
	v_max3_f32 v18, v18, v80, v81
	ds_bpermute_b32 v19, v221, v18
	v_mul_f32_e32 v26, v10, v1
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[204:205], v[50:65]
	v_mul_f32_e32 v27, v11, v1
	v_mul_f32_e32 v28, v12, v1
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v129, v173, v18, v19
	v_mul_f32_e32 v18, v2, v1
	v_mul_f32_e32 v19, v3, v1
	v_mul_f32_e32 v29, v13, v1
	v_fmac_f32_e32 v178, 0xbe0293ee, v129
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[124:125], v[50:65]
	ds_read2_b64 v[174:177], v0 offset0:106 offset1:108
	v_mfma_f32_32x32x8_f16 v[18:33], v[196:197], v[204:205], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[198:199], v[124:125], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[184:185], v[126:127], v[50:65]
	v_mfma_f32_32x32x8_f16 v[98:113], v[140:141], v[114:115], v[98:113]
	v_mul_f32_e32 v140, 0xbe0293ee, v129
	v_fmamk_f32 v2, v34, 0x3e0293ee, v140
	v_fmamk_f32 v3, v35, 0x3e0293ee, v140
	v_fmamk_f32 v4, v36, 0x3e0293ee, v140
	v_fmamk_f32 v5, v37, 0x3e0293ee, v140
	v_fmamk_f32 v6, v38, 0x3e0293ee, v140
	v_fmamk_f32 v7, v39, 0x3e0293ee, v140
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[174:175], v[126:127], v[18:33]
	v_exp_f32_e32 v126, v2
	v_exp_f32_e32 v127, v3
	s_waitcnt vmcnt(0)
	v_perm_b32 v2, v146, v142, s2
	v_perm_b32 v3, v146, v142, s3
	v_exp_f32_e32 v174, v4
	v_exp_f32_e32 v175, v5
	v_fmamk_f32 v38, v40, 0x3e0293ee, v140
	v_mfma_f32_32x32x8_f16 v[50:65], v[186:187], v[118:119], v[50:65]
	ds_read2_b64 v[184:187], v0 offset0:110 offset1:112
	ds_read2_b64 v[200:203], v0 offset0:114 offset1:116
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write2_b32 v250, v2, v3 offset1:34
	v_perm_b32 v2, v147, v143, s2
	v_perm_b32 v3, v147, v143, s3
	ds_write2_b32 v250, v2, v3 offset0:68 offset1:102
	v_perm_b32 v2, v148, v144, s2
	v_perm_b32 v3, v148, v144, s3
	ds_write2_b32 v250, v2, v3 offset0:136 offset1:170
	v_perm_b32 v2, v149, v145, s2
	v_perm_b32 v3, v149, v145, s3
	ds_write2_b32 v250, v2, v3 offset0:204 offset1:238
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b64 v[34:37], v136 offset1:2
	v_mfma_f32_32x32x8_f16 v[18:33], v[176:177], v[118:119], v[18:33]
	v_exp_f32_e32 v118, v178
	v_fmamk_f32 v40, v42, 0x3e0293ee, v140
	v_fmamk_f32 v42, v43, 0x3e0293ee, v140
	v_fmamk_f32 v43, v44, 0x3e0293ee, v140
	v_fmamk_f32 v44, v45, 0x3e0293ee, v140
	v_fmamk_f32 v45, v46, 0x3e0293ee, v140
	v_fmamk_f32 v46, v47, 0x3e0293ee, v140
	v_fmamk_f32 v47, v48, 0x3e0293ee, v140
	v_fmamk_f32 v48, v49, 0x3e0293ee, v140
	v_fmamk_f32 v49, v66, 0x3e0293ee, v140
	v_fmamk_f32 v130, v67, 0x3e0293ee, v140
	v_exp_f32_e32 v179, v6
	v_exp_f32_e32 v181, v7
	v_cvt_pkrtz_f16_f32 v66, v126, v127
	v_cvt_pkrtz_f16_f32 v67, v174, v175
	v_mul_f32_e32 v2, v98, v118
	v_mul_f32_e32 v3, v99, v118
	v_mul_f32_e32 v4, v100, v118
	v_mul_f32_e32 v5, v101, v118
	v_mul_f32_e32 v6, v102, v118
	v_mul_f32_e32 v7, v103, v118
	v_mul_f32_e32 v8, v104, v118
	v_mul_f32_e32 v9, v105, v118
	v_mul_f32_e32 v10, v106, v118
	v_mul_f32_e32 v11, v107, v118
	v_mul_f32_e32 v12, v108, v118
	v_mul_f32_e32 v13, v109, v118
	v_mul_f32_e32 v14, v110, v118
	v_mul_f32_e32 v15, v111, v118
	v_mul_f32_e32 v16, v112, v118
	v_mul_f32_e32 v17, v113, v118
	v_fmamk_f32 v39, v41, 0x3e0293ee, v140
	v_mfma_f32_32x32x8_f16 v[50:65], v[188:189], v[120:121], v[50:65]
	v_exp_f32_e32 v104, v38
	v_exp_f32_e32 v105, v39
	v_fmamk_f32 v188, v74, 0x3e0293ee, v140
	v_fmamk_f32 v141, v75, 0x3e0293ee, v140
	v_exp_f32_e32 v106, v40
	v_cvt_pkrtz_f16_f32 v74, v179, v181
	v_cvt_pkrtz_f16_f32 v75, v104, v105
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[66:67], v[2:17]
	ds_read2_b64 v[38:41], v136 offset0:4 offset1:6
	v_exp_f32_e32 v107, v42
	v_exp_f32_e32 v108, v43
	v_exp_f32_e32 v109, v44
	v_exp_f32_e32 v110, v45
	v_cvt_pkrtz_f16_f32 v98, v106, v107
	v_exp_f32_e32 v111, v46
	v_mfma_f32_32x32x8_f16 v[18:33], v[184:185], v[120:121], v[18:33]
	v_cvt_pkrtz_f16_f32 v99, v108, v109
	v_exp_f32_e32 v112, v47
	v_exp_f32_e32 v113, v48
	v_fmamk_f32 v124, v72, 0x3e0293ee, v140
	v_fmamk_f32 v125, v73, 0x3e0293ee, v140
	v_cvt_pkrtz_f16_f32 v72, v110, v111
	v_cvt_pkrtz_f16_f32 v73, v112, v113
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[74:75], v[2:17]
	ds_read2_b64 v[34:37], v136 offset0:8 offset1:10
	v_fmamk_f32 v68, v68, 0x3e0293ee, v140
	v_fmamk_f32 v69, v69, 0x3e0293ee, v140
	v_fmamk_f32 v173, v78, 0x3e0293ee, v140
	v_fmamk_f32 v100, v79, 0x3e0293ee, v140
	v_fmamk_f32 v70, v70, 0x3e0293ee, v140
	v_fmamk_f32 v71, v71, 0x3e0293ee, v140
	v_mfma_f32_32x32x8_f16 v[50:65], v[190:191], v[122:123], v[50:65]
	v_exp_f32_e32 v119, v70
	v_exp_f32_e32 v120, v71
	v_exp_f32_e32 v121, v124
	v_fmamk_f32 v76, v76, 0x3e0293ee, v140
	v_fmamk_f32 v77, v77, 0x3e0293ee, v140
	v_cvt_pkrtz_f16_f32 v70, v119, v120
	v_exp_f32_e32 v124, v141
	v_mfma_f32_32x32x8_f16 v[18:33], v[186:187], v[122:123], v[18:33]
	v_exp_f32_e32 v122, v125
	v_exp_f32_e32 v123, v188
	v_exp_f32_e32 v125, v76
	v_exp_f32_e32 v141, v100
	v_cvt_pkrtz_f16_f32 v71, v121, v122
	v_cvt_pkrtz_f16_f32 v76, v123, v124
	ds_read2_b64 v[100:103], v182 offset0:34 offset1:36
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[38:39], v[98:99], v[2:17]
	v_fmamk_f32 v42, v80, 0x3e0293ee, v140
	v_fmac_f32_e32 v140, 0x3e0293ee, v81
	v_exp_f32_e32 v143, v173
	v_exp_f32_e32 v142, v42
	v_exp_f32_e32 v140, v140
	v_mul_f32_e32 v42, v90, v118
	v_mul_f32_e32 v43, v91, v118
	v_mfma_f32_32x32x8_f16 v[50:65], v[192:193], v[116:117], v[50:65]
	v_mul_f32_e32 v44, v92, v118
	v_mul_f32_e32 v45, v93, v118
	v_mul_f32_e32 v46, v94, v118
	v_mul_f32_e32 v47, v95, v118
	v_mul_f32_e32 v48, v96, v118
	v_mfma_f32_32x32x8_f16 v[18:33], v[200:201], v[116:117], v[18:33]
	v_exp_f32_e32 v116, v68
	v_exp_f32_e32 v117, v69
	v_cvt_pkrtz_f16_f32 v68, v143, v141
	v_cvt_pkrtz_f16_f32 v69, v142, v140
	v_cvt_pkrtz_f16_f32 v79, v116, v117
	v_mfma_f32_32x32x8_f16 v[2:17], v[40:41], v[72:73], v[2:17]
	ds_read2_b64 v[38:41], v136 offset0:12 offset1:14
	v_mfma_f32_32x32x8_f16 v[50:65], v[194:195], v[114:115], v[50:65]
	v_mfma_f32_32x32x8_f16 v[18:33], v[202:203], v[114:115], v[18:33]
	v_exp_f32_e32 v114, v49
	v_exp_f32_e32 v115, v130
	v_exp_f32_e32 v130, v77
	v_mul_f32_e32 v49, v97, v118
	s_nop 5
	v_mul_f32_e32 v50, v50, v118
	v_cvt_pkrtz_f16_f32 v78, v114, v115
	v_cvt_pkrtz_f16_f32 v77, v125, v130
	v_mul_f32_e32 v51, v51, v118
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[78:79], v[2:17]
	v_mul_f32_e32 v34, v82, v118
	v_mul_f32_e32 v35, v83, v118
	ds_read2_b64 v[80:83], v182 offset0:38 offset1:40
	v_mul_f32_e32 v52, v52, v118
	v_mul_f32_e32 v53, v53, v118
	v_mul_f32_e32 v54, v54, v118
	v_mul_f32_e32 v55, v55, v118
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[70:71], v[2:17]
	v_mul_f32_e32 v36, v84, v118
	v_mul_f32_e32 v37, v85, v118
	v_add_f32_e32 v84, v223, v230
	v_add_f32_e32 v84, v84, v225
	v_add_f32_e32 v84, v84, v229
	v_mul_f32_e32 v56, v56, v118
	v_mul_f32_e32 v57, v57, v118
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[38:39], v[76:77], v[2:17]
	v_mul_f32_e32 v38, v86, v118
	v_mul_f32_e32 v39, v87, v118
	v_mul_f32_e32 v58, v58, v118
	v_mul_f32_e32 v59, v59, v118
	v_mul_f32_e32 v60, v60, v118
	v_mul_f32_e32 v61, v61, v118
	v_mul_f32_e32 v62, v62, v118
	v_mfma_f32_32x32x8_f16 v[2:17], v[40:41], v[68:69], v[2:17]
	v_mul_f32_e32 v40, v88, v118
	v_mul_f32_e32 v41, v89, v118
	v_mul_f32_e32 v63, v63, v118
	v_mul_f32_e32 v64, v64, v118
	v_mul_f32_e32 v65, v65, v118
	v_mul_f32_e32 v18, v18, v118
	v_mul_f32_e32 v19, v19, v118
	v_mfma_f32_32x32x8_f16 v[34:49], v[100:101], v[66:67], v[34:49]
	v_mul_f32_e32 v20, v20, v118
	v_mul_f32_e32 v21, v21, v118
	v_mul_f32_e32 v22, v22, v118
	v_mul_f32_e32 v23, v23, v118
	v_mul_f32_e32 v24, v24, v118
	v_mul_f32_e32 v25, v25, v118
	v_mul_f32_e32 v26, v26, v118
	v_mfma_f32_32x32x8_f16 v[34:49], v[102:103], v[74:75], v[34:49]
	v_mul_f32_e32 v27, v27, v118
	v_mul_f32_e32 v28, v28, v118
	v_mul_f32_e32 v29, v29, v118
	v_mul_f32_e32 v30, v30, v118
	v_mul_f32_e32 v31, v31, v118
	v_mul_f32_e32 v32, v32, v118
	v_mul_f32_e32 v33, v33, v118
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[80:81], v[98:99], v[34:49]
	v_add_f32_e32 v80, v84, v222
	v_add_f32_e32 v80, v80, v228
	v_add_f32_e32 v80, v80, v227
	ds_read2_b64 v[84:87], v182 offset0:42 offset1:44
	v_add_f32_e32 v80, v80, v235
	v_add_f32_e32 v80, v80, v234
	v_add_f32_e32 v80, v80, v216
	v_mfma_f32_32x32x8_f16 v[34:49], v[82:83], v[72:73], v[34:49]
	v_add_f32_e32 v80, v80, v224
	v_add_f32_e32 v80, v80, v226
	v_add_f32_e32 v80, v80, v233
	v_add_f32_e32 v80, v80, v161
	v_add_f32_e32 v80, v80, v160
	v_add_f32_e32 v80, v80, v245
	v_add_f32_e32 v80, v80, v131
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[84:85], v[78:79], v[34:49]
	v_add_f32_e32 v80, v80, v232
	v_add_f32_e32 v80, v80, v231
	v_add_f32_e32 v80, v80, v244
	v_add_f32_e32 v80, v80, v243
	v_add_f32_e32 v80, v80, v248
	v_add_f32_e32 v84, v80, v247
	ds_read2_b64 v[80:83], v182 offset0:46 offset1:48
	v_mfma_f32_32x32x8_f16 v[34:49], v[86:87], v[70:71], v[34:49]
	v_add_f32_e32 v84, v84, v237
	v_add_f32_e32 v84, v84, v236
	v_add_f32_e32 v84, v84, v242
	v_add_f32_e32 v88, v84, v246
	ds_read2_b64 v[84:87], v183 offset0:68 offset1:70
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[80:81], v[76:77], v[34:49]
	scratch_load_dword v80, off, off offset:176 ; 4-byte Folded Reload
	scratch_load_dword v81, off, off offset:180 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v80, v88, v80
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[84:85], v[66:67], v[50:65]
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v80, v80, v81
	v_add_f32_e32 v80, v80, v219
	v_add_f32_e32 v80, v80, v218
	v_add_f32_e32 v81, v80, v217
	scratch_load_dword v80, off, off offset:160 ; 4-byte Folded Reload
	ds_bpermute_b32 v88, v221, v81
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v81, v81, v88
	v_mfma_f32_32x32x8_f16 v[50:65], v[86:87], v[74:75], v[50:65]
	scratch_load_dword v86, off, off offset:156 ; 4-byte Folded Reload
	v_fmac_f32_e32 v81, v213, v215
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v80, 1, v80
	v_mfma_f32_32x32x8_f16 v[34:49], v[82:83], v[68:69], v[34:49]
	ds_read2_b64 v[82:85], v183 offset0:72 offset1:74
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v80, v80, v86
	v_add_f32_e32 v86, v132, v133
	v_add_f32_e32 v86, v134, v86
	v_add_f32_e32 v86, v135, v86
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[82:83], v[98:99], v[50:65]
	v_add_f32_e32 v82, v180, v86
	v_add_f32_e32 v82, v150, v82
	v_add_f32_e32 v82, v151, v82
	v_add_f32_e32 v82, v152, v82
	v_add_f32_e32 v82, v153, v82
	v_add_f32_e32 v82, v157, v82
	ds_read2_b64 v[86:89], v183 offset0:76 offset1:78
	v_add_f32_e32 v82, v154, v82
	v_mfma_f32_32x32x8_f16 v[50:65], v[84:85], v[72:73], v[50:65]
	v_add_f32_e32 v82, v155, v82
	v_add_f32_e32 v82, v156, v82
	v_add_f32_e32 v82, v158, v82
	v_add_f32_e32 v82, v159, v82
	v_add_f32_e32 v82, v206, v82
	v_add_f32_e32 v82, v207, v82
	v_add_f32_e32 v82, v162, v82
	v_add_f32_e32 v82, v163, v82
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[86:87], v[78:79], v[50:65]
	v_add_f32_e32 v82, v164, v82
	v_add_f32_e32 v82, v165, v82
	v_add_f32_e32 v82, v168, v82
	v_add_f32_e32 v86, v169, v82
	v_add_f32_e32 v86, v166, v86
	v_add_f32_e32 v86, v167, v86
	v_add_f32_e32 v86, v170, v86
	ds_read2_b64 v[82:85], v183 offset0:80 offset1:82
	v_add_f32_e32 v86, v171, v86
	v_mfma_f32_32x32x8_f16 v[50:65], v[88:89], v[70:71], v[50:65]
	v_add_f32_e32 v90, v172, v86
	ds_read2_b64 v[86:89], v0 offset0:102 offset1:104
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[50:65], v[82:83], v[76:77], v[50:65]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[86:87], v[66:67], v[18:33]
	v_add_f32_e32 v86, v126, v127
	v_add_f32_e32 v66, v128, v90
	v_add_f32_e32 v66, v137, v66
	v_add_f32_e32 v66, v138, v66
	v_add_f32_e32 v66, v139, v66
	ds_bpermute_b32 v67, v221, v66
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v66, v66, v67
	v_mfma_f32_32x32x8_f16 v[50:65], v[84:85], v[68:69], v[50:65]
	ds_read2_b64 v[82:85], v0 offset0:106 offset1:108
	v_fmac_f32_e32 v66, v81, v1
	v_lshl_add_u32 v1, v80, 2, 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[88:89], v[74:75], v[18:33]
	v_add_f32_e32 v74, v174, v86
	v_add_f32_e32 v74, v175, v74
	v_add_f32_e32 v74, v179, v74
	v_add_f32_e32 v74, v181, v74
	v_add_f32_e32 v74, v104, v74
	v_add_f32_e32 v74, v105, v74
	v_add_f32_e32 v74, v106, v74
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[82:83], v[98:99], v[18:33]
	v_add_f32_e32 v74, v107, v74
	v_add_f32_e32 v74, v108, v74
	v_add_f32_e32 v74, v109, v74
	ds_read2_b64 v[86:89], v0 offset0:110 offset1:112
	v_add_f32_e32 v74, v110, v74
	v_add_f32_e32 v74, v111, v74
	v_add_f32_e32 v74, v112, v74
	v_mfma_f32_32x32x8_f16 v[18:33], v[84:85], v[72:73], v[18:33]
	v_add_f32_e32 v72, v113, v74
	v_add_f32_e32 v72, v114, v72
	v_add_f32_e32 v72, v115, v72
	v_add_f32_e32 v72, v116, v72
	v_add_f32_e32 v72, v117, v72
	v_add_f32_e32 v72, v119, v72
	v_add_f32_e32 v72, v120, v72
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[86:87], v[78:79], v[18:33]
	v_add_f32_e32 v72, v121, v72
	v_add_f32_e32 v72, v122, v72
	v_add_f32_e32 v72, v123, v72
	v_add_f32_e32 v72, v124, v72
	v_add_f32_e32 v72, v125, v72
	v_add_f32_e32 v78, v130, v72
	ds_read2_b64 v[72:75], v0 offset0:114 offset1:116
	v_mfma_f32_32x32x8_f16 v[18:33], v[88:89], v[70:71], v[18:33]
	v_add_f32_e32 v0, v143, v78
	v_add_f32_e32 v0, v141, v0
	v_add_f32_e32 v0, v142, v0
	v_add_f32_e32 v0, v140, v0
	ds_bpermute_b32 v70, v221, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]
	v_add_f32_e32 v0, v0, v70
	v_fmac_f32_e32 v0, v66, v118
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[68:69], v[18:33]
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	scratch_load_dword v69, off, off offset:20 ; 4-byte Folded Reload
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v0
	v_mov_b32_e32 v67, 0x42000000
	v_or_b32_e32 v66, s28, v80
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v0, v68
	v_log_f32_e32 v68, v68
	s_movk_i32 s1, 0x4000
	v_cndmask_b32_e32 v67, 0, v67, vcc
	v_cmp_gt_i32_e64 s[4:5], s1, v66
	v_sub_f32_e32 v66, v68, v67
	v_add_f32_e32 v66, v129, v66
	ds_write_b32 v1, v66
	v_mov_b32_e32 v66, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_sub_i32 s1, 0x4000, s28
	s_mov_b32 s2, s6
	s_mov_b32 s3, s7
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v66, v66, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v67, 0, v66
	ds_read_b32 v67, v67
	v_and_b32_e32 v68, 0x100, v69
	v_cmp_lt_i32_sdwa s[12:13], v69, s1 src0_sel:BYTE_0 src1_sel:DWORD
	v_cmp_eq_u32_e32 vcc, 0, v68
	v_bfrev_b32_e32 v68, 1
	s_and_b64 vcc, vcc, s[12:13]
	s_and_b32 s1, s8, 0xffff
	v_cndmask_b32_e32 v66, v68, v66, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v66, s[0:3], 0 offen
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr4_sgpr5
.LBB0_9:
	s_mov_b32 s1, 0x800000
	v_cmp_gt_f32_e32 vcc, s1, v0
	v_mov_b32_e32 v66, 0x42000000
	s_movk_i32 s6, 0x100
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v0, v67
	v_log_f32_e32 v67, v67
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_and_b32 s1, s8, 0xffff
	s_mov_b32 s3, 0x27000
	v_sub_f32_e32 v66, v67, v66
	v_add_f32_e32 v66, v129, v66
	ds_write_b32 v1, v66
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v68, off, off offset:20 ; 4-byte Folded Reload
	v_mov_b32_e32 v1, 2
	v_bfrev_b32_e32 v67, 1
	s_mov_b32 s2, 0x7ffffffe
	s_or_b64 s[4:5], s[4:5], exec
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v1, v1, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v1
	ds_read_b32 v66, v66
	v_cmp_gt_u32_e32 vcc, s6, v68
	s_nop 1
	v_cndmask_b32_e32 v1, v67, v1, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v1, s[0:3], 0 offen
.LBB0_10:                               ; %.critedge
	v_div_scale_f32 v1, s[0:1], v0, v0, 1.0
	v_rcp_f32_e32 v66, v1
	v_div_scale_f32 v67, vcc, 1.0, v0, 1.0
	s_mul_i32 s0, s20, s18
	v_fma_f32 v68, -v1, v66, 1.0
	v_fmac_f32_e32 v66, v68, v66
	v_mul_f32_e32 v68, v67, v66
	v_fma_f32 v69, -v1, v68, v67
	v_fmac_f32_e32 v68, v69, v66
	v_fma_f32 v1, -v1, v68, v67
	v_div_fmas_f32 v1, v1, v66, v68
	v_div_fixup_f32 v0, v1, v0, 1.0
	v_mov_b32_e32 v66, v31
	v_mov_b32_e32 v67, v32
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
	v_mov_b32_e32 v18, v15
	v_mov_b32_e32 v19, v16
	v_fma_mixlo_f16 v23, v0, v65, 0
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
	v_fma_mixlo_f16 v62, v0, v62, 0
	v_fma_mixlo_f16 v61, v0, v61, 0
	v_fma_mixlo_f16 v58, v0, v58, 0
	v_fma_mixlo_f16 v57, v0, v57, 0
	v_fma_mixlo_f16 v54, v0, v54, 0
	v_fma_mixlo_f16 v53, v0, v53, 0
	v_fma_mixlo_f16 v50, v0, v50, 0
	v_fma_mixlo_f16 v49, v0, v49, 0
	v_fma_mixlo_f16 v46, v0, v46, 0
	v_fma_mixlo_f16 v45, v0, v45, 0
	v_fma_mixlo_f16 v42, v0, v42, 0
	v_fma_mixlo_f16 v41, v0, v41, 0
	v_fma_mixlo_f16 v38, v0, v38, 0
	v_fma_mixlo_f16 v37, v0, v37, 0
	v_fma_mixlo_f16 v34, v0, v34, 0
	v_pk_mul_f32 v[4:5], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v0, v0, v2, 0
	scratch_load_dword v2, off, off offset:24 ; 4-byte Folded Reload
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s21, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s16, s28
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_cvt_f16_f32_e32 v3, v5
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_f16_f32_e32 v4, v4
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s16, 0x3fff
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v8, v8
	v_mul_lo_u32 v5, s16, v80
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s6, 0x5040100
	s_or_b32 s1, s2, s1
	v_perm_b32 v1, v1, v3, s6
	v_bfrev_b32_e32 v3, 1
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
	v_lshrrev_b32_e32 v2, 3, v2
	v_add_lshl_u32 v2, v5, v2, 1
	v_cndmask_b32_e64 v4, v3, v2, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 16, v2
	v_pack_b32_f16 v0, v10, v8
	v_perm_b32 v1, v15, v9, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 32, v2
	v_pack_b32_f16 v0, v14, v12
	v_perm_b32 v1, v19, v13, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 48, v2
	v_pack_b32_f16 v0, v18, v16
	v_perm_b32 v1, v65, v17, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 64, v2
	v_pack_b32_f16 v0, v34, v36
	v_perm_b32 v1, v37, v35, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x50, v2
	v_pack_b32_f16 v0, v38, v40
	v_perm_b32 v1, v41, v39, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x60, v2
	v_pack_b32_f16 v0, v42, v44
	v_perm_b32 v1, v45, v43, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x70, v2
	v_pack_b32_f16 v0, v46, v48
	v_perm_b32 v1, v49, v47, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x80, v2
	v_pack_b32_f16 v0, v50, v52
	v_perm_b32 v1, v53, v51, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x90, v2
	v_pack_b32_f16 v0, v54, v56
	v_perm_b32 v1, v57, v55, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xa0, v2
	v_pack_b32_f16 v0, v58, v60
	v_perm_b32 v1, v61, v59, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xb0, v2
	v_pack_b32_f16 v0, v62, v64
	v_perm_b32 v1, v23, v63, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xc0, v2
	v_pack_b32_f16 v0, v22, v20
	v_perm_b32 v1, v27, v21, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xd0, v2
	v_pack_b32_f16 v0, v26, v24
	v_perm_b32 v1, v31, v25, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xe0, v2
	v_pack_b32_f16 v0, v30, v28
	v_perm_b32 v1, v67, v29, s6
	v_cndmask_b32_e64 v4, v3, v4, s[4:5]
	v_add_u32_e32 v2, 0xf0, v2
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_pack_b32_f16 v0, v66, v32
	v_perm_b32 v1, v68, v33, s6
	v_cndmask_b32_e64 v2, v3, v2, s[4:5]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 188
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
		.amdhsa_next_free_sgpr 54
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
	.set attn_fwd.numbered_sgpr, 54
	.set attn_fwd.private_seg_size, 188
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13248
; TotalNumSgprs: 60
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 188
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 60
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
	.short	680                             ; DW_AT_call_line
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
    .private_segment_fixed_size: 188
    .sgpr_count:     60
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 46
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
