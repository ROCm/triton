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
; %bb.28:
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.29:
.LBB0_0:
	s_load_dwordx4 s[20:23], s[0:1], 0x70
	s_ashr_i32 s19, s18, 31
	s_lshl_b32 s60, s16, 8
	s_lshl_b64 s[24:25], s[18:19], 2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s20, s20, s24
	s_addc_u32 s21, s21, s25
	s_load_dwordx2 s[68:69], s[20:21], 0x0
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s33, s69, s68
	s_cmp_gt_i32 s60, s33
	s_cbranch_scc1 .LBB0_27
; %bb.1:
	s_add_u32 s20, s22, s24
	s_addc_u32 s21, s23, s25
	s_load_dwordx2 s[66:67], s[20:21], 0x0
	s_load_dwordx8 s[36:43], s[0:1], 0x38
	v_lshrrev_b32_e32 v40, 4, v0
	v_or_b32_e32 v9, s60, v40
	v_and_b32_e32 v1, 0x100, v0
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s67, s67, s66
	s_add_i32 s16, s67, 63
	s_ashr_i32 s19, s16, 31
	s_lshr_b32 s19, s19, 26
	s_add_i32 s16, s16, s19
	s_sub_i32 s19, s60, s33
	s_add_i32 s19, s19, s67
	s_addk_i32 s19, 0x13f
	s_ashr_i32 s20, s19, 31
	s_lshr_b32 s20, s20, 26
	s_add_i32 s19, s19, s20
	s_ashr_i32 s16, s16, 6
	s_ashr_i32 s19, s19, 6
	v_or_b32_e32 v8, 32, v9
	v_or_b32_e32 v6, 64, v9
	v_or_b32_e32 v4, 0x60, v9
	v_or_b32_e32 v7, 0x80, v9
	v_or_b32_e32 v5, 0xa0, v9
	v_or_b32_e32 v3, 0xc0, v9
	v_or_b32_e32 v2, 0xe0, v9
	s_min_i32 s69, s16, s19
	s_mov_b32 s44, 0
	v_cmp_eq_u32_e64 s[0:1], 0, v1
	s_cmp_gt_i32 s69, 0
	s_mul_i32 s58, s41, s18
	s_mul_i32 s56, s42, s17
	s_mul_i32 s54, s68, s43
	s_mul_i32 s52, s43, s60
	v_cmp_gt_i32_e64 s[26:27], s33, v9
	v_cmp_gt_i32_e64 s[24:25], s33, v8
	v_cmp_gt_i32_e64 s[22:23], s33, v6
	v_cmp_gt_i32_e64 s[20:21], s33, v4
	v_cmp_gt_i32_e64 s[30:31], s33, v7
	v_cmp_gt_i32_e64 s[28:29], s33, v5
	v_cmp_gt_i32_e64 s[34:35], s33, v3
	v_cmp_gt_i32_e32 vcc, s33, v2
	v_lshlrev_b32_e32 v206, 3, v0
	s_mul_i32 s64, s18, 0xeb200
	s_mul_i32 s62, s17, 0x1d64
	s_cbranch_scc1 .LBB0_3
; %bb.2:
	s_ashr_i32 s59, s58, 31
	s_lshl_b64 s[46:47], s[58:59], 1
	s_add_u32 s16, s10, s46
	s_addc_u32 s19, s11, s47
	s_ashr_i32 s57, s56, 31
	s_lshl_b64 s[46:47], s[56:57], 1
	s_add_u32 s16, s16, s46
	s_addc_u32 s19, s19, s47
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[46:47], s[54:55], 1
	s_add_u32 s16, s16, s46
	s_addc_u32 s19, s19, s47
	s_ashr_i32 s53, s52, 31
	s_lshl_b32 s41, s43, 5
	s_lshl_b64 s[46:47], s[52:53], 1
	s_add_u32 s48, s16, s46
	v_and_b32_e32 v10, 0x78, v206
	s_addc_u32 s16, s19, s47
	v_mad_u64_u32 v[10:11], s[46:47], s43, v40, v[10:11]
	s_and_b32 s19, s43, 0x3fff
	v_add_u32_e32 v14, s41, v10
	s_bitset1_b32 s19, 14
	v_lshlrev_b32_e32 v10, 1, v10
	v_bfrev_b32_e32 v20, 1
	s_mov_b32 s45, s44
	v_add_u32_e32 v15, s41, v14
	s_and_b32 s16, s16, 0xffff
	s_lshl_b32 s19, s19, 16
	v_cndmask_b32_e64 v21, v20, v10, s[26:27]
	s_mov_b32 s46, s44
	s_mov_b32 s47, s44
	v_mov_b64_e32 v[10:11], s[44:45]
	v_lshlrev_b32_e32 v14, 1, v14
	s_or_b32 s49, s16, s19
	s_mov_b32 s51, 0x27000
	s_mov_b32 s50, 0x7ffffffe
	v_mov_b64_e32 v[12:13], s[46:47]
	v_cndmask_b32_e64 v14, v20, v14, s[24:25]
	buffer_store_dwordx4 v[10:13], v21, s[48:51], 0 offen
	buffer_store_dwordx4 v[10:13], v14, s[48:51], 0 offen
	v_lshlrev_b32_e32 v14, 1, v15
	v_add_u32_e32 v16, s41, v15
	v_cndmask_b32_e64 v14, v20, v14, s[22:23]
	buffer_store_dwordx4 v[10:13], v14, s[48:51], 0 offen
	v_lshlrev_b32_e32 v14, 1, v16
	v_add_u32_e32 v17, s41, v16
	v_cndmask_b32_e64 v14, v20, v14, s[20:21]
	buffer_store_dwordx4 v[10:13], v14, s[48:51], 0 offen
	v_lshlrev_b32_e32 v14, 1, v17
	s_ashr_i32 s65, s64, 31
	v_add_u32_e32 v18, s41, v17
	v_cndmask_b32_e64 v14, v20, v14, s[30:31]
	s_lshl_b64 s[20:21], s[64:65], 2
	buffer_store_dwordx4 v[10:13], v14, s[48:51], 0 offen
	v_lshlrev_b32_e32 v14, 1, v18
	s_add_u32 s16, s8, s20
	v_add_u32_e32 v19, s41, v18
	v_cndmask_b32_e64 v14, v20, v14, s[28:29]
	s_addc_u32 s19, s9, s21
	s_ashr_i32 s63, s62, 31
	buffer_store_dwordx4 v[10:13], v14, s[48:51], 0 offen
	v_lshlrev_b32_e32 v14, 1, v19
	s_lshl_b64 s[20:21], s[62:63], 2
	v_cndmask_b32_e64 v14, v20, v14, s[34:35]
	s_add_u32 s16, s16, s20
	buffer_store_dwordx4 v[10:13], v14, s[48:51], 0 offen
	v_add_lshl_u32 v14, v19, s41, 1
	s_addc_u32 s19, s19, s21
	s_ashr_i32 s61, s60, 31
	v_or_b32_sdwa v1, s60, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_cndmask_b32_e32 v14, v20, v14, vcc
	s_movk_i32 s22, 0x1d64
	s_lshl_b64 s[20:21], s[60:61], 2
	buffer_store_dwordx4 v[10:13], v14, s[48:51], 0 offen
	s_add_u32 s48, s16, s20
	v_cmp_gt_i32_e32 vcc, s22, v1
	v_mov_b32_e32 v1, 2
	s_addc_u32 s16, s19, s21
	v_lshlrev_b32_sdwa v1, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	s_and_b64 vcc, s[0:1], vcc
	s_and_b32 s49, s16, 0xffff
	v_cndmask_b32_e32 v1, v20, v1, vcc
	v_mov_b32_e32 v10, 0x7f800000
	buffer_store_dword v10, v1, s[48:51], 0 offen
.LBB0_3:
	s_cmp_lt_i32 s69, 1
	s_cbranch_scc1 .LBB0_27
; %bb.4:
	s_ashr_i32 s16, s17, 31
	s_lshr_b32 s16, s16, 28
	s_add_i32 s16, s17, s16
	s_ashr_i32 s28, s16, 4
	s_and_b32 s16, s67, 63
	s_sub_i32 s19, 64, s67
	s_cmp_lt_i32 s67, 64
	s_mul_i32 s20, s12, s18
	s_cselect_b32 s41, s19, s16
	s_ashr_i32 s21, s20, 31
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s12, s2, s20
	s_mul_i32 s2, s13, s17
	s_addc_u32 s16, s3, s21
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s68, s14
	s_addc_u32 s13, s16, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s60
	s_addc_u32 s13, s13, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s16, s14, 5
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s20, s12, s2
	v_and_b32_e32 v68, 0x78, v206
	s_addc_u32 s12, s13, s3
	v_mad_u64_u32 v[10:11], s[2:3], s14, v40, v[68:69]
	v_add_u32_e32 v11, s16, v10
	v_lshlrev_b32_e32 v10, 1, v10
	v_bfrev_b32_e32 v1, 1
	v_cmp_gt_i32_e32 vcc, s33, v9
	v_add_u32_e32 v16, s16, v11
	s_and_b32 s2, s14, 0x3fff
	v_cndmask_b32_e32 v18, v1, v10, vcc
	v_lshlrev_b32_e32 v9, 1, v11
	v_cmp_gt_i32_e32 vcc, s33, v8
	v_add_u32_e32 v17, s16, v16
	s_bitset1_b32 s2, 14
	v_cndmask_b32_e32 v19, v1, v9, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	v_cmp_gt_i32_e32 vcc, s33, v6
	s_and_b32 s3, s12, 0xffff
	s_lshl_b32 s2, s2, 16
	v_cndmask_b32_e32 v6, v1, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v17
	v_cmp_gt_i32_e32 vcc, s33, v4
	v_add_u32_e32 v24, s16, v17
	s_or_b32 s21, s3, s2
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v4, v1, v16, vcc
	v_add_u32_e32 v25, s16, v24
	buffer_load_dwordx4 v[8:11], v18, s[20:23], 0 offen
	buffer_load_dwordx4 v[12:15], v19, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[16:19], v6, s[20:23], 0 offen
	buffer_load_dwordx4 v[20:23], v4, s[20:23], 0 offen
	v_lshlrev_b32_e32 v4, 1, v24
	v_cmp_gt_i32_e32 vcc, s33, v7
	v_add_u32_e32 v32, s16, v25
	v_lshlrev_b32_e32 v6, 1, v25
	v_cndmask_b32_e32 v4, v1, v4, vcc
	v_cmp_gt_i32_e32 vcc, s33, v5
	v_lshrrev_b32_e32 v44, 1, v0
	v_lshlrev_b32_e32 v41, 7, v40
	v_cndmask_b32_e32 v5, v1, v6, vcc
	buffer_load_dwordx4 v[24:27], v4, s[20:23], 0 offen
	buffer_load_dwordx4 v[28:31], v5, s[20:23], 0 offen
	v_lshlrev_b32_e32 v4, 1, v32
	v_cmp_gt_i32_e32 vcc, s33, v3
	s_movk_i32 s12, 0x78
	v_or_b32_e32 v7, 0x1000, v41
	v_cndmask_b32_e32 v3, v1, v4, vcc
	v_add_lshl_u32 v4, v32, s16, 1
	v_cmp_gt_i32_e32 vcc, s33, v2
	v_mad_u64_u32 v[200:201], s[2:3], s40, v40, v[68:69]
	s_nop 0
	v_cndmask_b32_e32 v2, v1, v4, vcc
	buffer_load_dwordx4 v[32:35], v3, s[20:23], 0 offen
	buffer_load_dwordx4 v[36:39], v2, s[20:23], 0 offen
	v_and_b32_e32 v3, 0x78, v44
	v_lshrrev_b32_e32 v2, 3, v0
	v_bitop3_b32 v3, v3, v41, v68 bitop3:0xde
	v_and_b32_e32 v2, 4, v2
	v_lshlrev_b32_e32 v6, 1, v3
	scratch_store_dword off, v2, off offset:8 ; 4-byte Folded Spill
	v_bitop3_b32 v2, v44, v206, s12 bitop3:0x28
	v_add_u32_e32 v3, 0, v6
	s_barrier
	scratch_store_dword off, v3, off offset:28 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v4, 8, v40
	v_mad_u64_u32 v[202:203], s[2:3], s37, v40, v[68:69]
	v_and_b32_e32 v43, 31, v0
	s_movk_i32 s2, 0xe0
	s_mul_i32 s16, s15, s18
	s_mul_i32 s24, s36, s28
	s_mul_i32 s26, s66, s37
	s_mul_i32 s18, s38, s18
	s_mul_i32 s28, s39, s28
	s_mul_i32 s30, s66, s40
	s_ashr_i32 s17, s16, 31
	s_ashr_i32 s25, s24, 31
	s_ashr_i32 s27, s26, 31
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s29, s28, 31
	s_ashr_i32 s31, s30, 31
	v_and_b32_e32 v216, 16, v0
	v_and_b32_e32 v208, 32, v206
	v_and_b32_e32 v207, 64, v206
	v_lshl_add_u32 v42, s37, 5, v202
	v_lshl_add_u32 v209, s40, 5, v200
	s_movk_i32 s3, 0x60
	v_lshlrev_b32_e32 v67, 1, v0
	s_waitcnt vmcnt(9)
	ds_write_b128 v3, v[8:11]
	v_or_b32_e32 v3, v7, v2
	v_lshlrev_b32_e32 v5, 1, v3
	v_lshlrev_b32_e32 v3, 1, v2
	v_add_u32_e32 v201, 0, v5
	v_add3_u32 v3, 0, v3, v4
	s_waitcnt vmcnt(8)
	ds_write_b128 v201, v[12:15]
	s_waitcnt vmcnt(7)
	ds_write_b128 v3, v[16:19] offset:16384
	s_waitcnt vmcnt(6)
	ds_write_b128 v3, v[20:23] offset:24576
	s_waitcnt vmcnt(5)
	ds_write_b128 v3, v[24:27] offset:32768
	s_waitcnt vmcnt(4)
	ds_write_b128 v3, v[28:31] offset:40960
	s_waitcnt vmcnt(3)
	ds_write_b128 v3, v[32:35] offset:49152
	s_waitcnt vmcnt(2)
	ds_write_b128 v3, v[36:39] offset:57344
	v_lshrrev_b32_e32 v3, 2, v0
	v_and_b32_e32 v3, 8, v3
	v_or_b32_e32 v9, 16, v3
	v_or_b32_e32 v10, 32, v3
	v_or_b32_e32 v11, 48, v3
	v_or_b32_e32 v12, 64, v3
	v_or_b32_e32 v13, 0x50, v3
	v_or_b32_e32 v14, 0x60, v3
	v_or_b32_e32 v15, 0x70, v3
	v_and_b32_e32 v3, 15, v0
	v_lshrrev_b32_e32 v4, 5, v0
	v_bitop3_b32 v4, v4, v3, 1 bitop3:0x6c
	v_lshrrev_b32_e32 v3, 3, v9
	v_lshrrev_b32_e32 v9, 3, v10
	v_bitop3_b32 v23, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v11
	v_bitop3_b32 v22, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v12
	v_bitop3_b32 v21, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v13
	v_and_or_b32 v8, v44, s2, v43
	v_bitop3_b32 v20, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v14
	v_bitop3_b32 v3, v3, v0, 15 bitop3:0x78
	v_bitop3_b32 v19, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v15
	v_lshl_add_u32 v8, v8, 8, 0
	v_lshlrev_b32_e32 v218, 4, v4
	v_bitop3_b32 v18, v9, v0, 15 bitop3:0x78
	v_add_u32_e32 v9, v8, v218
	v_lshlrev_b32_e32 v219, 4, v3
	v_lshlrev_b32_e32 v220, 4, v23
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v10, v8, v219
	ds_read_b128 v[130:133], v9
	ds_read_b128 v[134:137], v10
	v_add_u32_e32 v9, v8, v220
	v_lshlrev_b32_e32 v221, 4, v22
	v_lshlrev_b32_e32 v223, 4, v21
	v_lshlrev_b32_e32 v229, 4, v20
	v_lshlrev_b32_e32 v198, 4, v19
	v_lshlrev_b32_e32 v217, 4, v18
	v_add_u32_e32 v10, v8, v221
	v_add_u32_e32 v11, v8, v223
	v_add_u32_e32 v12, v8, v229
	v_add_u32_e32 v13, v8, v198
	v_add_u32_e32 v8, v8, v217
	ds_read_b128 v[138:141], v9
	ds_read_b128 v[142:145], v10
	ds_read_b128 v[146:149], v11
	ds_read_b128 v[150:153], v12
	ds_read_b128 v[154:157], v13
	ds_read_b128 v[158:161], v8
	s_and_b32 s2, s33, 0xff
	s_or_b32 s2, s41, s2
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, 4, 5
	v_mov_b32_e32 v14, s2
	v_sub_u32_e64 v14, s69, v14 clamp
	s_cmp_le_u32 s69, s2
	v_readfirstlane_b32 s36, v14
	v_lshlrev_b32_e32 v32, 8, v43
	scratch_store_dwordx2 off, v[68:69], off offset:20 ; 8-byte Folded Spill
	scratch_store_dword off, v41, off offset:12 ; 4-byte Folded Spill
	scratch_store_dword off, v42, off offset:16 ; 4-byte Folded Spill
	scratch_store_dword off, v32, off offset:32 ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_11
; %bb.5:
	v_or_b32_e32 v231, v41, v68
	v_lshlrev_b32_e32 v224, 1, v231
	v_or_b32_e32 v7, v7, v68
	v_sub_u32_e32 v6, v6, v224
	v_lshlrev_b32_e32 v225, 1, v7
	v_ashrrev_i16_e32 v7, 15, v6
	v_lshrrev_b16_e32 v7, 12, v7
	v_add_u16_e32 v6, v6, v7
	v_and_b32_e32 v230, 63, v0
	v_ashrrev_i16_e32 v6, 4, v6
	s_lshl_b64 s[12:13], s[16:17], 1
	v_add_u32_sdwa v6, v230, sext(v6) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_add_u32 s2, s4, s12
	v_lshlrev_b32_e32 v7, 2, v6
	s_addc_u32 s14, s5, s13
	s_lshl_b64 s[12:13], s[24:25], 1
	ds_bpermute_b32 v8, v7, v202
	v_lshrrev_b64 v[6:7], v6, exec
	s_add_u32 s2, s2, s12
	v_and_b32_e32 v6, 1, v6
	v_sub_u32_e32 v5, v5, v225
	s_addc_u32 s14, s14, s13
	s_lshl_b64 s[12:13], s[26:27], 1
	v_cmp_eq_u32_e32 vcc, 1, v6
	v_ashrrev_i16_e32 v6, 15, v5
	s_add_u32 s20, s2, s12
	v_lshrrev_b16_e32 v6, 12, v6
	s_addc_u32 s34, s14, s13
	s_lshl_b64 s[12:13], s[18:19], 1
	v_add_u16_e32 v5, v5, v6
	s_add_u32 s2, s6, s12
	v_ashrrev_i16_e32 v5, 4, v5
	s_addc_u32 s14, s7, s13
	s_lshl_b64 s[12:13], s[28:29], 1
	v_add_u32_sdwa v5, v230, sext(v5) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_add_u32 s2, s2, s12
	v_lshlrev_b32_e32 v6, 2, v5
	s_addc_u32 s14, s14, s13
	s_lshl_b64 s[12:13], s[30:31], 1
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v7, 1, v8
	ds_bpermute_b32 v8, v6, v42
	s_add_u32 s38, s2, s12
	v_sub_u32_e32 v2, v2, v68
	s_addc_u32 s39, s14, s13
	s_and_b32 s13, s37, 0x3fff
	v_ashrrev_i32_e32 v2, 3, v2
	s_bitset1_b32 s13, 14
	v_cndmask_b32_e32 v64, v1, v7, vcc
	v_lshrrev_b64 v[6:7], v5, exec
	v_add_u32_e32 v2, v2, v230
	v_add_u32_e32 v62, 0, v224
	s_and_b32 s14, s34, 0xffff
	s_lshl_b32 s48, s13, 16
	v_and_b32_e32 v5, 1, v6
	v_lshlrev_b32_e32 v246, 2, v2
	s_lshl_b32 s12, s37, 6
	v_add_u32_e32 v63, 0, v225
	s_or_b32 s21, s14, s48
	v_readfirstlane_b32 s14, v62
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v6, 1, v8
	v_cmp_eq_u32_e32 vcc, 1, v5
	ds_bpermute_b32 v9, v246, v202
	s_mov_b32 m0, s14
	v_cndmask_b32_e32 v65, v1, v6, vcc
	v_readfirstlane_b32 s15, v63
	s_ashr_i32 s13, s12, 31
	v_lshrrev_b64 v[6:7], v2, exec
	s_lshl_b32 s2, s40, 6
	buffer_load_dwordx4 v64, s[20:23], 0 offen lds
	s_mov_b32 m0, s15
	s_lshl_b64 s[44:45], s[12:13], 1
	ds_bpermute_b32 v7, v246, v42
	buffer_load_dwordx4 v65, s[20:23], 0 offen lds
	s_add_u32 s20, s20, s44
	s_addc_u32 s13, s34, s45
	v_add_u32_e32 v5, 0x4000, v62
	v_and_b32_e32 v2, 1, v6
	s_and_b32 s21, s13, 0xffff
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v6, 1, v9
	v_cmp_eq_u32_e32 vcc, 1, v2
	v_readfirstlane_b32 s34, v5
	v_add_u32_e32 v8, 0x4000, v63
	s_or_b32 s21, s21, s48
	v_cndmask_b32_e32 v2, v1, v6, vcc
	s_mov_b32 m0, s34
	v_readfirstlane_b32 s34, v8
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v7
	v_lshl_or_b32 v70, v4, 4, v32
	v_cndmask_b32_e32 v2, v1, v2, vcc
	s_mov_b32 m0, s34
	v_add_u32_e32 v66, 0, v70
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b128 v[4:7], v66
	ds_read_b128 v[24:27], v66 offset:8192
	v_lshl_or_b32 v71, v3, 4, v32
	v_add_u32_e32 v72, 0, v71
	ds_read_b128 v[28:31], v72
	ds_read_b128 v[34:37], v72 offset:8192
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[4:7], v[130:133], 0
	v_lshl_or_b32 v73, v23, 4, v32
	v_add_u32_e32 v74, 0, v73
	v_lshl_or_b32 v75, v22, 4, v32
	v_add_u32_e32 v76, 0, v75
	ds_read_b128 v[38:41], v74 offset:8192
	v_lshl_or_b32 v77, v21, 4, v32
	v_add_u32_e32 v78, 0, v77
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[134:137], v[2:17]
	ds_read_b128 v[28:31], v74
	ds_read_b128 v[42:45], v76 offset:8192
	v_lshl_or_b32 v79, v20, 4, v32
	v_add_u32_e32 v80, 0, v79
	ds_read_b128 v[20:23], v80
	ds_read_b128 v[50:53], v80 offset:8192
	ds_read_b128 v[46:49], v78 offset:8192
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[138:141], v[2:17]
	ds_read_b128 v[28:31], v76
	v_lshl_or_b32 v81, v19, 4, v32
	v_add_u32_e32 v82, 0, v81
	ds_read_b128 v[54:57], v82 offset:8192
	v_lshl_or_b32 v83, v18, 4, v32
	v_add_u32_e32 v84, 0, v83
	v_bitop3_b32 v211, v68, v67, s3 bitop3:0x78
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[142:145], v[2:17]
	ds_read_b128 v[28:31], v78
	s_and_b32 s3, s40, 0x3fff
	s_bitset1_b32 s3, 14
	s_and_b32 s21, s39, 0xffff
	s_lshl_b32 s42, s3, 16
	s_or_b32 s73, s21, s42
	s_mov_b32 s72, s38
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[146:149], v[2:17]
	s_mov_b32 s74, s22
	s_mov_b32 s75, s23
	s_add_u32 s20, s20, s44
	s_addc_u32 s13, s13, s45
	v_mfma_f32_32x32x16_f16 v[2:17], v[20:23], v[150:153], v[2:17]
	ds_read_b128 v[20:23], v82
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[20:23], v[154:157], v[2:17]
	ds_read_b128 v[18:21], v84
	ds_read_b128 v[58:61], v84 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[158:161], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[24:27], v[130:133], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[34:37], v[134:137], v[18:33]
	v_sub_u32_e32 v34, v211, v68
	v_ashrrev_i32_e32 v34, 3, v34
	v_add_u32_e32 v205, v34, v230
	v_lshlrev_b32_e32 v199, 2, v205
	v_lshrrev_b64 v[34:35], v205, exec
	v_and_b32_e32 v34, 1, v34
	v_cmp_eq_u32_e32 vcc, 1, v34
	v_mfma_f32_32x32x16_f16 v[18:33], v[38:41], v[138:141], v[18:33]
	ds_bpermute_b32 v38, v199, v200
	v_add_u32_e32 v36, 0x8000, v62
	v_add_u32_e32 v37, 0x8000, v63
	v_readfirstlane_b32 s3, v36
	s_mov_b32 m0, s3
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v35, 1, v38
	v_cndmask_b32_e32 v34, v1, v35, vcc
	ds_bpermute_b32 v35, v199, v209
	v_readfirstlane_b32 s3, v37
	buffer_load_dwordx4 v34, s[72:75], 0 offen lds
	s_mov_b32 m0, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[34:35], s[2:3], 1
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v35, 1, v35
	s_add_u32 s46, s38, s34
	v_cndmask_b32_e32 v1, v1, v35, vcc
	s_addc_u32 s47, s39, s35
	s_and_b32 s2, s13, 0xffff
	buffer_load_dwordx4 v1, s[72:75], 0 offen lds
	s_or_b32 s21, s2, s48
	s_mov_b32 m0, s14
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v35, v3, v3
	v_max_f32_e32 v36, v2, v2
	buffer_load_dwordx4 v64, s[20:23], 0 offen lds
	s_mov_b32 m0, s15
	v_max_f32_e32 v35, v36, v35
	v_add_u32_e32 v36, 0xc000, v62
	s_and_b32 s2, s47, 0xffff
	buffer_load_dwordx4 v65, s[20:23], 0 offen lds
	v_add_u32_e32 v37, 0xc000, v63
	s_or_b32 s21, s2, s42
	v_readfirstlane_b32 s2, v36
	s_mov_b32 s20, s46
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v37
	s_waitcnt vmcnt(4)
	s_barrier
	buffer_load_dwordx4 v34, s[20:23], 0 offen lds
	s_mov_b32 m0, s2
	v_mfma_f32_32x32x16_f16 v[18:33], v[42:45], v[142:145], v[18:33]
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	v_max3_f32 v35, v35, v4, v5
	v_max3_f32 v35, v35, v6, v7
	v_max3_f32 v35, v35, v8, v9
	v_max3_f32 v35, v35, v10, v11
	v_max3_f32 v35, v35, v12, v13
	v_max3_f32 v35, v35, v14, v15
	v_mfma_f32_32x32x16_f16 v[18:33], v[46:49], v[146:149], v[18:33]
	v_max3_f32 v35, v35, v16, v17
	v_mov_b32_e32 v34, 0xff800000
	s_add_i32 s2, 0, 0x4000
	ds_read_b128 v[66:69], v66 offset:16384
	ds_read_b128 v[186:189], v72 offset:16384
	ds_read_b128 v[182:185], v74 offset:16384
	ds_read_b128 v[178:181], v76 offset:16384
	ds_read_b128 v[174:177], v78 offset:16384
	ds_read_b128 v[106:109], v80 offset:16384
	v_add_u32_e32 v36, s2, v73
	v_add_u32_e32 v37, s2, v75
	v_add_u32_e32 v38, s2, v77
	v_mfma_f32_32x32x16_f16 v[18:33], v[50:53], v[150:153], v[18:33]
	v_add_u32_e32 v39, s2, v79
	v_add_u32_e32 v40, s2, v81
	v_add_u32_e32 v41, s2, v83
	v_add_u32_e32 v42, 0xff, v0
	v_mfma_f32_32x32x16_f16 v[18:33], v[54:57], v[154:157], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[58:61], v[158:161], v[18:33]
	s_nop 7
	s_nop 3
	v_max3_f32 v35, v35, v18, v19
	v_max3_f32 v1, v35, v20, v21
	v_max3_f32 v1, v1, v22, v23
	v_max3_f32 v1, v1, v24, v25
	v_max3_f32 v1, v1, v26, v27
	v_max3_f32 v1, v1, v28, v29
	v_max3_f32 v1, v1, v30, v31
	v_max3_f32 v1, v1, v32, v33
	v_mov_b32_e32 v35, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v35
	v_max3_f32 v194, v1, v35, v34
	v_mul_f32_e32 v35, 0xbe0293ee, v194
	v_fmamk_f32 v1, v2, 0x3e0293ee, v35
	v_fmamk_f32 v2, v3, 0x3e0293ee, v35
	v_fmamk_f32 v3, v4, 0x3e0293ee, v35
	v_fmamk_f32 v4, v5, 0x3e0293ee, v35
	v_fmamk_f32 v5, v6, 0x3e0293ee, v35
	v_fmamk_f32 v6, v7, 0x3e0293ee, v35
	v_fmamk_f32 v7, v8, 0x3e0293ee, v35
	v_fmamk_f32 v8, v9, 0x3e0293ee, v35
	v_fmamk_f32 v9, v10, 0x3e0293ee, v35
	v_fmamk_f32 v10, v11, 0x3e0293ee, v35
	v_fmamk_f32 v11, v12, 0x3e0293ee, v35
	v_fmamk_f32 v12, v13, 0x3e0293ee, v35
	v_fmamk_f32 v13, v14, 0x3e0293ee, v35
	v_fmamk_f32 v14, v15, 0x3e0293ee, v35
	v_fmamk_f32 v15, v16, 0x3e0293ee, v35
	v_fmamk_f32 v16, v17, 0x3e0293ee, v35
	v_fmamk_f32 v17, v18, 0x3e0293ee, v35
	v_fmamk_f32 v18, v19, 0x3e0293ee, v35
	v_fmamk_f32 v19, v20, 0x3e0293ee, v35
	v_fmamk_f32 v20, v21, 0x3e0293ee, v35
	v_fmamk_f32 v21, v22, 0x3e0293ee, v35
	v_fmamk_f32 v22, v23, 0x3e0293ee, v35
	v_fmamk_f32 v23, v24, 0x3e0293ee, v35
	v_fmamk_f32 v24, v25, 0x3e0293ee, v35
	v_fmamk_f32 v25, v26, 0x3e0293ee, v35
	v_fmamk_f32 v26, v27, 0x3e0293ee, v35
	v_fmamk_f32 v27, v28, 0x3e0293ee, v35
	v_fmamk_f32 v28, v29, 0x3e0293ee, v35
	v_fmamk_f32 v29, v30, 0x3e0293ee, v35
	v_fmamk_f32 v30, v31, 0x3e0293ee, v35
	v_fmamk_f32 v31, v32, 0x3e0293ee, v35
	v_fmac_f32_e32 v35, 0x3e0293ee, v33
	v_add_u32_e32 v32, s2, v70
	v_add_u32_e32 v33, s2, v71
	ds_read_b128 v[110:113], v82 offset:16384
	ds_read_b128 v[102:105], v84 offset:16384
	ds_read_b128 v[98:101], v32 offset:8192
	ds_read_b128 v[170:173], v33 offset:8192
	ds_read_b128 v[166:169], v36 offset:8192
	ds_read_b128 v[162:165], v37 offset:8192
	ds_read_b128 v[126:129], v38 offset:8192
	ds_read_b128 v[122:125], v39 offset:8192
	ds_read_b128 v[118:121], v40 offset:8192
	ds_read_b128 v[114:117], v41 offset:8192
	s_movk_i32 s2, 0x1ff
	v_cmp_gt_u32_e32 vcc, s2, v42
	s_movk_i32 s2, 0x1fe
	v_fmac_f32_e32 v34, 0xbe0293ee, v194
	v_cmp_lt_u32_e64 s[2:3], s2, v42
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[14:15], s[2:3]
	s_cbranch_execz .LBB0_7
; %bb.6:
	s_barrier
.LBB0_7:
	s_or_b64 exec, exec, s[14:15]
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v227, v2
	v_exp_f32_e32 v247, v3
	v_exp_f32_e32 v249, v4
	v_exp_f32_e32 v248, v5
	v_exp_f32_e32 v251, v6
	v_exp_f32_e32 v250, v7
	v_exp_f32_e32 v253, v8
	v_exp_f32_e32 v252, v9
	v_exp_f32_e32 v255, v10
	v_exp_f32_e32 v254, v11
	v_exp_f32_e32 v222, v12
	v_exp_f32_e32 v191, v13
	v_exp_f32_e32 v193, v14
	v_exp_f32_e32 v192, v15
	v_exp_f32_e32 v196, v16
	v_exp_f32_e32 v195, v17
	v_exp_f32_e32 v232, v18
	v_exp_f32_e32 v197, v19
	v_exp_f32_e32 v237, v20
	v_exp_f32_e32 v236, v21
	v_exp_f32_e32 v239, v22
	v_exp_f32_e32 v233, v23
	v_exp_f32_e32 v235, v24
	v_exp_f32_e32 v234, v25
	v_exp_f32_e32 v238, v26
	v_exp_f32_e32 v240, v27
	v_exp_f32_e32 v242, v28
	v_exp_f32_e32 v241, v29
	v_exp_f32_e32 v244, v30
	v_exp_f32_e32 v243, v31
	v_exp_f32_e32 v245, v35
	v_exp_f32_e32 v190, v34
	v_and_b32_e32 v2, 31, v0
	v_mov_b32_e32 v49, 0
	s_cmp_lt_u32 s36, 4
	v_lshlrev_b32_e32 v228, 7, v2
	scratch_store_dword off, v223, off      ; 4-byte Folded Spill
	scratch_store_dword off, v229, off offset:4 ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_12
; %bb.8:                                ; %.lr.ph
	scratch_load_dword v3, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v210, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v215, off, off offset:16 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v2, 2, v0
	s_add_i32 s2, s36, -3
	v_and_b32_e32 v226, 12, v2
	s_movk_i32 s3, 0x60
	v_bitop3_b32 v213, v206, v226, s3 bitop3:0x4e
	s_mul_hi_i32 s3, s12, 6
	s_mul_i32 s14, s12, 6
	s_add_u32 s12, s16, s24
	s_addc_u32 s13, s17, s25
	s_add_u32 s12, s12, s26
	s_addc_u32 s13, s13, s27
	s_lshl_b64 s[12:13], s[12:13], 1
	v_lshrrev_b32_e32 v4, 2, v0
	s_add_u32 s12, s14, s12
	v_or_b32_e32 v2, v208, v226
	s_addc_u32 s13, s3, s13
	v_bitop3_b32 v212, v2, v207, 64 bitop3:0x36
	v_and_b32_e32 v2, 0x78, v206
	s_add_u32 s3, s4, s12
	v_mov_b32_e32 v34, 0
	scratch_store_dword off, v205, off offset:36 ; 4-byte Folded Spill
	s_addc_u32 s49, s5, s13
	s_mov_b32 s22, 0
	s_add_i32 s20, 0, 0x8000
	s_add_i32 s21, 0, 0xc000
	v_mov_b32_e32 v203, 1.0
	v_mov_b32_e32 v214, v198
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_mov_b32 s50, 0x3e0293ee
	s_mov_b32 s53, 0
	s_mov_b32 s51, 0
	v_mov_b32_e32 v35, v34
	v_mov_b32_e32 v36, v34
	v_mov_b32_e32 v37, v34
	v_mov_b32_e32 v38, v34
	v_mov_b32_e32 v39, v34
	v_mov_b32_e32 v40, v34
	v_mov_b32_e32 v41, v34
	v_mov_b32_e32 v42, v34
	v_mov_b32_e32 v43, v34
	v_mov_b32_e32 v44, v34
	v_mov_b32_e32 v45, v34
	v_mov_b32_e32 v46, v34
	v_mov_b32_e32 v47, v34
	v_mov_b32_e32 v48, v34
	v_mov_b32_e32 v49, v34
	v_mov_b32_e32 v5, v34
	v_mov_b32_e32 v6, v34
	v_mov_b32_e32 v7, v34
	v_mov_b32_e32 v8, v34
	v_mov_b32_e32 v9, v34
	v_mov_b32_e32 v10, v34
	v_mov_b32_e32 v11, v34
	v_mov_b32_e32 v12, v34
	v_mov_b32_e32 v13, v34
	v_mov_b32_e32 v14, v34
	v_mov_b32_e32 v15, v34
	v_mov_b32_e32 v16, v34
	v_mov_b32_e32 v17, v34
	v_mov_b32_e32 v18, v34
	v_mov_b32_e32 v19, v34
	v_mov_b32_e32 v20, v34
	v_mov_b32_e32 v21, v34
	v_mov_b32_e32 v22, v34
	v_mov_b32_e32 v23, v34
	v_mov_b32_e32 v24, v34
	v_mov_b32_e32 v25, v34
	v_mov_b32_e32 v26, v34
	v_mov_b32_e32 v27, v34
	v_mov_b32_e32 v28, v34
	v_mov_b32_e32 v29, v34
	v_mov_b32_e32 v30, v34
	v_mov_b32_e32 v31, v34
	v_mov_b32_e32 v32, v34
	v_mov_b32_e32 v33, v34
	v_mov_b32_e32 v50, v34
	s_waitcnt vmcnt(3)
	v_and_or_b32 v3, v4, 3, v3
	v_lshlrev_b32_e32 v3, 7, v3
	v_bitop3_b32 v4, v226, v208, 32 bitop3:0x36
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v205, v210, v2
	v_lshlrev_b32_e32 v198, 1, v3
	v_lshlrev_b32_e32 v204, 1, v4
	v_mov_b32_e32 v2, v34
	v_mov_b32_e32 v3, v34
	v_mov_b32_e32 v4, v34
	v_mov_b32_e32 v51, v34
	v_mov_b32_e32 v52, v34
	v_mov_b32_e32 v53, v34
	v_mov_b32_e32 v54, v34
	v_mov_b32_e32 v55, v34
	v_mov_b32_e32 v56, v34
	v_mov_b32_e32 v57, v34
	v_mov_b32_e32 v58, v34
	v_mov_b32_e32 v59, v34
	v_mov_b32_e32 v60, v34
	v_mov_b32_e32 v61, v34
	v_mov_b32_e32 v62, v34
	v_mov_b32_e32 v63, v34
	v_mov_b32_e32 v64, v34
	v_mov_b32_e32 v65, v34
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[130:133], 0
	v_mov_b32_e32 v229, v203
	v_mov_b32_e32 v223, v194
	s_mov_b64 s[38:39], s[46:47]
	s_mov_b32 s59, s20
	s_mov_b32 s20, s21
	s_mov_b32 s55, s22
	s_setprio 0
	v_add_f32_e32 v82, v1, v227
	v_add_f32_e32 v82, v82, v247
	v_add_f32_e32 v82, v82, v249
	v_add_f32_e32 v82, v82, v248
	v_add_f32_e32 v82, v82, v251
	v_add_f32_e32 v82, v82, v250
	v_add_f32_e32 v82, v82, v253
	v_add_f32_e32 v82, v82, v252
	v_add_f32_e32 v82, v82, v255
	v_add_f32_e32 v82, v82, v254
	v_add_f32_e32 v82, v82, v222
	v_add_f32_e32 v82, v82, v191
	v_add_f32_e32 v82, v82, v193
	v_add_f32_e32 v82, v82, v192
	v_add_f32_e32 v82, v82, v196
	v_add_f32_e32 v82, v82, v195
	v_add_f32_e32 v82, v82, v232
	v_add_f32_e32 v82, v82, v197
	v_add_f32_e32 v82, v82, v237
	v_add_f32_e32 v82, v82, v236
	v_add_f32_e32 v82, v82, v239
	v_add_f32_e32 v82, v82, v233
	v_add_f32_e32 v82, v82, v235
	v_add_f32_e32 v82, v82, v234
	v_add_f32_e32 v82, v82, v238
	v_add_f32_e32 v82, v82, v240
	v_add_f32_e32 v82, v82, v242
	v_add_f32_e32 v82, v82, v241
	v_add_f32_e32 v82, v82, v244
	v_add_f32_e32 v82, v82, v243
	v_add_f32_e32 v82, v82, v245
	v_mov_b32_e32 v83, v82
	s_nop 1
	v_permlane32_swap_b32_e32 v82, v83
	v_add_f32_e32 v203, v82, v83
	v_fmac_f32_e32 v203, v229, v190
	scratch_load_dword v229, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[134:137], v[66:81]
	v_mul_f32_e32 v34, v34, v190
	v_mul_f32_e32 v35, v35, v190
	v_mul_f32_e32 v36, v36, v190
	v_mul_f32_e32 v37, v37, v190
	v_mul_f32_e32 v38, v38, v190
	v_mul_f32_e32 v39, v39, v190
	v_mul_f32_e32 v40, v40, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[130:133], 0
	v_mul_f32_e32 v41, v41, v190
	v_mul_f32_e32 v42, v42, v190
	v_mul_f32_e32 v43, v43, v190
	v_mul_f32_e32 v44, v44, v190
	v_mul_f32_e32 v45, v45, v190
	v_mul_f32_e32 v46, v46, v190
	v_mul_f32_e32 v47, v47, v190
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[138:141], v[66:81]
	v_mul_f32_e32 v48, v48, v190
	v_mul_f32_e32 v49, v49, v190
	v_mul_f32_e32 v50, v50, v190
	v_mul_f32_e32 v51, v51, v190
	v_mul_f32_e32 v52, v52, v190
	v_mul_f32_e32 v53, v53, v190
	v_mul_f32_e32 v54, v54, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[134:137], v[82:97]
	v_mul_f32_e32 v55, v55, v190
	v_mul_f32_e32 v56, v56, v190
	v_mul_f32_e32 v57, v57, v190
	v_mul_f32_e32 v58, v58, v190
	v_mul_f32_e32 v59, v59, v190
	v_mul_f32_e32 v60, v60, v190
	v_mul_f32_e32 v61, v61, v190
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[142:145], v[66:81]
	v_mul_f32_e32 v62, v62, v190
	v_mul_f32_e32 v63, v63, v190
	v_mul_f32_e32 v64, v64, v190
	v_mul_f32_e32 v65, v65, v190
	v_mul_f32_e32 v18, v18, v190
	v_mul_f32_e32 v19, v19, v190
	v_mul_f32_e32 v20, v20, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[138:141], v[82:97]
	v_mul_f32_e32 v21, v21, v190
	v_mul_f32_e32 v22, v22, v190
	v_mul_f32_e32 v23, v23, v190
	v_mul_f32_e32 v24, v24, v190
	v_mul_f32_e32 v25, v25, v190
	v_mul_f32_e32 v26, v26, v190
	v_mul_f32_e32 v27, v27, v190
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[146:149], v[66:81]
	v_mul_f32_e32 v28, v28, v190
	v_mul_f32_e32 v29, v29, v190
	v_mul_f32_e32 v30, v30, v190
	v_mul_f32_e32 v31, v31, v190
	v_mul_f32_e32 v32, v32, v190
	v_mul_f32_e32 v33, v33, v190
	v_mul_f32_e32 v2, v2, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[142:145], v[82:97]
	v_mul_f32_e32 v3, v3, v190
	v_mul_f32_e32 v4, v4, v190
	v_mul_f32_e32 v5, v5, v190
	v_mul_f32_e32 v6, v6, v190
	v_mul_f32_e32 v7, v7, v190
	v_mul_f32_e32 v8, v8, v190
	v_mul_f32_e32 v9, v9, v190
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[150:153], v[66:81]
	v_mul_f32_e32 v10, v10, v190
	v_mul_f32_e32 v11, v11, v190
	v_mul_f32_e32 v12, v12, v190
	v_mul_f32_e32 v13, v13, v190
	v_mul_f32_e32 v14, v14, v190
	v_mul_f32_e32 v15, v15, v190
	v_mul_f32_e32 v16, v16, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[146:149], v[82:97]
	v_mul_f32_e32 v17, v17, v190
	v_cvt_pk_f16_f32 v106, v252, v255
	v_cvt_pk_f16_f32 v107, v254, v222
	v_cvt_pk_f16_f32 v108, v191, v193
	v_cvt_pk_f16_f32 v109, v192, v196
	v_cvt_pk_f16_f32 v98, v234, v238
	v_cvt_pk_f16_f32 v99, v240, v242
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[154:157], v[66:81]
	v_cvt_pk_f16_f32 v110, v1, v227
	v_cvt_pk_f16_f32 v111, v247, v249
	v_cvt_pk_f16_f32 v112, v248, v251
	v_cvt_pk_f16_f32 v113, v250, v253
	v_cvt_pk_f16_f32 v100, v241, v244
	v_cvt_pk_f16_f32 v101, v243, v245
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[150:153], v[82:97]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[158:161], v[66:81]
	v_cvt_pk_f16_f32 v102, v195, v232
	v_cvt_pk_f16_f32 v103, v197, v237
	v_cvt_pk_f16_f32 v104, v236, v239
	v_cvt_pk_f16_f32 v105, v233, v235
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[154:157], v[82:97]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[114:117], v[158:161], v[82:97]
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s46, s38, s34
	s_addc_u32 s47, s39, s35
	s_add_i32 s12, s53, 1
	s_cmp_lt_i32 s12, 2
	s_cselect_b32 s57, s12, 0
	ds_bpermute_b32 v115, v246, v202
	s_lshl_b32 s23, s57, 14
	s_waitcnt vmcnt(2)
	ds_bpermute_b32 v116, v246, v215
	s_add_i32 s22, s23, 0
	v_add_u32_e32 v1, s22, v224
	v_add_u32_e32 v114, s22, v225
	s_and_b32 s12, s49, 0xffff
	v_readfirstlane_b32 s21, v1
	s_or_b32 s13, s12, s48
	s_mov_b32 s12, s3
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v115, 1, v115
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v114
	buffer_load_dwordx4 v115, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v116
	s_mov_b32 m0, s21
	v_lshlrev_b32_e32 v114, 1, v208
	buffer_load_dwordx4 v1, s[12:15], 0 offen lds
	v_lshl_add_u32 v1, v226, 1, s59
	v_lshlrev_b32_e32 v115, 1, v207
	v_add3_u32 v1, v1, v114, v115
	v_lshlrev_b32_e32 v114, 1, v216
	v_add3_u32 v1, v1, v114, v198
	ds_read_b64_tr_b16 v[232:233], v1
	ds_read_b64_tr_b16 v[234:235], v1 offset:2048
	ds_read_b64_tr_b16 v[236:237], v1 offset:4096
	ds_read_b64_tr_b16 v[238:239], v1 offset:6144
	ds_read_b64_tr_b16 v[240:241], v1 offset:8192
	ds_read_b64_tr_b16 v[242:243], v1 offset:10240
	ds_read_b64_tr_b16 v[194:195], v1 offset:12288
	ds_read_b64_tr_b16 v[196:197], v1 offset:14336
	v_add3_u32 v1, s59, v204, v115
	v_add3_u32 v1, v1, v114, v198
	ds_read_b64_tr_b16 v[190:191], v1
	ds_read_b64_tr_b16 v[192:193], v1 offset:2048
	ds_read_b64_tr_b16 v[186:187], v1 offset:4096
	ds_read_b64_tr_b16 v[188:189], v1 offset:6144
	ds_read_b64_tr_b16 v[182:183], v1 offset:8192
	ds_read_b64_tr_b16 v[184:185], v1 offset:10240
	ds_read_b64_tr_b16 v[178:179], v1 offset:12288
	ds_read_b64_tr_b16 v[180:181], v1 offset:14336
	v_lshl_add_u32 v1, v212, 1, s59
	v_add3_u32 v1, v1, v114, v198
	ds_read_b64_tr_b16 v[174:175], v1
	ds_read_b64_tr_b16 v[176:177], v1 offset:2048
	ds_read_b64_tr_b16 v[170:171], v1 offset:4096
	ds_read_b64_tr_b16 v[172:173], v1 offset:6144
	ds_read_b64_tr_b16 v[166:167], v1 offset:8192
	ds_read_b64_tr_b16 v[168:169], v1 offset:10240
	ds_read_b64_tr_b16 v[162:163], v1 offset:12288
	ds_read_b64_tr_b16 v[164:165], v1 offset:14336
	v_lshl_add_u32 v1, v213, 1, s59
	v_add3_u32 v1, v1, v114, v198
	ds_read_b64_tr_b16 v[126:127], v1
	ds_read_b64_tr_b16 v[128:129], v1 offset:2048
	ds_read_b64_tr_b16 v[122:123], v1 offset:4096
	ds_read_b64_tr_b16 v[124:125], v1 offset:6144
	ds_read_b64_tr_b16 v[118:119], v1 offset:8192
	ds_read_b64_tr_b16 v[120:121], v1 offset:10240
	ds_read_b64_tr_b16 v[114:115], v1 offset:12288
	ds_read_b64_tr_b16 v[116:117], v1 offset:14336
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[34:49], v[232:235], v[110:113], v[34:49]
	s_barrier
	s_setprio 0
	v_max_f32_e32 v1, v67, v67
	v_max_f32_e32 v222, v66, v66
	v_max_f32_e32 v1, v222, v1
	v_max3_f32 v1, v1, v68, v69
	v_mfma_f32_32x32x16_f16 v[34:49], v[236:239], v[106:109], v[34:49]
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_max3_f32 v1, v1, v74, v75
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_mfma_f32_32x32x16_f16 v[50:65], v[190:193], v[110:113], v[50:65]
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_mfma_f32_32x32x16_f16 v[34:49], v[240:243], v[102:105], v[34:49]
	v_mfma_f32_32x32x16_f16 v[50:65], v[186:189], v[106:109], v[50:65]
	v_mov_b32_e32 v186, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v186
	v_mfma_f32_32x32x16_f16 v[34:49], v[194:197], v[98:101], v[34:49]
	v_max3_f32 v194, v223, v1, v186
	v_mul_f32_e32 v186, 0x3e0293ee, v194
	v_fma_f32 v1, v66, s50, -v186
	v_fma_f32 v66, v67, s50, -v186
	v_exp_f32_e32 v227, v66
	v_fma_f32 v66, v223, s50, -v186
	scratch_load_dword v223, off, off       ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[110:113], v[18:33]
	v_fma_f32 v67, v68, s50, -v186
	v_fma_f32 v68, v69, s50, -v186
	v_fma_f32 v69, v70, s50, -v186
	v_fma_f32 v70, v71, s50, -v186
	v_fma_f32 v71, v72, s50, -v186
	v_fma_f32 v72, v73, s50, -v186
	v_fma_f32 v73, v74, s50, -v186
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[126:129], v[110:113], v[2:17]
	v_fma_f32 v74, v75, s50, -v186
	v_fma_f32 v75, v76, s50, -v186
	v_fma_f32 v76, v77, s50, -v186
	v_fma_f32 v77, v78, s50, -v186
	v_fma_f32 v78, v79, s50, -v186
	v_fma_f32 v79, v80, s50, -v186
	v_fma_f32 v80, v81, s50, -v186
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[106:109], v[18:33]
	v_fma_f32 v81, v82, s50, -v186
	v_fma_f32 v82, v83, s50, -v186
	v_fma_f32 v83, v84, s50, -v186
	v_fma_f32 v84, v85, s50, -v186
	v_fma_f32 v85, v86, s50, -v186
	v_fma_f32 v86, v87, s50, -v186
	v_fma_f32 v87, v88, s50, -v186
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[122:125], v[106:109], v[2:17]
	v_fma_f32 v88, v89, s50, -v186
	v_fma_f32 v89, v90, s50, -v186
	v_fma_f32 v90, v91, s50, -v186
	v_fma_f32 v91, v92, s50, -v186
	v_fma_f32 v92, v93, s50, -v186
	v_fma_f32 v93, v94, s50, -v186
	v_fma_f32 v94, v95, s50, -v186
	v_mfma_f32_32x32x16_f16 v[50:65], v[182:185], v[102:105], v[50:65]
	v_fma_f32 v95, v96, s50, -v186
	v_fma_f32 v96, v97, s50, -v186
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v247, v67
	v_exp_f32_e32 v249, v68
	v_exp_f32_e32 v248, v69
	v_exp_f32_e32 v251, v70
	v_mfma_f32_32x32x16_f16 v[18:33], v[166:169], v[102:105], v[18:33]
	v_exp_f32_e32 v250, v71
	v_exp_f32_e32 v253, v72
	v_exp_f32_e32 v252, v73
	v_exp_f32_e32 v255, v74
	v_exp_f32_e32 v254, v75
	v_exp_f32_e32 v222, v76
	v_exp_f32_e32 v191, v77
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[118:121], v[102:105], v[2:17]
	v_exp_f32_e32 v193, v78
	v_exp_f32_e32 v192, v79
	v_exp_f32_e32 v196, v80
	v_exp_f32_e32 v195, v81
	v_exp_f32_e32 v232, v82
	v_exp_f32_e32 v197, v83
	v_exp_f32_e32 v237, v84
	v_mfma_f32_32x32x16_f16 v[50:65], v[178:181], v[98:101], v[50:65]
	v_exp_f32_e32 v236, v85
	v_exp_f32_e32 v239, v86
	v_exp_f32_e32 v233, v87
	v_exp_f32_e32 v235, v88
	v_exp_f32_e32 v234, v89
	v_exp_f32_e32 v238, v90
	v_exp_f32_e32 v240, v91
	v_mfma_f32_32x32x16_f16 v[18:33], v[162:165], v[98:101], v[18:33]
	v_exp_f32_e32 v242, v92
	v_exp_f32_e32 v241, v93
	v_exp_f32_e32 v244, v94
	v_exp_f32_e32 v243, v95
	v_exp_f32_e32 v245, v96
	v_exp_f32_e32 v190, v66
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[114:117], v[98:101], v[2:17]
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s12, s53, 14
	s_add_i32 s12, s12, 0
	s_add_i32 s21, s12, 0x8000
	v_lshlrev_b32_e32 v66, 1, v211
	v_lshlrev_b32_e32 v67, 1, v210
	v_add3_u32 v66, s12, v66, v67
	v_lshl_add_u32 v67, v231, 1, s21
	v_lshl_add_u32 v68, v205, 1, s21
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
	v_add_lshl_u32 v69, v69, v230, 2
	v_ashrrev_i32_e32 v66, 4, v66
	ds_bpermute_b32 v69, v69, v200
	v_add_lshl_u32 v66, v66, v230, 2
	ds_bpermute_b32 v66, v66, v209
	s_and_b32 s12, s47, 0xffff
	v_readfirstlane_b32 s53, v67
	s_or_b32 s13, s12, s42
	s_mov_b32 s12, s46
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v69, 1, v69
	s_mov_b32 m0, s53
	v_readfirstlane_b32 s53, v68
	buffer_load_dwordx4 v69, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v66, s[12:15], 0 offen lds
	v_lshlrev_b32_e32 v66, 1, v228
	v_add3_u32 v70, s55, v218, v66
	v_add3_u32 v71, s55, v219, v66
	v_add3_u32 v72, s55, v220, v66
	v_add3_u32 v73, s55, v221, v66
	s_waitcnt vmcnt(2)
	v_add3_u32 v74, s55, v223, v66
	v_add3_u32 v75, s55, v229, v66
	v_add3_u32 v76, s55, v214, v66
	v_add3_u32 v77, s55, v217, v66
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[98:101], v70 offset:8192
	ds_read_b128 v[186:189], v71
	ds_read_b128 v[170:173], v71 offset:8192
	ds_read_b128 v[182:185], v72
	ds_read_b128 v[166:169], v72 offset:8192
	ds_read_b128 v[178:181], v73
	ds_read_b128 v[162:165], v73 offset:8192
	ds_read_b128 v[174:177], v74
	ds_read_b128 v[126:129], v74 offset:8192
	ds_read_b128 v[106:109], v75
	ds_read_b128 v[122:125], v75 offset:8192
	ds_read_b128 v[110:113], v76
	ds_read_b128 v[118:121], v76 offset:8192
	ds_read_b128 v[102:105], v77
	ds_read_b128 v[114:117], v77 offset:8192
	; sched_barrier mask(0x00000000)
	s_add_i32 s51, s51, 1
	s_add_u32 s3, s3, s44
	s_addc_u32 s49, s49, s45
	s_cmp_lt_i32 s51, s2
	s_mov_b32 s53, s57
	s_barrier
	s_cbranch_scc1 .LBB0_9
; %bb.10:                               ; %Flow1108
	scratch_load_dword v205, off, off offset:36 ; 4-byte Folded Reload
	v_mov_b32_e32 v198, v214
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execnz .LBB0_13
	s_branch .LBB0_14
.LBB0_11:
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v162, 0xff800000
	v_mov_b32_e32 v98, 1.0
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v11, v3
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v13, v3
	v_mov_b32_e32 v12, v3
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v14, v3
	v_mov_b32_e32 v17, v3
	v_mov_b32_e32 v16, v3
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v21, v3
	v_mov_b32_e32 v20, v3
	v_mov_b32_e32 v23, v3
	v_mov_b32_e32 v22, v3
	v_mov_b32_e32 v25, v3
	v_mov_b32_e32 v24, v3
	v_mov_b32_e32 v27, v3
	v_mov_b32_e32 v26, v3
	v_mov_b32_e32 v29, v3
	v_mov_b32_e32 v28, v3
	v_mov_b32_e32 v31, v3
	v_mov_b32_e32 v30, v3
	v_mov_b32_e32 v33, v3
	v_mov_b32_e32 v32, v3
	v_mov_b32_e32 v35, v3
	v_mov_b32_e32 v34, v3
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v36, v3
	v_mov_b32_e32 v39, v3
	v_mov_b32_e32 v38, v3
	v_mov_b32_e32 v41, v3
	v_mov_b32_e32 v40, v3
	v_mov_b32_e32 v43, v3
	v_mov_b32_e32 v42, v3
	v_mov_b32_e32 v45, v3
	v_mov_b32_e32 v44, v3
	v_mov_b32_e32 v47, v3
	v_mov_b32_e32 v46, v3
	v_mov_b32_e32 v49, v3
	v_mov_b32_e32 v48, v3
	v_mov_b32_e32 v51, v3
	v_mov_b32_e32 v50, v3
	v_mov_b32_e32 v53, v3
	v_mov_b32_e32 v52, v3
	v_mov_b32_e32 v55, v3
	v_mov_b32_e32 v54, v3
	v_mov_b32_e32 v57, v3
	v_mov_b32_e32 v56, v3
	v_mov_b32_e32 v59, v3
	v_mov_b32_e32 v58, v3
	v_mov_b32_e32 v61, v3
	v_mov_b32_e32 v60, v3
	v_mov_b32_e32 v63, v3
	v_mov_b32_e32 v62, v3
	v_mov_b32_e32 v65, v3
	v_mov_b32_e32 v64, v3
	s_branch .LBB0_15
.LBB0_12:
	s_mov_b32 s22, 0
	s_add_i32 s20, 0, 0x8000
	s_add_i32 s21, 0, 0xc000
	v_mov_b32_e32 v203, 1.0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	s_mov_b32 s23, 0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_14
.LBB0_13:
	s_barrier
.LBB0_14:
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[130:133], 0
	v_mul_f32_e32 v68, v36, v190
	v_mul_f32_e32 v36, v52, v190
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	scratch_load_dword v52, off, off offset:8 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[134:137], v[82:97]
	v_add_f32_e32 v66, v1, v227
	v_add_f32_e32 v66, v66, v247
	v_add_f32_e32 v66, v66, v249
	v_add_f32_e32 v66, v66, v248
	v_add_f32_e32 v66, v66, v251
	v_add_f32_e32 v66, v66, v250
	v_add_f32_e32 v66, v66, v253
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[138:141], v[82:97]
	v_add_f32_e32 v66, v66, v252
	v_add_f32_e32 v66, v66, v255
	v_add_f32_e32 v66, v66, v254
	v_add_f32_e32 v66, v66, v222
	v_add_f32_e32 v66, v66, v191
	v_add_f32_e32 v66, v66, v193
	v_add_f32_e32 v66, v66, v192
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[142:145], v[82:97]
	v_add_f32_e32 v66, v66, v196
	v_add_f32_e32 v66, v66, v195
	v_add_f32_e32 v66, v66, v232
	v_add_f32_e32 v66, v66, v197
	v_add_f32_e32 v66, v66, v237
	v_add_f32_e32 v66, v66, v236
	v_add_f32_e32 v66, v66, v239
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[146:149], v[82:97]
	v_add_f32_e32 v66, v66, v233
	v_add_f32_e32 v66, v66, v235
	v_add_f32_e32 v66, v66, v234
	v_add_f32_e32 v66, v66, v238
	v_add_f32_e32 v66, v66, v240
	v_add_f32_e32 v66, v66, v242
	v_add_f32_e32 v66, v66, v241
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[150:153], v[82:97]
	v_add_f32_e32 v66, v66, v244
	v_add_f32_e32 v66, v66, v243
	v_add_f32_e32 v73, v66, v245
	v_mul_f32_e32 v66, v34, v190
	v_mul_f32_e32 v69, v37, v190
	v_mul_f32_e32 v34, v50, v190
	v_mul_f32_e32 v37, v53, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[154:157], v[82:97]
	v_cvt_pk_f16_f32 v50, v1, v227
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshrrev_b32_e32 v53, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_mul_f32_e32 v67, v35, v190
	v_mul_f32_e32 v35, v51, v190
	v_or_b32_e32 v51, v207, v216
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[158:161], v[82:97]
	v_mov_b32_e32 v74, v73
	s_nop 1
	v_permlane32_swap_b32_e32 v73, v74
	v_mul_f32_e32 v80, v48, v190
	v_mul_f32_e32 v48, v64, v190
	v_mul_f32_e32 v72, v40, v190
	v_add_f32_e32 v174, v73, v74
	v_mfma_f32_32x32x16_f16 v[98:113], v[98:101], v[130:133], 0
	v_mul_f32_e32 v73, v41, v190
	v_mul_f32_e32 v74, v42, v190
	v_mul_f32_e32 v75, v43, v190
	v_mul_f32_e32 v40, v56, v190
	v_mul_f32_e32 v41, v57, v190
	v_mul_f32_e32 v42, v58, v190
	v_mul_f32_e32 v43, v59, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[134:137], v[98:113]
	v_mul_f32_e32 v70, v38, v190
	v_mul_f32_e32 v71, v39, v190
	v_mul_f32_e32 v76, v44, v190
	v_mul_f32_e32 v77, v45, v190
	v_mul_f32_e32 v78, v46, v190
	v_mul_f32_e32 v79, v47, v190
	v_mul_f32_e32 v81, v49, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[138:141], v[98:113]
	v_mul_f32_e32 v44, v60, v190
	v_mul_f32_e32 v45, v61, v190
	v_mul_f32_e32 v46, v62, v190
	v_mul_f32_e32 v47, v63, v190
	v_mul_f32_e32 v38, v54, v190
	v_mul_f32_e32 v39, v55, v190
	s_waitcnt vmcnt(0)
	v_and_or_b32 v52, v53, 3, v52
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[142:145], v[98:113]
	v_lshlrev_b32_e32 v162, 7, v52
	v_cvt_pk_f16_f32 v52, v248, v251
	v_cvt_pk_f16_f32 v53, v250, v253
	v_cvt_pk_f16_f32 v54, v252, v255
	v_cvt_pk_f16_f32 v55, v254, v222
	v_mul_f32_e32 v49, v65, v190
	v_fmac_f32_e32 v174, v203, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[126:129], v[146:149], v[98:113]
	v_mul_f32_e32 v18, v18, v190
	v_mul_f32_e32 v19, v19, v190
	v_mul_f32_e32 v20, v20, v190
	v_mul_f32_e32 v21, v21, v190
	v_mul_f32_e32 v22, v22, v190
	v_mul_f32_e32 v23, v23, v190
	v_mul_f32_e32 v24, v24, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[122:125], v[150:153], v[98:113]
	v_or_b32_e32 v122, v1, v208
	v_or3_b32 v51, v51, v122, v162
	v_lshlrev_b32_e32 v51, 1, v51
	v_mov_b32_e32 v175, v51
	v_add_u32_e32 v64, s20, v51
	scratch_store_dword off, v175, off offset:72 ; 4-byte Folded Spill
	ds_read_b64_tr_b16 v[56:57], v64
	ds_read_b64_tr_b16 v[58:59], v64 offset:2048
	v_cvt_pk_f16_f32 v51, v247, v249
	ds_read_b64_tr_b16 v[60:61], v64 offset:4096
	ds_read_b64_tr_b16 v[62:63], v64 offset:6144
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[66:81], v[56:59], v[50:53], v[66:81]
	v_cvt_pk_f16_f32 v56, v191, v193
	v_cvt_pk_f16_f32 v57, v192, v196
	v_cvt_pk_f16_f32 v58, v234, v238
	v_cvt_pk_f16_f32 v59, v240, v242
	v_mul_f32_e32 v25, v25, v190
	v_mul_f32_e32 v26, v26, v190
	v_mul_f32_e32 v27, v27, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[118:121], v[154:157], v[98:113]
	ds_read_b64_tr_b16 v[118:119], v64 offset:8192
	ds_read_b64_tr_b16 v[120:121], v64 offset:10240
	v_mul_f32_e32 v28, v28, v190
	v_mul_f32_e32 v29, v29, v190
	v_mul_f32_e32 v30, v30, v190
	v_mul_f32_e32 v31, v31, v190
	v_mul_f32_e32 v32, v32, v190
	v_mul_f32_e32 v33, v33, v190
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[66:81], v[60:63], v[54:57], v[66:81]
	ds_read_b64_tr_b16 v[62:63], v64 offset:12288
	ds_read_b64_tr_b16 v[64:65], v64 offset:14336
	v_cvt_pk_f16_f32 v60, v241, v244
	v_cvt_pk_f16_f32 v61, v243, v245
	s_movk_i32 s12, 0x60
	v_max_f32_e32 v170, v83, v83
	v_mul_f32_e32 v2, v2, v190
	v_mul_f32_e32 v3, v3, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[114:117], v[158:161], v[98:113]
	v_cvt_pk_f16_f32 v114, v195, v232
	v_cvt_pk_f16_f32 v115, v197, v237
	v_cvt_pk_f16_f32 v116, v236, v239
	v_cvt_pk_f16_f32 v117, v233, v235
	v_mul_f32_e32 v4, v4, v190
	v_mul_f32_e32 v5, v5, v190
	v_mul_f32_e32 v6, v6, v190
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[66:81], v[118:121], v[114:117], v[66:81]
	v_or_b32_e32 v118, 32, v1
	v_bitop3_b32 v118, v216, v118, v208 bitop3:0xf6
	v_or3_b32 v118, v118, v207, v162
	v_lshlrev_b32_e32 v118, 1, v118
	v_mov_b32_e32 v196, v118
	v_add_u32_e32 v123, s20, v118
	scratch_store_dword off, v196, off offset:64 ; 4-byte Folded Spill
	ds_read_b64_tr_b16 v[118:119], v123
	ds_read_b64_tr_b16 v[120:121], v123 offset:2048
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[118:121], v[50:53], v[34:49]
	v_bitop3_b32 v1, v206, v1, s12 bitop3:0x4e
	v_or3_b32 v1, v1, v216, v162
	v_lshlrev_b32_e32 v1, 1, v1
	v_add_u32_e32 v168, s20, v1
	v_mul_f32_e32 v7, v7, v190
	v_mul_f32_e32 v8, v8, v190
	v_mul_f32_e32 v9, v9, v190
	v_mfma_f32_32x32x16_f16 v[66:81], v[62:65], v[58:61], v[66:81]
	ds_read_b64_tr_b16 v[62:63], v123 offset:4096
	ds_read_b64_tr_b16 v[64:65], v123 offset:6144
	ds_read_b64_tr_b16 v[118:119], v123 offset:8192
	ds_read_b64_tr_b16 v[120:121], v123 offset:10240
	v_mul_f32_e32 v10, v10, v190
	v_mul_f32_e32 v11, v11, v190
	v_mul_f32_e32 v12, v12, v190
	v_mul_f32_e32 v13, v13, v190
	v_mul_f32_e32 v14, v14, v190
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[62:65], v[54:57], v[34:49]
	ds_read_b64_tr_b16 v[62:63], v123 offset:12288
	ds_read_b64_tr_b16 v[64:65], v123 offset:14336
	scratch_store_dword off, v174, off offset:60 ; 4-byte Folded Spill
	v_mul_f32_e32 v15, v15, v190
	v_mul_f32_e32 v16, v16, v190
	v_mul_f32_e32 v17, v17, v190
	s_add_u32 s2, s38, s34
	s_addc_u32 s3, s39, s35
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[118:121], v[114:117], v[34:49]
	v_bitop3_b32 v118, v122, v207, 64 bitop3:0x36
	v_or3_b32 v118, v118, v216, v162
	v_lshlrev_b32_e32 v118, 1, v118
	v_mov_b32_e32 v195, v118
	v_add_u32_e32 v128, s20, v118
	scratch_store_dword off, v195, off offset:68 ; 4-byte Folded Spill
	s_add_u32 s12, s2, s34
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[62:65], v[58:61], v[34:49]
	ds_read_b64_tr_b16 v[62:63], v128
	ds_read_b64_tr_b16 v[64:65], v128 offset:2048
	ds_read_b64_tr_b16 v[118:119], v128 offset:4096
	ds_read_b64_tr_b16 v[120:121], v128 offset:6144
	ds_read_b64_tr_b16 v[122:123], v128 offset:8192
	ds_read_b64_tr_b16 v[124:125], v128 offset:10240
	ds_read_b64_tr_b16 v[126:127], v128 offset:12288
	ds_read_b64_tr_b16 v[128:129], v128 offset:14336
	scratch_store_dword off, v1, off offset:76 ; 4-byte Folded Spill
	s_addc_u32 s13, s3, s35
	s_add_i32 s2, s23, 0
	s_add_i32 s3, s2, 0x8000
	s_and_b32 s13, s13, 0xffff
	s_or_b32 s13, s13, s42
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[62:65], v[50:53], v[18:33]
	ds_read_b64_tr_b16 v[62:63], v168
	ds_read_b64_tr_b16 v[64:65], v168 offset:2048
	ds_read_b64_tr_b16 v[162:163], v168 offset:4096
	ds_read_b64_tr_b16 v[164:165], v168 offset:6144
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_add_u32_e32 v203, s21, v196
	v_mov_b32_e32 v197, v198
	v_mov_b32_e32 v211, v216
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[18:33], v[118:121], v[54:57], v[18:33]
	ds_read_b64_tr_b16 v[118:119], v168 offset:8192
	ds_read_b64_tr_b16 v[120:121], v168 offset:10240
	ds_read_b64_tr_b16 v[166:167], v168 offset:12288
	ds_read_b64_tr_b16 v[168:169], v168 offset:14336
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[18:33], v[122:125], v[114:117], v[18:33]
	v_max_f32_e32 v122, v82, v82
	v_max_f32_e32 v122, v122, v170
	v_max3_f32 v122, v122, v84, v85
	v_max3_f32 v122, v122, v86, v87
	v_max3_f32 v122, v122, v88, v89
	v_max3_f32 v122, v122, v90, v91
	v_max3_f32 v122, v122, v92, v93
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[62:65], v[50:53], v[2:17]
	v_max3_f32 v122, v122, v94, v95
	v_max3_f32 v122, v122, v96, v97
	v_max3_f32 v122, v122, v98, v99
	v_max3_f32 v122, v122, v100, v101
	v_max3_f32 v122, v122, v102, v103
	v_max3_f32 v122, v122, v104, v105
	v_max3_f32 v122, v122, v106, v107
	v_max3_f32 v50, v122, v108, v109
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[162:165], v[54:57], v[2:17]
	v_max3_f32 v50, v50, v110, v111
	v_max3_f32 v50, v50, v112, v113
	v_mov_b32_e32 v51, v50
	s_nop 1
	v_permlane32_swap_b32_e32 v50, v51
	v_max3_f32 v1, v194, v50, v51
	v_lshlrev_b32_e32 v50, 1, v228
	v_add3_u32 v54, s22, v218, v50
	scratch_store_dword off, v1, off offset:80 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_add3_u32 v55, s22, v219, v50
	v_add3_u32 v56, s22, v220, v50
	v_add3_u32 v57, s22, v221, v50
	v_add3_u32 v170, s22, v223, v50
	v_add3_u32 v171, s22, v229, v50
	v_add3_u32 v172, s22, v198, v50
	v_add3_u32 v173, s22, v217, v50
	ds_read_b128 v[50:53], v54
	v_mfma_f32_32x32x16_f16 v[2:17], v[118:121], v[114:117], v[2:17]
	v_mul_f32_e32 v204, 0x3e0293ee, v1
	v_mov_b32_e32 v198, v209
	v_mfma_f32_32x32x16_f16 v[18:33], v[126:129], v[58:61], v[18:33]
	ds_read_b128 v[114:117], v54 offset:8192
	ds_read_b128 v[118:121], v55
	ds_read_b128 v[190:193], v55 offset:8192
	ds_read_b128 v[122:125], v56
	ds_read_b128 v[182:185], v56 offset:8192
	ds_read_b128 v[126:129], v57
	ds_read_b128 v[178:181], v57 offset:8192
	v_mfma_f32_32x32x16_f16 v[2:17], v[166:169], v[58:61], v[2:17]
	ds_read_b128 v[162:165], v170
	ds_read_b128 v[226:229], v170 offset:8192
	ds_read_b128 v[166:169], v171
	ds_read_b128 v[230:233], v171 offset:8192
	ds_read_b128 v[234:237], v172
	ds_read_b128 v[238:241], v172 offset:8192
	ds_read_b128 v[242:245], v173
	ds_read_b128 v[186:189], v173 offset:8192
	ds_bpermute_b32 v170, v199, v200
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[50:53], v[130:133], 0
	v_mfma_f32_32x32x16_f16 v[50:65], v[118:121], v[134:137], v[50:65]
	v_lshrrev_b64 v[118:119], v205, exec
	v_add_u32_e32 v120, s3, v224
	v_and_b32_e32 v118, 1, v118
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v119, 1, v170
	v_cmp_eq_u32_e32 vcc, 1, v118
	v_readfirstlane_b32 s20, v120
	v_add_u32_e32 v121, s3, v225
	v_mfma_f32_32x32x16_f16 v[50:65], v[122:125], v[138:141], v[50:65]
	ds_bpermute_b32 v123, v199, v209
	v_bfrev_b32_e32 v122, 1
	v_cndmask_b32_e32 v118, v122, v119, vcc
	s_mov_b32 m0, s20
	v_readfirstlane_b32 s20, v121
	buffer_load_dwordx4 v118, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v118, 1, v123
	v_mfma_f32_32x32x16_f16 v[50:65], v[126:129], v[142:145], v[50:65]
	v_cndmask_b32_e32 v118, v122, v118, vcc
	s_mov_b32 m0, s20
	v_mov_b32_e32 v199, v217
	buffer_load_dwordx4 v118, s[12:15], 0 offen lds
	v_add_u32_e32 v118, s21, v175
	s_waitcnt vmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[162:165], v[146:149], v[50:65]
	s_barrier
	s_mov_b32 s12, 0x3e0293ee
	v_fma_f32 v82, v82, s12, -v204
	v_exp_f32_e32 v252, v82
	v_fma_f32 v82, v194, s12, -v204
	v_fma_f32 v98, v98, s12, -v204
	v_mfma_f32_32x32x16_f16 v[50:65], v[166:169], v[150:153], v[50:65]
	ds_read_b64_tr_b16 v[174:175], v118
	ds_read_b64_tr_b16 v[176:177], v118 offset:2048
	ds_read_b64_tr_b16 v[170:171], v118 offset:4096
	ds_read_b64_tr_b16 v[172:173], v118 offset:6144
	ds_read_b64_tr_b16 v[166:167], v118 offset:8192
	ds_read_b64_tr_b16 v[168:169], v118 offset:10240
	ds_read_b64_tr_b16 v[162:163], v118 offset:12288
	ds_read_b64_tr_b16 v[164:165], v118 offset:14336
	v_fma_f32 v99, v99, s12, -v204
	v_fma_f32 v100, v100, s12, -v204
	v_fma_f32 v101, v101, s12, -v204
	v_fma_f32 v102, v102, s12, -v204
	v_fma_f32 v103, v103, s12, -v204
	v_fma_f32 v104, v104, s12, -v204
	v_mfma_f32_32x32x16_f16 v[114:129], v[114:117], v[130:133], 0
	v_fma_f32 v205, v105, s12, -v204
	v_exp_f32_e32 v210, v98
	v_exp_f32_e32 v206, v99
	v_fma_f32 v83, v83, s12, -v204
	v_fma_f32 v84, v84, s12, -v204
	v_fma_f32 v85, v85, s12, -v204
	v_fma_f32 v86, v86, s12, -v204
	v_mfma_f32_32x32x16_f16 v[114:129], v[190:193], v[134:137], v[114:129]
	v_fma_f32 v87, v87, s12, -v204
	v_fma_f32 v88, v88, s12, -v204
	v_fma_f32 v89, v89, s12, -v204
	v_exp_f32_e32 v253, v83
	v_exp_f32_e32 v254, v84
	v_exp_f32_e32 v255, v85
	v_exp_f32_e32 v1, v86
	v_mfma_f32_32x32x16_f16 v[114:129], v[182:185], v[138:141], v[114:129]
	v_exp_f32_e32 v209, v88
	v_exp_f32_e32 v214, v89
	v_fma_f32 v90, v90, s12, -v204
	v_fma_f32 v91, v91, s12, -v204
	v_fma_f32 v92, v92, s12, -v204
	v_fma_f32 v93, v93, s12, -v204
	v_fma_f32 v94, v94, s12, -v204
	v_mfma_f32_32x32x16_f16 v[114:129], v[178:181], v[142:145], v[114:129]
	v_fma_f32 v95, v95, s12, -v204
	v_fma_f32 v96, v96, s12, -v204
	v_fma_f32 v97, v97, s12, -v204
	v_exp_f32_e32 v215, v90
	v_exp_f32_e32 v216, v91
	v_exp_f32_e32 v217, v92
	v_exp_f32_e32 v196, v94
	v_mfma_f32_32x32x16_f16 v[114:129], v[226:229], v[146:149], v[114:129]
	v_cvt_pk_f16_f32 v226, v252, v253
	v_cvt_pk_f16_f32 v227, v254, v255
	v_cvt_pk_f16_f32 v229, v209, v214
	v_fma_f32 v212, v106, s12, -v204
	v_fma_f32 v213, v107, s12, -v204
	v_fma_f32 v246, v108, s12, -v204
	v_fma_f32 v247, v109, s12, -v204
	v_mfma_f32_32x32x16_f16 v[114:129], v[230:233], v[150:153], v[114:129]
	v_fma_f32 v248, v110, s12, -v204
	v_fma_f32 v249, v111, s12, -v204
	v_fma_f32 v250, v112, s12, -v204
	v_fma_f32 v251, v113, s12, -v204
	v_exp_f32_e32 v194, v205
	v_exp_f32_e32 v204, v212
	v_exp_f32_e32 v205, v213
	v_mfma_f32_32x32x16_f16 v[114:129], v[238:241], v[154:157], v[114:129]
	v_exp_f32_e32 v212, v246
	v_exp_f32_e32 v213, v247
	v_exp_f32_e32 v246, v248
	v_exp_f32_e32 v247, v249
	v_exp_f32_e32 v248, v250
	v_exp_f32_e32 v249, v251
	v_mfma_f32_32x32x16_f16 v[50:65], v[234:237], v[154:157], v[50:65]
	ds_read_b64_tr_b16 v[222:223], v203
	ds_read_b64_tr_b16 v[224:225], v203 offset:2048
	ds_read_b64_tr_b16 v[234:235], v203 offset:4096
	ds_read_b64_tr_b16 v[236:237], v203 offset:6144
	v_mfma_f32_32x32x16_f16 v[114:129], v[186:189], v[158:161], v[114:129]
	v_exp_f32_e32 v186, v82
	v_exp_f32_e32 v187, v102
	v_exp_f32_e32 v188, v103
	v_exp_f32_e32 v189, v104
	v_mul_f32_e32 v98, v66, v186
	v_mul_f32_e32 v99, v67, v186
	v_mul_f32_e32 v102, v70, v186
	v_mfma_f32_32x32x16_f16 v[50:65], v[242:245], v[158:161], v[50:65]
	ds_read_b64_tr_b16 v[242:243], v203 offset:8192
	ds_read_b64_tr_b16 v[244:245], v203 offset:10240
	ds_read_b64_tr_b16 v[190:191], v203 offset:12288
	ds_read_b64_tr_b16 v[192:193], v203 offset:14336
	v_add_u32_e32 v203, s21, v195
	ds_read_b64_tr_b16 v[182:183], v203
	ds_read_b64_tr_b16 v[184:185], v203 offset:2048
	ds_read_b64_tr_b16 v[178:179], v203 offset:4096
	ds_read_b64_tr_b16 v[180:181], v203 offset:6144
	scratch_store_dword off, v218, off offset:44 ; 4-byte Folded Spill
	scratch_store_dword off, v219, off offset:48 ; 4-byte Folded Spill
	scratch_store_dword off, v220, off offset:52 ; 4-byte Folded Spill
	scratch_store_dword off, v221, off offset:56 ; 4-byte Folded Spill
	scratch_store_dword off, v207, off offset:40 ; 4-byte Folded Spill
	scratch_store_dword off, v208, off offset:36 ; 4-byte Folded Spill
	v_exp_f32_e32 v207, v100
	v_exp_f32_e32 v208, v101
	v_mul_f32_e32 v100, v68, v186
	v_mul_f32_e32 v101, v69, v186
	v_mul_f32_e32 v103, v71, v186
	v_mul_f32_e32 v104, v72, v186
	v_mul_f32_e32 v105, v73, v186
	ds_read_b64_tr_b16 v[66:67], v203 offset:8192
	ds_read_b64_tr_b16 v[68:69], v203 offset:10240
	ds_read_b64_tr_b16 v[70:71], v203 offset:12288
	ds_read_b64_tr_b16 v[72:73], v203 offset:14336
	scratch_load_dword v203, off, off offset:76 ; 4-byte Folded Reload
	v_exp_f32_e32 v195, v87
	v_exp_f32_e32 v218, v93
	v_exp_f32_e32 v219, v95
	v_exp_f32_e32 v220, v96
	v_exp_f32_e32 v221, v97
	v_cvt_pk_f16_f32 v228, v1, v195
	v_mul_f32_e32 v82, v34, v186
	v_mul_f32_e32 v83, v35, v186
	v_mul_f32_e32 v84, v36, v186
	v_mul_f32_e32 v85, v37, v186
	v_mul_f32_e32 v86, v38, v186
	v_mul_f32_e32 v87, v39, v186
	v_mul_f32_e32 v88, v40, v186
	v_mul_f32_e32 v89, v41, v186
	v_mul_f32_e32 v90, v42, v186
	v_mul_f32_e32 v91, v43, v186
	v_mul_f32_e32 v92, v44, v186
	v_mul_f32_e32 v93, v45, v186
	v_mul_f32_e32 v94, v46, v186
	v_mul_f32_e32 v95, v47, v186
	v_mul_f32_e32 v96, v48, v186
	v_mul_f32_e32 v97, v49, v186
	v_mul_f32_e32 v106, v74, v186
	v_mul_f32_e32 v107, v75, v186
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[82:97], v[222:225], v[226:229], v[82:97]
	v_mul_f32_e32 v108, v76, v186
	v_mul_f32_e32 v109, v77, v186
	v_mul_f32_e32 v110, v78, v186
	v_mul_f32_e32 v111, v79, v186
	v_mul_f32_e32 v112, v80, v186
	v_mul_f32_e32 v113, v81, v186
	v_max_f32_e32 v35, v50, v50
	v_mul_f32_e32 v36, v20, v186
	v_mfma_f32_32x32x16_f16 v[98:113], v[174:177], v[226:229], v[98:113]
	v_cvt_pk_f16_f32 v174, v215, v216
	v_cvt_pk_f16_f32 v175, v217, v218
	v_cvt_pk_f16_f32 v176, v196, v219
	v_cvt_pk_f16_f32 v177, v220, v221
	v_mul_f32_e32 v37, v21, v186
	v_mul_f32_e32 v38, v22, v186
	v_mul_f32_e32 v39, v23, v186
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[82:97], v[234:237], v[174:177], v[82:97]
	v_mul_f32_e32 v40, v24, v186
	v_mul_f32_e32 v41, v25, v186
	v_mul_f32_e32 v42, v26, v186
	v_mul_f32_e32 v43, v27, v186
	v_mul_f32_e32 v44, v28, v186
	v_mul_f32_e32 v45, v29, v186
	v_mul_f32_e32 v46, v30, v186
	v_mul_f32_e32 v47, v31, v186
	v_mul_f32_e32 v48, v32, v186
	v_mul_f32_e32 v49, v33, v186
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[174:177], v[98:113]
	v_cvt_pk_f16_f32 v170, v210, v206
	v_cvt_pk_f16_f32 v171, v207, v208
	v_cvt_pk_f16_f32 v172, v187, v188
	v_cvt_pk_f16_f32 v173, v189, v194
	v_mul_f32_e32 v75, v11, v186
	v_mul_f32_e32 v76, v12, v186
	v_mul_f32_e32 v77, v13, v186
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[170:173], v[98:113]
	v_cvt_pk_f16_f32 v166, v204, v205
	v_cvt_pk_f16_f32 v167, v212, v213
	v_cvt_pk_f16_f32 v168, v246, v247
	v_cvt_pk_f16_f32 v169, v248, v249
	v_mul_f32_e32 v78, v14, v186
	v_mul_f32_e32 v79, v15, v186
	v_mul_f32_e32 v80, v16, v186
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[166:169], v[98:113]
	v_mul_f32_e32 v81, v17, v186
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v34, s21, v203
	ds_read_b64_tr_b16 v[222:223], v34
	ds_read_b64_tr_b16 v[224:225], v34 offset:2048
	ds_read_b64_tr_b16 v[230:231], v34 offset:4096
	ds_read_b64_tr_b16 v[232:233], v34 offset:6144
	ds_read_b64_tr_b16 v[234:235], v34 offset:8192
	ds_read_b64_tr_b16 v[236:237], v34 offset:10240
	ds_read_b64_tr_b16 v[238:239], v34 offset:12288
	ds_read_b64_tr_b16 v[240:241], v34 offset:14336
	v_max_f32_e32 v34, v51, v51
	v_max_f32_e32 v34, v35, v34
	v_max3_f32 v34, v34, v52, v53
	v_max3_f32 v34, v34, v54, v55
	v_max3_f32 v34, v34, v56, v57
	v_max3_f32 v34, v34, v58, v59
	v_max3_f32 v34, v34, v60, v61
	v_max3_f32 v34, v34, v62, v63
	v_max3_f32 v34, v34, v64, v65
	v_max3_f32 v34, v34, v114, v115
	v_max3_f32 v74, v34, v116, v117
	v_mul_f32_e32 v34, v18, v186
	v_mul_f32_e32 v35, v19, v186
	v_max3_f32 v18, v74, v118, v119
	v_max3_f32 v18, v18, v120, v121
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[34:49], v[182:185], v[226:229], v[34:49]
	v_max3_f32 v18, v18, v122, v123
	v_max3_f32 v18, v18, v124, v125
	v_max3_f32 v18, v18, v126, v127
	v_max3_f32 v18, v18, v128, v129
	v_mov_b32_e32 v19, v18
	s_nop 1
	v_permlane32_swap_b32_e32 v18, v19
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[34:49], v[178:181], v[174:177], v[34:49]
	scratch_load_dword v180, off, off offset:80 ; 4-byte Folded Reload
	v_mul_f32_e32 v74, v10, v186
	s_waitcnt vmcnt(0)
	v_max3_f32 v162, v180, v18, v19
	v_add_f32_e32 v19, v252, v253
	v_add_f32_e32 v19, v254, v19
	v_add_f32_e32 v19, v255, v19
	v_add_f32_e32 v1, v1, v19
	v_add_f32_e32 v1, v195, v1
	v_add_f32_e32 v1, v209, v1
	v_add_f32_e32 v1, v214, v1
	v_add_f32_e32 v1, v215, v1
	v_add_f32_e32 v1, v216, v1
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[34:49], v[66:69], v[170:173], v[34:49]
	v_add_f32_e32 v1, v217, v1
	v_add_f32_e32 v1, v218, v1
	v_add_f32_e32 v1, v196, v1
	v_add_f32_e32 v1, v219, v1
	v_add_f32_e32 v1, v220, v1
	v_add_f32_e32 v1, v221, v1
	v_add_f32_e32 v1, v210, v1
	v_mul_f32_e32 v18, 0x3e0293ee, v162
	v_add_f32_e32 v1, v206, v1
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[34:49], v[70:73], v[166:169], v[34:49]
	v_mul_f32_e32 v66, v2, v186
	v_mul_f32_e32 v67, v3, v186
	v_mul_f32_e32 v68, v4, v186
	v_mul_f32_e32 v69, v5, v186
	v_mul_f32_e32 v70, v6, v186
	v_mul_f32_e32 v71, v7, v186
	v_mul_f32_e32 v72, v8, v186
	v_mul_f32_e32 v73, v9, v186
	v_add_f32_e32 v1, v207, v1
	v_fma_f32 v3, v51, s12, -v18
	scratch_load_dword v218, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[66:81], v[222:225], v[226:229], v[66:81]
	scratch_load_dword v229, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v223, off, off       ; 4-byte Folded Reload
	scratch_load_dword v219, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v220, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v221, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v207, off, off offset:40 ; 4-byte Folded Reload
	v_add_f32_e32 v1, v208, v1
	scratch_load_dword v208, off, off offset:36 ; 4-byte Folded Reload
	v_fma_f32 v11, v59, s12, -v18
	v_exp_f32_e32 v59, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	scratch_load_dword v3, off, off offset:72 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[66:81], v[230:233], v[174:177], v[66:81]
	v_fma_f32 v2, v50, s12, -v18
	v_fma_f32 v4, v52, s12, -v18
	v_fma_f32 v5, v53, s12, -v18
	v_fma_f32 v6, v54, s12, -v18
	v_fma_f32 v7, v55, s12, -v18
	v_fma_f32 v8, v56, s12, -v18
	v_fma_f32 v9, v57, s12, -v18
	v_mfma_f32_32x32x16_f16 v[82:97], v[242:245], v[170:173], v[82:97]
	v_fma_f32 v10, v58, s12, -v18
	v_exp_f32_e32 v58, v2
	v_fma_f32 v2, v180, s12, -v18
	v_fma_f32 v12, v60, s12, -v18
	v_fma_f32 v13, v61, s12, -v18
	v_fma_f32 v14, v62, s12, -v18
	v_fma_f32 v15, v63, s12, -v18
	v_mfma_f32_32x32x16_f16 v[66:81], v[234:237], v[170:173], v[66:81]
	v_fma_f32 v16, v64, s12, -v18
	v_fma_f32 v17, v65, s12, -v18
	v_exp_f32_e32 v60, v4
	v_exp_f32_e32 v61, v5
	v_exp_f32_e32 v62, v6
	v_exp_f32_e32 v63, v7
	v_exp_f32_e32 v64, v8
	v_exp_f32_e32 v65, v9
	v_exp_f32_e32 v180, v2
	v_fma_f32 v19, v114, s12, -v18
	v_fma_f32 v20, v115, s12, -v18
	v_fma_f32 v21, v116, s12, -v18
	v_fma_f32 v22, v117, s12, -v18
	v_fma_f32 v23, v118, s12, -v18
	v_fma_f32 v24, v119, s12, -v18
	v_fma_f32 v25, v120, s12, -v18
	v_fma_f32 v26, v121, s12, -v18
	v_mfma_f32_32x32x16_f16 v[82:97], v[190:193], v[166:169], v[82:97]
	v_fma_f32 v27, v122, s12, -v18
	v_fma_f32 v28, v123, s12, -v18
	v_fma_f32 v29, v124, s12, -v18
	v_fma_f32 v30, v125, s12, -v18
	v_fma_f32 v31, v126, s12, -v18
	v_fma_f32 v32, v127, s12, -v18
	v_fma_f32 v33, v128, s12, -v18
	v_mfma_f32_32x32x16_f16 v[66:81], v[238:241], v[166:169], v[66:81]
	v_fma_f32 v50, v129, s12, -v18
	v_exp_f32_e32 v164, v19
	v_exp_f32_e32 v165, v20
	v_exp_f32_e32 v166, v21
	v_exp_f32_e32 v167, v22
	v_exp_f32_e32 v168, v23
	v_exp_f32_e32 v169, v24
	v_exp_f32_e32 v170, v25
	v_exp_f32_e32 v171, v26
	v_exp_f32_e32 v122, v10
	v_exp_f32_e32 v123, v11
	v_exp_f32_e32 v124, v12
	v_exp_f32_e32 v125, v13
	v_exp_f32_e32 v126, v14
	v_exp_f32_e32 v127, v15
	v_exp_f32_e32 v128, v16
	v_exp_f32_e32 v129, v17
	v_cvt_pk_f16_f32 v118, v58, v59
	v_cvt_pk_f16_f32 v119, v60, v61
	v_cvt_pk_f16_f32 v120, v62, v63
	v_cvt_pk_f16_f32 v121, v64, v65
	v_mul_f32_e32 v4, v100, v180
	v_mul_f32_e32 v5, v101, v180
	v_mul_f32_e32 v6, v102, v180
	v_mul_f32_e32 v7, v103, v180
	v_mul_f32_e32 v8, v104, v180
	v_mul_f32_e32 v9, v105, v180
	v_mul_f32_e32 v10, v106, v180
	v_mul_f32_e32 v11, v107, v180
	v_mul_f32_e32 v12, v108, v180
	v_mul_f32_e32 v13, v109, v180
	v_mul_f32_e32 v14, v110, v180
	v_mul_f32_e32 v15, v111, v180
	v_mul_f32_e32 v16, v112, v180
	v_mul_f32_e32 v17, v113, v180
	v_cvt_pk_f16_f32 v114, v122, v123
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v2, s2, v3
	v_add_u32_e32 v26, s3, v3
	ds_read_b64_tr_b16 v[18:19], v2 offset:32768
	ds_read_b64_tr_b16 v[20:21], v26 offset:2048
	ds_read_b64_tr_b16 v[22:23], v26 offset:4096
	ds_read_b64_tr_b16 v[24:25], v26 offset:6144
	v_mul_f32_e32 v2, v98, v180
	v_mul_f32_e32 v3, v99, v180
	v_cvt_pk_f16_f32 v115, v124, v125
	v_cvt_pk_f16_f32 v116, v126, v127
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[118:121], v[2:17]
	v_cvt_pk_f16_f32 v117, v128, v129
	ds_read_b64_tr_b16 v[18:19], v26 offset:8192
	ds_read_b64_tr_b16 v[20:21], v26 offset:10240
	v_cvt_pk_f16_f32 v98, v164, v165
	v_cvt_pk_f16_f32 v99, v166, v167
	v_cvt_pk_f16_f32 v100, v168, v169
	v_cvt_pk_f16_f32 v101, v170, v171
	v_exp_f32_e32 v172, v27
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[114:117], v[2:17]
	ds_read_b64_tr_b16 v[22:23], v26 offset:12288
	ds_read_b64_tr_b16 v[24:25], v26 offset:14336
	v_exp_f32_e32 v173, v28
	v_exp_f32_e32 v174, v29
	v_exp_f32_e32 v175, v30
	v_exp_f32_e32 v176, v31
	v_exp_f32_e32 v177, v32
	v_exp_f32_e32 v178, v33
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[98:101], v[2:17]
	scratch_load_dword v19, off, off offset:64 ; 4-byte Folded Reload
	v_exp_f32_e32 v179, v50
	v_cvt_pk_f16_f32 v102, v172, v173
	v_cvt_pk_f16_f32 v103, v174, v175
	v_cvt_pk_f16_f32 v104, v176, v177
	v_cvt_pk_f16_f32 v105, v178, v179
	v_mul_f32_e32 v20, v84, v180
	v_mul_f32_e32 v21, v85, v180
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[102:105], v[2:17]
	v_mul_f32_e32 v22, v86, v180
	v_mul_f32_e32 v23, v87, v180
	v_mul_f32_e32 v24, v88, v180
	v_mul_f32_e32 v25, v89, v180
	v_mul_f32_e32 v26, v90, v180
	v_mul_f32_e32 v27, v91, v180
	v_mul_f32_e32 v28, v92, v180
	v_mul_f32_e32 v29, v93, v180
	v_mul_f32_e32 v30, v94, v180
	v_mul_f32_e32 v31, v95, v180
	v_mul_f32_e32 v32, v96, v180
	v_mul_f32_e32 v33, v97, v180
	v_mul_f32_e32 v34, v34, v180
	v_mul_f32_e32 v35, v35, v180
	v_mul_f32_e32 v36, v36, v180
	v_mul_f32_e32 v37, v37, v180
	v_mul_f32_e32 v38, v38, v180
	v_mul_f32_e32 v39, v39, v180
	v_mul_f32_e32 v40, v40, v180
	v_mul_f32_e32 v41, v41, v180
	v_mul_f32_e32 v42, v42, v180
	v_mul_f32_e32 v43, v43, v180
	v_mul_f32_e32 v44, v44, v180
	v_mul_f32_e32 v45, v45, v180
	v_mul_f32_e32 v46, v46, v180
	v_mul_f32_e32 v47, v47, v180
	v_mul_f32_e32 v48, v48, v180
	v_mul_f32_e32 v49, v49, v180
	v_add_f32_e32 v1, v187, v1
	v_add_f32_e32 v1, v188, v1
	v_add_f32_e32 v1, v189, v1
	v_add_f32_e32 v1, v194, v1
	v_add_f32_e32 v1, v204, v1
	v_add_f32_e32 v1, v205, v1
	v_add_f32_e32 v1, v212, v1
	v_add_f32_e32 v1, v213, v1
	v_add_f32_e32 v1, v246, v1
	v_add_f32_e32 v1, v247, v1
	v_add_f32_e32 v1, v248, v1
	v_add_f32_e32 v1, v249, v1
	v_mov_b32_e32 v163, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v163
	v_add_f32_e32 v1, v1, v163
	v_mov_b32_e32 v209, v198
	v_mov_b32_e32 v198, v197
	v_mov_b32_e32 v216, v211
	v_lshlrev_b32_e32 v206, 3, v0
	v_mov_b32_e32 v217, v199
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v18, s2, v19
	v_add_u32_e32 v106, s3, v19
	ds_read_b64_tr_b16 v[50:51], v18 offset:32768
	ds_read_b64_tr_b16 v[52:53], v106 offset:2048
	ds_read_b64_tr_b16 v[54:55], v106 offset:4096
	ds_read_b64_tr_b16 v[56:57], v106 offset:6144
	v_mul_f32_e32 v18, v82, v180
	v_mul_f32_e32 v19, v83, v180
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[50:53], v[118:121], v[18:33]
	v_add_f32_e32 v50, v58, v59
	v_add_f32_e32 v50, v60, v50
	v_add_f32_e32 v58, v61, v50
	ds_read_b64_tr_b16 v[50:51], v106 offset:8192
	ds_read_b64_tr_b16 v[52:53], v106 offset:10240
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[54:57], v[114:117], v[18:33]
	v_add_f32_e32 v54, v62, v58
	v_add_f32_e32 v54, v63, v54
	v_add_f32_e32 v54, v64, v54
	v_add_f32_e32 v54, v65, v54
	v_add_f32_e32 v58, v122, v54
	ds_read_b64_tr_b16 v[54:55], v106 offset:12288
	ds_read_b64_tr_b16 v[56:57], v106 offset:14336
	v_add_f32_e32 v62, v123, v58
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[50:53], v[98:101], v[18:33]
	scratch_load_dword v51, off, off offset:68 ; 4-byte Folded Reload
	v_mul_f32_e32 v64, v80, v180
	v_mul_f32_e32 v65, v81, v180
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v50, s2, v51
	v_add_u32_e32 v63, s3, v51
	ds_read_b64_tr_b16 v[50:51], v50 offset:32768
	ds_read_b64_tr_b16 v[52:53], v63 offset:2048
	ds_read_b64_tr_b16 v[58:59], v63 offset:4096
	ds_read_b64_tr_b16 v[60:61], v63 offset:6144
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[50:53], v[118:121], v[34:49]
	v_add_f32_e32 v50, v124, v62
	v_add_f32_e32 v50, v125, v50
	v_add_f32_e32 v50, v126, v50
	v_add_f32_e32 v50, v127, v50
	v_mul_f32_e32 v62, v78, v180
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[58:61], v[114:117], v[34:49]
	v_mul_f32_e32 v59, v75, v180
	v_mul_f32_e32 v60, v76, v180
	v_mul_f32_e32 v61, v77, v180
	v_mfma_f32_32x32x16_f16 v[18:33], v[54:57], v[102:105], v[18:33]
	v_add_f32_e32 v54, v128, v50
	ds_read_b64_tr_b16 v[50:51], v63 offset:8192
	ds_read_b64_tr_b16 v[52:53], v63 offset:10240
	v_add_f32_e32 v54, v129, v54
	v_add_f32_e32 v54, v164, v54
	v_add_f32_e32 v54, v165, v54
	v_add_f32_e32 v54, v166, v54
	v_add_f32_e32 v58, v167, v54
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[50:53], v[98:101], v[34:49]
	ds_read_b64_tr_b16 v[54:55], v63 offset:12288
	ds_read_b64_tr_b16 v[56:57], v63 offset:14336
	v_mov_b32_e32 v51, v203
	v_add_u32_e32 v50, s2, v51
	v_add_u32_e32 v91, s3, v51
	ds_read_b64_tr_b16 v[82:83], v50 offset:32768
	ds_read_b64_tr_b16 v[84:85], v91 offset:2048
	ds_read_b64_tr_b16 v[86:87], v91 offset:4096
	ds_read_b64_tr_b16 v[88:89], v91 offset:6144
	v_add_f32_e32 v90, v168, v58
	v_mul_f32_e32 v50, v66, v180
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[34:49], v[54:57], v[102:105], v[34:49]
	v_mul_f32_e32 v51, v67, v180
	v_mul_f32_e32 v52, v68, v180
	v_mul_f32_e32 v53, v69, v180
	v_mul_f32_e32 v54, v70, v180
	v_mul_f32_e32 v55, v71, v180
	v_mul_f32_e32 v56, v72, v180
	v_mul_f32_e32 v57, v73, v180
	v_mul_f32_e32 v58, v74, v180
	v_mul_f32_e32 v63, v79, v180
	v_add_f32_e32 v66, v169, v90
	v_add_f32_e32 v66, v170, v66
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[82:85], v[118:121], v[50:65]
	v_add_f32_e32 v66, v171, v66
	v_add_f32_e32 v66, v172, v66
	v_add_f32_e32 v70, v173, v66
	ds_read_b64_tr_b16 v[66:67], v91 offset:8192
	ds_read_b64_tr_b16 v[68:69], v91 offset:10240
	v_add_f32_e32 v70, v174, v70
	v_add_f32_e32 v70, v175, v70
	v_add_f32_e32 v70, v176, v70
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[86:89], v[114:117], v[50:65]
	v_add_f32_e32 v70, v177, v70
	v_add_f32_e32 v74, v178, v70
	ds_read_b64_tr_b16 v[70:71], v91 offset:12288
	ds_read_b64_tr_b16 v[72:73], v91 offset:14336
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[66:69], v[98:101], v[50:65]
	v_add_f32_e32 v66, v179, v74
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v98, v66, v67
	scratch_load_dword v66, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v1, v66, v186
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[102:105], v[50:65]
	v_fmac_f32_e32 v98, v1, v180
.LBB0_15:                               ; %Flow1110
	v_lshrrev_b32_e32 v66, 1, v0
	v_and_b32_e32 v1, 64, v66
	v_and_b32_e32 v66, 0xa0, v66
	v_and_b32_e32 v67, 31, v0
	v_or3_b32 v99, v66, v67, v1
	s_sub_i32 s20, s33, s67
	s_sub_i32 s21, s69, s36
	v_or_b32_e32 v100, s60, v99
	s_cmp_lt_i32 s21, 1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_20
; %bb.16:                               ; %.lr.ph958
	scratch_load_dword v70, off, off offset:16 ; 4-byte Folded Reload
	s_cmp_lg_u32 s41, 0
	s_cselect_b64 s[2:3], -1, 0
	s_lshl_b32 s13, s36, 6
	s_mul_i32 s34, s13, s40
	s_mul_i32 s38, s13, s37
	s_and_b32 s13, s37, 0x3fff
	s_or_b32 s36, s13, 0x4000
	s_and_b32 s13, s40, 0x3fff
	s_lshl_b32 s14, s37, 6
	s_or_b32 s37, s13, 0x4000
	s_movk_i32 s13, 0x60
	v_lshl_add_u32 v66, v216, 1, 0
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v67, 12, v1
	v_lshl_add_u32 v69, v207, 1, v66
	s_lshl_b32 s12, s40, 6
	v_lshlrev_b32_e32 v72, 1, v208
	s_lshl_b32 s41, s69, 6
	v_or_b32_e32 v68, v208, v67
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s39, s38, 31
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[30:31], s[30:31], 1
	s_lshl_b64 s[28:29], s[28:29], 1
	v_bitop3_b32 v68, v68, v207, 64 bitop3:0x36
	v_lshlrev_b32_e32 v68, 1, v68
	s_mov_b32 s22, 0
	v_lshlrev_b32_e32 v1, 1, v202
	v_lshlrev_b32_e32 v110, 1, v200
	v_lshlrev_b32_e32 v111, 1, v209
	v_bfrev_b32_e32 v120, 1
	v_mov_b32_e32 v121, 0xff800000
	scratch_load_dword v199, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v73, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v101, 1, v70
	scratch_load_dword v70, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_or_b32_e32 v118, s41, v73
	s_waitcnt vmcnt(0)
	v_add3_u32 v102, 0, v218, v70
	v_add3_u32 v103, 0, v219, v70
	v_add3_u32 v104, 0, v220, v70
	v_add3_u32 v105, 0, v221, v70
	v_add3_u32 v106, 0, v223, v70
	v_add3_u32 v107, 0, v229, v70
	v_add3_u32 v108, 0, v198, v70
	v_add3_u32 v109, 0, v217, v70
	scratch_load_dwordx2 v[70:71], off, off offset:20 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v71, 1, v0
	v_bitop3_b32 v70, v70, v71, s13 bitop3:0x78
	scratch_load_dword v71, off, off offset:12 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v70, 1, v70
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v71, 1, v71
	v_add3_u32 v112, 0, v70, v71
	v_lshrrev_b32_e32 v70, 2, v0
	v_and_or_b32 v70, v70, 3, v73
	v_lshl_add_u32 v71, v67, 1, v69
	v_lshlrev_b32_e32 v70, 8, v70
	v_add3_u32 v113, v71, v72, v70
	v_bitop3_b32 v71, v67, v208, 32 bitop3:0x36
	v_bitop3_b32 v67, v206, v67, s13 bitop3:0x4e
	s_ashr_i32 s13, s12, 31
	s_add_u32 s23, s30, s28
	s_addc_u32 s28, s31, s29
	s_lshl_b64 s[18:19], s[18:19], 1
	s_add_u32 s23, s23, s18
	s_addc_u32 s28, s28, s19
	s_lshl_b64 s[18:19], s[34:35], 1
	s_add_u32 s18, s23, s18
	s_addc_u32 s19, s28, s19
	s_add_u32 s23, s6, s18
	s_addc_u32 s28, s7, s19
	s_lshl_b64 s[6:7], s[12:13], 1
	s_lshl_b64 s[12:13], s[26:27], 1
	s_lshl_b64 s[18:19], s[24:25], 1
	s_add_u32 s18, s12, s18
	s_addc_u32 s19, s13, s19
	s_lshl_b64 s[12:13], s[16:17], 1
	s_add_u32 s16, s18, s12
	s_addc_u32 s17, s19, s13
	s_lshl_b64 s[12:13], s[38:39], 1
	s_add_u32 s12, s16, s12
	s_addc_u32 s13, s17, s13
	s_add_u32 s24, s4, s12
	v_lshlrev_b32_e32 v67, 1, v67
	s_addc_u32 s25, s5, s13
	s_lshl_b32 s4, s21, 6
	v_lshlrev_b32_e32 v71, 1, v71
	v_add3_u32 v115, v66, v68, v70
	v_add3_u32 v116, v66, v67, v70
	s_sub_i32 s26, 0, s4
	s_add_i32 s4, s20, s41
	v_lshrrev_b32_e32 v66, 4, v0
	v_add3_u32 v114, v69, v71, v70
	s_lshl_b64 s[16:17], s[14:15], 1
	v_add_u32_e32 v117, s4, v73
	v_or_b32_e32 v119, s41, v66
	s_movk_i32 s27, 0xffc0
	s_lshl_b32 s29, s36, 16
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_xor_b64 s[18:19], s[2:3], -1
	s_mov_b32 s30, 0x3e0293ee
	s_lshl_b32 s31, s37, 16
	s_branch .LBB0_18
.LBB0_17:                               ;   in Loop: Header=BB0_18 Depth=1
	v_add_u32_e32 v163, s26, v117
	v_add_u32_e32 v164, 1, v163
	v_cmp_ge_i32_e32 vcc, v100, v163
	v_add_u32_e32 v165, 2, v163
	v_add_u32_e32 v166, 3, v163
	v_cndmask_b32_e32 v82, v121, v82, vcc
	v_cmp_ge_i32_e32 vcc, v100, v164
	v_add_u32_e32 v167, 8, v163
	v_add_u32_e32 v168, 9, v163
	v_cndmask_b32_e32 v83, v121, v83, vcc
	v_cmp_ge_i32_e32 vcc, v100, v165
	v_add_u32_e32 v169, 10, v163
	v_add_u32_e32 v170, 11, v163
	v_cndmask_b32_e32 v84, v121, v84, vcc
	v_cmp_ge_i32_e32 vcc, v100, v166
	v_add_u32_e32 v171, 16, v163
	v_add_u32_e32 v172, 17, v163
	v_cndmask_b32_e32 v85, v121, v85, vcc
	v_cmp_ge_i32_e32 vcc, v100, v167
	s_barrier
	s_nop 0
	v_cndmask_b32_e32 v86, v121, v86, vcc
	v_cmp_ge_i32_e32 vcc, v100, v168
	s_waitcnt vmcnt(1)
	ds_write_b128 v199, v[66:69]
	s_waitcnt vmcnt(0)
	ds_write_b128 v201, v[70:73]
	v_cndmask_b32_e32 v87, v121, v87, vcc
	v_cmp_ge_i32_e32 vcc, v100, v169
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cndmask_b32_e32 v88, v121, v88, vcc
	v_cmp_ge_i32_e32 vcc, v100, v170
	ds_read_b128 v[66:69], v102
	s_nop 0
	v_cndmask_b32_e32 v89, v121, v89, vcc
	v_cmp_ge_i32_e32 vcc, v100, v171
	v_add_u32_e32 v173, 18, v163
	v_add_u32_e32 v174, 19, v163
	v_cndmask_b32_e32 v90, v121, v90, vcc
	v_cmp_ge_i32_e32 vcc, v100, v172
	v_add_u32_e32 v175, 24, v163
	v_add_u32_e32 v176, 25, v163
	v_cndmask_b32_e32 v91, v121, v91, vcc
	v_cmp_ge_i32_e32 vcc, v100, v173
	v_add_u32_e32 v177, 26, v163
	v_add_u32_e32 v178, 27, v163
	v_cndmask_b32_e32 v92, v121, v92, vcc
	v_cmp_ge_i32_e32 vcc, v100, v174
	ds_read_b128 v[164:167], v102 offset:8192
	ds_read_b128 v[70:73], v103
	v_cndmask_b32_e32 v93, v121, v93, vcc
	v_cmp_ge_i32_e32 vcc, v100, v175
	v_add_u32_e32 v184, 32, v163
	v_add_u32_e32 v185, 33, v163
	v_cndmask_b32_e32 v94, v121, v94, vcc
	v_cmp_ge_i32_e32 vcc, v100, v176
	v_add_u32_e32 v186, 34, v163
	v_add_u32_e32 v192, 35, v163
	v_cndmask_b32_e32 v95, v121, v95, vcc
	v_cmp_ge_i32_e32 vcc, v100, v177
	v_add_u32_e32 v193, 40, v163
	v_add_u32_e32 v194, 41, v163
	v_cndmask_b32_e32 v96, v121, v96, vcc
	v_cmp_ge_i32_e32 vcc, v100, v178
	v_add_u32_e32 v196, 42, v163
	v_add_u32_e32 v197, 43, v163
	v_cndmask_b32_e32 v97, v121, v97, vcc
	v_cmp_ge_i32_e32 vcc, v100, v184
	v_add_u32_e32 v198, 48, v163
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[130:133], v[82:97]
	ds_read_b128 v[168:171], v103 offset:8192
	ds_read_b128 v[66:69], v104
	v_add_u32_e32 v200, 49, v163
	v_add_u32_e32 v206, 50, v163
	v_add_u32_e32 v207, 51, v163
	v_add_u32_e32 v208, 56, v163
	v_add_u32_e32 v209, 57, v163
	v_add_u32_e32 v210, 58, v163
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[70:73], v[134:137], v[82:97]
	ds_read_b128 v[172:175], v104 offset:8192
	ds_read_b128 v[70:73], v105
	v_add_u32_e32 v163, 59, v163
	ds_read_b128 v[176:179], v106
	ds_read_b128 v[180:183], v105 offset:8192
	s_and_b32 s12, s28, 0xffff
	s_or_b32 s13, s12, s31
	s_mov_b32 s12, s23
	s_add_i32 s22, s22, 1
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[138:141], v[82:97]
	v_cndmask_b32_e32 v66, v121, v123, vcc
	v_cmp_ge_i32_e32 vcc, v100, v185
	s_add_u32 s23, s23, s6
	s_addc_u32 s28, s28, s7
	v_cndmask_b32_e32 v67, v121, v127, vcc
	v_cmp_ge_i32_e32 vcc, v100, v186
	ds_read_b128 v[184:187], v107
	ds_read_b128 v[188:191], v106 offset:8192
	v_cndmask_b32_e32 v68, v121, v129, vcc
	v_cmp_ge_i32_e32 vcc, v100, v192
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[70:73], v[142:145], v[82:97]
	s_add_u32 s24, s24, s16
	v_cndmask_b32_e32 v69, v121, v77, vcc
	v_cmp_ge_i32_e32 vcc, v100, v193
	s_addc_u32 s25, s25, s17
	s_sub_i32 s27, s27, 64
	v_cndmask_b32_e32 v70, v121, v124, vcc
	v_cmp_ge_i32_e32 vcc, v100, v194
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[82:97], v[176:179], v[146:149], v[82:97]
	ds_read_b128 v[176:179], v108
	ds_read_b128 v[192:195], v107 offset:8192
	v_cndmask_b32_e32 v71, v121, v128, vcc
	v_cmp_ge_i32_e32 vcc, v100, v196
	v_add_u32_e32 v117, 64, v117
	v_add_u32_e32 v118, 64, v118
	v_cndmask_b32_e32 v72, v121, v74, vcc
	v_cmp_ge_i32_e32 vcc, v100, v197
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[82:97], v[184:187], v[150:153], v[82:97]
	ds_read_b128 v[184:187], v109
	ds_read_b128 v[202:205], v108 offset:8192
	v_cndmask_b32_e32 v73, v121, v80, vcc
	v_cmp_ge_i32_e32 vcc, v100, v198
	s_cmp_lt_i32 s22, s21
	v_add_u32_e32 v119, 64, v119
	v_cndmask_b32_e32 v74, v121, v125, vcc
	v_cmp_ge_i32_e32 vcc, v100, v200
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[82:97], v[176:179], v[154:157], v[82:97]
	ds_read_b128 v[176:179], v109 offset:8192
	v_cndmask_b32_e32 v75, v121, v75, vcc
	v_cmp_ge_i32_e32 vcc, v100, v206
	s_nop 1
	v_cndmask_b32_e32 v76, v121, v76, vcc
	v_cmp_ge_i32_e32 vcc, v100, v207
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[184:187], v[158:161], v[82:97]
	v_cndmask_b32_e32 v77, v121, v81, vcc
	v_cmp_ge_i32_e32 vcc, v100, v208
	s_nop 1
	v_cndmask_b32_e32 v78, v121, v78, vcc
	v_cmp_ge_i32_e32 vcc, v100, v209
	s_nop 5
	v_max_f32_e32 v123, v82, v82
	v_cndmask_b32_e32 v79, v121, v79, vcc
	v_cmp_ge_i32_e32 vcc, v100, v210
	s_nop 1
	v_cndmask_b32_e32 v80, v121, v122, vcc
	v_cmp_ge_i32_e32 vcc, v100, v163
	v_max_f32_e32 v122, v83, v83
	v_max_f32_e32 v122, v123, v122
	v_cndmask_b32_e32 v81, v121, v126, vcc
	v_max3_f32 v122, v122, v84, v85
	v_max3_f32 v122, v122, v86, v87
	v_mfma_f32_32x32x16_f16 v[66:81], v[164:167], v[130:133], v[66:81]
	v_max3_f32 v122, v122, v88, v89
	v_max3_f32 v122, v122, v90, v91
	v_max3_f32 v122, v122, v92, v93
	v_max3_f32 v122, v122, v94, v95
	v_max3_f32 v122, v122, v96, v97
	v_mfma_f32_32x32x16_f16 v[66:81], v[168:171], v[134:137], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[172:175], v[138:141], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[180:183], v[142:145], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[188:191], v[146:149], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[192:195], v[150:153], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[202:205], v[154:157], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[176:179], v[158:161], v[66:81]
	s_nop 7
	s_nop 3
	v_max3_f32 v122, v122, v66, v67
	v_max3_f32 v122, v122, v68, v69
	v_max3_f32 v122, v122, v70, v71
	v_max3_f32 v122, v122, v72, v73
	v_max3_f32 v122, v122, v74, v75
	v_max3_f32 v122, v122, v76, v77
	v_max3_f32 v122, v122, v78, v79
	v_max3_f32 v122, v122, v80, v81
	v_mov_b32_e32 v123, v122
	s_nop 1
	v_permlane32_swap_b32_e32 v122, v123
	v_max3_f32 v122, v162, v122, v123
	v_mul_f32_e32 v123, 0x3e0293ee, v122
	v_fma_f32 v124, v66, s30, -v123
	v_fma_f32 v128, v70, s30, -v123
	v_cndmask_b32_e64 v66, v120, v110, s[2:3]
	v_cndmask_b32_e64 v70, v120, v111, s[4:5]
	v_fma_f32 v125, v67, s30, -v123
	v_fma_f32 v126, v68, s30, -v123
	v_fma_f32 v127, v69, s30, -v123
	v_fma_f32 v129, v71, s30, -v123
	v_fma_f32 v163, v72, s30, -v123
	v_fma_f32 v164, v73, s30, -v123
	buffer_load_dwordx4 v[66:69], v66, s[12:15], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[70:73], v70, s[12:15], 0 offen
	v_fma_f32 v82, v82, s30, -v123
	v_fma_f32 v83, v83, s30, -v123
	v_fma_f32 v84, v84, s30, -v123
	v_exp_f32_e32 v82, v82
	v_exp_f32_e32 v83, v83
	v_fma_f32 v85, v85, s30, -v123
	v_exp_f32_e32 v84, v84
	v_fma_f32 v86, v86, s30, -v123
	v_exp_f32_e32 v85, v85
	v_fma_f32 v87, v87, s30, -v123
	v_fma_f32 v74, v74, s30, -v123
	v_exp_f32_e32 v86, v86
	v_fma_f32 v88, v88, s30, -v123
	v_fma_f32 v75, v75, s30, -v123
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v165, v74
	v_add_f32_e32 v74, v82, v83
	v_fma_f32 v89, v89, s30, -v123
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v166, v75
	v_add_f32_e32 v74, v84, v74
	v_fma_f32 v75, v162, s30, -v123
	v_fma_f32 v90, v90, s30, -v123
	v_fma_f32 v91, v91, s30, -v123
	v_fma_f32 v92, v92, s30, -v123
	v_fma_f32 v93, v93, s30, -v123
	v_fma_f32 v94, v94, s30, -v123
	v_fma_f32 v95, v95, s30, -v123
	v_fma_f32 v96, v96, s30, -v123
	v_fma_f32 v97, v97, s30, -v123
	v_fma_f32 v76, v76, s30, -v123
	v_fma_f32 v77, v77, s30, -v123
	v_fma_f32 v78, v78, s30, -v123
	v_fma_f32 v79, v79, s30, -v123
	v_fma_f32 v80, v80, s30, -v123
	v_fma_f32 v81, v81, s30, -v123
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v74, v85, v74
	v_exp_f32_e32 v123, v75
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b128 v112, v[66:69]
	s_waitcnt vmcnt(0)
	ds_write_b128 v112, v[70:73] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[70:71], v113
	ds_read_b64_tr_b16 v[72:73], v113 offset:2048
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v74, v86, v74
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v74, v87, v74
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v74, v88, v74
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v74, v89, v74
	v_mul_f32_e32 v2, v2, v123
	v_mul_f32_e32 v3, v3, v123
	v_mul_f32_e32 v4, v4, v123
	v_mul_f32_e32 v5, v5, v123
	v_mul_f32_e32 v6, v6, v123
	v_mul_f32_e32 v7, v7, v123
	v_mul_f32_e32 v8, v8, v123
	v_mul_f32_e32 v9, v9, v123
	v_mul_f32_e32 v10, v10, v123
	v_mul_f32_e32 v11, v11, v123
	v_mul_f32_e32 v12, v12, v123
	v_mul_f32_e32 v13, v13, v123
	v_mul_f32_e32 v14, v14, v123
	v_mul_f32_e32 v15, v15, v123
	v_mul_f32_e32 v16, v16, v123
	v_mul_f32_e32 v17, v17, v123
	v_cvt_pk_f16_f32 v66, v82, v83
	v_cvt_pk_f16_f32 v67, v84, v85
	v_cvt_pk_f16_f32 v68, v86, v87
	v_cvt_pk_f16_f32 v69, v88, v89
	v_add_f32_e32 v74, v90, v74
	v_add_f32_e32 v74, v91, v74
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[70:73], v[66:69], v[2:17]
	v_add_f32_e32 v74, v92, v74
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_exp_f32_e32 v167, v76
	v_exp_f32_e32 v168, v77
	v_add_f32_e32 v162, v93, v74
	ds_read_b64_tr_b16 v[74:75], v113 offset:4096
	ds_read_b64_tr_b16 v[76:77], v113 offset:6144
	v_cvt_pk_f16_f32 v70, v90, v91
	v_cvt_pk_f16_f32 v71, v92, v93
	v_cvt_pk_f16_f32 v72, v94, v95
	v_cvt_pk_f16_f32 v73, v96, v97
	v_exp_f32_e32 v124, v124
	v_exp_f32_e32 v125, v125
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[74:77], v[70:73], v[2:17]
	v_exp_f32_e32 v126, v126
	v_exp_f32_e32 v127, v127
	v_exp_f32_e32 v128, v128
	v_exp_f32_e32 v129, v129
	v_exp_f32_e32 v163, v163
	v_exp_f32_e32 v164, v164
	v_exp_f32_e32 v169, v78
	v_exp_f32_e32 v170, v79
	v_exp_f32_e32 v171, v80
	v_exp_f32_e32 v172, v81
	ds_read_b64_tr_b16 v[78:79], v113 offset:8192
	ds_read_b64_tr_b16 v[80:81], v113 offset:10240
	v_cvt_pk_f16_f32 v74, v124, v125
	v_cvt_pk_f16_f32 v75, v126, v127
	v_cvt_pk_f16_f32 v76, v128, v129
	v_cvt_pk_f16_f32 v77, v163, v164
	ds_read_b64_tr_b16 v[82:83], v113 offset:12288
	ds_read_b64_tr_b16 v[84:85], v113 offset:14336
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[78:81], v[74:77], v[2:17]
	ds_read_b64_tr_b16 v[86:87], v114
	ds_read_b64_tr_b16 v[88:89], v114 offset:2048
	v_mul_f32_e32 v18, v18, v123
	v_mul_f32_e32 v19, v19, v123
	v_mul_f32_e32 v20, v20, v123
	v_mul_f32_e32 v21, v21, v123
	v_mul_f32_e32 v22, v22, v123
	v_mul_f32_e32 v23, v23, v123
	v_mul_f32_e32 v24, v24, v123
	v_mul_f32_e32 v25, v25, v123
	v_mul_f32_e32 v26, v26, v123
	v_mul_f32_e32 v27, v27, v123
	v_mul_f32_e32 v28, v28, v123
	v_mul_f32_e32 v29, v29, v123
	v_mul_f32_e32 v30, v30, v123
	v_mul_f32_e32 v31, v31, v123
	v_mul_f32_e32 v32, v32, v123
	v_mul_f32_e32 v33, v33, v123
	v_cvt_pk_f16_f32 v78, v165, v166
	v_cvt_pk_f16_f32 v79, v167, v168
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[86:89], v[66:69], v[18:33]
	v_cvt_pk_f16_f32 v80, v169, v170
	v_cvt_pk_f16_f32 v81, v171, v172
	v_add_f32_e32 v86, v94, v162
	v_add_f32_e32 v86, v95, v86
	v_add_f32_e32 v86, v96, v86
	v_add_f32_e32 v90, v97, v86
	v_mul_f32_e32 v34, v34, v123
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[78:81], v[2:17]
	ds_read_b64_tr_b16 v[82:83], v114 offset:4096
	ds_read_b64_tr_b16 v[84:85], v114 offset:6144
	ds_read_b64_tr_b16 v[86:87], v114 offset:8192
	ds_read_b64_tr_b16 v[88:89], v114 offset:10240
	v_mul_f32_e32 v35, v35, v123
	v_mul_f32_e32 v36, v36, v123
	v_mul_f32_e32 v37, v37, v123
	v_mul_f32_e32 v38, v38, v123
	v_mul_f32_e32 v39, v39, v123
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[70:73], v[18:33]
	ds_read_b64_tr_b16 v[82:83], v114 offset:12288
	ds_read_b64_tr_b16 v[84:85], v114 offset:14336
	v_mul_f32_e32 v40, v40, v123
	v_mul_f32_e32 v41, v41, v123
	v_mul_f32_e32 v42, v42, v123
	v_mul_f32_e32 v43, v43, v123
	v_mul_f32_e32 v44, v44, v123
	v_mul_f32_e32 v45, v45, v123
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[86:89], v[74:77], v[18:33]
	ds_read_b64_tr_b16 v[86:87], v115
	ds_read_b64_tr_b16 v[88:89], v115 offset:2048
	v_mul_f32_e32 v46, v46, v123
	v_mul_f32_e32 v47, v47, v123
	v_mul_f32_e32 v48, v48, v123
	v_mul_f32_e32 v49, v49, v123
	v_mul_f32_e32 v50, v50, v123
	v_mul_f32_e32 v51, v51, v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[86:89], v[66:69], v[34:49]
	v_add_f32_e32 v86, v124, v90
	v_add_f32_e32 v86, v125, v86
	v_add_f32_e32 v86, v126, v86
	v_add_f32_e32 v90, v127, v86
	v_mul_f32_e32 v52, v52, v123
	v_mul_f32_e32 v53, v53, v123
	v_mul_f32_e32 v54, v54, v123
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[78:81], v[18:33]
	ds_read_b64_tr_b16 v[82:83], v115 offset:4096
	ds_read_b64_tr_b16 v[84:85], v115 offset:6144
	ds_read_b64_tr_b16 v[86:87], v115 offset:8192
	ds_read_b64_tr_b16 v[88:89], v115 offset:10240
	v_mul_f32_e32 v55, v55, v123
	v_mul_f32_e32 v56, v56, v123
	v_mul_f32_e32 v57, v57, v123
	v_mul_f32_e32 v58, v58, v123
	v_mul_f32_e32 v59, v59, v123
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[70:73], v[34:49]
	ds_read_b64_tr_b16 v[82:83], v115 offset:12288
	ds_read_b64_tr_b16 v[84:85], v115 offset:14336
	v_mul_f32_e32 v60, v60, v123
	v_mul_f32_e32 v61, v61, v123
	v_mul_f32_e32 v62, v62, v123
	v_mul_f32_e32 v63, v63, v123
	v_mul_f32_e32 v64, v64, v123
	v_mul_f32_e32 v65, v65, v123
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[86:89], v[74:77], v[34:49]
	ds_read_b64_tr_b16 v[86:87], v116
	ds_read_b64_tr_b16 v[88:89], v116 offset:2048
	v_mov_b32_e32 v162, v122
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[86:89], v[66:69], v[50:65]
	v_add_f32_e32 v66, v128, v90
	v_add_f32_e32 v66, v129, v66
	v_add_f32_e32 v66, v163, v66
	v_add_f32_e32 v66, v164, v66
	v_add_f32_e32 v86, v165, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[78:81], v[34:49]
	ds_read_b64_tr_b16 v[82:83], v116 offset:4096
	ds_read_b64_tr_b16 v[84:85], v116 offset:6144
	ds_read_b64_tr_b16 v[66:67], v116 offset:8192
	ds_read_b64_tr_b16 v[68:69], v116 offset:10240
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[82:85], v[70:73], v[50:65]
	v_add_f32_e32 v70, v166, v86
	v_add_f32_e32 v70, v167, v70
	v_add_f32_e32 v70, v168, v70
	v_add_f32_e32 v70, v169, v70
	v_add_f32_e32 v82, v170, v70
	ds_read_b64_tr_b16 v[70:71], v116 offset:12288
	ds_read_b64_tr_b16 v[72:73], v116 offset:14336
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[66:69], v[74:77], v[50:65]
	v_add_f32_e32 v66, v171, v82
	v_add_f32_e32 v66, v172, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v98, v123
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[78:81], v[50:65]
	v_mov_b32_e32 v98, v66
	s_cbranch_scc0 .LBB0_20
.LBB0_18:                               ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v66, s26, v119
	v_add_u32_e32 v67, 32, v66
	s_and_b32 s2, s25, 0xffff
	s_or_b32 s13, s2, s29
	v_cmp_gt_i32_e64 s[2:3], s67, v66
	v_cmp_gt_i32_e64 s[4:5], s67, v67
	s_mov_b32 s12, s24
	v_cndmask_b32_e64 v66, v120, v1, s[2:3]
	v_cndmask_b32_e64 v70, v120, v101, s[4:5]
	buffer_load_dwordx4 v[66:69], v66, s[12:15], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[70:73], v70, s[12:15], 0 offen
	s_cmp_lg_u32 s26, s27
	s_cselect_b64 s[12:13], -1, 0
	s_or_b64 s[12:13], s[18:19], s[12:13]
	v_mov_b32_e32 v82, 0
	s_and_b64 vcc, exec, s[12:13]
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v126, 0
	s_cbranch_vccnz .LBB0_17
; %bb.19:                               ;   in Loop: Header=BB0_18 Depth=1
	v_add_u32_e32 v74, s26, v118
	v_add_u32_e32 v75, 1, v74
	v_cmp_gt_i32_e32 vcc, s67, v74
	v_add_u32_e32 v76, 2, v74
	v_add_u32_e32 v77, 3, v74
	v_cndmask_b32_e64 v82, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v75
	v_add_u32_e32 v78, 8, v74
	v_add_u32_e32 v79, 9, v74
	v_cndmask_b32_e64 v83, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v76
	v_add_u32_e32 v80, 10, v74
	v_add_u32_e32 v81, 11, v74
	v_cndmask_b32_e64 v84, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v77
	v_add_u32_e32 v90, 16, v74
	v_add_u32_e32 v91, 17, v74
	v_cndmask_b32_e64 v85, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v78
	v_add_u32_e32 v92, 18, v74
	v_add_u32_e32 v93, 19, v74
	v_cndmask_b32_e64 v86, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v79
	v_add_u32_e32 v94, 24, v74
	v_add_u32_e32 v95, 25, v74
	v_cndmask_b32_e64 v87, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v80
	v_add_u32_e32 v96, 26, v74
	v_add_u32_e32 v97, 27, v74
	v_cndmask_b32_e64 v88, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v81
	v_add_u32_e32 v122, 32, v74
	v_add_u32_e32 v124, 33, v74
	v_cndmask_b32_e64 v89, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v90
	v_add_u32_e32 v125, 34, v74
	v_add_u32_e32 v126, 35, v74
	v_cndmask_b32_e64 v90, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v91
	v_add_u32_e32 v128, 40, v74
	v_add_u32_e32 v163, 41, v74
	v_cndmask_b32_e64 v91, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v92
	v_add_u32_e32 v164, 42, v74
	v_add_u32_e32 v165, 43, v74
	v_cndmask_b32_e64 v92, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v93
	v_add_u32_e32 v166, 48, v74
	v_add_u32_e32 v167, 49, v74
	v_cndmask_b32_e64 v93, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v94
	v_add_u32_e32 v168, 50, v74
	v_add_u32_e32 v169, 51, v74
	v_cndmask_b32_e64 v94, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v95
	v_add_u32_e32 v170, 56, v74
	v_add_u32_e32 v171, 57, v74
	v_cndmask_b32_e64 v95, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v96
	v_add_u32_e32 v172, 58, v74
	v_add_u32_e32 v173, 59, v74
	v_cndmask_b32_e64 v96, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v97
	s_nop 1
	v_cndmask_b32_e64 v97, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v122
	s_nop 1
	v_cndmask_b32_e64 v123, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v124
	s_nop 1
	v_cndmask_b32_e64 v127, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v125
	s_nop 1
	v_cndmask_b32_e64 v129, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v126
	s_nop 1
	v_cndmask_b32_e64 v77, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v128
	s_nop 1
	v_cndmask_b32_e64 v124, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v163
	s_nop 1
	v_cndmask_b32_e64 v128, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v164
	s_nop 1
	v_cndmask_b32_e64 v74, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v165
	s_nop 1
	v_cndmask_b32_e64 v80, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v166
	s_nop 1
	v_cndmask_b32_e64 v125, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v167
	s_nop 1
	v_cndmask_b32_e64 v75, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v168
	s_nop 1
	v_cndmask_b32_e64 v76, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v169
	s_nop 1
	v_cndmask_b32_e64 v81, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v170
	s_nop 1
	v_cndmask_b32_e64 v78, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v171
	s_nop 1
	v_cndmask_b32_e64 v79, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v172
	s_nop 1
	v_cndmask_b32_e64 v122, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v173
	s_nop 1
	v_cndmask_b32_e64 v126, v121, 0, vcc
	s_branch .LBB0_17
.LBB0_20:                               ; %._crit_edge959
	v_div_scale_f32 v1, s[2:3], v98, v98, 1.0
	v_rcp_f32_e32 v1, v1
	v_div_scale_f32 v66, vcc, 1.0, v98, 1.0
	s_add_i32 s2, s60, 0x100
	v_mul_f32_e32 v1, v66, v1
	s_cmp_le_i32 s20, s60
	s_nop 0
	v_div_fmas_f32 v1, 0, 0, v1
	v_div_fixup_f32 v66, v1, v98, 1.0
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_ge_i32 s20, s2
	v_pk_mul_f32 v[2:3], v[2:3], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[34:35], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[36:37], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[38:39], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[40:41], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[46:47], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[64:65], v[66:67] op_sel_hi:[1,0]
	s_cselect_b64 s[6:7], -1, 0
	v_cvt_pk_f16_f32 v61, v2, v3
	v_cvt_pk_f16_f32 v63, v4, v5
	v_cvt_pk_f16_f32 v55, v6, v7
	v_cvt_pk_f16_f32 v59, v8, v9
	v_cvt_pk_f16_f32 v51, v10, v11
	v_cvt_pk_f16_f32 v56, v12, v13
	v_cvt_pk_f16_f32 v47, v14, v15
	v_cvt_pk_f16_f32 v52, v16, v17
	v_cvt_pk_f16_f32 v43, v18, v19
	v_cvt_pk_f16_f32 v48, v20, v21
	v_cvt_pk_f16_f32 v39, v22, v23
	v_cvt_pk_f16_f32 v44, v24, v25
	v_cvt_pk_f16_f32 v35, v26, v27
	v_cvt_pk_f16_f32 v40, v28, v29
	v_cvt_pk_f16_f32 v31, v30, v31
	v_cvt_pk_f16_f32 v36, v68, v69
	v_cvt_pk_f16_f32 v27, v70, v71
	v_cvt_pk_f16_f32 v32, v72, v73
	v_cvt_pk_f16_f32 v23, v74, v75
	v_cvt_pk_f16_f32 v28, v76, v77
	v_cvt_pk_f16_f32 v19, v78, v79
	v_cvt_pk_f16_f32 v24, v80, v81
	v_cvt_pk_f16_f32 v15, v82, v83
	v_cvt_pk_f16_f32 v20, v84, v85
	v_cvt_pk_f16_f32 v11, v86, v87
	v_cvt_pk_f16_f32 v16, v88, v89
	v_cvt_pk_f16_f32 v7, v90, v91
	v_cvt_pk_f16_f32 v12, v92, v93
	v_cvt_pk_f16_f32 v3, v94, v95
	v_cvt_pk_f16_f32 v8, v96, v97
	v_cvt_pk_f16_f32 v1, v102, v103
	v_cvt_pk_f16_f32 v4, v66, v67
	s_or_b64 s[4:5], s[4:5], s[6:7]
	v_lshrrev_b32_e32 v62, 16, v61
	v_lshrrev_b32_e32 v64, 16, v63
	v_lshrrev_b32_e32 v57, 16, v55
	v_lshrrev_b32_e32 v60, 16, v59
	v_lshrrev_b32_e32 v53, 16, v51
	v_lshrrev_b32_e32 v58, 16, v56
	v_lshrrev_b32_e32 v49, 16, v47
	v_lshrrev_b32_e32 v54, 16, v52
	v_lshrrev_b32_e32 v45, 16, v43
	v_lshrrev_b32_e32 v50, 16, v48
	v_lshrrev_b32_e32 v41, 16, v39
	v_lshrrev_b32_e32 v46, 16, v44
	v_lshrrev_b32_e32 v37, 16, v35
	v_lshrrev_b32_e32 v42, 16, v40
	v_lshrrev_b32_e32 v33, 16, v31
	v_lshrrev_b32_e32 v38, 16, v36
	v_lshrrev_b32_e32 v29, 16, v27
	v_lshrrev_b32_e32 v34, 16, v32
	v_lshrrev_b32_e32 v25, 16, v23
	v_lshrrev_b32_e32 v30, 16, v28
	v_lshrrev_b32_e32 v21, 16, v19
	v_lshrrev_b32_e32 v26, 16, v24
	v_lshrrev_b32_e32 v17, 16, v15
	v_lshrrev_b32_e32 v22, 16, v20
	v_lshrrev_b32_e32 v13, 16, v11
	v_lshrrev_b32_e32 v18, 16, v16
	v_lshrrev_b32_e32 v9, 16, v7
	v_lshrrev_b32_e32 v14, 16, v12
	v_lshrrev_b32_e32 v5, 16, v3
	v_lshrrev_b32_e32 v10, 16, v8
	v_lshrrev_b32_e32 v2, 16, v1
	v_lshrrev_b32_e32 v6, 16, v4
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_22
; %bb.21:
	v_cmp_gt_i32_e32 vcc, s20, v100
	s_nop 1
	v_cndmask_b32_e64 v61, v61, 0, vcc
	v_cndmask_b32_e64 v62, v62, 0, vcc
	v_cndmask_b32_e64 v63, v63, 0, vcc
	v_cndmask_b32_e64 v64, v64, 0, vcc
	v_cndmask_b32_e64 v55, v55, 0, vcc
	v_cndmask_b32_e64 v57, v57, 0, vcc
	v_cndmask_b32_e64 v59, v59, 0, vcc
	v_cndmask_b32_e64 v60, v60, 0, vcc
	v_cndmask_b32_e64 v51, v51, 0, vcc
	v_cndmask_b32_e64 v53, v53, 0, vcc
	v_cndmask_b32_e64 v56, v56, 0, vcc
	v_cndmask_b32_e64 v58, v58, 0, vcc
	v_cndmask_b32_e64 v47, v47, 0, vcc
	v_cndmask_b32_e64 v49, v49, 0, vcc
	v_cndmask_b32_e64 v52, v52, 0, vcc
	v_cndmask_b32_e64 v54, v54, 0, vcc
	v_cndmask_b32_e64 v43, v43, 0, vcc
	v_cndmask_b32_e64 v45, v45, 0, vcc
	v_cndmask_b32_e64 v48, v48, 0, vcc
	v_cndmask_b32_e64 v50, v50, 0, vcc
	v_cndmask_b32_e64 v39, v39, 0, vcc
	v_cndmask_b32_e64 v41, v41, 0, vcc
	v_cndmask_b32_e64 v44, v44, 0, vcc
	v_cndmask_b32_e64 v46, v46, 0, vcc
	v_cndmask_b32_e64 v35, v35, 0, vcc
	v_cndmask_b32_e64 v37, v37, 0, vcc
	v_cndmask_b32_e64 v40, v40, 0, vcc
	v_cndmask_b32_e64 v42, v42, 0, vcc
	v_cndmask_b32_e64 v31, v31, 0, vcc
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cndmask_b32_e64 v36, v36, 0, vcc
	v_cndmask_b32_e64 v38, v38, 0, vcc
	v_cndmask_b32_e64 v27, v27, 0, vcc
	v_cndmask_b32_e64 v29, v29, 0, vcc
	v_cndmask_b32_e64 v32, v32, 0, vcc
	v_cndmask_b32_e64 v34, v34, 0, vcc
	v_cndmask_b32_e64 v23, v23, 0, vcc
	v_cndmask_b32_e64 v25, v25, 0, vcc
	v_cndmask_b32_e64 v28, v28, 0, vcc
	v_cndmask_b32_e64 v30, v30, 0, vcc
	v_cndmask_b32_e64 v19, v19, 0, vcc
	v_cndmask_b32_e64 v21, v21, 0, vcc
	v_cndmask_b32_e64 v24, v24, 0, vcc
	v_cndmask_b32_e64 v26, v26, 0, vcc
	v_cndmask_b32_e64 v15, v15, 0, vcc
	v_cndmask_b32_e64 v17, v17, 0, vcc
	v_cndmask_b32_e64 v20, v20, 0, vcc
	v_cndmask_b32_e64 v22, v22, 0, vcc
	v_cndmask_b32_e64 v11, v11, 0, vcc
	v_cndmask_b32_e64 v13, v13, 0, vcc
	v_cndmask_b32_e64 v16, v16, 0, vcc
	v_cndmask_b32_e64 v18, v18, 0, vcc
	v_cndmask_b32_e64 v7, v7, 0, vcc
	v_cndmask_b32_e64 v9, v9, 0, vcc
	v_cndmask_b32_e64 v12, v12, 0, vcc
	v_cndmask_b32_e64 v14, v14, 0, vcc
	v_cndmask_b32_e64 v3, v3, 0, vcc
	v_cndmask_b32_e64 v5, v5, 0, vcc
	v_cndmask_b32_e64 v8, v8, 0, vcc
	v_cndmask_b32_e64 v10, v10, 0, vcc
	v_cndmask_b32_e64 v1, v1, 0, vcc
	v_cndmask_b32_e64 v2, v2, 0, vcc
	v_cndmask_b32_e64 v4, v4, 0, vcc
	v_cndmask_b32_e64 v6, v6, 0, vcc
.LBB0_22:
	s_ashr_i32 s65, s64, 31
	s_lshl_b64 s[4:5], s[64:65], 2
	s_add_u32 s3, s8, s4
	s_addc_u32 s6, s9, s5
	s_ashr_i32 s63, s62, 31
	s_lshl_b64 s[4:5], s[62:63], 2
	s_add_u32 s3, s3, s4
	s_addc_u32 s6, s6, s5
	s_ashr_i32 s61, s60, 31
	s_lshl_b64 s[4:5], s[60:61], 2
	s_add_u32 s4, s3, s4
	s_addc_u32 s12, s6, s5
	s_sub_i32 s2, s2, s33
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB0_24
; %bb.23:
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v98
	v_mov_b32_e32 v65, 0x42000000
	s_nop 0
	v_cndmask_b32_e64 v66, 0, 32, vcc
	v_ldexp_f32 v66, v98, v66
	v_log_f32_e32 v66, v66
	v_cndmask_b32_e32 v65, 0, v65, vcc
	s_barrier
	v_sub_f32_e32 v65, v66, v65
	v_add_f32_e32 v65, v162, v65
	v_lshl_add_u32 v66, v99, 2, 0
	ds_write_b32 v66, v65
	v_mov_b32_e32 v65, 2
	v_lshlrev_b32_sdwa v65, v65, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v65
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v66, v66
	s_sub_i32 s2, 0x100, s2
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	v_bfrev_b32_e32 v67, 1
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cmp_gt_i32_e64 s[8:9], s33, v100
	s_and_b32 s5, s12, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e32 v65, v67, v65, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v65, s[4:7], 0 offen
	s_cbranch_execz .LBB0_25
	s_branch .LBB0_26
.LBB0_24:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_25:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v98
	v_mov_b32_e32 v65, 0x42000000
	s_nop 0
	v_cndmask_b32_e64 v66, 0, 32, vcc
	v_ldexp_f32 v66, v98, v66
	v_log_f32_e32 v66, v66
	v_cndmask_b32_e32 v65, 0, v65, vcc
	s_barrier
	v_sub_f32_e32 v65, v66, v65
	v_add_f32_e32 v65, v162, v65
	v_lshl_add_u32 v66, v99, 2, 0
	ds_write_b32 v66, v65
	v_mov_b32_e32 v65, 2
	v_lshlrev_b32_sdwa v0, v65, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v65, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v65, v65
	v_bfrev_b32_e32 v66, 1
	s_and_b32 s5, s12, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e64 v0, v66, v0, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v65, v0, s[4:7], 0 offen
.LBB0_26:
	s_mov_b32 s4, 0x5040100
	v_perm_b32 v62, v62, v61, s4
	scratch_load_dword v61, off, off offset:8 ; 4-byte Folded Reload
	s_ashr_i32 s59, s58, 31
	s_lshl_b64 s[0:1], s[58:59], 1
	s_add_u32 s2, s10, s0
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s57, s56, 31
	s_lshl_b64 s[0:1], s[56:57], 1
	s_add_u32 s2, s2, s0
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[0:1], s[54:55], 1
	s_add_u32 s2, s2, s0
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s53, s52, 31
	s_lshl_b64 s[0:1], s[52:53], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s43, 0x3fff
	v_mul_lo_u32 v0, s43, v99
	s_bitset1_b32 s2, 14
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_perm_b32 v63, v64, v63, s4
	v_bfrev_b32_e32 v64, 1
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_perm_b32 v2, v2, v1, s4
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v0, v0, v61, 1
	v_cndmask_b32_e64 v61, v64, v0, s[8:9]
	buffer_store_dwordx2 v[62:63], v61, s[0:3], 0 offen
	v_perm_b32 v61, v60, v59, s4
	v_perm_b32 v60, v57, v55, s4
	v_add_u32_e32 v55, 16, v0
	v_perm_b32 v57, v58, v56, s4
	v_perm_b32 v56, v53, v51, s4
	v_add_u32_e32 v51, 32, v0
	v_perm_b32 v53, v54, v52, s4
	v_perm_b32 v52, v49, v47, s4
	v_add_u32_e32 v47, 48, v0
	v_perm_b32 v49, v50, v48, s4
	v_perm_b32 v48, v45, v43, s4
	v_add_u32_e32 v43, 64, v0
	v_perm_b32 v45, v46, v44, s4
	v_perm_b32 v44, v41, v39, s4
	v_add_u32_e32 v39, 0x50, v0
	v_perm_b32 v41, v42, v40, s4
	v_perm_b32 v40, v37, v35, s4
	v_add_u32_e32 v35, 0x60, v0
	v_perm_b32 v37, v38, v36, s4
	v_perm_b32 v36, v33, v31, s4
	v_add_u32_e32 v31, 0x70, v0
	v_perm_b32 v33, v34, v32, s4
	v_perm_b32 v32, v29, v27, s4
	v_add_u32_e32 v27, 0x80, v0
	v_perm_b32 v29, v30, v28, s4
	v_perm_b32 v28, v25, v23, s4
	v_add_u32_e32 v23, 0x90, v0
	v_perm_b32 v25, v26, v24, s4
	v_perm_b32 v24, v21, v19, s4
	v_add_u32_e32 v19, 0xa0, v0
	v_perm_b32 v21, v22, v20, s4
	v_perm_b32 v20, v17, v15, s4
	v_add_u32_e32 v15, 0xb0, v0
	v_perm_b32 v17, v18, v16, s4
	v_perm_b32 v16, v13, v11, s4
	v_add_u32_e32 v11, 0xc0, v0
	v_perm_b32 v13, v14, v12, s4
	v_perm_b32 v12, v9, v7, s4
	v_add_u32_e32 v7, 0xd0, v0
	v_perm_b32 v9, v10, v8, s4
	v_perm_b32 v8, v5, v3, s4
	v_add_u32_e32 v3, 0xe0, v0
	v_cndmask_b32_e64 v55, v64, v55, s[8:9]
	v_cndmask_b32_e64 v51, v64, v51, s[8:9]
	v_cndmask_b32_e64 v47, v64, v47, s[8:9]
	v_cndmask_b32_e64 v43, v64, v43, s[8:9]
	v_cndmask_b32_e64 v39, v64, v39, s[8:9]
	v_cndmask_b32_e64 v35, v64, v35, s[8:9]
	v_cndmask_b32_e64 v31, v64, v31, s[8:9]
	v_cndmask_b32_e64 v27, v64, v27, s[8:9]
	v_cndmask_b32_e64 v23, v64, v23, s[8:9]
	v_cndmask_b32_e64 v19, v64, v19, s[8:9]
	v_cndmask_b32_e64 v15, v64, v15, s[8:9]
	v_cndmask_b32_e64 v11, v64, v11, s[8:9]
	v_cndmask_b32_e64 v7, v64, v7, s[8:9]
	v_cndmask_b32_e64 v3, v64, v3, s[8:9]
	v_add_u32_e32 v0, 0xf0, v0
	buffer_store_dwordx2 v[60:61], v55, s[0:3], 0 offen
	buffer_store_dwordx2 v[56:57], v51, s[0:3], 0 offen
	buffer_store_dwordx2 v[52:53], v47, s[0:3], 0 offen
	buffer_store_dwordx2 v[48:49], v43, s[0:3], 0 offen
	buffer_store_dwordx2 v[44:45], v39, s[0:3], 0 offen
	buffer_store_dwordx2 v[40:41], v35, s[0:3], 0 offen
	buffer_store_dwordx2 v[36:37], v31, s[0:3], 0 offen
	buffer_store_dwordx2 v[32:33], v27, s[0:3], 0 offen
	buffer_store_dwordx2 v[28:29], v23, s[0:3], 0 offen
	buffer_store_dwordx2 v[24:25], v19, s[0:3], 0 offen
	buffer_store_dwordx2 v[20:21], v15, s[0:3], 0 offen
	buffer_store_dwordx2 v[16:17], v11, s[0:3], 0 offen
	buffer_store_dwordx2 v[12:13], v7, s[0:3], 0 offen
	buffer_store_dwordx2 v[8:9], v3, s[0:3], 0 offen
	v_perm_b32 v3, v6, v4, s4
	v_cndmask_b32_e64 v0, v64, v0, s[8:9]
	buffer_store_dwordx2 v[2:3], v0, s[0:3], 0 offen
.LBB0_27:                               ; %.critedge
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 88
		.amdhsa_kernarg_size 160
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
		.amdhsa_next_free_sgpr 76
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
	.set attn_fwd.numbered_sgpr, 76
	.set attn_fwd.private_seg_size, 88
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 18896
; TotalNumSgprs: 82
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 88
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 10
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 82
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
	.byte	1                               ; Abbrev [1] 0xb:0x6c DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x46 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	529                             ; DW_AT_call_line
	.byte	41                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	536                             ; DW_AT_call_line
	.byte	89                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x5b:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	683                             ; DW_AT_call_line
	.byte	52                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x68:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	58                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
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
	.quad	0
	.quad	0
.Ldebug_ranges3:
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
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .offset:         128
        .size:           4
        .value_kind:     by_value
      - .offset:         132
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     global_buffer
      - .offset:         144
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 160
    .max_flat_workgroup_size: 512
    .name:           attn_fwd
    .private_segment_fixed_size: 88
    .sgpr_count:     82
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 22
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
