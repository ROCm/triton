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
; %bb.23:
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.24:
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
	s_cbranch_scc1 .LBB0_22
; %bb.1:
	s_add_u32 s20, s22, s24
	s_addc_u32 s21, s23, s25
	s_load_dwordx2 s[66:67], s[20:21], 0x0
	s_load_dwordx8 s[36:43], s[0:1], 0x38
	v_and_b32_e32 v1, 0x100, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v1
	v_lshrrev_b32_e32 v1, 4, v0
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
	v_or_b32_e32 v9, s60, v1
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
	v_lshlrev_b32_e32 v200, 3, v0
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
	v_and_b32_e32 v10, 0x78, v200
	s_addc_u32 s16, s19, s47
	v_mad_u64_u32 v[10:11], s[46:47], s43, v1, v[10:11]
	s_and_b32 s19, s43, 0x3fff
	v_add_u32_e32 v15, s41, v10
	s_bitset1_b32 s19, 14
	v_lshlrev_b32_e32 v10, 1, v10
	v_bfrev_b32_e32 v21, 1
	s_mov_b32 s45, s44
	v_add_u32_e32 v16, s41, v15
	s_and_b32 s16, s16, 0xffff
	s_lshl_b32 s19, s19, 16
	v_cndmask_b32_e64 v22, v21, v10, s[26:27]
	s_mov_b32 s46, s44
	s_mov_b32 s47, s44
	v_mov_b64_e32 v[10:11], s[44:45]
	v_lshlrev_b32_e32 v15, 1, v15
	s_or_b32 s49, s16, s19
	s_mov_b32 s51, 0x27000
	s_mov_b32 s50, 0x7ffffffe
	v_mov_b64_e32 v[12:13], s[46:47]
	v_cndmask_b32_e64 v15, v21, v15, s[24:25]
	buffer_store_dwordx4 v[10:13], v22, s[48:51], 0 offen
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v16
	v_add_u32_e32 v17, s41, v16
	v_cndmask_b32_e64 v15, v21, v15, s[22:23]
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v17
	v_add_u32_e32 v18, s41, v17
	v_cndmask_b32_e64 v15, v21, v15, s[20:21]
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v18
	s_ashr_i32 s65, s64, 31
	v_add_u32_e32 v19, s41, v18
	v_cndmask_b32_e64 v15, v21, v15, s[30:31]
	s_lshl_b64 s[20:21], s[64:65], 2
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v19
	s_add_u32 s16, s8, s20
	v_add_u32_e32 v20, s41, v19
	v_cndmask_b32_e64 v15, v21, v15, s[28:29]
	s_addc_u32 s19, s9, s21
	s_ashr_i32 s63, s62, 31
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v20
	s_lshl_b64 s[20:21], s[62:63], 2
	v_cndmask_b32_e64 v15, v21, v15, s[34:35]
	s_add_u32 s16, s16, s20
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_add_lshl_u32 v15, v20, s41, 1
	s_addc_u32 s19, s19, s21
	s_ashr_i32 s61, s60, 31
	v_or_b32_sdwa v14, s60, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_cndmask_b32_e32 v15, v21, v15, vcc
	s_movk_i32 s22, 0x1d64
	s_lshl_b64 s[20:21], s[60:61], 2
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	s_add_u32 s48, s16, s20
	v_cmp_gt_i32_e32 vcc, s22, v14
	v_mov_b32_e32 v10, 2
	s_addc_u32 s16, s19, s21
	v_lshlrev_b32_sdwa v10, v10, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	s_and_b64 vcc, s[0:1], vcc
	s_and_b32 s49, s16, 0xffff
	v_cndmask_b32_e32 v10, v21, v10, vcc
	v_mov_b32_e32 v11, 0x7f800000
	buffer_store_dword v11, v10, s[48:51], 0 offen
.LBB0_3:
	s_cmp_lt_i32 s69, 1
	s_cbranch_scc1 .LBB0_22
; %bb.4:
	s_and_b32 s16, s67, 63
	s_sub_i32 s19, 64, s67
	s_cmp_lt_i32 s67, 64
	s_mul_i32 s20, s12, s18
	s_cselect_b32 s16, s19, s16
	s_ashr_i32 s21, s20, 31
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s12, s2, s20
	s_mul_i32 s2, s13, s17
	s_addc_u32 s19, s3, s21
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s68, s14
	s_addc_u32 s13, s19, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s60
	s_addc_u32 s13, s13, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s19, s14, 5
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s20, s12, s2
	v_and_b32_e32 v34, 0x78, v200
	s_addc_u32 s12, s13, s3
	v_mad_u64_u32 v[10:11], s[2:3], s14, v1, v[34:35]
	v_add_u32_e32 v11, s19, v10
	v_lshlrev_b32_e32 v10, 1, v10
	v_bfrev_b32_e32 v35, 1
	v_cmp_gt_i32_e32 vcc, s33, v9
	v_add_u32_e32 v16, s19, v11
	s_and_b32 s2, s14, 0x3fff
	v_cndmask_b32_e32 v18, v35, v10, vcc
	v_lshlrev_b32_e32 v9, 1, v11
	v_cmp_gt_i32_e32 vcc, s33, v8
	v_add_u32_e32 v17, s19, v16
	s_bitset1_b32 s2, 14
	v_cndmask_b32_e32 v19, v35, v9, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	v_cmp_gt_i32_e32 vcc, s33, v6
	s_and_b32 s3, s12, 0xffff
	s_lshl_b32 s2, s2, 16
	v_cndmask_b32_e32 v6, v35, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v17
	v_cmp_gt_i32_e32 vcc, s33, v4
	v_add_u32_e32 v24, s19, v17
	s_or_b32 s21, s3, s2
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v4, v35, v16, vcc
	v_add_u32_e32 v25, s19, v24
	buffer_load_dwordx4 v[8:11], v18, s[20:23], 0 offen
	buffer_load_dwordx4 v[12:15], v19, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[16:19], v6, s[20:23], 0 offen
	buffer_load_dwordx4 v[20:23], v4, s[20:23], 0 offen
	v_lshlrev_b32_e32 v4, 1, v24
	v_cmp_gt_i32_e32 vcc, s33, v7
	v_add_u32_e32 v32, s19, v25
	v_lshlrev_b32_e32 v6, 1, v25
	v_cndmask_b32_e32 v4, v35, v4, vcc
	v_cmp_gt_i32_e32 vcc, s33, v5
	v_lshrrev_b32_e32 v33, 1, v0
	s_movk_i32 s2, 0x78
	v_cndmask_b32_e32 v5, v35, v6, vcc
	buffer_load_dwordx4 v[24:27], v4, s[20:23], 0 offen
	buffer_load_dwordx4 v[28:31], v5, s[20:23], 0 offen
	v_lshlrev_b32_e32 v4, 1, v32
	v_cmp_gt_i32_e32 vcc, s33, v3
	v_lshlrev_b32_e32 v223, 7, v1
	v_or_b32_e32 v5, 0x1000, v223
	v_cndmask_b32_e32 v3, v35, v4, vcc
	v_add_lshl_u32 v4, v32, s19, 1
	v_cmp_gt_i32_e32 vcc, s33, v2
	v_and_b32_e32 v254, 31, v0
	v_lshlrev_b32_e32 v6, 8, v1
	v_cndmask_b32_e32 v2, v35, v4, vcc
	buffer_load_dwordx4 v[36:39], v3, s[20:23], 0 offen
	buffer_load_dwordx4 v[40:43], v2, s[20:23], 0 offen
	v_lshrrev_b32_e32 v2, 3, v0
	v_and_b32_e32 v255, 4, v2
	v_bitop3_b32 v2, v33, v200, s2 bitop3:0x28
	s_and_b32 s2, s33, 0xff
	v_and_b32_e32 v3, 0x78, v33
	s_or_b32 s2, s16, s2
	v_bitop3_b32 v3, v3, v223, v34 bitop3:0xde
	s_cmp_eq_u32 s2, 0
	v_or_b32_e32 v7, v5, v2
	v_lshlrev_b32_e32 v4, 1, v3
	s_cselect_b32 s31, 4, 5
	v_lshlrev_b32_e32 v32, 1, v2
	v_lshlrev_b32_e32 v3, 1, v7
	v_add_u32_e32 v7, 0, v4
	s_cmp_le_u32 s69, s31
	v_add3_u32 v6, 0, v32, v6
	v_add_u32_e32 v32, 0, v3
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v7, v[8:11]
	s_waitcnt vmcnt(6)
	ds_write_b128 v32, v[12:15]
	s_waitcnt vmcnt(5)
	ds_write_b128 v6, v[16:19] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v6, v[20:23] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v6, v[24:27] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v6, v[28:31] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v6, v[36:39] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v6, v[40:43] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_11
; %bb.5:
	v_mad_u64_u32 v[198:199], s[2:3], s37, v1, v[34:35]
	v_lshrrev_b32_e32 v204, 2, v0
	v_and_b32_e32 v6, 8, v204
	s_ashr_i32 s2, s17, 31
	v_or_b32_e32 v7, 16, v6
	v_or_b32_e32 v8, 32, v6
	v_or_b32_e32 v9, 48, v6
	v_or_b32_e32 v10, 64, v6
	v_or_b32_e32 v11, 0x50, v6
	v_or_b32_e32 v12, 0x60, v6
	v_or_b32_e32 v6, 0x70, v6
	s_lshr_b32 s2, s2, 28
	s_add_i32 s17, s17, s2
	s_movk_i32 s2, 0xe0
	v_lshrrev_b32_e32 v6, 3, v6
	v_and_or_b32 v13, v33, s2, v254
	v_bitop3_b32 v6, v6, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v12, 3, v12
	v_lshl_add_u32 v13, v13, 8, 0
	v_lshlrev_b32_e32 v199, 4, v6
	v_bitop3_b32 v12, v12, v0, 15 bitop3:0x78
	v_add_u32_e32 v6, v13, v199
	v_lshlrev_b32_e32 v201, 4, v12
	v_add_u32_e32 v12, v13, v201
	ds_read_b128 v[130:133], v6
	ds_read_b128 v[134:137], v12
	v_lshrrev_b32_e32 v6, 3, v11
	v_bitop3_b32 v6, v6, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v10, 3, v10
	v_lshlrev_b32_e32 v206, 4, v6
	v_bitop3_b32 v10, v10, v0, 15 bitop3:0x78
	v_add_u32_e32 v6, v13, v206
	v_lshlrev_b32_e32 v207, 4, v10
	v_or_b32_e32 v227, v223, v34
	v_add_u32_e32 v10, v13, v207
	ds_read_b128 v[138:141], v6
	ds_read_b128 v[142:145], v10
	v_lshrrev_b32_e32 v6, 3, v9
	v_lshlrev_b32_e32 v209, 1, v227
	v_bitop3_b32 v6, v6, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v8, 3, v8
	v_or_b32_e32 v5, v5, v34
	v_sub_u32_e32 v4, v4, v209
	v_lshlrev_b32_e32 v208, 4, v6
	v_bitop3_b32 v8, v8, v0, 15 bitop3:0x78
	s_mul_i32 s2, s38, s18
	v_lshlrev_b32_e32 v210, 1, v5
	v_ashrrev_i16_e32 v5, 15, v4
	v_add_u32_e32 v6, v13, v208
	v_lshlrev_b32_e32 v211, 4, v8
	s_ashr_i32 s3, s2, 31
	v_lshrrev_b16_e32 v5, 12, v5
	s_ashr_i32 s12, s17, 4
	v_add_u32_e32 v8, v13, v211
	ds_read_b128 v[146:149], v6
	ds_read_b128 v[150:153], v8
	v_lshrrev_b32_e32 v6, 3, v7
	s_lshl_b64 s[2:3], s[2:3], 1
	v_add_u16_e32 v4, v4, v5
	v_and_b32_e32 v226, 63, v0
	v_and_b32_e32 v14, 15, v0
	v_bitop3_b32 v6, v6, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v7, 5, v0
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s39, s12
	v_ashrrev_i16_e32 v4, 4, v4
	v_lshlrev_b32_e32 v212, 4, v6
	v_bitop3_b32 v7, v7, v14, 1 bitop3:0x6c
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	v_add_u32_sdwa v4, v226, sext(v4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v6, v13, v212
	v_lshlrev_b32_e32 v213, 4, v7
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v5, 2, v4
	v_add_u32_e32 v7, v13, v213
	ds_read_b128 v[154:157], v6
	ds_read_b128 v[114:117], v7
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s66, s40
	ds_bpermute_b32 v6, v5, v198
	v_lshrrev_b64 v[4:5], v4, exec
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	v_and_b32_e32 v4, 1, v4
	v_sub_u32_e32 v3, v3, v210
	s_lshl_b64 s[2:3], s[2:3], 1
	v_cmp_eq_u32_e32 vcc, 1, v4
	v_ashrrev_i16_e32 v4, 15, v3
	s_add_u32 s16, s6, s2
	s_mul_i32 s6, s15, s18
	v_lshrrev_b16_e32 v4, 12, v4
	s_addc_u32 s17, s7, s3
	s_ashr_i32 s7, s6, 31
	v_add_u16_e32 v3, v3, v4
	s_lshl_b64 s[2:3], s[6:7], 1
	v_ashrrev_i16_e32 v3, 4, v3
	s_add_u32 s13, s4, s2
	s_mul_i32 s24, s36, s12
	v_add_u32_sdwa v3, v226, sext(v3) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshl_add_u32 v225, s37, 5, v198
	s_addc_u32 s14, s5, s3
	s_ashr_i32 s25, s24, 31
	v_lshlrev_b32_e32 v4, 2, v3
	s_lshl_b64 s[2:3], s[24:25], 1
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v5, 1, v6
	ds_bpermute_b32 v6, v4, v225
	s_add_u32 s12, s13, s2
	s_mul_i32 s26, s66, s37
	v_sub_u32_e32 v2, v2, v34
	s_addc_u32 s13, s14, s3
	s_ashr_i32 s27, s26, 31
	v_ashrrev_i32_e32 v2, 3, v2
	s_lshl_b64 s[2:3], s[26:27], 1
	v_add_u32_e32 v2, v2, v226
	s_add_u32 s20, s12, s2
	v_cndmask_b32_e32 v66, v35, v5, vcc
	v_lshrrev_b64 v[4:5], v3, exec
	v_lshlrev_b32_e32 v241, 2, v2
	s_addc_u32 s3, s13, s3
	s_and_b32 s12, s37, 0x3fff
	v_and_b32_e32 v3, 1, v4
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v4, 1, v6
	ds_bpermute_b32 v6, v241, v198
	v_add_u32_e32 v64, 0, v209
	s_bitset1_b32 s12, 14
	s_lshl_b32 s28, s37, 6
	v_add_u32_e32 v65, 0, v210
	s_and_b32 s13, s3, 0xffff
	s_lshl_b32 s30, s12, 16
	v_readfirstlane_b32 s34, v64
	s_or_b32 s21, s13, s30
	s_mov_b32 m0, s34
	v_cmp_eq_u32_e32 vcc, 1, v3
	v_readfirstlane_b32 s35, v65
	s_ashr_i32 s29, s28, 31
	s_lshl_b32 s2, s40, 6
	buffer_load_dwordx4 v66, s[20:23], 0 offen lds
	v_cndmask_b32_e32 v67, v35, v4, vcc
	s_mov_b32 m0, s35
	s_lshl_b64 s[14:15], s[28:29], 1
	v_lshrrev_b64 v[2:3], v2, exec
	buffer_load_dwordx4 v67, s[20:23], 0 offen lds
	s_add_u32 s20, s20, s14
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v6
	ds_bpermute_b32 v6, v241, v225
	s_addc_u32 s3, s3, s15
	v_add_u32_e32 v4, 0x4000, v64
	s_and_b32 s12, s3, 0xffff
	v_and_b32_e32 v2, 1, v2
	s_or_b32 s21, s12, s30
	v_cmp_eq_u32_e32 vcc, 1, v2
	v_readfirstlane_b32 s12, v4
	v_add_u32_e32 v5, 0x4000, v65
	v_cndmask_b32_e32 v2, v35, v3, vcc
	s_mov_b32 m0, s12
	v_lshlrev_b32_e32 v26, 8, v254
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v6
	v_readfirstlane_b32 s12, v5
	v_or_b32_e32 v68, v213, v26
	v_cndmask_b32_e32 v2, v35, v2, vcc
	s_mov_b32 m0, s12
	v_add_u32_e32 v69, 0, v68
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b128 v[2:5], v69
	ds_read_b128 v[18:21], v69 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[114:117], 0
	v_or_b32_e32 v70, v212, v26
	v_add_u32_e32 v71, 0, v70
	ds_read_b128 v[22:25], v71
	ds_read_b128 v[36:39], v71 offset:8192
	v_or_b32_e32 v72, v211, v26
	v_add_u32_e32 v73, 0, v72
	v_or_b32_e32 v74, v208, v26
	v_add_u32_e32 v75, 0, v74
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[154:157], v[2:17]
	ds_read_b128 v[22:25], v73
	ds_read_b128 v[40:43], v73 offset:8192
	v_or_b32_e32 v76, v207, v26
	v_add_u32_e32 v77, 0, v76
	v_or_b32_e32 v78, v206, v26
	v_add_u32_e32 v79, 0, v78
	v_or_b32_e32 v80, v201, v26
	v_add_u32_e32 v81, 0, v80
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[150:153], v[2:17]
	ds_read_b128 v[22:25], v75
	ds_read_b128 v[44:47], v75 offset:8192
	v_or_b32_e32 v82, v199, v26
	v_add_u32_e32 v83, 0, v82
	s_movk_i32 s12, 0x60
	s_and_b32 s18, s40, 0x3fff
	s_bitset1_b32 s18, 14
	s_and_b32 s19, s17, 0xffff
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[146:149], v[2:17]
	ds_read_b128 v[22:25], v77
	ds_read_b128 v[48:51], v77 offset:8192
	s_lshl_b32 s29, s18, 16
	s_or_b32 s37, s19, s29
	s_add_u32 s20, s20, s14
	s_mov_b32 s36, s16
	s_mov_b32 s38, s22
	s_mov_b32 s39, s23
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[142:145], v[2:17]
	ds_read_b128 v[22:25], v79
	ds_read_b128 v[52:55], v79 offset:8192
	s_addc_u32 s21, s3, s15
	s_ashr_i32 s3, s2, 31
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[138:141], v[2:17]
	ds_read_b128 v[22:25], v81
	ds_read_b128 v[56:59], v81 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[134:137], v[2:17]
	ds_read_b128 v[22:25], v83
	ds_read_b128 v[60:63], v83 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[130:133], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[114:117], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[36:39], v[154:157], v[18:33]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_nop 7
	s_nop 1
	v_max_f32_e32 v36, v3, v3
	v_max_f32_e32 v37, v2, v2
	v_max_f32_e32 v36, v37, v36
	v_max3_f32 v36, v36, v4, v5
	v_max3_f32 v36, v36, v6, v7
	v_max3_f32 v36, v36, v8, v9
	v_max3_f32 v36, v36, v10, v11
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[150:153], v[18:33]
	v_max3_f32 v36, v36, v12, v13
	v_max3_f32 v36, v36, v14, v15
	v_max3_f32 v36, v36, v16, v17
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[146:149], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[48:51], v[142:145], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[52:55], v[138:141], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[56:59], v[134:137], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[60:63], v[130:133], v[18:33]
	s_nop 7
	s_nop 3
	v_max3_f32 v36, v36, v18, v19
	v_max3_f32 v36, v36, v20, v21
	v_max3_f32 v36, v36, v22, v23
	v_max3_f32 v38, v36, v24, v25
	v_lshlrev_b32_e32 v36, 1, v0
	v_bitop3_b32 v254, v34, v36, s12 bitop3:0x78
	v_sub_u32_e32 v36, v254, v34
	v_ashrrev_i32_e32 v36, 3, v36
	v_add_u32_e32 v220, v36, v226
	v_mad_u64_u32 v[202:203], s[12:13], s40, v1, v[34:35]
	v_lshlrev_b32_e32 v222, 2, v220
	v_lshl_add_u32 v216, s40, 5, v202
	ds_bpermute_b32 v1, v222, v202
	ds_bpermute_b32 v39, v222, v216
	v_lshrrev_b64 v[36:37], v220, exec
	v_and_b32_e32 v36, 1, v36
	v_add_u32_e32 v37, 0x8000, v64
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v1, 1, v1
	v_cmp_eq_u32_e32 vcc, 1, v36
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v36, 1, v39
	v_readfirstlane_b32 s12, v37
	v_cndmask_b32_e32 v1, v35, v1, vcc
	v_cndmask_b32_e32 v35, v35, v36, vcc
	v_add_u32_e32 v36, 0x8000, v65
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s13, v36
	buffer_load_dwordx4 v1, s[36:39], 0 offen lds
	s_mov_b32 m0, s13
	s_lshl_b64 s[12:13], s[2:3], 1
	s_add_u32 s18, s16, s12
	s_addc_u32 s19, s17, s13
	s_and_b32 s2, s21, 0xffff
	buffer_load_dwordx4 v35, s[36:39], 0 offen lds
	s_or_b32 s21, s2, s30
	s_mov_b32 m0, s34
	v_add_u32_e32 v36, 0xc000, v64
	buffer_load_dwordx4 v66, s[20:23], 0 offen lds
	s_mov_b32 m0, s35
	v_readfirstlane_b32 s2, v36
	s_and_b32 s3, s19, 0xffff
	buffer_load_dwordx4 v67, s[20:23], 0 offen lds
	s_or_b32 s21, s3, s29
	s_mov_b32 s20, s18
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(4)
	s_barrier
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	v_add_u32_e32 v1, 0xc000, v65
	s_nop 0
	v_readfirstlane_b32 s2, v1
	s_mov_b32 m0, s2
	v_max3_f32 v1, v38, v26, v27
	buffer_load_dwordx4 v35, s[20:23], 0 offen lds
	v_max3_f32 v1, v1, v28, v29
	v_max3_f32 v1, v1, v30, v31
	v_max3_f32 v35, v1, v32, v33
	v_mov_b32_e32 v36, v35
	s_nop 1
	v_permlane32_swap_b32_e32 v35, v36
	v_mov_b32_e32 v1, 0xff800000
	v_max3_f32 v203, v35, v36, v1
	v_mul_f32_e32 v35, 0xbe0293ee, v203
	v_fmamk_f32 v2, v2, 0x3e0293ee, v35
	v_fmamk_f32 v3, v3, 0x3e0293ee, v35
	v_fmamk_f32 v4, v4, 0x3e0293ee, v35
	v_fmamk_f32 v5, v5, 0x3e0293ee, v35
	v_fmamk_f32 v6, v6, 0x3e0293ee, v35
	v_fmamk_f32 v7, v7, 0x3e0293ee, v35
	v_fmamk_f32 v8, v8, 0x3e0293ee, v35
	v_fmamk_f32 v9, v9, 0x3e0293ee, v35
	v_fmamk_f32 v10, v10, 0x3e0293ee, v35
	v_fmamk_f32 v11, v11, 0x3e0293ee, v35
	v_fmamk_f32 v12, v12, 0x3e0293ee, v35
	v_fmamk_f32 v13, v13, 0x3e0293ee, v35
	v_fmamk_f32 v14, v14, 0x3e0293ee, v35
	v_fmamk_f32 v15, v15, 0x3e0293ee, v35
	v_fmamk_f32 v16, v16, 0x3e0293ee, v35
	v_fmamk_f32 v17, v17, 0x3e0293ee, v35
	v_fmamk_f32 v18, v18, 0x3e0293ee, v35
	v_fmamk_f32 v19, v19, 0x3e0293ee, v35
	v_fmamk_f32 v20, v20, 0x3e0293ee, v35
	v_fmamk_f32 v21, v21, 0x3e0293ee, v35
	v_fmamk_f32 v22, v22, 0x3e0293ee, v35
	v_fmamk_f32 v23, v23, 0x3e0293ee, v35
	v_fmamk_f32 v24, v24, 0x3e0293ee, v35
	v_fmamk_f32 v25, v25, 0x3e0293ee, v35
	v_fmamk_f32 v26, v26, 0x3e0293ee, v35
	v_fmamk_f32 v27, v27, 0x3e0293ee, v35
	v_fmamk_f32 v28, v28, 0x3e0293ee, v35
	v_fmamk_f32 v29, v29, 0x3e0293ee, v35
	v_fmamk_f32 v30, v30, 0x3e0293ee, v35
	v_fmamk_f32 v31, v31, 0x3e0293ee, v35
	v_fmamk_f32 v32, v32, 0x3e0293ee, v35
	v_fmac_f32_e32 v35, 0x3e0293ee, v33
	v_mov_b32_e32 v33, s31
	v_sub_u32_e64 v33, s69, v33 clamp
	s_movk_i32 s2, 0x1ff
	v_readfirstlane_b32 s22, v33
	v_add_u32_e32 v33, 0xff, v0
	v_cmp_gt_u32_e32 vcc, s2, v33
	s_movk_i32 s2, 0x1fe
	s_add_i32 s20, 0, 0x4000
	v_cmp_lt_u32_e64 s[2:3], s2, v33
	v_add_u32_e32 v33, s20, v68
	v_add_u32_e32 v36, s20, v70
	v_add_u32_e32 v37, s20, v72
	v_add_u32_e32 v38, s20, v74
	v_add_u32_e32 v39, s20, v76
	v_add_u32_e32 v40, s20, v78
	v_add_u32_e32 v41, s20, v80
	v_add_u32_e32 v42, s20, v82
	ds_read_b128 v[66:69], v69 offset:16384
	ds_read_b128 v[186:189], v71 offset:16384
	ds_read_b128 v[182:185], v73 offset:16384
	ds_read_b128 v[178:181], v75 offset:16384
	ds_read_b128 v[174:177], v77 offset:16384
	ds_read_b128 v[110:113], v79 offset:16384
	ds_read_b128 v[106:109], v81 offset:16384
	ds_read_b128 v[102:105], v83 offset:16384
	ds_read_b128 v[98:101], v33 offset:8192
	ds_read_b128 v[170:173], v36 offset:8192
	ds_read_b128 v[166:169], v37 offset:8192
	ds_read_b128 v[162:165], v38 offset:8192
	ds_read_b128 v[158:161], v39 offset:8192
	ds_read_b128 v[126:129], v40 offset:8192
	ds_read_b128 v[122:125], v41 offset:8192
	ds_read_b128 v[118:121], v42 offset:8192
	v_fmac_f32_e32 v1, 0xbe0293ee, v203
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], s[2:3]
	s_cbranch_execz .LBB0_7
; %bb.6:
	s_barrier
.LBB0_7:
	s_or_b64 exec, exec, s[20:21]
	v_exp_f32_e32 v252, v2
	v_exp_f32_e32 v253, v3
	v_exp_f32_e32 v242, v4
	v_exp_f32_e32 v244, v5
	v_exp_f32_e32 v243, v6
	v_exp_f32_e32 v246, v7
	v_exp_f32_e32 v245, v8
	v_exp_f32_e32 v248, v9
	v_exp_f32_e32 v247, v10
	v_exp_f32_e32 v250, v11
	v_exp_f32_e32 v249, v12
	v_exp_f32_e32 v251, v13
	v_exp_f32_e32 v191, v14
	v_exp_f32_e32 v193, v15
	v_exp_f32_e32 v192, v16
	v_exp_f32_e32 v195, v17
	v_exp_f32_e32 v194, v18
	v_exp_f32_e32 v197, v19
	v_exp_f32_e32 v196, v20
	v_exp_f32_e32 v232, v21
	v_exp_f32_e32 v231, v22
	v_exp_f32_e32 v234, v23
	v_exp_f32_e32 v228, v24
	v_exp_f32_e32 v230, v25
	v_exp_f32_e32 v229, v26
	v_exp_f32_e32 v233, v27
	v_exp_f32_e32 v235, v28
	v_exp_f32_e32 v237, v29
	v_exp_f32_e32 v236, v30
	v_exp_f32_e32 v239, v31
	v_exp_f32_e32 v238, v32
	v_exp_f32_e32 v240, v35
	v_exp_f32_e32 v190, v1
	v_and_b32_e32 v36, 31, v0
	v_mov_b32_e32 v33, 0
	v_lshlrev_b32_e32 v217, 7, v36
	v_and_b32_e32 v221, 32, v200
	v_and_b32_e32 v219, 64, v200
	v_and_b32_e32 v218, 16, v0
	s_cmp_lt_u32 s22, 4
	v_lshlrev_b32_e32 v214, 2, v0
	s_cbranch_scc1 .LBB0_12
; %bb.8:                                ; %.lr.ph
	s_add_i32 s2, s22, -3
	s_add_u32 s6, s6, s24
	s_addc_u32 s7, s7, s25
	s_add_u32 s6, s6, s26
	s_addc_u32 s7, s7, s27
	v_mov_b32_e32 v2, v255
	v_and_b32_e32 v255, 12, v214
	s_movk_i32 s3, 0x60
	s_mul_i32 s16, s28, 6
	s_lshl_b64 s[6:7], s[6:7], 1
	v_bitop3_b32 v224, v200, v255, s3 bitop3:0x4e
	s_mul_hi_i32 s3, s28, 6
	s_add_u32 s6, s16, s6
	scratch_store_dword off, v222, off offset:8 ; 4-byte Folded Spill
	scratch_store_dword off, v220, off offset:4 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off        ; 4-byte Folded Spill
	v_and_or_b32 v2, v204, 3, v2
	s_addc_u32 s7, s3, s7
	v_or_b32_e32 v1, v221, v255
	v_lshlrev_b32_e32 v2, 7, v2
	v_bitop3_b32 v3, v255, v221, 32 bitop3:0x36
	s_add_u32 s3, s4, s6
	v_mov_b32_e32 v18, 0
	v_bitop3_b32 v1, v1, v219, 64 bitop3:0x36
	v_add_u32_e32 v215, v223, v34
	s_addc_u32 s24, s5, s7
	s_mov_b32 s22, 0
	s_add_i32 s20, 0, 0x8000
	s_add_i32 s21, 0, 0xc000
	v_mov_b32_e32 v205, 1.0
	v_lshlrev_b32_e32 v214, 1, v2
	v_lshlrev_b32_e32 v204, 1, v3
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	s_mov_b32 s25, 0x3e0293ee
	s_mov_b32 s27, 0
	s_mov_b32 s26, 0
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
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[114:117], 0
	s_mov_b64 s[16:17], s[18:19]
	s_mov_b32 s34, s20
	s_mov_b32 s20, s21
	v_mov_b32_e32 v222, v205
	v_mov_b32_e32 v220, v203
	s_mov_b32 s28, s22
	s_setprio 0
	v_add_f32_e32 v82, v252, v253
	v_add_f32_e32 v82, v82, v242
	v_add_f32_e32 v82, v82, v244
	v_add_f32_e32 v82, v82, v243
	v_add_f32_e32 v82, v82, v246
	v_add_f32_e32 v82, v82, v245
	v_add_f32_e32 v82, v82, v248
	v_add_f32_e32 v82, v82, v247
	v_add_f32_e32 v82, v82, v250
	v_add_f32_e32 v82, v82, v249
	v_add_f32_e32 v82, v82, v251
	v_add_f32_e32 v82, v82, v191
	v_add_f32_e32 v82, v82, v193
	v_add_f32_e32 v82, v82, v192
	v_add_f32_e32 v82, v82, v195
	v_add_f32_e32 v82, v82, v194
	v_add_f32_e32 v82, v82, v197
	v_add_f32_e32 v82, v82, v196
	v_add_f32_e32 v82, v82, v232
	v_add_f32_e32 v82, v82, v231
	v_add_f32_e32 v82, v82, v234
	v_add_f32_e32 v82, v82, v228
	v_add_f32_e32 v82, v82, v230
	v_add_f32_e32 v82, v82, v229
	v_add_f32_e32 v82, v82, v233
	v_add_f32_e32 v82, v82, v235
	v_add_f32_e32 v82, v82, v237
	v_add_f32_e32 v82, v82, v236
	v_add_f32_e32 v82, v82, v239
	v_add_f32_e32 v82, v82, v238
	v_add_f32_e32 v82, v82, v240
	v_mov_b32_e32 v83, v82
	s_nop 1
	v_permlane32_swap_b32_e32 v82, v83
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[154:157], v[66:81]
	v_add_f32_e32 v205, v82, v83
	v_mul_f32_e32 v18, v18, v190
	v_mul_f32_e32 v19, v19, v190
	v_mul_f32_e32 v20, v20, v190
	v_mul_f32_e32 v21, v21, v190
	v_mul_f32_e32 v22, v22, v190
	v_mul_f32_e32 v23, v23, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[114:117], 0
	v_mul_f32_e32 v24, v24, v190
	v_mul_f32_e32 v25, v25, v190
	v_mul_f32_e32 v26, v26, v190
	v_mul_f32_e32 v27, v27, v190
	v_mul_f32_e32 v28, v28, v190
	v_mul_f32_e32 v29, v29, v190
	v_mul_f32_e32 v30, v30, v190
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[150:153], v[66:81]
	v_mul_f32_e32 v31, v31, v190
	v_mul_f32_e32 v32, v32, v190
	v_mul_f32_e32 v33, v33, v190
	v_mul_f32_e32 v50, v50, v190
	v_mul_f32_e32 v51, v51, v190
	v_mul_f32_e32 v52, v52, v190
	v_mul_f32_e32 v53, v53, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[154:157], v[82:97]
	v_mul_f32_e32 v54, v54, v190
	v_mul_f32_e32 v55, v55, v190
	v_mul_f32_e32 v56, v56, v190
	v_mul_f32_e32 v57, v57, v190
	v_mul_f32_e32 v58, v58, v190
	v_mul_f32_e32 v59, v59, v190
	v_mul_f32_e32 v60, v60, v190
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[146:149], v[66:81]
	v_mul_f32_e32 v61, v61, v190
	v_mul_f32_e32 v62, v62, v190
	v_mul_f32_e32 v63, v63, v190
	v_mul_f32_e32 v64, v64, v190
	v_mul_f32_e32 v65, v65, v190
	v_mul_f32_e32 v34, v34, v190
	v_mul_f32_e32 v35, v35, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[150:153], v[82:97]
	v_mul_f32_e32 v36, v36, v190
	v_mul_f32_e32 v37, v37, v190
	v_mul_f32_e32 v38, v38, v190
	v_mul_f32_e32 v39, v39, v190
	v_mul_f32_e32 v40, v40, v190
	v_mul_f32_e32 v41, v41, v190
	v_mul_f32_e32 v42, v42, v190
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[142:145], v[66:81]
	v_mul_f32_e32 v43, v43, v190
	v_mul_f32_e32 v44, v44, v190
	v_mul_f32_e32 v45, v45, v190
	v_mul_f32_e32 v46, v46, v190
	v_mul_f32_e32 v47, v47, v190
	v_mul_f32_e32 v48, v48, v190
	v_mul_f32_e32 v49, v49, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[146:149], v[82:97]
	v_mul_f32_e32 v2, v2, v190
	v_mul_f32_e32 v3, v3, v190
	v_mul_f32_e32 v4, v4, v190
	v_mul_f32_e32 v5, v5, v190
	v_mul_f32_e32 v6, v6, v190
	v_mul_f32_e32 v7, v7, v190
	v_mul_f32_e32 v8, v8, v190
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[138:141], v[66:81]
	v_mul_f32_e32 v9, v9, v190
	v_mul_f32_e32 v10, v10, v190
	v_mul_f32_e32 v11, v11, v190
	v_mul_f32_e32 v12, v12, v190
	v_mul_f32_e32 v13, v13, v190
	v_mul_f32_e32 v14, v14, v190
	v_mul_f32_e32 v15, v15, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[158:161], v[142:145], v[82:97]
	v_mul_f32_e32 v16, v16, v190
	v_mul_f32_e32 v17, v17, v190
	v_fmac_f32_e32 v205, v222, v190
	v_cvt_pk_f16_f32 v110, v252, v253
	v_cvt_pk_f16_f32 v111, v242, v244
	v_cvt_pk_f16_f32 v112, v243, v246
	v_cvt_pk_f16_f32 v113, v245, v248
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[134:137], v[66:81]
	v_cvt_pk_f16_f32 v106, v247, v250
	v_cvt_pk_f16_f32 v107, v249, v251
	v_cvt_pk_f16_f32 v108, v191, v193
	v_cvt_pk_f16_f32 v109, v192, v195
	v_cvt_pk_f16_f32 v98, v229, v233
	v_cvt_pk_f16_f32 v99, v235, v237
	v_cvt_pk_f16_f32 v100, v236, v239
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[138:141], v[82:97]
	v_cvt_pk_f16_f32 v101, v238, v240
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[130:133], v[66:81]
	v_cvt_pk_f16_f32 v102, v194, v197
	v_cvt_pk_f16_f32 v103, v196, v232
	v_cvt_pk_f16_f32 v104, v231, v234
	v_cvt_pk_f16_f32 v105, v228, v230
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[134:137], v[82:97]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[130:133], v[82:97]
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s18, s16, s12
	s_addc_u32 s19, s17, s13
	s_add_i32 s4, s27, 1
	s_cmp_lt_i32 s4, 2
	s_cselect_b32 s31, s4, 0
	ds_bpermute_b32 v120, v241, v198
	s_lshl_b32 s23, s31, 14
	ds_bpermute_b32 v121, v241, v225
	s_add_i32 s22, s23, 0
	v_add_u32_e32 v118, s22, v209
	v_add_u32_e32 v119, s22, v210
	s_and_b32 s4, s24, 0xffff
	v_readfirstlane_b32 s21, v118
	s_or_b32 s5, s4, s30
	s_mov_b32 s4, s3
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v120, 1, v120
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v119
	buffer_load_dwordx4 v120, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v118, 1, v121
	s_mov_b32 m0, s21
	v_lshlrev_b32_e32 v119, 1, v221
	buffer_load_dwordx4 v118, s[4:7], 0 offen lds
	v_lshl_add_u32 v118, v255, 1, s34
	v_lshlrev_b32_e32 v120, 1, v219
	v_add3_u32 v118, v118, v119, v120
	v_lshlrev_b32_e32 v119, 1, v218
	v_add3_u32 v118, v118, v119, v214
	ds_read_b64_tr_b16 v[228:229], v118
	ds_read_b64_tr_b16 v[230:231], v118 offset:2048
	ds_read_b64_tr_b16 v[232:233], v118 offset:4096
	ds_read_b64_tr_b16 v[234:235], v118 offset:6144
	ds_read_b64_tr_b16 v[236:237], v118 offset:8192
	ds_read_b64_tr_b16 v[238:239], v118 offset:10240
	ds_read_b64_tr_b16 v[194:195], v118 offset:12288
	ds_read_b64_tr_b16 v[196:197], v118 offset:14336
	v_add3_u32 v118, s34, v204, v120
	v_add3_u32 v118, v118, v119, v214
	ds_read_b64_tr_b16 v[190:191], v118
	ds_read_b64_tr_b16 v[192:193], v118 offset:2048
	ds_read_b64_tr_b16 v[186:187], v118 offset:4096
	ds_read_b64_tr_b16 v[188:189], v118 offset:6144
	ds_read_b64_tr_b16 v[182:183], v118 offset:8192
	ds_read_b64_tr_b16 v[184:185], v118 offset:10240
	ds_read_b64_tr_b16 v[178:179], v118 offset:12288
	ds_read_b64_tr_b16 v[180:181], v118 offset:14336
	v_lshl_add_u32 v118, v1, 1, s34
	v_add3_u32 v118, v118, v119, v214
	ds_read_b64_tr_b16 v[174:175], v118
	ds_read_b64_tr_b16 v[176:177], v118 offset:2048
	ds_read_b64_tr_b16 v[170:171], v118 offset:4096
	ds_read_b64_tr_b16 v[172:173], v118 offset:6144
	ds_read_b64_tr_b16 v[166:167], v118 offset:8192
	ds_read_b64_tr_b16 v[168:169], v118 offset:10240
	ds_read_b64_tr_b16 v[162:163], v118 offset:12288
	ds_read_b64_tr_b16 v[164:165], v118 offset:14336
	v_lshl_add_u32 v118, v224, 1, s34
	v_add3_u32 v120, v118, v119, v214
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
	v_mfma_f32_32x32x16_f16 v[18:33], v[228:231], v[110:113], v[18:33]
	s_barrier
	s_setprio 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[232:235], v[106:109], v[18:33]
	v_max_f32_e32 v203, v67, v67
	v_max_f32_e32 v222, v66, v66
	v_max_f32_e32 v203, v222, v203
	v_max3_f32 v203, v203, v68, v69
	v_mfma_f32_32x32x16_f16 v[18:33], v[236:239], v[102:105], v[18:33]
	v_mfma_f32_32x32x16_f16 v[50:65], v[190:193], v[110:113], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[174:177], v[110:113], v[34:49]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[110:113], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[194:197], v[98:101], v[18:33]
	v_max3_f32 v194, v203, v70, v71
	v_max3_f32 v194, v194, v72, v73
	v_max3_f32 v194, v194, v74, v75
	v_max3_f32 v194, v194, v76, v77
	v_max3_f32 v194, v194, v78, v79
	v_max3_f32 v194, v194, v80, v81
	v_max3_f32 v194, v194, v82, v83
	v_mfma_f32_32x32x16_f16 v[50:65], v[186:189], v[106:109], v[50:65]
	v_max3_f32 v190, v194, v84, v85
	v_max3_f32 v190, v190, v86, v87
	v_max3_f32 v190, v190, v88, v89
	v_max3_f32 v190, v190, v90, v91
	v_max3_f32 v190, v190, v92, v93
	v_max3_f32 v190, v190, v94, v95
	v_max3_f32 v190, v190, v96, v97
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[106:109], v[34:49]
	v_mov_b32_e32 v186, v190
	s_nop 1
	v_permlane32_swap_b32_e32 v190, v186
	v_max3_f32 v203, v220, v190, v186
	v_mul_f32_e32 v186, 0x3e0293ee, v203
	v_fma_f32 v66, v66, s25, -v186
	v_fma_f32 v67, v67, s25, -v186
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[126:129], v[106:109], v[2:17]
	v_fma_f32 v68, v68, s25, -v186
	v_fma_f32 v69, v69, s25, -v186
	v_fma_f32 v70, v70, s25, -v186
	v_fma_f32 v71, v71, s25, -v186
	v_fma_f32 v72, v72, s25, -v186
	v_fma_f32 v73, v73, s25, -v186
	v_fma_f32 v74, v74, s25, -v186
	v_mfma_f32_32x32x16_f16 v[50:65], v[182:185], v[102:105], v[50:65]
	v_fma_f32 v75, v75, s25, -v186
	v_fma_f32 v76, v76, s25, -v186
	v_fma_f32 v77, v77, s25, -v186
	v_fma_f32 v78, v78, s25, -v186
	v_fma_f32 v79, v79, s25, -v186
	v_fma_f32 v80, v80, s25, -v186
	v_fma_f32 v81, v81, s25, -v186
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[102:105], v[34:49]
	v_fma_f32 v82, v82, s25, -v186
	v_fma_f32 v83, v83, s25, -v186
	v_fma_f32 v84, v84, s25, -v186
	v_fma_f32 v85, v85, s25, -v186
	v_fma_f32 v86, v86, s25, -v186
	v_fma_f32 v87, v87, s25, -v186
	v_fma_f32 v88, v88, s25, -v186
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[122:125], v[102:105], v[2:17]
	v_fma_f32 v89, v89, s25, -v186
	v_fma_f32 v90, v90, s25, -v186
	v_fma_f32 v91, v91, s25, -v186
	v_fma_f32 v92, v92, s25, -v186
	v_fma_f32 v93, v93, s25, -v186
	v_fma_f32 v94, v94, s25, -v186
	v_fma_f32 v95, v95, s25, -v186
	v_mfma_f32_32x32x16_f16 v[50:65], v[178:181], v[98:101], v[50:65]
	v_fma_f32 v96, v96, s25, -v186
	v_fma_f32 v97, v97, s25, -v186
	v_exp_f32_e32 v252, v66
	v_fma_f32 v66, v220, s25, -v186
	v_exp_f32_e32 v253, v67
	v_exp_f32_e32 v242, v68
	v_exp_f32_e32 v244, v69
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[98:101], v[34:49]
	v_exp_f32_e32 v243, v70
	v_exp_f32_e32 v246, v71
	v_exp_f32_e32 v245, v72
	v_exp_f32_e32 v248, v73
	v_exp_f32_e32 v247, v74
	v_exp_f32_e32 v250, v75
	v_exp_f32_e32 v249, v76
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[118:121], v[98:101], v[2:17]
	v_exp_f32_e32 v251, v77
	v_exp_f32_e32 v191, v78
	v_exp_f32_e32 v193, v79
	v_exp_f32_e32 v192, v80
	v_exp_f32_e32 v195, v81
	v_exp_f32_e32 v194, v82
	v_exp_f32_e32 v197, v83
	v_exp_f32_e32 v196, v84
	v_exp_f32_e32 v232, v85
	v_exp_f32_e32 v231, v86
	v_exp_f32_e32 v234, v87
	v_exp_f32_e32 v228, v88
	v_exp_f32_e32 v230, v89
	v_exp_f32_e32 v229, v90
	v_exp_f32_e32 v233, v91
	v_exp_f32_e32 v235, v92
	v_exp_f32_e32 v237, v93
	v_exp_f32_e32 v236, v94
	v_exp_f32_e32 v239, v95
	v_exp_f32_e32 v238, v96
	v_exp_f32_e32 v240, v97
	v_exp_f32_e32 v190, v66
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s4, s27, 14
	s_add_i32 s4, s4, 0
	s_add_i32 s21, s4, 0x8000
	v_lshlrev_b32_e32 v66, 1, v254
	v_lshlrev_b32_e32 v67, 1, v223
	v_add3_u32 v66, s4, v66, v67
	v_lshl_add_u32 v67, v227, 1, s21
	v_lshl_add_u32 v68, v215, 1, s21
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
	v_add_lshl_u32 v69, v69, v226, 2
	v_ashrrev_i32_e32 v66, 4, v66
	ds_bpermute_b32 v69, v69, v202
	v_add_lshl_u32 v66, v66, v226, 2
	ds_bpermute_b32 v66, v66, v216
	s_and_b32 s4, s19, 0xffff
	v_readfirstlane_b32 s27, v67
	s_or_b32 s5, s4, s29
	s_mov_b32 s4, s18
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v69, 1, v69
	s_mov_b32 m0, s27
	v_readfirstlane_b32 s27, v68
	buffer_load_dwordx4 v69, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v66, s[4:7], 0 offen lds
	v_lshlrev_b32_e32 v66, 1, v217
	v_add3_u32 v70, s28, v213, v66
	v_add3_u32 v71, s28, v212, v66
	v_add3_u32 v72, s28, v211, v66
	v_add3_u32 v73, s28, v208, v66
	v_add3_u32 v74, s28, v207, v66
	v_add3_u32 v75, s28, v206, v66
	v_add3_u32 v76, s28, v201, v66
	v_add3_u32 v77, s28, v199, v66
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[98:101], v70 offset:8192
	ds_read_b128 v[186:189], v71
	ds_read_b128 v[170:173], v71 offset:8192
	ds_read_b128 v[182:185], v72
	ds_read_b128 v[166:169], v72 offset:8192
	ds_read_b128 v[178:181], v73
	ds_read_b128 v[162:165], v73 offset:8192
	ds_read_b128 v[174:177], v74
	ds_read_b128 v[158:161], v74 offset:8192
	ds_read_b128 v[110:113], v75
	ds_read_b128 v[126:129], v75 offset:8192
	ds_read_b128 v[106:109], v76
	ds_read_b128 v[122:125], v76 offset:8192
	ds_read_b128 v[102:105], v77
	ds_read_b128 v[118:121], v77 offset:8192
	; sched_barrier mask(0x00000000)
	s_add_i32 s26, s26, 1
	s_add_u32 s3, s3, s14
	s_addc_u32 s24, s24, s15
	s_cmp_lt_i32 s26, s2
	s_mov_b32 s27, s31
	s_barrier
	s_cbranch_scc1 .LBB0_9
; %bb.10:                               ; %Flow542
	scratch_load_dword v255, off, off       ; 4-byte Folded Reload
	scratch_load_dword v220, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v222, off, off offset:8 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v204, 2, v0
	v_lshlrev_b32_e32 v214, 2, v0
	v_and_b32_e32 v254, 31, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execnz .LBB0_13
	s_branch .LBB0_14
.LBB0_11:
	v_mov_b32_e32 v3, 0
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
	v_mov_b32_e32 v1, 0xff800000
	v_mov_b32_e32 v66, 1.0
	s_branch .LBB0_15
.LBB0_12:
	s_mov_b32 s22, 0
	s_add_i32 s20, 0, 0x8000
	s_add_i32 s21, 0, 0xc000
	v_mov_b32_e32 v205, 1.0
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
	v_mov_b32_e32 v49, 0
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
	v_and_b32_e32 v254, 31, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_14
.LBB0_13:
	s_barrier
.LBB0_14:
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[114:117], 0
	v_add_f32_e32 v1, v252, v253
	v_add_f32_e32 v1, v1, v242
	v_add_f32_e32 v1, v1, v244
	v_add_f32_e32 v1, v1, v243
	v_add_f32_e32 v1, v1, v246
	v_add_f32_e32 v1, v1, v245
	v_add_f32_e32 v1, v1, v248
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[154:157], v[82:97]
	v_add_f32_e32 v1, v1, v247
	v_add_f32_e32 v1, v1, v250
	v_add_f32_e32 v1, v1, v249
	v_add_f32_e32 v1, v1, v251
	v_add_f32_e32 v1, v1, v191
	v_add_f32_e32 v1, v1, v193
	v_add_f32_e32 v1, v1, v192
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[150:153], v[82:97]
	v_add_f32_e32 v1, v1, v195
	v_add_f32_e32 v1, v1, v194
	v_add_f32_e32 v1, v1, v197
	v_add_f32_e32 v1, v1, v196
	v_add_f32_e32 v1, v1, v232
	v_add_f32_e32 v1, v1, v231
	v_add_f32_e32 v1, v1, v234
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[146:149], v[82:97]
	v_add_f32_e32 v1, v1, v228
	v_add_f32_e32 v1, v1, v230
	v_add_f32_e32 v1, v1, v229
	v_add_f32_e32 v1, v1, v233
	v_add_f32_e32 v1, v1, v235
	v_add_f32_e32 v1, v1, v237
	v_add_f32_e32 v1, v1, v236
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[142:145], v[82:97]
	v_add_f32_e32 v1, v1, v239
	v_add_f32_e32 v1, v1, v238
	v_add_f32_e32 v1, v1, v240
	v_mov_b32_e32 v72, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v72
	v_mul_f32_e32 v67, v19, v190
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[138:141], v[82:97]
	v_add_f32_e32 v223, v1, v72
	v_mul_f32_e32 v19, v35, v190
	v_and_b32_e32 v1, 12, v214
	s_waitcnt vmcnt(2)
	v_and_or_b32 v35, v204, 3, v255
	v_mul_f32_e32 v81, v33, v190
	v_or_b32_e32 v33, v219, v218
	v_mul_f32_e32 v72, v24, v190
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[134:137], v[82:97]
	v_mul_f32_e32 v73, v25, v190
	v_mul_f32_e32 v74, v26, v190
	v_mul_f32_e32 v75, v27, v190
	v_mul_f32_e32 v24, v40, v190
	v_mul_f32_e32 v25, v41, v190
	v_mul_f32_e32 v26, v42, v190
	v_mul_f32_e32 v27, v43, v190
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[130:133], v[82:97]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mul_f32_e32 v66, v18, v190
	v_mul_f32_e32 v68, v20, v190
	v_mul_f32_e32 v69, v21, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[98:101], v[114:117], 0
	v_mul_f32_e32 v70, v22, v190
	v_mul_f32_e32 v71, v23, v190
	v_mul_f32_e32 v76, v28, v190
	v_mul_f32_e32 v77, v29, v190
	v_mul_f32_e32 v78, v30, v190
	v_mul_f32_e32 v79, v31, v190
	v_mul_f32_e32 v80, v32, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[154:157], v[98:113]
	v_mul_f32_e32 v18, v34, v190
	v_mul_f32_e32 v20, v36, v190
	v_mul_f32_e32 v21, v37, v190
	v_cvt_pk_f16_f32 v34, v252, v253
	v_cvt_pk_f16_f32 v36, v243, v246
	v_cvt_pk_f16_f32 v37, v245, v248
	v_mul_f32_e32 v28, v44, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[150:153], v[98:113]
	v_mul_f32_e32 v29, v45, v190
	v_mul_f32_e32 v30, v46, v190
	v_mul_f32_e32 v31, v47, v190
	v_mul_f32_e32 v22, v38, v190
	v_mul_f32_e32 v23, v39, v190
	v_cvt_pk_f16_f32 v38, v247, v250
	v_cvt_pk_f16_f32 v39, v249, v251
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[146:149], v[98:113]
	v_lshlrev_b32_e32 v162, 7, v35
	v_cvt_pk_f16_f32 v35, v242, v244
	v_mul_f32_e32 v50, v50, v190
	v_mul_f32_e32 v51, v51, v190
	v_mul_f32_e32 v52, v52, v190
	v_mul_f32_e32 v53, v53, v190
	v_mul_f32_e32 v54, v54, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[158:161], v[142:145], v[98:113]
	v_or_b32_e32 v158, v1, v221
	v_or3_b32 v33, v33, v158, v162
	v_lshlrev_b32_e32 v215, 1, v33
	v_add_u32_e32 v33, s20, v215
	ds_read_b64_tr_b16 v[40:41], v33
	ds_read_b64_tr_b16 v[42:43], v33 offset:2048
	ds_read_b64_tr_b16 v[44:45], v33 offset:4096
	ds_read_b64_tr_b16 v[46:47], v33 offset:6144
	v_mul_f32_e32 v55, v55, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[126:129], v[138:141], v[98:113]
	v_mul_f32_e32 v56, v56, v190
	v_mul_f32_e32 v57, v57, v190
	v_mul_f32_e32 v58, v58, v190
	v_mul_f32_e32 v59, v59, v190
	v_mul_f32_e32 v60, v60, v190
	v_mul_f32_e32 v61, v61, v190
	v_mul_f32_e32 v62, v62, v190
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[66:81], v[40:43], v[34:37], v[66:81]
	v_cvt_pk_f16_f32 v40, v191, v193
	v_cvt_pk_f16_f32 v41, v192, v195
	v_mul_f32_e32 v63, v63, v190
	v_mul_f32_e32 v64, v64, v190
	v_mul_f32_e32 v65, v65, v190
	v_mul_f32_e32 v32, v48, v190
	v_cvt_pk_f16_f32 v42, v229, v233
	v_mfma_f32_32x32x16_f16 v[98:113], v[122:125], v[134:137], v[98:113]
	ds_read_b64_tr_b16 v[122:123], v33 offset:8192
	ds_read_b64_tr_b16 v[124:125], v33 offset:10240
	ds_read_b64_tr_b16 v[126:127], v33 offset:12288
	ds_read_b64_tr_b16 v[128:129], v33 offset:14336
	v_or_b32_e32 v33, 32, v1
	v_bitop3_b32 v33, v218, v33, v221 bitop3:0xf6
	v_or3_b32 v33, v33, v219, v162
	v_lshlrev_b32_e32 v221, 1, v33
	v_add_u32_e32 v159, s20, v221
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[66:81], v[44:47], v[38:41], v[66:81]
	v_mul_f32_e32 v33, v49, v190
	v_cvt_pk_f16_f32 v43, v235, v237
	v_cvt_pk_f16_f32 v44, v236, v239
	v_cvt_pk_f16_f32 v45, v238, v240
	s_movk_i32 s4, 0x60
	v_bitop3_b32 v1, v200, v1, s4 bitop3:0x4e
	v_or3_b32 v1, v1, v218, v162
	v_mfma_f32_32x32x16_f16 v[98:113], v[118:121], v[130:133], v[98:113]
	v_cvt_pk_f16_f32 v118, v194, v197
	v_cvt_pk_f16_f32 v119, v196, v232
	v_cvt_pk_f16_f32 v120, v231, v234
	v_cvt_pk_f16_f32 v121, v228, v230
	v_lshlrev_b32_e32 v204, 1, v1
	v_add_u32_e32 v1, s20, v204
	v_mul_f32_e32 v2, v2, v190
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[66:81], v[122:125], v[118:121], v[66:81]
	ds_read_b64_tr_b16 v[122:123], v159
	ds_read_b64_tr_b16 v[124:125], v159 offset:2048
	ds_read_b64_tr_b16 v[46:47], v159 offset:4096
	ds_read_b64_tr_b16 v[48:49], v159 offset:6144
	v_mul_f32_e32 v3, v3, v190
	v_mul_f32_e32 v4, v4, v190
	v_mul_f32_e32 v5, v5, v190
	v_mul_f32_e32 v6, v6, v190
	v_mul_f32_e32 v7, v7, v190
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[122:125], v[34:37], v[50:65]
	ds_read_b64_tr_b16 v[122:123], v159 offset:8192
	ds_read_b64_tr_b16 v[124:125], v159 offset:10240
	v_mul_f32_e32 v8, v8, v190
	v_mul_f32_e32 v9, v9, v190
	v_mul_f32_e32 v10, v10, v190
	v_mul_f32_e32 v11, v11, v190
	v_mul_f32_e32 v12, v12, v190
	v_mul_f32_e32 v13, v13, v190
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[46:49], v[38:41], v[50:65]
	ds_read_b64_tr_b16 v[46:47], v159 offset:12288
	ds_read_b64_tr_b16 v[48:49], v159 offset:14336
	v_mul_f32_e32 v14, v14, v190
	v_mul_f32_e32 v15, v15, v190
	v_mul_f32_e32 v16, v16, v190
	v_mul_f32_e32 v17, v17, v190
	s_add_u32 s2, s16, s12
	s_addc_u32 s3, s17, s13
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[122:125], v[118:121], v[50:65]
	v_bitop3_b32 v122, v158, v219, 64 bitop3:0x36
	v_or3_b32 v122, v122, v218, v162
	v_lshlrev_b32_e32 v214, 1, v122
	v_add_u32_e32 v160, s20, v214
	s_add_u32 s4, s2, s12
	s_addc_u32 s5, s3, s13
	s_add_i32 s2, s23, 0
	v_mfma_f32_32x32x16_f16 v[66:81], v[126:129], v[42:45], v[66:81]
	s_add_i32 s3, s2, 0x8000
	s_and_b32 s5, s5, 0xffff
	s_or_b32 s5, s5, s29
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_fmac_f32_e32 v223, v205, v190
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[46:49], v[42:45], v[50:65]
	ds_read_b64_tr_b16 v[46:47], v160
	ds_read_b64_tr_b16 v[48:49], v160 offset:2048
	ds_read_b64_tr_b16 v[122:123], v160 offset:4096
	ds_read_b64_tr_b16 v[124:125], v160 offset:6144
	ds_read_b64_tr_b16 v[126:127], v160 offset:8192
	ds_read_b64_tr_b16 v[128:129], v160 offset:10240
	ds_read_b64_tr_b16 v[158:159], v160 offset:12288
	ds_read_b64_tr_b16 v[160:161], v160 offset:14336
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[46:49], v[34:37], v[18:33]
	ds_read_b64_tr_b16 v[46:47], v1
	ds_read_b64_tr_b16 v[48:49], v1 offset:2048
	ds_read_b64_tr_b16 v[162:163], v1 offset:4096
	ds_read_b64_tr_b16 v[164:165], v1 offset:6144
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[18:33], v[122:125], v[38:41], v[18:33]
	ds_read_b64_tr_b16 v[122:123], v1 offset:8192
	ds_read_b64_tr_b16 v[124:125], v1 offset:10240
	ds_read_b64_tr_b16 v[166:167], v1 offset:12288
	ds_read_b64_tr_b16 v[168:169], v1 offset:14336
	v_max_f32_e32 v1, v83, v83
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[18:33], v[126:129], v[118:121], v[18:33]
	v_max_f32_e32 v126, v82, v82
	v_max_f32_e32 v1, v126, v1
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_mfma_f32_32x32x16_f16 v[2:17], v[46:49], v[34:37], v[2:17]
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_max3_f32 v1, v1, v98, v99
	v_max3_f32 v1, v1, v100, v101
	v_max3_f32 v1, v1, v102, v103
	v_max3_f32 v1, v1, v104, v105
	v_max3_f32 v1, v1, v106, v107
	v_max3_f32 v1, v1, v108, v109
	v_mfma_f32_32x32x16_f16 v[2:17], v[162:165], v[38:41], v[2:17]
	v_max3_f32 v1, v1, v110, v111
	v_max3_f32 v1, v1, v112, v113
	v_mov_b32_e32 v34, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v34
	v_max3_f32 v218, v203, v1, v34
	v_lshlrev_b32_e32 v1, 1, v217
	v_add3_u32 v38, s22, v213, v1
	ds_read_b128 v[34:37], v38
	v_mfma_f32_32x32x16_f16 v[2:17], v[122:125], v[118:121], v[2:17]
	v_add3_u32 v39, s22, v212, v1
	v_add3_u32 v40, s22, v211, v1
	v_add3_u32 v41, s22, v208, v1
	v_add3_u32 v170, s22, v207, v1
	v_add3_u32 v171, s22, v206, v1
	v_add3_u32 v174, s22, v201, v1
	v_add3_u32 v1, s22, v199, v1
	v_mfma_f32_32x32x16_f16 v[18:33], v[158:161], v[42:45], v[18:33]
	ds_read_b128 v[118:121], v38 offset:8192
	ds_read_b128 v[122:125], v39
	ds_read_b128 v[224:227], v39 offset:8192
	ds_read_b128 v[126:129], v40
	ds_read_b128 v[228:231], v40 offset:8192
	ds_read_b128 v[158:161], v41
	ds_read_b128 v[198:201], v41 offset:8192
	v_mfma_f32_32x32x16_f16 v[2:17], v[166:169], v[42:45], v[2:17]
	ds_read_b128 v[162:165], v170
	ds_read_b128 v[232:235], v170 offset:8192
	ds_read_b128 v[166:169], v171
	ds_read_b128 v[236:239], v171 offset:8192
	ds_read_b128 v[170:173], v174
	ds_read_b128 v[240:243], v174 offset:8192
	ds_read_b128 v[244:247], v1
	ds_read_b128 v[248:251], v1 offset:8192
	v_add_u32_e32 v1, s3, v209
	s_nop 0
	v_readfirstlane_b32 s12, v1
	s_mov_b32 m0, s12
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[34:49], v[34:37], v[114:117], 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[154:157], v[34:49]
	ds_bpermute_b32 v125, v222, v202
	v_lshrrev_b64 v[122:123], v220, exec
	v_and_b32_e32 v122, 1, v122
	v_cmp_eq_u32_e32 vcc, 1, v122
	v_add_u32_e32 v124, s3, v210
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v123, 1, v125
	v_bfrev_b32_e32 v125, 1
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[150:153], v[34:49]
	ds_bpermute_b32 v126, v222, v216
	v_cndmask_b32_e32 v122, v125, v123, vcc
	buffer_load_dwordx4 v122, s[4:7], 0 offen lds
	v_readfirstlane_b32 s12, v124
	s_mov_b32 m0, s12
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v126
	v_cndmask_b32_e32 v1, v125, v1, vcc
	v_mfma_f32_32x32x16_f16 v[114:129], v[118:121], v[114:117], 0
	buffer_load_dwordx4 v1, s[4:7], 0 offen lds
	s_mov_b32 s4, 0x3e0293ee
	v_mul_f32_e32 v202, 0x3e0293ee, v218
	v_fma_f32 v82, v82, s4, -v202
	v_fma_f32 v83, v83, s4, -v202
	v_fma_f32 v84, v84, s4, -v202
	v_fma_f32 v85, v85, s4, -v202
	v_mfma_f32_32x32x16_f16 v[114:129], v[224:227], v[154:157], v[114:129]
	v_fma_f32 v86, v86, s4, -v202
	v_fma_f32 v87, v87, s4, -v202
	v_fma_f32 v88, v88, s4, -v202
	v_fma_f32 v89, v89, s4, -v202
	v_exp_f32_e32 v205, v86
	v_exp_f32_e32 v206, v87
	v_exp_f32_e32 v207, v88
	v_mfma_f32_32x32x16_f16 v[34:49], v[158:161], v[146:149], v[34:49]
	v_exp_f32_e32 v208, v89
	v_fma_f32 v90, v90, s4, -v202
	v_fma_f32 v91, v91, s4, -v202
	v_fma_f32 v92, v92, s4, -v202
	v_fma_f32 v93, v93, s4, -v202
	v_fma_f32 v94, v94, s4, -v202
	v_fma_f32 v95, v95, s4, -v202
	v_mfma_f32_32x32x16_f16 v[114:129], v[228:231], v[150:153], v[114:129]
	v_fma_f32 v150, v106, s4, -v202
	v_fma_f32 v96, v96, s4, -v202
	v_fma_f32 v97, v97, s4, -v202
	v_fma_f32 v98, v98, s4, -v202
	v_fma_f32 v99, v99, s4, -v202
	v_fma_f32 v100, v100, s4, -v202
	v_fma_f32 v101, v101, s4, -v202
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[142:145], v[34:49]
	v_add_u32_e32 v1, s21, v215
	v_exp_f32_e32 v209, v90
	v_exp_f32_e32 v210, v91
	v_exp_f32_e32 v211, v92
	v_exp_f32_e32 v212, v93
	v_exp_f32_e32 v213, v94
	v_exp_f32_e32 v216, v95
	v_mfma_f32_32x32x16_f16 v[114:129], v[198:201], v[146:149], v[114:129]
	v_exp_f32_e32 v198, v82
	v_fma_f32 v82, v203, s4, -v202
	v_exp_f32_e32 v199, v83
	v_exp_f32_e32 v200, v84
	v_exp_f32_e32 v201, v85
	v_exp_f32_e32 v106, v82
	v_exp_f32_e32 v217, v96
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[138:141], v[34:49]
	v_exp_f32_e32 v219, v97
	v_exp_f32_e32 v220, v98
	v_exp_f32_e32 v222, v99
	v_exp_f32_e32 v224, v100
	v_exp_f32_e32 v225, v101
	v_cvt_pk_f16_f32 v98, v198, v199
	v_cvt_pk_f16_f32 v99, v200, v201
	v_mfma_f32_32x32x16_f16 v[114:129], v[232:235], v[142:145], v[114:129]
	v_cvt_pk_f16_f32 v100, v205, v206
	v_cvt_pk_f16_f32 v101, v207, v208
	v_mul_f32_e32 v82, v66, v106
	v_mul_f32_e32 v83, v67, v106
	v_mul_f32_e32 v84, v68, v106
	v_mul_f32_e32 v85, v69, v106
	v_mul_f32_e32 v86, v70, v106
	v_mul_f32_e32 v87, v71, v106
	v_mul_f32_e32 v88, v72, v106
	v_mul_f32_e32 v89, v73, v106
	v_mul_f32_e32 v90, v74, v106
	v_mul_f32_e32 v91, v75, v106
	v_mul_f32_e32 v92, v76, v106
	v_mul_f32_e32 v93, v77, v106
	v_mul_f32_e32 v94, v78, v106
	v_mul_f32_e32 v95, v79, v106
	v_mul_f32_e32 v96, v80, v106
	v_mul_f32_e32 v97, v81, v106
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b64_tr_b16 v[194:195], v1
	ds_read_b64_tr_b16 v[196:197], v1 offset:2048
	ds_read_b64_tr_b16 v[190:191], v1 offset:4096
	ds_read_b64_tr_b16 v[192:193], v1 offset:6144
	ds_read_b64_tr_b16 v[186:187], v1 offset:8192
	ds_read_b64_tr_b16 v[188:189], v1 offset:10240
	ds_read_b64_tr_b16 v[182:183], v1 offset:12288
	ds_read_b64_tr_b16 v[184:185], v1 offset:14336
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[134:137], v[34:49]
	v_fma_f32 v102, v102, s4, -v202
	v_fma_f32 v103, v103, s4, -v202
	v_fma_f32 v104, v104, s4, -v202
	v_fma_f32 v105, v105, s4, -v202
	v_exp_f32_e32 v226, v102
	v_exp_f32_e32 v227, v103
	v_exp_f32_e32 v228, v104
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[82:97], v[194:197], v[98:101], v[82:97]
	v_exp_f32_e32 v194, v105
	v_cvt_pk_f16_f32 v102, v209, v210
	v_cvt_pk_f16_f32 v103, v211, v212
	v_cvt_pk_f16_f32 v104, v213, v216
	v_cvt_pk_f16_f32 v105, v217, v219
	v_add_u32_e32 v1, s21, v221
	ds_read_b64_tr_b16 v[178:179], v1
	ds_read_b64_tr_b16 v[180:181], v1 offset:2048
	ds_read_b64_tr_b16 v[174:175], v1 offset:4096
	ds_read_b64_tr_b16 v[176:177], v1 offset:6144
	ds_read_b64_tr_b16 v[170:171], v1 offset:8192
	ds_read_b64_tr_b16 v[172:173], v1 offset:10240
	ds_read_b64_tr_b16 v[166:167], v1 offset:12288
	ds_read_b64_tr_b16 v[168:169], v1 offset:14336
	v_mfma_f32_32x32x16_f16 v[114:129], v[236:239], v[138:141], v[114:129]
	v_add_u32_e32 v1, s21, v214
	v_fma_f32 v108, v108, s4, -v202
	v_fma_f32 v109, v109, s4, -v202
	v_fma_f32 v110, v110, s4, -v202
	ds_read_b64_tr_b16 v[162:163], v1
	ds_read_b64_tr_b16 v[164:165], v1 offset:2048
	ds_read_b64_tr_b16 v[158:159], v1 offset:4096
	ds_read_b64_tr_b16 v[160:161], v1 offset:6144
	v_fma_f32 v146, v111, s4, -v202
	v_cvt_pk_f16_f32 v111, v228, v194
	v_mfma_f32_32x32x16_f16 v[34:49], v[244:247], v[130:133], v[34:49]
	v_exp_f32_e32 v195, v150
	v_mul_f32_e32 v66, v50, v106
	v_mul_f32_e32 v67, v51, v106
	v_mul_f32_e32 v68, v52, v106
	v_mul_f32_e32 v69, v53, v106
	v_mul_f32_e32 v70, v54, v106
	v_mul_f32_e32 v71, v55, v106
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[82:97], v[190:193], v[102:105], v[82:97]
	v_exp_f32_e32 v190, v108
	v_exp_f32_e32 v191, v109
	v_exp_f32_e32 v192, v110
	v_cvt_pk_f16_f32 v108, v220, v222
	v_cvt_pk_f16_f32 v109, v224, v225
	v_cvt_pk_f16_f32 v110, v226, v227
	v_max_f32_e32 v50, v34, v34
	v_mfma_f32_32x32x16_f16 v[114:129], v[240:243], v[134:137], v[114:129]
	ds_read_b64_tr_b16 v[134:135], v1 offset:8192
	ds_read_b64_tr_b16 v[136:137], v1 offset:10240
	ds_read_b64_tr_b16 v[138:139], v1 offset:12288
	ds_read_b64_tr_b16 v[140:141], v1 offset:14336
	v_add_u32_e32 v1, s21, v204
	v_mul_f32_e32 v72, v56, v106
	v_mul_f32_e32 v73, v57, v106
	v_mul_f32_e32 v74, v58, v106
	v_mul_f32_e32 v75, v59, v106
	v_mul_f32_e32 v76, v60, v106
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[108:111], v[82:97]
	v_exp_f32_e32 v186, v146
	ds_read_b64_tr_b16 v[142:143], v1
	ds_read_b64_tr_b16 v[144:145], v1 offset:2048
	ds_read_b64_tr_b16 v[146:147], v1 offset:4096
	ds_read_b64_tr_b16 v[148:149], v1 offset:6144
	ds_read_b64_tr_b16 v[150:151], v1 offset:8192
	ds_read_b64_tr_b16 v[152:153], v1 offset:10240
	ds_read_b64_tr_b16 v[154:155], v1 offset:12288
	ds_read_b64_tr_b16 v[156:157], v1 offset:14336
	v_max_f32_e32 v1, v35, v35
	v_max_f32_e32 v1, v50, v1
	v_max3_f32 v1, v1, v36, v37
	v_max3_f32 v1, v1, v38, v39
	v_mfma_f32_32x32x16_f16 v[114:129], v[248:251], v[130:133], v[114:129]
	v_max3_f32 v1, v1, v40, v41
	v_max3_f32 v1, v1, v42, v43
	v_max3_f32 v1, v1, v44, v45
	v_max3_f32 v1, v1, v46, v47
	v_max3_f32 v1, v1, v48, v49
	v_mul_f32_e32 v77, v61, v106
	v_mul_f32_e32 v78, v62, v106
	s_nop 4
	v_max3_f32 v1, v1, v114, v115
	v_max3_f32 v1, v1, v116, v117
	v_max3_f32 v1, v1, v118, v119
	v_mul_f32_e32 v79, v63, v106
	v_mul_f32_e32 v80, v64, v106
	v_mul_f32_e32 v81, v65, v106
	v_mul_f32_e32 v50, v18, v106
	v_mul_f32_e32 v51, v19, v106
	v_mul_f32_e32 v52, v20, v106
	v_mul_f32_e32 v53, v21, v106
	v_mul_f32_e32 v54, v22, v106
	v_mul_f32_e32 v55, v23, v106
	v_mul_f32_e32 v56, v24, v106
	v_mul_f32_e32 v57, v25, v106
	v_mul_f32_e32 v58, v26, v106
	v_mul_f32_e32 v59, v27, v106
	v_mul_f32_e32 v60, v28, v106
	v_mul_f32_e32 v61, v29, v106
	v_mul_f32_e32 v62, v30, v106
	v_mul_f32_e32 v63, v31, v106
	v_mul_f32_e32 v64, v32, v106
	v_mul_f32_e32 v65, v33, v106
	v_max3_f32 v1, v1, v120, v121
	v_max3_f32 v1, v1, v122, v123
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[162:165], v[98:101], v[50:65]
	v_max3_f32 v1, v1, v124, v125
	v_max3_f32 v1, v1, v126, v127
	v_max3_f32 v1, v1, v128, v129
	v_mov_b32_e32 v18, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v18
	v_max3_f32 v1, v218, v1, v18
	v_add_f32_e32 v18, v198, v199
	v_add_f32_e32 v18, v200, v18
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[50:65], v[158:161], v[102:105], v[50:65]
	v_add_f32_e32 v18, v201, v18
	v_add_f32_e32 v18, v205, v18
	v_add_f32_e32 v18, v206, v18
	v_add_f32_e32 v18, v207, v18
	v_add_f32_e32 v18, v208, v18
	v_add_f32_e32 v18, v209, v18
	v_add_f32_e32 v18, v210, v18
	v_add_f32_e32 v18, v211, v18
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[50:65], v[134:137], v[108:111], v[50:65]
	v_add_f32_e32 v134, v212, v18
	v_mul_f32_e32 v18, v2, v106
	v_mul_f32_e32 v19, v3, v106
	v_mul_f32_e32 v20, v4, v106
	v_mul_f32_e32 v21, v5, v106
	v_mul_f32_e32 v22, v6, v106
	v_mul_f32_e32 v23, v7, v106
	v_mul_f32_e32 v24, v8, v106
	v_mul_f32_e32 v25, v9, v106
	v_mul_f32_e32 v26, v10, v106
	v_mul_f32_e32 v27, v11, v106
	v_mul_f32_e32 v28, v12, v106
	v_mul_f32_e32 v29, v13, v106
	v_mul_f32_e32 v30, v14, v106
	v_mul_f32_e32 v31, v15, v106
	v_mul_f32_e32 v32, v16, v106
	v_mul_f32_e32 v33, v17, v106
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[98:101], v[66:81]
	v_add_f32_e32 v2, v213, v134
	v_add_f32_e32 v2, v216, v2
	v_add_f32_e32 v2, v217, v2
	v_add_f32_e32 v2, v219, v2
	v_add_f32_e32 v2, v220, v2
	v_add_f32_e32 v2, v222, v2
	v_add_f32_e32 v2, v224, v2
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[142:145], v[98:101], v[18:33]
	v_add_f32_e32 v2, v225, v2
	v_fma_f32 v107, v107, s4, -v202
	v_add_f32_e32 v2, v226, v2
	v_exp_f32_e32 v107, v107
	v_add_f32_e32 v2, v227, v2
	v_add_f32_e32 v2, v228, v2
	v_fma_f32 v112, v112, s4, -v202
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[102:105], v[66:81]
	v_fma_f32 v113, v113, s4, -v202
	v_add_f32_e32 v2, v194, v2
	v_exp_f32_e32 v112, v112
	v_exp_f32_e32 v113, v113
	v_add_f32_e32 v2, v195, v2
	v_add_f32_e32 v2, v107, v2
	v_add_f32_e32 v2, v190, v2
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[18:33], v[146:149], v[102:105], v[18:33]
	v_add_f32_e32 v2, v191, v2
	v_cvt_pk_f16_f32 v130, v195, v107
	v_cvt_pk_f16_f32 v131, v190, v191
	v_cvt_pk_f16_f32 v132, v192, v186
	v_cvt_pk_f16_f32 v133, v112, v113
	v_add_f32_e32 v2, v192, v2
	v_add_f32_e32 v2, v186, v2
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[108:111], v[66:81]
	v_mul_f32_e32 v158, 0x3e0293ee, v1
	v_add_f32_e32 v2, v112, v2
	v_add_f32_e32 v107, v113, v2
	v_fma_f32 v2, v34, s4, -v158
	v_fma_f32 v3, v35, s4, -v158
	v_fma_f32 v4, v36, s4, -v158
	v_fma_f32 v5, v37, s4, -v158
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[150:153], v[108:111], v[18:33]
	v_fma_f32 v6, v38, s4, -v158
	v_fma_f32 v7, v39, s4, -v158
	v_fma_f32 v8, v40, s4, -v158
	v_fma_f32 v9, v41, s4, -v158
	v_exp_f32_e32 v109, v2
	v_fma_f32 v2, v218, s4, -v158
	v_fma_f32 v34, v114, s4, -v158
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[130:133], v[82:97]
	v_fma_f32 v35, v115, s4, -v158
	v_fma_f32 v36, v116, s4, -v158
	v_exp_f32_e32 v110, v3
	v_exp_f32_e32 v111, v4
	v_exp_f32_e32 v112, v5
	v_exp_f32_e32 v113, v6
	v_exp_f32_e32 v114, v7
	v_mfma_f32_32x32x16_f16 v[50:65], v[138:141], v[130:133], v[50:65]
	v_exp_f32_e32 v115, v8
	v_exp_f32_e32 v116, v9
	v_exp_f32_e32 v141, v2
	v_fma_f32 v10, v42, s4, -v158
	v_fma_f32 v37, v117, s4, -v158
	v_fma_f32 v38, v118, s4, -v158
	v_fma_f32 v39, v119, s4, -v158
	v_fma_f32 v40, v120, s4, -v158
	v_fma_f32 v41, v121, s4, -v158
	v_fma_f32 v42, v122, s4, -v158
	v_add_u32_e32 v2, s2, v215
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[130:133], v[66:81]
	v_fma_f32 v13, v45, s4, -v158
	v_fma_f32 v14, v46, s4, -v158
	v_fma_f32 v15, v47, s4, -v158
	v_fma_f32 v16, v48, s4, -v158
	v_fma_f32 v17, v49, s4, -v158
	v_fma_f32 v45, v125, s4, -v158
	v_fma_f32 v46, v126, s4, -v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[154:157], v[130:133], v[18:33]
	v_fma_f32 v47, v127, s4, -v158
	v_fma_f32 v48, v128, s4, -v158
	v_fma_f32 v49, v129, s4, -v158
	v_exp_f32_e32 v125, v34
	v_exp_f32_e32 v126, v35
	v_exp_f32_e32 v127, v36
	v_exp_f32_e32 v128, v37
	v_exp_f32_e32 v129, v38
	v_exp_f32_e32 v130, v39
	v_exp_f32_e32 v131, v40
	v_exp_f32_e32 v132, v41
	v_exp_f32_e32 v133, v42
	s_waitcnt vmcnt(0)
	s_barrier
	v_add_u32_e32 v42, s3, v215
	ds_read_b64_tr_b16 v[34:35], v2 offset:32768
	ds_read_b64_tr_b16 v[36:37], v42 offset:2048
	ds_read_b64_tr_b16 v[38:39], v42 offset:4096
	ds_read_b64_tr_b16 v[40:41], v42 offset:6144
	v_fma_f32 v11, v43, s4, -v158
	v_fma_f32 v12, v44, s4, -v158
	v_fma_f32 v43, v123, s4, -v158
	v_fma_f32 v44, v124, s4, -v158
	v_exp_f32_e32 v117, v10
	v_exp_f32_e32 v118, v11
	v_exp_f32_e32 v119, v12
	v_exp_f32_e32 v120, v13
	v_exp_f32_e32 v121, v14
	v_exp_f32_e32 v122, v15
	v_exp_f32_e32 v123, v16
	v_exp_f32_e32 v124, v17
	v_cvt_pk_f16_f32 v102, v109, v110
	v_cvt_pk_f16_f32 v103, v111, v112
	v_cvt_pk_f16_f32 v104, v113, v114
	v_cvt_pk_f16_f32 v105, v115, v116
	v_mul_f32_e32 v2, v82, v141
	v_mul_f32_e32 v3, v83, v141
	v_mul_f32_e32 v4, v84, v141
	v_mul_f32_e32 v5, v85, v141
	v_mul_f32_e32 v6, v86, v141
	v_mul_f32_e32 v7, v87, v141
	v_mul_f32_e32 v8, v88, v141
	v_mul_f32_e32 v9, v89, v141
	v_mul_f32_e32 v10, v90, v141
	v_mul_f32_e32 v11, v91, v141
	v_mul_f32_e32 v12, v92, v141
	v_mul_f32_e32 v13, v93, v141
	v_mul_f32_e32 v14, v94, v141
	v_mul_f32_e32 v15, v95, v141
	v_mul_f32_e32 v16, v96, v141
	v_mul_f32_e32 v17, v97, v141
	v_cvt_pk_f16_f32 v98, v117, v118
	v_cvt_pk_f16_f32 v99, v119, v120
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[34:37], v[102:105], v[2:17]
	v_cvt_pk_f16_f32 v100, v121, v122
	v_cvt_pk_f16_f32 v101, v123, v124
	ds_read_b64_tr_b16 v[34:35], v42 offset:8192
	ds_read_b64_tr_b16 v[36:37], v42 offset:10240
	v_cvt_pk_f16_f32 v82, v125, v126
	v_cvt_pk_f16_f32 v83, v127, v128
	v_cvt_pk_f16_f32 v84, v129, v130
	v_cvt_pk_f16_f32 v85, v131, v132
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[38:41], v[98:101], v[2:17]
	v_exp_f32_e32 v134, v43
	v_exp_f32_e32 v135, v44
	v_exp_f32_e32 v136, v45
	v_exp_f32_e32 v137, v46
	v_exp_f32_e32 v138, v47
	v_exp_f32_e32 v139, v48
	v_exp_f32_e32 v140, v49
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[34:37], v[82:85], v[2:17]
	ds_read_b64_tr_b16 v[38:39], v42 offset:12288
	ds_read_b64_tr_b16 v[40:41], v42 offset:14336
	v_add_u32_e32 v34, s2, v221
	v_add_u32_e32 v142, s3, v221
	ds_read_b64_tr_b16 v[90:91], v34 offset:32768
	ds_read_b64_tr_b16 v[92:93], v142 offset:2048
	ds_read_b64_tr_b16 v[94:95], v142 offset:4096
	ds_read_b64_tr_b16 v[96:97], v142 offset:6144
	v_cvt_pk_f16_f32 v86, v133, v134
	v_cvt_pk_f16_f32 v87, v135, v136
	v_cvt_pk_f16_f32 v88, v137, v138
	v_cvt_pk_f16_f32 v89, v139, v140
	v_mul_f32_e32 v34, v66, v141
	v_mul_f32_e32 v35, v67, v141
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[38:41], v[86:89], v[2:17]
	v_mul_f32_e32 v36, v68, v141
	v_mul_f32_e32 v37, v69, v141
	v_mul_f32_e32 v38, v70, v141
	v_mul_f32_e32 v39, v71, v141
	v_mul_f32_e32 v40, v72, v141
	v_mul_f32_e32 v41, v73, v141
	v_mul_f32_e32 v42, v74, v141
	v_mul_f32_e32 v43, v75, v141
	v_mul_f32_e32 v44, v76, v141
	v_mul_f32_e32 v45, v77, v141
	v_mul_f32_e32 v46, v78, v141
	v_mul_f32_e32 v47, v79, v141
	v_mul_f32_e32 v48, v80, v141
	v_mul_f32_e32 v49, v81, v141
	v_add_f32_e32 v66, v109, v110
	v_add_f32_e32 v66, v111, v66
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[90:93], v[102:105], v[34:49]
	v_add_f32_e32 v70, v112, v66
	ds_read_b64_tr_b16 v[66:67], v142 offset:8192
	ds_read_b64_tr_b16 v[68:69], v142 offset:10240
	v_add_f32_e32 v70, v113, v70
	v_add_f32_e32 v70, v114, v70
	v_add_f32_e32 v70, v115, v70
	v_add_f32_e32 v70, v116, v70
	v_add_f32_e32 v74, v117, v70
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[94:97], v[98:101], v[34:49]
	ds_read_b64_tr_b16 v[70:71], v142 offset:12288
	ds_read_b64_tr_b16 v[72:73], v142 offset:14336
	v_add_f32_e32 v79, v118, v74
	v_add_u32_e32 v80, s3, v214
	v_mul_f32_e32 v50, v50, v141
	v_mul_f32_e32 v51, v51, v141
	v_mul_f32_e32 v52, v52, v141
	v_mul_f32_e32 v53, v53, v141
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[66:69], v[82:85], v[34:49]
	v_add_u32_e32 v66, s2, v214
	ds_read_b64_tr_b16 v[66:67], v66 offset:32768
	ds_read_b64_tr_b16 v[68:69], v80 offset:2048
	ds_read_b64_tr_b16 v[74:75], v80 offset:4096
	ds_read_b64_tr_b16 v[76:77], v80 offset:6144
	v_mul_f32_e32 v54, v54, v141
	v_mul_f32_e32 v55, v55, v141
	v_mul_f32_e32 v56, v56, v141
	v_mul_f32_e32 v57, v57, v141
	v_mul_f32_e32 v58, v58, v141
	v_mul_f32_e32 v59, v59, v141
	v_mul_f32_e32 v60, v60, v141
	v_mul_f32_e32 v61, v61, v141
	v_mul_f32_e32 v62, v62, v141
	v_mul_f32_e32 v63, v63, v141
	v_mul_f32_e32 v64, v64, v141
	v_mul_f32_e32 v65, v65, v141
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[34:49], v[70:73], v[86:89], v[34:49]
	v_mul_f32_e32 v18, v18, v141
	v_mul_f32_e32 v19, v19, v141
	v_mul_f32_e32 v20, v20, v141
	v_mul_f32_e32 v21, v21, v141
	v_mul_f32_e32 v22, v22, v141
	v_mul_f32_e32 v23, v23, v141
	v_mul_f32_e32 v24, v24, v141
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[66:69], v[102:105], v[50:65]
	v_add_f32_e32 v66, v119, v79
	v_add_f32_e32 v66, v120, v66
	v_add_f32_e32 v66, v121, v66
	v_add_f32_e32 v66, v122, v66
	v_add_f32_e32 v70, v123, v66
	ds_read_b64_tr_b16 v[66:67], v80 offset:8192
	ds_read_b64_tr_b16 v[68:69], v80 offset:10240
	v_add_f32_e32 v70, v124, v70
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[74:77], v[98:101], v[50:65]
	v_add_f32_e32 v70, v125, v70
	v_add_f32_e32 v70, v126, v70
	v_add_f32_e32 v70, v127, v70
	v_add_f32_e32 v74, v128, v70
	ds_read_b64_tr_b16 v[70:71], v80 offset:12288
	ds_read_b64_tr_b16 v[72:73], v80 offset:14336
	v_add_f32_e32 v79, v129, v74
	v_add_u32_e32 v80, s3, v204
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[66:69], v[82:85], v[50:65]
	v_add_u32_e32 v66, s2, v204
	ds_read_b64_tr_b16 v[66:67], v66 offset:32768
	ds_read_b64_tr_b16 v[68:69], v80 offset:2048
	ds_read_b64_tr_b16 v[74:75], v80 offset:4096
	ds_read_b64_tr_b16 v[76:77], v80 offset:6144
	v_mul_f32_e32 v25, v25, v141
	v_mul_f32_e32 v26, v26, v141
	v_mul_f32_e32 v27, v27, v141
	v_mul_f32_e32 v28, v28, v141
	v_mul_f32_e32 v29, v29, v141
	v_mul_f32_e32 v30, v30, v141
	v_mul_f32_e32 v31, v31, v141
	v_mul_f32_e32 v32, v32, v141
	v_mul_f32_e32 v33, v33, v141
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[86:89], v[50:65]
	v_mov_b32_e32 v108, v107
	s_nop 1
	v_permlane32_swap_b32_e32 v107, v108
	v_add_f32_e32 v78, v107, v108
	v_fmac_f32_e32 v78, v223, v106
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[66:69], v[102:105], v[18:33]
	v_add_f32_e32 v66, v130, v79
	v_add_f32_e32 v66, v131, v66
	v_add_f32_e32 v66, v132, v66
	v_add_f32_e32 v66, v133, v66
	v_add_f32_e32 v70, v134, v66
	ds_read_b64_tr_b16 v[66:67], v80 offset:8192
	ds_read_b64_tr_b16 v[68:69], v80 offset:10240
	v_add_f32_e32 v70, v135, v70
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[74:77], v[98:101], v[18:33]
	v_add_f32_e32 v70, v136, v70
	v_add_f32_e32 v70, v137, v70
	v_add_f32_e32 v70, v138, v70
	v_add_f32_e32 v74, v139, v70
	ds_read_b64_tr_b16 v[70:71], v80 offset:12288
	ds_read_b64_tr_b16 v[72:73], v80 offset:14336
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[66:69], v[82:85], v[18:33]
	v_add_f32_e32 v66, v140, v74
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v78, v141
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[70:73], v[86:89], v[18:33]
.LBB0_15:                               ; %Flow544
	v_div_scale_f32 v69, s[4:5], v66, v66, 1.0
	v_rcp_f32_e32 v69, v69
	v_div_scale_f32 v70, vcc, 1.0, v66, 1.0
	s_add_i32 s2, s60, 0x100
	v_mul_f32_e32 v69, v70, v69
	s_sub_i32 s3, s33, s67
	s_nop 0
	v_div_fmas_f32 v69, 0, 0, v69
	s_cmp_le_i32 s3, s60
	v_lshrrev_b32_e32 v68, 1, v0
	v_div_fixup_f32 v70, v69, v66, 1.0
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_ge_i32 s3, s2
	v_and_b32_e32 v67, 64, v68
	v_and_b32_e32 v68, 0xa0, v68
	v_pk_mul_f32 v[2:3], v[2:3], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[40:41], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[42:43], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[44:45], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[46:47], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[48:49], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[50:51], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[52:53], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[54:55], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[56:57], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[58:59], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[60:61], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[62:63], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[64:65], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[18:19], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[20:21], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[22:23], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[24:25], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[26:27], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[28:29], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[30:31], v[70:71] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[32:33], v[70:71] op_sel_hi:[1,0]
	s_cselect_b64 s[6:7], -1, 0
	v_or3_b32 v67, v68, v254, v67
	v_cvt_pk_f16_f32 v62, v2, v3
	v_cvt_pk_f16_f32 v64, v4, v5
	v_cvt_pk_f16_f32 v56, v6, v7
	v_cvt_pk_f16_f32 v60, v8, v9
	v_cvt_pk_f16_f32 v52, v10, v11
	v_cvt_pk_f16_f32 v57, v12, v13
	v_cvt_pk_f16_f32 v48, v14, v15
	v_cvt_pk_f16_f32 v53, v16, v17
	v_cvt_pk_f16_f32 v44, v34, v35
	v_cvt_pk_f16_f32 v49, v36, v37
	v_cvt_pk_f16_f32 v40, v38, v39
	v_cvt_pk_f16_f32 v45, v72, v73
	v_cvt_pk_f16_f32 v36, v74, v75
	v_cvt_pk_f16_f32 v41, v76, v77
	v_cvt_pk_f16_f32 v32, v78, v79
	v_cvt_pk_f16_f32 v37, v80, v81
	v_cvt_pk_f16_f32 v28, v82, v83
	v_cvt_pk_f16_f32 v33, v84, v85
	v_cvt_pk_f16_f32 v24, v86, v87
	v_cvt_pk_f16_f32 v29, v88, v89
	v_cvt_pk_f16_f32 v20, v90, v91
	v_cvt_pk_f16_f32 v25, v92, v93
	v_cvt_pk_f16_f32 v16, v94, v95
	v_cvt_pk_f16_f32 v21, v96, v97
	v_cvt_pk_f16_f32 v12, v98, v99
	v_cvt_pk_f16_f32 v17, v100, v101
	v_cvt_pk_f16_f32 v8, v102, v103
	v_cvt_pk_f16_f32 v13, v104, v105
	v_cvt_pk_f16_f32 v4, v106, v107
	v_cvt_pk_f16_f32 v9, v108, v109
	v_cvt_pk_f16_f32 v2, v110, v111
	v_cvt_pk_f16_f32 v5, v70, v71
	s_or_b64 s[4:5], s[4:5], s[6:7]
	v_or_b32_e32 v68, s60, v67
	v_lshrrev_b32_e32 v63, 16, v62
	v_lshrrev_b32_e32 v65, 16, v64
	v_lshrrev_b32_e32 v58, 16, v56
	v_lshrrev_b32_e32 v61, 16, v60
	v_lshrrev_b32_e32 v54, 16, v52
	v_lshrrev_b32_e32 v59, 16, v57
	v_lshrrev_b32_e32 v50, 16, v48
	v_lshrrev_b32_e32 v55, 16, v53
	v_lshrrev_b32_e32 v46, 16, v44
	v_lshrrev_b32_e32 v51, 16, v49
	v_lshrrev_b32_e32 v42, 16, v40
	v_lshrrev_b32_e32 v47, 16, v45
	v_lshrrev_b32_e32 v38, 16, v36
	v_lshrrev_b32_e32 v43, 16, v41
	v_lshrrev_b32_e32 v34, 16, v32
	v_lshrrev_b32_e32 v39, 16, v37
	v_lshrrev_b32_e32 v30, 16, v28
	v_lshrrev_b32_e32 v35, 16, v33
	v_lshrrev_b32_e32 v26, 16, v24
	v_lshrrev_b32_e32 v31, 16, v29
	v_lshrrev_b32_e32 v22, 16, v20
	v_lshrrev_b32_e32 v27, 16, v25
	v_lshrrev_b32_e32 v18, 16, v16
	v_lshrrev_b32_e32 v23, 16, v21
	v_lshrrev_b32_e32 v14, 16, v12
	v_lshrrev_b32_e32 v19, 16, v17
	v_lshrrev_b32_e32 v10, 16, v8
	v_lshrrev_b32_e32 v15, 16, v13
	v_lshrrev_b32_e32 v6, 16, v4
	v_lshrrev_b32_e32 v11, 16, v9
	v_lshrrev_b32_e32 v3, 16, v2
	v_lshrrev_b32_e32 v7, 16, v5
	s_and_b64 vcc, exec, s[4:5]
	s_barrier
	s_cbranch_vccnz .LBB0_17
; %bb.16:
	v_cmp_gt_i32_e32 vcc, s3, v68
	s_nop 1
	v_cndmask_b32_e64 v62, v62, 0, vcc
	v_cndmask_b32_e64 v63, v63, 0, vcc
	v_cndmask_b32_e64 v64, v64, 0, vcc
	v_cndmask_b32_e64 v65, v65, 0, vcc
	v_cndmask_b32_e64 v56, v56, 0, vcc
	v_cndmask_b32_e64 v58, v58, 0, vcc
	v_cndmask_b32_e64 v60, v60, 0, vcc
	v_cndmask_b32_e64 v61, v61, 0, vcc
	v_cndmask_b32_e64 v52, v52, 0, vcc
	v_cndmask_b32_e64 v54, v54, 0, vcc
	v_cndmask_b32_e64 v57, v57, 0, vcc
	v_cndmask_b32_e64 v59, v59, 0, vcc
	v_cndmask_b32_e64 v48, v48, 0, vcc
	v_cndmask_b32_e64 v50, v50, 0, vcc
	v_cndmask_b32_e64 v53, v53, 0, vcc
	v_cndmask_b32_e64 v55, v55, 0, vcc
	v_cndmask_b32_e64 v44, v44, 0, vcc
	v_cndmask_b32_e64 v46, v46, 0, vcc
	v_cndmask_b32_e64 v49, v49, 0, vcc
	v_cndmask_b32_e64 v51, v51, 0, vcc
	v_cndmask_b32_e64 v40, v40, 0, vcc
	v_cndmask_b32_e64 v42, v42, 0, vcc
	v_cndmask_b32_e64 v45, v45, 0, vcc
	v_cndmask_b32_e64 v47, v47, 0, vcc
	v_cndmask_b32_e64 v36, v36, 0, vcc
	v_cndmask_b32_e64 v38, v38, 0, vcc
	v_cndmask_b32_e64 v41, v41, 0, vcc
	v_cndmask_b32_e64 v43, v43, 0, vcc
	v_cndmask_b32_e64 v32, v32, 0, vcc
	v_cndmask_b32_e64 v34, v34, 0, vcc
	v_cndmask_b32_e64 v37, v37, 0, vcc
	v_cndmask_b32_e64 v39, v39, 0, vcc
	v_cndmask_b32_e64 v28, v28, 0, vcc
	v_cndmask_b32_e64 v30, v30, 0, vcc
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cndmask_b32_e64 v35, v35, 0, vcc
	v_cndmask_b32_e64 v24, v24, 0, vcc
	v_cndmask_b32_e64 v26, v26, 0, vcc
	v_cndmask_b32_e64 v29, v29, 0, vcc
	v_cndmask_b32_e64 v31, v31, 0, vcc
	v_cndmask_b32_e64 v20, v20, 0, vcc
	v_cndmask_b32_e64 v22, v22, 0, vcc
	v_cndmask_b32_e64 v25, v25, 0, vcc
	v_cndmask_b32_e64 v27, v27, 0, vcc
	v_cndmask_b32_e64 v16, v16, 0, vcc
	v_cndmask_b32_e64 v18, v18, 0, vcc
	v_cndmask_b32_e64 v21, v21, 0, vcc
	v_cndmask_b32_e64 v23, v23, 0, vcc
	v_cndmask_b32_e64 v12, v12, 0, vcc
	v_cndmask_b32_e64 v14, v14, 0, vcc
	v_cndmask_b32_e64 v17, v17, 0, vcc
	v_cndmask_b32_e64 v19, v19, 0, vcc
	v_cndmask_b32_e64 v8, v8, 0, vcc
	v_cndmask_b32_e64 v10, v10, 0, vcc
	v_cndmask_b32_e64 v13, v13, 0, vcc
	v_cndmask_b32_e64 v15, v15, 0, vcc
	v_cndmask_b32_e64 v4, v4, 0, vcc
	v_cndmask_b32_e64 v6, v6, 0, vcc
	v_cndmask_b32_e64 v9, v9, 0, vcc
	v_cndmask_b32_e64 v11, v11, 0, vcc
	v_cndmask_b32_e64 v2, v2, 0, vcc
	v_cndmask_b32_e64 v3, v3, 0, vcc
	v_cndmask_b32_e64 v5, v5, 0, vcc
	v_cndmask_b32_e64 v7, v7, 0, vcc
.LBB0_17:
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
	v_lshl_add_u32 v69, v67, 2, 0
	s_cbranch_scc1 .LBB0_19
; %bb.18:
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v66
	v_mov_b32_e32 v70, 0x42000000
	v_cmp_gt_i32_e64 s[8:9], s33, v68
	v_cndmask_b32_e64 v71, 0, 32, vcc
	v_ldexp_f32 v71, v66, v71
	v_log_f32_e32 v71, v71
	v_cndmask_b32_e32 v70, 0, v70, vcc
	s_sub_i32 s2, 0x100, s2
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	v_sub_f32_e32 v68, v71, v70
	v_add_f32_e32 v68, v1, v68
	ds_write_b32 v69, v68
	v_mov_b32_e32 v68, 2
	v_lshlrev_b32_sdwa v68, v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v70, 0, v68
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v70, v70
	v_bfrev_b32_e32 v71, 1
	s_and_b64 vcc, s[0:1], s[2:3]
	s_and_b32 s5, s12, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e32 v68, v71, v68, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v70, v68, s[4:7], 0 offen
	s_cbranch_execz .LBB0_20
	s_branch .LBB0_21
.LBB0_19:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_20:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v66
	v_mov_b32_e32 v68, 0x42000000
	s_and_b32 s5, s12, 0xffff
	v_cndmask_b32_e64 v70, 0, 32, vcc
	v_ldexp_f32 v66, v66, v70
	v_log_f32_e32 v66, v66
	v_cndmask_b32_e32 v68, 0, v68, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_sub_f32_e32 v66, v66, v68
	v_add_f32_e32 v1, v1, v66
	ds_write_b32 v69, v1
	v_mov_b32_e32 v1, 2
	v_lshlrev_b32_sdwa v0, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v1, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v1, v1
	v_bfrev_b32_e32 v66, 1
	v_cndmask_b32_e64 v0, v66, v0, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v1, v0, s[4:7], 0 offen
.LBB0_21:
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
	v_mul_lo_u32 v66, s43, v67
	s_bitset1_b32 s2, 14
	s_mov_b32 s4, 0x5040100
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_perm_b32 v0, v63, v62, s4
	v_add_lshl_u32 v62, v66, v255, 1
	v_bfrev_b32_e32 v63, 1
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_perm_b32 v1, v65, v64, s4
	v_cndmask_b32_e64 v64, v63, v62, s[8:9]
	buffer_store_dwordx2 v[0:1], v64, s[0:3], 0 offen
	v_perm_b32 v0, v58, v56, s4
	v_add_u32_e32 v56, 16, v62
	v_perm_b32 v1, v61, v60, s4
	v_cndmask_b32_e64 v56, v63, v56, s[8:9]
	buffer_store_dwordx2 v[0:1], v56, s[0:3], 0 offen
	v_perm_b32 v0, v54, v52, s4
	v_add_u32_e32 v52, 32, v62
	v_perm_b32 v1, v59, v57, s4
	v_cndmask_b32_e64 v52, v63, v52, s[8:9]
	buffer_store_dwordx2 v[0:1], v52, s[0:3], 0 offen
	v_perm_b32 v0, v50, v48, s4
	v_add_u32_e32 v48, 48, v62
	v_perm_b32 v1, v55, v53, s4
	v_cndmask_b32_e64 v48, v63, v48, s[8:9]
	buffer_store_dwordx2 v[0:1], v48, s[0:3], 0 offen
	v_perm_b32 v0, v46, v44, s4
	v_add_u32_e32 v44, 64, v62
	v_perm_b32 v1, v51, v49, s4
	v_cndmask_b32_e64 v44, v63, v44, s[8:9]
	buffer_store_dwordx2 v[0:1], v44, s[0:3], 0 offen
	v_perm_b32 v0, v42, v40, s4
	v_add_u32_e32 v40, 0x50, v62
	v_perm_b32 v1, v47, v45, s4
	v_cndmask_b32_e64 v40, v63, v40, s[8:9]
	buffer_store_dwordx2 v[0:1], v40, s[0:3], 0 offen
	v_perm_b32 v0, v38, v36, s4
	v_add_u32_e32 v36, 0x60, v62
	v_perm_b32 v1, v43, v41, s4
	v_cndmask_b32_e64 v36, v63, v36, s[8:9]
	buffer_store_dwordx2 v[0:1], v36, s[0:3], 0 offen
	v_perm_b32 v0, v34, v32, s4
	v_add_u32_e32 v32, 0x70, v62
	v_perm_b32 v1, v39, v37, s4
	v_cndmask_b32_e64 v32, v63, v32, s[8:9]
	buffer_store_dwordx2 v[0:1], v32, s[0:3], 0 offen
	v_perm_b32 v0, v30, v28, s4
	v_add_u32_e32 v28, 0x80, v62
	v_perm_b32 v1, v35, v33, s4
	v_cndmask_b32_e64 v28, v63, v28, s[8:9]
	buffer_store_dwordx2 v[0:1], v28, s[0:3], 0 offen
	v_perm_b32 v0, v26, v24, s4
	v_add_u32_e32 v24, 0x90, v62
	v_perm_b32 v1, v31, v29, s4
	v_cndmask_b32_e64 v24, v63, v24, s[8:9]
	buffer_store_dwordx2 v[0:1], v24, s[0:3], 0 offen
	v_perm_b32 v0, v22, v20, s4
	v_add_u32_e32 v20, 0xa0, v62
	v_perm_b32 v1, v27, v25, s4
	v_cndmask_b32_e64 v20, v63, v20, s[8:9]
	buffer_store_dwordx2 v[0:1], v20, s[0:3], 0 offen
	v_perm_b32 v0, v18, v16, s4
	v_add_u32_e32 v16, 0xb0, v62
	v_perm_b32 v1, v23, v21, s4
	v_cndmask_b32_e64 v16, v63, v16, s[8:9]
	buffer_store_dwordx2 v[0:1], v16, s[0:3], 0 offen
	v_perm_b32 v0, v14, v12, s4
	v_add_u32_e32 v12, 0xc0, v62
	v_perm_b32 v1, v19, v17, s4
	v_cndmask_b32_e64 v12, v63, v12, s[8:9]
	buffer_store_dwordx2 v[0:1], v12, s[0:3], 0 offen
	v_perm_b32 v0, v10, v8, s4
	v_add_u32_e32 v8, 0xd0, v62
	v_perm_b32 v1, v15, v13, s4
	v_cndmask_b32_e64 v8, v63, v8, s[8:9]
	buffer_store_dwordx2 v[0:1], v8, s[0:3], 0 offen
	v_perm_b32 v0, v6, v4, s4
	v_add_u32_e32 v4, 0xe0, v62
	v_perm_b32 v1, v11, v9, s4
	v_cndmask_b32_e64 v4, v63, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_perm_b32 v0, v3, v2, s4
	v_add_u32_e32 v2, 0xf0, v62
	v_perm_b32 v1, v7, v5, s4
	v_cndmask_b32_e64 v2, v63, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
.LBB0_22:                               ; %.critedge
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 16
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
		.amdhsa_next_free_sgpr 70
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
	.set attn_fwd.numbered_sgpr, 70
	.set attn_fwd.private_seg_size, 16
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14680
; TotalNumSgprs: 76
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 16
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 9
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 76
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
	.byte	1                               ; Abbrev [1] 0xb:0x5f DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x39 DW_TAG_subprogram
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
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
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
    .private_segment_fixed_size: 16
    .sgpr_count:     76
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 3
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
