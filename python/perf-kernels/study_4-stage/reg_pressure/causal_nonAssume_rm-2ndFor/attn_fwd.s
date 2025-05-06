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
; %bb.31:
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.32:
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
	s_cbranch_scc1 .LBB0_30
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
	s_min_i32 s16, s16, s19
	s_mov_b32 s44, 0
	s_cmp_gt_i32 s16, 0
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
	v_lshlrev_b32_e32 v225, 3, v0
	s_mul_i32 s64, s18, 0xeb200
	s_mul_i32 s62, s17, 0x1d64
	s_cbranch_scc1 .LBB0_3
; %bb.2:
	s_ashr_i32 s59, s58, 31
	s_lshl_b64 s[46:47], s[58:59], 1
	s_add_u32 s19, s10, s46
	s_addc_u32 s41, s11, s47
	s_ashr_i32 s57, s56, 31
	s_lshl_b64 s[46:47], s[56:57], 1
	s_add_u32 s19, s19, s46
	s_addc_u32 s41, s41, s47
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[46:47], s[54:55], 1
	s_add_u32 s19, s19, s46
	s_addc_u32 s41, s41, s47
	s_ashr_i32 s53, s52, 31
	s_lshl_b32 s42, s43, 5
	s_lshl_b64 s[46:47], s[52:53], 1
	s_add_u32 s48, s19, s46
	v_and_b32_e32 v10, 0x78, v225
	s_addc_u32 s19, s41, s47
	v_mad_u64_u32 v[10:11], s[46:47], s43, v1, v[10:11]
	s_and_b32 s41, s43, 0x3fff
	v_add_u32_e32 v15, s42, v10
	s_bitset1_b32 s41, 14
	v_lshlrev_b32_e32 v10, 1, v10
	v_bfrev_b32_e32 v21, 1
	s_mov_b32 s45, s44
	v_add_u32_e32 v16, s42, v15
	s_and_b32 s19, s19, 0xffff
	s_lshl_b32 s41, s41, 16
	v_cndmask_b32_e64 v22, v21, v10, s[26:27]
	s_mov_b32 s46, s44
	s_mov_b32 s47, s44
	v_mov_b64_e32 v[10:11], s[44:45]
	v_lshlrev_b32_e32 v15, 1, v15
	s_or_b32 s49, s19, s41
	s_mov_b32 s51, 0x27000
	s_mov_b32 s50, 0x7ffffffe
	v_mov_b64_e32 v[12:13], s[46:47]
	v_cndmask_b32_e64 v15, v21, v15, s[24:25]
	buffer_store_dwordx4 v[10:13], v22, s[48:51], 0 offen
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v16
	v_add_u32_e32 v17, s42, v16
	v_cndmask_b32_e64 v15, v21, v15, s[22:23]
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v17
	v_add_u32_e32 v18, s42, v17
	v_cndmask_b32_e64 v15, v21, v15, s[20:21]
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v18
	s_ashr_i32 s65, s64, 31
	v_add_u32_e32 v19, s42, v18
	v_cndmask_b32_e64 v15, v21, v15, s[30:31]
	s_lshl_b64 s[20:21], s[64:65], 2
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v19
	s_add_u32 s19, s8, s20
	v_add_u32_e32 v20, s42, v19
	v_cndmask_b32_e64 v15, v21, v15, s[28:29]
	s_addc_u32 s22, s9, s21
	s_ashr_i32 s63, s62, 31
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_lshlrev_b32_e32 v15, 1, v20
	s_lshl_b64 s[20:21], s[62:63], 2
	v_cndmask_b32_e64 v15, v21, v15, s[34:35]
	s_add_u32 s19, s19, s20
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	v_add_lshl_u32 v15, v20, s42, 1
	s_addc_u32 s22, s22, s21
	s_ashr_i32 s61, s60, 31
	v_or_b32_sdwa v14, s60, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_cndmask_b32_e32 v15, v21, v15, vcc
	s_movk_i32 s23, 0x1d64
	s_lshl_b64 s[20:21], s[60:61], 2
	buffer_store_dwordx4 v[10:13], v15, s[48:51], 0 offen
	s_add_u32 s48, s19, s20
	v_cmp_gt_i32_e32 vcc, s23, v14
	v_mov_b32_e32 v10, 2
	s_addc_u32 s19, s22, s21
	v_lshlrev_b32_sdwa v10, v10, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	s_and_b64 vcc, s[0:1], vcc
	s_and_b32 s49, s19, 0xffff
	v_cndmask_b32_e32 v10, v21, v10, vcc
	v_mov_b32_e32 v11, 0x7f800000
	buffer_store_dword v11, v10, s[48:51], 0 offen
.LBB0_3:
	s_cmp_lt_i32 s16, 1
	s_cbranch_scc1 .LBB0_30
; %bb.4:
	s_and_b32 s19, s67, 63
	s_sub_i32 s20, 64, s67
	s_cmp_lt_i32 s67, 64
	s_cselect_b32 s19, s20, s19
	s_mul_i32 s20, s12, s18
	s_ashr_i32 s21, s20, 31
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s12, s2, s20
	s_mul_i32 s2, s13, s17
	s_addc_u32 s20, s3, s21
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s68, s14
	s_addc_u32 s13, s20, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s60
	s_addc_u32 s13, s13, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s24, s14, 5
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s20, s12, s2
	v_and_b32_e32 v34, 0x78, v225
	s_addc_u32 s12, s13, s3
	v_mad_u64_u32 v[10:11], s[2:3], s14, v1, v[34:35]
	v_add_u32_e32 v11, s24, v10
	v_lshlrev_b32_e32 v10, 1, v10
	v_bfrev_b32_e32 v35, 1
	v_cmp_gt_i32_e32 vcc, s33, v9
	v_add_u32_e32 v16, s24, v11
	s_and_b32 s2, s14, 0x3fff
	v_cndmask_b32_e32 v18, v35, v10, vcc
	v_lshlrev_b32_e32 v9, 1, v11
	v_cmp_gt_i32_e32 vcc, s33, v8
	v_add_u32_e32 v17, s24, v16
	s_bitset1_b32 s2, 14
	v_cndmask_b32_e32 v19, v35, v9, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	v_cmp_gt_i32_e32 vcc, s33, v6
	s_and_b32 s3, s12, 0xffff
	s_lshl_b32 s2, s2, 16
	v_cndmask_b32_e32 v6, v35, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v17
	v_cmp_gt_i32_e32 vcc, s33, v4
	v_add_u32_e32 v24, s24, v17
	s_or_b32 s21, s3, s2
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v4, v35, v16, vcc
	v_add_u32_e32 v25, s24, v24
	buffer_load_dwordx4 v[8:11], v18, s[20:23], 0 offen
	buffer_load_dwordx4 v[12:15], v19, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[16:19], v6, s[20:23], 0 offen
	buffer_load_dwordx4 v[20:23], v4, s[20:23], 0 offen
	v_lshlrev_b32_e32 v4, 1, v24
	v_cmp_gt_i32_e32 vcc, s33, v7
	v_add_u32_e32 v32, s24, v25
	v_lshlrev_b32_e32 v6, 1, v25
	v_cndmask_b32_e32 v4, v35, v4, vcc
	v_cmp_gt_i32_e32 vcc, s33, v5
	v_lshrrev_b32_e32 v221, 1, v0
	s_movk_i32 s2, 0x78
	v_cndmask_b32_e32 v5, v35, v6, vcc
	buffer_load_dwordx4 v[24:27], v4, s[20:23], 0 offen
	buffer_load_dwordx4 v[28:31], v5, s[20:23], 0 offen
	v_lshlrev_b32_e32 v4, 1, v32
	v_cmp_gt_i32_e32 vcc, s33, v3
	v_lshlrev_b32_e32 v101, 7, v1
	v_or_b32_e32 v5, 0x1000, v101
	v_cndmask_b32_e32 v3, v35, v4, vcc
	v_add_lshl_u32 v4, v32, s24, 1
	v_cmp_gt_i32_e32 vcc, s33, v2
	v_and_b32_e32 v219, 31, v0
	v_lshlrev_b32_e32 v6, 8, v1
	v_cndmask_b32_e32 v2, v35, v4, vcc
	buffer_load_dwordx4 v[36:39], v3, s[20:23], 0 offen
	buffer_load_dwordx4 v[40:43], v2, s[20:23], 0 offen
	v_lshrrev_b32_e32 v2, 3, v0
	v_and_b32_e32 v224, 4, v2
	v_bitop3_b32 v2, v221, v225, s2 bitop3:0x28
	s_and_b32 s2, s33, 0xff
	v_and_b32_e32 v3, 0x78, v221
	s_or_b32 s2, s19, s2
	v_bitop3_b32 v3, v3, v101, v34 bitop3:0xde
	s_cmp_eq_u32 s2, 0
	v_or_b32_e32 v7, v5, v2
	v_lshlrev_b32_e32 v4, 1, v3
	s_cselect_b32 s2, 4, 5
	v_lshlrev_b32_e32 v32, 1, v2
	v_lshlrev_b32_e32 v3, 1, v7
	v_add_u32_e32 v7, 0, v4
	s_cmp_le_u32 s16, s2
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
	v_lshrrev_b32_e32 v99, 2, v0
	v_and_b32_e32 v6, 8, v99
	v_or_b32_e32 v8, 16, v6
	s_movk_i32 s3, 0xe0
	v_or_b32_e32 v9, 32, v6
	v_and_b32_e32 v14, 15, v0
	v_lshrrev_b32_e32 v15, 5, v0
	v_lshrrev_b32_e32 v8, 3, v8
	v_and_or_b32 v7, v221, s3, v219
	v_or_b32_e32 v10, 48, v6
	v_or_b32_e32 v11, 64, v6
	v_bitop3_b32 v14, v15, v14, 1 bitop3:0x6c
	v_bitop3_b32 v8, v8, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v9
	v_or_b32_e32 v12, 0x50, v6
	v_or_b32_e32 v13, 0x60, v6
	v_or_b32_e32 v6, 0x70, v6
	v_bitop3_b32 v9, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v10, 3, v10
	v_lshrrev_b32_e32 v11, 3, v11
	v_lshl_add_u32 v7, v7, 8, 0
	v_lshlrev_b32_e32 v252, 4, v14
	v_lshlrev_b32_e32 v222, 4, v8
	v_bitop3_b32 v10, v10, v0, 15 bitop3:0x78
	v_bitop3_b32 v11, v11, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v12, 3, v12
	v_lshrrev_b32_e32 v13, 3, v13
	v_lshrrev_b32_e32 v6, 3, v6
	v_add_u32_e32 v14, v7, v252
	v_add_u32_e32 v8, v7, v222
	v_lshlrev_b32_e32 v218, 4, v9
	v_bitop3_b32 v12, v12, v0, 15 bitop3:0x78
	v_bitop3_b32 v13, v13, v0, 15 bitop3:0x78
	v_bitop3_b32 v6, v6, v0, 15 bitop3:0x78
	ds_read_b128 v[134:137], v14
	ds_read_b128 v[130:133], v8
	v_add_u32_e32 v8, v7, v218
	v_lshlrev_b32_e32 v29, 4, v10
	v_lshlrev_b32_e32 v28, 4, v11
	v_add_u32_e32 v9, v7, v29
	ds_read_b128 v[142:145], v8
	ds_read_b128 v[138:141], v9
	v_add_u32_e32 v8, v7, v28
	v_lshlrev_b32_e32 v27, 4, v12
	v_lshlrev_b32_e32 v228, 4, v13
	v_lshlrev_b32_e32 v227, 4, v6
	v_add_u32_e32 v9, v7, v27
	ds_read_b128 v[150:153], v8
	ds_read_b128 v[146:149], v9
	v_add_u32_e32 v8, v7, v228
	v_add_u32_e32 v6, v7, v227
	ds_read_b128 v[158:161], v8
	ds_read_b128 v[154:157], v6
	v_mov_b32_e32 v6, s2
	s_mul_i32 s2, s38, s18
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s6, s6, s2
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s2, s17, 31
	s_lshr_b32 s2, s2, 28
	v_mad_u64_u32 v[208:209], s[12:13], s37, v1, v[34:35]
	s_add_i32 s17, s17, s2
	s_ashr_i32 s12, s17, 4
	s_mul_i32 s2, s39, s12
	s_ashr_i32 s3, s2, 31
	v_or_b32_e32 v102, v101, v34
	v_sub_u32_e64 v36, s16, v6 clamp
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v6, 1, v102
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s66, s40
	v_or_b32_e32 v5, v5, v34
	v_sub_u32_e32 v4, v4, v6
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	v_lshlrev_b32_e32 v98, 1, v5
	v_ashrrev_i16_e32 v5, 15, v4
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshrrev_b16_e32 v5, 12, v5
	s_add_u32 s16, s6, s2
	s_mul_i32 s24, s15, s18
	v_add_u16_e32 v4, v4, v5
	v_and_b32_e32 v253, 63, v0
	s_addc_u32 s17, s7, s3
	s_ashr_i32 s25, s24, 31
	v_ashrrev_i16_e32 v4, 4, v4
	s_lshl_b64 s[2:3], s[24:25], 1
	v_add_u32_sdwa v67, v253, sext(v4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_add_u32 s6, s4, s2
	s_mul_i32 s26, s36, s12
	v_lshlrev_b32_e32 v4, 2, v67
	s_addc_u32 s7, s5, s3
	s_ashr_i32 s27, s26, 31
	v_add_u32_e32 v37, 0, v6
	scratch_store_dword off, v6, off offset:60 ; 4-byte Folded Spill
	ds_bpermute_b32 v6, v4, v208
	v_lshrrev_b64 v[4:5], v67, exec
	v_sub_u32_e32 v3, v3, v98
	s_lshl_b64 s[2:3], s[26:27], 1
	v_ashrrev_i16_e32 v5, 15, v3
	s_add_u32 s6, s6, s2
	s_mul_i32 s28, s66, s37
	v_lshrrev_b16_e32 v5, 12, v5
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s29, s28, 31
	v_add_u16_e32 v3, v3, v5
	s_lshl_b64 s[2:3], s[28:29], 1
	v_ashrrev_i16_e32 v3, 4, v3
	s_add_u32 s20, s6, s2
	v_add_u32_sdwa v69, v253, sext(v3) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshl_add_u32 v116, s37, 5, v208
	s_addc_u32 s2, s7, s3
	s_and_b32 s3, s37, 0x3fff
	v_lshlrev_b32_e32 v3, 2, v69
	s_bitset1_b32 s3, 14
	v_and_b32_e32 v4, 1, v4
	ds_bpermute_b32 v3, v3, v116
	s_and_b32 s7, s2, 0xffff
	s_lshl_b32 s34, s3, 16
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v68, 1, v6
	v_cmp_eq_u32_e32 vcc, 1, v4
	v_readfirstlane_b32 s35, v37
	v_sub_u32_e32 v2, v2, v34
	s_or_b32 s21, s7, s34
	v_cndmask_b32_e32 v4, v35, v68, vcc
	s_mov_b32 m0, s35
	v_ashrrev_i32_e32 v2, 3, v2
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_lshrrev_b64 v[4:5], v69, exec
	v_add_u32_e32 v2, v2, v253
	v_mov_b32_e32 v100, v224
	s_lshl_b32 s30, s37, 6
	v_add_u32_e32 v66, 0, v98
	v_and_b32_e32 v4, 1, v4
	v_lshlrev_b32_e32 v224, 2, v2
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v3
	v_cmp_eq_u32_e32 vcc, 1, v4
	v_readfirstlane_b32 s36, v66
	s_ashr_i32 s31, s30, 31
	ds_bpermute_b32 v5, v224, v208
	s_lshl_b32 s6, s40, 6
	v_cndmask_b32_e32 v3, v35, v70, vcc
	s_mov_b32 m0, s36
	s_lshl_b64 s[14:15], s[30:31], 1
	buffer_load_dwordx4 v3, s[20:23], 0 offen lds
	s_add_u32 s20, s20, s14
	v_cmp_ne_u32_e32 vcc, 1, v36
	ds_bpermute_b32 v6, v224, v116
	s_addc_u32 s7, s2, s15
	v_lshrrev_b64 v[2:3], v2, vcc
	v_add_u32_e32 v4, 0x4000, v37
	v_and_b32_e32 v2, 1, v2
	s_and_b32 s2, s7, 0xffff
	s_or_b32 s21, s2, s34
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v5, 1, v5
	v_cmp_eq_u32_e64 s[2:3], 1, v2
	v_readfirstlane_b32 s12, v4
	s_mov_b32 m0, s12
	v_cndmask_b32_e64 v2, v35, v5, s[2:3]
	v_add_u32_e32 v3, 0x4000, v66
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v6
	v_lshlrev_b32_e32 v26, 8, v219
	v_cndmask_b32_e64 v2, v35, v2, s[2:3]
	v_readfirstlane_b32 s2, v3
	v_or_b32_e32 v71, v252, v26
	s_mov_b32 m0, s2
	v_add_u32_e32 v72, 0, v71
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b128 v[2:5], v72
	ds_read_b128 v[18:21], v72 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[134:137], 0
	v_or_b32_e32 v73, v222, v26
	v_add_u32_e32 v74, 0, v73
	ds_read_b128 v[22:25], v74
	ds_read_b128 v[38:41], v74 offset:8192
	v_or_b32_e32 v75, v218, v26
	v_add_u32_e32 v76, 0, v75
	v_or_b32_e32 v77, v29, v26
	v_add_u32_e32 v78, 0, v77
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[130:133], v[2:17]
	ds_read_b128 v[22:25], v76
	ds_read_b128 v[42:45], v76 offset:8192
	scratch_store_dword off, v29, off offset:4 ; 4-byte Folded Spill
	v_or_b32_e32 v79, v28, v26
	v_add_u32_e32 v80, 0, v79
	v_or_b32_e32 v81, v27, v26
	v_add_u32_e32 v82, 0, v81
	v_or_b32_e32 v83, v228, v26
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[142:145], v[2:17]
	ds_read_b128 v[22:25], v78
	ds_read_b128 v[46:49], v78 offset:8192
	v_add_u32_e32 v84, 0, v83
	v_or_b32_e32 v85, v227, v26
	v_add_u32_e32 v86, 0, v85
	v_mov_b32_e32 v104, v28
	v_mov_b32_e32 v103, v27
	s_movk_i32 s2, 0x60
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[138:141], v[2:17]
	ds_read_b128 v[22:25], v80
	ds_read_b128 v[50:53], v80 offset:8192
	s_mov_b32 s44, s16
	s_mov_b32 s46, s22
	s_mov_b32 s47, s23
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[150:153], v[2:17]
	ds_read_b128 v[22:25], v82
	ds_read_b128 v[54:57], v82 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[146:149], v[2:17]
	ds_read_b128 v[22:25], v84
	ds_read_b128 v[58:61], v84 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[158:161], v[2:17]
	ds_read_b128 v[22:25], v86
	ds_read_b128 v[62:65], v86 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[154:157], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[134:137], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[38:41], v[130:133], v[18:33]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_nop 7
	s_nop 1
	v_max_f32_e32 v38, v3, v3
	v_max_f32_e32 v39, v2, v2
	v_max_f32_e32 v38, v39, v38
	v_max3_f32 v38, v38, v4, v5
	v_max3_f32 v38, v38, v6, v7
	v_max3_f32 v38, v38, v8, v9
	v_max3_f32 v38, v38, v10, v11
	v_mfma_f32_32x32x16_f16 v[18:33], v[42:45], v[142:145], v[18:33]
	v_max3_f32 v38, v38, v12, v13
	v_max3_f32 v38, v38, v14, v15
	v_max3_f32 v38, v38, v16, v17
	v_mfma_f32_32x32x16_f16 v[18:33], v[46:49], v[138:141], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[50:53], v[150:153], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[54:57], v[146:149], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[58:61], v[158:161], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[62:65], v[154:157], v[18:33]
	s_nop 7
	s_nop 3
	v_max3_f32 v38, v38, v18, v19
	v_max3_f32 v40, v38, v20, v21
	v_lshlrev_b32_e32 v38, 1, v0
	v_bitop3_b32 v105, v34, v38, s2 bitop3:0x78
	v_sub_u32_e32 v38, v105, v34
	v_ashrrev_i32_e32 v38, 3, v38
	v_add_u32_e32 v42, v38, v253
	v_mad_u64_u32 v[206:207], s[2:3], s40, v1, v[34:35]
	v_lshlrev_b32_e32 v41, 2, v42
	ds_bpermute_b32 v1, v41, v206
	v_lshrrev_b64 v[38:39], v42, exec
	v_add_u32_e32 v39, 0x8000, v37
	s_and_b32 s2, s40, 0x3fff
	v_lshl_add_u32 v43, s40, 5, v206
	v_readfirstlane_b32 s12, v39
	s_bitset1_b32 s2, 14
	ds_bpermute_b32 v39, v41, v43
	v_and_b32_e32 v38, 1, v38
	s_and_b32 s3, s17, 0xffff
	s_lshl_b32 s31, s2, 16
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v1, 1, v1
	s_or_b32 s45, s3, s31
	v_cmp_eq_u32_e64 s[2:3], 1, v38
	s_mov_b32 m0, s12
	s_add_u32 s20, s20, s14
	v_cndmask_b32_e64 v38, v35, v1, s[2:3]
	buffer_load_dwordx4 v38, s[44:47], 0 offen lds
	v_add_u32_e32 v38, 0x8000, v66
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v41, 1, v39
	v_readfirstlane_b32 s12, v38
	s_addc_u32 s21, s7, s15
	s_ashr_i32 s7, s6, 31
	v_cndmask_b32_e64 v38, v35, v41, s[2:3]
	s_mov_b32 m0, s12
	v_cmp_lt_u32_e64 s[2:3], 2, v36
	s_lshl_b64 s[12:13], s[6:7], 1
	buffer_load_dwordx4 v38, s[44:47], 0 offen lds
	v_lshrrev_b64 v[38:39], v67, s[2:3]
	s_add_u32 s18, s16, s12
	v_and_b32_e32 v38, 1, v38
	s_addc_u32 s19, s17, s13
	s_and_b32 s6, s21, 0xffff
	s_or_b32 s21, s6, s34
	v_cmp_eq_u32_e64 s[6:7], 1, v38
	s_mov_b32 m0, s35
	v_add_u32_e32 v37, 0xc000, v37
	v_cndmask_b32_e64 v38, v35, v68, s[6:7]
	buffer_load_dwordx4 v38, s[20:23], 0 offen lds
	v_lshrrev_b64 v[38:39], v69, s[2:3]
	v_and_b32_e32 v38, 1, v38
	v_cmp_eq_u32_e64 s[6:7], 1, v38
	s_mov_b32 m0, s36
	scratch_store_dword off, v42, off offset:64 ; 4-byte Folded Spill
	v_cndmask_b32_e64 v38, v35, v70, s[6:7]
	buffer_load_dwordx4 v38, s[20:23], 0 offen lds
	v_lshrrev_b64 v[38:39], v42, vcc
	v_and_b32_e32 v38, 1, v38
	v_readfirstlane_b32 s6, v37
	s_and_b32 s7, s19, 0xffff
	v_cmp_eq_u32_e32 vcc, 1, v38
	s_or_b32 s21, s7, s31
	s_mov_b32 s20, s18
	v_cndmask_b32_e32 v1, v35, v1, vcc
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(4)
	s_barrier
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	v_cndmask_b32_e32 v1, v35, v41, vcc
	v_add_u32_e32 v35, 0xc000, v66
	s_nop 0
	v_readfirstlane_b32 s6, v35
	s_mov_b32 m0, s6
	v_mov_b32_e32 v35, 0xff800000
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	v_max3_f32 v1, v40, v22, v23
	v_max3_f32 v1, v1, v24, v25
	v_max3_f32 v1, v1, v26, v27
	v_max3_f32 v1, v1, v28, v29
	v_max3_f32 v1, v1, v30, v31
	v_max3_f32 v1, v1, v32, v33
	v_mov_b32_e32 v37, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v37
	v_max3_f32 v231, v1, v37, v35
	v_mul_f32_e32 v37, 0xbe0293ee, v231
	v_fmamk_f32 v1, v2, 0x3e0293ee, v37
	v_fmamk_f32 v2, v3, 0x3e0293ee, v37
	v_fmamk_f32 v3, v4, 0x3e0293ee, v37
	v_fmamk_f32 v4, v5, 0x3e0293ee, v37
	v_fmamk_f32 v5, v6, 0x3e0293ee, v37
	v_fmamk_f32 v6, v7, 0x3e0293ee, v37
	v_fmamk_f32 v7, v8, 0x3e0293ee, v37
	v_fmamk_f32 v8, v9, 0x3e0293ee, v37
	v_fmamk_f32 v9, v10, 0x3e0293ee, v37
	v_fmamk_f32 v10, v11, 0x3e0293ee, v37
	v_fmamk_f32 v11, v12, 0x3e0293ee, v37
	v_fmamk_f32 v12, v13, 0x3e0293ee, v37
	v_fmamk_f32 v13, v14, 0x3e0293ee, v37
	v_fmamk_f32 v14, v15, 0x3e0293ee, v37
	v_fmamk_f32 v15, v16, 0x3e0293ee, v37
	v_fmamk_f32 v16, v17, 0x3e0293ee, v37
	s_movk_i32 s6, 0x1ff
	v_add_u32_e32 v17, 0xff, v0
	v_cmp_gt_u32_e32 vcc, s6, v17
	s_movk_i32 s6, 0x1fe
	s_add_i32 s20, 0, 0x4000
	v_fmamk_f32 v18, v18, 0x3e0293ee, v37
	v_fmamk_f32 v19, v19, 0x3e0293ee, v37
	v_fmamk_f32 v20, v20, 0x3e0293ee, v37
	v_fmamk_f32 v21, v21, 0x3e0293ee, v37
	v_fmamk_f32 v22, v22, 0x3e0293ee, v37
	v_fmamk_f32 v23, v23, 0x3e0293ee, v37
	v_fmamk_f32 v24, v24, 0x3e0293ee, v37
	v_fmamk_f32 v25, v25, 0x3e0293ee, v37
	v_fmamk_f32 v26, v26, 0x3e0293ee, v37
	v_fmamk_f32 v27, v27, 0x3e0293ee, v37
	v_fmamk_f32 v28, v28, 0x3e0293ee, v37
	v_fmamk_f32 v29, v29, 0x3e0293ee, v37
	v_fmamk_f32 v30, v30, 0x3e0293ee, v37
	v_fmamk_f32 v31, v31, 0x3e0293ee, v37
	v_fmamk_f32 v32, v32, 0x3e0293ee, v37
	v_fmac_f32_e32 v37, 0x3e0293ee, v33
	v_readfirstlane_b32 s23, v36
	v_cmp_lt_u32_e64 s[6:7], s6, v17
	v_add_u32_e32 v17, s20, v71
	v_add_u32_e32 v33, s20, v73
	v_add_u32_e32 v36, s20, v75
	v_add_u32_e32 v38, s20, v77
	v_add_u32_e32 v39, s20, v79
	v_add_u32_e32 v40, s20, v81
	v_add_u32_e32 v41, s20, v83
	v_add_u32_e32 v42, s20, v85
	ds_read_b128 v[68:71], v72 offset:16384
	ds_read_b128 v[202:205], v74 offset:16384
	ds_read_b128 v[198:201], v76 offset:16384
	ds_read_b128 v[194:197], v78 offset:16384
	ds_read_b128 v[190:193], v80 offset:16384
	ds_read_b128 v[94:97], v82 offset:16384
	ds_read_b128 v[90:93], v84 offset:16384
	ds_read_b128 v[86:89], v86 offset:16384
	ds_read_b128 v[82:85], v17 offset:8192
	ds_read_b128 v[186:189], v33 offset:8192
	ds_read_b128 v[182:185], v36 offset:8192
	ds_read_b128 v[178:181], v38 offset:8192
	ds_read_b128 v[174:177], v39 offset:8192
	ds_read_b128 v[170:173], v40 offset:8192
	ds_read_b128 v[166:169], v41 offset:8192
	ds_read_b128 v[162:165], v42 offset:8192
	v_fmac_f32_e32 v35, 0xbe0293ee, v231
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], s[6:7]
	s_cbranch_execz .LBB0_7
; %bb.6:
	s_barrier
.LBB0_7:
	s_or_b64 exec, exec, s[20:21]
	v_exp_f32_e32 v211, v1
	v_exp_f32_e32 v212, v2
	v_exp_f32_e32 v209, v3
	v_exp_f32_e32 v213, v4
	v_exp_f32_e32 v214, v5
	v_exp_f32_e32 v215, v6
	v_exp_f32_e32 v216, v7
	v_exp_f32_e32 v217, v8
	v_exp_f32_e32 v254, v9
	v_exp_f32_e32 v255, v10
	v_exp_f32_e32 v1, v11
	v_exp_f32_e32 v250, v12
	v_exp_f32_e32 v223, v13
	v_exp_f32_e32 v210, v14
	v_exp_f32_e32 v232, v15
	v_exp_f32_e32 v234, v16
	v_exp_f32_e32 v233, v18
	v_exp_f32_e32 v236, v19
	v_exp_f32_e32 v235, v20
	v_exp_f32_e32 v240, v21
	v_exp_f32_e32 v239, v22
	v_exp_f32_e32 v238, v23
	v_exp_f32_e32 v237, v24
	v_exp_f32_e32 v242, v25
	v_exp_f32_e32 v241, v26
	v_exp_f32_e32 v243, v27
	v_exp_f32_e32 v245, v28
	v_exp_f32_e32 v248, v29
	v_exp_f32_e32 v244, v30
	v_exp_f32_e32 v247, v31
	v_exp_f32_e32 v246, v32
	v_exp_f32_e32 v249, v37
	v_exp_f32_e32 v229, v35
	v_mov_b32_e32 v17, 0
	v_lshlrev_b32_e32 v33, 7, v219
	v_and_b32_e32 v220, 32, v225
	v_and_b32_e32 v230, 64, v225
	v_and_b32_e32 v251, 16, v0
	s_cmp_lt_u32 s23, 4
	v_lshlrev_b32_e32 v72, 2, v0
	scratch_store_dword off, v43, off offset:56 ; 4-byte Folded Spill
	scratch_store_dword off, v98, off offset:52 ; 4-byte Folded Spill
	scratch_store_dword off, v33, off       ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_12
; %bb.8:                                ; %.lr.ph
	v_and_b32_e32 v219, 12, v72
	v_or_b32_e32 v2, v220, v219
	v_and_or_b32 v3, v99, 3, v100
	v_bitop3_b32 v2, v2, v230, 64 bitop3:0x36
	s_movk_i32 s6, 0x60
	v_lshlrev_b32_e32 v3, 7, v3
	v_bitop3_b32 v4, v219, v220, 32 bitop3:0x36
	scratch_store_dword off, v2, off offset:36 ; 4-byte Folded Spill
	v_bitop3_b32 v2, v225, v219, s6 bitop3:0x4e
	scratch_store_dword off, v100, off offset:68 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off offset:40 ; 4-byte Folded Spill
	scratch_store_dword off, v228, off offset:8 ; 4-byte Folded Spill
	scratch_store_dword off, v227, off offset:12 ; 4-byte Folded Spill
	v_add_u32_e32 v2, v101, v34
	v_lshlrev_b32_e32 v225, 1, v3
	v_lshlrev_b32_e32 v3, 1, v4
	scratch_store_dword off, v2, off offset:44 ; 4-byte Folded Spill
	scratch_store_dword off, v222, off offset:16 ; 4-byte Folded Spill
	scratch_store_dword off, v252, off offset:20 ; 4-byte Folded Spill
	scratch_store_dword off, v218, off offset:24 ; 4-byte Folded Spill
	scratch_store_dword off, v104, off offset:32 ; 4-byte Folded Spill
	scratch_store_dword off, v103, off offset:28 ; 4-byte Folded Spill
	scratch_store_dword off, v3, off offset:48 ; 4-byte Folded Spill
	scratch_load_dword v252, off, off offset:52 ; 4-byte Folded Reload
	s_nop 0
	scratch_load_dword v218, off, off offset:56 ; 4-byte Folded Reload
	s_add_i32 s35, s23, -3
	s_add_u32 s6, s24, s26
	s_addc_u32 s7, s25, s27
	s_add_u32 s6, s6, s28
	s_addc_u32 s7, s7, s29
	s_mul_i32 s17, s30, 6
	s_lshl_b64 s[6:7], s[6:7], 1
	s_mul_hi_i32 s16, s30, 6
	s_add_u32 s6, s17, s6
	s_addc_u32 s7, s16, s7
	s_add_u32 s25, s4, s6
	v_mov_b32_e32 v2, 0
	s_addc_u32 s26, s5, s7
	s_mov_b32 s20, 0
	s_add_i32 s21, 0, 0x8000
	s_add_i32 s22, 0, 0xc000
	v_mov_b32_e32 v226, 1.0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	s_mov_b32 s27, 0x3e0293ee
	s_mov_b32 s29, 0
	s_mov_b32 s28, 0
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
	v_lshlrev_b32_e32 v222, 1, v102
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[68:71], v[134:137], 0
	s_mov_b64 s[16:17], s[18:19]
	s_mov_b32 s37, s21
	s_mov_b32 s21, s22
	v_mov_b32_e32 v98, v226
	v_mov_b32_e32 v207, v231
	s_mov_b32 s30, s20
	s_setprio 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[202:205], v[130:133], v[66:81]
	v_add_f32_e32 v99, v211, v212
	v_add_f32_e32 v99, v99, v209
	v_add_f32_e32 v99, v99, v213
	v_add_f32_e32 v99, v99, v214
	v_add_f32_e32 v99, v99, v215
	v_add_f32_e32 v99, v99, v216
	v_add_f32_e32 v99, v99, v217
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[198:201], v[142:145], v[66:81]
	v_add_f32_e32 v99, v99, v254
	v_add_f32_e32 v99, v99, v255
	v_add_f32_e32 v99, v99, v1
	v_add_f32_e32 v99, v99, v250
	v_add_f32_e32 v99, v99, v223
	v_add_f32_e32 v99, v99, v210
	v_add_f32_e32 v99, v99, v232
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[194:197], v[138:141], v[66:81]
	v_add_f32_e32 v99, v99, v234
	v_add_f32_e32 v99, v99, v233
	v_add_f32_e32 v99, v99, v236
	v_add_f32_e32 v99, v99, v235
	v_add_f32_e32 v99, v99, v240
	v_add_f32_e32 v99, v99, v239
	v_add_f32_e32 v99, v99, v238
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[190:193], v[150:153], v[66:81]
	v_add_f32_e32 v99, v99, v237
	v_add_f32_e32 v99, v99, v242
	v_add_f32_e32 v99, v99, v241
	v_add_f32_e32 v99, v99, v243
	v_add_f32_e32 v99, v99, v245
	v_add_f32_e32 v99, v99, v248
	v_add_f32_e32 v99, v99, v244
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[146:149], v[66:81]
	v_add_f32_e32 v99, v99, v247
	v_add_f32_e32 v99, v99, v246
	v_add_f32_e32 v99, v99, v249
	v_mov_b32_e32 v100, v99
	s_nop 1
	v_permlane32_swap_b32_e32 v99, v100
	v_add_f32_e32 v226, v99, v100
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[158:161], v[66:81]
	v_mul_f32_e32 v2, v2, v229
	v_mul_f32_e32 v3, v3, v229
	v_mul_f32_e32 v4, v4, v229
	v_mul_f32_e32 v5, v5, v229
	v_mul_f32_e32 v6, v6, v229
	v_mul_f32_e32 v7, v7, v229
	v_mul_f32_e32 v8, v8, v229
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[154:157], v[66:81]
	v_mul_f32_e32 v9, v9, v229
	v_mul_f32_e32 v10, v10, v229
	v_mul_f32_e32 v11, v11, v229
	v_mul_f32_e32 v12, v12, v229
	v_mul_f32_e32 v13, v13, v229
	v_mul_f32_e32 v14, v14, v229
	v_mul_f32_e32 v15, v15, v229
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[134:137], 0
	v_mul_f32_e32 v16, v16, v229
	v_mul_f32_e32 v17, v17, v229
	v_mul_f32_e32 v34, v34, v229
	v_mul_f32_e32 v35, v35, v229
	v_mul_f32_e32 v36, v36, v229
	v_mul_f32_e32 v37, v37, v229
	v_mul_f32_e32 v38, v38, v229
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[130:133], v[82:97]
	v_mul_f32_e32 v39, v39, v229
	v_mul_f32_e32 v40, v40, v229
	v_mul_f32_e32 v41, v41, v229
	v_mul_f32_e32 v42, v42, v229
	v_mul_f32_e32 v43, v43, v229
	v_mul_f32_e32 v44, v44, v229
	v_mul_f32_e32 v45, v45, v229
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[142:145], v[82:97]
	v_mul_f32_e32 v46, v46, v229
	v_mul_f32_e32 v47, v47, v229
	v_mul_f32_e32 v48, v48, v229
	v_mul_f32_e32 v49, v49, v229
	v_mul_f32_e32 v50, v50, v229
	v_mul_f32_e32 v51, v51, v229
	v_mul_f32_e32 v52, v52, v229
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[138:141], v[82:97]
	v_mul_f32_e32 v53, v53, v229
	v_mul_f32_e32 v54, v54, v229
	v_mul_f32_e32 v55, v55, v229
	v_mul_f32_e32 v56, v56, v229
	v_mul_f32_e32 v57, v57, v229
	v_mul_f32_e32 v58, v58, v229
	v_mul_f32_e32 v59, v59, v229
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[150:153], v[82:97]
	v_mul_f32_e32 v60, v60, v229
	v_mul_f32_e32 v61, v61, v229
	v_mul_f32_e32 v62, v62, v229
	v_mul_f32_e32 v63, v63, v229
	v_mul_f32_e32 v64, v64, v229
	v_mul_f32_e32 v65, v65, v229
	v_mul_f32_e32 v18, v18, v229
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[146:149], v[82:97]
	v_mul_f32_e32 v19, v19, v229
	v_mul_f32_e32 v20, v20, v229
	v_mul_f32_e32 v21, v21, v229
	v_mul_f32_e32 v22, v22, v229
	v_mul_f32_e32 v23, v23, v229
	v_mul_f32_e32 v24, v24, v229
	v_mul_f32_e32 v25, v25, v229
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[158:161], v[82:97]
	v_mul_f32_e32 v26, v26, v229
	v_mul_f32_e32 v27, v27, v229
	v_mul_f32_e32 v28, v28, v229
	v_mul_f32_e32 v29, v29, v229
	v_mul_f32_e32 v30, v30, v229
	v_mul_f32_e32 v31, v31, v229
	v_mul_f32_e32 v32, v32, v229
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[154:157], v[82:97]
	v_mul_f32_e32 v33, v33, v229
	v_fmac_f32_e32 v226, v98, v229
	v_mov_b32_e32 v229, v102
	v_cvt_pk_f16_f32 v102, v233, v236
	v_mov_b32_e32 v228, v105
	v_cvt_pk_f16_f32 v105, v237, v242
	v_mov_b32_e32 v221, v101
	v_cvt_pk_f16_f32 v101, v246, v249
	v_cvt_pk_f16_f32 v110, v211, v212
	v_cvt_pk_f16_f32 v111, v209, v213
	v_cvt_pk_f16_f32 v112, v214, v215
	v_cvt_pk_f16_f32 v113, v216, v217
	v_cvt_pk_f16_f32 v106, v254, v255
	v_cvt_pk_f16_f32 v107, v1, v250
	v_cvt_pk_f16_f32 v108, v223, v210
	v_cvt_pk_f16_f32 v109, v232, v234
	v_cvt_pk_f16_f32 v103, v235, v240
	v_cvt_pk_f16_f32 v104, v239, v238
	v_cvt_pk_f16_f32 v98, v241, v243
	v_cvt_pk_f16_f32 v99, v245, v248
	v_cvt_pk_f16_f32 v100, v244, v247
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s18, s16, s12
	s_addc_u32 s19, s17, s13
	s_add_i32 s4, s29, 1
	s_cmp_lt_i32 s4, 2
	s_cselect_b32 s36, s4, 0
	ds_bpermute_b32 v115, v224, v208
	s_lshl_b32 s24, s36, 14
	v_mov_b32_e32 v227, v116
	ds_bpermute_b32 v116, v224, v116
	s_add_i32 s20, s24, 0
	v_add_u32_e32 v1, s20, v222
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v114, s20, v252
	s_and_b32 s4, s26, 0xffff
	v_readfirstlane_b32 s22, v1
	s_or_b32 s5, s4, s34
	s_mov_b32 s4, s25
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v115, 1, v115
	s_mov_b32 m0, s22
	v_readfirstlane_b32 s22, v114
	buffer_load_dwordx4 v115, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v116
	s_mov_b32 m0, s22
	v_lshlrev_b32_e32 v114, 1, v220
	buffer_load_dwordx4 v1, s[4:7], 0 offen lds
	v_lshl_add_u32 v1, v219, 1, s37
	v_lshlrev_b32_e32 v115, 1, v230
	v_add3_u32 v1, v1, v114, v115
	v_lshlrev_b32_e32 v114, 1, v251
	v_add3_u32 v1, v1, v114, v225
	ds_read_b64_tr_b16 v[198:199], v1
	ds_read_b64_tr_b16 v[200:201], v1 offset:2048
	ds_read_b64_tr_b16 v[202:203], v1 offset:4096
	ds_read_b64_tr_b16 v[204:205], v1 offset:6144
	ds_read_b64_tr_b16 v[210:211], v1 offset:8192
	ds_read_b64_tr_b16 v[212:213], v1 offset:10240
	ds_read_b64_tr_b16 v[194:195], v1 offset:12288
	ds_read_b64_tr_b16 v[196:197], v1 offset:14336
	scratch_load_dword v1, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v1, s37, v1, v115
	v_add3_u32 v1, v1, v114, v225
	ds_read_b64_tr_b16 v[190:191], v1
	ds_read_b64_tr_b16 v[192:193], v1 offset:2048
	ds_read_b64_tr_b16 v[186:187], v1 offset:4096
	ds_read_b64_tr_b16 v[188:189], v1 offset:6144
	ds_read_b64_tr_b16 v[182:183], v1 offset:8192
	ds_read_b64_tr_b16 v[184:185], v1 offset:10240
	ds_read_b64_tr_b16 v[178:179], v1 offset:12288
	ds_read_b64_tr_b16 v[180:181], v1 offset:14336
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v1, 1, s37
	v_add3_u32 v1, v1, v114, v225
	ds_read_b64_tr_b16 v[174:175], v1
	ds_read_b64_tr_b16 v[176:177], v1 offset:2048
	ds_read_b64_tr_b16 v[170:171], v1 offset:4096
	ds_read_b64_tr_b16 v[172:173], v1 offset:6144
	ds_read_b64_tr_b16 v[166:167], v1 offset:8192
	ds_read_b64_tr_b16 v[168:169], v1 offset:10240
	ds_read_b64_tr_b16 v[162:163], v1 offset:12288
	ds_read_b64_tr_b16 v[164:165], v1 offset:14336
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v1, 1, s37
	v_add3_u32 v1, v1, v114, v225
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
	v_mfma_f32_32x32x16_f16 v[2:17], v[198:201], v[110:113], v[2:17]
	s_barrier
	s_setprio 0
	v_mfma_f32_32x32x16_f16 v[34:49], v[190:193], v[110:113], v[34:49]
	v_max_f32_e32 v1, v67, v67
	v_max_f32_e32 v198, v66, v66
	v_max_f32_e32 v1, v198, v1
	v_max3_f32 v1, v1, v68, v69
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_max3_f32 v1, v1, v74, v75
	v_mfma_f32_32x32x16_f16 v[50:65], v[174:177], v[110:113], v[50:65]
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[126:129], v[110:113], v[18:33]
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_mfma_f32_32x32x16_f16 v[2:17], v[202:205], v[106:109], v[2:17]
	v_mfma_f32_32x32x16_f16 v[34:49], v[186:189], v[106:109], v[34:49]
	v_mov_b32_e32 v186, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v186
	v_max3_f32 v231, v207, v1, v186
	v_mul_f32_e32 v186, 0x3e0293ee, v231
	v_fma_f32 v1, v66, s27, -v186
	v_fma_f32 v66, v67, s27, -v186
	v_mfma_f32_32x32x16_f16 v[50:65], v[170:173], v[106:109], v[50:65]
	v_fma_f32 v67, v68, s27, -v186
	v_fma_f32 v68, v69, s27, -v186
	v_fma_f32 v69, v70, s27, -v186
	v_fma_f32 v70, v71, s27, -v186
	v_fma_f32 v71, v72, s27, -v186
	v_fma_f32 v72, v73, s27, -v186
	v_fma_f32 v73, v74, s27, -v186
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[18:33], v[122:125], v[106:109], v[18:33]
	v_fma_f32 v74, v75, s27, -v186
	v_fma_f32 v75, v76, s27, -v186
	v_fma_f32 v76, v77, s27, -v186
	v_fma_f32 v77, v78, s27, -v186
	v_fma_f32 v78, v79, s27, -v186
	v_fma_f32 v79, v80, s27, -v186
	v_fma_f32 v80, v81, s27, -v186
	v_mfma_f32_32x32x16_f16 v[2:17], v[210:213], v[102:105], v[2:17]
	v_fma_f32 v81, v82, s27, -v186
	v_fma_f32 v82, v83, s27, -v186
	v_fma_f32 v83, v84, s27, -v186
	v_fma_f32 v84, v85, s27, -v186
	v_fma_f32 v85, v86, s27, -v186
	v_fma_f32 v86, v87, s27, -v186
	v_fma_f32 v87, v88, s27, -v186
	v_mfma_f32_32x32x16_f16 v[34:49], v[182:185], v[102:105], v[34:49]
	v_fma_f32 v88, v89, s27, -v186
	v_fma_f32 v89, v90, s27, -v186
	v_fma_f32 v90, v91, s27, -v186
	v_fma_f32 v91, v92, s27, -v186
	v_fma_f32 v92, v93, s27, -v186
	v_fma_f32 v93, v94, s27, -v186
	v_fma_f32 v94, v95, s27, -v186
	v_mfma_f32_32x32x16_f16 v[50:65], v[166:169], v[102:105], v[50:65]
	v_fma_f32 v95, v96, s27, -v186
	v_fma_f32 v96, v97, s27, -v186
	v_exp_f32_e32 v212, v66
	v_fma_f32 v66, v207, s27, -v186
	v_exp_f32_e32 v211, v1
	v_exp_f32_e32 v209, v67
	v_exp_f32_e32 v213, v68
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[118:121], v[102:105], v[18:33]
	v_exp_f32_e32 v214, v69
	v_exp_f32_e32 v215, v70
	v_exp_f32_e32 v216, v71
	v_exp_f32_e32 v217, v72
	v_exp_f32_e32 v254, v73
	v_exp_f32_e32 v255, v74
	v_exp_f32_e32 v1, v75
	v_mfma_f32_32x32x16_f16 v[2:17], v[194:197], v[98:101], v[2:17]
	v_exp_f32_e32 v250, v76
	v_exp_f32_e32 v223, v77
	v_exp_f32_e32 v210, v78
	v_exp_f32_e32 v232, v79
	v_exp_f32_e32 v234, v80
	v_exp_f32_e32 v233, v81
	v_exp_f32_e32 v236, v82
	v_mfma_f32_32x32x16_f16 v[34:49], v[178:181], v[98:101], v[34:49]
	v_exp_f32_e32 v235, v83
	v_exp_f32_e32 v240, v84
	v_exp_f32_e32 v239, v85
	v_exp_f32_e32 v238, v86
	v_exp_f32_e32 v237, v87
	v_exp_f32_e32 v242, v88
	v_exp_f32_e32 v241, v89
	v_mfma_f32_32x32x16_f16 v[50:65], v[162:165], v[98:101], v[50:65]
	v_exp_f32_e32 v243, v90
	v_mov_b32_e32 v102, v229
	v_exp_f32_e32 v245, v91
	v_exp_f32_e32 v248, v92
	v_exp_f32_e32 v244, v93
	v_exp_f32_e32 v247, v94
	v_exp_f32_e32 v246, v95
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[114:117], v[98:101], v[18:33]
	v_exp_f32_e32 v249, v96
	v_exp_f32_e32 v229, v66
	v_mov_b32_e32 v105, v228
	v_mov_b32_e32 v116, v227
	v_mov_b32_e32 v101, v221
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	scratch_load_dword v68, off, off offset:44 ; 4-byte Folded Reload
	s_lshl_b32 s4, s29, 14
	s_add_i32 s4, s4, 0
	s_add_i32 s22, s4, 0x8000
	v_lshlrev_b32_e32 v66, 1, v105
	v_lshlrev_b32_e32 v67, 1, v101
	v_add3_u32 v66, s4, v66, v67
	v_lshl_add_u32 v67, v102, 1, s22
	v_sub_u32_e32 v69, v66, v67
	v_add_u32_e32 v69, 0x8000, v69
	v_ashrrev_i32_e32 v70, 31, v69
	v_lshrrev_b32_e32 v70, 28, v70
	v_add_u32_e32 v69, v69, v70
	v_ashrrev_i32_e32 v69, 4, v69
	v_add_lshl_u32 v69, v69, v253, 2
	ds_bpermute_b32 v69, v69, v206
	s_and_b32 s4, s19, 0xffff
	v_readfirstlane_b32 s29, v67
	s_or_b32 s5, s4, s31
	s_mov_b32 s4, s18
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v69, 1, v69
	s_mov_b32 m0, s29
	scratch_load_dword v67, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshl_add_u32 v68, v68, 1, s22
	v_add_u32_e32 v68, 0x2000, v68
	v_sub_u32_e32 v66, v66, v68
	v_add_u32_e32 v66, 0xa000, v66
	v_ashrrev_i32_e32 v70, 31, v66
	v_lshrrev_b32_e32 v70, 28, v70
	v_add_u32_e32 v66, v66, v70
	v_ashrrev_i32_e32 v66, 4, v66
	v_add_lshl_u32 v66, v66, v253, 2
	ds_bpermute_b32 v66, v66, v218
	v_readfirstlane_b32 s29, v68
	buffer_load_dwordx4 v69, s[4:7], 0 offen lds
	s_mov_b32 m0, s29
	scratch_load_dword v68, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	buffer_load_dwordx4 v66, s[4:7], 0 offen lds
	scratch_load_dword v66, off, off        ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	v_add3_u32 v72, s30, v68, v66
	scratch_load_dword v68, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v73, s30, v68, v66
	scratch_load_dword v68, off, off offset:4 ; 4-byte Folded Reload
	v_add3_u32 v67, s30, v67, v66
	s_waitcnt vmcnt(0)
	v_add3_u32 v74, s30, v68, v66
	scratch_load_dword v68, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v75, s30, v68, v66
	scratch_load_dword v68, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v76, s30, v68, v66
	scratch_load_dword v68, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v77, s30, v68, v66
	scratch_load_dword v68, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v66, s30, v68, v66
	ds_read_b128 v[68:71], v67
	ds_read_b128 v[82:85], v67 offset:8192
	ds_read_b128 v[202:205], v72
	ds_read_b128 v[186:189], v72 offset:8192
	ds_read_b128 v[198:201], v73
	ds_read_b128 v[182:185], v73 offset:8192
	ds_read_b128 v[194:197], v74
	ds_read_b128 v[178:181], v74 offset:8192
	ds_read_b128 v[190:193], v75
	ds_read_b128 v[174:177], v75 offset:8192
	ds_read_b128 v[94:97], v76
	ds_read_b128 v[170:173], v76 offset:8192
	ds_read_b128 v[90:93], v77
	ds_read_b128 v[166:169], v77 offset:8192
	ds_read_b128 v[86:89], v66
	ds_read_b128 v[162:165], v66 offset:8192
	; sched_barrier mask(0x00000000)
	s_add_i32 s28, s28, 1
	s_add_u32 s25, s25, s14
	s_addc_u32 s26, s26, s15
	s_cmp_lt_i32 s28, s35
	s_mov_b32 s29, s36
	s_barrier
	s_cbranch_scc1 .LBB0_9
; %bb.10:                               ; %Flow797
	scratch_load_dword v224, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v227, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v228, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v208, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v253, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v218, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v252, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v222, off, off offset:16 ; 4-byte Folded Reload
	v_and_b32_e32 v219, 31, v0
	v_lshrrev_b32_e32 v221, 1, v0
	v_lshlrev_b32_e32 v225, 3, v0
	v_lshlrev_b32_e32 v72, 2, v0
	scratch_load_dword v207, off, off offset:60 ; 4-byte Folded Reload
	s_and_saveexec_b64 s[4:5], vcc
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
	v_mov_b32_e32 v194, 1.0
	s_branch .LBB0_23
.LBB0_12:
	s_mov_b32 s20, 0
	s_add_i32 s21, 0, 0x8000
	s_add_i32 s22, 0, 0xc000
	v_mov_b32_e32 v226, 1.0
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
	s_mov_b32 s24, 0
	v_mov_b32_e32 v224, v100
	v_mov_b32_e32 v208, v103
	v_mov_b32_e32 v253, v104
	scratch_load_dword v207, off, off offset:60 ; 4-byte Folded Reload
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_14
.LBB0_13:
	s_barrier
.LBB0_14:
	s_or_b64 exec, exec, s[4:5]
	s_cmp_eq_u32 s23, 1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lg_u32 s23, 1
	s_cselect_b64 s[18:19], -1, 0
	v_mov_b32_e32 v66, 0
	s_and_b64 vcc, exec, s[4:5]
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	s_cbranch_vccnz .LBB0_16
; %bb.15:
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[98:113], v[68:71], v[134:137], 0
	v_mfma_f32_32x32x16_f16 v[114:129], v[82:85], v[134:137], 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[98:113], v[202:205], v[130:133], v[98:113]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[114:129], v[186:189], v[130:133], v[114:129]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[98:113], v[198:201], v[142:145], v[98:113]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[114:129], v[182:185], v[142:145], v[114:129]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[98:113], v[194:197], v[138:141], v[98:113]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[114:129], v[178:181], v[138:141], v[114:129]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[98:113], v[190:193], v[150:153], v[98:113]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[114:129], v[174:177], v[150:153], v[114:129]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[98:113], v[94:97], v[146:149], v[98:113]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[114:129], v[170:173], v[146:149], v[114:129]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[98:113], v[90:93], v[158:161], v[98:113]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[114:129], v[166:169], v[158:161], v[114:129]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[86:89], v[154:157], v[98:113]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[114:129], v[162:165], v[154:157], v[114:129]
.LBB0_16:
	s_waitcnt lgkmcnt(14)
	v_lshrrev_b32_e32 v70, 2, v0
	s_waitcnt lgkmcnt(5)
	v_and_b32_e32 v96, 12, v72
	s_waitcnt vmcnt(8)
	v_and_or_b32 v70, v70, 3, v224
	v_or_b32_e32 v68, v96, v220
	v_or_b32_e32 v69, v230, v251
	v_lshlrev_b32_e32 v97, 7, v70
	v_or3_b32 v195, v69, v68, v97
	v_lshl_add_u32 v69, v195, 1, s21
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[162:163], v69
	ds_read_b64_tr_b16 v[164:165], v69 offset:2048
	ds_read_b64_tr_b16 v[166:167], v69 offset:4096
	ds_read_b64_tr_b16 v[168:169], v69 offset:6144
	ds_read_b64_tr_b16 v[170:171], v69 offset:8192
	ds_read_b64_tr_b16 v[172:173], v69 offset:10240
	ds_read_b64_tr_b16 v[174:175], v69 offset:12288
	ds_read_b64_tr_b16 v[176:177], v69 offset:14336
	v_or_b32_e32 v69, 32, v96
	v_bitop3_b32 v69, v251, v69, v220 bitop3:0xf6
	v_or3_b32 v196, v69, v230, v97
	v_lshl_add_u32 v69, v196, 1, s21
	v_bitop3_b32 v68, v68, v230, 64 bitop3:0x36
	ds_read_b64_tr_b16 v[92:93], v69
	ds_read_b64_tr_b16 v[94:95], v69 offset:2048
	ds_read_b64_tr_b16 v[88:89], v69 offset:4096
	ds_read_b64_tr_b16 v[90:91], v69 offset:6144
	ds_read_b64_tr_b16 v[84:85], v69 offset:8192
	ds_read_b64_tr_b16 v[86:87], v69 offset:10240
	ds_read_b64_tr_b16 v[80:81], v69 offset:12288
	ds_read_b64_tr_b16 v[82:83], v69 offset:14336
	v_or3_b32 v197, v68, v251, v97
	v_cvt_pk_f16_f32 v71, v216, v217
	v_cvt_pk_f16_f32 v70, v214, v215
	v_cvt_pk_f16_f32 v69, v209, v213
	v_cvt_pk_f16_f32 v68, v211, v212
	v_mul_f32_e32 v16, v16, v229
	v_mul_f32_e32 v17, v17, v229
	v_mul_f32_e32 v14, v14, v229
	v_mul_f32_e32 v15, v15, v229
	v_mul_f32_e32 v12, v12, v229
	v_mul_f32_e32 v13, v13, v229
	v_mul_f32_e32 v10, v10, v229
	v_mul_f32_e32 v11, v11, v229
	v_mul_f32_e32 v8, v8, v229
	v_mul_f32_e32 v9, v9, v229
	v_mul_f32_e32 v6, v6, v229
	v_mul_f32_e32 v7, v7, v229
	v_mul_f32_e32 v4, v4, v229
	v_mul_f32_e32 v5, v5, v229
	v_mul_f32_e32 v2, v2, v229
	v_mul_f32_e32 v3, v3, v229
	v_mul_f32_e32 v48, v48, v229
	v_mul_f32_e32 v49, v49, v229
	v_mul_f32_e32 v46, v46, v229
	v_mul_f32_e32 v47, v47, v229
	v_mul_f32_e32 v44, v44, v229
	v_mul_f32_e32 v45, v45, v229
	v_mul_f32_e32 v42, v42, v229
	v_mul_f32_e32 v43, v43, v229
	v_mul_f32_e32 v40, v40, v229
	v_mul_f32_e32 v41, v41, v229
	v_mul_f32_e32 v38, v38, v229
	v_mul_f32_e32 v39, v39, v229
	v_mul_f32_e32 v36, v36, v229
	v_mul_f32_e32 v37, v37, v229
	v_mul_f32_e32 v34, v34, v229
	v_mul_f32_e32 v35, v35, v229
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[2:17], v[162:165], v[68:71], v[2:17]
	v_cvt_pk_f16_f32 v165, v232, v234
	v_cvt_pk_f16_f32 v164, v223, v210
	v_cvt_pk_f16_f32 v163, v1, v250
	v_cvt_pk_f16_f32 v162, v254, v255
	v_add_f32_e32 v67, v211, v212
	v_lshl_add_u32 v182, v197, 1, s21
	v_add_f32_e32 v67, v67, v209
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[34:49], v[92:95], v[68:71], v[34:49]
	ds_read_b64_tr_b16 v[76:77], v182
	ds_read_b64_tr_b16 v[78:79], v182 offset:2048
	ds_read_b64_tr_b16 v[72:73], v182 offset:4096
	ds_read_b64_tr_b16 v[74:75], v182 offset:6144
	v_add_f32_e32 v67, v67, v213
	v_add_f32_e32 v67, v67, v214
	v_mul_f32_e32 v64, v64, v229
	v_mul_f32_e32 v65, v65, v229
	v_mul_f32_e32 v62, v62, v229
	v_mul_f32_e32 v63, v63, v229
	v_mfma_f32_32x32x16_f16 v[2:17], v[166:169], v[162:165], v[2:17]
	v_cvt_pk_f16_f32 v169, v237, v242
	v_cvt_pk_f16_f32 v168, v239, v238
	v_cvt_pk_f16_f32 v167, v235, v240
	v_cvt_pk_f16_f32 v166, v233, v236
	v_mul_f32_e32 v60, v60, v229
	v_mul_f32_e32 v61, v61, v229
	v_mul_f32_e32 v58, v58, v229
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[34:49], v[88:91], v[162:165], v[34:49]
	v_mul_f32_e32 v59, v59, v229
	v_mul_f32_e32 v56, v56, v229
	v_mul_f32_e32 v57, v57, v229
	v_mul_f32_e32 v54, v54, v229
	v_mul_f32_e32 v55, v55, v229
	v_mul_f32_e32 v52, v52, v229
	v_mul_f32_e32 v53, v53, v229
	v_mul_f32_e32 v50, v50, v229
	v_mul_f32_e32 v51, v51, v229
	v_add_f32_e32 v67, v67, v215
	v_mfma_f32_32x32x16_f16 v[2:17], v[170:173], v[166:169], v[2:17]
	v_add_f32_e32 v67, v67, v216
	v_add_f32_e32 v67, v67, v217
	v_add_f32_e32 v67, v67, v254
	v_add_f32_e32 v67, v67, v255
	s_movk_i32 s14, 0x60
	v_add_f32_e32 v67, v67, v1
	v_bitop3_b32 v1, v225, v96, s14 bitop3:0x4e
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[34:49], v[84:87], v[166:169], v[34:49]
	s_add_u32 s6, s16, s12
	v_or3_b32 v1, v1, v251, v97
	s_addc_u32 s7, s17, s13
	v_cvt_pk_f16_f32 v181, v246, v249
	v_cvt_pk_f16_f32 v180, v244, v247
	v_cvt_pk_f16_f32 v179, v245, v248
	v_cvt_pk_f16_f32 v178, v241, v243
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[76:79], v[68:71], v[50:65]
	v_lshl_add_u32 v96, v1, 1, s21
	ds_read_b64_tr_b16 v[92:93], v182 offset:8192
	ds_read_b64_tr_b16 v[94:95], v182 offset:10240
	ds_read_b64_tr_b16 v[170:171], v182 offset:12288
	ds_read_b64_tr_b16 v[172:173], v182 offset:14336
	s_add_u32 s12, s6, s12
	s_addc_u32 s6, s7, s13
	s_add_i32 s16, s24, 0
	v_cndmask_b32_e64 v85, 0, 1, s[2:3]
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_mfma_f32_32x32x16_f16 v[2:17], v[174:177], v[178:181], v[2:17]
	ds_read_b64_tr_b16 v[88:89], v96
	ds_read_b64_tr_b16 v[90:91], v96 offset:2048
	ds_read_b64_tr_b16 v[174:175], v96 offset:4096
	ds_read_b64_tr_b16 v[176:177], v96 offset:6144
	s_and_b32 s6, s6, 0xffff
	s_or_b32 s13, s6, s31
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_add_f32_e32 v67, v67, v250
	v_add_f32_e32 v67, v67, v223
	v_mfma_f32_32x32x16_f16 v[34:49], v[80:83], v[178:181], v[34:49]
	ds_read_b64_tr_b16 v[76:77], v96 offset:8192
	ds_read_b64_tr_b16 v[78:79], v96 offset:10240
	ds_read_b64_tr_b16 v[80:81], v96 offset:12288
	ds_read_b64_tr_b16 v[82:83], v96 offset:14336
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	scratch_load_dword v87, off, off offset:56 ; 4-byte Folded Reload
	v_add_f32_e32 v67, v67, v210
	v_mfma_f32_32x32x16_f16 v[50:65], v[72:75], v[162:165], v[50:65]
	scratch_load_dword v73, off, off offset:64 ; 4-byte Folded Reload
	v_add_u32_e32 v72, s16, v207
	v_add_u32_e32 v74, 0x8000, v72
	scratch_load_dword v72, off, off offset:52 ; 4-byte Folded Reload
	v_readfirstlane_b32 s6, v74
	s_mov_b32 m0, s6
	v_add_f32_e32 v67, v67, v232
	v_add_f32_e32 v67, v67, v234
	v_add_f32_e32 v67, v67, v233
	v_mul_f32_e32 v32, v32, v229
	v_mul_f32_e32 v33, v33, v229
	v_mul_f32_e32 v30, v30, v229
	v_mul_f32_e32 v31, v31, v229
	v_mul_f32_e32 v28, v28, v229
	v_mul_f32_e32 v29, v29, v229
	v_mul_f32_e32 v26, v26, v229
	v_mul_f32_e32 v27, v27, v229
	v_mul_f32_e32 v24, v24, v229
	v_mul_f32_e32 v25, v25, v229
	v_mul_f32_e32 v22, v22, v229
	v_mul_f32_e32 v23, v23, v229
	v_mul_f32_e32 v20, v20, v229
	v_mul_f32_e32 v21, v21, v229
	v_mul_f32_e32 v18, v18, v229
	v_mul_f32_e32 v19, v19, v229
	v_add_f32_e32 v67, v67, v236
	v_add_f32_e32 v67, v67, v235
	v_mfma_f32_32x32x16_f16 v[18:33], v[88:91], v[68:71], v[18:33]
	v_add_f32_e32 v67, v67, v240
	v_add_f32_e32 v67, v67, v239
	v_add_f32_e32 v67, v67, v238
	v_add_f32_e32 v67, v67, v237
	v_add_f32_e32 v67, v67, v242
	v_add_f32_e32 v67, v67, v241
	v_add_f32_e32 v67, v67, v243
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[162:165], v[18:33]
	v_add_f32_e32 v67, v67, v245
	v_add_f32_e32 v67, v67, v248
	v_add_f32_e32 v67, v67, v244
	v_add_f32_e32 v67, v67, v247
	v_add_f32_e32 v67, v67, v246
	v_add_f32_e32 v162, v67, v249
	v_max_f32_e32 v67, v99, v99
	v_max_f32_e32 v68, v98, v98
	v_max_f32_e32 v67, v68, v67
	v_max3_f32 v67, v67, v100, v101
	v_mfma_f32_32x32x16_f16 v[50:65], v[92:95], v[166:169], v[50:65]
	v_max3_f32 v67, v67, v102, v103
	v_max3_f32 v67, v67, v104, v105
	v_max3_f32 v67, v67, v106, v107
	v_max3_f32 v67, v67, v108, v109
	v_max3_f32 v67, v67, v110, v111
	v_max3_f32 v67, v67, v112, v113
	v_max3_f32 v67, v67, v114, v115
	v_mfma_f32_32x32x16_f16 v[18:33], v[76:79], v[166:169], v[18:33]
	v_max3_f32 v67, v67, v116, v117
	v_max3_f32 v67, v67, v118, v119
	v_max3_f32 v67, v67, v120, v121
	v_max3_f32 v67, v67, v122, v123
	v_max3_f32 v67, v67, v124, v125
	v_max3_f32 v67, v67, v126, v127
	v_max3_f32 v164, v67, v128, v129
	v_mfma_f32_32x32x16_f16 v[50:65], v[170:173], v[178:181], v[50:65]
	v_mov_b32_e32 v163, v162
	v_mov_b32_e32 v165, v164
	s_nop 0
	v_permlane32_swap_b32_e32 v162, v163
	v_permlane32_swap_b32_e32 v164, v165
	v_mov_b32_e32 v67, 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[80:83], v[178:181], v[18:33]
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v86, 2, v73
	ds_bpermute_b32 v84, v86, v206
	ds_bpermute_b32 v86, v86, v87
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v72, s16, v72
	v_add_u32_e32 v75, 0x8000, v72
	v_lshrrev_b64 v[72:73], v73, vcc
	v_and_b32_e32 v72, 1, v72
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v73, 1, v84
	v_bfrev_b32_e32 v84, 1
	v_cmp_eq_u32_e32 vcc, 1, v72
	v_readfirstlane_b32 s6, v75
	v_mov_b32_e32 v75, 0
	v_cndmask_b32_e32 v72, v84, v73, vcc
	buffer_load_dwordx4 v72, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v72, 1, v86
	v_cndmask_b32_e32 v72, v84, v72, vcc
	s_mov_b32 m0, s6
	v_cmp_ne_u32_e64 s[6:7], 1, v85
	buffer_load_dwordx4 v72, s[12:15], 0 offen lds
	s_andn2_b64 vcc, exec, s[2:3]
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
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
	s_cbranch_vccnz .LBB0_18
; %bb.17:
	scratch_load_dword v66, off, off        ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v90, 1, v66
	v_add3_u32 v91, s20, v222, v90
	ds_read_b128 v[86:89], v91
	ds_read_b128 v[166:169], v91 offset:8192
	scratch_load_dword v91, off, off offset:4 ; 4-byte Folded Reload
	v_add3_u32 v70, s20, v252, v90
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[82:85], v70 offset:8192
	v_add3_u32 v92, s20, v218, v90
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[134:137], 0
	ds_read_b128 v[170:173], v92 offset:8192
	s_waitcnt vmcnt(0)
	v_add3_u32 v91, s20, v91, v90
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[130:133], v[66:81]
	ds_read_b128 v[86:89], v92
	v_add3_u32 v92, s20, v253, v90
	ds_read_b128 v[174:177], v91 offset:8192
	ds_read_b128 v[178:181], v92 offset:8192
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[142:145], v[66:81]
	ds_read_b128 v[86:89], v91
	v_add3_u32 v91, s20, v208, v90
	ds_read_b128 v[182:185], v91 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[138:141], v[66:81]
	ds_read_b128 v[86:89], v92
	v_add3_u32 v92, s20, v228, v90
	v_add3_u32 v90, s20, v227, v90
	ds_read_b128 v[186:189], v92 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[150:153], v[66:81]
	ds_read_b128 v[86:89], v91
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[146:149], v[66:81]
	ds_read_b128 v[86:89], v92
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[158:161], v[66:81]
	ds_read_b128 v[86:89], v90
	ds_read_b128 v[190:193], v90 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[154:157], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[134:137], 0
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[130:133], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[142:145], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[138:141], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[150:153], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[146:149], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[158:161], v[82:97]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[190:193], v[154:157], v[82:97]
.LBB0_18:
	v_lshl_add_u32 v130, v195, 1, s22
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b64_tr_b16 v[190:191], v130
	ds_read_b64_tr_b16 v[192:193], v130 offset:2048
	ds_read_b64_tr_b16 v[186:187], v130 offset:4096
	ds_read_b64_tr_b16 v[188:189], v130 offset:6144
	ds_read_b64_tr_b16 v[182:183], v130 offset:8192
	ds_read_b64_tr_b16 v[184:185], v130 offset:10240
	ds_read_b64_tr_b16 v[178:179], v130 offset:12288
	ds_read_b64_tr_b16 v[180:181], v130 offset:14336
	v_lshl_add_u32 v130, v196, 1, s22
	v_max3_f32 v198, v231, v164, v165
	v_add_f32_e32 v194, v162, v163
	ds_read_b64_tr_b16 v[174:175], v130
	ds_read_b64_tr_b16 v[176:177], v130 offset:2048
	ds_read_b64_tr_b16 v[170:171], v130 offset:4096
	ds_read_b64_tr_b16 v[172:173], v130 offset:6144
	ds_read_b64_tr_b16 v[166:167], v130 offset:8192
	ds_read_b64_tr_b16 v[168:169], v130 offset:10240
	ds_read_b64_tr_b16 v[162:163], v130 offset:12288
	ds_read_b64_tr_b16 v[164:165], v130 offset:14336
	v_lshl_add_u32 v130, v197, 1, s22
	v_lshl_add_u32 v132, v1, 1, s22
	ds_read_b64_tr_b16 v[158:159], v130
	ds_read_b64_tr_b16 v[160:161], v130 offset:2048
	ds_read_b64_tr_b16 v[154:155], v130 offset:4096
	ds_read_b64_tr_b16 v[156:157], v130 offset:6144
	ds_read_b64_tr_b16 v[150:151], v130 offset:8192
	ds_read_b64_tr_b16 v[152:153], v130 offset:10240
	ds_read_b64_tr_b16 v[146:147], v130 offset:12288
	ds_read_b64_tr_b16 v[148:149], v130 offset:14336
	ds_read_b64_tr_b16 v[142:143], v132
	ds_read_b64_tr_b16 v[144:145], v132 offset:2048
	ds_read_b64_tr_b16 v[138:139], v132 offset:4096
	ds_read_b64_tr_b16 v[140:141], v132 offset:6144
	ds_read_b64_tr_b16 v[134:135], v132 offset:8192
	ds_read_b64_tr_b16 v[136:137], v132 offset:10240
	ds_read_b64_tr_b16 v[130:131], v132 offset:12288
	ds_read_b64_tr_b16 v[132:133], v132 offset:14336
	s_andn2_b64 vcc, exec, s[18:19]
	v_fmac_f32_e32 v194, v226, v229
	s_cbranch_vccnz .LBB0_20
; %bb.19:
	s_mov_b32 s12, 0x3e0293ee
	v_mul_f32_e32 v199, 0x3e0293ee, v198
	v_fma_f32 v98, v98, s12, -v199
	v_fma_f32 v99, v99, s12, -v199
	v_fma_f32 v100, v100, s12, -v199
	v_fma_f32 v101, v101, s12, -v199
	v_fma_f32 v102, v102, s12, -v199
	v_fma_f32 v103, v103, s12, -v199
	v_fma_f32 v104, v104, s12, -v199
	v_fma_f32 v105, v105, s12, -v199
	v_exp_f32_e32 v200, v98
	v_fma_f32 v98, v231, s12, -v199
	v_fma_f32 v106, v106, s12, -v199
	v_fma_f32 v107, v107, s12, -v199
	v_fma_f32 v108, v108, s12, -v199
	v_fma_f32 v109, v109, s12, -v199
	v_fma_f32 v110, v110, s12, -v199
	v_fma_f32 v111, v111, s12, -v199
	v_fma_f32 v112, v112, s12, -v199
	v_fma_f32 v113, v113, s12, -v199
	v_fma_f32 v114, v114, s12, -v199
	v_fma_f32 v115, v115, s12, -v199
	v_fma_f32 v116, v116, s12, -v199
	v_fma_f32 v117, v117, s12, -v199
	v_fma_f32 v118, v118, s12, -v199
	v_fma_f32 v119, v119, s12, -v199
	v_fma_f32 v120, v120, s12, -v199
	v_fma_f32 v121, v121, s12, -v199
	v_fma_f32 v122, v122, s12, -v199
	v_fma_f32 v123, v123, s12, -v199
	v_fma_f32 v124, v124, s12, -v199
	v_fma_f32 v125, v125, s12, -v199
	v_fma_f32 v126, v126, s12, -v199
	v_fma_f32 v127, v127, s12, -v199
	v_fma_f32 v128, v128, s12, -v199
	v_fma_f32 v129, v129, s12, -v199
	v_exp_f32_e32 v201, v99
	v_exp_f32_e32 v202, v100
	v_exp_f32_e32 v203, v101
	v_exp_f32_e32 v204, v102
	v_exp_f32_e32 v205, v103
	v_exp_f32_e32 v206, v104
	v_exp_f32_e32 v207, v105
	v_exp_f32_e32 v199, v98
	v_cvt_pk_f16_f32 v100, v204, v205
	v_cvt_pk_f16_f32 v99, v202, v203
	v_cvt_pk_f16_f32 v101, v206, v207
	v_cvt_pk_f16_f32 v98, v200, v201
	v_mul_f32_e32 v50, v50, v199
	v_mul_f32_e32 v51, v51, v199
	v_mul_f32_e32 v52, v52, v199
	v_mul_f32_e32 v53, v53, v199
	v_mul_f32_e32 v54, v54, v199
	v_mul_f32_e32 v55, v55, v199
	v_mul_f32_e32 v56, v56, v199
	v_mul_f32_e32 v57, v57, v199
	v_mul_f32_e32 v58, v58, v199
	v_mul_f32_e32 v59, v59, v199
	v_mul_f32_e32 v60, v60, v199
	v_mul_f32_e32 v61, v61, v199
	v_mul_f32_e32 v62, v62, v199
	v_mul_f32_e32 v63, v63, v199
	v_mul_f32_e32 v64, v64, v199
	v_mul_f32_e32 v65, v65, v199
	v_exp_f32_e32 v208, v106
	v_exp_f32_e32 v209, v107
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[158:161], v[98:101], v[50:65]
	v_exp_f32_e32 v210, v108
	v_exp_f32_e32 v211, v109
	v_exp_f32_e32 v212, v110
	v_exp_f32_e32 v213, v111
	v_exp_f32_e32 v214, v112
	v_exp_f32_e32 v215, v113
	v_cvt_pk_f16_f32 v103, v210, v211
	v_cvt_pk_f16_f32 v104, v212, v213
	v_cvt_pk_f16_f32 v102, v208, v209
	v_cvt_pk_f16_f32 v105, v214, v215
	v_add_f32_e32 v158, v200, v201
	v_exp_f32_e32 v114, v114
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[50:65], v[154:157], v[102:105], v[50:65]
	v_add_f32_e32 v154, v202, v158
	v_add_f32_e32 v154, v203, v154
	v_exp_f32_e32 v115, v115
	v_exp_f32_e32 v116, v116
	v_exp_f32_e32 v117, v117
	v_exp_f32_e32 v118, v118
	v_exp_f32_e32 v119, v119
	v_exp_f32_e32 v120, v120
	v_mul_f32_e32 v2, v2, v199
	v_mul_f32_e32 v3, v3, v199
	v_mul_f32_e32 v4, v4, v199
	v_mul_f32_e32 v5, v5, v199
	v_mul_f32_e32 v6, v6, v199
	v_mul_f32_e32 v7, v7, v199
	v_mul_f32_e32 v8, v8, v199
	v_mul_f32_e32 v9, v9, v199
	v_mul_f32_e32 v10, v10, v199
	v_mul_f32_e32 v11, v11, v199
	v_mul_f32_e32 v12, v12, v199
	v_mul_f32_e32 v13, v13, v199
	v_mul_f32_e32 v14, v14, v199
	v_mul_f32_e32 v15, v15, v199
	v_mul_f32_e32 v16, v16, v199
	v_mul_f32_e32 v17, v17, v199
	v_exp_f32_e32 v121, v121
	v_mul_f32_e32 v34, v34, v199
	v_mul_f32_e32 v35, v35, v199
	v_mul_f32_e32 v36, v36, v199
	v_mul_f32_e32 v37, v37, v199
	v_mul_f32_e32 v38, v38, v199
	v_mul_f32_e32 v39, v39, v199
	v_mul_f32_e32 v40, v40, v199
	v_mul_f32_e32 v41, v41, v199
	v_mul_f32_e32 v42, v42, v199
	v_mul_f32_e32 v43, v43, v199
	v_mul_f32_e32 v44, v44, v199
	v_mul_f32_e32 v45, v45, v199
	v_mul_f32_e32 v46, v46, v199
	v_mul_f32_e32 v47, v47, v199
	v_mul_f32_e32 v48, v48, v199
	v_mul_f32_e32 v49, v49, v199
	v_add_f32_e32 v154, v204, v154
	v_mul_f32_e32 v18, v18, v199
	v_mul_f32_e32 v19, v19, v199
	v_mul_f32_e32 v20, v20, v199
	v_mul_f32_e32 v21, v21, v199
	v_mul_f32_e32 v22, v22, v199
	v_mul_f32_e32 v23, v23, v199
	v_mul_f32_e32 v24, v24, v199
	v_mul_f32_e32 v25, v25, v199
	v_mul_f32_e32 v26, v26, v199
	v_mul_f32_e32 v27, v27, v199
	v_mul_f32_e32 v28, v28, v199
	v_mul_f32_e32 v29, v29, v199
	v_mul_f32_e32 v30, v30, v199
	v_mul_f32_e32 v31, v31, v199
	v_mul_f32_e32 v32, v32, v199
	v_mul_f32_e32 v33, v33, v199
	v_mfma_f32_32x32x16_f16 v[2:17], v[190:193], v[98:101], v[2:17]
	v_add_f32_e32 v154, v205, v154
	v_add_f32_e32 v154, v206, v154
	v_add_f32_e32 v154, v207, v154
	v_cvt_pk_f16_f32 v109, v120, v121
	v_cvt_pk_f16_f32 v108, v118, v119
	v_cvt_pk_f16_f32 v107, v116, v117
	v_cvt_pk_f16_f32 v106, v114, v115
	v_mfma_f32_32x32x16_f16 v[34:49], v[174:177], v[98:101], v[34:49]
	v_add_f32_e32 v154, v208, v154
	v_exp_f32_e32 v122, v122
	v_exp_f32_e32 v123, v123
	v_exp_f32_e32 v124, v124
	v_exp_f32_e32 v125, v125
	v_exp_f32_e32 v126, v126
	v_exp_f32_e32 v128, v128
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[142:145], v[98:101], v[18:33]
	v_exp_f32_e32 v129, v129
	v_exp_f32_e32 v127, v127
	v_cvt_pk_f16_f32 v111, v124, v125
	v_cvt_pk_f16_f32 v110, v122, v123
	v_cvt_pk_f16_f32 v113, v128, v129
	v_cvt_pk_f16_f32 v112, v126, v127
	v_mfma_f32_32x32x16_f16 v[50:65], v[150:153], v[106:109], v[50:65]
	v_add_f32_e32 v150, v209, v154
	v_add_f32_e32 v150, v210, v150
	v_add_f32_e32 v150, v211, v150
	v_add_f32_e32 v150, v212, v150
	v_add_f32_e32 v150, v213, v150
	v_add_f32_e32 v150, v214, v150
	v_add_f32_e32 v150, v215, v150
	v_mfma_f32_32x32x16_f16 v[2:17], v[186:189], v[102:105], v[2:17]
	v_add_f32_e32 v98, v114, v150
	v_add_f32_e32 v98, v115, v98
	v_add_f32_e32 v98, v116, v98
	v_add_f32_e32 v98, v117, v98
	v_add_f32_e32 v98, v118, v98
	v_add_f32_e32 v98, v119, v98
	v_add_f32_e32 v98, v120, v98
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[102:105], v[34:49]
	v_add_f32_e32 v98, v121, v98
	v_add_f32_e32 v98, v122, v98
	v_add_f32_e32 v98, v123, v98
	v_add_f32_e32 v98, v124, v98
	v_add_f32_e32 v98, v125, v98
	v_add_f32_e32 v98, v126, v98
	v_add_f32_e32 v98, v127, v98
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[18:33], v[138:141], v[102:105], v[18:33]
	v_add_f32_e32 v98, v128, v98
	v_add_f32_e32 v98, v129, v98
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_add_f32_e32 v98, v98, v99
	v_fmac_f32_e32 v98, v194, v199
	v_mfma_f32_32x32x16_f16 v[2:17], v[182:185], v[106:109], v[2:17]
	v_mov_b32_e32 v194, v98
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[106:109], v[34:49]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[134:137], v[106:109], v[18:33]
	v_mfma_f32_32x32x16_f16 v[2:17], v[178:181], v[110:113], v[2:17]
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[110:113], v[34:49]
	v_mfma_f32_32x32x16_f16 v[50:65], v[146:149], v[110:113], v[50:65]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[130:133], v[110:113], v[18:33]
.LBB0_20:
	v_max_f32_e32 v98, v67, v67
	v_max_f32_e32 v99, v66, v66
	v_max_f32_e32 v98, v99, v98
	v_max3_f32 v98, v98, v68, v69
	v_max3_f32 v98, v98, v70, v71
	v_max3_f32 v98, v98, v72, v73
	v_max3_f32 v98, v98, v74, v75
	v_max3_f32 v98, v98, v76, v77
	v_max3_f32 v98, v98, v78, v79
	v_max3_f32 v98, v98, v80, v81
	v_max3_f32 v98, v98, v82, v83
	v_max3_f32 v98, v98, v84, v85
	v_max3_f32 v98, v98, v86, v87
	v_max3_f32 v98, v98, v88, v89
	v_max3_f32 v98, v98, v90, v91
	v_max3_f32 v98, v98, v92, v93
	v_max3_f32 v98, v98, v94, v95
	s_add_i32 s12, s16, 0x8000
	s_waitcnt lgkmcnt(14)
	v_max3_f32 v163, v98, v96, v97
	v_lshl_add_u32 v98, v195, 1, s16
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_lshl_add_u32 v99, v195, 1, s12
	ds_read_b64_tr_b16 v[158:159], v98 offset:32768
	ds_read_b64_tr_b16 v[160:161], v99 offset:2048
	ds_read_b64_tr_b16 v[154:155], v99 offset:4096
	ds_read_b64_tr_b16 v[156:157], v99 offset:6144
	ds_read_b64_tr_b16 v[150:151], v99 offset:8192
	ds_read_b64_tr_b16 v[152:153], v99 offset:10240
	ds_read_b64_tr_b16 v[146:147], v99 offset:12288
	ds_read_b64_tr_b16 v[148:149], v99 offset:14336
	v_lshl_add_u32 v98, v196, 1, s16
	v_lshl_add_u32 v99, v196, 1, s12
	ds_read_b64_tr_b16 v[142:143], v98 offset:32768
	ds_read_b64_tr_b16 v[144:145], v99 offset:2048
	ds_read_b64_tr_b16 v[138:139], v99 offset:4096
	ds_read_b64_tr_b16 v[140:141], v99 offset:6144
	ds_read_b64_tr_b16 v[134:135], v99 offset:8192
	ds_read_b64_tr_b16 v[136:137], v99 offset:10240
	ds_read_b64_tr_b16 v[130:131], v99 offset:12288
	ds_read_b64_tr_b16 v[132:133], v99 offset:14336
	v_lshl_add_u32 v98, v197, 1, s16
	v_lshl_add_u32 v99, v197, 1, s12
	ds_read_b64_tr_b16 v[126:127], v98 offset:32768
	ds_read_b64_tr_b16 v[128:129], v99 offset:2048
	ds_read_b64_tr_b16 v[122:123], v99 offset:4096
	ds_read_b64_tr_b16 v[124:125], v99 offset:6144
	ds_read_b64_tr_b16 v[118:119], v99 offset:8192
	ds_read_b64_tr_b16 v[120:121], v99 offset:10240
	ds_read_b64_tr_b16 v[114:115], v99 offset:12288
	ds_read_b64_tr_b16 v[116:117], v99 offset:14336
	v_lshl_add_u32 v98, v1, 1, s16
	v_lshl_add_u32 v1, v1, 1, s12
	ds_read_b64_tr_b16 v[110:111], v98 offset:32768
	ds_read_b64_tr_b16 v[112:113], v1 offset:2048
	ds_read_b64_tr_b16 v[106:107], v1 offset:4096
	ds_read_b64_tr_b16 v[108:109], v1 offset:6144
	ds_read_b64_tr_b16 v[102:103], v1 offset:8192
	ds_read_b64_tr_b16 v[104:105], v1 offset:10240
	ds_read_b64_tr_b16 v[98:99], v1 offset:12288
	ds_read_b64_tr_b16 v[100:101], v1 offset:14336
	v_mov_b32_e32 v164, v163
	v_cndmask_b32_e64 v162, v198, v231, s[4:5]
	s_nop 0
	v_permlane32_swap_b32_e32 v163, v164
	s_and_b64 vcc, exec, s[6:7]
	v_max3_f32 v1, v162, v163, v164
	s_cbranch_vccnz .LBB0_22
; %bb.21:
	s_mov_b32 s4, 0x3e0293ee
	v_mul_f32_e32 v163, 0x3e0293ee, v1
	v_fma_f32 v66, v66, s4, -v163
	v_fma_f32 v67, v67, s4, -v163
	v_fma_f32 v68, v68, s4, -v163
	v_fma_f32 v69, v69, s4, -v163
	v_fma_f32 v70, v70, s4, -v163
	v_fma_f32 v71, v71, s4, -v163
	v_fma_f32 v72, v72, s4, -v163
	v_fma_f32 v73, v73, s4, -v163
	v_exp_f32_e32 v164, v66
	v_fma_f32 v66, v162, s4, -v163
	v_fma_f32 v74, v74, s4, -v163
	v_fma_f32 v75, v75, s4, -v163
	v_fma_f32 v76, v76, s4, -v163
	v_fma_f32 v77, v77, s4, -v163
	v_fma_f32 v78, v78, s4, -v163
	v_fma_f32 v79, v79, s4, -v163
	v_fma_f32 v80, v80, s4, -v163
	v_fma_f32 v81, v81, s4, -v163
	v_fma_f32 v82, v82, s4, -v163
	v_fma_f32 v83, v83, s4, -v163
	v_fma_f32 v84, v84, s4, -v163
	v_fma_f32 v85, v85, s4, -v163
	v_fma_f32 v86, v86, s4, -v163
	v_fma_f32 v87, v87, s4, -v163
	v_fma_f32 v88, v88, s4, -v163
	v_fma_f32 v89, v89, s4, -v163
	v_fma_f32 v90, v90, s4, -v163
	v_fma_f32 v91, v91, s4, -v163
	v_fma_f32 v92, v92, s4, -v163
	v_fma_f32 v93, v93, s4, -v163
	v_fma_f32 v94, v94, s4, -v163
	v_fma_f32 v95, v95, s4, -v163
	v_fma_f32 v96, v96, s4, -v163
	v_fma_f32 v97, v97, s4, -v163
	v_exp_f32_e32 v165, v67
	v_exp_f32_e32 v166, v68
	v_exp_f32_e32 v167, v69
	v_exp_f32_e32 v168, v70
	v_exp_f32_e32 v169, v71
	v_exp_f32_e32 v170, v72
	v_exp_f32_e32 v171, v73
	v_exp_f32_e32 v163, v66
	v_cvt_pk_f16_f32 v68, v168, v169
	v_cvt_pk_f16_f32 v67, v166, v167
	v_cvt_pk_f16_f32 v69, v170, v171
	v_cvt_pk_f16_f32 v66, v164, v165
	v_mul_f32_e32 v48, v48, v163
	v_mul_f32_e32 v49, v49, v163
	v_mul_f32_e32 v46, v46, v163
	v_mul_f32_e32 v47, v47, v163
	v_mul_f32_e32 v44, v44, v163
	v_mul_f32_e32 v45, v45, v163
	v_mul_f32_e32 v42, v42, v163
	v_mul_f32_e32 v43, v43, v163
	v_mul_f32_e32 v40, v40, v163
	v_mul_f32_e32 v41, v41, v163
	v_mul_f32_e32 v38, v38, v163
	v_mul_f32_e32 v39, v39, v163
	v_mul_f32_e32 v36, v36, v163
	v_mul_f32_e32 v37, v37, v163
	v_mul_f32_e32 v34, v34, v163
	v_mul_f32_e32 v35, v35, v163
	v_exp_f32_e32 v172, v74
	v_exp_f32_e32 v173, v75
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[34:49], v[142:145], v[66:69], v[34:49]
	v_exp_f32_e32 v174, v76
	v_exp_f32_e32 v175, v77
	v_exp_f32_e32 v176, v78
	v_exp_f32_e32 v177, v79
	v_exp_f32_e32 v178, v80
	v_exp_f32_e32 v179, v81
	v_cvt_pk_f16_f32 v71, v174, v175
	v_cvt_pk_f16_f32 v72, v176, v177
	v_cvt_pk_f16_f32 v70, v172, v173
	v_cvt_pk_f16_f32 v73, v178, v179
	v_mul_f32_e32 v64, v64, v163
	v_mul_f32_e32 v65, v65, v163
	v_mfma_f32_32x32x16_f16 v[34:49], v[138:141], v[70:73], v[34:49]
	v_add_f32_e32 v138, v164, v165
	v_add_f32_e32 v138, v166, v138
	v_mul_f32_e32 v62, v62, v163
	v_mul_f32_e32 v63, v63, v163
	v_mul_f32_e32 v60, v60, v163
	v_mul_f32_e32 v61, v61, v163
	v_mul_f32_e32 v58, v58, v163
	v_mul_f32_e32 v59, v59, v163
	v_mul_f32_e32 v56, v56, v163
	v_mul_f32_e32 v57, v57, v163
	v_mul_f32_e32 v54, v54, v163
	v_mul_f32_e32 v55, v55, v163
	v_mul_f32_e32 v52, v52, v163
	v_mul_f32_e32 v53, v53, v163
	v_mul_f32_e32 v50, v50, v163
	v_mul_f32_e32 v51, v51, v163
	v_add_f32_e32 v138, v167, v138
	v_mul_f32_e32 v16, v16, v163
	v_mfma_f32_32x32x16_f16 v[50:65], v[126:129], v[66:69], v[50:65]
	v_mul_f32_e32 v17, v17, v163
	v_mul_f32_e32 v14, v14, v163
	v_mul_f32_e32 v15, v15, v163
	v_mul_f32_e32 v12, v12, v163
	v_mul_f32_e32 v13, v13, v163
	v_mul_f32_e32 v10, v10, v163
	v_mul_f32_e32 v11, v11, v163
	v_mul_f32_e32 v8, v8, v163
	v_mul_f32_e32 v9, v9, v163
	v_mul_f32_e32 v6, v6, v163
	v_mul_f32_e32 v7, v7, v163
	v_mul_f32_e32 v4, v4, v163
	v_mul_f32_e32 v5, v5, v163
	v_mul_f32_e32 v2, v2, v163
	v_mul_f32_e32 v3, v3, v163
	v_add_f32_e32 v126, v168, v138
	v_mul_f32_e32 v32, v32, v163
	v_mul_f32_e32 v33, v33, v163
	v_mul_f32_e32 v30, v30, v163
	v_mul_f32_e32 v31, v31, v163
	v_mul_f32_e32 v28, v28, v163
	v_mul_f32_e32 v29, v29, v163
	v_mul_f32_e32 v26, v26, v163
	v_mul_f32_e32 v27, v27, v163
	v_mul_f32_e32 v24, v24, v163
	v_mul_f32_e32 v25, v25, v163
	v_mul_f32_e32 v22, v22, v163
	v_mul_f32_e32 v23, v23, v163
	v_mul_f32_e32 v20, v20, v163
	v_mul_f32_e32 v21, v21, v163
	v_mul_f32_e32 v18, v18, v163
	v_mul_f32_e32 v19, v19, v163
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[66:69], v[2:17]
	v_add_f32_e32 v126, v169, v126
	v_add_f32_e32 v126, v170, v126
	v_add_f32_e32 v126, v171, v126
	v_add_f32_e32 v126, v172, v126
	v_add_f32_e32 v126, v173, v126
	v_add_f32_e32 v126, v174, v126
	v_exp_f32_e32 v82, v82
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[110:113], v[66:69], v[18:33]
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v85, v85
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	v_mfma_f32_32x32x16_f16 v[50:65], v[122:125], v[70:73], v[50:65]
	v_add_f32_e32 v122, v175, v126
	v_add_f32_e32 v122, v176, v122
	v_add_f32_e32 v122, v177, v122
	v_add_f32_e32 v122, v178, v122
	v_add_f32_e32 v122, v179, v122
	v_add_f32_e32 v66, v82, v122
	v_add_f32_e32 v66, v83, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[154:157], v[70:73], v[2:17]
	v_add_f32_e32 v66, v84, v66
	v_add_f32_e32 v66, v85, v66
	v_exp_f32_e32 v90, v90
	v_cvt_pk_f16_f32 v77, v88, v89
	v_cvt_pk_f16_f32 v76, v86, v87
	v_cvt_pk_f16_f32 v75, v84, v85
	v_cvt_pk_f16_f32 v74, v82, v83
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[18:33], v[106:109], v[70:73], v[18:33]
	v_add_f32_e32 v66, v86, v66
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v66, v87, v66
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v66, v88, v66
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v66, v89, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[150:153], v[74:77], v[2:17]
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_add_f32_e32 v66, v90, v66
	v_add_f32_e32 v66, v91, v66
	v_add_f32_e32 v66, v92, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[134:137], v[74:77], v[34:49]
	v_add_f32_e32 v66, v93, v66
	v_cvt_pk_f16_f32 v81, v96, v97
	v_cvt_pk_f16_f32 v80, v94, v95
	v_cvt_pk_f16_f32 v79, v92, v93
	v_cvt_pk_f16_f32 v78, v90, v91
	v_add_f32_e32 v66, v94, v66
	v_add_f32_e32 v66, v95, v66
	v_mfma_f32_32x32x16_f16 v[50:65], v[118:121], v[74:77], v[50:65]
	v_add_f32_e32 v66, v96, v66
	v_add_f32_e32 v66, v97, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v194, v163
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[102:105], v[74:77], v[18:33]
	v_mov_b32_e32 v194, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[146:149], v[78:81], v[2:17]
	v_mfma_f32_32x32x16_f16 v[34:49], v[130:133], v[78:81], v[34:49]
	v_mfma_f32_32x32x16_f16 v[50:65], v[114:117], v[78:81], v[50:65]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[98:101], v[78:81], v[18:33]
.LBB0_22:                               ; %Flow796
	v_cndmask_b32_e64 v1, v162, v1, s[2:3]
.LBB0_23:                               ; %Flow799
	v_div_scale_f32 v68, s[4:5], v194, v194, 1.0
	v_rcp_f32_e32 v68, v68
	v_div_scale_f32 v69, vcc, 1.0, v194, 1.0
	s_add_i32 s2, s60, 0x100
	v_mul_f32_e32 v68, v69, v68
	s_sub_i32 s3, s33, s67
	s_nop 0
	v_div_fmas_f32 v68, 0, 0, v68
	s_cmp_le_i32 s3, s60
	v_div_fixup_f32 v68, v68, v194, 1.0
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_ge_i32 s3, s2
	v_and_b32_e32 v66, 64, v221
	v_and_b32_e32 v67, 0xa0, v221
	v_pk_mul_f32 v[2:3], v[2:3], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[40:41], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[42:43], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[44:45], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[46:47], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[48:49], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[50:51], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[52:53], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[54:55], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[56:57], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[58:59], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[60:61], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[62:63], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[64:65], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[18:19], v[68:69] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[98:99], v[20:21], v[68:69] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[100:101], v[22:23], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[24:25], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[26:27], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[28:29], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[30:31], v[68:69] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[32:33], v[68:69] op_sel_hi:[1,0]
	s_cselect_b64 s[6:7], -1, 0
	v_or3_b32 v66, v67, v219, v66
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
	v_cvt_pk_f16_f32 v45, v70, v71
	v_cvt_pk_f16_f32 v36, v72, v73
	v_cvt_pk_f16_f32 v41, v74, v75
	v_cvt_pk_f16_f32 v32, v76, v77
	v_cvt_pk_f16_f32 v37, v78, v79
	v_cvt_pk_f16_f32 v28, v80, v81
	v_cvt_pk_f16_f32 v33, v82, v83
	v_cvt_pk_f16_f32 v24, v84, v85
	v_cvt_pk_f16_f32 v29, v86, v87
	v_cvt_pk_f16_f32 v20, v88, v89
	v_cvt_pk_f16_f32 v25, v90, v91
	v_cvt_pk_f16_f32 v16, v92, v93
	v_cvt_pk_f16_f32 v21, v94, v95
	v_cvt_pk_f16_f32 v12, v96, v97
	v_cvt_pk_f16_f32 v17, v98, v99
	v_cvt_pk_f16_f32 v8, v100, v101
	v_cvt_pk_f16_f32 v13, v102, v103
	v_cvt_pk_f16_f32 v4, v104, v105
	v_cvt_pk_f16_f32 v9, v106, v107
	v_cvt_pk_f16_f32 v2, v108, v109
	v_cvt_pk_f16_f32 v5, v68, v69
	s_or_b64 s[4:5], s[4:5], s[6:7]
	v_or_b32_e32 v67, s60, v66
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
	s_cbranch_vccnz .LBB0_25
; %bb.24:
	v_cmp_gt_i32_e32 vcc, s3, v67
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
.LBB0_25:
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
	v_lshl_add_u32 v68, v66, 2, 0
	s_cbranch_scc1 .LBB0_27
; %bb.26:
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v194
	v_mov_b32_e32 v69, 0x42000000
	v_cmp_gt_i32_e64 s[8:9], s33, v67
	v_cndmask_b32_e64 v70, 0, 32, vcc
	v_ldexp_f32 v70, v194, v70
	v_log_f32_e32 v70, v70
	v_cndmask_b32_e32 v69, 0, v69, vcc
	s_sub_i32 s2, 0x100, s2
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	v_sub_f32_e32 v67, v70, v69
	v_add_f32_e32 v67, v1, v67
	ds_write_b32 v68, v67
	v_mov_b32_e32 v67, 2
	v_lshlrev_b32_sdwa v67, v67, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v69, 0, v67
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v69, v69
	v_bfrev_b32_e32 v70, 1
	s_and_b64 vcc, s[0:1], s[2:3]
	s_and_b32 s5, s12, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e32 v67, v70, v67, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v69, v67, s[4:7], 0 offen
	s_cbranch_execz .LBB0_28
	s_branch .LBB0_29
.LBB0_27:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_28:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v194
	v_mov_b32_e32 v67, 0x42000000
	s_and_b32 s5, s12, 0xffff
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v194, v69
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v67, 0, v67, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_sub_f32_e32 v67, v69, v67
	v_add_f32_e32 v1, v1, v67
	ds_write_b32 v68, v1
	v_mov_b32_e32 v1, 2
	v_lshlrev_b32_sdwa v0, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v1, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v1, v1
	v_bfrev_b32_e32 v67, 1
	v_cndmask_b32_e64 v0, v67, v0, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v1, v0, s[4:7], 0 offen
.LBB0_29:
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
	v_mul_lo_u32 v66, s43, v66
	s_bitset1_b32 s2, 14
	s_mov_b32 s4, 0x5040100
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_perm_b32 v0, v63, v62, s4
	v_add_lshl_u32 v62, v66, v224, 1
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
.LBB0_30:                               ; %.critedge
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 76
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
	.set attn_fwd.private_seg_size, 76
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15660
; TotalNumSgprs: 76
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 76
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
    .private_segment_fixed_size: 76
    .sgpr_count:     76
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 18
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
