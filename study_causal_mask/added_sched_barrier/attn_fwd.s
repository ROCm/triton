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
; %bb.63:
	.file	1 "/app/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.64:
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
	s_cbranch_scc1 .LBB0_62
; %bb.1:
	s_add_u32 s20, s22, s24
	s_addc_u32 s21, s23, s25
	s_load_dwordx2 s[66:67], s[20:21], 0x0
	s_load_dwordx8 s[36:43], s[0:1], 0x38
	v_lshrrev_b32_e32 v35, 4, v0
	v_and_b32_e32 v1, 0x100, v0
	v_or_b32_e32 v79, 32, v35
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
	v_or_b32_e32 v8, s60, v35
	s_ashr_i32 s16, s16, 6
	s_ashr_i32 s19, s19, 6
	v_cmp_eq_u32_e64 s[0:1], 0, v1
	v_or_b32_e32 v7, s60, v79
	v_or_b32_e32 v5, 64, v8
	v_or_b32_e32 v3, 0x60, v8
	v_or_b32_e32 v6, 0x80, v8
	v_or_b32_e32 v4, 0xa0, v8
	v_or_b32_e32 v2, 0xc0, v8
	v_or_b32_e32 v1, 0xe0, v8
	s_min_i32 s69, s16, s19
	s_mov_b32 s44, 0
	s_cmp_gt_i32 s69, 0
	s_mul_i32 s58, s41, s18
	s_mul_i32 s56, s42, s17
	s_mul_i32 s54, s68, s43
	s_mul_i32 s52, s43, s60
	v_cmp_gt_i32_e64 s[26:27], s33, v8
	v_cmp_gt_i32_e64 s[24:25], s33, v7
	v_cmp_gt_i32_e64 s[22:23], s33, v5
	v_cmp_gt_i32_e64 s[20:21], s33, v3
	v_cmp_gt_i32_e64 s[30:31], s33, v6
	v_cmp_gt_i32_e64 s[28:29], s33, v4
	v_cmp_gt_i32_e64 s[34:35], s33, v2
	v_cmp_gt_i32_e32 vcc, s33, v1
	v_lshlrev_b32_e32 v101, 3, v0
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
	v_and_b32_e32 v10, 0x78, v101
	s_addc_u32 s16, s19, s47
	v_mad_u64_u32 v[10:11], s[46:47], s43, v35, v[10:11]
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
	v_or_b32_sdwa v9, s60, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_cndmask_b32_e32 v14, v20, v14, vcc
	s_movk_i32 s22, 0x1d64
	s_lshl_b64 s[20:21], s[60:61], 2
	buffer_store_dwordx4 v[10:13], v14, s[48:51], 0 offen
	s_add_u32 s48, s16, s20
	v_cmp_gt_i32_e32 vcc, s22, v9
	v_mov_b32_e32 v9, 2
	s_addc_u32 s16, s19, s21
	v_lshlrev_b32_sdwa v9, v9, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	s_and_b64 vcc, s[0:1], vcc
	s_and_b32 s49, s16, 0xffff
	v_cndmask_b32_e32 v9, v20, v9, vcc
	v_mov_b32_e32 v10, 0x7f800000
	buffer_store_dword v10, v9, s[48:51], 0 offen
.LBB0_3:
	s_cmp_lt_i32 s69, 1
	s_cbranch_scc1 .LBB0_62
; %bb.4:
	s_ashr_i32 s16, s17, 31
	s_lshr_b32 s16, s16, 28
	s_add_i32 s16, s17, s16
	s_ashr_i32 s19, s16, 4
	s_and_b32 s16, s67, 63
	s_sub_i32 s20, 64, s67
	s_cmp_lt_i32 s67, 64
	s_cselect_b32 s41, s20, s16
	s_mul_i32 s20, s12, s18
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
	s_lshl_b32 s17, s14, 5
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	v_and_b32_e32 v34, 0x78, v101
	s_mul_i32 s24, s15, s18
	s_addc_u32 s13, s13, s3
	v_mad_u64_u32 v[10:11], s[2:3], s14, v35, v[34:35]
	s_ashr_i32 s25, s24, 31
	s_lshl_b64 s[2:3], s[24:25], 1
	s_add_u32 s15, s4, s2
	s_mul_i32 s26, s36, s19
	s_addc_u32 s16, s5, s3
	s_ashr_i32 s27, s26, 31
	s_lshl_b64 s[2:3], s[26:27], 1
	s_add_u32 s15, s15, s2
	s_mul_i32 s28, s66, s37
	s_addc_u32 s20, s16, s3
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[2:3], s[28:29], 1
	s_add_u32 s16, s15, s2
	s_mul_i32 s30, s38, s18
	s_addc_u32 s36, s20, s3
	s_ashr_i32 s31, s30, 31
	s_lshl_b64 s[2:3], s[30:31], 1
	s_add_u32 s15, s6, s2
	s_mul_i32 s34, s39, s19
	s_addc_u32 s18, s7, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 1
	s_add_u32 s15, s15, s2
	s_mul_i32 s38, s66, s40
	s_addc_u32 s18, s18, s3
	s_ashr_i32 s39, s38, 31
	s_lshl_b64 s[2:3], s[38:39], 1
	s_add_u32 s48, s15, s2
	s_addc_u32 s49, s18, s3
	s_and_b32 s2, s14, 0x3fff
	v_add_u32_e32 v9, s17, v10
	s_bitset1_b32 s2, 14
	v_lshlrev_b32_e32 v10, 1, v10
	v_bfrev_b32_e32 v33, 1
	v_cmp_gt_i32_e32 vcc, s33, v8
	s_and_b32 s3, s13, 0xffff
	s_lshl_b32 s2, s2, 16
	v_cndmask_b32_e32 v18, v33, v10, vcc
	v_lshlrev_b32_e32 v8, 1, v9
	v_cmp_gt_i32_e32 vcc, s33, v7
	v_add_u32_e32 v16, s17, v9
	s_or_b32 s13, s3, s2
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_cndmask_b32_e32 v7, v33, v8, vcc
	v_add_u32_e32 v17, s17, v16
	buffer_load_dwordx4 v[8:11], v18, s[12:15], 0 offen
	buffer_load_dwordx4 v[12:15], v7, s[12:15], 0 offen
	v_lshlrev_b32_e32 v7, 1, v16
	v_cmp_gt_i32_e32 vcc, s33, v5
	v_add_u32_e32 v24, s17, v17
	v_add_u32_e32 v25, s17, v24
	v_cndmask_b32_e32 v5, v33, v7, vcc
	v_lshlrev_b32_e32 v7, 1, v17
	v_cmp_gt_i32_e32 vcc, s33, v3
	v_add_u32_e32 v32, s17, v25
	v_lshrrev_b32_e32 v80, 1, v0
	v_cndmask_b32_e32 v3, v33, v7, vcc
	buffer_load_dwordx4 v[16:19], v5, s[12:15], 0 offen
	buffer_load_dwordx4 v[20:23], v3, s[12:15], 0 offen
	v_lshlrev_b32_e32 v3, 1, v24
	v_cmp_gt_i32_e32 vcc, s33, v6
	v_lshlrev_b32_e32 v5, 1, v25
	v_mad_u64_u32 v[226:227], s[2:3], s37, v35, v[34:35]
	v_cndmask_b32_e32 v3, v33, v3, vcc
	v_cmp_gt_i32_e32 vcc, s33, v4
	v_mad_u64_u32 v[228:229], s[2:3], s40, v35, v[34:35]
	s_nop 0
	v_cndmask_b32_e32 v4, v33, v5, vcc
	buffer_load_dwordx4 v[24:27], v3, s[12:15], 0 offen
	buffer_load_dwordx4 v[28:31], v4, s[12:15], 0 offen
	v_lshlrev_b32_e32 v3, 1, v32
	v_cmp_gt_i32_e32 vcc, s33, v2
	v_and_b32_e32 v248, 31, v0
	s_movk_i32 s2, 0xe0
	v_cndmask_b32_e32 v2, v33, v3, vcc
	v_add_lshl_u32 v3, v32, s17, 1
	v_cmp_gt_i32_e32 vcc, s33, v1
	s_and_b32 s3, s33, 0xff
	s_or_b32 s3, s41, s3
	v_cndmask_b32_e32 v1, v33, v3, vcc
	buffer_load_dwordx4 v[36:39], v2, s[12:15], 0 offen
	buffer_load_dwordx4 v[40:43], v1, s[12:15], 0 offen
	v_and_b32_e32 v1, 0x78, v80
	v_lshlrev_b32_e32 v33, 7, v35
	v_bitop3_b32 v1, v1, v33, v34 bitop3:0xde
	s_movk_i32 s12, 0x78
	v_lshlrev_b32_e32 v5, 1, v1
	v_bitop3_b32 v6, v80, v101, s12 bitop3:0x28
	v_add_u32_e32 v1, 0, v5
	v_or_b32_e32 v7, 0x1000, v33
	s_barrier
	v_lshlrev_b32_e32 v2, 8, v35
	scratch_store_dword off, v35, off offset:92 ; 4-byte Folded Spill
	v_and_b32_e32 v3, 15, v0
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s12, 4, 5
	v_and_b32_e32 v249, 63, v0
	v_and_b32_e32 v32, 32, v0
	s_mov_b32 s42, 0
	s_cmp_gt_u32 s69, s12
	v_readfirstlane_b32 s44, v0
	v_or_b32_e32 v253, v33, v34
	v_lshlrev_b32_e32 v35, 1, v0
	s_waitcnt vmcnt(8)
	ds_write_b128 v1, v[8:11]
	v_or_b32_e32 v1, v7, v6
	v_lshlrev_b32_e32 v4, 1, v1
	v_add_u32_e32 v1, 0, v4
	s_waitcnt vmcnt(7)
	ds_write_b128 v1, v[12:15]
	v_lshlrev_b32_e32 v1, 1, v6
	v_add3_u32 v1, 0, v1, v2
	s_waitcnt vmcnt(6)
	ds_write_b128 v1, v[16:19] offset:16384
	s_waitcnt vmcnt(5)
	ds_write_b128 v1, v[20:23] offset:24576
	s_waitcnt vmcnt(4)
	ds_write_b128 v1, v[24:27] offset:32768
	s_waitcnt vmcnt(3)
	ds_write_b128 v1, v[28:31] offset:40960
	s_waitcnt vmcnt(2)
	ds_write_b128 v1, v[36:39] offset:49152
	s_waitcnt vmcnt(1)
	ds_write_b128 v1, v[40:43] offset:57344
	v_lshrrev_b32_e32 v1, 2, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v1, off offset:84 ; 4-byte Folded Spill
	v_and_b32_e32 v1, 8, v1
	v_or_b32_e32 v9, 32, v1
	v_or_b32_e32 v10, 48, v1
	v_lshrrev_b32_e32 v9, 3, v9
	v_or_b32_e32 v2, 16, v1
	v_or_b32_e32 v11, 64, v1
	v_bitop3_b32 v22, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v10
	v_or_b32_e32 v12, 0x50, v1
	v_lshrrev_b32_e32 v14, 5, v0
	v_lshrrev_b32_e32 v2, 3, v2
	v_bitop3_b32 v21, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v11
	v_and_or_b32 v8, v80, s2, v248
	v_or_b32_e32 v13, 0x60, v1
	v_bitop3_b32 v3, v14, v3, 1 bitop3:0x6c
	v_bitop3_b32 v2, v2, v0, 15 bitop3:0x78
	v_bitop3_b32 v20, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v12
	v_or_b32_e32 v1, 0x70, v1
	v_bitop3_b32 v19, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v13
	v_lshl_add_u32 v8, v8, 8, 0
	v_lshlrev_b32_e32 v254, 4, v3
	v_lshlrev_b32_e32 v10, 4, v2
	v_bitop3_b32 v18, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v1, 3, v1
	v_add_u32_e32 v9, v8, v254
	scratch_store_dword off, v10, off offset:28 ; 4-byte Folded Spill
	v_add_u32_e32 v10, v8, v10
	v_lshlrev_b32_e32 v250, 4, v22
	v_lshlrev_b32_e32 v255, 4, v21
	v_bitop3_b32 v1, v1, v0, 15 bitop3:0x78
	ds_read_b128 v[134:137], v9
	ds_read_b128 v[130:133], v10
	v_add_u32_e32 v9, v8, v250
	v_add_u32_e32 v10, v8, v255
	ds_read_b128 v[142:145], v9
	ds_read_b128 v[138:141], v10
	v_lshlrev_b32_e32 v9, 4, v20
	v_lshlrev_b32_e32 v10, 4, v19
	v_lshlrev_b32_e32 v11, 4, v18
	v_lshlrev_b32_e32 v12, 4, v1
	scratch_store_dword off, v9, off offset:24 ; 4-byte Folded Spill
	v_add_u32_e32 v9, v8, v9
	scratch_store_dword off, v10, off offset:68 ; 4-byte Folded Spill
	scratch_store_dword off, v11, off offset:72 ; 4-byte Folded Spill
	scratch_store_dword off, v12, off offset:76 ; 4-byte Folded Spill
	v_add_u32_e32 v10, v8, v10
	v_add_u32_e32 v11, v8, v11
	v_add_u32_e32 v8, v8, v12
	ds_read_b128 v[158:161], v9
	ds_read_b128 v[154:157], v10
	ds_read_b128 v[150:153], v11
	ds_read_b128 v[146:149], v8
	v_sub_u32_e32 v6, v6, v34
	s_movk_i32 s2, 0x60
	v_readfirstlane_b32 s45, v1
	v_or_b32_e32 v8, v7, v34
	v_ashrrev_i32_e32 v6, 3, v6
	scratch_store_dword off, v8, off offset:96 ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_6
; %bb.5:                                ; %._crit_edge2233
	v_or_b32_e32 v66, v33, v34
	v_bitop3_b32 v81, v34, v35, s2 bitop3:0x78
	v_lshlrev_b32_e32 v251, 1, v66
	v_sub_u32_e32 v9, v81, v34
	v_or_b32_e32 v67, v7, v34
	v_sub_u32_e32 v7, v5, v251
	v_ashrrev_i32_e32 v9, 3, v9
	v_add_u32_e32 v82, v9, v249
	v_ashrrev_i16_e32 v9, 15, v7
	v_lshrrev_b16_e32 v9, 12, v9
	v_lshlrev_b32_e32 v252, 1, v67
	v_add_u16_e32 v7, v7, v9
	v_sub_u32_e32 v8, v4, v252
	v_ashrrev_i16_e32 v7, 4, v7
	v_add_u32_sdwa v72, v249, sext(v7) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_ashrrev_i16_e32 v7, 15, v8
	v_lshrrev_b16_e32 v7, 12, v7
	v_add_u16_e32 v7, v8, v7
	v_ashrrev_i16_e32 v7, 4, v7
	s_lshl_b32 s44, s37, 6
	s_lshl_b32 s46, s40, 6
	s_and_b32 s3, s37, 0x3fff
	s_and_b32 s2, s40, 0x3fff
	v_add_u32_e32 v230, v6, v249
	v_add_u32_sdwa v74, v249, sext(v7) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_or_b32 s55, s3, 0x4000
	s_or_b32 s57, s2, 0x4000
	v_lshlrev_b32_e32 v238, 2, v82
	s_ashr_i32 s45, s44, 31
	s_ashr_i32 s47, s46, 31
	v_lshlrev_b32_e32 v246, 2, v230
	v_lshlrev_b32_e32 v70, 2, v72
	v_lshlrev_b32_e32 v71, 2, v74
	s_mov_b64 s[2:3], 0
	s_branch .LBB0_7
.LBB0_6:
	s_mov_b64 s[2:3], -1
                                        ; implicit-def: $vgpr66
                                        ; implicit-def: $vgpr251
                                        ; implicit-def: $vgpr67
                                        ; implicit-def: $vgpr252
                                        ; implicit-def: $sgpr55
                                        ; implicit-def: $vgpr81
                                        ; implicit-def: $sgpr57
                                        ; implicit-def: $vgpr238
                                        ; implicit-def: $vgpr82_vgpr83
	s_mov_b64 s[46:47], s[44:45]
                                        ; implicit-def: $vgpr246
                                        ; implicit-def: $vgpr230_vgpr231
                                        ; implicit-def: $vgpr70
                                        ; implicit-def: $vgpr72_vgpr73
                                        ; implicit-def: $vgpr71
                                        ; implicit-def: $vgpr74_vgpr75
.LBB0_7:                                ; %Flow2120
	v_and_b32_e32 v7, 16, v0
	v_lshrrev_b32_e32 v100, 3, v32
	v_and_b32_e32 v234, 32, v101
	v_and_b32_e32 v233, 64, v101
	v_lshl_add_u32 v78, s37, 5, v226
	s_andn2_b64 vcc, exec, s[2:3]
	v_lshl_add_u32 v235, s40, 5, v228
	scratch_store_dword off, v33, off offset:12 ; 4-byte Folded Spill
	scratch_store_dword off, v7, off offset:80 ; 4-byte Folded Spill
	scratch_store_dword off, v32, off offset:112 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:108 ; 4-byte Folded Spill
	scratch_store_dword off, v100, off offset:20 ; 4-byte Folded Spill
	scratch_store_dword off, v101, off offset:88 ; 4-byte Folded Spill
	scratch_store_dword off, v233, off      ; 4-byte Folded Spill
	scratch_store_dword off, v234, off offset:4 ; 4-byte Folded Spill
	scratch_store_dword off, v78, off offset:16 ; 4-byte Folded Spill
	s_cbranch_vccnz .LBB0_14
; %bb.8:
	scratch_store_dword off, v80, off offset:124 ; 4-byte Folded Spill
	scratch_store_dword off, v79, off offset:120 ; 4-byte Folded Spill
	v_mov_b32_e32 v7, s12
	v_sub_u32_e64 v64, s69, v7 clamp
	scratch_load_dword v7, off, off offset:96 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v251, 1, v253
	v_sub_u32_e32 v5, v5, v251
	s_and_b32 s2, s37, 0x3fff
	v_add_u32_e32 v65, 0, v251
	s_or_b32 s55, s2, 0x4000
	s_and_b32 s2, s36, 0xffff
	s_lshl_b32 s65, s55, 16
	v_bfrev_b32_e32 v68, 1
	v_readfirstlane_b32 s42, v65
	v_add_u32_e32 v230, v6, v249
	s_or_b32 s17, s2, s65
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	s_mov_b32 m0, s42
	v_lshlrev_b32_e32 v246, 2, v230
	ds_bpermute_b32 v6, v246, v226
	s_lshl_b32 s44, s37, 6
	s_ashr_i32 s45, s44, 31
	s_lshl_b32 s46, s40, 6
	s_lshl_b64 s[22:23], s[44:45], 1
	s_add_u32 s12, s16, s22
	s_addc_u32 s20, s36, s23
	s_and_b32 s2, s20, 0xffff
	s_or_b32 s13, s2, s65
	s_mov_b32 s14, s18
	s_mov_b32 s15, s19
	v_lshlrev_b32_e32 v32, 8, v248
	v_lshl_or_b32 v72, v3, 4, v32
	v_add_u32_e32 v70, 0, v72
	v_lshl_or_b32 v73, v2, 4, v32
	v_add_u32_e32 v74, 0, v73
	v_lshl_or_b32 v75, v22, 4, v32
	v_add_u32_e32 v76, 0, v75
	v_lshl_or_b32 v77, v21, 4, v32
	v_lshl_or_b32 v79, v20, 4, v32
	v_add_u32_e32 v80, 0, v79
	v_lshl_or_b32 v81, v19, 4, v32
	v_add_u32_e32 v82, 0, v81
	v_lshl_or_b32 v98, v18, 4, v32
	v_add_u32_e32 v83, 0, v98
	v_lshl_or_b32 v99, v1, 4, v32
	v_add_u32_e32 v84, 0, v99
	v_add_u32_e32 v1, 0x8000, v65
	s_mov_b32 s72, s48
	s_mov_b32 s74, s18
	s_mov_b32 s75, s19
	v_readfirstlane_b32 s61, v64
	s_mov_b32 s53, 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v252, 1, v7
	v_ashrrev_i16_e32 v7, 15, v5
	v_lshrrev_b16_e32 v7, 12, v7
	v_add_u16_e32 v5, v5, v7
	v_ashrrev_i16_e32 v5, 4, v5
	v_add_u32_sdwa v86, v249, sext(v5) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshrrev_b64 v[8:9], v86, exec
	v_and_b32_e32 v7, 1, v8
	v_sub_u32_e32 v4, v4, v252
	v_cmp_eq_u32_e32 vcc, 1, v7
	v_ashrrev_i16_e32 v7, 15, v4
	v_lshlrev_b32_e32 v5, 2, v86
	v_lshrrev_b16_e32 v7, 12, v7
	scratch_store_dword off, v5, off offset:136 ; 4-byte Folded Spill
	ds_bpermute_b32 v5, v5, v226
	v_add_u16_e32 v4, v4, v7
	v_ashrrev_i16_e32 v4, 4, v4
	v_add_u32_sdwa v88, v249, sext(v4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshlrev_b32_e32 v4, 2, v88
	ds_bpermute_b32 v7, v4, v78
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v67, 1, v5
	v_cndmask_b32_e32 v5, v68, v67, vcc
	scratch_store_dword off, v4, off offset:148 ; 4-byte Folded Spill
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	v_lshrrev_b64 v[4:5], v88, exec
	v_add_u32_e32 v66, 0, v252
	v_and_b32_e32 v4, 1, v4
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v69, 1, v7
	v_cmp_eq_u32_e32 vcc, 1, v4
	v_readfirstlane_b32 s59, v66
	s_mov_b32 m0, s59
	v_cndmask_b32_e32 v4, v68, v69, vcc
	v_cmp_ne_u32_e32 vcc, 1, v64
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	v_add_u32_e32 v7, 0x4000, v65
	v_lshrrev_b64 v[4:5], v230, vcc
	v_lshlrev_b32_e32 v5, 1, v6
	ds_bpermute_b32 v6, v246, v78
	v_and_b32_e32 v4, 1, v4
	v_cmp_eq_u32_e64 s[2:3], 1, v4
	v_readfirstlane_b32 s17, v7
	s_mov_b32 m0, s17
	v_cndmask_b32_e64 v4, v68, v5, s[2:3]
	v_add_u32_e32 v8, 0x4000, v66
	buffer_load_dwordx4 v4, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v4, 1, v6
	v_cndmask_b32_e64 v4, v68, v4, s[2:3]
	v_readfirstlane_b32 s2, v8
	s_mov_b32 m0, s2
	v_add_u32_e32 v78, 0, v77
	buffer_load_dwordx4 v4, s[12:15], 0 offen lds
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b128 v[4:7], v70
	ds_read_b128 v[24:27], v70 offset:8192
	ds_read_b128 v[28:31], v74
	ds_read_b128 v[36:39], v74 offset:8192
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[4:7], v[134:137], 0
	ds_read_b128 v[40:43], v76 offset:8192
	ds_read_b128 v[20:23], v80
	ds_read_b128 v[48:51], v80 offset:8192
	ds_read_b128 v[44:47], v78 offset:8192
	s_movk_i32 s2, 0x60
	ds_read_b128 v[52:55], v82 offset:8192
	v_readfirstlane_b32 s13, v1
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[130:133], v[2:17]
	ds_read_b128 v[28:31], v76
	s_mov_b32 m0, s13
	ds_read_b128 v[56:59], v83 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[142:145], v[2:17]
	ds_read_b128 v[28:31], v78
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[138:141], v[2:17]
	v_mfma_f32_32x32x16_f16 v[2:17], v[20:23], v[158:161], v[2:17]
	ds_read_b128 v[20:23], v82
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[20:23], v[154:157], v[2:17]
	ds_read_b128 v[18:21], v83
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[150:153], v[2:17]
	ds_read_b128 v[18:21], v84
	ds_read_b128 v[60:63], v84 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[146:149], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[24:27], v[134:137], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[36:39], v[130:133], v[18:33]
	v_bitop3_b32 v36, v34, v35, s2 bitop3:0x78
	scratch_store_dword off, v36, off offset:116 ; 4-byte Folded Spill
	v_sub_u32_e32 v36, v36, v34
	v_ashrrev_i32_e32 v36, 3, v36
	s_and_b32 s2, s40, 0x3fff
	s_or_b32 s57, s2, 0x4000
	s_and_b32 s2, s49, 0xffff
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[142:145], v[18:33]
	v_add_u32_e32 v42, v36, v249
	v_lshlrev_b32_e32 v238, 2, v42
	ds_bpermute_b32 v38, v238, v228
	v_lshrrev_b64 v[36:37], v42, exec
	ds_bpermute_b32 v37, v238, v235
	s_lshl_b32 s17, s57, 16
	v_and_b32_e32 v36, 1, v36
	s_or_b32 s73, s2, s17
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v38, 1, v38
	v_cmp_eq_u32_e64 s[2:3], 1, v36
	v_add_u32_e32 v35, 0x8000, v66
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v37
	v_cndmask_b32_e64 v36, v68, v38, s[2:3]
	buffer_load_dwordx4 v36, s[72:75], 0 offen lds
	v_cndmask_b32_e64 v36, v68, v1, s[2:3]
	v_readfirstlane_b32 s2, v35
	s_mov_b32 m0, s2
	v_cmp_lt_u32_e64 s[2:3], 2, v64
	buffer_load_dwordx4 v36, s[72:75], 0 offen lds
	s_add_u32 s72, s12, s22
	s_addc_u32 s12, s20, s23
	s_ashr_i32 s47, s46, 31
	s_lshl_b64 s[20:21], s[46:47], 1
	v_mov_b32_e32 v36, v86
	s_add_u32 s50, s48, s20
	scratch_store_dwordx2 off, v[36:37], off offset:128 ; 8-byte Folded Spill
	v_lshrrev_b64 v[36:37], v86, s[2:3]
	s_addc_u32 s51, s49, s21
	s_and_b32 s12, s12, 0xffff
	v_and_b32_e32 v35, 1, v36
	s_or_b32 s73, s12, s65
	v_cmp_eq_u32_e64 s[12:13], 1, v35
	v_mov_b32_e32 v36, v88
	s_mov_b32 m0, s42
	v_cndmask_b32_e64 v35, v68, v67, s[12:13]
	scratch_store_dwordx2 off, v[36:37], off offset:140 ; 8-byte Folded Spill
	v_lshrrev_b64 v[36:37], v88, s[2:3]
	buffer_load_dwordx4 v35, s[72:75], 0 offen lds
	v_and_b32_e32 v35, 1, v36
	v_cmp_eq_u32_e64 s[12:13], 1, v35
	s_mov_b32 m0, s59
	.file	2 "/app/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v36, v2, v2
	v_cndmask_b32_e64 v35, v68, v69, s[12:13]
	buffer_load_dwordx4 v35, s[72:75], 0 offen lds
	v_max_f32_e32 v35, v3, v3
	v_max_f32_e32 v35, v36, v35
	v_mov_b32_e32 v36, v42
	s_waitcnt vmcnt(4)
	s_barrier
	scratch_store_dwordx2 off, v[36:37], off offset:100 ; 8-byte Folded Spill
	v_lshrrev_b64 v[36:37], v42, vcc
	v_add_u32_e32 v39, 0xc000, v65
	v_and_b32_e32 v36, 1, v36
	v_add_u32_e32 v40, 0xc000, v66
	s_and_b32 s12, s51, 0xffff
	v_cmp_eq_u32_e32 vcc, 1, v36
	v_readfirstlane_b32 s18, v39
	s_or_b32 s13, s12, s17
	s_mov_b32 s12, s50
	v_cndmask_b32_e32 v36, v68, v38, vcc
	s_mov_b32 m0, s18
	v_readfirstlane_b32 s18, v40
	buffer_load_dwordx4 v36, s[12:15], 0 offen lds
	v_cndmask_b32_e32 v1, v68, v1, vcc
	s_mov_b32 m0, s18
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[138:141], v[18:33]
	buffer_load_dwordx4 v1, s[12:15], 0 offen lds
	v_max3_f32 v35, v35, v4, v5
	v_max3_f32 v35, v35, v6, v7
	v_max3_f32 v35, v35, v8, v9
	v_max3_f32 v35, v35, v10, v11
	v_max3_f32 v35, v35, v12, v13
	v_max3_f32 v35, v35, v14, v15
	v_mfma_f32_32x32x16_f16 v[18:33], v[48:51], v[158:161], v[18:33]
	v_max3_f32 v35, v35, v16, v17
	ds_read_b128 v[68:71], v70 offset:16384
	ds_read_b128 v[202:205], v74 offset:16384
	ds_read_b128 v[198:201], v76 offset:16384
	ds_read_b128 v[194:197], v78 offset:16384
	ds_read_b128 v[190:193], v80 offset:16384
	ds_read_b128 v[94:97], v82 offset:16384
	ds_read_b128 v[90:93], v83 offset:16384
	ds_read_b128 v[86:89], v84 offset:16384
	s_add_i32 s12, 0, 0x4000
	v_mfma_f32_32x32x16_f16 v[18:33], v[52:55], v[154:157], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[56:59], v[150:153], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[60:63], v[146:149], v[18:33]
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
	v_mov_b32_e32 v36, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v36
	v_mov_b32_e32 v35, 0xff800000
	v_max3_f32 v231, v1, v36, v35
	v_mul_f32_e32 v36, 0xbe0293ee, v231
	v_fmamk_f32 v1, v2, 0x3e0293ee, v36
	v_fmamk_f32 v2, v3, 0x3e0293ee, v36
	v_fmamk_f32 v3, v4, 0x3e0293ee, v36
	v_fmamk_f32 v4, v5, 0x3e0293ee, v36
	v_fmamk_f32 v5, v6, 0x3e0293ee, v36
	v_fmamk_f32 v6, v7, 0x3e0293ee, v36
	v_fmamk_f32 v7, v8, 0x3e0293ee, v36
	v_fmamk_f32 v8, v9, 0x3e0293ee, v36
	v_fmamk_f32 v9, v10, 0x3e0293ee, v36
	v_fmamk_f32 v10, v11, 0x3e0293ee, v36
	v_fmamk_f32 v11, v12, 0x3e0293ee, v36
	v_fmamk_f32 v12, v13, 0x3e0293ee, v36
	v_fmamk_f32 v13, v14, 0x3e0293ee, v36
	v_fmamk_f32 v14, v15, 0x3e0293ee, v36
	v_fmamk_f32 v15, v16, 0x3e0293ee, v36
	v_fmamk_f32 v16, v17, 0x3e0293ee, v36
	v_fmamk_f32 v17, v18, 0x3e0293ee, v36
	v_fmamk_f32 v18, v19, 0x3e0293ee, v36
	v_fmamk_f32 v19, v20, 0x3e0293ee, v36
	v_fmamk_f32 v20, v21, 0x3e0293ee, v36
	v_fmamk_f32 v21, v22, 0x3e0293ee, v36
	v_fmamk_f32 v22, v23, 0x3e0293ee, v36
	v_fmamk_f32 v23, v24, 0x3e0293ee, v36
	v_fmamk_f32 v24, v25, 0x3e0293ee, v36
	v_fmamk_f32 v25, v26, 0x3e0293ee, v36
	v_fmamk_f32 v26, v27, 0x3e0293ee, v36
	v_fmamk_f32 v27, v28, 0x3e0293ee, v36
	v_fmamk_f32 v28, v29, 0x3e0293ee, v36
	v_fmamk_f32 v29, v30, 0x3e0293ee, v36
	v_fmamk_f32 v30, v31, 0x3e0293ee, v36
	v_fmamk_f32 v31, v32, 0x3e0293ee, v36
	v_add_u32_e32 v32, s12, v72
	v_fmac_f32_e32 v36, 0x3e0293ee, v33
	v_add_u32_e32 v33, s12, v73
	ds_read_b128 v[82:85], v32 offset:8192
	ds_read_b128 v[186:189], v33 offset:8192
	v_add_u32_e32 v32, s12, v75
	v_add_u32_e32 v33, s12, v77
	ds_read_b128 v[182:185], v32 offset:8192
	ds_read_b128 v[178:181], v33 offset:8192
	v_add_u32_e32 v32, s12, v79
	v_add_u32_e32 v33, s12, v81
	ds_read_b128 v[174:177], v32 offset:8192
	ds_read_b128 v[170:173], v33 offset:8192
	v_add_u32_e32 v32, s12, v98
	v_add_u32_e32 v33, s12, v99
	ds_read_b128 v[166:169], v32 offset:8192
	ds_read_b128 v[162:165], v33 offset:8192
	v_add_u32_e32 v32, 0xff, v0
	s_movk_i32 s12, 0x1ff
	v_cmp_gt_u32_e32 vcc, s12, v32
	s_movk_i32 s12, 0x1fe
	v_fmac_f32_e32 v35, 0xbe0293ee, v231
	v_cmp_lt_u32_e64 s[12:13], s12, v32
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[14:15], s[12:13]
	s_cbranch_execz .LBB0_10
; %bb.9:
	s_barrier
.LBB0_10:
	s_or_b64 exec, exec, s[14:15]
	v_exp_f32_e32 v241, v1
	v_exp_f32_e32 v242, v2
	v_exp_f32_e32 v243, v3
	v_exp_f32_e32 v244, v4
	v_exp_f32_e32 v206, v5
	v_exp_f32_e32 v232, v6
	v_exp_f32_e32 v237, v7
	v_exp_f32_e32 v1, v8
	v_exp_f32_e32 v223, v9
	v_exp_f32_e32 v224, v10
	v_exp_f32_e32 v225, v11
	v_exp_f32_e32 v236, v12
	v_exp_f32_e32 v227, v13
	v_exp_f32_e32 v229, v14
	v_exp_f32_e32 v239, v15
	v_exp_f32_e32 v240, v16
	v_exp_f32_e32 v247, v17
	v_exp_f32_e32 v245, v18
	v_exp_f32_e32 v209, v19
	v_exp_f32_e32 v217, v20
	v_exp_f32_e32 v216, v21
	v_exp_f32_e32 v221, v22
	v_exp_f32_e32 v220, v23
	v_exp_f32_e32 v212, v24
	v_exp_f32_e32 v211, v25
	v_exp_f32_e32 v215, v26
	v_exp_f32_e32 v219, v27
	v_exp_f32_e32 v222, v28
	v_exp_f32_e32 v210, v29
	v_exp_f32_e32 v214, v30
	v_exp_f32_e32 v213, v31
	v_exp_f32_e32 v218, v36
	v_exp_f32_e32 v208, v35
	s_lshl_b32 s42, s61, 6
	s_cmp_lt_u32 s61, 4
	v_lshlrev_b32_e32 v2, 7, v248
	scratch_store_dword off, v2, off offset:8 ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_15
; %bb.11:                               ; %.lr.ph
	v_lshlrev_b32_e32 v2, 2, v0
	scratch_load_dword v0, off, off offset:84 ; 4-byte Folded Reload
	v_and_b32_e32 v5, 12, v2
	v_or_b32_e32 v2, v234, v5
	s_movk_i32 s12, 0x60
	v_bitop3_b32 v4, v5, v234, 32 bitop3:0x36
	scratch_store_dword off, v5, off offset:48 ; 4-byte Folded Spill
	scratch_store_dword off, v238, off offset:152 ; 4-byte Folded Spill
	s_add_i32 s68, s42, 0xffffff40
	s_mul_i32 s15, s44, 6
	s_mul_hi_i32 s14, s44, 6
	v_mov_b32_e32 v207, 1.0
	s_mov_b32 s72, 0x3e0293ee
	s_mov_b32 s74, 0
	s_mov_b32 s73, 0
	s_waitcnt vmcnt(2)
	v_and_or_b32 v3, v0, 3, v100
	v_bitop3_b32 v0, v2, v233, 64 bitop3:0x36
	scratch_store_dword off, v0, off offset:52 ; 4-byte Folded Spill
	v_bitop3_b32 v0, v101, v5, s12 bitop3:0x4e
	scratch_store_dword off, v0, off offset:56 ; 4-byte Folded Spill
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	v_mov_b32_e32 v5, v254
	s_add_u32 s12, s24, s26
	s_addc_u32 s13, s25, s27
	s_add_u32 s12, s12, s28
	s_addc_u32 s13, s13, s29
	s_lshl_b64 s[12:13], s[12:13], 1
	s_add_u32 s12, s15, s12
	s_addc_u32 s13, s14, s13
	v_lshlrev_b32_e32 v3, 7, v3
	s_add_u32 s70, s4, s12
	v_mov_b32_e32 v2, 0
	s_addc_u32 s71, s5, s13
	s_add_i32 s59, 0, 0x8000
	s_add_i32 s63, 0, 0xc000
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
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
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v254, v0
	v_add_u32_e32 v0, v0, v34
	scratch_store_dword off, v0, off offset:60 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 1, v4
	scratch_store_dword off, v250, off offset:32 ; 4-byte Folded Spill
	scratch_store_dword off, v5, off offset:36 ; 4-byte Folded Spill
	scratch_store_dword off, v255, off offset:40 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:64 ; 4-byte Folded Spill
	scratch_load_dword v238, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v233, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v248, off, off offset:76 ; 4-byte Folded Reload
	s_nop 0
	scratch_load_dword v255, off, off offset:80 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v250, 1, v3
	scratch_store_dword off, v235, off offset:44 ; 4-byte Folded Spill
	scratch_load_dword v235, off, off offset:16 ; 4-byte Folded Reload
	s_nop 0
	scratch_load_dword v0, off, off offset:116 ; 4-byte Folded Reload
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v34, v2
.LBB0_12:                               ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[68:71], v[134:137], 0
	s_mov_b64 s[18:19], s[50:51]
	s_mov_b32 s77, s59
	s_mov_b32 s59, s63
	v_mov_b32_e32 v98, v207
	v_mov_b32_e32 v234, v231
	s_mov_b32 s75, s53
	s_setprio 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[202:205], v[130:133], v[66:81]
	v_add_f32_e32 v99, v241, v242
	v_add_f32_e32 v99, v99, v243
	v_add_f32_e32 v99, v99, v244
	v_add_f32_e32 v99, v99, v206
	v_add_f32_e32 v99, v99, v232
	v_add_f32_e32 v99, v99, v237
	v_add_f32_e32 v99, v99, v1
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[198:201], v[142:145], v[66:81]
	v_add_f32_e32 v99, v99, v223
	v_add_f32_e32 v99, v99, v224
	v_add_f32_e32 v99, v99, v225
	v_add_f32_e32 v99, v99, v236
	v_add_f32_e32 v99, v99, v227
	v_add_f32_e32 v99, v99, v229
	v_add_f32_e32 v99, v99, v239
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[194:197], v[138:141], v[66:81]
	v_add_f32_e32 v99, v99, v240
	v_add_f32_e32 v99, v99, v247
	v_add_f32_e32 v99, v99, v245
	v_add_f32_e32 v99, v99, v209
	v_add_f32_e32 v99, v99, v217
	v_add_f32_e32 v99, v99, v216
	v_add_f32_e32 v99, v99, v221
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[190:193], v[158:161], v[66:81]
	v_add_f32_e32 v99, v99, v220
	v_add_f32_e32 v99, v99, v212
	v_add_f32_e32 v99, v99, v211
	v_add_f32_e32 v99, v99, v215
	v_add_f32_e32 v99, v99, v219
	v_add_f32_e32 v99, v99, v222
	v_add_f32_e32 v99, v99, v210
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[154:157], v[66:81]
	v_add_f32_e32 v99, v99, v214
	v_add_f32_e32 v99, v99, v213
	v_add_f32_e32 v99, v99, v218
	v_mov_b32_e32 v100, v99
	s_nop 1
	v_permlane32_swap_b32_e32 v99, v100
	v_add_f32_e32 v207, v99, v100
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[150:153], v[66:81]
	v_mul_f32_e32 v2, v2, v208
	v_mul_f32_e32 v3, v3, v208
	v_mul_f32_e32 v4, v4, v208
	v_mul_f32_e32 v5, v5, v208
	v_mul_f32_e32 v6, v6, v208
	v_mul_f32_e32 v7, v7, v208
	v_mul_f32_e32 v8, v8, v208
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[146:149], v[66:81]
	v_mul_f32_e32 v9, v9, v208
	v_mul_f32_e32 v10, v10, v208
	v_mul_f32_e32 v11, v11, v208
	v_mul_f32_e32 v12, v12, v208
	v_mul_f32_e32 v13, v13, v208
	v_mul_f32_e32 v14, v14, v208
	v_mul_f32_e32 v15, v15, v208
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[134:137], 0
	v_mul_f32_e32 v16, v16, v208
	v_mul_f32_e32 v17, v17, v208
	v_mul_f32_e32 v18, v18, v208
	v_mul_f32_e32 v19, v19, v208
	v_mul_f32_e32 v20, v20, v208
	v_mul_f32_e32 v21, v21, v208
	v_mul_f32_e32 v22, v22, v208
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[130:133], v[82:97]
	v_mul_f32_e32 v23, v23, v208
	v_mul_f32_e32 v24, v24, v208
	v_mul_f32_e32 v25, v25, v208
	v_mul_f32_e32 v26, v26, v208
	v_mul_f32_e32 v27, v27, v208
	v_mul_f32_e32 v28, v28, v208
	v_mul_f32_e32 v29, v29, v208
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[142:145], v[82:97]
	v_mul_f32_e32 v30, v30, v208
	v_mul_f32_e32 v31, v31, v208
	v_mul_f32_e32 v32, v32, v208
	v_mul_f32_e32 v33, v33, v208
	v_mul_f32_e32 v34, v34, v208
	v_mul_f32_e32 v35, v35, v208
	v_mul_f32_e32 v36, v36, v208
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[138:141], v[82:97]
	v_mul_f32_e32 v37, v37, v208
	v_mul_f32_e32 v38, v38, v208
	v_mul_f32_e32 v39, v39, v208
	v_mul_f32_e32 v40, v40, v208
	v_mul_f32_e32 v41, v41, v208
	v_mul_f32_e32 v42, v42, v208
	v_mul_f32_e32 v43, v43, v208
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[158:161], v[82:97]
	v_mul_f32_e32 v44, v44, v208
	v_mul_f32_e32 v45, v45, v208
	v_mul_f32_e32 v46, v46, v208
	v_mul_f32_e32 v47, v47, v208
	v_mul_f32_e32 v48, v48, v208
	v_mul_f32_e32 v49, v49, v208
	v_mul_f32_e32 v50, v50, v208
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[154:157], v[82:97]
	v_mul_f32_e32 v51, v51, v208
	v_mul_f32_e32 v52, v52, v208
	v_mul_f32_e32 v53, v53, v208
	v_mul_f32_e32 v54, v54, v208
	v_mul_f32_e32 v55, v55, v208
	v_mul_f32_e32 v56, v56, v208
	v_mul_f32_e32 v57, v57, v208
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[150:153], v[82:97]
	v_mul_f32_e32 v58, v58, v208
	v_mul_f32_e32 v59, v59, v208
	v_mul_f32_e32 v60, v60, v208
	v_mul_f32_e32 v61, v61, v208
	v_mul_f32_e32 v62, v62, v208
	v_mul_f32_e32 v63, v63, v208
	v_mul_f32_e32 v64, v64, v208
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[146:149], v[82:97]
	v_mul_f32_e32 v65, v65, v208
	v_fmac_f32_e32 v207, v98, v208
	v_cvt_pk_f16_f32 v110, v241, v242
	v_cvt_pk_f16_f32 v111, v243, v244
	v_cvt_pk_f16_f32 v112, v206, v232
	v_cvt_pk_f16_f32 v113, v237, v1
	v_cvt_pk_f16_f32 v106, v223, v224
	v_cvt_pk_f16_f32 v107, v225, v236
	v_cvt_pk_f16_f32 v108, v227, v229
	v_cvt_pk_f16_f32 v109, v239, v240
	v_cvt_pk_f16_f32 v102, v247, v245
	v_cvt_pk_f16_f32 v103, v209, v217
	v_cvt_pk_f16_f32 v104, v216, v221
	v_cvt_pk_f16_f32 v105, v220, v212
	v_cvt_pk_f16_f32 v98, v211, v215
	v_cvt_pk_f16_f32 v99, v219, v222
	v_cvt_pk_f16_f32 v100, v210, v214
	v_cvt_pk_f16_f32 v101, v213, v218
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s50, s18, s20
	s_addc_u32 s51, s19, s21
	s_add_i32 s12, s74, 1
	s_cmp_lt_i32 s12, 2
	s_cselect_b32 s76, s12, 0
	ds_bpermute_b32 v115, v246, v226
	s_lshl_b32 s66, s76, 14
	s_waitcnt vmcnt(1)
	ds_bpermute_b32 v116, v246, v235
	s_add_i32 s53, s66, 0
	v_add_u32_e32 v1, s53, v251
	v_add_u32_e32 v114, s53, v252
	s_and_b32 s12, s71, 0xffff
	v_readfirstlane_b32 s63, v1
	s_or_b32 s13, s12, s65
	s_mov_b32 s12, s70
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v115, 1, v115
	s_mov_b32 m0, s63
	v_readfirstlane_b32 s63, v114
	buffer_load_dwordx4 v115, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v116
	s_mov_b32 m0, s63
	scratch_load_dword v114, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v115, off, off       ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v114, 1, v114
	buffer_load_dwordx4 v1, s[12:15], 0 offen lds
	scratch_load_dword v1, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v115, 1, v115
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v1, 1, s77
	v_add3_u32 v1, v1, v114, v115
	v_lshlrev_b32_e32 v114, 1, v255
	v_add3_u32 v1, v1, v114, v250
	ds_read_b64_tr_b16 v[198:199], v1
	ds_read_b64_tr_b16 v[200:201], v1 offset:2048
	ds_read_b64_tr_b16 v[202:203], v1 offset:4096
	ds_read_b64_tr_b16 v[204:205], v1 offset:6144
	ds_read_b64_tr_b16 v[208:209], v1 offset:8192
	ds_read_b64_tr_b16 v[210:211], v1 offset:10240
	ds_read_b64_tr_b16 v[194:195], v1 offset:12288
	ds_read_b64_tr_b16 v[196:197], v1 offset:14336
	scratch_load_dword v1, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v1, s77, v1, v115
	v_add3_u32 v1, v1, v114, v250
	ds_read_b64_tr_b16 v[190:191], v1
	ds_read_b64_tr_b16 v[192:193], v1 offset:2048
	ds_read_b64_tr_b16 v[186:187], v1 offset:4096
	ds_read_b64_tr_b16 v[188:189], v1 offset:6144
	ds_read_b64_tr_b16 v[182:183], v1 offset:8192
	ds_read_b64_tr_b16 v[184:185], v1 offset:10240
	ds_read_b64_tr_b16 v[178:179], v1 offset:12288
	ds_read_b64_tr_b16 v[180:181], v1 offset:14336
	scratch_load_dword v1, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v1, 1, s77
	v_add3_u32 v1, v1, v114, v250
	ds_read_b64_tr_b16 v[174:175], v1
	ds_read_b64_tr_b16 v[176:177], v1 offset:2048
	ds_read_b64_tr_b16 v[170:171], v1 offset:4096
	ds_read_b64_tr_b16 v[172:173], v1 offset:6144
	ds_read_b64_tr_b16 v[166:167], v1 offset:8192
	ds_read_b64_tr_b16 v[168:169], v1 offset:10240
	ds_read_b64_tr_b16 v[162:163], v1 offset:12288
	ds_read_b64_tr_b16 v[164:165], v1 offset:14336
	scratch_load_dword v1, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v1, 1, s77
	v_add3_u32 v1, v1, v114, v250
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
	v_mfma_f32_32x32x16_f16 v[18:33], v[190:193], v[110:113], v[18:33]
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
	v_mfma_f32_32x32x16_f16 v[50:65], v[126:129], v[110:113], v[50:65]
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_mfma_f32_32x32x16_f16 v[2:17], v[202:205], v[106:109], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[186:189], v[106:109], v[18:33]
	v_mov_b32_e32 v186, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v186
	v_max3_f32 v231, v234, v1, v186
	v_mul_f32_e32 v186, 0x3e0293ee, v231
	v_fma_f32 v1, v66, s72, -v186
	v_fma_f32 v66, v67, s72, -v186
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[106:109], v[34:49]
	v_fma_f32 v67, v68, s72, -v186
	v_fma_f32 v68, v69, s72, -v186
	v_fma_f32 v69, v70, s72, -v186
	v_fma_f32 v70, v71, s72, -v186
	v_fma_f32 v71, v72, s72, -v186
	v_fma_f32 v72, v73, s72, -v186
	v_fma_f32 v73, v74, s72, -v186
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[122:125], v[106:109], v[50:65]
	v_fma_f32 v74, v75, s72, -v186
	v_fma_f32 v75, v76, s72, -v186
	v_fma_f32 v76, v77, s72, -v186
	v_fma_f32 v77, v78, s72, -v186
	v_fma_f32 v78, v79, s72, -v186
	v_fma_f32 v79, v80, s72, -v186
	v_fma_f32 v80, v81, s72, -v186
	v_mfma_f32_32x32x16_f16 v[2:17], v[208:211], v[102:105], v[2:17]
	v_fma_f32 v81, v82, s72, -v186
	v_fma_f32 v82, v83, s72, -v186
	v_fma_f32 v83, v84, s72, -v186
	v_fma_f32 v84, v85, s72, -v186
	v_fma_f32 v85, v86, s72, -v186
	v_fma_f32 v86, v87, s72, -v186
	v_fma_f32 v87, v88, s72, -v186
	v_mfma_f32_32x32x16_f16 v[18:33], v[182:185], v[102:105], v[18:33]
	v_fma_f32 v88, v89, s72, -v186
	v_fma_f32 v89, v90, s72, -v186
	v_fma_f32 v90, v91, s72, -v186
	v_fma_f32 v91, v92, s72, -v186
	v_fma_f32 v92, v93, s72, -v186
	v_fma_f32 v93, v94, s72, -v186
	v_fma_f32 v94, v95, s72, -v186
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[102:105], v[34:49]
	v_fma_f32 v95, v96, s72, -v186
	v_fma_f32 v96, v97, s72, -v186
	v_exp_f32_e32 v242, v66
	v_fma_f32 v66, v234, s72, -v186
	v_exp_f32_e32 v241, v1
	v_exp_f32_e32 v243, v67
	v_exp_f32_e32 v244, v68
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[118:121], v[102:105], v[50:65]
	v_exp_f32_e32 v206, v69
	v_exp_f32_e32 v232, v70
	v_exp_f32_e32 v237, v71
	v_exp_f32_e32 v1, v72
	v_exp_f32_e32 v223, v73
	v_exp_f32_e32 v224, v74
	v_exp_f32_e32 v225, v75
	v_mfma_f32_32x32x16_f16 v[2:17], v[194:197], v[98:101], v[2:17]
	v_exp_f32_e32 v236, v76
	v_exp_f32_e32 v227, v77
	v_exp_f32_e32 v229, v78
	v_exp_f32_e32 v239, v79
	v_exp_f32_e32 v240, v80
	v_exp_f32_e32 v247, v81
	v_exp_f32_e32 v245, v82
	v_mfma_f32_32x32x16_f16 v[18:33], v[178:181], v[98:101], v[18:33]
	v_exp_f32_e32 v209, v83
	v_exp_f32_e32 v217, v84
	v_exp_f32_e32 v216, v85
	v_exp_f32_e32 v221, v86
	v_exp_f32_e32 v220, v87
	v_exp_f32_e32 v212, v88
	v_exp_f32_e32 v211, v89
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[98:101], v[34:49]
	v_exp_f32_e32 v215, v90
	v_exp_f32_e32 v219, v91
	v_exp_f32_e32 v222, v92
	v_exp_f32_e32 v210, v93
	v_exp_f32_e32 v214, v94
	v_exp_f32_e32 v213, v95
	v_exp_f32_e32 v218, v96
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[114:117], v[98:101], v[50:65]
	v_exp_f32_e32 v208, v66
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	scratch_load_dword v68, off, off offset:60 ; 4-byte Folded Reload
	s_lshl_b32 s12, s74, 14
	s_add_i32 s12, s12, 0
	s_add_i32 s63, s12, 0x8000
	v_lshlrev_b32_e32 v66, 1, v0
	v_lshlrev_b32_e32 v67, 1, v254
	v_add3_u32 v66, s12, v66, v67
	v_lshl_add_u32 v67, v253, 1, s63
	v_sub_u32_e32 v69, v66, v67
	v_add_u32_e32 v69, 0x8000, v69
	v_ashrrev_i32_e32 v70, 31, v69
	v_lshrrev_b32_e32 v70, 28, v70
	v_add_u32_e32 v69, v69, v70
	v_ashrrev_i32_e32 v69, 4, v69
	v_add_lshl_u32 v69, v69, v249, 2
	ds_bpermute_b32 v69, v69, v228
	s_and_b32 s12, s51, 0xffff
	v_readfirstlane_b32 s74, v67
	s_or_b32 s13, s12, s17
	s_mov_b32 s12, s50
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v69, 1, v69
	s_mov_b32 m0, s74
	scratch_load_dword v67, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshl_add_u32 v68, v68, 1, s63
	v_add_u32_e32 v68, 0x2000, v68
	v_sub_u32_e32 v66, v66, v68
	v_add_u32_e32 v66, 0xa000, v66
	v_ashrrev_i32_e32 v70, 31, v66
	v_lshrrev_b32_e32 v70, 28, v70
	v_add_u32_e32 v66, v66, v70
	scratch_load_dword v70, off, off offset:44 ; 4-byte Folded Reload
	v_ashrrev_i32_e32 v66, 4, v66
	v_add_lshl_u32 v66, v66, v249, 2
	v_readfirstlane_b32 s74, v68
	buffer_load_dwordx4 v69, s[12:15], 0 offen lds
	s_mov_b32 m0, s74
	scratch_load_dword v68, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	ds_bpermute_b32 v66, v66, v70
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	buffer_load_dwordx4 v66, s[12:15], 0 offen lds
	scratch_load_dword v66, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	v_add3_u32 v72, s75, v68, v66
	scratch_load_dword v68, off, off offset:32 ; 4-byte Folded Reload
	v_add3_u32 v76, s75, v238, v66
	v_add3_u32 v77, s75, v233, v66
	s_waitcnt vmcnt(0)
	v_add3_u32 v73, s75, v68, v66
	scratch_load_dword v68, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v74, s75, v68, v66
	scratch_load_dword v68, off, off offset:24 ; 4-byte Folded Reload
	v_add3_u32 v67, s75, v67, v66
	s_waitcnt vmcnt(0)
	v_add3_u32 v75, s75, v68, v66
	v_add3_u32 v66, s75, v248, v66
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
	s_add_i32 s73, s73, 64
	s_add_u32 s70, s70, s22
	s_addc_u32 s71, s71, s23
	s_cmp_lt_i32 s73, s68
	s_mov_b32 s74, s76
	s_barrier
	s_cbranch_scc1 .LBB0_12
; %bb.13:                               ; %Flow2118
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	scratch_load_dword v254, off, off offset:36 ; 4-byte Folded Reload
	scratch_load_dword v250, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v255, off, off offset:40 ; 4-byte Folded Reload
	scratch_load_dword v233, off, off       ; 4-byte Folded Reload
	scratch_load_dword v234, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v235, off, off offset:44 ; 4-byte Folded Reload
	scratch_load_dword v238, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v248, 31, v0
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execnz .LBB0_16
	s_branch .LBB0_17
.LBB0_14:
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v198, 0xff800000
	v_mov_b32_e32 v247, 1.0
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
	v_mov_b32_e32 v253, v66
	scratch_store_dword off, v67, off offset:96 ; 4-byte Folded Spill
	s_branch .LBB0_26
.LBB0_15:
	v_mov_b32_e32 v17, 0
	s_add_i32 s59, 0, 0x8000
	s_add_i32 s63, 0, 0xc000
	v_mov_b32_e32 v207, 1.0
	v_mov_b32_e32 v16, v17
	v_mov_b32_e32 v15, v17
	v_mov_b32_e32 v14, v17
	v_mov_b32_e32 v13, v17
	v_mov_b32_e32 v12, v17
	v_mov_b32_e32 v11, v17
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v9, v17
	v_mov_b32_e32 v8, v17
	v_mov_b32_e32 v7, v17
	v_mov_b32_e32 v6, v17
	v_mov_b32_e32 v5, v17
	v_mov_b32_e32 v4, v17
	v_mov_b32_e32 v3, v17
	v_mov_b32_e32 v2, v17
	v_mov_b32_e32 v33, v17
	v_mov_b32_e32 v32, v17
	v_mov_b32_e32 v31, v17
	v_mov_b32_e32 v30, v17
	v_mov_b32_e32 v29, v17
	v_mov_b32_e32 v28, v17
	v_mov_b32_e32 v27, v17
	v_mov_b32_e32 v26, v17
	v_mov_b32_e32 v25, v17
	v_mov_b32_e32 v24, v17
	v_mov_b32_e32 v23, v17
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v21, v17
	v_mov_b32_e32 v20, v17
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v49, v17
	v_mov_b32_e32 v48, v17
	v_mov_b32_e32 v47, v17
	v_mov_b32_e32 v46, v17
	v_mov_b32_e32 v45, v17
	v_mov_b32_e32 v44, v17
	v_mov_b32_e32 v43, v17
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v41, v17
	v_mov_b32_e32 v40, v17
	v_mov_b32_e32 v39, v17
	v_mov_b32_e32 v38, v17
	v_mov_b32_e32 v37, v17
	v_mov_b32_e32 v36, v17
	v_mov_b32_e32 v35, v17
	v_mov_b32_e32 v34, v17
	v_mov_b32_e32 v65, v17
	v_mov_b32_e32 v64, v17
	v_mov_b32_e32 v63, v17
	v_mov_b32_e32 v62, v17
	v_mov_b32_e32 v61, v17
	v_mov_b32_e32 v60, v17
	v_mov_b32_e32 v59, v17
	v_mov_b32_e32 v58, v17
	v_mov_b32_e32 v57, v17
	v_mov_b32_e32 v56, v17
	v_mov_b32_e32 v55, v17
	v_mov_b32_e32 v54, v17
	v_mov_b32_e32 v53, v17
	v_mov_b32_e32 v52, v17
	v_mov_b32_e32 v51, v17
	v_mov_b32_e32 v50, v17
	s_mov_b32 s66, 0
	s_mov_b64 s[18:19], s[48:49]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_17
.LBB0_16:
	s_barrier
.LBB0_17:
	s_or_b64 exec, exec, s[12:13]
	s_cmp_eq_u32 s61, 1
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lg_u32 s61, 1
	s_cselect_b64 s[50:51], -1, 0
	v_mov_b32_e32 v66, 0
	s_and_b64 vcc, exec, s[12:13]
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
	s_cbranch_vccnz .LBB0_19
; %bb.18:
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
	v_mfma_f32_32x32x16_f16 v[98:113], v[190:193], v[158:161], v[98:113]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[114:129], v[174:177], v[158:161], v[114:129]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[98:113], v[94:97], v[154:157], v[98:113]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[114:129], v[170:173], v[154:157], v[114:129]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[98:113], v[90:93], v[150:153], v[98:113]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[114:129], v[166:169], v[150:153], v[114:129]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[86:89], v[146:149], v[98:113]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[114:129], v[162:165], v[146:149], v[114:129]
.LBB0_19:
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	scratch_load_dword v175, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v69, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v70, off, off offset:84 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v68, 2, v0
	v_and_b32_e32 v96, 12, v68
	v_or_b32_e32 v97, v96, v234
	v_cvt_pk_f16_f32 v71, v237, v1
	v_mul_f32_e32 v16, v16, v208
	v_mul_f32_e32 v17, v17, v208
	v_mul_f32_e32 v14, v14, v208
	v_mul_f32_e32 v15, v15, v208
	v_mul_f32_e32 v12, v12, v208
	v_mul_f32_e32 v13, v13, v208
	v_mul_f32_e32 v10, v10, v208
	v_mul_f32_e32 v11, v11, v208
	v_mul_f32_e32 v8, v8, v208
	v_mul_f32_e32 v9, v9, v208
	v_mul_f32_e32 v6, v6, v208
	v_mul_f32_e32 v7, v7, v208
	v_mul_f32_e32 v4, v4, v208
	v_mul_f32_e32 v5, v5, v208
	v_mul_f32_e32 v2, v2, v208
	v_mul_f32_e32 v3, v3, v208
	v_cvt_pk_f16_f32 v173, v239, v240
	v_cvt_pk_f16_f32 v172, v227, v229
	v_cvt_pk_f16_f32 v171, v225, v236
	v_cvt_pk_f16_f32 v170, v223, v224
	v_add_f32_e32 v67, v241, v242
	v_mul_f32_e32 v32, v32, v208
	v_mul_f32_e32 v33, v33, v208
	v_mul_f32_e32 v30, v30, v208
	v_mul_f32_e32 v31, v31, v208
	v_mul_f32_e32 v28, v28, v208
	v_mul_f32_e32 v29, v29, v208
	v_mul_f32_e32 v26, v26, v208
	v_mul_f32_e32 v27, v27, v208
	v_mul_f32_e32 v24, v24, v208
	v_mul_f32_e32 v25, v25, v208
	v_mul_f32_e32 v22, v22, v208
	v_mul_f32_e32 v23, v23, v208
	v_mul_f32_e32 v20, v20, v208
	v_mul_f32_e32 v21, v21, v208
	v_mul_f32_e32 v18, v18, v208
	v_mul_f32_e32 v19, v19, v208
	v_add_f32_e32 v67, v67, v243
	v_add_f32_e32 v67, v67, v244
	v_add_f32_e32 v67, v67, v206
	v_add_f32_e32 v67, v67, v232
	v_add_f32_e32 v67, v67, v237
	v_add_f32_e32 v67, v67, v1
	v_bitop3_b32 v1, v97, v233, 64 bitop3:0x36
	v_mul_f32_e32 v48, v48, v208
	v_mul_f32_e32 v49, v49, v208
	v_mul_f32_e32 v46, v46, v208
	v_mul_f32_e32 v47, v47, v208
	v_mul_f32_e32 v44, v44, v208
	v_mul_f32_e32 v45, v45, v208
	v_mul_f32_e32 v42, v42, v208
	v_mul_f32_e32 v43, v43, v208
	v_mul_f32_e32 v40, v40, v208
	v_mul_f32_e32 v41, v41, v208
	v_mul_f32_e32 v38, v38, v208
	v_mul_f32_e32 v39, v39, v208
	v_mul_f32_e32 v36, v36, v208
	v_mul_f32_e32 v37, v37, v208
	v_mul_f32_e32 v34, v34, v208
	v_mul_f32_e32 v35, v35, v208
	s_add_u32 s14, s18, s20
	v_add_f32_e32 v67, v67, v223
	s_addc_u32 s15, s19, s21
	v_add_f32_e32 v67, v67, v224
	v_add_f32_e32 v67, v67, v225
	s_movk_i32 s18, 0x60
	s_add_u32 s20, s14, s20
	v_add_f32_e32 v67, v67, v236
	s_addc_u32 s14, s15, s21
	v_add_f32_e32 v67, v67, v227
	s_and_b32 s14, s14, 0xffff
	s_or_b32 s21, s14, s17
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	s_waitcnt vmcnt(2)
	v_or_b32_e32 v68, v233, v175
	s_waitcnt vmcnt(0)
	v_and_or_b32 v69, v70, 3, v69
	v_lshlrev_b32_e32 v174, 7, v69
	v_or3_b32 v194, v68, v97, v174
	v_lshl_add_u32 v68, v194, 1, s59
	ds_read_b64_tr_b16 v[88:89], v68
	ds_read_b64_tr_b16 v[90:91], v68 offset:2048
	ds_read_b64_tr_b16 v[92:93], v68 offset:4096
	ds_read_b64_tr_b16 v[94:95], v68 offset:6144
	ds_read_b64_tr_b16 v[162:163], v68 offset:8192
	ds_read_b64_tr_b16 v[164:165], v68 offset:10240
	ds_read_b64_tr_b16 v[166:167], v68 offset:12288
	ds_read_b64_tr_b16 v[168:169], v68 offset:14336
	v_or_b32_e32 v68, 32, v96
	v_bitop3_b32 v68, v175, v68, v234 bitop3:0xf6
	v_or3_b32 v195, v68, v233, v174
	v_lshl_add_u32 v68, v195, 1, s59
	ds_read_b64_tr_b16 v[84:85], v68
	ds_read_b64_tr_b16 v[86:87], v68 offset:2048
	ds_read_b64_tr_b16 v[80:81], v68 offset:4096
	ds_read_b64_tr_b16 v[82:83], v68 offset:6144
	ds_read_b64_tr_b16 v[76:77], v68 offset:8192
	ds_read_b64_tr_b16 v[78:79], v68 offset:10240
	ds_read_b64_tr_b16 v[72:73], v68 offset:12288
	ds_read_b64_tr_b16 v[74:75], v68 offset:14336
	v_cvt_pk_f16_f32 v70, v206, v232
	v_cvt_pk_f16_f32 v69, v243, v244
	v_cvt_pk_f16_f32 v68, v241, v242
	v_or3_b32 v1, v1, v175, v174
	v_lshl_add_u32 v97, v1, 1, s59
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[2:17], v[88:91], v[68:71], v[2:17]
	v_cvt_pk_f16_f32 v91, v213, v218
	v_cvt_pk_f16_f32 v90, v210, v214
	v_cvt_pk_f16_f32 v89, v219, v222
	v_cvt_pk_f16_f32 v88, v211, v215
	v_add_f32_e32 v67, v67, v229
	v_add_f32_e32 v67, v67, v239
	v_mul_f32_e32 v64, v64, v208
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[2:17], v[92:95], v[170:173], v[2:17]
	v_cvt_pk_f16_f32 v95, v220, v212
	v_cvt_pk_f16_f32 v94, v216, v221
	v_cvt_pk_f16_f32 v93, v209, v217
	v_cvt_pk_f16_f32 v92, v247, v245
	v_mul_f32_e32 v65, v65, v208
	v_mul_f32_e32 v62, v62, v208
	v_mul_f32_e32 v63, v63, v208
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[2:17], v[162:165], v[92:95], v[2:17]
	v_mul_f32_e32 v60, v60, v208
	v_mul_f32_e32 v61, v61, v208
	v_mul_f32_e32 v58, v58, v208
	v_mul_f32_e32 v59, v59, v208
	v_mul_f32_e32 v56, v56, v208
	v_mul_f32_e32 v57, v57, v208
	v_mul_f32_e32 v54, v54, v208
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[84:87], v[68:71], v[18:33]
	ds_read_b64_tr_b16 v[84:85], v97
	ds_read_b64_tr_b16 v[86:87], v97 offset:2048
	ds_read_b64_tr_b16 v[162:163], v97 offset:4096
	ds_read_b64_tr_b16 v[164:165], v97 offset:6144
	v_mul_f32_e32 v55, v55, v208
	v_mul_f32_e32 v52, v52, v208
	v_mul_f32_e32 v53, v53, v208
	v_mul_f32_e32 v50, v50, v208
	v_mul_f32_e32 v51, v51, v208
	v_add_f32_e32 v67, v67, v240
	v_mfma_f32_32x32x16_f16 v[2:17], v[166:169], v[88:91], v[2:17]
	v_add_f32_e32 v67, v67, v247
	v_add_f32_e32 v67, v67, v245
	v_add_f32_e32 v67, v67, v209
	v_add_f32_e32 v67, v67, v217
	v_add_f32_e32 v67, v67, v216
	v_add_f32_e32 v67, v67, v221
	v_add_f32_e32 v67, v67, v220
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[18:33], v[80:83], v[170:173], v[18:33]
	ds_read_b64_tr_b16 v[80:81], v97 offset:8192
	ds_read_b64_tr_b16 v[82:83], v97 offset:10240
	ds_read_b64_tr_b16 v[166:167], v97 offset:12288
	ds_read_b64_tr_b16 v[168:169], v97 offset:14336
	scratch_load_dword v97, off, off offset:88 ; 4-byte Folded Reload
	v_add_f32_e32 v67, v67, v212
	v_add_f32_e32 v67, v67, v211
	v_add_f32_e32 v67, v67, v215
	v_add_f32_e32 v67, v67, v219
	v_add_f32_e32 v67, v67, v222
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[34:49], v[84:87], v[68:71], v[34:49]
	ds_bpermute_b32 v84, v238, v228
	v_cndmask_b32_e64 v85, 0, 1, s[2:3]
	v_cmp_ne_u32_e32 vcc, 0, v85
	ds_bpermute_b32 v86, v238, v235
	v_add_f32_e32 v67, v67, v210
	v_add_f32_e32 v67, v67, v214
	v_add_f32_e32 v67, v67, v213
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[170:173], v[34:49]
	v_add_f32_e32 v162, v67, v218
	v_max_f32_e32 v67, v99, v99
	v_mov_b32_e32 v163, v162
	s_nop 1
	v_permlane32_swap_b32_e32 v162, v163
	v_mov_b32_e32 v87, 0
	s_waitcnt vmcnt(0)
	v_bitop3_b32 v96, v97, v96, s18 bitop3:0x4e
	v_mfma_f32_32x32x16_f16 v[18:33], v[76:79], v[92:95], v[18:33]
	s_add_i32 s18, s66, 0
	v_or3_b32 v227, v96, v175, v174
	v_lshl_add_u32 v96, v227, 1, s59
	v_mov_b32_e32 v97, 0
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[34:49], v[80:83], v[92:95], v[34:49]
	v_add_u32_e32 v80, s18, v251
	v_add_u32_e32 v82, 0x8000, v80
	v_add_u32_e32 v80, s18, v252
	v_add_u32_e32 v83, 0x8000, v80
	v_readfirstlane_b32 s14, v82
	s_mov_b32 m0, s14
	v_readfirstlane_b32 s14, v83
	v_mfma_f32_32x32x16_f16 v[18:33], v[72:75], v[88:91], v[18:33]
	ds_read_b64_tr_b16 v[72:73], v96
	ds_read_b64_tr_b16 v[74:75], v96 offset:2048
	ds_read_b64_tr_b16 v[76:77], v96 offset:4096
	ds_read_b64_tr_b16 v[78:79], v96 offset:6144
	scratch_load_dwordx2 v[80:81], off, off offset:100 ; 8-byte Folded Reload
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	s_waitcnt vmcnt(0)
	v_lshrrev_b64 v[80:81], v80, vcc
	v_and_b32_e32 v80, 1, v80
	s_waitcnt lgkmcnt(5)
	v_lshlrev_b32_e32 v81, 1, v84
	v_bfrev_b32_e32 v84, 1
	v_cmp_eq_u32_e32 vcc, 1, v80
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[72:75], v[68:71], v[50:65]
	v_cndmask_b32_e32 v80, v84, v81, vcc
	buffer_load_dwordx4 v80, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v80, 1, v86
	v_cndmask_b32_e32 v80, v84, v80, vcc
	s_mov_b32 m0, s14
	v_cmp_ne_u32_e64 s[14:15], 1, v85
	buffer_load_dwordx4 v80, s[20:23], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[76:79], v[170:173], v[50:65]
	ds_read_b64_tr_b16 v[68:69], v96 offset:8192
	ds_read_b64_tr_b16 v[70:71], v96 offset:10240
	ds_read_b64_tr_b16 v[72:73], v96 offset:12288
	ds_read_b64_tr_b16 v[74:75], v96 offset:14336
	s_andn2_b64 vcc, exec, s[2:3]
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[68:71], v[92:95], v[50:65]
	v_max_f32_e32 v68, v98, v98
	v_max_f32_e32 v67, v68, v67
	v_max3_f32 v67, v67, v100, v101
	v_max3_f32 v67, v67, v102, v103
	v_max3_f32 v67, v67, v104, v105
	v_max3_f32 v67, v67, v106, v107
	v_max3_f32 v67, v67, v108, v109
	v_max3_f32 v67, v67, v110, v111
	v_max3_f32 v67, v67, v112, v113
	v_max3_f32 v67, v67, v114, v115
	v_max3_f32 v67, v67, v116, v117
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[88:91], v[34:49]
	v_max3_f32 v67, v67, v118, v119
	v_max3_f32 v67, v67, v120, v121
	v_max3_f32 v67, v67, v122, v123
	v_max3_f32 v67, v67, v124, v125
	v_max3_f32 v67, v67, v126, v127
	v_max3_f32 v164, v67, v128, v129
	v_mov_b32_e32 v165, v164
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[72:75], v[88:91], v[50:65]
	v_permlane32_swap_b32_e32 v164, v165
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	s_cbranch_vccnz .LBB0_21
; %bb.20:
	scratch_load_dword v66, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v71, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v90, 1, v66
	v_add3_u32 v70, s53, v254, v90
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[82:85], v70 offset:8192
	s_waitcnt vmcnt(0)
	v_add3_u32 v91, s53, v71, v90
	ds_read_b128 v[86:89], v91
	ds_read_b128 v[166:169], v91 offset:8192
	v_add3_u32 v92, s53, v250, v90
	v_add3_u32 v91, s53, v255, v90
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[134:137], 0
	ds_read_b128 v[170:173], v92 offset:8192
	ds_read_b128 v[174:177], v91 offset:8192
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[130:133], v[66:81]
	ds_read_b128 v[86:89], v92
	scratch_load_dword v92, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[142:145], v[66:81]
	ds_read_b128 v[86:89], v91
	scratch_load_dword v91, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_add3_u32 v92, s53, v92, v90
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[138:141], v[66:81]
	ds_read_b128 v[86:89], v92
	ds_read_b128 v[178:181], v92 offset:8192
	scratch_load_dword v92, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_add3_u32 v91, s53, v91, v90
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[158:161], v[66:81]
	ds_read_b128 v[86:89], v91
	ds_read_b128 v[182:185], v91 offset:8192
	scratch_load_dword v91, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_add3_u32 v92, s53, v92, v90
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[154:157], v[66:81]
	ds_read_b128 v[86:89], v92
	ds_read_b128 v[186:189], v92 offset:8192
	s_waitcnt vmcnt(0)
	v_add3_u32 v90, s53, v91, v90
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[150:153], v[66:81]
	ds_read_b128 v[86:89], v90
	ds_read_b128 v[190:193], v90 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[146:149], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[134:137], 0
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[130:133], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[142:145], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[138:141], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[158:161], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[154:157], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[150:153], v[82:97]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[190:193], v[146:149], v[82:97]
.LBB0_21:
	v_add_f32_e32 v247, v162, v163
	v_lshl_add_u32 v162, v194, 1, s63
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[222:223], v162
	ds_read_b64_tr_b16 v[224:225], v162 offset:2048
	ds_read_b64_tr_b16 v[218:219], v162 offset:4096
	ds_read_b64_tr_b16 v[220:221], v162 offset:6144
	ds_read_b64_tr_b16 v[214:215], v162 offset:8192
	ds_read_b64_tr_b16 v[216:217], v162 offset:10240
	ds_read_b64_tr_b16 v[210:211], v162 offset:12288
	ds_read_b64_tr_b16 v[212:213], v162 offset:14336
	scratch_store_dword off, v195, off offset:8 ; 4-byte Folded Spill
	v_lshl_add_u32 v162, v195, 1, s63
	v_fmac_f32_e32 v247, v207, v208
	v_max3_f32 v229, v231, v164, v165
	v_mov_b32_e32 v245, v194
	ds_read_b64_tr_b16 v[206:207], v162
	ds_read_b64_tr_b16 v[208:209], v162 offset:2048
	ds_read_b64_tr_b16 v[202:203], v162 offset:4096
	ds_read_b64_tr_b16 v[204:205], v162 offset:6144
	ds_read_b64_tr_b16 v[198:199], v162 offset:8192
	ds_read_b64_tr_b16 v[200:201], v162 offset:10240
	ds_read_b64_tr_b16 v[194:195], v162 offset:12288
	ds_read_b64_tr_b16 v[196:197], v162 offset:14336
	v_lshl_add_u32 v162, v1, 1, s63
	v_lshl_add_u32 v164, v227, 1, s63
	ds_read_b64_tr_b16 v[190:191], v162
	ds_read_b64_tr_b16 v[192:193], v162 offset:2048
	ds_read_b64_tr_b16 v[186:187], v162 offset:4096
	ds_read_b64_tr_b16 v[188:189], v162 offset:6144
	ds_read_b64_tr_b16 v[182:183], v162 offset:8192
	ds_read_b64_tr_b16 v[184:185], v162 offset:10240
	ds_read_b64_tr_b16 v[178:179], v162 offset:12288
	ds_read_b64_tr_b16 v[180:181], v162 offset:14336
	ds_read_b64_tr_b16 v[174:175], v164
	ds_read_b64_tr_b16 v[176:177], v164 offset:2048
	ds_read_b64_tr_b16 v[170:171], v164 offset:4096
	ds_read_b64_tr_b16 v[172:173], v164 offset:6144
	ds_read_b64_tr_b16 v[166:167], v164 offset:8192
	ds_read_b64_tr_b16 v[168:169], v164 offset:10240
	ds_read_b64_tr_b16 v[162:163], v164 offset:12288
	ds_read_b64_tr_b16 v[164:165], v164 offset:14336
	s_mov_b32 s17, 0x3e0293ee
	s_andn2_b64 vcc, exec, s[50:51]
	s_cbranch_vccnz .LBB0_23
; %bb.22:
	v_mul_f32_e32 v232, 0x3e0293ee, v229
	v_fma_f32 v98, v98, s17, -v232
	v_fma_f32 v99, v99, s17, -v232
	v_fma_f32 v100, v100, s17, -v232
	v_exp_f32_e32 v233, v98
	v_exp_f32_e32 v234, v99
	v_fma_f32 v101, v101, s17, -v232
	v_mov_b32_e32 v243, v235
	v_exp_f32_e32 v235, v100
	v_fma_f32 v102, v102, s17, -v232
	v_exp_f32_e32 v236, v101
	v_fma_f32 v103, v103, s17, -v232
	v_fma_f32 v114, v114, s17, -v232
	v_exp_f32_e32 v237, v102
	v_fma_f32 v104, v104, s17, -v232
	v_mov_b32_e32 v244, v238
	v_exp_f32_e32 v238, v103
	v_exp_f32_e32 v102, v114
	v_add_f32_e32 v114, v233, v234
	v_fma_f32 v105, v105, s17, -v232
	v_exp_f32_e32 v239, v104
	v_add_f32_e32 v114, v235, v114
	v_fma_f32 v106, v106, s17, -v232
	v_exp_f32_e32 v240, v105
	v_add_f32_e32 v114, v236, v114
	v_fma_f32 v107, v107, s17, -v232
	v_exp_f32_e32 v106, v106
	v_add_f32_e32 v114, v237, v114
	v_fma_f32 v108, v108, s17, -v232
	v_exp_f32_e32 v241, v107
	v_add_f32_e32 v114, v238, v114
	v_fma_f32 v109, v109, s17, -v232
	v_exp_f32_e32 v107, v108
	v_add_f32_e32 v114, v239, v114
	v_fma_f32 v110, v110, s17, -v232
	v_exp_f32_e32 v242, v109
	v_add_f32_e32 v114, v240, v114
	v_fma_f32 v111, v111, s17, -v232
	v_exp_f32_e32 v108, v110
	v_add_f32_e32 v114, v106, v114
	v_exp_f32_e32 v110, v111
	v_add_f32_e32 v114, v241, v114
	v_add_f32_e32 v114, v107, v114
	v_add_f32_e32 v114, v242, v114
	v_add_f32_e32 v114, v108, v114
	v_add_f32_e32 v114, v110, v114
	v_cvt_pk_f16_f32 v108, v108, v110
	v_cvt_pk_f16_f32 v110, v233, v234
	scratch_load_dword v234, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v233, off, off       ; 4-byte Folded Reload
	v_fma_f32 v112, v112, s17, -v232
	v_fma_f32 v113, v113, s17, -v232
	v_exp_f32_e32 v109, v112
	v_exp_f32_e32 v111, v113
	v_fma_f32 v115, v115, s17, -v232
	v_fma_f32 v116, v116, s17, -v232
	v_fma_f32 v122, v122, s17, -v232
	v_exp_f32_e32 v112, v115
	v_fma_f32 v117, v117, s17, -v232
	v_exp_f32_e32 v103, v116
	v_exp_f32_e32 v98, v122
	v_add_f32_e32 v114, v109, v114
	v_fma_f32 v122, v231, s17, -v232
	v_exp_f32_e32 v113, v117
	v_add_f32_e32 v114, v111, v114
	v_exp_f32_e32 v122, v122
	v_add_f32_e32 v114, v102, v114
	v_add_f32_e32 v114, v112, v114
	v_add_f32_e32 v114, v103, v114
	v_add_f32_e32 v114, v113, v114
	v_cvt_pk_f16_f32 v103, v103, v113
	v_cvt_pk_f16_f32 v102, v102, v112
	v_cvt_pk_f16_f32 v109, v109, v111
	v_cvt_pk_f16_f32 v113, v239, v240
	v_cvt_pk_f16_f32 v112, v237, v238
	v_cvt_pk_f16_f32 v111, v235, v236
	v_mul_f32_e32 v2, v2, v122
	v_mul_f32_e32 v3, v3, v122
	v_mul_f32_e32 v4, v4, v122
	v_mul_f32_e32 v5, v5, v122
	v_mul_f32_e32 v6, v6, v122
	v_mul_f32_e32 v7, v7, v122
	v_mul_f32_e32 v8, v8, v122
	v_mul_f32_e32 v9, v9, v122
	v_mul_f32_e32 v10, v10, v122
	v_mul_f32_e32 v11, v11, v122
	v_mul_f32_e32 v12, v12, v122
	v_mul_f32_e32 v13, v13, v122
	v_mul_f32_e32 v14, v14, v122
	v_mul_f32_e32 v15, v15, v122
	v_mul_f32_e32 v16, v16, v122
	v_mul_f32_e32 v17, v17, v122
	v_mul_f32_e32 v18, v18, v122
	v_mul_f32_e32 v19, v19, v122
	v_mul_f32_e32 v20, v20, v122
	v_mul_f32_e32 v21, v21, v122
	v_mul_f32_e32 v22, v22, v122
	v_mul_f32_e32 v23, v23, v122
	v_mul_f32_e32 v24, v24, v122
	v_mul_f32_e32 v25, v25, v122
	v_mul_f32_e32 v26, v26, v122
	v_mul_f32_e32 v27, v27, v122
	v_mul_f32_e32 v28, v28, v122
	v_mul_f32_e32 v29, v29, v122
	v_mul_f32_e32 v30, v30, v122
	v_mul_f32_e32 v31, v31, v122
	v_mul_f32_e32 v32, v32, v122
	v_mul_f32_e32 v33, v33, v122
	v_mul_f32_e32 v34, v34, v122
	v_mul_f32_e32 v35, v35, v122
	v_mul_f32_e32 v36, v36, v122
	v_mul_f32_e32 v37, v37, v122
	v_mul_f32_e32 v38, v38, v122
	v_mul_f32_e32 v39, v39, v122
	v_mul_f32_e32 v40, v40, v122
	v_mul_f32_e32 v41, v41, v122
	v_mul_f32_e32 v42, v42, v122
	v_mul_f32_e32 v43, v43, v122
	v_mul_f32_e32 v44, v44, v122
	v_mul_f32_e32 v45, v45, v122
	v_mul_f32_e32 v46, v46, v122
	v_mul_f32_e32 v47, v47, v122
	v_mul_f32_e32 v48, v48, v122
	v_mul_f32_e32 v49, v49, v122
	v_mul_f32_e32 v50, v50, v122
	v_mul_f32_e32 v51, v51, v122
	v_mul_f32_e32 v52, v52, v122
	v_mul_f32_e32 v53, v53, v122
	v_mul_f32_e32 v54, v54, v122
	v_mul_f32_e32 v55, v55, v122
	v_mul_f32_e32 v56, v56, v122
	v_mul_f32_e32 v57, v57, v122
	v_mul_f32_e32 v58, v58, v122
	v_mul_f32_e32 v59, v59, v122
	v_mul_f32_e32 v60, v60, v122
	v_mul_f32_e32 v61, v61, v122
	v_mul_f32_e32 v62, v62, v122
	v_mul_f32_e32 v63, v63, v122
	v_mul_f32_e32 v64, v64, v122
	v_mul_f32_e32 v65, v65, v122
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[2:17], v[222:225], v[110:113], v[2:17]
	v_cvt_pk_f16_f32 v107, v107, v242
	v_cvt_pk_f16_f32 v106, v106, v241
	v_fma_f32 v118, v118, s17, -v232
	v_fma_f32 v119, v119, s17, -v232
	v_exp_f32_e32 v104, v118
	v_fma_f32 v120, v120, s17, -v232
	v_fma_f32 v121, v121, s17, -v232
	v_mfma_f32_32x32x16_f16 v[18:33], v[206:209], v[110:113], v[18:33]
	v_exp_f32_e32 v115, v119
	v_exp_f32_e32 v105, v120
	v_exp_f32_e32 v116, v121
	v_fma_f32 v123, v123, s17, -v232
	v_add_f32_e32 v114, v104, v114
	v_fma_f32 v124, v124, s17, -v232
	v_exp_f32_e32 v117, v123
	v_mfma_f32_32x32x16_f16 v[34:49], v[190:193], v[110:113], v[34:49]
	v_add_f32_e32 v114, v115, v114
	v_fma_f32 v125, v125, s17, -v232
	v_exp_f32_e32 v99, v124
	v_add_f32_e32 v114, v105, v114
	v_cvt_pk_f16_f32 v105, v105, v116
	v_cvt_pk_f16_f32 v104, v104, v115
	v_fma_f32 v126, v126, s17, -v232
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[50:65], v[174:177], v[110:113], v[50:65]
	v_exp_f32_e32 v118, v125
	v_add_f32_e32 v114, v116, v114
	v_fma_f32 v127, v127, s17, -v232
	v_exp_f32_e32 v100, v126
	v_add_f32_e32 v114, v98, v114
	v_fma_f32 v128, v128, s17, -v232
	v_fma_f32 v129, v129, s17, -v232
	v_mfma_f32_32x32x16_f16 v[2:17], v[218:221], v[106:109], v[2:17]
	v_exp_f32_e32 v119, v127
	v_add_f32_e32 v114, v117, v114
	v_exp_f32_e32 v101, v128
	v_exp_f32_e32 v120, v129
	v_add_f32_e32 v114, v99, v114
	v_add_f32_e32 v114, v118, v114
	v_add_f32_e32 v114, v100, v114
	v_mfma_f32_32x32x16_f16 v[18:33], v[202:205], v[106:109], v[18:33]
	v_add_f32_e32 v114, v119, v114
	v_add_f32_e32 v114, v101, v114
	v_cvt_pk_f16_f32 v101, v101, v120
	v_cvt_pk_f16_f32 v100, v100, v119
	v_cvt_pk_f16_f32 v99, v99, v118
	v_cvt_pk_f16_f32 v98, v98, v117
	v_add_f32_e32 v114, v120, v114
	v_mfma_f32_32x32x16_f16 v[34:49], v[186:189], v[106:109], v[34:49]
	v_mov_b32_e32 v121, v114
	s_nop 1
	v_permlane32_swap_b32_e32 v114, v121
	v_add_f32_e32 v114, v114, v121
	v_fmac_f32_e32 v114, v247, v122
	v_mov_b32_e32 v238, v244
	v_mov_b32_e32 v235, v243
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[170:173], v[106:109], v[50:65]
	v_mov_b32_e32 v247, v114
	v_mfma_f32_32x32x16_f16 v[2:17], v[214:217], v[102:105], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[198:201], v[102:105], v[18:33]
	v_mfma_f32_32x32x16_f16 v[34:49], v[182:185], v[102:105], v[34:49]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[166:169], v[102:105], v[50:65]
	v_mfma_f32_32x32x16_f16 v[2:17], v[210:213], v[98:101], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[194:197], v[98:101], v[18:33]
	v_mfma_f32_32x32x16_f16 v[34:49], v[178:181], v[98:101], v[34:49]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[162:165], v[98:101], v[50:65]
.LBB0_23:
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
	s_add_i32 s19, s18, 0x8000
	v_max3_f32 v98, v98, v94, v95
	s_waitcnt lgkmcnt(14)
	v_max3_f32 v195, v98, v96, v97
	v_lshl_add_u32 v98, v245, 1, s18
	v_lshl_add_u32 v99, v245, 1, s19
	ds_read_b64_tr_b16 v[190:191], v98 offset:32768
	ds_read_b64_tr_b16 v[192:193], v99 offset:2048
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[186:187], v99 offset:4096
	ds_read_b64_tr_b16 v[188:189], v99 offset:6144
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[182:183], v99 offset:8192
	ds_read_b64_tr_b16 v[184:185], v99 offset:10240
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[178:179], v99 offset:12288
	ds_read_b64_tr_b16 v[180:181], v99 offset:14336
	scratch_load_dword v99, off, off offset:8 ; 4-byte Folded Reload
	v_lshl_add_u32 v100, v227, 1, s19
	v_mov_b32_e32 v196, v195
	v_cndmask_b32_e64 v194, v229, v231, s[12:13]
	s_nop 0
	v_permlane32_swap_b32_e32 v195, v196
	s_and_b64 vcc, exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v98, v99, 1, s18
	v_lshl_add_u32 v99, v99, 1, s19
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[174:175], v98 offset:32768
	ds_read_b64_tr_b16 v[176:177], v99 offset:2048
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[170:171], v99 offset:4096
	ds_read_b64_tr_b16 v[172:173], v99 offset:6144
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[166:167], v99 offset:8192
	ds_read_b64_tr_b16 v[168:169], v99 offset:10240
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[162:163], v99 offset:12288
	ds_read_b64_tr_b16 v[164:165], v99 offset:14336
	v_lshl_add_u32 v98, v1, 1, s18
	v_lshl_add_u32 v1, v1, 1, s19
	ds_read_b64_tr_b16 v[126:127], v98 offset:32768
	ds_read_b64_tr_b16 v[128:129], v1 offset:2048
	ds_read_b64_tr_b16 v[122:123], v1 offset:4096
	ds_read_b64_tr_b16 v[124:125], v1 offset:6144
	ds_read_b64_tr_b16 v[118:119], v1 offset:8192
	ds_read_b64_tr_b16 v[120:121], v1 offset:10240
	ds_read_b64_tr_b16 v[114:115], v1 offset:12288
	ds_read_b64_tr_b16 v[116:117], v1 offset:14336
	v_lshl_add_u32 v1, v227, 1, s18
	ds_read_b64_tr_b16 v[110:111], v1 offset:32768
	ds_read_b64_tr_b16 v[112:113], v100 offset:2048
	ds_read_b64_tr_b16 v[106:107], v100 offset:4096
	ds_read_b64_tr_b16 v[108:109], v100 offset:6144
	ds_read_b64_tr_b16 v[102:103], v100 offset:8192
	ds_read_b64_tr_b16 v[104:105], v100 offset:10240
	ds_read_b64_tr_b16 v[98:99], v100 offset:12288
	ds_read_b64_tr_b16 v[100:101], v100 offset:14336
	v_max3_f32 v1, v194, v195, v196
	s_cbranch_vccnz .LBB0_25
; %bb.24:
	v_mul_f32_e32 v195, 0x3e0293ee, v1
	v_fma_f32 v66, v66, s17, -v195
	s_mov_b32 s12, 0x3e0293ee
	v_fma_f32 v67, v67, s17, -v195
	v_fma_f32 v68, v68, s17, -v195
	v_fma_f32 v69, v69, s17, -v195
	v_fma_f32 v70, v70, s17, -v195
	v_fma_f32 v71, v71, s17, -v195
	v_fma_f32 v72, v72, s17, -v195
	v_fma_f32 v73, v73, s17, -v195
	v_exp_f32_e32 v196, v66
	v_fma_f32 v66, v194, s12, -v195
	v_fma_f32 v74, v74, s17, -v195
	v_fma_f32 v75, v75, s17, -v195
	v_fma_f32 v76, v76, s17, -v195
	v_fma_f32 v77, v77, s17, -v195
	v_fma_f32 v78, v78, s17, -v195
	v_fma_f32 v79, v79, s17, -v195
	v_fma_f32 v80, v80, s17, -v195
	v_fma_f32 v81, v81, s17, -v195
	v_fma_f32 v82, v82, s17, -v195
	v_fma_f32 v83, v83, s17, -v195
	v_fma_f32 v84, v84, s17, -v195
	v_fma_f32 v85, v85, s17, -v195
	v_fma_f32 v86, v86, s17, -v195
	v_fma_f32 v87, v87, s17, -v195
	v_fma_f32 v88, v88, s17, -v195
	v_fma_f32 v89, v89, s17, -v195
	v_fma_f32 v90, v90, s17, -v195
	v_fma_f32 v91, v91, s17, -v195
	v_fma_f32 v92, v92, s17, -v195
	v_fma_f32 v93, v93, s17, -v195
	v_fma_f32 v94, v94, s17, -v195
	v_fma_f32 v95, v95, s17, -v195
	v_fma_f32 v96, v96, s17, -v195
	v_fma_f32 v97, v97, s17, -v195
	v_exp_f32_e32 v197, v67
	v_exp_f32_e32 v198, v68
	v_exp_f32_e32 v199, v69
	v_exp_f32_e32 v200, v70
	v_exp_f32_e32 v201, v71
	v_exp_f32_e32 v202, v72
	v_exp_f32_e32 v203, v73
	v_exp_f32_e32 v195, v66
	v_cvt_pk_f16_f32 v68, v200, v201
	v_cvt_pk_f16_f32 v67, v198, v199
	v_cvt_pk_f16_f32 v69, v202, v203
	v_cvt_pk_f16_f32 v66, v196, v197
	v_mul_f32_e32 v32, v32, v195
	v_mul_f32_e32 v33, v33, v195
	v_mul_f32_e32 v30, v30, v195
	v_mul_f32_e32 v31, v31, v195
	v_mul_f32_e32 v28, v28, v195
	v_mul_f32_e32 v29, v29, v195
	v_mul_f32_e32 v26, v26, v195
	v_mul_f32_e32 v27, v27, v195
	v_mul_f32_e32 v24, v24, v195
	v_mul_f32_e32 v25, v25, v195
	v_mul_f32_e32 v22, v22, v195
	v_mul_f32_e32 v23, v23, v195
	v_mul_f32_e32 v20, v20, v195
	v_mul_f32_e32 v21, v21, v195
	v_mul_f32_e32 v18, v18, v195
	v_mul_f32_e32 v19, v19, v195
	v_exp_f32_e32 v204, v74
	v_exp_f32_e32 v205, v75
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[66:69], v[18:33]
	v_exp_f32_e32 v206, v76
	v_exp_f32_e32 v207, v77
	v_exp_f32_e32 v208, v78
	v_exp_f32_e32 v209, v79
	v_exp_f32_e32 v210, v80
	v_exp_f32_e32 v211, v81
	v_cvt_pk_f16_f32 v71, v206, v207
	v_cvt_pk_f16_f32 v72, v208, v209
	v_cvt_pk_f16_f32 v70, v204, v205
	v_cvt_pk_f16_f32 v73, v210, v211
	v_mul_f32_e32 v48, v48, v195
	v_mul_f32_e32 v49, v49, v195
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[70:73], v[18:33]
	v_add_f32_e32 v170, v196, v197
	v_add_f32_e32 v170, v198, v170
	v_mul_f32_e32 v46, v46, v195
	v_mul_f32_e32 v47, v47, v195
	v_mul_f32_e32 v44, v44, v195
	v_mul_f32_e32 v45, v45, v195
	v_mul_f32_e32 v42, v42, v195
	v_mul_f32_e32 v43, v43, v195
	v_mul_f32_e32 v40, v40, v195
	v_mul_f32_e32 v41, v41, v195
	v_mul_f32_e32 v38, v38, v195
	v_mul_f32_e32 v39, v39, v195
	v_mul_f32_e32 v36, v36, v195
	v_mul_f32_e32 v37, v37, v195
	v_mul_f32_e32 v34, v34, v195
	v_mul_f32_e32 v35, v35, v195
	v_add_f32_e32 v170, v199, v170
	v_mul_f32_e32 v16, v16, v195
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[66:69], v[34:49]
	v_mul_f32_e32 v17, v17, v195
	v_mul_f32_e32 v14, v14, v195
	v_mul_f32_e32 v15, v15, v195
	v_mul_f32_e32 v12, v12, v195
	v_mul_f32_e32 v13, v13, v195
	v_mul_f32_e32 v10, v10, v195
	v_mul_f32_e32 v11, v11, v195
	v_mul_f32_e32 v8, v8, v195
	v_mul_f32_e32 v9, v9, v195
	v_mul_f32_e32 v6, v6, v195
	v_mul_f32_e32 v7, v7, v195
	v_mul_f32_e32 v4, v4, v195
	v_mul_f32_e32 v5, v5, v195
	v_mul_f32_e32 v2, v2, v195
	v_mul_f32_e32 v3, v3, v195
	v_add_f32_e32 v126, v200, v170
	v_mul_f32_e32 v64, v64, v195
	v_mul_f32_e32 v65, v65, v195
	v_mul_f32_e32 v62, v62, v195
	v_mul_f32_e32 v63, v63, v195
	v_mul_f32_e32 v60, v60, v195
	v_mul_f32_e32 v61, v61, v195
	v_mul_f32_e32 v58, v58, v195
	v_mul_f32_e32 v59, v59, v195
	v_mul_f32_e32 v56, v56, v195
	v_mul_f32_e32 v57, v57, v195
	v_mul_f32_e32 v54, v54, v195
	v_mul_f32_e32 v55, v55, v195
	v_mul_f32_e32 v52, v52, v195
	v_mul_f32_e32 v53, v53, v195
	v_mul_f32_e32 v50, v50, v195
	v_mul_f32_e32 v51, v51, v195
	v_mfma_f32_32x32x16_f16 v[2:17], v[190:193], v[66:69], v[2:17]
	v_add_f32_e32 v126, v201, v126
	v_add_f32_e32 v126, v202, v126
	v_add_f32_e32 v126, v203, v126
	v_add_f32_e32 v126, v204, v126
	v_add_f32_e32 v126, v205, v126
	v_add_f32_e32 v126, v206, v126
	v_exp_f32_e32 v82, v82
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[50:65], v[110:113], v[66:69], v[50:65]
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v85, v85
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[70:73], v[34:49]
	v_add_f32_e32 v122, v207, v126
	v_add_f32_e32 v122, v208, v122
	v_add_f32_e32 v122, v209, v122
	v_add_f32_e32 v122, v210, v122
	v_add_f32_e32 v122, v211, v122
	v_add_f32_e32 v66, v82, v122
	v_add_f32_e32 v66, v83, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[186:189], v[70:73], v[2:17]
	v_add_f32_e32 v66, v84, v66
	v_add_f32_e32 v66, v85, v66
	v_exp_f32_e32 v90, v90
	v_cvt_pk_f16_f32 v77, v88, v89
	v_cvt_pk_f16_f32 v76, v86, v87
	v_cvt_pk_f16_f32 v75, v84, v85
	v_cvt_pk_f16_f32 v74, v82, v83
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[106:109], v[70:73], v[50:65]
	v_add_f32_e32 v66, v86, v66
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v66, v87, v66
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v66, v88, v66
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v66, v89, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[182:185], v[74:77], v[2:17]
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_add_f32_e32 v66, v90, v66
	v_add_f32_e32 v66, v91, v66
	v_add_f32_e32 v66, v92, v66
	v_mfma_f32_32x32x16_f16 v[18:33], v[166:169], v[74:77], v[18:33]
	v_add_f32_e32 v66, v93, v66
	v_cvt_pk_f16_f32 v81, v96, v97
	v_cvt_pk_f16_f32 v80, v94, v95
	v_cvt_pk_f16_f32 v79, v92, v93
	v_cvt_pk_f16_f32 v78, v90, v91
	v_add_f32_e32 v66, v94, v66
	v_add_f32_e32 v66, v95, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[118:121], v[74:77], v[34:49]
	v_add_f32_e32 v66, v96, v66
	v_add_f32_e32 v66, v97, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v247, v195
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[102:105], v[74:77], v[50:65]
	v_mov_b32_e32 v247, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[178:181], v[78:81], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[162:165], v[78:81], v[18:33]
	v_mfma_f32_32x32x16_f16 v[34:49], v[114:117], v[78:81], v[34:49]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[98:101], v[78:81], v[50:65]
.LBB0_25:                               ; %Flow2117
	s_waitcnt lgkmcnt(0)
	scratch_load_dword v100, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v101, off, off offset:88 ; 4-byte Folded Reload
	scratch_load_dword v78, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v79, off, off offset:120 ; 4-byte Folded Reload
	scratch_load_dword v80, off, off offset:124 ; 4-byte Folded Reload
	scratch_load_dword v81, off, off offset:116 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[82:83], off, off offset:100 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[72:73], off, off offset:128 ; 8-byte Folded Reload
	scratch_load_dword v70, off, off offset:136 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[74:75], off, off offset:140 ; 8-byte Folded Reload
	scratch_load_dword v71, off, off offset:148 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v198, v194, v1, s[2:3]
.LBB0_26:                               ; %Flow2121
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v77, off, off offset:92 ; 4-byte Folded Reload
	s_mul_i32 s14, s42, s37
	s_ashr_i32 s15, s14, 31
	s_lshl_b32 s59, s69, 6
	s_sub_i32 s53, s33, s67
	s_lshl_b64 s[2:3], s[14:15], 1
	s_add_u32 s16, s16, s2
	s_mul_i32 s50, s42, s40
	s_addc_u32 s65, s36, s3
	s_ashr_i32 s51, s50, 31
	s_lshl_b64 s[2:3], s[50:51], 1
	s_add_u32 s20, s48, s2
	s_addc_u32 s66, s49, s3
	s_cmp_lg_u32 s41, 0
	s_cselect_b64 s[36:37], -1, 0
	s_cmp_gt_u32 s59, s42
	s_cselect_b64 s[22:23], -1, 0
	s_and_b32 s12, s65, 0xffff
	s_lshl_b32 s63, s55, 16
	s_or_b32 s17, s12, s63
	s_waitcnt vmcnt(3)
	ds_bpermute_b32 v70, v70, v226
	v_or_b32_e32 v67, s42, v79
	v_cmp_gt_i32_e64 s[2:3], s67, v67
	v_add_u32_e32 v68, 0, v251
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	s_waitcnt vmcnt(1)
	ds_bpermute_b32 v71, v71, v78
	s_and_b64 s[2:3], s[22:23], s[2:3]
	v_add_u32_e32 v69, 0, v252
	ds_bpermute_b32 v73, v238, v228
	s_add_i32 s61, 0, 0x10000
	s_lshl_b32 s57, s57, 16
	ds_bpermute_b32 v75, v246, v226
	v_and_b32_e32 v1, 64, v80
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v73, 1, v73
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v75, 1, v75
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v66, s42, v77
	v_cmp_gt_i32_e32 vcc, s67, v66
	s_and_b64 s[12:13], s[22:23], vcc
	v_cndmask_b32_e64 v66, 0, 1, s[12:13]
	v_cmp_ne_u32_e32 vcc, 0, v66
	s_mov_b32 s22, s18
	s_mov_b32 s23, s19
	v_lshrrev_b64 v[66:67], v72, vcc
	v_and_b32_e32 v66, 1, v66
	v_lshlrev_b32_e32 v67, 1, v70
	v_bfrev_b32_e32 v70, 1
	v_cmp_eq_u32_e64 s[12:13], 1, v66
	v_add_u32_e32 v72, s61, v252
	s_nop 0
	v_cndmask_b32_e64 v66, v70, v67, s[12:13]
	v_readfirstlane_b32 s12, v68
	s_mov_b32 m0, s12
	s_nop 0
	buffer_load_dwordx4 v66, s[16:19], 0 offen lds
	v_cndmask_b32_e64 v66, 0, 1, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v66
	s_nop 1
	v_lshrrev_b64 v[66:67], v74, s[2:3]
	v_and_b32_e32 v66, 1, v66
	v_lshlrev_b32_e32 v67, 1, v71
	v_cmp_eq_u32_e64 s[12:13], 1, v66
	v_add_u32_e32 v71, s61, v251
	v_add_u32_e32 v74, 0x4000, v69
	v_cndmask_b32_e64 v66, v70, v67, s[12:13]
	v_readfirstlane_b32 s12, v69
	s_mov_b32 m0, s12
	s_and_b32 s12, s66, 0xffff
	buffer_load_dwordx4 v66, s[16:19], 0 offen lds
	v_lshrrev_b64 v[66:67], v82, vcc
	s_or_b32 s21, s12, s57
	v_and_b32_e32 v66, 1, v66
	v_readfirstlane_b32 s12, v71
	ds_bpermute_b32 v71, v238, v235
	v_cmp_eq_u32_e32 vcc, 1, v66
	s_mov_b32 m0, s12
	v_add_u32_e32 v69, 0x8000, v69
	v_cndmask_b32_e32 v66, v70, v73, vcc
	buffer_load_dwordx4 v66, s[20:23], 0 offen lds
	v_lshrrev_b64 v[66:67], v82, s[2:3]
	v_and_b32_e32 v66, 1, v66
	v_readfirstlane_b32 s2, v72
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v71, 1, v71
	v_cmp_eq_u32_e32 vcc, 1, v66
	s_mov_b32 m0, s2
	s_add_i32 s2, s42, 64
	v_cndmask_b32_e32 v66, v70, v71, vcc
	s_cmp_lt_u32 s2, s59
	buffer_load_dwordx4 v66, s[20:23], 0 offen lds
	s_cselect_b64 s[22:23], -1, 0
	s_lshl_b64 s[40:41], s[44:45], 1
	s_add_u32 s16, s16, s40
	s_addc_u32 s65, s65, s41
	s_lshl_b64 s[48:49], s[46:47], 1
	s_add_u32 s20, s20, s48
	v_or_b32_e32 v66, s2, v77
	s_addc_u32 s66, s66, s49
	v_cmp_gt_i32_e32 vcc, s67, v66
	s_and_b32 s12, s65, 0xffff
	s_or_b32 s17, s12, s63
	s_and_b64 s[12:13], s[22:23], vcc
	v_cndmask_b32_e64 v66, 0, 1, s[12:13]
	v_or_b32_e32 v67, s2, v79
	v_cmp_ne_u32_e32 vcc, 0, v66
	v_cmp_gt_i32_e64 s[2:3], s67, v67
	v_add_u32_e32 v72, 0x4000, v68
	v_lshrrev_b64 v[66:67], v230, vcc
	v_and_b32_e32 v66, 1, v66
	v_cmp_eq_u32_e64 s[12:13], 1, v66
	s_and_b64 s[2:3], s[22:23], s[2:3]
	s_add_i32 s55, 0, 0x14000
	v_cndmask_b32_e64 v66, v70, v75, s[12:13]
	v_readfirstlane_b32 s12, v72
	s_mov_b32 m0, s12
	ds_bpermute_b32 v72, v246, v78
	buffer_load_dwordx4 v66, s[16:19], 0 offen lds
	v_cndmask_b32_e64 v66, 0, 1, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v66
	s_mov_b32 s22, s18
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v72, 1, v72
	v_lshrrev_b64 v[66:67], v230, s[2:3]
	v_and_b32_e32 v66, 1, v66
	v_cmp_eq_u32_e64 s[12:13], 1, v66
	s_mov_b32 s23, s19
	v_add_u32_e32 v76, s55, v252
	v_cndmask_b32_e64 v66, v70, v72, s[12:13]
	v_readfirstlane_b32 s12, v74
	s_mov_b32 m0, s12
	v_add_u32_e32 v74, s55, v251
	buffer_load_dwordx4 v66, s[16:19], 0 offen lds
	v_lshrrev_b64 v[66:67], v82, vcc
	s_and_b32 s12, s66, 0xffff
	v_and_b32_e32 v66, 1, v66
	s_or_b32 s21, s12, s57
	v_cmp_eq_u32_e32 vcc, 1, v66
	v_readfirstlane_b32 s12, v74
	s_mov_b32 m0, s12
	v_cndmask_b32_e32 v66, v70, v73, vcc
	buffer_load_dwordx4 v66, s[20:23], 0 offen lds
	v_lshrrev_b64 v[66:67], v82, s[2:3]
	v_and_b32_e32 v66, 1, v66
	v_readfirstlane_b32 s2, v76
	v_cmp_eq_u32_e32 vcc, 1, v66
	s_mov_b32 m0, s2
	s_add_i32 s2, s42, 0x80
	v_cndmask_b32_e32 v66, v70, v71, vcc
	s_cmp_lt_u32 s2, s59
	buffer_load_dwordx4 v66, s[20:23], 0 offen lds
	s_cselect_b64 s[22:23], -1, 0
	s_add_u32 s16, s16, s40
	s_addc_u32 s12, s65, s41
	s_add_u32 s20, s20, s48
	v_or_b32_e32 v66, s2, v77
	s_addc_u32 s21, s66, s49
	v_cmp_gt_i32_e32 vcc, s67, v66
	s_and_b32 s12, s12, 0xffff
	s_or_b32 s17, s12, s63
	s_and_b64 s[12:13], s[22:23], vcc
	v_cndmask_b32_e64 v66, 0, 1, s[12:13]
	v_or_b32_e32 v67, s2, v79
	v_cmp_ne_u32_e32 vcc, 0, v66
	v_cmp_gt_i32_e64 s[2:3], s67, v67
	v_add_u32_e32 v68, 0x8000, v68
	v_lshrrev_b64 v[66:67], v230, vcc
	v_and_b32_e32 v66, 1, v66
	v_cmp_eq_u32_e64 s[12:13], 1, v66
	s_and_b64 s[2:3], s[22:23], s[2:3]
	s_mov_b32 s22, s18
	v_cndmask_b32_e64 v66, v70, v75, s[12:13]
	v_readfirstlane_b32 s12, v68
	s_mov_b32 m0, s12
	s_mov_b32 s23, s19
	buffer_load_dwordx4 v66, s[16:19], 0 offen lds
	v_cndmask_b32_e64 v66, 0, 1, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v66
	s_nop 1
	v_lshrrev_b64 v[66:67], v230, s[2:3]
	v_and_b32_e32 v66, 1, v66
	v_cmp_eq_u32_e64 s[12:13], 1, v66
	s_nop 1
	v_cndmask_b32_e64 v66, v70, v72, s[12:13]
	v_readfirstlane_b32 s12, v69
	s_mov_b32 m0, s12
	s_and_b32 s12, s21, 0xffff
	buffer_load_dwordx4 v66, s[16:19], 0 offen lds
	s_add_i32 s16, 0, 0x18000
	v_lshrrev_b64 v[66:67], v82, vcc
	v_add_u32_e32 v68, s16, v251
	v_and_b32_e32 v66, 1, v66
	s_or_b32 s21, s12, s57
	v_cmp_eq_u32_e32 vcc, 1, v66
	v_readfirstlane_b32 s12, v68
	s_mov_b32 m0, s12
	v_cndmask_b32_e32 v66, v70, v73, vcc
	buffer_load_dwordx4 v66, s[20:23], 0 offen lds
	v_lshrrev_b64 v[66:67], v82, s[2:3]
	v_add_u32_e32 v69, s16, v252
	v_and_b32_e32 v66, 1, v66
	v_cmp_eq_u32_e32 vcc, 1, v66
	v_readfirstlane_b32 s2, v69
	s_mov_b32 m0, s2
	v_cndmask_b32_e32 v66, v70, v71, vcc
	buffer_load_dwordx4 v66, s[20:23], 0 offen lds
	v_and_b32_e32 v66, 0xa0, v80
	s_add_i32 s20, s59, 0xffffff40
	v_or3_b32 v67, v66, v248, v1
	s_mov_b32 s19, 0
	s_cmp_lt_i32 s42, s20
	v_lshlrev_b32_e32 v1, 7, v248
	s_cbranch_scc1 .LBB0_28
; %bb.27:                               ; %.._crit_edge2045_crit_edge
	v_lshlrev_b32_e32 v66, 7, v248
	s_mov_b64 s[2:3], 0
	s_branch .LBB0_29
.LBB0_28:
	s_mov_b64 s[2:3], -1
                                        ; implicit-def: $vgpr66
.LBB0_29:                               ; %Flow2115
	scratch_load_dword v240, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v241, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v242, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v243, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v244, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v248, off, off offset:96 ; 4-byte Folded Reload
	scratch_load_dword v245, off, off offset:92 ; 4-byte Folded Reload
	v_or_b32_e32 v196, s60, v67
	v_add_u32_e32 v68, s53, v100
	s_andn2_b64 vcc, exec, s[2:3]
	v_lshlrev_b32_e32 v197, 2, v0
	scratch_store_dword off, v68, off offset:8 ; 4-byte Folded Spill
	scratch_store_dword off, v197, off offset:100 ; 4-byte Folded Spill
	scratch_store_dword off, v67, off offset:116 ; 4-byte Folded Spill
	s_cbranch_vccnz .LBB0_35
; %bb.30:                               ; %.lr.ph2044
	scratch_load_dword v0, off, off offset:84 ; 4-byte Folded Reload
	v_and_b32_e32 v194, 12, v197
	s_movk_i32 s2, 0x60
	v_bitop3_b32 v199, v101, v194, s2 bitop3:0x4e
	s_mul_i32 s2, s47, 6
	s_mul_hi_u32 s3, s46, 6
	s_add_i32 s12, s3, s2
	s_add_u32 s2, s30, s34
	s_addc_u32 s3, s31, s35
	s_add_u32 s2, s2, s38
	s_addc_u32 s3, s3, s39
	s_add_u32 s2, s2, s50
	s_addc_u32 s3, s3, s51
	s_mul_i32 s13, s46, 6
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s2, s13, s2
	s_addc_u32 s3, s12, s3
	s_add_u32 s22, s6, s2
	s_addc_u32 s23, s7, s3
	s_mul_i32 s2, s45, 6
	s_mul_hi_u32 s3, s44, 6
	s_add_i32 s6, s3, s2
	s_add_u32 s2, s24, s26
	s_addc_u32 s3, s25, s27
	s_add_u32 s2, s2, s28
	s_addc_u32 s3, s3, s29
	s_add_u32 s2, s2, s14
	s_addc_u32 s3, s3, s15
	s_mul_i32 s7, s44, 6
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s2, s7, s2
	s_addc_u32 s3, s6, s3
	v_or_b32_e32 v66, v234, v194
	v_bitop3_b32 v68, v194, v234, 32 bitop3:0x36
	s_add_u32 s24, s4, s2
	s_mov_b32 s21, 2
	v_bitop3_b32 v195, v66, v233, 64 bitop3:0x36
	s_addc_u32 s25, s5, s3
	s_add_i32 s18, 0, 0x4000
	s_add_i32 s17, 0, 0x8000
	s_add_i32 s28, 0, 0x10000
	s_add_i32 s55, 0, 0x14000
	s_add_i32 s16, 0, 0x18000
	s_sub_i32 s26, s59, 64
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_xor_b64 s[6:7], s[36:37], -1
	s_mov_b32 s27, 0x3e0293ee
	v_lshlrev_b32_e32 v200, 1, v81
	v_lshlrev_b32_e32 v202, 1, v68
	v_bfrev_b32_e32 v203, 1
	v_mov_b32_e32 v204, 0xff800000
	s_mov_b32 s30, s42
	s_waitcnt vmcnt(0)
	v_and_or_b32 v67, v0, 3, v100
	scratch_load_dword v0, off, off offset:28 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v67, 7, v67
	v_lshlrev_b32_e32 v201, 1, v67
.LBB0_31:                               ; =>This Inner Loop Header: Depth=1
	s_add_i32 s2, s21, 1
	s_cmp_lt_i32 s2, 4
	s_cselect_b32 s21, s2, 0
	v_add_u32_e32 v66, s30, v245
	ds_bpermute_b32 v70, v246, v226
	v_add_u32_e32 v67, 0xc0, v66
	s_lshl_b32 s12, s21, 14
	s_mov_b32 s29, s19
	s_mov_b32 s19, s18
	s_mov_b32 s18, s17
	v_add_u32_e32 v66, 0xe0, v66
	v_cmp_gt_i32_e64 s[4:5], s67, v67
	s_add_i32 s17, s12, 0
	v_cmp_gt_i32_e64 s[2:3], s67, v66
	v_add_u32_e32 v68, s17, v251
	v_lshrrev_b64 v[66:67], v230, s[4:5]
	v_and_b32_e32 v66, 1, v66
	v_readfirstlane_b32 s31, v68
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v68, v246, v78
	s_and_b32 s12, s25, 0xffff
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v67, 1, v70
	v_cmp_eq_u32_e32 vcc, 1, v66
	s_or_b32 s13, s12, s63
	s_mov_b32 s12, s24
	v_cndmask_b32_e32 v66, v203, v67, vcc
	s_mov_b32 m0, s31
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v66, s[12:15], 0 offen lds
	v_lshrrev_b64 v[66:67], v230, s[2:3]
	v_add_u32_e32 v69, s17, v252
	v_and_b32_e32 v66, 1, v66
	v_lshlrev_b32_e32 v67, 1, v68
	v_cmp_eq_u32_e32 vcc, 1, v66
	v_readfirstlane_b32 s31, v69
	s_mov_b32 m0, s31
	v_cndmask_b32_e32 v66, v203, v67, vcc
	buffer_load_dwordx4 v66, s[12:15], 0 offen lds
	v_lshlrev_b32_e32 v66, 1, v1
	v_add3_u32 v67, s29, v254, v66
	s_waitcnt vmcnt(2)
	v_add3_u32 v68, s29, v0, v66
	v_add3_u32 v69, s29, v250, v66
	v_add3_u32 v70, s29, v255, v66
	v_add3_u32 v71, s29, v240, v66
	v_add3_u32 v72, s29, v241, v66
	v_add3_u32 v73, s29, v242, v66
	v_add3_u32 v66, s29, v243, v66
	ds_read_b128 v[162:165], v67
	ds_read_b128 v[106:109], v67 offset:8192
	ds_read_b128 v[166:169], v68
	ds_read_b128 v[110:113], v68 offset:8192
	ds_read_b128 v[170:173], v69
	ds_read_b128 v[114:117], v69 offset:8192
	ds_read_b128 v[174:177], v70
	ds_read_b128 v[118:121], v70 offset:8192
	ds_read_b128 v[178:181], v71
	ds_read_b128 v[122:125], v71 offset:8192
	ds_read_b128 v[182:185], v72
	ds_read_b128 v[126:129], v72 offset:8192
	ds_read_b128 v[186:189], v73
	ds_read_b128 v[102:105], v73 offset:8192
	ds_read_b128 v[190:193], v66
	ds_read_b128 v[98:101], v66 offset:8192
	s_lshl_b32 s12, s21, 13
	s_cmp_lg_u32 s26, s30
	s_cselect_b64 s[34:35], -1, 0
	s_or_b64 s[34:35], s[6:7], s[34:35]
	v_mov_b32_e32 v197, v235
	s_mov_b32 s61, s55
	s_mov_b32 s55, s16
	v_mov_b32_e32 v66, 0
	s_and_b64 vcc, exec, s[34:35]
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
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
	s_cbranch_vccnz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_31 Depth=1
	scratch_load_dword v66, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v66, s26, v66
	v_add_u32_e32 v67, 1, v66
	v_cmp_gt_i32_e32 vcc, s67, v66
	v_add_u32_e32 v68, 2, v66
	v_add_u32_e32 v69, 3, v66
	v_add_u32_e32 v70, 8, v66
	v_add_u32_e32 v71, 9, v66
	v_add_u32_e32 v72, 10, v66
	v_add_u32_e32 v73, 11, v66
	v_add_u32_e32 v74, 16, v66
	v_add_u32_e32 v75, 17, v66
	v_add_u32_e32 v76, 18, v66
	v_add_u32_e32 v77, 19, v66
	v_add_u32_e32 v78, 24, v66
	v_add_u32_e32 v79, 25, v66
	v_add_u32_e32 v80, 26, v66
	v_add_u32_e32 v81, 27, v66
	v_add_u32_e32 v82, 32, v66
	v_add_u32_e32 v83, 33, v66
	v_add_u32_e32 v84, 34, v66
	v_add_u32_e32 v85, 35, v66
	v_add_u32_e32 v86, 40, v66
	v_add_u32_e32 v87, 41, v66
	v_add_u32_e32 v88, 42, v66
	v_add_u32_e32 v89, 43, v66
	v_add_u32_e32 v90, 48, v66
	v_add_u32_e32 v91, 49, v66
	v_add_u32_e32 v92, 50, v66
	v_add_u32_e32 v93, 51, v66
	v_add_u32_e32 v94, 56, v66
	v_add_u32_e32 v95, 57, v66
	v_add_u32_e32 v96, 58, v66
	v_add_u32_e32 v97, 59, v66
	v_cndmask_b32_e64 v66, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v67
	s_nop 1
	v_cndmask_b32_e64 v67, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v68
	s_nop 1
	v_cndmask_b32_e64 v68, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v69
	s_nop 1
	v_cndmask_b32_e64 v69, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v70
	s_nop 1
	v_cndmask_b32_e64 v70, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v71
	s_nop 1
	v_cndmask_b32_e64 v71, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v72
	s_nop 1
	v_cndmask_b32_e64 v72, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v73
	s_nop 1
	v_cndmask_b32_e64 v73, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v74
	s_nop 1
	v_cndmask_b32_e64 v74, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v75
	s_nop 1
	v_cndmask_b32_e64 v75, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v76
	s_nop 1
	v_cndmask_b32_e64 v76, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v77
	s_nop 1
	v_cndmask_b32_e64 v77, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v78
	s_nop 1
	v_cndmask_b32_e64 v78, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v79
	s_nop 1
	v_cndmask_b32_e64 v79, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v80
	s_nop 1
	v_cndmask_b32_e64 v80, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v81
	s_nop 1
	v_cndmask_b32_e64 v81, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v82
	s_nop 1
	v_cndmask_b32_e64 v82, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v83
	s_nop 1
	v_cndmask_b32_e64 v83, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v84
	s_nop 1
	v_cndmask_b32_e64 v84, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v85
	s_nop 1
	v_cndmask_b32_e64 v85, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v86
	s_nop 1
	v_cndmask_b32_e64 v86, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v87
	s_nop 1
	v_cndmask_b32_e64 v87, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v88
	s_nop 1
	v_cndmask_b32_e64 v88, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v89
	s_nop 1
	v_cndmask_b32_e64 v89, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v90
	s_nop 1
	v_cndmask_b32_e64 v90, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v91
	s_nop 1
	v_cndmask_b32_e64 v91, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v92
	s_nop 1
	v_cndmask_b32_e64 v92, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v93
	s_nop 1
	v_cndmask_b32_e64 v93, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v94
	s_nop 1
	v_cndmask_b32_e64 v94, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v95
	s_nop 1
	v_cndmask_b32_e64 v95, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v96
	s_nop 1
	v_cndmask_b32_e64 v96, v204, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v97
	s_nop 1
	v_cndmask_b32_e64 v97, v204, 0, vcc
.LBB0_33:                               ;   in Loop: Header=BB0_31 Depth=1
	s_add_i32 s29, s30, 64
	; sched_barrier mask(0x00000000)
	scratch_load_dword v205, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v205, s30, v205
	v_add_u32_e32 v206, 1, v205
	v_cmp_ge_i32_e32 vcc, v196, v205
	v_add_u32_e32 v207, 2, v205
	v_add_u32_e32 v208, 3, v205
	v_cndmask_b32_e32 v66, v204, v66, vcc
	v_cmp_ge_i32_e32 vcc, v196, v206
	v_add_u32_e32 v209, 8, v205
	v_add_u32_e32 v210, 9, v205
	v_cndmask_b32_e32 v67, v204, v67, vcc
	v_cmp_ge_i32_e32 vcc, v196, v207
	v_add_u32_e32 v211, 10, v205
	v_add_u32_e32 v212, 11, v205
	v_cndmask_b32_e32 v68, v204, v68, vcc
	v_cmp_ge_i32_e32 vcc, v196, v208
	v_add_u32_e32 v213, 16, v205
	v_add_u32_e32 v214, 17, v205
	v_cndmask_b32_e32 v69, v204, v69, vcc
	v_cmp_ge_i32_e32 vcc, v196, v209
	v_add_u32_e32 v215, 18, v205
	v_add_u32_e32 v216, 19, v205
	v_cndmask_b32_e32 v70, v204, v70, vcc
	v_cmp_ge_i32_e32 vcc, v196, v210
	v_add_u32_e32 v217, 24, v205
	v_add_u32_e32 v218, 25, v205
	v_cndmask_b32_e32 v71, v204, v71, vcc
	v_cmp_ge_i32_e32 vcc, v196, v211
	v_add_u32_e32 v219, 26, v205
	v_add_u32_e32 v220, 27, v205
	v_cndmask_b32_e32 v72, v204, v72, vcc
	v_cmp_ge_i32_e32 vcc, v196, v212
	v_add_u32_e32 v221, 32, v205
	v_add_u32_e32 v222, 33, v205
	v_cndmask_b32_e32 v73, v204, v73, vcc
	v_cmp_ge_i32_e32 vcc, v196, v213
	v_add_u32_e32 v223, 34, v205
	v_add_u32_e32 v224, 35, v205
	v_cndmask_b32_e32 v74, v204, v74, vcc
	v_cmp_ge_i32_e32 vcc, v196, v214
	v_add_u32_e32 v225, 40, v205
	v_add_u32_e32 v227, 41, v205
	v_cndmask_b32_e32 v75, v204, v75, vcc
	v_cmp_ge_i32_e32 vcc, v196, v215
	v_add_u32_e32 v229, 42, v205
	v_add_u32_e32 v231, 43, v205
	v_cndmask_b32_e32 v76, v204, v76, vcc
	v_cmp_ge_i32_e32 vcc, v196, v216
	v_add_u32_e32 v232, 48, v205
	v_add_u32_e32 v233, 49, v205
	v_cndmask_b32_e32 v77, v204, v77, vcc
	v_cmp_ge_i32_e32 vcc, v196, v217
	v_add_u32_e32 v234, 50, v205
	v_add_u32_e32 v235, 51, v205
	v_cndmask_b32_e32 v78, v204, v78, vcc
	v_cmp_ge_i32_e32 vcc, v196, v218
	v_add_u32_e32 v236, 56, v205
	v_add_u32_e32 v237, 57, v205
	v_cndmask_b32_e32 v79, v204, v79, vcc
	v_cmp_ge_i32_e32 vcc, v196, v219
	v_add_u32_e32 v238, 58, v205
	v_add_u32_e32 v239, 59, v205
	v_cndmask_b32_e32 v80, v204, v80, vcc
	v_cmp_ge_i32_e32 vcc, v196, v220
	s_nop 1
	v_cndmask_b32_e32 v81, v204, v81, vcc
	v_cmp_ge_i32_e32 vcc, v196, v221
	s_nop 1
	v_cndmask_b32_e32 v82, v204, v82, vcc
	v_cmp_ge_i32_e32 vcc, v196, v222
	s_nop 1
	v_cndmask_b32_e32 v83, v204, v83, vcc
	v_cmp_ge_i32_e32 vcc, v196, v223
	s_nop 1
	v_cndmask_b32_e32 v84, v204, v84, vcc
	v_cmp_ge_i32_e32 vcc, v196, v224
	s_nop 1
	v_cndmask_b32_e32 v85, v204, v85, vcc
	v_cmp_ge_i32_e32 vcc, v196, v225
	s_nop 1
	v_cndmask_b32_e32 v86, v204, v86, vcc
	v_cmp_ge_i32_e32 vcc, v196, v227
	s_nop 1
	v_cndmask_b32_e32 v87, v204, v87, vcc
	v_cmp_ge_i32_e32 vcc, v196, v229
	s_nop 1
	v_cndmask_b32_e32 v88, v204, v88, vcc
	v_cmp_ge_i32_e32 vcc, v196, v231
	s_nop 1
	v_cndmask_b32_e32 v89, v204, v89, vcc
	v_cmp_ge_i32_e32 vcc, v196, v232
	s_nop 1
	v_cndmask_b32_e32 v90, v204, v90, vcc
	v_cmp_ge_i32_e32 vcc, v196, v233
	s_nop 1
	v_cndmask_b32_e32 v91, v204, v91, vcc
	v_cmp_ge_i32_e32 vcc, v196, v234
	s_nop 1
	v_cndmask_b32_e32 v92, v204, v92, vcc
	v_cmp_ge_i32_e32 vcc, v196, v235
	s_nop 1
	v_cndmask_b32_e32 v93, v204, v93, vcc
	v_cmp_ge_i32_e32 vcc, v196, v236
	s_nop 1
	v_cndmask_b32_e32 v94, v204, v94, vcc
	v_cmp_ge_i32_e32 vcc, v196, v237
	s_nop 1
	v_cndmask_b32_e32 v95, v204, v95, vcc
	v_cmp_ge_i32_e32 vcc, v196, v238
	s_nop 1
	v_cndmask_b32_e32 v96, v204, v96, vcc
	v_cmp_ge_i32_e32 vcc, v196, v239
	s_nop 1
	v_cndmask_b32_e32 v97, v204, v97, vcc
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[134:137], v[66:81]
	s_lshl_b32 s12, s12, 1
	s_add_i32 s12, s12, 0
	s_add_i32 s16, s12, 0x10000
	s_and_b32 s12, s23, 0xffff
	s_or_b32 s13, s12, s57
	s_mov_b32 s12, s22
	s_add_u32 s22, s22, s48
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[130:133], v[66:81]
	s_addc_u32 s23, s23, s49
	s_add_u32 s24, s24, s40
	s_addc_u32 s25, s25, s41
	s_cmp_lt_i32 s29, s20
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[134:137], v[82:97]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[142:145], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[130:133], v[82:97]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[138:141], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[114:117], v[142:145], v[82:97]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[158:161], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[138:141], v[82:97]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[154:157], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[158:161], v[82:97]
	v_lshl_add_u32 v123, v248, 1, s16
	v_lshlrev_b32_e32 v124, 1, v244
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[150:153], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[154:157], v[82:97]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[190:193], v[146:149], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[150:153], v[82:97]
	s_nop 7
	s_nop 2
	v_max_f32_e32 v106, v67, v67
	v_max_f32_e32 v107, v66, v66
	v_max_f32_e32 v106, v107, v106
	v_max3_f32 v102, v106, v68, v69
	v_max3_f32 v102, v102, v70, v71
	v_max3_f32 v102, v102, v72, v73
	v_max3_f32 v102, v102, v74, v75
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[146:149], v[82:97]
	v_max3_f32 v102, v102, v76, v77
	v_max3_f32 v102, v102, v78, v79
	v_max3_f32 v102, v102, v80, v81
	s_nop 7
	s_nop 0
	v_max3_f32 v98, v102, v82, v83
	v_max3_f32 v98, v98, v84, v85
	v_max3_f32 v98, v98, v86, v87
	v_max3_f32 v98, v98, v88, v89
	v_max3_f32 v98, v98, v90, v91
	v_max3_f32 v98, v98, v92, v93
	v_max3_f32 v98, v98, v94, v95
	v_max3_f32 v98, v98, v96, v97
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_max3_f32 v98, v198, v98, v99
	v_mul_f32_e32 v99, 0x3e0293ee, v98
	v_fma_f32 v66, v66, s27, -v99
	v_fma_f32 v67, v67, s27, -v99
	v_fma_f32 v68, v68, s27, -v99
	v_exp_f32_e32 v100, v66
	v_exp_f32_e32 v101, v67
	v_fma_f32 v69, v69, s27, -v99
	v_exp_f32_e32 v68, v68
	v_fma_f32 v70, v70, s27, -v99
	v_exp_f32_e32 v69, v69
	v_fma_f32 v71, v71, s27, -v99
	v_exp_f32_e32 v102, v70
	v_fma_f32 v72, v72, s27, -v99
	v_exp_f32_e32 v103, v71
	v_add_f32_e32 v66, v100, v101
	v_fma_f32 v73, v73, s27, -v99
	v_exp_f32_e32 v104, v72
	v_add_f32_e32 v66, v68, v66
	v_fma_f32 v74, v74, s27, -v99
	v_exp_f32_e32 v105, v73
	v_add_f32_e32 v66, v69, v66
	v_fma_f32 v75, v75, s27, -v99
	v_exp_f32_e32 v106, v74
	v_add_f32_e32 v66, v102, v66
	v_fma_f32 v76, v76, s27, -v99
	v_exp_f32_e32 v107, v75
	v_add_f32_e32 v66, v103, v66
	v_fma_f32 v77, v77, s27, -v99
	v_exp_f32_e32 v108, v76
	v_add_f32_e32 v66, v104, v66
	v_fma_f32 v78, v78, s27, -v99
	v_exp_f32_e32 v77, v77
	v_add_f32_e32 v66, v105, v66
	v_fma_f32 v79, v79, s27, -v99
	v_exp_f32_e32 v109, v78
	v_add_f32_e32 v66, v106, v66
	v_fma_f32 v80, v80, s27, -v99
	v_exp_f32_e32 v110, v79
	v_add_f32_e32 v66, v107, v66
	v_fma_f32 v81, v81, s27, -v99
	v_exp_f32_e32 v111, v80
	v_add_f32_e32 v66, v108, v66
	v_fma_f32 v82, v82, s27, -v99
	v_exp_f32_e32 v112, v81
	v_add_f32_e32 v66, v77, v66
	v_fma_f32 v83, v83, s27, -v99
	v_exp_f32_e32 v113, v82
	v_add_f32_e32 v66, v109, v66
	v_fma_f32 v84, v84, s27, -v99
	v_exp_f32_e32 v114, v83
	v_add_f32_e32 v66, v110, v66
	v_fma_f32 v85, v85, s27, -v99
	v_exp_f32_e32 v115, v84
	v_add_f32_e32 v66, v111, v66
	v_fma_f32 v86, v86, s27, -v99
	v_exp_f32_e32 v116, v85
	v_add_f32_e32 v66, v112, v66
	v_fma_f32 v87, v87, s27, -v99
	v_exp_f32_e32 v117, v86
	v_add_f32_e32 v66, v113, v66
	v_fma_f32 v88, v88, s27, -v99
	v_exp_f32_e32 v118, v87
	v_add_f32_e32 v66, v114, v66
	v_exp_f32_e32 v119, v88
	v_add_f32_e32 v66, v115, v66
	v_add_f32_e32 v66, v116, v66
	v_add_f32_e32 v66, v117, v66
	v_add_f32_e32 v66, v118, v66
	v_add_f32_e32 v122, v119, v66
	scratch_load_dword v66, off, off offset:12 ; 4-byte Folded Reload
	v_lshl_add_u32 v79, v253, 1, s16
	v_fma_f32 v67, v198, s27, -v99
	v_fma_f32 v89, v89, s27, -v99
	v_fma_f32 v90, v90, s27, -v99
	v_fma_f32 v91, v91, s27, -v99
	v_fma_f32 v92, v92, s27, -v99
	v_fma_f32 v93, v93, s27, -v99
	v_fma_f32 v94, v94, s27, -v99
	v_fma_f32 v95, v95, s27, -v99
	v_fma_f32 v96, v96, s27, -v99
	v_fma_f32 v97, v97, s27, -v99
	v_exp_f32_e32 v99, v67
	v_exp_f32_e32 v121, v90
	v_exp_f32_e32 v72, v91
	v_exp_f32_e32 v120, v89
	v_mul_f32_e32 v2, v2, v99
	v_mul_f32_e32 v3, v3, v99
	v_mul_f32_e32 v4, v4, v99
	v_mul_f32_e32 v5, v5, v99
	v_mul_f32_e32 v6, v6, v99
	v_mul_f32_e32 v7, v7, v99
	v_mul_f32_e32 v8, v8, v99
	v_mul_f32_e32 v9, v9, v99
	v_mul_f32_e32 v10, v10, v99
	v_mul_f32_e32 v11, v11, v99
	v_mul_f32_e32 v12, v12, v99
	v_mul_f32_e32 v13, v13, v99
	v_mul_f32_e32 v14, v14, v99
	v_mul_f32_e32 v15, v15, v99
	v_mul_f32_e32 v16, v16, v99
	v_mul_f32_e32 v17, v17, v99
	v_exp_f32_e32 v73, v92
	v_exp_f32_e32 v74, v93
	v_cvt_pk_f16_f32 v92, v117, v118
	v_cvt_pk_f16_f32 v93, v119, v120
	v_exp_f32_e32 v75, v94
	v_exp_f32_e32 v76, v95
	v_exp_f32_e32 v70, v96
	v_exp_f32_e32 v71, v97
	v_mul_f32_e32 v18, v18, v99
	v_mul_f32_e32 v19, v19, v99
	v_mul_f32_e32 v20, v20, v99
	v_mul_f32_e32 v21, v21, v99
	v_mul_f32_e32 v22, v22, v99
	v_mul_f32_e32 v23, v23, v99
	v_mul_f32_e32 v24, v24, v99
	v_mul_f32_e32 v25, v25, v99
	v_mul_f32_e32 v26, v26, v99
	v_mul_f32_e32 v27, v27, v99
	v_mul_f32_e32 v28, v28, v99
	v_mul_f32_e32 v29, v29, v99
	v_mul_f32_e32 v30, v30, v99
	v_mul_f32_e32 v31, v31, v99
	v_mul_f32_e32 v32, v32, v99
	v_mul_f32_e32 v33, v33, v99
	v_mul_f32_e32 v34, v34, v99
	v_mul_f32_e32 v35, v35, v99
	v_mul_f32_e32 v36, v36, v99
	v_mul_f32_e32 v37, v37, v99
	v_mul_f32_e32 v38, v38, v99
	v_mul_f32_e32 v39, v39, v99
	v_mul_f32_e32 v40, v40, v99
	v_mul_f32_e32 v41, v41, v99
	v_mul_f32_e32 v42, v42, v99
	v_mul_f32_e32 v43, v43, v99
	v_mul_f32_e32 v44, v44, v99
	v_mul_f32_e32 v45, v45, v99
	v_mul_f32_e32 v46, v46, v99
	v_mul_f32_e32 v47, v47, v99
	v_mul_f32_e32 v48, v48, v99
	v_mul_f32_e32 v49, v49, v99
	v_mul_f32_e32 v50, v50, v99
	v_mul_f32_e32 v51, v51, v99
	v_mul_f32_e32 v52, v52, v99
	v_mul_f32_e32 v53, v53, v99
	v_mul_f32_e32 v54, v54, v99
	v_mul_f32_e32 v55, v55, v99
	v_mul_f32_e32 v56, v56, v99
	v_mul_f32_e32 v57, v57, v99
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	v_add3_u32 v78, s16, v200, v66
	v_sub_u32_e32 v66, v78, v79
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshrrev_b32_e32 v67, 28, v67
	v_add_u32_e32 v66, v66, v67
	v_ashrrev_i32_e32 v66, 4, v66
	v_add_u32_e32 v66, v66, v249
	v_lshlrev_b32_e32 v67, 2, v66
	ds_bpermute_b32 v80, v67, v228
	v_cndmask_b32_e64 v67, 0, 1, s[4:5]
	v_cmp_ne_u32_e32 vcc, 0, v67
	v_readfirstlane_b32 s4, v79
	s_mov_b32 m0, s4
	v_lshrrev_b64 v[66:67], v66, vcc
	v_and_b32_e32 v66, 1, v66
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v67, 1, v80
	v_cmp_eq_u32_e32 vcc, 1, v66
	scratch_load_dword v79, off, off        ; 4-byte Folded Reload
	v_mul_f32_e32 v58, v58, v99
	v_cndmask_b32_e32 v66, v203, v67, vcc
	buffer_load_dwordx4 v66, s[12:15], 0 offen lds
	v_sub_u32_e32 v66, v78, v123
	scratch_load_dword v78, off, off offset:4 ; 4-byte Folded Reload
	v_lshl_add_u32 v67, v194, 1, s28
	v_add_u32_e32 v125, 0x2000, v66
	v_cvt_pk_f16_f32 v66, v100, v101
	v_ashrrev_i32_e32 v86, 31, v125
	v_lshrrev_b32_e32 v100, 28, v86
	v_mul_f32_e32 v59, v59, v99
	v_mul_f32_e32 v60, v60, v99
	v_mul_f32_e32 v61, v61, v99
	v_mul_f32_e32 v62, v62, v99
	v_mul_f32_e32 v63, v63, v99
	v_mul_f32_e32 v64, v64, v99
	v_mul_f32_e32 v65, v65, v99
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v90, 1, v79
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v78, 1, v78
	v_add3_u32 v67, v67, v78, v90
	v_add3_u32 v91, v67, v124, v201
	ds_read_b64_tr_b16 v[78:79], v91
	ds_read_b64_tr_b16 v[80:81], v91 offset:2048
	v_cvt_pk_f16_f32 v67, v68, v69
	v_cvt_pk_f16_f32 v68, v102, v103
	v_cvt_pk_f16_f32 v69, v104, v105
	ds_read_b64_tr_b16 v[82:83], v91 offset:4096
	ds_read_b64_tr_b16 v[84:85], v91 offset:6144
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[78:81], v[66:69], v[2:17]
	v_cvt_pk_f16_f32 v78, v106, v107
	v_cvt_pk_f16_f32 v79, v108, v77
	v_cvt_pk_f16_f32 v80, v109, v110
	v_cvt_pk_f16_f32 v81, v111, v112
	ds_read_b64_tr_b16 v[86:87], v91 offset:8192
	ds_read_b64_tr_b16 v[88:89], v91 offset:10240
	v_add3_u32 v77, s28, v202, v90
	v_cvt_pk_f16_f32 v90, v113, v114
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[78:81], v[2:17]
	ds_read_b64_tr_b16 v[82:83], v91 offset:12288
	ds_read_b64_tr_b16 v[84:85], v91 offset:14336
	v_cvt_pk_f16_f32 v91, v115, v116
	v_add3_u32 v77, v77, v124, v201
	ds_read_b64_tr_b16 v[94:95], v77
	ds_read_b64_tr_b16 v[96:97], v77 offset:2048
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[86:89], v[90:93], v[2:17]
	v_cvt_pk_f16_f32 v86, v121, v72
	v_cvt_pk_f16_f32 v87, v73, v74
	v_cvt_pk_f16_f32 v88, v75, v76
	v_cvt_pk_f16_f32 v89, v70, v71
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[86:89], v[2:17]
	v_add_u32_e32 v82, v125, v100
	v_ashrrev_i32_e32 v82, 4, v82
	v_add_u32_e32 v100, v82, v249
	v_lshlrev_b32_e32 v82, 2, v100
	ds_bpermute_b32 v101, v82, v197
	ds_read_b64_tr_b16 v[82:83], v77 offset:4096
	ds_read_b64_tr_b16 v[84:85], v77 offset:6144
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v101, 1, v101
	v_mfma_f32_32x32x16_f16 v[18:33], v[94:97], v[66:69], v[18:33]
	v_cndmask_b32_e64 v94, 0, 1, s[2:3]
	v_cmp_ne_u32_e32 vcc, 0, v94
	v_readfirstlane_b32 s2, v123
	s_mov_b32 m0, s2
	v_lshrrev_b64 v[94:95], v100, vcc
	v_and_b32_e32 v100, 1, v94
	v_cmp_eq_u32_e32 vcc, 1, v100
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[78:81], v[18:33]
	ds_read_b64_tr_b16 v[94:95], v77 offset:8192
	ds_read_b64_tr_b16 v[96:97], v77 offset:10240
	v_cndmask_b32_e32 v82, v203, v101, vcc
	buffer_load_dwordx4 v82, s[12:15], 0 offen lds
	ds_read_b64_tr_b16 v[82:83], v77 offset:12288
	ds_read_b64_tr_b16 v[84:85], v77 offset:14336
	v_lshl_add_u32 v77, v195, 1, s28
	v_add3_u32 v77, v77, v124, v201
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[94:97], v[90:93], v[18:33]
	ds_read_b64_tr_b16 v[94:95], v77
	ds_read_b64_tr_b16 v[96:97], v77 offset:2048
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[94:97], v[66:69], v[34:49]
	v_add_f32_e32 v94, v120, v122
	v_add_f32_e32 v100, v121, v94
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[86:89], v[18:33]
	ds_read_b64_tr_b16 v[82:83], v77 offset:4096
	ds_read_b64_tr_b16 v[84:85], v77 offset:6144
	ds_read_b64_tr_b16 v[94:95], v77 offset:8192
	ds_read_b64_tr_b16 v[96:97], v77 offset:10240
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[78:81], v[34:49]
	ds_read_b64_tr_b16 v[82:83], v77 offset:12288
	ds_read_b64_tr_b16 v[84:85], v77 offset:14336
	v_lshl_add_u32 v77, v199, 1, s28
	v_add3_u32 v77, v77, v124, v201
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[94:97], v[90:93], v[34:49]
	ds_read_b64_tr_b16 v[94:95], v77
	ds_read_b64_tr_b16 v[96:97], v77 offset:2048
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[94:97], v[66:69], v[50:65]
	v_add_f32_e32 v66, v72, v100
	v_add_f32_e32 v66, v73, v66
	v_add_f32_e32 v66, v74, v66
	v_add_f32_e32 v66, v75, v66
	v_add_f32_e32 v72, v76, v66
	v_add_f32_e32 v70, v70, v72
	v_add_f32_e32 v74, v71, v70
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[86:89], v[34:49]
	ds_read_b64_tr_b16 v[82:83], v77 offset:4096
	ds_read_b64_tr_b16 v[84:85], v77 offset:6144
	ds_read_b64_tr_b16 v[66:67], v77 offset:8192
	ds_read_b64_tr_b16 v[68:69], v77 offset:10240
	ds_read_b64_tr_b16 v[70:71], v77 offset:12288
	ds_read_b64_tr_b16 v[72:73], v77 offset:14336
	v_mov_b32_e32 v75, v74
	s_nop 1
	v_permlane32_swap_b32_e32 v74, v75
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[82:85], v[78:81], v[50:65]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[66:69], v[90:93], v[50:65]
	v_add_f32_e32 v66, v74, v75
	v_fmac_f32_e32 v66, v247, v99
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[86:89], v[50:65]
	s_cbranch_scc0 .LBB0_36
; %bb.34:                               ;   in Loop: Header=BB0_31 Depth=1
	scratch_load_dword v78, off, off offset:16 ; 4-byte Folded Reload
	v_mov_b32_e32 v235, v197
	s_mov_b32 s28, s61
	v_mov_b32_e32 v198, v98
	v_mov_b32_e32 v247, v66
	s_mov_b32 s30, s29
	s_branch .LBB0_31
.LBB0_35:
	scratch_load_dword v0, off, off offset:28 ; 4-byte Folded Reload
	s_add_i32 s18, 0, 0x4000
	s_add_i32 s17, 0, 0x8000
	s_branch .LBB0_37
.LBB0_36:                               ; %._crit_edge2045.loopexit
	scratch_load_dword v100, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v197, off, off offset:100 ; 4-byte Folded Reload
	v_mov_b32_e32 v198, v98
	v_mov_b32_e32 v247, v66
	v_mov_b32_e32 v66, v1
.LBB0_37:                               ; %._crit_edge2045
	v_lshlrev_b32_e32 v232, 1, v66
	v_add3_u32 v1, s19, v254, v232
	s_waitcnt vmcnt(1)
	v_or_b32_e32 v202, 1, v100
	v_or_b32_e32 v203, 2, v100
	v_or_b32_e32 v204, 3, v100
	v_or_b32_e32 v205, 8, v100
	v_or_b32_e32 v209, 9, v100
	v_or_b32_e32 v195, 10, v100
	v_or_b32_e32 v199, 11, v100
	v_or_b32_e32 v200, 16, v100
	v_or_b32_e32 v201, 17, v100
	v_or_b32_e32 v210, 18, v100
	v_or_b32_e32 v211, 19, v100
	v_or_b32_e32 v212, 24, v100
	v_or_b32_e32 v213, 25, v100
	v_or_b32_e32 v214, 26, v100
	v_or_b32_e32 v215, 27, v100
	v_or_b32_e32 v216, 32, v100
	v_or_b32_e32 v217, 33, v100
	v_or_b32_e32 v218, 34, v100
	v_or_b32_e32 v219, 35, v100
	v_or_b32_e32 v220, 40, v100
	v_or_b32_e32 v221, 41, v100
	v_or_b32_e32 v222, 42, v100
	v_or_b32_e32 v223, 43, v100
	v_or_b32_e32 v224, 48, v100
	v_or_b32_e32 v225, 49, v100
	v_or_b32_e32 v226, 50, v100
	v_or_b32_e32 v227, 51, v100
	v_or_b32_e32 v228, 56, v100
	v_or_b32_e32 v229, 57, v100
	v_or_b32_e32 v230, 58, v100
	v_or_b32_e32 v231, 59, v100
	s_waitcnt vmcnt(0)
	s_barrier
	v_add3_u32 v66, s19, v0, v232
	v_add3_u32 v67, s19, v250, v232
	v_add3_u32 v68, s19, v255, v232
	v_add3_u32 v69, s19, v240, v232
	v_add3_u32 v70, s19, v241, v232
	v_add3_u32 v71, s19, v242, v232
	v_add3_u32 v72, s19, v243, v232
	ds_read_b128 v[162:165], v1
	ds_read_b128 v[98:101], v1 offset:8192
	ds_read_b128 v[166:169], v66
	ds_read_b128 v[102:105], v66 offset:8192
	ds_read_b128 v[170:173], v67
	ds_read_b128 v[106:109], v67 offset:8192
	ds_read_b128 v[174:177], v68
	ds_read_b128 v[110:113], v68 offset:8192
	ds_read_b128 v[178:181], v69
	ds_read_b128 v[114:117], v69 offset:8192
	ds_read_b128 v[182:185], v70
	ds_read_b128 v[118:121], v70 offset:8192
	ds_read_b128 v[186:189], v71
	ds_read_b128 v[122:125], v71 offset:8192
	ds_read_b128 v[190:193], v72
	ds_read_b128 v[126:129], v72 offset:8192
	s_sub_i32 s14, s59, s42
	s_ashr_i32 s2, s14, 31
	s_or_b32 s6, s14, 63
	s_lshr_b32 s2, s2, 26
	s_add_i32 s2, s6, s2
	s_ashr_i32 s2, s2, 6
	s_max_i32 s2, s2, 3
	s_lshl_b32 s2, s2, 6
	s_add_i32 s7, s42, s2
	s_add_i32 s13, s7, 0xffffff40
	s_add_i32 s12, s7, 0xffffff80
	s_cmp_lg_u32 s12, s59
	s_cselect_b64 s[4:5], -1, 0
	s_xor_b64 s[2:3], s[36:37], -1
	s_or_b64 s[4:5], s[2:3], s[4:5]
	v_mov_b32_e32 v206, v254
	v_mov_b32_e32 v207, v250
	v_mov_b32_e32 v208, v255
	v_mov_b32_e32 v66, 0
	s_and_b64 vcc, exec, s[4:5]
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
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
	s_cbranch_vccnz .LBB0_39
; %bb.38:
	scratch_load_dword v1, off, off offset:20 ; 4-byte Folded Reload
	v_or_b32_e32 v67, s13, v202
	v_mov_b32_e32 v194, 0xff800000
	v_or_b32_e32 v68, s13, v203
	v_or_b32_e32 v69, s13, v204
	v_or_b32_e32 v70, s13, v205
	v_or_b32_e32 v71, s13, v209
	v_or_b32_e32 v72, s13, v195
	v_or_b32_e32 v73, s13, v199
	v_or_b32_e32 v74, s13, v200
	v_or_b32_e32 v75, s13, v201
	v_or_b32_e32 v76, s13, v210
	v_or_b32_e32 v77, s13, v211
	v_or_b32_e32 v78, s13, v212
	v_or_b32_e32 v79, s13, v213
	v_or_b32_e32 v80, s13, v214
	v_or_b32_e32 v81, s13, v215
	v_or_b32_e32 v82, s13, v216
	v_or_b32_e32 v83, s13, v217
	v_or_b32_e32 v84, s13, v218
	v_or_b32_e32 v85, s13, v219
	v_or_b32_e32 v86, s13, v220
	v_or_b32_e32 v87, s13, v221
	v_or_b32_e32 v88, s13, v222
	v_or_b32_e32 v89, s13, v223
	v_or_b32_e32 v90, s13, v224
	v_or_b32_e32 v91, s13, v225
	v_or_b32_e32 v92, s13, v226
	v_or_b32_e32 v93, s13, v227
	v_or_b32_e32 v94, s13, v228
	v_or_b32_e32 v95, s13, v229
	v_or_b32_e32 v96, s13, v230
	v_or_b32_e32 v97, s13, v231
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v1, s13, v1
	v_cmp_gt_i32_e32 vcc, s67, v1
	s_nop 1
	v_cndmask_b32_e64 v66, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v67
	s_nop 1
	v_cndmask_b32_e64 v67, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v68
	s_nop 1
	v_cndmask_b32_e64 v68, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v69
	s_nop 1
	v_cndmask_b32_e64 v69, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v70
	s_nop 1
	v_cndmask_b32_e64 v70, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v71
	s_nop 1
	v_cndmask_b32_e64 v71, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v72
	s_nop 1
	v_cndmask_b32_e64 v72, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v73
	s_nop 1
	v_cndmask_b32_e64 v73, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v74
	s_nop 1
	v_cndmask_b32_e64 v74, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v75
	s_nop 1
	v_cndmask_b32_e64 v75, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v76
	s_nop 1
	v_cndmask_b32_e64 v76, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v77
	s_nop 1
	v_cndmask_b32_e64 v77, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v78
	s_nop 1
	v_cndmask_b32_e64 v78, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v79
	s_nop 1
	v_cndmask_b32_e64 v79, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v80
	s_nop 1
	v_cndmask_b32_e64 v80, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v81
	s_nop 1
	v_cndmask_b32_e64 v81, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v82
	s_nop 1
	v_cndmask_b32_e64 v82, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v83
	s_nop 1
	v_cndmask_b32_e64 v83, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v84
	s_nop 1
	v_cndmask_b32_e64 v84, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v85
	s_nop 1
	v_cndmask_b32_e64 v85, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v86
	s_nop 1
	v_cndmask_b32_e64 v86, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v87
	s_nop 1
	v_cndmask_b32_e64 v87, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v88
	s_nop 1
	v_cndmask_b32_e64 v88, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v89
	s_nop 1
	v_cndmask_b32_e64 v89, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v90
	s_nop 1
	v_cndmask_b32_e64 v90, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v91
	s_nop 1
	v_cndmask_b32_e64 v91, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v92
	s_nop 1
	v_cndmask_b32_e64 v92, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v93
	s_nop 1
	v_cndmask_b32_e64 v93, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v94
	s_nop 1
	v_cndmask_b32_e64 v94, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v95
	s_nop 1
	v_cndmask_b32_e64 v95, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v96
	s_nop 1
	v_cndmask_b32_e64 v96, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v97
	s_nop 1
	v_cndmask_b32_e64 v97, v194, 0, vcc
.LBB0_39:
	scratch_load_dword v1, off, off offset:8 ; 4-byte Folded Reload
	v_add_u32_e32 v233, s53, v202
	scratch_store_dword off, v201, off offset:136 ; 4-byte Folded Spill
	v_add_u32_e32 v244, s53, v201
	scratch_store_dword off, v214, off offset:160 ; 4-byte Folded Spill
	v_add_u32_e32 v194, s53, v214
	v_add_u32_e32 v201, s53, v218
	v_add_u32_e32 v235, s53, v203
	scratch_store_dword off, v212, off offset:152 ; 4-byte Folded Spill
	v_add_u32_e32 v254, s53, v212
	scratch_store_dword off, v220, off offset:184 ; 4-byte Folded Spill
	v_add_u32_e32 v212, s53, v220
	scratch_store_dword off, v227, off offset:212 ; 4-byte Folded Spill
	v_add_u32_e32 v249, s53, v227
	v_add_u32_e32 v227, s13, v233
	scratch_store_dword off, v201, off offset:36 ; 4-byte Folded Spill
	v_add_u32_e32 v220, s13, v201
	v_mov_b32_e32 v201, 0xff800000
	v_add_u32_e32 v237, s53, v204
	scratch_store_dword off, v229, off offset:220 ; 4-byte Folded Spill
	v_add_u32_e32 v251, s53, v229
	v_add_u32_e32 v229, s13, v235
	v_add_u32_e32 v239, s53, v205
	scratch_store_dword off, v231, off offset:228 ; 4-byte Folded Spill
	v_add_u32_e32 v238, s53, v231
	v_add_u32_e32 v231, s13, v237
	v_add_u32_e32 v240, s53, v209
	v_mov_b32_e32 v236, v233
	v_add_u32_e32 v233, s13, v239
	v_add_u32_e32 v241, s53, v195
	scratch_store_dword off, v235, off offset:272 ; 4-byte Folded Spill
	v_add_u32_e32 v235, s13, v240
	v_add_u32_e32 v242, s53, v199
	scratch_store_dword off, v237, off offset:268 ; 4-byte Folded Spill
	v_add_u32_e32 v237, s13, v241
	v_add_u32_e32 v243, s53, v200
	scratch_store_dword off, v239, off offset:244 ; 4-byte Folded Spill
	v_add_u32_e32 v239, s13, v242
	scratch_store_dword off, v240, off offset:248 ; 4-byte Folded Spill
	v_add_u32_e32 v240, s13, v243
	v_add_u32_e32 v253, s53, v210
	scratch_store_dword off, v241, off offset:252 ; 4-byte Folded Spill
	v_add_u32_e32 v241, s13, v244
	v_add_u32_e32 v255, s53, v211
	scratch_store_dword off, v242, off offset:256 ; 4-byte Folded Spill
	v_add_u32_e32 v242, s13, v253
	v_add_u32_e32 v245, s53, v213
	scratch_store_dword off, v243, off offset:260 ; 4-byte Folded Spill
	v_add_u32_e32 v243, s13, v255
	scratch_store_dword off, v244, off offset:264 ; 4-byte Folded Spill
	scratch_store_dword off, v254, off offset:236 ; 4-byte Folded Spill
	v_add_u32_e32 v244, s13, v254
	v_mov_b32_e32 v254, v245
	scratch_store_dword off, v195, off offset:120 ; 4-byte Folded Spill
	scratch_store_dword off, v215, off offset:164 ; 4-byte Folded Spill
	v_add_u32_e32 v195, s53, v215
	v_add_u32_e32 v215, s13, v254
	scratch_store_dword off, v199, off offset:124 ; 4-byte Folded Spill
	scratch_store_dword off, v216, off offset:168 ; 4-byte Folded Spill
	v_add_u32_e32 v199, s53, v216
	v_add_u32_e32 v216, s13, v194
	scratch_store_dword off, v200, off offset:128 ; 4-byte Folded Spill
	scratch_store_dword off, v217, off offset:172 ; 4-byte Folded Spill
	v_add_u32_e32 v200, s53, v217
	v_add_u32_e32 v217, s13, v195
	scratch_store_dword off, v218, off offset:176 ; 4-byte Folded Spill
	v_add_u32_e32 v218, s13, v199
	scratch_store_dword off, v210, off offset:140 ; 4-byte Folded Spill
	scratch_store_dword off, v219, off offset:180 ; 4-byte Folded Spill
	v_add_u32_e32 v210, s53, v219
	v_add_u32_e32 v219, s13, v200
	scratch_store_dword off, v211, off offset:148 ; 4-byte Folded Spill
	scratch_store_dword off, v221, off offset:188 ; 4-byte Folded Spill
	v_add_u32_e32 v211, s53, v221
	scratch_store_dword off, v222, off offset:192 ; 4-byte Folded Spill
	v_add_u32_e32 v222, s53, v222
	scratch_store_dword off, v210, off offset:40 ; 4-byte Folded Spill
	v_add_u32_e32 v221, s13, v210
	v_mov_b32_e32 v210, v212
	v_mov_b32_e32 v212, v222
	s_waitcnt vmcnt(30)
	v_add_u32_e32 v214, s13, v1
	v_cmp_ge_i32_e32 vcc, v196, v214
	v_add_u32_e32 v222, s13, v210
	scratch_store_dword off, v213, off offset:156 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v66, v201, v66, vcc
	v_cmp_ge_i32_e32 vcc, v196, v227
	scratch_store_dword off, v223, off offset:196 ; 4-byte Folded Spill
	v_add_u32_e32 v213, s53, v223
	v_cndmask_b32_e32 v67, v201, v67, vcc
	v_cmp_ge_i32_e32 vcc, v196, v229
	v_add_u32_e32 v223, s13, v211
	scratch_store_dword off, v224, off offset:200 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v68, v201, v68, vcc
	v_cmp_ge_i32_e32 vcc, v196, v231
	v_add_u32_e32 v234, s53, v224
	v_add_u32_e32 v224, s13, v212
	v_cndmask_b32_e32 v69, v201, v69, vcc
	v_cmp_ge_i32_e32 vcc, v196, v233
	scratch_store_dword off, v225, off offset:204 ; 4-byte Folded Spill
	v_add_u32_e32 v246, s53, v225
	v_cndmask_b32_e32 v70, v201, v70, vcc
	v_cmp_ge_i32_e32 vcc, v196, v235
	scratch_store_dword off, v213, off offset:44 ; 4-byte Folded Spill
	v_add_u32_e32 v225, s13, v213
	v_cndmask_b32_e32 v71, v201, v71, vcc
	v_cmp_ge_i32_e32 vcc, v196, v237
	v_mov_b32_e32 v213, v234
	scratch_store_dword off, v226, off offset:208 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v72, v201, v72, vcc
	v_cmp_ge_i32_e32 vcc, v196, v239
	v_add_u32_e32 v248, s53, v226
	v_add_u32_e32 v226, s13, v213
	v_cndmask_b32_e32 v73, v201, v73, vcc
	v_cmp_ge_i32_e32 vcc, v196, v240
	scratch_store_dword off, v228, off offset:216 ; 4-byte Folded Spill
	v_add_u32_e32 v250, s53, v228
	v_cndmask_b32_e32 v74, v201, v74, vcc
	v_cmp_ge_i32_e32 vcc, v196, v241
	v_add_u32_e32 v228, s13, v246
	scratch_store_dword off, v230, off offset:224 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v75, v201, v75, vcc
	v_cmp_ge_i32_e32 vcc, v196, v242
	v_add_u32_e32 v252, s53, v230
	v_add_u32_e32 v230, s13, v248
	v_cndmask_b32_e32 v76, v201, v76, vcc
	v_cmp_ge_i32_e32 vcc, v196, v243
	v_add_u32_e32 v1, s13, v249
	scratch_store_dword off, v194, off offset:12 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v77, v201, v77, vcc
	v_cmp_ge_i32_e32 vcc, v196, v244
	v_add_u32_e32 v194, s13, v250
	scratch_store_dword off, v195, off offset:16 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v78, v201, v78, vcc
	v_cmp_ge_i32_e32 vcc, v196, v215
	v_add_u32_e32 v195, s13, v251
	scratch_store_dword off, v199, off offset:28 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v79, v201, v79, vcc
	v_cmp_ge_i32_e32 vcc, v196, v216
	v_add_u32_e32 v199, s13, v252
	scratch_store_dword off, v200, off offset:32 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v80, v201, v80, vcc
	v_cmp_ge_i32_e32 vcc, v196, v217
	v_add_u32_e32 v200, s13, v238
	s_cmp_gt_i32 s14, 0
	v_cndmask_b32_e32 v81, v201, v81, vcc
	v_cmp_ge_i32_e32 vcc, v196, v218
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s14, 1
	v_cndmask_b32_e32 v82, v201, v82, vcc
	v_cmp_ge_i32_e32 vcc, v196, v219
	scratch_store_dword off, v253, off offset:240 ; 4-byte Folded Spill
	scratch_store_dword off, v255, off offset:232 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v83, v201, v83, vcc
	v_cmp_ge_i32_e32 vcc, v196, v220
	scratch_store_dword off, v246, off offset:48 ; 4-byte Folded Spill
	scratch_store_dword off, v248, off offset:52 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v84, v201, v84, vcc
	v_cmp_ge_i32_e32 vcc, v196, v221
	scratch_store_dword off, v249, off offset:56 ; 4-byte Folded Spill
	scratch_store_dword off, v250, off offset:60 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v85, v201, v85, vcc
	v_cmp_ge_i32_e32 vcc, v196, v222
	scratch_store_dword off, v251, off offset:64 ; 4-byte Folded Spill
	scratch_store_dword off, v252, off offset:92 ; 4-byte Folded Spill
	v_cndmask_b32_e32 v86, v201, v86, vcc
	v_cmp_ge_i32_e32 vcc, v196, v223
	scratch_store_dword off, v238, off offset:96 ; 4-byte Folded Spill
	s_nop 0
	v_cndmask_b32_e32 v87, v201, v87, vcc
	v_cmp_ge_i32_e32 vcc, v196, v224
	s_nop 1
	v_cndmask_b32_e32 v88, v201, v88, vcc
	v_cmp_ge_i32_e32 vcc, v196, v225
	s_nop 1
	v_cndmask_b32_e32 v89, v201, v89, vcc
	v_cmp_ge_i32_e32 vcc, v196, v226
	s_nop 1
	v_cndmask_b32_e32 v90, v201, v90, vcc
	v_cmp_ge_i32_e32 vcc, v196, v228
	s_nop 1
	v_cndmask_b32_e32 v91, v201, v91, vcc
	v_cmp_ge_i32_e32 vcc, v196, v230
	s_nop 1
	v_cndmask_b32_e32 v92, v201, v92, vcc
	v_cmp_ge_i32_e32 vcc, v196, v1
	s_nop 1
	v_cndmask_b32_e32 v93, v201, v93, vcc
	v_cmp_ge_i32_e32 vcc, v196, v194
	s_nop 1
	v_cndmask_b32_e32 v94, v201, v94, vcc
	v_cmp_ge_i32_e32 vcc, v196, v195
	s_nop 1
	v_cndmask_b32_e32 v95, v201, v95, vcc
	v_cmp_ge_i32_e32 vcc, v196, v199
	s_nop 1
	v_cndmask_b32_e32 v96, v201, v96, vcc
	v_cmp_ge_i32_e32 vcc, v196, v200
	s_nop 1
	v_cndmask_b32_e32 v97, v201, v97, vcc
	s_cbranch_scc1 .LBB0_41
; %bb.40:
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[134:137], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[134:137], v[82:97]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[130:133], v[66:81]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[130:133], v[82:97]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[142:145], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[142:145], v[82:97]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[138:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[138:141], v[82:97]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[158:161], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[82:97], v[114:117], v[158:161], v[82:97]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[154:157], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[154:157], v[82:97]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[150:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[150:153], v[82:97]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[190:193], v[146:149], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[146:149], v[82:97]
.LBB0_41:
	s_waitcnt lgkmcnt(12)
	scratch_load_dword v103, off, off       ; 4-byte Folded Reload
	scratch_load_dword v104, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v102, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v100, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v101, off, off offset:84 ; 4-byte Folded Reload
	v_and_b32_e32 v1, 12, v197
	s_movk_i32 s13, 0x60
	s_andn2_b64 vcc, exec, s[4:5]
	v_mov_b32_e32 v234, v236
	s_waitcnt vmcnt(3)
	v_or_b32_e32 v98, v1, v104
	s_waitcnt vmcnt(2)
	v_or_b32_e32 v99, v103, v102
	s_waitcnt vmcnt(0)
	v_and_or_b32 v100, v101, 3, v100
	v_lshlrev_b32_e32 v100, 7, v100
	v_or3_b32 v214, v99, v98, v100
	v_lshl_add_u32 v99, v214, 1, s61
	s_waitcnt lgkmcnt(1)
	ds_read_b64_tr_b16 v[190:191], v99
	ds_read_b64_tr_b16 v[192:193], v99 offset:2048
	ds_read_b64_tr_b16 v[186:187], v99 offset:4096
	ds_read_b64_tr_b16 v[188:189], v99 offset:6144
	ds_read_b64_tr_b16 v[182:183], v99 offset:8192
	ds_read_b64_tr_b16 v[184:185], v99 offset:10240
	ds_read_b64_tr_b16 v[178:179], v99 offset:12288
	ds_read_b64_tr_b16 v[180:181], v99 offset:14336
	v_or_b32_e32 v99, 32, v1
	v_bitop3_b32 v99, v102, v99, v104 bitop3:0xf6
	v_bitop3_b32 v98, v98, v103, 64 bitop3:0x36
	v_or3_b32 v237, v99, v103, v100
	v_or3_b32 v235, v98, v102, v100
	v_lshl_add_u32 v99, v237, 1, s61
	v_lshl_add_u32 v98, v235, 1, s61
	ds_read_b64_tr_b16 v[174:175], v99
	ds_read_b64_tr_b16 v[176:177], v99 offset:2048
	ds_read_b64_tr_b16 v[170:171], v99 offset:4096
	ds_read_b64_tr_b16 v[172:173], v99 offset:6144
	ds_read_b64_tr_b16 v[166:167], v99 offset:8192
	ds_read_b64_tr_b16 v[168:169], v99 offset:10240
	ds_read_b64_tr_b16 v[162:163], v99 offset:12288
	ds_read_b64_tr_b16 v[164:165], v99 offset:14336
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[126:127], v98
	ds_read_b64_tr_b16 v[128:129], v98 offset:2048
	ds_read_b64_tr_b16 v[122:123], v98 offset:4096
	ds_read_b64_tr_b16 v[124:125], v98 offset:6144
	ds_read_b64_tr_b16 v[118:119], v98 offset:8192
	ds_read_b64_tr_b16 v[120:121], v98 offset:10240
	ds_read_b64_tr_b16 v[114:115], v98 offset:12288
	ds_read_b64_tr_b16 v[116:117], v98 offset:14336
	scratch_load_dword v98, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_bitop3_b32 v1, v98, v1, s13 bitop3:0x4e
	v_or3_b32 v233, v1, v102, v100
	v_lshl_add_u32 v1, v233, 1, s61
	ds_read_b64_tr_b16 v[110:111], v1
	ds_read_b64_tr_b16 v[112:113], v1 offset:2048
	ds_read_b64_tr_b16 v[106:107], v1 offset:4096
	ds_read_b64_tr_b16 v[108:109], v1 offset:6144
	ds_read_b64_tr_b16 v[102:103], v1 offset:8192
	ds_read_b64_tr_b16 v[104:105], v1 offset:10240
	ds_read_b64_tr_b16 v[98:99], v1 offset:12288
	ds_read_b64_tr_b16 v[100:101], v1 offset:14336
	s_cbranch_vccnz .LBB0_43
; %bb.42:
	v_max_f32_e32 v1, v67, v67
	v_max_f32_e32 v194, v66, v66
	v_max_f32_e32 v1, v194, v1
	v_max3_f32 v1, v1, v68, v69
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_max3_f32 v1, v1, v74, v75
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_mov_b32_e32 v194, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v194
	v_max3_f32 v227, v198, v1, v194
	s_mov_b32 s4, 0x3e0293ee
	v_mul_f32_e32 v1, 0x3e0293ee, v227
	v_fma_f32 v66, v66, s4, -v1
	v_fma_f32 v67, v67, s4, -v1
	v_fma_f32 v68, v68, s4, -v1
	v_fma_f32 v69, v69, s4, -v1
	v_fma_f32 v70, v70, s4, -v1
	v_fma_f32 v71, v71, s4, -v1
	v_fma_f32 v72, v72, s4, -v1
	v_fma_f32 v73, v73, s4, -v1
	v_fma_f32 v74, v74, s4, -v1
	v_fma_f32 v75, v75, s4, -v1
	v_fma_f32 v76, v76, s4, -v1
	v_fma_f32 v77, v77, s4, -v1
	v_fma_f32 v78, v78, s4, -v1
	v_fma_f32 v79, v79, s4, -v1
	v_fma_f32 v80, v80, s4, -v1
	v_fma_f32 v81, v81, s4, -v1
	v_fma_f32 v82, v82, s4, -v1
	v_fma_f32 v83, v83, s4, -v1
	v_fma_f32 v84, v84, s4, -v1
	v_fma_f32 v85, v85, s4, -v1
	v_fma_f32 v86, v86, s4, -v1
	v_fma_f32 v87, v87, s4, -v1
	v_fma_f32 v88, v88, s4, -v1
	v_fma_f32 v89, v89, s4, -v1
	v_fma_f32 v90, v90, s4, -v1
	v_fma_f32 v91, v91, s4, -v1
	v_fma_f32 v92, v92, s4, -v1
	v_fma_f32 v93, v93, s4, -v1
	v_fma_f32 v94, v94, s4, -v1
	v_fma_f32 v95, v95, s4, -v1
	v_fma_f32 v96, v96, s4, -v1
	v_fma_f32 v97, v97, s4, -v1
	v_fma_f32 v1, v198, s4, -v1
	v_exp_f32_e32 v194, v66
	v_exp_f32_e32 v195, v67
	v_exp_f32_e32 v199, v68
	v_exp_f32_e32 v200, v69
	v_exp_f32_e32 v201, v70
	v_exp_f32_e32 v215, v71
	v_exp_f32_e32 v216, v72
	v_exp_f32_e32 v217, v73
	v_exp_f32_e32 v1, v1
	v_cvt_pk_f16_f32 v68, v201, v215
	v_cvt_pk_f16_f32 v67, v199, v200
	v_cvt_pk_f16_f32 v69, v216, v217
	v_cvt_pk_f16_f32 v66, v194, v195
	v_mul_f32_e32 v32, v32, v1
	v_mul_f32_e32 v33, v33, v1
	v_mul_f32_e32 v30, v30, v1
	v_mul_f32_e32 v31, v31, v1
	v_mul_f32_e32 v28, v28, v1
	v_mul_f32_e32 v29, v29, v1
	v_mul_f32_e32 v26, v26, v1
	v_mul_f32_e32 v27, v27, v1
	v_mul_f32_e32 v24, v24, v1
	v_mul_f32_e32 v25, v25, v1
	v_mul_f32_e32 v22, v22, v1
	v_mul_f32_e32 v23, v23, v1
	v_mul_f32_e32 v20, v20, v1
	v_mul_f32_e32 v21, v21, v1
	v_mul_f32_e32 v18, v18, v1
	v_mul_f32_e32 v19, v19, v1
	v_exp_f32_e32 v218, v74
	v_exp_f32_e32 v219, v75
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[66:69], v[18:33]
	v_exp_f32_e32 v220, v76
	v_exp_f32_e32 v221, v77
	v_exp_f32_e32 v222, v78
	v_exp_f32_e32 v223, v79
	v_exp_f32_e32 v224, v80
	v_exp_f32_e32 v225, v81
	v_cvt_pk_f16_f32 v71, v220, v221
	v_cvt_pk_f16_f32 v72, v222, v223
	v_cvt_pk_f16_f32 v70, v218, v219
	v_cvt_pk_f16_f32 v73, v224, v225
	v_mul_f32_e32 v48, v48, v1
	v_mul_f32_e32 v49, v49, v1
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[70:73], v[18:33]
	v_add_f32_e32 v170, v194, v195
	v_add_f32_e32 v170, v199, v170
	v_mul_f32_e32 v46, v46, v1
	v_mul_f32_e32 v47, v47, v1
	v_mul_f32_e32 v44, v44, v1
	v_mul_f32_e32 v45, v45, v1
	v_mul_f32_e32 v42, v42, v1
	v_mul_f32_e32 v43, v43, v1
	v_mul_f32_e32 v40, v40, v1
	v_mul_f32_e32 v41, v41, v1
	v_mul_f32_e32 v38, v38, v1
	v_mul_f32_e32 v39, v39, v1
	v_mul_f32_e32 v36, v36, v1
	v_mul_f32_e32 v37, v37, v1
	v_mul_f32_e32 v34, v34, v1
	v_mul_f32_e32 v35, v35, v1
	v_add_f32_e32 v170, v200, v170
	v_mul_f32_e32 v16, v16, v1
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[66:69], v[34:49]
	v_mul_f32_e32 v17, v17, v1
	v_mul_f32_e32 v14, v14, v1
	v_mul_f32_e32 v15, v15, v1
	v_mul_f32_e32 v12, v12, v1
	v_mul_f32_e32 v13, v13, v1
	v_mul_f32_e32 v10, v10, v1
	v_mul_f32_e32 v11, v11, v1
	v_mul_f32_e32 v8, v8, v1
	v_mul_f32_e32 v9, v9, v1
	v_mul_f32_e32 v6, v6, v1
	v_mul_f32_e32 v7, v7, v1
	v_mul_f32_e32 v4, v4, v1
	v_mul_f32_e32 v5, v5, v1
	v_mul_f32_e32 v2, v2, v1
	v_mul_f32_e32 v3, v3, v1
	v_add_f32_e32 v126, v201, v170
	v_mul_f32_e32 v64, v64, v1
	v_mul_f32_e32 v65, v65, v1
	v_mul_f32_e32 v62, v62, v1
	v_mul_f32_e32 v63, v63, v1
	v_mul_f32_e32 v60, v60, v1
	v_mul_f32_e32 v61, v61, v1
	v_mul_f32_e32 v58, v58, v1
	v_mul_f32_e32 v59, v59, v1
	v_mul_f32_e32 v56, v56, v1
	v_mul_f32_e32 v57, v57, v1
	v_mul_f32_e32 v54, v54, v1
	v_mul_f32_e32 v55, v55, v1
	v_mul_f32_e32 v52, v52, v1
	v_mul_f32_e32 v53, v53, v1
	v_mul_f32_e32 v50, v50, v1
	v_mul_f32_e32 v51, v51, v1
	v_mfma_f32_32x32x16_f16 v[2:17], v[190:193], v[66:69], v[2:17]
	v_add_f32_e32 v126, v215, v126
	v_add_f32_e32 v126, v216, v126
	v_add_f32_e32 v126, v217, v126
	v_add_f32_e32 v126, v218, v126
	v_add_f32_e32 v126, v219, v126
	v_add_f32_e32 v126, v220, v126
	v_exp_f32_e32 v82, v82
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[50:65], v[110:113], v[66:69], v[50:65]
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v85, v85
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[70:73], v[34:49]
	v_add_f32_e32 v122, v221, v126
	v_add_f32_e32 v122, v222, v122
	v_add_f32_e32 v122, v223, v122
	v_add_f32_e32 v122, v224, v122
	v_add_f32_e32 v122, v225, v122
	v_add_f32_e32 v66, v82, v122
	v_add_f32_e32 v66, v83, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[186:189], v[70:73], v[2:17]
	v_add_f32_e32 v66, v84, v66
	v_add_f32_e32 v66, v85, v66
	v_exp_f32_e32 v90, v90
	v_cvt_pk_f16_f32 v77, v88, v89
	v_cvt_pk_f16_f32 v76, v86, v87
	v_cvt_pk_f16_f32 v75, v84, v85
	v_cvt_pk_f16_f32 v74, v82, v83
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[106:109], v[70:73], v[50:65]
	v_add_f32_e32 v66, v86, v66
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v66, v87, v66
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v66, v88, v66
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v66, v89, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[182:185], v[74:77], v[2:17]
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_add_f32_e32 v66, v90, v66
	v_add_f32_e32 v66, v91, v66
	v_add_f32_e32 v66, v92, v66
	v_mfma_f32_32x32x16_f16 v[18:33], v[166:169], v[74:77], v[18:33]
	v_add_f32_e32 v66, v93, v66
	v_cvt_pk_f16_f32 v81, v96, v97
	v_cvt_pk_f16_f32 v80, v94, v95
	v_cvt_pk_f16_f32 v79, v92, v93
	v_cvt_pk_f16_f32 v78, v90, v91
	v_add_f32_e32 v66, v94, v66
	v_add_f32_e32 v66, v95, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[118:121], v[74:77], v[34:49]
	v_add_f32_e32 v66, v96, v66
	v_add_f32_e32 v66, v97, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v247, v1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[102:105], v[74:77], v[50:65]
	v_mov_b32_e32 v198, v227
	v_mov_b32_e32 v247, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[178:181], v[78:81], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[162:165], v[78:81], v[18:33]
	v_mfma_f32_32x32x16_f16 v[34:49], v[114:117], v[78:81], v[34:49]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[98:101], v[78:81], v[50:65]
.LBB0_43:
	scratch_load_dword v236, off, off offset:272 ; 4-byte Folded Reload
	scratch_load_dword v72, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v69, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v70, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v71, off, off offset:72 ; 4-byte Folded Reload
	v_add3_u32 v1, s18, v206, v232
	v_add3_u32 v66, s18, v0, v232
	v_add3_u32 v67, s18, v207, v232
	v_add3_u32 v68, s18, v208, v232
	s_sub_i32 s13, s7, 64
	s_cmp_lg_u32 s13, s59
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[4:5], s[2:3], s[4:5]
	s_and_b64 vcc, exec, s[4:5]
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
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
	s_waitcnt vmcnt(3)
	v_add3_u32 v72, s18, v72, v232
	s_waitcnt vmcnt(2)
	v_add3_u32 v69, s18, v69, v232
	s_waitcnt vmcnt(1)
	v_add3_u32 v70, s18, v70, v232
	s_waitcnt vmcnt(0)
	v_add3_u32 v71, s18, v71, v232
	s_waitcnt lgkmcnt(14)
	ds_read_b128 v[162:165], v1
	s_waitcnt lgkmcnt(1)
	ds_read_b128 v[98:101], v1 offset:8192
	ds_read_b128 v[166:169], v66
	ds_read_b128 v[102:105], v66 offset:8192
	ds_read_b128 v[170:173], v67
	ds_read_b128 v[106:109], v67 offset:8192
	ds_read_b128 v[174:177], v68
	ds_read_b128 v[110:113], v68 offset:8192
	ds_read_b128 v[178:181], v69
	ds_read_b128 v[114:117], v69 offset:8192
	ds_read_b128 v[182:185], v70
	ds_read_b128 v[118:121], v70 offset:8192
	ds_read_b128 v[186:189], v71
	ds_read_b128 v[122:125], v71 offset:8192
	ds_read_b128 v[190:193], v72
	ds_read_b128 v[126:129], v72 offset:8192
	scratch_load_dword v238, off, off offset:268 ; 4-byte Folded Reload
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	s_cbranch_vccnz .LBB0_45
; %bb.44:
	scratch_load_dword v1, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v66, off, off offset:120 ; 4-byte Folded Reload
	v_or_b32_e32 v67, s12, v202
	v_mov_b32_e32 v194, 0xff800000
	v_or_b32_e32 v68, s12, v203
	v_or_b32_e32 v69, s12, v204
	v_or_b32_e32 v70, s12, v205
	v_or_b32_e32 v71, s12, v209
	s_waitcnt vmcnt(1)
	v_or_b32_e32 v1, s12, v1
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v72, s12, v66
	scratch_load_dword v66, off, off offset:124 ; 4-byte Folded Reload
	v_cmp_gt_i32_e32 vcc, s67, v1
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v73, s12, v66
	scratch_load_dword v66, off, off offset:128 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v74, s12, v66
	scratch_load_dword v66, off, off offset:136 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v75, s12, v66
	scratch_load_dword v66, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v76, s12, v66
	scratch_load_dword v66, off, off offset:148 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v77, s12, v66
	scratch_load_dword v66, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v78, s12, v66
	scratch_load_dword v66, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v79, s12, v66
	scratch_load_dword v66, off, off offset:160 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v80, s12, v66
	scratch_load_dword v66, off, off offset:164 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v81, s12, v66
	scratch_load_dword v66, off, off offset:168 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v82, s12, v66
	scratch_load_dword v66, off, off offset:172 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v83, s12, v66
	scratch_load_dword v66, off, off offset:176 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v84, s12, v66
	scratch_load_dword v66, off, off offset:180 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v85, s12, v66
	scratch_load_dword v66, off, off offset:184 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v86, s12, v66
	scratch_load_dword v66, off, off offset:188 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v87, s12, v66
	scratch_load_dword v66, off, off offset:192 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v88, s12, v66
	scratch_load_dword v66, off, off offset:196 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v89, s12, v66
	scratch_load_dword v66, off, off offset:200 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v90, s12, v66
	scratch_load_dword v66, off, off offset:204 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v91, s12, v66
	scratch_load_dword v66, off, off offset:208 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v92, s12, v66
	scratch_load_dword v66, off, off offset:212 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v93, s12, v66
	scratch_load_dword v66, off, off offset:216 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v94, s12, v66
	scratch_load_dword v66, off, off offset:220 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v95, s12, v66
	scratch_load_dword v66, off, off offset:224 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v96, s12, v66
	scratch_load_dword v66, off, off offset:228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v97, s12, v66
	v_cndmask_b32_e64 v66, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v67
	s_nop 1
	v_cndmask_b32_e64 v67, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v68
	s_nop 1
	v_cndmask_b32_e64 v68, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v69
	s_nop 1
	v_cndmask_b32_e64 v69, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v70
	s_nop 1
	v_cndmask_b32_e64 v70, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v71
	s_nop 1
	v_cndmask_b32_e64 v71, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v72
	s_nop 1
	v_cndmask_b32_e64 v72, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v73
	s_nop 1
	v_cndmask_b32_e64 v73, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v74
	s_nop 1
	v_cndmask_b32_e64 v74, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v75
	s_nop 1
	v_cndmask_b32_e64 v75, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v76
	s_nop 1
	v_cndmask_b32_e64 v76, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v77
	s_nop 1
	v_cndmask_b32_e64 v77, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v78
	s_nop 1
	v_cndmask_b32_e64 v78, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v79
	s_nop 1
	v_cndmask_b32_e64 v79, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v80
	s_nop 1
	v_cndmask_b32_e64 v80, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v81
	s_nop 1
	v_cndmask_b32_e64 v81, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v82
	s_nop 1
	v_cndmask_b32_e64 v82, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v83
	s_nop 1
	v_cndmask_b32_e64 v83, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v84
	s_nop 1
	v_cndmask_b32_e64 v84, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v85
	s_nop 1
	v_cndmask_b32_e64 v85, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v86
	s_nop 1
	v_cndmask_b32_e64 v86, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v87
	s_nop 1
	v_cndmask_b32_e64 v87, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v88
	s_nop 1
	v_cndmask_b32_e64 v88, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v89
	s_nop 1
	v_cndmask_b32_e64 v89, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v90
	s_nop 1
	v_cndmask_b32_e64 v90, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v91
	s_nop 1
	v_cndmask_b32_e64 v91, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v92
	s_nop 1
	v_cndmask_b32_e64 v92, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v93
	s_nop 1
	v_cndmask_b32_e64 v93, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v94
	s_nop 1
	v_cndmask_b32_e64 v94, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v95
	s_nop 1
	v_cndmask_b32_e64 v95, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v96
	s_nop 1
	v_cndmask_b32_e64 v96, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v97
	s_nop 1
	v_cndmask_b32_e64 v97, v194, 0, vcc
.LBB0_45:
	scratch_store_dword off, v209, off offset:80 ; 4-byte Folded Spill
	scratch_store_dword off, v205, off offset:4 ; 4-byte Folded Spill
	scratch_store_dword off, v204, off      ; 4-byte Folded Spill
	v_mov_b32_e32 v197, v202
	scratch_load_dword v1, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v202, off, off offset:12 ; 4-byte Folded Reload
	v_mov_b32_e32 v209, v203
	scratch_load_dword v203, off, off offset:92 ; 4-byte Folded Reload
	scratch_load_dword v255, off, off offset:232 ; 4-byte Folded Reload
	scratch_load_dword v245, off, off offset:236 ; 4-byte Folded Reload
	scratch_load_dword v253, off, off offset:240 ; 4-byte Folded Reload
	scratch_load_dword v246, off, off offset:244 ; 4-byte Folded Reload
	scratch_load_dword v248, off, off offset:248 ; 4-byte Folded Reload
	scratch_load_dword v249, off, off offset:252 ; 4-byte Folded Reload
	scratch_load_dword v250, off, off offset:256 ; 4-byte Folded Reload
	scratch_load_dword v251, off, off offset:260 ; 4-byte Folded Reload
	scratch_load_dword v252, off, off offset:264 ; 4-byte Folded Reload
	v_add_u32_e32 v194, s12, v234
	v_mov_b32_e32 v205, 0xff800000
	v_add_u32_e32 v195, s12, v236
	s_waitcnt vmcnt(15)
	v_add_u32_e32 v199, s12, v238
	v_add_u32_e32 v222, s12, v254
	v_add_u32_e32 v229, s12, v210
	v_add_u32_e32 v230, s12, v211
	v_add_u32_e32 v231, s12, v212
	v_add_u32_e32 v240, s12, v213
	s_cmpk_gt_i32 s6, 0x7f
	s_cselect_b64 s[4:5], -1, 0
	s_cmpk_lt_i32 s6, 0x80
	scratch_load_dword v204, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(12)
	v_add_u32_e32 v1, s12, v1
	s_waitcnt vmcnt(11)
	v_add_u32_e32 v223, s12, v202
	scratch_load_dword v202, off, off offset:16 ; 4-byte Folded Reload
	v_cmp_ge_i32_e32 vcc, v196, v1
	s_waitcnt vmcnt(11)
	v_add_u32_e32 v203, s12, v203
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v204, s12, v204
	v_cndmask_b32_e32 v66, v205, v66, vcc
	v_cmp_ge_i32_e32 vcc, v196, v194
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v224, s12, v202
	scratch_load_dword v202, off, off offset:28 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v67, v205, v67, vcc
	v_cmp_ge_i32_e32 vcc, v196, v195
	v_add_u32_e32 v220, s12, v255
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v225, s12, v202
	scratch_load_dword v202, off, off offset:32 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v68, v205, v68, vcc
	v_cmp_ge_i32_e32 vcc, v196, v199
	v_add_u32_e32 v221, s12, v245
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v226, s12, v202
	scratch_load_dword v202, off, off offset:36 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v69, v205, v69, vcc
	v_add_u32_e32 v219, s12, v253
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v227, s12, v202
	scratch_load_dword v202, off, off offset:40 ; 4-byte Folded Reload
	v_add_u32_e32 v200, s12, v246
	v_cmp_ge_i32_e32 vcc, v196, v200
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v228, s12, v202
	scratch_load_dword v202, off, off offset:44 ; 4-byte Folded Reload
	v_add_u32_e32 v201, s12, v248
	v_cndmask_b32_e32 v70, v205, v70, vcc
	v_cmp_ge_i32_e32 vcc, v196, v201
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v239, s12, v202
	scratch_load_dword v202, off, off offset:48 ; 4-byte Folded Reload
	v_add_u32_e32 v215, s12, v249
	v_cndmask_b32_e32 v71, v205, v71, vcc
	v_cmp_ge_i32_e32 vcc, v196, v215
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v241, s12, v202
	scratch_load_dword v202, off, off offset:52 ; 4-byte Folded Reload
	v_add_u32_e32 v216, s12, v250
	v_cndmask_b32_e32 v72, v205, v72, vcc
	v_cmp_ge_i32_e32 vcc, v196, v216
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v242, s12, v202
	scratch_load_dword v202, off, off offset:56 ; 4-byte Folded Reload
	v_add_u32_e32 v217, s12, v251
	v_cndmask_b32_e32 v73, v205, v73, vcc
	v_cmp_ge_i32_e32 vcc, v196, v217
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v243, s12, v202
	scratch_load_dword v202, off, off offset:60 ; 4-byte Folded Reload
	v_add_u32_e32 v218, s12, v252
	v_cndmask_b32_e32 v74, v205, v74, vcc
	v_cmp_ge_i32_e32 vcc, v196, v218
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v244, s12, v202
	scratch_load_dword v202, off, off offset:64 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v75, v205, v75, vcc
	v_cmp_ge_i32_e32 vcc, v196, v219
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v202, s12, v202
	v_cndmask_b32_e32 v76, v205, v76, vcc
	v_cmp_ge_i32_e32 vcc, v196, v220
	s_nop 1
	v_cndmask_b32_e32 v77, v205, v77, vcc
	v_cmp_ge_i32_e32 vcc, v196, v221
	s_nop 1
	v_cndmask_b32_e32 v78, v205, v78, vcc
	v_cmp_ge_i32_e32 vcc, v196, v222
	s_nop 1
	v_cndmask_b32_e32 v79, v205, v79, vcc
	v_cmp_ge_i32_e32 vcc, v196, v223
	s_nop 1
	v_cndmask_b32_e32 v80, v205, v80, vcc
	v_cmp_ge_i32_e32 vcc, v196, v224
	s_nop 1
	v_cndmask_b32_e32 v81, v205, v81, vcc
	v_cmp_ge_i32_e32 vcc, v196, v225
	s_nop 1
	v_cndmask_b32_e32 v82, v205, v82, vcc
	v_cmp_ge_i32_e32 vcc, v196, v226
	s_nop 1
	v_cndmask_b32_e32 v83, v205, v83, vcc
	v_cmp_ge_i32_e32 vcc, v196, v227
	s_nop 1
	v_cndmask_b32_e32 v84, v205, v84, vcc
	v_cmp_ge_i32_e32 vcc, v196, v228
	s_nop 1
	v_cndmask_b32_e32 v85, v205, v85, vcc
	v_cmp_ge_i32_e32 vcc, v196, v229
	s_nop 1
	v_cndmask_b32_e32 v86, v205, v86, vcc
	v_cmp_ge_i32_e32 vcc, v196, v230
	s_nop 1
	v_cndmask_b32_e32 v87, v205, v87, vcc
	v_cmp_ge_i32_e32 vcc, v196, v231
	s_nop 1
	v_cndmask_b32_e32 v88, v205, v88, vcc
	v_cmp_ge_i32_e32 vcc, v196, v239
	s_nop 1
	v_cndmask_b32_e32 v89, v205, v89, vcc
	v_cmp_ge_i32_e32 vcc, v196, v240
	s_nop 1
	v_cndmask_b32_e32 v90, v205, v90, vcc
	v_cmp_ge_i32_e32 vcc, v196, v241
	s_nop 1
	v_cndmask_b32_e32 v91, v205, v91, vcc
	v_cmp_ge_i32_e32 vcc, v196, v242
	s_nop 1
	v_cndmask_b32_e32 v92, v205, v92, vcc
	v_cmp_ge_i32_e32 vcc, v196, v243
	s_nop 1
	v_cndmask_b32_e32 v93, v205, v93, vcc
	v_cmp_ge_i32_e32 vcc, v196, v244
	s_nop 1
	v_cndmask_b32_e32 v94, v205, v94, vcc
	v_cmp_ge_i32_e32 vcc, v196, v202
	s_nop 1
	v_cndmask_b32_e32 v95, v205, v95, vcc
	v_cmp_ge_i32_e32 vcc, v196, v203
	s_nop 1
	v_cndmask_b32_e32 v96, v205, v96, vcc
	v_cmp_ge_i32_e32 vcc, v196, v204
	s_nop 1
	v_cndmask_b32_e32 v97, v205, v97, vcc
	s_cbranch_scc1 .LBB0_47
; %bb.46:
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[134:137], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[134:137], v[82:97]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[130:133], v[66:81]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[130:133], v[82:97]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[142:145], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[142:145], v[82:97]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[138:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[138:141], v[82:97]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[158:161], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[82:97], v[114:117], v[158:161], v[82:97]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[154:157], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[154:157], v[82:97]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[150:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[150:153], v[82:97]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[190:193], v[146:149], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[146:149], v[82:97]
.LBB0_47:
	v_lshl_add_u32 v1, v214, 1, s55
	s_waitcnt lgkmcnt(1)
	ds_read_b64_tr_b16 v[190:191], v1
	ds_read_b64_tr_b16 v[192:193], v1 offset:2048
	ds_read_b64_tr_b16 v[186:187], v1 offset:4096
	ds_read_b64_tr_b16 v[188:189], v1 offset:6144
	ds_read_b64_tr_b16 v[182:183], v1 offset:8192
	ds_read_b64_tr_b16 v[184:185], v1 offset:10240
	ds_read_b64_tr_b16 v[178:179], v1 offset:12288
	ds_read_b64_tr_b16 v[180:181], v1 offset:14336
	v_lshl_add_u32 v1, v237, 1, s55
	ds_read_b64_tr_b16 v[174:175], v1
	ds_read_b64_tr_b16 v[176:177], v1 offset:2048
	ds_read_b64_tr_b16 v[170:171], v1 offset:4096
	ds_read_b64_tr_b16 v[172:173], v1 offset:6144
	ds_read_b64_tr_b16 v[166:167], v1 offset:8192
	ds_read_b64_tr_b16 v[168:169], v1 offset:10240
	ds_read_b64_tr_b16 v[162:163], v1 offset:12288
	ds_read_b64_tr_b16 v[164:165], v1 offset:14336
	v_lshl_add_u32 v1, v235, 1, s55
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[126:127], v1
	ds_read_b64_tr_b16 v[128:129], v1 offset:2048
	ds_read_b64_tr_b16 v[122:123], v1 offset:4096
	ds_read_b64_tr_b16 v[124:125], v1 offset:6144
	ds_read_b64_tr_b16 v[118:119], v1 offset:8192
	ds_read_b64_tr_b16 v[120:121], v1 offset:10240
	ds_read_b64_tr_b16 v[114:115], v1 offset:12288
	ds_read_b64_tr_b16 v[116:117], v1 offset:14336
	v_lshl_add_u32 v1, v233, 1, s55
	ds_read_b64_tr_b16 v[110:111], v1
	ds_read_b64_tr_b16 v[112:113], v1 offset:2048
	ds_read_b64_tr_b16 v[106:107], v1 offset:4096
	ds_read_b64_tr_b16 v[108:109], v1 offset:6144
	ds_read_b64_tr_b16 v[102:103], v1 offset:8192
	ds_read_b64_tr_b16 v[104:105], v1 offset:10240
	ds_read_b64_tr_b16 v[98:99], v1 offset:12288
	ds_read_b64_tr_b16 v[100:101], v1 offset:14336
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_49
; %bb.48:
	v_max_f32_e32 v1, v67, v67
	v_max_f32_e32 v194, v66, v66
	v_max_f32_e32 v1, v194, v1
	v_max3_f32 v1, v1, v68, v69
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_max3_f32 v1, v1, v74, v75
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_mov_b32_e32 v194, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v194
	v_max3_f32 v227, v198, v1, v194
	s_mov_b32 s4, 0x3e0293ee
	v_mul_f32_e32 v1, 0x3e0293ee, v227
	v_fma_f32 v66, v66, s4, -v1
	v_fma_f32 v67, v67, s4, -v1
	v_fma_f32 v68, v68, s4, -v1
	v_fma_f32 v69, v69, s4, -v1
	v_fma_f32 v70, v70, s4, -v1
	v_fma_f32 v71, v71, s4, -v1
	v_fma_f32 v72, v72, s4, -v1
	v_fma_f32 v73, v73, s4, -v1
	v_fma_f32 v74, v74, s4, -v1
	v_fma_f32 v75, v75, s4, -v1
	v_fma_f32 v76, v76, s4, -v1
	v_fma_f32 v77, v77, s4, -v1
	v_fma_f32 v78, v78, s4, -v1
	v_fma_f32 v79, v79, s4, -v1
	v_fma_f32 v80, v80, s4, -v1
	v_fma_f32 v81, v81, s4, -v1
	v_fma_f32 v82, v82, s4, -v1
	v_fma_f32 v83, v83, s4, -v1
	v_fma_f32 v84, v84, s4, -v1
	v_fma_f32 v85, v85, s4, -v1
	v_fma_f32 v86, v86, s4, -v1
	v_fma_f32 v87, v87, s4, -v1
	v_fma_f32 v88, v88, s4, -v1
	v_fma_f32 v89, v89, s4, -v1
	v_fma_f32 v90, v90, s4, -v1
	v_fma_f32 v91, v91, s4, -v1
	v_fma_f32 v92, v92, s4, -v1
	v_fma_f32 v93, v93, s4, -v1
	v_fma_f32 v94, v94, s4, -v1
	v_fma_f32 v95, v95, s4, -v1
	v_fma_f32 v96, v96, s4, -v1
	v_fma_f32 v97, v97, s4, -v1
	v_fma_f32 v1, v198, s4, -v1
	v_exp_f32_e32 v194, v66
	v_exp_f32_e32 v195, v67
	v_exp_f32_e32 v199, v68
	v_exp_f32_e32 v200, v69
	v_exp_f32_e32 v201, v70
	v_exp_f32_e32 v202, v71
	v_exp_f32_e32 v203, v72
	v_exp_f32_e32 v204, v73
	v_exp_f32_e32 v1, v1
	v_cvt_pk_f16_f32 v68, v201, v202
	v_cvt_pk_f16_f32 v67, v199, v200
	v_cvt_pk_f16_f32 v69, v203, v204
	v_cvt_pk_f16_f32 v66, v194, v195
	v_mul_f32_e32 v32, v32, v1
	v_mul_f32_e32 v33, v33, v1
	v_mul_f32_e32 v30, v30, v1
	v_mul_f32_e32 v31, v31, v1
	v_mul_f32_e32 v28, v28, v1
	v_mul_f32_e32 v29, v29, v1
	v_mul_f32_e32 v26, v26, v1
	v_mul_f32_e32 v27, v27, v1
	v_mul_f32_e32 v24, v24, v1
	v_mul_f32_e32 v25, v25, v1
	v_mul_f32_e32 v22, v22, v1
	v_mul_f32_e32 v23, v23, v1
	v_mul_f32_e32 v20, v20, v1
	v_mul_f32_e32 v21, v21, v1
	v_mul_f32_e32 v18, v18, v1
	v_mul_f32_e32 v19, v19, v1
	v_exp_f32_e32 v205, v74
	v_exp_f32_e32 v215, v75
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[66:69], v[18:33]
	v_exp_f32_e32 v216, v76
	v_exp_f32_e32 v217, v77
	v_exp_f32_e32 v218, v78
	v_exp_f32_e32 v219, v79
	v_exp_f32_e32 v220, v80
	v_exp_f32_e32 v221, v81
	v_cvt_pk_f16_f32 v71, v216, v217
	v_cvt_pk_f16_f32 v72, v218, v219
	v_cvt_pk_f16_f32 v70, v205, v215
	v_cvt_pk_f16_f32 v73, v220, v221
	v_mul_f32_e32 v48, v48, v1
	v_mul_f32_e32 v49, v49, v1
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[70:73], v[18:33]
	v_add_f32_e32 v170, v194, v195
	v_add_f32_e32 v170, v199, v170
	v_mul_f32_e32 v46, v46, v1
	v_mul_f32_e32 v47, v47, v1
	v_mul_f32_e32 v44, v44, v1
	v_mul_f32_e32 v45, v45, v1
	v_mul_f32_e32 v42, v42, v1
	v_mul_f32_e32 v43, v43, v1
	v_mul_f32_e32 v40, v40, v1
	v_mul_f32_e32 v41, v41, v1
	v_mul_f32_e32 v38, v38, v1
	v_mul_f32_e32 v39, v39, v1
	v_mul_f32_e32 v36, v36, v1
	v_mul_f32_e32 v37, v37, v1
	v_mul_f32_e32 v34, v34, v1
	v_mul_f32_e32 v35, v35, v1
	v_add_f32_e32 v170, v200, v170
	v_mul_f32_e32 v16, v16, v1
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[66:69], v[34:49]
	v_mul_f32_e32 v17, v17, v1
	v_mul_f32_e32 v14, v14, v1
	v_mul_f32_e32 v15, v15, v1
	v_mul_f32_e32 v12, v12, v1
	v_mul_f32_e32 v13, v13, v1
	v_mul_f32_e32 v10, v10, v1
	v_mul_f32_e32 v11, v11, v1
	v_mul_f32_e32 v8, v8, v1
	v_mul_f32_e32 v9, v9, v1
	v_mul_f32_e32 v6, v6, v1
	v_mul_f32_e32 v7, v7, v1
	v_mul_f32_e32 v4, v4, v1
	v_mul_f32_e32 v5, v5, v1
	v_mul_f32_e32 v2, v2, v1
	v_mul_f32_e32 v3, v3, v1
	v_add_f32_e32 v126, v201, v170
	v_mul_f32_e32 v64, v64, v1
	v_mul_f32_e32 v65, v65, v1
	v_mul_f32_e32 v62, v62, v1
	v_mul_f32_e32 v63, v63, v1
	v_mul_f32_e32 v60, v60, v1
	v_mul_f32_e32 v61, v61, v1
	v_mul_f32_e32 v58, v58, v1
	v_mul_f32_e32 v59, v59, v1
	v_mul_f32_e32 v56, v56, v1
	v_mul_f32_e32 v57, v57, v1
	v_mul_f32_e32 v54, v54, v1
	v_mul_f32_e32 v55, v55, v1
	v_mul_f32_e32 v52, v52, v1
	v_mul_f32_e32 v53, v53, v1
	v_mul_f32_e32 v50, v50, v1
	v_mul_f32_e32 v51, v51, v1
	v_mfma_f32_32x32x16_f16 v[2:17], v[190:193], v[66:69], v[2:17]
	v_add_f32_e32 v126, v202, v126
	v_add_f32_e32 v126, v203, v126
	v_add_f32_e32 v126, v204, v126
	v_add_f32_e32 v126, v205, v126
	v_add_f32_e32 v126, v215, v126
	v_add_f32_e32 v126, v216, v126
	v_exp_f32_e32 v82, v82
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[50:65], v[110:113], v[66:69], v[50:65]
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v85, v85
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[70:73], v[34:49]
	v_add_f32_e32 v122, v217, v126
	v_add_f32_e32 v122, v218, v122
	v_add_f32_e32 v122, v219, v122
	v_add_f32_e32 v122, v220, v122
	v_add_f32_e32 v122, v221, v122
	v_add_f32_e32 v66, v82, v122
	v_add_f32_e32 v66, v83, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[186:189], v[70:73], v[2:17]
	v_add_f32_e32 v66, v84, v66
	v_add_f32_e32 v66, v85, v66
	v_exp_f32_e32 v90, v90
	v_cvt_pk_f16_f32 v77, v88, v89
	v_cvt_pk_f16_f32 v76, v86, v87
	v_cvt_pk_f16_f32 v75, v84, v85
	v_cvt_pk_f16_f32 v74, v82, v83
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[106:109], v[70:73], v[50:65]
	v_add_f32_e32 v66, v86, v66
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v66, v87, v66
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v66, v88, v66
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v66, v89, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[182:185], v[74:77], v[2:17]
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_add_f32_e32 v66, v90, v66
	v_add_f32_e32 v66, v91, v66
	v_add_f32_e32 v66, v92, v66
	v_mfma_f32_32x32x16_f16 v[18:33], v[166:169], v[74:77], v[18:33]
	v_add_f32_e32 v66, v93, v66
	v_cvt_pk_f16_f32 v81, v96, v97
	v_cvt_pk_f16_f32 v80, v94, v95
	v_cvt_pk_f16_f32 v79, v92, v93
	v_cvt_pk_f16_f32 v78, v90, v91
	v_add_f32_e32 v66, v94, v66
	v_add_f32_e32 v66, v95, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[118:121], v[74:77], v[34:49]
	v_add_f32_e32 v66, v96, v66
	v_add_f32_e32 v66, v97, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v247, v1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[102:105], v[74:77], v[50:65]
	v_mov_b32_e32 v198, v227
	v_mov_b32_e32 v247, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[178:181], v[78:81], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[162:165], v[78:81], v[18:33]
	v_mfma_f32_32x32x16_f16 v[34:49], v[114:117], v[78:81], v[34:49]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[98:101], v[78:81], v[50:65]
.LBB0_49:
	v_add3_u32 v66, s17, v0, v232
	scratch_load_dword v0, off, off offset:24 ; 4-byte Folded Reload
	v_add3_u32 v1, s17, v206, v232
	v_add3_u32 v67, s17, v207, v232
	v_add3_u32 v68, s17, v208, v232
	s_cmp_lg_u32 s7, s59
	s_cselect_b64 s[4:5], -1, 0
	s_or_b64 s[2:3], s[2:3], s[4:5]
	s_and_b64 vcc, exec, s[2:3]
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
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
	s_waitcnt vmcnt(0)
	v_add3_u32 v69, s17, v0, v232
	scratch_load_dword v0, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v70, s17, v0, v232
	scratch_load_dword v0, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v71, s17, v0, v232
	scratch_load_dword v0, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v72, s17, v0, v232
	s_waitcnt lgkmcnt(14)
	ds_read_b128 v[162:165], v1
	s_waitcnt lgkmcnt(1)
	ds_read_b128 v[98:101], v1 offset:8192
	ds_read_b128 v[166:169], v66
	ds_read_b128 v[102:105], v66 offset:8192
	ds_read_b128 v[170:173], v67
	ds_read_b128 v[106:109], v67 offset:8192
	ds_read_b128 v[174:177], v68
	ds_read_b128 v[110:113], v68 offset:8192
	ds_read_b128 v[178:181], v69
	ds_read_b128 v[114:117], v69 offset:8192
	ds_read_b128 v[182:185], v70
	ds_read_b128 v[118:121], v70 offset:8192
	ds_read_b128 v[186:189], v71
	ds_read_b128 v[122:125], v71 offset:8192
	ds_read_b128 v[190:193], v72
	ds_read_b128 v[126:129], v72 offset:8192
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	s_cbranch_vccnz .LBB0_51
; %bb.50:
	scratch_load_dword v0, off, off offset:20 ; 4-byte Folded Reload
	v_or_b32_e32 v67, s13, v197
	v_mov_b32_e32 v194, 0xff800000
	v_or_b32_e32 v68, s13, v209
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v1, s13, v0
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	v_cmp_gt_i32_e32 vcc, s67, v1
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v69, s13, v0
	scratch_load_dword v0, off, off offset:4 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v66, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v67
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v70, s13, v0
	scratch_load_dword v0, off, off offset:80 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v67, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v68
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v71, s13, v0
	scratch_load_dword v0, off, off offset:120 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v68, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v69
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v72, s13, v0
	scratch_load_dword v0, off, off offset:124 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v69, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v70
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v73, s13, v0
	scratch_load_dword v0, off, off offset:128 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v70, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v71
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v74, s13, v0
	scratch_load_dword v0, off, off offset:136 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v71, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v72
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v75, s13, v0
	scratch_load_dword v0, off, off offset:140 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v72, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v73
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v76, s13, v0
	scratch_load_dword v0, off, off offset:148 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v73, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v74
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v77, s13, v0
	scratch_load_dword v0, off, off offset:152 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v74, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v75
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v78, s13, v0
	scratch_load_dword v0, off, off offset:156 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v75, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v76
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v79, s13, v0
	scratch_load_dword v0, off, off offset:160 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v76, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v77
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v80, s13, v0
	scratch_load_dword v0, off, off offset:164 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v77, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v78
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v81, s13, v0
	scratch_load_dword v0, off, off offset:168 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v78, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v79
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v82, s13, v0
	scratch_load_dword v0, off, off offset:172 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v79, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v80
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v83, s13, v0
	scratch_load_dword v0, off, off offset:176 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v80, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v81
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v84, s13, v0
	scratch_load_dword v0, off, off offset:180 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v81, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v82
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v85, s13, v0
	scratch_load_dword v0, off, off offset:184 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v82, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v83
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v86, s13, v0
	scratch_load_dword v0, off, off offset:188 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v83, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v84
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v87, s13, v0
	scratch_load_dword v0, off, off offset:192 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v84, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v85
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v88, s13, v0
	scratch_load_dword v0, off, off offset:196 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v85, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v86
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v89, s13, v0
	scratch_load_dword v0, off, off offset:200 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v86, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v87
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v90, s13, v0
	scratch_load_dword v0, off, off offset:204 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v87, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v88
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v91, s13, v0
	scratch_load_dword v0, off, off offset:208 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v88, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v89
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v92, s13, v0
	scratch_load_dword v0, off, off offset:212 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v89, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v90
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v93, s13, v0
	scratch_load_dword v0, off, off offset:216 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v90, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v91
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v94, s13, v0
	scratch_load_dword v0, off, off offset:220 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v91, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v92
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v95, s13, v0
	scratch_load_dword v0, off, off offset:224 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v92, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v93
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v96, s13, v0
	scratch_load_dword v0, off, off offset:228 ; 4-byte Folded Reload
	v_cndmask_b32_e64 v93, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v94
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v97, s13, v0
	v_cndmask_b32_e64 v94, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v95
	s_nop 1
	v_cndmask_b32_e64 v95, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v96
	s_nop 1
	v_cndmask_b32_e64 v96, v194, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v97
	s_nop 1
	v_cndmask_b32_e64 v97, v194, 0, vcc
.LBB0_51:
	scratch_load_dword v0, off, off offset:8 ; 4-byte Folded Reload
	v_add_u32_e32 v224, s13, v210
	v_add_u32_e32 v225, s13, v211
	v_add_u32_e32 v226, s13, v212
	v_add_u32_e32 v194, s13, v234
	v_mov_b32_e32 v228, 0xff800000
	v_add_u32_e32 v195, s13, v236
	v_add_u32_e32 v197, s13, v238
	v_add_u32_e32 v199, s13, v246
	v_add_u32_e32 v200, s13, v248
	v_add_u32_e32 v201, s13, v249
	v_add_u32_e32 v202, s13, v250
	v_add_u32_e32 v203, s13, v251
	v_add_u32_e32 v204, s13, v252
	v_add_u32_e32 v205, s13, v253
	v_add_u32_e32 v215, s13, v255
	v_add_u32_e32 v216, s13, v245
	v_add_u32_e32 v217, s13, v254
	v_add_u32_e32 v206, s13, v213
	s_cmpk_gt_i32 s6, 0xbf
	s_cselect_b64 s[2:3], -1, 0
	s_cmpk_lt_i32 s6, 0xc0
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v1, s13, v0
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	v_cmp_ge_i32_e32 vcc, v196, v1
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v218, s13, v0
	scratch_load_dword v0, off, off offset:16 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v66, v228, v66, vcc
	v_cmp_ge_i32_e32 vcc, v196, v194
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v219, s13, v0
	scratch_load_dword v0, off, off offset:28 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v67, v228, v67, vcc
	v_cmp_ge_i32_e32 vcc, v196, v195
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v220, s13, v0
	scratch_load_dword v0, off, off offset:32 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v68, v228, v68, vcc
	v_cmp_ge_i32_e32 vcc, v196, v197
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v221, s13, v0
	scratch_load_dword v0, off, off offset:36 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v69, v228, v69, vcc
	v_cmp_ge_i32_e32 vcc, v196, v199
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v222, s13, v0
	scratch_load_dword v0, off, off offset:40 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v70, v228, v70, vcc
	v_cmp_ge_i32_e32 vcc, v196, v200
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v223, s13, v0
	scratch_load_dword v0, off, off offset:44 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v71, v228, v71, vcc
	v_cmp_ge_i32_e32 vcc, v196, v201
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v227, s13, v0
	scratch_load_dword v0, off, off offset:48 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v72, v228, v72, vcc
	v_cmp_ge_i32_e32 vcc, v196, v202
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v207, s13, v0
	scratch_load_dword v0, off, off offset:52 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v73, v228, v73, vcc
	v_cmp_ge_i32_e32 vcc, v196, v203
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v208, s13, v0
	scratch_load_dword v0, off, off offset:56 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v74, v228, v74, vcc
	v_cmp_ge_i32_e32 vcc, v196, v204
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v209, s13, v0
	scratch_load_dword v0, off, off offset:60 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v75, v228, v75, vcc
	v_cmp_ge_i32_e32 vcc, v196, v205
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v210, s13, v0
	scratch_load_dword v0, off, off offset:64 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v76, v228, v76, vcc
	v_cmp_ge_i32_e32 vcc, v196, v215
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v211, s13, v0
	scratch_load_dword v0, off, off offset:92 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v77, v228, v77, vcc
	v_cmp_ge_i32_e32 vcc, v196, v216
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v212, s13, v0
	scratch_load_dword v0, off, off offset:96 ; 4-byte Folded Reload
	v_cndmask_b32_e32 v78, v228, v78, vcc
	v_cmp_ge_i32_e32 vcc, v196, v217
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v213, s13, v0
	v_cndmask_b32_e32 v79, v228, v79, vcc
	v_cmp_ge_i32_e32 vcc, v196, v218
	s_nop 1
	v_cndmask_b32_e32 v80, v228, v80, vcc
	v_cmp_ge_i32_e32 vcc, v196, v219
	s_nop 1
	v_cndmask_b32_e32 v81, v228, v81, vcc
	v_cmp_ge_i32_e32 vcc, v196, v220
	s_nop 1
	v_cndmask_b32_e32 v82, v228, v82, vcc
	v_cmp_ge_i32_e32 vcc, v196, v221
	s_nop 1
	v_cndmask_b32_e32 v83, v228, v83, vcc
	v_cmp_ge_i32_e32 vcc, v196, v222
	s_nop 1
	v_cndmask_b32_e32 v84, v228, v84, vcc
	v_cmp_ge_i32_e32 vcc, v196, v223
	s_nop 1
	v_cndmask_b32_e32 v85, v228, v85, vcc
	v_cmp_ge_i32_e32 vcc, v196, v224
	s_nop 1
	v_cndmask_b32_e32 v86, v228, v86, vcc
	v_cmp_ge_i32_e32 vcc, v196, v225
	s_nop 1
	v_cndmask_b32_e32 v87, v228, v87, vcc
	v_cmp_ge_i32_e32 vcc, v196, v226
	s_nop 1
	v_cndmask_b32_e32 v88, v228, v88, vcc
	v_cmp_ge_i32_e32 vcc, v196, v227
	s_nop 1
	v_cndmask_b32_e32 v89, v228, v89, vcc
	v_cmp_ge_i32_e32 vcc, v196, v206
	s_nop 1
	v_cndmask_b32_e32 v90, v228, v90, vcc
	v_cmp_ge_i32_e32 vcc, v196, v207
	s_nop 1
	v_cndmask_b32_e32 v91, v228, v91, vcc
	v_cmp_ge_i32_e32 vcc, v196, v208
	s_nop 1
	v_cndmask_b32_e32 v92, v228, v92, vcc
	v_cmp_ge_i32_e32 vcc, v196, v209
	s_nop 1
	v_cndmask_b32_e32 v93, v228, v93, vcc
	v_cmp_ge_i32_e32 vcc, v196, v210
	s_nop 1
	v_cndmask_b32_e32 v94, v228, v94, vcc
	v_cmp_ge_i32_e32 vcc, v196, v211
	s_nop 1
	v_cndmask_b32_e32 v95, v228, v95, vcc
	v_cmp_ge_i32_e32 vcc, v196, v212
	s_nop 1
	v_cndmask_b32_e32 v96, v228, v96, vcc
	v_cmp_ge_i32_e32 vcc, v196, v213
	s_nop 1
	v_cndmask_b32_e32 v97, v228, v97, vcc
	s_cbranch_scc1 .LBB0_53
; %bb.52:
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[134:137], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[134:137], v[82:97]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[130:133], v[66:81]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[130:133], v[82:97]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[142:145], v[66:81]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[142:145], v[82:97]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[138:141], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[138:141], v[82:97]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[158:161], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[82:97], v[114:117], v[158:161], v[82:97]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[154:157], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[154:157], v[82:97]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[150:153], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[150:153], v[82:97]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[190:193], v[146:149], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[146:149], v[82:97]
.LBB0_53:
	v_lshl_add_u32 v1, v214, 1, s16
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	ds_read_b64_tr_b16 v[158:159], v1
	ds_read_b64_tr_b16 v[160:161], v1 offset:2048
	ds_read_b64_tr_b16 v[154:155], v1 offset:4096
	ds_read_b64_tr_b16 v[156:157], v1 offset:6144
	ds_read_b64_tr_b16 v[150:151], v1 offset:8192
	ds_read_b64_tr_b16 v[152:153], v1 offset:10240
	ds_read_b64_tr_b16 v[146:147], v1 offset:12288
	ds_read_b64_tr_b16 v[148:149], v1 offset:14336
	v_lshl_add_u32 v1, v237, 1, s16
	ds_read_b64_tr_b16 v[142:143], v1
	ds_read_b64_tr_b16 v[144:145], v1 offset:2048
	ds_read_b64_tr_b16 v[138:139], v1 offset:4096
	ds_read_b64_tr_b16 v[140:141], v1 offset:6144
	ds_read_b64_tr_b16 v[134:135], v1 offset:8192
	ds_read_b64_tr_b16 v[136:137], v1 offset:10240
	ds_read_b64_tr_b16 v[130:131], v1 offset:12288
	ds_read_b64_tr_b16 v[132:133], v1 offset:14336
	v_lshl_add_u32 v1, v235, 1, s16
	s_waitcnt lgkmcnt(14)
	ds_read_b64_tr_b16 v[126:127], v1
	ds_read_b64_tr_b16 v[128:129], v1 offset:2048
	ds_read_b64_tr_b16 v[122:123], v1 offset:4096
	ds_read_b64_tr_b16 v[124:125], v1 offset:6144
	ds_read_b64_tr_b16 v[118:119], v1 offset:8192
	ds_read_b64_tr_b16 v[120:121], v1 offset:10240
	ds_read_b64_tr_b16 v[114:115], v1 offset:12288
	ds_read_b64_tr_b16 v[116:117], v1 offset:14336
	v_lshl_add_u32 v1, v233, 1, s16
	ds_read_b64_tr_b16 v[110:111], v1
	ds_read_b64_tr_b16 v[112:113], v1 offset:2048
	ds_read_b64_tr_b16 v[106:107], v1 offset:4096
	ds_read_b64_tr_b16 v[108:109], v1 offset:6144
	ds_read_b64_tr_b16 v[102:103], v1 offset:8192
	ds_read_b64_tr_b16 v[104:105], v1 offset:10240
	ds_read_b64_tr_b16 v[98:99], v1 offset:12288
	ds_read_b64_tr_b16 v[100:101], v1 offset:14336
	scratch_load_dword v180, off, off offset:100 ; 4-byte Folded Reload
	s_andn2_b64 vcc, exec, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v179, 31, v0
	s_cbranch_vccnz .LBB0_55
; %bb.54:
	v_max_f32_e32 v1, v67, v67
	v_max_f32_e32 v162, v66, v66
	v_max_f32_e32 v1, v162, v1
	v_max3_f32 v1, v1, v68, v69
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_max3_f32 v1, v1, v74, v75
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_mov_b32_e32 v162, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v162
	v_max3_f32 v1, v198, v1, v162
	s_mov_b32 s2, 0x3e0293ee
	v_mul_f32_e32 v162, 0x3e0293ee, v1
	v_fma_f32 v66, v66, s2, -v162
	v_fma_f32 v67, v67, s2, -v162
	v_fma_f32 v68, v68, s2, -v162
	v_fma_f32 v69, v69, s2, -v162
	v_fma_f32 v70, v70, s2, -v162
	v_fma_f32 v71, v71, s2, -v162
	v_fma_f32 v72, v72, s2, -v162
	v_fma_f32 v73, v73, s2, -v162
	v_exp_f32_e32 v163, v66
	v_fma_f32 v66, v198, s2, -v162
	v_fma_f32 v74, v74, s2, -v162
	v_fma_f32 v75, v75, s2, -v162
	v_fma_f32 v76, v76, s2, -v162
	v_fma_f32 v77, v77, s2, -v162
	v_fma_f32 v78, v78, s2, -v162
	v_fma_f32 v79, v79, s2, -v162
	v_fma_f32 v80, v80, s2, -v162
	v_fma_f32 v81, v81, s2, -v162
	v_fma_f32 v82, v82, s2, -v162
	v_fma_f32 v83, v83, s2, -v162
	v_fma_f32 v84, v84, s2, -v162
	v_fma_f32 v85, v85, s2, -v162
	v_fma_f32 v86, v86, s2, -v162
	v_fma_f32 v87, v87, s2, -v162
	v_fma_f32 v88, v88, s2, -v162
	v_fma_f32 v89, v89, s2, -v162
	v_fma_f32 v90, v90, s2, -v162
	v_fma_f32 v91, v91, s2, -v162
	v_fma_f32 v92, v92, s2, -v162
	v_fma_f32 v93, v93, s2, -v162
	v_fma_f32 v94, v94, s2, -v162
	v_fma_f32 v95, v95, s2, -v162
	v_fma_f32 v96, v96, s2, -v162
	v_fma_f32 v97, v97, s2, -v162
	v_exp_f32_e32 v164, v67
	v_exp_f32_e32 v165, v68
	v_exp_f32_e32 v166, v69
	v_exp_f32_e32 v167, v70
	v_exp_f32_e32 v168, v71
	v_exp_f32_e32 v169, v72
	v_exp_f32_e32 v170, v73
	v_exp_f32_e32 v162, v66
	v_cvt_pk_f16_f32 v68, v167, v168
	v_cvt_pk_f16_f32 v67, v165, v166
	v_cvt_pk_f16_f32 v69, v169, v170
	v_cvt_pk_f16_f32 v66, v163, v164
	v_mul_f32_e32 v32, v32, v162
	v_mul_f32_e32 v33, v33, v162
	v_mul_f32_e32 v30, v30, v162
	v_mul_f32_e32 v31, v31, v162
	v_mul_f32_e32 v28, v28, v162
	v_mul_f32_e32 v29, v29, v162
	v_mul_f32_e32 v26, v26, v162
	v_mul_f32_e32 v27, v27, v162
	v_mul_f32_e32 v24, v24, v162
	v_mul_f32_e32 v25, v25, v162
	v_mul_f32_e32 v22, v22, v162
	v_mul_f32_e32 v23, v23, v162
	v_mul_f32_e32 v20, v20, v162
	v_mul_f32_e32 v21, v21, v162
	v_mul_f32_e32 v18, v18, v162
	v_mul_f32_e32 v19, v19, v162
	v_exp_f32_e32 v171, v74
	v_exp_f32_e32 v172, v75
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[18:33], v[142:145], v[66:69], v[18:33]
	v_exp_f32_e32 v173, v76
	v_exp_f32_e32 v174, v77
	v_exp_f32_e32 v175, v78
	v_exp_f32_e32 v176, v79
	v_exp_f32_e32 v177, v80
	v_exp_f32_e32 v178, v81
	v_cvt_pk_f16_f32 v71, v173, v174
	v_cvt_pk_f16_f32 v72, v175, v176
	v_cvt_pk_f16_f32 v70, v171, v172
	v_cvt_pk_f16_f32 v73, v177, v178
	v_mul_f32_e32 v48, v48, v162
	v_mul_f32_e32 v49, v49, v162
	v_mfma_f32_32x32x16_f16 v[18:33], v[138:141], v[70:73], v[18:33]
	v_add_f32_e32 v138, v163, v164
	v_add_f32_e32 v138, v165, v138
	v_mul_f32_e32 v46, v46, v162
	v_mul_f32_e32 v47, v47, v162
	v_mul_f32_e32 v44, v44, v162
	v_mul_f32_e32 v45, v45, v162
	v_mul_f32_e32 v42, v42, v162
	v_mul_f32_e32 v43, v43, v162
	v_mul_f32_e32 v40, v40, v162
	v_mul_f32_e32 v41, v41, v162
	v_mul_f32_e32 v38, v38, v162
	v_mul_f32_e32 v39, v39, v162
	v_mul_f32_e32 v36, v36, v162
	v_mul_f32_e32 v37, v37, v162
	v_mul_f32_e32 v34, v34, v162
	v_mul_f32_e32 v35, v35, v162
	v_add_f32_e32 v138, v166, v138
	v_mul_f32_e32 v16, v16, v162
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[66:69], v[34:49]
	v_mul_f32_e32 v17, v17, v162
	v_mul_f32_e32 v14, v14, v162
	v_mul_f32_e32 v15, v15, v162
	v_mul_f32_e32 v12, v12, v162
	v_mul_f32_e32 v13, v13, v162
	v_mul_f32_e32 v10, v10, v162
	v_mul_f32_e32 v11, v11, v162
	v_mul_f32_e32 v8, v8, v162
	v_mul_f32_e32 v9, v9, v162
	v_mul_f32_e32 v6, v6, v162
	v_mul_f32_e32 v7, v7, v162
	v_mul_f32_e32 v4, v4, v162
	v_mul_f32_e32 v5, v5, v162
	v_mul_f32_e32 v2, v2, v162
	v_mul_f32_e32 v3, v3, v162
	v_add_f32_e32 v126, v167, v138
	v_mul_f32_e32 v64, v64, v162
	v_mul_f32_e32 v65, v65, v162
	v_mul_f32_e32 v62, v62, v162
	v_mul_f32_e32 v63, v63, v162
	v_mul_f32_e32 v60, v60, v162
	v_mul_f32_e32 v61, v61, v162
	v_mul_f32_e32 v58, v58, v162
	v_mul_f32_e32 v59, v59, v162
	v_mul_f32_e32 v56, v56, v162
	v_mul_f32_e32 v57, v57, v162
	v_mul_f32_e32 v54, v54, v162
	v_mul_f32_e32 v55, v55, v162
	v_mul_f32_e32 v52, v52, v162
	v_mul_f32_e32 v53, v53, v162
	v_mul_f32_e32 v50, v50, v162
	v_mul_f32_e32 v51, v51, v162
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[66:69], v[2:17]
	v_add_f32_e32 v126, v168, v126
	v_add_f32_e32 v126, v169, v126
	v_add_f32_e32 v126, v170, v126
	v_add_f32_e32 v126, v171, v126
	v_add_f32_e32 v126, v172, v126
	v_add_f32_e32 v126, v173, v126
	v_exp_f32_e32 v82, v82
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[50:65], v[110:113], v[66:69], v[50:65]
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v85, v85
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[70:73], v[34:49]
	v_add_f32_e32 v122, v174, v126
	v_add_f32_e32 v122, v175, v122
	v_add_f32_e32 v122, v176, v122
	v_add_f32_e32 v122, v177, v122
	v_add_f32_e32 v122, v178, v122
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
	v_mfma_f32_32x32x16_f16 v[50:65], v[106:109], v[70:73], v[50:65]
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
	v_mfma_f32_32x32x16_f16 v[18:33], v[134:137], v[74:77], v[18:33]
	v_add_f32_e32 v66, v93, v66
	v_cvt_pk_f16_f32 v81, v96, v97
	v_cvt_pk_f16_f32 v80, v94, v95
	v_cvt_pk_f16_f32 v79, v92, v93
	v_cvt_pk_f16_f32 v78, v90, v91
	v_add_f32_e32 v66, v94, v66
	v_add_f32_e32 v66, v95, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[118:121], v[74:77], v[34:49]
	v_add_f32_e32 v66, v96, v66
	v_add_f32_e32 v66, v97, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v247, v162
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[102:105], v[74:77], v[50:65]
	v_mov_b32_e32 v198, v1
	v_mov_b32_e32 v247, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[146:149], v[78:81], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[130:133], v[78:81], v[18:33]
	v_mfma_f32_32x32x16_f16 v[34:49], v[114:117], v[78:81], v[34:49]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[98:101], v[78:81], v[50:65]
.LBB0_55:
	v_div_scale_f32 v1, s[2:3], v247, v247, 1.0
	v_rcp_f32_e32 v1, v1
	v_div_scale_f32 v66, vcc, 1.0, v247, 1.0
	s_add_i32 s2, s60, 0x100
	v_mul_f32_e32 v1, v66, v1
	s_cmp_le_i32 s53, s60
	s_nop 0
	v_div_fmas_f32 v1, 0, 0, v1
	v_div_fixup_f32 v66, v1, v247, 1.0
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_ge_i32 s53, s2
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[66:67], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[66:67], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[66:67], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[66:67], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[66:67], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[66:67], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[66:67], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[66:67], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[66:67], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[66:67], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[66:67], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[66:67], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[66:67], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[66:67], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[66:67], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[66:67], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[66:67], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[66:67], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[66:67], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[66:67], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[66:67], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[66:67], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[66:67], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[66:67], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[66:67], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[66:67], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[66:67], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[94:95], v[66:67], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[60:61], v[66:67] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[98:99], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[64:65], v[66:67] op_sel_hi:[1,0]
	s_cselect_b64 s[6:7], -1, 0
	v_cvt_pk_f16_f32 v62, v2, v3
	v_cvt_pk_f16_f32 v58, v4, v5
	v_cvt_pk_f16_f32 v64, v6, v7
	v_cvt_pk_f16_f32 v61, v8, v9
	v_cvt_pk_f16_f32 v53, v10, v11
	v_cvt_pk_f16_f32 v50, v12, v13
	v_cvt_pk_f16_f32 v56, v14, v15
	v_cvt_pk_f16_f32 v54, v16, v17
	v_cvt_pk_f16_f32 v45, v18, v19
	v_cvt_pk_f16_f32 v42, v20, v21
	v_cvt_pk_f16_f32 v48, v22, v23
	v_cvt_pk_f16_f32 v46, v24, v25
	v_cvt_pk_f16_f32 v37, v26, v27
	v_cvt_pk_f16_f32 v34, v28, v29
	v_cvt_pk_f16_f32 v40, v30, v31
	v_cvt_pk_f16_f32 v38, v68, v69
	v_cvt_pk_f16_f32 v29, v70, v71
	v_cvt_pk_f16_f32 v26, v72, v73
	v_cvt_pk_f16_f32 v32, v74, v75
	v_cvt_pk_f16_f32 v30, v76, v77
	v_cvt_pk_f16_f32 v21, v78, v79
	v_cvt_pk_f16_f32 v18, v80, v81
	v_cvt_pk_f16_f32 v24, v82, v83
	v_cvt_pk_f16_f32 v22, v84, v85
	v_cvt_pk_f16_f32 v13, v86, v87
	v_cvt_pk_f16_f32 v10, v88, v89
	v_cvt_pk_f16_f32 v16, v90, v91
	v_cvt_pk_f16_f32 v14, v92, v93
	v_cvt_pk_f16_f32 v5, v94, v95
	v_cvt_pk_f16_f32 v2, v96, v97
	v_cvt_pk_f16_f32 v8, v98, v99
	v_cvt_pk_f16_f32 v6, v66, v67
	s_or_b64 s[4:5], s[4:5], s[6:7]
	v_lshrrev_b32_e32 v59, 16, v62
	v_lshrrev_b32_e32 v57, 16, v58
	v_lshrrev_b32_e32 v63, 16, v64
	v_lshrrev_b32_e32 v60, 16, v61
	v_lshrrev_b32_e32 v51, 16, v53
	v_lshrrev_b32_e32 v49, 16, v50
	v_lshrrev_b32_e32 v55, 16, v56
	v_lshrrev_b32_e32 v52, 16, v54
	v_lshrrev_b32_e32 v43, 16, v45
	v_lshrrev_b32_e32 v41, 16, v42
	v_lshrrev_b32_e32 v47, 16, v48
	v_lshrrev_b32_e32 v44, 16, v46
	v_lshrrev_b32_e32 v35, 16, v37
	v_lshrrev_b32_e32 v33, 16, v34
	v_lshrrev_b32_e32 v39, 16, v40
	v_lshrrev_b32_e32 v36, 16, v38
	v_lshrrev_b32_e32 v27, 16, v29
	v_lshrrev_b32_e32 v25, 16, v26
	v_lshrrev_b32_e32 v31, 16, v32
	v_lshrrev_b32_e32 v28, 16, v30
	v_lshrrev_b32_e32 v19, 16, v21
	v_lshrrev_b32_e32 v17, 16, v18
	v_lshrrev_b32_e32 v23, 16, v24
	v_lshrrev_b32_e32 v20, 16, v22
	v_lshrrev_b32_e32 v11, 16, v13
	v_lshrrev_b32_e32 v9, 16, v10
	v_lshrrev_b32_e32 v15, 16, v16
	v_lshrrev_b32_e32 v12, 16, v14
	v_lshrrev_b32_e32 v3, 16, v5
	v_lshrrev_b32_e32 v1, 16, v2
	v_lshrrev_b32_e32 v7, 16, v8
	v_lshrrev_b32_e32 v4, 16, v6
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_57
; %bb.56:
	v_cmp_gt_i32_e32 vcc, s53, v196
	s_nop 1
	v_cndmask_b32_e64 v62, v62, 0, vcc
	v_cndmask_b32_e64 v59, v59, 0, vcc
	v_cndmask_b32_e64 v58, v58, 0, vcc
	v_cndmask_b32_e64 v57, v57, 0, vcc
	v_cndmask_b32_e64 v64, v64, 0, vcc
	v_cndmask_b32_e64 v63, v63, 0, vcc
	v_cndmask_b32_e64 v61, v61, 0, vcc
	v_cndmask_b32_e64 v60, v60, 0, vcc
	v_cndmask_b32_e64 v53, v53, 0, vcc
	v_cndmask_b32_e64 v51, v51, 0, vcc
	v_cndmask_b32_e64 v50, v50, 0, vcc
	v_cndmask_b32_e64 v49, v49, 0, vcc
	v_cndmask_b32_e64 v56, v56, 0, vcc
	v_cndmask_b32_e64 v55, v55, 0, vcc
	v_cndmask_b32_e64 v54, v54, 0, vcc
	v_cndmask_b32_e64 v52, v52, 0, vcc
	v_cndmask_b32_e64 v45, v45, 0, vcc
	v_cndmask_b32_e64 v43, v43, 0, vcc
	v_cndmask_b32_e64 v42, v42, 0, vcc
	v_cndmask_b32_e64 v41, v41, 0, vcc
	v_cndmask_b32_e64 v48, v48, 0, vcc
	v_cndmask_b32_e64 v47, v47, 0, vcc
	v_cndmask_b32_e64 v46, v46, 0, vcc
	v_cndmask_b32_e64 v44, v44, 0, vcc
	v_cndmask_b32_e64 v37, v37, 0, vcc
	v_cndmask_b32_e64 v35, v35, 0, vcc
	v_cndmask_b32_e64 v34, v34, 0, vcc
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cndmask_b32_e64 v40, v40, 0, vcc
	v_cndmask_b32_e64 v39, v39, 0, vcc
	v_cndmask_b32_e64 v38, v38, 0, vcc
	v_cndmask_b32_e64 v36, v36, 0, vcc
	v_cndmask_b32_e64 v29, v29, 0, vcc
	v_cndmask_b32_e64 v27, v27, 0, vcc
	v_cndmask_b32_e64 v26, v26, 0, vcc
	v_cndmask_b32_e64 v25, v25, 0, vcc
	v_cndmask_b32_e64 v32, v32, 0, vcc
	v_cndmask_b32_e64 v31, v31, 0, vcc
	v_cndmask_b32_e64 v30, v30, 0, vcc
	v_cndmask_b32_e64 v28, v28, 0, vcc
	v_cndmask_b32_e64 v21, v21, 0, vcc
	v_cndmask_b32_e64 v19, v19, 0, vcc
	v_cndmask_b32_e64 v18, v18, 0, vcc
	v_cndmask_b32_e64 v17, v17, 0, vcc
	v_cndmask_b32_e64 v24, v24, 0, vcc
	v_cndmask_b32_e64 v23, v23, 0, vcc
	v_cndmask_b32_e64 v22, v22, 0, vcc
	v_cndmask_b32_e64 v20, v20, 0, vcc
	v_cndmask_b32_e64 v13, v13, 0, vcc
	v_cndmask_b32_e64 v11, v11, 0, vcc
	v_cndmask_b32_e64 v10, v10, 0, vcc
	v_cndmask_b32_e64 v9, v9, 0, vcc
	v_cndmask_b32_e64 v16, v16, 0, vcc
	v_cndmask_b32_e64 v15, v15, 0, vcc
	v_cndmask_b32_e64 v14, v14, 0, vcc
	v_cndmask_b32_e64 v12, v12, 0, vcc
	v_cndmask_b32_e64 v5, v5, 0, vcc
	v_cndmask_b32_e64 v3, v3, 0, vcc
	v_cndmask_b32_e64 v2, v2, 0, vcc
	v_cndmask_b32_e64 v1, v1, 0, vcc
	v_cndmask_b32_e64 v8, v8, 0, vcc
	v_cndmask_b32_e64 v7, v7, 0, vcc
	v_cndmask_b32_e64 v6, v6, 0, vcc
	v_cndmask_b32_e64 v4, v4, 0, vcc
.LBB0_57:
	scratch_load_dword v69, off, off offset:116 ; 4-byte Folded Reload
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
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v65, v69, 2, 0
	s_cbranch_scc1 .LBB0_59
; %bb.58:
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v247
	v_mov_b32_e32 v66, 0x42000000
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v247, v67
	v_log_f32_e32 v67, v67
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_barrier
	v_sub_f32_e32 v66, v67, v66
	v_add_f32_e32 v66, v198, v66
	ds_write_b32 v65, v66
	v_mov_b32_e32 v66, 2
	v_lshlrev_b32_sdwa v66, v66, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v67, 0, v66
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v67, v67
	s_sub_i32 s2, 0x100, s2
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	v_bfrev_b32_e32 v68, 1
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cmp_gt_i32_e64 s[8:9], s33, v196
	s_and_b32 s5, s12, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e32 v66, v68, v66, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v66, s[4:7], 0 offen
	s_cbranch_execz .LBB0_60
	s_branch .LBB0_61
.LBB0_59:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_60:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v247
	v_mov_b32_e32 v66, 0x42000000
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v247, v67
	v_log_f32_e32 v67, v67
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_barrier
	v_sub_f32_e32 v66, v67, v66
	v_add_f32_e32 v66, v198, v66
	ds_write_b32 v65, v66
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
.LBB0_61:
	scratch_load_dword v71, off, off offset:112 ; 4-byte Folded Reload
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
	v_mul_lo_u32 v66, s43, v69
	v_and_b32_e32 v67, 0xfc, v180
	s_add_u32 s2, s2, s0
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s53, s52, 31
	s_lshl_b64 s[0:1], s[52:53], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s43, 0x3fff
	s_bitset1_b32 s2, 14
	s_mov_b32 s4, 0x5040100
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_waitcnt vmcnt(0)
	v_cmp_eq_u32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v0, v64, v62, vcc
	v_cndmask_b32_e32 v68, v63, v59, vcc
	v_cndmask_b32_e32 v62, v62, v64, vcc
	v_bitop3_b32 v64, v179, v71, 32 bitop3:0x36
	v_cndmask_b32_e32 v59, v59, v63, vcc
	v_lshrrev_b32_e32 v65, 2, v71
	v_bfe_i32 v0, v0, 0, 16
	v_bfe_i32 v68, v68, 0, 16
	v_cndmask_b32_e32 v69, v61, v58, vcc
	v_cndmask_b32_e32 v70, v60, v57, vcc
	v_bfe_i32 v62, v62, 0, 16
	v_lshlrev_b32_e32 v64, 2, v64
	v_bfe_i32 v59, v59, 0, 16
	v_cndmask_b32_e32 v58, v58, v61, vcc
	v_cndmask_b32_e32 v57, v57, v60, vcc
	v_cndmask_b32_e32 v60, v56, v53, vcc
	v_cndmask_b32_e32 v61, v55, v51, vcc
	v_cndmask_b32_e32 v63, v54, v50, vcc
	v_cndmask_b32_e32 v71, v52, v49, vcc
	v_cndmask_b32_e32 v53, v53, v56, vcc
	v_cndmask_b32_e32 v51, v51, v55, vcc
	v_cndmask_b32_e32 v50, v50, v54, vcc
	v_cndmask_b32_e32 v49, v49, v52, vcc
	v_cndmask_b32_e32 v52, v48, v45, vcc
	v_cndmask_b32_e32 v54, v47, v43, vcc
	v_cndmask_b32_e32 v55, v46, v42, vcc
	v_cndmask_b32_e32 v56, v44, v41, vcc
	v_cndmask_b32_e32 v45, v45, v48, vcc
	v_cndmask_b32_e32 v43, v43, v47, vcc
	v_cndmask_b32_e32 v42, v42, v46, vcc
	v_cndmask_b32_e32 v41, v41, v44, vcc
	v_cndmask_b32_e32 v44, v40, v37, vcc
	v_cndmask_b32_e32 v46, v39, v35, vcc
	v_cndmask_b32_e32 v47, v38, v34, vcc
	v_cndmask_b32_e32 v48, v36, v33, vcc
	v_cndmask_b32_e32 v37, v37, v40, vcc
	v_cndmask_b32_e32 v35, v35, v39, vcc
	v_cndmask_b32_e32 v34, v34, v38, vcc
	v_cndmask_b32_e32 v33, v33, v36, vcc
	v_cndmask_b32_e32 v36, v32, v29, vcc
	v_cndmask_b32_e32 v38, v31, v27, vcc
	v_cndmask_b32_e32 v39, v30, v26, vcc
	v_cndmask_b32_e32 v40, v28, v25, vcc
	v_cndmask_b32_e32 v29, v29, v32, vcc
	v_cndmask_b32_e32 v27, v27, v31, vcc
	v_cndmask_b32_e32 v26, v26, v30, vcc
	v_cndmask_b32_e32 v25, v25, v28, vcc
	v_cndmask_b32_e32 v28, v24, v21, vcc
	v_cndmask_b32_e32 v30, v23, v19, vcc
	v_cndmask_b32_e32 v31, v22, v18, vcc
	v_cndmask_b32_e32 v32, v20, v17, vcc
	v_cndmask_b32_e32 v21, v21, v24, vcc
	v_cndmask_b32_e32 v19, v19, v23, vcc
	v_cndmask_b32_e32 v18, v18, v22, vcc
	v_cndmask_b32_e32 v17, v17, v20, vcc
	v_cndmask_b32_e32 v20, v16, v13, vcc
	v_cndmask_b32_e32 v22, v15, v11, vcc
	v_cndmask_b32_e32 v23, v14, v10, vcc
	v_cndmask_b32_e32 v24, v12, v9, vcc
	v_cndmask_b32_e32 v13, v13, v16, vcc
	v_cndmask_b32_e32 v11, v11, v15, vcc
	v_cndmask_b32_e32 v10, v10, v14, vcc
	v_cndmask_b32_e32 v9, v9, v12, vcc
	v_cndmask_b32_e32 v12, v8, v5, vcc
	v_cndmask_b32_e32 v14, v7, v3, vcc
	v_cndmask_b32_e32 v15, v6, v2, vcc
	v_cndmask_b32_e32 v5, v5, v8, vcc
	v_cndmask_b32_e32 v3, v3, v7, vcc
	v_cndmask_b32_e32 v2, v2, v6, vcc
	ds_bpermute_b32 v0, v67, v0
	ds_bpermute_b32 v68, v67, v68
	v_bfe_i32 v69, v69, 0, 16
	v_bfe_i32 v70, v70, 0, 16
	ds_bpermute_b32 v62, v64, v62
	ds_bpermute_b32 v59, v64, v59
	v_bfe_i32 v58, v58, 0, 16
	v_bfe_i32 v57, v57, 0, 16
	v_bfe_i32 v60, v60, 0, 16
	v_bfe_i32 v61, v61, 0, 16
	v_bfe_i32 v63, v63, 0, 16
	v_bfe_i32 v53, v53, 0, 16
	v_bfe_i32 v51, v51, 0, 16
	v_bfe_i32 v50, v50, 0, 16
	v_bfe_i32 v52, v52, 0, 16
	v_bfe_i32 v54, v54, 0, 16
	v_bfe_i32 v55, v55, 0, 16
	v_bfe_i32 v45, v45, 0, 16
	v_bfe_i32 v43, v43, 0, 16
	v_bfe_i32 v42, v42, 0, 16
	v_bfe_i32 v44, v44, 0, 16
	v_bfe_i32 v46, v46, 0, 16
	v_bfe_i32 v47, v47, 0, 16
	v_bfe_i32 v37, v37, 0, 16
	v_bfe_i32 v35, v35, 0, 16
	v_bfe_i32 v34, v34, 0, 16
	v_bfe_i32 v36, v36, 0, 16
	v_bfe_i32 v38, v38, 0, 16
	v_bfe_i32 v39, v39, 0, 16
	v_bfe_i32 v29, v29, 0, 16
	v_bfe_i32 v27, v27, 0, 16
	v_bfe_i32 v26, v26, 0, 16
	v_bfe_i32 v28, v28, 0, 16
	v_bfe_i32 v30, v30, 0, 16
	v_bfe_i32 v31, v31, 0, 16
	v_bfe_i32 v21, v21, 0, 16
	v_bfe_i32 v19, v19, 0, 16
	v_bfe_i32 v18, v18, 0, 16
	v_bfe_i32 v20, v20, 0, 16
	v_bfe_i32 v22, v22, 0, 16
	v_bfe_i32 v23, v23, 0, 16
	v_bfe_i32 v13, v13, 0, 16
	v_bfe_i32 v11, v11, 0, 16
	v_bfe_i32 v10, v10, 0, 16
	v_bfe_i32 v12, v12, 0, 16
	v_bfe_i32 v14, v14, 0, 16
	v_bfe_i32 v15, v15, 0, 16
	v_cndmask_b32_e32 v16, v4, v1, vcc
	v_bfe_i32 v5, v5, 0, 16
	v_bfe_i32 v3, v3, 0, 16
	v_bfe_i32 v2, v2, 0, 16
	v_cndmask_b32_e32 v1, v1, v4, vcc
	ds_bpermute_b32 v69, v67, v69
	ds_bpermute_b32 v70, v67, v70
	ds_bpermute_b32 v58, v64, v58
	ds_bpermute_b32 v57, v64, v57
	ds_bpermute_b32 v60, v67, v60
	ds_bpermute_b32 v61, v67, v61
	ds_bpermute_b32 v63, v67, v63
	ds_bpermute_b32 v53, v64, v53
	ds_bpermute_b32 v51, v64, v51
	ds_bpermute_b32 v50, v64, v50
	ds_bpermute_b32 v52, v67, v52
	ds_bpermute_b32 v54, v67, v54
	ds_bpermute_b32 v55, v67, v55
	v_bfe_i32 v56, v56, 0, 16
	ds_bpermute_b32 v45, v64, v45
	ds_bpermute_b32 v43, v64, v43
	ds_bpermute_b32 v42, v64, v42
	v_bfe_i32 v41, v41, 0, 16
	ds_bpermute_b32 v44, v67, v44
	ds_bpermute_b32 v46, v67, v46
	ds_bpermute_b32 v47, v67, v47
	v_bfe_i32 v48, v48, 0, 16
	ds_bpermute_b32 v37, v64, v37
	ds_bpermute_b32 v35, v64, v35
	ds_bpermute_b32 v34, v64, v34
	v_bfe_i32 v33, v33, 0, 16
	ds_bpermute_b32 v36, v67, v36
	ds_bpermute_b32 v38, v67, v38
	ds_bpermute_b32 v39, v67, v39
	v_bfe_i32 v40, v40, 0, 16
	ds_bpermute_b32 v29, v64, v29
	ds_bpermute_b32 v27, v64, v27
	ds_bpermute_b32 v26, v64, v26
	v_bfe_i32 v25, v25, 0, 16
	ds_bpermute_b32 v28, v67, v28
	ds_bpermute_b32 v30, v67, v30
	ds_bpermute_b32 v31, v67, v31
	v_bfe_i32 v32, v32, 0, 16
	ds_bpermute_b32 v21, v64, v21
	ds_bpermute_b32 v19, v64, v19
	ds_bpermute_b32 v18, v64, v18
	v_bfe_i32 v17, v17, 0, 16
	ds_bpermute_b32 v20, v67, v20
	ds_bpermute_b32 v22, v67, v22
	ds_bpermute_b32 v23, v67, v23
	v_bfe_i32 v24, v24, 0, 16
	ds_bpermute_b32 v13, v64, v13
	ds_bpermute_b32 v11, v64, v11
	ds_bpermute_b32 v10, v64, v10
	v_bfe_i32 v9, v9, 0, 16
	ds_bpermute_b32 v12, v67, v12
	ds_bpermute_b32 v14, v67, v14
	ds_bpermute_b32 v15, v67, v15
	v_bfe_i32 v16, v16, 0, 16
	ds_bpermute_b32 v5, v64, v5
	ds_bpermute_b32 v3, v64, v3
	ds_bpermute_b32 v2, v64, v2
	v_bfe_i32 v1, v1, 0, 16
	v_bfe_i32 v71, v71, 0, 16
	v_bfe_i32 v49, v49, 0, 16
	ds_bpermute_b32 v56, v67, v56
	ds_bpermute_b32 v41, v64, v41
	ds_bpermute_b32 v48, v67, v48
	ds_bpermute_b32 v33, v64, v33
	ds_bpermute_b32 v40, v67, v40
	ds_bpermute_b32 v25, v64, v25
	ds_bpermute_b32 v32, v67, v32
	ds_bpermute_b32 v17, v64, v17
	ds_bpermute_b32 v24, v67, v24
	ds_bpermute_b32 v9, v64, v9
	ds_bpermute_b32 v16, v67, v16
	ds_bpermute_b32 v1, v64, v1
	ds_bpermute_b32 v71, v67, v71
	ds_bpermute_b32 v49, v64, v49
	s_waitcnt lgkmcnt(14)
	v_cndmask_b32_e32 v4, v62, v0, vcc
	v_cndmask_b32_e32 v6, v59, v68, vcc
	v_cndmask_b32_e32 v0, v0, v62, vcc
	v_cndmask_b32_e32 v59, v68, v59, vcc
	v_cndmask_b32_e32 v7, v58, v69, vcc
	v_cndmask_b32_e32 v8, v57, v70, vcc
	v_cndmask_b32_e32 v58, v69, v58, vcc
	v_cndmask_b32_e32 v57, v70, v57, vcc
	v_cndmask_b32_e32 v62, v53, v60, vcc
	v_cndmask_b32_e32 v64, v51, v61, vcc
	v_cndmask_b32_e32 v67, v50, v63, vcc
	v_cndmask_b32_e32 v53, v60, v53, vcc
	v_cndmask_b32_e32 v51, v61, v51, vcc
	v_cndmask_b32_e32 v50, v63, v50, vcc
	v_cndmask_b32_e32 v60, v45, v52, vcc
	v_cndmask_b32_e32 v61, v43, v54, vcc
	v_cndmask_b32_e32 v63, v42, v55, vcc
	v_cndmask_b32_e32 v45, v52, v45, vcc
	v_cndmask_b32_e32 v43, v54, v43, vcc
	v_cndmask_b32_e32 v42, v55, v42, vcc
	v_cndmask_b32_e32 v52, v37, v44, vcc
	v_cndmask_b32_e32 v54, v35, v46, vcc
	v_cndmask_b32_e32 v55, v34, v47, vcc
	v_cndmask_b32_e32 v37, v44, v37, vcc
	v_cndmask_b32_e32 v35, v46, v35, vcc
	v_cndmask_b32_e32 v34, v47, v34, vcc
	v_cndmask_b32_e32 v44, v29, v36, vcc
	v_cndmask_b32_e32 v46, v27, v38, vcc
	v_cndmask_b32_e32 v47, v26, v39, vcc
	v_cndmask_b32_e32 v29, v36, v29, vcc
	v_cndmask_b32_e32 v27, v38, v27, vcc
	v_cndmask_b32_e32 v26, v39, v26, vcc
	v_cndmask_b32_e32 v36, v21, v28, vcc
	v_cndmask_b32_e32 v38, v19, v30, vcc
	v_cndmask_b32_e32 v39, v18, v31, vcc
	v_cndmask_b32_e32 v21, v28, v21, vcc
	v_cndmask_b32_e32 v19, v30, v19, vcc
	v_cndmask_b32_e32 v18, v31, v18, vcc
	v_cndmask_b32_e32 v28, v13, v20, vcc
	v_cndmask_b32_e32 v30, v11, v22, vcc
	v_cndmask_b32_e32 v31, v10, v23, vcc
	v_cndmask_b32_e32 v13, v20, v13, vcc
	v_cndmask_b32_e32 v11, v22, v11, vcc
	v_cndmask_b32_e32 v10, v23, v10, vcc
	v_cndmask_b32_e32 v20, v5, v12, vcc
	v_cndmask_b32_e32 v22, v3, v14, vcc
	v_cndmask_b32_e32 v23, v2, v15, vcc
	v_cndmask_b32_e32 v5, v12, v5, vcc
	v_cndmask_b32_e32 v12, v14, v3, vcc
	v_cndmask_b32_e32 v14, v15, v2, vcc
	v_perm_b32 v2, v59, v0, s4
	v_perm_b32 v0, v6, v4, s4
	v_add_lshl_u32 v4, v66, v65, 1
	v_bfrev_b32_e32 v6, 1
	s_waitcnt lgkmcnt(12)
	v_cndmask_b32_e32 v69, v41, v56, vcc
	v_cndmask_b32_e32 v41, v56, v41, vcc
	s_waitcnt lgkmcnt(10)
	v_cndmask_b32_e32 v56, v33, v48, vcc
	v_cndmask_b32_e32 v33, v48, v33, vcc
	s_waitcnt lgkmcnt(8)
	v_cndmask_b32_e32 v48, v25, v40, vcc
	v_cndmask_b32_e32 v25, v40, v25, vcc
	s_waitcnt lgkmcnt(6)
	v_cndmask_b32_e32 v40, v17, v32, vcc
	v_cndmask_b32_e32 v17, v32, v17, vcc
	s_waitcnt lgkmcnt(4)
	v_cndmask_b32_e32 v32, v9, v24, vcc
	v_cndmask_b32_e32 v9, v24, v9, vcc
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e32 v24, v1, v16, vcc
	v_cndmask_b32_e32 v15, v16, v1, vcc
	v_perm_b32 v3, v57, v58, s4
	v_perm_b32 v1, v8, v7, s4
	v_cndmask_b32_e64 v7, v6, v4, s[8:9]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e32 v68, v49, v71, vcc
	v_cndmask_b32_e32 v49, v71, v49, vcc
	buffer_store_dwordx4 v[0:3], v7, s[0:3], 0 offen
	v_add_u32_e32 v7, 32, v4
	v_cndmask_b32_e64 v7, v6, v7, s[8:9]
	v_perm_b32 v3, v49, v50, s4
	v_perm_b32 v2, v51, v53, s4
	v_perm_b32 v1, v68, v67, s4
	v_perm_b32 v0, v64, v62, s4
	buffer_store_dwordx4 v[0:3], v7, s[0:3], 0 offen
	v_add_u32_e32 v7, 64, v4
	v_cndmask_b32_e64 v7, v6, v7, s[8:9]
	v_perm_b32 v3, v41, v42, s4
	v_perm_b32 v2, v43, v45, s4
	v_perm_b32 v1, v69, v63, s4
	v_perm_b32 v0, v61, v60, s4
	buffer_store_dwordx4 v[0:3], v7, s[0:3], 0 offen
	v_add_u32_e32 v7, 0x60, v4
	v_cndmask_b32_e64 v7, v6, v7, s[8:9]
	v_perm_b32 v3, v33, v34, s4
	v_perm_b32 v2, v35, v37, s4
	v_perm_b32 v1, v56, v55, s4
	v_perm_b32 v0, v54, v52, s4
	buffer_store_dwordx4 v[0:3], v7, s[0:3], 0 offen
	v_add_u32_e32 v7, 0x80, v4
	v_cndmask_b32_e64 v7, v6, v7, s[8:9]
	v_perm_b32 v3, v25, v26, s4
	v_perm_b32 v2, v27, v29, s4
	v_perm_b32 v1, v48, v47, s4
	v_perm_b32 v0, v46, v44, s4
	buffer_store_dwordx4 v[0:3], v7, s[0:3], 0 offen
	v_add_u32_e32 v7, 0xa0, v4
	v_cndmask_b32_e64 v7, v6, v7, s[8:9]
	v_perm_b32 v3, v17, v18, s4
	v_perm_b32 v2, v19, v21, s4
	v_perm_b32 v1, v40, v39, s4
	v_perm_b32 v0, v38, v36, s4
	buffer_store_dwordx4 v[0:3], v7, s[0:3], 0 offen
	v_add_u32_e32 v7, 0xc0, v4
	v_cndmask_b32_e64 v7, v6, v7, s[8:9]
	v_perm_b32 v3, v9, v10, s4
	v_perm_b32 v2, v11, v13, s4
	v_perm_b32 v1, v32, v31, s4
	v_perm_b32 v0, v30, v28, s4
	v_add_u32_e32 v4, 0xe0, v4
	buffer_store_dwordx4 v[0:3], v7, s[0:3], 0 offen
	v_cndmask_b32_e64 v4, v6, v4, s[8:9]
	s_nop 0
	v_perm_b32 v3, v15, v14, s4
	v_perm_b32 v2, v12, v5, s4
	v_perm_b32 v1, v24, v23, s4
	v_perm_b32 v0, v22, v20, s4
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
.LBB0_62:                               ; %.critedge
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 280
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
		.amdhsa_next_free_sgpr 78
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
	.set attn_fwd.numbered_sgpr, 78
	.set attn_fwd.private_seg_size, 280
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 34944
; TotalNumSgprs: 84
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 280
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 10
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 84
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
	.short	523                             ; DW_AT_call_line
	.byte	41                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	530                             ; DW_AT_call_line
	.byte	89                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x5b:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	701                             ; DW_AT_call_line
	.byte	58                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x68:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
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
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
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
    .private_segment_fixed_size: 280
    .sgpr_count:     84
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 98
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
