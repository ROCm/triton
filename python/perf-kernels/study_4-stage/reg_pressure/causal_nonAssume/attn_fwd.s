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
; %bb.36:
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.37:
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
	s_cbranch_scc1 .LBB0_35
; %bb.1:
	s_add_u32 s20, s22, s24
	s_addc_u32 s21, s23, s25
	s_load_dwordx2 s[66:67], s[20:21], 0x0
	s_load_dwordx8 s[36:43], s[0:1], 0x38
	v_lshrrev_b32_e32 v234, 4, v0
	v_or_b32_e32 v9, s60, v234
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
	v_lshlrev_b32_e32 v235, 3, v0
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
	v_and_b32_e32 v10, 0x78, v235
	s_addc_u32 s16, s19, s47
	v_mad_u64_u32 v[10:11], s[46:47], s43, v234, v[10:11]
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
	s_cbranch_scc1 .LBB0_35
; %bb.4:
	s_ashr_i32 s16, s17, 31
	s_lshr_b32 s16, s16, 28
	s_add_i32 s16, s17, s16
	s_ashr_i32 s16, s16, 4
	s_and_b32 s19, s67, 63
	s_sub_i32 s20, 64, s67
	s_cmp_lt_i32 s67, 64
	s_cselect_b32 s41, s20, s19
	s_mul_i32 s20, s12, s18
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
	s_lshl_b32 s17, s14, 5
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s20, s12, s2
	v_and_b32_e32 v112, 0x78, v235
	s_addc_u32 s12, s13, s3
	v_mad_u64_u32 v[10:11], s[2:3], s14, v234, v[112:113]
	v_add_u32_e32 v11, s17, v10
	v_lshlrev_b32_e32 v10, 1, v10
	v_bfrev_b32_e32 v1, 1
	v_cmp_gt_i32_e32 vcc, s33, v9
	v_add_u32_e32 v16, s17, v11
	s_and_b32 s2, s14, 0x3fff
	v_cndmask_b32_e32 v18, v1, v10, vcc
	v_lshlrev_b32_e32 v9, 1, v11
	v_cmp_gt_i32_e32 vcc, s33, v8
	v_add_u32_e32 v17, s17, v16
	s_bitset1_b32 s2, 14
	v_cndmask_b32_e32 v19, v1, v9, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	v_cmp_gt_i32_e32 vcc, s33, v6
	s_and_b32 s3, s12, 0xffff
	s_lshl_b32 s2, s2, 16
	v_cndmask_b32_e32 v6, v1, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v17
	v_cmp_gt_i32_e32 vcc, s33, v4
	v_add_u32_e32 v24, s17, v17
	s_or_b32 s21, s3, s2
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v4, v1, v16, vcc
	v_add_u32_e32 v25, s17, v24
	buffer_load_dwordx4 v[8:11], v18, s[20:23], 0 offen
	buffer_load_dwordx4 v[12:15], v19, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[16:19], v6, s[20:23], 0 offen
	buffer_load_dwordx4 v[20:23], v4, s[20:23], 0 offen
	v_lshlrev_b32_e32 v4, 1, v24
	v_cmp_gt_i32_e32 vcc, s33, v7
	v_add_u32_e32 v32, s17, v25
	v_lshlrev_b32_e32 v6, 1, v25
	v_cndmask_b32_e32 v4, v1, v4, vcc
	v_cmp_gt_i32_e32 vcc, s33, v5
	v_lshrrev_b32_e32 v66, 1, v0
	s_movk_i32 s12, 0x78
	v_cndmask_b32_e32 v5, v1, v6, vcc
	buffer_load_dwordx4 v[24:27], v4, s[20:23], 0 offen
	buffer_load_dwordx4 v[28:31], v5, s[20:23], 0 offen
	v_lshlrev_b32_e32 v4, 1, v32
	v_cmp_gt_i32_e32 vcc, s33, v3
	v_lshlrev_b32_e32 v228, 7, v234
	v_or_b32_e32 v7, 0x1000, v228
	v_cndmask_b32_e32 v3, v1, v4, vcc
	v_add_lshl_u32 v4, v32, s17, 1
	v_cmp_gt_i32_e32 vcc, s33, v2
	v_lshrrev_b32_e32 v248, 2, v0
	v_mad_u64_u32 v[72:73], s[2:3], s37, v234, v[112:113]
	v_cndmask_b32_e32 v2, v1, v4, vcc
	buffer_load_dwordx4 v[32:35], v3, s[20:23], 0 offen
	buffer_load_dwordx4 v[36:39], v2, s[20:23], 0 offen
	v_lshrrev_b32_e32 v2, 3, v0
	v_and_b32_e32 v3, 0x78, v66
	v_and_b32_e32 v252, 4, v2
	v_bitop3_b32 v2, v66, v235, s12 bitop3:0x28
	v_bitop3_b32 v3, v3, v228, v112 bitop3:0xde
	v_lshlrev_b32_e32 v6, 1, v3
	v_or_b32_e32 v3, v7, v2
	v_lshlrev_b32_e32 v5, 1, v3
	v_lshlrev_b32_e32 v3, 1, v2
	v_lshlrev_b32_e32 v4, 8, v234
	v_add_u32_e32 v249, 0, v6
	v_add_u32_e32 v227, 0, v5
	v_add3_u32 v3, 0, v3, v4
	s_barrier
	v_lshrrev_b32_e32 v4, 5, v0
	v_mad_u64_u32 v[70:71], s[2:3], s40, v234, v[112:113]
	v_and_b32_e32 v98, 31, v0
	s_movk_i32 s2, 0xe0
	s_mul_i32 s24, s15, s18
	s_mul_i32 s26, s36, s16
	s_mul_i32 s28, s66, s37
	s_mul_i32 s30, s38, s18
	s_mul_i32 s34, s39, s16
	s_mul_i32 s38, s66, s40
	s_and_b32 s3, s33, 0xff
	s_ashr_i32 s25, s24, 31
	s_ashr_i32 s27, s26, 31
	s_ashr_i32 s29, s28, 31
	s_ashr_i32 s31, s30, 31
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s39, s38, 31
	s_or_b32 s3, s41, s3
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s3, 4, 5
	v_and_b32_e32 v238, 32, v235
	v_and_b32_e32 v236, 64, v235
	v_lshl_add_u32 v230, s37, 5, v72
	v_lshl_add_u32 v206, s40, 5, v70
	s_cmp_le_u32 s69, s3
	v_lshlrev_b32_e32 v85, 8, v98
	v_lshlrev_b32_e32 v86, 1, v0
	s_waitcnt vmcnt(7)
	ds_write_b128 v249, v[8:11]
	s_waitcnt vmcnt(6)
	ds_write_b128 v227, v[12:15]
	s_waitcnt vmcnt(5)
	ds_write_b128 v3, v[16:19] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v3, v[20:23] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v3, v[24:27] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v3, v[28:31] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v3, v[32:35] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v3, v[36:39] offset:57344
	v_and_b32_e32 v3, 8, v248
	v_or_b32_e32 v9, 16, v3
	v_or_b32_e32 v10, 32, v3
	v_or_b32_e32 v11, 48, v3
	v_or_b32_e32 v12, 64, v3
	v_or_b32_e32 v13, 0x50, v3
	v_or_b32_e32 v14, 0x60, v3
	v_or_b32_e32 v15, 0x70, v3
	v_and_b32_e32 v3, 15, v0
	v_bitop3_b32 v4, v4, v3, 1 bitop3:0x6c
	v_lshrrev_b32_e32 v3, 3, v9
	v_lshrrev_b32_e32 v9, 3, v10
	v_bitop3_b32 v23, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v11
	v_bitop3_b32 v22, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v12
	v_bitop3_b32 v21, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v13
	v_and_or_b32 v8, v66, s2, v98
	v_bitop3_b32 v20, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v14
	v_bitop3_b32 v3, v3, v0, 15 bitop3:0x78
	v_bitop3_b32 v19, v9, v0, 15 bitop3:0x78
	v_lshrrev_b32_e32 v9, 3, v15
	v_lshl_add_u32 v8, v8, 8, 0
	v_lshlrev_b32_e32 v71, 4, v4
	v_bitop3_b32 v18, v9, v0, 15 bitop3:0x78
	v_add_u32_e32 v9, v8, v71
	v_lshlrev_b32_e32 v73, 4, v3
	v_lshlrev_b32_e32 v74, 4, v23
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v10, v8, v73
	ds_read_b128 v[130:133], v9
	ds_read_b128 v[134:137], v10
	v_add_u32_e32 v9, v8, v74
	v_lshlrev_b32_e32 v75, 4, v22
	v_lshlrev_b32_e32 v76, 4, v21
	v_lshlrev_b32_e32 v77, 4, v20
	v_lshlrev_b32_e32 v78, 4, v19
	v_lshlrev_b32_e32 v79, 4, v18
	v_add_u32_e32 v10, v8, v75
	v_add_u32_e32 v11, v8, v76
	v_add_u32_e32 v12, v8, v77
	v_add_u32_e32 v13, v8, v78
	v_add_u32_e32 v8, v8, v79
	ds_read_b128 v[138:141], v9
	ds_read_b128 v[142:145], v10
	ds_read_b128 v[146:149], v11
	ds_read_b128 v[150:153], v12
	ds_read_b128 v[154:157], v13
	ds_read_b128 v[158:161], v8
	v_mov_b32_e32 v14, s3
	v_sub_u32_e64 v14, s69, v14 clamp
	s_movk_i32 s2, 0x60
	v_readfirstlane_b32 s36, v14
	s_cbranch_scc1 .LBB0_11
; %bb.5:
	v_or_b32_e32 v99, v228, v112
	v_lshlrev_b32_e32 v207, 1, v99
	v_or_b32_e32 v7, v7, v112
	v_sub_u32_e32 v6, v6, v207
	s_lshl_b64 s[12:13], s[24:25], 1
	v_lshlrev_b32_e32 v210, 1, v7
	v_ashrrev_i16_e32 v7, 15, v6
	s_add_u32 s3, s4, s12
	v_lshrrev_b16_e32 v7, 12, v7
	s_addc_u32 s14, s5, s13
	s_lshl_b64 s[12:13], s[26:27], 1
	v_add_u16_e32 v6, v6, v7
	s_add_u32 s3, s3, s12
	v_and_b32_e32 v226, 63, v0
	v_ashrrev_i16_e32 v6, 4, v6
	s_addc_u32 s14, s14, s13
	s_lshl_b64 s[12:13], s[28:29], 1
	v_add_u32_sdwa v64, v226, sext(v6) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_add_u32 s20, s3, s12
	v_lshlrev_b32_e32 v6, 2, v64
	s_addc_u32 s3, s14, s13
	s_lshl_b64 s[12:13], s[30:31], 1
	ds_bpermute_b32 v8, v6, v72
	v_lshrrev_b64 v[6:7], v64, exec
	v_sub_u32_e32 v5, v5, v210
	s_add_u32 s14, s6, s12
	v_ashrrev_i16_e32 v7, 15, v5
	s_addc_u32 s15, s7, s13
	s_lshl_b64 s[12:13], s[34:35], 1
	v_lshrrev_b16_e32 v7, 12, v7
	s_add_u32 s14, s14, s12
	v_add_u16_e32 v5, v5, v7
	s_addc_u32 s15, s15, s13
	s_lshl_b64 s[12:13], s[38:39], 1
	v_ashrrev_i16_e32 v5, 4, v5
	s_add_u32 s18, s14, s12
	v_add_u32_sdwa v66, v226, sext(v5) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_addc_u32 s19, s15, s13
	s_and_b32 s12, s37, 0x3fff
	v_lshlrev_b32_e32 v5, 2, v66
	v_add_u32_e32 v62, 0, v207
	s_bitset1_b32 s12, 14
	v_and_b32_e32 v6, 1, v6
	ds_bpermute_b32 v5, v5, v230
	s_and_b32 s13, s3, 0xffff
	s_lshl_b32 s49, s12, 16
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v65, 1, v8
	v_cmp_eq_u32_e32 vcc, 1, v6
	v_readfirstlane_b32 s48, v62
	v_sub_u32_e32 v2, v2, v112
	scratch_store_dword off, v79, off offset:88 ; 4-byte Folded Spill
	scratch_store_dword off, v78, off offset:84 ; 4-byte Folded Spill
	scratch_store_dword off, v77, off       ; 4-byte Folded Spill
	scratch_store_dword off, v76, off offset:80 ; 4-byte Folded Spill
	scratch_store_dword off, v75, off offset:76 ; 4-byte Folded Spill
	scratch_store_dword off, v74, off offset:72 ; 4-byte Folded Spill
	scratch_store_dword off, v73, off offset:68 ; 4-byte Folded Spill
	scratch_store_dword off, v71, off offset:64 ; 4-byte Folded Spill
	s_or_b32 s21, s13, s49
	v_cndmask_b32_e32 v6, v1, v65, vcc
	s_mov_b32 m0, s48
	v_ashrrev_i32_e32 v2, 3, v2
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_lshrrev_b64 v[6:7], v66, exec
	v_add_u32_e32 v2, v2, v226
	s_lshl_b32 s14, s37, 6
	v_add_u32_e32 v63, 0, v210
	v_and_b32_e32 v6, 1, v6
	v_lshlrev_b32_e32 v242, 2, v2
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v67, 1, v5
	v_cmp_eq_u32_e32 vcc, 1, v6
	v_readfirstlane_b32 s50, v63
	s_ashr_i32 s15, s14, 31
	v_mov_b32_e32 v6, v72
	ds_bpermute_b32 v9, v242, v72
	v_cmp_ne_u32_e64 s[12:13], s36, 1
	s_lshl_b32 s16, s40, 6
	v_cndmask_b32_e32 v5, v1, v67, vcc
	s_mov_b32 m0, s50
	s_lshl_b64 s[44:45], s[14:15], 1
	scratch_store_dwordx2 off, v[6:7], off offset:92 ; 8-byte Folded Spill
	v_lshrrev_b64 v[6:7], v2, s[12:13]
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_u32 s20, s20, s44
	ds_bpermute_b32 v7, v242, v230
	s_addc_u32 s15, s3, s45
	v_add_u32_e32 v5, 0x4000, v62
	s_and_b32 s3, s15, 0xffff
	v_and_b32_e32 v2, 1, v6
	s_or_b32 s21, s3, s49
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v6, 1, v9
	v_cmp_eq_u32_e32 vcc, 1, v2
	v_readfirstlane_b32 s3, v5
	v_add_u32_e32 v8, 0x4000, v63
	v_cndmask_b32_e32 v2, v1, v6, vcc
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v8
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v7
	v_lshl_or_b32 v72, v4, 4, v85
	v_cndmask_b32_e32 v2, v1, v2, vcc
	s_mov_b32 m0, s3
	v_add_u32_e32 v68, 0, v72
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b128 v[4:7], v68
	ds_read_b128 v[24:27], v68 offset:8192
	v_lshl_or_b32 v73, v3, 4, v85
	v_add_u32_e32 v74, 0, v73
	ds_read_b128 v[28:31], v74
	ds_read_b128 v[34:37], v74 offset:8192
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[4:7], v[130:133], 0
	v_lshl_or_b32 v75, v23, 4, v85
	v_add_u32_e32 v76, 0, v75
	v_lshl_or_b32 v77, v22, 4, v85
	v_add_u32_e32 v78, 0, v77
	ds_read_b128 v[38:41], v76 offset:8192
	v_lshl_or_b32 v79, v21, 4, v85
	v_add_u32_e32 v80, 0, v79
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[134:137], v[2:17]
	ds_read_b128 v[28:31], v76
	ds_read_b128 v[42:45], v78 offset:8192
	v_lshl_or_b32 v81, v20, 4, v85
	v_add_u32_e32 v82, 0, v81
	ds_read_b128 v[20:23], v82
	ds_read_b128 v[50:53], v82 offset:8192
	ds_read_b128 v[46:49], v80 offset:8192
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[138:141], v[2:17]
	ds_read_b128 v[28:31], v78
	v_lshl_or_b32 v83, v19, 4, v85
	v_add_u32_e32 v84, 0, v83
	ds_read_b128 v[54:57], v84 offset:8192
	scratch_store_dword off, v85, off offset:108 ; 4-byte Folded Spill
	v_lshl_or_b32 v85, v18, 4, v85
	v_add_u32_e32 v90, 0, v85
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[142:145], v[2:17]
	ds_read_b128 v[28:31], v80
	v_bitop3_b32 v100, v112, v86, s2 bitop3:0x78
	s_and_b32 s2, s40, 0x3fff
	s_bitset1_b32 s2, 14
	s_and_b32 s3, s19, 0xffff
	s_lshl_b32 s42, s2, 16
	s_or_b32 s73, s3, s42
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[28:31], v[146:149], v[2:17]
	s_mov_b32 s72, s18
	s_mov_b32 s74, s22
	s_mov_b32 s75, s23
	s_cmp_gt_u32 s36, 2
	v_cmp_gt_u32_e64 s[70:71], s36, 2
	v_mfma_f32_32x32x16_f16 v[2:17], v[20:23], v[150:153], v[2:17]
	ds_read_b128 v[20:23], v84
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[20:23], v[154:157], v[2:17]
	ds_read_b128 v[18:21], v90
	ds_read_b128 v[58:61], v90 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[158:161], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[24:27], v[130:133], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[34:37], v[134:137], v[18:33]
	v_sub_u32_e32 v34, v100, v112
	v_ashrrev_i32_e32 v34, 3, v34
	v_add_u32_e32 v36, 0x8000, v62
	v_add_u32_e32 v37, 0x8000, v63
	v_readfirstlane_b32 s2, v36
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v37
	v_mfma_f32_32x32x16_f16 v[18:33], v[38:41], v[138:141], v[18:33]
	v_add_u32_e32 v41, v34, v226
	v_lshlrev_b32_e32 v39, 2, v41
	v_mov_b32_e32 v34, v70
	ds_bpermute_b32 v38, v39, v70
	scratch_store_dwordx2 off, v[34:35], off offset:56 ; 8-byte Folded Spill
	v_lshrrev_b64 v[34:35], v41, exec
	ds_bpermute_b32 v35, v39, v206
	v_mfma_f32_32x32x16_f16 v[18:33], v[42:45], v[142:145], v[18:33]
	v_and_b32_e32 v34, 1, v34
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v38, 1, v38
	v_cmp_eq_u32_e32 vcc, 1, v34
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v36, 1, v35
	v_add_u32_e32 v39, 0xc000, v62
	v_cndmask_b32_e32 v34, v1, v38, vcc
	buffer_load_dwordx4 v34, s[72:75], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[18:33], v[46:49], v[146:149], v[18:33]
	s_mov_b32 m0, s2
	s_cselect_b64 s[2:3], -1, 0
	s_add_u32 s20, s20, s44
	v_cndmask_b32_e32 v34, v1, v36, vcc
	s_addc_u32 s15, s15, s45
	s_ashr_i32 s17, s16, 31
	buffer_load_dwordx4 v34, s[72:75], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[18:33], v[50:53], v[150:153], v[18:33]
	s_lshl_b64 s[16:17], s[16:17], 1
	v_lshrrev_b64 v[34:35], v64, s[70:71]
	s_add_u32 s46, s18, s16
	v_and_b32_e32 v34, 1, v34
	s_addc_u32 s47, s19, s17
	s_and_b32 s15, s15, 0xffff
	v_cmp_eq_u32_e32 vcc, 1, v34
	v_mfma_f32_32x32x16_f16 v[18:33], v[54:57], v[154:157], v[18:33]
	s_or_b32 s21, s15, s49
	v_cndmask_b32_e32 v34, v1, v65, vcc
	s_mov_b32 m0, s48
	v_add_u32_e32 v40, 0xc000, v63
	buffer_load_dwordx4 v34, s[20:23], 0 offen lds
	v_lshrrev_b64 v[34:35], v66, s[70:71]
	v_and_b32_e32 v34, 1, v34
	v_cmp_eq_u32_e32 vcc, 1, v34
	s_mov_b32 m0, s50
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v35, v2, v2
	v_cndmask_b32_e32 v34, v1, v67, vcc
	buffer_load_dwordx4 v34, s[20:23], 0 offen lds
	v_max_f32_e32 v34, v3, v3
	v_mfma_f32_32x32x16_f16 v[18:33], v[58:61], v[158:161], v[18:33]
	v_max_f32_e32 v34, v35, v34
	v_max3_f32 v34, v34, v4, v5
	v_max3_f32 v34, v34, v6, v7
	v_max3_f32 v34, v34, v8, v9
	v_max3_f32 v34, v34, v10, v11
	v_max3_f32 v34, v34, v12, v13
	v_max3_f32 v34, v34, v14, v15
	v_max3_f32 v34, v34, v16, v17
	s_nop 3
	v_max3_f32 v37, v34, v18, v19
	v_lshrrev_b64 v[34:35], v41, s[12:13]
	v_and_b32_e32 v34, 1, v34
	s_and_b32 s15, s47, 0xffff
	v_cmp_eq_u32_e32 vcc, 1, v34
	v_readfirstlane_b32 s12, v39
	s_or_b32 s21, s15, s42
	s_mov_b32 s20, s46
	v_cndmask_b32_e32 v34, v1, v38, vcc
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v40
	s_waitcnt vmcnt(4)
	s_barrier
	buffer_load_dwordx4 v34, s[20:23], 0 offen lds
	v_cndmask_b32_e32 v1, v1, v36, vcc
	s_mov_b32 m0, s12
	v_mov_b32_e32 v34, 0xff800000
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	v_max3_f32 v1, v37, v20, v21
	v_max3_f32 v1, v1, v22, v23
	v_max3_f32 v1, v1, v24, v25
	v_max3_f32 v1, v1, v26, v27
	v_max3_f32 v1, v1, v28, v29
	v_max3_f32 v1, v1, v30, v31
	v_max3_f32 v1, v1, v32, v33
	v_mov_b32_e32 v35, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v35
	v_max3_f32 v229, v1, v35, v34
	scratch_store_dword off, v41, off offset:112 ; 4-byte Folded Spill
	v_mul_f32_e32 v1, 0xbe0293ee, v229
	s_add_i32 s12, 0, 0x4000
	v_fmamk_f32 v2, v2, 0x3e0293ee, v1
	v_fmamk_f32 v3, v3, 0x3e0293ee, v1
	v_fmamk_f32 v4, v4, 0x3e0293ee, v1
	v_fmamk_f32 v5, v5, 0x3e0293ee, v1
	v_fmamk_f32 v6, v6, 0x3e0293ee, v1
	v_fmamk_f32 v7, v7, 0x3e0293ee, v1
	v_fmamk_f32 v8, v8, 0x3e0293ee, v1
	v_fmamk_f32 v9, v9, 0x3e0293ee, v1
	v_fmamk_f32 v10, v10, 0x3e0293ee, v1
	v_fmamk_f32 v11, v11, 0x3e0293ee, v1
	v_fmamk_f32 v12, v12, 0x3e0293ee, v1
	v_fmamk_f32 v13, v13, 0x3e0293ee, v1
	v_fmamk_f32 v14, v14, 0x3e0293ee, v1
	v_fmamk_f32 v15, v15, 0x3e0293ee, v1
	v_fmamk_f32 v16, v16, 0x3e0293ee, v1
	v_fmamk_f32 v17, v17, 0x3e0293ee, v1
	v_fmamk_f32 v18, v18, 0x3e0293ee, v1
	v_fmamk_f32 v19, v19, 0x3e0293ee, v1
	v_fmamk_f32 v20, v20, 0x3e0293ee, v1
	v_fmamk_f32 v21, v21, 0x3e0293ee, v1
	v_fmamk_f32 v22, v22, 0x3e0293ee, v1
	v_fmamk_f32 v23, v23, 0x3e0293ee, v1
	v_fmamk_f32 v24, v24, 0x3e0293ee, v1
	v_fmamk_f32 v25, v25, 0x3e0293ee, v1
	v_fmamk_f32 v26, v26, 0x3e0293ee, v1
	v_fmamk_f32 v27, v27, 0x3e0293ee, v1
	v_fmamk_f32 v28, v28, 0x3e0293ee, v1
	v_fmamk_f32 v29, v29, 0x3e0293ee, v1
	v_fmamk_f32 v30, v30, 0x3e0293ee, v1
	v_fmamk_f32 v31, v31, 0x3e0293ee, v1
	v_fmamk_f32 v32, v32, 0x3e0293ee, v1
	v_fmac_f32_e32 v1, 0x3e0293ee, v33
	ds_read_b128 v[68:71], v68 offset:16384
	ds_read_b128 v[202:205], v74 offset:16384
	ds_read_b128 v[198:201], v76 offset:16384
	ds_read_b128 v[194:197], v78 offset:16384
	ds_read_b128 v[94:97], v80 offset:16384
	ds_read_b128 v[86:89], v82 offset:16384
	v_add_u32_e32 v33, s12, v72
	v_add_u32_e32 v35, s12, v73
	v_add_u32_e32 v36, s12, v75
	v_add_u32_e32 v37, s12, v77
	v_add_u32_e32 v38, s12, v79
	v_add_u32_e32 v39, s12, v81
	v_add_u32_e32 v40, s12, v83
	v_add_u32_e32 v41, s12, v85
	ds_read_b128 v[190:193], v84 offset:16384
	ds_read_b128 v[90:93], v90 offset:16384
	ds_read_b128 v[82:85], v33 offset:8192
	ds_read_b128 v[186:189], v35 offset:8192
	ds_read_b128 v[182:185], v36 offset:8192
	ds_read_b128 v[178:181], v37 offset:8192
	ds_read_b128 v[174:177], v38 offset:8192
	ds_read_b128 v[170:173], v39 offset:8192
	ds_read_b128 v[166:169], v40 offset:8192
	ds_read_b128 v[162:165], v41 offset:8192
	v_add_u32_e32 v42, 0xff, v0
	s_movk_i32 s12, 0x1ff
	v_cmp_gt_u32_e32 vcc, s12, v42
	s_movk_i32 s12, 0x1fe
	v_fmac_f32_e32 v34, 0xbe0293ee, v229
	v_cmp_lt_u32_e64 s[12:13], s12, v42
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], s[12:13]
	s_cbranch_execz .LBB0_7
; %bb.6:
	s_barrier
.LBB0_7:
	s_or_b64 exec, exec, s[20:21]
	v_exp_f32_e32 v251, v2
	v_exp_f32_e32 v231, v3
	v_exp_f32_e32 v237, v4
	v_exp_f32_e32 v243, v5
	v_exp_f32_e32 v244, v6
	v_exp_f32_e32 v245, v7
	v_exp_f32_e32 v246, v8
	v_exp_f32_e32 v247, v9
	v_exp_f32_e32 v239, v10
	v_exp_f32_e32 v240, v11
	v_exp_f32_e32 v208, v12
	v_exp_f32_e32 v209, v13
	v_exp_f32_e32 v241, v14
	v_exp_f32_e32 v233, v15
	v_exp_f32_e32 v213, v16
	v_exp_f32_e32 v215, v17
	v_exp_f32_e32 v214, v18
	v_exp_f32_e32 v217, v19
	v_exp_f32_e32 v216, v20
	v_exp_f32_e32 v221, v21
	v_exp_f32_e32 v220, v22
	v_exp_f32_e32 v219, v23
	v_exp_f32_e32 v218, v24
	v_exp_f32_e32 v223, v25
	v_exp_f32_e32 v222, v26
	v_exp_f32_e32 v224, v27
	v_exp_f32_e32 v253, v28
	v_exp_f32_e32 v250, v29
	v_exp_f32_e32 v225, v30
	v_exp_f32_e32 v255, v31
	v_exp_f32_e32 v254, v32
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v212, v34
	v_mov_b32_e32 v17, 0
	s_cmp_lt_u32 s36, 4
	v_lshlrev_b32_e32 v2, 7, v98
	scratch_store_dword off, v2, off offset:8 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[112:113], off offset:100 ; 8-byte Folded Spill
	scratch_store_dword off, v228, off offset:4 ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_12
; %bb.8:                                ; %.lr.ph
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v5, 12, v2
	v_or_b32_e32 v2, v238, v5
	v_and_or_b32 v3, v248, 3, v252
	v_bitop3_b32 v2, v2, v236, 64 bitop3:0x36
	s_movk_i32 s12, 0x60
	scratch_store_dword off, v227, off offset:124 ; 4-byte Folded Spill
	scratch_store_dword off, v249, off offset:120 ; 4-byte Folded Spill
	scratch_store_dword off, v252, off offset:116 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v3, 7, v3
	v_bitop3_b32 v4, v5, v238, 32 bitop3:0x36
	scratch_store_dword off, v206, off offset:20 ; 4-byte Folded Spill
	scratch_store_dword off, v230, off offset:12 ; 4-byte Folded Spill
	scratch_store_dword off, v238, off offset:16 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off offset:40 ; 4-byte Folded Spill
	v_bitop3_b32 v2, v235, v5, s12 bitop3:0x4e
	scratch_store_dword off, v2, off offset:44 ; 4-byte Folded Spill
	v_add_u32_e32 v2, v228, v112
	scratch_store_dword off, v236, off offset:24 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v236, 1, v3
	v_lshlrev_b32_e32 v3, 1, v4
	scratch_store_dword off, v5, off offset:36 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off offset:48 ; 4-byte Folded Spill
	scratch_store_dword off, v3, off offset:52 ; 4-byte Folded Spill
	scratch_load_dwordx2 v[66:67], off, off offset:56 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[234:235], off, off offset:92 ; 8-byte Folded Reload
	scratch_load_dword v232, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v230, off, off offset:68 ; 4-byte Folded Reload
	s_add_i32 s20, s36, -3
	scratch_store_dword off, v210, off offset:28 ; 4-byte Folded Spill
	scratch_load_dword v210, off, off offset:72 ; 4-byte Folded Reload
	s_nop 0
	scratch_load_dword v238, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v248, off, off offset:80 ; 4-byte Folded Reload
	s_add_u32 s12, s24, s26
	scratch_store_dword off, v207, off offset:32 ; 4-byte Folded Spill
	scratch_load_dword v227, off, off offset:84 ; 4-byte Folded Reload
	scratch_load_dword v228, off, off offset:88 ; 4-byte Folded Reload
	s_addc_u32 s13, s25, s27
	s_add_u32 s12, s12, s28
	s_addc_u32 s13, s13, s29
	s_mul_hi_i32 s15, s14, 6
	s_mul_i32 s14, s14, 6
	s_lshl_b64 s[12:13], s[12:13], 1
	s_add_u32 s12, s14, s12
	s_addc_u32 s13, s15, s13
	s_add_u32 s21, s4, s12
	v_mov_b32_e32 v2, 0
	s_addc_u32 s51, s5, s13
	s_mov_b32 s22, 0
	s_add_i32 s23, 0, 0x8000
	s_add_i32 s48, 0, 0xc000
	v_mov_b32_e32 v211, 1.0
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_mov_b32 s53, 0x3e0293ee
	s_mov_b32 s57, 0
	s_mov_b32 s55, 0
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
	s_waitcnt vmcnt(10)
	v_mov_b32_e32 v206, v66
	s_waitcnt vmcnt(9)
	v_and_b32_e32 v235, 16, v0
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[68:71], v[130:133], 0
	s_mov_b64 s[18:19], s[46:47]
	s_mov_b32 s63, s23
	s_mov_b32 s23, s48
	v_mov_b32_e32 v98, v211
	v_mov_b32_e32 v252, v229
	s_mov_b32 s59, s22
	s_setprio 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[202:205], v[134:137], v[66:81]
	v_mov_b32_e32 v249, v99
	v_add_f32_e32 v99, v251, v231
	v_add_f32_e32 v99, v99, v237
	v_add_f32_e32 v99, v99, v243
	v_add_f32_e32 v99, v99, v244
	v_add_f32_e32 v99, v99, v245
	v_add_f32_e32 v99, v99, v246
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[198:201], v[138:141], v[66:81]
	v_add_f32_e32 v99, v99, v247
	v_add_f32_e32 v99, v99, v239
	v_add_f32_e32 v99, v99, v240
	v_add_f32_e32 v99, v99, v208
	v_add_f32_e32 v99, v99, v209
	v_add_f32_e32 v99, v99, v241
	v_add_f32_e32 v99, v99, v233
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[194:197], v[142:145], v[66:81]
	v_add_f32_e32 v99, v99, v213
	v_add_f32_e32 v99, v99, v215
	v_add_f32_e32 v99, v99, v214
	v_add_f32_e32 v99, v99, v217
	v_add_f32_e32 v99, v99, v216
	v_add_f32_e32 v99, v99, v221
	v_add_f32_e32 v99, v99, v220
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[146:149], v[66:81]
	v_add_f32_e32 v99, v99, v219
	v_add_f32_e32 v99, v99, v218
	v_add_f32_e32 v99, v99, v223
	v_add_f32_e32 v99, v99, v222
	v_add_f32_e32 v99, v99, v224
	v_add_f32_e32 v99, v99, v253
	v_add_f32_e32 v99, v99, v250
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[150:153], v[66:81]
	v_add_f32_e32 v94, v99, v225
	v_add_f32_e32 v94, v94, v255
	v_add_f32_e32 v94, v94, v254
	v_add_f32_e32 v94, v94, v1
	v_mov_b32_e32 v95, v94
	s_nop 1
	v_permlane32_swap_b32_e32 v94, v95
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[190:193], v[154:157], v[66:81]
	v_add_f32_e32 v211, v94, v95
	v_mul_f32_e32 v2, v2, v212
	v_mul_f32_e32 v3, v3, v212
	v_mul_f32_e32 v4, v4, v212
	v_mul_f32_e32 v5, v5, v212
	v_mul_f32_e32 v6, v6, v212
	v_mul_f32_e32 v7, v7, v212
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[158:161], v[66:81]
	v_mul_f32_e32 v8, v8, v212
	v_mul_f32_e32 v9, v9, v212
	v_mul_f32_e32 v10, v10, v212
	v_mul_f32_e32 v11, v11, v212
	v_mul_f32_e32 v12, v12, v212
	v_mul_f32_e32 v13, v13, v212
	v_mul_f32_e32 v14, v14, v212
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[130:133], 0
	v_mul_f32_e32 v15, v15, v212
	v_mul_f32_e32 v16, v16, v212
	v_mul_f32_e32 v17, v17, v212
	v_mul_f32_e32 v18, v18, v212
	v_mul_f32_e32 v19, v19, v212
	v_mul_f32_e32 v20, v20, v212
	v_mul_f32_e32 v21, v21, v212
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[134:137], v[82:97]
	v_mul_f32_e32 v22, v22, v212
	v_mul_f32_e32 v23, v23, v212
	v_mul_f32_e32 v24, v24, v212
	v_mul_f32_e32 v25, v25, v212
	v_mul_f32_e32 v26, v26, v212
	v_mul_f32_e32 v27, v27, v212
	v_mul_f32_e32 v28, v28, v212
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[138:141], v[82:97]
	v_mul_f32_e32 v29, v29, v212
	v_mul_f32_e32 v30, v30, v212
	v_mul_f32_e32 v31, v31, v212
	v_mul_f32_e32 v32, v32, v212
	v_mul_f32_e32 v33, v33, v212
	v_mul_f32_e32 v34, v34, v212
	v_mul_f32_e32 v35, v35, v212
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[142:145], v[82:97]
	v_mul_f32_e32 v36, v36, v212
	v_mul_f32_e32 v37, v37, v212
	v_mul_f32_e32 v38, v38, v212
	v_mul_f32_e32 v39, v39, v212
	v_mul_f32_e32 v40, v40, v212
	v_mul_f32_e32 v41, v41, v212
	v_mul_f32_e32 v42, v42, v212
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[146:149], v[82:97]
	v_mul_f32_e32 v43, v43, v212
	v_mul_f32_e32 v44, v44, v212
	v_mul_f32_e32 v45, v45, v212
	v_mul_f32_e32 v46, v46, v212
	v_mul_f32_e32 v47, v47, v212
	v_mul_f32_e32 v48, v48, v212
	v_mul_f32_e32 v49, v49, v212
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[150:153], v[82:97]
	v_mul_f32_e32 v50, v50, v212
	v_mul_f32_e32 v51, v51, v212
	v_mul_f32_e32 v52, v52, v212
	v_mul_f32_e32 v53, v53, v212
	v_mul_f32_e32 v54, v54, v212
	v_mul_f32_e32 v55, v55, v212
	v_mul_f32_e32 v56, v56, v212
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[154:157], v[82:97]
	v_mul_f32_e32 v57, v57, v212
	v_mul_f32_e32 v58, v58, v212
	v_mul_f32_e32 v59, v59, v212
	v_mul_f32_e32 v60, v60, v212
	v_mul_f32_e32 v61, v61, v212
	v_mul_f32_e32 v62, v62, v212
	v_mul_f32_e32 v63, v63, v212
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[158:161], v[82:97]
	v_mul_f32_e32 v64, v64, v212
	v_mul_f32_e32 v65, v65, v212
	v_fmac_f32_e32 v211, v98, v212
	v_cvt_pk_f16_f32 v99, v253, v250
	v_mov_b32_e32 v207, v100
	v_cvt_pk_f16_f32 v100, v225, v255
	v_cvt_pk_f16_f32 v110, v251, v231
	v_cvt_pk_f16_f32 v111, v237, v243
	v_cvt_pk_f16_f32 v112, v244, v245
	v_cvt_pk_f16_f32 v113, v246, v247
	v_cvt_pk_f16_f32 v106, v239, v240
	v_cvt_pk_f16_f32 v107, v208, v209
	v_cvt_pk_f16_f32 v108, v241, v233
	v_cvt_pk_f16_f32 v109, v213, v215
	v_cvt_pk_f16_f32 v102, v214, v217
	v_cvt_pk_f16_f32 v103, v216, v221
	v_cvt_pk_f16_f32 v104, v220, v219
	v_cvt_pk_f16_f32 v105, v218, v223
	v_cvt_pk_f16_f32 v98, v222, v224
	v_cvt_pk_f16_f32 v101, v254, v1
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v114, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v116, off, off offset:12 ; 4-byte Folded Reload
	s_add_u32 s46, s18, s16
	s_addc_u32 s47, s19, s17
	s_add_i32 s12, s57, 1
	s_cmp_lt_i32 s12, 2
	s_cselect_b32 s61, s12, 0
	ds_bpermute_b32 v115, v242, v234
	s_lshl_b32 s50, s61, 14
	s_add_i32 s22, s50, 0
	s_and_b32 s12, s51, 0xffff
	s_or_b32 s13, s12, s49
	s_mov_b32 s12, s21
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v115, 1, v115
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v1, s22, v1
	s_nop 0
	v_readfirstlane_b32 s48, v1
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v114, s22, v114
	s_mov_b32 m0, s48
	v_readfirstlane_b32 s48, v114
	scratch_load_dword v114, off, off offset:16 ; 4-byte Folded Reload
	s_nop 0
	buffer_load_dwordx4 v115, s[12:15], 0 offen lds
	scratch_load_dword v115, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(3)
	ds_bpermute_b32 v116, v242, v116
	s_mov_b32 m0, s48
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v116
	buffer_load_dwordx4 v1, s[12:15], 0 offen lds
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v114, 1, v114
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v115, 1, v115
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v1, 1, s63
	v_add3_u32 v1, v1, v114, v115
	v_lshlrev_b32_e32 v114, 1, v235
	v_add3_u32 v1, v1, v114, v236
	ds_read_b64_tr_b16 v[198:199], v1
	ds_read_b64_tr_b16 v[200:201], v1 offset:2048
	ds_read_b64_tr_b16 v[202:203], v1 offset:4096
	ds_read_b64_tr_b16 v[204:205], v1 offset:6144
	ds_read_b64_tr_b16 v[212:213], v1 offset:8192
	ds_read_b64_tr_b16 v[214:215], v1 offset:10240
	ds_read_b64_tr_b16 v[194:195], v1 offset:12288
	ds_read_b64_tr_b16 v[196:197], v1 offset:14336
	scratch_load_dword v1, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add3_u32 v1, s63, v1, v115
	v_add3_u32 v1, v1, v114, v236
	ds_read_b64_tr_b16 v[190:191], v1
	ds_read_b64_tr_b16 v[192:193], v1 offset:2048
	ds_read_b64_tr_b16 v[186:187], v1 offset:4096
	ds_read_b64_tr_b16 v[188:189], v1 offset:6144
	ds_read_b64_tr_b16 v[182:183], v1 offset:8192
	ds_read_b64_tr_b16 v[184:185], v1 offset:10240
	ds_read_b64_tr_b16 v[178:179], v1 offset:12288
	ds_read_b64_tr_b16 v[180:181], v1 offset:14336
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v1, 1, s63
	v_add3_u32 v1, v1, v114, v236
	ds_read_b64_tr_b16 v[174:175], v1
	ds_read_b64_tr_b16 v[176:177], v1 offset:2048
	ds_read_b64_tr_b16 v[170:171], v1 offset:4096
	ds_read_b64_tr_b16 v[172:173], v1 offset:6144
	ds_read_b64_tr_b16 v[166:167], v1 offset:8192
	ds_read_b64_tr_b16 v[168:169], v1 offset:10240
	ds_read_b64_tr_b16 v[162:163], v1 offset:12288
	ds_read_b64_tr_b16 v[164:165], v1 offset:14336
	scratch_load_dword v1, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v1, 1, s63
	v_add3_u32 v1, v1, v114, v236
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
	v_max3_f32 v229, v252, v1, v186
	v_mul_f32_e32 v186, 0x3e0293ee, v229
	v_fma_f32 v1, v66, s53, -v186
	v_fma_f32 v66, v67, s53, -v186
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[106:109], v[34:49]
	v_fma_f32 v67, v68, s53, -v186
	v_fma_f32 v68, v69, s53, -v186
	v_fma_f32 v69, v70, s53, -v186
	v_fma_f32 v70, v71, s53, -v186
	v_fma_f32 v71, v72, s53, -v186
	v_fma_f32 v72, v73, s53, -v186
	v_fma_f32 v73, v74, s53, -v186
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[122:125], v[106:109], v[50:65]
	v_fma_f32 v74, v75, s53, -v186
	v_fma_f32 v75, v76, s53, -v186
	v_fma_f32 v76, v77, s53, -v186
	v_fma_f32 v77, v78, s53, -v186
	v_fma_f32 v78, v79, s53, -v186
	v_fma_f32 v79, v80, s53, -v186
	v_fma_f32 v80, v81, s53, -v186
	v_mfma_f32_32x32x16_f16 v[2:17], v[212:215], v[102:105], v[2:17]
	v_fma_f32 v81, v82, s53, -v186
	v_fma_f32 v82, v83, s53, -v186
	v_fma_f32 v83, v84, s53, -v186
	v_fma_f32 v84, v85, s53, -v186
	v_fma_f32 v85, v86, s53, -v186
	v_fma_f32 v86, v87, s53, -v186
	v_fma_f32 v87, v88, s53, -v186
	v_mfma_f32_32x32x16_f16 v[18:33], v[182:185], v[102:105], v[18:33]
	v_fma_f32 v88, v89, s53, -v186
	v_fma_f32 v89, v90, s53, -v186
	v_fma_f32 v90, v91, s53, -v186
	v_fma_f32 v91, v92, s53, -v186
	v_fma_f32 v92, v93, s53, -v186
	v_fma_f32 v93, v94, s53, -v186
	v_fma_f32 v94, v95, s53, -v186
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[102:105], v[34:49]
	v_fma_f32 v95, v96, s53, -v186
	v_fma_f32 v96, v97, s53, -v186
	v_exp_f32_e32 v231, v66
	v_fma_f32 v66, v252, s53, -v186
	v_exp_f32_e32 v251, v1
	v_exp_f32_e32 v237, v67
	v_exp_f32_e32 v243, v68
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[118:121], v[102:105], v[50:65]
	v_exp_f32_e32 v244, v69
	v_exp_f32_e32 v245, v70
	v_exp_f32_e32 v246, v71
	v_exp_f32_e32 v247, v72
	v_exp_f32_e32 v239, v73
	v_exp_f32_e32 v240, v74
	v_exp_f32_e32 v208, v75
	v_mfma_f32_32x32x16_f16 v[2:17], v[194:197], v[98:101], v[2:17]
	v_exp_f32_e32 v209, v76
	v_exp_f32_e32 v241, v77
	v_exp_f32_e32 v233, v78
	v_exp_f32_e32 v213, v79
	v_exp_f32_e32 v215, v80
	v_exp_f32_e32 v214, v81
	v_exp_f32_e32 v217, v82
	v_mfma_f32_32x32x16_f16 v[18:33], v[178:181], v[98:101], v[18:33]
	v_exp_f32_e32 v216, v83
	v_exp_f32_e32 v221, v84
	v_exp_f32_e32 v220, v85
	v_exp_f32_e32 v219, v86
	v_exp_f32_e32 v218, v87
	v_exp_f32_e32 v223, v88
	v_exp_f32_e32 v222, v89
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[98:101], v[34:49]
	v_exp_f32_e32 v224, v90
	v_exp_f32_e32 v253, v91
	v_exp_f32_e32 v250, v92
	v_exp_f32_e32 v225, v93
	v_exp_f32_e32 v255, v94
	v_exp_f32_e32 v254, v95
	v_exp_f32_e32 v1, v96
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[114:117], v[98:101], v[50:65]
	v_exp_f32_e32 v212, v66
	v_mov_b32_e32 v100, v207
	v_mov_b32_e32 v99, v249
	s_setprio 1
	s_waitcnt vmcnt(4)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	scratch_load_dword v67, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v68, off, off offset:48 ; 4-byte Folded Reload
	s_lshl_b32 s12, s57, 14
	s_add_i32 s12, s12, 0
	s_add_i32 s48, s12, 0x8000
	v_lshlrev_b32_e32 v66, 1, v100
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v67, 1, v67
	v_add3_u32 v66, s12, v66, v67
	v_lshl_add_u32 v67, v99, 1, s48
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v68, v68, 1, s48
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
	v_add_u32_e32 v66, v66, v70
	scratch_load_dword v70, off, off offset:20 ; 4-byte Folded Reload
	v_ashrrev_i32_e32 v69, 4, v69
	v_add_lshl_u32 v69, v69, v226, 2
	v_ashrrev_i32_e32 v66, 4, v66
	ds_bpermute_b32 v69, v69, v206
	v_add_lshl_u32 v66, v66, v226, 2
	s_and_b32 s12, s47, 0xffff
	v_readfirstlane_b32 s57, v67
	s_or_b32 s13, s12, s42
	s_mov_b32 s12, s46
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v69, 1, v69
	s_mov_b32 m0, s57
	v_readfirstlane_b32 s57, v68
	buffer_load_dwordx4 v69, s[12:15], 0 offen lds
	s_mov_b32 m0, s57
	scratch_load_dword v68, off, off        ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	ds_bpermute_b32 v66, v66, v70
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	buffer_load_dwordx4 v66, s[12:15], 0 offen lds
	scratch_load_dword v66, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v66, 1, v66
	v_add3_u32 v67, s59, v232, v66
	v_add3_u32 v72, s59, v230, v66
	v_add3_u32 v73, s59, v210, v66
	v_add3_u32 v74, s59, v238, v66
	v_add3_u32 v75, s59, v248, v66
	v_add3_u32 v76, s59, v68, v66
	v_add3_u32 v77, s59, v227, v66
	v_add3_u32 v66, s59, v228, v66
	ds_read_b128 v[68:71], v67
	ds_read_b128 v[82:85], v67 offset:8192
	ds_read_b128 v[202:205], v72
	ds_read_b128 v[186:189], v72 offset:8192
	ds_read_b128 v[198:201], v73
	ds_read_b128 v[182:185], v73 offset:8192
	ds_read_b128 v[194:197], v74
	ds_read_b128 v[178:181], v74 offset:8192
	ds_read_b128 v[94:97], v75
	ds_read_b128 v[174:177], v75 offset:8192
	ds_read_b128 v[86:89], v76
	ds_read_b128 v[170:173], v76 offset:8192
	ds_read_b128 v[190:193], v77
	ds_read_b128 v[166:169], v77 offset:8192
	ds_read_b128 v[90:93], v66
	ds_read_b128 v[162:165], v66 offset:8192
	; sched_barrier mask(0x00000000)
	s_add_i32 s55, s55, 1
	s_add_u32 s21, s21, s44
	s_addc_u32 s51, s51, s45
	s_cmp_lt_i32 s55, s20
	s_mov_b32 s57, s61
	s_barrier
	s_cbranch_scc1 .LBB0_9
; %bb.10:                               ; %Flow1363
	scratch_load_dword v252, off, off offset:116 ; 4-byte Folded Reload
	scratch_load_dword v249, off, off offset:120 ; 4-byte Folded Reload
	scratch_load_dword v227, off, off offset:124 ; 4-byte Folded Reload
	scratch_load_dword v236, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v238, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v206, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v230, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v207, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v210, off, off offset:28 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v234, 4, v0
	v_lshlrev_b32_e32 v235, 3, v0
	v_lshrrev_b32_e32 v248, 2, v0
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execnz .LBB0_13
	s_branch .LBB0_14
.LBB0_11:
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v99, 0xff800000
	v_mov_b32_e32 v253, 1.0
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
	s_branch .LBB0_23
.LBB0_12:
	s_mov_b32 s22, 0
	s_add_i32 s23, 0, 0x8000
	s_add_i32 s48, 0, 0xc000
	v_mov_b32_e32 v211, 1.0
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
	s_mov_b32 s50, 0
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_14
.LBB0_13:
	s_barrier
.LBB0_14:
	s_or_b64 exec, exec, s[12:13]
	s_cmp_eq_u32 s36, 1
	s_cselect_b64 s[12:13], -1, 0
	s_cmp_lg_u32 s36, 1
	s_cselect_b64 s[20:21], -1, 0
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
	s_cbranch_vccnz .LBB0_16
; %bb.15:
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[98:113], v[68:71], v[130:133], 0
	v_mfma_f32_32x32x16_f16 v[114:129], v[82:85], v[130:133], 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[98:113], v[202:205], v[134:137], v[98:113]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[114:129], v[186:189], v[134:137], v[114:129]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[98:113], v[198:201], v[138:141], v[98:113]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[114:129], v[182:185], v[138:141], v[114:129]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[98:113], v[194:197], v[142:145], v[98:113]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[114:129], v[178:181], v[142:145], v[114:129]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[98:113], v[94:97], v[146:149], v[98:113]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[114:129], v[174:177], v[146:149], v[114:129]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[98:113], v[86:89], v[150:153], v[98:113]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[114:129], v[170:173], v[150:153], v[114:129]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[98:113], v[190:193], v[154:157], v[98:113]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[114:129], v[166:169], v[154:157], v[114:129]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[90:93], v[158:161], v[98:113]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[114:129], v[162:165], v[158:161], v[114:129]
.LBB0_16:
	s_waitcnt lgkmcnt(14)
	v_lshlrev_b32_e32 v68, 2, v0
	s_waitcnt lgkmcnt(7)
	v_and_b32_e32 v96, 12, v68
	v_and_b32_e32 v183, 16, v0
	s_waitcnt vmcnt(8)
	v_and_or_b32 v70, v248, 3, v252
	s_waitcnt vmcnt(4)
	v_or_b32_e32 v68, v96, v238
	v_or_b32_e32 v69, v236, v183
	v_lshlrev_b32_e32 v97, 7, v70
	v_or3_b32 v226, v69, v68, v97
	v_lshl_add_u32 v69, v226, 1, s23
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
	v_bitop3_b32 v69, v183, v69, v238 bitop3:0xf6
	v_or3_b32 v232, v69, v236, v97
	v_lshl_add_u32 v69, v232, 1, s23
	v_bitop3_b32 v68, v68, v236, 64 bitop3:0x36
	ds_read_b64_tr_b16 v[92:93], v69
	ds_read_b64_tr_b16 v[94:95], v69 offset:2048
	ds_read_b64_tr_b16 v[88:89], v69 offset:4096
	ds_read_b64_tr_b16 v[90:91], v69 offset:6144
	ds_read_b64_tr_b16 v[84:85], v69 offset:8192
	ds_read_b64_tr_b16 v[86:87], v69 offset:10240
	ds_read_b64_tr_b16 v[80:81], v69 offset:12288
	ds_read_b64_tr_b16 v[82:83], v69 offset:14336
	v_or3_b32 v242, v68, v183, v97
	v_cvt_pk_f16_f32 v71, v246, v247
	v_cvt_pk_f16_f32 v70, v244, v245
	v_cvt_pk_f16_f32 v69, v237, v243
	v_cvt_pk_f16_f32 v68, v251, v231
	v_mul_f32_e32 v16, v16, v212
	v_mul_f32_e32 v17, v17, v212
	v_mul_f32_e32 v14, v14, v212
	v_mul_f32_e32 v15, v15, v212
	v_mul_f32_e32 v12, v12, v212
	v_mul_f32_e32 v13, v13, v212
	v_mul_f32_e32 v10, v10, v212
	v_mul_f32_e32 v11, v11, v212
	v_mul_f32_e32 v8, v8, v212
	v_mul_f32_e32 v9, v9, v212
	v_mul_f32_e32 v6, v6, v212
	v_mul_f32_e32 v7, v7, v212
	v_mul_f32_e32 v4, v4, v212
	v_mul_f32_e32 v5, v5, v212
	v_mul_f32_e32 v2, v2, v212
	v_mul_f32_e32 v3, v3, v212
	v_mul_f32_e32 v32, v32, v212
	v_mul_f32_e32 v33, v33, v212
	v_mul_f32_e32 v30, v30, v212
	v_mul_f32_e32 v31, v31, v212
	v_mul_f32_e32 v28, v28, v212
	v_mul_f32_e32 v29, v29, v212
	v_mul_f32_e32 v26, v26, v212
	v_mul_f32_e32 v27, v27, v212
	v_mul_f32_e32 v24, v24, v212
	v_mul_f32_e32 v25, v25, v212
	v_mul_f32_e32 v22, v22, v212
	v_mul_f32_e32 v23, v23, v212
	v_mul_f32_e32 v20, v20, v212
	v_mul_f32_e32 v21, v21, v212
	v_mul_f32_e32 v18, v18, v212
	v_mul_f32_e32 v19, v19, v212
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[2:17], v[162:165], v[68:71], v[2:17]
	v_cvt_pk_f16_f32 v165, v213, v215
	v_cvt_pk_f16_f32 v164, v241, v233
	v_cvt_pk_f16_f32 v163, v208, v209
	v_cvt_pk_f16_f32 v162, v239, v240
	v_lshl_add_u32 v182, v242, 1, s23
	ds_read_b64_tr_b16 v[76:77], v182
	ds_read_b64_tr_b16 v[78:79], v182 offset:2048
	ds_read_b64_tr_b16 v[72:73], v182 offset:4096
	ds_read_b64_tr_b16 v[74:75], v182 offset:6144
	v_mul_f32_e32 v48, v48, v212
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[18:33], v[92:95], v[68:71], v[18:33]
	v_mul_f32_e32 v49, v49, v212
	v_mul_f32_e32 v46, v46, v212
	v_mul_f32_e32 v47, v47, v212
	v_mul_f32_e32 v44, v44, v212
	v_mul_f32_e32 v45, v45, v212
	v_mul_f32_e32 v42, v42, v212
	v_mul_f32_e32 v43, v43, v212
	v_mfma_f32_32x32x16_f16 v[2:17], v[166:169], v[162:165], v[2:17]
	v_mul_f32_e32 v40, v40, v212
	v_mul_f32_e32 v41, v41, v212
	v_mul_f32_e32 v38, v38, v212
	v_mul_f32_e32 v39, v39, v212
	v_mul_f32_e32 v36, v36, v212
	v_mul_f32_e32 v37, v37, v212
	v_mul_f32_e32 v34, v34, v212
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[18:33], v[88:91], v[162:165], v[18:33]
	v_mul_f32_e32 v35, v35, v212
	v_cvt_pk_f16_f32 v169, v218, v223
	v_cvt_pk_f16_f32 v168, v220, v219
	v_cvt_pk_f16_f32 v167, v216, v221
	v_cvt_pk_f16_f32 v166, v214, v217
	s_add_u32 s14, s18, s16
	s_movk_i32 s18, 0x60
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[76:79], v[68:71], v[34:49]
	s_addc_u32 s15, s19, s17
	v_bitop3_b32 v96, v235, v96, s18 bitop3:0x4e
	v_add_f32_e32 v67, v251, v231
	v_or3_b32 v231, v96, v183, v97
	s_add_u32 s16, s14, s16
	v_lshl_add_u32 v96, v231, 1, s23
	s_addc_u32 s14, s15, s17
	v_mfma_f32_32x32x16_f16 v[2:17], v[170:173], v[166:169], v[2:17]
	s_add_i32 s23, s50, 0
	v_cvt_pk_f16_f32 v181, v254, v1
	v_cvt_pk_f16_f32 v180, v225, v255
	v_cvt_pk_f16_f32 v179, v253, v250
	v_cvt_pk_f16_f32 v178, v222, v224
	ds_read_b64_tr_b16 v[92:93], v182 offset:8192
	ds_read_b64_tr_b16 v[94:95], v182 offset:10240
	ds_read_b64_tr_b16 v[170:171], v182 offset:12288
	ds_read_b64_tr_b16 v[172:173], v182 offset:14336
	s_and_b32 s14, s14, 0xffff
	v_mfma_f32_32x32x16_f16 v[18:33], v[84:87], v[166:169], v[18:33]
	v_cndmask_b32_e64 v85, 0, 1, s[2:3]
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_or_b32 s17, s14, s42
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	v_add_f32_e32 v67, v67, v237
	v_add_f32_e32 v67, v67, v243
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[34:49], v[72:75], v[162:165], v[34:49]
	v_add_u32_e32 v72, s23, v207
	v_add_u32_e32 v74, 0x8000, v72
	v_add_u32_e32 v72, s23, v210
	v_add_u32_e32 v75, 0x8000, v72
	v_readfirstlane_b32 s14, v74
	s_mov_b32 m0, s14
	v_readfirstlane_b32 s14, v75
	v_mfma_f32_32x32x16_f16 v[2:17], v[174:177], v[178:181], v[2:17]
	ds_read_b64_tr_b16 v[88:89], v96
	ds_read_b64_tr_b16 v[90:91], v96 offset:2048
	ds_read_b64_tr_b16 v[174:175], v96 offset:4096
	ds_read_b64_tr_b16 v[176:177], v96 offset:6144
	v_add_f32_e32 v67, v67, v244
	v_add_f32_e32 v67, v67, v245
	v_add_f32_e32 v67, v67, v246
	v_add_f32_e32 v67, v67, v247
	v_add_f32_e32 v67, v67, v239
	v_add_f32_e32 v67, v67, v240
	v_mfma_f32_32x32x16_f16 v[18:33], v[80:83], v[178:181], v[18:33]
	ds_read_b64_tr_b16 v[76:77], v96 offset:8192
	ds_read_b64_tr_b16 v[78:79], v96 offset:10240
	ds_read_b64_tr_b16 v[80:81], v96 offset:12288
	ds_read_b64_tr_b16 v[82:83], v96 offset:14336
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	scratch_load_dwordx2 v[72:73], off, off offset:56 ; 8-byte Folded Reload
	scratch_load_dword v87, off, off offset:112 ; 4-byte Folded Reload
	v_add_f32_e32 v67, v67, v208
	v_add_f32_e32 v67, v67, v209
	v_add_f32_e32 v67, v67, v241
	v_add_f32_e32 v67, v67, v233
	v_add_f32_e32 v67, v67, v213
	v_add_f32_e32 v67, v67, v215
	v_add_f32_e32 v67, v67, v214
	v_mul_f32_e32 v64, v64, v212
	v_mul_f32_e32 v65, v65, v212
	v_mul_f32_e32 v62, v62, v212
	v_mul_f32_e32 v63, v63, v212
	v_mul_f32_e32 v60, v60, v212
	v_mul_f32_e32 v61, v61, v212
	v_mul_f32_e32 v58, v58, v212
	v_mul_f32_e32 v59, v59, v212
	v_mul_f32_e32 v56, v56, v212
	v_mul_f32_e32 v57, v57, v212
	v_mul_f32_e32 v54, v54, v212
	v_mul_f32_e32 v55, v55, v212
	v_mul_f32_e32 v52, v52, v212
	v_mul_f32_e32 v53, v53, v212
	v_mul_f32_e32 v50, v50, v212
	v_mul_f32_e32 v51, v51, v212
	v_add_f32_e32 v67, v67, v217
	v_add_f32_e32 v67, v67, v216
	v_mfma_f32_32x32x16_f16 v[50:65], v[88:91], v[68:71], v[50:65]
	v_add_f32_e32 v67, v67, v221
	v_add_f32_e32 v67, v67, v220
	v_add_f32_e32 v67, v67, v219
	v_add_f32_e32 v67, v67, v218
	v_add_f32_e32 v67, v67, v223
	v_add_f32_e32 v67, v67, v222
	v_add_f32_e32 v67, v67, v224
	v_mfma_f32_32x32x16_f16 v[50:65], v[174:177], v[162:165], v[50:65]
	v_add_f32_e32 v67, v67, v253
	v_add_f32_e32 v67, v67, v250
	v_add_f32_e32 v67, v67, v225
	v_add_f32_e32 v67, v67, v255
	v_add_f32_e32 v67, v67, v254
	v_add_f32_e32 v162, v67, v1
	v_max_f32_e32 v1, v99, v99
	v_max_f32_e32 v67, v98, v98
	v_max_f32_e32 v1, v67, v1
	v_max3_f32 v1, v1, v100, v101
	v_mfma_f32_32x32x16_f16 v[34:49], v[92:95], v[166:169], v[34:49]
	v_max3_f32 v1, v1, v102, v103
	v_max3_f32 v1, v1, v104, v105
	v_max3_f32 v1, v1, v106, v107
	v_max3_f32 v1, v1, v108, v109
	v_max3_f32 v1, v1, v110, v111
	v_max3_f32 v1, v1, v112, v113
	v_max3_f32 v1, v1, v114, v115
	v_mfma_f32_32x32x16_f16 v[50:65], v[76:79], v[166:169], v[50:65]
	v_max3_f32 v1, v1, v116, v117
	v_max3_f32 v1, v1, v118, v119
	v_max3_f32 v1, v1, v120, v121
	v_max3_f32 v1, v1, v122, v123
	v_max3_f32 v1, v1, v124, v125
	v_max3_f32 v1, v1, v126, v127
	v_max3_f32 v1, v1, v128, v129
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[178:181], v[34:49]
	v_mov_b32_e32 v163, v162
	v_mov_b32_e32 v164, v1
	v_mov_b32_e32 v251, v206
	v_permlane32_swap_b32_e32 v162, v163
	v_permlane32_swap_b32_e32 v1, v164
	v_mfma_f32_32x32x16_f16 v[50:65], v[80:83], v[178:181], v[50:65]
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v86, 2, v87
	ds_bpermute_b32 v84, v86, v72
	ds_bpermute_b32 v86, v86, v206
	v_lshrrev_b64 v[72:73], v87, vcc
	v_and_b32_e32 v72, 1, v72
	v_cmp_eq_u32_e32 vcc, 1, v72
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v73, 1, v84
	v_bfrev_b32_e32 v84, 1
	v_cndmask_b32_e32 v72, v84, v73, vcc
	buffer_load_dwordx4 v72, s[16:19], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v72, 1, v86
	v_cndmask_b32_e32 v72, v84, v72, vcc
	s_mov_b32 m0, s14
	v_cmp_ne_u32_e64 s[14:15], 1, v85
	buffer_load_dwordx4 v72, s[16:19], 0 offen lds
	s_andn2_b64 vcc, exec, s[2:3]
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
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
	s_cbranch_vccnz .LBB0_18
; %bb.17:
	scratch_load_dword v66, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v71, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v92, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v90, 1, v66
	scratch_load_dword v66, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_add3_u32 v91, s22, v71, v90
	ds_read_b128 v[86:89], v91
	ds_read_b128 v[166:169], v91 offset:8192
	scratch_load_dword v91, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_add3_u32 v92, s22, v92, v90
	ds_read_b128 v[170:173], v92 offset:8192
	s_waitcnt vmcnt(1)
	v_add3_u32 v70, s22, v66, v90
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[82:85], v70 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[130:133], 0
	s_waitcnt vmcnt(0)
	v_add3_u32 v91, s22, v91, v90
	ds_read_b128 v[174:177], v91 offset:8192
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[134:137], v[66:81]
	ds_read_b128 v[86:89], v92
	scratch_load_dword v92, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[138:141], v[66:81]
	ds_read_b128 v[86:89], v91
	scratch_load_dword v91, off, off        ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_add3_u32 v92, s22, v92, v90
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[142:145], v[66:81]
	ds_read_b128 v[86:89], v92
	ds_read_b128 v[178:181], v92 offset:8192
	scratch_load_dword v92, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_add3_u32 v91, s22, v91, v90
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[146:149], v[66:81]
	ds_read_b128 v[86:89], v91
	ds_read_b128 v[182:185], v91 offset:8192
	scratch_load_dword v91, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_add3_u32 v92, s22, v92, v90
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[150:153], v[66:81]
	ds_read_b128 v[86:89], v92
	ds_read_b128 v[186:189], v92 offset:8192
	s_waitcnt vmcnt(0)
	v_add3_u32 v90, s22, v91, v90
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[154:157], v[66:81]
	ds_read_b128 v[86:89], v90
	ds_read_b128 v[190:193], v90 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[158:161], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[130:133], 0
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[134:137], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[138:141], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[142:145], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[146:149], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[150:153], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[154:157], v[82:97]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[190:193], v[158:161], v[82:97]
.LBB0_18:
	v_add_f32_e32 v253, v162, v163
	v_lshl_add_u32 v162, v226, 1, s48
	v_fmac_f32_e32 v253, v211, v212
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b64_tr_b16 v[222:223], v162
	ds_read_b64_tr_b16 v[224:225], v162 offset:2048
	ds_read_b64_tr_b16 v[218:219], v162 offset:4096
	ds_read_b64_tr_b16 v[220:221], v162 offset:6144
	ds_read_b64_tr_b16 v[214:215], v162 offset:8192
	ds_read_b64_tr_b16 v[216:217], v162 offset:10240
	ds_read_b64_tr_b16 v[210:211], v162 offset:12288
	ds_read_b64_tr_b16 v[212:213], v162 offset:14336
	v_lshl_add_u32 v162, v232, 1, s48
	v_max3_f32 v1, v229, v1, v164
	ds_read_b64_tr_b16 v[206:207], v162
	ds_read_b64_tr_b16 v[208:209], v162 offset:2048
	ds_read_b64_tr_b16 v[202:203], v162 offset:4096
	ds_read_b64_tr_b16 v[204:205], v162 offset:6144
	ds_read_b64_tr_b16 v[198:199], v162 offset:8192
	ds_read_b64_tr_b16 v[200:201], v162 offset:10240
	ds_read_b64_tr_b16 v[194:195], v162 offset:12288
	ds_read_b64_tr_b16 v[196:197], v162 offset:14336
	v_lshl_add_u32 v162, v242, 1, s48
	v_lshl_add_u32 v164, v231, 1, s48
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
	s_andn2_b64 vcc, exec, s[20:21]
	s_cbranch_vccnz .LBB0_20
; %bb.19:
	s_mov_b32 s16, 0x3e0293ee
	v_mov_b32_e32 v244, v227
	v_mul_f32_e32 v227, 0x3e0293ee, v1
	v_fma_f32 v98, v98, s16, -v227
	v_fma_f32 v99, v99, s16, -v227
	v_fma_f32 v100, v100, s16, -v227
	v_exp_f32_e32 v233, v98
	v_mov_b32_e32 v245, v234
	v_exp_f32_e32 v234, v99
	v_fma_f32 v101, v101, s16, -v227
	v_mov_b32_e32 v246, v235
	v_exp_f32_e32 v235, v100
	v_fma_f32 v102, v102, s16, -v227
	v_fma_f32 v114, v114, s16, -v227
	v_mov_b32_e32 v247, v236
	v_exp_f32_e32 v236, v101
	v_fma_f32 v103, v103, s16, -v227
	v_fma_f32 v109, v109, s16, -v227
	v_fma_f32 v112, v112, s16, -v227
	v_fma_f32 v115, v115, s16, -v227
	v_exp_f32_e32 v237, v102
	v_exp_f32_e32 v102, v114
	v_fma_f32 v114, v229, s16, -v227
	v_fma_f32 v104, v104, s16, -v227
	v_mov_b32_e32 v228, v248
	v_mov_b32_e32 v248, v238
	v_exp_f32_e32 v238, v103
	v_exp_f32_e32 v243, v109
	v_exp_f32_e32 v109, v112
	v_exp_f32_e32 v112, v115
	v_exp_f32_e32 v115, v114
	v_add_f32_e32 v114, v233, v234
	v_fma_f32 v105, v105, s16, -v227
	v_exp_f32_e32 v239, v104
	v_add_f32_e32 v114, v235, v114
	v_fma_f32 v106, v106, s16, -v227
	v_exp_f32_e32 v240, v105
	v_add_f32_e32 v114, v236, v114
	v_fma_f32 v107, v107, s16, -v227
	v_exp_f32_e32 v106, v106
	v_add_f32_e32 v114, v237, v114
	v_fma_f32 v108, v108, s16, -v227
	v_exp_f32_e32 v241, v107
	v_add_f32_e32 v114, v238, v114
	v_exp_f32_e32 v107, v108
	v_add_f32_e32 v114, v239, v114
	v_fma_f32 v110, v110, s16, -v227
	v_add_f32_e32 v114, v240, v114
	v_fma_f32 v111, v111, s16, -v227
	v_exp_f32_e32 v108, v110
	v_add_f32_e32 v114, v106, v114
	v_exp_f32_e32 v110, v111
	v_add_f32_e32 v114, v241, v114
	v_fma_f32 v113, v113, s16, -v227
	v_add_f32_e32 v114, v107, v114
	v_exp_f32_e32 v111, v113
	v_add_f32_e32 v114, v243, v114
	v_add_f32_e32 v114, v108, v114
	v_fma_f32 v116, v116, s16, -v227
	v_add_f32_e32 v114, v110, v114
	v_fma_f32 v117, v117, s16, -v227
	v_exp_f32_e32 v103, v116
	v_add_f32_e32 v114, v109, v114
	v_exp_f32_e32 v113, v117
	v_add_f32_e32 v114, v111, v114
	v_add_f32_e32 v114, v102, v114
	v_add_f32_e32 v114, v112, v114
	v_add_f32_e32 v114, v103, v114
	v_add_f32_e32 v114, v113, v114
	v_cvt_pk_f16_f32 v103, v103, v113
	v_cvt_pk_f16_f32 v102, v102, v112
	v_cvt_pk_f16_f32 v109, v109, v111
	v_cvt_pk_f16_f32 v108, v108, v110
	v_cvt_pk_f16_f32 v113, v239, v240
	v_cvt_pk_f16_f32 v112, v237, v238
	v_cvt_pk_f16_f32 v111, v235, v236
	v_cvt_pk_f16_f32 v110, v233, v234
	v_mul_f32_e32 v2, v2, v115
	v_mul_f32_e32 v3, v3, v115
	v_mul_f32_e32 v4, v4, v115
	v_mul_f32_e32 v5, v5, v115
	v_mul_f32_e32 v6, v6, v115
	v_mul_f32_e32 v7, v7, v115
	v_mul_f32_e32 v8, v8, v115
	v_mul_f32_e32 v9, v9, v115
	v_mul_f32_e32 v10, v10, v115
	v_mul_f32_e32 v11, v11, v115
	v_mul_f32_e32 v12, v12, v115
	v_mul_f32_e32 v13, v13, v115
	v_mul_f32_e32 v14, v14, v115
	v_mul_f32_e32 v15, v15, v115
	v_mul_f32_e32 v16, v16, v115
	v_mul_f32_e32 v17, v17, v115
	v_mul_f32_e32 v18, v18, v115
	v_mul_f32_e32 v19, v19, v115
	v_mul_f32_e32 v20, v20, v115
	v_mul_f32_e32 v21, v21, v115
	v_mul_f32_e32 v22, v22, v115
	v_mul_f32_e32 v23, v23, v115
	v_mul_f32_e32 v24, v24, v115
	v_mul_f32_e32 v25, v25, v115
	v_mul_f32_e32 v26, v26, v115
	v_mul_f32_e32 v27, v27, v115
	v_mul_f32_e32 v28, v28, v115
	v_mul_f32_e32 v29, v29, v115
	v_mul_f32_e32 v30, v30, v115
	v_mul_f32_e32 v31, v31, v115
	v_mul_f32_e32 v32, v32, v115
	v_mul_f32_e32 v33, v33, v115
	v_mul_f32_e32 v34, v34, v115
	v_mul_f32_e32 v35, v35, v115
	v_mul_f32_e32 v36, v36, v115
	v_mul_f32_e32 v37, v37, v115
	v_mul_f32_e32 v38, v38, v115
	v_mul_f32_e32 v39, v39, v115
	v_mul_f32_e32 v40, v40, v115
	v_mul_f32_e32 v41, v41, v115
	v_mul_f32_e32 v42, v42, v115
	v_mul_f32_e32 v43, v43, v115
	v_mul_f32_e32 v44, v44, v115
	v_mul_f32_e32 v45, v45, v115
	v_mul_f32_e32 v46, v46, v115
	v_mul_f32_e32 v47, v47, v115
	v_mul_f32_e32 v48, v48, v115
	v_mul_f32_e32 v49, v49, v115
	v_mul_f32_e32 v50, v50, v115
	v_mul_f32_e32 v51, v51, v115
	v_mul_f32_e32 v52, v52, v115
	v_mul_f32_e32 v53, v53, v115
	v_mul_f32_e32 v54, v54, v115
	v_mul_f32_e32 v55, v55, v115
	v_mul_f32_e32 v56, v56, v115
	v_mul_f32_e32 v57, v57, v115
	v_mul_f32_e32 v58, v58, v115
	v_mul_f32_e32 v59, v59, v115
	v_mul_f32_e32 v60, v60, v115
	v_mul_f32_e32 v61, v61, v115
	v_mul_f32_e32 v62, v62, v115
	v_mul_f32_e32 v63, v63, v115
	v_mul_f32_e32 v64, v64, v115
	v_mul_f32_e32 v65, v65, v115
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[2:17], v[222:225], v[110:113], v[2:17]
	v_cvt_pk_f16_f32 v107, v107, v243
	v_cvt_pk_f16_f32 v106, v106, v241
	v_fma_f32 v118, v118, s16, -v227
	v_fma_f32 v119, v119, s16, -v227
	v_exp_f32_e32 v104, v118
	v_fma_f32 v120, v120, s16, -v227
	v_fma_f32 v121, v121, s16, -v227
	v_mfma_f32_32x32x16_f16 v[18:33], v[206:209], v[110:113], v[18:33]
	v_exp_f32_e32 v116, v119
	v_exp_f32_e32 v105, v120
	v_exp_f32_e32 v117, v121
	v_fma_f32 v122, v122, s16, -v227
	v_fma_f32 v123, v123, s16, -v227
	v_exp_f32_e32 v98, v122
	v_add_f32_e32 v114, v104, v114
	v_mfma_f32_32x32x16_f16 v[34:49], v[190:193], v[110:113], v[34:49]
	v_fma_f32 v124, v124, s16, -v227
	v_exp_f32_e32 v118, v123
	v_add_f32_e32 v114, v116, v114
	v_fma_f32 v125, v125, s16, -v227
	v_exp_f32_e32 v99, v124
	v_add_f32_e32 v114, v105, v114
	v_cvt_pk_f16_f32 v105, v105, v117
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[50:65], v[174:177], v[110:113], v[50:65]
	v_cvt_pk_f16_f32 v104, v104, v116
	v_fma_f32 v126, v126, s16, -v227
	v_exp_f32_e32 v119, v125
	v_add_f32_e32 v114, v117, v114
	v_fma_f32 v127, v127, s16, -v227
	v_exp_f32_e32 v100, v126
	v_add_f32_e32 v114, v98, v114
	v_mfma_f32_32x32x16_f16 v[2:17], v[218:221], v[106:109], v[2:17]
	v_fma_f32 v128, v128, s16, -v227
	v_fma_f32 v129, v129, s16, -v227
	v_exp_f32_e32 v120, v127
	v_add_f32_e32 v114, v118, v114
	v_exp_f32_e32 v101, v128
	v_exp_f32_e32 v121, v129
	v_add_f32_e32 v114, v99, v114
	v_mfma_f32_32x32x16_f16 v[18:33], v[202:205], v[106:109], v[18:33]
	v_add_f32_e32 v114, v119, v114
	v_add_f32_e32 v114, v100, v114
	v_add_f32_e32 v114, v120, v114
	v_add_f32_e32 v114, v101, v114
	v_cvt_pk_f16_f32 v101, v101, v121
	v_cvt_pk_f16_f32 v100, v100, v120
	v_cvt_pk_f16_f32 v99, v99, v119
	v_mfma_f32_32x32x16_f16 v[34:49], v[186:189], v[106:109], v[34:49]
	v_cvt_pk_f16_f32 v98, v98, v118
	v_add_f32_e32 v114, v121, v114
	v_mov_b32_e32 v122, v114
	s_nop 1
	v_permlane32_swap_b32_e32 v114, v122
	v_add_f32_e32 v114, v114, v122
	v_fmac_f32_e32 v114, v253, v115
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[170:173], v[106:109], v[50:65]
	v_mov_b32_e32 v227, v244
	v_mov_b32_e32 v238, v248
	v_mov_b32_e32 v248, v228
	v_mov_b32_e32 v236, v247
	v_mov_b32_e32 v235, v246
	v_mov_b32_e32 v234, v245
	v_mov_b32_e32 v253, v114
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
	s_add_i32 s16, s23, 0x8000
	s_waitcnt lgkmcnt(14)
	v_max3_f32 v194, v98, v96, v97
	v_lshl_add_u32 v98, v226, 1, s23
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_lshl_add_u32 v99, v226, 1, s16
	ds_read_b64_tr_b16 v[190:191], v98 offset:32768
	ds_read_b64_tr_b16 v[192:193], v99 offset:2048
	ds_read_b64_tr_b16 v[186:187], v99 offset:4096
	ds_read_b64_tr_b16 v[188:189], v99 offset:6144
	ds_read_b64_tr_b16 v[182:183], v99 offset:8192
	ds_read_b64_tr_b16 v[184:185], v99 offset:10240
	ds_read_b64_tr_b16 v[178:179], v99 offset:12288
	ds_read_b64_tr_b16 v[180:181], v99 offset:14336
	v_lshl_add_u32 v98, v232, 1, s23
	v_lshl_add_u32 v99, v232, 1, s16
	ds_read_b64_tr_b16 v[174:175], v98 offset:32768
	ds_read_b64_tr_b16 v[176:177], v99 offset:2048
	ds_read_b64_tr_b16 v[170:171], v99 offset:4096
	ds_read_b64_tr_b16 v[172:173], v99 offset:6144
	ds_read_b64_tr_b16 v[166:167], v99 offset:8192
	ds_read_b64_tr_b16 v[168:169], v99 offset:10240
	ds_read_b64_tr_b16 v[162:163], v99 offset:12288
	ds_read_b64_tr_b16 v[164:165], v99 offset:14336
	v_lshl_add_u32 v98, v242, 1, s23
	v_lshl_add_u32 v99, v242, 1, s16
	ds_read_b64_tr_b16 v[126:127], v98 offset:32768
	ds_read_b64_tr_b16 v[128:129], v99 offset:2048
	ds_read_b64_tr_b16 v[122:123], v99 offset:4096
	ds_read_b64_tr_b16 v[124:125], v99 offset:6144
	ds_read_b64_tr_b16 v[118:119], v99 offset:8192
	ds_read_b64_tr_b16 v[120:121], v99 offset:10240
	ds_read_b64_tr_b16 v[114:115], v99 offset:12288
	ds_read_b64_tr_b16 v[116:117], v99 offset:14336
	v_lshl_add_u32 v98, v231, 1, s23
	v_lshl_add_u32 v100, v231, 1, s16
	ds_read_b64_tr_b16 v[110:111], v98 offset:32768
	ds_read_b64_tr_b16 v[112:113], v100 offset:2048
	ds_read_b64_tr_b16 v[106:107], v100 offset:4096
	ds_read_b64_tr_b16 v[108:109], v100 offset:6144
	ds_read_b64_tr_b16 v[102:103], v100 offset:8192
	ds_read_b64_tr_b16 v[104:105], v100 offset:10240
	ds_read_b64_tr_b16 v[98:99], v100 offset:12288
	ds_read_b64_tr_b16 v[100:101], v100 offset:14336
	v_mov_b32_e32 v195, v194
	v_cndmask_b32_e64 v1, v1, v229, s[12:13]
	s_nop 0
	v_permlane32_swap_b32_e32 v194, v195
	s_and_b64 vcc, exec, s[14:15]
	v_max3_f32 v194, v1, v194, v195
	s_cbranch_vccnz .LBB0_22
; %bb.21:
	s_mov_b32 s12, 0x3e0293ee
	v_mul_f32_e32 v195, 0x3e0293ee, v194
	v_fma_f32 v66, v66, s12, -v195
	v_fma_f32 v67, v67, s12, -v195
	v_fma_f32 v68, v68, s12, -v195
	v_fma_f32 v69, v69, s12, -v195
	v_fma_f32 v70, v70, s12, -v195
	v_fma_f32 v71, v71, s12, -v195
	v_fma_f32 v72, v72, s12, -v195
	v_fma_f32 v73, v73, s12, -v195
	v_exp_f32_e32 v196, v66
	v_fma_f32 v66, v1, s12, -v195
	v_fma_f32 v74, v74, s12, -v195
	v_fma_f32 v75, v75, s12, -v195
	v_fma_f32 v76, v76, s12, -v195
	v_fma_f32 v77, v77, s12, -v195
	v_fma_f32 v78, v78, s12, -v195
	v_fma_f32 v79, v79, s12, -v195
	v_fma_f32 v80, v80, s12, -v195
	v_fma_f32 v81, v81, s12, -v195
	v_fma_f32 v82, v82, s12, -v195
	v_fma_f32 v83, v83, s12, -v195
	v_fma_f32 v84, v84, s12, -v195
	v_fma_f32 v85, v85, s12, -v195
	v_fma_f32 v86, v86, s12, -v195
	v_fma_f32 v87, v87, s12, -v195
	v_fma_f32 v88, v88, s12, -v195
	v_fma_f32 v89, v89, s12, -v195
	v_fma_f32 v90, v90, s12, -v195
	v_fma_f32 v91, v91, s12, -v195
	v_fma_f32 v92, v92, s12, -v195
	v_fma_f32 v93, v93, s12, -v195
	v_fma_f32 v94, v94, s12, -v195
	v_fma_f32 v95, v95, s12, -v195
	v_fma_f32 v96, v96, s12, -v195
	v_fma_f32 v97, v97, s12, -v195
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
	v_fmac_f32_e32 v66, v253, v195
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[102:105], v[74:77], v[50:65]
	v_mov_b32_e32 v253, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[178:181], v[78:81], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[162:165], v[78:81], v[18:33]
	v_mfma_f32_32x32x16_f16 v[34:49], v[114:117], v[78:81], v[34:49]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[98:101], v[78:81], v[50:65]
.LBB0_22:                               ; %Flow1362
	s_waitcnt lgkmcnt(6)
	scratch_load_dwordx2 v[112:113], off, off offset:100 ; 8-byte Folded Reload
	scratch_load_dword v228, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[70:71], off, off offset:56 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[72:73], off, off offset:92 ; 8-byte Folded Reload
	scratch_load_dword v71, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v73, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v74, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v75, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v76, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v77, off, off        ; 4-byte Folded Reload
	scratch_load_dword v78, off, off offset:84 ; 4-byte Folded Reload
	scratch_load_dword v79, off, off offset:88 ; 4-byte Folded Reload
	scratch_load_dword v85, off, off offset:108 ; 4-byte Folded Reload
	v_mov_b32_e32 v206, v251
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v98, 31, v0
	v_lshrrev_b32_e32 v66, 1, v0
	v_lshlrev_b32_e32 v86, 1, v0
	v_cndmask_b32_e64 v99, v1, v194, s[2:3]
.LBB0_23:                               ; %Flow1365
	v_and_b32_e32 v1, 64, v66
	v_and_b32_e32 v66, 0xa0, v66
	v_or3_b32 v98, v66, v98, v1
	s_sub_i32 s20, s33, s67
	s_sub_i32 s21, s69, s36
	v_or_b32_e32 v1, s60, v98
	s_cmp_lt_i32 s21, 1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_28
; %bb.24:                               ; %.lr.ph1215
	s_cmp_lg_u32 s41, 0
	s_cselect_b64 s[2:3], -1, 0
	s_lshl_b32 s13, s36, 6
	s_mul_i32 s16, s13, s40
	s_mul_i32 s18, s13, s37
	s_and_b32 s13, s37, 0x3fff
	s_or_b32 s42, s13, 0x4000
	s_and_b32 s13, s40, 0x3fff
	s_lshl_b32 s12, s40, 6
	s_or_b32 s40, s13, 0x4000
	s_movk_i32 s13, 0x60
	v_and_b32_e32 v66, 16, v0
	s_waitcnt vmcnt(10)
	v_lshlrev_b32_e32 v110, 1, v70
	v_bitop3_b32 v70, v112, v86, s13 bitop3:0x78
	v_lshl_add_u32 v66, v66, 1, 0
	v_lshlrev_b32_e32 v67, 2, v0
	s_waitcnt vmcnt(0)
	v_add3_u32 v102, 0, v71, v85
	v_lshlrev_b32_e32 v70, 1, v70
	v_lshlrev_b32_e32 v71, 1, v228
	v_and_b32_e32 v67, 12, v67
	v_lshl_add_u32 v69, v236, 1, v66
	v_add3_u32 v112, 0, v70, v71
	v_and_or_b32 v70, v248, 3, v252
	s_lshl_b32 s14, s37, 6
	v_lshlrev_b32_e32 v100, 1, v72
	v_lshl_add_u32 v71, v67, 1, v69
	v_lshlrev_b32_e32 v72, 1, v238
	v_lshlrev_b32_e32 v70, 8, v70
	s_lshl_b32 s41, s69, 6
	v_or_b32_e32 v68, v238, v67
	s_ashr_i32 s17, s16, 31
	s_ashr_i32 s19, s18, 31
	v_add3_u32 v113, v71, v72, v70
	v_bitop3_b32 v71, v67, v238, 32 bitop3:0x36
	v_bitop3_b32 v67, v235, v67, s13 bitop3:0x4e
	s_ashr_i32 s15, s14, 31
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[36:37], s[38:39], 1
	s_lshl_b64 s[34:35], s[34:35], 1
	s_add_u32 s23, s36, s34
	s_addc_u32 s34, s37, s35
	s_lshl_b64 s[30:31], s[30:31], 1
	s_add_u32 s23, s23, s30
	s_addc_u32 s30, s34, s31
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s16, s23, s16
	s_addc_u32 s17, s30, s17
	s_add_u32 s23, s6, s16
	s_addc_u32 s30, s7, s17
	s_lshl_b64 s[6:7], s[12:13], 1
	s_lshl_b64 s[12:13], s[28:29], 1
	s_lshl_b64 s[16:17], s[26:27], 1
	s_add_u32 s16, s12, s16
	s_addc_u32 s17, s13, s17
	s_lshl_b64 s[12:13], s[24:25], 1
	s_add_u32 s16, s16, s12
	s_addc_u32 s17, s17, s13
	s_lshl_b64 s[12:13], s[18:19], 1
	s_add_u32 s12, s16, s12
	s_addc_u32 s13, s17, s13
	s_add_u32 s24, s4, s12
	v_bitop3_b32 v68, v68, v236, 64 bitop3:0x36
	s_addc_u32 s25, s5, s13
	s_lshl_b32 s4, s21, 6
	v_lshlrev_b32_e32 v71, 1, v71
	v_lshlrev_b32_e32 v68, 1, v68
	v_lshlrev_b32_e32 v67, 1, v67
	s_sub_i32 s26, 0, s4
	s_add_i32 s4, s20, s41
	s_mov_b32 s22, 0
	v_lshlrev_b32_e32 v101, 1, v230
	v_add3_u32 v103, 0, v73, v85
	v_add3_u32 v104, 0, v74, v85
	v_add3_u32 v105, 0, v75, v85
	v_add3_u32 v106, 0, v76, v85
	v_add3_u32 v107, 0, v77, v85
	v_add3_u32 v108, 0, v78, v85
	v_add3_u32 v109, 0, v79, v85
	v_lshlrev_b32_e32 v111, 1, v206
	v_add3_u32 v114, v69, v71, v70
	v_add3_u32 v115, v66, v68, v70
	v_add3_u32 v116, v66, v67, v70
	s_lshl_b64 s[16:17], s[14:15], 1
	v_add_u32_e32 v117, s4, v252
	v_or_b32_e32 v118, s41, v252
	v_or_b32_e32 v119, s41, v234
	s_movk_i32 s27, 0xffc0
	s_lshl_b32 s28, s42, 16
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_xor_b64 s[18:19], s[2:3], -1
	s_mov_b32 s29, 0x3e0293ee
	s_lshl_b32 s31, s40, 16
	v_bfrev_b32_e32 v120, 1
	v_mov_b32_e32 v121, 0xff800000
	s_branch .LBB0_26
.LBB0_25:                               ;   in Loop: Header=BB0_26 Depth=1
	v_add_u32_e32 v174, s26, v117
	v_add_u32_e32 v162, 1, v174
	v_cmp_ge_i32_e32 vcc, v1, v174
	v_add_u32_e32 v163, 2, v174
	v_add_u32_e32 v164, 3, v174
	v_cndmask_b32_e32 v82, v121, v82, vcc
	v_cmp_ge_i32_e32 vcc, v1, v162
	v_add_u32_e32 v165, 8, v174
	v_add_u32_e32 v166, 9, v174
	v_cndmask_b32_e32 v83, v121, v83, vcc
	v_cmp_ge_i32_e32 vcc, v1, v163
	v_add_u32_e32 v167, 10, v174
	v_add_u32_e32 v168, 11, v174
	v_cndmask_b32_e32 v84, v121, v84, vcc
	v_cmp_ge_i32_e32 vcc, v1, v164
	v_add_u32_e32 v169, 16, v174
	v_add_u32_e32 v170, 17, v174
	v_cndmask_b32_e32 v85, v121, v85, vcc
	v_cmp_ge_i32_e32 vcc, v1, v165
	s_barrier
	s_nop 0
	v_cndmask_b32_e32 v86, v121, v86, vcc
	v_cmp_ge_i32_e32 vcc, v1, v166
	s_waitcnt vmcnt(1)
	ds_write_b128 v249, v[66:69]
	s_waitcnt vmcnt(0)
	ds_write_b128 v227, v[70:73]
	v_cndmask_b32_e32 v87, v121, v87, vcc
	v_cmp_ge_i32_e32 vcc, v1, v167
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cndmask_b32_e32 v88, v121, v88, vcc
	v_cmp_ge_i32_e32 vcc, v1, v168
	ds_read_b128 v[66:69], v102
	s_nop 0
	v_cndmask_b32_e32 v89, v121, v89, vcc
	v_cmp_ge_i32_e32 vcc, v1, v169
	v_add_u32_e32 v171, 18, v174
	v_add_u32_e32 v172, 19, v174
	v_cndmask_b32_e32 v90, v121, v90, vcc
	v_cmp_ge_i32_e32 vcc, v1, v170
	v_add_u32_e32 v173, 24, v174
	v_add_u32_e32 v175, 25, v174
	v_cndmask_b32_e32 v91, v121, v91, vcc
	v_cmp_ge_i32_e32 vcc, v1, v171
	v_add_u32_e32 v176, 26, v174
	v_add_u32_e32 v177, 27, v174
	v_cndmask_b32_e32 v92, v121, v92, vcc
	v_cmp_ge_i32_e32 vcc, v1, v172
	ds_read_b128 v[162:165], v102 offset:8192
	ds_read_b128 v[70:73], v103
	v_cndmask_b32_e32 v93, v121, v93, vcc
	v_cmp_ge_i32_e32 vcc, v1, v173
	v_add_u32_e32 v182, 32, v174
	v_add_u32_e32 v183, 33, v174
	v_cndmask_b32_e32 v94, v121, v94, vcc
	v_cmp_ge_i32_e32 vcc, v1, v175
	v_add_u32_e32 v184, 34, v174
	v_add_u32_e32 v190, 35, v174
	v_cndmask_b32_e32 v95, v121, v95, vcc
	v_cmp_ge_i32_e32 vcc, v1, v176
	v_add_u32_e32 v191, 40, v174
	v_add_u32_e32 v192, 41, v174
	v_cndmask_b32_e32 v96, v121, v96, vcc
	v_cmp_ge_i32_e32 vcc, v1, v177
	v_add_u32_e32 v194, 42, v174
	v_add_u32_e32 v195, 43, v174
	v_cndmask_b32_e32 v97, v121, v97, vcc
	v_cmp_ge_i32_e32 vcc, v1, v182
	v_add_u32_e32 v196, 48, v174
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[130:133], v[82:97]
	ds_read_b128 v[166:169], v103 offset:8192
	ds_read_b128 v[66:69], v104
	v_add_u32_e32 v198, 49, v174
	v_add_u32_e32 v199, 50, v174
	v_add_u32_e32 v200, 51, v174
	v_add_u32_e32 v201, 56, v174
	v_add_u32_e32 v202, 57, v174
	v_add_u32_e32 v203, 58, v174
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[70:73], v[134:137], v[82:97]
	ds_read_b128 v[170:173], v104 offset:8192
	ds_read_b128 v[70:73], v105
	v_add_u32_e32 v204, 59, v174
	ds_read_b128 v[174:177], v106
	ds_read_b128 v[178:181], v105 offset:8192
	s_and_b32 s12, s30, 0xffff
	s_or_b32 s13, s12, s31
	s_mov_b32 s12, s23
	s_add_i32 s22, s22, 1
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[138:141], v[82:97]
	v_cndmask_b32_e32 v66, v121, v123, vcc
	v_cmp_ge_i32_e32 vcc, v1, v183
	s_add_u32 s23, s23, s6
	s_addc_u32 s30, s30, s7
	v_cndmask_b32_e32 v67, v121, v127, vcc
	v_cmp_ge_i32_e32 vcc, v1, v184
	ds_read_b128 v[182:185], v107
	ds_read_b128 v[186:189], v106 offset:8192
	v_cndmask_b32_e32 v68, v121, v129, vcc
	v_cmp_ge_i32_e32 vcc, v1, v190
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[70:73], v[142:145], v[82:97]
	s_add_u32 s24, s24, s16
	v_cndmask_b32_e32 v69, v121, v77, vcc
	v_cmp_ge_i32_e32 vcc, v1, v191
	s_addc_u32 s25, s25, s17
	s_sub_i32 s27, s27, 64
	v_cndmask_b32_e32 v70, v121, v124, vcc
	v_cmp_ge_i32_e32 vcc, v1, v192
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[146:149], v[82:97]
	ds_read_b128 v[174:177], v108
	ds_read_b128 v[190:193], v107 offset:8192
	v_cndmask_b32_e32 v71, v121, v128, vcc
	v_cmp_ge_i32_e32 vcc, v1, v194
	v_add_u32_e32 v117, 64, v117
	v_add_u32_e32 v118, 64, v118
	v_cndmask_b32_e32 v72, v121, v74, vcc
	v_cmp_ge_i32_e32 vcc, v1, v195
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[150:153], v[82:97]
	s_cmp_lt_i32 s22, s21
	v_cndmask_b32_e32 v73, v121, v80, vcc
	v_cmp_ge_i32_e32 vcc, v1, v196
	ds_read_b128 v[182:185], v109
	ds_read_b128 v[194:197], v108 offset:8192
	v_cndmask_b32_e32 v74, v121, v125, vcc
	v_cmp_ge_i32_e32 vcc, v1, v198
	v_add_u32_e32 v119, 64, v119
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[154:157], v[82:97]
	v_cndmask_b32_e32 v75, v121, v75, vcc
	v_cmp_ge_i32_e32 vcc, v1, v199
	ds_read_b128 v[174:177], v109 offset:8192
	s_nop 0
	v_cndmask_b32_e32 v76, v121, v76, vcc
	v_cmp_ge_i32_e32 vcc, v1, v200
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[158:161], v[82:97]
	v_cndmask_b32_e32 v77, v121, v81, vcc
	v_cmp_ge_i32_e32 vcc, v1, v201
	s_nop 1
	v_cndmask_b32_e32 v78, v121, v78, vcc
	v_cmp_ge_i32_e32 vcc, v1, v202
	s_nop 5
	v_max_f32_e32 v123, v82, v82
	v_cndmask_b32_e32 v79, v121, v79, vcc
	v_cmp_ge_i32_e32 vcc, v1, v203
	s_nop 1
	v_cndmask_b32_e32 v80, v121, v122, vcc
	v_cmp_ge_i32_e32 vcc, v1, v204
	v_max_f32_e32 v122, v83, v83
	v_max_f32_e32 v122, v123, v122
	v_cndmask_b32_e32 v81, v121, v126, vcc
	v_max3_f32 v122, v122, v84, v85
	v_max3_f32 v122, v122, v86, v87
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[130:133], v[66:81]
	v_max3_f32 v122, v122, v88, v89
	v_max3_f32 v122, v122, v90, v91
	v_max3_f32 v122, v122, v92, v93
	v_max3_f32 v122, v122, v94, v95
	v_max3_f32 v122, v122, v96, v97
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[134:137], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[138:141], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[142:145], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[146:149], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[190:193], v[150:153], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[194:197], v[154:157], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[158:161], v[66:81]
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
	v_max3_f32 v122, v99, v122, v123
	v_mul_f32_e32 v123, 0x3e0293ee, v122
	v_fma_f32 v124, v66, s29, -v123
	v_fma_f32 v128, v70, s29, -v123
	v_cndmask_b32_e64 v66, v120, v110, s[2:3]
	v_cndmask_b32_e64 v70, v120, v111, s[4:5]
	v_fma_f32 v125, v67, s29, -v123
	v_fma_f32 v126, v68, s29, -v123
	v_fma_f32 v127, v69, s29, -v123
	v_fma_f32 v129, v71, s29, -v123
	v_fma_f32 v162, v72, s29, -v123
	v_fma_f32 v163, v73, s29, -v123
	buffer_load_dwordx4 v[66:69], v66, s[12:15], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[70:73], v70, s[12:15], 0 offen
	v_fma_f32 v82, v82, s29, -v123
	v_fma_f32 v83, v83, s29, -v123
	v_fma_f32 v84, v84, s29, -v123
	v_exp_f32_e32 v82, v82
	v_exp_f32_e32 v83, v83
	v_fma_f32 v85, v85, s29, -v123
	v_exp_f32_e32 v84, v84
	v_fma_f32 v86, v86, s29, -v123
	v_exp_f32_e32 v85, v85
	v_fma_f32 v87, v87, s29, -v123
	v_fma_f32 v74, v74, s29, -v123
	v_exp_f32_e32 v86, v86
	v_fma_f32 v88, v88, s29, -v123
	v_fma_f32 v75, v75, s29, -v123
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v164, v74
	v_add_f32_e32 v74, v82, v83
	v_fma_f32 v89, v89, s29, -v123
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v165, v75
	v_add_f32_e32 v74, v84, v74
	v_fma_f32 v75, v99, s29, -v123
	v_fma_f32 v90, v90, s29, -v123
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v74, v85, v74
	v_exp_f32_e32 v99, v75
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b128 v112, v[66:69]
	s_waitcnt vmcnt(0)
	ds_write_b128 v112, v[70:73] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[70:71], v113
	ds_read_b64_tr_b16 v[72:73], v113 offset:2048
	v_fma_f32 v91, v91, s29, -v123
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v74, v86, v74
	v_fma_f32 v92, v92, s29, -v123
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v74, v87, v74
	v_fma_f32 v93, v93, s29, -v123
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v74, v88, v74
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v74, v89, v74
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
	v_cvt_pk_f16_f32 v66, v82, v83
	v_cvt_pk_f16_f32 v67, v84, v85
	v_cvt_pk_f16_f32 v68, v86, v87
	v_cvt_pk_f16_f32 v69, v88, v89
	v_add_f32_e32 v74, v90, v74
	v_add_f32_e32 v74, v91, v74
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[70:73], v[66:69], v[2:17]
	v_fma_f32 v94, v94, s29, -v123
	v_fma_f32 v95, v95, s29, -v123
	v_fma_f32 v96, v96, s29, -v123
	v_fma_f32 v97, v97, s29, -v123
	v_fma_f32 v76, v76, s29, -v123
	v_fma_f32 v77, v77, s29, -v123
	v_add_f32_e32 v74, v92, v74
	v_fma_f32 v78, v78, s29, -v123
	v_fma_f32 v79, v79, s29, -v123
	v_fma_f32 v80, v80, s29, -v123
	v_fma_f32 v81, v81, s29, -v123
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_exp_f32_e32 v166, v76
	v_exp_f32_e32 v167, v77
	v_add_f32_e32 v123, v93, v74
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
	v_exp_f32_e32 v162, v162
	v_exp_f32_e32 v163, v163
	v_exp_f32_e32 v168, v78
	v_exp_f32_e32 v169, v79
	v_exp_f32_e32 v170, v80
	v_exp_f32_e32 v171, v81
	ds_read_b64_tr_b16 v[78:79], v113 offset:8192
	ds_read_b64_tr_b16 v[80:81], v113 offset:10240
	v_cvt_pk_f16_f32 v74, v124, v125
	v_cvt_pk_f16_f32 v75, v126, v127
	v_cvt_pk_f16_f32 v76, v128, v129
	v_cvt_pk_f16_f32 v77, v162, v163
	ds_read_b64_tr_b16 v[82:83], v113 offset:12288
	ds_read_b64_tr_b16 v[84:85], v113 offset:14336
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[78:81], v[74:77], v[2:17]
	ds_read_b64_tr_b16 v[86:87], v114
	ds_read_b64_tr_b16 v[88:89], v114 offset:2048
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
	v_cvt_pk_f16_f32 v78, v164, v165
	v_cvt_pk_f16_f32 v79, v166, v167
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[86:89], v[66:69], v[18:33]
	v_cvt_pk_f16_f32 v80, v168, v169
	v_cvt_pk_f16_f32 v81, v170, v171
	v_add_f32_e32 v86, v94, v123
	v_add_f32_e32 v86, v95, v86
	v_add_f32_e32 v86, v96, v86
	v_add_f32_e32 v90, v97, v86
	v_mul_f32_e32 v34, v34, v99
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[78:81], v[2:17]
	ds_read_b64_tr_b16 v[82:83], v114 offset:4096
	ds_read_b64_tr_b16 v[84:85], v114 offset:6144
	ds_read_b64_tr_b16 v[86:87], v114 offset:8192
	ds_read_b64_tr_b16 v[88:89], v114 offset:10240
	v_mul_f32_e32 v35, v35, v99
	v_mul_f32_e32 v36, v36, v99
	v_mul_f32_e32 v37, v37, v99
	v_mul_f32_e32 v38, v38, v99
	v_mul_f32_e32 v39, v39, v99
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[70:73], v[18:33]
	ds_read_b64_tr_b16 v[82:83], v114 offset:12288
	ds_read_b64_tr_b16 v[84:85], v114 offset:14336
	v_mul_f32_e32 v40, v40, v99
	v_mul_f32_e32 v41, v41, v99
	v_mul_f32_e32 v42, v42, v99
	v_mul_f32_e32 v43, v43, v99
	v_mul_f32_e32 v44, v44, v99
	v_mul_f32_e32 v45, v45, v99
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[86:89], v[74:77], v[18:33]
	ds_read_b64_tr_b16 v[86:87], v115
	ds_read_b64_tr_b16 v[88:89], v115 offset:2048
	v_mul_f32_e32 v46, v46, v99
	v_mul_f32_e32 v47, v47, v99
	v_mul_f32_e32 v48, v48, v99
	v_mul_f32_e32 v49, v49, v99
	v_mul_f32_e32 v50, v50, v99
	v_mul_f32_e32 v51, v51, v99
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[86:89], v[66:69], v[34:49]
	v_add_f32_e32 v86, v124, v90
	v_add_f32_e32 v86, v125, v86
	v_add_f32_e32 v86, v126, v86
	v_add_f32_e32 v90, v127, v86
	v_mul_f32_e32 v52, v52, v99
	v_mul_f32_e32 v53, v53, v99
	v_mul_f32_e32 v54, v54, v99
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[78:81], v[18:33]
	ds_read_b64_tr_b16 v[82:83], v115 offset:4096
	ds_read_b64_tr_b16 v[84:85], v115 offset:6144
	ds_read_b64_tr_b16 v[86:87], v115 offset:8192
	ds_read_b64_tr_b16 v[88:89], v115 offset:10240
	v_mul_f32_e32 v55, v55, v99
	v_mul_f32_e32 v56, v56, v99
	v_mul_f32_e32 v57, v57, v99
	v_mul_f32_e32 v58, v58, v99
	v_mul_f32_e32 v59, v59, v99
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[70:73], v[34:49]
	ds_read_b64_tr_b16 v[82:83], v115 offset:12288
	ds_read_b64_tr_b16 v[84:85], v115 offset:14336
	v_mul_f32_e32 v60, v60, v99
	v_mul_f32_e32 v61, v61, v99
	v_mul_f32_e32 v62, v62, v99
	v_mul_f32_e32 v63, v63, v99
	v_mul_f32_e32 v64, v64, v99
	v_mul_f32_e32 v65, v65, v99
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[86:89], v[74:77], v[34:49]
	ds_read_b64_tr_b16 v[86:87], v116
	ds_read_b64_tr_b16 v[88:89], v116 offset:2048
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[86:89], v[66:69], v[50:65]
	v_add_f32_e32 v66, v128, v90
	v_add_f32_e32 v66, v129, v66
	v_add_f32_e32 v66, v162, v66
	v_add_f32_e32 v66, v163, v66
	v_add_f32_e32 v86, v164, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[78:81], v[34:49]
	ds_read_b64_tr_b16 v[82:83], v116 offset:4096
	ds_read_b64_tr_b16 v[84:85], v116 offset:6144
	ds_read_b64_tr_b16 v[66:67], v116 offset:8192
	ds_read_b64_tr_b16 v[68:69], v116 offset:10240
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[82:85], v[70:73], v[50:65]
	v_add_f32_e32 v70, v165, v86
	v_add_f32_e32 v70, v166, v70
	v_add_f32_e32 v70, v167, v70
	v_add_f32_e32 v70, v168, v70
	v_add_f32_e32 v82, v169, v70
	ds_read_b64_tr_b16 v[70:71], v116 offset:12288
	ds_read_b64_tr_b16 v[72:73], v116 offset:14336
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[66:69], v[74:77], v[50:65]
	v_add_f32_e32 v66, v170, v82
	v_add_f32_e32 v66, v171, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v253, v99
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[78:81], v[50:65]
	v_mov_b32_e32 v99, v122
	v_mov_b32_e32 v253, v66
	s_cbranch_scc0 .LBB0_28
.LBB0_26:                               ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v66, s26, v119
	v_add_u32_e32 v67, 32, v66
	s_and_b32 s2, s25, 0xffff
	s_or_b32 s13, s2, s28
	v_cmp_gt_i32_e64 s[2:3], s67, v66
	v_cmp_gt_i32_e64 s[4:5], s67, v67
	s_mov_b32 s12, s24
	v_cndmask_b32_e64 v66, v120, v100, s[2:3]
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
	s_cbranch_vccnz .LBB0_25
; %bb.27:                               ;   in Loop: Header=BB0_26 Depth=1
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
	v_add_u32_e32 v162, 41, v74
	v_cndmask_b32_e64 v91, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v92
	v_add_u32_e32 v163, 42, v74
	v_add_u32_e32 v164, 43, v74
	v_cndmask_b32_e64 v92, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v93
	v_add_u32_e32 v165, 48, v74
	v_add_u32_e32 v166, 49, v74
	v_cndmask_b32_e64 v93, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v94
	v_add_u32_e32 v167, 50, v74
	v_add_u32_e32 v168, 51, v74
	v_cndmask_b32_e64 v94, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v95
	v_add_u32_e32 v169, 56, v74
	v_add_u32_e32 v170, 57, v74
	v_cndmask_b32_e64 v95, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v96
	v_add_u32_e32 v171, 58, v74
	v_add_u32_e32 v172, 59, v74
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
	v_cmp_gt_i32_e32 vcc, s67, v162
	s_nop 1
	v_cndmask_b32_e64 v128, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v163
	s_nop 1
	v_cndmask_b32_e64 v74, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v164
	s_nop 1
	v_cndmask_b32_e64 v80, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v165
	s_nop 1
	v_cndmask_b32_e64 v125, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v166
	s_nop 1
	v_cndmask_b32_e64 v75, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v167
	s_nop 1
	v_cndmask_b32_e64 v76, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v168
	s_nop 1
	v_cndmask_b32_e64 v81, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v169
	s_nop 1
	v_cndmask_b32_e64 v78, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v170
	s_nop 1
	v_cndmask_b32_e64 v79, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v171
	s_nop 1
	v_cndmask_b32_e64 v122, v121, 0, vcc
	v_cmp_gt_i32_e32 vcc, s67, v172
	s_nop 1
	v_cndmask_b32_e64 v126, v121, 0, vcc
	s_branch .LBB0_25
.LBB0_28:                               ; %._crit_edge1216
	v_div_scale_f32 v66, s[2:3], v253, v253, 1.0
	v_rcp_f32_e32 v66, v66
	v_div_scale_f32 v67, vcc, 1.0, v253, 1.0
	s_add_i32 s2, s60, 0x100
	v_mul_f32_e32 v66, v67, v66
	s_cmp_le_i32 s20, s60
	s_nop 0
	v_div_fmas_f32 v66, 0, 0, v66
	v_div_fixup_f32 v66, v66, v253, 1.0
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
	s_waitcnt vmcnt(8)
	v_pk_mul_f32 v[70:71], v[34:35], v[66:67] op_sel_hi:[1,0]
	s_waitcnt vmcnt(7)
	v_pk_mul_f32 v[72:73], v[36:37], v[66:67] op_sel_hi:[1,0]
	s_waitcnt vmcnt(5)
	v_pk_mul_f32 v[74:75], v[38:39], v[66:67] op_sel_hi:[1,0]
	s_waitcnt vmcnt(3)
	v_pk_mul_f32 v[76:77], v[40:41], v[66:67] op_sel_hi:[1,0]
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[78:79], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[46:47], v[66:67] op_sel_hi:[1,0]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[84:85], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[64:65], v[66:67] op_sel_hi:[1,0]
	s_cselect_b64 s[6:7], -1, 0
	v_cvt_pk_f16_f32 v62, v2, v3
	v_cvt_pk_f16_f32 v64, v4, v5
	v_cvt_pk_f16_f32 v56, v6, v7
	v_cvt_pk_f16_f32 v60, v8, v9
	v_cvt_pk_f16_f32 v52, v10, v11
	v_cvt_pk_f16_f32 v57, v12, v13
	v_cvt_pk_f16_f32 v48, v14, v15
	v_cvt_pk_f16_f32 v53, v16, v17
	v_cvt_pk_f16_f32 v44, v18, v19
	v_cvt_pk_f16_f32 v49, v20, v21
	v_cvt_pk_f16_f32 v40, v22, v23
	v_cvt_pk_f16_f32 v45, v24, v25
	v_cvt_pk_f16_f32 v36, v26, v27
	v_cvt_pk_f16_f32 v41, v28, v29
	v_cvt_pk_f16_f32 v32, v30, v31
	v_cvt_pk_f16_f32 v37, v68, v69
	v_cvt_pk_f16_f32 v28, v70, v71
	v_cvt_pk_f16_f32 v33, v72, v73
	v_cvt_pk_f16_f32 v24, v74, v75
	v_cvt_pk_f16_f32 v29, v76, v77
	v_cvt_pk_f16_f32 v20, v78, v79
	v_cvt_pk_f16_f32 v25, v80, v81
	v_cvt_pk_f16_f32 v16, v82, v83
	v_cvt_pk_f16_f32 v21, v84, v85
	v_cvt_pk_f16_f32 v12, v86, v87
	v_cvt_pk_f16_f32 v17, v88, v89
	v_cvt_pk_f16_f32 v8, v90, v91
	v_cvt_pk_f16_f32 v13, v92, v93
	v_cvt_pk_f16_f32 v4, v94, v95
	v_cvt_pk_f16_f32 v9, v96, v97
	v_cvt_pk_f16_f32 v2, v100, v101
	v_cvt_pk_f16_f32 v5, v66, v67
	s_or_b64 s[4:5], s[4:5], s[6:7]
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
	s_cbranch_vccnz .LBB0_30
; %bb.29:
	v_cmp_gt_i32_e32 vcc, s20, v1
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
.LBB0_30:
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
	s_cbranch_scc1 .LBB0_32
; %bb.31:
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v253
	v_mov_b32_e32 v66, 0x42000000
	v_cmp_gt_i32_e64 s[8:9], s33, v1
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v253, v67
	v_log_f32_e32 v67, v67
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_barrier
	v_sub_f32_e32 v1, v67, v66
	v_add_f32_e32 v1, v99, v1
	v_lshl_add_u32 v66, v98, 2, 0
	ds_write_b32 v66, v1
	v_mov_b32_e32 v1, 2
	v_lshlrev_b32_sdwa v1, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v66, v66
	s_sub_i32 s2, 0x100, s2
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	v_bfrev_b32_e32 v67, 1
	s_and_b64 vcc, s[0:1], s[2:3]
	s_and_b32 s5, s12, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e32 v1, v67, v1, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v1, s[4:7], 0 offen
	s_cbranch_execz .LBB0_33
	s_branch .LBB0_34
.LBB0_32:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_33:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v253
	v_mov_b32_e32 v1, 0x42000000
	s_nop 0
	v_cndmask_b32_e64 v66, 0, 32, vcc
	v_ldexp_f32 v66, v253, v66
	v_log_f32_e32 v66, v66
	v_cndmask_b32_e32 v1, 0, v1, vcc
	s_barrier
	v_sub_f32_e32 v1, v66, v1
	v_add_f32_e32 v1, v99, v1
	v_lshl_add_u32 v66, v98, 2, 0
	ds_write_b32 v66, v1
	v_mov_b32_e32 v1, 2
	v_lshlrev_b32_sdwa v0, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v1, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v1, v1
	v_bfrev_b32_e32 v66, 1
	s_and_b32 s5, s12, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e64 v0, v66, v0, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v1, v0, s[4:7], 0 offen
.LBB0_34:
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
	v_mul_lo_u32 v66, s43, v98
	s_bitset1_b32 s2, 14
	s_mov_b32 s4, 0x5040100
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_perm_b32 v0, v63, v62, s4
	v_add_lshl_u32 v62, v66, v252, 1
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
.LBB0_35:                               ; %.critedge
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 132
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
	.set attn_fwd.private_seg_size, 132
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 19796
; TotalNumSgprs: 82
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 132
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
	.quad	0
	.quad	0
.Ldebug_ranges3:
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
    .private_segment_fixed_size: 132
    .sgpr_count:     82
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 32
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
