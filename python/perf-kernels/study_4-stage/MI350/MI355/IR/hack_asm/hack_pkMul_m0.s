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
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.12:
.LBB0_0:
	s_ashr_i32 s19, s16, 31
	s_lshr_b32 s19, s19, 29
	s_add_i32 s19, s16, s19
	s_ashr_i32 s19, s19, 3
	s_mul_i32 s24, s12, s18
	s_lshl_b32 s16, s16, 3
	s_mulk_i32 s19, 0xffc1
	s_ashr_i32 s25, s24, 31
	s_add_i32 s19, s19, s16
	s_lshl_b32 s16, s17, 8
	s_lshl_b64 s[24:25], s[24:25], 1
	s_add_u32 s12, s2, s24
	s_mul_i32 s2, s13, s19
	s_addc_u32 s17, s3, s25
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s16
	s_addc_u32 s13, s17, s3
	s_ashr_i32 s3, s2, 31
	s_load_dwordx4 s[20:23], s[0:1], 0x38
	s_load_dword s33, s[0:1], 0x48
	v_lshrrev_b32_e32 v190, 4, v0
	s_lshl_b32 s25, s14, 6
	s_lshl_b64 s[2:3], s[2:3], 1
	v_and_b32_e32 v38, 15, v0
	v_or_b32_e32 v2, 0xa0, v190
	v_or_b32_e32 v3, 0xe0, v190
	s_add_u32 s12, s12, s2
	v_lshlrev_b32_e32 v194, 3, v38
	s_mul_i32 s34, s15, s18
	v_or_b32_e32 v9, s16, v2
	v_or_b32_e32 v11, s16, v3
	v_mul_lo_u32 v13, s14, v2
	v_mul_lo_u32 v14, s14, v3
	s_addc_u32 s13, s13, s3
	v_mad_u64_u32 v[2:3], s[2:3], s14, v190, v[194:195]
	s_ashr_i32 s35, s34, 31
	v_or_b32_e32 v191, 32, v190
	v_or_b32_e32 v1, 0x60, v190
	s_lshl_b64 s[2:3], s[34:35], 1
	v_or_b32_e32 v7, s16, v1
	v_mul_lo_u32 v12, s14, v191
	v_mul_lo_u32 v1, s14, v1
	s_add_u32 s14, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s36, s20, s19
	s_addc_u32 s15, s5, s3
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[2:3], s[36:37], 1
	s_add_u32 s24, s14, s2
	s_mul_i32 s14, s22, s18
	s_addc_u32 s3, s15, s3
	s_ashr_i32 s15, s14, 31
	s_lshl_b32 s17, s21, 3
	s_lshl_b64 s[14:15], s[14:15], 1
	s_add_u32 s2, s6, s14
	s_mul_i32 s6, s23, s19
	s_addc_u32 s14, s7, s15
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	v_or_b32_e32 v4, s16, v190
	s_add_u32 s20, s2, s6
	s_movk_i32 s2, 0x4000
	v_or_b32_e32 v5, s16, v191
	v_add_u32_e32 v3, s25, v2
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v16, 1
	v_cmp_gt_i32_e32 vcc, s2, v4
	v_or_b32_e32 v6, 64, v4
	v_or_b32_e32 v8, 0x80, v4
	v_cndmask_b32_e32 v34, v16, v2, vcc
	v_add_lshl_u32 v2, v12, v194, 1
	v_cmp_gt_i32_e32 vcc, s2, v5
	v_add_u32_e32 v15, s25, v3
	v_add_lshl_u32 v1, v1, v194, 1
	v_cndmask_b32_e32 v35, v16, v2, vcc
	v_lshlrev_b32_e32 v2, 1, v3
	v_cmp_gt_i32_e32 vcc, s2, v6
	v_or_b32_e32 v10, 0xc0, v4
	s_addc_u32 s38, s14, s7
	v_cndmask_b32_e32 v36, v16, v2, vcc
	v_cmp_gt_i32_e32 vcc, s2, v7
	v_lshlrev_b32_e32 v2, 1, v15
	s_and_b32 s13, s13, 0xffff
	v_cndmask_b32_e32 v1, v16, v1, vcc
	v_cmp_gt_i32_e32 vcc, s2, v8
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_cndmask_b32_e32 v37, v16, v2, vcc
	v_add_lshl_u32 v2, v13, v194, 1
	v_cmp_gt_i32_e32 vcc, s2, v9
	v_and_b32_e32 v207, 0x1c0, v0
	v_mov_b32_e32 v51, 0x2000
	v_cndmask_b32_e32 v39, v16, v2, vcc
	v_add_lshl_u32 v2, v15, s25, 1
	v_cmp_gt_i32_e32 vcc, s2, v10
	v_and_b32_e32 v206, 31, v0
	v_lshlrev_b32_e32 v198, 4, v38
	v_cndmask_b32_e32 v40, v16, v2, vcc
	v_add_lshl_u32 v2, v14, v194, 1
	v_cmp_gt_i32_e32 vcc, s2, v11
	s_movk_i32 s22, 0xa0
	s_movk_i32 s23, 0xe0
	v_cndmask_b32_e32 v41, v16, v2, vcc
	buffer_load_dwordx4 v[2:5], v34, s[12:15], 0 offen
	buffer_load_dwordx4 v[6:9], v35, s[12:15], 0 offen
	buffer_load_dwordx4 v[10:13], v36, s[12:15], 0 offen
	buffer_load_dwordx4 v[14:17], v1, s[12:15], 0 offen
	buffer_load_dwordx4 v[18:21], v37, s[12:15], 0 offen
	buffer_load_dwordx4 v[22:25], v39, s[12:15], 0 offen
	buffer_load_dwordx4 v[26:29], v40, s[12:15], 0 offen
	buffer_load_dwordx4 v[30:33], v41, s[12:15], 0 offen
	v_and_b32_e32 v1, 32, v0
	v_and_b32_e32 v39, 16, v0
	v_lshrrev_b32_e32 v35, 6, v0
	v_lshlrev_b32_e32 v34, 1, v39
	v_lshrrev_b32_e32 v40, 1, v1
	v_or3_b32 v36, v35, v34, v40
	v_mad_u64_u32 v[36:37], s[6:7], s21, v36, v[194:195]
	v_lshlrev_b32_e32 v41, 7, v207
	v_lshl_or_b32 v51, v35, 10, v51
	s_movk_i32 s13, 0x60
	s_movk_i32 s25, 0x80
	s_movk_i32 s26, 0xc0
	v_lshl_or_b32 v41, v206, 8, v41
	v_xor_b32_e32 v42, v198, v40
	s_movk_i32 s6, 0x410
	v_lshrrev_b32_e32 v52, 6, v51
	s_lshl_b32 s12, s21, 6
	v_or_b32_e32 v43, v41, v42
	v_bitop3_b32 v46, v41, s13, v42 bitop3:0x36
	v_bitop3_b32 v47, v41, s25, v42 bitop3:0x36
	v_bitop3_b32 v48, v41, s22, v42 bitop3:0x36
	v_bitop3_b32 v49, v41, s26, v42 bitop3:0x36
	v_bitop3_b32 v41, v41, s23, v42 bitop3:0x36
	v_mad_u32_u24 v42, v35, s6, 0
	v_or_b32_e32 v193, v52, v51
        v_readfirstlane_b32 s47, v193
	v_add_u32_e32 v50, 0x87c0, v42
	v_add_u32_e32 v52, 0, v193
	s_ashr_i32 s13, s12, 31
	s_lshl_b32 s39, s33, 3
	s_lshl_b32 s2, s33, 6
	v_lshlrev_b32_e32 v199, 4, v0
	v_and_b32_e32 v37, 0xf0, v0
	s_and_b32 s25, s3, 0xffff
	v_add_u32_e32 v53, 0x87c0, v52
	s_lshl_b64 s[6:7], s[12:13], 1
	v_readfirstlane_b32 s13, v50
	v_xad_u32 v37, v199, v37, 0
	v_add_u32_e32 v44, 0, v43
	s_mov_b32 s26, s14
	s_mov_b32 s27, s15
	v_lshlrev_b32_e32 v192, 1, v36
	v_add_lshl_u32 v195, v36, s17, 1
	s_add_u32 s28, s24, s6
	v_add_u32_e32 v36, 0xc8c0, v42
	v_lshlrev_b32_e32 v38, 10, v38
	v_lshlrev_b32_e32 v39, 5, v39
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s44, v53
	v_xad_u32 v45, v43, 32, 0
	v_xad_u32 v43, v43, 64, 0
	v_add_u32_e32 v46, 0, v46
	v_add_u32_e32 v47, 0, v47
	v_add_u32_e32 v48, 0, v48
	v_add_u32_e32 v49, 0, v49
	v_add_u32_e32 v41, 0, v41
	s_addc_u32 s3, s3, s7
	v_add_u32_e32 v42, 0xc8c0, v52
	v_or3_b32 v38, v38, v39, v40
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v37, v[2:5]
	s_waitcnt vmcnt(6)
	ds_write_b128 v37, v[6:9] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v37, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v37, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v37, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v37, v[22:25] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v37, v[26:29] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v37, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[158:161], v44
	ds_read_b128 v[154:157], v45
	ds_read_b128 v[150:153], v43
	ds_read_b128 v[146:149], v46
	ds_read_b128 v[142:145], v47
	ds_read_b128 v[138:141], v48
	ds_read_b128 v[134:137], v49
	ds_read_b128 v[130:133], v41
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v192, s[24:27], 0 offen lds
	s_mov_b32 m0, s44
	v_readfirstlane_b32 s17, v36
	s_and_b32 s29, s3, 0xffff
	s_mov_b32 s30, s14
	s_mov_b32 s31, s15
	v_add_u32_e32 v202, v38, v198
	buffer_load_dwordx4 v195, s[24:27], 0 offen lds
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s17, v42
	v_add_u32_e32 v52, 0, v202
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v192, s[28:31], 0 offen lds
	s_mov_b32 m0, s17
	v_and_b32_e32 v209, 48, v0
	buffer_load_dwordx4 v195, s[28:31], 0 offen lds
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[2:5], v52 offset:34752
	ds_read_b128 v[18:21], v52 offset:34784
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[158:161], 0
	v_or_b32_e32 v40, v35, v209
	v_mad_u64_u32 v[44:45], s[22:23], s33, v40, v[194:195]
	s_and_b32 s21, s38, 0xffff
	s_add_u32 s40, s28, s6
	v_lshrrev_b32_e32 v45, 4, v51
	s_movk_i32 s24, 0x440
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[154:157], v[2:17]
	ds_read_b128 v[18:21], v52 offset:34816
	ds_read_b128 v[22:25], v52 offset:34848
	s_addc_u32 s3, s3, s7
	v_mad_u32_u24 v46, v35, s24, 0
	v_or_b32_e32 v200, v45, v51
        v_readfirstlane_b32 s49, v200
	s_and_b32 s41, s3, 0xffff
	s_ashr_i32 s3, s2, 31
	v_add_u32_e32 v45, 0, v200
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[150:153], v[2:17]
	s_lshl_b64 s[28:29], s[2:3], 1
	v_readfirstlane_b32 s3, v46
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	v_lshlrev_b32_e32 v196, 1, v44
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v45
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[146:149], v[2:17]
	ds_read_b128 v[18:21], v52 offset:34880
	ds_read_b128 v[22:25], v52 offset:34912
	v_add_lshl_u32 v197, v44, s39, 1
	s_mov_b32 s42, s14
	s_mov_b32 s43, s15
	s_add_u32 s30, s20, s28
	v_add_u32_e32 v44, 0x4400, v46
	s_addc_u32 s31, s38, s29
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[142:145], v[2:17]
	v_add_u32_e32 v47, 0x4400, v45
	s_and_b32 s25, s31, 0xffff
	s_mov_b32 s24, s30
	s_movk_i32 s2, 0x100
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_movk_i32 s2, 0xff
	s_mov_b32 s17, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[138:141], v[2:17]
	ds_read_b128 v[18:21], v52 offset:34944
	ds_read_b128 v[22:25], v52 offset:34976
	ds_read_b128 v[36:39], v52 offset:35040
	ds_read_b128 v[40:43], v52 offset:35072
	buffer_load_dwordx4 v196, s[20:23], 0 offen lds
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v44
	buffer_load_dwordx4 v197, s[20:23], 0 offen lds
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[134:137], v[2:17]
	ds_read_b128 v[18:21], v52 offset:35008
	s_mov_b32 m0, s13
	s_mov_b32 s23, 0x3e0293ee
	buffer_load_dwordx4 v192, s[40:43], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v195, s[40:43], 0 offen lds
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[130:133], v[2:17]
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v47
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[158:161], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[36:39], v[154:157], v[18:33]
	ds_read_b128 v[36:39], v52 offset:35104
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[150:153], v[18:33]
	ds_read_b128 v[40:43], v52 offset:35136
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[36:39], v[146:149], v[18:33]
	ds_read_b128 v[36:39], v52 offset:35168
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[142:145], v[18:33]
	ds_read_b128 v[40:43], v52 offset:35200
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[36:39], v[138:141], v[18:33]
	ds_read_b128 v[36:39], v52 offset:35232
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v196, s[24:27], 0 offen lds
	s_mov_b32 m0, s3
	v_cmp_lt_u32_e64 s[2:3], s2, v0
	buffer_load_dwordx4 v197, s[24:27], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[134:137], v[18:33]
	ds_read_b128 v[66:69], v52 offset:51392
	ds_read_b128 v[186:189], v52 offset:51424
	ds_read_b128 v[182:185], v52 offset:51456
	ds_read_b128 v[178:181], v52 offset:51488
	ds_read_b128 v[122:125], v52 offset:51520
	ds_read_b128 v[118:121], v52 offset:51552
	ds_read_b128 v[114:117], v52 offset:51584
	ds_read_b128 v[110:113], v52 offset:51616
	ds_read_b128 v[126:129], v52 offset:51648
	ds_read_b128 v[174:177], v52 offset:51680
	ds_read_b128 v[170:173], v52 offset:51712
	ds_read_b128 v[166:169], v52 offset:51744
	ds_read_b128 v[162:165], v52 offset:51776
	ds_read_b128 v[106:109], v52 offset:51808
	ds_read_b128 v[102:105], v52 offset:51840
	ds_read_b128 v[98:101], v52 offset:51872
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[18:33], v[36:39], v[130:133], v[18:33]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_maximum3_f32 v36, v2, v3, v3
	v_maximum3_f32 v36, v36, v4, v5
	v_maximum3_f32 v36, v36, v6, v7
	v_maximum3_f32 v36, v36, v8, v9
	v_maximum3_f32 v36, v36, v10, v11
	v_maximum3_f32 v36, v36, v12, v13
	v_maximum3_f32 v36, v36, v14, v15
	v_maximum3_f32 v36, v36, v16, v17
	s_nop 3
	v_maximum3_f32 v36, v36, v18, v19
	v_maximum3_f32 v36, v36, v20, v21
	v_maximum3_f32 v36, v36, v22, v23
	v_maximum3_f32 v36, v36, v24, v25
	v_maximum3_f32 v36, v36, v26, v27
	v_maximum3_f32 v36, v36, v28, v29
	v_maximum3_f32 v36, v36, v30, v31
	v_maximum3_f32 v36, v36, v32, v33
	v_mov_b32_e32 v37, v36
	s_nop 1
	v_permlane32_swap_b32_e32 v36, v37
	v_maximum3_f32 v217, v36, v37, v37
	v_mul_f32_e32 v36, 0xbe0293ee, v217
	v_fmamk_f32 v2, v2, 0x3e0293ee, v36
	v_fmamk_f32 v3, v3, 0x3e0293ee, v36
	v_fmamk_f32 v4, v4, 0x3e0293ee, v36
	v_fmamk_f32 v5, v5, 0x3e0293ee, v36
	v_fmamk_f32 v6, v6, 0x3e0293ee, v36
	v_fmamk_f32 v7, v7, 0x3e0293ee, v36
	v_fmamk_f32 v8, v8, 0x3e0293ee, v36
	v_fmamk_f32 v9, v9, 0x3e0293ee, v36
	v_fmamk_f32 v10, v10, 0x3e0293ee, v36
	v_fmamk_f32 v11, v11, 0x3e0293ee, v36
	v_fmamk_f32 v12, v12, 0x3e0293ee, v36
	v_fmamk_f32 v13, v13, 0x3e0293ee, v36
	v_fmamk_f32 v14, v14, 0x3e0293ee, v36
	v_fmamk_f32 v15, v15, 0x3e0293ee, v36
	v_fmamk_f32 v16, v16, 0x3e0293ee, v36
	v_fmamk_f32 v17, v17, 0x3e0293ee, v36
	v_fmamk_f32 v18, v18, 0x3e0293ee, v36
	v_fmamk_f32 v19, v19, 0x3e0293ee, v36
	v_fmamk_f32 v20, v20, 0x3e0293ee, v36
	v_fmamk_f32 v21, v21, 0x3e0293ee, v36
	v_fmamk_f32 v22, v22, 0x3e0293ee, v36
	v_fmamk_f32 v23, v23, 0x3e0293ee, v36
	v_fmamk_f32 v24, v24, 0x3e0293ee, v36
	v_fmamk_f32 v25, v25, 0x3e0293ee, v36
	v_fmamk_f32 v26, v26, 0x3e0293ee, v36
	v_fmamk_f32 v27, v27, 0x3e0293ee, v36
	v_fmamk_f32 v28, v28, 0x3e0293ee, v36
	v_fmamk_f32 v29, v29, 0x3e0293ee, v36
	v_fmamk_f32 v30, v30, 0x3e0293ee, v36
	v_fmamk_f32 v31, v31, 0x3e0293ee, v36
	v_fmamk_f32 v32, v32, 0x3e0293ee, v36
	v_fmac_f32_e32 v36, 0x3e0293ee, v33
	v_mov_b32_e32 v33, 0xff800000
	v_fmac_f32_e32 v33, 0xbe0293ee, v217
	s_and_saveexec_b64 s[20:21], s[2:3]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:                                ; %.preheader
	s_or_b64 exec, exec, s[20:21]
	s_load_dwordx2 s[20:21], s[0:1], 0x4c
	s_load_dword s22, s[0:1], 0x54
	v_exp_f32_e32 v243, v2
	v_exp_f32_e32 v228, v3
	v_lshlrev_b32_e32 v2, 8, v0
	v_lshlrev_b32_e32 v3, 3, v0
	s_add_u32 s0, s34, s36
	v_exp_f32_e32 v235, v4
	v_and_b32_e32 v2, 0xc00, v2
	v_and_b32_e32 v3, 24, v3
	v_lshlrev_b32_e32 v4, 7, v1
	s_addc_u32 s1, s35, s37
	v_or3_b32 v2, v2, v3, v4
	s_mul_i32 s3, s12, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_or_b32_e32 v214, v2, v34
	v_lshrrev_b32_e32 v3, 4, v2
	v_or_b32_e32 v2, 0x2000, v2
	s_mul_hi_i32 s2, s12, 6
	s_add_u32 s0, s3, s0
	v_lshrrev_b32_e32 v2, 4, v2
	s_addc_u32 s1, s2, s1
	v_exp_f32_e32 v229, v5
	v_exp_f32_e32 v231, v6
	v_exp_f32_e32 v230, v7
	v_exp_f32_e32 v239, v8
	v_exp_f32_e32 v233, v9
	v_exp_f32_e32 v232, v10
	v_exp_f32_e32 v234, v11
	v_exp_f32_e32 v237, v12
	v_exp_f32_e32 v236, v13
	v_exp_f32_e32 v244, v14
	v_exp_f32_e32 v238, v15
	v_exp_f32_e32 v245, v16
	v_exp_f32_e32 v225, v17
	v_exp_f32_e32 v224, v18
	v_exp_f32_e32 v226, v19
	v_exp_f32_e32 v223, v20
	v_exp_f32_e32 v240, v21
	v_exp_f32_e32 v219, v22
	v_exp_f32_e32 v220, v23
	v_exp_f32_e32 v221, v24
	v_exp_f32_e32 v241, v25
	v_exp_f32_e32 v218, v26
	v_exp_f32_e32 v222, v27
	v_exp_f32_e32 v242, v28
	v_exp_f32_e32 v210, v29
	v_exp_f32_e32 v211, v30
	v_exp_f32_e32 v212, v31
	v_exp_f32_e32 v204, v32
	v_exp_f32_e32 v213, v36
	v_exp_f32_e32 v205, v33
	v_and_b32_e32 v216, 0x3c0, v2
	s_add_u32 s0, s4, s0
	v_mov_b32_e32 v2, 0
	v_mul_u32_u24_e32 v208, 0x410, v35
        v_readfirstlane_b32 s46, v208
	v_mul_u32_u24_e32 v203, 0x440, v35
        v_readfirstlane_b32 s48, v203
	v_and_b32_e32 v215, 0x1c0, v3
	s_addc_u32 s1, s5, s1
	s_add_i32 s2, 0, 0x4400
	s_add_i32 s4, 0, 0x87c0
	v_mov_b32_e32 v201, 1.0
	s_movk_i32 s3, 0xffc0
	s_mov_b32 s5, 0
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
	v_mov_b32_e32 v227, v201
	s_mov_b32 s35, s17
	s_mov_b32 s17, s2
	v_mov_b32_e32 v252, v217
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[158:161], 0
	v_mul_f32_e32 v48, v48, v205
	v_add_f32_e32 v82, v243, v228
	v_mul_f32_e32 v40, v40, v205
	v_add_f32_e32 v82, v82, v235
	v_mul_f32_e32 v34, v34, v205
	v_add_f32_e32 v82, v82, v229
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[154:157], v[66:81]
	v_add_f32_e32 v82, v82, v231
	v_add_f32_e32 v82, v82, v230
	v_add_f32_e32 v201, v82, v239
	v_mul_f32_e32 v65, v65, v205
	v_mul_f32_e32 v17, v17, v205
	v_mul_f32_e32 v64, v64, v205
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[158:161], 0
	v_mul_f32_e32 v62, v62, v205
	v_cvt_pk_f16_f32 v129, v204, v213
	v_cvt_pk_f16_f32 v128, v211, v212
	v_cvt_pk_f16_f32 v127, v242, v210
	v_add_f32_e32 v126, v201, v233
	v_mul_f32_e32 v63, v63, v205
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[150:153], v[66:81]
	v_mul_f32_e32 v61, v61, v205
	v_add_f32_e32 v126, v126, v232
	v_mul_f32_e32 v60, v60, v205
	v_add_f32_e32 v126, v126, v234
	v_mul_f32_e32 v59, v59, v205
	v_add_f32_e32 v126, v126, v237
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[146:149], v[66:81]
	v_add_f32_e32 v126, v126, v236
	v_mul_f32_e32 v58, v58, v205
	v_add_f32_e32 v126, v126, v244
	v_mul_f32_e32 v57, v57, v205
	v_add_f32_e32 v126, v126, v238
	v_mul_f32_e32 v56, v56, v205
	v_mfma_f32_32x32x16_f16 v[66:81], v[122:125], v[142:145], v[66:81]
	v_mul_f32_e32 v55, v55, v205
	v_add_f32_e32 v126, v126, v245
	v_mul_f32_e32 v54, v54, v205
	v_add_f32_e32 v126, v126, v225
	v_mul_f32_e32 v53, v53, v205
	v_add_f32_e32 v126, v126, v224
	v_mfma_f32_32x32x16_f16 v[66:81], v[118:121], v[138:141], v[66:81]
	v_add_f32_e32 v126, v126, v226
	v_mul_f32_e32 v52, v52, v205
	v_add_f32_e32 v126, v126, v223
	v_mul_f32_e32 v51, v51, v205
	v_add_f32_e32 v126, v126, v240
	v_mul_f32_e32 v50, v50, v205
	v_mfma_f32_32x32x16_f16 v[66:81], v[114:117], v[134:137], v[66:81]
	v_add_f32_e32 v126, v126, v219
	v_mul_f32_e32 v16, v16, v205
	v_add_f32_e32 v126, v126, v220
	v_mul_f32_e32 v35, v35, v205
	v_add_f32_e32 v126, v126, v221
	v_mul_f32_e32 v36, v36, v205
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[154:157], v[82:97]
	v_mul_f32_e32 v37, v37, v205
	v_add_f32_e32 v126, v126, v241
	v_mul_f32_e32 v38, v38, v205
	v_add_f32_e32 v126, v126, v218
	v_mul_f32_e32 v39, v39, v205
	v_add_f32_e32 v126, v126, v222
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[150:153], v[82:97]
	v_add_f32_e32 v126, v126, v242
	v_mul_f32_e32 v41, v41, v205
	v_add_f32_e32 v126, v126, v210
	v_mul_f32_e32 v42, v42, v205
	v_add_f32_e32 v126, v126, v211
	v_mul_f32_e32 v43, v43, v205
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[146:149], v[82:97]
	v_mul_f32_e32 v44, v44, v205
	v_add_f32_e32 v126, v126, v212
	v_mul_f32_e32 v45, v45, v205
	v_add_f32_e32 v126, v126, v204
	v_mul_f32_e32 v46, v46, v205
	v_add_f32_e32 v126, v126, v213
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[142:145], v[82:97]
	v_mul_f32_e32 v49, v49, v205
	v_cvt_pk_f16_f32 v165, v239, v233
	v_cvt_pk_f16_f32 v164, v231, v230
	v_cvt_pk_f16_f32 v163, v235, v229
	v_mov_b32_e32 v162, v126
	v_mul_f32_e32 v47, v47, v205
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[138:141], v[82:97]
	v_cvt_pk_f16_f32 v109, v221, v241
	v_cvt_pk_f16_f32 v108, v219, v220
	v_cvt_pk_f16_f32 v107, v223, v240
	v_cvt_pk_f16_f32 v106, v224, v226
	v_permlane32_swap_b32_e32 v126, v162
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[134:137], v[82:97]
	v_add_f32_e32 v201, v126, v162
	v_cvt_pk_f16_f32 v105, v245, v225
	v_cvt_pk_f16_f32 v162, v243, v228
	v_cvt_pk_f16_f32 v126, v218, v222
	v_fmac_f32_e32 v201, v227, v205
	v_cvt_pk_f16_f32 v104, v244, v238
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[130:133], v[66:81]
	v_cvt_pk_f16_f32 v103, v237, v236
	v_cvt_pk_f16_f32 v102, v232, v234
	v_mul_f32_e32 v20, v20, v205
	v_mul_f32_e32 v22, v22, v205
	v_mul_f32_e32 v21, v21, v205
	v_mul_f32_e32 v23, v23, v205
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[130:133], v[82:97]
	v_mul_f32_e32 v25, v25, v205
	v_mul_f32_e32 v26, v26, v205
	v_mul_f32_e32 v28, v28, v205
	v_mul_f32_e32 v27, v27, v205
	v_mul_f32_e32 v29, v29, v205
	v_mul_f32_e32 v24, v24, v205

        v_pk_mul_f32 v[18:19], v[204:205], v[18:19] op_sel:[1,0]
        v_pk_mul_f32 v[2:3], v[204:205], v[2:3] op_sel:[1,0]
        v_pk_mul_f32 v[4:5], v[204:205], v[4:5] op_sel:[1,0]
        v_pk_mul_f32 v[6:7], v[204:205], v[6:7] op_sel:[1,0]
        v_pk_mul_f32 v[8:9], v[204:205], v[8:9] op_sel:[1,0]
        v_pk_mul_f32 v[10:11], v[204:205], v[10:11] op_sel:[1,0]
        v_pk_mul_f32 v[12:13], v[204:205], v[12:13] op_sel:[1,0]
        v_pk_mul_f32 v[14:15], v[204:205], v[14:15] op_sel:[1,0]
        v_pk_mul_f32 v[30:31], v[204:205], v[30:31] op_sel:[1,0]
        v_pk_mul_f32 v[32:33], v[204:205], v[32:33] op_sel:[1,0]
	; iglp_opt mask(0x0000000A)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_add_i32 s2, s5, 1
	s_cmp_lt_i32 s2, 2
	s_cselect_b32 s27, s2, 0
	s_lshl_b32 s25, s27, 13
	s_lshl_b32 s2, s27, 14
	s_add_i32 s26, s2, 0
	s_ashr_i32 s2, s25, 5
	s_add_i32 s24, s26, s2
	s_add_i32 s34, s24, 0x87c0

        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ;; Old pattern
        ;; v_mul_u32_u24_e32 v208, 0x410, v35
        ;; loop:
        ;;   v_add_u32_e32 v98, s34, v208
        ;;   v_readfirstlane_b32 s2, v98
        ;;   s_mov_b32 m0, s2
        ;;
        ;; New pattern
        ;; v_mul_u32_u24_e32 v208, 0x410, v35
        ;; v_readfirstlane_b32 s46, v208
        ;; loop:
        ;;   s_add_i32 m0, s34, s46
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        s_add_i32 m0, s34, s46
	;;v_add_u32_e32 v98, s34, v208
	;;s_nop 0
	;;v_readfirstlane_b32 s2, v98
	;;v_add_u32_e32 v98, s34, v193
	s_and_b32 s13, s1, 0xffff
	s_mov_b32 s12, s0
	;;s_mov_b32 m0, s2
	;;v_readfirstlane_b32 s2, v98
	buffer_load_dwordx4 v192, s[12:15], 0 offen lds
	;; s_mov_b32 m0, s2
        s_add_i32 m0, s34, s47
	s_nop 0
	buffer_load_dwordx4 v195, s[12:15], 0 offen lds
	v_add_u32_e32 v98, s35, v214
	v_add_u32_e32 v204, v98, v215
	v_add_u32_e32 v205, v98, v216
	ds_read_b64_tr_b16 v[98:99], v204
	ds_read_b64_tr_b16 v[110:111], v204 offset:64
	ds_read_b64_tr_b16 v[114:115], v204 offset:128
	ds_read_b64_tr_b16 v[118:119], v204 offset:192
	ds_read_b64_tr_b16 v[100:101], v205 offset:8192
	ds_read_b64_tr_b16 v[112:113], v205 offset:8256
	ds_read_b64_tr_b16 v[116:117], v205 offset:8320
	ds_read_b64_tr_b16 v[120:121], v205 offset:8384
	ds_read_b64_tr_b16 v[122:123], v204 offset:256
	ds_read_b64_tr_b16 v[166:167], v204 offset:320
	ds_read_b64_tr_b16 v[170:171], v204 offset:384
	ds_read_b64_tr_b16 v[174:175], v204 offset:448
	ds_read_b64_tr_b16 v[124:125], v205 offset:8448
	ds_read_b64_tr_b16 v[168:169], v205 offset:8512
	ds_read_b64_tr_b16 v[172:173], v205 offset:8576
	ds_read_b64_tr_b16 v[176:177], v205 offset:8640
	ds_read_b64_tr_b16 v[178:179], v204 offset:512
	ds_read_b64_tr_b16 v[182:183], v204 offset:576
	ds_read_b64_tr_b16 v[186:187], v204 offset:640
	ds_read_b64_tr_b16 v[210:211], v204 offset:704
	ds_read_b64_tr_b16 v[180:181], v205 offset:8704
	ds_read_b64_tr_b16 v[184:185], v205 offset:8768
	ds_read_b64_tr_b16 v[188:189], v205 offset:8832
	ds_read_b64_tr_b16 v[212:213], v205 offset:8896
	ds_read_b64_tr_b16 v[236:237], v204 offset:768
	ds_read_b64_tr_b16 v[240:241], v204 offset:832
	ds_read_b64_tr_b16 v[244:245], v204 offset:896
	ds_read_b64_tr_b16 v[238:239], v205 offset:8960
	ds_read_b64_tr_b16 v[242:243], v205 offset:9024
	ds_read_b64_tr_b16 v[248:249], v204 offset:960
	ds_read_b64_tr_b16 v[246:247], v205 offset:9088
	ds_read_b64_tr_b16 v[250:251], v205 offset:9152
	; sched_barrier mask(0x00000000)
	s_setprio 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_f16 v[50:65], v[98:101], v[162:165], v[50:65]
	v_maximum3_f32 v98, v66, v67, v67
	v_maximum3_f32 v98, v98, v68, v69
	v_maximum3_f32 v98, v98, v70, v71
	v_maximum3_f32 v98, v98, v72, v73
	v_maximum3_f32 v98, v98, v74, v75
	v_maximum3_f32 v98, v98, v76, v77
	v_mfma_f32_32x32x16_f16 v[34:49], v[110:113], v[162:165], v[34:49]
	v_maximum3_f32 v98, v98, v78, v79
	v_maximum3_f32 v98, v98, v80, v81
	v_maximum3_f32 v98, v98, v82, v83
	v_maximum3_f32 v98, v98, v84, v85
	v_maximum3_f32 v98, v98, v86, v87
	v_maximum3_f32 v98, v98, v88, v89
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[102:105], v[34:49]
	v_maximum3_f32 v98, v98, v90, v91
	v_maximum3_f32 v98, v98, v92, v93
	v_maximum3_f32 v98, v98, v94, v95
	v_maximum3_f32 v98, v98, v96, v97
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_mfma_f32_32x32x16_f16 v[34:49], v[182:185], v[106:109], v[34:49]
	v_maximum3_f32 v217, v252, v98, v99
	v_mul_f32_e32 v227, 0x3e0293ee, v217
	v_fma_f32 v67, v67, s23, -v227
	v_fma_f32 v98, v252, s23, -v227
	v_fma_f32 v68, v68, s23, -v227
	v_fma_f32 v69, v69, s23, -v227
	v_mfma_f32_32x32x16_f16 v[34:49], v[240:243], v[126:129], v[34:49]
	v_fma_f32 v70, v70, s23, -v227
	v_fma_f32 v66, v66, s23, -v227
	v_fma_f32 v71, v71, s23, -v227
	v_fma_f32 v73, v73, s23, -v227
	v_fma_f32 v74, v74, s23, -v227
	v_fma_f32 v75, v75, s23, -v227
	v_mfma_f32_32x32x16_f16 v[18:33], v[114:117], v[162:165], v[18:33]
	v_fma_f32 v81, v81, s23, -v227
	v_fma_f32 v82, v82, s23, -v227
	v_fma_f32 v83, v83, s23, -v227
	v_fma_f32 v84, v84, s23, -v227
	v_fma_f32 v85, v85, s23, -v227
	v_fma_f32 v86, v86, s23, -v227
	v_mfma_f32_32x32x16_f16 v[2:17], v[118:121], v[162:165], v[2:17]
	v_fma_f32 v87, v87, s23, -v227
	v_fma_f32 v88, v88, s23, -v227
	v_fma_f32 v89, v89, s23, -v227
	v_fma_f32 v90, v90, s23, -v227
	v_fma_f32 v91, v91, s23, -v227
	v_fma_f32 v92, v92, s23, -v227
	v_mfma_f32_32x32x16_f16 v[50:65], v[122:125], v[102:105], v[50:65]
	v_fma_f32 v96, v96, s23, -v227
	v_exp_f32_e32 v229, v69
	v_exp_f32_e32 v204, v96
	v_exp_f32_e32 v234, v75
	v_mfma_f32_32x32x16_f16 v[50:65], v[178:181], v[106:109], v[50:65]
	v_exp_f32_e32 v219, v86
	v_exp_f32_e32 v242, v92
	v_exp_f32_e32 v228, v67
	v_mfma_f32_32x32x16_f16 v[50:65], v[236:239], v[126:129], v[50:65]
	v_fma_f32 v72, v72, s23, -v227
	v_fma_f32 v76, v76, s23, -v227
	v_fma_f32 v77, v77, s23, -v227
	v_fma_f32 v79, v79, s23, -v227
	v_exp_f32_e32 v222, v91
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[102:105], v[18:33]
	v_exp_f32_e32 v238, v79
	v_exp_f32_e32 v205, v98
	v_exp_f32_e32 v221, v88
	v_mfma_f32_32x32x16_f16 v[18:33], v[186:189], v[106:109], v[18:33]
	v_exp_f32_e32 v220, v87
	v_exp_f32_e32 v235, v68
	v_exp_f32_e32 v240, v85
	v_mfma_f32_32x32x16_f16 v[18:33], v[244:247], v[126:129], v[18:33]
	v_fma_f32 v78, v78, s23, -v227
	v_fma_f32 v80, v80, s23, -v227
	v_exp_f32_e32 v226, v83
	v_exp_f32_e32 v245, v80
	v_mfma_f32_32x32x16_f16 v[2:17], v[174:177], v[102:105], v[2:17]
	v_exp_f32_e32 v241, v89
	v_exp_f32_e32 v225, v81
	v_exp_f32_e32 v244, v78
	v_mfma_f32_32x32x16_f16 v[2:17], v[210:213], v[106:109], v[2:17]
	v_fma_f32 v97, v97, s23, -v227
	v_fma_f32 v95, v95, s23, -v227
	v_fma_f32 v94, v94, s23, -v227
	v_fma_f32 v93, v93, s23, -v227
	v_exp_f32_e32 v213, v97
	v_mfma_f32_32x32x16_f16 v[2:17], v[248:251], v[126:129], v[2:17]
	v_exp_f32_e32 v237, v76
	v_exp_f32_e32 v239, v72
	v_exp_f32_e32 v232, v74
	v_exp_f32_e32 v218, v90
	v_exp_f32_e32 v230, v71
	v_exp_f32_e32 v231, v70
	v_exp_f32_e32 v223, v84
	v_exp_f32_e32 v233, v73
	v_exp_f32_e32 v236, v77
	v_exp_f32_e32 v224, v82
	v_exp_f32_e32 v243, v66
	v_exp_f32_e32 v210, v93
	v_exp_f32_e32 v211, v94
	v_exp_f32_e32 v212, v95
	; iglp_opt mask(0x0000000A)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s30, s28
	s_addc_u32 s35, s31, s29
	s_lshl_b32 s2, s5, 13
	s_lshl_b32 s5, s5, 14
	s_add_i32 s5, s5, 0
	s_ashr_i32 s2, s2, 3
	s_add_i32 s2, s5, s2
	;;v_add_u32_e32 v66, s2, v203
	s_and_b32 s13, s35, 0xffff
	;;v_readfirstlane_b32 s5, v66
	;;v_add_u32_e32 v66, s2, v200
	;;s_mov_b32 m0, s5
        s_add_i32 m0, s2, s48
	;;v_readfirstlane_b32 s5, v66
	buffer_load_dwordx4 v196, s[12:15], 0 offen lds
	;;s_mov_b32 m0, s5
        s_add_i32 m0, s2, s49
	v_add_u32_e32 v70, s4, v202
	buffer_load_dwordx4 v197, s[12:15], 0 offen lds
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[186:189], v70 offset:32
	ds_read_b128 v[182:185], v70 offset:64
	ds_read_b128 v[178:181], v70 offset:96
	ds_read_b128 v[122:125], v70 offset:128
	ds_read_b128 v[118:121], v70 offset:160
	ds_read_b128 v[114:117], v70 offset:192
	ds_read_b128 v[110:113], v70 offset:224
	ds_read_b128 v[126:129], v70 offset:256
	ds_read_b128 v[174:177], v70 offset:288
	ds_read_b128 v[170:173], v70 offset:320
	ds_read_b128 v[166:169], v70 offset:352
	ds_read_b128 v[162:165], v70 offset:384
	ds_read_b128 v[106:109], v70 offset:416
	ds_read_b128 v[102:105], v70 offset:448
	ds_read_b128 v[98:101], v70 offset:480
	; sched_barrier mask(0x00000000)
	s_setprio 0
	s_add_u32 s30, s30, s28
	s_addc_u32 s31, s31, s29
	s_add_u32 s0, s0, s6
	s_addc_u32 s1, s1, s7
	s_add_i32 s3, s3, 64
	s_cmpk_lt_u32 s3, 0x3f00
	s_mov_b32 s4, s34
	s_mov_b32 s5, s27
	s_waitcnt lgkmcnt(0)
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[158:161], 0
	s_add_i32 s0, s16, 0xffffc100
	s_add_i32 s6, 0, 0x10a00
	v_or_b32_e32 v66, 5, v194
	v_mul_lo_u32 v69, s33, v190
	v_mul_lo_u32 v74, s33, v191
	s_add_u32 s4, s12, s28
	s_addc_u32 s1, s35, s29
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[154:157], v[82:97]
	v_add_u32_e32 v187, v69, v66
	s_ashr_i32 s3, s25, 3
	s_add_i32 s3, s26, s3
	s_and_b32 s5, s1, 0xffff
	s_cmp_lt_i32 s0, 1
	v_lshrrev_b32_e32 v70, 1, v207
	v_or_b32_e32 v208, v70, v206
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[150:153], v[82:97]
	v_add_u32_e32 v183, v74, v66
	v_mov_b32_e32 v66, 2
	v_or_b32_e32 v70, 1, v194
	v_or_b32_e32 v71, 2, v194
	v_or_b32_e32 v72, 3, v194
	v_or_b32_e32 v73, 4, v194
	v_or_b32_e32 v67, 6, v194
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[146:149], v[82:97]
	v_lshlrev_b32_sdwa v178, v66, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_and_b32_e32 v66, 0x100, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v66
	v_add_f32_e32 v66, v243, v228
	v_add_f32_e32 v66, v66, v235
	v_add_f32_e32 v66, v66, v229
	v_add_f32_e32 v66, v66, v231
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[142:145], v[82:97]
	v_add_f32_e32 v66, v66, v230
	v_add_f32_e32 v66, v66, v239
	v_add_f32_e32 v66, v66, v233
	v_add_f32_e32 v66, v66, v232
	v_add_f32_e32 v66, v66, v234
	v_add_f32_e32 v66, v66, v237
	v_add_f32_e32 v66, v66, v236
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[138:141], v[82:97]
	v_or_b32_e32 v181, v215, v214
	v_add_f32_e32 v66, v66, v244
	v_or_b32_e32 v68, 7, v194
	v_add_f32_e32 v66, v66, v238
	v_add_u32_e32 v76, s17, v181
	v_add_u32_e32 v190, v69, v194
	v_add_u32_e32 v191, v69, v70
	v_mfma_f32_32x32x16_f16 v[82:97], v[114:117], v[134:137], v[82:97]
	v_add_u32_e32 v192, v69, v71
	v_add_u32_e32 v193, v69, v72
	v_add_u32_e32 v186, v69, v73
	v_add_u32_e32 v188, v69, v67
	v_add_u32_e32 v189, v69, v68
	v_add_u32_e32 v184, v74, v67
	v_add_u32_e32 v185, v74, v68
	v_mfma_f32_32x32x16_f16 v[114:129], v[126:129], v[158:161], 0
	v_lshlrev_b32_e32 v246, 2, v207
	v_lshlrev_b32_e32 v179, 2, v206
	v_lshlrev_b32_e32 v180, 1, v207
	v_add_f32_e32 v206, v66, v245
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[114:129], v[174:177], v[154:157], v[114:129]
	v_mul_f32_e32 v50, v50, v205
	v_mul_f32_e32 v51, v51, v205
	v_mul_f32_e32 v52, v52, v205
	v_mul_f32_e32 v53, v53, v205
	v_mul_f32_e32 v54, v54, v205
	v_mul_f32_e32 v55, v55, v205
	v_mul_f32_e32 v56, v56, v205
	v_mfma_f32_32x32x16_f16 v[114:129], v[170:173], v[150:153], v[114:129]
	v_add_u32_e32 v170, v216, v214
	v_add_u32_e32 v207, s17, v170
	ds_read_b64_tr_b16 v[66:67], v76
	ds_read_b64_tr_b16 v[68:69], v207 offset:8192
	v_mul_f32_e32 v57, v57, v205
	v_mul_f32_e32 v58, v58, v205
	v_mul_f32_e32 v59, v59, v205
	v_mul_f32_e32 v60, v60, v205
	v_mfma_f32_32x32x16_f16 v[114:129], v[166:169], v[146:149], v[114:129]
	v_mul_f32_e32 v61, v61, v205
	v_mul_f32_e32 v62, v62, v205
	v_mul_f32_e32 v63, v63, v205
	v_mul_f32_e32 v64, v64, v205
	v_mul_f32_e32 v65, v65, v205
	v_cvt_pk_f16_f32 v166, v243, v228
	v_cvt_pk_f16_f32 v167, v235, v229
	v_mfma_f32_32x32x16_f16 v[114:129], v[162:165], v[142:145], v[114:129]
	v_cvt_pk_f16_f32 v168, v231, v230
	v_cvt_pk_f16_f32 v169, v239, v233
	v_add_u32_e32 v171, v214, v215
	v_add_u32_e32 v214, s17, v171
	v_add_u32_e32 v194, v74, v194
	v_add_u32_e32 v195, v74, v70
	v_add_u32_e32 v196, v74, v71
	v_mfma_f32_32x32x16_f16 v[114:129], v[106:109], v[138:141], v[114:129]
	v_add_u32_e32 v197, v74, v72
	v_add_u32_e32 v182, v74, v73
	ds_read_b64_tr_b16 v[70:71], v214 offset:256
	ds_read_b64_tr_b16 v[72:73], v207 offset:8448
	ds_read_b64_tr_b16 v[74:75], v207 offset:8256
	ds_read_b64_tr_b16 v[76:77], v76 offset:512
	v_cvt_pk_f16_f32 v162, v232, v234
	v_cvt_pk_f16_f32 v163, v237, v236
	v_cvt_pk_f16_f32 v164, v244, v238
	v_cvt_pk_f16_f32 v165, v245, v225
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[66:69], v[166:169], v[50:65]
	v_mul_f32_e32 v34, v34, v205
	v_mul_f32_e32 v35, v35, v205
	v_mul_f32_e32 v36, v36, v205
	v_mul_f32_e32 v37, v37, v205
	v_mul_f32_e32 v38, v38, v205
	v_mul_f32_e32 v39, v39, v205
	v_mul_f32_e32 v40, v40, v205
	v_mfma_f32_32x32x16_f16 v[114:129], v[102:105], v[134:137], v[114:129]
	v_mul_f32_e32 v41, v41, v205
	v_mul_f32_e32 v42, v42, v205
	v_mul_f32_e32 v43, v43, v205
	v_mul_f32_e32 v44, v44, v205
	v_mul_f32_e32 v45, v45, v205
	v_mul_f32_e32 v46, v46, v205
	v_mul_f32_e32 v47, v47, v205
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[162:165], v[50:65]
	v_mul_f32_e32 v48, v48, v205
	v_mul_f32_e32 v49, v49, v205
	v_add_f32_e32 v70, v206, v225
	v_add_f32_e32 v70, v70, v224
	v_add_f32_e32 v70, v70, v226
	v_cvt_pk_f16_f32 v106, v218, v222
	v_cvt_pk_f16_f32 v107, v242, v210
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[130:133], v[82:97]
	v_cvt_pk_f16_f32 v110, v224, v226
	v_cvt_pk_f16_f32 v111, v223, v240
	v_cvt_pk_f16_f32 v112, v219, v220
	v_cvt_pk_f16_f32 v113, v221, v241
	v_cvt_pk_f16_f32 v108, v211, v212
	v_cvt_pk_f16_f32 v109, v204, v213
	v_add_f32_e32 v70, v70, v223
	v_mfma_f32_32x32x16_f16 v[114:129], v[98:101], v[130:133], v[114:129]
	ds_read_b64_tr_b16 v[78:79], v207 offset:8704
	ds_read_b64_tr_b16 v[66:67], v214 offset:320
	ds_read_b64_tr_b16 v[98:99], v214 offset:384
	ds_read_b64_tr_b16 v[102:103], v214 offset:192
	ds_read_b64_tr_b16 v[68:69], v207 offset:8512
	ds_read_b64_tr_b16 v[176:177], v207 offset:8320
	ds_read_b64_tr_b16 v[104:105], v207 offset:8384
	ds_read_b64_tr_b16 v[172:173], v214 offset:768
	ds_read_b64_tr_b16 v[174:175], v207 offset:8960
	ds_read_b64_tr_b16 v[80:81], v207 offset:8768
	ds_read_b64_tr_b16 v[100:101], v207 offset:8576
	ds_read_b64_tr_b16 v[230:231], v207 offset:8640
	ds_read_b64_tr_b16 v[72:73], v214 offset:64
	ds_read_b64_tr_b16 v[234:235], v207 offset:9024
	ds_read_b64_tr_b16 v[238:239], v207 offset:8832
	ds_read_b64_tr_b16 v[244:245], v207 offset:8896
	ds_read_b64_tr_b16 v[232:233], v214 offset:832
	ds_read_b64_tr_b16 v[248:249], v214 offset:896
	ds_read_b64_tr_b16 v[252:253], v214 offset:960
	v_add_f32_e32 v70, v70, v240
	v_add_f32_e32 v70, v70, v219
	v_add_f32_e32 v70, v70, v220
	v_add_f32_e32 v70, v70, v221
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[76:79], v[110:113], v[50:65]
	v_add_f32_e32 v70, v70, v241
	v_add_f32_e32 v70, v70, v218
	v_mul_f32_e32 v71, v23, v205
	v_mul_f32_e32 v76, v28, v205
	v_mul_f32_e32 v77, v29, v205
	v_mul_f32_e32 v23, v7, v205
	v_mul_f32_e32 v28, v12, v205
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[34:49], v[72:75], v[166:169], v[34:49]
	v_mul_f32_e32 v72, v24, v205
	v_mul_f32_e32 v73, v25, v205
	v_mul_f32_e32 v74, v26, v205
	v_mul_f32_e32 v75, v27, v205
	v_mul_f32_e32 v24, v8, v205
	v_mul_f32_e32 v25, v9, v205
	v_mul_f32_e32 v26, v10, v205
	v_mfma_f32_32x32x16_f16 v[50:65], v[172:175], v[106:109], v[50:65]
	ds_read_b64_tr_b16 v[174:175], v214 offset:128
	ds_read_b64_tr_b16 v[250:251], v207 offset:9088
	ds_read_b64_tr_b16 v[254:255], v207 offset:9152
	ds_read_b64_tr_b16 v[78:79], v214 offset:576
	v_mul_f32_e32 v27, v11, v205
	v_mul_f32_e32 v29, v13, v205
	s_mov_b32 s7, 0x3e0293ee
	v_lshl_add_u32 v209, v209, 8, s6
	v_mfma_f32_32x32x16_f16 v[34:49], v[66:69], v[162:165], v[34:49]
	v_add_f32_e32 v66, v70, v222
	v_add_f32_e32 v172, v66, v242
	v_mul_f32_e32 v66, v18, v205
	v_mul_f32_e32 v18, v2, v205
	v_maximum3_f32 v2, v82, v83, v83
	v_maximum3_f32 v2, v2, v84, v85
	v_mul_f32_e32 v67, v19, v205
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[78:81], v[110:113], v[34:49]
	v_mul_f32_e32 v68, v20, v205
	v_mul_f32_e32 v69, v21, v205
	v_mul_f32_e32 v70, v22, v205
	v_mul_f32_e32 v78, v30, v205
	v_mul_f32_e32 v79, v31, v205
	v_mul_f32_e32 v80, v32, v205
	v_mul_f32_e32 v81, v33, v205
	v_mul_f32_e32 v19, v3, v205
	v_mul_f32_e32 v20, v4, v205
	v_mul_f32_e32 v21, v5, v205
	v_mul_f32_e32 v22, v6, v205
	v_mul_f32_e32 v30, v14, v205
	v_mul_f32_e32 v31, v15, v205
	v_mul_f32_e32 v32, v16, v205
	v_mul_f32_e32 v33, v17, v205
	v_maximum3_f32 v2, v2, v86, v87
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[166:169], v[66:81]
	v_maximum3_f32 v2, v2, v88, v89
	v_maximum3_f32 v2, v2, v90, v91
	v_maximum3_f32 v2, v2, v92, v93
	ds_read_b64_tr_b16 v[236:237], v214 offset:640
	ds_read_b64_tr_b16 v[242:243], v214 offset:704
	ds_read_b64_tr_b16 v[228:229], v214 offset:448
	v_maximum3_f32 v2, v2, v94, v95
	v_maximum3_f32 v2, v2, v96, v97
	v_maximum3_f32 v2, v2, v114, v115
	v_mfma_f32_32x32x16_f16 v[18:33], v[102:105], v[166:169], v[18:33]
	v_maximum3_f32 v2, v2, v116, v117
	v_maximum3_f32 v2, v2, v118, v119
	v_maximum3_f32 v2, v2, v120, v121
	v_maximum3_f32 v2, v2, v122, v123
	v_maximum3_f32 v2, v2, v124, v125
	v_maximum3_f32 v2, v2, v126, v127
	v_maximum3_f32 v2, v2, v128, v129
	v_mfma_f32_32x32x16_f16 v[66:81], v[98:101], v[162:165], v[66:81]
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v2, v3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[18:33], v[228:231], v[162:165], v[18:33]
	v_add_u32_e32 v164, s24, v202
	v_maximum3_f32 v163, v217, v2, v3
	ds_read_b128 v[2:5], v164 offset:34752
	ds_read_b128 v[6:9], v164 offset:34784
	v_mul_f32_e32 v162, 0x3e0293ee, v163
	v_fma_f32 v10, v82, s7, -v162
	v_fma_f32 v11, v83, s7, -v162
	v_mfma_f32_32x32x16_f16 v[66:81], v[236:239], v[110:113], v[66:81]
	v_fma_f32 v12, v84, s7, -v162
	v_fma_f32 v13, v85, s7, -v162
	v_fma_f32 v14, v86, s7, -v162
	v_fma_f32 v15, v87, s7, -v162
	v_fma_f32 v16, v88, s7, -v162
	v_fma_f32 v17, v89, s7, -v162
	v_fma_f32 v82, v90, s7, -v162
	v_mfma_f32_32x32x16_f16 v[18:33], v[242:245], v[110:113], v[18:33]
	v_fma_f32 v83, v91, s7, -v162
	v_fma_f32 v84, v92, s7, -v162
	v_fma_f32 v85, v93, s7, -v162
	v_fma_f32 v86, v94, s7, -v162
	v_fma_f32 v87, v95, s7, -v162
	v_fma_f32 v88, v96, s7, -v162
	v_fma_f32 v89, v97, s7, -v162
	v_mfma_f32_32x32x16_f16 v[34:49], v[232:235], v[106:109], v[34:49]
	v_fma_f32 v90, v114, s7, -v162
	v_fma_f32 v91, v115, s7, -v162
	v_fma_f32 v92, v116, s7, -v162
	v_fma_f32 v93, v117, s7, -v162
	v_fma_f32 v94, v118, s7, -v162
	v_fma_f32 v95, v119, s7, -v162
	v_fma_f32 v96, v120, s7, -v162
	v_mfma_f32_32x32x16_f16 v[66:81], v[248:251], v[106:109], v[66:81]
	v_fma_f32 v97, v121, s7, -v162
	v_exp_f32_e32 v215, v82
	v_exp_f32_e32 v216, v83
	v_exp_f32_e32 v217, v84
	v_exp_f32_e32 v218, v85
	v_exp_f32_e32 v219, v86
	v_exp_f32_e32 v220, v87
	v_mfma_f32_32x32x16_f16 v[18:33], v[252:255], v[106:109], v[18:33]
	v_exp_f32_e32 v221, v88
	v_exp_f32_e32 v222, v89
	v_exp_f32_e32 v223, v90
	v_exp_f32_e32 v224, v91
	v_exp_f32_e32 v225, v92
	v_exp_f32_e32 v226, v93
	v_exp_f32_e32 v228, v94
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[2:5], v[158:161], 0
	ds_read_b128 v[2:5], v164 offset:34816
	v_exp_f32_e32 v229, v95
	v_exp_f32_e32 v230, v96
	v_exp_f32_e32 v231, v97
	v_fma_f32 v116, v124, s7, -v162
	v_fma_f32 v117, v125, s7, -v162
	v_fma_f32 v118, v126, s7, -v162
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[6:9], v[154:157], v[98:113]
	ds_read_b128 v[6:9], v164 offset:34848
	v_fma_f32 v119, v127, s7, -v162
	v_exp_f32_e32 v168, v10
	v_exp_f32_e32 v169, v11
	v_exp_f32_e32 v176, v12
	v_exp_f32_e32 v177, v13
	v_exp_f32_e32 v202, v14
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[2:5], v[150:153], v[98:113]
	ds_read_b128 v[2:5], v164 offset:34880
	v_exp_f32_e32 v206, v15
	v_exp_f32_e32 v207, v16
	v_exp_f32_e32 v214, v17
	v_exp_f32_e32 v234, v116
	v_exp_f32_e32 v235, v117
	v_fma_f32 v114, v122, s7, -v162
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[6:9], v[146:149], v[98:113]
	ds_read_b128 v[6:9], v164 offset:34912
	v_exp_f32_e32 v232, v114
	v_add_u32_e32 v114, s6, v199
	v_fma_f32 v115, v123, s7, -v162
	v_fma_f32 v120, v128, s7, -v162
	v_fma_f32 v121, v129, s7, -v162
	v_fmac_f32_e32 v227, 0xbe0293ee, v163
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[2:5], v[142:145], v[98:113]
	ds_read_b128 v[2:5], v164 offset:34944
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_exp_f32_e32 v233, v115
	v_exp_f32_e32 v236, v121
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[6:9], v[138:141], v[98:113]
	ds_read_b128 v[6:9], v164 offset:34976
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[2:5], v[134:137], v[98:113]
	ds_read_b128 v[2:5], v164 offset:35008
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[98:113], v[6:9], v[130:133], v[98:113]
	ds_read_b128 v[6:9], v164 offset:35040
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[82:97], v[2:5], v[158:161], 0
	ds_read_b128 v[2:5], v164 offset:35072
	v_exp_f32_e32 v159, v118
	v_exp_f32_e32 v160, v119
	v_exp_f32_e32 v158, v227
	v_exp_f32_e32 v161, v120
	v_mul_f32_e32 v115, v51, v158
	v_mul_f32_e32 v120, v56, v158
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[82:97], v[6:9], v[154:157], v[82:97]
	ds_read_b128 v[6:9], v164 offset:35104
	ds_read_b128 v[10:13], v164 offset:35136
	ds_read_b128 v[14:17], v164 offset:35168
	ds_read_b128 v[116:119], v164 offset:35200
	ds_read_b128 v[124:127], v164 offset:35232
	ds_write_b128 v114, v[190:193]
	ds_write_b128 v114, v[194:197] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mul_f32_e32 v121, v57, v158
	v_mfma_f32_32x32x16_f16 v[82:97], v[2:5], v[150:153], v[82:97]
	v_add3_u32 v2, v209, v246, v198
	ds_read2st64_b32 v[2:3], v2 offset1:8
	v_add_u32_e32 v4, s3, v203
	s_nop 0
	v_readfirstlane_b32 s12, v4
	s_mov_b32 m0, s12
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 1, v2
	v_mfma_f32_32x32x16_f16 v[82:97], v[6:9], v[146:149], v[82:97]
	s_barrier
	ds_write_b128 v114, v[186:189]
	ds_write_b128 v114, v[182:185] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[82:97], v[10:13], v[142:145], v[82:97]
	v_add_u32_e32 v2, s3, v200
	s_nop 0
	v_readfirstlane_b32 s12, v2
	v_lshlrev_b32_e32 v3, 1, v3
	s_mov_b32 m0, s12
	v_mul_f32_e32 v8, v40, v158
	v_add_u32_e32 v40, s2, v181
	buffer_load_dwordx4 v3, s[4:7], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[82:97], v[14:17], v[138:141], v[82:97]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v146, s2, v170
	ds_read_b64_tr_b16 v[14:15], v40
	ds_read_b64_tr_b16 v[16:17], v146 offset:8192
	v_mul_f32_e32 v114, v50, v158
	v_mul_f32_e32 v122, v58, v158
	v_mfma_f32_32x32x16_f16 v[82:97], v[116:119], v[134:137], v[82:97]
	v_mul_f32_e32 v116, v52, v158
	v_mul_f32_e32 v117, v53, v158
	v_mul_f32_e32 v118, v54, v158
	v_mul_f32_e32 v119, v55, v158
	v_mul_f32_e32 v123, v59, v158
	v_mul_f32_e32 v128, v64, v158
	v_mul_f32_e32 v129, v65, v158
	v_mfma_f32_32x32x16_f16 v[82:97], v[124:127], v[130:133], v[82:97]
	v_mul_f32_e32 v124, v60, v158
	v_mul_f32_e32 v125, v61, v158
	v_mul_f32_e32 v126, v62, v158
	v_mul_f32_e32 v127, v63, v158
	v_cvt_pk_f16_f32 v142, v168, v169
	v_cvt_pk_f16_f32 v143, v176, v177
	v_cvt_pk_f16_f32 v144, v202, v206
	v_cvt_pk_f16_f32 v145, v207, v214
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[114:129], v[14:17], v[142:145], v[114:129]
	v_add_u32_e32 v147, s2, v171
	v_mul_f32_e32 v2, v34, v158
	v_mul_f32_e32 v3, v35, v158
	v_mul_f32_e32 v4, v36, v158
	v_mul_f32_e32 v5, v37, v158
	v_mul_f32_e32 v6, v38, v158
	v_mul_f32_e32 v7, v39, v158
	v_mul_f32_e32 v9, v41, v158
	ds_read_b64_tr_b16 v[34:35], v147 offset:256
	ds_read_b64_tr_b16 v[36:37], v146 offset:8448
	ds_read_b64_tr_b16 v[38:39], v146 offset:8256
	ds_read_b64_tr_b16 v[40:41], v40 offset:512
	v_cvt_pk_f16_f32 v134, v215, v216
	v_cvt_pk_f16_f32 v135, v217, v218
	v_cvt_pk_f16_f32 v136, v219, v220
	v_cvt_pk_f16_f32 v137, v221, v222
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[114:129], v[34:37], v[134:137], v[114:129]
	v_cvt_pk_f16_f32 v130, v223, v224
	v_cvt_pk_f16_f32 v131, v225, v226
	v_cvt_pk_f16_f32 v132, v228, v229
	v_cvt_pk_f16_f32 v133, v230, v231
	v_mul_f32_e32 v10, v42, v158
	v_mul_f32_e32 v11, v43, v158
	ds_read_b64_tr_b16 v[42:43], v146 offset:8704
	ds_read_b64_tr_b16 v[50:51], v147 offset:320
	ds_read_b64_tr_b16 v[54:55], v147 offset:384
	ds_read_b64_tr_b16 v[148:149], v147 offset:192
	ds_read_b64_tr_b16 v[52:53], v146 offset:8512
	ds_read_b64_tr_b16 v[60:61], v146 offset:8320
	ds_read_b64_tr_b16 v[150:151], v146 offset:8384
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[114:129], v[40:43], v[130:133], v[114:129]
	v_mul_f32_e32 v12, v44, v158
	v_mul_f32_e32 v13, v45, v158
	ds_read_b64_tr_b16 v[14:15], v147 offset:768
	ds_read_b64_tr_b16 v[16:17], v146 offset:8960
	ds_read_b64_tr_b16 v[44:45], v146 offset:8768
	ds_read_b64_tr_b16 v[56:57], v146 offset:8576
	ds_read_b64_tr_b16 v[154:155], v146 offset:8640
	v_cvt_pk_f16_f32 v138, v232, v233
	v_cvt_pk_f16_f32 v139, v234, v235
	v_cvt_pk_f16_f32 v140, v159, v160
	v_cvt_pk_f16_f32 v141, v161, v236
	s_waitcnt lgkmcnt(3)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[114:129], v[14:17], v[138:141], v[114:129]
	v_mul_f32_e32 v14, v46, v158
	v_mul_f32_e32 v15, v47, v158
	v_mul_f32_e32 v16, v48, v158
	v_mul_f32_e32 v17, v49, v158
	ds_read_b64_tr_b16 v[36:37], v147 offset:64
	ds_read_b64_tr_b16 v[64:65], v146 offset:9024
	ds_read_b64_tr_b16 v[166:167], v146 offset:8832
	ds_read_b64_tr_b16 v[174:175], v146 offset:8896
	ds_read_b64_tr_b16 v[62:63], v147 offset:832
	ds_read_b64_tr_b16 v[182:183], v147 offset:896
	ds_read_b64_tr_b16 v[186:187], v147 offset:960
	ds_read_b64_tr_b16 v[58:59], v147 offset:128
	ds_read_b64_tr_b16 v[184:185], v146 offset:9088
	ds_read_b64_tr_b16 v[188:189], v146 offset:9152
	ds_read_b64_tr_b16 v[42:43], v147 offset:576
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[2:17], v[36:39], v[142:145], v[2:17]
	v_add_f32_e32 v34, v172, v210
	v_add_f32_e32 v34, v34, v211
	v_add_f32_e32 v34, v34, v212
	v_add_f32_e32 v34, v34, v204
	v_add_f32_e32 v146, v34, v213
	v_mul_f32_e32 v34, v66, v158
	v_mul_f32_e32 v35, v67, v158
	v_mfma_f32_32x32x16_f16 v[2:17], v[50:53], v[134:137], v[2:17]
	v_add_f32_e32 v50, v168, v169
	v_add_f32_e32 v50, v176, v50
	v_add_f32_e32 v50, v177, v50
	v_add_f32_e32 v50, v202, v50
	v_mul_f32_e32 v36, v68, v158
	v_mul_f32_e32 v37, v69, v158
	v_mul_f32_e32 v38, v70, v158
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[42:45], v[130:133], v[2:17]
	v_mul_f32_e32 v39, v71, v158
	v_mul_f32_e32 v40, v72, v158
	v_mul_f32_e32 v41, v73, v158
	v_mul_f32_e32 v42, v74, v158
	v_mul_f32_e32 v43, v75, v158
	v_mul_f32_e32 v44, v76, v158
	v_mul_f32_e32 v45, v77, v158
	v_mul_f32_e32 v46, v78, v158
	v_mul_f32_e32 v47, v79, v158
	v_mul_f32_e32 v48, v80, v158
	v_mul_f32_e32 v49, v81, v158
	v_add_f32_e32 v50, v206, v50
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[34:49], v[58:61], v[142:145], v[34:49]
	v_add_f32_e32 v50, v207, v50
	v_add_f32_e32 v50, v214, v50
	v_add_f32_e32 v50, v215, v50
	v_add_f32_e32 v50, v216, v50
	v_add_f32_e32 v66, v217, v50
	v_mul_f32_e32 v50, v18, v158
	v_add_f32_e32 v18, v218, v66
	v_add_f32_e32 v18, v219, v18
	v_add_f32_e32 v18, v220, v18
	v_mfma_f32_32x32x16_f16 v[2:17], v[62:65], v[138:141], v[2:17]
	v_mul_f32_e32 v51, v19, v158
	v_mul_f32_e32 v52, v20, v158
	v_mul_f32_e32 v53, v21, v158
	v_mul_f32_e32 v58, v26, v158
	v_mul_f32_e32 v59, v27, v158
	v_mul_f32_e32 v60, v28, v158
	v_mul_f32_e32 v61, v29, v158
	v_mfma_f32_32x32x16_f16 v[34:49], v[54:57], v[134:137], v[34:49]
	v_mul_f32_e32 v54, v22, v158
	v_mul_f32_e32 v55, v23, v158
	v_mul_f32_e32 v56, v24, v158
	v_mul_f32_e32 v57, v25, v158
	v_mul_f32_e32 v62, v30, v158
	v_mul_f32_e32 v63, v31, v158
	v_mul_f32_e32 v64, v32, v158
	v_mul_f32_e32 v65, v33, v158
	v_add_f32_e32 v18, v221, v18
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[50:65], v[148:151], v[142:145], v[50:65]
	v_add_f32_e32 v18, v222, v18
	v_add_f32_e32 v18, v223, v18
	v_add_f32_e32 v18, v224, v18
	ds_read_b64_tr_b16 v[164:165], v147 offset:640
	ds_read_b64_tr_b16 v[172:173], v147 offset:704
	ds_read_b64_tr_b16 v[152:153], v147 offset:448
	v_add_f32_e32 v18, v225, v18
	v_add_f32_e32 v18, v226, v18
	v_add_f32_e32 v18, v228, v18
	v_add_f32_e32 v18, v229, v18
	v_add_f32_e32 v18, v230, v18
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[152:155], v[134:137], v[50:65]
	v_add_f32_e32 v18, v231, v18
	v_add_f32_e32 v18, v232, v18
	v_add_f32_e32 v18, v233, v18
	v_add_f32_e32 v18, v234, v18
	v_add_f32_e32 v18, v235, v18
	v_add_f32_e32 v18, v159, v18
	v_add_f32_e32 v18, v160, v18
	v_add_f32_e32 v18, v161, v18
	v_mfma_f32_32x32x16_f16 v[34:49], v[164:167], v[130:133], v[34:49]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v167, s3, v171
	v_mov_b32_e32 v156, v146
	s_nop 1
	v_permlane32_swap_b32_e32 v146, v156
	v_mfma_f32_32x32x16_f16 v[50:65], v[172:175], v[130:133], v[50:65]
	v_add_f32_e32 v131, v236, v18
	v_maximum3_f32 v18, v98, v99, v99
	v_maximum3_f32 v18, v18, v100, v101
	v_maximum3_f32 v18, v18, v102, v103
	v_maximum3_f32 v18, v18, v104, v105
	v_maximum3_f32 v18, v18, v106, v107
	v_maximum3_f32 v18, v18, v108, v109
	v_maximum3_f32 v18, v18, v110, v111
	v_maximum3_f32 v18, v18, v112, v113
	v_maximum3_f32 v18, v18, v82, v83
	v_maximum3_f32 v18, v18, v84, v85
	v_maximum3_f32 v18, v18, v86, v87
	v_maximum3_f32 v18, v18, v88, v89
	v_maximum3_f32 v18, v18, v90, v91
	v_maximum3_f32 v18, v18, v92, v93
	v_maximum3_f32 v18, v18, v94, v95
	v_maximum3_f32 v18, v18, v96, v97
	v_mov_b32_e32 v19, v18
	s_nop 1
	v_permlane32_swap_b32_e32 v18, v19
	v_maximum3_f32 v130, v163, v18, v19
	v_mul_f32_e32 v18, 0xbe0293ee, v130
	v_fmac_f32_e32 v162, 0xbe0293ee, v130
	v_fmamk_f32 v76, v91, 0x3e0293ee, v18
	v_exp_f32_e32 v162, v162
	v_fmamk_f32 v19, v98, 0x3e0293ee, v18
	v_fmamk_f32 v20, v99, 0x3e0293ee, v18
	v_fmamk_f32 v21, v100, 0x3e0293ee, v18
	v_fmamk_f32 v22, v101, 0x3e0293ee, v18
	v_fmamk_f32 v23, v102, 0x3e0293ee, v18
	v_fmamk_f32 v24, v103, 0x3e0293ee, v18
	v_fmamk_f32 v25, v104, 0x3e0293ee, v18
	v_fmamk_f32 v26, v105, 0x3e0293ee, v18
	v_fmamk_f32 v27, v106, 0x3e0293ee, v18
	v_fmamk_f32 v28, v107, 0x3e0293ee, v18
	v_fmamk_f32 v29, v108, 0x3e0293ee, v18
	v_fmamk_f32 v30, v109, 0x3e0293ee, v18
	v_fmamk_f32 v31, v110, 0x3e0293ee, v18
	v_fmamk_f32 v32, v111, 0x3e0293ee, v18
	v_fmamk_f32 v33, v112, 0x3e0293ee, v18
	v_fmamk_f32 v66, v113, 0x3e0293ee, v18
	v_fmamk_f32 v67, v82, 0x3e0293ee, v18
	v_fmamk_f32 v68, v83, 0x3e0293ee, v18
	v_fmamk_f32 v69, v84, 0x3e0293ee, v18
	v_fmamk_f32 v70, v85, 0x3e0293ee, v18
	v_fmamk_f32 v71, v86, 0x3e0293ee, v18
	v_fmamk_f32 v72, v87, 0x3e0293ee, v18
	v_fmamk_f32 v73, v88, 0x3e0293ee, v18
	v_fmamk_f32 v74, v89, 0x3e0293ee, v18
	v_fmamk_f32 v75, v90, 0x3e0293ee, v18
	v_fmamk_f32 v77, v92, 0x3e0293ee, v18
	v_fmamk_f32 v78, v93, 0x3e0293ee, v18
	v_fmamk_f32 v79, v94, 0x3e0293ee, v18
	v_fmamk_f32 v80, v95, 0x3e0293ee, v18
	v_fmamk_f32 v81, v96, 0x3e0293ee, v18
	v_fmac_f32_e32 v18, 0x3e0293ee, v97
	v_exp_f32_e32 v159, v76
	v_add_u32_e32 v76, s3, v181
	v_exp_f32_e32 v166, v18
	v_mul_f32_e32 v18, v114, v162
	v_add_u32_e32 v114, s3, v170
	ds_read_b64_tr_b16 v[82:83], v76
	ds_read_b64_tr_b16 v[84:85], v114 offset:8192
	v_mfma_f32_32x32x16_f16 v[34:49], v[182:185], v[138:141], v[34:49]
	v_exp_f32_e32 v132, v19
	v_exp_f32_e32 v133, v20
	v_exp_f32_e32 v134, v21
	v_exp_f32_e32 v135, v22
	v_exp_f32_e32 v136, v23
	v_exp_f32_e32 v137, v24
	v_exp_f32_e32 v142, v29
	v_mfma_f32_32x32x16_f16 v[50:65], v[186:189], v[138:141], v[50:65]
	v_exp_f32_e32 v138, v25
	v_exp_f32_e32 v139, v26
	v_exp_f32_e32 v140, v27
	v_exp_f32_e32 v141, v28
	v_exp_f32_e32 v143, v30
	v_exp_f32_e32 v144, v31
	v_exp_f32_e32 v145, v32
	v_exp_f32_e32 v147, v33
	v_exp_f32_e32 v161, v78
	v_exp_f32_e32 v163, v79
	v_exp_f32_e32 v164, v80
	v_exp_f32_e32 v165, v81
	v_mul_f32_e32 v19, v115, v162
	v_mul_f32_e32 v20, v116, v162
	v_mul_f32_e32 v21, v117, v162
	v_mul_f32_e32 v22, v118, v162
	v_mul_f32_e32 v23, v119, v162
	v_mul_f32_e32 v24, v120, v162
	v_mul_f32_e32 v25, v121, v162
	v_mul_f32_e32 v26, v122, v162
	v_mul_f32_e32 v27, v123, v162
	v_mul_f32_e32 v28, v124, v162
	v_mul_f32_e32 v29, v125, v162
	v_mul_f32_e32 v30, v126, v162
	v_mul_f32_e32 v31, v127, v162
	v_mul_f32_e32 v32, v128, v162
	v_mul_f32_e32 v33, v129, v162
	v_cvt_pk_f16_f32 v78, v132, v133
	v_cvt_pk_f16_f32 v79, v134, v135
	v_cvt_pk_f16_f32 v80, v136, v137
	v_cvt_pk_f16_f32 v81, v138, v139
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[78:81], v[18:33]
	v_exp_f32_e32 v148, v66
	ds_read_b64_tr_b16 v[86:87], v167 offset:256
	ds_read_b64_tr_b16 v[88:89], v114 offset:8448
	ds_read_b64_tr_b16 v[90:91], v114 offset:8256
	ds_read_b64_tr_b16 v[92:93], v76 offset:512
	v_exp_f32_e32 v152, v70
	v_exp_f32_e32 v153, v71
	v_exp_f32_e32 v154, v72
	v_exp_f32_e32 v155, v73
	v_cvt_pk_f16_f32 v70, v140, v141
	v_cvt_pk_f16_f32 v71, v142, v143
	v_cvt_pk_f16_f32 v72, v144, v145
	v_cvt_pk_f16_f32 v73, v147, v148
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[86:89], v[70:73], v[18:33]
	v_add_f32_e32 v146, v146, v156
	v_exp_f32_e32 v149, v67
	v_exp_f32_e32 v150, v68
	v_exp_f32_e32 v151, v69
	v_exp_f32_e32 v156, v74
	v_mul_f32_e32 v2, v2, v162
	v_mul_f32_e32 v3, v3, v162
	v_mul_f32_e32 v4, v4, v162
	v_mul_f32_e32 v5, v5, v162
	v_mul_f32_e32 v6, v6, v162
	v_mul_f32_e32 v7, v7, v162
	v_mul_f32_e32 v8, v8, v162
	v_mul_f32_e32 v9, v9, v162
	v_mul_f32_e32 v10, v10, v162
	v_mul_f32_e32 v11, v11, v162
	v_mul_f32_e32 v12, v12, v162
	v_mul_f32_e32 v13, v13, v162
	v_cvt_pk_f16_f32 v66, v149, v150
	v_cvt_pk_f16_f32 v67, v151, v152
	v_cvt_pk_f16_f32 v68, v153, v154
	v_cvt_pk_f16_f32 v69, v155, v156
	ds_read_b64_tr_b16 v[94:95], v114 offset:8704
	ds_read_b64_tr_b16 v[82:83], v167 offset:320
	ds_read_b64_tr_b16 v[96:97], v167 offset:384
	ds_read_b64_tr_b16 v[100:101], v167 offset:192
	ds_read_b64_tr_b16 v[84:85], v114 offset:8512
	ds_read_b64_tr_b16 v[106:107], v114 offset:8320
	ds_read_b64_tr_b16 v[102:103], v114 offset:8384
	v_mul_f32_e32 v14, v14, v162
	v_mul_f32_e32 v15, v15, v162
	v_mul_f32_e32 v16, v16, v162
	v_mul_f32_e32 v17, v17, v162
	ds_read_b64_tr_b16 v[108:109], v167 offset:768
	ds_read_b64_tr_b16 v[110:111], v114 offset:8960
	ds_read_b64_tr_b16 v[112:113], v114 offset:8768
	ds_read_b64_tr_b16 v[98:99], v114 offset:8576
	ds_read_b64_tr_b16 v[86:87], v114 offset:8640
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[18:33], v[92:95], v[66:69], v[18:33]
	ds_read_b64_tr_b16 v[88:89], v167 offset:64
	ds_read_b64_tr_b16 v[94:95], v114 offset:9024
	ds_read_b64_tr_b16 v[116:117], v114 offset:8832
	ds_read_b64_tr_b16 v[120:121], v114 offset:8896
	ds_read_b64_tr_b16 v[92:93], v167 offset:832
	ds_read_b64_tr_b16 v[122:123], v167 offset:896
	ds_read_b64_tr_b16 v[126:127], v167 offset:960
	ds_read_b64_tr_b16 v[104:105], v167 offset:128
	ds_read_b64_tr_b16 v[124:125], v114 offset:9088
	ds_read_b64_tr_b16 v[128:129], v114 offset:9152
	v_mul_f32_e32 v34, v34, v162
	v_mul_f32_e32 v35, v35, v162
	v_mul_f32_e32 v36, v36, v162
	v_mul_f32_e32 v37, v37, v162
	v_mul_f32_e32 v38, v38, v162
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[2:17], v[88:91], v[78:81], v[2:17]
	v_add_f32_e32 v89, v132, v133
	v_add_f32_e32 v89, v134, v89
	v_mul_f32_e32 v39, v39, v162
	v_mul_f32_e32 v40, v40, v162
	v_mul_f32_e32 v41, v41, v162
	v_mul_f32_e32 v42, v42, v162
	v_mul_f32_e32 v43, v43, v162
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[70:73], v[2:17]
	v_add_f32_e32 v82, v135, v89
	v_add_f32_e32 v82, v136, v82
	v_add_f32_e32 v82, v137, v82
	v_add_f32_e32 v82, v138, v82
	v_add_f32_e32 v82, v139, v82
	v_add_f32_e32 v82, v140, v82
	v_add_f32_e32 v82, v141, v82
	v_mul_f32_e32 v44, v44, v162
	v_mul_f32_e32 v45, v45, v162
	v_mul_f32_e32 v46, v46, v162
	v_mul_f32_e32 v47, v47, v162
	v_mul_f32_e32 v48, v48, v162
	v_mul_f32_e32 v49, v49, v162
	v_add_f32_e32 v82, v142, v82
	v_mul_f32_e32 v50, v50, v162
	v_mul_f32_e32 v51, v51, v162
	v_mul_f32_e32 v52, v52, v162
	v_mul_f32_e32 v53, v53, v162
	v_mul_f32_e32 v54, v54, v162
	v_mul_f32_e32 v55, v55, v162
	v_mul_f32_e32 v56, v56, v162
	v_mul_f32_e32 v57, v57, v162
	v_mul_f32_e32 v58, v58, v162
	v_mul_f32_e32 v59, v59, v162
	v_mul_f32_e32 v60, v60, v162
	v_mul_f32_e32 v61, v61, v162
	v_mul_f32_e32 v62, v62, v162
	v_mul_f32_e32 v63, v63, v162
	v_mul_f32_e32 v64, v64, v162
	v_mul_f32_e32 v65, v65, v162
	v_exp_f32_e32 v157, v75
	v_exp_f32_e32 v160, v77
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[104:107], v[78:81], v[34:49]
	v_add_f32_e32 v82, v143, v82
	v_add_f32_e32 v82, v144, v82
	v_cvt_pk_f16_f32 v74, v157, v159
	v_cvt_pk_f16_f32 v75, v160, v161
	v_cvt_pk_f16_f32 v76, v163, v164
	v_cvt_pk_f16_f32 v77, v165, v166
	v_add_f32_e32 v82, v145, v82
	v_mfma_f32_32x32x16_f16 v[50:65], v[100:103], v[78:81], v[50:65]
	v_add_f32_e32 v82, v147, v82
	v_add_f32_e32 v82, v148, v82
	v_add_f32_e32 v82, v149, v82
	v_add_f32_e32 v78, v150, v82
	v_add_f32_e32 v78, v151, v78
	v_add_f32_e32 v78, v152, v78
	v_add_f32_e32 v78, v153, v78
	v_mfma_f32_32x32x16_f16 v[18:33], v[108:111], v[74:77], v[18:33]
	ds_read_b64_tr_b16 v[110:111], v167 offset:576
	ds_read_b64_tr_b16 v[114:115], v167 offset:640
	ds_read_b64_tr_b16 v[118:119], v167 offset:704
	ds_read_b64_tr_b16 v[84:85], v167 offset:448
	v_add_f32_e32 v78, v154, v78
	v_add_f32_e32 v78, v155, v78
	v_add_f32_e32 v78, v156, v78
	v_mov_b32_e32 v88, v131
	s_nop 1
	v_permlane32_swap_b32_e32 v131, v88
	v_mfma_f32_32x32x16_f16 v[34:49], v[96:99], v[70:73], v[34:49]
	v_fmac_f32_e32 v146, v201, v205
	v_add_f32_e32 v88, v131, v88
	v_fmac_f32_e32 v88, v146, v158
	s_mov_b32 s2, 0x800000
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[50:65], v[84:87], v[70:73], v[50:65]
	v_add_f32_e32 v70, v157, v78
	v_add_f32_e32 v70, v159, v70
	v_add_f32_e32 v70, v160, v70
	v_add_f32_e32 v70, v161, v70
	v_add_f32_e32 v70, v163, v70
	v_add_f32_e32 v70, v164, v70
	v_add_f32_e32 v70, v165, v70
	v_mfma_f32_32x32x16_f16 v[2:17], v[110:113], v[66:69], v[2:17]
	v_mfma_f32_32x32x16_f16 v[34:49], v[114:117], v[66:69], v[34:49]
	v_mfma_f32_32x32x16_f16 v[50:65], v[118:121], v[66:69], v[50:65]
	v_add_f32_e32 v66, v166, v70
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v67
	v_add_f32_e32 v66, v66, v67
	v_fmac_f32_e32 v66, v88, v162
	v_cmp_gt_f32_e32 vcc, s2, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[92:95], v[74:77], v[2:17]
	s_nop 0
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v66, v67
	v_log_f32_e32 v67, v67
	v_mov_b32_e32 v68, 0x42000000
	v_cndmask_b32_e32 v68, 0, v68, vcc
	v_sub_f32_e32 v67, v67, v68
	v_add_f32_e32 v67, v130, v67
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[74:77], v[34:49]
	v_add3_u32 v68, 0, v179, v180
	ds_write_b32 v68, v67
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[126:129], v[74:77], v[50:65]
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	v_or_b32_e32 v67, s16, v208
	s_movk_i32 s2, 0x4000
	v_cmp_gt_i32_e64 s[4:5], s2, v67
	s_sub_i32 s2, 0x4000, s16
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	v_bfrev_b32_e32 v0, 1
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v0, v0, v178, vcc
	s_barrier
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr4_sgpr5
                                        ; implicit-def: $vgpr0
.LBB0_9:
	v_bfrev_b32_e32 v0, 1
	v_cndmask_b32_e64 v0, v0, v178, s[0:1]
	s_or_b64 s[4:5], s[4:5], exec
	s_barrier
.LBB0_10:
	v_div_scale_f32 v67, s[0:1], v66, v66, 1.0
	v_rcp_f32_e32 v68, v67
	s_lshl_b32 s0, s18, 20
	v_fma_f32 v69, -v67, v68, 1.0
	v_fmac_f32_e32 v68, v69, v68
	v_div_scale_f32 v69, vcc, 1.0, v66, 1.0
	v_mul_f32_e32 v70, v69, v68
	v_fma_f32 v71, -v67, v70, v69
	v_fmac_f32_e32 v70, v71, v68
	v_fma_f32 v67, -v67, v70, v69
	v_div_fmas_f32 v67, v67, v68, v70
	s_ashr_i32 s1, s0, 31
	v_div_fixup_f32 v66, v67, v66, 1.0
	s_lshl_b64 s[0:1], s[0:1], 2
	v_add_u32_e32 v67, 0, v178
	s_add_u32 s2, s8, s0
	ds_read_b32 v67, v67
	s_addc_u32 s3, s9, s1
	s_lshl_b32 s0, s19, 14
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s2, s2, s0
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s17, s16, 31
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[8:9], v[66:67], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[66:67], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	s_lshl_b64 s[0:1], s[16:17], 2
	v_cvt_pk_f16_f32 v9, v8, v9
	v_cvt_pk_f16_f32 v8, v6, v7
	v_pk_mul_f32 v[4:5], v[66:67], v[4:5] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v6, v2, v3
	v_pk_mul_f32 v[2:3], v[66:67], v[32:33] op_sel_hi:[0,1]
	s_add_u32 s0, s2, s0
	v_pk_mul_f32 v[16:17], v[66:67], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[66:67], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[66:67], v[10:11] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v7, v4, v5
	v_cvt_pk_f16_f32 v5, v2, v3
	v_pk_mul_f32 v[2:3], v[66:67], v[30:31] op_sel_hi:[0,1]
	s_addc_u32 s1, s3, s1
	v_cvt_pk_f16_f32 v17, v16, v17
	v_cvt_pk_f16_f32 v16, v14, v15
	v_cvt_pk_f16_f32 v14, v10, v11
	v_cvt_pk_f16_f32 v4, v2, v3
	v_pk_mul_f32 v[2:3], v[66:67], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[66:67], v[26:27] op_sel_hi:[0,1]
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_pk_mul_f32 v[12:13], v[66:67], v[12:13] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v3, v2, v3
	v_cvt_pk_f16_f32 v2, v10, v11
	v_pk_mul_f32 v[10:11], v[66:67], v[24:25] op_sel_hi:[0,1]
	buffer_store_dword v67, v0, s[0:3], 0 offen
	v_cvt_pk_f16_f32 v15, v12, v13
	v_cvt_pk_f16_f32 v13, v10, v11
	v_pk_mul_f32 v[10:11], v[66:67], v[22:23] op_sel_hi:[0,1]
	v_lshrrev_b32_e32 v0, 2, v1
	v_mul_lo_u32 v1, s22, v208
	s_mul_i32 s0, s20, s18
	v_cvt_pk_f16_f32 v12, v10, v11
	v_pk_mul_f32 v[10:11], v[66:67], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[66:67], v[18:19] op_sel_hi:[0,1]
	v_add_lshl_u32 v0, v1, v0, 1
	s_ashr_i32 s1, s0, 31
	v_cvt_pk_f16_f32 v11, v10, v11
	v_cvt_pk_f16_f32 v10, v18, v19
	v_add_u32_e32 v1, 0xe0, v0
	v_bfrev_b32_e32 v18, 1
	v_add_u32_e32 v19, 0xc0, v0
	v_add_u32_e32 v20, 0xa0, v0
	v_add_u32_e32 v21, 0x80, v0
	v_add_u32_e32 v22, 0x60, v0
	v_add_u32_e32 v23, 64, v0
	v_add_u32_e32 v24, 32, v0
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cndmask_b32_e64 v1, v18, v1, s[4:5]
	v_cndmask_b32_e64 v19, v18, v19, s[4:5]
	v_cndmask_b32_e64 v20, v18, v20, s[4:5]
	v_cndmask_b32_e64 v21, v18, v21, s[4:5]
	v_cndmask_b32_e64 v22, v18, v22, s[4:5]
	v_cndmask_b32_e64 v23, v18, v23, s[4:5]
	v_cndmask_b32_e64 v24, v18, v24, s[4:5]
	v_cndmask_b32_e64 v0, v18, v0, s[4:5]
	s_add_u32 s4, s10, s0
	s_mul_i32 s0, s21, s19
	s_addc_u32 s5, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s4, s4, s0
	s_mul_i32 s0, s22, s16
	s_addc_u32 s5, s5, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s4, s0
	v_pk_mul_f32 v[64:65], v[66:67], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[66:67], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[66:67], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[66:67], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[66:67], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[66:67], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[66:67], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[66:67], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[66:67], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[66:67], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[66:67], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[66:67], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[66:67], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[66:67], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[66:67], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[66:67], v[34:35] op_sel_hi:[0,1]
	s_addc_u32 s1, s5, s1
	v_cvt_pk_f16_f32 v65, v64, v65
	v_cvt_pk_f16_f32 v64, v62, v63
	v_cvt_pk_f16_f32 v63, v60, v61
	v_cvt_pk_f16_f32 v62, v58, v59
	v_cvt_pk_f16_f32 v57, v56, v57
	v_cvt_pk_f16_f32 v56, v54, v55
	v_cvt_pk_f16_f32 v55, v52, v53
	v_cvt_pk_f16_f32 v54, v50, v51
	v_cvt_pk_f16_f32 v49, v48, v49
	v_cvt_pk_f16_f32 v48, v46, v47
	v_cvt_pk_f16_f32 v47, v44, v45
	v_cvt_pk_f16_f32 v46, v42, v43
	v_cvt_pk_f16_f32 v41, v40, v41
	v_cvt_pk_f16_f32 v40, v38, v39
	v_cvt_pk_f16_f32 v39, v36, v37
	v_cvt_pk_f16_f32 v38, v34, v35
	s_and_b32 s1, s1, 0xffff
	v_permlane32_swap_b32_e32 v10, v12
	v_permlane32_swap_b32_e32 v11, v13
	v_permlane32_swap_b32_e32 v2, v4
	v_permlane32_swap_b32_e32 v3, v5
	v_permlane32_swap_b32_e32 v6, v8
	v_permlane32_swap_b32_e32 v7, v9
	v_permlane32_swap_b32_e32 v14, v16
	v_permlane32_swap_b32_e32 v15, v17
	v_permlane32_swap_b32_e32 v38, v40
	v_permlane32_swap_b32_e32 v39, v41
	v_permlane32_swap_b32_e32 v46, v48
	v_permlane32_swap_b32_e32 v47, v49
	v_permlane32_swap_b32_e32 v54, v56
	v_permlane32_swap_b32_e32 v55, v57
	v_permlane32_swap_b32_e32 v62, v64
	v_permlane32_swap_b32_e32 v63, v65
	buffer_store_dwordx4 v[10:13], v0, s[0:3], 0 offen
	buffer_store_dwordx4 v[2:5], v24, s[0:3], 0 offen
	buffer_store_dwordx4 v[6:9], v23, s[0:3], 0 offen
	buffer_store_dwordx4 v[14:17], v22, s[0:3], 0 offen
	buffer_store_dwordx4 v[38:41], v21, s[0:3], 0 offen
	buffer_store_dwordx4 v[46:49], v20, s[0:3], 0 offen
	buffer_store_dwordx4 v[54:57], v19, s[0:3], 0 offen
	buffer_store_dwordx4 v[62:65], v1, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 152
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
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 96
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
	.set attn_fwd.numbered_sgpr, 45
	.set attn_fwd.num_named_barrier, 0
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11148
; TotalNumSgprs: 51
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 256
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
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
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
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
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
	.byte	1                               ; Abbrev [1] 0xb:0xa2 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x7c DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	519                             ; DW_AT_call_line
	.byte	41                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4e:0x5d DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	708                             ; DW_AT_call_line
	.byte	61                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x5b:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	286                             ; DW_AT_call_line
	.byte	69                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x68:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	69                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x75:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	334                             ; DW_AT_call_line
	.byte	42                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x82:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	189                             ; DW_AT_call_line
	.byte	40                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x8f:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	340                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x9c:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges7                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	291                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp291-.Lfunc_begin0
	.quad	.Ltmp292-.Lfunc_begin0
	.quad	.Ltmp299-.Lfunc_begin0
	.quad	.Ltmp300-.Lfunc_begin0
	.quad	.Ltmp301-.Lfunc_begin0
	.quad	.Ltmp302-.Lfunc_begin0
	.quad	.Ltmp303-.Lfunc_begin0
	.quad	.Ltmp304-.Lfunc_begin0
	.quad	.Ltmp305-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
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
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp195-.Lfunc_begin0
	.quad	.Ltmp196-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp216-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp218-.Lfunc_begin0
	.quad	.Ltmp219-.Lfunc_begin0
	.quad	.Ltmp220-.Lfunc_begin0
	.quad	.Ltmp221-.Lfunc_begin0
	.quad	.Ltmp222-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp227-.Lfunc_begin0
	.quad	.Ltmp228-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	.Ltmp232-.Lfunc_begin0
	.quad	.Ltmp233-.Lfunc_begin0
	.quad	.Ltmp234-.Lfunc_begin0
	.quad	.Ltmp248-.Lfunc_begin0
	.quad	.Ltmp249-.Lfunc_begin0
	.quad	.Ltmp253-.Lfunc_begin0
	.quad	.Ltmp254-.Lfunc_begin0
	.quad	.Ltmp260-.Lfunc_begin0
	.quad	.Ltmp261-.Lfunc_begin0
	.quad	.Ltmp262-.Lfunc_begin0
	.quad	.Ltmp263-.Lfunc_begin0
	.quad	.Ltmp264-.Lfunc_begin0
	.quad	.Ltmp265-.Lfunc_begin0
	.quad	.Ltmp268-.Lfunc_begin0
	.quad	.Ltmp269-.Lfunc_begin0
	.quad	.Ltmp270-.Lfunc_begin0
	.quad	.Ltmp271-.Lfunc_begin0
	.quad	.Ltmp272-.Lfunc_begin0
	.quad	.Ltmp273-.Lfunc_begin0
	.quad	.Ltmp286-.Lfunc_begin0
	.quad	.Ltmp287-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp257-.Lfunc_begin0
	.quad	.Ltmp259-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp257-.Lfunc_begin0
	.quad	.Ltmp258-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
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
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp235-.Lfunc_begin0
	.quad	.Ltmp236-.Lfunc_begin0
	.quad	.Ltmp237-.Lfunc_begin0
	.quad	.Ltmp238-.Lfunc_begin0
	.quad	.Ltmp239-.Lfunc_begin0
	.quad	.Ltmp240-.Lfunc_begin0
	.quad	.Ltmp241-.Lfunc_begin0
	.quad	.Ltmp242-.Lfunc_begin0
	.quad	.Ltmp243-.Lfunc_begin0
	.quad	.Ltmp244-.Lfunc_begin0
	.quad	.Ltmp245-.Lfunc_begin0
	.quad	.Ltmp246-.Lfunc_begin0
	.quad	.Ltmp247-.Lfunc_begin0
	.quad	.Ltmp248-.Lfunc_begin0
	.quad	.Ltmp249-.Lfunc_begin0
	.quad	.Ltmp250-.Lfunc_begin0
	.quad	.Ltmp251-.Lfunc_begin0
	.quad	.Ltmp252-.Lfunc_begin0
	.quad	.Ltmp254-.Lfunc_begin0
	.quad	.Ltmp255-.Lfunc_begin0
	.quad	.Ltmp256-.Lfunc_begin0
	.quad	.Ltmp257-.Lfunc_begin0
	.quad	.Ltmp266-.Lfunc_begin0
	.quad	.Ltmp267-.Lfunc_begin0
	.quad	.Ltmp274-.Lfunc_begin0
	.quad	.Ltmp275-.Lfunc_begin0
	.quad	.Ltmp276-.Lfunc_begin0
	.quad	.Ltmp277-.Lfunc_begin0
	.quad	.Ltmp278-.Lfunc_begin0
	.quad	.Ltmp279-.Lfunc_begin0
	.quad	.Ltmp280-.Lfunc_begin0
	.quad	.Ltmp281-.Lfunc_begin0
	.quad	.Ltmp282-.Lfunc_begin0
	.quad	.Ltmp283-.Lfunc_begin0
	.quad	.Ltmp284-.Lfunc_begin0
	.quad	.Ltmp285-.Lfunc_begin0
	.quad	.Ltmp287-.Lfunc_begin0
	.quad	.Ltmp289-.Lfunc_begin0
	.quad	.Ltmp290-.Lfunc_begin0
	.quad	.Ltmp291-.Lfunc_begin0
	.quad	.Ltmp293-.Lfunc_begin0
	.quad	.Ltmp294-.Lfunc_begin0
	.quad	.Ltmp295-.Lfunc_begin0
	.quad	.Ltmp298-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
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
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp235-.Lfunc_begin0
	.quad	.Ltmp236-.Lfunc_begin0
	.quad	.Ltmp237-.Lfunc_begin0
	.quad	.Ltmp238-.Lfunc_begin0
	.quad	.Ltmp239-.Lfunc_begin0
	.quad	.Ltmp240-.Lfunc_begin0
	.quad	.Ltmp241-.Lfunc_begin0
	.quad	.Ltmp242-.Lfunc_begin0
	.quad	.Ltmp243-.Lfunc_begin0
	.quad	.Ltmp244-.Lfunc_begin0
	.quad	.Ltmp245-.Lfunc_begin0
	.quad	.Ltmp246-.Lfunc_begin0
	.quad	.Ltmp247-.Lfunc_begin0
	.quad	.Ltmp248-.Lfunc_begin0
	.quad	.Ltmp249-.Lfunc_begin0
	.quad	.Ltmp250-.Lfunc_begin0
	.quad	.Ltmp251-.Lfunc_begin0
	.quad	.Ltmp252-.Lfunc_begin0
	.quad	.Ltmp256-.Lfunc_begin0
	.quad	.Ltmp257-.Lfunc_begin0
	.quad	.Ltmp266-.Lfunc_begin0
	.quad	.Ltmp267-.Lfunc_begin0
	.quad	.Ltmp274-.Lfunc_begin0
	.quad	.Ltmp275-.Lfunc_begin0
	.quad	.Ltmp276-.Lfunc_begin0
	.quad	.Ltmp277-.Lfunc_begin0
	.quad	.Ltmp278-.Lfunc_begin0
	.quad	.Ltmp279-.Lfunc_begin0
	.quad	.Ltmp280-.Lfunc_begin0
	.quad	.Ltmp281-.Lfunc_begin0
	.quad	.Ltmp282-.Lfunc_begin0
	.quad	.Ltmp283-.Lfunc_begin0
	.quad	.Ltmp284-.Lfunc_begin0
	.quad	.Ltmp285-.Lfunc_begin0
	.quad	.Ltmp287-.Lfunc_begin0
	.quad	.Ltmp288-.Lfunc_begin0
	.quad	.Ltmp290-.Lfunc_begin0
	.quad	.Ltmp291-.Lfunc_begin0
	.quad	.Ltmp293-.Lfunc_begin0
	.quad	.Ltmp294-.Lfunc_begin0
	.quad	.Ltmp295-.Lfunc_begin0
	.quad	.Ltmp296-.Lfunc_begin0
	.quad	.Ltmp297-.Lfunc_begin0
	.quad	.Ltmp298-.Lfunc_begin0
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
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 152
    .max_flat_workgroup_size: 512
    .name:           attn_fwd
    .private_segment_fixed_size: 0
    .sgpr_count:     51
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
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
