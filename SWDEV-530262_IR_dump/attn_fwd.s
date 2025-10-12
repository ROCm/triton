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
	.file	1 "/app/OAI-triton/fav3_kernel" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.12:
.LBB0_0:
	s_mov_b64 s[36:37], s[2:3]
	s_ashr_i32 s2, s16, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s2, s16, s2
	s_ashr_i32 s2, s2, 3
	s_lshl_b32 s3, s16, 3
	s_mul_i32 s19, s2, 0xffffffc1
	s_add_i32 s19, s19, s3
	s_lshl_b32 s16, s17, 8
	s_mul_i32 s12, s12, s18
	s_mul_i32 s2, s13, s19
	s_add_i32 s2, s2, s12
	s_mul_i32 s3, s14, s16
	v_and_b32_e32 v39, 15, v0
	v_lshrrev_b32_e32 v34, 4, v0
	v_lshlrev_b32_e32 v214, 3, v39
	s_add_i32 s2, s2, s3
	s_mov_b64 s[24:25], s[6:7]
	v_or_b32_e32 v35, 32, v34
	v_or_b32_e32 v4, s16, v34
	v_mul_lo_u32 v12, s14, v34
	s_lshl_b32 s6, s14, 6
	v_add_u32_e32 v17, s2, v214
	s_movk_i32 s2, 0x4000
	v_or_b32_e32 v5, s16, v35
	v_mul_lo_u32 v13, s14, v35
	v_add_u32_e32 v14, s6, v12
	v_add_lshl_u32 v12, v17, v12, 1
	v_bfrev_b32_e32 v18, 1
	v_cmp_gt_i32_e32 vcc, s2, v4
	v_or_b32_e32 v1, 0x60, v34
	v_or_b32_e32 v6, 64, v4
	v_or_b32_e32 v8, 0x80, v4
	v_or_b32_e32 v10, 0xc0, v4
	v_cndmask_b32_e32 v36, v18, v12, vcc
	v_add_lshl_u32 v4, v17, v13, 1
	v_cmp_gt_i32_e32 vcc, s2, v5
	v_or_b32_e32 v7, s16, v1
	v_mul_lo_u32 v1, s14, v1
	v_cndmask_b32_e32 v37, v18, v4, vcc
	v_add_lshl_u32 v4, v17, v14, 1
	v_cmp_gt_i32_e32 vcc, s2, v6
	v_or_b32_e32 v2, 0xa0, v34
	v_add_u32_e32 v15, s6, v14
	v_cndmask_b32_e32 v38, v18, v4, vcc
	v_add_lshl_u32 v1, v17, v1, 1
	v_cmp_gt_i32_e32 vcc, s2, v7
	v_or_b32_e32 v9, s16, v2
	v_mul_lo_u32 v2, s14, v2
	v_cndmask_b32_e32 v1, v18, v1, vcc
	v_add_lshl_u32 v4, v17, v15, 1
	v_cmp_gt_i32_e32 vcc, s2, v8
	v_or_b32_e32 v3, 0xe0, v34
	v_add_u32_e32 v16, s6, v15
	v_cndmask_b32_e32 v40, v18, v4, vcc
	v_add_lshl_u32 v2, v17, v2, 1
	v_cmp_gt_i32_e32 vcc, s2, v9
	v_or_b32_e32 v11, s16, v3
	v_mul_lo_u32 v3, s14, v3
	v_cndmask_b32_e32 v41, v18, v2, vcc
	v_add_lshl_u32 v2, v17, v16, 1
	v_cmp_gt_i32_e32 vcc, s2, v10
	s_and_b32 s37, s37, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e32 v42, v18, v2, vcc
	v_add_lshl_u32 v2, v17, v3, 1
	v_cmp_gt_i32_e32 vcc, s2, v11
	v_and_b32_e32 v215, 32, v0
	v_and_b32_e32 v198, 0x1c0, v0
	v_cndmask_b32_e32 v43, v18, v2, vcc
	buffer_load_dwordx4 v[2:5], v36, s[36:39], 0 offen
	buffer_load_dwordx4 v[6:9], v37, s[36:39], 0 offen
	buffer_load_dwordx4 v[10:13], v38, s[36:39], 0 offen
	buffer_load_dwordx4 v[14:17], v1, s[36:39], 0 offen
	buffer_load_dwordx4 v[18:21], v40, s[36:39], 0 offen
	buffer_load_dwordx4 v[22:25], v41, s[36:39], 0 offen
	buffer_load_dwordx4 v[26:29], v42, s[36:39], 0 offen
	buffer_load_dwordx4 v[30:33], v43, s[36:39], 0 offen
	s_load_dwordx4 s[28:31], s[0:1], 0x38
	s_load_dword s22, s[0:1], 0x48
	v_and_b32_e32 v42, 16, v0
	v_lshrrev_b32_e32 v36, 6, v0
	v_lshlrev_b32_e32 v37, 1, v42
	v_lshrrev_b32_e32 v38, 1, v215
	v_or3_b32 v40, v36, v37, v38
	v_mov_b32_e32 v55, 0x2000
	v_and_b32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[40:41], s[2:3], s29, v40, v[214:215]
	v_lshlrev_b32_e32 v44, 7, v198
	v_lshlrev_b32_e32 v218, 4, v39
	v_lshl_or_b32 v55, v36, 10, v55
	s_mov_b64 s[20:21], s[10:11]
	s_movk_i32 s6, 0x60
	s_movk_i32 s7, 0xa0
	s_movk_i32 s10, 0xe0
	s_movk_i32 s11, 0x80
	s_movk_i32 s12, 0xc0
	v_lshl_or_b32 v44, v1, 8, v44
	v_xor_b32_e32 v45, v218, v38
	s_movk_i32 s2, 0x410
	v_lshrrev_b32_e32 v56, 6, v55
	v_or_b32_e32 v46, v44, v45
	v_bitop3_b32 v49, v44, s6, v45 bitop3:0x36
	v_bitop3_b32 v50, v44, s11, v45 bitop3:0x36
	v_bitop3_b32 v51, v44, s7, v45 bitop3:0x36
	v_bitop3_b32 v52, v44, s12, v45 bitop3:0x36
	v_bitop3_b32 v44, v44, s10, v45 bitop3:0x36
	v_mad_u32_u24 v45, v36, s2, 0
	v_or_b32_e32 v199, v56, v55
	s_mul_i32 s33, s15, s18
	s_mul_i32 s23, s28, s19
	v_add_u32_e32 v53, 0x87c0, v45
	v_add_u32_e32 v56, 0, v199
	s_add_i32 s35, s23, s33
	s_lshl_b32 s36, s29, 6
	v_lshlrev_b32_e32 v219, 4, v0
	v_and_b32_e32 v43, 0xf0, v0
	s_and_b32 s5, s5, 0xffff
	v_add_u32_e32 v57, 0x87c0, v56
	v_readfirstlane_b32 s10, v53
	v_lshl_add_u32 v41, s29, 3, v40
	v_xad_u32 v43, v219, v43, 0
	v_add_u32_e32 v47, 0, v46
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s38
	s_mov_b32 s15, s39
	v_add_lshl_u32 v54, v40, s35, 1
	v_add_u32_e32 v59, s36, v40
	v_add_u32_e32 v40, 0xc8c0, v45
	v_lshlrev_b32_e32 v39, 10, v39
	v_lshlrev_b32_e32 v42, 5, v42
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s37, v57
	v_xad_u32 v48, v46, 32, 0
	v_xad_u32 v46, v46, 64, 0
	v_add_u32_e32 v49, 0, v49
	v_add_u32_e32 v50, 0, v50
	v_add_u32_e32 v51, 0, v51
	v_add_u32_e32 v52, 0, v52
	v_add_u32_e32 v44, 0, v44
	v_add_lshl_u32 v58, v41, s35, 1
	v_add_u32_e32 v45, 0xc8c0, v56
	v_or3_b32 v39, v39, v42, v38
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v43, v[2:5]
	s_waitcnt vmcnt(6)
	ds_write_b128 v43, v[6:9] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v43, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v43, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v43, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v43, v[22:25] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v43, v[26:29] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v43, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[142:145], v47
	ds_read_b128 v[138:141], v48
	ds_read_b128 v[134:137], v46
	ds_read_b128 v[130:133], v49
	ds_read_b128 v[126:129], v50
	ds_read_b128 v[122:125], v51
	ds_read_b128 v[118:121], v52
	ds_read_b128 v[114:117], v44
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v54, s[12:15], 0 offen lds
	s_mov_b32 m0, s37
	v_readfirstlane_b32 s2, v40
	v_add_u32_e32 v60, s36, v41
	v_add_lshl_u32 v41, v59, s35, 1
	v_add_u32_e32 v221, v39, v218
	buffer_load_dwordx4 v58, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v45
	v_add_lshl_u32 v56, v60, s35, 1
	v_add_u32_e32 v39, 0, v221
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v41, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
	v_and_b32_e32 v200, 48, v0
	buffer_load_dwordx4 v56, s[12:15], 0 offen lds
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[2:5], v39 offset:34752
	ds_read_b128 v[18:21], v39 offset:34784
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[142:145], 0
	v_or_b32_e32 v44, v36, v200
	v_mad_u64_u32 v[48:49], s[2:3], s22, v44, v[214:215]
	v_lshrrev_b32_e32 v50, 4, v55
	s_movk_i32 s2, 0x440
	s_mul_i32 s34, s30, s18
	s_mul_i32 s30, s31, s19
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[138:141], v[2:17]
	ds_read_b128 v[18:21], v39 offset:34816
	ds_read_b128 v[22:25], v39 offset:34848
	v_mad_u32_u24 v51, v36, s2, 0
	v_or_b32_e32 v220, v50, v55
	s_add_i32 s28, s30, s34
	v_add_u32_e32 v50, 0, v220
	v_readfirstlane_b32 s31, v51
	v_lshl_add_u32 v49, s22, 3, v48
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[134:137], v[2:17]
	s_and_b32 s25, s25, 0xffff
	s_mov_b32 s26, s38
	s_mov_b32 s27, s39
	v_add_lshl_u32 v52, v48, s28, 1
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s31, v50
	v_add_lshl_u32 v53, v49, s28, 1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[130:133], v[2:17]
	ds_read_b128 v[18:21], v39 offset:34880
	ds_read_b128 v[22:25], v39 offset:34912
	s_add_i32 s35, s35, s36
	s_lshl_b32 s11, s22, 6
	v_add_lshl_u32 v54, v59, s35, 1
	v_add_u32_e32 v56, 0x4400, v51
	v_add_lshl_u32 v55, v60, s35, 1
	s_add_i32 s2, s28, s11
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[126:129], v[2:17]
	v_add_u32_e32 v57, 0x4400, v50
	v_add_lshl_u32 v48, s2, v48, 1
	v_add_lshl_u32 v49, s2, v49, 1
	s_movk_i32 s2, 0x100
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_movk_i32 s2, 0xff
	s_mov_b32 s17, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[122:125], v[2:17]
	ds_read_b128 v[18:21], v39 offset:34944
	ds_read_b128 v[22:25], v39 offset:34976
	ds_read_b128 v[40:43], v39 offset:35040
	ds_read_b128 v[44:47], v39 offset:35072
	buffer_load_dwordx4 v52, s[24:27], 0 offen lds
	s_mov_b32 m0, s31
	s_mov_b32 s6, s38
	buffer_load_dwordx4 v53, s[24:27], 0 offen lds
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[118:121], v[2:17]
	ds_read_b128 v[18:21], v39 offset:35008
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v56
	buffer_load_dwordx4 v54, s[12:15], 0 offen lds
	s_mov_b32 m0, s37
	s_mov_b32 s7, s39
	buffer_load_dwordx4 v55, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[114:117], v[2:17]
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v57
	v_cmp_lt_u32_e64 s[2:3], s2, v0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[142:145], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[138:141], v[18:33]
	ds_read_b128 v[40:43], v39 offset:35104
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[134:137], v[18:33]
	ds_read_b128 v[44:47], v39 offset:35136
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[130:133], v[18:33]
	ds_read_b128 v[40:43], v39 offset:35168
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[126:129], v[18:33]
	ds_read_b128 v[44:47], v39 offset:35200
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[122:125], v[18:33]
	ds_read_b128 v[40:43], v39 offset:35232
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v48, s[24:27], 0 offen lds
	s_mov_b32 m0, s10
	s_mov_b32 s10, 0xff800000
	buffer_load_dwordx4 v49, s[24:27], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[118:121], v[18:33]
	ds_read_b128 v[66:69], v39 offset:51392
	ds_read_b128 v[186:189], v39 offset:51424
	ds_read_b128 v[182:185], v39 offset:51456
	ds_read_b128 v[178:181], v39 offset:51488
	ds_read_b128 v[174:177], v39 offset:51520
	ds_read_b128 v[110:113], v39 offset:51552
	ds_read_b128 v[106:109], v39 offset:51584
	ds_read_b128 v[102:105], v39 offset:51616
	ds_read_b128 v[98:101], v39 offset:51648
	ds_read_b128 v[170:173], v39 offset:51680
	ds_read_b128 v[166:169], v39 offset:51712
	ds_read_b128 v[162:165], v39 offset:51744
	ds_read_b128 v[158:161], v39 offset:51776
	ds_read_b128 v[154:157], v39 offset:51808
	ds_read_b128 v[150:153], v39 offset:51840
	ds_read_b128 v[146:149], v39 offset:51872
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[114:117], v[18:33]
	.file	2 "/app/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v40, v3, v3
	v_max_f32_e32 v41, v2, v2
	v_max_f32_e32 v40, v41, v40
	v_max3_f32 v40, v40, v4, v5
	v_max3_f32 v40, v40, v6, v7
	v_max3_f32 v40, v40, v8, v9
	v_max3_f32 v40, v40, v10, v11
	v_max3_f32 v40, v40, v12, v13
	v_max3_f32 v40, v40, v14, v15
	v_max3_f32 v40, v40, v16, v17
	s_nop 1
	v_max3_f32 v40, v40, v18, v19
	v_max3_f32 v40, v40, v20, v21
	v_max3_f32 v40, v40, v22, v23
	v_max3_f32 v40, v40, v24, v25
	v_max3_f32 v40, v40, v26, v27
	v_max3_f32 v40, v40, v28, v29
	v_max3_f32 v40, v40, v30, v31
	v_max3_f32 v40, v40, v32, v33
	v_mov_b32_e32 v41, v40
	s_nop 1
	v_permlane32_swap_b32_e32 v40, v41
	v_max3_f32 v217, v40, v41, s10
	v_mov_b32_e32 v216, v33
	s_mov_b32 s10, 0x3e0293ee
	v_pk_mul_f32 v[40:41], v[216:217], s[10:11] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v2, v2, s10, -v41
	v_fma_f32 v3, v3, s10, -v41
	v_fma_f32 v4, v4, s10, -v41
	v_fma_f32 v5, v5, s10, -v41
	v_fma_f32 v6, v6, s10, -v41
	v_fma_f32 v7, v7, s10, -v41
	v_fma_f32 v8, v8, s10, -v41
	v_fma_f32 v9, v9, s10, -v41
	v_fma_f32 v10, v10, s10, -v41
	v_fma_f32 v11, v11, s10, -v41
	v_fma_f32 v12, v12, s10, -v41
	v_fma_f32 v13, v13, s10, -v41
	v_fma_f32 v14, v14, s10, -v41
	v_fma_f32 v15, v15, s10, -v41
	v_fma_f32 v16, v16, s10, -v41
	v_fma_f32 v17, v17, s10, -v41
	v_fma_f32 v18, v18, s10, -v41
	v_fma_f32 v19, v19, s10, -v41
	v_fma_f32 v20, v20, s10, -v41
	v_fma_f32 v21, v21, s10, -v41
	v_fma_f32 v22, v22, s10, -v41
	v_fma_f32 v23, v23, s10, -v41
	v_fma_f32 v24, v24, s10, -v41
	v_fma_f32 v25, v25, s10, -v41
	v_fma_f32 v26, v26, s10, -v41
	v_fma_f32 v27, v27, s10, -v41
	v_fma_f32 v28, v28, s10, -v41
	v_fma_f32 v29, v29, s10, -v41
	v_fma_f32 v30, v30, s10, -v41
	v_fma_f32 v31, v31, s10, -v41
	v_fma_f32 v32, v32, s10, -v41
	v_sub_f32_e32 v33, v40, v41
	v_sub_f32_e32 v39, 0xff800000, v41
	s_and_saveexec_b64 s[12:13], s[2:3]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:                                ; %.preheader
	s_or_b64 exec, exec, s[12:13]
	v_exp_f32_e32 v190, v2
	v_exp_f32_e32 v192, v3
	v_lshlrev_b32_e32 v2, 8, v0
	v_lshlrev_b32_e32 v3, 3, v0
	v_exp_f32_e32 v191, v4
	v_and_b32_e32 v2, 0xc00, v2
	v_and_b32_e32 v3, 24, v3
	v_lshlrev_b32_e32 v4, 7, v215
	v_or3_b32 v2, v2, v3, v4
	v_or_b32_e32 v201, v2, v37
	v_lshrrev_b32_e32 v3, 4, v2
	v_or_b32_e32 v2, 0x2000, v2
	v_lshrrev_b32_e32 v2, 4, v2
	v_and_b32_e32 v203, 0x3c0, v2
	v_add_u32_e32 v2, v200, v36
	s_load_dwordx2 s[2:3], s[0:1], 0x4c
	s_load_dword s12, s[0:1], 0x54
	v_and_b32_e32 v202, 0x1c0, v3
	s_lshl_b32 s0, s34, 1
	v_add_u32_e32 v3, 0x88, v2
	s_lshl_b32 s1, s30, 1
	v_or_b32_e32 v2, 0x80, v2
	s_add_i32 s1, s1, s0
	v_mul_lo_u32 v2, s22, v2
	v_mul_lo_u32 v3, s22, v3
	v_lshl_add_u32 v207, v2, 1, s1
	v_add3_u32 v2, v37, v38, v36
	v_lshl_add_u32 v206, v3, 1, s1
	s_lshl_b32 s1, s33, 1
	v_add_u32_e32 v3, 0xc8, v2
	s_lshl_b32 s13, s23, 1
	v_add_u32_e32 v2, 0xc0, v2
	s_add_i32 s13, s13, s1
	v_mul_lo_u32 v2, s29, v2
	v_lshl_add_u32 v209, v2, 1, s13
	v_or_b32_e32 v2, 64, v35
	v_exp_f32_e32 v194, v5
	v_exp_f32_e32 v193, v6
	v_exp_f32_e32 v196, v7
	v_exp_f32_e32 v195, v8
	v_exp_f32_e32 v210, v9
	v_exp_f32_e32 v197, v10
	v_exp_f32_e32 v212, v11
	v_exp_f32_e32 v211, v12
	v_exp_f32_e32 v232, v13
	v_exp_f32_e32 v213, v14
	v_exp_f32_e32 v235, v15
	v_exp_f32_e32 v233, v16
	v_exp_f32_e32 v237, v17
	v_exp_f32_e32 v236, v18
	v_exp_f32_e32 v239, v19
	v_exp_f32_e32 v238, v20
	v_exp_f32_e32 v241, v21
	v_exp_f32_e32 v240, v22
	v_exp_f32_e32 v242, v23
	v_exp_f32_e32 v243, v24
	v_exp_f32_e32 v245, v25
	v_exp_f32_e32 v244, v26
	v_exp_f32_e32 v247, v27
	v_exp_f32_e32 v246, v28
	v_exp_f32_e32 v249, v29
	v_exp_f32_e32 v248, v30
	v_exp_f32_e32 v251, v31
	v_exp_f32_e32 v250, v32
	v_exp_f32_e32 v252, v33
	v_exp_f32_e32 v216, v39
	v_mul_lo_u32 v2, s22, v2
	v_add_u32_e32 v223, s28, v2
	v_or_b32_e32 v2, 64, v34
	v_mul_lo_u32 v3, s29, v3
	v_mul_lo_u32 v2, s22, v2
	v_mov_b32_e32 v18, 0
	v_mul_u32_u24_e32 v204, 0x410, v36
	v_mul_u32_u24_e32 v222, 0x440, v36
	s_lshl_b32 s0, s22, 7
	v_lshl_add_u32 v208, v3, 1, s13
	s_lshl_b32 s1, s29, 7
	s_add_i32 s13, 0, 0x4400
	s_add_i32 s22, 0, 0x87c0
	v_add_u32_e32 v224, s28, v2
	v_mov_b32_e32 v205, 1.0
	s_movk_i32 s15, 0xffc0
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	s_mov_b32 s23, 0
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
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[142:145], 0
	s_mov_b32 s33, s17
	s_mov_b32 s17, s13
	v_mov_b32_e32 v226, v205
	v_mov_b32_e32 v225, v217
	s_setprio 0
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[138:141], v[66:81]
	v_add_f32_e32 v82, v190, v192
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[134:137], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[130:133], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[126:129], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[122:125], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[118:121], v[66:81]
	v_cvt_pk_f16_f32 v106, v236, v239
	v_cvt_pk_f16_f32 v107, v238, v241
	v_cvt_pk_f16_f32 v108, v240, v242
	v_cvt_pk_f16_f32 v109, v243, v245
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[114:117], v[66:81]
	v_add_f32_e32 v102, v82, v191
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[142:145], 0
	v_add_f32_e32 v98, v102, v194
	v_add_f32_e32 v98, v98, v193
	v_add_f32_e32 v98, v98, v196
	v_add_f32_e32 v98, v98, v195
	v_add_f32_e32 v98, v98, v210
	v_add_f32_e32 v98, v98, v197
	v_add_f32_e32 v98, v98, v212
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[138:141], v[82:97]
	v_add_f32_e32 v98, v98, v211
	v_add_f32_e32 v98, v98, v232
	v_add_f32_e32 v98, v98, v213
	v_add_f32_e32 v98, v98, v235
	v_add_f32_e32 v98, v98, v233
	v_add_f32_e32 v98, v98, v237
	v_add_f32_e32 v98, v98, v236
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[134:137], v[82:97]
	v_add_f32_e32 v98, v98, v239
	v_add_f32_e32 v98, v98, v238
	v_add_f32_e32 v98, v98, v241
	v_add_f32_e32 v98, v98, v240
	v_add_f32_e32 v98, v98, v242
	v_add_f32_e32 v98, v98, v243
	v_add_f32_e32 v98, v98, v245
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[130:133], v[82:97]
	v_add_f32_e32 v98, v98, v244
	v_add_f32_e32 v98, v98, v247
	v_add_f32_e32 v98, v98, v246
	v_add_f32_e32 v98, v98, v249
	v_add_f32_e32 v98, v98, v248
	v_add_f32_e32 v98, v98, v251
	v_add_f32_e32 v98, v98, v250
	v_mfma_f32_32x32x16_f16 v[82:97], v[158:161], v[126:129], v[82:97]
	v_add_f32_e32 v98, v98, v252
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_add_f32_e32 v205, v98, v99
	v_fmac_f32_e32 v205, v226, v216
	v_cvt_pk_f16_f32 v158, v190, v192
	v_mfma_f32_32x32x16_f16 v[82:97], v[154:157], v[122:125], v[82:97]
	v_cvt_pk_f16_f32 v159, v191, v194
	v_cvt_pk_f16_f32 v160, v193, v196
	v_cvt_pk_f16_f32 v161, v195, v210
	v_cvt_pk_f16_f32 v154, v197, v212
	v_cvt_pk_f16_f32 v155, v211, v232
	v_cvt_pk_f16_f32 v156, v213, v235
	v_cvt_pk_f16_f32 v157, v233, v237
	v_mfma_f32_32x32x16_f16 v[82:97], v[150:153], v[118:121], v[82:97]
	v_cvt_pk_f16_f32 v98, v244, v247
	v_cvt_pk_f16_f32 v99, v246, v249
	v_cvt_pk_f16_f32 v100, v248, v251
	v_cvt_pk_f16_f32 v101, v250, v252
	v_mfma_f32_32x32x16_f16 v[82:97], v[146:149], v[114:117], v[82:97]
	s_setprio 1
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s13, s23, 1
	s_cmp_lt_i32 s13, 2
	s_cselect_b32 s30, s13, 0
	s_lshl_b32 s28, s30, 13
	s_lshl_b32 s13, s30, 14
	s_add_i32 s29, s13, 0
	s_ashr_i32 s13, s28, 5
	s_add_i32 s14, s29, s13
	s_add_i32 s31, s14, 0x87c0
	v_add_u32_e32 v102, s31, v204
	v_add_u32_e32 v103, v218, v209
	v_readfirstlane_b32 s13, v102
	v_add_u32_e32 v102, s31, v199
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v102
	buffer_load_dwordx4 v103, s[4:7], 0 offen lds
	v_add_u32_e32 v103, v218, v208
	s_mov_b32 m0, s13
	v_add_u32_e32 v102, s33, v201
	buffer_load_dwordx4 v103, s[4:7], 0 offen lds
	v_add_u32_e32 v103, v102, v202
	v_add_u32_e32 v104, v102, v203
	ds_read_b64_tr_b16 v[210:211], v103
	ds_read_b64_tr_b16 v[226:227], v103 offset:64
	ds_read_b64_tr_b16 v[194:195], v103 offset:128
	ds_read_b64_tr_b16 v[170:171], v103 offset:192
	ds_read_b64_tr_b16 v[212:213], v104 offset:8192
	ds_read_b64_tr_b16 v[228:229], v104 offset:8256
	ds_read_b64_tr_b16 v[196:197], v104 offset:8320
	ds_read_b64_tr_b16 v[172:173], v104 offset:8384
	ds_read_b64_tr_b16 v[230:231], v103 offset:256
	ds_read_b64_tr_b16 v[182:183], v103 offset:320
	ds_read_b64_tr_b16 v[178:179], v103 offset:384
	ds_read_b64_tr_b16 v[174:175], v103 offset:448
	ds_read_b64_tr_b16 v[232:233], v104 offset:8448
	ds_read_b64_tr_b16 v[184:185], v104 offset:8512
	ds_read_b64_tr_b16 v[180:181], v104 offset:8576
	ds_read_b64_tr_b16 v[176:177], v104 offset:8640
	ds_read_b64_tr_b16 v[186:187], v103 offset:512
	ds_read_b64_tr_b16 v[166:167], v103 offset:576
	ds_read_b64_tr_b16 v[162:163], v103 offset:640
	ds_read_b64_tr_b16 v[150:151], v103 offset:704
	ds_read_b64_tr_b16 v[188:189], v104 offset:8704
	ds_read_b64_tr_b16 v[168:169], v104 offset:8768
	ds_read_b64_tr_b16 v[164:165], v104 offset:8832
	ds_read_b64_tr_b16 v[152:153], v104 offset:8896
	ds_read_b64_tr_b16 v[190:191], v103 offset:768
	ds_read_b64_tr_b16 v[146:147], v103 offset:832
	ds_read_b64_tr_b16 v[110:111], v103 offset:896
	ds_read_b64_tr_b16 v[102:103], v103 offset:960
	ds_read_b64_tr_b16 v[192:193], v104 offset:8960
	ds_read_b64_tr_b16 v[148:149], v104 offset:9024
	ds_read_b64_tr_b16 v[112:113], v104 offset:9088
	ds_read_b64_tr_b16 v[104:105], v104 offset:9152
	; sched_barrier mask(0x00000000)
	v_pk_mul_f32 v[32:33], v[32:33], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[216:217] op_sel_hi:[1,0]
	s_barrier
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[18:33], v[210:213], v[158:161], v[18:33]
	s_setprio 0
	v_mul_f32_e64 v64, v64, v216
	v_mul_f32_e64 v65, v65, v216
	v_mul_f32_e64 v62, v62, v216
	v_mul_f32_e64 v63, v63, v216
	v_pk_mul_f32 v[60:61], v[60:61], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[216:217] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[50:65], v[226:229], v[158:161], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[194:197], v[158:161], v[34:49]
	v_mfma_f32_32x32x16_f16 v[2:17], v[170:173], v[158:161], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[230:233], v[154:157], v[18:33]
	v_mfma_f32_32x32x16_f16 v[50:65], v[182:185], v[154:157], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[178:181], v[154:157], v[34:49]
	v_mfma_f32_32x32x16_f16 v[2:17], v[174:177], v[154:157], v[2:17]
	v_max_f32_e32 v154, v67, v67
	v_max_f32_e32 v155, v66, v66
	v_max_f32_e32 v154, v155, v154
	v_max3_f32 v154, v154, v68, v69
	v_max3_f32 v154, v154, v70, v71
	v_max3_f32 v154, v154, v72, v73
	v_max3_f32 v154, v154, v74, v75
	v_max3_f32 v154, v154, v76, v77
	v_max3_f32 v154, v154, v78, v79
	v_max3_f32 v154, v154, v80, v81
	v_max3_f32 v154, v154, v82, v83
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[18:33], v[186:189], v[106:109], v[18:33]
	v_max3_f32 v154, v154, v84, v85
	v_max3_f32 v154, v154, v86, v87
	v_max3_f32 v154, v154, v88, v89
	v_max3_f32 v154, v154, v90, v91
	v_max3_f32 v154, v154, v92, v93
	v_max3_f32 v154, v154, v94, v95
	v_max3_f32 v154, v154, v96, v97
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[50:65], v[166:169], v[106:109], v[50:65]
	v_mov_b32_e32 v155, v154
	s_nop 1
	v_permlane32_swap_b32_e32 v154, v155
	v_max3_f32 v217, v225, v154, v155
	v_mul_f32_e32 v234, 0x3e0293ee, v217
	v_fma_f32 v66, v66, s10, -v234
	v_fma_f32 v67, v67, s10, -v234
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[106:109], v[34:49]
	v_fma_f32 v68, v68, s10, -v234
	v_fma_f32 v69, v69, s10, -v234
	v_fma_f32 v70, v70, s10, -v234
	v_fma_f32 v71, v71, s10, -v234
	v_fma_f32 v72, v72, s10, -v234
	v_fma_f32 v73, v73, s10, -v234
	v_fma_f32 v74, v74, s10, -v234
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[2:17], v[150:153], v[106:109], v[2:17]
	v_fma_f32 v75, v75, s10, -v234
	v_fma_f32 v76, v76, s10, -v234
	v_fma_f32 v77, v77, s10, -v234
	v_fma_f32 v78, v78, s10, -v234
	v_fma_f32 v79, v79, s10, -v234
	v_fma_f32 v80, v80, s10, -v234
	v_fma_f32 v81, v81, s10, -v234
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[18:33], v[190:193], v[98:101], v[18:33]
	v_fma_f32 v82, v82, s10, -v234
	v_fma_f32 v83, v83, s10, -v234
	v_fma_f32 v84, v84, s10, -v234
	v_fma_f32 v85, v85, s10, -v234
	v_fma_f32 v86, v86, s10, -v234
	v_fma_f32 v87, v87, s10, -v234
	v_fma_f32 v88, v88, s10, -v234
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[146:149], v[98:101], v[50:65]
	v_fma_f32 v89, v89, s10, -v234
	v_fma_f32 v90, v90, s10, -v234
	v_fma_f32 v91, v91, s10, -v234
	v_fma_f32 v92, v92, s10, -v234
	v_fma_f32 v93, v93, s10, -v234
	v_fma_f32 v94, v94, s10, -v234
	v_fma_f32 v95, v95, s10, -v234
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[34:49], v[110:113], v[98:101], v[34:49]
	v_fma_f32 v96, v96, s10, -v234
	v_fma_f32 v97, v97, s10, -v234
	v_fma_f32 v106, v225, s10, -v234
	v_exp_f32_e32 v190, v66
	v_exp_f32_e32 v192, v67
	v_exp_f32_e32 v191, v68
	v_exp_f32_e32 v194, v69
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[102:105], v[98:101], v[2:17]
	v_exp_f32_e32 v193, v70
	v_exp_f32_e32 v196, v71
	v_exp_f32_e32 v195, v72
	v_exp_f32_e32 v210, v73
	v_exp_f32_e32 v197, v74
	v_exp_f32_e32 v212, v75
	v_exp_f32_e32 v211, v76
	v_exp_f32_e32 v232, v77
	v_exp_f32_e32 v213, v78
	v_exp_f32_e32 v235, v79
	v_exp_f32_e32 v233, v80
	v_exp_f32_e32 v237, v81
	v_exp_f32_e32 v236, v82
	v_exp_f32_e32 v239, v83
	v_exp_f32_e32 v238, v84
	v_exp_f32_e32 v241, v85
	v_exp_f32_e32 v240, v86
	v_exp_f32_e32 v242, v87
	v_exp_f32_e32 v243, v88
	v_exp_f32_e32 v245, v89
	v_exp_f32_e32 v244, v90
	v_exp_f32_e32 v247, v91
	v_exp_f32_e32 v246, v92
	v_exp_f32_e32 v249, v93
	v_exp_f32_e32 v248, v94
	v_exp_f32_e32 v251, v95
	v_exp_f32_e32 v250, v96
	v_exp_f32_e32 v252, v97
	v_exp_f32_e32 v216, v106
	s_setprio 1
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s13, s23, 13
	s_lshl_b32 s23, s23, 14
	s_add_i32 s23, s23, 0
	s_ashr_i32 s13, s13, 3
	s_add_i32 s13, s23, s13
	v_add_u32_e32 v66, s13, v222
	v_add_u32_e32 v67, v218, v207
	v_readfirstlane_b32 s23, v66
	v_add_u32_e32 v66, s13, v220
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s23, v66
	buffer_load_dwordx4 v67, s[24:27], 0 offen lds
	v_add_u32_e32 v67, v218, v206
	s_mov_b32 m0, s23
	v_add_u32_e32 v70, s22, v221
	buffer_load_dwordx4 v67, s[24:27], 0 offen lds
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[186:189], v70 offset:32
	ds_read_b128 v[182:185], v70 offset:64
	ds_read_b128 v[178:181], v70 offset:96
	ds_read_b128 v[174:177], v70 offset:128
	ds_read_b128 v[110:113], v70 offset:160
	ds_read_b128 v[106:109], v70 offset:192
	ds_read_b128 v[102:105], v70 offset:224
	ds_read_b128 v[98:101], v70 offset:256
	ds_read_b128 v[170:173], v70 offset:288
	ds_read_b128 v[166:169], v70 offset:320
	ds_read_b128 v[162:165], v70 offset:352
	ds_read_b128 v[158:161], v70 offset:384
	ds_read_b128 v[154:157], v70 offset:416
	ds_read_b128 v[150:153], v70 offset:448
	ds_read_b128 v[146:149], v70 offset:480
	; sched_barrier mask(0x00000000)
	s_add_i32 s15, s15, 64
	v_add_u32_e32 v206, s0, v206
	v_add_u32_e32 v207, s0, v207
	v_add_u32_e32 v208, s1, v208
	v_add_u32_e32 v209, s1, v209
	v_add_u32_e32 v223, s11, v223
	v_add_u32_e32 v224, s11, v224
	s_cmpk_lt_u32 s15, 0x3f00
	s_mov_b32 s22, s31
	s_mov_b32 s23, s30
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[142:145], 0
	v_lshrrev_b32_e32 v66, 1, v198
	v_and_b32_e32 v226, 0xff, v0
	s_ashr_i32 s1, s28, 3
	v_or_b32_e32 v225, v66, v1
	s_add_i32 s0, s16, 0xffffc100
	v_lshl_or_b32 v66, s18, 20, v226
	s_add_i32 s10, 0, 0x10a00
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[138:141], v[82:97]
	s_add_i32 s5, s29, s1
	v_and_b32_e32 v0, 0x100, v0
	s_cmp_lt_i32 s0, 1
	v_lshlrev_b32_e32 v228, 2, v1
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	v_add_lshl_u32 v0, v66, s16, 2
	v_add_f32_e32 v1, v190, v192
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[134:137], v[82:97]
	v_lshl_add_u32 v227, s19, 16, v0
	v_add_f32_e32 v0, v1, v191
	v_add_f32_e32 v0, v0, v194
	v_add_f32_e32 v0, v0, v193
	v_add_f32_e32 v0, v0, v196
	v_add_f32_e32 v0, v0, v195
	v_add_f32_e32 v0, v0, v210
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[130:133], v[82:97]
	v_add_f32_e32 v0, v0, v197
	v_add_f32_e32 v0, v0, v212
	v_add_f32_e32 v0, v0, v211
	v_add_f32_e32 v0, v0, v232
	v_add_f32_e32 v0, v0, v213
	v_add_f32_e32 v0, v0, v235
	v_add_f32_e32 v0, v0, v233
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[126:129], v[82:97]
	v_add_f32_e32 v0, v0, v237
	v_add_f32_e32 v0, v0, v236
	v_add_f32_e32 v0, v0, v239
	v_add_f32_e32 v0, v0, v238
	v_add_f32_e32 v0, v0, v241
	v_add_f32_e32 v0, v0, v240
	v_add_f32_e32 v0, v0, v242
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[122:125], v[82:97]
	v_add_f32_e32 v0, v0, v243
	v_add_f32_e32 v0, v0, v245
	v_add_f32_e32 v0, v0, v244
	v_add_f32_e32 v0, v0, v247
	v_add_f32_e32 v0, v0, v246
	v_add_f32_e32 v0, v0, v249
	v_add_f32_e32 v0, v0, v248
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[118:121], v[82:97]
	v_add_f32_e32 v0, v0, v251
	v_add_f32_e32 v0, v0, v250
	v_add_f32_e32 v0, v0, v252
	v_mov_b32_e32 v1, v0
	v_or_b32_e32 v231, v202, v201
	s_nop 0
	v_permlane32_swap_b32_e32 v0, v1
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[114:117], v[82:97]
	v_add_f32_e32 v230, v0, v1
	v_cvt_pk_f16_f32 v179, v211, v232
	v_add_u32_e32 v0, s17, v231
	v_add_u32_e32 v232, v203, v201
	v_cvt_pk_f16_f32 v182, v190, v192
	v_cvt_pk_f16_f32 v183, v191, v194
	v_cvt_pk_f16_f32 v184, v193, v196
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[98:113], v[98:101], v[142:145], 0
	v_cvt_pk_f16_f32 v185, v195, v210
	v_cvt_pk_f16_f32 v178, v197, v212
	v_cvt_pk_f16_f32 v180, v213, v235
	v_cvt_pk_f16_f32 v181, v233, v237
	v_cvt_pk_f16_f32 v174, v236, v239
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[138:141], v[98:113]
	s_barrier
	v_add_u32_e32 v1, s17, v232
	v_add_u32_e32 v233, v201, v202
	ds_read_b64_tr_b16 v[192:193], v0
	ds_read_b64_tr_b16 v[194:195], v1 offset:8192
	ds_read_b64_tr_b16 v[212:213], v1 offset:8256
	ds_read_b64_tr_b16 v[236:237], v0 offset:512
	v_add_u32_e32 v235, s17, v233
	v_lshlrev_b32_e32 v253, 2, v198
	v_lshl_add_u32 v254, v200, 8, s10
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[134:137], v[98:113]
	v_lshlrev_b32_e32 v229, 1, v198
	v_cvt_pk_f16_f32 v175, v238, v241
	v_cvt_pk_f16_f32 v176, v240, v242
	v_cvt_pk_f16_f32 v177, v243, v245
	v_mul_f32_e64 v80, v32, v216
	v_mul_f32_e64 v81, v33, v216
	v_pk_mul_f32 v[78:79], v[30:31], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[28:29], v[216:217] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[130:133], v[98:113]
	v_mul_f32_e64 v74, v26, v216
	v_mul_f32_e64 v75, v27, v216
	v_mul_f32_e64 v72, v24, v216
	v_mul_f32_e64 v73, v25, v216
	v_mul_f32_e64 v70, v22, v216
	v_mul_f32_e64 v71, v23, v216
	v_pk_mul_f32 v[68:69], v[20:21], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[18:19], v[216:217] op_sel_hi:[1,0]
	v_fmac_f32_e32 v230, v205, v216
	v_cvt_pk_f16_f32 v170, v244, v247
	v_mfma_f32_32x32x16_f16 v[98:113], v[158:161], v[126:129], v[98:113]
	ds_read_b64_tr_b16 v[240:241], v235 offset:256
	ds_read_b64_tr_b16 v[198:199], v235 offset:320
	ds_read_b64_tr_b16 v[166:167], v235 offset:384
	ds_read_b64_tr_b16 v[158:159], v235 offset:192
	ds_read_b64_tr_b16 v[242:243], v1 offset:8448
	ds_read_b64_tr_b16 v[200:201], v1 offset:8512
	ds_read_b64_tr_b16 v[196:197], v1 offset:8320
	ds_read_b64_tr_b16 v[160:161], v1 offset:8384
	v_cvt_pk_f16_f32 v171, v246, v249
	v_pk_mul_f32 v[32:33], v[48:49], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[46:47], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[44:45], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[42:43], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[40:41], v[216:217] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[98:113], v[154:157], v[122:125], v[98:113]
	ds_read_b64_tr_b16 v[202:203], v235 offset:576
	ds_read_b64_tr_b16 v[186:187], v235 offset:640
	ds_read_b64_tr_b16 v[154:155], v235 offset:704
	ds_read_b64_tr_b16 v[162:163], v235 offset:448
	ds_read_b64_tr_b16 v[238:239], v1 offset:8704
	ds_read_b64_tr_b16 v[204:205], v1 offset:8768
	ds_read_b64_tr_b16 v[168:169], v1 offset:8576
	ds_read_b64_tr_b16 v[164:165], v1 offset:8640
	v_pk_mul_f32 v[22:23], v[38:39], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[36:37], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[34:35], v[216:217] op_sel_hi:[1,0]
	v_max_f32_e32 v0, v83, v83
	v_pk_mul_f32 v[16:17], v[16:17], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[216:217] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[98:113], v[150:153], v[118:121], v[98:113]
	ds_read_b64_tr_b16 v[246:247], v1 offset:8960
	ds_read_b64_tr_b16 v[208:209], v1 offset:9024
	ds_read_b64_tr_b16 v[188:189], v1 offset:8832
	ds_read_b64_tr_b16 v[156:157], v1 offset:8896
	ds_read_b64_tr_b16 v[244:245], v235 offset:768
	ds_read_b64_tr_b16 v[206:207], v235 offset:832
	ds_read_b64_tr_b16 v[190:191], v235 offset:896
	ds_read_b64_tr_b16 v[150:151], v235 offset:960
	v_pk_mul_f32 v[12:13], v[12:13], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[216:217] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[192:195], v[182:185], v[66:81]
	ds_read_b64_tr_b16 v[210:211], v235 offset:64
	ds_read_b64_tr_b16 v[194:195], v235 offset:128
	ds_read_b64_tr_b16 v[192:193], v1 offset:9088
	ds_read_b64_tr_b16 v[152:153], v1 offset:9152
	v_max_f32_e32 v1, v82, v82
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[194:197], v[182:185], v[18:33]
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_mul_f32_e64 v64, v64, v216
	v_mul_f32_e64 v65, v65, v216
	v_pk_mul_f32 v[62:63], v[62:63], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[216:217] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[98:113], v[146:149], v[114:117], v[98:113]
	v_mul_f32_e64 v56, v56, v216
	v_mul_f32_e64 v57, v57, v216
	v_mul_f32_e64 v54, v54, v216
	v_mul_f32_e64 v55, v55, v216
	v_mul_f32_e64 v52, v52, v216
	v_mul_f32_e64 v53, v53, v216
	v_pk_mul_f32 v[50:51], v[50:51], v[216:217] op_sel_hi:[1,0]
	s_mov_b32 s4, 0x3e0293ee
	v_cvt_pk_f16_f32 v172, v248, v251
	v_cvt_pk_f16_f32 v173, v250, v252
	v_mfma_f32_32x32x16_f16 v[18:33], v[166:169], v[178:181], v[18:33]
	s_nop 0
	v_max3_f32 v0, v0, v98, v99
	v_max3_f32 v0, v0, v100, v101
	v_max3_f32 v0, v0, v102, v103
	v_max3_f32 v0, v0, v104, v105
	v_max3_f32 v0, v0, v106, v107
	v_max3_f32 v0, v0, v108, v109
	v_max3_f32 v0, v0, v110, v111
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[182:185], v[2:17]
	v_max3_f32 v0, v0, v112, v113
	v_mov_b32_e32 v1, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v1
	v_max3_f32 v147, v217, v0, v1
	v_mov_b32_e32 v146, v113
	v_pk_mul_f32 v[0:1], v[146:147], s[4:5] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[18:33], v[186:189], v[174:177], v[18:33]
	v_fma_f32 v34, v82, s4, -v1
	v_fma_f32 v35, v83, s4, -v1
	v_fma_f32 v36, v84, s4, -v1
	v_fma_f32 v37, v85, s4, -v1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[2:17], v[162:165], v[178:181], v[2:17]
	v_fma_f32 v82, v98, s4, -v1
	v_fma_f32 v83, v99, s4, -v1
	v_fma_f32 v84, v100, s4, -v1
	v_fma_f32 v85, v101, s4, -v1
	v_fma_f32 v38, v86, s4, -v1
	v_fma_f32 v39, v87, s4, -v1
	v_fma_f32 v40, v88, s4, -v1
	v_mfma_f32_32x32x16_f16 v[50:65], v[210:213], v[182:185], v[50:65]
	v_add_u32_e32 v182, s14, v221
	v_fma_f32 v41, v89, s4, -v1
	v_fma_f32 v42, v90, s4, -v1
	v_fma_f32 v43, v91, s4, -v1
	v_fma_f32 v44, v92, s4, -v1
	v_fma_f32 v45, v93, s4, -v1
	v_fma_f32 v46, v94, s4, -v1
	v_mfma_f32_32x32x16_f16 v[18:33], v[190:193], v[170:173], v[18:33]
	v_exp_f32_e32 v190, v34
	v_exp_f32_e32 v191, v35
	v_exp_f32_e32 v192, v36
	v_exp_f32_e32 v193, v37
	ds_read_b128 v[34:37], v182 offset:34752
	v_fma_f32 v47, v95, s4, -v1
	v_fma_f32 v48, v96, s4, -v1
	v_mfma_f32_32x32x16_f16 v[2:17], v[154:157], v[174:177], v[2:17]
	v_fma_f32 v49, v97, s4, -v1
	v_exp_f32_e32 v194, v38
	v_exp_f32_e32 v195, v39
	v_exp_f32_e32 v165, v40
	v_exp_f32_e32 v196, v41
	v_exp_f32_e32 v197, v42
	v_exp_f32_e32 v146, v48
	v_mfma_f32_32x32x16_f16 v[50:65], v[198:201], v[178:181], v[50:65]
	v_exp_f32_e32 v198, v43
	v_exp_f32_e32 v199, v44
	v_exp_f32_e32 v200, v45
	v_exp_f32_e32 v201, v46
	v_exp_f32_e32 v149, v49
	v_fma_f32 v86, v102, s4, -v1
	v_fma_f32 v87, v103, s4, -v1
	v_mfma_f32_32x32x16_f16 v[2:17], v[150:153], v[170:173], v[2:17]
	v_exp_f32_e32 v150, v82
	v_exp_f32_e32 v151, v83
	v_exp_f32_e32 v152, v84
	v_exp_f32_e32 v153, v85
	ds_read_b128 v[82:85], v182 offset:34784
	v_fma_f32 v88, v104, s4, -v1
	v_fma_f32 v89, v105, s4, -v1
	v_mfma_f32_32x32x16_f16 v[50:65], v[202:205], v[174:177], v[50:65]
	v_exp_f32_e32 v202, v47
	v_exp_f32_e32 v148, v86
	v_exp_f32_e32 v162, v87
	v_exp_f32_e32 v163, v88
	v_exp_f32_e32 v164, v89
	ds_read_b128 v[86:89], v182 offset:34816
	v_fma_f32 v90, v106, s4, -v1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[34:37], v[142:145], 0
	v_fma_f32 v91, v107, s4, -v1
	v_fma_f32 v92, v108, s4, -v1
	v_fma_f32 v93, v109, s4, -v1
	v_fma_f32 v94, v110, s4, -v1
	v_fma_f32 v95, v111, s4, -v1
	v_fma_f32 v96, v112, s4, -v1
	v_exp_f32_e32 v155, v90
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[138:141], v[34:49]
	ds_read_b128 v[82:85], v182 offset:34848
	v_exp_f32_e32 v156, v91
	v_exp_f32_e32 v157, v92
	v_exp_f32_e32 v158, v93
	v_exp_f32_e32 v159, v94
	v_exp_f32_e32 v160, v95
	v_exp_f32_e32 v161, v96
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[34:49], v[86:89], v[134:137], v[34:49]
	ds_read_b128 v[86:89], v182 offset:34880
	ds_read_b128 v[90:93], v182 offset:34912
	ds_read_b128 v[94:97], v182 offset:34944
	ds_read_b128 v[98:101], v182 offset:34976
	ds_read_b128 v[102:105], v182 offset:35008
	ds_read_b128 v[106:109], v182 offset:35040
	v_add3_u32 v186, v214, v223, s11
	v_add_u32_e32 v187, 1, v186
	v_add_u32_e32 v188, 2, v186
	v_add_u32_e32 v189, 3, v186
	s_mov_b32 s26, s6
	v_sub_f32_e32 v0, v0, v1
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[34:49], v[82:85], v[130:133], v[34:49]
	v_add3_u32 v82, v214, v224, s11
	v_add_u32_e32 v83, 1, v82
	v_add_u32_e32 v84, 2, v82
	v_add_u32_e32 v85, 3, v82
	s_mov_b32 s27, s7
	v_exp_f32_e32 v154, v0
	v_sub_f32_e32 v0, v234, v1
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[34:49], v[86:89], v[126:129], v[34:49]
	v_add_u32_e32 v86, 4, v82
	v_add_u32_e32 v87, 5, v82
	v_add_u32_e32 v88, 6, v82
	v_add_u32_e32 v89, 7, v82
	v_exp_f32_e32 v0, v0
	s_nop 0
	v_pk_mul_f32 v[32:33], v[32:33], v[0:1] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[240:243], v[178:181], v[66:81]
	v_mul_f32_e64 v30, v30, v0
	v_mul_f32_e64 v31, v31, v0
	v_mul_f32_e64 v28, v28, v0
	v_mul_f32_e64 v29, v29, v0
	v_mul_f32_e64 v26, v26, v0
	v_mul_f32_e64 v27, v27, v0
	v_pk_mul_f32 v[24:25], v[24:25], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[0:1] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[34:49], v[90:93], v[122:125], v[34:49]
	v_add_u32_e32 v90, 4, v186
	v_add_u32_e32 v91, 5, v186
	v_add_u32_e32 v92, 6, v186
	v_add_u32_e32 v93, 7, v186
	v_mul_f32_e64 v16, v16, v0
	v_mul_f32_e64 v17, v17, v0
	v_pk_mul_f32 v[14:15], v[14:15], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[0:1] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[236:239], v[174:177], v[66:81]
	v_mul_f32_e64 v10, v10, v0
	v_mul_f32_e64 v11, v11, v0
	v_mul_f32_e64 v8, v8, v0
	v_mul_f32_e64 v9, v9, v0
	v_mul_f32_e64 v6, v6, v0
	v_mul_f32_e64 v7, v7, v0
	v_pk_mul_f32 v[4:5], v[4:5], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[0:1] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[34:49], v[94:97], v[118:121], v[34:49]
	v_add_u32_e32 v94, s10, v219
	v_mfma_f32_32x32x16_f16 v[66:81], v[244:247], v[170:173], v[66:81]
	v_mfma_f32_32x32x16_f16 v[50:65], v[206:209], v[170:173], v[50:65]
	ds_read_b128 v[110:113], v182 offset:35072
	ds_read_b128 v[166:169], v182 offset:35104
	ds_read_b128 v[170:173], v182 offset:35136
	ds_read_b128 v[174:177], v182 offset:35168
	ds_read_b128 v[178:181], v182 offset:35200
	ds_read_b128 v[182:185], v182 offset:35232
	ds_write_b128 v94, v[82:85]
	ds_write_b128 v94, v[186:189] offset:8192
	v_add3_u32 v82, v254, v253, v218
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_pk_mul_f32 v[80:81], v[80:81], v[0:1] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[34:49], v[98:101], v[114:117], v[34:49]
	ds_read2st64_b32 v[98:99], v82 offset1:8
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v94, v[86:89]
	ds_write_b128 v94, v[90:93] offset:8192
	v_add_u32_e32 v100, s5, v222
	v_lshlrev_b32_e32 v98, 1, v98
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[142:145], 0
	v_readfirstlane_b32 s6, v100
	s_mov_b32 m0, s6
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v98, s[24:27], 0 offen lds
	v_add_u32_e32 v98, s5, v220
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[138:141], v[82:97]
	v_readfirstlane_b32 s6, v98
	v_lshlrev_b32_e32 v99, 1, v99
	s_mov_b32 m0, s6
	v_add_u32_e32 v144, s13, v233
	buffer_load_dwordx4 v99, s[24:27], 0 offen lds
	s_waitcnt vmcnt(2) lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[134:137], v[82:97]
	s_barrier
	v_add_u32_e32 v141, s13, v232
	v_cvt_pk_f16_f32 v110, v190, v191
	v_cvt_pk_f16_f32 v111, v192, v193
	v_cvt_pk_f16_f32 v112, v194, v195
	v_cvt_pk_f16_f32 v113, v165, v196
	v_pk_mul_f32 v[78:79], v[78:79], v[0:1] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[130:133], v[82:97]
	v_mul_f32_e64 v76, v76, v0
	v_mul_f32_e64 v77, v77, v0
	v_mul_f32_e64 v74, v74, v0
	v_mul_f32_e64 v75, v75, v0
	v_mul_f32_e64 v72, v72, v0
	v_mul_f32_e64 v73, v73, v0
	v_pk_mul_f32 v[70:71], v[70:71], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[0:1] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v106, v197, v198
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[126:129], v[82:97]
	v_add_u32_e32 v128, s13, v231
	v_cvt_pk_f16_f32 v107, v199, v200
	v_cvt_pk_f16_f32 v108, v201, v202
	v_cvt_pk_f16_f32 v109, v146, v149
	v_cvt_pk_f16_f32 v102, v150, v151
	v_cvt_pk_f16_f32 v103, v152, v153
	v_cvt_pk_f16_f32 v104, v148, v162
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[122:125], v[82:97]
	v_cvt_pk_f16_f32 v105, v163, v164
	v_mul_f32_e64 v64, v64, v0
	v_mul_f32_e64 v65, v65, v0
	v_mul_f32_e64 v62, v62, v0
	v_mul_f32_e64 v63, v63, v0
	v_pk_mul_f32 v[60:61], v[60:61], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[0:1] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[118:121], v[82:97]
	ds_read_b64_tr_b16 v[118:119], v144 offset:256
	ds_read_b64_tr_b16 v[120:121], v141 offset:8448
	ds_read_b64_tr_b16 v[122:123], v128
	ds_read_b64_tr_b16 v[124:125], v141 offset:8192
	ds_read_b64_tr_b16 v[126:127], v141 offset:8256
	ds_read_b64_tr_b16 v[128:129], v128 offset:512
	v_pk_mul_f32 v[52:53], v[52:53], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[0:1] op_sel_hi:[1,0]
	v_add_f32_e32 v98, v190, v191
	v_add_f32_e32 v98, v192, v98
	v_add_f32_e32 v98, v193, v98
	v_add_f32_e32 v98, v194, v98
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[66:81], v[122:125], v[110:113], v[66:81]
	v_add_f32_e32 v98, v195, v98
	v_add_f32_e32 v98, v165, v98
	v_add_f32_e32 v140, v196, v98
	v_cvt_pk_f16_f32 v98, v155, v156
	v_cvt_pk_f16_f32 v99, v157, v158
	v_cvt_pk_f16_f32 v100, v159, v160
	v_cvt_pk_f16_f32 v101, v161, v154
	v_mfma_f32_32x32x16_f16 v[66:81], v[118:121], v[106:109], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[114:117], v[82:97]
	ds_read_b64_tr_b16 v[130:131], v141 offset:8704
	ds_read_b64_tr_b16 v[114:115], v144 offset:320
	ds_read_b64_tr_b16 v[132:133], v144 offset:384
	ds_read_b64_tr_b16 v[136:137], v144 offset:192
	ds_read_b64_tr_b16 v[116:117], v141 offset:8512
	ds_read_b64_tr_b16 v[122:123], v141 offset:8320
	ds_read_b64_tr_b16 v[138:139], v141 offset:8384
	ds_read_b64_tr_b16 v[142:143], v141 offset:8768
	ds_read_b64_tr_b16 v[134:135], v141 offset:8576
	ds_read_b64_tr_b16 v[118:119], v141 offset:8640
	ds_read_b64_tr_b16 v[168:169], v141 offset:8960
	ds_read_b64_tr_b16 v[172:173], v141 offset:9024
	ds_read_b64_tr_b16 v[176:177], v141 offset:8832
	ds_read_b64_tr_b16 v[180:181], v141 offset:8896
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[128:131], v[102:105], v[66:81]
	ds_read_b64_tr_b16 v[166:167], v144 offset:768
	ds_read_b64_tr_b16 v[170:171], v144 offset:832
	ds_read_b64_tr_b16 v[128:129], v144 offset:896
	ds_read_b64_tr_b16 v[182:183], v144 offset:960
	ds_read_b64_tr_b16 v[124:125], v144 offset:64
	ds_read_b64_tr_b16 v[120:121], v144 offset:128
	ds_read_b64_tr_b16 v[130:131], v141 offset:9088
	ds_read_b64_tr_b16 v[184:185], v141 offset:9152
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[50:65], v[124:127], v[110:113], v[50:65]
	v_add_f32_e32 v124, v197, v140
	ds_read_b64_tr_b16 v[140:141], v144 offset:576
	v_add_f32_e32 v124, v198, v124
	v_add_f32_e32 v124, v199, v124
	v_add_f32_e32 v124, v200, v124
	v_add_f32_e32 v124, v201, v124
	v_add_f32_e32 v145, v202, v124
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[18:33], v[120:123], v[110:113], v[18:33]
	v_mfma_f32_32x32x16_f16 v[2:17], v[136:139], v[110:113], v[2:17]
	v_mfma_f32_32x32x16_f16 v[50:65], v[114:117], v[106:109], v[50:65]
	v_max_f32_e32 v114, v35, v35
	v_max_f32_e32 v115, v34, v34
	ds_read_b64_tr_b16 v[174:175], v144 offset:640
	ds_read_b64_tr_b16 v[178:179], v144 offset:704
	ds_read_b64_tr_b16 v[116:117], v144 offset:448
	v_max_f32_e32 v114, v115, v114
	v_max3_f32 v114, v114, v36, v37
	v_max3_f32 v114, v114, v38, v39
	v_max3_f32 v114, v114, v40, v41
	v_max3_f32 v114, v114, v42, v43
	v_max3_f32 v114, v114, v44, v45
	v_mfma_f32_32x32x16_f16 v[18:33], v[132:135], v[106:109], v[18:33]
	v_max3_f32 v114, v114, v46, v47
	v_max3_f32 v114, v114, v48, v49
	v_max3_f32 v114, v114, v82, v83
	v_max3_f32 v114, v114, v84, v85
	v_max3_f32 v114, v114, v86, v87
	v_max3_f32 v114, v114, v88, v89
	v_max3_f32 v114, v114, v90, v91
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[116:119], v[106:109], v[2:17]
	v_max3_f32 v114, v114, v92, v93
	v_max3_f32 v114, v114, v94, v95
	v_max3_f32 v114, v114, v96, v97
	v_mov_b32_e32 v115, v114
	s_nop 1
	v_permlane32_swap_b32_e32 v114, v115
	v_max3_f32 v115, v147, v114, v115
	v_mfma_f32_32x32x16_f16 v[50:65], v[140:143], v[102:105], v[50:65]
	v_mov_b32_e32 v114, v97
	v_mul_f32_e64 v120, v114, s4
	v_mul_f32_e64 v121, v115, s4
	s_waitcnt vmcnt(0)
	v_fma_f32 v34, v34, s4, -v121
	v_fma_f32 v35, v35, s4, -v121
	v_fma_f32 v36, v36, s4, -v121
	v_fma_f32 v37, v37, s4, -v121
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[102:105], v[18:33]
	v_fma_f32 v38, v38, s4, -v121
	v_fma_f32 v39, v39, s4, -v121
	v_fma_f32 v40, v40, s4, -v121
	v_fma_f32 v41, v41, s4, -v121
	v_sub_f32_e32 v1, v1, v121
	v_exp_f32_e32 v114, v35
	v_exp_f32_e32 v140, v36
	v_mfma_f32_32x32x16_f16 v[2:17], v[178:181], v[102:105], v[2:17]
	v_exp_f32_e32 v141, v37
	v_exp_f32_e32 v142, v38
	v_exp_f32_e32 v143, v39
	v_exp_f32_e32 v144, v40
	v_exp_f32_e32 v147, v41
	v_sub_f32_e32 v97, v120, v121
	v_add_u32_e32 v120, s5, v233
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[98:101], v[66:81]
	v_fma_f32 v42, v42, s4, -v121
	v_fma_f32 v43, v43, s4, -v121
	v_fma_f32 v44, v44, s4, -v121
	v_fma_f32 v45, v45, s4, -v121
	v_fma_f32 v46, v46, s4, -v121
	v_fma_f32 v47, v47, s4, -v121
	v_fma_f32 v48, v48, s4, -v121
	v_mfma_f32_32x32x16_f16 v[50:65], v[170:173], v[98:101], v[50:65]
	v_fma_f32 v49, v49, s4, -v121
	v_fma_f32 v94, v94, s4, -v121
	v_fma_f32 v95, v95, s4, -v121
	v_fma_f32 v96, v96, s4, -v121
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v116, s5, v232
	v_mfma_f32_32x32x16_f16 v[18:33], v[128:131], v[98:101], v[18:33]
	v_exp_f32_e32 v165, v42
	v_exp_f32_e32 v166, v43
	v_exp_f32_e32 v167, v44
	v_exp_f32_e32 v168, v45
	v_exp_f32_e32 v169, v46
	v_exp_f32_e32 v170, v47
	v_exp_f32_e32 v171, v48
	v_mfma_f32_32x32x16_f16 v[2:17], v[182:185], v[98:101], v[2:17]
	v_exp_f32_e32 v99, v34
	v_exp_f32_e32 v98, v1
	v_add_u32_e32 v1, s5, v231
	ds_read_b64_tr_b16 v[100:101], v120 offset:256
	ds_read_b64_tr_b16 v[102:103], v116 offset:8448
	ds_read_b64_tr_b16 v[104:105], v1
	ds_read_b64_tr_b16 v[106:107], v116 offset:8192
	ds_read_b64_tr_b16 v[108:109], v116 offset:8256
	ds_read_b64_tr_b16 v[110:111], v1 offset:512
	ds_read_b64_tr_b16 v[112:113], v116 offset:8704
	v_exp_f32_e32 v172, v49
	v_exp_f32_e32 v185, v94
	v_exp_f32_e32 v186, v95
	v_exp_f32_e32 v187, v96
	v_exp_f32_e32 v188, v97
	v_cvt_pk_f16_f32 v94, v99, v114
	v_cvt_pk_f16_f32 v95, v140, v141
	v_cvt_pk_f16_f32 v96, v142, v143
	v_cvt_pk_f16_f32 v97, v144, v147
	v_pk_mul_f32 v[48:49], v[80:81], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[78:79], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[76:77], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[74:75], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[72:73], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[70:71], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[68:69], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[66:67], v[98:99] op_sel_hi:[1,0]
	v_fma_f32 v90, v90, s4, -v121
	v_fma_f32 v91, v91, s4, -v121
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[34:49], v[104:107], v[94:97], v[34:49]
	v_fma_f32 v92, v92, s4, -v121
	v_fma_f32 v93, v93, s4, -v121
	v_exp_f32_e32 v181, v90
	v_exp_f32_e32 v182, v91
	v_exp_f32_e32 v183, v92
	v_exp_f32_e32 v184, v93
	v_cvt_pk_f16_f32 v90, v165, v166
	v_cvt_pk_f16_f32 v91, v167, v168
	v_cvt_pk_f16_f32 v92, v169, v170
	v_cvt_pk_f16_f32 v93, v171, v172
	v_fma_f32 v82, v82, s4, -v121
	v_fma_f32 v83, v83, s4, -v121
	v_mfma_f32_32x32x16_f16 v[34:49], v[100:103], v[90:93], v[34:49]
	v_fma_f32 v84, v84, s4, -v121
	v_fma_f32 v85, v85, s4, -v121
	v_fma_f32 v86, v86, s4, -v121
	v_fma_f32 v87, v87, s4, -v121
	v_fma_f32 v88, v88, s4, -v121
	v_fma_f32 v89, v89, s4, -v121
	v_exp_f32_e32 v173, v82
	v_exp_f32_e32 v174, v83
	v_exp_f32_e32 v175, v84
	v_exp_f32_e32 v176, v85
	v_exp_f32_e32 v177, v86
	v_exp_f32_e32 v178, v87
	v_exp_f32_e32 v179, v88
	v_exp_f32_e32 v180, v89
	v_cvt_pk_f16_f32 v86, v173, v174
	v_cvt_pk_f16_f32 v87, v175, v176
	v_cvt_pk_f16_f32 v88, v177, v178
	v_cvt_pk_f16_f32 v89, v179, v180
	ds_read_b64_tr_b16 v[66:67], v120 offset:768
	ds_read_b64_tr_b16 v[70:71], v120 offset:320
	ds_read_b64_tr_b16 v[74:75], v120 offset:384
	ds_read_b64_tr_b16 v[78:79], v120 offset:192
	ds_read_b64_tr_b16 v[72:73], v116 offset:8512
	ds_read_b64_tr_b16 v[104:105], v116 offset:8320
	ds_read_b64_tr_b16 v[80:81], v116 offset:8384
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[34:49], v[110:113], v[86:89], v[34:49]
	ds_read_b64_tr_b16 v[118:119], v116 offset:8768
	ds_read_b64_tr_b16 v[76:77], v116 offset:8576
	ds_read_b64_tr_b16 v[122:123], v116 offset:8640
	ds_read_b64_tr_b16 v[68:69], v116 offset:8960
	ds_read_b64_tr_b16 v[126:127], v116 offset:9024
	ds_read_b64_tr_b16 v[130:131], v116 offset:8832
	ds_read_b64_tr_b16 v[134:135], v116 offset:8896
	v_cvt_pk_f16_f32 v82, v181, v182
	v_cvt_pk_f16_f32 v83, v183, v184
	v_cvt_pk_f16_f32 v84, v185, v186
	v_cvt_pk_f16_f32 v85, v187, v188
	ds_read_b64_tr_b16 v[124:125], v120 offset:832
	ds_read_b64_tr_b16 v[110:111], v120 offset:896
	ds_read_b64_tr_b16 v[136:137], v120 offset:960
	ds_read_b64_tr_b16 v[106:107], v120 offset:64
	ds_read_b64_tr_b16 v[102:103], v120 offset:128
	ds_read_b64_tr_b16 v[112:113], v116 offset:9088
	ds_read_b64_tr_b16 v[138:139], v116 offset:9152
	v_add_f32_e32 v1, v146, v145
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[34:49], v[66:69], v[82:85], v[34:49]
	v_mul_f32_e32 v67, v230, v0
	v_add_f32_e32 v0, v99, v114
	v_add_f32_e32 v0, v140, v0
	v_add_f32_e32 v0, v141, v0
	v_add_f32_e32 v0, v142, v0
	v_add_f32_e32 v0, v143, v0
	v_add_f32_e32 v0, v144, v0
	v_add_f32_e32 v0, v147, v0
	v_add_f32_e32 v0, v165, v0
	v_add_f32_e32 v0, v166, v0
	v_add_f32_e32 v0, v167, v0
	v_add_f32_e32 v0, v168, v0
	v_pk_mul_f32 v[64:65], v[64:65], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[98:99] op_sel_hi:[1,0]
	v_add_f32_e32 v1, v149, v1
	v_pk_mul_f32 v[32:33], v[32:33], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[98:99] op_sel_hi:[1,0]
	v_add_f32_e32 v0, v169, v0
	v_pk_mul_f32 v[16:17], v[16:17], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[98:99] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[50:65], v[106:109], v[94:97], v[50:65]
	v_add_f32_e32 v1, v150, v1
	v_add_f32_e32 v0, v170, v0
	v_add_f32_e32 v1, v151, v1
	v_add_f32_e32 v0, v171, v0
	v_add_f32_e32 v1, v152, v1
	v_add_f32_e32 v0, v172, v0
	v_add_f32_e32 v1, v153, v1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[102:105], v[94:97], v[18:33]
	ds_read_b64_tr_b16 v[116:117], v120 offset:576
	ds_read_b64_tr_b16 v[128:129], v120 offset:640
	ds_read_b64_tr_b16 v[132:133], v120 offset:704
	ds_read_b64_tr_b16 v[120:121], v120 offset:448
	v_add_f32_e32 v0, v173, v0
	v_add_f32_e32 v1, v148, v1
	v_add_f32_e32 v0, v174, v0
	v_add_f32_e32 v1, v162, v1
	v_add_f32_e32 v0, v175, v0
	v_mfma_f32_32x32x16_f16 v[2:17], v[78:81], v[94:97], v[2:17]
	v_add_f32_e32 v1, v163, v1
	v_add_f32_e32 v0, v176, v0
	v_add_f32_e32 v1, v164, v1
	v_add_f32_e32 v0, v177, v0
	v_add_f32_e32 v1, v155, v1
	v_add_f32_e32 v0, v178, v0
	v_add_f32_e32 v1, v156, v1
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[90:93], v[50:65]
	v_add_f32_e32 v0, v179, v0
	v_add_f32_e32 v1, v157, v1
	v_add_f32_e32 v0, v180, v0
	v_add_f32_e32 v1, v158, v1
	v_add_f32_e32 v0, v181, v0
	v_add_f32_e32 v1, v159, v1
	v_add_f32_e32 v0, v182, v0
	v_mfma_f32_32x32x16_f16 v[18:33], v[74:77], v[90:93], v[18:33]
	v_add_f32_e32 v1, v160, v1
	v_add_f32_e32 v0, v183, v0
	v_add_f32_e32 v1, v161, v1
	v_add_f32_e32 v0, v184, v0
	v_add_f32_e32 v1, v154, v1
	v_add_f32_e32 v0, v185, v0
	v_mov_b32_e32 v66, v1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[120:123], v[90:93], v[2:17]
	v_add_f32_e32 v0, v186, v0
	v_permlane32_swap_b32_e32 v1, v66
	v_add_f32_e32 v0, v187, v0
	v_add_f32_e32 v1, v1, v66
	v_add_f32_e32 v66, v188, v0
	v_mov_b32_e32 v0, v66
	v_mfma_f32_32x32x16_f16 v[50:65], v[116:119], v[86:89], v[50:65]
	s_nop 0
	v_permlane32_swap_b32_e32 v66, v0
	v_add_f32_e64 v0, v66, v0
	v_add_f32_e64 v1, v67, v1
	s_mov_b32 s4, 0x800000
	v_fmac_f32_e32 v0, v1, v98
	v_cmp_gt_f32_e32 vcc, s4, v0
	v_mov_b32_e32 v66, 0x42000000
	v_mfma_f32_32x32x16_f16 v[18:33], v[128:131], v[86:89], v[18:33]
	v_cndmask_b32_e64 v1, 0, 32, vcc
	v_ldexp_f32 v1, v0, v1
	v_log_f32_e32 v1, v1
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_barrier
	v_sub_f32_e32 v1, v1, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[132:135], v[86:89], v[2:17]
	v_add_f32_e32 v1, v115, v1
	v_add3_u32 v66, 0, v228, v229
	ds_write_b32 v66, v1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[124:127], v[82:85], v[50:65]
	v_mfma_f32_32x32x16_f16 v[18:33], v[110:113], v[82:85], v[18:33]
	v_mfma_f32_32x32x16_f16 v[2:17], v[136:139], v[82:85], v[2:17]
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	s_sub_i32 s6, 0x4000, s16
	v_or_b32_e32 v1, s16, v225
	s_movk_i32 s4, 0x4000
	v_cmp_gt_i32_e32 vcc, s6, v226
	v_cmp_gt_i32_e64 s[4:5], s4, v1
	v_bfrev_b32_e32 v1, 1
	s_and_b64 vcc, s[0:1], vcc
	v_cndmask_b32_e32 v1, v1, v227, vcc
	s_barrier
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr4_sgpr5
                                        ; implicit-def: $vgpr1
.LBB0_9:
	v_bfrev_b32_e32 v1, 1
	v_cndmask_b32_e64 v1, v1, v227, s[0:1]
	s_or_b64 s[4:5], s[4:5], exec
	s_barrier
.LBB0_10:
	v_div_scale_f32 v66, s[0:1], v0, v0, 1.0
	v_rcp_f32_e32 v67, v66
	v_div_scale_f32 v68, vcc, 1.0, v0, 1.0
	v_fma_f32 v69, -v66, v67, 1.0
	v_fmac_f32_e32 v67, v69, v67
	v_mul_f32_e32 v69, v68, v67
	v_fma_f32 v70, -v66, v69, v68
	v_fmac_f32_e32 v69, v70, v67
	v_fma_f32 v66, -v66, v69, v68
	v_div_fmas_f32 v66, v66, v67, v69
	v_lshl_add_u32 v67, v226, 2, 0
	ds_read_b32 v67, v67
	v_div_fixup_f32 v66, v66, v0, 1.0
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v1, s[8:11], 0 offen
	v_pk_mul_f32 v[0:1], v[66:67], v[16:17] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v17, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[14:15] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v16, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[12:13] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v15, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[10:11] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v14, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[8:9] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v9, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[6:7] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v8, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[4:5] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v7, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v6, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[32:33] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v3, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[30:31] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v2, v0, v1
	v_pk_mul_f32 v[0:1], v[66:67], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[66:67], v[26:27] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v1, v0, v1
	v_cvt_pk_f16_f32 v0, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[24:25] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v13, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[22:23] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v12, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[20:21] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v11, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[18:19] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v10, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[64:65] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v21, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[62:63] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v20, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[60:61] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v19, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[58:59] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v18, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[56:57] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v25, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[54:55] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v24, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[52:53] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v23, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[50:51] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v22, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[48:49] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v29, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[46:47] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v28, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[44:45] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v27, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[42:43] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v26, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[40:41] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v33, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[38:39] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v32, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[36:37] op_sel_hi:[0,1]
	s_mul_i32 s0, s3, s19
	s_mul_i32 s1, s2, s18
	v_cvt_pk_f16_f32 v31, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[34:35] op_sel_hi:[0,1]
	s_add_i32 s0, s0, s1
	s_mul_i32 s1, s12, s16
	v_cvt_pk_f16_f32 v30, v4, v5
	s_add_i32 s0, s0, s1
	v_mul_lo_u32 v4, s12, v225
	v_add_u32_e32 v4, s0, v4
	v_lshrrev_b32_e32 v5, 2, v215
	v_add_lshl_u32 v4, v4, v5, 1
	v_bfrev_b32_e32 v34, 1
	v_add_u32_e32 v5, 0xe0, v4
	v_add_u32_e32 v35, 0xc0, v4
	v_add_u32_e32 v36, 0xa0, v4
	v_add_u32_e32 v37, 0x80, v4
	v_add_u32_e32 v38, 0x60, v4
	v_add_u32_e32 v39, 64, v4
	v_add_u32_e32 v40, 32, v4
	v_cndmask_b32_e64 v4, v34, v4, s[4:5]
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s22, s10
	s_mov_b32 s23, s11
	v_permlane32_swap_b32_e32 v30, v32
	v_permlane32_swap_b32_e32 v31, v33
	v_cndmask_b32_e64 v5, v34, v5, s[4:5]
	v_cndmask_b32_e64 v35, v34, v35, s[4:5]
	v_cndmask_b32_e64 v36, v34, v36, s[4:5]
	v_cndmask_b32_e64 v37, v34, v37, s[4:5]
	v_cndmask_b32_e64 v38, v34, v38, s[4:5]
	v_cndmask_b32_e64 v39, v34, v39, s[4:5]
	v_cndmask_b32_e64 v40, v34, v40, s[4:5]
	v_permlane32_swap_b32_e32 v26, v28
	v_permlane32_swap_b32_e32 v27, v29
	v_permlane32_swap_b32_e32 v22, v24
	v_permlane32_swap_b32_e32 v23, v25
	v_permlane32_swap_b32_e32 v18, v20
	v_permlane32_swap_b32_e32 v19, v21
	v_permlane32_swap_b32_e32 v10, v12
	v_permlane32_swap_b32_e32 v11, v13
	v_permlane32_swap_b32_e32 v0, v2
	v_permlane32_swap_b32_e32 v1, v3
	v_permlane32_swap_b32_e32 v6, v8
	v_permlane32_swap_b32_e32 v7, v9
	v_permlane32_swap_b32_e32 v14, v16
	v_permlane32_swap_b32_e32 v15, v17
	buffer_store_dwordx4 v[30:33], v4, s[20:23], 0 offen
	buffer_store_dwordx4 v[26:29], v40, s[20:23], 0 offen
	buffer_store_dwordx4 v[22:25], v39, s[20:23], 0 offen
	buffer_store_dwordx4 v[18:21], v38, s[20:23], 0 offen
	buffer_store_dwordx4 v[10:13], v37, s[20:23], 0 offen
	buffer_store_dwordx4 v[0:3], v36, s[20:23], 0 offen
	buffer_store_dwordx4 v[6:9], v35, s[20:23], 0 offen
	buffer_store_dwordx4 v[14:17], v5, s[20:23], 0 offen
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
		.amdhsa_next_free_vgpr 255
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
	.set attn_fwd.num_vgpr, 255
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 40
	.set attn_fwd.num_named_barrier, 0
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11316
; TotalNumSgprs: 46
; NumVgprs: 255
; NumAgprs: 0
; TotalNumVgprs: 255
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 255
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
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
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
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
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
	.byte	1                               ; Abbrev [1] 0xb:0xaa DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x84 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	519                             ; DW_AT_call_line
	.byte	41                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x56:0x5d DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	708                             ; DW_AT_call_line
	.byte	61                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x63:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	286                             ; DW_AT_call_line
	.byte	69                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x70:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	69                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7d:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	334                             ; DW_AT_call_line
	.byte	42                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x8a:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	189                             ; DW_AT_call_line
	.byte	40                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x97:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	340                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa4:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
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
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp227-.Lfunc_begin0
	.quad	.Ltmp228-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
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
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
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
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
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
	.quad	.Ltmp63-.Lfunc_begin0
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
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp216-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp221-.Lfunc_begin0
	.quad	.Ltmp222-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
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
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp218-.Lfunc_begin0
	.quad	.Ltmp219-.Lfunc_begin0
	.quad	.Ltmp220-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"flash-attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/app/OAI-triton/fav3_kernel"   ; string offset=26
.Linfo_string3:
	.asciz	"attn_fwd"                      ; string offset=54
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
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     255
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
