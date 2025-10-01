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
	.file	1 "/app/OAI-triton/fa" "flash-attention.py"
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
	v_and_b32_e32 v37, 15, v0
	v_lshrrev_b32_e32 v34, 4, v0
	v_lshlrev_b32_e32 v198, 3, v37
	s_add_i32 s2, s2, s3
	s_mov_b64 s[24:25], s[6:7]
	v_or_b32_e32 v35, 32, v34
	v_or_b32_e32 v4, s16, v34
	v_mul_lo_u32 v12, s14, v34
	s_lshl_b32 s6, s14, 6
	v_add_u32_e32 v17, s2, v198
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
	v_cndmask_b32_e32 v38, v18, v4, vcc
	v_add_lshl_u32 v4, v17, v14, 1
	v_cmp_gt_i32_e32 vcc, s2, v6
	v_or_b32_e32 v2, 0xa0, v34
	v_add_u32_e32 v15, s6, v14
	v_cndmask_b32_e32 v39, v18, v4, vcc
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
	v_and_b32_e32 v209, 48, v0
	v_and_b32_e32 v199, 32, v0
	v_cndmask_b32_e32 v43, v18, v2, vcc
	buffer_load_dwordx4 v[2:5], v36, s[36:39], 0 offen
	buffer_load_dwordx4 v[6:9], v38, s[36:39], 0 offen
	buffer_load_dwordx4 v[10:13], v39, s[36:39], 0 offen
	buffer_load_dwordx4 v[14:17], v1, s[36:39], 0 offen
	buffer_load_dwordx4 v[18:21], v40, s[36:39], 0 offen
	buffer_load_dwordx4 v[22:25], v41, s[36:39], 0 offen
	buffer_load_dwordx4 v[26:29], v42, s[36:39], 0 offen
	buffer_load_dwordx4 v[30:33], v43, s[36:39], 0 offen
	s_load_dwordx4 s[28:31], s[0:1], 0x38
	s_load_dword s33, s[0:1], 0x48
	v_lshrrev_b32_e32 v36, 6, v0
	v_and_b32_e32 v1, 0x1c0, v0
	v_or_b32_e32 v42, v36, v209
	v_mov_b32_e32 v54, 0x2000
	v_and_b32_e32 v208, 31, v0
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[38:39], s[2:3], s29, v42, v[198:199]
	v_lshlrev_b32_e32 v41, 7, v1
	v_lshlrev_b32_e32 v200, 4, v37
	v_lshrrev_b32_e32 v43, 1, v199
	v_lshl_or_b32 v54, v36, 10, v54
	s_mov_b64 s[20:21], s[10:11]
	s_movk_i32 s6, 0x60
	s_movk_i32 s7, 0xa0
	s_movk_i32 s10, 0xe0
	s_movk_i32 s11, 0x80
	s_movk_i32 s12, 0xc0
	v_lshl_or_b32 v41, v208, 8, v41
	v_xor_b32_e32 v44, v200, v43
	s_movk_i32 s2, 0x410
	v_lshrrev_b32_e32 v55, 6, v54
	v_or_b32_e32 v45, v41, v44
	v_bitop3_b32 v48, v41, s6, v44 bitop3:0x36
	v_bitop3_b32 v49, v41, s11, v44 bitop3:0x36
	v_bitop3_b32 v50, v41, s7, v44 bitop3:0x36
	v_bitop3_b32 v51, v41, s12, v44 bitop3:0x36
	v_bitop3_b32 v41, v41, s10, v44 bitop3:0x36
	v_mad_u32_u24 v44, v36, s2, 0
	v_or_b32_e32 v210, v55, v54
	s_mul_i32 s34, s15, s18
	s_mul_i32 s28, s28, s19
	v_add_u32_e32 v52, 0x87c0, v44
	v_add_u32_e32 v55, 0, v210
	s_add_i32 s36, s28, s34
	s_lshl_b32 s37, s29, 6
	v_lshlrev_b32_e32 v201, 4, v0
	v_and_b32_e32 v40, 0xf0, v0
	s_and_b32 s5, s5, 0xffff
	v_add_u32_e32 v56, 0x87c0, v55
	v_lshlrev_b32_e32 v60, 10, v37
	v_and_b32_e32 v37, 16, v0
	v_readfirstlane_b32 s10, v52
	v_lshl_add_u32 v39, s29, 3, v38
	v_xad_u32 v40, v201, v40, 0
	v_add_u32_e32 v46, 0, v45
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s38
	s_mov_b32 s15, s39
	v_add_lshl_u32 v53, v38, s36, 1
	v_add_u32_e32 v58, s37, v38
	v_add_u32_e32 v38, 0xc8c0, v44
	v_lshlrev_b32_e32 v61, 4, v37
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s11, v56
	v_xad_u32 v47, v45, 32, 0
	v_xad_u32 v45, v45, 64, 0
	v_add_u32_e32 v48, 0, v48
	v_add_u32_e32 v49, 0, v49
	v_add_u32_e32 v50, 0, v50
	v_add_u32_e32 v51, 0, v51
	v_add_u32_e32 v41, 0, v41
	v_add_lshl_u32 v57, v39, s36, 1
	v_add_u32_e32 v44, 0xc8c0, v55
	v_or3_b32 v43, v60, v61, v43
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v40, v[2:5]
	s_waitcnt vmcnt(6)
	ds_write_b128 v40, v[6:9] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v40, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v40, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v40, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v40, v[22:25] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v40, v[26:29] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v40, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[114:117], v46
	ds_read_b128 v[154:157], v47
	ds_read_b128 v[150:153], v45
	ds_read_b128 v[146:149], v48
	ds_read_b128 v[142:145], v49
	ds_read_b128 v[138:141], v50
	ds_read_b128 v[134:137], v51
	ds_read_b128 v[130:133], v41
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v53, s[12:15], 0 offen lds
	s_mov_b32 m0, s11
	v_readfirstlane_b32 s2, v38
	v_add_u32_e32 v59, s37, v39
	v_add_lshl_u32 v39, v58, s36, 1
	v_add_u32_e32 v204, v43, v200
	buffer_load_dwordx4 v57, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v44
	v_add_lshl_u32 v55, v59, s36, 1
	v_add_u32_e32 v60, 0, v204
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v39, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
	s_add_i32 s23, 0, 0x87c0
	buffer_load_dwordx4 v55, s[12:15], 0 offen lds
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[2:5], v60 offset:34752
	v_add_u32_e32 v48, s23, v204
	ds_read_b128 v[18:21], v48 offset:32
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[114:117], 0
	v_mad_u64_u32 v[46:47], s[2:3], s33, v42, v[198:199]
	v_lshrrev_b32_e32 v49, 4, v54
	s_movk_i32 s2, 0x440
	s_mul_i32 s35, s30, s18
	s_mul_i32 s31, s31, s19
	v_mad_u32_u24 v50, v36, s2, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[154:157], v[2:17]
	ds_read_b128 v[18:21], v48 offset:64
	ds_read_b128 v[22:25], v48 offset:96
	v_or_b32_e32 v202, v49, v54
	s_add_i32 s30, s31, s35
	v_add_u32_e32 v49, 0, v202
	v_readfirstlane_b32 s3, v50
	v_lshl_add_u32 v47, s33, 3, v46
	s_and_b32 s25, s25, 0xffff
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[150:153], v[2:17]
	s_mov_b32 s26, s38
	s_mov_b32 s27, s39
	v_add_lshl_u32 v51, v46, s30, 1
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v49
	v_add_lshl_u32 v52, v47, s30, 1
	s_add_i32 s36, s36, s37
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[146:149], v[2:17]
	ds_read_b128 v[18:21], v48 offset:128
	ds_read_b128 v[22:25], v48 offset:160
	s_lshl_b32 s17, s33, 6
	v_add_lshl_u32 v53, v58, s36, 1
	s_add_i32 s2, 0, 0xc8c0
	v_add_u32_e32 v56, 0x4400, v50
	v_add_lshl_u32 v54, v59, s36, 1
	v_add_u32_e32 v55, s2, v204
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[142:145], v[2:17]
	s_add_i32 s2, s30, s17
	v_add_u32_e32 v57, 0x4400, v49
	v_add_lshl_u32 v46, s2, v46, 1
	v_add_lshl_u32 v47, s2, v47, 1
	s_movk_i32 s2, 0x100
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_movk_i32 s2, 0xff
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[138:141], v[2:17]
	ds_read_b128 v[18:21], v48 offset:192
	ds_read_b128 v[22:25], v48 offset:224
	ds_read_b128 v[38:41], v48 offset:544
	ds_read_b128 v[42:45], v48 offset:576
	buffer_load_dwordx4 v51, s[24:27], 0 offen lds
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v56
	buffer_load_dwordx4 v52, s[24:27], 0 offen lds
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[134:137], v[2:17]
	ds_read_b128 v[18:21], v48 offset:512
	s_mov_b32 m0, s10
	s_mov_b32 s22, 0
	buffer_load_dwordx4 v53, s[12:15], 0 offen lds
	s_mov_b32 m0, s11
	s_mov_b32 s6, s38
	buffer_load_dwordx4 v54, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[130:133], v[2:17]
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v57
	s_mov_b32 s7, s39
	s_mov_b32 s13, 0x3e0293ee
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[114:117], 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[38:41], v[154:157], v[18:33]
	ds_read_b128 v[38:41], v48 offset:608
	v_mfma_f32_32x32x16_f16 v[18:33], v[42:45], v[150:153], v[18:33]
	ds_read_b128 v[42:45], v48 offset:640
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[38:41], v[146:149], v[18:33]
	ds_read_b128 v[38:41], v48 offset:672
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[42:45], v[142:145], v[18:33]
	ds_read_b128 v[42:45], v48 offset:704
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[38:41], v[138:141], v[18:33]
	ds_read_b128 v[38:41], v48 offset:736
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v46, s[24:27], 0 offen lds
	s_mov_b32 m0, s3
	v_cmp_lt_u32_e64 s[2:3], s2, v0
	buffer_load_dwordx4 v47, s[24:27], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[18:33], v[42:45], v[134:137], v[18:33]
	ds_read_b128 v[66:69], v60 offset:51392
	ds_read_b128 v[186:189], v55 offset:32
	ds_read_b128 v[182:185], v55 offset:64
	ds_read_b128 v[178:181], v55 offset:96
	ds_read_b128 v[174:177], v55 offset:128
	ds_read_b128 v[110:113], v55 offset:160
	ds_read_b128 v[106:109], v55 offset:192
	ds_read_b128 v[102:105], v55 offset:224
	ds_read_b128 v[98:101], v55 offset:512
	ds_read_b128 v[170:173], v55 offset:544
	ds_read_b128 v[166:169], v55 offset:576
	ds_read_b128 v[162:165], v55 offset:608
	ds_read_b128 v[158:161], v55 offset:640
	ds_read_b128 v[126:129], v55 offset:672
	ds_read_b128 v[122:125], v55 offset:704
	ds_read_b128 v[118:121], v55 offset:736
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[18:33], v[38:41], v[130:133], v[18:33]
	.file	2 "/app/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v38, v3, v3
	v_max_f32_e32 v39, v2, v2
	v_max_f32_e32 v38, v39, v38
	v_max3_f32 v38, v38, v4, v5
	v_max3_f32 v38, v38, v6, v7
	v_max3_f32 v38, v38, v8, v9
	v_max3_f32 v38, v38, v10, v11
	v_max3_f32 v38, v38, v12, v13
	v_max3_f32 v38, v38, v14, v15
	v_max3_f32 v38, v38, v16, v17
	s_nop 1
	v_max3_f32 v38, v38, v18, v19
	v_max3_f32 v38, v38, v20, v21
	v_max3_f32 v38, v38, v22, v23
	v_max3_f32 v38, v38, v24, v25
	v_max3_f32 v38, v38, v26, v27
	v_max3_f32 v38, v38, v28, v29
	v_max3_f32 v38, v38, v30, v31
	v_max3_f32 v39, v38, v32, v33
	v_mov_b32_e32 v40, v39
	s_nop 1
	v_permlane32_swap_b32_e32 v39, v40
	v_mov_b32_e32 v38, 0xff800000
	v_max3_f32 v214, v39, v40, v38
	v_mul_f32_e32 v39, 0xbe0293ee, v214
	v_fmamk_f32 v2, v2, 0x3e0293ee, v39
	v_fmamk_f32 v3, v3, 0x3e0293ee, v39
	v_fmamk_f32 v4, v4, 0x3e0293ee, v39
	v_fmamk_f32 v5, v5, 0x3e0293ee, v39
	v_fmamk_f32 v6, v6, 0x3e0293ee, v39
	v_fmamk_f32 v7, v7, 0x3e0293ee, v39
	v_fmamk_f32 v8, v8, 0x3e0293ee, v39
	v_fmamk_f32 v9, v9, 0x3e0293ee, v39
	v_fmamk_f32 v10, v10, 0x3e0293ee, v39
	v_fmamk_f32 v11, v11, 0x3e0293ee, v39
	v_fmamk_f32 v12, v12, 0x3e0293ee, v39
	v_fmamk_f32 v13, v13, 0x3e0293ee, v39
	v_fmamk_f32 v14, v14, 0x3e0293ee, v39
	v_fmamk_f32 v15, v15, 0x3e0293ee, v39
	v_fmamk_f32 v16, v16, 0x3e0293ee, v39
	v_fmamk_f32 v17, v17, 0x3e0293ee, v39
	v_fmamk_f32 v18, v18, 0x3e0293ee, v39
	v_fmamk_f32 v19, v19, 0x3e0293ee, v39
	v_fmamk_f32 v20, v20, 0x3e0293ee, v39
	v_fmamk_f32 v21, v21, 0x3e0293ee, v39
	v_fmamk_f32 v22, v22, 0x3e0293ee, v39
	v_fmamk_f32 v23, v23, 0x3e0293ee, v39
	v_fmamk_f32 v24, v24, 0x3e0293ee, v39
	v_fmamk_f32 v25, v25, 0x3e0293ee, v39
	v_fmamk_f32 v26, v26, 0x3e0293ee, v39
	v_fmamk_f32 v27, v27, 0x3e0293ee, v39
	v_fmamk_f32 v28, v28, 0x3e0293ee, v39
	v_fmamk_f32 v29, v29, 0x3e0293ee, v39
	v_fmamk_f32 v30, v30, 0x3e0293ee, v39
	v_fmamk_f32 v31, v31, 0x3e0293ee, v39
	v_fmamk_f32 v32, v32, 0x3e0293ee, v39
	v_fmac_f32_e32 v39, 0x3e0293ee, v33
	v_fmac_f32_e32 v38, 0xbe0293ee, v214
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:                                ; %.preheader
	s_or_b64 exec, exec, s[10:11]
	v_exp_f32_e32 v222, v3
	v_lshlrev_b32_e32 v3, 3, v0
	s_load_dwordx2 s[2:3], s[0:1], 0x4c
	s_load_dword s12, s[0:1], 0x54
	v_exp_f32_e32 v220, v2
	v_lshlrev_b32_e32 v2, 8, v0
	v_and_b32_e32 v3, 24, v3
	s_movk_i32 s0, 0xc00
	v_exp_f32_e32 v221, v4
	v_exp_f32_e32 v224, v5
	v_lshlrev_b32_e32 v4, 1, v37
	v_lshlrev_b32_e32 v5, 7, v199
	v_and_or_b32 v2, v2, s0, v3
	v_or3_b32 v211, v4, v5, v2
	v_lshrrev_b32_e32 v2, 4, v211
	v_and_b32_e32 v212, 0x1c0, v2
	v_or_b32_e32 v2, 0x2000, v211
	v_lshrrev_b32_e32 v2, 4, v2
	v_and_b32_e32 v213, 0x3c0, v2
	v_add_u32_e32 v2, v209, v36
	s_lshl_b32 s0, s35, 1
	v_add_u32_e32 v3, 0x88, v2
	s_lshl_b32 s1, s31, 1
	v_mul_lo_u32 v3, s33, v3
	s_add_i32 s1, s1, s0
	v_lshl_add_u32 v216, v3, 1, s1
	v_or_b32_e32 v3, 0x80, v2
	v_mul_lo_u32 v3, s33, v3
	v_lshl_add_u32 v217, v3, 1, s1
	s_lshl_b32 s1, s34, 1
	v_add_u32_e32 v3, 0xc8, v2
	s_lshl_b32 s10, s28, 1
	v_add_u32_e32 v2, 0xc0, v2
	s_add_i32 s10, s10, s1
	v_mul_lo_u32 v2, s29, v2
	v_lshl_add_u32 v219, v2, 1, s10
	v_or_b32_e32 v2, 64, v35
	v_exp_f32_e32 v223, v6
	v_exp_f32_e32 v226, v7
	v_exp_f32_e32 v225, v8
	v_exp_f32_e32 v228, v9
	v_exp_f32_e32 v227, v10
	v_exp_f32_e32 v230, v11
	v_exp_f32_e32 v229, v12
	v_exp_f32_e32 v231, v13
	v_exp_f32_e32 v232, v14
	v_exp_f32_e32 v235, v15
	v_exp_f32_e32 v234, v16
	v_exp_f32_e32 v237, v17
	v_exp_f32_e32 v236, v18
	v_exp_f32_e32 v239, v19
	v_exp_f32_e32 v238, v20
	v_exp_f32_e32 v241, v21
	v_exp_f32_e32 v240, v22
	v_exp_f32_e32 v243, v23
	v_exp_f32_e32 v242, v24
	v_exp_f32_e32 v245, v25
	v_exp_f32_e32 v244, v26
	v_exp_f32_e32 v246, v27
	v_exp_f32_e32 v233, v28
	v_exp_f32_e32 v192, v29
	v_exp_f32_e32 v191, v30
	v_exp_f32_e32 v194, v31
	v_exp_f32_e32 v193, v32
	v_exp_f32_e32 v195, v39
	v_exp_f32_e32 v190, v38
	v_mul_lo_u32 v2, s33, v2
	v_add_u32_e32 v206, s30, v2
	v_or_b32_e32 v2, 64, v34
	v_mul_lo_u32 v3, s29, v3
	v_mul_lo_u32 v2, s33, v2
	v_mov_b32_e32 v34, 0
	v_mul_u32_u24_e32 v215, 0x410, v36
	v_mul_u32_u24_e32 v205, 0x440, v36
	s_lshl_b32 s0, s33, 7
	v_lshl_add_u32 v218, v3, 1, s10
	s_lshl_b32 s1, s29, 7
	s_add_i32 s10, 0, 0x4400
	v_add_u32_e32 v207, s30, v2
	v_mov_b32_e32 v203, 1.0
	s_movk_i32 s11, 0xffc0
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	s_mov_b32 s15, 0
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
	v_mov_b32_e32 v2, v34
	v_mov_b32_e32 v3, v34
	v_mov_b32_e32 v4, v34
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
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[114:117], 0
	s_mov_b32 s33, s22
	s_mov_b32 s22, s10
	v_mov_b32_e32 v196, v203
	v_mov_b32_e32 v248, v214
	s_setprio 0
	v_add_f32_e32 v82, v220, v222
	v_add_f32_e32 v82, v82, v221
	v_add_f32_e32 v82, v82, v224
	v_add_f32_e32 v82, v82, v223
	v_add_f32_e32 v82, v82, v226
	v_add_f32_e32 v82, v82, v225
	v_add_f32_e32 v82, v82, v228
	v_add_f32_e32 v82, v82, v227
	v_add_f32_e32 v82, v82, v230
	v_add_f32_e32 v82, v82, v229
	v_add_f32_e32 v82, v82, v231
	v_add_f32_e32 v82, v82, v232
	v_add_f32_e32 v82, v82, v235
	v_add_f32_e32 v82, v82, v234
	v_add_f32_e32 v82, v82, v237
	v_add_f32_e32 v82, v82, v236
	v_add_f32_e32 v82, v82, v239
	v_add_f32_e32 v82, v82, v238
	v_add_f32_e32 v82, v82, v241
	v_add_f32_e32 v82, v82, v240
	v_add_f32_e32 v82, v82, v243
	v_add_f32_e32 v82, v82, v242
	v_add_f32_e32 v82, v82, v245
	v_add_f32_e32 v82, v82, v244
	v_add_f32_e32 v82, v82, v246
	v_add_f32_e32 v82, v82, v233
	v_add_f32_e32 v82, v82, v192
	v_add_f32_e32 v82, v82, v191
	v_add_f32_e32 v82, v82, v194
	v_add_f32_e32 v82, v82, v193
	v_add_f32_e32 v82, v82, v195
	v_mov_b32_e32 v83, v82
	s_nop 1
	v_permlane32_swap_b32_e32 v82, v83
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[154:157], v[66:81]
	v_add_f32_e32 v203, v82, v83
	v_mul_f32_e32 v34, v34, v190
	v_mul_f32_e32 v35, v35, v190
	v_mul_f32_e32 v36, v36, v190
	v_mul_f32_e32 v37, v37, v190
	v_mul_f32_e32 v38, v38, v190
	v_mul_f32_e32 v39, v39, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[114:117], 0
	v_mul_f32_e32 v40, v40, v190
	v_mul_f32_e32 v41, v41, v190
	v_mul_f32_e32 v42, v42, v190
	v_mul_f32_e32 v43, v43, v190
	v_mul_f32_e32 v44, v44, v190
	v_mul_f32_e32 v45, v45, v190
	v_mul_f32_e32 v46, v46, v190
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[150:153], v[66:81]
	v_mul_f32_e32 v47, v47, v190
	v_mul_f32_e32 v48, v48, v190
	v_mul_f32_e32 v49, v49, v190
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
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[146:149], v[66:81]
	v_mul_f32_e32 v61, v61, v190
	v_mul_f32_e32 v62, v62, v190
	v_mul_f32_e32 v63, v63, v190
	v_mul_f32_e32 v64, v64, v190
	v_mul_f32_e32 v65, v65, v190
	v_mul_f32_e32 v18, v18, v190
	v_mul_f32_e32 v19, v19, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[150:153], v[82:97]
	v_mul_f32_e32 v20, v20, v190
	v_mul_f32_e32 v21, v21, v190
	v_mul_f32_e32 v22, v22, v190
	v_mul_f32_e32 v23, v23, v190
	v_mul_f32_e32 v24, v24, v190
	v_mul_f32_e32 v25, v25, v190
	v_mul_f32_e32 v26, v26, v190
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[142:145], v[66:81]
	v_mul_f32_e32 v27, v27, v190
	v_mul_f32_e32 v28, v28, v190
	v_mul_f32_e32 v29, v29, v190
	v_mul_f32_e32 v30, v30, v190
	v_mul_f32_e32 v31, v31, v190
	v_mul_f32_e32 v32, v32, v190
	v_mul_f32_e32 v33, v33, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[146:149], v[82:97]
	v_mul_f32_e32 v2, v2, v190
	v_mul_f32_e32 v3, v3, v190
	v_mul_f32_e32 v4, v4, v190
	v_mul_f32_e32 v5, v5, v190
	v_mul_f32_e32 v6, v6, v190
	v_mul_f32_e32 v7, v7, v190
	v_mul_f32_e32 v8, v8, v190
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
	v_fmac_f32_e32 v203, v196, v190
	v_cvt_pk_f16_f32 v110, v220, v222
	v_cvt_pk_f16_f32 v111, v221, v224
	v_cvt_pk_f16_f32 v112, v223, v226
	v_cvt_pk_f16_f32 v113, v225, v228
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[134:137], v[66:81]
	v_cvt_pk_f16_f32 v106, v227, v230
	v_cvt_pk_f16_f32 v107, v229, v231
	v_cvt_pk_f16_f32 v108, v232, v235
	v_cvt_pk_f16_f32 v109, v234, v237
	v_cvt_pk_f16_f32 v98, v244, v246
	v_cvt_pk_f16_f32 v99, v233, v192
	v_cvt_pk_f16_f32 v100, v191, v194
	v_mfma_f32_32x32x16_f16 v[82:97], v[126:129], v[138:141], v[82:97]
	v_cvt_pk_f16_f32 v101, v193, v195
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[130:133], v[66:81]
	v_cvt_pk_f16_f32 v102, v236, v239
	v_cvt_pk_f16_f32 v103, v238, v241
	v_cvt_pk_f16_f32 v104, v240, v243
	v_cvt_pk_f16_f32 v105, v242, v245
	v_mfma_f32_32x32x16_f16 v[82:97], v[122:125], v[134:137], v[82:97]
	v_mfma_f32_32x32x16_f16 v[82:97], v[118:121], v[130:133], v[82:97]
	s_setprio 1
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s10, s15, 1
	s_cmp_lt_i32 s10, 2
	s_cselect_b32 s30, s10, 0
	s_lshl_b32 s28, s30, 13
	s_lshl_b32 s10, s30, 14
	s_add_i32 s29, s10, 0
	s_ashr_i32 s10, s28, 5
	s_add_i32 s14, s29, s10
	s_add_i32 s31, s14, 0x87c0
	v_add_u32_e32 v118, s31, v215
	v_add_u32_e32 v119, v200, v219
	v_readfirstlane_b32 s10, v118
	v_add_u32_e32 v118, s31, v210
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v118
	buffer_load_dwordx4 v119, s[4:7], 0 offen lds
	v_add_u32_e32 v119, v200, v218
	s_mov_b32 m0, s10
	v_add_u32_e32 v118, s33, v211
	buffer_load_dwordx4 v119, s[4:7], 0 offen lds
	v_add_u32_e32 v119, v118, v212
	v_add_u32_e32 v120, v118, v213
	ds_read_b64_tr_b16 v[220:221], v119
	ds_read_b64_tr_b16 v[190:191], v119 offset:64
	ds_read_b64_tr_b16 v[174:175], v119 offset:128
	ds_read_b64_tr_b16 v[158:159], v119 offset:192
	ds_read_b64_tr_b16 v[222:223], v120 offset:8192
	ds_read_b64_tr_b16 v[192:193], v120 offset:8256
	ds_read_b64_tr_b16 v[176:177], v120 offset:8320
	ds_read_b64_tr_b16 v[160:161], v120 offset:8384
	ds_read_b64_tr_b16 v[224:225], v119 offset:256
	ds_read_b64_tr_b16 v[186:187], v119 offset:320
	ds_read_b64_tr_b16 v[170:171], v119 offset:384
	ds_read_b64_tr_b16 v[126:127], v119 offset:448
	ds_read_b64_tr_b16 v[226:227], v120 offset:8448
	ds_read_b64_tr_b16 v[188:189], v120 offset:8512
	ds_read_b64_tr_b16 v[172:173], v120 offset:8576
	ds_read_b64_tr_b16 v[128:129], v120 offset:8640
	ds_read_b64_tr_b16 v[228:229], v119 offset:512
	ds_read_b64_tr_b16 v[182:183], v119 offset:576
	ds_read_b64_tr_b16 v[166:167], v119 offset:640
	ds_read_b64_tr_b16 v[122:123], v119 offset:704
	ds_read_b64_tr_b16 v[230:231], v120 offset:8704
	ds_read_b64_tr_b16 v[184:185], v120 offset:8768
	ds_read_b64_tr_b16 v[168:169], v120 offset:8832
	ds_read_b64_tr_b16 v[124:125], v120 offset:8896
	ds_read_b64_tr_b16 v[194:195], v119 offset:768
	ds_read_b64_tr_b16 v[178:179], v119 offset:832
	ds_read_b64_tr_b16 v[162:163], v119 offset:896
	ds_read_b64_tr_b16 v[118:119], v119 offset:960
	ds_read_b64_tr_b16 v[196:197], v120 offset:8960
	ds_read_b64_tr_b16 v[180:181], v120 offset:9024
	ds_read_b64_tr_b16 v[164:165], v120 offset:9088
	ds_read_b64_tr_b16 v[120:121], v120 offset:9152
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[34:49], v[220:223], v[110:113], v[34:49]
	s_barrier
	s_setprio 0
	v_mfma_f32_32x32x16_f16 v[34:49], v[224:227], v[106:109], v[34:49]
	v_max_f32_e32 v214, v67, v67
	v_max_f32_e32 v220, v66, v66
	v_max_f32_e32 v214, v220, v214
	v_max3_f32 v214, v214, v68, v69
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[34:49], v[228:231], v[102:105], v[34:49]
	v_mfma_f32_32x32x16_f16 v[50:65], v[190:193], v[110:113], v[50:65]
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[110:113], v[18:33]
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[110:113], v[2:17]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[34:49], v[194:197], v[98:101], v[34:49]
	v_max3_f32 v194, v214, v70, v71
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
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[106:109], v[18:33]
	v_mov_b32_e32 v186, v190
	s_nop 1
	v_permlane32_swap_b32_e32 v190, v186
	v_max3_f32 v214, v248, v190, v186
	v_mul_f32_e32 v247, 0x3e0293ee, v214
	v_fma_f32 v66, v66, s13, -v247
	v_fma_f32 v67, v67, s13, -v247
	v_mfma_f32_32x32x16_f16 v[2:17], v[126:129], v[106:109], v[2:17]
	v_fma_f32 v68, v68, s13, -v247
	v_fma_f32 v69, v69, s13, -v247
	v_fma_f32 v70, v70, s13, -v247
	v_fma_f32 v71, v71, s13, -v247
	v_fma_f32 v72, v72, s13, -v247
	v_fma_f32 v73, v73, s13, -v247
	v_fma_f32 v74, v74, s13, -v247
	v_mfma_f32_32x32x16_f16 v[50:65], v[182:185], v[102:105], v[50:65]
	v_fma_f32 v75, v75, s13, -v247
	v_fma_f32 v76, v76, s13, -v247
	v_fma_f32 v77, v77, s13, -v247
	v_fma_f32 v78, v78, s13, -v247
	v_fma_f32 v79, v79, s13, -v247
	v_fma_f32 v80, v80, s13, -v247
	v_fma_f32 v81, v81, s13, -v247
	v_mfma_f32_32x32x16_f16 v[18:33], v[166:169], v[102:105], v[18:33]
	v_fma_f32 v82, v82, s13, -v247
	v_fma_f32 v83, v83, s13, -v247
	v_fma_f32 v84, v84, s13, -v247
	v_fma_f32 v85, v85, s13, -v247
	v_fma_f32 v86, v86, s13, -v247
	v_fma_f32 v87, v87, s13, -v247
	v_fma_f32 v88, v88, s13, -v247
	v_mfma_f32_32x32x16_f16 v[2:17], v[122:125], v[102:105], v[2:17]
	v_fma_f32 v89, v89, s13, -v247
	v_fma_f32 v90, v90, s13, -v247
	v_fma_f32 v91, v91, s13, -v247
	v_fma_f32 v92, v92, s13, -v247
	v_fma_f32 v93, v93, s13, -v247
	v_fma_f32 v94, v94, s13, -v247
	v_fma_f32 v95, v95, s13, -v247
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[50:65], v[178:181], v[98:101], v[50:65]
	v_fma_f32 v96, v96, s13, -v247
	v_fma_f32 v97, v97, s13, -v247
	v_exp_f32_e32 v220, v66
	v_fma_f32 v66, v248, s13, -v247
	v_exp_f32_e32 v222, v67
	v_exp_f32_e32 v221, v68
	v_exp_f32_e32 v224, v69
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[162:165], v[98:101], v[18:33]
	v_exp_f32_e32 v223, v70
	v_exp_f32_e32 v226, v71
	v_exp_f32_e32 v225, v72
	v_exp_f32_e32 v228, v73
	v_exp_f32_e32 v227, v74
	v_exp_f32_e32 v230, v75
	v_exp_f32_e32 v229, v76
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[118:121], v[98:101], v[2:17]
	v_exp_f32_e32 v231, v77
	v_exp_f32_e32 v232, v78
	v_exp_f32_e32 v235, v79
	v_exp_f32_e32 v234, v80
	v_exp_f32_e32 v237, v81
	v_exp_f32_e32 v236, v82
	v_exp_f32_e32 v239, v83
	v_exp_f32_e32 v238, v84
	v_exp_f32_e32 v241, v85
	v_exp_f32_e32 v240, v86
	v_exp_f32_e32 v243, v87
	v_exp_f32_e32 v242, v88
	v_exp_f32_e32 v245, v89
	v_exp_f32_e32 v244, v90
	v_exp_f32_e32 v246, v91
	v_exp_f32_e32 v233, v92
	v_exp_f32_e32 v192, v93
	v_exp_f32_e32 v191, v94
	v_exp_f32_e32 v194, v95
	v_exp_f32_e32 v193, v96
	v_exp_f32_e32 v195, v97
	v_exp_f32_e32 v190, v66
	s_setprio 1
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s10, s15, 13
	s_lshl_b32 s15, s15, 14
	s_add_i32 s15, s15, 0
	s_ashr_i32 s10, s10, 3
	s_add_i32 s10, s15, s10
	v_add_u32_e32 v66, s10, v205
	v_add_u32_e32 v67, v200, v217
	v_readfirstlane_b32 s15, v66
	v_add_u32_e32 v66, s10, v202
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v66
	buffer_load_dwordx4 v67, s[24:27], 0 offen lds
	v_add_u32_e32 v67, v200, v216
	s_mov_b32 m0, s15
	v_add_u32_e32 v70, s23, v204
	buffer_load_dwordx4 v67, s[24:27], 0 offen lds
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[186:189], v70 offset:32
	ds_read_b128 v[182:185], v70 offset:64
	ds_read_b128 v[178:181], v70 offset:96
	ds_read_b128 v[174:177], v70 offset:128
	ds_read_b128 v[110:113], v70 offset:160
	ds_read_b128 v[106:109], v70 offset:192
	ds_read_b128 v[102:105], v70 offset:224
	ds_read_b128 v[98:101], v70 offset:512
	ds_read_b128 v[170:173], v70 offset:544
	ds_read_b128 v[166:169], v70 offset:576
	ds_read_b128 v[162:165], v70 offset:608
	ds_read_b128 v[158:161], v70 offset:640
	ds_read_b128 v[126:129], v70 offset:672
	ds_read_b128 v[122:125], v70 offset:704
	ds_read_b128 v[118:121], v70 offset:736
	; sched_barrier mask(0x00000000)
	s_add_i32 s11, s11, 64
	v_add_u32_e32 v216, s0, v216
	v_add_u32_e32 v217, s0, v217
	v_add_u32_e32 v218, s1, v218
	v_add_u32_e32 v219, s1, v219
	v_add_u32_e32 v206, s17, v206
	v_add_u32_e32 v207, s17, v207
	s_cmpk_lt_u32 s11, 0x3f00
	s_mov_b32 s23, s31
	s_mov_b32 s15, s30
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
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[114:117], 0
	v_lshrrev_b32_e32 v70, 1, v1
	v_and_b32_e32 v197, 0xff, v0
	s_ashr_i32 s1, s28, 3
	v_or_b32_e32 v196, v70, v208
	v_lshl_or_b32 v70, s18, 20, v197
	s_add_i32 s0, s16, 0xffffc100
	s_add_i32 s5, 0, 0x10a00
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[154:157], v[82:97]
	s_add_i32 s4, s29, s1
	v_and_b32_e32 v0, 0x100, v0
	s_cmp_lt_i32 s0, 1
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	v_add_lshl_u32 v0, v70, s16, 2
	v_lshl_add_u32 v186, s19, 16, v0
	v_add_f32_e32 v0, v220, v222
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[150:153], v[82:97]
	v_add_f32_e32 v0, v0, v221
	v_add_f32_e32 v0, v0, v224
	v_add_f32_e32 v0, v0, v223
	v_add_f32_e32 v0, v0, v226
	v_add_f32_e32 v0, v0, v225
	v_add_f32_e32 v0, v0, v228
	v_add_f32_e32 v0, v0, v227
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[146:149], v[82:97]
	v_add_f32_e32 v0, v0, v230
	v_add_f32_e32 v0, v0, v229
	v_add_f32_e32 v0, v0, v231
	v_add_f32_e32 v0, v0, v232
	v_add_f32_e32 v0, v0, v235
	v_add_f32_e32 v0, v0, v234
	v_add_f32_e32 v0, v0, v237
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[142:145], v[82:97]
	v_add_f32_e32 v0, v0, v236
	v_add_f32_e32 v0, v0, v239
	v_add_f32_e32 v0, v0, v238
	v_add_f32_e32 v0, v0, v241
	v_add_f32_e32 v0, v0, v240
	v_add_f32_e32 v0, v0, v243
	v_add_f32_e32 v0, v0, v242
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[138:141], v[82:97]
	v_add_f32_e32 v0, v0, v245
	v_add_f32_e32 v0, v0, v244
	v_or_b32_e32 v210, v212, v211
	v_add_f32_e32 v0, v0, v246
	v_add_f32_e32 v174, v0, v233
	v_add_u32_e32 v0, s22, v210
	s_waitcnt vmcnt(0) lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[134:137], v[82:97]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mul_f32_e32 v66, v34, v190
	v_mul_f32_e32 v67, v35, v190
	v_mul_f32_e32 v68, v36, v190
	v_mul_f32_e32 v69, v37, v190
	v_mul_f32_e32 v70, v38, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[130:133], v[82:97]
	v_mul_f32_e32 v71, v39, v190
	v_mul_f32_e32 v72, v40, v190
	v_mul_f32_e32 v73, v41, v190
	v_mul_f32_e32 v74, v42, v190
	v_mul_f32_e32 v75, v43, v190
	v_mul_f32_e32 v76, v44, v190
	v_mul_f32_e32 v77, v45, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[98:101], v[114:117], 0
	v_mul_f32_e32 v78, v46, v190
	v_mul_f32_e32 v79, v47, v190
	v_mul_f32_e32 v80, v48, v190
	v_mul_f32_e32 v81, v49, v190
	v_mul_f32_e32 v46, v62, v190
	v_mul_f32_e32 v47, v63, v190
	v_mul_f32_e32 v48, v64, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[154:157], v[98:113]
	v_mul_f32_e32 v49, v65, v190
	v_cvt_pk_f16_f32 v62, v220, v222
	v_cvt_pk_f16_f32 v63, v221, v224
	v_cvt_pk_f16_f32 v64, v223, v226
	v_cvt_pk_f16_f32 v65, v225, v228
	v_mul_f32_e32 v42, v58, v190
	v_mul_f32_e32 v43, v59, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[150:153], v[98:113]
	v_mul_f32_e32 v44, v60, v190
	v_mul_f32_e32 v45, v61, v190
	v_cvt_pk_f16_f32 v58, v227, v230
	v_cvt_pk_f16_f32 v59, v229, v231
	v_cvt_pk_f16_f32 v60, v232, v235
	v_cvt_pk_f16_f32 v61, v234, v237
	v_mul_f32_e32 v38, v54, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[146:149], v[98:113]
	v_mul_f32_e32 v39, v55, v190
	v_mul_f32_e32 v40, v56, v190
	v_mul_f32_e32 v41, v57, v190
	v_cvt_pk_f16_f32 v54, v236, v239
	v_cvt_pk_f16_f32 v55, v238, v241
	v_cvt_pk_f16_f32 v56, v240, v243
	v_cvt_pk_f16_f32 v57, v242, v245
	v_mfma_f32_32x32x16_f16 v[98:113], v[158:161], v[142:145], v[98:113]
	v_add_u32_e32 v158, v213, v211
	v_add_u32_e32 v172, s22, v158
	v_add_u32_e32 v159, v212, v211
	v_add_u32_e32 v173, s22, v159
	v_mul_f32_e32 v34, v50, v190
	v_mul_f32_e32 v35, v51, v190
	v_mul_f32_e32 v36, v52, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[126:129], v[138:141], v[98:113]
	v_mul_f32_e32 v37, v53, v190
	v_cvt_pk_f16_f32 v50, v244, v246
	v_cvt_pk_f16_f32 v51, v233, v192
	v_cvt_pk_f16_f32 v52, v191, v194
	v_cvt_pk_f16_f32 v53, v193, v195
	v_lshlrev_b32_e32 v215, 2, v1
	v_lshlrev_b32_e32 v187, 1, v1
	v_mfma_f32_32x32x16_f16 v[98:113], v[122:125], v[134:137], v[98:113]
	ds_read_b64_tr_b16 v[122:123], v0
	ds_read_b64_tr_b16 v[124:125], v172 offset:8192
	v_mul_f32_e32 v18, v18, v190
	v_mul_f32_e32 v19, v19, v190
	v_mul_f32_e32 v20, v20, v190
	v_mul_f32_e32 v21, v21, v190
	v_mul_f32_e32 v22, v22, v190
	v_mul_f32_e32 v23, v23, v190
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[122:125], v[62:65], v[66:81]
	v_mul_f32_e32 v24, v24, v190
	v_mul_f32_e32 v25, v25, v190
	v_mul_f32_e32 v26, v26, v190
	v_mul_f32_e32 v27, v27, v190
	v_mul_f32_e32 v28, v28, v190
	v_mul_f32_e32 v29, v29, v190
	v_mul_f32_e32 v30, v30, v190
	v_mfma_f32_32x32x16_f16 v[98:113], v[118:121], v[130:133], v[98:113]
	ds_read_b64_tr_b16 v[118:119], v173 offset:256
	ds_read_b64_tr_b16 v[120:121], v172 offset:8448
	ds_read_b64_tr_b16 v[126:127], v172 offset:8256
	ds_read_b64_tr_b16 v[160:161], v0 offset:512
	ds_read_b64_tr_b16 v[162:163], v172 offset:8704
	ds_read_b64_tr_b16 v[164:165], v173 offset:320
	ds_read_b64_tr_b16 v[168:169], v173 offset:384
	ds_read_b64_tr_b16 v[176:177], v173 offset:192
	ds_read_b64_tr_b16 v[166:167], v172 offset:8512
	ds_read_b64_tr_b16 v[122:123], v172 offset:8320
	ds_read_b64_tr_b16 v[178:179], v172 offset:8384
	v_mul_f32_e32 v31, v31, v190
	v_mul_f32_e32 v32, v32, v190
	v_mul_f32_e32 v33, v33, v190
	v_mul_f32_e32 v0, v2, v190
	v_mul_f32_e32 v1, v3, v190
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[118:121], v[58:61], v[66:81]
	ds_read_b64_tr_b16 v[118:119], v173 offset:768
	ds_read_b64_tr_b16 v[120:121], v172 offset:8960
	ds_read_b64_tr_b16 v[128:129], v172 offset:8768
	ds_read_b64_tr_b16 v[170:171], v172 offset:8576
	ds_read_b64_tr_b16 v[182:183], v172 offset:8640
	v_mul_f32_e32 v2, v4, v190
	v_mul_f32_e32 v3, v5, v190
	v_mul_f32_e32 v4, v6, v190
	v_mul_f32_e32 v5, v7, v190
	v_mul_f32_e32 v6, v8, v190
	v_mul_f32_e32 v7, v9, v190
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[160:163], v[54:57], v[66:81]
	ds_read_b64_tr_b16 v[124:125], v173 offset:64
	ds_read_b64_tr_b16 v[162:163], v172 offset:9024
	ds_read_b64_tr_b16 v[218:219], v172 offset:8832
	ds_read_b64_tr_b16 v[222:223], v172 offset:8896
	ds_read_b64_tr_b16 v[160:161], v173 offset:832
	ds_read_b64_tr_b16 v[224:225], v173 offset:896
	ds_read_b64_tr_b16 v[228:229], v173 offset:960
	v_mul_f32_e32 v8, v10, v190
	v_mul_f32_e32 v9, v11, v190
	v_mul_f32_e32 v10, v12, v190
	v_mul_f32_e32 v11, v13, v190
	v_mul_f32_e32 v12, v14, v190
	v_mul_f32_e32 v13, v15, v190
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[66:81], v[118:121], v[50:53], v[66:81]
	ds_read_b64_tr_b16 v[120:121], v173 offset:128
	ds_read_b64_tr_b16 v[226:227], v172 offset:9088
	ds_read_b64_tr_b16 v[230:231], v172 offset:9152
	v_mul_f32_e32 v14, v16, v190
	v_mul_f32_e32 v15, v17, v190
	v_max_f32_e32 v16, v83, v83
	v_max_f32_e32 v17, v82, v82
	v_max_f32_e32 v16, v17, v16
	v_max3_f32 v16, v16, v84, v85
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[34:49], v[124:127], v[62:65], v[34:49]
	ds_read_b64_tr_b16 v[126:127], v173 offset:576
	ds_read_b64_tr_b16 v[216:217], v173 offset:640
	ds_read_b64_tr_b16 v[220:221], v173 offset:704
	ds_read_b64_tr_b16 v[180:181], v173 offset:448
	v_max3_f32 v16, v16, v86, v87
	v_max3_f32 v16, v16, v88, v89
	v_max3_f32 v16, v16, v90, v91
	v_max3_f32 v16, v16, v92, v93
	v_max3_f32 v16, v16, v94, v95
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[18:33], v[120:123], v[62:65], v[18:33]
	v_max3_f32 v16, v16, v96, v97
	v_max3_f32 v16, v16, v98, v99
	v_max3_f32 v16, v16, v100, v101
	v_max3_f32 v16, v16, v102, v103
	v_max3_f32 v16, v16, v104, v105
	v_add_u32_e32 v124, s14, v204
	v_max3_f32 v16, v16, v106, v107
	v_mfma_f32_32x32x16_f16 v[0:15], v[176:179], v[62:65], v[0:15]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_max3_f32 v16, v16, v108, v109
	v_max3_f32 v16, v16, v110, v111
	v_max3_f32 v16, v16, v112, v113
	v_mov_b32_e32 v17, v16
	v_mfma_f32_32x32x16_f16 v[34:49], v[164:167], v[58:61], v[34:49]
	s_nop 0
	v_permlane32_swap_b32_e32 v16, v17
	v_max3_f32 v17, v214, v16, v17
	s_mov_b32 s11, 0x3e0293ee
	v_mul_f32_e32 v16, 0x3e0293ee, v17
	v_fma_f32 v118, v86, s11, -v16
	v_fma_f32 v119, v87, s11, -v16
	v_mfma_f32_32x32x16_f16 v[18:33], v[168:171], v[58:61], v[18:33]
	v_fma_f32 v120, v88, s11, -v16
	v_fma_f32 v121, v89, s11, -v16
	ds_read_b128 v[86:89], v124 offset:34784
	v_fma_f32 v90, v90, s11, -v16
	v_fma_f32 v91, v91, s11, -v16
	v_fma_f32 v122, v92, s11, -v16
	v_fma_f32 v123, v93, s11, -v16
	v_mfma_f32_32x32x16_f16 v[0:15], v[180:183], v[58:61], v[0:15]
	v_exp_f32_e32 v189, v90
	v_exp_f32_e32 v211, v91
	ds_read_b128 v[90:93], v124 offset:34816
	v_fma_f32 v94, v94, s11, -v16
	v_fma_f32 v95, v95, s11, -v16
	v_fma_f32 v96, v96, s11, -v16
	v_fma_f32 v97, v97, s11, -v16
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[54:57], v[34:49]
	v_fma_f32 v98, v98, s11, -v16
	v_fma_f32 v99, v99, s11, -v16
	v_fma_f32 v100, v100, s11, -v16
	v_fma_f32 v101, v101, s11, -v16
	v_fma_f32 v102, v102, s11, -v16
	v_fma_f32 v103, v103, s11, -v16
	v_fma_f32 v104, v104, s11, -v16
	v_mfma_f32_32x32x16_f16 v[18:33], v[216:219], v[54:57], v[18:33]
	v_fma_f32 v105, v105, s11, -v16
	v_fma_f32 v106, v106, s11, -v16
	v_fma_f32 v107, v107, s11, -v16
	v_fma_f32 v108, v108, s11, -v16
	v_fma_f32 v109, v109, s11, -v16
	v_fma_f32 v110, v110, s11, -v16
	v_fma_f32 v111, v111, s11, -v16
	v_mfma_f32_32x32x16_f16 v[0:15], v[220:223], v[54:57], v[0:15]
	ds_read_b128 v[54:57], v124 offset:34752
	v_fma_f32 v112, v112, s11, -v16
	v_fma_f32 v113, v113, s11, -v16
	v_exp_f32_e32 v183, v118
	v_exp_f32_e32 v184, v119
	v_exp_f32_e32 v185, v120
	v_exp_f32_e32 v188, v121
	v_mfma_f32_32x32x16_f16 v[34:49], v[160:163], v[50:53], v[34:49]
	v_exp_f32_e32 v232, v122
	v_exp_f32_e32 v161, v123
	v_exp_f32_e32 v162, v94
	v_exp_f32_e32 v163, v95
	v_exp_f32_e32 v164, v96
	v_exp_f32_e32 v165, v97
	v_exp_f32_e32 v173, v98
	v_mfma_f32_32x32x16_f16 v[18:33], v[224:227], v[50:53], v[18:33]
	v_exp_f32_e32 v175, v99
	v_exp_f32_e32 v166, v100
	v_exp_f32_e32 v167, v101
	v_exp_f32_e32 v168, v102
	v_exp_f32_e32 v169, v103
	v_exp_f32_e32 v176, v104
	v_exp_f32_e32 v177, v105
	v_mfma_f32_32x32x16_f16 v[0:15], v[228:231], v[50:53], v[0:15]
	v_exp_f32_e32 v178, v106
	v_exp_f32_e32 v170, v107
	v_exp_f32_e32 v171, v108
	v_exp_f32_e32 v172, v109
	v_exp_f32_e32 v179, v110
	v_exp_f32_e32 v180, v111
	v_exp_f32_e32 v181, v112
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[54:57], v[114:117], 0
	v_exp_f32_e32 v182, v113
	v_add3_u32 v224, v198, v207, s17
	v_lshl_add_u32 v209, v209, 8, s5
	v_add_u32_e32 v225, 1, v224
	v_add_u32_e32 v226, 2, v224
	v_add_u32_e32 v227, 3, v224
	v_fmac_f32_e32 v247, 0xbe0293ee, v17
	v_mfma_f32_32x32x16_f16 v[50:65], v[86:89], v[154:157], v[50:65]
	ds_read_b128 v[86:89], v124 offset:34848
	v_exp_f32_e32 v160, v247
	v_add_u32_e32 v228, 4, v224
	v_add_u32_e32 v229, 5, v224
	v_add_u32_e32 v230, 6, v224
	v_add_u32_e32 v231, 7, v224
	s_mov_b32 s26, s6
	v_mfma_f32_32x32x16_f16 v[50:65], v[90:93], v[150:153], v[50:65]
	ds_read_b128 v[90:93], v124 offset:34880
	s_mov_b32 s27, s7
	v_fma_f32 v82, v82, s11, -v16
	v_fma_f32 v83, v83, s11, -v16
	v_fma_f32 v84, v84, s11, -v16
	v_fma_f32 v85, v85, s11, -v16
	v_exp_f32_e32 v82, v82
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[50:65], v[86:89], v[146:149], v[50:65]
	ds_read_b128 v[86:89], v124 offset:34912
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v85, v85
	v_mul_f32_e32 v34, v34, v160
	v_mul_f32_e32 v35, v35, v160
	v_mul_f32_e32 v36, v36, v160
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[50:65], v[90:93], v[142:145], v[50:65]
	ds_read_b128 v[90:93], v124 offset:34944
	v_mul_f32_e32 v37, v37, v160
	v_mul_f32_e32 v38, v38, v160
	v_mul_f32_e32 v39, v39, v160
	v_mul_f32_e32 v40, v40, v160
	v_mul_f32_e32 v41, v41, v160
	v_mul_f32_e32 v42, v42, v160
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[50:65], v[86:89], v[138:141], v[50:65]
	ds_read_b128 v[86:89], v124 offset:34976
	ds_read_b128 v[94:97], v124 offset:35264
	ds_read_b128 v[98:101], v124 offset:35296
	ds_read_b128 v[102:105], v124 offset:35328
	ds_read_b128 v[106:109], v124 offset:35360
	ds_read_b128 v[110:113], v124 offset:35392
	ds_read_b128 v[216:219], v124 offset:35424
	v_mul_f32_e32 v43, v43, v160
	v_mul_f32_e32 v44, v44, v160
	v_mul_f32_e32 v45, v45, v160
	v_mul_f32_e32 v46, v46, v160
	v_mul_f32_e32 v47, v47, v160
	v_mul_f32_e32 v48, v48, v160
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[50:65], v[90:93], v[134:137], v[50:65]
	ds_read_b128 v[90:93], v124 offset:35456
	ds_read_b128 v[220:223], v124 offset:35488
	v_mul_f32_e32 v49, v49, v160
	v_lshlrev_b32_e32 v208, 2, v208
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[114:129], v[94:97], v[114:117], 0
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[114:129], v[98:101], v[154:157], v[114:129]
	v_mul_f32_e32 v98, v66, v160
	v_mul_f32_e32 v99, v67, v160
	v_mul_f32_e32 v100, v68, v160
	v_mul_f32_e32 v101, v69, v160
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[114:129], v[102:105], v[150:153], v[114:129]
	v_mul_f32_e32 v102, v70, v160
	v_mul_f32_e32 v103, v71, v160
	v_mul_f32_e32 v104, v72, v160
	v_mul_f32_e32 v105, v73, v160
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[114:129], v[106:109], v[146:149], v[114:129]
	v_mul_f32_e32 v108, v76, v160
	v_add_u32_e32 v76, s10, v210
	v_add_u32_e32 v146, s10, v158
	v_mul_f32_e32 v106, v74, v160
	v_mul_f32_e32 v107, v75, v160
	v_mul_f32_e32 v109, v77, v160
	v_add_u32_e32 v147, s10, v159
	v_mfma_f32_32x32x16_f16 v[50:65], v[86:89], v[130:133], v[50:65]
	v_add3_u32 v86, v198, v206, s17
	v_add_u32_e32 v198, s5, v201
	v_add_u32_e32 v87, 1, v86
	v_add_u32_e32 v88, 2, v86
	v_add_u32_e32 v89, 3, v86
	v_add_u32_e32 v94, 4, v86
	v_add_u32_e32 v95, 5, v86
	v_add_u32_e32 v96, 6, v86
	v_add_u32_e32 v97, 7, v86
	ds_write_b128 v198, v[224:227]
	ds_write_b128 v198, v[86:89] offset:8192
	v_add3_u32 v86, v209, v215, v200
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[114:129], v[110:113], v[142:145], v[114:129]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b32 v[86:87], v86 offset1:8
	v_add_u32_e32 v88, s4, v205
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s5, v88
	s_mov_b32 m0, s5
	v_lshlrev_b32_e32 v86, 1, v86
	s_barrier
	ds_write_b128 v198, v[228:231]
	ds_write_b128 v198, v[94:97] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v86, s[24:27], 0 offen lds
	v_add_u32_e32 v86, s4, v202
	v_mfma_f32_32x32x16_f16 v[114:129], v[216:219], v[138:141], v[114:129]
	v_readfirstlane_b32 s5, v86
	v_lshlrev_b32_e32 v87, 1, v87
	s_mov_b32 m0, s5
	v_mul_f32_e32 v110, v78, v160
	buffer_load_dwordx4 v87, s[24:27], 0 offen lds
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[66:67], v76
	ds_read_b64_tr_b16 v[68:69], v146 offset:8192
	v_mul_f32_e32 v111, v79, v160
	v_mul_f32_e32 v112, v80, v160
	v_mul_f32_e32 v113, v81, v160
	v_cvt_pk_f16_f32 v142, v82, v83
	v_cvt_pk_f16_f32 v143, v84, v85
	v_cvt_pk_f16_f32 v144, v183, v184
	v_cvt_pk_f16_f32 v145, v185, v188
	v_mfma_f32_32x32x16_f16 v[114:129], v[90:93], v[134:137], v[114:129]
	ds_read_b64_tr_b16 v[70:71], v147 offset:256
	ds_read_b64_tr_b16 v[72:73], v146 offset:8448
	ds_read_b64_tr_b16 v[74:75], v146 offset:8256
	ds_read_b64_tr_b16 v[76:77], v76 offset:512
	v_cvt_pk_f16_f32 v134, v189, v211
	v_cvt_pk_f16_f32 v135, v232, v161
	v_cvt_pk_f16_f32 v136, v162, v163
	v_cvt_pk_f16_f32 v137, v164, v165
	v_cvt_pk_f16_f32 v138, v178, v170
	v_cvt_pk_f16_f32 v139, v171, v172
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[98:113], v[66:69], v[142:145], v[98:113]
	ds_read_b64_tr_b16 v[78:79], v146 offset:8704
	ds_read_b64_tr_b16 v[66:67], v147 offset:320
	ds_read_b64_tr_b16 v[86:87], v147 offset:384
	ds_read_b64_tr_b16 v[148:149], v147 offset:192
	ds_read_b64_tr_b16 v[68:69], v146 offset:8512
	ds_read_b64_tr_b16 v[92:93], v146 offset:8320
	ds_read_b64_tr_b16 v[150:151], v146 offset:8384
	ds_read_b64_tr_b16 v[94:95], v147 offset:768
	ds_read_b64_tr_b16 v[96:97], v146 offset:8960
	ds_read_b64_tr_b16 v[80:81], v146 offset:8768
	ds_read_b64_tr_b16 v[88:89], v146 offset:8576
	ds_read_b64_tr_b16 v[154:155], v146 offset:8640
	v_cvt_pk_f16_f32 v140, v179, v180
	v_cvt_pk_f16_f32 v141, v181, v182
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[98:113], v[70:73], v[134:137], v[98:113]
	v_add_f32_e32 v70, v174, v192
	v_add_f32_e32 v70, v70, v191
	v_add_f32_e32 v70, v70, v194
	v_add_f32_e32 v70, v70, v193
	v_mul_f32_e32 v71, v23, v160
	v_mfma_f32_32x32x16_f16 v[114:129], v[220:223], v[130:133], v[114:129]
	ds_read_b64_tr_b16 v[72:73], v147 offset:64
	ds_read_b64_tr_b16 v[206:207], v146 offset:9024
	ds_read_b64_tr_b16 v[214:215], v146 offset:8832
	ds_read_b64_tr_b16 v[218:219], v146 offset:8896
	ds_read_b64_tr_b16 v[204:205], v147 offset:832
	ds_read_b64_tr_b16 v[220:221], v147 offset:896
	ds_read_b64_tr_b16 v[224:225], v147 offset:960
	v_cvt_pk_f16_f32 v130, v173, v175
	v_cvt_pk_f16_f32 v131, v166, v167
	v_cvt_pk_f16_f32 v132, v168, v169
	v_cvt_pk_f16_f32 v133, v176, v177
	ds_read_b64_tr_b16 v[90:91], v147 offset:128
	ds_read_b64_tr_b16 v[222:223], v146 offset:9088
	ds_read_b64_tr_b16 v[226:227], v146 offset:9152
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[34:49], v[72:75], v[142:145], v[34:49]
	v_mul_f32_e32 v72, v24, v160
	v_mul_f32_e32 v73, v25, v160
	v_mul_f32_e32 v74, v26, v160
	v_mul_f32_e32 v75, v27, v160
	v_mfma_f32_32x32x16_f16 v[98:113], v[76:79], v[130:133], v[98:113]
	ds_read_b64_tr_b16 v[78:79], v147 offset:576
	v_mul_f32_e32 v76, v28, v160
	v_mul_f32_e32 v77, v29, v160
	ds_read_b64_tr_b16 v[212:213], v147 offset:640
	ds_read_b64_tr_b16 v[216:217], v147 offset:704
	ds_read_b64_tr_b16 v[152:153], v147 offset:448
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[34:49], v[66:69], v[134:137], v[34:49]
	v_mul_f32_e32 v66, v18, v160
	v_add_f32_e32 v18, v82, v83
	v_add_f32_e32 v18, v84, v18
	v_add_f32_e32 v18, v85, v18
	v_add_f32_e32 v18, v183, v18
	v_mul_f32_e32 v67, v19, v160
	v_mul_f32_e32 v68, v20, v160
	v_mfma_f32_32x32x16_f16 v[98:113], v[94:97], v[138:141], v[98:113]
	v_add_f32_e32 v94, v70, v195
	v_mul_f32_e32 v69, v21, v160
	v_mul_f32_e32 v70, v22, v160
	v_add_f32_e32 v18, v184, v18
	v_add_f32_e32 v18, v185, v18
	v_add_f32_e32 v18, v188, v18
	v_add_f32_e32 v18, v189, v18
	v_mfma_f32_32x32x16_f16 v[34:49], v[78:81], v[130:133], v[34:49]
	v_mul_f32_e32 v78, v30, v160
	v_mul_f32_e32 v79, v31, v160
	v_mul_f32_e32 v80, v32, v160
	v_mul_f32_e32 v81, v33, v160
	v_add_f32_e32 v18, v211, v18
	v_add_f32_e32 v18, v232, v18
	v_mul_f32_e32 v82, v0, v160
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[142:145], v[66:81]
	v_add_f32_e32 v0, v161, v18
	v_mov_b32_e32 v95, v94
	v_add_f32_e32 v0, v162, v0
	s_nop 0
	v_permlane32_swap_b32_e32 v94, v95
	v_add_f32_e32 v0, v163, v0
	v_add_f32_e32 v146, v94, v95
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[134:137], v[66:81]
	v_mul_f32_e32 v83, v1, v160
	v_mul_f32_e32 v84, v2, v160
	v_mul_f32_e32 v85, v3, v160
	v_mul_f32_e32 v86, v4, v160
	v_mul_f32_e32 v87, v5, v160
	v_mul_f32_e32 v88, v6, v160
	v_mul_f32_e32 v89, v7, v160
	v_mul_f32_e32 v90, v8, v160
	v_mul_f32_e32 v91, v9, v160
	v_mul_f32_e32 v92, v10, v160
	v_mul_f32_e32 v93, v11, v160
	v_mul_f32_e32 v94, v12, v160
	v_mul_f32_e32 v95, v13, v160
	v_mul_f32_e32 v96, v14, v160
	v_mul_f32_e32 v97, v15, v160
	v_add_f32_e32 v0, v164, v0
	v_add_f32_e32 v0, v165, v0
	v_mfma_f32_32x32x16_f16 v[82:97], v[148:151], v[142:145], v[82:97]
	v_add_f32_e32 v0, v173, v0
	v_add_f32_e32 v0, v175, v0
	v_add_f32_e32 v0, v166, v0
	v_add_f32_e32 v0, v167, v0
	v_add_f32_e32 v0, v168, v0
	v_add_f32_e32 v0, v169, v0
	v_add_f32_e32 v0, v176, v0
	v_mfma_f32_32x32x16_f16 v[82:97], v[152:155], v[134:137], v[82:97]
	v_add_f32_e32 v0, v177, v0
	v_add_f32_e32 v0, v178, v0
	v_add_f32_e32 v0, v170, v0
	v_add_f32_e32 v0, v171, v0
	v_add_f32_e32 v0, v172, v0
	v_add_f32_e32 v0, v179, v0
	v_add_f32_e32 v0, v180, v0
	v_add_f32_e32 v0, v181, v0
	v_mfma_f32_32x32x16_f16 v[66:81], v[212:215], v[130:133], v[66:81]
	v_max_f32_e32 v1, v50, v50
	v_fmac_f32_e32 v146, v203, v190
	v_mfma_f32_32x32x16_f16 v[82:97], v[216:219], v[130:133], v[82:97]
	v_add_f32_e32 v131, v182, v0
	v_max_f32_e32 v0, v51, v51
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v52, v53
	v_max3_f32 v0, v0, v54, v55
	v_max3_f32 v0, v0, v56, v57
	v_max3_f32 v0, v0, v58, v59
	v_max3_f32 v0, v0, v60, v61
	v_max3_f32 v0, v0, v62, v63
	v_max3_f32 v0, v0, v64, v65
	v_max3_f32 v0, v0, v114, v115
	v_max3_f32 v0, v0, v116, v117
	v_max3_f32 v0, v0, v118, v119
	v_max3_f32 v0, v0, v120, v121
	v_max3_f32 v0, v0, v122, v123
	v_max3_f32 v0, v0, v124, v125
	v_max3_f32 v0, v0, v126, v127
	v_max3_f32 v0, v0, v128, v129
	v_mfma_f32_32x32x16_f16 v[34:49], v[204:207], v[138:141], v[34:49]
	v_mov_b32_e32 v1, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v1
	v_max3_f32 v130, v17, v0, v1
	v_fmac_f32_e32 v16, 0xbe0293ee, v130
	v_exp_f32_e32 v168, v16
	v_mul_f32_e32 v0, 0xbe0293ee, v130
	v_fmamk_f32 v20, v116, 0x3e0293ee, v0
	v_fmamk_f32 v1, v50, 0x3e0293ee, v0
	v_fmamk_f32 v2, v51, 0x3e0293ee, v0
	v_fmamk_f32 v3, v52, 0x3e0293ee, v0
	v_fmamk_f32 v4, v53, 0x3e0293ee, v0
	v_fmamk_f32 v5, v54, 0x3e0293ee, v0
	v_fmamk_f32 v6, v55, 0x3e0293ee, v0
	v_fmamk_f32 v7, v56, 0x3e0293ee, v0
	v_fmamk_f32 v8, v57, 0x3e0293ee, v0
	v_fmamk_f32 v28, v124, 0x3e0293ee, v0
	v_fmamk_f32 v29, v125, 0x3e0293ee, v0
	v_fmamk_f32 v30, v126, 0x3e0293ee, v0
	v_fmamk_f32 v31, v127, 0x3e0293ee, v0
	v_exp_f32_e32 v154, v20
	v_mul_f32_e32 v20, v38, v168
	v_add_u32_e32 v38, s4, v210
	v_mfma_f32_32x32x16_f16 v[66:81], v[220:223], v[138:141], v[66:81]
	v_fmamk_f32 v9, v58, 0x3e0293ee, v0
	v_fmamk_f32 v10, v59, 0x3e0293ee, v0
	v_fmamk_f32 v22, v118, 0x3e0293ee, v0
	v_fmamk_f32 v23, v119, 0x3e0293ee, v0
	v_fmamk_f32 v26, v122, 0x3e0293ee, v0
	v_fmamk_f32 v27, v123, 0x3e0293ee, v0
	v_exp_f32_e32 v58, v1
	v_mfma_f32_32x32x16_f16 v[82:97], v[224:227], v[138:141], v[82:97]
	v_exp_f32_e32 v59, v2
	v_exp_f32_e32 v118, v3
	v_exp_f32_e32 v119, v4
	v_exp_f32_e32 v122, v5
	v_exp_f32_e32 v140, v6
	v_exp_f32_e32 v141, v7
	v_exp_f32_e32 v142, v8
	v_exp_f32_e32 v165, v28
	v_exp_f32_e32 v166, v29
	v_exp_f32_e32 v167, v30
	v_exp_f32_e32 v169, v31
	v_add_u32_e32 v123, s4, v158
	ds_read_b64_tr_b16 v[28:29], v38
	ds_read_b64_tr_b16 v[30:31], v123 offset:8192
	v_fmamk_f32 v11, v60, 0x3e0293ee, v0
	v_fmamk_f32 v12, v61, 0x3e0293ee, v0
	v_fmamk_f32 v13, v62, 0x3e0293ee, v0
	v_fmamk_f32 v14, v63, 0x3e0293ee, v0
	v_fmamk_f32 v15, v64, 0x3e0293ee, v0
	v_fmamk_f32 v17, v65, 0x3e0293ee, v0
	v_fmamk_f32 v18, v114, 0x3e0293ee, v0
	v_fmamk_f32 v19, v115, 0x3e0293ee, v0
	v_fmamk_f32 v21, v117, 0x3e0293ee, v0
	v_fmamk_f32 v24, v120, 0x3e0293ee, v0
	v_fmamk_f32 v25, v121, 0x3e0293ee, v0
	v_fmamk_f32 v32, v128, 0x3e0293ee, v0
	v_fmac_f32_e32 v0, 0x3e0293ee, v129
	v_exp_f32_e32 v143, v9
	v_exp_f32_e32 v144, v10
	v_exp_f32_e32 v145, v11
	v_exp_f32_e32 v147, v12
	v_exp_f32_e32 v148, v13
	v_exp_f32_e32 v149, v14
	v_exp_f32_e32 v150, v15
	v_exp_f32_e32 v171, v0
	v_mul_f32_e32 v0, v98, v168
	v_mul_f32_e32 v1, v99, v168
	v_mul_f32_e32 v2, v100, v168
	v_mul_f32_e32 v3, v101, v168
	v_mul_f32_e32 v4, v102, v168
	v_mul_f32_e32 v5, v103, v168
	v_mul_f32_e32 v6, v104, v168
	v_mul_f32_e32 v7, v105, v168
	v_mul_f32_e32 v8, v106, v168
	v_mul_f32_e32 v9, v107, v168
	v_mul_f32_e32 v10, v108, v168
	v_mul_f32_e32 v11, v109, v168
	v_mul_f32_e32 v12, v110, v168
	v_mul_f32_e32 v13, v111, v168
	v_mul_f32_e32 v14, v112, v168
	v_mul_f32_e32 v15, v113, v168
	v_cvt_pk_f16_f32 v110, v58, v59
	v_cvt_pk_f16_f32 v111, v118, v119
	v_cvt_pk_f16_f32 v112, v122, v140
	v_cvt_pk_f16_f32 v113, v141, v142
	v_exp_f32_e32 v151, v17
	v_add_u32_e32 v158, s4, v159
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[0:15], v[28:31], v[110:113], v[0:15]
	v_exp_f32_e32 v152, v18
	v_exp_f32_e32 v153, v19
	v_exp_f32_e32 v155, v21
	v_exp_f32_e32 v170, v32
	v_mul_f32_e32 v16, v34, v168
	v_mul_f32_e32 v17, v35, v168
	v_mul_f32_e32 v18, v36, v168
	v_mul_f32_e32 v19, v37, v168
	v_mul_f32_e32 v21, v39, v168
	ds_read_b64_tr_b16 v[32:33], v158 offset:256
	ds_read_b64_tr_b16 v[34:35], v123 offset:8448
	ds_read_b64_tr_b16 v[36:37], v123 offset:8256
	ds_read_b64_tr_b16 v[38:39], v38 offset:512
	v_cvt_pk_f16_f32 v102, v143, v144
	v_cvt_pk_f16_f32 v103, v145, v147
	v_cvt_pk_f16_f32 v104, v148, v149
	v_cvt_pk_f16_f32 v105, v150, v151
	v_exp_f32_e32 v156, v22
	v_exp_f32_e32 v157, v23
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[0:15], v[32:35], v[102:105], v[0:15]
	v_exp_f32_e32 v161, v24
	v_exp_f32_e32 v162, v25
	v_cvt_pk_f16_f32 v98, v152, v153
	v_cvt_pk_f16_f32 v99, v154, v155
	v_cvt_pk_f16_f32 v100, v156, v157
	v_cvt_pk_f16_f32 v101, v161, v162
	v_exp_f32_e32 v163, v26
	v_exp_f32_e32 v164, v27
	v_mul_f32_e32 v22, v40, v168
	v_mul_f32_e32 v23, v41, v168
	v_mul_f32_e32 v24, v42, v168
	v_mul_f32_e32 v25, v43, v168
	v_mul_f32_e32 v26, v44, v168
	v_mul_f32_e32 v27, v45, v168
	ds_read_b64_tr_b16 v[40:41], v123 offset:8704
	ds_read_b64_tr_b16 v[42:43], v158 offset:320
	ds_read_b64_tr_b16 v[50:51], v158 offset:384
	ds_read_b64_tr_b16 v[114:115], v158 offset:192
	ds_read_b64_tr_b16 v[44:45], v123 offset:8512
	ds_read_b64_tr_b16 v[56:57], v123 offset:8320
	ds_read_b64_tr_b16 v[116:117], v123 offset:8384
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[0:15], v[38:41], v[98:101], v[0:15]
	ds_read_b64_tr_b16 v[28:29], v158 offset:768
	ds_read_b64_tr_b16 v[30:31], v123 offset:8960
	ds_read_b64_tr_b16 v[60:61], v123 offset:8768
	ds_read_b64_tr_b16 v[52:53], v123 offset:8576
	ds_read_b64_tr_b16 v[120:121], v123 offset:8640
	v_cvt_pk_f16_f32 v106, v163, v164
	v_cvt_pk_f16_f32 v107, v165, v166
	v_cvt_pk_f16_f32 v108, v167, v169
	v_cvt_pk_f16_f32 v109, v170, v171
	ds_read_b64_tr_b16 v[34:35], v158 offset:64
	ds_read_b64_tr_b16 v[64:65], v123 offset:9024
	ds_read_b64_tr_b16 v[124:125], v123 offset:8832
	ds_read_b64_tr_b16 v[128:129], v123 offset:8896
	ds_read_b64_tr_b16 v[62:63], v158 offset:832
	ds_read_b64_tr_b16 v[132:133], v158 offset:896
	ds_read_b64_tr_b16 v[136:137], v158 offset:960
	v_mov_b32_e32 v32, v131
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[0:15], v[28:31], v[106:109], v[0:15]
	v_mul_f32_e32 v28, v46, v168
	v_mul_f32_e32 v29, v47, v168
	v_mul_f32_e32 v30, v48, v168
	v_mul_f32_e32 v31, v49, v168
	v_permlane32_swap_b32_e32 v131, v32
	ds_read_b64_tr_b16 v[54:55], v158 offset:128
	ds_read_b64_tr_b16 v[134:135], v123 offset:9088
	ds_read_b64_tr_b16 v[138:139], v123 offset:9152
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[16:31], v[34:37], v[110:113], v[16:31]
	v_add_f32_e32 v131, v131, v32
	v_add_f32_e32 v32, v58, v59
	ds_read_b64_tr_b16 v[58:59], v158 offset:576
	v_add_f32_e32 v32, v118, v32
	v_add_f32_e32 v32, v119, v32
	v_add_f32_e32 v48, v122, v32
	v_add_f32_e32 v48, v140, v48
	v_mfma_f32_32x32x16_f16 v[16:31], v[42:45], v[102:105], v[16:31]
	v_add_f32_e32 v48, v141, v48
	v_mul_f32_e32 v32, v66, v168
	v_mul_f32_e32 v33, v67, v168
	v_mul_f32_e32 v34, v68, v168
	v_mul_f32_e32 v35, v69, v168
	v_mul_f32_e32 v36, v70, v168
	v_mul_f32_e32 v37, v71, v168
	v_mul_f32_e32 v38, v72, v168
	v_mul_f32_e32 v39, v73, v168
	v_mul_f32_e32 v40, v74, v168
	v_mul_f32_e32 v41, v75, v168
	v_mul_f32_e32 v42, v76, v168
	v_mul_f32_e32 v43, v77, v168
	v_mul_f32_e32 v44, v78, v168
	v_mul_f32_e32 v45, v79, v168
	v_mul_f32_e32 v46, v80, v168
	v_mul_f32_e32 v47, v81, v168
	v_add_f32_e32 v48, v142, v48
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[16:31], v[58:61], v[98:101], v[16:31]
	v_add_f32_e32 v48, v143, v48
	v_add_f32_e32 v48, v144, v48
	v_add_f32_e32 v48, v145, v48
	v_add_f32_e32 v48, v147, v48
	v_add_f32_e32 v48, v148, v48
	v_add_f32_e32 v48, v149, v48
	v_add_f32_e32 v48, v150, v48
	v_mfma_f32_32x32x16_f16 v[32:47], v[54:57], v[110:113], v[32:47]
	v_add_f32_e32 v48, v151, v48
	v_mul_f32_e32 v49, v83, v168
	v_mul_f32_e32 v54, v88, v168
	v_mul_f32_e32 v55, v89, v168
	v_mul_f32_e32 v56, v90, v168
	v_mul_f32_e32 v57, v91, v168
	v_mul_f32_e32 v58, v92, v168
	v_mfma_f32_32x32x16_f16 v[16:31], v[62:65], v[106:109], v[16:31]
	v_add_f32_e32 v64, v152, v48
	v_mul_f32_e32 v48, v82, v168
	v_mul_f32_e32 v59, v93, v168
	v_mul_f32_e32 v60, v94, v168
	v_mul_f32_e32 v61, v95, v168
	v_mul_f32_e32 v62, v96, v168
	v_mul_f32_e32 v63, v97, v168
	v_mfma_f32_32x32x16_f16 v[32:47], v[50:53], v[102:105], v[32:47]
	v_mul_f32_e32 v50, v84, v168
	v_mul_f32_e32 v51, v85, v168
	v_mul_f32_e32 v52, v86, v168
	v_mul_f32_e32 v53, v87, v168
	ds_read_b64_tr_b16 v[122:123], v158 offset:640
	ds_read_b64_tr_b16 v[126:127], v158 offset:704
	ds_read_b64_tr_b16 v[118:119], v158 offset:448
	v_add_f32_e32 v64, v153, v64
	v_add_f32_e32 v64, v154, v64
	v_mfma_f32_32x32x16_f16 v[48:63], v[114:117], v[110:113], v[48:63]
	v_add_f32_e32 v64, v155, v64
	v_add_f32_e32 v64, v156, v64
	v_add_f32_e32 v64, v157, v64
	v_add_f32_e32 v64, v161, v64
	v_add_f32_e32 v64, v162, v64
	v_add_f32_e32 v64, v163, v64
	v_add_f32_e32 v64, v164, v64
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[48:63], v[118:121], v[102:105], v[48:63]
	v_add_f32_e32 v64, v165, v64
	v_add_f32_e32 v64, v166, v64
	v_add_f32_e32 v64, v167, v64
	v_add_f32_e32 v64, v169, v64
	v_add_f32_e32 v64, v170, v64
	v_add_f32_e32 v64, v171, v64
	v_mov_b32_e32 v65, v64
	v_mfma_f32_32x32x16_f16 v[32:47], v[122:125], v[98:101], v[32:47]
	s_nop 0
	v_permlane32_swap_b32_e32 v64, v65
	v_add_f32_e32 v64, v64, v65
	v_fmac_f32_e32 v131, v146, v160
	v_fmac_f32_e32 v64, v131, v168
	s_mov_b32 s4, 0x800000
	v_cmp_gt_f32_e32 vcc, s4, v64
	v_mfma_f32_32x32x16_f16 v[48:63], v[126:129], v[98:101], v[48:63]
	v_mov_b32_e32 v66, 0x42000000
	v_cndmask_b32_e64 v65, 0, 32, vcc
	v_ldexp_f32 v65, v64, v65
	v_log_f32_e32 v65, v65
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_barrier
	v_mfma_f32_32x32x16_f16 v[32:47], v[132:135], v[106:109], v[32:47]
	v_sub_f32_e32 v65, v65, v66
	v_add_f32_e32 v65, v130, v65
	v_add3_u32 v66, 0, v208, v187
	ds_write_b32 v66, v65
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[48:63], v[136:139], v[106:109], v[48:63]
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	s_sub_i32 s6, 0x4000, s16
	v_or_b32_e32 v65, s16, v196
	s_movk_i32 s4, 0x4000
	v_cmp_gt_i32_e32 vcc, s6, v197
	v_cmp_gt_i32_e64 s[4:5], s4, v65
	v_bfrev_b32_e32 v65, 1
	s_and_b64 vcc, s[0:1], vcc
	v_cndmask_b32_e32 v65, v65, v186, vcc
	s_barrier
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr4_sgpr5
                                        ; implicit-def: $vgpr65
.LBB0_9:
	v_bfrev_b32_e32 v65, 1
	v_cndmask_b32_e64 v65, v65, v186, s[0:1]
	s_or_b64 s[4:5], s[4:5], exec
	s_barrier
.LBB0_10:
	v_div_scale_f32 v66, s[0:1], v64, v64, 1.0
	v_rcp_f32_e32 v67, v66
	v_div_scale_f32 v68, vcc, 1.0, v64, 1.0
	s_mul_i32 s0, s3, s19
	v_fma_f32 v69, -v66, v67, 1.0
	v_fmac_f32_e32 v67, v69, v67
	v_mul_f32_e32 v69, v68, v67
	v_fma_f32 v70, -v66, v69, v68
	v_fmac_f32_e32 v69, v70, v67
	v_fma_f32 v66, -v66, v69, v68
	v_div_fmas_f32 v66, v66, v67, v69
	v_lshl_add_u32 v67, v197, 2, 0
	v_div_fixup_f32 v64, v66, v64, 1.0
	s_mul_i32 s1, s2, s18
	ds_read_b32 v67, v67
	v_pk_mul_f32 v[6:7], v[64:65], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[64:65], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[64:65], v[0:1] op_sel_hi:[0,1]
	s_add_i32 s0, s0, s1
	s_mul_i32 s1, s12, s16
	v_cvt_pk_f16_f32 v7, v6, v7
	v_cvt_pk_f16_f32 v6, v4, v5
	v_cvt_pk_f16_f32 v4, v0, v1
	s_add_i32 s0, s0, s1
	v_mul_lo_u32 v0, s12, v196
	v_pk_mul_f32 v[2:3], v[64:65], v[2:3] op_sel_hi:[0,1]
	v_add_u32_e32 v0, s0, v0
	v_lshrrev_b32_e32 v1, 2, v199
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_pk_mul_f32 v[62:63], v[64:65], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[64:65], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[64:65], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[64:65], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[64:65], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[64:65], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[64:65], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[64:65], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[64:65], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[64:65], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[64:65], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[64:65], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[64:65], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[64:65], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[64:65], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[64:65], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[64:65], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[64:65], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[64:65], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[64:65], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[64:65], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[64:65], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[64:65], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[64:65], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[64:65], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[64:65], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[64:65], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[64:65], v[8:9] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v5, v2, v3
	v_add_lshl_u32 v0, v0, v1, 1
	v_bfrev_b32_e32 v2, 1
	s_and_b32 s9, s9, 0xffff
	v_cvt_pk_f16_f32 v63, v62, v63
	v_cvt_pk_f16_f32 v62, v60, v61
	v_cvt_pk_f16_f32 v61, v58, v59
	v_cvt_pk_f16_f32 v60, v56, v57
	v_cvt_pk_f16_f32 v55, v54, v55
	v_cvt_pk_f16_f32 v54, v52, v53
	v_cvt_pk_f16_f32 v53, v50, v51
	v_cvt_pk_f16_f32 v52, v48, v49
	v_cvt_pk_f16_f32 v47, v46, v47
	v_cvt_pk_f16_f32 v46, v44, v45
	v_cvt_pk_f16_f32 v45, v42, v43
	v_cvt_pk_f16_f32 v44, v40, v41
	v_cvt_pk_f16_f32 v39, v38, v39
	v_cvt_pk_f16_f32 v38, v36, v37
	v_cvt_pk_f16_f32 v37, v34, v35
	v_cvt_pk_f16_f32 v36, v32, v33
	v_cvt_pk_f16_f32 v31, v30, v31
	v_cvt_pk_f16_f32 v30, v28, v29
	v_cvt_pk_f16_f32 v29, v26, v27
	v_cvt_pk_f16_f32 v28, v24, v25
	v_cvt_pk_f16_f32 v23, v22, v23
	v_cvt_pk_f16_f32 v22, v20, v21
	v_cvt_pk_f16_f32 v21, v18, v19
	v_cvt_pk_f16_f32 v20, v16, v17
	v_cvt_pk_f16_f32 v15, v14, v15
	v_cvt_pk_f16_f32 v14, v12, v13
	v_cvt_pk_f16_f32 v13, v10, v11
	v_cvt_pk_f16_f32 v12, v8, v9
	v_add_u32_e32 v1, 0xe0, v0
	v_add_u32_e32 v3, 0xc0, v0
	v_add_u32_e32 v8, 0xa0, v0
	v_add_u32_e32 v9, 0x80, v0
	v_add_u32_e32 v10, 0x60, v0
	v_add_u32_e32 v11, 64, v0
	v_add_u32_e32 v16, 32, v0
	v_cndmask_b32_e64 v0, v2, v0, s[4:5]
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s22, s10
	s_mov_b32 s23, s11
	v_permlane32_swap_b32_e32 v4, v6
	v_permlane32_swap_b32_e32 v5, v7
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v65, s[8:11], 0 offen
	v_cndmask_b32_e64 v1, v2, v1, s[4:5]
	v_cndmask_b32_e64 v3, v2, v3, s[4:5]
	v_cndmask_b32_e64 v8, v2, v8, s[4:5]
	v_cndmask_b32_e64 v9, v2, v9, s[4:5]
	v_cndmask_b32_e64 v10, v2, v10, s[4:5]
	v_cndmask_b32_e64 v11, v2, v11, s[4:5]
	v_cndmask_b32_e64 v16, v2, v16, s[4:5]
	v_permlane32_swap_b32_e32 v12, v14
	v_permlane32_swap_b32_e32 v13, v15
	v_permlane32_swap_b32_e32 v20, v22
	v_permlane32_swap_b32_e32 v21, v23
	v_permlane32_swap_b32_e32 v28, v30
	v_permlane32_swap_b32_e32 v29, v31
	v_permlane32_swap_b32_e32 v36, v38
	v_permlane32_swap_b32_e32 v37, v39
	v_permlane32_swap_b32_e32 v44, v46
	v_permlane32_swap_b32_e32 v45, v47
	v_permlane32_swap_b32_e32 v52, v54
	v_permlane32_swap_b32_e32 v53, v55
	v_permlane32_swap_b32_e32 v60, v62
	v_permlane32_swap_b32_e32 v61, v63
	buffer_store_dwordx4 v[4:7], v0, s[20:23], 0 offen
	buffer_store_dwordx4 v[12:15], v16, s[20:23], 0 offen
	buffer_store_dwordx4 v[20:23], v11, s[20:23], 0 offen
	buffer_store_dwordx4 v[28:31], v10, s[20:23], 0 offen
	buffer_store_dwordx4 v[36:39], v9, s[20:23], 0 offen
	buffer_store_dwordx4 v[44:47], v8, s[20:23], 0 offen
	buffer_store_dwordx4 v[52:55], v3, s[20:23], 0 offen
	buffer_store_dwordx4 v[60:63], v1, s[20:23], 0 offen
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
		.amdhsa_next_free_vgpr 249
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 252
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
	.set attn_fwd.num_vgpr, 249
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 40
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11116
; TotalNumSgprs: 46
; NumVgprs: 249
; NumAgprs: 0
; TotalNumVgprs: 249
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 249
; AccumOffset: 252
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 62
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
	.short	517                             ; DW_AT_call_line
	.byte	41                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x56:0x5d DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	706                             ; DW_AT_call_line
	.byte	61                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x63:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	284                             ; DW_AT_call_line
	.byte	69                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x70:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	351                             ; DW_AT_call_line
	.byte	69                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7d:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	332                             ; DW_AT_call_line
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
	.short	338                             ; DW_AT_call_line
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
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp235-.Lfunc_begin0
	.quad	.Ltmp236-.Lfunc_begin0
	.quad	.Ltmp237-.Lfunc_begin0
	.quad	.Ltmp238-.Lfunc_begin0
	.quad	.Ltmp239-.Lfunc_begin0
	.quad	.Ltmp240-.Lfunc_begin0
	.quad	.Ltmp241-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
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
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
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
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp195-.Lfunc_begin0
	.quad	.Ltmp196-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
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
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
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
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp227-.Lfunc_begin0
	.quad	.Ltmp228-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	.Ltmp232-.Lfunc_begin0
	.quad	.Ltmp234-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
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
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
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
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
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
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp227-.Lfunc_begin0
	.quad	.Ltmp228-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp233-.Lfunc_begin0
	.quad	.Ltmp234-.Lfunc_begin0
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
    .vgpr_count:     249
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
