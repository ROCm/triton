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
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v23, 4, v0
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s14, s34
	v_or_b32_e32 v2, 0x60, v23
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_or_b32_e32 v11, s34, v2
	s_lshl_b32 s16, s14, 6
	v_mul_lo_u32 v12, s14, v2
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v2, 3, v0
	v_or_b32_e32 v3, 0xa0, v23
	s_add_u32 s0, s2, s0
	v_and_b32_e32 v22, 0x78, v2
	s_mul_i32 s36, s15, s18
	v_or_b32_e32 v19, s34, v3
	v_mul_lo_u32 v20, s14, v3
	s_addc_u32 s1, s3, s1
	v_mad_u64_u32 v[2:3], s[2:3], s14, v23, v[22:23]
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[2:3], s[36:37], 1
	s_add_u32 s12, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s38, s20, s17
	s_addc_u32 s13, s5, s3
	s_ashr_i32 s39, s38, 31
	s_lshl_b64 s[2:3], s[38:39], 1
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
	v_or_b32_e32 v1, 32, v23
	v_or_b32_e32 v5, s34, v23
	s_addc_u32 s33, s7, s3
	s_movk_i32 s6, 0x4000
	s_and_b32 s2, s14, 0x3fff
	v_or_b32_e32 v6, s34, v1
	v_mul_lo_u32 v1, s14, v1
	v_add_u32_e32 v13, s16, v2
	s_bitset1_b32 s2, 14
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v32, 1
	v_cmp_gt_i32_e32 vcc, s6, v5
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_cndmask_b32_e32 v14, v32, v2, vcc
	v_add_lshl_u32 v1, v1, v22, 1
	v_cmp_gt_i32_e32 vcc, s6, v6
	v_or_b32_e32 v4, 0xe0, v23
	v_or_b32_e32 v10, 64, v5
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_or_b32_e32 v18, 0x80, v5
	v_or_b32_e32 v24, 0xc0, v5
	v_or_b32_e32 v25, s34, v4
	v_mul_lo_u32 v30, s14, v4
	buffer_load_dwordx4 v[2:5], v14, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v1, s[0:3], 0 offen
	v_lshlrev_b32_e32 v1, 1, v13
	v_cmp_gt_i32_e32 vcc, s6, v10
	v_add_u32_e32 v31, s16, v13
	v_add_lshl_u32 v10, v12, v22, 1
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_cmp_gt_i32_e32 vcc, s6, v11
	v_lshrrev_b32_e32 v176, 1, v0
	s_mov_b32 s30, s2
	v_cndmask_b32_e32 v21, v32, v10, vcc
	buffer_load_dwordx4 v[10:13], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v21, s[0:3], 0 offen
	v_lshlrev_b32_e32 v1, 1, v31
	v_cmp_gt_i32_e32 vcc, s6, v18
	v_add_lshl_u32 v18, v20, v22, 1
	s_mov_b32 s31, s3
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_cmp_gt_i32_e32 vcc, s6, v19
	v_and_b32_e32 v174, 31, v0
	v_lshlrev_b32_e32 v40, 8, v174
	v_cndmask_b32_e32 v33, v32, v18, vcc
	buffer_load_dwordx4 v[18:21], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[26:29], v33, s[0:3], 0 offen
	v_add_lshl_u32 v1, v31, s16, 1
	v_cmp_gt_i32_e32 vcc, s6, v24
	v_add_lshl_u32 v24, v30, v22, 1
	s_mov_b32 s14, s2
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_cmp_gt_i32_e32 vcc, s6, v25
	v_mul_lo_u32 v25, s21, v23
	v_add_lshl_u32 v82, v25, v22, 1
	v_cndmask_b32_e32 v24, v32, v24, vcc
	buffer_load_dwordx4 v[30:33], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[34:37], v24, s[0:3], 0 offen
	v_and_b32_e32 v24, 56, v176
	v_and_b32_e32 v1, 64, v176
	v_xor_b32_e32 v24, v24, v22
	s_and_b32 s0, s21, 0x3fff
	v_xor_b32_e32 v24, v24, v1
	s_bitset1_b32 s0, 14
	v_lshlrev_b32_e32 v38, 1, v24
	v_lshlrev_b32_e32 v24, 8, v23
	s_and_b32 s1, s13, 0xffff
	s_lshl_b32 s20, s0, 16
	v_add3_u32 v179, 0, v38, v24
	s_or_b32 s29, s1, s20
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v179, v[2:5]
	s_waitcnt vmcnt(6)
	ds_write_b128 v179, v[6:9] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v179, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v179, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v179, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v179, v[26:29] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v179, v[30:33] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v179, v[34:37] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[2:5], v82, s[28:31], 0 offen
	s_movk_i32 s0, 0xe0
	s_lshl_b32 s30, s21, 5
	v_bfe_u32 v6, v0, 5, 1
	v_and_b32_e32 v8, 15, v0
	v_and_or_b32 v7, v176, s0, v174
	v_xor_b32_e32 v9, v6, v8
	v_or_b32_e32 v10, 2, v6
	v_or_b32_e32 v11, 4, v6
	v_or_b32_e32 v12, 6, v6
	v_or_b32_e32 v13, 8, v6
	v_or_b32_e32 v14, 10, v6
	v_or_b32_e32 v15, 12, v6
	v_or_b32_e32 v6, 14, v6
	s_ashr_i32 s31, s30, 31
	s_lshl_b32 s6, s24, 5
	v_xor_b32_e32 v10, v10, v8
	v_xor_b32_e32 v11, v11, v8
	v_xor_b32_e32 v12, v12, v8
	v_xor_b32_e32 v13, v13, v8
	v_xor_b32_e32 v14, v14, v8
	v_xor_b32_e32 v15, v15, v8
	v_xor_b32_e32 v6, v6, v8
	v_lshl_add_u32 v7, v7, 8, 0
	v_lshlrev_b32_e32 v8, 4, v9
	s_lshl_b64 s[22:23], s[30:31], 1
	v_add_u32_e32 v9, v7, v8
	v_lshlrev_b32_e32 v10, 4, v10
	v_lshlrev_b32_e32 v25, 4, v11
	s_add_u32 s0, s28, s22
	v_add_u32_e32 v16, v7, v10
	ds_read_b128 v[126:129], v9
	ds_read_b128 v[122:125], v16
	v_add_u32_e32 v9, v7, v25
	v_lshlrev_b32_e32 v34, 4, v12
	v_lshlrev_b32_e32 v35, 4, v13
	s_addc_u32 s7, s13, s23
	v_add_u32_e32 v11, v7, v34
	ds_read_b128 v[118:121], v9
	ds_read_b128 v[114:117], v11
	v_add_u32_e32 v9, v7, v35
	v_lshlrev_b32_e32 v36, 4, v14
	v_lshlrev_b32_e32 v38, 4, v15
	s_and_b32 s1, s7, 0xffff
	v_add_u32_e32 v11, v7, v36
	ds_read_b128 v[110:113], v9
	ds_read_b128 v[106:109], v11
	v_add_u32_e32 v9, v7, v38
	v_lshlrev_b32_e32 v39, 4, v6
	s_or_b32 s1, s1, s20
	v_add3_u32 v180, 0, v8, v40
	v_add_u32_e32 v6, v7, v39
	ds_read_b128 v[102:105], v9
	ds_read_b128 v[98:101], v6
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[18:21], v82, s[0:3], 0 offen
	v_add3_u32 v184, 0, v10, v40
	v_add3_u32 v182, 0, v25, v40
	v_add3_u32 v181, 0, v34, v40
	v_add3_u32 v192, 0, v35, v40
	s_and_b32 s1, s24, 0x3fff
	s_bitset1_b32 s1, 14
	v_mul_lo_u32 v23, s24, v23
	s_and_b32 s13, s33, 0xffff
	s_lshl_b32 s19, s1, 16
	s_mov_b32 s15, s3
	v_add_lshl_u32 v199, v23, v22, 1
	s_or_b32 s13, s13, s19
	v_add3_u32 v200, 0, v36, v40
	v_lshlrev_b32_e32 v25, 4, v0
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_lshlrev_b32_e32 v23, 2, v0
	v_and_b32_e32 v205, 0x200, v25
	s_movk_i32 s1, 0x2000
	v_xor_b32_e32 v175, 0x80, v23
	v_lshlrev_b32_e32 v23, 1, v174
	v_lshl_add_u32 v25, v205, 1, 0
	s_add_u32 s0, s0, s22
	v_add3_u32 v83, v25, v23, s1
	s_addc_u32 s1, s7, s23
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	v_lshlrev_b32_e32 v22, 1, v22
	s_waitcnt vmcnt(1)
	ds_write_b128 v179, v[2:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[26:29], v180
	ds_read_b128 v[30:33], v184
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[126:127], 0
	buffer_load_dwordx4 v[34:37], v199, s[12:15], 0 offen
	s_mul_i32 s15, s21, 0xc0
	s_add_u32 s21, s12, s6
	s_addc_u32 s29, s33, s7
	s_and_b32 s1, s1, 0xffff
	v_add3_u32 v201, 0, v38, v40
	v_add3_u32 v202, 0, v39, v40
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[128:129], v[2:17]
	ds_read_b128 v[26:29], v182
	s_or_b32 s1, s1, s20
	s_and_b32 s12, s29, 0xffff
	v_add3_u32 v177, 0, v22, v24
	ds_read_b128 v[22:25], v201
	s_mov_b32 s14, 0xff800000
	s_mov_b32 s16, 0x3e0293ee
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[122:123], v[2:17]
	s_mul_hi_i32 s30, s30, 6
	s_movk_i32 s24, 0xffe0
	s_mov_b32 s28, 0x5040100
	v_mov_b32_e32 v178, 1.0
	v_mfma_f32_32x32x8_f16 v[2:17], v[32:33], v[124:125], v[2:17]
	ds_read_b128 v[30:33], v181
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[118:119], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[120:121], v[2:17]
	ds_read_b128 v[26:29], v192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[114:115], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[32:33], v[116:117], v[2:17]
	ds_read_b128 v[30:33], v200
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[110:111], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[112:113], v[2:17]
	ds_read_b128 v[26:29], v202
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[134:137], v82, s[0:3], 0 offen
	s_or_b32 s1, s12, s19
	s_mov_b32 s0, s21
	s_waitcnt vmcnt(2)
	ds_write_b128 v179, v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[130:133], v199, s[0:3], 0 offen
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[106:107], v[2:17]
	s_waitcnt vmcnt(2)
	ds_write_b128 v177, v[34:37] offset:8192
	ds_read_b128 v[166:169], v180
	ds_read_b128 v[162:165], v184
	ds_read_b128 v[158:161], v182
	ds_read_b128 v[154:157], v181
	ds_read_b128 v[150:153], v192
	ds_read_b128 v[146:149], v200
	ds_read_b128 v[142:145], v201
	ds_read_b128 v[138:141], v202
	s_add_u32 s12, s36, s38
	s_addc_u32 s13, s37, s39
	s_lshl_b64 s[0:1], s[12:13], 1
	s_add_u32 s0, s15, s0
	s_addc_u32 s1, s30, s1
	v_mfma_f32_32x32x8_f16 v[2:17], v[32:33], v[108:109], v[2:17]
	s_add_u32 s4, s4, s0
	s_addc_u32 s5, s5, s1
	v_mfma_f32_32x32x8_f16 v[2:17], v[22:23], v[102:103], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[24:25], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[100:101], v[2:17]
	s_nop 7
	s_nop 2
	v_max_f32_e32 v18, v3, v3
	v_max_f32_e32 v19, v2, v2
	v_max_f32_e32 v18, v19, v18
	v_max3_f32 v18, v18, v4, v5
	v_max3_f32 v18, v18, v6, v7
	v_max3_f32 v18, v18, v8, v9
	v_max3_f32 v18, v18, v10, v11
	v_max3_f32 v18, v18, v12, v13
	v_max3_f32 v18, v18, v14, v15
	v_max3_f32 v18, v18, v16, v17
	ds_bpermute_b32 v19, v175, v18
	v_mov_b32_e32 v172, v17
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v173, v18, v19, s14
	v_pk_mul_f32 v[18:19], v[172:173], s[16:17] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v2, v2, s16, -v19
	v_fma_f32 v3, v3, s16, -v19
	v_fma_f32 v4, v4, s16, -v19
	v_fma_f32 v5, v5, s16, -v19
	v_fma_f32 v6, v6, s16, -v19
	v_fma_f32 v7, v7, s16, -v19
	v_fma_f32 v8, v8, s16, -v19
	v_fma_f32 v9, v9, s16, -v19
	v_fma_f32 v10, v10, s16, -v19
	v_fma_f32 v11, v11, s16, -v19
	v_fma_f32 v12, v12, s16, -v19
	v_fma_f32 v13, v13, s16, -v19
	v_fma_f32 v14, v14, s16, -v19
	v_fma_f32 v15, v15, s16, -v19
	v_fma_f32 v16, v16, s16, -v19
	v_sub_f32_e32 v17, v18, v19
	v_sub_f32_e32 v64, 0xff800000, v19
	v_exp_f32_e32 v203, v2
	v_exp_f32_e32 v204, v3
	v_exp_f32_e32 v198, v4
	v_exp_f32_e32 v197, v5
	v_exp_f32_e32 v196, v6
	v_exp_f32_e32 v195, v7
	v_exp_f32_e32 v194, v8
	v_exp_f32_e32 v193, v9
	v_exp_f32_e32 v191, v10
	v_exp_f32_e32 v190, v11
	v_exp_f32_e32 v189, v12
	v_exp_f32_e32 v188, v13
	v_exp_f32_e32 v187, v14
	v_exp_f32_e32 v186, v15
	v_exp_f32_e32 v185, v16
	v_exp_f32_e32 v183, v17
	v_exp_f32_e32 v170, v64
	v_mov_b32_e32 v2, 0
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
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_mfma_f32_32x32x8_f16 v[66:81], v[166:167], v[126:127], 0
	v_cvt_f16_f32_e32 v84, v203
	v_cvt_f16_f32_e32 v86, v204
	v_cvt_f16_f32_e32 v85, v198
	v_cvt_f16_f32_e32 v87, v197
	v_pk_mul_f32 v[50:51], v[50:51], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[170:171] op_sel_hi:[1,0]
	s_barrier
	ds_read_u16 v171, v83
	ds_read_u16 v172, v83 offset:64
	ds_read_u16 v206, v83 offset:128
	ds_read_u16 v207, v83 offset:512
	ds_read_u16 v208, v83 offset:576
	ds_read_u16 v209, v83 offset:640
	ds_read_u16 v210, v83 offset:2048
	ds_read_u16 v211, v83 offset:2112
	ds_read_u16 v212, v83 offset:2176
	ds_read_u16 v213, v83 offset:2560
	ds_read_u16 v214, v83 offset:2624
	ds_read_u16 v215, v83 offset:2688
	ds_read_u16 v216, v83 offset:4096
	ds_read_u16 v217, v83 offset:4160
	ds_read_u16 v218, v83 offset:4224
	ds_read_u16 v219, v83 offset:4608
	ds_read_u16 v220, v83 offset:4672
	ds_read_u16 v221, v83 offset:4736
	ds_read_u16 v222, v83 offset:6144
	ds_read_u16 v223, v83 offset:6208
	ds_read_u16 v224, v83 offset:6272
	ds_read_u16 v225, v83 offset:6656
	ds_read_u16 v226, v83 offset:6720
	ds_read_u16 v227, v83 offset:6784
	ds_read_u16 v228, v83 offset:320
	ds_read_u16 v229, v83 offset:384
	ds_read_u16 v230, v83 offset:448
	ds_read_u16 v231, v83 offset:256
	ds_read_u16 v232, v83 offset:192
	ds_read_u16 v233, v83 offset:832
	ds_read_u16 v234, v83 offset:896
	ds_read_u16 v235, v83 offset:960
	ds_read_u16 v236, v83 offset:768
	ds_read_u16 v237, v83 offset:704
	ds_read_u16 v238, v83 offset:2368
	ds_read_u16 v239, v83 offset:2432
	ds_read_u16 v240, v83 offset:2496
	v_pack_b32_f16 v85, v85, v87
	v_pack_b32_f16 v84, v84, v86
	s_waitcnt lgkmcnt(4)
	v_perm_b32 v87, v236, v207, s28
	v_perm_b32 v86, v231, v171, s28
	v_mfma_f32_32x32x8_f16 v[66:81], v[168:169], v[128:129], v[66:81]
	ds_read_u16 v168, v83 offset:2304
	ds_read_u16 v171, v83 offset:2880
	ds_read_u16 v169, v83 offset:2240
	v_cvt_f16_f32_e32 v88, v196
	v_cvt_f16_f32_e32 v89, v195
	v_cvt_f16_f32_e32 v90, v194
	v_cvt_f16_f32_e32 v91, v193
	v_mfma_f32_32x32x8_f16 v[50:65], v[86:87], v[84:85], v[50:65]
	v_perm_b32 v87, v233, v208, s28
	v_perm_b32 v86, v228, v172, s28
	v_cvt_f16_f32_e32 v92, v191
	v_cvt_f16_f32_e32 v93, v190
	v_cvt_f16_f32_e32 v94, v189
	v_cvt_f16_f32_e32 v95, v188
	s_add_u32 s12, s21, s6
	v_mfma_f32_32x32x8_f16 v[34:49], v[86:87], v[84:85], v[34:49]
	v_perm_b32 v87, v234, v209, s28
	v_perm_b32 v86, v229, v206, s28
	s_addc_u32 s30, s29, s7
	s_and_b32 s1, s5, 0xffff
	s_and_b32 s13, s30, 0xffff
	s_mov_b32 s0, s4
	s_mov_b32 s14, s2
	v_mfma_f32_32x32x8_f16 v[66:81], v[162:163], v[122:123], v[66:81]
	s_mov_b32 s15, s3
	s_or_b32 s1, s1, s20
	s_or_b32 s13, s13, s19
	v_cvt_f16_f32_e32 v96, v187
	v_cvt_f16_f32_e32 v97, v186
	v_cvt_f16_f32_e32 v166, v185
	v_cvt_f16_f32_e32 v167, v183
	v_mfma_f32_32x32x8_f16 v[18:33], v[86:87], v[84:85], v[18:33]
	s_waitcnt lgkmcnt(6)
	v_perm_b32 v87, v235, v237, s28
	v_perm_b32 v86, v230, v232, s28
	s_add_u32 s21, s21, s6
	s_addc_u32 s29, s29, s7
	s_add_u32 s4, s4, s22
	s_addc_u32 s5, s5, s23
	s_add_i32 s24, s24, 32
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[84:85], v[2:17]
	ds_read_u16 v172, v83 offset:2944
	ds_read_u16 v206, v83 offset:3008
	ds_read_u16 v86, v83 offset:2816
	ds_read_u16 v207, v83 offset:2752
	ds_read_u16 v208, v83 offset:4416
	ds_read_u16 v209, v83 offset:4480
	v_pack_b32_f16 v85, v90, v91
	s_waitcnt lgkmcnt(3)
	v_perm_b32 v87, v86, v213, s28
	v_perm_b32 v86, v168, v210, s28
	v_pack_b32_f16 v84, v88, v89
	ds_read_u16 v88, v83 offset:4544
	ds_read_u16 v89, v83 offset:4352
	ds_read_u16 v90, v83 offset:4288
	v_mfma_f32_32x32x8_f16 v[66:81], v[164:165], v[124:125], v[66:81]
	s_cmpk_lt_u32 s24, 0x1f80
	v_mfma_f32_32x32x8_f16 v[50:65], v[86:87], v[84:85], v[50:65]
	v_perm_b32 v87, v171, v214, s28
	v_perm_b32 v86, v238, v211, s28
	v_mfma_f32_32x32x8_f16 v[66:81], v[158:159], v[118:119], v[66:81]
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[34:49], v[86:87], v[84:85], v[34:49]
	v_perm_b32 v87, v172, v215, s28
	v_perm_b32 v86, v239, v212, s28
	v_mov_b32_e32 v172, v173
	v_mfma_f32_32x32x8_f16 v[66:81], v[160:161], v[120:121], v[66:81]
	v_mfma_f32_32x32x8_f16 v[18:33], v[86:87], v[84:85], v[18:33]
	s_waitcnt lgkmcnt(5)
	v_perm_b32 v87, v206, v207, s28
	v_perm_b32 v86, v240, v169, s28
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[84:85], v[2:17]
	ds_read_u16 v91, v83 offset:4928
	ds_read_u16 v162, v83 offset:4992
	ds_read_u16 v163, v83 offset:5056
	ds_read_u16 v86, v83 offset:4864
	ds_read_u16 v168, v83 offset:4800
	ds_read_u16 v169, v83 offset:6464
	v_pack_b32_f16 v85, v94, v95
	v_pack_b32_f16 v84, v92, v93
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v87, v86, v219, s28
	v_perm_b32 v86, v89, v216, s28
	ds_read_u16 v89, v83 offset:6528
	v_mfma_f32_32x32x8_f16 v[66:81], v[154:155], v[114:115], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[86:87], v[84:85], v[50:65]
	v_perm_b32 v87, v91, v220, s28
	ds_read_u16 v91, v83 offset:6592
	v_perm_b32 v86, v208, v217, s28
	ds_read_u16 v92, v83 offset:6400
	v_mfma_f32_32x32x8_f16 v[66:81], v[156:157], v[116:117], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[86:87], v[84:85], v[34:49]
	v_perm_b32 v87, v162, v221, s28
	v_perm_b32 v86, v209, v218, s28
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[86:87], v[84:85], v[18:33]
	s_waitcnt lgkmcnt(4)
	v_perm_b32 v87, v163, v168, s28
	v_perm_b32 v86, v88, v90, s28
	v_mfma_f32_32x32x8_f16 v[66:81], v[150:151], v[110:111], v[66:81]
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[84:85], v[2:17]
	ds_read_u16 v88, v83 offset:6336
	ds_read_u16 v90, v83 offset:6976
	ds_read_u16 v93, v83 offset:7040
	ds_read_u16 v94, v83 offset:7104
	ds_read_u16 v86, v83 offset:6912
	ds_read_u16 v95, v83 offset:6848
	s_waitcnt vmcnt(1)
	ds_write_b128 v179, v[134:137]
	buffer_load_dwordx4 v[134:137], v82, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b128 v177, v[130:133] offset:8192
	buffer_load_dwordx4 v[130:133], v199, s[12:15], 0 offen
	v_mfma_f32_32x32x8_f16 v[66:81], v[152:153], v[112:113], v[66:81]
	v_perm_b32 v87, v86, v225, s28
	v_perm_b32 v86, v92, v222, s28
	v_pack_b32_f16 v85, v166, v167
	v_pack_b32_f16 v84, v96, v97
	v_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[106:107], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[148:149], v[108:109], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[86:87], v[84:85], v[50:65]
	v_perm_b32 v87, v90, v226, s28
	v_perm_b32 v86, v169, v223, s28
	ds_read_b128 v[166:169], v180
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[86:87], v[84:85], v[34:49]
	v_perm_b32 v87, v93, v227, s28
	v_perm_b32 v86, v89, v224, s28
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[86:87], v[84:85], v[18:33]
	v_perm_b32 v87, v94, v95, s28
	v_perm_b32 v86, v91, v88, s28
	v_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[104:105], v[66:81]
	ds_read_b128 v[162:165], v184
	ds_read_b128 v[158:161], v182
	ds_read_b128 v[154:157], v181
	ds_read_b128 v[150:153], v192
	ds_read_b128 v[146:149], v200
	ds_read_b128 v[142:145], v201
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[84:85], v[2:17]
	v_add_f32_e32 v85, v203, v204
	v_add_f32_e32 v85, v85, v198
	v_add_f32_e32 v85, v85, v197
	v_add_f32_e32 v85, v85, v196
	v_add_f32_e32 v85, v85, v195
	v_add_f32_e32 v85, v85, v194
	v_add_f32_e32 v85, v85, v193
	v_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[98:99], v[66:81]
	v_add_f32_e32 v85, v85, v191
	v_add_f32_e32 v85, v85, v190
	v_add_f32_e32 v85, v85, v189
	v_add_f32_e32 v85, v85, v188
	v_add_f32_e32 v85, v85, v187
	v_add_f32_e32 v85, v85, v186
	v_add_f32_e32 v85, v85, v185
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[100:101], v[66:81]
	v_add_f32_e32 v85, v85, v183
	ds_bpermute_b32 v86, v175, v85
	v_mov_b32_e32 v84, v178
	ds_read_b128 v[138:141], v202
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v178, v85, v86
	v_fmac_f32_e32 v178, v84, v170
	s_nop 3
	v_max_f32_e32 v84, v67, v67
	v_max_f32_e32 v85, v66, v66
	v_max_f32_e32 v84, v85, v84
	v_max3_f32 v84, v84, v68, v69
	v_max3_f32 v84, v84, v70, v71
	v_max3_f32 v84, v84, v72, v73
	v_max3_f32 v84, v84, v74, v75
	v_max3_f32 v84, v84, v76, v77
	v_max3_f32 v84, v84, v78, v79
	v_max3_f32 v84, v84, v80, v81
	ds_bpermute_b32 v85, v175, v84
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v173, v172, v84, v85
	v_pk_mul_f32 v[170:171], v[172:173], s[16:17] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v66, v66, s16, -v171
	v_fma_f32 v67, v67, s16, -v171
	v_fma_f32 v68, v68, s16, -v171
	v_fma_f32 v69, v69, s16, -v171
	v_fma_f32 v70, v70, s16, -v171
	v_fma_f32 v71, v71, s16, -v171
	v_fma_f32 v72, v72, s16, -v171
	v_fma_f32 v73, v73, s16, -v171
	v_fma_f32 v74, v74, s16, -v171
	v_fma_f32 v75, v75, s16, -v171
	v_fma_f32 v76, v76, s16, -v171
	v_fma_f32 v77, v77, s16, -v171
	v_fma_f32 v78, v78, s16, -v171
	v_fma_f32 v79, v79, s16, -v171
	v_fma_f32 v80, v80, s16, -v171
	v_fma_f32 v81, v81, s16, -v171
	v_sub_f32_e32 v84, v170, v171
	v_exp_f32_e32 v203, v66
	v_exp_f32_e32 v204, v67
	v_exp_f32_e32 v198, v68
	v_exp_f32_e32 v197, v69
	v_exp_f32_e32 v196, v70
	v_exp_f32_e32 v195, v71
	v_exp_f32_e32 v194, v72
	v_exp_f32_e32 v193, v73
	v_exp_f32_e32 v191, v74
	v_exp_f32_e32 v190, v75
	v_exp_f32_e32 v189, v76
	v_exp_f32_e32 v188, v77
	v_exp_f32_e32 v187, v78
	v_exp_f32_e32 v186, v79
	v_exp_f32_e32 v185, v80
	v_exp_f32_e32 v183, v81
	v_exp_f32_e32 v170, v84
	s_cbranch_scc1 .LBB0_1
; %bb.2:
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
	s_barrier
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[126:127], v[66:81]
	s_mov_b32 s5, 0x5040100
	v_pk_mul_f32 v[64:65], v[64:65], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[170:171] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[128:129], v[82:97]
	v_pk_mul_f32 v[54:55], v[54:55], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[170:171] op_sel_hi:[1,0]
	v_cvt_f16_f32_e32 v166, v194
	v_cvt_f16_f32_e32 v167, v193
	v_cvt_f16_f32_e32 v168, v191
	v_cvt_f16_f32_e32 v169, v190
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[122:123], v[82:97]
	v_cvt_f16_f32_e32 v162, v203
	v_cvt_f16_f32_e32 v163, v204
	v_cvt_f16_f32_e32 v172, v187
	v_cvt_f16_f32_e32 v206, v186
	v_cvt_f16_f32_e32 v207, v185
	v_cvt_f16_f32_e32 v208, v183
	v_pk_mul_f32 v[48:49], v[48:49], v[170:171] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[124:125], v[82:97]
	v_cvt_f16_f32_e32 v164, v198
	v_cvt_f16_f32_e32 v165, v195
	v_pk_mul_f32 v[46:47], v[46:47], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[170:171] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[158:159], v[118:119], v[82:97]
	v_cvt_f16_f32_e32 v158, v197
	v_cvt_f16_f32_e32 v159, v196
	v_pk_mul_f32 v[36:37], v[36:37], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[170:171] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[160:161], v[120:121], v[82:97]
	v_cvt_f16_f32_e32 v160, v189
	v_cvt_f16_f32_e32 v161, v188
	v_pk_mul_f32 v[26:27], v[26:27], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[170:171] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[154:155], v[114:115], v[82:97]
	v_or_b32_e32 v154, v205, v174
	v_lshl_add_u32 v154, v154, 1, 0
	ds_read_u16 v155, v154 offset:8192
	ds_read_u16 v205, v154 offset:8256
	ds_read_u16 v209, v154 offset:8320
	v_pk_mul_f32 v[16:17], v[16:17], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[170:171] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[156:157], v[116:117], v[82:97]
	ds_read_u16 v156, v154 offset:8704
	ds_read_u16 v157, v154 offset:8768
	ds_read_u16 v210, v154 offset:8832
	ds_read_u16 v211, v154 offset:10240
	ds_read_u16 v212, v154 offset:10304
	ds_read_u16 v213, v154 offset:10368
	ds_read_u16 v214, v154 offset:10752
	ds_read_u16 v215, v154 offset:10816
	ds_read_u16 v216, v154 offset:10880
	v_pk_mul_f32 v[8:9], v[8:9], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[170:171] op_sel_hi:[1,0]
	s_mul_i32 s0, s18, 0xc0000
	s_ashr_i32 s1, s0, 31
	v_mfma_f32_32x32x8_f16 v[82:97], v[150:151], v[110:111], v[82:97]
	ds_read_u16 v150, v154 offset:12288
	ds_read_u16 v151, v154 offset:12352
	ds_read_u16 v217, v154 offset:12416
	ds_read_u16 v218, v154 offset:12800
	ds_read_u16 v219, v154 offset:12864
	ds_read_u16 v220, v154 offset:12928
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s2, s8, s0
	s_addc_u32 s3, s9, s1
	s_lshl_b32 s0, s17, 14
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	v_mfma_f32_32x32x8_f16 v[82:97], v[152:153], v[112:113], v[82:97]
	ds_read_u16 v152, v154 offset:14336
	ds_read_u16 v153, v154 offset:14400
	ds_read_u16 v221, v154 offset:14464
	ds_read_u16 v222, v154 offset:14848
	ds_read_u16 v223, v154 offset:14912
	ds_read_u16 v224, v154 offset:14976
	s_add_u32 s2, s2, s0
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[0:1], s[34:35], 2
	s_add_u32 s4, s2, s0
	s_addc_u32 s13, s3, s1
	v_mfma_f32_32x32x8_f16 v[82:97], v[146:147], v[106:107], v[82:97]
	ds_read_u16 v225, v154 offset:8576
	ds_read_u16 v226, v154 offset:8640
	ds_read_u16 v227, v154 offset:8512
	ds_read_u16 v146, v154 offset:8448
	ds_read_u16 v228, v154 offset:8384
	s_add_i32 s8, s34, 0xffffc100
	s_add_u32 s0, s12, s6
	s_mov_b32 s6, 0x3e0293ee
	s_addc_u32 s1, s30, s7
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s1, s19
	v_mfma_f32_32x32x8_f16 v[82:97], v[148:149], v[108:109], v[82:97]
	ds_read_u16 v229, v154 offset:9088
	ds_read_u16 v230, v154 offset:9152
	ds_read_u16 v148, v154 offset:9024
	ds_read_u16 v147, v154 offset:8960
	ds_read_u16 v231, v154 offset:8896
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_cmp_lt_i32 s8, 1
	v_mfma_f32_32x32x8_f16 v[82:97], v[142:143], v[102:103], v[82:97]
	ds_read_u16 v232, v154 offset:10624
	ds_read_u16 v233, v154 offset:10688
	ds_read_u16 v234, v154 offset:10560
	ds_read_u16 v142, v154 offset:10496
	ds_read_u16 v235, v154 offset:10432
	ds_read_u16 v236, v154 offset:11136
	ds_read_u16 v237, v154 offset:11200
	ds_read_u16 v238, v154 offset:11072
	ds_read_u16 v143, v154 offset:11008
	ds_read_u16 v239, v154 offset:10944
	ds_read_u16 v240, v154 offset:12672
	ds_read_u16 v241, v154 offset:12736
	ds_read_u16 v242, v154 offset:12608
	ds_read_u16 v149, v154 offset:12544
	ds_read_u16 v243, v154 offset:12480
	s_waitcnt lgkmcnt(6)
	v_perm_b32 v143, v143, v214, s5
	v_perm_b32 v142, v142, v211, s5
	ds_read_u16 v244, v154 offset:13184
	ds_read_u16 v245, v154 offset:13248
	v_mfma_f32_32x32x8_f16 v[82:97], v[144:145], v[104:105], v[82:97]
	v_pack_b32_f16 v145, v166, v167
	v_pack_b32_f16 v144, v159, v165
	v_mfma_f32_32x32x8_f16 v[82:97], v[138:139], v[98:99], v[82:97]
	v_perm_b32 v139, v147, v156, s5
	v_perm_b32 v138, v146, v155, s5
	ds_read_u16 v155, v154 offset:13120
	ds_read_u16 v146, v154 offset:13056
	ds_read_u16 v156, v154 offset:12992
	v_mfma_f32_32x32x8_f16 v[82:97], v[140:141], v[100:101], v[82:97]
	v_pack_b32_f16 v141, v164, v158
	v_pack_b32_f16 v140, v162, v163
	ds_read_u16 v158, v154 offset:14720
	ds_read_u16 v162, v154 offset:14784
	ds_read_u16 v163, v154 offset:14656
	ds_read_u16 v164, v154 offset:14592
	ds_read_u16 v211, v154 offset:14528
	ds_read_u16 v159, v154 offset:15232
	ds_read_u16 v165, v154 offset:15296
	v_mfma_f32_32x32x8_f16 v[50:65], v[138:139], v[140:141], v[50:65]
	s_waitcnt lgkmcnt(8)
	v_perm_b32 v139, v146, v218, s5
	v_perm_b32 v138, v149, v150, s5
	ds_read_u16 v150, v154 offset:15168
	ds_read_u16 v146, v154 offset:15104
	ds_read_u16 v166, v154 offset:15040
	v_perm_b32 v149, v148, v157, s5
	v_perm_b32 v148, v227, v205, s5
	s_waitcnt vmcnt(1)
	ds_write_b128 v179, v[134:137]
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v147, v146, v222, s5
	v_mfma_f32_32x32x8_f16 v[50:65], v[142:143], v[144:145], v[50:65]
	v_pack_b32_f16 v143, v160, v161
	v_pack_b32_f16 v142, v168, v169
	v_perm_b32 v146, v164, v152, s5
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v136, v97
	v_mfma_f32_32x32x8_f16 v[50:65], v[138:139], v[142:143], v[50:65]
	v_pack_b32_f16 v139, v207, v208
	v_pack_b32_f16 v138, v172, v206
	v_mfma_f32_32x32x8_f16 v[34:49], v[148:149], v[140:141], v[34:49]
	v_perm_b32 v149, v229, v210, s5
	v_perm_b32 v148, v225, v209, s5
	v_mfma_f32_32x32x8_f16 v[50:65], v[146:147], v[138:139], v[50:65]
	v_perm_b32 v147, v238, v215, s5
	v_perm_b32 v146, v234, v212, s5
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[34:49], v[146:147], v[144:145], v[34:49]
	v_perm_b32 v147, v155, v219, s5
	v_perm_b32 v146, v242, v151, s5
	v_perm_b32 v151, v230, v231, s5
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[34:49], v[146:147], v[142:143], v[34:49]
	v_perm_b32 v147, v150, v223, s5
	v_perm_b32 v146, v163, v153, s5
	v_perm_b32 v150, v226, v228, s5
	v_mfma_f32_32x32x8_f16 v[18:33], v[148:149], v[140:141], v[18:33]
	v_perm_b32 v149, v159, v224, s5
	v_perm_b32 v148, v158, v221, s5
	v_mfma_f32_32x32x8_f16 v[34:49], v[146:147], v[138:139], v[34:49]
	v_perm_b32 v147, v236, v216, s5
	v_perm_b32 v146, v232, v213, s5
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[146:147], v[144:145], v[18:33]
	v_perm_b32 v147, v244, v220, s5
	v_perm_b32 v146, v240, v217, s5
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[146:147], v[142:143], v[18:33]
	v_perm_b32 v147, v237, v239, s5
	v_perm_b32 v146, v233, v235, s5
	v_mfma_f32_32x32x8_f16 v[2:17], v[150:151], v[140:141], v[2:17]
	v_perm_b32 v141, v245, v156, s5
	v_mfma_f32_32x32x8_f16 v[18:33], v[148:149], v[138:139], v[18:33]
	v_max_f32_e32 v148, v83, v83
	v_max_f32_e32 v149, v82, v82
	v_max_f32_e32 v148, v149, v148
	v_max3_f32 v140, v148, v84, v85
	v_max3_f32 v140, v140, v86, v87
	v_max3_f32 v140, v140, v88, v89
	v_max3_f32 v140, v140, v90, v91
	v_mfma_f32_32x32x8_f16 v[2:17], v[146:147], v[144:145], v[2:17]
	v_max3_f32 v140, v140, v92, v93
	v_max3_f32 v140, v140, v94, v95
	v_max3_f32 v148, v140, v96, v97
	v_perm_b32 v140, v241, v243, s5
	ds_bpermute_b32 v146, v175, v148
	v_perm_b32 v145, v165, v166, s5
	v_perm_b32 v144, v162, v211, s5
	v_mfma_f32_32x32x8_f16 v[2:17], v[140:141], v[142:143], v[2:17]
	ds_read_b128 v[140:143], v180
	s_waitcnt lgkmcnt(1)
	v_max3_f32 v137, v173, v148, v146
	v_pk_mul_f32 v[134:135], v[136:137], s[6:7] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v97, v82, s6, -v135
	v_fma_f32 v136, v83, s6, -v135
	v_fma_f32 v93, v93, s6, -v135
	v_mfma_f32_32x32x8_f16 v[2:17], v[144:145], v[138:139], v[2:17]
	v_fma_f32 v138, v84, s6, -v135
	v_fma_f32 v139, v85, s6, -v135
	ds_read_b128 v[82:85], v184
	v_fma_f32 v144, v92, s6, -v135
	v_fma_f32 v145, v94, s6, -v135
	v_fma_f32 v146, v95, s6, -v135
	v_fma_f32 v147, v96, s6, -v135
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[126:127], v[66:81]
	v_exp_f32_e32 v94, v144
	v_exp_f32_e32 v95, v93
	v_exp_f32_e32 v96, v145
	v_exp_f32_e32 v93, v147
	v_fma_f32 v86, v86, s6, -v135
	v_fma_f32 v87, v87, s6, -v135
	v_fma_f32 v88, v88, s6, -v135
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[128:129], v[66:81]
	ds_read_b128 v[140:143], v182
	v_exp_f32_e32 v128, v97
	v_exp_f32_e32 v97, v146
	ds_read_b128 v[144:147], v181
	v_fma_f32 v89, v89, s6, -v135
	v_fma_f32 v91, v91, s6, -v135
	v_sub_f32_e32 v148, v134, v135
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[66:81], v[82:83], v[122:123], v[66:81]
	v_exp_f32_e32 v134, v136
	v_exp_f32_e32 v126, v138
	v_exp_f32_e32 v136, v86
	v_exp_f32_e32 v138, v87
	v_exp_f32_e32 v129, v88
	v_exp_f32_e32 v122, v89
	v_exp_f32_e32 v92, v91
	v_mfma_f32_32x32x8_f16 v[66:81], v[84:85], v[124:125], v[66:81]
	v_exp_f32_e32 v91, v148
	ds_read_b128 v[148:151], v192
	ds_read_b128 v[156:159], v200
	ds_read_b128 v[160:163], v201
	ds_read_b128 v[86:89], v202
	v_fma_f32 v90, v90, s6, -v135
	v_exp_f32_e32 v127, v139
	v_sub_f32_e32 v82, v171, v135
	v_exp_f32_e32 v123, v90
	v_exp_f32_e32 v90, v82
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[118:119], v[66:81]
	s_waitcnt vmcnt(0)
	ds_write_b128 v177, v[130:133] offset:8192
	buffer_load_dwordx4 v[82:85], v199, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cvt_f16_f32_e32 v118, v128
	v_cvt_f16_f32_e32 v119, v134
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[120:121], v[66:81]
	v_cvt_f16_f32_e32 v120, v126
	v_cvt_f16_f32_e32 v121, v127
	v_pk_mul_f32 v[64:65], v[64:65], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[90:91] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[114:115], v[66:81]
	v_pk_mul_f32 v[54:55], v[54:55], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[90:91] op_sel_hi:[1,0]
	v_cvt_f16_f32_e32 v114, v136
	v_cvt_f16_f32_e32 v115, v138
	v_cvt_f16_f32_e32 v124, v129
	v_cvt_f16_f32_e32 v125, v122
	v_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[116:117], v[66:81]
	v_cvt_f16_f32_e32 v130, v123
	v_cvt_f16_f32_e32 v131, v92
	v_cvt_f16_f32_e32 v132, v94
	v_cvt_f16_f32_e32 v116, v95
	v_cvt_f16_f32_e32 v117, v96
	v_cvt_f16_f32_e32 v133, v97
	v_cvt_f16_f32_e32 v139, v93
	v_mfma_f32_32x32x8_f16 v[66:81], v[148:149], v[110:111], v[66:81]
	ds_read_u16 v110, v154 offset:8192
	ds_read_u16 v111, v154 offset:8256
	ds_read_u16 v141, v154 offset:8320
	ds_read_u16 v142, v154 offset:8704
	ds_read_u16 v143, v154 offset:8768
	ds_read_u16 v144, v154 offset:8832
	v_cvt_f16_f32_e32 v140, v91
	v_pk_mul_f32 v[48:49], v[48:49], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[90:91] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[150:151], v[112:113], v[66:81]
	ds_read_u16 v112, v154 offset:10240
	ds_read_u16 v113, v154 offset:10304
	ds_read_u16 v145, v154 offset:10368
	ds_read_u16 v146, v154 offset:10752
	ds_read_u16 v147, v154 offset:10816
	ds_read_u16 v148, v154 offset:10880
	ds_read_u16 v149, v154 offset:12288
	ds_read_u16 v150, v154 offset:12352
	ds_read_u16 v151, v154 offset:12416
	v_pk_mul_f32 v[38:39], v[38:39], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[90:91] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[156:157], v[106:107], v[66:81]
	ds_read_u16 v106, v154 offset:12800
	ds_read_u16 v107, v154 offset:12864
	ds_read_u16 v152, v154 offset:12928
	ds_read_u16 v153, v154 offset:14336
	ds_read_u16 v155, v154 offset:14400
	ds_read_u16 v156, v154 offset:14464
	ds_read_u16 v157, v154 offset:14848
	ds_read_u16 v164, v154 offset:14912
	ds_read_u16 v165, v154 offset:14976
	v_pk_mul_f32 v[26:27], v[26:27], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[90:91] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[158:159], v[108:109], v[66:81]
	ds_read_u16 v108, v154 offset:8576
	ds_read_u16 v109, v154 offset:8640
	ds_read_u16 v158, v154 offset:8512
	ds_read_u16 v159, v154 offset:8448
	ds_read_u16 v166, v154 offset:8384
	ds_read_u16 v167, v154 offset:9088
	ds_read_u16 v168, v154 offset:9152
	v_pk_mul_f32 v[14:15], v[14:15], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[90:91] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[160:161], v[102:103], v[66:81]
	ds_read_u16 v160, v154 offset:9024
	ds_read_u16 v102, v154 offset:8960
	ds_read_u16 v161, v154 offset:8896
	ds_read_u16 v169, v154 offset:10624
	ds_read_u16 v171, v154 offset:10688
	ds_read_u16 v103, v154 offset:10496
	ds_read_u16 v172, v154 offset:10432
	ds_read_u16 v173, v154 offset:11136
	ds_read_u16 v179, v154 offset:11200
	v_pk_mul_f32 v[2:3], v[2:3], v[90:91] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[66:81], v[162:163], v[104:105], v[66:81]
	ds_read_u16 v162, v154 offset:11072
	ds_read_u16 v104, v154 offset:11008
	ds_read_u16 v163, v154 offset:10944
	ds_read_u16 v180, v154 offset:12672
	ds_read_u16 v181, v154 offset:12736
	ds_read_u16 v105, v154 offset:12544
	ds_read_u16 v182, v154 offset:12480
	ds_read_u16 v184, v154 offset:13184
	ds_read_u16 v192, v154 offset:13248
	ds_read_u16 v199, v154 offset:13056
	ds_read_u16 v200, v154 offset:12992
	v_mfma_f32_32x32x8_f16 v[66:81], v[86:87], v[98:99], v[66:81]
	s_waitcnt lgkmcnt(14)
	v_perm_b32 v99, v102, v142, s5
	v_perm_b32 v98, v159, v110, s5
	v_pack_b32_f16 v87, v120, v121
	v_pack_b32_f16 v86, v118, v119
	ds_read_u16 v110, v154 offset:14720
	ds_read_u16 v118, v154 offset:14784
	v_mfma_f32_32x32x8_f16 v[50:65], v[98:99], v[86:87], v[50:65]
	v_pack_b32_f16 v99, v124, v125
	v_pack_b32_f16 v98, v114, v115
	v_mfma_f32_32x32x8_f16 v[66:81], v[88:89], v[100:101], v[66:81]
	s_waitcnt lgkmcnt(11)
	v_perm_b32 v89, v104, v146, s5
	v_perm_b32 v88, v103, v112, s5
	ds_read_u16 v102, v154 offset:14592
	ds_read_u16 v112, v154 offset:14528
	ds_read_u16 v114, v154 offset:15232
	ds_read_u16 v115, v154 offset:15296
	ds_read_u16 v103, v154 offset:15104
	v_pack_b32_f16 v101, v132, v116
	v_pack_b32_f16 v100, v130, v131
	v_perm_b32 v104, v158, v111, s5
	v_mfma_f32_32x32x8_f16 v[50:65], v[88:89], v[98:99], v[50:65]
	s_waitcnt lgkmcnt(8)
	v_perm_b32 v89, v199, v106, s5
	v_perm_b32 v88, v105, v149, s5
	v_perm_b32 v105, v160, v143, s5
	ds_read_u16 v116, v154 offset:15040
	v_mfma_f32_32x32x8_f16 v[50:65], v[88:89], v[100:101], v[50:65]
	s_waitcnt lgkmcnt(1)
	v_perm_b32 v89, v103, v157, s5
	v_perm_b32 v88, v102, v153, s5
	v_pack_b32_f16 v103, v139, v140
	v_pack_b32_f16 v102, v117, v133
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[88:89], v[102:103], v[50:65]
	ds_read_u16 v88, v154 offset:10560
	v_perm_b32 v89, v162, v147, s5
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v88, v88, v113, s5
	v_mfma_f32_32x32x8_f16 v[34:49], v[104:105], v[86:87], v[34:49]
	ds_read_u16 v104, v154 offset:13120
	ds_read_u16 v106, v154 offset:12608
	ds_read_u16 v111, v154 offset:14656
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v105, v104, v107, s5
	s_waitcnt lgkmcnt(1)
	v_perm_b32 v104, v106, v150, s5
	v_add_f32_e32 v106, v203, v204
	v_add_f32_e32 v106, v106, v198
	v_mfma_f32_32x32x8_f16 v[34:49], v[88:89], v[98:99], v[34:49]
	ds_read_u16 v89, v154 offset:15168
	v_perm_b32 v107, v167, v144, s5
	v_and_b32_e32 v88, 0x100, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v88
	v_and_b32_e32 v88, 0xa0, v176
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[34:49], v[104:105], v[100:101], v[34:49]
	v_perm_b32 v105, v89, v164, s5
	v_add_f32_e32 v89, v106, v197
	v_perm_b32 v106, v108, v141, s5
	v_perm_b32 v104, v111, v155, s5
	v_perm_b32 v108, v171, v172, s5
	s_waitcnt vmcnt(0)
	ds_write_b128 v177, v[82:85] offset:8192
	v_mfma_f32_32x32x8_f16 v[18:33], v[106:107], v[86:87], v[18:33]
	v_perm_b32 v107, v168, v161, s5
	v_perm_b32 v106, v109, v166, s5
	v_perm_b32 v109, v179, v163, s5
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_f32_e32 v89, v89, v196
	v_mfma_f32_32x32x8_f16 v[34:49], v[104:105], v[102:103], v[34:49]
	v_perm_b32 v105, v173, v148, s5
	v_perm_b32 v104, v169, v145, s5
	v_add_f32_e32 v89, v89, v195
	v_add_f32_e32 v89, v89, v194
	v_add_f32_e32 v89, v89, v193
	v_add_f32_e32 v89, v89, v191
	v_add_f32_e32 v89, v89, v190
	v_mfma_f32_32x32x8_f16 v[18:33], v[104:105], v[98:99], v[18:33]
	v_perm_b32 v105, v184, v152, s5
	v_perm_b32 v104, v180, v151, s5
	v_add_f32_e32 v89, v89, v189
	v_add_f32_e32 v89, v89, v188
	v_add_f32_e32 v89, v89, v187
	v_add_f32_e32 v89, v89, v186
	v_add_f32_e32 v89, v89, v185
	v_mfma_f32_32x32x8_f16 v[18:33], v[104:105], v[100:101], v[18:33]
	v_perm_b32 v105, v114, v165, s5
	v_perm_b32 v104, v110, v156, s5
	v_add_f32_e32 v89, v89, v183
	v_or3_b32 v1, v88, v174, v1
	v_mfma_f32_32x32x8_f16 v[2:17], v[106:107], v[86:87], v[2:17]
	v_perm_b32 v87, v192, v200, s5
	v_perm_b32 v86, v181, v182, s5
	v_mfma_f32_32x32x8_f16 v[18:33], v[104:105], v[102:103], v[18:33]
	v_max_f32_e32 v104, v67, v67
	v_max_f32_e32 v105, v66, v66
	v_max_f32_e32 v104, v105, v104
	v_max3_f32 v104, v104, v68, v69
	v_mfma_f32_32x32x8_f16 v[2:17], v[108:109], v[98:99], v[2:17]
	v_max3_f32 v98, v104, v70, v71
	v_max3_f32 v98, v98, v72, v73
	v_max3_f32 v98, v98, v74, v75
	v_max3_f32 v98, v98, v76, v77
	v_max3_f32 v98, v98, v78, v79
	v_max3_f32 v104, v98, v80, v81
	ds_bpermute_b32 v105, v175, v104
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[100:101], v[2:17]
	v_perm_b32 v99, v115, v116, s5
	v_perm_b32 v98, v118, v112, s5
	v_mov_b32_e32 v86, v81
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v87, v137, v104, v105
	v_pk_mul_f32 v[100:101], v[86:87], s[6:7] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v66, v66, s6, -v101
	v_fma_f32 v67, v67, s6, -v101
	v_fma_f32 v68, v68, s6, -v101
	v_fma_f32 v69, v69, s6, -v101
	v_mfma_f32_32x32x8_f16 v[2:17], v[98:99], v[102:103], v[2:17]
	v_exp_f32_e32 v86, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v98, v68
	v_exp_f32_e32 v99, v69
	v_fma_f32 v70, v70, s6, -v101
	v_fma_f32 v71, v71, s6, -v101
	v_sub_f32_e32 v81, v100, v101
	v_exp_f32_e32 v100, v70
	v_exp_f32_e32 v102, v71
	v_cvt_f16_f32_e32 v68, v86
	v_cvt_f16_f32_e32 v70, v67
	v_cvt_f16_f32_e32 v69, v98
	v_cvt_f16_f32_e32 v71, v99
	v_sub_f32_e32 v66, v135, v101
	v_fma_f32 v72, v72, s6, -v101
	v_fma_f32 v73, v73, s6, -v101
	v_fma_f32 v74, v74, s6, -v101
	v_fma_f32 v75, v75, s6, -v101
	v_fma_f32 v76, v76, s6, -v101
	v_fma_f32 v77, v77, s6, -v101
	v_fma_f32 v78, v78, s6, -v101
	v_fma_f32 v79, v79, s6, -v101
	v_fma_f32 v80, v80, s6, -v101
	v_exp_f32_e32 v66, v66
	ds_read_u16 v101, v154 offset:8192
	ds_read_u16 v111, v154 offset:8256
	ds_read_u16 v112, v154 offset:8320
	ds_read_u16 v113, v154 offset:8704
	ds_read_u16 v114, v154 offset:8768
	ds_read_u16 v115, v154 offset:8832
	ds_read_u16 v116, v154 offset:10240
	ds_read_u16 v117, v154 offset:10304
	ds_read_u16 v118, v154 offset:10368
	ds_read_u16 v119, v154 offset:10752
	ds_read_u16 v120, v154 offset:10816
	ds_read_u16 v121, v154 offset:10880
	ds_read_u16 v124, v154 offset:12288
	ds_read_u16 v125, v154 offset:12352
	ds_read_u16 v130, v154 offset:12416
	ds_read_u16 v131, v154 offset:12800
	ds_read_u16 v132, v154 offset:12864
	ds_read_u16 v133, v154 offset:12928
	ds_read_u16 v135, v154 offset:14336
	ds_read_u16 v137, v154 offset:14400
	ds_read_u16 v139, v154 offset:14464
	ds_read_u16 v140, v154 offset:14848
	ds_read_u16 v141, v154 offset:14912
	ds_read_u16 v142, v154 offset:14976
	ds_read_u16 v143, v154 offset:8576
	ds_read_u16 v144, v154 offset:8640
	ds_read_u16 v145, v154 offset:8512
	ds_read_u16 v146, v154 offset:8448
	ds_read_u16 v147, v154 offset:8384
	ds_read_u16 v148, v154 offset:9088
	ds_read_u16 v149, v154 offset:9152
	ds_read_u16 v150, v154 offset:9024
	ds_read_u16 v151, v154 offset:8960
	ds_read_u16 v152, v154 offset:8896
	ds_read_u16 v153, v154 offset:10624
	ds_read_u16 v155, v154 offset:10688
	ds_read_u16 v156, v154 offset:10560
	ds_read_u16 v157, v154 offset:10496
	ds_read_u16 v158, v154 offset:10432
	ds_read_u16 v159, v154 offset:11136
	ds_read_u16 v160, v154 offset:11200
	ds_read_u16 v161, v154 offset:11072
	ds_read_u16 v162, v154 offset:11008
	ds_read_u16 v163, v154 offset:10944
	ds_read_u16 v164, v154 offset:12672
	ds_read_u16 v165, v154 offset:12736
	ds_read_u16 v166, v154 offset:12608
	ds_read_u16 v167, v154 offset:12544
	ds_read_u16 v168, v154 offset:12480
	ds_read_u16 v169, v154 offset:13184
	ds_read_u16 v171, v154 offset:13248
	ds_read_u16 v172, v154 offset:13120
	ds_read_u16 v173, v154 offset:13056
	ds_read_u16 v176, v154 offset:12992
	ds_read_u16 v177, v154 offset:14720
	ds_read_u16 v179, v154 offset:14784
	v_pack_b32_f16 v69, v69, v71
	v_pack_b32_f16 v68, v68, v70
	s_waitcnt lgkmcnt(14)
	v_perm_b32 v71, v151, v113, s5
	v_perm_b32 v70, v146, v101, s5
	v_exp_f32_e32 v103, v72
	v_exp_f32_e32 v104, v73
	v_pk_mul_f32 v[64:65], v[64:65], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_exp_f32_e32 v107, v76
	v_exp_f32_e32 v108, v77
	v_mfma_f32_32x32x8_f16 v[50:65], v[70:71], v[68:69], v[50:65]
	v_exp_f32_e32 v105, v74
	v_exp_f32_e32 v106, v75
	v_cvt_f16_f32_e32 v72, v100
	v_cvt_f16_f32_e32 v74, v102
	v_cvt_f16_f32_e32 v73, v103
	v_cvt_f16_f32_e32 v75, v104
	s_waitcnt lgkmcnt(13)
	v_perm_b32 v71, v162, v119, s5
	v_perm_b32 v70, v157, v116, s5
	v_exp_f32_e32 v109, v78
	v_exp_f32_e32 v110, v79
	v_cvt_f16_f32_e32 v78, v107
	v_cvt_f16_f32_e32 v79, v108
	v_pack_b32_f16 v73, v73, v75
	v_pack_b32_f16 v72, v72, v74
	v_cvt_f16_f32_e32 v76, v105
	v_cvt_f16_f32_e32 v77, v106
	v_mfma_f32_32x32x8_f16 v[50:65], v[70:71], v[72:73], v[50:65]
	v_pack_b32_f16 v71, v78, v79
	v_perm_b32 v79, v150, v114, s5
	v_perm_b32 v78, v145, v111, s5
	s_waitcnt lgkmcnt(3)
	v_perm_b32 v75, v173, v131, s5
	v_perm_b32 v74, v167, v124, s5
	v_pk_mul_f32 v[48:49], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[66:67] op_sel_hi:[1,0]
	ds_read_u16 v101, v154 offset:14656
	ds_read_u16 v113, v154 offset:14592
	ds_read_u16 v116, v154 offset:14528
	ds_read_u16 v119, v154 offset:15232
	ds_read_u16 v124, v154 offset:15296
	ds_read_u16 v131, v154 offset:15168
	ds_read_u16 v146, v154 offset:15104
	v_mfma_f32_32x32x8_f16 v[34:49], v[78:79], v[68:69], v[34:49]
	ds_bpermute_b32 v78, v175, v89
	v_exp_f32_e32 v80, v80
	v_exp_f32_e32 v81, v81
	v_cvt_f16_f32_e32 v82, v109
	v_cvt_f16_f32_e32 v83, v110
	v_pack_b32_f16 v70, v76, v77
	v_cvt_f16_f32_e32 v84, v80
	v_cvt_f16_f32_e32 v85, v81
	v_mfma_f32_32x32x8_f16 v[50:65], v[74:75], v[70:71], v[50:65]
	s_waitcnt lgkmcnt(1)
	v_perm_b32 v75, v146, v140, s5
	v_perm_b32 v74, v113, v135, s5
	v_pack_b32_f16 v76, v82, v83
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v82, v89, v78
	v_add_f32_e32 v78, v128, v134
	v_add_f32_e32 v78, v126, v78
	v_add_f32_e32 v78, v127, v78
	v_add_f32_e32 v78, v136, v78
	v_pack_b32_f16 v77, v84, v85
	v_add_f32_e32 v78, v138, v78
	v_add_f32_e32 v78, v129, v78
	v_mfma_f32_32x32x8_f16 v[50:65], v[74:75], v[76:77], v[50:65]
	v_perm_b32 v75, v161, v120, s5
	v_perm_b32 v74, v156, v117, s5
	v_add_f32_e32 v78, v122, v78
	v_add_f32_e32 v83, v123, v78
	v_perm_b32 v79, v148, v115, s5
	v_perm_b32 v78, v143, v112, s5
	v_pk_mul_f32 v[32:33], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[34:49], v[74:75], v[72:73], v[34:49]
	v_pk_mul_f32 v[30:31], v[30:31], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[66:67] op_sel_hi:[1,0]
	v_perm_b32 v75, v172, v132, s5
	v_perm_b32 v74, v166, v125, s5
	v_mfma_f32_32x32x8_f16 v[18:33], v[78:79], v[68:69], v[18:33]
	v_add_f32_e32 v78, v92, v83
	v_add_f32_e32 v78, v94, v78
	v_add_f32_e32 v78, v95, v78
	v_add_f32_e32 v78, v96, v78
	v_add_f32_e32 v78, v97, v78
	v_add_f32_e32 v78, v93, v78
	v_add_f32_e32 v67, v86, v67
	v_mfma_f32_32x32x8_f16 v[34:49], v[74:75], v[70:71], v[34:49]
	v_add_f32_e32 v83, v91, v78
	v_add_f32_e32 v67, v98, v67
	v_perm_b32 v79, v149, v152, s5
	v_perm_b32 v78, v144, v147, s5
	v_add_f32_e32 v67, v99, v67
	v_add_f32_e32 v67, v100, v67
	v_perm_b32 v75, v131, v141, s5
	v_perm_b32 v74, v101, v137, s5
	v_add_f32_e32 v67, v102, v67
	v_pk_mul_f32 v[16:17], v[16:17], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[34:49], v[74:75], v[76:77], v[34:49]
	v_perm_b32 v75, v159, v121, s5
	v_perm_b32 v74, v153, v118, s5
	ds_read_u16 v151, v154 offset:15040
	v_add_f32_e32 v67, v103, v67
	v_add_f32_e32 v67, v104, v67
	v_add_f32_e32 v67, v105, v67
	v_add_f32_e32 v67, v106, v67
	v_mfma_f32_32x32x8_f16 v[2:17], v[78:79], v[68:69], v[2:17]
	v_perm_b32 v69, v160, v163, s5
	v_perm_b32 v68, v155, v158, s5
	v_add_f32_e32 v67, v107, v67
	v_add_f32_e32 v67, v108, v67
	v_add_f32_e32 v67, v109, v67
	v_add_f32_e32 v67, v110, v67
	v_add_f32_e32 v67, v80, v67
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[72:73], v[18:33]
	v_perm_b32 v75, v169, v133, s5
	v_perm_b32 v74, v164, v130, s5
	ds_bpermute_b32 v84, v175, v83
	v_add_f32_e32 v67, v81, v67
	v_fmac_f32_e32 v82, v178, v170
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[72:73], v[2:17]
	v_perm_b32 v69, v171, v176, s5
	v_perm_b32 v68, v165, v168, s5
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[70:71], v[18:33]
	v_perm_b32 v75, v119, v142, s5
	v_perm_b32 v74, v177, v139, s5
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[70:71], v[2:17]
	v_perm_b32 v69, v124, v151, s5
	v_perm_b32 v68, v179, v116, s5
	ds_bpermute_b32 v70, v175, v67
	v_add_f32_e32 v71, v83, v84
	v_fmac_f32_e32 v71, v82, v90
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v67, v67, v70
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[76:77], v[18:33]
	v_fmac_f32_e32 v67, v71, v66
	v_lshl_add_u32 v66, v1, 2, 0
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[76:77], v[2:17]
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	s_mov_b32 s6, 0x800000
	v_cmp_gt_f32_e32 vcc, s6, v67
	v_mov_b32_e32 v69, 0x42000000
	v_or_b32_e32 v68, s34, v1
	v_cndmask_b32_e64 v70, 0, 32, vcc
	v_ldexp_f32 v70, v67, v70
	v_log_f32_e32 v70, v70
	s_movk_i32 s5, 0x4000
	v_cndmask_b32_e32 v69, 0, v69, vcc
	v_cmp_gt_i32_e64 s[8:9], s5, v68
	v_sub_f32_e32 v68, v70, v69
	v_add_f32_e32 v68, v87, v68
	ds_write_b32 v66, v68
	v_mov_b32_e32 v68, 2
	v_lshlrev_b32_sdwa v68, v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v69, 0, v68
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v69, v69
	s_sub_i32 s5, 0x4000, s34
	v_cmp_lt_i32_sdwa s[14:15], v0, s5 src0_sel:BYTE_0 src1_sel:DWORD
	v_bfrev_b32_e32 v70, 1
	s_and_b64 vcc, s[0:1], s[14:15]
	s_and_b32 s5, s13, 0xffff
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	v_cndmask_b32_e32 v68, v70, v68, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v69, v68, s[4:7], 0 offen
	s_cbranch_execz .LBB0_5
	s_branch .LBB0_6
.LBB0_4:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_5:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v67
	v_mov_b32_e32 v68, 0x42000000
	s_and_b32 s5, s13, 0xffff
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v67, v69
	v_log_f32_e32 v69, v69
	v_cndmask_b32_e32 v68, 0, v68, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_sub_f32_e32 v68, v69, v68
	v_add_f32_e32 v68, v87, v68
	ds_write_b32 v66, v68
	v_mov_b32_e32 v66, 2
	v_lshlrev_b32_sdwa v66, v66, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v68, 0, v66
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v68, v68
	v_bfrev_b32_e32 v69, 1
	v_cndmask_b32_e64 v66, v69, v66, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v68, v66, s[4:7], 0 offen
.LBB0_6:                                ; %.critedge
	v_div_scale_f32 v66, s[0:1], v67, v67, 1.0
	v_rcp_f32_e32 v66, v66
	v_div_scale_f32 v68, vcc, 1.0, v67, 1.0
	v_mov_b32_e32 v69, v16
	v_mul_f32_e32 v66, v68, v66
	v_mov_b32_e32 v68, v15
	s_nop 0
	v_div_fmas_f32 v66, 0, 0, v66
	v_div_fixup_f32 v66, v66, v67, 1.0
	v_fma_mixlo_f16 v67, v66, v17, 0
	v_pk_mul_f32 v[16:17], v[66:67], v[68:69] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v68, v66, v14, 0
	v_mov_b32_e32 v14, v11
	v_mov_b32_e32 v15, v12
	v_fma_mixlo_f16 v69, v66, v13, 0
	v_pk_mul_f32 v[12:13], v[66:67], v[14:15] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v14, v66, v10, 0
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v8
	v_fma_mixlo_f16 v15, v66, v9, 0
	v_pk_mul_f32 v[8:9], v[66:67], v[10:11] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v10, v66, v6, 0
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v4
	v_fma_mixlo_f16 v11, v66, v5, 0
	v_pk_mul_f32 v[4:5], v[66:67], v[6:7] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v6, v66, v2, 0
	v_mov_b32_e32 v2, v31
	v_mov_b32_e32 v3, v32
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v31, v3
	v_cvt_f16_f32_e32 v32, v2
	v_mov_b32_e32 v2, v27
	v_mov_b32_e32 v3, v28
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v27, v3
	v_cvt_f16_f32_e32 v28, v2
	v_mov_b32_e32 v2, v23
	v_mov_b32_e32 v3, v24
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v23, v3
	v_cvt_f16_f32_e32 v24, v2
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v3, v20
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v19, v3
	v_cvt_f16_f32_e32 v20, v2
	v_mov_b32_e32 v2, v47
	v_mov_b32_e32 v3, v48
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v47, v3
	v_cvt_f16_f32_e32 v48, v2
	v_mov_b32_e32 v2, v43
	v_mov_b32_e32 v3, v44
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v43, v3
	v_cvt_f16_f32_e32 v44, v2
	v_mov_b32_e32 v2, v39
	v_mov_b32_e32 v3, v40
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v39, v3
	v_cvt_f16_f32_e32 v40, v2
	v_mov_b32_e32 v2, v35
	v_mov_b32_e32 v3, v36
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v35, v3
	v_cvt_f16_f32_e32 v36, v2
	v_mov_b32_e32 v2, v63
	v_mov_b32_e32 v3, v64
	s_mul_i32 s0, s25, s18
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	s_ashr_i32 s1, s0, 31
	v_cvt_f16_f32_e32 v63, v3
	v_cvt_f16_f32_e32 v64, v2
	v_mov_b32_e32 v2, v59
	v_mov_b32_e32 v3, v60
	s_lshl_b64 s[0:1], s[0:1], 1
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s26, s17
	v_cvt_f16_f32_e32 v59, v3
	v_cvt_f16_f32_e32 v60, v2
	v_mov_b32_e32 v2, v55
	v_mov_b32_e32 v3, v56
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_f16_f32_e32 v55, v3
	v_cvt_f16_f32_e32 v56, v2
	v_mov_b32_e32 v2, v51
	v_mov_b32_e32 v3, v52
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s27, s34
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v2, v2
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	v_lshrrev_b32_e32 v0, 3, v0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s27, 0x3fff
	v_fma_mixlo_f16 v53, v66, v53, 0
	v_fma_mixlo_f16 v50, v66, v50, 0
	v_and_b32_e32 v51, 4, v0
	v_mul_lo_u32 v52, s27, v1
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s4, 0x5040100
	s_or_b32 s1, s2, s1
	v_pack_b32_f16 v0, v50, v2
	v_perm_b32 v1, v53, v3, s4
	v_add_lshl_u32 v2, v52, v51, 1
	v_bfrev_b32_e32 v3, 1
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e64 v50, v3, v2, s[8:9]
	v_fma_mixlo_f16 v57, v66, v57, 0
	v_fma_mixlo_f16 v54, v66, v54, 0
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_add_u32_e32 v50, 16, v2
	v_pack_b32_f16 v0, v54, v56
	v_perm_b32 v1, v57, v55, s4
	v_cndmask_b32_e64 v50, v3, v50, s[8:9]
	v_fma_mixlo_f16 v61, v66, v61, 0
	v_fma_mixlo_f16 v58, v66, v58, 0
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_add_u32_e32 v50, 32, v2
	v_fma_mixlo_f16 v7, v66, v33, 0
	v_fma_mixlo_f16 v33, v66, v49, 0
	v_fma_mixlo_f16 v49, v66, v65, 0
	v_pack_b32_f16 v0, v58, v60
	v_perm_b32 v1, v61, v59, s4
	v_cndmask_b32_e64 v50, v3, v50, s[8:9]
	v_fma_mixlo_f16 v62, v66, v62, 0
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_perm_b32 v1, v49, v63, s4
	v_add_u32_e32 v49, 48, v2
	v_fma_mixlo_f16 v34, v66, v34, 0
	v_pack_b32_f16 v0, v62, v64
	v_cndmask_b32_e64 v49, v3, v49, s[8:9]
	v_fma_mixlo_f16 v37, v66, v37, 0
	buffer_store_dwordx2 v[0:1], v49, s[0:3], 0 offen
	v_pack_b32_f16 v0, v34, v36
	v_add_u32_e32 v34, 64, v2
	v_perm_b32 v1, v37, v35, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	v_fma_mixlo_f16 v41, v66, v41, 0
	v_fma_mixlo_f16 v38, v66, v38, 0
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_add_u32_e32 v34, 0x50, v2
	v_pack_b32_f16 v0, v38, v40
	v_perm_b32 v1, v41, v39, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	v_fma_mixlo_f16 v45, v66, v45, 0
	v_fma_mixlo_f16 v42, v66, v42, 0
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_add_u32_e32 v34, 0x60, v2
	v_pack_b32_f16 v0, v42, v44
	v_perm_b32 v1, v45, v43, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	v_fma_mixlo_f16 v46, v66, v46, 0
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_perm_b32 v1, v33, v47, s4
	v_add_u32_e32 v33, 0x70, v2
	v_fma_mixlo_f16 v18, v66, v18, 0
	v_pack_b32_f16 v0, v46, v48
	v_cndmask_b32_e64 v33, v3, v33, s[8:9]
	v_fma_mixlo_f16 v21, v66, v21, 0
	buffer_store_dwordx2 v[0:1], v33, s[0:3], 0 offen
	v_pack_b32_f16 v0, v18, v20
	v_add_u32_e32 v18, 0x80, v2
	v_perm_b32 v1, v21, v19, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	v_fma_mixlo_f16 v25, v66, v25, 0
	v_fma_mixlo_f16 v22, v66, v22, 0
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_add_u32_e32 v18, 0x90, v2
	v_pack_b32_f16 v0, v22, v24
	v_perm_b32 v1, v25, v23, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	v_cvt_f16_f32_e32 v4, v4
	v_fma_mixlo_f16 v29, v66, v29, 0
	v_fma_mixlo_f16 v26, v66, v26, 0
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_add_u32_e32 v18, 0xa0, v2
	v_cvt_f16_f32_e32 v5, v5
	v_pack_b32_f16 v0, v26, v28
	v_perm_b32 v1, v29, v27, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	v_fma_mixlo_f16 v30, v66, v30, 0
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_perm_b32 v1, v7, v31, s4
	v_add_u32_e32 v7, 0xb0, v2
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v8, v8
	v_pack_b32_f16 v0, v30, v32
	v_cndmask_b32_e64 v7, v3, v7, s[8:9]
	buffer_store_dwordx2 v[0:1], v7, s[0:3], 0 offen
	v_pack_b32_f16 v0, v6, v4
	v_add_u32_e32 v4, 0xc0, v2
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v12, v12
	v_perm_b32 v1, v11, v5, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v16, v16
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xd0, v2
	v_pack_b32_f16 v0, v10, v8
	v_perm_b32 v1, v15, v9, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xe0, v2
	v_pack_b32_f16 v0, v14, v12
	v_perm_b32 v1, v69, v13, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	v_add_u32_e32 v2, 0xf0, v2
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_pack_b32_f16 v0, v68, v16
	v_perm_b32 v1, v67, v17, s4
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
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
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 246
		.amdhsa_next_free_sgpr 52
		.amdhsa_accum_offset 248
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
	.set attn_fwd.num_vgpr, 246
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 52
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11736
; TotalNumSgprs: 58
; NumVgprs: 246
; NumAgprs: 0
; TotalNumVgprs: 246
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 30
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 246
; AccumOffset: 248
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 61
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
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     246
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
