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
	v_lshrrev_b32_e32 v23, 4, v0
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s28
	v_or_b32_e32 v2, 0x60, v23
	s_addc_u32 s13, s16, s3
	s_ashr_i32 s3, s2, 31
	s_load_dwordx4 s[20:23], s[0:1], 0x38
	s_load_dword s19, s[0:1], 0x48
	v_or_b32_e32 v11, s28, v2
	s_lshl_b32 s25, s14, 6
	v_mul_lo_u32 v12, s14, v2
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v2, 3, v0
	v_or_b32_e32 v3, 0xa0, v23
	s_add_u32 s12, s12, s2
	v_and_b32_e32 v22, 0x78, v2
	s_mul_i32 s30, s15, s18
	v_or_b32_e32 v19, s28, v3
	v_mul_lo_u32 v20, s14, v3
	s_addc_u32 s13, s13, s3
	v_mad_u64_u32 v[2:3], s[2:3], s14, v23, v[22:23]
	s_ashr_i32 s31, s30, 31
	s_lshl_b64 s[2:3], s[30:31], 1
	s_add_u32 s15, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s34, s20, s17
	s_addc_u32 s16, s5, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 1
	s_add_u32 s24, s15, s2
	s_mul_i32 s26, s22, s18
	s_addc_u32 s3, s16, s3
	s_ashr_i32 s27, s26, 31
	s_lshl_b64 s[26:27], s[26:27], 1
	s_add_u32 s2, s6, s26
	s_mul_i32 s6, s23, s17
	s_addc_u32 s15, s7, s27
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_u32 s20, s2, s6
	v_or_b32_e32 v1, 32, v23
	v_or_b32_e32 v5, s28, v23
	s_addc_u32 s16, s15, s7
	s_movk_i32 s2, 0x4000
	s_and_b32 s6, s14, 0x3fff
	v_or_b32_e32 v6, s28, v1
	v_mul_lo_u32 v1, s14, v1
	v_add_u32_e32 v13, s25, v2
	s_bitset1_b32 s6, 14
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v32, 1
	v_cmp_gt_i32_e32 vcc, s2, v5
	v_or_b32_e32 v4, 0xe0, v23
	s_and_b32 s7, s13, 0xffff
	s_lshl_b32 s6, s6, 16
	v_cndmask_b32_e32 v14, v32, v2, vcc
	v_add_lshl_u32 v1, v1, v22, 1
	v_cmp_gt_i32_e32 vcc, s2, v6
	v_or_b32_e32 v10, 64, v5
	v_mul_lo_u32 v30, s14, v4
	s_or_b32 s13, s7, s6
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_or_b32_e32 v18, 0x80, v5
	v_or_b32_e32 v28, 0xc0, v5
	v_or_b32_e32 v29, s28, v4
	buffer_load_dwordx4 v[2:5], v14, s[12:15], 0 offen
	buffer_load_dwordx4 v[6:9], v1, s[12:15], 0 offen
	v_lshlrev_b32_e32 v1, 1, v13
	v_cmp_gt_i32_e32 vcc, s2, v10
	v_add_u32_e32 v31, s25, v13
	v_add_lshl_u32 v10, v12, v22, 1
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_cmp_gt_i32_e32 vcc, s2, v11
	v_lshrrev_b32_e32 v158, 1, v0
	v_and_b32_e32 v37, 56, v158
	v_cndmask_b32_e32 v21, v32, v10, vcc
	buffer_load_dwordx4 v[10:13], v1, s[12:15], 0 offen
	buffer_load_dwordx4 v[14:17], v21, s[12:15], 0 offen
	v_lshlrev_b32_e32 v1, 1, v31
	v_cmp_gt_i32_e32 vcc, s2, v18
	v_add_lshl_u32 v18, v20, v22, 1
	v_xor_b32_e32 v37, v37, v22
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_cmp_gt_i32_e32 vcc, s2, v19
	v_lshlrev_b32_e32 v40, 8, v23
	s_and_b32 s6, s3, 0xffff
	v_cndmask_b32_e32 v33, v32, v18, vcc
	buffer_load_dwordx4 v[18:21], v1, s[12:15], 0 offen
	buffer_load_dwordx4 v[24:27], v33, s[12:15], 0 offen
	v_add_lshl_u32 v1, v31, s25, 1
	v_cmp_gt_i32_e32 vcc, s2, v28
	v_add_lshl_u32 v28, v30, v22, 1
	s_mov_b32 s26, s14
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_cmp_gt_i32_e32 vcc, s2, v29
	s_and_b32 s2, s21, 0x3fff
	s_bitset1_b32 s2, 14
	v_cndmask_b32_e32 v36, v32, v28, vcc
	buffer_load_dwordx4 v[28:31], v1, s[12:15], 0 offen
	buffer_load_dwordx4 v[32:35], v36, s[12:15], 0 offen
	v_and_b32_e32 v1, 64, v158
	v_xor_b32_e32 v37, v37, v1
	v_mul_lo_u32 v36, s21, v23
	v_lshlrev_b32_e32 v37, 1, v37
	s_lshl_b32 s29, s2, 16
	v_add3_u32 v162, 0, v37, v40
	s_or_b32 s25, s6, s29
	s_mov_b32 s27, s15
	v_add_lshl_u32 v154, v36, v22, 1
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v162, v[2:5]
	s_waitcnt vmcnt(6)
	ds_write_b128 v162, v[6:9] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v162, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v162, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v162, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v162, v[24:27] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v162, v[28:31] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v162, v[32:35] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[2:5], v154, s[24:27], 0 offen
	v_and_b32_e32 v156, 31, v0
	s_movk_i32 s6, 0xe0
	s_lshl_b32 s36, s21, 5
	v_bfe_u32 v6, v0, 5, 1
	v_and_b32_e32 v8, 15, v0
	v_and_or_b32 v7, v158, s6, v156
	v_xor_b32_e32 v9, v6, v8
	v_or_b32_e32 v10, 2, v6
	v_or_b32_e32 v11, 4, v6
	v_or_b32_e32 v12, 6, v6
	v_or_b32_e32 v13, 8, v6
	v_or_b32_e32 v14, 10, v6
	v_or_b32_e32 v15, 12, v6
	v_or_b32_e32 v6, 14, v6
	s_ashr_i32 s37, s36, 31
	s_lshl_b32 s2, s19, 5
	v_xor_b32_e32 v10, v10, v8
	v_xor_b32_e32 v11, v11, v8
	v_xor_b32_e32 v12, v12, v8
	v_xor_b32_e32 v13, v13, v8
	v_xor_b32_e32 v14, v14, v8
	v_xor_b32_e32 v15, v15, v8
	v_xor_b32_e32 v6, v6, v8
	v_lshl_add_u32 v7, v7, 8, 0
	v_lshlrev_b32_e32 v8, 4, v9
	s_lshl_b64 s[26:27], s[36:37], 1
	v_add_u32_e32 v9, v7, v8
	v_lshlrev_b32_e32 v10, 4, v10
	v_lshlrev_b32_e32 v32, 4, v11
	s_add_u32 s12, s24, s26
	v_add_u32_e32 v16, v7, v10
	ds_read_b128 v[126:129], v9
	ds_read_b128 v[122:125], v16
	v_add_u32_e32 v9, v7, v32
	v_lshlrev_b32_e32 v33, 4, v12
	v_lshlrev_b32_e32 v34, 4, v13
	s_addc_u32 s3, s3, s27
	v_add_u32_e32 v11, v7, v33
	ds_read_b128 v[118:121], v9
	ds_read_b128 v[114:117], v11
	v_add_u32_e32 v9, v7, v34
	v_lshlrev_b32_e32 v35, 4, v14
	v_lshlrev_b32_e32 v36, 4, v15
	s_and_b32 s6, s3, 0xffff
	v_lshlrev_b32_e32 v38, 8, v156
	v_add_u32_e32 v11, v7, v35
	ds_read_b128 v[110:113], v9
	ds_read_b128 v[106:109], v11
	v_add_u32_e32 v9, v7, v36
	v_lshlrev_b32_e32 v37, 4, v6
	s_or_b32 s13, s6, s29
	v_add3_u32 v163, 0, v8, v38
	v_add_u32_e32 v6, v7, v37
	ds_read_b128 v[102:105], v9
	ds_read_b128 v[98:101], v6
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[18:21], v154, s[12:15], 0 offen
	v_add3_u32 v164, 0, v10, v38
	v_add3_u32 v165, 0, v32, v38
	v_add3_u32 v166, 0, v33, v38
	v_add3_u32 v167, 0, v34, v38
	v_add3_u32 v172, 0, v35, v38
	s_and_b32 s6, s19, 0x3fff
	s_bitset1_b32 s6, 14
	v_mul_lo_u32 v23, s19, v23
	s_and_b32 s7, s16, 0xffff
	s_lshl_b32 s19, s6, 16
	s_or_b32 s21, s7, s19
	s_add_u32 s12, s12, s26
	s_addc_u32 s13, s3, s27
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	v_add_lshl_u32 v168, v23, v22, 1
	s_lshl_b64 s[6:7], s[2:3], 1
	v_add3_u32 v170, 0, v36, v38
	v_add3_u32 v171, 0, v37, v38
	s_movk_i32 s3, 0xff
	s_mov_b32 s24, 0x3e0293ee
	s_waitcnt vmcnt(1)
	ds_write_b128 v162, v[2:5]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[24:27], v163
	ds_read_b128 v[28:31], v164
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[24:25], v[126:127], 0
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[128:129], v[2:17]
	ds_read_b128 v[24:27], v165
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[122:123], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[124:125], v[2:17]
	ds_read_b128 v[28:31], v166
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[24:25], v[118:119], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[120:121], v[2:17]
	ds_read_b128 v[24:27], v167
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[114:115], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[116:117], v[2:17]
	ds_read_b128 v[28:31], v172
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[24:25], v[110:111], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[112:113], v[2:17]
	buffer_load_dwordx4 v[24:27], v168, s[20:23], 0 offen
	s_add_u32 s22, s20, s6
	s_addc_u32 s23, s16, s7
	s_and_b32 s2, s13, 0xffff
	s_or_b32 s13, s2, s29
	s_and_b32 s2, s23, 0xffff
	ds_read_b128 v[32:35], v170
	ds_read_b128 v[36:39], v171
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[86:89], v154, s[12:15], 0 offen
	s_or_b32 s13, s2, s19
	s_mov_b32 s12, s22
	s_waitcnt vmcnt(2)
	ds_write_b128 v162, v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v168, s[12:15], 0 offen
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[106:107], v[2:17]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_lshlrev_b32_e32 v19, 2, v0
	v_xor_b32_e32 v157, 0x80, v19
	v_lshlrev_b32_e32 v19, 1, v22
	v_add3_u32 v160, 0, v19, v40
	ds_read_b128 v[150:153], v163
	ds_read_b128 v[146:149], v164
	ds_read_b128 v[142:145], v165
	ds_read_b128 v[138:141], v166
	ds_read_b128 v[134:137], v167
	ds_read_b128 v[130:133], v172
	ds_read_b128 v[94:97], v170
	ds_read_b128 v[90:93], v171
	v_mov_b32_e32 v18, 0xff800000
	s_movk_i32 s2, 0x100
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[108:109], v[2:17]
	v_cmp_gt_u32_e32 vcc, s2, v0
	v_cmp_lt_u32_e64 s[2:3], s3, v0
	s_waitcnt vmcnt(2)
	ds_write_b128 v160, v[24:27] offset:8192
	v_mfma_f32_32x32x8_f16 v[2:17], v[32:33], v[102:103], v[2:17]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[36:37], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[38:39], v[100:101], v[2:17]
	s_nop 7
	s_nop 2
	v_max_f32_e32 v19, v3, v3
	v_max_f32_e32 v20, v2, v2
	v_max_f32_e32 v19, v20, v19
	v_max3_f32 v19, v19, v4, v5
	v_max3_f32 v19, v19, v6, v7
	v_max3_f32 v19, v19, v8, v9
	v_max3_f32 v19, v19, v10, v11
	v_max3_f32 v19, v19, v12, v13
	v_max3_f32 v19, v19, v14, v15
	v_max3_f32 v19, v19, v16, v17
	ds_bpermute_b32 v20, v157, v19
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v188, v19, v20, v18
	v_mul_f32_e32 v19, 0xbe0293ee, v188
	v_fmac_f32_e32 v18, 0xbe0293ee, v188
	v_fmamk_f32 v20, v2, 0x3e0293ee, v19
	v_fmamk_f32 v3, v3, 0x3e0293ee, v19
	v_fmamk_f32 v4, v4, 0x3e0293ee, v19
	v_fmamk_f32 v5, v5, 0x3e0293ee, v19
	v_fmamk_f32 v6, v6, 0x3e0293ee, v19
	v_fmamk_f32 v7, v7, 0x3e0293ee, v19
	v_fmamk_f32 v8, v8, 0x3e0293ee, v19
	v_fmamk_f32 v9, v9, 0x3e0293ee, v19
	v_fmamk_f32 v10, v10, 0x3e0293ee, v19
	v_fmamk_f32 v11, v11, 0x3e0293ee, v19
	v_fmamk_f32 v12, v12, 0x3e0293ee, v19
	v_fmamk_f32 v13, v13, 0x3e0293ee, v19
	v_fmamk_f32 v14, v14, 0x3e0293ee, v19
	v_fmamk_f32 v15, v15, 0x3e0293ee, v19
	v_fmamk_f32 v16, v16, 0x3e0293ee, v19
	v_fmac_f32_e32 v19, 0x3e0293ee, v17
	v_mov_b32_e32 v2, 0
	s_and_saveexec_b64 s[12:13], s[2:3]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[12:13]
	v_exp_f32_e32 v179, v3
	v_lshlrev_b32_e32 v3, 4, v0
	v_and_b32_e32 v189, 0x200, v3
	s_load_dwordx2 s[20:21], s[0:1], 0x4c
	s_load_dword s16, s[0:1], 0x54
	v_exp_f32_e32 v182, v4
	v_lshl_add_u32 v3, v189, 1, 0
	v_lshlrev_b32_e32 v4, 1, v156
	s_movk_i32 s0, 0x2000
	v_add3_u32 v155, v3, v4, s0
	s_add_u32 s0, s30, s34
	s_addc_u32 s1, s31, s35
	s_mul_i32 s3, s36, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_exp_f32_e32 v178, v20
	v_exp_f32_e32 v180, v5
	v_exp_f32_e32 v187, v6
	v_exp_f32_e32 v174, v7
	v_exp_f32_e32 v185, v8
	v_exp_f32_e32 v186, v9
	v_exp_f32_e32 v175, v10
	v_exp_f32_e32 v184, v11
	v_exp_f32_e32 v181, v12
	v_exp_f32_e32 v176, v13
	v_exp_f32_e32 v177, v14
	v_exp_f32_e32 v183, v15
	v_exp_f32_e32 v169, v16
	v_exp_f32_e32 v173, v19
	v_exp_f32_e32 v161, v18
	s_mul_hi_i32 s2, s36, 6
	s_add_u32 s0, s3, s0
	s_addc_u32 s1, s2, s1
	s_add_u32 s4, s4, s0
	s_addc_u32 s5, s5, s1
	v_mov_b32_e32 v159, 1.0
	s_movk_i32 s25, 0xffe0
	s_mov_b32 s30, 0x5040100
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
	v_mov_b32_e32 v190, v159
	v_mov_b32_e32 v191, v188
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[150:151], v[126:127], 0
	s_setprio 0
	v_add_f32_e32 v150, v178, v179
	v_mul_f32_e32 v17, v17, v161
	v_add_f32_e32 v150, v150, v182
	v_mul_f32_e32 v55, v55, v161
	v_add_f32_e32 v150, v150, v180
	v_mul_f32_e32 v50, v50, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[152:153], v[128:129], v[66:81]
	v_mul_f32_e32 v51, v51, v161
	v_add_f32_e32 v150, v150, v187
	v_mul_f32_e32 v52, v52, v161
	v_add_f32_e32 v150, v150, v174
	v_mul_f32_e32 v53, v53, v161
	v_add_f32_e32 v150, v150, v185
	v_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[122:123], v[66:81]
	v_mul_f32_e32 v56, v56, v161
	v_cvt_pkrtz_f16_f32 v147, v169, v173
	v_add_f32_e32 v146, v150, v186
	v_mul_f32_e32 v54, v54, v161
	v_add_f32_e32 v146, v146, v175
	v_mul_f32_e32 v57, v57, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[148:149], v[124:125], v[66:81]
	v_mul_f32_e32 v58, v58, v161
	v_add_f32_e32 v146, v146, v184
	v_mul_f32_e32 v59, v59, v161
	v_add_f32_e32 v146, v146, v181
	v_mul_f32_e32 v61, v61, v161
	v_add_f32_e32 v146, v146, v176
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[118:119], v[66:81]
	v_add_f32_e32 v146, v146, v177
	v_cvt_pkrtz_f16_f32 v143, v181, v176
	v_add_f32_e32 v146, v146, v183
	v_add_f32_e32 v146, v146, v169
	v_add_f32_e32 v148, v146, v173
	ds_bpermute_b32 v149, v157, v148
	v_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[120:121], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v159, v148, v149
	v_cvt_pkrtz_f16_f32 v142, v175, v184
	v_cvt_pkrtz_f16_f32 v149, v185, v186
	v_cvt_pkrtz_f16_f32 v148, v187, v174
	v_fmac_f32_e32 v159, v190, v161
	v_cvt_pkrtz_f16_f32 v144, v178, v179
	v_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[114:115], v[66:81]
	v_cvt_pkrtz_f16_f32 v146, v177, v183
	v_mul_f32_e32 v60, v60, v161
	v_cvt_pkrtz_f16_f32 v145, v182, v180
	v_mul_f32_e32 v62, v62, v161
	v_mul_f32_e32 v63, v63, v161
	v_mul_f32_e32 v64, v64, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[116:117], v[66:81]
	v_mul_f32_e32 v35, v35, v161
	v_mul_f32_e32 v65, v65, v161
	v_mul_f32_e32 v34, v34, v161
	v_mul_f32_e32 v36, v36, v161
	v_mul_f32_e32 v37, v37, v161
	v_mul_f32_e32 v38, v38, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[110:111], v[66:81]
	v_mul_f32_e32 v41, v41, v161
	v_mul_f32_e32 v39, v39, v161
	v_mul_f32_e32 v40, v40, v161
	v_mul_f32_e32 v42, v42, v161
	v_mul_f32_e32 v43, v43, v161
	v_mul_f32_e32 v44, v44, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[112:113], v[66:81]
	v_mul_f32_e32 v47, v47, v161
	v_mul_f32_e32 v45, v45, v161
	v_mul_f32_e32 v46, v46, v161
	v_mul_f32_e32 v48, v48, v161
	v_mul_f32_e32 v49, v49, v161
	v_mul_f32_e32 v18, v18, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[106:107], v[66:81]
	v_mul_f32_e32 v21, v21, v161
	v_mul_f32_e32 v19, v19, v161
	v_mul_f32_e32 v20, v20, v161
	v_mul_f32_e32 v22, v22, v161
	v_mul_f32_e32 v23, v23, v161
	v_mul_f32_e32 v24, v24, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[108:109], v[66:81]
	v_mul_f32_e32 v27, v27, v161
	v_mul_f32_e32 v25, v25, v161
	v_mul_f32_e32 v26, v26, v161
	v_mul_f32_e32 v28, v28, v161
	v_mul_f32_e32 v29, v29, v161
	v_mul_f32_e32 v30, v30, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[102:103], v[66:81]
	v_mul_f32_e32 v33, v33, v161
	v_mul_f32_e32 v31, v31, v161
	v_mul_f32_e32 v32, v32, v161
	v_mul_f32_e32 v2, v2, v161
	v_mul_f32_e32 v3, v3, v161
	v_mul_f32_e32 v4, v4, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[96:97], v[104:105], v[66:81]
	v_mul_f32_e32 v7, v7, v161
	v_mul_f32_e32 v5, v5, v161
	v_mul_f32_e32 v6, v6, v161
	v_mul_f32_e32 v8, v8, v161
	v_mul_f32_e32 v9, v9, v161
	v_mul_f32_e32 v10, v10, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[98:99], v[66:81]
	v_mul_f32_e32 v13, v13, v161
	v_mul_f32_e32 v11, v11, v161
	v_mul_f32_e32 v12, v12, v161
	v_mul_f32_e32 v14, v14, v161
	v_mul_f32_e32 v15, v15, v161
	v_mul_f32_e32 v16, v16, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[100:101], v[66:81]
	; iglp_opt mask(0x0000000A)
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s22, s6
	s_addc_u32 s31, s23, s7
	s_and_b32 s1, s5, 0xffff
	s_mov_b32 s12, s4
	s_or_b32 s13, s1, s29
	s_barrier
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b128 v162, v[86:89]
	buffer_load_dwordx4 v[86:89], v154, s[12:15], 0 offen
	ds_read_u16 v90, v155
	ds_read_u16 v94, v155 offset:64
	ds_read_u16 v95, v155 offset:128
	ds_read_u16 v91, v155 offset:512
	ds_read_u16 v96, v155 offset:576
	ds_read_u16 v97, v155 offset:640
	ds_read_u16 v92, v155 offset:2048
	ds_read_u16 v130, v155 offset:2112
	ds_read_u16 v131, v155 offset:2176
	ds_read_u16 v93, v155 offset:2560
	ds_read_u16 v132, v155 offset:2624
	ds_read_u16 v133, v155 offset:2688
	ds_read_u16 v134, v155 offset:4096
	ds_read_u16 v135, v155 offset:4160
	ds_read_u16 v136, v155 offset:4224
	ds_read_u16 v137, v155 offset:4608
	ds_read_u16 v138, v155 offset:4672
	ds_read_u16 v139, v155 offset:4736
	ds_read_u16 v140, v155 offset:6144
	ds_read_u16 v141, v155 offset:6208
	ds_read_u16 v150, v155 offset:6272
	ds_read_u16 v151, v155 offset:6656
	ds_read_u16 v152, v155 offset:6720
	ds_read_u16 v153, v155 offset:6784
	ds_read_u16 v161, v155 offset:320
	ds_read_u16 v178, v155 offset:384
	ds_read_u16 v186, v155 offset:448
	ds_read_u16 v169, v155 offset:256
	ds_read_u16 v187, v155 offset:192
	ds_read_u16 v173, v155 offset:832
	ds_read_u16 v179, v155 offset:896
	ds_read_u16 v174, v155 offset:768
	ds_read_u16 v175, v155 offset:2368
	ds_read_u16 v180, v155 offset:2432
	ds_read_u16 v176, v155 offset:2304
	ds_read_u16 v177, v155 offset:2880
	ds_read_u16 v181, v155 offset:2944
	ds_read_u16 v182, v155 offset:2816
	ds_read_u16 v183, v155 offset:4416
	ds_read_u16 v184, v155 offset:4480
	ds_read_u16 v185, v155 offset:4352
	ds_read_u16 v188, v155 offset:4864
	ds_read_u16 v190, v155 offset:7104
	ds_read_u16 v192, v155 offset:960
	ds_read_u16 v193, v155 offset:704
	ds_read_u16 v194, v155 offset:2496
	ds_read_u16 v195, v155 offset:2240
	ds_read_u16 v196, v155 offset:3008
	ds_read_u16 v197, v155 offset:2752
	ds_read_u16 v198, v155 offset:4544
	ds_read_u16 v199, v155 offset:4288
	ds_read_u16 v200, v155 offset:4928
	ds_read_u16 v201, v155 offset:4992
	ds_read_u16 v202, v155 offset:5056
	ds_read_u16 v203, v155 offset:4800
	ds_read_u16 v204, v155 offset:6464
	ds_read_u16 v205, v155 offset:6528
	ds_read_u16 v206, v155 offset:6592
	ds_read_u16 v207, v155 offset:6400
	ds_read_u16 v208, v155 offset:6336
	ds_read_u16 v209, v155 offset:6976
	ds_read_u16 v210, v155 offset:7040
	ds_read_u16 v211, v155 offset:6912
	ds_read_u16 v212, v155 offset:6848
	; sched_barrier mask(0x00000000)
	s_barrier
	s_setprio 0
	s_waitcnt lgkmcnt(14)
	v_perm_b32 v91, v174, v91, s30
	v_perm_b32 v90, v169, v90, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[90:91], v[144:145], v[50:65]
	v_max_f32_e32 v90, v66, v66
	v_max_f32_e32 v91, v67, v67
	v_max_f32_e32 v90, v90, v91
	v_perm_b32 v91, v182, v93, s30
	v_max3_f32 v90, v90, v68, v69
	v_max3_f32 v90, v90, v70, v71
	v_max3_f32 v169, v90, v72, v73
	v_perm_b32 v90, v176, v92, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[90:91], v[148:149], v[50:65]
	v_perm_b32 v93, v188, v137, s30
	v_perm_b32 v92, v185, v134, s30
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v91, v190, v212, s30
	v_max3_f32 v90, v169, v74, v75
	v_max3_f32 v90, v90, v76, v77
	v_max3_f32 v90, v90, v78, v79
	v_mfma_f32_32x32x8_f16 v[50:65], v[92:93], v[142:143], v[50:65]
	v_max3_f32 v90, v90, v80, v81
	ds_bpermute_b32 v169, v157, v90
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v188, v191, v90, v169
	v_mul_f32_e32 v190, 0x3e0293ee, v188
	v_fma_f32 v134, v80, s24, -v190
	v_fma_f32 v72, v72, s24, -v190
	v_fma_f32 v93, v81, s24, -v190
	v_perm_b32 v81, v211, v151, s30
	v_perm_b32 v80, v207, v140, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[80:81], v[146:147], v[50:65]
	v_fma_f32 v77, v77, s24, -v190
	v_fma_f32 v80, v67, s24, -v190
	v_fma_f32 v68, v68, s24, -v190
	v_perm_b32 v67, v173, v96, s30
	v_fma_f32 v92, v66, s24, -v190
	v_perm_b32 v66, v161, v94, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[34:49], v[66:67], v[144:145], v[34:49]
	v_fma_f32 v90, v191, s24, -v190
	v_exp_f32_e32 v161, v90
	v_exp_f32_e32 v169, v134
	v_perm_b32 v67, v177, v132, s30
	v_perm_b32 v66, v175, v130, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[34:49], v[66:67], v[148:149], v[34:49]
	v_fma_f32 v74, v74, s24, -v190
	v_fma_f32 v78, v78, s24, -v190
	v_fma_f32 v66, v71, s24, -v190
	v_exp_f32_e32 v174, v66
	v_perm_b32 v67, v200, v138, s30
	v_perm_b32 v66, v183, v135, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[34:49], v[66:67], v[142:143], v[34:49]
	v_fma_f32 v79, v79, s24, -v190
	v_exp_f32_e32 v177, v78
	v_exp_f32_e32 v183, v79
	v_perm_b32 v67, v209, v152, s30
	v_perm_b32 v66, v204, v141, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[34:49], v[66:67], v[146:147], v[34:49]
	v_perm_b32 v66, v178, v95, s30
	v_perm_b32 v67, v179, v97, s30
	v_exp_f32_e32 v173, v93
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[144:145], v[18:33]
	v_exp_f32_e32 v179, v80
	v_exp_f32_e32 v176, v77
	v_perm_b32 v67, v181, v133, s30
	v_perm_b32 v66, v180, v131, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[148:149], v[18:33]
	v_fma_f32 v69, v69, s24, -v190
	v_fma_f32 v76, v76, s24, -v190
	v_exp_f32_e32 v178, v92
	v_perm_b32 v67, v201, v139, s30
	v_perm_b32 v66, v184, v136, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[142:143], v[18:33]
	v_fma_f32 v75, v75, s24, -v190
	v_exp_f32_e32 v181, v76
	v_exp_f32_e32 v184, v75
	v_perm_b32 v67, v210, v153, s30
	v_perm_b32 v66, v205, v150, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[146:147], v[18:33]
	v_exp_f32_e32 v182, v68
	v_exp_f32_e32 v175, v74
	v_perm_b32 v67, v192, v193, s30
	v_perm_b32 v66, v186, v187, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[144:145], v[2:17]
	v_fma_f32 v73, v73, s24, -v190
	v_fma_f32 v70, v70, s24, -v190
	v_exp_f32_e32 v180, v69
	v_perm_b32 v67, v196, v197, s30
	v_perm_b32 v66, v194, v195, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[148:149], v[2:17]
	v_exp_f32_e32 v186, v73
	v_exp_f32_e32 v187, v70
	v_perm_b32 v67, v202, v203, s30
	v_perm_b32 v66, v198, v199, s30
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[142:143], v[2:17]
	v_perm_b32 v90, v206, v208, s30
	v_exp_f32_e32 v185, v72
	; iglp_opt mask(0x0000000A)
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[2:17], v[90:91], v[146:147], v[2:17]
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_and_b32 s1, s31, 0xffff
	s_mov_b32 s2, s14
	s_mov_b32 s3, s15
	s_or_b32 s1, s1, s19
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b128 v160, v[82:85] offset:8192
	buffer_load_dwordx4 v[82:85], v168, s[0:3], 0 offen
	ds_read_b128 v[150:153], v163
	ds_read_b128 v[146:149], v164
	ds_read_b128 v[142:145], v165
	ds_read_b128 v[138:141], v166
	ds_read_b128 v[134:137], v167
	ds_read_b128 v[94:97], v170
	ds_read_b128 v[130:133], v172
	ds_read_b128 v[90:93], v171
	; sched_barrier mask(0x00000000)
	s_add_u32 s22, s22, s6
	s_addc_u32 s23, s23, s7
	s_add_u32 s4, s4, s26
	s_addc_u32 s5, s5, s27
	s_add_i32 s25, s25, 32
	s_cmpk_lt_u32 s25, 0x1f80
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 v[66:81], v[150:151], v[126:127], 0
	s_mul_i32 s2, s18, 0xc0000
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s1, s8, s2
	s_addc_u32 s4, s9, s3
	s_lshl_b32 s2, s17, 14
	s_ashr_i32 s3, s2, 31
	v_mfma_f32_32x32x8_f16 v[66:81], v[152:153], v[128:129], v[66:81]
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s1, s1, s2
	s_addc_u32 s5, s4, s3
	s_ashr_i32 s29, s28, 31
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[122:123], v[66:81]
	s_lshl_b64 s[2:3], s[28:29], 2
	s_add_u32 s4, s1, s2
	s_addc_u32 s12, s5, s3
	s_mov_b32 s5, 0x5040100
	v_cvt_pkrtz_f16_f32 v154, v178, v179
	v_cvt_pkrtz_f16_f32 v155, v182, v180
	v_mul_f32_e32 v50, v50, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[148:149], v[124:125], v[66:81]
	v_or_b32_e32 v148, v189, v156
	v_lshl_add_u32 v148, v148, 1, 0
	v_mul_f32_e32 v51, v51, v161
	v_mul_f32_e32 v52, v52, v161
	v_mul_f32_e32 v53, v53, v161
	v_mul_f32_e32 v54, v54, v161
	v_mul_f32_e32 v55, v55, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[118:119], v[66:81]
	ds_read_u16 v142, v148 offset:8192
	ds_read_u16 v143, v148 offset:8256
	ds_read_u16 v149, v148 offset:8320
	ds_read_u16 v189, v148 offset:8704
	ds_read_u16 v191, v148 offset:8768
	ds_read_u16 v192, v148 offset:8832
	v_mul_f32_e32 v56, v56, v161
	v_mul_f32_e32 v57, v57, v161
	v_mul_f32_e32 v58, v58, v161
	v_mul_f32_e32 v59, v59, v161
	v_mul_f32_e32 v60, v60, v161
	v_mul_f32_e32 v61, v61, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[120:121], v[66:81]
	ds_read_u16 v144, v148 offset:10240
	ds_read_u16 v145, v148 offset:10304
	ds_read_u16 v193, v148 offset:10368
	ds_read_u16 v194, v148 offset:10752
	ds_read_u16 v195, v148 offset:10816
	ds_read_u16 v196, v148 offset:10880
	v_mul_f32_e32 v62, v62, v161
	v_mul_f32_e32 v63, v63, v161
	v_mul_f32_e32 v64, v64, v161
	v_mul_f32_e32 v65, v65, v161
	v_cvt_pkrtz_f16_f32 v152, v187, v174
	v_cvt_pkrtz_f16_f32 v153, v185, v186
	v_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[114:115], v[66:81]
	ds_read_u16 v138, v148 offset:12288
	ds_read_u16 v139, v148 offset:12352
	ds_read_u16 v197, v148 offset:12416
	ds_read_u16 v198, v148 offset:12800
	ds_read_u16 v199, v148 offset:12864
	ds_read_u16 v200, v148 offset:12928
	v_mul_f32_e32 v34, v34, v161
	v_mul_f32_e32 v35, v35, v161
	v_mul_f32_e32 v36, v36, v161
	v_mul_f32_e32 v37, v37, v161
	v_mul_f32_e32 v38, v38, v161
	v_mul_f32_e32 v39, v39, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[116:117], v[66:81]
	ds_read_u16 v140, v148 offset:14336
	ds_read_u16 v141, v148 offset:14400
	ds_read_u16 v201, v148 offset:14464
	ds_read_u16 v202, v148 offset:14848
	ds_read_u16 v203, v148 offset:14912
	ds_read_u16 v204, v148 offset:14976
	v_mul_f32_e32 v40, v40, v161
	v_mul_f32_e32 v41, v41, v161
	v_mul_f32_e32 v42, v42, v161
	v_mul_f32_e32 v43, v43, v161
	v_mul_f32_e32 v44, v44, v161
	v_mul_f32_e32 v45, v45, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[110:111], v[66:81]
	ds_read_u16 v134, v148 offset:8512
	ds_read_u16 v135, v148 offset:8576
	ds_read_u16 v205, v148 offset:8640
	ds_read_u16 v206, v148 offset:8448
	ds_read_u16 v207, v148 offset:8384
	ds_read_u16 v208, v148 offset:9024
	ds_read_u16 v209, v148 offset:9088
	ds_read_u16 v210, v148 offset:9152
	v_mul_f32_e32 v46, v46, v161
	v_mul_f32_e32 v47, v47, v161
	v_mul_f32_e32 v48, v48, v161
	v_mul_f32_e32 v49, v49, v161
	v_cvt_pkrtz_f16_f32 v150, v175, v184
	v_cvt_pkrtz_f16_f32 v151, v181, v176
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[112:113], v[66:81]
	ds_read_u16 v136, v148 offset:8960
	ds_read_u16 v137, v148 offset:8896
	ds_read_u16 v211, v148 offset:10560
	ds_read_u16 v212, v148 offset:10624
	ds_read_u16 v213, v148 offset:10688
	ds_read_u16 v214, v148 offset:10496
	ds_read_u16 v215, v148 offset:10432
	v_mul_f32_e32 v18, v18, v161
	v_mul_f32_e32 v19, v19, v161
	v_mul_f32_e32 v20, v20, v161
	v_mul_f32_e32 v21, v21, v161
	v_mul_f32_e32 v22, v22, v161
	v_mul_f32_e32 v23, v23, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[106:107], v[66:81]
	ds_read_u16 v130, v148 offset:11072
	ds_read_u16 v131, v148 offset:11136
	ds_read_u16 v216, v148 offset:11200
	ds_read_u16 v217, v148 offset:11008
	ds_read_u16 v218, v148 offset:10944
	ds_read_u16 v219, v148 offset:12608
	ds_read_u16 v220, v148 offset:12672
	ds_read_u16 v221, v148 offset:12736
	v_mul_f32_e32 v24, v24, v161
	v_mul_f32_e32 v25, v25, v161
	v_mul_f32_e32 v26, v26, v161
	v_mul_f32_e32 v27, v27, v161
	v_mul_f32_e32 v28, v28, v161
	v_mul_f32_e32 v29, v29, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[108:109], v[66:81]
	ds_read_u16 v132, v148 offset:12544
	ds_read_u16 v133, v148 offset:12480
	ds_read_u16 v222, v148 offset:13120
	ds_read_u16 v223, v148 offset:13184
	ds_read_u16 v224, v148 offset:13248
	ds_read_u16 v225, v148 offset:13056
	ds_read_u16 v226, v148 offset:12992
	ds_read_u16 v227, v148 offset:14656
	ds_read_u16 v228, v148 offset:14720
	ds_read_u16 v229, v148 offset:14784
	ds_read_u16 v230, v148 offset:14592
	ds_read_u16 v231, v148 offset:14528
	v_mul_f32_e32 v30, v30, v161
	v_mul_f32_e32 v31, v31, v161
	v_mul_f32_e32 v32, v32, v161
	v_mul_f32_e32 v33, v33, v161
	v_cvt_pkrtz_f16_f32 v146, v177, v183
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[102:103], v[66:81]
	s_waitcnt lgkmcnt(14)
	v_perm_b32 v95, v136, v189, s5
	v_perm_b32 v94, v206, v142, s5
	v_cvt_pkrtz_f16_f32 v147, v169, v173
	v_mul_f32_e32 v2, v2, v161
	v_mul_f32_e32 v3, v3, v161
	v_mul_f32_e32 v4, v4, v161
	v_mul_f32_e32 v5, v5, v161
	v_mfma_f32_32x32x8_f16 v[50:65], v[94:95], v[154:155], v[50:65]
	v_perm_b32 v95, v208, v191, s5
	v_perm_b32 v94, v134, v143, s5
	v_mul_f32_e32 v6, v6, v161
	v_mul_f32_e32 v7, v7, v161
	v_mul_f32_e32 v8, v8, v161
	v_mul_f32_e32 v9, v9, v161
	v_mul_f32_e32 v10, v10, v161
	v_mfma_f32_32x32x8_f16 v[66:81], v[96:97], v[104:105], v[66:81]
	v_perm_b32 v97, v217, v194, s5
	v_perm_b32 v96, v214, v144, s5
	v_mul_f32_e32 v11, v11, v161
	v_mul_f32_e32 v12, v12, v161
	v_mul_f32_e32 v13, v13, v161
	v_mul_f32_e32 v14, v14, v161
	v_mul_f32_e32 v15, v15, v161
	v_mfma_f32_32x32x8_f16 v[50:65], v[96:97], v[152:153], v[50:65]
	v_mul_f32_e32 v16, v16, v161
	v_mul_f32_e32 v17, v17, v161
	s_add_i32 s8, s28, 0xffffc100
	s_add_u32 s0, s0, s6
	s_addc_u32 s1, s31, s7
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s1, s19
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[98:99], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_perm_b32 v91, v225, v198, s5
	v_perm_b32 v90, v132, v138, s5
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_cmp_lt_i32 s8, 1
	v_mfma_f32_32x32x8_f16 v[34:49], v[94:95], v[154:155], v[34:49]
	v_perm_b32 v95, v216, v218, s5
	v_perm_b32 v94, v213, v215, s5
	v_mfma_f32_32x32x8_f16 v[50:65], v[90:91], v[150:151], v[50:65]
	v_perm_b32 v91, v130, v195, s5
	v_perm_b32 v90, v211, v145, s5
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[100:101], v[66:81]
	ds_read_u16 v136, v148 offset:15168
	ds_read_u16 v142, v148 offset:15232
	ds_read_u16 v144, v148 offset:15296
	ds_read_u16 v92, v148 offset:15104
	ds_read_u16 v189, v148 offset:15040
	s_waitcnt vmcnt(1)
	ds_write_b128 v162, v[86:89]
	v_perm_b32 v87, v222, v199, s5
	v_perm_b32 v86, v219, v139, s5
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v93, v92, v202, s5
	v_perm_b32 v92, v230, v140, s5
	v_perm_b32 v89, v136, v203, s5
	v_mfma_f32_32x32x8_f16 v[34:49], v[90:91], v[152:153], v[34:49]
	v_perm_b32 v91, v209, v192, s5
	v_perm_b32 v90, v135, v149, s5
	v_perm_b32 v88, v227, v141, s5
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[18:33], v[90:91], v[154:155], v[18:33]
	v_perm_b32 v91, v210, v137, s5
	v_perm_b32 v90, v205, v207, s5
	v_mfma_f32_32x32x8_f16 v[50:65], v[92:93], v[146:147], v[50:65]
	v_perm_b32 v93, v131, v196, s5
	v_perm_b32 v92, v212, v193, s5
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[92:93], v[152:153], v[18:33]
	v_mfma_f32_32x32x8_f16 v[34:49], v[86:87], v[150:151], v[34:49]
	v_perm_b32 v87, v223, v200, s5
	v_perm_b32 v86, v220, v197, s5
	v_mfma_f32_32x32x8_f16 v[2:17], v[90:91], v[154:155], v[2:17]
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[86:87], v[150:151], v[18:33]
	v_max_f32_e32 v86, v67, v67
	v_max_f32_e32 v87, v66, v66
	v_max_f32_e32 v86, v87, v86
	v_max3_f32 v86, v86, v68, v69
	v_max3_f32 v86, v86, v70, v71
	v_max3_f32 v86, v86, v72, v73
	v_max3_f32 v86, v86, v74, v75
	v_mfma_f32_32x32x8_f16 v[2:17], v[94:95], v[152:153], v[2:17]
	v_max3_f32 v86, v86, v76, v77
	v_max3_f32 v86, v86, v78, v79
	v_max3_f32 v90, v86, v80, v81
	v_perm_b32 v87, v224, v226, s5
	v_perm_b32 v86, v221, v133, s5
	ds_bpermute_b32 v91, v157, v90
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v149, v188, v90, v91
	v_mfma_f32_32x32x8_f16 v[34:49], v[88:89], v[146:147], v[34:49]
	v_perm_b32 v89, v142, v204, s5
	v_perm_b32 v88, v228, v201, s5
	v_fmac_f32_e32 v190, 0xbe0293ee, v149
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[150:151], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[88:89], v[146:147], v[18:33]
	v_perm_b32 v89, v144, v189, s5
	v_perm_b32 v88, v229, v231, s5
	ds_read_b128 v[150:153], v163
	ds_read_b128 v[192:195], v164
	ds_read_b128 v[162:165], v165
	ds_read_b128 v[196:199], v166
	ds_read_b128 v[200:203], v167
	ds_read_b128 v[142:145], v172
	ds_read_b128 v[138:141], v170
	ds_read_b128 v[134:137], v171
	s_waitcnt vmcnt(0)
	ds_write_b128 v160, v[82:85] offset:8192
	buffer_load_dwordx4 v[130:133], v168, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[88:89], v[146:147], v[2:17]
	s_barrier
	s_mov_b32 s0, 0x3e0293ee
	v_mfma_f32_32x32x8_f16 v[82:97], v[150:151], v[126:127], 0
	ds_read_u16 v126, v148 offset:8192
	ds_read_u16 v127, v148 offset:8256
	ds_read_u16 v146, v148 offset:8320
	ds_read_u16 v147, v148 offset:8704
	ds_read_u16 v150, v148 offset:8768
	ds_read_u16 v151, v148 offset:8832
	v_mfma_f32_32x32x8_f16 v[82:97], v[152:153], v[128:129], v[82:97]
	ds_read_u16 v128, v148 offset:10240
	ds_read_u16 v129, v148 offset:10304
	ds_read_u16 v152, v148 offset:10368
	ds_read_u16 v153, v148 offset:10752
	ds_read_u16 v154, v148 offset:10816
	ds_read_u16 v155, v148 offset:10880
	ds_read_u16 v166, v148 offset:12288
	ds_read_u16 v167, v148 offset:12352
	ds_read_u16 v168, v148 offset:12416
	v_mfma_f32_32x32x8_f16 v[82:97], v[192:193], v[122:123], v[82:97]
	ds_read_u16 v122, v148 offset:12800
	ds_read_u16 v123, v148 offset:12864
	ds_read_u16 v170, v148 offset:12928
	ds_read_u16 v171, v148 offset:14336
	ds_read_u16 v172, v148 offset:14400
	ds_read_u16 v188, v148 offset:14464
	ds_read_u16 v189, v148 offset:14848
	ds_read_u16 v191, v148 offset:14912
	ds_read_u16 v192, v148 offset:14976
	v_mfma_f32_32x32x8_f16 v[82:97], v[194:195], v[124:125], v[82:97]
	ds_read_u16 v124, v148 offset:8512
	ds_read_u16 v125, v148 offset:8576
	ds_read_u16 v193, v148 offset:8640
	ds_read_u16 v194, v148 offset:8448
	ds_read_u16 v195, v148 offset:8384
	ds_read_u16 v204, v148 offset:9024
	ds_read_u16 v205, v148 offset:9088
	ds_read_u16 v206, v148 offset:9152
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[118:119], v[82:97]
	ds_read_u16 v118, v148 offset:8960
	ds_read_u16 v162, v148 offset:8896
	ds_read_u16 v163, v148 offset:10560
	ds_read_u16 v207, v148 offset:10624
	ds_read_u16 v208, v148 offset:10688
	ds_read_u16 v119, v148 offset:10496
	ds_read_u16 v209, v148 offset:10432
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[120:121], v[82:97]
	ds_read_u16 v120, v148 offset:11072
	ds_read_u16 v121, v148 offset:11136
	ds_read_u16 v164, v148 offset:11200
	ds_read_u16 v165, v148 offset:11008
	ds_read_u16 v210, v148 offset:10944
	ds_read_u16 v211, v148 offset:12608
	ds_read_u16 v212, v148 offset:12672
	ds_read_u16 v213, v148 offset:12736
	v_mfma_f32_32x32x8_f16 v[82:97], v[196:197], v[114:115], v[82:97]
	ds_read_u16 v196, v148 offset:12544
	ds_read_u16 v197, v148 offset:12480
	ds_read_u16 v214, v148 offset:13120
	ds_read_u16 v215, v148 offset:13184
	ds_read_u16 v216, v148 offset:13248
	ds_read_u16 v217, v148 offset:13056
	ds_read_u16 v218, v148 offset:12992
	s_waitcnt lgkmcnt(11)
	v_perm_b32 v115, v165, v153, s5
	v_perm_b32 v114, v119, v128, s5
	v_perm_b32 v119, v204, v150, s5
	v_mfma_f32_32x32x8_f16 v[82:97], v[198:199], v[116:117], v[82:97]
	s_waitcnt lgkmcnt(1)
	v_perm_b32 v117, v217, v122, s5
	v_mul_f32_e32 v122, 0x3e0293ee, v149
	v_fma_f32 v66, v66, s0, -v122
	v_fma_f32 v67, v67, s0, -v122
	v_fma_f32 v68, v68, s0, -v122
	v_fma_f32 v69, v69, s0, -v122
	v_fma_f32 v70, v70, s0, -v122
	v_mfma_f32_32x32x8_f16 v[82:97], v[200:201], v[110:111], v[82:97]
	v_perm_b32 v111, v118, v147, s5
	v_perm_b32 v110, v194, v126, s5
	v_perm_b32 v118, v124, v127, s5
	v_fma_f32 v124, v76, s0, -v122
	v_fma_f32 v126, v77, s0, -v122
	v_exp_f32_e32 v76, v66
	v_exp_f32_e32 v77, v67
	v_mfma_f32_32x32x8_f16 v[82:97], v[202:203], v[112:113], v[82:97]
	v_fma_f32 v71, v71, s0, -v122
	v_fma_f32 v127, v78, s0, -v122
	v_cvt_pkrtz_f16_f32 v66, v76, v77
	v_exp_f32_e32 v78, v71
	v_perm_b32 v116, v196, v166, s5
	v_fma_f32 v128, v79, s0, -v122
	v_exp_f32_e32 v79, v124
	v_mfma_f32_32x32x8_f16 v[82:97], v[142:143], v[106:107], v[82:97]
	v_fma_f32 v106, v72, s0, -v122
	v_fma_f32 v107, v73, s0, -v122
	v_exp_f32_e32 v73, v68
	v_exp_f32_e32 v72, v190
	ds_read_u16 v198, v148 offset:14656
	ds_read_u16 v199, v148 offset:14720
	ds_read_u16 v219, v148 offset:14784
	ds_read_u16 v220, v148 offset:14592
	ds_read_u16 v221, v148 offset:14528
	ds_read_u16 v222, v148 offset:15168
	ds_read_u16 v223, v148 offset:15232
	ds_read_u16 v224, v148 offset:15296
	ds_read_u16 v200, v148 offset:15104
	ds_read_u16 v201, v148 offset:15040
	v_fma_f32 v80, v80, s0, -v122
	v_mfma_f32_32x32x8_f16 v[82:97], v[144:145], v[108:109], v[82:97]
	v_fma_f32 v108, v74, s0, -v122
	v_exp_f32_e32 v74, v69
	v_mul_f32_e32 v50, v50, v72
	v_mul_f32_e32 v51, v51, v72
	v_mul_f32_e32 v52, v52, v72
	v_cvt_pkrtz_f16_f32 v67, v73, v74
	v_mul_f32_e32 v53, v53, v72
	v_mfma_f32_32x32x8_f16 v[82:97], v[138:139], v[102:103], v[82:97]
	v_mul_f32_e32 v54, v54, v72
	v_mul_f32_e32 v55, v55, v72
	v_mul_f32_e32 v56, v56, v72
	v_mul_f32_e32 v57, v57, v72
	v_mul_f32_e32 v58, v58, v72
	v_mul_f32_e32 v59, v59, v72
	v_mul_f32_e32 v60, v60, v72
	v_mul_f32_e32 v61, v61, v72
	v_mul_f32_e32 v62, v62, v72
	v_mul_f32_e32 v63, v63, v72
	v_mul_f32_e32 v64, v64, v72
	v_mul_f32_e32 v65, v65, v72
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[110:111], v[66:67], v[50:65]
	v_fma_f32 v109, v75, s0, -v122
	v_exp_f32_e32 v75, v70
	v_exp_f32_e32 v103, v107
	v_fma_f32 v102, v81, s0, -v122
	v_exp_f32_e32 v81, v109
	v_cvt_pkrtz_f16_f32 v68, v75, v78
	v_mul_f32_e32 v34, v34, v72
	v_mfma_f32_32x32x8_f16 v[82:97], v[140:141], v[104:105], v[82:97]
	v_exp_f32_e32 v105, v106
	v_exp_f32_e32 v104, v108
	v_mul_f32_e32 v35, v35, v72
	v_mul_f32_e32 v36, v36, v72
	v_cvt_pkrtz_f16_f32 v69, v105, v103
	v_mul_f32_e32 v37, v37, v72
	v_mul_f32_e32 v38, v38, v72
	v_mfma_f32_32x32x8_f16 v[50:65], v[114:115], v[68:69], v[50:65]
	v_mul_f32_e32 v39, v39, v72
	v_mul_f32_e32 v40, v40, v72
	v_mul_f32_e32 v41, v41, v72
	v_mul_f32_e32 v42, v42, v72
	v_mul_f32_e32 v43, v43, v72
	v_mul_f32_e32 v44, v44, v72
	v_mul_f32_e32 v45, v45, v72
	v_mfma_f32_32x32x8_f16 v[82:97], v[134:135], v[98:99], v[82:97]
	v_exp_f32_e32 v99, v126
	v_mul_f32_e32 v46, v46, v72
	v_mul_f32_e32 v47, v47, v72
	v_mul_f32_e32 v48, v48, v72
	v_mul_f32_e32 v49, v49, v72
	v_cvt_pkrtz_f16_f32 v70, v104, v81
	v_cvt_pkrtz_f16_f32 v71, v79, v99
	v_mfma_f32_32x32x8_f16 v[34:49], v[118:119], v[66:67], v[34:49]
	v_perm_b32 v109, v120, v154, s5
	v_perm_b32 v108, v163, v129, s5
	s_waitcnt lgkmcnt(1)
	v_perm_b32 v113, v200, v189, s5
	v_perm_b32 v112, v220, v171, s5
	v_exp_f32_e32 v98, v80
	v_exp_f32_e32 v80, v102
	v_mul_f32_e32 v18, v18, v72
	v_mfma_f32_32x32x8_f16 v[50:65], v[116:117], v[70:71], v[50:65]
	v_mul_f32_e32 v19, v19, v72
	v_cvt_pkrtz_f16_f32 v107, v98, v80
	v_mul_f32_e32 v20, v20, v72
	v_mul_f32_e32 v21, v21, v72
	v_mul_f32_e32 v22, v22, v72
	v_mul_f32_e32 v23, v23, v72
	v_mul_f32_e32 v24, v24, v72
	v_mfma_f32_32x32x8_f16 v[82:97], v[136:137], v[100:101], v[82:97]
	v_exp_f32_e32 v100, v127
	v_exp_f32_e32 v101, v128
	v_mul_f32_e32 v25, v25, v72
	v_mul_f32_e32 v26, v26, v72
	v_mul_f32_e32 v27, v27, v72
	v_cvt_pkrtz_f16_f32 v106, v100, v101
	v_mul_f32_e32 v28, v28, v72
	v_mfma_f32_32x32x8_f16 v[34:49], v[108:109], v[68:69], v[34:49]
	v_perm_b32 v109, v214, v123, s5
	v_perm_b32 v108, v211, v167, s5
	v_mul_f32_e32 v29, v29, v72
	v_mul_f32_e32 v30, v30, v72
	v_mul_f32_e32 v31, v31, v72
	v_mul_f32_e32 v32, v32, v72
	v_mul_f32_e32 v33, v33, v72
	v_mfma_f32_32x32x8_f16 v[50:65], v[112:113], v[106:107], v[50:65]
	v_perm_b32 v113, v205, v151, s5
	v_perm_b32 v112, v125, v146, s5
	v_perm_b32 v111, v222, v191, s5
	v_perm_b32 v110, v198, v172, s5
	v_mul_f32_e32 v2, v2, v72
	v_mul_f32_e32 v3, v3, v72
	v_mul_f32_e32 v4, v4, v72
	v_mfma_f32_32x32x8_f16 v[34:49], v[108:109], v[70:71], v[34:49]
	v_perm_b32 v109, v121, v155, s5
	v_perm_b32 v108, v207, v152, s5
	v_mul_f32_e32 v5, v5, v72
	v_mul_f32_e32 v6, v6, v72
	v_mul_f32_e32 v7, v7, v72
	v_mul_f32_e32 v8, v8, v72
	v_mul_f32_e32 v9, v9, v72
	v_mfma_f32_32x32x8_f16 v[18:33], v[112:113], v[66:67], v[18:33]
	v_mul_f32_e32 v10, v10, v72
	v_mul_f32_e32 v11, v11, v72
	v_mul_f32_e32 v12, v12, v72
	v_mul_f32_e32 v13, v13, v72
	v_mul_f32_e32 v14, v14, v72
	v_mul_f32_e32 v15, v15, v72
	v_mul_f32_e32 v16, v16, v72
	v_mfma_f32_32x32x8_f16 v[34:49], v[110:111], v[106:107], v[34:49]
	v_perm_b32 v111, v206, v162, s5
	v_perm_b32 v110, v193, v195, s5
	v_mul_f32_e32 v17, v17, v72
	v_and_b32_e32 v102, 0x100, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v102
	v_add_f32_e32 v102, v178, v179
	v_add_f32_e32 v102, v102, v182
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[68:69], v[18:33]
	v_perm_b32 v109, v215, v170, s5
	v_perm_b32 v108, v212, v168, s5
	v_add_f32_e32 v102, v102, v180
	v_add_f32_e32 v102, v102, v187
	v_add_f32_e32 v102, v102, v174
	v_add_f32_e32 v102, v102, v185
	v_add_f32_e32 v102, v102, v186
	v_mfma_f32_32x32x8_f16 v[2:17], v[110:111], v[66:67], v[2:17]
	v_perm_b32 v67, v164, v210, s5
	v_perm_b32 v66, v208, v209, s5
	v_max_f32_e32 v110, v83, v83
	v_max_f32_e32 v111, v82, v82
	v_max_f32_e32 v110, v111, v110
	v_add_f32_e32 v102, v102, v175
	v_add_f32_e32 v102, v102, v184
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[70:71], v[18:33]
	v_perm_b32 v109, v223, v192, s5
	v_perm_b32 v108, v199, v188, s5
	v_add_f32_e32 v102, v102, v181
	v_add_f32_e32 v102, v102, v176
	v_add_f32_e32 v102, v102, v177
	v_add_f32_e32 v102, v102, v183
	v_and_b32_e32 v114, 0xa0, v158
	v_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[68:69], v[2:17]
	v_max3_f32 v66, v110, v84, v85
	v_max3_f32 v66, v66, v86, v87
	v_max3_f32 v66, v66, v88, v89
	v_max3_f32 v66, v66, v90, v91
	v_max3_f32 v66, v66, v92, v93
	v_max3_f32 v66, v66, v94, v95
	v_max3_f32 v68, v66, v96, v97
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[106:107], v[18:33]
	v_perm_b32 v109, v216, v218, s5
	v_perm_b32 v108, v213, v197, s5
	ds_bpermute_b32 v69, v157, v68
	s_waitcnt lgkmcnt(1)
	v_perm_b32 v67, v224, v201, s5
	v_perm_b32 v66, v219, v221, s5
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[2:17], v[108:109], v[70:71], v[2:17]
	v_add_f32_e32 v70, v102, v169
	v_max3_f32 v102, v149, v68, v69
	v_mul_f32_e32 v68, 0xbe0293ee, v102
	v_add_f32_e32 v108, v70, v173
	v_fmamk_f32 v69, v84, 0x3e0293ee, v68
	v_fmamk_f32 v70, v85, 0x3e0293ee, v68
	v_fmamk_f32 v71, v86, 0x3e0293ee, v68
	v_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[106:107], v[2:17]
	v_fmamk_f32 v66, v82, 0x3e0293ee, v68
	v_fmamk_f32 v67, v83, 0x3e0293ee, v68
	v_fmamk_f32 v82, v87, 0x3e0293ee, v68
	v_fmamk_f32 v83, v88, 0x3e0293ee, v68
	v_fmamk_f32 v84, v89, 0x3e0293ee, v68
	v_fmamk_f32 v85, v90, 0x3e0293ee, v68
	v_fmamk_f32 v86, v91, 0x3e0293ee, v68
	v_fmamk_f32 v87, v92, 0x3e0293ee, v68
	v_fmac_f32_e32 v122, 0xbe0293ee, v102
	v_fmamk_f32 v88, v93, 0x3e0293ee, v68
	v_fmamk_f32 v89, v94, 0x3e0293ee, v68
	v_fmamk_f32 v90, v95, 0x3e0293ee, v68
	v_fmamk_f32 v91, v96, 0x3e0293ee, v68
	v_fmac_f32_e32 v68, 0x3e0293ee, v97
	v_exp_f32_e32 v97, v82
	v_exp_f32_e32 v106, v83
	v_exp_f32_e32 v107, v84
	v_exp_f32_e32 v109, v85
	v_exp_f32_e32 v110, v86
	v_exp_f32_e32 v111, v87
	v_exp_f32_e32 v113, v122
	s_waitcnt vmcnt(0)
	ds_write_b128 v160, v[130:133] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v82, v148 offset:8192
	ds_read_u16 v86, v148 offset:8256
	ds_read_u16 v115, v148 offset:8320
	ds_read_u16 v83, v148 offset:8704
	ds_read_u16 v87, v148 offset:8768
	ds_read_u16 v116, v148 offset:8832
	ds_read_u16 v84, v148 offset:10240
	ds_read_u16 v117, v148 offset:10304
	ds_read_u16 v118, v148 offset:10368
	ds_read_u16 v85, v148 offset:10752
	ds_read_u16 v119, v148 offset:10816
	ds_read_u16 v120, v148 offset:10880
	ds_read_u16 v121, v148 offset:12288
	ds_read_u16 v122, v148 offset:12352
	ds_read_u16 v123, v148 offset:12416
	ds_read_u16 v124, v148 offset:12800
	ds_read_u16 v125, v148 offset:12864
	ds_read_u16 v126, v148 offset:12928
	ds_read_u16 v127, v148 offset:14336
	ds_read_u16 v128, v148 offset:14400
	ds_read_u16 v129, v148 offset:14464
	ds_read_u16 v130, v148 offset:14848
	ds_read_u16 v131, v148 offset:14912
	ds_read_u16 v132, v148 offset:14976
	ds_read_u16 v133, v148 offset:8512
	ds_read_u16 v134, v148 offset:8576
	ds_read_u16 v135, v148 offset:8640
	ds_read_u16 v136, v148 offset:8448
	ds_read_u16 v137, v148 offset:8384
	ds_read_u16 v138, v148 offset:9024
	ds_read_u16 v139, v148 offset:9088
	ds_read_u16 v140, v148 offset:9152
	ds_read_u16 v141, v148 offset:8960
	ds_read_u16 v142, v148 offset:8896
	ds_read_u16 v143, v148 offset:10560
	ds_read_u16 v144, v148 offset:10624
	ds_read_u16 v145, v148 offset:10688
	ds_read_u16 v146, v148 offset:10496
	ds_read_u16 v147, v148 offset:10432
	ds_read_u16 v149, v148 offset:11072
	ds_read_u16 v150, v148 offset:11136
	ds_read_u16 v151, v148 offset:11200
	ds_read_u16 v152, v148 offset:11008
	ds_read_u16 v153, v148 offset:10944
	ds_read_u16 v154, v148 offset:12608
	ds_read_u16 v155, v148 offset:12672
	ds_read_u16 v158, v148 offset:12736
	ds_read_u16 v160, v148 offset:12544
	ds_read_u16 v162, v148 offset:12480
	ds_read_u16 v163, v148 offset:13120
	ds_read_u16 v164, v148 offset:13184
	ds_read_u16 v165, v148 offset:13248
	v_exp_f32_e32 v92, v66
	v_exp_f32_e32 v93, v67
	v_exp_f32_e32 v94, v69
	v_exp_f32_e32 v95, v70
	s_waitcnt lgkmcnt(14)
	v_perm_b32 v83, v141, v83, s5
	v_perm_b32 v82, v136, v82, s5
	v_exp_f32_e32 v96, v71
	v_cvt_pkrtz_f16_f32 v70, v92, v93
	v_cvt_pkrtz_f16_f32 v71, v94, v95
	v_mul_f32_e32 v50, v50, v113
	v_mul_f32_e32 v51, v51, v113
	v_mul_f32_e32 v52, v52, v113
	v_mul_f32_e32 v53, v53, v113
	v_mul_f32_e32 v54, v54, v113
	v_mul_f32_e32 v55, v55, v113
	v_mul_f32_e32 v56, v56, v113
	v_mul_f32_e32 v57, v57, v113
	v_mul_f32_e32 v58, v58, v113
	v_mul_f32_e32 v59, v59, v113
	v_mul_f32_e32 v60, v60, v113
	v_mul_f32_e32 v61, v61, v113
	v_mul_f32_e32 v62, v62, v113
	v_mul_f32_e32 v63, v63, v113
	v_mul_f32_e32 v64, v64, v113
	v_mul_f32_e32 v65, v65, v113
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[82:83], v[70:71], v[50:65]
	s_waitcnt lgkmcnt(9)
	v_perm_b32 v83, v152, v85, s5
	v_perm_b32 v82, v146, v84, s5
	v_exp_f32_e32 v112, v68
	v_cvt_pkrtz_f16_f32 v68, v96, v97
	v_cvt_pkrtz_f16_f32 v69, v106, v107
	ds_read_u16 v84, v148 offset:13056
	ds_read_u16 v136, v148 offset:12992
	ds_read_u16 v141, v148 offset:14656
	ds_read_u16 v146, v148 offset:14720
	ds_read_u16 v152, v148 offset:14784
	v_exp_f32_e32 v88, v88
	v_mfma_f32_32x32x8_f16 v[50:65], v[82:83], v[68:69], v[50:65]
	s_waitcnt lgkmcnt(4)
	v_perm_b32 v83, v84, v124, s5
	v_perm_b32 v82, v160, v121, s5
	v_cvt_pkrtz_f16_f32 v66, v109, v110
	v_cvt_pkrtz_f16_f32 v67, v111, v88
	ds_read_u16 v85, v148 offset:14592
	ds_read_u16 v166, v148 offset:14528
	v_perm_b32 v87, v138, v87, s5
	v_perm_b32 v86, v133, v86, s5
	v_mfma_f32_32x32x8_f16 v[50:65], v[82:83], v[66:67], v[50:65]
	ds_read_u16 v84, v148 offset:15104
	ds_read_u16 v121, v148 offset:15168
	ds_read_u16 v124, v148 offset:15232
	ds_read_u16 v160, v148 offset:15296
	v_exp_f32_e32 v89, v89
	v_exp_f32_e32 v90, v90
	v_exp_f32_e32 v91, v91
	s_waitcnt lgkmcnt(3)
	v_perm_b32 v83, v84, v130, s5
	v_perm_b32 v82, v85, v127, s5
	v_mul_f32_e32 v34, v34, v113
	v_mul_f32_e32 v35, v35, v113
	v_mul_f32_e32 v36, v36, v113
	v_mul_f32_e32 v37, v37, v113
	v_mul_f32_e32 v38, v38, v113
	v_mul_f32_e32 v39, v39, v113
	v_mul_f32_e32 v40, v40, v113
	v_mul_f32_e32 v41, v41, v113
	v_mul_f32_e32 v42, v42, v113
	v_mul_f32_e32 v43, v43, v113
	v_mul_f32_e32 v44, v44, v113
	v_mul_f32_e32 v45, v45, v113
	v_mul_f32_e32 v46, v46, v113
	v_mul_f32_e32 v47, v47, v113
	v_mul_f32_e32 v48, v48, v113
	v_mul_f32_e32 v49, v49, v113
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[34:49], v[86:87], v[70:71], v[34:49]
	v_cvt_pkrtz_f16_f32 v84, v89, v90
	v_cvt_pkrtz_f16_f32 v85, v91, v112
	v_mul_f32_e32 v18, v18, v113
	v_mul_f32_e32 v19, v19, v113
	v_mul_f32_e32 v20, v20, v113
	v_mul_f32_e32 v21, v21, v113
	v_mul_f32_e32 v22, v22, v113
	v_mfma_f32_32x32x8_f16 v[50:65], v[82:83], v[84:85], v[50:65]
	v_perm_b32 v83, v149, v119, s5
	v_perm_b32 v82, v143, v117, s5
	v_mul_f32_e32 v23, v23, v113
	v_mul_f32_e32 v24, v24, v113
	v_mul_f32_e32 v25, v25, v113
	v_mul_f32_e32 v26, v26, v113
	v_mul_f32_e32 v27, v27, v113
	v_mfma_f32_32x32x8_f16 v[34:49], v[82:83], v[68:69], v[34:49]
	v_add_f32_e32 v82, v76, v77
	v_perm_b32 v77, v163, v125, s5
	v_perm_b32 v76, v154, v122, s5
	v_add_f32_e32 v73, v73, v82
	v_add_f32_e32 v73, v74, v73
	v_add_f32_e32 v73, v75, v73
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v75, v121, v131, s5
	v_mfma_f32_32x32x8_f16 v[34:49], v[76:77], v[66:67], v[34:49]
	v_perm_b32 v77, v139, v116, s5
	v_perm_b32 v76, v134, v115, s5
	v_perm_b32 v74, v141, v128, s5
	v_add_f32_e32 v73, v78, v73
	v_mul_f32_e32 v28, v28, v113
	v_mul_f32_e32 v29, v29, v113
	v_mul_f32_e32 v30, v30, v113
	v_mul_f32_e32 v31, v31, v113
	v_mul_f32_e32 v32, v32, v113
	v_mul_f32_e32 v33, v33, v113
	v_add_f32_e32 v73, v105, v73
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[76:77], v[70:71], v[18:33]
	v_add_f32_e32 v76, v92, v93
	v_add_f32_e32 v73, v103, v73
	v_add_f32_e32 v76, v94, v76
	v_add_f32_e32 v73, v104, v73
	v_add_f32_e32 v76, v95, v76
	v_add_f32_e32 v73, v81, v73
	v_add_f32_e32 v76, v96, v76
	v_mfma_f32_32x32x8_f16 v[34:49], v[74:75], v[84:85], v[34:49]
	v_perm_b32 v75, v150, v120, s5
	v_perm_b32 v74, v144, v118, s5
	v_add_f32_e32 v73, v79, v73
	v_add_f32_e32 v79, v97, v76
	v_perm_b32 v77, v140, v142, s5
	v_perm_b32 v76, v135, v137, s5
	v_mul_f32_e32 v2, v2, v113
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[68:69], v[18:33]
	v_mul_f32_e32 v3, v3, v113
	v_mul_f32_e32 v4, v4, v113
	v_mul_f32_e32 v5, v5, v113
	v_mul_f32_e32 v6, v6, v113
	v_mul_f32_e32 v7, v7, v113
	v_mul_f32_e32 v8, v8, v113
	v_mul_f32_e32 v9, v9, v113
	v_mul_f32_e32 v10, v10, v113
	v_mul_f32_e32 v11, v11, v113
	v_mul_f32_e32 v12, v12, v113
	v_mul_f32_e32 v13, v13, v113
	v_mul_f32_e32 v14, v14, v113
	v_mul_f32_e32 v15, v15, v113
	v_mul_f32_e32 v16, v16, v113
	v_mul_f32_e32 v17, v17, v113
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[2:17], v[76:77], v[70:71], v[2:17]
	v_perm_b32 v75, v164, v126, s5
	v_perm_b32 v74, v155, v123, s5
	v_perm_b32 v71, v151, v153, s5
	v_perm_b32 v70, v145, v147, s5
	ds_read_u16 v148, v148 offset:15040
	v_add_f32_e32 v73, v99, v73
	v_add_f32_e32 v73, v100, v73
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[66:67], v[18:33]
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v75, v124, v132, s5
	v_perm_b32 v74, v146, v129, s5
	v_add_f32_e32 v73, v101, v73
	v_add_f32_e32 v73, v98, v73
	ds_bpermute_b32 v86, v157, v108
	v_add_f32_e32 v73, v80, v73
	ds_bpermute_b32 v78, v157, v73
	v_mfma_f32_32x32x8_f16 v[2:17], v[70:71], v[68:69], v[2:17]
	v_perm_b32 v69, v165, v136, s5
	v_perm_b32 v68, v158, v162, s5
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v86, v108, v86
	v_fmac_f32_e32 v86, v159, v161
	v_or3_b32 v1, v114, v156, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[84:85], v[18:33]
	v_add_f32_e32 v74, v106, v79
	v_add_f32_e32 v74, v107, v74
	v_add_f32_e32 v74, v109, v74
	v_add_f32_e32 v74, v110, v74
	v_add_f32_e32 v74, v111, v74
	v_add_f32_e32 v70, v88, v74
	v_add_f32_e32 v70, v89, v70
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[66:67], v[2:17]
	v_perm_b32 v69, v160, v148, s5
	v_perm_b32 v68, v152, v166, s5
	v_add_f32_e32 v70, v90, v70
	v_add_f32_e32 v70, v91, v70
	v_add_f32_e32 v70, v112, v70
	ds_bpermute_b32 v66, v157, v70
	v_add_f32_e32 v67, v73, v78
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[84:85], v[2:17]
	v_fmac_f32_e32 v67, v86, v72
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v66, v70, v66
	v_fmac_f32_e32 v66, v67, v113
	v_lshl_add_u32 v67, v1, 2, 0
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	s_mov_b32 s6, 0x800000
	v_cmp_gt_f32_e32 vcc, s6, v66
	s_nop 1
	v_cndmask_b32_e64 v70, 0, 32, vcc
	v_ldexp_f32 v70, v66, v70
	v_log_f32_e32 v70, v70
	v_mov_b32_e32 v69, 0x42000000
	v_or_b32_e32 v68, s28, v1
	s_movk_i32 s5, 0x4000
	v_cndmask_b32_e32 v69, 0, v69, vcc
	v_cmp_gt_i32_e64 s[8:9], s5, v68
	v_sub_f32_e32 v68, v70, v69
	v_add_f32_e32 v68, v102, v68
	ds_write_b32 v67, v68
	v_mov_b32_e32 v68, 2
	v_lshlrev_b32_sdwa v68, v68, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v69, 0, v68
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v69, v69
	s_sub_i32 s5, 0x4000, s28
	v_cmp_lt_i32_sdwa s[14:15], v0, s5 src0_sel:BYTE_0 src1_sel:DWORD
	v_bfrev_b32_e32 v70, 1
	s_and_b64 vcc, s[0:1], s[14:15]
	s_and_b32 s5, s12, 0xffff
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	v_cndmask_b32_e32 v68, v70, v68, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v69, v68, s[4:7], 0 offen
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_9:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v66
	s_nop 1
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v66, v69
	v_log_f32_e32 v69, v69
	v_mov_b32_e32 v68, 0x42000000
	v_cndmask_b32_e32 v68, 0, v68, vcc
	s_and_b32 s5, s12, 0xffff
	v_sub_f32_e32 v68, v69, v68
	v_add_f32_e32 v68, v102, v68
	ds_write_b32 v67, v68
	v_mov_b32_e32 v67, 2
	v_lshlrev_b32_sdwa v67, v67, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v68, 0, v67
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v68, v68
	v_bfrev_b32_e32 v69, 1
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e64 v67, v69, v67, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v68, v67, s[4:7], 0 offen
.LBB0_10:                               ; %.critedge
	v_div_scale_f32 v67, s[0:1], v66, v66, 1.0
	v_rcp_f32_e32 v67, v67
	v_div_scale_f32 v68, vcc, 1.0, v66, 1.0
	v_mov_b32_e32 v69, v16
	v_mul_f32_e32 v67, v68, v67
	s_nop 1
	v_div_fmas_f32 v67, 0, 0, v67
	v_div_fixup_f32 v66, v67, v66, 1.0
	v_fma_mixlo_f16 v67, v66, v17, 0
	v_mov_b32_e32 v68, v15
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
	s_mul_i32 s0, s20, s18
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	s_ashr_i32 s1, s0, 31
	v_cvt_f16_f32_e32 v63, v3
	v_cvt_f16_f32_e32 v64, v2
	v_mov_b32_e32 v2, v59
	v_mov_b32_e32 v3, v60
	s_lshl_b64 s[0:1], s[0:1], 1
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s21, s17
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
	s_mul_i32 s0, s16, s28
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v2, v2
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	v_lshrrev_b32_e32 v0, 3, v0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s16, 0x3fff
	v_fma_mixlo_f16 v53, v66, v53, 0
	v_fma_mixlo_f16 v50, v66, v50, 0
	v_and_b32_e32 v51, 4, v0
	v_mul_lo_u32 v52, s16, v1
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
		.amdhsa_next_free_vgpr 232
		.amdhsa_next_free_sgpr 38
		.amdhsa_accum_offset 232
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
	.set attn_fwd.num_vgpr, 232
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 38
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11508
; TotalNumSgprs: 44
; NumVgprs: 232
; NumAgprs: 0
; TotalNumVgprs: 232
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 28
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 232
; AccumOffset: 232
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 57
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
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     232
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
