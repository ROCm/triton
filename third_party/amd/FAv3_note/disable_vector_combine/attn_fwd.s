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
	s_mul_i32 s20, s12, s18
	s_ashr_i32 s21, s20, 31
	s_lshl_b32 s34, s16, 8
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s12, s2, s20
	s_mul_i32 s2, s13, s17
	s_addc_u32 s16, s3, s21
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshrrev_b32_e32 v35, 4, v0
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s34
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	v_or_b32_e32 v2, 0x60, v35
	s_addc_u32 s13, s16, s3
	s_ashr_i32 s3, s2, 31
	v_or_b32_e32 v11, s34, v2
	s_lshl_b32 s16, s14, 6
	v_mul_lo_u32 v12, s14, v2
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v2, 3, v0
	v_or_b32_e32 v3, 0xa0, v35
	s_add_u32 s20, s12, s2
	v_and_b32_e32 v34, 0x78, v2
	s_mul_i32 s36, s15, s18
	v_or_b32_e32 v19, s34, v3
	v_mul_lo_u32 v20, s14, v3
	s_addc_u32 s12, s13, s3
	v_mad_u64_u32 v[2:3], s[2:3], s14, v35, v[34:35]
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[2:3], s[36:37], 1
	s_add_u32 s13, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s38, s24, s17
	s_addc_u32 s15, s5, s3
	s_ashr_i32 s39, s38, 31
	s_lshl_b64 s[2:3], s[38:39], 1
	s_add_u32 s28, s13, s2
	s_mul_i32 s2, s26, s18
	s_addc_u32 s43, s15, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s27, s17
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	v_or_b32_e32 v1, 32, v35
	v_or_b32_e32 v5, s34, v35
	s_add_u32 s24, s6, s2
	s_movk_i32 s2, 0x4000
	v_or_b32_e32 v6, s34, v1
	v_mul_lo_u32 v7, s14, v1
	v_add_u32_e32 v13, s16, v2
	s_addc_u32 s33, s7, s3
	s_and_b32 s3, s14, 0x3fff
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v30, 1
	v_cmp_gt_i32_e32 vcc, s2, v5
	v_or_b32_e32 v10, 64, v5
	s_bitset1_b32 s3, 14
	v_cndmask_b32_e32 v14, v30, v2, vcc
	v_add_lshl_u32 v2, v7, v34, 1
	v_cmp_gt_i32_e32 vcc, s2, v6
	v_add_u32_e32 v29, s16, v13
	s_and_b32 s6, s12, 0xffff
	s_lshl_b32 s3, s3, 16
	v_cndmask_b32_e32 v15, v30, v2, vcc
	v_lshlrev_b32_e32 v13, 1, v13
	v_cmp_gt_i32_e32 vcc, s2, v10
	v_or_b32_e32 v18, 0x80, v5
	s_or_b32 s21, s6, s3
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v21, v30, v13, vcc
	v_add_lshl_u32 v10, v12, v34, 1
	v_cmp_gt_i32_e32 vcc, s2, v11
	v_or_b32_e32 v4, 0xe0, v35
	s_nop 0
	v_cndmask_b32_e32 v22, v30, v10, vcc
	buffer_load_dwordx4 v[10:13], v21, s[20:23], 0 offen
                                        ; kill: killed $vgpr21
	v_lshlrev_b32_e32 v21, 1, v29
	v_cmp_gt_i32_e32 vcc, s2, v18
	v_or_b32_e32 v26, 0xc0, v5
	v_or_b32_e32 v27, s34, v4
	v_mul_lo_u32 v28, s14, v4
	buffer_load_dwordx4 v[2:5], v14, s[20:23], 0 offen
	v_cndmask_b32_e32 v31, v30, v21, vcc
	v_add_lshl_u32 v18, v20, v34, 1
	v_cmp_gt_i32_e32 vcc, s2, v19
	buffer_load_dwordx4 v[6:9], v15, s[20:23], 0 offen
	s_nop 0
	v_cndmask_b32_e32 v32, v30, v18, vcc
	v_add_lshl_u32 v29, v29, s16, 1
	v_cmp_gt_i32_e32 vcc, s2, v26
	s_nop 1
	v_cndmask_b32_e32 v36, v30, v29, vcc
	v_add_lshl_u32 v26, v28, v34, 1
	v_cmp_gt_i32_e32 vcc, s2, v27
                                        ; kill: killed $vgpr14
                                        ; kill: killed $vgpr15
	buffer_load_dwordx4 v[14:17], v22, s[20:23], 0 offen
	s_nop 0
	v_cndmask_b32_e32 v37, v30, v26, vcc
                                        ; kill: killed $vgpr22
	buffer_load_dwordx4 v[18:21], v31, s[20:23], 0 offen
	buffer_load_dwordx4 v[22:25], v32, s[20:23], 0 offen
                                        ; kill: killed $vgpr32
                                        ; kill: killed $vgpr31
	buffer_load_dwordx4 v[26:29], v36, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[30:33], v37, s[20:23], 0 offen
                                        ; kill: killed $vgpr36
	v_and_b32_e32 v36, 1, v0
	v_cmp_eq_u32_e64 s[2:3], 0, v36
	v_and_b32_e32 v36, 2, v0
	v_cmp_eq_u32_e64 s[6:7], 0, v36
	v_and_b32_e32 v36, 4, v0
	v_cmp_eq_u32_e64 s[12:13], 0, v36
	v_and_b32_e32 v36, 8, v0
	v_cmp_eq_u32_e64 s[14:15], 0, v36
	v_and_b32_e32 v36, 0x80, v0
	v_and_b32_e32 v39, 0x100, v0
	v_lshrrev_b32_e32 v41, 1, v0
	v_lshrrev_b32_e32 v40, 1, v36
	scratch_store_dword off, v39, off offset:144 ; 4-byte Folded Spill
	v_or_b32_e32 v36, v36, v39
	v_and_b32_e32 v39, 56, v41
	v_xor_b32_e32 v39, v39, v34
	v_xor_b32_e32 v39, v39, v40
                                        ; kill: killed $vgpr37
	v_mul_lo_u32 v37, s25, v35
	v_lshl_add_u32 v39, v39, 1, 0
	v_lshlrev_b32_e32 v35, 8, v35
	v_add_u32_e32 v245, v39, v35
	v_and_b32_e32 v54, 31, v0
	s_movk_i32 s19, 0xe0
	s_barrier
	scratch_store_dword off, v40, off offset:140 ; 4-byte Folded Spill
	s_load_dword s35, s[0:1], 0x48
	v_and_b32_e32 v50, 16, v0
	v_lshrrev_b32_e32 v51, 3, v0
	v_lshrrev_b32_e32 v36, 3, v36
	v_mul_lo_u32 v38, s25, v1
	v_lshrrev_b32_e32 v1, 3, v50
	v_and_or_b32 v52, v51, 12, v36
	v_or_b32_e32 v36, v52, v1
                                        ; kill: killed $sgpr20_sgpr21
	v_bfe_i32 v102, v0, 0, 1
	v_bfe_i32 v103, v0, 1, 1
	v_bfe_i32 v104, v0, 2, 1
	s_lshl_b32 s40, s25, 6
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s42, s35, 6
	v_bfe_i32 v53, v0, 3, 1
	s_movk_i32 s16, 0x100
	s_waitcnt vmcnt(9)
	ds_write_b128 v245, v[10:13] offset:16384
	s_waitcnt vmcnt(8)
	ds_write_b128 v245, v[2:5]
	v_bfe_u32 v2, v0, 5, 1
	v_and_b32_e32 v4, 15, v0
	v_and_or_b32 v3, v41, s19, v54
	s_waitcnt vmcnt(7)
	ds_write_b128 v245, v[6:9] offset:8192
	v_xor_b32_e32 v5, v2, v4
	v_or_b32_e32 v6, 2, v2
	v_or_b32_e32 v7, 4, v2
	v_xor_b32_e32 v6, v6, v4
	v_xor_b32_e32 v7, v7, v4
	v_or_b32_e32 v8, 6, v2
	v_or_b32_e32 v9, 8, v2
	v_or_b32_e32 v10, 10, v2
	v_or_b32_e32 v11, 12, v2
	v_or_b32_e32 v2, 14, v2
	v_lshl_add_u32 v3, v3, 8, 0
	v_lshlrev_b32_e32 v12, 4, v5
	s_waitcnt vmcnt(6)
	ds_write_b128 v245, v[14:17] offset:24576
	v_xor_b32_e32 v8, v8, v4
	v_xor_b32_e32 v9, v9, v4
	v_xor_b32_e32 v10, v10, v4
	v_xor_b32_e32 v11, v11, v4
	v_xor_b32_e32 v2, v2, v4
	v_add_u32_e32 v4, v3, v12
	v_lshlrev_b32_e32 v13, 4, v6
	v_lshlrev_b32_e32 v14, 4, v7
	s_waitcnt vmcnt(5)
	ds_write_b128 v245, v[18:21] offset:32768
	s_waitcnt vmcnt(4)
	ds_write_b128 v245, v[22:25] offset:40960
	s_waitcnt vmcnt(3)
	ds_write_b128 v245, v[26:29] offset:49152
	s_waitcnt vmcnt(2)
	ds_write_b128 v245, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v5, v3, v13
	ds_read_b128 v[158:161], v4
	v_add_u32_e32 v4, v3, v14
	v_lshlrev_b32_e32 v15, 4, v8
	v_lshlrev_b32_e32 v16, 4, v9
	ds_read_b128 v[142:145], v5
	v_add_u32_e32 v5, v3, v15
	ds_read_b128 v[154:157], v4
	v_add_u32_e32 v4, v3, v16
	v_lshlrev_b32_e32 v17, 4, v10
	v_lshlrev_b32_e32 v18, 4, v11
	v_lshlrev_b32_e32 v19, 4, v2
	ds_read_b128 v[138:141], v5
	v_add_u32_e32 v5, v3, v17
	ds_read_b128 v[150:153], v4
	v_add_u32_e32 v4, v3, v18
	v_add_u32_e32 v2, v3, v19
	ds_read_b128 v[134:137], v5
	ds_read_b128 v[146:149], v4
	ds_read_b128 v[130:133], v2
	v_mad_u64_u32 v[10:11], s[20:21], s35, v36, v[34:35]
	scratch_store_dword off, v41, off offset:148 ; 4-byte Folded Spill
	; sched_barrier mask(0x00000000)
	s_and_b32 s19, s25, 0x3fff
	s_bitset1_b32 s19, 14
	s_and_b32 s20, s43, 0xffff
	s_lshl_b32 s19, s19, 16
	s_or_b32 s29, s20, s19
	s_mov_b32 s30, s22
	s_mov_b32 s31, s23
	v_add_lshl_u32 v106, v37, v34, 1
	v_add_lshl_u32 v107, v38, v34, 1
	buffer_load_dwordx4 v[2:5], v106, s[28:31], 0 offen
	buffer_load_dwordx4 v[6:9], v107, s[28:31], 0 offen
                                        ; kill: killed $sgpr30_sgpr31 killed $sgpr29
	; sched_barrier mask(0x00000000)
	s_ashr_i32 s41, s40, 31
	s_lshl_b64 s[30:31], s[40:41], 1
	s_add_u32 s20, s28, s30
	s_addc_u32 s29, s43, s31
	s_and_b32 s21, s29, 0xffff
	s_or_b32 s21, s21, s19
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[42:45], v106, s[20:23], 0 offen
	buffer_load_dwordx4 v[46:49], v107, s[20:23], 0 offen
                                        ; kill: killed $sgpr21
	s_waitcnt vmcnt(3)
	ds_write_b128 v245, v[2:5]
	s_waitcnt vmcnt(2)
	ds_write_b128 v245, v[6:9] offset:8192
	; sched_barrier mask(0x00000000)
	s_and_b32 s21, s35, 0x3fff
	s_bitset1_b32 s21, 14
	s_and_b32 s25, s33, 0xffff
	s_lshl_b32 s28, s21, 16
	s_or_b32 s25, s25, s28
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	v_lshlrev_b32_e32 v252, 1, v10
	v_add_lshl_u32 v105, v10, s35, 1
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[34:37], v252, s[24:27], 0 offen
	buffer_load_dwordx4 v[38:41], v105, s[24:27], 0 offen
	v_lshlrev_b32_e32 v2, 8, v54
	v_add3_u32 v244, 0, v12, v2
	v_add3_u32 v243, 0, v13, v2
	v_add3_u32 v242, 0, v14, v2
	v_add3_u32 v240, 0, v15, v2
	v_add3_u32 v238, 0, v16, v2
	v_add3_u32 v237, 0, v17, v2
	v_add3_u32 v236, 0, v18, v2
	v_add3_u32 v235, 0, v19, v2
	ds_read_b128 v[16:19], v244
	ds_read_b128 v[20:23], v243
	ds_read_b128 v[24:27], v242
	ds_read_b128 v[28:31], v240
	ds_read_b128 v[66:69], v240 offset:8192
	ds_read_b128 v[70:73], v238
	ds_read_b128 v[74:77], v238 offset:8192
	ds_read_b128 v[78:81], v237
	ds_read_b128 v[82:85], v237 offset:8192
	ds_read_b128 v[86:89], v236
	ds_read_b128 v[90:93], v236 offset:8192
	ds_read_b128 v[94:97], v235
	ds_read_b128 v[98:101], v235 offset:8192
	scratch_store_dword off, v54, off offset:136 ; 4-byte Folded Spill
	ds_read_b128 v[54:57], v244 offset:8192
	ds_read_b128 v[58:61], v243 offset:8192
	ds_read_b128 v[62:65], v242 offset:8192
                                        ; kill: killed $sgpr26_sgpr27 killed $sgpr25
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[2:17], v[16:17], v[158:159], 0
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[160:161], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[142:143], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[22:23], v[144:145], v[2:17]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[2:17], v[24:25], v[154:155], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[156:157], v[2:17]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[138:139], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[140:141], v[2:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[54:55], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[56:57], v[160:161], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[70:71], v[150:151], v[2:17]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[18:33], v[58:59], v[142:143], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[72:73], v[152:153], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[60:61], v[144:145], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[78:79], v[134:135], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[62:63], v[154:155], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[80:81], v[136:137], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[64:65], v[156:157], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[146:147], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[138:139], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[88:89], v[148:149], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[68:69], v[140:141], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[150:151], v[18:33]
	v_mov_b32_e32 v74, v105
	v_mfma_f32_32x32x8_f16 v[2:17], v[94:95], v[130:131], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[96:97], v[132:133], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[76:77], v[152:153], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[82:83], v[134:135], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[84:85], v[136:137], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[90:91], v[146:147], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[92:93], v[148:149], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[98:99], v[130:131], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[100:101], v[132:133], v[18:33]
	; sched_barrier mask(0x00000000)
	s_add_u32 s20, s20, s30
	s_addc_u32 s21, s29, s31
	s_ashr_i32 s43, s42, 31
	s_lshl_b64 s[26:27], s[42:43], 1
	s_add_u32 s29, s24, s26
	s_addc_u32 s33, s33, s27
	s_and_b32 s21, s21, 0xffff
	s_or_b32 s21, s21, s19
	s_barrier
	v_mov_b32_e32 v233, v106
	buffer_load_dwordx4 v[110:113], v106, s[20:23], 0 offen
	v_mov_b32_e32 v234, v107
	buffer_load_dwordx4 v[106:109], v107, s[20:23], 0 offen
	s_waitcnt vmcnt(6)
	ds_write_b128 v245, v[42:45]
	s_waitcnt vmcnt(5)
	ds_write_b128 v245, v[46:49] offset:8192
	; sched_barrier mask(0x00000000)
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v42, v3, v3
	v_max_f32_e32 v43, v2, v2
	v_max_f32_e32 v42, v43, v42
	v_max3_f32 v42, v42, v4, v5
	v_max3_f32 v42, v42, v6, v7
	v_max3_f32 v42, v42, v8, v9
	v_max3_f32 v42, v42, v10, v11
	v_max3_f32 v42, v42, v12, v13
	v_max3_f32 v42, v42, v14, v15
	v_max3_f32 v42, v42, v16, v17
	v_max3_f32 v42, v42, v18, v19
	v_max3_f32 v42, v42, v20, v21
	v_max3_f32 v42, v42, v22, v23
	v_max3_f32 v42, v42, v24, v25
	v_max3_f32 v42, v42, v26, v27
	v_max3_f32 v42, v42, v28, v29
	v_max3_f32 v42, v42, v30, v31
	v_max3_f32 v43, v42, v32, v33
	v_lshlrev_b32_e32 v42, 2, v0
	v_xor_b32_e32 v194, 0x80, v42
	ds_bpermute_b32 v44, v194, v43
	v_mov_b32_e32 v42, 0xff800000
	s_mov_b32 s35, 0x3e0293ee
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v223, v43, v44, v42
	v_mul_f32_e32 v43, 0xbe0293ee, v223
	v_fmamk_f32 v2, v2, 0x3e0293ee, v43
	v_fmamk_f32 v3, v3, 0x3e0293ee, v43
	v_fmamk_f32 v4, v4, 0x3e0293ee, v43
	v_fmamk_f32 v5, v5, 0x3e0293ee, v43
	v_fmamk_f32 v6, v6, 0x3e0293ee, v43
	v_fmamk_f32 v7, v7, 0x3e0293ee, v43
	v_fmamk_f32 v8, v8, 0x3e0293ee, v43
	v_fmamk_f32 v9, v9, 0x3e0293ee, v43
	v_fmamk_f32 v10, v10, 0x3e0293ee, v43
	v_fmamk_f32 v11, v11, 0x3e0293ee, v43
	v_fmamk_f32 v12, v12, 0x3e0293ee, v43
	v_fmamk_f32 v13, v13, 0x3e0293ee, v43
	v_fmamk_f32 v14, v14, 0x3e0293ee, v43
	v_fmamk_f32 v15, v15, 0x3e0293ee, v43
	v_fmamk_f32 v16, v16, 0x3e0293ee, v43
	v_fmamk_f32 v17, v17, 0x3e0293ee, v43
	v_fmamk_f32 v18, v18, 0x3e0293ee, v43
	v_fmamk_f32 v19, v19, 0x3e0293ee, v43
	v_fmamk_f32 v20, v20, 0x3e0293ee, v43
	v_fmamk_f32 v21, v21, 0x3e0293ee, v43
	v_fmamk_f32 v22, v22, 0x3e0293ee, v43
	v_fmamk_f32 v23, v23, 0x3e0293ee, v43
	v_fmamk_f32 v24, v24, 0x3e0293ee, v43
	v_fmamk_f32 v25, v25, 0x3e0293ee, v43
	v_fmamk_f32 v26, v26, 0x3e0293ee, v43
	v_fmamk_f32 v27, v27, 0x3e0293ee, v43
	v_fmamk_f32 v28, v28, 0x3e0293ee, v43
	v_fmamk_f32 v29, v29, 0x3e0293ee, v43
	v_fmamk_f32 v30, v30, 0x3e0293ee, v43
	v_fmamk_f32 v31, v31, 0x3e0293ee, v43
	v_fmamk_f32 v32, v32, 0x3e0293ee, v43
	v_fmac_f32_e32 v43, 0x3e0293ee, v33
	v_fmac_f32_e32 v42, 0xbe0293ee, v223
	; sched_barrier mask(0x00000000)
	s_and_b32 s20, s33, 0xffff
	s_or_b32 s21, s20, s28
	s_mov_b32 s20, s29
	s_barrier
	v_and_b32_e32 v33, 0x220, v102
	v_and_b32_e32 v44, 0x404, v103
	v_and_b32_e32 v46, 0x808, v104
	buffer_load_dwordx4 v[98:101], v252, s[20:23], 0 offen
	v_or_b32_e32 v45, v33, v44
	buffer_load_dwordx4 v[102:105], v105, s[20:23], 0 offen
	v_or_b32_e32 v47, v45, v46
	v_and_b32_e32 v48, 0x1010, v53
	v_or3_b32 v49, v48, v1, v47
	v_xor_b32_e32 v49, v52, v49
	s_mov_b32 s41, 0x5040100
	v_lshl_add_u32 v67, v49, 1, 0
	s_waitcnt vmcnt(5)
	v_perm_b32 v49, v38, v34, s41
	ds_write_b32 v67, v49 offset:16384
	v_or_b32_e32 v49, 0x44, v33
	v_xor_b32_e32 v49, v49, v44
	v_or_b32_e32 v53, v48, v46
	v_or3_b32 v49, v1, v49, v53
	v_xor_b32_e32 v49, v52, v49
	s_mov_b32 s42, 0x7060302
	v_lshl_add_u32 v68, v49, 1, 0
	v_perm_b32 v34, v38, v34, s42
	ds_read_b128 v[80:83], v244
	ds_read_b128 v[122:125], v244 offset:8192
	ds_read_b128 v[88:91], v243
	ds_read_b128 v[118:121], v243 offset:8192
	ds_read_b128 v[84:87], v242
	ds_read_b128 v[114:117], v242 offset:8192
	ds_read_b128 v[190:193], v240
	ds_read_b128 v[174:177], v240 offset:8192
	ds_read_b128 v[186:189], v238
	ds_read_b128 v[170:173], v238 offset:8192
	ds_read_b128 v[182:185], v237
	ds_read_b128 v[166:169], v237 offset:8192
	ds_read_b128 v[178:181], v236
	ds_read_b128 v[162:165], v236 offset:8192
	ds_read_b128 v[92:95], v235
	ds_read_b128 v[126:129], v235 offset:8192
	ds_write_b32 v68, v34 offset:16384
	v_or_b32_e32 v34, 0x88, v45
	v_xor_b32_e32 v34, v34, v46
	v_or3_b32 v34, v1, v34, v48
	v_xor_b32_e32 v34, v52, v34
	v_perm_b32 v38, v39, v35, s41
	v_lshl_add_u32 v69, v34, 1, 0
	ds_write_b32 v69, v38 offset:16384
	v_or_b32_e32 v34, 0xcc, v33
	v_or_b32_e32 v38, v46, v44
	v_xor_b32_e32 v34, v38, v34
	v_or3_b32 v34, v1, v34, v48
	v_xor_b32_e32 v34, v52, v34
	v_lshl_add_u32 v70, v34, 1, 0
	v_or_b32_e32 v34, 0x110, v47
	v_xor_b32_e32 v34, v34, v48
	v_or_b32_e32 v34, v34, v1
	v_xor_b32_e32 v34, v52, v34
	v_lshl_add_u32 v71, v34, 1, 0
	v_or_b32_e32 v34, 0x154, v33
	v_xor_b32_e32 v34, v34, v44
	v_or_b32_e32 v34, v34, v46
	v_xor_b32_e32 v34, v34, v48
	v_or_b32_e32 v34, v34, v1
	v_xor_b32_e32 v34, v52, v34
	v_lshl_add_u32 v72, v34, 1, 0
	v_or_b32_e32 v34, 0x198, v45
	v_xor_b32_e32 v34, v53, v34
	v_or_b32_e32 v34, v34, v1
	v_xor_b32_e32 v34, v52, v34
	v_lshl_add_u32 v73, v34, 1, 0
	v_or_b32_e32 v34, v38, v48
	v_or_b32_e32 v33, 0x1dc, v33
	v_xor_b32_e32 v33, v34, v33
	v_or_b32_e32 v1, v33, v1
	v_xor_b32_e32 v1, v52, v1
	v_cmp_gt_u32_e32 vcc, s16, v0
	s_movk_i32 s16, 0xff
	v_perm_b32 v35, v39, v35, s42
	v_perm_b32 v39, v40, v36, s41
	v_perm_b32 v36, v40, v36, s42
	v_perm_b32 v40, v41, v37, s41
	v_perm_b32 v37, v41, v37, s42
	v_lshl_add_u32 v66, v1, 1, 0
	v_cmp_lt_u32_e64 s[20:21], s16, v0
	ds_write_b32 v70, v35 offset:16384
	ds_write_b32 v71, v39 offset:16384
	ds_write_b32 v72, v36 offset:16384
	ds_write_b32 v73, v40 offset:16384
	ds_write_b32 v66, v37 offset:16384
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v0, off offset:128 ; 4-byte Folded Spill
	s_and_saveexec_b64 s[24:25], s[20:21]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[24:25]
	v_exp_f32_e32 v249, v2
	v_exp_f32_e32 v1, v3
	v_mov_b32_e32 v2, 0x44
	v_mov_b32_e32 v3, 0x88
	v_exp_f32_e32 v196, v5
	v_cndmask_b32_e64 v2, v2, 0, s[2:3]
	v_cndmask_b32_e64 v3, v3, 0, s[6:7]
	v_mov_b32_e32 v5, 0x110
	s_load_dwordx2 s[24:25], s[0:1], 0x4c
	s_load_dword s16, s[0:1], 0x54
	v_exp_f32_e32 v246, v4
	v_exp_f32_e32 v198, v7
	v_exp_f32_e32 v200, v9
	v_exp_f32_e32 v201, v12
	v_or_b32_e32 v4, v2, v3
	v_cndmask_b32_e64 v5, v5, 0, s[12:13]
	v_mov_b32_e32 v7, 0x220
	v_mov_b32_e32 v9, 0x404
	v_cmp_eq_u32_e64 s[0:1], 0, v50
	v_or_b32_e32 v12, 8, v2
	v_exp_f32_e32 v195, v6
	v_exp_f32_e32 v204, v13
	v_and_b32_e32 v40, 4, v51
	v_or_b32_e32 v6, v4, v5
	v_cndmask_b32_e64 v7, v7, 0, s[14:15]
	v_cndmask_b32_e64 v9, v9, 0, s[0:1]
	v_xor_b32_e32 v12, v12, v3
	v_or_b32_e32 v13, 16, v4
	v_exp_f32_e32 v197, v8
	v_exp_f32_e32 v199, v10
	v_exp_f32_e32 v203, v14
	v_exp_f32_e32 v206, v15
	v_or_b32_e32 v8, v6, v7
	v_xor_b32_e32 v10, v9, v40
	v_or3_b32 v12, v5, v12, v7
	v_xor_b32_e32 v13, v13, v5
	v_or_b32_e32 v14, 24, v2
	v_or_b32_e32 v15, v5, v3
	v_exp_f32_e32 v202, v11
	v_xor_b32_e32 v11, v10, v8
	v_xor_b32_e32 v12, v40, v12
	v_or_b32_e32 v13, v13, v7
	v_xor_b32_e32 v14, v15, v14
	v_xor_b32_e32 v12, v12, v9
	v_xor_b32_e32 v13, v40, v13
	v_or_b32_e32 v14, v14, v7
	v_lshl_add_u32 v0, v11, 1, 0
	v_exp_f32_e32 v208, v17
	v_xor_b32_e32 v13, v13, v9
	v_xor_b32_e32 v14, v40, v14
	v_or_b32_e32 v17, 40, v2
	scratch_store_dword off, v0, off offset:80 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v12, 1, 0
	v_exp_f32_e32 v205, v16
	v_xor_b32_e32 v14, v14, v9
	v_or_b32_e32 v16, 32, v6
	v_or_b32_e32 v10, v10, v7
	v_xor_b32_e32 v17, v17, v3
	scratch_store_dword off, v0, off offset:84 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v13, 1, 0
	v_exp_f32_e32 v211, v22
	v_xor_b32_e32 v16, v10, v16
	v_or_b32_e32 v17, v17, v5
	v_or_b32_e32 v22, 0x808, v2
	scratch_store_dword off, v0, off offset:88 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v14, 1, 0
	v_exp_f32_e32 v207, v18
	v_exp_f32_e32 v210, v19
	v_xor_b32_e32 v17, v10, v17
	v_or_b32_e32 v18, 48, v4
	v_or_b32_e32 v19, v10, v5
	v_xor_b32_e32 v22, v22, v3
	scratch_store_dword off, v0, off offset:92 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v16, 1, 0
	v_exp_f32_e32 v209, v20
	v_exp_f32_e32 v212, v21
	v_exp_f32_e32 v213, v24
	v_xor_b32_e32 v18, v19, v18
	v_or_b32_e32 v20, 56, v2
	v_or_b32_e32 v21, v19, v3
	v_or3_b32 v22, v5, v22, v7
	v_or_b32_e32 v24, 0x818, v2
	scratch_store_dword off, v0, off offset:96 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v17, 1, 0
	v_exp_f32_e32 v214, v23
	v_exp_f32_e32 v216, v25
	v_xor_b32_e32 v20, v21, v20
	v_xor_b32_e32 v22, v40, v22
	v_or_b32_e32 v23, 0x800, v8
	v_xor_b32_e32 v24, v15, v24
	v_or_b32_e32 v25, 0x810, v4
	scratch_store_dword off, v0, off offset:100 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v18, 1, 0
	v_xor_b32_e32 v22, v22, v9
	v_xor_b32_e32 v23, v40, v23
	v_or_b32_e32 v24, v24, v7
	v_xor_b32_e32 v25, v25, v5
	scratch_store_dword off, v0, off offset:104 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v20, 1, 0
	v_exp_f32_e32 v215, v26
	v_xor_b32_e32 v23, v23, v9
	v_xor_b32_e32 v24, v40, v24
	v_or_b32_e32 v25, v25, v7
	v_or_b32_e32 v26, 0x828, v2
	scratch_store_dword off, v0, off offset:108 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v22, 1, 0
	v_xor_b32_e32 v24, v24, v9
	v_xor_b32_e32 v25, v40, v25
	v_xor_b32_e32 v26, v26, v3
	scratch_store_dword off, v0, off offset:28 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v23, 1, 0
	v_xor_b32_e32 v25, v25, v9
	v_or_b32_e32 v26, v26, v5
	scratch_store_dword off, v0, off offset:32 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v24, 1, 0
	v_exp_f32_e32 v218, v27
	v_exp_f32_e32 v219, v30
	v_xor_b32_e32 v26, v10, v26
	v_or_b32_e32 v27, 0x820, v6
	v_or_b32_e32 v30, 0x1010, v4
	scratch_store_dword off, v0, off        ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v25, 1, 0
	v_exp_f32_e32 v217, v28
	v_exp_f32_e32 v221, v31
	v_xor_b32_e32 v27, v10, v27
	v_or_b32_e32 v28, 0x838, v2
	v_xor_b32_e32 v30, v30, v5
	v_or_b32_e32 v31, 0x1018, v2
	scratch_store_dword off, v0, off offset:36 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v26, 1, 0
	v_exp_f32_e32 v220, v29
	v_xor_b32_e32 v28, v21, v28
	v_or_b32_e32 v29, 0x830, v4
	v_or_b32_e32 v30, v30, v7
	v_xor_b32_e32 v31, v15, v31
	scratch_store_dword off, v0, off offset:40 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v27, 1, 0
	v_xor_b32_e32 v29, v19, v29
	v_xor_b32_e32 v30, v40, v30
	v_or_b32_e32 v31, v31, v7
	v_or_b32_e32 v33, 0x1008, v2
	scratch_store_dword off, v0, off offset:44 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v28, 1, 0
	v_exp_f32_e32 v247, v32
	v_xor_b32_e32 v30, v30, v9
	v_xor_b32_e32 v31, v40, v31
	v_or_b32_e32 v32, 0x1000, v8
	v_xor_b32_e32 v33, v33, v3
	scratch_store_dword off, v0, off offset:48 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v29, 1, 0
	v_xor_b32_e32 v31, v31, v9
	v_xor_b32_e32 v32, v40, v32
	v_or3_b32 v33, v5, v33, v7
	scratch_store_dword off, v0, off offset:52 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v30, 1, 0
	v_xor_b32_e32 v32, v32, v9
	v_xor_b32_e32 v33, v40, v33
	v_or_b32_e32 v38, 0x1818, v2
	scratch_store_dword off, v0, off offset:56 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v31, 1, 0
	v_xor_b32_e32 v33, v33, v9
	v_or_b32_e32 v34, 0x1030, v4
	v_xor_b32_e32 v15, v15, v38
	v_or_b32_e32 v38, 0x1810, v4
	v_or_b32_e32 v39, 0x1808, v2
	scratch_store_dword off, v0, off offset:60 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v32, 1, 0
	v_xor_b32_e32 v34, v19, v34
	v_or_b32_e32 v35, 0x1038, v2
	v_or_b32_e32 v37, 0x1028, v2
	v_xor_b32_e32 v38, v38, v5
	v_xor_b32_e32 v39, v39, v3
	scratch_store_dword off, v0, off offset:64 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v33, 1, 0
	v_xor_b32_e32 v35, v21, v35
	v_or_b32_e32 v36, 0x1020, v6
	v_xor_b32_e32 v37, v37, v3
	v_or_b32_e32 v15, v15, v7
	v_or_b32_e32 v38, v38, v7
	v_or3_b32 v7, v5, v39, v7
	v_or_b32_e32 v8, 0x1800, v8
	scratch_store_dword off, v0, off offset:68 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v34, 1, 0
	v_xor_b32_e32 v36, v10, v36
	v_or_b32_e32 v37, v37, v5
	v_xor_b32_e32 v15, v40, v15
	v_xor_b32_e32 v38, v40, v38
	v_xor_b32_e32 v7, v40, v7
	v_xor_b32_e32 v8, v40, v8
	scratch_store_dword off, v0, off offset:72 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v35, 1, 0
	s_add_u32 s0, s36, s38
	v_xor_b32_e32 v37, v10, v37
	v_xor_b32_e32 v15, v15, v9
	v_xor_b32_e32 v38, v38, v9
	v_xor_b32_e32 v7, v7, v9
	v_xor_b32_e32 v8, v8, v9
	v_or_b32_e32 v9, 0x1838, v2
	v_or_b32_e32 v2, 0x1828, v2
	scratch_store_dword off, v0, off offset:76 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v36, 1, 0
	s_addc_u32 s1, s37, s39
	v_xor_b32_e32 v2, v2, v3
	scratch_store_dword off, v0, off offset:4 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v37, 1, 0
	s_mul_i32 s3, s40, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_exp_f32_e32 v248, v43
	v_exp_f32_e32 v222, v42
	v_or_b32_e32 v2, v2, v5
	scratch_store_dword off, v0, off offset:8 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v15, 1, 0
	s_mul_hi_i32 s2, s40, 6
	s_add_u32 s0, s3, s0
	v_or_b32_e32 v4, 0x1830, v4
	v_xor_b32_e32 v2, v10, v2
	v_or_b32_e32 v3, 0x1820, v6
	scratch_store_dword off, v0, off offset:12 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v38, 1, 0
	s_addc_u32 s1, s2, s1
	v_xor_b32_e32 v9, v21, v9
	v_xor_b32_e32 v4, v19, v4
	v_xor_b32_e32 v3, v10, v3
	scratch_store_dword off, v0, off offset:16 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v7, 1, 0
	v_lshl_add_u32 v250, v2, 1, 0
	s_add_u32 s0, s4, s0
	v_mov_b32_e32 v2, 0
	s_waitcnt vmcnt(27)
	v_lshrrev_b32_e32 v254, 16, v102
	scratch_store_dword off, v40, off offset:124 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:20 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v8, 1, 0
	v_lshl_add_u32 v232, v9, 1, 0
	v_lshl_add_u32 v241, v4, 1, 0
	v_lshl_add_u32 v251, v3, 1, 0
	s_addc_u32 s1, s5, s1
	v_mov_b32_e32 v255, 1.0
	s_movk_i32 s2, 0xffc0
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
	scratch_store_dword off, v0, off offset:24 ; 4-byte Folded Spill
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v224, v67
	v_mov_b32_e32 v225, v68
	v_mov_b32_e32 v226, v69
	v_mov_b32_e32 v227, v70
	v_mov_b32_e32 v228, v71
	v_mov_b32_e32 v229, v72
	v_mov_b32_e32 v230, v73
	v_mov_b32_e32 v231, v66
	v_mov_b32_e32 v239, v74
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[80:81], v[158:159], 0
	v_mov_b32_e32 v253, v255
	v_mov_b32_e32 v0, v252
	v_mov_b32_e32 v252, v223
	s_setprio 0
	v_mfma_f32_32x32x8_f16 v[66:81], v[82:83], v[160:161], v[66:81]
	v_mul_f32_e32 v17, v17, v222
	v_mul_f32_e32 v50, v50, v222
	v_mul_f32_e32 v16, v16, v222
	v_mul_f32_e32 v15, v15, v222
	v_mul_f32_e32 v14, v14, v222
	v_mul_f32_e32 v13, v13, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[88:89], v[142:143], v[66:81]
	v_mul_f32_e32 v11, v11, v222
	v_mul_f32_e32 v12, v12, v222
	v_mul_f32_e32 v10, v10, v222
	v_mul_f32_e32 v9, v9, v222
	v_mul_f32_e32 v8, v8, v222
	v_mul_f32_e32 v7, v7, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[144:145], v[66:81]
	v_mul_f32_e32 v5, v5, v222
	v_mul_f32_e32 v6, v6, v222
	v_mul_f32_e32 v4, v4, v222
	v_mul_f32_e32 v3, v3, v222
	v_mul_f32_e32 v2, v2, v222
	v_mul_f32_e32 v33, v33, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[84:85], v[154:155], v[66:81]
	v_mul_f32_e32 v31, v31, v222
	v_mul_f32_e32 v32, v32, v222
	v_mul_f32_e32 v30, v30, v222
	v_mul_f32_e32 v29, v29, v222
	v_mul_f32_e32 v28, v28, v222
	v_mul_f32_e32 v27, v27, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[86:87], v[156:157], v[66:81]
	v_mul_f32_e32 v25, v25, v222
	v_mul_f32_e32 v26, v26, v222
	v_mul_f32_e32 v24, v24, v222
	v_mul_f32_e32 v23, v23, v222
	v_mul_f32_e32 v22, v22, v222
	v_mul_f32_e32 v21, v21, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[138:139], v[66:81]
	v_mul_f32_e32 v19, v19, v222
	v_mul_f32_e32 v20, v20, v222
	v_mul_f32_e32 v18, v18, v222
	v_mul_f32_e32 v49, v49, v222
	v_mul_f32_e32 v48, v48, v222
	v_mul_f32_e32 v47, v47, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[192:193], v[140:141], v[66:81]
	v_mul_f32_e32 v45, v45, v222
	v_mul_f32_e32 v46, v46, v222
	v_mul_f32_e32 v44, v44, v222
	v_mul_f32_e32 v43, v43, v222
	v_mul_f32_e32 v42, v42, v222
	v_mul_f32_e32 v41, v41, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[150:151], v[66:81]
	v_mul_f32_e32 v39, v39, v222
	v_mul_f32_e32 v40, v40, v222
	v_mul_f32_e32 v38, v38, v222
	v_mul_f32_e32 v37, v37, v222
	v_mul_f32_e32 v36, v36, v222
	v_mul_f32_e32 v35, v35, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[152:153], v[66:81]
	v_mul_f32_e32 v65, v65, v222
	v_mul_f32_e32 v34, v34, v222
	v_mul_f32_e32 v64, v64, v222
	v_mul_f32_e32 v63, v63, v222
	v_mul_f32_e32 v62, v62, v222
	v_mul_f32_e32 v61, v61, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[134:135], v[66:81]
	v_mul_f32_e32 v59, v59, v222
	v_mul_f32_e32 v60, v60, v222
	v_mul_f32_e32 v58, v58, v222
	v_mul_f32_e32 v57, v57, v222
	v_mul_f32_e32 v56, v56, v222
	v_mul_f32_e32 v55, v55, v222
	v_mfma_f32_32x32x8_f16 v[66:81], v[184:185], v[136:137], v[66:81]
	v_mul_f32_e32 v53, v53, v222
	v_mul_f32_e32 v54, v54, v222
	v_mul_f32_e32 v52, v52, v222
	v_mul_f32_e32 v51, v51, v222
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[180:181], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[132:133], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[122:123], v[158:159], 0
	v_cvt_pkrtz_f16_f32 v122, v207, v210
	v_cvt_pkrtz_f16_f32 v123, v209, v212
	v_mfma_f32_32x32x8_f16 v[82:97], v[124:125], v[160:161], v[82:97]
	v_cvt_pkrtz_f16_f32 v124, v211, v214
	v_cvt_pkrtz_f16_f32 v125, v213, v216
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[142:143], v[82:97]
	v_cvt_pkrtz_f16_f32 v118, v199, v202
	v_cvt_pkrtz_f16_f32 v119, v201, v204
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[144:145], v[82:97]
	v_cvt_pkrtz_f16_f32 v120, v203, v206
	v_cvt_pkrtz_f16_f32 v121, v205, v208
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[154:155], v[82:97]
	v_add_f32_e32 v114, v249, v1
	v_add_f32_e32 v114, v114, v246
	v_add_f32_e32 v114, v114, v196
	v_add_f32_e32 v114, v114, v195
	v_add_f32_e32 v114, v114, v198
	v_add_f32_e32 v114, v114, v197
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[156:157], v[82:97]
	v_add_f32_e32 v114, v114, v200
	v_cvt_pkrtz_f16_f32 v117, v197, v200
	v_add_f32_e32 v114, v114, v199
	v_cvt_pkrtz_f16_f32 v116, v195, v198
	v_add_f32_e32 v114, v114, v202
	v_add_f32_e32 v114, v114, v201
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[138:139], v[82:97]
	v_add_f32_e32 v114, v114, v204
	v_add_f32_e32 v114, v114, v203
	v_add_f32_e32 v114, v114, v206
	v_add_f32_e32 v114, v114, v205
	v_add_f32_e32 v114, v114, v208
	v_add_f32_e32 v114, v114, v207
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[140:141], v[82:97]
	v_add_f32_e32 v114, v114, v210
	v_add_f32_e32 v114, v114, v209
	v_add_f32_e32 v114, v114, v212
	v_add_f32_e32 v114, v114, v211
	v_add_f32_e32 v114, v114, v214
	v_add_f32_e32 v114, v114, v213
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[150:151], v[82:97]
	v_add_f32_e32 v114, v114, v216
	v_add_f32_e32 v114, v114, v215
	v_add_f32_e32 v114, v114, v218
	v_add_f32_e32 v114, v114, v217
	v_add_f32_e32 v114, v114, v220
	v_add_f32_e32 v114, v114, v219
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[152:153], v[82:97]
	v_add_f32_e32 v114, v114, v221
	v_add_f32_e32 v114, v114, v247
	v_add_f32_e32 v114, v114, v248
	ds_bpermute_b32 v115, v194, v114
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v255, v114, v115
	v_fmac_f32_e32 v255, v253, v222
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[134:135], v[82:97]
	v_cvt_pkrtz_f16_f32 v114, v249, v1
	v_cvt_pkrtz_f16_f32 v115, v246, v196
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[136:137], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[146:147], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[126:127], v[130:131], v[82:97]
	v_cvt_pkrtz_f16_f32 v126, v215, v218
	v_cvt_pkrtz_f16_f32 v127, v217, v220
	v_mfma_f32_32x32x8_f16 v[82:97], v[128:129], v[132:133], v[82:97]
	v_cvt_pkrtz_f16_f32 v128, v219, v221
	v_cvt_pkrtz_f16_f32 v129, v247, v248
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_barrier
	s_barrier
	scratch_load_dword v1, off, off offset:80 ; 4-byte Folded Reload
	s_add_u32 s12, s29, s26
	s_addc_u32 s3, s33, s27
	s_and_b32 s4, s1, 0xffff
	s_or_b32 s21, s4, s19
	s_mov_b32 s20, s0
	ds_read_b64 v[220:221], v232 offset:16384
	ds_read_b64 v[222:223], v241 offset:16384
	ds_read_b64 v[246:247], v250 offset:16384
	ds_read_b64 v[248:249], v251 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[162:163], v1 offset:16384
	scratch_load_dword v1, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[164:165], v1 offset:16384
	scratch_load_dword v1, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[166:167], v1 offset:16384
	scratch_load_dword v1, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[168:169], v1 offset:16384
	scratch_load_dword v1, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v1 offset:16384
	scratch_load_dword v1, off, off offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[172:173], v1 offset:16384
	scratch_load_dword v1, off, off offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[174:175], v1 offset:16384
	scratch_load_dword v1, off, off offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[176:177], v1 offset:16384
	scratch_load_dword v1, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[178:179], v1 offset:16384
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[180:181], v1 offset:16384
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[182:183], v1 offset:16384
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[184:185], v1 offset:16384
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[186:187], v1 offset:16384
	scratch_load_dword v1, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[188:189], v1 offset:16384
	scratch_load_dword v1, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[190:191], v1 offset:16384
	scratch_load_dword v1, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[192:193], v1 offset:16384
	scratch_load_dword v1, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[196:197], v1 offset:16384
	scratch_load_dword v1, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[198:199], v1 offset:16384
	scratch_load_dword v1, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[200:201], v1 offset:16384
	scratch_load_dword v1, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[202:203], v1 offset:16384
	scratch_load_dword v1, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[204:205], v1 offset:16384
	scratch_load_dword v1, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[206:207], v1 offset:16384
	scratch_load_dword v1, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[208:209], v1 offset:16384
	scratch_load_dword v1, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[210:211], v1 offset:16384
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v1 offset:16384
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v1 offset:16384
	scratch_load_dword v1, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[216:217], v1 offset:16384
	scratch_load_dword v1, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[218:219], v1 offset:16384
	ds_write_b128 v245, v[110:113]
	ds_write_b128 v245, v[106:109] offset:8192
	buffer_load_dwordx4 v[110:113], v233, s[20:23], 0 offen
	buffer_load_dwordx4 v[106:109], v234, s[20:23], 0 offen
	; sched_barrier mask(0x00000000)
	s_barrier
	s_setprio 0
	v_max_f32_e32 v1, v67, v67
	; iglp_opt mask(0x0000000A)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[50:65], v[162:163], v[114:115], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[114:115], v[34:49]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[18:33], v[196:197], v[114:115], v[18:33]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[2:17], v[212:213], v[114:115], v[2:17]
	v_max_f32_e32 v114, v66, v66
	v_max_f32_e32 v1, v114, v1
	v_max3_f32 v1, v1, v68, v69
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_max3_f32 v1, v1, v74, v75
	v_mfma_f32_32x32x8_f16 v[18:33], v[198:199], v[116:117], v[18:33]
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_mfma_f32_32x32x8_f16 v[18:33], v[200:201], v[118:119], v[18:33]
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	ds_bpermute_b32 v114, v194, v1
	v_mfma_f32_32x32x8_f16 v[18:33], v[202:203], v[120:121], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[164:165], v[116:117], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[116:117], v[34:49]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[2:17], v[214:215], v[116:117], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[166:167], v[118:119], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[118:119], v[34:49]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[2:17], v[216:217], v[118:119], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[168:169], v[120:121], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[120:121], v[34:49]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[2:17], v[218:219], v[120:121], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[170:171], v[122:123], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[122:123], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[204:205], v[122:123], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[220:221], v[122:123], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[222:223], v[124:125], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v223, v252, v1, v114
	v_mul_f32_e32 v253, 0x3e0293ee, v223
	v_fma_f32 v1, v66, s35, -v253
	v_fma_f32 v66, v67, s35, -v253
	v_fma_f32 v67, v68, s35, -v253
	v_fma_f32 v68, v69, s35, -v253
	v_mfma_f32_32x32x8_f16 v[2:17], v[246:247], v[126:127], v[2:17]
	v_fma_f32 v69, v70, s35, -v253
	v_fma_f32 v70, v71, s35, -v253
	v_fma_f32 v71, v72, s35, -v253
	v_fma_f32 v72, v73, s35, -v253
	v_fma_f32 v73, v74, s35, -v253
	v_fma_f32 v74, v75, s35, -v253
	v_mfma_f32_32x32x8_f16 v[2:17], v[248:249], v[128:129], v[2:17]
	v_fma_f32 v75, v76, s35, -v253
	v_fma_f32 v76, v77, s35, -v253
	v_fma_f32 v77, v78, s35, -v253
	v_exp_f32_e32 v246, v67
	v_mfma_f32_32x32x8_f16 v[18:33], v[206:207], v[124:125], v[18:33]
	v_fma_f32 v78, v79, s35, -v253
	v_fma_f32 v79, v80, s35, -v253
	v_exp_f32_e32 v196, v68
	v_mfma_f32_32x32x8_f16 v[18:33], v[208:209], v[126:127], v[18:33]
	v_fma_f32 v80, v81, s35, -v253
	v_fma_f32 v81, v82, s35, -v253
	v_exp_f32_e32 v249, v1
	v_mfma_f32_32x32x8_f16 v[18:33], v[210:211], v[128:129], v[18:33]
	v_fma_f32 v82, v83, s35, -v253
	v_fma_f32 v83, v84, s35, -v253
	v_fma_f32 v84, v85, s35, -v253
	v_fma_f32 v85, v86, s35, -v253
	v_fma_f32 v86, v87, s35, -v253
	v_fma_f32 v87, v88, s35, -v253
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[124:125], v[50:65]
	v_fma_f32 v88, v89, s35, -v253
	v_fma_f32 v89, v90, s35, -v253
	v_fma_f32 v90, v91, s35, -v253
	v_fma_f32 v91, v92, s35, -v253
	v_fma_f32 v92, v93, s35, -v253
	v_fma_f32 v93, v94, s35, -v253
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[126:127], v[50:65]
	v_fma_f32 v94, v95, s35, -v253
	v_fma_f32 v95, v96, s35, -v253
	v_exp_f32_e32 v1, v66
	v_mfma_f32_32x32x8_f16 v[34:49], v[188:189], v[124:125], v[34:49]
	v_fma_f32 v66, v252, s35, -v253
	v_fma_f32 v96, v97, s35, -v253
	v_exp_f32_e32 v195, v69
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[126:127], v[34:49]
	v_exp_f32_e32 v198, v70
	v_exp_f32_e32 v197, v71
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[128:129], v[50:65]
	v_exp_f32_e32 v200, v72
	v_exp_f32_e32 v199, v73
	v_mfma_f32_32x32x8_f16 v[34:49], v[192:193], v[128:129], v[34:49]
	v_exp_f32_e32 v202, v74
	v_exp_f32_e32 v207, v81
	v_exp_f32_e32 v201, v75
	v_exp_f32_e32 v204, v76
	v_exp_f32_e32 v203, v77
	v_exp_f32_e32 v206, v78
	v_exp_f32_e32 v205, v79
	v_exp_f32_e32 v208, v80
	v_exp_f32_e32 v210, v82
	v_exp_f32_e32 v209, v83
	v_exp_f32_e32 v212, v84
	v_exp_f32_e32 v211, v85
	v_exp_f32_e32 v214, v86
	v_exp_f32_e32 v213, v87
	v_exp_f32_e32 v216, v88
	v_exp_f32_e32 v215, v89
	v_exp_f32_e32 v218, v90
	v_exp_f32_e32 v217, v91
	v_exp_f32_e32 v220, v92
	v_exp_f32_e32 v219, v93
	v_exp_f32_e32 v221, v94
	v_exp_f32_e32 v247, v95
	v_exp_f32_e32 v248, v96
	v_exp_f32_e32 v222, v66
	v_mov_b32_e32 v67, v224
	v_mov_b32_e32 v68, v225
	v_mov_b32_e32 v69, v226
	v_mov_b32_e32 v70, v227
	v_mov_b32_e32 v71, v228
	v_mov_b32_e32 v72, v229
	v_mov_b32_e32 v73, v230
	v_mov_b32_e32 v74, v239
	v_mov_b32_e32 v252, v0
	s_setprio 1
	; sched_barrier mask(0x00000000)
	v_perm_b32 v66, v102, v98, s41
	s_barrier
	ds_write_b32 v224, v66 offset:16384
	v_alignbit_b32 v66, v254, v98, 16
	ds_read_b128 v[80:83], v244
	ds_read_b128 v[88:91], v243
	ds_read_b128 v[84:87], v242
	ds_read_b128 v[190:193], v240
	ds_read_b128 v[186:189], v238
	ds_read_b128 v[182:185], v237
	ds_read_b128 v[178:181], v236
	ds_read_b128 v[92:95], v235
	ds_read_b128 v[122:125], v244 offset:8192
	ds_read_b128 v[118:121], v243 offset:8192
	ds_read_b128 v[114:117], v242 offset:8192
	ds_read_b128 v[174:177], v240 offset:8192
	ds_read_b128 v[170:173], v238 offset:8192
	ds_read_b128 v[166:169], v237 offset:8192
	ds_read_b128 v[162:165], v236 offset:8192
	ds_read_b128 v[126:129], v235 offset:8192
	ds_write_b32 v225, v66 offset:16384
	v_perm_b32 v66, v103, v99, s41
	ds_write_b32 v226, v66 offset:16384
	v_perm_b32 v66, v103, v99, s42
	ds_write_b32 v227, v66 offset:16384
	v_perm_b32 v66, v104, v100, s41
	ds_write_b32 v228, v66 offset:16384
	v_perm_b32 v66, v104, v100, s42
	s_and_b32 s4, s3, 0xffff
	ds_write_b32 v229, v66 offset:16384
	v_perm_b32 v66, v105, v101, s41
	s_or_b32 s13, s4, s28
	s_mov_b32 s14, s22
	s_mov_b32 s15, s23
	ds_write_b32 v230, v66 offset:16384
	v_perm_b32 v66, v105, v101, s42
	buffer_load_dwordx4 v[102:105], v239, s[12:15], 0 offen
	buffer_load_dwordx4 v[98:101], v0, s[12:15], 0 offen
	ds_write_b32 v231, v66 offset:16384
	v_mov_b32_e32 v66, v231
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v254, 16, v102
	; sched_barrier mask(0x00000000)
	s_add_u32 s29, s29, s26
	s_addc_u32 s33, s33, s27
	s_add_u32 s0, s0, s30
	s_addc_u32 s1, s1, s31
	s_add_i32 s2, s2, 64
	s_cmpk_lt_u32 s2, 0x1f00
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	scratch_store_dword off, v251, off offset:132 ; 4-byte Folded Spill
	scratch_store_dword off, v250, off offset:120 ; 4-byte Folded Spill
	scratch_store_dword off, v241, off offset:116 ; 4-byte Folded Spill
	scratch_store_dword off, v232, off offset:112 ; 4-byte Folded Spill
	scratch_load_dword v233, off, off offset:92 ; 4-byte Folded Reload
	s_nop 0
	scratch_load_dword v232, off, off offset:88 ; 4-byte Folded Reload
	scratch_load_dword v251, off, off offset:84 ; 4-byte Folded Reload
	scratch_load_dword v254, off, off offset:80 ; 4-byte Folded Reload
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dword v66, off, off offset:144 ; 4-byte Folded Reload
	scratch_load_dword v67, off, off offset:136 ; 4-byte Folded Reload
	scratch_load_dword v68, off, off offset:140 ; 4-byte Folded Reload
	v_mul_f32_e32 v17, v17, v222
	v_mul_f32_e32 v15, v15, v222
	v_mul_f32_e32 v14, v14, v222
	s_mul_i32 s4, s18, 0xc0000
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s2, s8, s4
	s_addc_u32 s6, s9, s5
	s_lshl_b32 s4, s17, 14
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s2, s2, s4
	s_addc_u32 s6, s6, s5
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[4:5], s[34:35], 2
	s_add_u32 s4, s2, s4
	s_addc_u32 s19, s6, s5
	s_add_i32 s2, s34, 0xffffc100
	s_add_u32 s12, s12, s26
	s_addc_u32 s3, s3, s27
	s_and_b32 s13, s3, 0xffff
	v_mul_f32_e32 v50, v50, v222
	v_mul_f32_e32 v51, v51, v222
	v_mul_f32_e32 v52, v52, v222
	v_mul_f32_e32 v53, v53, v222
	v_mul_f32_e32 v54, v54, v222
	v_mul_f32_e32 v55, v55, v222
	v_mul_f32_e32 v56, v56, v222
	v_mul_f32_e32 v57, v57, v222
	v_mul_f32_e32 v58, v58, v222
	v_mul_f32_e32 v59, v59, v222
	v_mul_f32_e32 v60, v60, v222
	v_mul_f32_e32 v61, v61, v222
	v_mul_f32_e32 v62, v62, v222
	v_mul_f32_e32 v63, v63, v222
	v_mul_f32_e32 v64, v64, v222
	v_mul_f32_e32 v65, v65, v222
	v_mul_f32_e32 v34, v34, v222
	v_mul_f32_e32 v35, v35, v222
	v_mul_f32_e32 v36, v36, v222
	v_mul_f32_e32 v37, v37, v222
	v_mul_f32_e32 v38, v38, v222
	v_mul_f32_e32 v39, v39, v222
	v_mul_f32_e32 v40, v40, v222
	v_mul_f32_e32 v41, v41, v222
	v_mul_f32_e32 v42, v42, v222
	v_mul_f32_e32 v43, v43, v222
	v_mul_f32_e32 v44, v44, v222
	v_mul_f32_e32 v45, v45, v222
	v_mul_f32_e32 v46, v46, v222
	v_mul_f32_e32 v47, v47, v222
	v_mul_f32_e32 v48, v48, v222
	v_mul_f32_e32 v49, v49, v222
	v_mul_f32_e32 v18, v18, v222
	v_mul_f32_e32 v19, v19, v222
	v_mul_f32_e32 v20, v20, v222
	v_mul_f32_e32 v21, v21, v222
	v_mul_f32_e32 v22, v22, v222
	v_mul_f32_e32 v23, v23, v222
	v_mul_f32_e32 v24, v24, v222
	v_mul_f32_e32 v25, v25, v222
	v_mul_f32_e32 v26, v26, v222
	v_mul_f32_e32 v27, v27, v222
	v_mul_f32_e32 v28, v28, v222
	v_mul_f32_e32 v29, v29, v222
	v_mul_f32_e32 v30, v30, v222
	v_mul_f32_e32 v31, v31, v222
	v_mul_f32_e32 v32, v32, v222
	v_mul_f32_e32 v33, v33, v222
	v_mul_f32_e32 v2, v2, v222
	v_mul_f32_e32 v3, v3, v222
	v_mul_f32_e32 v4, v4, v222
	v_mul_f32_e32 v5, v5, v222
	v_mul_f32_e32 v6, v6, v222
	v_mul_f32_e32 v7, v7, v222
	v_mul_f32_e32 v8, v8, v222
	v_mul_f32_e32 v9, v9, v222
	s_waitcnt vmcnt(2)
	v_cmp_eq_u32_e64 s[0:1], 0, v66
	scratch_load_dword v66, off, off offset:148 ; 4-byte Folded Reload
	v_mul_f32_e32 v10, v10, v222
	v_mul_f32_e32 v11, v11, v222
	v_mul_f32_e32 v12, v12, v222
	v_mul_f32_e32 v13, v13, v222
	v_mul_f32_e32 v16, v16, v222
	s_cmp_lt_i32 s2, 1
	; iglp_opt mask(0x0000000A)
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v66, 0xa0, v66
	v_or3_b32 v66, v66, v67, v68
	scratch_store_dword off, v66, off offset:136 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[66:81], v[80:81], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[66:81], v[82:83], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[88:89], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[84:85], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[86:87], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[192:193], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[184:185], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[180:181], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[132:133], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[122:123], v[158:159], 0
	v_cvt_pkrtz_f16_f32 v122, v203, v206
	v_cvt_pkrtz_f16_f32 v123, v205, v208
	v_mfma_f32_32x32x8_f16 v[82:97], v[124:125], v[160:161], v[82:97]
	v_cvt_pkrtz_f16_f32 v124, v199, v202
	v_cvt_pkrtz_f16_f32 v125, v201, v204
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[142:143], v[82:97]
	v_cvt_pkrtz_f16_f32 v118, v211, v214
	v_cvt_pkrtz_f16_f32 v119, v213, v216
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[144:145], v[82:97]
	v_cvt_pkrtz_f16_f32 v120, v207, v210
	v_cvt_pkrtz_f16_f32 v121, v209, v212
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[154:155], v[82:97]
	v_add_f32_e32 v114, v249, v1
	v_add_f32_e32 v114, v114, v246
	v_add_f32_e32 v114, v114, v196
	v_add_f32_e32 v114, v114, v195
	v_add_f32_e32 v114, v114, v198
	v_add_f32_e32 v114, v114, v197
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[156:157], v[82:97]
	v_add_f32_e32 v114, v114, v200
	v_cvt_pkrtz_f16_f32 v117, v217, v220
	v_add_f32_e32 v114, v114, v199
	v_cvt_pkrtz_f16_f32 v116, v215, v218
	v_add_f32_e32 v114, v114, v202
	v_add_f32_e32 v114, v114, v201
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[138:139], v[82:97]
	v_add_f32_e32 v114, v114, v204
	v_add_f32_e32 v114, v114, v203
	v_add_f32_e32 v114, v114, v206
	v_add_f32_e32 v114, v114, v205
	v_add_f32_e32 v114, v114, v208
	v_add_f32_e32 v114, v114, v207
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[140:141], v[82:97]
	v_add_f32_e32 v114, v114, v210
	v_add_f32_e32 v114, v114, v209
	v_add_f32_e32 v114, v114, v212
	v_add_f32_e32 v114, v114, v211
	v_add_f32_e32 v114, v114, v214
	v_add_f32_e32 v114, v114, v213
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[150:151], v[82:97]
	v_add_f32_e32 v114, v114, v216
	v_add_f32_e32 v114, v114, v215
	v_add_f32_e32 v114, v114, v218
	v_add_f32_e32 v114, v114, v217
	v_add_f32_e32 v114, v114, v220
	v_add_f32_e32 v114, v114, v219
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[152:153], v[82:97]
	v_add_f32_e32 v114, v114, v221
	v_add_f32_e32 v114, v114, v247
	v_add_f32_e32 v114, v114, v248
	ds_bpermute_b32 v115, v194, v114
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v114, v114, v115
	v_fmac_f32_e32 v114, v255, v222
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[134:135], v[82:97]
	scratch_store_dword off, v114, off offset:140 ; 4-byte Folded Spill
	v_cvt_pkrtz_f16_f32 v115, v247, v248
	v_cvt_pkrtz_f16_f32 v114, v219, v221
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[136:137], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[146:147], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[126:127], v[130:131], v[82:97]
	v_cvt_pkrtz_f16_f32 v126, v195, v198
	v_cvt_pkrtz_f16_f32 v127, v197, v200
	v_mfma_f32_32x32x8_f16 v[82:97], v[128:129], v[132:133], v[82:97]
	v_cvt_pkrtz_f16_f32 v128, v249, v1
	v_cvt_pkrtz_f16_f32 v129, v246, v196
	; sched_barrier mask(0x00000000)
	s_barrier
	s_barrier
	scratch_load_dword v1, off, off offset:100 ; 4-byte Folded Reload
	ds_read_b64 v[176:177], v254 offset:16384
	v_mov_b32_e32 v193, v251
	ds_read_b64 v[178:179], v251 offset:16384
	scratch_load_dword v195, off, off offset:96 ; 4-byte Folded Reload
	scratch_load_dword v234, off, off offset:108 ; 4-byte Folded Reload
	scratch_load_dword v252, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v192, off, off offset:132 ; 4-byte Folded Reload
	ds_read_b64 v[180:181], v232 offset:16384
	ds_read_b64 v[182:183], v233 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[186:187], v1 offset:16384
	scratch_load_dword v1, off, off offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(4)
	ds_read_b64 v[184:185], v195 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[190:191], v234 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[196:197], v252 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[162:163], v192 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[188:189], v1 offset:16384
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[198:199], v1 offset:16384
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[200:201], v1 offset:16384
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[202:203], v1 offset:16384
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[204:205], v1 offset:16384
	scratch_load_dword v1, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[206:207], v1 offset:16384
	scratch_load_dword v1, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[208:209], v1 offset:16384
	scratch_load_dword v1, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[210:211], v1 offset:16384
	scratch_load_dword v1, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v1 offset:16384
	scratch_load_dword v1, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v1 offset:16384
	scratch_load_dword v1, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[216:217], v1 offset:16384
	scratch_load_dword v1, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[218:219], v1 offset:16384
	scratch_load_dword v1, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[220:221], v1 offset:16384
	scratch_load_dword v1, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[246:247], v1 offset:16384
	scratch_load_dword v1, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[248:249], v1 offset:16384
	scratch_load_dword v1, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[254:255], v1 offset:16384
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[250:251], v1 offset:16384
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[174:175], v1 offset:16384
	scratch_load_dword v1, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[172:173], v1 offset:16384
	scratch_load_dword v1, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v1 offset:16384
	scratch_load_dword v1, off, off offset:112 ; 4-byte Folded Reload
	ds_write_b128 v245, v[110:113]
	ds_write_b128 v245, v[106:109] offset:8192
	s_waitcnt vmcnt(0)
	ds_read_b64 v[168:169], v1 offset:16384
	scratch_load_dword v1, off, off offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[166:167], v1 offset:16384
	scratch_load_dword v1, off, off offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[164:165], v1 offset:16384
	; sched_barrier mask(0x00000000)
	s_barrier
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_f16 v[2:17], v[250:251], v[128:129], v[2:17]
	v_max_f32_e32 v1, v67, v67
	v_max_f32_e32 v106, v66, v66
	v_max_f32_e32 v1, v106, v1
	v_max3_f32 v1, v1, v68, v69
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[128:129], v[50:65]
	v_max3_f32 v1, v1, v74, v75
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_mfma_f32_32x32x8_f16 v[50:65], v[178:179], v[126:127], v[50:65]
	v_max3_f32 v1, v1, v86, v87
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	v_mfma_f32_32x32x8_f16 v[50:65], v[180:181], v[124:125], v[50:65]
	ds_bpermute_b32 v106, v194, v1
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v250, v223, v1, v106
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[18:33], v[212:213], v[128:129], v[18:33]
	v_mfma_f32_32x32x8_f16 v[34:49], v[196:197], v[128:129], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[214:215], v[126:127], v[18:33]
	v_mfma_f32_32x32x8_f16 v[34:49], v[198:199], v[126:127], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[174:175], v[126:127], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[216:217], v[124:125], v[18:33]
	v_mfma_f32_32x32x8_f16 v[34:49], v[200:201], v[124:125], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[172:173], v[124:125], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[182:183], v[122:123], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[202:203], v[122:123], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[218:219], v[122:123], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[170:171], v[122:123], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[184:185], v[120:121], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[204:205], v[120:121], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[220:221], v[120:121], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[168:169], v[120:121], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[186:187], v[118:119], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[206:207], v[118:119], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[246:247], v[118:119], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[166:167], v[118:119], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[188:189], v[116:117], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[208:209], v[116:117], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[248:249], v[116:117], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[164:165], v[116:117], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[190:191], v[114:115], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[210:211], v[114:115], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[254:255], v[114:115], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[162:163], v[114:115], v[2:17]
	; sched_barrier mask(0x00000000)
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_barrier
	s_barrier
	buffer_load_dwordx4 v[162:165], v0, s[12:15], 0 offen
	buffer_load_dwordx4 v[166:169], v239, s[12:15], 0 offen
	s_mov_b32 s2, 0x5040100
	v_perm_b32 v1, v102, v98, s2
	s_mov_b32 s3, 0x7060302
	ds_write_b32 v224, v1 offset:16384
	v_perm_b32 v1, v102, v98, s3
	ds_read_b128 v[106:109], v244
	ds_read_b128 v[170:173], v244 offset:8192
	ds_read_b128 v[110:113], v243
	ds_read_b128 v[174:177], v243 offset:8192
	ds_read_b128 v[178:181], v242
	ds_read_b128 v[182:185], v242 offset:8192
	ds_read_b128 v[186:189], v240
	ds_read_b128 v[196:199], v240 offset:8192
	ds_read_b128 v[200:203], v238
	ds_read_b128 v[204:207], v238 offset:8192
	ds_read_b128 v[208:211], v237
	ds_read_b128 v[212:215], v237 offset:8192
	ds_read_b128 v[216:219], v236
	ds_read_b128 v[220:223], v236 offset:8192
	ds_read_b128 v[236:239], v235
	ds_read_b128 v[240:243], v235 offset:8192
	ds_write_b32 v225, v1 offset:16384
	v_perm_b32 v1, v103, v99, s2
	ds_write_b32 v226, v1 offset:16384
	v_perm_b32 v1, v103, v99, s3
	ds_write_b32 v227, v1 offset:16384
	v_perm_b32 v1, v104, v100, s2
	ds_write_b32 v228, v1 offset:16384
	v_perm_b32 v1, v104, v100, s3
	ds_write_b32 v229, v1 offset:16384
	v_perm_b32 v1, v105, v101, s2
	ds_write_b32 v230, v1 offset:16384
	v_perm_b32 v1, v105, v101, s3
	scratch_store_dword off, v224, off offset:152 ; 4-byte Folded Spill
	scratch_store_dword off, v225, off offset:156 ; 4-byte Folded Spill
	scratch_store_dword off, v226, off offset:160 ; 4-byte Folded Spill
	scratch_store_dword off, v227, off offset:164 ; 4-byte Folded Spill
	scratch_store_dword off, v228, off offset:168 ; 4-byte Folded Spill
	scratch_store_dword off, v229, off offset:172 ; 4-byte Folded Spill
	scratch_store_dword off, v230, off offset:176 ; 4-byte Folded Spill
	scratch_store_dword off, v231, off offset:180 ; 4-byte Folded Spill
	ds_write_b32 v231, v1 offset:16384
	; sched_barrier mask(0x00000000)
	s_barrier
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[114:129], v[106:107], v[158:159], 0
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[114:129], v[108:109], v[160:161], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[110:111], v[142:143], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[112:113], v[144:145], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[170:171], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[98:113], v[172:173], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[178:179], v[154:155], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[174:175], v[142:143], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[180:181], v[156:157], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[176:177], v[144:145], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[186:187], v[138:139], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[182:183], v[154:155], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[188:189], v[140:141], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[184:185], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[200:201], v[150:151], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[196:197], v[138:139], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[202:203], v[152:153], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[198:199], v[140:141], v[98:113]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[114:129], v[208:209], v[134:135], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[204:205], v[150:151], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[210:211], v[136:137], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[206:207], v[152:153], v[98:113]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[114:129], v[216:217], v[146:147], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[212:213], v[134:135], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[218:219], v[148:149], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[214:215], v[136:137], v[98:113]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_f16 v[114:129], v[236:237], v[130:131], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[220:221], v[146:147], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[238:239], v[132:133], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[222:223], v[148:149], v[98:113]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 v[98:113], v[240:241], v[130:131], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[242:243], v[132:133], v[98:113]
	; sched_barrier mask(0x00000000)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	scratch_load_dword v255, off, off offset:80 ; 4-byte Folded Reload
	ds_read_b64 v[172:173], v232 offset:16384
	ds_read_b64 v[156:157], v233 offset:16384
	scratch_load_dword v254, off, off offset:100 ; 4-byte Folded Reload
	scratch_load_dword v232, off, off offset:104 ; 4-byte Folded Reload
	ds_read_b64 v[144:145], v234 offset:16384
	ds_read_b64 v[198:199], v252 offset:16384
	scratch_load_dword v224, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v225, off, off offset:36 ; 4-byte Folded Reload
	scratch_load_dword v226, off, off offset:40 ; 4-byte Folded Reload
	scratch_load_dword v227, off, off offset:44 ; 4-byte Folded Reload
	scratch_load_dword v228, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v229, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v230, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v231, off, off offset:60 ; 4-byte Folded Reload
	scratch_load_dword v233, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v234, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v251, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v252, off, off offset:76 ; 4-byte Folded Reload
	ds_read_b64 v[178:179], v193 offset:16384
	ds_read_b64 v[152:153], v195 offset:16384
	ds_read_b64 v[130:131], v192 offset:16384
	s_waitcnt vmcnt(15)
	ds_read_b64 v[180:181], v0 offset:16384
	scratch_load_dword v0, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(15)
	ds_read_b64 v[196:197], v255 offset:16384
	s_waitcnt vmcnt(14)
	ds_read_b64 v[148:149], v254 offset:16384
	s_waitcnt vmcnt(13)
	ds_read_b64 v[146:147], v232 offset:16384
	s_waitcnt vmcnt(12)
	ds_read_b64 v[186:187], v224 offset:16384
	s_waitcnt vmcnt(11)
	ds_read_b64 v[170:171], v225 offset:16384
	s_waitcnt vmcnt(10)
	ds_read_b64 v[154:155], v226 offset:16384
	s_waitcnt vmcnt(9)
	ds_read_b64 v[150:151], v227 offset:16384
	s_waitcnt vmcnt(8)
	ds_read_b64 v[142:143], v228 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[140:141], v229 offset:16384
	s_waitcnt vmcnt(6)
	ds_read_b64 v[200:201], v230 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[188:189], v231 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[182:183], v233 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[174:175], v234 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[158:159], v251 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[138:139], v252 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[136:137], v0 offset:16384
	scratch_load_dword v0, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[134:135], v0 offset:16384
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[202:203], v0 offset:16384
	scratch_load_dword v0, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[204:205], v0 offset:16384
	scratch_load_dword v0, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[190:191], v0 offset:16384
	scratch_load_dword v0, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[184:185], v0 offset:16384
	scratch_load_dword v0, off, off offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[176:177], v0 offset:16384
	scratch_load_dword v0, off, off offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[160:161], v0 offset:16384
	scratch_load_dword v0, off, off offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[132:133], v0 offset:16384
	; sched_barrier mask(0x00000000)
	v_fmac_f32_e32 v253, 0xbe0293ee, v250
	v_exp_f32_e32 v193, v253
	s_mov_b32 s5, 0x3e0293ee
	v_mul_f32_e32 v206, 0x3e0293ee, v250
	v_fma_f32 v66, v66, s5, -v206
	v_fma_f32 v67, v67, s5, -v206
	v_fma_f32 v68, v68, s5, -v206
	v_exp_f32_e32 v207, v66
	v_exp_f32_e32 v208, v67
	v_fma_f32 v69, v69, s5, -v206
	v_exp_f32_e32 v209, v68
	v_fma_f32 v70, v70, s5, -v206
	v_exp_f32_e32 v210, v69
	v_fma_f32 v71, v71, s5, -v206
	v_exp_f32_e32 v211, v70
	v_fma_f32 v72, v72, s5, -v206
	v_exp_f32_e32 v212, v71
	v_add_f32_e32 v66, v207, v208
	v_fma_f32 v73, v73, s5, -v206
	v_exp_f32_e32 v213, v72
	v_add_f32_e32 v66, v209, v66
	v_fma_f32 v74, v74, s5, -v206
	v_exp_f32_e32 v214, v73
	v_add_f32_e32 v66, v210, v66
	v_fma_f32 v75, v75, s5, -v206
	v_exp_f32_e32 v215, v74
	v_add_f32_e32 v66, v211, v66
	v_fma_f32 v76, v76, s5, -v206
	v_exp_f32_e32 v216, v75
	v_add_f32_e32 v66, v212, v66
	v_fma_f32 v77, v77, s5, -v206
	v_exp_f32_e32 v217, v76
	v_add_f32_e32 v66, v213, v66
	v_fma_f32 v78, v78, s5, -v206
	v_exp_f32_e32 v218, v77
	v_add_f32_e32 v66, v214, v66
	v_fma_f32 v79, v79, s5, -v206
	v_exp_f32_e32 v219, v78
	v_add_f32_e32 v66, v215, v66
	v_fma_f32 v80, v80, s5, -v206
	v_exp_f32_e32 v220, v79
	v_add_f32_e32 v66, v216, v66
	v_fma_f32 v81, v81, s5, -v206
	v_exp_f32_e32 v221, v80
	v_add_f32_e32 v66, v217, v66
	v_fma_f32 v82, v82, s5, -v206
	v_exp_f32_e32 v222, v81
	v_add_f32_e32 v66, v218, v66
	v_fma_f32 v83, v83, s5, -v206
	v_exp_f32_e32 v223, v82
	v_add_f32_e32 v66, v219, v66
	v_fma_f32 v84, v84, s5, -v206
	v_exp_f32_e32 v235, v83
	v_add_f32_e32 v66, v220, v66
	v_fma_f32 v85, v85, s5, -v206
	v_exp_f32_e32 v236, v84
	v_add_f32_e32 v66, v221, v66
	v_max_f32_e32 v1, v115, v115
	v_max_f32_e32 v195, v114, v114
	v_fma_f32 v86, v86, s5, -v206
	v_exp_f32_e32 v237, v85
	v_add_f32_e32 v66, v222, v66
	v_max_f32_e32 v1, v195, v1
	v_fma_f32 v87, v87, s5, -v206
	v_exp_f32_e32 v238, v86
	v_add_f32_e32 v66, v223, v66
	v_max3_f32 v1, v1, v116, v117
	v_fma_f32 v88, v88, s5, -v206
	v_exp_f32_e32 v239, v87
	v_add_f32_e32 v66, v235, v66
	v_max3_f32 v1, v1, v118, v119
	v_fma_f32 v89, v89, s5, -v206
	v_exp_f32_e32 v240, v88
	v_add_f32_e32 v66, v236, v66
	v_max3_f32 v1, v1, v120, v121
	v_fma_f32 v90, v90, s5, -v206
	v_exp_f32_e32 v241, v89
	v_add_f32_e32 v66, v237, v66
	v_max3_f32 v1, v1, v122, v123
	v_fma_f32 v91, v91, s5, -v206
	v_exp_f32_e32 v242, v90
	v_add_f32_e32 v66, v238, v66
	v_max3_f32 v1, v1, v124, v125
	v_fma_f32 v92, v92, s5, -v206
	v_exp_f32_e32 v243, v91
	v_add_f32_e32 v66, v239, v66
	s_barrier
	v_max3_f32 v1, v1, v126, v127
	v_fma_f32 v93, v93, s5, -v206
	v_exp_f32_e32 v244, v92
	v_add_f32_e32 v66, v240, v66
	scratch_load_dword v0, off, off offset:140 ; 4-byte Folded Reload
	v_max3_f32 v1, v1, v128, v129
	v_fma_f32 v94, v94, s5, -v206
	v_exp_f32_e32 v245, v93
	v_add_f32_e32 v66, v241, v66
	v_max3_f32 v1, v1, v98, v99
	v_fma_f32 v95, v95, s5, -v206
	v_exp_f32_e32 v246, v94
	v_add_f32_e32 v66, v242, v66
	v_max3_f32 v1, v1, v100, v101
	v_fma_f32 v96, v96, s5, -v206
	v_exp_f32_e32 v247, v95
	v_add_f32_e32 v66, v243, v66
	v_max3_f32 v1, v1, v102, v103
	v_fma_f32 v97, v97, s5, -v206
	v_exp_f32_e32 v248, v96
	v_add_f32_e32 v66, v244, v66
	v_max3_f32 v1, v1, v104, v105
	v_exp_f32_e32 v249, v97
	v_add_f32_e32 v66, v245, v66
	v_max3_f32 v1, v1, v106, v107
	v_add_f32_e32 v66, v246, v66
	v_max3_f32 v1, v1, v108, v109
	v_add_f32_e32 v66, v247, v66
	v_max3_f32 v1, v1, v110, v111
	v_add_f32_e32 v66, v248, v66
	v_max3_f32 v1, v1, v112, v113
	v_add_f32_e32 v66, v249, v66
	ds_bpermute_b32 v195, v194, v1
	ds_bpermute_b32 v67, v194, v66
	v_mul_f32_e32 v50, v50, v193
	v_mul_f32_e32 v51, v51, v193
	v_mul_f32_e32 v52, v52, v193
	s_waitcnt lgkmcnt(1)
	v_max3_f32 v1, v250, v1, v195
	v_mul_f32_e32 v53, v53, v193
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v250, v66, v67
	v_mul_f32_e32 v54, v54, v193
	v_mul_f32_e32 v55, v55, v193
	v_mul_f32_e32 v56, v56, v193
	v_mul_f32_e32 v57, v57, v193
	v_mul_f32_e32 v58, v58, v193
	v_mul_f32_e32 v59, v59, v193
	v_mul_f32_e32 v60, v60, v193
	v_mul_f32_e32 v61, v61, v193
	v_mul_f32_e32 v62, v62, v193
	v_mul_f32_e32 v63, v63, v193
	v_mul_f32_e32 v64, v64, v193
	v_mul_f32_e32 v65, v65, v193
	v_mul_f32_e32 v82, v34, v193
	v_mul_f32_e32 v83, v35, v193
	v_mul_f32_e32 v84, v36, v193
	v_mul_f32_e32 v85, v37, v193
	v_mul_f32_e32 v86, v38, v193
	v_mul_f32_e32 v87, v39, v193
	v_mul_f32_e32 v88, v40, v193
	v_mul_f32_e32 v89, v41, v193
	v_mul_f32_e32 v90, v42, v193
	v_mul_f32_e32 v91, v43, v193
	v_mul_f32_e32 v92, v44, v193
	v_mul_f32_e32 v93, v45, v193
	v_mul_f32_e32 v94, v46, v193
	v_mul_f32_e32 v95, v47, v193
	v_mul_f32_e32 v96, v48, v193
	v_mul_f32_e32 v97, v49, v193
	v_mul_f32_e32 v66, v18, v193
	v_mul_f32_e32 v67, v19, v193
	v_mul_f32_e32 v68, v20, v193
	v_mul_f32_e32 v69, v21, v193
	v_mul_f32_e32 v70, v22, v193
	v_mul_f32_e32 v71, v23, v193
	v_mul_f32_e32 v72, v24, v193
	v_mul_f32_e32 v73, v25, v193
	v_mul_f32_e32 v74, v26, v193
	v_mul_f32_e32 v75, v27, v193
	v_mul_f32_e32 v76, v28, v193
	v_mul_f32_e32 v77, v29, v193
	v_mul_f32_e32 v78, v30, v193
	v_mul_f32_e32 v79, v31, v193
	v_mul_f32_e32 v80, v32, v193
	v_mul_f32_e32 v81, v33, v193
	v_mul_f32_e32 v34, v2, v193
	v_mul_f32_e32 v35, v3, v193
	v_mul_f32_e32 v36, v4, v193
	v_mul_f32_e32 v37, v5, v193
	v_mul_f32_e32 v38, v6, v193
	v_mul_f32_e32 v39, v7, v193
	v_mul_f32_e32 v40, v8, v193
	v_mul_f32_e32 v41, v9, v193
	v_mul_f32_e32 v42, v10, v193
	v_mul_f32_e32 v43, v11, v193
	v_mul_f32_e32 v44, v12, v193
	v_mul_f32_e32 v45, v13, v193
	v_mul_f32_e32 v46, v14, v193
	v_mul_f32_e32 v47, v15, v193
	v_mul_f32_e32 v48, v16, v193
	v_mul_f32_e32 v49, v17, v193
	v_cvt_pkrtz_f16_f32 v2, v207, v208
	v_cvt_pkrtz_f16_f32 v3, v209, v210
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[196:197], v[2:3], v[50:65]
	v_mul_f32_e32 v195, 0xbe0293ee, v1
	v_fmamk_f32 v19, v98, 0x3e0293ee, v195
	v_fmamk_f32 v23, v112, 0x3e0293ee, v195
	v_fmamk_f32 v26, v109, 0x3e0293ee, v195
	v_fmamk_f32 v10, v122, 0x3e0293ee, v195
	v_fmamk_f32 v8, v124, 0x3e0293ee, v195
	v_mfma_f32_32x32x8_f16 v[82:97], v[198:199], v[2:3], v[82:97]
	v_fmamk_f32 v5, v127, 0x3e0293ee, v195
	v_fmamk_f32 v30, v106, 0x3e0293ee, v195
	v_fmamk_f32 v32, v104, 0x3e0293ee, v195
	v_exp_f32_e32 v106, v5
	v_mfma_f32_32x32x8_f16 v[34:49], v[202:203], v[2:3], v[34:49]
	v_fmamk_f32 v15, v117, 0x3e0293ee, v195
	v_fmamk_f32 v6, v126, 0x3e0293ee, v195
	v_exp_f32_e32 v124, v10
	v_mfma_f32_32x32x8_f16 v[66:81], v[200:201], v[2:3], v[66:81]
	v_exp_f32_e32 v126, v15
	v_exp_f32_e32 v104, v30
	v_cvt_pkrtz_f16_f32 v2, v211, v212
	v_cvt_pkrtz_f16_f32 v3, v213, v214
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[178:179], v[2:3], v[50:65]
	v_exp_f32_e32 v122, v26
	v_exp_f32_e32 v127, v8
	v_mfma_f32_32x32x8_f16 v[82:97], v[186:187], v[2:3], v[82:97]
	v_exp_f32_e32 v112, v19
	v_exp_f32_e32 v109, v23
	v_mfma_f32_32x32x8_f16 v[34:49], v[204:205], v[2:3], v[34:49]
	v_exp_f32_e32 v117, v32
	v_fmac_f32_e32 v206, 0xbe0293ee, v1
	v_fmamk_f32 v14, v118, 0x3e0293ee, v195
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[2:3], v[66:81]
	v_exp_f32_e32 v118, v14
	v_fmamk_f32 v13, v119, 0x3e0293ee, v195
	v_exp_f32_e32 v119, v13
	v_cvt_pkrtz_f16_f32 v2, v215, v216
	v_cvt_pkrtz_f16_f32 v3, v217, v218
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[2:3], v[66:81]
	v_fmamk_f32 v16, v116, 0x3e0293ee, v195
	v_fmamk_f32 v17, v115, 0x3e0293ee, v195
	v_fmamk_f32 v21, v114, 0x3e0293ee, v195
	v_fmamk_f32 v25, v110, 0x3e0293ee, v195
	v_fmamk_f32 v22, v100, 0x3e0293ee, v195
	v_exp_f32_e32 v116, v21
	v_mfma_f32_32x32x8_f16 v[82:97], v[180:181], v[2:3], v[82:97]
	v_exp_f32_e32 v110, v22
	v_exp_f32_e32 v114, v25
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[2:3], v[34:49]
	v_exp_f32_e32 v115, v16
	v_fmamk_f32 v20, v129, 0x3e0293ee, v195
	v_fmamk_f32 v18, v99, 0x3e0293ee, v195
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[2:3], v[50:65]
	v_fmamk_f32 v9, v123, 0x3e0293ee, v195
	v_fmamk_f32 v11, v121, 0x3e0293ee, v195
	v_fmamk_f32 v29, v107, 0x3e0293ee, v195
	v_fmamk_f32 v27, v101, 0x3e0293ee, v195
	v_fmamk_f32 v4, v128, 0x3e0293ee, v195
	v_fmamk_f32 v7, v125, 0x3e0293ee, v195
	v_cvt_pkrtz_f16_f32 v2, v219, v220
	v_cvt_pkrtz_f16_f32 v3, v221, v222
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[66:81], v[174:175], v[2:3], v[66:81]
	v_exp_f32_e32 v128, v7
	v_exp_f32_e32 v101, v4
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[2:3], v[82:97]
	v_fmamk_f32 v12, v120, 0x3e0293ee, v195
	v_fmamk_f32 v28, v108, 0x3e0293ee, v195
	v_fmamk_f32 v24, v111, 0x3e0293ee, v195
	v_exp_f32_e32 v107, v27
	v_mfma_f32_32x32x8_f16 v[50:65], v[156:157], v[2:3], v[50:65]
	v_exp_f32_e32 v111, v20
	v_exp_f32_e32 v108, v24
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[2:3], v[34:49]
	v_exp_f32_e32 v120, v28
	v_exp_f32_e32 v129, v18
	v_cvt_pkrtz_f16_f32 v2, v223, v235
	v_cvt_pkrtz_f16_f32 v3, v236, v237
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[66:81], v[158:159], v[2:3], v[66:81]
	v_exp_f32_e32 v121, v29
	v_exp_f32_e32 v99, v9
	v_mfma_f32_32x32x8_f16 v[82:97], v[154:155], v[2:3], v[82:97]
	v_exp_f32_e32 v123, v11
	v_exp_f32_e32 v125, v12
	v_mfma_f32_32x32x8_f16 v[34:49], v[176:177], v[2:3], v[34:49]
	v_fmamk_f32 v98, v102, 0x3e0293ee, v195
	v_exp_f32_e32 v102, v98
	v_fmamk_f32 v31, v105, 0x3e0293ee, v195
	v_mfma_f32_32x32x8_f16 v[50:65], v[152:153], v[2:3], v[50:65]
	v_fmamk_f32 v33, v103, 0x3e0293ee, v195
	v_fmac_f32_e32 v195, 0x3e0293ee, v113
	v_exp_f32_e32 v105, v33
	v_cvt_pkrtz_f16_f32 v2, v238, v239
	v_cvt_pkrtz_f16_f32 v3, v240, v241
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[82:97], v[150:151], v[2:3], v[82:97]
	v_exp_f32_e32 v151, v6
	v_exp_f32_e32 v103, v31
	v_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[2:3], v[66:81]
	v_exp_f32_e32 v100, v195
	v_exp_f32_e32 v113, v206
	v_mfma_f32_32x32x8_f16 v[34:49], v[160:161], v[2:3], v[34:49]
	v_exp_f32_e32 v150, v17
	v_cvt_pkrtz_f16_f32 v6, v242, v243
	v_cvt_pkrtz_f16_f32 v7, v244, v245
	v_mfma_f32_32x32x8_f16 v[50:65], v[148:149], v[2:3], v[50:65]
	v_cvt_pkrtz_f16_f32 v4, v246, v247
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v250, v0, v193
	v_cvt_pkrtz_f16_f32 v5, v248, v249
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[50:65], v[146:147], v[6:7], v[50:65]
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[6:7], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[132:133], v[6:7], v[34:49]
	v_mfma_f32_32x32x8_f16 v[50:65], v[144:145], v[4:5], v[50:65]
	v_mfma_f32_32x32x8_f16 v[82:97], v[142:143], v[6:7], v[82:97]
	v_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[4:5], v[66:81]
	v_mfma_f32_32x32x8_f16 v[34:49], v[130:131], v[4:5], v[34:49]
	v_mfma_f32_32x32x8_f16 v[82:97], v[140:141], v[4:5], v[82:97]
	; sched_barrier mask(0x00000000)
	s_barrier
	s_barrier
	scratch_load_dword v0, off, off offset:152 ; 4-byte Folded Reload
	v_perm_b32 v2, v166, v162, s2
	s_waitcnt vmcnt(0)
	ds_write_b32 v0, v2 offset:16384
	scratch_load_dword v0, off, off offset:156 ; 4-byte Folded Reload
	v_perm_b32 v2, v166, v162, s3
	s_waitcnt vmcnt(0)
	ds_write_b32 v0, v2 offset:16384
	scratch_load_dword v0, off, off offset:160 ; 4-byte Folded Reload
	v_perm_b32 v2, v167, v163, s2
	s_waitcnt vmcnt(0)
	ds_write_b32 v0, v2 offset:16384
	scratch_load_dword v0, off, off offset:164 ; 4-byte Folded Reload
	v_perm_b32 v2, v167, v163, s3
	s_waitcnt vmcnt(0)
	ds_write_b32 v0, v2 offset:16384
	scratch_load_dword v0, off, off offset:168 ; 4-byte Folded Reload
	v_perm_b32 v2, v168, v164, s2
	s_waitcnt vmcnt(0)
	ds_write_b32 v0, v2 offset:16384
	scratch_load_dword v0, off, off offset:172 ; 4-byte Folded Reload
	v_perm_b32 v2, v168, v164, s3
	s_waitcnt vmcnt(0)
	ds_write_b32 v0, v2 offset:16384
	scratch_load_dword v0, off, off offset:176 ; 4-byte Folded Reload
	v_perm_b32 v2, v169, v165, s2
	s_waitcnt vmcnt(0)
	ds_write_b32 v0, v2 offset:16384
	scratch_load_dword v0, off, off offset:180 ; 4-byte Folded Reload
	v_perm_b32 v2, v169, v165, s3
	s_waitcnt vmcnt(0)
	ds_write_b32 v0, v2 offset:16384
	; sched_barrier mask(0x00000000)
	v_add_f32_e32 v2, v116, v150
	v_add_f32_e32 v2, v115, v2
	v_add_f32_e32 v2, v126, v2
	v_add_f32_e32 v2, v118, v2
	v_add_f32_e32 v2, v119, v2
	v_add_f32_e32 v2, v125, v2
	v_add_f32_e32 v2, v123, v2
	v_add_f32_e32 v2, v124, v2
	v_add_f32_e32 v2, v99, v2
	v_add_f32_e32 v2, v127, v2
	v_add_f32_e32 v2, v128, v2
	v_add_f32_e32 v2, v151, v2
	v_add_f32_e32 v2, v106, v2
	v_add_f32_e32 v2, v101, v2
	v_add_f32_e32 v2, v111, v2
	v_add_f32_e32 v2, v112, v2
	v_add_f32_e32 v2, v129, v2
	v_add_f32_e32 v2, v110, v2
	v_add_f32_e32 v2, v107, v2
	v_add_f32_e32 v2, v102, v2
	v_add_f32_e32 v2, v105, v2
	v_add_f32_e32 v2, v117, v2
	v_add_f32_e32 v2, v103, v2
	v_add_f32_e32 v2, v104, v2
	v_add_f32_e32 v2, v121, v2
	v_add_f32_e32 v2, v120, v2
	v_add_f32_e32 v2, v122, v2
	v_add_f32_e32 v2, v114, v2
	v_add_f32_e32 v2, v108, v2
	v_add_f32_e32 v2, v109, v2
	v_add_f32_e32 v5, v100, v2
	ds_bpermute_b32 v6, v194, v5
	v_mul_f32_e32 v2, v50, v113
	v_mul_f32_e32 v3, v51, v113
	v_mul_f32_e32 v4, v52, v113
	v_mul_f32_e32 v7, v55, v113
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v98, v5, v6
	v_mul_f32_e32 v5, v53, v113
	v_mul_f32_e32 v6, v54, v113
	v_mul_f32_e32 v8, v56, v113
	v_mul_f32_e32 v9, v57, v113
	v_mul_f32_e32 v10, v58, v113
	v_mul_f32_e32 v11, v59, v113
	v_mul_f32_e32 v12, v60, v113
	v_mul_f32_e32 v13, v61, v113
	v_mul_f32_e32 v14, v62, v113
	v_mul_f32_e32 v15, v63, v113
	v_mul_f32_e32 v16, v64, v113
	v_mul_f32_e32 v17, v65, v113
	v_mul_f32_e32 v18, v82, v113
	v_mul_f32_e32 v19, v83, v113
	v_mul_f32_e32 v20, v84, v113
	v_mul_f32_e32 v21, v85, v113
	v_mul_f32_e32 v22, v86, v113
	v_mul_f32_e32 v23, v87, v113
	v_mul_f32_e32 v24, v88, v113
	v_mul_f32_e32 v25, v89, v113
	v_mul_f32_e32 v26, v90, v113
	v_mul_f32_e32 v27, v91, v113
	v_mul_f32_e32 v28, v92, v113
	v_mul_f32_e32 v29, v93, v113
	v_mul_f32_e32 v30, v94, v113
	v_mul_f32_e32 v31, v95, v113
	v_mul_f32_e32 v32, v96, v113
	v_mul_f32_e32 v33, v97, v113
	v_mul_f32_e32 v50, v66, v113
	v_mul_f32_e32 v51, v67, v113
	v_mul_f32_e32 v52, v68, v113
	v_mul_f32_e32 v53, v69, v113
	v_mul_f32_e32 v54, v70, v113
	v_mul_f32_e32 v55, v71, v113
	v_mul_f32_e32 v56, v72, v113
	v_mul_f32_e32 v57, v73, v113
	v_mul_f32_e32 v58, v74, v113
	v_mul_f32_e32 v59, v75, v113
	v_mul_f32_e32 v60, v76, v113
	v_mul_f32_e32 v61, v77, v113
	v_mul_f32_e32 v62, v78, v113
	v_mul_f32_e32 v63, v79, v113
	v_mul_f32_e32 v64, v80, v113
	v_mul_f32_e32 v65, v81, v113
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
	v_fmac_f32_e32 v98, v250, v113
	v_cvt_pkrtz_f16_f32 v66, v116, v150
	v_cvt_pkrtz_f16_f32 v70, v124, v99
	s_barrier
	v_cvt_pkrtz_f16_f32 v67, v115, v126
	v_cvt_pkrtz_f16_f32 v68, v118, v119
	v_cvt_pkrtz_f16_f32 v69, v125, v123
	v_cvt_pkrtz_f16_f32 v71, v127, v128
	v_cvt_pkrtz_f16_f32 v72, v151, v106
	v_cvt_pkrtz_f16_f32 v73, v101, v111
	v_cvt_pkrtz_f16_f32 v74, v112, v129
	v_cvt_pkrtz_f16_f32 v75, v110, v107
	v_cvt_pkrtz_f16_f32 v76, v102, v105
	v_cvt_pkrtz_f16_f32 v77, v117, v103
	v_cvt_pkrtz_f16_f32 v78, v104, v121
	v_cvt_pkrtz_f16_f32 v79, v120, v122
	v_cvt_pkrtz_f16_f32 v80, v114, v108
	v_cvt_pkrtz_f16_f32 v81, v109, v100
	; sched_barrier mask(0x00000000)
	s_barrier
	s_barrier
	scratch_load_dword v0, off, off offset:84 ; 4-byte Folded Reload
	ds_read_b64 v[82:83], v255 offset:16384
	ds_read_b64 v[92:93], v254 offset:16384
	ds_read_b64 v[94:95], v232 offset:16384
	ds_read_b64 v[102:103], v224 offset:16384
	ds_read_b64 v[106:107], v225 offset:16384
	ds_read_b64 v[108:109], v226 offset:16384
	ds_read_b64 v[110:111], v227 offset:16384
	ds_read_b64 v[112:113], v228 offset:16384
	ds_read_b64 v[114:115], v229 offset:16384
	ds_read_b64 v[116:117], v230 offset:16384
	ds_read_b64 v[118:119], v231 offset:16384
	ds_read_b64 v[120:121], v233 offset:16384
	ds_read_b64 v[122:123], v234 offset:16384
	ds_read_b64 v[124:125], v251 offset:16384
	ds_read_b64 v[126:127], v252 offset:16384
	ds_read_b64 v[146:147], v192 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[84:85], v0 offset:16384
	scratch_load_dword v0, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[86:87], v0 offset:16384
	scratch_load_dword v0, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[88:89], v0 offset:16384
	scratch_load_dword v0, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[90:91], v0 offset:16384
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[96:97], v0 offset:16384
	scratch_load_dword v0, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[100:101], v0 offset:16384
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[104:105], v0 offset:16384
	scratch_load_dword v0, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[128:129], v0 offset:16384
	scratch_load_dword v0, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[130:131], v0 offset:16384
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[132:133], v0 offset:16384
	scratch_load_dword v0, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[134:135], v0 offset:16384
	scratch_load_dword v0, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[136:137], v0 offset:16384
	scratch_load_dword v0, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[138:139], v0 offset:16384
	scratch_load_dword v0, off, off offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[140:141], v0 offset:16384
	scratch_load_dword v0, off, off offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[142:143], v0 offset:16384
	scratch_load_dword v0, off, off offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[144:145], v0 offset:16384
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[2:17], v[82:83], v[66:67], v[2:17]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[18:33], v[100:101], v[66:67], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[116:117], v[66:67], v[50:65]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[34:49], v[132:133], v[66:67], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[84:85], v[68:69], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[102:103], v[68:69], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[118:119], v[68:69], v[50:65]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[34:49], v[134:135], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[70:71], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[104:105], v[70:71], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[120:121], v[70:71], v[50:65]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[34:49], v[136:137], v[70:71], v[34:49]
	scratch_load_dword v70, off, off offset:136 ; 4-byte Folded Reload
	s_barrier
	v_mfma_f32_32x32x8_f16 v[2:17], v[88:89], v[72:73], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[106:107], v[72:73], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[122:123], v[72:73], v[50:65]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[34:49], v[138:139], v[72:73], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[90:91], v[74:75], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[74:75], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[124:125], v[74:75], v[50:65]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[34:49], v[140:141], v[74:75], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[92:93], v[76:77], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[110:111], v[76:77], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[126:127], v[76:77], v[50:65]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[34:49], v[142:143], v[76:77], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[94:95], v[78:79], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[112:113], v[78:79], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[128:129], v[78:79], v[50:65]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[144:145], v[78:79], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[96:97], v[80:81], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[114:115], v[80:81], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[130:131], v[80:81], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[146:147], v[80:81], v[34:49]
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v66, v70, 2, 0
	s_barrier
	s_barrier
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	scratch_load_dword v0, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v98
	s_nop 1
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v98, v69
	v_log_f32_e32 v69, v69
	v_mov_b32_e32 v68, 0x42000000
	v_or_b32_e32 v67, s34, v70
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v68, 0, v68, vcc
	v_cmp_gt_i32_e64 s[8:9], s2, v67
	v_sub_f32_e32 v67, v69, v68
	v_add_f32_e32 v67, v1, v67
	ds_write_b32 v66, v67
	v_mov_b32_e32 v67, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_sub_i32 s2, 0x4000, s34
	v_bfrev_b32_e32 v69, 1
	s_and_b32 s5, s19, 0xffff
	s_mov_b32 s6, s14
	s_mov_b32 s7, s15
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v67, v67, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v68, 0, v67
	ds_read_b32 v68, v68
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v67, v69, v67, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v68, v67, s[4:7], 0 offen
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_9:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v98
	s_nop 1
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v98, v68
	v_log_f32_e32 v68, v68
	v_mov_b32_e32 v67, 0x42000000
	v_cndmask_b32_e32 v67, 0, v67, vcc
	s_and_b32 s5, s19, 0xffff
	v_sub_f32_e32 v67, v68, v67
	v_add_f32_e32 v1, v1, v67
	ds_write_b32 v66, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v0, off, off offset:128 ; 4-byte Folded Reload
	v_mov_b32_e32 v1, 2
	v_bfrev_b32_e32 v66, 1
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v0, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v1, 0, v0
	ds_read_b32 v1, v1
	v_cndmask_b32_e64 v0, v66, v0, s[0:1]
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v1, v0, s[4:7], 0 offen
.LBB0_10:                               ; %.critedge
	v_div_scale_f32 v0, s[0:1], v98, v98, 1.0
	v_rcp_f32_e32 v0, v0
	v_div_scale_f32 v1, vcc, 1.0, v98, 1.0
	v_mov_b32_e32 v66, v47
	v_mul_f32_e32 v0, v1, v0
	s_nop 1
	v_div_fmas_f32 v0, 0, 0, v0
	v_div_fixup_f32 v0, v0, v98, 1.0
	v_mov_b32_e32 v67, v48
	v_fma_mixlo_f16 v68, v0, v49, 0
	v_pk_mul_f32 v[48:49], v[0:1], v[66:67] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v66, v0, v46, 0
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v44
	v_fma_mixlo_f16 v67, v0, v45, 0
	v_pk_mul_f32 v[44:45], v[0:1], v[46:47] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v46, v0, v42, 0
	v_mov_b32_e32 v42, v39
	v_mov_b32_e32 v43, v40
	v_fma_mixlo_f16 v47, v0, v41, 0
	v_pk_mul_f32 v[40:41], v[0:1], v[42:43] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v42, v0, v38, 0
	v_mov_b32_e32 v38, v35
	v_mov_b32_e32 v39, v36
	v_fma_mixlo_f16 v43, v0, v37, 0
	v_pk_mul_f32 v[36:37], v[0:1], v[38:39] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v38, v0, v34, 0
	v_mov_b32_e32 v34, v63
	v_mov_b32_e32 v35, v64
	v_pk_mul_f32 v[34:35], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v63, v35
	v_cvt_f16_f32_e32 v64, v34
	v_mov_b32_e32 v34, v59
	v_mov_b32_e32 v35, v60
	v_pk_mul_f32 v[34:35], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v59, v35
	v_cvt_f16_f32_e32 v60, v34
	v_mov_b32_e32 v34, v55
	v_mov_b32_e32 v35, v56
	v_pk_mul_f32 v[34:35], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v55, v35
	v_cvt_f16_f32_e32 v56, v34
	v_mov_b32_e32 v34, v51
	v_mov_b32_e32 v35, v52
	v_pk_mul_f32 v[34:35], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v51, v35
	v_cvt_f16_f32_e32 v52, v34
	v_mov_b32_e32 v34, v31
	v_mov_b32_e32 v35, v32
	v_fma_mixlo_f16 v39, v0, v65, 0
	v_fma_mixlo_f16 v65, v0, v33, 0
	v_pk_mul_f32 v[32:33], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v34, v0, v30, 0
	v_mov_b32_e32 v30, v27
	v_mov_b32_e32 v31, v28
	v_fma_mixlo_f16 v35, v0, v29, 0
	v_pk_mul_f32 v[28:29], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v30, v0, v26, 0
	v_mov_b32_e32 v26, v23
	v_mov_b32_e32 v27, v24
	v_fma_mixlo_f16 v31, v0, v25, 0
	v_pk_mul_f32 v[24:25], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v26, v0, v22, 0
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v20
	v_fma_mixlo_f16 v27, v0, v21, 0
	v_pk_mul_f32 v[20:21], v[0:1], v[22:23] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v22, v0, v18, 0
	v_mov_b32_e32 v18, v15
	v_mov_b32_e32 v19, v16
	v_fma_mixlo_f16 v23, v0, v17, 0
	v_pk_mul_f32 v[16:17], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v18, v0, v14, 0
	v_mov_b32_e32 v14, v11
	v_mov_b32_e32 v15, v12
	v_fma_mixlo_f16 v19, v0, v13, 0
	v_pk_mul_f32 v[12:13], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v14, v0, v10, 0
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v8
	v_fma_mixlo_f16 v15, v0, v9, 0
	v_pk_mul_f32 v[8:9], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v10, v0, v6, 0
	v_fma_mixlo_f16 v1, v0, v5, 0
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v4
	v_pk_mul_f32 v[4:5], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v3, v5
	s_mov_b32 s4, 0x5040100
	s_mul_i32 s0, s24, s18
	s_ashr_i32 s1, s0, 31
	v_perm_b32 v1, v1, v3, s4
	scratch_load_dword v3, off, off offset:124 ; 4-byte Folded Reload
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s25, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s16, s34
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_f16_f32_e32 v4, v4
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s16, 0x3fff
	v_fma_mixlo_f16 v62, v0, v62, 0
	v_fma_mixlo_f16 v61, v0, v61, 0
	v_fma_mixlo_f16 v58, v0, v58, 0
	v_fma_mixlo_f16 v57, v0, v57, 0
	v_fma_mixlo_f16 v54, v0, v54, 0
	v_fma_mixlo_f16 v53, v0, v53, 0
	v_fma_mixlo_f16 v50, v0, v50, 0
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v8, v8
	v_fma_mixlo_f16 v0, v0, v2, 0
	v_mul_lo_u32 v2, s16, v70
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s2, s1
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v12, v12
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_pack_b32_f16 v0, v0, v4
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v29, v29
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v33, v33
	v_cvt_f16_f32_e32 v32, v32
	v_cvt_f16_f32_e32 v37, v37
	v_cvt_f16_f32_e32 v36, v36
	v_cvt_f16_f32_e32 v41, v41
	v_cvt_f16_f32_e32 v40, v40
	v_cvt_f16_f32_e32 v45, v45
	v_cvt_f16_f32_e32 v44, v44
	v_cvt_f16_f32_e32 v49, v49
	v_cvt_f16_f32_e32 v48, v48
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v2, v2, v3, 1
	v_bfrev_b32_e32 v3, 1
	v_cndmask_b32_e64 v4, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 16, v2
	v_pack_b32_f16 v0, v10, v8
	v_perm_b32 v1, v15, v9, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 32, v2
	v_pack_b32_f16 v0, v14, v12
	v_perm_b32 v1, v19, v13, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 48, v2
	v_pack_b32_f16 v0, v18, v16
	v_perm_b32 v1, v23, v17, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 64, v2
	v_pack_b32_f16 v0, v22, v20
	v_perm_b32 v1, v27, v21, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x50, v2
	v_pack_b32_f16 v0, v26, v24
	v_perm_b32 v1, v31, v25, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x60, v2
	v_pack_b32_f16 v0, v30, v28
	v_perm_b32 v1, v35, v29, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x70, v2
	v_pack_b32_f16 v0, v34, v32
	v_perm_b32 v1, v65, v33, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x80, v2
	v_pack_b32_f16 v0, v50, v52
	v_perm_b32 v1, v53, v51, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x90, v2
	v_pack_b32_f16 v0, v54, v56
	v_perm_b32 v1, v57, v55, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xa0, v2
	v_pack_b32_f16 v0, v58, v60
	v_perm_b32 v1, v61, v59, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xb0, v2
	v_pack_b32_f16 v0, v62, v64
	v_perm_b32 v1, v39, v63, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xc0, v2
	v_pack_b32_f16 v0, v38, v36
	v_perm_b32 v1, v43, v37, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xd0, v2
	v_pack_b32_f16 v0, v42, v40
	v_perm_b32 v1, v47, v41, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xe0, v2
	v_pack_b32_f16 v0, v46, v44
	v_perm_b32 v1, v67, v45, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	v_add_u32_e32 v2, 0xf0, v2
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_pack_b32_f16 v0, v66, v48
	v_perm_b32 v1, v68, v49, s4
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 188
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 44
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
	.set attn_fwd.numbered_sgpr, 44
	.set attn_fwd.private_seg_size, 188
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16056
; TotalNumSgprs: 50
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 188
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 50
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
    .private_segment_fixed_size: 188
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 48
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
