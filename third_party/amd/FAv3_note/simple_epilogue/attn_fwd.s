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
	s_mul_i32 s8, s12, s18
	s_ashr_i32 s9, s8, 31
	s_lshl_b32 s16, s16, 8
	s_lshl_b64 s[8:9], s[8:9], 1
	s_add_u32 s8, s2, s8
	s_mul_i32 s2, s13, s17
	s_addc_u32 s9, s3, s9
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshrrev_b32_e32 v3, 4, v0
	s_add_u32 s8, s8, s2
	s_mul_i32 s2, s14, s16
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	v_or_b32_e32 v2, 0xa0, v3
	s_addc_u32 s9, s9, s3
	s_ashr_i32 s3, s2, 31
	v_or_b32_e32 v21, s16, v2
	s_lshl_b32 s12, s14, 6
	v_mul_lo_u32 v22, s14, v2
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v2, 3, v0
	v_or_b32_e32 v4, 0xe0, v3
	s_add_u32 s20, s8, s2
	v_and_b32_e32 v2, 0x78, v2
	s_mul_i32 s34, s15, s18
	v_or_b32_e32 v29, s16, v4
	v_mul_lo_u32 v30, s14, v4
	s_addc_u32 s8, s9, s3
	v_mad_u64_u32 v[4:5], s[2:3], s14, v3, v[2:3]
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 1
	s_add_u32 s9, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s36, s24, s17
	s_addc_u32 s13, s5, s3
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[2:3], s[36:37], 1
	s_add_u32 s28, s9, s2
	s_mul_i32 s2, s26, s18
	s_addc_u32 s45, s13, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[38:39], s[2:3], 1
	s_add_u32 s9, s6, s38
	s_mul_i32 s2, s27, s17
	s_addc_u32 s13, s7, s39
	s_ashr_i32 s3, s2, 31
	v_or_b32_e32 v36, 32, v3
	v_or_b32_e32 v6, s16, v3
	s_lshl_b64 s[40:41], s[2:3], 1
	s_movk_i32 s2, 0x4000
	v_or_b32_e32 v7, s16, v36
	v_mul_lo_u32 v8, s14, v36
	v_add_u32_e32 v14, s12, v4
	s_add_u32 s24, s9, s40
	v_lshlrev_b32_e32 v4, 1, v4
	v_bfrev_b32_e32 v32, 1
	v_cmp_gt_i32_e32 vcc, s2, v6
	v_or_b32_e32 v1, 0x60, v3
	v_or_b32_e32 v12, 64, v6
	s_addc_u32 s46, s13, s41
	s_and_b32 s3, s14, 0x3fff
	v_cndmask_b32_e32 v15, v32, v4, vcc
	v_add_lshl_u32 v4, v8, v2, 1
	v_cmp_gt_i32_e32 vcc, s2, v7
	v_or_b32_e32 v13, s16, v1
	v_mul_lo_u32 v1, s14, v1
	v_add_u32_e32 v31, s12, v14
	s_bitset1_b32 s3, 14
	v_cndmask_b32_e32 v16, v32, v4, vcc
	v_lshlrev_b32_e32 v14, 1, v14
	v_cmp_gt_i32_e32 vcc, s2, v12
	s_and_b32 s8, s8, 0xffff
	s_lshl_b32 s3, s3, 16
	v_cndmask_b32_e32 v23, v32, v14, vcc
	v_add_lshl_u32 v1, v1, v2, 1
	v_cmp_gt_i32_e32 vcc, s2, v13
	v_or_b32_e32 v20, 0x80, v6
	s_or_b32 s21, s8, s3
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v1, v32, v1, vcc
	buffer_load_dwordx4 v[8:11], v16, s[20:23], 0 offen
                                        ; kill: killed $vgpr16
	s_nop 0
	buffer_load_dwordx4 v[16:19], v1, s[20:23], 0 offen
                                        ; kill: killed $vgpr1
	v_lshlrev_b32_e32 v1, 1, v31
	v_cmp_gt_i32_e32 vcc, s2, v20
	v_or_b32_e32 v28, 0xc0, v6
	buffer_load_dwordx4 v[4:7], v15, s[20:23], 0 offen
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_add_lshl_u32 v20, v22, v2, 1
	v_cmp_gt_i32_e32 vcc, s2, v21
                                        ; kill: killed $vgpr15
	buffer_load_dwordx4 v[12:15], v23, s[20:23], 0 offen
                                        ; kill: killed $vgpr23
	s_nop 0
	v_cndmask_b32_e32 v33, v32, v20, vcc
	buffer_load_dwordx4 v[20:23], v1, s[20:23], 0 offen
                                        ; kill: killed $vgpr1
	v_add_lshl_u32 v1, v31, s12, 1
	v_cmp_gt_i32_e32 vcc, s2, v28
	s_nop 1
	v_cndmask_b32_e32 v1, v32, v1, vcc
	v_add_lshl_u32 v28, v30, v2, 1
	v_cmp_gt_i32_e32 vcc, s2, v29
	s_nop 1
	v_cndmask_b32_e32 v37, v32, v28, vcc
	buffer_load_dwordx4 v[24:27], v33, s[20:23], 0 offen
                                        ; kill: killed $vgpr33
	buffer_load_dwordx4 v[28:31], v1, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[32:35], v37, s[20:23], 0 offen
	v_and_b32_e32 v40, 0x100, v0
	scratch_store_dword off, v40, off offset:124 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v53, 3, v40
	v_lshrrev_b32_e32 v40, 1, v0
                                        ; kill: killed $vgpr37
	v_and_b32_e32 v37, 64, v0
	v_and_b32_e32 v41, 24, v40
	v_lshrrev_b32_e32 v42, 1, v37
	v_and_b32_e32 v38, 0x80, v0
	v_xor_b32_e32 v41, v41, v2
	v_lshrrev_b32_e32 v43, 1, v38
	v_xor_b32_e32 v41, v41, v42
	v_lshrrev_b32_e32 v39, 3, v0
	v_xor_b32_e32 v41, v41, v43
	v_and_b32_e32 v234, 4, v39
	v_mul_lo_u32 v39, s25, v3
	v_lshl_add_u32 v41, v41, 1, 0
	v_lshlrev_b32_e32 v3, 8, v3
	v_add_u32_e32 v197, v41, v3
	v_and_b32_e32 v54, 31, v0
	s_movk_i32 s19, 0xe0
	v_bfe_u32 v3, v0, 5, 1
                                        ; kill: killed $vgpr1
	v_and_b32_e32 v1, 1, v0
	v_cmp_eq_u32_e64 s[2:3], 0, v1
	v_and_b32_e32 v1, 2, v0
	v_cmp_eq_u32_e64 s[8:9], 0, v1
	v_and_b32_e32 v1, 4, v0
	scratch_store_dword off, v42, off offset:116 ; 4-byte Folded Spill
	scratch_store_dword off, v43, off offset:120 ; 4-byte Folded Spill
	v_cmp_eq_u32_e64 s[12:13], 0, v1
	v_and_b32_e32 v1, 8, v0
	s_load_dword s33, s[0:1], 0x48
	v_cmp_eq_u32_e64 s[14:15], 0, v1
	v_and_b32_e32 v1, 16, v0
	v_lshrrev_b32_e32 v50, 3, v1
	v_lshrrev_b32_e32 v51, 3, v37
	v_or3_b32 v37, v50, v234, v51
	v_lshrrev_b32_e32 v52, 3, v38
	v_or3_b32 v37, v37, v52, v53
	v_or_b32_e32 v38, 1, v37
	v_bfe_i32 v136, v0, 2, 1
	v_bfe_i32 v137, v0, 3, 1
	s_lshl_b32 s42, s25, 6
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s44, s33, 6
                                        ; kill: killed $sgpr20_sgpr21
	v_bfe_i32 v94, v0, 0, 1
	v_bfe_i32 v95, v0, 1, 1
	s_movk_i32 s47, 0x100
	v_mul_lo_u32 v36, s25, v36
	s_waitcnt vmcnt(10)
	ds_write_b128 v197, v[8:11] offset:8192
	v_or_b32_e32 v8, 4, v3
	v_or_b32_e32 v9, 6, v3
	v_or_b32_e32 v10, 8, v3
	v_or_b32_e32 v11, 10, v3
	s_waitcnt vmcnt(8)
	ds_write_b128 v197, v[4:7]
	v_and_b32_e32 v5, 15, v0
	v_and_or_b32 v4, v40, s19, v54
	v_xor_b32_e32 v6, v3, v5
	v_or_b32_e32 v7, 2, v3
	s_waitcnt vmcnt(7)
	ds_write_b128 v197, v[12:15] offset:16384
	v_xor_b32_e32 v7, v7, v5
	v_xor_b32_e32 v8, v8, v5
	v_or_b32_e32 v12, 12, v3
	v_or_b32_e32 v3, 14, v3
	v_lshl_add_u32 v4, v4, 8, 0
	v_lshlrev_b32_e32 v13, 4, v6
	v_xor_b32_e32 v9, v9, v5
	v_xor_b32_e32 v10, v10, v5
	v_xor_b32_e32 v11, v11, v5
	v_xor_b32_e32 v12, v12, v5
	v_xor_b32_e32 v3, v3, v5
	v_add_u32_e32 v5, v4, v13
	v_lshlrev_b32_e32 v14, 4, v7
	v_lshlrev_b32_e32 v15, 4, v8
	ds_write_b128 v197, v[16:19] offset:24576
	s_waitcnt vmcnt(6)
	ds_write_b128 v197, v[20:23] offset:32768
	s_waitcnt vmcnt(5)
	ds_write_b128 v197, v[24:27] offset:40960
	s_waitcnt vmcnt(4)
	ds_write_b128 v197, v[28:31] offset:49152
	s_waitcnt vmcnt(3)
	ds_write_b128 v197, v[32:35] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v6, v4, v14
	ds_read_b128 v[96:99], v5
	v_add_u32_e32 v5, v4, v15
	v_lshlrev_b32_e32 v16, 4, v9
	v_lshlrev_b32_e32 v17, 4, v10
	ds_read_b128 v[100:103], v6
	v_add_u32_e32 v6, v4, v16
	ds_read_b128 v[104:107], v5
	v_add_u32_e32 v5, v4, v17
	v_lshlrev_b32_e32 v18, 4, v11
	v_lshlrev_b32_e32 v12, 4, v12
	v_lshlrev_b32_e32 v3, 4, v3
	ds_read_b128 v[108:111], v6
	v_add_u32_e32 v6, v4, v18
	ds_read_b128 v[112:115], v5
	v_add_u32_e32 v5, v4, v12
	v_add_u32_e32 v4, v4, v3
	ds_read_b128 v[116:119], v6
	ds_read_b128 v[120:123], v5
	ds_read_b128 v[124:127], v4
	v_mul_lo_u32 v19, s33, v37
	v_mul_lo_u32 v20, s33, v38
	; sched_barrier mask(0x00000000)
	s_and_b32 s19, s25, 0x3fff
	s_bitset1_b32 s19, 14
	s_and_b32 s20, s45, 0xffff
	s_lshl_b32 s19, s19, 16
	s_or_b32 s29, s20, s19
	s_mov_b32 s30, s22
	s_mov_b32 s31, s23
	v_add_lshl_u32 v138, v39, v2, 1
	v_add_lshl_u32 v139, v36, v2, 1
	buffer_load_dwordx4 v[4:7], v138, s[28:31], 0 offen
	buffer_load_dwordx4 v[8:11], v139, s[28:31], 0 offen
                                        ; kill: killed $sgpr30_sgpr31 killed $sgpr29
	; sched_barrier mask(0x00000000)
	s_ashr_i32 s43, s42, 31
	s_lshl_b64 s[30:31], s[42:43], 1
	s_add_u32 s20, s28, s30
	s_addc_u32 s28, s45, s31
	s_and_b32 s21, s28, 0xffff
	s_or_b32 s21, s21, s19
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[42:45], v138, s[20:23], 0 offen
	buffer_load_dwordx4 v[46:49], v139, s[20:23], 0 offen
                                        ; kill: killed $sgpr21
	s_waitcnt vmcnt(3)
	ds_write_b128 v197, v[4:7]
	s_waitcnt vmcnt(2)
	ds_write_b128 v197, v[8:11] offset:8192
	; sched_barrier mask(0x00000000)
	s_and_b32 s21, s33, 0x3fff
	s_bitset1_b32 s21, 14
	s_and_b32 s25, s46, 0xffff
	s_lshl_b32 s33, s21, 16
	s_or_b32 s25, s25, s33
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	v_add_lshl_u32 v235, v19, v2, 1
	v_add_lshl_u32 v238, v20, v2, 1
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[34:37], v235, s[24:27], 0 offen
	buffer_load_dwordx4 v[38:41], v238, s[24:27], 0 offen
	v_lshlrev_b32_e32 v2, 8, v54
	v_add3_u32 v202, 0, v13, v2
	v_add3_u32 v203, 0, v14, v2
	v_add3_u32 v204, 0, v15, v2
	v_add3_u32 v205, 0, v16, v2
	v_add3_u32 v207, 0, v18, v2
	v_add3_u32 v208, 0, v12, v2
	v_add3_u32 v209, 0, v3, v2
	v_add3_u32 v206, 0, v17, v2
	ds_read_b128 v[16:19], v202
	ds_read_b128 v[20:23], v203
	ds_read_b128 v[24:27], v204
	ds_read_b128 v[28:31], v205
	ds_read_b128 v[78:81], v207
	ds_read_b128 v[82:85], v207 offset:8192
	ds_read_b128 v[86:89], v208
	ds_read_b128 v[90:93], v208 offset:8192
	ds_read_b128 v[128:131], v209
	ds_read_b128 v[132:135], v209 offset:8192
	scratch_store_dword off, v54, off offset:112 ; 4-byte Folded Spill
	ds_read_b128 v[54:57], v202 offset:8192
	ds_read_b128 v[58:61], v203 offset:8192
	ds_read_b128 v[62:65], v204 offset:8192
	ds_read_b128 v[66:69], v205 offset:8192
	ds_read_b128 v[70:73], v206
	ds_read_b128 v[74:77], v206 offset:8192
                                        ; kill: killed $sgpr26_sgpr27 killed $sgpr25
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[2:17], v[16:17], v[96:97], 0
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[98:99], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[100:101], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[22:23], v[102:103], v[2:17]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[2:17], v[24:25], v[104:105], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[106:107], v[2:17]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[108:109], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[110:111], v[2:17]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[18:33], v[54:55], v[96:97], 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[56:57], v[98:99], v[18:33]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[70:71], v[112:113], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[58:59], v[100:101], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[72:73], v[114:115], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[60:61], v[102:103], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[78:79], v[116:117], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[62:63], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[80:81], v[118:119], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[64:65], v[106:107], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[120:121], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[108:109], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[88:89], v[122:123], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[68:69], v[110:111], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[128:129], v[124:125], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[112:113], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[130:131], v[126:127], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[76:77], v[114:115], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[82:83], v[116:117], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[84:85], v[118:119], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[90:91], v[120:121], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[92:93], v[122:123], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[132:133], v[124:125], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[134:135], v[126:127], v[18:33]
	; sched_barrier mask(0x00000000)
	s_add_u32 s20, s20, s30
	s_addc_u32 s21, s28, s31
	s_ashr_i32 s45, s44, 31
	s_lshl_b64 s[28:29], s[44:45], 1
	s_add_u32 s24, s24, s28
	s_addc_u32 s25, s46, s29
	s_and_b32 s21, s21, 0xffff
	s_or_b32 s21, s21, s19
	s_barrier
	buffer_load_dwordx4 v[132:135], v138, s[20:23], 0 offen
	buffer_load_dwordx4 v[128:131], v139, s[20:23], 0 offen
	v_mov_b32_e32 v236, v138
	v_mov_b32_e32 v237, v139
	s_waitcnt vmcnt(6)
	ds_write_b128 v197, v[42:45]
	s_waitcnt vmcnt(5)
	ds_write_b128 v197, v[46:49] offset:8192
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
	v_xor_b32_e32 v210, 0x80, v42
	ds_bpermute_b32 v44, v210, v43
	v_mov_b32_e32 v42, 0xff800000
	s_mov_b32 s43, 0x3e0293ee
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v253, v43, v44, v42
	v_mul_f32_e32 v43, 0xbe0293ee, v253
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
	v_fmac_f32_e32 v42, 0xbe0293ee, v253
	; sched_barrier mask(0x00000000)
	s_and_b32 s20, s25, 0xffff
	s_or_b32 s25, s20, s33
	s_barrier
	v_and_b32_e32 v46, 0x808, v136
	v_and_b32_e32 v48, 0x1010, v137
	buffer_load_dwordx4 v[136:139], v235, s[24:27], 0 offen
	buffer_load_dwordx4 v[140:143], v238, s[24:27], 0 offen
	v_and_b32_e32 v33, 0x220, v94
	v_and_b32_e32 v44, 0x404, v95
	v_or_b32_e32 v45, v33, v44
	v_or_b32_e32 v47, v45, v46
	v_or_b32_e32 v51, v51, v234
	v_or3_b32 v49, v48, v50, v47
	v_or3_b32 v51, v51, v52, v53
	v_xor_b32_e32 v49, v51, v49
	s_mov_b32 s27, 0x5040100
	v_lshl_add_u32 v239, v49, 1, 0
	s_waitcnt vmcnt(5)
	v_perm_b32 v49, v38, v34, s27
	ds_write_b32 v239, v49 offset:16384
	v_or_b32_e32 v49, 0x44, v33
	s_mov_b32 s46, 0x7060302
	v_xor_b32_e32 v49, v49, v44
	v_perm_b32 v34, v38, v34, s46
	v_perm_b32 v38, v39, v35, s27
	v_perm_b32 v35, v39, v35, s46
	v_perm_b32 v39, v40, v36, s27
	v_perm_b32 v36, v40, v36, s46
	v_perm_b32 v40, v41, v37, s27
	v_perm_b32 v37, v41, v37, s46
	v_or_b32_e32 v41, v48, v46
	v_or3_b32 v49, v50, v49, v41
	v_xor_b32_e32 v49, v51, v49
	v_lshl_add_u32 v240, v49, 1, 0
	ds_read_b128 v[78:81], v202
	ds_read_b128 v[144:147], v202 offset:8192
	ds_read_b128 v[82:85], v203
	ds_read_b128 v[148:151], v203 offset:8192
	ds_read_b128 v[86:89], v204
	ds_read_b128 v[152:155], v204 offset:8192
	ds_read_b128 v[90:93], v205
	ds_read_b128 v[156:159], v205 offset:8192
	ds_read_b128 v[176:179], v206
	ds_read_b128 v[160:163], v206 offset:8192
	ds_read_b128 v[180:183], v207
	ds_read_b128 v[164:167], v207 offset:8192
	ds_read_b128 v[184:187], v208
	ds_read_b128 v[168:171], v208 offset:8192
	ds_read_b128 v[188:191], v209
	ds_read_b128 v[172:175], v209 offset:8192
	ds_write_b32 v240, v34 offset:16384
	v_or_b32_e32 v34, 0x88, v45
	v_xor_b32_e32 v34, v34, v46
	v_or3_b32 v34, v50, v34, v48
	v_xor_b32_e32 v34, v51, v34
	v_lshl_add_u32 v241, v34, 1, 0
	ds_write_b32 v241, v38 offset:16384
	v_or_b32_e32 v34, 0xcc, v33
	v_or_b32_e32 v38, v46, v44
	v_xor_b32_e32 v34, v38, v34
	v_or3_b32 v34, v50, v34, v48
	v_xor_b32_e32 v34, v51, v34
	v_lshl_add_u32 v242, v34, 1, 0
	v_or_b32_e32 v34, 0x110, v47
	v_xor_b32_e32 v34, v34, v48
	v_or_b32_e32 v34, v34, v50
	v_xor_b32_e32 v34, v51, v34
	v_lshl_add_u32 v243, v34, 1, 0
	v_or_b32_e32 v34, 0x154, v33
	v_xor_b32_e32 v34, v34, v44
	v_or_b32_e32 v34, v34, v46
	v_xor_b32_e32 v34, v34, v48
	v_or_b32_e32 v34, v34, v50
	v_xor_b32_e32 v34, v51, v34
	v_lshl_add_u32 v244, v34, 1, 0
	v_or_b32_e32 v34, 0x198, v45
	v_xor_b32_e32 v34, v41, v34
	v_or_b32_e32 v34, v34, v50
	v_xor_b32_e32 v34, v51, v34
	v_lshl_add_u32 v245, v34, 1, 0
	v_or_b32_e32 v34, v38, v48
	v_or_b32_e32 v33, 0x1dc, v33
	v_xor_b32_e32 v33, v34, v33
	v_or_b32_e32 v33, v33, v50
	v_xor_b32_e32 v33, v51, v33
	s_movk_i32 s20, 0xff
	v_lshl_add_u32 v246, v33, 1, 0
	v_cmp_gt_u32_e32 vcc, s47, v0
	v_cmp_lt_u32_e64 s[20:21], s20, v0
	ds_write_b32 v242, v35 offset:16384
	ds_write_b32 v243, v39 offset:16384
	ds_write_b32 v244, v36 offset:16384
	ds_write_b32 v245, v40 offset:16384
	ds_write_b32 v246, v37 offset:16384
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[24:25], s[20:21]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[24:25]
	v_exp_f32_e32 v254, v2
	v_exp_f32_e32 v194, v4
	v_mov_b32_e32 v0, 0x44
	v_mov_b32_e32 v2, 0x88
	v_mov_b32_e32 v4, 0x110
	v_cndmask_b32_e64 v0, v0, 0, s[2:3]
	v_cndmask_b32_e64 v2, v2, 0, s[8:9]
	v_cndmask_b32_e64 v4, v4, 0, s[12:13]
	s_load_dwordx2 s[24:25], s[0:1], 0x4c
	s_load_dword s26, s[0:1], 0x54
	v_exp_f32_e32 v255, v3
	v_exp_f32_e32 v196, v6
	v_exp_f32_e32 v192, v8
	v_exp_f32_e32 v199, v12
	v_exp_f32_e32 v211, v13
	v_exp_f32_e32 v221, v22
	v_exp_f32_e32 v228, v29
	v_or_b32_e32 v3, v0, v2
	v_mov_b32_e32 v6, 0x220
	v_mov_b32_e32 v8, 0x404
	v_cmp_eq_u32_e64 s[0:1], 0, v1
	v_or_b32_e32 v12, 24, v0
	v_or_b32_e32 v13, v4, v2
	v_or_b32_e32 v22, 0x818, v0
	v_or_b32_e32 v29, 0x1018, v0
	v_or_b32_e32 v36, 0x1818, v0
	v_exp_f32_e32 v195, v5
	v_exp_f32_e32 v201, v10
	v_exp_f32_e32 v198, v11
	v_exp_f32_e32 v218, v20
	v_exp_f32_e32 v222, v23
	v_exp_f32_e32 v227, v28
	v_exp_f32_e32 v231, v31
	v_or_b32_e32 v5, v3, v4
	v_cndmask_b32_e64 v6, v6, 0, s[14:15]
	v_cndmask_b32_e64 v1, v8, 0, s[0:1]
	v_or_b32_e32 v10, 8, v0
	v_or_b32_e32 v11, 16, v3
	v_xor_b32_e32 v12, v13, v12
	v_or_b32_e32 v20, 0x808, v0
	v_xor_b32_e32 v22, v13, v22
	v_or_b32_e32 v23, 0x810, v3
	v_or_b32_e32 v28, 0x1010, v3
	v_xor_b32_e32 v29, v13, v29
	v_or_b32_e32 v31, 0x1008, v0
	v_xor_b32_e32 v13, v13, v36
	v_or_b32_e32 v36, 0x1810, v3
	v_or_b32_e32 v37, 0x1808, v0
	v_exp_f32_e32 v193, v7
	v_or_b32_e32 v7, v5, v6
	v_xor_b32_e32 v8, v1, v234
	v_xor_b32_e32 v10, v10, v2
	v_xor_b32_e32 v11, v11, v4
	v_xor_b32_e32 v20, v20, v2
	v_xor_b32_e32 v23, v23, v4
	v_xor_b32_e32 v28, v28, v4
	v_xor_b32_e32 v31, v31, v2
	v_xor_b32_e32 v36, v36, v4
	v_xor_b32_e32 v37, v37, v2
	v_exp_f32_e32 v200, v9
	v_exp_f32_e32 v220, v21
	v_exp_f32_e32 v229, v30
	v_xor_b32_e32 v9, v8, v7
	v_or3_b32 v10, v4, v10, v6
	v_or_b32_e32 v11, v11, v6
	v_or_b32_e32 v12, v12, v6
	v_or_b32_e32 v8, v8, v6
	v_or3_b32 v20, v4, v20, v6
	v_or_b32_e32 v21, 0x800, v7
	v_or_b32_e32 v22, v22, v6
	v_or_b32_e32 v23, v23, v6
	v_or_b32_e32 v28, v28, v6
	v_or_b32_e32 v29, v29, v6
	v_or_b32_e32 v30, 0x1000, v7
	v_or3_b32 v31, v4, v31, v6
	v_or_b32_e32 v13, v13, v6
	v_or_b32_e32 v36, v36, v6
	v_or3_b32 v6, v4, v37, v6
	v_or_b32_e32 v7, 0x1800, v7
	v_xor_b32_e32 v10, v234, v10
	v_xor_b32_e32 v11, v234, v11
	v_xor_b32_e32 v12, v234, v12
	v_xor_b32_e32 v20, v234, v20
	v_xor_b32_e32 v21, v234, v21
	v_xor_b32_e32 v22, v234, v22
	v_xor_b32_e32 v23, v234, v23
	v_xor_b32_e32 v28, v234, v28
	v_xor_b32_e32 v29, v234, v29
	v_xor_b32_e32 v30, v234, v30
	v_xor_b32_e32 v31, v234, v31
	v_xor_b32_e32 v13, v234, v13
	v_xor_b32_e32 v36, v234, v36
	v_xor_b32_e32 v6, v234, v6
	v_xor_b32_e32 v7, v234, v7
	v_exp_f32_e32 v213, v15
	v_exp_f32_e32 v216, v18
	v_exp_f32_e32 v223, v24
	v_exp_f32_e32 v225, v26
	v_xor_b32_e32 v10, v10, v1
	v_xor_b32_e32 v11, v11, v1
	v_xor_b32_e32 v12, v12, v1
	v_or_b32_e32 v15, 40, v0
	v_or_b32_e32 v18, 56, v0
	v_xor_b32_e32 v20, v20, v1
	v_xor_b32_e32 v21, v21, v1
	v_xor_b32_e32 v22, v22, v1
	v_xor_b32_e32 v23, v23, v1
	v_or_b32_e32 v24, 0x828, v0
	v_or_b32_e32 v26, 0x838, v0
	v_xor_b32_e32 v28, v28, v1
	v_xor_b32_e32 v29, v29, v1
	v_xor_b32_e32 v30, v30, v1
	v_xor_b32_e32 v31, v31, v1
	v_or_b32_e32 v33, 0x1038, v0
	v_or_b32_e32 v35, 0x1028, v0
	v_xor_b32_e32 v13, v13, v1
	v_xor_b32_e32 v36, v36, v1
	v_xor_b32_e32 v6, v6, v1
	v_xor_b32_e32 v1, v7, v1
	v_or_b32_e32 v7, 0x1838, v0
	v_or_b32_e32 v0, 0x1828, v0
	v_xor_b32_e32 v15, v15, v2
	v_xor_b32_e32 v24, v24, v2
	v_xor_b32_e32 v35, v35, v2
	v_xor_b32_e32 v0, v0, v2
	v_exp_f32_e32 v215, v17
	v_or_b32_e32 v15, v15, v4
	v_or_b32_e32 v17, v8, v4
	v_or_b32_e32 v24, v24, v4
	v_or_b32_e32 v35, v35, v4
	v_or_b32_e32 v0, v0, v4
	v_lshl_add_u32 v4, v9, 1, 0
	scratch_store_dword off, v4, off        ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v10, 1, 0
	v_exp_f32_e32 v212, v14
	v_or_b32_e32 v14, 32, v5
	scratch_store_dword off, v4, off offset:4 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v11, 1, 0
	v_xor_b32_e32 v14, v8, v14
	scratch_store_dword off, v4, off offset:8 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v12, 1, 0
	v_exp_f32_e32 v214, v16
	v_xor_b32_e32 v15, v8, v15
	v_or_b32_e32 v16, 48, v3
	scratch_store_dword off, v4, off offset:12 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v14, 1, 0
	v_exp_f32_e32 v217, v19
	v_xor_b32_e32 v16, v17, v16
	v_or_b32_e32 v19, v17, v2
	scratch_store_dword off, v4, off offset:16 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v15, 1, 0
	v_xor_b32_e32 v18, v19, v18
	scratch_store_dword off, v4, off offset:20 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v16, 1, 0
	scratch_store_dword off, v4, off offset:24 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v18, 1, 0
	scratch_store_dword off, v4, off offset:28 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v20, 1, 0
	scratch_store_dword off, v4, off offset:32 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v21, 1, 0
	scratch_store_dword off, v4, off offset:36 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v22, 1, 0
	v_exp_f32_e32 v224, v25
	v_xor_b32_e32 v24, v8, v24
	v_or_b32_e32 v25, 0x820, v5
	scratch_store_dword off, v4, off offset:40 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v23, 1, 0
	v_xor_b32_e32 v25, v8, v25
	scratch_store_dword off, v4, off offset:44 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v24, 1, 0
	v_exp_f32_e32 v226, v27
	v_xor_b32_e32 v26, v19, v26
	v_or_b32_e32 v27, 0x830, v3
	scratch_store_dword off, v4, off offset:48 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v25, 1, 0
	v_xor_b32_e32 v27, v17, v27
	scratch_store_dword off, v4, off offset:52 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v26, 1, 0
	s_lshl_b64 s[0:1], s[44:45], 2
	scratch_store_dword off, v4, off offset:56 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v27, 1, 0
	s_add_u32 s0, s0, s40
	scratch_store_dword off, v4, off offset:60 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v28, 1, 0
	s_addc_u32 s1, s1, s41
	scratch_store_dword off, v4, off offset:64 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v29, 1, 0
	s_add_u32 s0, s0, s38
	v_exp_f32_e32 v230, v32
	v_or_b32_e32 v32, 0x1030, v3
	scratch_store_dword off, v4, off offset:68 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v30, 1, 0
	s_addc_u32 s1, s1, s39
	v_xor_b32_e32 v32, v17, v32
	scratch_store_dword off, v4, off offset:72 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v31, 1, 0
	s_add_u32 s0, s6, s0
	v_xor_b32_e32 v33, v19, v33
	v_or_b32_e32 v34, 0x1020, v5
	scratch_store_dword off, v4, off offset:76 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v32, 1, 0
	s_addc_u32 s1, s7, s1
	v_xor_b32_e32 v34, v8, v34
	scratch_store_dword off, v4, off offset:80 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v33, 1, 0
	s_add_u32 s2, s34, s36
	v_xor_b32_e32 v35, v8, v35
	scratch_store_dword off, v4, off offset:84 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v34, 1, 0
	s_addc_u32 s3, s35, s37
	scratch_store_dword off, v4, off offset:88 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v35, 1, 0
	s_mul_i32 s7, s42, 6
	s_lshl_b64 s[2:3], s[2:3], 1
	v_exp_f32_e32 v232, v43
	v_exp_f32_e32 v233, v42
	scratch_store_dword off, v4, off offset:92 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v13, 1, 0
	s_mul_hi_i32 s6, s42, 6
	s_add_u32 s2, s7, s2
	v_or_b32_e32 v3, 0x1830, v3
	v_xor_b32_e32 v0, v8, v0
	v_or_b32_e32 v2, 0x1820, v5
	scratch_store_dword off, v4, off offset:96 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v36, 1, 0
	s_addc_u32 s3, s6, s3
	v_xor_b32_e32 v7, v19, v7
	v_xor_b32_e32 v3, v17, v3
	v_xor_b32_e32 v2, v8, v2
	scratch_store_dword off, v4, off offset:100 ; 4-byte Folded Spill
	v_lshl_add_u32 v4, v6, 1, 0
	v_lshl_add_u32 v250, v0, 1, 0
	s_add_u32 s2, s4, s2
	v_mov_b32_e32 v0, 0
	s_waitcnt vmcnt(26)
	v_lshrrev_b32_e32 v219, 16, v140
	scratch_store_dword off, v4, off offset:104 ; 4-byte Folded Spill
	v_lshl_add_u32 v247, v1, 1, 0
	v_lshl_add_u32 v248, v7, 1, 0
	v_lshl_add_u32 v249, v3, 1, 0
	v_lshl_add_u32 v251, v2, 1, 0
	s_addc_u32 s3, s5, s3
	v_mov_b32_e32 v252, 1.0
	s_movk_i32 s4, 0xffc0
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v60, v0
	v_mov_b32_e32 v61, v0
	v_mov_b32_e32 v62, v0
	v_mov_b32_e32 v63, v0
	scratch_store_dword off, v234, off offset:108 ; 4-byte Folded Spill
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[64:79], v[78:79], v[96:97], 0
	v_mov_b32_e32 v234, v252
	s_setprio 0
	v_mfma_f32_32x32x8_f16 v[64:79], v[80:81], v[98:99], v[64:79]
	v_mul_f32_e32 v15, v15, v233
	v_mul_f32_e32 v48, v48, v233
	v_mul_f32_e32 v14, v14, v233
	v_mul_f32_e32 v13, v13, v233
	v_mul_f32_e32 v12, v12, v233
	v_mul_f32_e32 v11, v11, v233
	v_mfma_f32_32x32x8_f16 v[64:79], v[82:83], v[100:101], v[64:79]
	v_mul_f32_e32 v9, v9, v233
	v_mul_f32_e32 v10, v10, v233
	v_mul_f32_e32 v8, v8, v233
	v_mul_f32_e32 v7, v7, v233
	v_mul_f32_e32 v6, v6, v233
	v_mul_f32_e32 v5, v5, v233
	v_mfma_f32_32x32x8_f16 v[64:79], v[84:85], v[102:103], v[64:79]
	v_mul_f32_e32 v3, v3, v233
	v_mul_f32_e32 v4, v4, v233
	v_mul_f32_e32 v2, v2, v233
	v_mul_f32_e32 v1, v1, v233
	v_mul_f32_e32 v0, v0, v233
	v_mul_f32_e32 v31, v31, v233
	v_mfma_f32_32x32x8_f16 v[64:79], v[86:87], v[104:105], v[64:79]
	v_mul_f32_e32 v29, v29, v233
	v_mul_f32_e32 v30, v30, v233
	v_mul_f32_e32 v28, v28, v233
	v_mul_f32_e32 v27, v27, v233
	v_mul_f32_e32 v26, v26, v233
	v_mul_f32_e32 v25, v25, v233
	v_mfma_f32_32x32x8_f16 v[64:79], v[88:89], v[106:107], v[64:79]
	v_mul_f32_e32 v23, v23, v233
	v_mul_f32_e32 v24, v24, v233
	v_mul_f32_e32 v22, v22, v233
	v_mul_f32_e32 v21, v21, v233
	v_mul_f32_e32 v20, v20, v233
	v_mul_f32_e32 v19, v19, v233
	v_mfma_f32_32x32x8_f16 v[64:79], v[90:91], v[108:109], v[64:79]
	v_mul_f32_e32 v17, v17, v233
	v_mul_f32_e32 v18, v18, v233
	v_mul_f32_e32 v16, v16, v233
	v_mul_f32_e32 v47, v47, v233
	v_mul_f32_e32 v46, v46, v233
	v_mul_f32_e32 v45, v45, v233
	v_mfma_f32_32x32x8_f16 v[64:79], v[92:93], v[110:111], v[64:79]
	v_mul_f32_e32 v43, v43, v233
	v_mul_f32_e32 v44, v44, v233
	v_mul_f32_e32 v49, v49, v233
	v_mul_f32_e32 v42, v42, v233
	v_mul_f32_e32 v41, v41, v233
	v_mul_f32_e32 v40, v40, v233
	v_mfma_f32_32x32x8_f16 v[80:95], v[144:145], v[96:97], 0
	v_mul_f32_e32 v38, v38, v233
	v_add_f32_e32 v144, v254, v255
	v_mul_f32_e32 v39, v39, v233
	v_add_f32_e32 v144, v144, v194
	v_mul_f32_e32 v37, v37, v233
	v_add_f32_e32 v144, v144, v195
	v_mfma_f32_32x32x8_f16 v[80:95], v[146:147], v[98:99], v[80:95]
	v_add_f32_e32 v144, v144, v196
	v_cvt_pkrtz_f16_f32 v147, v192, v200
	v_add_f32_e32 v144, v144, v193
	v_cvt_pkrtz_f16_f32 v146, v196, v193
	v_add_f32_e32 v144, v144, v192
	v_mul_f32_e32 v36, v36, v233
	v_mfma_f32_32x32x8_f16 v[80:95], v[148:149], v[100:101], v[80:95]
	v_mul_f32_e32 v35, v35, v233
	v_cvt_pkrtz_f16_f32 v149, v199, v211
	v_cvt_pkrtz_f16_f32 v148, v201, v198
	v_add_f32_e32 v144, v144, v200
	v_mul_f32_e32 v34, v34, v233
	v_add_f32_e32 v144, v144, v201
	v_mfma_f32_32x32x8_f16 v[80:95], v[150:151], v[102:103], v[80:95]
	v_add_f32_e32 v144, v144, v198
	v_cvt_pkrtz_f16_f32 v151, v214, v215
	v_add_f32_e32 v144, v144, v199
	v_cvt_pkrtz_f16_f32 v150, v212, v213
	v_add_f32_e32 v144, v144, v211
	v_mul_f32_e32 v33, v33, v233
	v_mfma_f32_32x32x8_f16 v[80:95], v[152:153], v[104:105], v[80:95]
	v_mul_f32_e32 v32, v32, v233
	v_cvt_pkrtz_f16_f32 v153, v218, v220
	v_cvt_pkrtz_f16_f32 v152, v216, v217
	v_add_f32_e32 v144, v144, v212
	v_mul_f32_e32 v63, v63, v233
	v_add_f32_e32 v144, v144, v213
	v_mfma_f32_32x32x8_f16 v[80:95], v[154:155], v[106:107], v[80:95]
	v_add_f32_e32 v144, v144, v214
	v_cvt_pkrtz_f16_f32 v155, v223, v224
	v_add_f32_e32 v144, v144, v215
	v_cvt_pkrtz_f16_f32 v154, v221, v222
	v_add_f32_e32 v144, v144, v216
	v_mul_f32_e32 v62, v62, v233
	v_mfma_f32_32x32x8_f16 v[80:95], v[156:157], v[108:109], v[80:95]
	v_mul_f32_e32 v61, v61, v233
	v_cvt_pkrtz_f16_f32 v157, v227, v228
	v_cvt_pkrtz_f16_f32 v156, v225, v226
	v_add_f32_e32 v144, v144, v217
	v_mul_f32_e32 v60, v60, v233
	v_add_f32_e32 v144, v144, v218
	v_mfma_f32_32x32x8_f16 v[80:95], v[158:159], v[110:111], v[80:95]
	v_add_f32_e32 v144, v144, v220
	v_cvt_pkrtz_f16_f32 v159, v230, v232
	v_add_f32_e32 v144, v144, v221
	v_cvt_pkrtz_f16_f32 v158, v229, v231
	v_add_f32_e32 v144, v144, v222
	v_mul_f32_e32 v59, v59, v233
	v_mfma_f32_32x32x8_f16 v[80:95], v[160:161], v[112:113], v[80:95]
	v_mul_f32_e32 v58, v58, v233
	v_add_f32_e32 v144, v144, v223
	v_mul_f32_e32 v57, v57, v233
	v_add_f32_e32 v144, v144, v224
	v_mul_f32_e32 v56, v56, v233
	v_add_f32_e32 v144, v144, v225
	v_mfma_f32_32x32x8_f16 v[80:95], v[162:163], v[114:115], v[80:95]
	v_add_f32_e32 v144, v144, v226
	v_mul_f32_e32 v55, v55, v233
	v_add_f32_e32 v144, v144, v227
	v_mul_f32_e32 v54, v54, v233
	v_add_f32_e32 v144, v144, v228
	v_mul_f32_e32 v53, v53, v233
	v_mfma_f32_32x32x8_f16 v[80:95], v[164:165], v[116:117], v[80:95]
	v_mul_f32_e32 v52, v52, v233
	v_add_f32_e32 v144, v144, v229
	v_mul_f32_e32 v51, v51, v233
	v_add_f32_e32 v144, v144, v231
	v_mul_f32_e32 v50, v50, v233
	v_add_f32_e32 v144, v144, v230
	v_mfma_f32_32x32x8_f16 v[80:95], v[166:167], v[118:119], v[80:95]
	v_add_f32_e32 v144, v144, v232
	ds_bpermute_b32 v145, v210, v144
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v252, v144, v145
	v_fmac_f32_e32 v252, v234, v233
	v_cvt_pkrtz_f16_f32 v145, v194, v195
	v_cvt_pkrtz_f16_f32 v144, v254, v255
	v_mfma_f32_32x32x8_f16 v[80:95], v[168:169], v[120:121], v[80:95]
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[64:79], v[176:177], v[112:113], v[64:79]
	v_mfma_f32_32x32x8_f16 v[64:79], v[178:179], v[114:115], v[64:79]
	v_mfma_f32_32x32x8_f16 v[64:79], v[180:181], v[116:117], v[64:79]
	v_mfma_f32_32x32x8_f16 v[64:79], v[182:183], v[118:119], v[64:79]
	v_mfma_f32_32x32x8_f16 v[64:79], v[184:185], v[120:121], v[64:79]
	v_mfma_f32_32x32x8_f16 v[64:79], v[186:187], v[122:123], v[64:79]
	v_mfma_f32_32x32x8_f16 v[80:95], v[170:171], v[122:123], v[80:95]
	v_mfma_f32_32x32x8_f16 v[64:79], v[188:189], v[124:125], v[64:79]
	v_mfma_f32_32x32x8_f16 v[80:95], v[172:173], v[124:125], v[80:95]
	v_mfma_f32_32x32x8_f16 v[64:79], v[190:191], v[126:127], v[64:79]
	v_mfma_f32_32x32x8_f16 v[80:95], v[174:175], v[126:127], v[80:95]
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_barrier
	s_barrier
	scratch_load_dword v196, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v162, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v164, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v176, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v178, off, off offset:36 ; 4-byte Folded Reload
	scratch_load_dword v182, off, off offset:44 ; 4-byte Folded Reload
	scratch_load_dword v192, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v160, off, off       ; 4-byte Folded Reload
	scratch_load_dword v166, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v168, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v170, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v172, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v174, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v180, off, off offset:40 ; 4-byte Folded Reload
	scratch_load_dword v184, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v186, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v188, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v190, off, off offset:60 ; 4-byte Folded Reload
	scratch_load_dword v194, off, off offset:68 ; 4-byte Folded Reload
	s_and_b32 s5, s3, 0xffff
	s_or_b32 s21, s5, s19
	s_mov_b32 s20, s2
	ds_read_b64 v[228:229], v247 offset:16384
	ds_read_b64 v[230:231], v248 offset:16384
	ds_read_b64 v[232:233], v249 offset:16384
	ds_read_b64 v[254:255], v250 offset:16384
	s_waitcnt vmcnt(18)
	ds_read_b64 v[198:199], v196 offset:16384
	scratch_load_dword v196, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(18)
	ds_read_b64 v[162:163], v162 offset:16384
	s_waitcnt vmcnt(17)
	ds_read_b64 v[164:165], v164 offset:16384
	s_waitcnt vmcnt(16)
	ds_read_b64 v[176:177], v176 offset:16384
	s_waitcnt vmcnt(15)
	ds_read_b64 v[178:179], v178 offset:16384
	s_waitcnt vmcnt(14)
	ds_read_b64 v[182:183], v182 offset:16384
	s_waitcnt vmcnt(13)
	ds_read_b64 v[192:193], v192 offset:16384
	s_waitcnt vmcnt(12)
	ds_read_b64 v[160:161], v160 offset:16384
	s_waitcnt vmcnt(11)
	ds_read_b64 v[166:167], v166 offset:16384
	s_waitcnt vmcnt(10)
	ds_read_b64 v[168:169], v168 offset:16384
	s_waitcnt vmcnt(9)
	ds_read_b64 v[170:171], v170 offset:16384
	s_waitcnt vmcnt(8)
	ds_read_b64 v[172:173], v172 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[174:175], v174 offset:16384
	s_waitcnt vmcnt(6)
	ds_read_b64 v[180:181], v180 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[184:185], v184 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[186:187], v186 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[188:189], v188 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[190:191], v190 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[194:195], v194 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[200:201], v196 offset:16384
	scratch_load_dword v196, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v196 offset:16384
	scratch_load_dword v196, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v196 offset:16384
	scratch_load_dword v196, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[216:217], v196 offset:16384
	scratch_load_dword v196, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[220:221], v196 offset:16384
	scratch_load_dword v196, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[222:223], v196 offset:16384
	scratch_load_dword v196, off, off offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[224:225], v196 offset:16384
	scratch_load_dword v196, off, off offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[226:227], v196 offset:16384
	ds_write_b128 v197, v[132:135]
	ds_write_b128 v197, v[128:131] offset:8192
	buffer_load_dwordx4 v[132:135], v236, s[20:23], 0 offen
	buffer_load_dwordx4 v[128:131], v237, s[20:23], 0 offen
	v_mov_b32_e32 v196, v235
	ds_read_b64 v[234:235], v251 offset:16384
	; sched_barrier mask(0x00000000)
	s_barrier
	s_setprio 0
	; iglp_opt mask(0x0000000A)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[48:63], v[160:161], v[144:145], v[48:63]
	v_mfma_f32_32x32x8_f16 v[32:47], v[176:177], v[144:145], v[32:47]
	v_mfma_f32_32x32x8_f16 v[16:31], v[192:193], v[144:145], v[16:31]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[0:15], v[222:223], v[144:145], v[0:15]
	v_max_f32_e32 v144, v65, v65
	v_max_f32_e32 v145, v64, v64
	v_max_f32_e32 v144, v145, v144
	v_max3_f32 v144, v144, v66, v67
	v_max3_f32 v144, v144, v68, v69
	v_max3_f32 v144, v144, v70, v71
	v_mfma_f32_32x32x8_f16 v[16:31], v[194:195], v[146:147], v[16:31]
	v_max3_f32 v144, v144, v72, v73
	v_max3_f32 v144, v144, v74, v75
	v_max3_f32 v144, v144, v76, v77
	v_max3_f32 v144, v144, v78, v79
	v_max3_f32 v144, v144, v80, v81
	v_max3_f32 v144, v144, v82, v83
	v_mfma_f32_32x32x8_f16 v[16:31], v[198:199], v[148:149], v[16:31]
	v_max3_f32 v144, v144, v84, v85
	v_max3_f32 v144, v144, v86, v87
	v_max3_f32 v144, v144, v88, v89
	v_max3_f32 v144, v144, v90, v91
	v_max3_f32 v144, v144, v92, v93
	v_max3_f32 v144, v144, v94, v95
	v_mfma_f32_32x32x8_f16 v[16:31], v[200:201], v[150:151], v[16:31]
	ds_bpermute_b32 v145, v210, v144
	v_mfma_f32_32x32x8_f16 v[48:63], v[162:163], v[146:147], v[48:63]
	v_mfma_f32_32x32x8_f16 v[32:47], v[178:179], v[146:147], v[32:47]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[0:15], v[224:225], v[146:147], v[0:15]
	v_mfma_f32_32x32x8_f16 v[48:63], v[164:165], v[148:149], v[48:63]
	v_mfma_f32_32x32x8_f16 v[32:47], v[180:181], v[148:149], v[32:47]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[0:15], v[226:227], v[148:149], v[0:15]
	v_mfma_f32_32x32x8_f16 v[48:63], v[166:167], v[150:151], v[48:63]
	v_mfma_f32_32x32x8_f16 v[32:47], v[182:183], v[150:151], v[32:47]
	v_mfma_f32_32x32x8_f16 v[0:15], v[228:229], v[150:151], v[0:15]
	v_mfma_f32_32x32x8_f16 v[48:63], v[168:169], v[152:153], v[48:63]
	v_mfma_f32_32x32x8_f16 v[32:47], v[184:185], v[152:153], v[32:47]
	v_mfma_f32_32x32x8_f16 v[16:31], v[212:213], v[152:153], v[16:31]
	v_mfma_f32_32x32x8_f16 v[0:15], v[230:231], v[152:153], v[0:15]
	v_mfma_f32_32x32x8_f16 v[48:63], v[170:171], v[154:155], v[48:63]
	v_mfma_f32_32x32x8_f16 v[32:47], v[186:187], v[154:155], v[32:47]
	v_mfma_f32_32x32x8_f16 v[16:31], v[214:215], v[154:155], v[16:31]
	v_mfma_f32_32x32x8_f16 v[0:15], v[232:233], v[154:155], v[0:15]
	v_mfma_f32_32x32x8_f16 v[48:63], v[172:173], v[156:157], v[48:63]
	v_mfma_f32_32x32x8_f16 v[32:47], v[188:189], v[156:157], v[32:47]
	v_mfma_f32_32x32x8_f16 v[16:31], v[216:217], v[156:157], v[16:31]
	v_mfma_f32_32x32x8_f16 v[0:15], v[254:255], v[156:157], v[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[0:15], v[234:235], v[158:159], v[0:15]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v234, v253, v144, v145
	v_mul_f32_e32 v144, 0x3e0293ee, v234
	v_fma_f32 v95, v95, s43, -v144
	v_fma_f32 v64, v64, s43, -v144
	v_fma_f32 v94, v94, s43, -v144
	v_exp_f32_e32 v254, v64
	v_mfma_f32_32x32x8_f16 v[16:31], v[220:221], v[158:159], v[16:31]
	v_fma_f32 v64, v253, s43, -v144
	v_fma_f32 v65, v65, s43, -v144
	v_fma_f32 v93, v93, s43, -v144
	v_fma_f32 v92, v92, s43, -v144
	v_fma_f32 v91, v91, s43, -v144
	v_fma_f32 v90, v90, s43, -v144
	v_mfma_f32_32x32x8_f16 v[48:63], v[174:175], v[158:159], v[48:63]
	v_fma_f32 v89, v89, s43, -v144
	v_fma_f32 v66, v66, s43, -v144
	v_fma_f32 v88, v88, s43, -v144
	v_fma_f32 v87, v87, s43, -v144
	v_fma_f32 v86, v86, s43, -v144
	v_fma_f32 v85, v85, s43, -v144
	v_mfma_f32_32x32x8_f16 v[32:47], v[190:191], v[158:159], v[32:47]
	v_fma_f32 v84, v84, s43, -v144
	v_fma_f32 v67, v67, s43, -v144
	v_fma_f32 v83, v83, s43, -v144
	v_fma_f32 v82, v82, s43, -v144
	v_fma_f32 v81, v81, s43, -v144
	v_fma_f32 v80, v80, s43, -v144
	v_fma_f32 v68, v68, s43, -v144
	v_fma_f32 v69, v69, s43, -v144
	v_fma_f32 v70, v70, s43, -v144
	v_fma_f32 v71, v71, s43, -v144
	v_fma_f32 v72, v72, s43, -v144
	v_fma_f32 v73, v73, s43, -v144
	v_fma_f32 v74, v74, s43, -v144
	v_fma_f32 v75, v75, s43, -v144
	v_fma_f32 v76, v76, s43, -v144
	v_fma_f32 v77, v77, s43, -v144
	v_fma_f32 v78, v78, s43, -v144
	v_fma_f32 v79, v79, s43, -v144
	v_mov_b32_e32 v235, v196
	v_exp_f32_e32 v255, v65
	v_exp_f32_e32 v194, v66
	v_exp_f32_e32 v195, v67
	v_exp_f32_e32 v196, v68
	v_exp_f32_e32 v193, v69
	v_exp_f32_e32 v192, v70
	v_exp_f32_e32 v200, v71
	v_exp_f32_e32 v201, v72
	v_exp_f32_e32 v198, v73
	v_exp_f32_e32 v199, v74
	v_exp_f32_e32 v211, v75
	v_exp_f32_e32 v212, v76
	v_exp_f32_e32 v213, v77
	v_exp_f32_e32 v214, v78
	v_exp_f32_e32 v215, v79
	v_exp_f32_e32 v216, v80
	v_exp_f32_e32 v217, v81
	v_exp_f32_e32 v218, v82
	v_exp_f32_e32 v220, v83
	v_exp_f32_e32 v221, v84
	v_exp_f32_e32 v222, v85
	v_exp_f32_e32 v223, v86
	v_exp_f32_e32 v224, v87
	v_exp_f32_e32 v225, v88
	v_exp_f32_e32 v226, v89
	v_exp_f32_e32 v227, v90
	v_exp_f32_e32 v228, v91
	v_exp_f32_e32 v229, v92
	v_exp_f32_e32 v231, v93
	v_exp_f32_e32 v230, v94
	v_exp_f32_e32 v232, v95
	v_exp_f32_e32 v233, v64
	s_setprio 1
	; sched_barrier mask(0x00000000)
	v_perm_b32 v64, v140, v136, s27
	s_barrier
	ds_write_b32 v239, v64 offset:16384
	v_alignbit_b32 v64, v219, v136, 16
	ds_read_b128 v[78:81], v202
	ds_read_b128 v[82:85], v203
	ds_read_b128 v[86:89], v204
	ds_read_b128 v[90:93], v205
	ds_read_b128 v[176:179], v206
	ds_read_b128 v[180:183], v207
	ds_read_b128 v[184:187], v208
	ds_read_b128 v[188:191], v209
	ds_read_b128 v[144:147], v202 offset:8192
	ds_read_b128 v[148:151], v203 offset:8192
	ds_read_b128 v[152:155], v204 offset:8192
	ds_read_b128 v[156:159], v205 offset:8192
	ds_read_b128 v[160:163], v206 offset:8192
	ds_read_b128 v[164:167], v207 offset:8192
	ds_read_b128 v[168:171], v208 offset:8192
	ds_read_b128 v[172:175], v209 offset:8192
	ds_write_b32 v240, v64 offset:16384
	v_perm_b32 v64, v141, v137, s27
	ds_write_b32 v241, v64 offset:16384
	v_perm_b32 v64, v141, v137, s46
	ds_write_b32 v242, v64 offset:16384
	v_perm_b32 v64, v142, v138, s27
	ds_write_b32 v243, v64 offset:16384
	v_perm_b32 v64, v142, v138, s46
	s_and_b32 s5, s1, 0xffff
	ds_write_b32 v244, v64 offset:16384
	v_perm_b32 v64, v143, v139, s27
	s_or_b32 s21, s5, s33
	s_mov_b32 s20, s0
	ds_write_b32 v245, v64 offset:16384
	v_perm_b32 v64, v143, v139, s46
	buffer_load_dwordx4 v[140:143], v238, s[20:23], 0 offen
	buffer_load_dwordx4 v[136:139], v235, s[20:23], 0 offen
	ds_write_b32 v246, v64 offset:16384
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v219, 16, v140
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s0, s28
	s_addc_u32 s1, s1, s29
	s_add_u32 s2, s2, s30
	s_addc_u32 s3, s3, s31
	s_add_i32 s4, s4, 64
	s_cmpk_lt_u32 s4, 0x1f00
	v_mov_b32_e32 v253, v234
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:                                ; %.critedge
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dword v64, off, off offset:124 ; 4-byte Folded Reload
	scratch_load_dword v65, off, off offset:112 ; 4-byte Folded Reload
	s_add_i32 s0, s16, 0xffffc100
	s_cmp_lt_i32 s0, 1
	s_movk_i32 s2, 0x4000
	s_cselect_b64 s[0:1], -1, 0
	v_mul_f32_e32 v48, v48, v252
	v_mul_f32_e32 v49, v49, v252
	v_mul_f32_e32 v50, v50, v252
	v_mul_f32_e32 v51, v51, v252
	v_mul_f32_e32 v52, v52, v252
	v_mul_f32_e32 v53, v53, v252
	v_mul_f32_e32 v54, v54, v252
	v_mul_f32_e32 v55, v55, v252
	v_mul_f32_e32 v56, v56, v252
	v_mul_f32_e32 v57, v57, v252
	v_mul_f32_e32 v58, v58, v252
	v_mul_f32_e32 v59, v59, v252
	v_mul_f32_e32 v60, v60, v252
	v_mul_f32_e32 v61, v61, v252
	v_mul_f32_e32 v62, v62, v252
	v_mul_f32_e32 v63, v63, v252
	v_mul_f32_e32 v32, v32, v252
	v_mul_f32_e32 v33, v33, v252
	v_mul_f32_e32 v34, v34, v252
	v_mul_f32_e32 v35, v35, v252
	v_mul_f32_e32 v36, v36, v252
	v_mul_f32_e32 v37, v37, v252
	v_mul_f32_e32 v38, v38, v252
	v_mul_f32_e32 v39, v39, v252
	v_mul_f32_e32 v40, v40, v252
	v_mul_f32_e32 v41, v41, v252
	v_mul_f32_e32 v42, v42, v252
	v_mul_f32_e32 v43, v43, v252
	v_mul_f32_e32 v44, v44, v252
	v_mul_f32_e32 v45, v45, v252
	v_mul_f32_e32 v46, v46, v252
	v_mul_f32_e32 v47, v47, v252
	v_mul_f32_e32 v16, v16, v252
	v_mul_f32_e32 v17, v17, v252
	v_mul_f32_e32 v18, v18, v252
	v_mul_f32_e32 v19, v19, v252
	v_mul_f32_e32 v20, v20, v252
	v_mul_f32_e32 v21, v21, v252
	v_mul_f32_e32 v22, v22, v252
	v_mul_f32_e32 v23, v23, v252
	v_mul_f32_e32 v24, v24, v252
	v_mul_f32_e32 v25, v25, v252
	v_mul_f32_e32 v26, v26, v252
	v_mul_f32_e32 v27, v27, v252
	v_mul_f32_e32 v28, v28, v252
	v_mul_f32_e32 v29, v29, v252
	v_mul_f32_e32 v30, v30, v252
	v_mul_f32_e32 v31, v31, v252
	v_mul_f32_e32 v67, v2, v252
	v_mul_f32_e32 v68, v3, v252
	v_mul_f32_e32 v69, v4, v252
	v_mul_f32_e32 v70, v5, v252
	v_mul_f32_e32 v71, v6, v252
	v_mul_f32_e32 v72, v7, v252
	v_mul_f32_e32 v73, v8, v252
	v_mul_f32_e32 v74, v9, v252
	v_mul_f32_e32 v75, v10, v252
	v_mul_f32_e32 v76, v11, v252
	v_mul_f32_e32 v77, v12, v252
	s_waitcnt lgkmcnt(14)
	v_mul_f32_e32 v78, v13, v252
	v_mul_f32_e32 v79, v14, v252
	v_mul_f32_e32 v80, v15, v252
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v64, 1, v64
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v64, v64, v65
	scratch_load_dword v65, off, off offset:116 ; 4-byte Folded Reload
	scratch_load_dword v66, off, off offset:120 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	v_or3_b32 v64, v64, v65, v66
	v_or_b32_e32 v65, s16, v64
	v_cmp_gt_i32_e32 vcc, s2, v65
	s_or_b64 vcc, s[0:1], vcc
	s_mul_i32 s0, s24, s18
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s25, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s26, s16
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	v_mul_lo_u32 v64, s26, v64
	s_addc_u32 s1, s3, s1
	v_mul_f32_e32 v65, v0, v252
	v_mul_f32_e32 v66, v1, v252
	; sched_barrier mask(0x00000000)
	v_cvt_pkrtz_f16_f32 v0, v48, v49
	v_cvt_pkrtz_f16_f32 v1, v50, v51
	v_cvt_pkrtz_f16_f32 v2, v52, v53
	v_cvt_pkrtz_f16_f32 v3, v54, v55
	v_cvt_pkrtz_f16_f32 v4, v56, v57
	v_cvt_pkrtz_f16_f32 v5, v58, v59
	v_cvt_pkrtz_f16_f32 v6, v60, v61
	v_cvt_pkrtz_f16_f32 v7, v62, v63
	v_cvt_pkrtz_f16_f32 v8, v32, v33
	v_cvt_pkrtz_f16_f32 v9, v34, v35
	v_cvt_pkrtz_f16_f32 v10, v36, v37
	v_cvt_pkrtz_f16_f32 v11, v38, v39
	v_cvt_pkrtz_f16_f32 v12, v40, v41
	v_cvt_pkrtz_f16_f32 v13, v42, v43
	v_cvt_pkrtz_f16_f32 v14, v44, v45
	v_cvt_pkrtz_f16_f32 v15, v46, v47
	v_cvt_pkrtz_f16_f32 v16, v16, v17
	v_cvt_pkrtz_f16_f32 v17, v18, v19
	v_cvt_pkrtz_f16_f32 v18, v20, v21
	v_cvt_pkrtz_f16_f32 v19, v22, v23
	v_cvt_pkrtz_f16_f32 v20, v24, v25
	v_cvt_pkrtz_f16_f32 v21, v26, v27
	v_cvt_pkrtz_f16_f32 v22, v28, v29
	v_cvt_pkrtz_f16_f32 v23, v30, v31
	v_cvt_pkrtz_f16_f32 v24, v65, v66
	v_cvt_pkrtz_f16_f32 v25, v67, v68
	v_cvt_pkrtz_f16_f32 v26, v69, v70
	v_cvt_pkrtz_f16_f32 v27, v71, v72
	v_cvt_pkrtz_f16_f32 v28, v73, v74
	v_cvt_pkrtz_f16_f32 v29, v75, v76
	v_cvt_pkrtz_f16_f32 v30, v77, v78
	v_cvt_pkrtz_f16_f32 v31, v79, v80
	; sched_barrier mask(0x00000000)
	scratch_load_dword v32, off, off offset:108 ; 4-byte Folded Reload
	s_and_b32 s2, s26, 0x3fff
	s_bitset1_b32 s2, 14
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_bfrev_b32_e32 v33, 1
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v32, v64, v32, 1
	v_cndmask_b32_e32 v34, v33, v32, vcc
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_add_u32_e32 v0, 16, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[2:3], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 32, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[4:5], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 48, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[6:7], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 64, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[8:9], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x50, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[10:11], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x60, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[12:13], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x70, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[14:15], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x80, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[16:17], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x90, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[18:19], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xa0, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[20:21], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xb0, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[22:23], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xc0, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[24:25], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xd0, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[26:27], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xe0, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[28:29], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xf0, v32
	v_cndmask_b32_e32 v0, v33, v0, vcc
	buffer_store_dwordx2 v[30:31], v0, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 132
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
		.amdhsa_next_free_sgpr 48
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
	.set attn_fwd.numbered_sgpr, 48
	.set attn_fwd.private_seg_size, 132
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8616
; TotalNumSgprs: 54
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 132
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 54
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
    .private_segment_fixed_size: 132
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 32
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
