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
; %bb.7:
	.file	1 "/app/OAI-triton/fa" "flash-attention.py"
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
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s14, s34
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_lshrrev_b32_e32 v1, 4, v0
	s_lshl_b32 s16, s14, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v115, 3, v0
	v_or_b32_e32 v59, 32, v1
	v_or_b32_e32 v2, 0xa0, v1
	v_or_b32_e32 v3, 0xe0, v1
	s_add_u32 s0, s2, s0
	v_and_b32_e32 v58, 0x78, v115
	s_mul_i32 s52, s15, s18
	v_or_b32_e32 v19, s34, v2
	v_or_b32_e32 v27, s34, v3
	v_mul_lo_u32 v20, s14, v2
	v_mul_lo_u32 v28, s14, v3
	s_addc_u32 s1, s3, s1
	v_mad_u64_u32 v[2:3], s[2:3], s14, v1, v[58:59]
	s_ashr_i32 s53, s52, 31
	s_lshl_b64 s[2:3], s[52:53], 1
	s_add_u32 s12, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s54, s20, s17
	s_addc_u32 s13, s5, s3
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[2:3], s[54:55], 1
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
	v_or_b32_e32 v4, s34, v1
	s_add_u32 s12, s6, s2
	s_movk_i32 s6, 0x4000
	v_or_b32_e32 v5, s34, v59
	v_mul_lo_u32 v6, s14, v59
	v_add_u32_e32 v13, s16, v2
	s_addc_u32 s33, s7, s3
	s_and_b32 s2, s14, 0x3fff
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v30, 1
	v_cmp_gt_i32_e32 vcc, s6, v4
	v_or_b32_e32 v7, 0x60, v1
	v_or_b32_e32 v10, 64, v4
	s_bitset1_b32 s2, 14
	v_cndmask_b32_e32 v14, v30, v2, vcc
	v_add_lshl_u32 v2, v6, v58, 1
	v_cmp_gt_i32_e32 vcc, s6, v5
	v_or_b32_e32 v11, s34, v7
	v_mul_lo_u32 v12, s14, v7
	v_add_u32_e32 v29, s16, v13
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_cndmask_b32_e32 v15, v30, v2, vcc
	v_lshlrev_b32_e32 v13, 1, v13
	v_cmp_gt_i32_e32 vcc, s6, v10
	v_or_b32_e32 v18, 0x80, v4
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v21, v30, v13, vcc
	v_add_lshl_u32 v10, v12, v58, 1
	v_cmp_gt_i32_e32 vcc, s6, v11
	v_or_b32_e32 v26, 0xc0, v4
	buffer_load_dwordx4 v[2:5], v14, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v15, s[0:3], 0 offen
	v_cndmask_b32_e32 v22, v30, v10, vcc
	buffer_load_dwordx4 v[10:13], v21, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v22, s[0:3], 0 offen
	v_lshlrev_b32_e32 v21, 1, v29
	v_cmp_gt_i32_e32 vcc, s6, v18
	v_add_lshl_u32 v18, v20, v58, 1
	v_add_lshl_u32 v29, v29, s16, 1
	v_cndmask_b32_e32 v31, v30, v21, vcc
	v_cmp_gt_i32_e32 vcc, s6, v19
	v_lshrrev_b32_e32 v122, 1, v0
	s_movk_i32 s23, 0x78
	v_cndmask_b32_e32 v32, v30, v18, vcc
	v_cmp_gt_i32_e32 vcc, s6, v26
	v_add_lshl_u32 v26, v28, v58, 1
	buffer_load_dwordx4 v[18:21], v31, s[0:3], 0 offen
	buffer_load_dwordx4 v[22:25], v32, s[0:3], 0 offen
	v_cndmask_b32_e32 v34, v30, v29, vcc
	v_cmp_gt_i32_e32 vcc, s6, v27
	v_bitop3_b32 v36, v122, v115, s23 bitop3:0x28
	v_lshlrev_b32_e32 v36, 1, v36
	v_cndmask_b32_e32 v35, v30, v26, vcc
	buffer_load_dwordx4 v[26:29], v34, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v35, s[0:3], 0 offen
	s_and_b32 s0, s21, 0x3fff
	s_bitset1_b32 s0, 14
	v_mul_lo_u32 v34, s21, v1
	v_lshlrev_b32_e32 v92, 8, v1
	s_and_b32 s1, s13, 0xffff
	s_lshl_b32 s19, s0, 16
	v_mul_lo_u32 v35, s21, v59
	v_add3_u32 v113, 0, v36, v92
	s_or_b32 s29, s1, s19
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	v_add_lshl_u32 v114, v34, v58, 1
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v113, v[2:5]
	s_waitcnt vmcnt(6)
	ds_write_b128 v113, v[6:9] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v113, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v113, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v113, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v113, v[22:25] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v113, v[26:29] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v113, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_lshl_u32 v116, v35, v58, 1
	buffer_load_dwordx4 v[2:5], v114, s[28:31], 0 offen
	buffer_load_dwordx4 v[6:9], v116, s[28:31], 0 offen
	v_and_b32_e32 v21, 31, v0
	s_movk_i32 s0, 0xe0
	v_lshrrev_b32_e32 v10, 5, v0
	v_and_b32_e32 v13, 15, v0
	s_lshl_b32 s30, s21, 6
	v_bfe_u32 v11, v0, 5, 1
	v_and_or_b32 v12, v122, s0, v21
	v_bitop3_b32 v10, v10, v13, 1 bitop3:0x6c
	v_bitop3_b32 v14, v11, v13, 2 bitop3:0x36
	v_bitop3_b32 v15, v11, v13, 4 bitop3:0x36
	v_lshl_add_u32 v12, v12, 8, 0
	v_lshlrev_b32_e32 v10, 4, v10
	s_ashr_i32 s31, s30, 31
	s_lshl_b32 s22, s24, 6
	v_bitop3_b32 v16, v11, v13, 6 bitop3:0x36
	v_bitop3_b32 v17, v11, v13, 8 bitop3:0x36
	v_bitop3_b32 v18, v11, v13, 10 bitop3:0x36
	v_bitop3_b32 v19, v11, v13, 12 bitop3:0x36
	v_bitop3_b32 v11, v11, v13, 14 bitop3:0x36
	v_add_u32_e32 v13, v12, v10
	v_lshlrev_b32_e32 v14, 4, v14
	v_lshlrev_b32_e32 v22, 4, v15
	s_lshl_b64 s[6:7], s[30:31], 1
	v_add_u32_e32 v20, v12, v14
	ds_read_b128 v[126:129], v13
	ds_read_b128 v[94:97], v20
	v_add_u32_e32 v13, v12, v22
	v_lshlrev_b32_e32 v23, 4, v16
	v_lshlrev_b32_e32 v24, 4, v17
	s_add_u32 s0, s28, s6
	v_add_u32_e32 v15, v12, v23
	ds_read_b128 v[98:101], v13
	ds_read_b128 v[102:105], v15
	v_add_u32_e32 v13, v12, v24
	v_lshlrev_b32_e32 v25, 4, v18
	v_lshlrev_b32_e32 v26, 4, v19
	s_addc_u32 s28, s13, s7
	v_add_u32_e32 v15, v12, v25
	ds_read_b128 v[130:133], v13
	ds_read_b128 v[134:137], v15
	v_add_u32_e32 v13, v12, v26
	v_lshlrev_b32_e32 v27, 4, v11
	s_and_b32 s1, s28, 0xffff
	v_lshlrev_b32_e32 v28, 8, v21
	v_add_u32_e32 v11, v12, v27
	ds_read_b128 v[138:141], v13
	ds_read_b128 v[142:145], v11
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b128 v113, v[2:5]
	s_waitcnt vmcnt(0)
	ds_write_b128 v113, v[6:9] offset:8192
	s_or_b32 s1, s1, s19
	v_add3_u32 v93, 0, v10, v28
	buffer_load_dwordx4 v[50:53], v114, s[0:3], 0 offen
	buffer_load_dwordx4 v[54:57], v116, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[18:21], v93
	ds_read_b128 v[60:63], v93 offset:8192
	s_mov_b32 s36, 0
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
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	v_mov_b64_e32 v[34:35], s[36:37]
	v_mov_b64_e32 v[36:37], s[38:39]
	v_mov_b64_e32 v[38:39], s[40:41]
	v_mov_b64_e32 v[40:41], s[42:43]
	v_mov_b64_e32 v[42:43], s[44:45]
	v_mov_b64_e32 v[44:45], s[46:47]
	v_mov_b64_e32 v[46:47], s[48:49]
	v_mov_b64_e32 v[48:49], s[50:51]
	v_add3_u32 v106, 0, v14, v28
	v_add3_u32 v107, 0, v22, v28
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[126:129], v[34:49]
	ds_read_b128 v[18:21], v106
	ds_read_b128 v[64:67], v106 offset:8192
	v_add3_u32 v108, 0, v23, v28
	v_add3_u32 v109, 0, v24, v28
	v_add3_u32 v110, 0, v25, v28
	v_add3_u32 v111, 0, v26, v28
	v_add3_u32 v112, 0, v27, v28
	s_and_b32 s1, s24, 0x3fff
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[94:97], v[2:17]
	ds_read_b128 v[18:21], v107
	ds_read_b128 v[68:71], v107 offset:8192
	s_bitset1_b32 s1, 14
	v_mul_lo_u32 v1, s24, v1
	s_and_b32 s13, s33, 0xffff
	s_lshl_b32 s20, s1, 16
	s_or_b32 s13, s13, s20
	s_mov_b32 s14, s2
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[98:101], v[2:17]
	ds_read_b128 v[18:21], v108
	ds_read_b128 v[72:75], v108 offset:8192
	s_mov_b32 s15, s3
	s_mov_b32 s1, 0xff800000
	s_mov_b32 s16, 0x3e0293ee
	s_add_u32 s0, s0, s6
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[102:105], v[2:17]
	ds_read_b128 v[18:21], v109
	ds_read_b128 v[76:79], v109 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[130:133], v[2:17]
	ds_read_b128 v[18:21], v110
	ds_read_b128 v[80:83], v110 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[134:137], v[2:17]
	ds_read_b128 v[18:21], v111
	ds_read_b128 v[84:87], v111 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[138:141], v[2:17]
	ds_read_b128 v[18:21], v112
	ds_read_b128 v[88:91], v112 offset:8192
	scratch_store_dwordx4 off, v[94:97], off ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[98:101], off offset:16 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[102:105], off offset:32 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[142:145], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[60:63], v[126:129], v[34:49]
	s_nop 6
	v_mul_lo_u32 v34, s24, v59
	v_add_lshl_u32 v44, v1, v58, 1
	v_add_lshl_u32 v45, v34, v58, 1
	buffer_load_dwordx4 v[34:37], v44, s[12:15], 0 offen
	buffer_load_dwordx4 v[38:41], v45, s[12:15], 0 offen
	v_mfma_f32_32x32x16_f16 v[18:33], v[64:67], v[94:97], v[18:33]
	.file	2 "/app/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v42, v3, v3
	v_max_f32_e32 v43, v2, v2
	v_max_f32_e32 v1, v43, v42
	v_max3_f32 v1, v1, v4, v5
	v_max3_f32 v1, v1, v6, v7
	v_max3_f32 v1, v1, v8, v9
	v_max3_f32 v1, v1, v10, v11
	v_mfma_f32_32x32x16_f16 v[18:33], v[68:71], v[98:101], v[18:33]
	v_max3_f32 v1, v1, v12, v13
	v_max3_f32 v1, v1, v14, v15
	v_max3_f32 v1, v1, v16, v17
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(6)
	ds_write_b128 v113, v[50:53]
	v_mfma_f32_32x32x16_f16 v[18:33], v[72:75], v[102:105], v[18:33]
	scratch_store_dword off, v113, off offset:92 ; 4-byte Folded Spill
	s_waitcnt vmcnt(6)
	ds_write_b128 v113, v[54:57] offset:8192
	scratch_store_dword off, v114, off offset:96 ; 4-byte Folded Spill
	scratch_store_dword off, v116, off offset:100 ; 4-byte Folded Spill
	v_and_b32_e32 v72, 64, v122
	s_mul_i32 s14, s21, 0x180
	s_mul_hi_i32 s15, s30, 6
	v_mfma_f32_32x32x16_f16 v[18:33], v[76:79], v[130:133], v[18:33]
	s_movk_i32 s21, 0xffc0
	v_mov_b32_e32 v122, 1.0
	v_mfma_f32_32x32x16_f16 v[18:33], v[80:83], v[134:137], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[84:87], v[138:141], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[88:91], v[142:145], v[18:33]
	s_nop 7
	s_nop 3
	v_max3_f32 v1, v1, v18, v19
	v_max3_f32 v1, v1, v20, v21
	v_max3_f32 v1, v1, v22, v23
	v_max3_f32 v1, v1, v24, v25
	v_max3_f32 v1, v1, v26, v27
	v_max3_f32 v1, v1, v28, v29
	v_max3_f32 v1, v1, v30, v31
	v_max3_f32 v1, v1, v32, v33
	v_mov_b32_e32 v42, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v1, v42
	v_max3_f32 v243, v1, v42, s1
	v_mov_b32_e32 v242, v33
	v_pk_mul_f32 v[42:43], v[242:243], s[16:17] op_sel_hi:[1,0]
	s_addc_u32 s1, s28, s7
	v_fma_f32 v1, v2, s16, -v43
	v_fma_f32 v2, v3, s16, -v43
	v_fma_f32 v3, v4, s16, -v43
	v_fma_f32 v4, v5, s16, -v43
	v_fma_f32 v5, v6, s16, -v43
	v_fma_f32 v6, v7, s16, -v43
	v_fma_f32 v7, v8, s16, -v43
	v_fma_f32 v8, v9, s16, -v43
	v_fma_f32 v9, v10, s16, -v43
	v_fma_f32 v10, v11, s16, -v43
	v_fma_f32 v11, v12, s16, -v43
	v_fma_f32 v12, v13, s16, -v43
	v_fma_f32 v13, v14, s16, -v43
	v_fma_f32 v14, v15, s16, -v43
	v_fma_f32 v15, v16, s16, -v43
	v_fma_f32 v16, v17, s16, -v43
	v_fma_f32 v17, v18, s16, -v43
	v_fma_f32 v18, v19, s16, -v43
	v_fma_f32 v19, v20, s16, -v43
	v_fma_f32 v20, v21, s16, -v43
	v_fma_f32 v21, v22, s16, -v43
	v_fma_f32 v22, v23, s16, -v43
	v_fma_f32 v23, v24, s16, -v43
	v_fma_f32 v24, v25, s16, -v43
	v_fma_f32 v25, v26, s16, -v43
	v_fma_f32 v26, v27, s16, -v43
	v_fma_f32 v27, v28, s16, -v43
	v_fma_f32 v28, v29, s16, -v43
	v_fma_f32 v29, v30, s16, -v43
	v_fma_f32 v30, v31, s16, -v43
	v_fma_f32 v31, v32, s16, -v43
	v_lshlrev_b32_e32 v32, 1, v0
	v_and_b32_e32 v32, 0x60, v32
	v_bitop3_b32 v32, v115, v32, s23 bitop3:0x6c
	s_ashr_i32 s23, s22, 31
	s_lshl_b64 s[22:23], s[22:23], 1
	s_add_u32 s24, s12, s22
	v_lshlrev_b32_e32 v32, 1, v32
	s_addc_u32 s28, s33, s23
	s_and_b32 s1, s1, 0xffff
	v_add3_u32 v67, 0, v32, v92
	s_or_b32 s1, s1, s19
	v_and_b32_e32 v33, 32, v0
	buffer_load_dwordx4 v[158:161], v114, s[0:3], 0 offen
	buffer_load_dwordx4 v[154:157], v116, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(6)
	ds_write_b128 v67, v[34:37] offset:16384
	s_waitcnt vmcnt(5)
	ds_write_b128 v67, v[38:41] offset:24576
	v_lshrrev_b32_e32 v32, 2, v0
	scratch_store_dword off, v33, off offset:116 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v33, 3, v33
	v_and_or_b32 v32, v32, 3, v33
	v_lshlrev_b32_e32 v33, 2, v0
	v_and_b32_e32 v33, 12, v33
	v_and_b32_e32 v35, 64, v115
	v_and_or_b32 v37, v115, 32, v33
	v_bitop3_b32 v36, v33, v115, 32 bitop3:0x72
                                        ; kill: killed $sgpr0_sgpr1
	v_lshlrev_b32_e32 v34, 1, v37
	scratch_store_dword off, v35, off offset:124 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v35, 1, v35
	scratch_store_dword off, v36, off offset:144 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v36, 1, v36
	s_movk_i32 s0, 0x60
	v_add3_u32 v34, 0, v34, v35
	v_add3_u32 v35, 0, v36, v35
	v_bitop3_b32 v36, v115, v33, s0 bitop3:0x4e
	s_and_b32 s0, s28, 0xffff
	s_or_b32 s1, s0, s20
	s_mov_b32 s0, s24
	buffer_load_dwordx4 v[150:153], v44, s[0:3], 0 offen
	buffer_load_dwordx4 v[146:149], v45, s[0:3], 0 offen
	v_sub_f32_e32 v66, 0xff800000, v43
	scratch_store_dword off, v0, off offset:112 ; 4-byte Folded Spill
	v_and_b32_e32 v0, 16, v0
	scratch_store_dword off, v37, off offset:128 ; 4-byte Folded Spill
	v_bitop3_b32 v37, v37, v115, 64 bitop3:0x72
	scratch_store_dword off, v0, off offset:136 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 1, v0
	v_lshlrev_b32_e32 v33, 8, v32
	v_exp_f32_e32 v247, v2
	v_lshlrev_b32_e32 v2, 7, v32
	v_exp_f32_e32 v244, v66
	v_add_u32_e32 v66, 0x4000, v67
	v_add3_u32 v68, v34, v0, v33
	scratch_store_dword off, v37, off offset:140 ; 4-byte Folded Spill
	scratch_store_dword off, v36, off offset:132 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off offset:148 ; 4-byte Folded Spill
	scratch_store_dword off, v44, off offset:80 ; 4-byte Folded Spill
	scratch_store_dword off, v45, off offset:84 ; 4-byte Folded Spill
	scratch_store_dword off, v72, off offset:152 ; 4-byte Folded Spill
	scratch_store_dword off, v67, off offset:88 ; 4-byte Folded Spill
	scratch_store_dword off, v66, off offset:120 ; 4-byte Folded Spill
	scratch_store_dword off, v68, off offset:104 ; 4-byte Folded Spill
	scratch_store_dword off, v93, off offset:48 ; 4-byte Folded Spill
	ds_read_b128 v[206:209], v93
	ds_read_b128 v[170:173], v93 offset:8192
	ds_read_b128 v[210:213], v106
	scratch_store_dword off, v106, off offset:52 ; 4-byte Folded Spill
	ds_read_b128 v[162:165], v106 offset:8192
	ds_read_b128 v[214:217], v107
	scratch_store_dword off, v107, off offset:56 ; 4-byte Folded Spill
	ds_read_b128 v[166:169], v107 offset:8192
	ds_read_b128 v[218:221], v108
	scratch_store_dword off, v108, off offset:60 ; 4-byte Folded Spill
	ds_read_b128 v[174:177], v108 offset:8192
	ds_read_b128 v[222:225], v109
	scratch_store_dword off, v109, off offset:64 ; 4-byte Folded Spill
	v_lshl_add_u32 v34, v37, 1, 0
	s_add_u32 s12, s52, s54
	ds_read_b128 v[178:181], v109 offset:8192
	ds_read_b128 v[194:197], v110
	scratch_store_dword off, v110, off offset:68 ; 4-byte Folded Spill
	v_add3_u32 v70, v34, v0, v33
	v_lshl_add_u32 v34, v36, 1, 0
	s_addc_u32 s13, s53, s55
	ds_read_b128 v[182:185], v110 offset:8192
	ds_read_b128 v[198:201], v111
	scratch_store_dword off, v111, off offset:72 ; 4-byte Folded Spill
	v_add3_u32 v69, v35, v0, v33
	v_add3_u32 v71, v34, v0, v33
	v_sub_f32_e32 v33, v42, v43
	s_lshl_b64 s[12:13], s[12:13], 1
	ds_read_b128 v[186:189], v111 offset:8192
	ds_read_b128 v[202:205], v112
	ds_read_b128 v[190:193], v112 offset:8192
	v_exp_f32_e32 v241, v1
	v_exp_f32_e32 v246, v3
	v_exp_f32_e32 v230, v4
	v_exp_f32_e32 v240, v5
	v_exp_f32_e32 v227, v6
	v_exp_f32_e32 v235, v7
	v_exp_f32_e32 v229, v8
	v_exp_f32_e32 v231, v9
	v_exp_f32_e32 v228, v10
	v_exp_f32_e32 v226, v11
	v_exp_f32_e32 v1, v12
	v_exp_f32_e32 v232, v13
	v_exp_f32_e32 v248, v14
	v_exp_f32_e32 v234, v15
	v_exp_f32_e32 v250, v16
	v_exp_f32_e32 v249, v17
	v_exp_f32_e32 v254, v18
	v_exp_f32_e32 v233, v19
	v_exp_f32_e32 v253, v20
	v_exp_f32_e32 v252, v21
	v_exp_f32_e32 v114, v22
	v_exp_f32_e32 v251, v23
	v_exp_f32_e32 v0, v24
	v_exp_f32_e32 v255, v25
	v_exp_f32_e32 v115, v26
	v_exp_f32_e32 v116, v27
	v_exp_f32_e32 v118, v28
	v_exp_f32_e32 v117, v29
	v_exp_f32_e32 v120, v30
	v_exp_f32_e32 v119, v31
	v_exp_f32_e32 v121, v33
	s_add_u32 s12, s14, s12
	s_addc_u32 s13, s15, s13
	s_add_u32 s4, s4, s12
	v_mov_b32_e32 v2, 0
	s_addc_u32 s5, s5, s13
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
	v_add_u32_e32 v236, 0x4000, v68
	v_add_u32_e32 v237, 0x4000, v69
	v_mov_b32_e32 v124, v70
	v_add_u32_e32 v238, 0x4000, v70
	v_mov_b32_e32 v125, v71
	v_add_u32_e32 v239, 0x4000, v71
                                        ; kill: killed $sgpr0_sgpr1
	scratch_store_dword off, v112, off offset:76 ; 4-byte Folded Spill
	scratch_store_dword off, v69, off offset:108 ; 4-byte Folded Spill
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b64_e32 v[112:113], s[50:51]
	v_mov_b64_e32 v[110:111], s[48:49]
	v_mov_b64_e32 v[108:109], s[46:47]
	v_mov_b64_e32 v[106:107], s[44:45]
	v_mov_b64_e32 v[104:105], s[42:43]
	v_mov_b64_e32 v[102:103], s[40:41]
	v_mov_b64_e32 v[100:101], s[38:39]
	v_mov_b64_e32 v[98:99], s[36:37]
	v_mov_b32_e32 v123, v122
	v_pk_mul_f32 v[50:51], v[50:51], v[244:245] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[206:209], v[126:129], v[98:113]
	scratch_load_dwordx4 v[206:209], off, off ; 16-byte Folded Reload
	v_pk_mul_f32 v[52:53], v[52:53], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[244:245] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[126:129], v[98:113]
	v_pk_mul_f32 v[64:65], v[64:65], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[244:245] op_sel_hi:[1,0]
	s_nop 1
	v_add_f32_e32 v98, v241, v247
	v_add_f32_e32 v98, v98, v246
	v_add_f32_e32 v98, v98, v230
	v_add_f32_e32 v98, v98, v240
	v_add_f32_e32 v98, v98, v227
	v_add_f32_e32 v98, v98, v235
	v_add_f32_e32 v98, v98, v229
	v_add_f32_e32 v98, v98, v231
	v_add_f32_e32 v98, v98, v228
	v_add_f32_e32 v98, v98, v226
	v_add_f32_e32 v98, v98, v1
	v_add_f32_e32 v98, v98, v232
	v_add_f32_e32 v98, v98, v248
	v_add_f32_e32 v98, v98, v234
	v_add_f32_e32 v98, v98, v250
	v_add_f32_e32 v98, v98, v249
	v_add_f32_e32 v98, v98, v254
	v_add_f32_e32 v98, v98, v233
	v_add_f32_e32 v98, v98, v253
	v_add_f32_e32 v98, v98, v252
	v_add_f32_e32 v98, v98, v114
	v_add_f32_e32 v98, v98, v251
	v_add_f32_e32 v98, v98, v0
	v_add_f32_e32 v98, v98, v255
	v_add_f32_e32 v98, v98, v115
	v_add_f32_e32 v98, v98, v116
	v_add_f32_e32 v98, v98, v118
	v_add_f32_e32 v98, v98, v117
	v_add_f32_e32 v98, v98, v120
	v_add_f32_e32 v98, v98, v119
	v_add_f32_e32 v98, v98, v121
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_add_f32_e32 v122, v98, v99
	v_pk_mul_f32 v[42:43], v[42:43], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[244:245] op_sel_hi:[1,0]
	v_fmac_f32_e32 v122, v123, v244
	v_mov_b32_e32 v242, v243
	v_cvt_pk_f16_f32 v110, v241, v247
	s_waitcnt vmcnt(0) lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[210:213], v[206:209], v[66:81]
	scratch_load_dwordx4 v[210:213], off, off offset:16 ; 16-byte Folded Reload
	v_cvt_pk_f16_f32 v111, v246, v230
	v_cvt_pk_f16_f32 v112, v240, v227
	v_cvt_pk_f16_f32 v113, v235, v229
	v_cvt_pk_f16_f32 v106, v231, v228
	v_cvt_pk_f16_f32 v107, v226, v1
	v_cvt_pk_f16_f32 v108, v232, v248
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[206:209], v[82:97]
	v_cvt_pk_f16_f32 v109, v234, v250
	v_cvt_pk_f16_f32 v102, v249, v254
	v_cvt_pk_f16_f32 v103, v233, v253
	v_cvt_pk_f16_f32 v104, v252, v114
	v_cvt_pk_f16_f32 v105, v251, v0
	v_cvt_pk_f16_f32 v98, v255, v115
	v_cvt_pk_f16_f32 v99, v116, v118
	v_cvt_pk_f16_f32 v100, v117, v120
	v_cvt_pk_f16_f32 v101, v119, v121
	s_waitcnt vmcnt(0) lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[214:217], v[210:213], v[66:81]
	scratch_load_dwordx4 v[214:217], off, off offset:32 ; 16-byte Folded Reload
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[210:213], v[82:97]
	s_waitcnt vmcnt(0) lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[218:221], v[214:217], v[66:81]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[214:217], v[82:97]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[222:225], v[130:133], v[66:81]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[130:133], v[82:97]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[194:197], v[134:137], v[66:81]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[134:137], v[82:97]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[66:81], v[198:201], v[138:141], v[66:81]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[138:141], v[82:97]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[66:81], v[202:205], v[142:145], v[66:81]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[190:193], v[142:145], v[82:97]
	; sched_barrier mask(0x00000000)
	s_barrier
	scratch_load_dword v0, off, off offset:104 ; 4-byte Folded Reload
	s_add_u32 s12, s24, s22
	s_addc_u32 s29, s28, s23
	s_and_b32 s0, s5, 0xffff
	s_or_b32 s1, s0, s19
	s_mov_b32 s0, s4
	s_waitcnt vmcnt(0)
	ds_read_b64_tr_b16 v[222:223], v0 offset:16384
	ds_read_b64_tr_b16 v[224:225], v236 offset:2048
	ds_read_b64_tr_b16 v[218:219], v236 offset:4096
	ds_read_b64_tr_b16 v[220:221], v236 offset:6144
	ds_read_b64_tr_b16 v[214:215], v236 offset:8192
	ds_read_b64_tr_b16 v[216:217], v236 offset:10240
	ds_read_b64_tr_b16 v[210:211], v236 offset:12288
	ds_read_b64_tr_b16 v[212:213], v236 offset:14336
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64_tr_b16 v[206:207], v0 offset:16384
	ds_read_b64_tr_b16 v[208:209], v237 offset:2048
	ds_read_b64_tr_b16 v[202:203], v237 offset:4096
	ds_read_b64_tr_b16 v[204:205], v237 offset:6144
	ds_read_b64_tr_b16 v[198:199], v237 offset:8192
	ds_read_b64_tr_b16 v[200:201], v237 offset:10240
	ds_read_b64_tr_b16 v[194:195], v237 offset:12288
	ds_read_b64_tr_b16 v[196:197], v237 offset:14336
	ds_read_b64_tr_b16 v[190:191], v124 offset:16384
	ds_read_b64_tr_b16 v[192:193], v238 offset:2048
	ds_read_b64_tr_b16 v[186:187], v238 offset:4096
	ds_read_b64_tr_b16 v[188:189], v238 offset:6144
	ds_read_b64_tr_b16 v[182:183], v238 offset:8192
	ds_read_b64_tr_b16 v[184:185], v238 offset:10240
	ds_read_b64_tr_b16 v[178:179], v238 offset:12288
	ds_read_b64_tr_b16 v[180:181], v238 offset:14336
	ds_read_b64_tr_b16 v[174:175], v125 offset:16384
	ds_read_b64_tr_b16 v[176:177], v239 offset:2048
	ds_read_b64_tr_b16 v[170:171], v239 offset:4096
	ds_read_b64_tr_b16 v[172:173], v239 offset:6144
	ds_read_b64_tr_b16 v[166:167], v239 offset:8192
	ds_read_b64_tr_b16 v[168:169], v239 offset:10240
	ds_read_b64_tr_b16 v[162:163], v239 offset:12288
	ds_read_b64_tr_b16 v[164:165], v239 offset:14336
	scratch_load_dword v0, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_write_b128 v0, v[158:161]
	ds_write_b128 v0, v[154:157] offset:8192
	scratch_load_dword v0, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[158:161], v0, s[0:3], 0 offen
	s_nop 0
	scratch_load_dword v0, off, off offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[154:157], v0, s[0:3], 0 offen
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[222:225], v[110:113], v[50:65]
	v_max_f32_e32 v0, v67, v67
	v_max_f32_e32 v1, v66, v66
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v68, v69
	v_max3_f32 v0, v0, v70, v71
	v_max3_f32 v0, v0, v72, v73
	v_max3_f32 v0, v0, v74, v75
	v_mfma_f32_32x32x16_f16 v[34:49], v[206:209], v[110:113], v[34:49]
	v_max3_f32 v0, v0, v76, v77
	v_max3_f32 v0, v0, v78, v79
	v_max3_f32 v0, v0, v80, v81
	v_max3_f32 v0, v0, v82, v83
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_mfma_f32_32x32x16_f16 v[18:33], v[190:193], v[110:113], v[18:33]
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_mov_b32_e32 v1, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v1
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[2:17], v[174:177], v[110:113], v[2:17]
	v_max3_f32 v243, v242, v0, v1
	v_pk_mul_f32 v[244:245], v[242:243], s[16:17] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v0, v66, s16, -v245
	v_fma_f32 v66, v68, s16, -v245
	v_fma_f32 v1, v67, s16, -v245
	v_fma_f32 v67, v69, s16, -v245
	v_mfma_f32_32x32x16_f16 v[50:65], v[218:221], v[106:109], v[50:65]
	v_fma_f32 v68, v70, s16, -v245
	v_fma_f32 v69, v71, s16, -v245
	v_fma_f32 v70, v72, s16, -v245
	v_fma_f32 v71, v73, s16, -v245
	v_fma_f32 v72, v74, s16, -v245
	v_fma_f32 v73, v75, s16, -v245
	v_fma_f32 v74, v76, s16, -v245
	v_mfma_f32_32x32x16_f16 v[34:49], v[202:205], v[106:109], v[34:49]
	v_fma_f32 v75, v77, s16, -v245
	v_fma_f32 v76, v78, s16, -v245
	v_fma_f32 v77, v79, s16, -v245
	v_fma_f32 v78, v80, s16, -v245
	v_fma_f32 v79, v81, s16, -v245
	v_fma_f32 v80, v82, s16, -v245
	v_fma_f32 v81, v83, s16, -v245
	v_mfma_f32_32x32x16_f16 v[18:33], v[186:189], v[106:109], v[18:33]
	v_fma_f32 v82, v84, s16, -v245
	v_fma_f32 v83, v85, s16, -v245
	v_fma_f32 v84, v86, s16, -v245
	v_fma_f32 v85, v87, s16, -v245
	v_fma_f32 v86, v88, s16, -v245
	v_fma_f32 v87, v89, s16, -v245
	v_fma_f32 v88, v90, s16, -v245
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[170:173], v[106:109], v[2:17]
	v_fma_f32 v89, v91, s16, -v245
	v_fma_f32 v90, v92, s16, -v245
	v_fma_f32 v91, v93, s16, -v245
	v_fma_f32 v92, v94, s16, -v245
	v_fma_f32 v93, v95, s16, -v245
	v_fma_f32 v94, v96, s16, -v245
	v_fma_f32 v95, v97, s16, -v245
	v_mfma_f32_32x32x16_f16 v[50:65], v[214:217], v[102:105], v[50:65]
	v_exp_f32_e32 v246, v66
	v_sub_f32_e32 v66, v244, v245
	v_exp_f32_e32 v241, v0
	v_exp_f32_e32 v247, v1
	v_exp_f32_e32 v230, v67
	v_exp_f32_e32 v240, v68
	v_exp_f32_e32 v227, v69
	v_mfma_f32_32x32x16_f16 v[34:49], v[198:201], v[102:105], v[34:49]
	v_exp_f32_e32 v235, v70
	v_exp_f32_e32 v229, v71
	v_exp_f32_e32 v231, v72
	v_exp_f32_e32 v228, v73
	v_exp_f32_e32 v226, v74
	v_exp_f32_e32 v1, v75
	v_exp_f32_e32 v232, v76
	v_mfma_f32_32x32x16_f16 v[18:33], v[182:185], v[102:105], v[18:33]
	v_exp_f32_e32 v248, v77
	v_exp_f32_e32 v234, v78
	v_exp_f32_e32 v250, v79
	v_exp_f32_e32 v249, v80
	v_exp_f32_e32 v254, v81
	v_exp_f32_e32 v233, v82
	v_exp_f32_e32 v253, v83
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[166:169], v[102:105], v[2:17]
	v_exp_f32_e32 v252, v84
	v_exp_f32_e32 v114, v85
	v_exp_f32_e32 v251, v86
	v_exp_f32_e32 v0, v87
	v_exp_f32_e32 v255, v88
	v_exp_f32_e32 v115, v89
	v_exp_f32_e32 v116, v90
	v_mfma_f32_32x32x16_f16 v[50:65], v[210:213], v[98:101], v[50:65]
	v_exp_f32_e32 v118, v91
	v_exp_f32_e32 v117, v92
	v_exp_f32_e32 v120, v93
	v_exp_f32_e32 v119, v94
	v_exp_f32_e32 v121, v95
	v_exp_f32_e32 v244, v66
	v_mfma_f32_32x32x16_f16 v[34:49], v[194:197], v[98:101], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[178:181], v[98:101], v[18:33]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[162:165], v[98:101], v[2:17]
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v66, off, off offset:88 ; 4-byte Folded Reload
	s_and_b32 s0, s29, 0xffff
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_or_b32 s13, s0, s20
	s_add_u32 s24, s24, s22
	s_addc_u32 s28, s28, s23
	s_add_u32 s4, s4, s6
	s_addc_u32 s5, s5, s7
	s_add_i32 s21, s21, 64
	s_cmpk_lt_u32 s21, 0x1f00
	s_waitcnt vmcnt(0)
	ds_write_b128 v66, v[150:153] offset:16384
	v_add_u32_e32 v66, 0x4000, v66
	ds_write_b128 v66, v[146:149] offset:8192
	scratch_load_dword v66, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[150:153], v66, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v66, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[146:149], v66, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v66, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 v[206:209], v66
	ds_read_b128 v[170:173], v66 offset:8192
	scratch_load_dword v66, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 v[210:213], v66
	ds_read_b128 v[162:165], v66 offset:8192
	scratch_load_dword v66, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 v[214:217], v66
	ds_read_b128 v[166:169], v66 offset:8192
	scratch_load_dword v66, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 v[218:221], v66
	ds_read_b128 v[174:177], v66 offset:8192
	scratch_load_dword v66, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 v[222:225], v66
	ds_read_b128 v[178:181], v66 offset:8192
	scratch_load_dword v66, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 v[194:197], v66
	ds_read_b128 v[182:185], v66 offset:8192
	scratch_load_dword v66, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 v[198:201], v66
	ds_read_b128 v[186:189], v66 offset:8192
	scratch_load_dword v66, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b128 v[202:205], v66
	ds_read_b128 v[190:193], v66 offset:8192
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	scratch_load_dword v67, off, off offset:112 ; 4-byte Folded Reload
	scratch_load_dword v68, off, off offset:152 ; 4-byte Folded Reload
	scratch_load_dwordx4 v[98:101], off, off ; 16-byte Folded Reload
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
	v_add_f32_e32 v123, v241, v247
	v_add_f32_e32 v123, v123, v246
	v_add_f32_e32 v123, v123, v230
	v_add_f32_e32 v123, v123, v240
	v_add_f32_e32 v123, v123, v227
	v_add_f32_e32 v123, v123, v235
	v_add_f32_e32 v123, v123, v229
	v_add_f32_e32 v123, v123, v231
	v_add_f32_e32 v123, v123, v228
	v_add_f32_e32 v123, v123, v226
	v_add_f32_e32 v123, v123, v1
	v_add_f32_e32 v123, v123, v232
	v_add_f32_e32 v123, v123, v248
	v_add_f32_e32 v123, v123, v234
	v_add_f32_e32 v123, v123, v250
	v_add_f32_e32 v123, v123, v249
	v_add_f32_e32 v123, v123, v254
	v_add_f32_e32 v123, v123, v233
	v_add_f32_e32 v123, v123, v253
	v_add_f32_e32 v123, v123, v252
	v_add_f32_e32 v123, v123, v114
	v_add_f32_e32 v123, v123, v251
	v_add_f32_e32 v123, v123, v0
	v_add_f32_e32 v123, v123, v255
	v_add_f32_e32 v123, v123, v115
	s_mul_i32 s2, s18, 0xc0000
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s8, s2
	s_addc_u32 s5, s9, s3
	s_lshl_b32 s2, s17, 14
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s4, s2
	v_pk_mul_f32 v[16:17], v[16:17], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[244:245] op_sel_hi:[1,0]
	s_addc_u32 s5, s5, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 2
	s_add_u32 s4, s4, s2
	s_addc_u32 s16, s5, s3
	s_add_i32 s2, s34, 0xffffc100
	v_pk_mul_f32 v[64:65], v[64:65], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[244:245] op_sel_hi:[1,0]
	s_add_u32 s12, s12, s22
	s_addc_u32 s3, s29, s23
	s_and_b32 s3, s3, 0xffff
	s_or_b32 s13, s3, s20
	v_add_f32_e32 v123, v123, v116
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v66, 0x100, v67
	v_cmp_eq_u32_e64 s[0:1], 0, v66
	v_lshrrev_b32_e32 v66, 1, v67
	v_and_b32_e32 v66, 0xa0, v66
	v_and_b32_e32 v67, 31, v67
	s_waitcnt vmcnt(1)
	v_or3_b32 v66, v66, v67, v68
	scratch_store_dword off, v66, off offset:96 ; 4-byte Folded Spill
	v_mov_b64_e32 v[80:81], s[50:51]
	v_mov_b64_e32 v[78:79], s[48:49]
	v_mov_b64_e32 v[76:77], s[46:47]
	v_mov_b64_e32 v[74:75], s[44:45]
	v_mov_b64_e32 v[72:73], s[42:43]
	v_mov_b64_e32 v[70:71], s[40:41]
	v_mov_b64_e32 v[68:69], s[38:39]
	v_mov_b64_e32 v[66:67], s[36:37]
	s_cmp_lt_i32 s2, 1
	v_add_f32_e32 v123, v123, v118
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[82:97], v[206:209], v[126:129], v[66:81]
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[208:209], v[100:101]
	v_mov_b64_e32 v[206:207], v[98:99]
	s_mov_b32 s2, 0x3e0293ee
	v_add_f32_e32 v123, v123, v117
	v_add_f32_e32 v123, v123, v120
	v_add_f32_e32 v123, v123, v119
	v_add_f32_e32 v123, v123, v121
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[82:97], v[210:213], v[98:101], v[82:97]
	scratch_load_dwordx4 v[98:101], off, off offset:16 ; 16-byte Folded Reload
	v_pk_mul_f32 v[48:49], v[48:49], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[244:245] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[244:245] op_sel_hi:[1,0]
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_mov_b32_e32 v124, v123
	s_nop 1
	v_permlane32_swap_b32_e32 v123, v124
	v_add_f32_e32 v123, v123, v124
	v_fmac_f32_e32 v123, v122, v244
	s_waitcnt vmcnt(0) lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[82:97], v[214:217], v[98:101], v[82:97]
	v_mov_b64_e32 v[212:213], v[100:101]
	v_mov_b64_e32 v[210:211], v[98:99]
	scratch_load_dwordx4 v[98:101], off, off offset:32 ; 16-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[216:217], v[100:101]
	v_mfma_f32_32x32x16_f16 v[82:97], v[218:221], v[98:101], v[82:97]
	v_mov_b64_e32 v[214:215], v[98:99]
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[126:129], v[66:81]
	v_cvt_pk_f16_f32 v173, v251, v0
	v_cvt_pk_f16_f32 v172, v252, v114
	v_cvt_pk_f16_f32 v171, v233, v253
	v_cvt_pk_f16_f32 v170, v249, v254
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[206:209], v[98:113]
	v_cvt_pk_f16_f32 v162, v241, v247
	v_cvt_pk_f16_f32 v163, v246, v230
	v_cvt_pk_f16_f32 v164, v240, v227
	v_cvt_pk_f16_f32 v165, v235, v229
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[210:213], v[98:113]
	v_cvt_pk_f16_f32 v167, v226, v1
	v_cvt_pk_f16_f32 v166, v231, v228
	v_cvt_pk_f16_f32 v168, v232, v248
	v_cvt_pk_f16_f32 v169, v234, v250
	v_mfma_f32_32x32x16_f16 v[98:113], v[174:177], v[214:217], v[98:113]
	v_cvt_pk_f16_f32 v174, v255, v115
	scratch_load_dword v1, off, off offset:124 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:128 ; 4-byte Folded Reload
	scratch_load_dword v115, off, off offset:136 ; 4-byte Folded Reload
	scratch_load_dword v114, off, off offset:148 ; 4-byte Folded Reload
	v_cvt_pk_f16_f32 v175, v116, v118
	v_cvt_pk_f16_f32 v176, v117, v120
	v_cvt_pk_f16_f32 v177, v119, v121
	s_waitcnt vmcnt(1)
	v_or3_b32 v0, v1, v115, v0
	v_lshlrev_b32_e32 v0, 1, v0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v114, 1, v114
	v_add3_u32 v242, 0, v0, v114
	ds_read_b64_tr_b16 v[238:239], v242 offset:16384
	ds_read_b64_tr_b16 v[240:241], v242 offset:18432
	ds_read_b64_tr_b16 v[234:235], v242 offset:20480
	ds_read_b64_tr_b16 v[236:237], v242 offset:22528
	ds_read_b64_tr_b16 v[226:227], v242 offset:24576
	ds_read_b64_tr_b16 v[228:229], v242 offset:26624
	ds_read_b64_tr_b16 v[230:231], v242 offset:28672
	ds_read_b64_tr_b16 v[232:233], v242 offset:30720
	scratch_load_dword v0, off, off offset:144 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[82:97], v[222:225], v[130:133], v[82:97]
	s_waitcnt vmcnt(0)
	v_or3_b32 v0, v115, v0, v1
	v_lshlrev_b32_e32 v0, 1, v0
	v_add3_u32 v247, 0, v0, v114
	ds_read_b64_tr_b16 v[222:223], v247 offset:16384
	ds_read_b64_tr_b16 v[224:225], v247 offset:18432
	ds_read_b64_tr_b16 v[218:219], v247 offset:20480
	ds_read_b64_tr_b16 v[220:221], v247 offset:22528
	ds_read_b64_tr_b16 v[210:211], v247 offset:24576
	ds_read_b64_tr_b16 v[212:213], v247 offset:26624
	ds_read_b64_tr_b16 v[214:215], v247 offset:28672
	ds_read_b64_tr_b16 v[216:217], v247 offset:30720
	scratch_load_dword v0, off, off offset:140 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[82:97], v[194:197], v[134:137], v[82:97]
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v0, v0, v115
	v_mfma_f32_32x32x16_f16 v[82:97], v[198:201], v[138:141], v[82:97]
	v_lshlrev_b32_e32 v0, 1, v0
	v_add3_u32 v1, 0, v0, v114
	v_mfma_f32_32x32x16_f16 v[82:97], v[202:205], v[142:145], v[82:97]
	ds_read_b64_tr_b16 v[194:195], v1 offset:16384
	ds_read_b64_tr_b16 v[196:197], v1 offset:18432
	ds_read_b64_tr_b16 v[198:199], v1 offset:20480
	ds_read_b64_tr_b16 v[200:201], v1 offset:22528
	ds_read_b64_tr_b16 v[202:203], v1 offset:24576
	ds_read_b64_tr_b16 v[204:205], v1 offset:26624
	ds_read_b64_tr_b16 v[206:207], v1 offset:28672
	ds_read_b64_tr_b16 v[208:209], v1 offset:30720
	scratch_load_dword v0, off, off offset:132 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v0, v0, v115
	v_mfma_f32_32x32x16_f16 v[98:113], v[178:181], v[130:133], v[98:113]
	v_lshlrev_b32_e32 v0, 1, v0
	v_add3_u32 v246, 0, v0, v114
	v_max_f32_e32 v0, v83, v83
	v_max_f32_e32 v114, v82, v82
	v_max_f32_e32 v0, v114, v0
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_mfma_f32_32x32x16_f16 v[98:113], v[182:185], v[134:137], v[98:113]
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_mfma_f32_32x32x16_f16 v[98:113], v[186:189], v[138:141], v[98:113]
	v_mfma_f32_32x32x16_f16 v[98:113], v[190:193], v[142:145], v[98:113]
	ds_read_b64_tr_b16 v[178:179], v246 offset:16384
	ds_read_b64_tr_b16 v[180:181], v246 offset:18432
	ds_read_b64_tr_b16 v[182:183], v246 offset:20480
	ds_read_b64_tr_b16 v[184:185], v246 offset:22528
	ds_read_b64_tr_b16 v[186:187], v246 offset:24576
	ds_read_b64_tr_b16 v[188:189], v246 offset:26624
	ds_read_b64_tr_b16 v[190:191], v246 offset:28672
	ds_read_b64_tr_b16 v[192:193], v246 offset:30720
	scratch_load_dword v125, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_write_b128 v125, v[158:161]
	ds_write_b128 v125, v[154:157] offset:8192
	v_max3_f32 v0, v0, v98, v99
	v_max3_f32 v0, v0, v100, v101
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[2:17], v[178:181], v[162:165], v[2:17]
	v_max3_f32 v0, v0, v102, v103
	v_max3_f32 v0, v0, v104, v105
	v_max3_f32 v0, v0, v106, v107
	v_max3_f32 v0, v0, v108, v109
	v_max3_f32 v0, v0, v110, v111
	v_max3_f32 v0, v0, v112, v113
	v_mov_b32_e32 v114, v0
	v_mfma_f32_32x32x16_f16 v[50:65], v[238:241], v[162:165], v[50:65]
	s_nop 0
	v_permlane32_swap_b32_e32 v0, v114
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[182:185], v[166:169], v[2:17]
	v_max3_f32 v185, v243, v0, v114
	v_mov_b32_e32 v184, v113
	v_pk_mul_f32 v[182:183], v[184:185], s[2:3] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v0, v82, s2, -v183
	v_fma_f32 v82, v83, s2, -v183
	v_fma_f32 v83, v84, s2, -v183
	v_fma_f32 v84, v85, s2, -v183
	v_fma_f32 v85, v86, s2, -v183
	v_fma_f32 v86, v87, s2, -v183
	v_fma_f32 v87, v88, s2, -v183
	v_fma_f32 v88, v89, s2, -v183
	v_fma_f32 v89, v90, s2, -v183
	v_exp_f32_e32 v115, v82
	v_sub_f32_e32 v82, v245, v183
	v_mfma_f32_32x32x16_f16 v[50:65], v[234:237], v[166:169], v[50:65]
	v_fma_f32 v90, v91, s2, -v183
	v_fma_f32 v91, v92, s2, -v183
	v_fma_f32 v92, v93, s2, -v183
	v_fma_f32 v93, v94, s2, -v183
	v_fma_f32 v94, v95, s2, -v183
	v_fma_f32 v95, v96, s2, -v183
	v_fma_f32 v96, v97, s2, -v183
	v_fma_f32 v97, v98, s2, -v183
	v_fma_f32 v98, v99, s2, -v183
	v_fma_f32 v99, v100, s2, -v183
	v_fma_f32 v100, v101, s2, -v183
	v_fma_f32 v101, v102, s2, -v183
	v_fma_f32 v102, v103, s2, -v183
	v_fma_f32 v103, v104, s2, -v183
	v_fma_f32 v104, v105, s2, -v183
	v_fma_f32 v105, v106, s2, -v183
	v_fma_f32 v106, v107, s2, -v183
	v_fma_f32 v107, v108, s2, -v183
	v_fma_f32 v108, v109, s2, -v183
	v_fma_f32 v109, v110, s2, -v183
	v_fma_f32 v110, v111, s2, -v183
	v_fma_f32 v111, v112, s2, -v183
	v_sub_f32_e32 v112, v182, v183
	v_exp_f32_e32 v114, v83
	v_exp_f32_e32 v116, v84
	v_exp_f32_e32 v117, v85
	v_exp_f32_e32 v118, v86
	v_exp_f32_e32 v119, v87
	v_exp_f32_e32 v120, v88
	v_exp_f32_e32 v121, v89
	scratch_load_dword v234, off, off offset:88 ; 4-byte Folded Reload
	v_exp_f32_e32 v182, v82
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v82, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v83, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v84, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v85, off, off offset:60 ; 4-byte Folded Reload
	scratch_load_dword v86, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v87, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v88, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v89, off, off offset:76 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[34:49], v[222:225], v[162:165], v[34:49]
	v_exp_f32_e32 v122, v90
	v_exp_f32_e32 v124, v91
	v_exp_f32_e32 v125, v92
	v_exp_f32_e32 v0, v0
	s_waitcnt vmcnt(3)
	ds_read_b128 v[222:225], v86
	v_mfma_f32_32x32x16_f16 v[18:33], v[194:197], v[162:165], v[18:33]
	v_exp_f32_e32 v194, v101
	v_exp_f32_e32 v195, v102
	v_exp_f32_e32 v196, v103
	v_exp_f32_e32 v197, v104
	v_mfma_f32_32x32x16_f16 v[34:49], v[218:221], v[166:169], v[34:49]
	ds_read_b128 v[218:221], v85
	v_mfma_f32_32x32x16_f16 v[18:33], v[198:201], v[166:169], v[18:33]
	v_exp_f32_e32 v199, v106
	v_exp_f32_e32 v200, v107
	v_exp_f32_e32 v201, v108
	v_exp_f32_e32 v198, v105
	v_mfma_f32_32x32x16_f16 v[50:65], v[226:229], v[170:173], v[50:65]
	s_waitcnt vmcnt(2)
	ds_read_b128 v[226:229], v87
	v_mfma_f32_32x32x16_f16 v[34:49], v[210:213], v[170:173], v[34:49]
	ds_read_b128 v[210:213], v83
	v_mfma_f32_32x32x16_f16 v[18:33], v[202:205], v[170:173], v[18:33]
	v_exp_f32_e32 v202, v109
	v_exp_f32_e32 v203, v110
	v_exp_f32_e32 v204, v111
	v_exp_f32_e32 v205, v112
	v_mfma_f32_32x32x16_f16 v[2:17], v[186:189], v[170:173], v[2:17]
	v_exp_f32_e32 v186, v93
	v_exp_f32_e32 v187, v94
	v_exp_f32_e32 v188, v95
	v_exp_f32_e32 v189, v96
	v_mfma_f32_32x32x16_f16 v[50:65], v[230:233], v[174:177], v[50:65]
	s_waitcnt vmcnt(1)
	ds_read_b128 v[230:233], v88
	v_mfma_f32_32x32x16_f16 v[34:49], v[214:217], v[174:177], v[34:49]
	ds_read_b128 v[214:217], v84
	s_nop 7
	v_pk_mul_f32 v[64:65], v[64:65], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[182:183] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[18:33], v[206:209], v[174:177], v[18:33]
	ds_read_b128 v[206:209], v82
	v_pk_mul_f32 v[52:53], v[52:53], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[182:183] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[2:17], v[190:193], v[174:177], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b128 v[178:181], v89
	ds_read_b128 v[174:177], v82 offset:8192
	ds_read_b128 v[106:109], v83 offset:8192
	ds_read_b128 v[110:113], v84 offset:8192
	ds_read_b128 v[154:157], v85 offset:8192
	ds_read_b128 v[158:161], v86 offset:8192
	ds_read_b128 v[162:165], v87 offset:8192
	ds_read_b128 v[166:169], v88 offset:8192
	ds_read_b128 v[170:173], v89 offset:8192
	scratch_load_dword v82, off, off offset:80 ; 4-byte Folded Reload
	v_exp_f32_e32 v191, v98
	v_exp_f32_e32 v192, v99
	v_exp_f32_e32 v193, v100
	scratch_load_dword v235, off, off offset:120 ; 4-byte Folded Reload
	v_exp_f32_e32 v190, v97
	ds_write_b128 v234, v[150:153] offset:16384
	scratch_load_dwordx4 v[150:153], off, off offset:16 ; 16-byte Folded Reload
	v_pk_mul_f32 v[40:41], v[40:41], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[182:183] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[182:183] op_sel_hi:[1,0]
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[98:101], v82, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v82, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[102:105], v82, s[12:15], 0 offen
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[82:97], v[206:209], v[126:129], v[66:81]
	scratch_load_dwordx4 v[206:209], off, off offset:32 ; 16-byte Folded Reload
	ds_write_b128 v235, v[146:149] offset:8192
	scratch_load_dwordx4 v[146:149], off, off ; 16-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_f16 v[82:97], v[210:213], v[146:149], v[82:97]
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[126:129], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[214:217], v[150:153], v[82:97]
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[146:149], v[66:81]
	v_add_f32_e32 v106, v0, v115
	v_add_f32_e32 v106, v114, v106
	v_add_f32_e32 v106, v116, v106
	v_add_f32_e32 v106, v117, v106
	v_add_f32_e32 v106, v118, v106
	v_add_f32_e32 v106, v119, v106
	v_add_f32_e32 v106, v120, v106
	v_mfma_f32_32x32x16_f16 v[82:97], v[218:221], v[206:209], v[82:97]
	v_add_f32_e32 v106, v121, v106
	v_add_f32_e32 v106, v122, v106
	v_add_f32_e32 v106, v124, v106
	v_add_f32_e32 v106, v125, v106
	v_add_f32_e32 v106, v186, v106
	v_add_f32_e32 v106, v187, v106
	v_add_f32_e32 v106, v188, v106
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[150:153], v[66:81]
	v_add_f32_e32 v106, v189, v106
	v_add_f32_e32 v106, v190, v106
	v_add_f32_e32 v106, v191, v106
	v_add_f32_e32 v106, v192, v106
	v_add_f32_e32 v106, v193, v106
	v_add_f32_e32 v106, v194, v106
	v_add_f32_e32 v106, v195, v106
	v_mfma_f32_32x32x16_f16 v[82:97], v[222:225], v[130:133], v[82:97]
	v_add_f32_e32 v106, v196, v106
	v_add_f32_e32 v106, v197, v106
	v_add_f32_e32 v106, v198, v106
	v_add_f32_e32 v106, v199, v106
	v_add_f32_e32 v106, v200, v106
	v_add_f32_e32 v106, v201, v106
	v_add_f32_e32 v106, v202, v106
	v_mfma_f32_32x32x16_f16 v[66:81], v[154:157], v[206:209], v[66:81]
	v_add_f32_e32 v106, v203, v106
	v_add_f32_e32 v106, v204, v106
	v_add_f32_e32 v106, v205, v106
	v_mov_b32_e32 v107, v106
	s_nop 1
	v_permlane32_swap_b32_e32 v106, v107
	v_add_f32_e32 v184, v106, v107
	v_mfma_f32_32x32x16_f16 v[82:97], v[226:229], v[134:137], v[82:97]
	v_cvt_pk_f16_f32 v106, v0, v115
	v_cvt_pk_f16_f32 v107, v114, v116
	v_cvt_pk_f16_f32 v108, v117, v118
	v_cvt_pk_f16_f32 v109, v119, v120
	v_fmac_f32_e32 v184, v123, v182
	v_cvt_pk_f16_f32 v110, v121, v122
	v_cvt_pk_f16_f32 v111, v124, v125
	v_mfma_f32_32x32x16_f16 v[66:81], v[158:161], v[130:133], v[66:81]
	v_cvt_pk_f16_f32 v112, v186, v187
	v_cvt_pk_f16_f32 v113, v188, v189
	v_cvt_pk_f16_f32 v114, v190, v191
	v_cvt_pk_f16_f32 v115, v192, v193
	v_cvt_pk_f16_f32 v116, v194, v195
	v_cvt_pk_f16_f32 v117, v196, v197
	v_cvt_pk_f16_f32 v118, v198, v199
	v_mfma_f32_32x32x16_f16 v[82:97], v[230:233], v[138:141], v[82:97]
	v_cvt_pk_f16_f32 v119, v200, v201
	v_cvt_pk_f16_f32 v120, v202, v203
	v_cvt_pk_f16_f32 v121, v204, v205
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[134:137], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[142:145], v[82:97]
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[138:141], v[66:81]
	s_nop 7
	s_nop 2
	v_max_f32_e32 v0, v83, v83
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[142:145], v[66:81]
	ds_read_b64_tr_b16 v[186:187], v242 offset:16384
	ds_read_b64_tr_b16 v[188:189], v242 offset:18432
	ds_read_b64_tr_b16 v[178:179], v242 offset:20480
	ds_read_b64_tr_b16 v[180:181], v242 offset:22528
	ds_read_b64_tr_b16 v[170:171], v242 offset:24576
	ds_read_b64_tr_b16 v[172:173], v242 offset:26624
	ds_read_b64_tr_b16 v[174:175], v242 offset:28672
	ds_read_b64_tr_b16 v[176:177], v242 offset:30720
	ds_read_b64_tr_b16 v[166:167], v247 offset:16384
	ds_read_b64_tr_b16 v[168:169], v247 offset:18432
	ds_read_b64_tr_b16 v[162:163], v247 offset:20480
	ds_read_b64_tr_b16 v[164:165], v247 offset:22528
	ds_read_b64_tr_b16 v[154:155], v247 offset:24576
	ds_read_b64_tr_b16 v[156:157], v247 offset:26624
	ds_read_b64_tr_b16 v[158:159], v247 offset:28672
	ds_read_b64_tr_b16 v[160:161], v247 offset:30720
	ds_read_b64_tr_b16 v[138:139], v1 offset:16384
	ds_read_b64_tr_b16 v[140:141], v1 offset:18432
	ds_read_b64_tr_b16 v[142:143], v1 offset:20480
	ds_read_b64_tr_b16 v[144:145], v1 offset:22528
	ds_read_b64_tr_b16 v[146:147], v1 offset:24576
	ds_read_b64_tr_b16 v[148:149], v1 offset:26624
	ds_read_b64_tr_b16 v[150:151], v1 offset:28672
	ds_read_b64_tr_b16 v[152:153], v1 offset:30720
	ds_read_b64_tr_b16 v[122:123], v246 offset:16384
	ds_read_b64_tr_b16 v[124:125], v246 offset:18432
	ds_read_b64_tr_b16 v[126:127], v246 offset:20480
	ds_read_b64_tr_b16 v[128:129], v246 offset:22528
	ds_read_b64_tr_b16 v[130:131], v246 offset:24576
	ds_read_b64_tr_b16 v[132:133], v246 offset:26624
	ds_read_b64_tr_b16 v[134:135], v246 offset:28672
	ds_read_b64_tr_b16 v[136:137], v246 offset:30720
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v234, v[98:101] offset:16384
	ds_write_b128 v235, v[102:105] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x16_f16 v[50:65], v[186:189], v[106:109], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[106:109], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[138:141], v[106:109], v[18:33]
	v_mfma_f32_32x32x16_f16 v[2:17], v[122:125], v[106:109], v[2:17]
	v_max_f32_e32 v106, v82, v82
	v_max_f32_e32 v0, v106, v0
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_max3_f32 v0, v0, v66, v67
	v_max3_f32 v0, v0, v68, v69
	v_max3_f32 v0, v0, v70, v71
	v_max3_f32 v0, v0, v72, v73
	v_max3_f32 v0, v0, v74, v75
	v_max3_f32 v0, v0, v76, v77
	v_max3_f32 v0, v0, v78, v79
	v_max3_f32 v0, v0, v80, v81
	v_mov_b32_e32 v106, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v106
	v_mfma_f32_32x32x16_f16 v[18:33], v[142:145], v[110:113], v[18:33]
	v_max3_f32 v143, v185, v0, v106
	v_mov_b32_e32 v142, v81
	v_pk_mul_f32 v[106:107], v[142:143], s[2:3] op_sel_hi:[1,0]
	s_nop 0
	v_fma_f32 v0, v82, s2, -v107
	v_fma_f32 v81, v83, s2, -v107
	v_fma_f32 v82, v84, s2, -v107
	v_fma_f32 v83, v85, s2, -v107
	v_fma_f32 v85, v87, s2, -v107
	v_fma_f32 v87, v89, s2, -v107
	v_fma_f32 v89, v91, s2, -v107
	v_fma_f32 v91, v93, s2, -v107
	v_fma_f32 v93, v95, s2, -v107
	v_fma_f32 v95, v97, s2, -v107
	v_exp_f32_e32 v97, v0
	v_exp_f32_e32 v81, v81
	v_exp_f32_e32 v82, v82
	v_fma_f32 v84, v86, s2, -v107
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v84, v84
	v_sub_f32_e32 v0, v183, v107
	v_fma_f32 v86, v88, s2, -v107
	v_exp_f32_e32 v85, v85
	v_exp_f32_e32 v142, v0
	v_add_f32_e32 v0, v97, v81
	v_exp_f32_e32 v86, v86
	v_add_f32_e32 v0, v82, v0
	v_fma_f32 v88, v90, s2, -v107
	v_exp_f32_e32 v87, v87
	v_add_f32_e32 v0, v83, v0
	v_exp_f32_e32 v88, v88
	v_add_f32_e32 v0, v84, v0
	v_fma_f32 v90, v92, s2, -v107
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v0, v85, v0
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v0, v86, v0
	v_fma_f32 v92, v94, s2, -v107
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v0, v87, v0
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v0, v88, v0
	v_mfma_f32_32x32x16_f16 v[50:65], v[178:181], v[110:113], v[50:65]
	v_fma_f32 v94, v96, s2, -v107
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v0, v89, v0
	v_exp_f32_e32 v94, v94
	v_add_f32_e32 v0, v90, v0
	v_fma_f32 v66, v66, s2, -v107
	v_exp_f32_e32 v95, v95
	v_mfma_f32_32x32x16_f16 v[34:49], v[162:165], v[110:113], v[34:49]
	v_add_f32_e32 v0, v91, v0
	v_fma_f32 v67, v67, s2, -v107
	v_exp_f32_e32 v66, v66
	v_add_f32_e32 v0, v92, v0
	v_fma_f32 v68, v68, s2, -v107
	v_exp_f32_e32 v67, v67
	v_add_f32_e32 v0, v93, v0
	v_mfma_f32_32x32x16_f16 v[2:17], v[126:129], v[110:113], v[2:17]
	v_fma_f32 v69, v69, s2, -v107
	v_exp_f32_e32 v68, v68
	v_add_f32_e32 v0, v94, v0
	v_fma_f32 v70, v70, s2, -v107
	v_exp_f32_e32 v69, v69
	v_add_f32_e32 v0, v95, v0
	v_fma_f32 v71, v71, s2, -v107
	v_sub_f32_e32 v96, v106, v107
	v_exp_f32_e32 v106, v70
	v_add_f32_e32 v0, v66, v0
	v_mfma_f32_32x32x16_f16 v[50:65], v[170:173], v[114:117], v[50:65]
	v_fma_f32 v72, v72, s2, -v107
	v_exp_f32_e32 v108, v71
	v_add_f32_e32 v0, v67, v0
	v_fma_f32 v73, v73, s2, -v107
	v_exp_f32_e32 v109, v72
	v_add_f32_e32 v0, v68, v0
	v_fma_f32 v74, v74, s2, -v107
	v_mfma_f32_32x32x16_f16 v[34:49], v[154:157], v[114:117], v[34:49]
	v_exp_f32_e32 v73, v73
	v_add_f32_e32 v0, v69, v0
	v_fma_f32 v75, v75, s2, -v107
	v_exp_f32_e32 v110, v74
	v_add_f32_e32 v0, v106, v0
	v_fma_f32 v76, v76, s2, -v107
	v_exp_f32_e32 v111, v75
	v_mfma_f32_32x32x16_f16 v[18:33], v[146:149], v[114:117], v[18:33]
	v_add_f32_e32 v0, v108, v0
	v_fma_f32 v77, v77, s2, -v107
	v_exp_f32_e32 v112, v76
	v_add_f32_e32 v0, v109, v0
	v_fma_f32 v78, v78, s2, -v107
	v_exp_f32_e32 v113, v77
	v_add_f32_e32 v0, v73, v0
	v_mfma_f32_32x32x16_f16 v[2:17], v[130:133], v[114:117], v[2:17]
	v_fma_f32 v79, v79, s2, -v107
	v_exp_f32_e32 v114, v78
	v_add_f32_e32 v0, v110, v0
	v_fma_f32 v80, v80, s2, -v107
	v_exp_f32_e32 v115, v79
	v_add_f32_e32 v0, v111, v0
	v_exp_f32_e32 v116, v80
	v_mfma_f32_32x32x16_f16 v[50:65], v[174:177], v[118:121], v[50:65]
	v_add_f32_e32 v0, v112, v0
	v_exp_f32_e32 v96, v96
	v_add_f32_e32 v0, v113, v0
	v_add_f32_e32 v0, v114, v0
	v_add_f32_e32 v0, v115, v0
	v_add_f32_e32 v0, v116, v0
	v_add_f32_e32 v0, v96, v0
	v_mfma_f32_32x32x16_f16 v[34:49], v[158:161], v[118:121], v[34:49]
	v_mov_b32_e32 v70, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v70
	v_cvt_pk_f16_f32 v78, v97, v81
	v_cvt_pk_f16_f32 v79, v82, v83
	v_cvt_pk_f16_f32 v80, v84, v85
	v_cvt_pk_f16_f32 v81, v86, v87
	v_mfma_f32_32x32x16_f16 v[18:33], v[150:153], v[118:121], v[18:33]
	v_pk_mul_f32 v[64:65], v[64:65], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[142:143] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[2:17], v[134:137], v[118:121], v[2:17]
	v_pk_mul_f32 v[50:51], v[50:51], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[142:143] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[142:143] op_sel_hi:[1,0]
	v_add_f32_e32 v0, v0, v70
	v_cvt_pk_f16_f32 v74, v88, v89
	v_cvt_pk_f16_f32 v75, v90, v91
	v_cvt_pk_f16_f32 v76, v92, v93
	v_cvt_pk_f16_f32 v77, v94, v95
	v_cvt_pk_f16_f32 v70, v66, v67
	v_cvt_pk_f16_f32 v71, v68, v69
	v_cvt_pk_f16_f32 v72, v106, v108
	v_cvt_pk_f16_f32 v73, v109, v73
	v_cvt_pk_f16_f32 v66, v110, v111
	v_cvt_pk_f16_f32 v67, v112, v113
	v_cvt_pk_f16_f32 v68, v114, v115
	v_cvt_pk_f16_f32 v69, v116, v96
	ds_read_b64_tr_b16 v[144:145], v242 offset:16384
	ds_read_b64_tr_b16 v[146:147], v242 offset:18432
	ds_read_b64_tr_b16 v[138:139], v242 offset:20480
	ds_read_b64_tr_b16 v[140:141], v242 offset:22528
	ds_read_b64_tr_b16 v[134:135], v242 offset:24576
	ds_read_b64_tr_b16 v[136:137], v242 offset:26624
	ds_read_b64_tr_b16 v[130:131], v242 offset:28672
	ds_read_b64_tr_b16 v[132:133], v242 offset:30720
	ds_read_b64_tr_b16 v[126:127], v247 offset:16384
	ds_read_b64_tr_b16 v[128:129], v247 offset:18432
	ds_read_b64_tr_b16 v[122:123], v247 offset:20480
	ds_read_b64_tr_b16 v[124:125], v247 offset:22528
	ds_read_b64_tr_b16 v[118:119], v247 offset:24576
	ds_read_b64_tr_b16 v[120:121], v247 offset:26624
	ds_read_b64_tr_b16 v[114:115], v247 offset:28672
	ds_read_b64_tr_b16 v[116:117], v247 offset:30720
	ds_read_b64_tr_b16 v[110:111], v1 offset:16384
	ds_read_b64_tr_b16 v[112:113], v1 offset:18432
	ds_read_b64_tr_b16 v[106:107], v1 offset:20480
	ds_read_b64_tr_b16 v[108:109], v1 offset:22528
	ds_read_b64_tr_b16 v[102:103], v1 offset:24576
	ds_read_b64_tr_b16 v[104:105], v1 offset:26624
	ds_read_b64_tr_b16 v[98:99], v1 offset:28672
	ds_read_b64_tr_b16 v[100:101], v1 offset:30720
	ds_read_b64_tr_b16 v[94:95], v246 offset:16384
	ds_read_b64_tr_b16 v[96:97], v246 offset:18432
	ds_read_b64_tr_b16 v[90:91], v246 offset:20480
	ds_read_b64_tr_b16 v[92:93], v246 offset:22528
	ds_read_b64_tr_b16 v[86:87], v246 offset:24576
	ds_read_b64_tr_b16 v[88:89], v246 offset:26624
	ds_read_b64_tr_b16 v[82:83], v246 offset:28672
	ds_read_b64_tr_b16 v[84:85], v246 offset:30720
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[144:147], v[78:81], v[50:65]
	v_fmac_f32_e32 v0, v184, v142
	v_mfma_f32_32x32x16_f16 v[34:49], v[126:129], v[78:81], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[110:113], v[78:81], v[18:33]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[2:17], v[94:97], v[78:81], v[2:17]
	v_mfma_f32_32x32x16_f16 v[50:65], v[138:141], v[74:77], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[122:125], v[74:77], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[106:109], v[74:77], v[18:33]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[2:17], v[90:93], v[74:77], v[2:17]
	v_mfma_f32_32x32x16_f16 v[50:65], v[134:137], v[70:73], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[118:121], v[70:73], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[102:105], v[70:73], v[18:33]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[86:89], v[70:73], v[2:17]
	v_mfma_f32_32x32x16_f16 v[50:65], v[130:133], v[66:69], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[114:117], v[66:69], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[98:101], v[66:69], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[66:69], v[2:17]
	scratch_load_dword v69, off, off offset:96 ; 4-byte Folded Reload
	s_barrier
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v69, 2, 0
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	scratch_load_dword v70, off, off offset:112 ; 4-byte Folded Reload
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v0
	v_mov_b32_e32 v67, 0x42000000
	v_or_b32_e32 v66, s34, v69
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v0, v68
	v_log_f32_e32 v68, v68
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v67, 0, v67, vcc
	v_cmp_gt_i32_e64 s[8:9], s2, v66
	v_sub_f32_e32 v66, v68, v67
	v_add_f32_e32 v66, v143, v66
	ds_write_b32 v1, v66
	v_mov_b32_e32 v66, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_sub_i32 s2, 0x4000, s34
	v_bfrev_b32_e32 v68, 1
	s_and_b32 s5, s16, 0xffff
	s_mov_b32 s6, s14
	s_mov_b32 s7, s15
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v66, v66, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v67, 0, v66
	ds_read_b32 v67, v67
	v_cmp_lt_i32_sdwa s[2:3], v70, s2 src0_sel:BYTE_0 src1_sel:DWORD
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v66, v68, v66, vcc
	v_mov_b32_e32 v68, v70
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v66, s[4:7], 0 offen
	s_cbranch_execz .LBB0_5
	s_branch .LBB0_6
.LBB0_4:
	scratch_load_dword v68, off, off offset:112 ; 4-byte Folded Reload
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_5:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v0
	v_mov_b32_e32 v66, 0x42000000
	s_and_b32 s5, s16, 0xffff
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v0, v67
	v_log_f32_e32 v67, v67
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_sub_f32_e32 v66, v67, v66
	v_add_f32_e32 v66, v143, v66
	ds_write_b32 v1, v66
	v_mov_b32_e32 v1, 2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v1, v1, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v66, v66
	v_bfrev_b32_e32 v67, 1
	v_cndmask_b32_e64 v1, v67, v1, s[0:1]
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v1, s[4:7], 0 offen
.LBB0_6:                                ; %.critedge
	v_div_scale_f32 v1, s[0:1], v0, v0, 1.0
	v_rcp_f32_e32 v1, v1
	v_div_scale_f32 v66, vcc, 1.0, v0, 1.0
	s_mul_i32 s0, s25, s18
	v_mul_f32_e32 v1, v66, v1
	s_ashr_i32 s1, s0, 31
	s_nop 0
	v_div_fmas_f32 v1, 0, 0, v1
	v_div_fixup_f32 v66, v1, v0, 1.0
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
	v_pk_mul_f32 v[4:5], v[66:67], v[48:49] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v21, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[46:47] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v20, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[44:45] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v19, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[42:43] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v18, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[40:41] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v25, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[38:39] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v24, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[36:37] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v23, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[34:35] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v22, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[64:65] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v29, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[62:63] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v28, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[60:61] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v27, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[58:59] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v26, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[56:57] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v33, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[54:55] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v32, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[52:53] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v31, v4, v5
	v_pk_mul_f32 v[4:5], v[66:67], v[50:51] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v30, v4, v5
	scratch_load_dword v4, off, off offset:116 ; 4-byte Folded Reload
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s26, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s27, s34
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s27, 0x3fff
	v_mul_lo_u32 v5, s27, v69
	s_bitset1_b32 s2, 14
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_permlane32_swap_b32_e32 v30, v32
	v_permlane32_swap_b32_e32 v31, v33
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
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
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v4, 2, v4
	v_add_lshl_u32 v4, v5, v4, 1
	v_bfrev_b32_e32 v5, 1
	v_cndmask_b32_e64 v34, v5, v4, s[8:9]
	buffer_store_dwordx4 v[30:33], v34, s[0:3], 0 offen
	s_nop 1
	v_add_u32_e32 v30, 32, v4
	v_cndmask_b32_e64 v30, v5, v30, s[8:9]
	buffer_store_dwordx4 v[26:29], v30, s[0:3], 0 offen
	s_nop 1
	v_add_u32_e32 v26, 64, v4
	v_cndmask_b32_e64 v26, v5, v26, s[8:9]
	buffer_store_dwordx4 v[22:25], v26, s[0:3], 0 offen
	s_nop 1
	v_add_u32_e32 v22, 0x60, v4
	v_cndmask_b32_e64 v22, v5, v22, s[8:9]
	buffer_store_dwordx4 v[18:21], v22, s[0:3], 0 offen
	s_nop 1
	v_add_u32_e32 v18, 0x80, v4
	v_cndmask_b32_e64 v18, v5, v18, s[8:9]
	buffer_store_dwordx4 v[10:13], v18, s[0:3], 0 offen
	s_nop 1
	v_add_u32_e32 v10, 0xa0, v4
	v_cndmask_b32_e64 v10, v5, v10, s[8:9]
	buffer_store_dwordx4 v[0:3], v10, s[0:3], 0 offen
	s_nop 1
	v_add_u32_e32 v0, 0xc0, v4
	v_cndmask_b32_e64 v0, v5, v0, s[8:9]
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xe0, v4
	v_cndmask_b32_e64 v0, v5, v0, s[8:9]
	buffer_store_dwordx4 v[14:17], v0, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 160
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
		.amdhsa_next_free_sgpr 56
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
	.set attn_fwd.numbered_sgpr, 56
	.set attn_fwd.private_seg_size, 160
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12128
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 160
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 62
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
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .max_flat_workgroup_size: 512
    .name:           attn_fwd
    .private_segment_fixed_size: 160
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 40
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
