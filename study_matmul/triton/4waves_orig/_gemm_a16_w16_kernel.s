	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0 ; -- Begin function _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
	.p2align	8
	.type	_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0,@function
_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0: ; @_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.10:
	.file	1 "/var/lib/jenkins/aiter/aiter/ops/triton/_triton_kernels" "gemm_a16w16.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.11:
.LBB0_0:
	s_mov_b32 s36, s11
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_addk_i32 s11, 0xff
	s_ashr_i32 s15, s11, 31
	s_lshr_b32 s15, s15, 24
	s_add_i32 s11, s11, s15
	s_ashr_i32 s15, s11, 8
	.file	3 "/var/lib/jenkins/aiter/aiter/ops/triton/utils/_triton" "pid_preprocessing.py"
	s_lshl_b32 s15, s15, 4
	s_abs_i32 s17, s15
	v_cvt_f32_u32_e32 v1, s17
	s_mov_b64 s[20:21], s[6:7]
	s_ashr_i32 s6, s16, 31
	s_lshr_b32 s6, s6, 29
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s6, s16, s6
	s_ashr_i32 s6, s6, 3
	s_lshl_b32 s7, s16, 5
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mulk_i32 s6, 0xff01
	s_add_i32 s6, s6, s7
	s_add_i32 s7, s10, 0xff
	s_sub_i32 s18, 0, s17
	v_readfirstlane_b32 s19, v1
	s_ashr_i32 s16, s7, 31
	s_mul_i32 s18, s18, s19
	s_lshr_b32 s16, s16, 24
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s7, s7, s16
	s_abs_i32 s16, s6
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s16, s19
	s_mul_i32 s19, s18, s17
	s_ashr_i32 s22, s7, 8
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s11, s11, 31
	s_sub_i32 s19, s16, s19
	s_xor_b32 s11, s7, s11
	s_add_i32 s23, s18, 1
	s_sub_i32 s24, s19, s17
	s_cmp_ge_u32 s19, s17
	s_cselect_b32 s18, s23, s18
	s_cselect_b32 s19, s24, s19
	s_add_i32 s23, s18, 1
	s_cmp_ge_u32 s19, s17
	s_cselect_b32 s17, s23, s18
	s_xor_b32 s17, s17, s11
	s_sub_i32 s19, s17, s11
	s_lshl_b32 s18, s19, 4
	s_sub_i32 s11, s22, s18
	s_min_i32 s22, s11, 16
	s_abs_i32 s17, s22
	v_cvt_f32_u32_e32 v1, s17
	v_readfirstlane_b32 s11, v0
	s_cmp_lt_i32 s12, 1
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s23, v1
	s_cbranch_scc1 .LBB0_9
; %bb.1:
	s_sub_i32 s24, 0, s17
	s_mul_i32 s24, s24, s23
	s_mul_hi_u32 s24, s23, s24
	s_add_i32 s23, s23, s24
	s_mul_hi_u32 s24, s16, s23
	s_mul_i32 s19, s19, s15
	s_sub_i32 s6, s6, s19
	s_mul_i32 s24, s24, s17
	s_ashr_i32 s15, s6, 31
	s_abs_i32 s6, s6
	s_sub_i32 s16, s16, s24
	s_ashr_i32 s22, s22, 31
	s_mul_hi_u32 s19, s6, s23
	s_sub_i32 s23, s16, s17
	s_cmp_ge_u32 s16, s17
	s_cselect_b32 s16, s23, s16
	s_sub_i32 s23, s16, s17
	s_cmp_ge_u32 s16, s17
	s_cselect_b32 s16, s23, s16
	s_xor_b32 s16, s16, s7
	s_sub_i32 s7, s16, s7
	s_add_i32 s16, s18, s7
	s_xor_b32 s7, s15, s22
	s_mul_i32 s15, s19, s17
	s_sub_i32 s6, s6, s15
	s_add_i32 s15, s19, 1
	s_sub_i32 s18, s6, s17
	s_cmp_ge_u32 s6, s17
	s_cselect_b32 s15, s15, s19
	s_cselect_b32 s6, s18, s6
	s_add_i32 s18, s15, 1
	s_cmp_ge_u32 s6, s17
	s_cselect_b32 s6, s18, s15
	s_abs_i32 s15, s36
	v_cvt_f32_u32_e32 v1, s15
	s_xor_b32 s6, s6, s7
	s_bfe_u32 s24, s11, 0x20006
	s_sub_i32 s18, s6, s7
	v_rcp_iflag_f32_e32 v1, v1
	v_and_b32_e32 v22, 63, v0
	s_lshl_b32 s6, s24, 6
	v_or_b32_e32 v2, s6, v22
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_lshl_b32 s17, s18, 8
	s_sub_i32 s7, 0, s15
	v_or_b32_e32 v2, s17, v2
	v_mul_lo_u32 v3, s7, v1
	s_bfe_i32 s7, s18, 0x10017
	v_mul_hi_u32 v3, v1, v3
	v_add_u32_e32 v2, s7, v2
	v_add_u32_e32 v1, v1, v3
	v_xor_b32_e32 v2, s7, v2
	v_mul_hi_u32 v3, v2, v1
	v_mul_lo_u32 v3, v3, s15
	v_sub_u32_e32 v2, v2, v3
	v_subrev_u32_e32 v3, s15, v2
	v_cmp_le_u32_e32 vcc, s15, v2
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s23, 0x27000
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s15, v2
	v_cmp_le_u32_e32 vcc, s15, v2
	s_mov_b32 s22, 0x7ffffffe
	v_lshlrev_b32_e32 v4, 3, v0
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_xor_b32_e32 v2, s7, v2
	v_subrev_u32_e32 v2, s7, v2
	v_lshlrev_b32_e32 v2, 1, v2
	buffer_load_ushort v37, v2, s[20:23], 0 offen
	v_lshlrev_b32_e32 v5, 2, v0
	v_and_b32_e32 v10, 56, v4
	v_and_b32_e32 v4, 0xe0, v4
	v_and_b32_e32 v5, 12, v5
	s_abs_i32 s21, s10
	v_add3_u32 v4, 0, v4, v5
	v_cvt_f32_u32_e32 v5, s21
	v_lshlrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v2, 0x70, v2
	s_and_b32 s19, s11, 64
	v_rcp_iflag_f32_e32 v5, v5
	s_lshl_b32 s25, s16, 8
	s_bfe_i32 s20, s16, 0x10017
	v_or_b32_e32 v2, s24, v2
	s_sub_i32 s28, 0, s21
	s_lshr_b32 s29, s19, 5
	v_mov_b32_e32 v3, 0x70
	v_or_b32_e32 v7, 4, v2
	v_or_b32_e32 v8, 8, v2
	v_or_b32_e32 v9, 12, v2
	v_or_b32_e32 v11, 0x80, v2
	v_or_b32_e32 v12, 0x84, v2
	v_or_b32_e32 v13, 0x88, v2
	v_or_b32_e32 v14, 0x8c, v2
	s_cmp_lt_u32 s24, 2
	v_mul_f32_e32 v5, 0x4f7ffffe, v5
	s_mul_i32 s11, s24, 0x420
	v_bitop3_b32 v38, s6, v3, v22 bitop3:0xc8
	v_or_b32_e32 v3, s25, v2
	v_or_b32_e32 v15, s25, v7
	v_or_b32_e32 v16, s25, v8
	v_or_b32_e32 v17, s25, v9
	v_or_b32_e32 v18, s25, v11
	v_or_b32_e32 v19, s25, v12
	v_or_b32_e32 v20, s25, v13
	v_or_b32_e32 v21, s25, v14
	s_cselect_b64 s[24:25], -1, 0
	v_cvt_u32_f32_e32 v5, v5
	s_and_b64 s[26:27], s[24:25], exec
	v_lshrrev_b32_e32 v6, 1, v0
	s_cselect_b32 s26, 0, 0x100
	v_and_b32_e32 v6, 16, v6
	s_or_b32 s26, s26, s29
	v_add3_u32 v4, v4, v6, s26
	v_mul_lo_u32 v6, s28, v5
	v_add_u32_e32 v3, s20, v3
	v_or_b32_e32 v2, s17, v2
	v_or_b32_e32 v7, s17, v7
	v_or_b32_e32 v8, s17, v8
	v_or_b32_e32 v9, s17, v9
	v_or_b32_e32 v11, s17, v11
	v_or_b32_e32 v12, s17, v12
	v_or_b32_e32 v13, s17, v13
	v_or_b32_e32 v14, s17, v14
	v_mul_hi_u32 v6, v5, v6
	v_xor_b32_e32 v3, s20, v3
	v_add_u32_e32 v15, s20, v15
	v_add_u32_e32 v2, s7, v2
	v_add_u32_e32 v7, s7, v7
	v_add_u32_e32 v8, s7, v8
	v_add_u32_e32 v9, s7, v9
	v_add_u32_e32 v11, s7, v11
	v_add_u32_e32 v12, s7, v12
	v_add_u32_e32 v13, s7, v13
	v_add_u32_e32 v14, s7, v14
	v_add_u32_e32 v5, v5, v6
	v_add_u32_e32 v16, s20, v16
	v_add_u32_e32 v17, s20, v17
	v_add_u32_e32 v18, s20, v18
	v_add_u32_e32 v19, s20, v19
	v_add_u32_e32 v20, s20, v20
	v_xor_b32_e32 v15, s20, v15
	v_xor_b32_e32 v2, s7, v2
	v_xor_b32_e32 v7, s7, v7
	v_xor_b32_e32 v8, s7, v8
	v_xor_b32_e32 v9, s7, v9
	v_xor_b32_e32 v11, s7, v11
	v_xor_b32_e32 v12, s7, v12
	v_xor_b32_e32 v13, s7, v13
	v_xor_b32_e32 v14, s7, v14
	v_mul_hi_u32 v6, v3, v5
	v_add_u32_e32 v21, s20, v21
	v_xor_b32_e32 v16, s20, v16
	v_xor_b32_e32 v17, s20, v17
	v_xor_b32_e32 v18, s20, v18
	v_xor_b32_e32 v19, s20, v19
	v_xor_b32_e32 v20, s20, v20
	v_mul_hi_u32 v24, v15, v5
	v_mul_hi_u32 v30, v2, v1
	v_mul_hi_u32 v31, v7, v1
	v_mul_hi_u32 v32, v8, v1
	v_mul_hi_u32 v33, v9, v1
	v_mul_hi_u32 v34, v11, v1
	v_mul_hi_u32 v35, v12, v1
	v_mul_hi_u32 v36, v13, v1
	v_mul_hi_u32 v1, v14, v1
	v_mul_lo_u32 v6, v6, s21
	v_xor_b32_e32 v21, s20, v21
	v_mul_hi_u32 v25, v16, v5
	v_mul_hi_u32 v26, v17, v5
	v_mul_hi_u32 v27, v18, v5
	v_mul_hi_u32 v28, v19, v5
	v_mul_hi_u32 v29, v20, v5
	v_mul_lo_u32 v24, v24, s21
	v_mul_lo_u32 v1, v1, s15
	v_sub_u32_e32 v3, v3, v6
	v_mul_hi_u32 v5, v21, v5
	v_mul_lo_u32 v25, v25, s21
	v_mul_lo_u32 v26, v26, s21
	v_mul_lo_u32 v27, v27, s21
	v_mul_lo_u32 v28, v28, s21
	v_mul_lo_u32 v29, v29, s21
	v_sub_u32_e32 v6, v15, v24
	v_sub_u32_e32 v1, v14, v1
	v_subrev_u32_e32 v14, s21, v3
	v_cmp_le_u32_e32 vcc, s21, v3
	v_mul_lo_u32 v5, v5, s21
	v_sub_u32_e32 v15, v16, v25
	v_sub_u32_e32 v16, v17, v26
	v_sub_u32_e32 v17, v18, v27
	v_sub_u32_e32 v18, v19, v28
	v_sub_u32_e32 v19, v20, v29
	v_subrev_u32_e32 v20, s21, v6
	v_cndmask_b32_e32 v3, v3, v14, vcc
	v_cmp_le_u32_e32 vcc, s21, v6
	v_sub_u32_e32 v5, v21, v5
	v_subrev_u32_e32 v21, s21, v15
	v_cndmask_b32_e32 v6, v6, v20, vcc
	v_cmp_le_u32_e32 vcc, s21, v15
	v_subrev_u32_e32 v24, s21, v16
	v_subrev_u32_e32 v25, s21, v17
	v_cndmask_b32_e32 v14, v15, v21, vcc
	v_cmp_le_u32_e32 vcc, s21, v16
	v_subrev_u32_e32 v26, s21, v18
	v_mul_lo_u32 v30, v30, s15
	v_cndmask_b32_e32 v15, v16, v24, vcc
	v_cmp_le_u32_e32 vcc, s21, v17
	v_subrev_u32_e32 v27, s21, v19
	v_mul_lo_u32 v31, v31, s15
	v_cndmask_b32_e32 v16, v17, v25, vcc
	v_cmp_le_u32_e32 vcc, s21, v18
	v_sub_u32_e32 v2, v2, v30
	v_subrev_u32_e32 v28, s21, v5
	v_cndmask_b32_e32 v17, v18, v26, vcc
	v_cmp_le_u32_e32 vcc, s21, v19
	v_mul_lo_u32 v32, v32, s15
	v_sub_u32_e32 v7, v7, v31
	v_cndmask_b32_e32 v18, v19, v27, vcc
	v_cmp_le_u32_e32 vcc, s21, v5
	v_subrev_u32_e32 v29, s15, v2
	v_mul_lo_u32 v33, v33, s15
	v_cndmask_b32_e32 v5, v5, v28, vcc
	v_cmp_le_u32_e32 vcc, s15, v2
	v_sub_u32_e32 v8, v8, v32
	v_subrev_u32_e32 v30, s15, v7
	v_cndmask_b32_e32 v2, v2, v29, vcc
	v_cmp_le_u32_e32 vcc, s15, v7
	v_mul_lo_u32 v34, v34, s15
	v_sub_u32_e32 v9, v9, v33
	v_subrev_u32_e32 v31, s15, v8
	v_cndmask_b32_e32 v7, v7, v30, vcc
	v_cmp_le_u32_e32 vcc, s15, v8
	v_mul_lo_u32 v35, v35, s15
	v_sub_u32_e32 v11, v11, v34
	v_subrev_u32_e32 v32, s15, v9
	v_cndmask_b32_e32 v8, v8, v31, vcc
	v_cmp_le_u32_e32 vcc, s15, v9
	v_mul_lo_u32 v36, v36, s15
	v_sub_u32_e32 v12, v12, v35
	v_subrev_u32_e32 v33, s15, v11
	v_cndmask_b32_e32 v9, v9, v32, vcc
	v_cmp_le_u32_e32 vcc, s15, v11
	v_sub_u32_e32 v13, v13, v36
	v_subrev_u32_e32 v34, s15, v12
	v_cndmask_b32_e32 v11, v11, v33, vcc
	v_cmp_le_u32_e32 vcc, s15, v12
	v_subrev_u32_e32 v35, s15, v13
	v_subrev_u32_e32 v36, s15, v1
	v_cndmask_b32_e32 v12, v12, v34, vcc
	v_cmp_le_u32_e32 vcc, s15, v13
	v_subrev_u32_e32 v19, s21, v3
	v_subrev_u32_e32 v20, s21, v6
	v_cndmask_b32_e32 v13, v13, v35, vcc
	v_cmp_le_u32_e32 vcc, s15, v1
	v_subrev_u32_e32 v21, s21, v14
	v_subrev_u32_e32 v24, s21, v15
	v_cndmask_b32_e32 v1, v1, v36, vcc
	v_cmp_le_u32_e32 vcc, s21, v3
	v_subrev_u32_e32 v25, s21, v16
	v_subrev_u32_e32 v26, s21, v17
	v_cndmask_b32_e32 v3, v3, v19, vcc
	v_cmp_le_u32_e32 vcc, s21, v6
	v_subrev_u32_e32 v27, s21, v18
	v_subrev_u32_e32 v28, s21, v5
	v_cndmask_b32_e32 v6, v6, v20, vcc
	v_cmp_le_u32_e32 vcc, s21, v14
	v_subrev_u32_e32 v29, s15, v2
	v_subrev_u32_e32 v30, s15, v7
	v_cndmask_b32_e32 v14, v14, v21, vcc
	v_cmp_le_u32_e32 vcc, s21, v15
	v_subrev_u32_e32 v31, s15, v8
	v_subrev_u32_e32 v32, s15, v9
	v_cndmask_b32_e32 v15, v15, v24, vcc
	v_cmp_le_u32_e32 vcc, s21, v16
	v_subrev_u32_e32 v33, s15, v11
	v_subrev_u32_e32 v34, s15, v12
	v_cndmask_b32_e32 v16, v16, v25, vcc
	v_cmp_le_u32_e32 vcc, s21, v17
	v_subrev_u32_e32 v35, s15, v13
	v_subrev_u32_e32 v36, s15, v1
	v_cndmask_b32_e32 v17, v17, v26, vcc
	v_cmp_le_u32_e32 vcc, s21, v18
	v_xor_b32_e32 v3, s20, v3
	v_xor_b32_e32 v6, s20, v6
	v_cndmask_b32_e32 v18, v18, v27, vcc
	v_cmp_le_u32_e32 vcc, s21, v5
	v_xor_b32_e32 v14, s20, v14
	v_xor_b32_e32 v15, s20, v15
	v_cndmask_b32_e32 v5, v5, v28, vcc
	v_cmp_le_u32_e32 vcc, s15, v2
	v_xor_b32_e32 v16, s20, v16
	v_xor_b32_e32 v17, s20, v17
	v_cndmask_b32_e32 v2, v2, v29, vcc
	v_cmp_le_u32_e32 vcc, s15, v7
	v_subrev_u32_e32 v3, s20, v3
	s_add_i32 s17, s11, 0
	v_cndmask_b32_e32 v7, v7, v30, vcc
	v_cmp_le_u32_e32 vcc, s15, v8
	v_xor_b32_e32 v7, s7, v7
	v_xor_b32_e32 v18, s20, v18
	v_cndmask_b32_e32 v8, v8, v31, vcc
	v_cmp_le_u32_e32 vcc, s15, v9
	v_xor_b32_e32 v8, s7, v8
	v_xor_b32_e32 v5, s20, v5
	v_cndmask_b32_e32 v9, v9, v32, vcc
	v_cmp_le_u32_e32 vcc, s15, v11
	v_xor_b32_e32 v9, s7, v9
	v_subrev_u32_e32 v6, s20, v6
	v_cndmask_b32_e32 v11, v11, v33, vcc
	v_cmp_le_u32_e32 vcc, s15, v12
	v_subrev_u32_e32 v14, s20, v14
	v_subrev_u32_e32 v15, s20, v15
	v_cndmask_b32_e32 v12, v12, v34, vcc
	v_cmp_le_u32_e32 vcc, s15, v13
	v_subrev_u32_e32 v16, s20, v16
	v_subrev_u32_e32 v17, s20, v17
	v_cndmask_b32_e32 v13, v13, v35, vcc
	v_cmp_le_u32_e32 vcc, s15, v1
	v_subrev_u32_e32 v7, s7, v7
	v_subrev_u32_e32 v8, s7, v8
	v_cndmask_b32_e32 v1, v1, v36, vcc
	v_xor_b32_e32 v1, s7, v1
	v_subrev_u32_e32 v9, s7, v9
	v_subrev_u32_e32 v1, s7, v1
	v_mul_lo_u32 v3, v3, s13
	v_lshl_add_u32 v23, v38, 1, 0
	v_subrev_u32_e32 v18, s20, v18
	v_subrev_u32_e32 v5, s20, v5
	v_mul_lo_u32 v6, v6, s13
	v_mul_lo_u32 v14, v14, s13
	v_mul_lo_u32 v15, v15, s13
	v_mul_lo_u32 v16, v16, s13
	v_mul_lo_u32 v17, v17, s13
	v_mul_lo_u32 v7, v7, s14
	v_mul_lo_u32 v8, v8, s14
	v_mul_lo_u32 v9, v9, s14
	v_mul_lo_u32 v19, v1, s14
	s_and_b32 s21, s3, 0xffff
	s_mov_b32 s20, s2
	v_add_lshl_u32 v1, v3, v10, 1
	s_mov_b32 m0, s17
	v_add_lshl_u32 v40, v6, v10, 1
	v_add_lshl_u32 v41, v14, v10, 1
	v_add_lshl_u32 v46, v15, v10, 1
	v_add_lshl_u32 v47, v16, v10, 1
	v_add_lshl_u32 v48, v17, v10, 1
	v_add_lshl_u32 v34, v7, v10, 1
	v_add_lshl_u32 v35, v8, v10, 1
	v_add_lshl_u32 v36, v9, v10, 1
	s_waitcnt vmcnt(0)
	ds_write_b16 v4, v37
	v_xor_b32_e32 v2, s7, v2
	v_mul_lo_u32 v18, v18, s13
	v_subrev_u32_e32 v2, s7, v2
	v_mul_lo_u32 v5, v5, s13
	v_add_lshl_u32 v49, v18, v10, 1
	v_mul_lo_u32 v2, v2, s14
	v_add_lshl_u32 v222, v5, v10, 1
	v_add_lshl_u32 v25, v2, v10, 1
	v_xor_b32_e32 v11, s7, v11
	v_xor_b32_e32 v12, s7, v12
	v_subrev_u32_e32 v11, s7, v11
	v_xor_b32_e32 v13, s7, v13
	v_subrev_u32_e32 v12, s7, v12
	v_mul_lo_u32 v11, v11, s14
	v_subrev_u32_e32 v13, s7, v13
	v_mul_lo_u32 v12, v12, s14
	v_add_lshl_u32 v42, v11, v10, 1
	v_mul_lo_u32 v13, v13, s14
	v_add_lshl_u32 v43, v12, v10, 1
	v_add_lshl_u32 v44, v13, v10, 1
	v_add_lshl_u32 v37, v19, v10, 1
	v_and_b32_e32 v24, 15, v0
	v_and_b32_e32 v39, 48, v0
	v_lshlrev_b32_e32 v254, 10, v24
	v_lshlrev_b32_e32 v255, 5, v24
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[6:9], v23
	ds_read_b128 v[14:17], v23 offset:16
	ds_read_b128 v[26:29], v23 offset:256
	ds_read_b128 v[30:33], v23 offset:272
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x1080
	v_lshlrev_b32_e32 v55, 16, v7
	buffer_load_dwordx4 v40, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x2100
	v_lshlrev_b32_e32 v54, 16, v6
	buffer_load_dwordx4 v41, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x3180
	v_lshlrev_b32_e32 v57, 16, v9
	buffer_load_dwordx4 v46, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x4200
	v_lshlrev_b32_e32 v56, 16, v8
	buffer_load_dwordx4 v47, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x5280
	v_lshlrev_b32_e32 v3, 16, v15
	buffer_load_dwordx4 v48, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x6300
	v_lshlrev_b32_e32 v2, 16, v14
	buffer_load_dwordx4 v49, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x7380
	v_lshlrev_b32_e32 v5, 16, v17
	buffer_load_dwordx4 v222, s[20:23], 0 offen lds
	s_and_b32 s21, s5, 0xffff
	s_mov_b32 s20, s4
	s_add_i32 m0, s17, 0x107e0
	v_lshlrev_b32_e32 v4, 16, v16
	buffer_load_dwordx4 v25, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x11860
	v_and_b32_e32 v11, 0xffff0000, v7
	buffer_load_dwordx4 v34, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x128e0
	v_and_b32_e32 v10, 0xffff0000, v6
	buffer_load_dwordx4 v35, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x13960
	v_and_b32_e32 v13, 0xffff0000, v9
	buffer_load_dwordx4 v36, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x149e0
	v_and_b32_e32 v12, 0xffff0000, v8
	buffer_load_dwordx4 v42, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x15a60
	v_and_b32_e32 v7, 0xffff0000, v15
	buffer_load_dwordx4 v43, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x16ae0
	v_and_b32_e32 v6, 0xffff0000, v14
	buffer_load_dwordx4 v44, s[20:23], 0 offen lds
	s_add_i32 m0, s17, 0x17b60
	s_add_u32 s13, s2, 0x80
	buffer_load_dwordx4 v37, s[20:23], 0 offen lds
	s_addc_u32 s14, s3, 0
	s_add_u32 s15, s4, 0x80
	s_addc_u32 s17, s5, 0
	v_and_b32_e32 v9, 0xffff0000, v17
	v_and_b32_e32 v8, 0xffff0000, v16
	v_lshlrev_b32_e32 v19, 16, v27
	v_lshlrev_b32_e32 v18, 16, v26
	v_lshlrev_b32_e32 v21, 16, v29
	v_lshlrev_b32_e32 v20, 16, v28
	v_lshlrev_b32_e32 v15, 16, v31
	v_lshlrev_b32_e32 v14, 16, v30
	v_lshlrev_b32_e32 v17, 16, v33
	v_lshlrev_b32_e32 v16, 16, v32
	v_and_b32_e32 v27, 0xffff0000, v27
	v_and_b32_e32 v26, 0xffff0000, v26
	v_and_b32_e32 v29, 0xffff0000, v29
	v_and_b32_e32 v28, 0xffff0000, v28
	v_and_b32_e32 v31, 0xffff0000, v31
	v_and_b32_e32 v30, 0xffff0000, v30
	v_and_b32_e32 v33, 0xffff0000, v33
	v_and_b32_e32 v32, 0xffff0000, v32
	s_cmp_gt_u32 s12, 64
	v_mov_b32_e32 v23, 0xb0
	s_cbranch_scc1 .LBB0_4
; %bb.2:                                ; %.._crit_edge_crit_edge
	s_lshl_b32 s4, s19, 1
	v_mov_b32_e32 v0, s4
	v_bitop3_b32 v22, s6, v23, v22 bitop3:0xc8
	s_cbranch_execz .LBB0_5
; %bb.3:
	v_mov_b32_e32 v37, v33
	v_mov_b32_e32 v36, v32
	v_mov_b32_e32 v35, v31
	v_mov_b32_e32 v34, v30
	v_accvgpr_write_b32 a139, v37
	v_accvgpr_write_b32 a138, v36
	v_accvgpr_write_b32 a137, v35
	v_accvgpr_write_b32 a136, v34
	v_mov_b32_e32 v37, v17
	v_mov_b32_e32 v36, v16
	v_mov_b32_e32 v35, v15
	v_mov_b32_e32 v34, v14
	v_accvgpr_write_b32 a135, v37
	v_accvgpr_write_b32 a134, v36
	v_accvgpr_write_b32 a133, v35
	v_accvgpr_write_b32 a132, v34
	v_mov_b32_e32 v37, v21
	v_mov_b32_e32 v36, v20
	v_mov_b32_e32 v35, v19
	v_mov_b32_e32 v34, v18
	v_accvgpr_write_b32 a155, v37
	v_accvgpr_write_b32 a154, v36
	v_accvgpr_write_b32 a153, v35
	v_accvgpr_write_b32 a152, v34
	v_mov_b32_e32 v37, v9
	v_mov_b32_e32 v36, v8
	v_mov_b32_e32 v35, v7
	v_mov_b32_e32 v34, v6
	v_accvgpr_write_b32 a147, v37
	v_accvgpr_write_b32 a146, v36
	v_accvgpr_write_b32 a145, v35
	v_accvgpr_write_b32 a144, v34
	v_mov_b32_e32 v37, v13
	v_mov_b32_e32 v36, v12
	v_mov_b32_e32 v35, v11
	v_mov_b32_e32 v34, v10
	v_accvgpr_write_b32 a159, v37
	v_accvgpr_write_b32 a158, v36
	v_accvgpr_write_b32 a157, v35
	v_accvgpr_write_b32 a156, v34
	v_mov_b32_e32 v37, v5
	v_mov_b32_e32 v36, v4
	v_mov_b32_e32 v35, v3
	v_mov_b32_e32 v34, v2
	v_accvgpr_write_b32 a143, v37
	v_accvgpr_write_b32 a142, v36
	v_accvgpr_write_b32 a141, v35
	v_accvgpr_write_b32 a140, v34
	v_mov_b32_e32 v37, v57
	v_mov_b32_e32 v36, v56
	v_mov_b32_e32 v35, v55
	v_mov_b32_e32 v34, v54
	v_accvgpr_write_b32 a151, v37
	v_accvgpr_write_b32 a150, v36
	v_accvgpr_write_b32 a149, v35
	v_accvgpr_write_b32 a148, v34
	v_mov_b32_e32 v37, v13
	v_mov_b32_e32 v36, v12
	v_mov_b32_e32 v35, v11
	v_mov_b32_e32 v34, v10
	v_accvgpr_write_b32 a117, v37
	v_accvgpr_write_b32 a116, v36
	v_accvgpr_write_b32 a115, v35
	v_accvgpr_write_b32 a114, v34
	v_mov_b32_e32 v37, v5
	v_mov_b32_e32 v36, v4
	v_mov_b32_e32 v35, v3
	v_mov_b32_e32 v34, v2
	v_accvgpr_write_b32 a121, v37
	v_accvgpr_write_b32 a120, v36
	v_accvgpr_write_b32 a119, v35
	v_accvgpr_write_b32 a118, v34
	v_mov_b32_e32 v37, v17
	v_mov_b32_e32 v36, v16
	v_mov_b32_e32 v35, v15
	v_mov_b32_e32 v34, v14
	v_accvgpr_write_b32 a125, v37
	v_accvgpr_write_b32 a124, v36
	v_accvgpr_write_b32 a123, v35
	v_accvgpr_write_b32 a122, v34
	v_mov_b32_e32 v37, v9
	v_mov_b32_e32 v36, v8
	v_mov_b32_e32 v35, v7
	v_mov_b32_e32 v34, v6
	v_accvgpr_write_b32 a129, v37
	s_mov_b32 s2, 0
	s_add_i32 s3, 0, 0x107e0
	v_accvgpr_write_b32 a167, v29
	v_accvgpr_write_b32 a166, v28
	v_accvgpr_write_b32 a165, v27
	v_accvgpr_write_b32 a164, v26
	v_accvgpr_write_b32 a103, v33
	v_accvgpr_write_b32 a102, v32
	v_accvgpr_write_b32 a101, v31
	v_accvgpr_write_b32 a100, v30
	v_accvgpr_write_b32 a171, v29
	v_accvgpr_write_b32 a170, v28
	v_accvgpr_write_b32 a169, v27
	v_accvgpr_write_b32 a168, v26
	v_accvgpr_write_b32 a107, v17
	v_accvgpr_write_b32 a106, v16
	v_accvgpr_write_b32 a105, v15
	v_accvgpr_write_b32 a104, v14
	v_mov_b32_e32 v69, v21
	v_mov_b32_e32 v68, v20
	v_mov_b32_e32 v67, v19
	v_mov_b32_e32 v66, v18
	v_accvgpr_write_b32 a99, v9
	v_accvgpr_write_b32 a98, v8
	v_accvgpr_write_b32 a97, v7
	v_accvgpr_write_b32 a96, v6
	v_mov_b32_e32 v53, v5
	v_mov_b32_e32 v52, v4
	v_mov_b32_e32 v51, v3
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v89, v57
	v_mov_b32_e32 v88, v56
	v_mov_b32_e32 v87, v55
	v_mov_b32_e32 v86, v54
	v_mov_b32_e32 v61, v33
	v_mov_b32_e32 v60, v32
	v_mov_b32_e32 v59, v31
	v_mov_b32_e32 v58, v30
	v_mov_b32_e32 v85, v29
	v_mov_b32_e32 v84, v28
	v_mov_b32_e32 v83, v27
	v_mov_b32_e32 v82, v26
	v_accvgpr_write_b32 a111, v17
	v_accvgpr_write_b32 a110, v16
	v_accvgpr_write_b32 a109, v15
	v_accvgpr_write_b32 a108, v14
	v_mov_b32_e32 v93, v21
	v_mov_b32_e32 v92, v20
	v_mov_b32_e32 v91, v19
	v_mov_b32_e32 v90, v18
	v_mov_b32_e32 v77, v9
	v_mov_b32_e32 v76, v8
	v_mov_b32_e32 v75, v7
	v_mov_b32_e32 v74, v6
	v_mov_b32_e32 v105, v13
	v_mov_b32_e32 v104, v12
	v_mov_b32_e32 v103, v11
	v_mov_b32_e32 v102, v10
	v_mov_b32_e32 v125, v57
	v_mov_b32_e32 v124, v56
	v_mov_b32_e32 v123, v55
	v_mov_b32_e32 v122, v54
	v_mov_b32_e32 v81, v33
	v_mov_b32_e32 v80, v32
	v_mov_b32_e32 v79, v31
	v_mov_b32_e32 v78, v30
	v_mov_b32_e32 v113, v29
	v_mov_b32_e32 v112, v28
	v_mov_b32_e32 v111, v27
	v_mov_b32_e32 v110, v26
	v_mov_b32_e32 v101, v17
	v_mov_b32_e32 v100, v16
	v_mov_b32_e32 v99, v15
	v_mov_b32_e32 v98, v14
	v_mov_b32_e32 v129, v21
	v_mov_b32_e32 v128, v20
	v_mov_b32_e32 v127, v19
	v_mov_b32_e32 v126, v18
	v_mov_b32_e32 v97, v9
	v_mov_b32_e32 v96, v8
	v_mov_b32_e32 v95, v7
	v_mov_b32_e32 v94, v6
	v_mov_b32_e32 v137, v13
	v_mov_b32_e32 v136, v12
	v_mov_b32_e32 v135, v11
	v_mov_b32_e32 v134, v10
	v_mov_b32_e32 v121, v5
	v_mov_b32_e32 v120, v4
	v_mov_b32_e32 v119, v3
	v_mov_b32_e32 v118, v2
	v_mov_b32_e32 v161, v57
	v_mov_b32_e32 v160, v56
	v_mov_b32_e32 v159, v55
	v_mov_b32_e32 v158, v54
	v_mov_b32_e32 v117, v33
	v_mov_b32_e32 v116, v32
	v_mov_b32_e32 v115, v31
	v_mov_b32_e32 v114, v30
	v_mov_b32_e32 v149, v29
	v_mov_b32_e32 v148, v28
	v_mov_b32_e32 v147, v27
	v_mov_b32_e32 v146, v26
	v_mov_b32_e32 v157, v21
	v_mov_b32_e32 v156, v20
	v_mov_b32_e32 v155, v19
	v_mov_b32_e32 v154, v18
	v_mov_b32_e32 v141, v9
	v_mov_b32_e32 v140, v8
	v_mov_b32_e32 v139, v7
	v_mov_b32_e32 v138, v6
	v_mov_b32_e32 v169, v13
	v_mov_b32_e32 v168, v12
	v_mov_b32_e32 v167, v11
	v_mov_b32_e32 v166, v10
	v_mov_b32_e32 v133, v5
	v_mov_b32_e32 v132, v4
	v_mov_b32_e32 v131, v3
	v_mov_b32_e32 v130, v2
	v_mov_b32_e32 v193, v57
	v_mov_b32_e32 v192, v56
	v_mov_b32_e32 v191, v55
	v_mov_b32_e32 v190, v54
	v_mov_b32_e32 v145, v33
	v_mov_b32_e32 v144, v32
	v_mov_b32_e32 v143, v31
	v_mov_b32_e32 v142, v30
	v_mov_b32_e32 v173, v29
	v_mov_b32_e32 v172, v28
	v_mov_b32_e32 v171, v27
	v_mov_b32_e32 v170, v26
	v_mov_b32_e32 v165, v17
	v_mov_b32_e32 v164, v16
	v_mov_b32_e32 v163, v15
	v_mov_b32_e32 v162, v14
	v_mov_b32_e32 v185, v21
	v_mov_b32_e32 v184, v20
	v_mov_b32_e32 v183, v19
	v_mov_b32_e32 v182, v18
	v_accvgpr_write_b32 a128, v36
	v_accvgpr_write_b32 a127, v35
	v_accvgpr_write_b32 a126, v34
	v_mov_b32_e32 v197, v13
	v_mov_b32_e32 v196, v12
	v_mov_b32_e32 v195, v11
	v_mov_b32_e32 v194, v10
	v_mov_b32_e32 v189, v5
	v_mov_b32_e32 v188, v4
	v_mov_b32_e32 v187, v3
	v_mov_b32_e32 v186, v2
	v_mov_b32_e32 v217, v57
	v_mov_b32_e32 v216, v56
	v_mov_b32_e32 v215, v55
	v_mov_b32_e32 v214, v54
	v_mov_b32_e32 v181, v33
	v_mov_b32_e32 v180, v32
	v_mov_b32_e32 v179, v31
	v_mov_b32_e32 v178, v30
	v_mov_b32_e32 v205, v29
	v_mov_b32_e32 v204, v28
	v_mov_b32_e32 v203, v27
	v_mov_b32_e32 v202, v26
	v_mov_b32_e32 v177, v17
	v_mov_b32_e32 v176, v16
	v_mov_b32_e32 v175, v15
	v_mov_b32_e32 v174, v14
	v_mov_b32_e32 v209, v21
	v_mov_b32_e32 v208, v20
	v_mov_b32_e32 v207, v19
	v_mov_b32_e32 v206, v18
	v_mov_b32_e32 v201, v9
	v_mov_b32_e32 v200, v8
	v_mov_b32_e32 v199, v7
	v_mov_b32_e32 v198, v6
	v_mov_b32_e32 v213, v13
	v_mov_b32_e32 v212, v12
	v_mov_b32_e32 v211, v11
	v_mov_b32_e32 v210, v10
	v_mov_b32_e32 v221, v5
	v_mov_b32_e32 v220, v4
	v_mov_b32_e32 v219, v3
	v_mov_b32_e32 v218, v2
	v_mov_b32_e32 v37, v57
	v_mov_b32_e32 v36, v56
	v_mov_b32_e32 v35, v55
	v_mov_b32_e32 v34, v54
	s_branch .LBB0_8
.LBB0_4:
                                        ; implicit-def: $vgpr0
	v_bitop3_b32 v22, s6, v23, v22 bitop3:0xc8
.LBB0_5:                                ; %.lr.ph
	s_min_u32 s2, s12, 0x1000
	s_add_i32 s2, s2, 63
	v_mov_b32_e32 v53, v13
	v_mov_b32_e32 v81, v5
	v_mov_b32_e32 v133, v17
	v_mov_b32_e32 v174, v6
	s_lshr_b32 s4, s2, 6
	v_mov_b32_e32 v52, v12
	v_mov_b32_e32 v51, v11
	v_mov_b32_e32 v50, v10
	v_accvgpr_write_b32 a135, v53
	v_mov_b32_e32 v80, v4
	v_mov_b32_e32 v79, v3
	v_mov_b32_e32 v78, v2
	v_accvgpr_write_b32 a139, v81
	v_mov_b32_e32 v132, v16
	v_mov_b32_e32 v131, v15
	v_mov_b32_e32 v130, v14
	v_accvgpr_write_b32 a143, v133
	v_mov_b32_e32 v177, v9
	v_mov_b32_e32 v176, v8
	v_mov_b32_e32 v175, v7
	v_accvgpr_write_b32 a126, v174
	v_accvgpr_write_b32 a113, v24
	v_accvgpr_write_b32 a112, v38
	s_lshl_b32 s12, s19, 1
	s_mov_b32 s2, 0
	s_add_i32 s3, 0, 0x107e0
	s_add_i32 s19, s4, -1
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	s_mov_b32 s20, 0
	v_accvgpr_write_b32 a103, v33
	v_accvgpr_write_b32 a121, v44
	v_accvgpr_write_b32 a102, v32
	v_accvgpr_write_b32 a122, v43
	v_accvgpr_write_b32 a101, v31
	v_accvgpr_write_b32 a123, v42
	v_accvgpr_write_b32 a100, v30
	v_accvgpr_write_b32 a171, v29
	v_accvgpr_write_b32 a170, v28
	v_accvgpr_write_b32 a169, v27
	v_accvgpr_write_b32 a168, v26
	v_accvgpr_write_b32 a107, v17
	v_accvgpr_write_b32 a106, v16
	v_accvgpr_write_b32 a105, v15
	v_accvgpr_write_b32 a104, v14
	v_mov_b32_e32 v69, v21
	v_mov_b32_e32 v68, v20
	v_mov_b32_e32 v67, v19
	v_mov_b32_e32 v66, v18
	v_accvgpr_write_b32 a99, v9
	v_accvgpr_write_b32 a98, v8
	v_accvgpr_write_b32 a97, v7
	v_accvgpr_write_b32 a96, v6
	v_accvgpr_write_b32 a134, v52
	v_accvgpr_write_b32 a133, v51
	v_accvgpr_write_b32 a132, v50
	v_mov_b32_e32 v53, v5
	v_mov_b32_e32 v52, v4
	v_mov_b32_e32 v51, v3
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v89, v57
	v_mov_b32_e32 v88, v56
	v_mov_b32_e32 v87, v55
	v_mov_b32_e32 v86, v54
	v_mov_b32_e32 v61, v33
	v_mov_b32_e32 v60, v32
	v_mov_b32_e32 v59, v31
	v_mov_b32_e32 v58, v30
	v_mov_b32_e32 v85, v29
	v_mov_b32_e32 v84, v28
	v_mov_b32_e32 v83, v27
	v_mov_b32_e32 v82, v26
	v_accvgpr_write_b32 a111, v17
	v_accvgpr_write_b32 a110, v16
	v_accvgpr_write_b32 a109, v15
	v_accvgpr_write_b32 a108, v14
	v_mov_b32_e32 v93, v21
	v_mov_b32_e32 v92, v20
	v_mov_b32_e32 v91, v19
	v_mov_b32_e32 v90, v18
	v_mov_b32_e32 v77, v9
	v_mov_b32_e32 v76, v8
	v_mov_b32_e32 v75, v7
	v_mov_b32_e32 v74, v6
	v_mov_b32_e32 v105, v13
	v_mov_b32_e32 v104, v12
	v_mov_b32_e32 v103, v11
	v_mov_b32_e32 v102, v10
	v_accvgpr_write_b32 a138, v80
	v_accvgpr_write_b32 a137, v79
	v_accvgpr_write_b32 a136, v78
	v_mov_b32_e32 v125, v57
	v_mov_b32_e32 v124, v56
	v_mov_b32_e32 v123, v55
	v_mov_b32_e32 v122, v54
	v_mov_b32_e32 v81, v33
	v_mov_b32_e32 v80, v32
	v_mov_b32_e32 v79, v31
	v_mov_b32_e32 v78, v30
	v_mov_b32_e32 v113, v29
	v_mov_b32_e32 v112, v28
	v_mov_b32_e32 v111, v27
	v_mov_b32_e32 v110, v26
	v_mov_b32_e32 v101, v17
	v_mov_b32_e32 v100, v16
	v_mov_b32_e32 v99, v15
	v_mov_b32_e32 v98, v14
	v_mov_b32_e32 v129, v21
	v_mov_b32_e32 v128, v20
	v_mov_b32_e32 v127, v19
	v_mov_b32_e32 v126, v18
	v_mov_b32_e32 v97, v9
	v_mov_b32_e32 v96, v8
	v_mov_b32_e32 v95, v7
	v_mov_b32_e32 v94, v6
	v_mov_b32_e32 v137, v13
	v_mov_b32_e32 v136, v12
	v_mov_b32_e32 v135, v11
	v_mov_b32_e32 v134, v10
	v_mov_b32_e32 v121, v5
	v_mov_b32_e32 v120, v4
	v_mov_b32_e32 v119, v3
	v_mov_b32_e32 v118, v2
	v_mov_b32_e32 v161, v57
	v_mov_b32_e32 v160, v56
	v_mov_b32_e32 v159, v55
	v_mov_b32_e32 v158, v54
	v_mov_b32_e32 v117, v33
	v_mov_b32_e32 v116, v32
	v_mov_b32_e32 v115, v31
	v_mov_b32_e32 v114, v30
	v_mov_b32_e32 v149, v29
	v_mov_b32_e32 v148, v28
	v_mov_b32_e32 v147, v27
	v_mov_b32_e32 v146, v26
	v_accvgpr_write_b32 a142, v132
	v_accvgpr_write_b32 a141, v131
	v_accvgpr_write_b32 a140, v130
	v_mov_b32_e32 v157, v21
	v_mov_b32_e32 v156, v20
	v_mov_b32_e32 v155, v19
	v_mov_b32_e32 v154, v18
	v_mov_b32_e32 v141, v9
	v_mov_b32_e32 v140, v8
	v_mov_b32_e32 v139, v7
	v_mov_b32_e32 v138, v6
	v_mov_b32_e32 v169, v13
	v_mov_b32_e32 v168, v12
	v_mov_b32_e32 v167, v11
	v_mov_b32_e32 v166, v10
	v_mov_b32_e32 v133, v5
	v_mov_b32_e32 v132, v4
	v_mov_b32_e32 v131, v3
	v_mov_b32_e32 v130, v2
	v_mov_b32_e32 v193, v57
	v_mov_b32_e32 v192, v56
	v_mov_b32_e32 v191, v55
	v_mov_b32_e32 v190, v54
	v_mov_b32_e32 v145, v33
	v_mov_b32_e32 v144, v32
	v_mov_b32_e32 v143, v31
	v_mov_b32_e32 v142, v30
	v_mov_b32_e32 v173, v29
	v_mov_b32_e32 v172, v28
	v_mov_b32_e32 v171, v27
	v_mov_b32_e32 v170, v26
	v_mov_b32_e32 v165, v17
	v_mov_b32_e32 v164, v16
	v_mov_b32_e32 v163, v15
	v_mov_b32_e32 v162, v14
	v_mov_b32_e32 v185, v21
	v_mov_b32_e32 v184, v20
	v_mov_b32_e32 v183, v19
	v_mov_b32_e32 v182, v18
	v_accvgpr_write_b32 a127, v175
	v_accvgpr_write_b32 a128, v176
	v_accvgpr_write_b32 a129, v177
	v_mov_b32_e32 v197, v13
	v_mov_b32_e32 v196, v12
	v_mov_b32_e32 v195, v11
	v_mov_b32_e32 v194, v10
	v_mov_b32_e32 v189, v5
	v_mov_b32_e32 v188, v4
	v_mov_b32_e32 v187, v3
	v_mov_b32_e32 v186, v2
	v_mov_b32_e32 v217, v57
	v_mov_b32_e32 v216, v56
	v_mov_b32_e32 v215, v55
	v_mov_b32_e32 v214, v54
	v_mov_b32_e32 v181, v33
	v_mov_b32_e32 v180, v32
	v_mov_b32_e32 v179, v31
	v_mov_b32_e32 v178, v30
	v_mov_b32_e32 v205, v29
	v_mov_b32_e32 v204, v28
	v_mov_b32_e32 v203, v27
	v_mov_b32_e32 v202, v26
	v_mov_b32_e32 v177, v17
	v_mov_b32_e32 v176, v16
	v_mov_b32_e32 v175, v15
	v_mov_b32_e32 v174, v14
	v_mov_b32_e32 v209, v21
	v_mov_b32_e32 v208, v20
	v_mov_b32_e32 v207, v19
	v_mov_b32_e32 v206, v18
	v_mov_b32_e32 v201, v9
	v_mov_b32_e32 v200, v8
	v_mov_b32_e32 v199, v7
	v_mov_b32_e32 v198, v6
	v_mov_b32_e32 v213, v13
	v_mov_b32_e32 v212, v12
	v_mov_b32_e32 v211, v11
	v_mov_b32_e32 v210, v10
	v_mov_b32_e32 v221, v5
	v_mov_b32_e32 v220, v4
	v_mov_b32_e32 v219, v3
	v_mov_b32_e32 v218, v2
	v_accvgpr_write_b32 a124, v37
	v_mov_b32_e32 v37, v57
	v_accvgpr_write_b32 a125, v36
	v_mov_b32_e32 v36, v56
	v_accvgpr_write_b32 a131, v35
	v_mov_b32_e32 v35, v55
	v_accvgpr_write_b32 a144, v34
	v_mov_b32_e32 v34, v54
	v_mov_b32_e32 v229, v33
	v_mov_b32_e32 v228, v32
	v_mov_b32_e32 v227, v31
	v_mov_b32_e32 v226, v30
	v_mov_b32_e32 v233, v29
	v_mov_b32_e32 v232, v28
	v_mov_b32_e32 v231, v27
	v_mov_b32_e32 v230, v26
	v_mov_b32_e32 v237, v17
	v_mov_b32_e32 v236, v16
	v_mov_b32_e32 v235, v15
	v_mov_b32_e32 v234, v14
	v_mov_b32_e32 v241, v21
	v_mov_b32_e32 v240, v20
	v_mov_b32_e32 v239, v19
	v_mov_b32_e32 v238, v18
	v_mov_b32_e32 v245, v9
	v_mov_b32_e32 v244, v8
	v_mov_b32_e32 v243, v7
	v_mov_b32_e32 v242, v6
	v_mov_b32_e32 v249, v13
	v_mov_b32_e32 v248, v12
	v_mov_b32_e32 v247, v11
	v_mov_b32_e32 v246, v10
	v_mov_b32_e32 v253, v5
	v_mov_b32_e32 v252, v4
	v_mov_b32_e32 v251, v3
	v_mov_b32_e32 v250, v2
	v_accvgpr_write_b32 a145, v25
	v_mov_b32_e32 v25, v57
	v_mov_b32_e32 v24, v56
	v_mov_b32_e32 v23, v55
	v_accvgpr_write_b32 a130, v22
	v_mov_b32_e32 v22, v54
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s22, s2
	s_add_i32 s2, s20, 1
	s_cmp_lt_i32 s2, 2
	s_cselect_b32 s20, s2, 0
	s_lshl_b32 s2, s20, 14
	s_mov_b32 s21, s3
	s_ashr_i32 s3, s2, 5
	s_add_i32 s3, s3, s2
	s_lshl1_add_u32 s2, s3, 0
	s_add_i32 s3, s2, s11
	s_and_b32 s5, s14, 0xffff
	s_mov_b32 s4, s13
	s_mov_b32 m0, s3
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v1, s[4:7], 0 offen lds
	s_add_i32 m0, s3, 0x1080
	v_accvgpr_write_b32 a153, v29
	buffer_load_dwordx4 v40, s[4:7], 0 offen lds
	s_add_i32 m0, s3, 0x2100
	v_add_u32_e32 v38, s22, v254
	buffer_load_dwordx4 v41, s[4:7], 0 offen lds
	s_add_i32 m0, s3, 0x3180
	v_accvgpr_read_b32 v0, a130
	buffer_load_dwordx4 v46, s[4:7], 0 offen lds
	s_add_i32 m0, s3, 0x4200
	v_accvgpr_write_b32 a152, v28
	buffer_load_dwordx4 v47, s[4:7], 0 offen lds
	s_add_i32 m0, s3, 0x5280
	v_accvgpr_write_b32 a151, v27
	buffer_load_dwordx4 v48, s[4:7], 0 offen lds
	s_add_i32 m0, s3, 0x6300
	v_accvgpr_write_b32 a150, v26
	buffer_load_dwordx4 v49, s[4:7], 0 offen lds
	s_add_i32 m0, s3, 0x7380
	s_add_i32 s3, s2, 0x107e0
	v_mov_b64_e32 v[28:29], v[8:9]
	v_accvgpr_mov_b32 a173, a171
	s_add_i32 s22, s3, s11
	buffer_load_dwordx4 v222, s[4:7], 0 offen lds
	v_add3_u32 v38, v38, v0, v255
	v_accvgpr_mov_b32 a161, a103
	v_accvgpr_mov_b32 a157, a99
	v_mov_b64_e32 v[26:27], v[6:7]
	v_accvgpr_mov_b32 a172, a170
	v_accvgpr_mov_b32 a171, a169
	v_accvgpr_mov_b32 a170, a168
	v_accvgpr_mov_b32 a165, a107
	v_accvgpr_mov_b32 a169, a111
	s_and_b32 s5, s17, 0xffff
	s_mov_b32 s4, s15
	s_mov_b32 m0, s22
	v_accvgpr_read_b32 v9, a145
	v_accvgpr_mov_b32 a160, a102
	v_accvgpr_mov_b32 a159, a101
	v_accvgpr_mov_b32 a158, a100
	v_accvgpr_mov_b32 a156, a98
	v_accvgpr_mov_b32 a155, a97
	v_accvgpr_mov_b32 a154, a96
	v_accvgpr_mov_b32 a164, a106
	v_accvgpr_mov_b32 a163, a105
	v_accvgpr_mov_b32 a162, a104
	v_accvgpr_mov_b32 a168, a110
	v_accvgpr_mov_b32 a167, a109
	v_accvgpr_mov_b32 a166, a108
	ds_read_b128 v[150:153], v38 offset:320
	ds_read_b128 a[96:99], v38 offset:512
	ds_read_b128 a[100:103], v38 offset:576
	ds_read_b128 a[104:107], v38 offset:768
	ds_read_b128 a[108:111], v38 offset:832
	ds_read_b128 a[88:91], v38 offset:16896
	ds_read_b128 a[92:95], v38 offset:16960
	ds_read_b128 a[72:75], v38 offset:17152
	ds_read_b128 a[76:79], v38 offset:17216
	ds_read_b128 a[64:67], v38 offset:17408
	ds_read_b128 a[68:71], v38 offset:17472
	ds_read_b128 a[8:11], v38 offset:17664
	ds_read_b128 a[0:3], v38 offset:17728
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 m0, s22, 0x1080
	v_accvgpr_read_b32 v9, a144
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 m0, s22, 0x2100
	v_accvgpr_read_b32 v9, a131
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 m0, s22, 0x3180
	v_accvgpr_read_b32 v9, a125
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 m0, s22, 0x4200
	v_accvgpr_read_b32 v9, a123
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 m0, s22, 0x5280
	v_accvgpr_read_b32 v9, a122
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 m0, s22, 0x6300
	v_accvgpr_read_b32 v9, a121
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 m0, s22, 0x7380
	v_accvgpr_read_b32 v9, a124
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 s4, s12, s21
	v_accvgpr_write_b32 a116, v46
	v_accvgpr_write_b32 a117, v47
	v_accvgpr_write_b32 a118, v48
	v_accvgpr_write_b32 a119, v49
	ds_read_b128 v[46:49], v38
	ds_read_b128 v[70:73], v38 offset:64
	ds_read_b128 v[106:109], v38 offset:256
	v_add_u32_e32 v38, s4, v39
	v_add3_u32 v38, v38, v254, v255
	v_accvgpr_write_b32 a120, v222
	v_mov_b64_e32 v[224:225], v[16:17]
	ds_read_b128 a[80:83], v38
	ds_read_b128 a[84:87], v38 offset:64
	ds_read_b128 a[56:59], v38 offset:256
	ds_read_b128 a[60:63], v38 offset:320
	ds_read_b128 a[48:51], v38 offset:512
	ds_read_b128 a[52:55], v38 offset:576
	ds_read_b128 a[40:43], v38 offset:768
	ds_read_b128 a[44:47], v38 offset:832
	ds_read_b128 a[32:35], v38 offset:16896
	ds_read_b128 a[36:39], v38 offset:16960
	ds_read_b128 a[24:27], v38 offset:17152
	ds_read_b128 a[28:31], v38 offset:17216
	ds_read_b128 a[16:19], v38 offset:17408
	ds_read_b128 a[20:23], v38 offset:17472
	ds_read_b128 a[12:15], v38 offset:17664
	ds_read_b128 a[4:7], v38 offset:17728
	v_mov_b64_e32 v[222:223], v[14:15]
	v_mov_b64_e32 v[16:17], v[4:5]
	v_mov_b64_e32 v[14:15], v[2:3]
	v_mov_b32_e32 v8, v1
	v_mov_b64_e32 v[0:1], v[66:67]
	v_accvgpr_write_b32 a149, v33
	v_mov_b64_e32 v[2:3], v[68:69]
	v_accvgpr_write_b32 a148, v32
	v_accvgpr_write_b32 a147, v31
	v_accvgpr_write_b32 a146, v30
	v_mov_b64_e32 v[32:33], v[20:21]
	v_accvgpr_write_b32 a114, v40
	v_accvgpr_write_b32 a115, v41
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_bf16 v[22:25], a[80:83], v[46:49], v[22:25]
	v_mov_b64_e32 v[30:31], v[18:19]
	v_mov_b64_e32 v[20:21], v[12:13]
	v_mov_b64_e32 v[18:19], v[10:11]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_bf16 v[250:253], a[56:59], v[46:49], v[250:253]
	v_mov_b64_e32 v[10:11], v[54:55]
	v_mov_b64_e32 v[68:69], v[52:53]
	v_mov_b64_e32 v[12:13], v[56:57]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_bf16 v[246:249], a[48:51], v[46:49], v[246:249]
	v_mov_b64_e32 v[66:67], v[50:51]
	s_add_u32 s15, s15, 0x80
	s_addc_u32 s17, s17, 0
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_bf16 v[242:245], a[40:43], v[46:49], v[242:245]
	s_add_u32 s13, s13, 0x80
	s_addc_u32 s14, s14, 0
	s_add_i32 s19, s19, -1
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 v[238:241], a[32:35], v[46:49], v[238:241]
	s_cmp_lg_u32 s19, 0
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 v[234:237], a[24:27], v[46:49], v[234:237]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 v[230:233], a[16:19], v[46:49], v[230:233]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[226:229], a[12:15], v[46:49], v[226:229]
	v_mov_b32_e32 v46, v39
	v_accvgpr_read_b32 v49, a119
	v_accvgpr_read_b32 v48, a118
	v_mfma_f32_16x16x32_bf16 v[38:41], a[32:35], a[64:67], v[0:3]
	v_accvgpr_read_b32 v47, a117
	v_mfma_f32_16x16x32_bf16 v[214:217], a[80:83], a[96:99], v[214:217]
	s_nop 0
	v_mov_b32_e32 v1, v8
	v_mfma_f32_16x16x32_bf16 v[186:189], a[56:59], a[96:99], v[186:189]
	v_mfma_f32_16x16x32_bf16 v[194:197], a[48:51], a[96:99], v[194:197]
	v_mfma_f32_16x16x32_bf16 a[126:129], a[40:43], a[96:99], a[126:129]
	v_mfma_f32_16x16x32_bf16 v[182:185], a[32:35], a[96:99], v[182:185]
	v_mfma_f32_16x16x32_bf16 v[162:165], a[24:27], a[96:99], v[162:165]
	v_mfma_f32_16x16x32_bf16 v[170:173], a[16:19], a[96:99], v[170:173]
	v_mfma_f32_16x16x32_bf16 v[142:145], a[12:15], a[96:99], v[142:145]
	v_mfma_f32_16x16x32_bf16 v[190:193], a[80:83], a[104:107], v[190:193]
	v_mfma_f32_16x16x32_bf16 v[130:133], a[56:59], a[104:107], v[130:133]
	v_mfma_f32_16x16x32_bf16 v[166:169], a[48:51], a[104:107], v[166:169]
	v_mfma_f32_16x16x32_bf16 v[138:141], a[40:43], a[104:107], v[138:141]
	v_mfma_f32_16x16x32_bf16 v[154:157], a[32:35], a[104:107], v[154:157]
	v_mfma_f32_16x16x32_bf16 a[140:143], a[24:27], a[104:107], a[140:143]
	v_mfma_f32_16x16x32_bf16 v[146:149], a[16:19], a[104:107], v[146:149]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], a[104:107], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[50:53], a[56:59], a[64:67], v[66:69]
	v_mfma_f32_16x16x32_bf16 v[66:69], a[36:39], a[68:71], v[38:41]
	v_mfma_f32_16x16x32_bf16 v[38:41], a[80:83], a[8:11], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[10:13], a[48:51], a[8:11], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[6:9], a[40:43], a[8:11], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[18:21], a[32:35], a[8:11], v[30:33]
	s_nop 1
	v_accvgpr_read_b32 v26, a150
	v_accvgpr_read_b32 v27, a151
	v_accvgpr_read_b32 v28, a152
	v_accvgpr_read_b32 v30, a146
	v_accvgpr_read_b32 v29, a153
	v_accvgpr_read_b32 v31, a147
	v_accvgpr_read_b32 v32, a148
	v_accvgpr_read_b32 v33, a149
	v_mfma_f32_16x16x32_bf16 v[34:37], a[80:83], v[106:109], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[218:221], a[56:59], v[106:109], v[218:221]
	v_mfma_f32_16x16x32_bf16 v[210:213], a[48:51], v[106:109], v[210:213]
	v_mfma_f32_16x16x32_bf16 v[198:201], a[40:43], v[106:109], v[198:201]
	v_mfma_f32_16x16x32_bf16 v[206:209], a[32:35], v[106:109], v[206:209]
	v_mfma_f32_16x16x32_bf16 v[174:177], a[24:27], v[106:109], v[174:177]
	v_mfma_f32_16x16x32_bf16 v[202:205], a[16:19], v[106:109], v[202:205]
	v_mfma_f32_16x16x32_bf16 v[178:181], a[12:15], v[106:109], v[178:181]
	v_mfma_f32_16x16x32_bf16 v[214:217], a[84:87], a[100:103], v[214:217]
	v_mfma_f32_16x16x32_bf16 v[186:189], a[60:63], a[100:103], v[186:189]
	v_mfma_f32_16x16x32_bf16 v[194:197], a[52:55], a[100:103], v[194:197]
	v_mfma_f32_16x16x32_bf16 a[126:129], a[44:47], a[100:103], a[126:129]
	v_mfma_f32_16x16x32_bf16 v[182:185], a[36:39], a[100:103], v[182:185]
	v_mfma_f32_16x16x32_bf16 v[162:165], a[28:31], a[100:103], v[162:165]
	v_mfma_f32_16x16x32_bf16 v[170:173], a[20:23], a[100:103], v[170:173]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[142:145], a[4:7], a[100:103], v[142:145]
	v_mfma_f32_16x16x32_bf16 v[190:193], a[84:87], a[108:111], v[190:193]
	v_mfma_f32_16x16x32_bf16 v[130:133], a[60:63], a[108:111], v[130:133]
	v_mfma_f32_16x16x32_bf16 v[166:169], a[52:55], a[108:111], v[166:169]
	v_mfma_f32_16x16x32_bf16 v[138:141], a[44:47], a[108:111], v[138:141]
	v_mfma_f32_16x16x32_bf16 v[154:157], a[36:39], a[108:111], v[154:157]
	v_mfma_f32_16x16x32_bf16 a[140:143], a[28:31], a[108:111], a[140:143]
	v_mfma_f32_16x16x32_bf16 v[146:149], a[20:23], a[108:111], v[146:149]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], a[108:111], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[158:161], a[80:83], a[88:91], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[118:121], a[56:59], a[88:91], v[118:121]
	v_mfma_f32_16x16x32_bf16 v[134:137], a[48:51], a[88:91], v[134:137]
	v_mfma_f32_16x16x32_bf16 v[94:97], a[40:43], a[88:91], v[94:97]
	v_mfma_f32_16x16x32_bf16 v[126:129], a[32:35], a[88:91], v[126:129]
	v_mfma_f32_16x16x32_bf16 v[98:101], a[24:27], a[88:91], v[98:101]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], a[88:91], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[78:81], a[12:15], a[88:91], v[78:81]
	v_mfma_f32_16x16x32_bf16 v[122:125], a[80:83], a[72:75], v[122:125]
	v_mfma_f32_16x16x32_bf16 a[136:139], a[56:59], a[72:75], a[136:139]
	v_mfma_f32_16x16x32_bf16 v[102:105], a[48:51], a[72:75], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[74:77], a[40:43], a[72:75], v[74:77]
	v_mfma_f32_16x16x32_bf16 v[90:93], a[32:35], a[72:75], v[90:93]
	v_mfma_f32_16x16x32_bf16 a[108:111], a[24:27], a[72:75], a[166:169]
	v_mfma_f32_16x16x32_bf16 v[82:85], a[16:19], a[72:75], v[82:85]
	v_mfma_f32_16x16x32_bf16 v[58:61], a[12:15], a[72:75], v[58:61]
	v_mfma_f32_16x16x32_bf16 v[86:89], a[80:83], a[64:67], v[86:89]
	v_mfma_f32_16x16x32_bf16 a[132:135], a[48:51], a[64:67], a[132:135]
	v_mfma_f32_16x16x32_bf16 a[96:99], a[40:43], a[64:67], a[154:157]
	v_mfma_f32_16x16x32_bf16 a[104:107], a[24:27], a[64:67], a[162:165]
	v_mfma_f32_16x16x32_bf16 a[164:167], a[16:19], a[64:67], a[170:173]
	v_mfma_f32_16x16x32_bf16 a[100:103], a[12:15], a[64:67], a[158:161]
	v_mfma_f32_16x16x32_bf16 v[2:5], a[56:59], a[8:11], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[14:17], a[24:27], a[8:11], v[222:225]
	v_mfma_f32_16x16x32_bf16 v[26:29], a[16:19], a[8:11], v[26:29]
	s_nop 1
	v_accvgpr_read_b32 v222, a120
	v_mfma_f32_16x16x32_bf16 v[30:33], a[12:15], a[8:11], v[30:33]
	v_mfma_f32_16x16x32_bf16 v[22:25], a[84:87], v[70:73], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[250:253], a[60:63], v[70:73], v[250:253]
	v_mfma_f32_16x16x32_bf16 v[246:249], a[52:55], v[70:73], v[246:249]
	v_mfma_f32_16x16x32_bf16 v[242:245], a[44:47], v[70:73], v[242:245]
	v_mfma_f32_16x16x32_bf16 v[238:241], a[36:39], v[70:73], v[238:241]
	v_mfma_f32_16x16x32_bf16 v[234:237], a[28:31], v[70:73], v[234:237]
	v_mfma_f32_16x16x32_bf16 v[230:233], a[20:23], v[70:73], v[230:233]
	v_mfma_f32_16x16x32_bf16 v[226:229], a[4:7], v[70:73], v[226:229]
	v_mfma_f32_16x16x32_bf16 v[34:37], a[84:87], v[150:153], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[218:221], a[60:63], v[150:153], v[218:221]
	v_mfma_f32_16x16x32_bf16 v[210:213], a[52:55], v[150:153], v[210:213]
	v_mfma_f32_16x16x32_bf16 v[198:201], a[44:47], v[150:153], v[198:201]
	v_mfma_f32_16x16x32_bf16 v[206:209], a[36:39], v[150:153], v[206:209]
	v_mfma_f32_16x16x32_bf16 v[174:177], a[28:31], v[150:153], v[174:177]
	v_mfma_f32_16x16x32_bf16 v[202:205], a[20:23], v[150:153], v[202:205]
	v_mfma_f32_16x16x32_bf16 v[178:181], a[4:7], v[150:153], v[178:181]
	v_mfma_f32_16x16x32_bf16 v[158:161], a[84:87], a[92:95], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[118:121], a[60:63], a[92:95], v[118:121]
	v_mfma_f32_16x16x32_bf16 v[134:137], a[52:55], a[92:95], v[134:137]
	v_mfma_f32_16x16x32_bf16 v[94:97], a[44:47], a[92:95], v[94:97]
	v_mfma_f32_16x16x32_bf16 v[126:129], a[36:39], a[92:95], v[126:129]
	v_mfma_f32_16x16x32_bf16 v[98:101], a[28:31], a[92:95], v[98:101]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[20:23], a[92:95], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[78:81], a[4:7], a[92:95], v[78:81]
	v_mfma_f32_16x16x32_bf16 v[122:125], a[84:87], a[76:79], v[122:125]
	v_mfma_f32_16x16x32_bf16 a[136:139], a[60:63], a[76:79], a[136:139]
	v_mfma_f32_16x16x32_bf16 v[102:105], a[52:55], a[76:79], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[74:77], a[44:47], a[76:79], v[74:77]
	v_mfma_f32_16x16x32_bf16 v[90:93], a[36:39], a[76:79], v[90:93]
	v_mfma_f32_16x16x32_bf16 a[108:111], a[28:31], a[76:79], a[108:111]
	v_mfma_f32_16x16x32_bf16 v[82:85], a[20:23], a[76:79], v[82:85]
	v_mfma_f32_16x16x32_bf16 v[58:61], a[4:7], a[76:79], v[58:61]
	v_mfma_f32_16x16x32_bf16 v[86:89], a[84:87], a[68:71], v[86:89]
	v_mfma_f32_16x16x32_bf16 v[50:53], a[60:63], a[68:71], v[50:53]
	v_mfma_f32_16x16x32_bf16 a[132:135], a[52:55], a[68:71], a[132:135]
	v_mfma_f32_16x16x32_bf16 a[96:99], a[44:47], a[68:71], a[96:99]
	v_mfma_f32_16x16x32_bf16 a[104:107], a[28:31], a[68:71], a[104:107]
	v_mfma_f32_16x16x32_bf16 a[168:171], a[20:23], a[68:71], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[100:103], a[4:7], a[68:71], a[100:103]
	v_mfma_f32_16x16x32_bf16 v[54:57], a[84:87], a[0:3], v[38:41]
	v_mfma_f32_16x16x32_bf16 v[2:5], a[60:63], a[0:3], v[2:5]
	s_nop 1
	v_accvgpr_read_b32 v41, a115
	v_accvgpr_read_b32 v40, a114
	v_mov_b32_e32 v39, v46
	v_mfma_f32_16x16x32_bf16 v[10:13], a[52:55], a[0:3], v[10:13]
	v_accvgpr_read_b32 v46, a116
	v_mfma_f32_16x16x32_bf16 v[6:9], a[44:47], a[0:3], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[18:21], a[36:39], a[0:3], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[14:17], a[28:31], a[0:3], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[26:29], a[20:23], a[0:3], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[30:33], a[4:7], a[0:3], v[30:33]
	s_cbranch_scc1 .LBB0_6
; %bb.7:                                ; %._crit_edge.loopexit
	v_accvgpr_mov_b32 a122, a140
	v_accvgpr_mov_b32 a118, a136
	v_accvgpr_mov_b32 a114, a132
	v_accvgpr_mov_b32 a123, a141
	v_accvgpr_mov_b32 a124, a142
	v_accvgpr_mov_b32 a125, a143
	v_accvgpr_mov_b32 a119, a137
	v_accvgpr_mov_b32 a120, a138
	v_accvgpr_mov_b32 a121, a139
	v_accvgpr_mov_b32 a115, a133
	v_accvgpr_mov_b32 a116, a134
	v_accvgpr_mov_b32 a117, a135
                                        ; kill: def $vgpr33 killed $vgpr33 killed $exec
                                        ; kill: def $vgpr32 killed $vgpr32 killed $exec
                                        ; kill: def $vgpr31 killed $vgpr31 killed $exec
                                        ; kill: def $vgpr30 killed $vgpr30 killed $exec
	v_accvgpr_write_b32 a139, v33
                                        ; kill: def $vgpr29 killed $vgpr29 killed $exec
                                        ; kill: def $vgpr28 killed $vgpr28 killed $exec
                                        ; kill: def $vgpr27 killed $vgpr27 killed $exec
                                        ; kill: def $vgpr26 killed $vgpr26 killed $exec
                                        ; kill: def $vgpr17 killed $vgpr17 killed $exec
                                        ; kill: def $vgpr16 killed $vgpr16 killed $exec
                                        ; kill: def $vgpr15 killed $vgpr15 killed $exec
                                        ; kill: def $vgpr14 killed $vgpr14 killed $exec
	v_accvgpr_write_b32 a135, v17
                                        ; kill: def $vgpr21 killed $vgpr21 killed $exec
                                        ; kill: def $vgpr20 killed $vgpr20 killed $exec
                                        ; kill: def $vgpr19 killed $vgpr19 killed $exec
                                        ; kill: def $vgpr18 killed $vgpr18 killed $exec
	v_accvgpr_write_b32 a155, v21
                                        ; kill: def $vgpr9 killed $vgpr9 killed $exec
                                        ; kill: def $vgpr8 killed $vgpr8 killed $exec
                                        ; kill: def $vgpr7 killed $vgpr7 killed $exec
                                        ; kill: def $vgpr6 killed $vgpr6 killed $exec
	v_accvgpr_write_b32 a147, v9
                                        ; kill: def $vgpr13 killed $vgpr13 killed $exec
                                        ; kill: def $vgpr12 killed $vgpr12 killed $exec
                                        ; kill: def $vgpr11 killed $vgpr11 killed $exec
                                        ; kill: def $vgpr10 killed $vgpr10 killed $exec
	v_accvgpr_write_b32 a159, v13
                                        ; kill: def $vgpr5 killed $vgpr5 killed $exec
                                        ; kill: def $vgpr4 killed $vgpr4 killed $exec
                                        ; kill: def $vgpr3 killed $vgpr3 killed $exec
                                        ; kill: def $vgpr2 killed $vgpr2 killed $exec
	v_accvgpr_write_b32 a143, v5
                                        ; kill: def $vgpr57 killed $vgpr57 killed $exec
                                        ; kill: def $vgpr56 killed $vgpr56 killed $exec
                                        ; kill: def $vgpr55 killed $vgpr55 killed $exec
                                        ; kill: def $vgpr54 killed $vgpr54 killed $exec
	v_accvgpr_write_b32 a151, v57
	v_accvgpr_write_b32 a167, v29
	v_mov_b32_e32 v0, s12
	v_accvgpr_write_b32 a138, v32
	v_accvgpr_write_b32 a137, v31
	v_accvgpr_write_b32 a136, v30
	v_accvgpr_write_b32 a134, v16
	v_accvgpr_write_b32 a133, v15
	v_accvgpr_write_b32 a132, v14
	v_accvgpr_write_b32 a154, v20
	v_accvgpr_write_b32 a153, v19
	v_accvgpr_write_b32 a152, v18
	v_accvgpr_write_b32 a146, v8
	v_accvgpr_write_b32 a145, v7
	v_accvgpr_write_b32 a144, v6
	v_accvgpr_write_b32 a158, v12
	v_accvgpr_write_b32 a157, v11
	v_accvgpr_write_b32 a156, v10
	v_accvgpr_write_b32 a142, v4
	v_accvgpr_write_b32 a141, v3
	v_accvgpr_write_b32 a140, v2
	v_accvgpr_write_b32 a150, v56
	v_accvgpr_write_b32 a149, v55
	v_accvgpr_write_b32 a148, v54
	v_mov_b32_e32 v55, v23
	v_mov_b32_e32 v54, v22
	v_mov_b32_e32 v57, v25
	v_mov_b32_e32 v56, v24
	v_mov_b32_e32 v3, v251
	v_mov_b32_e32 v2, v250
	v_mov_b32_e32 v5, v253
	v_mov_b32_e32 v4, v252
	v_mov_b32_e32 v11, v247
	v_mov_b32_e32 v10, v246
	v_mov_b32_e32 v13, v249
	v_mov_b32_e32 v12, v248
	v_accvgpr_write_b32 a166, v28
	v_accvgpr_write_b32 a165, v27
	v_accvgpr_write_b32 a164, v26
	v_mov_b32_e32 v7, v243
	v_mov_b32_e32 v6, v242
	v_mov_b32_e32 v9, v245
	v_mov_b32_e32 v8, v244
	v_mov_b32_e32 v19, v239
	v_mov_b32_e32 v18, v238
	v_mov_b32_e32 v21, v241
	v_mov_b32_e32 v20, v240
	v_mov_b32_e32 v15, v235
	v_mov_b32_e32 v14, v234
	v_mov_b32_e32 v17, v237
	v_mov_b32_e32 v16, v236
	v_mov_b32_e32 v27, v231
	v_mov_b32_e32 v26, v230
	v_mov_b32_e32 v29, v233
	v_mov_b32_e32 v28, v232
	v_mov_b32_e32 v31, v227
	v_mov_b32_e32 v30, v226
	v_mov_b32_e32 v33, v229
	v_mov_b32_e32 v32, v228
	v_accvgpr_read_b32 v38, a112
	v_accvgpr_read_b32 v24, a113
	v_accvgpr_read_b32 v22, a130
.LBB0_8:                                ; %Flow330
	v_add_u32_e32 v1, s3, v0
	v_add_u32_e32 v1, v1, v39
	v_add_u32_e32 v0, s2, v254
	v_add3_u32 v1, v1, v254, v255
	v_add3_u32 v0, v0, v22, v255
	v_lshrrev_b32_e32 v226, 2, v38
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[0:3], v1 offset:17728
	ds_read_b128 a[8:11], v1 offset:17664
	ds_read_b128 v[38:41], v1 offset:17472
	ds_read_b128 v[46:49], v1 offset:17408
	ds_read_b128 a[12:15], v1 offset:17216
	ds_read_b128 a[16:19], v1 offset:17152
	ds_read_b128 v[70:73], v1 offset:16960
	ds_read_b128 v[106:109], v1 offset:16896
	ds_read_b128 a[20:23], v1 offset:832
	ds_read_b128 a[24:27], v1 offset:768
	ds_read_b128 v[150:153], v1 offset:576
	ds_read_b128 v[228:231], v1 offset:512
	ds_read_b128 a[28:31], v1 offset:320
	ds_read_b128 a[32:35], v1 offset:256
	ds_read_b128 v[232:235], v1 offset:64
	ds_read_b128 v[244:247], v1
	ds_read_b128 a[4:7], v0 offset:17728
	ds_read_b128 a[36:39], v0 offset:17664
	ds_read_b128 a[40:43], v0 offset:17472
	ds_read_b128 a[44:47], v0 offset:17408
	ds_read_b128 a[48:51], v0 offset:17216
	ds_read_b128 a[52:55], v0 offset:17152
	ds_read_b128 a[56:59], v0 offset:16960
	ds_read_b128 a[60:63], v0 offset:16896
	ds_read_b128 a[64:67], v0 offset:832
	ds_read_b128 a[68:71], v0 offset:768
	ds_read_b128 a[72:75], v0 offset:576
	ds_read_b128 a[76:79], v0 offset:512
	ds_read_b128 a[80:83], v0 offset:320
	ds_read_b128 a[84:87], v0 offset:256
	ds_read_b128 a[88:91], v0 offset:64
	ds_read_b128 a[92:95], v0
	s_and_b64 s[4:5], s[24:25], exec
	s_load_dword s0, s[0:1], 0x38
	s_cselect_b32 s1, 0, 16
	v_or_b32_e32 v227, s1, v24
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[22:25], v[244:247], a[92:95], v[54:57]
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[2:3], s[16:17], 8
	s_ashr_i32 s19, s18, 31
	v_mfma_f32_16x16x32_bf16 v[62:65], v[46:49], a[52:55], v[82:85]
	s_ashr_i32 s11, s10, 31
	s_ashr_i32 s37, s36, 31
	s_and_b32 s9, s9, 0xffff
	v_mfma_f32_16x16x32_bf16 v[82:85], v[228:231], a[52:55], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[228:231], a[60:63], v[134:137]
	v_mfma_f32_16x16x32_bf16 v[222:225], v[232:235], a[88:91], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[22:25], a[32:35], a[84:87], v[218:221]
	v_mfma_f32_16x16x32_bf16 v[134:137], v[150:153], a[56:59], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[46:49], a[68:71], v[146:149]
	v_mfma_f32_16x16x32_bf16 v[236:239], a[28:31], a[80:83], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[244:247], a[84:87], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[146:149], v[38:41], a[64:67], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[106:109], a[68:71], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[218:221], v[232:235], a[80:83], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[244:247], a[76:79], v[214:217]
	v_mfma_f32_16x16x32_bf16 v[154:157], v[70:73], a[64:67], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[228:231], a[68:71], v[166:169]
	v_mfma_f32_16x16x32_bf16 v[214:217], v[232:235], a[72:75], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[244:247], a[68:71], v[190:193]
	v_mfma_f32_16x16x32_bf16 v[166:169], v[150:153], a[64:67], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[46:49], a[76:79], v[170:173]
	v_mfma_f32_16x16x32_bf16 v[190:193], v[232:235], a[64:67], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[244:247], a[60:63], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[170:173], v[38:41], a[72:75], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[106:109], a[76:79], v[182:185]
	v_mfma_f32_16x16x32_bf16 v[158:161], v[232:235], a[56:59], v[22:25]
	s_nop 5
	v_cvt_pk_bf16_f32 v170, v170, v171
	v_cvt_pk_bf16_f32 v171, v172, v173
	v_mfma_f32_16x16x32_bf16 v[22:25], v[244:247], a[52:55], v[122:125]
	v_mfma_f32_16x16x32_bf16 v[182:185], v[70:73], a[72:75], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[228:231], a[76:79], v[194:197]
	v_mfma_f32_16x16x32_bf16 v[240:243], v[232:235], a[48:51], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[244:247], a[44:47], v[86:89]
	v_mfma_f32_16x16x32_bf16 v[86:89], v[46:49], a[60:63], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[194:197], v[150:153], a[72:75], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[46:49], a[84:87], v[202:205]
	v_mfma_f32_16x16x32_bf16 v[110:113], v[228:231], a[84:87], v[210:213]
	s_nop 5
	v_cvt_pk_bf16_f32 v194, v194, v195
	v_cvt_pk_bf16_f32 v195, v196, v197
	v_cvt_pk_bf16_f32 v196, v182, v183
	v_mfma_f32_16x16x32_bf16 v[10:13], v[228:231], a[92:95], v[10:13]
	v_cvt_pk_bf16_f32 v182, v190, v191
	v_cvt_pk_bf16_f32 v191, v168, v169
	v_cvt_pk_bf16_f32 v169, v156, v157
	v_mfma_f32_16x16x32_bf16 v[54:57], v[106:109], a[44:47], v[66:69]
	v_cvt_pk_bf16_f32 v156, v134, v135
	v_cvt_pk_bf16_f32 v157, v136, v137
	v_cvt_pk_bf16_f32 v197, v184, v185
	v_mfma_f32_16x16x32_bf16 v[202:205], v[38:41], a[80:83], v[102:105]
	v_cvt_pk_bf16_f32 v183, v192, v193
	v_cvt_pk_bf16_f32 v190, v166, v167
	v_cvt_pk_bf16_f32 v168, v154, v155
	v_mfma_f32_16x16x32_bf16 v[102:105], v[106:109], a[84:87], v[206:209]
	v_mfma_f32_16x16x32_bf16 v[206:209], v[150:153], a[80:83], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], v[150:153], a[88:91], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[10:13], a[32:35], a[76:79], v[186:189]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[8:11], a[68:71], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[252:255], v[232:235], a[40:43], v[22:25]
	s_nop 2
	v_accvgpr_read_b32 v22, a114
	v_accvgpr_read_b32 v23, a115
	v_accvgpr_read_b32 v24, a116
	v_accvgpr_read_b32 v25, a117
	v_mfma_f32_16x16x32_bf16 a[148:151], v[244:247], a[36:39], a[148:151]
	v_accvgpr_mov_b32 a112, a136
	v_accvgpr_mov_b32 a113, a137
	v_accvgpr_mov_b32 a114, a138
	v_mfma_f32_16x16x32_bf16 a[156:159], v[228:231], a[36:39], a[156:159]
	v_accvgpr_mov_b32 a115, a139
	v_mfma_f32_16x16x32_bf16 v[248:251], v[70:73], a[40:43], v[54:57]
	v_mfma_f32_16x16x32_bf16 v[54:57], v[228:231], a[44:47], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[186:189], a[28:31], a[72:75], v[10:13]
	s_nop 1
	v_accvgpr_read_b32 v22, a96
	v_accvgpr_read_b32 v23, a97
	v_accvgpr_read_b32 v24, a98
	v_mfma_f32_16x16x32_bf16 v[10:13], a[32:35], a[60:63], v[118:121]
	v_accvgpr_read_b32 v25, a99
	v_mfma_f32_16x16x32_bf16 v[228:231], a[0:3], a[64:67], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[24:27], a[68:71], v[138:141]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[232:235], a[4:7], a[148:151]
	v_mfma_f32_16x16x32_bf16 v[210:213], a[28:31], a[56:59], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[10:13], a[32:35], a[44:47], v[50:53]
	v_mfma_f32_16x16x32_bf16 v[232:235], a[20:23], a[64:67], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[16:19], a[76:79], v[162:165]
	v_mfma_f32_16x16x32_bf16 v[244:247], v[150:153], a[40:43], v[54:57]
	s_nop 5
	v_cvt_pk_bf16_f32 v172, v232, v233
	v_cvt_pk_bf16_f32 v173, v234, v235
	v_mfma_f32_16x16x32_bf16 v[54:57], a[28:31], a[40:43], v[10:13]
	s_nop 2
	v_accvgpr_read_b32 v10, a144
	v_mfma_f32_16x16x32_bf16 v[162:165], a[12:15], a[72:75], v[114:117]
	v_accvgpr_read_b32 v11, a145
	v_accvgpr_read_b32 v12, a146
	v_accvgpr_read_b32 v13, a147
	v_mfma_f32_16x16x32_bf16 v[114:117], a[8:11], a[84:87], v[178:181]
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[50:53], a[24:27], a[36:39], v[10:13]
	s_nop 0
	v_or_b32_e32 v181, 32, v227
	v_or_b32_e32 v180, 0x60, v227
	v_accvgpr_read_b32 v10, a104
	v_accvgpr_read_b32 v11, a105
	v_accvgpr_read_b32 v12, a106
	v_accvgpr_read_b32 v13, a107
	v_mfma_f32_16x16x32_bf16 a[148:151], v[46:49], a[36:39], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[152:155], v[106:109], a[36:39], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[46:49], a[44:47], a[168:171]
	v_mfma_f32_16x16x32_bf16 v[66:69], v[106:109], a[52:55], v[90:93]
	v_mfma_f32_16x16x32_bf16 v[90:93], v[106:109], a[60:63], v[126:129]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[46:49], a[92:95], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[18:21], v[106:109], a[92:95], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[42:45], a[16:19], a[44:47], v[10:13]
	s_nop 2
	v_accvgpr_read_b32 v10, a100
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0:3], a[80:83], v[114:117]
	v_accvgpr_read_b32 v11, a101
	v_accvgpr_read_b32 v12, a102
	v_accvgpr_read_b32 v13, a103
	v_mfma_f32_16x16x32_bf16 v[114:117], a[24:27], a[84:87], v[198:201]
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[118:121], a[8:11], a[44:47], v[10:13]
	s_nop 2
	v_accvgpr_read_b32 v10, a118
	v_accvgpr_read_b32 v11, a119
	v_accvgpr_read_b32 v12, a120
	v_accvgpr_read_b32 v13, a121
	v_mfma_f32_16x16x32_bf16 a[152:155], v[70:73], a[4:7], a[152:155]
	v_mfma_f32_16x16x32_bf16 v[66:69], v[70:73], a[48:51], v[66:69]
	v_mfma_f32_16x16x32_bf16 v[90:93], v[70:73], a[56:59], v[90:93]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[70:73], a[80:83], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[18:21], v[70:73], a[88:91], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[70:73], a[20:23], a[80:83], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[32:35], a[52:55], v[10:13]
	s_nop 5
	v_cvt_pk_bf16_f32 v179, v20, v21
	v_cvt_pk_bf16_f32 v178, v18, v19
	v_accvgpr_read_b32 v10, a132
	v_accvgpr_read_b32 v11, a133
	v_accvgpr_read_b32 v12, a134
	v_accvgpr_read_b32 v13, a135
	v_mfma_f32_16x16x32_bf16 a[148:151], v[38:41], a[4:7], a[148:151]
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[138:141], a[16:19], a[36:39], v[10:13]
	s_nop 2
	v_accvgpr_read_b32 v10, a108
	v_accvgpr_read_b32 v11, a109
	v_accvgpr_read_b32 v12, a110
	v_accvgpr_read_b32 v13, a111
	v_mfma_f32_16x16x32_bf16 a[164:167], v[38:41], a[40:43], a[164:167]
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[126:129], a[16:19], a[52:55], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[122:125], v[38:41], a[48:51], v[62:65]
	s_nop 1
	v_or_b32_e32 v13, 0xc0, v227
	v_or_b32_e32 v12, 64, v226
	v_mfma_f32_16x16x32_bf16 v[86:89], v[38:41], a[56:59], v[86:89]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[38:41], a[88:91], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[0:3], a[32:35], a[92:95], v[2:5]
	v_mfma_f32_16x16x32_bf16 v[38:41], a[12:15], a[48:51], v[126:129]
	s_nop 5
	v_cvt_pk_bf16_f32 v200, v26, v27
	v_cvt_pk_bf16_f32 v201, v28, v29
	v_cvt_pk_bf16_f32 v28, v248, v249
	v_mfma_f32_16x16x32_bf16 v[126:129], a[8:11], a[76:79], v[142:145]
	v_cvt_pk_bf16_f32 v29, v250, v251
	v_mfma_f32_16x16x32_bf16 v[0:3], a[28:31], a[88:91], v[0:3]
	v_mfma_f32_16x16x32_bf16 v[106:109], a[0:3], a[72:75], v[126:129]
	v_mfma_f32_16x16x32_bf16 v[126:129], a[16:19], a[84:87], v[174:177]
	v_mfma_f32_16x16x32_bf16 v[4:7], a[24:27], a[92:95], v[6:9]
	s_nop 5
	v_cvt_pk_bf16_f32 v184, v106, v107
	v_cvt_pk_bf16_f32 v185, v108, v109
	v_accvgpr_read_b32 v8, a140
	v_mfma_f32_16x16x32_bf16 v[62:65], a[24:27], a[44:47], v[22:25]
	v_accvgpr_read_b32 v9, a141
	v_accvgpr_read_b32 v10, a142
	v_accvgpr_read_b32 v11, a143
	v_mfma_f32_16x16x32_bf16 v[22:25], a[32:35], a[68:71], v[130:133]
	v_mfma_f32_16x16x32_bf16 v[34:37], a[28:31], a[48:51], v[114:117]
	s_nop 1
	v_cvt_pk_bf16_f32 v130, v0, v1
	v_cvt_pk_bf16_f32 v131, v2, v3
	v_cvt_pk_bf16_f32 v133, v224, v225
	v_mfma_f32_16x16x32_bf16 v[114:117], a[32:35], a[36:39], v[8:11]
	v_or_b32_e32 v225, 0x80, v227
	v_or_b32_e32 v224, 0xe0, v227
	v_cvt_pk_bf16_f32 v132, v222, v223
	v_accvgpr_read_b32 v8, a122
	v_accvgpr_read_b32 v9, a123
	v_accvgpr_read_b32 v10, a124
	v_accvgpr_read_b32 v11, a125
	v_mfma_f32_16x16x32_bf16 a[156:159], v[150:153], a[4:7], a[156:159]
	v_or_b32_e32 v223, 32, v226
	v_or_b32_e32 v222, 0x60, v226
	v_mfma_f32_16x16x32_bf16 v[82:85], v[150:153], a[48:51], v[82:85]
	v_mfma_f32_16x16x32_bf16 v[150:153], a[12:15], a[80:83], v[126:129]
	v_mfma_f32_16x16x32_bf16 v[4:7], a[20:23], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_bf16 v[126:129], a[16:19], a[68:71], v[8:11]
	v_mfma_f32_16x16x32_bf16 v[30:33], a[8:11], a[92:95], v[30:33]
	s_nop 1
	v_accvgpr_read_b32 v8, a126
	v_accvgpr_read_b32 v9, a127
	v_accvgpr_read_b32 v10, a128
	v_mfma_f32_16x16x32_bf16 v[0:3], a[28:31], a[64:67], v[22:25]
	v_accvgpr_read_b32 v11, a129
	v_mfma_f32_16x16x32_bf16 v[14:17], a[16:19], a[92:95], v[14:17]
	s_nop 0
	v_accvgpr_read_b32 v22, a148
	v_accvgpr_read_b32 v23, a149
	v_accvgpr_read_b32 v24, a150
	v_mfma_f32_16x16x32_bf16 v[30:33], a[0:3], a[88:91], v[30:33]
	v_accvgpr_read_b32 v25, a151
	v_mfma_f32_16x16x32_bf16 v[174:177], a[12:15], a[64:67], v[126:129]
	s_nop 2
	v_cvt_pk_bf16_f32 v127, v6, v7
	v_cvt_pk_bf16_f32 v7, v104, v105
	v_cvt_pk_bf16_f32 v104, v202, v203
	v_cvt_pk_bf16_f32 v202, v186, v187
	v_cvt_pk_bf16_f32 v187, v164, v165
	v_cvt_pk_bf16_f32 v164, v0, v1
	v_cvt_pk_bf16_f32 v165, v2, v3
	v_mfma_f32_16x16x32_bf16 v[0:3], a[20:23], a[40:43], v[62:65]
	v_cvt_pk_bf16_f32 v198, v30, v31
	v_cvt_pk_bf16_f32 v203, v188, v189
	v_cvt_pk_bf16_f32 v129, v112, v113
	v_mfma_f32_16x16x32_bf16 v[14:17], a[12:15], a[88:91], v[14:17]
	v_cvt_pk_bf16_f32 v113, v208, v209
	v_cvt_pk_bf16_f32 v6, v102, v103
	v_cvt_pk_bf16_f32 v209, v152, v153
	v_mfma_f32_16x16x32_bf16 v[142:145], a[24:27], a[76:79], v[8:11]
	v_cvt_pk_bf16_f32 v30, v0, v1
	v_cvt_pk_bf16_f32 v31, v2, v3
	v_accvgpr_read_b32 v0, a164
	v_accvgpr_read_b32 v1, a165
	v_accvgpr_read_b32 v2, a166
	v_accvgpr_read_b32 v3, a167
	v_cvt_pk_bf16_f32 v20, v0, v1
	v_cvt_pk_bf16_f32 v21, v2, v3
	v_accvgpr_read_b32 v0, a160
	v_accvgpr_read_b32 v1, a161
	v_accvgpr_read_b32 v2, a162
	v_accvgpr_read_b32 v3, a163
	v_mfma_f32_16x16x32_bf16 v[8:11], a[20:23], a[72:75], v[142:145]
	v_cvt_pk_bf16_f32 v103, v48, v49
	v_cvt_pk_bf16_f32 v152, v86, v87
	v_cvt_pk_bf16_f32 v86, v34, v35
	v_cvt_pk_bf16_f32 v145, v16, v17
	v_cvt_pk_bf16_f32 v16, v0, v1
	v_cvt_pk_bf16_f32 v17, v2, v3
	v_mfma_f32_16x16x32_bf16 v[0:3], a[28:31], a[4:7], v[114:117]
	v_cvt_pk_bf16_f32 v144, v14, v15
	v_cvt_pk_bf16_f32 v188, v8, v9
	v_or_b32_e32 v143, 64, v227
	v_mfma_f32_16x16x32_bf16 v[42:45], a[12:15], a[40:43], v[42:45]
	v_or_b32_e32 v142, 0xa0, v227
	v_cvt_pk_bf16_f32 v48, v82, v83
	v_cvt_pk_bf16_f32 v82, v66, v67
	v_mfma_f32_16x16x32_bf16 a[112:115], a[8:11], a[36:39], a[112:115]
	v_cvt_pk_bf16_f32 v14, v0, v1
	v_cvt_pk_bf16_f32 v15, v2, v3
	v_accvgpr_read_b32 v0, a156
	v_accvgpr_read_b32 v1, a157
	v_accvgpr_read_b32 v2, a158
	v_accvgpr_read_b32 v3, a159
	v_cvt_pk_bf16_f32 v8, v0, v1
	v_cvt_pk_bf16_f32 v9, v2, v3
	v_accvgpr_read_b32 v0, a152
	v_accvgpr_read_b32 v1, a153
	v_accvgpr_read_b32 v2, a154
	v_accvgpr_read_b32 v3, a155
	v_mfma_f32_16x16x32_bf16 v[58:61], a[8:11], a[52:55], v[58:61]
	v_cvt_pk_bf16_f32 v26, v42, v43
	v_cvt_pk_bf16_f32 v42, v0, v1
	v_cvt_pk_bf16_f32 v43, v2, v3
	v_mfma_f32_16x16x32_bf16 v[0:3], a[12:15], a[4:7], v[138:141]
	v_cvt_pk_bf16_f32 v66, v38, v39
	v_cvt_pk_bf16_f32 v34, v54, v55
	v_cvt_pk_bf16_f32 v35, v56, v57
	v_mfma_f32_16x16x32_bf16 v[74:77], a[24:27], a[52:55], v[74:77]
	v_cvt_pk_bf16_f32 v27, v44, v45
	v_or_b32_e32 v54, s2, v143
	v_mov_b32_e32 v55, s3
	v_mfma_f32_16x16x32_bf16 v[78:81], a[8:11], a[60:63], v[78:81]
	v_cvt_pk_bf16_f32 v135, v2, v3
	v_cvt_pk_bf16_f32 v2, v22, v23
	v_cvt_pk_bf16_f32 v3, v24, v25
	v_mfma_f32_16x16x32_bf16 v[50:53], a[20:23], a[4:7], v[50:53]
	v_accvgpr_read_b32 v22, a112
	v_accvgpr_read_b32 v23, a113
	v_accvgpr_read_b32 v24, a114
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0:3], a[48:51], v[58:61]
	v_accvgpr_read_b32 v25, a115
	v_or_b32_e32 v56, s2, v180
	v_mov_b32_e32 v57, s3
	v_mfma_f32_16x16x32_bf16 v[22:25], a[0:3], a[4:7], v[22:25]
	v_cvt_pk_bf16_f32 v44, v50, v51
	v_cvt_pk_bf16_f32 v45, v52, v53
	v_or_b32_e32 v50, s2, v227
	v_mfma_f32_16x16x32_bf16 v[74:77], a[20:23], a[48:51], v[74:77]
	v_cvt_pk_bf16_f32 v38, v58, v59
	v_cvt_pk_bf16_f32 v39, v60, v61
	v_mov_b32_e32 v51, s3
	v_mfma_f32_16x16x32_bf16 v[78:81], a[0:3], a[56:59], v[78:81]
	v_or_b32_e32 v52, s2, v181
	v_mov_b32_e32 v53, s3
	v_or_b32_e32 v58, s2, v225
	v_mov_b32_e32 v59, s3
	v_or_b32_e32 v60, s2, v142
	v_mov_b32_e32 v61, s3
	v_or_b32_e32 v62, s2, v13
	v_mov_b32_e32 v63, s3
	v_or_b32_e32 v64, s2, v224
	v_mov_b32_e32 v65, s3
	s_lshl_b64 s[2:3], s[18:19], 8
	v_cvt_pk_bf16_f32 v126, v4, v5
	v_cvt_pk_bf16_f32 v5, v220, v221
	v_or_b32_e32 v221, 0x80, v226
	v_or_b32_e32 v220, 0xa0, v226
	v_cvt_pk_bf16_f32 v112, v206, v207
	v_or_b32_e32 v207, 0xc0, v226
	v_or_b32_e32 v206, 0xe0, v226
	v_or_b32_e32 v142, s2, v226
	v_mov_b32_e32 v143, s3
	v_cvt_pk_bf16_f32 v189, v10, v11
	v_cvt_pk_bf16_f32 v83, v68, v69
	v_cvt_pk_bf16_f32 v134, v0, v1
	v_cvt_pk_bf16_f32 v0, v22, v23
	v_cvt_pk_bf16_f32 v1, v24, v25
	v_or_b32_e32 v140, s2, v223
	v_mov_b32_e32 v141, s3
	v_or_b32_e32 v138, s2, v12
	v_mov_b32_e32 v139, s3
	v_or_b32_e32 v136, s2, v222
	v_mov_b32_e32 v137, s3
	v_or_b32_e32 v24, s2, v221
	v_mov_b32_e32 v25, s3
	v_or_b32_e32 v22, s2, v220
	v_mov_b32_e32 v23, s3
	v_or_b32_e32 v12, s2, v207
	v_mov_b32_e32 v13, s3
	v_or_b32_e32 v10, s2, v206
	v_mov_b32_e32 v11, s3
	v_mul_lo_u32 v68, v50, s0
	v_cmp_gt_i64_e64 s[34:35], s[10:11], v[50:51]
	v_cmp_gt_i64_e64 s[18:19], s[36:37], v[142:143]
	v_cvt_pk_bf16_f32 v208, v150, v151
	v_cvt_pk_bf16_f32 v102, v46, v47
	v_cvt_pk_bf16_f32 v150, v78, v79
	v_cvt_pk_bf16_f32 v46, v74, v75
	v_cvt_pk_bf16_f32 v47, v76, v77
	v_mul_lo_u32 v69, v52, s0
	v_mul_lo_u32 v74, v54, s0
	v_mul_lo_u32 v75, v56, s0
	v_mul_lo_u32 v76, v58, s0
	v_mul_lo_u32 v77, v60, s0
	v_mul_lo_u32 v78, v62, s0
	v_mul_lo_u32 v79, v64, s0
	v_cmp_gt_i64_e64 s[16:17], s[36:37], v[140:141]
	v_cmp_gt_i64_e64 s[14:15], s[36:37], v[138:139]
	v_cmp_gt_i64_e64 s[12:13], s[36:37], v[136:137]
	v_cmp_gt_i64_e64 s[6:7], s[36:37], v[24:25]
	v_cmp_gt_i64_e64 s[4:5], s[36:37], v[22:23]
	v_cmp_gt_i64_e64 s[2:3], s[36:37], v[12:13]
	v_cmp_gt_i64_e64 s[0:1], s[36:37], v[10:11]
	v_add_lshl_u32 v11, v142, v68, 1
	v_bfrev_b32_e32 v13, 1
	s_and_b64 s[36:37], s[34:35], s[18:19]
	v_cmp_gt_i64_e64 s[30:31], s[10:11], v[52:53]
	v_cmp_gt_i64_e64 s[28:29], s[10:11], v[54:55]
	v_cmp_gt_i64_e64 s[26:27], s[10:11], v[56:57]
	v_cmp_gt_i64_e64 s[24:25], s[10:11], v[58:59]
	v_cmp_gt_i64_e64 s[22:23], s[10:11], v[60:61]
	v_cmp_gt_i64_e64 s[20:21], s[10:11], v[62:63]
	v_cmp_gt_i64_e32 vcc, s[10:11], v[64:65]
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_cndmask_b32_e64 v11, v13, v11, s[36:37]
	buffer_store_dwordx2 v[132:133], v11, s[8:11], 0 offen
	v_add_lshl_u32 v11, v140, v68, 1
	s_and_b64 s[36:37], s[34:35], s[16:17]
	v_cndmask_b32_e64 v11, v13, v11, s[36:37]
	buffer_store_dwordx2 v[130:131], v11, s[8:11], 0 offen
	v_add_lshl_u32 v11, v138, v68, 1
	s_and_b64 s[36:37], s[34:35], s[14:15]
	v_cvt_pk_bf16_f32 v128, v110, v111
	v_cndmask_b32_e64 v11, v13, v11, s[36:37]
	buffer_store_dwordx2 v[128:129], v11, s[8:11], 0 offen
	v_add_lshl_u32 v11, v136, v68, 1
	s_and_b64 s[36:37], s[34:35], s[12:13]
	v_cndmask_b32_e64 v11, v13, v11, s[36:37]
	buffer_store_dwordx2 v[126:127], v11, s[8:11], 0 offen
	v_add_lshl_u32 v11, v24, v68, 1
	s_and_b64 s[36:37], s[34:35], s[6:7]
	v_cndmask_b32_e64 v11, v13, v11, s[36:37]
	buffer_store_dwordx2 v[178:179], v11, s[8:11], 0 offen
	v_add_lshl_u32 v11, v22, v68, 1
	s_and_b64 s[36:37], s[34:35], s[4:5]
	v_cndmask_b32_e64 v11, v13, v11, s[36:37]
	buffer_store_dwordx2 v[144:145], v11, s[8:11], 0 offen
	v_add_lshl_u32 v11, v12, v68, 1
	s_and_b64 s[36:37], s[34:35], s[2:3]
	v_cndmask_b32_e64 v11, v13, v11, s[36:37]
	buffer_store_dwordx2 v[200:201], v11, s[8:11], 0 offen
	v_add_lshl_u32 v11, v10, v68, 1
	s_and_b64 s[34:35], s[34:35], s[0:1]
	v_cvt_pk_bf16_f32 v199, v32, v33
	v_cndmask_b32_e64 v11, v13, v11, s[34:35]
	buffer_store_dwordx2 v[198:199], v11, s[8:11], 0 offen
	v_add_lshl_u32 v11, v142, v69, 1
	s_and_b64 s[34:35], s[30:31], s[18:19]
	v_cvt_pk_bf16_f32 v4, v218, v219
	v_cndmask_b32_e64 v11, v13, v11, s[34:35]
	buffer_store_dwordx2 v[4:5], v11, s[8:11], 0 offen
	v_add_lshl_u32 v4, v140, v69, 1
	s_and_b64 s[34:35], s[30:31], s[16:17]
	v_cvt_pk_bf16_f32 v218, v236, v237
	v_cvt_pk_bf16_f32 v219, v238, v239
	v_cndmask_b32_e64 v4, v13, v4, s[34:35]
	buffer_store_dwordx2 v[218:219], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v138, v69, 1
	s_and_b64 s[34:35], s[30:31], s[14:15]
	v_cndmask_b32_e64 v4, v13, v4, s[34:35]
	buffer_store_dwordx2 v[112:113], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v136, v69, 1
	s_and_b64 s[34:35], s[30:31], s[12:13]
	v_cvt_pk_bf16_f32 v110, v70, v71
	v_cvt_pk_bf16_f32 v111, v72, v73
	v_cndmask_b32_e64 v4, v13, v4, s[34:35]
	buffer_store_dwordx2 v[110:111], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v24, v69, 1
	s_and_b64 s[34:35], s[30:31], s[6:7]
	v_cndmask_b32_e64 v4, v13, v4, s[34:35]
	buffer_store_dwordx2 v[6:7], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v22, v69, 1
	s_and_b64 s[34:35], s[30:31], s[4:5]
	v_cndmask_b32_e64 v4, v13, v4, s[34:35]
	buffer_store_dwordx2 v[208:209], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v12, v69, 1
	s_and_b64 s[34:35], s[30:31], s[2:3]
	v_cvt_pk_bf16_f32 v105, v204, v205
	v_cndmask_b32_e64 v4, v13, v4, s[34:35]
	buffer_store_dwordx2 v[104:105], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v10, v69, 1
	s_and_b64 s[30:31], s[30:31], s[0:1]
	v_cndmask_b32_e64 v4, v13, v4, s[30:31]
	buffer_store_dwordx2 v[102:103], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v142, v74, 1
	s_and_b64 s[30:31], s[28:29], s[18:19]
	v_cvt_pk_bf16_f32 v204, v214, v215
	v_cvt_pk_bf16_f32 v205, v216, v217
	v_cndmask_b32_e64 v4, v13, v4, s[30:31]
	buffer_store_dwordx2 v[204:205], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v140, v74, 1
	s_and_b64 s[30:31], s[28:29], s[16:17]
	v_cndmask_b32_e64 v4, v13, v4, s[30:31]
	buffer_store_dwordx2 v[202:203], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v138, v74, 1
	s_and_b64 s[30:31], s[28:29], s[14:15]
	v_cndmask_b32_e64 v4, v13, v4, s[30:31]
	buffer_store_dwordx2 v[194:195], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v136, v74, 1
	s_and_b64 s[30:31], s[28:29], s[12:13]
	v_cndmask_b32_e64 v4, v13, v4, s[30:31]
	buffer_store_dwordx2 v[188:189], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v24, v74, 1
	s_and_b64 s[30:31], s[28:29], s[6:7]
	v_cndmask_b32_e64 v4, v13, v4, s[30:31]
	buffer_store_dwordx2 v[196:197], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v22, v74, 1
	s_and_b64 s[30:31], s[28:29], s[4:5]
	v_cvt_pk_bf16_f32 v186, v162, v163
	v_cndmask_b32_e64 v4, v13, v4, s[30:31]
	buffer_store_dwordx2 v[186:187], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v12, v74, 1
	s_and_b64 s[30:31], s[28:29], s[2:3]
	v_cndmask_b32_e64 v4, v13, v4, s[30:31]
	buffer_store_dwordx2 v[170:171], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v10, v74, 1
	s_and_b64 s[28:29], s[28:29], s[0:1]
	v_cndmask_b32_e64 v4, v13, v4, s[28:29]
	buffer_store_dwordx2 v[184:185], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v142, v75, 1
	s_and_b64 s[28:29], s[26:27], s[18:19]
	v_cndmask_b32_e64 v4, v13, v4, s[28:29]
	buffer_store_dwordx2 v[182:183], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v140, v75, 1
	s_and_b64 s[28:29], s[26:27], s[16:17]
	v_cndmask_b32_e64 v4, v13, v4, s[28:29]
	buffer_store_dwordx2 v[164:165], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v138, v75, 1
	s_and_b64 s[28:29], s[26:27], s[14:15]
	v_cndmask_b32_e64 v4, v13, v4, s[28:29]
	buffer_store_dwordx2 v[190:191], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v136, v75, 1
	s_and_b64 s[28:29], s[26:27], s[12:13]
	v_cndmask_b32_e64 v4, v13, v4, s[28:29]
	buffer_store_dwordx2 v[172:173], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v24, v75, 1
	s_and_b64 s[28:29], s[26:27], s[6:7]
	v_cndmask_b32_e64 v4, v13, v4, s[28:29]
	buffer_store_dwordx2 v[168:169], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v22, v75, 1
	s_and_b64 s[28:29], s[26:27], s[4:5]
	v_cvt_pk_bf16_f32 v166, v174, v175
	v_cvt_pk_bf16_f32 v167, v176, v177
	v_cndmask_b32_e64 v4, v13, v4, s[28:29]
	buffer_store_dwordx2 v[166:167], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v12, v75, 1
	s_and_b64 s[28:29], s[26:27], s[2:3]
	v_cvt_pk_bf16_f32 v72, v146, v147
	v_cvt_pk_bf16_f32 v73, v148, v149
	v_cndmask_b32_e64 v4, v13, v4, s[28:29]
	buffer_store_dwordx2 v[72:73], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v10, v75, 1
	s_and_b64 s[26:27], s[26:27], s[0:1]
	v_mfma_f32_16x16x32_bf16 v[94:97], a[24:27], a[60:63], v[94:97]
	v_cvt_pk_bf16_f32 v70, v228, v229
	v_cvt_pk_bf16_f32 v71, v230, v231
	v_cndmask_b32_e64 v4, v13, v4, s[26:27]
	buffer_store_dwordx2 v[70:71], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v142, v76, 1
	s_and_b64 s[26:27], s[24:25], s[18:19]
	v_cvt_pk_bf16_f32 v162, v158, v159
	v_cvt_pk_bf16_f32 v163, v160, v161
	v_cndmask_b32_e64 v4, v13, v4, s[26:27]
	buffer_store_dwordx2 v[162:163], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v140, v76, 1
	s_and_b64 s[26:27], s[24:25], s[16:17]
	v_mfma_f32_16x16x32_bf16 v[98:101], a[16:19], a[60:63], v[98:101]
	v_cvt_pk_bf16_f32 v158, v210, v211
	v_cvt_pk_bf16_f32 v159, v212, v213
	v_cndmask_b32_e64 v4, v13, v4, s[26:27]
	v_mfma_f32_16x16x32_bf16 v[94:97], a[20:23], a[56:59], v[94:97]
	buffer_store_dwordx2 v[158:159], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v138, v76, 1
	s_and_b64 s[26:27], s[24:25], s[14:15]
	v_cndmask_b32_e64 v4, v13, v4, s[26:27]
	buffer_store_dwordx2 v[156:157], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v136, v76, 1
	s_and_b64 s[26:27], s[24:25], s[12:13]
	v_mfma_f32_16x16x32_bf16 v[98:101], a[12:15], a[56:59], v[98:101]
	v_cvt_pk_bf16_f32 v154, v94, v95
	v_cvt_pk_bf16_f32 v155, v96, v97
	v_cndmask_b32_e64 v4, v13, v4, s[26:27]
	buffer_store_dwordx2 v[154:155], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v24, v76, 1
	s_and_b64 s[26:27], s[24:25], s[6:7]
	v_cvt_pk_bf16_f32 v146, v90, v91
	v_cvt_pk_bf16_f32 v147, v92, v93
	v_cndmask_b32_e64 v4, v13, v4, s[26:27]
	buffer_store_dwordx2 v[146:147], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v22, v76, 1
	s_and_b64 s[26:27], s[24:25], s[4:5]
	v_cvt_pk_bf16_f32 v90, v98, v99
	v_cvt_pk_bf16_f32 v91, v100, v101
	v_cndmask_b32_e64 v4, v13, v4, s[26:27]
	buffer_store_dwordx2 v[90:91], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v12, v76, 1
	s_and_b64 s[26:27], s[24:25], s[2:3]
	v_cvt_pk_bf16_f32 v153, v88, v89
	v_cndmask_b32_e64 v4, v13, v4, s[26:27]
	buffer_store_dwordx2 v[152:153], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v10, v76, 1
	s_and_b64 s[24:25], s[24:25], s[0:1]
	v_cvt_pk_bf16_f32 v151, v80, v81
	v_cndmask_b32_e64 v4, v13, v4, s[24:25]
	buffer_store_dwordx2 v[150:151], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v142, v77, 1
	s_and_b64 s[24:25], s[22:23], s[18:19]
	v_cvt_pk_bf16_f32 v88, v240, v241
	v_cvt_pk_bf16_f32 v89, v242, v243
	v_cndmask_b32_e64 v4, v13, v4, s[24:25]
	buffer_store_dwordx2 v[88:89], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v140, v77, 1
	s_and_b64 s[24:25], s[22:23], s[16:17]
	v_cvt_pk_bf16_f32 v87, v36, v37
	v_cndmask_b32_e64 v4, v13, v4, s[24:25]
	buffer_store_dwordx2 v[86:87], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v138, v77, 1
	s_and_b64 s[24:25], s[22:23], s[14:15]
	v_cvt_pk_bf16_f32 v49, v84, v85
	v_cndmask_b32_e64 v4, v13, v4, s[24:25]
	buffer_store_dwordx2 v[48:49], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v136, v77, 1
	s_and_b64 s[24:25], s[22:23], s[12:13]
	v_cndmask_b32_e64 v4, v13, v4, s[24:25]
	buffer_store_dwordx2 v[46:47], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v24, v77, 1
	s_and_b64 s[24:25], s[22:23], s[6:7]
	v_cndmask_b32_e64 v4, v13, v4, s[24:25]
	buffer_store_dwordx2 v[82:83], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v22, v77, 1
	s_and_b64 s[24:25], s[22:23], s[4:5]
	v_cvt_pk_bf16_f32 v67, v40, v41
	v_cndmask_b32_e64 v4, v13, v4, s[24:25]
	buffer_store_dwordx2 v[66:67], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v12, v77, 1
	s_and_b64 s[24:25], s[22:23], s[2:3]
	v_cvt_pk_bf16_f32 v40, v122, v123
	v_cvt_pk_bf16_f32 v41, v124, v125
	v_cndmask_b32_e64 v4, v13, v4, s[24:25]
	buffer_store_dwordx2 v[40:41], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v10, v77, 1
	s_and_b64 s[22:23], s[22:23], s[0:1]
	v_cndmask_b32_e64 v4, v13, v4, s[22:23]
	buffer_store_dwordx2 v[38:39], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v142, v78, 1
	s_and_b64 s[22:23], s[20:21], s[18:19]
	v_cvt_pk_bf16_f32 v36, v252, v253
	v_cvt_pk_bf16_f32 v37, v254, v255
	v_cndmask_b32_e64 v4, v13, v4, s[22:23]
	buffer_store_dwordx2 v[36:37], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v140, v78, 1
	s_and_b64 s[22:23], s[20:21], s[16:17]
	v_cndmask_b32_e64 v4, v13, v4, s[22:23]
	buffer_store_dwordx2 v[34:35], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v138, v78, 1
	s_and_b64 s[22:23], s[20:21], s[14:15]
	v_cvt_pk_bf16_f32 v32, v244, v245
	v_cvt_pk_bf16_f32 v33, v246, v247
	v_cndmask_b32_e64 v4, v13, v4, s[22:23]
	buffer_store_dwordx2 v[32:33], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v136, v78, 1
	s_and_b64 s[22:23], s[20:21], s[12:13]
	v_cndmask_b32_e64 v4, v13, v4, s[22:23]
	buffer_store_dwordx2 v[30:31], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v24, v78, 1
	s_and_b64 s[22:23], s[20:21], s[6:7]
	v_cndmask_b32_e64 v4, v13, v4, s[22:23]
	buffer_store_dwordx2 v[28:29], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v22, v78, 1
	s_and_b64 s[22:23], s[20:21], s[4:5]
	v_mfma_f32_16x16x32_bf16 v[118:121], a[0:3], a[40:43], v[118:121]
	v_cndmask_b32_e64 v4, v13, v4, s[22:23]
	buffer_store_dwordx2 v[26:27], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v12, v78, 1
	s_and_b64 s[22:23], s[20:21], s[2:3]
	v_cndmask_b32_e64 v4, v13, v4, s[22:23]
	buffer_store_dwordx2 v[20:21], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v10, v78, 1
	s_and_b64 s[20:21], s[20:21], s[0:1]
	v_cvt_pk_bf16_f32 v18, v118, v119
	v_cvt_pk_bf16_f32 v19, v120, v121
	v_cndmask_b32_e64 v4, v13, v4, s[20:21]
	buffer_store_dwordx2 v[18:19], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v142, v79, 1
	s_and_b64 s[18:19], vcc, s[18:19]
	v_cndmask_b32_e64 v4, v13, v4, s[18:19]
	buffer_store_dwordx2 v[16:17], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v140, v79, 1
	s_and_b64 s[16:17], vcc, s[16:17]
	v_cndmask_b32_e64 v4, v13, v4, s[16:17]
	buffer_store_dwordx2 v[14:15], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v138, v79, 1
	s_and_b64 s[14:15], vcc, s[14:15]
	v_cndmask_b32_e64 v4, v13, v4, s[14:15]
	buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v136, v79, 1
	s_and_b64 s[12:13], vcc, s[12:13]
	v_cndmask_b32_e64 v4, v13, v4, s[12:13]
	buffer_store_dwordx2 v[44:45], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v24, v79, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cndmask_b32_e64 v4, v13, v4, s[6:7]
	buffer_store_dwordx2 v[42:43], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v22, v79, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cndmask_b32_e64 v4, v13, v4, s[4:5]
	buffer_store_dwordx2 v[134:135], v4, s[8:11], 0 offen
	v_add_lshl_u32 v4, v12, v79, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_cndmask_b32_e64 v4, v13, v4, s[2:3]
	buffer_store_dwordx2 v[2:3], v4, s[8:11], 0 offen
	v_add_lshl_u32 v2, v10, v79, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v2, v13, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen
.LBB0_9:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 430
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
	.size	_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0, .Lfunc_end0-_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
	.cfi_endproc
                                        ; -- End function
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.num_vgpr, 256
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.num_agpr, 174
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.numbered_sgpr, 38
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.num_named_barrier, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.private_seg_size, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.uses_vcc, 1
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.uses_flat_scratch, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.has_dyn_sized_stack, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.has_recursion, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13500
; TotalNumSgprs: 44
; NumVgprs: 256
; NumAgprs: 174
; TotalNumVgprs: 430
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 53
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 430
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
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
	.set amdgpu.max_num_named_barrier, 0
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
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
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
	.byte	1                               ; Abbrev [1] 0xb:0x84 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x5e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	101                             ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	104                             ; DW_AT_call_line
	.byte	48                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x61:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	97                              ; DW_AT_call_line
	.byte	41                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6d:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	100                             ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x79:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp27                         ; DW_AT_low_pc
	.long	.Ltmp28-.Ltmp27                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	142                             ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
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
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
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
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"gemm_a16w16.py"                ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/aiter/aiter/ops/triton/_triton_kernels" ; string offset=22
.Linfo_string3:
	.asciz	"_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0" ; string offset=78
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     174
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
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
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
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .max_flat_workgroup_size: 256
    .name:           _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     430
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
