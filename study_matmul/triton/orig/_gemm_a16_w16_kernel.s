	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0 ; -- Begin function _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
	.p2align	8
	.type	_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0,@function
_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0: ; @_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.13:
	.file	1 "/var/lib/jenkins/aiter/aiter/ops/triton/_triton_kernels" "gemm_a16w16.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.14:
.LBB0_0:
	s_mov_b64 s[24:25], s[6:7]
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s6, s11, 0xff
	s_ashr_i32 s7, s6, 31
	s_lshr_b32 s7, s7, 24
	s_add_i32 s7, s6, s7
	s_ashr_i32 s6, s7, 8
	.file	3 "/var/lib/jenkins/aiter/aiter/ops/triton/utils/_triton" "pid_preprocessing.py"
	s_lshl_b32 s6, s6, 4
	s_mov_b32 s28, s11
	s_abs_i32 s11, s6
	v_cvt_f32_u32_e32 v1, s11
	s_mov_b64 s[20:21], s[2:3]
	s_ashr_i32 s2, s16, 31
	s_lshr_b32 s2, s2, 29
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s2, s16, s2
	s_ashr_i32 s2, s2, 3
	s_lshl_b32 s3, s16, 5
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mulk_i32 s2, 0xff01
	s_add_i32 s2, s2, s3
	s_add_i32 s3, s10, 0xff
	s_ashr_i32 s15, s3, 31
	s_lshr_b32 s15, s15, 24
	s_sub_i32 s17, 0, s11
	v_readfirstlane_b32 s18, v1
	s_add_i32 s3, s3, s15
	s_mul_i32 s17, s17, s18
	s_ashr_i32 s15, s3, 8
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s7, s7, 31
	s_mul_hi_u32 s17, s18, s17
	s_xor_b32 s16, s3, s7
	s_abs_i32 s7, s2
	s_add_i32 s18, s18, s17
	s_mul_hi_u32 s17, s7, s18
	s_mul_i32 s18, s17, s11
	s_sub_i32 s18, s7, s18
	s_add_i32 s19, s17, 1
	s_sub_i32 s22, s18, s11
	s_cmp_ge_u32 s18, s11
	s_cselect_b32 s17, s19, s17
	s_cselect_b32 s18, s22, s18
	s_add_i32 s19, s17, 1
	s_cmp_ge_u32 s18, s11
	s_cselect_b32 s11, s19, s17
	s_xor_b32 s11, s11, s16
	s_sub_i32 s17, s11, s16
	s_lshl_b32 s16, s17, 4
	s_sub_i32 s11, s15, s16
	s_min_i32 s18, s11, 16
	s_abs_i32 s11, s18
	v_cvt_f32_u32_e32 v1, s11
	s_movk_i32 s15, 0xff
	s_cmp_lt_i32 s12, 1
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s19, v1
	s_cbranch_scc1 .LBB0_11
; %bb.1:
	s_ashr_i32 s22, s18, 31
	s_sub_i32 s18, 0, s11
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s7, s19
	s_mul_i32 s17, s17, s6
	s_mul_i32 s18, s18, s11
	s_sub_i32 s2, s2, s17
	s_sub_i32 s7, s7, s18
	s_ashr_i32 s6, s2, 31
	s_abs_i32 s2, s2
	s_sub_i32 s18, s7, s11
	s_cmp_ge_u32 s7, s11
	s_cselect_b32 s7, s18, s7
	s_sub_i32 s18, s7, s11
	s_cmp_ge_u32 s7, s11
	s_cselect_b32 s7, s18, s7
	s_xor_b32 s7, s7, s3
	s_mul_hi_u32 s17, s2, s19
	s_sub_i32 s3, s7, s3
	s_add_i32 s18, s16, s3
	s_xor_b32 s3, s6, s22
	s_mul_i32 s6, s17, s11
	s_sub_i32 s2, s2, s6
	s_add_i32 s6, s17, 1
	s_sub_i32 s7, s2, s11
	s_cmp_ge_u32 s2, s11
	s_cselect_b32 s6, s6, s17
	s_cselect_b32 s2, s7, s2
	s_add_i32 s7, s6, 1
	s_cmp_ge_u32 s2, s11
	s_cselect_b32 s2, s7, s6
	s_xor_b32 s2, s2, s3
	s_sub_i32 s16, s2, s3
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshrrev_b32_e32 v2, 6, v0
	s_movk_i32 s3, 0xf0
	v_and_or_b32 v2, v1, s3, v2
	s_abs_i32 s3, s10
	v_cvt_f32_u32_e32 v4, s3
	s_lshl_b32 s2, s18, 8
	v_or_b32_e32 v3, 8, v2
	s_sub_i32 s6, 0, s3
	v_rcp_iflag_f32_e32 v4, v4
	v_or_b32_e32 v5, s2, v2
	v_or_b32_e32 v6, s2, v3
	s_bfe_i32 s2, s18, 0x10017
	v_mul_f32_e32 v4, 0x4f7ffffe, v4
	v_cvt_u32_f32_e32 v4, v4
	v_add_u32_e32 v5, s2, v5
	v_xor_b32_e32 v5, s2, v5
	v_add_u32_e32 v6, s2, v6
	v_mul_lo_u32 v7, s6, v4
	v_mul_hi_u32 v7, v4, v7
	v_add_u32_e32 v4, v4, v7
	v_mul_hi_u32 v7, v5, v4
	v_mul_lo_u32 v7, v7, s3
	v_sub_u32_e32 v5, v5, v7
	v_xor_b32_e32 v6, s2, v6
	v_subrev_u32_e32 v7, s3, v5
	v_cmp_le_u32_e32 vcc, s3, v5
	v_mul_hi_u32 v4, v6, v4
	v_mul_lo_u32 v4, v4, s3
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_subrev_u32_e32 v7, s3, v5
	v_cmp_le_u32_e32 vcc, s3, v5
	v_sub_u32_e32 v4, v6, v4
	v_subrev_u32_e32 v6, s3, v4
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_cmp_le_u32_e32 vcc, s3, v4
	v_and_b32_e32 v1, 0xf0, v0
	v_lshrrev_b32_e32 v130, 2, v1
	v_cndmask_b32_e32 v4, v4, v6, vcc
	v_subrev_u32_e32 v6, s3, v4
	v_cmp_le_u32_e32 vcc, s3, v4
	s_abs_i32 s3, s28
	v_cvt_f32_u32_e32 v7, s3
	v_cndmask_b32_e32 v4, v4, v6, vcc
	v_xor_b32_e32 v5, s2, v5
	v_xor_b32_e32 v4, s2, v4
	v_rcp_iflag_f32_e32 v7, v7
	v_or_b32_e32 v133, 64, v130
	v_or_b32_e32 v132, 0x80, v130
	v_or_b32_e32 v131, 0xc0, v130
	v_mul_f32_e32 v7, 0x4f7ffffe, v7
	v_cvt_u32_f32_e32 v7, v7
	v_subrev_u32_e32 v5, s2, v5
	v_subrev_u32_e32 v4, s2, v4
	s_lshl_b32 s2, s16, 8
	s_sub_i32 s6, 0, s3
	v_or_b32_e32 v6, s2, v130
	v_or_b32_e32 v8, s2, v133
	v_or_b32_e32 v9, s2, v132
	v_or_b32_e32 v10, s2, v131
	v_or_b32_e32 v2, s2, v2
	v_or_b32_e32 v3, s2, v3
	s_bfe_i32 s2, s16, 0x10017
	v_mul_lo_u32 v11, s6, v7
	v_add_u32_e32 v6, s2, v6
	v_mul_hi_u32 v11, v7, v11
	v_xor_b32_e32 v6, s2, v6
	v_add_u32_e32 v7, v7, v11
	v_mul_hi_u32 v11, v6, v7
	v_mul_lo_u32 v11, v11, s3
	v_sub_u32_e32 v6, v6, v11
	v_subrev_u32_e32 v11, s3, v6
	v_cmp_le_u32_e32 vcc, s3, v6
	v_add_u32_e32 v8, s2, v8
	v_xor_b32_e32 v8, s2, v8
	v_cndmask_b32_e32 v6, v6, v11, vcc
	v_subrev_u32_e32 v11, s3, v6
	v_cmp_le_u32_e32 vcc, s3, v6
	v_add_u32_e32 v9, s2, v9
	v_xor_b32_e32 v9, s2, v9
	v_cndmask_b32_e32 v6, v6, v11, vcc
	v_mul_hi_u32 v11, v8, v7
	v_mul_lo_u32 v11, v11, s3
	v_sub_u32_e32 v8, v8, v11
	v_subrev_u32_e32 v11, s3, v8
	v_cmp_le_u32_e32 vcc, s3, v8
	v_add_u32_e32 v10, s2, v10
	v_xor_b32_e32 v10, s2, v10
	v_cndmask_b32_e32 v8, v8, v11, vcc
	v_subrev_u32_e32 v11, s3, v8
	v_cmp_le_u32_e32 vcc, s3, v8
	v_add_u32_e32 v2, s2, v2
	v_xor_b32_e32 v2, s2, v2
	v_cndmask_b32_e32 v8, v8, v11, vcc
	v_mul_hi_u32 v11, v9, v7
	v_mul_lo_u32 v11, v11, s3
	v_sub_u32_e32 v9, v9, v11
	v_subrev_u32_e32 v11, s3, v9
	v_cmp_le_u32_e32 vcc, s3, v9
	v_xor_b32_e32 v6, s2, v6
	v_xor_b32_e32 v8, s2, v8
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_subrev_u32_e32 v11, s3, v9
	v_cmp_le_u32_e32 vcc, s3, v9
	v_subrev_u32_e32 v6, s2, v6
	v_subrev_u32_e32 v8, s2, v8
	v_cndmask_b32_e32 v9, v9, v11, vcc
	v_mul_hi_u32 v11, v10, v7
	v_mul_lo_u32 v11, v11, s3
	v_sub_u32_e32 v10, v10, v11
	v_subrev_u32_e32 v11, s3, v10
	v_cmp_le_u32_e32 vcc, s3, v10
	v_xor_b32_e32 v9, s2, v9
	v_subrev_u32_e32 v9, s2, v9
	v_cndmask_b32_e32 v10, v10, v11, vcc
	v_subrev_u32_e32 v11, s3, v10
	v_cmp_le_u32_e32 vcc, s3, v10
	s_and_b32 s25, s25, 0xffff
	s_mov_b32 s27, 0x27000
	v_cndmask_b32_e32 v10, v10, v11, vcc
	v_mul_hi_u32 v11, v2, v7
	v_mul_lo_u32 v11, v11, s3
	v_sub_u32_e32 v2, v2, v11
	v_subrev_u32_e32 v11, s3, v2
	v_cmp_le_u32_e32 vcc, s3, v2
	v_xor_b32_e32 v10, s2, v10
	v_subrev_u32_e32 v10, s2, v10
	v_cndmask_b32_e32 v2, v2, v11, vcc
	v_subrev_u32_e32 v11, s3, v2
	v_cmp_le_u32_e32 vcc, s3, v2
	s_mov_b32 s26, 0x7ffffffe
	v_lshlrev_b32_e32 v8, 1, v8
	v_cndmask_b32_e32 v2, v2, v11, vcc
	v_xor_b32_e32 v2, s2, v2
	v_subrev_u32_e32 v11, s2, v2
	v_add_u32_e32 v2, s2, v3
	v_xor_b32_e32 v2, s2, v2
	v_mul_hi_u32 v3, v2, v7
	v_mul_lo_u32 v3, v3, s3
	v_sub_u32_e32 v2, v2, v3
	v_subrev_u32_e32 v3, s3, v2
	v_cmp_le_u32_e32 vcc, s3, v2
	v_lshlrev_b32_e32 v9, 1, v9
	v_lshlrev_b32_e32 v10, 1, v10
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s3, v2
	v_cmp_le_u32_e32 vcc, s3, v2
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s22, s26
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_xor_b32_e32 v2, s2, v2
	v_subrev_u32_e32 v3, s2, v2
	v_lshlrev_b32_e32 v2, 3, v0
	v_and_b32_e32 v2, 24, v2
	v_mad_u64_u32 v[22:23], s[2:3], v11, s14, v[2:3]
	v_lshlrev_b32_e32 v11, 1, v6
	v_mad_u64_u32 v[18:19], s[2:3], v5, s13, v[2:3]
	v_mad_u64_u32 v[20:21], s[2:3], v4, s13, v[2:3]
	v_mad_u64_u32 v[24:25], s[2:3], v3, s14, v[2:3]
	buffer_load_dwordx2 v[6:7], v11, s[24:27], 0 offen
	buffer_load_dwordx2 v[4:5], v8, s[24:27], 0 offen
	buffer_load_dwordx2 v[2:3], v9, s[24:27], 0 offen
	buffer_load_dwordx2 v[26:27], v10, s[24:27], 0 offen
	v_lshlrev_b32_e32 v8, 4, v0
	v_and_b32_e32 v8, 0x1c00, v8
	v_or_b32_e32 v11, 0x2000, v8
	v_lshrrev_b32_e32 v9, 5, v8
	v_lshrrev_b32_e32 v12, 5, v11
	v_or_b32_e32 v139, v9, v8
	v_or_b32_e32 v140, v12, v8
	v_add_u32_e32 v9, 0, v139
	v_add_u32_e32 v8, 0, v140
	v_readfirstlane_b32 s2, v9
	v_or_b32_e32 v11, v12, v11
	v_add_u32_e32 v12, 0x2000, v8
	s_mov_b32 s23, s27
	v_lshlrev_b32_e32 v10, 1, v18
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v12
	v_add_u32_e32 v12, 0xc5e0, v9
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v13, 1, v20
	s_mov_b32 m0, s2
	s_and_b32 s5, s5, 0xffff
	v_readfirstlane_b32 s2, v12
	v_add_u32_e32 v12, 0xe5e0, v8
	buffer_load_dwordx4 v13, s[20:23], 0 offen lds
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_lshlrev_b32_e32 v14, 1, v22
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v12
	s_cmp_gt_u32 s12, 32
	v_add_u32_e32 v9, 0x4200, v9
	buffer_load_dwordx4 v14, s[4:7], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v24
	s_mov_b32 m0, s2
	v_add_u32_e32 v10, 64, v10
	v_bfrev_b32_e32 v12, 1
	s_cselect_b64 vcc, -1, 0
	v_readfirstlane_b32 s2, v9
	v_add_u32_e32 v8, 0x6200, v8
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	v_cndmask_b32_e32 v10, v12, v10, vcc
	s_mov_b32 m0, s2
	v_add_u32_e32 v9, 64, v13
	v_readfirstlane_b32 s2, v8
	s_add_i32 s11, 0, 0x107e0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_cndmask_b32_e32 v9, v12, v9, vcc
	s_mov_b32 m0, s2
	v_add_u32_e32 v8, s11, v139
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_add_u32_e32 v9, 64, v14
	v_readfirstlane_b32 s2, v8
	v_cndmask_b32_e32 v9, v12, v9, vcc
	s_mov_b32 m0, s2
	v_add_u32_e32 v8, s11, v11
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	v_add_u32_e32 v9, 64, v15
	v_readfirstlane_b32 s2, v8
	v_cndmask_b32_e32 v9, v12, v9, vcc
	s_mov_b32 m0, s2
	s_movk_i32 s2, 0x100
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	v_cmp_gt_u32_e64 s[2:3], s2, v0
	v_cmp_lt_u32_e64 s[6:7], s15, v0
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[14:15], s[6:7]
	s_cbranch_execz .LBB0_3
; %bb.2:
	s_barrier
.LBB0_3:
	s_or_b64 exec, exec, s[14:15]
	v_and_b32_e32 v134, 15, v0
	v_and_b32_e32 v135, 0x100, v0
	v_and_b32_e32 v15, 0xffff0000, v6
	v_lshlrev_b32_e32 v14, 16, v6
	v_and_b32_e32 v17, 0xffff0000, v7
	v_lshlrev_b32_e32 v16, 16, v7
	v_and_b32_e32 v11, 0xffff0000, v4
	v_lshlrev_b32_e32 v10, 16, v4
	v_and_b32_e32 v13, 0xffff0000, v5
	v_lshlrev_b32_e32 v12, 16, v5
	v_and_b32_e32 v7, 0xffff0000, v2
	v_lshlrev_b32_e32 v6, 16, v2
	v_and_b32_e32 v9, 0xffff0000, v3
	v_lshlrev_b32_e32 v8, 16, v3
	v_and_b32_e32 v3, 0xffff0000, v26
	v_lshlrev_b32_e32 v2, 16, v26
	v_and_b32_e32 v5, 0xffff0000, v27
	v_lshlrev_b32_e32 v4, 16, v27
	s_cmpk_lt_u32 s12, 0x41
	v_lshlrev_b32_e32 v136, 10, v134
	v_and_b32_e32 v0, 48, v0
	v_lshrrev_b32_e32 v137, 2, v135
	v_lshlrev_b32_e32 v138, 5, v134
	s_cbranch_scc1 .LBB0_12
; %bb.4:                                ; %.lr.ph
	s_min_u32 s6, s12, 0x1000
	s_add_i32 s6, s6, 31
	s_lshr_b32 s6, s6, 5
	s_movk_i32 s7, 0x80
	v_lshl_add_u32 v141, v24, 1, s7
	v_lshl_add_u32 v142, v22, 1, s7
	v_lshl_add_u32 v143, v20, 1, s7
	v_lshl_add_u32 v144, v18, 1, s7
	s_mov_b32 s13, 0
	s_add_i32 s12, 0, 0x4200
	s_add_i32 s14, 0, 0xc5e0
	s_add_i32 s11, 0, 0x107e0
	s_add_i32 s15, s6, -2
	s_mov_b32 s17, 1
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_mov_b32_e32 v21, v5
	v_mov_b32_e32 v20, v4
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v25, v9
	v_mov_b32_e32 v24, v8
	v_mov_b32_e32 v23, v7
	v_mov_b32_e32 v22, v6
	v_mov_b32_e32 v29, v13
	v_mov_b32_e32 v28, v12
	v_mov_b32_e32 v27, v11
	v_mov_b32_e32 v26, v10
	v_mov_b32_e32 v33, v17
	v_mov_b32_e32 v32, v16
	v_mov_b32_e32 v31, v15
	v_mov_b32_e32 v30, v14
	v_mov_b32_e32 v37, v5
	v_mov_b32_e32 v36, v4
	v_mov_b32_e32 v35, v3
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v41, v9
	v_mov_b32_e32 v40, v8
	v_mov_b32_e32 v39, v7
	v_mov_b32_e32 v38, v6
	v_mov_b32_e32 v45, v13
	v_mov_b32_e32 v44, v12
	v_mov_b32_e32 v43, v11
	v_mov_b32_e32 v42, v10
	v_mov_b32_e32 v49, v17
	v_mov_b32_e32 v48, v16
	v_mov_b32_e32 v47, v15
	v_mov_b32_e32 v46, v14
	v_mov_b32_e32 v53, v5
	v_mov_b32_e32 v52, v4
	v_mov_b32_e32 v51, v3
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v57, v9
	v_mov_b32_e32 v56, v8
	v_mov_b32_e32 v55, v7
	v_mov_b32_e32 v54, v6
	v_mov_b32_e32 v61, v13
	v_mov_b32_e32 v60, v12
	v_mov_b32_e32 v59, v11
	v_mov_b32_e32 v58, v10
	v_mov_b32_e32 v65, v17
	v_mov_b32_e32 v64, v16
	v_mov_b32_e32 v63, v15
	v_mov_b32_e32 v62, v14
	v_mov_b32_e32 v69, v5
	v_mov_b32_e32 v68, v4
	v_mov_b32_e32 v67, v3
	v_mov_b32_e32 v66, v2
	v_mov_b32_e32 v73, v9
	v_mov_b32_e32 v72, v8
	v_mov_b32_e32 v71, v7
	v_mov_b32_e32 v70, v6
	v_mov_b32_e32 v77, v13
	v_mov_b32_e32 v76, v12
	v_mov_b32_e32 v75, v11
	v_mov_b32_e32 v74, v10
	v_mov_b32_e32 v81, v17
	v_mov_b32_e32 v80, v16
	v_mov_b32_e32 v79, v15
	v_mov_b32_e32 v78, v14
	v_mov_b32_e32 v85, v5
	v_mov_b32_e32 v84, v4
	v_mov_b32_e32 v83, v3
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v89, v9
	v_mov_b32_e32 v88, v8
	v_mov_b32_e32 v87, v7
	v_mov_b32_e32 v86, v6
	v_mov_b32_e32 v93, v13
	v_mov_b32_e32 v92, v12
	v_mov_b32_e32 v91, v11
	v_mov_b32_e32 v90, v10
	v_mov_b32_e32 v97, v17
	v_mov_b32_e32 v96, v16
	v_mov_b32_e32 v95, v15
	v_mov_b32_e32 v94, v14
	v_mov_b32_e32 v101, v5
	v_mov_b32_e32 v100, v4
	v_mov_b32_e32 v99, v3
	v_mov_b32_e32 v98, v2
	v_mov_b32_e32 v105, v9
	v_mov_b32_e32 v104, v8
	v_mov_b32_e32 v103, v7
	v_mov_b32_e32 v102, v6
	v_mov_b32_e32 v109, v13
	v_mov_b32_e32 v108, v12
	v_mov_b32_e32 v107, v11
	v_mov_b32_e32 v106, v10
	v_mov_b32_e32 v113, v17
	v_mov_b32_e32 v112, v16
	v_mov_b32_e32 v111, v15
	v_mov_b32_e32 v110, v14
	v_mov_b32_e32 v117, v5
	v_mov_b32_e32 v116, v4
	v_mov_b32_e32 v115, v3
	v_mov_b32_e32 v114, v2
	v_mov_b32_e32 v121, v9
	v_mov_b32_e32 v120, v8
	v_mov_b32_e32 v119, v7
	v_mov_b32_e32 v118, v6
	v_mov_b32_e32 v125, v13
	v_mov_b32_e32 v124, v12
	v_mov_b32_e32 v123, v11
	v_mov_b32_e32 v122, v10
	v_mov_b32_e32 v129, v17
	v_mov_b32_e32 v128, v16
	v_mov_b32_e32 v127, v15
	v_mov_b32_e32 v126, v14
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s19, s13
	s_mov_b32 s13, s12
	s_mov_b32 s12, s14
	s_mov_b32 s14, s11
	s_add_i32 s11, s17, 1
	s_cmp_lt_i32 s11, 3
	v_add3_u32 v145, s19, v137, v0
	v_add_u32_e32 v146, s12, v136
	s_cselect_b32 s17, s11, 0
	v_add3_u32 v145, v145, v136, v138
	v_add3_u32 v146, v146, v1, v138
	s_lshl_b32 s11, s17, 13
	ds_read_b128 v[148:151], v145
	ds_read_b128 v[152:155], v145 offset:128
	ds_read_b128 v[156:159], v145 offset:256
	ds_read_b128 v[160:163], v145 offset:384
	ds_read_b128 v[164:167], v145 offset:512
	ds_read_b128 v[168:171], v145 offset:640
	ds_read_b128 v[172:175], v145 offset:768
	ds_read_b128 v[176:179], v145 offset:896
	ds_read_b128 v[180:183], v146
	ds_read_b128 v[184:187], v146 offset:256
	ds_read_b128 v[188:191], v146 offset:512
	ds_read_b128 v[192:195], v146 offset:768
	s_ashr_i32 s12, s11, 5
	s_add_i32 s12, s12, s11
	s_lshl1_add_u32 s12, s12, 0
	s_add_i32 s11, s12, 0xc5e0
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v145, s12, v139
	v_add_u32_e32 v146, s12, v140
	v_readfirstlane_b32 s19, v145
	v_add_u32_e32 v147, 0x2000, v146
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s19, v147
	buffer_load_dwordx4 v144, s[20:23], 0 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v143, s[20:23], 0 offen lds
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v145, 0xc5e0, v145
	v_mfma_f32_16x16x32_bf16 v[126:129], v[180:183], v[148:151], v[126:129]
	v_readfirstlane_b32 s19, v145
	v_add_u32_e32 v145, 0xe5e0, v146
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s19, v145
	buffer_load_dwordx4 v142, s[4:7], 0 offen lds
	s_mov_b32 m0, s19
	v_mfma_f32_16x16x32_bf16 v[122:125], v[184:187], v[148:151], v[122:125]
	buffer_load_dwordx4 v141, s[4:7], 0 offen lds
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(3) SyncID(0)
	v_mfma_f32_16x16x32_bf16 v[118:121], v[188:191], v[148:151], v[118:121]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[192:195], v[148:151], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[110:113], v[180:183], v[152:155], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[106:109], v[184:187], v[152:155], v[106:109]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[188:191], v[152:155], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[98:101], v[192:195], v[152:155], v[98:101]
	v_mfma_f32_16x16x32_bf16 v[94:97], v[180:183], v[156:159], v[94:97]
	v_mfma_f32_16x16x32_bf16 v[90:93], v[184:187], v[156:159], v[90:93]
	v_mfma_f32_16x16x32_bf16 v[86:89], v[188:191], v[156:159], v[86:89]
	v_mfma_f32_16x16x32_bf16 v[82:85], v[192:195], v[156:159], v[82:85]
	v_mfma_f32_16x16x32_bf16 v[78:81], v[180:183], v[160:163], v[78:81]
	v_mfma_f32_16x16x32_bf16 v[74:77], v[184:187], v[160:163], v[74:77]
	v_mfma_f32_16x16x32_bf16 v[70:73], v[188:191], v[160:163], v[70:73]
	v_mfma_f32_16x16x32_bf16 v[66:69], v[192:195], v[160:163], v[66:69]
	v_mfma_f32_16x16x32_bf16 v[62:65], v[180:183], v[164:167], v[62:65]
	v_mfma_f32_16x16x32_bf16 v[58:61], v[184:187], v[164:167], v[58:61]
	v_mfma_f32_16x16x32_bf16 v[54:57], v[188:191], v[164:167], v[54:57]
	v_mfma_f32_16x16x32_bf16 v[50:53], v[192:195], v[164:167], v[50:53]
	v_mfma_f32_16x16x32_bf16 v[46:49], v[180:183], v[168:171], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[42:45], v[184:187], v[168:171], v[42:45]
	v_mfma_f32_16x16x32_bf16 v[38:41], v[188:191], v[168:171], v[38:41]
	v_mfma_f32_16x16x32_bf16 v[34:37], v[192:195], v[168:171], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[30:33], v[180:183], v[172:175], v[30:33]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[184:187], v[172:175], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[188:191], v[172:175], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[18:21], v[192:195], v[172:175], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[180:183], v[176:179], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[184:187], v[176:179], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[6:9], v[188:191], v[176:179], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[2:5], v[192:195], v[176:179], v[2:5]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s15, s15, -1
	v_add_u32_e32 v141, 64, v141
	v_add_u32_e32 v142, 64, v142
	v_add_u32_e32 v143, 64, v143
	s_cmp_lg_u32 s15, 0
	v_add_u32_e32 v144, 64, v144
	s_cbranch_scc1 .LBB0_5
; %bb.6:                                ; %Flow
	s_load_dword s4, s[0:1], 0x38
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_8
.LBB0_7:
	s_barrier
.LBB0_8:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, v136, v1
	v_add_u32_e32 v139, v1, v138
	v_add_u32_e32 v1, s14, v139
	ds_read_b128 v[140:143], v1
	ds_read_b128 v[148:151], v1 offset:256
	ds_read_b128 v[152:155], v1 offset:512
	ds_read_b128 v[156:159], v1 offset:768
	v_or3_b32 v0, v137, v0, v136
	v_add_u32_e32 v136, v0, v138
	v_add_u32_e32 v0, s13, v136
	ds_read_b128 v[144:147], v0
	s_andn2_b64 vcc, exec, vcc
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[126:129], v[140:143], v[144:147], v[126:129]
	v_mfma_f32_16x16x32_bf16 v[122:125], v[148:151], v[144:147], v[122:125]
	v_mfma_f32_16x16x32_bf16 v[118:121], v[152:155], v[144:147], v[118:121]
	v_mfma_f32_16x16x32_bf16 v[114:117], v[156:159], v[144:147], v[114:117]
	ds_read_b128 v[144:147], v0 offset:128
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[110:113], v[140:143], v[144:147], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[106:109], v[148:151], v[144:147], v[106:109]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[152:155], v[144:147], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[98:101], v[156:159], v[144:147], v[98:101]
	ds_read_b128 v[144:147], v0 offset:256
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[94:97], v[140:143], v[144:147], v[94:97]
	v_mfma_f32_16x16x32_bf16 v[90:93], v[148:151], v[144:147], v[90:93]
	v_mfma_f32_16x16x32_bf16 v[86:89], v[152:155], v[144:147], v[86:89]
	v_mfma_f32_16x16x32_bf16 v[82:85], v[156:159], v[144:147], v[82:85]
	ds_read_b128 v[144:147], v0 offset:384
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[78:81], v[140:143], v[144:147], v[78:81]
	v_mfma_f32_16x16x32_bf16 v[74:77], v[148:151], v[144:147], v[74:77]
	v_mfma_f32_16x16x32_bf16 v[70:73], v[152:155], v[144:147], v[70:73]
	v_mfma_f32_16x16x32_bf16 v[66:69], v[156:159], v[144:147], v[66:69]
	ds_read_b128 v[144:147], v0 offset:512
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[140:143], v[144:147], v[62:65]
	v_mfma_f32_16x16x32_bf16 v[58:61], v[148:151], v[144:147], v[58:61]
	v_mfma_f32_16x16x32_bf16 v[54:57], v[152:155], v[144:147], v[54:57]
	v_mfma_f32_16x16x32_bf16 v[50:53], v[156:159], v[144:147], v[50:53]
	ds_read_b128 v[144:147], v0 offset:640
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[46:49], v[140:143], v[144:147], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[42:45], v[148:151], v[144:147], v[42:45]
	v_mfma_f32_16x16x32_bf16 v[38:41], v[152:155], v[144:147], v[38:41]
	v_mfma_f32_16x16x32_bf16 v[34:37], v[156:159], v[144:147], v[34:37]
	ds_read_b128 v[144:147], v0 offset:768
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[30:33], v[140:143], v[144:147], v[30:33]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[148:151], v[144:147], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[152:155], v[144:147], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[18:21], v[156:159], v[144:147], v[18:21]
	ds_read_b128 v[144:147], v0 offset:896
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[14:17], v[140:143], v[144:147], v[14:17]
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[10:13], v[148:151], v[144:147], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[6:9], v[152:155], v[144:147], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[0:3], v[156:159], v[144:147], v[2:5]
	s_cbranch_vccnz .LBB0_10
; %bb.9:
	s_nop 1
	v_add_u32_e32 v4, s11, v139
	ds_read_b128 v[138:141], v4
	ds_read_b128 v[146:149], v4 offset:256
	ds_read_b128 v[150:153], v4 offset:512
	ds_read_b128 v[154:157], v4 offset:768
	v_add_u32_e32 v5, s12, v136
	ds_read_b128 v[142:145], v5
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[126:129], v[138:141], v[142:145], v[126:129]
	v_mfma_f32_16x16x32_bf16 v[122:125], v[146:149], v[142:145], v[122:125]
	v_mfma_f32_16x16x32_bf16 v[118:121], v[150:153], v[142:145], v[118:121]
	v_mfma_f32_16x16x32_bf16 v[114:117], v[154:157], v[142:145], v[114:117]
	ds_read_b128 v[142:145], v5 offset:128
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[110:113], v[138:141], v[142:145], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[106:109], v[146:149], v[142:145], v[106:109]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[150:153], v[142:145], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[98:101], v[154:157], v[142:145], v[98:101]
	ds_read_b128 v[142:145], v5 offset:256
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[94:97], v[138:141], v[142:145], v[94:97]
	v_mfma_f32_16x16x32_bf16 v[90:93], v[146:149], v[142:145], v[90:93]
	v_mfma_f32_16x16x32_bf16 v[86:89], v[150:153], v[142:145], v[86:89]
	v_mfma_f32_16x16x32_bf16 v[82:85], v[154:157], v[142:145], v[82:85]
	ds_read_b128 v[142:145], v5 offset:384
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[78:81], v[138:141], v[142:145], v[78:81]
	v_mfma_f32_16x16x32_bf16 v[74:77], v[146:149], v[142:145], v[74:77]
	v_mfma_f32_16x16x32_bf16 v[70:73], v[150:153], v[142:145], v[70:73]
	v_mfma_f32_16x16x32_bf16 v[66:69], v[154:157], v[142:145], v[66:69]
	ds_read_b128 v[142:145], v5 offset:512
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[138:141], v[142:145], v[62:65]
	v_mfma_f32_16x16x32_bf16 v[58:61], v[146:149], v[142:145], v[58:61]
	v_mfma_f32_16x16x32_bf16 v[54:57], v[150:153], v[142:145], v[54:57]
	v_mfma_f32_16x16x32_bf16 v[50:53], v[154:157], v[142:145], v[50:53]
	ds_read_b128 v[142:145], v5 offset:640
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[46:49], v[138:141], v[142:145], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[42:45], v[146:149], v[142:145], v[42:45]
	v_mfma_f32_16x16x32_bf16 v[38:41], v[150:153], v[142:145], v[38:41]
	v_mfma_f32_16x16x32_bf16 v[34:37], v[154:157], v[142:145], v[34:37]
	ds_read_b128 v[142:145], v5 offset:768
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[30:33], v[138:141], v[142:145], v[30:33]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[146:149], v[142:145], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[150:153], v[142:145], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[18:21], v[154:157], v[142:145], v[18:21]
	ds_read_b128 v[142:145], v5 offset:896
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[14:17], v[138:141], v[142:145], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[146:149], v[142:145], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[6:9], v[150:153], v[142:145], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[0:3], v[154:157], v[142:145], v[0:3]
.LBB0_10:
	s_nop 1
	v_lshrrev_b32_e32 v4, 4, v135
	v_or_b32_e32 v134, v4, v134
	s_ashr_i32 s19, s18, 31
	v_or_b32_e32 v135, 32, v134
	v_or_b32_e32 v136, 64, v134
	v_or_b32_e32 v137, 0x60, v134
	v_or_b32_e32 v138, 0x80, v134
	v_or_b32_e32 v139, 0xa0, v134
	v_or_b32_e32 v140, 0xc0, v134
	v_or_b32_e32 v141, 0xe0, v134
	s_lshl_b64 s[0:1], s[18:19], 8
	s_ashr_i32 s17, s16, 31
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	v_cvt_pk_bf16_f32 v26, v26, v27
	v_cvt_pk_bf16_f32 v27, v28, v29
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_cvt_pk_bf16_f32 v23, v24, v25
	v_cvt_pk_bf16_f32 v18, v18, v19
	v_cvt_pk_bf16_f32 v19, v20, v21
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_cvt_pk_bf16_f32 v15, v16, v17
	v_cvt_pk_bf16_f32 v10, v10, v11
	v_cvt_pk_bf16_f32 v11, v12, v13
	v_cvt_pk_bf16_f32 v4, v6, v7
	v_cvt_pk_bf16_f32 v5, v8, v9
	v_or_b32_e32 v6, s0, v134
	v_mov_b32_e32 v7, s1
	v_or_b32_e32 v8, s0, v135
	v_mov_b32_e32 v9, s1
	v_or_b32_e32 v12, s0, v136
	v_mov_b32_e32 v13, s1
	v_or_b32_e32 v16, s0, v137
	v_mov_b32_e32 v17, s1
	v_or_b32_e32 v20, s0, v138
	v_mov_b32_e32 v21, s1
	v_or_b32_e32 v24, s0, v139
	v_mov_b32_e32 v25, s1
	v_or_b32_e32 v28, s0, v140
	v_mov_b32_e32 v29, s1
	v_or_b32_e32 v32, s0, v141
	v_mov_b32_e32 v33, s1
	s_lshl_b64 s[0:1], s[16:17], 8
	v_cvt_pk_bf16_f32 v34, v34, v35
	v_cvt_pk_bf16_f32 v35, v36, v37
	v_or_b32_e32 v36, s0, v130
	v_mov_b32_e32 v37, s1
	s_ashr_i32 s11, s10, 31
	s_ashr_i32 s29, s28, 31
	v_cvt_pk_bf16_f32 v46, v46, v47
	v_cvt_pk_bf16_f32 v47, v48, v49
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_cvt_pk_bf16_f32 v1, v2, v3
	v_or_b32_e32 v2, s0, v131
	v_mov_b32_e32 v3, s1
	v_mul_lo_u32 v48, v6, s4
	v_cmp_gt_i64_e64 s[22:23], s[10:11], v[6:7]
	v_cmp_gt_i64_e64 s[6:7], s[28:29], v[36:37]
	v_cvt_pk_bf16_f32 v42, v42, v43
	v_cvt_pk_bf16_f32 v43, v44, v45
	v_cvt_pk_bf16_f32 v38, v38, v39
	v_cvt_pk_bf16_f32 v39, v40, v41
	v_or_b32_e32 v40, s0, v133
	v_mov_b32_e32 v41, s1
	v_or_b32_e32 v44, s0, v132
	v_mov_b32_e32 v45, s1
	v_cmp_gt_i64_e64 s[0:1], s[28:29], v[2:3]
	v_add_lshl_u32 v3, v36, v48, 1
	v_bfrev_b32_e32 v6, 1
	s_and_b64 s[26:27], s[22:23], s[6:7]
	v_cvt_pk_bf16_f32 v126, v126, v127
	v_cvt_pk_bf16_f32 v127, v128, v129
	v_cvt_pk_bf16_f32 v58, v58, v59
	v_cvt_pk_bf16_f32 v59, v60, v61
	v_cvt_pk_bf16_f32 v54, v54, v55
	v_cvt_pk_bf16_f32 v55, v56, v57
	v_cvt_pk_bf16_f32 v50, v50, v51
	v_cvt_pk_bf16_f32 v51, v52, v53
	v_mul_lo_u32 v49, v8, s4
	v_mul_lo_u32 v52, v12, s4
	v_mul_lo_u32 v53, v16, s4
	v_mul_lo_u32 v56, v20, s4
	v_mul_lo_u32 v57, v24, s4
	v_mul_lo_u32 v60, v28, s4
	v_mul_lo_u32 v61, v32, s4
	v_cmp_gt_i64_e64 s[24:25], s[10:11], v[8:9]
	v_cmp_gt_i64_e64 s[20:21], s[10:11], v[12:13]
	v_cmp_gt_i64_e64 s[18:19], s[10:11], v[16:17]
	v_cmp_gt_i64_e64 s[16:17], s[10:11], v[20:21]
	v_cmp_gt_i64_e64 s[14:15], s[10:11], v[24:25]
	v_cmp_gt_i64_e64 s[12:13], s[10:11], v[28:29]
	v_cmp_gt_i64_e32 vcc, s[10:11], v[32:33]
	v_cmp_gt_i64_e64 s[4:5], s[28:29], v[40:41]
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_cndmask_b32_e64 v3, v6, v3, s[26:27]
	buffer_store_dwordx2 v[126:127], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v40, v48, 1
	s_and_b64 s[26:27], s[22:23], s[4:5]
	v_cvt_pk_bf16_f32 v122, v122, v123
	v_cvt_pk_bf16_f32 v123, v124, v125
	v_cmp_gt_i64_e64 s[2:3], s[28:29], v[44:45]
	v_cndmask_b32_e64 v3, v6, v3, s[26:27]
	buffer_store_dwordx2 v[122:123], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v44, v48, 1
	s_and_b64 s[26:27], s[22:23], s[2:3]
	v_cvt_pk_bf16_f32 v118, v118, v119
	v_cvt_pk_bf16_f32 v119, v120, v121
	v_cndmask_b32_e64 v3, v6, v3, s[26:27]
	buffer_store_dwordx2 v[118:119], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v2, v48, 1
	s_and_b64 s[22:23], s[22:23], s[0:1]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	v_cndmask_b32_e64 v3, v6, v3, s[22:23]
	buffer_store_dwordx2 v[114:115], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v36, v49, 1
	s_and_b64 s[22:23], s[24:25], s[6:7]
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	v_cndmask_b32_e64 v3, v6, v3, s[22:23]
	buffer_store_dwordx2 v[110:111], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v40, v49, 1
	s_and_b64 s[22:23], s[24:25], s[4:5]
	v_cvt_pk_bf16_f32 v106, v106, v107
	v_cvt_pk_bf16_f32 v107, v108, v109
	v_cndmask_b32_e64 v3, v6, v3, s[22:23]
	buffer_store_dwordx2 v[106:107], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v44, v49, 1
	s_and_b64 s[22:23], s[24:25], s[2:3]
	v_cvt_pk_bf16_f32 v102, v102, v103
	v_cvt_pk_bf16_f32 v103, v104, v105
	v_cndmask_b32_e64 v3, v6, v3, s[22:23]
	buffer_store_dwordx2 v[102:103], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v2, v49, 1
	s_and_b64 s[22:23], s[24:25], s[0:1]
	v_cvt_pk_bf16_f32 v98, v98, v99
	v_cvt_pk_bf16_f32 v99, v100, v101
	v_cndmask_b32_e64 v3, v6, v3, s[22:23]
	buffer_store_dwordx2 v[98:99], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v36, v52, 1
	s_and_b64 s[22:23], s[20:21], s[6:7]
	v_cvt_pk_bf16_f32 v94, v94, v95
	v_cvt_pk_bf16_f32 v95, v96, v97
	v_cndmask_b32_e64 v3, v6, v3, s[22:23]
	buffer_store_dwordx2 v[94:95], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v40, v52, 1
	s_and_b64 s[22:23], s[20:21], s[4:5]
	v_cvt_pk_bf16_f32 v90, v90, v91
	v_cvt_pk_bf16_f32 v91, v92, v93
	v_cndmask_b32_e64 v3, v6, v3, s[22:23]
	buffer_store_dwordx2 v[90:91], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v44, v52, 1
	s_and_b64 s[22:23], s[20:21], s[2:3]
	v_cvt_pk_bf16_f32 v86, v86, v87
	v_cvt_pk_bf16_f32 v87, v88, v89
	v_cndmask_b32_e64 v3, v6, v3, s[22:23]
	buffer_store_dwordx2 v[86:87], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v2, v52, 1
	s_and_b64 s[20:21], s[20:21], s[0:1]
	v_cvt_pk_bf16_f32 v82, v82, v83
	v_cvt_pk_bf16_f32 v83, v84, v85
	v_cndmask_b32_e64 v3, v6, v3, s[20:21]
	buffer_store_dwordx2 v[82:83], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v36, v53, 1
	s_and_b64 s[20:21], s[18:19], s[6:7]
	v_cvt_pk_bf16_f32 v78, v78, v79
	v_cvt_pk_bf16_f32 v79, v80, v81
	v_cndmask_b32_e64 v3, v6, v3, s[20:21]
	buffer_store_dwordx2 v[78:79], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v40, v53, 1
	s_and_b64 s[20:21], s[18:19], s[4:5]
	v_cvt_pk_bf16_f32 v74, v74, v75
	v_cvt_pk_bf16_f32 v75, v76, v77
	v_cndmask_b32_e64 v3, v6, v3, s[20:21]
	buffer_store_dwordx2 v[74:75], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v44, v53, 1
	s_and_b64 s[20:21], s[18:19], s[2:3]
	v_cvt_pk_bf16_f32 v70, v70, v71
	v_cvt_pk_bf16_f32 v71, v72, v73
	v_cndmask_b32_e64 v3, v6, v3, s[20:21]
	buffer_store_dwordx2 v[70:71], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v2, v53, 1
	s_and_b64 s[18:19], s[18:19], s[0:1]
	v_cvt_pk_bf16_f32 v66, v66, v67
	v_cvt_pk_bf16_f32 v67, v68, v69
	v_cndmask_b32_e64 v3, v6, v3, s[18:19]
	buffer_store_dwordx2 v[66:67], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v36, v56, 1
	s_and_b64 s[18:19], s[16:17], s[6:7]
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cndmask_b32_e64 v3, v6, v3, s[18:19]
	buffer_store_dwordx2 v[62:63], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v40, v56, 1
	s_and_b64 s[18:19], s[16:17], s[4:5]
	v_cndmask_b32_e64 v3, v6, v3, s[18:19]
	buffer_store_dwordx2 v[58:59], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v44, v56, 1
	s_and_b64 s[18:19], s[16:17], s[2:3]
	v_cndmask_b32_e64 v3, v6, v3, s[18:19]
	buffer_store_dwordx2 v[54:55], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v2, v56, 1
	s_and_b64 s[16:17], s[16:17], s[0:1]
	v_cndmask_b32_e64 v3, v6, v3, s[16:17]
	buffer_store_dwordx2 v[50:51], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v36, v57, 1
	s_and_b64 s[16:17], s[14:15], s[6:7]
	v_cndmask_b32_e64 v3, v6, v3, s[16:17]
	buffer_store_dwordx2 v[46:47], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v40, v57, 1
	s_and_b64 s[16:17], s[14:15], s[4:5]
	v_cndmask_b32_e64 v3, v6, v3, s[16:17]
	buffer_store_dwordx2 v[42:43], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v44, v57, 1
	s_and_b64 s[16:17], s[14:15], s[2:3]
	v_cndmask_b32_e64 v3, v6, v3, s[16:17]
	buffer_store_dwordx2 v[38:39], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v2, v57, 1
	s_and_b64 s[14:15], s[14:15], s[0:1]
	v_cndmask_b32_e64 v3, v6, v3, s[14:15]
	buffer_store_dwordx2 v[34:35], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v36, v60, 1
	s_and_b64 s[14:15], s[12:13], s[6:7]
	v_cndmask_b32_e64 v3, v6, v3, s[14:15]
	buffer_store_dwordx2 v[30:31], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v40, v60, 1
	s_and_b64 s[14:15], s[12:13], s[4:5]
	v_cndmask_b32_e64 v3, v6, v3, s[14:15]
	buffer_store_dwordx2 v[26:27], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v44, v60, 1
	s_and_b64 s[14:15], s[12:13], s[2:3]
	v_cndmask_b32_e64 v3, v6, v3, s[14:15]
	buffer_store_dwordx2 v[22:23], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v2, v60, 1
	s_and_b64 s[12:13], s[12:13], s[0:1]
	v_cndmask_b32_e64 v3, v6, v3, s[12:13]
	buffer_store_dwordx2 v[18:19], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v36, v61, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cndmask_b32_e64 v3, v6, v3, s[6:7]
	buffer_store_dwordx2 v[14:15], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v40, v61, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cndmask_b32_e64 v3, v6, v3, s[4:5]
	buffer_store_dwordx2 v[10:11], v3, s[8:11], 0 offen
	v_add_lshl_u32 v3, v44, v61, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_add_lshl_u32 v2, v2, v61, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e64 v3, v6, v3, s[2:3]
	v_cndmask_b32_e32 v2, v6, v2, vcc
	buffer_store_dwordx2 v[4:5], v3, s[8:11], 0 offen
	buffer_store_dwordx2 v[0:1], v2, s[8:11], 0 offen
.LBB0_11:
	s_endpgm
.LBB0_12:
	s_mov_b32 s13, 0
	s_add_i32 s12, 0, 0x4200
	s_add_i32 s14, 0, 0xc5e0
	v_mov_b32_e32 v21, v5
	v_mov_b32_e32 v20, v4
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v25, v9
	v_mov_b32_e32 v24, v8
	v_mov_b32_e32 v23, v7
	v_mov_b32_e32 v22, v6
	v_mov_b32_e32 v29, v13
	v_mov_b32_e32 v28, v12
	v_mov_b32_e32 v27, v11
	v_mov_b32_e32 v26, v10
	v_mov_b32_e32 v33, v17
	v_mov_b32_e32 v32, v16
	v_mov_b32_e32 v31, v15
	v_mov_b32_e32 v30, v14
	v_mov_b32_e32 v37, v5
	v_mov_b32_e32 v36, v4
	v_mov_b32_e32 v35, v3
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v41, v9
	v_mov_b32_e32 v40, v8
	v_mov_b32_e32 v39, v7
	v_mov_b32_e32 v38, v6
	v_mov_b32_e32 v45, v13
	v_mov_b32_e32 v44, v12
	v_mov_b32_e32 v43, v11
	v_mov_b32_e32 v42, v10
	v_mov_b32_e32 v49, v17
	v_mov_b32_e32 v48, v16
	v_mov_b32_e32 v47, v15
	v_mov_b32_e32 v46, v14
	v_mov_b32_e32 v53, v5
	v_mov_b32_e32 v52, v4
	v_mov_b32_e32 v51, v3
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v57, v9
	v_mov_b32_e32 v56, v8
	v_mov_b32_e32 v55, v7
	v_mov_b32_e32 v54, v6
	v_mov_b32_e32 v61, v13
	v_mov_b32_e32 v60, v12
	v_mov_b32_e32 v59, v11
	v_mov_b32_e32 v58, v10
	v_mov_b32_e32 v65, v17
	v_mov_b32_e32 v64, v16
	v_mov_b32_e32 v63, v15
	v_mov_b32_e32 v62, v14
	v_mov_b32_e32 v69, v5
	v_mov_b32_e32 v68, v4
	v_mov_b32_e32 v67, v3
	v_mov_b32_e32 v66, v2
	v_mov_b32_e32 v73, v9
	v_mov_b32_e32 v72, v8
	v_mov_b32_e32 v71, v7
	v_mov_b32_e32 v70, v6
	v_mov_b32_e32 v77, v13
	v_mov_b32_e32 v76, v12
	v_mov_b32_e32 v75, v11
	v_mov_b32_e32 v74, v10
	v_mov_b32_e32 v81, v17
	v_mov_b32_e32 v80, v16
	v_mov_b32_e32 v79, v15
	v_mov_b32_e32 v78, v14
	v_mov_b32_e32 v85, v5
	v_mov_b32_e32 v84, v4
	v_mov_b32_e32 v83, v3
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v89, v9
	v_mov_b32_e32 v88, v8
	v_mov_b32_e32 v87, v7
	v_mov_b32_e32 v86, v6
	v_mov_b32_e32 v93, v13
	v_mov_b32_e32 v92, v12
	v_mov_b32_e32 v91, v11
	v_mov_b32_e32 v90, v10
	v_mov_b32_e32 v97, v17
	v_mov_b32_e32 v96, v16
	v_mov_b32_e32 v95, v15
	v_mov_b32_e32 v94, v14
	v_mov_b32_e32 v101, v5
	v_mov_b32_e32 v100, v4
	v_mov_b32_e32 v99, v3
	v_mov_b32_e32 v98, v2
	v_mov_b32_e32 v105, v9
	v_mov_b32_e32 v104, v8
	v_mov_b32_e32 v103, v7
	v_mov_b32_e32 v102, v6
	v_mov_b32_e32 v109, v13
	v_mov_b32_e32 v108, v12
	v_mov_b32_e32 v107, v11
	v_mov_b32_e32 v106, v10
	v_mov_b32_e32 v113, v17
	v_mov_b32_e32 v112, v16
	v_mov_b32_e32 v111, v15
	v_mov_b32_e32 v110, v14
	v_mov_b32_e32 v117, v5
	v_mov_b32_e32 v116, v4
	v_mov_b32_e32 v115, v3
	v_mov_b32_e32 v114, v2
	v_mov_b32_e32 v121, v9
	v_mov_b32_e32 v120, v8
	v_mov_b32_e32 v119, v7
	v_mov_b32_e32 v118, v6
	v_mov_b32_e32 v125, v13
	v_mov_b32_e32 v124, v12
	v_mov_b32_e32 v123, v11
	v_mov_b32_e32 v122, v10
	v_mov_b32_e32 v129, v17
	v_mov_b32_e32 v128, v16
	v_mov_b32_e32 v127, v15
	v_mov_b32_e32 v126, v14
	s_load_dword s4, s[0:1], 0x38
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_7
	s_branch .LBB0_8
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
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
		.amdhsa_next_free_vgpr 196
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 196
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
	.size	_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0, .Lfunc_end0-_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
	.cfi_endproc
                                        ; -- End function
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.num_vgpr, 196
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.num_agpr, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.numbered_sgpr, 30
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.num_named_barrier, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.private_seg_size, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.uses_vcc, 1
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.uses_flat_scratch, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.has_dyn_sized_stack, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.has_recursion, 0
	.set _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6204
; TotalNumSgprs: 36
; NumVgprs: 196
; NumAgprs: 0
; TotalNumVgprs: 196
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 196
; AccumOffset: 196
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 48
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
	.quad	.Ltmp19                         ; DW_AT_low_pc
	.long	.Ltmp20-.Ltmp19                 ; DW_AT_high_pc
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
	.asciz	"_gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0" ; string offset=78
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
    .max_flat_workgroup_size: 512
    .name:           _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0
    .private_segment_fixed_size: 0
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         _gemm_a16_w16_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_32_GROUP_SIZE_M_16_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_GRID_MN_256_cache_modifier_NONE_activation_NONE_use_activation_0_ADD_BIAS_1_SKIP_REDUCE_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     196
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
