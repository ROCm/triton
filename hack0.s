	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_kernel                   ; -- Begin function matmul_kernel
	.p2align	8
	.type	matmul_kernel,@function
matmul_kernel:                          ; @matmul_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.8:
	.file	1 "/var/lib/jenkins/OAI-triton/issues" "cache_swizzling.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.9:
.LBB0_0:
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 15
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 28
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 4
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s18, 0, s1
	s_add_i32 s14, s8, 15
	s_abs_i32 s17, s16
	v_rcp_iflag_f32_e32 v1, v1
	s_ashr_i32 s15, s14, 31
	s_lshr_b32 s15, s15, 28
	s_add_i32 s14, s14, s15
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s15, s16, s0
	s_ashr_i32 s14, s14, 4
	s_ashr_i32 s15, s15, 31
	v_readfirstlane_b32 s19, v1
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s17, s19
	s_mul_i32 s19, s18, s1
	s_sub_i32 s17, s17, s19
	s_add_i32 s19, s18, 1
	s_sub_i32 s20, s17, s1
	s_cmp_ge_u32 s17, s1
	s_cselect_b32 s18, s19, s18
	s_cselect_b32 s17, s20, s17
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s17, s1
	s_cselect_b32 s1, s19, s18
	s_xor_b32 s1, s1, s15
	s_sub_i32 s1, s1, s15
	s_sub_i32 s14, s14, s1
	s_min_i32 s14, s14, 1
	s_abs_i32 s15, s14
	v_cvt_f32_u32_e32 v1, s15
	s_sub_i32 s18, 0, s15
	s_mul_i32 s0, s1, s0
	s_sub_i32 s16, s16, s0
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s17, s16
	s_xor_b32 s0, s16, s14
	s_ashr_i32 s0, s0, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s19, v1
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s17, s19
	s_mul_i32 s19, s18, s15
	s_sub_i32 s17, s17, s19
	s_add_i32 s19, s18, 1
	s_sub_i32 s20, s17, s15
	s_cmp_ge_u32 s17, s15
	s_cselect_b32 s18, s19, s18
	s_cselect_b32 s17, s20, s17
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s17, s15
	s_cselect_b32 s15, s19, s18
	s_xor_b32 s15, s15, s0
	s_sub_i32 s0, s15, s0
	s_mul_i32 s14, s0, s14
	s_sub_i32 s14, s16, s14
	s_add_i32 s1, s14, s1
	v_and_b32_e32 v1, 48, v0
	s_add_i32 s10, s10, 15
	s_cmp_gt_i32 s10, 15
	v_lshrrev_b32_e32 v6, 2, v1
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	v_lshrrev_b32_e32 v2, 2, v1
	s_mov_b64 s[14:15], 0
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b64 s[14:15], -1
                                        ; implicit-def: $vgpr2
.LBB0_3:                                ; %Flow
	s_lshl_b32 s1, s1, 4
	v_and_b32_e32 v7, 15, v0
	s_lshl_b32 s0, s0, 4
	v_mov_b32_e32 v3, 0
	s_andn2_b64 vcc, exec, s[14:15]
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	s_cbranch_vccnz .LBB0_7
; %bb.4:                                ; %.lr.ph
	s_abs_i32 s15, s9
	v_cvt_f32_u32_e32 v2, s15
	v_lshlrev_b32_e32 v3, 2, v0
	v_and_b32_e32 v4, 12, v3
	s_sub_i32 s17, 0, s15
	v_rcp_iflag_f32_e32 v2, v2
	v_or_b32_e32 v3, s0, v4
	s_ashr_i32 s16, s0, 31
	v_add_u32_e32 v3, s16, v3
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_xor_b32_e32 v3, s16, v3
	s_ashr_i32 s14, s10, 31
	s_lshr_b32 s14, s14, 28
	v_mul_lo_u32 v5, s17, v2
	v_mul_hi_u32 v5, v2, v5
	v_add_u32_e32 v2, v2, v5
	v_mul_hi_u32 v2, v3, v2
	v_mul_lo_u32 v2, v2, s15
	v_sub_u32_e32 v2, v3, v2
	v_subrev_u32_e32 v3, s15, v2
	v_cmp_le_u32_e32 vcc, s15, v2
	v_lshrrev_b32_e32 v5, 2, v0
	s_add_i32 s10, s10, s14
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s15, v2
	v_cmp_le_u32_e32 vcc, s15, v2
	s_abs_i32 s15, s8
	s_lshl_b32 s14, s12, 4
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_cvt_f32_u32_e32 v3, s15
	v_xor_b32_e32 v2, s16, v2
	v_subrev_u32_e32 v2, s16, v2
	v_lshlrev_b32_e32 v10, 3, v0
	v_rcp_iflag_f32_e32 v8, v3
	v_mad_u64_u32 v[2:3], s[16:17], s12, v5, v[2:3]
	v_or_b32_e32 v3, s1, v5
	v_mul_f32_e32 v5, 0x4f7ffffe, v8
	v_cvt_u32_f32_e32 v5, v5
	s_sub_i32 s16, 0, s15
	s_ashr_i32 s12, s1, 31
	v_add_u32_e32 v3, s12, v3
	v_mul_lo_u32 v8, s16, v5
	v_mul_hi_u32 v8, v5, v8
	v_xor_b32_e32 v3, s12, v3
	v_add_u32_e32 v5, v5, v8
	v_mul_hi_u32 v5, v3, v5
	v_mul_lo_u32 v5, v5, s15
	v_sub_u32_e32 v3, v3, v5
	v_subrev_u32_e32 v5, s15, v3
	v_cmp_le_u32_e32 vcc, s15, v3
	v_lshlrev_b32_e32 v11, 1, v7
	v_lshl_add_u32 v1, v1, 3, 0
	v_cndmask_b32_e32 v3, v3, v5, vcc
	v_subrev_u32_e32 v5, s15, v3
	v_cmp_le_u32_e32 vcc, s15, v3
	s_ashr_i32 s15, s14, 31
	v_mov_b32_e32 v0, 0
	v_cndmask_b32_e32 v3, v3, v5, vcc
	v_xor_b32_e32 v3, s12, v3
	v_subrev_u32_e32 v3, s12, v3
	v_mul_lo_u32 v3, v3, s11
	v_add_lshl_u32 v8, v3, v4, 1
	v_ashrrev_i32_e32 v3, 31, v2
	s_ashr_i32 s10, s10, 4
	v_lshl_or_b32 v9, v7, 4, v6
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[4:5]
	s_lshl_b64 s[4:5], s[14:15], 1
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	v_add_u32_e32 v10, 0, v10
	v_add_u32_e32 v11, v1, v11
	s_mov_b32 s12, 0x5040100
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1

        s_and_b32 s14, s3, 0xffff
	s_mov_b32 s16, s2


	s_and_b32 s11, s11, 0x3fff
        s_mov_b32 s11, 0x00000800
        s_bitset1_b32 s11, 14
	s_lshl_b32 s11, s11, 16

	s_or_b32 s17, s14, s11


        
	buffer_load_dwordx2 v[14:15], v8, s[16:19], 0 offen

        s_mov_b32 s11, 0x00000200
        s_bitset1_b32 s11, 14
	s_lshl_b32 s11, s11, 16

	s_or_b32 s17, s14, s11

	;; buffer_load_dwordx2 v[12:13], v8, s[16:19], 0 offen
        global_load_dwordx2 v[12:13], v[4:5], off
	; wave barrier
	s_add_u32 s2, s2, 32
	s_addc_u32 s3, s3, 0
	s_add_i32 s10, s10, -1
	s_cmp_lg_u32 s10, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[4:5]
	s_waitcnt vmcnt(0)
	ds_write_b64 v10, v[12:13]
	s_waitcnt lgkmcnt(0)
	; wave barrier
	ds_read_u16 v16, v11
	ds_read_u16 v17, v11 offset:64
	ds_read_u16 v18, v11 offset:96
	ds_read_u16 v19, v11 offset:32
	ds_bpermute_b32 v12, v9, v14
	ds_bpermute_b32 v13, v9, v15
	s_waitcnt lgkmcnt(3)
	v_perm_b32 v15, v18, v17, s12
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v14, v19, v16, s12
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x16_f16 v[0:3], v[14:15], v[12:13], v[0:3]
	s_cbranch_scc1 .LBB0_5
; %bb.6:                                ; %._crit_edge.loopexit
	s_nop 5
	v_cvt_f16_f32_e32 v5, v0
	v_cvt_f16_f32_e32 v4, v1
	v_cvt_f16_f32_e32 v8, v2
	v_cvt_f16_f32_e32 v3, v3
	v_mov_b32_e32 v2, v6
.LBB0_7:                                ; %._crit_edge
	s_mul_i32 s2, s1, s13
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s2, s6, s2
	v_or_b32_e32 v1, s1, v7
	s_addc_u32 s3, s7, s3
	s_ashr_i32 s1, s0, 31
	v_or_b32_e32 v0, s0, v2
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s4, s2, s0
	s_addc_u32 s2, s3, s1
	s_and_b32 s3, s13, 0x3fff
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_mul_lo_u32 v6, s13, v7
	v_cmp_gt_i32_e32 vcc, s8, v1
	v_cmp_gt_i32_e64 s[0:1], s9, v0
	s_or_b32 s5, s2, s3
	s_mov_b32 s2, 0x5040100
	v_perm_b32 v1, v3, v8, s2
	v_add_lshl_u32 v2, v6, v2, 1
	v_bfrev_b32_e32 v3, 1
	s_and_b64 vcc, vcc, s[0:1]
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_perm_b32 v0, v4, v5, s2
	v_cndmask_b32_e32 v2, v3, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[4:7], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
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
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 21
		.amdhsa_accum_offset 20
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
	.size	matmul_kernel, .Lfunc_end0-matmul_kernel
	.cfi_endproc
                                        ; -- End function
	.set matmul_kernel.num_vgpr, 20
	.set matmul_kernel.num_agpr, 0
	.set matmul_kernel.numbered_sgpr, 21
	.set matmul_kernel.private_seg_size, 0
	.set matmul_kernel.uses_vcc, 1
	.set matmul_kernel.uses_flat_scratch, 0
	.set matmul_kernel.has_dyn_sized_stack, 0
	.set matmul_kernel.has_recursion, 0
	.set matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1368
; TotalNumSgprs: 27
; NumVgprs: 20
; NumAgprs: 0
; TotalNumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 27
; NumVGPRsForWavesPerEU: 20
; AccumOffset: 20
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 4
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
	.byte	1                               ; Abbrev [1] 0xb:0x64 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x3e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_call_file
	.byte	45                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	44                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x61:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	90                              ; DW_AT_call_line
	.byte	33                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"cache_swizzling.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton/issues" ; string offset=26
.Linfo_string3:
	.asciz	"matmul_kernel"                 ; string offset=61
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
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
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
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 64
    .name:           matmul_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     27
    .sgpr_spill_count: 0
    .symbol:         matmul_kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     20
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
