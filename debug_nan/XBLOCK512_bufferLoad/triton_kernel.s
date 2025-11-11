	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	triton_kernel                   ; -- Begin function triton_kernel
	.p2align	8
	.type	triton_kernel,@function
triton_kernel:                          ; @triton_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.1:
	.file	1 "/var/lib/jenkins/OAI-triton" "test_load_nan.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.2:
.LBB0_0:
	s_mov_b64 s[0:1], s[4:5]
	s_lshl_b32 s4, s14, 9
	s_mov_b64 s[12:13], s[2:3]
	v_lshl_or_b32 v1, v0, 1, s4
	s_mov_b32 s2, 0x66666667
	v_mul_hi_i32 v2, v1, s2
	v_lshrrev_b32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v2, 8, v2
	v_add_u32_e32 v2, v2, v3
	v_lshlrev_b32_e32 v3, 1, v1
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_lshl_add_u32 v3, v2, 9, v3
	v_bfrev_b32_e32 v4, 1
	v_cmp_gt_i32_e32 vcc, s10, v1
	v_lshlrev_b32_e32 v2, 3, v2
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, s14
	s_mov_b32 s3, s15
	v_cndmask_b32_e32 v2, v4, v2, vcc
	v_cndmask_b32_e32 v1, v4, v3, vcc
	buffer_load_dwordx2 v[2:3], v2, s[0:3], 0 offen
	s_and_b32 s13, s13, 0xffff
	buffer_load_dword v5, v1, s[12:15], 0 offen
	v_lshlrev_b32_e32 v9, 2, v0
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[0:1], s[4:5], 1
	s_add_u32 s12, s6, s0
	s_mov_b32 s2, 0x800000
	s_addc_u32 s0, s7, s1
	s_and_b32 s13, s0, 0xffff
	s_mov_b32 s3, 0x3f317217
	s_mov_b32 s8, 0x7f800000
	v_mov_b32_e32 v7, 0x41b17218
	v_mov_b32_e32 v6, 1.0
	s_movk_i32 s9, 0x7fff
	v_mov_b32_e32 v8, 0x7fff
	s_mov_b32 s10, 0x5040100
	s_waitcnt vmcnt(1)
	v_xor_b32_e32 v0, v2, v3
	v_ffbh_i32_e32 v1, v3
	v_ashrrev_i32_e32 v0, 31, v0
	v_add_u32_e32 v1, -1, v1
	v_add_u32_e32 v0, 32, v0
	v_min_u32_e32 v10, v1, v0
	v_lshlrev_b64 v[0:1], v10, v[2:3]
	v_min_u32_e32 v0, 1, v0
	v_or_b32_e32 v0, v1, v0
	v_cvt_f32_i32_e32 v0, v0
	v_sub_u32_e32 v1, 32, v10
	v_ldexp_f32 v0, v0, v1
	v_add_f32_e32 v0, 1.0, v0
	v_mul_f32_e32 v0, 0x39000000, v0
	v_floor_f32_e32 v0, v0
	v_add_f32_e32 v0, 1.0, v0
	v_cmp_gt_f32_e64 s[0:1], s2, v0
	s_nop 1
	v_cndmask_b32_e64 v1, 0, 32, s[0:1]
	v_ldexp_f32 v0, v0, v1
	v_log_f32_e32 v2, v0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffff0000, v5
	v_lshlrev_b32_e32 v0, 16, v5
	v_cndmask_b32_e64 v3, 0, v7, s[0:1]
	v_mul_f32_e32 v5, 0x3f317217, v2
	v_fma_f32 v5, v2, s3, -v5
	v_fmamk_f32 v5, v2, 0x3377d1cf, v5
	v_fmac_f32_e32 v5, 0x3f317217, v2
	v_cmp_lt_f32_e64 s[0:1], |v2|, s8
	s_nop 1
	v_cndmask_b32_e64 v2, v2, v5, s[0:1]
	v_sub_f32_e32 v2, v2, v3
	v_fmamk_f32 v2, v2, 0x3dcccccd, v6
	v_pk_mul_f32 v[0:1], v[2:3], v[0:1] op_sel_hi:[0,1]
	v_bfe_u32 v2, v0, 16, 1
	v_bfe_u32 v3, v1, 16, 1
	v_add3_u32 v2, v0, v2, s9
	v_add3_u32 v3, v1, v3, s9
	v_lshrrev_b32_e32 v2, 16, v2
	v_cmp_o_f32_e64 s[0:1], v0, v0
	v_lshrrev_b32_e32 v3, 16, v3
	s_nop 0
	v_cndmask_b32_e64 v0, v8, v2, s[0:1]
	v_cmp_o_f32_e64 s[0:1], v1, v1
	s_nop 1
	v_cndmask_b32_e64 v1, v8, v3, s[0:1]
	v_perm_b32 v0, v1, v0, s10
	v_cndmask_b32_e32 v1, v4, v9, vcc
	buffer_store_dword v0, v1, s[12:15], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel triton_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 48
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 12
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 11
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 12
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
	.size	triton_kernel, .Lfunc_end0-triton_kernel
	.cfi_endproc
                                        ; -- End function
	.set triton_kernel.num_vgpr, 11
	.set triton_kernel.num_agpr, 0
	.set triton_kernel.numbered_sgpr, 16
	.set triton_kernel.private_seg_size, 0
	.set triton_kernel.uses_vcc, 1
	.set triton_kernel.uses_flat_scratch, 0
	.set triton_kernel.has_dyn_sized_stack, 0
	.set triton_kernel.has_recursion, 0
	.set triton_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 788
; TotalNumSgprs: 22
; NumVgprs: 11
; NumAgprs: 0
; TotalNumVgprs: 11
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 11
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
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
	.byte	0                               ; DW_CHILDREN_no
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"test_load_nan.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton"   ; string offset=24
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
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name:           triton_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         triton_kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     11
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
