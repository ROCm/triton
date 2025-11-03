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
; %bb.13:
	.file	1 "/var/lib/jenkins/OAI-triton" "test_load_nan.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.14:
.LBB0_0:
	s_lshl_b32 s0, s14, 10
	v_lshl_or_b32 v0, v0, 2, s0
	s_mov_b32 s0, 0x66666667
	v_mul_hi_i32 v1, v0, s0
	v_lshrrev_b32_e32 v2, 31, v1
	v_ashrrev_i32_e32 v1, 8, v1
	v_cmp_gt_i32_e32 vcc, s10, v0
	v_add_u32_e32 v6, v1, v2
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_lshl_add_u32 v2, v6, 8, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[2:3]
	global_load_dwordx2 v[8:9], v[2:3], off
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffff0000, v8
	v_and_b32_e32 v5, 0xffff0000, v9
	v_lshlrev_b32_e32 v2, 16, v8
	v_lshlrev_b32_e32 v4, 16, v9
.LBB0_2:
	s_or_b64 exec, exec, s[0:1]
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[14:15], v[6:7], 3, s[4:5]
	v_mov_b64_e32 v[10:11], 0
	v_mov_b64_e32 v[12:13], 0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	global_load_dwordx2 v[12:13], v[14:15], off
.LBB0_4:
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	global_load_dwordx2 v[10:11], v[14:15], off
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	v_mov_b64_e32 v[6:7], 0
	v_mov_b64_e32 v[8:9], 0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execnz .LBB0_10
; %bb.7:
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execnz .LBB0_11
.LBB0_8:
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execnz .LBB0_12
.LBB0_9:
	s_endpgm
.LBB0_10:
	global_load_dwordx2 v[8:9], v[14:15], off
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_8
.LBB0_11:
	global_load_dwordx2 v[6:7], v[14:15], off
	s_or_b64 exec, exec, s[0:1]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_9
.LBB0_12:
	s_waitcnt vmcnt(0)
	v_xor_b32_e32 v14, v12, v13
	v_ffbh_i32_e32 v1, v13
	v_ashrrev_i32_e32 v14, 31, v14
	v_add_u32_e32 v1, -1, v1
	v_add_u32_e32 v14, 32, v14
	v_min_u32_e32 v1, v1, v14
	v_lshlrev_b64 v[12:13], v1, v[12:13]
	v_min_u32_e32 v12, 1, v12
	v_or_b32_e32 v12, v13, v12
	v_cvt_f32_i32_e32 v12, v12
	v_sub_u32_e32 v1, 32, v1
	v_xor_b32_e32 v14, v10, v11
	v_ffbh_i32_e32 v13, v11
	v_ldexp_f32 v1, v12, v1
	v_add_f32_e32 v1, 1.0, v1
	v_ashrrev_i32_e32 v14, 31, v14
	v_mul_f32_e32 v1, 0x39000000, v1
	v_add_u32_e32 v13, -1, v13
	v_add_u32_e32 v14, 32, v14
	v_floor_f32_e32 v1, v1
	v_min_u32_e32 v13, v13, v14
	v_add_f32_e32 v1, 1.0, v1
	s_mov_b32 s4, 0x800000
	v_lshlrev_b64 v[10:11], v13, v[10:11]
	v_cmp_gt_f32_e32 vcc, s4, v1
	v_min_u32_e32 v10, 1, v10
	v_or_b32_e32 v10, v11, v10
	v_cndmask_b32_e64 v12, 0, 32, vcc
	v_ldexp_f32 v1, v1, v12
	v_cvt_f32_i32_e32 v10, v10
	v_log_f32_e32 v1, v1
	v_xor_b32_e32 v14, v8, v9
	v_sub_u32_e32 v11, 32, v13
	v_ffbh_i32_e32 v13, v9
	v_ashrrev_i32_e32 v14, 31, v14
	v_ldexp_f32 v10, v10, v11
	v_add_u32_e32 v13, -1, v13
	v_add_u32_e32 v14, 32, v14
	s_mov_b32 s5, 0x3f317217
	v_mul_f32_e32 v12, 0x3f317217, v1
	v_add_f32_e32 v10, 1.0, v10
	v_min_u32_e32 v13, v13, v14
	v_fma_f32 v12, v1, s5, -v12
	v_mul_f32_e32 v10, 0x39000000, v10
	v_lshlrev_b64 v[8:9], v13, v[8:9]
	v_fmac_f32_e32 v12, 0x3377d1cf, v1
	s_mov_b32 s8, 0x7f800000
	v_floor_f32_e32 v10, v10
	v_min_u32_e32 v8, 1, v8
	v_fmac_f32_e32 v12, 0x3f317217, v1
	v_cmp_lt_f32_e64 s[0:1], |v1|, s8
	v_add_f32_e32 v10, 1.0, v10
	v_or_b32_e32 v8, v9, v8
	v_cndmask_b32_e64 v1, v1, v12, s[0:1]
	v_cmp_gt_f32_e64 s[0:1], s4, v10
	v_cvt_f32_i32_e32 v8, v8
	v_sub_u32_e32 v9, 32, v13
	v_cndmask_b32_e64 v11, 0, 32, s[0:1]
	v_ldexp_f32 v10, v10, v11
	v_log_f32_e32 v11, v10
	v_ldexp_f32 v8, v8, v9
	v_mov_b32_e32 v12, 0x41b17218
	v_add_f32_e32 v8, 1.0, v8
	v_cndmask_b32_e32 v10, 0, v12, vcc
	v_mul_f32_e32 v8, 0x39000000, v8
	v_sub_f32_e32 v10, v1, v10
	v_mul_f32_e32 v1, 0x3f317217, v11
	v_floor_f32_e32 v8, v8
	v_fma_f32 v1, v11, s5, -v1
	v_add_f32_e32 v8, 1.0, v8
	v_fmac_f32_e32 v1, 0x3377d1cf, v11
	v_cmp_gt_f32_e32 vcc, s4, v8
	v_fmac_f32_e32 v1, 0x3f317217, v11
	v_cmp_lt_f32_e64 s[2:3], |v11|, s8
	v_cndmask_b32_e64 v9, 0, 32, vcc
	v_ldexp_f32 v8, v8, v9
	v_cndmask_b32_e64 v1, v11, v1, s[2:3]
	v_cndmask_b32_e64 v9, 0, v12, s[0:1]
	v_xor_b32_e32 v13, v6, v7
	v_sub_f32_e32 v11, v1, v9
	v_ffbh_i32_e32 v9, v7
	v_ashrrev_i32_e32 v13, 31, v13
	v_add_u32_e32 v9, -1, v9
	v_add_u32_e32 v13, 32, v13
	v_min_u32_e32 v9, v9, v13
	v_lshlrev_b64 v[6:7], v9, v[6:7]
	v_min_u32_e32 v6, 1, v6
	v_or_b32_e32 v6, v7, v6
	v_cvt_f32_i32_e32 v6, v6
	v_sub_u32_e32 v7, 32, v9
	v_log_f32_e32 v8, v8
	v_ldexp_f32 v6, v6, v7
	v_add_f32_e32 v6, 1.0, v6
	v_mul_f32_e32 v6, 0x39000000, v6
	v_floor_f32_e32 v6, v6
	v_add_f32_e32 v6, 1.0, v6
	v_cmp_gt_f32_e64 s[0:1], s4, v6
	v_mul_f32_e32 v1, 0x3f317217, v8
	v_fma_f32 v1, v8, s5, -v1
	v_cndmask_b32_e64 v7, 0, 32, s[0:1]
	v_ldexp_f32 v6, v6, v7
	v_log_f32_e32 v7, v6
	v_fmac_f32_e32 v1, 0x3377d1cf, v8
	v_fmac_f32_e32 v1, 0x3f317217, v8
	v_cmp_lt_f32_e64 s[2:3], |v8|, s8
	v_cndmask_b32_e32 v6, 0, v12, vcc
	v_cmp_lt_f32_e64 vcc, |v7|, s8
	v_cndmask_b32_e64 v1, v8, v1, s[2:3]
	v_sub_f32_e32 v6, v1, v6
	v_mul_f32_e32 v1, 0x3f317217, v7
	v_fma_f32 v1, v7, s5, -v1
	v_fmac_f32_e32 v1, 0x3377d1cf, v7
	v_fmac_f32_e32 v1, 0x3f317217, v7
	v_cndmask_b32_e32 v1, v7, v1, vcc
	v_cndmask_b32_e64 v7, 0, v12, s[0:1]
	v_sub_f32_e32 v7, v1, v7
	s_mov_b32 s0, 0x3dcccccd
	v_pk_fma_f32 v[8:9], v[10:11], s[0:1], 1.0 op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[6:7], v[6:7], s[0:1], 1.0 op_sel_hi:[1,0,0]
	v_ashrrev_i32_e32 v1, 31, v0
	v_pk_mul_f32 v[4:5], v[6:7], v[4:5]
	v_pk_mul_f32 v[2:3], v[8:9], v[2:3]
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	global_store_dwordx4 v[0:1], v[2:5], off
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
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 15
		.amdhsa_accum_offset 16
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
	.set triton_kernel.num_vgpr, 16
	.set triton_kernel.num_agpr, 0
	.set triton_kernel.numbered_sgpr, 15
	.set triton_kernel.private_seg_size, 0
	.set triton_kernel.uses_vcc, 1
	.set triton_kernel.uses_flat_scratch, 0
	.set triton_kernel.has_dyn_sized_stack, 0
	.set triton_kernel.has_recursion, 0
	.set triton_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1264
; TotalNumSgprs: 21
; NumVgprs: 16
; NumAgprs: 0
; TotalNumVgprs: 16
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 21
; NumVGPRsForWavesPerEU: 16
; AccumOffset: 16
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 3
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
    .sgpr_count:     21
    .sgpr_spill_count: 0
    .symbol:         triton_kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     16
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
