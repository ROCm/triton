	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v10                             ; -- Begin function v10
	.p2align	8
	.type	v10,@function
v10:                                    ; @v10
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.3:
	.file	1 "/var/lib/jenkins/OAI-triton/study_matmul/gluon/matmul_kernels" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.4:
.LBB0_0:
	v_mov_b32_e32 v11, v0
	s_nop 0
	v_readfirstlane_b32 s51, v11
	s_bfe_u32 s1, s51, 0x20006
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s8, 0xff
	s_ashr_i32 s8, s0, 31
	s_lshr_b32 s8, s8, 24
	s_add_i32 s0, s0, s8
	s_ashr_i32 s0, s0, 8
	s_add_i32 s8, s9, 0xff
	s_ashr_i32 s9, s8, 31
	s_lshr_b32 s9, s9, 24
	s_add_i32 s8, s8, s9
	s_ashr_i32 s8, s8, 8
	s_ashr_i32 s9, s16, 31
	s_lshr_b32 s9, s9, 29
	s_add_i32 s9, s16, s9
	s_ashr_i32 s9, s9, 3
	s_lshl_b32 s13, s16, 5
	s_mulk_i32 s9, 0xff01
	s_add_i32 s9, s9, s13
	s_lshl_b32 s13, s8, 2
	s_xor_b32 s8, s9, s8
	s_ashr_i32 s8, s8, 31
	s_abs_i32 s14, s9
	s_abs_i32 s15, s13
	v_cvt_f32_u32_e32 v0, s15
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_sub_i32 s16, 0, s15
	v_readfirstlane_b32 s17, v0
	s_mul_i32 s16, s16, s17
	s_mul_hi_u32 s16, s17, s16
	s_add_i32 s17, s17, s16
	s_mul_hi_u32 s16, s14, s17
	s_mul_i32 s17, s16, s15
	s_sub_i32 s14, s14, s17
	s_add_i32 s17, s16, 1
	s_sub_i32 s18, s14, s15
	s_cmp_ge_u32 s14, s15
	s_cselect_b32 s16, s17, s16
	s_cselect_b32 s14, s18, s14
	s_add_i32 s17, s16, 1
	s_cmp_ge_u32 s14, s15
	s_cselect_b32 s14, s17, s16
	s_xor_b32 s14, s14, s8
	s_sub_i32 s8, s14, s8
	s_lshl_b32 s14, s8, 2
	s_sub_i32 s0, s0, s14
	s_min_i32 s0, s0, 4
	s_mul_i32 s8, s8, s13
	s_sub_i32 s8, s9, s8
	s_xor_b32 s9, s8, s0
	s_ashr_i32 s9, s9, 31
	s_abs_i32 s13, s8
	s_abs_i32 s15, s0
	v_cvt_f32_u32_e32 v0, s15
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_sub_i32 s16, 0, s15
	v_readfirstlane_b32 s17, v0
	s_mul_i32 s16, s16, s17
	s_mul_hi_u32 s16, s17, s16
	s_add_i32 s17, s17, s16
	s_mul_hi_u32 s16, s13, s17
	s_mul_i32 s17, s16, s15
	s_sub_i32 s13, s13, s17
	s_add_i32 s17, s16, 1
	s_sub_i32 s18, s13, s15
	s_cmp_ge_u32 s13, s15
	s_cselect_b32 s16, s17, s16
	s_cselect_b32 s13, s18, s13
	s_add_i32 s17, s16, 1
	s_cmp_ge_u32 s13, s15
	s_cselect_b32 s13, s17, s16
	s_xor_b32 s13, s13, s9
	s_sub_i32 s9, s13, s9
	s_mul_i32 s0, s9, s0
	s_sub_i32 s52, s8, s0
	s_add_i32 s52, s52, s14
	v_lshlrev_b32_e32 v0, 1, v11
	v_and_b32_e32 v0, 0x70, v0
	v_or_b32_e32 v1, s1, v0
	v_lshlrev_b32_e32 v0, 3, v11
	v_mov_b32_e32 v116, v0
	v_and_b32_e32 v0, 56, v0
	s_lshl_b32 s8, s52, 8
	s_mul_i32 s14, s8, s10
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[14:15], s[14:15], 1
	s_add_u32 s16, s2, s14
	s_addc_u32 s31, s3, s15
	s_lshl_b32 s0, s9, 8
	s_mul_i32 s14, s0, s11
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[14:15], s[14:15], 1
	s_add_u32 s20, s4, s14
	s_addc_u32 s14, s5, s15
	s_lshl_b32 s15, s10, 2
	s_mul_i32 s9, s10, 0x74
	v_mad_u64_u32 v[2:3], s[4:5], v1, s10, v[0:1]
	v_add_u32_e32 v3, s15, v2
	v_add_u32_e32 v4, s15, v3
	v_add_u32_e32 v5, s15, v4
	v_add_u32_e32 v6, s9, v5
	v_add_u32_e32 v7, s15, v6
	v_add_u32_e32 v8, s15, v7
	s_lshl_b32 s9, s11, 2
	v_mad_u64_u32 v[0:1], s[4:5], v1, s11, v[0:1]
	v_add_u32_e32 v1, s9, v0
	v_add_u32_e32 v9, s9, v1
	v_add_u32_e32 v10, s9, v9
	s_lshl_b32 s4, s11, 8
	s_ashr_i32 s36, s4, 1
	s_and_b32 s17, s31, 0xffff
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	s_mul_i32 s48, s1, 0x420
	s_add_i32 s4, s48, 0
	v_lshlrev_b32_e32 v12, 1, v2
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_add_i32 s49, s48, 0x1080
	s_add_i32 s5, s4, 0x1080
	v_lshlrev_b32_e32 v13, 1, v3
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v13, s[16:19], 0 offen lds
	s_add_i32 s50, s48, 0x2100
	s_add_i32 s9, s4, 0x2100
	v_lshlrev_b32_e32 v14, 1, v4
	s_mov_b32 m0, s9
	s_nop 0
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds
	s_add_i32 s53, s48, 0x3180
	s_add_i32 s11, s4, 0x3180
	v_lshlrev_b32_e32 v15, 1, v5
	s_mov_b32 m0, s11
	s_nop 0
	buffer_load_dwordx4 v15, s[16:19], 0 offen lds
	s_add_i32 s13, s4, 0x4200
	v_lshlrev_b32_e32 v6, 1, v6
	s_mov_b32 m0, s13
	s_nop 0
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	s_add_i32 s24, s4, 0x5280
	v_lshlrev_b32_e32 v7, 1, v7
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 s25, s4, 0x6300
	v_lshlrev_b32_e32 v16, 1, v8
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v16, s[16:19], 0 offen lds
	s_add_i32 s26, s4, 0x7380
	v_add_lshl_u32 v8, v8, s15, 1
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v8, s[16:19], 0 offen lds
	s_and_b32 s21, s14, 0xffff
	s_mov_b32 s22, s18
	s_mov_b32 s23, s19
	s_add_i32 s54, 0, 0x107e0
	s_add_i32 s27, s54, s48
	v_lshlrev_b32_e32 v2, 1, v0
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_add_i32 s28, s54, s49
	v_lshlrev_b32_e32 v3, 1, v1
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v3, s[20:23], 0 offen lds
	s_add_i32 s29, s54, s50
	v_lshlrev_b32_e32 v4, 1, v9
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	s_add_i32 s30, s54, s53
	v_lshlrev_b32_e32 v5, 1, v10
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_u32 s16, s16, 0x80
	s_addc_u32 s15, s31, 0
	s_add_i32 s35, 0, 0x18bc0
	s_add_i32 s31, s35, s48
	v_add_lshl_u32 v0, v0, s36, 1
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	s_add_i32 s33, s35, s49
	v_add_lshl_u32 v1, v1, s36, 1
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	s_add_i32 s34, s35, s50
	v_add_lshl_u32 v9, v9, s36, 1
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	s_add_i32 s35, s35, s53
	v_add_lshl_u32 v10, v10, s36, 1
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b32 s17, s15, 0xffff
	s_add_i32 s36, s4, 0x8400
	s_mov_b32 m0, s36
	scratch_store_dword off, v12, off offset:4 ; 4-byte Folded Spill
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_add_i32 s37, s4, 0x9480
	s_mov_b32 m0, s37
	scratch_store_dword off, v13, off       ; 4-byte Folded Spill
	buffer_load_dwordx4 v13, s[16:19], 0 offen lds
	s_add_i32 s38, s4, 0xa500
	s_mov_b32 m0, s38
	scratch_store_dword off, v14, off offset:8 ; 4-byte Folded Spill
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds
	s_add_i32 s39, s4, 0xb580
	s_mov_b32 m0, s39
	scratch_store_dword off, v15, off offset:12 ; 4-byte Folded Spill
	buffer_load_dwordx4 v15, s[16:19], 0 offen lds
	s_add_i32 s40, s4, 0xc600
	s_mov_b32 m0, s40
	scratch_store_dword off, v6, off offset:16 ; 4-byte Folded Spill
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	s_add_i32 s41, s4, 0xd680
	s_mov_b32 m0, s41
	scratch_store_dword off, v7, off offset:20 ; 4-byte Folded Spill
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 s42, s4, 0xe700
	s_mov_b32 m0, s42
	scratch_store_dword off, v16, off offset:24 ; 4-byte Folded Spill
	buffer_load_dwordx4 v16, s[16:19], 0 offen lds
	s_add_i32 s43, s4, 0xf780
	s_mov_b32 m0, s43
	scratch_store_dword off, v8, off offset:28 ; 4-byte Folded Spill
	buffer_load_dwordx4 v8, s[16:19], 0 offen lds
	s_add_i32 s47, 0, 0x149e0
	s_add_i32 s44, s47, s48
	v_add_u32_e32 v2, 0x80, v2
	s_mov_b32 m0, s44
	scratch_store_dword off, v2, off offset:32 ; 4-byte Folded Spill
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_add_i32 s45, s47, s49
	v_add_u32_e32 v2, 0x80, v3
	s_mov_b32 m0, s45
	v_accvgpr_write_b32 a156, v2
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_add_i32 s46, s47, s50
	v_add_u32_e32 v2, 0x80, v4
	s_mov_b32 m0, s46
	v_accvgpr_write_b32 a157, v2
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_add_i32 s47, s47, s53
	v_add_u32_e32 v2, 0x80, v5
	s_mov_b32 m0, s47
	v_accvgpr_write_b32 a166, v2
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_add_u32 s21, s20, 0x80
	s_addc_u32 s22, s14, 0
	s_and_b32 s17, s22, 0xffff
	s_mov_b32 s16, s21
	s_add_i32 s14, 0, 0x1cdc0
	s_add_i32 s23, s14, s48
	s_mov_b32 m0, s23
	v_accvgpr_write_b32 a167, v0
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_add_i32 s48, s14, s49
	s_mov_b32 m0, s48
	v_accvgpr_write_b32 a176, v1
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_add_i32 s49, s14, s50
	s_mov_b32 m0, s49
	v_accvgpr_write_b32 a177, v9
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	s_add_i32 s50, s14, s53
	s_mov_b32 m0, s50
	v_accvgpr_write_b32 a210, v10
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	s_waitcnt vmcnt(20) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v3, 15, v11
	v_lshlrev_b32_e32 v1, 10, v3
	v_mov_b32_e32 v117, v11
	v_and_b32_e32 v2, 48, v11
	s_cmp_lt_u32 s1, 2
	s_cselect_b64 s[14:15], -1, 0
	s_and_b64 s[16:17], s[14:15], exec
	s_cselect_b32 s16, 0, 0x100
	v_or3_b32 v0, v2, s16, v1
	scratch_store_dword off, v3, off offset:36 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v3, 5, v3
	v_add_u32_e32 v0, v0, v3
	v_add_u32_e32 v255, 0, v0
	ds_read_b128 a[56:59], v255
	ds_read_b128 a[60:63], v255 offset:64
	ds_read_b128 a[48:51], v255 offset:128
	ds_read_b128 a[52:55], v255 offset:192
	ds_read_b128 a[40:43], v255 offset:512
	ds_read_b128 a[44:47], v255 offset:576
	ds_read_b128 a[32:35], v255 offset:640
	ds_read_b128 a[36:39], v255 offset:704
	ds_read_b128 a[24:27], v255 offset:16896
	ds_read_b128 a[28:31], v255 offset:16960
	ds_read_b128 a[16:19], v255 offset:17024
	ds_read_b128 a[20:23], v255 offset:17088
	ds_read_b128 a[8:11], v255 offset:17408
	ds_read_b128 a[12:15], v255 offset:17472
	ds_read_b128 a[0:3], v255 offset:17536
	ds_read_b128 a[4:7], v255 offset:17600
	s_and_b32 s20, s51, 64
	v_or_b32_e32 v1, v1, v2
	v_add_u32_e32 v1, v1, v3
	v_lshl_add_u32 v1, s20, 2, v1
	v_accvgpr_write_b32 a211, v1
	v_add_u32_e32 v1, s54, v1
	ds_read_b128 a[68:71], v1
	ds_read_b128 a[76:79], v1 offset:64
	ds_read_b128 a[64:67], v1 offset:128
	ds_read_b128 a[72:75], v1 offset:192
	ds_read_b128 a[80:83], v1 offset:512
	ds_read_b128 a[84:87], v1 offset:576
	ds_read_b128 a[88:91], v1 offset:640
	ds_read_b128 a[92:95], v1 offset:704
	s_mul_i32 s10, s10, s52
	s_lshl_b32 s16, s10, 8
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s2, s2, s16
	s_addc_u32 s3, s3, s17
	s_add_u32 s2, s2, 0x180
	s_addc_u32 s3, s3, 0
	s_mov_b32 s10, -2
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a215, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_add_u32 s16, s2, 0xffffff80
	s_addc_u32 s17, s3, -1
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[56:59], v[80:83]
	v_mfma_f32_16x16x32_f16 v[6:9], a[76:79], a[60:63], v[226:229]
	s_nop 7
	v_accvgpr_write_b32 a147, v9
	v_accvgpr_write_b32 a146, v8
	v_accvgpr_write_b32 a145, v7
	v_accvgpr_write_b32 a144, v6
	v_mfma_f32_16x16x32_f16 v[230:233], a[64:67], a[56:59], v[230:233]
	v_mfma_f32_16x16x32_f16 v[6:9], a[72:75], a[60:63], v[230:233]
	s_nop 7
	v_accvgpr_write_b32 a151, v9
	v_accvgpr_write_b32 a150, v8
	v_accvgpr_write_b32 a149, v7
	v_accvgpr_write_b32 a148, v6
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	v_accvgpr_read_b32 v1, a211
	v_add_u32_e32 v254, 0, v1
	v_add_u32_e32 v6, 0x18bc0, v254
	ds_read_b128 v[118:121], v6
	v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[56:59], v[234:237]
	ds_read_b128 v[122:125], v6 offset:64
	v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[60:63], v[234:237]
	ds_read_b128 v[126:129], v6 offset:128
	v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[56:59], v[238:241]
	ds_read_b128 v[226:229], v6 offset:192
	v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[60:63], v[238:241]
	ds_read_b128 v[230:233], v6 offset:512
	v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[48:51], v[242:245]
	ds_read_b128 a[96:99], v6 offset:576
	v_mfma_f32_16x16x32_f16 v[242:245], a[76:79], a[52:55], v[242:245]
	ds_read_b128 a[100:103], v6 offset:640
	v_mfma_f32_16x16x32_f16 v[246:249], a[64:67], a[48:51], v[246:249]
	ds_read_b128 a[104:107], v6 offset:704
	v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[52:55], v[246:249]
	v_mfma_f32_16x16x32_f16 v[2:5], a[80:83], a[48:51], v[2:5]
	v_mfma_f32_16x16x32_f16 v[2:5], a[84:87], a[52:55], v[2:5]
	s_cmp_eq_u32 s10, 60
	s_cselect_b64 vcc, -1, 0
	s_and_b32 s17, s17, 0xffff
	v_bfrev_b32_e32 v0, 1
	scratch_load_dword v1, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a216, v1
	v_mfma_f32_16x16x32_f16 v[250:253], a[88:91], a[48:51], v[250:253]
	v_mfma_f32_16x16x32_f16 v[250:253], a[92:95], a[52:55], v[250:253]
	v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[40:43], v[194:197]
	v_mfma_f32_16x16x32_f16 v[194:197], a[76:79], a[44:47], v[194:197]
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a217, v1
	v_mfma_f32_16x16x32_f16 v[198:201], a[64:67], a[40:43], v[198:201]
	v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[44:47], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[40:43], v[202:205]
	v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[44:47], v[202:205]
	scratch_load_dword v1, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s9
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a218, v1
	v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[40:43], v[206:209]
	v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[44:47], v[206:209]
	v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[32:35], v[210:213]
	v_mfma_f32_16x16x32_f16 v[210:213], a[76:79], a[36:39], v[210:213]
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s11
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a219, v1
	v_mfma_f32_16x16x32_f16 v[214:217], a[64:67], a[32:35], v[214:217]
	v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[36:39], v[214:217]
	v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[32:35], v[218:221]
	v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[36:39], v[218:221]
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s13
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a220, v1
	v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[32:35], v[222:225]
	v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[36:39], v[222:225]
	v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[24:27], v[162:165]
	v_mfma_f32_16x16x32_f16 v[162:165], a[76:79], a[28:31], v[162:165]
	scratch_load_dword v1, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a221, v1
	v_mfma_f32_16x16x32_f16 v[166:169], a[64:67], a[24:27], v[166:169]
	v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[28:31], v[166:169]
	v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[24:27], v[170:173]
	v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[28:31], v[170:173]
	scratch_load_dword v1, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a222, v1
	v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[24:27], v[174:177]
	v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[28:31], v[174:177]
	v_mfma_f32_16x16x32_f16 v[182:185], a[68:71], a[16:19], v[182:185]
	v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[20:23], v[182:185]
	scratch_load_dword v1, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a223, v1
	v_mfma_f32_16x16x32_f16 v[186:189], a[64:67], a[16:19], v[186:189]
	v_mfma_f32_16x16x32_f16 v[186:189], a[72:75], a[20:23], v[186:189]
	v_mfma_f32_16x16x32_f16 v[190:193], a[80:83], a[16:19], v[190:193]
	v_mfma_f32_16x16x32_f16 v[190:193], a[84:87], a[20:23], v[190:193]
	s_and_b32 s17, s22, 0xffff
	s_mov_b32 s16, s21
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a224, v1
	v_mfma_f32_16x16x32_f16 v[178:181], a[88:91], a[16:19], v[178:181]
	v_mfma_f32_16x16x32_f16 v[178:181], a[92:95], a[20:23], v[178:181]
	v_mfma_f32_16x16x32_f16 v[130:133], a[68:71], a[8:11], v[130:133]
	v_mfma_f32_16x16x32_f16 v[130:133], a[76:79], a[12:15], v[130:133]
	v_accvgpr_read_b32 v1, a156
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a225, v1
	v_mfma_f32_16x16x32_f16 v[134:137], a[64:67], a[8:11], v[134:137]
	v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], a[12:15], v[134:137]
	v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[8:11], v[138:141]
	v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], a[12:15], v[138:141]
	v_accvgpr_read_b32 v1, a157
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a226, v1
	v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[8:11], v[142:145]
	v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[12:15], v[142:145]
	v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[0:3], v[146:149]
	v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], a[4:7], v[146:149]
	v_accvgpr_read_b32 v1, a166
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_accvgpr_write_b32 a227, v1
	v_mfma_f32_16x16x32_f16 v[150:153], a[64:67], a[0:3], v[150:153]
	v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[4:7], v[150:153]
	v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], a[0:3], v[154:157]
	v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], a[4:7], v[154:157]
	v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], a[0:3], v[158:161]
	v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[4:7], v[158:161]
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v6, a158
	v_accvgpr_read_b32 v7, a159
	v_accvgpr_read_b32 v8, a160
	v_accvgpr_read_b32 v9, a161
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[6:9], v[118:121], a[56:59], v[6:9]
	v_mfma_f32_16x16x32_f16 v[6:9], v[122:125], a[60:63], v[6:9]
	v_accvgpr_mov_b32 a64, a212
	v_accvgpr_mov_b32 a65, a213
	v_accvgpr_mov_b32 a66, a214
	v_accvgpr_mov_b32 a67, a215
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[64:67], v[126:129], a[56:59], a[64:67]
	v_mfma_f32_16x16x32_f16 a[158:161], v[226:229], a[60:63], a[64:67]
	v_mfma_f32_16x16x32_f16 v[14:17], v[230:233], a[56:59], v[42:45]
	v_mfma_f32_16x16x32_f16 v[10:13], a[96:99], a[60:63], v[14:17]
	s_nop 7
	v_accvgpr_write_b32 a215, v13
	v_accvgpr_write_b32 a214, v12
	v_accvgpr_write_b32 a213, v11
	v_accvgpr_write_b32 a212, v10
	v_accvgpr_mov_b32 a64, a162
	v_accvgpr_mov_b32 a65, a163
	v_accvgpr_mov_b32 a66, a164
	v_accvgpr_mov_b32 a67, a165
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[56:59], a[100:103], a[56:59], a[64:67]
	v_mfma_f32_16x16x32_f16 a[162:165], a[104:107], a[60:63], a[56:59]
	v_mfma_f32_16x16x32_f16 v[22:25], v[118:121], a[48:51], v[38:41]
	v_mfma_f32_16x16x32_f16 v[10:13], v[122:125], a[52:55], v[22:25]
	s_nop 7
	v_accvgpr_write_b32 a231, v13
	v_accvgpr_write_b32 a230, v12
	v_accvgpr_write_b32 a229, v11
	v_accvgpr_write_b32 a228, v10
	v_mfma_f32_16x16x32_f16 v[26:29], v[126:129], a[48:51], v[60:63]
	v_mfma_f32_16x16x32_f16 v[10:13], v[226:229], a[52:55], v[26:29]
	s_nop 7
	v_accvgpr_write_b32 a235, v13
	v_accvgpr_write_b32 a234, v12
	v_accvgpr_write_b32 a233, v11
	v_accvgpr_write_b32 a232, v10
	v_mfma_f32_16x16x32_f16 v[30:33], v[230:233], a[48:51], v[64:67]
	v_mfma_f32_16x16x32_f16 v[10:13], a[96:99], a[52:55], v[30:33]
	s_nop 7
	v_accvgpr_write_b32 a239, v13
	v_accvgpr_write_b32 a238, v12
	v_accvgpr_write_b32 a237, v11
	v_accvgpr_write_b32 a236, v10
	v_mfma_f32_16x16x32_f16 v[34:37], a[100:103], a[48:51], v[68:71]
	v_mfma_f32_16x16x32_f16 v[10:13], a[104:107], a[52:55], v[34:37]
	s_nop 7
	v_accvgpr_write_b32 a243, v13
	v_accvgpr_write_b32 a242, v12
	v_accvgpr_write_b32 a241, v11
	v_accvgpr_write_b32 a240, v10
	v_mfma_f32_16x16x32_f16 v[38:41], v[118:121], a[40:43], v[72:75]
	v_mfma_f32_16x16x32_f16 v[10:13], v[122:125], a[44:47], v[38:41]
	s_nop 7
	v_accvgpr_write_b32 a247, v13
	v_accvgpr_write_b32 a246, v12
	v_accvgpr_write_b32 a245, v11
	v_accvgpr_write_b32 a244, v10
	v_mfma_f32_16x16x32_f16 v[42:45], v[126:129], a[40:43], v[76:79]
	v_mfma_f32_16x16x32_f16 v[10:13], v[226:229], a[44:47], v[42:45]
	s_nop 7
	v_accvgpr_write_b32 a251, v13
	v_accvgpr_write_b32 a250, v12
	v_accvgpr_write_b32 a249, v11
	v_accvgpr_write_b32 a248, v10
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[48:51], v255 offset:33792
	v_mfma_f32_16x16x32_f16 v[46:49], v[230:233], a[40:43], v[46:49]
	ds_read_b128 a[52:55], v255 offset:33856
	v_mfma_f32_16x16x32_f16 v[14:17], a[96:99], a[44:47], v[46:49]
	ds_read_b128 a[56:59], v255 offset:33920
	v_mfma_f32_16x16x32_f16 v[50:53], a[100:103], a[40:43], v[50:53]
	ds_read_b128 a[40:43], v255 offset:33984
	v_mfma_f32_16x16x32_f16 v[18:21], a[104:107], a[44:47], v[50:53]
	ds_read_b128 a[44:47], v255 offset:34304
	v_mfma_f32_16x16x32_f16 v[54:57], v[118:121], a[32:35], v[54:57]
	ds_read_b128 a[64:67], v255 offset:34368
	v_mfma_f32_16x16x32_f16 v[22:25], v[122:125], a[36:39], v[54:57]
	ds_read_b128 a[68:71], v255 offset:34432
	v_mfma_f32_16x16x32_f16 v[58:61], v[126:129], a[32:35], v[92:95]
	ds_read_b128 a[72:75], v255 offset:34496
	v_mfma_f32_16x16x32_f16 v[26:29], v[226:229], a[36:39], v[58:61]
	ds_read_b128 a[80:83], v255 offset:50688
	v_mfma_f32_16x16x32_f16 v[62:65], v[230:233], a[32:35], v[88:91]
	ds_read_b128 a[84:87], v255 offset:50752
	v_mfma_f32_16x16x32_f16 v[30:33], a[96:99], a[36:39], v[62:65]
	ds_read_b128 a[108:111], v255 offset:50816
	v_mfma_f32_16x16x32_f16 v[66:69], a[100:103], a[32:35], v[84:87]
	ds_read_b128 a[112:115], v255 offset:50880
	v_mfma_f32_16x16x32_f16 v[34:37], a[104:107], a[36:39], v[66:69]
	ds_read_b128 a[116:119], v255 offset:51200
	v_mfma_f32_16x16x32_f16 v[70:73], v[118:121], a[24:27], v[104:107]
	ds_read_b128 a[120:123], v255 offset:51264
	v_mfma_f32_16x16x32_f16 v[10:13], v[122:125], a[28:31], v[70:73]
	s_nop 7
	v_accvgpr_write_b32 a255, v13
	v_accvgpr_write_b32 a254, v12
	v_accvgpr_write_b32 a253, v11
	v_accvgpr_write_b32 a252, v10
	ds_read_b128 a[124:127], v255 offset:51328
	v_mfma_f32_16x16x32_f16 v[74:77], v[126:129], a[24:27], v[96:99]
	ds_read_b128 a[128:131], v255 offset:51392
	v_mfma_f32_16x16x32_f16 v[10:13], v[226:229], a[28:31], v[74:77]
	s_nop 7
	v_accvgpr_write_b32 a155, v13
	v_accvgpr_write_b32 a154, v12
	v_accvgpr_write_b32 a153, v11
	v_accvgpr_write_b32 a152, v10
	v_add_u32_e32 v94, 0x149e0, v254
	ds_read_b128 a[32:35], v94
	v_mfma_f32_16x16x32_f16 v[78:81], v[230:233], a[24:27], v[100:103]
	ds_read_b128 a[36:39], v94 offset:64
	v_mfma_f32_16x16x32_f16 v[10:13], a[96:99], a[28:31], v[78:81]
	s_nop 7
	v_accvgpr_write_b32 a209, v13
	v_accvgpr_write_b32 a208, v12
	v_accvgpr_write_b32 a207, v11
	v_accvgpr_write_b32 a206, v10
	ds_read_b128 a[60:63], v94 offset:128
	v_mfma_f32_16x16x32_f16 v[82:85], a[100:103], a[24:27], v[108:111]
	ds_read_b128 a[24:27], v94 offset:192
	v_mfma_f32_16x16x32_f16 v[10:13], a[104:107], a[28:31], v[82:85]
	s_nop 7
	v_accvgpr_write_b32 a205, v13
	v_accvgpr_write_b32 a204, v12
	v_accvgpr_write_b32 a203, v11
	v_accvgpr_write_b32 a202, v10
	ds_read_b128 a[28:31], v94 offset:512
	v_mfma_f32_16x16x32_f16 v[86:89], v[118:121], a[16:19], v[112:115]
	ds_read_b128 a[76:79], v94 offset:576
	v_mfma_f32_16x16x32_f16 v[86:89], v[122:125], a[20:23], v[86:89]
	ds_read_b128 a[88:91], v94 offset:640
	v_mfma_f32_16x16x32_f16 a[168:171], v[126:129], a[16:19], a[168:171]
	ds_read_b128 a[92:95], v94 offset:704
	v_mfma_f32_16x16x32_f16 a[168:171], v[226:229], a[20:23], a[168:171]
	v_mfma_f32_16x16x32_f16 a[172:175], v[230:233], a[16:19], a[172:175]
	v_mfma_f32_16x16x32_f16 a[172:175], a[96:99], a[20:23], a[172:175]
	s_add_u32 s16, s21, 0x80
	s_addc_u32 s17, s22, 0
	s_and_b32 s17, s17, 0xffff
	v_accvgpr_read_b32 v1, a167
	v_cndmask_b32_e32 v10, v1, v0, vcc
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[16:19], a[100:103], a[16:19], a[178:181]
	v_mfma_f32_16x16x32_f16 a[178:181], a[104:107], a[20:23], a[16:19]
	s_nop 6
	v_accvgpr_mov_b32 a16, a182
	v_accvgpr_mov_b32 a17, a183
	v_accvgpr_mov_b32 a18, a184
	v_accvgpr_mov_b32 a19, a185
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[16:19], v[118:121], a[8:11], a[16:19]
	v_mfma_f32_16x16x32_f16 a[182:185], v[122:125], a[12:15], a[16:19]
	v_accvgpr_read_b32 v1, a176
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_nop 1
	v_accvgpr_mov_b32 a16, a186
	v_accvgpr_mov_b32 a17, a187
	v_accvgpr_mov_b32 a18, a188
	v_accvgpr_mov_b32 a19, a189
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[16:19], v[126:129], a[8:11], a[16:19]
	v_mfma_f32_16x16x32_f16 a[186:189], v[226:229], a[12:15], a[16:19]
	v_mfma_f32_16x16x32_f16 a[190:193], v[230:233], a[8:11], a[190:193]
	v_mfma_f32_16x16x32_f16 a[190:193], a[96:99], a[12:15], a[190:193]
	v_accvgpr_read_b32 v11, a177
	v_cndmask_b32_e32 v11, v11, v0, vcc
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[194:197], a[100:103], a[8:11], a[194:197]
	v_mfma_f32_16x16x32_f16 a[194:197], a[104:107], a[12:15], a[194:197]
	v_mfma_f32_16x16x32_f16 a[198:201], v[118:121], a[0:3], a[198:201]
	v_mfma_f32_16x16x32_f16 a[198:201], v[122:125], a[4:7], a[198:201]
	v_accvgpr_read_b32 v12, a210
	v_cndmask_b32_e32 v12, v12, v0, vcc
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	v_accvgpr_read_b32 v38, a132
	v_accvgpr_read_b32 v39, a133
	v_accvgpr_read_b32 v40, a134
	v_accvgpr_read_b32 v41, a135
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[122:125], v[126:129], a[0:3], v[38:41]
	v_mfma_f32_16x16x32_f16 v[122:125], v[226:229], a[4:7], v[122:125]
	s_nop 1
	v_accvgpr_read_b32 v38, a136
	v_accvgpr_read_b32 v39, a137
	v_accvgpr_read_b32 v40, a138
	v_accvgpr_read_b32 v41, a139
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[126:129], v[230:233], a[0:3], v[38:41]
	v_mfma_f32_16x16x32_f16 v[126:129], a[96:99], a[4:7], v[126:129]
	s_nop 1
	v_accvgpr_read_b32 v38, a140
	v_accvgpr_read_b32 v39, a141
	v_accvgpr_read_b32 v40, a142
	v_accvgpr_read_b32 v41, a143
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[226:229], a[100:103], a[0:3], v[38:41]
	v_mfma_f32_16x16x32_f16 v[226:229], a[104:107], a[4:7], v[226:229]
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_accvgpr_read_b32 v38, a144
	v_accvgpr_read_b32 v39, a145
	v_accvgpr_read_b32 v40, a146
	v_accvgpr_read_b32 v41, a147
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[230:233], a[32:35], a[48:51], v[38:41]
	v_mfma_f32_16x16x32_f16 v[80:83], a[36:39], a[52:55], v[230:233]
	s_nop 1
	v_accvgpr_read_b32 v38, a148
	v_accvgpr_read_b32 v39, a149
	v_accvgpr_read_b32 v40, a150
	v_accvgpr_read_b32 v41, a151
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[230:233], a[60:63], a[48:51], v[38:41]
	v_mfma_f32_16x16x32_f16 v[230:233], a[24:27], a[52:55], v[230:233]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v13, 0x1cdc0, v254
	ds_read_b128 a[96:99], v13
	v_mfma_f32_16x16x32_f16 v[234:237], a[28:31], a[48:51], v[234:237]
	ds_read_b128 a[100:103], v13 offset:64
	v_mfma_f32_16x16x32_f16 v[234:237], a[76:79], a[52:55], v[234:237]
	ds_read_b128 a[104:107], v13 offset:128
	v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[48:51], v[238:241]
	ds_read_b128 a[132:135], v13 offset:192
	v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[52:55], v[238:241]
	ds_read_b128 a[136:139], v13 offset:512
	v_mfma_f32_16x16x32_f16 v[242:245], a[32:35], a[56:59], v[242:245]
	ds_read_b128 a[140:143], v13 offset:576
	v_mfma_f32_16x16x32_f16 v[242:245], a[36:39], a[40:43], v[242:245]
	ds_read_b128 a[144:147], v13 offset:640
	v_mfma_f32_16x16x32_f16 v[246:249], a[60:63], a[56:59], v[246:249]
	ds_read_b128 a[148:151], v13 offset:704
	v_mfma_f32_16x16x32_f16 v[246:249], a[24:27], a[40:43], v[246:249]
	v_mfma_f32_16x16x32_f16 v[2:5], a[28:31], a[56:59], v[2:5]
	v_mfma_f32_16x16x32_f16 v[2:5], a[76:79], a[40:43], v[2:5]
	s_and_b32 s53, s3, 0xffff
	s_mov_b32 s52, s2
	s_mov_b32 s54, s18
	s_mov_b32 s55, s19
	s_mov_b32 m0, s36
	v_accvgpr_read_b32 v0, a216
	buffer_load_dwordx4 v0, s[52:55], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[250:253], a[88:91], a[56:59], v[250:253]
	v_mfma_f32_16x16x32_f16 v[250:253], a[92:95], a[40:43], v[250:253]
	v_mfma_f32_16x16x32_f16 v[194:197], a[32:35], a[44:47], v[194:197]
	v_mfma_f32_16x16x32_f16 v[194:197], a[36:39], a[64:67], v[194:197]
	s_mov_b32 m0, s37
	v_accvgpr_read_b32 v0, a217
	buffer_load_dwordx4 v0, s[52:55], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[198:201], a[60:63], a[44:47], v[198:201]
	v_mfma_f32_16x16x32_f16 v[198:201], a[24:27], a[64:67], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[28:31], a[44:47], v[202:205]
	v_mfma_f32_16x16x32_f16 v[202:205], a[76:79], a[64:67], v[202:205]
	s_mov_b32 m0, s38
	v_accvgpr_read_b32 v0, a218
	buffer_load_dwordx4 v0, s[52:55], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[44:47], v[206:209]
	v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[64:67], v[206:209]
	v_mfma_f32_16x16x32_f16 v[210:213], a[32:35], a[68:71], v[210:213]
	v_mfma_f32_16x16x32_f16 v[210:213], a[36:39], a[72:75], v[210:213]
	s_mov_b32 m0, s39
	v_accvgpr_read_b32 v0, a219
	buffer_load_dwordx4 v0, s[52:55], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[214:217], a[60:63], a[68:71], v[214:217]
	v_mfma_f32_16x16x32_f16 v[214:217], a[24:27], a[72:75], v[214:217]
	v_mfma_f32_16x16x32_f16 v[218:221], a[28:31], a[68:71], v[218:221]
	v_mfma_f32_16x16x32_f16 v[218:221], a[76:79], a[72:75], v[218:221]
	s_mov_b32 m0, s40
	v_accvgpr_read_b32 v0, a220
	buffer_load_dwordx4 v0, s[52:55], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[68:71], v[222:225]
	v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[72:75], v[222:225]
	v_mfma_f32_16x16x32_f16 v[162:165], a[32:35], a[80:83], v[162:165]
	v_mfma_f32_16x16x32_f16 v[162:165], a[36:39], a[84:87], v[162:165]
	s_mov_b32 m0, s41
	v_accvgpr_read_b32 v0, a221
	buffer_load_dwordx4 v0, s[52:55], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[166:169], a[60:63], a[80:83], v[166:169]
	v_mfma_f32_16x16x32_f16 v[166:169], a[24:27], a[84:87], v[166:169]
	v_mfma_f32_16x16x32_f16 v[170:173], a[28:31], a[80:83], v[170:173]
	v_mfma_f32_16x16x32_f16 v[170:173], a[76:79], a[84:87], v[170:173]
	s_mov_b32 m0, s42
	v_accvgpr_read_b32 v0, a222
	buffer_load_dwordx4 v0, s[52:55], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[80:83], v[174:177]
	v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[84:87], v[174:177]
	v_mfma_f32_16x16x32_f16 v[182:185], a[32:35], a[108:111], v[182:185]
	v_mfma_f32_16x16x32_f16 v[182:185], a[36:39], a[112:115], v[182:185]
	s_mov_b32 m0, s43
	v_accvgpr_read_b32 v0, a223
	buffer_load_dwordx4 v0, s[52:55], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[186:189], a[60:63], a[108:111], v[186:189]
	v_mfma_f32_16x16x32_f16 v[186:189], a[24:27], a[112:115], v[186:189]
	v_mfma_f32_16x16x32_f16 v[190:193], a[28:31], a[108:111], v[190:193]
	v_mfma_f32_16x16x32_f16 v[190:193], a[76:79], a[112:115], v[190:193]
	s_mov_b32 m0, s44
	v_accvgpr_read_b32 v0, a224
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[178:181], a[88:91], a[108:111], v[178:181]
	v_mfma_f32_16x16x32_f16 v[178:181], a[92:95], a[112:115], v[178:181]
	v_mfma_f32_16x16x32_f16 v[130:133], a[32:35], a[116:119], v[130:133]
	v_mfma_f32_16x16x32_f16 v[130:133], a[36:39], a[120:123], v[130:133]
	s_mov_b32 m0, s45
	v_accvgpr_read_b32 v0, a225
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[134:137], a[60:63], a[116:119], v[134:137]
	v_mfma_f32_16x16x32_f16 v[134:137], a[24:27], a[120:123], v[134:137]
	v_mfma_f32_16x16x32_f16 v[138:141], a[28:31], a[116:119], v[138:141]
	v_mfma_f32_16x16x32_f16 v[138:141], a[76:79], a[120:123], v[138:141]
	s_mov_b32 m0, s46
	v_accvgpr_read_b32 v0, a226
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[116:119], v[142:145]
	v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[120:123], v[142:145]
	v_mfma_f32_16x16x32_f16 v[146:149], a[32:35], a[124:127], v[146:149]
	v_mfma_f32_16x16x32_f16 v[146:149], a[36:39], a[128:131], v[146:149]
	s_mov_b32 m0, s47
	v_accvgpr_read_b32 v0, a227
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[150:153], a[60:63], a[124:127], v[150:153]
	v_mfma_f32_16x16x32_f16 v[150:153], a[24:27], a[128:131], v[150:153]
	v_mfma_f32_16x16x32_f16 v[154:157], a[28:31], a[124:127], v[154:157]
	v_mfma_f32_16x16x32_f16 v[154:157], a[76:79], a[128:131], v[154:157]
	v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], a[124:127], v[158:161]
	v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[128:131], v[158:161]
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[6:9], a[96:99], a[48:51], v[6:9]
	v_mfma_f32_16x16x32_f16 v[40:43], a[100:103], a[52:55], v[6:9]
	v_accvgpr_mov_b32 a0, a158
	v_accvgpr_mov_b32 a1, a159
	v_accvgpr_mov_b32 a2, a160
	v_accvgpr_mov_b32 a3, a161
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[0:3], a[104:107], a[48:51], a[0:3]
	v_mfma_f32_16x16x32_f16 a[0:3], a[132:135], a[52:55], a[0:3]
	v_accvgpr_mov_b32 a4, a212
	v_accvgpr_mov_b32 a5, a213
	v_accvgpr_mov_b32 a6, a214
	v_accvgpr_mov_b32 a7, a215
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[4:7], a[136:139], a[48:51], a[4:7]
	v_mfma_f32_16x16x32_f16 a[4:7], a[140:143], a[52:55], a[4:7]
	v_mfma_f32_16x16x32_f16 a[162:165], a[144:147], a[48:51], a[162:165]
	v_mfma_f32_16x16x32_f16 a[162:165], a[148:151], a[52:55], a[162:165]
	v_accvgpr_mov_b32 a8, a228
	v_accvgpr_mov_b32 a9, a229
	v_accvgpr_mov_b32 a10, a230
	v_accvgpr_mov_b32 a11, a231
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[8:11], a[96:99], a[56:59], a[8:11]
	v_mfma_f32_16x16x32_f16 a[8:11], a[100:103], a[40:43], a[8:11]
	v_accvgpr_mov_b32 a12, a232
	v_accvgpr_mov_b32 a13, a233
	v_accvgpr_mov_b32 a14, a234
	v_accvgpr_mov_b32 a15, a235
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[12:15], a[104:107], a[56:59], a[12:15]
	v_mfma_f32_16x16x32_f16 a[12:15], a[132:135], a[40:43], a[12:15]
	v_accvgpr_mov_b32 a16, a236
	v_accvgpr_mov_b32 a17, a237
	v_accvgpr_mov_b32 a18, a238
	v_accvgpr_mov_b32 a19, a239
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[16:19], a[136:139], a[56:59], a[16:19]
	v_mfma_f32_16x16x32_f16 a[20:23], a[140:143], a[40:43], a[16:19]
	s_nop 6
	v_accvgpr_mov_b32 a16, a240
	v_accvgpr_mov_b32 a17, a241
	v_accvgpr_mov_b32 a18, a242
	v_accvgpr_mov_b32 a19, a243
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[16:19], a[144:147], a[56:59], a[16:19]
	v_mfma_f32_16x16x32_f16 a[76:79], a[148:151], a[40:43], a[16:19]
	s_nop 6
	v_accvgpr_mov_b32 a16, a244
	v_accvgpr_mov_b32 a17, a245
	v_accvgpr_mov_b32 a18, a246
	v_accvgpr_mov_b32 a19, a247
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[16:19], a[96:99], a[44:47], a[16:19]
	v_mfma_f32_16x16x32_f16 a[88:91], a[100:103], a[64:67], a[16:19]
	s_nop 6
	v_accvgpr_mov_b32 a16, a248
	v_accvgpr_mov_b32 a17, a249
	v_accvgpr_mov_b32 a18, a250
	v_accvgpr_mov_b32 a19, a251
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[16:19], a[104:107], a[44:47], a[16:19]
	v_accvgpr_write_b32 a161, v43
	v_accvgpr_write_b32 a160, v42
	v_accvgpr_write_b32 a159, v41
	v_accvgpr_write_b32 a158, v40
	v_mfma_f32_16x16x32_f16 a[92:95], a[132:135], a[64:67], a[16:19]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[56:59], v255
	v_mfma_f32_16x16x32_f16 v[6:9], a[136:139], a[44:47], v[14:17]
	v_accvgpr_mov_b32 a215, a3
	v_accvgpr_mov_b32 a214, a2
	v_accvgpr_mov_b32 a213, a1
	v_accvgpr_mov_b32 a212, a0
	ds_read_b128 a[60:63], v255 offset:64
	v_mfma_f32_16x16x32_f16 v[46:49], a[140:143], a[64:67], v[6:9]
	ds_read_b128 a[48:51], v255 offset:128
	v_mfma_f32_16x16x32_f16 v[6:9], a[144:147], a[44:47], v[18:21]
	v_accvgpr_read_b32 v45, a7
	v_accvgpr_read_b32 v44, a6
	v_accvgpr_read_b32 v43, a5
	v_accvgpr_read_b32 v42, a4
	ds_read_b128 a[52:55], v255 offset:192
	v_mfma_f32_16x16x32_f16 v[50:53], a[148:151], a[64:67], v[6:9]
	ds_read_b128 a[40:43], v255 offset:512
	v_mfma_f32_16x16x32_f16 v[6:9], a[96:99], a[68:71], v[22:25]
	ds_read_b128 a[44:47], v255 offset:576
	v_mfma_f32_16x16x32_f16 v[54:57], a[100:103], a[72:75], v[6:9]
	ds_read_b128 a[32:35], v255 offset:640
	v_mfma_f32_16x16x32_f16 v[6:9], a[104:107], a[68:71], v[26:29]
	v_accvgpr_read_b32 v41, a11
	v_accvgpr_read_b32 v40, a10
	v_accvgpr_read_b32 v39, a9
	v_accvgpr_read_b32 v38, a8
	ds_read_b128 a[36:39], v255 offset:704
	v_mfma_f32_16x16x32_f16 v[92:95], a[132:135], a[72:75], v[6:9]
	ds_read_b128 a[24:27], v255 offset:16896
	v_mfma_f32_16x16x32_f16 v[6:9], a[136:139], a[68:71], v[30:33]
	v_accvgpr_read_b32 v63, a15
	v_accvgpr_read_b32 v62, a14
	v_accvgpr_read_b32 v61, a13
	v_accvgpr_read_b32 v60, a12
	ds_read_b128 a[28:31], v255 offset:16960
	v_mfma_f32_16x16x32_f16 v[30:33], a[140:143], a[72:75], v[6:9]
	ds_read_b128 a[16:19], v255 offset:17024
	v_mfma_f32_16x16x32_f16 v[6:9], a[144:147], a[68:71], v[34:37]
	v_accvgpr_read_b32 v67, a23
	v_accvgpr_read_b32 v66, a22
	v_accvgpr_read_b32 v65, a21
	v_accvgpr_read_b32 v64, a20
	ds_read_b128 a[20:23], v255 offset:17088
	v_mfma_f32_16x16x32_f16 v[34:37], a[148:151], a[72:75], v[6:9]
	ds_read_b128 a[8:11], v255 offset:17408
	v_accvgpr_mov_b32 a0, a252
	v_accvgpr_mov_b32 a1, a253
	v_accvgpr_mov_b32 a2, a254
	v_accvgpr_mov_b32 a3, a255
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[0:3], a[96:99], a[80:83], a[0:3]
	v_accvgpr_read_b32 v68, a76
	v_accvgpr_read_b32 v69, a77
	v_accvgpr_read_b32 v70, a78
	v_accvgpr_read_b32 v71, a79
	ds_read_b128 a[12:15], v255 offset:17472
	v_mfma_f32_16x16x32_f16 a[216:219], a[100:103], a[84:87], a[0:3]
	s_nop 2
	ds_read_b128 a[0:3], v255 offset:17536
	v_accvgpr_mov_b32 a4, a152
	v_accvgpr_mov_b32 a5, a153
	v_accvgpr_mov_b32 a6, a154
	v_accvgpr_mov_b32 a7, a155
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[64:67], a[104:107], a[80:83], a[4:7]
	v_accvgpr_read_b32 v72, a88
	v_accvgpr_read_b32 v73, a89
	v_accvgpr_read_b32 v74, a90
	v_accvgpr_read_b32 v75, a91
	ds_read_b128 a[4:7], v255 offset:17600
	v_mfma_f32_16x16x32_f16 a[152:155], a[132:135], a[84:87], a[64:67]
	v_add_u32_e32 v13, 0x107e0, v254
	ds_read_b128 a[68:71], v13
	s_nop 0
	v_accvgpr_mov_b32 a64, a206
	v_accvgpr_mov_b32 a65, a207
	v_accvgpr_mov_b32 a66, a208
	v_accvgpr_mov_b32 a67, a209
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[64:67], a[136:139], a[80:83], a[64:67]
	v_accvgpr_read_b32 v76, a92
	v_accvgpr_read_b32 v77, a93
	v_accvgpr_read_b32 v78, a94
	v_accvgpr_read_b32 v79, a95
	ds_read_b128 a[76:79], v13 offset:64
	v_mfma_f32_16x16x32_f16 a[206:209], a[140:143], a[84:87], a[64:67]
	s_nop 2
	ds_read_b128 a[64:67], v13 offset:128
	v_accvgpr_mov_b32 a72, a202
	v_accvgpr_mov_b32 a73, a203
	v_accvgpr_mov_b32 a74, a204
	v_accvgpr_mov_b32 a75, a205
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[80:83], a[144:147], a[80:83], a[72:75]
	s_nop 2
	ds_read_b128 a[72:75], v13 offset:192
	v_mfma_f32_16x16x32_f16 a[202:205], a[148:151], a[84:87], a[80:83]
	s_nop 2
	ds_read_b128 a[80:83], v13 offset:512
	v_mfma_f32_16x16x32_f16 v[6:9], a[96:99], a[108:111], v[86:89]
	ds_read_b128 a[84:87], v13 offset:576
	v_mfma_f32_16x16x32_f16 v[112:115], a[100:103], a[112:115], v[6:9]
	ds_read_b128 a[88:91], v13 offset:640
	v_mfma_f32_16x16x32_f16 a[168:171], a[104:107], a[108:111], a[168:171]
	ds_read_b128 a[92:95], v13 offset:704
	v_mfma_f32_16x16x32_f16 a[168:171], a[132:135], a[112:115], a[168:171]
	v_mfma_f32_16x16x32_f16 a[172:175], a[136:139], a[108:111], a[172:175]
	v_mfma_f32_16x16x32_f16 a[172:175], a[140:143], a[112:115], a[172:175]
	s_add_u32 s21, s21, 0x100
	s_addc_u32 s22, s22, 0
	s_and_b32 s17, s22, 0xffff
	s_mov_b32 s16, s21
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[178:181], a[144:147], a[108:111], a[178:181]
	v_mov_b64_e32 v[90:91], v[32:33]
	v_mov_b64_e32 v[88:89], v[30:31]
	v_mfma_f32_16x16x32_f16 a[178:181], a[148:151], a[112:115], a[178:181]
	v_mfma_f32_16x16x32_f16 a[182:185], a[96:99], a[116:119], a[182:185]
	v_mov_b64_e32 v[86:87], v[36:37]
	v_mov_b64_e32 v[84:85], v[34:35]
	v_mfma_f32_16x16x32_f16 a[182:185], a[100:103], a[120:123], a[182:185]
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[186:189], a[104:107], a[116:119], a[186:189]
	v_accvgpr_read_b32 v100, a206
	v_accvgpr_read_b32 v101, a207
	v_accvgpr_read_b32 v102, a208
	v_accvgpr_read_b32 v103, a209
	v_accvgpr_read_b32 v96, a152
	v_accvgpr_read_b32 v97, a153
	v_accvgpr_read_b32 v98, a154
	v_accvgpr_read_b32 v99, a155
	v_accvgpr_read_b32 v104, a216
	v_accvgpr_read_b32 v105, a217
	v_accvgpr_read_b32 v106, a218
	v_accvgpr_read_b32 v107, a219
	v_mfma_f32_16x16x32_f16 a[186:189], a[132:135], a[120:123], a[186:189]
	v_mfma_f32_16x16x32_f16 a[190:193], a[136:139], a[116:119], a[190:193]
	v_accvgpr_read_b32 v108, a202
	v_accvgpr_read_b32 v109, a203
	v_accvgpr_read_b32 v110, a204
	v_accvgpr_read_b32 v111, a205
	v_mfma_f32_16x16x32_f16 a[190:193], a[140:143], a[120:123], a[190:193]
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[194:197], a[144:147], a[116:119], a[194:197]
	v_mfma_f32_16x16x32_f16 a[194:197], a[148:151], a[120:123], a[194:197]
	v_mfma_f32_16x16x32_f16 a[198:201], a[96:99], a[124:127], a[198:201]
	v_mfma_f32_16x16x32_f16 a[198:201], a[100:103], a[128:131], a[198:201]
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[6:9], a[104:107], a[124:127], v[122:125]
	v_mfma_f32_16x16x32_f16 v[6:9], a[132:135], a[128:131], v[6:9]
	s_nop 7
	v_accvgpr_write_b32 a135, v9
	v_accvgpr_write_b32 a134, v8
	v_accvgpr_write_b32 a133, v7
	v_accvgpr_write_b32 a132, v6
	v_mfma_f32_16x16x32_f16 v[6:9], a[136:139], a[124:127], v[126:129]
	v_mfma_f32_16x16x32_f16 v[6:9], a[140:143], a[128:131], v[6:9]
	s_nop 7
	v_accvgpr_write_b32 a139, v9
	v_accvgpr_write_b32 a138, v8
	v_accvgpr_write_b32 a137, v7
	v_accvgpr_write_b32 a136, v6
	v_mfma_f32_16x16x32_f16 v[6:9], a[144:147], a[124:127], v[226:229]
	v_mfma_f32_16x16x32_f16 v[6:9], a[148:151], a[128:131], v[6:9]
	s_nop 7
	v_accvgpr_write_b32 a143, v9
	v_accvgpr_write_b32 a142, v8
	v_accvgpr_write_b32 a141, v7
	v_accvgpr_write_b32 a140, v6
	; sched_barrier mask(0x00000000)
	s_add_u32 s2, s2, 0x100
	s_addc_u32 s3, s3, 0
	s_add_i32 s10, s10, 2
	s_cmp_lt_u32 s10, 61
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	s_lshl_b32 s5, s1, 6
	v_mov_b32_e32 v58, v117
	v_and_b32_e32 v34, 63, v58
	v_or_b32_e32 v0, s5, v34
	v_lshrrev_b32_e32 v0, 4, v0
	s_mul_i32 s2, s8, s12
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s2, s6, s2
	s_addc_u32 s3, s7, s3
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s4, s2, s0
	s_addc_u32 s9, s3, s1
	s_add_u32 s0, s4, 0x100
	s_addc_u32 s1, s9, 0
	v_mul_lo_u32 v0, v0, s12
	s_lshl_b32 s21, s12, 4
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v35, v1, 3, v0
	v_add_u32_e32 v36, s21, v35
	v_add_u32_e32 v37, s21, v36
	v_add_u32_e32 v59, s21, v37
	v_add_u32_e32 v0, s21, v59
	v_add_u32_e32 v1, s21, v0
	v_add_u32_e32 v255, s21, v1
	v_add_u32_e32 v11, s21, v255
	v_add_u32_e32 v10, s21, v11
	v_accvgpr_write_b32 a103, v11
	v_add_u32_e32 v11, s21, v10
	v_accvgpr_write_b32 a102, v10
	v_add_u32_e32 v10, s21, v11
	v_accvgpr_write_b32 a101, v11
	v_add_u32_e32 v11, s21, v10
	v_accvgpr_write_b32 a100, v10
	v_add_u32_e32 v10, s21, v11
	v_accvgpr_write_b32 a99, v11
	v_add_u32_e32 v11, s21, v10
	v_accvgpr_write_b32 a98, v10
	v_add_u32_e32 v10, s21, v11
	v_accvgpr_write_b32 a97, v11
	v_accvgpr_write_b32 a96, v10
	s_lshl_b32 s2, s12, 6
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[6:7], s[2:3], 1
	s_add_u32 s16, s4, s6
	s_addc_u32 s10, s9, s7
	s_add_u32 s12, s16, s6
	s_addc_u32 s3, s10, s7
	s_add_u32 s8, s12, s6
	s_addc_u32 s2, s3, s7
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[56:59], v[80:83]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[6:9], a[76:79], a[60:63], v[6:9]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[56:59], v[230:233]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[60:63], v[10:13]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[56:59], v[234:237]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[60:63], v[14:17]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[18:21], a[88:91], a[56:59], v[238:241]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[18:21], a[92:95], a[60:63], v[18:21]
	v_mfma_f32_16x16x32_f16 v[22:25], a[68:71], a[48:51], v[242:245]
	v_mfma_f32_16x16x32_f16 v[22:25], a[76:79], a[52:55], v[22:25]
	v_mfma_f32_16x16x32_f16 v[26:29], a[64:67], a[48:51], v[246:249]
	v_mfma_f32_16x16x32_f16 v[26:29], a[72:75], a[52:55], v[26:29]
	v_mfma_f32_16x16x32_f16 v[2:5], a[80:83], a[48:51], v[2:5]
	v_mfma_f32_16x16x32_f16 v[2:5], a[84:87], a[52:55], v[2:5]
	v_mfma_f32_16x16x32_f16 v[30:33], a[88:91], a[48:51], v[250:253]
	v_mfma_f32_16x16x32_f16 v[30:33], a[92:95], a[52:55], v[30:33]
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v7, v8, v9
	v_cvt_pk_f16_f32 v10, v10, v11
	v_cvt_pk_f16_f32 v11, v12, v13
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_cvt_pk_f16_f32 v8, v22, v23
	v_cvt_pk_f16_f32 v9, v24, v25
	v_cvt_pk_f16_f32 v12, v26, v27
	v_cvt_pk_f16_f32 v13, v28, v29
	v_cvt_pk_f16_f32 v16, v2, v3
	v_cvt_pk_f16_f32 v17, v4, v5
	v_cvt_pk_f16_f32 v20, v30, v31
	v_cvt_pk_f16_f32 v21, v32, v33
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v2, 8, v58
	v_mov_b32_e32 v25, v116
	v_and_b32_e32 v22, 0x70, v25
	v_and_b32_e32 v23, 1, v58
	v_lshlrev_b32_e32 v3, 12, v23
	v_and_b32_e32 v24, 16, v58
	v_lshlrev_b32_e32 v4, 4, v24
	s_lshr_b32 s11, s20, 1
	s_and_b64 s[6:7], s[14:15], exec
	s_cselect_b32 s6, 0, 0x80
	s_movk_i32 s7, 0x2e00
	v_and_or_b32 v2, v2, s7, v3
	v_or3_b32 v5, s6, v4, v2
	v_mov_b32_e32 v2, 0x70
	v_bitop3_b32 v25, s11, v25, v2 bitop3:0x78
	v_or_b32_e32 v4, v5, v25
	v_add_u32_e32 v2, 0, v4
	ds_write_b128 v2, v[6:9]
	v_xad_u32 v3, v4, 16, 0
	ds_write_b128 v3, v[10:13]
	v_xad_u32 v4, v4, 64, 0
	ds_write_b128 v4, v[14:17]
	s_movk_i32 s6, 0x50
	v_bitop3_b32 v5, v5, s6, v25 bitop3:0x36
	v_add_u32_e32 v5, 0, v5
	ds_write_b128 v5, v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v6, 0xe0
	v_bitop3_b32 v6, s5, v6, v34 bitop3:0xc8
	v_lshlrev_b32_e32 v7, 4, v6
	v_lshrrev_b32_e32 v6, 1, v6
	v_lshlrev_b32_e32 v8, 8, v24
	v_bitop3_b32 v6, v7, v6, v22 bitop3:0x36
	v_lshl_add_u32 v7, v23, 13, 0
	v_add3_u32 v226, v7, v8, v6
	ds_read_b128 v[6:9], v226
	ds_read_b128 v[12:15], v226 offset:256
	ds_read_b128 v[16:19], v226 offset:128
	ds_read_b128 v[22:25], v226 offset:384
	s_and_b32 s5, s9, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v10, v6
	v_mov_b32_e32 v11, v7
	v_lshlrev_b32_e32 v227, 1, v35
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v227, s[4:7], 0 offen
	s_nop 1
	v_mov_b32_e32 v10, v14
	v_mov_b32_e32 v11, v15
	v_lshlrev_b32_e32 v228, 1, v36
	buffer_store_dwordx4 v[8:11], v228, s[4:7], 0 offen
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v20, v16
	v_mov_b32_e32 v21, v17
	v_lshlrev_b32_e32 v229, 1, v37
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v229, s[4:7], 0 offen
	s_nop 1
	v_mov_b32_e32 v20, v24
	v_mov_b32_e32 v21, v25
	v_lshlrev_b32_e32 v230, 1, v59
	buffer_store_dwordx4 v[18:21], v230, s[4:7], 0 offen
	v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[40:43], v[194:197]
	v_mfma_f32_16x16x32_f16 v[6:9], a[76:79], a[44:47], v[6:9]
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[40:43], v[198:201]
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[44:47], v[10:13]
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[40:43], v[202:205]
	v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[44:47], v[14:17]
	v_mfma_f32_16x16x32_f16 v[18:21], a[88:91], a[40:43], v[206:209]
	v_mfma_f32_16x16x32_f16 v[18:21], a[92:95], a[44:47], v[18:21]
	v_mfma_f32_16x16x32_f16 v[22:25], a[68:71], a[32:35], v[210:213]
	v_mfma_f32_16x16x32_f16 v[22:25], a[76:79], a[36:39], v[22:25]
	v_mfma_f32_16x16x32_f16 v[26:29], a[64:67], a[32:35], v[214:217]
	v_mfma_f32_16x16x32_f16 v[26:29], a[72:75], a[36:39], v[26:29]
	v_mfma_f32_16x16x32_f16 v[30:33], a[80:83], a[32:35], v[218:221]
	v_mfma_f32_16x16x32_f16 v[30:33], a[84:87], a[36:39], v[30:33]
	v_mfma_f32_16x16x32_f16 v[34:37], a[88:91], a[32:35], v[222:225]
	v_mfma_f32_16x16x32_f16 v[34:37], a[92:95], a[36:39], v[34:37]
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v7, v8, v9
	v_cvt_pk_f16_f32 v10, v10, v11
	v_cvt_pk_f16_f32 v11, v12, v13
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_cvt_pk_f16_f32 v8, v22, v23
	v_cvt_pk_f16_f32 v9, v24, v25
	v_cvt_pk_f16_f32 v12, v26, v27
	v_cvt_pk_f16_f32 v13, v28, v29
	v_cvt_pk_f16_f32 v16, v30, v31
	v_cvt_pk_f16_f32 v17, v32, v33
	v_cvt_pk_f16_f32 v20, v34, v35
	v_cvt_pk_f16_f32 v21, v36, v37
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v2, v[6:9]
	ds_write_b128 v3, v[10:13]
	ds_write_b128 v4, v[14:17]
	ds_write_b128 v5, v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[6:9], v226
	ds_read_b128 v[12:15], v226 offset:256
	ds_read_b128 v[16:19], v226 offset:128
	ds_read_b128 v[22:25], v226 offset:384
	s_and_b32 s17, s10, 0xffff
	s_mov_b32 s18, s6
	s_mov_b32 s19, s7
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v10, v6
	v_mov_b32_e32 v11, v7
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v227, s[16:19], 0 offen
	s_nop 1
	v_mov_b32_e32 v10, v14
	v_mov_b32_e32 v11, v15
	buffer_store_dwordx4 v[8:11], v228, s[16:19], 0 offen
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v20, v16
	v_mov_b32_e32 v21, v17
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v229, s[16:19], 0 offen
	s_nop 1
	v_mov_b32_e32 v20, v24
	v_mov_b32_e32 v21, v25
	buffer_store_dwordx4 v[18:21], v230, s[16:19], 0 offen
	v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[24:27], v[162:165]
	v_mfma_f32_16x16x32_f16 v[6:9], a[76:79], a[28:31], v[6:9]
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[24:27], v[166:169]
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[28:31], v[10:13]
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[24:27], v[170:173]
	v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[28:31], v[14:17]
	v_mfma_f32_16x16x32_f16 v[18:21], a[88:91], a[24:27], v[174:177]
	v_mfma_f32_16x16x32_f16 v[18:21], a[92:95], a[28:31], v[18:21]
	v_mfma_f32_16x16x32_f16 v[22:25], a[68:71], a[16:19], v[182:185]
	v_mfma_f32_16x16x32_f16 v[22:25], a[76:79], a[20:23], v[22:25]
	v_mfma_f32_16x16x32_f16 v[26:29], a[64:67], a[16:19], v[186:189]
	v_mfma_f32_16x16x32_f16 v[26:29], a[72:75], a[20:23], v[26:29]
	v_mfma_f32_16x16x32_f16 v[30:33], a[80:83], a[16:19], v[190:193]
	v_mfma_f32_16x16x32_f16 v[30:33], a[84:87], a[20:23], v[30:33]
	v_mfma_f32_16x16x32_f16 v[34:37], a[88:91], a[16:19], v[178:181]
	v_mfma_f32_16x16x32_f16 v[34:37], a[92:95], a[20:23], v[34:37]
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v7, v8, v9
	v_cvt_pk_f16_f32 v10, v10, v11
	v_cvt_pk_f16_f32 v11, v12, v13
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_cvt_pk_f16_f32 v8, v22, v23
	v_cvt_pk_f16_f32 v9, v24, v25
	v_cvt_pk_f16_f32 v12, v26, v27
	v_cvt_pk_f16_f32 v13, v28, v29
	v_cvt_pk_f16_f32 v16, v30, v31
	v_cvt_pk_f16_f32 v17, v32, v33
	v_cvt_pk_f16_f32 v20, v34, v35
	v_cvt_pk_f16_f32 v21, v36, v37
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v2, v[6:9]
	ds_write_b128 v3, v[10:13]
	ds_write_b128 v4, v[14:17]
	ds_write_b128 v5, v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[6:9], v226
	ds_read_b128 v[12:15], v226 offset:256
	ds_read_b128 v[16:19], v226 offset:128
	ds_read_b128 v[22:25], v226 offset:384
	s_and_b32 s13, s3, 0xffff
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v10, v6
	v_mov_b32_e32 v11, v7
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v227, s[12:15], 0 offen
	s_nop 1
	v_mov_b32_e32 v10, v14
	v_mov_b32_e32 v11, v15
	buffer_store_dwordx4 v[8:11], v228, s[12:15], 0 offen
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v20, v16
	v_mov_b32_e32 v21, v17
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v229, s[12:15], 0 offen
	s_nop 1
	v_mov_b32_e32 v20, v24
	v_mov_b32_e32 v21, v25
	buffer_store_dwordx4 v[18:21], v230, s[12:15], 0 offen
	v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[8:11], v[130:133]
	v_mfma_f32_16x16x32_f16 v[6:9], a[76:79], a[12:15], v[6:9]
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[8:11], v[134:137]
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[12:15], v[10:13]
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[8:11], v[138:141]
	v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[12:15], v[14:17]
	v_mfma_f32_16x16x32_f16 v[18:21], a[88:91], a[8:11], v[142:145]
	v_mfma_f32_16x16x32_f16 v[18:21], a[92:95], a[12:15], v[18:21]
	v_mfma_f32_16x16x32_f16 v[22:25], a[68:71], a[0:3], v[146:149]
	v_mfma_f32_16x16x32_f16 v[22:25], a[76:79], a[4:7], v[22:25]
	v_mfma_f32_16x16x32_f16 v[26:29], a[64:67], a[0:3], v[150:153]
	v_mfma_f32_16x16x32_f16 v[26:29], a[72:75], a[4:7], v[26:29]
	v_mfma_f32_16x16x32_f16 v[30:33], a[80:83], a[0:3], v[154:157]
	v_mfma_f32_16x16x32_f16 v[30:33], a[84:87], a[4:7], v[30:33]
	v_mfma_f32_16x16x32_f16 v[34:37], a[88:91], a[0:3], v[158:161]
	v_mfma_f32_16x16x32_f16 v[34:37], a[92:95], a[4:7], v[34:37]
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v7, v8, v9
	v_cvt_pk_f16_f32 v10, v10, v11
	v_cvt_pk_f16_f32 v11, v12, v13
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_cvt_pk_f16_f32 v8, v22, v23
	v_cvt_pk_f16_f32 v9, v24, v25
	v_cvt_pk_f16_f32 v12, v26, v27
	v_cvt_pk_f16_f32 v13, v28, v29
	v_cvt_pk_f16_f32 v16, v30, v31
	v_cvt_pk_f16_f32 v17, v32, v33
	v_cvt_pk_f16_f32 v20, v34, v35
	v_cvt_pk_f16_f32 v21, v36, v37
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v2, v[6:9]
	ds_write_b128 v3, v[10:13]
	ds_write_b128 v4, v[14:17]
	ds_write_b128 v5, v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[6:9], v226
	ds_read_b128 v[12:15], v226 offset:256
	ds_read_b128 v[16:19], v226 offset:128
	ds_read_b128 v[22:25], v226 offset:384
	s_and_b32 s9, s2, 0xffff
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v10, v6
	v_mov_b32_e32 v11, v7
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[10:13], v227, s[8:11], 0 offen
	s_nop 1
	v_mov_b32_e32 v10, v14
	v_mov_b32_e32 v11, v15
	buffer_store_dwordx4 v[8:11], v228, s[8:11], 0 offen
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v20, v16
	v_mov_b32_e32 v21, v17
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v229, s[8:11], 0 offen
	s_nop 1
	v_mov_b32_e32 v20, v24
	v_mov_b32_e32 v21, v25
	buffer_store_dwordx4 v[18:21], v230, s[8:11], 0 offen
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v6, 0x1cdc0, v254
	ds_read_b128 v[118:121], v6
	ds_read_b128 v[122:125], v6 offset:64
	ds_read_b128 v[126:129], v6 offset:128
	ds_read_b128 v[130:133], v6 offset:192
	ds_read_b128 v[134:137], v6 offset:512
	ds_read_b128 v[138:141], v6 offset:576
	ds_read_b128 v[142:145], v6 offset:640
	ds_read_b128 v[146:149], v6 offset:704
	v_accvgpr_read_b32 v6, a158
	v_accvgpr_read_b32 v7, a159
	v_accvgpr_read_b32 v8, a160
	v_accvgpr_read_b32 v9, a161
	s_waitcnt lgkmcnt(7)
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[6:9], v[118:121], a[56:59], v[6:9]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[6:9], v[122:125], a[60:63], v[6:9]
	v_accvgpr_read_b32 v10, a212
	v_accvgpr_read_b32 v11, a213
	v_accvgpr_read_b32 v12, a214
	v_accvgpr_read_b32 v13, a215
	s_waitcnt lgkmcnt(5)
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[10:13], v[126:129], a[56:59], v[10:13]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[10:13], v[130:133], a[60:63], v[10:13]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[14:17], v[134:137], a[56:59], v[42:45]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[14:17], v[138:141], a[60:63], v[14:17]
	v_accvgpr_read_b32 v18, a162
	v_accvgpr_read_b32 v19, a163
	v_accvgpr_read_b32 v20, a164
	v_accvgpr_read_b32 v21, a165
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[18:21], v[142:145], a[56:59], v[18:21]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[18:21], v[146:149], a[60:63], v[18:21]
	v_mfma_f32_16x16x32_f16 v[22:25], v[118:121], a[48:51], v[38:41]
	v_mfma_f32_16x16x32_f16 v[22:25], v[122:125], a[52:55], v[22:25]
	v_mfma_f32_16x16x32_f16 v[26:29], v[126:129], a[48:51], v[60:63]
	v_mfma_f32_16x16x32_f16 v[26:29], v[130:133], a[52:55], v[26:29]
	v_mfma_f32_16x16x32_f16 v[30:33], v[134:137], a[48:51], v[64:67]
	v_mfma_f32_16x16x32_f16 v[30:33], v[138:141], a[52:55], v[30:33]
	v_mfma_f32_16x16x32_f16 v[34:37], v[142:145], a[48:51], v[68:71]
	v_mfma_f32_16x16x32_f16 v[34:37], v[146:149], a[52:55], v[34:37]
	v_mfma_f32_16x16x32_f16 v[38:41], v[118:121], a[40:43], v[72:75]
	v_mfma_f32_16x16x32_f16 v[38:41], v[122:125], a[44:47], v[38:41]
	v_mfma_f32_16x16x32_f16 v[42:45], v[126:129], a[40:43], v[76:79]
	v_mfma_f32_16x16x32_f16 v[42:45], v[130:133], a[44:47], v[42:45]
	v_mfma_f32_16x16x32_f16 v[46:49], v[134:137], a[40:43], v[46:49]
	v_mfma_f32_16x16x32_f16 v[46:49], v[138:141], a[44:47], v[46:49]
	v_mfma_f32_16x16x32_f16 v[50:53], v[142:145], a[40:43], v[50:53]
	v_mfma_f32_16x16x32_f16 v[50:53], v[146:149], a[44:47], v[50:53]
	v_mfma_f32_16x16x32_f16 v[54:57], v[118:121], a[32:35], v[54:57]
	v_mfma_f32_16x16x32_f16 v[54:57], v[122:125], a[36:39], v[54:57]
	v_mfma_f32_16x16x32_f16 v[58:61], v[126:129], a[32:35], v[92:95]
	v_mfma_f32_16x16x32_f16 v[58:61], v[130:133], a[36:39], v[58:61]
	v_mfma_f32_16x16x32_f16 v[62:65], v[134:137], a[32:35], v[88:91]
	v_mfma_f32_16x16x32_f16 v[62:65], v[138:141], a[36:39], v[62:65]
	v_mfma_f32_16x16x32_f16 v[66:69], v[142:145], a[32:35], v[84:87]
	v_mfma_f32_16x16x32_f16 v[66:69], v[146:149], a[36:39], v[66:69]
	v_mfma_f32_16x16x32_f16 v[70:73], v[118:121], a[24:27], v[104:107]
	v_mfma_f32_16x16x32_f16 v[70:73], v[122:125], a[28:31], v[70:73]
	v_mfma_f32_16x16x32_f16 v[74:77], v[126:129], a[24:27], v[96:99]
	v_mfma_f32_16x16x32_f16 v[74:77], v[130:133], a[28:31], v[74:77]
	v_mfma_f32_16x16x32_f16 v[78:81], v[134:137], a[24:27], v[100:103]
	v_mfma_f32_16x16x32_f16 v[78:81], v[138:141], a[28:31], v[78:81]
	v_mfma_f32_16x16x32_f16 v[82:85], v[142:145], a[24:27], v[108:111]
	v_mfma_f32_16x16x32_f16 v[82:85], v[146:149], a[28:31], v[82:85]
	v_mfma_f32_16x16x32_f16 v[86:89], v[118:121], a[16:19], v[112:115]
	v_mfma_f32_16x16x32_f16 v[86:89], v[122:125], a[20:23], v[86:89]
	v_accvgpr_read_b32 v90, a168
	v_accvgpr_read_b32 v91, a169
	v_accvgpr_read_b32 v92, a170
	v_accvgpr_read_b32 v93, a171
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[90:93], v[126:129], a[16:19], v[90:93]
	v_mfma_f32_16x16x32_f16 v[90:93], v[130:133], a[20:23], v[90:93]
	v_accvgpr_read_b32 v94, a172
	v_accvgpr_read_b32 v95, a173
	v_accvgpr_read_b32 v96, a174
	v_accvgpr_read_b32 v97, a175
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[94:97], v[134:137], a[16:19], v[94:97]
	v_mfma_f32_16x16x32_f16 v[94:97], v[138:141], a[20:23], v[94:97]
	v_accvgpr_read_b32 v98, a178
	v_accvgpr_read_b32 v99, a179
	v_accvgpr_read_b32 v100, a180
	v_accvgpr_read_b32 v101, a181
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[98:101], v[142:145], a[16:19], v[98:101]
	v_mfma_f32_16x16x32_f16 v[98:101], v[146:149], a[20:23], v[98:101]
	v_accvgpr_read_b32 v102, a182
	v_accvgpr_read_b32 v103, a183
	v_accvgpr_read_b32 v104, a184
	v_accvgpr_read_b32 v105, a185
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[102:105], v[118:121], a[8:11], v[102:105]
	v_mfma_f32_16x16x32_f16 v[102:105], v[122:125], a[12:15], v[102:105]
	v_accvgpr_read_b32 v106, a186
	v_accvgpr_read_b32 v107, a187
	v_accvgpr_read_b32 v108, a188
	v_accvgpr_read_b32 v109, a189
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[106:109], v[126:129], a[8:11], v[106:109]
	v_mfma_f32_16x16x32_f16 v[106:109], v[130:133], a[12:15], v[106:109]
	v_accvgpr_read_b32 v110, a190
	v_accvgpr_read_b32 v111, a191
	v_accvgpr_read_b32 v112, a192
	v_accvgpr_read_b32 v113, a193
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[110:113], v[134:137], a[8:11], v[110:113]
	v_mfma_f32_16x16x32_f16 v[110:113], v[138:141], a[12:15], v[110:113]
	v_accvgpr_read_b32 v114, a194
	v_accvgpr_read_b32 v115, a195
	v_accvgpr_read_b32 v116, a196
	v_accvgpr_read_b32 v117, a197
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[114:117], v[142:145], a[8:11], v[114:117]
	v_mfma_f32_16x16x32_f16 v[114:117], v[146:149], a[12:15], v[114:117]
	v_accvgpr_read_b32 v150, a198
	v_accvgpr_read_b32 v151, a199
	v_accvgpr_read_b32 v152, a200
	v_accvgpr_read_b32 v153, a201
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[118:121], v[118:121], a[0:3], v[150:153]
	v_mfma_f32_16x16x32_f16 v[118:121], v[122:125], a[4:7], v[118:121]
	v_accvgpr_read_b32 v122, a132
	v_accvgpr_read_b32 v123, a133
	v_accvgpr_read_b32 v124, a134
	v_accvgpr_read_b32 v125, a135
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[122:125], v[126:129], a[0:3], v[122:125]
	v_mfma_f32_16x16x32_f16 v[122:125], v[130:133], a[4:7], v[122:125]
	v_accvgpr_read_b32 v126, a136
	v_accvgpr_read_b32 v127, a137
	v_accvgpr_read_b32 v128, a138
	v_accvgpr_read_b32 v129, a139
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[126:129], v[134:137], a[0:3], v[126:129]
	v_mfma_f32_16x16x32_f16 v[126:129], v[138:141], a[4:7], v[126:129]
	v_accvgpr_read_b32 v130, a140
	v_accvgpr_read_b32 v131, a141
	v_accvgpr_read_b32 v132, a142
	v_accvgpr_read_b32 v133, a143
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[130:133], v[142:145], a[0:3], v[130:133]
	v_mfma_f32_16x16x32_f16 v[130:133], v[146:149], a[4:7], v[130:133]
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v7, v8, v9
	v_cvt_pk_f16_f32 v10, v10, v11
	v_cvt_pk_f16_f32 v11, v12, v13
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_cvt_pk_f16_f32 v8, v22, v23
	v_cvt_pk_f16_f32 v9, v24, v25
	v_cvt_pk_f16_f32 v12, v26, v27
	v_cvt_pk_f16_f32 v13, v28, v29
	v_cvt_pk_f16_f32 v16, v30, v31
	v_cvt_pk_f16_f32 v17, v32, v33
	v_cvt_pk_f16_f32 v20, v34, v35
	v_cvt_pk_f16_f32 v21, v36, v37
	v_cvt_pk_f16_f32 v22, v38, v39
	v_cvt_pk_f16_f32 v23, v40, v41
	v_cvt_pk_f16_f32 v26, v42, v43
	v_cvt_pk_f16_f32 v27, v44, v45
	v_cvt_pk_f16_f32 v30, v46, v47
	v_cvt_pk_f16_f32 v31, v48, v49
	v_cvt_pk_f16_f32 v34, v50, v51
	v_cvt_pk_f16_f32 v35, v52, v53
	v_cvt_pk_f16_f32 v24, v54, v55
	v_cvt_pk_f16_f32 v25, v56, v57
	v_cvt_pk_f16_f32 v28, v58, v59
	v_cvt_pk_f16_f32 v29, v60, v61
	v_cvt_pk_f16_f32 v32, v62, v63
	v_cvt_pk_f16_f32 v33, v64, v65
	v_cvt_pk_f16_f32 v36, v66, v67
	v_cvt_pk_f16_f32 v37, v68, v69
	v_cvt_pk_f16_f32 v38, v70, v71
	v_cvt_pk_f16_f32 v39, v72, v73
	v_cvt_pk_f16_f32 v42, v74, v75
	v_cvt_pk_f16_f32 v43, v76, v77
	v_cvt_pk_f16_f32 v46, v78, v79
	v_cvt_pk_f16_f32 v47, v80, v81
	v_cvt_pk_f16_f32 v50, v82, v83
	v_cvt_pk_f16_f32 v51, v84, v85
	v_cvt_pk_f16_f32 v40, v86, v87
	v_cvt_pk_f16_f32 v41, v88, v89
	v_cvt_pk_f16_f32 v44, v90, v91
	v_cvt_pk_f16_f32 v45, v92, v93
	v_cvt_pk_f16_f32 v48, v94, v95
	v_cvt_pk_f16_f32 v49, v96, v97
	v_cvt_pk_f16_f32 v52, v98, v99
	v_cvt_pk_f16_f32 v53, v100, v101
	v_cvt_pk_f16_f32 v54, v102, v103
	v_cvt_pk_f16_f32 v55, v104, v105
	v_cvt_pk_f16_f32 v58, v106, v107
	v_cvt_pk_f16_f32 v59, v108, v109
	v_cvt_pk_f16_f32 v62, v110, v111
	v_cvt_pk_f16_f32 v63, v112, v113
	v_cvt_pk_f16_f32 v66, v114, v115
	v_cvt_pk_f16_f32 v67, v116, v117
	v_cvt_pk_f16_f32 v56, v118, v119
	v_cvt_pk_f16_f32 v57, v120, v121
	v_cvt_pk_f16_f32 v60, v122, v123
	v_cvt_pk_f16_f32 v61, v124, v125
	v_cvt_pk_f16_f32 v64, v126, v127
	v_cvt_pk_f16_f32 v65, v128, v129
	v_cvt_pk_f16_f32 v68, v130, v131
	v_cvt_pk_f16_f32 v69, v132, v133
	ds_write_b128 v2, v[6:9]
	ds_write_b128 v3, v[10:13]
	ds_write_b128 v4, v[14:17]
	ds_write_b128 v5, v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[6:9], v226
	ds_read_b128 v[12:15], v226 offset:256
	ds_read_b128 v[16:19], v226 offset:128
	ds_read_b128 v[70:73], v226 offset:384
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v2, v[22:25]
	ds_write_b128 v3, v[26:29]
	ds_write_b128 v4, v[30:33]
	ds_write_b128 v5, v[34:37]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[22:25], v226
	ds_read_b128 v[28:31], v226 offset:256
	ds_read_b128 v[32:35], v226 offset:128
	ds_read_b128 v[74:77], v226 offset:384
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v2, v[38:41]
	ds_write_b128 v3, v[42:45]
	ds_write_b128 v4, v[46:49]
	ds_write_b128 v5, v[50:53]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[38:41], v226
	ds_read_b128 v[44:47], v226 offset:256
	ds_read_b128 v[48:51], v226 offset:128
	ds_read_b128 v[78:81], v226 offset:384
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v2, v[54:57]
	ds_write_b128 v3, v[58:61]
	ds_write_b128 v4, v[62:65]
	ds_write_b128 v5, v[66:69]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[2:5], v226
	ds_read_b128 v[54:57], v226 offset:256
	ds_read_b128 v[58:61], v226 offset:128
	ds_read_b128 v[64:67], v226 offset:384
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, s6
	s_mov_b32 s3, s7
	v_mov_b32_e32 v10, v6
	v_mov_b32_e32 v11, v7
	buffer_store_dwordx4 v[10:13], v227, s[0:3], 0 offen
	s_nop 1
	v_mov_b32_e32 v10, v14
	v_mov_b32_e32 v11, v15
	buffer_store_dwordx4 v[8:11], v228, s[0:3], 0 offen
	v_mov_b32_e32 v68, v16
	v_mov_b32_e32 v69, v17
	buffer_store_dwordx4 v[68:71], v229, s[0:3], 0 offen
	v_mov_b32_e32 v20, v72
	v_mov_b32_e32 v21, v73
	buffer_store_dwordx4 v[18:21], v230, s[0:3], 0 offen
	v_mov_b32_e32 v26, v22
	v_mov_b32_e32 v27, v23
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_dwordx4 v[26:29], v0, s[0:3], 0 offen
	s_nop 1
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v31
	v_lshlrev_b32_e32 v0, 1, v1
	buffer_store_dwordx4 v[24:27], v0, s[0:3], 0 offen
	v_mov_b32_e32 v72, v32
	v_mov_b32_e32 v73, v33
	v_lshlrev_b32_e32 v0, 1, v255
	buffer_store_dwordx4 v[72:75], v0, s[0:3], 0 offen
	v_mov_b32_e32 v36, v76
	v_mov_b32_e32 v37, v77
	v_accvgpr_read_b32 v0, a103
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_dwordx4 v[34:37], v0, s[0:3], 0 offen
	v_mov_b32_e32 v42, v38
	v_mov_b32_e32 v43, v39
	v_accvgpr_read_b32 v0, a102
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_dwordx4 v[42:45], v0, s[0:3], 0 offen
	s_nop 1
	v_mov_b32_e32 v42, v46
	v_mov_b32_e32 v43, v47
	v_accvgpr_read_b32 v0, a101
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_dwordx4 v[40:43], v0, s[0:3], 0 offen
	v_mov_b32_e32 v76, v48
	v_mov_b32_e32 v77, v49
	v_accvgpr_read_b32 v0, a100
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_dwordx4 v[76:79], v0, s[0:3], 0 offen
	v_mov_b32_e32 v52, v80
	v_mov_b32_e32 v53, v81
	v_accvgpr_read_b32 v0, a99
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_dwordx4 v[50:53], v0, s[0:3], 0 offen
	s_waitcnt lgkmcnt(3)
	s_nop 0
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v3
	v_accvgpr_read_b32 v0, a98
	v_lshlrev_b32_e32 v0, 1, v0
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[52:55], v0, s[0:3], 0 offen
	v_mov_b32_e32 v6, v56
	v_mov_b32_e32 v7, v57
	v_accvgpr_read_b32 v0, a97
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_dwordx4 v[4:7], v0, s[0:3], 0 offen
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v62, v58
	v_mov_b32_e32 v63, v59
	v_accvgpr_read_b32 v1, a96
	v_lshlrev_b32_e32 v0, 1, v1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[62:65], v0, s[0:3], 0 offen
	s_nop 1
	v_mov_b32_e32 v62, v66
	v_mov_b32_e32 v63, v67
	v_add_lshl_u32 v0, v1, s21, 1
	buffer_store_dwordx4 v[60:63], v0, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v10
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 44
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
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
	.size	v10, .Lfunc_end0-v10
	.cfi_endproc
                                        ; -- End function
	.set v10.num_vgpr, 256
	.set v10.num_agpr, 256
	.set v10.numbered_sgpr, 56
	.set v10.num_named_barrier, 0
	.set v10.private_seg_size, 44
	.set v10.uses_vcc, 1
	.set v10.uses_flat_scratch, 0
	.set v10.has_dyn_sized_stack, 0
	.set v10.has_recursion, 0
	.set v10.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13264
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 44
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 512
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x76 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x50 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x3e DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1330                            ; DW_AT_call_line
	.byte	71                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x56:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	13                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6a:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp3                          ; DW_AT_low_pc
	.long	.Ltmp4-.Ltmp3                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	14                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton/study_matmul/gluon/matmul_kernels" ; string offset=24
.Linfo_string3:
	.asciz	"v10"                           ; string offset=86
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
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
    .max_flat_workgroup_size: 256
    .name:           v10
    .private_segment_fixed_size: 44
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         v10.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 10
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
