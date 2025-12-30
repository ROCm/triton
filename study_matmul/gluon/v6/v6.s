	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v6                              ; -- Begin function v6
	.p2align	8
	.type	v6,@function
v6:                                     ; @v6
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.6:
	.file	1 "/var/lib/jenkins/OAI-triton/study_matmul/gluon/matmul_kernels" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.7:
.LBB0_0:
	v_mov_b32_e32 v194, v0
	s_nop 0
	v_readfirstlane_b32 s23, v194
	s_bfe_u32 s24, s23, 0x20006
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
	s_xor_b32 s1, s16, s0
	s_ashr_i32 s1, s1, 31
	s_abs_i32 s8, s16
	s_abs_i32 s9, s0
	v_cvt_f32_u32_e32 v0, s9
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_mov_b32 s20, 0
	s_sub_i32 s14, 0, s9
	v_readfirstlane_b32 s15, v0
	s_mul_i32 s14, s14, s15
	s_mul_hi_u32 s14, s15, s14
	s_add_i32 s15, s15, s14
	s_mul_hi_u32 s14, s8, s15
	s_mul_i32 s15, s14, s9
	s_sub_i32 s8, s8, s15
	s_add_i32 s15, s14, 1
	s_sub_i32 s17, s8, s9
	s_cmp_ge_u32 s8, s9
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s8, s17, s8
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s8, s9
	s_cselect_b32 s8, s15, s14
	s_xor_b32 s8, s8, s1
	s_sub_i32 s1, s8, s1
	s_mul_i32 s0, s1, s0
	s_sub_i32 s8, s16, s0
	v_lshlrev_b32_e32 v0, 1, v194
	v_and_b32_e32 v0, 0x70, v0
	v_or_b32_e32 v1, s24, v0
	v_lshlrev_b32_e32 v0, 3, v194
	v_and_b32_e32 v0, 56, v0
	s_lshl_b32 s15, s1, 8
	s_mul_i32 s0, s15, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s25, s3, s1
	s_lshl_b32 s14, s8, 8
	s_mul_i32 s2, s14, s12
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s8, s4, s2
	s_addc_u32 s26, s5, s3
	s_lshl_b32 s4, s11, 2
	s_mul_i32 s1, s11, 0x74
	v_mad_u64_u32 v[2:3], s[2:3], v1, s11, v[0:1]
	v_add_u32_e32 v3, s4, v2
	v_add_u32_e32 v4, s4, v3
	v_add_u32_e32 v5, s4, v4
	v_add_u32_e32 v6, s1, v5
	v_add_u32_e32 v7, s4, v6
	v_add_u32_e32 v8, s4, v7
	s_lshl_b32 s1, s12, 2
	v_mad_u64_u32 v[0:1], s[2:3], v1, s12, v[0:1]
	v_add_u32_e32 v1, s1, v0
	v_add_u32_e32 v9, s1, v1
	v_add_u32_e32 v10, s1, v9
	s_lshl_b32 s1, s12, 8
	s_ashr_i32 s5, s1, 1
	s_add_i32 s33, s10, 63
	s_ashr_i32 s1, s33, 31
	s_lshr_b32 s1, s1, 26
	s_add_i32 s1, s33, s1
	s_ashr_i32 s27, s1, 6
	s_and_b32 s1, s25, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_mul_i32 s12, s24, 0x420
	s_add_i32 s21, s12, 0
	v_lshlrev_b32_e32 v54, 1, v2
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v54, s[0:3], 0 offen lds
	s_add_i32 s34, s12, 0x1080
	s_add_i32 m0, s21, 0x1080
	v_lshlrev_b32_e32 v55, 1, v3
	buffer_load_dwordx4 v55, s[0:3], 0 offen lds
	s_add_i32 s35, s12, 0x2100
	s_add_i32 m0, s21, 0x2100
	v_lshlrev_b32_e32 v56, 1, v4
	buffer_load_dwordx4 v56, s[0:3], 0 offen lds
	s_add_i32 s36, s12, 0x3180
	s_add_i32 m0, s21, 0x3180
	v_lshlrev_b32_e32 v57, 1, v5
	buffer_load_dwordx4 v57, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x4200
	v_lshlrev_b32_e32 v58, 1, v6
	buffer_load_dwordx4 v58, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x5280
	v_lshlrev_b32_e32 v59, 1, v7
	buffer_load_dwordx4 v59, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x6300
	v_lshlrev_b32_e32 v60, 1, v8
	buffer_load_dwordx4 v60, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x7380
	v_add_lshl_u32 v61, v8, s4, 1
	buffer_load_dwordx4 v61, s[0:3], 0 offen lds
	s_and_b32 s9, s26, 0xffff
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	s_add_i32 s1, 0, 0x107e0
	s_add_i32 s22, s1, s12
	v_lshlrev_b32_e32 v62, 1, v0
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v62, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s34
	v_lshlrev_b32_e32 v63, 1, v1
	buffer_load_dwordx4 v63, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s35
	v_lshlrev_b32_e32 v64, 1, v9
	buffer_load_dwordx4 v64, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s36
	v_lshlrev_b32_e32 v65, 1, v10
	buffer_load_dwordx4 v65, s[8:11], 0 offen lds
	s_add_i32 s4, 0, 0x18bc0
	s_add_i32 m0, s4, s12
	v_add_lshl_u32 v5, v0, s5, 1
	buffer_load_dwordx4 v5, s[8:11], 0 offen lds
	s_add_i32 m0, s4, s34
	v_add_lshl_u32 v6, v1, s5, 1
	buffer_load_dwordx4 v6, s[8:11], 0 offen lds
	s_add_i32 m0, s4, s35
	v_add_lshl_u32 v7, v9, s5, 1
	buffer_load_dwordx4 v7, s[8:11], 0 offen lds
	s_add_i32 m0, s4, s36
	v_add_lshl_u32 v66, v10, s5, 1
	buffer_load_dwordx4 v66, s[8:11], 0 offen lds
	s_add_u32 s28, s0, 0x80
	s_addc_u32 s4, s25, 0
	s_add_u32 s16, s8, 0x80
	s_addc_u32 s5, s26, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b32 s29, s4, 0xffff
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	s_add_i32 m0, s21, 0x8400
	s_nop 0
	buffer_load_dwordx4 v54, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0x9480
	s_nop 0
	buffer_load_dwordx4 v55, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xa500
	s_nop 0
	buffer_load_dwordx4 v56, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xb580
	s_nop 0
	buffer_load_dwordx4 v57, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xc600
	s_nop 0
	buffer_load_dwordx4 v58, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xd680
	s_nop 0
	buffer_load_dwordx4 v59, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xe700
	s_nop 0
	buffer_load_dwordx4 v60, s[28:31], 0 offen lds
	s_add_i32 m0, s21, 0xf780
	s_nop 0
	buffer_load_dwordx4 v61, s[28:31], 0 offen lds
	s_and_b32 s17, s5, 0xffff
	s_mov_b32 s18, s2
	s_mov_b32 s19, s3
	s_add_i32 s4, 0, 0x149e0
	s_add_i32 m0, s4, s12
	s_nop 0
	buffer_load_dwordx4 v62, s[16:19], 0 offen lds
	s_add_i32 m0, s4, s34
	s_nop 0
	buffer_load_dwordx4 v63, s[16:19], 0 offen lds
	s_add_i32 m0, s4, s35
	s_nop 0
	buffer_load_dwordx4 v64, s[16:19], 0 offen lds
	s_add_i32 m0, s4, s36
	s_nop 0
	buffer_load_dwordx4 v65, s[16:19], 0 offen lds
	s_add_i32 s4, 0, 0x1cdc0
	s_add_i32 m0, s4, s12
	s_nop 0
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s4, s34
	s_nop 0
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	s_add_i32 m0, s4, s35
	s_nop 0
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_add_i32 m0, s4, s36
	s_nop 0
	buffer_load_dwordx4 v66, s[16:19], 0 offen lds
	s_waitcnt vmcnt(20) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v195, 15, v194
	v_lshlrev_b32_e32 v1, 10, v195
	v_and_b32_e32 v2, 48, v194
	s_cmp_lt_u32 s24, 2
	s_cselect_b64 s[4:5], -1, 0
	s_and_b64 s[10:11], s[4:5], exec
	s_cselect_b32 s9, 0, 0x100
	v_or3_b32 v3, v2, s9, v1
	v_lshlrev_b32_e32 v4, 5, v195
	v_add3_u32 v255, v3, v4, 0
	ds_read_b128 a[16:19], v255
	ds_read_b128 a[20:23], v255 offset:64
	ds_read_b128 a[8:11], v255 offset:128
	ds_read_b128 a[12:15], v255 offset:192
	ds_read_b128 a[0:3], v255 offset:512
	ds_read_b128 a[4:7], v255 offset:576
	ds_read_b128 a[92:95], v255 offset:640
	ds_read_b128 a[88:91], v255 offset:704
	ds_read_b128 a[84:87], v255 offset:16896
	ds_read_b128 a[80:83], v255 offset:16960
	ds_read_b128 a[76:79], v255 offset:17024
	ds_read_b128 a[72:75], v255 offset:17088
	ds_read_b128 a[68:71], v255 offset:17408
	ds_read_b128 a[64:67], v255 offset:17472
	ds_read_b128 a[60:63], v255 offset:17536
	ds_read_b128 a[56:59], v255 offset:17600
	s_and_b32 s10, s23, 64
	v_or_b32_e32 v1, v1, v2
	v_add_u32_e32 v1, v1, v4
	v_lshl_add_u32 v0, s10, 2, v1
	v_add_u32_e32 v254, s1, v0
	ds_read_b128 a[48:51], v254
	ds_read_b128 a[52:55], v254 offset:64
	ds_read_b128 a[40:43], v254 offset:128
	ds_read_b128 a[44:47], v254 offset:192
	ds_read_b128 a[32:35], v254 offset:512
	ds_read_b128 a[36:39], v254 offset:576
	ds_read_b128 a[24:27], v254 offset:640
	ds_read_b128 a[28:31], v254 offset:704
	s_add_i32 s9, s27, -1
	s_cmpk_lt_i32 s33, 0x80
	s_cbranch_scc1 .LBB0_4
; %bb.1:                                ; %.lr.ph
	v_accvgpr_write_b32 a165, v195
	v_accvgpr_write_b32 a164, v194
	s_add_u32 s8, s8, 0x100
	s_addc_u32 s11, s26, 0
	s_add_u32 s16, s0, 0x100
	s_addc_u32 s17, s25, 0
	s_add_i32 s18, s27, -2
	s_max_i32 s19, s9, 1
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a177, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a183, v6
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a188, v7
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a189, v5
	v_accvgpr_write_b32 a197, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_bfrev_b32_e32 v1, 1
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s0, s20, 1
	s_xor_b32 s23, s0, 1
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[194:197], a[48:51], a[16:19], a[194:197]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 a[194:197], a[52:55], a[20:23], a[194:197]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[250:253], a[40:43], a[16:19], v[248:251]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[2:5], a[44:47], a[20:23], v[250:253]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[246:249], a[32:35], a[16:19], v[244:247]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[250:253], a[36:39], a[20:23], v[246:249]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[242:245], a[24:27], a[16:19], v[240:243]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[246:249], a[28:31], a[20:23], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[48:51], a[8:11], v[236:239]
	v_mfma_f32_16x16x32_f16 v[242:245], a[52:55], a[12:15], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[40:43], a[8:11], v[232:235]
	v_mfma_f32_16x16x32_f16 v[238:241], a[44:47], a[12:15], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], a[32:35], a[8:11], v[228:231]
	v_mfma_f32_16x16x32_f16 v[234:237], a[36:39], a[12:15], v[230:233]
	v_mfma_f32_16x16x32_f16 v[226:229], a[24:27], a[8:11], v[224:227]
	v_mfma_f32_16x16x32_f16 v[230:233], a[28:31], a[12:15], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[48:51], a[0:3], v[220:223]
	v_mfma_f32_16x16x32_f16 v[226:229], a[52:55], a[4:7], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[40:43], a[0:3], v[216:219]
	v_mfma_f32_16x16x32_f16 v[222:225], a[44:47], a[4:7], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[32:35], a[0:3], v[212:215]
	v_mfma_f32_16x16x32_f16 v[218:221], a[36:39], a[4:7], v[214:217]
	v_mfma_f32_16x16x32_f16 v[210:213], a[24:27], a[0:3], v[208:211]
	v_mfma_f32_16x16x32_f16 v[214:217], a[28:31], a[4:7], v[210:213]
	v_mfma_f32_16x16x32_f16 v[206:209], a[48:51], a[92:95], v[204:207]
	v_mfma_f32_16x16x32_f16 v[210:213], a[52:55], a[88:91], v[206:209]
	v_mfma_f32_16x16x32_f16 v[202:205], a[40:43], a[92:95], v[200:203]
	v_mfma_f32_16x16x32_f16 v[206:209], a[44:47], a[88:91], v[202:205]
	v_mfma_f32_16x16x32_f16 v[198:201], a[32:35], a[92:95], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[36:39], a[88:91], v[198:201]
	v_mfma_f32_16x16x32_f16 a[190:193], a[24:27], a[92:95], a[190:193]
	v_mfma_f32_16x16x32_f16 a[190:193], a[28:31], a[88:91], a[190:193]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	s_mul_i32 s24, s0, 0x4200
	s_add_i32 s1, s24, 0
	s_add_i32 s25, s1, 0x18bc0
	v_accvgpr_mov_b32 a201, a157
	v_accvgpr_mov_b32 a200, a156
	v_accvgpr_mov_b32 a199, a155
	v_accvgpr_mov_b32 a198, a154
	v_add_u32_e32 v6, s25, v0
	ds_read_b128 v[8:11], v6
	v_mfma_f32_16x16x32_f16 v[190:193], a[48:51], a[84:87], v[190:193]
	ds_read_b128 a[136:139], v6 offset:64
	v_mfma_f32_16x16x32_f16 v[190:193], a[52:55], a[80:83], v[190:193]
	ds_read_b128 a[140:143], v6 offset:128
	v_mfma_f32_16x16x32_f16 v[186:189], a[40:43], a[84:87], v[186:189]
	ds_read_b128 a[144:147], v6 offset:192
	v_mfma_f32_16x16x32_f16 v[186:189], a[44:47], a[80:83], v[186:189]
	ds_read_b128 a[148:151], v6 offset:512
	v_mfma_f32_16x16x32_f16 v[182:185], a[32:35], a[84:87], v[182:185]
	ds_read_b128 a[152:155], v6 offset:576
	v_mfma_f32_16x16x32_f16 v[182:185], a[36:39], a[80:83], v[182:185]
	ds_read_b128 a[156:159], v6 offset:640
	v_mfma_f32_16x16x32_f16 v[178:181], a[24:27], a[84:87], v[178:181]
	ds_read_b128 a[160:163], v6 offset:704
	v_mfma_f32_16x16x32_f16 v[178:181], a[28:31], a[80:83], v[178:181]
	s_cmp_eq_u32 s18, s20
	s_cselect_b64 vcc, -1, 0
	s_mul_i32 s26, s0, 0x8400
	s_and_b32 s1, s17, 0xffff
	s_mov_b32 s0, s16
	s_add_i32 s26, s21, s26
	v_cndmask_b32_e32 v6, v54, v1, vcc
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[174:177], a[48:51], a[76:79], v[174:177]
	v_mfma_f32_16x16x32_f16 v[174:177], a[52:55], a[72:75], v[174:177]
	v_mfma_f32_16x16x32_f16 v[170:173], a[40:43], a[76:79], v[170:173]
	v_mfma_f32_16x16x32_f16 v[170:173], a[44:47], a[72:75], v[170:173]
	s_add_i32 m0, s26, 0x1080
	v_cndmask_b32_e32 v6, v55, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[166:169], a[32:35], a[76:79], v[166:169]
	v_mfma_f32_16x16x32_f16 v[166:169], a[36:39], a[72:75], v[166:169]
	v_mfma_f32_16x16x32_f16 v[162:165], a[24:27], a[76:79], v[162:165]
	v_mfma_f32_16x16x32_f16 v[162:165], a[28:31], a[72:75], v[162:165]
	s_add_i32 m0, s26, 0x2100
	v_cndmask_b32_e32 v6, v56, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[158:161], a[48:51], a[68:71], v[158:161]
	v_mfma_f32_16x16x32_f16 v[158:161], a[52:55], a[64:67], v[158:161]
	v_mfma_f32_16x16x32_f16 v[154:157], a[40:43], a[68:71], v[154:157]
	v_mfma_f32_16x16x32_f16 v[154:157], a[44:47], a[64:67], v[154:157]
	s_add_i32 m0, s26, 0x3180
	v_cndmask_b32_e32 v6, v57, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[150:153], a[32:35], a[68:71], v[150:153]
	v_mfma_f32_16x16x32_f16 v[150:153], a[36:39], a[64:67], v[150:153]
	v_mfma_f32_16x16x32_f16 v[146:149], a[24:27], a[68:71], v[146:149]
	v_mfma_f32_16x16x32_f16 v[146:149], a[28:31], a[64:67], v[146:149]
	s_add_i32 m0, s26, 0x4200
	v_cndmask_b32_e32 v6, v58, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[142:145], a[48:51], a[60:63], v[142:145]
	v_mfma_f32_16x16x32_f16 v[142:145], a[52:55], a[56:59], v[142:145]
	v_mfma_f32_16x16x32_f16 v[138:141], a[40:43], a[60:63], v[138:141]
	v_mfma_f32_16x16x32_f16 v[138:141], a[44:47], a[56:59], v[138:141]
	s_add_i32 m0, s26, 0x5280
	v_cndmask_b32_e32 v6, v59, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[134:137], a[32:35], a[60:63], v[134:137]
	v_mfma_f32_16x16x32_f16 v[134:137], a[36:39], a[56:59], v[134:137]
	v_mfma_f32_16x16x32_f16 v[130:133], a[24:27], a[60:63], v[130:133]
	v_mfma_f32_16x16x32_f16 v[130:133], a[28:31], a[56:59], v[130:133]
	s_add_i32 m0, s26, 0x6300
	v_cndmask_b32_e32 v6, v60, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[126:129], v[8:11], a[16:19], v[126:129]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[126:129], a[136:139], a[20:23], v[126:129]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[122:125], a[140:143], a[16:19], v[122:125]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[122:125], a[144:147], a[20:23], v[122:125]
	s_add_i32 m0, s26, 0x7380
	v_cndmask_b32_e32 v6, v61, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[118:121], a[148:151], a[16:19], v[118:121]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[118:121], a[152:155], a[20:23], v[118:121]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[114:117], a[156:159], a[16:19], v[114:117]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[114:117], a[160:163], a[20:23], v[114:117]
	s_and_b32 s1, s11, 0xffff
	s_mov_b32 s0, s8
	s_add_i32 s24, s22, s24
	v_cndmask_b32_e32 v6, v62, v1, vcc
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[110:113], v[8:11], a[8:11], v[110:113]
	v_mfma_f32_16x16x32_f16 v[110:113], a[136:139], a[12:15], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[140:143], a[8:11], v[106:109]
	v_mfma_f32_16x16x32_f16 v[106:109], a[144:147], a[12:15], v[106:109]
	s_add_i32 m0, s24, 0x1080
	v_cndmask_b32_e32 v6, v63, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[102:105], a[148:151], a[8:11], v[102:105]
	v_mfma_f32_16x16x32_f16 v[102:105], a[152:155], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[156:159], a[8:11], v[98:101]
	v_mfma_f32_16x16x32_f16 v[98:101], a[160:163], a[12:15], v[98:101]
	s_add_i32 m0, s24, 0x2100
	v_cndmask_b32_e32 v6, v64, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[94:97], v[8:11], a[0:3], v[94:97]
	v_mfma_f32_16x16x32_f16 v[94:97], a[136:139], a[4:7], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[140:143], a[0:3], v[90:93]
	v_mfma_f32_16x16x32_f16 v[90:93], a[144:147], a[4:7], v[90:93]
	s_add_i32 m0, s24, 0x3180
	v_cndmask_b32_e32 v6, v65, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[86:89], a[148:151], a[0:3], v[86:89]
	v_mfma_f32_16x16x32_f16 v[86:89], a[152:155], a[4:7], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[156:159], a[0:3], v[82:85]
	v_mfma_f32_16x16x32_f16 v[82:85], a[160:163], a[4:7], v[82:85]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	s_mul_i32 s24, s23, 0x8400
	v_add_u32_e32 v6, s24, v255
	ds_read_b128 a[16:19], v6
	v_mfma_f32_16x16x32_f16 v[78:81], v[8:11], a[92:95], v[78:81]
	ds_read_b128 a[20:23], v6 offset:64
	v_mfma_f32_16x16x32_f16 v[78:81], a[136:139], a[88:91], v[78:81]
	ds_read_b128 a[8:11], v6 offset:128
	v_mfma_f32_16x16x32_f16 v[74:77], a[140:143], a[92:95], v[74:77]
	ds_read_b128 a[12:15], v6 offset:192
	v_mfma_f32_16x16x32_f16 v[74:77], a[144:147], a[88:91], v[74:77]
	ds_read_b128 a[0:3], v6 offset:512
	v_mfma_f32_16x16x32_f16 v[70:73], a[148:151], a[92:95], v[70:73]
	ds_read_b128 a[4:7], v6 offset:576
	v_mfma_f32_16x16x32_f16 v[70:73], a[152:155], a[88:91], v[70:73]
	ds_read_b128 a[96:99], v6 offset:640
	v_accvgpr_write_b32 a182, v66
	v_mfma_f32_16x16x32_f16 a[184:187], a[156:159], a[92:95], a[184:187]
	ds_read_b128 a[100:103], v6 offset:704
	v_mfma_f32_16x16x32_f16 a[184:187], a[160:163], a[88:91], a[184:187]
	ds_read_b128 a[104:107], v6 offset:16896
	v_accvgpr_write_b32 a92, v62
	v_accvgpr_write_b32 a93, v63
	v_accvgpr_write_b32 a94, v64
	v_accvgpr_write_b32 a95, v65
	v_mfma_f32_16x16x32_f16 a[166:169], v[8:11], a[84:87], a[166:169]
	ds_read_b128 a[108:111], v6 offset:16960
	v_mfma_f32_16x16x32_f16 a[166:169], a[136:139], a[80:83], a[166:169]
	ds_read_b128 a[112:115], v6 offset:17024
	v_accvgpr_write_b32 a88, v58
	v_accvgpr_write_b32 a89, v59
	v_accvgpr_write_b32 a90, v60
	v_accvgpr_write_b32 a91, v61
	v_mfma_f32_16x16x32_f16 v[58:61], a[140:143], a[84:87], v[46:49]
	ds_read_b128 a[116:119], v6 offset:17088
	v_mfma_f32_16x16x32_f16 v[66:69], a[144:147], a[80:83], v[58:61]
	ds_read_b128 a[120:123], v6 offset:17408
	v_mfma_f32_16x16x32_f16 a[170:173], a[148:151], a[84:87], a[170:173]
	ds_read_b128 a[124:127], v6 offset:17472
	v_mfma_f32_16x16x32_f16 a[170:173], a[152:155], a[80:83], a[170:173]
	ds_read_b128 a[128:131], v6 offset:17536
	v_mfma_f32_16x16x32_f16 v[50:53], a[156:159], a[84:87], v[42:45]
	ds_read_b128 a[132:135], v6 offset:17600
	v_mfma_f32_16x16x32_f16 v[62:65], a[160:163], a[80:83], v[50:53]
	s_mulk_i32 s23, 0x4200
	v_add_u32_e32 v6, s23, v254
	ds_read_b128 a[48:51], v6
	v_mfma_f32_16x16x32_f16 a[174:177], v[8:11], a[76:79], a[174:177]
	ds_read_b128 a[52:55], v6 offset:64
	v_mfma_f32_16x16x32_f16 a[174:177], a[136:139], a[72:75], a[174:177]
	ds_read_b128 a[40:43], v6 offset:128
	v_mfma_f32_16x16x32_f16 v[42:45], a[140:143], a[76:79], v[38:41]
	ds_read_b128 a[44:47], v6 offset:192
	v_mfma_f32_16x16x32_f16 v[38:41], a[144:147], a[72:75], v[42:45]
	ds_read_b128 a[32:35], v6 offset:512
	v_mfma_f32_16x16x32_f16 a[178:181], a[148:151], a[76:79], a[178:181]
	ds_read_b128 a[36:39], v6 offset:576
	v_mfma_f32_16x16x32_f16 a[178:181], a[152:155], a[72:75], a[178:181]
	ds_read_b128 a[24:27], v6 offset:640
	v_accvgpr_mov_b32 a80, a214
	v_accvgpr_mov_b32 a81, a215
	v_accvgpr_mov_b32 a82, a216
	v_accvgpr_mov_b32 a83, a217
	v_mfma_f32_16x16x32_f16 a[76:79], a[156:159], a[76:79], a[218:221]
	ds_read_b128 a[28:31], v6 offset:704
	v_mfma_f32_16x16x32_f16 a[218:221], a[160:163], a[72:75], a[76:79]
	s_add_i32 s23, s25, s12
	v_accvgpr_read_b32 v6, a189
	v_cndmask_b32_e32 v6, v6, v1, vcc
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[30:33], v[8:11], a[68:71], v[30:33]
	v_mfma_f32_16x16x32_f16 v[46:49], a[136:139], a[64:67], v[30:33]
	v_mfma_f32_16x16x32_f16 v[26:29], a[140:143], a[68:71], v[26:29]
	v_mfma_f32_16x16x32_f16 v[26:29], a[144:147], a[64:67], v[26:29]
	s_add_i32 m0, s23, 0x1080
	v_accvgpr_read_b32 v6, a183
	v_cndmask_b32_e32 v6, v6, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_accvgpr_mov_b32 a72, a210
	v_accvgpr_mov_b32 a73, a211
	v_accvgpr_mov_b32 a74, a212
	v_accvgpr_mov_b32 a75, a213
	v_mfma_f32_16x16x32_f16 a[76:79], a[148:151], a[68:71], a[80:83]
	v_accvgpr_read_b32 v61, a91
	v_accvgpr_read_b32 v60, a90
	v_accvgpr_read_b32 v59, a89
	v_accvgpr_read_b32 v58, a88
	v_mfma_f32_16x16x32_f16 a[214:217], a[152:155], a[64:67], a[76:79]
	v_mfma_f32_16x16x32_f16 v[18:21], a[156:159], a[68:71], v[18:21]
	v_mfma_f32_16x16x32_f16 v[18:21], a[160:163], a[64:67], v[18:21]
	s_add_i32 m0, s23, 0x2100
	v_accvgpr_read_b32 v6, a188
	v_cndmask_b32_e32 v6, v6, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_accvgpr_mov_b32 a64, a206
	v_accvgpr_mov_b32 a65, a207
	v_accvgpr_mov_b32 a66, a208
	v_accvgpr_mov_b32 a67, a209
	v_mfma_f32_16x16x32_f16 a[68:71], v[8:11], a[60:63], a[72:75]
	v_mov_b64_e32 v[42:43], v[62:63]
	v_mov_b64_e32 v[44:45], v[64:65]
	v_accvgpr_read_b32 v65, a95
	v_accvgpr_read_b32 v64, a94
	v_accvgpr_read_b32 v63, a93
	v_accvgpr_read_b32 v62, a92
	v_mfma_f32_16x16x32_f16 a[210:213], a[136:139], a[56:59], a[68:71]
	v_mfma_f32_16x16x32_f16 a[64:67], a[140:143], a[60:63], a[64:67]
	v_mov_b64_e32 v[30:31], v[46:47]
	v_mov_b64_e32 v[32:33], v[48:49]
	v_mov_b64_e32 v[46:47], v[66:67]
	v_mov_b64_e32 v[48:49], v[68:69]
	v_accvgpr_read_b32 v66, a182
	v_mfma_f32_16x16x32_f16 a[206:209], a[144:147], a[56:59], a[64:67]
	s_add_i32 m0, s23, 0x3180
	v_cndmask_b32_e32 v6, v66, v1, vcc
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[64:67], a[148:151], a[60:63], a[202:205]
	v_mfma_f32_16x16x32_f16 a[202:205], a[152:155], a[56:59], a[64:67]
	v_mfma_f32_16x16x32_f16 a[60:63], a[156:159], a[60:63], a[198:201]
	v_mov_b64_e32 v[200:201], v[206:207]
	v_mov_b64_e32 v[202:203], v[208:209]
	v_mov_b64_e32 v[204:205], v[210:211]
	v_mov_b64_e32 v[206:207], v[212:213]
	v_mov_b64_e32 v[208:209], v[214:215]
	v_mov_b64_e32 v[210:211], v[216:217]
	v_mov_b64_e32 v[212:213], v[218:219]
	v_mov_b64_e32 v[214:215], v[220:221]
	v_mov_b64_e32 v[216:217], v[222:223]
	v_mov_b64_e32 v[218:219], v[224:225]
	v_mov_b64_e32 v[220:221], v[226:227]
	v_mov_b64_e32 v[222:223], v[228:229]
	v_mov_b64_e32 v[224:225], v[230:231]
	v_mov_b64_e32 v[226:227], v[232:233]
	v_mov_b64_e32 v[228:229], v[234:235]
	v_mov_b64_e32 v[230:231], v[236:237]
	v_mov_b64_e32 v[232:233], v[238:239]
	v_mov_b64_e32 v[234:235], v[240:241]
	v_mov_b64_e32 v[236:237], v[242:243]
	v_mov_b64_e32 v[238:239], v[244:245]
	v_mov_b64_e32 v[240:241], v[246:247]
	v_mov_b64_e32 v[242:243], v[248:249]
	v_mov_b64_e32 v[244:245], v[250:251]
	v_mov_b64_e32 v[246:247], v[252:253]
	v_mov_b64_e32 v[250:251], v[4:5]
	v_mov_b64_e32 v[248:249], v[2:3]
	v_mfma_f32_16x16x32_f16 a[154:157], a[160:163], a[56:59], a[60:63]
	s_add_u32 s16, s16, 0x80
	s_addc_u32 s17, s17, 0
	s_add_u32 s8, s8, 0x80
	s_addc_u32 s11, s11, 0
	s_add_i32 s20, s20, 1
	s_cmp_lg_u32 s19, s20
	s_waitcnt lgkmcnt(14)
	v_accvgpr_mov_b32 a92, a96
	v_accvgpr_mov_b32 a93, a97
	v_accvgpr_mov_b32 a94, a98
	v_accvgpr_mov_b32 a95, a99
	v_accvgpr_mov_b32 a88, a100
	v_accvgpr_mov_b32 a89, a101
	v_accvgpr_mov_b32 a90, a102
	v_accvgpr_mov_b32 a91, a103
	v_accvgpr_mov_b32 a84, a104
	v_accvgpr_mov_b32 a85, a105
	v_accvgpr_mov_b32 a86, a106
	v_accvgpr_mov_b32 a87, a107
	v_accvgpr_mov_b32 a80, a108
	v_accvgpr_mov_b32 a81, a109
	v_accvgpr_mov_b32 a82, a110
	v_accvgpr_mov_b32 a83, a111
	s_waitcnt lgkmcnt(13)
	v_accvgpr_mov_b32 a76, a112
	v_accvgpr_mov_b32 a77, a113
	v_accvgpr_mov_b32 a78, a114
	v_accvgpr_mov_b32 a79, a115
	s_waitcnt lgkmcnt(12)
	v_accvgpr_mov_b32 a72, a116
	v_accvgpr_mov_b32 a73, a117
	v_accvgpr_mov_b32 a74, a118
	v_accvgpr_mov_b32 a75, a119
	s_waitcnt lgkmcnt(11)
	v_accvgpr_mov_b32 a68, a120
	v_accvgpr_mov_b32 a69, a121
	v_accvgpr_mov_b32 a70, a122
	v_accvgpr_mov_b32 a71, a123
	s_waitcnt lgkmcnt(10)
	v_accvgpr_mov_b32 a64, a124
	v_accvgpr_mov_b32 a65, a125
	v_accvgpr_mov_b32 a66, a126
	v_accvgpr_mov_b32 a67, a127
	s_waitcnt lgkmcnt(9)
	v_accvgpr_mov_b32 a60, a128
	v_accvgpr_mov_b32 a61, a129
	v_accvgpr_mov_b32 a62, a130
	v_accvgpr_mov_b32 a63, a131
	s_waitcnt lgkmcnt(8)
	v_accvgpr_mov_b32 a56, a132
	v_accvgpr_mov_b32 a57, a133
	v_accvgpr_mov_b32 a58, a134
	v_accvgpr_mov_b32 a59, a135
	s_cbranch_scc1 .LBB0_2
; %bb.3:                                ; %Flow
	v_accvgpr_mov_b32 a150, a190
	v_accvgpr_mov_b32 a151, a191
	v_accvgpr_mov_b32 a152, a192
	v_accvgpr_mov_b32 a153, a193
	v_accvgpr_write_b32 a148, v0
	v_accvgpr_mov_b32 a144, a184
	v_accvgpr_mov_b32 a145, a185
	v_accvgpr_mov_b32 a146, a186
	v_accvgpr_mov_b32 a147, a187
	v_accvgpr_mov_b32 a59, a135
	v_accvgpr_mov_b32 a58, a134
	v_accvgpr_mov_b32 a57, a133
	v_accvgpr_mov_b32 a56, a132
	v_accvgpr_mov_b32 a63, a131
	v_accvgpr_mov_b32 a62, a130
	v_accvgpr_mov_b32 a61, a129
	v_accvgpr_mov_b32 a60, a128
	v_accvgpr_mov_b32 a67, a127
	v_accvgpr_mov_b32 a66, a126
	v_accvgpr_mov_b32 a65, a125
	v_accvgpr_mov_b32 a64, a124
	v_accvgpr_mov_b32 a71, a123
	v_accvgpr_mov_b32 a70, a122
	v_accvgpr_mov_b32 a69, a121
	v_accvgpr_mov_b32 a68, a120
	v_accvgpr_mov_b32 a75, a119
	v_accvgpr_mov_b32 a74, a118
	v_accvgpr_mov_b32 a73, a117
	v_accvgpr_mov_b32 a72, a116
	v_accvgpr_mov_b32 a79, a115
	v_accvgpr_mov_b32 a78, a114
	v_accvgpr_mov_b32 a77, a113
	v_accvgpr_mov_b32 a76, a112
	v_accvgpr_mov_b32 a83, a111
	v_accvgpr_mov_b32 a82, a110
	v_accvgpr_mov_b32 a81, a109
	v_accvgpr_mov_b32 a80, a108
	v_accvgpr_mov_b32 a87, a107
	v_accvgpr_mov_b32 a86, a106
	v_accvgpr_mov_b32 a85, a105
	v_accvgpr_mov_b32 a84, a104
	v_accvgpr_mov_b32 a91, a103
	v_accvgpr_mov_b32 a90, a102
	v_accvgpr_mov_b32 a89, a101
	v_accvgpr_mov_b32 a88, a100
	v_accvgpr_mov_b32 a95, a99
	v_accvgpr_mov_b32 a94, a98
	v_accvgpr_mov_b32 a93, a97
	v_accvgpr_mov_b32 a92, a96
	v_accvgpr_read_b32 v62, a170
	v_accvgpr_read_b32 v63, a171
	v_accvgpr_read_b32 v64, a172
	v_accvgpr_read_b32 v65, a173
	v_accvgpr_read_b32 v66, a166
	v_accvgpr_read_b32 v67, a167
	v_accvgpr_read_b32 v68, a168
	v_accvgpr_read_b32 v69, a169
	v_accvgpr_read_b32 v194, a164
	v_accvgpr_read_b32 v195, a165
	s_branch .LBB0_5
.LBB0_4:
	v_accvgpr_write_b32 a148, v0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a210, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a214, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a178, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a144, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a150, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v248, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a194, 0
.LBB0_5:                                ; %._crit_edge
	v_accvgpr_write_b32 a143, v49
	v_accvgpr_write_b32 a142, v48
	v_accvgpr_write_b32 a141, v47
	v_accvgpr_write_b32 a140, v46
	v_accvgpr_write_b32 a139, v45
	v_accvgpr_write_b32 a138, v44
	v_accvgpr_write_b32 a137, v43
	v_accvgpr_write_b32 a136, v42
	v_accvgpr_write_b32 a135, v41
	v_accvgpr_write_b32 a134, v40
	v_accvgpr_write_b32 a133, v39
	v_accvgpr_write_b32 a132, v38
	v_accvgpr_write_b32 a127, v33
	v_accvgpr_write_b32 a126, v32
	v_accvgpr_write_b32 a125, v31
	v_accvgpr_write_b32 a124, v30
	v_accvgpr_write_b32 a123, v29
	v_accvgpr_write_b32 a122, v28
	v_accvgpr_write_b32 a121, v27
	v_accvgpr_write_b32 a120, v26
	v_accvgpr_write_b32 a115, v21
	v_accvgpr_write_b32 a114, v20
	v_accvgpr_write_b32 a113, v19
	v_accvgpr_write_b32 a112, v18
	v_accvgpr_mov_b32 a96, a154
	v_accvgpr_mov_b32 a97, a155
	v_accvgpr_mov_b32 a98, a156
	v_accvgpr_mov_b32 a99, a157
	s_and_b64 s[0:1], s[4:5], exec
	s_cselect_b32 s0, 0, 32
	v_or_b32_e32 v0, s0, v195
	v_and_b32_e32 v1, 16, v194
	v_lshrrev_b32_e32 v6, 2, v194
	v_and_b32_e32 v6, 8, v6
	s_lshr_b32 s0, s10, 1
	v_or3_b32 v255, v6, v1, s0
	v_or_b32_e32 v254, 64, v255
	s_mul_i32 s0, s15, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[0:1], s[14:15], 1
	s_add_u32 s4, s2, s0
	s_addc_u32 s2, s3, s1
	s_add_u32 s0, s4, 0x100
	s_addc_u32 s1, s2, 0
	v_mul_lo_u32 v0, v0, s13
	s_lshl_b32 s3, s13, 4
	v_add_u32_e32 v1, s3, v0
	v_accvgpr_write_b32 a160, v0
	s_mul_i32 s13, s13, 48
	v_add_u32_e32 v0, s13, v1
	v_accvgpr_write_b32 a159, v1
	v_add_u32_e32 v1, s3, v0
	v_accvgpr_write_b32 a158, v0
	v_add_u32_e32 v0, s13, v1
	v_accvgpr_write_b32 a157, v1
	v_add_u32_e32 v1, s3, v0
	v_accvgpr_write_b32 a156, v0
	v_add_u32_e32 v0, s13, v1
	v_accvgpr_write_b32 a155, v1
	v_add_u32_e32 v1, s3, v0
	v_accvgpr_write_b32 a154, v1
	v_accvgpr_write_b32 a149, v0
	s_lshr_b32 s3, s9, 31
	s_add_i32 s3, s9, s3
	s_and_b32 s3, s3, -2
	s_sub_i32 s3, s9, s3
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[194:197], a[48:51], a[16:19], a[194:197]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 a[194:197], a[52:55], a[20:23], a[194:197]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[16:19], v[248:251]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[58:61], a[44:47], a[20:23], v[4:7]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[246:249], a[32:35], a[16:19], v[244:247]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[246:249], a[36:39], a[20:23], v[246:249]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[242:245], a[24:27], a[16:19], v[240:243]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[242:245], a[28:31], a[20:23], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[48:51], a[8:11], v[236:239]
	v_mfma_f32_16x16x32_f16 v[238:241], a[52:55], a[12:15], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[40:43], a[8:11], v[232:235]
	v_mfma_f32_16x16x32_f16 v[234:237], a[44:47], a[12:15], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], a[32:35], a[8:11], v[228:231]
	v_mfma_f32_16x16x32_f16 v[230:233], a[36:39], a[12:15], v[230:233]
	v_mfma_f32_16x16x32_f16 v[226:229], a[24:27], a[8:11], v[224:227]
	v_mfma_f32_16x16x32_f16 v[226:229], a[28:31], a[12:15], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[48:51], a[0:3], v[220:223]
	v_mfma_f32_16x16x32_f16 v[222:225], a[52:55], a[4:7], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[40:43], a[0:3], v[216:219]
	v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[4:7], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[32:35], a[0:3], v[212:215]
	v_mfma_f32_16x16x32_f16 v[214:217], a[36:39], a[4:7], v[214:217]
	v_mfma_f32_16x16x32_f16 v[210:213], a[24:27], a[0:3], v[208:211]
	v_mfma_f32_16x16x32_f16 v[210:213], a[28:31], a[4:7], v[210:213]
	v_mfma_f32_16x16x32_f16 v[206:209], a[48:51], a[92:95], v[204:207]
	v_mfma_f32_16x16x32_f16 v[206:209], a[52:55], a[88:91], v[206:209]
	v_mfma_f32_16x16x32_f16 v[202:205], a[40:43], a[92:95], v[200:203]
	v_mfma_f32_16x16x32_f16 v[202:205], a[44:47], a[88:91], v[202:205]
	v_mfma_f32_16x16x32_f16 v[198:201], a[32:35], a[92:95], v[196:199]
	v_mfma_f32_16x16x32_f16 v[198:201], a[36:39], a[88:91], v[198:201]
	v_accvgpr_read_b32 v0, a150
	v_accvgpr_read_b32 v1, a151
	v_accvgpr_read_b32 v2, a152
	v_accvgpr_read_b32 v3, a153
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[194:197], a[24:27], a[92:95], v[0:3]
	v_mfma_f32_16x16x32_f16 v[194:197], a[28:31], a[88:91], v[194:197]
	v_mfma_f32_16x16x32_f16 v[190:193], a[48:51], a[84:87], v[190:193]
	v_mfma_f32_16x16x32_f16 v[190:193], a[52:55], a[80:83], v[190:193]
	v_mfma_f32_16x16x32_f16 v[186:189], a[40:43], a[84:87], v[186:189]
	v_mfma_f32_16x16x32_f16 v[250:253], a[44:47], a[80:83], v[186:189]
	v_mfma_f32_16x16x32_f16 v[182:185], a[32:35], a[84:87], v[182:185]
	v_mfma_f32_16x16x32_f16 v[0:3], a[36:39], a[80:83], v[182:185]
	v_mfma_f32_16x16x32_f16 v[178:181], a[24:27], a[84:87], v[178:181]
	v_mfma_f32_16x16x32_f16 v[8:11], a[28:31], a[80:83], v[178:181]
	v_mfma_f32_16x16x32_f16 v[174:177], a[48:51], a[76:79], v[174:177]
	v_mfma_f32_16x16x32_f16 v[12:15], a[52:55], a[72:75], v[174:177]
	v_mfma_f32_16x16x32_f16 v[170:173], a[40:43], a[76:79], v[170:173]
	v_mfma_f32_16x16x32_f16 v[16:19], a[44:47], a[72:75], v[170:173]
	v_mfma_f32_16x16x32_f16 v[166:169], a[32:35], a[76:79], v[166:169]
	v_mfma_f32_16x16x32_f16 v[20:23], a[36:39], a[72:75], v[166:169]
	v_mfma_f32_16x16x32_f16 v[162:165], a[24:27], a[76:79], v[162:165]
	v_mfma_f32_16x16x32_f16 v[24:27], a[28:31], a[72:75], v[162:165]
	v_mfma_f32_16x16x32_f16 v[158:161], a[48:51], a[68:71], v[158:161]
	v_mfma_f32_16x16x32_f16 v[28:31], a[52:55], a[64:67], v[158:161]
	v_mfma_f32_16x16x32_f16 v[154:157], a[40:43], a[68:71], v[154:157]
	v_mfma_f32_16x16x32_f16 v[32:35], a[44:47], a[64:67], v[154:157]
	v_mfma_f32_16x16x32_f16 v[150:153], a[32:35], a[68:71], v[150:153]
	v_mfma_f32_16x16x32_f16 v[36:39], a[36:39], a[64:67], v[150:153]
	v_mfma_f32_16x16x32_f16 v[146:149], a[24:27], a[68:71], v[146:149]
	v_mfma_f32_16x16x32_f16 v[146:149], a[28:31], a[64:67], v[146:149]
	v_mfma_f32_16x16x32_f16 v[142:145], a[48:51], a[60:63], v[142:145]
	v_mfma_f32_16x16x32_f16 v[4:7], a[52:55], a[56:59], v[142:145]
	v_mfma_f32_16x16x32_f16 v[138:141], a[40:43], a[60:63], v[138:141]
	v_mfma_f32_16x16x32_f16 v[40:43], a[44:47], a[56:59], v[138:141]
	v_mfma_f32_16x16x32_f16 v[134:137], a[32:35], a[60:63], v[134:137]
	v_mfma_f32_16x16x32_f16 v[44:47], a[36:39], a[56:59], v[134:137]
	v_mfma_f32_16x16x32_f16 v[130:133], a[24:27], a[60:63], v[130:133]
	v_mfma_f32_16x16x32_f16 v[48:51], a[28:31], a[56:59], v[130:133]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_mulk_i32 s3, 0x4200
	s_add_i32 s3, s3, 0
	v_accvgpr_read_b32 v52, a148
	v_add_u32_e32 v52, s3, v52
	v_add_u32_e32 v52, 0x18bc0, v52
	ds_read_b128 a[24:27], v52
	ds_read_b128 a[28:31], v52 offset:64
	ds_read_b128 a[32:35], v52 offset:128
	ds_read_b128 a[36:39], v52 offset:192
	ds_read_b128 a[40:43], v52 offset:512
	ds_read_b128 a[44:47], v52 offset:576
	ds_read_b128 a[48:51], v52 offset:640
	ds_read_b128 a[52:55], v52 offset:704
	v_accvgpr_read_b32 v54, a194
	v_accvgpr_read_b32 v55, a195
	v_accvgpr_read_b32 v56, a196
	v_accvgpr_read_b32 v57, a197
	v_cvt_pk_f16_f32 v182, v54, v55
	v_cvt_pk_f16_f32 v183, v56, v57
	v_cvt_pk_f16_f32 v184, v58, v59
	v_cvt_pk_f16_f32 v185, v60, v61
	v_cvt_pk_f16_f32 v166, v246, v247
	v_cvt_pk_f16_f32 v167, v248, v249
	v_cvt_pk_f16_f32 v168, v242, v243
	v_cvt_pk_f16_f32 v169, v244, v245
	v_cvt_pk_f16_f32 v186, v238, v239
	v_cvt_pk_f16_f32 v187, v240, v241
	v_cvt_pk_f16_f32 v188, v234, v235
	v_cvt_pk_f16_f32 v189, v236, v237
	v_cvt_pk_f16_f32 v174, v230, v231
	v_cvt_pk_f16_f32 v175, v232, v233
	v_cvt_pk_f16_f32 v176, v226, v227
	v_cvt_pk_f16_f32 v177, v228, v229
	v_cvt_pk_f16_f32 v170, v222, v223
	v_cvt_pk_f16_f32 v171, v224, v225
	v_cvt_pk_f16_f32 v172, v218, v219
	v_cvt_pk_f16_f32 v173, v220, v221
	v_cvt_pk_f16_f32 v150, v214, v215
	v_cvt_pk_f16_f32 v151, v216, v217
	v_cvt_pk_f16_f32 v152, v210, v211
	v_cvt_pk_f16_f32 v153, v212, v213
	v_cvt_pk_f16_f32 v178, v206, v207
	v_cvt_pk_f16_f32 v179, v208, v209
	v_cvt_pk_f16_f32 v180, v202, v203
	v_cvt_pk_f16_f32 v181, v204, v205
	v_cvt_pk_f16_f32 v158, v198, v199
	v_cvt_pk_f16_f32 v159, v200, v201
	v_cvt_pk_f16_f32 v160, v194, v195
	v_cvt_pk_f16_f32 v161, v196, v197
	v_cvt_pk_f16_f32 v154, v190, v191
	v_cvt_pk_f16_f32 v155, v192, v193
	v_cvt_pk_f16_f32 v156, v250, v251
	v_cvt_pk_f16_f32 v157, v252, v253
	v_cvt_pk_f16_f32 v134, v0, v1
	v_cvt_pk_f16_f32 v135, v2, v3
	v_cvt_pk_f16_f32 v136, v8, v9
	v_cvt_pk_f16_f32 v137, v10, v11
	v_cvt_pk_f16_f32 v162, v12, v13
	v_cvt_pk_f16_f32 v163, v14, v15
	v_cvt_pk_f16_f32 v164, v16, v17
	v_cvt_pk_f16_f32 v165, v18, v19
	v_cvt_pk_f16_f32 v142, v20, v21
	v_cvt_pk_f16_f32 v143, v22, v23
	v_cvt_pk_f16_f32 v144, v24, v25
	v_cvt_pk_f16_f32 v145, v26, v27
	v_cvt_pk_f16_f32 v138, v28, v29
	v_cvt_pk_f16_f32 v139, v30, v31
	v_cvt_pk_f16_f32 v140, v32, v33
	v_cvt_pk_f16_f32 v141, v34, v35
	v_cvt_pk_f16_f32 v0, v36, v37
	v_cvt_pk_f16_f32 v1, v38, v39
	v_cvt_pk_f16_f32 v2, v146, v147
	v_cvt_pk_f16_f32 v3, v148, v149
	v_cvt_pk_f16_f32 v146, v4, v5
	v_cvt_pk_f16_f32 v147, v6, v7
	v_cvt_pk_f16_f32 v148, v40, v41
	v_cvt_pk_f16_f32 v149, v42, v43
	v_cvt_pk_f16_f32 v130, v44, v45
	v_cvt_pk_f16_f32 v131, v46, v47
	v_cvt_pk_f16_f32 v132, v48, v49
	v_cvt_pk_f16_f32 v133, v50, v51
	v_permlane16_swap_b32_e32 v182, v184
	v_permlane16_swap_b32_e32 v183, v185
	v_permlane16_swap_b32_e32 v186, v188
	v_permlane16_swap_b32_e32 v187, v189
	v_permlane16_swap_b32_e32 v166, v168
	v_permlane16_swap_b32_e32 v167, v169
	v_permlane16_swap_b32_e32 v174, v176
	v_permlane16_swap_b32_e32 v175, v177
	v_permlane16_swap_b32_e32 v170, v172
	v_permlane16_swap_b32_e32 v171, v173
	v_permlane16_swap_b32_e32 v178, v180
	v_permlane16_swap_b32_e32 v179, v181
	v_permlane16_swap_b32_e32 v150, v152
	v_permlane16_swap_b32_e32 v151, v153
	v_permlane16_swap_b32_e32 v158, v160
	v_permlane16_swap_b32_e32 v159, v161
	v_permlane16_swap_b32_e32 v154, v156
	v_permlane16_swap_b32_e32 v155, v157
	v_permlane16_swap_b32_e32 v162, v164
	v_permlane16_swap_b32_e32 v163, v165
	v_permlane16_swap_b32_e32 v134, v136
	v_permlane16_swap_b32_e32 v135, v137
	v_permlane16_swap_b32_e32 v142, v144
	v_permlane16_swap_b32_e32 v143, v145
	v_permlane16_swap_b32_e32 v138, v140
	v_permlane16_swap_b32_e32 v139, v141
	v_permlane16_swap_b32_e32 v146, v148
	v_permlane16_swap_b32_e32 v147, v149
	v_permlane16_swap_b32_e32 v0, v2
	v_permlane16_swap_b32_e32 v1, v3
	v_permlane16_swap_b32_e32 v130, v132
	v_permlane16_swap_b32_e32 v131, v133
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[16:19], v[126:129]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[126:129], a[28:31], a[20:23], v[4:7]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], a[16:19], v[122:125]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[122:125], a[36:39], a[20:23], v[4:7]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[16:19], v[118:121]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[118:121], a[44:47], a[20:23], v[4:7]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[4:7], a[48:51], a[16:19], v[114:117]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[114:117], a[52:55], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[8:11], v[110:113]
	v_mfma_f32_16x16x32_f16 v[110:113], a[28:31], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], a[8:11], v[106:109]
	v_mfma_f32_16x16x32_f16 v[106:109], a[36:39], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[8:11], v[102:105]
	v_mfma_f32_16x16x32_f16 v[102:105], a[44:47], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[48:51], a[8:11], v[98:101]
	v_mfma_f32_16x16x32_f16 v[98:101], a[52:55], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[0:3], v[94:97]
	v_mfma_f32_16x16x32_f16 v[94:97], a[28:31], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], a[0:3], v[90:93]
	v_mfma_f32_16x16x32_f16 v[90:93], a[36:39], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[0:3], v[86:89]
	v_mfma_f32_16x16x32_f16 v[4:7], a[44:47], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[8:11], a[48:51], a[0:3], v[82:85]
	v_mfma_f32_16x16x32_f16 v[8:11], a[52:55], a[4:7], v[8:11]
	v_mfma_f32_16x16x32_f16 v[12:15], a[24:27], a[92:95], v[78:81]
	v_mfma_f32_16x16x32_f16 v[12:15], a[28:31], a[88:91], v[12:15]
	v_mfma_f32_16x16x32_f16 v[16:19], a[32:35], a[92:95], v[74:77]
	v_mfma_f32_16x16x32_f16 v[16:19], a[36:39], a[88:91], v[16:19]
	v_mfma_f32_16x16x32_f16 v[20:23], a[40:43], a[92:95], v[70:73]
	v_mfma_f32_16x16x32_f16 v[20:23], a[44:47], a[88:91], v[20:23]
	v_accvgpr_read_b32 v24, a144
	v_accvgpr_read_b32 v25, a145
	v_accvgpr_read_b32 v26, a146
	v_accvgpr_read_b32 v27, a147
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[24:27], a[48:51], a[92:95], v[24:27]
	v_mfma_f32_16x16x32_f16 v[24:27], a[52:55], a[88:91], v[24:27]
	v_mfma_f32_16x16x32_f16 v[28:31], a[24:27], a[84:87], v[66:69]
	v_mfma_f32_16x16x32_f16 v[28:31], a[28:31], a[80:83], v[28:31]
	v_accvgpr_read_b32 v32, a140
	v_accvgpr_read_b32 v33, a141
	v_accvgpr_read_b32 v34, a142
	v_accvgpr_read_b32 v35, a143
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[32:35], a[32:35], a[84:87], v[32:35]
	v_mfma_f32_16x16x32_f16 v[32:35], a[36:39], a[80:83], v[32:35]
	v_mfma_f32_16x16x32_f16 v[36:39], a[40:43], a[84:87], v[62:65]
	v_mfma_f32_16x16x32_f16 v[36:39], a[44:47], a[80:83], v[36:39]
	v_accvgpr_read_b32 v40, a136
	v_accvgpr_read_b32 v41, a137
	v_accvgpr_read_b32 v42, a138
	v_accvgpr_read_b32 v43, a139
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[40:43], a[48:51], a[84:87], v[40:43]
	v_mfma_f32_16x16x32_f16 v[40:43], a[52:55], a[80:83], v[40:43]
	v_accvgpr_read_b32 v44, a174
	v_accvgpr_read_b32 v45, a175
	v_accvgpr_read_b32 v46, a176
	v_accvgpr_read_b32 v47, a177
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[44:47], a[24:27], a[76:79], v[44:47]
	v_mfma_f32_16x16x32_f16 v[44:47], a[28:31], a[72:75], v[44:47]
	v_accvgpr_read_b32 v48, a132
	v_accvgpr_read_b32 v49, a133
	v_accvgpr_read_b32 v50, a134
	v_accvgpr_read_b32 v51, a135
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[48:51], a[32:35], a[76:79], v[48:51]
	v_mfma_f32_16x16x32_f16 v[48:51], a[36:39], a[72:75], v[48:51]
	v_accvgpr_read_b32 v52, a178
	v_accvgpr_read_b32 v53, a179
	v_accvgpr_read_b32 v54, a180
	v_accvgpr_read_b32 v55, a181
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[52:55], a[40:43], a[76:79], v[52:55]
	v_mfma_f32_16x16x32_f16 v[52:55], a[44:47], a[72:75], v[52:55]
	v_accvgpr_read_b32 v56, a218
	v_accvgpr_read_b32 v57, a219
	v_accvgpr_read_b32 v58, a220
	v_accvgpr_read_b32 v59, a221
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[56:59], a[48:51], a[76:79], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[52:55], a[72:75], v[56:59]
	v_accvgpr_read_b32 v60, a124
	v_accvgpr_read_b32 v61, a125
	v_accvgpr_read_b32 v62, a126
	v_accvgpr_read_b32 v63, a127
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[60:63], a[24:27], a[68:71], v[60:63]
	v_mfma_f32_16x16x32_f16 v[60:63], a[28:31], a[64:67], v[60:63]
	v_accvgpr_read_b32 v64, a120
	v_accvgpr_read_b32 v65, a121
	v_accvgpr_read_b32 v66, a122
	v_accvgpr_read_b32 v67, a123
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[64:67], a[32:35], a[68:71], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[36:39], a[64:67], v[64:67]
	v_accvgpr_read_b32 v68, a214
	v_accvgpr_read_b32 v69, a215
	v_accvgpr_read_b32 v70, a216
	v_accvgpr_read_b32 v71, a217
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[68:71], a[40:43], a[68:71], v[68:71]
	v_mfma_f32_16x16x32_f16 v[68:71], a[44:47], a[64:67], v[68:71]
	v_accvgpr_read_b32 v72, a112
	v_accvgpr_read_b32 v73, a113
	v_accvgpr_read_b32 v74, a114
	v_accvgpr_read_b32 v75, a115
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[72:75], a[48:51], a[68:71], v[72:75]
	v_mfma_f32_16x16x32_f16 v[72:75], a[52:55], a[64:67], v[72:75]
	v_accvgpr_read_b32 v76, a210
	v_accvgpr_read_b32 v77, a211
	v_accvgpr_read_b32 v78, a212
	v_accvgpr_read_b32 v79, a213
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[76:79], a[24:27], a[60:63], v[76:79]
	v_mfma_f32_16x16x32_f16 v[76:79], a[28:31], a[56:59], v[76:79]
	v_accvgpr_read_b32 v80, a206
	v_accvgpr_read_b32 v81, a207
	v_accvgpr_read_b32 v82, a208
	v_accvgpr_read_b32 v83, a209
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[80:83], a[32:35], a[60:63], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[36:39], a[56:59], v[80:83]
	v_accvgpr_read_b32 v84, a202
	v_accvgpr_read_b32 v85, a203
	v_accvgpr_read_b32 v86, a204
	v_accvgpr_read_b32 v87, a205
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[84:87], a[40:43], a[60:63], v[84:87]
	v_mfma_f32_16x16x32_f16 v[84:87], a[44:47], a[56:59], v[84:87]
	v_accvgpr_read_b32 v193, a99
	v_accvgpr_read_b32 v192, a98
	v_accvgpr_read_b32 v191, a97
	v_accvgpr_read_b32 v190, a96
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[190:193], a[48:51], a[60:63], v[190:193]
	v_mfma_f32_16x16x32_f16 v[190:193], a[52:55], a[56:59], v[190:193]
	s_and_b32 s5, s2, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_accvgpr_read_b32 v89, a160
	v_add_lshl_u32 v194, v89, v255, 1
	buffer_store_dwordx4 v[182:185], v194, s[4:7], 0 offen
	v_accvgpr_read_b32 v88, a159
	s_nop 0
	v_add_lshl_u32 v182, v88, v255, 1
	buffer_store_dwordx4 v[186:189], v182, s[4:7], 0 offen
	v_add_lshl_u32 v183, v89, v254, 1
	buffer_store_dwordx4 v[166:169], v183, s[4:7], 0 offen
	s_nop 1
	v_add_lshl_u32 v166, v88, v254, 1
	buffer_store_dwordx4 v[174:177], v166, s[4:7], 0 offen
	v_accvgpr_read_b32 v89, a158
	v_add_lshl_u32 v167, v89, v255, 1
	buffer_store_dwordx4 v[170:173], v167, s[4:7], 0 offen
	v_accvgpr_read_b32 v88, a157
	v_add_lshl_u32 v168, v88, v255, 1
	buffer_store_dwordx4 v[178:181], v168, s[4:7], 0 offen
	v_add_lshl_u32 v169, v89, v254, 1
	buffer_store_dwordx4 v[150:153], v169, s[4:7], 0 offen
	s_nop 1
	v_add_lshl_u32 v150, v88, v254, 1
	buffer_store_dwordx4 v[158:161], v150, s[4:7], 0 offen
	v_accvgpr_read_b32 v89, a156
	v_add_lshl_u32 v151, v89, v255, 1
	buffer_store_dwordx4 v[154:157], v151, s[4:7], 0 offen
	v_accvgpr_read_b32 v88, a155
	v_add_lshl_u32 v152, v88, v255, 1
	buffer_store_dwordx4 v[162:165], v152, s[4:7], 0 offen
	v_add_lshl_u32 v153, v89, v254, 1
	buffer_store_dwordx4 v[134:137], v153, s[4:7], 0 offen
	s_nop 1
	v_add_lshl_u32 v134, v88, v254, 1
	buffer_store_dwordx4 v[142:145], v134, s[4:7], 0 offen
	v_accvgpr_read_b32 v89, a149
	v_add_lshl_u32 v135, v89, v255, 1
	buffer_store_dwordx4 v[138:141], v135, s[4:7], 0 offen
	v_accvgpr_read_b32 v88, a154
	v_add_lshl_u32 v136, v88, v255, 1
	buffer_store_dwordx4 v[146:149], v136, s[4:7], 0 offen
	v_add_lshl_u32 v137, v89, v254, 1
	buffer_store_dwordx4 v[0:3], v137, s[4:7], 0 offen
	v_add_lshl_u32 v138, v88, v254, 1
	buffer_store_dwordx4 v[130:133], v138, s[4:7], 0 offen
	v_cvt_pk_f16_f32 v0, v126, v127
	v_cvt_pk_f16_f32 v1, v128, v129
	v_cvt_pk_f16_f32 v2, v122, v123
	v_cvt_pk_f16_f32 v3, v124, v125
	v_cvt_pk_f16_f32 v118, v118, v119
	v_cvt_pk_f16_f32 v119, v120, v121
	v_cvt_pk_f16_f32 v120, v114, v115
	v_cvt_pk_f16_f32 v121, v116, v117
	v_cvt_pk_f16_f32 v110, v110, v111
	v_cvt_pk_f16_f32 v111, v112, v113
	v_cvt_pk_f16_f32 v112, v106, v107
	v_cvt_pk_f16_f32 v113, v108, v109
	v_cvt_pk_f16_f32 v102, v102, v103
	v_cvt_pk_f16_f32 v103, v104, v105
	v_cvt_pk_f16_f32 v104, v98, v99
	v_cvt_pk_f16_f32 v105, v100, v101
	v_cvt_pk_f16_f32 v88, v94, v95
	v_cvt_pk_f16_f32 v89, v96, v97
	v_cvt_pk_f16_f32 v90, v90, v91
	v_cvt_pk_f16_f32 v91, v92, v93
	v_cvt_pk_f16_f32 v4, v4, v5
	v_cvt_pk_f16_f32 v5, v6, v7
	v_cvt_pk_f16_f32 v6, v8, v9
	v_cvt_pk_f16_f32 v7, v10, v11
	v_cvt_pk_f16_f32 v8, v12, v13
	v_cvt_pk_f16_f32 v9, v14, v15
	v_cvt_pk_f16_f32 v10, v16, v17
	v_cvt_pk_f16_f32 v11, v18, v19
	v_cvt_pk_f16_f32 v12, v20, v21
	v_cvt_pk_f16_f32 v13, v22, v23
	v_cvt_pk_f16_f32 v14, v24, v25
	v_cvt_pk_f16_f32 v15, v26, v27
	v_cvt_pk_f16_f32 v16, v28, v29
	v_cvt_pk_f16_f32 v17, v30, v31
	v_cvt_pk_f16_f32 v18, v32, v33
	v_cvt_pk_f16_f32 v19, v34, v35
	v_cvt_pk_f16_f32 v20, v36, v37
	v_cvt_pk_f16_f32 v21, v38, v39
	v_cvt_pk_f16_f32 v22, v40, v41
	v_cvt_pk_f16_f32 v23, v42, v43
	v_cvt_pk_f16_f32 v24, v44, v45
	v_cvt_pk_f16_f32 v25, v46, v47
	v_cvt_pk_f16_f32 v26, v48, v49
	v_cvt_pk_f16_f32 v27, v50, v51
	v_cvt_pk_f16_f32 v28, v52, v53
	v_cvt_pk_f16_f32 v29, v54, v55
	v_cvt_pk_f16_f32 v30, v56, v57
	v_cvt_pk_f16_f32 v31, v58, v59
	v_cvt_pk_f16_f32 v32, v60, v61
	v_cvt_pk_f16_f32 v33, v62, v63
	v_cvt_pk_f16_f32 v34, v64, v65
	v_cvt_pk_f16_f32 v35, v66, v67
	v_cvt_pk_f16_f32 v36, v68, v69
	v_cvt_pk_f16_f32 v37, v70, v71
	v_cvt_pk_f16_f32 v38, v72, v73
	v_cvt_pk_f16_f32 v39, v74, v75
	v_cvt_pk_f16_f32 v40, v76, v77
	v_cvt_pk_f16_f32 v41, v78, v79
	v_cvt_pk_f16_f32 v42, v80, v81
	v_cvt_pk_f16_f32 v43, v82, v83
	v_cvt_pk_f16_f32 v44, v84, v85
	v_cvt_pk_f16_f32 v45, v86, v87
	v_cvt_pk_f16_f32 v46, v190, v191
	v_cvt_pk_f16_f32 v47, v192, v193
	v_permlane16_swap_b32_e32 v0, v2
	v_permlane16_swap_b32_e32 v1, v3
	v_permlane16_swap_b32_e32 v110, v112
	v_permlane16_swap_b32_e32 v111, v113
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v102, v104
	v_permlane16_swap_b32_e32 v103, v105
	v_permlane16_swap_b32_e32 v88, v90
	v_permlane16_swap_b32_e32 v89, v91
	v_permlane16_swap_b32_e32 v8, v10
	v_permlane16_swap_b32_e32 v9, v11
	v_permlane16_swap_b32_e32 v4, v6
	v_permlane16_swap_b32_e32 v5, v7
	v_permlane16_swap_b32_e32 v12, v14
	v_permlane16_swap_b32_e32 v13, v15
	v_permlane16_swap_b32_e32 v16, v18
	v_permlane16_swap_b32_e32 v17, v19
	v_permlane16_swap_b32_e32 v24, v26
	v_permlane16_swap_b32_e32 v25, v27
	v_permlane16_swap_b32_e32 v20, v22
	v_permlane16_swap_b32_e32 v21, v23
	v_permlane16_swap_b32_e32 v28, v30
	v_permlane16_swap_b32_e32 v29, v31
	v_permlane16_swap_b32_e32 v32, v34
	v_permlane16_swap_b32_e32 v33, v35
	v_permlane16_swap_b32_e32 v40, v42
	v_permlane16_swap_b32_e32 v41, v43
	v_permlane16_swap_b32_e32 v36, v38
	v_permlane16_swap_b32_e32 v37, v39
	v_permlane16_swap_b32_e32 v44, v46
	v_permlane16_swap_b32_e32 v45, v47
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, s6
	s_mov_b32 s3, s7
	buffer_store_dwordx4 v[0:3], v194, s[0:3], 0 offen
	buffer_store_dwordx4 v[110:113], v182, s[0:3], 0 offen
	buffer_store_dwordx4 v[118:121], v183, s[0:3], 0 offen
	buffer_store_dwordx4 v[102:105], v166, s[0:3], 0 offen
	buffer_store_dwordx4 v[88:91], v167, s[0:3], 0 offen
	buffer_store_dwordx4 v[8:11], v168, s[0:3], 0 offen
	buffer_store_dwordx4 v[4:7], v169, s[0:3], 0 offen
	buffer_store_dwordx4 v[12:15], v150, s[0:3], 0 offen
	buffer_store_dwordx4 v[16:19], v151, s[0:3], 0 offen
	buffer_store_dwordx4 v[24:27], v152, s[0:3], 0 offen
	buffer_store_dwordx4 v[20:23], v153, s[0:3], 0 offen
	buffer_store_dwordx4 v[28:31], v134, s[0:3], 0 offen
	buffer_store_dwordx4 v[32:35], v135, s[0:3], 0 offen
	buffer_store_dwordx4 v[40:43], v136, s[0:3], 0 offen
	buffer_store_dwordx4 v[36:39], v137, s[0:3], 0 offen
	buffer_store_dwordx4 v[44:47], v138, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v6
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
		.amdhsa_next_free_vgpr 478
		.amdhsa_next_free_sgpr 37
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
	.size	v6, .Lfunc_end0-v6
	.cfi_endproc
                                        ; -- End function
	.set v6.num_vgpr, 256
	.set v6.num_agpr, 222
	.set v6.numbered_sgpr, 37
	.set v6.num_named_barrier, 0
	.set v6.private_seg_size, 0
	.set v6.uses_vcc, 1
	.set v6.uses_flat_scratch, 0
	.set v6.has_dyn_sized_stack, 0
	.set v6.has_recursion, 0
	.set v6.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10828
; TotalNumSgprs: 43
; NumVgprs: 256
; NumAgprs: 222
; TotalNumVgprs: 478
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 59
; NumSGPRsForWavesPerEU: 43
; NumVGPRsForWavesPerEU: 478
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
	.byte	6                               ; Abbreviation Code
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
	.byte	1                               ; Abbrev [1] 0xb:0x77 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x51 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x2a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp4-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	488                             ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x56:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	12                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x6b:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp5                          ; DW_AT_low_pc
	.long	.Ltmp6-.Ltmp5                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	537                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
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
	.asciz	"v6"                            ; string offset=86
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     222
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
    .max_flat_workgroup_size: 256
    .name:           v6
    .private_segment_fixed_size: 0
    .sgpr_count:     43
    .sgpr_spill_count: 0
    .symbol:         v6.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     478
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
