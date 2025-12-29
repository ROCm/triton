	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v5                              ; -- Begin function v5
	.p2align	8
	.type	v5,@function
v5:                                     ; @v5
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.5:
	.file	1 "/var/lib/jenkins/OAI-triton/study_matmul/gluon/matmul_kernels" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.6:
.LBB0_0:
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s14, 0, s1
	s_abs_i32 s9, s16
	v_readfirstlane_b32 s20, v0
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s8, s16, s0
	s_bfe_u32 s23, s20, 0x20006
	s_ashr_i32 s8, s8, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshlrev_b32_e32 v2, 3, v0
	v_and_b32_e32 v2, 56, v2
	s_mul_i32 s25, s23, 0x420
	v_readfirstlane_b32 s15, v1
	s_mul_i32 s14, s14, s15
	s_mul_hi_u32 s14, s15, s14
	s_add_i32 s15, s15, s14
	s_mul_hi_u32 s14, s9, s15
	s_mul_i32 s15, s14, s1
	s_sub_i32 s9, s9, s15
	s_add_i32 s15, s14, 1
	s_sub_i32 s17, s9, s1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s9, s17, s9
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s9, s1
	s_cselect_b32 s1, s15, s14
	s_xor_b32 s1, s1, s8
	s_sub_i32 s1, s1, s8
	s_mul_i32 s0, s1, s0
	s_lshl_b32 s15, s1, 8
	s_sub_i32 s8, s16, s0
	s_mul_i32 s0, s15, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s21, s3, s1
	s_lshl_b32 s14, s8, 8
	s_mul_i32 s2, s14, s12
	v_lshlrev_b32_e32 v1, 1, v0
	s_ashr_i32 s3, s2, 31
	v_and_b32_e32 v1, 0x70, v1
	s_lshl_b64 s[2:3], s[2:3], 1
	v_or_b32_e32 v1, s23, v1
	s_add_u32 s8, s4, s2
	s_addc_u32 s22, s5, s3
	s_lshl_b32 s9, s11, 2
	v_mad_u64_u32 v[4:5], s[2:3], v1, s11, v[2:3]
	v_add_u32_e32 v5, s9, v4
	s_lshl_b32 s16, s12, 2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s12, v[2:3]
	v_add_u32_e32 v6, s9, v5
	v_add_u32_e32 v1, s16, v2
	s_mul_i32 s1, s11, 0x74
	v_add_u32_e32 v7, s9, v6
	v_add_u32_e32 v3, s16, v1
	v_add_u32_e32 v8, s1, v7
	s_mul_i32 s1, s12, 0x74
	v_add_u32_e32 v11, s16, v3
	s_add_i32 s24, s10, 63
	v_add_u32_e32 v12, s1, v11
	s_ashr_i32 s1, s24, 31
	s_lshr_b32 s1, s1, 26
	s_add_i32 s1, s24, s1
	s_add_i32 s5, s25, 0
	s_ashr_i32 s4, s1, 6
	s_and_b32 s1, s21, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_lshlrev_b32_e32 v4, 1, v4
	s_mov_b32 m0, s5
	v_lshlrev_b32_e32 v5, 1, v5
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x1080
	v_lshlrev_b32_e32 v6, 1, v6
	buffer_load_dwordx4 v5, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x2100
	v_lshlrev_b32_e32 v7, 1, v7
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x3180
	v_add_u32_e32 v9, s9, v8
	buffer_load_dwordx4 v7, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x4200
	v_lshlrev_b32_e32 v8, 1, v8
	v_add_u32_e32 v10, s9, v9
	buffer_load_dwordx4 v8, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x5280
	v_lshlrev_b32_e32 v9, 1, v9
	buffer_load_dwordx4 v9, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x6300
	v_lshlrev_b32_e32 v15, 1, v10
	buffer_load_dwordx4 v15, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x7380
	v_add_lshl_u32 v10, v10, s9, 1
	buffer_load_dwordx4 v10, s[0:3], 0 offen lds
	s_add_i32 s1, 0, 0x107e0
	s_add_i32 s12, s1, s25
	s_add_i32 s26, s25, 0x1080
	s_and_b32 s9, s22, 0xffff
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	v_lshlrev_b32_e32 v16, 1, v2
	s_mov_b32 m0, s12
	s_add_i32 s27, s25, 0x2100
	buffer_load_dwordx4 v16, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s26
	v_lshlrev_b32_e32 v17, 1, v1
	s_add_i32 s33, s25, 0x3180
	buffer_load_dwordx4 v17, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s27
	v_lshlrev_b32_e32 v18, 1, v3
	s_add_i32 s34, s25, 0x4200
	buffer_load_dwordx4 v18, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s33
	v_lshlrev_b32_e32 v11, 1, v11
	v_add_u32_e32 v13, s16, v12
	s_add_i32 s35, s25, 0x5280
	buffer_load_dwordx4 v11, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s34
	v_lshlrev_b32_e32 v12, 1, v12
	v_add_u32_e32 v14, s16, v13
	s_add_i32 s36, s25, 0x6300
	buffer_load_dwordx4 v12, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s35
	v_lshlrev_b32_e32 v13, 1, v13
	s_add_i32 s37, s25, 0x7380
	buffer_load_dwordx4 v13, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s36
	v_lshlrev_b32_e32 v19, 1, v14
	buffer_load_dwordx4 v19, s[8:11], 0 offen lds
	s_add_i32 m0, s1, s37
	v_add_lshl_u32 v14, v14, s16, 1
	s_add_u32 s28, s0, 0x80
	buffer_load_dwordx4 v14, s[8:11], 0 offen lds
	s_addc_u32 s9, s21, 0
	s_add_u32 s16, s8, 0x80
	s_addc_u32 s10, s22, 0
	s_and_b32 s29, s9, 0xffff
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	s_add_i32 m0, s5, 0x8400
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v4, s[28:31], 0 offen lds
	s_add_i32 m0, s5, 0x9480
	s_add_i32 s9, 0, 0x18be0
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	s_add_i32 m0, s5, 0xa500
	s_and_b32 s17, s10, 0xffff
	buffer_load_dwordx4 v6, s[28:31], 0 offen lds
	s_add_i32 m0, s5, 0xb580
	s_mov_b32 s18, s2
	buffer_load_dwordx4 v7, s[28:31], 0 offen lds
	s_add_i32 m0, s5, 0xc600
	s_mov_b32 s19, s3
	buffer_load_dwordx4 v8, s[28:31], 0 offen lds
	s_add_i32 m0, s5, 0xd680
	v_and_b32_e32 v1, 63, v0
	buffer_load_dwordx4 v9, s[28:31], 0 offen lds
	s_add_i32 m0, s5, 0xe700
	v_and_b32_e32 v3, 15, v0
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	s_add_i32 m0, s5, 0xf780
	v_lshl_or_b32 v2, s23, 6, v1
	buffer_load_dwordx4 v10, s[28:31], 0 offen lds
	s_add_i32 m0, s9, s25
	v_lshlrev_b32_e32 v1, 10, v3
	buffer_load_dwordx4 v16, s[16:19], 0 offen lds
	s_add_i32 m0, s9, s26
	v_accvgpr_write_b32 a128, v2
	buffer_load_dwordx4 v17, s[16:19], 0 offen lds
	s_add_i32 m0, s9, s27
	v_and_b32_e32 v0, 48, v0
	buffer_load_dwordx4 v18, s[16:19], 0 offen lds
	s_add_i32 m0, s9, s33
	v_accvgpr_write_b32 a129, v3
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_add_i32 m0, s9, s34
	v_lshlrev_b32_e32 v3, 5, v3
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_add_i32 m0, s9, s35
	s_nop 0
	buffer_load_dwordx4 v13, s[16:19], 0 offen lds
	s_add_i32 m0, s9, s36
	s_nop 0
	buffer_load_dwordx4 v19, s[16:19], 0 offen lds
	s_add_i32 m0, s9, s37
	s_movk_i32 s9, 0xb0
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds
	v_and_or_b32 v2, v2, s9, v1
	s_lshl_b32 s9, s20, 1
	s_and_b32 s9, s9, 0x80
	v_or3_b32 v0, v0, s9, v1
	v_add_u32_e32 v2, v2, v3
	v_add_u32_e32 v0, v0, v3
	v_accvgpr_write_b32 a130, v2
	v_add_u32_e32 v2, 0, v2
	v_accvgpr_write_b32 a131, v0
	v_add_u32_e32 v0, s1, v0
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[124:127], v2
	ds_read_b128 a[108:111], v2 offset:64
	ds_read_b128 a[120:123], v2 offset:256
	ds_read_b128 a[96:99], v2 offset:320
	ds_read_b128 a[116:119], v2 offset:512
	ds_read_b128 a[56:59], v2 offset:576
	ds_read_b128 a[112:115], v2 offset:768
	ds_read_b128 a[48:51], v2 offset:832
	ds_read_b128 a[100:103], v2 offset:16896
	ds_read_b128 a[44:47], v2 offset:16960
	ds_read_b128 a[60:63], v2 offset:17152
	ds_read_b128 a[12:15], v2 offset:17216
	ds_read_b128 a[52:55], v2 offset:17408
	ds_read_b128 a[8:11], v2 offset:17472
	ds_read_b128 a[104:107], v2 offset:17664
	ds_read_b128 a[0:3], v2 offset:17728
	ds_read_b128 a[72:75], v0
	ds_read_b128 a[16:19], v0 offset:64
	ds_read_b128 a[76:79], v0 offset:256
	ds_read_b128 a[24:27], v0 offset:320
	ds_read_b128 a[80:83], v0 offset:512
	ds_read_b128 a[28:31], v0 offset:576
	ds_read_b128 a[84:87], v0 offset:768
	ds_read_b128 a[32:35], v0 offset:832
	ds_read_b128 a[88:91], v0 offset:16896
	ds_read_b128 a[36:39], v0 offset:16960
	ds_read_b128 a[92:95], v0 offset:17152
	ds_read_b128 a[40:43], v0 offset:17216
	ds_read_b128 a[64:67], v0 offset:17408
	ds_read_b128 a[20:23], v0 offset:17472
	ds_read_b128 a[68:71], v0 offset:17664
	ds_read_b128 a[4:7], v0 offset:17728
	s_cmpk_lt_i32 s24, 0xc0
	s_mov_b32 s9, 0
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %.lr.ph.preheader
	s_add_u32 s8, s8, 0x100
	s_addc_u32 s10, s22, 0
	s_add_u32 s11, s0, 0x100
	v_accvgpr_write_b32 a134, v18
	v_mov_b32_e32 v18, 0
	v_accvgpr_write_b32 a138, v14
	v_mov_b32_e32 v14, 0
	v_accvgpr_write_b32 a146, v10
	v_mov_b32_e32 v10, 0
	v_accvgpr_write_b32 a150, v6
	v_mov_b32_e32 v6, 0
	s_addc_u32 s16, s21, 0
	s_max_i32 s17, s4, 3
	v_accvgpr_write_b32 a132, v16
	v_mov_b32_e32 v16, 0
	v_accvgpr_write_b32 a133, v17
	v_mov_b32_e32 v17, 0
	v_accvgpr_write_b32 a172, v18
	v_accvgpr_write_b32 a136, v12
	v_mov_b32_e32 v12, 0
	v_accvgpr_write_b32 a137, v13
	v_mov_b32_e32 v13, 0
	v_accvgpr_write_b32 a176, v14
	v_accvgpr_write_b32 a144, v8
	v_mov_b32_e32 v8, 0
	v_accvgpr_write_b32 a145, v9
	v_mov_b32_e32 v9, 0
	v_accvgpr_write_b32 a180, v10
	v_accvgpr_write_b32 a148, v4
	v_mov_b32_e32 v4, 0
	v_accvgpr_write_b32 a149, v5
	v_mov_b32_e32 v5, 0
	v_accvgpr_write_b32 a184, v6
	s_add_i32 s17, s17, -2
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
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
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
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
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a135, v19
	v_accvgpr_write_b32 a171, v17
	v_accvgpr_write_b32 a170, v16
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a139, v15
	v_accvgpr_write_b32 a175, v13
	v_accvgpr_write_b32 a174, v12
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a147, v11
	v_accvgpr_write_b32 a179, v9
	v_accvgpr_write_b32 a178, v8
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a151, v7
	v_accvgpr_write_b32 a183, v5
	v_accvgpr_write_b32 a182, v4
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a161, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_accvgpr_write_b32 a152, v0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a153, v2
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
.LBB0_2:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	s_and_b32 s18, s9, 1
	s_mul_i32 s19, s18, 0x8400
	s_add_i32 s21, s5, s19
	s_mov_b32 m0, s21
	s_and_b32 s1, s16, 0xffff
	s_mov_b32 s0, s11
	v_accvgpr_read_b32 v48, a148
	buffer_load_dwordx4 v48, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x1080
	v_accvgpr_read_b32 v48, a149
	buffer_load_dwordx4 v48, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x2100
	v_accvgpr_read_b32 v48, a150
	buffer_load_dwordx4 v48, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x3180
	v_accvgpr_read_b32 v48, a151
	buffer_load_dwordx4 v48, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x4200
	v_accvgpr_read_b32 v48, a144
	buffer_load_dwordx4 v48, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x5280
	v_accvgpr_read_b32 v48, a145
	v_accvgpr_write_b32 a165, v59
	buffer_load_dwordx4 v48, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x6300
	v_accvgpr_write_b32 a164, v58
	v_accvgpr_write_b32 a163, v57
	v_accvgpr_write_b32 a162, v56
	v_accvgpr_read_b32 v56, a139
	buffer_load_dwordx4 v56, s[0:3], 0 offen lds
	s_add_i32 m0, s21, 0x7380
	s_add_i32 s19, s12, s19
	v_accvgpr_read_b32 v48, a146
	buffer_load_dwordx4 v48, s[0:3], 0 offen lds
	s_mov_b32 m0, s19
	s_and_b32 s1, s10, 0xffff
	s_mov_b32 s0, s8
	v_accvgpr_read_b32 v4, a132
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s19, 0x1080
	v_accvgpr_read_b32 v4, a133
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s19, 0x2100
	v_accvgpr_read_b32 v4, a134
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s19, 0x3180
	v_accvgpr_read_b32 v4, a147
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s19, 0x4200
	v_accvgpr_read_b32 v4, a136
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s19, 0x5280
	v_accvgpr_read_b32 v4, a137
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s19, 0x6300
	v_accvgpr_read_b32 v4, a135
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s19, 0x7380
	v_accvgpr_read_b32 v4, a138
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[132:135], a[72:75], a[124:127], v[132:135]
	v_mov_b64_e32 v[48:49], v[46:47]
	v_mov_b64_e32 v[46:47], v[44:45]
	v_mov_b64_e32 v[44:45], v[42:43]
	v_mov_b64_e32 v[42:43], v[40:41]
	v_mov_b64_e32 v[40:41], v[38:39]
	v_mov_b64_e32 v[38:39], v[36:37]
	v_mov_b64_e32 v[36:37], v[34:35]
	v_mfma_f32_16x16x32_f16 v[56:59], a[16:19], a[108:111], v[132:135]
	v_mov_b64_e32 v[34:35], v[32:33]
	v_mov_b64_e32 v[32:33], v[30:31]
	v_mov_b64_e32 v[30:31], v[28:29]
	v_accvgpr_mov_b32 a189, a161
	s_lshl_b32 s0, s18, 14
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[28:31], a[92:95], a[104:107], v[30:33]
	v_accvgpr_mov_b32 a188, a160
	v_accvgpr_mov_b32 a187, a159
	v_accvgpr_write_b32 a157, v59
	v_mfma_f32_16x16x32_f16 v[32:35], a[88:91], a[104:107], v[34:37]
	v_accvgpr_write_b32 a156, v58
	v_accvgpr_write_b32 a155, v57
	v_accvgpr_write_b32 a154, v56
	v_mfma_f32_16x16x32_f16 v[36:39], a[84:87], a[104:107], v[38:41]
	v_accvgpr_mov_b32 a186, a158
	s_xor_b32 s0, s0, 0x4000
	s_lshr_b32 s1, s0, 5
	v_mfma_f32_16x16x32_f16 v[40:43], a[80:83], a[104:107], v[42:45]
	s_or_b32 s0, s1, s0
	s_lshl_b32 s0, s0, 1
	s_waitcnt vmcnt(16)
	v_mfma_f32_16x16x32_f16 v[44:47], a[76:79], a[104:107], v[46:49]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_add_u32 s11, s11, 0x80
	v_accvgpr_read_b32 v48, a162
	v_accvgpr_read_b32 v49, a163
	v_accvgpr_read_b32 v50, a164
	v_accvgpr_read_b32 v51, a165
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[104:107], v[52:55]
	s_addc_u32 s16, s16, 0
	s_add_u32 s8, s8, 0x80
	v_accvgpr_read_b32 v132, a154
	v_mfma_f32_16x16x32_f16 v[56:59], a[68:71], a[60:63], v[48:51]
	s_addc_u32 s10, s10, 0
	s_add_i32 s9, s9, 1
	v_accvgpr_read_b32 v133, a155
	v_mov_b64_e32 v[50:51], v[26:27]
	v_mov_b64_e32 v[48:49], v[24:25]
	v_mfma_f32_16x16x32_f16 a[190:193], a[72:75], a[104:107], a[190:193]
	v_accvgpr_read_b32 v134, a156
	v_accvgpr_read_b32 v135, a157
	s_cmp_lg_u32 s17, s9
	v_mfma_f32_16x16x32_f16 a[182:185], a[92:95], a[52:55], a[182:185]
	v_mfma_f32_16x16x32_f16 a[178:181], a[88:91], a[52:55], a[178:181]
	v_mfma_f32_16x16x32_f16 a[174:177], a[84:87], a[52:55], a[174:177]
	v_mfma_f32_16x16x32_f16 a[170:173], a[80:83], a[52:55], a[170:173]
	v_mfma_f32_16x16x32_f16 a[166:169], a[76:79], a[52:55], a[166:169]
	v_mfma_f32_16x16x32_f16 v[188:191], a[72:75], a[52:55], v[188:191]
	v_mfma_f32_16x16x32_f16 a[140:143], a[64:67], a[104:107], a[140:143]
	v_mfma_f32_16x16x32_f16 v[248:251], a[76:79], a[124:127], v[248:251]
	v_mfma_f32_16x16x32_f16 v[244:247], a[80:83], a[124:127], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[84:87], a[124:127], v[240:243]
	v_mfma_f32_16x16x32_f16 v[236:239], a[88:91], a[124:127], v[236:239]
	v_mfma_f32_16x16x32_f16 v[232:235], a[92:95], a[124:127], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[64:67], a[124:127], v[228:231]
	v_mfma_f32_16x16x32_f16 v[224:227], a[68:71], a[124:127], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[72:75], a[120:123], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[76:79], a[120:123], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[80:83], a[120:123], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[84:87], a[120:123], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[88:91], a[120:123], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[92:95], a[120:123], v[200:203]
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[120:123], v[196:199]
	v_mfma_f32_16x16x32_f16 v[192:195], a[68:71], a[120:123], v[192:195]
	v_mfma_f32_16x16x32_f16 v[184:187], a[72:75], a[116:119], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[76:79], a[116:119], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[80:83], a[116:119], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[84:87], a[116:119], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[88:91], a[116:119], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[92:95], a[116:119], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[64:67], a[116:119], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[68:71], a[116:119], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[72:75], a[112:115], v[152:155]
	v_mfma_f32_16x16x32_f16 v[148:151], a[76:79], a[112:115], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[80:83], a[112:115], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[84:87], a[112:115], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[88:91], a[112:115], v[136:139]
	v_mfma_f32_16x16x32_f16 v[128:131], a[92:95], a[112:115], v[128:131]
	v_mfma_f32_16x16x32_f16 v[124:127], a[64:67], a[112:115], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[68:71], a[112:115], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[72:75], a[100:103], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[76:79], a[100:103], v[112:115]
	v_mfma_f32_16x16x32_f16 v[108:111], a[80:83], a[100:103], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[84:87], a[100:103], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[88:91], a[100:103], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[92:95], a[100:103], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[64:67], a[100:103], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[68:71], a[100:103], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[72:75], a[60:63], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[76:79], a[60:63], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[80:83], a[60:63], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[84:87], a[60:63], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[88:91], a[60:63], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[92:95], a[60:63], v[64:67]
	v_mfma_f32_16x16x32_f16 v[60:63], a[64:67], a[60:63], v[60:63]
	v_mfma_f32_16x16x32_f16 a[60:63], a[64:67], a[52:55], a[186:189]
	v_mfma_f32_16x16x32_f16 v[48:51], a[68:71], a[52:55], v[48:51]
	v_mfma_f32_16x16x32_f16 v[52:55], a[4:7], a[0:3], v[0:3]
	s_nop 2
	v_accvgpr_read_b32 v0, a153
	v_add_u32_e32 v0, s0, v0
	v_mfma_f32_16x16x32_f16 a[190:193], a[16:19], a[0:3], a[190:193]
	v_mfma_f32_16x16x32_f16 a[182:185], a[40:43], a[8:11], a[182:185]
	v_mfma_f32_16x16x32_f16 a[178:181], a[36:39], a[8:11], a[178:181]
	v_mfma_f32_16x16x32_f16 a[174:177], a[32:35], a[8:11], a[174:177]
	v_mfma_f32_16x16x32_f16 a[170:173], a[28:31], a[8:11], a[170:173]
	v_mfma_f32_16x16x32_f16 a[166:169], a[24:27], a[8:11], a[166:169]
	v_mfma_f32_16x16x32_f16 a[140:143], a[20:23], a[0:3], a[140:143]
	v_mfma_f32_16x16x32_f16 v[28:31], a[40:43], a[0:3], v[28:31]
	v_mfma_f32_16x16x32_f16 v[32:35], a[36:39], a[0:3], v[32:35]
	v_mfma_f32_16x16x32_f16 v[36:39], a[32:35], a[0:3], v[36:39]
	v_mfma_f32_16x16x32_f16 v[40:43], a[28:31], a[0:3], v[40:43]
	v_mfma_f32_16x16x32_f16 v[44:47], a[24:27], a[0:3], v[44:47]
	v_mfma_f32_16x16x32_f16 v[248:251], a[24:27], a[108:111], v[248:251]
	v_mfma_f32_16x16x32_f16 v[244:247], a[28:31], a[108:111], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[32:35], a[108:111], v[240:243]
	v_mfma_f32_16x16x32_f16 v[236:239], a[36:39], a[108:111], v[236:239]
	v_mfma_f32_16x16x32_f16 v[232:235], a[40:43], a[108:111], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[20:23], a[108:111], v[228:231]
	v_mfma_f32_16x16x32_f16 v[224:227], a[4:7], a[108:111], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[16:19], a[96:99], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[24:27], a[96:99], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[28:31], a[96:99], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[32:35], a[96:99], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[36:39], a[96:99], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[40:43], a[96:99], v[200:203]
	v_mfma_f32_16x16x32_f16 v[196:199], a[20:23], a[96:99], v[196:199]
	v_mfma_f32_16x16x32_f16 v[192:195], a[4:7], a[96:99], v[192:195]
	v_mfma_f32_16x16x32_f16 v[184:187], a[16:19], a[56:59], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[24:27], a[56:59], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[28:31], a[56:59], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[32:35], a[56:59], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[36:39], a[56:59], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[40:43], a[56:59], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[20:23], a[56:59], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[4:7], a[56:59], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[16:19], a[48:51], v[152:155]
	v_mfma_f32_16x16x32_f16 v[148:151], a[24:27], a[48:51], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[28:31], a[48:51], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[32:35], a[48:51], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[36:39], a[48:51], v[136:139]
	v_mfma_f32_16x16x32_f16 v[128:131], a[40:43], a[48:51], v[128:131]
	v_mfma_f32_16x16x32_f16 v[124:127], a[20:23], a[48:51], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[4:7], a[48:51], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[16:19], a[44:47], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[24:27], a[44:47], v[112:115]
	v_mfma_f32_16x16x32_f16 v[108:111], a[28:31], a[44:47], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[32:35], a[44:47], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[36:39], a[44:47], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[40:43], a[44:47], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[20:23], a[44:47], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[4:7], a[44:47], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[16:19], a[12:15], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[24:27], a[12:15], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[28:31], a[12:15], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[32:35], a[12:15], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[36:39], a[12:15], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[40:43], a[12:15], v[64:67]
	v_mfma_f32_16x16x32_f16 v[60:63], a[20:23], a[12:15], v[60:63]
	v_mfma_f32_16x16x32_f16 v[56:59], a[4:7], a[12:15], v[56:59]
	v_mfma_f32_16x16x32_f16 v[188:191], a[16:19], a[8:11], v[188:191]
	v_mfma_f32_16x16x32_f16 a[158:161], a[20:23], a[8:11], a[60:63]
	v_mfma_f32_16x16x32_f16 v[24:27], a[4:7], a[8:11], v[48:51]
	ds_read_b128 a[124:127], v0
	ds_read_b128 a[108:111], v0 offset:64
	ds_read_b128 a[120:123], v0 offset:256
	ds_read_b128 a[96:99], v0 offset:320
	ds_read_b128 a[116:119], v0 offset:512
	ds_read_b128 a[56:59], v0 offset:576
	ds_read_b128 a[112:115], v0 offset:768
	ds_read_b128 a[48:51], v0 offset:832
	ds_read_b128 a[100:103], v0 offset:16896
	ds_read_b128 a[44:47], v0 offset:16960
	ds_read_b128 a[60:63], v0 offset:17152
	ds_read_b128 a[12:15], v0 offset:17216
	ds_read_b128 a[52:55], v0 offset:17408
	ds_read_b128 a[8:11], v0 offset:17472
	ds_read_b128 a[104:107], v0 offset:17664
	ds_read_b128 a[0:3], v0 offset:17728
	v_accvgpr_read_b32 v0, a152
	v_add_u32_e32 v0, s0, v0
	ds_read_b128 a[72:75], v0
	ds_read_b128 a[16:19], v0 offset:64
	ds_read_b128 a[76:79], v0 offset:256
	ds_read_b128 a[24:27], v0 offset:320
	ds_read_b128 a[80:83], v0 offset:512
	ds_read_b128 a[28:31], v0 offset:576
	ds_read_b128 a[84:87], v0 offset:768
	ds_read_b128 a[32:35], v0 offset:832
	ds_read_b128 a[88:91], v0 offset:16896
	ds_read_b128 a[36:39], v0 offset:16960
	ds_read_b128 a[92:95], v0 offset:17152
	ds_read_b128 a[40:43], v0 offset:17216
	ds_read_b128 a[64:67], v0 offset:17408
	ds_read_b128 a[20:23], v0 offset:17472
	ds_read_b128 a[68:71], v0 offset:17664
	ds_read_b128 a[4:7], v0 offset:17728
	s_cbranch_scc1 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a140, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a190, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v192, 0
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
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
.LBB0_4:                                ; %Flow390
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[104:107], v[44:47]
	s_add_i32 s1, s4, -1
	s_lshr_b32 s2, s1, 31
	s_add_i32 s2, s1, s2
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[44:47], a[24:27], a[0:3], v[0:3]
	s_and_b32 s2, s2, -2
	s_sub_i32 s1, s1, s2
	s_mul_i32 s1, s1, 0x8400
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[104:107], v[40:43]
	s_add_i32 s1, s1, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[40:43], a[28:31], a[0:3], v[0:3]
	s_barrier
	s_lshr_b32 s0, s20, 6
	s_lshl_b32 s0, s0, 3
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[104:107], v[36:39]
	v_mfma_f32_16x16x32_f16 v[36:39], a[32:35], a[0:3], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[104:107], v[32:35]
	v_mfma_f32_16x16x32_f16 v[132:135], a[72:75], a[124:127], v[132:135]
	v_mfma_f32_16x16x32_f16 v[248:251], a[76:79], a[124:127], v[248:251]
	v_mfma_f32_16x16x32_f16 v[244:247], a[80:83], a[124:127], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[84:87], a[124:127], v[240:243]
	v_mfma_f32_16x16x32_f16 v[236:239], a[88:91], a[124:127], v[236:239]
	v_mfma_f32_16x16x32_f16 v[232:235], a[92:95], a[124:127], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[64:67], a[124:127], v[228:231]
	v_mfma_f32_16x16x32_f16 v[224:227], a[68:71], a[124:127], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[72:75], a[120:123], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[76:79], a[120:123], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[80:83], a[120:123], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[84:87], a[120:123], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[88:91], a[120:123], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[92:95], a[120:123], v[200:203]
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[120:123], v[196:199]
	v_mfma_f32_16x16x32_f16 v[192:195], a[68:71], a[120:123], v[192:195]
	v_mfma_f32_16x16x32_f16 v[184:187], a[72:75], a[116:119], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[76:79], a[116:119], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[80:83], a[116:119], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[84:87], a[116:119], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[88:91], a[116:119], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[92:95], a[116:119], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[64:67], a[116:119], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[68:71], a[116:119], v[156:159]
	v_mfma_f32_16x16x32_f16 v[188:191], a[72:75], a[52:55], v[188:191]
	v_mfma_f32_16x16x32_f16 a[116:119], a[76:79], a[52:55], a[166:169]
	v_mfma_f32_16x16x32_f16 a[120:123], a[80:83], a[52:55], a[170:173]
	v_mfma_f32_16x16x32_f16 a[124:127], a[84:87], a[52:55], a[174:177]
	v_mfma_f32_16x16x32_f16 a[132:135], a[88:91], a[52:55], a[178:181]
	v_mfma_f32_16x16x32_f16 a[136:139], a[92:95], a[52:55], a[182:185]
	v_mfma_f32_16x16x32_f16 a[144:147], a[64:67], a[52:55], a[158:161]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[52:55], v[24:27]
	v_mfma_f32_16x16x32_f16 v[32:35], a[36:39], a[0:3], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[104:107], v[28:31]
	v_mfma_f32_16x16x32_f16 v[188:191], a[16:19], a[8:11], v[188:191]
	v_mfma_f32_16x16x32_f16 a[116:119], a[24:27], a[8:11], a[116:119]
	v_mfma_f32_16x16x32_f16 a[120:123], a[28:31], a[8:11], a[120:123]
	v_mfma_f32_16x16x32_f16 a[124:127], a[32:35], a[8:11], a[124:127]
	v_mfma_f32_16x16x32_f16 a[132:135], a[36:39], a[8:11], a[132:135]
	v_mfma_f32_16x16x32_f16 a[136:139], a[40:43], a[8:11], a[136:139]
	v_mfma_f32_16x16x32_f16 a[144:147], a[20:23], a[8:11], a[144:147]
	v_mfma_f32_16x16x32_f16 v[48:51], a[4:7], a[8:11], v[4:7]
	v_accvgpr_mov_b32 a8, a140
	v_accvgpr_mov_b32 a9, a141
	v_accvgpr_mov_b32 a10, a142
	v_mfma_f32_16x16x32_f16 v[28:31], a[40:43], a[0:3], v[0:3]
	v_accvgpr_mov_b32 a11, a143
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[104:107], v[52:55]
	v_mfma_f32_16x16x32_f16 v[120:123], a[68:71], a[112:115], v[120:123]
	s_nop 1
	v_accvgpr_read_b32 v52, a116
	v_accvgpr_read_b32 v53, a117
	v_accvgpr_read_b32 v54, a118
	v_mfma_f32_16x16x32_f16 v[88:91], a[68:71], a[100:103], v[88:91]
	v_accvgpr_read_b32 v55, a119
	v_mfma_f32_16x16x32_f16 v[56:59], a[68:71], a[60:63], v[56:59]
	v_mfma_f32_16x16x32_f16 a[148:151], a[72:75], a[104:107], a[190:193]
	v_mfma_f32_16x16x32_f16 a[8:11], a[64:67], a[104:107], a[8:11]
	v_mfma_f32_16x16x32_f16 v[252:255], a[4:7], a[0:3], v[0:3]
	s_nop 2
	v_accvgpr_read_b32 v0, a130
	v_mfma_f32_16x16x32_f16 v[248:251], a[24:27], a[108:111], v[248:251]
	v_add_u32_e32 v0, s1, v0
	v_mfma_f32_16x16x32_f16 v[152:155], a[72:75], a[112:115], v[152:155]
	v_mfma_f32_16x16x32_f16 v[148:151], a[76:79], a[112:115], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[80:83], a[112:115], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[84:87], a[112:115], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[88:91], a[112:115], v[136:139]
	v_mfma_f32_16x16x32_f16 v[128:131], a[92:95], a[112:115], v[128:131]
	v_mfma_f32_16x16x32_f16 v[124:127], a[64:67], a[112:115], v[124:127]
	v_mfma_f32_16x16x32_f16 v[116:119], a[72:75], a[100:103], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[76:79], a[100:103], v[112:115]
	v_mfma_f32_16x16x32_f16 v[108:111], a[80:83], a[100:103], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[84:87], a[100:103], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[88:91], a[100:103], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[92:95], a[100:103], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[64:67], a[100:103], v[92:95]
	v_mfma_f32_16x16x32_f16 v[84:87], a[72:75], a[60:63], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[76:79], a[60:63], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[80:83], a[60:63], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[84:87], a[60:63], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[88:91], a[60:63], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[92:95], a[60:63], v[64:67]
	v_mfma_f32_16x16x32_f16 v[60:63], a[64:67], a[60:63], v[60:63]
	v_mfma_f32_16x16x32_f16 v[132:135], a[16:19], a[108:111], v[132:135]
	v_mfma_f32_16x16x32_f16 v[244:247], a[28:31], a[108:111], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[32:35], a[108:111], v[240:243]
	v_mfma_f32_16x16x32_f16 v[236:239], a[36:39], a[108:111], v[236:239]
	v_mfma_f32_16x16x32_f16 v[232:235], a[40:43], a[108:111], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[20:23], a[108:111], v[228:231]
	v_mfma_f32_16x16x32_f16 v[224:227], a[4:7], a[108:111], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[16:19], a[96:99], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[24:27], a[96:99], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[28:31], a[96:99], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[32:35], a[96:99], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[36:39], a[96:99], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[40:43], a[96:99], v[200:203]
	v_mfma_f32_16x16x32_f16 v[196:199], a[20:23], a[96:99], v[196:199]
	v_mfma_f32_16x16x32_f16 v[192:195], a[4:7], a[96:99], v[192:195]
	v_mfma_f32_16x16x32_f16 v[156:159], a[4:7], a[56:59], v[156:159]
	v_mfma_f32_16x16x32_f16 v[120:123], a[4:7], a[48:51], v[120:123]
	v_mfma_f32_16x16x32_f16 v[88:91], a[4:7], a[44:47], v[88:91]
	v_mfma_f32_16x16x32_f16 v[56:59], a[4:7], a[12:15], v[56:59]
	v_mfma_f32_16x16x32_f16 a[148:151], a[16:19], a[0:3], a[148:151]
	v_mfma_f32_16x16x32_f16 a[140:143], a[20:23], a[0:3], a[8:11]
	ds_read_b128 v[8:11], v0
	ds_read_b128 v[12:15], v0 offset:64
	ds_read_b128 a[108:111], v0 offset:256
	ds_read_b128 a[112:115], v0 offset:320
	ds_read_b128 a[100:103], v0 offset:512
	ds_read_b128 a[104:107], v0 offset:576
	ds_read_b128 a[92:95], v0 offset:768
	ds_read_b128 a[96:99], v0 offset:832
	ds_read_b128 a[84:87], v0 offset:16896
	ds_read_b128 a[88:91], v0 offset:16960
	ds_read_b128 a[76:79], v0 offset:17152
	ds_read_b128 a[80:83], v0 offset:17216
	ds_read_b128 v[24:27], v0 offset:17408
	ds_read_b128 a[8:11], v0 offset:17472
	ds_read_b128 a[0:3], v0 offset:17664
	ds_read_b128 a[4:7], v0 offset:17728
	v_accvgpr_read_b32 v0, a131
	v_add_u32_e32 v0, s1, v0
	v_add_u32_e32 v0, 0x107e0, v0
	v_mfma_f32_16x16x32_f16 v[184:187], a[16:19], a[56:59], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[24:27], a[56:59], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[28:31], a[56:59], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[32:35], a[56:59], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[36:39], a[56:59], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[40:43], a[56:59], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[20:23], a[56:59], v[160:163]
	v_mfma_f32_16x16x32_f16 v[152:155], a[16:19], a[48:51], v[152:155]
	v_mfma_f32_16x16x32_f16 v[148:151], a[24:27], a[48:51], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[28:31], a[48:51], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[32:35], a[48:51], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[36:39], a[48:51], v[136:139]
	v_mfma_f32_16x16x32_f16 v[128:131], a[40:43], a[48:51], v[128:131]
	v_mfma_f32_16x16x32_f16 v[124:127], a[20:23], a[48:51], v[124:127]
	v_mfma_f32_16x16x32_f16 v[116:119], a[16:19], a[44:47], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[24:27], a[44:47], v[112:115]
	v_mfma_f32_16x16x32_f16 v[108:111], a[28:31], a[44:47], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[32:35], a[44:47], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[36:39], a[44:47], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[40:43], a[44:47], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[20:23], a[44:47], v[92:95]
	v_mfma_f32_16x16x32_f16 v[84:87], a[16:19], a[12:15], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[24:27], a[12:15], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[28:31], a[12:15], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[32:35], a[12:15], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[36:39], a[12:15], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[40:43], a[12:15], v[64:67]
	v_mfma_f32_16x16x32_f16 v[60:63], a[20:23], a[12:15], v[60:63]
	ds_read_b128 a[12:15], v0
	ds_read_b128 a[16:19], v0 offset:64
	ds_read_b128 a[20:23], v0 offset:256
	ds_read_b128 a[24:27], v0 offset:320
	ds_read_b128 a[28:31], v0 offset:512
	ds_read_b128 a[32:35], v0 offset:576
	ds_read_b128 a[36:39], v0 offset:768
	ds_read_b128 a[40:43], v0 offset:832
	ds_read_b128 a[44:47], v0 offset:16896
	ds_read_b128 a[48:51], v0 offset:16960
	ds_read_b128 a[52:55], v0 offset:17152
	ds_read_b128 a[56:59], v0 offset:17216
	ds_read_b128 a[60:63], v0 offset:17408
	ds_read_b128 a[64:67], v0 offset:17472
	ds_read_b128 a[68:71], v0 offset:17664
	ds_read_b128 a[72:75], v0 offset:17728
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[4:7], a[20:23], v[8:11], v[248:251]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[248:251], a[24:27], v[12:15], v[4:7]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[4:7], a[28:31], v[8:11], v[244:247]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[244:247], a[32:35], v[12:15], v[4:7]
	s_nop 3
	v_cvt_pk_f16_f32 v248, v248, v249
	v_cvt_pk_f16_f32 v249, v250, v251
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[4:7], a[36:39], v[8:11], v[240:243]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[240:243], a[40:43], v[12:15], v[4:7]
	v_cvt_pk_f16_f32 v244, v244, v245
	v_cvt_pk_f16_f32 v245, v246, v247
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[4:7], a[44:47], v[8:11], v[236:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[236:239], a[48:51], v[12:15], v[4:7]
	s_nop 1
	v_cvt_pk_f16_f32 v240, v240, v241
	v_cvt_pk_f16_f32 v241, v242, v243
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[4:7], a[52:55], v[8:11], v[232:235]
	v_mfma_f32_16x16x32_f16 v[116:119], a[12:15], a[84:87], v[116:119]
	s_nop 0
	v_cvt_pk_f16_f32 v236, v236, v237
	v_cvt_pk_f16_f32 v237, v238, v239
	v_mfma_f32_16x16x32_f16 v[84:87], a[12:15], a[76:79], v[84:87]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[56:59], a[68:71], a[76:79], v[56:59]
	v_mfma_f32_16x16x32_f16 v[0:3], a[12:15], v[8:11], v[132:135]
	v_mfma_f32_16x16x32_f16 v[232:235], a[56:59], v[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], v[8:11], v[228:231]
	v_mfma_f32_16x16x32_f16 v[8:11], a[68:71], v[8:11], v[224:227]
	s_nop 5
	v_cvt_pk_f16_f32 v232, v232, v233
	v_cvt_pk_f16_f32 v233, v234, v235
	v_mfma_f32_16x16x32_f16 v[16:19], a[20:23], a[108:111], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[16:19], a[88:91], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[20:23], a[84:87], v[112:115]
	v_mfma_f32_16x16x32_f16 v[88:91], a[68:71], a[84:87], v[88:91]
	v_mfma_f32_16x16x32_f16 v[116:119], a[16:19], a[80:83], v[84:87]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[84:87], a[72:75], a[80:83], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[12:15], v[24:27], v[188:191]
	v_mfma_f32_16x16x32_f16 v[0:3], a[16:19], v[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], v[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[8:11], a[72:75], v[12:15], v[8:11]
	s_nop 5
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	v_mfma_f32_16x16x32_f16 v[12:15], a[12:15], a[108:111], v[220:223]
	v_cvt_pk_f16_f32 v235, v6, v7
	v_cvt_pk_f16_f32 v234, v4, v5
	v_mfma_f32_16x16x32_f16 v[220:223], a[24:27], a[88:91], v[112:115]
	v_cvt_pk_f16_f32 v238, v8, v9
	v_cvt_pk_f16_f32 v239, v10, v11
	v_mfma_f32_16x16x32_f16 v[108:111], a[28:31], a[84:87], v[108:111]
	v_mfma_f32_16x16x32_f16 v[92:95], a[60:63], a[84:87], v[92:95]
	v_mfma_f32_16x16x32_f16 v[112:115], a[72:75], a[88:91], v[88:91]
	v_mfma_f32_16x16x32_f16 v[88:91], a[16:19], a[8:11], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[20:23], v[24:27], v[52:55]
	s_nop 5
	v_cvt_pk_f16_f32 v112, v112, v113
	v_cvt_pk_f16_f32 v113, v114, v115
	v_accvgpr_read_b32 v52, a120
	v_accvgpr_read_b32 v53, a121
	v_accvgpr_read_b32 v54, a122
	v_accvgpr_read_b32 v55, a123
	v_mfma_f32_16x16x32_f16 v[224:227], a[32:35], a[88:91], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[36:39], a[84:87], v[104:107]
	v_mfma_f32_16x16x32_f16 v[96:99], a[52:55], a[84:87], v[96:99]
	v_mfma_f32_16x16x32_f16 v[108:111], a[64:67], a[88:91], v[92:95]
	v_mfma_f32_16x16x32_f16 v[92:95], a[24:27], a[8:11], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[28:31], v[24:27], v[52:55]
	s_nop 2
	v_accvgpr_read_b32 v52, a124
	v_accvgpr_read_b32 v53, a125
	v_accvgpr_read_b32 v54, a126
	v_accvgpr_read_b32 v55, a127
	v_mfma_f32_16x16x32_f16 v[124:127], a[60:63], a[92:95], v[124:127]
	v_mfma_f32_16x16x32_f16 v[228:231], a[40:43], a[88:91], v[104:107]
	v_mfma_f32_16x16x32_f16 v[104:107], a[56:59], a[88:91], v[96:99]
	v_mfma_f32_16x16x32_f16 v[76:79], a[28:31], a[76:79], v[76:79]
	v_mfma_f32_16x16x32_f16 v[64:67], a[52:55], a[76:79], v[64:67]
	v_mfma_f32_16x16x32_f16 v[96:99], a[32:35], a[8:11], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[36:39], v[24:27], v[52:55]
	v_mfma_f32_16x16x32_f16 v[120:123], a[68:71], a[92:95], v[120:123]
	s_nop 1
	v_accvgpr_read_b32 v52, a132
	v_accvgpr_read_b32 v53, a133
	v_accvgpr_read_b32 v54, a134
	v_mfma_f32_16x16x32_f16 v[80:83], a[20:23], a[76:79], v[80:83]
	v_accvgpr_read_b32 v55, a135
	v_mfma_f32_16x16x32_f16 v[60:63], a[60:63], a[76:79], v[60:63]
	v_mfma_f32_16x16x32_f16 v[132:135], a[36:39], a[108:111], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[64:67], a[96:99], v[124:127]
	v_mfma_f32_16x16x32_f16 v[124:127], a[32:35], a[80:83], v[76:79]
	v_mfma_f32_16x16x32_f16 v[76:79], a[56:59], a[80:83], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[40:43], a[8:11], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[44:47], v[24:27], v[52:55]
	s_nop 2
	v_accvgpr_read_b32 v52, a136
	v_accvgpr_read_b32 v53, a137
	v_accvgpr_read_b32 v54, a138
	v_accvgpr_read_b32 v55, a139
	v_mfma_f32_16x16x32_f16 v[20:23], a[28:31], a[108:111], v[212:215]
	v_cvt_pk_f16_f32 v64, v64, v65
	v_cvt_pk_f16_f32 v65, v66, v67
	v_mfma_f32_16x16x32_f16 v[212:215], a[72:75], a[96:99], v[120:123]
	v_mfma_f32_16x16x32_f16 v[120:123], a[24:27], a[80:83], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[64:67], a[80:83], v[60:63]
	v_mfma_f32_16x16x32_f16 v[60:63], a[52:55], v[24:27], v[52:55]
	s_nop 2
	v_accvgpr_read_b32 v52, a144
	v_accvgpr_read_b32 v53, a145
	v_accvgpr_read_b32 v54, a146
	v_accvgpr_read_b32 v55, a147
	v_mfma_f32_16x16x32_f16 v[12:15], a[16:19], a[112:115], v[12:15]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[52:55], a[60:63], v[24:27], v[52:55]
	v_mfma_f32_16x16x32_f16 v[24:27], a[68:71], v[24:27], v[48:51]
	s_nop 4
	v_cvt_pk_f16_f32 v243, v14, v15
	v_cvt_pk_f16_f32 v242, v12, v13
	v_mfma_f32_16x16x32_f16 v[48:51], a[72:75], a[8:11], v[24:27]
	v_mfma_f32_16x16x32_f16 v[36:39], a[36:39], a[0:3], v[36:39]
	s_nop 1
	v_accvgpr_read_b32 v24, a148
	v_accvgpr_read_b32 v25, a149
	v_accvgpr_read_b32 v26, a150
	v_accvgpr_read_b32 v27, a151
	v_mfma_f32_16x16x32_f16 v[188:191], a[40:43], a[4:7], v[36:39]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[24:27], a[12:15], a[0:3], v[24:27]
	s_nop 0
	v_accvgpr_read_b32 v36, a140
	v_accvgpr_read_b32 v37, a141
	v_accvgpr_read_b32 v38, a142
	v_mfma_f32_16x16x32_f16 v[24:27], a[16:19], a[4:7], v[24:27]
	v_accvgpr_read_b32 v39, a143
	v_cvt_pk_f16_f32 v10, v188, v189
	v_cvt_pk_f16_f32 v11, v190, v191
	v_mfma_f32_16x16x32_f16 v[32:35], a[44:47], a[0:3], v[32:35]
	v_mfma_f32_16x16x32_f16 v[36:39], a[60:63], a[0:3], v[36:39]
	s_nop 2
	v_cvt_pk_f16_f32 v15, v26, v27
	v_accvgpr_read_b32 v26, a129
	v_and_or_b32 v26, s0, 16, v26
	s_mul_i32 s0, s15, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	v_mfma_f32_16x16x32_f16 v[32:35], a[48:51], a[4:7], v[32:35]
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[0:1], s[14:15], 1
	v_mfma_f32_16x16x32_f16 v[40:43], a[28:31], a[0:3], v[40:43]
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	v_mul_lo_u32 v26, v26, s13
	v_mfma_f32_16x16x32_f16 v[36:39], a[64:67], a[4:7], v[36:39]
	s_lshl_b32 s2, s13, 5
	v_cvt_pk_f16_f32 v7, v34, v35
	v_add_u32_e32 v35, s2, v26
	v_mfma_f32_16x16x32_f16 v[28:31], a[52:55], a[0:3], v[28:31]
	v_cvt_pk_f16_f32 v14, v24, v25
	v_accvgpr_read_b32 v27, a128
	v_lshrrev_b32_e32 v27, 2, v27
	v_mfma_f32_16x16x32_f16 v[40:43], a[32:35], a[4:7], v[40:43]
	v_cvt_pk_f16_f32 v24, v36, v37
	v_add_u32_e32 v36, s2, v35
	v_add_u32_e32 v37, s2, v36
	v_mfma_f32_16x16x32_f16 v[28:31], a[56:59], a[4:7], v[28:31]
	v_cvt_pk_f16_f32 v25, v38, v39
	v_add_u32_e32 v38, s2, v37
	v_add_u32_e32 v39, s2, v38
	v_and_b32_e32 v27, 28, v27
	v_cvt_pk_f16_f32 v8, v40, v41
	v_add_u32_e32 v40, s2, v39
	v_cvt_pk_f16_f32 v9, v42, v43
	v_add_u32_e32 v41, s2, v40
	v_cvt_pk_f16_f32 v4, v28, v29
	v_or_b32_e32 v28, 32, v27
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_add_lshl_u32 v42, v27, v26, 1
	v_or_b32_e32 v29, 64, v27
	buffer_store_dwordx2 v[0:1], v42, s[0:3], 0 offen
	v_add_lshl_u32 v0, v28, v26, 1
	v_cvt_pk_f16_f32 v5, v30, v31
	v_or_b32_e32 v30, 0x60, v27
	buffer_store_dwordx2 v[248:249], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v29, v26, 1
	v_or_b32_e32 v31, 0x80, v27
	buffer_store_dwordx2 v[244:245], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v30, v26, 1
	v_mfma_f32_16x16x32_f16 v[204:207], a[44:47], a[108:111], v[204:207]
	v_cvt_pk_f16_f32 v6, v32, v33
	v_or_b32_e32 v32, 0xa0, v27
	buffer_store_dwordx2 v[240:241], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v31, v26, 1
	v_mfma_f32_16x16x32_f16 v[16:19], a[24:27], a[112:115], v[16:19]
	v_or_b32_e32 v33, 0xc0, v27
	buffer_store_dwordx2 v[236:237], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v32, v26, 1
	v_mfma_f32_16x16x32_f16 v[200:203], a[52:55], a[108:111], v[200:203]
	v_or_b32_e32 v34, 0xe0, v27
	buffer_store_dwordx2 v[232:233], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v33, v26, 1
	v_mfma_f32_16x16x32_f16 v[20:23], a[32:35], a[112:115], v[20:23]
	buffer_store_dwordx2 v[234:235], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v34, v26, 1
	buffer_store_dwordx2 v[238:239], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[196:199], a[60:63], a[108:111], v[196:199]
	v_add_lshl_u32 v0, v35, v27, 1
	v_cvt_pk_f16_f32 v246, v16, v17
	v_cvt_pk_f16_f32 v247, v18, v19
	v_mfma_f32_16x16x32_f16 v[132:135], a[40:43], a[112:115], v[132:135]
	buffer_store_dwordx2 v[242:243], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v28, v35, 1
	v_cvt_pk_f16_f32 v250, v20, v21
	v_mfma_f32_16x16x32_f16 v[192:195], a[68:71], a[108:111], v[192:195]
	v_cvt_pk_f16_f32 v251, v22, v23
	buffer_store_dwordx2 v[246:247], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v29, v35, 1
	v_mfma_f32_16x16x32_f16 v[204:207], a[48:51], a[112:115], v[204:207]
	v_cvt_pk_f16_f32 v132, v132, v133
	v_cvt_pk_f16_f32 v133, v134, v135
	buffer_store_dwordx2 v[250:251], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[184:187], a[12:15], a[100:103], v[184:187]
	v_add_lshl_u32 v0, v30, v35, 1
	buffer_store_dwordx2 v[132:133], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v31, v35, 1
	v_mfma_f32_16x16x32_f16 v[200:203], a[56:59], a[112:115], v[200:203]
	v_cvt_pk_f16_f32 v134, v204, v205
	v_cvt_pk_f16_f32 v135, v206, v207
	buffer_store_dwordx2 v[134:135], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[180:183], a[20:23], a[100:103], v[180:183]
	v_add_lshl_u32 v0, v32, v35, 1
	v_cvt_pk_f16_f32 v16, v48, v49
	v_cvt_pk_f16_f32 v17, v50, v51
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[112:115], v[196:199]
	v_cvt_pk_f16_f32 v200, v200, v201
	v_cvt_pk_f16_f32 v201, v202, v203
	buffer_store_dwordx2 v[200:201], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[176:179], a[28:31], a[100:103], v[176:179]
	v_add_lshl_u32 v0, v33, v35, 1
	v_mfma_f32_16x16x32_f16 v[192:195], a[72:75], a[112:115], v[192:195]
	s_nop 1
	v_cvt_pk_f16_f32 v196, v196, v197
	v_cvt_pk_f16_f32 v197, v198, v199
	buffer_store_dwordx2 v[196:197], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[172:175], a[36:39], a[100:103], v[172:175]
	v_add_lshl_u32 v0, v34, v35, 1
	v_mfma_f32_16x16x32_f16 v[184:187], a[16:19], a[104:107], v[184:187]
	v_cvt_pk_f16_f32 v192, v192, v193
	v_cvt_pk_f16_f32 v193, v194, v195
	buffer_store_dwordx2 v[192:193], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[168:171], a[44:47], a[100:103], v[168:171]
	v_add_lshl_u32 v0, v36, v27, 1
	v_mfma_f32_16x16x32_f16 v[180:183], a[24:27], a[104:107], v[180:183]
	s_nop 1
	v_cvt_pk_f16_f32 v184, v184, v185
	v_cvt_pk_f16_f32 v185, v186, v187
	buffer_store_dwordx2 v[184:185], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[164:167], a[52:55], a[100:103], v[164:167]
	v_add_lshl_u32 v0, v28, v36, 1
	v_mfma_f32_16x16x32_f16 v[176:179], a[32:35], a[104:107], v[176:179]
	v_cvt_pk_f16_f32 v180, v180, v181
	v_cvt_pk_f16_f32 v181, v182, v183
	buffer_store_dwordx2 v[180:181], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[160:163], a[60:63], a[100:103], v[160:163]
	v_add_lshl_u32 v0, v29, v36, 1
	v_mfma_f32_16x16x32_f16 v[172:175], a[40:43], a[104:107], v[172:175]
	s_nop 1
	v_cvt_pk_f16_f32 v176, v176, v177
	v_cvt_pk_f16_f32 v177, v178, v179
	buffer_store_dwordx2 v[176:177], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[156:159], a[68:71], a[100:103], v[156:159]
	v_add_lshl_u32 v0, v30, v36, 1
	v_mfma_f32_16x16x32_f16 v[168:171], a[48:51], a[104:107], v[168:171]
	v_cvt_pk_f16_f32 v172, v172, v173
	v_cvt_pk_f16_f32 v173, v174, v175
	buffer_store_dwordx2 v[172:173], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[152:155], a[12:15], a[92:95], v[152:155]
	v_add_lshl_u32 v0, v31, v36, 1
	v_mfma_f32_16x16x32_f16 v[164:167], a[56:59], a[104:107], v[164:167]
	s_nop 1
	v_cvt_pk_f16_f32 v168, v168, v169
	v_cvt_pk_f16_f32 v169, v170, v171
	buffer_store_dwordx2 v[168:169], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[148:151], a[20:23], a[92:95], v[148:151]
	v_add_lshl_u32 v0, v32, v36, 1
	v_mfma_f32_16x16x32_f16 v[160:163], a[64:67], a[104:107], v[160:163]
	v_cvt_pk_f16_f32 v164, v164, v165
	v_cvt_pk_f16_f32 v165, v166, v167
	buffer_store_dwordx2 v[164:165], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[144:147], a[28:31], a[92:95], v[144:147]
	v_add_lshl_u32 v0, v33, v36, 1
	v_mfma_f32_16x16x32_f16 v[156:159], a[72:75], a[104:107], v[156:159]
	s_nop 1
	v_cvt_pk_f16_f32 v160, v160, v161
	v_cvt_pk_f16_f32 v161, v162, v163
	buffer_store_dwordx2 v[160:161], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[140:143], a[36:39], a[92:95], v[140:143]
	v_add_lshl_u32 v0, v34, v36, 1
	v_cvt_pk_f16_f32 v162, v228, v229
	v_cvt_pk_f16_f32 v163, v230, v231
	v_mfma_f32_16x16x32_f16 v[152:155], a[16:19], a[96:99], v[152:155]
	v_cvt_pk_f16_f32 v156, v156, v157
	v_cvt_pk_f16_f32 v157, v158, v159
	buffer_store_dwordx2 v[156:157], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[136:139], a[44:47], a[92:95], v[136:139]
	v_add_lshl_u32 v0, v37, v27, 1
	v_cvt_pk_f16_f32 v158, v224, v225
	v_cvt_pk_f16_f32 v159, v226, v227
	v_mfma_f32_16x16x32_f16 v[148:151], a[24:27], a[96:99], v[148:151]
	v_cvt_pk_f16_f32 v152, v152, v153
	v_cvt_pk_f16_f32 v153, v154, v155
	buffer_store_dwordx2 v[152:153], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[128:131], a[52:55], a[92:95], v[128:131]
	v_add_lshl_u32 v0, v28, v37, 1
	v_cvt_pk_f16_f32 v154, v220, v221
	v_cvt_pk_f16_f32 v155, v222, v223
	v_mfma_f32_16x16x32_f16 v[144:147], a[32:35], a[96:99], v[144:147]
	v_cvt_pk_f16_f32 v148, v148, v149
	v_cvt_pk_f16_f32 v149, v150, v151
	buffer_store_dwordx2 v[148:149], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[140:143], a[40:43], a[96:99], v[140:143]
	v_add_lshl_u32 v0, v29, v37, 1
	v_cvt_pk_f16_f32 v150, v216, v217
	v_cvt_pk_f16_f32 v151, v218, v219
	v_mfma_f32_16x16x32_f16 v[136:139], a[48:51], a[96:99], v[136:139]
	v_cvt_pk_f16_f32 v144, v144, v145
	v_cvt_pk_f16_f32 v145, v146, v147
	buffer_store_dwordx2 v[144:145], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[128:131], a[56:59], a[96:99], v[128:131]
	v_cvt_pk_f16_f32 v140, v140, v141
	v_cvt_pk_f16_f32 v141, v142, v143
	v_add_lshl_u32 v0, v30, v37, 1
	v_mfma_f32_16x16x32_f16 v[100:103], a[44:47], a[84:87], v[100:103]
	v_cvt_pk_f16_f32 v136, v136, v137
	v_cvt_pk_f16_f32 v137, v138, v139
	buffer_store_dwordx2 v[140:141], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v31, v37, 1
	v_cvt_pk_f16_f32 v138, v128, v129
	v_cvt_pk_f16_f32 v139, v130, v131
	buffer_store_dwordx2 v[136:137], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v32, v37, 1
	v_cvt_pk_f16_f32 v142, v208, v209
	v_cvt_pk_f16_f32 v143, v210, v211
	buffer_store_dwordx2 v[138:139], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v33, v37, 1
	v_cvt_pk_f16_f32 v146, v212, v213
	v_cvt_pk_f16_f32 v147, v214, v215
	buffer_store_dwordx2 v[142:143], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v34, v37, 1
	v_mfma_f32_16x16x32_f16 v[100:103], a[48:51], a[88:91], v[100:103]
	buffer_store_dwordx2 v[146:147], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v38, v27, 1
	buffer_store_dwordx2 v[150:151], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v28, v38, 1
	buffer_store_dwordx2 v[154:155], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v29, v38, 1
	v_mfma_f32_16x16x32_f16 v[72:75], a[36:39], a[76:79], v[72:75]
	buffer_store_dwordx2 v[158:159], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v30, v38, 1
	v_cvt_pk_f16_f32 v166, v100, v101
	v_mfma_f32_16x16x32_f16 v[68:71], a[44:47], a[76:79], v[68:71]
	v_cvt_pk_f16_f32 v167, v102, v103
	buffer_store_dwordx2 v[162:163], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v31, v38, 1
	v_cvt_pk_f16_f32 v130, v104, v105
	v_cvt_pk_f16_f32 v131, v106, v107
	buffer_store_dwordx2 v[166:167], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v32, v38, 1
	v_cvt_pk_f16_f32 v128, v108, v109
	v_cvt_pk_f16_f32 v129, v110, v111
	buffer_store_dwordx2 v[130:131], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v33, v38, 1
	v_mfma_f32_16x16x32_f16 v[72:75], a[40:43], a[80:83], v[72:75]
	buffer_store_dwordx2 v[128:129], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v34, v38, 1
	v_cvt_pk_f16_f32 v110, v116, v117
	v_mfma_f32_16x16x32_f16 v[68:71], a[48:51], a[80:83], v[68:71]
	v_cvt_pk_f16_f32 v111, v118, v119
	buffer_store_dwordx2 v[112:113], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v39, v27, 1
	v_cvt_pk_f16_f32 v108, v120, v121
	v_cvt_pk_f16_f32 v109, v122, v123
	buffer_store_dwordx2 v[110:111], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v28, v39, 1
	v_cvt_pk_f16_f32 v104, v124, v125
	v_cvt_pk_f16_f32 v105, v126, v127
	buffer_store_dwordx2 v[108:109], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v29, v39, 1
	v_cvt_pk_f16_f32 v106, v72, v73
	v_cvt_pk_f16_f32 v107, v74, v75
	buffer_store_dwordx2 v[104:105], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v30, v39, 1
	v_cvt_pk_f16_f32 v102, v68, v69
	v_cvt_pk_f16_f32 v103, v70, v71
	buffer_store_dwordx2 v[106:107], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v31, v39, 1
	v_cvt_pk_f16_f32 v100, v76, v77
	v_cvt_pk_f16_f32 v101, v78, v79
	buffer_store_dwordx2 v[102:103], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v32, v39, 1
	v_cvt_pk_f16_f32 v76, v80, v81
	v_cvt_pk_f16_f32 v77, v82, v83
	buffer_store_dwordx2 v[100:101], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v33, v39, 1
	v_cvt_pk_f16_f32 v74, v84, v85
	v_cvt_pk_f16_f32 v75, v86, v87
	buffer_store_dwordx2 v[76:77], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v34, v39, 1
	v_mfma_f32_16x16x32_f16 v[56:59], a[48:51], a[8:11], v[56:59]
	v_cvt_pk_f16_f32 v72, v88, v89
	v_cvt_pk_f16_f32 v73, v90, v91
	buffer_store_dwordx2 v[74:75], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v40, v27, 1
	v_mfma_f32_16x16x32_f16 v[60:63], a[56:59], a[8:11], v[60:63]
	v_cvt_pk_f16_f32 v70, v92, v93
	v_cvt_pk_f16_f32 v71, v94, v95
	buffer_store_dwordx2 v[72:73], v0, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[44:47], a[20:23], a[0:3], v[44:47]
	v_add_lshl_u32 v0, v28, v40, 1
	v_cvt_pk_f16_f32 v68, v96, v97
	v_cvt_pk_f16_f32 v69, v98, v99
	v_mfma_f32_16x16x32_f16 v[52:55], a[64:67], a[8:11], v[52:55]
	buffer_store_dwordx2 v[70:71], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v29, v40, 1
	buffer_store_dwordx2 v[68:69], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v30, v40, 1
	v_cvt_pk_f16_f32 v22, v56, v57
	v_cvt_pk_f16_f32 v23, v58, v59
	buffer_store_dwordx2 v[64:65], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v31, v40, 1
	v_mfma_f32_16x16x32_f16 v[44:47], a[24:27], a[4:7], v[44:47]
	v_cvt_pk_f16_f32 v20, v60, v61
	v_cvt_pk_f16_f32 v21, v62, v63
	buffer_store_dwordx2 v[22:23], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v32, v40, 1
	v_cvt_pk_f16_f32 v18, v52, v53
	v_cvt_pk_f16_f32 v19, v54, v55
	buffer_store_dwordx2 v[20:21], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v33, v40, 1
	v_mfma_f32_16x16x32_f16 v[252:255], a[68:71], a[0:3], v[252:255]
	buffer_store_dwordx2 v[18:19], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v34, v40, 1
	buffer_store_dwordx2 v[16:17], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v41, v27, 1
	v_cvt_pk_f16_f32 v12, v44, v45
	v_cvt_pk_f16_f32 v13, v46, v47
	buffer_store_dwordx2 v[14:15], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v28, v41, 1
	buffer_store_dwordx2 v[12:13], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v29, v41, 1
	v_mfma_f32_16x16x32_f16 v[252:255], a[72:75], a[4:7], v[252:255]
	buffer_store_dwordx2 v[8:9], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v30, v41, 1
	buffer_store_dwordx2 v[10:11], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v31, v41, 1
	buffer_store_dwordx2 v[6:7], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v32, v41, 1
	buffer_store_dwordx2 v[4:5], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v33, v41, 1
	v_cvt_pk_f16_f32 v2, v252, v253
	v_cvt_pk_f16_f32 v3, v254, v255
	buffer_store_dwordx2 v[24:25], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v34, v41, 1
	buffer_store_dwordx2 v[2:3], v0, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v5
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
		.amdhsa_next_free_vgpr 450
		.amdhsa_next_free_sgpr 38
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 0
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
	.size	v5, .Lfunc_end0-v5
	.cfi_endproc
                                        ; -- End function
	.set v5.num_vgpr, 256
	.set v5.num_agpr, 194
	.set v5.numbered_sgpr, 38
	.set v5.num_named_barrier, 0
	.set v5.private_seg_size, 0
	.set v5.uses_vcc, 0
	.set v5.uses_flat_scratch, 0
	.set v5.has_dyn_sized_stack, 0
	.set v5.has_recursion, 0
	.set v5.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11256
; TotalNumSgprs: 44
; NumVgprs: 256
; NumAgprs: 194
; TotalNumVgprs: 450
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 56
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 450
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
	.byte	1                               ; Abbrev [1] 0xb:0x81 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x5b DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x22 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	378                             ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4e:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	11                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x63:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	477                             ; DW_AT_call_line
	.byte	76                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x70:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	383                             ; DW_AT_call_line
	.byte	115                             ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x7d:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	404                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
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
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
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
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton/study_matmul/gluon/matmul_kernels" ; string offset=24
.Linfo_string3:
	.asciz	"v5"                            ; string offset=86
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     194
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
    .name:           v5
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         v5.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     450
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
