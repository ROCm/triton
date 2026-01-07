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
; %bb.9:
	.file	1 "/var/lib/jenkins/OAI-triton/study_matmul/gluon/matmul_kernels" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.10:
.LBB0_0:
	v_readfirstlane_b32 s20, v0
	s_bfe_u32 s17, s20, 0x20006
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
	v_cvt_f32_u32_e32 v1, s9
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mov_b32 s22, 0
	s_sub_i32 s14, 0, s9
	v_readfirstlane_b32 s15, v1
	s_mul_i32 s14, s14, s15
	s_mul_hi_u32 s14, s15, s14
	s_add_i32 s15, s15, s14
	s_mul_hi_u32 s14, s8, s15
	s_mul_i32 s15, s14, s9
	s_sub_i32 s8, s8, s15
	s_add_i32 s15, s14, 1
	s_sub_i32 s18, s8, s9
	s_cmp_ge_u32 s8, s9
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s8, s18, s8
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s8, s9
	s_cselect_b32 s8, s15, s14
	s_xor_b32 s8, s8, s1
	s_sub_i32 s1, s8, s1
	s_mul_i32 s0, s1, s0
	s_sub_i32 s9, s16, s0
	v_and_b32_e32 v1, 63, v0
	v_lshl_or_b32 v15, s17, 6, v1
	v_lshlrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v1, 0x70, v1
	v_or_b32_e32 v1, s17, v1
	v_lshlrev_b32_e32 v2, 3, v0
	v_and_b32_e32 v2, 56, v2
	s_lshl_b32 s15, s1, 8
	s_mul_i32 s0, s15, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s8, s2, s0
	s_addc_u32 s23, s3, s1
	s_lshl_b32 s14, s9, 8
	s_mul_i32 s0, s14, s12
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s4, s0
	s_addc_u32 s24, s5, s1
	s_lshl_b32 s1, s11, 2
	s_mul_i32 s4, s11, 0x74
	v_mad_u64_u32 v[4:5], s[2:3], v1, s11, v[2:3]
	v_add_u32_e32 v5, s1, v4
	v_add_u32_e32 v6, s1, v5
	v_add_u32_e32 v7, s1, v6
	v_add_u32_e32 v8, s4, v7
	v_add_u32_e32 v9, s1, v8
	v_add_u32_e32 v10, s1, v9
	s_lshl_b32 s16, s12, 2
	s_mul_i32 s4, s12, 0x74
	v_mad_u64_u32 v[2:3], s[2:3], v1, s12, v[2:3]
	v_add_u32_e32 v1, s16, v2
	v_add_u32_e32 v3, s16, v1
	v_add_u32_e32 v11, s16, v3
	v_add_u32_e32 v12, s4, v11
	v_add_u32_e32 v13, s16, v12
	v_add_u32_e32 v14, s16, v13
	s_add_i32 s25, s10, 63
	s_and_b32 s9, s23, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	s_mul_i32 s5, s17, 0x420
	s_add_i32 s12, s5, 0
	v_lshlrev_b32_e32 v4, 1, v4
	s_mov_b32 m0, s12
	s_nop 0
	buffer_load_dwordx4 v4, s[8:11], 0 offen lds
	s_add_i32 s26, s5, 0x1080
	s_add_i32 m0, s12, 0x1080
	v_lshlrev_b32_e32 v5, 1, v5
	buffer_load_dwordx4 v5, s[8:11], 0 offen lds
	s_add_i32 s27, s5, 0x2100
	s_add_i32 m0, s12, 0x2100
	v_lshlrev_b32_e32 v16, 1, v6
	v_mov_b32_e32 v6, v4
	v_mov_b32_e32 v4, v16
	buffer_load_dwordx4 v4, s[8:11], 0 offen lds
	s_add_i32 s33, s5, 0x3180
	s_add_i32 m0, s12, 0x3180
	v_lshlrev_b32_e32 v16, 1, v7
	v_mov_b32_e32 v7, v5
	v_mov_b32_e32 v5, v16
	buffer_load_dwordx4 v5, s[8:11], 0 offen lds
	s_add_i32 s34, s5, 0x4200
	s_add_i32 m0, s12, 0x4200
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_load_dwordx4 v8, s[8:11], 0 offen lds
	s_add_i32 s35, s5, 0x5280
	s_add_i32 m0, s12, 0x5280
	v_lshlrev_b32_e32 v9, 1, v9
	buffer_load_dwordx4 v9, s[8:11], 0 offen lds
	s_add_i32 s36, s5, 0x6300
	s_add_i32 m0, s12, 0x6300
	v_lshlrev_b32_e32 v16, 1, v10
	buffer_load_dwordx4 v16, s[8:11], 0 offen lds
	s_add_i32 s37, s5, 0x7380
	s_add_i32 m0, s12, 0x7380
	v_add_lshl_u32 v10, v10, s1, 1
	buffer_load_dwordx4 v10, s[8:11], 0 offen lds
	s_and_b32 s1, s24, 0xffff
	s_mov_b32 s2, s10
	s_mov_b32 s3, s11
	s_add_i32 s4, 0, 0x107e0
	s_add_i32 s21, s4, s5
	v_lshlrev_b32_e32 v17, 1, v2
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v17, s[0:3], 0 offen lds
	s_add_i32 m0, s4, s26
	v_lshlrev_b32_e32 v18, 1, v1
	buffer_load_dwordx4 v18, s[0:3], 0 offen lds
	s_add_i32 m0, s4, s27
	v_lshlrev_b32_e32 v19, 1, v3
	buffer_load_dwordx4 v19, s[0:3], 0 offen lds
	s_add_i32 m0, s4, s33
	v_lshlrev_b32_e32 v11, 1, v11
	buffer_load_dwordx4 v11, s[0:3], 0 offen lds
	s_add_i32 m0, s4, s34
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_load_dwordx4 v12, s[0:3], 0 offen lds
	s_add_i32 m0, s4, s35
	v_lshlrev_b32_e32 v13, 1, v13
	buffer_load_dwordx4 v13, s[0:3], 0 offen lds
	s_add_i32 m0, s4, s36
	v_lshlrev_b32_e32 v20, 1, v14
	buffer_load_dwordx4 v20, s[0:3], 0 offen lds
	s_add_i32 m0, s4, s37
	v_add_lshl_u32 v14, v14, s16, 1
	buffer_load_dwordx4 v14, s[0:3], 0 offen lds
	s_add_u32 s28, s8, 0x80
	s_addc_u32 s1, s23, 0
	s_add_u32 s16, s0, 0x80
	s_addc_u32 s2, s24, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b32 s29, s1, 0xffff
	s_mov_b32 s30, s10
	s_mov_b32 s31, s11
	s_add_i32 m0, s12, 0x8400
	v_accvgpr_write_b32 a210, v6
	buffer_load_dwordx4 v6, s[28:31], 0 offen lds
	s_add_i32 m0, s12, 0x9480
	v_accvgpr_write_b32 a212, v7
	buffer_load_dwordx4 v7, s[28:31], 0 offen lds
	s_add_i32 m0, s12, 0xa500
	s_nop 0
	buffer_load_dwordx4 v4, s[28:31], 0 offen lds
	s_add_i32 m0, s12, 0xb580
	s_nop 0
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	s_add_i32 m0, s12, 0xc600
	s_nop 0
	buffer_load_dwordx4 v8, s[28:31], 0 offen lds
	s_add_i32 m0, s12, 0xd680
	s_nop 0
	buffer_load_dwordx4 v9, s[28:31], 0 offen lds
	s_add_i32 m0, s12, 0xe700
	s_nop 0
	buffer_load_dwordx4 v16, s[28:31], 0 offen lds
	s_add_i32 m0, s12, 0xf780
	s_nop 0
	buffer_load_dwordx4 v10, s[28:31], 0 offen lds
	s_and_b32 s17, s2, 0xffff
	s_mov_b32 s18, s10
	s_mov_b32 s19, s11
	s_add_i32 s1, 0, 0x18be0
	s_add_i32 m0, s1, s5
	s_nop 0
	buffer_load_dwordx4 v17, s[16:19], 0 offen lds
	s_add_i32 m0, s1, s26
	s_nop 0
	buffer_load_dwordx4 v18, s[16:19], 0 offen lds
	s_add_i32 m0, s1, s27
	s_nop 0
	buffer_load_dwordx4 v19, s[16:19], 0 offen lds
	s_add_i32 m0, s1, s33
	s_nop 0
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_add_i32 m0, s1, s34
	s_nop 0
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_add_i32 m0, s1, s35
	s_nop 0
	buffer_load_dwordx4 v13, s[16:19], 0 offen lds
	s_add_i32 m0, s1, s36
	s_nop 0
	buffer_load_dwordx4 v20, s[16:19], 0 offen lds
	s_add_i32 m0, s1, s37
	s_nop 0
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v3, 15, v0
	v_lshlrev_b32_e32 v1, 10, v3
	s_movk_i32 s1, 0xb0
	v_accvgpr_write_b32 a208, v15
	v_and_or_b32 v2, v15, s1, v1
	v_accvgpr_write_b32 a209, v3
	v_lshlrev_b32_e32 v3, 5, v3
	v_add_u32_e32 v6, v2, v3
	v_add_u32_e32 v2, 0, v6
	ds_read_b128 a[40:43], v2
	ds_read_b128 a[44:47], v2 offset:64
	ds_read_b128 a[32:35], v2 offset:256
	ds_read_b128 a[36:39], v2 offset:320
	ds_read_b128 a[24:27], v2 offset:512
	ds_read_b128 a[28:31], v2 offset:576
	ds_read_b128 a[16:19], v2 offset:768
	ds_read_b128 a[20:23], v2 offset:832
	ds_read_b128 a[8:11], v2 offset:16896
	ds_read_b128 a[12:15], v2 offset:16960
	ds_read_b128 a[0:3], v2 offset:17152
	ds_read_b128 a[4:7], v2 offset:17216
	ds_read_b128 a[124:127], v2 offset:17408
	ds_read_b128 a[116:119], v2 offset:17472
	ds_read_b128 a[52:55], v2 offset:17664
	ds_read_b128 a[48:51], v2 offset:17728
	v_and_b32_e32 v0, 48, v0
	s_lshl_b32 s1, s20, 1
	s_and_b32 s1, s1, 0x80
	v_or3_b32 v0, v0, s1, v1
	v_add_u32_e32 v7, v0, v3
	v_add_u32_e32 v0, s4, v7
	ds_read_b128 a[120:123], v0
	ds_read_b128 a[112:115], v0 offset:64
	ds_read_b128 a[108:111], v0 offset:256
	ds_read_b128 a[104:107], v0 offset:320
	ds_read_b128 a[100:103], v0 offset:512
	ds_read_b128 a[96:99], v0 offset:576
	ds_read_b128 a[92:95], v0 offset:768
	ds_read_b128 a[88:91], v0 offset:832
	ds_read_b128 a[84:87], v0 offset:16896
	ds_read_b128 a[80:83], v0 offset:16960
	ds_read_b128 a[76:79], v0 offset:17152
	ds_read_b128 a[72:75], v0 offset:17216
	ds_read_b128 a[68:71], v0 offset:17408
	ds_read_b128 a[64:67], v0 offset:17472
	ds_read_b128 a[60:63], v0 offset:17664
	ds_read_b128 a[56:59], v0 offset:17728
	s_cmpk_lt_i32 s25, 0x80
	s_cbranch_scc1 .LBB0_5
; %bb.1:                                ; %.lr.ph
	v_accvgpr_write_b32 a223, v14
	v_accvgpr_write_b32 a222, v20
	v_accvgpr_write_b32 a221, v13
	v_accvgpr_write_b32 a220, v12
	v_accvgpr_write_b32 a219, v11
	v_accvgpr_write_b32 a218, v19
	v_accvgpr_write_b32 a217, v18
	v_accvgpr_write_b32 a216, v17
	v_accvgpr_write_b32 a215, v10
	v_accvgpr_write_b32 a214, v16
	v_accvgpr_write_b32 a213, v9
	v_accvgpr_write_b32 a211, v8
	s_lshr_b32 s3, s25, 6
	s_add_u32 s0, s0, 0x100
	s_addc_u32 s1, s24, 0
	s_add_u32 s8, s8, 0x100
	s_addc_u32 s9, s23, 0
	s_add_i32 s2, s3, -2
	s_cmp_eq_u32 s3, 2
	s_cbranch_scc1 .LBB0_6
; %bb.2:                                ; %.lr.ph.split
	v_accvgpr_write_b32 a225, v7
	v_accvgpr_write_b32 a224, v6
	s_mov_b32 s3, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v255, 0
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
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
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
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
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
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
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
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	v_accvgpr_mov_b32 a234, a211
	v_accvgpr_mov_b32 a235, a213
	v_accvgpr_read_b32 v6, a214
	v_accvgpr_read_b32 v7, a215
	v_accvgpr_mov_b32 a236, a216
	v_accvgpr_mov_b32 a240, a217
	v_accvgpr_mov_b32 a237, a218
	v_accvgpr_mov_b32 a241, a219
	v_accvgpr_mov_b32 a238, a220
	v_accvgpr_mov_b32 a242, a221
	v_accvgpr_mov_b32 a239, a222
	v_accvgpr_mov_b32 a243, a223
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s10, s3, 1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[252:255], a[120:123], a[40:43], v[252:255]
	v_mfma_f32_16x16x32_f16 v[252:255], a[112:115], a[44:47], v[252:255]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[248:251], a[108:111], a[40:43], v[248:251]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[248:251], a[104:107], a[44:47], v[248:251]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[244:247], a[100:103], a[40:43], v[244:247]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[244:247], a[96:99], a[44:47], v[244:247]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[240:243], a[92:95], a[40:43], v[240:243]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[240:243], a[88:91], a[44:47], v[240:243]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[236:239], a[84:87], a[40:43], v[236:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[236:239], a[80:83], a[44:47], v[236:239]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[232:235], a[76:79], a[40:43], v[232:235]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[232:235], a[72:75], a[44:47], v[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[228:231], a[68:71], a[40:43], v[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[228:231], a[64:67], a[44:47], v[228:231]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[224:227], a[60:63], a[40:43], v[224:227]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[224:227], a[56:59], a[44:47], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[120:123], a[32:35], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[112:115], a[36:39], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[108:111], a[32:35], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[104:107], a[36:39], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[100:103], a[32:35], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[96:99], a[36:39], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[92:95], a[32:35], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[88:91], a[36:39], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[84:87], a[32:35], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[80:83], a[36:39], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[76:79], a[32:35], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[72:75], a[36:39], v[200:203]
	v_mfma_f32_16x16x32_f16 v[196:199], a[68:71], a[32:35], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[36:39], v[196:199]
	v_mfma_f32_16x16x32_f16 v[192:195], a[60:63], a[32:35], v[192:195]
	v_mfma_f32_16x16x32_f16 v[192:195], a[56:59], a[36:39], v[192:195]
	s_and_b32 s17, s9, 0xffff
	s_mov_b32 s16, s8
	s_mul_i32 s11, s10, 0x8400
	s_add_i32 s22, s12, s11
	s_mov_b32 m0, s22
	v_accvgpr_read_b32 v1, a210
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[188:191], a[120:123], a[24:27], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[112:115], a[28:31], v[188:191]
	v_mfma_f32_16x16x32_f16 v[184:187], a[108:111], a[24:27], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[104:107], a[28:31], v[184:187]
	s_add_i32 m0, s22, 0x1080
	v_accvgpr_read_b32 v1, a212
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[180:183], a[100:103], a[24:27], v[180:183]
	v_mfma_f32_16x16x32_f16 v[180:183], a[96:99], a[28:31], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[92:95], a[24:27], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[88:91], a[28:31], v[176:179]
	s_add_i32 m0, s22, 0x2100
	s_nop 0
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[172:175], a[84:87], a[24:27], v[172:175]
	v_mfma_f32_16x16x32_f16 v[172:175], a[80:83], a[28:31], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[76:79], a[24:27], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[72:75], a[28:31], v[168:171]
	s_add_i32 m0, s22, 0x3180
	s_nop 0
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[164:167], a[68:71], a[24:27], v[164:167]
	v_mfma_f32_16x16x32_f16 v[164:167], a[64:67], a[28:31], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[60:63], a[24:27], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[56:59], a[28:31], v[160:163]
	s_add_i32 m0, s22, 0x4200
	v_accvgpr_read_b32 v1, a234
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[156:159], a[120:123], a[16:19], v[156:159]
	v_mfma_f32_16x16x32_f16 v[156:159], a[112:115], a[20:23], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[108:111], a[16:19], v[152:155]
	v_mfma_f32_16x16x32_f16 v[152:155], a[104:107], a[20:23], v[152:155]
	s_add_i32 m0, s22, 0x5280
	v_accvgpr_read_b32 v1, a235
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[148:151], a[100:103], a[16:19], v[148:151]
	v_mfma_f32_16x16x32_f16 v[148:151], a[96:99], a[20:23], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[92:95], a[16:19], v[144:147]
	v_mfma_f32_16x16x32_f16 v[144:147], a[88:91], a[20:23], v[144:147]
	s_add_i32 m0, s22, 0x6300
	s_nop 0
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[140:143], a[84:87], a[16:19], v[140:143]
	v_mfma_f32_16x16x32_f16 v[140:143], a[80:83], a[20:23], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[76:79], a[16:19], v[136:139]
	v_mfma_f32_16x16x32_f16 v[136:139], a[72:75], a[20:23], v[136:139]
	s_add_i32 m0, s22, 0x7380
	s_nop 0
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[132:135], a[68:71], a[16:19], v[132:135]
	v_mfma_f32_16x16x32_f16 v[132:135], a[64:67], a[20:23], v[132:135]
	v_mfma_f32_16x16x32_f16 v[128:131], a[60:63], a[16:19], v[128:131]
	v_mfma_f32_16x16x32_f16 v[128:131], a[56:59], a[20:23], v[128:131]
	s_and_b32 s17, s1, 0xffff
	s_mov_b32 s16, s0
	s_add_i32 s11, s21, s11
	s_mov_b32 m0, s11
	v_accvgpr_read_b32 v1, a236
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[124:127], a[120:123], a[8:11], v[124:127]
	v_mfma_f32_16x16x32_f16 v[124:127], a[112:115], a[12:15], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[108:111], a[8:11], v[120:123]
	v_mfma_f32_16x16x32_f16 v[120:123], a[104:107], a[12:15], v[120:123]
	s_add_i32 m0, s11, 0x1080
	v_accvgpr_read_b32 v1, a240
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[116:119], a[100:103], a[8:11], v[116:119]
	v_mfma_f32_16x16x32_f16 v[116:119], a[96:99], a[12:15], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[92:95], a[8:11], v[112:115]
	v_mfma_f32_16x16x32_f16 v[112:115], a[88:91], a[12:15], v[112:115]
	s_add_i32 m0, s11, 0x2100
	v_accvgpr_read_b32 v1, a237
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[108:111], a[84:87], a[8:11], v[108:111]
	v_mfma_f32_16x16x32_f16 v[108:111], a[80:83], a[12:15], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[76:79], a[8:11], v[104:107]
	v_mfma_f32_16x16x32_f16 v[104:107], a[72:75], a[12:15], v[104:107]
	s_add_i32 m0, s11, 0x3180
	v_accvgpr_read_b32 v1, a241
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[100:103], a[68:71], a[8:11], v[100:103]
	v_mfma_f32_16x16x32_f16 v[100:103], a[64:67], a[12:15], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[60:63], a[8:11], v[96:99]
	v_mfma_f32_16x16x32_f16 v[96:99], a[56:59], a[12:15], v[96:99]
	s_add_i32 m0, s11, 0x4200
	v_accvgpr_read_b32 v1, a238
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[92:95], a[120:123], a[0:3], v[92:95]
	v_mfma_f32_16x16x32_f16 v[92:95], a[112:115], a[4:7], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[108:111], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[88:91], a[104:107], a[4:7], v[88:91]
	s_add_i32 m0, s11, 0x5280
	v_accvgpr_read_b32 v1, a242
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[84:87], a[100:103], a[0:3], v[84:87]
	v_mfma_f32_16x16x32_f16 v[84:87], a[96:99], a[4:7], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[92:95], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[88:91], a[4:7], v[80:83]
	s_add_i32 m0, s11, 0x6300
	v_accvgpr_read_b32 v1, a239
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[76:79], a[84:87], a[0:3], v[76:79]
	v_mfma_f32_16x16x32_f16 v[76:79], a[80:83], a[4:7], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[76:79], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[72:75], a[72:75], a[4:7], v[72:75]
	s_add_i32 m0, s11, 0x7380
	v_accvgpr_read_b32 v1, a243
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[68:71], a[68:71], a[0:3], v[68:71]
	v_mfma_f32_16x16x32_f16 v[68:71], a[64:67], a[4:7], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[60:63], a[0:3], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[56:59], a[4:7], v[64:67]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s10, s10, 14
	s_xor_b32 s10, s10, 0x4000
	s_lshr_b32 s11, s10, 5
	s_or_b32 s10, s11, s10
	s_lshl_b32 s10, s10, 1
	v_add_u32_e32 v1, s10, v2
	ds_read_b128 a[40:43], v1
	v_mfma_f32_16x16x32_f16 v[60:63], a[120:123], a[124:127], v[60:63]
	ds_read_b128 a[44:47], v1 offset:64
	v_mfma_f32_16x16x32_f16 v[60:63], a[112:115], a[116:119], v[60:63]
	ds_read_b128 a[32:35], v1 offset:256
	v_mfma_f32_16x16x32_f16 v[56:59], a[108:111], a[124:127], v[56:59]
	ds_read_b128 a[36:39], v1 offset:320
	v_mfma_f32_16x16x32_f16 v[56:59], a[104:107], a[116:119], v[56:59]
	ds_read_b128 a[24:27], v1 offset:512
	v_mfma_f32_16x16x32_f16 v[52:55], a[100:103], a[124:127], v[52:55]
	ds_read_b128 a[28:31], v1 offset:576
	v_mfma_f32_16x16x32_f16 v[52:55], a[96:99], a[116:119], v[52:55]
	ds_read_b128 a[16:19], v1 offset:768
	v_mfma_f32_16x16x32_f16 v[48:51], a[92:95], a[124:127], v[48:51]
	ds_read_b128 a[20:23], v1 offset:832
	v_mfma_f32_16x16x32_f16 v[48:51], a[88:91], a[116:119], v[48:51]
	ds_read_b128 a[8:11], v1 offset:16896
	v_mfma_f32_16x16x32_f16 v[44:47], a[84:87], a[124:127], v[44:47]
	ds_read_b128 a[12:15], v1 offset:16960
	v_mfma_f32_16x16x32_f16 v[44:47], a[80:83], a[116:119], v[44:47]
	ds_read_b128 a[0:3], v1 offset:17152
	v_mfma_f32_16x16x32_f16 v[40:43], a[76:79], a[124:127], v[40:43]
	ds_read_b128 a[4:7], v1 offset:17216
	v_mfma_f32_16x16x32_f16 v[40:43], a[72:75], a[116:119], v[40:43]
	ds_read_b128 a[128:131], v1 offset:17408
	v_mfma_f32_16x16x32_f16 v[36:39], a[68:71], a[124:127], v[36:39]
	ds_read_b128 a[132:135], v1 offset:17472
	v_mfma_f32_16x16x32_f16 v[36:39], a[64:67], a[116:119], v[36:39]
	ds_read_b128 a[136:139], v1 offset:17664
	v_mfma_f32_16x16x32_f16 v[32:35], a[60:63], a[124:127], v[32:35]
	ds_read_b128 a[140:143], v1 offset:17728
	v_mfma_f32_16x16x32_f16 v[32:35], a[56:59], a[116:119], v[32:35]
	v_add_u32_e32 v1, s10, v0
	ds_read_b128 a[144:147], v1
	v_mfma_f32_16x16x32_f16 v[28:31], a[120:123], a[52:55], v[28:31]
	ds_read_b128 a[148:151], v1 offset:64
	v_mfma_f32_16x16x32_f16 v[28:31], a[112:115], a[48:51], v[28:31]
	ds_read_b128 a[152:155], v1 offset:256
	v_mfma_f32_16x16x32_f16 v[24:27], a[108:111], a[52:55], v[24:27]
	ds_read_b128 a[156:159], v1 offset:320
	v_mfma_f32_16x16x32_f16 v[24:27], a[104:107], a[48:51], v[24:27]
	ds_read_b128 a[160:163], v1 offset:512
	v_mfma_f32_16x16x32_f16 v[20:23], a[100:103], a[52:55], v[20:23]
	ds_read_b128 a[164:167], v1 offset:576
	v_mfma_f32_16x16x32_f16 v[20:23], a[96:99], a[48:51], v[20:23]
	ds_read_b128 a[168:171], v1 offset:768
	v_mfma_f32_16x16x32_f16 a[92:95], a[92:95], a[52:55], a[252:255]
	ds_read_b128 a[172:175], v1 offset:832
	v_mfma_f32_16x16x32_f16 a[252:255], a[88:91], a[48:51], a[92:95]
	ds_read_b128 a[176:179], v1 offset:16896
	v_mfma_f32_16x16x32_f16 a[84:87], a[84:87], a[52:55], a[248:251]
	ds_read_b128 a[180:183], v1 offset:16960
	v_mfma_f32_16x16x32_f16 a[248:251], a[80:83], a[48:51], a[84:87]
	ds_read_b128 a[184:187], v1 offset:17152
	v_mfma_f32_16x16x32_f16 a[76:79], a[76:79], a[52:55], a[244:247]
	ds_read_b128 a[188:191], v1 offset:17216
	v_mfma_f32_16x16x32_f16 a[244:247], a[72:75], a[48:51], a[76:79]
	ds_read_b128 a[192:195], v1 offset:17408
	v_mfma_f32_16x16x32_f16 a[230:233], a[68:71], a[52:55], a[230:233]
	ds_read_b128 a[196:199], v1 offset:17472
	v_mfma_f32_16x16x32_f16 a[230:233], a[64:67], a[48:51], a[230:233]
	ds_read_b128 a[200:203], v1 offset:17664
	v_mfma_f32_16x16x32_f16 a[226:229], a[60:63], a[52:55], a[226:229]
	ds_read_b128 a[204:207], v1 offset:17728
	v_mfma_f32_16x16x32_f16 a[226:229], a[56:59], a[48:51], a[226:229]
	s_add_u32 s8, s8, 0x80
	s_addc_u32 s9, s9, 0
	s_add_u32 s0, s0, 0x80
	s_addc_u32 s1, s1, 0
	s_add_i32 s3, s3, 1
	s_cmp_lg_u32 s2, s3
	s_waitcnt lgkmcnt(14)
	v_accvgpr_mov_b32 a124, a128
	v_accvgpr_mov_b32 a125, a129
	v_accvgpr_mov_b32 a126, a130
	v_accvgpr_mov_b32 a127, a131
	v_accvgpr_mov_b32 a116, a132
	v_accvgpr_mov_b32 a117, a133
	v_accvgpr_mov_b32 a118, a134
	v_accvgpr_mov_b32 a119, a135
	v_accvgpr_mov_b32 a52, a136
	v_accvgpr_mov_b32 a53, a137
	v_accvgpr_mov_b32 a54, a138
	v_accvgpr_mov_b32 a55, a139
	v_accvgpr_mov_b32 a48, a140
	v_accvgpr_mov_b32 a49, a141
	v_accvgpr_mov_b32 a50, a142
	v_accvgpr_mov_b32 a51, a143
	v_accvgpr_mov_b32 a120, a144
	v_accvgpr_mov_b32 a121, a145
	v_accvgpr_mov_b32 a122, a146
	v_accvgpr_mov_b32 a123, a147
	v_accvgpr_mov_b32 a112, a148
	v_accvgpr_mov_b32 a113, a149
	v_accvgpr_mov_b32 a114, a150
	v_accvgpr_mov_b32 a115, a151
	s_waitcnt lgkmcnt(13)
	v_accvgpr_mov_b32 a108, a152
	v_accvgpr_mov_b32 a109, a153
	v_accvgpr_mov_b32 a110, a154
	v_accvgpr_mov_b32 a111, a155
	s_waitcnt lgkmcnt(12)
	v_accvgpr_mov_b32 a104, a156
	v_accvgpr_mov_b32 a105, a157
	v_accvgpr_mov_b32 a106, a158
	v_accvgpr_mov_b32 a107, a159
	s_waitcnt lgkmcnt(11)
	v_accvgpr_mov_b32 a100, a160
	v_accvgpr_mov_b32 a101, a161
	v_accvgpr_mov_b32 a102, a162
	v_accvgpr_mov_b32 a103, a163
	s_waitcnt lgkmcnt(10)
	v_accvgpr_mov_b32 a96, a164
	v_accvgpr_mov_b32 a97, a165
	v_accvgpr_mov_b32 a98, a166
	v_accvgpr_mov_b32 a99, a167
	s_waitcnt lgkmcnt(9)
	v_accvgpr_mov_b32 a92, a168
	v_accvgpr_mov_b32 a93, a169
	v_accvgpr_mov_b32 a94, a170
	v_accvgpr_mov_b32 a95, a171
	s_waitcnt lgkmcnt(8)
	v_accvgpr_mov_b32 a88, a172
	v_accvgpr_mov_b32 a89, a173
	v_accvgpr_mov_b32 a90, a174
	v_accvgpr_mov_b32 a91, a175
	s_waitcnt lgkmcnt(7)
	v_accvgpr_mov_b32 a84, a176
	v_accvgpr_mov_b32 a85, a177
	v_accvgpr_mov_b32 a86, a178
	v_accvgpr_mov_b32 a87, a179
	s_waitcnt lgkmcnt(6)
	v_accvgpr_mov_b32 a80, a180
	v_accvgpr_mov_b32 a81, a181
	v_accvgpr_mov_b32 a82, a182
	v_accvgpr_mov_b32 a83, a183
	s_waitcnt lgkmcnt(5)
	v_accvgpr_mov_b32 a76, a184
	v_accvgpr_mov_b32 a77, a185
	v_accvgpr_mov_b32 a78, a186
	v_accvgpr_mov_b32 a79, a187
	s_waitcnt lgkmcnt(4)
	v_accvgpr_mov_b32 a72, a188
	v_accvgpr_mov_b32 a73, a189
	v_accvgpr_mov_b32 a74, a190
	v_accvgpr_mov_b32 a75, a191
	s_waitcnt lgkmcnt(3)
	v_accvgpr_mov_b32 a68, a192
	v_accvgpr_mov_b32 a69, a193
	v_accvgpr_mov_b32 a70, a194
	v_accvgpr_mov_b32 a71, a195
	s_waitcnt lgkmcnt(2)
	v_accvgpr_mov_b32 a64, a196
	v_accvgpr_mov_b32 a65, a197
	v_accvgpr_mov_b32 a66, a198
	v_accvgpr_mov_b32 a67, a199
	s_waitcnt lgkmcnt(1)
	v_accvgpr_mov_b32 a60, a200
	v_accvgpr_mov_b32 a61, a201
	v_accvgpr_mov_b32 a62, a202
	v_accvgpr_mov_b32 a63, a203
	s_waitcnt lgkmcnt(0)
	v_accvgpr_mov_b32 a56, a204
	v_accvgpr_mov_b32 a57, a205
	v_accvgpr_mov_b32 a58, a206
	v_accvgpr_mov_b32 a59, a207
	s_cbranch_scc1 .LBB0_3
; %bb.4:                                ; %Flow
	v_accvgpr_mov_b32 a59, a207
	v_accvgpr_mov_b32 a58, a206
	v_accvgpr_mov_b32 a57, a205
	v_accvgpr_mov_b32 a56, a204
	v_accvgpr_mov_b32 a63, a203
	v_accvgpr_mov_b32 a62, a202
	v_accvgpr_mov_b32 a61, a201
	v_accvgpr_mov_b32 a60, a200
	v_accvgpr_mov_b32 a67, a199
	v_accvgpr_mov_b32 a66, a198
	v_accvgpr_mov_b32 a65, a197
	v_accvgpr_mov_b32 a64, a196
	v_accvgpr_mov_b32 a71, a195
	v_accvgpr_mov_b32 a70, a194
	v_accvgpr_mov_b32 a69, a193
	v_accvgpr_mov_b32 a68, a192
	v_accvgpr_mov_b32 a75, a191
	v_accvgpr_mov_b32 a74, a190
	v_accvgpr_mov_b32 a73, a189
	v_accvgpr_mov_b32 a72, a188
	v_accvgpr_mov_b32 a79, a187
	v_accvgpr_mov_b32 a78, a186
	v_accvgpr_mov_b32 a77, a185
	v_accvgpr_mov_b32 a76, a184
	v_accvgpr_mov_b32 a83, a183
	v_accvgpr_mov_b32 a82, a182
	v_accvgpr_mov_b32 a81, a181
	v_accvgpr_mov_b32 a80, a180
	v_accvgpr_mov_b32 a87, a179
	v_accvgpr_mov_b32 a86, a178
	v_accvgpr_mov_b32 a85, a177
	v_accvgpr_mov_b32 a84, a176
	v_accvgpr_mov_b32 a91, a175
	v_accvgpr_mov_b32 a90, a174
	v_accvgpr_mov_b32 a89, a173
	v_accvgpr_mov_b32 a88, a172
	v_accvgpr_mov_b32 a95, a171
	v_accvgpr_mov_b32 a94, a170
	v_accvgpr_mov_b32 a93, a169
	v_accvgpr_mov_b32 a92, a168
	v_accvgpr_mov_b32 a99, a167
	v_accvgpr_mov_b32 a98, a166
	v_accvgpr_mov_b32 a97, a165
	v_accvgpr_mov_b32 a96, a164
	v_accvgpr_mov_b32 a103, a163
	v_accvgpr_mov_b32 a102, a162
	v_accvgpr_mov_b32 a101, a161
	v_accvgpr_mov_b32 a100, a160
	v_accvgpr_mov_b32 a107, a159
	v_accvgpr_mov_b32 a106, a158
	v_accvgpr_mov_b32 a105, a157
	v_accvgpr_mov_b32 a104, a156
	v_accvgpr_mov_b32 a111, a155
	v_accvgpr_mov_b32 a110, a154
	v_accvgpr_mov_b32 a109, a153
	v_accvgpr_mov_b32 a108, a152
	v_accvgpr_mov_b32 a115, a151
	v_accvgpr_mov_b32 a114, a150
	v_accvgpr_mov_b32 a113, a149
	v_accvgpr_mov_b32 a112, a148
	v_accvgpr_mov_b32 a123, a147
	v_accvgpr_mov_b32 a122, a146
	v_accvgpr_mov_b32 a121, a145
	v_accvgpr_mov_b32 a120, a144
	v_accvgpr_mov_b32 a51, a143
	v_accvgpr_mov_b32 a50, a142
	v_accvgpr_mov_b32 a49, a141
	v_accvgpr_mov_b32 a48, a140
	v_accvgpr_mov_b32 a55, a139
	v_accvgpr_mov_b32 a54, a138
	v_accvgpr_mov_b32 a53, a137
	v_accvgpr_mov_b32 a52, a136
	v_accvgpr_mov_b32 a119, a135
	v_accvgpr_mov_b32 a118, a134
	v_accvgpr_mov_b32 a117, a133
	v_accvgpr_mov_b32 a116, a132
	v_accvgpr_mov_b32 a127, a131
	v_accvgpr_mov_b32 a126, a130
	v_accvgpr_mov_b32 a125, a129
	v_accvgpr_mov_b32 a124, a128
	s_mov_b32 s22, s2
	v_accvgpr_read_b32 v6, a224
	v_accvgpr_read_b32 v7, a225
	s_branch .LBB0_7
.LBB0_5:
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a144, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
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
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
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
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
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
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
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
	v_mov_b32_e32 v255, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v252, 0
	s_branch .LBB0_8
.LBB0_6:
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
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
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
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
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
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
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
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
	v_mov_b32_e32 v255, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v252, 0
.LBB0_7:                                ; %._crit_edge.loopexit.peel.begin
	s_and_b32 s12, s22, 1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[40:43], v[252:255]
	v_mfma_f32_16x16x32_f16 v[252:255], a[112:115], a[44:47], v[0:3]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[40:43], v[248:251]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[248:251], a[104:107], a[44:47], v[0:3]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[40:43], v[244:247]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[244:247], a[96:99], a[44:47], v[0:3]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[40:43], v[240:243]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[240:243], a[88:91], a[44:47], v[0:3]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[40:43], v[236:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[236:239], a[80:83], a[44:47], v[0:3]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[40:43], v[232:235]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[232:235], a[72:75], a[44:47], v[0:3]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[40:43], v[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[228:231], a[64:67], a[44:47], v[0:3]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[40:43], v[224:227]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[224:227], a[56:59], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[32:35], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[112:115], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[32:35], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[104:107], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[32:35], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[96:99], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[32:35], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[88:91], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[32:35], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[80:83], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[32:35], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[72:75], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[32:35], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[32:35], v[192:195]
	v_mfma_f32_16x16x32_f16 v[192:195], a[56:59], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[24:27], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[112:115], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[24:27], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[104:107], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[24:27], v[180:183]
	v_mfma_f32_16x16x32_f16 v[180:183], a[96:99], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[24:27], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[88:91], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[24:27], v[172:175]
	v_mfma_f32_16x16x32_f16 v[172:175], a[80:83], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[24:27], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[72:75], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[24:27], v[164:167]
	v_mfma_f32_16x16x32_f16 v[164:167], a[64:67], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[24:27], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[56:59], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[16:19], v[156:159]
	v_mfma_f32_16x16x32_f16 v[156:159], a[112:115], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[16:19], v[152:155]
	v_mfma_f32_16x16x32_f16 v[152:155], a[104:107], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[16:19], v[148:151]
	v_mfma_f32_16x16x32_f16 v[148:151], a[96:99], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[16:19], v[144:147]
	v_mfma_f32_16x16x32_f16 v[144:147], a[88:91], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[16:19], v[140:143]
	v_mfma_f32_16x16x32_f16 v[140:143], a[80:83], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[16:19], v[136:139]
	v_mfma_f32_16x16x32_f16 v[136:139], a[72:75], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[16:19], v[132:135]
	v_mfma_f32_16x16x32_f16 v[132:135], a[64:67], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[16:19], v[128:131]
	v_mfma_f32_16x16x32_f16 v[128:131], a[56:59], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[8:11], v[124:127]
	v_mfma_f32_16x16x32_f16 v[124:127], a[112:115], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[8:11], v[120:123]
	v_mfma_f32_16x16x32_f16 v[120:123], a[104:107], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[8:11], v[116:119]
	v_mfma_f32_16x16x32_f16 v[116:119], a[96:99], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[8:11], v[112:115]
	v_mfma_f32_16x16x32_f16 v[112:115], a[88:91], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[8:11], v[108:111]
	v_mfma_f32_16x16x32_f16 v[108:111], a[80:83], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[8:11], v[104:107]
	v_mfma_f32_16x16x32_f16 v[104:107], a[72:75], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[8:11], v[100:103]
	v_mfma_f32_16x16x32_f16 v[100:103], a[64:67], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[8:11], v[96:99]
	v_mfma_f32_16x16x32_f16 v[96:99], a[56:59], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[0:3], v[92:95]
	v_mfma_f32_16x16x32_f16 v[92:95], a[112:115], a[4:7], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[88:91], a[104:107], a[4:7], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[0:3], v[84:87]
	v_mfma_f32_16x16x32_f16 v[84:87], a[96:99], a[4:7], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[88:91], a[4:7], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[0:3], v[76:79]
	v_mfma_f32_16x16x32_f16 v[76:79], a[80:83], a[4:7], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[72:75], a[72:75], a[4:7], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[0:3], v[68:71]
	v_mfma_f32_16x16x32_f16 v[68:71], a[64:67], a[4:7], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[0:3], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[56:59], a[4:7], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[124:127], v[60:63]
	v_mfma_f32_16x16x32_f16 v[60:63], a[112:115], a[116:119], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[124:127], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[104:107], a[116:119], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[124:127], v[52:55]
	v_mfma_f32_16x16x32_f16 v[52:55], a[96:99], a[116:119], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[124:127], v[48:51]
	v_mfma_f32_16x16x32_f16 v[48:51], a[88:91], a[116:119], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[124:127], v[44:47]
	v_mfma_f32_16x16x32_f16 v[44:47], a[80:83], a[116:119], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[124:127], v[40:43]
	v_mfma_f32_16x16x32_f16 v[40:43], a[72:75], a[116:119], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[124:127], v[36:39]
	v_mfma_f32_16x16x32_f16 v[36:39], a[64:67], a[116:119], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[60:63], a[124:127], v[32:35]
	v_mfma_f32_16x16x32_f16 v[32:35], a[56:59], a[116:119], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[52:55], v[28:31]
	v_mfma_f32_16x16x32_f16 v[28:31], a[112:115], a[48:51], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[52:55], v[24:27]
	v_mfma_f32_16x16x32_f16 v[24:27], a[104:107], a[48:51], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[52:55], v[20:23]
	v_mfma_f32_16x16x32_f16 v[20:23], a[96:99], a[48:51], v[0:3]
	v_mfma_f32_16x16x32_f16 a[144:147], a[92:95], a[52:55], a[252:255]
	v_mfma_f32_16x16x32_f16 a[144:147], a[88:91], a[48:51], a[144:147]
	v_mfma_f32_16x16x32_f16 a[140:143], a[84:87], a[52:55], a[248:251]
	v_mfma_f32_16x16x32_f16 a[140:143], a[80:83], a[48:51], a[140:143]
	v_mfma_f32_16x16x32_f16 a[136:139], a[76:79], a[52:55], a[244:247]
	v_mfma_f32_16x16x32_f16 a[136:139], a[72:75], a[48:51], a[136:139]
	v_accvgpr_mov_b32 a0, a230
	v_accvgpr_mov_b32 a1, a231
	v_accvgpr_mov_b32 a2, a232
	v_accvgpr_mov_b32 a3, a233
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[0:3], a[68:71], a[52:55], a[0:3]
	v_mfma_f32_16x16x32_f16 a[132:135], a[64:67], a[48:51], a[0:3]
	s_nop 6
	v_accvgpr_mov_b32 a0, a226
	v_accvgpr_mov_b32 a1, a227
	v_accvgpr_mov_b32 a2, a228
	v_accvgpr_mov_b32 a3, a229
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[0:3], a[60:63], a[52:55], a[0:3]
	v_mfma_f32_16x16x32_f16 a[128:131], a[56:59], a[48:51], a[0:3]
	s_cmp_eq_u32 s22, s2
	s_cselect_b64 vcc, -1, 0
	s_mul_i32 s2, s12, 0x8400
	s_add_i32 s3, s2, 0
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	s_add_i32 s3, s3, s5
	v_bfrev_b32_e32 v0, 1
	v_accvgpr_read_b32 v1, a210
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s3
	s_nop 0
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s3, 0x1080
	v_accvgpr_read_b32 v1, a212
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s3, 0x2100
	v_cndmask_b32_e32 v1, v4, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s3, 0x3180
	v_cndmask_b32_e32 v1, v5, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s3, 0x4200
	v_accvgpr_read_b32 v1, a211
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s3, 0x5280
	v_accvgpr_read_b32 v1, a213
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s3, 0x6300
	v_accvgpr_read_b32 v1, a214
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 m0, s3, 0x7380
	v_accvgpr_read_b32 v1, a215
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[8:11], 0 offen lds
	s_add_i32 s8, s4, s2
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, s10
	s_mov_b32 s3, s11
	s_add_i32 s5, s8, s5
	v_accvgpr_read_b32 v1, a216
	v_cndmask_b32_e32 v1, v1, v0, vcc
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x1080
	v_accvgpr_read_b32 v1, a217
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x2100
	v_accvgpr_read_b32 v1, a218
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x3180
	v_accvgpr_read_b32 v1, a219
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x4200
	v_accvgpr_read_b32 v1, a220
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x5280
	v_accvgpr_read_b32 v1, a221
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x6300
	v_accvgpr_read_b32 v1, a222
	v_cndmask_b32_e32 v1, v1, v0, vcc
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_add_i32 m0, s5, 0x7380
	v_accvgpr_read_b32 v1, a223
	v_cndmask_b32_e32 v0, v1, v0, vcc
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	s_lshl_b32 s0, s12, 14
	s_xor_b32 s0, s0, 0x4000
	s_lshr_b32 s1, s0, 5
	s_or_b32 s0, s1, s0
	s_lshl_b32 s0, s0, 1
	s_add_i32 s1, s0, 0
	v_add_u32_e32 v0, s1, v6
	ds_read_b128 a[40:43], v0
	ds_read_b128 a[44:47], v0 offset:64
	ds_read_b128 a[32:35], v0 offset:256
	ds_read_b128 a[36:39], v0 offset:320
	ds_read_b128 a[24:27], v0 offset:512
	ds_read_b128 a[28:31], v0 offset:576
	ds_read_b128 a[16:19], v0 offset:768
	ds_read_b128 a[20:23], v0 offset:832
	ds_read_b128 a[8:11], v0 offset:16896
	ds_read_b128 a[12:15], v0 offset:16960
	ds_read_b128 a[0:3], v0 offset:17152
	ds_read_b128 a[4:7], v0 offset:17216
	ds_read_b128 a[124:127], v0 offset:17408
	ds_read_b128 a[116:119], v0 offset:17472
	ds_read_b128 a[52:55], v0 offset:17664
	ds_read_b128 a[48:51], v0 offset:17728
	s_add_i32 s4, s4, s0
	v_add_u32_e32 v0, s4, v7
	ds_read_b128 a[120:123], v0
	ds_read_b128 a[112:115], v0 offset:64
	ds_read_b128 a[108:111], v0 offset:256
	ds_read_b128 a[104:107], v0 offset:320
	ds_read_b128 a[100:103], v0 offset:512
	ds_read_b128 a[96:99], v0 offset:576
	ds_read_b128 a[92:95], v0 offset:768
	ds_read_b128 a[88:91], v0 offset:832
	ds_read_b128 a[84:87], v0 offset:16896
	ds_read_b128 a[80:83], v0 offset:16960
	ds_read_b128 a[76:79], v0 offset:17152
	ds_read_b128 a[72:75], v0 offset:17216
	ds_read_b128 a[68:71], v0 offset:17408
	ds_read_b128 a[64:67], v0 offset:17472
	ds_read_b128 a[60:63], v0 offset:17664
	ds_read_b128 a[56:59], v0 offset:17728
.LBB0_8:                                ; %Flow394
	s_lshr_b32 s0, s20, 6
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[252:255], a[120:123], a[40:43], v[252:255]
	v_mfma_f32_16x16x32_f16 v[252:255], a[112:115], a[44:47], v[252:255]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[248:251], a[108:111], a[40:43], v[248:251]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[248:251], a[104:107], a[44:47], v[248:251]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[244:247], a[100:103], a[40:43], v[244:247]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[244:247], a[96:99], a[44:47], v[244:247]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[240:243], a[92:95], a[40:43], v[240:243]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[240:243], a[88:91], a[44:47], v[240:243]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[236:239], a[84:87], a[40:43], v[236:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[236:239], a[80:83], a[44:47], v[236:239]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[232:235], a[76:79], a[40:43], v[232:235]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[232:235], a[72:75], a[44:47], v[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[228:231], a[68:71], a[40:43], v[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[228:231], a[64:67], a[44:47], v[228:231]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[224:227], a[60:63], a[40:43], v[224:227]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[224:227], a[56:59], a[44:47], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[120:123], a[32:35], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[112:115], a[36:39], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[108:111], a[32:35], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[104:107], a[36:39], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[100:103], a[32:35], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[96:99], a[36:39], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[92:95], a[32:35], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[88:91], a[36:39], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[84:87], a[32:35], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[80:83], a[36:39], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[76:79], a[32:35], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[72:75], a[36:39], v[200:203]
	v_mfma_f32_16x16x32_f16 v[196:199], a[68:71], a[32:35], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[36:39], v[196:199]
	v_mfma_f32_16x16x32_f16 v[192:195], a[60:63], a[32:35], v[192:195]
	v_mfma_f32_16x16x32_f16 v[192:195], a[56:59], a[36:39], v[192:195]
	v_mfma_f32_16x16x32_f16 v[188:191], a[120:123], a[24:27], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[112:115], a[28:31], v[188:191]
	v_mfma_f32_16x16x32_f16 v[184:187], a[108:111], a[24:27], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[104:107], a[28:31], v[184:187]
	v_mfma_f32_16x16x32_f16 v[180:183], a[100:103], a[24:27], v[180:183]
	v_mfma_f32_16x16x32_f16 v[180:183], a[96:99], a[28:31], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[92:95], a[24:27], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[88:91], a[28:31], v[176:179]
	v_mfma_f32_16x16x32_f16 v[172:175], a[84:87], a[24:27], v[172:175]
	v_mfma_f32_16x16x32_f16 v[172:175], a[80:83], a[28:31], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[76:79], a[24:27], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[72:75], a[28:31], v[168:171]
	v_mfma_f32_16x16x32_f16 v[164:167], a[68:71], a[24:27], v[164:167]
	v_mfma_f32_16x16x32_f16 v[164:167], a[64:67], a[28:31], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[60:63], a[24:27], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[56:59], a[28:31], v[160:163]
	v_mfma_f32_16x16x32_f16 v[156:159], a[120:123], a[16:19], v[156:159]
	v_mfma_f32_16x16x32_f16 v[156:159], a[112:115], a[20:23], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[108:111], a[16:19], v[152:155]
	v_mfma_f32_16x16x32_f16 v[152:155], a[104:107], a[20:23], v[152:155]
	v_mfma_f32_16x16x32_f16 v[148:151], a[100:103], a[16:19], v[148:151]
	v_mfma_f32_16x16x32_f16 v[148:151], a[96:99], a[20:23], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[92:95], a[16:19], v[144:147]
	v_mfma_f32_16x16x32_f16 v[144:147], a[88:91], a[20:23], v[144:147]
	v_mfma_f32_16x16x32_f16 v[140:143], a[84:87], a[16:19], v[140:143]
	v_mfma_f32_16x16x32_f16 v[140:143], a[80:83], a[20:23], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[76:79], a[16:19], v[136:139]
	v_mfma_f32_16x16x32_f16 v[136:139], a[72:75], a[20:23], v[136:139]
	v_mfma_f32_16x16x32_f16 v[132:135], a[68:71], a[16:19], v[132:135]
	v_mfma_f32_16x16x32_f16 v[132:135], a[64:67], a[20:23], v[132:135]
	v_mfma_f32_16x16x32_f16 v[128:131], a[60:63], a[16:19], v[128:131]
	v_mfma_f32_16x16x32_f16 v[128:131], a[56:59], a[20:23], v[128:131]
	v_mfma_f32_16x16x32_f16 v[124:127], a[120:123], a[8:11], v[124:127]
	v_mfma_f32_16x16x32_f16 v[124:127], a[112:115], a[12:15], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[108:111], a[8:11], v[120:123]
	v_mfma_f32_16x16x32_f16 v[120:123], a[104:107], a[12:15], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[100:103], a[8:11], v[116:119]
	v_mfma_f32_16x16x32_f16 v[116:119], a[96:99], a[12:15], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[92:95], a[8:11], v[112:115]
	v_mfma_f32_16x16x32_f16 v[112:115], a[88:91], a[12:15], v[112:115]
	v_mfma_f32_16x16x32_f16 v[108:111], a[84:87], a[8:11], v[108:111]
	v_mfma_f32_16x16x32_f16 v[108:111], a[80:83], a[12:15], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[76:79], a[8:11], v[104:107]
	v_mfma_f32_16x16x32_f16 v[104:107], a[72:75], a[12:15], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[68:71], a[8:11], v[100:103]
	v_mfma_f32_16x16x32_f16 v[100:103], a[64:67], a[12:15], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[60:63], a[8:11], v[96:99]
	v_mfma_f32_16x16x32_f16 v[96:99], a[56:59], a[12:15], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[120:123], a[0:3], v[92:95]
	v_mfma_f32_16x16x32_f16 v[92:95], a[112:115], a[4:7], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[108:111], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[88:91], a[104:107], a[4:7], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[100:103], a[0:3], v[84:87]
	v_mfma_f32_16x16x32_f16 v[84:87], a[96:99], a[4:7], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[92:95], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[88:91], a[4:7], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[84:87], a[0:3], v[76:79]
	v_mfma_f32_16x16x32_f16 v[76:79], a[80:83], a[4:7], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[76:79], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[72:75], a[72:75], a[4:7], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[68:71], a[0:3], v[68:71]
	v_mfma_f32_16x16x32_f16 v[68:71], a[64:67], a[4:7], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[60:63], a[0:3], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[56:59], a[4:7], v[64:67]
	v_mfma_f32_16x16x32_f16 v[60:63], a[120:123], a[124:127], v[60:63]
	v_mfma_f32_16x16x32_f16 v[60:63], a[112:115], a[116:119], v[60:63]
	v_mfma_f32_16x16x32_f16 v[56:59], a[108:111], a[124:127], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[104:107], a[116:119], v[56:59]
	v_mfma_f32_16x16x32_f16 v[52:55], a[100:103], a[124:127], v[52:55]
	v_mfma_f32_16x16x32_f16 v[52:55], a[96:99], a[116:119], v[52:55]
	v_mfma_f32_16x16x32_f16 v[48:51], a[92:95], a[124:127], v[48:51]
	v_mfma_f32_16x16x32_f16 v[48:51], a[88:91], a[116:119], v[48:51]
	v_mfma_f32_16x16x32_f16 v[44:47], a[84:87], a[124:127], v[44:47]
	v_mfma_f32_16x16x32_f16 v[44:47], a[80:83], a[116:119], v[44:47]
	v_mfma_f32_16x16x32_f16 v[40:43], a[76:79], a[124:127], v[40:43]
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[116:119], v[40:43]
	v_mfma_f32_16x16x32_f16 v[36:39], a[68:71], a[124:127], v[36:39]
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], a[116:119], v[36:39]
	v_mfma_f32_16x16x32_f16 v[32:35], a[60:63], a[124:127], v[32:35]
	v_mfma_f32_16x16x32_f16 v[8:11], a[56:59], a[116:119], v[32:35]
	v_mfma_f32_16x16x32_f16 v[28:31], a[120:123], a[52:55], v[28:31]
	v_mfma_f32_16x16x32_f16 v[12:15], a[112:115], a[48:51], v[28:31]
	v_mfma_f32_16x16x32_f16 v[24:27], a[108:111], a[52:55], v[24:27]
	v_mfma_f32_16x16x32_f16 v[16:19], a[104:107], a[48:51], v[24:27]
	v_mfma_f32_16x16x32_f16 v[20:23], a[100:103], a[52:55], v[20:23]
	v_mfma_f32_16x16x32_f16 v[20:23], a[96:99], a[48:51], v[20:23]
	s_nop 4
	v_accvgpr_read_b32 v24, a144
	v_accvgpr_read_b32 v25, a145
	v_accvgpr_read_b32 v26, a146
	v_accvgpr_read_b32 v27, a147
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[24:27], a[92:95], a[52:55], v[24:27]
	v_mfma_f32_16x16x32_f16 v[24:27], a[88:91], a[48:51], v[24:27]
	v_accvgpr_read_b32 v28, a140
	v_accvgpr_read_b32 v29, a141
	v_accvgpr_read_b32 v30, a142
	v_accvgpr_read_b32 v31, a143
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[28:31], a[84:87], a[52:55], v[28:31]
	v_mfma_f32_16x16x32_f16 v[28:31], a[80:83], a[48:51], v[28:31]
	v_accvgpr_read_b32 v32, a136
	v_accvgpr_read_b32 v33, a137
	v_accvgpr_read_b32 v34, a138
	v_accvgpr_read_b32 v35, a139
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[32:35], a[76:79], a[52:55], v[32:35]
	v_mfma_f32_16x16x32_f16 v[32:35], a[72:75], a[48:51], v[32:35]
	v_accvgpr_read_b32 v36, a132
	v_accvgpr_read_b32 v37, a133
	v_accvgpr_read_b32 v38, a134
	v_accvgpr_read_b32 v39, a135
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[36:39], a[68:71], a[52:55], v[36:39]
	v_mfma_f32_16x16x32_f16 v[36:39], a[64:67], a[48:51], v[36:39]
	v_accvgpr_read_b32 v40, a128
	v_accvgpr_read_b32 v41, a129
	v_accvgpr_read_b32 v42, a130
	v_accvgpr_read_b32 v43, a131
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[40:43], a[60:63], a[52:55], v[40:43]
	v_mfma_f32_16x16x32_f16 v[40:43], a[56:59], a[48:51], v[40:43]
	v_cvt_pk_f16_f32 v252, v252, v253
	v_cvt_pk_f16_f32 v253, v254, v255
	v_cvt_pk_f16_f32 v248, v248, v249
	v_cvt_pk_f16_f32 v249, v250, v251
	v_cvt_pk_f16_f32 v244, v244, v245
	v_cvt_pk_f16_f32 v245, v246, v247
	v_cvt_pk_f16_f32 v240, v240, v241
	v_cvt_pk_f16_f32 v241, v242, v243
	v_cvt_pk_f16_f32 v236, v236, v237
	v_cvt_pk_f16_f32 v237, v238, v239
	v_cvt_pk_f16_f32 v232, v232, v233
	v_cvt_pk_f16_f32 v233, v234, v235
	v_cvt_pk_f16_f32 v228, v228, v229
	v_cvt_pk_f16_f32 v229, v230, v231
	v_cvt_pk_f16_f32 v224, v224, v225
	v_cvt_pk_f16_f32 v225, v226, v227
	v_cvt_pk_f16_f32 v220, v220, v221
	v_cvt_pk_f16_f32 v221, v222, v223
	v_cvt_pk_f16_f32 v216, v216, v217
	v_cvt_pk_f16_f32 v217, v218, v219
	v_cvt_pk_f16_f32 v212, v212, v213
	v_cvt_pk_f16_f32 v213, v214, v215
	v_cvt_pk_f16_f32 v208, v208, v209
	v_cvt_pk_f16_f32 v209, v210, v211
	v_cvt_pk_f16_f32 v204, v204, v205
	v_cvt_pk_f16_f32 v205, v206, v207
	v_cvt_pk_f16_f32 v200, v200, v201
	v_cvt_pk_f16_f32 v201, v202, v203
	v_cvt_pk_f16_f32 v196, v196, v197
	v_cvt_pk_f16_f32 v197, v198, v199
	v_cvt_pk_f16_f32 v192, v192, v193
	v_cvt_pk_f16_f32 v193, v194, v195
	v_cvt_pk_f16_f32 v188, v188, v189
	v_cvt_pk_f16_f32 v189, v190, v191
	v_cvt_pk_f16_f32 v184, v184, v185
	v_cvt_pk_f16_f32 v185, v186, v187
	v_cvt_pk_f16_f32 v180, v180, v181
	v_cvt_pk_f16_f32 v181, v182, v183
	v_cvt_pk_f16_f32 v176, v176, v177
	v_cvt_pk_f16_f32 v177, v178, v179
	v_cvt_pk_f16_f32 v172, v172, v173
	v_cvt_pk_f16_f32 v173, v174, v175
	v_cvt_pk_f16_f32 v168, v168, v169
	v_cvt_pk_f16_f32 v169, v170, v171
	v_cvt_pk_f16_f32 v164, v164, v165
	v_cvt_pk_f16_f32 v165, v166, v167
	v_cvt_pk_f16_f32 v160, v160, v161
	v_cvt_pk_f16_f32 v161, v162, v163
	v_cvt_pk_f16_f32 v156, v156, v157
	v_cvt_pk_f16_f32 v157, v158, v159
	v_cvt_pk_f16_f32 v152, v152, v153
	v_cvt_pk_f16_f32 v153, v154, v155
	v_cvt_pk_f16_f32 v148, v148, v149
	v_cvt_pk_f16_f32 v149, v150, v151
	v_cvt_pk_f16_f32 v144, v144, v145
	v_cvt_pk_f16_f32 v145, v146, v147
	v_cvt_pk_f16_f32 v140, v140, v141
	v_cvt_pk_f16_f32 v141, v142, v143
	v_cvt_pk_f16_f32 v136, v136, v137
	v_cvt_pk_f16_f32 v137, v138, v139
	v_cvt_pk_f16_f32 v132, v132, v133
	v_cvt_pk_f16_f32 v133, v134, v135
	v_cvt_pk_f16_f32 v128, v128, v129
	v_cvt_pk_f16_f32 v129, v130, v131
	v_cvt_pk_f16_f32 v124, v124, v125
	v_cvt_pk_f16_f32 v125, v126, v127
	v_cvt_pk_f16_f32 v120, v120, v121
	v_cvt_pk_f16_f32 v121, v122, v123
	v_cvt_pk_f16_f32 v116, v116, v117
	v_cvt_pk_f16_f32 v117, v118, v119
	v_cvt_pk_f16_f32 v112, v112, v113
	v_cvt_pk_f16_f32 v113, v114, v115
	v_cvt_pk_f16_f32 v108, v108, v109
	v_cvt_pk_f16_f32 v109, v110, v111
	v_cvt_pk_f16_f32 v104, v104, v105
	v_cvt_pk_f16_f32 v105, v106, v107
	v_cvt_pk_f16_f32 v100, v100, v101
	v_cvt_pk_f16_f32 v101, v102, v103
	v_cvt_pk_f16_f32 v96, v96, v97
	v_cvt_pk_f16_f32 v97, v98, v99
	v_cvt_pk_f16_f32 v92, v92, v93
	v_cvt_pk_f16_f32 v93, v94, v95
	v_cvt_pk_f16_f32 v88, v88, v89
	v_cvt_pk_f16_f32 v89, v90, v91
	v_cvt_pk_f16_f32 v90, v84, v85
	v_cvt_pk_f16_f32 v91, v86, v87
	v_cvt_pk_f16_f32 v84, v80, v81
	v_cvt_pk_f16_f32 v85, v82, v83
	v_cvt_pk_f16_f32 v80, v76, v77
	v_cvt_pk_f16_f32 v81, v78, v79
	v_cvt_pk_f16_f32 v76, v72, v73
	v_cvt_pk_f16_f32 v77, v74, v75
	v_cvt_pk_f16_f32 v72, v68, v69
	v_cvt_pk_f16_f32 v73, v70, v71
	v_cvt_pk_f16_f32 v68, v64, v65
	v_cvt_pk_f16_f32 v69, v66, v67
	v_cvt_pk_f16_f32 v64, v60, v61
	v_cvt_pk_f16_f32 v65, v62, v63
	v_cvt_pk_f16_f32 v60, v56, v57
	v_cvt_pk_f16_f32 v61, v58, v59
	v_cvt_pk_f16_f32 v56, v52, v53
	v_cvt_pk_f16_f32 v57, v54, v55
	v_cvt_pk_f16_f32 v52, v48, v49
	v_cvt_pk_f16_f32 v53, v50, v51
	v_cvt_pk_f16_f32 v50, v44, v45
	v_cvt_pk_f16_f32 v51, v46, v47
	v_cvt_pk_f16_f32 v48, v0, v1
	v_cvt_pk_f16_f32 v49, v2, v3
	v_cvt_pk_f16_f32 v46, v4, v5
	v_cvt_pk_f16_f32 v47, v6, v7
	v_cvt_pk_f16_f32 v44, v8, v9
	v_cvt_pk_f16_f32 v45, v10, v11
	v_cvt_pk_f16_f32 v54, v12, v13
	v_cvt_pk_f16_f32 v55, v14, v15
	v_cvt_pk_f16_f32 v12, v16, v17
	v_cvt_pk_f16_f32 v13, v18, v19
	v_cvt_pk_f16_f32 v10, v20, v21
	v_cvt_pk_f16_f32 v11, v22, v23
	v_cvt_pk_f16_f32 v8, v24, v25
	v_cvt_pk_f16_f32 v9, v26, v27
	v_cvt_pk_f16_f32 v6, v28, v29
	v_cvt_pk_f16_f32 v7, v30, v31
	v_cvt_pk_f16_f32 v4, v32, v33
	v_cvt_pk_f16_f32 v5, v34, v35
	v_cvt_pk_f16_f32 v2, v36, v37
	v_cvt_pk_f16_f32 v3, v38, v39
	v_cvt_pk_f16_f32 v0, v40, v41
	v_cvt_pk_f16_f32 v1, v42, v43
	s_lshl_b32 s0, s0, 3
	v_accvgpr_read_b32 v14, a209
	v_and_or_b32 v14, s0, 16, v14
	v_accvgpr_read_b32 v15, a208
	v_lshrrev_b32_e32 v15, 2, v15
	v_and_b32_e32 v15, 28, v15
	v_or_b32_e32 v16, 32, v15
	v_or_b32_e32 v17, 64, v15
	v_or_b32_e32 v18, 0x60, v15
	v_or_b32_e32 v19, 0x80, v15
	v_or_b32_e32 v20, 0xa0, v15
	v_or_b32_e32 v21, 0xc0, v15
	v_or_b32_e32 v22, 0xe0, v15
	s_mul_i32 s0, s15, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[0:1], s[14:15], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	v_mul_lo_u32 v14, v14, s13
	s_lshl_b32 s2, s13, 5
	v_add_u32_e32 v23, s2, v14
	v_add_u32_e32 v24, s2, v23
	v_add_u32_e32 v25, s2, v24
	v_add_u32_e32 v26, s2, v25
	v_add_u32_e32 v27, s2, v26
	v_add_u32_e32 v28, s2, v27
	v_add_u32_e32 v29, s2, v28
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_add_lshl_u32 v30, v15, v14, 1
	buffer_store_dwordx2 v[252:253], v30, s[0:3], 0 offen
	v_add_lshl_u32 v30, v16, v14, 1
	buffer_store_dwordx2 v[248:249], v30, s[0:3], 0 offen
	v_add_lshl_u32 v30, v17, v14, 1
	buffer_store_dwordx2 v[244:245], v30, s[0:3], 0 offen
	v_add_lshl_u32 v30, v18, v14, 1
	buffer_store_dwordx2 v[240:241], v30, s[0:3], 0 offen
	v_add_lshl_u32 v30, v19, v14, 1
	buffer_store_dwordx2 v[236:237], v30, s[0:3], 0 offen
	v_add_lshl_u32 v30, v20, v14, 1
	buffer_store_dwordx2 v[232:233], v30, s[0:3], 0 offen
	v_add_lshl_u32 v30, v21, v14, 1
	buffer_store_dwordx2 v[228:229], v30, s[0:3], 0 offen
	v_add_lshl_u32 v14, v22, v14, 1
	buffer_store_dwordx2 v[224:225], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v23, v15, 1
	buffer_store_dwordx2 v[220:221], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v16, v23, 1
	buffer_store_dwordx2 v[216:217], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v17, v23, 1
	buffer_store_dwordx2 v[212:213], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v18, v23, 1
	buffer_store_dwordx2 v[208:209], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v19, v23, 1
	buffer_store_dwordx2 v[204:205], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v20, v23, 1
	buffer_store_dwordx2 v[200:201], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v21, v23, 1
	buffer_store_dwordx2 v[196:197], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v22, v23, 1
	buffer_store_dwordx2 v[192:193], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v24, v15, 1
	buffer_store_dwordx2 v[188:189], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v16, v24, 1
	buffer_store_dwordx2 v[184:185], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v17, v24, 1
	buffer_store_dwordx2 v[180:181], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v18, v24, 1
	buffer_store_dwordx2 v[176:177], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v19, v24, 1
	buffer_store_dwordx2 v[172:173], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v20, v24, 1
	buffer_store_dwordx2 v[168:169], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v21, v24, 1
	buffer_store_dwordx2 v[164:165], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v22, v24, 1
	buffer_store_dwordx2 v[160:161], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v25, v15, 1
	buffer_store_dwordx2 v[156:157], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v16, v25, 1
	buffer_store_dwordx2 v[152:153], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v17, v25, 1
	buffer_store_dwordx2 v[148:149], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v18, v25, 1
	buffer_store_dwordx2 v[144:145], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v19, v25, 1
	buffer_store_dwordx2 v[140:141], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v20, v25, 1
	buffer_store_dwordx2 v[136:137], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v21, v25, 1
	buffer_store_dwordx2 v[132:133], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v22, v25, 1
	buffer_store_dwordx2 v[128:129], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v26, v15, 1
	buffer_store_dwordx2 v[124:125], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v16, v26, 1
	buffer_store_dwordx2 v[120:121], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v17, v26, 1
	buffer_store_dwordx2 v[116:117], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v18, v26, 1
	buffer_store_dwordx2 v[112:113], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v19, v26, 1
	buffer_store_dwordx2 v[108:109], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v20, v26, 1
	buffer_store_dwordx2 v[104:105], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v21, v26, 1
	buffer_store_dwordx2 v[100:101], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v22, v26, 1
	buffer_store_dwordx2 v[96:97], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v27, v15, 1
	buffer_store_dwordx2 v[92:93], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v16, v27, 1
	buffer_store_dwordx2 v[88:89], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v17, v27, 1
	buffer_store_dwordx2 v[90:91], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v18, v27, 1
	buffer_store_dwordx2 v[84:85], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v19, v27, 1
	buffer_store_dwordx2 v[80:81], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v20, v27, 1
	buffer_store_dwordx2 v[76:77], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v21, v27, 1
	buffer_store_dwordx2 v[72:73], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v22, v27, 1
	buffer_store_dwordx2 v[68:69], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v28, v15, 1
	buffer_store_dwordx2 v[64:65], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v16, v28, 1
	buffer_store_dwordx2 v[60:61], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v17, v28, 1
	buffer_store_dwordx2 v[56:57], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v18, v28, 1
	buffer_store_dwordx2 v[52:53], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v19, v28, 1
	buffer_store_dwordx2 v[50:51], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v20, v28, 1
	buffer_store_dwordx2 v[48:49], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v21, v28, 1
	buffer_store_dwordx2 v[46:47], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v22, v28, 1
	buffer_store_dwordx2 v[44:45], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v29, v15, 1
	buffer_store_dwordx2 v[54:55], v14, s[0:3], 0 offen
	v_add_lshl_u32 v14, v16, v29, 1
	buffer_store_dwordx2 v[12:13], v14, s[0:3], 0 offen
	v_add_lshl_u32 v12, v17, v29, 1
	buffer_store_dwordx2 v[10:11], v12, s[0:3], 0 offen
	v_add_lshl_u32 v10, v18, v29, 1
	buffer_store_dwordx2 v[8:9], v10, s[0:3], 0 offen
	v_add_lshl_u32 v8, v19, v29, 1
	buffer_store_dwordx2 v[6:7], v8, s[0:3], 0 offen
	v_add_lshl_u32 v6, v20, v29, 1
	buffer_store_dwordx2 v[4:5], v6, s[0:3], 0 offen
	v_add_lshl_u32 v4, v21, v29, 1
	buffer_store_dwordx2 v[2:3], v4, s[0:3], 0 offen
	v_add_lshl_u32 v2, v22, v29, 1
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
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
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 38
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
	.size	v5, .Lfunc_end0-v5
	.cfi_endproc
                                        ; -- End function
	.set v5.num_vgpr, 256
	.set v5.num_agpr, 256
	.set v5.numbered_sgpr, 38
	.set v5.num_named_barrier, 0
	.set v5.private_seg_size, 0
	.set v5.uses_vcc, 1
	.set v5.uses_flat_scratch, 0
	.set v5.has_dyn_sized_stack, 0
	.set v5.has_recursion, 0
	.set v5.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13288
; TotalNumSgprs: 44
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 512
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
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
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
	.byte	1                               ; Abbrev [1] 0xb:0x91 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x6b DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	469                             ; DW_AT_call_line
	.byte	76                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4e:0x2a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp4-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	371                             ; DW_AT_call_line
	.byte	43                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x63:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	12                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x78:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp4                          ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	375                             ; DW_AT_call_line
	.byte	115                             ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x8d:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	397                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
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
    .vgpr_count:     512
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
