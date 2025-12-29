	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v4                              ; -- Begin function v4
	.p2align	8
	.type	v4,@function
v4:                                     ; @v4
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.8:
	.file	1 "/var/lib/jenkins/OAI-triton/study_matmul/gluon/matmul_kernels" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.9:
.LBB0_0:
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
	s_abs_i32 s1, s0
	v_mov_b32_e32 v22, v0
	v_cvt_f32_u32_e32 v0, s1
	s_sub_i32 s14, 0, s1
	s_abs_i32 s9, s16
	v_readfirstlane_b32 s24, v22
	v_rcp_iflag_f32_e32 v0, v0
	s_xor_b32 s8, s16, s0
	s_bfe_u32 s25, s24, 0x20006
	s_ashr_i32 s8, s8, 31
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_mul_i32 s21, s11, 0x74
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_readfirstlane_b32 s15, v0
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
	s_lshl_b32 s9, s1, 8
	s_sub_i32 s8, s16, s0
	s_mul_i32 s0, s9, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s20, s2, s0
	s_addc_u32 s17, s3, s1
	s_lshl_b32 s8, s8, 8
	v_lshlrev_b32_e32 v0, 1, v22
	s_mul_i32 s14, s8, s12
	v_and_b32_e32 v0, 0x70, v0
	s_ashr_i32 s15, s14, 31
	v_or_b32_e32 v1, s25, v0
	v_lshlrev_b32_e32 v0, 3, v22
	s_lshl_b64 s[14:15], s[14:15], 1
	v_and_b32_e32 v0, 56, v0
	s_add_u32 s16, s4, s14
	s_addc_u32 s26, s5, s15
	s_lshl_b32 s27, s11, 2
	v_mad_u64_u32 v[2:3], s[18:19], v1, s11, v[0:1]
	v_add_u32_e32 v3, s27, v2
	v_add_u32_e32 v4, s27, v3
	s_lshl_b32 s11, s12, 2
	v_mad_u64_u32 v[0:1], s[18:19], v1, s12, v[0:1]
	v_add_u32_e32 v5, s27, v4
	v_add_u32_e32 v1, s11, v0
	v_add_u32_e32 v6, s21, v5
	s_mul_i32 s21, s12, 0x74
	v_add_u32_e32 v9, s11, v1
	s_mul_i32 s12, s25, 0x420
	v_add_u32_e32 v10, s11, v9
	s_add_i32 s12, s12, 0
	v_add_u32_e32 v11, s21, v10
	s_and_b32 s21, s17, 0xffff
	v_lshlrev_b32_e32 v32, 1, v2
	s_mov_b32 m0, s12
	v_lshlrev_b32_e32 v33, 1, v3
	buffer_load_dwordx4 v32, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x1080
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_load_dwordx4 v33, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x2100
	v_lshlrev_b32_e32 v5, 1, v5
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x3180
	v_add_u32_e32 v7, s27, v6
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x4200
	v_lshlrev_b32_e32 v2, 1, v6
	v_add_u32_e32 v8, s27, v7
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x5280
	v_lshlrev_b32_e32 v6, 1, v7
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x6300
	v_lshlrev_b32_e32 v7, 1, v8
	buffer_load_dwordx4 v7, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x7380
	v_add_lshl_u32 v8, v8, s27, 1
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	s_add_i32 s20, s12, 0x107e0
	s_and_b32 s17, s26, 0xffff
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	v_lshlrev_b32_e32 v14, 1, v0
	s_mov_b32 m0, s20
	v_lshlrev_b32_e32 v15, 1, v1
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x11860
	v_lshlrev_b32_e32 v9, 1, v9
	buffer_load_dwordx4 v15, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x128e0
	v_lshlrev_b32_e32 v10, 1, v10
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x13960
	v_add_u32_e32 v12, s11, v11
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x149e0
	v_lshlrev_b32_e32 v11, 1, v11
	v_add_u32_e32 v13, s11, v12
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x15a60
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x16ae0
	v_lshlrev_b32_e32 v255, 1, v13
	buffer_load_dwordx4 v255, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x17b60
	v_add_lshl_u32 v254, v13, s11, 1
	buffer_load_dwordx4 v254, s[16:19], 0 offen lds
	s_add_i32 s10, s10, 63
	s_ashr_i32 s11, s10, 31
	s_lshr_b32 s11, s11, 26
	s_add_i32 s11, s10, s11
	v_and_b32_e32 v0, 63, v22
	s_ashr_i32 s16, s11, 6
	v_lshl_or_b32 v1, s25, 6, v0
	s_cmpk_gt_i32 s10, 0x7f
	v_and_b32_e32 v3, 48, v22
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %.._crit_edge_crit_edge
	s_lshl_b32 s10, s24, 1
	s_and_b32 s17, s10, 0x80
	v_and_b32_e32 v23, 0xb0, v1
	v_mov_b32_e32 v0, s17
	s_add_i32 s10, s16, -1
	s_cbranch_execz .LBB0_4
; %bb.2:
	v_accvgpr_write_b32 a116, v1
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a118, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v242, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	s_branch .LBB0_7
.LBB0_3:
                                        ; implicit-def: $vgpr23
                                        ; implicit-def: $vgpr0
	s_add_i32 s10, s16, -1
.LBB0_4:                                ; %.lr.ph
	v_lshlrev_b32_e32 v0, 10, v22
	s_lshl_b32 s16, s24, 1
	v_and_b32_e32 v0, 0x3c00, v0
	s_and_b32 s16, s16, 0x80
	v_accvgpr_write_b32 a116, v1
	v_accvgpr_write_b32 a119, v6
	v_and_b32_e32 v6, 0xb0, v1
	v_accvgpr_write_b32 a120, v2
	v_add_u32_e32 v2, 0, v0
	s_add_i32 s17, s16, 0
	s_add_i32 s17, s17, 0x107e0
	v_lshrrev_b32_e32 v1, 5, v0
	v_add3_u32 v2, v2, v6, v1
	v_accvgpr_write_b32 a121, v2
	v_add_u32_e32 v2, s17, v3
	v_add3_u32 v0, v2, v0, v1
	v_accvgpr_write_b32 a112, v22
	s_mov_b32 s11, 0
	s_max_i32 s17, s10, 1
	s_add_u32 s4, s4, s14
	s_addc_u32 s5, s5, s15
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	v_accvgpr_write_b32 a114, v6
	v_accvgpr_write_b32 a113, v3
	v_accvgpr_write_b32 a126, v0
	s_add_u32 s14, s0, 0x80
	s_addc_u32 s15, s1, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_accvgpr_write_b32 a127, v10
	v_mov_b32_e32 v10, 0
	v_accvgpr_write_b32 a128, v11
	v_mov_b32_e32 v11, 0
	v_accvgpr_write_b32 a129, v12
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_accvgpr_write_b32 a130, v14
	v_mov_b32_e32 v14, 0
	v_accvgpr_write_b32 a131, v15
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a132, v7
	v_mov_b32_e32 v7, 0
	v_accvgpr_write_b32 a133, v8
	v_mov_b32_e32 v8, 0
	v_accvgpr_write_b32 a134, v9
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
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
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s18, s11, 1
	s_lshl_b32 s0, s18, 14
	s_xor_b32 s0, s0, 0x4000
	s_lshr_b32 s1, s0, 5
	s_or_b32 s0, s1, s0
	s_lshl_b32 s19, s0, 1
	s_add_i32 s21, s12, s19
	s_mov_b32 m0, s21
	s_and_b32 s1, s15, 0xffff
	s_mov_b32 s0, s14
	s_waitcnt lgkmcnt(0)
	s_barrier
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v32, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s21, 0x1080
	v_accvgpr_read_b32 v0, a120
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v33, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s21, 0x2100
	s_add_i32 s19, s20, s19
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s21, 0x3180
	s_mul_i32 s18, s18, 0x8400
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v5, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s21, 0x4200
	v_accvgpr_write_b32 a115, v32
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s21, 0x5280
	v_accvgpr_read_b32 v0, a119
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s21, 0x6300
	v_accvgpr_read_b32 v0, a132
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s21, 0x7380
	v_accvgpr_read_b32 v0, a133
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_mov_b32 m0, s19
	s_and_b32 s1, s5, 0xffff
	s_mov_b32 s0, s4
	v_accvgpr_read_b32 v0, a130
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s19, 0x1080
	v_accvgpr_read_b32 v0, a131
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s19, 0x2100
	v_accvgpr_read_b32 v0, a134
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s19, 0x3180
	v_accvgpr_read_b32 v0, a127
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s19, 0x4200
	v_accvgpr_read_b32 v0, a128
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s19, 0x5280
	v_accvgpr_read_b32 v0, a129
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s19, 0x6300
	v_accvgpr_read_b32 v0, a126
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v255, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_add_i32 m0, s19, 0x7380
	v_add_u32_e32 v0, s18, v0
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	buffer_load_dwordx4 v254, s[0:3], 0 offen lds
        v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[26:29], v0
	v_accvgpr_write_b32 a117, v33
	ds_read_b128 v[30:33], v0 offset:256
	v_accvgpr_mov_b32 a145, a3
	v_accvgpr_mov_b32 a161, a7
	v_accvgpr_mov_b32 a149, a15
	v_accvgpr_mov_b32 a153, a19
	v_accvgpr_mov_b32 a144, a2
	v_accvgpr_mov_b32 a143, a1
	v_accvgpr_mov_b32 a142, a0
	v_accvgpr_mov_b32 a160, a6
	v_accvgpr_mov_b32 a159, a5
	v_accvgpr_mov_b32 a158, a4
	v_accvgpr_mov_b32 a148, a14
	v_accvgpr_mov_b32 a147, a13
	v_accvgpr_mov_b32 a146, a12
	v_accvgpr_mov_b32 a152, a18
	v_accvgpr_mov_b32 a151, a17
	v_accvgpr_mov_b32 a150, a16
	ds_read_b128 v[70:73], v0 offset:512
	ds_read_b128 a[36:39], v0 offset:768
	ds_read_b128 a[40:43], v0 offset:16896
	ds_read_b128 a[44:47], v0 offset:17152
	ds_read_b128 a[48:51], v0 offset:17408
	ds_read_b128 a[52:55], v0 offset:17664
	ds_read_b128 a[0:3], v0 offset:17728
	ds_read_b128 a[4:7], v0 offset:64
	ds_read_b128 a[8:11], v0 offset:320
	ds_read_b128 a[12:15], v0 offset:576
	ds_read_b128 a[16:19], v0 offset:832
	ds_read_b128 a[20:23], v0 offset:16960
	ds_read_b128 a[24:27], v0 offset:17216
	ds_read_b128 a[28:31], v0 offset:17472
	v_accvgpr_read_b32 v0, a121
	v_add_u32_e32 v0, s18, v0
	ds_read_b128 a[56:59], v0
	ds_read_b128 v[22:25], v0 offset:256
	ds_read_b128 a[60:63], v0 offset:512
	ds_read_b128 a[64:67], v0 offset:768
	ds_read_b128 a[68:71], v0 offset:16896
	ds_read_b128 a[72:75], v0 offset:17152
	ds_read_b128 a[76:79], v0 offset:17408
	ds_read_b128 a[80:83], v0 offset:17664
	ds_read_b128 a[32:35], v0 offset:17728
	ds_read_b128 a[84:87], v0 offset:64
	ds_read_b128 a[88:91], v0 offset:320
	ds_read_b128 a[92:95], v0 offset:576
	ds_read_b128 a[96:99], v0 offset:832
	ds_read_b128 a[100:103], v0 offset:16960
	ds_read_b128 a[104:107], v0 offset:17216
	ds_read_b128 a[108:111], v0 offset:17472
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[202:205], a[36:39], v[22:25], v[202:205]
	s_add_i32 s11, s11, 1
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	v_mfma_f32_16x16x32_f16 v[226:229], a[40:43], v[22:25], v[226:229]
	s_add_u32 s14, s14, 0x80
	s_addc_u32 s15, s15, 0
	s_cmp_lg_u32 s17, s11
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[174:177], a[36:39], a[60:63], v[174:177]
	v_mfma_f32_16x16x32_f16 v[182:185], a[40:43], a[60:63], v[182:185]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[130:133], a[36:39], a[64:67], v[130:133]
	v_mfma_f32_16x16x32_f16 v[134:137], a[40:43], a[64:67], v[134:137]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[114:117], a[36:39], a[68:71], v[114:117]
	v_mfma_f32_16x16x32_f16 v[86:89], a[40:43], a[68:71], v[86:89]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[98:101], a[36:39], a[72:75], v[98:101]
	v_mfma_f32_16x16x32_f16 v[106:109], a[40:43], a[72:75], v[106:109]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[110:113], a[36:39], a[76:79], v[110:113]
	v_mfma_f32_16x16x32_f16 v[122:125], a[40:43], a[76:79], v[122:125]
	v_mfma_f32_16x16x32_f16 v[18:21], a[40:43], a[56:59], v[18:21]
	v_mfma_f32_16x16x32_f16 a[138:141], a[36:39], a[56:59], a[138:141]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[54:57], a[36:39], a[80:83], v[54:57]
	v_mfma_f32_16x16x32_f16 a[36:39], a[40:43], a[80:83], a[158:161]
	v_accvgpr_mov_b32 a40, a146
	v_accvgpr_mov_b32 a41, a147
	v_accvgpr_mov_b32 a42, a148
	v_accvgpr_mov_b32 a43, a149
	v_mfma_f32_16x16x32_f16 v[230:233], a[44:47], v[22:25], v[230:233]
	v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], a[60:63], v[218:221]
	v_mfma_f32_16x16x32_f16 v[142:145], a[44:47], a[64:67], v[142:145]
	v_mfma_f32_16x16x32_f16 v[118:121], a[44:47], a[68:71], v[118:121]
	v_mfma_f32_16x16x32_f16 v[126:129], a[44:47], a[72:75], v[126:129]
	v_mfma_f32_16x16x32_f16 v[46:49], a[44:47], a[76:79], v[46:49]
	v_mfma_f32_16x16x32_f16 v[42:45], a[44:47], a[56:59], v[42:45]
	v_mfma_f32_16x16x32_f16 a[40:43], a[44:47], a[80:83], a[40:43]
	v_accvgpr_mov_b32 a44, a150
	v_accvgpr_mov_b32 a45, a151
	v_accvgpr_mov_b32 a46, a152
	v_accvgpr_mov_b32 a47, a153
	v_mfma_f32_16x16x32_f16 a[154:157], v[30:33], a[56:59], a[154:157]
	v_mfma_f32_16x16x32_f16 v[234:237], a[48:51], v[22:25], v[234:237]
	v_mfma_f32_16x16x32_f16 v[214:217], a[48:51], a[60:63], v[214:217]
	v_mfma_f32_16x16x32_f16 v[94:97], a[48:51], a[64:67], v[94:97]
	v_mfma_f32_16x16x32_f16 v[170:173], a[48:51], a[68:71], v[170:173]
	v_mfma_f32_16x16x32_f16 v[138:141], a[48:51], a[72:75], v[138:141]
	v_mfma_f32_16x16x32_f16 v[62:65], a[48:51], a[76:79], v[62:65]
	v_mfma_f32_16x16x32_f16 v[10:13], a[48:51], a[56:59], v[10:13]
	v_mfma_f32_16x16x32_f16 a[44:47], a[48:51], a[80:83], a[44:47]
	v_accvgpr_mov_b32 a48, a142
	v_accvgpr_mov_b32 a49, a143
	v_accvgpr_mov_b32 a50, a144
	v_accvgpr_mov_b32 a51, a145
	v_mfma_f32_16x16x32_f16 v[242:245], v[70:73], a[56:59], v[242:245]
	v_mfma_f32_16x16x32_f16 v[198:201], v[70:73], v[22:25], v[198:201]
	v_mfma_f32_16x16x32_f16 v[238:241], a[52:55], v[22:25], v[238:241]
	v_mfma_f32_16x16x32_f16 v[250:253], v[26:29], a[60:63], v[250:253]
	v_mfma_f32_16x16x32_f16 v[186:189], v[30:33], a[60:63], v[186:189]
	v_mfma_f32_16x16x32_f16 v[166:169], v[70:73], a[60:63], v[166:169]
	v_mfma_f32_16x16x32_f16 v[146:149], a[52:55], a[60:63], v[146:149]
	v_mfma_f32_16x16x32_f16 v[246:249], v[26:29], a[64:67], v[246:249]
	v_mfma_f32_16x16x32_f16 v[190:193], v[30:33], a[64:67], v[190:193]
	v_mfma_f32_16x16x32_f16 v[158:161], v[70:73], a[64:67], v[158:161]
	v_mfma_f32_16x16x32_f16 v[194:197], a[52:55], a[64:67], v[194:197]
	v_mfma_f32_16x16x32_f16 v[222:225], v[26:29], a[68:71], v[222:225]
	v_mfma_f32_16x16x32_f16 v[78:81], v[30:33], a[68:71], v[78:81]
	v_mfma_f32_16x16x32_f16 v[150:153], v[70:73], a[68:71], v[150:153]
	v_mfma_f32_16x16x32_f16 v[178:181], a[52:55], a[68:71], v[178:181]
	v_mfma_f32_16x16x32_f16 v[210:213], v[26:29], a[72:75], v[210:213]
	v_mfma_f32_16x16x32_f16 v[50:53], v[30:33], a[72:75], v[50:53]
	v_mfma_f32_16x16x32_f16 v[102:105], v[70:73], a[72:75], v[102:105]
	v_mfma_f32_16x16x32_f16 v[82:85], a[52:55], a[72:75], v[82:85]
	v_mfma_f32_16x16x32_f16 v[206:209], v[26:29], a[76:79], v[206:209]
	v_mfma_f32_16x16x32_f16 v[90:93], v[30:33], a[76:79], v[90:93]
	v_mfma_f32_16x16x32_f16 v[58:61], v[70:73], a[76:79], v[58:61]
	v_mfma_f32_16x16x32_f16 v[162:165], a[52:55], a[76:79], v[162:165]
	v_mfma_f32_16x16x32_f16 v[6:9], v[30:33], v[22:25], v[6:9]
	v_mfma_f32_16x16x32_f16 v[74:77], v[26:29], v[22:25], v[74:77]
	v_mfma_f32_16x16x32_f16 v[14:17], a[52:55], a[56:59], v[14:17]
	v_mfma_f32_16x16x32_f16 a[122:125], v[26:29], a[56:59], a[122:125]
	v_mfma_f32_16x16x32_f16 v[154:157], v[26:29], a[80:83], v[154:157]
	v_mfma_f32_16x16x32_f16 v[38:41], v[30:33], a[80:83], v[38:41]
	v_accvgpr_read_b32 v33, a117
	v_accvgpr_read_b32 v32, a115
	v_mfma_f32_16x16x32_f16 v[34:37], v[70:73], a[80:83], v[34:37]
	v_mfma_f32_16x16x32_f16 a[48:51], a[52:55], a[80:83], a[48:51]
	v_accvgpr_mov_b32 a52, a154
	v_accvgpr_mov_b32 a53, a155
	v_accvgpr_mov_b32 a54, a156
	v_accvgpr_mov_b32 a55, a157
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 a[122:125], a[4:7], a[84:87], a[122:125]
	v_mfma_f32_16x16x32_f16 a[154:157], a[8:11], a[84:87], a[52:55]
	v_mfma_f32_16x16x32_f16 v[242:245], a[12:15], a[84:87], v[242:245]
	v_mfma_f32_16x16x32_f16 a[138:141], a[16:19], a[84:87], a[138:141]
	v_mfma_f32_16x16x32_f16 v[18:21], a[20:23], a[84:87], v[18:21]
	v_mfma_f32_16x16x32_f16 v[42:45], a[24:27], a[84:87], v[42:45]
	v_mfma_f32_16x16x32_f16 v[10:13], a[28:31], a[84:87], v[10:13]
	v_mfma_f32_16x16x32_f16 v[14:17], a[0:3], a[84:87], v[14:17]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[74:77], a[4:7], a[88:91], v[74:77]
	v_mfma_f32_16x16x32_f16 v[6:9], a[8:11], a[88:91], v[6:9]
	v_mfma_f32_16x16x32_f16 v[198:201], a[12:15], a[88:91], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[16:19], a[88:91], v[202:205]
	v_mfma_f32_16x16x32_f16 v[226:229], a[20:23], a[88:91], v[226:229]
	v_mfma_f32_16x16x32_f16 v[230:233], a[24:27], a[88:91], v[230:233]
	v_mfma_f32_16x16x32_f16 v[234:237], a[28:31], a[88:91], v[234:237]
	v_mfma_f32_16x16x32_f16 v[238:241], a[0:3], a[88:91], v[238:241]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[250:253], a[4:7], a[92:95], v[250:253]
	v_mfma_f32_16x16x32_f16 v[186:189], a[8:11], a[92:95], v[186:189]
	v_mfma_f32_16x16x32_f16 v[166:169], a[12:15], a[92:95], v[166:169]
	v_mfma_f32_16x16x32_f16 v[174:177], a[16:19], a[92:95], v[174:177]
	v_mfma_f32_16x16x32_f16 v[182:185], a[20:23], a[92:95], v[182:185]
	v_mfma_f32_16x16x32_f16 v[218:221], a[24:27], a[92:95], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[28:31], a[92:95], v[214:217]
	v_mfma_f32_16x16x32_f16 v[146:149], a[0:3], a[92:95], v[146:149]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[246:249], a[4:7], a[96:99], v[246:249]
	v_mfma_f32_16x16x32_f16 v[190:193], a[8:11], a[96:99], v[190:193]
	v_mfma_f32_16x16x32_f16 v[158:161], a[12:15], a[96:99], v[158:161]
	v_mfma_f32_16x16x32_f16 v[130:133], a[16:19], a[96:99], v[130:133]
	v_mfma_f32_16x16x32_f16 v[134:137], a[20:23], a[96:99], v[134:137]
	v_mfma_f32_16x16x32_f16 v[142:145], a[24:27], a[96:99], v[142:145]
	v_mfma_f32_16x16x32_f16 v[94:97], a[28:31], a[96:99], v[94:97]
	v_mfma_f32_16x16x32_f16 v[194:197], a[0:3], a[96:99], v[194:197]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[222:225], a[4:7], a[100:103], v[222:225]
	v_mfma_f32_16x16x32_f16 v[78:81], a[8:11], a[100:103], v[78:81]
	v_mfma_f32_16x16x32_f16 v[150:153], a[12:15], a[100:103], v[150:153]
	v_mfma_f32_16x16x32_f16 v[114:117], a[16:19], a[100:103], v[114:117]
	v_mfma_f32_16x16x32_f16 v[86:89], a[20:23], a[100:103], v[86:89]
	v_mfma_f32_16x16x32_f16 v[118:121], a[24:27], a[100:103], v[118:121]
	v_mfma_f32_16x16x32_f16 v[170:173], a[28:31], a[100:103], v[170:173]
	v_mfma_f32_16x16x32_f16 v[178:181], a[0:3], a[100:103], v[178:181]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[210:213], a[4:7], a[104:107], v[210:213]
	v_mfma_f32_16x16x32_f16 v[50:53], a[8:11], a[104:107], v[50:53]
	v_mfma_f32_16x16x32_f16 v[102:105], a[12:15], a[104:107], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[16:19], a[104:107], v[98:101]
	v_mfma_f32_16x16x32_f16 v[106:109], a[20:23], a[104:107], v[106:109]
	v_mfma_f32_16x16x32_f16 v[126:129], a[24:27], a[104:107], v[126:129]
	v_mfma_f32_16x16x32_f16 v[138:141], a[28:31], a[104:107], v[138:141]
	v_mfma_f32_16x16x32_f16 v[82:85], a[0:3], a[104:107], v[82:85]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[206:209], a[4:7], a[108:111], v[206:209]
	v_mfma_f32_16x16x32_f16 v[90:93], a[8:11], a[108:111], v[90:93]
	v_mfma_f32_16x16x32_f16 v[58:61], a[12:15], a[108:111], v[58:61]
	v_mfma_f32_16x16x32_f16 v[110:113], a[16:19], a[108:111], v[110:113]
	v_mfma_f32_16x16x32_f16 v[122:125], a[20:23], a[108:111], v[122:125]
	v_mfma_f32_16x16x32_f16 v[46:49], a[24:27], a[108:111], v[46:49]
	v_mfma_f32_16x16x32_f16 v[62:65], a[28:31], a[108:111], v[62:65]
	v_mfma_f32_16x16x32_f16 v[162:165], a[0:3], a[108:111], v[162:165]
	v_mfma_f32_16x16x32_f16 v[154:157], a[4:7], a[32:35], v[154:157]
	v_mfma_f32_16x16x32_f16 v[38:41], a[8:11], a[32:35], v[38:41]
	v_mfma_f32_16x16x32_f16 v[34:37], a[12:15], a[32:35], v[34:37]
	v_mfma_f32_16x16x32_f16 v[54:57], a[16:19], a[32:35], v[54:57]
	v_mfma_f32_16x16x32_f16 a[4:7], a[20:23], a[32:35], a[36:39]
	v_mfma_f32_16x16x32_f16 a[12:15], a[24:27], a[32:35], a[40:43]
	v_mfma_f32_16x16x32_f16 a[16:19], a[28:31], a[32:35], a[44:47]
	v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[32:35], a[48:51]
	s_cbranch_scc1 .LBB0_5
; %bb.6:                                ; %Flow
	v_accvgpr_mov_b32 a8, a154
	s_nop 5
	v_accvgpr_mov_b32 a137, a3
	v_accvgpr_mov_b32 a118, a138
	v_accvgpr_read_b32 v66, a122
	v_accvgpr_mov_b32 a9, a155
	v_accvgpr_mov_b32 a10, a156
	v_accvgpr_mov_b32 a11, a157
	v_accvgpr_mov_b32 a136, a2
	v_accvgpr_mov_b32 a135, a1
	v_accvgpr_mov_b32 a134, a0
	v_accvgpr_mov_b32 a119, a139
	v_accvgpr_mov_b32 a120, a140
	v_accvgpr_mov_b32 a121, a141
	v_mov_b32_e32 v0, s16
	v_accvgpr_read_b32 v22, a112
	v_accvgpr_read_b32 v3, a113
	v_accvgpr_read_b32 v23, a114
	v_accvgpr_read_b32 v67, a123
	v_accvgpr_read_b32 v68, a124
	v_accvgpr_read_b32 v69, a125
.LBB0_7:                                ; %Flow265
	s_lshr_b32 s1, s10, 31
	s_add_i32 s1, s10, s1
	s_and_b32 s1, s1, -2
	s_sub_i32 s1, s10, s1
	s_mul_i32 s1, s1, 0x8400
	s_add_i32 s1, s1, 0
	v_add_u32_e32 v0, s1, v0
	v_and_b32_e32 v4, 15, v22
	v_add_u32_e32 v0, 0x107e0, v0
	v_add_u32_e32 v0, v0, v3
	v_lshlrev_b32_e32 v1, 10, v4
	v_lshlrev_b32_e32 v2, 5, v4
	v_accvgpr_write_b32 a117, v4
	v_add3_u32 v4, v0, v1, v2
	v_add_u32_e32 v0, s1, v1
	v_accvgpr_mov_b32 a133, a19
	v_accvgpr_mov_b32 a129, a15
	v_accvgpr_mov_b32 a125, a7
	v_add3_u32 v0, v0, v23, v2
	v_accvgpr_mov_b32 a132, a18
	v_accvgpr_mov_b32 a131, a17
	v_accvgpr_mov_b32 a130, a16
	v_accvgpr_mov_b32 a128, a14
	v_accvgpr_mov_b32 a127, a13
	v_accvgpr_mov_b32 a126, a12
	v_accvgpr_mov_b32 a124, a6
	v_accvgpr_mov_b32 a123, a5
	v_accvgpr_mov_b32 a122, a4
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
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
	ds_read_b128 a[0:3], v0 offset:17728
	ds_read_b128 a[4:7], v4
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[0:3], a[4:7], a[12:15], v[66:69]
	ds_read_b128 a[72:75], v4 offset:256
	ds_read_b128 a[76:79], v4 offset:320
	ds_read_b128 a[80:83], v4 offset:512
	ds_read_b128 a[96:99], v4 offset:16896
	ds_read_b128 a[100:103], v4 offset:16960
	ds_read_b128 a[104:107], v4 offset:17152
	ds_read_b128 a[84:87], v4 offset:576
	ds_read_b128 a[88:91], v4 offset:768
	ds_read_b128 a[92:95], v4 offset:832
	v_mfma_f32_16x16x32_f16 v[22:25], a[4:7], a[36:39], v[246:249]
	s_lshr_b32 s0, s24, 6
	s_lshl_b32 s0, s0, 3
	v_mfma_f32_16x16x32_f16 v[26:29], a[4:7], a[68:71], v[154:157]
	v_mfma_f32_16x16x32_f16 v[30:33], a[4:7], a[60:63], v[206:209]
	v_mfma_f32_16x16x32_f16 v[70:73], a[4:7], a[44:47], v[222:225]
	v_mfma_f32_16x16x32_f16 v[154:157], a[4:7], a[28:31], v[250:253]
	v_mfma_f32_16x16x32_f16 v[206:209], a[4:7], a[52:55], v[210:213]
	v_mfma_f32_16x16x32_f16 v[74:77], a[4:7], a[20:23], v[74:77]
	ds_read_b128 a[4:7], v4 offset:64
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[246:249], a[4:7], a[40:43], v[22:25]
	s_nop 2
	v_accvgpr_read_b32 v25, a11
	v_accvgpr_read_b32 v24, a10
	v_accvgpr_read_b32 v23, a9
	v_accvgpr_read_b32 v22, a8
	v_mfma_f32_16x16x32_f16 v[210:213], a[4:7], a[0:3], v[26:29]
	ds_read_b128 v[66:69], v4 offset:17664
	ds_read_b128 a[8:11], v4 offset:17728
	v_mfma_f32_16x16x32_f16 v[22:25], a[72:75], a[12:15], v[22:25]
	v_mfma_f32_16x16x32_f16 v[26:29], a[4:7], a[48:51], v[70:73]
	v_mfma_f32_16x16x32_f16 v[70:73], a[76:79], a[16:19], v[22:25]
	v_mfma_f32_16x16x32_f16 v[22:25], a[104:107], a[12:15], v[42:45]
	v_mfma_f32_16x16x32_f16 v[222:225], a[4:7], a[24:27], v[74:77]
	s_nop 5
	v_cvt_pk_f16_f32 v254, v70, v71
	v_cvt_pk_f16_f32 v255, v72, v73
	v_mfma_f32_16x16x32_f16 v[250:253], a[4:7], a[16:19], v[0:3]
	v_mfma_f32_16x16x32_f16 v[74:77], a[4:7], a[64:67], v[30:33]
	v_mfma_f32_16x16x32_f16 v[30:33], a[4:7], a[32:35], v[154:157]
	s_nop 5
	v_cvt_pk_f16_f32 v250, v250, v251
	v_cvt_pk_f16_f32 v251, v252, v253
	v_mfma_f32_16x16x32_f16 v[0:3], a[4:7], a[56:59], v[206:209]
	ds_read_b128 a[4:7], v4 offset:17216
	ds_read_b128 a[108:111], v4 offset:17408
	ds_read_b128 a[112:115], v4 offset:17472
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[42:45], a[4:7], a[16:19], v[22:25]
	v_mfma_f32_16x16x32_f16 v[22:25], v[66:69], a[36:39], v[194:197]
	v_mfma_f32_16x16x32_f16 v[194:197], v[66:69], a[20:23], v[238:241]
	v_mfma_f32_16x16x32_f16 v[238:241], a[8:11], a[40:43], v[22:25]
	v_mfma_f32_16x16x32_f16 v[22:25], v[66:69], a[60:63], v[162:165]
	v_mfma_f32_16x16x32_f16 v[162:165], a[8:11], a[64:67], v[22:25]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[22:25], a[108:111], a[20:23], v[234:237]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[234:237], a[112:115], a[24:27], v[22:25]
	v_mfma_f32_16x16x32_f16 v[22:25], a[108:111], a[52:55], v[138:141]
	v_mfma_f32_16x16x32_f16 v[138:141], a[108:111], a[28:31], v[214:217]
	v_mfma_f32_16x16x32_f16 v[214:217], a[112:115], a[32:35], v[138:141]
	v_mfma_f32_16x16x32_f16 v[138:141], a[108:111], a[44:47], v[170:173]
	v_mfma_f32_16x16x32_f16 v[170:173], a[112:115], a[48:51], v[138:141]
	v_mfma_f32_16x16x32_f16 v[138:141], a[104:107], a[28:31], v[218:221]
	v_mfma_f32_16x16x32_f16 v[218:221], a[4:7], a[32:35], v[138:141]
	s_nop 5
	v_cvt_pk_f16_f32 v70, v170, v171
	v_cvt_pk_f16_f32 v71, v172, v173
	v_mfma_f32_16x16x32_f16 v[138:141], a[104:107], a[20:23], v[230:233]
	v_mfma_f32_16x16x32_f16 v[18:21], a[96:99], a[12:15], v[18:21]
	v_mfma_f32_16x16x32_f16 v[54:57], a[88:91], a[68:71], v[54:57]
	v_mfma_f32_16x16x32_f16 v[230:233], a[4:7], a[24:27], v[138:141]
	v_mfma_f32_16x16x32_f16 v[138:141], a[100:103], a[16:19], v[18:21]
	v_mfma_f32_16x16x32_f16 v[18:21], a[96:99], a[52:55], v[106:109]
	v_mfma_f32_16x16x32_f16 v[106:109], a[96:99], a[20:23], v[226:229]
	v_mfma_f32_16x16x32_f16 v[226:229], a[92:95], a[0:3], v[54:57]
	v_mfma_f32_16x16x32_f16 v[54:57], a[88:91], a[52:55], v[98:101]
	v_mfma_f32_16x16x32_f16 v[98:101], a[80:83], a[52:55], v[102:105]
	v_mfma_f32_16x16x32_f16 v[102:105], a[80:83], a[44:47], v[150:153]
	v_mfma_f32_16x16x32_f16 v[150:153], a[80:83], a[12:15], v[242:245]
	v_mfma_f32_16x16x32_f16 v[242:245], a[84:87], a[16:19], v[150:153]
	v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[36:39], v[190:193]
	v_mfma_f32_16x16x32_f16 v[190:193], a[108:111], a[12:15], v[10:13]
	s_nop 5
	v_cvt_pk_f16_f32 v252, v242, v243
	v_cvt_pk_f16_f32 v242, v138, v139
	v_cvt_pk_f16_f32 v243, v140, v141
	v_mfma_f32_16x16x32_f16 v[10:13], a[104:107], a[36:39], v[142:145]
	v_cvt_pk_f16_f32 v140, v42, v43
	v_cvt_pk_f16_f32 v253, v244, v245
	v_cvt_pk_f16_f32 v141, v44, v45
	v_mfma_f32_16x16x32_f16 v[14:17], v[66:69], a[12:15], v[14:17]
	v_cvt_pk_f16_f32 v44, v210, v211
	v_cvt_pk_f16_f32 v45, v212, v213
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[20:23], v[6:9]
	v_mfma_f32_16x16x32_f16 v[142:145], a[4:7], a[40:43], v[10:13]
	s_nop 2
	v_accvgpr_read_b32 v10, a126
	v_accvgpr_read_b32 v11, a127
	v_accvgpr_read_b32 v12, a128
	v_accvgpr_read_b32 v13, a129
	v_mfma_f32_16x16x32_f16 v[154:157], a[8:11], a[16:19], v[14:17]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[10:13], a[104:107], a[68:71], v[10:13]
	s_nop 0
	v_accvgpr_read_b32 v14, a130
	v_accvgpr_read_b32 v15, a131
	v_accvgpr_read_b32 v16, a132
	v_mfma_f32_16x16x32_f16 v[206:209], a[112:115], a[16:19], v[190:193]
	v_accvgpr_read_b32 v17, a133
	v_mfma_f32_16x16x32_f16 v[62:65], a[108:111], a[60:63], v[62:65]
	s_nop 0
	v_accvgpr_read_b32 v193, a125
	v_accvgpr_read_b32 v192, a124
	v_accvgpr_read_b32 v191, a123
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[24:27], v[4:7]
	s_nop 0
	v_cvt_pk_f16_f32 v138, v206, v207
	v_cvt_pk_f16_f32 v206, v154, v155
	v_cvt_pk_f16_f32 v155, v224, v225
	v_mfma_f32_16x16x32_f16 a[134:137], v[66:69], a[68:71], a[134:137]
	v_cvt_pk_f16_f32 v139, v208, v209
	v_cvt_pk_f16_f32 v207, v156, v157
	v_cvt_pk_f16_f32 v154, v222, v223
	v_mfma_f32_16x16x32_f16 v[18:21], a[100:103], a[56:59], v[18:21]
	v_cvt_pk_f16_f32 v224, v4, v5
	v_cvt_pk_f16_f32 v225, v6, v7
	v_cvt_pk_f16_f32 v156, v234, v235
	v_mfma_f32_16x16x32_f16 v[54:57], a[92:95], a[56:59], v[54:57]
	v_cvt_pk_f16_f32 v157, v236, v237
	v_accvgpr_read_b32 v190, a122
	v_mfma_f32_16x16x32_f16 v[102:105], a[84:87], a[48:51], v[102:105]
	v_mfma_f32_16x16x32_f16 v[4:7], a[112:115], a[64:67], v[62:65]
	v_mfma_f32_16x16x32_f16 v[10:13], a[4:7], a[0:3], v[10:13]
	s_nop 5
	v_cvt_pk_f16_f32 v102, v102, v103
	v_cvt_pk_f16_f32 v103, v104, v105
	v_cvt_pk_f16_f32 v105, v2, v3
	v_cvt_pk_f16_f32 v2, v54, v55
	v_cvt_pk_f16_f32 v55, v20, v21
	v_cvt_pk_f16_f32 v20, v74, v75
	v_cvt_pk_f16_f32 v75, v6, v7
	v_cvt_pk_f16_f32 v104, v0, v1
	v_cvt_pk_f16_f32 v6, v10, v11
	v_cvt_pk_f16_f32 v7, v12, v13
	v_accvgpr_read_b32 v10, a134
	v_accvgpr_read_b32 v11, a135
	v_accvgpr_read_b32 v12, a136
	v_accvgpr_read_b32 v13, a137
	v_mfma_f32_16x16x32_f16 v[86:89], a[96:99], a[44:47], v[86:89]
	v_cvt_pk_f16_f32 v74, v4, v5
	v_cvt_pk_f16_f32 v3, v56, v57
	v_cvt_pk_f16_f32 v54, v18, v19
	v_mfma_f32_16x16x32_f16 v[10:13], a[8:11], a[0:3], v[10:13]
	v_cvt_pk_f16_f32 v21, v76, v77
	v_cvt_pk_f16_f32 v76, v162, v163
	v_cvt_pk_f16_f32 v77, v164, v165
	v_mfma_f32_16x16x32_f16 v[202:205], a[88:91], a[20:23], v[202:205]
	v_mfma_f32_16x16x32_f16 v[78:81], a[72:75], a[44:47], v[78:81]
	s_nop 2
	v_cvt_pk_f16_f32 v0, v10, v11
	v_accvgpr_read_b32 v10, a117
	v_and_or_b32 v10, s0, 16, v10
	v_mfma_f32_16x16x32_f16 v[50:53], a[72:75], a[52:55], v[50:53]
	s_mul_i32 s0, s9, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_mfma_f32_16x16x32_f16 v[38:41], a[72:75], a[68:71], v[38:41]
	s_add_u32 s2, s6, s0
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s9, s8, 31
	v_mfma_f32_16x16x32_f16 v[34:37], a[80:83], a[68:71], v[34:37]
	s_lshl_b64 s[0:1], s[8:9], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	v_mfma_f32_16x16x32_f16 v[22:25], a[112:115], a[56:59], v[22:25]
	v_mul_lo_u32 v10, v10, s13
	s_lshl_b32 s2, s13, 5
	v_accvgpr_read_b32 v11, a116
	v_mfma_f32_16x16x32_f16 v[86:89], a[100:103], a[48:51], v[86:89]
	v_lshrrev_b32_e32 v11, 2, v11
	v_and_b32_e32 v11, 28, v11
	v_cvt_pk_f16_f32 v1, v12, v13
	v_mfma_f32_16x16x32_f16 v[202:205], a[92:95], a[24:27], v[202:205]
	v_or_b32_e32 v12, 32, v11
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	v_mfma_f32_16x16x32_f16 v[78:81], a[76:79], a[48:51], v[78:81]
	v_or_b32_e32 v13, 64, v11
	v_mfma_f32_16x16x32_f16 v[50:53], a[76:79], a[56:59], v[50:53]
	s_nop 1
	v_cvt_pk_f16_f32 v209, v204, v205
	v_cvt_pk_f16_f32 v208, v202, v203
	v_mfma_f32_16x16x32_f16 v[38:41], a[76:79], a[0:3], v[38:41]
	s_nop 0
	v_cvt_pk_f16_f32 v204, v78, v79
	v_cvt_pk_f16_f32 v79, v88, v89
	v_cvt_pk_f16_f32 v205, v80, v81
	v_mfma_f32_16x16x32_f16 v[34:37], a[84:87], a[0:3], v[34:37]
	v_cvt_pk_f16_f32 v88, v50, v51
	v_cvt_pk_f16_f32 v51, v24, v25
	v_add_u32_e32 v25, s2, v10
	v_mfma_f32_16x16x32_f16 v[46:49], a[104:107], a[60:63], v[46:49]
	v_cvt_pk_f16_f32 v42, v38, v39
	v_cvt_pk_f16_f32 v43, v40, v41
	v_or_b32_e32 v24, 0xe0, v11
	v_mfma_f32_16x16x32_f16 v[178:181], v[66:69], a[44:47], v[178:181]
	v_cvt_pk_f16_f32 v39, v36, v37
	v_add_u32_e32 v36, s2, v25
	v_add_u32_e32 v37, s2, v36
	v_mfma_f32_16x16x32_f16 v[82:85], v[66:69], a[52:55], v[82:85]
	v_add_u32_e32 v40, s2, v37
	v_add_u32_e32 v41, s2, v40
	v_cvt_pk_f16_f32 v78, v86, v87
	v_mfma_f32_16x16x32_f16 v[146:149], v[66:69], a[28:31], v[146:149]
	v_accvgpr_read_b32 v66, a118
	v_accvgpr_read_b32 v67, a119
	v_accvgpr_read_b32 v68, a120
	v_mfma_f32_16x16x32_f16 v[14:17], a[108:111], a[68:71], v[14:17]
	v_accvgpr_read_b32 v69, a121
	v_cvt_pk_f16_f32 v89, v52, v53
	v_cvt_pk_f16_f32 v50, v22, v23
	v_mfma_f32_16x16x32_f16 v[66:69], a[88:91], a[12:15], v[66:69]
	v_cvt_pk_f16_f32 v38, v34, v35
	v_cvt_pk_f16_f32 v34, v226, v227
	v_cvt_pk_f16_f32 v35, v228, v229
	v_mfma_f32_16x16x32_f16 v[46:49], a[4:7], a[64:67], v[46:49]
	v_mfma_f32_16x16x32_f16 v[14:17], a[112:115], a[0:3], v[14:17]
	v_mfma_f32_16x16x32_f16 v[66:69], a[92:95], a[16:19], v[66:69]
	s_nop 5
	v_cvt_pk_f16_f32 v8, v46, v47
	v_add_u32_e32 v46, s2, v41
	v_cvt_pk_f16_f32 v9, v48, v49
	v_add_u32_e32 v47, s2, v46
	s_mov_b32 s2, 0x7ffffffe
	v_add_lshl_u32 v48, v11, v10, 1
	v_mfma_f32_16x16x32_f16 v[198:201], a[80:83], a[20:23], v[198:201]
	buffer_store_dwordx2 v[250:251], v48, s[0:3], 0 offen
	v_add_lshl_u32 v48, v12, v10, 1
	v_cvt_pk_f16_f32 v4, v14, v15
	v_or_b32_e32 v14, 0x60, v11
	buffer_store_dwordx2 v[254:255], v48, s[0:3], 0 offen
	v_add_lshl_u32 v48, v13, v10, 1
	v_cvt_pk_f16_f32 v244, v66, v67
	v_cvt_pk_f16_f32 v245, v68, v69
	v_or_b32_e32 v15, 0x80, v11
	buffer_store_dwordx2 v[252:253], v48, s[0:3], 0 offen
	v_add_lshl_u32 v48, v14, v10, 1
	v_cvt_pk_f16_f32 v5, v16, v17
	v_or_b32_e32 v16, 0xa0, v11
	buffer_store_dwordx2 v[244:245], v48, s[0:3], 0 offen
	v_add_lshl_u32 v48, v15, v10, 1
	v_mfma_f32_16x16x32_f16 v[198:201], a[84:87], a[24:27], v[198:201]
	v_or_b32_e32 v17, 0xc0, v11
	buffer_store_dwordx2 v[242:243], v48, s[0:3], 0 offen
	v_add_lshl_u32 v48, v16, v10, 1
	buffer_store_dwordx2 v[140:141], v48, s[0:3], 0 offen
	v_add_lshl_u32 v48, v17, v10, 1
	v_add_lshl_u32 v10, v24, v10, 1
	v_mfma_f32_16x16x32_f16 v[106:109], a[100:103], a[24:27], v[106:109]
	buffer_store_dwordx2 v[138:139], v48, s[0:3], 0 offen
	buffer_store_dwordx2 v[206:207], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v25, v11, 1
	v_mfma_f32_16x16x32_f16 v[186:189], a[72:75], a[28:31], v[186:189]
	buffer_store_dwordx2 v[154:155], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v12, v25, 1
	v_cvt_pk_f16_f32 v222, v198, v199
	v_mfma_f32_16x16x32_f16 v[166:169], a[80:83], a[28:31], v[166:169]
	v_cvt_pk_f16_f32 v223, v200, v201
	buffer_store_dwordx2 v[224:225], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v13, v25, 1
	v_mfma_f32_16x16x32_f16 v[194:197], a[8:11], a[24:27], v[194:197]
	buffer_store_dwordx2 v[222:223], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v14, v25, 1
	v_cvt_pk_f16_f32 v198, v106, v107
	v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[28:31], v[174:177]
	v_cvt_pk_f16_f32 v199, v108, v109
	buffer_store_dwordx2 v[208:209], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v15, v25, 1
	v_mfma_f32_16x16x32_f16 v[182:185], a[96:99], a[28:31], v[182:185]
	v_cvt_pk_f16_f32 v200, v230, v231
	v_cvt_pk_f16_f32 v201, v232, v233
	buffer_store_dwordx2 v[198:199], v10, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[186:189], a[76:79], a[32:35], v[186:189]
	v_add_lshl_u32 v10, v16, v25, 1
	buffer_store_dwordx2 v[200:201], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v17, v25, 1
	v_mfma_f32_16x16x32_f16 v[166:169], a[84:87], a[32:35], v[166:169]
	v_cvt_pk_f16_f32 v194, v194, v195
	v_cvt_pk_f16_f32 v195, v196, v197
	buffer_store_dwordx2 v[156:157], v10, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[32:35], v[174:177]
	v_add_lshl_u32 v10, v24, v25, 1
	v_cvt_pk_f16_f32 v230, v30, v31
	v_cvt_pk_f16_f32 v231, v32, v33
	v_mfma_f32_16x16x32_f16 v[182:185], a[100:103], a[32:35], v[182:185]
	buffer_store_dwordx2 v[194:195], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v36, v11, 1
	v_cvt_pk_f16_f32 v202, v186, v187
	v_cvt_pk_f16_f32 v203, v188, v189
	buffer_store_dwordx2 v[230:231], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v12, v36, 1
	v_mfma_f32_16x16x32_f16 v[158:161], a[80:83], a[36:39], v[158:161]
	v_cvt_pk_f16_f32 v196, v166, v167
	v_cvt_pk_f16_f32 v197, v168, v169
	buffer_store_dwordx2 v[202:203], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v13, v36, 1
	v_mfma_f32_16x16x32_f16 v[130:133], a[88:91], a[36:39], v[130:133]
	v_cvt_pk_f16_f32 v174, v174, v175
	v_cvt_pk_f16_f32 v175, v176, v177
	buffer_store_dwordx2 v[196:197], v10, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[146:149], a[8:11], a[32:35], v[146:149]
	v_add_lshl_u32 v10, v14, v36, 1
	v_cvt_pk_f16_f32 v168, v182, v183
	v_cvt_pk_f16_f32 v169, v184, v185
	v_mfma_f32_16x16x32_f16 v[134:137], a[96:99], a[36:39], v[134:137]
	buffer_store_dwordx2 v[174:175], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v15, v36, 1
	v_cvt_pk_f16_f32 v108, v218, v219
	v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], a[40:43], v[150:153]
	v_cvt_pk_f16_f32 v109, v220, v221
	buffer_store_dwordx2 v[168:169], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v16, v36, 1
	v_mfma_f32_16x16x32_f16 v[158:161], a[84:87], a[40:43], v[158:161]
	v_cvt_pk_f16_f32 v106, v214, v215
	v_cvt_pk_f16_f32 v107, v216, v217
	buffer_store_dwordx2 v[108:109], v10, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[94:97], a[108:111], a[36:39], v[94:97]
	v_add_lshl_u32 v10, v17, v36, 1
	v_cvt_pk_f16_f32 v186, v146, v147
	v_cvt_pk_f16_f32 v187, v148, v149
	v_mfma_f32_16x16x32_f16 v[130:133], a[92:95], a[40:43], v[130:133]
	buffer_store_dwordx2 v[106:107], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v24, v36, 1
	v_cvt_pk_f16_f32 v184, v246, v247
	v_mfma_f32_16x16x32_f16 v[134:137], a[100:103], a[40:43], v[134:137]
	v_cvt_pk_f16_f32 v185, v248, v249
	buffer_store_dwordx2 v[186:187], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v37, v11, 1
	v_cvt_pk_f16_f32 v182, v150, v151
	v_cvt_pk_f16_f32 v183, v152, v153
	buffer_store_dwordx2 v[184:185], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v12, v37, 1
	v_mfma_f32_16x16x32_f16 v[94:97], a[112:115], a[40:43], v[94:97]
	v_cvt_pk_f16_f32 v176, v158, v159
	v_cvt_pk_f16_f32 v177, v160, v161
	buffer_store_dwordx2 v[182:183], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v13, v37, 1
	v_mfma_f32_16x16x32_f16 v[114:117], a[88:91], a[44:47], v[114:117]
	v_cvt_pk_f16_f32 v68, v130, v131
	v_cvt_pk_f16_f32 v69, v132, v133
	buffer_store_dwordx2 v[176:177], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v14, v37, 1
	v_cvt_pk_f16_f32 v66, v134, v135
	v_cvt_pk_f16_f32 v67, v136, v137
	buffer_store_dwordx2 v[68:69], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v15, v37, 1
	v_mfma_f32_16x16x32_f16 v[118:121], a[104:107], a[44:47], v[118:121]
	v_cvt_pk_f16_f32 v152, v142, v143
	v_cvt_pk_f16_f32 v153, v144, v145
	buffer_store_dwordx2 v[66:67], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v16, v37, 1
	v_cvt_pk_f16_f32 v150, v94, v95
	v_cvt_pk_f16_f32 v151, v96, v97
	buffer_store_dwordx2 v[152:153], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v17, v37, 1
	v_mfma_f32_16x16x32_f16 v[114:117], a[92:95], a[48:51], v[114:117]
	v_cvt_pk_f16_f32 v136, v238, v239
	v_cvt_pk_f16_f32 v137, v240, v241
	buffer_store_dwordx2 v[150:151], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v24, v37, 1
	v_cvt_pk_f16_f32 v134, v26, v27
	v_cvt_pk_f16_f32 v135, v28, v29
	buffer_store_dwordx2 v[136:137], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v40, v11, 1
	v_mfma_f32_16x16x32_f16 v[118:121], a[4:7], a[48:51], v[118:121]
	buffer_store_dwordx2 v[134:135], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v12, v40, 1
	buffer_store_dwordx2 v[204:205], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v13, v40, 1
	v_mfma_f32_16x16x32_f16 v[178:181], a[8:11], a[48:51], v[178:181]
	v_cvt_pk_f16_f32 v80, v114, v115
	v_cvt_pk_f16_f32 v81, v116, v117
	buffer_store_dwordx2 v[102:103], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v14, v40, 1
	buffer_store_dwordx2 v[80:81], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v15, v40, 1
	v_mfma_f32_16x16x32_f16 v[126:129], a[104:107], a[52:55], v[126:129]
	v_cvt_pk_f16_f32 v72, v118, v119
	v_cvt_pk_f16_f32 v73, v120, v121
	buffer_store_dwordx2 v[78:79], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v16, v40, 1
	v_mfma_f32_16x16x32_f16 v[98:101], a[84:87], a[56:59], v[98:101]
	buffer_store_dwordx2 v[72:73], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v17, v40, 1
	v_cvt_pk_f16_f32 v114, v178, v179
	v_cvt_pk_f16_f32 v115, v180, v181
	buffer_store_dwordx2 v[70:71], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v24, v40, 1
	buffer_store_dwordx2 v[114:115], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v41, v11, 1
	v_mfma_f32_16x16x32_f16 v[126:129], a[4:7], a[56:59], v[126:129]
	buffer_store_dwordx2 v[104:105], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v12, v41, 1
	v_cvt_pk_f16_f32 v86, v98, v99
	v_mfma_f32_16x16x32_f16 v[90:93], a[72:75], a[60:63], v[90:93]
	v_cvt_pk_f16_f32 v87, v100, v101
	buffer_store_dwordx2 v[88:89], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v13, v41, 1
	v_mfma_f32_16x16x32_f16 v[58:61], a[80:83], a[60:63], v[58:61]
	buffer_store_dwordx2 v[86:87], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v14, v41, 1
	buffer_store_dwordx2 v[2:3], v10, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[110:113], a[88:91], a[60:63], v[110:113]
	v_add_lshl_u32 v2, v15, v41, 1
	v_cvt_pk_f16_f32 v52, v126, v127
	v_cvt_pk_f16_f32 v53, v128, v129
	v_mfma_f32_16x16x32_f16 v[82:85], a[8:11], a[56:59], v[82:85]
	buffer_store_dwordx2 v[54:55], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v16, v41, 1
	buffer_store_dwordx2 v[52:53], v2, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[122:125], a[96:99], a[60:63], v[122:125]
	v_add_lshl_u32 v2, v17, v41, 1
	buffer_store_dwordx2 v[50:51], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v24, v41, 1
	v_mfma_f32_16x16x32_f16 v[90:93], a[76:79], a[64:67], v[90:93]
	v_cvt_pk_f16_f32 v22, v82, v83
	v_cvt_pk_f16_f32 v23, v84, v85
	buffer_store_dwordx2 v[22:23], v2, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[58:61], a[84:87], a[64:67], v[58:61]
	v_add_lshl_u32 v2, v46, v11, 1
	buffer_store_dwordx2 v[20:21], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v12, v46, 1
	v_mfma_f32_16x16x32_f16 v[110:113], a[92:95], a[64:67], v[110:113]
	v_cvt_pk_f16_f32 v32, v90, v91
	v_cvt_pk_f16_f32 v33, v92, v93
	buffer_store_dwordx2 v[32:33], v2, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[122:125], a[100:103], a[64:67], v[122:125]
	v_cvt_pk_f16_f32 v30, v58, v59
	v_cvt_pk_f16_f32 v31, v60, v61
	v_add_lshl_u32 v2, v13, v46, 1
	buffer_store_dwordx2 v[30:31], v2, s[0:3], 0 offen
	v_cvt_pk_f16_f32 v28, v110, v111
	v_cvt_pk_f16_f32 v29, v112, v113
	v_add_lshl_u32 v2, v14, v46, 1
	v_mfma_f32_16x16x32_f16 v[190:193], a[96:99], a[68:71], v[190:193]
	v_cvt_pk_f16_f32 v26, v122, v123
	v_cvt_pk_f16_f32 v27, v124, v125
	buffer_store_dwordx2 v[28:29], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v15, v46, 1
	buffer_store_dwordx2 v[26:27], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v16, v46, 1
	buffer_store_dwordx2 v[8:9], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v17, v46, 1
	buffer_store_dwordx2 v[74:75], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v24, v46, 1
	v_mfma_f32_16x16x32_f16 v[190:193], a[100:103], a[0:3], v[190:193]
	buffer_store_dwordx2 v[76:77], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v47, v11, 1
	buffer_store_dwordx2 v[44:45], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v12, v47, 1
	buffer_store_dwordx2 v[42:43], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v13, v47, 1
	buffer_store_dwordx2 v[38:39], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v14, v47, 1
	v_cvt_pk_f16_f32 v18, v190, v191
	v_cvt_pk_f16_f32 v19, v192, v193
	buffer_store_dwordx2 v[34:35], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v15, v47, 1
	buffer_store_dwordx2 v[18:19], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v16, v47, 1
	buffer_store_dwordx2 v[6:7], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v17, v47, 1
	buffer_store_dwordx2 v[4:5], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v24, v47, 1
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v4
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
		.amdhsa_next_free_vgpr 418
		.amdhsa_next_free_sgpr 28
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
	.size	v4, .Lfunc_end0-v4
	.cfi_endproc
                                        ; -- End function
	.set v4.num_vgpr, 256
	.set v4.num_agpr, 162
	.set v4.numbered_sgpr, 28
	.set v4.num_named_barrier, 0
	.set v4.private_seg_size, 0
	.set v4.uses_vcc, 0
	.set v4.uses_flat_scratch, 0
	.set v4.has_dyn_sized_stack, 0
	.set v4.has_recursion, 0
	.set v4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9576
; TotalNumSgprs: 34
; NumVgprs: 256
; NumAgprs: 162
; TotalNumVgprs: 418
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 52
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 418
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
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton/study_matmul/gluon/matmul_kernels" ; string offset=24
.Linfo_string3:
	.asciz	"v4"                            ; string offset=86
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     162
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
    .name:           v4
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         v4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     418
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
