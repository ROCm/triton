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
; %bb.13:
	.file	1 "/var/lib/jenkins/AMD-triton/python/perf-kernels/tools/tune_gemm" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_load_dword s14, s[0:1], 0x30
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.14:
.LBB0_0:
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s8, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_add_i32 s1, s9, 0xff
	s_ashr_i32 s14, s1, 31
	s_lshr_b32 s14, s14, 24
	s_add_i32 s1, s1, s14
	s_ashr_i32 s1, s1, 8
	s_lshl_b32 s16, s1, 2
	s_abs_i32 s17, s16
	v_cvt_f32_u32_e32 v1, s17
	s_ashr_i32 s14, s15, 31
	s_lshr_b32 s14, s14, 29
	s_add_i32 s14, s15, s14
	v_rcp_iflag_f32_e32 v1, v1
	s_ashr_i32 s14, s14, 3
	s_sub_i32 s18, 0, s17
	s_mulk_i32 s15, 0x98
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mulk_i32 s14, 0xfb41
	s_add_i32 s14, s14, s15
	s_abs_i32 s15, s14
	v_readfirstlane_b32 s19, v1
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s15, s19
	s_mul_i32 s19, s18, s17
	s_xor_b32 s1, s14, s1
	s_sub_i32 s15, s15, s19
	s_ashr_i32 s0, s0, 8
	s_ashr_i32 s1, s1, 31
	s_add_i32 s19, s18, 1
	s_sub_i32 s20, s15, s17
	s_cmp_ge_u32 s15, s17
	s_cselect_b32 s18, s19, s18
	s_cselect_b32 s15, s20, s15
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s15, s17
	s_cselect_b32 s15, s19, s18
	s_xor_b32 s15, s15, s1
	s_sub_i32 s1, s15, s1
	s_lshl_b32 s15, s1, 2
	s_sub_i32 s0, s0, s15
	s_min_i32 s0, s0, 4
	s_abs_i32 s17, s0
	v_cvt_f32_u32_e32 v1, s17
	s_sub_i32 s18, 0, s17
	s_mul_i32 s1, s1, s16
	s_sub_i32 s1, s14, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s16, s1
	s_xor_b32 s14, s1, s0
	s_ashr_i32 s14, s14, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshlrev_b32_e32 v3, 3, v0
	v_and_b32_e32 v34, 56, v3
	v_bfrev_b32_e32 v30, 1
	v_readfirstlane_b32 s19, v1
	s_mul_i32 s18, s18, s19
	s_mul_hi_u32 s18, s19, s18
	s_add_i32 s19, s19, s18
	s_mul_hi_u32 s18, s16, s19
	s_mul_i32 s19, s18, s17
	s_sub_i32 s16, s16, s19
	s_add_i32 s19, s18, 1
	s_sub_i32 s20, s16, s17
	s_cmp_ge_u32 s16, s17
	s_cselect_b32 s18, s19, s18
	s_cselect_b32 s16, s20, s16
	s_add_i32 s19, s18, 1
	s_cmp_ge_u32 s16, s17
	s_cselect_b32 s16, s19, s18
	s_xor_b32 s16, s16, s14
	s_sub_i32 s14, s16, s14
	s_mul_i32 s0, s14, s0
	s_sub_i32 s23, s1, s0
	s_add_i32 s23, s23, s15
	s_lshl_b32 s15, s23, 8
	s_mul_i32 s0, s15, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s14, s14, 8
	s_lshl_b32 s22, s11, 7
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s16, s2, s0
	s_addc_u32 s17, s3, s1
	s_add_i32 s24, s10, 63
	v_lshrrev_b32_e32 v1, 3, v0
	s_cmp_gt_i32 s24, 63
	v_mad_u64_u32 v[10:11], s[0:1], s11, v1, v[34:35]
	s_cselect_b64 vcc, -1, 0
	s_cmp_lt_i32 s24, 64
	s_cselect_b64 s[20:21], -1, 0
	s_and_b32 s0, s11, 0x3fff
	v_or_b32_e32 v18, 64, v1
	s_bitset1_b32 s0, 14
	v_or_b32_e32 v28, 0xc0, v1
	v_mul_lo_u32 v2, s11, v18
	s_and_b32 s1, s17, 0xffff
	s_lshl_b32 s10, s0, 16
	s_mul_i32 s0, s14, s12
	v_mul_lo_u32 v12, s11, v28
	v_lshlrev_b32_e32 v157, 1, v10
	s_or_b32 s17, s1, s10
	v_add_lshl_u32 v159, v2, v34, 1
	v_add_lshl_u32 v160, v10, s22, 1
	s_ashr_i32 s1, s0, 31
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	v_cndmask_b32_e32 v11, v30, v157, vcc
	v_cndmask_b32_e32 v13, v30, v159, vcc
	v_cndmask_b32_e32 v19, v30, v160, vcc
	v_add_lshl_u32 v161, v12, v34, 1
	s_lshl_b32 s25, s12, 7
	s_lshl_b64 s[0:1], s[0:1], 1
	buffer_load_dwordx4 v[2:5], v11, s[16:19], 0 offen
	buffer_load_dwordx4 v[6:9], v13, s[16:19], 0 offen
	v_cndmask_b32_e32 v20, v30, v161, vcc
	buffer_load_dwordx4 v[10:13], v19, s[16:19], 0 offen
	buffer_load_dwordx4 v[14:17], v20, s[16:19], 0 offen
	s_add_u32 s16, s4, s0
	s_addc_u32 s17, s5, s1
	s_and_b32 s22, s12, 0x3fff
	v_mad_u64_u32 v[26:27], s[26:27], s12, v1, v[34:35]
	s_bitset1_b32 s22, 14
	v_mul_lo_u32 v22, s12, v18
	s_and_b32 s17, s17, 0xffff
	s_lshl_b32 s22, s22, 16
	v_lshlrev_b32_e32 v162, 1, v26
	v_mul_lo_u32 v31, s12, v28
	v_add_lshl_u32 v164, v26, s25, 1
	s_or_b32 s17, s17, s22
	v_cndmask_b32_e32 v18, v30, v162, vcc
	v_add_lshl_u32 v163, v22, v34, 1
	v_cndmask_b32_e32 v26, v30, v164, vcc
	v_add_lshl_u32 v165, v31, v34, 1
	buffer_load_dwordx4 v[18:21], v18, s[16:19], 0 offen
	v_cndmask_b32_e32 v22, v30, v163, vcc
	buffer_load_dwordx4 v[26:29], v26, s[16:19], 0 offen
	v_cndmask_b32_e32 v30, v30, v165, vcc
	buffer_load_dwordx4 v[30:33], v30, s[16:19], 0 offen
	v_lshrrev_b32_e32 v35, 1, v0
	buffer_load_dwordx4 v[22:25], v22, s[16:19], 0 offen
	v_and_or_b32 v36, v35, 4, v34
	v_and_b32_e32 v37, 56, v35
	v_xor_b32_e32 v36, v37, v36
	v_lshlrev_b32_e32 v1, 6, v1
	v_or_b32_e32 v37, v36, v1
	v_lshlrev_b32_e32 v37, 1, v37
	v_add_u32_e32 v166, 0, v37
	s_add_i32 s12, 0, 0x8000
	v_add_u32_e32 v172, s12, v37
	s_cmpk_gt_i32 s24, 0x7f
	v_lshrrev_b32_e32 v158, 2, v0
	s_waitcnt vmcnt(7)
	ds_write_b64 v166, v[2:3]
	v_or_b32_e32 v2, 4, v34
	v_and_b32_e32 v3, 60, v35
	v_xor_b32_e32 v2, v3, v2
	v_or_b32_e32 v34, 0x1000, v1
	v_or_b32_e32 v3, v2, v1
	v_or_b32_e32 v35, v36, v34
	v_lshlrev_b32_e32 v3, 1, v3
	v_lshl_add_u32 v168, v35, 1, 0
	v_or_b32_e32 v1, 0x3000, v1
	v_add_u32_e32 v167, 0, v3
	s_waitcnt vmcnt(6)
	ds_write_b64 v168, v[6:7]
	v_or_b32_e32 v6, v2, v34
	s_waitcnt vmcnt(5)
	ds_write_b64 v166, v[10:11] offset:16384
	ds_write2st64_b64 v167, v[4:5], v[12:13] offset1:32
	v_or_b32_e32 v4, v36, v1
	v_or_b32_e32 v1, v2, v1
	v_lshl_add_u32 v169, v6, 1, 0
	v_lshl_add_u32 v170, v4, 1, 0
	v_lshl_add_u32 v171, v1, 1, 0
	v_add_u32_e32 v173, s12, v3
	s_waitcnt vmcnt(4)
	ds_write_b64 v170, v[14:15]
	s_waitcnt vmcnt(3)
	ds_write_b64 v166, v[18:19] offset:32768
	ds_write_b64 v167, v[20:21] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b64 v168, v[22:23] offset:32768
	ds_write2st64_b64 v169, v[8:9], v[24:25] offset1:64
	ds_write_b64 v172, v[26:27] offset:16384
	ds_write_b64 v173, v[28:29] offset:16384
	ds_write_b64 v170, v[30:31] offset:32768
	ds_write2st64_b64 v171, v[16:17], v[32:33] offset1:64
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	v_lshrrev_b32_e32 v174, 2, v0
	s_mov_b64 s[16:17], 0
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b64 s[16:17], -1
                                        ; implicit-def: $vgpr174
.LBB0_3:                                ; %Flow188
	v_lshrrev_b32_e32 v1, 6, v0
	v_mov_b32_e32 v21, 0
	s_andn2_b64 vcc, exec, s[16:17]
	v_lshlrev_b32_e32 v154, 2, v1
	v_lshlrev_b32_e32 v155, 4, v1
	v_mov_b32_e32 v20, v21
	v_mov_b32_e32 v19, v21
	v_mov_b32_e32 v18, v21
	v_mov_b32_e32 v29, v21
	v_mov_b32_e32 v28, v21
	v_mov_b32_e32 v27, v21
	v_mov_b32_e32 v26, v21
	v_mov_b32_e32 v5, v21
	v_mov_b32_e32 v4, v21
	v_mov_b32_e32 v3, v21
	v_mov_b32_e32 v2, v21
	v_mov_b32_e32 v9, v21
	v_mov_b32_e32 v8, v21
	v_mov_b32_e32 v7, v21
	v_mov_b32_e32 v6, v21
	v_mov_b32_e32 v17, v21
	v_mov_b32_e32 v16, v21
	v_mov_b32_e32 v15, v21
	v_mov_b32_e32 v14, v21
	v_mov_b32_e32 v13, v21
	v_mov_b32_e32 v12, v21
	v_mov_b32_e32 v11, v21
	v_mov_b32_e32 v10, v21
	v_mov_b32_e32 v25, v21
	v_mov_b32_e32 v24, v21
	v_mov_b32_e32 v23, v21
	v_mov_b32_e32 v22, v21
	v_mov_b32_e32 v33, v21
	v_mov_b32_e32 v32, v21
	v_mov_b32_e32 v31, v21
	v_mov_b32_e32 v30, v21
	v_mov_b32_e32 v37, v21
	v_mov_b32_e32 v36, v21
	v_mov_b32_e32 v35, v21
	v_mov_b32_e32 v34, v21
	v_mov_b32_e32 v41, v21
	v_mov_b32_e32 v40, v21
	v_mov_b32_e32 v39, v21
	v_mov_b32_e32 v38, v21
	v_mov_b32_e32 v45, v21
	v_mov_b32_e32 v44, v21
	v_mov_b32_e32 v43, v21
	v_mov_b32_e32 v42, v21
	v_mov_b32_e32 v49, v21
	v_mov_b32_e32 v48, v21
	v_mov_b32_e32 v47, v21
	v_mov_b32_e32 v46, v21
	v_mov_b32_e32 v53, v21
	v_mov_b32_e32 v52, v21
	v_mov_b32_e32 v51, v21
	v_mov_b32_e32 v50, v21
	v_mov_b32_e32 v57, v21
	v_mov_b32_e32 v56, v21
	v_mov_b32_e32 v55, v21
	v_mov_b32_e32 v54, v21
	v_mov_b32_e32 v61, v21
	v_mov_b32_e32 v60, v21
	v_mov_b32_e32 v59, v21
	v_mov_b32_e32 v58, v21
	v_mov_b32_e32 v65, v21
	v_mov_b32_e32 v64, v21
	v_mov_b32_e32 v63, v21
	v_mov_b32_e32 v62, v21
	v_mov_b32_e32 v69, v21
	v_mov_b32_e32 v68, v21
	v_mov_b32_e32 v67, v21
	v_mov_b32_e32 v66, v21
	v_mov_b32_e32 v73, v21
	v_mov_b32_e32 v72, v21
	v_mov_b32_e32 v71, v21
	v_mov_b32_e32 v70, v21
	v_mov_b32_e32 v77, v21
	v_mov_b32_e32 v76, v21
	v_mov_b32_e32 v75, v21
	v_mov_b32_e32 v74, v21
	v_mov_b32_e32 v81, v21
	v_mov_b32_e32 v80, v21
	v_mov_b32_e32 v79, v21
	v_mov_b32_e32 v78, v21
	v_mov_b32_e32 v85, v21
	v_mov_b32_e32 v84, v21
	v_mov_b32_e32 v83, v21
	v_mov_b32_e32 v82, v21
	v_mov_b32_e32 v89, v21
	v_mov_b32_e32 v88, v21
	v_mov_b32_e32 v87, v21
	v_mov_b32_e32 v86, v21
	v_mov_b32_e32 v93, v21
	v_mov_b32_e32 v92, v21
	v_mov_b32_e32 v91, v21
	v_mov_b32_e32 v90, v21
	v_mov_b32_e32 v97, v21
	v_mov_b32_e32 v96, v21
	v_mov_b32_e32 v95, v21
	v_mov_b32_e32 v94, v21
	v_mov_b32_e32 v101, v21
	v_mov_b32_e32 v100, v21
	v_mov_b32_e32 v99, v21
	v_mov_b32_e32 v98, v21
	v_mov_b32_e32 v105, v21
	v_mov_b32_e32 v104, v21
	v_mov_b32_e32 v103, v21
	v_mov_b32_e32 v102, v21
	v_mov_b32_e32 v109, v21
	v_mov_b32_e32 v108, v21
	v_mov_b32_e32 v107, v21
	v_mov_b32_e32 v106, v21
	v_mov_b32_e32 v113, v21
	v_mov_b32_e32 v112, v21
	v_mov_b32_e32 v111, v21
	v_mov_b32_e32 v110, v21
	v_mov_b32_e32 v117, v21
	v_mov_b32_e32 v116, v21
	v_mov_b32_e32 v115, v21
	v_mov_b32_e32 v114, v21
	v_mov_b32_e32 v121, v21
	v_mov_b32_e32 v120, v21
	v_mov_b32_e32 v119, v21
	v_mov_b32_e32 v118, v21
	v_mov_b32_e32 v125, v21
	v_mov_b32_e32 v124, v21
	v_mov_b32_e32 v123, v21
	v_mov_b32_e32 v122, v21
	v_mov_b32_e32 v129, v21
	v_mov_b32_e32 v128, v21
	v_mov_b32_e32 v127, v21
	v_mov_b32_e32 v126, v21
	v_and_b32_e32 v1, 15, v0
	v_lshlrev_b32_e32 v156, 2, v0
	s_cbranch_vccnz .LBB0_7
; %bb.4:                                ; %.lr.ph
	s_lshr_b32 s16, s24, 6
	v_and_b32_e32 v2, 15, v0
	v_and_b32_e32 v3, 12, v158
	s_add_u32 s0, s4, s0
	v_and_or_b32 v4, v154, 16, v2
	v_or_b32_e32 v5, 16, v3
	v_or_b32_e32 v6, 32, v3
	v_or_b32_e32 v7, 48, v3
	v_lshlrev_b32_e32 v8, 2, v2
	s_addc_u32 s1, s5, s1
	v_xor_b32_e32 v9, v3, v8
	v_xor_b32_e32 v10, v5, v8
	v_xor_b32_e32 v11, v6, v8
	v_xor_b32_e32 v8, v7, v8
	v_lshl_add_u32 v4, v4, 7, 0
	s_add_u32 s4, s0, 0x80
	s_mul_i32 s11, s11, s23
	v_lshl_add_u32 v174, v9, 1, v4
	v_lshl_add_u32 v175, v10, 1, v4
	v_lshl_add_u32 v176, v11, 1, v4
	v_lshl_add_u32 v177, v8, 1, v4
	v_and_b32_e32 v4, 60, v156
	s_addc_u32 s5, s1, 0
	s_lshl_b32 s0, s11, 8
	v_and_or_b32 v2, v155, 48, v2
	v_xor_b32_e32 v3, v4, v3
	s_ashr_i32 s1, s0, 31
	v_xor_b32_e32 v5, v4, v5
	v_lshlrev_b32_e32 v3, 1, v3
	v_lshlrev_b32_e32 v2, 7, v2
	s_lshl_b64 s[0:1], s[0:1], 1
	v_xor_b32_e32 v6, v4, v6
	v_add3_u32 v178, 0, v3, v2
	v_lshlrev_b32_e32 v3, 1, v5
	s_add_u32 s0, s0, s2
	v_xor_b32_e32 v4, v4, v7
	v_add3_u32 v179, 0, v3, v2
	v_lshlrev_b32_e32 v3, 1, v6
	s_addc_u32 s1, s1, s3
	v_add3_u32 v180, 0, v3, v2
	v_lshlrev_b32_e32 v3, 1, v4
	s_add_u32 s11, s0, 0x80
	v_mov_b32_e32 v126, 0
	v_add3_u32 v181, 0, v3, v2
	s_addc_u32 s12, s1, 0
	s_add_i32 s16, s16, -1
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_mov_b32_e32 v127, v126
	v_mov_b32_e32 v128, v126
	v_mov_b32_e32 v129, v126
	v_mov_b32_e32 v122, v126
	v_mov_b32_e32 v123, v126
	v_mov_b32_e32 v124, v126
	v_mov_b32_e32 v125, v126
	v_mov_b32_e32 v118, v126
	v_mov_b32_e32 v119, v126
	v_mov_b32_e32 v120, v126
	v_mov_b32_e32 v121, v126
	v_mov_b32_e32 v114, v126
	v_mov_b32_e32 v115, v126
	v_mov_b32_e32 v116, v126
	v_mov_b32_e32 v117, v126
	v_mov_b32_e32 v110, v126
	v_mov_b32_e32 v111, v126
	v_mov_b32_e32 v112, v126
	v_mov_b32_e32 v113, v126
	v_mov_b32_e32 v106, v126
	v_mov_b32_e32 v107, v126
	v_mov_b32_e32 v108, v126
	v_mov_b32_e32 v109, v126
	v_mov_b32_e32 v102, v126
	v_mov_b32_e32 v103, v126
	v_mov_b32_e32 v104, v126
	v_mov_b32_e32 v105, v126
	v_mov_b32_e32 v98, v126
	v_mov_b32_e32 v99, v126
	v_mov_b32_e32 v100, v126
	v_mov_b32_e32 v101, v126
	v_mov_b32_e32 v94, v126
	v_mov_b32_e32 v95, v126
	v_mov_b32_e32 v96, v126
	v_mov_b32_e32 v97, v126
	v_mov_b32_e32 v90, v126
	v_mov_b32_e32 v91, v126
	v_mov_b32_e32 v92, v126
	v_mov_b32_e32 v93, v126
	v_mov_b32_e32 v86, v126
	v_mov_b32_e32 v87, v126
	v_mov_b32_e32 v88, v126
	v_mov_b32_e32 v89, v126
	v_mov_b32_e32 v82, v126
	v_mov_b32_e32 v83, v126
	v_mov_b32_e32 v84, v126
	v_mov_b32_e32 v85, v126
	v_mov_b32_e32 v78, v126
	v_mov_b32_e32 v79, v126
	v_mov_b32_e32 v80, v126
	v_mov_b32_e32 v81, v126
	v_mov_b32_e32 v74, v126
	v_mov_b32_e32 v75, v126
	v_mov_b32_e32 v76, v126
	v_mov_b32_e32 v77, v126
	v_mov_b32_e32 v70, v126
	v_mov_b32_e32 v71, v126
	v_mov_b32_e32 v72, v126
	v_mov_b32_e32 v73, v126
	v_mov_b32_e32 v66, v126
	v_mov_b32_e32 v67, v126
	v_mov_b32_e32 v68, v126
	v_mov_b32_e32 v69, v126
	v_mov_b32_e32 v62, v126
	v_mov_b32_e32 v63, v126
	v_mov_b32_e32 v64, v126
	v_mov_b32_e32 v65, v126
	v_mov_b32_e32 v58, v126
	v_mov_b32_e32 v59, v126
	v_mov_b32_e32 v60, v126
	v_mov_b32_e32 v61, v126
	v_mov_b32_e32 v54, v126
	v_mov_b32_e32 v55, v126
	v_mov_b32_e32 v56, v126
	v_mov_b32_e32 v57, v126
	v_mov_b32_e32 v50, v126
	v_mov_b32_e32 v51, v126
	v_mov_b32_e32 v52, v126
	v_mov_b32_e32 v53, v126
	v_mov_b32_e32 v46, v126
	v_mov_b32_e32 v47, v126
	v_mov_b32_e32 v48, v126
	v_mov_b32_e32 v49, v126
	v_mov_b32_e32 v42, v126
	v_mov_b32_e32 v43, v126
	v_mov_b32_e32 v44, v126
	v_mov_b32_e32 v45, v126
	v_mov_b32_e32 v38, v126
	v_mov_b32_e32 v39, v126
	v_mov_b32_e32 v40, v126
	v_mov_b32_e32 v41, v126
	v_mov_b32_e32 v34, v126
	v_mov_b32_e32 v35, v126
	v_mov_b32_e32 v36, v126
	v_mov_b32_e32 v37, v126
	v_mov_b32_e32 v30, v126
	v_mov_b32_e32 v31, v126
	v_mov_b32_e32 v32, v126
	v_mov_b32_e32 v33, v126
	v_mov_b32_e32 v22, v126
	v_mov_b32_e32 v23, v126
	v_mov_b32_e32 v24, v126
	v_mov_b32_e32 v25, v126
	v_mov_b32_e32 v10, v126
	v_mov_b32_e32 v11, v126
	v_mov_b32_e32 v12, v126
	v_mov_b32_e32 v13, v126
	v_mov_b32_e32 v14, v126
	v_mov_b32_e32 v15, v126
	v_mov_b32_e32 v16, v126
	v_mov_b32_e32 v17, v126
	v_mov_b32_e32 v6, v126
	v_mov_b32_e32 v7, v126
	v_mov_b32_e32 v8, v126
	v_mov_b32_e32 v9, v126
	v_mov_b32_e32 v2, v126
	v_mov_b32_e32 v3, v126
	v_mov_b32_e32 v4, v126
	v_mov_b32_e32 v5, v126
	v_mov_b32_e32 v26, v126
	v_mov_b32_e32 v27, v126
	v_mov_b32_e32 v28, v126
	v_mov_b32_e32 v29, v126
	v_mov_b32_e32 v18, v126
	v_mov_b32_e32 v19, v126
	v_mov_b32_e32 v20, v126
	v_mov_b32_e32 v21, v126
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s1, s12, 0xffff
	s_mov_b32 s0, s11
	s_or_b32 s1, s1, s10
	buffer_load_dwordx4 v[138:141], v157, s[0:3], 0 offen
	buffer_load_dwordx4 v[130:133], v159, s[0:3], 0 offen
	buffer_load_dwordx4 v[142:145], v160, s[0:3], 0 offen
	buffer_load_dwordx4 v[134:137], v161, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[146:149], v178 offset0:64 offset1:80
	ds_read2st64_b64 v[182:185], v174 offset1:8
	ds_read2st64_b64 v[150:153], v178 offset0:96 offset1:112
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[126:129], v[146:147], v[182:183], v[126:129]
	s_and_b32 s0, s5, 0xffff
	s_or_b32 s1, s0, s22
	s_mov_b32 s0, s4
	v_mfma_f32_16x16x16_f16 v[122:125], v[148:149], v[182:183], v[122:125]
	buffer_load_dwordx4 v[186:189], v163, s[0:3], 0 offen
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[118:121], v[150:151], v[182:183], v[118:121]
	s_add_u32 s11, s11, 0x80
	s_addc_u32 s12, s12, 0
	s_add_i32 s16, s16, -1
	v_mfma_f32_16x16x16_f16 v[114:117], v[152:153], v[182:183], v[114:117]
	s_cmp_lg_u32 s16, 0
	v_mfma_f32_16x16x16_f16 v[110:113], v[146:147], v[184:185], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[148:149], v[184:185], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[150:151], v[184:185], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[152:153], v[184:185], v[98:101]
	ds_read2st64_b64 v[182:185], v174 offset0:16 offset1:24
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[94:97], v[146:147], v[182:183], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[148:149], v[182:183], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[150:151], v[182:183], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[152:153], v[182:183], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[146:147], v[184:185], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[148:149], v[184:185], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[150:151], v[184:185], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[152:153], v[184:185], v[66:69]
	ds_read2st64_b64 v[182:185], v174 offset0:32 offset1:40
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[62:65], v[146:147], v[182:183], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[148:149], v[182:183], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[150:151], v[182:183], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[152:153], v[182:183], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[146:147], v[184:185], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[148:149], v[184:185], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[150:151], v[184:185], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[152:153], v[184:185], v[34:37]
	ds_read2st64_b64 v[182:185], v174 offset0:48 offset1:56
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[30:33], v[146:147], v[182:183], v[30:33]
	v_mfma_f32_16x16x16_f16 v[22:25], v[148:149], v[182:183], v[22:25]
	v_mfma_f32_16x16x16_f16 v[6:9], v[146:147], v[184:185], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[148:149], v[184:185], v[2:5]
	ds_read2st64_b64 v[146:149], v179 offset0:64 offset1:80
	v_mfma_f32_16x16x16_f16 v[10:13], v[150:151], v[182:183], v[10:13]
	v_mfma_f32_16x16x16_f16 v[14:17], v[152:153], v[182:183], v[14:17]
	v_mfma_f32_16x16x16_f16 v[26:29], v[150:151], v[184:185], v[26:29]
	v_mfma_f32_16x16x16_f16 v[18:21], v[152:153], v[184:185], v[18:21]
	ds_read2st64_b64 v[182:185], v179 offset0:96 offset1:112
	ds_read2st64_b64 v[150:153], v175 offset1:8
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[126:129], v[146:147], v[150:151], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[148:149], v[150:151], v[122:125]
	v_mfma_f32_16x16x16_f16 v[118:121], v[182:183], v[150:151], v[118:121]
	v_mfma_f32_16x16x16_f16 v[114:117], v[184:185], v[150:151], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[146:147], v[152:153], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[148:149], v[152:153], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[182:183], v[152:153], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[184:185], v[152:153], v[98:101]
	ds_read2st64_b64 v[150:153], v175 offset0:16 offset1:24
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[94:97], v[146:147], v[150:151], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[148:149], v[150:151], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[182:183], v[150:151], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[184:185], v[150:151], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[146:147], v[152:153], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[148:149], v[152:153], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[182:183], v[152:153], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[184:185], v[152:153], v[66:69]
	ds_read2st64_b64 v[150:153], v175 offset0:32 offset1:40
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[62:65], v[146:147], v[150:151], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[148:149], v[150:151], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[182:183], v[150:151], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[184:185], v[150:151], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[146:147], v[152:153], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[148:149], v[152:153], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[182:183], v[152:153], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[184:185], v[152:153], v[34:37]
	ds_read2st64_b64 v[150:153], v175 offset0:48 offset1:56
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[30:33], v[146:147], v[150:151], v[30:33]
	v_mfma_f32_16x16x16_f16 v[22:25], v[148:149], v[150:151], v[22:25]
	v_mfma_f32_16x16x16_f16 v[6:9], v[146:147], v[152:153], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[148:149], v[152:153], v[2:5]
	ds_read2st64_b64 v[146:149], v180 offset0:64 offset1:80
	v_mfma_f32_16x16x16_f16 v[10:13], v[182:183], v[150:151], v[10:13]
	v_mfma_f32_16x16x16_f16 v[14:17], v[184:185], v[150:151], v[14:17]
	v_mfma_f32_16x16x16_f16 v[26:29], v[182:183], v[152:153], v[26:29]
	v_mfma_f32_16x16x16_f16 v[18:21], v[184:185], v[152:153], v[18:21]
	ds_read2st64_b64 v[182:185], v180 offset0:96 offset1:112
	ds_read2st64_b64 v[150:153], v176 offset1:8
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[126:129], v[146:147], v[150:151], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[148:149], v[150:151], v[122:125]
	v_mfma_f32_16x16x16_f16 v[118:121], v[182:183], v[150:151], v[118:121]
	v_mfma_f32_16x16x16_f16 v[114:117], v[184:185], v[150:151], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[146:147], v[152:153], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[148:149], v[152:153], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[182:183], v[152:153], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[184:185], v[152:153], v[98:101]
	ds_read2st64_b64 v[150:153], v176 offset0:16 offset1:24
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[94:97], v[146:147], v[150:151], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[148:149], v[150:151], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[182:183], v[150:151], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[184:185], v[150:151], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[146:147], v[152:153], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[148:149], v[152:153], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[182:183], v[152:153], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[184:185], v[152:153], v[66:69]
	ds_read2st64_b64 v[150:153], v176 offset0:32 offset1:40
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[62:65], v[146:147], v[150:151], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[148:149], v[150:151], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[182:183], v[150:151], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[184:185], v[150:151], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[146:147], v[152:153], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[148:149], v[152:153], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[182:183], v[152:153], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[184:185], v[152:153], v[34:37]
	ds_read2st64_b64 v[150:153], v176 offset0:48 offset1:56
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[30:33], v[146:147], v[150:151], v[30:33]
	v_mfma_f32_16x16x16_f16 v[22:25], v[148:149], v[150:151], v[22:25]
	v_mfma_f32_16x16x16_f16 v[6:9], v[146:147], v[152:153], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[148:149], v[152:153], v[2:5]
	ds_read2st64_b64 v[146:149], v181 offset0:64 offset1:80
	v_mfma_f32_16x16x16_f16 v[10:13], v[182:183], v[150:151], v[10:13]
	v_mfma_f32_16x16x16_f16 v[14:17], v[184:185], v[150:151], v[14:17]
	v_mfma_f32_16x16x16_f16 v[26:29], v[182:183], v[152:153], v[26:29]
	v_mfma_f32_16x16x16_f16 v[18:21], v[184:185], v[152:153], v[18:21]
	ds_read2st64_b64 v[182:185], v181 offset0:96 offset1:112
	ds_read2st64_b64 v[150:153], v177 offset1:8
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[126:129], v[146:147], v[150:151], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[148:149], v[150:151], v[122:125]
	v_mfma_f32_16x16x16_f16 v[118:121], v[182:183], v[150:151], v[118:121]
	v_mfma_f32_16x16x16_f16 v[114:117], v[184:185], v[150:151], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[146:147], v[152:153], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[148:149], v[152:153], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[182:183], v[152:153], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[184:185], v[152:153], v[98:101]
	ds_read2st64_b64 v[150:153], v177 offset0:16 offset1:24
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[94:97], v[146:147], v[150:151], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[148:149], v[150:151], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[182:183], v[150:151], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[184:185], v[150:151], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[146:147], v[152:153], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[148:149], v[152:153], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[182:183], v[152:153], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[184:185], v[152:153], v[66:69]
	ds_read2st64_b64 v[150:153], v177 offset0:32 offset1:40
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[62:65], v[146:147], v[150:151], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[148:149], v[150:151], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[182:183], v[150:151], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[184:185], v[150:151], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[146:147], v[152:153], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[148:149], v[152:153], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[182:183], v[152:153], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[184:185], v[152:153], v[34:37]
	ds_read2st64_b64 v[150:153], v177 offset0:48 offset1:56
	buffer_load_dwordx4 v[190:193], v165, s[0:3], 0 offen
	buffer_load_dwordx4 v[194:197], v162, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[30:33], v[146:147], v[150:151], v[30:33]
	v_mfma_f32_16x16x16_f16 v[22:25], v[148:149], v[150:151], v[22:25]
	v_mfma_f32_16x16x16_f16 v[6:9], v[146:147], v[152:153], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[148:149], v[152:153], v[2:5]
	buffer_load_dwordx4 v[146:149], v164, s[0:3], 0 offen
	s_barrier
	v_mfma_f32_16x16x16_f16 v[10:13], v[182:183], v[150:151], v[10:13]
	s_waitcnt vmcnt(7)
	ds_write_b64 v166, v[138:139]
	s_waitcnt vmcnt(5)
	ds_write_b64 v166, v[142:143] offset:16384
	ds_write2st64_b64 v167, v[140:141], v[144:145] offset1:32
	ds_write_b64 v168, v[130:131]
	s_waitcnt vmcnt(4)
	ds_write_b64 v170, v[134:135]
	s_waitcnt vmcnt(3)
	ds_write_b64 v168, v[186:187] offset:32768
	ds_write2st64_b64 v169, v[132:133], v[188:189] offset1:64
	s_waitcnt vmcnt(2)
	ds_write_b64 v170, v[190:191] offset:32768
	ds_write2st64_b64 v171, v[136:137], v[192:193] offset1:64
	s_waitcnt vmcnt(1)
	ds_write_b64 v166, v[194:195] offset:32768
	ds_write_b64 v167, v[196:197] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b64 v172, v[146:147] offset:16384
	ds_write_b64 v173, v[148:149] offset:16384
	v_mfma_f32_16x16x16_f16 v[14:17], v[184:185], v[150:151], v[14:17]
	v_mfma_f32_16x16x16_f16 v[26:29], v[182:183], v[152:153], v[26:29]
	v_mfma_f32_16x16x16_f16 v[18:21], v[184:185], v[152:153], v[18:21]
	s_cbranch_scc1 .LBB0_5
; %bb.6:                                ; %Flow186
	v_mov_b32_e32 v174, v158
.LBB0_7:                                ; %._crit_edge
	s_andn2_b64 vcc, exec, s[20:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_10
; %bb.8:                                ; %._crit_edge._crit_edge
	v_and_b32_e32 v130, 15, v0
	s_cbranch_execz .LBB0_11
; %bb.9:
	v_mov_b32_e32 v1, v130
	s_branch .LBB0_12
.LBB0_10:
                                        ; implicit-def: $vgpr130
.LBB0_11:
	v_and_b32_e32 v138, 12, v174
	v_or_b32_e32 v142, 48, v138
	v_and_b32_e32 v131, 60, v156
	v_and_or_b32 v130, v155, 48, v1
	v_xor_b32_e32 v132, v131, v142
	v_lshlrev_b32_e32 v132, 1, v132
	v_lshlrev_b32_e32 v130, 7, v130
	v_or_b32_e32 v143, 32, v138
	v_add3_u32 v158, 0, v132, v130
	v_xor_b32_e32 v132, v131, v143
	v_lshlrev_b32_e32 v132, 1, v132
	v_or_b32_e32 v144, 16, v138
	v_add3_u32 v155, 0, v132, v130
	v_xor_b32_e32 v132, v131, v144
	v_xor_b32_e32 v131, v131, v138
	v_and_or_b32 v139, v154, 16, v1
	v_lshlrev_b32_e32 v151, 2, v1
	v_lshlrev_b32_e32 v132, 1, v132
	v_lshlrev_b32_e32 v131, 1, v131
	v_lshl_add_u32 v152, v139, 7, 0
	v_xor_b32_e32 v138, v138, v151
	v_add3_u32 v150, 0, v132, v130
	v_add3_u32 v145, 0, v131, v130
	v_lshl_add_u32 v175, v138, 1, v152
	v_xor_b32_e32 v144, v144, v151
	v_xor_b32_e32 v143, v143, v151
	v_xor_b32_e32 v142, v142, v151
	ds_read2st64_b64 v[130:133], v150 offset0:64 offset1:80
	ds_read2st64_b64 v[134:137], v145 offset0:64 offset1:80
	ds_read2st64_b64 v[162:165], v175 offset1:8
	ds_read2st64_b64 v[138:141], v155 offset0:64 offset1:80
	v_lshl_add_u32 v180, v144, 1, v152
	v_lshl_add_u32 v181, v143, 1, v152
	v_lshl_add_u32 v182, v142, 1, v152
	ds_read2st64_b64 v[142:145], v145 offset0:96 offset1:112
	ds_read2st64_b64 v[150:153], v150 offset0:96 offset1:112
	ds_read2st64_b64 v[166:169], v180 offset1:8
	ds_read2st64_b64 v[154:157], v155 offset0:96 offset1:112
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x16_f16 v[126:129], v[134:135], v[162:163], v[126:129]
	ds_read2st64_b64 v[170:173], v181 offset1:8
	ds_read2st64_b64 v[146:149], v158 offset0:64 offset1:80
	ds_read2st64_b64 v[158:161], v158 offset0:96 offset1:112
	v_mfma_f32_16x16x16_f16 v[122:125], v[136:137], v[162:163], v[122:125]
	ds_read2st64_b64 v[176:179], v182 offset1:8
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x16_f16 v[118:121], v[142:143], v[162:163], v[118:121]
	v_mfma_f32_16x16x16_f16 v[114:117], v[144:145], v[162:163], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[134:135], v[164:165], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[136:137], v[164:165], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[142:143], v[164:165], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[144:145], v[164:165], v[98:101]
	ds_read2st64_b64 v[162:165], v175 offset0:16 offset1:24
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x16_f16 v[126:129], v[130:131], v[166:167], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[132:133], v[166:167], v[122:125]
	v_mfma_f32_16x16x16_f16 v[118:121], v[150:151], v[166:167], v[118:121]
	v_mfma_f32_16x16x16_f16 v[114:117], v[152:153], v[166:167], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[130:131], v[168:169], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[132:133], v[168:169], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[150:151], v[168:169], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[152:153], v[168:169], v[98:101]
	ds_read2st64_b64 v[166:169], v180 offset0:16 offset1:24
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[94:97], v[134:135], v[162:163], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[136:137], v[162:163], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[142:143], v[162:163], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[144:145], v[162:163], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[134:135], v[164:165], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[136:137], v[164:165], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[142:143], v[164:165], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[144:145], v[164:165], v[66:69]
	ds_read2st64_b64 v[162:165], v175 offset0:32 offset1:40
	v_mfma_f32_16x16x16_f16 v[126:129], v[138:139], v[170:171], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[140:141], v[170:171], v[122:125]
	v_mfma_f32_16x16x16_f16 v[118:121], v[154:155], v[170:171], v[118:121]
	v_mfma_f32_16x16x16_f16 v[114:117], v[156:157], v[170:171], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[138:139], v[172:173], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[140:141], v[172:173], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[154:155], v[172:173], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[156:157], v[172:173], v[98:101]
	ds_read2st64_b64 v[170:173], v181 offset0:16 offset1:24
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[94:97], v[130:131], v[166:167], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[132:133], v[166:167], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[150:151], v[166:167], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[152:153], v[166:167], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[130:131], v[168:169], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[132:133], v[168:169], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[150:151], v[168:169], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[152:153], v[168:169], v[66:69]
	ds_read2st64_b64 v[166:169], v180 offset0:32 offset1:40
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[62:65], v[134:135], v[162:163], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[136:137], v[162:163], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[142:143], v[162:163], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[144:145], v[162:163], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[134:135], v[164:165], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[136:137], v[164:165], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[142:143], v[164:165], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[144:145], v[164:165], v[34:37]
	ds_read2st64_b64 v[162:165], v175 offset0:48 offset1:56
	v_mfma_f32_16x16x16_f16 v[126:129], v[146:147], v[176:177], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[148:149], v[176:177], v[122:125]
	v_mfma_f32_16x16x16_f16 v[118:121], v[158:159], v[176:177], v[118:121]
	v_mfma_f32_16x16x16_f16 v[114:117], v[160:161], v[176:177], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[146:147], v[178:179], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[148:149], v[178:179], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[158:159], v[178:179], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[160:161], v[178:179], v[98:101]
	ds_read2st64_b64 v[176:179], v182 offset0:16 offset1:24
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 v[94:97], v[138:139], v[170:171], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[140:141], v[170:171], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[154:155], v[170:171], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[156:157], v[170:171], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[138:139], v[172:173], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[140:141], v[172:173], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[154:155], v[172:173], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[156:157], v[172:173], v[66:69]
	ds_read2st64_b64 v[170:173], v181 offset0:32 offset1:40
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 v[62:65], v[130:131], v[166:167], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[132:133], v[166:167], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[150:151], v[166:167], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[152:153], v[166:167], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[130:131], v[168:169], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[132:133], v[168:169], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[150:151], v[168:169], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[152:153], v[168:169], v[34:37]
	ds_read2st64_b64 v[166:169], v180 offset0:48 offset1:56
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 v[30:33], v[134:135], v[162:163], v[30:33]
	v_mfma_f32_16x16x16_f16 v[22:25], v[136:137], v[162:163], v[22:25]
	v_mfma_f32_16x16x16_f16 v[10:13], v[142:143], v[162:163], v[10:13]
	v_mfma_f32_16x16x16_f16 v[14:17], v[144:145], v[162:163], v[14:17]
	v_mfma_f32_16x16x16_f16 v[6:9], v[134:135], v[164:165], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[136:137], v[164:165], v[2:5]
	v_mfma_f32_16x16x16_f16 v[26:29], v[142:143], v[164:165], v[26:29]
	v_mfma_f32_16x16x16_f16 v[18:21], v[144:145], v[164:165], v[18:21]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[94:97], v[146:147], v[176:177], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[148:149], v[176:177], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[158:159], v[176:177], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[160:161], v[176:177], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[146:147], v[178:179], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[148:149], v[178:179], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[158:159], v[178:179], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[160:161], v[178:179], v[66:69]
	ds_read2st64_b64 v[176:179], v182 offset0:32 offset1:40
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[62:65], v[138:139], v[170:171], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[140:141], v[170:171], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[154:155], v[170:171], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[156:157], v[170:171], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[138:139], v[172:173], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[140:141], v[172:173], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[154:155], v[172:173], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[156:157], v[172:173], v[34:37]
	ds_read2st64_b64 v[170:173], v181 offset0:48 offset1:56
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[30:33], v[130:131], v[166:167], v[30:33]
	v_mfma_f32_16x16x16_f16 v[22:25], v[132:133], v[166:167], v[22:25]
	v_mfma_f32_16x16x16_f16 v[10:13], v[150:151], v[166:167], v[10:13]
	v_mfma_f32_16x16x16_f16 v[14:17], v[152:153], v[166:167], v[14:17]
	v_mfma_f32_16x16x16_f16 v[6:9], v[130:131], v[168:169], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[132:133], v[168:169], v[2:5]
	v_mfma_f32_16x16x16_f16 v[26:29], v[150:151], v[168:169], v[26:29]
	v_mfma_f32_16x16x16_f16 v[18:21], v[152:153], v[168:169], v[18:21]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[62:65], v[146:147], v[176:177], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[148:149], v[176:177], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[158:159], v[176:177], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[160:161], v[176:177], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[146:147], v[178:179], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[148:149], v[178:179], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[158:159], v[178:179], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[160:161], v[178:179], v[34:37]
	ds_read2st64_b64 v[176:179], v182 offset0:48 offset1:56
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[30:33], v[138:139], v[170:171], v[30:33]
	v_mfma_f32_16x16x16_f16 v[22:25], v[140:141], v[170:171], v[22:25]
	v_mfma_f32_16x16x16_f16 v[10:13], v[154:155], v[170:171], v[10:13]
	v_mfma_f32_16x16x16_f16 v[14:17], v[156:157], v[170:171], v[14:17]
	v_mfma_f32_16x16x16_f16 v[6:9], v[138:139], v[172:173], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[140:141], v[172:173], v[2:5]
	v_mfma_f32_16x16x16_f16 v[26:29], v[154:155], v[172:173], v[26:29]
	v_mfma_f32_16x16x16_f16 v[18:21], v[156:157], v[172:173], v[18:21]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[30:33], v[146:147], v[176:177], v[30:33]
	v_mfma_f32_16x16x16_f16 v[22:25], v[148:149], v[176:177], v[22:25]
	v_mfma_f32_16x16x16_f16 v[10:13], v[158:159], v[176:177], v[10:13]
	v_mfma_f32_16x16x16_f16 v[14:17], v[160:161], v[176:177], v[14:17]
	v_mfma_f32_16x16x16_f16 v[6:9], v[146:147], v[178:179], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[148:149], v[178:179], v[2:5]
	v_mfma_f32_16x16x16_f16 v[26:29], v[158:159], v[178:179], v[26:29]
	v_mfma_f32_16x16x16_f16 v[18:21], v[160:161], v[178:179], v[18:21]
.LBB0_12:
	v_lshrrev_b32_e32 v0, 4, v0
	v_and_or_b32 v0, v0, 16, v1
	v_mul_lo_u32 v145, s13, v0
	s_lshl_b32 s1, s13, 5
	v_add_u32_e32 v146, s1, v145
	v_add_u32_e32 v147, s1, v146
	v_add_u32_e32 v148, s1, v147
	v_add_u32_e32 v149, s1, v148
	v_add_u32_e32 v150, s1, v149
	s_mul_i32 s0, s15, s13
	v_add_u32_e32 v151, s1, v150
	v_add_u32_e32 v152, s1, v151
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	v_or_b32_e32 v1, s15, v0
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	v_and_b32_e32 v130, 60, v174
	s_lshl_b64 s[0:1], s[14:15], 1
	v_or_b32_e32 v131, 0xc0, v130
	v_or_b32_e32 v133, 0x80, v130
	v_or_b32_e32 v135, 64, v130
	v_or_b32_e32 v138, 0xe0, v1
	v_or_b32_e32 v139, 0xc0, v1
	v_or_b32_e32 v140, 0xa0, v1
	v_or_b32_e32 v141, 0x80, v1
	v_or_b32_e32 v142, 0x60, v1
	v_or_b32_e32 v143, 64, v1
	v_or_b32_e32 v144, 32, v1
	v_cvt_f16_f32_e32 v126, v126
	v_cvt_f16_f32_e32 v127, v127
	s_add_u32 s28, s2, s0
	v_or_b32_e32 v132, s14, v131
	v_or_b32_e32 v134, s14, v133
	v_or_b32_e32 v136, s14, v135
	v_or_b32_e32 v137, s14, v130
	v_cvt_f16_f32_e32 v128, v128
	v_cvt_f16_f32_e32 v129, v129
	s_addc_u32 s12, s3, s1
	v_cmp_gt_i32_e64 s[24:25], s8, v1
	v_cmp_gt_i32_e64 s[18:19], s8, v144
	v_cmp_gt_i32_e64 s[16:17], s8, v143
	v_cmp_gt_i32_e64 s[14:15], s8, v142
	v_cmp_gt_i32_e64 s[20:21], s8, v141
	v_cmp_gt_i32_e64 s[10:11], s8, v140
	v_cmp_gt_i32_e64 s[22:23], s8, v139
	v_cmp_gt_i32_e32 vcc, s8, v138
	s_and_b32 s8, s13, 0x3fff
	v_cvt_f16_f32_e32 v122, v122
	v_cvt_f16_f32_e32 v123, v123
	s_bitset1_b32 s8, 14
	v_cvt_f16_f32_e32 v124, v124
	v_cvt_f16_f32_e32 v125, v125
	v_cmp_gt_i32_e64 s[6:7], s9, v137
	v_cmp_gt_i32_e64 s[4:5], s9, v136
	v_cmp_gt_i32_e64 s[2:3], s9, v134
	v_cmp_gt_i32_e64 s[0:1], s9, v132
	s_and_b32 s9, s12, 0xffff
	s_lshl_b32 s8, s8, 16
	v_cvt_f16_f32_e32 v118, v118
	v_cvt_f16_f32_e32 v119, v119
	s_or_b32 s29, s9, s8
	v_pack_b32_f16 v0, v126, v127
	v_add_lshl_u32 v126, v145, v130, 1
	v_bfrev_b32_e32 v127, 1
	s_and_b64 s[8:9], s[24:25], s[6:7]
	v_cvt_f16_f32_e32 v120, v120
	v_cvt_f16_f32_e32 v121, v121
	s_mov_b32 s31, 0x27000
	s_mov_b32 s30, 0x7ffffffe
	v_pack_b32_f16 v1, v128, v129
	v_cndmask_b32_e64 v126, v127, v126, s[8:9]
	v_cvt_f16_f32_e32 v114, v114
	v_cvt_f16_f32_e32 v115, v115
	buffer_store_dwordx2 v[0:1], v126, s[28:31], 0 offen
	v_pack_b32_f16 v0, v122, v123
	v_add_lshl_u32 v122, v145, v135, 1
	s_and_b64 s[8:9], s[24:25], s[4:5]
	v_cvt_f16_f32_e32 v116, v116
	v_cvt_f16_f32_e32 v117, v117
	v_pack_b32_f16 v1, v124, v125
	v_cndmask_b32_e64 v122, v127, v122, s[8:9]
	v_cvt_f16_f32_e32 v110, v110
	v_cvt_f16_f32_e32 v111, v111
	buffer_store_dwordx2 v[0:1], v122, s[28:31], 0 offen
	v_pack_b32_f16 v0, v118, v119
	v_add_lshl_u32 v118, v145, v133, 1
	s_and_b64 s[8:9], s[24:25], s[2:3]
	v_cvt_f16_f32_e32 v112, v112
	v_cvt_f16_f32_e32 v113, v113
	v_pack_b32_f16 v1, v120, v121
	v_cndmask_b32_e64 v118, v127, v118, s[8:9]
	v_cvt_f16_f32_e32 v106, v106
	v_cvt_f16_f32_e32 v107, v107
	buffer_store_dwordx2 v[0:1], v118, s[28:31], 0 offen
	v_pack_b32_f16 v0, v114, v115
	v_add_lshl_u32 v114, v145, v131, 1
	s_and_b64 s[8:9], s[24:25], s[0:1]
	v_cvt_f16_f32_e32 v108, v108
	v_cvt_f16_f32_e32 v109, v109
	v_pack_b32_f16 v1, v116, v117
	v_cndmask_b32_e64 v114, v127, v114, s[8:9]
	v_cvt_f16_f32_e32 v102, v102
	v_cvt_f16_f32_e32 v103, v103
	buffer_store_dwordx2 v[0:1], v114, s[28:31], 0 offen
	v_pack_b32_f16 v0, v110, v111
	v_add_lshl_u32 v110, v146, v130, 1
	s_and_b64 s[8:9], s[18:19], s[6:7]
	v_cvt_f16_f32_e32 v104, v104
	v_cvt_f16_f32_e32 v105, v105
	v_pack_b32_f16 v1, v112, v113
	v_cndmask_b32_e64 v110, v127, v110, s[8:9]
	v_cvt_f16_f32_e32 v98, v98
	v_cvt_f16_f32_e32 v99, v99
	buffer_store_dwordx2 v[0:1], v110, s[28:31], 0 offen
	v_pack_b32_f16 v0, v106, v107
	v_add_lshl_u32 v106, v146, v135, 1
	s_and_b64 s[8:9], s[18:19], s[4:5]
	v_cvt_f16_f32_e32 v100, v100
	v_cvt_f16_f32_e32 v101, v101
	v_pack_b32_f16 v1, v108, v109
	v_cndmask_b32_e64 v106, v127, v106, s[8:9]
	v_cvt_f16_f32_e32 v94, v94
	v_cvt_f16_f32_e32 v95, v95
	buffer_store_dwordx2 v[0:1], v106, s[28:31], 0 offen
	v_pack_b32_f16 v0, v102, v103
	v_add_lshl_u32 v102, v146, v133, 1
	s_and_b64 s[8:9], s[18:19], s[2:3]
	v_cvt_f16_f32_e32 v96, v96
	v_cvt_f16_f32_e32 v97, v97
	v_pack_b32_f16 v1, v104, v105
	v_cndmask_b32_e64 v102, v127, v102, s[8:9]
	v_cvt_f16_f32_e32 v90, v90
	v_cvt_f16_f32_e32 v91, v91
	buffer_store_dwordx2 v[0:1], v102, s[28:31], 0 offen
	v_pack_b32_f16 v0, v98, v99
	v_add_lshl_u32 v98, v146, v131, 1
	s_and_b64 s[8:9], s[18:19], s[0:1]
	v_cvt_f16_f32_e32 v92, v92
	v_cvt_f16_f32_e32 v93, v93
	v_pack_b32_f16 v1, v100, v101
	v_cndmask_b32_e64 v98, v127, v98, s[8:9]
	v_cvt_f16_f32_e32 v86, v86
	v_cvt_f16_f32_e32 v87, v87
	buffer_store_dwordx2 v[0:1], v98, s[28:31], 0 offen
	v_pack_b32_f16 v0, v94, v95
	v_add_lshl_u32 v94, v147, v130, 1
	s_and_b64 s[8:9], s[16:17], s[6:7]
	v_cvt_f16_f32_e32 v88, v88
	v_cvt_f16_f32_e32 v89, v89
	v_pack_b32_f16 v1, v96, v97
	v_cndmask_b32_e64 v94, v127, v94, s[8:9]
	v_cvt_f16_f32_e32 v82, v82
	v_cvt_f16_f32_e32 v83, v83
	buffer_store_dwordx2 v[0:1], v94, s[28:31], 0 offen
	v_pack_b32_f16 v0, v90, v91
	v_add_lshl_u32 v90, v147, v135, 1
	s_and_b64 s[8:9], s[16:17], s[4:5]
	v_cvt_f16_f32_e32 v84, v84
	v_cvt_f16_f32_e32 v85, v85
	v_pack_b32_f16 v1, v92, v93
	v_cndmask_b32_e64 v90, v127, v90, s[8:9]
	v_cvt_f16_f32_e32 v78, v78
	v_cvt_f16_f32_e32 v79, v79
	buffer_store_dwordx2 v[0:1], v90, s[28:31], 0 offen
	v_pack_b32_f16 v0, v86, v87
	v_add_lshl_u32 v86, v147, v133, 1
	s_and_b64 s[8:9], s[16:17], s[2:3]
	v_cvt_f16_f32_e32 v80, v80
	v_cvt_f16_f32_e32 v81, v81
	v_pack_b32_f16 v1, v88, v89
	v_cndmask_b32_e64 v86, v127, v86, s[8:9]
	v_cvt_f16_f32_e32 v74, v74
	v_cvt_f16_f32_e32 v75, v75
	buffer_store_dwordx2 v[0:1], v86, s[28:31], 0 offen
	v_pack_b32_f16 v0, v82, v83
	v_add_lshl_u32 v82, v147, v131, 1
	s_and_b64 s[8:9], s[16:17], s[0:1]
	v_cvt_f16_f32_e32 v76, v76
	v_cvt_f16_f32_e32 v77, v77
	v_pack_b32_f16 v1, v84, v85
	v_cndmask_b32_e64 v82, v127, v82, s[8:9]
	v_cvt_f16_f32_e32 v70, v70
	v_cvt_f16_f32_e32 v71, v71
	buffer_store_dwordx2 v[0:1], v82, s[28:31], 0 offen
	v_pack_b32_f16 v0, v78, v79
	v_add_lshl_u32 v78, v148, v130, 1
	s_and_b64 s[8:9], s[14:15], s[6:7]
	v_cvt_f16_f32_e32 v72, v72
	v_cvt_f16_f32_e32 v73, v73
	v_pack_b32_f16 v1, v80, v81
	v_cndmask_b32_e64 v78, v127, v78, s[8:9]
	v_cvt_f16_f32_e32 v66, v66
	v_cvt_f16_f32_e32 v67, v67
	buffer_store_dwordx2 v[0:1], v78, s[28:31], 0 offen
	v_pack_b32_f16 v0, v74, v75
	v_add_lshl_u32 v74, v148, v135, 1
	s_and_b64 s[8:9], s[14:15], s[4:5]
	v_cvt_f16_f32_e32 v68, v68
	v_cvt_f16_f32_e32 v69, v69
	v_pack_b32_f16 v1, v76, v77
	v_cndmask_b32_e64 v74, v127, v74, s[8:9]
	v_cvt_f16_f32_e32 v62, v62
	v_cvt_f16_f32_e32 v63, v63
	buffer_store_dwordx2 v[0:1], v74, s[28:31], 0 offen
	v_pack_b32_f16 v0, v70, v71
	v_add_lshl_u32 v70, v148, v133, 1
	s_and_b64 s[8:9], s[14:15], s[2:3]
	v_cvt_f16_f32_e32 v64, v64
	v_cvt_f16_f32_e32 v65, v65
	v_pack_b32_f16 v1, v72, v73
	v_cndmask_b32_e64 v70, v127, v70, s[8:9]
	v_cvt_f16_f32_e32 v58, v58
	v_cvt_f16_f32_e32 v59, v59
	buffer_store_dwordx2 v[0:1], v70, s[28:31], 0 offen
	v_pack_b32_f16 v0, v66, v67
	v_add_lshl_u32 v66, v148, v131, 1
	s_and_b64 s[8:9], s[14:15], s[0:1]
	v_cvt_f16_f32_e32 v60, v60
	v_cvt_f16_f32_e32 v61, v61
	v_pack_b32_f16 v1, v68, v69
	v_cndmask_b32_e64 v66, v127, v66, s[8:9]
	v_cvt_f16_f32_e32 v54, v54
	v_cvt_f16_f32_e32 v55, v55
	buffer_store_dwordx2 v[0:1], v66, s[28:31], 0 offen
	v_pack_b32_f16 v0, v62, v63
	v_add_lshl_u32 v62, v149, v130, 1
	s_and_b64 s[8:9], s[20:21], s[6:7]
	v_cvt_f16_f32_e32 v56, v56
	v_cvt_f16_f32_e32 v57, v57
	v_pack_b32_f16 v1, v64, v65
	v_cndmask_b32_e64 v62, v127, v62, s[8:9]
	v_cvt_f16_f32_e32 v50, v50
	v_cvt_f16_f32_e32 v51, v51
	buffer_store_dwordx2 v[0:1], v62, s[28:31], 0 offen
	v_pack_b32_f16 v0, v58, v59
	v_add_lshl_u32 v58, v149, v135, 1
	s_and_b64 s[8:9], s[20:21], s[4:5]
	v_cvt_f16_f32_e32 v52, v52
	v_cvt_f16_f32_e32 v53, v53
	v_pack_b32_f16 v1, v60, v61
	v_cndmask_b32_e64 v58, v127, v58, s[8:9]
	v_cvt_f16_f32_e32 v46, v46
	v_cvt_f16_f32_e32 v47, v47
	buffer_store_dwordx2 v[0:1], v58, s[28:31], 0 offen
	v_pack_b32_f16 v0, v54, v55
	v_add_lshl_u32 v54, v149, v133, 1
	s_and_b64 s[8:9], s[20:21], s[2:3]
	v_cvt_f16_f32_e32 v48, v48
	v_cvt_f16_f32_e32 v49, v49
	v_pack_b32_f16 v1, v56, v57
	v_cndmask_b32_e64 v54, v127, v54, s[8:9]
	v_cvt_f16_f32_e32 v42, v42
	v_cvt_f16_f32_e32 v43, v43
	buffer_store_dwordx2 v[0:1], v54, s[28:31], 0 offen
	v_pack_b32_f16 v0, v50, v51
	v_add_lshl_u32 v50, v149, v131, 1
	s_and_b64 s[8:9], s[20:21], s[0:1]
	v_cvt_f16_f32_e32 v44, v44
	v_cvt_f16_f32_e32 v45, v45
	v_pack_b32_f16 v1, v52, v53
	v_cndmask_b32_e64 v50, v127, v50, s[8:9]
	v_cvt_f16_f32_e32 v38, v38
	v_cvt_f16_f32_e32 v39, v39
	buffer_store_dwordx2 v[0:1], v50, s[28:31], 0 offen
	v_pack_b32_f16 v0, v46, v47
	v_add_lshl_u32 v46, v150, v130, 1
	s_and_b64 s[8:9], s[10:11], s[6:7]
	v_cvt_f16_f32_e32 v40, v40
	v_cvt_f16_f32_e32 v41, v41
	v_pack_b32_f16 v1, v48, v49
	v_cndmask_b32_e64 v46, v127, v46, s[8:9]
	v_cvt_f16_f32_e32 v34, v34
	v_cvt_f16_f32_e32 v35, v35
	buffer_store_dwordx2 v[0:1], v46, s[28:31], 0 offen
	v_pack_b32_f16 v0, v42, v43
	v_add_lshl_u32 v42, v150, v135, 1
	s_and_b64 s[8:9], s[10:11], s[4:5]
	v_cvt_f16_f32_e32 v36, v36
	v_cvt_f16_f32_e32 v37, v37
	v_pack_b32_f16 v1, v44, v45
	v_cndmask_b32_e64 v42, v127, v42, s[8:9]
	v_cvt_f16_f32_e32 v30, v30
	v_cvt_f16_f32_e32 v31, v31
	buffer_store_dwordx2 v[0:1], v42, s[28:31], 0 offen
	v_pack_b32_f16 v0, v38, v39
	v_add_lshl_u32 v38, v150, v133, 1
	s_and_b64 s[8:9], s[10:11], s[2:3]
	v_cvt_f16_f32_e32 v32, v32
	v_cvt_f16_f32_e32 v33, v33
	v_pack_b32_f16 v1, v40, v41
	v_cndmask_b32_e64 v38, v127, v38, s[8:9]
	v_cvt_f16_f32_e32 v22, v22
	v_cvt_f16_f32_e32 v23, v23
	buffer_store_dwordx2 v[0:1], v38, s[28:31], 0 offen
	v_pack_b32_f16 v0, v34, v35
	v_add_lshl_u32 v34, v150, v131, 1
	s_and_b64 s[8:9], s[10:11], s[0:1]
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v25, v25
	v_pack_b32_f16 v1, v36, v37
	v_cndmask_b32_e64 v34, v127, v34, s[8:9]
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v11, v11
	buffer_store_dwordx2 v[0:1], v34, s[28:31], 0 offen
	v_pack_b32_f16 v0, v30, v31
	v_add_lshl_u32 v30, v151, v130, 1
	s_and_b64 s[8:9], s[22:23], s[6:7]
	v_cvt_f16_f32_e32 v12, v12
	v_cvt_f16_f32_e32 v13, v13
	v_pack_b32_f16 v1, v32, v33
	v_cndmask_b32_e64 v30, v127, v30, s[8:9]
	buffer_store_dwordx2 v[0:1], v30, s[28:31], 0 offen
	v_pack_b32_f16 v0, v22, v23
	v_add_lshl_u32 v22, v151, v135, 1
	s_and_b64 s[8:9], s[22:23], s[4:5]
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_pack_b32_f16 v1, v24, v25
	v_cndmask_b32_e64 v22, v127, v22, s[8:9]
	v_cvt_f16_f32_e32 v6, v6
	v_cvt_f16_f32_e32 v7, v7
	buffer_store_dwordx2 v[0:1], v22, s[28:31], 0 offen
	v_pack_b32_f16 v0, v10, v11
	v_add_lshl_u32 v10, v151, v133, 1
	s_and_b64 s[8:9], s[22:23], s[2:3]
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v9, v9
	v_pack_b32_f16 v1, v12, v13
	v_cndmask_b32_e64 v10, v127, v10, s[8:9]
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	buffer_store_dwordx2 v[0:1], v10, s[28:31], 0 offen
	v_add_lshl_u32 v10, v151, v131, 1
	s_and_b64 s[8:9], s[22:23], s[0:1]
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v5, v5
	v_pack_b32_f16 v1, v16, v17
	v_pack_b32_f16 v0, v14, v15
	v_cndmask_b32_e64 v10, v127, v10, s[8:9]
	buffer_store_dwordx2 v[0:1], v10, s[28:31], 0 offen
	v_pack_b32_f16 v0, v6, v7
	v_add_lshl_u32 v6, v152, v130, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cvt_f16_f32_e32 v26, v26
	v_cvt_f16_f32_e32 v27, v27
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v29, v29
	v_pack_b32_f16 v1, v8, v9
	v_cndmask_b32_e64 v6, v127, v6, s[6:7]
	buffer_store_dwordx2 v[0:1], v6, s[28:31], 0 offen
	v_pack_b32_f16 v0, v2, v3
	v_add_lshl_u32 v2, v152, v135, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v21, v21
	v_pack_b32_f16 v1, v4, v5
	v_cndmask_b32_e64 v2, v127, v2, s[4:5]
	buffer_store_dwordx2 v[0:1], v2, s[28:31], 0 offen
	v_add_lshl_u32 v2, v152, v133, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_pack_b32_f16 v1, v28, v29
	v_pack_b32_f16 v0, v26, v27
	v_cndmask_b32_e64 v2, v127, v2, s[2:3]
	buffer_store_dwordx2 v[0:1], v2, s[28:31], 0 offen
	v_add_lshl_u32 v2, v152, v131, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_pack_b32_f16 v1, v20, v21
	v_pack_b32_f16 v0, v18, v19
	v_cndmask_b32_e32 v2, v127, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[28:31], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 15
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 13
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 198
		.amdhsa_next_free_sgpr 32
		.amdhsa_accum_offset 200
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
	.set matmul_kernel.num_vgpr, 198
	.set matmul_kernel.num_agpr, 0
	.set matmul_kernel.numbered_sgpr, 32
	.set matmul_kernel.private_seg_size, 0
	.set matmul_kernel.uses_vcc, 1
	.set matmul_kernel.uses_flat_scratch, 0
	.set matmul_kernel.has_dyn_sized_stack, 0
	.set matmul_kernel.has_recursion, 0
	.set matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7832
; TotalNumSgprs: 38
; NumVgprs: 198
; NumAgprs: 0
; TotalNumVgprs: 198
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 38
; NumVGPRsForWavesPerEU: 198
; AccumOffset: 200
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 49
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
	.byte	4                               ; Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	21                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4d:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	22                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x61:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	73                              ; DW_AT_call_line
	.byte	33                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/AMD-triton/python/perf-kernels/tools/tune_gemm" ; string offset=24
.Linfo_string3:
	.asciz	"matmul_kernel"                 ; string offset=88
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
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 512
    .name:           matmul_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     38
    .sgpr_spill_count: 0
    .symbol:         matmul_kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     198
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
