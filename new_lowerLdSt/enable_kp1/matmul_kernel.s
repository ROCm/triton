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
; %bb.10:
	.file	1 "/var/lib/jenkins/AMD-triton/python/perf-kernels/tools/tune_gemm" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_load_dword s14, s[0:1], 0x30
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.11:
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
	v_lshlrev_b32_e32 v146, 3, v0
	v_and_b32_e32 v26, 56, v146
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
	s_sub_i32 s24, s1, s0
	s_add_i32 s24, s24, s15
	s_lshl_b32 s21, s24, 8
	s_mul_i32 s0, s21, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s20, s14, 8
	s_lshl_b32 s14, s11, 7
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s16, s2, s0
	v_lshrrev_b32_e32 v1, 3, v0
	s_addc_u32 s15, s3, s1
	s_add_i32 s25, s10, 63
	v_mad_u64_u32 v[10:11], s[0:1], s11, v1, v[26:27]
	s_cmp_gt_i32 s25, 63
	s_cselect_b64 vcc, -1, 0
	s_and_b32 s0, s11, 0x3fff
	v_or_b32_e32 v18, 64, v1
	s_bitset1_b32 s0, 14
	v_or_b32_e32 v19, 0xc0, v1
	v_mul_lo_u32 v2, s11, v18
	s_and_b32 s1, s15, 0xffff
	s_lshl_b32 s10, s0, 16
	s_mul_i32 s0, s20, s12
	v_mul_lo_u32 v12, s11, v19
	v_lshlrev_b32_e32 v148, 1, v10
	v_bfrev_b32_e32 v27, 1
	s_or_b32 s17, s1, s10
	v_add_lshl_u32 v149, v2, v26, 1
	v_add_lshl_u32 v150, v10, s14, 1
	s_ashr_i32 s1, s0, 31
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	v_cndmask_b32_e32 v11, v27, v148, vcc
	v_cndmask_b32_e32 v13, v27, v149, vcc
	v_cndmask_b32_e32 v20, v27, v150, vcc
	v_add_lshl_u32 v151, v12, v26, 1
	s_lshl_b32 s14, s12, 7
	s_lshl_b64 s[22:23], s[0:1], 1
	buffer_load_dwordx4 v[2:5], v11, s[16:19], 0 offen
	buffer_load_dwordx4 v[6:9], v13, s[16:19], 0 offen
	v_cndmask_b32_e32 v21, v27, v151, vcc
	buffer_load_dwordx4 v[10:13], v20, s[16:19], 0 offen
	buffer_load_dwordx4 v[14:17], v21, s[16:19], 0 offen
	s_add_u32 s16, s4, s22
	v_mad_u64_u32 v[28:29], s[0:1], s12, v1, v[26:27]
	s_addc_u32 s15, s5, s23
	s_and_b32 s0, s12, 0x3fff
	s_bitset1_b32 s0, 14
	v_mul_lo_u32 v18, s12, v18
	v_mul_lo_u32 v30, s12, v19
	s_and_b32 s1, s15, 0xffff
	s_lshl_b32 s12, s0, 16
	v_lshlrev_b32_e32 v152, 1, v28
	s_or_b32 s17, s1, s12
	v_cndmask_b32_e32 v1, v27, v152, vcc
	v_add_lshl_u32 v153, v18, v26, 1
	v_add_lshl_u32 v154, v28, s14, 1
	v_cndmask_b32_e32 v29, v27, v153, vcc
	buffer_load_dwordx4 v[18:21], v1, s[16:19], 0 offen
	buffer_load_dwordx4 v[22:25], v29, s[16:19], 0 offen
	v_cndmask_b32_e32 v1, v27, v154, vcc
	v_add_lshl_u32 v155, v30, v26, 1
	v_cndmask_b32_e32 v34, v27, v155, vcc
	buffer_load_dwordx4 v[26:29], v1, s[16:19], 0 offen
	buffer_load_dwordx4 v[30:33], v34, s[16:19], 0 offen
	v_lshlrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v34, 0x78, v0
	v_xor_b32_e32 v1, v1, v34
	s_add_i32 s0, 0, 0x8000
	v_add_u32_e32 v156, 0, v1
	v_xor_b32_e32 v34, 8, v1
	v_add_u32_e32 v1, s0, v1
	s_movk_i32 s14, 0xff
	v_add_u32_e32 v157, 0, v34
	v_cmp_lt_u32_e64 s[14:15], s14, v0
	s_waitcnt vmcnt(6)
	ds_write2st64_b64 v156, v[2:3], v[6:7] offset1:16
	s_waitcnt vmcnt(4)
	ds_write2st64_b64 v156, v[10:11], v[14:15] offset0:32 offset1:48
	ds_write2st64_b64 v157, v[4:5], v[8:9] offset1:16
	ds_write2st64_b64 v157, v[12:13], v[16:17] offset0:32 offset1:48
	s_waitcnt vmcnt(3)
	ds_write_b64 v156, v[18:19] offset:32768
	s_waitcnt vmcnt(1)
	ds_write2st64_b64 v1, v[22:23], v[26:27] offset0:16 offset1:32
	s_waitcnt vmcnt(0)
	ds_write_b64 v1, v[30:31] offset:24576
	ds_write_b64 v157, v[20:21] offset:32768
	v_add_u32_e32 v1, s0, v34
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e64 s[0:1], s0, v0
	ds_write2st64_b64 v1, v[24:25], v[28:29] offset0:16 offset1:32
	ds_write_b64 v1, v[32:33] offset:24576
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[16:17], s[14:15]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[16:17]
	v_mov_b32_e32 v5, 0
	s_cmpk_lt_i32 s25, 0x80
	v_mov_b32_e32 v4, v5
	v_mov_b32_e32 v3, v5
	v_mov_b32_e32 v2, v5
	v_mov_b32_e32 v9, v5
	v_mov_b32_e32 v8, v5
	v_mov_b32_e32 v7, v5
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v13, v5
	v_mov_b32_e32 v12, v5
	v_mov_b32_e32 v11, v5
	v_mov_b32_e32 v10, v5
	v_mov_b32_e32 v17, v5
	v_mov_b32_e32 v16, v5
	v_mov_b32_e32 v15, v5
	v_mov_b32_e32 v14, v5
	v_mov_b32_e32 v21, v5
	v_mov_b32_e32 v20, v5
	v_mov_b32_e32 v19, v5
	v_mov_b32_e32 v18, v5
	v_mov_b32_e32 v25, v5
	v_mov_b32_e32 v24, v5
	v_mov_b32_e32 v23, v5
	v_mov_b32_e32 v22, v5
	v_mov_b32_e32 v29, v5
	v_mov_b32_e32 v28, v5
	v_mov_b32_e32 v27, v5
	v_mov_b32_e32 v26, v5
	v_mov_b32_e32 v33, v5
	v_mov_b32_e32 v32, v5
	v_mov_b32_e32 v31, v5
	v_mov_b32_e32 v30, v5
	v_mov_b32_e32 v37, v5
	v_mov_b32_e32 v36, v5
	v_mov_b32_e32 v35, v5
	v_mov_b32_e32 v34, v5
	v_mov_b32_e32 v41, v5
	v_mov_b32_e32 v40, v5
	v_mov_b32_e32 v39, v5
	v_mov_b32_e32 v38, v5
	v_mov_b32_e32 v45, v5
	v_mov_b32_e32 v44, v5
	v_mov_b32_e32 v43, v5
	v_mov_b32_e32 v42, v5
	v_mov_b32_e32 v49, v5
	v_mov_b32_e32 v48, v5
	v_mov_b32_e32 v47, v5
	v_mov_b32_e32 v46, v5
	v_mov_b32_e32 v53, v5
	v_mov_b32_e32 v52, v5
	v_mov_b32_e32 v51, v5
	v_mov_b32_e32 v50, v5
	v_mov_b32_e32 v57, v5
	v_mov_b32_e32 v56, v5
	v_mov_b32_e32 v55, v5
	v_mov_b32_e32 v54, v5
	v_mov_b32_e32 v61, v5
	v_mov_b32_e32 v60, v5
	v_mov_b32_e32 v59, v5
	v_mov_b32_e32 v58, v5
	v_mov_b32_e32 v65, v5
	v_mov_b32_e32 v64, v5
	v_mov_b32_e32 v63, v5
	v_mov_b32_e32 v62, v5
	v_mov_b32_e32 v69, v5
	v_mov_b32_e32 v68, v5
	v_mov_b32_e32 v67, v5
	v_mov_b32_e32 v66, v5
	v_mov_b32_e32 v73, v5
	v_mov_b32_e32 v72, v5
	v_mov_b32_e32 v71, v5
	v_mov_b32_e32 v70, v5
	v_mov_b32_e32 v77, v5
	v_mov_b32_e32 v76, v5
	v_mov_b32_e32 v75, v5
	v_mov_b32_e32 v74, v5
	v_mov_b32_e32 v81, v5
	v_mov_b32_e32 v80, v5
	v_mov_b32_e32 v79, v5
	v_mov_b32_e32 v78, v5
	v_mov_b32_e32 v85, v5
	v_mov_b32_e32 v84, v5
	v_mov_b32_e32 v83, v5
	v_mov_b32_e32 v82, v5
	v_mov_b32_e32 v89, v5
	v_mov_b32_e32 v88, v5
	v_mov_b32_e32 v87, v5
	v_mov_b32_e32 v86, v5
	v_mov_b32_e32 v93, v5
	v_mov_b32_e32 v92, v5
	v_mov_b32_e32 v91, v5
	v_mov_b32_e32 v90, v5
	v_mov_b32_e32 v97, v5
	v_mov_b32_e32 v96, v5
	v_mov_b32_e32 v95, v5
	v_mov_b32_e32 v94, v5
	v_mov_b32_e32 v101, v5
	v_mov_b32_e32 v100, v5
	v_mov_b32_e32 v99, v5
	v_mov_b32_e32 v98, v5
	v_mov_b32_e32 v105, v5
	v_mov_b32_e32 v104, v5
	v_mov_b32_e32 v103, v5
	v_mov_b32_e32 v102, v5
	v_mov_b32_e32 v109, v5
	v_mov_b32_e32 v108, v5
	v_mov_b32_e32 v107, v5
	v_mov_b32_e32 v106, v5
	v_mov_b32_e32 v113, v5
	v_mov_b32_e32 v112, v5
	v_mov_b32_e32 v111, v5
	v_mov_b32_e32 v110, v5
	v_mov_b32_e32 v117, v5
	v_mov_b32_e32 v116, v5
	v_mov_b32_e32 v115, v5
	v_mov_b32_e32 v114, v5
	v_mov_b32_e32 v125, v5
	v_mov_b32_e32 v124, v5
	v_mov_b32_e32 v123, v5
	v_mov_b32_e32 v122, v5
	v_mov_b32_e32 v129, v5
	v_mov_b32_e32 v128, v5
	v_mov_b32_e32 v127, v5
	v_mov_b32_e32 v126, v5
	v_mov_b32_e32 v121, v5
	v_mov_b32_e32 v120, v5
	v_mov_b32_e32 v119, v5
	v_mov_b32_e32 v118, v5
	v_and_b32_e32 v1, 15, v0
	v_and_b32_e32 v162, 0x100, v0
	v_lshrrev_b32_e32 v147, 1, v0
	s_cbranch_scc1 .LBB0_5
; %bb.3:                                ; %.lr.ph
	s_ashr_i32 s14, s25, 31
	s_lshr_b32 s14, s14, 26
	s_add_i32 s25, s25, s14
	s_ashr_i32 s14, s25, 6
	s_max_i32 s16, s14, 2
	s_add_u32 s4, s4, s22
	s_addc_u32 s5, s5, s23
	s_add_u32 s4, s4, 0x80
	s_mul_i32 s11, s11, s24
	v_lshlrev_b32_e32 v2, 5, v1
	v_and_b32_e32 v3, 24, v146
	v_lshlrev_b32_e32 v4, 1, v162
	s_movk_i32 s15, 0x618
	s_addc_u32 s5, s5, 0
	s_lshl_b32 s14, s11, 8
	v_or3_b32 v3, v3, v4, v2
	v_and_or_b32 v2, v146, s15, v2
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[14:15], s[14:15], 1
	v_and_b32_e32 v4, 24, v147
	s_add_u32 s2, s14, s2
	v_xor_b32_e32 v3, v3, v4
	v_xor_b32_e32 v2, v2, v4
	s_addc_u32 s3, s15, s3
	s_add_u32 s2, s2, 0x80
	v_mov_b32_e32 v118, 0
	v_add_u32_e32 v158, 0, v3
	v_add_u32_e32 v159, 0, v2
	s_addc_u32 s3, s3, 0
	s_add_i32 s11, s16, -1
	v_add_u32_e32 v160, 32, v158
	v_add_u32_e32 v161, 32, v159
	v_add_u32_e32 v163, 0x800, v158
	v_add_u32_e32 v164, 0x1000, v158
	v_add_u32_e32 v165, 0x1800, v158
	v_add_u32_e32 v166, 0x8800, v159
	v_add_u32_e32 v167, 0x9000, v159
	v_add_u32_e32 v168, 0x9800, v159
	v_add_u32_e32 v169, 0x8000, v159
	v_mov_b32_e32 v119, v118
	v_mov_b32_e32 v120, v118
	v_mov_b32_e32 v121, v118
	v_mov_b32_e32 v126, v118
	v_mov_b32_e32 v127, v118
	v_mov_b32_e32 v128, v118
	v_mov_b32_e32 v129, v118
	v_mov_b32_e32 v122, v118
	v_mov_b32_e32 v123, v118
	v_mov_b32_e32 v124, v118
	v_mov_b32_e32 v125, v118
	v_mov_b32_e32 v114, v118
	v_mov_b32_e32 v115, v118
	v_mov_b32_e32 v116, v118
	v_mov_b32_e32 v117, v118
	v_mov_b32_e32 v110, v118
	v_mov_b32_e32 v111, v118
	v_mov_b32_e32 v112, v118
	v_mov_b32_e32 v113, v118
	v_mov_b32_e32 v106, v118
	v_mov_b32_e32 v107, v118
	v_mov_b32_e32 v108, v118
	v_mov_b32_e32 v109, v118
	v_mov_b32_e32 v102, v118
	v_mov_b32_e32 v103, v118
	v_mov_b32_e32 v104, v118
	v_mov_b32_e32 v105, v118
	v_mov_b32_e32 v98, v118
	v_mov_b32_e32 v99, v118
	v_mov_b32_e32 v100, v118
	v_mov_b32_e32 v101, v118
	v_mov_b32_e32 v94, v118
	v_mov_b32_e32 v95, v118
	v_mov_b32_e32 v96, v118
	v_mov_b32_e32 v97, v118
	v_mov_b32_e32 v90, v118
	v_mov_b32_e32 v91, v118
	v_mov_b32_e32 v92, v118
	v_mov_b32_e32 v93, v118
	v_mov_b32_e32 v86, v118
	v_mov_b32_e32 v87, v118
	v_mov_b32_e32 v88, v118
	v_mov_b32_e32 v89, v118
	v_mov_b32_e32 v82, v118
	v_mov_b32_e32 v83, v118
	v_mov_b32_e32 v84, v118
	v_mov_b32_e32 v85, v118
	v_mov_b32_e32 v78, v118
	v_mov_b32_e32 v79, v118
	v_mov_b32_e32 v80, v118
	v_mov_b32_e32 v81, v118
	v_mov_b32_e32 v74, v118
	v_mov_b32_e32 v75, v118
	v_mov_b32_e32 v76, v118
	v_mov_b32_e32 v77, v118
	v_mov_b32_e32 v70, v118
	v_mov_b32_e32 v71, v118
	v_mov_b32_e32 v72, v118
	v_mov_b32_e32 v73, v118
	v_mov_b32_e32 v66, v118
	v_mov_b32_e32 v67, v118
	v_mov_b32_e32 v68, v118
	v_mov_b32_e32 v69, v118
	v_mov_b32_e32 v62, v118
	v_mov_b32_e32 v63, v118
	v_mov_b32_e32 v64, v118
	v_mov_b32_e32 v65, v118
	v_mov_b32_e32 v58, v118
	v_mov_b32_e32 v59, v118
	v_mov_b32_e32 v60, v118
	v_mov_b32_e32 v61, v118
	v_mov_b32_e32 v54, v118
	v_mov_b32_e32 v55, v118
	v_mov_b32_e32 v56, v118
	v_mov_b32_e32 v57, v118
	v_mov_b32_e32 v50, v118
	v_mov_b32_e32 v51, v118
	v_mov_b32_e32 v52, v118
	v_mov_b32_e32 v53, v118
	v_mov_b32_e32 v46, v118
	v_mov_b32_e32 v47, v118
	v_mov_b32_e32 v48, v118
	v_mov_b32_e32 v49, v118
	v_mov_b32_e32 v42, v118
	v_mov_b32_e32 v43, v118
	v_mov_b32_e32 v44, v118
	v_mov_b32_e32 v45, v118
	v_mov_b32_e32 v38, v118
	v_mov_b32_e32 v39, v118
	v_mov_b32_e32 v40, v118
	v_mov_b32_e32 v41, v118
	v_mov_b32_e32 v34, v118
	v_mov_b32_e32 v35, v118
	v_mov_b32_e32 v36, v118
	v_mov_b32_e32 v37, v118
	v_mov_b32_e32 v30, v118
	v_mov_b32_e32 v31, v118
	v_mov_b32_e32 v32, v118
	v_mov_b32_e32 v33, v118
	v_mov_b32_e32 v26, v118
	v_mov_b32_e32 v27, v118
	v_mov_b32_e32 v28, v118
	v_mov_b32_e32 v29, v118
	v_mov_b32_e32 v22, v118
	v_mov_b32_e32 v23, v118
	v_mov_b32_e32 v24, v118
	v_mov_b32_e32 v25, v118
	v_mov_b32_e32 v18, v118
	v_mov_b32_e32 v19, v118
	v_mov_b32_e32 v20, v118
	v_mov_b32_e32 v21, v118
	v_mov_b32_e32 v14, v118
	v_mov_b32_e32 v15, v118
	v_mov_b32_e32 v16, v118
	v_mov_b32_e32 v17, v118
	v_mov_b32_e32 v10, v118
	v_mov_b32_e32 v11, v118
	v_mov_b32_e32 v12, v118
	v_mov_b32_e32 v13, v118
	v_mov_b32_e32 v6, v118
	v_mov_b32_e32 v7, v118
	v_mov_b32_e32 v8, v118
	v_mov_b32_e32 v9, v118
	v_mov_b32_e32 v2, v118
	v_mov_b32_e32 v3, v118
	v_mov_b32_e32 v4, v118
	v_mov_b32_e32 v5, v118
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s14, s3, 0xffff
	s_or_b32 s17, s14, s10
	s_mov_b32 s16, s2
	buffer_load_dwordx4 v[138:141], v148, s[16:19], 0 offen
	buffer_load_dwordx4 v[142:145], v149, s[16:19], 0 offen
	buffer_load_dwordx4 v[130:133], v150, s[16:19], 0 offen
	buffer_load_dwordx4 v[134:137], v151, s[16:19], 0 offen
	ds_read2st64_b64 v[170:173], v158 offset1:2
	ds_read2st64_b64 v[174:177], v158 offset0:4 offset1:6
	ds_read2st64_b64 v[178:181], v158 offset0:8 offset1:10
	ds_read2st64_b64 v[182:185], v158 offset0:12 offset1:14
	ds_read2st64_b64 v[186:189], v159 offset0:64 offset1:68
	ds_read2st64_b64 v[190:193], v159 offset0:72 offset1:76
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 v[118:121], v[186:187], v[170:171], v[118:121]
	s_setprio 1
	v_mfma_f32_16x16x16_f16 v[126:129], v[188:189], v[170:171], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[190:191], v[170:171], v[122:125]
	v_mfma_f32_16x16x16_f16 v[114:117], v[192:193], v[170:171], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[186:187], v[172:173], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[188:189], v[172:173], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[190:191], v[172:173], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[192:193], v[172:173], v[98:101]
	v_mfma_f32_16x16x16_f16 v[94:97], v[186:187], v[174:175], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[188:189], v[174:175], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[190:191], v[174:175], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[192:193], v[174:175], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[186:187], v[176:177], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[188:189], v[176:177], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[190:191], v[176:177], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[192:193], v[176:177], v[66:69]
	v_mfma_f32_16x16x16_f16 v[62:65], v[186:187], v[178:179], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[188:189], v[178:179], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[190:191], v[178:179], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[192:193], v[178:179], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[186:187], v[180:181], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[188:189], v[180:181], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[190:191], v[180:181], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[192:193], v[180:181], v[34:37]
	v_mfma_f32_16x16x16_f16 v[30:33], v[186:187], v[182:183], v[30:33]
	v_mfma_f32_16x16x16_f16 v[26:29], v[188:189], v[182:183], v[26:29]
	v_mfma_f32_16x16x16_f16 v[22:25], v[190:191], v[182:183], v[22:25]
	v_mfma_f32_16x16x16_f16 v[18:21], v[192:193], v[182:183], v[18:21]
	v_mfma_f32_16x16x16_f16 v[14:17], v[186:187], v[184:185], v[14:17]
	v_mfma_f32_16x16x16_f16 v[10:13], v[188:189], v[184:185], v[10:13]
	v_mfma_f32_16x16x16_f16 v[6:9], v[190:191], v[184:185], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[192:193], v[184:185], v[2:5]
	s_setprio 0
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_and_b32 s14, s5, 0xffff
	s_or_b32 s17, s14, s12
	s_mov_b32 s16, s4
	buffer_load_dwordx4 v[170:173], v152, s[16:19], 0 offen
	buffer_load_dwordx4 v[174:177], v153, s[16:19], 0 offen
	buffer_load_dwordx4 v[178:181], v154, s[16:19], 0 offen
	buffer_load_dwordx4 v[182:185], v155, s[16:19], 0 offen
	ds_read2_b64 v[186:189], v158 offset0:4 offset1:132
	ds_read2st64_b64 v[190:193], v160 offset0:4 offset1:6
	ds_read2st64_b64 v[194:197], v160 offset0:8 offset1:10
	ds_read2st64_b64 v[198:201], v160 offset0:12 offset1:14
	ds_read2st64_b64 v[202:205], v161 offset0:64 offset1:68
	ds_read2st64_b64 v[206:209], v161 offset0:72 offset1:76
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 v[118:121], v[202:203], v[186:187], v[118:121]
	s_setprio 1
	v_mfma_f32_16x16x16_f16 v[126:129], v[204:205], v[186:187], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[206:207], v[186:187], v[122:125]
	v_mfma_f32_16x16x16_f16 v[114:117], v[208:209], v[186:187], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[202:203], v[188:189], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[204:205], v[188:189], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[206:207], v[188:189], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[208:209], v[188:189], v[98:101]
	v_mfma_f32_16x16x16_f16 v[94:97], v[202:203], v[190:191], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[204:205], v[190:191], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[206:207], v[190:191], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[208:209], v[190:191], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[202:203], v[192:193], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[204:205], v[192:193], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[206:207], v[192:193], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[208:209], v[192:193], v[66:69]
	v_mfma_f32_16x16x16_f16 v[62:65], v[202:203], v[194:195], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[204:205], v[194:195], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[206:207], v[194:195], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[208:209], v[194:195], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[202:203], v[196:197], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[204:205], v[196:197], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[206:207], v[196:197], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[208:209], v[196:197], v[34:37]
	v_mfma_f32_16x16x16_f16 v[30:33], v[202:203], v[198:199], v[30:33]
	v_mfma_f32_16x16x16_f16 v[26:29], v[204:205], v[198:199], v[26:29]
	v_mfma_f32_16x16x16_f16 v[22:25], v[206:207], v[198:199], v[22:25]
	v_mfma_f32_16x16x16_f16 v[18:21], v[208:209], v[198:199], v[18:21]
	v_mfma_f32_16x16x16_f16 v[14:17], v[202:203], v[200:201], v[14:17]
	v_mfma_f32_16x16x16_f16 v[10:13], v[204:205], v[200:201], v[10:13]
	v_mfma_f32_16x16x16_f16 v[6:9], v[206:207], v[200:201], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[208:209], v[200:201], v[2:5]
	s_setprio 0
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	ds_read2_b64 v[186:189], v158 offset0:136 offset1:140
	ds_read2_b64 v[190:193], v163 offset0:8 offset1:12
	ds_read2_b64 v[194:197], v163 offset0:136 offset1:140
	ds_read2_b64 v[198:201], v164 offset0:8 offset1:12
	ds_read2_b64 v[202:205], v164 offset0:136 offset1:140
	ds_read2_b64 v[206:209], v165 offset0:8 offset1:12
	ds_read2_b64 v[210:213], v165 offset0:136 offset1:140
	ds_read2_b64 v[214:217], v166 offset0:8 offset1:12
	ds_read2_b64 v[218:221], v167 offset0:8 offset1:12
	ds_read2_b64 v[222:225], v168 offset0:8 offset1:12
	ds_read2_b64 v[226:229], v158 offset0:8 offset1:12
	ds_read2_b64 v[230:233], v169 offset0:8 offset1:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 v[118:121], v[230:231], v[226:227], v[118:121]
	s_setprio 1
	v_mfma_f32_16x16x16_f16 v[126:129], v[214:215], v[226:227], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[218:219], v[226:227], v[122:125]
	v_mfma_f32_16x16x16_f16 v[114:117], v[222:223], v[226:227], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[230:231], v[186:187], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[214:215], v[186:187], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[218:219], v[186:187], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[222:223], v[186:187], v[98:101]
	v_mfma_f32_16x16x16_f16 v[94:97], v[230:231], v[190:191], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[214:215], v[190:191], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[218:219], v[190:191], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[222:223], v[190:191], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[230:231], v[194:195], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[214:215], v[194:195], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[218:219], v[194:195], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[222:223], v[194:195], v[66:69]
	v_mfma_f32_16x16x16_f16 v[62:65], v[230:231], v[198:199], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[214:215], v[198:199], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[218:219], v[198:199], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[222:223], v[198:199], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[230:231], v[202:203], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[214:215], v[202:203], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[218:219], v[202:203], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[222:223], v[202:203], v[34:37]
	v_mfma_f32_16x16x16_f16 v[30:33], v[230:231], v[206:207], v[30:33]
	v_mfma_f32_16x16x16_f16 v[26:29], v[214:215], v[206:207], v[26:29]
	v_mfma_f32_16x16x16_f16 v[22:25], v[218:219], v[206:207], v[22:25]
	v_mfma_f32_16x16x16_f16 v[18:21], v[222:223], v[206:207], v[18:21]
	v_mfma_f32_16x16x16_f16 v[14:17], v[230:231], v[210:211], v[14:17]
	v_mfma_f32_16x16x16_f16 v[10:13], v[214:215], v[210:211], v[10:13]
	v_mfma_f32_16x16x16_f16 v[6:9], v[218:219], v[210:211], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[222:223], v[210:211], v[2:5]
	s_setprio 0
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(6)
	ds_write2st64_b64 v156, v[138:139], v[142:143] offset1:16
	s_waitcnt vmcnt(4)
	ds_write2st64_b64 v156, v[130:131], v[134:135] offset0:32 offset1:48
	ds_write2st64_b64 v157, v[140:141], v[144:145] offset1:16
	ds_write2st64_b64 v157, v[132:133], v[136:137] offset0:32 offset1:48
	s_waitcnt vmcnt(2)
	ds_write2st64_b64 v156, v[170:171], v[174:175] offset0:64 offset1:80
	s_waitcnt vmcnt(0)
	ds_write2st64_b64 v156, v[178:179], v[182:183] offset0:96 offset1:112
	ds_write2st64_b64 v157, v[172:173], v[176:177] offset0:64 offset1:80
	ds_write2st64_b64 v157, v[180:181], v[184:185] offset0:96 offset1:112
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 v[118:121], v[232:233], v[228:229], v[118:121]
	s_setprio 1
	v_mfma_f32_16x16x16_f16 v[126:129], v[216:217], v[228:229], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[220:221], v[228:229], v[122:125]
	v_mfma_f32_16x16x16_f16 v[114:117], v[224:225], v[228:229], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[232:233], v[188:189], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[216:217], v[188:189], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[220:221], v[188:189], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[224:225], v[188:189], v[98:101]
	v_mfma_f32_16x16x16_f16 v[94:97], v[232:233], v[192:193], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[216:217], v[192:193], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[220:221], v[192:193], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[224:225], v[192:193], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[232:233], v[196:197], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[216:217], v[196:197], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[220:221], v[196:197], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[224:225], v[196:197], v[66:69]
	v_mfma_f32_16x16x16_f16 v[62:65], v[232:233], v[200:201], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[216:217], v[200:201], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[220:221], v[200:201], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[224:225], v[200:201], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[232:233], v[204:205], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[216:217], v[204:205], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[220:221], v[204:205], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[224:225], v[204:205], v[34:37]
	v_mfma_f32_16x16x16_f16 v[30:33], v[232:233], v[208:209], v[30:33]
	v_mfma_f32_16x16x16_f16 v[26:29], v[216:217], v[208:209], v[26:29]
	v_mfma_f32_16x16x16_f16 v[22:25], v[220:221], v[208:209], v[22:25]
	v_mfma_f32_16x16x16_f16 v[18:21], v[224:225], v[208:209], v[18:21]
	v_mfma_f32_16x16x16_f16 v[14:17], v[232:233], v[212:213], v[14:17]
	v_mfma_f32_16x16x16_f16 v[10:13], v[216:217], v[212:213], v[10:13]
	v_mfma_f32_16x16x16_f16 v[6:9], v[220:221], v[212:213], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[224:225], v[212:213], v[2:5]
	s_setprio 0
	s_barrier
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_u32 s2, s2, 0x80
	s_addc_u32 s3, s3, 0
	s_add_i32 s11, s11, -1
	s_cmp_lg_u32 s11, 0
	s_cbranch_scc1 .LBB0_4
.LBB0_5:                                ; %._crit_edge
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_7
; %bb.6:
	s_barrier
.LBB0_7:
	s_or_b64 exec, exec, s[2:3]
	s_andn2_b64 vcc, exec, vcc
	s_cbranch_vccnz .LBB0_9
; %bb.8:
	v_lshlrev_b32_e32 v130, 5, v0
	v_and_b32_e32 v130, 0x1800, v130
	v_lshlrev_b32_e32 v131, 3, v1
	v_lshlrev_b32_e32 v138, 7, v1
	v_or3_b32 v130, v130, v131, v138
	v_and_b32_e32 v139, 24, v147
	s_movk_i32 s0, 0x878
	v_xor_b32_e32 v140, v130, v139
	v_and_or_b32 v138, v146, s0, v138
	v_xor_b32_e32 v142, 0x60, v140
	v_xor_b32_e32 v144, v138, v139
	v_add_u32_e32 v143, 0, v140
	v_add_u32_e32 v158, 0, v142
	v_xor_b32_e32 v142, 0x60, v144
	ds_read2st64_b64 v[134:137], v143 offset0:64 offset1:80
	v_add_u32_e32 v163, 0, v144
	v_xad_u32 v180, v144, 32, 0
	v_xad_u32 v181, v144, 64, 0
	v_add_u32_e32 v182, 0, v142
	ds_read2st64_b64 v[142:145], v143 offset0:96 offset1:112
	v_xad_u32 v150, v140, 32, 0
	ds_read2st64_b64 v[130:133], v150 offset0:64 offset1:80
	ds_read2st64_b64 v[164:167], v163 offset1:8
	ds_read2st64_b64 v[150:153], v150 offset0:96 offset1:112
	v_xad_u32 v154, v140, 64, 0
	ds_read2st64_b64 v[138:141], v154 offset0:64 offset1:80
	ds_read2st64_b64 v[168:171], v180 offset1:8
	ds_read2st64_b64 v[154:157], v154 offset0:96 offset1:112
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 v[118:121], v[134:135], v[164:165], v[118:121]
	ds_read2st64_b64 v[172:175], v181 offset1:8
	ds_read2st64_b64 v[146:149], v158 offset0:64 offset1:80
	ds_read2st64_b64 v[158:161], v158 offset0:96 offset1:112
	v_mfma_f32_16x16x16_f16 v[126:129], v[136:137], v[164:165], v[126:129]
	ds_read2st64_b64 v[176:179], v182 offset1:8
	v_mfma_f32_16x16x16_f16 v[122:125], v[142:143], v[164:165], v[122:125]
	v_mfma_f32_16x16x16_f16 v[114:117], v[144:145], v[164:165], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[134:135], v[166:167], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[136:137], v[166:167], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[142:143], v[166:167], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[144:145], v[166:167], v[98:101]
	ds_read2st64_b64 v[164:167], v163 offset0:16 offset1:24
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x16_f16 v[118:121], v[130:131], v[168:169], v[118:121]
	v_mfma_f32_16x16x16_f16 v[126:129], v[132:133], v[168:169], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[150:151], v[168:169], v[122:125]
	v_mfma_f32_16x16x16_f16 v[114:117], v[152:153], v[168:169], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[130:131], v[170:171], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[132:133], v[170:171], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[150:151], v[170:171], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[152:153], v[170:171], v[98:101]
	ds_read2st64_b64 v[168:171], v180 offset0:16 offset1:24
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[94:97], v[134:135], v[164:165], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[136:137], v[164:165], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[142:143], v[164:165], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[144:145], v[164:165], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[134:135], v[166:167], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[136:137], v[166:167], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[142:143], v[166:167], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[144:145], v[166:167], v[66:69]
	ds_read2st64_b64 v[164:167], v163 offset0:32 offset1:40
	v_mfma_f32_16x16x16_f16 v[118:121], v[138:139], v[172:173], v[118:121]
	v_mfma_f32_16x16x16_f16 v[126:129], v[140:141], v[172:173], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[154:155], v[172:173], v[122:125]
	v_mfma_f32_16x16x16_f16 v[114:117], v[156:157], v[172:173], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[138:139], v[174:175], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[140:141], v[174:175], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[154:155], v[174:175], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[156:157], v[174:175], v[98:101]
	ds_read2st64_b64 v[172:175], v181 offset0:16 offset1:24
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[94:97], v[130:131], v[168:169], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[132:133], v[168:169], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[150:151], v[168:169], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[152:153], v[168:169], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[130:131], v[170:171], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[132:133], v[170:171], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[150:151], v[170:171], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[152:153], v[170:171], v[66:69]
	ds_read2st64_b64 v[168:171], v180 offset0:32 offset1:40
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[62:65], v[134:135], v[164:165], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[136:137], v[164:165], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[142:143], v[164:165], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[144:145], v[164:165], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[134:135], v[166:167], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[136:137], v[166:167], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[142:143], v[166:167], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[144:145], v[166:167], v[34:37]
	ds_read2st64_b64 v[164:167], v163 offset0:48 offset1:56
	v_mfma_f32_16x16x16_f16 v[118:121], v[146:147], v[176:177], v[118:121]
	v_mfma_f32_16x16x16_f16 v[126:129], v[148:149], v[176:177], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[158:159], v[176:177], v[122:125]
	v_mfma_f32_16x16x16_f16 v[114:117], v[160:161], v[176:177], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], v[146:147], v[178:179], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], v[148:149], v[178:179], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], v[158:159], v[178:179], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], v[160:161], v[178:179], v[98:101]
	ds_read2st64_b64 v[176:179], v182 offset0:16 offset1:24
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 v[94:97], v[138:139], v[172:173], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[140:141], v[172:173], v[90:93]
	v_mfma_f32_16x16x16_f16 v[86:89], v[154:155], v[172:173], v[86:89]
	v_mfma_f32_16x16x16_f16 v[82:85], v[156:157], v[172:173], v[82:85]
	v_mfma_f32_16x16x16_f16 v[78:81], v[138:139], v[174:175], v[78:81]
	v_mfma_f32_16x16x16_f16 v[74:77], v[140:141], v[174:175], v[74:77]
	v_mfma_f32_16x16x16_f16 v[70:73], v[154:155], v[174:175], v[70:73]
	v_mfma_f32_16x16x16_f16 v[66:69], v[156:157], v[174:175], v[66:69]
	ds_read2st64_b64 v[172:175], v181 offset0:32 offset1:40
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 v[62:65], v[130:131], v[168:169], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[132:133], v[168:169], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[150:151], v[168:169], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[152:153], v[168:169], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[130:131], v[170:171], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[132:133], v[170:171], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[150:151], v[170:171], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[152:153], v[170:171], v[34:37]
	ds_read2st64_b64 v[168:171], v180 offset0:48 offset1:56
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 v[30:33], v[134:135], v[164:165], v[30:33]
	v_mfma_f32_16x16x16_f16 v[26:29], v[136:137], v[164:165], v[26:29]
	v_mfma_f32_16x16x16_f16 v[22:25], v[142:143], v[164:165], v[22:25]
	v_mfma_f32_16x16x16_f16 v[18:21], v[144:145], v[164:165], v[18:21]
	v_mfma_f32_16x16x16_f16 v[14:17], v[134:135], v[166:167], v[14:17]
	v_mfma_f32_16x16x16_f16 v[10:13], v[136:137], v[166:167], v[10:13]
	v_mfma_f32_16x16x16_f16 v[6:9], v[142:143], v[166:167], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[144:145], v[166:167], v[2:5]
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
	v_mfma_f32_16x16x16_f16 v[62:65], v[138:139], v[172:173], v[62:65]
	v_mfma_f32_16x16x16_f16 v[58:61], v[140:141], v[172:173], v[58:61]
	v_mfma_f32_16x16x16_f16 v[54:57], v[154:155], v[172:173], v[54:57]
	v_mfma_f32_16x16x16_f16 v[50:53], v[156:157], v[172:173], v[50:53]
	v_mfma_f32_16x16x16_f16 v[46:49], v[138:139], v[174:175], v[46:49]
	v_mfma_f32_16x16x16_f16 v[42:45], v[140:141], v[174:175], v[42:45]
	v_mfma_f32_16x16x16_f16 v[38:41], v[154:155], v[174:175], v[38:41]
	v_mfma_f32_16x16x16_f16 v[34:37], v[156:157], v[174:175], v[34:37]
	ds_read2st64_b64 v[172:175], v181 offset0:48 offset1:56
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[30:33], v[130:131], v[168:169], v[30:33]
	v_mfma_f32_16x16x16_f16 v[26:29], v[132:133], v[168:169], v[26:29]
	v_mfma_f32_16x16x16_f16 v[22:25], v[150:151], v[168:169], v[22:25]
	v_mfma_f32_16x16x16_f16 v[18:21], v[152:153], v[168:169], v[18:21]
	v_mfma_f32_16x16x16_f16 v[14:17], v[130:131], v[170:171], v[14:17]
	v_mfma_f32_16x16x16_f16 v[10:13], v[132:133], v[170:171], v[10:13]
	v_mfma_f32_16x16x16_f16 v[6:9], v[150:151], v[170:171], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[152:153], v[170:171], v[2:5]
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
	v_mfma_f32_16x16x16_f16 v[30:33], v[138:139], v[172:173], v[30:33]
	v_mfma_f32_16x16x16_f16 v[26:29], v[140:141], v[172:173], v[26:29]
	v_mfma_f32_16x16x16_f16 v[22:25], v[154:155], v[172:173], v[22:25]
	v_mfma_f32_16x16x16_f16 v[18:21], v[156:157], v[172:173], v[18:21]
	v_mfma_f32_16x16x16_f16 v[14:17], v[138:139], v[174:175], v[14:17]
	v_mfma_f32_16x16x16_f16 v[10:13], v[140:141], v[174:175], v[10:13]
	v_mfma_f32_16x16x16_f16 v[6:9], v[154:155], v[174:175], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[156:157], v[174:175], v[2:5]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[30:33], v[146:147], v[176:177], v[30:33]
	v_mfma_f32_16x16x16_f16 v[26:29], v[148:149], v[176:177], v[26:29]
	v_mfma_f32_16x16x16_f16 v[22:25], v[158:159], v[176:177], v[22:25]
	v_mfma_f32_16x16x16_f16 v[18:21], v[160:161], v[176:177], v[18:21]
	v_mfma_f32_16x16x16_f16 v[14:17], v[146:147], v[178:179], v[14:17]
	v_mfma_f32_16x16x16_f16 v[10:13], v[148:149], v[178:179], v[10:13]
	v_mfma_f32_16x16x16_f16 v[6:9], v[158:159], v[178:179], v[6:9]
	v_mfma_f32_16x16x16_f16 v[2:5], v[160:161], v[178:179], v[2:5]
.LBB0_9:                                ; %._crit_edge4
	v_lshrrev_b32_e32 v137, 4, v162
	v_or_b32_e32 v1, v137, v1
	v_mul_lo_u32 v145, s13, v1
	s_lshl_b32 s1, s13, 5
	v_add_u32_e32 v146, s1, v145
	v_add_u32_e32 v147, s1, v146
	v_add_u32_e32 v148, s1, v147
	v_add_u32_e32 v149, s1, v148
	v_add_u32_e32 v150, s1, v149
	s_mul_i32 s0, s21, s13
	v_add_u32_e32 v151, s1, v150
	v_add_u32_e32 v152, s1, v151
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	v_lshrrev_b32_e32 v0, 2, v0
	v_or_b32_e32 v137, s21, v1
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s21, s20, 31
	v_and_b32_e32 v130, 60, v0
	s_lshl_b64 s[0:1], s[20:21], 1
	v_or_b32_e32 v131, 0xc0, v130
	v_or_b32_e32 v132, 0x80, v130
	v_or_b32_e32 v134, 64, v130
	v_or_b32_e32 v138, 0xe0, v137
	v_or_b32_e32 v139, 0xc0, v137
	v_or_b32_e32 v140, 0xa0, v137
	v_or_b32_e32 v141, 0x80, v137
	v_or_b32_e32 v142, 0x60, v137
	v_or_b32_e32 v143, 64, v137
	v_or_b32_e32 v144, 32, v137
	v_cvt_f16_f32_e32 v118, v118
	v_cvt_f16_f32_e32 v119, v119
	s_add_u32 s28, s2, s0
	v_or_b32_e32 v0, s20, v131
	v_or_b32_e32 v133, s20, v132
	v_or_b32_e32 v135, s20, v134
	v_or_b32_e32 v136, s20, v130
	v_cvt_f16_f32_e32 v120, v120
	v_cvt_f16_f32_e32 v121, v121
	s_addc_u32 s12, s3, s1
	v_cmp_gt_i32_e64 s[24:25], s8, v137
	v_cmp_gt_i32_e64 s[18:19], s8, v144
	v_cmp_gt_i32_e64 s[16:17], s8, v143
	v_cmp_gt_i32_e64 s[14:15], s8, v142
	v_cmp_gt_i32_e64 s[20:21], s8, v141
	v_cmp_gt_i32_e64 s[10:11], s8, v140
	v_cmp_gt_i32_e64 s[22:23], s8, v139
	v_cmp_gt_i32_e32 vcc, s8, v138
	s_and_b32 s8, s13, 0x3fff
	s_bitset1_b32 s8, 14
	v_cvt_f16_f32_e32 v126, v126
	v_cvt_f16_f32_e32 v127, v127
	v_cvt_f16_f32_e32 v128, v128
	v_cvt_f16_f32_e32 v129, v129
	v_cmp_gt_i32_e64 s[6:7], s9, v136
	v_cmp_gt_i32_e64 s[4:5], s9, v135
	v_cmp_gt_i32_e64 s[2:3], s9, v133
	v_cmp_gt_i32_e64 s[0:1], s9, v0
	s_and_b32 s9, s12, 0xffff
	s_lshl_b32 s8, s8, 16
	s_or_b32 s29, s9, s8
	v_pack_b32_f16 v0, v118, v119
	v_add_lshl_u32 v118, v145, v130, 1
	v_bfrev_b32_e32 v119, 1
	s_and_b64 s[8:9], s[24:25], s[6:7]
	v_cvt_f16_f32_e32 v122, v122
	v_cvt_f16_f32_e32 v123, v123
	v_cvt_f16_f32_e32 v124, v124
	v_cvt_f16_f32_e32 v125, v125
	s_mov_b32 s31, 0x27000
	s_mov_b32 s30, 0x7ffffffe
	v_pack_b32_f16 v1, v120, v121
	v_cndmask_b32_e64 v118, v119, v118, s[8:9]
	v_cvt_f16_f32_e32 v114, v114
	v_cvt_f16_f32_e32 v115, v115
	buffer_store_dwordx2 v[0:1], v118, s[28:31], 0 offen
	v_add_lshl_u32 v118, v145, v134, 1
	s_and_b64 s[8:9], s[24:25], s[4:5]
	v_cvt_f16_f32_e32 v116, v116
	v_cvt_f16_f32_e32 v117, v117
	v_pack_b32_f16 v1, v128, v129
	v_pack_b32_f16 v0, v126, v127
	v_cndmask_b32_e64 v118, v119, v118, s[8:9]
	v_cvt_f16_f32_e32 v110, v110
	v_cvt_f16_f32_e32 v111, v111
	buffer_store_dwordx2 v[0:1], v118, s[28:31], 0 offen
	v_add_lshl_u32 v118, v145, v132, 1
	s_and_b64 s[8:9], s[24:25], s[2:3]
	v_cvt_f16_f32_e32 v112, v112
	v_cvt_f16_f32_e32 v113, v113
	v_pack_b32_f16 v1, v124, v125
	v_pack_b32_f16 v0, v122, v123
	v_cndmask_b32_e64 v118, v119, v118, s[8:9]
	v_cvt_f16_f32_e32 v106, v106
	v_cvt_f16_f32_e32 v107, v107
	buffer_store_dwordx2 v[0:1], v118, s[28:31], 0 offen
	v_pack_b32_f16 v0, v114, v115
	v_add_lshl_u32 v114, v145, v131, 1
	s_and_b64 s[8:9], s[24:25], s[0:1]
	v_cvt_f16_f32_e32 v108, v108
	v_cvt_f16_f32_e32 v109, v109
	v_pack_b32_f16 v1, v116, v117
	v_cndmask_b32_e64 v114, v119, v114, s[8:9]
	v_cvt_f16_f32_e32 v102, v102
	v_cvt_f16_f32_e32 v103, v103
	buffer_store_dwordx2 v[0:1], v114, s[28:31], 0 offen
	v_pack_b32_f16 v0, v110, v111
	v_add_lshl_u32 v110, v146, v130, 1
	s_and_b64 s[8:9], s[18:19], s[6:7]
	v_cvt_f16_f32_e32 v104, v104
	v_cvt_f16_f32_e32 v105, v105
	v_pack_b32_f16 v1, v112, v113
	v_cndmask_b32_e64 v110, v119, v110, s[8:9]
	v_cvt_f16_f32_e32 v98, v98
	v_cvt_f16_f32_e32 v99, v99
	buffer_store_dwordx2 v[0:1], v110, s[28:31], 0 offen
	v_pack_b32_f16 v0, v106, v107
	v_add_lshl_u32 v106, v146, v134, 1
	s_and_b64 s[8:9], s[18:19], s[4:5]
	v_cvt_f16_f32_e32 v100, v100
	v_cvt_f16_f32_e32 v101, v101
	v_pack_b32_f16 v1, v108, v109
	v_cndmask_b32_e64 v106, v119, v106, s[8:9]
	v_cvt_f16_f32_e32 v94, v94
	v_cvt_f16_f32_e32 v95, v95
	buffer_store_dwordx2 v[0:1], v106, s[28:31], 0 offen
	v_pack_b32_f16 v0, v102, v103
	v_add_lshl_u32 v102, v146, v132, 1
	s_and_b64 s[8:9], s[18:19], s[2:3]
	v_cvt_f16_f32_e32 v96, v96
	v_cvt_f16_f32_e32 v97, v97
	v_pack_b32_f16 v1, v104, v105
	v_cndmask_b32_e64 v102, v119, v102, s[8:9]
	v_cvt_f16_f32_e32 v90, v90
	v_cvt_f16_f32_e32 v91, v91
	buffer_store_dwordx2 v[0:1], v102, s[28:31], 0 offen
	v_pack_b32_f16 v0, v98, v99
	v_add_lshl_u32 v98, v146, v131, 1
	s_and_b64 s[8:9], s[18:19], s[0:1]
	v_cvt_f16_f32_e32 v92, v92
	v_cvt_f16_f32_e32 v93, v93
	v_pack_b32_f16 v1, v100, v101
	v_cndmask_b32_e64 v98, v119, v98, s[8:9]
	v_cvt_f16_f32_e32 v86, v86
	v_cvt_f16_f32_e32 v87, v87
	buffer_store_dwordx2 v[0:1], v98, s[28:31], 0 offen
	v_pack_b32_f16 v0, v94, v95
	v_add_lshl_u32 v94, v147, v130, 1
	s_and_b64 s[8:9], s[16:17], s[6:7]
	v_cvt_f16_f32_e32 v88, v88
	v_cvt_f16_f32_e32 v89, v89
	v_pack_b32_f16 v1, v96, v97
	v_cndmask_b32_e64 v94, v119, v94, s[8:9]
	v_cvt_f16_f32_e32 v82, v82
	v_cvt_f16_f32_e32 v83, v83
	buffer_store_dwordx2 v[0:1], v94, s[28:31], 0 offen
	v_pack_b32_f16 v0, v90, v91
	v_add_lshl_u32 v90, v147, v134, 1
	s_and_b64 s[8:9], s[16:17], s[4:5]
	v_cvt_f16_f32_e32 v84, v84
	v_cvt_f16_f32_e32 v85, v85
	v_pack_b32_f16 v1, v92, v93
	v_cndmask_b32_e64 v90, v119, v90, s[8:9]
	v_cvt_f16_f32_e32 v78, v78
	v_cvt_f16_f32_e32 v79, v79
	buffer_store_dwordx2 v[0:1], v90, s[28:31], 0 offen
	v_pack_b32_f16 v0, v86, v87
	v_add_lshl_u32 v86, v147, v132, 1
	s_and_b64 s[8:9], s[16:17], s[2:3]
	v_cvt_f16_f32_e32 v80, v80
	v_cvt_f16_f32_e32 v81, v81
	v_pack_b32_f16 v1, v88, v89
	v_cndmask_b32_e64 v86, v119, v86, s[8:9]
	v_cvt_f16_f32_e32 v74, v74
	v_cvt_f16_f32_e32 v75, v75
	buffer_store_dwordx2 v[0:1], v86, s[28:31], 0 offen
	v_pack_b32_f16 v0, v82, v83
	v_add_lshl_u32 v82, v147, v131, 1
	s_and_b64 s[8:9], s[16:17], s[0:1]
	v_cvt_f16_f32_e32 v76, v76
	v_cvt_f16_f32_e32 v77, v77
	v_pack_b32_f16 v1, v84, v85
	v_cndmask_b32_e64 v82, v119, v82, s[8:9]
	v_cvt_f16_f32_e32 v70, v70
	v_cvt_f16_f32_e32 v71, v71
	buffer_store_dwordx2 v[0:1], v82, s[28:31], 0 offen
	v_pack_b32_f16 v0, v78, v79
	v_add_lshl_u32 v78, v148, v130, 1
	s_and_b64 s[8:9], s[14:15], s[6:7]
	v_cvt_f16_f32_e32 v72, v72
	v_cvt_f16_f32_e32 v73, v73
	v_pack_b32_f16 v1, v80, v81
	v_cndmask_b32_e64 v78, v119, v78, s[8:9]
	v_cvt_f16_f32_e32 v66, v66
	v_cvt_f16_f32_e32 v67, v67
	buffer_store_dwordx2 v[0:1], v78, s[28:31], 0 offen
	v_pack_b32_f16 v0, v74, v75
	v_add_lshl_u32 v74, v148, v134, 1
	s_and_b64 s[8:9], s[14:15], s[4:5]
	v_cvt_f16_f32_e32 v68, v68
	v_cvt_f16_f32_e32 v69, v69
	v_pack_b32_f16 v1, v76, v77
	v_cndmask_b32_e64 v74, v119, v74, s[8:9]
	v_cvt_f16_f32_e32 v62, v62
	v_cvt_f16_f32_e32 v63, v63
	buffer_store_dwordx2 v[0:1], v74, s[28:31], 0 offen
	v_pack_b32_f16 v0, v70, v71
	v_add_lshl_u32 v70, v148, v132, 1
	s_and_b64 s[8:9], s[14:15], s[2:3]
	v_cvt_f16_f32_e32 v64, v64
	v_cvt_f16_f32_e32 v65, v65
	v_pack_b32_f16 v1, v72, v73
	v_cndmask_b32_e64 v70, v119, v70, s[8:9]
	v_cvt_f16_f32_e32 v58, v58
	v_cvt_f16_f32_e32 v59, v59
	buffer_store_dwordx2 v[0:1], v70, s[28:31], 0 offen
	v_pack_b32_f16 v0, v66, v67
	v_add_lshl_u32 v66, v148, v131, 1
	s_and_b64 s[8:9], s[14:15], s[0:1]
	v_cvt_f16_f32_e32 v60, v60
	v_cvt_f16_f32_e32 v61, v61
	v_pack_b32_f16 v1, v68, v69
	v_cndmask_b32_e64 v66, v119, v66, s[8:9]
	v_cvt_f16_f32_e32 v54, v54
	v_cvt_f16_f32_e32 v55, v55
	buffer_store_dwordx2 v[0:1], v66, s[28:31], 0 offen
	v_pack_b32_f16 v0, v62, v63
	v_add_lshl_u32 v62, v149, v130, 1
	s_and_b64 s[8:9], s[20:21], s[6:7]
	v_cvt_f16_f32_e32 v56, v56
	v_cvt_f16_f32_e32 v57, v57
	v_pack_b32_f16 v1, v64, v65
	v_cndmask_b32_e64 v62, v119, v62, s[8:9]
	v_cvt_f16_f32_e32 v50, v50
	v_cvt_f16_f32_e32 v51, v51
	buffer_store_dwordx2 v[0:1], v62, s[28:31], 0 offen
	v_pack_b32_f16 v0, v58, v59
	v_add_lshl_u32 v58, v149, v134, 1
	s_and_b64 s[8:9], s[20:21], s[4:5]
	v_cvt_f16_f32_e32 v52, v52
	v_cvt_f16_f32_e32 v53, v53
	v_pack_b32_f16 v1, v60, v61
	v_cndmask_b32_e64 v58, v119, v58, s[8:9]
	v_cvt_f16_f32_e32 v46, v46
	v_cvt_f16_f32_e32 v47, v47
	buffer_store_dwordx2 v[0:1], v58, s[28:31], 0 offen
	v_pack_b32_f16 v0, v54, v55
	v_add_lshl_u32 v54, v149, v132, 1
	s_and_b64 s[8:9], s[20:21], s[2:3]
	v_cvt_f16_f32_e32 v48, v48
	v_cvt_f16_f32_e32 v49, v49
	v_pack_b32_f16 v1, v56, v57
	v_cndmask_b32_e64 v54, v119, v54, s[8:9]
	v_cvt_f16_f32_e32 v42, v42
	v_cvt_f16_f32_e32 v43, v43
	buffer_store_dwordx2 v[0:1], v54, s[28:31], 0 offen
	v_pack_b32_f16 v0, v50, v51
	v_add_lshl_u32 v50, v149, v131, 1
	s_and_b64 s[8:9], s[20:21], s[0:1]
	v_cvt_f16_f32_e32 v44, v44
	v_cvt_f16_f32_e32 v45, v45
	v_pack_b32_f16 v1, v52, v53
	v_cndmask_b32_e64 v50, v119, v50, s[8:9]
	v_cvt_f16_f32_e32 v38, v38
	v_cvt_f16_f32_e32 v39, v39
	buffer_store_dwordx2 v[0:1], v50, s[28:31], 0 offen
	v_pack_b32_f16 v0, v46, v47
	v_add_lshl_u32 v46, v150, v130, 1
	s_and_b64 s[8:9], s[10:11], s[6:7]
	v_cvt_f16_f32_e32 v40, v40
	v_cvt_f16_f32_e32 v41, v41
	v_pack_b32_f16 v1, v48, v49
	v_cndmask_b32_e64 v46, v119, v46, s[8:9]
	v_cvt_f16_f32_e32 v34, v34
	v_cvt_f16_f32_e32 v35, v35
	buffer_store_dwordx2 v[0:1], v46, s[28:31], 0 offen
	v_pack_b32_f16 v0, v42, v43
	v_add_lshl_u32 v42, v150, v134, 1
	s_and_b64 s[8:9], s[10:11], s[4:5]
	v_cvt_f16_f32_e32 v36, v36
	v_cvt_f16_f32_e32 v37, v37
	v_pack_b32_f16 v1, v44, v45
	v_cndmask_b32_e64 v42, v119, v42, s[8:9]
	v_cvt_f16_f32_e32 v30, v30
	v_cvt_f16_f32_e32 v31, v31
	buffer_store_dwordx2 v[0:1], v42, s[28:31], 0 offen
	v_pack_b32_f16 v0, v38, v39
	v_add_lshl_u32 v38, v150, v132, 1
	s_and_b64 s[8:9], s[10:11], s[2:3]
	v_cvt_f16_f32_e32 v32, v32
	v_cvt_f16_f32_e32 v33, v33
	v_pack_b32_f16 v1, v40, v41
	v_cndmask_b32_e64 v38, v119, v38, s[8:9]
	v_cvt_f16_f32_e32 v26, v26
	v_cvt_f16_f32_e32 v27, v27
	buffer_store_dwordx2 v[0:1], v38, s[28:31], 0 offen
	v_pack_b32_f16 v0, v34, v35
	v_add_lshl_u32 v34, v150, v131, 1
	s_and_b64 s[8:9], s[10:11], s[0:1]
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v29, v29
	v_pack_b32_f16 v1, v36, v37
	v_cndmask_b32_e64 v34, v119, v34, s[8:9]
	v_cvt_f16_f32_e32 v22, v22
	v_cvt_f16_f32_e32 v23, v23
	buffer_store_dwordx2 v[0:1], v34, s[28:31], 0 offen
	v_pack_b32_f16 v0, v30, v31
	v_add_lshl_u32 v30, v151, v130, 1
	s_and_b64 s[8:9], s[22:23], s[6:7]
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v25, v25
	v_pack_b32_f16 v1, v32, v33
	v_cndmask_b32_e64 v30, v119, v30, s[8:9]
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	buffer_store_dwordx2 v[0:1], v30, s[28:31], 0 offen
	v_pack_b32_f16 v0, v26, v27
	v_add_lshl_u32 v26, v151, v134, 1
	s_and_b64 s[8:9], s[22:23], s[4:5]
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v21, v21
	v_pack_b32_f16 v1, v28, v29
	v_cndmask_b32_e64 v26, v119, v26, s[8:9]
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v15, v15
	buffer_store_dwordx2 v[0:1], v26, s[28:31], 0 offen
	v_pack_b32_f16 v0, v22, v23
	v_add_lshl_u32 v22, v151, v132, 1
	s_and_b64 s[8:9], s[22:23], s[2:3]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_pack_b32_f16 v1, v24, v25
	v_cndmask_b32_e64 v22, v119, v22, s[8:9]
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v11, v11
	buffer_store_dwordx2 v[0:1], v22, s[28:31], 0 offen
	v_pack_b32_f16 v0, v18, v19
	v_add_lshl_u32 v18, v151, v131, 1
	s_and_b64 s[8:9], s[22:23], s[0:1]
	v_cvt_f16_f32_e32 v12, v12
	v_cvt_f16_f32_e32 v13, v13
	v_pack_b32_f16 v1, v20, v21
	v_cndmask_b32_e64 v18, v119, v18, s[8:9]
	v_cvt_f16_f32_e32 v6, v6
	v_cvt_f16_f32_e32 v7, v7
	buffer_store_dwordx2 v[0:1], v18, s[28:31], 0 offen
	v_pack_b32_f16 v0, v14, v15
	v_add_lshl_u32 v14, v152, v130, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v9, v9
	v_pack_b32_f16 v1, v16, v17
	v_cndmask_b32_e64 v14, v119, v14, s[6:7]
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	buffer_store_dwordx2 v[0:1], v14, s[28:31], 0 offen
	v_pack_b32_f16 v0, v10, v11
	v_add_lshl_u32 v10, v152, v134, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v5, v5
	v_pack_b32_f16 v1, v12, v13
	v_cndmask_b32_e64 v10, v119, v10, s[4:5]
	buffer_store_dwordx2 v[0:1], v10, s[28:31], 0 offen
	v_pack_b32_f16 v0, v6, v7
	v_add_lshl_u32 v6, v152, v132, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_pack_b32_f16 v1, v8, v9
	v_cndmask_b32_e64 v6, v119, v6, s[2:3]
	buffer_store_dwordx2 v[0:1], v6, s[28:31], 0 offen
	v_pack_b32_f16 v0, v2, v3
	v_add_lshl_u32 v2, v152, v131, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_pack_b32_f16 v1, v4, v5
	v_cndmask_b32_e32 v2, v119, v2, vcc
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
		.amdhsa_next_free_vgpr 234
		.amdhsa_next_free_sgpr 32
		.amdhsa_accum_offset 236
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
	.set matmul_kernel.num_vgpr, 234
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
; codeLenInByte = 7580
; TotalNumSgprs: 38
; NumVgprs: 234
; NumAgprs: 0
; TotalNumVgprs: 234
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 38
; NumVGPRsForWavesPerEU: 234
; AccumOffset: 236
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 58
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
    .vgpr_count:     234
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
