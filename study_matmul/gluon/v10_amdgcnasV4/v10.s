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
s_load_dwordx2 s[2:3], s[0:1], 0x0
s_load_dwordx8 s[4:11], s[0:1], 0x8
s_load_dwordx4 s[12:15], s[0:1], 0x28
s_waitcnt lgkmcnt(0)
s_branch .LBB0_0
.p2align 8
; %bb.4:
.LBB0_0:
v_mov_b32_e32 v11, v0
s_nop 0
v_readfirstlane_b32 s51, v11
s_bfe_u32 s1, s51, 0x20006
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
v_accvgpr_write_b32 a153, v0
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
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
s_add_i32 s49, s48, 0x1080
s_add_i32 s5, s4, 0x1080
v_lshlrev_b32_e32 v13, 1, v3
s_mov_b32 m0, s5
s_nop 0
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s50, s48, 0x2100
s_add_i32 s9, s4, 0x2100
v_lshlrev_b32_e32 v14, 1, v4
s_mov_b32 m0, s9
s_nop 0
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_add_i32 s53, s48, 0x3180
s_add_i32 s11, s4, 0x3180
v_lshlrev_b32_e32 v15, 1, v5
s_mov_b32 m0, s11
s_nop 0
buffer_load_dwordx4 v15, s[16:19], 0, offen, lds
s_add_i32 s13, s4, 0x4200
v_lshlrev_b32_e32 v6, 1, v6
s_mov_b32 m0, s13
s_nop 0
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_add_i32 s24, s4, 0x5280
v_lshlrev_b32_e32 v7, 1, v7
s_mov_b32 m0, s24
s_nop 0
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
s_add_i32 s25, s4, 0x6300
v_lshlrev_b32_e32 v16, 1, v8
s_mov_b32 m0, s25
s_nop 0
buffer_load_dwordx4 v16, s[16:19], 0, offen, lds
s_add_i32 s26, s4, 0x7380
v_add_lshl_u32 v8, v8, s15, 1
s_mov_b32 m0, s26
s_nop 0
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
s_and_b32 s21, s14, 0xffff
s_mov_b32 s22, s18
s_mov_b32 s23, s19
s_add_i32 s54, 0, 0x107e0
s_add_i32 s27, s54, s48
v_lshlrev_b32_e32 v2, 1, v0
s_mov_b32 m0, s27
s_nop 0
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_i32 s28, s54, s49
v_lshlrev_b32_e32 v3, 1, v1
s_mov_b32 m0, s28
s_nop 0
buffer_load_dwordx4 v3, s[20:23], 0, offen, lds
s_add_i32 s29, s54, s50
v_lshlrev_b32_e32 v4, 1, v9
s_mov_b32 m0, s29
s_nop 0
buffer_load_dwordx4 v4, s[20:23], 0, offen, lds
s_add_i32 s30, s54, s53
v_lshlrev_b32_e32 v5, 1, v10
s_mov_b32 m0, s30
s_nop 0
buffer_load_dwordx4 v5, s[20:23], 0, offen, lds
s_add_u32 s16, s16, 0x80
s_addc_u32 s15, s31, 0
s_add_i32 s35, 0, 0x18bc0
s_add_i32 s31, s35, s48
v_add_lshl_u32 v0, v0, s36, 1
s_mov_b32 m0, s31
s_nop 0
buffer_load_dwordx4 v0, s[20:23], 0, offen, lds
s_add_i32 s33, s35, s49
v_add_lshl_u32 v1, v1, s36, 1
s_mov_b32 m0, s33
s_nop 0
buffer_load_dwordx4 v1, s[20:23], 0, offen, lds
s_add_i32 s34, s35, s50
v_add_lshl_u32 v9, v9, s36, 1
s_mov_b32 m0, s34
s_nop 0
buffer_load_dwordx4 v9, s[20:23], 0, offen, lds
s_add_i32 s35, s35, s53
v_add_lshl_u32 v10, v10, s36, 1
s_mov_b32 m0, s35
s_nop 0
buffer_load_dwordx4 v10, s[20:23], 0, offen, lds
s_waitcnt lgkmcnt(0)
s_barrier
s_and_b32 s17, s15, 0xffff
s_add_i32 s36, s4, 0x8400
s_mov_b32 m0, s36
v_accvgpr_write_b32 a162, v12
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
s_add_i32 s37, s4, 0x9480
s_mov_b32 m0, s37
v_accvgpr_write_b32 a163, v13
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s38, s4, 0xa500
s_mov_b32 m0, s38
v_accvgpr_write_b32 a164, v14
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_add_i32 s39, s4, 0xb580
s_mov_b32 m0, s39
v_accvgpr_write_b32 a165, v15
buffer_load_dwordx4 v15, s[16:19], 0, offen, lds
s_add_i32 s40, s4, 0xc600
s_mov_b32 m0, s40
v_accvgpr_write_b32 a166, v6
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_add_i32 s41, s4, 0xd680
s_mov_b32 m0, s41
v_accvgpr_write_b32 a167, v7
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
s_add_i32 s42, s4, 0xe700
s_mov_b32 m0, s42
v_accvgpr_write_b32 a168, v16
buffer_load_dwordx4 v16, s[16:19], 0, offen, lds
s_add_i32 s43, s4, 0xf780
s_mov_b32 m0, s43
v_accvgpr_write_b32 a169, v8
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
s_add_i32 s47, 0, 0x149e0
s_add_i32 s44, s47, s48
v_add_u32_e32 v2, 0x80, v2
s_mov_b32 m0, s44
v_accvgpr_write_b32 a174, v2
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_i32 s45, s47, s49
v_add_u32_e32 v2, 0x80, v3
s_mov_b32 m0, s45
v_accvgpr_write_b32 a175, v2
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_i32 s46, s47, s50
v_add_u32_e32 v2, 0x80, v4
s_mov_b32 m0, s46
v_accvgpr_write_b32 a177, v2
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_i32 s47, s47, s53
v_add_u32_e32 v2, 0x80, v5
s_mov_b32 m0, s47
v_accvgpr_write_b32 a178, v2
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_u32 s20, s20, 0x80
s_addc_u32 s22, s14, 0
s_and_b32 s17, s22, 0xffff
s_mov_b32 s16, s20
s_add_i32 s14, 0, 0x1cdc0
s_add_i32 s23, s14, s48
s_mov_b32 m0, s23
v_accvgpr_write_b32 a170, v0
buffer_load_dwordx4 v0, s[16:19], 0, offen, lds
s_add_i32 s48, s14, s49
s_mov_b32 m0, s48
v_accvgpr_write_b32 a171, v1
buffer_load_dwordx4 v1, s[16:19], 0, offen, lds
s_add_i32 s49, s14, s50
s_mov_b32 m0, s49
v_accvgpr_write_b32 a172, v9
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
s_add_i32 s50, s14, s53
s_mov_b32 m0, s50
v_accvgpr_write_b32 a173, v10
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
v_and_b32_e32 v3, 15, v11
v_lshlrev_b32_e32 v1, 10, v3
v_accvgpr_write_b32 a152, v11
v_and_b32_e32 v2, 48, v11
s_cmp_lt_u32 s1, 2
s_cselect_b64 s[14:15], -1, 0
s_and_b64 s[16:17], s[14:15], exec
s_cselect_b32 s16, 0, 0x100
v_or3_b32 v0, v2, s16, v1
v_accvgpr_write_b32 a176, v3
v_lshlrev_b32_e32 v3, 5, v3
v_add_u32_e32 v0, v0, v3
v_add_u32_e32 v254, 0, v0
ds_read_b128 a[56:59], v254
ds_read_b128 a[60:63], v254, offset:64
ds_read_b128 a[48:51], v254, offset:128
ds_read_b128 a[52:55], v254, offset:192
ds_read_b128 a[40:43], v254, offset:512
ds_read_b128 a[44:47], v254, offset:576
ds_read_b128 a[32:35], v254, offset:640
ds_read_b128 a[36:39], v254, offset:704
ds_read_b128 a[24:27], v254, offset:16896
ds_read_b128 a[28:31], v254, offset:16960
ds_read_b128 a[16:19], v254, offset:17024
ds_read_b128 a[20:23], v254, offset:17088
ds_read_b128 a[8:11], v254, offset:17408
ds_read_b128 a[12:15], v254, offset:17472
ds_read_b128 a[0:3], v254, offset:17536
ds_read_b128 a[4:7], v254, offset:17600
s_and_b32 s21, s51, 64
v_or_b32_e32 v1, v1, v2
v_add_u32_e32 v1, v1, v3
v_lshl_add_u32 v0, s21, 2, v1
v_accvgpr_write_b32 a179, v0
v_add_u32_e32 v1, s54, v0
ds_read_b128 a[68:71], v1
ds_read_b128 a[76:79], v1, offset:64
ds_read_b128 a[64:67], v1, offset:128
ds_read_b128 a[72:75], v1, offset:192
ds_read_b128 a[80:83], v1, offset:512
ds_read_b128 a[84:87], v1, offset:576
ds_read_b128 a[88:91], v1, offset:640
ds_read_b128 a[92:95], v1, offset:704
s_mul_i32 s10, s10, s52
s_lshl_b32 s16, s10, 8
s_ashr_i32 s17, s16, 31
s_lshl_b64 s[16:17], s[16:17], 1
s_add_u32 s2, s2, s16
s_addc_u32 s3, s3, s17
s_add_u32 s2, s2, 0x180
s_addc_u32 s3, s3, 0
s_mov_b32 s10, -2
v_mov_b32_e32 v98, 0
v_mov_b32_e32 v99, 0
v_mov_b32_e32 v100, 0
v_mov_b32_e32 v101, 0
v_mov_b32_e32 v102, 0
v_mov_b32_e32 v103, 0
v_mov_b32_e32 v104, 0
v_mov_b32_e32 v105, 0
v_mov_b32_e32 v106, 0
v_mov_b32_e32 v107, 0
v_mov_b32_e32 v108, 0
v_mov_b32_e32 v109, 0
v_mov_b32_e32 v110, 0
v_mov_b32_e32 v111, 0
v_mov_b32_e32 v112, 0
v_mov_b32_e32 v113, 0
v_mov_b32_e32 v114, 0
v_mov_b32_e32 v115, 0
v_mov_b32_e32 v116, 0
v_mov_b32_e32 v117, 0
v_mov_b32_e32 v118, 0
v_mov_b32_e32 v119, 0
v_mov_b32_e32 v120, 0
v_mov_b32_e32 v121, 0
v_mov_b32_e32 v122, 0
v_mov_b32_e32 v123, 0
v_mov_b32_e32 v124, 0
v_mov_b32_e32 v125, 0
v_mov_b32_e32 v126, 0
v_mov_b32_e32 v127, 0
v_mov_b32_e32 v128, 0
v_mov_b32_e32 v129, 0
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
v_mov_b32_e32 v80, 0
v_mov_b32_e32 v81, 0
v_mov_b32_e32 v82, 0
v_mov_b32_e32 v83, 0
v_mov_b32_e32 v84, 0
v_mov_b32_e32 v85, 0
v_mov_b32_e32 v86, 0
v_mov_b32_e32 v87, 0
v_mov_b32_e32 v88, 0
v_mov_b32_e32 v89, 0
v_mov_b32_e32 v90, 0
v_mov_b32_e32 v91, 0
v_mov_b32_e32 v92, 0
v_mov_b32_e32 v93, 0
v_mov_b32_e32 v94, 0
v_mov_b32_e32 v95, 0
v_mov_b32_e32 v96, 0
v_mov_b32_e32 v97, 0
v_accvgpr_write_b32 a235, 0
v_accvgpr_write_b32 a236, 0
v_accvgpr_write_b32 a237, 0
v_accvgpr_write_b32 a234, 0
v_accvgpr_write_b32 a230, 0
v_accvgpr_write_b32 a231, 0
v_accvgpr_write_b32 a232, 0
v_accvgpr_write_b32 a233, 0
v_accvgpr_write_b32 a227, 0
v_accvgpr_write_b32 a228, 0
v_accvgpr_write_b32 a229, 0
v_accvgpr_write_b32 a226, 0
v_accvgpr_write_b32 a222, 0
v_accvgpr_write_b32 a223, 0
v_accvgpr_write_b32 a224, 0
v_accvgpr_write_b32 a225, 0
v_accvgpr_write_b32 a219, 0
v_accvgpr_write_b32 a220, 0
v_accvgpr_write_b32 a221, 0
v_accvgpr_write_b32 a218, 0
v_accvgpr_write_b32 a214, 0
v_accvgpr_write_b32 a215, 0
v_accvgpr_write_b32 a216, 0
v_accvgpr_write_b32 a217, 0
v_accvgpr_write_b32 a211, 0
v_accvgpr_write_b32 a212, 0
v_accvgpr_write_b32 a213, 0
v_accvgpr_write_b32 a210, 0
v_accvgpr_write_b32 a192, 0
v_accvgpr_write_b32 a193, 0
v_accvgpr_write_b32 a194, 0
v_accvgpr_write_b32 a195, 0
v_accvgpr_write_b32 a154, 0
v_accvgpr_write_b32 a155, 0
v_accvgpr_write_b32 a156, 0
v_accvgpr_write_b32 a157, 0
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
v_accvgpr_write_b32 a158, 0
v_accvgpr_write_b32 a159, 0
v_accvgpr_write_b32 a160, 0
v_accvgpr_write_b32 a161, 0
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
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
v_mov_b32_e32 v18, 0
v_mov_b32_e32 v19, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
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
v_accvgpr_read_b32 v35, a179
v_add_u32_e32 v255, 0, v35
v_add_u32_e32 v37, 0x18bc0, v255
v_accvgpr_read_b32 v36, a162
v_bfrev_b32_e32 v38, 1
v_accvgpr_read_b32 v42, a163
v_accvgpr_read_b32 v20, a164
v_accvgpr_read_b32 v43, a165
v_accvgpr_read_b32 v15, a166
v_accvgpr_read_b32 v44, a167
v_accvgpr_read_b32 v34, a168
v_accvgpr_read_b32 v45, a169
v_accvgpr_read_b32 v41, a174
v_accvgpr_read_b32 v21, a175
v_accvgpr_read_b32 v39, a177
v_accvgpr_read_b32 v47, a178
v_add_u32_e32 v6, 0x149e0, v255
v_accvgpr_read_b32 v14, a170
v_accvgpr_read_b32 v40, a171
v_accvgpr_read_b32 v8, a172
v_accvgpr_read_b32 v46, a173
v_add_u32_e32 v9, 0x1cdc0, v255
s_mov_b32 s54, s18
s_mov_b32 s55, s19
v_add_u32_e32 v230, 0x107e0, v255
s_waitcnt lgkmcnt(0)
.LBB0_1:
v_mfma_f32_16x16x32_f16 v[16:19], a[68:71], a[56:59], v[16:19]
s_add_u32 s16, s2, 0xffffff80
s_addc_u32 s17, s3, -1
v_mfma_f32_16x16x32_f16 v[16:19], a[76:79], a[60:63], v[16:19]
v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[56:59], v[10:13]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[60:63], v[10:13]
s_barrier
ds_read_b128 a[96:99], v37
v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[56:59], v[234:237]
ds_read_b128 a[100:103], v37, offset:64
v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[60:63], v[234:237]
ds_read_b128 a[104:107], v37, offset:128
v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[56:59], v[238:241]
ds_read_b128 a[108:111], v37, offset:192
v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[60:63], v[238:241]
ds_read_b128 a[112:115], v37, offset:512
v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[48:51], v[242:245]
ds_read_b128 a[116:119], v37, offset:576
v_mfma_f32_16x16x32_f16 v[242:245], a[76:79], a[52:55], v[242:245]
ds_read_b128 a[120:123], v37, offset:640
v_mfma_f32_16x16x32_f16 v[246:249], a[64:67], a[48:51], v[246:249]
ds_read_b128 a[124:127], v37, offset:704
v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[52:55], v[246:249]
s_cmp_eq_u32 s10, 60
s_cselect_b64 vcc, -1, 0
v_mfma_f32_16x16x32_f16 v[2:5], a[80:83], a[48:51], v[2:5]
s_and_b32 s17, s17, 0xffff
v_cndmask_b32_e32 v36, v36, v38, vcc
v_mfma_f32_16x16x32_f16 v[2:5], a[84:87], a[52:55], v[2:5]
s_mov_b32 m0, s4
v_mfma_f32_16x16x32_f16 v[250:253], a[88:91], a[48:51], v[250:253]
buffer_load_dwordx4 v36, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[250:253], a[92:95], a[52:55], v[250:253]
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[40:43], v[194:197]
v_mfma_f32_16x16x32_f16 v[194:197], a[76:79], a[44:47], v[194:197]
v_cndmask_b32_e32 v42, v42, v38, vcc
s_mov_b32 m0, s5
v_mfma_f32_16x16x32_f16 v[198:201], a[64:67], a[40:43], v[198:201]
buffer_load_dwordx4 v42, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[44:47], v[198:201]
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[40:43], v[202:205]
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[44:47], v[202:205]
v_cndmask_b32_e32 v20, v20, v38, vcc
s_mov_b32 m0, s9
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[40:43], v[206:209]
buffer_load_dwordx4 v20, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[44:47], v[206:209]
v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[32:35], v[210:213]
v_mfma_f32_16x16x32_f16 v[210:213], a[76:79], a[36:39], v[210:213]
v_cndmask_b32_e32 v43, v43, v38, vcc
s_mov_b32 m0, s11
v_mfma_f32_16x16x32_f16 v[214:217], a[64:67], a[32:35], v[214:217]
buffer_load_dwordx4 v43, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[36:39], v[214:217]
v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[32:35], v[218:221]
v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[36:39], v[218:221]
v_cndmask_b32_e32 v15, v15, v38, vcc
s_mov_b32 m0, s13
v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[32:35], v[222:225]
buffer_load_dwordx4 v15, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[36:39], v[222:225]
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[24:27], v[162:165]
v_mfma_f32_16x16x32_f16 v[162:165], a[76:79], a[28:31], v[162:165]
v_cndmask_b32_e32 v44, v44, v38, vcc
s_mov_b32 m0, s24
v_mfma_f32_16x16x32_f16 v[166:169], a[64:67], a[24:27], v[166:169]
buffer_load_dwordx4 v44, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[28:31], v[166:169]
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[24:27], v[170:173]
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[28:31], v[170:173]
v_cndmask_b32_e32 v34, v34, v38, vcc
s_mov_b32 m0, s25
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[24:27], v[174:177]
buffer_load_dwordx4 v34, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[28:31], v[174:177]
v_mfma_f32_16x16x32_f16 v[182:185], a[68:71], a[16:19], v[182:185]
v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[20:23], v[182:185]
v_cndmask_b32_e32 v45, v45, v38, vcc
s_mov_b32 m0, s26
v_mfma_f32_16x16x32_f16 v[186:189], a[64:67], a[16:19], v[186:189]
buffer_load_dwordx4 v45, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[186:189], a[72:75], a[20:23], v[186:189]
s_and_b32 s17, s22, 0xffff
s_mov_b32 s16, s20
v_mfma_f32_16x16x32_f16 v[190:193], a[80:83], a[16:19], v[190:193]
v_cndmask_b32_e32 v41, v41, v38, vcc
s_mov_b32 m0, s27
v_mfma_f32_16x16x32_f16 v[190:193], a[84:87], a[20:23], v[190:193]
v_mfma_f32_16x16x32_f16 v[178:181], a[88:91], a[16:19], v[178:181]
buffer_load_dwordx4 v41, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[178:181], a[92:95], a[20:23], v[178:181]
v_mfma_f32_16x16x32_f16 v[130:133], a[68:71], a[8:11], v[130:133]
v_mfma_f32_16x16x32_f16 v[130:133], a[76:79], a[12:15], v[130:133]
v_cndmask_b32_e32 v21, v21, v38, vcc
s_mov_b32 m0, s28
v_mfma_f32_16x16x32_f16 v[134:137], a[64:67], a[8:11], v[134:137]
buffer_load_dwordx4 v21, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], a[12:15], v[134:137]
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[8:11], v[138:141]
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], a[12:15], v[138:141]
v_cndmask_b32_e32 v39, v39, v38, vcc
s_mov_b32 m0, s29
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[8:11], v[142:145]
buffer_load_dwordx4 v39, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[12:15], v[142:145]
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[0:3], v[146:149]
v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], a[4:7], v[146:149]
v_cndmask_b32_e32 v47, v47, v38, vcc
s_mov_b32 m0, s30
v_mfma_f32_16x16x32_f16 v[150:153], a[64:67], a[0:3], v[150:153]
buffer_load_dwordx4 v47, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[4:7], v[150:153]
v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], a[0:3], v[154:157]
v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], a[4:7], v[154:157]
v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], a[0:3], v[158:161]
v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[4:7], v[158:161]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[98:101], a[96:99], a[56:59], v[98:101]
v_mfma_f32_16x16x32_f16 v[98:101], a[100:103], a[60:63], v[98:101]
v_mfma_f32_16x16x32_f16 v[102:105], a[104:107], a[56:59], v[102:105]
v_mfma_f32_16x16x32_f16 v[102:105], a[108:111], a[60:63], v[102:105]
v_mfma_f32_16x16x32_f16 v[106:109], a[112:115], a[56:59], v[106:109]
v_mfma_f32_16x16x32_f16 v[106:109], a[116:119], a[60:63], v[106:109]
v_mfma_f32_16x16x32_f16 v[110:113], a[120:123], a[56:59], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[124:127], a[60:63], v[110:113]
v_mfma_f32_16x16x32_f16 v[114:117], a[96:99], a[48:51], v[114:117]
v_mfma_f32_16x16x32_f16 v[114:117], a[100:103], a[52:55], v[114:117]
v_mfma_f32_16x16x32_f16 v[118:121], a[104:107], a[48:51], v[118:121]
v_mfma_f32_16x16x32_f16 v[118:121], a[108:111], a[52:55], v[118:121]
v_mfma_f32_16x16x32_f16 v[122:125], a[112:115], a[48:51], v[122:125]
v_mfma_f32_16x16x32_f16 v[122:125], a[116:119], a[52:55], v[122:125]
v_mfma_f32_16x16x32_f16 v[126:129], a[120:123], a[48:51], v[126:129]
v_mfma_f32_16x16x32_f16 v[126:129], a[124:127], a[52:55], v[126:129]
v_mfma_f32_16x16x32_f16 v[66:69], a[96:99], a[40:43], v[66:69]
v_mfma_f32_16x16x32_f16 v[66:69], a[100:103], a[44:47], v[66:69]
v_mfma_f32_16x16x32_f16 v[70:73], a[104:107], a[40:43], v[70:73]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[70:73], a[108:111], a[44:47], v[70:73]
s_barrier
ds_read_b128 a[128:131], v254, offset:33792
v_mfma_f32_16x16x32_f16 v[74:77], a[112:115], a[40:43], v[74:77]
ds_read_b128 a[144:147], v254, offset:33856
v_mfma_f32_16x16x32_f16 v[74:77], a[116:119], a[44:47], v[74:77]
ds_read_b128 a[148:151], v254, offset:33920
v_mfma_f32_16x16x32_f16 v[78:81], a[120:123], a[40:43], v[78:81]
ds_read_b128 a[162:165], v254, offset:33984
v_mfma_f32_16x16x32_f16 v[78:81], a[124:127], a[44:47], v[78:81]
ds_read_b128 a[166:169], v254, offset:34304
v_mfma_f32_16x16x32_f16 v[82:85], a[96:99], a[32:35], v[82:85]
ds_read_b128 a[170:173], v254, offset:34368
v_mfma_f32_16x16x32_f16 v[82:85], a[100:103], a[36:39], v[82:85]
ds_read_b128 a[196:199], v254, offset:34432
v_mfma_f32_16x16x32_f16 v[86:89], a[104:107], a[32:35], v[86:89]
ds_read_b128 a[200:203], v254, offset:34496
v_mfma_f32_16x16x32_f16 v[86:89], a[108:111], a[36:39], v[86:89]
ds_read_b128 a[204:207], v254, offset:50688
v_mfma_f32_16x16x32_f16 v[90:93], a[112:115], a[32:35], v[90:93]
ds_read_b128 a[238:241], v254, offset:50752
v_mfma_f32_16x16x32_f16 v[90:93], a[116:119], a[36:39], v[90:93]
ds_read_b128 a[242:245], v254, offset:50816
v_mfma_f32_16x16x32_f16 v[94:97], a[120:123], a[32:35], v[94:97]
ds_read_b128 a[246:249], v254, offset:50880
v_mfma_f32_16x16x32_f16 v[94:97], a[124:127], a[36:39], v[94:97]
ds_read_b128 a[250:253], v254, offset:51200
v_mfma_f32_16x16x32_f16 a[234:237], a[96:99], a[24:27], a[234:237]
ds_read_b128 v[22:25], v254, offset:51264
v_mfma_f32_16x16x32_f16 a[234:237], a[100:103], a[28:31], a[234:237]
ds_read_b128 v[26:29], v254, offset:51328
v_mfma_f32_16x16x32_f16 a[230:233], a[104:107], a[24:27], a[230:233]
ds_read_b128 v[30:33], v254, offset:51392
v_mfma_f32_16x16x32_f16 a[230:233], a[108:111], a[28:31], a[230:233]
ds_read_b128 a[68:71], v6
v_mfma_f32_16x16x32_f16 a[226:229], a[112:115], a[24:27], a[226:229]
ds_read_b128 a[76:79], v6, offset:64
v_mfma_f32_16x16x32_f16 a[226:229], a[116:119], a[28:31], a[226:229]
ds_read_b128 a[64:67], v6, offset:128
v_mfma_f32_16x16x32_f16 a[222:225], a[120:123], a[24:27], a[222:225]
ds_read_b128 a[72:75], v6, offset:192
v_mfma_f32_16x16x32_f16 a[222:225], a[124:127], a[28:31], a[222:225]
ds_read_b128 a[80:83], v6, offset:512
v_mfma_f32_16x16x32_f16 a[218:221], a[96:99], a[16:19], a[218:221]
ds_read_b128 a[84:87], v6, offset:576
v_mfma_f32_16x16x32_f16 a[218:221], a[100:103], a[20:23], a[218:221]
ds_read_b128 a[88:91], v6, offset:640
v_mfma_f32_16x16x32_f16 a[214:217], a[104:107], a[16:19], a[214:217]
ds_read_b128 a[92:95], v6, offset:704
v_mfma_f32_16x16x32_f16 a[214:217], a[108:111], a[20:23], a[214:217]
s_add_u32 s16, s20, 0x80
s_addc_u32 s17, s22, 0
v_mfma_f32_16x16x32_f16 a[210:213], a[112:115], a[16:19], a[210:213]
s_and_b32 s17, s17, 0xffff
v_cndmask_b32_e32 v14, v14, v38, vcc
v_mfma_f32_16x16x32_f16 a[210:213], a[116:119], a[20:23], a[210:213]
s_mov_b32 m0, s31
v_mfma_f32_16x16x32_f16 a[192:195], a[120:123], a[16:19], a[192:195]
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[192:195], a[124:127], a[20:23], a[192:195]
v_mfma_f32_16x16x32_f16 a[154:157], a[96:99], a[8:11], a[154:157]
v_mfma_f32_16x16x32_f16 a[154:157], a[100:103], a[12:15], a[154:157]
v_cndmask_b32_e32 v40, v40, v38, vcc
s_mov_b32 m0, s33
v_mfma_f32_16x16x32_f16 a[180:183], a[104:107], a[8:11], a[180:183]
buffer_load_dwordx4 v40, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[180:183], a[108:111], a[12:15], a[180:183]
v_mfma_f32_16x16x32_f16 a[184:187], a[112:115], a[8:11], a[184:187]
v_mfma_f32_16x16x32_f16 a[184:187], a[116:119], a[12:15], a[184:187]
v_cndmask_b32_e32 v8, v8, v38, vcc
s_mov_b32 m0, s34
v_mfma_f32_16x16x32_f16 a[188:191], a[120:123], a[8:11], a[188:191]
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[188:191], a[124:127], a[12:15], a[188:191]
v_mfma_f32_16x16x32_f16 a[158:161], a[96:99], a[0:3], a[158:161]
v_mfma_f32_16x16x32_f16 a[158:161], a[100:103], a[4:7], a[158:161]
v_cndmask_b32_e32 v46, v46, v38, vcc
s_mov_b32 m0, s35
v_mfma_f32_16x16x32_f16 a[132:135], a[104:107], a[0:3], a[132:135]
buffer_load_dwordx4 v46, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[132:135], a[108:111], a[4:7], a[132:135]
v_mfma_f32_16x16x32_f16 a[136:139], a[112:115], a[0:3], a[136:139]
v_mfma_f32_16x16x32_f16 a[136:139], a[116:119], a[4:7], a[136:139]
v_mfma_f32_16x16x32_f16 a[140:143], a[120:123], a[0:3], a[140:143]
v_mfma_f32_16x16x32_f16 a[140:143], a[124:127], a[4:7], a[140:143]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[16:19], a[68:71], a[128:131], v[16:19]
v_mfma_f32_16x16x32_f16 v[16:19], a[76:79], a[144:147], v[16:19]
v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[128:131], v[10:13]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[144:147], v[10:13]
s_barrier
ds_read_b128 a[96:99], v9
v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[128:131], v[234:237]
ds_read_b128 a[100:103], v9, offset:64
v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[144:147], v[234:237]
ds_read_b128 a[104:107], v9, offset:128
v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[128:131], v[238:241]
ds_read_b128 a[108:111], v9, offset:192
v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[144:147], v[238:241]
ds_read_b128 a[112:115], v9, offset:512
v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[148:151], v[242:245]
ds_read_b128 a[116:119], v9, offset:576
v_mfma_f32_16x16x32_f16 v[242:245], a[76:79], a[162:165], v[242:245]
ds_read_b128 a[120:123], v9, offset:640
v_mfma_f32_16x16x32_f16 v[246:249], a[64:67], a[148:151], v[246:249]
ds_read_b128 a[124:127], v9, offset:704
v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[162:165], v[246:249]
s_and_b32 s53, s3, 0xffff
s_mov_b32 s52, s2
v_mfma_f32_16x16x32_f16 v[2:5], a[80:83], a[148:151], v[2:5]
s_mov_b32 m0, s36
v_mfma_f32_16x16x32_f16 v[2:5], a[84:87], a[162:165], v[2:5]
v_mfma_f32_16x16x32_f16 v[250:253], a[88:91], a[148:151], v[250:253]
buffer_load_dwordx4 v36, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[250:253], a[92:95], a[162:165], v[250:253]
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[166:169], v[194:197]
v_mfma_f32_16x16x32_f16 v[194:197], a[76:79], a[170:173], v[194:197]
s_mov_b32 m0, s37
v_mfma_f32_16x16x32_f16 v[198:201], a[64:67], a[166:169], v[198:201]
buffer_load_dwordx4 v42, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[170:173], v[198:201]
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[166:169], v[202:205]
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[170:173], v[202:205]
s_mov_b32 m0, s38
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[166:169], v[206:209]
buffer_load_dwordx4 v20, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[170:173], v[206:209]
v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[196:199], v[210:213]
v_mfma_f32_16x16x32_f16 v[210:213], a[76:79], a[200:203], v[210:213]
s_mov_b32 m0, s39
v_mfma_f32_16x16x32_f16 v[214:217], a[64:67], a[196:199], v[214:217]
buffer_load_dwordx4 v43, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[200:203], v[214:217]
v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[196:199], v[218:221]
v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[200:203], v[218:221]
s_mov_b32 m0, s40
v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[196:199], v[222:225]
buffer_load_dwordx4 v15, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[200:203], v[222:225]
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[204:207], v[162:165]
v_mfma_f32_16x16x32_f16 v[162:165], a[76:79], a[238:241], v[162:165]
s_mov_b32 m0, s41
v_mfma_f32_16x16x32_f16 v[166:169], a[64:67], a[204:207], v[166:169]
buffer_load_dwordx4 v44, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[238:241], v[166:169]
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[204:207], v[170:173]
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[238:241], v[170:173]
s_mov_b32 m0, s42
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[204:207], v[174:177]
buffer_load_dwordx4 v34, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[238:241], v[174:177]
v_mfma_f32_16x16x32_f16 v[182:185], a[68:71], a[242:245], v[182:185]
v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[246:249], v[182:185]
s_mov_b32 m0, s43
v_mfma_f32_16x16x32_f16 v[186:189], a[64:67], a[242:245], v[186:189]
buffer_load_dwordx4 v45, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[186:189], a[72:75], a[246:249], v[186:189]
v_mfma_f32_16x16x32_f16 v[190:193], a[80:83], a[242:245], v[190:193]
v_mfma_f32_16x16x32_f16 v[190:193], a[84:87], a[246:249], v[190:193]
s_mov_b32 m0, s44
v_mfma_f32_16x16x32_f16 v[178:181], a[88:91], a[242:245], v[178:181]
buffer_load_dwordx4 v41, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[178:181], a[92:95], a[246:249], v[178:181]
v_mfma_f32_16x16x32_f16 v[130:133], a[68:71], a[250:253], v[130:133]
v_mfma_f32_16x16x32_f16 v[130:133], a[76:79], v[22:25], v[130:133]
s_mov_b32 m0, s45
v_mfma_f32_16x16x32_f16 v[134:137], a[64:67], a[250:253], v[134:137]
buffer_load_dwordx4 v21, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], v[22:25], v[134:137]
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[250:253], v[138:141]
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], v[22:25], v[138:141]
s_mov_b32 m0, s46
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[250:253], v[142:145]
buffer_load_dwordx4 v39, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], v[22:25], v[142:145]
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], v[26:29], v[146:149]
v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], v[30:33], v[146:149]
s_mov_b32 m0, s47
v_mfma_f32_16x16x32_f16 v[150:153], a[64:67], v[26:29], v[150:153]
buffer_load_dwordx4 v47, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], v[30:33], v[150:153]
v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], v[26:29], v[154:157]
v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], v[30:33], v[154:157]
v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], v[26:29], v[158:161]
v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], v[30:33], v[158:161]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[98:101], a[96:99], a[128:131], v[98:101]
v_mfma_f32_16x16x32_f16 v[98:101], a[100:103], a[144:147], v[98:101]
v_mfma_f32_16x16x32_f16 v[102:105], a[104:107], a[128:131], v[102:105]
v_mfma_f32_16x16x32_f16 v[102:105], a[108:111], a[144:147], v[102:105]
v_mfma_f32_16x16x32_f16 v[106:109], a[112:115], a[128:131], v[106:109]
v_mfma_f32_16x16x32_f16 v[106:109], a[116:119], a[144:147], v[106:109]
v_mfma_f32_16x16x32_f16 v[110:113], a[120:123], a[128:131], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[124:127], a[144:147], v[110:113]
v_mfma_f32_16x16x32_f16 v[114:117], a[96:99], a[148:151], v[114:117]
v_mfma_f32_16x16x32_f16 v[114:117], a[100:103], a[162:165], v[114:117]
v_mfma_f32_16x16x32_f16 v[118:121], a[104:107], a[148:151], v[118:121]
v_mfma_f32_16x16x32_f16 v[118:121], a[108:111], a[162:165], v[118:121]
v_mfma_f32_16x16x32_f16 v[122:125], a[112:115], a[148:151], v[122:125]
v_mfma_f32_16x16x32_f16 v[122:125], a[116:119], a[162:165], v[122:125]
v_mfma_f32_16x16x32_f16 v[126:129], a[120:123], a[148:151], v[126:129]
v_mfma_f32_16x16x32_f16 v[126:129], a[124:127], a[162:165], v[126:129]
v_mfma_f32_16x16x32_f16 v[66:69], a[96:99], a[166:169], v[66:69]
v_mfma_f32_16x16x32_f16 v[66:69], a[100:103], a[170:173], v[66:69]
v_mfma_f32_16x16x32_f16 v[70:73], a[104:107], a[166:169], v[70:73]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[70:73], a[108:111], a[170:173], v[70:73]
s_barrier
ds_read_b128 a[56:59], v254
v_mfma_f32_16x16x32_f16 v[74:77], a[112:115], a[166:169], v[74:77]
ds_read_b128 a[60:63], v254, offset:64
v_mfma_f32_16x16x32_f16 v[74:77], a[116:119], a[170:173], v[74:77]
ds_read_b128 a[48:51], v254, offset:128
v_mfma_f32_16x16x32_f16 v[78:81], a[120:123], a[166:169], v[78:81]
ds_read_b128 a[52:55], v254, offset:192
v_mfma_f32_16x16x32_f16 v[78:81], a[124:127], a[170:173], v[78:81]
ds_read_b128 a[40:43], v254, offset:512
v_mfma_f32_16x16x32_f16 v[82:85], a[96:99], a[196:199], v[82:85]
ds_read_b128 a[44:47], v254, offset:576
v_mfma_f32_16x16x32_f16 v[82:85], a[100:103], a[200:203], v[82:85]
ds_read_b128 a[32:35], v254, offset:640
v_mfma_f32_16x16x32_f16 v[86:89], a[104:107], a[196:199], v[86:89]
ds_read_b128 a[36:39], v254, offset:704
v_mfma_f32_16x16x32_f16 v[86:89], a[108:111], a[200:203], v[86:89]
ds_read_b128 a[24:27], v254, offset:16896
v_mfma_f32_16x16x32_f16 v[90:93], a[112:115], a[196:199], v[90:93]
ds_read_b128 a[28:31], v254, offset:16960
v_mfma_f32_16x16x32_f16 v[90:93], a[116:119], a[200:203], v[90:93]
ds_read_b128 a[16:19], v254, offset:17024
v_mfma_f32_16x16x32_f16 v[94:97], a[120:123], a[196:199], v[94:97]
ds_read_b128 a[20:23], v254, offset:17088
v_mfma_f32_16x16x32_f16 v[94:97], a[124:127], a[200:203], v[94:97]
ds_read_b128 a[8:11], v254, offset:17408
v_mfma_f32_16x16x32_f16 a[234:237], a[96:99], a[204:207], a[234:237]
ds_read_b128 a[12:15], v254, offset:17472
v_mfma_f32_16x16x32_f16 a[234:237], a[100:103], a[238:241], a[234:237]
ds_read_b128 a[0:3], v254, offset:17536
v_mfma_f32_16x16x32_f16 a[230:233], a[104:107], a[204:207], a[230:233]
ds_read_b128 a[4:7], v254, offset:17600
v_mfma_f32_16x16x32_f16 a[230:233], a[108:111], a[238:241], a[230:233]
ds_read_b128 a[68:71], v230
v_mfma_f32_16x16x32_f16 a[226:229], a[112:115], a[204:207], a[226:229]
ds_read_b128 a[76:79], v230, offset:64
v_mfma_f32_16x16x32_f16 a[226:229], a[116:119], a[238:241], a[226:229]
ds_read_b128 a[64:67], v230, offset:128
v_mfma_f32_16x16x32_f16 a[222:225], a[120:123], a[204:207], a[222:225]
ds_read_b128 a[72:75], v230, offset:192
v_mfma_f32_16x16x32_f16 a[222:225], a[124:127], a[238:241], a[222:225]
ds_read_b128 a[80:83], v230, offset:512
v_mfma_f32_16x16x32_f16 a[218:221], a[96:99], a[242:245], a[218:221]
ds_read_b128 a[84:87], v230, offset:576
v_mfma_f32_16x16x32_f16 a[218:221], a[100:103], a[246:249], a[218:221]
ds_read_b128 a[88:91], v230, offset:640
v_mfma_f32_16x16x32_f16 a[214:217], a[104:107], a[242:245], a[214:217]
ds_read_b128 a[92:95], v230, offset:704
v_mfma_f32_16x16x32_f16 a[214:217], a[108:111], a[246:249], a[214:217]
s_add_u32 s20, s20, 0x100
s_addc_u32 s22, s22, 0
v_mfma_f32_16x16x32_f16 a[210:213], a[112:115], a[242:245], a[210:213]
s_and_b32 s17, s22, 0xffff
s_mov_b32 s16, s20
v_mfma_f32_16x16x32_f16 a[210:213], a[116:119], a[246:249], a[210:213]
s_mov_b32 m0, s23
v_mfma_f32_16x16x32_f16 a[192:195], a[120:123], a[242:245], a[192:195]
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[192:195], a[124:127], a[246:249], a[192:195]
v_mfma_f32_16x16x32_f16 a[154:157], a[96:99], a[250:253], a[154:157]
v_mfma_f32_16x16x32_f16 a[154:157], a[100:103], v[22:25], a[154:157]
s_mov_b32 m0, s48
v_mfma_f32_16x16x32_f16 a[180:183], a[104:107], a[250:253], a[180:183]
buffer_load_dwordx4 v40, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[180:183], a[108:111], v[22:25], a[180:183]
v_mfma_f32_16x16x32_f16 a[184:187], a[112:115], a[250:253], a[184:187]
v_mfma_f32_16x16x32_f16 a[184:187], a[116:119], v[22:25], a[184:187]
s_mov_b32 m0, s49
v_mfma_f32_16x16x32_f16 a[188:191], a[120:123], a[250:253], a[188:191]
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[188:191], a[124:127], v[22:25], a[188:191]
v_mfma_f32_16x16x32_f16 a[158:161], a[96:99], v[26:29], a[158:161]
v_mfma_f32_16x16x32_f16 a[158:161], a[100:103], v[30:33], a[158:161]
s_mov_b32 m0, s50
v_mfma_f32_16x16x32_f16 a[132:135], a[104:107], v[26:29], a[132:135]
buffer_load_dwordx4 v46, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[132:135], a[108:111], v[30:33], a[132:135]
s_add_u32 s2, s2, 0x100
s_addc_u32 s3, s3, 0
v_mfma_f32_16x16x32_f16 a[136:139], a[112:115], v[26:29], a[136:139]
s_add_i32 s10, s10, 2
s_cmp_lt_u32 s10, 61
v_mfma_f32_16x16x32_f16 a[136:139], a[116:119], v[30:33], a[136:139]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[140:143], a[120:123], v[26:29], a[140:143]
v_mfma_f32_16x16x32_f16 a[140:143], a[124:127], v[30:33], a[140:143]
s_cbranch_scc1 .LBB0_1
; %bb.2:
v_accvgpr_read_b32 v34, a234
v_accvgpr_read_b32 v35, a235
v_accvgpr_read_b32 v36, a236
v_accvgpr_read_b32 v37, a237
v_accvgpr_read_b32 v38, a230
v_accvgpr_read_b32 v39, a231
v_accvgpr_read_b32 v40, a232
v_accvgpr_read_b32 v41, a233
v_accvgpr_read_b32 v42, a226
v_accvgpr_read_b32 v43, a227
v_accvgpr_read_b32 v44, a228
v_accvgpr_read_b32 v45, a229
v_accvgpr_read_b32 v46, a222
v_accvgpr_read_b32 v47, a223
v_accvgpr_read_b32 v48, a224
v_accvgpr_read_b32 v49, a225
v_accvgpr_read_b32 v50, a218
v_accvgpr_read_b32 v51, a219
v_accvgpr_read_b32 v52, a220
v_accvgpr_read_b32 v53, a221
v_accvgpr_read_b32 v54, a214
v_accvgpr_read_b32 v55, a215
v_accvgpr_read_b32 v56, a216
v_accvgpr_read_b32 v57, a217
v_accvgpr_read_b32 v58, a210
v_accvgpr_read_b32 v59, a211
v_accvgpr_read_b32 v60, a212
v_accvgpr_read_b32 v61, a213
v_accvgpr_read_b32 v62, a192
v_accvgpr_read_b32 v63, a193
v_accvgpr_read_b32 v64, a194
v_accvgpr_read_b32 v65, a195
s_lshl_b32 s13, s1, 6
v_accvgpr_read_b32 v230, a152
v_and_b32_e32 v226, 63, v230
v_or_b32_e32 v0, s13, v226
s_waitcnt vmcnt(0), lgkmcnt(0)
s_barrier
v_lshrrev_b32_e32 v0, 4, v0
s_mul_i32 s2, s8, s12
s_ashr_i32 s3, s2, 31
s_lshl_b64 s[2:3], s[2:3], 1
s_add_u32 s2, s6, s2
s_addc_u32 s3, s7, s3
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s0, s2, s0
s_addc_u32 s1, s3, s1
s_lshl_b32 s2, s12, 6
s_ashr_i32 s3, s2, 31
s_lshl_b64 s[2:3], s[2:3], 1
s_add_u32 s28, s0, s2
s_addc_u32 s17, s1, s3
s_add_u32 s24, s28, s2
s_addc_u32 s11, s17, s3
s_add_u32 s20, s24, s2
s_addc_u32 s10, s11, s3
v_mul_lo_u32 v0, v0, s12
s_lshl_b32 s18, s12, 4
v_accvgpr_read_b32 v1, a176
v_lshl_add_u32 v227, v1, 3, v0
v_add_u32_e32 v228, s18, v227
v_add_u32_e32 v229, s18, v228
s_add_u32 s16, s0, 0x100
s_addc_u32 s9, s1, 0
s_add_u32 s12, s28, 0x100
s_addc_u32 s7, s17, 0
s_add_u32 s8, s24, 0x100
s_addc_u32 s6, s11, 0
s_add_u32 s4, s20, 0x100
s_addc_u32 s5, s10, 0
v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[56:59], v[16:19]
v_mfma_f32_16x16x32_f16 v[6:9], a[76:79], a[60:63], v[6:9]
v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[56:59], v[10:13]
v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[60:63], v[10:13]
v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[56:59], v[234:237]
v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[60:63], v[14:17]
s_nop 1
v_accvgpr_read_b32 v235, a191
v_accvgpr_read_b32 v234, a190
v_accvgpr_read_b32 v233, a189
v_accvgpr_read_b32 v232, a188
v_mfma_f32_16x16x32_f16 v[18:21], a[88:91], a[56:59], v[238:241]
v_mfma_f32_16x16x32_f16 v[18:21], a[92:95], a[60:63], v[18:21]
v_mfma_f32_16x16x32_f16 v[22:25], a[68:71], a[48:51], v[242:245]
v_mfma_f32_16x16x32_f16 v[22:25], a[76:79], a[52:55], v[22:25]
v_mfma_f32_16x16x32_f16 v[26:29], a[64:67], a[48:51], v[246:249]
v_mfma_f32_16x16x32_f16 v[26:29], a[72:75], a[52:55], v[26:29]
v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[48:51], v[2:5]
v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[52:55], v[0:3]
v_mfma_f32_16x16x32_f16 v[30:33], a[88:91], a[48:51], v[250:253]
v_mfma_f32_16x16x32_f16 v[30:33], a[92:95], a[52:55], v[30:33]
v_cvt_pk_f16_f32 v4, v6, v7
v_cvt_pk_f16_f32 v5, v8, v9
v_cvt_pk_f16_f32 v8, v10, v11
v_cvt_pk_f16_f32 v9, v12, v13
v_cvt_pk_f16_f32 v12, v14, v15
v_cvt_pk_f16_f32 v13, v16, v17
v_cvt_pk_f16_f32 v16, v18, v19
v_cvt_pk_f16_f32 v17, v20, v21
v_cvt_pk_f16_f32 v6, v22, v23
v_cvt_pk_f16_f32 v7, v24, v25
v_cvt_pk_f16_f32 v10, v26, v27
v_cvt_pk_f16_f32 v11, v28, v29
v_cvt_pk_f16_f32 v14, v0, v1
v_cvt_pk_f16_f32 v15, v2, v3
v_cvt_pk_f16_f32 v18, v30, v31
v_cvt_pk_f16_f32 v19, v32, v33
v_lshlrev_b32_e32 v0, 8, v230
v_accvgpr_read_b32 v23, a153
v_and_b32_e32 v20, 0x70, v23
v_and_b32_e32 v21, 1, v230
v_lshlrev_b32_e32 v1, 12, v21
v_and_b32_e32 v22, 16, v230
v_lshlrev_b32_e32 v2, 4, v22
s_lshr_b32 s19, s21, 1
s_and_b64 s[2:3], s[14:15], exec
s_cselect_b32 s2, 0, 0x80
s_movk_i32 s3, 0x2e00
v_and_or_b32 v0, v0, s3, v1
v_or3_b32 v3, s2, v2, v0
v_mov_b32_e32 v0, 0x70
v_bitop3_b32 v23, s19, v23, v0, bitop3:0x78
v_or_b32_e32 v2, v3, v23
v_add_u32_e32 v0, 0, v2
ds_write_b128 v0, v[4:7]
v_xad_u32 v1, v2, 16, 0
ds_write_b128 v1, v[8:11]
v_xad_u32 v2, v2, 64, 0
ds_write_b128 v2, v[12:15]
s_movk_i32 s2, 0x50
v_bitop3_b32 v3, v3, s2, v23, bitop3:0x36
v_add_u32_e32 v3, 0, v3
ds_write_b128 v3, v[16:19]
s_waitcnt lgkmcnt(0)
s_barrier
v_mov_b32_e32 v4, 0xe0
v_bitop3_b32 v4, s13, v4, v226, bitop3:0xc8
v_lshlrev_b32_e32 v5, 4, v4
v_lshrrev_b32_e32 v4, 1, v4
v_lshlrev_b32_e32 v6, 8, v22
v_bitop3_b32 v4, v5, v4, v20, bitop3:0x36
v_lshl_add_u32 v5, v21, 13, 0
v_add3_u32 v4, v5, v6, v4
ds_read_b128 v[6:9], v4
ds_read_b128 v[12:15], v4, offset:256
ds_read_b128 v[16:19], v4, offset:128
ds_read_b128 v[22:25], v4, offset:384
s_and_b32 s1, s1, 0xffff
s_mov_b32 s3, 0x27000
s_mov_b32 s2, 0x7ffffffe
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v10, v6
v_mov_b32_e32 v11, v7
v_lshlrev_b32_e32 v5, 1, v227
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[10:13], v5, s[0:3], 0, offen
s_nop 1
v_mov_b32_e32 v10, v14
v_mov_b32_e32 v11, v15
v_lshlrev_b32_e32 v226, 1, v228
buffer_store_dwordx4 v[8:11], v226, s[0:3], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
v_lshlrev_b32_e32 v227, 1, v229
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v227, s[0:3], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
v_add_lshl_u32 v228, v229, s18, 1
buffer_store_dwordx4 v[18:21], v228, s[0:3], 0, offen
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
v_mfma_f32_16x16x32_f16 v[194:197], a[88:91], a[32:35], v[222:225]
v_mfma_f32_16x16x32_f16 v[194:197], a[92:95], a[36:39], v[194:197]
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
v_cvt_pk_f16_f32 v20, v194, v195
v_cvt_pk_f16_f32 v21, v196, v197
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[6:9]
ds_write_b128 v1, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[6:9], v4
ds_read_b128 v[12:15], v4, offset:256
ds_read_b128 v[16:19], v4, offset:128
ds_read_b128 v[22:25], v4, offset:384
s_and_b32 s29, s17, 0xffff
s_mov_b32 s30, s2
s_mov_b32 s31, s3
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v10, v6
v_mov_b32_e32 v11, v7
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[10:13], v5, s[28:31], 0, offen
s_nop 1
v_mov_b32_e32 v10, v14
v_mov_b32_e32 v11, v15
buffer_store_dwordx4 v[8:11], v226, s[28:31], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v227, s[28:31], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
buffer_store_dwordx4 v[18:21], v228, s[28:31], 0, offen
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
v_mfma_f32_16x16x32_f16 v[162:165], a[88:91], a[16:19], v[178:181]
v_mfma_f32_16x16x32_f16 v[162:165], a[92:95], a[20:23], v[162:165]
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
v_cvt_pk_f16_f32 v20, v162, v163
v_cvt_pk_f16_f32 v21, v164, v165
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[6:9]
ds_write_b128 v1, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[6:9], v4
ds_read_b128 v[12:15], v4, offset:256
ds_read_b128 v[16:19], v4, offset:128
ds_read_b128 v[22:25], v4, offset:384
s_and_b32 s25, s11, 0xffff
s_mov_b32 s26, s2
s_mov_b32 s27, s3
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v10, v6
v_mov_b32_e32 v11, v7
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[10:13], v5, s[24:27], 0, offen
s_nop 1
v_mov_b32_e32 v10, v14
v_mov_b32_e32 v11, v15
buffer_store_dwordx4 v[8:11], v226, s[24:27], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v227, s[24:27], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
buffer_store_dwordx4 v[18:21], v228, s[24:27], 0, offen
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
v_mfma_f32_16x16x32_f16 v[130:133], a[88:91], a[0:3], v[158:161]
v_mfma_f32_16x16x32_f16 v[130:133], a[92:95], a[4:7], v[130:133]
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
v_cvt_pk_f16_f32 v20, v130, v131
v_cvt_pk_f16_f32 v21, v132, v133
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[6:9]
ds_write_b128 v1, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[6:9], v4
ds_read_b128 v[12:15], v4, offset:256
ds_read_b128 v[16:19], v4, offset:128
ds_read_b128 v[22:25], v4, offset:384
s_and_b32 s21, s10, 0xffff
s_mov_b32 s22, s2
s_mov_b32 s23, s3
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v10, v6
v_mov_b32_e32 v11, v7
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[10:13], v5, s[20:23], 0, offen
s_nop 1
v_mov_b32_e32 v10, v14
v_mov_b32_e32 v11, v15
buffer_store_dwordx4 v[8:11], v226, s[20:23], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v227, s[20:23], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
buffer_store_dwordx4 v[18:21], v228, s[20:23], 0, offen
v_add_u32_e32 v6, 0x1cdc0, v255
ds_read_b128 v[130:133], v6
ds_read_b128 v[134:137], v6, offset:64
ds_read_b128 v[138:141], v6, offset:128
ds_read_b128 v[142:145], v6, offset:192
ds_read_b128 v[146:149], v6, offset:512
ds_read_b128 v[150:153], v6, offset:576
ds_read_b128 v[154:157], v6, offset:640
ds_read_b128 v[158:161], v6, offset:704
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 v[6:9], v[130:133], a[56:59], v[98:101]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 v[6:9], v[134:137], a[60:63], v[6:9]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 v[10:13], v[138:141], a[56:59], v[102:105]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 v[10:13], v[142:145], a[60:63], v[10:13]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 v[14:17], v[146:149], a[56:59], v[106:109]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 v[14:17], v[150:153], a[60:63], v[14:17]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 v[18:21], v[154:157], a[56:59], v[110:113]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[18:21], v[158:161], a[60:63], v[18:21]
v_mfma_f32_16x16x32_f16 v[22:25], v[130:133], a[48:51], v[114:117]
v_mfma_f32_16x16x32_f16 v[22:25], v[134:137], a[52:55], v[22:25]
v_mfma_f32_16x16x32_f16 v[26:29], v[138:141], a[48:51], v[118:121]
v_mfma_f32_16x16x32_f16 v[26:29], v[142:145], a[52:55], v[26:29]
v_mfma_f32_16x16x32_f16 v[30:33], v[146:149], a[48:51], v[122:125]
v_mfma_f32_16x16x32_f16 v[30:33], v[150:153], a[52:55], v[30:33]
v_mfma_f32_16x16x32_f16 v[98:101], v[154:157], a[48:51], v[126:129]
v_mfma_f32_16x16x32_f16 v[98:101], v[158:161], a[52:55], v[98:101]
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
v_cvt_pk_f16_f32 v20, v98, v99
v_cvt_pk_f16_f32 v21, v100, v101
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[6:9]
ds_write_b128 v1, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[6:9], v4
ds_read_b128 v[12:15], v4, offset:256
ds_read_b128 v[16:19], v4, offset:128
ds_read_b128 v[22:25], v4, offset:384
s_and_b32 s17, s9, 0xffff
s_mov_b32 s18, s2
s_mov_b32 s19, s3
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v10, v6
v_mov_b32_e32 v11, v7
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[10:13], v5, s[16:19], 0, offen
s_nop 1
v_mov_b32_e32 v10, v14
v_mov_b32_e32 v11, v15
buffer_store_dwordx4 v[8:11], v226, s[16:19], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v227, s[16:19], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
buffer_store_dwordx4 v[18:21], v228, s[16:19], 0, offen
v_mfma_f32_16x16x32_f16 v[6:9], v[130:133], a[40:43], v[66:69]
v_mfma_f32_16x16x32_f16 v[6:9], v[134:137], a[44:47], v[6:9]
v_mfma_f32_16x16x32_f16 v[10:13], v[138:141], a[40:43], v[70:73]
v_mfma_f32_16x16x32_f16 v[10:13], v[142:145], a[44:47], v[10:13]
v_mfma_f32_16x16x32_f16 v[14:17], v[146:149], a[40:43], v[74:77]
v_mfma_f32_16x16x32_f16 v[14:17], v[150:153], a[44:47], v[14:17]
v_mfma_f32_16x16x32_f16 v[18:21], v[154:157], a[40:43], v[78:81]
v_mfma_f32_16x16x32_f16 v[18:21], v[158:161], a[44:47], v[18:21]
v_mfma_f32_16x16x32_f16 v[22:25], v[130:133], a[32:35], v[82:85]
v_mfma_f32_16x16x32_f16 v[22:25], v[134:137], a[36:39], v[22:25]
v_mfma_f32_16x16x32_f16 v[26:29], v[138:141], a[32:35], v[86:89]
v_mfma_f32_16x16x32_f16 v[26:29], v[142:145], a[36:39], v[26:29]
v_mfma_f32_16x16x32_f16 v[30:33], v[146:149], a[32:35], v[90:93]
v_mfma_f32_16x16x32_f16 v[30:33], v[150:153], a[36:39], v[30:33]
v_mfma_f32_16x16x32_f16 v[66:69], v[154:157], a[32:35], v[94:97]
v_mfma_f32_16x16x32_f16 v[66:69], v[158:161], a[36:39], v[66:69]
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
v_cvt_pk_f16_f32 v20, v66, v67
v_cvt_pk_f16_f32 v21, v68, v69
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[6:9]
ds_write_b128 v1, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[6:9], v4
ds_read_b128 v[12:15], v4, offset:256
ds_read_b128 v[16:19], v4, offset:128
ds_read_b128 v[22:25], v4, offset:384
s_and_b32 s13, s7, 0xffff
s_mov_b32 s14, s2
s_mov_b32 s15, s3
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v10, v6
v_mov_b32_e32 v11, v7
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[10:13], v5, s[12:15], 0, offen
s_nop 1
v_mov_b32_e32 v10, v14
v_mov_b32_e32 v11, v15
buffer_store_dwordx4 v[8:11], v226, s[12:15], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v227, s[12:15], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
buffer_store_dwordx4 v[18:21], v228, s[12:15], 0, offen
v_mfma_f32_16x16x32_f16 v[6:9], v[130:133], a[24:27], v[34:37]
v_mfma_f32_16x16x32_f16 v[6:9], v[134:137], a[28:31], v[6:9]
v_mfma_f32_16x16x32_f16 v[10:13], v[138:141], a[24:27], v[38:41]
v_mfma_f32_16x16x32_f16 v[10:13], v[142:145], a[28:31], v[10:13]
v_mfma_f32_16x16x32_f16 v[14:17], v[146:149], a[24:27], v[42:45]
v_mfma_f32_16x16x32_f16 v[14:17], v[150:153], a[28:31], v[14:17]
v_mfma_f32_16x16x32_f16 v[18:21], v[154:157], a[24:27], v[46:49]
v_mfma_f32_16x16x32_f16 v[18:21], v[158:161], a[28:31], v[18:21]
v_mfma_f32_16x16x32_f16 v[22:25], v[130:133], a[16:19], v[50:53]
v_mfma_f32_16x16x32_f16 v[22:25], v[134:137], a[20:23], v[22:25]
v_mfma_f32_16x16x32_f16 v[26:29], v[138:141], a[16:19], v[54:57]
v_mfma_f32_16x16x32_f16 v[26:29], v[142:145], a[20:23], v[26:29]
v_mfma_f32_16x16x32_f16 v[30:33], v[146:149], a[16:19], v[58:61]
v_mfma_f32_16x16x32_f16 v[30:33], v[150:153], a[20:23], v[30:33]
v_mfma_f32_16x16x32_f16 v[34:37], v[154:157], a[16:19], v[62:65]
v_mfma_f32_16x16x32_f16 v[34:37], v[158:161], a[20:23], v[34:37]
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
ds_write_b128 v0, v[6:9]
ds_write_b128 v1, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[6:9], v4
ds_read_b128 v[12:15], v4, offset:256
ds_read_b128 v[16:19], v4, offset:128
ds_read_b128 v[22:25], v4, offset:384
s_and_b32 s9, s6, 0xffff
s_mov_b32 s10, s2
s_mov_b32 s11, s3
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v10, v6
v_mov_b32_e32 v11, v7
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[10:13], v5, s[8:11], 0, offen
s_nop 1
v_mov_b32_e32 v10, v14
v_mov_b32_e32 v11, v15
buffer_store_dwordx4 v[8:11], v226, s[8:11], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v20, v16
v_mov_b32_e32 v21, v17
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v227, s[8:11], 0, offen
s_nop 1
v_mov_b32_e32 v20, v24
v_mov_b32_e32 v21, v25
buffer_store_dwordx4 v[18:21], v228, s[8:11], 0, offen
v_accvgpr_read_b32 v6, a154
v_accvgpr_read_b32 v7, a155
v_accvgpr_read_b32 v8, a156
v_accvgpr_read_b32 v9, a157
s_nop 1
v_mfma_f32_16x16x32_f16 v[6:9], v[130:133], a[8:11], v[6:9]
v_mfma_f32_16x16x32_f16 v[6:9], v[134:137], a[12:15], v[6:9]
v_accvgpr_read_b32 v10, a180
v_accvgpr_read_b32 v11, a181
v_accvgpr_read_b32 v12, a182
v_accvgpr_read_b32 v13, a183
s_nop 1
v_mfma_f32_16x16x32_f16 v[10:13], v[138:141], a[8:11], v[10:13]
v_mfma_f32_16x16x32_f16 v[10:13], v[142:145], a[12:15], v[10:13]
v_accvgpr_read_b32 v14, a184
v_accvgpr_read_b32 v15, a185
v_accvgpr_read_b32 v16, a186
v_accvgpr_read_b32 v17, a187
s_nop 1
v_mfma_f32_16x16x32_f16 v[14:17], v[146:149], a[8:11], v[14:17]
v_mfma_f32_16x16x32_f16 v[14:17], v[150:153], a[12:15], v[14:17]
v_mfma_f32_16x16x32_f16 v[18:21], v[154:157], a[8:11], v[232:235]
v_mfma_f32_16x16x32_f16 v[18:21], v[158:161], a[12:15], v[18:21]
v_accvgpr_read_b32 v22, a158
v_accvgpr_read_b32 v23, a159
v_accvgpr_read_b32 v24, a160
v_accvgpr_read_b32 v25, a161
s_nop 1
v_mfma_f32_16x16x32_f16 v[22:25], v[130:133], a[0:3], v[22:25]
v_mfma_f32_16x16x32_f16 v[22:25], v[134:137], a[4:7], v[22:25]
v_accvgpr_read_b32 v26, a132
v_accvgpr_read_b32 v27, a133
v_accvgpr_read_b32 v28, a134
v_accvgpr_read_b32 v29, a135
s_nop 1
v_mfma_f32_16x16x32_f16 v[26:29], v[138:141], a[0:3], v[26:29]
v_mfma_f32_16x16x32_f16 v[26:29], v[142:145], a[4:7], v[26:29]
v_accvgpr_read_b32 v30, a136
v_accvgpr_read_b32 v31, a137
v_accvgpr_read_b32 v32, a138
v_accvgpr_read_b32 v33, a139
s_nop 1
v_mfma_f32_16x16x32_f16 v[30:33], v[146:149], a[0:3], v[30:33]
v_mfma_f32_16x16x32_f16 v[30:33], v[150:153], a[4:7], v[30:33]
v_accvgpr_read_b32 v34, a140
v_accvgpr_read_b32 v35, a141
v_accvgpr_read_b32 v36, a142
v_accvgpr_read_b32 v37, a143
s_nop 1
v_mfma_f32_16x16x32_f16 v[34:37], v[154:157], a[0:3], v[34:37]
v_mfma_f32_16x16x32_f16 v[34:37], v[158:161], a[4:7], v[34:37]
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
ds_write_b128 v0, v[6:9]
ds_write_b128 v1, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[0:3], v4
ds_read_b128 v[8:11], v4, offset:256
ds_read_b128 v[12:15], v4, offset:128
ds_read_b128 v[18:21], v4, offset:384
s_and_b32 s5, s5, 0xffff
s_mov_b32 s6, s2
s_mov_b32 s7, s3
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v6, v0
v_mov_b32_e32 v7, v1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[6:9], v5, s[4:7], 0, offen
v_mov_b32_e32 v4, v10
v_mov_b32_e32 v5, v11
buffer_store_dwordx4 v[2:5], v226, s[4:7], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v16, v12
v_mov_b32_e32 v17, v13
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[16:19], v227, s[4:7], 0, offen
s_nop 1
v_mov_b32_e32 v16, v20
v_mov_b32_e32 v17, v21
buffer_store_dwordx4 v[14:17], v228, s[4:7], 0, offen
s_endpgm
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v10
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
	.set v10.num_agpr, 230
	.set v10.numbered_sgpr, 56
	.set v10.num_named_barrier, 0
	.set v10.private_seg_size, 0
	.set v10.uses_vcc, 1
	.set v10.uses_flat_scratch, 0
	.set v10.has_dyn_sized_stack, 0
	.set v10.has_recursion, 0
	.set v10.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11572
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 230
; TotalNumVgprs: 486
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 60
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 486
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
	.byte	0                               ; EOM(3)
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
  - .agpr_count:     230
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
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         v10.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 512
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
