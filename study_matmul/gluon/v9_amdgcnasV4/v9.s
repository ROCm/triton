	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v9                              ; -- Begin function v9
	.p2align	8
	.type	v9,@function
v9:                                     ; @v9
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
v_readfirstlane_b32 s49, v11
s_bfe_u32 s50, s49, 0x20006
s_add_i32 s0, s8, 0xff
s_ashr_i32 s1, s0, 31
s_lshr_b32 s1, s1, 24
s_add_i32 s0, s0, s1
s_ashr_i32 s0, s0, 8
s_add_i32 s1, s9, 0xff
s_ashr_i32 s8, s1, 31
s_lshr_b32 s8, s8, 24
s_add_i32 s1, s1, s8
s_ashr_i32 s1, s1, 8
s_ashr_i32 s8, s16, 31
s_lshr_b32 s8, s8, 29
s_add_i32 s8, s16, s8
s_ashr_i32 s8, s8, 3
s_lshl_b32 s9, s16, 5
s_mulk_i32 s8, 0xff01
s_add_i32 s8, s8, s9
s_lshl_b32 s9, s1, 2
s_xor_b32 s1, s8, s1
s_ashr_i32 s1, s1, 31
s_abs_i32 s13, s8
s_abs_i32 s14, s9
v_cvt_f32_u32_e32 v0, s14
v_rcp_iflag_f32_e32 v0, v0
s_nop 0
v_mul_f32_e32 v0, 0x4f7ffffe, v0
v_cvt_u32_f32_e32 v0, v0
s_sub_i32 s15, 0, s14
v_readfirstlane_b32 s16, v0
s_mul_i32 s15, s15, s16
s_mul_hi_u32 s15, s16, s15
s_add_i32 s16, s16, s15
s_mul_hi_u32 s15, s13, s16
s_mul_i32 s16, s15, s14
s_sub_i32 s13, s13, s16
s_add_i32 s16, s15, 1
s_sub_i32 s17, s13, s14
s_cmp_ge_u32 s13, s14
s_cselect_b32 s15, s16, s15
s_cselect_b32 s13, s17, s13
s_add_i32 s16, s15, 1
s_cmp_ge_u32 s13, s14
s_cselect_b32 s13, s16, s15
s_xor_b32 s13, s13, s1
s_sub_i32 s1, s13, s1
s_lshl_b32 s13, s1, 2
s_sub_i32 s0, s0, s13
s_min_i32 s0, s0, 4
s_mul_i32 s1, s1, s9
s_sub_i32 s1, s8, s1
s_xor_b32 s8, s1, s0
s_ashr_i32 s8, s8, 31
s_abs_i32 s9, s1
s_abs_i32 s14, s0
v_cvt_f32_u32_e32 v0, s14
v_rcp_iflag_f32_e32 v0, v0
s_nop 0
v_mul_f32_e32 v0, 0x4f7ffffe, v0
v_cvt_u32_f32_e32 v0, v0
s_sub_i32 s15, 0, s14
v_readfirstlane_b32 s16, v0
s_mul_i32 s15, s15, s16
s_mul_hi_u32 s15, s16, s15
s_add_i32 s16, s16, s15
s_mul_hi_u32 s15, s9, s16
s_mul_i32 s16, s15, s14
s_sub_i32 s9, s9, s16
s_add_i32 s16, s15, 1
s_sub_i32 s17, s9, s14
s_cmp_ge_u32 s9, s14
s_cselect_b32 s15, s16, s15
s_cselect_b32 s9, s17, s9
s_add_i32 s16, s15, 1
s_cmp_ge_u32 s9, s14
s_cselect_b32 s9, s16, s15
s_xor_b32 s9, s9, s8
s_sub_i32 s14, s9, s8
s_mul_i32 s0, s14, s0
s_sub_i32 s51, s1, s0
s_add_i32 s51, s51, s13
v_lshlrev_b32_e32 v0, 1, v11
v_and_b32_e32 v0, 0x70, v0
v_or_b32_e32 v1, s50, v0
v_lshlrev_b32_e32 v0, 3, v11
v_and_b32_e32 v0, 56, v0
s_lshl_b32 s1, s51, 8
s_mul_i32 s8, s1, s10
s_ashr_i32 s9, s8, 31
s_lshl_b64 s[8:9], s[8:9], 1
s_add_u32 s16, s2, s8
s_addc_u32 s30, s3, s9
s_lshl_b32 s0, s14, 8
s_mul_i32 s8, s0, s11
s_ashr_i32 s9, s8, 31
s_lshl_b64 s[8:9], s[8:9], 1
s_add_u32 s20, s4, s8
s_addc_u32 s47, s5, s9
s_lshl_b32 s21, s10, 2
s_mul_i32 s8, s10, 0x74
v_mad_u64_u32 v[2:3], s[4:5], v1, s10, v[0:1]
v_add_u32_e32 v3, s21, v2
v_add_u32_e32 v4, s21, v3
v_add_u32_e32 v5, s21, v4
v_add_u32_e32 v6, s8, v5
v_add_u32_e32 v7, s21, v6
v_add_u32_e32 v8, s21, v7
s_lshl_b32 s8, s11, 2
v_mad_u64_u32 v[0:1], s[4:5], v1, s11, v[0:1]
v_add_u32_e32 v1, s8, v0
v_add_u32_e32 v9, s8, v1
v_add_u32_e32 v10, s8, v9
s_lshl_b32 s4, s11, 8
s_ashr_i32 s4, s4, 1
s_and_b32 s17, s30, 0xffff
s_mov_b32 s19, 0x27000
s_mov_b32 s18, 0x7ffffffe
s_mul_i32 s5, s50, 0x420
s_add_i32 s8, s5, 0
v_lshlrev_b32_e32 v12, 1, v2
s_mov_b32 m0, s8
s_nop 0
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
s_add_i32 s48, s5, 0x1080
s_add_i32 s9, s8, 0x1080
v_lshlrev_b32_e32 v13, 1, v3
s_mov_b32 m0, s9
s_nop 0
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s52, s5, 0x2100
s_add_i32 s11, s8, 0x2100
v_lshlrev_b32_e32 v14, 1, v4
s_mov_b32 m0, s11
s_nop 0
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_add_i32 s53, s5, 0x3180
s_add_i32 s13, s8, 0x3180
v_lshlrev_b32_e32 v15, 1, v5
s_mov_b32 m0, s13
s_nop 0
buffer_load_dwordx4 v15, s[16:19], 0, offen, lds
s_add_i32 s14, s8, 0x4200
v_lshlrev_b32_e32 v6, 1, v6
s_mov_b32 m0, s14
s_nop 0
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_add_i32 s15, s8, 0x5280
v_lshlrev_b32_e32 v7, 1, v7
s_mov_b32 m0, s15
s_nop 0
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
s_add_i32 s24, s8, 0x6300
v_lshlrev_b32_e32 v16, 1, v8
s_mov_b32 m0, s24
s_nop 0
buffer_load_dwordx4 v16, s[16:19], 0, offen, lds
s_add_i32 s25, s8, 0x7380
v_add_lshl_u32 v8, v8, s21, 1
s_mov_b32 m0, s25
s_nop 0
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
s_and_b32 s21, s47, 0xffff
s_mov_b32 s22, s18
s_mov_b32 s23, s19
s_add_i32 s54, 0, 0x107e0
s_add_i32 s26, s54, s5
v_lshlrev_b32_e32 v2, 1, v0
s_mov_b32 m0, s26
s_nop 0
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_i32 s27, s54, s48
v_lshlrev_b32_e32 v3, 1, v1
s_mov_b32 m0, s27
s_nop 0
buffer_load_dwordx4 v3, s[20:23], 0, offen, lds
s_add_i32 s28, s54, s52
v_lshlrev_b32_e32 v4, 1, v9
s_mov_b32 m0, s28
s_nop 0
buffer_load_dwordx4 v4, s[20:23], 0, offen, lds
s_add_i32 s29, s54, s53
v_lshlrev_b32_e32 v5, 1, v10
s_mov_b32 m0, s29
s_nop 0
buffer_load_dwordx4 v5, s[20:23], 0, offen, lds
s_add_u32 s16, s16, 0x80
s_addc_u32 s17, s30, 0
s_add_i32 s34, 0, 0x18bc0
s_add_i32 s30, s34, s5
v_add_lshl_u32 v0, v0, s4, 1
s_mov_b32 m0, s30
s_nop 0
buffer_load_dwordx4 v0, s[20:23], 0, offen, lds
s_add_i32 s31, s34, s48
v_add_lshl_u32 v1, v1, s4, 1
s_mov_b32 m0, s31
s_nop 0
buffer_load_dwordx4 v1, s[20:23], 0, offen, lds
s_add_i32 s33, s34, s52
v_add_lshl_u32 v9, v9, s4, 1
s_mov_b32 m0, s33
s_nop 0
buffer_load_dwordx4 v9, s[20:23], 0, offen, lds
s_add_i32 s34, s34, s53
v_add_lshl_u32 v10, v10, s4, 1
s_mov_b32 m0, s34
s_nop 0
buffer_load_dwordx4 v10, s[20:23], 0, offen, lds
s_waitcnt lgkmcnt(0)
s_barrier
s_and_b32 s17, s17, 0xffff
s_add_i32 s35, s8, 0x8400
s_mov_b32 m0, s35
v_accvgpr_write_b32 a153, v12
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
s_add_i32 s36, s8, 0x9480
s_mov_b32 m0, s36
v_accvgpr_write_b32 a154, v13
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s37, s8, 0xa500
s_mov_b32 m0, s37
v_accvgpr_write_b32 a155, v14
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_add_i32 s38, s8, 0xb580
s_mov_b32 m0, s38
v_accvgpr_write_b32 a156, v15
buffer_load_dwordx4 v15, s[16:19], 0, offen, lds
s_add_i32 s39, s8, 0xc600
s_mov_b32 m0, s39
v_accvgpr_write_b32 a157, v6
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_add_i32 s40, s8, 0xd680
s_mov_b32 m0, s40
v_accvgpr_write_b32 a158, v7
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
s_add_i32 s41, s8, 0xe700
s_mov_b32 m0, s41
v_accvgpr_write_b32 a159, v16
buffer_load_dwordx4 v16, s[16:19], 0, offen, lds
s_add_i32 s42, s8, 0xf780
s_mov_b32 m0, s42
v_accvgpr_write_b32 a160, v8
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
s_add_i32 s46, 0, 0x149e0
s_add_i32 s43, s46, s5
v_add_u32_e32 v2, 0x80, v2
s_mov_b32 m0, s43
v_accvgpr_write_b32 a165, v2
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_i32 s44, s46, s48
v_add_u32_e32 v2, 0x80, v3
s_mov_b32 m0, s44
v_accvgpr_write_b32 a166, v2
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_i32 s45, s46, s52
v_add_u32_e32 v2, 0x80, v4
s_mov_b32 m0, s45
v_accvgpr_write_b32 a167, v2
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_i32 s46, s46, s53
v_add_u32_e32 v2, 0x80, v5
s_mov_b32 m0, s46
v_accvgpr_write_b32 a168, v2
buffer_load_dwordx4 v2, s[20:23], 0, offen, lds
s_add_u32 s20, s20, 0x80
s_addc_u32 s21, s47, 0
s_and_b32 s17, s21, 0xffff
s_mov_b32 s16, s20
s_add_i32 s4, 0, 0x1cdc0
s_add_i32 s22, s4, s5
s_mov_b32 m0, s22
v_accvgpr_write_b32 a161, v0
buffer_load_dwordx4 v0, s[16:19], 0, offen, lds
s_add_i32 s23, s4, s48
s_mov_b32 m0, s23
v_accvgpr_write_b32 a162, v1
buffer_load_dwordx4 v1, s[16:19], 0, offen, lds
s_add_i32 s47, s4, s52
s_mov_b32 m0, s47
v_accvgpr_write_b32 a163, v9
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
s_add_i32 s48, s4, s53
s_mov_b32 m0, s48
v_accvgpr_write_b32 a164, v10
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
v_and_b32_e32 v3, 15, v11
v_lshlrev_b32_e32 v1, 10, v3
v_accvgpr_write_b32 a152, v11
v_and_b32_e32 v2, 48, v11
s_cmp_lt_u32 s50, 2
s_cselect_b64 s[4:5], -1, 0
s_and_b64 s[16:17], s[4:5], exec
s_cselect_b32 s16, 0, 0x100
v_or3_b32 v0, v2, s16, v1
v_accvgpr_write_b32 a169, v3
v_lshlrev_b32_e32 v3, 5, v3
v_add_u32_e32 v0, v0, v3
v_add_u32_e32 v255, 0, v0
ds_read_b128 a[56:59], v255
ds_read_b128 a[60:63], v255, offset:64
ds_read_b128 a[48:51], v255, offset:128
ds_read_b128 a[52:55], v255, offset:192
ds_read_b128 a[40:43], v255, offset:512
ds_read_b128 a[44:47], v255, offset:576
ds_read_b128 a[32:35], v255, offset:640
ds_read_b128 a[36:39], v255, offset:704
ds_read_b128 a[24:27], v255, offset:16896
ds_read_b128 a[28:31], v255, offset:16960
ds_read_b128 a[16:19], v255, offset:17024
ds_read_b128 a[20:23], v255, offset:17088
ds_read_b128 a[8:11], v255, offset:17408
ds_read_b128 a[12:15], v255, offset:17472
ds_read_b128 a[0:3], v255, offset:17536
ds_read_b128 a[4:7], v255, offset:17600
s_and_b32 s49, s49, 64
v_or_b32_e32 v1, v1, v2
v_add_u32_e32 v1, v1, v3
v_lshl_add_u32 v1, s49, 2, v1
v_accvgpr_write_b32 a178, v1
v_add_u32_e32 v1, s54, v1
ds_read_b128 a[68:71], v1
ds_read_b128 a[76:79], v1, offset:64
ds_read_b128 a[64:67], v1, offset:128
ds_read_b128 a[72:75], v1, offset:192
ds_read_b128 a[80:83], v1, offset:512
ds_read_b128 a[84:87], v1, offset:576
ds_read_b128 a[88:91], v1, offset:640
ds_read_b128 a[92:95], v1, offset:704
s_mul_i32 s10, s10, s51
s_lshl_b32 s16, s10, 8
s_ashr_i32 s17, s16, 31
s_lshl_b64 s[16:17], s[16:17], 1
s_add_u32 s2, s2, s16
s_addc_u32 s3, s3, s17
s_add_u32 s2, s2, 0x180
s_addc_u32 s3, s3, 0
s_mov_b32 s10, -2
v_accvgpr_write_b32 a170, 0
v_accvgpr_write_b32 a171, 0
v_accvgpr_write_b32 a172, 0
v_accvgpr_write_b32 a173, 0
v_accvgpr_write_b32 a174, 0
v_accvgpr_write_b32 a175, 0
v_accvgpr_write_b32 a176, 0
v_accvgpr_write_b32 a177, 0
v_accvgpr_write_b32 a188, 0
v_accvgpr_write_b32 a189, 0
v_accvgpr_write_b32 a190, 0
v_accvgpr_write_b32 a191, 0
v_accvgpr_write_b32 a224, 0
v_accvgpr_write_b32 a225, 0
v_accvgpr_write_b32 a226, 0
v_accvgpr_write_b32 a227, 0
v_accvgpr_write_b32 a228, 0
v_accvgpr_write_b32 a229, 0
v_accvgpr_write_b32 a230, 0
v_accvgpr_write_b32 a231, 0
v_accvgpr_write_b32 a232, 0
v_accvgpr_write_b32 a233, 0
v_accvgpr_write_b32 a234, 0
v_accvgpr_write_b32 a235, 0
v_accvgpr_write_b32 a236, 0
v_accvgpr_write_b32 a237, 0
v_accvgpr_write_b32 a238, 0
v_accvgpr_write_b32 a239, 0
v_mov_b32_e32 v34, 0
v_mov_b32_e32 v35, 0
v_mov_b32_e32 v36, 0
v_mov_b32_e32 v37, 0
v_mov_b32_e32 v38, 0
v_mov_b32_e32 v39, 0
v_mov_b32_e32 v40, 0
v_mov_b32_e32 v41, 0
v_mov_b32_e32 v42, 0
v_mov_b32_e32 v43, 0
v_mov_b32_e32 v44, 0
v_mov_b32_e32 v45, 0
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
v_accvgpr_write_b32 a240, 0
v_accvgpr_write_b32 a241, 0
v_accvgpr_write_b32 a242, 0
v_accvgpr_write_b32 a243, 0
v_accvgpr_write_b32 a244, 0
v_accvgpr_write_b32 a245, 0
v_accvgpr_write_b32 a246, 0
v_accvgpr_write_b32 a247, 0
v_accvgpr_write_b32 a248, 0
v_accvgpr_write_b32 a249, 0
v_accvgpr_write_b32 a250, 0
v_accvgpr_write_b32 a251, 0
v_accvgpr_write_b32 a252, 0
v_accvgpr_write_b32 a253, 0
v_accvgpr_write_b32 a254, 0
v_accvgpr_write_b32 a255, 0
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
v_accvgpr_write_b32 a180, 0
v_accvgpr_write_b32 a181, 0
v_accvgpr_write_b32 a182, 0
v_accvgpr_write_b32 a183, 0
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
v_accvgpr_write_b32 a184, 0
v_accvgpr_write_b32 a185, 0
v_accvgpr_write_b32 a186, 0
v_accvgpr_write_b32 a187, 0
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
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
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
v_mov_b32_e32 v158, 0
v_mov_b32_e32 v159, 0
v_mov_b32_e32 v160, 0
v_mov_b32_e32 v161, 0
v_mov_b32_e32 v154, 0
v_mov_b32_e32 v155, 0
v_mov_b32_e32 v156, 0
v_mov_b32_e32 v157, 0
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
v_mov_b32_e32 v178, 0
v_mov_b32_e32 v179, 0
v_mov_b32_e32 v180, 0
v_mov_b32_e32 v181, 0
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
v_mov_b32_e32 v214, 0
v_mov_b32_e32 v215, 0
v_mov_b32_e32 v216, 0
v_mov_b32_e32 v217, 0
v_mov_b32_e32 v218, 0
v_mov_b32_e32 v219, 0
v_mov_b32_e32 v220, 0
v_mov_b32_e32 v221, 0
v_mov_b32_e32 v230, 0
v_mov_b32_e32 v231, 0
v_mov_b32_e32 v232, 0
v_mov_b32_e32 v233, 0
v_mov_b32_e32 v210, 0
v_mov_b32_e32 v211, 0
v_mov_b32_e32 v212, 0
v_mov_b32_e32 v213, 0
v_mov_b32_e32 v222, 0
v_mov_b32_e32 v223, 0
v_mov_b32_e32 v224, 0
v_mov_b32_e32 v225, 0
v_mov_b32_e32 v226, 0
v_mov_b32_e32 v227, 0
v_mov_b32_e32 v228, 0
v_mov_b32_e32 v229, 0
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
v_mov_b32_e32 v250, 0
v_mov_b32_e32 v251, 0
v_mov_b32_e32 v252, 0
v_mov_b32_e32 v253, 0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0
v_accvgpr_read_b32 v23, a178
v_add_u32_e32 v254, 0, v23
v_add_u32_e32 v6, 0x18bc0, v254
v_accvgpr_read_b32 v32, a153
v_bfrev_b32_e32 v33, 1
v_accvgpr_read_b32 v18, a154
v_accvgpr_read_b32 v26, a155
v_accvgpr_read_b32 v19, a156
v_accvgpr_read_b32 v27, a157
v_accvgpr_read_b32 v20, a158
v_accvgpr_read_b32 v29, a159
v_accvgpr_read_b32 v21, a160
v_accvgpr_read_b32 v9, a165
v_accvgpr_read_b32 v24, a166
v_accvgpr_read_b32 v8, a167
v_accvgpr_read_b32 v25, a168
v_add_u32_e32 v94, 0x149e0, v254
v_accvgpr_read_b32 v30, a161
v_accvgpr_read_b32 v7, a162
v_accvgpr_read_b32 v22, a163
v_accvgpr_read_b32 v28, a164
v_add_u32_e32 v31, 0x1cdc0, v254
s_mov_b32 s54, s18
s_mov_b32 s55, s19
v_add_u32_e32 v13, 0x107e0, v254
s_waitcnt lgkmcnt(0)
.LBB0_1:
v_mfma_f32_16x16x32_f16 v[14:17], a[68:71], a[56:59], v[14:17]
s_add_u32 s16, s2, 0xffffff80
s_addc_u32 s17, s3, -1
v_mfma_f32_16x16x32_f16 v[14:17], a[76:79], a[60:63], v[14:17]
v_mfma_f32_16x16x32_f16 v[134:137], a[64:67], a[56:59], v[134:137]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], a[60:63], v[134:137]
s_barrier
ds_read_b128 a[96:99], v6
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[56:59], v[138:141]
ds_read_b128 a[100:103], v6, offset:64
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], a[60:63], v[138:141]
ds_read_b128 a[104:107], v6, offset:128
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[56:59], v[142:145]
ds_read_b128 a[108:111], v6, offset:192
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[60:63], v[142:145]
ds_read_b128 a[112:115], v6, offset:512
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[48:51], v[146:149]
ds_read_b128 a[116:119], v6, offset:576
v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], a[52:55], v[146:149]
ds_read_b128 a[120:123], v6, offset:640
v_mfma_f32_16x16x32_f16 v[150:153], a[64:67], a[48:51], v[150:153]
ds_read_b128 a[124:127], v6, offset:704
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[52:55], v[150:153]
s_cmp_eq_u32 s10, 60
s_cselect_b64 vcc, -1, 0
v_mfma_f32_16x16x32_f16 v[158:161], a[80:83], a[48:51], v[158:161]
s_and_b32 s17, s17, 0xffff
v_cndmask_b32_e32 v32, v32, v33, vcc
v_mfma_f32_16x16x32_f16 v[158:161], a[84:87], a[52:55], v[158:161]
s_mov_b32 m0, s8
v_mfma_f32_16x16x32_f16 v[154:157], a[88:91], a[48:51], v[154:157]
buffer_load_dwordx4 v32, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[154:157], a[92:95], a[52:55], v[154:157]
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[40:43], v[162:165]
v_mfma_f32_16x16x32_f16 v[162:165], a[76:79], a[44:47], v[162:165]
v_cndmask_b32_e32 v18, v18, v33, vcc
s_mov_b32 m0, s9
v_mfma_f32_16x16x32_f16 v[166:169], a[64:67], a[40:43], v[166:169]
buffer_load_dwordx4 v18, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[44:47], v[166:169]
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[40:43], v[170:173]
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[44:47], v[170:173]
v_cndmask_b32_e32 v26, v26, v33, vcc
s_mov_b32 m0, s11
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[40:43], v[174:177]
buffer_load_dwordx4 v26, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[44:47], v[174:177]
v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[32:35], v[178:181]
v_mfma_f32_16x16x32_f16 v[178:181], a[76:79], a[36:39], v[178:181]
v_cndmask_b32_e32 v19, v19, v33, vcc
s_mov_b32 m0, s13
v_mfma_f32_16x16x32_f16 v[182:185], a[64:67], a[32:35], v[182:185]
buffer_load_dwordx4 v19, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[36:39], v[182:185]
v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[32:35], v[186:189]
v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[36:39], v[186:189]
v_cndmask_b32_e32 v27, v27, v33, vcc
s_mov_b32 m0, s14
v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[32:35], v[190:193]
buffer_load_dwordx4 v27, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[36:39], v[190:193]
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[24:27], v[194:197]
v_mfma_f32_16x16x32_f16 v[194:197], a[76:79], a[28:31], v[194:197]
v_cndmask_b32_e32 v20, v20, v33, vcc
s_mov_b32 m0, s15
v_mfma_f32_16x16x32_f16 v[198:201], a[64:67], a[24:27], v[198:201]
buffer_load_dwordx4 v20, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[28:31], v[198:201]
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[24:27], v[202:205]
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[28:31], v[202:205]
v_cndmask_b32_e32 v29, v29, v33, vcc
s_mov_b32 m0, s24
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[24:27], v[206:209]
buffer_load_dwordx4 v29, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[28:31], v[206:209]
v_mfma_f32_16x16x32_f16 v[214:217], a[68:71], a[16:19], v[214:217]
v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[20:23], v[214:217]
v_cndmask_b32_e32 v21, v21, v33, vcc
s_mov_b32 m0, s25
v_mfma_f32_16x16x32_f16 v[218:221], a[64:67], a[16:19], v[218:221]
buffer_load_dwordx4 v21, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[218:221], a[72:75], a[20:23], v[218:221]
s_and_b32 s17, s21, 0xffff
s_mov_b32 s16, s20
v_mfma_f32_16x16x32_f16 v[230:233], a[80:83], a[16:19], v[230:233]
v_cndmask_b32_e32 v9, v9, v33, vcc
s_mov_b32 m0, s26
v_mfma_f32_16x16x32_f16 v[230:233], a[84:87], a[20:23], v[230:233]
v_mfma_f32_16x16x32_f16 v[210:213], a[88:91], a[16:19], v[210:213]
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[210:213], a[92:95], a[20:23], v[210:213]
v_mfma_f32_16x16x32_f16 v[222:225], a[68:71], a[8:11], v[222:225]
v_mfma_f32_16x16x32_f16 v[222:225], a[76:79], a[12:15], v[222:225]
v_cndmask_b32_e32 v24, v24, v33, vcc
s_mov_b32 m0, s27
v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[8:11], v[226:229]
buffer_load_dwordx4 v24, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[226:229], a[72:75], a[12:15], v[226:229]
v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[8:11], v[234:237]
v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[12:15], v[234:237]
v_cndmask_b32_e32 v8, v8, v33, vcc
s_mov_b32 m0, s28
v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[8:11], v[238:241]
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[12:15], v[238:241]
v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[0:3], v[242:245]
v_mfma_f32_16x16x32_f16 v[242:245], a[76:79], a[4:7], v[242:245]
v_cndmask_b32_e32 v25, v25, v33, vcc
s_mov_b32 m0, s29
v_mfma_f32_16x16x32_f16 v[246:249], a[64:67], a[0:3], v[246:249]
buffer_load_dwordx4 v25, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[4:7], v[246:249]
v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], a[0:3], v[250:253]
v_mfma_f32_16x16x32_f16 v[250:253], a[84:87], a[4:7], v[250:253]
v_mfma_f32_16x16x32_f16 v[2:5], a[88:91], a[0:3], v[2:5]
v_mfma_f32_16x16x32_f16 v[2:5], a[92:95], a[4:7], v[2:5]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[170:173], a[96:99], a[56:59], a[170:173]
v_mfma_f32_16x16x32_f16 a[170:173], a[100:103], a[60:63], a[170:173]
v_mfma_f32_16x16x32_f16 a[174:177], a[104:107], a[56:59], a[174:177]
v_mfma_f32_16x16x32_f16 a[174:177], a[108:111], a[60:63], a[174:177]
v_mfma_f32_16x16x32_f16 a[188:191], a[112:115], a[56:59], a[188:191]
v_mfma_f32_16x16x32_f16 a[188:191], a[116:119], a[60:63], a[188:191]
v_mfma_f32_16x16x32_f16 a[224:227], a[120:123], a[56:59], a[224:227]
v_mfma_f32_16x16x32_f16 a[224:227], a[124:127], a[60:63], a[224:227]
v_mfma_f32_16x16x32_f16 a[228:231], a[96:99], a[48:51], a[228:231]
v_mfma_f32_16x16x32_f16 a[228:231], a[100:103], a[52:55], a[228:231]
v_mfma_f32_16x16x32_f16 a[232:235], a[104:107], a[48:51], a[232:235]
v_mfma_f32_16x16x32_f16 a[232:235], a[108:111], a[52:55], a[232:235]
v_mfma_f32_16x16x32_f16 a[236:239], a[112:115], a[48:51], a[236:239]
v_mfma_f32_16x16x32_f16 a[236:239], a[116:119], a[52:55], a[236:239]
v_mfma_f32_16x16x32_f16 v[34:37], a[120:123], a[48:51], v[34:37]
v_mfma_f32_16x16x32_f16 v[34:37], a[124:127], a[52:55], v[34:37]
v_mfma_f32_16x16x32_f16 v[38:41], a[96:99], a[40:43], v[38:41]
v_mfma_f32_16x16x32_f16 v[38:41], a[100:103], a[44:47], v[38:41]
v_mfma_f32_16x16x32_f16 v[42:45], a[104:107], a[40:43], v[42:45]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[42:45], a[108:111], a[44:47], v[42:45]
s_barrier
ds_read_b128 a[128:131], v255, offset:33792
v_mfma_f32_16x16x32_f16 v[46:49], a[112:115], a[40:43], v[46:49]
ds_read_b128 a[144:147], v255, offset:33856
v_mfma_f32_16x16x32_f16 v[46:49], a[116:119], a[44:47], v[46:49]
ds_read_b128 a[148:151], v255, offset:33920
v_mfma_f32_16x16x32_f16 v[50:53], a[120:123], a[40:43], v[50:53]
ds_read_b128 a[154:157], v255, offset:33984
v_mfma_f32_16x16x32_f16 v[50:53], a[124:127], a[44:47], v[50:53]
ds_read_b128 a[158:161], v255, offset:34304
v_mfma_f32_16x16x32_f16 v[54:57], a[96:99], a[32:35], v[54:57]
ds_read_b128 a[162:165], v255, offset:34368
v_mfma_f32_16x16x32_f16 v[54:57], a[100:103], a[36:39], v[54:57]
ds_read_b128 a[192:195], v255, offset:34432
v_mfma_f32_16x16x32_f16 a[240:243], a[104:107], a[32:35], a[240:243]
ds_read_b128 a[196:199], v255, offset:34496
v_mfma_f32_16x16x32_f16 a[240:243], a[108:111], a[36:39], a[240:243]
ds_read_b128 a[200:203], v255, offset:50688
v_mfma_f32_16x16x32_f16 a[244:247], a[112:115], a[32:35], a[244:247]
ds_read_b128 a[204:207], v255, offset:50752
v_mfma_f32_16x16x32_f16 a[244:247], a[116:119], a[36:39], a[244:247]
ds_read_b128 a[208:211], v255, offset:50816
v_mfma_f32_16x16x32_f16 a[248:251], a[120:123], a[32:35], a[248:251]
ds_read_b128 a[212:215], v255, offset:50880
v_mfma_f32_16x16x32_f16 a[248:251], a[124:127], a[36:39], a[248:251]
ds_read_b128 a[216:219], v255, offset:51200
v_mfma_f32_16x16x32_f16 a[252:255], a[96:99], a[24:27], a[252:255]
ds_read_b128 a[220:223], v255, offset:51264
v_mfma_f32_16x16x32_f16 a[252:255], a[100:103], a[28:31], a[252:255]
ds_read_b128 v[70:73], v255, offset:51328
v_mfma_f32_16x16x32_f16 v[74:77], a[104:107], a[24:27], v[74:77]
ds_read_b128 v[118:121], v255, offset:51392
v_mfma_f32_16x16x32_f16 v[74:77], a[108:111], a[28:31], v[74:77]
ds_read_b128 a[68:71], v94
v_mfma_f32_16x16x32_f16 v[78:81], a[112:115], a[24:27], v[78:81]
ds_read_b128 a[76:79], v94, offset:64
v_mfma_f32_16x16x32_f16 v[78:81], a[116:119], a[28:31], v[78:81]
ds_read_b128 a[64:67], v94, offset:128
v_mfma_f32_16x16x32_f16 v[82:85], a[120:123], a[24:27], v[82:85]
ds_read_b128 a[72:75], v94, offset:192
v_mfma_f32_16x16x32_f16 v[82:85], a[124:127], a[28:31], v[82:85]
ds_read_b128 a[80:83], v94, offset:512
v_mfma_f32_16x16x32_f16 v[86:89], a[96:99], a[16:19], v[86:89]
ds_read_b128 a[84:87], v94, offset:576
v_mfma_f32_16x16x32_f16 v[86:89], a[100:103], a[20:23], v[86:89]
ds_read_b128 a[88:91], v94, offset:640
v_mfma_f32_16x16x32_f16 v[90:93], a[104:107], a[16:19], v[90:93]
ds_read_b128 a[92:95], v94, offset:704
v_mfma_f32_16x16x32_f16 v[90:93], a[108:111], a[20:23], v[90:93]
s_add_u32 s16, s20, 0x80
s_addc_u32 s17, s21, 0
v_mfma_f32_16x16x32_f16 a[180:183], a[112:115], a[16:19], a[180:183]
s_and_b32 s17, s17, 0xffff
v_cndmask_b32_e32 v30, v30, v33, vcc
v_mfma_f32_16x16x32_f16 a[180:183], a[116:119], a[20:23], a[180:183]
s_mov_b32 m0, s30
v_mfma_f32_16x16x32_f16 v[98:101], a[120:123], a[16:19], v[98:101]
buffer_load_dwordx4 v30, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[98:101], a[124:127], a[20:23], v[98:101]
v_mfma_f32_16x16x32_f16 v[102:105], a[96:99], a[8:11], v[102:105]
v_mfma_f32_16x16x32_f16 v[102:105], a[100:103], a[12:15], v[102:105]
v_cndmask_b32_e32 v7, v7, v33, vcc
s_mov_b32 m0, s31
v_mfma_f32_16x16x32_f16 v[106:109], a[104:107], a[8:11], v[106:109]
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[106:109], a[108:111], a[12:15], v[106:109]
v_mfma_f32_16x16x32_f16 v[110:113], a[112:115], a[8:11], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[116:119], a[12:15], v[110:113]
v_cndmask_b32_e32 v22, v22, v33, vcc
s_mov_b32 m0, s33
v_mfma_f32_16x16x32_f16 v[114:117], a[120:123], a[8:11], v[114:117]
buffer_load_dwordx4 v22, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[114:117], a[124:127], a[12:15], v[114:117]
v_mfma_f32_16x16x32_f16 a[184:187], a[96:99], a[0:3], a[184:187]
v_mfma_f32_16x16x32_f16 a[184:187], a[100:103], a[4:7], a[184:187]
v_cndmask_b32_e32 v28, v28, v33, vcc
s_mov_b32 m0, s34
v_mfma_f32_16x16x32_f16 a[132:135], a[104:107], a[0:3], a[132:135]
buffer_load_dwordx4 v28, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[132:135], a[108:111], a[4:7], a[132:135]
v_mfma_f32_16x16x32_f16 a[136:139], a[112:115], a[0:3], a[136:139]
v_mfma_f32_16x16x32_f16 a[136:139], a[116:119], a[4:7], a[136:139]
v_mfma_f32_16x16x32_f16 a[140:143], a[120:123], a[0:3], a[140:143]
v_mfma_f32_16x16x32_f16 a[140:143], a[124:127], a[4:7], a[140:143]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[14:17], a[68:71], a[128:131], v[14:17]
v_mfma_f32_16x16x32_f16 v[14:17], a[76:79], a[144:147], v[14:17]
v_mfma_f32_16x16x32_f16 v[134:137], a[64:67], a[128:131], v[134:137]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], a[144:147], v[134:137]
s_barrier
ds_read_b128 a[96:99], v31
v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[128:131], v[138:141]
ds_read_b128 a[100:103], v31, offset:64
v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], a[144:147], v[138:141]
ds_read_b128 a[104:107], v31, offset:128
v_mfma_f32_16x16x32_f16 v[142:145], a[88:91], a[128:131], v[142:145]
ds_read_b128 a[108:111], v31, offset:192
v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[144:147], v[142:145]
ds_read_b128 a[112:115], v31, offset:512
v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[148:151], v[146:149]
ds_read_b128 a[116:119], v31, offset:576
v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], a[154:157], v[146:149]
ds_read_b128 a[120:123], v31, offset:640
v_mfma_f32_16x16x32_f16 v[150:153], a[64:67], a[148:151], v[150:153]
ds_read_b128 a[124:127], v31, offset:704
v_mfma_f32_16x16x32_f16 v[150:153], a[72:75], a[154:157], v[150:153]
s_and_b32 s53, s3, 0xffff
s_mov_b32 s52, s2
v_mfma_f32_16x16x32_f16 v[158:161], a[80:83], a[148:151], v[158:161]
s_mov_b32 m0, s35
v_mfma_f32_16x16x32_f16 v[158:161], a[84:87], a[154:157], v[158:161]
v_mfma_f32_16x16x32_f16 v[154:157], a[88:91], a[148:151], v[154:157]
buffer_load_dwordx4 v32, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[154:157], a[92:95], a[154:157], v[154:157]
v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[158:161], v[162:165]
v_mfma_f32_16x16x32_f16 v[162:165], a[76:79], a[162:165], v[162:165]
s_mov_b32 m0, s36
v_mfma_f32_16x16x32_f16 v[166:169], a[64:67], a[158:161], v[166:169]
buffer_load_dwordx4 v18, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[162:165], v[166:169]
v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[158:161], v[170:173]
v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[162:165], v[170:173]
s_mov_b32 m0, s37
v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[158:161], v[174:177]
buffer_load_dwordx4 v26, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[162:165], v[174:177]
v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[192:195], v[178:181]
v_mfma_f32_16x16x32_f16 v[178:181], a[76:79], a[196:199], v[178:181]
s_mov_b32 m0, s38
v_mfma_f32_16x16x32_f16 v[182:185], a[64:67], a[192:195], v[182:185]
buffer_load_dwordx4 v19, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[196:199], v[182:185]
v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[192:195], v[186:189]
v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[196:199], v[186:189]
s_mov_b32 m0, s39
v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[192:195], v[190:193]
buffer_load_dwordx4 v27, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[196:199], v[190:193]
v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[200:203], v[194:197]
v_mfma_f32_16x16x32_f16 v[194:197], a[76:79], a[204:207], v[194:197]
s_mov_b32 m0, s40
v_mfma_f32_16x16x32_f16 v[198:201], a[64:67], a[200:203], v[198:201]
buffer_load_dwordx4 v20, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[204:207], v[198:201]
v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[200:203], v[202:205]
v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[204:207], v[202:205]
s_mov_b32 m0, s41
v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[200:203], v[206:209]
buffer_load_dwordx4 v29, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[204:207], v[206:209]
v_mfma_f32_16x16x32_f16 v[214:217], a[68:71], a[208:211], v[214:217]
v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[212:215], v[214:217]
s_mov_b32 m0, s42
v_mfma_f32_16x16x32_f16 v[218:221], a[64:67], a[208:211], v[218:221]
buffer_load_dwordx4 v21, s[52:55], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[218:221], a[72:75], a[212:215], v[218:221]
v_mfma_f32_16x16x32_f16 v[230:233], a[80:83], a[208:211], v[230:233]
v_mfma_f32_16x16x32_f16 v[230:233], a[84:87], a[212:215], v[230:233]
s_mov_b32 m0, s43
v_mfma_f32_16x16x32_f16 v[210:213], a[88:91], a[208:211], v[210:213]
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[210:213], a[92:95], a[212:215], v[210:213]
v_mfma_f32_16x16x32_f16 v[222:225], a[68:71], a[216:219], v[222:225]
v_mfma_f32_16x16x32_f16 v[222:225], a[76:79], a[220:223], v[222:225]
s_mov_b32 m0, s44
v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[216:219], v[226:229]
buffer_load_dwordx4 v24, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[226:229], a[72:75], a[220:223], v[226:229]
v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[216:219], v[234:237]
v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[220:223], v[234:237]
s_mov_b32 m0, s45
v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[216:219], v[238:241]
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[220:223], v[238:241]
v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], v[70:73], v[242:245]
v_mfma_f32_16x16x32_f16 v[242:245], a[76:79], v[118:121], v[242:245]
s_mov_b32 m0, s46
v_mfma_f32_16x16x32_f16 v[246:249], a[64:67], v[70:73], v[246:249]
buffer_load_dwordx4 v25, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], v[118:121], v[246:249]
v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], v[70:73], v[250:253]
v_mfma_f32_16x16x32_f16 v[250:253], a[84:87], v[118:121], v[250:253]
v_mfma_f32_16x16x32_f16 v[2:5], a[88:91], v[70:73], v[2:5]
v_mfma_f32_16x16x32_f16 v[2:5], a[92:95], v[118:121], v[2:5]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[170:173], a[96:99], a[128:131], a[170:173]
v_mfma_f32_16x16x32_f16 a[170:173], a[100:103], a[144:147], a[170:173]
v_mfma_f32_16x16x32_f16 a[174:177], a[104:107], a[128:131], a[174:177]
v_mfma_f32_16x16x32_f16 a[174:177], a[108:111], a[144:147], a[174:177]
v_mfma_f32_16x16x32_f16 a[188:191], a[112:115], a[128:131], a[188:191]
v_mfma_f32_16x16x32_f16 a[188:191], a[116:119], a[144:147], a[188:191]
v_mfma_f32_16x16x32_f16 a[224:227], a[120:123], a[128:131], a[224:227]
v_mfma_f32_16x16x32_f16 a[224:227], a[124:127], a[144:147], a[224:227]
v_mfma_f32_16x16x32_f16 a[228:231], a[96:99], a[148:151], a[228:231]
v_mfma_f32_16x16x32_f16 a[228:231], a[100:103], a[154:157], a[228:231]
v_mfma_f32_16x16x32_f16 a[232:235], a[104:107], a[148:151], a[232:235]
v_mfma_f32_16x16x32_f16 a[232:235], a[108:111], a[154:157], a[232:235]
v_mfma_f32_16x16x32_f16 a[236:239], a[112:115], a[148:151], a[236:239]
v_mfma_f32_16x16x32_f16 a[236:239], a[116:119], a[154:157], a[236:239]
v_mfma_f32_16x16x32_f16 v[34:37], a[120:123], a[148:151], v[34:37]
v_mfma_f32_16x16x32_f16 v[34:37], a[124:127], a[154:157], v[34:37]
v_mfma_f32_16x16x32_f16 v[38:41], a[96:99], a[158:161], v[38:41]
v_mfma_f32_16x16x32_f16 v[38:41], a[100:103], a[162:165], v[38:41]
v_mfma_f32_16x16x32_f16 v[42:45], a[104:107], a[158:161], v[42:45]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[42:45], a[108:111], a[162:165], v[42:45]
s_barrier
ds_read_b128 a[56:59], v255
v_mfma_f32_16x16x32_f16 v[46:49], a[112:115], a[158:161], v[46:49]
ds_read_b128 a[60:63], v255, offset:64
v_mfma_f32_16x16x32_f16 v[46:49], a[116:119], a[162:165], v[46:49]
ds_read_b128 a[48:51], v255, offset:128
v_mfma_f32_16x16x32_f16 v[50:53], a[120:123], a[158:161], v[50:53]
ds_read_b128 a[52:55], v255, offset:192
v_mfma_f32_16x16x32_f16 v[50:53], a[124:127], a[162:165], v[50:53]
ds_read_b128 a[40:43], v255, offset:512
v_mfma_f32_16x16x32_f16 v[54:57], a[96:99], a[192:195], v[54:57]
ds_read_b128 a[44:47], v255, offset:576
v_mfma_f32_16x16x32_f16 v[54:57], a[100:103], a[196:199], v[54:57]
ds_read_b128 a[32:35], v255, offset:640
v_mfma_f32_16x16x32_f16 a[240:243], a[104:107], a[192:195], a[240:243]
ds_read_b128 a[36:39], v255, offset:704
v_mfma_f32_16x16x32_f16 a[240:243], a[108:111], a[196:199], a[240:243]
ds_read_b128 a[24:27], v255, offset:16896
v_mfma_f32_16x16x32_f16 a[244:247], a[112:115], a[192:195], a[244:247]
ds_read_b128 a[28:31], v255, offset:16960
v_mfma_f32_16x16x32_f16 a[244:247], a[116:119], a[196:199], a[244:247]
ds_read_b128 a[16:19], v255, offset:17024
v_mfma_f32_16x16x32_f16 a[248:251], a[120:123], a[192:195], a[248:251]
ds_read_b128 a[20:23], v255, offset:17088
v_mfma_f32_16x16x32_f16 a[248:251], a[124:127], a[196:199], a[248:251]
ds_read_b128 a[8:11], v255, offset:17408
v_mfma_f32_16x16x32_f16 a[252:255], a[96:99], a[200:203], a[252:255]
ds_read_b128 a[12:15], v255, offset:17472
v_mfma_f32_16x16x32_f16 a[252:255], a[100:103], a[204:207], a[252:255]
ds_read_b128 a[0:3], v255, offset:17536
v_mfma_f32_16x16x32_f16 v[74:77], a[104:107], a[200:203], v[74:77]
ds_read_b128 a[4:7], v255, offset:17600
v_mfma_f32_16x16x32_f16 v[74:77], a[108:111], a[204:207], v[74:77]
ds_read_b128 a[68:71], v13
v_mfma_f32_16x16x32_f16 v[78:81], a[112:115], a[200:203], v[78:81]
ds_read_b128 a[76:79], v13, offset:64
v_mfma_f32_16x16x32_f16 v[78:81], a[116:119], a[204:207], v[78:81]
ds_read_b128 a[64:67], v13, offset:128
v_mfma_f32_16x16x32_f16 v[82:85], a[120:123], a[200:203], v[82:85]
ds_read_b128 a[72:75], v13, offset:192
v_mfma_f32_16x16x32_f16 v[82:85], a[124:127], a[204:207], v[82:85]
ds_read_b128 a[80:83], v13, offset:512
v_mfma_f32_16x16x32_f16 v[86:89], a[96:99], a[208:211], v[86:89]
ds_read_b128 a[84:87], v13, offset:576
v_mfma_f32_16x16x32_f16 v[86:89], a[100:103], a[212:215], v[86:89]
ds_read_b128 a[88:91], v13, offset:640
v_mfma_f32_16x16x32_f16 v[90:93], a[104:107], a[208:211], v[90:93]
ds_read_b128 a[92:95], v13, offset:704
v_mfma_f32_16x16x32_f16 v[90:93], a[108:111], a[212:215], v[90:93]
s_add_u32 s20, s20, 0x100
s_addc_u32 s21, s21, 0
v_mfma_f32_16x16x32_f16 a[180:183], a[112:115], a[208:211], a[180:183]
s_and_b32 s17, s21, 0xffff
s_mov_b32 s16, s20
v_mfma_f32_16x16x32_f16 a[180:183], a[116:119], a[212:215], a[180:183]
s_mov_b32 m0, s22
v_mfma_f32_16x16x32_f16 v[98:101], a[120:123], a[208:211], v[98:101]
buffer_load_dwordx4 v30, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[98:101], a[124:127], a[212:215], v[98:101]
v_mfma_f32_16x16x32_f16 v[102:105], a[96:99], a[216:219], v[102:105]
v_mfma_f32_16x16x32_f16 v[102:105], a[100:103], a[220:223], v[102:105]
s_mov_b32 m0, s23
v_mfma_f32_16x16x32_f16 v[106:109], a[104:107], a[216:219], v[106:109]
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[106:109], a[108:111], a[220:223], v[106:109]
v_mfma_f32_16x16x32_f16 v[110:113], a[112:115], a[216:219], v[110:113]
v_mfma_f32_16x16x32_f16 v[110:113], a[116:119], a[220:223], v[110:113]
s_mov_b32 m0, s47
v_mfma_f32_16x16x32_f16 v[114:117], a[120:123], a[216:219], v[114:117]
buffer_load_dwordx4 v22, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 v[114:117], a[124:127], a[220:223], v[114:117]
v_mfma_f32_16x16x32_f16 a[184:187], a[96:99], v[70:73], a[184:187]
v_mfma_f32_16x16x32_f16 a[184:187], a[100:103], v[118:121], a[184:187]
s_mov_b32 m0, s48
v_mfma_f32_16x16x32_f16 a[132:135], a[104:107], v[70:73], a[132:135]
buffer_load_dwordx4 v28, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[132:135], a[108:111], v[118:121], a[132:135]
s_add_u32 s2, s2, 0x100
s_addc_u32 s3, s3, 0
v_mfma_f32_16x16x32_f16 a[136:139], a[112:115], v[70:73], a[136:139]
s_add_i32 s10, s10, 2
s_cmp_lt_u32 s10, 61
v_mfma_f32_16x16x32_f16 a[136:139], a[116:119], v[118:121], a[136:139]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[140:143], a[120:123], v[70:73], a[140:143]
v_mfma_f32_16x16x32_f16 a[140:143], a[124:127], v[118:121], a[140:143]
s_cbranch_scc1 .LBB0_1
; %bb.2:
v_accvgpr_write_b32 a179, v254
s_and_b64 s[2:3], s[4:5], exec
s_cselect_b32 s2, 0, 32
v_accvgpr_read_b32 v0, a169
v_or_b32_e32 v0, s2, v0
v_accvgpr_read_b32 v6, a152
v_and_b32_e32 v1, 16, v6
v_lshrrev_b32_e32 v6, 2, v6
v_and_b32_e32 v6, 8, v6
s_lshr_b32 s2, s49, 1
v_or3_b32 v254, v6, v1, s2
v_or_b32_e32 v255, 64, v254
s_mul_i32 s2, s1, s12
s_ashr_i32 s3, s2, 31
s_lshl_b64 s[2:3], s[2:3], 1
s_add_u32 s2, s6, s2
s_addc_u32 s3, s7, s3
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s4, s2, s0
s_addc_u32 s2, s3, s1
s_add_u32 s0, s4, 0x100
s_addc_u32 s1, s2, 0
v_mul_lo_u32 v0, v0, s12
s_lshl_b32 s3, s12, 4
v_add_u32_e32 v1, s3, v0
v_accvgpr_write_b32 a103, v0
s_mul_i32 s12, s12, 48
v_add_u32_e32 v0, s12, v1
v_accvgpr_write_b32 a102, v1
v_add_u32_e32 v1, s3, v0
v_accvgpr_write_b32 a101, v0
v_add_u32_e32 v0, s12, v1
v_accvgpr_write_b32 a100, v1
v_add_u32_e32 v1, s3, v0
v_accvgpr_write_b32 a99, v0
v_add_u32_e32 v0, s12, v1
v_accvgpr_write_b32 a98, v1
v_add_u32_e32 v1, s3, v0
v_accvgpr_write_b32 a97, v1
v_accvgpr_write_b32 a96, v0
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[56:59], v[14:17]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 v[6:9], a[76:79], a[60:63], v[6:9]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[56:59], v[134:137]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[60:63], v[10:13]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[56:59], v[138:141]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[60:63], v[14:17]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 v[18:21], a[88:91], a[56:59], v[142:145]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[18:21], a[92:95], a[60:63], v[18:21]
v_mfma_f32_16x16x32_f16 v[22:25], a[68:71], a[48:51], v[146:149]
v_mfma_f32_16x16x32_f16 v[22:25], a[76:79], a[52:55], v[22:25]
v_mfma_f32_16x16x32_f16 v[26:29], a[64:67], a[48:51], v[150:153]
v_mfma_f32_16x16x32_f16 v[26:29], a[72:75], a[52:55], v[26:29]
v_mfma_f32_16x16x32_f16 v[30:33], a[80:83], a[48:51], v[158:161]
v_mfma_f32_16x16x32_f16 v[30:33], a[84:87], a[52:55], v[30:33]
s_nop 1
v_mov_b64_e32 v[160:161], v[36:37]
v_mov_b64_e32 v[158:159], v[34:35]
v_mfma_f32_16x16x32_f16 v[34:37], a[88:91], a[48:51], v[154:157]
v_mfma_f32_16x16x32_f16 v[34:37], a[92:95], a[52:55], v[34:37]
s_nop 1
v_mov_b64_e32 v[156:157], v[40:41]
v_mov_b64_e32 v[154:155], v[38:39]
v_mfma_f32_16x16x32_f16 v[38:41], a[68:71], a[40:43], v[162:165]
v_mfma_f32_16x16x32_f16 v[38:41], a[76:79], a[44:47], v[38:41]
s_nop 1
v_mov_b64_e32 v[164:165], v[44:45]
v_mov_b64_e32 v[162:163], v[42:43]
v_mfma_f32_16x16x32_f16 v[42:45], a[64:67], a[40:43], v[166:169]
v_mfma_f32_16x16x32_f16 v[42:45], a[72:75], a[44:47], v[42:45]
s_nop 1
v_mov_b64_e32 v[168:169], v[48:49]
v_mov_b64_e32 v[166:167], v[46:47]
v_mfma_f32_16x16x32_f16 v[46:49], a[80:83], a[40:43], v[170:173]
v_mfma_f32_16x16x32_f16 v[46:49], a[84:87], a[44:47], v[46:49]
s_nop 1
v_mov_b64_e32 v[172:173], v[52:53]
v_mov_b64_e32 v[170:171], v[50:51]
v_mfma_f32_16x16x32_f16 v[50:53], a[88:91], a[40:43], v[174:177]
v_mfma_f32_16x16x32_f16 v[50:53], a[92:95], a[44:47], v[50:53]
s_nop 1
v_mov_b64_e32 v[176:177], v[56:57]
v_mov_b64_e32 v[174:175], v[54:55]
v_mfma_f32_16x16x32_f16 v[54:57], a[68:71], a[32:35], v[178:181]
v_mfma_f32_16x16x32_f16 v[54:57], a[76:79], a[36:39], v[54:57]
v_mfma_f32_16x16x32_f16 v[58:61], a[64:67], a[32:35], v[182:185]
v_mfma_f32_16x16x32_f16 v[58:61], a[72:75], a[36:39], v[58:61]
v_mfma_f32_16x16x32_f16 v[62:65], a[80:83], a[32:35], v[186:189]
v_mfma_f32_16x16x32_f16 v[62:65], a[84:87], a[36:39], v[62:65]
v_mfma_f32_16x16x32_f16 v[66:69], a[88:91], a[32:35], v[190:193]
v_mfma_f32_16x16x32_f16 v[66:69], a[92:95], a[36:39], v[66:69]
v_mfma_f32_16x16x32_f16 v[70:73], a[68:71], a[24:27], v[194:197]
v_mfma_f32_16x16x32_f16 v[70:73], a[76:79], a[28:31], v[70:73]
v_accvgpr_write_b32 a151, v77
v_accvgpr_write_b32 a150, v76
v_accvgpr_write_b32 a149, v75
v_accvgpr_write_b32 a148, v74
v_mfma_f32_16x16x32_f16 v[74:77], a[64:67], a[24:27], v[198:201]
v_mfma_f32_16x16x32_f16 v[74:77], a[72:75], a[28:31], v[74:77]
v_accvgpr_write_b32 a155, v81
v_accvgpr_write_b32 a154, v80
v_accvgpr_write_b32 a153, v79
v_accvgpr_write_b32 a152, v78
v_mfma_f32_16x16x32_f16 v[78:81], a[80:83], a[24:27], v[202:205]
v_mfma_f32_16x16x32_f16 v[78:81], a[84:87], a[28:31], v[78:81]
v_accvgpr_write_b32 a159, v85
v_accvgpr_write_b32 a158, v84
v_accvgpr_write_b32 a157, v83
v_accvgpr_write_b32 a156, v82
v_mfma_f32_16x16x32_f16 v[82:85], a[88:91], a[24:27], v[206:209]
v_mfma_f32_16x16x32_f16 v[82:85], a[92:95], a[28:31], v[82:85]
v_accvgpr_write_b32 a163, v89
v_accvgpr_write_b32 a162, v88
v_accvgpr_write_b32 a161, v87
v_accvgpr_write_b32 a160, v86
v_mfma_f32_16x16x32_f16 v[86:89], a[68:71], a[16:19], v[214:217]
v_mfma_f32_16x16x32_f16 v[86:89], a[76:79], a[20:23], v[86:89]
s_nop 1
v_mov_b64_e32 v[216:217], v[92:93]
v_mov_b64_e32 v[214:215], v[90:91]
v_mfma_f32_16x16x32_f16 v[90:93], a[64:67], a[16:19], v[218:221]
v_mfma_f32_16x16x32_f16 v[90:93], a[72:75], a[20:23], v[90:93]
v_mfma_f32_16x16x32_f16 v[94:97], a[80:83], a[16:19], v[230:233]
v_mfma_f32_16x16x32_f16 v[94:97], a[84:87], a[20:23], v[94:97]
v_mov_b64_e32 v[220:221], v[100:101]
v_mov_b64_e32 v[218:219], v[98:99]
v_mfma_f32_16x16x32_f16 v[98:101], a[88:91], a[16:19], v[210:213]
v_mfma_f32_16x16x32_f16 v[98:101], a[92:95], a[20:23], v[98:101]
s_nop 1
v_mov_b64_e32 v[212:213], v[104:105]
v_mov_b64_e32 v[210:211], v[102:103]
v_mfma_f32_16x16x32_f16 v[102:105], a[68:71], a[8:11], v[222:225]
v_mfma_f32_16x16x32_f16 v[102:105], a[76:79], a[12:15], v[102:105]
s_nop 1
v_mov_b64_e32 v[224:225], v[108:109]
v_mov_b64_e32 v[222:223], v[106:107]
v_mfma_f32_16x16x32_f16 v[106:109], a[64:67], a[8:11], v[226:229]
v_mfma_f32_16x16x32_f16 v[106:109], a[72:75], a[12:15], v[106:109]
s_nop 1
v_mov_b64_e32 v[228:229], v[112:113]
v_mov_b64_e32 v[226:227], v[110:111]
v_mfma_f32_16x16x32_f16 v[110:113], a[80:83], a[8:11], v[234:237]
v_mfma_f32_16x16x32_f16 v[110:113], a[84:87], a[12:15], v[110:113]
v_mov_b64_e32 v[232:233], v[116:117]
v_mov_b64_e32 v[230:231], v[114:115]
v_mfma_f32_16x16x32_f16 v[114:117], a[88:91], a[8:11], v[238:241]
v_mfma_f32_16x16x32_f16 v[114:117], a[92:95], a[12:15], v[114:117]
v_mfma_f32_16x16x32_f16 v[118:121], a[68:71], a[0:3], v[242:245]
v_mfma_f32_16x16x32_f16 v[118:121], a[76:79], a[4:7], v[118:121]
v_mfma_f32_16x16x32_f16 v[122:125], a[64:67], a[0:3], v[246:249]
v_mfma_f32_16x16x32_f16 v[122:125], a[72:75], a[4:7], v[122:125]
v_mfma_f32_16x16x32_f16 v[126:129], a[80:83], a[0:3], v[250:253]
v_mfma_f32_16x16x32_f16 v[126:129], a[84:87], a[4:7], v[126:129]
v_mfma_f32_16x16x32_f16 v[2:5], a[88:91], a[0:3], v[2:5]
v_mfma_f32_16x16x32_f16 v[150:153], a[92:95], a[4:7], v[2:5]
s_waitcnt vmcnt(0), lgkmcnt(0)
s_barrier
v_accvgpr_read_b32 v0, a179
v_add_u32_e32 v0, 0x1cdc0, v0
ds_read_b128 v[178:181], v0
ds_read_b128 v[182:185], v0, offset:64
ds_read_b128 v[186:189], v0, offset:128
ds_read_b128 v[190:193], v0, offset:192
ds_read_b128 v[194:197], v0, offset:512
ds_read_b128 v[198:201], v0, offset:576
ds_read_b128 v[202:205], v0, offset:640
ds_read_b128 v[206:209], v0, offset:704
v_cvt_pk_f16_f32 v4, v6, v7
v_cvt_pk_f16_f32 v5, v8, v9
v_cvt_pk_f16_f32 v6, v10, v11
v_cvt_pk_f16_f32 v7, v12, v13
v_cvt_pk_f16_f32 v8, v14, v15
v_cvt_pk_f16_f32 v9, v16, v17
v_cvt_pk_f16_f32 v10, v18, v19
v_cvt_pk_f16_f32 v11, v20, v21
v_cvt_pk_f16_f32 v12, v22, v23
v_cvt_pk_f16_f32 v13, v24, v25
v_cvt_pk_f16_f32 v14, v26, v27
v_cvt_pk_f16_f32 v15, v28, v29
v_cvt_pk_f16_f32 v16, v30, v31
v_cvt_pk_f16_f32 v17, v32, v33
v_cvt_pk_f16_f32 v18, v34, v35
v_cvt_pk_f16_f32 v19, v36, v37
v_cvt_pk_f16_f32 v20, v38, v39
v_cvt_pk_f16_f32 v21, v40, v41
v_cvt_pk_f16_f32 v22, v42, v43
v_cvt_pk_f16_f32 v23, v44, v45
v_cvt_pk_f16_f32 v24, v46, v47
v_cvt_pk_f16_f32 v25, v48, v49
v_cvt_pk_f16_f32 v26, v50, v51
v_cvt_pk_f16_f32 v27, v52, v53
v_cvt_pk_f16_f32 v28, v54, v55
v_cvt_pk_f16_f32 v29, v56, v57
v_cvt_pk_f16_f32 v30, v58, v59
v_cvt_pk_f16_f32 v31, v60, v61
v_cvt_pk_f16_f32 v32, v62, v63
v_cvt_pk_f16_f32 v33, v64, v65
v_cvt_pk_f16_f32 v34, v66, v67
v_cvt_pk_f16_f32 v35, v68, v69
v_cvt_pk_f16_f32 v36, v70, v71
v_cvt_pk_f16_f32 v37, v72, v73
v_cvt_pk_f16_f32 v38, v74, v75
v_cvt_pk_f16_f32 v39, v76, v77
v_cvt_pk_f16_f32 v134, v78, v79
v_cvt_pk_f16_f32 v135, v80, v81
v_cvt_pk_f16_f32 v136, v82, v83
v_cvt_pk_f16_f32 v137, v84, v85
v_cvt_pk_f16_f32 v40, v86, v87
v_cvt_pk_f16_f32 v41, v88, v89
v_cvt_pk_f16_f32 v42, v90, v91
v_cvt_pk_f16_f32 v43, v92, v93
v_cvt_pk_f16_f32 v142, v94, v95
v_cvt_pk_f16_f32 v143, v96, v97
v_cvt_pk_f16_f32 v144, v98, v99
v_cvt_pk_f16_f32 v145, v100, v101
v_cvt_pk_f16_f32 v138, v102, v103
v_cvt_pk_f16_f32 v139, v104, v105
v_cvt_pk_f16_f32 v140, v106, v107
v_cvt_pk_f16_f32 v141, v108, v109
v_cvt_pk_f16_f32 v0, v110, v111
v_cvt_pk_f16_f32 v1, v112, v113
v_cvt_pk_f16_f32 v2, v114, v115
v_cvt_pk_f16_f32 v3, v116, v117
v_cvt_pk_f16_f32 v146, v118, v119
v_cvt_pk_f16_f32 v147, v120, v121
v_cvt_pk_f16_f32 v148, v122, v123
v_cvt_pk_f16_f32 v149, v124, v125
v_cvt_pk_f16_f32 v130, v126, v127
v_cvt_pk_f16_f32 v131, v128, v129
v_cvt_pk_f16_f32 v132, v150, v151
v_cvt_pk_f16_f32 v133, v152, v153
v_permlane16_swap_b32_e32 v4, v6
v_permlane16_swap_b32_e32 v5, v7
v_permlane16_swap_b32_e32 v12, v14
v_permlane16_swap_b32_e32 v13, v15
v_permlane16_swap_b32_e32 v8, v10
v_permlane16_swap_b32_e32 v9, v11
v_permlane16_swap_b32_e32 v16, v18
v_permlane16_swap_b32_e32 v17, v19
v_permlane16_swap_b32_e32 v20, v22
v_permlane16_swap_b32_e32 v21, v23
v_permlane16_swap_b32_e32 v28, v30
v_permlane16_swap_b32_e32 v29, v31
v_permlane16_swap_b32_e32 v24, v26
v_permlane16_swap_b32_e32 v25, v27
v_permlane16_swap_b32_e32 v32, v34
v_permlane16_swap_b32_e32 v33, v35
v_permlane16_swap_b32_e32 v36, v38
v_permlane16_swap_b32_e32 v37, v39
v_permlane16_swap_b32_e32 v40, v42
v_permlane16_swap_b32_e32 v41, v43
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
v_accvgpr_read_b32 v44, a170
v_accvgpr_read_b32 v45, a171
v_accvgpr_read_b32 v46, a172
v_accvgpr_read_b32 v47, a173
s_waitcnt lgkmcnt(7)
s_nop 0
v_mfma_f32_16x16x32_f16 v[44:47], v[178:181], a[56:59], v[44:47]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 v[44:47], v[182:185], a[60:63], v[44:47]
v_accvgpr_read_b32 v48, a174
v_accvgpr_read_b32 v49, a175
v_accvgpr_read_b32 v50, a176
v_accvgpr_read_b32 v51, a177
s_waitcnt lgkmcnt(5)
s_nop 0
v_mfma_f32_16x16x32_f16 v[48:51], v[186:189], a[56:59], v[48:51]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 v[48:51], v[190:193], a[60:63], v[48:51]
v_accvgpr_read_b32 v52, a188
v_accvgpr_read_b32 v53, a189
v_accvgpr_read_b32 v54, a190
v_accvgpr_read_b32 v55, a191
s_waitcnt lgkmcnt(3)
s_nop 0
v_mfma_f32_16x16x32_f16 v[52:55], v[194:197], a[56:59], v[52:55]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 v[52:55], v[198:201], a[60:63], v[52:55]
v_accvgpr_read_b32 v56, a224
v_accvgpr_read_b32 v57, a225
v_accvgpr_read_b32 v58, a226
v_accvgpr_read_b32 v59, a227
s_waitcnt lgkmcnt(1)
s_nop 0
v_mfma_f32_16x16x32_f16 v[56:59], v[202:205], a[56:59], v[56:59]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 v[56:59], v[206:209], a[60:63], v[56:59]
v_accvgpr_read_b32 v60, a228
v_accvgpr_read_b32 v61, a229
v_accvgpr_read_b32 v62, a230
v_accvgpr_read_b32 v63, a231
s_nop 1
v_mfma_f32_16x16x32_f16 v[60:63], v[178:181], a[48:51], v[60:63]
v_mfma_f32_16x16x32_f16 v[60:63], v[182:185], a[52:55], v[60:63]
v_accvgpr_read_b32 v64, a232
v_accvgpr_read_b32 v65, a233
v_accvgpr_read_b32 v66, a234
v_accvgpr_read_b32 v67, a235
s_nop 1
v_mfma_f32_16x16x32_f16 v[64:67], v[186:189], a[48:51], v[64:67]
v_mfma_f32_16x16x32_f16 v[64:67], v[190:193], a[52:55], v[64:67]
v_accvgpr_read_b32 v68, a236
v_accvgpr_read_b32 v69, a237
v_accvgpr_read_b32 v70, a238
v_accvgpr_read_b32 v71, a239
s_nop 1
v_mfma_f32_16x16x32_f16 v[68:71], v[194:197], a[48:51], v[68:71]
v_mfma_f32_16x16x32_f16 v[68:71], v[198:201], a[52:55], v[68:71]
v_mfma_f32_16x16x32_f16 v[72:75], v[202:205], a[48:51], v[158:161]
v_mfma_f32_16x16x32_f16 v[72:75], v[206:209], a[52:55], v[72:75]
v_mfma_f32_16x16x32_f16 v[76:79], v[178:181], a[40:43], v[154:157]
v_mfma_f32_16x16x32_f16 v[76:79], v[182:185], a[44:47], v[76:79]
v_mfma_f32_16x16x32_f16 v[80:83], v[186:189], a[40:43], v[162:165]
v_mfma_f32_16x16x32_f16 v[80:83], v[190:193], a[44:47], v[80:83]
v_mfma_f32_16x16x32_f16 v[84:87], v[194:197], a[40:43], v[166:169]
v_mfma_f32_16x16x32_f16 v[84:87], v[198:201], a[44:47], v[84:87]
v_mfma_f32_16x16x32_f16 v[88:91], v[202:205], a[40:43], v[170:173]
v_mfma_f32_16x16x32_f16 v[88:91], v[206:209], a[44:47], v[88:91]
v_mfma_f32_16x16x32_f16 v[92:95], v[178:181], a[32:35], v[174:177]
v_mfma_f32_16x16x32_f16 v[92:95], v[182:185], a[36:39], v[92:95]
v_accvgpr_read_b32 v96, a240
v_accvgpr_read_b32 v97, a241
v_accvgpr_read_b32 v98, a242
v_accvgpr_read_b32 v99, a243
s_nop 1
v_mfma_f32_16x16x32_f16 v[96:99], v[186:189], a[32:35], v[96:99]
v_mfma_f32_16x16x32_f16 v[96:99], v[190:193], a[36:39], v[96:99]
v_accvgpr_read_b32 v100, a244
v_accvgpr_read_b32 v101, a245
v_accvgpr_read_b32 v102, a246
v_accvgpr_read_b32 v103, a247
s_nop 1
v_mfma_f32_16x16x32_f16 v[100:103], v[194:197], a[32:35], v[100:103]
v_mfma_f32_16x16x32_f16 v[100:103], v[198:201], a[36:39], v[100:103]
v_accvgpr_read_b32 v104, a248
v_accvgpr_read_b32 v105, a249
v_accvgpr_read_b32 v106, a250
v_accvgpr_read_b32 v107, a251
s_nop 1
v_mfma_f32_16x16x32_f16 v[104:107], v[202:205], a[32:35], v[104:107]
v_mfma_f32_16x16x32_f16 v[104:107], v[206:209], a[36:39], v[104:107]
v_accvgpr_read_b32 v108, a252
v_accvgpr_read_b32 v109, a253
v_accvgpr_read_b32 v110, a254
v_accvgpr_read_b32 v111, a255
s_nop 1
v_mfma_f32_16x16x32_f16 v[108:111], v[178:181], a[24:27], v[108:111]
v_mfma_f32_16x16x32_f16 v[108:111], v[182:185], a[28:31], v[108:111]
v_accvgpr_read_b32 v112, a148
v_accvgpr_read_b32 v113, a149
v_accvgpr_read_b32 v114, a150
v_accvgpr_read_b32 v115, a151
s_nop 1
v_mfma_f32_16x16x32_f16 v[112:115], v[186:189], a[24:27], v[112:115]
v_mfma_f32_16x16x32_f16 v[112:115], v[190:193], a[28:31], v[112:115]
v_accvgpr_read_b32 v116, a152
v_accvgpr_read_b32 v117, a153
v_accvgpr_read_b32 v118, a154
v_accvgpr_read_b32 v119, a155
s_nop 1
v_mfma_f32_16x16x32_f16 v[116:119], v[194:197], a[24:27], v[116:119]
v_mfma_f32_16x16x32_f16 v[116:119], v[198:201], a[28:31], v[116:119]
v_accvgpr_read_b32 v120, a156
v_accvgpr_read_b32 v121, a157
v_accvgpr_read_b32 v122, a158
v_accvgpr_read_b32 v123, a159
s_nop 1
v_mfma_f32_16x16x32_f16 v[120:123], v[202:205], a[24:27], v[120:123]
v_mfma_f32_16x16x32_f16 v[120:123], v[206:209], a[28:31], v[120:123]
v_accvgpr_read_b32 v124, a160
v_accvgpr_read_b32 v125, a161
v_accvgpr_read_b32 v126, a162
v_accvgpr_read_b32 v127, a163
s_nop 1
v_mfma_f32_16x16x32_f16 v[124:127], v[178:181], a[16:19], v[124:127]
v_mfma_f32_16x16x32_f16 v[124:127], v[182:185], a[20:23], v[124:127]
v_mfma_f32_16x16x32_f16 v[150:153], v[186:189], a[16:19], v[214:217]
v_mfma_f32_16x16x32_f16 v[150:153], v[190:193], a[20:23], v[150:153]
v_accvgpr_read_b32 v154, a180
v_accvgpr_read_b32 v155, a181
v_accvgpr_read_b32 v156, a182
v_accvgpr_read_b32 v157, a183
s_nop 1
v_mfma_f32_16x16x32_f16 v[154:157], v[194:197], a[16:19], v[154:157]
v_mfma_f32_16x16x32_f16 v[154:157], v[198:201], a[20:23], v[154:157]
v_mfma_f32_16x16x32_f16 v[158:161], v[202:205], a[16:19], v[218:221]
v_mfma_f32_16x16x32_f16 v[158:161], v[206:209], a[20:23], v[158:161]
v_mfma_f32_16x16x32_f16 v[162:165], v[178:181], a[8:11], v[210:213]
v_mfma_f32_16x16x32_f16 v[162:165], v[182:185], a[12:15], v[162:165]
v_mfma_f32_16x16x32_f16 v[166:169], v[186:189], a[8:11], v[222:225]
v_mfma_f32_16x16x32_f16 v[166:169], v[190:193], a[12:15], v[166:169]
v_mfma_f32_16x16x32_f16 v[170:173], v[194:197], a[8:11], v[226:229]
v_mfma_f32_16x16x32_f16 v[170:173], v[198:201], a[12:15], v[170:173]
v_mfma_f32_16x16x32_f16 v[174:177], v[202:205], a[8:11], v[230:233]
v_mfma_f32_16x16x32_f16 v[174:177], v[206:209], a[12:15], v[174:177]
v_accvgpr_read_b32 v213, a187
v_accvgpr_read_b32 v212, a186
v_accvgpr_read_b32 v211, a185
v_accvgpr_read_b32 v210, a184
s_nop 1
v_mfma_f32_16x16x32_f16 v[178:181], v[178:181], a[0:3], v[210:213]
v_mfma_f32_16x16x32_f16 v[178:181], v[182:185], a[4:7], v[178:181]
v_accvgpr_read_b32 v185, a135
v_accvgpr_read_b32 v184, a134
v_accvgpr_read_b32 v183, a133
v_accvgpr_read_b32 v182, a132
s_nop 1
v_mfma_f32_16x16x32_f16 v[182:185], v[186:189], a[0:3], v[182:185]
v_mfma_f32_16x16x32_f16 v[182:185], v[190:193], a[4:7], v[182:185]
v_accvgpr_read_b32 v189, a139
v_accvgpr_read_b32 v188, a138
v_accvgpr_read_b32 v187, a137
v_accvgpr_read_b32 v186, a136
s_nop 1
v_mfma_f32_16x16x32_f16 v[186:189], v[194:197], a[0:3], v[186:189]
v_mfma_f32_16x16x32_f16 v[186:189], v[198:201], a[4:7], v[186:189]
v_accvgpr_read_b32 v193, a143
v_accvgpr_read_b32 v192, a142
v_accvgpr_read_b32 v191, a141
v_accvgpr_read_b32 v190, a140
s_nop 1
v_mfma_f32_16x16x32_f16 v[190:193], v[202:205], a[0:3], v[190:193]
v_mfma_f32_16x16x32_f16 v[190:193], v[206:209], a[4:7], v[190:193]
s_and_b32 s5, s2, 0xffff
s_mov_b32 s7, 0x27000
s_mov_b32 s6, 0x7ffffffe
v_accvgpr_read_b32 v194, a103
v_add_lshl_u32 v128, v194, v254, 1
buffer_store_dwordx4 v[4:7], v128, s[4:7], 0, offen
s_nop 1
v_accvgpr_read_b32 v4, a102
v_add_lshl_u32 v129, v4, v254, 1
buffer_store_dwordx4 v[12:15], v129, s[4:7], 0, offen
v_add_lshl_u32 v194, v194, v255, 1
buffer_store_dwordx4 v[8:11], v194, s[4:7], 0, offen
v_add_lshl_u32 v195, v4, v255, 1
buffer_store_dwordx4 v[16:19], v195, s[4:7], 0, offen
v_accvgpr_read_b32 v5, a101
v_add_lshl_u32 v196, v5, v254, 1
buffer_store_dwordx4 v[20:23], v196, s[4:7], 0, offen
v_accvgpr_read_b32 v4, a100
v_add_lshl_u32 v197, v4, v254, 1
buffer_store_dwordx4 v[28:31], v197, s[4:7], 0, offen
v_add_lshl_u32 v198, v5, v255, 1
buffer_store_dwordx4 v[24:27], v198, s[4:7], 0, offen
v_add_lshl_u32 v199, v4, v255, 1
buffer_store_dwordx4 v[32:35], v199, s[4:7], 0, offen
v_accvgpr_read_b32 v5, a99
v_add_lshl_u32 v200, v5, v254, 1
buffer_store_dwordx4 v[36:39], v200, s[4:7], 0, offen
v_accvgpr_read_b32 v4, a98
v_add_lshl_u32 v201, v4, v254, 1
buffer_store_dwordx4 v[40:43], v201, s[4:7], 0, offen
v_add_lshl_u32 v202, v5, v255, 1
buffer_store_dwordx4 v[134:137], v202, s[4:7], 0, offen
s_nop 1
v_add_lshl_u32 v134, v4, v255, 1
buffer_store_dwordx4 v[142:145], v134, s[4:7], 0, offen
v_accvgpr_read_b32 v5, a96
v_add_lshl_u32 v135, v5, v254, 1
buffer_store_dwordx4 v[138:141], v135, s[4:7], 0, offen
v_accvgpr_read_b32 v4, a97
v_add_lshl_u32 v136, v4, v254, 1
buffer_store_dwordx4 v[146:149], v136, s[4:7], 0, offen
v_add_lshl_u32 v137, v5, v255, 1
buffer_store_dwordx4 v[0:3], v137, s[4:7], 0, offen
v_add_lshl_u32 v138, v4, v255, 1
buffer_store_dwordx4 v[130:133], v138, s[4:7], 0, offen
v_cvt_pk_f16_f32 v0, v44, v45
v_cvt_pk_f16_f32 v1, v46, v47
v_cvt_pk_f16_f32 v2, v48, v49
v_cvt_pk_f16_f32 v3, v50, v51
v_cvt_pk_f16_f32 v4, v52, v53
v_cvt_pk_f16_f32 v5, v54, v55
v_cvt_pk_f16_f32 v6, v56, v57
v_cvt_pk_f16_f32 v7, v58, v59
v_cvt_pk_f16_f32 v8, v60, v61
v_cvt_pk_f16_f32 v9, v62, v63
v_cvt_pk_f16_f32 v10, v64, v65
v_cvt_pk_f16_f32 v11, v66, v67
v_cvt_pk_f16_f32 v12, v68, v69
v_cvt_pk_f16_f32 v13, v70, v71
v_cvt_pk_f16_f32 v14, v72, v73
v_cvt_pk_f16_f32 v15, v74, v75
v_cvt_pk_f16_f32 v16, v76, v77
v_cvt_pk_f16_f32 v17, v78, v79
v_cvt_pk_f16_f32 v18, v80, v81
v_cvt_pk_f16_f32 v19, v82, v83
v_cvt_pk_f16_f32 v20, v84, v85
v_cvt_pk_f16_f32 v21, v86, v87
v_cvt_pk_f16_f32 v22, v88, v89
v_cvt_pk_f16_f32 v23, v90, v91
v_cvt_pk_f16_f32 v24, v92, v93
v_cvt_pk_f16_f32 v25, v94, v95
v_cvt_pk_f16_f32 v26, v96, v97
v_cvt_pk_f16_f32 v27, v98, v99
v_cvt_pk_f16_f32 v28, v100, v101
v_cvt_pk_f16_f32 v29, v102, v103
v_cvt_pk_f16_f32 v30, v104, v105
v_cvt_pk_f16_f32 v31, v106, v107
v_cvt_pk_f16_f32 v32, v108, v109
v_cvt_pk_f16_f32 v33, v110, v111
v_cvt_pk_f16_f32 v34, v112, v113
v_cvt_pk_f16_f32 v35, v114, v115
v_cvt_pk_f16_f32 v36, v116, v117
v_cvt_pk_f16_f32 v37, v118, v119
v_cvt_pk_f16_f32 v38, v120, v121
v_cvt_pk_f16_f32 v39, v122, v123
v_cvt_pk_f16_f32 v40, v124, v125
v_cvt_pk_f16_f32 v41, v126, v127
v_cvt_pk_f16_f32 v42, v150, v151
v_cvt_pk_f16_f32 v43, v152, v153
v_cvt_pk_f16_f32 v44, v154, v155
v_cvt_pk_f16_f32 v45, v156, v157
v_cvt_pk_f16_f32 v46, v158, v159
v_cvt_pk_f16_f32 v47, v160, v161
v_cvt_pk_f16_f32 v48, v162, v163
v_cvt_pk_f16_f32 v49, v164, v165
v_cvt_pk_f16_f32 v50, v166, v167
v_cvt_pk_f16_f32 v51, v168, v169
v_cvt_pk_f16_f32 v52, v170, v171
v_cvt_pk_f16_f32 v53, v172, v173
v_cvt_pk_f16_f32 v54, v174, v175
v_cvt_pk_f16_f32 v55, v176, v177
v_cvt_pk_f16_f32 v56, v178, v179
v_cvt_pk_f16_f32 v57, v180, v181
v_cvt_pk_f16_f32 v58, v182, v183
v_cvt_pk_f16_f32 v59, v184, v185
v_cvt_pk_f16_f32 v60, v186, v187
v_cvt_pk_f16_f32 v61, v188, v189
v_cvt_pk_f16_f32 v62, v190, v191
v_cvt_pk_f16_f32 v63, v192, v193
v_permlane16_swap_b32_e32 v0, v2
v_permlane16_swap_b32_e32 v1, v3
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
v_permlane16_swap_b32_e32 v48, v50
v_permlane16_swap_b32_e32 v49, v51
v_permlane16_swap_b32_e32 v56, v58
v_permlane16_swap_b32_e32 v57, v59
v_permlane16_swap_b32_e32 v52, v54
v_permlane16_swap_b32_e32 v53, v55
v_permlane16_swap_b32_e32 v60, v62
v_permlane16_swap_b32_e32 v61, v63
s_and_b32 s1, s1, 0xffff
s_mov_b32 s2, s6
s_mov_b32 s3, s7
buffer_store_dwordx4 v[0:3], v128, s[0:3], 0, offen
buffer_store_dwordx4 v[8:11], v129, s[0:3], 0, offen
buffer_store_dwordx4 v[4:7], v194, s[0:3], 0, offen
buffer_store_dwordx4 v[12:15], v195, s[0:3], 0, offen
buffer_store_dwordx4 v[16:19], v196, s[0:3], 0, offen
buffer_store_dwordx4 v[24:27], v197, s[0:3], 0, offen
buffer_store_dwordx4 v[20:23], v198, s[0:3], 0, offen
buffer_store_dwordx4 v[28:31], v199, s[0:3], 0, offen
buffer_store_dwordx4 v[32:35], v200, s[0:3], 0, offen
buffer_store_dwordx4 v[40:43], v201, s[0:3], 0, offen
buffer_store_dwordx4 v[36:39], v202, s[0:3], 0, offen
buffer_store_dwordx4 v[44:47], v134, s[0:3], 0, offen
buffer_store_dwordx4 v[48:51], v135, s[0:3], 0, offen
buffer_store_dwordx4 v[56:59], v136, s[0:3], 0, offen
buffer_store_dwordx4 v[52:55], v137, s[0:3], 0, offen
buffer_store_dwordx4 v[60:63], v138, s[0:3], 0, offen
s_endpgm
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v9
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
	.size	v9, .Lfunc_end0-v9
	.cfi_endproc
                                        ; -- End function
	.set v9.num_vgpr, 256
	.set v9.num_agpr, 256
	.set v9.numbered_sgpr, 56
	.set v9.num_named_barrier, 0
	.set v9.private_seg_size, 0
	.set v9.uses_vcc, 1
	.set v9.uses_flat_scratch, 0
	.set v9.has_dyn_sized_stack, 0
	.set v9.has_recursion, 0
	.set v9.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11432
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
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
	.asciz	"v9"                            ; string offset=86
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
    .name:           v9
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         v9.kd
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
