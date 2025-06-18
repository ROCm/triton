	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	attn_fwd                        ; -- Begin function attn_fwd
	.p2align	8
	.type	attn_fwd,@function
attn_fwd:                               ; @attn_fwd
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.7:
	.file	1 "/app/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.8:
.LBB0_0:
	s_load_dwordx8 s[20:27], s[0:1], 0x38
	s_mul_i32 s0, s12, s18
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s28, s16, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s13, s17
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s14, s28
	v_mov_b32_e32 v73, v0
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s12, s14, 2
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v69, 3, v73
	v_lshrrev_b32_e32 v3, 4, v73
	s_add_u32 s0, s2, s0
	v_and_b32_e32 v2, 0x78, v69
	s_addc_u32 s1, s3, s1
	v_mad_u64_u32 v[0:1], s[2:3], s14, v3, v[2:3]
	s_mul_i32 s2, s15, s18
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s4, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s20, s17
	s_addc_u32 s5, s5, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s13, s4, s2
	s_mul_i32 s2, s22, s18
	s_addc_u32 s15, s5, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s4, s21, 2
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s5, s6, s2
	s_mul_i32 s2, s23, s17
	s_addc_u32 s6, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s16, s5, s2
	s_addc_u32 s19, s6, s3
	s_and_b32 s2, s14, 0x3fff
	v_or_b32_e32 v4, s28, v3
	s_movk_i32 s5, 0x4000
	s_bitset1_b32 s2, 14
	v_add_u32_e32 v1, s12, v0
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_lshlrev_b32_e32 v0, 1, v0
	v_bfrev_b32_e32 v57, 1
	v_cmp_gt_i32_e32 vcc, s5, v4
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v0, v57, v0, vcc
	v_or_b32_e32 v8, 4, v4
	v_or_b32_e32 v12, 8, v4
	v_or_b32_e32 v16, 12, v4
	v_or_b32_e32 v20, 16, v4
	v_or_b32_e32 v24, 20, v4
	v_or_b32_e32 v28, 24, v4
	v_or_b32_e32 v32, 28, v4
	v_or_b32_e32 v36, 32, v4
	v_or_b32_e32 v40, 36, v4
	v_or_b32_e32 v44, 40, v4
	v_or_b32_e32 v45, 44, v4
	v_or_b32_e32 v52, 48, v4
	v_or_b32_e32 v53, 52, v4
	v_or_b32_e32 v54, 56, v4
	buffer_load_dwordx4 v[4:7], v0, s[0:3], 0 offen
	v_add_u32_e32 v13, s12, v1
	v_lshlrev_b32_e32 v1, 1, v1
	v_cmp_gt_i32_e32 vcc, s5, v8
	v_add_u32_e32 v17, s12, v13
	v_lshlrev_b32_e32 v13, 1, v13
	v_cndmask_b32_e32 v1, v57, v1, vcc
	buffer_load_dwordx4 v[8:11], v1, s[0:3], 0 offen
	v_cmp_gt_i32_e32 vcc, s5, v12
	v_add_u32_e32 v21, s12, v17
	v_lshlrev_b32_e32 v17, 1, v17
	v_cndmask_b32_e32 v12, v57, v13, vcc
	v_cmp_gt_i32_e32 vcc, s5, v16
	v_add_u32_e32 v0, s12, v21
	v_lshlrev_b32_e32 v21, 1, v21
	v_cndmask_b32_e32 v16, v57, v17, vcc
	v_cmp_gt_i32_e32 vcc, s5, v20
	v_add_u32_e32 v29, s12, v0
	v_lshlrev_b32_e32 v0, 1, v0
	v_cndmask_b32_e32 v20, v57, v21, vcc
	v_cmp_gt_i32_e32 vcc, s5, v24
	v_add_u32_e32 v33, s12, v29
	v_lshlrev_b32_e32 v29, 1, v29
	v_cndmask_b32_e32 v0, v57, v0, vcc
	v_cmp_gt_i32_e32 vcc, s5, v28
	v_add_u32_e32 v37, s12, v33
	buffer_load_dwordx4 v[12:15], v12, s[0:3], 0 offen
	v_cndmask_b32_e32 v28, v57, v29, vcc
	buffer_load_dwordx4 v[16:19], v16, s[0:3], 0 offen
	v_lshlrev_b32_e32 v33, 1, v33
	v_cmp_gt_i32_e32 vcc, s5, v32
	v_add_u32_e32 v1, s12, v37
	buffer_load_dwordx4 v[20:23], v20, s[0:3], 0 offen
	v_cndmask_b32_e32 v32, v57, v33, vcc
	buffer_load_dwordx4 v[28:31], v28, s[0:3], 0 offen
	v_lshlrev_b32_e32 v37, 1, v37
	v_cmp_gt_i32_e32 vcc, s5, v36
	v_add_u32_e32 v46, s12, v1
	v_lshlrev_b32_e32 v1, 1, v1
	v_cndmask_b32_e32 v36, v57, v37, vcc
	v_cmp_gt_i32_e32 vcc, s5, v40
	buffer_load_dwordx4 v[24:27], v0, s[0:3], 0 offen
	v_add_u32_e32 v47, s12, v46
	v_cndmask_b32_e32 v1, v57, v1, vcc
	buffer_load_dwordx4 v[40:43], v1, s[0:3], 0 offen
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_load_dwordx4 v[32:35], v32, s[0:3], 0 offen
	v_cmp_gt_i32_e32 vcc, s5, v44
	buffer_load_dwordx4 v[36:39], v36, s[0:3], 0 offen
	v_add_u32_e32 v0, s12, v47
	v_cndmask_b32_e32 v59, v57, v46, vcc
	v_lshlrev_b32_e32 v44, 1, v47
	v_cmp_gt_i32_e32 vcc, s5, v45
	v_add_u32_e32 v48, s12, v0
	v_lshlrev_b32_e32 v0, 1, v0
	v_cndmask_b32_e32 v60, v57, v44, vcc
	v_cmp_gt_i32_e32 vcc, s5, v52
	v_or_b32_e32 v68, 60, v3
	v_lshlrev_b32_e32 v58, 1, v48
	v_cndmask_b32_e32 v0, v57, v0, vcc
	v_cmp_gt_i32_e32 vcc, s5, v53
	v_or_b32_e32 v55, s28, v68
	v_mul_lo_u32 v56, s14, v68
	v_add_lshl_u32 v1, v48, s12, 1
	v_cndmask_b32_e32 v70, v57, v58, vcc
	v_cmp_gt_i32_e32 vcc, s5, v54
	v_add_lshl_u32 v52, v56, v2, 1
	buffer_load_dwordx4 v[44:47], v59, s[0:3], 0 offen
	buffer_load_dwordx4 v[48:51], v60, s[0:3], 0 offen
	v_cndmask_b32_e32 v1, v57, v1, vcc
	v_cmp_gt_i32_e32 vcc, s5, v55
	v_and_b32_e32 v72, 32, v73
	s_movk_i32 s5, 0x78
	v_cndmask_b32_e32 v71, v57, v52, vcc
	buffer_load_dwordx4 v[52:55], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[56:59], v70, s[0:3], 0 offen
	buffer_load_dwordx4 v[60:63], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[64:67], v71, s[0:3], 0 offen
	v_lshrrev_b32_e32 v0, 1, v73
	v_bitop3_b32 v1, v0, v2, 24 bitop3:0x6c
	v_lshlrev_b32_e32 v1, 1, v1
	v_lshlrev_b32_e32 v70, 8, v3
	v_add3_u32 v202, 0, v1, v70
	v_and_b32_e32 v1, 24, v69
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt vmcnt(15)
	ds_write_b128 v202, v[4:7]
	v_and_b32_e32 v6, 64, v69
	v_bitop3_b32 v4, v1, v69, 32 bitop3:0x72
	v_or_b32_e32 v4, v4, v6
	v_bitop3_b32 v4, v0, v4, 24 bitop3:0x6c
	v_lshlrev_b32_e32 v4, 1, v4
	v_add3_u32 v203, 0, v4, v70
	v_and_b32_e32 v0, 24, v0
	v_and_or_b32 v4, v69, 56, 64
	v_bitop3_b32 v4, v0, v4, v6 bitop3:0x36
	v_or_b32_e32 v0, v0, v6
	v_or_b32_e32 v1, 0x60, v1
	v_and_b32_e32 v7, 32, v69
	v_bitop3_b32 v0, v0, v1, v7 bitop3:0x36
	v_lshlrev_b32_e32 v4, 1, v4
	v_lshlrev_b32_e32 v0, 1, v0
	v_add3_u32 v204, 0, v4, v70
	v_add3_u32 v205, 0, v0, v70
	s_waitcnt vmcnt(14)
	ds_write_b128 v203, v[8:11] offset:1024
	v_mul_lo_u32 v9, s21, v68
	v_mad_u64_u32 v[0:1], s[0:1], s21, v3, v[2:3]
	v_mad_u64_u32 v[4:5], s[6:7], s24, v3, v[2:3]
	s_waitcnt vmcnt(13)
	ds_write_b128 v204, v[12:15] offset:2048
	s_waitcnt vmcnt(12)
	ds_write_b128 v205, v[16:19] offset:3072
	s_waitcnt vmcnt(11)
	ds_write_b128 v202, v[20:23] offset:4096
	v_mul_lo_u32 v22, s24, v68
	v_add_lshl_u32 v9, v9, v2, 1
	v_add_lshl_u32 v2, v22, v2, 1
	v_add_u32_e32 v1, s4, v0
	v_accvgpr_write_b32 a197, v2
	v_lshlrev_b32_e32 v2, 2, v73
	v_add_u32_e32 v10, s4, v1
	v_accvgpr_write_b32 a195, v2
	v_and_b32_e32 v2, 12, v2
	v_lshlrev_b32_e32 v1, 1, v1
	v_add_u32_e32 v11, s4, v10
	v_accvgpr_write_b32 a196, v9
	s_waitcnt vmcnt(8)
	ds_write_b128 v203, v[40:43] offset:9216
	v_or_b32_e32 v7, v2, v7
	v_lshrrev_b32_e32 v9, 2, v73
	v_lshrrev_b32_e32 v22, 3, v72
	v_bitop3_b32 v42, v2, v69, 32 bitop3:0x72
	v_accvgpr_write_b32 a208, v1
	v_lshlrev_b32_e32 v1, 1, v10
	v_and_b32_e32 v8, 16, v73
	v_add_u32_e32 v12, s4, v11
	s_movk_i32 s1, 0x60
	v_and_or_b32 v9, v9, 3, v22
	v_lshlrev_b32_e32 v22, 1, v7
	v_lshlrev_b32_e32 v6, 1, v6
	v_lshlrev_b32_e32 v42, 1, v42
	v_accvgpr_write_b32 a209, v1
	v_lshlrev_b32_e32 v1, 1, v11
	v_and_b32_e32 v71, 31, v73
	v_add_u32_e32 v13, s4, v12
	s_waitcnt vmcnt(7)
	ds_write_b128 v205, v[32:35] offset:7168
	v_lshrrev_b32_e32 v32, 5, v73
	v_and_b32_e32 v33, 15, v73
	v_add3_u32 v22, 0, v22, v6
	v_add3_u32 v6, 0, v42, v6
	v_lshlrev_b32_e32 v8, 1, v8
	v_lshlrev_b32_e32 v9, 8, v9
	v_bitop3_b32 v2, v69, v2, s1 bitop3:0x4e
	v_accvgpr_write_b32 a210, v1
	v_lshlrev_b32_e32 v1, 1, v12
	v_add_u32_e32 v14, s4, v13
	s_waitcnt vmcnt(6)
	ds_write_b128 v202, v[36:39] offset:8192
	v_bitop3_b32 v34, v32, v73, 15 bitop3:0x78
	v_bitop3_b32 v35, v32, v33, 2 bitop3:0x36
	v_bitop3_b32 v36, v32, v33, 4 bitop3:0x36
	v_bitop3_b32 v37, v32, v33, 6 bitop3:0x36
	v_bitop3_b32 v38, v32, v33, 8 bitop3:0x36
	v_bitop3_b32 v39, v32, v33, 10 bitop3:0x36
	v_bitop3_b32 v40, v32, v33, 12 bitop3:0x36
	v_bitop3_b32 v32, v32, v33, 14 bitop3:0x36
	v_lshlrev_b32_e32 v33, 8, v71
	v_add3_u32 v213, v6, v8, v9
	v_bitop3_b32 v6, v7, v69, 64 bitop3:0x72
	v_lshl_add_u32 v2, v2, 1, 0
	v_accvgpr_write_b32 a211, v1
	v_lshlrev_b32_e32 v1, 1, v13
	v_add_u32_e32 v15, s4, v14
	v_add_u32_e32 v41, 0, v33
	v_lshl_add_u32 v6, v6, 1, 0
	v_add3_u32 v215, v2, v8, v9
	v_lshlrev_b32_e32 v2, 4, v39
	v_accvgpr_write_b32 a212, v1
	v_lshlrev_b32_e32 v1, 1, v14
	v_add_u32_e32 v16, s4, v15
	v_add3_u32 v214, v6, v8, v9
	v_add_u32_e32 v217, v41, v2
	v_lshlrev_b32_e32 v6, 4, v40
	v_add3_u32 v2, 0, v2, v33
	v_accvgpr_write_b32 a213, v1
	v_lshlrev_b32_e32 v1, 1, v15
	v_add_u32_e32 v17, s4, v16
	v_accvgpr_write_b32 a204, v2
	v_add3_u32 v2, 0, v6, v33
	v_accvgpr_write_b32 a214, v1
	v_lshlrev_b32_e32 v1, 1, v16
	v_add_u32_e32 v18, s4, v17
	v_lshlrev_b32_e32 v34, 4, v34
	v_accvgpr_write_b32 a205, v2
	v_lshlrev_b32_e32 v2, 1, v4
	v_accvgpr_write_b32 a215, v1
	v_lshlrev_b32_e32 v1, 1, v17
	v_add_u32_e32 v19, s4, v18
	v_lshlrev_b32_e32 v35, 4, v35
	v_add3_u32 v7, 0, v34, v33
	v_accvgpr_write_b32 a207, v2
	v_lshlrev_b32_e32 v2, 1, v73
	v_accvgpr_write_b32 a216, v1
	v_lshlrev_b32_e32 v1, 1, v18
	v_add_u32_e32 v20, s4, v19
	v_lshlrev_b32_e32 v36, 4, v36
	v_accvgpr_write_b32 a199, v7
	v_add3_u32 v7, 0, v35, v33
	v_and_b32_e32 v2, 0x60, v2
	v_accvgpr_write_b32 a217, v1
	v_lshlrev_b32_e32 v1, 1, v19
	v_lshlrev_b32_e32 v37, 4, v37
	v_lshlrev_b32_e32 v0, 1, v0
	v_accvgpr_write_b32 a200, v7
	v_add3_u32 v7, 0, v36, v33
	v_bitop3_b32 v2, v69, v2, s5 bitop3:0x6c
	v_accvgpr_write_b32 a218, v1
	v_lshlrev_b32_e32 v1, 1, v20
	v_add_u32_e32 v206, v41, v34
	v_lshlrev_b32_e32 v38, 4, v38
	v_accvgpr_write_b32 a198, v0
	v_lshlrev_b32_e32 v0, 4, v32
	v_accvgpr_write_b32 a201, v7
	v_add3_u32 v7, 0, v37, v33
	v_accvgpr_write_b32 a219, v1
	v_lshlrev_b32_e32 v1, 1, v2
	ds_write_b128 v203, v[24:27] offset:5120
	ds_write_b128 v204, v[28:31] offset:6144
	s_waitcnt vmcnt(5)
	ds_write_b128 v204, v[44:47] offset:10240
	s_waitcnt vmcnt(4)
	ds_write_b128 v205, v[48:51] offset:11264
	s_waitcnt vmcnt(3)
	ds_write_b128 v202, v[52:55] offset:12288
	s_waitcnt vmcnt(2)
	ds_write_b128 v203, v[56:59] offset:13312
	s_waitcnt vmcnt(1)
	ds_write_b128 v204, v[60:63] offset:14336
	s_waitcnt vmcnt(0)
	ds_write_b128 v205, v[64:67] offset:15360
	v_accvgpr_write_b32 a192, v71
	v_add_u32_e32 v207, v41, v35
	v_add_u32_e32 v208, v41, v36
	v_accvgpr_write_b32 a193, v72
	v_add_u32_e32 v211, v41, v37
	v_add_u32_e32 v216, v41, v38
	v_add_u32_e32 v219, v41, v6
	v_add_u32_e32 v220, v41, v0
	v_accvgpr_write_b32 a202, v7
	v_add3_u32 v7, 0, v38, v33
	v_accvgpr_write_b32 a194, v73
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	v_add3_u32 v242, 0, v1, v70
	ds_read_b128 v[34:37], v206
	ds_read_b128 v[38:41], v206 offset:8192
	ds_read_b128 v[42:45], v207
	ds_read_b128 v[46:49], v207 offset:8192
	ds_read_b128 v[50:53], v208
	ds_read_b128 v[54:57], v208 offset:8192
	ds_read_b128 v[58:61], v211
	ds_read_b128 v[62:65], v211 offset:8192
	ds_read_b128 v[66:69], v216
	ds_read_b128 v[70:73], v216 offset:8192
	ds_read_b128 v[74:77], v217
	ds_read_b128 v[78:81], v217 offset:8192
	ds_read_b128 v[82:85], v219
	ds_read_b128 v[86:89], v219 offset:8192
	ds_read_b128 v[90:93], v220
	ds_read_b128 v[94:97], v220 offset:8192
	s_lshl_b32 s0, s24, 2
	v_add_u32_e32 v3, s0, v4
	v_add_u32_e32 v5, s0, v3
	v_add_u32_e32 v23, s0, v5
	v_add_u32_e32 v21, s4, v20
	v_add_u32_e32 v24, s0, v23
	v_add_u32_e32 v25, s0, v24
	v_lshlrev_b32_e32 v1, 1, v21
	v_add_u32_e32 v26, s0, v25
	v_accvgpr_write_b32 a220, v1
	v_add_lshl_u32 v1, v21, s4, 1
	v_add_u32_e32 v27, s0, v26
	v_accvgpr_write_b32 a221, v1
	v_lshlrev_b32_e32 v1, 1, v3
	v_add_u32_e32 v28, s0, v27
	v_accvgpr_write_b32 a222, v1
	v_lshlrev_b32_e32 v1, 1, v5
	v_add_u32_e32 v29, s0, v28
	v_accvgpr_write_b32 a223, v1
	v_lshlrev_b32_e32 v1, 1, v23
	v_add_u32_e32 v30, s0, v29
	v_accvgpr_write_b32 a224, v1
	v_lshlrev_b32_e32 v1, 1, v24
	v_add_u32_e32 v31, s0, v30
	v_add3_u32 v0, 0, v0, v33
	v_accvgpr_write_b32 a225, v1
	v_lshlrev_b32_e32 v1, 1, v25
	v_accvgpr_write_b32 a206, v0
	v_add_u32_e32 v0, s0, v31
	s_lshl_b32 s6, s21, 6
	s_lshl_b32 s20, s24, 6
	s_and_b32 s1, s21, 0x3fff
	s_and_b32 s4, s24, 0x3fff
	v_accvgpr_write_b32 a226, v1
	v_lshlrev_b32_e32 v1, 1, v26
	v_add_u32_e32 v4, s0, v0
	s_bitset1_b32 s1, 14
	s_or_b32 s22, s4, 0x4000
	v_accvgpr_write_b32 a227, v1
	v_lshlrev_b32_e32 v1, 1, v27
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s21, s20, 31
	v_mov_b32_e32 v102, 1.0
	v_add3_u32 v212, v22, v8, v9
	v_accvgpr_write_b32 a203, v7
	v_accvgpr_write_b32 a228, v1
	v_lshlrev_b32_e32 v252, 1, v28
	v_lshlrev_b32_e32 v253, 1, v29
	v_lshlrev_b32_e32 v254, 1, v30
	v_lshlrev_b32_e32 v255, 1, v31
	v_lshlrev_b32_e32 v201, 1, v0
	v_lshlrev_b32_e32 v1, 1, v4
	v_add_lshl_u32 v200, v4, s0, 1
	s_lshl_b64 s[4:5], s[20:21], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	s_movk_i32 s14, 0xffc0
	s_lshl_b32 s20, s1, 16
	s_mov_b32 s12, 0x3e0293ee
	s_lshl_b32 s21, s22, 16
	v_mov_b32_e32 v103, v102
	v_mov_b32_e32 v105, 0xff800000
	v_mov_b32_e32 v107, 0xff800000
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s1, s15, 0xffff
	s_mov_b32 s0, s13
	s_or_b32 s1, s1, s20
	v_accvgpr_read_b32 v0, a198
	buffer_load_dwordx4 v[2:5], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a208
	buffer_load_dwordx4 v[6:9], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a209
	buffer_load_dwordx4 v[10:13], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a210
	buffer_load_dwordx4 v[14:17], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a211
	buffer_load_dwordx4 v[18:21], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a212
	buffer_load_dwordx4 v[22:25], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a213
	buffer_load_dwordx4 v[26:29], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a214
	buffer_load_dwordx4 v[30:33], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a215
	buffer_load_dwordx4 v[98:101], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a216
	buffer_load_dwordx4 v[108:111], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a217
	buffer_load_dwordx4 v[112:115], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a218
	buffer_load_dwordx4 v[116:119], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a219
	buffer_load_dwordx4 v[120:123], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a220
	buffer_load_dwordx4 v[124:127], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a221
	buffer_load_dwordx4 v[128:131], v0, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a196
	buffer_load_dwordx4 v[132:135], v0, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	v_accvgpr_read_b32 v0, a199
	v_mov_b32_e32 v180, v107
	v_accvgpr_read_b32 v107, a95
	v_accvgpr_read_b32 v106, a94
	v_accvgpr_read_b32 v137, a69
	v_accvgpr_read_b32 v136, a68
	v_accvgpr_read_b32 v139, a67
	v_accvgpr_read_b32 v138, a66
	v_accvgpr_read_b32 v141, a65
	v_accvgpr_read_b32 v140, a64
	s_and_b32 s0, s19, 0xffff
	s_or_b32 s1, s0, s21
	s_mov_b32 s0, s16
	v_accvgpr_read_b32 v143, a117
	v_accvgpr_read_b32 v142, a116
	v_accvgpr_read_b32 v145, a115
	v_accvgpr_read_b32 v144, a114
	v_accvgpr_read_b32 v147, a113
	v_accvgpr_read_b32 v146, a112
	v_accvgpr_read_b32 v149, a25
	v_accvgpr_read_b32 v148, a24
	v_accvgpr_read_b32 v151, a23
	v_accvgpr_read_b32 v150, a22
	v_accvgpr_read_b32 v153, a21
	v_accvgpr_read_b32 v152, a20
	v_accvgpr_read_b32 v155, a19
	v_accvgpr_read_b32 v154, a18
	v_accvgpr_read_b32 v157, a17
	v_accvgpr_read_b32 v156, a16
	v_accvgpr_read_b32 v239, a41
	v_accvgpr_read_b32 v238, a40
	v_accvgpr_read_b32 v241, a39
	v_accvgpr_read_b32 v240, a38
	v_accvgpr_read_b32 v251, a1
	v_accvgpr_read_b32 v250, a0
	v_accvgpr_read_b32 v245, a7
	v_accvgpr_read_b32 v244, a6
	v_accvgpr_read_b32 v247, a5
	v_accvgpr_read_b32 v246, a4
	v_accvgpr_read_b32 v249, a3
	v_accvgpr_read_b32 v248, a2
	s_add_u32 s16, s16, s4
	s_addc_u32 s19, s19, s5
	s_waitcnt vmcnt(15)
	ds_write_b128 v202, v[2:5]
	s_waitcnt vmcnt(14)
	ds_write_b128 v203, v[6:9] offset:1024
	s_waitcnt vmcnt(13)
	ds_write_b128 v204, v[10:13] offset:2048
	s_waitcnt vmcnt(12)
	ds_write_b128 v205, v[14:17] offset:3072
	s_waitcnt vmcnt(11)
	ds_write_b128 v202, v[18:21] offset:4096
	s_waitcnt vmcnt(10)
	ds_write_b128 v203, v[22:25] offset:5120
	s_waitcnt vmcnt(9)
	ds_write_b128 v204, v[26:29] offset:6144
	s_waitcnt vmcnt(8)
	ds_write_b128 v205, v[30:33] offset:7168
	s_waitcnt vmcnt(7)
	ds_write_b128 v202, v[98:101] offset:8192
	s_waitcnt vmcnt(6)
	ds_write_b128 v203, v[108:111] offset:9216
	s_waitcnt vmcnt(5)
	ds_write_b128 v204, v[112:115] offset:10240
	s_waitcnt vmcnt(4)
	ds_write_b128 v205, v[116:119] offset:11264
	s_waitcnt vmcnt(3)
	ds_write_b128 v202, v[120:123] offset:12288
	s_waitcnt vmcnt(2)
	ds_write_b128 v203, v[124:127] offset:13312
	s_waitcnt vmcnt(1)
	ds_write_b128 v204, v[128:131] offset:14336
	s_waitcnt vmcnt(0)
	ds_write_b128 v205, v[132:135] offset:15360
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[2:5], v206
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[176:191], v[2:5], v[34:37], 0
	ds_read_b128 v[6:9], v0 offset:8192
	v_accvgpr_read_b32 v17, a111
	v_accvgpr_read_b32 v16, a110
	v_accvgpr_read_b32 v15, a109
	v_accvgpr_read_b32 v14, a108
	v_accvgpr_read_b32 v13, a107
	v_accvgpr_read_b32 v12, a106
	v_mfma_f32_32x32x16_f16 a[144:159], v[2:5], v[38:41], 0
	ds_read_b128 v[2:5], v207
	v_accvgpr_read_b32 v11, a105
	v_accvgpr_read_b32 v10, a104
	v_accvgpr_read_b32 v113, a93
	v_accvgpr_read_b32 v112, a92
	v_accvgpr_read_b32 v115, a91
	v_accvgpr_read_b32 v114, a90
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[176:191], v[2:5], v[42:45], a[176:191]
	v_accvgpr_read_b32 v117, a89
	v_accvgpr_read_b32 v116, a88
	v_accvgpr_read_b32 v119, a87
	v_accvgpr_read_b32 v118, a86
	v_accvgpr_read_b32 v121, a85
	v_accvgpr_read_b32 v120, a84
	v_accvgpr_read_b32 v133, a83
	v_mfma_f32_32x32x16_f16 a[144:159], v[2:5], v[46:49], a[144:159]
	ds_read_b128 v[2:5], v208
	v_accvgpr_read_b32 v132, a82
	v_accvgpr_read_b32 v135, a81
	v_accvgpr_read_b32 v134, a80
	v_accvgpr_read_b32 v123, a79
	v_accvgpr_read_b32 v122, a78
	v_accvgpr_read_b32 v125, a77
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[176:191], v[2:5], v[50:53], a[176:191]
	v_accvgpr_read_b32 v124, a76
	v_accvgpr_read_b32 v127, a75
	v_accvgpr_read_b32 v126, a74
	v_accvgpr_read_b32 v129, a73
	v_accvgpr_read_b32 v128, a72
	v_accvgpr_read_b32 v131, a71
	v_accvgpr_read_b32 v130, a70
	v_mfma_f32_32x32x16_f16 a[144:159], v[2:5], v[54:57], a[144:159]
	ds_read_b128 v[2:5], v211
	v_accvgpr_read_b32 v31, a125
	v_accvgpr_read_b32 v30, a124
	v_accvgpr_read_b32 v33, a123
	v_accvgpr_read_b32 v32, a122
	v_accvgpr_read_b32 v99, a121
	v_accvgpr_read_b32 v98, a120
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[176:191], v[2:5], v[58:61], a[176:191]
	v_accvgpr_read_b32 v101, a119
	v_accvgpr_read_b32 v100, a118
	v_accvgpr_read_b32 v29, a27
	v_accvgpr_read_b32 v28, a26
	v_mfma_f32_32x32x16_f16 a[144:159], v[2:5], v[62:65], a[144:159]
	v_accvgpr_read_b32 v2, a203
	ds_read_b128 v[22:25], v2 offset:8192
	v_accvgpr_read_b32 v0, a200
	v_accvgpr_read_b32 v5, a99
	v_accvgpr_read_b32 v4, a98
	v_accvgpr_read_b32 v3, a97
	v_accvgpr_read_b32 v2, a96
	v_mfma_f32_32x32x16_f16 a[160:175], v[6:9], v[34:37], 0
	v_mfma_f32_32x32x16_f16 a[128:143], v[6:9], v[38:41], 0
	ds_read_b128 v[6:9], v0 offset:8192
	v_accvgpr_read_b32 v0, a201
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[160:175], v[6:9], v[42:45], a[160:175]
	v_mfma_f32_32x32x16_f16 a[128:143], v[6:9], v[46:49], a[128:143]
	ds_read_b128 v[6:9], v0 offset:8192
	v_mov_b32_e32 v0, v105
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[160:175], v[6:9], v[50:53], a[160:175]
	v_mfma_f32_32x32x16_f16 a[128:143], v[6:9], v[54:57], a[128:143]
	v_accvgpr_read_b32 v6, a202
	ds_read_b128 v[18:21], v6 offset:8192
	v_accvgpr_read_b32 v9, a103
	v_accvgpr_read_b32 v8, a102
	v_accvgpr_read_b32 v7, a101
	v_accvgpr_read_b32 v6, a100
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[160:175], v[18:21], v[58:61], a[160:175]
	v_mfma_f32_32x32x16_f16 a[128:143], v[18:21], v[62:65], a[128:143]
	ds_read_b128 v[18:21], v216
	v_mfma_f32_32x32x16_f16 a[160:175], v[22:25], v[66:69], a[160:175]
	v_mfma_f32_32x32x16_f16 a[128:143], v[22:25], v[70:73], a[128:143]
	v_accvgpr_read_b32 v22, a204
	ds_read_b128 v[22:25], v22 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[176:191], v[18:21], v[66:69], a[176:191]
	v_mfma_f32_32x32x16_f16 a[144:159], v[18:21], v[70:73], a[144:159]
	ds_read_b128 v[18:21], v217
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[160:175], v[22:25], v[74:77], a[160:175]
	v_mfma_f32_32x32x16_f16 a[128:143], v[22:25], v[78:81], a[128:143]
	v_accvgpr_read_b32 v22, a205
	ds_read_b128 v[24:27], v22 offset:8192
	v_accvgpr_read_b32 v23, a127
	v_accvgpr_read_b32 v22, a126
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[176:191], v[18:21], v[74:77], a[176:191]
	v_mfma_f32_32x32x16_f16 a[144:159], v[18:21], v[78:81], a[144:159]
	ds_read_b128 v[18:21], v219
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[160:175], v[24:27], v[82:85], a[160:175]
	v_mfma_f32_32x32x16_f16 a[128:143], v[24:27], v[86:89], a[128:143]
	v_accvgpr_read_b32 v24, a206
	ds_read_b128 v[108:111], v24 offset:8192
	v_accvgpr_read_b32 v25, a31
	v_accvgpr_read_b32 v24, a30
	v_accvgpr_read_b32 v27, a29
	v_accvgpr_read_b32 v26, a28
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[176:191], v[18:21], v[82:85], a[176:191]
	v_mfma_f32_32x32x16_f16 a[144:159], v[18:21], v[86:89], a[144:159]
	ds_read_b128 v[18:21], v220
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[176:191], v[18:21], v[90:93], a[176:191]
	v_mfma_f32_32x32x16_f16 a[160:175], v[108:111], v[90:93], a[160:175]
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v181, a176
	v_accvgpr_read_b32 v182, a177
	v_accvgpr_read_b32 v183, a178
	v_accvgpr_read_b32 v184, a179
	v_accvgpr_read_b32 v185, a180
	v_mfma_f32_32x32x16_f16 a[144:159], v[18:21], v[94:97], a[144:159]
	.file	2 "/app/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v18, v182, v182
	v_max_f32_e32 v19, v181, v181
	v_max_f32_e32 v18, v19, v18
	v_accvgpr_read_b32 v186, a181
	v_max3_f32 v18, v18, v183, v184
	v_accvgpr_read_b32 v187, a182
	v_accvgpr_read_b32 v188, a183
	v_max3_f32 v18, v18, v185, v186
	v_accvgpr_read_b32 v173, a184
	v_accvgpr_read_b32 v174, a185
	v_max3_f32 v18, v18, v187, v188
	v_accvgpr_read_b32 v175, a186
	v_accvgpr_read_b32 v176, a187
	v_max3_f32 v18, v18, v173, v174
	v_accvgpr_read_b32 v177, a188
	v_accvgpr_read_b32 v178, a189
	v_max3_f32 v18, v18, v175, v176
	v_accvgpr_read_b32 v179, a190
	v_accvgpr_read_b32 v172, a191
	v_max3_f32 v18, v18, v177, v178
	v_accvgpr_read_b32 v171, a160
	v_accvgpr_read_b32 v170, a161
	v_max3_f32 v18, v18, v179, v172
	v_accvgpr_read_b32 v163, a162
	v_accvgpr_read_b32 v164, a163
	v_max3_f32 v18, v18, v171, v170
	v_accvgpr_read_b32 v165, a164
	v_accvgpr_read_b32 v166, a165
	v_max3_f32 v18, v18, v163, v164
	v_accvgpr_read_b32 v167, a166
	v_accvgpr_read_b32 v168, a167
	v_max3_f32 v18, v18, v165, v166
	v_mfma_f32_32x32x16_f16 a[128:143], v[108:111], v[94:97], a[128:143]
	v_accvgpr_read_b32 v169, a168
	v_accvgpr_read_b32 v109, a169
	v_max3_f32 v18, v18, v167, v168
	v_accvgpr_read_b32 v158, a170
	v_accvgpr_read_b32 v159, a171
	v_max3_f32 v18, v18, v169, v109
	v_accvgpr_read_b32 v160, a172
	v_accvgpr_read_b32 v161, a173
	v_max3_f32 v18, v18, v158, v159
	v_accvgpr_read_b32 v162, a174
	v_accvgpr_read_b32 v104, a175
	v_max3_f32 v18, v18, v160, v161
	v_max3_f32 v18, v18, v162, v104
	v_mov_b32_e32 v19, v18
	s_nop 1
	v_permlane32_swap_b32_e32 v18, v19
	v_max3_f32 v105, v0, v18, v19
	v_pk_mul_f32 v[110:111], v[104:105], s[12:13] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v195, a144
	v_fma_f32 v0, v0, s12, -v111
	v_exp_f32_e32 v108, v0
	v_accvgpr_read_b32 v196, a145
	v_accvgpr_read_b32 v197, a146
	v_accvgpr_read_b32 v198, a147
	v_pk_mul_f32 v[16:17], v[16:17], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[108:109] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v0, a207
	v_accvgpr_write_b32 a111, v17
	v_accvgpr_write_b32 a110, v16
	v_accvgpr_write_b32 a109, v15
	v_accvgpr_write_b32 a108, v14
	v_accvgpr_write_b32 a107, v13
	v_accvgpr_write_b32 a106, v12
	v_accvgpr_write_b32 a105, v11
	v_accvgpr_write_b32 a104, v10
	v_accvgpr_write_b32 a103, v9
	v_accvgpr_write_b32 a102, v8
	v_accvgpr_write_b32 a101, v7
	v_accvgpr_write_b32 a100, v6
	v_accvgpr_write_b32 a99, v5
	v_accvgpr_write_b32 a98, v4
	v_accvgpr_write_b32 a97, v3
	v_accvgpr_write_b32 a96, v2
	v_pk_mul_f32 v[16:17], v[106:107], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[134:135], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[132:133], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[120:121], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[118:119], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[116:117], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[114:115], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[112:113], v[108:109] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v199, a148
	v_accvgpr_write_b32 a95, v17
	v_accvgpr_write_b32 a94, v16
	v_accvgpr_write_b32 a93, v15
	v_accvgpr_write_b32 a92, v14
	v_accvgpr_write_b32 a91, v13
	v_accvgpr_write_b32 a90, v12
	v_accvgpr_write_b32 a89, v11
	v_accvgpr_write_b32 a88, v10
	v_accvgpr_write_b32 a87, v9
	v_accvgpr_write_b32 a86, v8
	v_accvgpr_write_b32 a85, v7
	v_accvgpr_write_b32 a84, v6
	v_accvgpr_write_b32 a83, v5
	v_accvgpr_write_b32 a82, v4
	v_accvgpr_write_b32 a81, v3
	v_accvgpr_write_b32 a80, v2
	v_pk_mul_f32 v[16:17], v[122:123], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[140:141], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[138:139], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[136:137], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[130:131], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[128:129], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[126:127], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[124:125], v[108:109] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v209, a149
	v_accvgpr_write_b32 a79, v17
	v_accvgpr_write_b32 a78, v16
	v_accvgpr_write_b32 a77, v15
	v_accvgpr_write_b32 a76, v14
	v_accvgpr_write_b32 a75, v13
	v_accvgpr_write_b32 a74, v12
	v_accvgpr_write_b32 a73, v11
	v_accvgpr_write_b32 a72, v10
	v_accvgpr_write_b32 a71, v9
	v_accvgpr_write_b32 a70, v8
	v_accvgpr_write_b32 a69, v7
	v_accvgpr_write_b32 a68, v6
	v_accvgpr_write_b32 a67, v5
	v_accvgpr_write_b32 a66, v4
	v_accvgpr_write_b32 a65, v3
	v_accvgpr_write_b32 a64, v2
	v_max_f32_e32 v2, v196, v196
	v_max_f32_e32 v3, v195, v195
	v_max_f32_e32 v2, v3, v2
	v_max3_f32 v2, v2, v197, v198
	v_accvgpr_read_b32 v8, a222
	buffer_load_dwordx4 v[18:21], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[136:139], v8, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a150
	v_accvgpr_read_b32 v210, a151
	v_max3_f32 v2, v2, v199, v209
	v_accvgpr_read_b32 v134, a152
	v_accvgpr_read_b32 v135, a153
	v_max3_f32 v2, v2, v0, v210
	v_accvgpr_read_b32 v189, a154
	v_accvgpr_read_b32 v190, a155
	v_max3_f32 v2, v2, v134, v135
	v_accvgpr_read_b32 v191, a156
	v_accvgpr_read_b32 v192, a157
	v_max3_f32 v2, v2, v189, v190
	v_accvgpr_read_b32 v193, a158
	v_accvgpr_read_b32 v194, a159
	v_max3_f32 v2, v2, v191, v192
	v_accvgpr_read_b32 v132, a128
	v_accvgpr_read_b32 v133, a129
	v_max3_f32 v2, v2, v193, v194
	v_accvgpr_read_b32 v116, a130
	v_accvgpr_read_b32 v117, a131
	v_max3_f32 v2, v2, v132, v133
	v_accvgpr_read_b32 v118, a132
	v_accvgpr_read_b32 v119, a133
	v_max3_f32 v2, v2, v116, v117
	v_accvgpr_read_b32 v120, a134
	v_accvgpr_read_b32 v121, a135
	v_max3_f32 v2, v2, v118, v119
	v_accvgpr_read_b32 v129, a136
	v_accvgpr_read_b32 v128, a137
	v_max3_f32 v2, v2, v120, v121
	v_accvgpr_read_b32 v126, a138
	v_accvgpr_read_b32 v127, a139
	v_max3_f32 v2, v2, v129, v128
	v_accvgpr_read_b32 v124, a140
	v_accvgpr_read_b32 v125, a141
	v_max3_f32 v2, v2, v126, v127
	v_accvgpr_read_b32 v122, a142
	v_accvgpr_read_b32 v106, a143
	v_max3_f32 v2, v2, v124, v125
	v_max3_f32 v2, v2, v122, v106
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v2, v3
	v_max3_f32 v107, v180, v2, v3
	v_accvgpr_read_b32 v12, a223
	v_accvgpr_read_b32 v16, a224
	v_pk_mul_f32 v[112:113], v[106:107], s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[100:101], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[98:99], v[108:109] op_sel_hi:[1,0]
	buffer_load_dwordx4 v[98:101], v12, s[0:3], 0 offen
	v_pk_mul_f32 v[12:13], v[32:33], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[30:31], v[108:109] op_sel_hi:[1,0]
	buffer_load_dwordx4 v[30:33], v16, s[0:3], 0 offen
	v_pk_mul_f32 v[16:17], v[22:23], v[108:109] op_sel_hi:[1,0]
	v_fma_f32 v22, v180, s12, -v113
	v_exp_f32_e32 v104, v22
	v_pk_mul_f32 v[2:3], v[146:147], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[144:145], v[108:109] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[142:143], v[108:109] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v22, a228
	v_accvgpr_write_b32 a127, v17
	v_accvgpr_write_b32 a126, v16
	v_accvgpr_write_b32 a125, v15
	v_accvgpr_write_b32 a124, v14
	v_accvgpr_write_b32 a123, v13
	v_accvgpr_write_b32 a122, v12
	v_accvgpr_write_b32 a121, v11
	v_accvgpr_write_b32 a120, v10
	v_accvgpr_write_b32 a119, v9
	v_accvgpr_write_b32 a118, v8
	v_accvgpr_write_b32 a117, v7
	v_accvgpr_write_b32 a116, v6
	v_accvgpr_write_b32 a115, v5
	v_accvgpr_write_b32 a114, v4
	v_accvgpr_write_b32 a113, v3
	v_accvgpr_write_b32 a112, v2
	v_accvgpr_read_b32 v6, a225
	v_accvgpr_read_b32 v10, a226
	v_accvgpr_read_b32 v14, a227
	v_pk_mul_f32 v[16:17], v[24:25], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[156:157], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[154:155], v[104:105] op_sel_hi:[1,0]
	buffer_load_dwordx4 v[140:143], v6, s[0:3], 0 offen
	buffer_load_dwordx4 v[144:147], v10, s[0:3], 0 offen
	v_pk_mul_f32 v[6:7], v[152:153], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[150:151], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[148:149], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[28:29], v[104:105] op_sel_hi:[1,0]
	buffer_load_dwordx4 v[148:151], v14, s[0:3], 0 offen
	v_fma_f32 v0, v0, s12, -v113
	buffer_load_dwordx4 v[22:25], v22, s[0:3], 0 offen
	v_pk_mul_f32 v[14:15], v[26:27], v[104:105] op_sel_hi:[1,0]
	v_exp_f32_e32 v0, v0
	v_accvgpr_write_b32 a31, v17
	v_accvgpr_write_b32 a30, v16
	v_accvgpr_write_b32 a29, v15
	v_accvgpr_write_b32 a28, v14
	v_accvgpr_write_b32 a27, v13
	v_accvgpr_write_b32 a26, v12
	v_accvgpr_write_b32 a25, v11
	v_accvgpr_write_b32 a24, v10
	v_accvgpr_write_b32 a23, v9
	v_accvgpr_write_b32 a22, v8
	v_accvgpr_write_b32 a21, v7
	v_accvgpr_write_b32 a20, v6
	v_accvgpr_write_b32 a19, v5
	v_accvgpr_write_b32 a18, v4
	v_accvgpr_write_b32 a17, v3
	v_accvgpr_write_b32 a16, v2
	buffer_load_dwordx4 v[2:5], v252, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v253, s[0:3], 0 offen
	buffer_load_dwordx4 v[26:29], v254, s[0:3], 0 offen
	buffer_load_dwordx4 v[152:155], v255, s[0:3], 0 offen
	buffer_load_dwordx4 v[222:225], v201, s[0:3], 0 offen
	buffer_load_dwordx4 v[226:229], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[230:233], v200, s[0:3], 0 offen
	v_accvgpr_read_b32 v10, a197
	buffer_load_dwordx4 v[234:237], v10, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt vmcnt(15)
	ds_write_b128 v242, v[18:21]
	s_waitcnt vmcnt(14)
	ds_write_b128 v242, v[136:139] offset:1024
	s_waitcnt vmcnt(13)
	ds_write_b128 v242, v[98:101] offset:2048
	s_waitcnt vmcnt(12)
	ds_write_b128 v242, v[30:33] offset:3072
	s_waitcnt vmcnt(11)
	ds_write_b128 v242, v[140:143] offset:4096
	v_accvgpr_read_b32 v99, a37
	v_accvgpr_read_b32 v98, a36
	v_accvgpr_read_b32 v33, a47
	v_accvgpr_read_b32 v32, a46
	s_waitcnt vmcnt(10)
	ds_write_b128 v242, v[144:147] offset:5120
	v_accvgpr_read_b32 v31, a45
	v_accvgpr_read_b32 v30, a44
	v_accvgpr_read_b32 v157, a43
	v_accvgpr_read_b32 v156, a42
	s_waitcnt vmcnt(9)
	ds_write_b128 v242, v[148:151] offset:6144
	v_accvgpr_read_b32 v21, a35
	s_waitcnt vmcnt(8)
	ds_write_b128 v242, v[22:25] offset:7168
	v_fma_f32 v22, v185, s12, -v111
	v_fma_f32 v23, v186, s12, -v111
	v_exp_f32_e32 v131, v22
	v_exp_f32_e32 v130, v23
	v_pk_mul_f32 v[22:23], v[98:99], v[104:105] op_sel_hi:[1,0]
	v_fma_f32 v24, v187, s12, -v111
	v_fma_f32 v25, v188, s12, -v111
	v_fma_f32 v188, v195, s12, -v113
	v_fma_f32 v195, v196, s12, -v113
	v_fma_f32 v196, v197, s12, -v113
	v_fma_f32 v197, v198, s12, -v113
	v_fma_f32 v198, v199, s12, -v113
	s_waitcnt vmcnt(7)
	ds_write_b128 v242, v[2:5] offset:8192
	s_waitcnt vmcnt(6)
	ds_write_b128 v242, v[6:9] offset:9216
	s_waitcnt vmcnt(5)
	ds_write_b128 v242, v[26:29] offset:10240
	s_waitcnt vmcnt(4)
	ds_write_b128 v242, v[152:155] offset:11264
	s_waitcnt vmcnt(3)
	ds_write_b128 v242, v[222:225] offset:12288
	s_waitcnt vmcnt(2)
	ds_write_b128 v242, v[226:229] offset:13312
	s_waitcnt vmcnt(1)
	ds_write_b128 v242, v[230:233] offset:14336
	s_waitcnt vmcnt(0)
	ds_write_b128 v242, v[234:237] offset:15360
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	ds_read_b64_tr_b16 v[140:141], v212
	ds_read_b64_tr_b16 v[142:143], v212 offset:2048
	ds_read_b64_tr_b16 v[144:145], v213
	ds_read_b64_tr_b16 v[146:147], v213 offset:2048
	ds_read_b64_tr_b16 v[148:149], v214
	ds_read_b64_tr_b16 v[150:151], v214 offset:2048
	ds_read_b64_tr_b16 v[98:99], v215
	ds_read_b64_tr_b16 v[100:101], v215 offset:2048
	v_fma_f32 v2, v181, s12, -v111
	v_fma_f32 v3, v182, s12, -v111
	v_fma_f32 v4, v183, s12, -v111
	v_fma_f32 v5, v184, s12, -v111
	v_fma_f32 v199, v209, s12, -v113
	v_fma_f32 v209, v210, s12, -v113
	v_exp_f32_e32 v138, v2
	v_exp_f32_e32 v139, v3
	v_exp_f32_e32 v137, v4
	v_exp_f32_e32 v136, v5
	v_exp_f32_e32 v123, v24
	v_exp_f32_e32 v106, v25
	v_exp_f32_e32 v188, v188
	v_exp_f32_e32 v195, v195
	v_exp_f32_e32 v210, v196
	v_exp_f32_e32 v243, v197
	v_exp_f32_e32 v218, v198
	v_exp_f32_e32 v221, v199
	v_exp_f32_e32 v209, v209
	v_accvgpr_read_b32 v20, a34
	v_accvgpr_read_b32 v19, a33
	v_accvgpr_read_b32 v18, a32
	v_pk_mul_f32 v[32:33], v[32:33], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[240:241], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[238:239], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[156:157], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[104:105] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v115, a15
	v_accvgpr_write_b32 a47, v33
	v_accvgpr_read_b32 v114, a14
	v_cvt_pk_f16_f32 v2, v138, v139
	v_cvt_pk_f16_f32 v3, v137, v136
	v_cvt_pk_f16_f32 v4, v131, v130
	v_cvt_pk_f16_f32 v5, v123, v106
	v_accvgpr_read_b32 v157, a13
	v_accvgpr_read_b32 v156, a12
	v_accvgpr_read_b32 v239, a11
	v_accvgpr_read_b32 v238, a10
	v_accvgpr_read_b32 v241, a9
	v_accvgpr_read_b32 v240, a8
	v_accvgpr_write_b32 a46, v32
	v_accvgpr_write_b32 a45, v31
	v_accvgpr_write_b32 a44, v30
	v_accvgpr_write_b32 a43, v29
	v_accvgpr_write_b32 a42, v28
	v_accvgpr_write_b32 a41, v27
	v_accvgpr_write_b32 a40, v26
	v_accvgpr_write_b32 a39, v25
	v_accvgpr_write_b32 a38, v24
	v_accvgpr_write_b32 a37, v23
	v_accvgpr_write_b32 a36, v22
	v_accvgpr_write_b32 a35, v21
	v_accvgpr_write_b32 a34, v20
	v_accvgpr_write_b32 a33, v19
	v_accvgpr_write_b32 a32, v18
	v_pk_mul_f32 v[18:19], v[250:251], v[104:105] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v196, v188, v195
	v_cvt_pk_f16_f32 v197, v210, v243
	v_cvt_pk_f16_f32 v198, v218, v221
	v_cvt_pk_f16_f32 v199, v0, v209
	v_fma_f32 v22, v173, s12, -v111
	v_fma_f32 v23, v174, s12, -v111
	v_fma_f32 v24, v175, s12, -v111
	v_fma_f32 v25, v176, s12, -v111
	v_fma_f32 v26, v177, s12, -v111
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[96:111], v[140:143], v[2:5], a[96:111]
	v_pk_mul_f32 v[20:21], v[248:249], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[238:239], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[156:157], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[114:115], v[104:105] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v17, a63
	v_accvgpr_read_b32 v16, a62
	v_accvgpr_read_b32 v15, a61
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 a[80:95], v[144:147], v[2:5], a[80:95]
	v_accvgpr_read_b32 v14, a60
	v_accvgpr_read_b32 v13, a59
	v_accvgpr_read_b32 v12, a58
	v_accvgpr_read_b32 v11, a57
	v_accvgpr_read_b32 v10, a56
	v_accvgpr_read_b32 v9, a55
	v_accvgpr_read_b32 v8, a54
	v_mfma_f32_32x32x16_f16 a[16:31], v[140:143], v[196:199], a[16:31]
	v_exp_f32_e32 v143, v22
	v_fma_f32 v140, v178, s12, -v111
	v_fma_f32 v141, v179, s12, -v111
	v_fma_f32 v142, v172, s12, -v111
	v_accvgpr_read_b32 v7, a53
	v_accvgpr_read_b32 v6, a52
	v_accvgpr_read_b32 v153, a51
	v_mfma_f32_32x32x16_f16 a[32:47], v[144:147], v[196:199], a[32:47]
	v_exp_f32_e32 v144, v23
	v_exp_f32_e32 v145, v24
	v_exp_f32_e32 v146, v25
	v_exp_f32_e32 v147, v26
	v_pk_mul_f32 v[22:23], v[246:247], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[244:245], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[240:241], v[104:105] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[64:79], v[148:151], v[2:5], a[64:79]
	v_accvgpr_write_b32 a0, v18
	v_accvgpr_write_b32 a1, v19
	v_accvgpr_write_b32 a2, v20
	v_accvgpr_write_b32 a3, v21
	v_accvgpr_write_b32 a4, v22
	v_accvgpr_write_b32 a5, v23
	v_accvgpr_write_b32 a6, v24
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[112:127], v[98:101], v[2:5], a[112:127]
	v_accvgpr_write_b32 a7, v25
	v_accvgpr_write_b32 a8, v26
	v_accvgpr_write_b32 a9, v27
	v_accvgpr_write_b32 a10, v28
	v_accvgpr_write_b32 a11, v29
	v_accvgpr_write_b32 a12, v30
	v_accvgpr_write_b32 a13, v31
	v_accvgpr_write_b32 a14, v32
	v_accvgpr_write_b32 a15, v33
	v_exp_f32_e32 v22, v140
	v_exp_f32_e32 v23, v141
	v_mfma_f32_32x32x16_f16 a[0:15], v[98:101], v[196:199], a[0:15]
	v_exp_f32_e32 v24, v142
	v_fma_f32 v20, v134, s12, -v113
	v_fma_f32 v21, v135, s12, -v113
	v_fma_f32 v25, v189, s12, -v113
	v_fma_f32 v26, v190, s12, -v113
	v_fma_f32 v27, v191, s12, -v113
	v_fma_f32 v28, v192, s12, -v113
	v_fma_f32 v29, v193, s12, -v113
	v_fma_f32 v30, v194, s12, -v113
	v_accvgpr_read_b32 v152, a50
	v_accvgpr_read_b32 v155, a49
	v_accvgpr_read_b32 v154, a48
	v_pk_mul_f32 v[16:17], v[16:17], v[104:105] op_sel_hi:[1,0]
	v_exp_f32_e32 v31, v20
	v_exp_f32_e32 v32, v21
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_pk_mul_f32 v[2:3], v[154:155], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[152:153], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[104:105] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[104:105] op_sel_hi:[1,0]
	ds_read_b64_tr_b16 v[152:153], v212 offset:4096
	ds_read_b64_tr_b16 v[154:155], v212 offset:6144
	ds_read_b64_tr_b16 v[180:181], v213 offset:4096
	ds_read_b64_tr_b16 v[182:183], v213 offset:6144
	ds_read_b64_tr_b16 v[184:185], v214 offset:4096
	ds_read_b64_tr_b16 v[186:187], v214 offset:6144
	ds_read_b64_tr_b16 v[222:223], v215 offset:4096
	ds_read_b64_tr_b16 v[224:225], v215 offset:6144
	ds_read_b64_tr_b16 v[226:227], v212 offset:8192
	ds_read_b64_tr_b16 v[228:229], v212 offset:10240
	ds_read_b64_tr_b16 v[230:231], v212 offset:12288
	ds_read_b64_tr_b16 v[232:233], v212 offset:14336
	ds_read_b64_tr_b16 v[234:235], v213 offset:8192
	ds_read_b64_tr_b16 v[236:237], v213 offset:10240
	ds_read_b64_tr_b16 a[128:129], v213 offset:12288
	ds_read_b64_tr_b16 a[130:131], v213 offset:14336
	ds_read_b64_tr_b16 a[132:133], v214 offset:8192
	ds_read_b64_tr_b16 a[134:135], v214 offset:10240
	v_accvgpr_write_b32 a63, v17
	v_accvgpr_write_b32 a62, v16
	v_accvgpr_write_b32 a61, v15
	v_accvgpr_write_b32 a60, v14
	v_accvgpr_write_b32 a59, v13
	v_accvgpr_write_b32 a58, v12
	v_accvgpr_write_b32 a57, v11
	v_accvgpr_write_b32 a56, v10
	v_accvgpr_write_b32 a55, v9
	v_accvgpr_write_b32 a54, v8
	v_accvgpr_write_b32 a53, v7
	v_accvgpr_write_b32 a52, v6
	v_accvgpr_write_b32 a51, v5
	v_accvgpr_write_b32 a50, v4
	v_accvgpr_write_b32 a49, v3
	v_accvgpr_write_b32 a48, v2
	v_cvt_pk_f16_f32 v2, v143, v144
	v_cvt_pk_f16_f32 v3, v145, v146
	v_mfma_f32_32x32x16_f16 a[48:63], v[148:151], v[196:199], a[48:63]
	v_cvt_pk_f16_f32 v4, v147, v22
	v_cvt_pk_f16_f32 v5, v23, v24
	v_fma_f32 v18, v171, s12, -v111
	v_fma_f32 v19, v170, s12, -v111
	v_fma_f32 v20, v163, s12, -v111
	v_fma_f32 v21, v164, s12, -v111
	v_fma_f32 v33, v165, s12, -v111
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 a[96:111], v[152:155], v[2:5], a[96:111]
	v_fma_f32 v98, v166, s12, -v111
	v_fma_f32 v99, v167, s12, -v111
	v_fma_f32 v100, v168, s12, -v111
	v_fma_f32 v101, v169, s12, -v111
	v_fma_f32 v109, v109, s12, -v111
	v_fma_f32 v114, v158, s12, -v111
	v_fma_f32 v115, v159, s12, -v111
	v_mfma_f32_32x32x16_f16 a[80:95], v[180:183], v[2:5], a[80:95]
	v_fma_f32 v134, v160, s12, -v111
	v_fma_f32 v135, v161, s12, -v111
	v_fma_f32 v140, v162, s12, -v111
	v_sub_f32_e32 v110, v110, v111
	v_fma_f32 v111, v132, s12, -v113
	v_fma_f32 v132, v133, s12, -v113
	v_exp_f32_e32 v133, v18
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 a[64:79], v[184:187], v[2:5], a[64:79]
	v_exp_f32_e32 v141, v19
	v_exp_f32_e32 v142, v20
	v_exp_f32_e32 v148, v21
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v98, v98
	v_exp_f32_e32 v99, v99
	v_exp_f32_e32 v100, v100
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 a[112:127], v[222:225], v[2:5], a[112:127]
	v_cvt_pk_f16_f32 v2, v31, v32
	v_cvt_pk_f16_f32 v3, v25, v26
	v_cvt_pk_f16_f32 v4, v27, v28
	v_cvt_pk_f16_f32 v5, v29, v30
	v_fma_f32 v18, v116, s12, -v113
	v_fma_f32 v19, v117, s12, -v113
	v_fma_f32 v20, v118, s12, -v113
	v_mfma_f32_32x32x16_f16 a[0:15], v[222:225], v[2:5], a[0:15]
	v_fma_f32 v21, v119, s12, -v113
	v_fma_f32 v116, v120, s12, -v113
	v_fma_f32 v117, v121, s12, -v113
	v_fma_f32 v118, v129, s12, -v113
	v_fma_f32 v119, v128, s12, -v113
	v_fma_f32 v120, v126, s12, -v113
	v_fma_f32 v121, v127, s12, -v113
	v_exp_f32_e32 v111, v111
	v_exp_f32_e32 v126, v132
	v_exp_f32_e32 v127, v18
	v_exp_f32_e32 v128, v19
	v_exp_f32_e32 v129, v20
	v_exp_f32_e32 v132, v21
	v_exp_f32_e32 v116, v116
	v_exp_f32_e32 v117, v117
	ds_read_b64_tr_b16 v[6:7], v214 offset:12288
	ds_read_b64_tr_b16 v[8:9], v214 offset:14336
	ds_read_b64_tr_b16 v[10:11], v215 offset:8192
	ds_read_b64_tr_b16 v[12:13], v215 offset:10240
	ds_read_b64_tr_b16 v[14:15], v215 offset:12288
	ds_read_b64_tr_b16 v[16:17], v215 offset:14336
	v_mfma_f32_32x32x16_f16 a[16:31], v[152:155], v[2:5], a[16:31]
	v_add_f32_e32 v138, v138, v139
	v_add_f32_e32 v139, v188, v195
	v_fma_f32 v124, v124, s12, -v113
	v_fma_f32 v125, v125, s12, -v113
	v_fma_f32 v122, v122, s12, -v113
	v_sub_f32_e32 v112, v112, v113
	v_exp_f32_e32 v101, v101
	v_mfma_f32_32x32x16_f16 a[32:47], v[180:183], v[2:5], a[32:47]
	v_exp_f32_e32 v113, v109
	v_exp_f32_e32 v114, v114
	v_exp_f32_e32 v115, v115
	v_exp_f32_e32 v134, v134
	v_exp_f32_e32 v135, v135
	v_exp_f32_e32 v140, v140
	v_exp_f32_e32 v110, v110
	v_mfma_f32_32x32x16_f16 a[48:63], v[184:187], v[2:5], a[48:63]
	v_cvt_pk_f16_f32 v2, v133, v141
	v_cvt_pk_f16_f32 v3, v142, v148
	v_cvt_pk_f16_f32 v4, v33, v98
	v_cvt_pk_f16_f32 v5, v99, v100
	v_exp_f32_e32 v118, v118
	v_exp_f32_e32 v119, v119
	v_exp_f32_e32 v120, v120
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 a[96:111], v[226:229], v[2:5], a[96:111]
	v_exp_f32_e32 v121, v121
	v_exp_f32_e32 v124, v124
	v_exp_f32_e32 v125, v125
	v_exp_f32_e32 v122, v122
	v_exp_f32_e32 v112, v112
	v_cvt_pk_f16_f32 v18, v101, v113
	v_cvt_pk_f16_f32 v19, v114, v115
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 a[80:95], v[234:237], v[2:5], a[80:95]
	v_cvt_pk_f16_f32 v20, v134, v135
	v_cvt_pk_f16_f32 v21, v140, v110
	s_add_u32 s13, s13, s6
	s_addc_u32 s15, s15, s7
	s_add_i32 s14, s14, 64
	v_mov_b32_e32 v109, v104
	s_cmpk_lt_u32 s14, 0x1fc0
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[64:79], a[132:135], v[2:5], a[64:79]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[112:127], v[10:13], v[2:5], a[112:127]
	v_cvt_pk_f16_f32 v2, v111, v126
	v_cvt_pk_f16_f32 v3, v127, v128
	v_cvt_pk_f16_f32 v4, v129, v132
	v_cvt_pk_f16_f32 v5, v116, v117
	s_nop 1
	v_mfma_f32_32x32x16_f16 a[0:15], v[10:13], v[2:5], a[0:15]
	v_add_f32_e32 v10, v137, v138
	v_add_f32_e32 v11, v210, v139
	v_add_f32_e32 v10, v136, v10
	v_add_f32_e32 v11, v243, v11
	v_add_f32_e32 v10, v131, v10
	v_add_f32_e32 v11, v218, v11
	v_add_f32_e32 v10, v130, v10
	v_add_f32_e32 v11, v221, v11
	v_add_f32_e32 v10, v123, v10
	v_add_f32_e32 v0, v0, v11
	v_add_f32_e32 v10, v106, v10
	v_add_f32_e32 v0, v209, v0
	v_add_f32_e32 v10, v143, v10
	v_add_f32_e32 v0, v31, v0
	v_add_f32_e32 v10, v144, v10
	v_add_f32_e32 v0, v32, v0
	v_add_f32_e32 v10, v145, v10
	v_add_f32_e32 v0, v25, v0
	v_add_f32_e32 v10, v146, v10
	v_add_f32_e32 v0, v26, v0
	v_add_f32_e32 v10, v147, v10
	v_add_f32_e32 v0, v27, v0
	v_add_f32_e32 v10, v22, v10
	v_add_f32_e32 v0, v28, v0
	v_add_f32_e32 v10, v23, v10
	v_add_f32_e32 v0, v29, v0
	v_add_f32_e32 v10, v24, v10
	v_add_f32_e32 v0, v30, v0
	v_add_f32_e32 v10, v133, v10
	v_add_f32_e32 v0, v111, v0
	v_mfma_f32_32x32x16_f16 a[48:63], a[132:135], v[2:5], a[48:63]
	v_add_f32_e32 v10, v141, v10
	v_add_f32_e32 v0, v126, v0
	v_add_f32_e32 v10, v142, v10
	v_add_f32_e32 v0, v127, v0
	v_add_f32_e32 v10, v148, v10
	v_add_f32_e32 v0, v128, v0
	v_add_f32_e32 v10, v33, v10
	v_mfma_f32_32x32x16_f16 a[16:31], v[226:229], v[2:5], a[16:31]
	v_add_f32_e32 v0, v129, v0
	v_add_f32_e32 v10, v98, v10
	v_add_f32_e32 v0, v132, v0
	v_add_f32_e32 v10, v99, v10
	v_add_f32_e32 v0, v116, v0
	v_add_f32_e32 v10, v100, v10
	v_add_f32_e32 v0, v117, v0
	v_mfma_f32_32x32x16_f16 a[32:47], v[234:237], v[2:5], a[32:47]
	v_cvt_pk_f16_f32 v2, v118, v119
	v_cvt_pk_f16_f32 v3, v120, v121
	v_cvt_pk_f16_f32 v4, v124, v125
	v_cvt_pk_f16_f32 v5, v122, v112
	v_add_f32_e32 v10, v101, v10
	v_add_f32_e32 v0, v118, v0
	v_add_f32_e32 v0, v119, v0
	v_mfma_f32_32x32x16_f16 a[64:79], v[6:9], v[18:21], a[64:79]
	v_add_f32_e32 v0, v120, v0
	v_add_f32_e32 v0, v121, v0
	v_add_f32_e32 v0, v124, v0
	v_add_f32_e32 v0, v125, v0
	v_add_f32_e32 v0, v122, v0
	v_mfma_f32_32x32x16_f16 a[48:63], v[6:9], v[2:5], a[48:63]
	v_add_f32_e32 v6, v113, v10
	v_add_f32_e32 v6, v114, v6
	v_add_f32_e32 v6, v115, v6
	v_mfma_f32_32x32x16_f16 a[96:111], v[230:233], v[18:21], a[96:111]
	v_mfma_f32_32x32x16_f16 a[80:95], a[128:131], v[18:21], a[80:95]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[112:127], v[14:17], v[18:21], a[112:127]
	v_mfma_f32_32x32x16_f16 a[16:31], v[230:233], v[2:5], a[16:31]
	v_mfma_f32_32x32x16_f16 a[32:47], a[128:131], v[2:5], a[32:47]
	v_mfma_f32_32x32x16_f16 a[0:15], v[14:17], v[2:5], a[0:15]
	v_add_f32_e32 v2, v134, v6
	v_add_f32_e32 v2, v135, v2
	v_add_f32_e32 v2, v140, v2
	v_add_f32_e32 v2, v110, v2
	v_add_f32_e32 v3, v112, v0
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v3
	s_nop 0
	v_permlane32_swap_b32_e32 v2, v4
	v_permlane32_swap_b32_e32 v3, v5
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
	s_nop 0
	v_pk_fma_f32 v[102:103], v[102:103], v[108:109], v[2:3]
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	s_mul_i32 s0, s18, 0xc0000
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s2, s8, s0
	s_addc_u32 s3, s9, s1
	s_lshl_b32 s0, s17, 14
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s2, s2, s0
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[0:1], s[28:29], 2
	s_add_u32 s0, s2, s0
	v_accvgpr_read_b32 v73, a192
	s_addc_u32 s8, s3, s1
	s_add_i32 s1, s28, 0xffffc040
	v_accvgpr_read_b32 v99, a111
	v_accvgpr_read_b32 v98, a110
	v_accvgpr_read_b32 v101, a109
	v_accvgpr_read_b32 v100, a108
	v_accvgpr_read_b32 v109, a107
	v_accvgpr_read_b32 v108, a106
	v_accvgpr_read_b32 v111, a105
	v_accvgpr_read_b32 v110, a104
	v_accvgpr_read_b32 v113, a103
	v_accvgpr_read_b32 v112, a102
	v_accvgpr_read_b32 v115, a101
	v_accvgpr_read_b32 v114, a100
	v_accvgpr_read_b32 v117, a99
	v_accvgpr_read_b32 v116, a98
	v_accvgpr_read_b32 v119, a97
	v_accvgpr_read_b32 v118, a96
	v_accvgpr_read_b32 v121, a95
	v_accvgpr_read_b32 v120, a94
	v_accvgpr_read_b32 v123, a93
	v_accvgpr_read_b32 v122, a92
	v_accvgpr_read_b32 v125, a91
	v_accvgpr_read_b32 v124, a90
	v_accvgpr_read_b32 v127, a89
	v_accvgpr_read_b32 v126, a88
	v_accvgpr_read_b32 v129, a87
	v_accvgpr_read_b32 v128, a86
	v_accvgpr_read_b32 v131, a85
	v_accvgpr_read_b32 v130, a84
	v_accvgpr_read_b32 v133, a83
	v_accvgpr_read_b32 v132, a82
	v_accvgpr_read_b32 v135, a81
	v_accvgpr_read_b32 v134, a80
	v_accvgpr_read_b32 v137, a79
	v_accvgpr_read_b32 v136, a78
	v_accvgpr_read_b32 v139, a77
	v_accvgpr_read_b32 v138, a76
	v_accvgpr_read_b32 v141, a75
	v_accvgpr_read_b32 v140, a74
	v_accvgpr_read_b32 v143, a73
	v_accvgpr_read_b32 v142, a72
	v_accvgpr_read_b32 v145, a71
	v_accvgpr_read_b32 v144, a70
	v_accvgpr_read_b32 v147, a69
	v_accvgpr_read_b32 v146, a68
	v_accvgpr_read_b32 v149, a67
	v_accvgpr_read_b32 v148, a66
	v_accvgpr_read_b32 v151, a65
	v_accvgpr_read_b32 v150, a64
	v_accvgpr_read_b32 v33, a127
	v_accvgpr_read_b32 v32, a126
	v_accvgpr_read_b32 v153, a125
	v_accvgpr_read_b32 v152, a124
	v_accvgpr_read_b32 v155, a123
	v_accvgpr_read_b32 v154, a122
	v_accvgpr_read_b32 v157, a121
	v_accvgpr_read_b32 v156, a120
	v_accvgpr_read_b32 v159, a119
	v_accvgpr_read_b32 v158, a118
	v_accvgpr_read_b32 v161, a117
	v_accvgpr_read_b32 v160, a116
	v_accvgpr_read_b32 v163, a115
	v_accvgpr_read_b32 v162, a114
	v_accvgpr_read_b32 v165, a113
	v_accvgpr_read_b32 v164, a112
	v_accvgpr_read_b32 v25, a31
	v_accvgpr_read_b32 v24, a30
	v_accvgpr_read_b32 v29, a29
	v_accvgpr_read_b32 v28, a28
	v_accvgpr_read_b32 v31, a27
	v_accvgpr_read_b32 v30, a26
	v_accvgpr_read_b32 v167, a25
	v_accvgpr_read_b32 v166, a24
	v_accvgpr_read_b32 v169, a23
	v_accvgpr_read_b32 v168, a22
	v_accvgpr_read_b32 v171, a21
	v_accvgpr_read_b32 v170, a20
	v_accvgpr_read_b32 v173, a19
	v_accvgpr_read_b32 v172, a18
	v_accvgpr_read_b32 v175, a17
	v_accvgpr_read_b32 v174, a16
	v_accvgpr_read_b32 v17, a47
	v_accvgpr_read_b32 v16, a46
	v_accvgpr_read_b32 v21, a45
	v_accvgpr_read_b32 v20, a44
	v_accvgpr_read_b32 v23, a43
	v_accvgpr_read_b32 v22, a42
	v_accvgpr_read_b32 v27, a41
	v_accvgpr_read_b32 v26, a40
	v_accvgpr_read_b32 v177, a39
	v_accvgpr_read_b32 v176, a38
	v_accvgpr_read_b32 v179, a37
	v_accvgpr_read_b32 v178, a36
	v_accvgpr_read_b32 v181, a35
	v_accvgpr_read_b32 v180, a34
	v_accvgpr_read_b32 v183, a33
	v_accvgpr_read_b32 v182, a32
	v_accvgpr_read_b32 v9, a63
	v_accvgpr_read_b32 v8, a62
	v_accvgpr_read_b32 v13, a61
	v_accvgpr_read_b32 v12, a60
	v_accvgpr_read_b32 v15, a59
	v_accvgpr_read_b32 v14, a58
	v_accvgpr_read_b32 v19, a57
	v_accvgpr_read_b32 v18, a56
	v_accvgpr_read_b32 v185, a55
	v_accvgpr_read_b32 v184, a54
	v_accvgpr_read_b32 v187, a53
	v_accvgpr_read_b32 v186, a52
	v_accvgpr_read_b32 v189, a51
	v_accvgpr_read_b32 v188, a50
	v_accvgpr_read_b32 v191, a49
	v_accvgpr_read_b32 v190, a48
	v_accvgpr_read_b32 v3, a15
	v_accvgpr_read_b32 v2, a14
	v_accvgpr_read_b32 v5, a13
	v_accvgpr_read_b32 v4, a12
	v_accvgpr_read_b32 v7, a11
	v_accvgpr_read_b32 v6, a10
	v_accvgpr_read_b32 v11, a9
	v_accvgpr_read_b32 v10, a8
	v_accvgpr_read_b32 v193, a7
	v_accvgpr_read_b32 v192, a6
	v_accvgpr_read_b32 v195, a5
	v_accvgpr_read_b32 v194, a4
	v_accvgpr_read_b32 v197, a3
	v_accvgpr_read_b32 v196, a2
	v_accvgpr_read_b32 v199, a1
	v_accvgpr_read_b32 v198, a0
	v_or_b32_e32 v36, 32, v73
	s_cmp_lt_i32 s1, 1
	v_lshl_add_u32 v0, v73, 2, 0
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	s_mov_b32 s2, 0x800000
	v_mov_b32_e32 v34, 0x42000000
	v_cmp_gt_f32_e32 vcc, s2, v102
	v_or_b32_e32 v1, s28, v36
	s_movk_i32 s1, 0x4000
	v_cndmask_b32_e32 v35, 0, v34, vcc
	v_cndmask_b32_e64 v37, 0, 32, vcc
	v_cmp_gt_f32_e32 vcc, s2, v103
	v_ldexp_f32 v37, v102, v37
	v_log_f32_e32 v37, v37
	v_cndmask_b32_e64 v38, 0, 32, vcc
	v_ldexp_f32 v38, v103, v38
	v_log_f32_e32 v38, v38
	v_cmp_gt_i32_e64 s[4:5], s1, v1
	v_or_b32_e32 v1, s28, v73
	v_cndmask_b32_e32 v34, 0, v34, vcc
	v_cmp_gt_i32_e64 s[6:7], s1, v1
	v_sub_f32_e32 v1, v37, v35
	v_sub_f32_e32 v34, v38, v34
	v_add_f32_e32 v1, v105, v1
	v_add_f32_e32 v34, v107, v34
	v_accvgpr_read_b32 v37, a195
	ds_write2_b32 v0, v1, v34 offset1:32
	v_add_u32_e32 v1, 0, v37
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v1, v1
	s_sub_i32 s9, 0x4000, s28
	v_accvgpr_read_b32 v35, a194
	v_bfrev_b32_e32 v34, 1
	v_cmp_gt_i32_e32 vcc, s9, v35
	s_and_b32 s1, s8, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v34, v34, v37, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v1, v34, s[0:3], 0 offen
	s_cbranch_execz .LBB0_5
	s_branch .LBB0_6
.LBB0_4:
                                        ; implicit-def: $sgpr4_sgpr5
                                        ; implicit-def: $sgpr6_sgpr7
.LBB0_5:
	s_mov_b32 s1, 0x800000
	v_mov_b32_e32 v1, 0x42000000
	v_cmp_gt_f32_e32 vcc, s1, v102
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v34, 0, v1, vcc
	v_cndmask_b32_e64 v35, 0, 32, vcc
	v_cmp_gt_f32_e32 vcc, s1, v103
	v_ldexp_f32 v35, v102, v35
	v_log_f32_e32 v35, v35
	v_cndmask_b32_e64 v37, 0, 32, vcc
	v_ldexp_f32 v37, v103, v37
	v_log_f32_e32 v37, v37
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_sub_f32_e32 v34, v35, v34
	v_add_f32_e32 v34, v105, v34
	v_sub_f32_e32 v1, v37, v1
	v_add_f32_e32 v1, v107, v1
	ds_write2_b32 v0, v34, v1 offset1:32
	v_accvgpr_read_b32 v1, a195
	v_add_u32_e32 v0, 0, v1
	s_waitcnt lgkmcnt(0)
	; wave barrier
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v0, v0
	s_and_b32 s1, s8, 0xffff
	s_or_b64 s[4:5], s[4:5], exec
	s_or_b64 s[6:7], s[6:7], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v0, v1, s[0:3], 0 offen
.LBB0_6:                                ; %.critedge
	v_div_scale_f32 v0, s[0:1], v103, v103, 1.0
	v_rcp_f32_e32 v0, v0
	v_div_scale_f32 v1, vcc, 1.0, v103, 1.0
	v_accvgpr_read_b32 v37, a193
	v_mul_f32_e32 v0, v1, v0
	v_div_scale_f32 v1, s[0:1], v102, v102, 1.0
	v_rcp_f32_e32 v1, v1
	v_div_fmas_f32 v0, 0, 0, v0
	v_div_fixup_f32 v34, v0, v103, 1.0
	v_div_scale_f32 v0, vcc, 1.0, v102, 1.0
	v_mul_f32_e32 v0, v0, v1
	v_pk_mul_f32 v[8:9], v[8:9], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[34:35] op_sel_hi:[1,0]
	s_nop 0
	v_div_fmas_f32 v0, 0, 0, v0
	v_div_fixup_f32 v62, v0, v102, 1.0
	v_pk_mul_f32 v[0:1], v[2:3], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v3, v0, v1
	v_pk_mul_f32 v[0:1], v[4:5], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[10:11], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v2, v0, v1
	v_pk_mul_f32 v[0:1], v[6:7], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[198:199], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v1, v0, v1
	v_cvt_pk_f16_f32 v0, v4, v5
	v_pk_mul_f32 v[4:5], v[192:193], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v7, v4, v5
	v_pk_mul_f32 v[4:5], v[194:195], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[156:157], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v6, v4, v5
	v_pk_mul_f32 v[4:5], v[196:197], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[164:165], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v5, v4, v5
	v_cvt_pk_f16_f32 v4, v10, v11
	v_cvt_pk_f16_f32 v11, v8, v9
	v_pk_mul_f32 v[8:9], v[12:13], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[18:19], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v10, v8, v9
	v_pk_mul_f32 v[8:9], v[14:15], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[190:191], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v9, v8, v9
	v_cvt_pk_f16_f32 v8, v12, v13
	v_pk_mul_f32 v[12:13], v[184:185], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[142:143], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v15, v12, v13
	v_pk_mul_f32 v[12:13], v[186:187], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[150:151], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v14, v12, v13
	v_pk_mul_f32 v[12:13], v[188:189], v[34:35] op_sel_hi:[1,0]
	s_mul_i32 s0, s25, s18
	v_cvt_pk_f16_f32 v13, v12, v13
	v_cvt_pk_f16_f32 v12, v18, v19
	v_cvt_pk_f16_f32 v19, v16, v17
	v_pk_mul_f32 v[16:17], v[20:21], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[26:27], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v18, v16, v17
	v_pk_mul_f32 v[16:17], v[22:23], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[182:183], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v17, v16, v17
	v_cvt_pk_f16_f32 v16, v20, v21
	v_pk_mul_f32 v[20:21], v[176:177], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[126:127], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v23, v20, v21
	v_pk_mul_f32 v[20:21], v[178:179], v[34:35] op_sel_hi:[1,0]
	s_ashr_i32 s1, s0, 31
	v_cvt_pk_f16_f32 v22, v20, v21
	v_pk_mul_f32 v[20:21], v[180:181], v[34:35] op_sel_hi:[1,0]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v21, v20, v21
	v_cvt_pk_f16_f32 v20, v26, v27
	v_cvt_pk_f16_f32 v27, v24, v25
	v_pk_mul_f32 v[24:25], v[28:29], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[166:167], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v26, v24, v25
	v_pk_mul_f32 v[24:25], v[30:31], v[34:35] op_sel_hi:[1,0]
	s_add_u32 s2, s10, s0
	v_cvt_pk_f16_f32 v25, v24, v25
	v_cvt_pk_f16_f32 v24, v28, v29
	v_pk_mul_f32 v[28:29], v[168:169], v[34:35] op_sel_hi:[1,0]
	s_mul_i32 s0, s26, s17
	v_cvt_pk_f16_f32 v31, v28, v29
	v_pk_mul_f32 v[28:29], v[170:171], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[134:135], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v30, v28, v29
	v_pk_mul_f32 v[28:29], v[172:173], v[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[174:175], v[34:35] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v29, v28, v29
	v_cvt_pk_f16_f32 v28, v34, v35
	v_cvt_pk_f16_f32 v35, v32, v33
	v_pk_mul_f32 v[32:33], v[152:153], v[62:63] op_sel_hi:[1,0]
	s_addc_u32 s3, s11, s1
	v_cvt_pk_f16_f32 v34, v32, v33
	v_pk_mul_f32 v[32:33], v[154:155], v[62:63] op_sel_hi:[1,0]
	s_ashr_i32 s1, s0, 31
	v_cvt_pk_f16_f32 v33, v32, v33
	v_cvt_pk_f16_f32 v32, v38, v39
	v_pk_mul_f32 v[38:39], v[158:159], v[62:63] op_sel_hi:[1,0]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v41, v38, v39
	v_pk_mul_f32 v[38:39], v[160:161], v[62:63] op_sel_hi:[1,0]
	s_add_u32 s2, s2, s0
	v_cvt_pk_f16_f32 v40, v38, v39
	v_pk_mul_f32 v[38:39], v[162:163], v[62:63] op_sel_hi:[1,0]
	s_mul_i32 s0, s27, s28
	v_cvt_pk_f16_f32 v39, v38, v39
	v_cvt_pk_f16_f32 v38, v42, v43
	v_pk_mul_f32 v[42:43], v[136:137], v[62:63] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[110:111], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v45, v42, v43
	v_pk_mul_f32 v[42:43], v[138:139], v[62:63] op_sel_hi:[1,0]
	s_addc_u32 s3, s3, s1
	v_cvt_pk_f16_f32 v44, v42, v43
	v_pk_mul_f32 v[42:43], v[140:141], v[62:63] op_sel_hi:[1,0]
	s_ashr_i32 s1, s0, 31
	v_cvt_pk_f16_f32 v43, v42, v43
	v_cvt_pk_f16_f32 v42, v46, v47
	v_pk_mul_f32 v[46:47], v[144:145], v[62:63] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[114:115], v[62:63] op_sel_hi:[1,0]
	v_cvt_pk_f16_f32 v49, v46, v47
	v_pk_mul_f32 v[46:47], v[146:147], v[62:63] op_sel_hi:[1,0]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v48, v46, v47
	v_pk_mul_f32 v[46:47], v[148:149], v[62:63] op_sel_hi:[1,0]
	s_add_u32 s0, s2, s0
	v_cvt_pk_f16_f32 v47, v46, v47
	v_cvt_pk_f16_f32 v46, v50, v51
	v_pk_mul_f32 v[50:51], v[120:121], v[62:63] op_sel_hi:[1,0]
	s_addc_u32 s1, s3, s1
	v_cvt_pk_f16_f32 v53, v50, v51
	v_pk_mul_f32 v[50:51], v[122:123], v[62:63] op_sel_hi:[1,0]
	s_and_b32 s2, s27, 0x3fff
	v_cvt_pk_f16_f32 v52, v50, v51
	v_pk_mul_f32 v[50:51], v[124:125], v[62:63] op_sel_hi:[1,0]
	v_lshrrev_b32_e32 v37, 2, v37
	v_cvt_pk_f16_f32 v51, v50, v51
	v_cvt_pk_f16_f32 v50, v54, v55
	v_pk_mul_f32 v[54:55], v[128:129], v[62:63] op_sel_hi:[1,0]
	v_mul_lo_u32 v73, s27, v73
	v_cvt_pk_f16_f32 v57, v54, v55
	v_pk_mul_f32 v[54:55], v[130:131], v[62:63] op_sel_hi:[1,0]
	s_bitset1_b32 s2, 14
	v_cvt_pk_f16_f32 v56, v54, v55
	v_pk_mul_f32 v[54:55], v[132:133], v[62:63] op_sel_hi:[1,0]
	s_and_b32 s1, s1, 0xffff
	v_cvt_pk_f16_f32 v55, v54, v55
	v_cvt_pk_f16_f32 v54, v58, v59
	v_pk_mul_f32 v[58:59], v[98:99], v[62:63] op_sel_hi:[1,0]
	s_lshl_b32 s2, s2, 16
	v_cvt_pk_f16_f32 v61, v58, v59
	v_pk_mul_f32 v[58:59], v[100:101], v[62:63] op_sel_hi:[1,0]
	v_add_lshl_u32 v74, v73, v37, 1
	v_cvt_pk_f16_f32 v60, v58, v59
	v_pk_mul_f32 v[58:59], v[108:109], v[62:63] op_sel_hi:[1,0]
	v_bfrev_b32_e32 v75, 1
	v_cvt_pk_f16_f32 v59, v58, v59
	v_cvt_pk_f16_f32 v58, v64, v65
	v_pk_mul_f32 v[64:65], v[112:113], v[62:63] op_sel_hi:[1,0]
	v_or_b32_e32 v72, 16, v37
	v_cvt_pk_f16_f32 v65, v64, v65
	v_cvt_pk_f16_f32 v64, v66, v67
	v_pk_mul_f32 v[66:67], v[116:117], v[62:63] op_sel_hi:[1,0]
	s_or_b32 s1, s1, s2
	v_cvt_pk_f16_f32 v63, v66, v67
	v_pk_mul_f32 v[66:67], v[118:119], v[62:63] op_sel_hi:[1,0]
	s_nop 0
	v_permlane32_swap_b32_e32 v63, v65
	v_cvt_pk_f16_f32 v62, v66, v67
	s_nop 1
	v_permlane32_swap_b32_e32 v62, v64
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e64 v74, v75, v74, s[6:7]
	buffer_store_dwordx4 v[62:65], v74, s[0:3], 0 offen
	v_or_b32_e32 v71, 32, v37
	v_permlane32_swap_b32_e32 v58, v60
	v_add_lshl_u32 v62, v73, v72, 1
	v_permlane32_swap_b32_e32 v59, v61
	v_cndmask_b32_e64 v62, v75, v62, s[6:7]
	buffer_store_dwordx4 v[58:61], v62, s[0:3], 0 offen
	v_or_b32_e32 v70, 48, v37
	v_permlane32_swap_b32_e32 v54, v56
	v_add_lshl_u32 v58, v73, v71, 1
	v_permlane32_swap_b32_e32 v55, v57
	v_cndmask_b32_e64 v58, v75, v58, s[6:7]
	buffer_store_dwordx4 v[54:57], v58, s[0:3], 0 offen
	v_or_b32_e32 v69, 64, v37
	v_permlane32_swap_b32_e32 v50, v52
	v_add_lshl_u32 v54, v73, v70, 1
	v_permlane32_swap_b32_e32 v51, v53
	v_cndmask_b32_e64 v54, v75, v54, s[6:7]
	buffer_store_dwordx4 v[50:53], v54, s[0:3], 0 offen
	v_or_b32_e32 v68, 0x50, v37
	v_permlane32_swap_b32_e32 v46, v48
	v_add_lshl_u32 v50, v73, v69, 1
	v_permlane32_swap_b32_e32 v47, v49
	v_cndmask_b32_e64 v50, v75, v50, s[6:7]
	buffer_store_dwordx4 v[46:49], v50, s[0:3], 0 offen
	v_or_b32_e32 v67, 0x60, v37
	v_permlane32_swap_b32_e32 v42, v44
	v_add_lshl_u32 v46, v73, v68, 1
	v_permlane32_swap_b32_e32 v43, v45
	v_cndmask_b32_e64 v46, v75, v46, s[6:7]
	buffer_store_dwordx4 v[42:45], v46, s[0:3], 0 offen
	v_or_b32_e32 v66, 0x70, v37
	v_permlane32_swap_b32_e32 v38, v40
	v_add_lshl_u32 v42, v73, v67, 1
	v_permlane32_swap_b32_e32 v39, v41
	v_cndmask_b32_e64 v42, v75, v42, s[6:7]
	buffer_store_dwordx4 v[38:41], v42, s[0:3], 0 offen
	v_mul_lo_u32 v36, s27, v36
	v_permlane32_swap_b32_e32 v32, v34
	v_add_lshl_u32 v38, v73, v66, 1
	v_permlane32_swap_b32_e32 v33, v35
	v_cndmask_b32_e64 v38, v75, v38, s[6:7]
	buffer_store_dwordx4 v[32:35], v38, s[0:3], 0 offen
	v_permlane32_swap_b32_e32 v28, v30
	s_nop 0
	v_add_lshl_u32 v32, v36, v37, 1
	v_permlane32_swap_b32_e32 v29, v31
	v_cndmask_b32_e64 v32, v75, v32, s[4:5]
	buffer_store_dwordx4 v[28:31], v32, s[0:3], 0 offen
	v_permlane32_swap_b32_e32 v24, v26
	s_nop 0
	v_add_lshl_u32 v28, v36, v72, 1
	v_permlane32_swap_b32_e32 v25, v27
	v_cndmask_b32_e64 v28, v75, v28, s[4:5]
	buffer_store_dwordx4 v[24:27], v28, s[0:3], 0 offen
	v_permlane32_swap_b32_e32 v20, v22
	s_nop 0
	v_add_lshl_u32 v24, v36, v71, 1
	v_permlane32_swap_b32_e32 v21, v23
	v_cndmask_b32_e64 v24, v75, v24, s[4:5]
	buffer_store_dwordx4 v[20:23], v24, s[0:3], 0 offen
	v_permlane32_swap_b32_e32 v16, v18
	s_nop 0
	v_add_lshl_u32 v20, v36, v70, 1
	v_permlane32_swap_b32_e32 v17, v19
	v_cndmask_b32_e64 v20, v75, v20, s[4:5]
	buffer_store_dwordx4 v[16:19], v20, s[0:3], 0 offen
	v_permlane32_swap_b32_e32 v12, v14
	s_nop 0
	v_add_lshl_u32 v16, v36, v69, 1
	v_permlane32_swap_b32_e32 v13, v15
	v_cndmask_b32_e64 v16, v75, v16, s[4:5]
	buffer_store_dwordx4 v[12:15], v16, s[0:3], 0 offen
	v_permlane32_swap_b32_e32 v8, v10
	s_nop 0
	v_add_lshl_u32 v12, v36, v68, 1
	v_permlane32_swap_b32_e32 v9, v11
	v_cndmask_b32_e64 v12, v75, v12, s[4:5]
	buffer_store_dwordx4 v[8:11], v12, s[0:3], 0 offen
	v_permlane32_swap_b32_e32 v4, v6
	s_nop 0
	v_add_lshl_u32 v8, v36, v67, 1
	v_permlane32_swap_b32_e32 v5, v7
	v_cndmask_b32_e64 v8, v75, v8, s[4:5]
	buffer_store_dwordx4 v[4:7], v8, s[0:3], 0 offen
	v_permlane32_swap_b32_e32 v0, v2
	s_nop 0
	v_add_lshl_u32 v4, v36, v66, 1
	v_permlane32_swap_b32_e32 v1, v3
	v_cndmask_b32_e64 v4, v75, v4, s[4:5]
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 144
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 485
		.amdhsa_next_free_sgpr 30
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
	.size	attn_fwd, .Lfunc_end0-attn_fwd
	.cfi_endproc
                                        ; -- End function
	.set attn_fwd.num_vgpr, 256
	.set attn_fwd.num_agpr, 229
	.set attn_fwd.numbered_sgpr, 30
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13600
; TotalNumSgprs: 36
; NumVgprs: 256
; NumAgprs: 229
; TotalNumVgprs: 485
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 60
; NumSGPRsForWavesPerEU: 36
; NumVGPRsForWavesPerEU: 485
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x45 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x1f DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	679                             ; DW_AT_call_line
	.byte	61                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"flash-attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/app/OAI-triton/fa"            ; string offset=26
.Linfo_string3:
	.asciz	"attn_fwd"                      ; string offset=45
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     229
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
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .offset:         108
        .size:           4
        .value_kind:     by_value
      - .offset:         112
        .size:           4
        .value_kind:     by_value
      - .offset:         116
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .offset:         128
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .max_flat_workgroup_size: 64
    .name:           attn_fwd
    .private_segment_fixed_size: 0
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     485
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
