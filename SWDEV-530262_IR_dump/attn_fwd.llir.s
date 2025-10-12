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
; %bb.11:
	.file	1 "/app/OAI-triton/fav3_kernel" "flash-attention.py"
	.loc	1 465 0 prologue_end            ; flash-attention.py:465:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.12:
.LBB0_0:
	s_mov_b64 s[36:37], s[2:3]
.Ltmp1:
	.loc	1 46 23 is_stmt 1               ; flash-attention.py:46:23 @[ flash-attention.py:519:41 ]
	s_ashr_i32 s2, s16, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s2, s16, s2
	s_ashr_i32 s2, s2, 3
	.loc	1 52 35                         ; flash-attention.py:52:35 @[ flash-attention.py:519:41 ]
	s_lshl_b32 s3, s16, 3
	s_mul_i32 s19, s2, 0xffffffc1
	s_add_i32 s19, s19, s3
.Ltmp2:
	.loc	1 523 27                        ; flash-attention.py:523:27
	s_lshl_b32 s16, s17, 8
	.loc	1 602 39                        ; flash-attention.py:602:39
	s_mul_i32 s12, s12, s18
	.loc	1 602 61 is_stmt 0              ; flash-attention.py:602:61
	s_mul_i32 s2, s13, s19
	.loc	1 602 51                        ; flash-attention.py:602:51
	s_add_i32 s2, s2, s12
	.loc	1 603 36 is_stmt 1              ; flash-attention.py:603:36
	s_mul_i32 s3, s14, s16
	.loc	1 603 73 is_stmt 0              ; flash-attention.py:603:73
	v_and_b32_e32 v39, 15, v0
	.loc	1 523 50 is_stmt 1              ; flash-attention.py:523:50
	v_lshrrev_b32_e32 v34, 4, v0
	.loc	1 603 73                        ; flash-attention.py:603:73
	v_lshlrev_b32_e32 v210, 3, v39
	.loc	1 603 66 is_stmt 0              ; flash-attention.py:603:66
	s_add_i32 s2, s2, s3
	s_mov_b64 s[24:25], s[6:7]
	.loc	1 523 50 is_stmt 1              ; flash-attention.py:523:50
	v_or_b32_e32 v35, 32, v34
	v_or_b32_e32 v4, s16, v34
	.loc	1 603 66                        ; flash-attention.py:603:66
	v_mul_lo_u32 v12, s14, v34
	s_lshl_b32 s6, s14, 6
	v_add_u32_e32 v17, s2, v210
	s_movk_i32 s2, 0x4000
	.loc	1 523 37                        ; flash-attention.py:523:37
	v_or_b32_e32 v5, s16, v35
	.loc	1 603 66                        ; flash-attention.py:603:66
	v_mul_lo_u32 v13, s14, v35
	v_add_u32_e32 v14, s6, v12
	v_add_lshl_u32 v12, v17, v12, 1
	v_bfrev_b32_e32 v18, 1
	.loc	1 653 48                        ; flash-attention.py:653:48
	v_cmp_gt_i32_e32 vcc, s2, v4
	.loc	1 523 50                        ; flash-attention.py:523:50
	v_or_b32_e32 v1, 0x60, v34
	.loc	1 523 37 is_stmt 0              ; flash-attention.py:523:37
	v_or_b32_e32 v6, 64, v4
	v_or_b32_e32 v8, 0x80, v4
	v_or_b32_e32 v10, 0xc0, v4
	v_cndmask_b32_e32 v36, v18, v12, vcc
	v_add_lshl_u32 v4, v17, v13, 1
	.loc	1 653 48 is_stmt 1              ; flash-attention.py:653:48
	v_cmp_gt_i32_e32 vcc, s2, v5
	.loc	1 523 37                        ; flash-attention.py:523:37
	v_or_b32_e32 v7, s16, v1
	.loc	1 603 66                        ; flash-attention.py:603:66
	v_mul_lo_u32 v1, s14, v1
	v_cndmask_b32_e32 v37, v18, v4, vcc
	v_add_lshl_u32 v4, v17, v14, 1
	.loc	1 653 48                        ; flash-attention.py:653:48
	v_cmp_gt_i32_e32 vcc, s2, v6
	.loc	1 523 50                        ; flash-attention.py:523:50
	v_or_b32_e32 v2, 0xa0, v34
	.loc	1 603 66                        ; flash-attention.py:603:66
	v_add_u32_e32 v15, s6, v14
	v_cndmask_b32_e32 v38, v18, v4, vcc
	v_add_lshl_u32 v1, v17, v1, 1
	.loc	1 653 48                        ; flash-attention.py:653:48
	v_cmp_gt_i32_e32 vcc, s2, v7
	.loc	1 523 37                        ; flash-attention.py:523:37
	v_or_b32_e32 v9, s16, v2
	.loc	1 603 66                        ; flash-attention.py:603:66
	v_mul_lo_u32 v2, s14, v2
	v_cndmask_b32_e32 v1, v18, v1, vcc
	v_add_lshl_u32 v4, v17, v15, 1
	.loc	1 653 48                        ; flash-attention.py:653:48
	v_cmp_gt_i32_e32 vcc, s2, v8
	.loc	1 523 50                        ; flash-attention.py:523:50
	v_or_b32_e32 v3, 0xe0, v34
	.loc	1 603 66                        ; flash-attention.py:603:66
	v_add_u32_e32 v16, s6, v15
	v_cndmask_b32_e32 v40, v18, v4, vcc
	v_add_lshl_u32 v2, v17, v2, 1
	.loc	1 653 48                        ; flash-attention.py:653:48
	v_cmp_gt_i32_e32 vcc, s2, v9
	.loc	1 523 37                        ; flash-attention.py:523:37
	v_or_b32_e32 v11, s16, v3
	.loc	1 603 66                        ; flash-attention.py:603:66
	v_mul_lo_u32 v3, s14, v3
	v_cndmask_b32_e32 v41, v18, v2, vcc
	v_add_lshl_u32 v2, v17, v16, 1
	.loc	1 653 48                        ; flash-attention.py:653:48
	v_cmp_gt_i32_e32 vcc, s2, v10
	s_and_b32 s37, s37, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e32 v42, v18, v2, vcc
	v_add_lshl_u32 v2, v17, v3, 1
	v_cmp_gt_i32_e32 vcc, s2, v11
	.loc	1 603 73                        ; flash-attention.py:603:73
	v_and_b32_e32 v211, 32, v0
	.loc	1 523 50                        ; flash-attention.py:523:50
	v_and_b32_e32 v198, 0x1c0, v0
	v_cndmask_b32_e32 v43, v18, v2, vcc
	.loc	1 656 28                        ; flash-attention.py:656:28
	buffer_load_dwordx4 v[2:5], v36, s[36:39], 0 offen
	buffer_load_dwordx4 v[6:9], v37, s[36:39], 0 offen
	buffer_load_dwordx4 v[10:13], v38, s[36:39], 0 offen
	buffer_load_dwordx4 v[14:17], v1, s[36:39], 0 offen
	buffer_load_dwordx4 v[18:21], v40, s[36:39], 0 offen
	buffer_load_dwordx4 v[22:25], v41, s[36:39], 0 offen
	buffer_load_dwordx4 v[26:29], v42, s[36:39], 0 offen
	buffer_load_dwordx4 v[30:33], v43, s[36:39], 0 offen
	s_load_dwordx4 s[28:31], s[0:1], 0x38
	s_load_dword s23, s[0:1], 0x48
	.loc	1 605 73                        ; flash-attention.py:605:73
	v_and_b32_e32 v42, 16, v0
	.loc	1 523 50                        ; flash-attention.py:523:50
	v_lshrrev_b32_e32 v36, 6, v0
	.loc	1 605 73                        ; flash-attention.py:605:73
	v_lshlrev_b32_e32 v37, 1, v42
	v_lshrrev_b32_e32 v38, 1, v211
	v_or3_b32 v40, v36, v37, v38
	v_mov_b32_e32 v55, 0x2000
	.loc	1 523 50                        ; flash-attention.py:523:50
	v_and_b32_e32 v1, 31, v0
	.loc	1 605 66                        ; flash-attention.py:605:66
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[40:41], s[2:3], s29, v40, v[210:211]
	v_lshlrev_b32_e32 v44, 7, v198
	v_lshlrev_b32_e32 v214, 4, v39
	v_lshl_or_b32 v55, v36, 10, v55
	s_mov_b64 s[20:21], s[10:11]
	s_movk_i32 s6, 0x60
	s_movk_i32 s7, 0xa0
	s_movk_i32 s10, 0xe0
	s_movk_i32 s11, 0x80
	s_movk_i32 s12, 0xc0
	v_lshl_or_b32 v44, v1, 8, v44
	v_xor_b32_e32 v45, v214, v38
	s_movk_i32 s2, 0x410
	v_lshrrev_b32_e32 v56, 6, v55
	v_or_b32_e32 v46, v44, v45
	v_bitop3_b32 v49, v44, s6, v45 bitop3:0x36
	v_bitop3_b32 v50, v44, s11, v45 bitop3:0x36
	v_bitop3_b32 v51, v44, s7, v45 bitop3:0x36
	v_bitop3_b32 v52, v44, s12, v45 bitop3:0x36
	v_bitop3_b32 v44, v44, s10, v45 bitop3:0x36
	v_mad_u32_u24 v45, v36, s2, 0
	v_or_b32_e32 v199, v56, v55
	.loc	1 604 39                        ; flash-attention.py:604:39
	s_mul_i32 s33, s15, s18
	.loc	1 604 61 is_stmt 0              ; flash-attention.py:604:61
	s_mul_i32 s28, s28, s19
	v_add_u32_e32 v53, 0x87c0, v45
	v_add_u32_e32 v56, 0, v199
	.loc	1 604 51                        ; flash-attention.py:604:51
	s_add_i32 s35, s28, s33
.Ltmp3:
	.loc	1 372 28 is_stmt 1              ; flash-attention.py:372:28 @[ flash-attention.py:708:61 ]
	s_lshl_b32 s36, s29, 6
	v_lshlrev_b32_e32 v215, 4, v0
	v_and_b32_e32 v43, 0xf0, v0
	s_and_b32 s5, s5, 0xffff
	v_add_u32_e32 v57, 0x87c0, v56
.Ltmp4:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	v_readfirstlane_b32 s10, v53
.Ltmp5:
	.loc	1 605 66                        ; flash-attention.py:605:66
	v_lshl_add_u32 v41, s29, 3, v40
	v_xad_u32 v43, v215, v43, 0
	v_add_u32_e32 v47, 0, v46
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s38
	s_mov_b32 s15, s39
	v_add_lshl_u32 v54, v40, s35, 1
	v_add_u32_e32 v59, s36, v40
	v_add_u32_e32 v40, 0xc8c0, v45
	v_lshlrev_b32_e32 v39, 10, v39
	v_lshlrev_b32_e32 v42, 5, v42
.Ltmp6:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s11, v57
	v_xad_u32 v48, v46, 32, 0
	v_xad_u32 v46, v46, 64, 0
	v_add_u32_e32 v49, 0, v49
	v_add_u32_e32 v50, 0, v50
	v_add_u32_e32 v51, 0, v51
	v_add_u32_e32 v52, 0, v52
	v_add_u32_e32 v44, 0, v44
	v_add_lshl_u32 v58, v41, s35, 1
	v_add_u32_e32 v45, 0xc8c0, v56
	v_or3_b32 v39, v39, v42, v38
.Ltmp7:
	.loc	1 656 28                        ; flash-attention.py:656:28
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(7)
	ds_write_b128 v43, v[2:5]
	s_waitcnt vmcnt(6)
	ds_write_b128 v43, v[6:9] offset:8192
	s_waitcnt vmcnt(5)
	ds_write_b128 v43, v[10:13] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b128 v43, v[14:17] offset:24576
	s_waitcnt vmcnt(3)
	ds_write_b128 v43, v[18:21] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v43, v[22:25] offset:40960
	s_waitcnt vmcnt(1)
	ds_write_b128 v43, v[26:29] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b128 v43, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[142:145], v47
	ds_read_b128 v[138:141], v48
	ds_read_b128 v[134:137], v46
	ds_read_b128 v[130:133], v49
	ds_read_b128 v[126:129], v50
	ds_read_b128 v[122:125], v51
	ds_read_b128 v[118:121], v52
	ds_read_b128 v[114:117], v44
.Ltmp8:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v54, s[12:15], 0 offen lds
	s_mov_b32 m0, s11
	v_readfirstlane_b32 s2, v40
	v_add_u32_e32 v60, s36, v41
	v_add_lshl_u32 v41, v59, s35, 1
	v_add_u32_e32 v217, v39, v214
	buffer_load_dwordx4 v58, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v45
	v_add_lshl_u32 v56, v60, s35, 1
	v_add_u32_e32 v39, 0, v217
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v41, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
.Ltmp9:
	.loc	1 607 43                        ; flash-attention.py:607:43
	v_and_b32_e32 v200, 48, v0
.Ltmp10:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	buffer_load_dwordx4 v56, s[12:15], 0 offen lds
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[2:5], v39 offset:34752
	ds_read_b128 v[18:21], v39 offset:34784
.Ltmp11:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[142:145], 0
	v_lshrrev_b32_e32 v50, 4, v55
.Ltmp12:
	.loc	1 606 39                        ; flash-attention.py:606:39
	s_mul_i32 s34, s30, s18
	.loc	1 606 61 is_stmt 0              ; flash-attention.py:606:61
	s_mul_i32 s31, s31, s19
	v_or_b32_e32 v216, v50, v55
	s_add_i32 s35, s35, s36
	.loc	1 606 51                        ; flash-attention.py:606:51
	s_add_i32 s30, s31, s34
	v_add_u32_e32 v50, 0, v216
.Ltmp13:
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[138:141], v[2:17]
.Ltmp14:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[18:21], v39 offset:34816
	ds_read_b128 v[22:25], v39 offset:34848
	v_add_lshl_u32 v54, v59, s35, 1
	v_add_lshl_u32 v55, v60, s35, 1
	s_and_b32 s25, s25, 0xffff
	s_mov_b32 s26, s38
	s_mov_b32 s27, s39
.Ltmp15:
	.loc	1 373 28                        ; flash-attention.py:373:28 @[ flash-attention.py:708:61 ]
	s_lshl_b32 s17, s23, 6
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[134:137], v[2:17]
	v_add_u32_e32 v57, 0x4400, v50
	s_mov_b32 s22, 0
	s_mov_b32 s6, s38
	s_mov_b32 s7, s39
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[130:133], v[2:17]
.Ltmp16:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[18:21], v39 offset:34880
	ds_read_b128 v[22:25], v39 offset:34912
.Ltmp17:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[126:129], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[122:125], v[2:17]
.Ltmp18:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[18:21], v39 offset:34944
	ds_read_b128 v[22:25], v39 offset:34976
.Ltmp19:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[18:21], v[118:121], v[2:17]
.Ltmp20:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[18:21], v39 offset:35008
	ds_read_b128 v[40:43], v39 offset:35040
	ds_read_b128 v[44:47], v39 offset:35072
.Ltmp21:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[114:117], v[2:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[142:145], 0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[138:141], v[18:33]
.Ltmp22:
	.loc	1 607 43                        ; flash-attention.py:607:43
	v_or_b32_e32 v40, v36, v200
	.loc	1 607 66 is_stmt 0              ; flash-attention.py:607:66
	v_mad_u64_u32 v[48:49], s[2:3], s23, v40, v[210:211]
.Ltmp23:
	.loc	1 213 25 is_stmt 1              ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[40:43], v39 offset:35104
	s_movk_i32 s2, 0x440
	v_mad_u32_u24 v51, v36, s2, 0
.Ltmp24:
	.loc	1 607 66                        ; flash-attention.py:607:66
	v_lshl_add_u32 v49, s23, 3, v48
.Ltmp25:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[134:137], v[18:33]
.Ltmp26:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[44:47], v39 offset:35136
.Ltmp27:
	.loc	1 213 25 is_stmt 0              ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_readfirstlane_b32 s35, v51
	v_add_lshl_u32 v52, v48, s30, 1
	s_mov_b32 m0, s35
	v_readfirstlane_b32 s35, v50
	v_add_lshl_u32 v53, v49, s30, 1
	buffer_load_dwordx4 v52, s[24:27], 0 offen lds
.Ltmp28:
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[130:133], v[18:33]
.Ltmp29:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[40:43], v39 offset:35168
.Ltmp30:
	.loc	1 213 25 is_stmt 0              ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_mov_b32 m0, s35
	v_add_u32_e32 v56, 0x4400, v51
	buffer_load_dwordx4 v53, s[24:27], 0 offen lds
.Ltmp31:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	s_mov_b32 m0, s10
	s_add_i32 s2, s30, s17
	buffer_load_dwordx4 v54, s[12:15], 0 offen lds
.Ltmp32:
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[126:129], v[18:33]
.Ltmp33:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	s_mov_b32 m0, s11
.Ltmp34:
	.loc	1 213 25 is_stmt 0              ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_readfirstlane_b32 s11, v56
	v_add_lshl_u32 v48, s2, v48, 1
.Ltmp35:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	buffer_load_dwordx4 v55, s[12:15], 0 offen lds
.Ltmp36:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_mov_b32 m0, s11
	v_readfirstlane_b32 s11, v57
.Ltmp37:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[44:47], v39 offset:35200
.Ltmp38:
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[122:125], v[18:33]
	v_add_lshl_u32 v49, s2, v49, 1
.Ltmp39:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[40:43], v39 offset:35232
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
.Ltmp40:
	.loc	1 213 25 is_stmt 0              ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	buffer_load_dwordx4 v48, s[24:27], 0 offen lds
	s_mov_b32 m0, s11
.Ltmp41:
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[118:121], v[18:33]
.Ltmp42:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	buffer_load_dwordx4 v49, s[24:27], 0 offen lds
.Ltmp43:
	.loc	1 213 25 is_stmt 0              ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[66:69], v39 offset:51392
	ds_read_b128 v[186:189], v39 offset:51424
	ds_read_b128 v[182:185], v39 offset:51456
	ds_read_b128 v[178:181], v39 offset:51488
	ds_read_b128 v[174:177], v39 offset:51520
	ds_read_b128 v[106:109], v39 offset:51552
	ds_read_b128 v[102:105], v39 offset:51584
	ds_read_b128 v[98:101], v39 offset:51616
	ds_read_b128 v[110:113], v39 offset:51648
	ds_read_b128 v[170:173], v39 offset:51680
	ds_read_b128 v[166:169], v39 offset:51712
	ds_read_b128 v[162:165], v39 offset:51744
	ds_read_b128 v[158:161], v39 offset:51776
	ds_read_b128 v[154:157], v39 offset:51808
	ds_read_b128 v[150:153], v39 offset:51840
	ds_read_b128 v[146:149], v39 offset:51872
	s_mov_b32 s10, 0xff800000
	s_movk_i32 s2, 0x100
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_movk_i32 s2, 0xff
	v_cmp_lt_u32_e64 s[2:3], s2, v0
.Ltmp44:
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[18:33], v[40:43], v[114:117], v[18:33]
.Ltmp45:
	.file	2 "/app/OAI-triton/python/triton/language" "standard.py"
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max_f32_e32 v40, v3, v3
	v_max_f32_e32 v41, v2, v2
	v_max_f32_e32 v40, v41, v40
	v_max3_f32 v40, v40, v4, v5
	v_max3_f32 v40, v40, v6, v7
	v_max3_f32 v40, v40, v8, v9
	v_max3_f32 v40, v40, v10, v11
	v_max3_f32 v40, v40, v12, v13
	v_max3_f32 v40, v40, v14, v15
	v_max3_f32 v40, v40, v16, v17
	s_nop 1
	v_max3_f32 v40, v40, v18, v19
	v_max3_f32 v40, v40, v20, v21
	v_max3_f32 v40, v40, v22, v23
	v_max3_f32 v40, v40, v24, v25
	v_max3_f32 v40, v40, v26, v27
	v_max3_f32 v40, v40, v28, v29
	v_max3_f32 v40, v40, v30, v31
	v_max3_f32 v40, v40, v32, v33
.Ltmp46:
	.loc	2 189 40                        ; standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ]
	v_mov_b32_e32 v41, v40
	s_nop 1
	v_permlane32_swap_b32_e32 v40, v41
.Ltmp47:
	.loc	1 334 31                        ; flash-attention.py:334:31 @[ flash-attention.py:708:61 ]
	v_max3_f32 v213, v40, v41, s10
	s_mov_b32 s10, 0x3e0293ee
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mov_b32_e32 v212, v33
	v_mul_f32_e32 v2, 0x3e0293ee, v2
	v_mul_f32_e32 v3, 0x3e0293ee, v3
	v_mul_f32_e32 v4, 0x3e0293ee, v4
	v_mul_f32_e32 v5, 0x3e0293ee, v5
	v_mul_f32_e32 v6, 0x3e0293ee, v6
	v_mul_f32_e32 v7, 0x3e0293ee, v7
	v_mul_f32_e32 v8, 0x3e0293ee, v8
	v_mul_f32_e32 v9, 0x3e0293ee, v9
	v_mul_f32_e32 v10, 0x3e0293ee, v10
	v_mul_f32_e32 v11, 0x3e0293ee, v11
	v_mul_f32_e32 v12, 0x3e0293ee, v12
	v_mul_f32_e32 v13, 0x3e0293ee, v13
	v_mul_f32_e32 v14, 0x3e0293ee, v14
	v_mul_f32_e32 v15, 0x3e0293ee, v15
	v_mul_f32_e32 v16, 0x3e0293ee, v16
	v_mul_f32_e32 v17, 0x3e0293ee, v17
	v_mul_f32_e32 v18, 0x3e0293ee, v18
	v_mul_f32_e32 v19, 0x3e0293ee, v19
	v_mul_f32_e32 v20, 0x3e0293ee, v20
	v_mul_f32_e32 v21, 0x3e0293ee, v21
	v_mul_f32_e32 v22, 0x3e0293ee, v22
	v_mul_f32_e32 v23, 0x3e0293ee, v23
	v_mul_f32_e32 v24, 0x3e0293ee, v24
	v_mul_f32_e32 v25, 0x3e0293ee, v25
	v_mul_f32_e32 v26, 0x3e0293ee, v26
	v_mul_f32_e32 v27, 0x3e0293ee, v27
	v_mul_f32_e32 v28, 0x3e0293ee, v28
	v_mul_f32_e32 v29, 0x3e0293ee, v29
	v_mul_f32_e32 v30, 0x3e0293ee, v30
	v_mul_f32_e32 v31, 0x3e0293ee, v31
	v_mul_f32_e32 v32, 0x3e0293ee, v32
	v_pk_mul_f32 v[40:41], v[212:213], s[10:11] op_sel_hi:[1,0]
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v2, v2, v41
	v_sub_f32_e32 v3, v3, v41
	v_sub_f32_e32 v4, v4, v41
	v_sub_f32_e32 v5, v5, v41
	v_sub_f32_e32 v6, v6, v41
	v_sub_f32_e32 v7, v7, v41
	v_sub_f32_e32 v8, v8, v41
	v_sub_f32_e32 v9, v9, v41
	v_sub_f32_e32 v10, v10, v41
	v_sub_f32_e32 v11, v11, v41
	v_sub_f32_e32 v12, v12, v41
	v_sub_f32_e32 v13, v13, v41
	v_sub_f32_e32 v14, v14, v41
	v_sub_f32_e32 v15, v15, v41
	v_sub_f32_e32 v16, v16, v41
	v_sub_f32_e32 v17, v17, v41
	v_sub_f32_e32 v18, v18, v41
	v_sub_f32_e32 v19, v19, v41
	v_sub_f32_e32 v20, v20, v41
	v_sub_f32_e32 v21, v21, v41
	v_sub_f32_e32 v22, v22, v41
	v_sub_f32_e32 v23, v23, v41
	v_sub_f32_e32 v24, v24, v41
	v_sub_f32_e32 v25, v25, v41
	v_sub_f32_e32 v26, v26, v41
	v_sub_f32_e32 v27, v27, v41
	v_sub_f32_e32 v28, v28, v41
	v_sub_f32_e32 v29, v29, v41
	v_sub_f32_e32 v30, v30, v41
	v_sub_f32_e32 v31, v31, v41
	v_sub_f32_e32 v32, v32, v41
	v_sub_f32_e32 v33, v40, v41
	.loc	1 350 46                        ; flash-attention.py:350:46 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v39, 0xff800000, v41
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_barrier
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:                                ; %.preheader
	.loc	1 0 59 is_stmt 0                ; flash-attention.py:0:59
	s_or_b64 exec, exec, s[10:11]
	v_exp_f32_e32 v250, v2
	v_exp_f32_e32 v251, v3
	v_lshlrev_b32_e32 v2, 8, v0
	v_lshlrev_b32_e32 v3, 3, v0
	v_exp_f32_e32 v191, v4
	v_and_b32_e32 v2, 0xc00, v2
	v_and_b32_e32 v3, 24, v3
	v_lshlrev_b32_e32 v4, 7, v211
	v_or3_b32 v2, v2, v3, v4
	v_or_b32_e32 v201, v2, v37
	v_lshrrev_b32_e32 v3, 4, v2
	v_or_b32_e32 v2, 0x2000, v2
	v_lshrrev_b32_e32 v2, 4, v2
	v_and_b32_e32 v203, 0x3c0, v2
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	v_add_u32_e32 v2, v200, v36
	s_load_dwordx2 s[2:3], s[0:1], 0x4c
	s_load_dword s12, s[0:1], 0x54
	v_and_b32_e32 v202, 0x1c0, v3
	s_lshl_b32 s0, s34, 1
	v_add_u32_e32 v3, 0x88, v2
	s_lshl_b32 s1, s31, 1
	v_or_b32_e32 v2, 0x80, v2
	s_add_i32 s1, s1, s0
	v_mul_lo_u32 v2, s23, v2
	v_mul_lo_u32 v3, s23, v3
	v_lshl_add_u32 v206, v2, 1, s1
	v_add3_u32 v2, v37, v38, v36
	v_lshl_add_u32 v205, v3, 1, s1
	s_lshl_b32 s1, s33, 1
	v_add_u32_e32 v3, 0xc8, v2
	s_lshl_b32 s10, s28, 1
	v_add_u32_e32 v2, 0xc0, v2
	s_add_i32 s10, s10, s1
	v_mul_lo_u32 v2, s29, v2
	v_lshl_add_u32 v208, v2, 1, s10
	v_or_b32_e32 v2, 64, v35
	v_exp_f32_e32 v197, v5
	v_exp_f32_e32 v196, v6
	v_exp_f32_e32 v248, v7
	v_exp_f32_e32 v190, v8
	v_exp_f32_e32 v195, v9
	v_exp_f32_e32 v194, v10
	v_exp_f32_e32 v193, v11
	v_exp_f32_e32 v192, v12
	v_exp_f32_e32 v247, v13
	v_exp_f32_e32 v209, v14
	v_exp_f32_e32 v239, v15
	v_exp_f32_e32 v238, v16
	v_exp_f32_e32 v240, v17
	v_exp_f32_e32 v237, v18
	v_exp_f32_e32 v236, v19
	v_exp_f32_e32 v235, v20
	v_exp_f32_e32 v233, v21
	v_exp_f32_e32 v232, v22
	v_exp_f32_e32 v231, v23
	v_exp_f32_e32 v234, v24
	v_exp_f32_e32 v230, v25
	v_exp_f32_e32 v229, v26
	v_exp_f32_e32 v228, v27
	v_exp_f32_e32 v227, v28
	v_exp_f32_e32 v226, v29
	v_exp_f32_e32 v225, v30
	v_exp_f32_e32 v224, v31
	v_exp_f32_e32 v223, v32
	v_exp_f32_e32 v222, v33
	v_exp_f32_e32 v212, v39
	v_mul_lo_u32 v2, s23, v2
	v_add_u32_e32 v220, s30, v2
	v_or_b32_e32 v2, 64, v34
	v_mul_lo_u32 v3, s29, v3
	v_mul_lo_u32 v2, s23, v2
	v_mov_b32_e32 v16, 0
	v_mul_u32_u24_e32 v204, 0x410, v36
	v_mul_u32_u24_e32 v219, 0x440, v36
	s_lshl_b32 s0, s23, 7
	v_lshl_add_u32 v207, v3, 1, s10
	s_lshl_b32 s1, s29, 7
	s_add_i32 s10, 0, 0x4400
	s_add_i32 s14, 0, 0x87c0
	v_add_u32_e32 v221, s30, v2
	v_mov_b32_e32 v218, 1.0
	s_movk_i32 s13, 0xffc0
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	s_mov_b32 s15, 0
	v_mov_b32_e32 v17, v16
	v_mov_b32_e32 v14, v16
	v_mov_b32_e32 v15, v16
	v_mov_b32_e32 v12, v16
	v_mov_b32_e32 v13, v16
	v_mov_b32_e32 v10, v16
	v_mov_b32_e32 v11, v16
	v_mov_b32_e32 v8, v16
	v_mov_b32_e32 v9, v16
	v_mov_b32_e32 v6, v16
	v_mov_b32_e32 v7, v16
	v_mov_b32_e32 v4, v16
	v_mov_b32_e32 v5, v16
	v_mov_b32_e32 v2, v16
	v_mov_b32_e32 v3, v16
	v_mov_b32_e32 v48, v16
	v_mov_b32_e32 v49, v16
	v_mov_b32_e32 v46, v16
	v_mov_b32_e32 v47, v16
	v_mov_b32_e32 v44, v16
	v_mov_b32_e32 v45, v16
	v_mov_b32_e32 v42, v16
	v_mov_b32_e32 v43, v16
	v_mov_b32_e32 v40, v16
	v_mov_b32_e32 v41, v16
	v_mov_b32_e32 v38, v16
	v_mov_b32_e32 v39, v16
	v_mov_b32_e32 v36, v16
	v_mov_b32_e32 v37, v16
	v_mov_b32_e32 v34, v16
	v_mov_b32_e32 v35, v16
	v_mov_b32_e32 v32, v16
	v_mov_b32_e32 v33, v16
	v_mov_b32_e32 v30, v16
	v_mov_b32_e32 v31, v16
	v_mov_b32_e32 v28, v16
	v_mov_b32_e32 v29, v16
	v_mov_b32_e32 v26, v16
	v_mov_b32_e32 v27, v16
	v_mov_b32_e32 v24, v16
	v_mov_b32_e32 v25, v16
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v23, v16
	v_mov_b32_e32 v20, v16
	v_mov_b32_e32 v21, v16
	v_mov_b32_e32 v18, v16
	v_mov_b32_e32 v19, v16
	v_mov_b32_e32 v64, v16
	v_mov_b32_e32 v65, v16
	v_mov_b32_e32 v62, v16
	v_mov_b32_e32 v63, v16
	v_mov_b32_e32 v60, v16
	v_mov_b32_e32 v61, v16
	v_mov_b32_e32 v58, v16
	v_mov_b32_e32 v59, v16
	v_mov_b32_e32 v56, v16
	v_mov_b32_e32 v57, v16
	v_mov_b32_e32 v54, v16
	v_mov_b32_e32 v55, v16
	v_mov_b32_e32 v52, v16
	v_mov_b32_e32 v53, v16
	v_mov_b32_e32 v50, v16
	v_mov_b32_e32 v51, v16
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[142:145], 0
	s_mov_b32 s31, s22
	s_mov_b32 s22, s10
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_setprio 0
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[138:141], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[134:137], v[66:81]
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[130:133], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[110:113], v[142:145], 0
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v110, v237, v236
	v_cvt_pk_f16_f32 v111, v235, v233
	v_cvt_pk_f16_f32 v112, v232, v231
	v_cvt_pk_f16_f32 v113, v234, v230
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[126:129], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[138:141], v[82:97]
.Ltmp48:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v170, v250, v251
.Ltmp49:
	.loc	1 355 20                        ; flash-attention.py:355:20 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v171, v218, v212
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[122:125], v[66:81]
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v106, v229, v228
	v_cvt_pk_f16_f32 v107, v227, v226
	v_cvt_pk_f16_f32 v108, v225, v224
	v_cvt_pk_f16_f32 v109, v223, v222
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[134:137], v[82:97]
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v166, v250, v251
	v_cvt_pk_f16_f32 v167, v191, v197
	v_cvt_pk_f16_f32 v168, v196, v248
	v_cvt_pk_f16_f32 v169, v190, v195
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[118:121], v[66:81]
.Ltmp50:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v102, v170, v191
	v_add_f32_e32 v102, v102, v197
	v_add_f32_e32 v102, v102, v196
	v_add_f32_e32 v102, v102, v248
	v_add_f32_e32 v102, v102, v190
	v_add_f32_e32 v102, v102, v195
	v_add_f32_e32 v102, v102, v194
.Ltmp51:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[130:133], v[82:97]
.Ltmp52:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v102, v102, v193
	v_add_f32_e32 v102, v102, v192
	v_add_f32_e32 v102, v102, v247
	v_add_f32_e32 v102, v102, v209
	v_add_f32_e32 v102, v102, v239
	v_add_f32_e32 v102, v102, v238
	v_add_f32_e32 v102, v102, v240
.Ltmp53:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[158:161], v[126:129], v[82:97]
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v162, v194, v193
	v_cvt_pk_f16_f32 v163, v192, v247
	v_cvt_pk_f16_f32 v164, v209, v239
	v_cvt_pk_f16_f32 v165, v238, v240
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[66:81], v[98:101], v[114:117], v[66:81]
.Ltmp54:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v98, v102, v237
	v_add_f32_e32 v98, v98, v236
	v_add_f32_e32 v98, v98, v235
	v_add_f32_e32 v98, v98, v233
	v_add_f32_e32 v98, v98, v232
	v_add_f32_e32 v98, v98, v231
	v_add_f32_e32 v98, v98, v234
.Ltmp55:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[154:157], v[122:125], v[82:97]
.Ltmp56:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v98, v98, v230
	v_add_f32_e32 v98, v98, v229
	v_add_f32_e32 v98, v98, v228
	v_add_f32_e32 v98, v98, v227
	v_add_f32_e32 v98, v98, v226
	v_add_f32_e32 v98, v98, v225
	v_add_f32_e32 v98, v98, v224
.Ltmp57:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[150:153], v[118:121], v[82:97]
.Ltmp58:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v98, v98, v223
	v_add_f32_e32 v98, v98, v222
.Ltmp59:
	.loc	2 291 36                        ; standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ]
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
.Ltmp60:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v98, v98, v99
.Ltmp61:
	.loc	1 355 28                        ; flash-attention.py:355:28 @[ flash-attention.py:708:61 ]
	v_add_f32_e32 v218, v171, v98
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[146:149], v[114:117], v[82:97]
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_setprio 1
.Ltmp62:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
.Ltmp63:
	; sched_barrier mask(0x00000000)
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_add_i32 s10, s15, 1
	s_cmp_lt_i32 s10, 2
	s_cselect_b32 s29, s10, 0
.Ltmp64:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	s_lshl_b32 s23, s29, 13
	s_lshl_b32 s10, s29, 14
	s_add_i32 s28, s10, 0
	s_ashr_i32 s10, s23, 5
	s_add_i32 s11, s28, s10
	s_add_i32 s30, s11, 0x87c0
	v_add_u32_e32 v98, s30, v204
	v_add_u32_e32 v99, v214, v208
	v_readfirstlane_b32 s10, v98
	v_add_u32_e32 v98, s30, v199
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v98
	buffer_load_dwordx4 v99, s[4:7], 0 offen lds
	v_add_u32_e32 v99, v214, v207
	s_mov_b32 m0, s10
.Ltmp65:
	.loc	1 213 25 is_stmt 0              ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v98, s31, v201
.Ltmp66:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	buffer_load_dwordx4 v99, s[4:7], 0 offen lds
.Ltmp67:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v99, v98, v202
	v_add_u32_e32 v100, v98, v203
	ds_read_b64_tr_b16 v[222:223], v99
	ds_read_b64_tr_b16 v[226:227], v99 offset:64
	ds_read_b64_tr_b16 v[194:195], v99 offset:128
	ds_read_b64_tr_b16 v[190:191], v99 offset:192
	ds_read_b64_tr_b16 v[224:225], v100 offset:8192
	ds_read_b64_tr_b16 v[228:229], v100 offset:8256
	ds_read_b64_tr_b16 v[196:197], v100 offset:8320
	ds_read_b64_tr_b16 v[192:193], v100 offset:8384
	ds_read_b64_tr_b16 v[230:231], v99 offset:256
	ds_read_b64_tr_b16 v[182:183], v99 offset:320
	ds_read_b64_tr_b16 v[178:179], v99 offset:384
	ds_read_b64_tr_b16 v[170:171], v99 offset:448
	ds_read_b64_tr_b16 v[232:233], v100 offset:8448
	ds_read_b64_tr_b16 v[184:185], v100 offset:8512
	ds_read_b64_tr_b16 v[180:181], v100 offset:8576
	ds_read_b64_tr_b16 v[172:173], v100 offset:8640
	ds_read_b64_tr_b16 v[186:187], v99 offset:512
	ds_read_b64_tr_b16 v[158:159], v99 offset:576
	ds_read_b64_tr_b16 v[154:155], v99 offset:640
	ds_read_b64_tr_b16 v[150:151], v99 offset:704
	ds_read_b64_tr_b16 v[188:189], v100 offset:8704
	ds_read_b64_tr_b16 v[160:161], v100 offset:8768
	ds_read_b64_tr_b16 v[156:157], v100 offset:8832
	ds_read_b64_tr_b16 v[152:153], v100 offset:8896
	ds_read_b64_tr_b16 v[174:175], v99 offset:768
	ds_read_b64_tr_b16 v[146:147], v99 offset:832
	ds_read_b64_tr_b16 v[102:103], v99 offset:896
	ds_read_b64_tr_b16 v[98:99], v99 offset:960
	ds_read_b64_tr_b16 v[176:177], v100 offset:8960
	ds_read_b64_tr_b16 v[148:149], v100 offset:9024
	ds_read_b64_tr_b16 v[104:105], v100 offset:9088
	ds_read_b64_tr_b16 v[100:101], v100 offset:9152
.Ltmp68:
	; sched_barrier mask(0x00000000)
	.loc	1 370 51 is_stmt 1              ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[64:65], v[64:65], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[212:213] op_sel_hi:[1,0]
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_barrier
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[222:225], v[166:169], v[50:65]
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_setprio 0
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v16, v16, v212
	v_mul_f32_e64 v17, v17, v212
	v_mul_f32_e64 v14, v14, v212
	v_mul_f32_e64 v15, v15, v212
	v_pk_mul_f32 v[12:13], v[12:13], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[212:213] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[2:17], v[226:229], v[166:169], v[2:17]
	v_mfma_f32_32x32x16_f16 v[34:49], v[194:197], v[166:169], v[34:49]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v194, 0x3e0293ee, v86
	v_mul_f32_e32 v195, 0x3e0293ee, v87
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[18:33], v[190:193], v[166:169], v[18:33]
.Ltmp69:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max_f32_e32 v166, v67, v67
	v_max_f32_e32 v167, v66, v66
	v_max_f32_e32 v166, v167, v166
.Ltmp70:
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v168, 0x3e0293ee, v68
.Ltmp71:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v68, v166, v68, v69
	v_max3_f32 v68, v68, v70, v71
	v_max3_f32 v68, v68, v72, v73
.Ltmp72:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[230:233], v[162:165], v[50:65]
.Ltmp73:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v68, v68, v74, v75
	v_max3_f32 v68, v68, v76, v77
	v_max3_f32 v68, v68, v78, v79
	v_max3_f32 v68, v68, v80, v81
	v_max3_f32 v68, v68, v82, v83
	v_max3_f32 v68, v68, v84, v85
	v_max3_f32 v68, v68, v86, v87
.Ltmp74:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[2:17], v[182:185], v[162:165], v[2:17]
.Ltmp75:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v68, v68, v88, v89
	v_max3_f32 v68, v68, v90, v91
	v_max3_f32 v68, v68, v92, v93
	v_max3_f32 v68, v68, v94, v95
	v_max3_f32 v68, v68, v96, v97
.Ltmp76:
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v169, 0x3e0293ee, v69
.Ltmp77:
	.loc	2 189 40                        ; standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ]
	v_mov_b32_e32 v69, v68
.Ltmp78:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[178:181], v[162:165], v[34:49]
.Ltmp79:
	.loc	2 189 40                        ; standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ]
	s_nop 0
	v_permlane32_swap_b32_e32 v68, v69
.Ltmp80:
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v66, 0x3e0293ee, v66
	v_mul_f32_e32 v67, 0x3e0293ee, v67
	v_mul_f32_e32 v182, 0x3e0293ee, v74
	v_mul_f32_e32 v183, 0x3e0293ee, v75
	v_mul_f32_e32 v184, 0x3e0293ee, v76
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[162:165], v[18:33]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v185, 0x3e0293ee, v77
	v_mul_f32_e32 v190, 0x3e0293ee, v78
	v_mul_f32_e32 v191, 0x3e0293ee, v79
	v_mul_f32_e32 v192, 0x3e0293ee, v80
	v_mul_f32_e32 v178, 0x3e0293ee, v81
	v_mul_f32_e32 v179, 0x3e0293ee, v82
	v_mul_f32_e32 v180, 0x3e0293ee, v83
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[50:65], v[186:189], v[110:113], v[50:65]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v186, 0x3e0293ee, v70
	v_mul_f32_e32 v187, 0x3e0293ee, v71
	v_mul_f32_e32 v188, 0x3e0293ee, v72
	v_mul_f32_e32 v189, 0x3e0293ee, v73
	v_mul_f32_e32 v181, 0x3e0293ee, v84
	v_mul_f32_e32 v193, 0x3e0293ee, v85
	v_mul_f32_e32 v162, 0x3e0293ee, v88
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[110:113], v[2:17]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v163, 0x3e0293ee, v89
	v_mul_f32_e32 v164, 0x3e0293ee, v90
	v_mul_f32_e32 v165, 0x3e0293ee, v91
	v_mul_f32_e32 v170, 0x3e0293ee, v92
	v_mul_f32_e32 v171, 0x3e0293ee, v93
	v_mul_f32_e32 v172, 0x3e0293ee, v94
	v_mul_f32_e32 v173, 0x3e0293ee, v95
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[34:49], v[154:157], v[110:113], v[34:49]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[18:33], v[150:153], v[110:113], v[18:33]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[50:65], v[174:177], v[106:109], v[50:65]
	.loc	1 350 35                        ; flash-attention.py:350:35 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v176, 0x3e0293ee, v213
	.loc	1 334 31                        ; flash-attention.py:334:31 @[ flash-attention.py:708:61 ]
	v_max3_f32 v213, v213, v68, v69
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v174, 0x3e0293ee, v96
	v_mul_f32_e32 v175, 0x3e0293ee, v97
	.loc	1 335 29                        ; flash-attention.py:335:29 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v249, 0x3e0293ee, v213
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v66, v66, v249
	v_sub_f32_e32 v67, v67, v249
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[2:17], v[146:149], v[106:109], v[2:17]
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v68, v168, v249
	v_sub_f32_e32 v69, v169, v249
	v_sub_f32_e32 v70, v186, v249
	v_sub_f32_e32 v71, v187, v249
	v_sub_f32_e32 v72, v188, v249
	v_sub_f32_e32 v73, v189, v249
	v_sub_f32_e32 v74, v182, v249
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[34:49], v[102:105], v[106:109], v[34:49]
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v75, v183, v249
	v_sub_f32_e32 v76, v184, v249
	v_sub_f32_e32 v77, v185, v249
	v_sub_f32_e32 v78, v190, v249
	v_sub_f32_e32 v79, v191, v249
	v_sub_f32_e32 v80, v192, v249
	v_sub_f32_e32 v81, v178, v249
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[98:101], v[106:109], v[18:33]
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v82, v179, v249
	v_sub_f32_e32 v83, v180, v249
	v_sub_f32_e32 v84, v181, v249
	v_sub_f32_e32 v85, v193, v249
	v_sub_f32_e32 v86, v194, v249
	v_sub_f32_e32 v87, v195, v249
	v_sub_f32_e32 v88, v162, v249
	v_sub_f32_e32 v89, v163, v249
	v_sub_f32_e32 v90, v164, v249
	v_sub_f32_e32 v91, v165, v249
	v_sub_f32_e32 v92, v170, v249
	v_sub_f32_e32 v93, v171, v249
	v_sub_f32_e32 v94, v172, v249
	v_sub_f32_e32 v95, v173, v249
	v_sub_f32_e32 v96, v174, v249
	v_sub_f32_e32 v97, v175, v249
	.loc	1 350 46                        ; flash-attention.py:350:46 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v110, v176, v249
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v250, v66
	v_exp_f32_e32 v251, v67
	v_exp_f32_e32 v191, v68
	v_exp_f32_e32 v197, v69
	v_exp_f32_e32 v196, v70
	v_exp_f32_e32 v248, v71
	v_exp_f32_e32 v190, v72
	v_exp_f32_e32 v195, v73
	v_exp_f32_e32 v194, v74
	v_exp_f32_e32 v193, v75
	v_exp_f32_e32 v192, v76
	v_exp_f32_e32 v247, v77
	v_exp_f32_e32 v209, v78
	v_exp_f32_e32 v239, v79
	v_exp_f32_e32 v238, v80
	v_exp_f32_e32 v240, v81
	v_exp_f32_e32 v237, v82
	v_exp_f32_e32 v236, v83
	v_exp_f32_e32 v235, v84
	v_exp_f32_e32 v233, v85
	v_exp_f32_e32 v232, v86
	v_exp_f32_e32 v231, v87
	v_exp_f32_e32 v234, v88
	v_exp_f32_e32 v230, v89
	v_exp_f32_e32 v229, v90
	v_exp_f32_e32 v228, v91
	v_exp_f32_e32 v227, v92
	v_exp_f32_e32 v226, v93
	v_exp_f32_e32 v225, v94
	v_exp_f32_e32 v224, v95
	v_exp_f32_e32 v223, v96
	v_exp_f32_e32 v222, v97
	.loc	1 350 29                        ; flash-attention.py:350:29 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v212, v110
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_setprio 1
.Ltmp81:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
.Ltmp82:
	; sched_barrier mask(0x00000000)
	.loc	1 213 25 is_stmt 0              ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_lshl_b32 s10, s15, 13
	s_lshl_b32 s15, s15, 14
	s_add_i32 s15, s15, 0
	s_ashr_i32 s10, s10, 3
	s_add_i32 s10, s15, s10
	v_add_u32_e32 v66, s10, v219
	v_add_u32_e32 v67, v214, v206
	v_readfirstlane_b32 s15, v66
	v_add_u32_e32 v66, s10, v216
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v66
	buffer_load_dwordx4 v67, s[24:27], 0 offen lds
	v_add_u32_e32 v67, v214, v205
	s_mov_b32 m0, s15
.Ltmp83:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v70, s14, v217
.Ltmp84:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	buffer_load_dwordx4 v67, s[24:27], 0 offen lds
.Ltmp85:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[186:189], v70 offset:32
	ds_read_b128 v[182:185], v70 offset:64
	ds_read_b128 v[178:181], v70 offset:96
	ds_read_b128 v[174:177], v70 offset:128
	ds_read_b128 v[106:109], v70 offset:160
	ds_read_b128 v[102:105], v70 offset:192
	ds_read_b128 v[98:101], v70 offset:224
	ds_read_b128 v[110:113], v70 offset:256
	ds_read_b128 v[170:173], v70 offset:288
	ds_read_b128 v[166:169], v70 offset:320
	ds_read_b128 v[162:165], v70 offset:352
	ds_read_b128 v[158:161], v70 offset:384
	ds_read_b128 v[154:157], v70 offset:416
	ds_read_b128 v[150:153], v70 offset:448
	ds_read_b128 v[146:149], v70 offset:480
.Ltmp86:
	; sched_barrier mask(0x00000000)
	.loc	1 278 59 is_stmt 1              ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	s_add_i32 s13, s13, 64
	v_add_u32_e32 v205, s0, v205
	v_add_u32_e32 v206, s0, v206
	v_add_u32_e32 v207, s1, v207
	v_add_u32_e32 v208, s1, v208
	v_add_u32_e32 v220, s17, v220
	v_add_u32_e32 v221, s17, v221
	s_cmpk_lt_u32 s13, 0x3f00
	s_mov_b32 s14, s30
	s_mov_b32 s15, s29
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	.loc	1 0 59 is_stmt 0                ; flash-attention.py:0:59
	s_or_b64 exec, exec, s[0:1]
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[142:145], 0
.Ltmp87:
	.loc	1 523 50                        ; flash-attention.py:523:50
	v_lshrrev_b32_e32 v66, 1, v198
	v_and_b32_e32 v242, 0xff, v0
.Ltmp88:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_ashr_i32 s1, s23, 3
.Ltmp89:
	.loc	1 523 50                        ; flash-attention.py:523:50
	v_or_b32_e32 v241, v66, v1
	.loc	1 765 44                        ; flash-attention.py:765:44
	s_add_i32 s0, s16, 0xffffc100
	.loc	1 762 84                        ; flash-attention.py:762:84
	v_lshl_or_b32 v66, s18, 20, v242
	s_add_i32 s13, 0, 0x10a00
.Ltmp90:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[138:141], v[82:97]
.Ltmp91:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_add_i32 s5, s28, s1
	v_and_b32_e32 v0, 0x100, v0
.Ltmp92:
	.loc	1 766 35                        ; flash-attention.py:766:35
	s_cmp_lt_i32 s0, 1
	v_lshlrev_b32_e32 v244, 2, v1
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	v_add_lshl_u32 v0, v66, s16, 2
.Ltmp93:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v1, v250, v251
.Ltmp94:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[134:137], v[82:97]
	v_lshl_add_u32 v243, s19, 16, v0
.Ltmp95:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v0, v1, v191
	v_add_f32_e32 v0, v0, v197
	v_add_f32_e32 v0, v0, v196
	v_add_f32_e32 v0, v0, v248
	v_add_f32_e32 v0, v0, v190
	v_add_f32_e32 v0, v0, v195
.Ltmp96:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[130:133], v[82:97]
.Ltmp97:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v0, v0, v194
	v_add_f32_e32 v0, v0, v193
	v_add_f32_e32 v0, v0, v192
	v_or_b32_e32 v246, v202, v201
	v_add_f32_e32 v0, v0, v247
.Ltmp98:
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v186, v250, v251
	v_cvt_pk_f16_f32 v187, v191, v197
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[126:129], v[82:97]
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v188, v196, v248
.Ltmp99:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v250, v0, v209
.Ltmp100:
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v189, v190, v195
	v_cvt_pk_f16_f32 v183, v192, v247
.Ltmp101:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v0, s22, v246
	v_add_u32_e32 v247, v203, v201
	v_add_u32_e32 v248, v201, v202
.Ltmp102:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[122:125], v[82:97]
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v80, v64, v212
	v_mul_f32_e64 v81, v65, v212
	v_mul_f32_e64 v78, v62, v212
	v_mul_f32_e64 v79, v63, v212
	v_mul_f32_e64 v76, v60, v212
	v_mul_f32_e64 v77, v61, v212
	v_pk_mul_f32 v[74:75], v[58:59], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[56:57], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[54:55], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[52:53], v[212:213] op_sel_hi:[1,0]
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[118:121], v[82:97]
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v66, v50, v212
	v_mul_f32_e64 v67, v51, v212
	v_lshlrev_b32_e32 v252, 2, v198
	v_lshlrev_b32_e32 v245, 1, v198
	.loc	1 370 31 is_stmt 0              ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v184, v209, v239
.Ltmp103:
	.loc	1 213 25 is_stmt 1              ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
.Ltmp104:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[114:117], v[82:97]
.Ltmp105:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v251, s22, v247
	v_add_u32_e32 v254, s22, v248
.Ltmp106:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v62, v14, v212
	v_mul_f32_e64 v63, v15, v212
	v_mul_f32_e64 v14, v32, v212
	v_mul_f32_e64 v15, v33, v212
	.loc	1 370 31 is_stmt 0              ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v182, v194, v193
	v_lshl_add_u32 v253, v200, 8, s13
	v_cvt_pk_f16_f32 v185, v238, v240
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[98:113], v[110:113], v[142:145], 0
.Ltmp107:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	s_nop 1
	v_max_f32_e32 v32, v83, v83
	v_max_f32_e32 v33, v82, v82
	v_max_f32_e32 v32, v33, v32
.Ltmp108:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v64, v16, v212
	v_mul_f32_e64 v65, v17, v212
	v_pk_mul_f32 v[60:61], v[12:13], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[10:11], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[8:9], v[212:213] op_sel_hi:[1,0]
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[98:113], v[170:173], v[138:141], v[98:113]
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v54, v6, v212
	v_mul_f32_e64 v55, v7, v212
	v_mul_f32_e64 v52, v4, v212
	v_mul_f32_e64 v53, v5, v212
	v_mul_f32_e64 v50, v2, v212
	v_mul_f32_e64 v51, v3, v212
	v_pk_mul_f32 v[12:13], v[30:31], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[28:29], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[26:27], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[24:25], v[212:213] op_sel_hi:[1,0]
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[98:113], v[166:169], v[134:137], v[98:113]
.Ltmp109:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[206:207], v0
	ds_read_b64_tr_b16 v[208:209], v251 offset:8192
	ds_read_b64_tr_b16 v[196:197], v251 offset:8256
	ds_read_b64_tr_b16 v[198:199], v0 offset:512
	ds_read_b64_tr_b16 v[202:203], v254 offset:256
	ds_read_b64_tr_b16 v[190:191], v254 offset:320
	ds_read_b64_tr_b16 v[170:171], v254 offset:384
	ds_read_b64_tr_b16 v[166:167], v254 offset:192
.Ltmp110:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[4:5], v[22:23], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[20:21], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[18:19], v[212:213] op_sel_hi:[1,0]
.Ltmp111:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v32, v32, v84, v85
	v_max3_f32 v32, v32, v86, v87
.Ltmp112:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[48:49], v[48:49], v[212:213] op_sel_hi:[1,0]
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[98:113], v[162:165], v[130:133], v[98:113]
.Ltmp113:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[204:205], v251 offset:8448
	ds_read_b64_tr_b16 v[192:193], v251 offset:8512
	ds_read_b64_tr_b16 v[162:163], v251 offset:8320
	ds_read_b64_tr_b16 v[168:169], v251 offset:8384
	ds_read_b64_tr_b16 v[16:17], v254 offset:576
	ds_read_b64_tr_b16 v[20:21], v254 offset:640
	ds_read_b64_tr_b16 v[24:25], v254 offset:704
	ds_read_b64_tr_b16 v[28:29], v254 offset:448
.Ltmp114:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[46:47], v[46:47], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[212:213] op_sel_hi:[1,0]
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[98:113], v[158:161], v[126:129], v[98:113]
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v36, v36, v212
	v_mul_f32_e64 v37, v37, v212
	v_mul_f32_e64 v34, v34, v212
	v_mul_f32_e64 v35, v35, v212
.Ltmp115:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v32, v32, v88, v89
	v_max3_f32 v32, v32, v90, v91
	v_max3_f32 v32, v32, v92, v93
	v_max3_f32 v32, v32, v94, v95
	v_max3_f32 v32, v32, v96, v97
.Ltmp116:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[98:113], v[154:157], v[122:125], v[98:113]
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v178, v237, v236
	v_cvt_pk_f16_f32 v179, v235, v233
	v_cvt_pk_f16_f32 v180, v232, v231
	v_cvt_pk_f16_f32 v181, v234, v230
	v_cvt_pk_f16_f32 v174, v229, v228
	v_cvt_pk_f16_f32 v175, v227, v226
	v_cvt_pk_f16_f32 v176, v225, v224
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[98:113], v[150:153], v[118:121], v[98:113]
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v177, v223, v222
	s_mov_b32 s4, 0x3e0293ee
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v92, 0x3e0293ee, v92
	v_mul_f32_e32 v93, 0x3e0293ee, v93
	v_mul_f32_e32 v94, 0x3e0293ee, v94
	v_mul_f32_e32 v95, 0x3e0293ee, v95
	v_mul_f32_e32 v96, 0x3e0293ee, v96
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[66:81], v[206:209], v[186:189], v[66:81]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v82, 0x3e0293ee, v82
	v_mul_f32_e32 v83, 0x3e0293ee, v83
	v_mul_f32_e32 v84, 0x3e0293ee, v84
	v_mul_f32_e32 v85, 0x3e0293ee, v85
	v_mul_f32_e32 v86, 0x3e0293ee, v86
	v_mul_f32_e32 v87, 0x3e0293ee, v87
	v_mul_f32_e32 v88, 0x3e0293ee, v88
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[98:113], v[146:149], v[114:117], v[98:113]
.Ltmp117:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[200:201], v251 offset:8704
	ds_read_b64_tr_b16 v[18:19], v251 offset:8768
	ds_read_b64_tr_b16 v[172:173], v251 offset:8576
	ds_read_b64_tr_b16 v[30:31], v251 offset:8640
	ds_read_b64_tr_b16 v[148:149], v251 offset:8960
	ds_read_b64_tr_b16 v[152:153], v251 offset:9024
	ds_read_b64_tr_b16 v[22:23], v251 offset:8832
	ds_read_b64_tr_b16 v[26:27], v251 offset:8896
	ds_read_b64_tr_b16 v[146:147], v254 offset:768
	ds_read_b64_tr_b16 v[150:151], v254 offset:832
	ds_read_b64_tr_b16 v[154:155], v254 offset:896
	ds_read_b64_tr_b16 v[206:207], v254 offset:960
	ds_read_b64_tr_b16 v[194:195], v254 offset:64
	ds_read_b64_tr_b16 v[160:161], v254 offset:128
	ds_read_b64_tr_b16 v[156:157], v251 offset:9088
	ds_read_b64_tr_b16 v[208:209], v251 offset:9152
.Ltmp118:
	.loc	1 213 25 is_stmt 0              ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
.Ltmp119:
	.loc	1 336 18 is_stmt 1              ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v89, 0x3e0293ee, v89
	v_mul_f32_e32 v90, 0x3e0293ee, v90
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[66:81], v[202:205], v[182:185], v[66:81]
.Ltmp120:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v32, v32, v98, v99
	v_max3_f32 v32, v32, v100, v101
	v_max3_f32 v32, v32, v102, v103
	v_max3_f32 v32, v32, v104, v105
	v_max3_f32 v32, v32, v106, v107
	v_max3_f32 v32, v32, v108, v109
	v_max3_f32 v32, v32, v110, v111
.Ltmp121:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[194:197], v[186:189], v[50:65]
.Ltmp122:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v32, v32, v112, v113
.Ltmp123:
	.loc	2 189 40                        ; standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ]
	v_mov_b32_e32 v33, v32
	s_nop 1
	v_permlane32_swap_b32_e32 v32, v33
.Ltmp124:
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v91, 0x3e0293ee, v91
.Ltmp125:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add3_u32 v204, v253, v252, v214
	s_mov_b32 s26, s6
.Ltmp126:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[160:163], v[186:189], v[34:49]
.Ltmp127:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_mov_b32 s27, s7
.Ltmp128:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[0:15], v[166:169], v[186:189], v[0:15]
	v_mfma_f32_32x32x16_f16 v[66:81], v[198:201], v[178:181], v[66:81]
	v_mfma_f32_32x32x16_f16 v[50:65], v[190:193], v[182:185], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[170:173], v[182:185], v[34:49]
	v_mfma_f32_32x32x16_f16 v[0:15], v[28:31], v[182:185], v[0:15]
	v_mfma_f32_32x32x16_f16 v[66:81], v[146:149], v[174:177], v[66:81]
	.loc	1 334 31                        ; flash-attention.py:334:31 @[ flash-attention.py:708:61 ]
	v_max3_f32 v147, v213, v32, v33
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mov_b32_e32 v146, v113
	v_mul_f32_e64 v32, v146, s4
	v_mul_f32_e64 v33, v147, s4
	.loc	1 336 29 is_stmt 0              ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v82, v82, v33
	v_sub_f32_e32 v83, v83, v33
	v_sub_f32_e32 v84, v84, v33
	.loc	1 370 51 is_stmt 1              ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[16:19], v[178:181], v[50:65]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v16, 0x3e0293ee, v97
	v_mul_f32_e32 v17, 0x3e0293ee, v98
	v_mul_f32_e32 v18, 0x3e0293ee, v99
	v_mul_f32_e32 v19, 0x3e0293ee, v100
	v_mul_f32_e32 v98, 0x3e0293ee, v102
	v_mul_f32_e32 v102, 0x3e0293ee, v106
	v_mul_f32_e32 v106, 0x3e0293ee, v110
.Ltmp129:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v110, s11, v217
.Ltmp130:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[20:23], v[178:181], v[34:49]
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v20, v92, v33
	v_sub_f32_e32 v21, v93, v33
	v_sub_f32_e32 v22, v94, v33
	v_sub_f32_e32 v23, v95, v33
	v_sub_f32_e32 v92, v96, v33
	v_sub_f32_e32 v93, v16, v33
	v_sub_f32_e32 v94, v17, v33
	v_sub_f32_e32 v95, v18, v33
	v_sub_f32_e32 v96, v19, v33
.Ltmp131:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[16:19], v110 offset:34752
.Ltmp132:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[0:15], v[24:27], v[178:181], v[0:15]
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v85, v85, v33
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v166, v82
	v_exp_f32_e32 v169, v83
	v_exp_f32_e32 v170, v84
	v_exp_f32_e32 v171, v85
.Ltmp133:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[82:85], v110 offset:34784
.Ltmp134:
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v164, v22
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[150:153], v[174:177], v[50:65]
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v165, v23
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v86, v86, v33
	v_sub_f32_e32 v87, v87, v33
	v_sub_f32_e32 v88, v88, v33
	v_sub_f32_e32 v89, v89, v33
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v172, v86
	v_exp_f32_e32 v173, v87
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[154:157], v[174:177], v[34:49]
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v167, v88
	v_exp_f32_e32 v168, v89
.Ltmp135:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[86:89], v110 offset:34816
.Ltmp136:
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v97, 0x3e0293ee, v101
	v_mul_f32_e32 v99, 0x3e0293ee, v103
	v_mul_f32_e32 v100, 0x3e0293ee, v104
	v_mul_f32_e32 v101, 0x3e0293ee, v105
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[0:15], v[206:209], v[174:177], v[0:15]
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v176, v20
	v_exp_f32_e32 v177, v21
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v103, 0x3e0293ee, v107
	v_mul_f32_e32 v104, 0x3e0293ee, v108
	v_mul_f32_e32 v105, 0x3e0293ee, v109
	.loc	1 336 29 is_stmt 0              ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v90, v90, v33
	v_sub_f32_e32 v91, v91, v33
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[16:31], v[16:19], v[142:145], 0
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v97, v97, v33
	v_sub_f32_e32 v98, v98, v33
	v_sub_f32_e32 v99, v99, v33
	v_sub_f32_e32 v100, v100, v33
	v_sub_f32_e32 v101, v101, v33
	v_sub_f32_e32 v102, v102, v33
	v_sub_f32_e32 v103, v103, v33
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[16:31], v[82:85], v[138:141], v[16:31]
.Ltmp137:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[82:85], v110 offset:34848
.Ltmp138:
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v104, v104, v33
	v_sub_f32_e32 v105, v105, v33
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v174, v90
	v_exp_f32_e32 v175, v91
	v_exp_f32_e32 v178, v92
	v_exp_f32_e32 v154, v93
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[16:31], v[86:89], v[134:137], v[16:31]
.Ltmp139:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[86:89], v110 offset:34880
.Ltmp140:
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v155, v94
	v_exp_f32_e32 v156, v95
	v_exp_f32_e32 v157, v96
	v_exp_f32_e32 v158, v97
	v_exp_f32_e32 v163, v98
	v_exp_f32_e32 v159, v99
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[16:31], v[82:85], v[130:133], v[16:31]
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v160, v100
	v_exp_f32_e32 v161, v101
	v_exp_f32_e32 v162, v102
	v_exp_f32_e32 v146, v103
	v_exp_f32_e32 v148, v104
	v_exp_f32_e32 v149, v105
.Ltmp141:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[82:85], v110 offset:34912
	ds_read_b128 v[90:93], v110 offset:34944
	ds_read_b128 v[94:97], v110 offset:34976
	ds_read_b128 v[98:101], v110 offset:35008
	ds_read_b128 v[102:105], v110 offset:35040
.Ltmp142:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[16:31], v[86:89], v[126:129], v[16:31]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v107, 0x3e0293ee, v111
	v_mul_f32_e32 v108, 0x3e0293ee, v112
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	v_add3_u32 v86, v210, v221, s17
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v106, v106, v33
	v_sub_f32_e32 v107, v107, v33
	v_sub_f32_e32 v108, v108, v33
	.loc	1 373 18                        ; flash-attention.py:373:18 @[ flash-attention.py:708:61 ]
	v_add_u32_e32 v87, 1, v86
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[16:31], v[82:85], v[122:125], v[16:31]
	.loc	1 278 59                        ; flash-attention.py:278:59 @[ flash-attention.py:708:61 ]
	v_add3_u32 v82, v210, v220, s17
	.loc	1 373 18                        ; flash-attention.py:373:18 @[ flash-attention.py:708:61 ]
	v_add_u32_e32 v88, 2, v86
	v_add_u32_e32 v89, 3, v86
	v_add_u32_e32 v83, 1, v82
	v_add_u32_e32 v84, 2, v82
	v_add_u32_e32 v85, 3, v82
.Ltmp143:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v179, s13, v215
.Ltmp144:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[16:31], v[90:93], v[118:121], v[16:31]
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v150, v106
	v_exp_f32_e32 v151, v107
	v_exp_f32_e32 v152, v108
.Ltmp145:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:286:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b128 v[106:109], v110 offset:35072
	ds_read_b128 v[180:183], v110 offset:35104
	ds_read_b128 v[184:187], v110 offset:35136
	ds_read_b128 v[188:191], v110 offset:35168
	ds_read_b128 v[192:195], v110 offset:35200
	ds_read_b128 v[196:199], v110 offset:35232
.Ltmp146:
	.loc	1 373 18                        ; flash-attention.py:373:18 @[ flash-attention.py:708:61 ]
	v_add_u32_e32 v110, 4, v86
	v_add_u32_e32 v111, 5, v86
	v_add_u32_e32 v112, 6, v86
	v_add_u32_e32 v113, 7, v86
	v_add_u32_e32 v200, 4, v82
	v_add_u32_e32 v201, 5, v82
	v_add_u32_e32 v202, 6, v82
	v_add_u32_e32 v203, 7, v82
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[16:31], v[94:97], v[114:117], v[16:31]
.Ltmp147:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_write_b128 v179, v[86:89]
	ds_write_b128 v179, v[82:85] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp148:
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v32, v32, v33
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v153, v32
	.loc	1 350 46                        ; flash-attention.py:350:46 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v32, v249, v33
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[98:101], v[142:145], 0
.Ltmp149:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read2st64_b32 v[98:99], v204 offset1:8
	v_add_u32_e32 v100, s5, v219
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s6, v100
	s_mov_b32 m0, s6
	v_lshlrev_b32_e32 v98, 1, v98
	s_barrier
.Ltmp150:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[102:105], v[138:141], v[82:97]
.Ltmp151:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_write_b128 v179, v[110:113]
	ds_write_b128 v179, v[200:203] offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v98, s[24:27], 0 offen lds
	v_add_u32_e32 v98, s5, v216
.Ltmp152:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[106:109], v[134:137], v[82:97]
	.loc	1 350 29                        ; flash-attention.py:350:29 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v32, v32
.Ltmp153:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_readfirstlane_b32 s6, v98
	v_lshlrev_b32_e32 v99, 1, v99
	s_mov_b32 m0, s6
	v_add_u32_e32 v144, s10, v248
	buffer_load_dwordx4 v99, s[24:27], 0 offen lds
	s_waitcnt vmcnt(2)
.Ltmp154:
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[180:183], v[130:133], v[82:97]
.Ltmp155:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v140, s10, v247
.Ltmp156:
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v110, v166, v169
	v_cvt_pk_f16_f32 v111, v170, v171
	v_cvt_pk_f16_f32 v112, v172, v173
	v_cvt_pk_f16_f32 v113, v167, v168
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[184:187], v[126:129], v[82:97]
.Ltmp157:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v128, s10, v246
.Ltmp158:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v80, v80, v32
	v_mul_f32_e64 v81, v81, v32
	v_mul_f32_e64 v78, v78, v32
	v_mul_f32_e64 v79, v79, v32
	v_pk_mul_f32 v[76:77], v[76:77], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[32:33] op_sel_hi:[1,0]
	.loc	1 315 28                        ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[188:191], v[122:125], v[82:97]
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mul_f32_e64 v68, v68, v32
	v_mul_f32_e64 v69, v69, v32
	v_mul_f32_e64 v66, v66, v32
	v_mul_f32_e64 v67, v67, v32
	.loc	1 370 31 is_stmt 0              ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v106, v174, v175
	v_cvt_pk_f16_f32 v107, v176, v177
	v_cvt_pk_f16_f32 v108, v164, v165
	v_cvt_pk_f16_f32 v109, v178, v154
	v_cvt_pk_f16_f32 v102, v155, v156
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[192:195], v[118:121], v[82:97]
.Ltmp159:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[118:119], v144 offset:256
	ds_read_b64_tr_b16 v[120:121], v140 offset:8448
	ds_read_b64_tr_b16 v[122:123], v128
	ds_read_b64_tr_b16 v[124:125], v140 offset:8192
	ds_read_b64_tr_b16 v[126:127], v140 offset:8256
	ds_read_b64_tr_b16 v[128:129], v128 offset:512
.Ltmp160:
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v103, v157, v158
	v_cvt_pk_f16_f32 v104, v163, v159
	v_cvt_pk_f16_f32 v105, v160, v161
	.loc	1 370 51 is_stmt 0              ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[64:65], v[64:65], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[32:33] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[66:81], v[122:125], v[110:113], v[66:81]
	v_mul_f32_e64 v58, v58, v32
	v_mul_f32_e64 v59, v59, v32
	v_mul_f32_e64 v56, v56, v32
	v_mul_f32_e64 v57, v57, v32
	v_mul_f32_e64 v54, v54, v32
	v_mul_f32_e64 v55, v55, v32
	v_pk_mul_f32 v[52:53], v[52:53], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[32:33] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[118:121], v[106:109], v[66:81]
	v_mul_f32_e64 v44, v44, v32
	v_mul_f32_e64 v45, v45, v32
	v_mul_f32_e64 v42, v42, v32
	v_mul_f32_e64 v43, v43, v32
	v_mul_f32_e64 v40, v40, v32
	v_mul_f32_e64 v41, v41, v32
	v_pk_mul_f32 v[38:39], v[38:39], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[32:33] op_sel_hi:[1,0]
	.loc	1 315 28 is_stmt 1              ; flash-attention.py:315:28 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[82:97], v[196:199], v[114:117], v[82:97]
.Ltmp161:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[130:131], v140 offset:8704
	ds_read_b64_tr_b16 v[114:115], v144 offset:320
	ds_read_b64_tr_b16 v[132:133], v144 offset:384
	ds_read_b64_tr_b16 v[136:137], v144 offset:192
	ds_read_b64_tr_b16 v[116:117], v140 offset:8512
	ds_read_b64_tr_b16 v[122:123], v140 offset:8320
	ds_read_b64_tr_b16 v[138:139], v140 offset:8384
	ds_read_b64_tr_b16 v[142:143], v140 offset:8768
	ds_read_b64_tr_b16 v[134:135], v140 offset:8576
	ds_read_b64_tr_b16 v[118:119], v140 offset:8640
	ds_read_b64_tr_b16 v[182:183], v140 offset:8960
	ds_read_b64_tr_b16 v[186:187], v140 offset:9024
	ds_read_b64_tr_b16 v[190:191], v140 offset:8832
	ds_read_b64_tr_b16 v[194:195], v140 offset:8896
.Ltmp162:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[12:13], v[12:13], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[32:33] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[128:131], v[102:105], v[66:81]
.Ltmp163:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[180:181], v144 offset:768
	ds_read_b64_tr_b16 v[184:185], v144 offset:832
	ds_read_b64_tr_b16 v[128:129], v144 offset:896
	ds_read_b64_tr_b16 v[196:197], v144 offset:960
	ds_read_b64_tr_b16 v[124:125], v144 offset:64
	ds_read_b64_tr_b16 v[120:121], v144 offset:128
	ds_read_b64_tr_b16 v[130:131], v140 offset:9088
	ds_read_b64_tr_b16 v[198:199], v140 offset:9152
.Ltmp164:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[2:3], v[2:3], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[32:33] op_sel_hi:[1,0]
.Ltmp165:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[140:141], v144 offset:576
.Ltmp166:
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v98, v162, v146
	v_cvt_pk_f16_f32 v99, v148, v149
	v_cvt_pk_f16_f32 v100, v150, v151
	.loc	1 370 51 is_stmt 0              ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[50:65], v[124:127], v[110:113], v[50:65]
.Ltmp167:
	.loc	2 261 15 is_stmt 1              ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v124, v250, v239
	v_add_f32_e32 v124, v124, v238
	v_add_f32_e32 v124, v124, v240
	v_add_f32_e32 v124, v124, v237
	v_add_f32_e32 v124, v124, v236
	v_add_f32_e32 v124, v124, v235
.Ltmp168:
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v101, v152, v153
	.loc	1 370 51 is_stmt 0              ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[114:117], v[106:109], v[50:65]
.Ltmp169:
	.loc	2 261 15 is_stmt 1              ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v114, v124, v233
	v_add_f32_e32 v114, v114, v232
	v_add_f32_e32 v114, v114, v231
	v_add_f32_e32 v114, v114, v234
	v_add_f32_e32 v114, v114, v230
	v_add_f32_e32 v114, v114, v229
	v_add_f32_e32 v114, v114, v228
	v_add_f32_e32 v114, v114, v227
	v_add_f32_e32 v114, v114, v226
	v_add_f32_e32 v114, v114, v225
	v_add_f32_e32 v114, v114, v224
	v_add_f32_e32 v114, v114, v223
	v_add_f32_e32 v114, v114, v222
.Ltmp170:
	.loc	2 291 36                        ; standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ]
	v_mov_b32_e32 v115, v114
	s_nop 1
	v_permlane32_swap_b32_e32 v114, v115
.Ltmp171:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v114, v114, v115
.Ltmp172:
	.loc	1 355 20                        ; flash-attention.py:355:20 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v115, v218, v212
	.loc	1 355 28 is_stmt 0              ; flash-attention.py:355:28 @[ flash-attention.py:708:61 ]
	v_add_f32_e32 v114, v115, v114
.Ltmp173:
	.loc	2 261 15 is_stmt 1              ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v115, v166, v169
	v_add_f32_e32 v115, v170, v115
	v_add_f32_e32 v115, v171, v115
	v_add_f32_e32 v115, v172, v115
	v_add_f32_e32 v115, v173, v115
	v_add_f32_e32 v115, v167, v115
.Ltmp174:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[34:49], v[120:123], v[110:113], v[34:49]
.Ltmp175:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v115, v168, v115
	v_add_f32_e32 v115, v174, v115
	v_add_f32_e32 v115, v175, v115
.Ltmp176:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[188:189], v144 offset:640
	ds_read_b64_tr_b16 v[192:193], v144 offset:704
	ds_read_b64_tr_b16 v[116:117], v144 offset:448
.Ltmp177:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v115, v176, v115
	v_add_f32_e32 v115, v177, v115
.Ltmp178:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v120, s5, v248
.Ltmp179:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[0:15], v[136:139], v[110:113], v[0:15]
.Ltmp180:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v110, v164, v115
	v_add_f32_e32 v110, v165, v110
	v_add_f32_e32 v115, v178, v110
.Ltmp181:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max_f32_e32 v110, v17, v17
	v_max_f32_e32 v111, v16, v16
	v_max_f32_e32 v110, v111, v110
	v_max3_f32 v110, v110, v18, v19
.Ltmp182:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[132:135], v[106:109], v[34:49]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v18, 0x3e0293ee, v18
	v_mul_f32_e32 v19, 0x3e0293ee, v19
.Ltmp183:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
.Ltmp184:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[0:15], v[116:119], v[106:109], v[0:15]
.Ltmp185:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v106, v110, v20, v21
	v_max3_f32 v106, v106, v22, v23
	v_max3_f32 v106, v106, v24, v25
	v_max3_f32 v106, v106, v26, v27
	v_max3_f32 v106, v106, v28, v29
	v_max3_f32 v106, v106, v30, v31
	v_max3_f32 v106, v106, v82, v83
.Ltmp186:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[140:143], v[102:105], v[50:65]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v20, 0x3e0293ee, v20
	v_mul_f32_e32 v21, 0x3e0293ee, v21
	v_mul_f32_e32 v22, 0x3e0293ee, v22
	v_mul_f32_e32 v23, 0x3e0293ee, v23
	v_mul_f32_e32 v24, 0x3e0293ee, v24
	v_mul_f32_e32 v25, 0x3e0293ee, v25
	v_mul_f32_e32 v26, 0x3e0293ee, v26
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[188:191], v[102:105], v[34:49]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v27, 0x3e0293ee, v27
	v_mul_f32_e32 v28, 0x3e0293ee, v28
	v_mul_f32_e32 v29, 0x3e0293ee, v29
	v_mul_f32_e32 v30, 0x3e0293ee, v30
	v_mul_f32_e32 v31, 0x3e0293ee, v31
	v_mul_f32_e32 v82, 0x3e0293ee, v82
	v_mul_f32_e32 v83, 0x3e0293ee, v83
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[0:15], v[192:195], v[102:105], v[0:15]
.Ltmp187:
	.loc	2 168 27                        ; standard.py:168:27 @[ standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ] ]
	v_max3_f32 v102, v106, v84, v85
	v_max3_f32 v102, v102, v86, v87
	v_max3_f32 v102, v102, v88, v89
	v_max3_f32 v102, v102, v90, v91
	v_max3_f32 v102, v102, v92, v93
	v_max3_f32 v102, v102, v94, v95
	v_max3_f32 v102, v102, v96, v97
.Ltmp188:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[66:81], v[180:183], v[98:101], v[66:81]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v94, 0x3e0293ee, v94
	v_mul_f32_e32 v95, 0x3e0293ee, v95
	v_mul_f32_e32 v96, 0x3e0293ee, v96
	v_mul_f32_e32 v84, 0x3e0293ee, v84
	v_mul_f32_e32 v85, 0x3e0293ee, v85
	v_mul_f32_e32 v86, 0x3e0293ee, v86
	v_mul_f32_e32 v87, 0x3e0293ee, v87
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[184:187], v[98:101], v[50:65]
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v88, 0x3e0293ee, v88
	v_mul_f32_e32 v89, 0x3e0293ee, v89
	v_mul_f32_e32 v90, 0x3e0293ee, v90
	v_mul_f32_e32 v91, 0x3e0293ee, v91
	v_mul_f32_e32 v92, 0x3e0293ee, v92
	v_mul_f32_e32 v93, 0x3e0293ee, v93
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[128:131], v[98:101], v[34:49]
	v_mfma_f32_32x32x16_f16 v[0:15], v[196:199], v[98:101], v[0:15]
.Ltmp189:
	.loc	2 189 40                        ; standard.py:189:40 @[ flash-attention.py:334:42 @[ flash-attention.py:708:61 ] ]
	v_mov_b32_e32 v98, v102
	s_nop 1
	v_permlane32_swap_b32_e32 v102, v98
.Ltmp190:
	.loc	1 334 31                        ; flash-attention.py:334:31 @[ flash-attention.py:708:61 ]
	v_max3_f32 v99, v147, v102, v98
	.loc	1 336 18                        ; flash-attention.py:336:18 @[ flash-attention.py:708:61 ]
	v_mov_b32_e32 v98, v97
	v_mul_f32_e32 v100, 0x3e0293ee, v16
	v_mul_f32_e32 v101, 0x3e0293ee, v17
	v_pk_mul_f32 v[16:17], v[98:99], s[4:5] op_sel_hi:[1,0]
	s_mov_b32 s4, 0x800000
	.loc	1 336 29 is_stmt 0              ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v16, v16, v17
	v_sub_f32_e32 v97, v100, v17
	v_sub_f32_e32 v98, v101, v17
	v_sub_f32_e32 v18, v18, v17
	v_sub_f32_e32 v19, v19, v17
	v_sub_f32_e32 v20, v20, v17
	v_sub_f32_e32 v21, v21, v17
	v_sub_f32_e32 v22, v22, v17
	v_sub_f32_e32 v23, v23, v17
	.loc	1 337 25 is_stmt 1              ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v188, v16
	.loc	1 350 46                        ; flash-attention.py:350:46 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v16, v33, v17
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v140, v97
	v_exp_f32_e32 v141, v98
	v_exp_f32_e32 v142, v18
	v_exp_f32_e32 v143, v19
	v_exp_f32_e32 v144, v20
	v_exp_f32_e32 v145, v21
	v_exp_f32_e32 v147, v22
	v_exp_f32_e32 v164, v23
	.loc	1 350 29                        ; flash-attention.py:350:29 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v98, v16
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v24, v24, v17
	v_sub_f32_e32 v25, v25, v17
	v_sub_f32_e32 v26, v26, v17
	v_sub_f32_e32 v27, v27, v17
	v_sub_f32_e32 v28, v28, v17
	v_sub_f32_e32 v29, v29, v17
	v_sub_f32_e32 v30, v30, v17
	v_sub_f32_e32 v31, v31, v17
	v_sub_f32_e32 v94, v94, v17
	v_sub_f32_e32 v95, v95, v17
	v_sub_f32_e32 v96, v96, v17
.Ltmp191:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	v_add_u32_e32 v16, s5, v246
	v_add_u32_e32 v33, s5, v247
	ds_read_b64_tr_b16 v[100:101], v120 offset:256
	ds_read_b64_tr_b16 v[102:103], v33 offset:8448
	ds_read_b64_tr_b16 v[104:105], v16
	ds_read_b64_tr_b16 v[106:107], v33 offset:8192
	ds_read_b64_tr_b16 v[108:109], v33 offset:8256
	ds_read_b64_tr_b16 v[110:111], v16 offset:512
	ds_read_b64_tr_b16 v[112:113], v33 offset:8704
.Ltmp192:
	.loc	1 336 29                        ; flash-attention.py:336:29 @[ flash-attention.py:708:61 ]
	v_sub_f32_e32 v82, v82, v17
	v_sub_f32_e32 v83, v83, v17
	v_sub_f32_e32 v84, v84, v17
	v_sub_f32_e32 v85, v85, v17
	v_sub_f32_e32 v86, v86, v17
	v_sub_f32_e32 v87, v87, v17
	v_sub_f32_e32 v88, v88, v17
	v_sub_f32_e32 v89, v89, v17
	v_sub_f32_e32 v90, v90, v17
	v_sub_f32_e32 v91, v91, v17
	v_sub_f32_e32 v92, v92, v17
	v_sub_f32_e32 v93, v93, v17
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v165, v24
	v_exp_f32_e32 v166, v25
	v_exp_f32_e32 v167, v26
	v_exp_f32_e32 v168, v27
	v_exp_f32_e32 v169, v28
	v_exp_f32_e32 v170, v29
	v_exp_f32_e32 v171, v30
	v_exp_f32_e32 v172, v31
	v_exp_f32_e32 v185, v94
	v_exp_f32_e32 v186, v95
	v_exp_f32_e32 v187, v96
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v94, v140, v141
	v_cvt_pk_f16_f32 v95, v142, v143
	v_cvt_pk_f16_f32 v96, v144, v145
	v_cvt_pk_f16_f32 v97, v147, v164
	.loc	1 370 51 is_stmt 0              ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[30:31], v[80:81], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[78:79], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[76:77], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[74:75], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[72:73], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[70:71], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[68:69], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[66:67], v[98:99] op_sel_hi:[1,0]
	.loc	1 337 25 is_stmt 1              ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v181, v90
	v_exp_f32_e32 v182, v91
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[16:31], v[104:107], v[94:97], v[16:31]
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v183, v92
	v_exp_f32_e32 v184, v93
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v90, v165, v166
	v_cvt_pk_f16_f32 v91, v167, v168
	v_cvt_pk_f16_f32 v92, v169, v170
	v_cvt_pk_f16_f32 v93, v171, v172
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v173, v82
	v_exp_f32_e32 v174, v83
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[16:31], v[100:103], v[90:93], v[16:31]
	.loc	1 337 25                        ; flash-attention.py:337:25 @[ flash-attention.py:708:61 ]
	v_exp_f32_e32 v175, v84
	v_exp_f32_e32 v176, v85
	v_exp_f32_e32 v177, v86
	v_exp_f32_e32 v178, v87
	v_exp_f32_e32 v179, v88
	v_exp_f32_e32 v180, v89
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v86, v173, v174
	v_cvt_pk_f16_f32 v87, v175, v176
	v_cvt_pk_f16_f32 v88, v177, v178
	v_cvt_pk_f16_f32 v89, v179, v180
.Ltmp193:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[66:67], v120 offset:768
	ds_read_b64_tr_b16 v[70:71], v120 offset:320
	ds_read_b64_tr_b16 v[74:75], v120 offset:384
	ds_read_b64_tr_b16 v[78:79], v120 offset:192
	ds_read_b64_tr_b16 v[72:73], v33 offset:8512
	ds_read_b64_tr_b16 v[104:105], v33 offset:8320
	ds_read_b64_tr_b16 v[80:81], v33 offset:8384
.Ltmp194:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[16:31], v[110:113], v[86:89], v[16:31]
.Ltmp195:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[118:119], v33 offset:8768
	ds_read_b64_tr_b16 v[76:77], v33 offset:8576
	ds_read_b64_tr_b16 v[122:123], v33 offset:8640
	ds_read_b64_tr_b16 v[68:69], v33 offset:8960
	ds_read_b64_tr_b16 v[126:127], v33 offset:9024
	ds_read_b64_tr_b16 v[130:131], v33 offset:8832
	ds_read_b64_tr_b16 v[134:135], v33 offset:8896
.Ltmp196:
	.loc	1 370 31                        ; flash-attention.py:370:31 @[ flash-attention.py:708:61 ]
	v_cvt_pk_f16_f32 v82, v181, v182
	v_cvt_pk_f16_f32 v83, v183, v184
	v_cvt_pk_f16_f32 v84, v185, v186
	v_cvt_pk_f16_f32 v85, v187, v188
.Ltmp197:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[124:125], v120 offset:832
	ds_read_b64_tr_b16 v[110:111], v120 offset:896
	ds_read_b64_tr_b16 v[136:137], v120 offset:960
	ds_read_b64_tr_b16 v[106:107], v120 offset:64
	ds_read_b64_tr_b16 v[102:103], v120 offset:128
	ds_read_b64_tr_b16 v[112:113], v33 offset:9088
	ds_read_b64_tr_b16 v[138:139], v33 offset:9152
.Ltmp198:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v33, v154, v115
.Ltmp199:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[16:31], v[66:69], v[82:85], v[16:31]
	.loc	1 355 20                        ; flash-attention.py:355:20 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v67, v114, v32
.Ltmp200:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v32, v140, v141
	v_add_f32_e32 v32, v142, v32
	v_add_f32_e32 v32, v143, v32
	v_add_f32_e32 v32, v144, v32
	v_add_f32_e32 v32, v145, v32
	v_add_f32_e32 v32, v147, v32
	v_add_f32_e32 v32, v164, v32
	v_add_f32_e32 v32, v165, v32
	v_add_f32_e32 v32, v166, v32
	v_add_f32_e32 v32, v167, v32
	v_add_f32_e32 v32, v168, v32
	v_add_f32_e32 v32, v169, v32
.Ltmp201:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[64:65], v[64:65], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[98:99] op_sel_hi:[1,0]
.Ltmp202:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v33, v155, v33
.Ltmp203:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[48:49], v[48:49], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[98:99] op_sel_hi:[1,0]
.Ltmp204:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v32, v170, v32
.Ltmp205:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_pk_mul_f32 v[14:15], v[14:15], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[98:99] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[98:99] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[50:65], v[106:109], v[94:97], v[50:65]
.Ltmp206:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v33, v156, v33
	v_add_f32_e32 v32, v171, v32
	v_add_f32_e32 v33, v157, v33
	v_add_f32_e32 v32, v172, v32
	v_add_f32_e32 v33, v158, v33
	v_add_f32_e32 v32, v173, v32
	v_add_f32_e32 v33, v163, v33
.Ltmp207:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[34:49], v[102:105], v[94:97], v[34:49]
.Ltmp208:
	.loc	1 213 25                        ; flash-attention.py:213:25 @[ flash-attention.py:353:69 @[ flash-attention.py:708:61 ] ]
	ds_read_b64_tr_b16 v[116:117], v120 offset:576
	ds_read_b64_tr_b16 v[128:129], v120 offset:640
	ds_read_b64_tr_b16 v[132:133], v120 offset:704
	ds_read_b64_tr_b16 v[120:121], v120 offset:448
.Ltmp209:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v32, v174, v32
	v_add_f32_e32 v33, v159, v33
	v_add_f32_e32 v32, v175, v32
	v_add_f32_e32 v33, v160, v33
	v_add_f32_e32 v32, v176, v32
.Ltmp210:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[0:15], v[78:81], v[94:97], v[0:15]
.Ltmp211:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v33, v161, v33
	v_add_f32_e32 v32, v177, v32
	v_add_f32_e32 v33, v162, v33
	v_add_f32_e32 v32, v178, v32
	v_add_f32_e32 v33, v146, v33
	v_add_f32_e32 v32, v179, v32
	v_add_f32_e32 v33, v148, v33
.Ltmp212:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[70:73], v[90:93], v[50:65]
.Ltmp213:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v32, v180, v32
	v_add_f32_e32 v33, v149, v33
	v_add_f32_e32 v32, v181, v32
	v_add_f32_e32 v33, v150, v33
	v_add_f32_e32 v32, v182, v32
	v_add_f32_e32 v33, v151, v33
	v_add_f32_e32 v32, v183, v32
.Ltmp214:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[74:77], v[90:93], v[34:49]
.Ltmp215:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v33, v152, v33
	v_add_f32_e32 v32, v184, v32
	v_add_f32_e32 v33, v153, v33
	v_add_f32_e32 v32, v185, v32
.Ltmp216:
	.loc	2 291 36                        ; standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ]
	v_mov_b32_e32 v66, v33
.Ltmp217:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v32, v186, v32
.Ltmp218:
	.loc	2 291 36                        ; standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ]
	s_nop 0
	v_permlane32_swap_b32_e32 v33, v66
.Ltmp219:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[0:15], v[120:123], v[90:93], v[0:15]
.Ltmp220:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_add_f32_e32 v32, v187, v32
	v_add_f32_e32 v33, v33, v66
	v_add_f32_e32 v66, v188, v32
.Ltmp221:
	.loc	2 291 36                        ; standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ]
	v_mov_b32_e32 v32, v66
	s_nop 1
	v_permlane32_swap_b32_e32 v66, v32
.Ltmp222:
	.loc	2 261 15                        ; standard.py:261:15 @[ standard.py:291:36 @[ flash-attention.py:340:25 @[ flash-attention.py:708:61 ] ] ]
	v_pk_add_f32 v[32:33], v[66:67], v[32:33]
.Ltmp223:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[116:119], v[86:89], v[50:65]
	.loc	1 355 20                        ; flash-attention.py:355:20 @[ flash-attention.py:708:61 ]
	v_mul_f32_e32 v33, v33, v98
	.loc	1 355 28 is_stmt 0              ; flash-attention.py:355:28 @[ flash-attention.py:708:61 ]
	v_add_f32_e32 v32, v32, v33
.Ltmp224:
	.loc	1 0 0                           ; flash-attention.py:0
	v_cmp_gt_f32_e32 vcc, s4, v32
	v_mov_b32_e32 v66, 0x42000000
	s_nop 0
	v_cndmask_b32_e64 v33, 0, 32, vcc
	v_ldexp_f32 v33, v32, v33
.Ltmp225:
	.loc	1 370 51 is_stmt 1              ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[34:49], v[128:131], v[86:89], v[34:49]
.Ltmp226:
	.loc	1 0 0 is_stmt 0                 ; flash-attention.py:0
	v_log_f32_e32 v33, v33
	v_cndmask_b32_e32 v66, 0, v66, vcc
	.loc	1 712 16 is_stmt 1              ; flash-attention.py:712:16
	s_barrier
	.loc	1 0 0 is_stmt 0                 ; flash-attention.py:0
	v_sub_f32_e32 v33, v33, v66
	v_add_f32_e32 v33, v99, v33
	v_add3_u32 v66, 0, v244, v245
.Ltmp227:
	.loc	1 370 51 is_stmt 1              ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[0:15], v[132:135], v[86:89], v[0:15]
.Ltmp228:
	.loc	1 0 0 is_stmt 0                 ; flash-attention.py:0
	ds_write_b32 v66, v33
	s_waitcnt lgkmcnt(0)
.Ltmp229:
	.loc	1 370 51                        ; flash-attention.py:370:51 @[ flash-attention.py:708:61 ]
	v_mfma_f32_32x32x16_f16 v[50:65], v[124:127], v[82:85], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[110:113], v[82:85], v[34:49]
	v_mfma_f32_32x32x16_f16 v[0:15], v[136:139], v[82:85], v[0:15]
.Ltmp230:
	.loc	1 766 19 is_stmt 1              ; flash-attention.py:766:19
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	.loc	1 0 19 is_stmt 0                ; flash-attention.py:0:19
	s_sub_i32 s6, 0x4000, s16
	.loc	1 523 37 is_stmt 1              ; flash-attention.py:523:37
	v_or_b32_e32 v33, s16, v241
	s_movk_i32 s4, 0x4000
	v_cmp_gt_i32_e32 vcc, s6, v242
	.loc	1 653 48                        ; flash-attention.py:653:48
	v_cmp_gt_i32_e64 s[4:5], s4, v33
	v_bfrev_b32_e32 v33, 1
	s_and_b64 vcc, s[0:1], vcc
	v_cndmask_b32_e32 v33, v33, v243, vcc
	.loc	1 769 37                        ; flash-attention.py:769:37
	s_barrier
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr4_sgpr5
                                        ; implicit-def: $vgpr33
.LBB0_9:
	.loc	1 0 37 is_stmt 0                ; flash-attention.py:0:37
	v_bfrev_b32_e32 v33, 1
	v_cndmask_b32_e64 v33, v33, v243, s[0:1]
	s_or_b64 s[4:5], s[4:5], exec
	.loc	1 771 37 is_stmt 1              ; flash-attention.py:771:37
	s_barrier
.LBB0_10:
	.loc	1 741 30                        ; flash-attention.py:741:30
	v_div_scale_f32 v66, s[0:1], v32, v32, 1.0
	v_rcp_f32_e32 v67, v66
	v_div_scale_f32 v68, vcc, 1.0, v32, 1.0
	v_fma_f32 v69, -v66, v67, 1.0
	v_fmac_f32_e32 v67, v69, v67
	v_mul_f32_e32 v69, v68, v67
	v_fma_f32 v70, -v66, v69, v68
	v_fmac_f32_e32 v69, v70, v67
	v_fma_f32 v66, -v66, v69, v68
	v_div_fmas_f32 v66, v66, v67, v69
	v_lshl_add_u32 v67, v242, 2, 0
	.loc	1 0 0 is_stmt 0                 ; flash-attention.py:0
	ds_read_b32 v67, v67
	.loc	1 741 30                        ; flash-attention.py:741:30
	v_div_fixup_f32 v66, v66, v32, 1.0
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	.loc	1 742 28 is_stmt 1              ; flash-attention.py:742:28
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[6:7], v[66:67], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[66:67], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[66:67], v[0:1] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v7, v6, v7
	v_cvt_pk_f16_f32 v6, v4, v5
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[2:3], v[66:67], v[2:3] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v4, v0, v1
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[0:1], v[66:67], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[66:67], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[66:67], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[66:67], v[8:9] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v5, v2, v3
	v_cvt_pk_f16_f32 v3, v0, v1
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[0:1], v[66:67], v[46:47] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v15, v14, v15
	v_cvt_pk_f16_f32 v14, v12, v13
	v_cvt_pk_f16_f32 v12, v8, v9
	v_cvt_pk_f16_f32 v2, v0, v1
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[0:1], v[66:67], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[66:67], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[66:67], v[10:11] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v1, v0, v1
	v_cvt_pk_f16_f32 v0, v8, v9
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[8:9], v[66:67], v[40:41] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v13, v10, v11
	v_cvt_pk_f16_f32 v11, v8, v9
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[8:9], v[66:67], v[38:39] op_sel_hi:[0,1]
	.loc	1 0 0 is_stmt 0                 ; flash-attention.py:0
	buffer_store_dword v67, v33, s[8:11], 0 offen
	.loc	1 753 29 is_stmt 1              ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v10, v8, v9
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[8:9], v[66:67], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[66:67], v[34:35] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v9, v8, v9
	v_cvt_pk_f16_f32 v8, v32, v33
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[32:33], v[66:67], v[64:65] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v35, v32, v33
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[32:33], v[66:67], v[62:63] op_sel_hi:[0,1]
	.loc	1 774 63                        ; flash-attention.py:774:63
	s_mul_i32 s0, s3, s19
	.loc	1 774 41 is_stmt 0              ; flash-attention.py:774:41
	s_mul_i32 s1, s2, s18
	.loc	1 753 29 is_stmt 1              ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v34, v32, v33
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[32:33], v[66:67], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[66:67], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[66:67], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[66:67], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[66:67], v[16:17] op_sel_hi:[0,1]
	.loc	1 774 53                        ; flash-attention.py:774:53
	s_add_i32 s0, s0, s1
	.loc	1 775 36                        ; flash-attention.py:775:36
	s_mul_i32 s1, s12, s16
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v33, v32, v33
	v_cvt_pk_f16_f32 v32, v36, v37
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[36:37], v[66:67], v[56:57] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v23, v22, v23
	v_cvt_pk_f16_f32 v22, v20, v21
	v_cvt_pk_f16_f32 v20, v16, v17
	.loc	1 775 66                        ; flash-attention.py:775:66
	s_add_i32 s0, s0, s1
	v_mul_lo_u32 v16, s12, v241
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v39, v36, v37
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[36:37], v[66:67], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[66:67], v[18:19] op_sel_hi:[0,1]
	.loc	1 603 73                        ; flash-attention.py:603:73
	v_add_u32_e32 v16, s0, v16
	v_lshrrev_b32_e32 v17, 2, v211
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v38, v36, v37
	.loc	1 742 28                        ; flash-attention.py:742:28
	v_pk_mul_f32 v[36:37], v[66:67], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[66:67], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[66:67], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[66:67], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[66:67], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[66:67], v[24:25] op_sel_hi:[0,1]
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v21, v18, v19
	.loc	1 775 66                        ; flash-attention.py:775:66
	v_add_lshl_u32 v16, v16, v17, 1
	v_bfrev_b32_e32 v18, 1
	.loc	1 753 29                        ; flash-attention.py:753:29
	v_cvt_pk_f16_f32 v37, v36, v37
	v_cvt_pk_f16_f32 v36, v40, v41
	v_cvt_pk_f16_f32 v31, v30, v31
	v_cvt_pk_f16_f32 v30, v28, v29
	v_cvt_pk_f16_f32 v29, v26, v27
	v_cvt_pk_f16_f32 v28, v24, v25
	v_add_u32_e32 v17, 0xe0, v16
	v_add_u32_e32 v19, 0xc0, v16
	v_add_u32_e32 v24, 0xa0, v16
	v_add_u32_e32 v25, 0x80, v16
	v_add_u32_e32 v26, 0x60, v16
	v_add_u32_e32 v27, 64, v16
	v_add_u32_e32 v40, 32, v16
	v_cndmask_b32_e64 v16, v18, v16, s[4:5]
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s22, s10
	s_mov_b32 s23, s11
	v_permlane32_swap_b32_e32 v20, v22
	v_permlane32_swap_b32_e32 v21, v23
	v_cndmask_b32_e64 v17, v18, v17, s[4:5]
	v_cndmask_b32_e64 v19, v18, v19, s[4:5]
	v_cndmask_b32_e64 v24, v18, v24, s[4:5]
	v_cndmask_b32_e64 v25, v18, v25, s[4:5]
	v_cndmask_b32_e64 v26, v18, v26, s[4:5]
	v_cndmask_b32_e64 v27, v18, v27, s[4:5]
	v_cndmask_b32_e64 v40, v18, v40, s[4:5]
	v_permlane32_swap_b32_e32 v28, v30
	v_permlane32_swap_b32_e32 v29, v31
	v_permlane32_swap_b32_e32 v36, v38
	v_permlane32_swap_b32_e32 v37, v39
	v_permlane32_swap_b32_e32 v32, v34
	v_permlane32_swap_b32_e32 v33, v35
	v_permlane32_swap_b32_e32 v8, v10
	v_permlane32_swap_b32_e32 v9, v11
	v_permlane32_swap_b32_e32 v0, v2
	v_permlane32_swap_b32_e32 v1, v3
	v_permlane32_swap_b32_e32 v4, v6
	v_permlane32_swap_b32_e32 v5, v7
	v_permlane32_swap_b32_e32 v12, v14
	v_permlane32_swap_b32_e32 v13, v15
	.loc	1 781 33                        ; flash-attention.py:781:33
	buffer_store_dwordx4 v[20:23], v16, s[20:23], 0 offen
	buffer_store_dwordx4 v[28:31], v40, s[20:23], 0 offen
	buffer_store_dwordx4 v[36:39], v27, s[20:23], 0 offen
	buffer_store_dwordx4 v[32:35], v26, s[20:23], 0 offen
	buffer_store_dwordx4 v[8:11], v25, s[20:23], 0 offen
	buffer_store_dwordx4 v[0:3], v24, s[20:23], 0 offen
	buffer_store_dwordx4 v[4:7], v19, s[20:23], 0 offen
	buffer_store_dwordx4 v[12:15], v17, s[20:23], 0 offen
	.loc	1 511 4                         ; flash-attention.py:511:4
	s_endpgm
.Ltmp231:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 152
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
		.amdhsa_next_free_vgpr 255
		.amdhsa_next_free_sgpr 96
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
	.set attn_fwd.num_vgpr, 255
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 40
	.set attn_fwd.num_named_barrier, 0
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11828
; TotalNumSgprs: 46
; NumVgprs: 255
; NumAgprs: 0
; TotalNumVgprs: 255
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 255
; AccumOffset: 256
; Occupancy: 2
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
	.byte	7                               ; Abbreviation Code
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0xaa DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x84 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	519                             ; DW_AT_call_line
	.byte	41                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x56:0x5d DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	708                             ; DW_AT_call_line
	.byte	61                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x63:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	286                             ; DW_AT_call_line
	.byte	69                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x70:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	353                             ; DW_AT_call_line
	.byte	69                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7d:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	334                             ; DW_AT_call_line
	.byte	42                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x8a:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	189                             ; DW_AT_call_line
	.byte	40                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x97:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	340                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa4:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	291                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp227-.Lfunc_begin0
	.quad	.Ltmp228-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
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
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
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
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
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
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp195-.Lfunc_begin0
	.quad	.Ltmp196-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
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
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
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
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp219-.Lfunc_begin0
	.quad	.Ltmp220-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
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
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
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
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp216-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp218-.Lfunc_begin0
	.quad	.Ltmp220-.Lfunc_begin0
	.quad	.Ltmp221-.Lfunc_begin0
	.quad	.Ltmp222-.Lfunc_begin0
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"flash-attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/app/OAI-triton/fav3_kernel"   ; string offset=26
.Linfo_string3:
	.asciz	"attn_fwd"                      ; string offset=54
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
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 152
    .max_flat_workgroup_size: 512
    .name:           attn_fwd
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     255
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
