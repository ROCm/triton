	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
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
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
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
	s_lshl_b32 s34, s16, 8
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s13, s17
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_mov_b32_e32 v79, v0
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v34, 4, v79
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s14, s34
	v_or_b32_e32 v0, 0x60, v34
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_or_b32_e32 v11, s34, v0
	s_lshl_b32 s16, s14, 6
	v_mul_lo_u32 v12, s14, v0
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v0, 3, v79
	v_or_b32_e32 v1, 0xa0, v34
	s_add_u32 s0, s2, s0
	v_and_b32_e32 v58, 0x78, v0
	s_mul_i32 s52, s15, s18
	v_or_b32_e32 v19, s34, v1
	v_mul_lo_u32 v20, s14, v1
	s_addc_u32 s1, s3, s1
	v_mad_u64_u32 v[0:1], s[2:3], s14, v34, v[58:59]
	s_ashr_i32 s53, s52, 31
	s_lshl_b64 s[2:3], s[52:53], 1
	s_add_u32 s12, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s54, s20, s17
	s_addc_u32 s13, s5, s3
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[2:3], s[54:55], 1
	s_add_u32 s28, s12, s2
	s_mul_i32 s2, s22, s18
	s_addc_u32 s13, s13, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s23, s17
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s12, s6, s2
	s_addc_u32 s33, s7, s3
	s_and_b32 s2, s14, 0x3fff
	v_or_b32_e32 v35, 32, v34
	v_or_b32_e32 v3, s34, v34
	s_movk_i32 s6, 0x4000
	s_bitset1_b32 s2, 14
	v_or_b32_e32 v2, 0xe0, v34
	v_or_b32_e32 v4, s34, v35
	v_mul_lo_u32 v5, s14, v35
	v_add_u32_e32 v1, s16, v0
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_lshlrev_b32_e32 v0, 1, v0
	v_bfrev_b32_e32 v30, 1
	v_cmp_gt_i32_e32 vcc, s6, v3
	v_or_b32_e32 v10, 64, v3
	v_or_b32_e32 v27, s34, v2
	v_mul_lo_u32 v28, s14, v2
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_add_lshl_u32 v2, v5, v58, 1
	v_cmp_gt_i32_e32 vcc, s6, v4
	v_or_b32_e32 v18, 0x80, v3
	v_or_b32_e32 v26, 0xc0, v3
	v_cndmask_b32_e32 v13, v30, v2, vcc
	buffer_load_dwordx4 v[2:5], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v13, s[0:3], 0 offen
	v_lshlrev_b32_e32 v0, 1, v1
	v_cmp_gt_i32_e32 vcc, s6, v10
	v_add_u32_e32 v29, s16, v1
	v_add_lshl_u32 v1, v12, v58, 1
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s6, v11
	v_lshrrev_b32_e32 v41, 2, v79
	v_and_b32_e32 v36, 31, v79
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[10:13], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v1, s[0:3], 0 offen
	v_lshlrev_b32_e32 v0, 1, v29
	v_cmp_gt_i32_e32 vcc, s6, v18
	v_add_lshl_u32 v1, v20, v58, 1
	v_lshrrev_b32_e32 v37, 1, v79
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s6, v19
	v_mul_lo_u32 v35, s21, v35
	s_mov_b32 s30, s2
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[18:21], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[22:25], v1, s[0:3], 0 offen
	v_add_lshl_u32 v0, v29, s16, 1
	v_cmp_gt_i32_e32 vcc, s6, v26
	v_add_lshl_u32 v1, v28, v58, 1
	s_mov_b32 s31, s3
	v_cndmask_b32_e32 v0, v30, v0, vcc
	v_cmp_gt_i32_e32 vcc, s6, v27
	v_add_lshl_u32 v92, v35, v58, 1
	s_lshl_b32 s22, s24, 6
	v_cndmask_b32_e32 v1, v30, v1, vcc
	buffer_load_dwordx4 v[26:29], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[30:33], v1, s[0:3], 0 offen
	v_and_b32_e32 v1, 16, v79
	v_and_b32_e32 v0, 0x80, v79
	v_lshrrev_b32_e32 v38, 2, v1
	v_lshrrev_b32_e32 v40, 2, v0
	v_or_b32_e32 v39, v58, v38
	v_and_or_b32 v40, v41, 24, v40
	s_movk_i32 s0, 0xe0
	v_xor_b32_e32 v39, v40, v39
	v_and_or_b32 v42, v37, s0, v36
	v_mul_lo_u32 v37, s21, v34
	v_lshlrev_b32_e32 v39, 1, v39
	v_lshlrev_b32_e32 v34, 8, v34
	v_add3_u32 v89, 0, v39, v34
	s_barrier
	s_and_b32 s0, s21, 0x3fff
	s_bitset1_b32 s0, 14
	s_and_b32 s1, s13, 0xffff
	s_lshl_b32 s19, s0, 16
	s_or_b32 s29, s1, s19
	v_add_lshl_u32 v91, v37, v58, 1
	v_lshlrev_b32_e32 v78, 8, v36
	s_mov_b32 s36, 0
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_mov_b32 s16, 0x3e0293ee
	v_mov_b32_e32 v201, v92
	v_mov_b32_e32 v198, 1.0
	s_waitcnt vmcnt(7)
	ds_write_b64 v89, v[2:3]
	v_or_b32_e32 v2, 4, v58
	v_or_b32_e32 v3, v40, v38
	v_xor_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v90, 0, v2, v34
	s_waitcnt vmcnt(6)
	ds_write_b64 v89, v[6:7] offset:8192
	ds_write2st64_b64 v90, v[4:5], v[8:9] offset1:16
	s_waitcnt vmcnt(5)
	ds_write_b64 v89, v[10:11] offset:16384
	s_waitcnt vmcnt(4)
	ds_write_b64 v89, v[14:15] offset:24576
	ds_write2st64_b64 v90, v[12:13], v[16:17] offset0:32 offset1:48
	s_waitcnt vmcnt(3)
	ds_write_b64 v89, v[18:19] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b64 v89, v[22:23] offset:40960
	ds_write2st64_b64 v90, v[20:21], v[24:25] offset0:64 offset1:80
	s_waitcnt vmcnt(1)
	ds_write_b64 v89, v[26:27] offset:49152
	s_waitcnt vmcnt(0)
	ds_write_b64 v89, v[30:31] offset:57344
	ds_write2st64_b64 v90, v[28:29], v[32:33] offset0:96 offset1:112
	v_bfe_u32 v11, v79, 5, 1
	v_and_b32_e32 v12, 15, v79
	v_or_b32_e32 v2, 2, v11
	v_xor_b32_e32 v14, v2, v12
	v_or_b32_e32 v2, 4, v11
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_xor_b32_e32 v15, v2, v12
	buffer_load_dwordx4 v[2:5], v91, s[28:31], 0 offen
	buffer_load_dwordx4 v[6:9], v92, s[28:31], 0 offen
	v_lshrrev_b32_e32 v10, 5, v79
	v_xor_b32_e32 v13, v11, v12
	v_or_b32_e32 v16, 6, v11
	v_or_b32_e32 v17, 8, v11
	v_or_b32_e32 v18, 10, v11
	v_or_b32_e32 v19, 12, v11
	v_or_b32_e32 v20, 14, v11
	v_or_b32_e32 v21, 16, v11
	v_or_b32_e32 v22, 18, v11
	v_or_b32_e32 v23, 20, v11
	v_or_b32_e32 v24, 22, v11
	v_or_b32_e32 v25, 24, v11
	v_or_b32_e32 v26, 26, v11
	v_or_b32_e32 v11, 28, v11
	v_or_b32_e32 v10, 30, v10
	s_lshl_b32 s30, s21, 6
	v_xor_b32_e32 v16, v16, v12
	v_xor_b32_e32 v17, v17, v12
	v_xor_b32_e32 v18, v18, v12
	v_xor_b32_e32 v19, v19, v12
	v_xor_b32_e32 v20, v20, v12
	v_xor_b32_e32 v21, v21, v12
	v_xor_b32_e32 v22, v22, v12
	v_xor_b32_e32 v23, v23, v12
	v_xor_b32_e32 v24, v24, v12
	v_xor_b32_e32 v25, v25, v12
	v_xor_b32_e32 v26, v26, v12
	v_xor_b32_e32 v11, v11, v12
	v_xor_b32_e32 v10, v10, v12
	v_lshl_add_u32 v12, v42, 8, 0
	v_lshlrev_b32_e32 v13, 3, v13
	v_lshlrev_b32_e32 v59, 3, v14
	s_ashr_i32 s31, s30, 31
	scratch_store_dword off, v42, off offset:284 ; 4-byte Folded Spill
	v_add_u32_e32 v27, v12, v13
	v_add_u32_e32 v14, v12, v59
	v_lshlrev_b32_e32 v64, 3, v15
	v_lshlrev_b32_e32 v65, 3, v16
	v_lshlrev_b32_e32 v66, 3, v17
	s_lshl_b64 s[6:7], s[30:31], 1
	v_add_u32_e32 v15, v12, v64
	v_add_u32_e32 v16, v12, v65
	ds_read_b64 v[224:225], v27
	ds_read_b64 v[222:223], v14
	ds_read_b64 v[220:221], v15
	ds_read_b64 v[218:219], v16
	v_add_u32_e32 v14, v12, v66
	v_lshlrev_b32_e32 v67, 3, v18
	v_lshlrev_b32_e32 v68, 3, v19
	v_lshlrev_b32_e32 v69, 3, v20
	v_lshlrev_b32_e32 v70, 3, v21
	s_add_u32 s0, s28, s6
	v_add_u32_e32 v15, v12, v67
	v_add_u32_e32 v16, v12, v68
	v_add_u32_e32 v17, v12, v69
	ds_read_b64 v[216:217], v14
	ds_read_b64 v[214:215], v15
	ds_read_b64 v[212:213], v16
	ds_read_b64 v[210:211], v17
	v_add_u32_e32 v14, v12, v70
	v_lshlrev_b32_e32 v71, 3, v22
	v_lshlrev_b32_e32 v72, 3, v23
	v_lshlrev_b32_e32 v73, 3, v24
	v_lshlrev_b32_e32 v74, 3, v25
	s_addc_u32 s23, s13, s7
	v_add_u32_e32 v15, v12, v71
	v_add_u32_e32 v16, v12, v72
	v_add_u32_e32 v17, v12, v73
	ds_read_b64 v[208:209], v14
	ds_read_b64 v[206:207], v15
	ds_read_b64 v[204:205], v16
	ds_read_b64 v[202:203], v17
	v_add_u32_e32 v14, v12, v74
	v_lshlrev_b32_e32 v75, 3, v26
	v_lshlrev_b32_e32 v76, 3, v11
	v_lshlrev_b32_e32 v77, 3, v10
	s_and_b32 s1, s23, 0xffff
	v_add_u32_e32 v15, v12, v75
	v_add_u32_e32 v11, v12, v76
	v_add_u32_e32 v10, v12, v77
	ds_read_b64 v[86:87], v14
	ds_read_b64 v[84:85], v15
	ds_read_b64 v[82:83], v11
	ds_read_b64 v[80:81], v10
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(2)
	ds_write_b64 v89, v[2:3]
	s_waitcnt vmcnt(1)
	ds_write_b64 v89, v[6:7] offset:8192
	ds_write2st64_b64 v90, v[4:5], v[8:9] offset1:16
	s_or_b32 s1, s1, s19
	v_add3_u32 v88, 0, v13, v78
	buffer_load_dwordx4 v[50:53], v91, s[0:3], 0 offen
	buffer_load_dwordx4 v[54:57], v92, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[60:63], v88 offset1:16
	v_mov_b64_e32 v[34:35], s[36:37]
	v_mov_b64_e32 v[36:37], s[38:39]
	v_mov_b64_e32 v[38:39], s[40:41]
	v_mov_b64_e32 v[40:41], s[42:43]
	v_mov_b64_e32 v[42:43], s[44:45]
	v_mov_b64_e32 v[44:45], s[46:47]
	v_mov_b64_e32 v[46:47], s[48:49]
	v_mov_b64_e32 v[48:49], s[50:51]
	s_and_b32 s1, s24, 0x3fff
	s_bitset1_b32 s1, 14
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[60:61], v[224:225], v[34:49]
	v_add3_u32 v60, 0, v59, v78
	v_add3_u32 v61, 0, v64, v78
	v_add3_u32 v64, 0, v67, v78
	v_add3_u32 v67, 0, v69, v78
	v_add3_u32 v69, 0, v71, v78
	v_add3_u32 v71, 0, v73, v78
	v_add3_u32 v73, 0, v75, v78
	v_mfma_f32_32x32x8_f16 v[18:33], v[62:63], v[224:225], v[34:49]
	v_add3_u32 v62, 0, v65, v78
	v_add3_u32 v63, 0, v66, v78
	v_add3_u32 v65, 0, v68, v78
	v_add3_u32 v68, 0, v70, v78
	v_add3_u32 v70, 0, v72, v78
	s_nop 1
	ds_read2st64_b64 v[34:37], v60 offset1:16
	v_add3_u32 v72, 0, v74, v78
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[222:223], v[2:17]
	v_add3_u32 v74, 0, v76, v78
	v_add3_u32 v75, 0, v77, v78
	ds_read2st64_b64 v[40:43], v75 offset1:16
	v_lshrrev_b32_e32 v44, 3, v1
	v_lshrrev_b32_e32 v1, 3, v79
	v_and_b32_e32 v59, 4, v1
	v_and_b32_e32 v45, 8, v1
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[222:223], v[18:33]
	ds_read2st64_b64 v[34:37], v61 offset1:16
	v_and_b32_e32 v1, 0x100, v79
	v_lshrrev_b32_e32 v46, 3, v0
	v_or3_b32 v0, v44, v59, v45
	v_lshrrev_b32_e32 v47, 3, v1
	v_or3_b32 v0, v0, v46, v47
	scratch_store_dword off, v1, off offset:292 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[220:221], v[2:17]
	v_or_b32_e32 v1, 1, v0
	v_mul_lo_u32 v0, s24, v0
	s_and_b32 s13, s33, 0xffff
	s_lshl_b32 s20, s1, 16
	scratch_store_dwordx2 off, v[86:87], off offset:140 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[84:85], off offset:188 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[82:83], off offset:132 ; 8-byte Folded Spill
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[220:221], v[18:33]
	ds_read2st64_b64 v[34:37], v62 offset1:16
	v_mul_lo_u32 v1, s24, v1
	v_add_lshl_u32 v76, v0, v58, 1
	s_or_b32 s13, s13, s20
	v_add_lshl_u32 v77, v1, v58, 1
	s_mov_b32 s1, 0xff800000
	s_add_u32 s0, s0, s6
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[218:219], v[2:17]
	s_mov_b32 s24, 0x5040100
	s_mov_b32 s28, 0x7060302
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[218:219], v[18:33]
	ds_read2st64_b64 v[34:37], v63 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[216:217], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[216:217], v[18:33]
	ds_read2st64_b64 v[34:37], v64 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[214:215], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[214:215], v[18:33]
	ds_read2st64_b64 v[34:37], v65 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[212:213], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[212:213], v[18:33]
	ds_read2st64_b64 v[34:37], v67 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[210:211], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[210:211], v[18:33]
	ds_read2st64_b64 v[34:37], v68 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[208:209], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[208:209], v[18:33]
	ds_read2st64_b64 v[34:37], v69 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[206:207], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[206:207], v[18:33]
	ds_read2st64_b64 v[34:37], v70 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[204:205], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[204:205], v[18:33]
	ds_read2st64_b64 v[34:37], v71 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[202:203], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[202:203], v[18:33]
	ds_read2st64_b64 v[34:37], v72 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[86:87], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[86:87], v[18:33]
	ds_read2st64_b64 v[34:37], v73 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[84:85], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[84:85], v[18:33]
	ds_read2st64_b64 v[34:37], v74 offset1:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[34:35], v[82:83], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[36:37], v[82:83], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[40:41], v[80:81], v[2:17]
	buffer_load_dwordx4 v[34:37], v76, s[12:15], 0 offen
	buffer_load_dwordx4 v[38:41], v77, s[12:15], 0 offen
	s_mul_i32 s14, s21, 0x180
	scratch_store_dwordx2 off, v[80:81], off offset:124 ; 8-byte Folded Spill
	s_mul_hi_i32 s15, s30, 6
	s_movk_i32 s21, 0xffc0
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_nop 4
	v_max_f32_e32 v0, v3, v3
	v_max_f32_e32 v1, v2, v2
	v_mfma_f32_32x32x8_f16 v[18:33], v[42:43], v[80:81], v[18:33]
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v4, v5
	v_max3_f32 v0, v0, v6, v7
	v_max3_f32 v0, v0, v8, v9
	v_max3_f32 v0, v0, v10, v11
	v_max3_f32 v0, v0, v12, v13
	v_max3_f32 v0, v0, v14, v15
	v_max3_f32 v0, v0, v16, v17
	s_nop 2
	v_max3_f32 v0, v0, v18, v19
	v_max3_f32 v0, v0, v20, v21
	v_max3_f32 v0, v0, v22, v23
	v_max3_f32 v0, v0, v24, v25
	v_max3_f32 v0, v0, v26, v27
	v_max3_f32 v0, v0, v28, v29
	v_max3_f32 v0, v0, v30, v31
	v_lshlrev_b32_e32 v1, 2, v79
	v_max3_f32 v0, v0, v32, v33
	v_xor_b32_e32 v1, 0x80, v1
	scratch_store_dword off, v1, off offset:152 ; 4-byte Folded Spill
	ds_bpermute_b32 v1, v1, v0
	v_mov_b32_e32 v226, v33
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_max3_f32 v227, v0, v1, s1
	v_pk_mul_f32 v[42:43], v[226:227], s[16:17] op_sel_hi:[1,0]
	s_addc_u32 s1, s23, s7
	s_ashr_i32 s23, s22, 31
	v_fma_f32 v0, v2, s16, -v43
	v_fma_f32 v1, v3, s16, -v43
	v_fma_f32 v2, v4, s16, -v43
	v_fma_f32 v3, v5, s16, -v43
	v_fma_f32 v4, v6, s16, -v43
	v_fma_f32 v5, v7, s16, -v43
	v_fma_f32 v6, v8, s16, -v43
	v_fma_f32 v7, v9, s16, -v43
	v_fma_f32 v8, v10, s16, -v43
	v_fma_f32 v9, v11, s16, -v43
	v_fma_f32 v10, v12, s16, -v43
	v_fma_f32 v11, v13, s16, -v43
	v_fma_f32 v12, v14, s16, -v43
	v_fma_f32 v13, v15, s16, -v43
	v_fma_f32 v14, v16, s16, -v43
	v_fma_f32 v15, v17, s16, -v43
	v_fma_f32 v16, v18, s16, -v43
	v_fma_f32 v17, v19, s16, -v43
	v_fma_f32 v18, v20, s16, -v43
	v_fma_f32 v19, v21, s16, -v43
	v_fma_f32 v20, v22, s16, -v43
	v_fma_f32 v21, v23, s16, -v43
	v_fma_f32 v22, v24, s16, -v43
	v_fma_f32 v23, v25, s16, -v43
	v_fma_f32 v24, v26, s16, -v43
	v_fma_f32 v25, v27, s16, -v43
	v_fma_f32 v26, v28, s16, -v43
	v_fma_f32 v27, v29, s16, -v43
	v_fma_f32 v28, v30, s16, -v43
	v_fma_f32 v29, v31, s16, -v43
	v_fma_f32 v30, v32, s16, -v43
	s_lshl_b64 s[22:23], s[22:23], 1
	v_sub_f32_e32 v66, 0xff800000, v43
	s_add_u32 s29, s12, s22
	s_addc_u32 s31, s33, s23
	s_and_b32 s1, s1, 0xffff
	s_waitcnt vmcnt(9)
	ds_write_b64 v89, v[50:51]
	s_or_b32 s1, s1, s19
	s_waitcnt vmcnt(8)
	ds_write_b64 v89, v[54:55] offset:8192
	ds_write2st64_b64 v90, v[52:53], v[56:57] offset1:16
	buffer_load_dwordx4 v[114:117], v91, s[0:3], 0 offen
	buffer_load_dwordx4 v[118:121], v92, s[0:3], 0 offen
	s_waitcnt vmcnt(4)
	v_perm_b32 v31, v38, v34, s24
	v_perm_b32 v32, v38, v34, s28
	v_perm_b32 v33, v39, v35, s24
	v_perm_b32 v34, v39, v35, s28
	v_perm_b32 v35, v40, v36, s24
	v_perm_b32 v36, v40, v36, s28
	v_perm_b32 v38, v41, v37, s24
	v_perm_b32 v37, v41, v37, s28
	v_or_b32_e32 v39, v45, v59
	v_sub_f32_e32 v40, v42, v43
	v_bfe_i32 v41, v79, 0, 1
	v_bfe_i32 v42, v79, 1, 1
	v_or3_b32 v39, v39, v46, v47
	v_bfe_i32 v43, v79, 2, 1
	v_and_b32_e32 v46, 0x220, v41
	v_and_b32_e32 v47, 0x404, v42
	v_bfe_i32 v45, v79, 3, 1
	v_or_b32_e32 v48, v46, v47
	v_and_b32_e32 v49, 0x808, v43
	v_or_b32_e32 v50, v48, v49
	v_and_b32_e32 v51, 0x1010, v45
	v_or3_b32 v52, v51, v44, v50
	s_and_b32 s0, s31, 0xffff
	v_xor_b32_e32 v52, v39, v52
	s_or_b32 s1, s0, s20
	s_mov_b32 s0, s29
	scratch_store_dword off, v89, off offset:268 ; 4-byte Folded Spill
	scratch_store_dword off, v90, off offset:272 ; 4-byte Folded Spill
	scratch_store_dword off, v91, off offset:276 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshl_add_u32 v52, v52, 1, 0
	buffer_load_dwordx4 v[122:125], v77, s[0:3], 0 offen
	buffer_load_dwordx4 v[126:129], v76, s[0:3], 0 offen
	ds_write_b32 v52, v31 offset:16384
	v_or_b32_e32 v31, 0x44, v46
	scratch_load_dword v200, off, off offset:152 ; 4-byte Folded Reload
	v_xor_b32_e32 v31, v31, v47
	scratch_store_dword off, v52, off offset:156 ; 4-byte Folded Spill
	v_or_b32_e32 v52, v51, v49
	v_or3_b32 v31, v44, v31, v52
	v_xor_b32_e32 v31, v39, v31
	v_lshl_add_u32 v31, v31, 1, 0
	ds_read2st64_b64 v[190:193], v88 offset1:16
	ds_read2st64_b64 v[130:133], v60 offset1:16
	ds_read2st64_b64 v[134:137], v61 offset1:16
	ds_read2st64_b64 v[138:141], v62 offset1:16
	ds_read2st64_b64 v[142:145], v63 offset1:16
	ds_read2st64_b64 v[146:149], v64 offset1:16
	ds_read2st64_b64 v[150:153], v65 offset1:16
	ds_read2st64_b64 v[154:157], v67 offset1:16
	ds_read2st64_b64 v[158:161], v68 offset1:16
	ds_read2st64_b64 v[162:165], v69 offset1:16
	ds_read2st64_b64 v[166:169], v70 offset1:16
	ds_read2st64_b64 v[170:173], v71 offset1:16
	ds_read2st64_b64 v[174:177], v72 offset1:16
	ds_read2st64_b64 v[178:181], v73 offset1:16
	ds_read2st64_b64 v[182:185], v74 offset1:16
	ds_read2st64_b64 v[186:189], v75 offset1:16
	scratch_store_dword off, v31, off offset:160 ; 4-byte Folded Spill
	ds_write_b32 v31, v32 offset:16384
	v_or_b32_e32 v31, 0x88, v48
	v_xor_b32_e32 v31, v31, v49
	v_or3_b32 v31, v44, v31, v51
	v_xor_b32_e32 v31, v39, v31
	v_lshl_add_u32 v31, v31, 1, 0
	scratch_store_dword off, v31, off offset:164 ; 4-byte Folded Spill
	ds_write_b32 v31, v33 offset:16384
	v_or_b32_e32 v31, 0xcc, v46
	v_or_b32_e32 v32, v49, v47
	v_xor_b32_e32 v31, v32, v31
	v_or3_b32 v31, v44, v31, v51
	v_xor_b32_e32 v31, v39, v31
	v_lshl_add_u32 v31, v31, 1, 0
	scratch_store_dword off, v31, off offset:168 ; 4-byte Folded Spill
	ds_write_b32 v31, v34 offset:16384
	v_or_b32_e32 v31, 0x110, v50
	v_xor_b32_e32 v31, v31, v51
	v_or_b32_e32 v31, v31, v44
	v_xor_b32_e32 v31, v39, v31
	v_lshl_add_u32 v31, v31, 1, 0
	scratch_store_dword off, v31, off offset:172 ; 4-byte Folded Spill
	ds_write_b32 v31, v35 offset:16384
	v_or_b32_e32 v31, 0x154, v46
	v_xor_b32_e32 v31, v31, v47
	v_or_b32_e32 v31, v31, v49
	v_xor_b32_e32 v31, v31, v51
	v_or_b32_e32 v31, v31, v44
	v_xor_b32_e32 v31, v39, v31
	v_lshl_add_u32 v31, v31, 1, 0
	scratch_store_dword off, v31, off offset:176 ; 4-byte Folded Spill
	ds_write_b32 v31, v36 offset:16384
	v_or_b32_e32 v31, 0x198, v48
	v_xor_b32_e32 v31, v52, v31
	v_or_b32_e32 v31, v31, v44
	v_xor_b32_e32 v31, v39, v31
	v_lshl_add_u32 v31, v31, 1, 0
	scratch_store_dword off, v31, off offset:180 ; 4-byte Folded Spill
	ds_write_b32 v31, v38 offset:16384
	v_or_b32_e32 v31, v32, v51
	v_or_b32_e32 v32, 0x1dc, v46
	v_xor_b32_e32 v31, v31, v32
	v_or_b32_e32 v31, v31, v44
	v_xor_b32_e32 v31, v39, v31
	v_lshl_add_u32 v31, v31, 1, 0
	scratch_store_dword off, v31, off offset:184 ; 4-byte Folded Spill
	ds_write_b32 v31, v37 offset:16384
	v_and_b32_e32 v31, 0x44, v41
	v_and_b32_e32 v32, 0x88, v42
	v_and_b32_e32 v33, 0x110, v43
	v_or_b32_e32 v34, 24, v31
	v_or_b32_e32 v35, v33, v32
	v_or_b32_e32 v36, 0x818, v31
	v_or_b32_e32 v37, 0x1018, v31
	v_or_b32_e32 v38, 0x1818, v31
	v_bfe_i32 v39, v79, 4, 1
	v_or_b32_e32 v42, v31, v32
	v_xor_b32_e32 v34, v35, v34
	v_xor_b32_e32 v36, v35, v36
	v_xor_b32_e32 v37, v35, v37
	v_xor_b32_e32 v35, v35, v38
	v_and_b32_e32 v38, 0x220, v45
	v_and_b32_e32 v39, 0x404, v39
	v_or_b32_e32 v41, 8, v31
	v_or_b32_e32 v43, 16, v42
	v_or_b32_e32 v44, 0x808, v31
	v_or_b32_e32 v45, 0x810, v42
	v_or_b32_e32 v46, 0x1010, v42
	v_or_b32_e32 v47, 0x1008, v31
	v_or_b32_e32 v48, 0x1810, v42
	v_or_b32_e32 v49, 0x1808, v31
	v_or_b32_e32 v50, v42, v33
	v_xor_b32_e32 v41, v41, v32
	v_xor_b32_e32 v43, v43, v33
	v_xor_b32_e32 v44, v44, v32
	v_xor_b32_e32 v45, v45, v33
	v_xor_b32_e32 v46, v46, v33
	v_xor_b32_e32 v47, v47, v32
	v_xor_b32_e32 v48, v48, v33
	v_xor_b32_e32 v49, v49, v32
	v_or_b32_e32 v51, v50, v38
	v_xor_b32_e32 v52, v39, v59
	v_or3_b32 v41, v33, v41, v38
	v_or_b32_e32 v43, v43, v38
	v_or_b32_e32 v34, v34, v38
	v_or3_b32 v44, v33, v44, v38
	v_or_b32_e32 v36, v36, v38
	v_or_b32_e32 v45, v45, v38
	v_or_b32_e32 v46, v46, v38
	v_or_b32_e32 v37, v37, v38
	v_or3_b32 v47, v33, v47, v38
	v_or_b32_e32 v35, v35, v38
	v_or_b32_e32 v48, v48, v38
	v_or3_b32 v49, v33, v49, v38
	v_or_b32_e32 v38, v52, v38
	v_xor_b32_e32 v52, v52, v51
	v_or_b32_e32 v53, 0x800, v51
	v_or_b32_e32 v54, 0x1000, v51
	v_or_b32_e32 v51, 0x1800, v51
	v_xor_b32_e32 v41, v59, v41
	v_xor_b32_e32 v43, v59, v43
	v_xor_b32_e32 v34, v59, v34
	v_xor_b32_e32 v44, v59, v44
	v_xor_b32_e32 v53, v59, v53
	v_xor_b32_e32 v36, v59, v36
	v_xor_b32_e32 v45, v59, v45
	v_xor_b32_e32 v46, v59, v46
	v_xor_b32_e32 v37, v59, v37
	v_xor_b32_e32 v54, v59, v54
	v_xor_b32_e32 v47, v59, v47
	v_xor_b32_e32 v35, v59, v35
	v_xor_b32_e32 v48, v59, v48
	v_xor_b32_e32 v49, v59, v49
	v_xor_b32_e32 v51, v59, v51
	v_xor_b32_e32 v41, v41, v39
	v_xor_b32_e32 v43, v43, v39
	v_xor_b32_e32 v34, v34, v39
	v_xor_b32_e32 v44, v44, v39
	v_xor_b32_e32 v53, v53, v39
	v_xor_b32_e32 v36, v36, v39
	v_xor_b32_e32 v45, v45, v39
	v_xor_b32_e32 v46, v46, v39
	v_xor_b32_e32 v37, v37, v39
	v_xor_b32_e32 v54, v54, v39
	v_xor_b32_e32 v47, v47, v39
	v_xor_b32_e32 v35, v35, v39
	v_xor_b32_e32 v48, v48, v39
	v_xor_b32_e32 v49, v49, v39
	v_xor_b32_e32 v39, v51, v39
	v_or_b32_e32 v51, v38, v33
	scratch_store_dword off, v59, off offset:288 ; 4-byte Folded Spill
	v_or_b32_e32 v55, 56, v31
	v_or_b32_e32 v56, v51, v32
	v_or_b32_e32 v57, 0x838, v31
	v_or_b32_e32 v58, 0x1038, v31
	v_or_b32_e32 v59, 0x1838, v31
	scratch_store_dword off, v60, off offset:200 ; 4-byte Folded Spill
	scratch_store_dword off, v61, off offset:204 ; 4-byte Folded Spill
	v_xor_b32_e32 v55, v56, v55
	v_xor_b32_e32 v57, v56, v57
	v_xor_b32_e32 v58, v56, v58
	v_xor_b32_e32 v56, v56, v59
	v_or_b32_e32 v59, 48, v42
	v_or_b32_e32 v60, 0x830, v42
	v_or_b32_e32 v61, 0x1030, v42
	v_or_b32_e32 v42, 0x1830, v42
	scratch_store_dword off, v62, off offset:208 ; 4-byte Folded Spill
	scratch_store_dword off, v63, off offset:212 ; 4-byte Folded Spill
	v_xor_b32_e32 v59, v51, v59
	v_xor_b32_e32 v60, v51, v60
	v_xor_b32_e32 v61, v51, v61
	v_xor_b32_e32 v42, v51, v42
	v_or_b32_e32 v51, 40, v31
	v_or_b32_e32 v62, 0x828, v31
	v_or_b32_e32 v63, 0x1028, v31
	v_or_b32_e32 v31, 0x1828, v31
	v_exp_f32_e32 v195, v2
	v_lshl_add_u32 v2, v52, 1, 0
	v_xor_b32_e32 v51, v51, v32
	v_xor_b32_e32 v62, v62, v32
	v_xor_b32_e32 v63, v63, v32
	v_xor_b32_e32 v31, v31, v32
	scratch_store_dword off, v2, off offset:148 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v41, 1, 0
	v_or_b32_e32 v32, v51, v33
	v_or_b32_e32 v51, v62, v33
	v_or_b32_e32 v62, v63, v33
	v_or_b32_e32 v31, v31, v33
	v_or_b32_e32 v33, 32, v50
	scratch_store_dword off, v2, off        ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v43, 1, 0
	v_xor_b32_e32 v33, v38, v33
	scratch_store_dword off, v2, off offset:4 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v34, 1, 0
	v_xor_b32_e32 v32, v38, v32
	scratch_store_dword off, v2, off offset:8 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v33, 1, 0
	scratch_store_dword off, v2, off offset:12 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v32, 1, 0
	scratch_store_dword off, v2, off offset:16 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v59, 1, 0
	scratch_store_dword off, v2, off offset:20 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v55, 1, 0
	scratch_store_dword off, v2, off offset:24 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v44, 1, 0
	scratch_store_dword off, v2, off offset:28 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v53, 1, 0
	scratch_store_dword off, v2, off offset:32 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v36, 1, 0
	v_xor_b32_e32 v51, v38, v51
	v_or_b32_e32 v63, 0x820, v50
	scratch_store_dword off, v2, off offset:36 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v45, 1, 0
	v_xor_b32_e32 v63, v38, v63
	scratch_store_dword off, v2, off offset:40 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v51, 1, 0
	scratch_store_dword off, v2, off offset:44 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v63, 1, 0
	scratch_store_dword off, v2, off offset:48 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v57, 1, 0
	scratch_store_dword off, v2, off offset:52 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v60, 1, 0
	scratch_store_dword off, v2, off offset:56 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v46, 1, 0
	scratch_store_dword off, v2, off offset:60 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v37, 1, 0
	scratch_store_dword off, v2, off offset:64 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v54, 1, 0
	scratch_store_dword off, v2, off offset:68 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v47, 1, 0
	scratch_store_dword off, v64, off offset:216 ; 4-byte Folded Spill
	v_or_b32_e32 v64, 0x1020, v50
	scratch_store_dword off, v2, off offset:72 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v61, 1, 0
	v_xor_b32_e32 v64, v38, v64
	scratch_store_dword off, v2, off offset:76 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v58, 1, 0
	v_xor_b32_e32 v62, v38, v62
	scratch_store_dword off, v2, off offset:80 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v64, 1, 0
	scratch_store_dword off, v2, off offset:84 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v62, 1, 0
	scratch_store_dword off, v2, off offset:88 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v35, 1, 0
	scratch_store_dword off, v2, off offset:92 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v48, 1, 0
	scratch_store_dword off, v2, off offset:96 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v49, 1, 0
	scratch_store_dword off, v2, off offset:100 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v39, 1, 0
	s_add_u32 s12, s52, s54
	scratch_store_dword off, v2, off offset:104 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v56, 1, 0
	s_addc_u32 s13, s53, s55
	v_xor_b32_e32 v31, v38, v31
	v_or_b32_e32 v50, 0x1820, v50
	scratch_store_dword off, v2, off offset:108 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v42, 1, 0
	s_lshl_b64 s[12:13], s[12:13], 1
	v_xor_b32_e32 v38, v38, v50
	v_exp_f32_e32 v197, v0
	v_exp_f32_e32 v196, v1
	v_exp_f32_e32 v194, v3
	v_exp_f32_e32 v247, v4
	v_exp_f32_e32 v231, v5
	v_exp_f32_e32 v250, v6
	v_exp_f32_e32 v249, v7
	v_exp_f32_e32 v248, v8
	v_exp_f32_e32 v246, v9
	v_exp_f32_e32 v245, v10
	v_exp_f32_e32 v244, v11
	v_exp_f32_e32 v243, v12
	v_exp_f32_e32 v242, v13
	v_exp_f32_e32 v241, v14
	v_exp_f32_e32 v240, v15
	v_exp_f32_e32 v239, v16
	v_exp_f32_e32 v238, v17
	v_exp_f32_e32 v237, v18
	v_exp_f32_e32 v236, v19
	v_exp_f32_e32 v235, v20
	v_exp_f32_e32 v234, v21
	v_exp_f32_e32 v233, v22
	v_exp_f32_e32 v232, v23
	v_exp_f32_e32 v230, v24
	v_exp_f32_e32 v1, v25
	v_exp_f32_e32 v0, v26
	v_exp_f32_e32 v255, v27
	v_exp_f32_e32 v254, v28
	v_exp_f32_e32 v253, v29
	v_exp_f32_e32 v252, v30
	v_exp_f32_e32 v251, v40
	scratch_store_dword off, v2, off offset:112 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v31, 1, 0
	s_add_u32 s12, s14, s12
	v_exp_f32_e32 v228, v66
	scratch_store_dword off, v2, off offset:116 ; 4-byte Folded Spill
	v_lshl_add_u32 v2, v38, 1, 0
	s_addc_u32 s13, s15, s13
	scratch_store_dword off, v2, off offset:120 ; 4-byte Folded Spill
	s_add_u32 s4, s4, s12
	v_mov_b32_e32 v2, 0
	scratch_store_dword off, v65, off offset:220 ; 4-byte Folded Spill
	s_addc_u32 s5, s5, s13
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	s_waitcnt vmcnt(49)
	v_lshrrev_b32_e32 v199, 16, v122
	scratch_store_dword off, v88, off offset:196 ; 4-byte Folded Spill
	scratch_store_dword off, v67, off offset:224 ; 4-byte Folded Spill
	scratch_store_dword off, v68, off offset:228 ; 4-byte Folded Spill
	scratch_store_dword off, v69, off offset:232 ; 4-byte Folded Spill
	scratch_store_dword off, v70, off offset:236 ; 4-byte Folded Spill
	scratch_store_dword off, v71, off offset:240 ; 4-byte Folded Spill
	scratch_store_dword off, v72, off offset:244 ; 4-byte Folded Spill
	scratch_store_dword off, v73, off offset:248 ; 4-byte Folded Spill
	scratch_store_dword off, v74, off offset:252 ; 4-byte Folded Spill
	scratch_store_dword off, v75, off offset:256 ; 4-byte Folded Spill
	scratch_store_dword off, v79, off offset:280 ; 4-byte Folded Spill
	scratch_store_dword off, v77, off offset:264 ; 4-byte Folded Spill
	scratch_store_dword off, v76, off offset:260 ; 4-byte Folded Spill
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b64_e32 v[112:113], s[50:51]
	v_mov_b64_e32 v[110:111], s[48:49]
	v_mov_b64_e32 v[108:109], s[46:47]
	v_mov_b64_e32 v[106:107], s[44:45]
	v_mov_b64_e32 v[104:105], s[42:43]
	v_mov_b64_e32 v[102:103], s[40:41]
	v_mov_b64_e32 v[100:101], s[38:39]
	v_mov_b64_e32 v[98:99], s[36:37]
	v_mov_b32_e32 v229, v198
	v_pk_mul_f32 v[50:51], v[50:51], v[228:229] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[224:225], v[98:113]
	v_pk_mul_f32 v[52:53], v[52:53], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[192:193], v[224:225], v[98:113]
	v_pk_mul_f32 v[34:35], v[34:35], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[228:229] op_sel_hi:[1,0]
	s_nop 1
	v_add_f32_e32 v98, v197, v196
	v_add_f32_e32 v98, v98, v195
	v_add_f32_e32 v98, v98, v194
	v_add_f32_e32 v98, v98, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[222:223], v[66:81]
	v_add_f32_e32 v98, v98, v231
	v_add_f32_e32 v98, v98, v250
	v_add_f32_e32 v98, v98, v249
	v_add_f32_e32 v98, v98, v248
	v_add_f32_e32 v98, v98, v246
	v_add_f32_e32 v98, v98, v245
	v_add_f32_e32 v98, v98, v244
	v_add_f32_e32 v98, v98, v243
	v_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[220:221], v[66:81]
	v_add_f32_e32 v98, v98, v242
	v_add_f32_e32 v98, v98, v241
	v_add_f32_e32 v98, v98, v240
	v_add_f32_e32 v98, v98, v239
	v_add_f32_e32 v98, v98, v238
	v_add_f32_e32 v98, v98, v237
	v_add_f32_e32 v98, v98, v236
	v_add_f32_e32 v98, v98, v235
	v_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[218:219], v[66:81]
	v_add_f32_e32 v98, v98, v234
	v_add_f32_e32 v98, v98, v233
	v_add_f32_e32 v98, v98, v232
	v_add_f32_e32 v98, v98, v230
	v_add_f32_e32 v98, v98, v1
	v_add_f32_e32 v98, v98, v0
	v_add_f32_e32 v98, v98, v255
	v_mfma_f32_32x32x8_f16 v[82:97], v[132:133], v[222:223], v[82:97]
	v_add_f32_e32 v98, v98, v254
	v_add_f32_e32 v98, v98, v253
	v_add_f32_e32 v98, v98, v252
	v_add_f32_e32 v98, v98, v251
	s_waitcnt vmcnt(60)
	ds_bpermute_b32 v99, v200, v98
	scratch_load_dwordx2 v[138:139], off, off offset:188 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[134:135], off, off offset:132 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[130:131], off, off offset:124 ; 8-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[216:217], v[66:81]
	scratch_load_dwordx2 v[142:143], off, off offset:140 ; 8-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v198, v98, v99
	s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(15)
	s_barrier
	scratch_load_dword v98, off, off        ; 4-byte Folded Reload
	scratch_load_dword v100, off, off offset:120 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[136:137], v[220:221], v[82:97]
	v_pk_mul_f32 v[44:45], v[44:45], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[140:141], v[218:219], v[82:97]
	v_pk_mul_f32 v[26:27], v[26:27], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[228:229] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[82:97], v[144:145], v[216:217], v[82:97]
	v_pk_mul_f32 v[8:9], v[8:9], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[228:229] op_sel_hi:[1,0]
	v_fmac_f32_e32 v198, v229, v228
	v_cvt_f16_f32_e32 v229, v230
	v_mfma_f32_32x32x8_f16 v[82:97], v[148:149], v[214:215], v[82:97]
	v_cvt_f16_f32_e32 v230, v1
	v_mov_b32_e32 v226, v227
	v_cvt_f16_f32_e32 v227, v233
	v_cvt_f16_f32_e32 v228, v232
	s_add_u32 s12, s29, s22
	s_addc_u32 s30, s31, s23
	s_and_b32 s0, s5, 0xffff
	v_mfma_f32_32x32x8_f16 v[82:97], v[152:153], v[212:213], v[82:97]
	s_or_b32 s1, s0, s19
	s_mov_b32 s0, s4
	v_cvt_f16_f32_e32 v190, v241
	v_cvt_f16_f32_e32 v191, v240
	v_cvt_f16_f32_e32 v192, v239
	v_cvt_f16_f32_e32 v193, v238
	v_cvt_f16_f32_e32 v232, v255
	v_mfma_f32_32x32x8_f16 v[82:97], v[156:157], v[210:211], v[82:97]
	v_cvt_f16_f32_e32 v233, v254
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	s_waitcnt vmcnt(1)
	ds_read_b64 v[106:107], v98 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[100:101], v100 offset:16384
	v_mfma_f32_32x32x8_f16 v[82:97], v[160:161], v[208:209], v[82:97]
	scratch_load_dword v98, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[108:109], v98 offset:16384
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[206:207], v[82:97]
	scratch_load_dword v98, off, off offset:8 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[204:205], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[202:203], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[142:143], v[82:97]
	v_cvt_f16_f32_e32 v176, v197
	v_cvt_f16_f32_e32 v177, v195
	v_cvt_f16_f32_e32 v195, v236
	v_cvt_f16_f32_e32 v197, v234
	v_cvt_f16_f32_e32 v234, v253
	v_cvt_f16_f32_e32 v236, v251
	v_mfma_f32_32x32x8_f16 v[82:97], v[180:181], v[138:139], v[82:97]
	v_cvt_f16_f32_e32 v181, v231
	v_cvt_f16_f32_e32 v231, v0
	scratch_load_dword v0, off, off offset:148 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v180, v247
	s_waitcnt vmcnt(0)
	ds_read_b64 v[0:1], v0 offset:16384
	v_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[214:215], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[150:151], v[212:213], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[154:155], v[210:211], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[158:159], v[208:209], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[162:163], v[206:207], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[166:167], v[204:205], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[170:171], v[202:203], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[174:175], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[138:139], v[66:81]
	v_cvt_f16_f32_e32 v179, v194
	v_cvt_f16_f32_e32 v194, v237
	scratch_load_dword v237, off, off offset:268 ; 4-byte Folded Reload
	ds_read_b64 v[110:111], v98 offset:16384
	scratch_load_dword v98, off, off offset:12 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v178, v196
	v_pack_b32_f16 v177, v177, v179
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[134:135], v[66:81]
	v_cvt_f16_f32_e32 v182, v250
	v_pack_b32_f16 v176, v176, v178
	v_cvt_f16_f32_e32 v183, v249
	v_pack_b32_f16 v178, v180, v181
	v_cvt_f16_f32_e32 v196, v235
	v_cvt_f16_f32_e32 v235, v252
	v_pack_b32_f16 v179, v182, v183
	v_mfma_f32_32x32x8_f16 v[82:97], v[184:185], v[134:135], v[82:97]
	v_cvt_f16_f32_e32 v184, v248
	v_cvt_f16_f32_e32 v185, v246
	v_pack_b32_f16 v183, v190, v191
	v_pack_b32_f16 v191, v235, v236
	v_pack_b32_f16 v190, v233, v234
	v_pack_b32_f16 v180, v184, v185
	v_pack_b32_f16 v185, v194, v195
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[130:131], v[66:81]
	v_cvt_f16_f32_e32 v186, v245
	v_cvt_f16_f32_e32 v187, v244
	v_pack_b32_f16 v184, v192, v193
	v_pack_b32_f16 v181, v186, v187
	v_pack_b32_f16 v187, v227, v228
	v_mfma_f32_32x32x8_f16 v[82:97], v[188:189], v[130:131], v[82:97]
	v_cvt_f16_f32_e32 v188, v243
	v_cvt_f16_f32_e32 v189, v242
	v_pack_b32_f16 v186, v196, v197
	v_pack_b32_f16 v182, v188, v189
	v_pack_b32_f16 v188, v229, v230
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[50:65], v[0:1], v[176:177], v[50:65]
	v_max_f32_e32 v0, v67, v67
	v_max_f32_e32 v1, v66, v66
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v68, v69
	v_max3_f32 v0, v0, v70, v71
	v_max3_f32 v0, v0, v72, v73
	v_max3_f32 v0, v0, v74, v75
	v_max3_f32 v0, v0, v76, v77
	v_max3_f32 v0, v0, v78, v79
	v_max3_f32 v0, v0, v80, v81
	v_max3_f32 v0, v0, v82, v83
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	ds_bpermute_b32 v1, v200, v0
	v_mfma_f32_32x32x8_f16 v[50:65], v[106:107], v[178:179], v[50:65]
	v_pack_b32_f16 v189, v231, v232
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v227, v226, v0, v1
	v_pk_mul_f32 v[228:229], v[226:227], s[16:17] op_sel_hi:[1,0]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[112:113], v98 offset:16384
	scratch_load_dword v98, off, off offset:16 ; 4-byte Folded Reload
	v_fma_f32 v0, v66, s16, -v229
	v_fma_f32 v66, v68, s16, -v229
	v_exp_f32_e32 v195, v66
	v_sub_f32_e32 v66, v228, v229
	v_exp_f32_e32 v228, v66
	v_mfma_f32_32x32x8_f16 v[50:65], v[108:109], v[180:181], v[50:65]
	v_fma_f32 v1, v67, s16, -v229
	v_fma_f32 v67, v69, s16, -v229
	v_exp_f32_e32 v194, v67
	v_fma_f32 v68, v70, s16, -v229
	v_fma_f32 v69, v71, s16, -v229
	v_fma_f32 v70, v72, s16, -v229
	v_fma_f32 v71, v73, s16, -v229
	v_mfma_f32_32x32x8_f16 v[50:65], v[110:111], v[182:183], v[50:65]
	v_fma_f32 v72, v74, s16, -v229
	v_fma_f32 v73, v75, s16, -v229
	v_fma_f32 v74, v76, s16, -v229
	v_fma_f32 v75, v77, s16, -v229
	v_fma_f32 v76, v78, s16, -v229
	v_fma_f32 v77, v79, s16, -v229
	v_fma_f32 v78, v80, s16, -v229
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[112:113], v[184:185], v[50:65]
	v_fma_f32 v79, v81, s16, -v229
	v_fma_f32 v80, v82, s16, -v229
	v_fma_f32 v81, v83, s16, -v229
	v_fma_f32 v82, v84, s16, -v229
	v_fma_f32 v83, v85, s16, -v229
	v_fma_f32 v84, v86, s16, -v229
	v_fma_f32 v85, v87, s16, -v229
	v_fma_f32 v86, v88, s16, -v229
	v_fma_f32 v87, v89, s16, -v229
	v_fma_f32 v88, v90, s16, -v229
	v_fma_f32 v89, v91, s16, -v229
	v_fma_f32 v90, v92, s16, -v229
	v_fma_f32 v91, v93, s16, -v229
	v_fma_f32 v92, v94, s16, -v229
	v_fma_f32 v93, v95, s16, -v229
	v_fma_f32 v94, v96, s16, -v229
	v_fma_f32 v95, v97, s16, -v229
	v_exp_f32_e32 v197, v0
	v_exp_f32_e32 v196, v1
	v_exp_f32_e32 v247, v68
	v_exp_f32_e32 v231, v69
	v_exp_f32_e32 v250, v70
	v_exp_f32_e32 v249, v71
	v_exp_f32_e32 v248, v72
	v_exp_f32_e32 v246, v73
	v_exp_f32_e32 v245, v74
	v_exp_f32_e32 v244, v75
	v_exp_f32_e32 v243, v76
	v_exp_f32_e32 v242, v77
	v_exp_f32_e32 v241, v78
	v_exp_f32_e32 v240, v79
	v_exp_f32_e32 v239, v80
	v_exp_f32_e32 v238, v81
	v_exp_f32_e32 v236, v83
	v_exp_f32_e32 v235, v84
	v_exp_f32_e32 v234, v85
	v_exp_f32_e32 v233, v86
	v_exp_f32_e32 v232, v87
	v_exp_f32_e32 v230, v88
	v_exp_f32_e32 v1, v89
	v_exp_f32_e32 v0, v90
	v_exp_f32_e32 v255, v91
	v_exp_f32_e32 v254, v92
	v_exp_f32_e32 v253, v93
	v_exp_f32_e32 v252, v94
	v_exp_f32_e32 v251, v95
	s_waitcnt vmcnt(0)
	ds_read_b64 v[130:131], v98 offset:16384
	scratch_load_dword v98, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[130:131], v[186:187], v[50:65]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[132:133], v98 offset:16384
	scratch_load_dword v98, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[132:133], v[188:189], v[50:65]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[134:135], v98 offset:16384
	scratch_load_dword v98, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[134:135], v[190:191], v[50:65]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[136:137], v98 offset:16384
	scratch_load_dword v98, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[136:137], v[176:177], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[138:139], v98 offset:16384
	scratch_load_dword v98, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[138:139], v[178:179], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[140:141], v98 offset:16384
	scratch_load_dword v98, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[140:141], v[180:181], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[142:143], v98 offset:16384
	scratch_load_dword v98, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[142:143], v[182:183], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[144:145], v98 offset:16384
	scratch_load_dword v98, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[144:145], v[184:185], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[146:147], v98 offset:16384
	scratch_load_dword v98, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[146:147], v[186:187], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[148:149], v98 offset:16384
	scratch_load_dword v98, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[148:149], v[188:189], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[150:151], v98 offset:16384
	scratch_load_dword v98, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[150:151], v[190:191], v[34:49]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[152:153], v98 offset:16384
	scratch_load_dword v98, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[152:153], v[176:177], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[154:155], v98 offset:16384
	scratch_load_dword v98, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[154:155], v[178:179], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[156:157], v98 offset:16384
	scratch_load_dword v98, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[156:157], v[180:181], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[158:159], v98 offset:16384
	scratch_load_dword v98, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[158:159], v[182:183], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[160:161], v98 offset:16384
	scratch_load_dword v98, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[160:161], v[184:185], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[162:163], v98 offset:16384
	scratch_load_dword v98, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[162:163], v[186:187], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[164:165], v98 offset:16384
	scratch_load_dword v98, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[164:165], v[188:189], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[166:167], v98 offset:16384
	scratch_load_dword v98, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[166:167], v[190:191], v[18:33]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[168:169], v98 offset:16384
	scratch_load_dword v98, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[168:169], v[176:177], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v98 offset:16384
	scratch_load_dword v98, off, off offset:100 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[170:171], v[178:179], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[172:173], v98 offset:16384
	scratch_load_dword v98, off, off offset:104 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[172:173], v[180:181], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[174:175], v98 offset:16384
	scratch_load_dword v98, off, off offset:108 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[174:175], v[182:183], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[104:105], v98 offset:16384
	scratch_load_dword v98, off, off offset:112 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[104:105], v[184:185], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[102:103], v98 offset:16384
	scratch_load_dword v98, off, off offset:116 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[2:17], v[102:103], v[186:187], v[2:17]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[98:99], v98 offset:16384
	ds_write_b64 v237, v[114:115]
	ds_write_b64 v237, v[118:119] offset:8192
	scratch_load_dword v114, off, off offset:272 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[98:99], v[188:189], v[2:17]
	v_exp_f32_e32 v237, v82
	s_waitcnt vmcnt(0)
	ds_write2st64_b64 v114, v[116:117], v[120:121] offset1:16
	scratch_load_dword v114, off, off offset:276 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[114:117], v114, s[0:3], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[118:121], v201, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v66, off, off offset:196 ; 4-byte Folded Reload
	scratch_load_dword v67, off, off offset:156 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[2:17], v[100:101], v[190:191], v[2:17]
	s_and_b32 s0, s30, 0xffff
	s_or_b32 s13, s0, s20
	s_add_u32 s29, s29, s22
	s_addc_u32 s31, s31, s23
	s_add_u32 s4, s4, s6
	s_addc_u32 s5, s5, s7
	s_add_i32 s21, s21, 64
	s_cmpk_lt_u32 s21, 0x1f00
	s_waitcnt vmcnt(1)
	ds_read2st64_b64 v[190:193], v66 offset1:16
	scratch_load_dword v66, off, off offset:200 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[130:133], v66 offset1:16
	scratch_load_dword v66, off, off offset:204 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[134:137], v66 offset1:16
	scratch_load_dword v66, off, off offset:208 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[138:141], v66 offset1:16
	scratch_load_dword v66, off, off offset:212 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[142:145], v66 offset1:16
	scratch_load_dword v66, off, off offset:216 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[146:149], v66 offset1:16
	scratch_load_dword v66, off, off offset:220 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[150:153], v66 offset1:16
	scratch_load_dword v66, off, off offset:224 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[154:157], v66 offset1:16
	scratch_load_dword v66, off, off offset:228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[158:161], v66 offset1:16
	scratch_load_dword v66, off, off offset:232 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[162:165], v66 offset1:16
	scratch_load_dword v66, off, off offset:236 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[166:169], v66 offset1:16
	scratch_load_dword v66, off, off offset:240 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[170:173], v66 offset1:16
	scratch_load_dword v66, off, off offset:244 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[174:177], v66 offset1:16
	scratch_load_dword v66, off, off offset:248 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[178:181], v66 offset1:16
	scratch_load_dword v66, off, off offset:252 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[182:185], v66 offset1:16
	scratch_load_dword v66, off, off offset:256 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[186:189], v66 offset1:16
	v_perm_b32 v66, v122, v126, s24
	ds_write_b32 v67, v66 offset:16384
	scratch_load_dword v67, off, off offset:160 ; 4-byte Folded Reload
	v_alignbit_b32 v66, v199, v126, 16
	s_waitcnt vmcnt(0)
	ds_write_b32 v67, v66 offset:16384
	scratch_load_dword v67, off, off offset:164 ; 4-byte Folded Reload
	v_perm_b32 v66, v123, v127, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v67, v66 offset:16384
	scratch_load_dword v67, off, off offset:168 ; 4-byte Folded Reload
	v_perm_b32 v66, v123, v127, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v67, v66 offset:16384
	scratch_load_dword v67, off, off offset:172 ; 4-byte Folded Reload
	v_perm_b32 v66, v124, v128, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v67, v66 offset:16384
	scratch_load_dword v67, off, off offset:176 ; 4-byte Folded Reload
	v_perm_b32 v66, v124, v128, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v67, v66 offset:16384
	scratch_load_dword v67, off, off offset:180 ; 4-byte Folded Reload
	v_perm_b32 v66, v125, v129, s24
	s_waitcnt vmcnt(0)
	ds_write_b32 v67, v66 offset:16384
	scratch_load_dword v67, off, off offset:184 ; 4-byte Folded Reload
	v_perm_b32 v66, v125, v129, s28
	s_waitcnt vmcnt(0)
	ds_write_b32 v67, v66 offset:16384
	scratch_load_dword v66, off, off offset:260 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[126:129], v66, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v66, off, off offset:264 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[122:125], v66, s[12:15], 0 offen
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v199, 16, v122
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	scratch_load_dword v66, off, off offset:292 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[98:99], off, off offset:140 ; 8-byte Folded Reload
	s_mov_b32 s36, 0
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	v_pk_mul_f32 v[64:65], v[64:65], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[228:229] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[228:229] op_sel_hi:[1,0]
	s_mul_i32 s2, s18, 0xc0000
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s8, s2
	s_addc_u32 s5, s9, s3
	s_lshl_b32 s2, s17, 14
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s4, s4, s2
	s_addc_u32 s5, s5, s3
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[2:3], s[34:35], 2
	s_add_u32 s4, s4, s2
	s_addc_u32 s16, s5, s3
	s_add_i32 s2, s34, 0xffffc100
	s_add_u32 s12, s12, s22
	s_addc_u32 s3, s30, s23
	s_and_b32 s13, s3, 0xffff
	s_cmp_lt_i32 s2, 1
	s_mov_b32 s2, 0x3e0293ee
	s_mov_b32 s5, 0x5040100
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	scratch_load_dword v201, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_cmp_eq_u32_e64 s[0:1], 0, v66
	v_mov_b64_e32 v[80:81], s[50:51]
	v_mov_b64_e32 v[78:79], s[48:49]
	v_mov_b64_e32 v[76:77], s[46:47]
	v_mov_b64_e32 v[74:75], s[44:45]
	v_mov_b64_e32 v[72:73], s[42:43]
	v_mov_b64_e32 v[70:71], s[40:41]
	v_mov_b64_e32 v[68:69], s[38:39]
	v_mov_b64_e32 v[66:67], s[36:37]
	s_waitcnt lgkmcnt(14)
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[82:97], v[190:191], v[224:225], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[130:131], v[222:223], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[134:135], v[220:221], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[138:139], v[218:219], v[82:97]
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[138:139], v[98:99]
	v_mfma_f32_32x32x8_f16 v[82:97], v[142:143], v[216:217], v[82:97]
	v_cvt_f16_f32_e32 v142, v242
	v_cvt_f16_f32_e32 v143, v244
	v_mfma_f32_32x32x8_f16 v[82:97], v[146:147], v[214:215], v[82:97]
	v_cvt_f16_f32_e32 v146, v238
	v_cvt_f16_f32_e32 v147, v240
	v_mfma_f32_32x32x8_f16 v[82:97], v[150:151], v[212:213], v[82:97]
	v_cvt_f16_f32_e32 v150, v234
	v_cvt_f16_f32_e32 v151, v236
	v_mfma_f32_32x32x8_f16 v[82:97], v[154:155], v[210:211], v[82:97]
	v_cvt_f16_f32_e32 v154, v232
	v_cvt_f16_f32_e32 v155, v0
	v_mfma_f32_32x32x8_f16 v[82:97], v[158:159], v[208:209], v[82:97]
	v_cvt_f16_f32_e32 v159, v251
	v_cvt_f16_f32_e32 v158, v252
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[206:207], v[82:97]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[204:205], v[82:97]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[202:203], v[82:97]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[98:99], v[82:97]
	scratch_load_dwordx2 v[174:175], off, off offset:188 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[98:99], off, off offset:132 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[134:135], v[98:99]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[82:97], v[178:179], v[174:175], v[82:97]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_f16 v[82:97], v[182:183], v[98:99], v[82:97]
	scratch_load_dwordx2 v[98:99], off, off offset:124 ; 8-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v162, off, off       ; 4-byte Folded Reload
	scratch_load_dword v166, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v170, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x8_f16 v[82:97], v[186:187], v[98:99], v[82:97]
	v_mov_b64_e32 v[130:131], v[98:99]
	scratch_load_dword v178, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v182, off, off offset:36 ; 4-byte Folded Reload
	scratch_load_dword v186, off, off offset:44 ; 4-byte Folded Reload
	scratch_load_dword v190, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v200, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(4)
	ds_read_b64 v[178:179], v178 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[192:193], v[224:225], v[66:81]
	scratch_load_dword v192, off, off offset:56 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[132:133], v[222:223], v[98:113]
	v_cvt_f16_f32_e32 v133, v195
	v_cvt_f16_f32_e32 v132, v247
	v_mfma_f32_32x32x8_f16 v[98:113], v[136:137], v[220:221], v[98:113]
	v_cvt_f16_f32_e32 v136, v248
	v_cvt_f16_f32_e32 v137, v250
	v_mfma_f32_32x32x8_f16 v[98:113], v[140:141], v[218:219], v[98:113]
	v_cvt_f16_f32_e32 v140, v243
	v_cvt_f16_f32_e32 v141, v245
	v_mfma_f32_32x32x8_f16 v[98:113], v[144:145], v[216:217], v[98:113]
	v_cvt_f16_f32_e32 v144, v239
	v_cvt_f16_f32_e32 v145, v241
	v_mfma_f32_32x32x8_f16 v[98:113], v[148:149], v[214:215], v[98:113]
	v_cvt_f16_f32_e32 v148, v235
	v_cvt_f16_f32_e32 v149, v237
	v_mfma_f32_32x32x8_f16 v[98:113], v[152:153], v[212:213], v[98:113]
	v_cvt_f16_f32_e32 v152, v230
	v_cvt_f16_f32_e32 v153, v233
	v_mfma_f32_32x32x8_f16 v[98:113], v[156:157], v[210:211], v[98:113]
	v_cvt_f16_f32_e32 v156, v253
	v_cvt_f16_f32_e32 v157, v255
	v_mfma_f32_32x32x8_f16 v[98:113], v[160:161], v[208:209], v[98:113]
	scratch_load_dword v160, off, off offset:148 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[160:161], v160 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[164:165], v[206:207], v[98:113]
	scratch_load_dword v164, off, off offset:4 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[168:169], v[204:205], v[98:113]
	scratch_load_dword v168, off, off offset:12 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[172:173], v[202:203], v[98:113]
	scratch_load_dword v172, off, off offset:20 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[98:113], v[176:177], v[138:139], v[98:113]
	scratch_load_dword v176, off, off offset:24 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v138, v246
	v_cvt_f16_f32_e32 v139, v249
	ds_read_b64 v[162:163], v162 offset:16384
	ds_read_b64 v[166:167], v166 offset:16384
	ds_read_b64 v[170:171], v170 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[164:165], v164 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[180:181], v[174:175], v[98:113]
	scratch_load_dword v180, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(3)
	ds_read_b64 v[168:169], v168 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[184:185], v[134:135], v[98:113]
	v_cvt_f16_f32_e32 v135, v194
	scratch_load_dword v184, off, off offset:40 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v134, v231
	s_waitcnt vmcnt(3)
	ds_read_b64 v[172:173], v172 offset:16384
	v_mfma_f32_32x32x8_f16 v[98:113], v[188:189], v[130:131], v[98:113]
	v_add_f32_e32 v130, v197, v196
	v_add_f32_e32 v130, v130, v195
	v_add_f32_e32 v130, v130, v194
	v_add_f32_e32 v130, v130, v247
	v_add_f32_e32 v130, v130, v231
	v_add_f32_e32 v130, v130, v250
	s_waitcnt vmcnt(2)
	ds_read_b64 v[176:177], v176 offset:16384
	v_add_f32_e32 v130, v130, v249
	v_add_f32_e32 v130, v130, v248
	v_add_f32_e32 v130, v130, v246
	v_add_f32_e32 v130, v130, v245
	v_add_f32_e32 v130, v130, v244
	v_add_f32_e32 v130, v130, v243
	v_add_f32_e32 v130, v130, v242
	v_add_f32_e32 v130, v130, v241
	v_add_f32_e32 v130, v130, v240
	v_add_f32_e32 v130, v130, v239
	v_add_f32_e32 v130, v130, v238
	v_add_f32_e32 v130, v130, v237
	v_add_f32_e32 v130, v130, v236
	v_add_f32_e32 v130, v130, v235
	v_add_f32_e32 v130, v130, v234
	v_add_f32_e32 v130, v130, v233
	v_add_f32_e32 v130, v130, v232
	v_add_f32_e32 v130, v130, v230
	v_add_f32_e32 v130, v130, v1
	v_add_f32_e32 v130, v130, v0
	v_add_f32_e32 v130, v130, v255
	v_add_f32_e32 v130, v130, v254
	v_add_f32_e32 v130, v130, v253
	v_add_f32_e32 v130, v130, v252
	v_add_f32_e32 v130, v130, v251
	ds_bpermute_b32 v131, v201, v130
	scratch_load_dword v188, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v194, off, off offset:60 ; 4-byte Folded Reload
	ds_read_b64 v[230:231], v200 offset:16384
	scratch_load_dword v200, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v226, v130, v131
	v_fmac_f32_e32 v226, v198, v228
	v_cvt_f16_f32_e32 v131, v196
	scratch_load_dword v196, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v198, off, off offset:68 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v130, v197
	v_cvt_f16_f32_e32 v0, v254
	ds_read_b64 v[182:183], v182 offset:16384
	ds_read_b64 v[186:187], v186 offset:16384
	ds_read_b64 v[190:191], v190 offset:16384
	ds_read_b64 v[192:193], v192 offset:16384
	v_cvt_f16_f32_e32 v1, v1
	v_pack_b32_f16 v0, v0, v156
	s_waitcnt vmcnt(6)
	ds_read_b64 v[180:181], v180 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[184:185], v184 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[188:189], v188 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[194:195], v194 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[232:233], v200 offset:16384
	scratch_load_dword v200, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	ds_read_b64 v[196:197], v196 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[198:199], v198 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[234:235], v200 offset:16384
	scratch_load_dword v200, off, off offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[236:237], v200 offset:16384
	scratch_load_dword v200, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[238:239], v200 offset:16384
	scratch_load_dword v200, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[240:241], v200 offset:16384
	scratch_load_dword v200, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[242:243], v200 offset:16384
	scratch_load_dword v200, off, off offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[244:245], v200 offset:16384
	scratch_load_dword v200, off, off offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[246:247], v200 offset:16384
	scratch_load_dword v200, off, off offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[248:249], v200 offset:16384
	scratch_load_dword v200, off, off offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[250:251], v200 offset:16384
	scratch_load_dword v200, off, off offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[252:253], v200 offset:16384
	scratch_load_dword v200, off, off offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[254:255], v200 offset:16384
	scratch_load_dword v200, off, off offset:268 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_write_b64 v200, v[114:115]
	ds_write_b64 v200, v[118:119] offset:8192
	scratch_load_dword v114, off, off offset:272 ; 4-byte Folded Reload
	v_pack_b32_f16 v115, v133, v135
	v_pack_b32_f16 v119, v141, v143
	v_pack_b32_f16 v118, v136, v138
	v_pack_b32_f16 v133, v153, v154
	v_pack_b32_f16 v135, v155, v157
	s_waitcnt vmcnt(0)
	ds_write2st64_b64 v114, v[116:117], v[120:121] offset1:16
	v_pack_b32_f16 v114, v130, v131
	v_pack_b32_f16 v117, v137, v139
	v_pack_b32_f16 v116, v132, v134
	v_mfma_f32_32x32x8_f16 v[50:65], v[160:161], v[114:115], v[50:65]
	v_pack_b32_f16 v121, v145, v147
	v_pack_b32_f16 v120, v140, v142
	v_pack_b32_f16 v131, v149, v151
	v_pack_b32_f16 v130, v144, v146
	v_pack_b32_f16 v132, v148, v150
	v_pack_b32_f16 v134, v152, v1
	v_pack_b32_f16 v1, v158, v159
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[114:115], v[34:49]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[18:33], v[194:195], v[114:115], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[240:241], v[114:115], v[2:17]
	scratch_load_dword v240, off, off offset:160 ; 4-byte Folded Reload
	scratch_load_dword v241, off, off offset:164 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[50:65], v[162:163], v[116:117], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[116:117], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[196:197], v[116:117], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[242:243], v[116:117], v[2:17]
	scratch_load_dword v243, off, off offset:172 ; 4-byte Folded Reload
	scratch_load_dword v242, off, off offset:168 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[50:65], v[164:165], v[118:119], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[118:119], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[198:199], v[118:119], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[244:245], v[118:119], v[2:17]
	scratch_load_dword v244, off, off offset:176 ; 4-byte Folded Reload
	scratch_load_dword v245, off, off offset:180 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[50:65], v[166:167], v[120:121], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[120:121], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[230:231], v[120:121], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[246:247], v[120:121], v[2:17]
	scratch_load_dword v246, off, off offset:184 ; 4-byte Folded Reload
	v_mov_b32_e32 v247, v201
	v_mfma_f32_32x32x8_f16 v[50:65], v[168:169], v[130:131], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[130:131], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[232:233], v[130:131], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[248:249], v[130:131], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[170:171], v[132:133], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[188:189], v[132:133], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[234:235], v[132:133], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[250:251], v[132:133], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[134:135], v[50:65]
	v_mov_b32_e32 v172, v113
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[134:135], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[236:237], v[134:135], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[252:253], v[134:135], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[0:1], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[192:193], v[0:1], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[238:239], v[0:1], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[254:255], v[0:1], v[2:17]
	v_max_f32_e32 v0, v83, v83
	v_max_f32_e32 v1, v82, v82
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	v_max3_f32 v0, v0, v94, v95
	v_max3_f32 v0, v0, v96, v97
	v_max3_f32 v0, v0, v98, v99
	v_max3_f32 v0, v0, v100, v101
	v_max3_f32 v0, v0, v102, v103
	v_max3_f32 v0, v0, v104, v105
	v_max3_f32 v0, v0, v106, v107
	v_max3_f32 v0, v0, v108, v109
	v_max3_f32 v0, v0, v110, v111
	v_max3_f32 v0, v0, v112, v113
	ds_bpermute_b32 v1, v201, v0
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v173, v227, v0, v1
	v_pk_mul_f32 v[170:171], v[172:173], s[2:3] op_sel_hi:[1,0]
	s_mov_b32 s3, 0x7060302
	v_fma_f32 v0, v82, s2, -v171
	v_fma_f32 v82, v84, s2, -v171
	v_exp_f32_e32 v172, v82
	v_sub_f32_e32 v82, v229, v171
	v_fma_f32 v1, v83, s2, -v171
	v_fma_f32 v83, v85, s2, -v171
	v_fma_f32 v85, v87, s2, -v171
	v_fma_f32 v87, v89, s2, -v171
	v_fma_f32 v89, v91, s2, -v171
	v_fma_f32 v91, v93, s2, -v171
	v_fma_f32 v93, v95, s2, -v171
	v_fma_f32 v95, v97, s2, -v171
	v_fma_f32 v97, v99, s2, -v171
	v_fma_f32 v99, v101, s2, -v171
	v_fma_f32 v101, v103, s2, -v171
	v_fma_f32 v103, v105, s2, -v171
	v_fma_f32 v105, v107, s2, -v171
	v_fma_f32 v107, v109, s2, -v171
	v_fma_f32 v109, v111, s2, -v171
	v_sub_f32_e32 v111, v170, v171
	v_exp_f32_e32 v170, v82
	scratch_load_dword v82, off, off offset:196 ; 4-byte Folded Reload
	scratch_load_dword v229, off, off offset:156 ; 4-byte Folded Reload
	v_fma_f32 v84, v86, s2, -v171
	v_fma_f32 v86, v88, s2, -v171
	v_fma_f32 v88, v90, s2, -v171
	v_fma_f32 v90, v92, s2, -v171
	v_fma_f32 v92, v94, s2, -v171
	v_fma_f32 v94, v96, s2, -v171
	v_fma_f32 v96, v98, s2, -v171
	v_fma_f32 v98, v100, s2, -v171
	v_fma_f32 v100, v102, s2, -v171
	v_fma_f32 v102, v104, s2, -v171
	v_fma_f32 v104, v106, s2, -v171
	v_fma_f32 v106, v108, s2, -v171
	v_fma_f32 v108, v110, s2, -v171
	v_fma_f32 v110, v112, s2, -v171
	v_exp_f32_e32 v228, v110
	v_exp_f32_e32 v230, v111
	v_exp_f32_e32 v198, v106
	v_exp_f32_e32 v199, v107
	v_exp_f32_e32 v231, v108
	v_exp_f32_e32 v227, v109
	v_exp_f32_e32 v190, v98
	v_exp_f32_e32 v191, v99
	v_exp_f32_e32 v192, v100
	v_exp_f32_e32 v193, v101
	v_exp_f32_e32 v201, v83
	v_exp_f32_e32 v176, v84
	v_exp_f32_e32 v177, v85
	v_exp_f32_e32 v178, v86
	v_exp_f32_e32 v179, v87
	v_exp_f32_e32 v180, v88
	v_exp_f32_e32 v181, v89
	v_exp_f32_e32 v182, v90
	v_exp_f32_e32 v183, v91
	v_exp_f32_e32 v184, v92
	v_exp_f32_e32 v185, v93
	v_exp_f32_e32 v186, v94
	v_exp_f32_e32 v187, v95
	v_exp_f32_e32 v188, v96
	v_exp_f32_e32 v189, v97
	v_exp_f32_e32 v194, v102
	v_exp_f32_e32 v195, v103
	v_exp_f32_e32 v196, v104
	v_exp_f32_e32 v197, v105
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v1, v1
	v_pk_mul_f32 v[64:65], v[64:65], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[170:171] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[170:171] op_sel_hi:[1,0]
	s_waitcnt vmcnt(1)
	ds_read2st64_b64 v[232:235], v82 offset1:16
	scratch_load_dword v82, off, off offset:200 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[236:239], v82 offset1:16
	scratch_load_dword v82, off, off offset:204 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[166:169], v82 offset1:16
	scratch_load_dword v82, off, off offset:208 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[162:165], v82 offset1:16
	scratch_load_dword v82, off, off offset:212 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[158:161], v82 offset1:16
	scratch_load_dword v82, off, off offset:216 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[154:157], v82 offset1:16
	scratch_load_dword v82, off, off offset:220 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[150:153], v82 offset1:16
	scratch_load_dword v82, off, off offset:224 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[146:149], v82 offset1:16
	scratch_load_dword v82, off, off offset:228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[142:145], v82 offset1:16
	scratch_load_dword v82, off, off offset:232 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[138:141], v82 offset1:16
	scratch_load_dword v82, off, off offset:236 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[134:137], v82 offset1:16
	scratch_load_dword v82, off, off offset:240 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[130:133], v82 offset1:16
	scratch_load_dword v82, off, off offset:244 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[118:121], v82 offset1:16
	scratch_load_dword v82, off, off offset:248 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[114:117], v82 offset1:16
	scratch_load_dword v82, off, off offset:252 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[110:113], v82 offset1:16
	scratch_load_dword v82, off, off offset:256 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read2st64_b64 v[106:109], v82 offset1:16
	v_perm_b32 v82, v122, v126, s5
	ds_write_b32 v229, v82 offset:16384
	v_perm_b32 v82, v122, v126, s3
	ds_write_b32 v240, v82 offset:16384
	v_perm_b32 v82, v123, v127, s5
	ds_write_b32 v241, v82 offset:16384
	v_perm_b32 v82, v123, v127, s3
	scratch_load_dwordx2 v[122:123], off, off offset:140 ; 8-byte Folded Reload
	ds_write_b32 v242, v82 offset:16384
	v_perm_b32 v82, v124, v128, s5
	ds_write_b32 v243, v82 offset:16384
	v_perm_b32 v82, v124, v128, s3
	ds_write_b32 v244, v82 offset:16384
	v_perm_b32 v82, v125, v129, s5
	ds_write_b32 v245, v82 offset:16384
	v_perm_b32 v82, v125, v129, s3
	ds_write_b32 v246, v82 offset:16384
	scratch_load_dword v82, off, off offset:260 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v125, v201
	v_cvt_f16_f32_e32 v124, v1
	v_cvt_f16_f32_e32 v128, v178
	v_cvt_f16_f32_e32 v129, v179
	v_cvt_f16_f32_e32 v126, v176
	v_cvt_f16_f32_e32 v127, v177
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[98:101], v82, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v82, off, off offset:264 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[102:105], v82, s[12:15], 0 offen
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[82:97], v[232:233], v[224:225], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[236:237], v[222:223], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[220:221], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[218:219], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[158:159], v[216:217], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[154:155], v[214:215], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[150:151], v[212:213], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[146:147], v[210:211], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[142:143], v[208:209], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[138:139], v[206:207], v[82:97]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[82:97], v[134:135], v[204:205], v[82:97]
	v_cvt_f16_f32_e32 v134, v184
	v_cvt_f16_f32_e32 v135, v185
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[82:97], v[130:131], v[202:203], v[82:97]
	v_cvt_f16_f32_e32 v130, v180
	v_cvt_f16_f32_e32 v131, v181
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[122:123], v[82:97]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[174:175], v[82:97]
	scratch_load_dwordx2 v[114:115], off, off offset:132 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0) lgkmcnt(9)
	v_mfma_f32_32x32x8_f16 v[82:97], v[110:111], v[114:115], v[82:97]
	scratch_load_dwordx2 v[110:111], off, off offset:124 ; 8-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v232, off, off offset:104 ; 4-byte Folded Reload
	scratch_load_dword v233, off, off offset:108 ; 4-byte Folded Reload
	scratch_load_dword v236, off, off offset:120 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[66:81], v[234:235], v[224:225], v[66:81]
	scratch_load_dword v234, off, off offset:112 ; 4-byte Folded Reload
	scratch_load_dword v235, off, off offset:116 ; 4-byte Folded Reload
	scratch_load_dword v224, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v225, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	ds_read_b64 v[118:119], v235 offset:16384
	v_mfma_f32_32x32x8_f16 v[66:81], v[238:239], v[222:223], v[66:81]
	scratch_load_dword v222, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v223, off, off offset:68 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[66:81], v[168:169], v[220:221], v[66:81]
	scratch_load_dword v220, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v221, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[138:139], v221 offset:16384
	v_mfma_f32_32x32x8_f16 v[66:81], v[164:165], v[218:219], v[66:81]
	scratch_load_dword v218, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v219, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[158:159], v219 offset:16384
	v_mfma_f32_32x32x8_f16 v[66:81], v[160:161], v[216:217], v[66:81]
	ds_read_b64 v[160:161], v220 offset:16384
	scratch_load_dword v216, off, off offset:40 ; 4-byte Folded Reload
	scratch_load_dword v217, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	ds_read_b64 v[168:169], v216 offset:16384
	v_mfma_f32_32x32x8_f16 v[66:81], v[156:157], v[214:215], v[66:81]
	scratch_load_dword v214, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v215, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	ds_read_b64 v[154:155], v217 offset:16384
	ds_read_b64 v[156:157], v218 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[164:165], v214 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[166:167], v215 offset:16384
	v_mfma_f32_32x32x8_f16 v[66:81], v[152:153], v[212:213], v[66:81]
	scratch_load_dword v212, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v213, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[162:163], v213 offset:16384
	v_mfma_f32_32x32x8_f16 v[82:97], v[106:107], v[110:111], v[82:97]
	v_add_f32_e32 v106, v0, v1
	v_add_f32_e32 v106, v172, v106
	v_add_f32_e32 v106, v201, v106
	v_add_f32_e32 v106, v176, v106
	v_add_f32_e32 v106, v177, v106
	v_add_f32_e32 v106, v178, v106
	v_add_f32_e32 v106, v179, v106
	v_mfma_f32_32x32x8_f16 v[66:81], v[148:149], v[210:211], v[66:81]
	v_add_f32_e32 v106, v180, v106
	v_add_f32_e32 v106, v181, v106
	v_add_f32_e32 v106, v182, v106
	v_add_f32_e32 v106, v183, v106
	v_add_f32_e32 v106, v184, v106
	v_add_f32_e32 v106, v185, v106
	v_add_f32_e32 v106, v186, v106
	v_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[208:209], v[66:81]
	v_add_f32_e32 v106, v187, v106
	v_add_f32_e32 v106, v188, v106
	v_add_f32_e32 v106, v189, v106
	v_add_f32_e32 v106, v190, v106
	v_add_f32_e32 v106, v191, v106
	v_add_f32_e32 v106, v192, v106
	v_add_f32_e32 v106, v193, v106
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[206:207], v[66:81]
	v_add_f32_e32 v106, v194, v106
	v_add_f32_e32 v106, v195, v106
	v_add_f32_e32 v106, v196, v106
	v_add_f32_e32 v106, v197, v106
	v_add_f32_e32 v106, v198, v106
	v_add_f32_e32 v106, v199, v106
	v_add_f32_e32 v106, v231, v106
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[204:205], v[66:81]
	scratch_load_dword v205, off, off offset:148 ; 4-byte Folded Reload
	v_add_f32_e32 v106, v227, v106
	v_add_f32_e32 v106, v228, v106
	v_add_f32_e32 v106, v230, v106
	ds_bpermute_b32 v107, v247, v106
	scratch_load_dword v206, off, off       ; 4-byte Folded Reload
	scratch_load_dword v207, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v208, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v209, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v210, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v211, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v200, v106, v107
	v_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[202:203], v[66:81]
	v_fmac_f32_e32 v200, v226, v170
	v_mov_b32_e32 v204, v229
	v_cvt_f16_f32_e32 v201, v227
	v_cvt_f16_f32_e32 v202, v228
	scratch_load_dword v226, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v227, off, off offset:84 ; 4-byte Folded Reload
	scratch_load_dword v228, off, off offset:88 ; 4-byte Folded Reload
	scratch_load_dword v229, off, off offset:92 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v203, v230
	scratch_load_dword v230, off, off offset:96 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[122:123], v[66:81]
	v_cvt_f16_f32_e32 v123, v172
	v_cvt_f16_f32_e32 v172, v188
	v_cvt_f16_f32_e32 v188, v189
	v_cvt_f16_f32_e32 v189, v190
	v_cvt_f16_f32_e32 v190, v191
	v_cvt_f16_f32_e32 v191, v192
	v_cvt_f16_f32_e32 v192, v193
	v_cvt_f16_f32_e32 v193, v194
	v_cvt_f16_f32_e32 v194, v195
	v_cvt_f16_f32_e32 v195, v196
	v_cvt_f16_f32_e32 v196, v197
	v_cvt_f16_f32_e32 v197, v198
	v_cvt_f16_f32_e32 v198, v199
	v_cvt_f16_f32_e32 v199, v231
	scratch_load_dword v231, off, off offset:100 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[174:175], v[66:81]
	v_cvt_f16_f32_e32 v122, v0
	v_pack_b32_f16 v123, v123, v125
	v_cvt_f16_f32_e32 v132, v182
	v_cvt_f16_f32_e32 v133, v183
	v_pack_b32_f16 v122, v122, v124
	v_pack_b32_f16 v125, v128, v129
	v_pack_b32_f16 v124, v126, v127
	v_mfma_f32_32x32x8_f16 v[66:81], v[112:113], v[114:115], v[66:81]
	ds_read_b64 v[112:113], v232 offset:16384
	ds_read_b64 v[140:141], v222 offset:16384
	ds_read_b64 v[142:143], v223 offset:16384
	ds_read_b64 v[144:145], v224 offset:16384
	v_pack_b32_f16 v127, v132, v133
	v_pack_b32_f16 v126, v130, v131
	v_cvt_f16_f32_e32 v136, v186
	v_mfma_f32_32x32x8_f16 v[66:81], v[108:109], v[110:111], v[66:81]
	v_cvt_f16_f32_e32 v137, v187
	ds_read_b64 v[174:175], v212 offset:16384
	ds_read_b64 v[146:147], v225 offset:16384
	ds_read_b64 v[114:115], v233 offset:16384
	ds_read_b64 v[116:117], v234 offset:16384
	ds_read_b64 v[120:121], v236 offset:16384
	v_pack_b32_f16 v129, v136, v137
	v_mfma_f32_32x32x8_f16 v[34:49], v[162:163], v[122:123], v[34:49]
	v_pack_b32_f16 v128, v134, v135
	v_pack_b32_f16 v131, v189, v190
	v_pack_b32_f16 v130, v172, v188
	v_pack_b32_f16 v133, v193, v194
	v_pack_b32_f16 v132, v191, v192
	v_pack_b32_f16 v135, v197, v198
	v_pack_b32_f16 v134, v195, v196
	v_mfma_f32_32x32x8_f16 v[18:33], v[138:139], v[122:123], v[18:33]
	v_pack_b32_f16 v137, v202, v203
	v_pack_b32_f16 v136, v199, v201
	s_waitcnt vmcnt(12)
	ds_read_b64 v[0:1], v205 offset:16384
	v_mfma_f32_32x32x8_f16 v[34:49], v[164:165], v[124:125], v[34:49]
	s_waitcnt vmcnt(11)
	ds_read_b64 v[178:179], v206 offset:16384
	s_waitcnt vmcnt(10)
	ds_read_b64 v[180:181], v207 offset:16384
	s_waitcnt vmcnt(9)
	ds_read_b64 v[182:183], v208 offset:16384
	s_waitcnt vmcnt(8)
	ds_read_b64 v[184:185], v209 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[186:187], v210 offset:16384
	s_waitcnt vmcnt(6)
	ds_read_b64 v[176:177], v211 offset:16384
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[50:65], v[0:1], v[122:123], v[50:65]
	v_max_f32_e32 v0, v83, v83
	v_max_f32_e32 v1, v82, v82
	v_max_f32_e32 v0, v1, v0
	v_max3_f32 v0, v0, v84, v85
	v_max3_f32 v0, v0, v86, v87
	v_max3_f32 v0, v0, v88, v89
	v_max3_f32 v0, v0, v90, v91
	v_max3_f32 v0, v0, v92, v93
	s_waitcnt vmcnt(2)
	ds_read_b64 v[106:107], v229 offset:16384
	v_max3_f32 v0, v0, v94, v95
	s_waitcnt vmcnt(1)
	ds_read_b64 v[108:109], v230 offset:16384
	v_max3_f32 v0, v0, v96, v97
	v_max3_f32 v0, v0, v66, v67
	v_max3_f32 v0, v0, v68, v69
	v_max3_f32 v0, v0, v70, v71
	v_max3_f32 v0, v0, v72, v73
	v_max3_f32 v0, v0, v74, v75
	v_max3_f32 v0, v0, v76, v77
	v_max3_f32 v0, v0, v78, v79
	v_max3_f32 v0, v0, v80, v81
	ds_bpermute_b32 v1, v247, v0
	ds_read_b64 v[148:149], v226 offset:16384
	ds_read_b64 v[150:151], v227 offset:16384
	ds_read_b64 v[152:153], v228 offset:16384
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_f16 v[50:65], v[178:179], v[124:125], v[50:65]
	s_waitcnt vmcnt(0)
	ds_read_b64 v[110:111], v231 offset:16384
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_32x32x8_f16 v[2:17], v[106:107], v[122:123], v[2:17]
	v_max3_f32 v107, v173, v0, v1
	v_mov_b32_e32 v106, v81
	v_pk_mul_f32 v[0:1], v[106:107], s[2:3] op_sel_hi:[1,0]
	s_nop 0
	v_sub_f32_e32 v0, v0, v1
	v_fma_f32 v66, v66, s2, -v1
	v_exp_f32_e32 v106, v0
	v_mfma_f32_32x32x8_f16 v[2:17], v[108:109], v[124:125], v[2:17]
	v_sub_f32_e32 v0, v171, v1
	v_fma_f32 v81, v82, s2, -v1
	v_fma_f32 v82, v83, s2, -v1
	v_fma_f32 v83, v84, s2, -v1
	v_fma_f32 v84, v85, s2, -v1
	v_fma_f32 v85, v86, s2, -v1
	v_fma_f32 v86, v87, s2, -v1
	v_mfma_f32_32x32x8_f16 v[2:17], v[110:111], v[126:127], v[2:17]
	v_fma_f32 v87, v88, s2, -v1
	v_fma_f32 v88, v89, s2, -v1
	v_fma_f32 v89, v90, s2, -v1
	v_fma_f32 v90, v91, s2, -v1
	v_fma_f32 v91, v92, s2, -v1
	v_fma_f32 v92, v93, s2, -v1
	v_fma_f32 v93, v94, s2, -v1
	v_fma_f32 v94, v95, s2, -v1
	v_fma_f32 v95, v96, s2, -v1
	v_fma_f32 v96, v97, s2, -v1
	v_exp_f32_e32 v97, v66
	v_exp_f32_e32 v66, v0
	v_perm_b32 v0, v102, v98, s5
	v_mfma_f32_32x32x8_f16 v[18:33], v[140:141], v[124:125], v[18:33]
	ds_write_b32 v204, v0 offset:16384
	v_perm_b32 v0, v102, v98, s3
	ds_write_b32 v240, v0 offset:16384
	v_perm_b32 v0, v103, v99, s5
	ds_write_b32 v241, v0 offset:16384
	v_perm_b32 v0, v103, v99, s3
	v_exp_f32_e32 v81, v81
	v_exp_f32_e32 v82, v82
	ds_write_b32 v242, v0 offset:16384
	v_perm_b32 v0, v104, v100, s5
	v_exp_f32_e32 v83, v83
	ds_write_b32 v243, v0 offset:16384
	v_perm_b32 v0, v104, v100, s3
	v_exp_f32_e32 v84, v84
	ds_write_b32 v244, v0 offset:16384
	v_perm_b32 v0, v105, v101, s5
	v_exp_f32_e32 v85, v85
	ds_write_b32 v245, v0 offset:16384
	v_perm_b32 v0, v105, v101, s3
	v_mfma_f32_32x32x8_f16 v[2:17], v[112:113], v[128:129], v[2:17]
	v_exp_f32_e32 v86, v86
	ds_write_b32 v246, v0 offset:16384
	v_add_f32_e32 v0, v81, v82
	v_exp_f32_e32 v87, v87
	v_add_f32_e32 v0, v83, v0
	v_exp_f32_e32 v88, v88
	v_add_f32_e32 v0, v84, v0
	v_mfma_f32_32x32x8_f16 v[50:65], v[180:181], v[126:127], v[50:65]
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v0, v85, v0
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v0, v86, v0
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v0, v87, v0
	v_exp_f32_e32 v92, v92
	v_mfma_f32_32x32x8_f16 v[34:49], v[166:167], v[126:127], v[34:49]
	v_add_f32_e32 v0, v88, v0
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v0, v89, v0
	v_exp_f32_e32 v94, v94
	v_add_f32_e32 v0, v90, v0
	v_exp_f32_e32 v95, v95
	v_add_f32_e32 v0, v91, v0
	v_mfma_f32_32x32x8_f16 v[18:33], v[142:143], v[126:127], v[18:33]
	v_exp_f32_e32 v96, v96
	v_add_f32_e32 v0, v92, v0
	v_fma_f32 v67, v67, s2, -v1
	v_add_f32_e32 v0, v93, v0
	v_fma_f32 v68, v68, s2, -v1
	v_exp_f32_e32 v67, v67
	v_add_f32_e32 v0, v94, v0
	v_mfma_f32_32x32x8_f16 v[2:17], v[114:115], v[130:131], v[2:17]
	v_fma_f32 v69, v69, s2, -v1
	v_exp_f32_e32 v68, v68
	v_add_f32_e32 v0, v95, v0
	v_fma_f32 v70, v70, s2, -v1
	v_exp_f32_e32 v69, v69
	v_add_f32_e32 v0, v96, v0
	v_fma_f32 v71, v71, s2, -v1
	v_mfma_f32_32x32x8_f16 v[50:65], v[182:183], v[128:129], v[50:65]
	v_exp_f32_e32 v70, v70
	v_add_f32_e32 v0, v97, v0
	v_fma_f32 v72, v72, s2, -v1
	v_exp_f32_e32 v71, v71
	v_add_f32_e32 v0, v67, v0
	v_fma_f32 v73, v73, s2, -v1
	v_exp_f32_e32 v72, v72
	v_mfma_f32_32x32x8_f16 v[34:49], v[168:169], v[128:129], v[34:49]
	v_add_f32_e32 v0, v68, v0
	v_fma_f32 v74, v74, s2, -v1
	v_exp_f32_e32 v73, v73
	v_add_f32_e32 v0, v69, v0
	v_fma_f32 v75, v75, s2, -v1
	v_exp_f32_e32 v74, v74
	v_add_f32_e32 v0, v70, v0
	v_mfma_f32_32x32x8_f16 v[18:33], v[144:145], v[128:129], v[18:33]
	v_fma_f32 v76, v76, s2, -v1
	v_exp_f32_e32 v75, v75
	v_add_f32_e32 v0, v71, v0
	v_fma_f32 v77, v77, s2, -v1
	v_exp_f32_e32 v76, v76
	v_add_f32_e32 v0, v72, v0
	v_fma_f32 v78, v78, s2, -v1
	v_mfma_f32_32x32x8_f16 v[2:17], v[116:117], v[132:133], v[2:17]
	v_exp_f32_e32 v77, v77
	v_add_f32_e32 v0, v73, v0
	v_fma_f32 v79, v79, s2, -v1
	v_exp_f32_e32 v78, v78
	v_add_f32_e32 v0, v74, v0
	v_fma_f32 v80, v80, s2, -v1
	v_exp_f32_e32 v79, v79
	v_mfma_f32_32x32x8_f16 v[50:65], v[184:185], v[130:131], v[50:65]
	v_add_f32_e32 v0, v75, v0
	v_exp_f32_e32 v80, v80
	v_add_f32_e32 v0, v76, v0
	v_add_f32_e32 v0, v77, v0
	v_add_f32_e32 v0, v78, v0
	v_add_f32_e32 v0, v79, v0
	v_add_f32_e32 v0, v80, v0
	v_mfma_f32_32x32x8_f16 v[34:49], v[154:155], v[130:131], v[34:49]
	v_add_f32_e32 v0, v106, v0
	ds_bpermute_b32 v1, v247, v0
	v_cvt_f16_f32_e32 v67, v67
	v_cvt_f16_f32_e32 v100, v85
	v_cvt_f16_f32_e32 v101, v88
	v_cvt_f16_f32_e32 v155, v68
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v1
	v_mfma_f32_32x32x8_f16 v[18:33], v[146:147], v[130:131], v[18:33]
	v_cvt_f16_f32_e32 v1, v81
	v_cvt_f16_f32_e32 v81, v82
	v_cvt_f16_f32_e32 v82, v83
	v_cvt_f16_f32_e32 v83, v84
	v_cvt_f16_f32_e32 v162, v75
	v_pack_b32_f16 v104, v1, v81
	v_cvt_f16_f32_e32 v163, v76
	v_mfma_f32_32x32x8_f16 v[2:17], v[118:119], v[134:135], v[2:17]
	v_pack_b32_f16 v105, v82, v83
	v_cvt_f16_f32_e32 v164, v77
	v_cvt_f16_f32_e32 v165, v78
	v_cvt_f16_f32_e32 v166, v79
	s_barrier
	v_mfma_f32_32x32x8_f16 v[50:65], v[186:187], v[132:133], v[50:65]
	v_cvt_f16_f32_e32 v86, v86
	v_cvt_f16_f32_e32 v87, v87
	v_cvt_f16_f32_e32 v90, v90
	v_cvt_f16_f32_e32 v91, v91
	v_pack_b32_f16 v100, v100, v86
	v_pack_b32_f16 v101, v87, v101
	v_cvt_f16_f32_e32 v92, v92
	v_mfma_f32_32x32x8_f16 v[34:49], v[156:157], v[132:133], v[34:49]
	v_cvt_f16_f32_e32 v156, v69
	v_cvt_f16_f32_e32 v157, v70
	v_cvt_f16_f32_e32 v154, v97
	v_pack_b32_f16 v97, v91, v92
	v_pack_b32_f16 v91, v155, v156
	v_cvt_f16_f32_e32 v167, v80
	v_pack_b32_f16 v81, v163, v164
	v_mfma_f32_32x32x8_f16 v[18:33], v[148:149], v[132:133], v[18:33]
	v_cvt_f16_f32_e32 v106, v106
	v_pack_b32_f16 v82, v165, v166
	v_fmac_f32_e32 v0, v200, v66
	v_pack_b32_f16 v83, v167, v106
	v_mfma_f32_32x32x8_f16 v[2:17], v[120:121], v[136:137], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[134:135], v[50:65]
	s_nop 7
	s_nop 1
	v_pk_mul_f32 v[16:17], v[16:17], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[34:49], v[158:159], v[134:135], v[34:49]
	v_pk_mul_f32 v[2:3], v[2:3], v[66:67] op_sel_hi:[1,0]
	v_cvt_f16_f32_e32 v158, v71
	v_cvt_f16_f32_e32 v159, v72
	v_pack_b32_f16 v86, v157, v158
	v_mfma_f32_32x32x8_f16 v[18:33], v[150:151], v[134:135], v[18:33]
	v_cvt_f16_f32_e32 v150, v89
	v_cvt_f16_f32_e32 v151, v93
	v_cvt_f16_f32_e32 v93, v95
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[136:137], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[160:161], v[136:137], v[34:49]
	v_cvt_f16_f32_e32 v160, v73
	v_cvt_f16_f32_e32 v161, v74
	s_nop 7
	v_pk_mul_f32 v[64:65], v[64:65], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pack_b32_f16 v87, v159, v160
	v_pack_b32_f16 v80, v161, v162
	v_pk_mul_f32 v[60:61], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[18:33], v[152:153], v[136:137], v[18:33]
	v_cvt_f16_f32_e32 v152, v94
	ds_read_b64 v[146:147], v205 offset:16384
	ds_read_b64 v[148:149], v206 offset:16384
	ds_read_b64 v[144:145], v207 offset:16384
	ds_read_b64 v[142:143], v208 offset:16384
	ds_read_b64 v[140:141], v209 offset:16384
	ds_read_b64 v[138:139], v210 offset:16384
	ds_read_b64 v[136:137], v211 offset:16384
	ds_read_b64 v[134:135], v212 offset:16384
	ds_read_b64 v[132:133], v213 offset:16384
	ds_read_b64 v[130:131], v214 offset:16384
	ds_read_b64 v[128:129], v215 offset:16384
	ds_read_b64 v[126:127], v216 offset:16384
	ds_read_b64 v[124:125], v217 offset:16384
	ds_read_b64 v[122:123], v218 offset:16384
	ds_read_b64 v[120:121], v219 offset:16384
	ds_read_b64 v[118:119], v220 offset:16384
	ds_read_b64 v[116:117], v221 offset:16384
	ds_read_b64 v[114:115], v222 offset:16384
	ds_read_b64 v[112:113], v223 offset:16384
	ds_read_b64 v[110:111], v224 offset:16384
	ds_read_b64 v[108:109], v225 offset:16384
	ds_read_b64 v[102:103], v226 offset:16384
	ds_read_b64 v[98:99], v227 offset:16384
	ds_read_b64 v[94:95], v228 offset:16384
	ds_read_b64 v[88:89], v229 offset:16384
	ds_read_b64 v[84:85], v230 offset:16384
	ds_read_b64 v[78:79], v231 offset:16384
	ds_read_b64 v[76:77], v232 offset:16384
	ds_read_b64 v[74:75], v233 offset:16384
	ds_read_b64 v[72:73], v234 offset:16384
	ds_read_b64 v[68:69], v235 offset:16384
	ds_read_b64 v[70:71], v236 offset:16384
	v_cvt_f16_f32_e32 v153, v96
	v_pack_b32_f16 v96, v150, v90
	v_pack_b32_f16 v92, v151, v152
	v_pack_b32_f16 v90, v154, v67
	v_pack_b32_f16 v93, v93, v153
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 v[2:17], v[88:89], v[104:105], v[2:17]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_pk_mul_f32 v[58:59], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[84:85], v[100:101], v[2:17]
	v_pk_mul_f32 v[50:51], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[78:79], v[96:97], v[2:17]
	v_pk_mul_f32 v[36:37], v[36:37], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[76:77], v[92:93], v[2:17]
	v_pk_mul_f32 v[22:23], v[22:23], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[66:67] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[90:91], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[72:73], v[86:87], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[80:81], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[70:71], v[82:83], v[2:17]
	scratch_load_dword v70, off, off offset:284 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v1, v70, 2, 0
	v_mfma_f32_32x32x8_f16 v[50:65], v[146:147], v[104:105], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[132:133], v[104:105], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[116:117], v[104:105], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[148:149], v[100:101], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[130:131], v[100:101], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[114:115], v[100:101], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[144:145], v[96:97], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[128:129], v[96:97], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[112:113], v[96:97], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[142:143], v[92:93], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[126:127], v[92:93], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[110:111], v[92:93], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[140:141], v[90:91], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[124:125], v[90:91], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[108:109], v[90:91], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[138:139], v[86:87], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[122:123], v[86:87], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[102:103], v[86:87], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[136:137], v[80:81], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[120:121], v[80:81], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[98:99], v[80:81], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[134:135], v[82:83], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[118:119], v[82:83], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[94:95], v[82:83], v[18:33]
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	scratch_load_dword v69, off, off offset:280 ; 4-byte Folded Reload
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v0
	v_mov_b32_e32 v67, 0x42000000
	v_or_b32_e32 v66, s34, v70
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v0, v68
	v_log_f32_e32 v68, v68
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v67, 0, v67, vcc
	v_cmp_gt_i32_e64 s[8:9], s2, v66
	v_sub_f32_e32 v66, v68, v67
	v_add_f32_e32 v66, v107, v66
	ds_write_b32 v1, v66
	v_mov_b32_e32 v66, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_sub_i32 s2, 0x4000, s34
	v_bfrev_b32_e32 v68, 1
	s_and_b32 s5, s16, 0xffff
	s_mov_b32 s6, s14
	s_mov_b32 s7, s15
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v66, v66, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v67, 0, v66
	ds_read_b32 v67, v67
	v_cmp_lt_i32_sdwa s[2:3], v69, s2 src0_sel:BYTE_0 src1_sel:DWORD
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v66, v68, v66, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v66, s[4:7], 0 offen
	s_cbranch_execz .LBB0_5
	s_branch .LBB0_6
.LBB0_4:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_5:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v0
	v_mov_b32_e32 v66, 0x42000000
	s_and_b32 s5, s16, 0xffff
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v0, v67
	v_log_f32_e32 v67, v67
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_sub_f32_e32 v66, v67, v66
	v_add_f32_e32 v66, v107, v66
	ds_write_b32 v1, v66
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v66, off, off offset:280 ; 4-byte Folded Reload
	v_mov_b32_e32 v1, 2
	v_bfrev_b32_e32 v67, 1
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v1, v1, v66 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v1
	ds_read_b32 v66, v66
	v_cndmask_b32_e64 v1, v67, v1, s[0:1]
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v1, s[4:7], 0 offen
.LBB0_6:                                ; %.critedge
	v_div_scale_f32 v1, s[0:1], v0, v0, 1.0
	v_rcp_f32_e32 v1, v1
	v_div_scale_f32 v66, vcc, 1.0, v0, 1.0
	v_mov_b32_e32 v67, v16
	v_mul_f32_e32 v1, v66, v1
	v_mov_b32_e32 v66, v15
	s_nop 0
	v_div_fmas_f32 v1, 0, 0, v1
	v_div_fixup_f32 v0, v1, v0, 1.0
	v_fma_mixlo_f16 v68, v0, v17, 0
	v_pk_mul_f32 v[16:17], v[0:1], v[66:67] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v66, v0, v14, 0
	v_mov_b32_e32 v14, v11
	v_mov_b32_e32 v15, v12
	v_fma_mixlo_f16 v67, v0, v13, 0
	v_pk_mul_f32 v[12:13], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v14, v0, v10, 0
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v8
	v_fma_mixlo_f16 v15, v0, v9, 0
	v_pk_mul_f32 v[8:9], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v10, v0, v6, 0
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v4
	v_fma_mixlo_f16 v11, v0, v5, 0
	v_pk_mul_f32 v[4:5], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v6, v0, v2, 0
	v_mov_b32_e32 v2, v31
	v_mov_b32_e32 v3, v32
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v31, v3
	v_cvt_f16_f32_e32 v32, v2
	v_mov_b32_e32 v2, v27
	v_mov_b32_e32 v3, v28
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v27, v3
	v_cvt_f16_f32_e32 v28, v2
	v_mov_b32_e32 v2, v23
	v_mov_b32_e32 v3, v24
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v23, v3
	v_cvt_f16_f32_e32 v24, v2
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v3, v20
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v19, v3
	v_cvt_f16_f32_e32 v20, v2
	v_mov_b32_e32 v2, v47
	v_mov_b32_e32 v3, v48
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v47, v3
	v_cvt_f16_f32_e32 v48, v2
	v_mov_b32_e32 v2, v43
	v_mov_b32_e32 v3, v44
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v43, v3
	v_cvt_f16_f32_e32 v44, v2
	v_mov_b32_e32 v2, v39
	v_mov_b32_e32 v3, v40
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v39, v3
	v_cvt_f16_f32_e32 v40, v2
	v_mov_b32_e32 v2, v35
	v_mov_b32_e32 v3, v36
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v35, v3
	v_cvt_f16_f32_e32 v36, v2
	v_mov_b32_e32 v2, v63
	v_mov_b32_e32 v3, v64
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v63, v3
	v_cvt_f16_f32_e32 v64, v2
	v_mov_b32_e32 v2, v59
	v_mov_b32_e32 v3, v60
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v59, v3
	v_cvt_f16_f32_e32 v60, v2
	v_mov_b32_e32 v2, v55
	v_mov_b32_e32 v3, v56
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v55, v3
	v_cvt_f16_f32_e32 v56, v2
	v_fma_mixlo_f16 v1, v0, v53, 0
	v_mov_b32_e32 v2, v51
	v_mov_b32_e32 v3, v52
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v2, v2
	v_fma_mixlo_f16 v7, v0, v33, 0
	v_fma_mixlo_f16 v30, v0, v30, 0
	v_fma_mixlo_f16 v29, v0, v29, 0
	v_fma_mixlo_f16 v26, v0, v26, 0
	v_fma_mixlo_f16 v25, v0, v25, 0
	v_fma_mixlo_f16 v22, v0, v22, 0
	v_fma_mixlo_f16 v21, v0, v21, 0
	v_fma_mixlo_f16 v18, v0, v18, 0
	v_fma_mixlo_f16 v33, v0, v49, 0
	v_fma_mixlo_f16 v46, v0, v46, 0
	v_fma_mixlo_f16 v45, v0, v45, 0
	v_fma_mixlo_f16 v42, v0, v42, 0
	v_fma_mixlo_f16 v41, v0, v41, 0
	v_fma_mixlo_f16 v38, v0, v38, 0
	v_fma_mixlo_f16 v37, v0, v37, 0
	v_fma_mixlo_f16 v34, v0, v34, 0
	v_fma_mixlo_f16 v49, v0, v65, 0
	v_fma_mixlo_f16 v62, v0, v62, 0
	v_fma_mixlo_f16 v61, v0, v61, 0
	v_fma_mixlo_f16 v58, v0, v58, 0
	v_fma_mixlo_f16 v57, v0, v57, 0
	v_fma_mixlo_f16 v54, v0, v54, 0
	v_fma_mixlo_f16 v0, v0, v50, 0
	v_pack_b32_f16 v0, v0, v2
	scratch_load_dword v2, off, off offset:288 ; 4-byte Folded Reload
	s_mul_i32 s0, s25, s18
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s26, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s27, s34
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_cvt_f16_f32_e32 v3, v3
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s27, 0x3fff
	v_mul_lo_u32 v50, s27, v70
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s4, 0x5040100
	s_or_b32 s1, s2, s1
	v_perm_b32 v1, v1, v3, s4
	v_bfrev_b32_e32 v3, 1
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v5, v5
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v12, v12
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v16, v16
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v2, v50, v2, 1
	v_cndmask_b32_e64 v50, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_add_u32_e32 v50, 16, v2
	v_pack_b32_f16 v0, v54, v56
	v_perm_b32 v1, v57, v55, s4
	v_cndmask_b32_e64 v50, v3, v50, s[8:9]
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_add_u32_e32 v50, 32, v2
	v_pack_b32_f16 v0, v58, v60
	v_perm_b32 v1, v61, v59, s4
	v_cndmask_b32_e64 v50, v3, v50, s[8:9]
	buffer_store_dwordx2 v[0:1], v50, s[0:3], 0 offen
	v_perm_b32 v1, v49, v63, s4
	v_add_u32_e32 v49, 48, v2
	v_pack_b32_f16 v0, v62, v64
	v_cndmask_b32_e64 v49, v3, v49, s[8:9]
	buffer_store_dwordx2 v[0:1], v49, s[0:3], 0 offen
	v_pack_b32_f16 v0, v34, v36
	v_add_u32_e32 v34, 64, v2
	v_perm_b32 v1, v37, v35, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_add_u32_e32 v34, 0x50, v2
	v_pack_b32_f16 v0, v38, v40
	v_perm_b32 v1, v41, v39, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_add_u32_e32 v34, 0x60, v2
	v_pack_b32_f16 v0, v42, v44
	v_perm_b32 v1, v45, v43, s4
	v_cndmask_b32_e64 v34, v3, v34, s[8:9]
	buffer_store_dwordx2 v[0:1], v34, s[0:3], 0 offen
	v_perm_b32 v1, v33, v47, s4
	v_add_u32_e32 v33, 0x70, v2
	v_pack_b32_f16 v0, v46, v48
	v_cndmask_b32_e64 v33, v3, v33, s[8:9]
	buffer_store_dwordx2 v[0:1], v33, s[0:3], 0 offen
	v_pack_b32_f16 v0, v18, v20
	v_add_u32_e32 v18, 0x80, v2
	v_perm_b32 v1, v21, v19, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_add_u32_e32 v18, 0x90, v2
	v_pack_b32_f16 v0, v22, v24
	v_perm_b32 v1, v25, v23, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_add_u32_e32 v18, 0xa0, v2
	v_pack_b32_f16 v0, v26, v28
	v_perm_b32 v1, v29, v27, s4
	v_cndmask_b32_e64 v18, v3, v18, s[8:9]
	buffer_store_dwordx2 v[0:1], v18, s[0:3], 0 offen
	v_perm_b32 v1, v7, v31, s4
	v_add_u32_e32 v7, 0xb0, v2
	v_pack_b32_f16 v0, v30, v32
	v_cndmask_b32_e64 v7, v3, v7, s[8:9]
	buffer_store_dwordx2 v[0:1], v7, s[0:3], 0 offen
	v_pack_b32_f16 v0, v6, v4
	v_add_u32_e32 v4, 0xc0, v2
	v_perm_b32 v1, v11, v5, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xd0, v2
	v_pack_b32_f16 v0, v10, v8
	v_perm_b32 v1, v15, v9, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xe0, v2
	v_pack_b32_f16 v0, v14, v12
	v_perm_b32 v1, v67, v13, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	v_add_u32_e32 v2, 0xf0, v2
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_pack_b32_f16 v0, v66, v16
	v_perm_b32 v1, v68, v17, s4
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 300
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
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
	.size	attn_fwd, .Lfunc_end0-attn_fwd
	.cfi_endproc
                                        ; -- End function
	.set attn_fwd.num_vgpr, 256
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 56
	.set attn_fwd.private_seg_size, 300
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 17636
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 300
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
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
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"flash-attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton/fa" ; string offset=26
.Linfo_string3:
	.asciz	"attn_fwd"                      ; string offset=57
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .max_flat_workgroup_size: 512
    .name:           attn_fwd
    .private_segment_fixed_size: 300
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 74
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
