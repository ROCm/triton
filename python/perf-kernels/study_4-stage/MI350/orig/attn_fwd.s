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
	.file	1 "/var/lib/jenkins/OAI-triton/python/../fa" "flash-attention.py"
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
	v_mov_b32_e32 v69, v0
	s_lshl_b64 s[0:1], s[0:1], 1
	v_and_b32_e32 v34, 0x80, v69
	v_and_b32_e32 v1, 0x100, v69
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s14, s34
	v_bfe_u32 v0, v69, 4, 3
	v_lshrrev_b32_e32 v37, 4, v34
	v_lshrrev_b32_e32 v36, 4, v1
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	v_or3_b32 v38, v0, v37, v36
	s_lshl_b32 s19, s14, 5
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v0, 3, v69
	s_add_u32 s12, s2, s0
	v_and_b32_e32 v86, 0x78, v0
	s_mul_i32 s36, s15, s18
	scratch_store_dword off, v1, off offset:160 ; 4-byte Folded Spill
	s_addc_u32 s13, s3, s1
	scratch_store_dword off, v0, off offset:92 ; 4-byte Folded Spill
	v_mad_u64_u32 v[0:1], s[0:1], s14, v38, v[86:87]
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[0:1], s[36:37], 1
	s_add_u32 s2, s4, s0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s38, s20, s17
	s_addc_u32 s3, s5, s1
	s_ashr_i32 s39, s38, 31
	s_lshl_b64 s[0:1], s[38:39], 1
	s_add_u32 s0, s2, s0
	s_mul_i32 s2, s22, s18
	s_addc_u32 s20, s3, s1
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s1, s6, s2
	s_mul_i32 s2, s23, s17
	s_addc_u32 s6, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s28, s1, s2
	s_addc_u32 s16, s6, s3
	s_and_b32 s2, s14, 0x3fff
	v_or_b32_e32 v2, s34, v38
	s_movk_i32 s1, 0x4000
	s_bitset1_b32 s2, 14
	v_or_b32_e32 v3, 32, v2
	v_add_u32_e32 v1, s19, v0
	s_and_b32 s3, s13, 0xffff
	s_lshl_b32 s2, s2, 16
	v_lshlrev_b32_e32 v0, 1, v0
	v_bfrev_b32_e32 v39, 1
	v_cmp_gt_i32_e32 vcc, s1, v2
	v_add_u32_e32 v12, s19, v1
	s_or_b32 s13, s3, s2
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_cndmask_b32_e32 v0, v39, v0, vcc
	v_lshlrev_b32_e32 v1, 1, v1
	v_cmp_gt_i32_e32 vcc, s1, v3
	v_or_b32_e32 v10, 64, v2
	v_or_b32_e32 v11, 0x60, v2
	v_or_b32_e32 v18, 0x80, v2
	v_or_b32_e32 v19, 0xa0, v2
	v_or_b32_e32 v26, 0xc0, v2
	v_or_b32_e32 v30, 0xe0, v2
	v_cndmask_b32_e32 v1, v39, v1, vcc
	buffer_load_dwordx4 v[2:5], v0, s[12:15], 0 offen
	buffer_load_dwordx4 v[6:9], v1, s[12:15], 0 offen
	v_add_u32_e32 v13, s19, v12
	v_lshlrev_b32_e32 v0, 1, v12
	v_cmp_gt_i32_e32 vcc, s1, v10
	v_lshlrev_b32_e32 v1, 1, v13
	v_add_u32_e32 v20, s19, v13
	v_cndmask_b32_e32 v0, v39, v0, vcc
	v_cmp_gt_i32_e32 vcc, s1, v11
	v_add_u32_e32 v21, s19, v20
	v_add_u32_e32 v31, s19, v21
	v_cndmask_b32_e32 v1, v39, v1, vcc
	buffer_load_dwordx4 v[10:13], v0, s[12:15], 0 offen
	buffer_load_dwordx4 v[14:17], v1, s[12:15], 0 offen
	v_lshlrev_b32_e32 v0, 1, v20
	v_cmp_gt_i32_e32 vcc, s1, v18
	v_lshlrev_b32_e32 v1, 1, v21
	v_lshrrev_b32_e32 v42, 1, v69
	v_cndmask_b32_e32 v0, v39, v0, vcc
	v_cmp_gt_i32_e32 vcc, s1, v19
	v_mad_u64_u32 v[220:221], s[2:3], s21, v38, v[86:87]
	s_nop 0
	v_cndmask_b32_e32 v1, v39, v1, vcc
	buffer_load_dwordx4 v[18:21], v0, s[12:15], 0 offen
	buffer_load_dwordx4 v[22:25], v1, s[12:15], 0 offen
	v_lshlrev_b32_e32 v0, 1, v31
	v_cmp_gt_i32_e32 vcc, s1, v26
	v_bitop3_b32 v1, v42, v86, 40 bitop3:0x6c
	v_lshrrev_b32_e32 v34, 1, v34
	v_cndmask_b32_e32 v0, v39, v0, vcc
	buffer_load_dwordx4 v[26:29], v0, s[12:15], 0 offen
	v_add_lshl_u32 v0, v31, s19, 1
	v_cmp_gt_i32_e32 vcc, s1, v30
	v_and_b32_e32 v41, 31, v69
	s_movk_i32 s1, 0xe0
	v_cndmask_b32_e32 v0, v39, v0, vcc
	buffer_load_dwordx4 v[30:33], v0, s[12:15], 0 offen
	v_and_b32_e32 v0, 32, v69
	v_lshrrev_b32_e32 v87, 1, v0
	s_barrier
	v_xor_b32_e32 v1, v1, v87
	scratch_store_dword off, v38, off offset:164 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v38, 7, v38
	v_xor_b32_e32 v35, v1, v34
	v_bitop3_b32 v1, v38, v1, v34 bitop3:0xf6
	v_lshlrev_b32_e32 v40, 1, v1
	v_add_u32_e32 v1, 0, v40
	scratch_store_dword off, v34, off offset:184 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v34, 2, v69
	v_and_b32_e32 v237, 63, v69
	v_lshl_add_u32 v43, s21, 5, v220
	s_and_b32 s2, s20, 0xffff
	s_lshl_b32 s6, s21, 6
	s_mov_b32 s3, s15
	s_ashr_i32 s7, s6, 31
	s_lshl_b32 s23, s24, 3
	s_lshl_b32 s22, s24, 6
	v_lshlrev_b32_e32 v242, 7, v41
	s_mov_b32 s30, s14
	s_mov_b32 s31, s15
	v_bitop3_b32 v214, v34, v86, 8 bitop3:0x6c
	v_mov_b32_e32 v215, 1.0
	s_waitcnt vmcnt(9)
	ds_write_b128 v1, v[2:5]
	v_or_b32_e32 v2, 0x1000, v38
	v_or_b32_e32 v1, v2, v35
	v_lshlrev_b32_e32 v4, 1, v1
	v_add_u32_e32 v1, 0, v4
	s_waitcnt vmcnt(8)
	ds_write_b128 v1, v[6:9]
	v_or_b32_e32 v1, 0x4000, v40
	v_add_u32_e32 v1, 0, v1
	v_and_or_b32 v3, v42, s1, v41
	v_lshlrev_b32_e32 v3, 7, v3
	v_or_b32_e32 v2, v2, v86
	s_waitcnt vmcnt(7)
	ds_write_b128 v1, v[10:13]
	v_or_b32_e32 v1, 0x6000, v40
	v_add_u32_e32 v1, 0, v1
	s_waitcnt vmcnt(6)
	ds_write_b128 v1, v[14:17]
	v_or_b32_e32 v1, 0x8000, v40
	v_add_u32_e32 v1, 0, v1
	v_lshlrev_b32_e32 v15, 1, v2
	s_and_b32 s1, s21, 0x3fff
	s_bitset1_b32 s1, 14
	s_lshl_b32 s19, s1, 16
	s_waitcnt vmcnt(5)
	ds_write_b128 v1, v[18:21]
	v_or_b32_e32 v1, 0xa000, v40
	v_add_u32_e32 v1, 0, v1
	s_waitcnt vmcnt(4)
	ds_write_b128 v1, v[22:25]
	v_or_b32_e32 v1, 0xc000, v40
	v_add_u32_e32 v1, 0, v1
	s_or_b32 s1, s2, s19
	s_waitcnt vmcnt(3)
	ds_write_b128 v1, v[26:29]
	v_or_b32_e32 v1, 0xe000, v40
	v_add_u32_e32 v1, 0, v1
	s_mov_b32 s2, s14
	s_waitcnt vmcnt(2)
	ds_write_b128 v1, v[30:33]
	v_and_b32_e32 v1, 8, v34
	v_or_b32_e32 v6, 16, v1
	v_or_b32_e32 v26, 32, v1
	v_or_b32_e32 v27, 48, v1
	v_or_b32_e32 v28, 64, v1
	v_or_b32_e32 v29, 0x50, v1
	v_or_b32_e32 v30, 0x60, v1
	v_or_b32_e32 v31, 0x70, v1
	v_bitop3_b32 v5, v3, v1, v86 bitop3:0xf6
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v42, off offset:188 ; 4-byte Folded Spill
	v_bitop3_b32 v7, v3, v6, v86 bitop3:0xf6
	v_bitop3_b32 v8, v3, v26, v86 bitop3:0xf6
	v_bitop3_b32 v9, v3, v27, v86 bitop3:0xf6
	v_bitop3_b32 v10, v3, v28, v86 bitop3:0xf6
	v_bitop3_b32 v11, v3, v29, v86 bitop3:0xf6
	v_bitop3_b32 v12, v3, v30, v86 bitop3:0xf6
	v_bitop3_b32 v13, v3, v31, v86 bitop3:0xf6
	v_lshl_add_u32 v3, v5, 1, 0
	v_lshl_add_u32 v5, v7, 1, 0
	ds_read_b128 v[110:113], v3
	ds_read_b128 v[106:109], v5
	v_lshl_add_u32 v3, v8, 1, 0
	v_lshl_add_u32 v5, v9, 1, 0
	ds_read_b128 v[102:105], v3
	ds_read_b128 v[98:101], v5
	v_or_b32_e32 v3, v38, v86
	v_lshlrev_b32_e32 v14, 1, v3
	v_sub_u32_e32 v2, v40, v14
	v_ashrrev_i16_e32 v3, 15, v2
	v_lshrrev_b16_e32 v3, 12, v3
	v_add_u16_e32 v2, v2, v3
	v_ashrrev_i16_e32 v2, 4, v2
	v_add_u32_sdwa v2, v237, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshlrev_b32_e32 v3, 2, v2
	ds_bpermute_b32 v8, v3, v220
	v_lshrrev_b64 v[2:3], v2, exec
	v_sub_u32_e32 v3, v4, v15
	v_ashrrev_i16_e32 v4, 15, v3
	v_lshrrev_b16_e32 v4, 12, v4
	v_add_u16_e32 v3, v3, v4
	v_ashrrev_i16_e32 v3, 4, v3
	v_add_u32_sdwa v3, v237, sext(v3) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshlrev_b32_e32 v4, 2, v3
	v_and_b32_e32 v9, 1, v2
	v_lshrrev_b64 v[2:3], v3, exec
	v_lshl_add_u32 v5, v10, 1, 0
	v_and_b32_e32 v10, 1, v2
	v_sub_u32_e32 v2, v35, v86
	v_ashrrev_i32_e32 v2, 3, v2
	v_add_u32_e32 v2, v2, v237
	v_lshlrev_b32_e32 v241, 2, v2
	v_lshl_add_u32 v7, v11, 1, 0
	ds_bpermute_b32 v11, v241, v220
	ds_bpermute_b32 v4, v4, v43
	v_lshrrev_b64 v[2:3], v2, exec
	ds_read_b128 v[126:129], v5
	ds_read_b128 v[118:121], v7
	v_lshl_add_u32 v3, v12, 1, 0
	s_waitcnt lgkmcnt(3)
	v_lshlrev_b32_e32 v7, 1, v11
	scratch_store_dword off, v14, off offset:8 ; 4-byte Folded Spill
	v_add_u32_e32 v11, 0, v14
	v_lshlrev_b32_e32 v8, 1, v8
	v_lshl_add_u32 v5, v13, 1, 0
	v_readfirstlane_b32 s33, v11
	ds_read_b128 v[122:125], v3
	ds_read_b128 v[114:117], v5
	v_cmp_eq_u32_e32 vcc, 1, v9
	v_add_u32_e32 v3, 0, v15
	s_waitcnt lgkmcnt(4)
	v_lshlrev_b32_e32 v4, 1, v4
	v_cndmask_b32_e32 v64, v39, v8, vcc
	s_mov_b32 m0, s33
	v_cmp_eq_u32_e32 vcc, 1, v10
	v_readfirstlane_b32 s35, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v64, s[0:3], 0 offen lds
	v_cndmask_b32_e32 v65, v39, v4, vcc
	s_mov_b32 m0, s35
	ds_bpermute_b32 v4, v241, v43
	buffer_load_dwordx4 v65, s[0:3], 0 offen lds
	s_lshl_b64 s[2:3], s[6:7], 1
	s_add_u32 s12, s0, s2
	v_and_b32_e32 v2, 1, v2
	v_add_u32_e32 v11, 0x4000, v11
	s_addc_u32 s7, s20, s3
	v_readfirstlane_b32 s29, v11
	s_and_b32 s0, s7, 0xffff
	v_cmp_eq_u32_e32 vcc, 1, v2
	s_or_b32 s13, s0, s19
	s_mov_b32 m0, s29
	v_cndmask_b32_e32 v2, v39, v7, vcc
	scratch_store_dword off, v15, off offset:12 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v2, s[12:15], 0 offen lds
	v_add_u32_e32 v2, 0x4000, v3
	scratch_store_dword off, v43, off offset:4 ; 4-byte Folded Spill
	v_readfirstlane_b32 s0, v2
	v_lshlrev_b32_e32 v2, 1, v4
	v_cndmask_b32_e32 v2, v39, v2, vcc
	s_mov_b32 m0, s0
	v_bitop3_b32 v6, v6, v242, v86 bitop3:0xde
	buffer_load_dwordx4 v2, s[12:15], 0 offen lds
	v_bitop3_b32 v2, v1, v242, v86 bitop3:0xde
	v_lshlrev_b32_e32 v35, 1, v2
	v_add_u32_e32 v66, 0, v35
	s_waitcnt vmcnt(2)
	s_barrier
	ds_read_b128 v[2:5], v66
	ds_read_b128 v[18:21], v66 offset:8192
	v_lshlrev_b32_e32 v38, 1, v6
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[2:5], v[110:113], 0
	scratch_store_dword off, v41, off offset:180 ; 4-byte Folded Spill
	v_add_u32_e32 v70, 0, v38
	ds_read_b128 v[22:25], v70
	ds_read_b128 v[44:47], v70 offset:8192
	s_mov_b32 s20, 0x3e0293ee
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[106:109], v[2:17]
	v_bitop3_b32 v22, v26, v242, v86 bitop3:0xde
	v_lshlrev_b32_e32 v40, 1, v22
	v_add_u32_e32 v71, 0, v40
	ds_read_b128 v[22:25], v71
	ds_read_b128 v[48:51], v71 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[102:105], v[2:17]
	v_bitop3_b32 v22, v27, v242, v86 bitop3:0xde
	v_lshlrev_b32_e32 v41, 1, v22
	v_add_u32_e32 v72, 0, v41
	ds_read_b128 v[22:25], v72
	ds_read_b128 v[52:55], v72 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[98:101], v[2:17]
	v_bitop3_b32 v22, v28, v242, v86 bitop3:0xde
	v_lshlrev_b32_e32 v42, 1, v22
	v_add_u32_e32 v73, 0, v42
	ds_read_b128 v[22:25], v73
	ds_read_b128 v[56:59], v73 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[126:129], v[2:17]
	v_bitop3_b32 v22, v29, v242, v86 bitop3:0xde
	v_lshlrev_b32_e32 v43, 1, v22
	v_add_u32_e32 v74, 0, v43
	ds_read_b128 v[22:25], v74
	ds_read_b128 v[60:63], v74 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[118:121], v[2:17]
	v_bitop3_b32 v22, v30, v242, v86 bitop3:0xde
	v_lshlrev_b32_e32 v67, 1, v22
	v_add_u32_e32 v75, 0, v67
	ds_read_b128 v[22:25], v75
	ds_read_b128 v[78:81], v75 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[122:125], v[2:17]
	v_bitop3_b32 v22, v31, v242, v86 bitop3:0xde
	v_lshlrev_b32_e32 v68, 1, v22
	v_add_u32_e32 v76, 0, v68
	ds_read_b128 v[22:25], v76
	ds_read_b128 v[82:85], v76 offset:8192
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[2:17], v[22:25], v[114:117], v[2:17]
	v_mfma_f32_32x32x16_f16 v[18:33], v[18:21], v[110:113], 0
	s_nop 7
	s_nop 2
	v_fma_f32 v4, v4, s20, 0
	v_fma_f32 v5, v5, s20, 0
	v_fma_f32 v6, v6, s20, 0
	v_fma_f32 v7, v7, s20, 0
	v_fma_f32 v8, v8, s20, 0
	v_fma_f32 v9, v9, s20, 0
	v_fma_f32 v10, v10, s20, 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[44:47], v[106:109], v[18:33]
	v_fma_f32 v44, v2, s20, 0
	v_lshlrev_b32_e32 v2, 1, v69
	v_fma_f32 v45, v3, s20, 0
	scratch_store_dword off, v2, off offset:132 ; 4-byte Folded Spill
	v_and_b32_e32 v2, 30, v2
	v_lshlrev_b32_e32 v46, 1, v0
	v_or_b32_e32 v47, 32, v2
	v_mfma_f32_32x32x16_f16 v[18:33], v[48:51], v[102:105], v[18:33]
	v_fma_f32 v11, v11, s20, 0
	v_fma_f32 v12, v12, s20, 0
	v_fma_f32 v13, v13, s20, 0
	v_fma_f32 v14, v14, s20, 0
	v_fma_f32 v15, v15, s20, 0
	v_fma_f32 v16, v16, s20, 0
	v_fma_f32 v17, v17, s20, 0
	v_mfma_f32_32x32x16_f16 v[18:33], v[52:55], v[98:101], v[18:33]
	v_mfma_f32_32x32x16_f16 v[18:33], v[56:59], v[126:129], v[18:33]
	v_and_b32_e32 v59, 16, v69
	v_lshlrev_b32_e32 v3, 1, v59
	v_or3_b32 v50, v2, v3, v46
	v_lshrrev_b32_e32 v2, 4, v69
	v_bitop3_b32 v47, v47, v46, v3 bitop3:0xde
	v_and_or_b32 v2, v2, 4, v37
	v_bitop3_b32 v47, v2, v47, v36 bitop3:0x36
	v_bitop3_b32 v2, v2, v50, v36 bitop3:0x36
	scratch_store_dword off, v2, off offset:20 ; 4-byte Folded Spill
	v_sub_u32_e32 v2, v2, v50
	v_lshrrev_b16_e32 v3, 7, v2
	v_and_b32_e32 v3, 1, v3
	v_add_u16_e32 v2, v2, v3
	v_mov_b32_e32 v36, 1
	v_ashrrev_i16_sdwa v2, v36, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_sdwa v54, v237, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshrrev_b64 v[2:3], v54, exec
	v_and_b32_e32 v37, 1, v2
	v_lshrrev_b32_e32 v2, 6, v69
	v_lshlrev_b32_e32 v3, 7, v2
	scratch_store_dword off, v3, off offset:24 ; 4-byte Folded Spill
	v_or_b32_e32 v3, v50, v3
	scratch_store_dword off, v3, off offset:28 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v57, 1, v3
	v_sub_u32_e32 v3, v47, v50
	scratch_store_dword off, v47, off offset:16 ; 4-byte Folded Spill
	v_lshrrev_b16_e32 v47, 7, v3
	v_and_b32_e32 v47, 1, v47
	v_add_u16_e32 v3, v3, v47
	v_ashrrev_i16_sdwa v3, v36, sext(v3) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_sdwa v48, v237, sext(v3) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_mad_u64_u32 v[222:223], s[0:1], s24, v2, v[50:51]
	v_mov_b32_e32 v36, v50
	v_add_u32_e32 v58, s23, v222
	v_lshlrev_b32_e32 v56, 2, v48
	scratch_store_dwordx2 off, v[36:37], off offset:144 ; 8-byte Folded Spill
	ds_bpermute_b32 v36, v56, v58
	v_add_u32_e32 v46, 0, v57
	v_add_u32_e32 v2, 0x8000, v46
	v_lshlrev_b32_e32 v55, 2, v54
	v_readfirstlane_b32 s13, v2
	v_mov_b32_e32 v2, v48
	scratch_store_dwordx2 off, v[2:3], off offset:108 ; 8-byte Folded Spill
	v_lshrrev_b64 v[2:3], v48, exec
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v36
	v_or_b32_e32 v36, 0x800, v57
	scratch_store_dword off, v36, off offset:120 ; 4-byte Folded Spill
	v_add_u32_e32 v36, 0, v36
	v_add_u32_e32 v47, 0x8000, v36
	v_mov_b32_e32 v52, v54
	v_readfirstlane_b32 s40, v47
	v_or_b32_e32 v47, 0x1000, v57
	scratch_store_dword off, v47, off offset:124 ; 4-byte Folded Spill
	v_add_u32_e32 v47, 0, v47
	v_add_u32_e32 v48, 0x8000, v47
	v_mfma_f32_32x32x16_f16 v[18:33], v[60:63], v[118:121], v[18:33]
	v_readfirstlane_b32 s41, v48
	v_or_b32_e32 v48, 0x1800, v57
	scratch_store_dword off, v48, off offset:128 ; 4-byte Folded Spill
	v_add_u32_e32 v48, 0, v48
	v_add_u32_e32 v49, 0x8000, v48
	scratch_store_dwordx2 off, v[52:53], off offset:96 ; 8-byte Folded Spill
	v_readfirstlane_b32 s42, v49
	v_or_b32_e32 v49, 0x2000, v57
	scratch_store_dword off, v49, off offset:136 ; 4-byte Folded Spill
	v_add_u32_e32 v49, 0, v49
	v_add_u32_e32 v50, 0x8000, v49
	v_or_b32_e32 v52, 0x3000, v57
	v_readfirstlane_b32 s43, v50
	v_or_b32_e32 v50, 0x2800, v57
	scratch_store_dword off, v50, off offset:140 ; 4-byte Folded Spill
	v_add_u32_e32 v50, 0, v50
	v_add_u32_e32 v51, 0x8000, v50
	scratch_store_dword off, v52, off offset:152 ; 4-byte Folded Spill
	v_readfirstlane_b32 s44, v51
	ds_bpermute_b32 v51, v55, v222
	v_add_u32_e32 v52, 0, v52
	v_cmp_eq_u32_e32 vcc, 1, v37
	v_add_u32_e32 v53, 0x8000, v52
	v_and_b32_e32 v2, 1, v2
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v51, 1, v51
	v_cndmask_b32_e32 v37, v39, v51, vcc
	v_add_u32_e32 v51, s23, v58
	v_readfirstlane_b32 s45, v53
	v_add_u32_e32 v53, s23, v51
	v_add_u32_e32 v54, s23, v53
	v_cmp_eq_u32_e64 s[0:1], 1, v2
	scratch_store_dword off, v58, off offset:32 ; 4-byte Folded Spill
	v_add_u32_e32 v58, s23, v54
	v_cndmask_b32_e64 v2, v39, v3, s[0:1]
	ds_bpermute_b32 v3, v55, v51
	scratch_store_dword off, v51, off offset:36 ; 4-byte Folded Spill
	ds_bpermute_b32 v51, v56, v53
	scratch_store_dword off, v53, off offset:40 ; 4-byte Folded Spill
	ds_bpermute_b32 v53, v55, v54
	scratch_store_dword off, v54, off offset:44 ; 4-byte Folded Spill
	ds_bpermute_b32 v54, v56, v58
	scratch_store_dword off, v58, off offset:48 ; 4-byte Folded Spill
	v_add_u32_e32 v58, s23, v58
	scratch_store_dword off, v55, off offset:104 ; 4-byte Folded Spill
	ds_bpermute_b32 v55, v55, v58
	scratch_store_dword off, v58, off offset:52 ; 4-byte Folded Spill
	v_add_u32_e32 v58, s23, v58
	v_mfma_f32_32x32x16_f16 v[18:33], v[78:81], v[122:125], v[18:33]
	ds_bpermute_b32 v56, v56, v58
	s_waitcnt lgkmcnt(5)
	v_lshlrev_b32_e32 v3, 1, v3
	s_waitcnt lgkmcnt(4)
	v_lshlrev_b32_e32 v51, 1, v51
	s_waitcnt lgkmcnt(3)
	v_lshlrev_b32_e32 v53, 1, v53
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v54, 1, v54
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v55, 1, v55
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v56, 1, v56
	v_cndmask_b32_e32 v3, v39, v3, vcc
	v_cndmask_b32_e64 v51, v39, v51, s[0:1]
	v_cndmask_b32_e32 v53, v39, v53, vcc
	v_cndmask_b32_e64 v54, v39, v54, s[0:1]
	v_cndmask_b32_e32 v55, v39, v55, vcc
	v_cndmask_b32_e64 v39, v39, v56, s[0:1]
	v_or_b32_e32 v56, 0x3800, v57
	scratch_store_dword off, v56, off offset:156 ; 4-byte Folded Spill
	v_add_u32_e32 v56, 0, v56
	v_mfma_f32_32x32x16_f16 v[18:33], v[82:85], v[114:117], v[18:33]
	scratch_store_dword off, v57, off offset:116 ; 4-byte Folded Spill
	v_add_u32_e32 v57, 0x8000, v56
	s_and_b32 s0, s24, 0x3fff
	v_readfirstlane_b32 s1, v57
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v57, v44, v45
	v_max3_f32 v57, v57, v4, v5
	v_max3_f32 v57, v57, v6, v7
	v_max3_f32 v57, v57, v8, v9
	v_max3_f32 v57, v57, v10, v11
	v_max3_f32 v57, v57, v12, v13
	v_max3_f32 v57, v57, v14, v15
	s_nop 0
	v_fma_f32 v18, v18, s20, 0
	v_fma_f32 v19, v19, s20, 0
	v_max3_f32 v57, v57, v16, v17
	v_fma_f32 v20, v20, s20, 0
	v_fma_f32 v21, v21, s20, 0
	s_bitset1_b32 s0, 14
	v_max3_f32 v57, v57, v18, v19
	v_fma_f32 v22, v22, s20, 0
	v_fma_f32 v23, v23, s20, 0
	s_and_b32 s23, s16, 0xffff
	s_lshl_b32 s0, s0, 16
	v_max3_f32 v57, v57, v20, v21
	v_fma_f32 v24, v24, s20, 0
	v_fma_f32 v25, v25, s20, 0
	s_or_b32 s29, s23, s0
	s_mov_b32 m0, s13
	v_max3_f32 v57, v57, v22, v23
	v_fma_f32 v26, v26, s20, 0
	v_fma_f32 v27, v27, s20, 0
	buffer_load_dword v37, s[28:31], 0 offen lds
	s_mov_b32 m0, s40
	v_max3_f32 v57, v57, v24, v25
	v_fma_f32 v28, v28, s20, 0
	v_fma_f32 v29, v29, s20, 0
	buffer_load_dword v2, s[28:31], 0 offen lds
	s_mov_b32 m0, s41
	v_max3_f32 v57, v57, v26, v27
	v_fma_f32 v30, v30, s20, 0
	v_fma_f32 v31, v31, s20, 0
	buffer_load_dword v3, s[28:31], 0 offen lds
	s_mov_b32 m0, s42
	v_max3_f32 v57, v57, v28, v29
	v_fma_f32 v32, v32, s20, 0
	v_fma_f32 v33, v33, s20, 0
	buffer_load_dword v51, s[28:31], 0 offen lds
	s_mov_b32 m0, s43
	s_add_u32 s12, s12, s2
	v_max3_f32 v57, v57, v30, v31
	v_lshlrev_b32_e32 v60, 2, v69
	buffer_load_dword v53, s[28:31], 0 offen lds
	s_mov_b32 m0, s44
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s23, s22, 31
	v_max3_f32 v57, v57, v32, v33
	v_xor_b32_e32 v211, 0x80, v60
	scratch_store_dword off, v58, off offset:56 ; 4-byte Folded Spill
	buffer_load_dword v54, s[28:31], 0 offen lds
	s_mov_b32 m0, s45
	s_lshl_b64 s[22:23], s[22:23], 1
	ds_bpermute_b32 v58, v211, v57
	buffer_load_dword v55, s[28:31], 0 offen lds
	s_mov_b32 m0, s1
	s_add_u32 s1, s28, s22
	buffer_load_dword v39, s[28:31], 0 offen lds
	s_addc_u32 s28, s16, s23
	s_and_b32 s7, s7, 0xffff
	s_or_b32 s13, s7, s19
	s_mov_b32 m0, s33
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v64, s[12:15], 0 offen lds
	s_mov_b32 m0, s35
	s_mov_b32 s7, 0xff800000
	v_add_u32_e32 v46, 0xc000, v46
	buffer_load_dwordx4 v65, s[12:15], 0 offen lds
	v_max3_f32 v226, v57, v58, s7
	v_readfirstlane_b32 s7, v46
	s_and_b32 s12, s28, 0xffff
	v_add_u32_e32 v36, 0xc000, v36
	s_or_b32 s13, s12, s0
	s_mov_b32 s12, s1
	s_mov_b32 m0, s7
	v_readfirstlane_b32 s7, v36
	s_waitcnt vmcnt(10)
	s_barrier
	buffer_load_dword v37, s[12:15], 0 offen lds
	s_mov_b32 m0, s7
	s_mov_b32 s16, 0
	buffer_load_dword v2, s[12:15], 0 offen lds
	v_add_u32_e32 v2, 0xc000, v47
	s_mov_b32 s29, 0x8000
	v_readfirstlane_b32 s7, v2
	v_add_u32_e32 v2, 0xc000, v48
	s_mov_b32 m0, s7
	v_readfirstlane_b32 s7, v2
	v_add_u32_e32 v2, 0xc000, v49
	buffer_load_dword v3, s[12:15], 0 offen lds
	s_mov_b32 m0, s7
	v_readfirstlane_b32 s7, v2
	v_add_u32_e32 v2, 0xc000, v50
	buffer_load_dword v51, s[12:15], 0 offen lds
	s_mov_b32 m0, s7
	v_readfirstlane_b32 s7, v2
	v_add_u32_e32 v2, 0xc000, v52
	buffer_load_dword v53, s[12:15], 0 offen lds
	s_mov_b32 m0, s7
	v_readfirstlane_b32 s7, v2
	v_add_u32_e32 v2, 0xc000, v56
	buffer_load_dword v54, s[12:15], 0 offen lds
	s_mov_b32 m0, s7
	v_readfirstlane_b32 s7, v2
	buffer_load_dword v55, s[12:15], 0 offen lds
	s_mov_b32 m0, s7
	s_movk_i32 s7, 0x60
	buffer_load_dword v39, s[12:15], 0 offen lds
	v_lshrrev_b32_e32 v39, 3, v0
	v_bitop3_b32 v0, v1, v86, 16 bitop3:0x36
	scratch_store_dword off, v0, off offset:60 ; 4-byte Folded Spill
	v_bitop3_b32 v0, v1, v86, 32 bitop3:0x36
	scratch_store_dword off, v0, off offset:64 ; 4-byte Folded Spill
	v_bitop3_b32 v0, v1, v86, 48 bitop3:0x36
	v_and_b32_e32 v2, 12, v60
	v_and_b32_e32 v3, 12, v69
	s_movk_i32 s12, 0x50
	scratch_store_dword off, v0, off offset:68 ; 4-byte Folded Spill
	v_bitop3_b32 v0, v1, v86, 64 bitop3:0x36
	v_bitop3_b32 v36, v2, v3, 32 bitop3:0x36
	v_bitop3_b32 v37, v2, v3, 64 bitop3:0x36
	v_bitop3_b32 v2, v2, v3, s7 bitop3:0x36
	scratch_store_dword off, v0, off offset:72 ; 4-byte Folded Spill
	v_bitop3_b32 v0, v1, v86, s12 bitop3:0x36
	v_sub_f32_e32 v3, v4, v226
	v_sub_f32_e32 v4, v5, v226
	scratch_store_dword off, v0, off offset:76 ; 4-byte Folded Spill
	v_bitop3_b32 v0, v1, v86, s7 bitop3:0x36
	v_sub_f32_e32 v5, v6, v226
	v_exp_f32_e32 v216, v4
	s_add_i32 s7, 0, 0x4000
	v_mov_b32_e32 v4, v86
	v_bitop3_b32 v2, v2, v87, v59 bitop3:0x36
	scratch_store_dword off, v0, off offset:80 ; 4-byte Folded Spill
	v_add_u32_e32 v83, s7, v67
	scratch_store_dword off, v69, off offset:84 ; 4-byte Folded Spill
	scratch_store_dword off, v60, off offset:196 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:168 ; 8-byte Folded Spill
	scratch_store_dword off, v39, off offset:88 ; 4-byte Folded Spill
	scratch_store_dword off, v87, off offset:176 ; 4-byte Folded Spill
	scratch_store_dword off, v59, off offset:192 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off        ; 4-byte Folded Spill
	v_sub_f32_e32 v67, 0xff800000, v226
	v_exp_f32_e32 v236, v3
	v_add_u32_e32 v77, s7, v35
	v_add_u32_e32 v78, s7, v38
	v_add_u32_e32 v79, s7, v40
	v_add_u32_e32 v80, s7, v41
	v_add_u32_e32 v81, s7, v42
	v_add_u32_e32 v82, s7, v43
	v_add_u32_e32 v84, s7, v68
	v_bitop3_b32 v3, v60, v69, 12 bitop3:0x28
	v_exp_f32_e32 v210, v67
	ds_read_b128 v[66:69], v66 offset:16384
	ds_read_b128 v[178:181], v70 offset:16384
	ds_read_b128 v[186:189], v71 offset:16384
	ds_read_b128 v[182:185], v72 offset:16384
	ds_read_b128 v[166:169], v73 offset:16384
	ds_read_b128 v[170:173], v74 offset:16384
	ds_read_b128 v[174:177], v75 offset:16384
	ds_read_b128 v[162:165], v76 offset:16384
	ds_read_b128 v[158:161], v77 offset:8192
	ds_read_b128 v[154:157], v78 offset:8192
	ds_read_b128 v[150:153], v79 offset:8192
	ds_read_b128 v[146:149], v80 offset:8192
	ds_read_b128 v[142:145], v81 offset:8192
	ds_read_b128 v[138:141], v82 offset:8192
	ds_read_b128 v[134:137], v83 offset:8192
	ds_read_b128 v[130:133], v84 offset:8192
	s_movk_i32 s13, 0x70
	v_bitop3_b32 v212, v1, v86, s13 bitop3:0x36
	s_mul_hi_i32 s13, s6, 6
	s_add_u32 s6, s36, s38
	s_addc_u32 s7, s37, s39
	v_sub_f32_e32 v0, v44, v226
	v_sub_f32_e32 v1, v45, v226
	v_sub_f32_e32 v6, v7, v226
	v_sub_f32_e32 v7, v8, v226
	v_sub_f32_e32 v8, v9, v226
	v_sub_f32_e32 v9, v10, v226
	v_sub_f32_e32 v10, v11, v226
	v_sub_f32_e32 v11, v12, v226
	v_sub_f32_e32 v12, v13, v226
	v_sub_f32_e32 v13, v14, v226
	v_sub_f32_e32 v14, v15, v226
	v_sub_f32_e32 v15, v16, v226
	v_sub_f32_e32 v16, v17, v226
	v_sub_f32_e32 v17, v18, v226
	v_sub_f32_e32 v18, v19, v226
	v_sub_f32_e32 v19, v20, v226
	v_sub_f32_e32 v20, v21, v226
	v_sub_f32_e32 v21, v22, v226
	v_sub_f32_e32 v22, v23, v226
	v_sub_f32_e32 v23, v24, v226
	v_sub_f32_e32 v24, v25, v226
	v_sub_f32_e32 v25, v26, v226
	v_sub_f32_e32 v26, v27, v226
	v_sub_f32_e32 v27, v28, v226
	v_sub_f32_e32 v28, v29, v226
	v_sub_f32_e32 v29, v30, v226
	v_sub_f32_e32 v30, v31, v226
	v_sub_f32_e32 v31, v32, v226
	v_sub_f32_e32 v32, v33, v226
	s_mul_i32 s12, s21, 0x180
	s_lshl_b64 s[6:7], s[6:7], 1
	v_exp_f32_e32 v235, v0
	v_exp_f32_e32 v247, v1
	v_exp_f32_e32 v232, v5
	v_exp_f32_e32 v217, v6
	v_exp_f32_e32 v233, v7
	v_exp_f32_e32 v243, v8
	v_exp_f32_e32 v219, v9
	v_exp_f32_e32 v238, v10
	v_exp_f32_e32 v221, v11
	v_exp_f32_e32 v239, v12
	v_exp_f32_e32 v0, v13
	v_exp_f32_e32 v248, v14
	v_exp_f32_e32 v240, v15
	v_exp_f32_e32 v250, v16
	v_exp_f32_e32 v249, v17
	v_exp_f32_e32 v245, v18
	v_exp_f32_e32 v251, v19
	v_exp_f32_e32 v218, v20
	v_exp_f32_e32 v246, v21
	v_exp_f32_e32 v252, v22
	v_exp_f32_e32 v224, v23
	v_exp_f32_e32 v254, v24
	v_exp_f32_e32 v253, v25
	v_exp_f32_e32 v1, v26
	v_exp_f32_e32 v255, v27
	v_exp_f32_e32 v227, v28
	v_exp_f32_e32 v223, v29
	v_exp_f32_e32 v229, v30
	v_exp_f32_e32 v228, v31
	v_exp_f32_e32 v230, v32
	s_add_u32 s6, s12, s6
	s_addc_u32 s7, s13, s7
	v_bitop3_b32 v3, v3, v87, v59 bitop3:0x36
	v_and_or_b32 v4, v34, 3, v39
	s_add_u32 s30, s4, s6
	v_mov_b32_e32 v2, 0
	v_lshlrev_b32_e32 v244, 7, v4
	v_bitop3_b32 v225, v36, v87, v59 bitop3:0x36
	v_bitop3_b32 v213, v37, v87, v59 bitop3:0x36
	s_addc_u32 s31, s5, s7
	s_add_i32 s21, 0, 0x8000
	s_add_i32 s13, 0, 0xc000
	s_movk_i32 s33, 0xffc0
	v_lshlrev_b32_e32 v234, 1, v3
	s_mov_b32 s5, 0
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
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[82:97], v[66:69], v[110:113], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_add_u32 s4, s1, s22
	s_addc_u32 s36, s28, s23
	s_add_i32 s12, s5, 1
	s_cmp_lt_i32 s12, 2
	v_mfma_f32_32x32x16_f16 v[66:81], v[158:161], v[110:113], 0
	s_cselect_b32 s35, s12, 0
	s_lshl_b32 s12, s35, 14
	s_mov_b32 s6, s16
	s_add_i32 s16, s12, 0
	s_and_b32 s12, s31, 0xffff
	s_mov_b32 s7, s21
	s_mov_b32 s21, s13
	v_mfma_f32_32x32x16_f16 v[66:81], v[154:157], v[106:109], v[66:81]
	s_or_b32 s13, s12, s19
	s_mov_b32 s12, s30
	v_mov_b32_e32 v190, v215
	v_mov_b32_e32 v231, v226
	v_pk_mul_f32 v[50:51], v[50:51], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[150:153], v[102:105], v[66:81]
	v_pk_mul_f32 v[56:57], v[56:57], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[146:149], v[98:101], v[66:81]
	ds_bpermute_b32 v146, v241, v220
	v_pk_mul_f32 v[38:39], v[38:39], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[210:211] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v146, 1, v146
	v_pk_mul_f32 v[46:47], v[46:47], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[142:145], v[126:129], v[66:81]
	v_cvt_pk_f16_f32 v142, v235, v247
	v_pk_mul_f32 v[48:49], v[48:49], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[138:141], v[118:121], v[66:81]
	v_cvt_pk_f16_f32 v140, v0, v248
	v_pk_mul_f32 v[28:29], v[28:29], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[134:137], v[122:125], v[66:81]
	v_cvt_pk_f16_f32 v134, v253, v1
	v_cvt_pk_f16_f32 v135, v255, v227
	v_pk_mul_f32 v[8:9], v[8:9], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[130:133], v[114:117], v[66:81]
	v_add_f32_e32 v130, v235, v247
	v_add_f32_e32 v130, v130, v236
	v_add_f32_e32 v130, v130, v216
	v_add_f32_e32 v130, v130, v232
	v_add_f32_e32 v130, v130, v217
	v_add_f32_e32 v130, v130, v233
	v_add_f32_e32 v130, v130, v243
	v_add_f32_e32 v130, v130, v219
	v_add_f32_e32 v130, v130, v238
	v_add_f32_e32 v130, v130, v221
	v_add_f32_e32 v130, v130, v239
	v_add_f32_e32 v130, v130, v0
	scratch_load_dword v0, off, off offset:8 ; 4-byte Folded Reload
	v_add_f32_e32 v130, v130, v248
	v_add_f32_e32 v130, v130, v240
	v_add_f32_e32 v130, v130, v250
	v_add_f32_e32 v130, v130, v249
	v_add_f32_e32 v130, v130, v245
	v_add_f32_e32 v130, v130, v251
	v_add_f32_e32 v130, v130, v218
	v_add_f32_e32 v130, v130, v246
	v_add_f32_e32 v130, v130, v252
	v_add_f32_e32 v130, v130, v224
	v_add_f32_e32 v130, v130, v254
	v_add_f32_e32 v130, v130, v253
	v_add_f32_e32 v130, v130, v1
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[106:109], v[82:97]
	v_add_f32_e32 v130, v130, v255
	v_add_f32_e32 v130, v130, v227
	v_add_f32_e32 v130, v130, v223
	v_add_f32_e32 v130, v130, v229
	v_add_f32_e32 v130, v130, v228
	v_add_f32_e32 v130, v130, v230
	ds_bpermute_b32 v131, v211, v130
	v_mfma_f32_32x32x16_f16 v[82:97], v[186:189], v[102:105], v[82:97]
	v_fma_f32 v66, v66, s20, 0
	v_fma_f32 v67, v67, s20, 0
	v_fma_f32 v68, v68, s20, 0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v215, v130, v131
	v_fmac_f32_e32 v215, v190, v210
	v_fma_f32 v69, v69, s20, 0
	v_fma_f32 v70, v70, s20, 0
	v_mfma_f32_32x32x16_f16 v[82:97], v[182:185], v[98:101], v[82:97]
	v_fma_f32 v71, v71, s20, 0
	v_fma_f32 v72, v72, s20, 0
	v_fma_f32 v73, v73, s20, 0
	v_fma_f32 v74, v74, s20, 0
	v_fma_f32 v75, v75, s20, 0
	v_fma_f32 v76, v76, s20, 0
	v_fma_f32 v77, v77, s20, 0
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[126:129], v[82:97]
	v_fma_f32 v78, v78, s20, 0
	v_fma_f32 v79, v79, s20, 0
	v_fma_f32 v80, v80, s20, 0
	v_fma_f32 v81, v81, s20, 0
	v_cvt_pk_f16_f32 v132, v246, v252
	v_cvt_pk_f16_f32 v133, v224, v254
	v_cvt_pk_f16_f32 v143, v236, v216
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[118:121], v[82:97]
	v_cvt_pk_f16_f32 v144, v232, v217
	v_cvt_pk_f16_f32 v145, v233, v243
	v_cvt_pk_f16_f32 v138, v219, v238
	v_cvt_pk_f16_f32 v139, v221, v239
	v_cvt_pk_f16_f32 v141, v240, v250
	v_cvt_pk_f16_f32 v130, v249, v245
	v_cvt_pk_f16_f32 v131, v251, v218
	v_mfma_f32_32x32x16_f16 v[82:97], v[174:177], v[122:125], v[82:97]
	v_cvt_pk_f16_f32 v136, v223, v229
	v_cvt_pk_f16_f32 v137, v228, v230
	s_lshl_b32 s5, s5, 14
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v0, s16, v0
	s_nop 0
	v_readfirstlane_b32 s37, v0
	scratch_load_dword v0, off, off offset:4 ; 4-byte Folded Reload
	s_mov_b32 m0, s37
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[114:117], v[82:97]
	buffer_load_dwordx4 v146, s[12:15], 0 offen lds
	v_lshlrev_b32_e32 v146, 1, v225
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v1, s16, v1
	s_nop 0
	v_readfirstlane_b32 s37, v1
	s_mov_b32 m0, s37
	s_add_i32 s37, s5, 0
	s_and_b32 s5, s36, 0xffff
	s_or_b32 s5, s5, s0
	s_waitcnt vmcnt(1)
	ds_bpermute_b32 v0, v241, v0
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_load_dwordx4 v0, s[12:15], 0 offen lds
	v_lshlrev_b32_e32 v0, 1, v244
	v_add3_u32 v1, s7, v234, v0
	v_add3_u32 v146, s7, v146, v0
	s_waitcnt vmcnt(10)
	s_barrier
	ds_read_b64_tr_b16 v[194:195], v1
	ds_read_b64_tr_b16 v[196:197], v146 offset:2048
	ds_read_b64_tr_b16 v[198:199], v1 offset:4096
	ds_read_b64_tr_b16 v[200:201], v146 offset:6144
	ds_read_b64_tr_b16 v[202:203], v1 offset:8192
	ds_read_b64_tr_b16 v[204:205], v146 offset:10240
	ds_read_b64_tr_b16 v[206:207], v1 offset:12288
	ds_read_b64_tr_b16 v[208:209], v146 offset:14336
	ds_read_b64_tr_b16 v[178:179], v146
	ds_read_b64_tr_b16 v[180:181], v1 offset:2048
	ds_read_b64_tr_b16 v[182:183], v146 offset:4096
	ds_read_b64_tr_b16 v[184:185], v1 offset:6144
	ds_read_b64_tr_b16 v[186:187], v146 offset:8192
	ds_read_b64_tr_b16 v[188:189], v1 offset:10240
	ds_read_b64_tr_b16 v[190:191], v146 offset:12288
	ds_read_b64_tr_b16 v[192:193], v1 offset:14336
	scratch_load_dword v146, off, off       ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v1, 1, v213
	v_add3_u32 v1, s7, v1, v0
	ds_read_b64_tr_b16 v[162:163], v1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[194:197], v[142:145], v[50:65]
	v_lshlrev_b32_e32 v195, 1, v214
	v_lshlrev_b32_e32 v194, 1, v242
	s_add_i32 s13, s37, 0x8000
	s_add_u32 s1, s1, s22
	s_addc_u32 s28, s28, s23
	s_add_u32 s30, s30, s2
	s_addc_u32 s31, s31, s3
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[34:49], v[178:181], v[142:145], v[34:49]
	s_add_i32 s33, s33, 64
	s_cmpk_lt_u32 s33, 0x1f00
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v146, 1, v146
	v_add3_u32 v0, s7, v146, v0
	ds_read_b64_tr_b16 v[164:165], v0 offset:2048
	ds_read_b64_tr_b16 v[166:167], v1 offset:4096
	ds_read_b64_tr_b16 v[168:169], v0 offset:6144
	ds_read_b64_tr_b16 v[170:171], v1 offset:8192
	ds_read_b64_tr_b16 v[172:173], v0 offset:10240
	ds_read_b64_tr_b16 v[174:175], v1 offset:12288
	ds_read_b64_tr_b16 v[176:177], v0 offset:14336
	ds_read_b64_tr_b16 v[158:159], v0
	ds_read_b64_tr_b16 v[160:161], v1 offset:2048
	ds_read_b64_tr_b16 v[154:155], v0 offset:4096
	ds_read_b64_tr_b16 v[156:157], v1 offset:6144
	ds_read_b64_tr_b16 v[146:147], v0 offset:8192
	ds_read_b64_tr_b16 v[148:149], v1 offset:10240
	ds_read_b64_tr_b16 v[150:151], v0 offset:12288
	ds_read_b64_tr_b16 v[152:153], v1 offset:14336
	v_fma_f32 v0, v82, s20, 0
	v_fma_f32 v1, v83, s20, 0
	v_fma_f32 v82, v84, s20, 0
	v_fma_f32 v83, v85, s20, 0
	v_fma_f32 v84, v86, s20, 0
	v_fma_f32 v86, v88, s20, 0
	v_fma_f32 v88, v90, s20, 0
	v_fma_f32 v90, v92, s20, 0
	v_fma_f32 v92, v94, s20, 0
	v_fma_f32 v94, v96, s20, 0
	v_max_f32_e32 v96, v0, v1
	v_fma_f32 v85, v87, s20, 0
	v_max3_f32 v96, v96, v82, v83
	v_fma_f32 v87, v89, s20, 0
	v_max3_f32 v96, v96, v84, v85
	v_fma_f32 v89, v91, s20, 0
	v_max3_f32 v96, v96, v86, v87
	v_fma_f32 v91, v93, s20, 0
	v_max3_f32 v96, v96, v88, v89
	v_fma_f32 v93, v95, s20, 0
	v_max3_f32 v96, v96, v90, v91
	v_fma_f32 v95, v97, s20, 0
	v_max3_f32 v96, v96, v92, v93
	v_max3_f32 v96, v96, v94, v95
	v_max3_f32 v96, v96, v66, v67
	v_max3_f32 v96, v96, v68, v69
	v_max3_f32 v96, v96, v70, v71
	v_max3_f32 v96, v96, v72, v73
	v_max3_f32 v96, v96, v74, v75
	v_max3_f32 v96, v96, v76, v77
	v_max3_f32 v96, v96, v78, v79
	v_max3_f32 v96, v96, v80, v81
	ds_bpermute_b32 v97, v211, v96
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[18:33], v[162:165], v[142:145], v[18:33]
	s_mov_b32 s7, s15
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v226, v231, v96, v97
	v_sub_f32_e32 v1, v1, v226
	v_sub_f32_e32 v71, v71, v226
	v_sub_f32_e32 v72, v72, v226
	v_sub_f32_e32 v73, v73, v226
	v_sub_f32_e32 v74, v74, v226
	v_sub_f32_e32 v75, v75, v226
	v_sub_f32_e32 v76, v76, v226
	v_exp_f32_e32 v247, v1
	v_exp_f32_e32 v252, v71
	v_exp_f32_e32 v224, v72
	v_exp_f32_e32 v254, v73
	v_exp_f32_e32 v253, v74
	v_exp_f32_e32 v1, v75
	v_exp_f32_e32 v255, v76
	scratch_load_dword v71, off, off offset:60 ; 4-byte Folded Reload
	scratch_load_dword v72, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v73, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v74, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v75, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v76, off, off offset:80 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[2:17], v[158:161], v[142:145], v[2:17]
	v_sub_f32_e32 v66, v66, v226
	v_sub_f32_e32 v70, v70, v226
	v_sub_f32_e32 v77, v77, v226
	v_sub_f32_e32 v67, v67, v226
	v_sub_f32_e32 v68, v68, v226
	v_sub_f32_e32 v69, v69, v226
	v_exp_f32_e32 v249, v66
	v_mfma_f32_32x32x16_f16 v[50:65], v[198:201], v[138:141], v[50:65]
	v_exp_f32_e32 v246, v70
	v_exp_f32_e32 v227, v77
	v_sub_f32_e32 v66, v231, v226
	v_add3_u32 v70, s6, v195, v194
	v_exp_f32_e32 v245, v67
	v_exp_f32_e32 v251, v68
	v_exp_f32_e32 v218, v69
	v_mfma_f32_32x32x16_f16 v[34:49], v[182:185], v[138:141], v[34:49]
	v_exp_f32_e32 v210, v66
	ds_read_b128 v[66:69], v70
	v_sub_f32_e32 v81, v81, v226
	v_exp_f32_e32 v230, v81
	v_sub_f32_e32 v78, v78, v226
	v_sub_f32_e32 v79, v79, v226
	v_sub_f32_e32 v80, v80, v226
	v_mfma_f32_32x32x16_f16 v[18:33], v[166:169], v[138:141], v[18:33]
	v_exp_f32_e32 v223, v78
	v_exp_f32_e32 v229, v79
	v_exp_f32_e32 v228, v80
	v_sub_f32_e32 v0, v0, v226
	v_sub_f32_e32 v82, v82, v226
	v_sub_f32_e32 v83, v83, v226
	v_sub_f32_e32 v84, v84, v226
	v_mfma_f32_32x32x16_f16 v[2:17], v[154:157], v[138:141], v[2:17]
	v_sub_f32_e32 v85, v85, v226
	v_sub_f32_e32 v86, v86, v226
	v_sub_f32_e32 v87, v87, v226
	v_sub_f32_e32 v88, v88, v226
	v_sub_f32_e32 v89, v89, v226
	v_sub_f32_e32 v90, v90, v226
	v_sub_f32_e32 v91, v91, v226
	v_mfma_f32_32x32x16_f16 v[50:65], v[202:205], v[130:133], v[50:65]
	v_lshlrev_b32_e32 v202, 1, v212
	v_add3_u32 v77, s6, v202, v194
	v_sub_f32_e32 v92, v92, v226
	v_sub_f32_e32 v93, v93, v226
	v_sub_f32_e32 v94, v94, v226
	v_sub_f32_e32 v95, v95, v226
	v_exp_f32_e32 v235, v0
	v_mfma_f32_32x32x16_f16 v[34:49], v[186:189], v[130:133], v[34:49]
	v_exp_f32_e32 v236, v82
	v_exp_f32_e32 v216, v83
	v_exp_f32_e32 v232, v84
	v_exp_f32_e32 v217, v85
	v_exp_f32_e32 v233, v86
	v_exp_f32_e32 v243, v87
	v_exp_f32_e32 v219, v88
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[130:133], v[18:33]
	v_exp_f32_e32 v238, v89
	v_exp_f32_e32 v221, v90
	v_exp_f32_e32 v239, v91
	v_exp_f32_e32 v0, v92
	v_exp_f32_e32 v248, v93
	v_exp_f32_e32 v240, v94
	v_exp_f32_e32 v250, v95
	v_mfma_f32_32x32x16_f16 v[2:17], v[146:149], v[130:133], v[2:17]
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v196, 1, v71
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v197, 1, v72
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v198, 1, v73
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v199, 1, v74
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v200, 1, v75
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v201, 1, v76
	v_add3_u32 v71, s6, v196, v194
	ds_read_b128 v[178:181], v71
	v_add3_u32 v72, s6, v197, v194
	v_add3_u32 v73, s6, v198, v194
	v_add3_u32 v74, s6, v199, v194
	v_add3_u32 v75, s6, v200, v194
	v_add3_u32 v76, s6, v201, v194
	v_mfma_f32_32x32x16_f16 v[50:65], v[206:209], v[134:137], v[50:65]
	ds_read_b128 v[186:189], v72
	ds_read_b128 v[182:185], v73
	ds_read_b128 v[166:169], v74
	ds_read_b128 v[170:173], v75
	s_mov_b32 s6, s14
	v_mfma_f32_32x32x16_f16 v[34:49], v[190:193], v[134:137], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[174:177], v[134:137], v[18:33]
	ds_read_b128 v[174:177], v76
	v_mfma_f32_32x32x16_f16 v[2:17], v[150:153], v[134:137], v[2:17]
	ds_read_b128 v[162:165], v77
	ds_read_b128 v[158:161], v70 offset:8192
	ds_read_b128 v[154:157], v71 offset:8192
	ds_read_b128 v[150:153], v72 offset:8192
	ds_read_b128 v[146:149], v73 offset:8192
	ds_read_b128 v[142:145], v74 offset:8192
	ds_read_b128 v[138:141], v75 offset:8192
	ds_read_b128 v[134:137], v76 offset:8192
	ds_read_b128 v[130:133], v77 offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v70, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v71, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v73, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v72, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v70, 1, v70
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v71, 1, v71
	v_add3_u32 v70, s37, v70, v71
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v72, v72, 1, s37
	v_lshl_add_u32 v73, v73, 1, s13
	v_add3_u32 v71, v72, v71, s29
	v_add_u32_e32 v72, 0x8000, v70
	v_sub_u32_e32 v70, v70, v73
	v_add_u32_e32 v70, 0x8000, v70
	v_ashrrev_i32_e32 v81, 31, v70
	v_lshrrev_b32_e32 v81, 30, v81
	v_add_u32_e32 v70, v70, v81
	v_lshrrev_b32_e32 v70, 2, v70
	v_add_lshl_u32 v70, v70, v237, 2
	ds_bpermute_b32 v70, v70, v222
	v_readfirstlane_b32 s12, v73
	v_add_u32_e32 v74, 0x800, v73
	s_mov_b32 m0, s12
	v_add_u32_e32 v75, 0x1000, v73
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_load_dword v70, s[4:7], 0 offen lds
	v_sub_u32_e32 v70, v71, v74
	v_add_u32_e32 v70, 0x800, v70
	v_add_u32_e32 v76, 0x1800, v73
	v_add_u32_e32 v77, 0x2000, v73
	v_add_u32_e32 v78, 0x2800, v73
	v_add_u32_e32 v79, 0x3000, v73
	v_add_u32_e32 v80, 0x3800, v73
	v_ashrrev_i32_e32 v73, 31, v70
	v_lshrrev_b32_e32 v73, 30, v73
	v_add_u32_e32 v70, v70, v73
	scratch_load_dword v73, off, off offset:32 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v70, 2, v70
	v_add_lshl_u32 v70, v70, v237, 2
	v_readfirstlane_b32 s12, v74
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v75
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v70, v70, v73
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_load_dword v70, s[4:7], 0 offen lds
	v_sub_u32_e32 v70, v72, v75
	v_add_u32_e32 v70, 0x1000, v70
	v_ashrrev_i32_e32 v73, 31, v70
	v_lshrrev_b32_e32 v73, 30, v73
	v_add_u32_e32 v70, v70, v73
	scratch_load_dword v73, off, off offset:36 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v70, 2, v70
	v_add_lshl_u32 v70, v70, v237, 2
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v76
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v70, v70, v73
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_load_dword v70, s[4:7], 0 offen lds
	v_sub_u32_e32 v70, v71, v76
	v_add_u32_e32 v70, 0x1800, v70
	v_ashrrev_i32_e32 v73, 31, v70
	v_lshrrev_b32_e32 v73, 30, v73
	v_add_u32_e32 v70, v70, v73
	scratch_load_dword v73, off, off offset:40 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v70, 2, v70
	v_add_lshl_u32 v70, v70, v237, 2
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v77
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v70, v70, v73
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_load_dword v70, s[4:7], 0 offen lds
	v_sub_u32_e32 v70, v72, v77
	v_add_u32_e32 v70, 0x2000, v70
	v_ashrrev_i32_e32 v73, 31, v70
	v_lshrrev_b32_e32 v73, 30, v73
	v_add_u32_e32 v70, v70, v73
	scratch_load_dword v73, off, off offset:44 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v70, 2, v70
	v_add_lshl_u32 v70, v70, v237, 2
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v78
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v70, v70, v73
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_load_dword v70, s[4:7], 0 offen lds
	v_sub_u32_e32 v70, v71, v78
	v_add_u32_e32 v70, 0x2800, v70
	v_ashrrev_i32_e32 v73, 31, v70
	v_lshrrev_b32_e32 v73, 30, v73
	v_add_u32_e32 v70, v70, v73
	scratch_load_dword v73, off, off offset:48 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v70, 2, v70
	v_add_lshl_u32 v70, v70, v237, 2
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v79
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v70, v70, v73
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_load_dword v70, s[4:7], 0 offen lds
	v_sub_u32_e32 v70, v72, v79
	v_add_u32_e32 v70, 0x3000, v70
	v_ashrrev_i32_e32 v72, 31, v70
	v_lshrrev_b32_e32 v72, 30, v72
	v_add_u32_e32 v70, v70, v72
	scratch_load_dword v72, off, off offset:52 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v70, 2, v70
	v_add_lshl_u32 v70, v70, v237, 2
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v80
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v70, v70, v72
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_load_dword v70, s[4:7], 0 offen lds
	v_sub_u32_e32 v70, v71, v80
	v_add_u32_e32 v70, 0x3800, v70
	v_ashrrev_i32_e32 v71, 31, v70
	v_lshrrev_b32_e32 v71, 30, v71
	v_add_u32_e32 v70, v70, v71
	scratch_load_dword v71, off, off offset:56 ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v70, 2, v70
	v_add_lshl_u32 v70, v70, v237, 2
	s_mov_b32 m0, s12
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v70, v70, v71
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_load_dword v70, s[4:7], 0 offen lds
	s_mov_b32 s5, s35
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	scratch_load_dword v82, off, off offset:188 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[84:85], off, off offset:168 ; 8-byte Folded Reload
	scratch_load_dword v86, off, off offset:164 ; 4-byte Folded Reload
	scratch_load_dword v89, off, off offset:180 ; 4-byte Folded Reload
	scratch_load_dword v90, off, off offset:184 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[110:113], 0
	s_mul_i32 s2, s18, 0xc0000
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s6, s17, 14
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s35, s34, 31
	s_add_i32 s14, s34, 0xffffc100
	s_lshl_b64 s[2:3], s[2:3], 2
	v_mfma_f32_32x32x16_f16 v[66:81], v[178:181], v[106:109], v[66:81]
	s_add_u32 s5, s8, s2
	s_addc_u32 s8, s9, s3
	s_lshl_b64 s[2:3], s[6:7], 2
	s_add_u32 s5, s5, s2
	s_addc_u32 s6, s8, s3
	s_lshl_b64 s[2:3], s[34:35], 2
	s_add_u32 s12, s5, s2
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[102:105], v[66:81]
	s_addc_u32 s8, s6, s3
	s_add_u32 s4, s4, s22
	v_add_lshl_u32 v207, v225, v244, 1
	s_addc_u32 s3, s36, s23
	s_add_i32 s9, s16, 0x8000
	s_add_i32 s2, 0, 0x10000
	s_and_b32 s5, s3, 0xffff
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[98:101], v[66:81]
	s_cmp_lt_i32 s14, 1
	s_mov_b32 s14, 0x3e0293ee
	v_pk_mul_f32 v[64:65], v[64:65], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[166:169], v[126:129], v[66:81]
	v_pk_mul_f32 v[54:55], v[54:55], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[170:173], v[118:121], v[66:81]
	v_pk_mul_f32 v[40:41], v[40:41], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[122:125], v[66:81]
	v_pk_mul_f32 v[26:27], v[26:27], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[210:211] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[162:165], v[114:117], v[66:81]
	v_pk_mul_f32 v[12:13], v[12:13], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[210:211] op_sel_hi:[1,0]
	s_mov_b32 s7, 0x27000
	s_nop 4
	v_fma_f32 v220, v70, s14, 0
	v_fma_f32 v214, v66, s14, 0
	v_fma_f32 v222, v72, s14, 0
	v_fma_f32 v231, v80, s14, 0
	s_mov_b32 s6, 0x7ffffffe
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v82, 0xa0, v82
	s_waitcnt vmcnt(3)
	v_mov_b32_e32 v88, v84
	s_waitcnt vmcnt(2)
	v_mul_lo_u32 v86, s24, v86
	v_lshl_add_u32 v87, s24, 5, v86
	s_waitcnt vmcnt(0)
	v_or3_b32 v203, v82, v89, v90
	v_or_b32_e32 v82, 7, v88
	v_add_u32_e32 v193, v86, v82
	v_add_u32_e32 v185, v87, v82
	scratch_load_dword v82, off, off offset:160 ; 4-byte Folded Reload
	v_or_b32_e32 v83, 1, v88
	v_or_b32_e32 v84, 3, v88
	v_or_b32_e32 v85, 5, v88
	v_add_u32_e32 v187, v86, v83
	v_add_u32_e32 v189, v86, v84
	v_add_u32_e32 v179, v87, v83
	v_add_u32_e32 v181, v87, v84
	v_or_b32_e32 v83, 4, v88
	v_or_b32_e32 v84, 6, v88
	v_add_u32_e32 v191, v86, v85
	v_add_u32_e32 v183, v87, v85
	v_add_u32_e32 v186, v86, v88
	v_add_u32_e32 v178, v87, v88
	v_add_u32_e32 v190, v86, v83
	v_add_u32_e32 v192, v86, v84
	v_add_u32_e32 v182, v87, v83
	v_add_u32_e32 v184, v87, v84
	s_waitcnt vmcnt(0)
	s_barrier
	v_cmp_eq_u32_e64 s[0:1], 0, v82
	v_or_b32_e32 v82, 2, v88
	v_add_u32_e32 v188, v86, v82
	v_add_u32_e32 v180, v87, v82
	v_add_f32_e32 v82, v235, v247
	v_add_f32_e32 v82, v82, v236
	v_add_f32_e32 v82, v82, v216
	v_add_f32_e32 v82, v82, v232
	v_add_f32_e32 v82, v82, v217
	v_add_f32_e32 v82, v82, v233
	v_add_f32_e32 v82, v82, v243
	v_add_f32_e32 v82, v82, v219
	v_add_f32_e32 v82, v82, v238
	v_add_f32_e32 v82, v82, v221
	v_add_f32_e32 v82, v82, v239
	v_add_f32_e32 v82, v82, v0
	v_add_f32_e32 v162, v82, v248
	v_mfma_f32_32x32x16_f16 v[82:97], v[158:161], v[110:113], 0
	v_add_f32_e32 v158, v162, v240
	v_add_f32_e32 v158, v158, v250
	v_add_f32_e32 v158, v158, v249
	v_add_f32_e32 v158, v158, v245
	v_add_f32_e32 v158, v158, v251
	v_add_f32_e32 v158, v158, v218
	v_add_f32_e32 v158, v158, v246
	v_mfma_f32_32x32x16_f16 v[82:97], v[154:157], v[106:109], v[82:97]
	v_add_f32_e32 v154, v158, v252
	v_add_f32_e32 v154, v154, v224
	v_add_f32_e32 v154, v154, v254
	v_add_f32_e32 v154, v154, v253
	v_add_f32_e32 v154, v154, v1
	v_add_f32_e32 v154, v154, v255
	v_add_f32_e32 v154, v154, v227
	v_mfma_f32_32x32x16_f16 v[82:97], v[150:153], v[102:105], v[82:97]
	v_cvt_pk_f16_f32 v152, v0, v248
	scratch_load_dword v242, off, off offset:84 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:196 ; 4-byte Folded Reload
	v_add_f32_e32 v150, v154, v223
	v_cvt_pk_f16_f32 v154, v235, v247
	v_cvt_pk_f16_f32 v155, v236, v216
	v_cvt_pk_f16_f32 v156, v232, v217
	v_cvt_pk_f16_f32 v157, v233, v243
	v_mfma_f32_32x32x16_f16 v[82:97], v[146:149], v[98:101], v[82:97]
	v_add_f32_e32 v150, v150, v229
	v_add_f32_e32 v150, v150, v228
	v_fma_f32 v217, v67, s14, 0
	v_add_f32_e32 v204, v150, v230
	v_cvt_pk_f16_f32 v150, v219, v238
	v_cvt_pk_f16_f32 v147, v251, v218
	v_fma_f32 v218, v68, s14, 0
	v_mfma_f32_32x32x16_f16 v[82:97], v[142:145], v[126:129], v[82:97]
	v_cvt_pk_f16_f32 v142, v253, v1
	v_fma_f32 v219, v69, s14, 0
	v_cvt_pk_f16_f32 v151, v221, v239
	v_fma_f32 v221, v71, s14, 0
	v_cvt_pk_f16_f32 v153, v240, v250
	v_cvt_pk_f16_f32 v144, v223, v229
	v_fma_f32 v223, v73, s14, 0
	v_mfma_f32_32x32x16_f16 v[82:97], v[138:141], v[118:121], v[82:97]
	scratch_load_dword v139, off, off offset:192 ; 4-byte Folded Reload
	scratch_load_dword v138, off, off offset:176 ; 4-byte Folded Reload
	v_cvt_pk_f16_f32 v149, v224, v254
	v_fma_f32 v224, v74, s14, 0
	v_cvt_pk_f16_f32 v143, v255, v227
	v_cvt_pk_f16_f32 v145, v228, v230
	v_fma_f32 v227, v76, s14, 0
	v_mfma_f32_32x32x16_f16 v[82:97], v[134:137], v[122:125], v[82:97]
	v_fma_f32 v228, v77, s14, 0
	v_fma_f32 v229, v78, s14, 0
	v_fma_f32 v230, v79, s14, 0
	v_fma_f32 v232, v81, s14, 0
	v_cvt_pk_f16_f32 v146, v249, v245
	v_cvt_pk_f16_f32 v148, v246, v252
	ds_bpermute_b32 v205, v211, v204
	v_mfma_f32_32x32x16_f16 v[82:97], v[130:133], v[114:117], v[82:97]
	s_waitcnt vmcnt(2)
	v_bitop3_b32 v0, v0, 12, v242 bitop3:0x48
	s_nop 7
	s_nop 1
	v_fma_f32 v233, v88, s14, 0
	v_fma_f32 v234, v89, s14, 0
	v_fma_f32 v235, v90, s14, 0
	v_fma_f32 v236, v95, s14, 0
	v_fma_f32 v237, v96, s14, 0
	v_fma_f32 v238, v97, s14, 0
	s_waitcnt vmcnt(1)
	v_or_b32_e32 v1, v0, v139
	s_waitcnt vmcnt(0)
	v_bitop3_b32 v1, v1, v244, v138 bitop3:0xde
	v_lshlrev_b32_e32 v206, 1, v1
	v_add_u32_e32 v1, s21, v206
	v_bitop3_b32 v0, v0, v138, v139 bitop3:0x36
	v_add_u32_e32 v138, s21, v207
	ds_read_b64_tr_b16 v[170:171], v1
	ds_read_b64_tr_b16 v[172:173], v138 offset:2048
	v_or_b32_e32 v1, v225, v244
	v_add_lshl_u32 v208, v0, v244, 1
	v_lshlrev_b32_e32 v1, 1, v1
	v_add_u32_e32 v0, s21, v208
	v_add_u32_e32 v132, s21, v1
	ds_read_b64_tr_b16 v[166:167], v0 offset:8192
	ds_read_b64_tr_b16 v[136:137], v0 offset:6144
	ds_read_b64_tr_b16 v[174:175], v0 offset:4096
	ds_read_b64_tr_b16 v[160:161], v0 offset:2048
	ds_read_b64_tr_b16 v[176:177], v138 offset:6144
	ds_read_b64_tr_b16 v[134:135], v138 offset:4096
	ds_read_b64_tr_b16 v[164:165], v138 offset:14336
	ds_read_b64_tr_b16 v[130:131], v138 offset:12288
	ds_read_b64_tr_b16 v[168:169], v138 offset:10240
	ds_read_b64_tr_b16 v[138:139], v138 offset:8192
	ds_read_b64_tr_b16 v[158:159], v132
	ds_read_b64_tr_b16 v[132:133], v0 offset:14336
	ds_read_b64_tr_b16 v[162:163], v0 offset:12288
	ds_read_b64_tr_b16 v[140:141], v0 offset:10240
	scratch_load_dword v70, off, off        ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[50:65], v[170:173], v[154:157], v[50:65]
	v_fma_f32 v172, v82, s14, 0
	v_max_f32_e32 v82, v214, v217
	v_max3_f32 v82, v82, v218, v219
	v_max3_f32 v82, v82, v220, v221
	v_fma_f32 v225, v75, s14, 0
	v_max3_f32 v82, v82, v222, v223
	v_max3_f32 v82, v82, v224, v225
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[50:65], v[174:177], v[150:153], v[50:65]
	v_max3_f32 v82, v82, v227, v228
	v_max3_f32 v82, v82, v229, v230
	v_fma_f32 v173, v83, s14, 0
	v_max3_f32 v82, v82, v231, v232
	v_fma_f32 v174, v84, s14, 0
	v_fma_f32 v175, v85, s14, 0
	v_max3_f32 v82, v82, v172, v173
	v_or_b32_e32 v0, v213, v244
	v_fma_f32 v176, v86, s14, 0
	v_fma_f32 v177, v87, s14, 0
	v_max3_f32 v82, v82, v174, v175
	v_lshlrev_b32_e32 v0, 1, v0
	v_max3_f32 v82, v82, v176, v177
	v_add_u32_e32 v209, s21, v0
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[50:65], v[166:169], v[146:149], v[50:65]
	v_fma_f32 v166, v91, s14, 0
	v_max3_f32 v82, v82, v233, v234
	v_add_lshl_u32 v213, v213, v244, 1
	v_fma_f32 v167, v92, s14, 0
	v_fma_f32 v168, v93, s14, 0
	v_max3_f32 v82, v82, v235, v166
	v_add_u32_e32 v170, s21, v213
	v_fma_f32 v169, v94, s14, 0
	v_max3_f32 v82, v82, v167, v168
	v_max3_f32 v82, v82, v169, v236
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[34:49], v[158:161], v[154:157], v[34:49]
	v_max3_f32 v158, v82, v237, v238
	ds_bpermute_b32 v159, v211, v158
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v212, v70, v244, 1
	v_add_u32_e32 v216, s21, v212
	ds_read_b64_tr_b16 v[66:67], v209
	ds_read_b64_tr_b16 v[68:69], v216 offset:2048
	v_or_b32_e32 v70, v70, v244
	v_lshlrev_b32_e32 v209, 1, v70
	v_add_u32_e32 v171, s21, v209
	ds_read_b64_tr_b16 v[70:71], v170 offset:8192
	ds_read_b64_tr_b16 v[76:77], v170 offset:6144
	ds_read_b64_tr_b16 v[78:79], v170 offset:4096
	ds_read_b64_tr_b16 v[84:85], v170 offset:2048
	ds_read_b64_tr_b16 v[80:81], v216 offset:6144
	ds_read_b64_tr_b16 v[74:75], v216 offset:4096
	ds_read_b64_tr_b16 v[88:89], v216 offset:14336
	ds_read_b64_tr_b16 v[90:91], v216 offset:12288
	ds_read_b64_tr_b16 v[72:73], v216 offset:10240
	ds_read_b64_tr_b16 v[94:95], v216 offset:8192
	ds_read_b64_tr_b16 v[82:83], v171
	ds_read_b64_tr_b16 v[92:93], v170 offset:14336
	ds_read_b64_tr_b16 v[86:87], v170 offset:12288
	ds_read_b64_tr_b16 v[96:97], v170 offset:10240
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[18:33], v[66:69], v[154:157], v[18:33]
	v_max3_f32 v216, v226, v158, v159
	v_sub_f32_e32 v66, v235, v216
	v_sub_f32_e32 v158, v214, v216
	v_sub_f32_e32 v159, v217, v216
	v_sub_f32_e32 v160, v218, v216
	v_sub_f32_e32 v161, v223, v216
	v_sub_f32_e32 v67, v166, v216
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[82:85], v[154:157], v[2:17]
	v_sub_f32_e32 v68, v167, v216
	v_sub_f32_e32 v69, v168, v216
	v_sub_f32_e32 v168, v237, v216
	v_sub_f32_e32 v170, v172, v216
	v_sub_f32_e32 v171, v177, v216
	v_sub_f32_e32 v172, v233, v216
	v_sub_f32_e32 v166, v169, v216
	v_mfma_f32_32x32x16_f16 v[34:49], v[134:137], v[150:153], v[34:49]
	v_sub_f32_e32 v134, v219, v216
	v_sub_f32_e32 v135, v220, v216
	v_sub_f32_e32 v136, v221, v216
	v_sub_f32_e32 v137, v222, v216
	v_sub_f32_e32 v167, v236, v216
	v_exp_f32_e32 v217, v158
	v_exp_f32_e32 v218, v159
	v_mfma_f32_32x32x16_f16 v[18:33], v[78:81], v[150:153], v[18:33]
	v_sub_f32_e32 v78, v238, v216
	v_exp_f32_e32 v238, v66
	v_sub_f32_e32 v66, v226, v216
	v_exp_f32_e32 v219, v160
	v_exp_f32_e32 v220, v134
	v_exp_f32_e32 v221, v135
	v_exp_f32_e32 v222, v136
	v_mfma_f32_32x32x16_f16 v[2:17], v[74:77], v[150:153], v[2:17]
	v_exp_f32_e32 v223, v137
	v_exp_f32_e32 v239, v67
	v_exp_f32_e32 v240, v68
	v_exp_f32_e32 v241, v69
	v_exp_f32_e32 v177, v168
	v_add3_u32 v74, s16, v199, v194
	v_add3_u32 v75, s16, v200, v194
	v_mfma_f32_32x32x16_f16 v[34:49], v[138:141], v[146:149], v[34:49]
	v_sub_f32_e32 v138, v227, v216
	v_sub_f32_e32 v139, v228, v216
	v_sub_f32_e32 v140, v229, v216
	v_sub_f32_e32 v141, v230, v216
	v_exp_f32_e32 v228, v138
	v_exp_f32_e32 v229, v139
	v_exp_f32_e32 v230, v140
	v_mfma_f32_32x32x16_f16 v[18:33], v[70:73], v[146:149], v[18:33]
	v_add3_u32 v70, s16, v195, v194
	v_add3_u32 v71, s16, v196, v194
	v_add3_u32 v72, s16, v197, v194
	v_add3_u32 v73, s16, v198, v194
	v_exp_f32_e32 v214, v78
	v_exp_f32_e32 v170, v170
	v_exp_f32_e32 v171, v171
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[2:17], v[94:97], v[146:149], v[2:17]
	v_exp_f32_e32 v172, v172
	v_mfma_f32_32x32x16_f16 v[50:65], v[162:165], v[142:145], v[50:65]
	v_sub_f32_e32 v162, v224, v216
	v_sub_f32_e32 v163, v225, v216
	v_sub_f32_e32 v164, v231, v216
	v_sub_f32_e32 v165, v232, v216
	v_exp_f32_e32 v225, v162
	v_add3_u32 v162, s16, v201, v194
	v_exp_f32_e32 v224, v161
	v_mfma_f32_32x32x16_f16 v[34:49], v[130:133], v[142:145], v[34:49]
	v_sub_f32_e32 v130, v173, v216
	v_sub_f32_e32 v131, v174, v216
	v_sub_f32_e32 v132, v175, v216
	v_sub_f32_e32 v133, v176, v216
	v_sub_f32_e32 v173, v234, v216
	v_exp_f32_e32 v227, v163
	v_exp_f32_e32 v231, v141
	v_mfma_f32_32x32x16_f16 v[18:33], v[86:89], v[142:145], v[18:33]
	v_exp_f32_e32 v232, v164
	v_exp_f32_e32 v233, v165
	v_exp_f32_e32 v234, v130
	v_exp_f32_e32 v235, v131
	v_exp_f32_e32 v236, v132
	v_exp_f32_e32 v237, v133
	v_exp_f32_e32 v174, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[90:93], v[142:145], v[2:17]
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[82:85], v70 offset:8192
	ds_read_b128 v[86:89], v71
	ds_read_b128 v[130:133], v71 offset:8192
	ds_read_b128 v[90:93], v72
	ds_read_b128 v[134:137], v72 offset:8192
	ds_read_b128 v[94:97], v73
	ds_read_b128 v[138:141], v73 offset:8192
	ds_read_b128 v[142:145], v74
	ds_read_b128 v[146:149], v74 offset:8192
	ds_read_b128 v[150:153], v75
	ds_read_b128 v[154:157], v75 offset:8192
	ds_read_b128 v[158:161], v162
	ds_read_b128 v[162:165], v162 offset:8192
	scratch_load_dword v168, off, off offset:92 ; 4-byte Folded Reload
	scratch_load_dword v169, off, off offset:132 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[196:197], off, off offset:144 ; 8-byte Folded Reload
	v_exp_f32_e32 v175, v166
	v_add3_u32 v166, s16, v202, v194
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[110:113], 0
	v_exp_f32_e32 v176, v167
	v_exp_f32_e32 v173, v173
	v_pk_mul_f32 v[64:65], v[64:65], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[174:175] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[106:109], v[66:81]
	v_pk_mul_f32 v[54:55], v[54:55], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[174:175] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[174:175] op_sel_hi:[1,0]
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v167, 0xff8, v168
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[66:81], v[90:93], v[102:105], v[66:81]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v194, 0x380, v169
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v195, v196, v194
	v_lshrrev_b32_e32 v168, 3, v168
	v_and_b32_e32 v168, 0x1f0, v168
	v_lshlrev_b32_e32 v86, 2, v167
	v_lshrrev_b32_e32 v90, 3, v194
	v_lshlrev_b32_e32 v91, 2, v195
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[66:81], v[94:97], v[98:101], v[66:81]
	v_or_b32_e32 v92, 0x400, v194
	v_or_b32_e32 v93, 0x800, v194
	v_or_b32_e32 v94, 0xc00, v194
	v_add3_u32 v196, s2, v168, v86
	v_add3_u32 v90, s2, v90, v91
	v_lshrrev_b32_e32 v92, 3, v92
	v_lshrrev_b32_e32 v93, 3, v93
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[142:145], v[126:129], v[66:81]
	v_lshrrev_b32_e32 v94, 3, v94
	ds_read_b128 v[86:89], v166
	ds_read_b128 v[166:169], v166 offset:8192
	ds_write_b128 v196, v[186:189]
	ds_write_b128 v196, v[190:193] offset:16
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add3_u32 v92, s2, v92, v91
	v_add3_u32 v93, s2, v93, v91
	v_add3_u32 v91, s2, v94, v91
	ds_read_b32 v94, v90
	ds_read_b32 v142, v92 offset:4096
	ds_read_b32 v143, v93 offset:8192
	ds_read_b32 v144, v91 offset:12288
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v196, v[178:181]
	ds_write_b128 v196, v[182:185] offset:16
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[66:81], v[150:153], v[118:121], v[66:81]
	s_barrier
	ds_read_b32 v145, v90
	ds_read_b32 v150, v92 offset:4096
	ds_read_b32 v151, v93 offset:8192
	ds_read_b32 v152, v91 offset:12288
	scratch_load_dword v90, off, off offset:116 ; 4-byte Folded Reload
	scratch_load_dword v184, off, off offset:104 ; 4-byte Folded Reload
	v_bfrev_b32_e32 v178, 1
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v153, s9, v90
	scratch_load_dword v90, off, off offset:120 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[66:81], v[158:161], v[122:125], v[66:81]
	v_readfirstlane_b32 s2, v153
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v158, s9, v90
	scratch_load_dword v90, off, off offset:124 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_f16 v[66:81], v[86:89], v[114:117], v[66:81]
	scratch_load_dwordx2 v[86:87], off, off offset:96 ; 8-byte Folded Reload
	ds_bpermute_b32 v88, v184, v94
	v_readfirstlane_b32 s15, v158
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v183, 1, v88
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v159, s9, v90
	scratch_load_dword v90, off, off offset:128 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshrrev_b64 v[86:87], v86, exec
	v_and_b32_e32 v182, 1, v86
	v_cmp_eq_u32_e32 vcc, 1, v182
	v_fma_f32 v66, v66, s14, 0
	v_fma_f32 v67, v67, s14, 0
	v_fma_f32 v68, v68, s14, 0
	v_fma_f32 v185, v73, s14, 0
	v_fma_f32 v186, v74, s14, 0
	v_fma_f32 v187, v75, s14, 0
	v_fma_f32 v188, v76, s14, 0
	v_fma_f32 v189, v77, s14, 0
	v_fma_f32 v190, v78, s14, 0
	v_fma_f32 v191, v79, s14, 0
	v_fma_f32 v192, v80, s14, 0
	v_fma_f32 v193, v81, s14, 0
	v_pk_mul_f32 v[80:81], v[174:175], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[174:175], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[174:175], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[174:175], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[174:175], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[174:175], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[174:175], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[174:175], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[174:175], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[174:175], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[174:175], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[174:175], v[10:11] op_sel_hi:[0,1]
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v160, s9, v90
	scratch_load_dword v90, off, off offset:136 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v161, s9, v90
	scratch_load_dword v90, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v179, s9, v90
	scratch_load_dword v90, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v180, s9, v90
	scratch_load_dword v90, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v181, s9, v90
	v_mfma_f32_32x32x16_f16 v[82:97], v[82:85], v[110:113], 0
	scratch_load_dwordx2 v[112:113], off, off offset:108 ; 8-byte Folded Reload
	v_cndmask_b32_e32 v110, v178, v183, vcc
	buffer_load_dword v110, s[4:7], 0 offen lds
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v159
	v_fma_f32 v183, v71, s14, 0
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v182, v112
	v_mfma_f32_32x32x16_f16 v[82:97], v[130:133], v[106:109], v[82:97]
	v_lshlrev_b32_e32 v113, 2, v182
	ds_bpermute_b32 v112, v113, v142
	ds_bpermute_b32 v107, v184, v143
	v_lshrrev_b64 v[110:111], v182, exec
	v_and_b32_e32 v110, 1, v110
	v_cmp_eq_u32_e64 s[2:3], 1, v110
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v106, 1, v112
	v_mfma_f32_32x32x16_f16 v[82:97], v[134:137], v[102:105], v[82:97]
	ds_bpermute_b32 v103, v113, v144
	v_cndmask_b32_e64 v106, v178, v106, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v102, 1, v107
	buffer_load_dword v106, s[4:7], 0 offen lds
	v_cndmask_b32_e32 v102, v178, v102, vcc
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v160
	v_mfma_f32_32x32x16_f16 v[82:97], v[138:141], v[98:101], v[82:97]
	buffer_load_dword v102, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v102, 1, v103
	ds_bpermute_b32 v98, v184, v145
	v_cndmask_b32_e64 v99, v178, v102, s[2:3]
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v161
	buffer_load_dword v99, s[4:7], 0 offen lds
	v_mfma_f32_32x32x16_f16 v[82:97], v[146:149], v[126:129], v[82:97]
	ds_bpermute_b32 v99, v113, v150
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v98, 1, v98
	v_cndmask_b32_e32 v98, v178, v98, vcc
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v179
	buffer_load_dword v98, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v98, 1, v99
	ds_bpermute_b32 v99, v184, v151
	v_mfma_f32_32x32x16_f16 v[82:97], v[154:157], v[118:121], v[82:97]
	v_cndmask_b32_e64 v98, v178, v98, s[2:3]
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v180
	buffer_load_dword v98, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v98, 1, v99
	ds_bpermute_b32 v99, v113, v152
	v_cndmask_b32_e32 v98, v178, v98, vcc
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[122:125], v[82:97]
	s_mov_b32 m0, s15
	v_fma_f32 v182, v70, s14, 0
	buffer_load_dword v98, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v98, 1, v99
	v_cndmask_b32_e64 v98, v178, v98, s[2:3]
	v_readfirstlane_b32 s2, v181
	s_mov_b32 m0, s2
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[114:117], v[82:97]
	buffer_load_dword v98, s[4:7], 0 offen lds
	v_add_f32_e32 v98, v217, v218
	v_add_f32_e32 v98, v219, v98
	v_add_f32_e32 v98, v220, v98
	v_add_f32_e32 v98, v221, v98
	v_add_f32_e32 v98, v222, v98
	v_add_f32_e32 v98, v223, v98
	v_fma_f32 v181, v69, s14, 0
	v_max_f32_e32 v69, v66, v67
	v_add_f32_e32 v98, v224, v98
	v_max3_f32 v69, v69, v68, v181
	v_add_f32_e32 v98, v225, v98
	v_fma_f32 v184, v72, s14, 0
	v_max3_f32 v69, v69, v182, v183
	v_add_f32_e32 v98, v227, v98
	v_max3_f32 v69, v69, v184, v185
	v_add_f32_e32 v98, v228, v98
	v_max3_f32 v69, v69, v186, v187
	v_add_f32_e32 v98, v229, v98
	v_max3_f32 v69, v69, v188, v189
	v_add_f32_e32 v98, v230, v98
	v_max3_f32 v69, v69, v190, v191
	v_add_f32_e32 v98, v231, v98
	v_fma_f32 v82, v82, s14, 0
	v_fma_f32 v83, v83, s14, 0
	v_max3_f32 v69, v69, v192, v193
	v_add_f32_e32 v98, v232, v98
	v_add_u32_e32 v114, s37, v206
	v_fma_f32 v84, v84, s14, 0
	v_fma_f32 v85, v85, s14, 0
	v_max3_f32 v69, v69, v82, v83
	v_add_f32_e32 v98, v233, v98
	s_waitcnt vmcnt(0)
	s_barrier
	v_add_u32_e32 v115, s13, v207
	ds_read_b64_tr_b16 v[166:167], v114 offset:32768
	ds_read_b64_tr_b16 v[168:169], v115 offset:2048
	v_fma_f32 v86, v86, s14, 0
	v_fma_f32 v87, v87, s14, 0
	v_max3_f32 v69, v69, v84, v85
	v_add_f32_e32 v98, v170, v98
	v_fma_f32 v88, v88, s14, 0
	v_fma_f32 v89, v89, s14, 0
	v_max3_f32 v69, v69, v86, v87
	v_add_f32_e32 v98, v234, v98
	v_fma_f32 v90, v90, s14, 0
	v_fma_f32 v194, v91, s14, 0
	v_max3_f32 v69, v69, v88, v89
	v_add_f32_e32 v98, v235, v98
	v_fma_f32 v92, v92, s14, 0
	v_fma_f32 v93, v93, s14, 0
	v_max3_f32 v69, v69, v90, v194
	v_add_f32_e32 v98, v236, v98
	v_cvt_pk_f16_f32 v110, v217, v218
	v_cvt_pk_f16_f32 v111, v219, v220
	v_cvt_pk_f16_f32 v112, v221, v222
	v_cvt_pk_f16_f32 v113, v223, v224
	v_fma_f32 v94, v94, s14, 0
	v_fma_f32 v95, v95, s14, 0
	v_max3_f32 v69, v69, v92, v93
	v_add_f32_e32 v98, v237, v98
	v_fma_f32 v96, v96, s14, 0
	v_fma_f32 v97, v97, s14, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[166:169], v[110:113], v[50:65]
	v_max3_f32 v69, v69, v94, v95
	v_add_f32_e32 v98, v171, v98
	v_max3_f32 v69, v69, v96, v97
	v_add_f32_e32 v98, v172, v98
	v_add_u32_e32 v114, s13, v208
	ds_bpermute_b32 v70, v211, v69
	v_add_f32_e32 v98, v173, v98
	v_cvt_pk_f16_f32 v102, v170, v234
	v_cvt_pk_f16_f32 v104, v237, v171
	v_cvt_pk_f16_f32 v105, v172, v173
	v_add_u32_e32 v116, s37, v1
	ds_read_b64_tr_b16 v[124:125], v114 offset:8192
	ds_read_b64_tr_b16 v[152:153], v114 offset:6144
	ds_read_b64_tr_b16 v[170:171], v114 offset:4096
	ds_read_b64_tr_b16 v[160:161], v114 offset:2048
	ds_read_b64_tr_b16 v[172:173], v115 offset:6144
	ds_read_b64_tr_b16 v[150:151], v115 offset:4096
	ds_read_b64_tr_b16 v[164:165], v115 offset:14336
	ds_read_b64_tr_b16 v[146:147], v115 offset:12288
	ds_read_b64_tr_b16 v[126:127], v115 offset:10240
	ds_read_b64_tr_b16 v[154:155], v115 offset:8192
	ds_read_b64_tr_b16 v[158:159], v116 offset:32768
	ds_read_b64_tr_b16 v[148:149], v114 offset:14336
	ds_read_b64_tr_b16 v[162:163], v114 offset:12288
	ds_read_b64_tr_b16 v[156:157], v114 offset:10240
	v_add_u32_e32 v114, s37, v0
	v_add_u32_e32 v116, s13, v212
	ds_read_b64_tr_b16 v[138:139], v114 offset:32768
	ds_read_b64_tr_b16 v[140:141], v116 offset:2048
	v_cvt_pk_f16_f32 v106, v225, v227
	v_cvt_pk_f16_f32 v107, v228, v229
	v_cvt_pk_f16_f32 v108, v230, v231
	v_cvt_pk_f16_f32 v109, v232, v233
	s_waitcnt lgkmcnt(14)
	v_max3_f32 v91, v216, v69, v70
	v_sub_f32_e32 v166, v66, v91
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[50:65], v[170:173], v[106:109], v[50:65]
	v_sub_f32_e32 v167, v67, v91
	v_sub_f32_e32 v168, v68, v91
	v_pk_mul_f32 v[72:73], v[174:175], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[174:175], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[174:175], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[174:175], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[174:175], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[174:175], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[174:175], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[174:175], v[18:19] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[66:81], v[158:161], v[110:113], v[66:81]
	v_cvt_pk_f16_f32 v103, v235, v236
	v_add_u32_e32 v180, s13, v213
	v_add_u32_e32 v117, s37, v209
	ds_read_b64_tr_b16 v[134:135], v180 offset:8192
	ds_read_b64_tr_b16 v[120:121], v180 offset:6144
	ds_read_b64_tr_b16 v[142:143], v180 offset:4096
	ds_read_b64_tr_b16 v[128:129], v180 offset:2048
	ds_read_b64_tr_b16 v[144:145], v116 offset:6144
	ds_read_b64_tr_b16 v[118:119], v116 offset:4096
	ds_read_b64_tr_b16 v[132:133], v116 offset:14336
	ds_read_b64_tr_b16 v[114:115], v116 offset:12288
	ds_read_b64_tr_b16 v[136:137], v116 offset:10240
	ds_read_b64_tr_b16 v[122:123], v116 offset:8192
	v_pk_mul_f32 v[24:25], v[174:175], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[174:175], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[174:175], v[4:5] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[34:49], v[138:141], v[110:113], v[34:49]
	v_pk_mul_f32 v[18:19], v[174:175], v[2:3] op_sel_hi:[0,1]
	v_add_f32_e32 v98, v238, v98
	v_add_f32_e32 v98, v239, v98
	v_add_f32_e32 v98, v240, v98
	v_add_f32_e32 v179, v241, v98
	v_cvt_pk_f16_f32 v98, v238, v239
	v_cvt_pk_f16_f32 v99, v240, v241
	v_mfma_f32_32x32x16_f16 v[50:65], v[124:127], v[102:105], v[50:65]
	ds_read_b64_tr_b16 v[126:127], v117 offset:32768
	ds_read_b64_tr_b16 v[116:117], v180 offset:14336
	ds_read_b64_tr_b16 v[130:131], v180 offset:12288
	ds_read_b64_tr_b16 v[124:125], v180 offset:10240
	v_cvt_pk_f16_f32 v100, v175, v176
	v_cvt_pk_f16_f32 v101, v177, v214
	v_sub_f32_e32 v90, v90, v91
	v_sub_f32_e32 v2, v216, v91
	v_sub_f32_e32 v158, v181, v91
	v_sub_f32_e32 v159, v182, v91
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[34:49], v[142:145], v[106:109], v[34:49]
	v_sub_f32_e32 v160, v183, v91
	v_sub_f32_e32 v161, v184, v91
	v_sub_f32_e32 v92, v92, v91
	v_sub_f32_e32 v93, v93, v91
	v_sub_f32_e32 v94, v94, v91
	v_sub_f32_e32 v95, v95, v91
	v_exp_f32_e32 v139, v166
	v_mfma_f32_32x32x16_f16 v[66:81], v[150:153], v[106:109], v[66:81]
	v_exp_f32_e32 v140, v167
	v_exp_f32_e32 v141, v168
	v_exp_f32_e32 v142, v158
	v_exp_f32_e32 v143, v159
	v_exp_f32_e32 v144, v160
	v_sub_f32_e32 v86, v86, v91
	v_sub_f32_e32 v87, v87, v91
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[18:33], v[126:129], v[110:113], v[18:33]
	v_sub_f32_e32 v88, v88, v91
	v_sub_f32_e32 v89, v89, v91
	v_sub_f32_e32 v150, v188, v91
	v_sub_f32_e32 v151, v189, v91
	v_sub_f32_e32 v152, v190, v91
	v_sub_f32_e32 v153, v191, v91
	v_sub_f32_e32 v169, v193, v91
	v_mfma_f32_32x32x16_f16 v[34:49], v[134:137], v[102:105], v[34:49]
	v_exp_f32_e32 v134, v161
	v_sub_f32_e32 v96, v96, v91
	v_sub_f32_e32 v97, v97, v91
	v_exp_f32_e32 v145, v150
	v_exp_f32_e32 v126, v153
	v_exp_f32_e32 v128, v169
	v_sub_f32_e32 v82, v82, v91
	v_mfma_f32_32x32x16_f16 v[66:81], v[154:157], v[102:105], v[66:81]
	v_sub_f32_e32 v83, v83, v91
	v_sub_f32_e32 v84, v84, v91
	v_sub_f32_e32 v85, v85, v91
	v_exp_f32_e32 v129, v82
	v_add_u32_e32 v1, s16, v1
	v_add_u32_e32 v0, s16, v0
	v_sub_f32_e32 v138, v194, v91
	v_mfma_f32_32x32x16_f16 v[18:33], v[118:121], v[106:109], v[18:33]
	v_exp_f32_e32 v118, v86
	v_exp_f32_e32 v119, v87
	v_exp_f32_e32 v120, v88
	v_exp_f32_e32 v121, v89
	v_cvt_pk_f16_f32 v86, v139, v140
	v_cvt_pk_f16_f32 v87, v141, v142
	v_cvt_pk_f16_f32 v88, v143, v144
	v_mfma_f32_32x32x16_f16 v[50:65], v[162:165], v[98:101], v[50:65]
	v_sub_f32_e32 v162, v185, v91
	v_exp_f32_e32 v135, v162
	v_sub_f32_e32 v163, v186, v91
	v_sub_f32_e32 v164, v187, v91
	v_sub_f32_e32 v165, v192, v91
	v_cvt_pk_f16_f32 v89, v134, v135
	v_exp_f32_e32 v136, v163
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[34:49], v[130:133], v[98:101], v[34:49]
	v_exp_f32_e32 v133, v90
	v_exp_f32_e32 v90, v2
	v_add_u32_e32 v2, s16, v206
	v_exp_f32_e32 v137, v164
	v_exp_f32_e32 v127, v165
	v_pk_mul_f32 v[16:17], v[64:65], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[62:63], v[90:91] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[146:149], v[98:101], v[66:81]
	v_exp_f32_e32 v148, v92
	v_pk_mul_f32 v[12:13], v[60:61], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[58:59], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[56:57], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[54:55], v[90:91] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[52:53], v[90:91] op_sel_hi:[1,0]
	v_exp_f32_e32 v146, v151
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[18:33], v[122:125], v[102:105], v[18:33]
	v_exp_f32_e32 v122, v93
	v_exp_f32_e32 v123, v94
	v_exp_f32_e32 v124, v95
	v_add_u32_e32 v104, s9, v207
	ds_read_b64_tr_b16 v[92:93], v2 offset:32768
	ds_read_b64_tr_b16 v[94:95], v104 offset:2048
	v_pk_mul_f32 v[2:3], v[50:51], v[90:91] op_sel_hi:[1,0]
	v_exp_f32_e32 v147, v152
	v_mfma_f32_32x32x16_f16 v[18:33], v[114:117], v[98:101], v[18:33]
	v_add_u32_e32 v114, s9, v208
	v_exp_f32_e32 v125, v96
	v_exp_f32_e32 v149, v97
	ds_read_b64_tr_b16 v[96:97], v114 offset:4096
	ds_read_b64_tr_b16 v[98:99], v104 offset:6144
	ds_read_b64_tr_b16 v[100:101], v104 offset:4096
	v_exp_f32_e32 v130, v83
	v_exp_f32_e32 v131, v84
	v_exp_f32_e32 v132, v85
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[2:17], v[92:95], v[86:89], v[2:17]
	v_cvt_pk_f16_f32 v82, v136, v137
	v_cvt_pk_f16_f32 v83, v145, v146
	v_cvt_pk_f16_f32 v84, v147, v126
	v_cvt_pk_f16_f32 v85, v127, v128
	ds_read_b64_tr_b16 v[52:53], v104 offset:10240
	ds_read_b64_tr_b16 v[50:51], v114 offset:8192
	ds_read_b64_tr_b16 v[102:103], v114 offset:6144
	ds_read_b64_tr_b16 v[106:107], v114 offset:2048
	v_cvt_pk_f16_f32 v92, v129, v130
	v_cvt_pk_f16_f32 v93, v131, v132
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[2:17], v[96:99], v[82:85], v[2:17]
	v_cvt_pk_f16_f32 v94, v118, v119
	v_cvt_pk_f16_f32 v95, v120, v121
	ds_read_b64_tr_b16 v[54:55], v114 offset:12288
	ds_read_b64_tr_b16 v[56:57], v104 offset:14336
	ds_read_b64_tr_b16 v[108:109], v104 offset:12288
	ds_read_b64_tr_b16 v[112:113], v104 offset:8192
	ds_read_b64_tr_b16 v[104:105], v1 offset:32768
	ds_read_b64_tr_b16 v[110:111], v114 offset:14336
	ds_read_b64_tr_b16 v[114:115], v114 offset:10240
	v_pk_mul_f32 v[62:63], v[90:91], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[90:91], v[76:77] op_sel_hi:[0,1]
	v_add_u32_e32 v78, s9, v212
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[2:17], v[50:53], v[92:95], v[2:17]
	v_pk_mul_f32 v[50:51], v[90:91], v[66:67] op_sel_hi:[0,1]
	v_add_f32_e32 v66, v139, v140
	v_add_f32_e32 v66, v141, v66
	v_add_f32_e32 v66, v142, v66
	v_add_f32_e32 v66, v143, v66
	v_add_f32_e32 v66, v144, v66
	v_add_f32_e32 v66, v134, v66
	v_add_f32_e32 v66, v135, v66
	v_pk_mul_f32 v[52:53], v[90:91], v[68:69] op_sel_hi:[0,1]
	v_add_f32_e32 v76, v136, v66
	ds_read_b64_tr_b16 v[66:67], v0 offset:32768
	ds_read_b64_tr_b16 v[68:69], v78 offset:2048
	v_exp_f32_e32 v138, v138
	v_pk_mul_f32 v[48:49], v[90:91], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[90:91], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[90:91], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[90:91], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[90:91], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[90:91], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[90:91], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[90:91], v[34:35] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v96, v133, v138
	v_cvt_pk_f16_f32 v97, v148, v122
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[34:49], v[66:69], v[86:89], v[34:49]
	v_cvt_pk_f16_f32 v98, v123, v124
	v_cvt_pk_f16_f32 v99, v125, v149
	v_add_u32_e32 v0, s9, v213
	v_pk_mul_f32 v[58:59], v[90:91], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[90:91], v[80:81] op_sel_hi:[0,1]
	v_add_f32_e32 v66, v137, v76
	v_add_f32_e32 v66, v145, v66
	v_mfma_f32_32x32x16_f16 v[2:17], v[54:57], v[96:99], v[2:17]
	v_pk_mul_f32 v[56:57], v[90:91], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[90:91], v[70:71] op_sel_hi:[0,1]
	ds_read_b64_tr_b16 v[70:71], v0 offset:4096
	ds_read_b64_tr_b16 v[72:73], v78 offset:6144
	ds_read_b64_tr_b16 v[74:75], v78 offset:4096
	v_add_f32_e32 v79, v146, v66
	ds_read_b64_tr_b16 v[68:69], v78 offset:10240
	ds_read_b64_tr_b16 v[66:67], v0 offset:8192
	ds_read_b64_tr_b16 v[76:77], v0 offset:6144
	ds_read_b64_tr_b16 v[80:81], v0 offset:2048
	v_pk_mul_f32 v[32:33], v[90:91], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[90:91], v[30:31] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[34:49], v[70:73], v[82:85], v[34:49]
	v_add_f32_e32 v70, v147, v79
	v_add_f32_e32 v70, v126, v70
	v_add_f32_e32 v79, v127, v70
	v_pk_mul_f32 v[28:29], v[90:91], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[90:91], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[90:91], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[90:91], v[22:23] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_f16 v[50:65], v[104:107], v[86:89], v[50:65]
	v_pk_mul_f32 v[20:21], v[90:91], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[90:91], v[18:19] op_sel_hi:[0,1]
	v_add_f32_e32 v1, v175, v179
	v_add_f32_e32 v1, v176, v1
	v_add_f32_e32 v1, v177, v1
	v_add_f32_e32 v1, v214, v1
	ds_bpermute_b32 v116, v211, v1
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[34:49], v[66:69], v[92:95], v[34:49]
	v_add_u32_e32 v67, s16, v209
	v_add_f32_e32 v66, v128, v79
	v_add_f32_e32 v66, v129, v66
	v_add_f32_e32 v66, v130, v66
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v116
	v_mfma_f32_32x32x16_f16 v[50:65], v[100:103], v[82:85], v[50:65]
	ds_read_b64_tr_b16 v[70:71], v0 offset:12288
	ds_read_b64_tr_b16 v[72:73], v78 offset:14336
	ds_read_b64_tr_b16 v[100:101], v78 offset:12288
	ds_read_b64_tr_b16 v[104:105], v78 offset:8192
	ds_read_b64_tr_b16 v[78:79], v67 offset:32768
	ds_read_b64_tr_b16 v[102:103], v0 offset:14336
	ds_read_b64_tr_b16 v[106:107], v0 offset:10240
	v_add_f32_e32 v0, v131, v66
	v_add_f32_e32 v0, v132, v0
	v_add_f32_e32 v0, v118, v0
	v_add_f32_e32 v0, v119, v0
	v_add_f32_e32 v0, v120, v0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[18:33], v[78:81], v[86:89], v[18:33]
	v_add_f32_e32 v0, v121, v0
	v_add_f32_e32 v0, v133, v0
	v_add_f32_e32 v0, v138, v0
	v_add_f32_e32 v0, v148, v0
	v_add_f32_e32 v0, v122, v0
	v_add_f32_e32 v0, v123, v0
	v_add_f32_e32 v0, v124, v0
	v_mfma_f32_32x32x16_f16 v[18:33], v[74:77], v[82:85], v[18:33]
	v_add_f32_e32 v0, v125, v0
	v_add_f32_e32 v0, v149, v0
	ds_bpermute_b32 v66, v211, v0
	v_add_f32_e32 v67, v204, v205
	v_fmac_f32_e32 v67, v215, v210
	v_fmac_f32_e32 v1, v67, v174
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[50:65], v[112:115], v[92:95], v[50:65]
	v_add_f32_e32 v0, v0, v66
	v_fmac_f32_e32 v0, v1, v90
	v_lshl_add_u32 v1, v203, 2, 0
	s_barrier
	v_mfma_f32_32x32x16_f16 v[18:33], v[104:107], v[92:95], v[18:33]
	v_mfma_f32_32x32x16_f16 v[50:65], v[108:111], v[96:99], v[50:65]
	v_mfma_f32_32x32x16_f16 v[34:49], v[70:73], v[96:99], v[34:49]
	v_mfma_f32_32x32x16_f16 v[18:33], v[100:103], v[96:99], v[18:33]
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v0
	v_mov_b32_e32 v67, 0x42000000
	v_or_b32_e32 v66, s34, v203
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v0, v68
	v_log_f32_e32 v68, v68
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v67, 0, v67, vcc
	v_cmp_gt_i32_e64 s[4:5], s2, v66
	v_sub_f32_e32 v66, v68, v67
	v_add_f32_e32 v66, v91, v66
	ds_write_b32 v1, v66
	v_mov_b32_e32 v66, 2
	v_lshlrev_b32_sdwa v66, v66, v242 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v67, 0, v66
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v67, v67
	s_sub_i32 s2, 0x4000, s34
	v_cmp_lt_i32_sdwa s[2:3], v242, s2 src0_sel:BYTE_0 src1_sel:DWORD
	s_and_b64 vcc, s[0:1], s[2:3]
	s_and_b32 s13, s8, 0xffff
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	v_cndmask_b32_e32 v66, v178, v66, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v66, s[12:15], 0 offen
	s_cbranch_execz .LBB0_5
	s_branch .LBB0_6
.LBB0_4:
                                        ; implicit-def: $sgpr4_sgpr5
.LBB0_5:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v0
	v_mov_b32_e32 v66, 0x42000000
	s_and_b32 s13, s8, 0xffff
	v_cndmask_b32_e64 v67, 0, 32, vcc
	v_ldexp_f32 v67, v0, v67
	v_log_f32_e32 v67, v67
	v_cndmask_b32_e32 v66, 0, v66, vcc
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	v_sub_f32_e32 v66, v67, v66
	v_add_f32_e32 v66, v91, v66
	ds_write_b32 v1, v66
	v_mov_b32_e32 v1, 2
	v_lshlrev_b32_sdwa v1, v1, v242 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v66, v66
	v_bfrev_b32_e32 v67, 1
	v_cndmask_b32_e64 v1, v67, v1, s[0:1]
	s_or_b64 s[4:5], s[4:5], exec
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v1, s[12:15], 0 offen
.LBB0_6:                                ; %.critedge
	v_div_scale_f32 v1, s[0:1], v0, v0, 1.0
	v_rcp_f32_e32 v1, v1
	v_div_scale_f32 v66, vcc, 1.0, v0, 1.0
	s_mul_i32 s0, s25, s18
	v_mul_f32_e32 v1, v66, v1
	s_ashr_i32 s1, s0, 31
	s_nop 0
	v_div_fmas_f32 v1, 0, 0, v1
	v_div_fixup_f32 v0, v1, v0, 1.0
	v_pk_mul_f32 v[24:25], v[0:1], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[0:1], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[0:1], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[0:1], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v25, v24, v25
	v_cvt_pk_f16_f32 v24, v22, v23
	v_cvt_pk_f16_f32 v21, v20, v21
	v_cvt_pk_f16_f32 v20, v18, v19
	v_pk_mul_f32 v[18:19], v[0:1], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[0:1], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[0:1], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v29, v28, v29
	v_cvt_pk_f16_f32 v28, v26, v27
	v_cvt_pk_f16_f32 v19, v18, v19
	v_cvt_pk_f16_f32 v18, v22, v23
	v_pk_mul_f32 v[22:23], v[0:1], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[0:1], v[42:43] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v33, v32, v33
	v_cvt_pk_f16_f32 v32, v30, v31
	v_cvt_pk_f16_f32 v23, v22, v23
	v_cvt_pk_f16_f32 v22, v26, v27
	v_pk_mul_f32 v[26:27], v[0:1], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[0:1], v[38:39] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v27, v26, v27
	v_cvt_pk_f16_f32 v26, v30, v31
	v_pk_mul_f32 v[30:31], v[0:1], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[0:1], v[34:35] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v31, v30, v31
	v_cvt_pk_f16_f32 v30, v34, v35
	v_pk_mul_f32 v[34:35], v[0:1], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[0:1], v[62:63] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v35, v34, v35
	v_cvt_pk_f16_f32 v34, v36, v37
	v_pk_mul_f32 v[36:37], v[0:1], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[0:1], v[58:59] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v37, v36, v37
	v_cvt_pk_f16_f32 v36, v38, v39
	v_pk_mul_f32 v[38:39], v[0:1], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[0:1], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[0:1], v[4:5] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v39, v38, v39
	v_cvt_pk_f16_f32 v38, v40, v41
	v_pk_mul_f32 v[40:41], v[0:1], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[0:1], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[0:1], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[0:1], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[0:1], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v1, v4, v5
	v_pk_mul_f32 v[2:3], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_cvt_pk_f16_f32 v0, v2, v3
	scratch_load_dword v3, off, off offset:88 ; 4-byte Folded Reload
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
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s27, 0x3fff
	v_mul_lo_u32 v2, s27, v203
	s_bitset1_b32 s2, 14
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cvt_pk_f16_f32 v9, v8, v9
	v_cvt_pk_f16_f32 v8, v6, v7
	v_cvt_pk_f16_f32 v13, v12, v13
	v_cvt_pk_f16_f32 v12, v10, v11
	v_cvt_pk_f16_f32 v17, v16, v17
	v_cvt_pk_f16_f32 v16, v14, v15
	v_cvt_pk_f16_f32 v41, v40, v41
	v_cvt_pk_f16_f32 v40, v42, v43
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v2, v2, v3, 1
	v_bfrev_b32_e32 v3, 1
	v_cndmask_b32_e64 v4, v3, v2, s[4:5]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v0, 16, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[8:9], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 32, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[12:13], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 48, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[16:17], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 64, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[40:41], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x50, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[38:39], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x60, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[36:37], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x70, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[34:35], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x80, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[30:31], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x90, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[26:27], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xa0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[22:23], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xb0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[18:19], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xc0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[20:21], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xd0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[24:25], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xe0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[28:29], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0xf0, v2
	v_cndmask_b32_e64 v0, v3, v0, s[4:5]
	buffer_store_dwordx2 v[32:33], v0, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 204
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
		.amdhsa_next_free_sgpr 46
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
	.set attn_fwd.numbered_sgpr, 46
	.set attn_fwd.private_seg_size, 204
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15408
; TotalNumSgprs: 52
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 204
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 52
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
	.short	676                             ; DW_AT_call_line
	.byte	52                              ; DW_AT_call_column
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
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"flash-attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/var/lib/jenkins/OAI-triton/python/../fa" ; string offset=26
.Linfo_string3:
	.asciz	"attn_fwd"                      ; string offset=67
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
    .private_segment_fixed_size: 204
    .sgpr_count:     52
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 50
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
