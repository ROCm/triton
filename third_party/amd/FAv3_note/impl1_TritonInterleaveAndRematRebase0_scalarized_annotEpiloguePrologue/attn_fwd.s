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
; %bb.11:
	.file	1 "/var/lib/jenkins/OAI-triton/fa" "flash-attention.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.12:
.LBB0_0:
	s_mul_i32 s20, s12, s18
	s_ashr_i32 s21, s20, 31
	s_lshl_b32 s34, s16, 8
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s12, s2, s20
	s_mul_i32 s2, s13, s17
	s_addc_u32 s16, s3, s21
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshrrev_b32_e32 v35, 4, v0
	s_add_u32 s12, s12, s2
	s_mul_i32 s2, s14, s34
	s_load_dwordx4 s[24:27], s[0:1], 0x38
	v_or_b32_e32 v2, 0x60, v35
	s_addc_u32 s13, s16, s3
	s_ashr_i32 s3, s2, 31
	v_or_b32_e32 v11, s34, v2
	s_lshl_b32 s16, s14, 6
	v_mul_lo_u32 v12, s14, v2
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v2, 3, v0
	v_or_b32_e32 v3, 0xa0, v35
	s_add_u32 s20, s12, s2
	v_and_b32_e32 v34, 0x78, v2
	s_mul_i32 s36, s15, s18
	v_or_b32_e32 v19, s34, v3
	v_mul_lo_u32 v20, s14, v3
	s_addc_u32 s12, s13, s3
	v_mad_u64_u32 v[2:3], s[2:3], s14, v35, v[34:35]
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[2:3], s[36:37], 1
	s_add_u32 s13, s4, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s38, s24, s17
	s_addc_u32 s15, s5, s3
	s_ashr_i32 s39, s38, 31
	s_lshl_b64 s[2:3], s[38:39], 1
	s_add_u32 s28, s13, s2
	s_mul_i32 s2, s26, s18
	s_addc_u32 s43, s15, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s6, s6, s2
	s_mul_i32 s2, s27, s17
	s_addc_u32 s7, s7, s3
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	v_or_b32_e32 v1, 32, v35
	v_or_b32_e32 v5, s34, v35
	s_add_u32 s24, s6, s2
	s_movk_i32 s2, 0x4000
	v_or_b32_e32 v6, s34, v1
	v_mul_lo_u32 v7, s14, v1
	v_add_u32_e32 v13, s16, v2
	s_addc_u32 s33, s7, s3
	s_and_b32 s3, s14, 0x3fff
	v_lshlrev_b32_e32 v2, 1, v2
	v_bfrev_b32_e32 v30, 1
	v_cmp_gt_i32_e32 vcc, s2, v5
	v_or_b32_e32 v10, 64, v5
	s_bitset1_b32 s3, 14
	v_cndmask_b32_e32 v14, v30, v2, vcc
	v_add_lshl_u32 v2, v7, v34, 1
	v_cmp_gt_i32_e32 vcc, s2, v6
	v_add_u32_e32 v29, s16, v13
	s_and_b32 s6, s12, 0xffff
	s_lshl_b32 s3, s3, 16
	v_cndmask_b32_e32 v15, v30, v2, vcc
	v_lshlrev_b32_e32 v13, 1, v13
	v_cmp_gt_i32_e32 vcc, s2, v10
	v_or_b32_e32 v18, 0x80, v5
	s_or_b32 s21, s6, s3
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v21, v30, v13, vcc
	v_add_lshl_u32 v10, v12, v34, 1
	v_cmp_gt_i32_e32 vcc, s2, v11
	v_or_b32_e32 v4, 0xe0, v35
	s_nop 0
	v_cndmask_b32_e32 v22, v30, v10, vcc
	buffer_load_dwordx4 v[10:13], v21, s[20:23], 0 offen
                                        ; kill: killed $vgpr21
	v_lshlrev_b32_e32 v21, 1, v29
	v_cmp_gt_i32_e32 vcc, s2, v18
	v_or_b32_e32 v26, 0xc0, v5
	v_or_b32_e32 v27, s34, v4
	v_mul_lo_u32 v28, s14, v4
	buffer_load_dwordx4 v[2:5], v14, s[20:23], 0 offen
	v_cndmask_b32_e32 v31, v30, v21, vcc
	v_add_lshl_u32 v18, v20, v34, 1
	v_cmp_gt_i32_e32 vcc, s2, v19
	buffer_load_dwordx4 v[6:9], v15, s[20:23], 0 offen
	s_nop 0
	v_cndmask_b32_e32 v32, v30, v18, vcc
	v_add_lshl_u32 v29, v29, s16, 1
	v_cmp_gt_i32_e32 vcc, s2, v26
	s_nop 1
	v_cndmask_b32_e32 v36, v30, v29, vcc
	v_add_lshl_u32 v26, v28, v34, 1
	v_cmp_gt_i32_e32 vcc, s2, v27
                                        ; kill: killed $vgpr15
                                        ; kill: killed $vgpr14
	buffer_load_dwordx4 v[14:17], v22, s[20:23], 0 offen
	s_nop 0
	v_cndmask_b32_e32 v37, v30, v26, vcc
                                        ; kill: killed $vgpr22
	buffer_load_dwordx4 v[18:21], v31, s[20:23], 0 offen
	buffer_load_dwordx4 v[22:25], v32, s[20:23], 0 offen
                                        ; kill: killed $vgpr31
                                        ; kill: killed $vgpr32
	buffer_load_dwordx4 v[26:29], v36, s[20:23], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[30:33], v37, s[20:23], 0 offen
                                        ; kill: killed $vgpr36
	v_and_b32_e32 v36, 1, v0
	v_cmp_eq_u32_e64 s[2:3], 0, v36
	v_and_b32_e32 v36, 2, v0
	v_cmp_eq_u32_e64 s[6:7], 0, v36
	v_and_b32_e32 v36, 4, v0
	v_cmp_eq_u32_e64 s[12:13], 0, v36
	v_and_b32_e32 v36, 8, v0
	v_cmp_eq_u32_e64 s[14:15], 0, v36
	v_and_b32_e32 v36, 0x80, v0
	v_and_b32_e32 v39, 0x100, v0
	v_lshrrev_b32_e32 v41, 1, v0
	v_lshrrev_b32_e32 v40, 1, v36
	scratch_store_dword off, v39, off offset:152 ; 4-byte Folded Spill
	v_or_b32_e32 v36, v36, v39
	v_and_b32_e32 v39, 56, v41
	v_xor_b32_e32 v39, v39, v34
	v_xor_b32_e32 v39, v39, v40
                                        ; kill: killed $vgpr37
	v_mul_lo_u32 v37, s25, v35
	v_lshl_add_u32 v39, v39, 1, 0
	v_lshlrev_b32_e32 v35, 8, v35
	v_add_u32_e32 v245, v39, v35
	v_and_b32_e32 v54, 31, v0
	s_movk_i32 s19, 0xe0
	s_barrier
	scratch_store_dword off, v40, off offset:148 ; 4-byte Folded Spill
	s_load_dword s35, s[0:1], 0x48
	v_and_b32_e32 v50, 16, v0
	v_lshrrev_b32_e32 v51, 3, v0
	v_lshrrev_b32_e32 v36, 3, v36
	v_mul_lo_u32 v38, s25, v1
	v_lshrrev_b32_e32 v1, 3, v50
	v_and_or_b32 v52, v51, 12, v36
	v_or_b32_e32 v36, v52, v1
                                        ; kill: killed $sgpr20_sgpr21
	v_bfe_i32 v102, v0, 0, 1
	v_bfe_i32 v103, v0, 1, 1
	v_bfe_i32 v104, v0, 2, 1
	s_lshl_b32 s40, s25, 6
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s42, s35, 6
	v_bfe_i32 v53, v0, 3, 1
	s_movk_i32 s16, 0x100
	s_waitcnt vmcnt(9)
	ds_write_b128 v245, v[10:13] offset:16384
	s_waitcnt vmcnt(8)
	ds_write_b128 v245, v[2:5]
	v_bfe_u32 v2, v0, 5, 1
	v_and_b32_e32 v4, 15, v0
	v_and_or_b32 v3, v41, s19, v54
	s_waitcnt vmcnt(7)
	ds_write_b128 v245, v[6:9] offset:8192
	v_xor_b32_e32 v5, v2, v4
	v_or_b32_e32 v6, 2, v2
	v_or_b32_e32 v7, 4, v2
	v_xor_b32_e32 v6, v6, v4
	v_xor_b32_e32 v7, v7, v4
	v_or_b32_e32 v8, 6, v2
	v_or_b32_e32 v9, 8, v2
	v_or_b32_e32 v10, 10, v2
	v_or_b32_e32 v11, 12, v2
	v_or_b32_e32 v2, 14, v2
	v_lshl_add_u32 v3, v3, 8, 0
	v_lshlrev_b32_e32 v12, 4, v5
	s_waitcnt vmcnt(6)
	ds_write_b128 v245, v[14:17] offset:24576
	v_xor_b32_e32 v8, v8, v4
	v_xor_b32_e32 v9, v9, v4
	v_xor_b32_e32 v10, v10, v4
	v_xor_b32_e32 v11, v11, v4
	v_xor_b32_e32 v2, v2, v4
	v_add_u32_e32 v4, v3, v12
	v_lshlrev_b32_e32 v13, 4, v6
	v_lshlrev_b32_e32 v14, 4, v7
	s_waitcnt vmcnt(5)
	ds_write_b128 v245, v[18:21] offset:32768
	s_waitcnt vmcnt(4)
	ds_write_b128 v245, v[22:25] offset:40960
	s_waitcnt vmcnt(3)
	ds_write_b128 v245, v[26:29] offset:49152
	s_waitcnt vmcnt(2)
	ds_write_b128 v245, v[30:33] offset:57344
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v5, v3, v13
	ds_read_b128 v[158:161], v4
	v_add_u32_e32 v4, v3, v14
	v_lshlrev_b32_e32 v15, 4, v8
	v_lshlrev_b32_e32 v16, 4, v9
	ds_read_b128 v[142:145], v5
	v_add_u32_e32 v5, v3, v15
	ds_read_b128 v[154:157], v4
	v_add_u32_e32 v4, v3, v16
	v_lshlrev_b32_e32 v17, 4, v10
	v_lshlrev_b32_e32 v18, 4, v11
	v_lshlrev_b32_e32 v19, 4, v2
	ds_read_b128 v[138:141], v5
	v_add_u32_e32 v5, v3, v17
	ds_read_b128 v[150:153], v4
	v_add_u32_e32 v4, v3, v18
	v_add_u32_e32 v2, v3, v19
	ds_read_b128 v[134:137], v5
	ds_read_b128 v[146:149], v4
	ds_read_b128 v[130:133], v2
	v_mad_u64_u32 v[10:11], s[20:21], s35, v36, v[34:35]
	scratch_store_dword off, v41, off offset:156 ; 4-byte Folded Spill


        ;;;;;; Prologue 0
        ;; GRK[0]
	; sched_barrier mask(0x00000000)
	s_and_b32 s19, s25, 0x3fff
	s_bitset1_b32 s19, 14
	s_and_b32 s20, s43, 0xffff
	s_lshl_b32 s19, s19, 16
	s_or_b32 s29, s20, s19
	s_mov_b32 s30, s22
	s_mov_b32 s31, s23
	v_add_lshl_u32 v105, v37, v34, 1
	v_add_lshl_u32 v106, v38, v34, 1
	buffer_load_dwordx4 v[2:5], v105, s[28:31], 0 offen
	buffer_load_dwordx4 v[6:9], v106, s[28:31], 0 offen
                                ; kill: killed $sgpr30_sgpr31 killed $sgpr29

        ;;;;;; Prologue 1
        ;; LWK[0] + GRK[1]
	; sched_barrier mask(0x00000000)
	s_ashr_i32 s41, s40, 31
	s_lshl_b64 s[30:31], s[40:41], 1
	s_add_u32 s20, s28, s30
	s_addc_u32 s29, s43, s31
	s_and_b32 s21, s29, 0xffff
	s_or_b32 s21, s21, s19
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[42:45], v105, s[20:23], 0 offen
	buffer_load_dwordx4 v[46:49], v106, s[20:23], 0 offen
                                        ; kill: killed $sgpr21
	s_waitcnt vmcnt(3)
	ds_write_b128 v245, v[2:5]
	s_waitcnt vmcnt(2)
	ds_write_b128 v245, v[6:9] offset:8192

        ;; LRK[0] + GRV[0]
	; sched_barrier mask(0x00000000)
	s_and_b32 s21, s35, 0x3fff
	s_bitset1_b32 s21, 14
	s_and_b32 s25, s33, 0xffff
	s_lshl_b32 s28, s21, 16
	s_or_b32 s25, s25, s28
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	v_lshlrev_b32_e32 v233, 1, v10
	v_add_lshl_u32 v225, v10, s35, 1
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[34:37], v233, s[24:27], 0 offen
	buffer_load_dwordx4 v[38:41], v225, s[24:27], 0 offen
	v_lshlrev_b32_e32 v2, 8, v54
	v_add3_u32 v244, 0, v12, v2
	v_add3_u32 v243, 0, v13, v2
	v_add3_u32 v242, 0, v14, v2
	v_add3_u32 v240, 0, v15, v2
	v_add3_u32 v238, 0, v16, v2
	v_add3_u32 v237, 0, v17, v2
	v_add3_u32 v236, 0, v18, v2
	v_add3_u32 v235, 0, v19, v2
	ds_read_b128 v[16:19], v244
	ds_read_b128 v[20:23], v243
	ds_read_b128 v[24:27], v242
	ds_read_b128 v[28:31], v240
	ds_read_b128 v[66:69], v240 offset:8192
	ds_read_b128 v[70:73], v238
	ds_read_b128 v[78:81], v237
	ds_read_b128 v[82:85], v237 offset:8192
	ds_read_b128 v[86:89], v236
	ds_read_b128 v[90:93], v236 offset:8192
	ds_read_b128 v[94:97], v235
	ds_read_b128 v[98:101], v235 offset:8192
	scratch_store_dword off, v54, off offset:144 ; 4-byte Folded Spill
	ds_read_b128 v[54:57], v244 offset:8192
	ds_read_b128 v[58:61], v243 offset:8192
	ds_read_b128 v[62:65], v242 offset:8192
	ds_read_b128 v[74:77], v238 offset:8192
                                ; kill: killed $sgpr26_sgpr27 killed $sgpr25


        ;;;;;; Prologue 2
        ;; DOT1[0] --> QK: 32;2:33
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[2:17], v[16:17], v[158:159], 0
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[2:17], v[18:19], v[160:161], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[20:21], v[142:143], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[22:23], v[144:145], v[2:17]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[2:17], v[24:25], v[154:155], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[26:27], v[156:157], v[2:17]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[2:17], v[28:29], v[138:139], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[30:31], v[140:141], v[2:17]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[18:33], v[54:55], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[18:33], v[56:57], v[160:161], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[70:71], v[150:151], v[2:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[58:59], v[142:143], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[72:73], v[152:153], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[60:61], v[144:145], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[78:79], v[134:135], v[2:17]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[18:33], v[62:63], v[154:155], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[80:81], v[136:137], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[64:65], v[156:157], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[86:87], v[146:147], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[66:67], v[138:139], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[88:89], v[148:149], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[68:69], v[140:141], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[94:95], v[130:131], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[150:151], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[96:97], v[132:133], v[2:17]
	v_mfma_f32_32x32x8_f16 v[18:33], v[76:77], v[152:153], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[82:83], v[134:135], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[84:85], v[136:137], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[90:91], v[146:147], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[92:93], v[148:149], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[98:99], v[130:131], v[18:33]
	v_mfma_f32_32x32x8_f16 v[18:33], v[100:101], v[132:133], v[18:33]


        ;; LWK[1] + GRK[2]
	; sched_barrier mask(0x00000000)
	s_add_u32 s20, s20, s30
	s_addc_u32 s21, s29, s31
	s_ashr_i32 s43, s42, 31
	s_lshl_b64 s[26:27], s[42:43], 1
	s_add_u32 s29, s24, s26
	s_addc_u32 s33, s33, s27
	s_and_b32 s21, s21, 0xffff
	s_or_b32 s21, s21, s19
	s_barrier
	buffer_load_dwordx4 v[110:113], v105, s[20:23], 0 offen
	v_mov_b32_e32 v234, v106
	buffer_load_dwordx4 v[106:109], v106, s[20:23], 0 offen
	v_mov_b32_e32 v250, v105
	s_waitcnt vmcnt(6)
	ds_write_b128 v245, v[42:45]
	s_waitcnt vmcnt(5)
	ds_write_b128 v245, v[46:49] offset:8192

        ;; VEC1[0]
        ;; v_exp escaped from this section ????
	; sched_barrier mask(0x00000000)
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	v_max_f32_e32 v42, v3, v3
	v_max_f32_e32 v43, v2, v2
	v_max_f32_e32 v42, v43, v42
	v_max3_f32 v42, v42, v4, v5
	v_max3_f32 v42, v42, v6, v7
	v_max3_f32 v42, v42, v8, v9
	v_max3_f32 v42, v42, v10, v11
	v_max3_f32 v42, v42, v12, v13
	v_max3_f32 v42, v42, v14, v15
	v_max3_f32 v42, v42, v16, v17
	v_max3_f32 v42, v42, v18, v19
	v_max3_f32 v42, v42, v20, v21
	v_max3_f32 v42, v42, v22, v23
	v_max3_f32 v42, v42, v24, v25
	v_max3_f32 v42, v42, v26, v27
	v_max3_f32 v42, v42, v28, v29
	v_max3_f32 v42, v42, v30, v31
	v_max3_f32 v43, v42, v32, v33
	v_lshlrev_b32_e32 v42, 2, v0
	v_xor_b32_e32 v194, 0x80, v42
	ds_bpermute_b32 v44, v194, v43
	v_mov_b32_e32 v42, 0xff800000
	s_mov_b32 s35, 0x3e0293ee
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v223, v43, v44, v42
	v_mul_f32_e32 v43, 0xbe0293ee, v223
	v_fmamk_f32 v2, v2, 0x3e0293ee, v43
	v_fmamk_f32 v3, v3, 0x3e0293ee, v43
	v_fmamk_f32 v4, v4, 0x3e0293ee, v43
	v_fmamk_f32 v5, v5, 0x3e0293ee, v43
	v_fmamk_f32 v6, v6, 0x3e0293ee, v43
	v_fmamk_f32 v7, v7, 0x3e0293ee, v43
	v_fmamk_f32 v8, v8, 0x3e0293ee, v43
	v_fmamk_f32 v9, v9, 0x3e0293ee, v43
	v_fmamk_f32 v10, v10, 0x3e0293ee, v43
	v_fmamk_f32 v11, v11, 0x3e0293ee, v43
	v_fmamk_f32 v12, v12, 0x3e0293ee, v43
	v_fmamk_f32 v13, v13, 0x3e0293ee, v43
	v_fmamk_f32 v14, v14, 0x3e0293ee, v43
	v_fmamk_f32 v15, v15, 0x3e0293ee, v43
	v_fmamk_f32 v16, v16, 0x3e0293ee, v43
	v_fmamk_f32 v17, v17, 0x3e0293ee, v43
	v_fmamk_f32 v18, v18, 0x3e0293ee, v43
	v_fmamk_f32 v19, v19, 0x3e0293ee, v43
	v_fmamk_f32 v20, v20, 0x3e0293ee, v43
	v_fmamk_f32 v21, v21, 0x3e0293ee, v43
	v_fmamk_f32 v22, v22, 0x3e0293ee, v43
	v_fmamk_f32 v23, v23, 0x3e0293ee, v43
	v_fmamk_f32 v24, v24, 0x3e0293ee, v43
	v_fmamk_f32 v25, v25, 0x3e0293ee, v43
	v_fmamk_f32 v26, v26, 0x3e0293ee, v43
	v_fmamk_f32 v27, v27, 0x3e0293ee, v43
	v_fmamk_f32 v28, v28, 0x3e0293ee, v43
	v_fmamk_f32 v29, v29, 0x3e0293ee, v43
	v_fmamk_f32 v30, v30, 0x3e0293ee, v43
	v_fmamk_f32 v31, v31, 0x3e0293ee, v43
	v_fmamk_f32 v32, v32, 0x3e0293ee, v43
	v_fmac_f32_e32 v43, 0x3e0293ee, v33
	v_fmac_f32_e32 v42, 0xbe0293ee, v223


        ;; GRV[1] + LWV[0] + LRK[1]
	; sched_barrier mask(0x00000000)
	s_and_b32 s20, s33, 0xffff
	s_or_b32 s21, s20, s28
	s_mov_b32 s20, s29
	s_barrier
	v_and_b32_e32 v33, 0x220, v102
	v_and_b32_e32 v44, 0x404, v103
	v_and_b32_e32 v46, 0x808, v104
	buffer_load_dwordx4 v[98:101], v233, s[20:23], 0 offen
	buffer_load_dwordx4 v[102:105], v225, s[20:23], 0 offen
	v_or_b32_e32 v45, v33, v44
	v_or_b32_e32 v47, v45, v46
	v_and_b32_e32 v48, 0x1010, v53
	v_or3_b32 v49, v48, v1, v47
	v_xor_b32_e32 v49, v52, v49
	s_mov_b32 s41, 0x5040100
	v_lshl_add_u32 v73, v49, 1, 0
	s_waitcnt vmcnt(5)
	v_perm_b32 v49, v38, v34, s41
	ds_write_b32 v73, v49 offset:16384
	v_or_b32_e32 v49, 0x44, v33
	v_xor_b32_e32 v49, v49, v44
	v_or_b32_e32 v53, v48, v46
	v_or3_b32 v49, v1, v49, v53
	v_xor_b32_e32 v49, v52, v49
	s_mov_b32 s42, 0x7060302
	v_lshl_add_u32 v67, v49, 1, 0
	v_perm_b32 v34, v38, v34, s42
	ds_read_b128 v[80:83], v244
	ds_read_b128 v[122:125], v244 offset:8192
	ds_read_b128 v[88:91], v243
	ds_read_b128 v[118:121], v243 offset:8192
	ds_read_b128 v[84:87], v242
	ds_read_b128 v[114:117], v242 offset:8192
	ds_read_b128 v[190:193], v240
	ds_read_b128 v[174:177], v240 offset:8192
	ds_read_b128 v[186:189], v238
	ds_read_b128 v[170:173], v238 offset:8192
	ds_read_b128 v[182:185], v237
	ds_read_b128 v[166:169], v237 offset:8192
	ds_read_b128 v[178:181], v236
	ds_read_b128 v[162:165], v236 offset:8192
	ds_read_b128 v[92:95], v235
	ds_read_b128 v[126:129], v235 offset:8192
	ds_write_b32 v67, v34 offset:16384
	v_or_b32_e32 v34, 0x88, v45
	v_xor_b32_e32 v34, v34, v46
	v_or3_b32 v34, v1, v34, v48
	v_xor_b32_e32 v34, v52, v34
	v_perm_b32 v38, v39, v35, s41
	v_lshl_add_u32 v68, v34, 1, 0
	ds_write_b32 v68, v38 offset:16384
	v_or_b32_e32 v34, 0xcc, v33
	v_or_b32_e32 v38, v46, v44
	v_xor_b32_e32 v34, v38, v34
	v_or3_b32 v34, v1, v34, v48
	v_xor_b32_e32 v34, v52, v34
	v_lshl_add_u32 v69, v34, 1, 0
	v_or_b32_e32 v34, 0x110, v47
	v_xor_b32_e32 v34, v34, v48
	v_or_b32_e32 v34, v34, v1
	v_xor_b32_e32 v34, v52, v34
	v_lshl_add_u32 v70, v34, 1, 0
	v_or_b32_e32 v34, 0x154, v33
	v_xor_b32_e32 v34, v34, v44
	v_or_b32_e32 v34, v34, v46
	v_xor_b32_e32 v34, v34, v48
	v_or_b32_e32 v34, v34, v1
	v_xor_b32_e32 v34, v52, v34
	v_lshl_add_u32 v71, v34, 1, 0
	v_or_b32_e32 v34, 0x198, v45
	v_xor_b32_e32 v34, v53, v34
	v_or_b32_e32 v34, v34, v1
	v_xor_b32_e32 v34, v52, v34
	v_lshl_add_u32 v72, v34, 1, 0
	v_or_b32_e32 v34, v38, v48
	v_or_b32_e32 v33, 0x1dc, v33
	v_xor_b32_e32 v33, v34, v33
	v_or_b32_e32 v1, v33, v1
	v_xor_b32_e32 v1, v52, v1
	v_cmp_gt_u32_e32 vcc, s16, v0
	s_movk_i32 s16, 0xff
	v_perm_b32 v35, v39, v35, s42
	v_perm_b32 v39, v40, v36, s41
	v_perm_b32 v36, v40, v36, s42
	v_perm_b32 v40, v41, v37, s41
	v_perm_b32 v37, v41, v37, s42
	v_lshl_add_u32 v66, v1, 1, 0
	v_cmp_lt_u32_e64 s[20:21], s16, v0
	ds_write_b32 v69, v35 offset:16384
	ds_write_b32 v70, v39 offset:16384
	ds_write_b32 v71, v36 offset:16384
	ds_write_b32 v72, v40 offset:16384
	ds_write_b32 v66, v37 offset:16384
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_store_dword off, v0, off offset:132 ; 4-byte Folded Spill
	s_and_saveexec_b64 s[24:25], s[20:21]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
        ;; section to prepare addr for LRV[0]
        ;; And v_exp escaped here ?????
        s_or_b64 exec, exec, s[24:25]
	v_exp_f32_e32 v248, v2
	v_exp_f32_e32 v246, v3
	v_mov_b32_e32 v2, 0x44
	v_mov_b32_e32 v3, 0x88
	v_cndmask_b32_e64 v2, v2, 0, s[2:3]
	v_cndmask_b32_e64 v3, v3, 0, s[6:7]
	v_exp_f32_e32 v249, v4
	v_exp_f32_e32 v195, v5
	v_exp_f32_e32 v197, v7
	v_exp_f32_e32 v202, v12
	v_or_b32_e32 v4, v2, v3
	v_mov_b32_e32 v5, 0x110
	v_mov_b32_e32 v7, 0x220
	v_or_b32_e32 v12, 8, v2
	v_exp_f32_e32 v205, v13
	v_cndmask_b32_e64 v5, v5, 0, s[12:13]
	v_cndmask_b32_e64 v7, v7, 0, s[14:15]
	v_xor_b32_e32 v12, v12, v3
	v_or_b32_e32 v13, 16, v4
	s_load_dwordx2 s[24:25], s[0:1], 0x4c
	s_load_dword s16, s[0:1], 0x54
	v_exp_f32_e32 v200, v9
	v_exp_f32_e32 v204, v14
	v_exp_f32_e32 v207, v15
	v_and_b32_e32 v40, 4, v51
	v_mov_b32_e32 v9, 0x404
	v_cmp_eq_u32_e64 s[0:1], 0, v50
	v_or3_b32 v12, v5, v12, v7
	v_xor_b32_e32 v13, v13, v5
	v_or_b32_e32 v14, 24, v2
	v_or_b32_e32 v15, v5, v3
	v_cndmask_b32_e64 v9, v9, 0, s[0:1]
	v_xor_b32_e32 v12, v40, v12
	v_or_b32_e32 v13, v13, v7
	v_xor_b32_e32 v14, v15, v14
	v_exp_f32_e32 v1, v6
	v_or_b32_e32 v6, v4, v5
	v_xor_b32_e32 v12, v12, v9
	v_xor_b32_e32 v13, v40, v13
	v_or_b32_e32 v14, v14, v7
	v_exp_f32_e32 v196, v8
	v_exp_f32_e32 v199, v10
	v_exp_f32_e32 v209, v17
	v_or_b32_e32 v8, v6, v7
	v_xor_b32_e32 v10, v9, v40
	v_xor_b32_e32 v13, v13, v9
	v_xor_b32_e32 v14, v40, v14
	v_or_b32_e32 v17, 40, v2
	v_lshl_add_u32 v0, v12, 1, 0
	v_exp_f32_e32 v203, v11
	v_exp_f32_e32 v206, v16
	v_xor_b32_e32 v11, v10, v8
	v_xor_b32_e32 v14, v14, v9
	v_or_b32_e32 v16, 32, v6
	v_or_b32_e32 v10, v10, v7
	v_xor_b32_e32 v17, v17, v3
	scratch_store_dword off, v0, off offset:84 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v13, 1, 0
	v_exp_f32_e32 v212, v22
	v_xor_b32_e32 v16, v10, v16
	v_or_b32_e32 v17, v17, v5
	v_or_b32_e32 v22, 0x808, v2
	scratch_store_dword off, v0, off offset:88 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v14, 1, 0
	v_exp_f32_e32 v208, v18
	v_exp_f32_e32 v211, v19
	v_xor_b32_e32 v17, v10, v17
	v_or_b32_e32 v18, 48, v4
	v_or_b32_e32 v19, v10, v5
	v_xor_b32_e32 v22, v22, v3
	scratch_store_dword off, v0, off offset:92 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v16, 1, 0
	v_exp_f32_e32 v210, v20
	v_exp_f32_e32 v213, v21
	v_exp_f32_e32 v214, v24
	v_xor_b32_e32 v18, v19, v18
	v_or_b32_e32 v20, 56, v2
	v_or_b32_e32 v21, v19, v3
	v_or3_b32 v22, v5, v22, v7
	v_or_b32_e32 v24, 0x818, v2
	scratch_store_dword off, v0, off offset:96 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v17, 1, 0
	v_exp_f32_e32 v215, v23
	v_exp_f32_e32 v217, v25
	v_xor_b32_e32 v20, v21, v20
	v_xor_b32_e32 v22, v40, v22
	v_or_b32_e32 v23, 0x800, v8
	v_xor_b32_e32 v24, v15, v24
	v_or_b32_e32 v25, 0x810, v4
	scratch_store_dword off, v0, off offset:76 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v18, 1, 0
	v_xor_b32_e32 v22, v22, v9
	v_xor_b32_e32 v23, v40, v23
	v_or_b32_e32 v24, v24, v7
	v_xor_b32_e32 v25, v25, v5
	scratch_store_dword off, v0, off offset:100 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v20, 1, 0
	v_exp_f32_e32 v216, v26
	v_xor_b32_e32 v23, v23, v9
	v_xor_b32_e32 v24, v40, v24
	v_or_b32_e32 v25, v25, v7
	v_or_b32_e32 v26, 0x828, v2
	scratch_store_dword off, v0, off offset:104 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v22, 1, 0
	v_xor_b32_e32 v24, v24, v9
	v_xor_b32_e32 v25, v40, v25
	v_xor_b32_e32 v26, v26, v3
	scratch_store_dword off, v0, off offset:108 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v23, 1, 0
	v_xor_b32_e32 v25, v25, v9
	v_or_b32_e32 v26, v26, v5
	scratch_store_dword off, v0, off offset:80 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v24, 1, 0
	v_exp_f32_e32 v219, v27
	v_exp_f32_e32 v220, v30
	v_xor_b32_e32 v26, v10, v26
	v_or_b32_e32 v27, 0x820, v6
	v_or_b32_e32 v30, 0x1010, v4
	scratch_store_dword off, v0, off        ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v25, 1, 0
	v_exp_f32_e32 v218, v28
	v_exp_f32_e32 v222, v31
	v_xor_b32_e32 v27, v10, v27
	v_or_b32_e32 v28, 0x838, v2
	v_xor_b32_e32 v30, v30, v5
	v_or_b32_e32 v31, 0x1018, v2
	scratch_store_dword off, v0, off offset:4 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v26, 1, 0
	v_exp_f32_e32 v221, v29
	v_xor_b32_e32 v28, v21, v28
	v_or_b32_e32 v29, 0x830, v4
	v_or_b32_e32 v30, v30, v7
	v_xor_b32_e32 v31, v15, v31
	scratch_store_dword off, v0, off offset:8 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v27, 1, 0
	v_xor_b32_e32 v29, v19, v29
	v_xor_b32_e32 v30, v40, v30
	v_or_b32_e32 v31, v31, v7
	v_or_b32_e32 v33, 0x1008, v2
	scratch_store_dword off, v0, off offset:12 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v28, 1, 0
	v_exp_f32_e32 v198, v32
	v_xor_b32_e32 v30, v30, v9
	v_xor_b32_e32 v31, v40, v31
	v_or_b32_e32 v32, 0x1000, v8
	v_xor_b32_e32 v33, v33, v3
	scratch_store_dword off, v0, off offset:16 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v29, 1, 0
	v_xor_b32_e32 v31, v31, v9
	v_xor_b32_e32 v32, v40, v32
	v_or3_b32 v33, v5, v33, v7
	scratch_store_dword off, v0, off offset:20 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v30, 1, 0
	v_xor_b32_e32 v32, v32, v9
	v_xor_b32_e32 v33, v40, v33
	scratch_store_dword off, v0, off offset:24 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v31, 1, 0
	v_xor_b32_e32 v33, v33, v9
	v_or_b32_e32 v34, 0x1030, v4
	scratch_store_dword off, v0, off offset:28 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v32, 1, 0
	v_xor_b32_e32 v34, v19, v34
	v_or_b32_e32 v35, 0x1038, v2
	v_or_b32_e32 v37, 0x1028, v2
	v_or_b32_e32 v38, 0x1818, v2
	scratch_store_dword off, v0, off offset:32 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v33, 1, 0
	v_xor_b32_e32 v35, v21, v35
	v_or_b32_e32 v36, 0x1020, v6
	v_xor_b32_e32 v37, v37, v3
	v_xor_b32_e32 v15, v15, v38
	v_or_b32_e32 v38, 0x1810, v4
	scratch_store_dword off, v0, off offset:36 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v34, 1, 0
	v_xor_b32_e32 v36, v10, v36
	v_or_b32_e32 v37, v37, v5
	v_or_b32_e32 v15, v15, v7
	v_xor_b32_e32 v38, v38, v5
	v_or_b32_e32 v39, 0x1808, v2
	scratch_store_dword off, v0, off offset:40 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v35, 1, 0
	v_xor_b32_e32 v37, v10, v37
	v_xor_b32_e32 v15, v40, v15
	v_or_b32_e32 v38, v38, v7
	v_xor_b32_e32 v39, v39, v3
	scratch_store_dword off, v0, off offset:44 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v36, 1, 0
	v_xor_b32_e32 v15, v15, v9
	v_xor_b32_e32 v38, v40, v38
	v_or3_b32 v7, v5, v39, v7
	v_or_b32_e32 v8, 0x1800, v8
	scratch_store_dword off, v0, off offset:48 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v37, 1, 0
	v_xor_b32_e32 v38, v38, v9
	v_xor_b32_e32 v7, v40, v7
	v_xor_b32_e32 v8, v40, v8
	scratch_store_dword off, v0, off offset:52 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v15, 1, 0
	s_add_u32 s0, s36, s38
	v_xor_b32_e32 v7, v7, v9
	v_xor_b32_e32 v8, v8, v9
	v_or_b32_e32 v9, 0x1838, v2
	v_or_b32_e32 v2, 0x1828, v2
	scratch_store_dword off, v0, off offset:56 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v38, 1, 0
	s_addc_u32 s1, s37, s39
	v_xor_b32_e32 v2, v2, v3
	scratch_store_dword off, v0, off offset:60 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v7, 1, 0
	s_mul_i32 s3, s40, 6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_exp_f32_e32 v201, v43
	v_exp_f32_e32 v247, v42
	v_xor_b32_e32 v9, v21, v9
	v_or_b32_e32 v2, v2, v5
	scratch_store_dword off, v0, off offset:64 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v8, 1, 0
	s_mul_hi_i32 s2, s40, 6
	s_add_u32 s0, s3, s0
	v_or_b32_e32 v4, 0x1830, v4
	v_xor_b32_e32 v2, v10, v2
	v_or_b32_e32 v3, 0x1820, v6
	scratch_store_dword off, v0, off offset:68 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v9, 1, 0
	s_addc_u32 s1, s2, s1
	v_xor_b32_e32 v4, v19, v4
	v_xor_b32_e32 v3, v10, v3
	scratch_store_dword off, v0, off offset:72 ; 4-byte Folded Spill
	v_lshl_add_u32 v0, v2, 1, 0
	s_add_u32 s0, s4, s0
	v_mov_b32_e32 v2, 0
	s_waitcnt vmcnt(29)
	v_lshrrev_b32_e32 v252, 16, v102
	scratch_store_dword off, v40, off offset:128 ; 4-byte Folded Spill
	v_lshl_add_u32 v224, v11, 1, 0
	v_lshl_add_u32 v241, v4, 1, 0
	v_lshl_add_u32 v251, v3, 1, 0
	s_addc_u32 s1, s5, s1
	v_mov_b32_e32 v255, 1.0
	s_movk_i32 s2, 0xffc0
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
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v226, v67
	v_mov_b32_e32 v227, v68
	v_mov_b32_e32 v228, v69
	v_mov_b32_e32 v229, v70
	v_mov_b32_e32 v230, v71
	v_mov_b32_e32 v231, v72
	v_mov_b32_e32 v232, v66
	v_mov_b32_e32 v239, v73
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[80:81], v[158:159], 0
	v_mov_b32_e32 v254, v224
	v_mov_b32_e32 v224, v255
	v_mov_b32_e32 v253, v223
	s_setprio 0
	v_mfma_f32_32x32x8_f16 v[66:81], v[82:83], v[160:161], v[66:81]
	v_mul_f32_e32 v17, v17, v247
	v_mul_f32_e32 v50, v50, v247
	v_mul_f32_e32 v16, v16, v247
	v_mul_f32_e32 v15, v15, v247
	v_mul_f32_e32 v14, v14, v247
	v_mul_f32_e32 v13, v13, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[88:89], v[142:143], v[66:81]
	v_mul_f32_e32 v11, v11, v247
	v_mul_f32_e32 v12, v12, v247
	v_mul_f32_e32 v10, v10, v247
	v_mul_f32_e32 v9, v9, v247
	v_mul_f32_e32 v8, v8, v247
	v_mul_f32_e32 v7, v7, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[144:145], v[66:81]
	v_mul_f32_e32 v5, v5, v247
	v_mul_f32_e32 v6, v6, v247
	v_mul_f32_e32 v4, v4, v247
	v_mul_f32_e32 v3, v3, v247
	v_mul_f32_e32 v2, v2, v247
	v_mul_f32_e32 v33, v33, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[84:85], v[154:155], v[66:81]
	v_mul_f32_e32 v31, v31, v247
	v_mul_f32_e32 v32, v32, v247
	v_mul_f32_e32 v30, v30, v247
	v_mul_f32_e32 v29, v29, v247
	v_mul_f32_e32 v28, v28, v247
	v_mul_f32_e32 v27, v27, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[86:87], v[156:157], v[66:81]
	v_mul_f32_e32 v25, v25, v247
	v_mul_f32_e32 v26, v26, v247
	v_mul_f32_e32 v24, v24, v247
	v_mul_f32_e32 v23, v23, v247
	v_mul_f32_e32 v22, v22, v247
	v_mul_f32_e32 v21, v21, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[138:139], v[66:81]
	v_mul_f32_e32 v19, v19, v247
	v_mul_f32_e32 v20, v20, v247
	v_mul_f32_e32 v18, v18, v247
	v_mul_f32_e32 v49, v49, v247
	v_mul_f32_e32 v48, v48, v247
	v_mul_f32_e32 v47, v47, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[192:193], v[140:141], v[66:81]
	v_mul_f32_e32 v45, v45, v247
	v_mul_f32_e32 v46, v46, v247
	v_mul_f32_e32 v44, v44, v247
	v_mul_f32_e32 v43, v43, v247
	v_mul_f32_e32 v42, v42, v247
	v_mul_f32_e32 v41, v41, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[150:151], v[66:81]
	v_mul_f32_e32 v39, v39, v247
	v_mul_f32_e32 v40, v40, v247
	v_mul_f32_e32 v38, v38, v247
	v_mul_f32_e32 v37, v37, v247
	v_mul_f32_e32 v36, v36, v247
	v_mul_f32_e32 v35, v35, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[152:153], v[66:81]
	v_mul_f32_e32 v65, v65, v247
	v_mul_f32_e32 v34, v34, v247
	v_mul_f32_e32 v64, v64, v247
	v_mul_f32_e32 v63, v63, v247
	v_mul_f32_e32 v62, v62, v247
	v_mul_f32_e32 v61, v61, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[134:135], v[66:81]
	v_mul_f32_e32 v59, v59, v247
	v_mul_f32_e32 v60, v60, v247
	v_mul_f32_e32 v58, v58, v247
	v_mul_f32_e32 v57, v57, v247
	v_mul_f32_e32 v56, v56, v247
	v_mul_f32_e32 v55, v55, v247
	v_mfma_f32_32x32x8_f16 v[66:81], v[184:185], v[136:137], v[66:81]
	v_mul_f32_e32 v53, v53, v247
	v_mul_f32_e32 v54, v54, v247
	v_mul_f32_e32 v52, v52, v247
	v_mul_f32_e32 v51, v51, v247
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[180:181], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[132:133], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[122:123], v[158:159], 0
	v_cvt_pkrtz_f16_f32 v122, v208, v211
	v_cvt_pkrtz_f16_f32 v123, v210, v213
	v_mfma_f32_32x32x8_f16 v[82:97], v[124:125], v[160:161], v[82:97]
	v_cvt_pkrtz_f16_f32 v124, v212, v215
	v_cvt_pkrtz_f16_f32 v125, v214, v217
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[142:143], v[82:97]
	v_cvt_pkrtz_f16_f32 v118, v199, v203
	v_cvt_pkrtz_f16_f32 v119, v202, v205
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[144:145], v[82:97]
	v_cvt_pkrtz_f16_f32 v120, v204, v207
	v_cvt_pkrtz_f16_f32 v121, v206, v209
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[154:155], v[82:97]
	v_add_f32_e32 v114, v248, v246
	v_add_f32_e32 v114, v114, v249
	v_add_f32_e32 v114, v114, v195
	v_add_f32_e32 v114, v114, v1
	v_add_f32_e32 v114, v114, v197
	v_add_f32_e32 v114, v114, v196
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[156:157], v[82:97]
	v_add_f32_e32 v114, v114, v200
	v_cvt_pkrtz_f16_f32 v117, v196, v200
	v_add_f32_e32 v114, v114, v199
	v_cvt_pkrtz_f16_f32 v116, v1, v197
	v_add_f32_e32 v114, v114, v203
	v_add_f32_e32 v114, v114, v202
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[138:139], v[82:97]
	v_add_f32_e32 v114, v114, v205
	v_add_f32_e32 v114, v114, v204
	v_add_f32_e32 v114, v114, v207
	v_add_f32_e32 v114, v114, v206
	v_add_f32_e32 v114, v114, v209
	v_add_f32_e32 v114, v114, v208
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[140:141], v[82:97]
	v_add_f32_e32 v114, v114, v211
	v_add_f32_e32 v114, v114, v210
	v_add_f32_e32 v114, v114, v213
	v_add_f32_e32 v114, v114, v212
	v_add_f32_e32 v114, v114, v215
	v_add_f32_e32 v114, v114, v214
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[150:151], v[82:97]
	v_add_f32_e32 v114, v114, v217
	v_add_f32_e32 v114, v114, v216
	v_add_f32_e32 v114, v114, v219
	v_add_f32_e32 v114, v114, v218
	v_add_f32_e32 v114, v114, v221
	v_add_f32_e32 v114, v114, v220
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[152:153], v[82:97]
	v_add_f32_e32 v114, v114, v222
	v_add_f32_e32 v114, v114, v198
	v_add_f32_e32 v114, v114, v201
	ds_bpermute_b32 v115, v194, v114
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v255, v114, v115
	v_fmac_f32_e32 v255, v224, v247
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[134:135], v[82:97]
	v_mov_b32_e32 v224, v254
	v_cvt_pkrtz_f16_f32 v115, v249, v195
	v_cvt_pkrtz_f16_f32 v114, v248, v246
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[136:137], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[146:147], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[126:127], v[130:131], v[82:97]
	v_cvt_pkrtz_f16_f32 v126, v216, v219
	v_cvt_pkrtz_f16_f32 v127, v218, v221
	v_mfma_f32_32x32x8_f16 v[82:97], v[128:129], v[132:133], v[82:97]
	v_cvt_pkrtz_f16_f32 v128, v220, v222
	v_cvt_pkrtz_f16_f32 v129, v198, v201
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_barrier
	s_barrier
	scratch_load_dword v1, off, off offset:84 ; 4-byte Folded Reload
	s_add_u32 s12, s29, s26
	s_addc_u32 s3, s33, s27
	s_and_b32 s4, s1, 0xffff
	s_or_b32 s21, s4, s19
	s_mov_b32 s20, s0
	ds_read_b64 v[162:163], v254 offset:16384
	ds_read_b64 v[222:223], v241 offset:16384
	ds_read_b64 v[246:247], v0 offset:16384
	ds_read_b64 v[248:249], v251 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[164:165], v1 offset:16384
	scratch_load_dword v1, off, off offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[166:167], v1 offset:16384
	scratch_load_dword v1, off, off offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[168:169], v1 offset:16384
	scratch_load_dword v1, off, off offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[170:171], v1 offset:16384
	scratch_load_dword v1, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[172:173], v1 offset:16384
	scratch_load_dword v1, off, off offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[174:175], v1 offset:16384
	scratch_load_dword v1, off, off offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[176:177], v1 offset:16384
	scratch_load_dword v1, off, off offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[178:179], v1 offset:16384
	scratch_load_dword v1, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[180:181], v1 offset:16384
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[182:183], v1 offset:16384
	scratch_load_dword v1, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[184:185], v1 offset:16384
	scratch_load_dword v1, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[186:187], v1 offset:16384
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[188:189], v1 offset:16384
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[190:191], v1 offset:16384
	scratch_load_dword v1, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[192:193], v1 offset:16384
	scratch_load_dword v1, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[196:197], v1 offset:16384
	scratch_load_dword v1, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[198:199], v1 offset:16384
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[200:201], v1 offset:16384
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[202:203], v1 offset:16384
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[204:205], v1 offset:16384
	scratch_load_dword v1, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[206:207], v1 offset:16384
	scratch_load_dword v1, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[208:209], v1 offset:16384
	scratch_load_dword v1, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[210:211], v1 offset:16384
	scratch_load_dword v1, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v1 offset:16384
	scratch_load_dword v1, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v1 offset:16384
	scratch_load_dword v1, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[216:217], v1 offset:16384
	scratch_load_dword v1, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[218:219], v1 offset:16384
	scratch_load_dword v1, off, off offset:72 ; 4-byte Folded Reload
	ds_write_b128 v245, v[110:113]
	ds_write_b128 v245, v[106:109] offset:8192
	buffer_load_dwordx4 v[110:113], v250, s[20:23], 0 offen
	buffer_load_dwordx4 v[106:109], v234, s[20:23], 0 offen
	s_waitcnt vmcnt(2)
	ds_read_b64 v[220:221], v1 offset:16384
	; sched_barrier mask(0x00000000)
	s_barrier
	s_setprio 0
	v_max_f32_e32 v1, v67, v67
	; iglp_opt mask(0x0000000A)
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[50:65], v[162:163], v[114:115], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[114:115], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[196:197], v[114:115], v[18:33]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[2:17], v[212:213], v[114:115], v[2:17]
	v_max_f32_e32 v114, v66, v66
	v_max_f32_e32 v1, v114, v1
	v_max3_f32 v1, v1, v68, v69
	v_max3_f32 v1, v1, v70, v71
	v_max3_f32 v1, v1, v72, v73
	v_max3_f32 v1, v1, v74, v75
	v_mfma_f32_32x32x8_f16 v[18:33], v[198:199], v[116:117], v[18:33]
	v_max3_f32 v1, v1, v76, v77
	v_max3_f32 v1, v1, v78, v79
	v_max3_f32 v1, v1, v80, v81
	v_max3_f32 v1, v1, v82, v83
	v_max3_f32 v1, v1, v84, v85
	v_max3_f32 v1, v1, v86, v87
	v_mfma_f32_32x32x8_f16 v[18:33], v[200:201], v[118:119], v[18:33]
	v_max3_f32 v1, v1, v88, v89
	v_max3_f32 v1, v1, v90, v91
	v_max3_f32 v1, v1, v92, v93
	v_max3_f32 v1, v1, v94, v95
	v_max3_f32 v1, v1, v96, v97
	ds_bpermute_b32 v114, v194, v1
	v_mfma_f32_32x32x8_f16 v[18:33], v[202:203], v[120:121], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[164:165], v[116:117], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[116:117], v[34:49]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[2:17], v[214:215], v[116:117], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[166:167], v[118:119], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[118:119], v[34:49]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[2:17], v[216:217], v[118:119], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[168:169], v[120:121], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[120:121], v[34:49]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[2:17], v[218:219], v[120:121], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[170:171], v[122:123], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[122:123], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[204:205], v[122:123], v[18:33]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[220:221], v[122:123], v[2:17]
	v_mfma_f32_32x32x8_f16 v[2:17], v[222:223], v[124:125], v[2:17]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v223, v253, v1, v114
	v_mul_f32_e32 v254, 0x3e0293ee, v223
	v_fma_f32 v1, v66, s35, -v254
	v_fma_f32 v66, v67, s35, -v254
	v_fma_f32 v67, v68, s35, -v254
	v_fma_f32 v68, v69, s35, -v254
	v_mfma_f32_32x32x8_f16 v[2:17], v[246:247], v[126:127], v[2:17]
	v_fma_f32 v69, v70, s35, -v254
	v_fma_f32 v70, v71, s35, -v254
	v_fma_f32 v71, v72, s35, -v254
	v_fma_f32 v72, v73, s35, -v254
	v_fma_f32 v73, v74, s35, -v254
	v_fma_f32 v74, v75, s35, -v254
	v_mfma_f32_32x32x8_f16 v[2:17], v[248:249], v[128:129], v[2:17]
	v_fma_f32 v75, v76, s35, -v254
	v_fma_f32 v76, v77, s35, -v254
	v_fma_f32 v77, v78, s35, -v254
	v_exp_f32_e32 v246, v66
	v_mfma_f32_32x32x8_f16 v[18:33], v[206:207], v[124:125], v[18:33]
	v_fma_f32 v78, v79, s35, -v254
	v_fma_f32 v79, v80, s35, -v254
	v_fma_f32 v66, v253, s35, -v254
	v_fma_f32 v80, v81, s35, -v254
	v_exp_f32_e32 v195, v68
	v_mfma_f32_32x32x8_f16 v[18:33], v[208:209], v[126:127], v[18:33]
	v_fma_f32 v81, v82, s35, -v254
	v_exp_f32_e32 v249, v67
	v_fma_f32 v82, v83, s35, -v254
	v_mfma_f32_32x32x8_f16 v[18:33], v[210:211], v[128:129], v[18:33]
	v_fma_f32 v83, v84, s35, -v254
	v_fma_f32 v84, v85, s35, -v254
	v_fma_f32 v85, v86, s35, -v254
	v_fma_f32 v86, v87, s35, -v254
	v_fma_f32 v87, v88, s35, -v254
	v_fma_f32 v88, v89, s35, -v254
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[124:125], v[50:65]
	v_fma_f32 v89, v90, s35, -v254
	v_fma_f32 v90, v91, s35, -v254
	v_fma_f32 v91, v92, s35, -v254
	v_fma_f32 v92, v93, s35, -v254
	v_fma_f32 v93, v94, s35, -v254
	v_fma_f32 v94, v95, s35, -v254
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[126:127], v[50:65]
	v_fma_f32 v95, v96, s35, -v254
	v_exp_f32_e32 v197, v70
	v_fma_f32 v96, v97, s35, -v254
	v_mfma_f32_32x32x8_f16 v[34:49], v[188:189], v[124:125], v[34:49]
	v_exp_f32_e32 v196, v71
	v_exp_f32_e32 v200, v72
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[126:127], v[34:49]
	v_exp_f32_e32 v199, v73
	v_exp_f32_e32 v248, v1
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[128:129], v[50:65]
	v_exp_f32_e32 v1, v69
	v_exp_f32_e32 v198, v95
	v_mfma_f32_32x32x8_f16 v[34:49], v[192:193], v[128:129], v[34:49]
	v_exp_f32_e32 v247, v66
	v_exp_f32_e32 v220, v93
	v_exp_f32_e32 v203, v74
	v_exp_f32_e32 v202, v75
	v_exp_f32_e32 v205, v76
	v_exp_f32_e32 v204, v77
	v_exp_f32_e32 v207, v78
	v_exp_f32_e32 v206, v79
	v_exp_f32_e32 v209, v80
	v_exp_f32_e32 v208, v81
	v_exp_f32_e32 v211, v82
	v_exp_f32_e32 v210, v83
	v_exp_f32_e32 v213, v84
	v_exp_f32_e32 v212, v85
	v_exp_f32_e32 v215, v86
	v_exp_f32_e32 v214, v87
	v_exp_f32_e32 v217, v88
	v_exp_f32_e32 v216, v89
	v_exp_f32_e32 v219, v90
	v_exp_f32_e32 v218, v91
	v_exp_f32_e32 v221, v92
	v_exp_f32_e32 v222, v94
	v_exp_f32_e32 v201, v96
	v_mov_b32_e32 v67, v226
	v_mov_b32_e32 v68, v227
	v_mov_b32_e32 v69, v228
	v_mov_b32_e32 v70, v229
	v_mov_b32_e32 v71, v230
	v_mov_b32_e32 v72, v231
	v_mov_b32_e32 v73, v239
	s_setprio 1
	; sched_barrier mask(0x00000000)
	v_perm_b32 v66, v102, v98, s41
	s_barrier
	ds_write_b32 v239, v66 offset:16384
	v_alignbit_b32 v66, v252, v98, 16
	ds_read_b128 v[80:83], v244
	ds_read_b128 v[88:91], v243
	ds_read_b128 v[84:87], v242
	ds_read_b128 v[190:193], v240
	ds_read_b128 v[186:189], v238
	ds_read_b128 v[182:185], v237
	ds_read_b128 v[178:181], v236
	ds_read_b128 v[92:95], v235
	ds_read_b128 v[122:125], v244 offset:8192
	ds_read_b128 v[118:121], v243 offset:8192
	ds_read_b128 v[114:117], v242 offset:8192
	ds_read_b128 v[174:177], v240 offset:8192
	ds_read_b128 v[170:173], v238 offset:8192
	ds_read_b128 v[166:169], v237 offset:8192
	ds_read_b128 v[162:165], v236 offset:8192
	ds_read_b128 v[126:129], v235 offset:8192
	ds_write_b32 v226, v66 offset:16384
	v_perm_b32 v66, v103, v99, s41
	ds_write_b32 v227, v66 offset:16384
	v_perm_b32 v66, v103, v99, s42
	ds_write_b32 v228, v66 offset:16384
	v_perm_b32 v66, v104, v100, s41
	ds_write_b32 v229, v66 offset:16384
	v_perm_b32 v66, v104, v100, s42
	s_and_b32 s4, s3, 0xffff
	ds_write_b32 v230, v66 offset:16384
	v_perm_b32 v66, v105, v101, s41
	s_or_b32 s13, s4, s28
	s_mov_b32 s14, s22
	s_mov_b32 s15, s23
	ds_write_b32 v231, v66 offset:16384
	v_perm_b32 v66, v105, v101, s42
	buffer_load_dwordx4 v[102:105], v225, s[12:15], 0 offen
	buffer_load_dwordx4 v[98:101], v233, s[12:15], 0 offen
	ds_write_b32 v232, v66 offset:16384
	v_mov_b32_e32 v66, v232
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v252, 16, v102
	; sched_barrier mask(0x00000000)
	s_add_u32 s29, s29, s26
	s_addc_u32 s33, s33, s27
	s_add_u32 s0, s0, s30
	s_addc_u32 s1, s1, s31
	s_add_i32 s2, s2, 64
	s_cmpk_lt_u32 s2, 0x1f00
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:
	scratch_store_dword off, v225, off offset:140 ; 4-byte Folded Spill
	scratch_store_dword off, v233, off offset:136 ; 4-byte Folded Spill
	scratch_store_dword off, v251, off offset:124 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:120 ; 4-byte Folded Spill
	scratch_store_dword off, v241, off offset:112 ; 4-byte Folded Spill
	scratch_load_dword v250, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v225, off, off offset:108 ; 4-byte Folded Reload
	scratch_load_dword v253, off, off offset:104 ; 4-byte Folded Reload
	scratch_load_dword v252, off, off offset:100 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v241, off, off offset:96 ; 4-byte Folded Reload
	scratch_load_dword v239, off, off offset:92 ; 4-byte Folded Reload
	scratch_load_dword v234, off, off offset:88 ; 4-byte Folded Reload
	scratch_load_dword v233, off, off offset:84 ; 4-byte Folded Reload
	s_nop 0
	scratch_store_dword off, v73, off offset:116 ; 4-byte Folded Spill
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
        ;; Epilogue

        ;; round 0
        ;; Cluster 0
        ;; DOT1[2] + VEC2[1] (add, mul, cvtrtz)
        s_or_b64 exec, exec, s[0:1]
	scratch_load_dword v66, off, off offset:152 ; 4-byte Folded Reload
	scratch_load_dword v67, off, off offset:144 ; 4-byte Folded Reload
	scratch_load_dword v68, off, off offset:148 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_cmp_eq_u32_e64 s[0:1], 0, v66
	scratch_load_dword v66, off, off offset:156 ; 4-byte Folded Reload
	s_mul_i32 s4, s18, 0xc0000
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s2, s8, s4
	s_addc_u32 s6, s9, s5
	s_lshl_b32 s4, s17, 14
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s2, s2, s4
	s_addc_u32 s6, s6, s5
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[4:5], s[34:35], 2
	s_add_u32 s4, s2, s4
	s_addc_u32 s19, s6, s5
	s_add_i32 s2, s34, 0xffffc100
	s_add_u32 s12, s12, s26
	s_addc_u32 s3, s3, s27
	s_and_b32 s13, s3, 0xffff
	s_cmp_lt_i32 s2, 1
	; iglp_opt mask(0x0000000A)
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v66, 0xa0, v66
	v_or3_b32 v66, v66, v67, v68
	scratch_store_dword off, v66, off offset:144 ; 4-byte Folded Spill
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[66:81], v[80:81], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[66:81], v[82:83], v[160:161], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[88:89], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[90:91], v[144:145], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[84:85], v[154:155], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[86:87], v[156:157], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[190:191], v[138:139], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[192:193], v[140:141], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[186:187], v[150:151], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[188:189], v[152:153], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[182:183], v[134:135], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[184:185], v[136:137], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[178:179], v[146:147], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[180:181], v[148:149], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[92:93], v[130:131], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[94:95], v[132:133], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[122:123], v[158:159], 0
	v_cvt_pkrtz_f16_f32 v122, v208, v211
	v_cvt_pkrtz_f16_f32 v123, v210, v213
	v_mfma_f32_32x32x8_f16 v[82:97], v[124:125], v[160:161], v[82:97]
	v_cvt_pkrtz_f16_f32 v124, v212, v215
	v_cvt_pkrtz_f16_f32 v125, v214, v217
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[82:97], v[118:119], v[142:143], v[82:97]
	v_cvt_pkrtz_f16_f32 v118, v199, v203
	v_cvt_pkrtz_f16_f32 v119, v202, v205
	v_mfma_f32_32x32x8_f16 v[82:97], v[120:121], v[144:145], v[82:97]
	v_cvt_pkrtz_f16_f32 v120, v204, v207
	v_cvt_pkrtz_f16_f32 v121, v206, v209
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x8_f16 v[82:97], v[114:115], v[154:155], v[82:97]
	v_add_f32_e32 v114, v248, v246
	v_add_f32_e32 v114, v114, v249
	v_add_f32_e32 v114, v114, v195
	v_add_f32_e32 v114, v114, v1
	v_add_f32_e32 v114, v114, v197
	v_add_f32_e32 v114, v114, v196
	v_mfma_f32_32x32x8_f16 v[82:97], v[116:117], v[156:157], v[82:97]
	v_add_f32_e32 v114, v114, v200
	v_cvt_pkrtz_f16_f32 v117, v196, v200
	v_add_f32_e32 v114, v114, v199
	v_cvt_pkrtz_f16_f32 v116, v1, v197
	v_add_f32_e32 v114, v114, v203
	v_add_f32_e32 v114, v114, v202
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_f16 v[82:97], v[174:175], v[138:139], v[82:97]
	v_add_f32_e32 v114, v114, v205
	v_add_f32_e32 v114, v114, v204
	v_add_f32_e32 v114, v114, v207
	v_add_f32_e32 v114, v114, v206
	v_add_f32_e32 v114, v114, v209
	v_add_f32_e32 v114, v114, v208
	v_mfma_f32_32x32x8_f16 v[82:97], v[176:177], v[140:141], v[82:97]
	v_add_f32_e32 v114, v114, v211
	v_add_f32_e32 v114, v114, v210
	v_add_f32_e32 v114, v114, v213
	v_add_f32_e32 v114, v114, v212
	v_add_f32_e32 v114, v114, v215
	v_add_f32_e32 v114, v114, v214
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x8_f16 v[82:97], v[170:171], v[150:151], v[82:97]
	v_add_f32_e32 v114, v114, v217
	v_add_f32_e32 v114, v114, v216
	v_add_f32_e32 v114, v114, v219
	v_add_f32_e32 v114, v114, v218
	v_add_f32_e32 v114, v114, v221
	v_add_f32_e32 v114, v114, v220
	v_mfma_f32_32x32x8_f16 v[82:97], v[172:173], v[152:153], v[82:97]
	v_add_f32_e32 v114, v114, v222
	v_add_f32_e32 v114, v114, v198
	v_add_f32_e32 v114, v114, v201
	ds_bpermute_b32 v115, v194, v114
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v114, v114, v115
	v_fmac_f32_e32 v114, v255, v247
	v_mfma_f32_32x32x8_f16 v[82:97], v[166:167], v[134:135], v[82:97]
	scratch_store_dword off, v114, off offset:148 ; 4-byte Folded Spill
	v_cvt_pkrtz_f16_f32 v115, v249, v195
	v_cvt_pkrtz_f16_f32 v114, v248, v246
	v_mfma_f32_32x32x8_f16 v[82:97], v[168:169], v[136:137], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[162:163], v[146:147], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[164:165], v[148:149], v[82:97]
	v_mfma_f32_32x32x8_f16 v[82:97], v[126:127], v[130:131], v[82:97]
	v_cvt_pkrtz_f16_f32 v126, v216, v219
	v_cvt_pkrtz_f16_f32 v127, v218, v221
	v_mfma_f32_32x32x8_f16 v[82:97], v[128:129], v[132:133], v[82:97]
	v_cvt_pkrtz_f16_f32 v128, v220, v222
	v_cvt_pkrtz_f16_f32 v129, v198, v201
        ;; where are the v_mul???
	; sched_barrier mask(0x00000000)


        ;; Cluster 1
	;; LRV[1] + LWK[3]
        s_barrier
	ds_read_b64 v[172:173], v0 offset:16384
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	v_mov_b32_e32 v246, v252
	ds_read_b64 v[174:175], v252 offset:16384
	v_mov_b32_e32 v255, v253
	ds_read_b64 v[176:177], v253 offset:16384
	ds_read_b64 v[162:163], v224 offset:16384
	ds_read_b64 v[164:165], v233 offset:16384
	ds_read_b64 v[166:167], v234 offset:16384
	ds_read_b64 v[168:169], v239 offset:16384
	ds_read_b64 v[170:171], v241 offset:16384
	v_mov_b32_e32 v222, v225
	ds_read_b64 v[178:179], v225 offset:16384
	ds_read_b64 v[180:181], v250 offset:16384
	v_mov_b32_e32 v195, v224
	scratch_store_dword off, v195, off offset:160 ; 4-byte Folded Spill
	s_waitcnt vmcnt(1)
	ds_read_b64 v[182:183], v0 offset:16384
	scratch_load_dword v0, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[184:185], v0 offset:16384
	scratch_load_dword v0, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[186:187], v0 offset:16384
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[188:189], v0 offset:16384
	scratch_load_dword v0, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[190:191], v0 offset:16384
	scratch_load_dword v0, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[192:193], v0 offset:16384
	scratch_load_dword v0, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[196:197], v0 offset:16384
	scratch_load_dword v0, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[198:199], v0 offset:16384
	scratch_load_dword v0, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[200:201], v0 offset:16384
	scratch_load_dword v0, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[202:203], v0 offset:16384
	scratch_load_dword v0, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[204:205], v0 offset:16384
	scratch_load_dword v0, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[206:207], v0 offset:16384
	scratch_load_dword v0, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[208:209], v0 offset:16384
	scratch_load_dword v0, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[210:211], v0 offset:16384
	scratch_load_dword v0, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[212:213], v0 offset:16384
	scratch_load_dword v0, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[214:215], v0 offset:16384
	scratch_load_dword v0, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[216:217], v0 offset:16384
	scratch_load_dword v0, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[218:219], v0 offset:16384
	scratch_load_dword v0, off, off offset:72 ; 4-byte Folded Reload
	ds_write_b128 v245, v[110:113]
	ds_write_b128 v245, v[106:109] offset:8192
	s_waitcnt vmcnt(0)
	ds_read_b64 v[220:221], v0 offset:16384
	scratch_load_dword v0, off, off offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[248:249], v0 offset:16384
	scratch_load_dword v0, off, off offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[252:253], v0 offset:16384
	scratch_load_dword v0, off, off offset:124 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[224:225], v0 offset:16384
	; sched_barrier mask(0x00000000)

        ;; Cluster 2
	;; DOT2[1] + VEC1[2]
	v_mul_f32_e32 v50, v50, v247
	v_max_f32_e32 v106, v66, v66
	v_max_f32_e32 v1, v67, v67
	v_mul_f32_e32 v51, v51, v247
	v_max_f32_e32 v1, v106, v1
	v_mul_f32_e32 v17, v17, v247
	v_mul_f32_e32 v52, v52, v247
	v_mul_f32_e32 v53, v53, v247
	v_mul_f32_e32 v54, v54, v247
	v_mul_f32_e32 v55, v55, v247
	v_mul_f32_e32 v56, v56, v247
	v_mul_f32_e32 v57, v57, v247
	v_mul_f32_e32 v58, v58, v247
	v_mul_f32_e32 v59, v59, v247
	v_mul_f32_e32 v60, v60, v247
	v_mul_f32_e32 v61, v61, v247
	v_mul_f32_e32 v62, v62, v247
	v_mul_f32_e32 v63, v63, v247
	v_mul_f32_e32 v64, v64, v247
	v_mul_f32_e32 v65, v65, v247
	s_waitcnt lgkmcnt(14)
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[50:65], v[162:163], v[114:115], v[50:65]
	v_max3_f32 v1, v1, v68, v69
	v_mul_f32_e32 v16, v16, v247
	v_max3_f32 v1, v1, v70, v71
	v_mul_f32_e32 v2, v2, v247
	v_max3_f32 v1, v1, v72, v73
	v_mul_f32_e32 v33, v33, v247
	v_mfma_f32_32x32x8_f16 v[50:65], v[164:165], v[116:117], v[50:65]
	v_mul_f32_e32 v32, v32, v247
	v_max3_f32 v1, v1, v74, v75
	v_mul_f32_e32 v31, v31, v247
	v_max3_f32 v1, v1, v76, v77
	v_mul_f32_e32 v30, v30, v247
	v_max3_f32 v1, v1, v78, v79
	v_mfma_f32_32x32x8_f16 v[50:65], v[166:167], v[118:119], v[50:65]
	v_max3_f32 v1, v1, v80, v81
	v_mul_f32_e32 v29, v29, v247
	v_max3_f32 v1, v1, v82, v83
	v_mul_f32_e32 v28, v28, v247
	v_max3_f32 v1, v1, v84, v85
	v_mul_f32_e32 v27, v27, v247
	v_mfma_f32_32x32x8_f16 v[50:65], v[168:169], v[120:121], v[50:65]
	v_mul_f32_e32 v26, v26, v247
	v_max3_f32 v1, v1, v86, v87
	v_mul_f32_e32 v25, v25, v247
	v_max3_f32 v1, v1, v88, v89
	v_mul_f32_e32 v24, v24, v247
	v_max3_f32 v1, v1, v90, v91
	v_mfma_f32_32x32x8_f16 v[50:65], v[170:171], v[122:123], v[50:65]
	v_max3_f32 v1, v1, v92, v93
	v_mul_f32_e32 v23, v23, v247
	v_max3_f32 v1, v1, v94, v95
	v_mul_f32_e32 v22, v22, v247
	v_max3_f32 v1, v1, v96, v97
	ds_bpermute_b32 v106, v194, v1
	v_mfma_f32_32x32x8_f16 v[50:65], v[172:173], v[124:125], v[50:65]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v1, v223, v1, v106
	v_mul_f32_e32 v21, v21, v247
	v_mul_f32_e32 v20, v20, v247
	v_mul_f32_e32 v19, v19, v247
	v_mul_f32_e32 v18, v18, v247
	v_mul_f32_e32 v49, v49, v247
	v_mfma_f32_32x32x8_f16 v[50:65], v[174:175], v[126:127], v[50:65]
	v_mul_f32_e32 v48, v48, v247
	v_mul_f32_e32 v15, v15, v247
	v_mul_f32_e32 v47, v47, v247
	v_mul_f32_e32 v46, v46, v247
	v_mul_f32_e32 v45, v45, v247
	v_mul_f32_e32 v44, v44, v247
	v_mfma_f32_32x32x8_f16 v[18:33], v[196:197], v[114:115], v[18:33]
	v_mul_f32_e32 v42, v42, v247
	v_mul_f32_e32 v43, v43, v247
	v_mul_f32_e32 v41, v41, v247
	v_mul_f32_e32 v40, v40, v247
	v_mul_f32_e32 v39, v39, v247
	v_mul_f32_e32 v38, v38, v247
	v_mfma_f32_32x32x8_f16 v[18:33], v[198:199], v[116:117], v[18:33]
	v_mul_f32_e32 v36, v36, v247
	v_mul_f32_e32 v37, v37, v247
	v_mul_f32_e32 v35, v35, v247
	v_mul_f32_e32 v34, v34, v247
	v_mul_f32_e32 v3, v3, v247
	v_mul_f32_e32 v14, v14, v247
	v_mfma_f32_32x32x8_f16 v[18:33], v[200:201], v[118:119], v[18:33]
	v_mul_f32_e32 v5, v5, v247
	v_mul_f32_e32 v4, v4, v247
	v_mul_f32_e32 v6, v6, v247
	v_mul_f32_e32 v7, v7, v247
	v_mul_f32_e32 v8, v8, v247
	v_mul_f32_e32 v9, v9, v247
	v_mfma_f32_32x32x8_f16 v[18:33], v[202:203], v[120:121], v[18:33]
	v_mul_f32_e32 v11, v11, v247
	v_mul_f32_e32 v10, v10, v247
	v_mul_f32_e32 v12, v12, v247
	v_mul_f32_e32 v13, v13, v247
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[114:115], v[34:49]
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[2:17], v[212:213], v[114:115], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[116:117], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[214:215], v[116:117], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[118:119], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[216:217], v[118:119], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[120:121], v[34:49]
	v_mfma_f32_32x32x8_f16 v[2:17], v[218:219], v[120:121], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[186:187], v[122:123], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[204:205], v[122:123], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[220:221], v[122:123], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[188:189], v[124:125], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[206:207], v[124:125], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[248:249], v[124:125], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[190:191], v[126:127], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[208:209], v[126:127], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[252:253], v[126:127], v[2:17]
	v_mfma_f32_32x32x8_f16 v[50:65], v[176:177], v[128:129], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[192:193], v[128:129], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[210:211], v[128:129], v[18:33]
	v_mfma_f32_32x32x8_f16 v[2:17], v[224:225], v[128:129], v[2:17]
        ;; where are v_fma??
	; sched_barrier mask(0x00000000)



        ;; Cluster 3
	;; LRK[3] + LWV[2] + GRV[3]
        s_barrier
	scratch_load_dword v106, off, off offset:136 ; 4-byte Folded Reload
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	scratch_load_dword v0, off, off offset:116 ; 4-byte Folded Reload
	s_mov_b32 s2, 0x5040100
	s_mov_b32 s3, 0x7060302
	ds_read_b128 v[112:115], v244
	ds_read_b128 v[170:173], v244 offset:8192
	ds_read_b128 v[116:119], v243
	ds_read_b128 v[174:177], v243 offset:8192
	ds_read_b128 v[120:123], v242
	ds_read_b128 v[178:181], v242 offset:8192
	ds_read_b128 v[124:127], v240
	ds_read_b128 v[182:185], v240 offset:8192
	ds_read_b128 v[186:189], v238
	ds_read_b128 v[190:193], v238 offset:8192
	ds_read_b128 v[196:199], v237
	ds_read_b128 v[200:203], v237 offset:8192
	ds_read_b128 v[204:207], v236
	ds_read_b128 v[208:211], v236 offset:8192
	ds_read_b128 v[212:215], v235
	ds_read_b128 v[216:219], v235 offset:8192
	v_mov_b32_e32 v250, v226
	v_mov_b32_e32 v251, v227
	s_waitcnt vmcnt(1)
	buffer_load_dwordx4 v[162:165], v106, s[12:15], 0 offen
	s_nop 0
	scratch_load_dword v106, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v[166:169], v106, s[12:15], 0 offen
	v_perm_b32 v106, v102, v98, s2
	v_perm_b32 v98, v102, v98, s3
	ds_write_b32 v0, v106 offset:16384
	ds_write_b32 v226, v98 offset:16384
	v_perm_b32 v98, v103, v99, s2
	ds_write_b32 v227, v98 offset:16384
	v_perm_b32 v98, v103, v99, s3
	ds_write_b32 v228, v98 offset:16384
	v_perm_b32 v98, v104, v100, s2
	ds_write_b32 v229, v98 offset:16384
	v_perm_b32 v98, v104, v100, s3
	ds_write_b32 v230, v98 offset:16384
	v_perm_b32 v98, v105, v101, s2
	ds_write_b32 v231, v98 offset:16384
	v_perm_b32 v98, v105, v101, s3
	ds_write_b32 v232, v98 offset:16384
	; sched_barrier mask(0x00000000)
	; iglp_opt mask(0x0000000A)




        ;; Epilogue round 1
        ;; cluster 1-0: DOT1[3] + VEC2[2]
        s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[98:113], v[112:113], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[98:113], v[114:115], v[160:161], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[116:117], v[142:143], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[118:119], v[144:145], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[120:121], v[154:155], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[122:123], v[156:157], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[124:125], v[138:139], v[98:113]
	v_mfma_f32_32x32x8_f16 v[98:113], v[126:127], v[140:141], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[170:171], v[158:159], 0
	v_mfma_f32_32x32x8_f16 v[114:129], v[172:173], v[160:161], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[186:187], v[150:151], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[174:175], v[142:143], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[188:189], v[152:153], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[176:177], v[144:145], v[114:129]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x8_f16 v[98:113], v[196:197], v[134:135], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[178:179], v[154:155], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[198:199], v[136:137], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[180:181], v[156:157], v[114:129]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x8_f16 v[98:113], v[204:205], v[146:147], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[182:183], v[138:139], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[206:207], v[148:149], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[184:185], v[140:141], v[114:129]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x8_f16 v[98:113], v[212:213], v[130:131], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[190:191], v[150:151], v[114:129]
	v_mfma_f32_32x32x8_f16 v[98:113], v[214:215], v[132:133], v[98:113]
	v_mfma_f32_32x32x8_f16 v[114:129], v[192:193], v[152:153], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[200:201], v[134:135], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[202:203], v[136:137], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[208:209], v[146:147], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[210:211], v[148:149], v[114:129]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x8_f16 v[114:129], v[216:217], v[130:131], v[114:129]
	v_mfma_f32_32x32x8_f16 v[114:129], v[218:219], v[132:133], v[114:129]
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)

        ;; Cluster 1-1: LRV[2]
	s_barrier
	scratch_load_dword v0, off, off offset:112 ; 4-byte Folded Reload
	scratch_load_dword v219, off, off offset:76 ; 4-byte Folded Reload
	scratch_load_dword v220, off, off offset:80 ; 4-byte Folded Reload
	scratch_load_dword v221, off, off       ; 4-byte Folded Reload
	scratch_load_dword v223, off, off offset:4 ; 4-byte Folded Reload
	scratch_load_dword v224, off, off offset:8 ; 4-byte Folded Reload
	scratch_load_dword v225, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v235, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v236, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dword v237, off, off offset:24 ; 4-byte Folded Reload
	scratch_load_dword v238, off, off offset:28 ; 4-byte Folded Reload
	scratch_load_dword v240, off, off offset:32 ; 4-byte Folded Reload
	scratch_load_dword v242, off, off offset:36 ; 4-byte Folded Reload
	scratch_load_dword v243, off, off offset:40 ; 4-byte Folded Reload
	scratch_load_dword v244, off, off offset:44 ; 4-byte Folded Reload
	scratch_load_dword v245, off, off offset:48 ; 4-byte Folded Reload
	scratch_load_dword v247, off, off offset:52 ; 4-byte Folded Reload
	scratch_load_dword v248, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v249, off, off offset:60 ; 4-byte Folded Reload
	scratch_load_dword v252, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v253, off, off offset:68 ; 4-byte Folded Reload
	scratch_load_dword v226, off, off offset:72 ; 4-byte Folded Reload
	scratch_load_dword v227, off, off offset:124 ; 4-byte Folded Reload
	ds_read_b64 v[196:197], v195 offset:16384
	ds_read_b64 v[198:199], v233 offset:16384
	ds_read_b64 v[200:201], v234 offset:16384
	ds_read_b64 v[202:203], v239 offset:16384
	ds_read_b64 v[192:193], v241 offset:16384
	ds_read_b64 v[188:189], v246 offset:16384
	ds_read_b64 v[186:187], v255 offset:16384
	ds_read_b64 v[184:185], v222 offset:16384
	s_waitcnt vmcnt(22)
	ds_read_b64 v[134:135], v0 offset:16384
	scratch_load_dword v0, off, off offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(22)
	ds_read_b64 v[190:191], v219 offset:16384
	s_waitcnt vmcnt(21)
	ds_read_b64 v[182:183], v220 offset:16384
	s_waitcnt vmcnt(20)
	ds_read_b64 v[180:181], v221 offset:16384
	s_waitcnt vmcnt(19)
	ds_read_b64 v[178:179], v223 offset:16384
	s_waitcnt vmcnt(18)
	ds_read_b64 v[176:177], v224 offset:16384
	s_waitcnt vmcnt(17)
	ds_read_b64 v[174:175], v225 offset:16384
	s_waitcnt vmcnt(16)
	ds_read_b64 v[172:173], v235 offset:16384
	s_waitcnt vmcnt(15)
	ds_read_b64 v[170:171], v236 offset:16384
	s_waitcnt vmcnt(14)
	ds_read_b64 v[160:161], v237 offset:16384
	s_waitcnt vmcnt(13)
	ds_read_b64 v[158:159], v238 offset:16384
	s_waitcnt vmcnt(12)
	ds_read_b64 v[156:157], v240 offset:16384
	s_waitcnt vmcnt(11)
	ds_read_b64 v[154:155], v242 offset:16384
	s_waitcnt vmcnt(10)
	ds_read_b64 v[152:153], v243 offset:16384
	s_waitcnt vmcnt(9)
	ds_read_b64 v[150:151], v244 offset:16384
	s_waitcnt vmcnt(8)
	ds_read_b64 v[148:149], v245 offset:16384
	s_waitcnt vmcnt(7)
	ds_read_b64 v[146:147], v247 offset:16384
	s_waitcnt vmcnt(6)
	ds_read_b64 v[144:145], v248 offset:16384
	s_waitcnt vmcnt(5)
	ds_read_b64 v[142:143], v249 offset:16384
	s_waitcnt vmcnt(4)
	ds_read_b64 v[140:141], v252 offset:16384
	s_waitcnt vmcnt(3)
	ds_read_b64 v[138:139], v253 offset:16384
	s_waitcnt vmcnt(2)
	ds_read_b64 v[136:137], v226 offset:16384
	s_waitcnt vmcnt(1)
	ds_read_b64 v[130:131], v227 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[132:133], v0 offset:16384
	; sched_barrier mask(0x00000000)
	v_max_f32_e32 v195, v99, v99
	v_max_f32_e32 v204, v98, v98
	v_fmac_f32_e32 v254, 0xbe0293ee, v1
	v_max_f32_e32 v195, v204, v195
	s_mov_b32 s5, 0x3e0293ee
	v_mul_f32_e32 v205, 0x3e0293ee, v1
	v_fma_f32 v66, v66, s5, -v205
	v_fma_f32 v67, v67, s5, -v205
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_fma_f32 v82, v82, s5, -v205
	v_fma_f32 v84, v84, s5, -v205
	v_exp_f32_e32 v206, v82
	v_exp_f32_e32 v208, v84
	v_add_f32_e32 v82, v66, v67
	v_cvt_pkrtz_f16_f32 v84, v66, v67
	scratch_load_dword v66, off, off offset:148 ; 4-byte Folded Reload
	v_fma_f32 v68, v68, s5, -v205
	v_fma_f32 v69, v69, s5, -v205
	v_exp_f32_e32 v68, v68
	v_fma_f32 v70, v70, s5, -v205
	v_exp_f32_e32 v69, v69
	v_fma_f32 v71, v71, s5, -v205
	v_exp_f32_e32 v70, v70
	v_fma_f32 v72, v72, s5, -v205
	v_exp_f32_e32 v71, v71
	v_fma_f32 v73, v73, s5, -v205
	v_exp_f32_e32 v72, v72
	v_add_f32_e32 v82, v68, v82
	v_fma_f32 v74, v74, s5, -v205
	v_exp_f32_e32 v73, v73
	v_add_f32_e32 v82, v69, v82
	v_fma_f32 v75, v75, s5, -v205
	v_exp_f32_e32 v74, v74
	v_add_f32_e32 v82, v70, v82
	v_fma_f32 v76, v76, s5, -v205
	v_exp_f32_e32 v75, v75
	v_add_f32_e32 v82, v71, v82
	v_fma_f32 v77, v77, s5, -v205
	v_exp_f32_e32 v76, v76
	v_add_f32_e32 v82, v72, v82
	v_fma_f32 v78, v78, s5, -v205
	v_exp_f32_e32 v77, v77
	v_add_f32_e32 v82, v73, v82
	v_fma_f32 v79, v79, s5, -v205
	v_exp_f32_e32 v78, v78
	v_add_f32_e32 v82, v74, v82
	v_fma_f32 v80, v80, s5, -v205
	v_exp_f32_e32 v79, v79
	v_add_f32_e32 v82, v75, v82
	v_fma_f32 v81, v81, s5, -v205
	v_exp_f32_e32 v80, v80
	v_add_f32_e32 v82, v76, v82
	v_exp_f32_e32 v81, v81
	v_add_f32_e32 v82, v77, v82
	v_fma_f32 v83, v83, s5, -v205
	v_add_f32_e32 v82, v78, v82
	v_exp_f32_e32 v207, v83
	v_add_f32_e32 v82, v79, v82
	v_max3_f32 v195, v195, v100, v101
	v_fma_f32 v85, v85, s5, -v205
	v_add_f32_e32 v82, v80, v82
	v_max3_f32 v195, v195, v102, v103
	v_fma_f32 v86, v86, s5, -v205
	v_exp_f32_e32 v209, v85
	v_add_f32_e32 v82, v81, v82
	v_max3_f32 v195, v195, v104, v105
	v_fma_f32 v87, v87, s5, -v205
	v_exp_f32_e32 v86, v86
	v_add_f32_e32 v82, v206, v82
	v_max3_f32 v195, v195, v106, v107
	v_fma_f32 v88, v88, s5, -v205
	v_exp_f32_e32 v87, v87
	v_add_f32_e32 v82, v207, v82
	v_max3_f32 v195, v195, v108, v109
	v_fma_f32 v89, v89, s5, -v205
	v_exp_f32_e32 v88, v88
	v_add_f32_e32 v82, v208, v82
	v_max3_f32 v195, v195, v110, v111
	v_fma_f32 v90, v90, s5, -v205
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v82, v209, v82
	v_max3_f32 v195, v195, v112, v113
	v_fma_f32 v91, v91, s5, -v205
	v_exp_f32_e32 v210, v90
	v_add_f32_e32 v82, v86, v82
	v_max3_f32 v195, v195, v114, v115
	v_fma_f32 v92, v92, s5, -v205
	v_exp_f32_e32 v211, v91
	v_add_f32_e32 v82, v87, v82
	v_max3_f32 v195, v195, v116, v117
	v_fma_f32 v93, v93, s5, -v205
	v_exp_f32_e32 v212, v92
	v_add_f32_e32 v82, v88, v82
	v_max3_f32 v195, v195, v118, v119
	v_fma_f32 v94, v94, s5, -v205
	v_exp_f32_e32 v213, v93
	v_add_f32_e32 v82, v89, v82
	v_max3_f32 v195, v195, v120, v121
	v_fma_f32 v95, v95, s5, -v205
	v_exp_f32_e32 v214, v94
	v_add_f32_e32 v82, v210, v82
	v_max3_f32 v195, v195, v122, v123
	v_fma_f32 v96, v96, s5, -v205
	v_exp_f32_e32 v215, v95
	v_add_f32_e32 v82, v211, v82
	v_max3_f32 v195, v195, v124, v125
	v_fma_f32 v97, v97, s5, -v205
	v_exp_f32_e32 v216, v96
	v_add_f32_e32 v82, v212, v82
	v_max3_f32 v195, v195, v126, v127
	v_exp_f32_e32 v217, v97
	v_add_f32_e32 v82, v213, v82
	v_max3_f32 v195, v195, v128, v129
	v_add_f32_e32 v82, v214, v82
	ds_bpermute_b32 v204, v194, v195
	v_add_f32_e32 v82, v215, v82
	v_add_f32_e32 v82, v216, v82
	v_add_f32_e32 v90, v217, v82
	ds_bpermute_b32 v91, v194, v90
	s_waitcnt lgkmcnt(1)
	v_max3_f32 v195, v1, v195, v204
	v_exp_f32_e32 v1, v254
	v_mul_f32_e32 v204, 0xbe0293ee, v195
	v_cvt_pkrtz_f16_f32 v85, v68, v69
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v218, v90, v91
	v_cvt_pkrtz_f16_f32 v82, v70, v71
	v_cvt_pkrtz_f16_f32 v83, v72, v73
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v218, v66, v1
	v_cvt_pkrtz_f16_f32 v96, v74, v75
	v_cvt_pkrtz_f16_f32 v97, v76, v77
	v_cvt_pkrtz_f16_f32 v94, v78, v79
	v_cvt_pkrtz_f16_f32 v95, v80, v81
	v_fmamk_f32 v66, v112, 0x3e0293ee, v204
	v_fmamk_f32 v67, v111, 0x3e0293ee, v204
	v_fmamk_f32 v68, v110, 0x3e0293ee, v204
	v_fmamk_f32 v69, v109, 0x3e0293ee, v204
	v_fmamk_f32 v70, v108, 0x3e0293ee, v204
	v_fmamk_f32 v71, v107, 0x3e0293ee, v204
	v_fmamk_f32 v72, v106, 0x3e0293ee, v204
	v_fmamk_f32 v73, v105, 0x3e0293ee, v204
	v_fmamk_f32 v74, v104, 0x3e0293ee, v204
	v_fmamk_f32 v75, v103, 0x3e0293ee, v204
	v_fmamk_f32 v76, v102, 0x3e0293ee, v204
	v_fmamk_f32 v77, v101, 0x3e0293ee, v204
	v_fmamk_f32 v78, v100, 0x3e0293ee, v204
	v_fmamk_f32 v79, v99, 0x3e0293ee, v204
	v_fmamk_f32 v80, v98, 0x3e0293ee, v204
	v_fmamk_f32 v81, v128, 0x3e0293ee, v204
	v_mul_f32_e32 v50, v50, v1
	v_mul_f32_e32 v51, v51, v1
	v_mul_f32_e32 v52, v52, v1
	v_mul_f32_e32 v53, v53, v1
	v_mul_f32_e32 v54, v54, v1
	v_mul_f32_e32 v55, v55, v1
	v_mul_f32_e32 v56, v56, v1
	v_mul_f32_e32 v57, v57, v1
	v_mul_f32_e32 v58, v58, v1
	v_mul_f32_e32 v59, v59, v1
	v_mul_f32_e32 v60, v60, v1
	v_mul_f32_e32 v61, v61, v1
	v_mul_f32_e32 v62, v62, v1
	v_mul_f32_e32 v63, v63, v1
	v_mul_f32_e32 v64, v64, v1
	v_mul_f32_e32 v65, v65, v1
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[196:197], v[84:85], v[50:65]
	v_cvt_pkrtz_f16_f32 v91, v88, v89
	v_cvt_pkrtz_f16_f32 v90, v86, v87
	v_fmamk_f32 v110, v115, 0x3e0293ee, v204
	v_fmamk_f32 v104, v121, 0x3e0293ee, v204
	v_exp_f32_e32 v115, v66
	v_mfma_f32_32x32x8_f16 v[50:65], v[198:199], v[82:83], v[50:65]
	v_exp_f32_e32 v121, v72
	v_fmamk_f32 v98, v127, 0x3e0293ee, v204
	v_fmamk_f32 v100, v125, 0x3e0293ee, v204
	v_mfma_f32_32x32x8_f16 v[50:65], v[200:201], v[96:97], v[50:65]
	v_exp_f32_e32 v125, v76
	v_exp_f32_e32 v127, v79
	v_mfma_f32_32x32x8_f16 v[50:65], v[202:203], v[94:95], v[50:65]
	v_fmamk_f32 v102, v123, 0x3e0293ee, v204
	v_fmamk_f32 v112, v113, 0x3e0293ee, v204
	v_exp_f32_e32 v123, v74
	v_fmamk_f32 v99, v126, 0x3e0293ee, v204
	v_fmamk_f32 v101, v124, 0x3e0293ee, v204
	v_fmamk_f32 v103, v122, 0x3e0293ee, v204
	v_fmamk_f32 v105, v120, 0x3e0293ee, v204
	v_fmamk_f32 v106, v119, 0x3e0293ee, v204
	v_fmamk_f32 v107, v118, 0x3e0293ee, v204
	v_fmamk_f32 v108, v117, 0x3e0293ee, v204
	v_fmamk_f32 v109, v116, 0x3e0293ee, v204
	v_fmamk_f32 v111, v114, 0x3e0293ee, v204
	v_fmac_f32_e32 v204, 0x3e0293ee, v129
	v_exp_f32_e32 v113, v71
	v_exp_f32_e32 v116, v67
	v_exp_f32_e32 v117, v68
	v_mul_f32_e32 v34, v34, v1
	v_mul_f32_e32 v35, v35, v1
	v_mul_f32_e32 v36, v36, v1
	v_mul_f32_e32 v37, v37, v1
	v_mul_f32_e32 v38, v38, v1
	v_mul_f32_e32 v39, v39, v1
	v_mul_f32_e32 v40, v40, v1
	v_mul_f32_e32 v41, v41, v1
	v_mul_f32_e32 v42, v42, v1
	v_mul_f32_e32 v43, v43, v1
	v_mul_f32_e32 v44, v44, v1
	v_mul_f32_e32 v45, v45, v1
	v_mul_f32_e32 v46, v46, v1
	v_mul_f32_e32 v47, v47, v1
	v_mul_f32_e32 v48, v48, v1
	v_mul_f32_e32 v49, v49, v1
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[34:49], v[184:185], v[84:85], v[34:49]
	v_exp_f32_e32 v129, v81
	v_exp_f32_e32 v118, v69
	v_mfma_f32_32x32x8_f16 v[34:49], v[182:183], v[82:83], v[34:49]
	v_exp_f32_e32 v128, v80
	v_exp_f32_e32 v119, v70
	v_mfma_f32_32x32x8_f16 v[34:49], v[180:181], v[96:97], v[34:49]
	v_exp_f32_e32 v122, v73
	v_exp_f32_e32 v124, v75
	v_mfma_f32_32x32x8_f16 v[34:49], v[178:179], v[94:95], v[34:49]
	v_exp_f32_e32 v126, v78
	v_exp_f32_e32 v120, v77
	v_mul_f32_e32 v18, v18, v1
	v_mul_f32_e32 v19, v19, v1
	v_mul_f32_e32 v20, v20, v1
	v_mul_f32_e32 v21, v21, v1
	v_mul_f32_e32 v22, v22, v1
	v_mul_f32_e32 v23, v23, v1
	v_mul_f32_e32 v24, v24, v1
	v_mul_f32_e32 v25, v25, v1
	v_mul_f32_e32 v26, v26, v1
	v_mul_f32_e32 v27, v27, v1
	v_mul_f32_e32 v28, v28, v1
	v_mul_f32_e32 v29, v29, v1
	v_mul_f32_e32 v30, v30, v1
	v_mul_f32_e32 v31, v31, v1
	v_mul_f32_e32 v32, v32, v1
	v_mul_f32_e32 v33, v33, v1
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[18:33], v[160:161], v[84:85], v[18:33]
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v108, v108
	v_mfma_f32_32x32x8_f16 v[18:33], v[158:159], v[82:83], v[18:33]
	v_exp_f32_e32 v106, v106
	v_exp_f32_e32 v107, v107
	v_mfma_f32_32x32x8_f16 v[18:33], v[156:157], v[96:97], v[18:33]
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v105, v105
	v_mfma_f32_32x32x8_f16 v[18:33], v[154:155], v[94:95], v[18:33]
	v_exp_f32_e32 v99, v99
	v_exp_f32_e32 v112, v112
	v_mul_f32_e32 v81, v17, v1
	v_mul_f32_e32 v79, v15, v1
	v_mul_f32_e32 v73, v9, v1
	v_mul_f32_e32 v75, v11, v1
	v_mul_f32_e32 v74, v10, v1
	v_mul_f32_e32 v66, v2, v1
	v_mul_f32_e32 v67, v3, v1
	v_mul_f32_e32 v72, v8, v1
	v_mul_f32_e32 v68, v4, v1
	v_mul_f32_e32 v78, v14, v1
	v_mul_f32_e32 v69, v5, v1
	v_mul_f32_e32 v76, v12, v1
	v_mul_f32_e32 v70, v6, v1
	v_mul_f32_e32 v71, v7, v1
	v_mul_f32_e32 v77, v13, v1
	v_mul_f32_e32 v80, v16, v1
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[84:85], v[66:81]
	v_exp_f32_e32 v104, v104
	v_exp_f32_e32 v102, v102
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[82:83], v[66:81]
	v_exp_f32_e32 v109, v109
	v_exp_f32_e32 v111, v111
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[96:97], v[66:81]
	v_exp_f32_e32 v114, v204
	v_exp_f32_e32 v101, v101
	v_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[94:95], v[66:81]
	v_exp_f32_e32 v110, v110
	v_exp_f32_e32 v98, v98
	v_cvt_pkrtz_f16_f32 v92, v206, v207
	v_cvt_pkrtz_f16_f32 v93, v208, v209
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[92:93], v[66:81]
	v_cvt_pkrtz_f16_f32 v87, v216, v217
	v_cvt_pkrtz_f16_f32 v89, v212, v213
	v_cvt_pkrtz_f16_f32 v86, v214, v215
	v_cvt_pkrtz_f16_f32 v88, v210, v211
	v_fmac_f32_e32 v205, 0xbe0293ee, v195
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x8_f16 v[50:65], v[192:193], v[92:93], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[176:177], v[92:93], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[152:153], v[92:93], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[190:191], v[90:91], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[174:175], v[90:91], v[34:49]
	v_mfma_f32_32x32x8_f16 v[34:49], v[172:173], v[88:89], v[34:49]
	v_exp_f32_e32 v172, v205
	v_mfma_f32_32x32x8_f16 v[18:33], v[150:151], v[90:91], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[90:91], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[188:189], v[88:89], v[50:65]
	v_mfma_f32_32x32x8_f16 v[18:33], v[148:149], v[88:89], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[88:89], v[66:81]
	v_mfma_f32_32x32x8_f16 v[50:65], v[186:187], v[86:87], v[50:65]
	v_mfma_f32_32x32x8_f16 v[34:49], v[170:171], v[86:87], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[146:147], v[86:87], v[18:33]
	v_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[86:87], v[66:81]
	; sched_barrier mask(0x00000000)
	s_barrier
	scratch_load_dword v2, off, off offset:116 ; 4-byte Folded Reload
	v_perm_b32 v1, v166, v162, s2
	s_waitcnt vmcnt(0)
	ds_write_b32 v2, v1 offset:16384
	v_perm_b32 v1, v166, v162, s3
	ds_write_b32 v250, v1 offset:16384
	v_perm_b32 v1, v167, v163, s2
	ds_write_b32 v251, v1 offset:16384
	v_perm_b32 v1, v167, v163, s3
	ds_write_b32 v228, v1 offset:16384
	v_perm_b32 v1, v168, v164, s2
	ds_write_b32 v229, v1 offset:16384
	v_perm_b32 v1, v168, v164, s3
	ds_write_b32 v230, v1 offset:16384
	v_perm_b32 v1, v169, v165, s2
	ds_write_b32 v231, v1 offset:16384
	v_perm_b32 v1, v169, v165, s3
	ds_write_b32 v232, v1 offset:16384
	; sched_barrier mask(0x00000000)
	v_add_f32_e32 v1, v128, v127
	v_add_f32_e32 v1, v126, v1
	v_add_f32_e32 v1, v120, v1
	v_add_f32_e32 v1, v125, v1
	v_add_f32_e32 v1, v124, v1
	v_add_f32_e32 v1, v123, v1
	v_add_f32_e32 v1, v122, v1
	v_add_f32_e32 v1, v121, v1
	v_add_f32_e32 v1, v113, v1
	v_add_f32_e32 v1, v119, v1
	v_add_f32_e32 v1, v118, v1
	v_add_f32_e32 v1, v117, v1
	v_add_f32_e32 v1, v116, v1
	v_add_f32_e32 v1, v115, v1
	v_add_f32_e32 v1, v112, v1
	v_add_f32_e32 v1, v111, v1
	v_add_f32_e32 v1, v110, v1
	v_add_f32_e32 v1, v109, v1
	v_add_f32_e32 v1, v108, v1
	v_add_f32_e32 v1, v107, v1
	v_add_f32_e32 v1, v106, v1
	v_add_f32_e32 v1, v105, v1
	v_add_f32_e32 v1, v104, v1
	v_add_f32_e32 v1, v103, v1
	v_add_f32_e32 v1, v102, v1
	v_add_f32_e32 v1, v101, v1
	v_add_f32_e32 v1, v100, v1
	v_add_f32_e32 v1, v99, v1
	v_add_f32_e32 v1, v98, v1
	v_add_f32_e32 v1, v129, v1
	v_add_f32_e32 v1, v114, v1
	ds_bpermute_b32 v2, v194, v1
	v_cvt_pkrtz_f16_f32 v82, v128, v127
	v_cvt_pkrtz_f16_f32 v83, v126, v120
	v_cvt_pkrtz_f16_f32 v84, v125, v124
	v_cvt_pkrtz_f16_f32 v85, v123, v122
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v2
	v_fmac_f32_e32 v1, v218, v172
	v_cvt_pkrtz_f16_f32 v86, v121, v113
	v_cvt_pkrtz_f16_f32 v87, v119, v118
	v_cvt_pkrtz_f16_f32 v88, v117, v116
	v_cvt_pkrtz_f16_f32 v89, v115, v112
	v_cvt_pkrtz_f16_f32 v90, v111, v110
	v_cvt_pkrtz_f16_f32 v91, v109, v108
	v_cvt_pkrtz_f16_f32 v92, v107, v106
	v_cvt_pkrtz_f16_f32 v93, v105, v104
	v_cvt_pkrtz_f16_f32 v94, v103, v102
	v_cvt_pkrtz_f16_f32 v95, v101, v100
	v_cvt_pkrtz_f16_f32 v96, v99, v98
	v_cvt_pkrtz_f16_f32 v97, v129, v114
	; sched_barrier mask(0x00000000)
	s_barrier
	scratch_load_dword v2, off, off offset:160 ; 4-byte Folded Reload
	ds_read_b64 v[100:101], v233 offset:16384
	ds_read_b64 v[102:103], v234 offset:16384
	ds_read_b64 v[104:105], v239 offset:16384
	ds_read_b64 v[106:107], v241 offset:16384
	ds_read_b64 v[108:109], v219 offset:16384
	ds_read_b64 v[110:111], v246 offset:16384
	ds_read_b64 v[112:113], v255 offset:16384
	ds_read_b64 v[114:115], v222 offset:16384
	ds_read_b64 v[116:117], v220 offset:16384
	ds_read_b64 v[118:119], v221 offset:16384
	ds_read_b64 v[120:121], v223 offset:16384
	ds_read_b64 v[122:123], v224 offset:16384
	ds_read_b64 v[124:125], v225 offset:16384
	ds_read_b64 v[126:127], v235 offset:16384
	ds_read_b64 v[128:129], v236 offset:16384
	ds_read_b64 v[130:131], v237 offset:16384
	ds_read_b64 v[132:133], v238 offset:16384
	ds_read_b64 v[134:135], v240 offset:16384
	ds_read_b64 v[136:137], v242 offset:16384
	ds_read_b64 v[138:139], v243 offset:16384
	ds_read_b64 v[140:141], v244 offset:16384
	ds_read_b64 v[142:143], v245 offset:16384
	ds_read_b64 v[144:145], v247 offset:16384
	ds_read_b64 v[146:147], v248 offset:16384
	ds_read_b64 v[148:149], v249 offset:16384
	ds_read_b64 v[150:151], v252 offset:16384
	ds_read_b64 v[152:153], v253 offset:16384
	ds_read_b64 v[154:155], v226 offset:16384
	ds_read_b64 v[158:159], v0 offset:16384
	ds_read_b64 v[160:161], v227 offset:16384
	s_waitcnt vmcnt(0)
	ds_read_b64 v[98:99], v2 offset:16384
	scratch_load_dword v2, off, off offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	ds_read_b64 v[156:157], v2 offset:16384
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v6, v54, v172
	v_mul_f32_e32 v54, v70, v172
	scratch_load_dword v70, off, off offset:144 ; 4-byte Folded Reload
	v_mul_f32_e32 v2, v50, v172
	v_mul_f32_e32 v3, v51, v172
	v_mul_f32_e32 v4, v52, v172
	v_mul_f32_e32 v5, v53, v172
	v_mul_f32_e32 v7, v55, v172
	v_mul_f32_e32 v8, v56, v172
	v_mul_f32_e32 v9, v57, v172
	v_mul_f32_e32 v10, v58, v172
	v_mul_f32_e32 v11, v59, v172
	v_mul_f32_e32 v12, v60, v172
	v_mul_f32_e32 v13, v61, v172
	v_mul_f32_e32 v14, v62, v172
	v_mul_f32_e32 v15, v63, v172
	v_mul_f32_e32 v16, v64, v172
	v_mul_f32_e32 v17, v65, v172
	v_mul_f32_e32 v34, v34, v172
	v_mul_f32_e32 v35, v35, v172
	v_mul_f32_e32 v36, v36, v172
	v_mul_f32_e32 v37, v37, v172
	v_mul_f32_e32 v38, v38, v172
	v_mul_f32_e32 v39, v39, v172
	v_mul_f32_e32 v40, v40, v172
	v_mul_f32_e32 v41, v41, v172
	v_mul_f32_e32 v42, v42, v172
	v_mul_f32_e32 v43, v43, v172
	v_mul_f32_e32 v44, v44, v172
	v_mul_f32_e32 v45, v45, v172
	v_mul_f32_e32 v46, v46, v172
	v_mul_f32_e32 v47, v47, v172
	v_mul_f32_e32 v48, v48, v172
	v_mul_f32_e32 v49, v49, v172
	v_mul_f32_e32 v18, v18, v172
	v_mul_f32_e32 v19, v19, v172
	v_mul_f32_e32 v20, v20, v172
	v_mul_f32_e32 v21, v21, v172
	v_mul_f32_e32 v22, v22, v172
	v_mul_f32_e32 v23, v23, v172
	v_mul_f32_e32 v24, v24, v172
	v_mul_f32_e32 v25, v25, v172
	v_mul_f32_e32 v26, v26, v172
	v_mul_f32_e32 v27, v27, v172
	v_mul_f32_e32 v28, v28, v172
	v_mul_f32_e32 v29, v29, v172
	v_mul_f32_e32 v30, v30, v172
	v_mul_f32_e32 v31, v31, v172
	v_mul_f32_e32 v32, v32, v172
	v_mul_f32_e32 v33, v33, v172
	v_mul_f32_e32 v50, v66, v172
	v_mul_f32_e32 v51, v67, v172
	v_mul_f32_e32 v52, v68, v172
	v_mul_f32_e32 v53, v69, v172
	v_mul_f32_e32 v55, v71, v172
	v_mul_f32_e32 v56, v72, v172
	v_mul_f32_e32 v57, v73, v172
	v_mul_f32_e32 v58, v74, v172
	v_mul_f32_e32 v59, v75, v172
	v_mul_f32_e32 v60, v76, v172
	v_mul_f32_e32 v61, v77, v172
	v_mul_f32_e32 v62, v78, v172
	v_mul_f32_e32 v63, v79, v172
	v_mul_f32_e32 v64, v80, v172
	v_mul_f32_e32 v65, v81, v172
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[98:99], v[82:83], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[114:115], v[82:83], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[130:131], v[82:83], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[146:147], v[82:83], v[50:65]
	v_mfma_f32_32x32x8_f16 v[2:17], v[100:101], v[84:85], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[116:117], v[84:85], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[132:133], v[84:85], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[148:149], v[84:85], v[50:65]
	v_mfma_f32_32x32x8_f16 v[2:17], v[102:103], v[86:87], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[118:119], v[86:87], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[134:135], v[86:87], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[150:151], v[86:87], v[50:65]
	v_mfma_f32_32x32x8_f16 v[2:17], v[104:105], v[88:89], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[120:121], v[88:89], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[136:137], v[88:89], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[152:153], v[88:89], v[50:65]
	v_mfma_f32_32x32x8_f16 v[2:17], v[106:107], v[90:91], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[122:123], v[90:91], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[138:139], v[90:91], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[154:155], v[90:91], v[50:65]
	v_mfma_f32_32x32x8_f16 v[2:17], v[108:109], v[92:93], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[124:125], v[92:93], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[140:141], v[92:93], v[18:33]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[50:65], v[156:157], v[92:93], v[50:65]
	v_mfma_f32_32x32x8_f16 v[2:17], v[110:111], v[94:95], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[126:127], v[94:95], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[142:143], v[94:95], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[158:159], v[94:95], v[50:65]
	v_mfma_f32_32x32x8_f16 v[2:17], v[112:113], v[96:97], v[2:17]
	v_mfma_f32_32x32x8_f16 v[34:49], v[128:129], v[96:97], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[144:145], v[96:97], v[18:33]
	v_mfma_f32_32x32x8_f16 v[50:65], v[160:161], v[96:97], v[50:65]
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v66, v70, 2, 0
	s_barrier
	s_cbranch_scc1 .LBB0_8
; %bb.7:
	scratch_load_dword v0, off, off offset:132 ; 4-byte Folded Reload
	s_mov_b32 s3, 0x800000
	v_cmp_gt_f32_e32 vcc, s3, v1
	s_nop 1
	v_cndmask_b32_e64 v69, 0, 32, vcc
	v_ldexp_f32 v69, v1, v69
	v_log_f32_e32 v69, v69
	v_mov_b32_e32 v68, 0x42000000
	v_or_b32_e32 v67, s34, v70
	s_movk_i32 s2, 0x4000
	v_cndmask_b32_e32 v68, 0, v68, vcc
	v_cmp_gt_i32_e64 s[8:9], s2, v67
	v_sub_f32_e32 v67, v69, v68
	v_add_f32_e32 v67, v195, v67
	ds_write_b32 v66, v67
	v_mov_b32_e32 v67, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_sub_i32 s2, 0x4000, s34
	v_bfrev_b32_e32 v69, 1
	s_and_b32 s5, s19, 0xffff
	s_mov_b32 s6, s14
	s_mov_b32 s7, s15
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v67, v67, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v68, 0, v67
	ds_read_b32 v68, v68
	v_cmp_lt_i32_sdwa s[2:3], v0, s2 src0_sel:BYTE_0 src1_sel:DWORD
	s_and_b64 vcc, s[0:1], s[2:3]
	v_cndmask_b32_e32 v67, v69, v67, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v68, v67, s[4:7], 0 offen
	s_cbranch_execz .LBB0_9
	s_branch .LBB0_10
.LBB0_8:
                                        ; implicit-def: $sgpr8_sgpr9
.LBB0_9:
	s_mov_b32 s2, 0x800000
	v_cmp_gt_f32_e32 vcc, s2, v1
	s_nop 1
	v_cndmask_b32_e64 v68, 0, 32, vcc
	v_ldexp_f32 v68, v1, v68
	v_log_f32_e32 v68, v68
	v_mov_b32_e32 v67, 0x42000000
	v_cndmask_b32_e32 v67, 0, v67, vcc
	s_and_b32 s5, s19, 0xffff
	v_sub_f32_e32 v67, v68, v67
	v_add_f32_e32 v67, v195, v67
	ds_write_b32 v66, v67
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v0, off, off offset:132 ; 4-byte Folded Reload
	v_mov_b32_e32 v66, 2
	v_bfrev_b32_e32 v67, 1
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	s_or_b64 s[8:9], s[8:9], exec
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_sdwa v0, v66, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_add_u32_e32 v66, 0, v0
	ds_read_b32 v66, v66
	v_cndmask_b32_e64 v0, v67, v0, s[0:1]
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v66, v0, s[4:7], 0 offen
.LBB0_10:                               ; %.critedge
	v_div_scale_f32 v0, s[0:1], v1, v1, 1.0
	v_rcp_f32_e32 v0, v0
	v_div_scale_f32 v66, vcc, 1.0, v1, 1.0
	v_mov_b32_e32 v67, v64
	v_mul_f32_e32 v0, v66, v0
	s_nop 1
	v_div_fmas_f32 v0, 0, 0, v0
	v_div_fixup_f32 v0, v0, v1, 1.0
	v_mov_b32_e32 v66, v63
	v_fma_mixlo_f16 v68, v0, v65, 0
	v_pk_mul_f32 v[64:65], v[0:1], v[66:67] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v66, v0, v62, 0
	v_mov_b32_e32 v62, v59
	v_mov_b32_e32 v63, v60
	v_fma_mixlo_f16 v67, v0, v61, 0
	v_pk_mul_f32 v[60:61], v[0:1], v[62:63] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v62, v0, v58, 0
	v_mov_b32_e32 v58, v55
	v_mov_b32_e32 v59, v56
	v_fma_mixlo_f16 v63, v0, v57, 0
	v_pk_mul_f32 v[56:57], v[0:1], v[58:59] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v58, v0, v54, 0
	v_mov_b32_e32 v54, v51
	v_mov_b32_e32 v55, v52
	v_fma_mixlo_f16 v59, v0, v53, 0
	v_pk_mul_f32 v[52:53], v[0:1], v[54:55] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v54, v0, v50, 0
	v_mov_b32_e32 v50, v31
	v_mov_b32_e32 v51, v32
	v_fma_mixlo_f16 v55, v0, v33, 0
	v_pk_mul_f32 v[32:33], v[0:1], v[50:51] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v50, v0, v30, 0
	v_mov_b32_e32 v30, v27
	v_mov_b32_e32 v31, v28
	v_fma_mixlo_f16 v51, v0, v29, 0
	v_pk_mul_f32 v[28:29], v[0:1], v[30:31] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v30, v0, v26, 0
	v_mov_b32_e32 v26, v23
	v_mov_b32_e32 v27, v24
	v_fma_mixlo_f16 v31, v0, v25, 0
	v_pk_mul_f32 v[24:25], v[0:1], v[26:27] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v26, v0, v22, 0
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v20
	v_fma_mixlo_f16 v27, v0, v21, 0
	v_pk_mul_f32 v[20:21], v[0:1], v[22:23] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v22, v0, v18, 0
	v_mov_b32_e32 v18, v47
	v_mov_b32_e32 v19, v48
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v47, v19
	v_cvt_f16_f32_e32 v48, v18
	v_mov_b32_e32 v18, v43
	v_mov_b32_e32 v19, v44
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v43, v19
	v_cvt_f16_f32_e32 v44, v18
	v_mov_b32_e32 v18, v39
	v_mov_b32_e32 v19, v40
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v39, v19
	v_cvt_f16_f32_e32 v40, v18
	v_mov_b32_e32 v18, v35
	v_mov_b32_e32 v19, v36
	v_pk_mul_f32 v[18:19], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v35, v19
	v_cvt_f16_f32_e32 v36, v18
	v_mov_b32_e32 v18, v15
	v_mov_b32_e32 v19, v16
	v_fma_mixlo_f16 v23, v0, v49, 0
	v_fma_mixlo_f16 v49, v0, v17, 0
	v_pk_mul_f32 v[16:17], v[0:1], v[18:19] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v18, v0, v14, 0
	v_mov_b32_e32 v14, v11
	v_mov_b32_e32 v15, v12
	v_fma_mixlo_f16 v19, v0, v13, 0
	v_pk_mul_f32 v[12:13], v[0:1], v[14:15] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v14, v0, v10, 0
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v8
	v_fma_mixlo_f16 v15, v0, v9, 0
	v_pk_mul_f32 v[8:9], v[0:1], v[10:11] op_sel_hi:[0,1]
	v_fma_mixlo_f16 v10, v0, v6, 0
	v_fma_mixlo_f16 v1, v0, v5, 0
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v4
	v_pk_mul_f32 v[4:5], v[0:1], v[6:7] op_sel_hi:[0,1]
	v_cvt_f16_f32_e32 v3, v5
	s_mov_b32 s4, 0x5040100
	s_mul_i32 s0, s24, s18
	s_ashr_i32 s1, s0, 31
	v_perm_b32 v1, v1, v3, s4
	scratch_load_dword v3, off, off offset:128 ; 4-byte Folded Reload
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	s_mul_i32 s0, s25, s17
	s_addc_u32 s3, s11, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	s_mul_i32 s0, s16, s34
	s_addc_u32 s3, s3, s1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_f16_f32_e32 v4, v4
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s16, 0x3fff
	v_fma_mixlo_f16 v46, v0, v46, 0
	v_fma_mixlo_f16 v45, v0, v45, 0
	v_fma_mixlo_f16 v42, v0, v42, 0
	v_fma_mixlo_f16 v41, v0, v41, 0
	v_fma_mixlo_f16 v38, v0, v38, 0
	v_fma_mixlo_f16 v37, v0, v37, 0
	v_fma_mixlo_f16 v34, v0, v34, 0
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v8, v8
	v_fma_mixlo_f16 v0, v0, v2, 0
	v_mul_lo_u32 v2, s16, v70
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s2, s1
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v12, v12
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_pack_b32_f16 v0, v0, v4
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v29, v29
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v33, v33
	v_cvt_f16_f32_e32 v32, v32
	v_cvt_f16_f32_e32 v53, v53
	v_cvt_f16_f32_e32 v52, v52
	v_cvt_f16_f32_e32 v57, v57
	v_cvt_f16_f32_e32 v56, v56
	v_cvt_f16_f32_e32 v61, v61
	v_cvt_f16_f32_e32 v60, v60
	v_cvt_f16_f32_e32 v65, v65
	v_cvt_f16_f32_e32 v64, v64
	s_waitcnt vmcnt(0)
	v_add_lshl_u32 v2, v2, v3, 1
	v_bfrev_b32_e32 v3, 1
	v_cndmask_b32_e64 v4, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 16, v2
	v_pack_b32_f16 v0, v10, v8
	v_perm_b32 v1, v15, v9, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 32, v2
	v_pack_b32_f16 v0, v14, v12
	v_perm_b32 v1, v19, v13, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 48, v2
	v_pack_b32_f16 v0, v18, v16
	v_perm_b32 v1, v49, v17, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 64, v2
	v_pack_b32_f16 v0, v34, v36
	v_perm_b32 v1, v37, v35, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x50, v2
	v_pack_b32_f16 v0, v38, v40
	v_perm_b32 v1, v41, v39, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x60, v2
	v_pack_b32_f16 v0, v42, v44
	v_perm_b32 v1, v45, v43, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x70, v2
	v_pack_b32_f16 v0, v46, v48
	v_perm_b32 v1, v23, v47, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x80, v2
	v_pack_b32_f16 v0, v22, v20
	v_perm_b32 v1, v27, v21, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0x90, v2
	v_pack_b32_f16 v0, v26, v24
	v_perm_b32 v1, v31, v25, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xa0, v2
	v_pack_b32_f16 v0, v30, v28
	v_perm_b32 v1, v51, v29, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xb0, v2
	v_pack_b32_f16 v0, v50, v32
	v_perm_b32 v1, v55, v33, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xc0, v2
	v_pack_b32_f16 v0, v54, v52
	v_perm_b32 v1, v59, v53, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xd0, v2
	v_pack_b32_f16 v0, v58, v56
	v_perm_b32 v1, v63, v57, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_add_u32_e32 v4, 0xe0, v2
	v_pack_b32_f16 v0, v62, v60
	v_perm_b32 v1, v67, v61, s4
	v_cndmask_b32_e64 v4, v3, v4, s[8:9]
	v_add_u32_e32 v2, 0xf0, v2
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_pack_b32_f16 v0, v66, v64
	v_perm_b32 v1, v68, v65, s4
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 168
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
		.amdhsa_next_free_sgpr 44
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
	.set attn_fwd.numbered_sgpr, 44
	.set attn_fwd.private_seg_size, 168
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15700
; TotalNumSgprs: 50
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 168
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 50
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
    .private_segment_fixed_size: 168
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 43
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
