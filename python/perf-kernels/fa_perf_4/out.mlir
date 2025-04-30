INFO: running a single config given from a file
250 1000
src= 	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	attn_fwd                        ; -- Begin function attn_fwd
	.p2align	8
	.type	attn_fwd,@function
attn_fwd:                               ; @attn_fwd
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.23:
	.file	1 "ttgir" "7.ttgir"
	.loc	1 47 0 prologue_end             ; 7.ttgir:47:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.24:
.LBB0_0:
.Ltmp1:
	.loc	1 97 10 is_stmt 1               ; 7.ttgir:97:10
	s_load_dwordx8 s[40:47], s[0:1], 0x38
	s_load_dwordx4 s[20:23], s[0:1], 0x70
	.loc	1 111 11                        ; 7.ttgir:111:11
	s_ashr_i32 s19, s18, 31
	.loc	1 100 10                        ; 7.ttgir:100:10
	s_lshl_b32 s52, s16, 7
	.loc	1 111 11                        ; 7.ttgir:111:11
	s_lshl_b64 s[0:1], s[18:19], 2
	.loc	1 153 11                        ; 7.ttgir:153:11
	v_lshlrev_b32_e32 v2, 3, v0
	.loc	1 111 11                        ; 7.ttgir:111:11
	s_waitcnt lgkmcnt(0)
	s_add_u32 s20, s20, s0
	s_addc_u32 s21, s21, s1
	.loc	1 112 11                        ; 7.ttgir:112:11
	s_load_dwordx2 s[54:55], s[20:21], 0x0
	.loc	1 102 10                        ; 7.ttgir:102:10
	v_lshrrev_b32_e32 v158, 4, v0
	.loc	1 153 11                        ; 7.ttgir:153:11
	v_and_b32_e32 v138, 0x78, v2
	.loc	1 102 10                        ; 7.ttgir:102:10
	v_or_b32_e32 v159, 16, v158
	v_or_b32_e32 v1, s52, v158
	.loc	1 115 11                        ; 7.ttgir:115:11
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s16, s55, s54
	.loc	1 116 11                        ; 7.ttgir:116:11
	s_add_u32 s0, s22, s0
	s_addc_u32 s1, s23, s1
	.loc	1 117 11                        ; 7.ttgir:117:11
	s_load_dwordx2 s[20:21], s[0:1], 0x0
	.loc	1 107 11                        ; 7.ttgir:107:11
	v_or_b32_e32 v4, s52, v159
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_bfrev_b32_e32 v31, 1
	.loc	1 190 11                        ; 7.ttgir:190:11
	v_cmp_gt_i32_e32 vcc, s16, v1
	.loc	1 107 11                        ; 7.ttgir:107:11
	v_or_b32_e32 v12, 32, v1
	.loc	1 120 11                        ; 7.ttgir:120:11
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s19, s21, s20
	.loc	1 121 11                        ; 7.ttgir:121:11
	s_add_i32 s0, s19, 31
	.loc	1 122 11                        ; 7.ttgir:122:11
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 27
	s_add_i32 s0, s0, s1
	.loc	1 126 11                        ; 7.ttgir:126:11
	s_sub_i32 s1, s52, s16
	s_add_i32 s1, s1, s19
	.loc	1 127 11                        ; 7.ttgir:127:11
	s_addk_i32 s1, 0x9f
	.loc	1 128 11                        ; 7.ttgir:128:11
	s_ashr_i32 s21, s1, 31
	s_lshr_b32 s21, s21, 27
	s_add_i32 s1, s1, s21
	.loc	1 122 11                        ; 7.ttgir:122:11
	s_ashr_i32 s0, s0, 5
	.loc	1 128 11                        ; 7.ttgir:128:11
	s_ashr_i32 s1, s1, 5
	.loc	1 129 11                        ; 7.ttgir:129:11
	s_min_i32 s72, s0, s1
	.loc	1 131 5                         ; 7.ttgir:131:5
	s_and_b32 s0, s19, 31
	s_sub_i32 s1, 32, s19
	.loc	1 130 11                        ; 7.ttgir:130:11
	s_cmp_lt_i32 s19, 32
	.loc	1 131 5                         ; 7.ttgir:131:5
	s_cselect_b32 s67, s1, s0
	.loc	1 139 11                        ; 7.ttgir:139:11
	s_mul_i32 s0, s12, s18
	.loc	1 140 11                        ; 7.ttgir:140:11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	.loc	1 141 11                        ; 7.ttgir:141:11
	s_mul_i32 s0, s13, s17
	.loc	1 140 11                        ; 7.ttgir:140:11
	s_addc_u32 s3, s3, s1
	.loc	1 142 11                        ; 7.ttgir:142:11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	.loc	1 143 11                        ; 7.ttgir:143:11
	s_mul_i32 s0, s54, s14
	.loc	1 142 11                        ; 7.ttgir:142:11
	s_addc_u32 s3, s3, s1
	.loc	1 144 11                        ; 7.ttgir:144:11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	.loc	1 148 11                        ; 7.ttgir:148:11
	s_mul_i32 s0, s14, s52
	.loc	1 144 11                        ; 7.ttgir:144:11
	s_addc_u32 s3, s3, s1
	.loc	1 151 11                        ; 7.ttgir:151:11
	s_ashr_i32 s1, s0, 31
	.loc	1 150 11                        ; 7.ttgir:150:11
	s_lshl_b32 s12, s14, 4
	.loc	1 151 11                        ; 7.ttgir:151:11
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	.loc	1 158 11                        ; 7.ttgir:158:11
	v_mad_u64_u32 v[2:3], s[2:3], s14, v158, v[138:139]
	.loc	1 159 11                        ; 7.ttgir:159:11
	s_mul_i32 s2, s15, s18
	.loc	1 160 11                        ; 7.ttgir:160:11
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[34:35], s[2:3], 1
	s_add_u32 s13, s4, s34
	.loc	1 161 11                        ; 7.ttgir:161:11
	s_mul_i32 s2, s40, s17
	.loc	1 160 11                        ; 7.ttgir:160:11
	s_addc_u32 s15, s5, s35
	.loc	1 162 11                        ; 7.ttgir:162:11
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[56:57], s[2:3], 1
	s_add_u32 s13, s13, s56
	.loc	1 163 11                        ; 7.ttgir:163:11
	s_mul_i32 s2, s20, s41
	.loc	1 162 11                        ; 7.ttgir:162:11
	s_addc_u32 s15, s15, s57
	.loc	1 164 11                        ; 7.ttgir:164:11
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[58:59], s[2:3], 1
	s_add_u32 s28, s13, s58
	.loc	1 174 11                        ; 7.ttgir:174:11
	s_mul_i32 s2, s42, s18
	.loc	1 164 11                        ; 7.ttgir:164:11
	s_addc_u32 s40, s15, s59
	.loc	1 175 11                        ; 7.ttgir:175:11
	s_ashr_i32 s3, s2, 31
	.loc	1 171 11                        ; 7.ttgir:171:11
	s_lshl_b32 s13, s41, 4
	.loc	1 175 11                        ; 7.ttgir:175:11
	s_lshl_b64 s[60:61], s[2:3], 1
	s_add_u32 s15, s6, s60
	.loc	1 176 11                        ; 7.ttgir:176:11
	s_mul_i32 s2, s43, s17
	.loc	1 175 11                        ; 7.ttgir:175:11
	s_addc_u32 s21, s7, s61
	.loc	1 177 11                        ; 7.ttgir:177:11
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[42:43], s[2:3], 1
	s_add_u32 s15, s15, s42
	.loc	1 178 11                        ; 7.ttgir:178:11
	s_mul_i32 s2, s20, s44
	.loc	1 177 11                        ; 7.ttgir:177:11
	s_addc_u32 s21, s21, s43
	.loc	1 179 11                        ; 7.ttgir:179:11
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[62:63], s[2:3], 1
	s_add_u32 s36, s15, s62
	s_addc_u32 s66, s21, s63
	.loc	1 194 11                        ; 7.ttgir:194:11
	s_and_b32 s2, s14, 0x3fff
	s_bitset1_b32 s2, 14
	.loc	1 158 11                        ; 7.ttgir:158:11
	v_add_u32_e32 v3, s12, v2
	.loc	1 194 11                        ; 7.ttgir:194:11
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	v_lshlrev_b32_e32 v2, 1, v2
	.loc	1 107 11                        ; 7.ttgir:107:11
	v_or_b32_e32 v13, 48, v1
	v_or_b32_e32 v20, 64, v1
	v_or_b32_e32 v21, 0x50, v1
	v_or_b32_e32 v28, 0x60, v1
	v_or_b32_e32 v29, 0x70, v1
	.loc	1 158 11                        ; 7.ttgir:158:11
	v_add_u32_e32 v14, s12, v3
	.loc	1 194 11                        ; 7.ttgir:194:11
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v1, v31, v2, vcc
	v_lshlrev_b32_e32 v2, 1, v3
	.loc	1 190 11                        ; 7.ttgir:190:11
	v_cmp_gt_i32_e32 vcc, s16, v4
	.loc	1 158 11                        ; 7.ttgir:158:11
	v_add_u32_e32 v15, s12, v14
	v_add_u32_e32 v22, s12, v15
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_cndmask_b32_e32 v2, v31, v2, vcc
	buffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[8:11], v2, s[0:3], 0 offen
	v_lshlrev_b32_e32 v1, 1, v14
	.loc	1 190 11                        ; 7.ttgir:190:11
	v_cmp_gt_i32_e32 vcc, s16, v12
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_lshlrev_b32_e32 v2, 1, v15
	.loc	1 158 11                        ; 7.ttgir:158:11
	v_add_u32_e32 v23, s12, v22
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_cndmask_b32_e32 v1, v31, v1, vcc
	.loc	1 190 11                        ; 7.ttgir:190:11
	v_cmp_gt_i32_e32 vcc, s16, v13
	.loc	1 158 11                        ; 7.ttgir:158:11
	v_add_u32_e32 v30, s12, v23
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_and_b32_e32 v38, 0x80, v0
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_cndmask_b32_e32 v2, v31, v2, vcc
	buffer_load_dwordx4 v[12:15], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[16:19], v2, s[0:3], 0 offen
	v_lshlrev_b32_e32 v1, 1, v22
	.loc	1 190 11                        ; 7.ttgir:190:11
	v_cmp_gt_i32_e32 vcc, s16, v20
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_lshlrev_b32_e32 v2, 1, v23
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_lshrrev_b32_e32 v152, 1, v38
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_cndmask_b32_e32 v1, v31, v1, vcc
	.loc	1 190 11                        ; 7.ttgir:190:11
	v_cmp_gt_i32_e32 vcc, s16, v21
	.loc	1 220 12                        ; 7.ttgir:220:12
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_cndmask_b32_e32 v2, v31, v2, vcc
	buffer_load_dwordx4 v[20:23], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[24:27], v2, s[0:3], 0 offen
	v_lshlrev_b32_e32 v1, 1, v30
	.loc	1 190 11                        ; 7.ttgir:190:11
	v_cmp_gt_i32_e32 vcc, s16, v28
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_add_lshl_u32 v2, v30, s12, 1
	.loc	1 222 12                        ; 7.ttgir:222:12
	s_mov_b32 s38, s2
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_cndmask_b32_e32 v1, v31, v1, vcc
	.loc	1 190 11                        ; 7.ttgir:190:11
	v_cmp_gt_i32_e32 vcc, s16, v29
	.loc	1 222 12                        ; 7.ttgir:222:12
	s_mov_b32 s39, s3
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_and_b32_e32 v153, 31, v0
	.loc	1 194 11                        ; 7.ttgir:194:11
	v_cndmask_b32_e32 v2, v31, v2, vcc
	buffer_load_dwordx4 v[28:31], v1, s[0:3], 0 offen
	buffer_load_dwordx4 v[32:35], v2, s[0:3], 0 offen
	.loc	1 173 11                        ; 7.ttgir:173:11
	v_mad_u64_u32 v[36:37], s[0:1], s41, v158, v[138:139]
	.loc	1 196 11                        ; 7.ttgir:196:11
	v_lshrrev_b32_e32 v37, 1, v0
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_and_b32_e32 v2, 64, v0
	.loc	1 196 11                        ; 7.ttgir:196:11
	v_and_b32_e32 v1, 24, v37
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_lshrrev_b32_e32 v151, 1, v2
	.loc	1 196 11                        ; 7.ttgir:196:11
	v_xor_b32_e32 v1, v1, v138
	v_xor_b32_e32 v1, v1, v151
	v_xor_b32_e32 v1, v1, v152
	.loc	1 202 11                        ; 7.ttgir:202:11
	s_and_b32 s0, s16, 0x7f
	.loc	1 196 11                        ; 7.ttgir:196:11
	v_lshlrev_b32_e32 v1, 1, v1
	.loc	1 203 11                        ; 7.ttgir:203:11
	s_or_b32 s0, s67, s0
	.loc	1 196 11                        ; 7.ttgir:196:11
	v_lshl_or_b32 v1, v158, 8, v1
	.loc	1 203 11                        ; 7.ttgir:203:11
	s_cmp_eq_u32 s0, 0
	.loc	1 196 11                        ; 7.ttgir:196:11
	v_add_u32_e32 v39, 0, v1
	.loc	1 206 12                        ; 7.ttgir:206:12
	s_cselect_b32 s12, 4, 5
	.loc	1 220 12                        ; 7.ttgir:220:12
	s_and_b32 s0, s41, 0x3fff
	.loc	1 196 11                        ; 7.ttgir:196:11
	s_waitcnt vmcnt(7)
	ds_write_b128 v39, v[4:7]
	s_waitcnt vmcnt(6)
	ds_write_b128 v39, v[8:11] offset:4096
	s_waitcnt vmcnt(5)
	ds_write_b128 v39, v[12:15] offset:8192
	s_waitcnt vmcnt(4)
	ds_write_b128 v39, v[16:19] offset:12288
	s_waitcnt vmcnt(3)
	ds_write_b128 v39, v[20:23] offset:16384
	s_waitcnt vmcnt(2)
	ds_write_b128 v39, v[24:27] offset:20480
	s_waitcnt vmcnt(1)
	ds_write_b128 v39, v[28:31] offset:24576
	s_waitcnt vmcnt(0)
	ds_write_b128 v39, v[32:35] offset:28672
	.loc	1 220 12                        ; 7.ttgir:220:12
	s_bitset1_b32 s0, 14
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_and_b32_e32 v7, 16, v0
	.loc	1 102 10                        ; 7.ttgir:102:10
	v_and_b32_e32 v8, 32, v0
	.loc	1 220 12                        ; 7.ttgir:220:12
	s_and_b32 s1, s40, 0xffff
	s_lshl_b32 s53, s0, 16
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_lshrrev_b32_e32 v1, 3, v8
	.loc	1 110 11                        ; 7.ttgir:110:11
	v_lshrrev_b32_e32 v3, 3, v7
	.loc	1 220 12                        ; 7.ttgir:220:12
	s_or_b32 s29, s1, s53
	v_lshlrev_b32_e32 v160, 1, v36
	.loc	1 110 11                        ; 7.ttgir:110:11
	v_or_b32_e32 v4, v3, v1
	v_lshrrev_b32_e32 v6, 3, v2
	v_lshrrev_b32_e32 v15, 3, v38
	.loc	1 197 5                         ; 7.ttgir:197:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 220 12                        ; 7.ttgir:220:12
	v_add_lshl_u32 v161, v36, s13, 1
	buffer_load_dwordx4 v[16:19], v160, s[28:31], 0 offen
	buffer_load_dwordx4 v[20:23], v161, s[28:31], 0 offen
	.loc	1 110 11                        ; 7.ttgir:110:11
	v_or3_b32 v162, v4, v6, v15
	.loc	1 186 11                        ; 7.ttgir:186:11
	v_mad_u64_u32 v[4:5], s[0:1], s44, v162, v[138:139]
	.loc	1 222 12                        ; 7.ttgir:222:12
	s_and_b32 s0, s44, 0x3fff
	s_bitset1_b32 s0, 14
	s_and_b32 s1, s66, 0xffff
	s_lshl_b32 s55, s0, 16
	s_or_b32 s37, s1, s55
	v_lshlrev_b32_e32 v163, 1, v4
	v_add_lshl_u32 v164, v4, s44, 1
	buffer_load_dwordx4 v[114:117], v163, s[36:39], 0 offen
	buffer_load_dwordx4 v[118:121], v164, s[36:39], 0 offen
	.loc	1 199 11                        ; 7.ttgir:199:11
	v_lshrrev_b32_e32 v5, 2, v0
	s_movk_i32 s13, 0x60
	v_and_b32_e32 v141, 8, v5
	v_and_b32_e32 v4, 15, v0
	v_bfe_u32 v24, v0, 5, 1
	v_and_or_b32 v5, v37, s13, v153
	v_or_b32_e32 v9, 32, v141
	v_xor_b32_e32 v25, v24, v4
	v_or_b32_e32 v24, 2, v24
	v_lshlrev_b32_e32 v139, 3, v4
	v_or_b32_e32 v10, 48, v141
	v_or_b32_e32 v11, 64, v141
	v_lshlrev_b32_e32 v25, 3, v25
	v_lshlrev_b32_e32 v5, 7, v5
	v_xor_b32_e32 v24, v24, v4
	v_xor_b32_e32 v9, v9, v139
	v_or_b32_e32 v12, 0x50, v141
	v_or_b32_e32 v13, 0x60, v141
	v_or_b32_e32 v26, v25, v5
	v_lshlrev_b32_e32 v24, 3, v24
	v_or_b32_e32 v4, v5, v9
	v_xor_b32_e32 v10, v10, v139
	v_xor_b32_e32 v11, v11, v139
	v_or_b32_e32 v14, 0x70, v141
	v_or_b32_e32 v27, v24, v5
	v_or_b32_e32 v28, v5, v10
	v_or_b32_e32 v29, v5, v11
	v_xor_b32_e32 v12, v12, v139
	v_xor_b32_e32 v13, v13, v139
	v_lshl_add_u32 v26, v26, 1, 0
	v_lshl_add_u32 v4, v4, 1, 0
	v_or_b32_e32 v30, v5, v12
	v_or_b32_e32 v31, v5, v13
	v_xor_b32_e32 v14, v14, v139
	v_lshl_add_u32 v27, v27, 1, 0
	ds_read_b128 v[110:113], v26
	ds_read_b128 v[106:109], v27
	v_lshl_add_u32 v26, v28, 1, 0
	ds_read_b128 v[102:105], v4
	ds_read_b128 v[98:101], v26
	v_lshl_add_u32 v4, v29, 1, 0
	v_or_b32_e32 v5, v5, v14
	v_lshl_add_u32 v26, v30, 1, 0
	ds_read_b128 v[94:97], v4
	ds_read_b128 v[90:93], v26
	v_lshl_add_u32 v4, v31, 1, 0
	v_lshl_add_u32 v5, v5, 1, 0
	ds_read_b128 v[86:89], v4
	ds_read_b128 v[82:85], v5
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_and_b32_e32 v4, 1, v0
	v_cmp_eq_u32_e64 s[24:25], 0, v4
	v_and_b32_e32 v4, 2, v0
	v_cmp_eq_u32_e64 s[26:27], 0, v4
	v_and_b32_e32 v4, 4, v0
	v_cmp_eq_u32_e64 s[20:21], 0, v4
	v_and_b32_e32 v4, 8, v0
	.loc	1 242 14                        ; 7.ttgir:242:14
	v_lshlrev_b32_e32 v140, 7, v153
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_cmp_eq_u32_e64 s[22:23], 0, v4
	.loc	1 242 14                        ; 7.ttgir:242:14
	v_or_b32_e32 v4, v25, v140
	.loc	1 224 5                         ; 7.ttgir:224:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(3)
	.loc	1 226 5                         ; 7.ttgir:226:5
	ds_write_b128 v39, v[16:19]
	s_waitcnt vmcnt(2)
	ds_write_b128 v39, v[20:23] offset:4096
	.loc	1 242 14                        ; 7.ttgir:242:14
	v_lshl_add_u32 v177, v4, 1, 0
	.loc	1 243 14                        ; 7.ttgir:243:14
	v_or_b32_e32 v4, v24, v140
	.loc	1 207 12                        ; 7.ttgir:207:12
	s_min_i32 s1, s12, s72
	.loc	1 243 14                        ; 7.ttgir:243:14
	v_lshl_add_u32 v179, v4, 1, 0
	.loc	1 242 14                        ; 7.ttgir:242:14
	ds_read_b128 v[126:129], v177
	.loc	1 243 14                        ; 7.ttgir:243:14
	ds_read_b128 v[122:125], v179
	.loc	1 208 12                        ; 7.ttgir:208:12
	s_sub_i32 s0, s72, s1
	.loc	1 209 12                        ; 7.ttgir:209:12
	s_lshl_b32 s33, s0, 5
	.loc	1 215 12                        ; 7.ttgir:215:12
	s_lshl_b32 s30, s41, 5
	.loc	1 211 12                        ; 7.ttgir:211:12
	s_or_b32 s2, s33, 31
	.loc	1 233 13                        ; 7.ttgir:233:13
	s_ashr_i32 s31, s30, 31
	.loc	1 101 10                        ; 7.ttgir:101:10
	v_cmp_eq_u32_e64 s[82:83], 0, v38
	v_or3_b32 v5, v6, v1, v15
	.loc	1 297 12                        ; 7.ttgir:297:12
	s_cmp_gt_i32 s2, 63
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v156, 8, v1
	v_or_b32_e32 v155, 16, v1
	v_or_b32_e32 v154, 24, v1
	v_or_b32_e32 v165, v9, v140
	v_or_b32_e32 v168, v10, v140
	v_or_b32_e32 v169, v11, v140
	v_or_b32_e32 v170, v12, v140
	v_or_b32_e32 v172, v13, v140
	v_or_b32_e32 v174, v14, v140
	v_lshlrev_b32_e32 v6, 2, v0
	v_lshlrev_b32_e32 v4, 5, v153
	.loc	1 298 5                         ; 7.ttgir:298:5
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	.loc	1 658 12                        ; 7.ttgir:658:12
	v_or_b32_e32 v135, v9, v140
	.loc	1 899 5                         ; 7.ttgir:899:5
	v_mov_b32_e32 v9, 0x440
	.loc	1 662 12                        ; 7.ttgir:662:12
	v_or_b32_e32 v133, v11, v140
	.loc	1 899 5                         ; 7.ttgir:899:5
	v_cndmask_b32_e64 v148, v9, 0, s[82:83]
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_mov_b32_e32 v9, 0x110
	v_mov_b32_e32 v11, 0x204
	.loc	1 660 12                        ; 7.ttgir:660:12
	v_or_b32_e32 v134, v10, v140
	.loc	1 666 12                        ; 7.ttgir:666:12
	v_or_b32_e32 v131, v13, v140
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_cndmask_b32_e64 v10, v9, 0, s[24:25]
	v_cndmask_b32_e64 v11, v11, 0, s[26:27]
	v_mov_b32_e32 v13, 0x408
	.loc	1 664 12                        ; 7.ttgir:664:12
	v_or_b32_e32 v132, v12, v140
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_or_b32_e32 v12, v10, v11
	v_cndmask_b32_e64 v13, v13, 0, s[20:21]
	v_mov_b32_e32 v15, 0x810
	.loc	1 668 12                        ; 7.ttgir:668:12
	v_or_b32_e32 v130, v14, v140
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_or_b32_e32 v14, v12, v13
	v_cndmask_b32_e64 v15, v15, 0, s[22:23]
	v_xor_b32_e32 v16, v14, v15
	v_or_b32_e32 v14, 32, v14
	v_xor_b32_e32 v14, v14, v15
	v_or_b32_e32 v14, v14, v3
	v_xor_b32_e32 v143, v5, v14
	v_or_b32_e32 v14, 0x44, v10
	v_xor_b32_e32 v14, v14, v11
	v_or_b32_e32 v14, v14, v13
	v_xor_b32_e32 v14, v14, v15
	v_or_b32_e32 v14, v14, v3
	v_xor_b32_e32 v144, v5, v14
	v_or_b32_e32 v14, 0x64, v10
	v_xor_b32_e32 v14, v14, v11
	v_or_b32_e32 v14, v14, v13
	v_xor_b32_e32 v14, v14, v15
	v_or_b32_e32 v14, v14, v3
	v_xor_b32_e32 v145, v5, v14
	v_or_b32_e32 v14, 0x88, v12
	v_or_b32_e32 v13, v15, v13
	v_or_b32_e32 v12, 0xa8, v12
	v_xor_b32_e32 v12, v13, v12
	v_or_b32_e32 v12, v12, v3
	v_xor_b32_e32 v147, v5, v12
	v_or_b32_e32 v12, 0xcc, v10
	v_or_b32_e32 v11, v13, v11
	v_xor_b32_e32 v12, v11, v12
	v_or_b32_e32 v10, 0xec, v10
	v_or_b32_e32 v12, v12, v3
	v_xor_b32_e32 v10, v11, v10
	v_xor_b32_e32 v14, v13, v14
	v_xor_b32_e32 v149, v5, v12
	v_or_b32_e32 v10, v10, v3
	.loc	1 788 12                        ; 7.ttgir:788:12
	v_mov_b32_e32 v12, 0x44
	v_mov_b32_e32 v13, 0x88
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_xor_b32_e32 v150, v5, v10
	.loc	1 788 12                        ; 7.ttgir:788:12
	v_and_b32_e32 v10, 32, v4
	v_cndmask_b32_e64 v12, v12, 0, s[26:27]
	v_cndmask_b32_e64 v13, v13, 0, s[20:21]
	v_or_b32_e32 v11, v1, v10
	v_or_b32_e32 v12, v13, v12
	v_cndmask_b32_e64 v9, v9, 0, s[22:23]
	v_xor_b32_e32 v11, v12, v11
	v_or_b32_e32 v167, v11, v9
	.loc	1 796 12                        ; 7.ttgir:796:12
	v_or_b32_e32 v11, v156, v10
	v_or_b32_e32 v13, v12, v9
	v_xor_b32_e32 v171, v13, v11
	.loc	1 798 12                        ; 7.ttgir:798:12
	v_xor_b32_e32 v11, v12, v11
	v_or_b32_e32 v173, v11, v9
	.loc	1 804 12                        ; 7.ttgir:804:12
	v_or_b32_e32 v11, v155, v10
	.loc	1 812 12                        ; 7.ttgir:812:12
	v_or_b32_e32 v10, v154, v10
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_or_b32_e32 v16, v16, v3
	v_or_b32_e32 v14, v14, v3
	.loc	1 804 12                        ; 7.ttgir:804:12
	v_xor_b32_e32 v175, v13, v11
	.loc	1 806 12                        ; 7.ttgir:806:12
	v_xor_b32_e32 v11, v12, v11
	.loc	1 812 12                        ; 7.ttgir:812:12
	v_xor_b32_e32 v166, v13, v10
	.loc	1 814 12                        ; 7.ttgir:814:12
	v_xor_b32_e32 v10, v12, v10
	.loc	1 707 12                        ; 7.ttgir:707:12
	v_xor_b32_e32 v157, 0x80, v6
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_xor_b32_e32 v142, v5, v16
	v_xor_b32_e32 v146, v5, v14
	.loc	1 806 12                        ; 7.ttgir:806:12
	v_xor_b32_e32 v178, v11, v9
	.loc	1 814 12                        ; 7.ttgir:814:12
	v_xor_b32_e32 v176, v10, v9
	s_mov_b64 s[2:3], 0
	s_branch .LBB0_3
.LBB0_2:
	.loc	1 0 12 is_stmt 0                ; 7.ttgir:0:12
	s_mov_b64 s[2:3], -1
                                        ; implicit-def: $vgpr135
                                        ; implicit-def: $vgpr134
                                        ; implicit-def: $vgpr133
                                        ; implicit-def: $vgpr132
                                        ; implicit-def: $vgpr131
                                        ; implicit-def: $vgpr130
                                        ; implicit-def: $vgpr157
                                        ; implicit-def: $vgpr148
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr143
                                        ; implicit-def: $vgpr144
                                        ; implicit-def: $vgpr145
                                        ; implicit-def: $vgpr146
                                        ; implicit-def: $vgpr147
                                        ; implicit-def: $vgpr149
                                        ; implicit-def: $vgpr150
                                        ; implicit-def: $vgpr167
                                        ; implicit-def: $vgpr171
                                        ; implicit-def: $vgpr173
                                        ; implicit-def: $vgpr175
                                        ; implicit-def: $vgpr178
                                        ; implicit-def: $vgpr166
                                        ; implicit-def: $vgpr176
.LBB0_3:                                ; %Flow394
	s_lshl_b32 s38, s44, 5
	.loc	1 298 5 is_stmt 1               ; 7.ttgir:298:5
	s_andn2_b64 vcc, exec, s[2:3]
	v_cmp_eq_u32_e64 s[14:15], 0, v7
	v_cmp_eq_u32_e64 s[12:13], 0, v8
	v_cmp_eq_u32_e64 s[2:3], 0, v2
	s_cbranch_vccnz .LBB0_7
; %bb.4:                                ; %.lr.ph
	.loc	1 212 12                        ; 7.ttgir:212:12
	s_and_b32 s29, s0, 0x7ffffff
	.loc	1 233 13                        ; 7.ttgir:233:13
	s_lshl_b64 s[64:65], s[30:31], 1
	s_add_u32 s48, s28, s64
	s_addc_u32 s37, s40, s65
	.loc	1 236 13                        ; 7.ttgir:236:13
	s_and_b32 s37, s37, 0xffff
	s_or_b32 s49, s37, s53
	s_mov_b32 s51, 0x27000
	s_mov_b32 s50, 0x7ffffffe
	.loc	1 237 13                        ; 7.ttgir:237:13
	buffer_load_dwordx4 v[130:133], v161, s[48:51], 0 offen
	.loc	1 236 13                        ; 7.ttgir:236:13
	buffer_load_dwordx4 v[134:137], v160, s[48:51], 0 offen
	v_mov_b32_e32 v2, 0x110
	v_mov_b32_e32 v8, 0x204
	v_cndmask_b32_e64 v7, v2, 0, s[24:25]
	v_cndmask_b32_e64 v9, v8, 0, s[26:27]
	v_mov_b32_e32 v11, 0x408
	v_or_b32_e32 v10, v7, v9
	v_cndmask_b32_e64 v11, v11, 0, s[20:21]
	v_mov_b32_e32 v13, 0x810
	v_or_b32_e32 v12, v10, v11
	v_cndmask_b32_e64 v13, v13, 0, s[22:23]
	v_xor_b32_e32 v14, v12, v13
	v_or_b32_e32 v12, 32, v12
	v_xor_b32_e32 v12, v12, v13
	v_or_b32_e32 v12, v12, v3
	v_xor_b32_e32 v143, v5, v12
	v_or_b32_e32 v12, 0x44, v7
	v_xor_b32_e32 v12, v12, v9
	v_or_b32_e32 v12, v12, v11
	v_xor_b32_e32 v12, v12, v13
	v_or_b32_e32 v12, v12, v3
	v_xor_b32_e32 v144, v5, v12
	v_or_b32_e32 v12, 0x64, v7
	v_xor_b32_e32 v12, v12, v9
	v_or_b32_e32 v12, v12, v11
	v_xor_b32_e32 v12, v12, v13
	v_or_b32_e32 v12, v12, v3
	v_xor_b32_e32 v145, v5, v12
	v_or_b32_e32 v12, 0x88, v10
	v_or_b32_e32 v11, v13, v11
	v_or_b32_e32 v10, 0xa8, v10
	v_xor_b32_e32 v10, v11, v10
	v_or_b32_e32 v10, v10, v3
	v_xor_b32_e32 v147, v5, v10
	v_or_b32_e32 v10, 0xcc, v7
	v_or_b32_e32 v9, v11, v9
	v_or_b32_e32 v7, 0xec, v7
	v_xor_b32_e32 v12, v11, v12
	v_xor_b32_e32 v10, v9, v10
	v_xor_b32_e32 v7, v9, v7
	v_or_b32_e32 v14, v14, v3
	v_or_b32_e32 v12, v12, v3
	v_or_b32_e32 v10, v10, v3
	v_or_b32_e32 v3, v7, v3
	v_xor_b32_e32 v150, v5, v3
	v_mov_b32_e32 v3, 0x88
	v_xor_b32_e32 v142, v5, v14
	v_xor_b32_e32 v146, v5, v12
	v_xor_b32_e32 v149, v5, v10
	v_xor_b32_e32 v157, 0x80, v6
	v_cndmask_b32_e64 v5, v3, 0, s[14:15]
	v_cndmask_b32_e64 v6, v2, 0, s[12:13]
	v_or_b32_e32 v5, v5, v6
	v_mov_b32_e32 v6, 0x220
	v_mov_b32_e32 v7, 0x440
	v_cndmask_b32_e64 v6, v6, 0, s[2:3]
	v_cndmask_b32_e64 v148, v7, 0, s[82:83]
	v_or3_b32 v5, v5, v6, v148
	v_mov_b32_e32 v6, 0x44
	v_xor_b32_e32 v5, v5, v138
	v_and_b32_e32 v4, 32, v4
	v_cndmask_b32_e64 v6, v6, 0, s[26:27]
	v_cndmask_b32_e64 v3, v3, 0, s[20:21]
	v_lshl_add_u32 v196, v5, 1, 0
	v_or_b32_e32 v5, v1, v4
	v_or_b32_e32 v3, v3, v6
	s_ashr_i32 s39, s38, 31
	v_cndmask_b32_e64 v2, v2, 0, s[22:23]
	v_xor_b32_e32 v5, v3, v5
	v_cndmask_b32_e64 v7, v8, 0, s[14:15]
	v_or_b32_e32 v167, v5, v2
	.loc	1 298 5                         ; 7.ttgir:298:5
	s_add_u32 s20, s62, s42
	v_xor_b32_e32 v5, v167, v7
	s_addc_u32 s21, s63, s43
	v_xor_b32_e32 v6, 0x408, v5
	s_add_u32 s22, s20, s60
	v_lshl_add_u32 v197, v5, 1, 0
	v_lshl_add_u32 v198, v6, 1, 0
	v_xor_b32_e32 v6, 0x810, v5
	v_xor_b32_e32 v5, 0xc18, v5
	s_addc_u32 s23, s21, s61
	s_lshl_b64 s[20:21], s[38:39], 1
	v_lshl_add_u32 v199, v6, 1, 0
	v_lshl_add_u32 v200, v5, 1, 0
	v_or_b32_e32 v5, v156, v4
	v_or_b32_e32 v6, v3, v2
	s_add_u32 s22, s22, s20
	v_xor_b32_e32 v171, v6, v5
	v_xor_b32_e32 v5, v3, v5
	s_addc_u32 s23, s23, s21
	v_or_b32_e32 v173, v5, v2
	s_add_u32 s22, s6, s22
	v_xor_b32_e32 v8, v171, v7
	v_xor_b32_e32 v5, v173, v7
	s_addc_u32 s23, s7, s23
	s_lshl_b64 s[24:25], s[30:31], 2
	v_lshl_add_u32 v202, v8, 1, 0
	v_xor_b32_e32 v8, 0x408, v5
	s_add_u32 s24, s24, s58
	v_lshl_add_u32 v203, v8, 1, 0
	v_xor_b32_e32 v8, 0x810, v5
	v_xor_b32_e32 v5, 0xc18, v5
	s_addc_u32 s25, s25, s59
	v_lshl_add_u32 v205, v5, 1, 0
	v_or_b32_e32 v5, v155, v4
	s_add_u32 s24, s24, s56
	v_xor_b32_e32 v175, v6, v5
	v_xor_b32_e32 v5, v3, v5
	v_or_b32_e32 v4, v154, v4
	s_addc_u32 s25, s25, s57
	v_xor_b32_e32 v178, v5, v2
	v_xor_b32_e32 v3, v3, v4
	s_add_u32 s24, s24, s34
	v_lshl_add_u32 v204, v8, 1, 0
	v_xor_b32_e32 v8, v175, v7
	v_xor_b32_e32 v5, v178, v7
	v_xor_b32_e32 v176, v3, v2
	s_addc_u32 s25, s25, s35
	v_lshl_add_u32 v206, v8, 1, 0
	v_xor_b32_e32 v8, 0x408, v5
	v_xor_b32_e32 v2, v176, v7
	s_add_u32 s24, s4, s24
	v_lshl_add_u32 v207, v8, 1, 0
	v_xor_b32_e32 v8, 0x810, v5
	v_xor_b32_e32 v5, 0xc18, v5
	v_xor_b32_e32 v166, v6, v4
	v_xor_b32_e32 v3, 0x408, v2
	s_addc_u32 s25, s5, s25
	s_min_u32 s26, s29, 2
	s_not_b32 s27, s29
	v_lshl_add_u32 v209, v5, 1, 0
	v_xor_b32_e32 v5, v166, v7
	v_lshl_add_u32 v211, v3, 1, 0
	v_xor_b32_e32 v3, 0x810, v2
	v_xor_b32_e32 v2, 0xc18, v2
	s_add_i32 s26, s27, s26
	v_mov_b32_e32 v50, 0
	.loc	1 222 12                        ; 7.ttgir:222:12
	s_waitcnt vmcnt(2)
	v_lshrrev_b32_e32 v187, 16, v118
	v_lshl_add_u32 v181, v165, 1, 0
	v_lshl_add_u32 v182, v168, 1, 0
	v_lshl_add_u32 v183, v169, 1, 0
	v_lshl_add_u32 v184, v170, 1, 0
	v_lshl_add_u32 v185, v172, 1, 0
	v_lshl_add_u32 v186, v174, 1, 0
	v_lshl_add_u32 v188, v142, 1, 0
	v_lshl_add_u32 v189, v143, 1, 0
	v_lshl_add_u32 v190, v144, 1, 0
	v_lshl_add_u32 v191, v145, 1, 0
	v_lshl_add_u32 v192, v146, 1, 0
	v_lshl_add_u32 v193, v147, 1, 0
	v_lshl_add_u32 v194, v149, 1, 0
	v_lshl_add_u32 v195, v150, 1, 0
	v_lshl_add_u32 v208, v8, 1, 0
	v_lshl_add_u32 v210, v5, 1, 0
	v_lshl_add_u32 v212, v3, 1, 0
	v_lshl_add_u32 v213, v2, 1, 0
	v_mov_b32_e32 v201, 1.0
	v_mov_b32_e32 v180, 0xff800000
	v_mov_b32_e32 v214, s26
	s_mov_b32 s26, 0x5040100
	s_mov_b32 s27, 0x7060302
	s_mov_b32 s29, 0x3e0293ee
	v_mov_b32_e32 v51, v50
	v_mov_b32_e32 v52, v50
	v_mov_b32_e32 v53, v50
	v_mov_b32_e32 v54, v50
	v_mov_b32_e32 v55, v50
	v_mov_b32_e32 v56, v50
	v_mov_b32_e32 v57, v50
	v_mov_b32_e32 v58, v50
	v_mov_b32_e32 v59, v50
	v_mov_b32_e32 v60, v50
	v_mov_b32_e32 v61, v50
	v_mov_b32_e32 v62, v50
	v_mov_b32_e32 v63, v50
	v_mov_b32_e32 v64, v50
	v_mov_b32_e32 v65, v50
	v_mov_b32_e32 v34, v50
	v_mov_b32_e32 v35, v50
	v_mov_b32_e32 v36, v50
	v_mov_b32_e32 v37, v50
	v_mov_b32_e32 v38, v50
	v_mov_b32_e32 v39, v50
	v_mov_b32_e32 v40, v50
	v_mov_b32_e32 v41, v50
	v_mov_b32_e32 v42, v50
	v_mov_b32_e32 v43, v50
	v_mov_b32_e32 v44, v50
	v_mov_b32_e32 v45, v50
	v_mov_b32_e32 v46, v50
	v_mov_b32_e32 v47, v50
	v_mov_b32_e32 v48, v50
	v_mov_b32_e32 v49, v50
	v_mov_b32_e32 v18, v50
	v_mov_b32_e32 v19, v50
	v_mov_b32_e32 v20, v50
	v_mov_b32_e32 v21, v50
	v_mov_b32_e32 v22, v50
	v_mov_b32_e32 v23, v50
	v_mov_b32_e32 v24, v50
	v_mov_b32_e32 v25, v50
	v_mov_b32_e32 v26, v50
	v_mov_b32_e32 v27, v50
	v_mov_b32_e32 v28, v50
	v_mov_b32_e32 v29, v50
	v_mov_b32_e32 v30, v50
	v_mov_b32_e32 v31, v50
	v_mov_b32_e32 v32, v50
	v_mov_b32_e32 v33, v50
	v_mov_b32_e32 v2, v50
	v_mov_b32_e32 v3, v50
	v_mov_b32_e32 v4, v50
	v_mov_b32_e32 v5, v50
	v_mov_b32_e32 v6, v50
	v_mov_b32_e32 v7, v50
	v_mov_b32_e32 v8, v50
	v_mov_b32_e32 v9, v50
	v_mov_b32_e32 v10, v50
	v_mov_b32_e32 v11, v50
	v_mov_b32_e32 v12, v50
	v_mov_b32_e32 v13, v50
	v_mov_b32_e32 v14, v50
	v_mov_b32_e32 v15, v50
	v_mov_b32_e32 v16, v50
	v_mov_b32_e32 v17, v50
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 320 12                        ; 7.ttgir:320:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[110:111], 0
	v_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[112:113], v[66:81]
	; sched_barrier mask(0x00000406)
	.loc	1 319 12                        ; 7.ttgir:319:12
	ds_read_b128 v[126:129], v181
	; sched_barrier mask(0x00000406)
	.loc	1 322 12                        ; 7.ttgir:322:12
	ds_read_b128 v[216:219], v182
	.loc	1 324 12                        ; 7.ttgir:324:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[106:107], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[108:109], v[66:81]
	.loc	1 325 12                        ; 7.ttgir:325:12
	ds_read_b128 v[122:125], v183
	; sched_barrier mask(0x00000406)
	.loc	1 327 12                        ; 7.ttgir:327:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[104:105], v[66:81]
	.loc	1 328 12                        ; 7.ttgir:328:12
	ds_read_b128 v[126:129], v184
	; sched_barrier mask(0x00000406)
	.loc	1 330 12                        ; 7.ttgir:330:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[66:81], v[216:217], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[218:219], v[100:101], v[66:81]
	.loc	1 331 12                        ; 7.ttgir:331:12
	ds_read_b128 v[216:219], v185
	; sched_barrier mask(0x00000406)
	.loc	1 333 12                        ; 7.ttgir:333:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[94:95], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[96:97], v[66:81]
	.loc	1 334 12                        ; 7.ttgir:334:12
	ds_read_b128 v[122:125], v186
	; sched_barrier mask(0x00000406)
	.loc	1 336 12                        ; 7.ttgir:336:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[90:91], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[92:93], v[66:81]
	; sched_barrier mask(0x00000406)
	.loc	1 338 12                        ; 7.ttgir:338:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[216:217], v[86:87], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[218:219], v[88:89], v[66:81]
	; sched_barrier mask(0x00000406)
	.loc	1 340 12                        ; 7.ttgir:340:12
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[82:83], v[66:81]
	v_mov_b32_e32 v122, v201
	v_mov_b32_e32 v123, v180
	v_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[84:85], v[66:81]
	; sched_barrier mask(0x00000000)
	.loc	1 342 5                         ; 7.ttgir:342:5
	s_setprio 0
	.loc	1 354 5                         ; 7.ttgir:354:5
	s_waitcnt vmcnt(2)
	v_perm_b32 v118, v118, v114, s26
	v_alignbit_b32 v114, v187, v114, 16
	.loc	1 347 5                         ; 7.ttgir:347:5
	s_barrier
	.loc	1 354 5                         ; 7.ttgir:354:5
	ds_write_b32 v188, v118 offset:8192
	ds_write_b32 v189, v114 offset:8192
	v_perm_b32 v114, v119, v115, s26
	ds_write_b32 v190, v114 offset:8192
	v_perm_b32 v114, v119, v115, s27
	ds_write_b32 v191, v114 offset:8192
	v_perm_b32 v114, v120, v116, s26
	ds_write_b32 v192, v114 offset:8192
	v_perm_b32 v114, v120, v116, s27
	ds_write_b32 v193, v114 offset:8192
	v_perm_b32 v114, v121, v117, s26
	ds_write_b32 v194, v114 offset:8192
	v_perm_b32 v114, v121, v117, s27
	.loc	1 359 13                        ; 7.ttgir:359:13
	s_and_b32 s37, s23, 0xffff
	.loc	1 354 5                         ; 7.ttgir:354:5
	ds_write_b32 v195, v114 offset:8192
	.loc	1 359 13                        ; 7.ttgir:359:13
	s_or_b32 s49, s37, s55
	s_mov_b32 s48, s22
	; sched_barrier mask(0x0000040F)
	buffer_load_dwordx4 v[118:121], v164, s[48:51], 0 offen
	buffer_load_dwordx4 v[114:117], v163, s[48:51], 0 offen
	.loc	1 379 12                        ; 7.ttgir:379:12
	v_fma_f32 v66, v66, s29, 0
	v_fma_f32 v67, v67, s29, 0
	v_fma_f32 v68, v68, s29, 0
	v_fma_f32 v69, v69, s29, 0
	.loc	1 389 15                        ; 7.ttgir:389:15
	v_max_f32_e32 v124, v66, v67
	.loc	1 381 12                        ; 7.ttgir:381:12
	v_fma_f32 v70, v70, s29, 0
	v_fma_f32 v71, v71, s29, 0
	.loc	1 389 15                        ; 7.ttgir:389:15
	v_max3_f32 v124, v124, v68, v69
	.loc	1 381 12                        ; 7.ttgir:381:12
	v_fma_f32 v72, v72, s29, 0
	v_fma_f32 v73, v73, s29, 0
	.loc	1 389 15                        ; 7.ttgir:389:15
	v_max3_f32 v124, v124, v70, v71
	.loc	1 383 12                        ; 7.ttgir:383:12
	v_fma_f32 v74, v74, s29, 0
	v_fma_f32 v75, v75, s29, 0
	.loc	1 389 15                        ; 7.ttgir:389:15
	v_max3_f32 v124, v124, v72, v73
	.loc	1 383 12                        ; 7.ttgir:383:12
	v_fma_f32 v76, v76, s29, 0
	v_fma_f32 v77, v77, s29, 0
	.loc	1 389 15                        ; 7.ttgir:389:15
	v_max3_f32 v124, v124, v74, v75
	.loc	1 385 12                        ; 7.ttgir:385:12
	v_fma_f32 v78, v78, s29, 0
	v_fma_f32 v79, v79, s29, 0
	.loc	1 389 15                        ; 7.ttgir:389:15
	v_max3_f32 v124, v124, v76, v77
	.loc	1 385 12                        ; 7.ttgir:385:12
	v_fma_f32 v80, v80, s29, 0
	v_fma_f32 v81, v81, s29, 0
	.loc	1 389 15                        ; 7.ttgir:389:15
	v_max3_f32 v124, v124, v78, v79
	v_max3_f32 v124, v124, v80, v81
	; sched_barrier mask(0x0000040F)
	.loc	1 387 12                        ; 7.ttgir:387:12
	ds_bpermute_b32 v125, v157, v124
	.loc	1 392 12                        ; 7.ttgir:392:12
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v180, v123, v124, v125
	.loc	1 359 13                        ; 7.ttgir:359:13
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v187, 16, v118
	; sched_barrier mask(0x00000000)
	.loc	1 416 14                        ; 7.ttgir:416:14
	s_and_b32 s37, s25, 0xffff
	s_mov_b32 s48, s24
	s_or_b32 s49, s37, s53
	.loc	1 407 5                         ; 7.ttgir:407:5
	ds_write_b128 v196, v[134:137]
	; sched_barrier mask(0x0000040F)
	.loc	1 416 14                        ; 7.ttgir:416:14
	buffer_load_dwordx4 v[134:137], v160, s[48:51], 0 offen
	; sched_barrier mask(0x0000040F)
	.loc	1 420 5                         ; 7.ttgir:420:5
	ds_write_b128 v196, v[130:133] offset:4096
	; sched_barrier mask(0x0000040F)
	.loc	1 423 14                        ; 7.ttgir:423:14
	buffer_load_dwordx4 v[130:133], v161, s[48:51], 0 offen
	.loc	1 426 12                        ; 7.ttgir:426:12
	v_sub_f32_e32 v66, v66, v180
	v_sub_f32_e32 v67, v67, v180
	v_sub_f32_e32 v68, v68, v180
	.loc	1 427 12                        ; 7.ttgir:427:12
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	.loc	1 426 12                        ; 7.ttgir:426:12
	v_sub_f32_e32 v69, v69, v180
	.loc	1 427 12                        ; 7.ttgir:427:12
	v_exp_f32_e32 v68, v68
	.loc	1 429 12                        ; 7.ttgir:429:12
	v_sub_f32_e32 v70, v70, v180
	.loc	1 427 12                        ; 7.ttgir:427:12
	v_exp_f32_e32 v69, v69
	.loc	1 429 12                        ; 7.ttgir:429:12
	v_sub_f32_e32 v71, v71, v180
	.loc	1 430 12                        ; 7.ttgir:430:12
	v_exp_f32_e32 v70, v70
	.loc	1 429 12                        ; 7.ttgir:429:12
	v_sub_f32_e32 v72, v72, v180
	.loc	1 430 12                        ; 7.ttgir:430:12
	v_exp_f32_e32 v71, v71
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v66, v67
	.loc	1 429 12                        ; 7.ttgir:429:12
	v_sub_f32_e32 v73, v73, v180
	.loc	1 430 12                        ; 7.ttgir:430:12
	v_exp_f32_e32 v72, v72
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v68, v124
	.loc	1 432 12                        ; 7.ttgir:432:12
	v_sub_f32_e32 v74, v74, v180
	.loc	1 430 12                        ; 7.ttgir:430:12
	v_exp_f32_e32 v73, v73
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v69, v124
	.loc	1 432 12                        ; 7.ttgir:432:12
	v_sub_f32_e32 v75, v75, v180
	.loc	1 433 12                        ; 7.ttgir:433:12
	v_exp_f32_e32 v74, v74
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v70, v124
	.loc	1 432 12                        ; 7.ttgir:432:12
	v_sub_f32_e32 v76, v76, v180
	.loc	1 433 12                        ; 7.ttgir:433:12
	v_exp_f32_e32 v75, v75
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v71, v124
	.loc	1 432 12                        ; 7.ttgir:432:12
	v_sub_f32_e32 v77, v77, v180
	.loc	1 433 12                        ; 7.ttgir:433:12
	v_exp_f32_e32 v76, v76
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v72, v124
	.loc	1 435 12                        ; 7.ttgir:435:12
	v_sub_f32_e32 v78, v78, v180
	.loc	1 433 12                        ; 7.ttgir:433:12
	v_exp_f32_e32 v77, v77
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v73, v124
	.loc	1 435 12                        ; 7.ttgir:435:12
	v_sub_f32_e32 v79, v79, v180
	.loc	1 436 12                        ; 7.ttgir:436:12
	v_exp_f32_e32 v78, v78
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v74, v124
	.loc	1 435 12                        ; 7.ttgir:435:12
	v_sub_f32_e32 v80, v80, v180
	.loc	1 436 12                        ; 7.ttgir:436:12
	v_exp_f32_e32 v79, v79
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v75, v124
	.loc	1 435 12                        ; 7.ttgir:435:12
	v_sub_f32_e32 v81, v81, v180
	.loc	1 436 12                        ; 7.ttgir:436:12
	v_exp_f32_e32 v80, v80
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v76, v124
	.loc	1 436 12                        ; 7.ttgir:436:12
	v_exp_f32_e32 v81, v81
	.loc	1 440 15                        ; 7.ttgir:440:15
	v_add_f32_e32 v124, v77, v124
	v_add_f32_e32 v124, v78, v124
	v_add_f32_e32 v124, v79, v124
	v_add_f32_e32 v124, v80, v124
	v_add_f32_e32 v124, v81, v124
	; sched_barrier mask(0x0000040F)
	.loc	1 438 12                        ; 7.ttgir:438:12
	ds_bpermute_b32 v125, v157, v124
	.loc	1 443 12                        ; 7.ttgir:443:12
	v_sub_f32_e32 v123, v123, v180
	.loc	1 444 12                        ; 7.ttgir:444:12
	v_exp_f32_e32 v123, v123
	.loc	1 440 15                        ; 7.ttgir:440:15
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v201, v124, v125
	; sched_barrier mask(0x00000000)
	.loc	1 483 12                        ; 7.ttgir:483:12
	v_cvt_f16_f32_e32 v124, v66
	v_cvt_f16_f32_e32 v125, v67
	v_cvt_f16_f32_e32 v126, v68
	v_cvt_f16_f32_e32 v127, v69
	.loc	1 484 12                        ; 7.ttgir:484:12
	v_cvt_f16_f32_e32 v128, v70
	v_cvt_f16_f32_e32 v129, v71
	.loc	1 488 5                         ; 7.ttgir:488:5
	s_barrier
	.loc	1 496 12                        ; 7.ttgir:496:12
	ds_read_b64 v[66:67], v197 offset:8192
	.loc	1 455 12                        ; 7.ttgir:455:12
	v_mul_f32_e32 v50, v50, v123
	v_mul_f32_e32 v51, v51, v123
	v_mul_f32_e32 v52, v52, v123
	v_mul_f32_e32 v53, v53, v123
	.loc	1 458 12                        ; 7.ttgir:458:12
	v_mul_f32_e32 v54, v54, v123
	v_mul_f32_e32 v55, v55, v123
	v_mul_f32_e32 v56, v56, v123
	v_mul_f32_e32 v57, v57, v123
	.loc	1 461 12                        ; 7.ttgir:461:12
	v_mul_f32_e32 v58, v58, v123
	v_mul_f32_e32 v59, v59, v123
	v_mul_f32_e32 v60, v60, v123
	v_mul_f32_e32 v61, v61, v123
	.loc	1 464 12                        ; 7.ttgir:464:12
	v_mul_f32_e32 v62, v62, v123
	v_mul_f32_e32 v63, v63, v123
	v_mul_f32_e32 v64, v64, v123
	v_mul_f32_e32 v65, v65, v123
	.loc	1 484 12                        ; 7.ttgir:484:12
	v_cvt_f16_f32_e32 v215, v72
	v_cvt_f16_f32_e32 v216, v73
	; sched_barrier mask(0x00000000)
	.loc	1 504 12                        ; 7.ttgir:504:12
	ds_read_b64 v[68:69], v198 offset:8192
	.loc	1 498 12                        ; 7.ttgir:498:12
	v_mul_f32_e32 v34, v34, v123
	v_mul_f32_e32 v35, v35, v123
	v_mul_f32_e32 v36, v36, v123
	v_mul_f32_e32 v37, v37, v123
	.loc	1 501 12                        ; 7.ttgir:501:12
	v_mul_f32_e32 v38, v38, v123
	v_mul_f32_e32 v39, v39, v123
	v_mul_f32_e32 v40, v40, v123
	v_mul_f32_e32 v41, v41, v123
	; sched_barrier mask(0x00000000)
	.loc	1 512 12                        ; 7.ttgir:512:12
	ds_read_b64 v[70:71], v199 offset:8192
	.loc	1 506 12                        ; 7.ttgir:506:12
	v_mul_f32_e32 v42, v42, v123
	v_mul_f32_e32 v43, v43, v123
	v_mul_f32_e32 v44, v44, v123
	v_mul_f32_e32 v45, v45, v123
	.loc	1 509 12                        ; 7.ttgir:509:12
	v_mul_f32_e32 v46, v46, v123
	v_mul_f32_e32 v47, v47, v123
	v_mul_f32_e32 v48, v48, v123
	v_mul_f32_e32 v49, v49, v123
	; sched_barrier mask(0x00000000)
	.loc	1 516 12                        ; 7.ttgir:516:12
	v_mul_f32_e32 v18, v18, v123
	v_mul_f32_e32 v19, v19, v123
	v_mul_f32_e32 v20, v20, v123
	v_mul_f32_e32 v21, v21, v123
	.loc	1 519 12                        ; 7.ttgir:519:12
	v_mul_f32_e32 v22, v22, v123
	v_mul_f32_e32 v23, v23, v123
	v_mul_f32_e32 v24, v24, v123
	v_mul_f32_e32 v25, v25, v123
	.loc	1 522 12                        ; 7.ttgir:522:12
	v_mul_f32_e32 v26, v26, v123
	v_mul_f32_e32 v27, v27, v123
	v_mul_f32_e32 v28, v28, v123
	v_mul_f32_e32 v29, v29, v123
	.loc	1 525 12                        ; 7.ttgir:525:12
	v_mul_f32_e32 v30, v30, v123
	v_mul_f32_e32 v31, v31, v123
	v_mul_f32_e32 v32, v32, v123
	v_mul_f32_e32 v33, v33, v123
	; sched_barrier mask(0x00000000)
	.loc	1 535 12                        ; 7.ttgir:535:12
	v_mul_f32_e32 v2, v2, v123
	v_mul_f32_e32 v3, v3, v123
	v_mul_f32_e32 v4, v4, v123
	v_mul_f32_e32 v5, v5, v123
	.loc	1 538 12                        ; 7.ttgir:538:12
	v_mul_f32_e32 v6, v6, v123
	v_mul_f32_e32 v7, v7, v123
	v_mul_f32_e32 v8, v8, v123
	v_mul_f32_e32 v9, v9, v123
	.loc	1 541 12                        ; 7.ttgir:541:12
	v_mul_f32_e32 v10, v10, v123
	v_mul_f32_e32 v11, v11, v123
	v_mul_f32_e32 v12, v12, v123
	v_mul_f32_e32 v13, v13, v123
	.loc	1 544 12                        ; 7.ttgir:544:12
	v_mul_f32_e32 v14, v14, v123
	v_mul_f32_e32 v15, v15, v123
	v_mul_f32_e32 v16, v16, v123
	v_mul_f32_e32 v17, v17, v123
	; sched_barrier mask(0x00000000)
	.loc	1 553 12                        ; 7.ttgir:553:12
	v_fmac_f32_e32 v201, v122, v123
	.loc	1 548 12                        ; 7.ttgir:548:12
	v_cvt_f16_f32_e32 v217, v74
	v_cvt_f16_f32_e32 v218, v75
	v_cvt_f16_f32_e32 v76, v76
	v_cvt_f16_f32_e32 v77, v77
	.loc	1 549 12                        ; 7.ttgir:549:12
	v_cvt_f16_f32_e32 v78, v78
	v_cvt_f16_f32_e32 v79, v79
	v_cvt_f16_f32_e32 v80, v80
	v_cvt_f16_f32_e32 v81, v81
	; sched_barrier mask(0x00000000)
	.loc	1 558 5                         ; 7.ttgir:558:5
	s_setprio 3
	; sched_barrier mask(0x00000000)
	.loc	1 561 12                        ; 7.ttgir:561:12
	v_pack_b32_f16 v73, v126, v127
	v_pack_b32_f16 v72, v124, v125
	.loc	1 297 12                        ; 7.ttgir:297:12
	s_add_u32 s22, s22, s20
	s_addc_u32 s23, s23, s21
	.loc	1 561 12                        ; 7.ttgir:561:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[50:65], v[66:67], v[72:73], v[50:65]
	.loc	1 562 12                        ; 7.ttgir:562:12
	ds_read_b64 v[66:67], v200 offset:8192
	; sched_barrier mask(0x00000406)
	.loc	1 297 12                        ; 7.ttgir:297:12
	s_add_u32 s24, s24, s64
	v_add_co_u32_e32 v214, vcc, 1, v214
	s_addc_u32 s25, s25, s65
	.loc	1 298 5                         ; 7.ttgir:298:5
	s_andn2_b64 vcc, exec, vcc
	.loc	1 564 12                        ; 7.ttgir:564:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[34:49], v[68:69], v[72:73], v[34:49]
	.loc	1 565 12                        ; 7.ttgir:565:12
	ds_read_b64 v[68:69], v202 offset:8192
	; sched_barrier mask(0x00000406)
	.loc	1 570 12                        ; 7.ttgir:570:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[72:73], v[2:17]
	.loc	1 573 12                        ; 7.ttgir:573:12
	v_pack_b32_f16 v67, v215, v216
	v_pack_b32_f16 v66, v128, v129
	.loc	1 567 12                        ; 7.ttgir:567:12
	v_mfma_f32_32x32x8_f16 v[18:33], v[70:71], v[72:73], v[18:33]
	.loc	1 568 12                        ; 7.ttgir:568:12
	ds_read_b64 v[70:71], v203 offset:8192
	; sched_barrier mask(0x00000406)
	.loc	1 571 12                        ; 7.ttgir:571:12
	ds_read_b64 v[74:75], v204 offset:8192
	; sched_barrier mask(0x00000406)
	.loc	1 573 12                        ; 7.ttgir:573:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[66:67], v[50:65]
	.loc	1 574 12                        ; 7.ttgir:574:12
	ds_read_b64 v[68:69], v205 offset:8192
	; sched_barrier mask(0x00000406)
	.loc	1 576 12                        ; 7.ttgir:576:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[66:67], v[34:49]
	.loc	1 577 12                        ; 7.ttgir:577:12
	ds_read_b64 v[70:71], v206 offset:8192
	; sched_barrier mask(0x00000406)
	.loc	1 579 12                        ; 7.ttgir:579:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[66:67], v[18:33]
	.loc	1 580 12                        ; 7.ttgir:580:12
	ds_read_b64 v[72:73], v207 offset:8192
	; sched_barrier mask(0x00000406)
	.loc	1 583 12                        ; 7.ttgir:583:12
	ds_read_b64 v[74:75], v208 offset:8192
	.loc	1 582 12                        ; 7.ttgir:582:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[66:67], v[2:17]
	; sched_barrier mask(0x00000406)
	.loc	1 586 12                        ; 7.ttgir:586:12
	ds_read_b64 v[68:69], v209 offset:8192
	.loc	1 585 12                        ; 7.ttgir:585:12
	v_pack_b32_f16 v67, v76, v77
	v_pack_b32_f16 v66, v217, v218
	s_waitcnt lgkmcnt(3)
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[50:65], v[70:71], v[66:67], v[50:65]
	; sched_barrier mask(0x00000406)
	.loc	1 589 12                        ; 7.ttgir:589:12
	ds_read_b64 v[70:71], v210 offset:8192
	.loc	1 588 12                        ; 7.ttgir:588:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[34:49], v[72:73], v[66:67], v[34:49]
	; sched_barrier mask(0x00000406)
	.loc	1 591 12                        ; 7.ttgir:591:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[66:67], v[18:33]
	.loc	1 592 12                        ; 7.ttgir:592:12
	ds_read_b64 v[72:73], v211 offset:8192
	; sched_barrier mask(0x00000406)
	.loc	1 595 12                        ; 7.ttgir:595:12
	ds_read_b64 v[74:75], v212 offset:8192
	.loc	1 594 12                        ; 7.ttgir:594:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[66:67], v[2:17]
	.loc	1 596 12                        ; 7.ttgir:596:12
	ds_read_b64 v[66:67], v213 offset:8192
	.loc	1 599 12                        ; 7.ttgir:599:12
	v_pack_b32_f16 v69, v80, v81
	v_pack_b32_f16 v68, v78, v79
	; sched_barrier mask(0x00000406)
	.loc	1 601 13                        ; 7.ttgir:601:13
	ds_read_b128 v[126:129], v177
	.loc	1 599 12                        ; 7.ttgir:599:12
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[50:65], v[70:71], v[68:69], v[50:65]
	; sched_barrier mask(0x00000406)
	.loc	1 606 13                        ; 7.ttgir:606:13
	ds_read_b128 v[122:125], v179
	.loc	1 604 12                        ; 7.ttgir:604:12
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[34:49], v[72:73], v[68:69], v[34:49]
	; sched_barrier mask(0x00000406)
	.loc	1 609 12                        ; 7.ttgir:609:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[68:69], v[18:33]
	.loc	1 610 12                        ; 7.ttgir:610:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[68:69], v[2:17]
	.loc	1 298 5                         ; 7.ttgir:298:5
	s_cbranch_vccnz .LBB0_5
; %bb.6:                                ; %._crit_edge.loopexit
	.loc	1 0 5 is_stmt 0                 ; 7.ttgir:0:5
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v135, v165
	v_mov_b32_e32 v134, v168
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v133, v169
	v_mov_b32_e32 v132, v170
	v_mov_b32_e32 v131, v172
	v_mov_b32_e32 v130, v174
	s_branch .LBB0_8
.LBB0_7:
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v201, 1.0
	v_mov_b32_e32 v180, 0xff800000
	v_mov_b32_e32 v16, v17
	v_mov_b32_e32 v15, v17
	v_mov_b32_e32 v14, v17
	v_mov_b32_e32 v13, v17
	v_mov_b32_e32 v12, v17
	v_mov_b32_e32 v11, v17
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v9, v17
	v_mov_b32_e32 v8, v17
	v_mov_b32_e32 v7, v17
	v_mov_b32_e32 v6, v17
	v_mov_b32_e32 v5, v17
	v_mov_b32_e32 v4, v17
	v_mov_b32_e32 v3, v17
	v_mov_b32_e32 v2, v17
	v_mov_b32_e32 v33, v17
	v_mov_b32_e32 v32, v17
	v_mov_b32_e32 v31, v17
	v_mov_b32_e32 v30, v17
	v_mov_b32_e32 v29, v17
	v_mov_b32_e32 v28, v17
	v_mov_b32_e32 v27, v17
	v_mov_b32_e32 v26, v17
	v_mov_b32_e32 v25, v17
	v_mov_b32_e32 v24, v17
	v_mov_b32_e32 v23, v17
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v21, v17
	v_mov_b32_e32 v20, v17
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v18, v17
	v_mov_b32_e32 v49, v17
	v_mov_b32_e32 v48, v17
	v_mov_b32_e32 v47, v17
	v_mov_b32_e32 v46, v17
	v_mov_b32_e32 v45, v17
	v_mov_b32_e32 v44, v17
	v_mov_b32_e32 v43, v17
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v41, v17
	v_mov_b32_e32 v40, v17
	v_mov_b32_e32 v39, v17
	v_mov_b32_e32 v38, v17
	v_mov_b32_e32 v37, v17
	v_mov_b32_e32 v36, v17
	v_mov_b32_e32 v35, v17
	v_mov_b32_e32 v34, v17
	v_mov_b32_e32 v65, v17
	v_mov_b32_e32 v64, v17
	v_mov_b32_e32 v63, v17
	v_mov_b32_e32 v62, v17
	v_mov_b32_e32 v61, v17
	v_mov_b32_e32 v60, v17
	v_mov_b32_e32 v59, v17
	v_mov_b32_e32 v58, v17
	v_mov_b32_e32 v57, v17
	v_mov_b32_e32 v56, v17
	v_mov_b32_e32 v55, v17
	v_mov_b32_e32 v54, v17
	v_mov_b32_e32 v53, v17
	v_mov_b32_e32 v52, v17
	v_mov_b32_e32 v51, v17
	v_mov_b32_e32 v50, v17
.LBB0_8:                                ; %._crit_edge
	.loc	1 101 10 is_stmt 1              ; 7.ttgir:101:10
	v_or3_b32 v165, v151, v153, v152
	.loc	1 110 11                        ; 7.ttgir:110:11
	v_or_b32_e32 v189, 1, v162
	.loc	1 651 5                         ; 7.ttgir:651:5
	s_setprio 3
	.loc	1 677 12                        ; 7.ttgir:677:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[110:111], 0
	.loc	1 658 12                        ; 7.ttgir:658:12
	v_lshl_add_u32 v183, v135, 1, 0
	.loc	1 652 5                         ; 7.ttgir:652:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(1)
	.loc	1 658 12                        ; 7.ttgir:658:12
	ds_read_b128 v[114:117], v183
	.loc	1 660 12                        ; 7.ttgir:660:12
	v_lshl_add_u32 v184, v134, 1, 0
	s_waitcnt vmcnt(0)
	ds_read_b128 v[118:121], v184
	.loc	1 677 12                        ; 7.ttgir:677:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[112:113], v[66:81]
	.loc	1 662 12                        ; 7.ttgir:662:12
	v_lshl_add_u32 v185, v133, 1, 0
	.loc	1 664 12                        ; 7.ttgir:664:12
	v_lshl_add_u32 v186, v132, 1, 0
	.loc	1 666 12                        ; 7.ttgir:666:12
	v_lshl_add_u32 v187, v131, 1, 0
	.loc	1 668 12                        ; 7.ttgir:668:12
	v_lshl_add_u32 v188, v130, 1, 0
	.loc	1 678 12                        ; 7.ttgir:678:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[106:107], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[108:109], v[66:81]
	.loc	1 679 12                        ; 7.ttgir:679:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[102:103], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[104:105], v[66:81]
	.loc	1 662 12                        ; 7.ttgir:662:12
	ds_read_b128 v[114:117], v185
	.loc	1 680 12                        ; 7.ttgir:680:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[98:99], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[100:101], v[66:81]
	.loc	1 664 12                        ; 7.ttgir:664:12
	ds_read_b128 v[118:121], v186
	.loc	1 681 12                        ; 7.ttgir:681:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[94:95], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[96:97], v[66:81]
	.loc	1 666 12                        ; 7.ttgir:666:12
	ds_read_b128 v[114:117], v187
	.loc	1 682 12                        ; 7.ttgir:682:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[90:91], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[92:93], v[66:81]
	.loc	1 668 12                        ; 7.ttgir:668:12
	ds_read_b128 v[118:121], v188
	.loc	1 683 12                        ; 7.ttgir:683:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[86:87], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[88:89], v[66:81]
	.loc	1 684 12                        ; 7.ttgir:684:12
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[82:83], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[84:85], v[66:81]
	.loc	1 685 5                         ; 7.ttgir:685:5
	s_setprio 0
	s_mov_b32 s20, 0x3e0293ee
	.loc	1 699 12                        ; 7.ttgir:699:12
	s_nop 7
	s_nop 0
	v_fma_f32 v66, v66, s20, 0
	v_fma_f32 v67, v67, s20, 0
	v_fma_f32 v68, v68, s20, 0
	v_fma_f32 v69, v69, s20, 0
	.loc	1 709 15                        ; 7.ttgir:709:15
	v_max_f32_e32 v114, v66, v67
	.loc	1 701 12                        ; 7.ttgir:701:12
	v_fma_f32 v70, v70, s20, 0
	v_fma_f32 v71, v71, s20, 0
	.loc	1 709 15                        ; 7.ttgir:709:15
	v_max3_f32 v114, v114, v68, v69
	.loc	1 701 12                        ; 7.ttgir:701:12
	v_fma_f32 v72, v72, s20, 0
	v_fma_f32 v73, v73, s20, 0
	.loc	1 709 15                        ; 7.ttgir:709:15
	v_max3_f32 v114, v114, v70, v71
	.loc	1 703 12                        ; 7.ttgir:703:12
	v_fma_f32 v74, v74, s20, 0
	v_fma_f32 v75, v75, s20, 0
	.loc	1 709 15                        ; 7.ttgir:709:15
	v_max3_f32 v114, v114, v72, v73
	.loc	1 703 12                        ; 7.ttgir:703:12
	v_fma_f32 v76, v76, s20, 0
	v_fma_f32 v77, v77, s20, 0
	.loc	1 709 15                        ; 7.ttgir:709:15
	v_max3_f32 v114, v114, v74, v75
	.loc	1 705 12                        ; 7.ttgir:705:12
	v_fma_f32 v78, v78, s20, 0
	v_fma_f32 v79, v79, s20, 0
	.loc	1 709 15                        ; 7.ttgir:709:15
	v_max3_f32 v114, v114, v76, v77
	.loc	1 705 12                        ; 7.ttgir:705:12
	v_fma_f32 v80, v80, s20, 0
	v_fma_f32 v81, v81, s20, 0
	.loc	1 709 15                        ; 7.ttgir:709:15
	v_max3_f32 v114, v114, v78, v79
	v_max3_f32 v114, v114, v80, v81
	.loc	1 707 12                        ; 7.ttgir:707:12
	ds_bpermute_b32 v115, v157, v114
	.loc	1 712 12                        ; 7.ttgir:712:12
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v123, v180, v114, v115
	.loc	1 716 12                        ; 7.ttgir:716:12
	v_sub_f32_e32 v66, v66, v123
	v_sub_f32_e32 v67, v67, v123
	v_sub_f32_e32 v68, v68, v123
	.loc	1 723 12                        ; 7.ttgir:723:12
	v_exp_f32_e32 v114, v66
	v_exp_f32_e32 v67, v67
	.loc	1 716 12                        ; 7.ttgir:716:12
	v_sub_f32_e32 v69, v69, v123
	.loc	1 723 12                        ; 7.ttgir:723:12
	v_exp_f32_e32 v115, v68
	.loc	1 718 12                        ; 7.ttgir:718:12
	v_sub_f32_e32 v70, v70, v123
	.loc	1 723 12                        ; 7.ttgir:723:12
	v_exp_f32_e32 v116, v69
	.loc	1 718 12                        ; 7.ttgir:718:12
	v_sub_f32_e32 v71, v71, v123
	.loc	1 724 12                        ; 7.ttgir:724:12
	v_exp_f32_e32 v117, v70
	.loc	1 718 12                        ; 7.ttgir:718:12
	v_sub_f32_e32 v72, v72, v123
	.loc	1 724 12                        ; 7.ttgir:724:12
	v_exp_f32_e32 v118, v71
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v114, v67
	.loc	1 718 12                        ; 7.ttgir:718:12
	v_sub_f32_e32 v73, v73, v123
	.loc	1 724 12                        ; 7.ttgir:724:12
	v_exp_f32_e32 v119, v72
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v115, v66
	.loc	1 720 12                        ; 7.ttgir:720:12
	v_sub_f32_e32 v74, v74, v123
	.loc	1 724 12                        ; 7.ttgir:724:12
	v_exp_f32_e32 v120, v73
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v116, v66
	.loc	1 720 12                        ; 7.ttgir:720:12
	v_sub_f32_e32 v75, v75, v123
	.loc	1 725 12                        ; 7.ttgir:725:12
	v_exp_f32_e32 v121, v74
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v117, v66
	.loc	1 720 12                        ; 7.ttgir:720:12
	v_sub_f32_e32 v76, v76, v123
	.loc	1 725 12                        ; 7.ttgir:725:12
	v_exp_f32_e32 v124, v75
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v118, v66
	.loc	1 720 12                        ; 7.ttgir:720:12
	v_sub_f32_e32 v77, v77, v123
	.loc	1 725 12                        ; 7.ttgir:725:12
	v_exp_f32_e32 v125, v76
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v119, v66
	.loc	1 722 12                        ; 7.ttgir:722:12
	v_sub_f32_e32 v78, v78, v123
	.loc	1 725 12                        ; 7.ttgir:725:12
	v_exp_f32_e32 v126, v77
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v120, v66
	.loc	1 722 12                        ; 7.ttgir:722:12
	v_sub_f32_e32 v79, v79, v123
	.loc	1 726 12                        ; 7.ttgir:726:12
	v_exp_f32_e32 v78, v78
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v121, v66
	.loc	1 722 12                        ; 7.ttgir:722:12
	v_sub_f32_e32 v80, v80, v123
	.loc	1 726 12                        ; 7.ttgir:726:12
	v_exp_f32_e32 v79, v79
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v124, v66
	.loc	1 722 12                        ; 7.ttgir:722:12
	v_sub_f32_e32 v81, v81, v123
	.loc	1 726 12                        ; 7.ttgir:726:12
	v_exp_f32_e32 v80, v80
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v125, v66
	.loc	1 726 12                        ; 7.ttgir:726:12
	v_exp_f32_e32 v81, v81
	.loc	1 730 15                        ; 7.ttgir:730:15
	v_add_f32_e32 v66, v126, v66
	v_add_f32_e32 v66, v78, v66
	v_add_f32_e32 v66, v79, v66
	v_add_f32_e32 v66, v80, v66
	v_add_f32_e32 v68, v81, v66
	.loc	1 728 12                        ; 7.ttgir:728:12
	ds_bpermute_b32 v69, v157, v68
	.loc	1 733 12                        ; 7.ttgir:733:12
	v_sub_f32_e32 v66, v180, v123
	.loc	1 734 12                        ; 7.ttgir:734:12
	v_exp_f32_e32 v66, v66
	.loc	1 730 15                        ; 7.ttgir:730:15
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v122, v68, v69
	.loc	1 737 5                         ; 7.ttgir:737:5
	s_setprio 2
	.loc	1 788 12                        ; 7.ttgir:788:12
	v_mov_b32_e32 v68, 0x204
	v_cndmask_b32_e64 v127, v68, 0, s[14:15]
	v_xor_b32_e32 v68, v167, v127
	.loc	1 790 12                        ; 7.ttgir:790:12
	v_xor_b32_e32 v69, 0x408, v68
	.loc	1 788 12                        ; 7.ttgir:788:12
	v_lshl_add_u32 v168, v68, 1, 0
	.loc	1 790 12                        ; 7.ttgir:790:12
	v_lshl_add_u32 v167, v69, 1, 0
	.loc	1 792 12                        ; 7.ttgir:792:12
	v_xor_b32_e32 v69, 0x810, v68
	.loc	1 794 12                        ; 7.ttgir:794:12
	v_xor_b32_e32 v68, 0xc18, v68
	.loc	1 792 12                        ; 7.ttgir:792:12
	v_lshl_add_u32 v169, v69, 1, 0
	.loc	1 794 12                        ; 7.ttgir:794:12
	v_lshl_add_u32 v170, v68, 1, 0
	.loc	1 788 12                        ; 7.ttgir:788:12
	ds_read_b64 v[68:69], v168 offset:8192
	.loc	1 821 12                        ; 7.ttgir:821:12
	v_cvt_f16_f32_e32 v76, v114
	v_cvt_f16_f32_e32 v67, v67
	v_cvt_f16_f32_e32 v77, v115
	v_cvt_f16_f32_e32 v114, v116
	.loc	1 740 12                        ; 7.ttgir:740:12
	v_mul_f32_e32 v50, v50, v66
	.loc	1 833 12                        ; 7.ttgir:833:12
	v_pack_b32_f16 v76, v76, v67
	.loc	1 796 12                        ; 7.ttgir:796:12
	v_xor_b32_e32 v67, v171, v127
	.loc	1 740 12                        ; 7.ttgir:740:12
	v_mul_f32_e32 v51, v51, v66
	v_mul_f32_e32 v52, v52, v66
	v_mul_f32_e32 v53, v53, v66
	.loc	1 743 12                        ; 7.ttgir:743:12
	v_mul_f32_e32 v54, v54, v66
	v_mul_f32_e32 v55, v55, v66
	v_mul_f32_e32 v56, v56, v66
	v_mul_f32_e32 v57, v57, v66
	.loc	1 746 12                        ; 7.ttgir:746:12
	v_mul_f32_e32 v58, v58, v66
	v_mul_f32_e32 v59, v59, v66
	v_mul_f32_e32 v60, v60, v66
	v_mul_f32_e32 v61, v61, v66
	.loc	1 749 12                        ; 7.ttgir:749:12
	v_mul_f32_e32 v62, v62, v66
	v_mul_f32_e32 v63, v63, v66
	v_mul_f32_e32 v64, v64, v66
	v_mul_f32_e32 v65, v65, v66
	.loc	1 833 12                        ; 7.ttgir:833:12
	v_pack_b32_f16 v77, v77, v114
	.loc	1 796 12                        ; 7.ttgir:796:12
	v_lshl_add_u32 v172, v67, 1, 0
	.loc	1 798 12                        ; 7.ttgir:798:12
	v_xor_b32_e32 v67, v173, v127
	.loc	1 790 12                        ; 7.ttgir:790:12
	ds_read_b64 v[70:71], v167 offset:8192
	.loc	1 792 12                        ; 7.ttgir:792:12
	ds_read_b64 v[72:73], v169 offset:8192
	.loc	1 794 12                        ; 7.ttgir:794:12
	ds_read_b64 v[74:75], v170 offset:8192
	.loc	1 833 12                        ; 7.ttgir:833:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]
	.loc	1 798 12                        ; 7.ttgir:798:12
	v_xor_b32_e32 v68, 0x408, v67
	v_lshl_add_u32 v171, v68, 1, 0
	.loc	1 800 12                        ; 7.ttgir:800:12
	v_xor_b32_e32 v68, 0x810, v67
	v_lshl_add_u32 v173, v68, 1, 0
	.loc	1 796 12                        ; 7.ttgir:796:12
	ds_read_b64 v[68:69], v172 offset:8192
	.loc	1 752 12                        ; 7.ttgir:752:12
	v_mul_f32_e32 v34, v34, v66
	v_mul_f32_e32 v35, v35, v66
	v_mul_f32_e32 v36, v36, v66
	v_mul_f32_e32 v37, v37, v66
	.loc	1 755 12                        ; 7.ttgir:755:12
	v_mul_f32_e32 v38, v38, v66
	v_mul_f32_e32 v39, v39, v66
	v_mul_f32_e32 v40, v40, v66
	v_mul_f32_e32 v41, v41, v66
	.loc	1 758 12                        ; 7.ttgir:758:12
	v_mul_f32_e32 v42, v42, v66
	v_mul_f32_e32 v43, v43, v66
	v_mul_f32_e32 v44, v44, v66
	v_mul_f32_e32 v45, v45, v66
	.loc	1 761 12                        ; 7.ttgir:761:12
	v_mul_f32_e32 v46, v46, v66
	v_mul_f32_e32 v47, v47, v66
	v_mul_f32_e32 v48, v48, v66
	v_mul_f32_e32 v49, v49, v66
	.loc	1 764 12                        ; 7.ttgir:764:12
	v_mul_f32_e32 v18, v18, v66
	v_mul_f32_e32 v19, v19, v66
	v_mul_f32_e32 v20, v20, v66
	v_mul_f32_e32 v21, v21, v66
	.loc	1 767 12                        ; 7.ttgir:767:12
	v_mul_f32_e32 v22, v22, v66
	v_mul_f32_e32 v23, v23, v66
	v_mul_f32_e32 v24, v24, v66
	v_mul_f32_e32 v25, v25, v66
	.loc	1 770 12                        ; 7.ttgir:770:12
	v_mul_f32_e32 v26, v26, v66
	v_mul_f32_e32 v27, v27, v66
	v_mul_f32_e32 v28, v28, v66
	v_mul_f32_e32 v29, v29, v66
	.loc	1 773 12                        ; 7.ttgir:773:12
	v_mul_f32_e32 v30, v30, v66
	v_mul_f32_e32 v31, v31, v66
	v_mul_f32_e32 v32, v32, v66
	v_mul_f32_e32 v33, v33, v66
	.loc	1 776 12                        ; 7.ttgir:776:12
	v_mul_f32_e32 v2, v2, v66
	v_mul_f32_e32 v3, v3, v66
	v_mul_f32_e32 v4, v4, v66
	v_mul_f32_e32 v5, v5, v66
	.loc	1 779 12                        ; 7.ttgir:779:12
	v_mul_f32_e32 v6, v6, v66
	v_mul_f32_e32 v7, v7, v66
	v_mul_f32_e32 v8, v8, v66
	v_mul_f32_e32 v9, v9, v66
	.loc	1 782 12                        ; 7.ttgir:782:12
	v_mul_f32_e32 v10, v10, v66
	v_mul_f32_e32 v11, v11, v66
	v_mul_f32_e32 v12, v12, v66
	v_mul_f32_e32 v13, v13, v66
	.loc	1 785 12                        ; 7.ttgir:785:12
	v_mul_f32_e32 v14, v14, v66
	v_mul_f32_e32 v15, v15, v66
	v_mul_f32_e32 v16, v16, v66
	v_mul_f32_e32 v17, v17, v66
	.loc	1 802 12                        ; 7.ttgir:802:12
	v_xor_b32_e32 v67, 0xc18, v67
	.loc	1 834 12                        ; 7.ttgir:834:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]
	.loc	1 802 12                        ; 7.ttgir:802:12
	v_lshl_add_u32 v174, v67, 1, 0
	.loc	1 822 12                        ; 7.ttgir:822:12
	v_cvt_f16_f32_e32 v67, v117
	v_cvt_f16_f32_e32 v114, v118
	.loc	1 857 12                        ; 7.ttgir:857:12
	s_mul_i32 s20, s33, s41
	.loc	1 858 12                        ; 7.ttgir:858:12
	s_ashr_i32 s21, s20, 31
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s64, s28, s20
	.loc	1 835 12                        ; 7.ttgir:835:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]
	.loc	1 859 12                        ; 7.ttgir:859:12
	s_mul_i32 s22, s33, s44
	.loc	1 858 12                        ; 7.ttgir:858:12
	s_addc_u32 s25, s40, s21
	.loc	1 860 12                        ; 7.ttgir:860:12
	s_ashr_i32 s23, s22, 31
	s_lshl_b64 s[22:23], s[22:23], 1
	s_add_u32 s68, s36, s22
	s_addc_u32 s26, s66, s23
	.loc	1 864 12                        ; 7.ttgir:864:12
	s_lshl_b32 s24, s1, 5
	.loc	1 836 12                        ; 7.ttgir:836:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]
	.loc	1 822 12                        ; 7.ttgir:822:12
	v_cvt_f16_f32_e32 v76, v119
	v_cvt_f16_f32_e32 v77, v120
	.loc	1 798 12                        ; 7.ttgir:798:12
	ds_read_b64 v[70:71], v171 offset:8192
	.loc	1 800 12                        ; 7.ttgir:800:12
	ds_read_b64 v[72:73], v173 offset:8192
	.loc	1 802 12                        ; 7.ttgir:802:12
	ds_read_b64 v[74:75], v174 offset:8192
	.loc	1 865 12                        ; 7.ttgir:865:12
	s_or_b32 s27, s24, 31
	.loc	1 871 12                        ; 7.ttgir:871:12
	s_cmp_lg_u32 s67, 0
	.loc	1 837 12                        ; 7.ttgir:837:12
	v_pack_b32_f16 v77, v76, v77
	v_pack_b32_f16 v76, v67, v114
	.loc	1 804 12                        ; 7.ttgir:804:12
	v_xor_b32_e32 v67, v175, v127
	v_lshl_add_u32 v177, v67, 1, 0
	.loc	1 806 12                        ; 7.ttgir:806:12
	v_xor_b32_e32 v67, v178, v127
	.loc	1 837 12                        ; 7.ttgir:837:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]
	.loc	1 806 12                        ; 7.ttgir:806:12
	v_xor_b32_e32 v68, 0x408, v67
	v_lshl_add_u32 v175, v68, 1, 0
	.loc	1 808 12                        ; 7.ttgir:808:12
	v_xor_b32_e32 v68, 0x810, v67
	v_lshl_add_u32 v179, v68, 1, 0
	.loc	1 804 12                        ; 7.ttgir:804:12
	ds_read_b64 v[68:69], v177 offset:8192
	.loc	1 810 12                        ; 7.ttgir:810:12
	v_xor_b32_e32 v67, 0xc18, v67
	v_lshl_add_u32 v181, v67, 1, 0
	.loc	1 838 12                        ; 7.ttgir:838:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]
	.loc	1 823 12                        ; 7.ttgir:823:12
	v_cvt_f16_f32_e32 v67, v121
	v_cvt_f16_f32_e32 v114, v124
	.loc	1 871 12                        ; 7.ttgir:871:12
	s_cselect_b64 s[48:49], -1, 0
	.loc	1 891 12                        ; 7.ttgir:891:12
	s_and_b32 s25, s25, 0xffff
	s_or_b32 s65, s25, s53
	s_mov_b32 s67, 0x27000
	s_mov_b32 s66, 0x7ffffffe
	.loc	1 839 12                        ; 7.ttgir:839:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]
	.loc	1 896 12                        ; 7.ttgir:896:12
	s_and_b32 s25, s26, 0xffff
	s_or_b32 s69, s25, s55
	s_mov_b32 s70, s66
	s_mov_b32 s71, s67
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_lshl_add_u32 v193, v142, 1, 0
	v_lshl_add_u32 v194, v143, 1, 0
	v_lshl_add_u32 v195, v144, 1, 0
	.loc	1 840 12                        ; 7.ttgir:840:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]
	.loc	1 823 12                        ; 7.ttgir:823:12
	v_cvt_f16_f32_e32 v76, v125
	v_cvt_f16_f32_e32 v77, v126
	.loc	1 806 12                        ; 7.ttgir:806:12
	ds_read_b64 v[70:71], v175 offset:8192
	.loc	1 808 12                        ; 7.ttgir:808:12
	ds_read_b64 v[72:73], v179 offset:8192
	.loc	1 810 12                        ; 7.ttgir:810:12
	ds_read_b64 v[74:75], v181 offset:8192
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_lshl_add_u32 v196, v145, 1, 0
	v_lshl_add_u32 v197, v146, 1, 0
	.loc	1 841 12                        ; 7.ttgir:841:12
	v_pack_b32_f16 v77, v76, v77
	v_pack_b32_f16 v76, v67, v114
	.loc	1 812 12                        ; 7.ttgir:812:12
	v_xor_b32_e32 v67, v166, v127
	v_lshl_add_u32 v178, v67, 1, 0
	.loc	1 814 12                        ; 7.ttgir:814:12
	v_xor_b32_e32 v67, v176, v127
	.loc	1 841 12                        ; 7.ttgir:841:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]
	.loc	1 814 12                        ; 7.ttgir:814:12
	v_xor_b32_e32 v68, 0x408, v67
	v_lshl_add_u32 v176, v68, 1, 0
	.loc	1 816 12                        ; 7.ttgir:816:12
	v_xor_b32_e32 v68, 0x810, v67
	v_lshl_add_u32 v180, v68, 1, 0
	.loc	1 812 12                        ; 7.ttgir:812:12
	ds_read_b64 v[68:69], v178 offset:8192
	.loc	1 818 12                        ; 7.ttgir:818:12
	v_xor_b32_e32 v67, 0xc18, v67
	v_lshl_add_u32 v182, v67, 1, 0
	.loc	1 843 12                        ; 7.ttgir:843:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]
	.loc	1 824 12                        ; 7.ttgir:824:12
	v_cvt_f16_f32_e32 v67, v78
	v_cvt_f16_f32_e32 v78, v79
	.loc	1 904 5                         ; 7.ttgir:904:5
	v_lshl_add_u32 v198, v147, 1, 0
	v_lshl_add_u32 v199, v149, 1, 0
	v_lshl_add_u32 v200, v150, 1, 0
	.loc	1 845 12                        ; 7.ttgir:845:12
	v_pack_b32_f16 v118, v67, v78
	.loc	1 878 12                        ; 7.ttgir:878:12
	v_or_b32_e32 v67, s33, v158
	.loc	1 842 12                        ; 7.ttgir:842:12
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]
	.loc	1 891 12                        ; 7.ttgir:891:12
	v_bfrev_b32_e32 v78, 1
	.loc	1 885 12                        ; 7.ttgir:885:12
	v_cmp_gt_i32_e32 vcc, s19, v67
	.loc	1 908 12                        ; 7.ttgir:908:12
	s_cmp_gt_i32 s27, 63
	.loc	1 891 12                        ; 7.ttgir:891:12
	s_nop 0
	v_cndmask_b32_e32 v67, v78, v160, vcc
	.loc	1 844 12                        ; 7.ttgir:844:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]
	.loc	1 824 12                        ; 7.ttgir:824:12
	v_cvt_f16_f32_e32 v74, v80
	v_cvt_f16_f32_e32 v75, v81
	.loc	1 814 12                        ; 7.ttgir:814:12
	ds_read_b64 v[70:71], v176 offset:8192
	.loc	1 816 12                        ; 7.ttgir:816:12
	ds_read_b64 v[72:73], v180 offset:8192
	.loc	1 818 12                        ; 7.ttgir:818:12
	ds_read_b64 v[80:81], v182 offset:8192
	.loc	1 879 12                        ; 7.ttgir:879:12
	v_or_b32_e32 v76, s33, v162
	v_or_b32_e32 v77, s33, v189
	.loc	1 845 12                        ; 7.ttgir:845:12
	v_pack_b32_f16 v119, v74, v75
	.loc	1 853 5                         ; 7.ttgir:853:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 845 12                        ; 7.ttgir:845:12
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[118:119], v[50:65]
	.loc	1 878 12                        ; 7.ttgir:878:12
	v_or_b32_e32 v68, s33, v159
	.loc	1 885 12                        ; 7.ttgir:885:12
	v_cmp_gt_i32_e32 vcc, s19, v68
	.loc	1 847 12                        ; 7.ttgir:847:12
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[118:119], v[18:33]
	.loc	1 892 12                        ; 7.ttgir:892:12
	s_nop 0
	v_cndmask_b32_e32 v72, v78, v161, vcc
	.loc	1 894 12                        ; 7.ttgir:894:12
	v_cmp_gt_i32_e32 vcc, s19, v76
	.loc	1 846 12                        ; 7.ttgir:846:12
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[118:119], v[34:49]
	.loc	1 891 12                        ; 7.ttgir:891:12
	buffer_load_dwordx4 v[68:71], v67, s[64:67], 0 offen
	.loc	1 892 12                        ; 7.ttgir:892:12
	s_nop 0
	buffer_load_dwordx4 v[72:75], v72, s[64:67], 0 offen
	.loc	1 896 12                        ; 7.ttgir:896:12
	v_cndmask_b32_e32 v67, v78, v163, vcc
	.loc	1 894 12                        ; 7.ttgir:894:12
	v_cmp_gt_i32_e32 vcc, s19, v77
	.loc	1 896 12                        ; 7.ttgir:896:12
	s_nop 1
	v_cndmask_b32_e32 v114, v78, v164, vcc
	buffer_load_dwordx4 v[76:79], v67, s[68:71], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[114:117], v114, s[68:71], 0 offen
	.loc	1 848 12                        ; 7.ttgir:848:12
	v_mfma_f32_32x32x8_f16 v[2:17], v[80:81], v[118:119], v[2:17]
	.loc	1 899 5                         ; 7.ttgir:899:5
	v_mov_b32_e32 v67, 0x88
	v_mov_b32_e32 v80, 0x110
	v_cndmask_b32_e64 v67, v67, 0, s[14:15]
	v_cndmask_b32_e64 v80, v80, 0, s[12:13]
	v_or_b32_e32 v67, v67, v80
	v_mov_b32_e32 v80, 0x220
	v_cndmask_b32_e64 v80, v80, 0, s[2:3]
	v_or3_b32 v67, v67, v80, v148
	v_xor_b32_e32 v67, v67, v138
	.loc	1 904 5                         ; 7.ttgir:904:5
	s_mov_b32 s2, 0x5040100
	.loc	1 899 5                         ; 7.ttgir:899:5
	v_lshl_add_u32 v192, v67, 1, 0
	.loc	1 904 5                         ; 7.ttgir:904:5
	s_mov_b32 s3, 0x7060302
	.loc	1 899 5                         ; 7.ttgir:899:5
	s_waitcnt vmcnt(3)
	ds_write_b128 v192, v[68:71]
	.loc	1 901 5                         ; 7.ttgir:901:5
	s_waitcnt vmcnt(2)
	ds_write_b128 v192, v[72:75] offset:4096
	v_or_b32_e32 v68, 16, v141
	.loc	1 904 5                         ; 7.ttgir:904:5
	s_waitcnt vmcnt(0)
	v_perm_b32 v67, v114, v76, s2
	ds_write_b32 v193, v67 offset:8192
	v_perm_b32 v67, v114, v76, s3
	ds_write_b32 v194, v67 offset:8192
	v_perm_b32 v67, v115, v77, s2
	ds_write_b32 v195, v67 offset:8192
	v_perm_b32 v67, v115, v77, s3
	ds_write_b32 v196, v67 offset:8192
	v_perm_b32 v67, v116, v78, s2
	ds_write_b32 v197, v67 offset:8192
	v_perm_b32 v67, v116, v78, s3
	ds_write_b32 v198, v67 offset:8192
	v_perm_b32 v67, v117, v79, s2
	ds_write_b32 v199, v67 offset:8192
	v_perm_b32 v67, v117, v79, s3
	ds_write_b32 v200, v67 offset:8192
	v_xor_b32_e32 v67, v141, v139
	v_or_b32_e32 v191, v67, v140
	.loc	1 909 5                         ; 7.ttgir:909:5
	s_cbranch_scc1 .LBB0_10
; %bb.9:                                ; %._crit_edge.._crit_edge669_crit_edge
	.loc	1 1204 12                       ; 7.ttgir:1204:12
	v_xor_b32_e32 v69, v68, v139
	.loc	1 861 12                        ; 7.ttgir:861:12
	s_add_i32 s12, s33, 32
	.loc	1 1202 12                       ; 7.ttgir:1202:12
	v_or_b32_e32 v67, v67, v140
	.loc	1 1204 12                       ; 7.ttgir:1204:12
	v_or_b32_e32 v190, v69, v140
	s_mov_b64 s[2:3], 0
	s_branch .LBB0_11
.LBB0_10:
	.loc	1 0 12 is_stmt 0                ; 7.ttgir:0:12
	s_mov_b64 s[2:3], -1
                                        ; implicit-def: $sgpr12
                                        ; implicit-def: $vgpr67
                                        ; implicit-def: $vgpr190
.LBB0_11:                               ; %Flow391
	v_or_b32_e32 v166, s52, v165
	v_fmac_f32_e32 v122, v201, v66
	s_lshl_b32 s92, s72, 5
	.loc	1 909 5 is_stmt 1               ; 7.ttgir:909:5
	s_andn2_b64 vcc, exec, s[2:3]
	.loc	1 0 0 is_stmt 0                 ; 7.ttgir:0
	s_sub_i32 s44, s16, s19
	.loc	1 909 5                         ; 7.ttgir:909:5
	s_cbranch_vccnz .LBB0_15
; %bb.12:                               ; %.lr.ph668
	.loc	1 866 12 is_stmt 1              ; 7.ttgir:866:12
	s_and_b32 s1, s1, 0x7ffffff
	s_ashr_i32 s39, s38, 31
	.loc	1 909 5                         ; 7.ttgir:909:5
	s_add_i32 s93, s0, 1
	s_add_u32 s0, s62, s42
	s_addc_u32 s2, s63, s43
	s_add_u32 s0, s0, s60
	s_addc_u32 s2, s2, s61
	s_add_u32 s0, s0, s22
	s_addc_u32 s2, s2, s23
	s_lshl_b64 s[50:51], s[38:39], 1
	s_add_u32 s0, s0, s50
	s_addc_u32 s2, s2, s51
	s_add_u32 s94, s6, s0
	s_addc_u32 s95, s7, s2
	s_sub_i32 s96, 0, s24
	s_add_i32 s0, s44, s92
	v_add_u32_e32 v203, s0, v1
	s_add_u32 s0, s58, s56
	s_addc_u32 s2, s59, s57
	s_add_u32 s0, s0, s34
	s_addc_u32 s2, s2, s35
	s_add_u32 s0, s0, s20
                                        ; implicit-def: $vgpr227 : SGPR spill to VGPR lane
	s_addc_u32 s2, s2, s21
	s_lshl_b64 s[56:57], s[30:31], 1
	v_writelane_b32 v227, s82, 0
	v_xor_b32_e32 v66, v68, v139
	s_add_u32 s0, s0, s56
	v_writelane_b32 v227, s83, 1
	v_or_b32_e32 v190, v66, v140
	s_addc_u32 s2, s2, s57
	v_sub_u32_e64 v66, s1, 2 clamp
	v_writelane_b32 v227, s10, 2
	s_add_u32 s97, s4, s0
	v_readfirstlane_b32 s0, v66
	v_writelane_b32 v227, s11, 3
	v_lshl_add_u32 v201, v191, 1, 0
	v_lshl_add_u32 v202, v190, 1, 0
	v_add_u32_e32 v204, s92, v1
	s_addc_u32 s98, s5, s2
	s_add_i32 s99, s0, 1
	s_movk_i32 s0, 0xffe0
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	v_bfrev_b32_e32 v205, 1
	s_xor_b64 s[58:59], s[48:49], -1
	v_mov_b32_e32 v206, 0xff800000
	s_mov_b32 s1, 0x5040100
	s_mov_b32 s10, 0x7060302
.LBB0_13:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 0 5 is_stmt 0                 ; 7.ttgir:0:5
	v_mov_b32_e32 v207, v122
	v_mov_b32_e32 v210, v123
	.loc	1 911 5 is_stmt 1               ; 7.ttgir:911:5
	s_setprio 0
	.loc	1 917 12                        ; 7.ttgir:917:12
	s_lshl_b32 s33, s93, 5
	.loc	1 921 12                        ; 7.ttgir:921:12
	v_or_b32_e32 v66, s33, v158
	v_or_b32_e32 v67, s33, v159
	.loc	1 928 12                        ; 7.ttgir:928:12
	v_cmp_gt_i32_e32 vcc, s19, v66
	.loc	1 934 12                        ; 7.ttgir:934:12
	s_and_b32 s4, s98, 0xffff
	.loc	1 928 12                        ; 7.ttgir:928:12
	v_cmp_gt_i32_e64 s[2:3], s19, v67
	.loc	1 934 12                        ; 7.ttgir:934:12
	s_or_b32 s41, s4, s53
	s_mov_b32 s40, s97
	v_cndmask_b32_e32 v66, v205, v160, vcc
	buffer_load_dwordx4 v[114:117], v66, s[40:43], 0 offen
	.loc	1 935 12                        ; 7.ttgir:935:12
	v_cndmask_b32_e64 v66, v205, v161, s[2:3]
	buffer_load_dwordx4 v[118:121], v66, s[40:43], 0 offen
	.loc	1 954 12                        ; 7.ttgir:954:12
	s_cmp_lg_u32 s96, s0
	.loc	1 959 12                        ; 7.ttgir:959:12
	v_add_u32_e32 v66, s96, v204
	.loc	1 954 12                        ; 7.ttgir:954:12
	s_cselect_b64 s[60:61], -1, 0
	.loc	1 959 12                        ; 7.ttgir:959:12
	v_add_u32_e32 v67, 1, v66
	v_add_u32_e32 v68, 2, v66
	.loc	1 962 12                        ; 7.ttgir:962:12
	v_add_u32_e32 v69, 3, v66
	v_add_u32_e32 v70, 8, v66
	v_add_u32_e32 v71, 9, v66
	v_add_u32_e32 v72, 10, v66
	.loc	1 965 12                        ; 7.ttgir:965:12
	v_add_u32_e32 v73, 11, v66
	v_add_u32_e32 v74, 16, v66
	v_add_u32_e32 v75, 17, v66
	v_add_u32_e32 v76, 18, v66
	.loc	1 968 12                        ; 7.ttgir:968:12
	v_add_u32_e32 v77, 19, v66
	v_add_u32_e32 v78, 24, v66
	v_add_u32_e32 v79, 25, v66
	v_add_u32_e32 v80, 26, v66
	.loc	1 970 12                        ; 7.ttgir:970:12
	v_add_u32_e32 v81, 27, v66
	v_cmp_gt_i32_e32 vcc, s19, v66
	.loc	1 974 12                        ; 7.ttgir:974:12
	v_add_u32_e32 v66, s96, v203
	.loc	1 970 12                        ; 7.ttgir:970:12
	v_cmp_gt_i32_e64 s[2:3], s19, v67
	v_cmp_gt_i32_e64 s[4:5], s19, v68
	v_cmp_gt_i32_e64 s[6:7], s19, v69
	v_cmp_gt_i32_e64 s[12:13], s19, v70
	v_cmp_gt_i32_e64 s[14:15], s19, v71
	v_cmp_gt_i32_e64 s[20:21], s19, v72
	v_cmp_gt_i32_e64 s[22:23], s19, v73
	v_cmp_gt_i32_e64 s[24:25], s19, v74
	v_cmp_gt_i32_e64 s[26:27], s19, v75
	v_cmp_gt_i32_e64 s[28:29], s19, v76
	v_cmp_gt_i32_e64 s[30:31], s19, v77
	v_cmp_gt_i32_e64 s[34:35], s19, v78
	v_cmp_gt_i32_e64 s[36:37], s19, v79
	v_cmp_gt_i32_e64 s[38:39], s19, v80
	v_cmp_gt_i32_e64 s[40:41], s19, v81
	.loc	1 972 12                        ; 7.ttgir:972:12
	s_or_b64 s[90:91], s[58:59], s[60:61]
	.loc	1 974 12                        ; 7.ttgir:974:12
	v_add_u32_e32 v67, 1, v66
	v_add_u32_e32 v68, 2, v66
	v_add_u32_e32 v69, 3, v66
	v_add_u32_e32 v70, 8, v66
	v_add_u32_e32 v71, 9, v66
	v_add_u32_e32 v72, 10, v66
	v_add_u32_e32 v73, 11, v66
	v_add_u32_e32 v74, 16, v66
	v_add_u32_e32 v75, 17, v66
	v_add_u32_e32 v76, 18, v66
	v_add_u32_e32 v77, 19, v66
	v_add_u32_e32 v78, 24, v66
	v_add_u32_e32 v79, 25, v66
	v_add_u32_e32 v80, 26, v66
	.loc	1 985 12                        ; 7.ttgir:985:12
	v_add_u32_e32 v81, 27, v66
	.loc	1 936 5                         ; 7.ttgir:936:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 938 12                        ; 7.ttgir:938:12
	ds_read_b128 v[150:153], v201
	.loc	1 940 12                        ; 7.ttgir:940:12
	ds_read_b128 v[146:149], v202
	.loc	1 942 12                        ; 7.ttgir:942:12
	ds_read_b128 v[142:145], v183
	.loc	1 944 12                        ; 7.ttgir:944:12
	ds_read_b128 v[138:141], v184
	.loc	1 946 12                        ; 7.ttgir:946:12
	ds_read_b128 v[134:137], v185
	.loc	1 948 12                        ; 7.ttgir:948:12
	ds_read_b128 v[130:133], v186
	.loc	1 950 12                        ; 7.ttgir:950:12
	ds_read_b128 v[126:129], v187
	.loc	1 952 12                        ; 7.ttgir:952:12
	ds_read_b128 v[122:125], v188
	.loc	1 972 12                        ; 7.ttgir:972:12
	s_or_b64 s[60:61], s[90:91], vcc
	s_or_b64 s[62:63], s[90:91], s[2:3]
	s_or_b64 s[64:65], s[90:91], s[4:5]
	s_or_b64 s[66:67], s[90:91], s[6:7]
	s_or_b64 s[68:69], s[90:91], s[12:13]
	s_or_b64 s[70:71], s[90:91], s[14:15]
	s_or_b64 s[72:73], s[90:91], s[20:21]
	s_or_b64 s[74:75], s[90:91], s[22:23]
	s_or_b64 s[76:77], s[90:91], s[24:25]
	s_or_b64 s[78:79], s[90:91], s[26:27]
	s_or_b64 s[80:81], s[90:91], s[28:29]
	s_or_b64 s[82:83], s[90:91], s[30:31]
	s_or_b64 s[84:85], s[90:91], s[34:35]
	s_or_b64 s[86:87], s[90:91], s[36:37]
	s_or_b64 s[88:89], s[90:91], s[38:39]
	s_or_b64 s[90:91], s[90:91], s[40:41]
	.loc	1 985 12                        ; 7.ttgir:985:12
	v_cmp_ge_i32_e32 vcc, v166, v66
	v_cmp_ge_i32_e64 s[2:3], v166, v67
	v_cmp_ge_i32_e64 s[4:5], v166, v68
	v_cmp_ge_i32_e64 s[6:7], v166, v69
	v_cmp_ge_i32_e64 s[12:13], v166, v70
	v_cmp_ge_i32_e64 s[14:15], v166, v71
	v_cmp_ge_i32_e64 s[20:21], v166, v72
	v_cmp_ge_i32_e64 s[22:23], v166, v73
	v_cmp_ge_i32_e64 s[24:25], v166, v74
	v_cmp_ge_i32_e64 s[26:27], v166, v75
	v_cmp_ge_i32_e64 s[28:29], v166, v76
	v_cmp_ge_i32_e64 s[30:31], v166, v77
	v_cmp_ge_i32_e64 s[34:35], v166, v78
	v_cmp_ge_i32_e64 s[36:37], v166, v79
	v_cmp_ge_i32_e64 s[38:39], v166, v80
	v_cmp_ge_i32_e64 s[40:41], v166, v81
	.loc	1 1008 12                       ; 7.ttgir:1008:12
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 v[66:81], v[150:151], v[110:111], 0
	.loc	1 988 12                        ; 7.ttgir:988:12
	s_and_b64 s[60:61], vcc, s[60:61]
	s_and_b64 s[2:3], s[2:3], s[62:63]
	s_and_b64 s[4:5], s[4:5], s[64:65]
	s_and_b64 s[6:7], s[6:7], s[66:67]
	.loc	1 999 12                        ; 7.ttgir:999:12
	v_cndmask_b32_e64 v211, v206, 0, s[60:61]
	v_cndmask_b32_e64 v212, v206, 0, s[2:3]
	.loc	1 991 12                        ; 7.ttgir:991:12
	s_and_b64 s[12:13], s[12:13], s[68:69]
	.loc	1 1008 12                       ; 7.ttgir:1008:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[152:153], v[112:113], v[66:81]
	.loc	1 991 12                        ; 7.ttgir:991:12
	s_and_b64 s[14:15], s[14:15], s[70:71]
	.loc	1 999 12                        ; 7.ttgir:999:12
	v_cndmask_b32_e64 v213, v206, 0, s[4:5]
	v_cndmask_b32_e64 v214, v206, 0, s[6:7]
	.loc	1 991 12                        ; 7.ttgir:991:12
	s_and_b64 s[20:21], s[20:21], s[72:73]
	s_and_b64 s[22:23], s[22:23], s[74:75]
	.loc	1 999 12                        ; 7.ttgir:999:12
	v_cndmask_b32_e64 v215, v206, 0, s[12:13]
	v_cndmask_b32_e64 v216, v206, 0, s[14:15]
	.loc	1 1009 12                       ; 7.ttgir:1009:12
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[106:107], v[66:81]
	.loc	1 994 12                        ; 7.ttgir:994:12
	s_and_b64 s[24:25], s[24:25], s[76:77]
	s_and_b64 s[26:27], s[26:27], s[78:79]
	.loc	1 999 12                        ; 7.ttgir:999:12
	v_cndmask_b32_e64 v217, v206, 0, s[20:21]
	v_cndmask_b32_e64 v218, v206, 0, s[22:23]
	.loc	1 994 12                        ; 7.ttgir:994:12
	s_and_b64 s[28:29], s[28:29], s[80:81]
	s_and_b64 s[30:31], s[30:31], s[82:83]
	.loc	1 999 12                        ; 7.ttgir:999:12
	v_cndmask_b32_e64 v219, v206, 0, s[24:25]
	.loc	1 1009 12                       ; 7.ttgir:1009:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[148:149], v[108:109], v[66:81]
	.loc	1 999 12                        ; 7.ttgir:999:12
	v_cndmask_b32_e64 v220, v206, 0, s[26:27]
	.loc	1 997 12                        ; 7.ttgir:997:12
	s_and_b64 s[34:35], s[34:35], s[84:85]
	s_and_b64 s[36:37], s[36:37], s[86:87]
	.loc	1 999 12                        ; 7.ttgir:999:12
	v_cndmask_b32_e64 v221, v206, 0, s[28:29]
	v_cndmask_b32_e64 v222, v206, 0, s[30:31]
	.loc	1 997 12                        ; 7.ttgir:997:12
	s_and_b64 s[38:39], s[38:39], s[88:89]
	s_and_b64 s[40:41], s[40:41], s[90:91]
	.loc	1 1010 12                       ; 7.ttgir:1010:12
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[102:103], v[66:81]
	.loc	1 999 12                        ; 7.ttgir:999:12
	v_cndmask_b32_e64 v223, v206, 0, s[34:35]
	v_cndmask_b32_e64 v224, v206, 0, s[36:37]
	v_cndmask_b32_e64 v225, v206, 0, s[38:39]
	v_cndmask_b32_e64 v226, v206, 0, s[40:41]
	.loc	1 922 12                        ; 7.ttgir:922:12
	v_or_b32_e32 v208, s33, v162
	v_or_b32_e32 v209, s33, v189
	.loc	1 1117 12                       ; 7.ttgir:1117:12
	v_cmp_gt_i32_e32 vcc, s19, v208
	.loc	1 1010 12                       ; 7.ttgir:1010:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[104:105], v[66:81]
	.loc	1 1117 12                       ; 7.ttgir:1117:12
	v_cmp_gt_i32_e64 s[2:3], s19, v209
	.loc	1 1119 12                       ; 7.ttgir:1119:12
	s_and_b32 s4, s95, 0xffff
	s_or_b32 s41, s4, s55
	s_mov_b32 s40, s94
	.loc	1 908 12                        ; 7.ttgir:908:12
	s_add_u32 s94, s94, s50
	s_addc_u32 s95, s95, s51
	s_sub_i32 s0, s0, 32
	.loc	1 1011 12                       ; 7.ttgir:1011:12
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[98:99], v[66:81]
	.loc	1 908 12                        ; 7.ttgir:908:12
	s_add_u32 s97, s97, s56
	s_addc_u32 s98, s98, s57
	s_add_i32 s93, s93, 1
	s_add_i32 s99, s99, -1
	v_add_u32_e32 v203, 32, v203
	v_add_u32_e32 v204, 32, v204
	s_cmp_lg_u32 s99, 0
	.loc	1 1011 12                       ; 7.ttgir:1011:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[100:101], v[66:81]
	.loc	1 1012 12                       ; 7.ttgir:1012:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[94:95], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[96:97], v[66:81]
	.loc	1 1013 12                       ; 7.ttgir:1013:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[90:91], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[92:93], v[66:81]
	.loc	1 1014 12                       ; 7.ttgir:1014:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[86:87], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[88:89], v[66:81]
	.loc	1 1015 12                       ; 7.ttgir:1015:12
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[82:83], v[66:81]
	v_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[84:85], v[66:81]
	.loc	1 1029 12                       ; 7.ttgir:1029:12
	s_nop 7
	s_nop 2
	v_fmac_f32_e32 v211, 0x3e0293ee, v66
	v_fmac_f32_e32 v212, 0x3e0293ee, v67
	v_fmac_f32_e32 v213, 0x3e0293ee, v68
	v_fmac_f32_e32 v214, 0x3e0293ee, v69
	.loc	1 1039 15                       ; 7.ttgir:1039:15
	v_max_f32_e32 v66, v211, v212
	.loc	1 1031 12                       ; 7.ttgir:1031:12
	v_fmac_f32_e32 v215, 0x3e0293ee, v70
	v_fmac_f32_e32 v216, 0x3e0293ee, v71
	.loc	1 1039 15                       ; 7.ttgir:1039:15
	v_max3_f32 v66, v66, v213, v214
	.loc	1 1031 12                       ; 7.ttgir:1031:12
	v_fmac_f32_e32 v217, 0x3e0293ee, v72
	v_fmac_f32_e32 v218, 0x3e0293ee, v73
	.loc	1 1039 15                       ; 7.ttgir:1039:15
	v_max3_f32 v66, v66, v215, v216
	.loc	1 1033 12                       ; 7.ttgir:1033:12
	v_fmac_f32_e32 v219, 0x3e0293ee, v74
	v_fmac_f32_e32 v220, 0x3e0293ee, v75
	.loc	1 1039 15                       ; 7.ttgir:1039:15
	v_max3_f32 v66, v66, v217, v218
	.loc	1 1033 12                       ; 7.ttgir:1033:12
	v_fmac_f32_e32 v221, 0x3e0293ee, v76
	v_fmac_f32_e32 v222, 0x3e0293ee, v77
	.loc	1 1039 15                       ; 7.ttgir:1039:15
	v_max3_f32 v66, v66, v219, v220
	.loc	1 1035 12                       ; 7.ttgir:1035:12
	v_fmac_f32_e32 v223, 0x3e0293ee, v78
	v_fmac_f32_e32 v224, 0x3e0293ee, v79
	.loc	1 1039 15                       ; 7.ttgir:1039:15
	v_max3_f32 v66, v66, v221, v222
	.loc	1 1035 12                       ; 7.ttgir:1035:12
	v_fmac_f32_e32 v225, 0x3e0293ee, v80
	v_fmac_f32_e32 v226, 0x3e0293ee, v81
	.loc	1 1039 15                       ; 7.ttgir:1039:15
	v_max3_f32 v66, v66, v223, v224
	v_max3_f32 v66, v66, v225, v226
	.loc	1 1037 12                       ; 7.ttgir:1037:12
	ds_bpermute_b32 v67, v157, v66
	.loc	1 1042 12                       ; 7.ttgir:1042:12
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v123, v210, v66, v67
	.loc	1 1046 12                       ; 7.ttgir:1046:12
	v_sub_f32_e32 v66, v211, v123
	v_sub_f32_e32 v67, v212, v123
	v_sub_f32_e32 v68, v213, v123
	.loc	1 1053 12                       ; 7.ttgir:1053:12
	v_exp_f32_e32 v148, v66
	v_exp_f32_e32 v149, v67
	.loc	1 1046 12                       ; 7.ttgir:1046:12
	v_sub_f32_e32 v69, v214, v123
	.loc	1 1053 12                       ; 7.ttgir:1053:12
	v_exp_f32_e32 v150, v68
	.loc	1 1048 12                       ; 7.ttgir:1048:12
	v_sub_f32_e32 v70, v215, v123
	.loc	1 1053 12                       ; 7.ttgir:1053:12
	v_exp_f32_e32 v151, v69
	.loc	1 1048 12                       ; 7.ttgir:1048:12
	v_sub_f32_e32 v71, v216, v123
	.loc	1 1054 12                       ; 7.ttgir:1054:12
	v_exp_f32_e32 v152, v70
	.loc	1 1048 12                       ; 7.ttgir:1048:12
	v_sub_f32_e32 v72, v217, v123
	.loc	1 1054 12                       ; 7.ttgir:1054:12
	v_exp_f32_e32 v153, v71
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v148, v149
	.loc	1 1048 12                       ; 7.ttgir:1048:12
	v_sub_f32_e32 v73, v218, v123
	.loc	1 1054 12                       ; 7.ttgir:1054:12
	v_exp_f32_e32 v211, v72
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v150, v66
	.loc	1 1050 12                       ; 7.ttgir:1050:12
	v_sub_f32_e32 v74, v219, v123
	.loc	1 1054 12                       ; 7.ttgir:1054:12
	v_exp_f32_e32 v212, v73
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v151, v66
	.loc	1 1050 12                       ; 7.ttgir:1050:12
	v_sub_f32_e32 v75, v220, v123
	.loc	1 1055 12                       ; 7.ttgir:1055:12
	v_exp_f32_e32 v213, v74
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v152, v66
	.loc	1 1050 12                       ; 7.ttgir:1050:12
	v_sub_f32_e32 v76, v221, v123
	.loc	1 1055 12                       ; 7.ttgir:1055:12
	v_exp_f32_e32 v214, v75
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v153, v66
	.loc	1 1050 12                       ; 7.ttgir:1050:12
	v_sub_f32_e32 v77, v222, v123
	.loc	1 1055 12                       ; 7.ttgir:1055:12
	v_exp_f32_e32 v215, v76
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v211, v66
	.loc	1 1052 12                       ; 7.ttgir:1052:12
	v_sub_f32_e32 v78, v223, v123
	.loc	1 1055 12                       ; 7.ttgir:1055:12
	v_exp_f32_e32 v216, v77
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v212, v66
	.loc	1 1052 12                       ; 7.ttgir:1052:12
	v_sub_f32_e32 v79, v224, v123
	.loc	1 1056 12                       ; 7.ttgir:1056:12
	v_exp_f32_e32 v217, v78
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v213, v66
	.loc	1 1052 12                       ; 7.ttgir:1052:12
	v_sub_f32_e32 v80, v225, v123
	.loc	1 1056 12                       ; 7.ttgir:1056:12
	v_exp_f32_e32 v218, v79
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v214, v66
	.loc	1 1052 12                       ; 7.ttgir:1052:12
	v_sub_f32_e32 v81, v226, v123
	.loc	1 1056 12                       ; 7.ttgir:1056:12
	v_exp_f32_e32 v219, v80
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v215, v66
	.loc	1 1056 12                       ; 7.ttgir:1056:12
	v_exp_f32_e32 v220, v81
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	v_add_f32_e32 v66, v216, v66
	v_add_f32_e32 v66, v217, v66
	v_add_f32_e32 v66, v218, v66
	v_add_f32_e32 v66, v219, v66
	v_add_f32_e32 v66, v220, v66
	.loc	1 1058 12                       ; 7.ttgir:1058:12
	ds_bpermute_b32 v67, v157, v66
	.loc	1 1119 12                       ; 7.ttgir:1119:12
	v_cndmask_b32_e64 v70, v205, v164, s[2:3]
	buffer_load_dwordx4 v[70:73], v70, s[40:43], 0 offen
	.loc	1 1154 12                       ; 7.ttgir:1154:12
	v_cvt_f16_f32_e32 v148, v148
	.loc	1 1155 12                       ; 7.ttgir:1155:12
	v_cvt_f16_f32_e32 v208, v212
	.loc	1 1060 15                       ; 7.ttgir:1060:15
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v122, v66, v67
	.loc	1 1063 12                       ; 7.ttgir:1063:12
	v_sub_f32_e32 v66, v210, v123
	.loc	1 1064 12                       ; 7.ttgir:1064:12
	v_exp_f32_e32 v210, v66
	.loc	1 1119 12                       ; 7.ttgir:1119:12
	v_cndmask_b32_e32 v66, v205, v163, vcc
	buffer_load_dwordx4 v[66:69], v66, s[40:43], 0 offen
	.loc	1 1121 12                       ; 7.ttgir:1121:12
	ds_read_b64 v[74:75], v168 offset:8192
	.loc	1 1123 12                       ; 7.ttgir:1123:12
	ds_read_b64 v[76:77], v167 offset:8192
	.loc	1 1125 12                       ; 7.ttgir:1125:12
	ds_read_b64 v[78:79], v169 offset:8192
	.loc	1 1127 12                       ; 7.ttgir:1127:12
	ds_read_b64 v[80:81], v170 offset:8192
	.loc	1 1129 12                       ; 7.ttgir:1129:12
	ds_read_b64 v[124:125], v172 offset:8192
	.loc	1 1131 12                       ; 7.ttgir:1131:12
	ds_read_b64 v[126:127], v171 offset:8192
	.loc	1 1133 12                       ; 7.ttgir:1133:12
	ds_read_b64 v[128:129], v173 offset:8192
	.loc	1 1135 12                       ; 7.ttgir:1135:12
	ds_read_b64 v[130:131], v174 offset:8192
	.loc	1 1137 12                       ; 7.ttgir:1137:12
	ds_read_b64 v[132:133], v177 offset:8192
	.loc	1 1139 12                       ; 7.ttgir:1139:12
	ds_read_b64 v[134:135], v175 offset:8192
	.loc	1 1141 12                       ; 7.ttgir:1141:12
	ds_read_b64 v[136:137], v179 offset:8192
	.loc	1 1143 12                       ; 7.ttgir:1143:12
	ds_read_b64 v[138:139], v181 offset:8192
	.loc	1 1145 12                       ; 7.ttgir:1145:12
	ds_read_b64 v[140:141], v178 offset:8192
	.loc	1 1147 12                       ; 7.ttgir:1147:12
	ds_read_b64 v[142:143], v176 offset:8192
	.loc	1 1149 12                       ; 7.ttgir:1149:12
	ds_read_b64 v[144:145], v180 offset:8192
	.loc	1 1151 12                       ; 7.ttgir:1151:12
	ds_read_b64 v[146:147], v182 offset:8192
	.loc	1 1153 12                       ; 7.ttgir:1153:12
	v_fmac_f32_e32 v122, v207, v210
	.loc	1 1154 12                       ; 7.ttgir:1154:12
	v_cvt_f16_f32_e32 v207, v149
	v_cvt_f16_f32_e32 v149, v150
	v_cvt_f16_f32_e32 v150, v151
	.loc	1 1069 12                       ; 7.ttgir:1069:12
	v_mul_f32_e32 v50, v50, v210
	v_mul_f32_e32 v51, v51, v210
	v_mul_f32_e32 v52, v52, v210
	v_mul_f32_e32 v53, v53, v210
	.loc	1 1072 12                       ; 7.ttgir:1072:12
	v_mul_f32_e32 v54, v54, v210
	v_mul_f32_e32 v55, v55, v210
	v_mul_f32_e32 v56, v56, v210
	v_mul_f32_e32 v57, v57, v210
	.loc	1 1075 12                       ; 7.ttgir:1075:12
	v_mul_f32_e32 v58, v58, v210
	v_mul_f32_e32 v59, v59, v210
	v_mul_f32_e32 v60, v60, v210
	v_mul_f32_e32 v61, v61, v210
	.loc	1 1078 12                       ; 7.ttgir:1078:12
	v_mul_f32_e32 v62, v62, v210
	v_mul_f32_e32 v63, v63, v210
	v_mul_f32_e32 v64, v64, v210
	v_mul_f32_e32 v65, v65, v210
	.loc	1 1081 12                       ; 7.ttgir:1081:12
	v_mul_f32_e32 v34, v34, v210
	v_mul_f32_e32 v35, v35, v210
	v_mul_f32_e32 v36, v36, v210
	v_mul_f32_e32 v37, v37, v210
	.loc	1 1084 12                       ; 7.ttgir:1084:12
	v_mul_f32_e32 v38, v38, v210
	v_mul_f32_e32 v39, v39, v210
	v_mul_f32_e32 v40, v40, v210
	v_mul_f32_e32 v41, v41, v210
	.loc	1 1087 12                       ; 7.ttgir:1087:12
	v_mul_f32_e32 v42, v42, v210
	v_mul_f32_e32 v43, v43, v210
	v_mul_f32_e32 v44, v44, v210
	v_mul_f32_e32 v45, v45, v210
	.loc	1 1090 12                       ; 7.ttgir:1090:12
	v_mul_f32_e32 v46, v46, v210
	v_mul_f32_e32 v47, v47, v210
	v_mul_f32_e32 v48, v48, v210
	v_mul_f32_e32 v49, v49, v210
	.loc	1 1093 12                       ; 7.ttgir:1093:12
	v_mul_f32_e32 v18, v18, v210
	v_mul_f32_e32 v19, v19, v210
	v_mul_f32_e32 v20, v20, v210
	v_mul_f32_e32 v21, v21, v210
	.loc	1 1096 12                       ; 7.ttgir:1096:12
	v_mul_f32_e32 v22, v22, v210
	v_mul_f32_e32 v23, v23, v210
	v_mul_f32_e32 v24, v24, v210
	v_mul_f32_e32 v25, v25, v210
	.loc	1 1099 12                       ; 7.ttgir:1099:12
	v_mul_f32_e32 v26, v26, v210
	v_mul_f32_e32 v27, v27, v210
	v_mul_f32_e32 v28, v28, v210
	v_mul_f32_e32 v29, v29, v210
	.loc	1 1102 12                       ; 7.ttgir:1102:12
	v_mul_f32_e32 v30, v30, v210
	v_mul_f32_e32 v31, v31, v210
	v_mul_f32_e32 v32, v32, v210
	v_mul_f32_e32 v33, v33, v210
	.loc	1 1105 12                       ; 7.ttgir:1105:12
	v_mul_f32_e32 v2, v2, v210
	v_mul_f32_e32 v3, v3, v210
	v_mul_f32_e32 v4, v4, v210
	v_mul_f32_e32 v5, v5, v210
	.loc	1 1108 12                       ; 7.ttgir:1108:12
	v_mul_f32_e32 v6, v6, v210
	v_mul_f32_e32 v7, v7, v210
	v_mul_f32_e32 v8, v8, v210
	v_mul_f32_e32 v9, v9, v210
	.loc	1 1111 12                       ; 7.ttgir:1111:12
	v_mul_f32_e32 v10, v10, v210
	v_mul_f32_e32 v11, v11, v210
	v_mul_f32_e32 v12, v12, v210
	v_mul_f32_e32 v13, v13, v210
	.loc	1 1114 12                       ; 7.ttgir:1114:12
	v_mul_f32_e32 v14, v14, v210
	v_mul_f32_e32 v15, v15, v210
	v_mul_f32_e32 v16, v16, v210
	v_mul_f32_e32 v17, v17, v210
	.loc	1 1166 12                       ; 7.ttgir:1166:12
	v_pack_b32_f16 v149, v149, v150
	v_pack_b32_f16 v148, v148, v207
	.loc	1 1155 12                       ; 7.ttgir:1155:12
	v_cvt_f16_f32_e32 v151, v152
	v_cvt_f16_f32_e32 v152, v153
	.loc	1 1166 12                       ; 7.ttgir:1166:12
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x8_f16 v[50:65], v[74:75], v[148:149], v[50:65]
	.loc	1 1155 12                       ; 7.ttgir:1155:12
	v_cvt_f16_f32_e32 v153, v211
	.loc	1 1156 12                       ; 7.ttgir:1156:12
	v_cvt_f16_f32_e32 v209, v213
	.loc	1 1170 12                       ; 7.ttgir:1170:12
	v_pack_b32_f16 v74, v151, v152
	.loc	1 1156 12                       ; 7.ttgir:1156:12
	v_cvt_f16_f32_e32 v210, v214
	.loc	1 1170 12                       ; 7.ttgir:1170:12
	v_pack_b32_f16 v75, v153, v208
	.loc	1 1156 12                       ; 7.ttgir:1156:12
	v_cvt_f16_f32_e32 v211, v215
	v_cvt_f16_f32_e32 v212, v216
	.loc	1 1167 12                       ; 7.ttgir:1167:12
	v_mfma_f32_32x32x8_f16 v[34:49], v[76:77], v[148:149], v[34:49]
	.loc	1 1157 12                       ; 7.ttgir:1157:12
	v_cvt_f16_f32_e32 v213, v217
	v_cvt_f16_f32_e32 v214, v218
	v_cvt_f16_f32_e32 v215, v219
	v_cvt_f16_f32_e32 v216, v220
	.loc	1 1187 5                        ; 7.ttgir:1187:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 1168 12                       ; 7.ttgir:1168:12
	v_mfma_f32_32x32x8_f16 v[18:33], v[78:79], v[148:149], v[18:33]
	.loc	1 1189 5                        ; 7.ttgir:1189:5
	s_waitcnt vmcnt(3)
	ds_write_b128 v192, v[114:117]
	.loc	1 1191 5                        ; 7.ttgir:1191:5
	s_waitcnt vmcnt(2)
	ds_write_b128 v192, v[118:121] offset:4096
	.loc	1 1169 12                       ; 7.ttgir:1169:12
	v_mfma_f32_32x32x8_f16 v[2:17], v[80:81], v[148:149], v[2:17]
	.loc	1 1170 12                       ; 7.ttgir:1170:12
	v_mfma_f32_32x32x8_f16 v[50:65], v[124:125], v[74:75], v[50:65]
	.loc	1 1171 12                       ; 7.ttgir:1171:12
	v_mfma_f32_32x32x8_f16 v[34:49], v[126:127], v[74:75], v[34:49]
	.loc	1 1172 12                       ; 7.ttgir:1172:12
	v_mfma_f32_32x32x8_f16 v[18:33], v[128:129], v[74:75], v[18:33]
	.loc	1 1173 12                       ; 7.ttgir:1173:12
	v_mfma_f32_32x32x8_f16 v[2:17], v[130:131], v[74:75], v[2:17]
	.loc	1 1174 12                       ; 7.ttgir:1174:12
	v_pack_b32_f16 v75, v211, v212
	v_pack_b32_f16 v74, v209, v210
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[132:133], v[74:75], v[50:65]
	.loc	1 1175 12                       ; 7.ttgir:1175:12
	v_mfma_f32_32x32x8_f16 v[34:49], v[134:135], v[74:75], v[34:49]
	.loc	1 1176 12                       ; 7.ttgir:1176:12
	v_mfma_f32_32x32x8_f16 v[18:33], v[136:137], v[74:75], v[18:33]
	.loc	1 1177 12                       ; 7.ttgir:1177:12
	v_mfma_f32_32x32x8_f16 v[2:17], v[138:139], v[74:75], v[2:17]
	.loc	1 1178 12                       ; 7.ttgir:1178:12
	v_pack_b32_f16 v75, v215, v216
	v_pack_b32_f16 v74, v213, v214
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[140:141], v[74:75], v[50:65]
	.loc	1 1179 12                       ; 7.ttgir:1179:12
	v_mfma_f32_32x32x8_f16 v[34:49], v[142:143], v[74:75], v[34:49]
	.loc	1 1180 12                       ; 7.ttgir:1180:12
	v_mfma_f32_32x32x8_f16 v[18:33], v[144:145], v[74:75], v[18:33]
	.loc	1 1181 12                       ; 7.ttgir:1181:12
	v_mfma_f32_32x32x8_f16 v[2:17], v[146:147], v[74:75], v[2:17]
	.loc	1 1194 5                        ; 7.ttgir:1194:5
	s_waitcnt vmcnt(0)
	v_perm_b32 v74, v70, v66, s1
	v_perm_b32 v66, v70, v66, s10
	ds_write_b32 v193, v74 offset:8192
	ds_write_b32 v194, v66 offset:8192
	v_perm_b32 v66, v71, v67, s1
	ds_write_b32 v195, v66 offset:8192
	v_perm_b32 v66, v71, v67, s10
	ds_write_b32 v196, v66 offset:8192
	v_perm_b32 v66, v72, v68, s1
	ds_write_b32 v197, v66 offset:8192
	v_perm_b32 v66, v72, v68, s10
	ds_write_b32 v198, v66 offset:8192
	v_perm_b32 v66, v73, v69, s1
	ds_write_b32 v199, v66 offset:8192
	v_perm_b32 v66, v73, v69, s10
	ds_write_b32 v200, v66 offset:8192
	.loc	1 909 5                         ; 7.ttgir:909:5
	s_cbranch_scc1 .LBB0_13
; %bb.14:                               ; %._crit_edge669.loopexit
	.loc	1 0 5 is_stmt 0                 ; 7.ttgir:0:5
	v_readlane_b32 s10, v227, 2
	v_readlane_b32 s82, v227, 0
	.loc	1 1217 12 is_stmt 1             ; 7.ttgir:1217:12
	s_add_i32 s12, s33, 32
	v_readlane_b32 s11, v227, 3
	v_readlane_b32 s83, v227, 1
	s_branch .LBB0_16
.LBB0_15:
	.loc	1 0 12 is_stmt 0                ; 7.ttgir:0:12
	v_mov_b32_e32 v191, v67
.LBB0_16:                               ; %._crit_edge669
	.loc	1 1202 12 is_stmt 1             ; 7.ttgir:1202:12
	v_lshl_add_u32 v66, v191, 1, 0
	.loc	1 1204 12                       ; 7.ttgir:1204:12
	v_lshl_add_u32 v67, v190, 1, 0
	.loc	1 1200 5                        ; 7.ttgir:1200:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 1202 12                       ; 7.ttgir:1202:12
	ds_read_b128 v[124:127], v66
	.loc	1 1204 12                       ; 7.ttgir:1204:12
	ds_read_b128 v[128:131], v67
	.loc	1 1206 12                       ; 7.ttgir:1206:12
	ds_read_b128 v[132:135], v183
	.loc	1 1208 12                       ; 7.ttgir:1208:12
	ds_read_b128 v[136:139], v184
	.loc	1 1210 12                       ; 7.ttgir:1210:12
	ds_read_b128 v[140:143], v185
	.loc	1 1212 12                       ; 7.ttgir:1212:12
	ds_read_b128 v[144:147], v186
	.loc	1 1214 12                       ; 7.ttgir:1214:12
	ds_read_b128 v[118:121], v187
	.loc	1 1216 12                       ; 7.ttgir:1216:12
	ds_read_b128 v[114:117], v188
	.loc	1 1272 12                       ; 7.ttgir:1272:12
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[110:111], 0
	.loc	1 124 11                        ; 7.ttgir:124:11
	s_add_i32 s0, s52, 0x80
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v148, 1, v1
	.loc	1 1218 12                       ; 7.ttgir:1218:12
	s_cmp_lg_u32 s12, s92
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v149, 2, v1
	.loc	1 1218 12                       ; 7.ttgir:1218:12
	s_cselect_b64 s[42:43], -1, 0
	.loc	1 1219 12                       ; 7.ttgir:1219:12
	s_xor_b64 s[48:49], s[48:49], -1
	.loc	1 1223 12                       ; 7.ttgir:1223:12
	v_or_b32_e32 v183, s33, v148
	.loc	1 1272 12                       ; 7.ttgir:1272:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[112:113], v[66:81]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	s_add_i32 s1, s33, s44
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v150, 3, v1
	.loc	1 1223 12                       ; 7.ttgir:1223:12
	v_or_b32_e32 v184, s33, v149
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[2:3], s19, v183
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[80:81], s[48:49], s[42:43]
	.loc	1 1223 12                       ; 7.ttgir:1223:12
	v_or_b32_e32 v185, s33, v150
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[4:5], s19, v184
	.loc	1 1273 12                       ; 7.ttgir:1273:12
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[106:107], v[66:81]
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[56:57], s[80:81], s[2:3]
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v151, 9, v1
	.loc	1 1226 12                       ; 7.ttgir:1226:12
	v_or_b32_e32 v186, s33, v156
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[6:7], s19, v185
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[58:59], s[80:81], s[4:5]
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v152, 10, v1
	.loc	1 1226 12                       ; 7.ttgir:1226:12
	v_or_b32_e32 v187, s33, v151
	.loc	1 1273 12                       ; 7.ttgir:1273:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[108:109], v[66:81]
	.loc	1 1232 12                       ; 7.ttgir:1232:12
	v_or_b32_e32 v190, s33, v154
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[12:13], s19, v186
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[60:61], s[80:81], s[6:7]
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v153, 11, v1
	.loc	1 1226 12                       ; 7.ttgir:1226:12
	v_or_b32_e32 v110, s33, v152
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[14:15], s19, v187
	v_cmp_gt_i32_e64 s[34:35], s19, v190
	.loc	1 1274 12                       ; 7.ttgir:1274:12
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[102:103], v[66:81]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v102, s1, v149
	v_add_u32_e32 v103, s1, v150
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[4:5], v166, v102
	v_cmp_ge_i32_e64 s[6:7], v166, v103
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[62:63], s[80:81], s[12:13]
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v161, 25, v1
	.loc	1 1226 12                       ; 7.ttgir:1226:12
	v_or_b32_e32 v111, s33, v153
	.loc	1 1274 12                       ; 7.ttgir:1274:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[104:105], v[66:81]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v104, s1, v156
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[12:13], v166, v104
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[20:21], s19, v110
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[64:65], s[80:81], s[14:15]
	s_or_b64 s[78:79], s[80:81], s[34:35]
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v158, 17, v1
	.loc	1 1223 12                       ; 7.ttgir:1223:12
	v_or_b32_e32 v164, s33, v1
	.loc	1 1275 12                       ; 7.ttgir:1275:12
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[98:99], v[66:81]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v99, s1, v148
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[2:3], v166, v99
	.loc	1 1252 12                       ; 7.ttgir:1252:12
	s_and_b64 s[2:3], s[2:3], s[56:57]
	.loc	1 1229 12                       ; 7.ttgir:1229:12
	v_or_b32_e32 v124, s33, v155
	.loc	1 1232 12                       ; 7.ttgir:1232:12
	v_or_b32_e32 v112, s33, v161
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[22:23], s19, v111
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[66:67], s[80:81], s[20:21]
	.loc	1 1275 12                       ; 7.ttgir:1275:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[100:101], v[66:81]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v100, s1, v151
	v_add_u32_e32 v101, s1, v152
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[14:15], v166, v100
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v105, s1, v153
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[20:21], v166, v101
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v159, 18, v1
	.loc	1 1229 12                       ; 7.ttgir:1229:12
	v_or_b32_e32 v125, s33, v158
	.loc	1 1276 12                       ; 7.ttgir:1276:12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[94:95], v[66:81]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v94, s1, v154
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[34:35], v166, v94
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e32 vcc, s19, v164
	v_cmp_gt_i32_e64 s[24:25], s19, v124
	v_cmp_gt_i32_e64 s[36:37], s19, v112
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[68:69], s[80:81], s[22:23]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v98, s1, v1
	.loc	1 1276 12                       ; 7.ttgir:1276:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[96:97], v[66:81]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v106, s1, v155
	v_add_u32_e32 v95, s1, v161
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[22:23], v166, v105
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v160, 19, v1
	.loc	1 1229 12                       ; 7.ttgir:1229:12
	v_or_b32_e32 v188, s33, v159
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[26:27], s19, v125
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[50:51], s[80:81], vcc
	.loc	1 1277 12                       ; 7.ttgir:1277:12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[90:91], v[66:81]
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_mov_b32_e32 v90, 0xff800000
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[70:71], s[80:81], s[24:25]
	s_or_b64 s[48:49], s[80:81], s[36:37]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v107, s1, v158
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e32 vcc, v166, v98
	v_cmp_ge_i32_e64 s[24:25], v166, v106
	v_cmp_ge_i32_e64 s[36:37], v166, v95
	.loc	1 1277 12                       ; 7.ttgir:1277:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[92:93], v[66:81]
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v92, v90, 0, s[2:3]
	.loc	1 1252 12                       ; 7.ttgir:1252:12
	s_and_b64 s[2:3], s[4:5], s[58:59]
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v93, v90, 0, s[2:3]
	.loc	1 1252 12                       ; 7.ttgir:1252:12
	s_and_b64 s[2:3], s[6:7], s[60:61]
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v94, v90, 0, s[2:3]
	.loc	1 1255 12                       ; 7.ttgir:1255:12
	s_and_b64 s[2:3], s[12:13], s[62:63]
	.loc	1 1229 12                       ; 7.ttgir:1229:12
	v_or_b32_e32 v189, s33, v160
	.loc	1 1278 12                       ; 7.ttgir:1278:12
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[86:87], v[66:81]
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v86, v90, 0, s[2:3]
	.loc	1 1255 12                       ; 7.ttgir:1255:12
	s_and_b64 s[2:3], s[14:15], s[64:65]
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v87, v90, 0, s[2:3]
	.loc	1 1255 12                       ; 7.ttgir:1255:12
	s_and_b64 s[2:3], s[20:21], s[66:67]
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v95, v90, 0, s[2:3]
	.loc	1 1255 12                       ; 7.ttgir:1255:12
	s_and_b64 s[2:3], s[22:23], s[68:69]
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[28:29], s19, v188
	.loc	1 1278 12                       ; 7.ttgir:1278:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[88:89], v[66:81]
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[72:73], s[80:81], s[26:27]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v108, s1, v159
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[26:27], v166, v107
	.loc	1 1252 12                       ; 7.ttgir:1252:12
	s_and_b64 s[50:51], vcc, s[50:51]
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v96, v90, 0, s[2:3]
	.loc	1 1258 12                       ; 7.ttgir:1258:12
	s_and_b64 s[2:3], s[24:25], s[70:71]
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[30:31], s19, v189
	.loc	1 1279 12                       ; 7.ttgir:1279:12
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[82:83], v[66:81]
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[74:75], s[80:81], s[28:29]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v109, s1, v160
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[28:29], v166, v108
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v91, v90, 0, s[50:51]
	v_cndmask_b32_e64 v88, v90, 0, s[2:3]
	.loc	1 1258 12                       ; 7.ttgir:1258:12
	s_and_b64 s[2:3], s[26:27], s[72:73]
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v162, 26, v1
	.loc	1 1279 12                       ; 7.ttgir:1279:12
	v_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[84:85], v[66:81]
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[76:77], s[80:81], s[30:31]
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[30:31], v166, v109
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v89, v90, 0, s[2:3]
	.loc	1 1258 12                       ; 7.ttgir:1258:12
	s_and_b64 s[2:3], s[28:29], s[74:75]
	.loc	1 109 11                        ; 7.ttgir:109:11
	v_or_b32_e32 v163, 27, v1
	.loc	1 1232 12                       ; 7.ttgir:1232:12
	v_or_b32_e32 v113, s33, v162
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v97, v90, 0, s[2:3]
	.loc	1 1293 12                       ; 7.ttgir:1293:12
	s_nop 3
	v_fmac_f32_e32 v91, 0x3e0293ee, v66
	v_fmac_f32_e32 v92, 0x3e0293ee, v67
	.loc	1 1258 12                       ; 7.ttgir:1258:12
	s_and_b64 s[2:3], s[30:31], s[76:77]
	.loc	1 1293 12                       ; 7.ttgir:1293:12
	v_fmac_f32_e32 v93, 0x3e0293ee, v68
	v_fmac_f32_e32 v94, 0x3e0293ee, v69
	.loc	1 1303 15                       ; 7.ttgir:1303:15
	v_max_f32_e32 v66, v91, v92
	.loc	1 1232 12                       ; 7.ttgir:1232:12
	v_or_b32_e32 v126, s33, v163
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[38:39], s19, v113
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v110, s1, v162
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v98, v90, 0, s[2:3]
	.loc	1 1261 12                       ; 7.ttgir:1261:12
	s_and_b64 s[2:3], s[34:35], s[78:79]
	.loc	1 1295 12                       ; 7.ttgir:1295:12
	v_fmac_f32_e32 v86, 0x3e0293ee, v70
	v_fmac_f32_e32 v87, 0x3e0293ee, v71
	.loc	1 1303 15                       ; 7.ttgir:1303:15
	v_max3_f32 v66, v66, v93, v94
	.loc	1 1234 12                       ; 7.ttgir:1234:12
	v_cmp_gt_i32_e64 s[40:41], s19, v126
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[42:43], s[80:81], s[38:39]
	.loc	1 1238 12                       ; 7.ttgir:1238:12
	v_add_u32_e32 v111, s1, v163
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[38:39], v166, v110
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v82, v90, 0, s[2:3]
	.loc	1 1261 12                       ; 7.ttgir:1261:12
	s_and_b64 s[2:3], s[36:37], s[48:49]
	.loc	1 1295 12                       ; 7.ttgir:1295:12
	v_fmac_f32_e32 v95, 0x3e0293ee, v72
	v_fmac_f32_e32 v96, 0x3e0293ee, v73
	.loc	1 1303 15                       ; 7.ttgir:1303:15
	v_max3_f32 v66, v66, v86, v87
	.loc	1 1236 12                       ; 7.ttgir:1236:12
	s_or_b64 s[80:81], s[80:81], s[40:41]
	.loc	1 1249 12                       ; 7.ttgir:1249:12
	v_cmp_ge_i32_e64 s[40:41], v166, v111
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v83, v90, 0, s[2:3]
	.loc	1 1261 12                       ; 7.ttgir:1261:12
	s_and_b64 s[2:3], s[38:39], s[42:43]
	.loc	1 1297 12                       ; 7.ttgir:1297:12
	v_fmac_f32_e32 v88, 0x3e0293ee, v74
	v_fmac_f32_e32 v89, 0x3e0293ee, v75
	.loc	1 1303 15                       ; 7.ttgir:1303:15
	v_max3_f32 v66, v66, v95, v96
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v99, v90, 0, s[2:3]
	.loc	1 1261 12                       ; 7.ttgir:1261:12
	s_and_b64 s[2:3], s[40:41], s[80:81]
	.loc	1 1297 12                       ; 7.ttgir:1297:12
	v_fmac_f32_e32 v97, 0x3e0293ee, v76
	v_fmac_f32_e32 v98, 0x3e0293ee, v77
	.loc	1 1303 15                       ; 7.ttgir:1303:15
	v_max3_f32 v66, v66, v88, v89
	.loc	1 1263 12                       ; 7.ttgir:1263:12
	v_cndmask_b32_e64 v90, v90, 0, s[2:3]
	.loc	1 1299 12                       ; 7.ttgir:1299:12
	v_fmac_f32_e32 v82, 0x3e0293ee, v78
	v_fmac_f32_e32 v83, 0x3e0293ee, v79
	.loc	1 1303 15                       ; 7.ttgir:1303:15
	v_max3_f32 v66, v66, v97, v98
	.loc	1 1299 12                       ; 7.ttgir:1299:12
	v_fmac_f32_e32 v99, 0x3e0293ee, v80
	v_fmac_f32_e32 v90, 0x3e0293ee, v81
	.loc	1 1303 15                       ; 7.ttgir:1303:15
	v_max3_f32 v66, v66, v82, v83
	v_max3_f32 v66, v66, v99, v90
	.loc	1 1301 12                       ; 7.ttgir:1301:12
	ds_bpermute_b32 v67, v157, v66
	.loc	1 1513 13                       ; 7.ttgir:1513:13
	s_cmp_le_i32 s44, s52
	s_mov_b64 s[2:3], -1
	.loc	1 1306 12                       ; 7.ttgir:1306:12
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v66, v123, v66, v67
	.loc	1 1310 12                       ; 7.ttgir:1310:12
	v_sub_f32_e32 v68, v92, v66
	v_sub_f32_e32 v67, v91, v66
	v_sub_f32_e32 v69, v93, v66
	v_sub_f32_e32 v70, v94, v66
	.loc	1 1317 12                       ; 7.ttgir:1317:12
	v_exp_f32_e32 v78, v68
	.loc	1 1327 12                       ; 7.ttgir:1327:12
	v_sub_f32_e32 v68, v123, v66
	.loc	1 1317 12                       ; 7.ttgir:1317:12
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v79, v69
	v_exp_f32_e32 v80, v70
	.loc	1 1328 12                       ; 7.ttgir:1328:12
	v_exp_f32_e32 v84, v68
	.loc	1 1381 13                       ; 7.ttgir:1381:13
	ds_read_b64 v[68:69], v168 offset:8192
	.loc	1 1414 13                       ; 7.ttgir:1414:13
	v_cvt_f16_f32_e32 v76, v67
	v_cvt_f16_f32_e32 v91, v78
	v_cvt_f16_f32_e32 v77, v79
	v_cvt_f16_f32_e32 v92, v80
	.loc	1 1312 12                       ; 7.ttgir:1312:12
	v_sub_f32_e32 v71, v86, v66
	v_sub_f32_e32 v72, v87, v66
	v_sub_f32_e32 v73, v95, v66
	v_sub_f32_e32 v74, v96, v66
	.loc	1 1318 12                       ; 7.ttgir:1318:12
	v_exp_f32_e32 v81, v71
	v_exp_f32_e32 v85, v72
	v_exp_f32_e32 v86, v73
	v_exp_f32_e32 v87, v74
	.loc	1 1383 13                       ; 7.ttgir:1383:13
	ds_read_b64 v[70:71], v167 offset:8192
	.loc	1 1385 13                       ; 7.ttgir:1385:13
	ds_read_b64 v[72:73], v169 offset:8192
	.loc	1 1387 13                       ; 7.ttgir:1387:13
	ds_read_b64 v[74:75], v170 offset:8192
	.loc	1 1333 12                       ; 7.ttgir:1333:12
	v_mul_f32_e32 v50, v50, v84
	v_mul_f32_e32 v51, v51, v84
	v_mul_f32_e32 v52, v52, v84
	v_mul_f32_e32 v53, v53, v84
	.loc	1 1336 12                       ; 7.ttgir:1336:12
	v_mul_f32_e32 v54, v54, v84
	v_mul_f32_e32 v55, v55, v84
	v_mul_f32_e32 v56, v56, v84
	v_mul_f32_e32 v57, v57, v84
	.loc	1 1339 12                       ; 7.ttgir:1339:12
	v_mul_f32_e32 v58, v58, v84
	v_mul_f32_e32 v59, v59, v84
	v_mul_f32_e32 v60, v60, v84
	v_mul_f32_e32 v61, v61, v84
	.loc	1 1342 12                       ; 7.ttgir:1342:12
	v_mul_f32_e32 v62, v62, v84
	v_mul_f32_e32 v63, v63, v84
	v_mul_f32_e32 v64, v64, v84
	v_mul_f32_e32 v65, v65, v84
	.loc	1 1426 13                       ; 7.ttgir:1426:13
	v_pack_b32_f16 v77, v77, v92
	v_pack_b32_f16 v76, v76, v91
	.loc	1 1345 12                       ; 7.ttgir:1345:12
	v_mul_f32_e32 v34, v34, v84
	v_mul_f32_e32 v35, v35, v84
	v_mul_f32_e32 v36, v36, v84
	v_mul_f32_e32 v37, v37, v84
	.loc	1 1348 12                       ; 7.ttgir:1348:12
	v_mul_f32_e32 v38, v38, v84
	v_mul_f32_e32 v39, v39, v84
	v_mul_f32_e32 v40, v40, v84
	v_mul_f32_e32 v41, v41, v84
	.loc	1 1351 12                       ; 7.ttgir:1351:12
	v_mul_f32_e32 v42, v42, v84
	v_mul_f32_e32 v43, v43, v84
	v_mul_f32_e32 v44, v44, v84
	v_mul_f32_e32 v45, v45, v84
	.loc	1 1354 12                       ; 7.ttgir:1354:12
	v_mul_f32_e32 v46, v46, v84
	v_mul_f32_e32 v47, v47, v84
	v_mul_f32_e32 v48, v48, v84
	v_mul_f32_e32 v49, v49, v84
	.loc	1 1426 13                       ; 7.ttgir:1426:13
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]
	.loc	1 1389 13                       ; 7.ttgir:1389:13
	ds_read_b64 v[68:69], v172 offset:8192
	.loc	1 1357 12                       ; 7.ttgir:1357:12
	v_mul_f32_e32 v18, v18, v84
	v_mul_f32_e32 v19, v19, v84
	v_mul_f32_e32 v20, v20, v84
	v_mul_f32_e32 v21, v21, v84
	.loc	1 1360 12                       ; 7.ttgir:1360:12
	v_mul_f32_e32 v22, v22, v84
	v_mul_f32_e32 v23, v23, v84
	.loc	1 1427 13                       ; 7.ttgir:1427:13
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]
	.loc	1 1360 12                       ; 7.ttgir:1360:12
	v_mul_f32_e32 v24, v24, v84
	v_mul_f32_e32 v25, v25, v84
	.loc	1 1363 12                       ; 7.ttgir:1363:12
	v_mul_f32_e32 v26, v26, v84
	v_mul_f32_e32 v27, v27, v84
	v_mul_f32_e32 v28, v28, v84
	v_mul_f32_e32 v29, v29, v84
	.loc	1 1366 13                       ; 7.ttgir:1366:13
	v_mul_f32_e32 v30, v30, v84
	v_mul_f32_e32 v31, v31, v84
	v_mul_f32_e32 v32, v32, v84
	v_mul_f32_e32 v33, v33, v84
	.loc	1 1369 13                       ; 7.ttgir:1369:13
	v_mul_f32_e32 v2, v2, v84
	v_mul_f32_e32 v3, v3, v84
	v_mul_f32_e32 v4, v4, v84
	v_mul_f32_e32 v5, v5, v84
	.loc	1 1372 13                       ; 7.ttgir:1372:13
	v_mul_f32_e32 v6, v6, v84
	v_mul_f32_e32 v7, v7, v84
	v_mul_f32_e32 v8, v8, v84
	v_mul_f32_e32 v9, v9, v84
	.loc	1 1375 13                       ; 7.ttgir:1375:13
	v_mul_f32_e32 v10, v10, v84
	v_mul_f32_e32 v11, v11, v84
	v_mul_f32_e32 v12, v12, v84
	v_mul_f32_e32 v13, v13, v84
	.loc	1 1378 13                       ; 7.ttgir:1378:13
	v_mul_f32_e32 v14, v14, v84
	v_mul_f32_e32 v15, v15, v84
	v_mul_f32_e32 v16, v16, v84
	v_mul_f32_e32 v17, v17, v84
	.loc	1 1428 13                       ; 7.ttgir:1428:13
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]
	.loc	1 1415 13                       ; 7.ttgir:1415:13
	v_cvt_f16_f32_e32 v91, v81
	v_cvt_f16_f32_e32 v92, v85
	.loc	1 1324 15                       ; 7.ttgir:1324:15
	v_add_f32_e32 v67, v67, v78
	v_add_f32_e32 v67, v79, v67
	v_add_f32_e32 v67, v80, v67
	v_add_f32_e32 v67, v81, v67
	v_add_f32_e32 v67, v85, v67
	.loc	1 1429 13                       ; 7.ttgir:1429:13
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]
	.loc	1 1415 13                       ; 7.ttgir:1415:13
	v_cvt_f16_f32_e32 v76, v86
	v_cvt_f16_f32_e32 v77, v87
	.loc	1 1391 13                       ; 7.ttgir:1391:13
	ds_read_b64 v[70:71], v171 offset:8192
	.loc	1 1393 13                       ; 7.ttgir:1393:13
	ds_read_b64 v[72:73], v173 offset:8192
	.loc	1 1395 13                       ; 7.ttgir:1395:13
	ds_read_b64 v[74:75], v174 offset:8192
	.loc	1 1324 15                       ; 7.ttgir:1324:15
	v_add_f32_e32 v67, v86, v67
	v_add_f32_e32 v67, v87, v67
	.loc	1 1430 13                       ; 7.ttgir:1430:13
	v_pack_b32_f16 v77, v76, v77
	v_pack_b32_f16 v76, v91, v92
	s_waitcnt lgkmcnt(3)
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]
	.loc	1 1314 12                       ; 7.ttgir:1314:12
	v_sub_f32_e32 v68, v88, v66
	v_sub_f32_e32 v69, v89, v66
	.loc	1 1319 12                       ; 7.ttgir:1319:12
	v_exp_f32_e32 v88, v68
	v_exp_f32_e32 v89, v69
	.loc	1 1397 13                       ; 7.ttgir:1397:13
	ds_read_b64 v[68:69], v177 offset:8192
	.loc	1 1416 13                       ; 7.ttgir:1416:13
	v_cvt_f16_f32_e32 v93, v88
	.loc	1 1431 13                       ; 7.ttgir:1431:13
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]
	.loc	1 1314 12                       ; 7.ttgir:1314:12
	v_sub_f32_e32 v70, v97, v66
	v_sub_f32_e32 v71, v98, v66
	.loc	1 1319 12                       ; 7.ttgir:1319:12
	v_exp_f32_e32 v91, v70
	v_exp_f32_e32 v92, v71
	.loc	1 1416 13                       ; 7.ttgir:1416:13
	v_cvt_f16_f32_e32 v94, v89
	.loc	1 1324 15                       ; 7.ttgir:1324:15
	v_add_f32_e32 v67, v88, v67
	v_add_f32_e32 v67, v89, v67
	.loc	1 1432 13                       ; 7.ttgir:1432:13
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]
	.loc	1 1324 15                       ; 7.ttgir:1324:15
	v_add_f32_e32 v67, v91, v67
	v_add_f32_e32 v67, v92, v67
	.loc	1 1433 13                       ; 7.ttgir:1433:13
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]
	.loc	1 1416 13                       ; 7.ttgir:1416:13
	v_cvt_f16_f32_e32 v76, v91
	v_cvt_f16_f32_e32 v77, v92
	.loc	1 1399 13                       ; 7.ttgir:1399:13
	ds_read_b64 v[70:71], v175 offset:8192
	.loc	1 1401 13                       ; 7.ttgir:1401:13
	ds_read_b64 v[72:73], v179 offset:8192
	.loc	1 1403 13                       ; 7.ttgir:1403:13
	ds_read_b64 v[74:75], v181 offset:8192
	.loc	1 1434 13                       ; 7.ttgir:1434:13
	v_pack_b32_f16 v77, v76, v77
	v_pack_b32_f16 v76, v93, v94
	s_waitcnt lgkmcnt(3)
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]
	.loc	1 1316 12                       ; 7.ttgir:1316:12
	v_sub_f32_e32 v68, v82, v66
	v_sub_f32_e32 v69, v83, v66
	.loc	1 1320 12                       ; 7.ttgir:1320:12
	v_exp_f32_e32 v82, v68
	v_exp_f32_e32 v83, v69
	.loc	1 1405 13                       ; 7.ttgir:1405:13
	ds_read_b64 v[68:69], v178 offset:8192
	.loc	1 1417 13                       ; 7.ttgir:1417:13
	v_cvt_f16_f32_e32 v78, v82
	.loc	1 1435 13                       ; 7.ttgir:1435:13
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]
	.loc	1 1316 12                       ; 7.ttgir:1316:12
	v_sub_f32_e32 v70, v99, v66
	v_sub_f32_e32 v71, v90, v66
	.loc	1 1320 12                       ; 7.ttgir:1320:12
	v_exp_f32_e32 v90, v70
	v_exp_f32_e32 v93, v71
	.loc	1 1417 13                       ; 7.ttgir:1417:13
	v_cvt_f16_f32_e32 v94, v83
	.loc	1 1324 15                       ; 7.ttgir:1324:15
	v_add_f32_e32 v67, v82, v67
	v_add_f32_e32 v67, v83, v67
	.loc	1 1436 13                       ; 7.ttgir:1436:13
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]
	.loc	1 1324 15                       ; 7.ttgir:1324:15
	v_add_f32_e32 v67, v90, v67
	v_add_f32_e32 v67, v93, v67
	.loc	1 1437 13                       ; 7.ttgir:1437:13
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]
	.loc	1 1417 13                       ; 7.ttgir:1417:13
	v_cvt_f16_f32_e32 v76, v90
	v_cvt_f16_f32_e32 v77, v93
	.loc	1 1407 13                       ; 7.ttgir:1407:13
	ds_read_b64 v[70:71], v176 offset:8192
	.loc	1 1409 13                       ; 7.ttgir:1409:13
	ds_read_b64 v[72:73], v180 offset:8192
	.loc	1 1411 13                       ; 7.ttgir:1411:13
	ds_read_b64 v[74:75], v182 offset:8192
	.loc	1 1438 13                       ; 7.ttgir:1438:13
	v_pack_b32_f16 v77, v76, v77
	v_pack_b32_f16 v76, v78, v94
	s_waitcnt lgkmcnt(3)
	s_nop 0
	v_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]
	.loc	1 1322 12                       ; 7.ttgir:1322:12
	ds_bpermute_b32 v68, v157, v67
	.loc	1 1324 15                       ; 7.ttgir:1324:15
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v79, v67, v68
	.loc	1 1413 13                       ; 7.ttgir:1413:13
	v_fmac_f32_e32 v79, v122, v84
	.loc	1 1439 13                       ; 7.ttgir:1439:13
	v_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]
	.loc	1 1446 13                       ; 7.ttgir:1446:13
	v_div_scale_f32 v67, s[4:5], v79, v79, 1.0
	v_rcp_f32_e32 v67, v67
	v_div_scale_f32 v68, vcc, 1.0, v79, 1.0
	.loc	1 1513 13                       ; 7.ttgir:1513:13
	s_cselect_b64 s[4:5], -1, 0
	.loc	1 1446 13                       ; 7.ttgir:1446:13
	v_mul_f32_e32 v67, v68, v67
	.loc	1 1440 13                       ; 7.ttgir:1440:13
	v_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]
	.loc	1 1514 13                       ; 7.ttgir:1514:13
	s_cmp_ge_i32 s44, s0
	.loc	1 1446 13                       ; 7.ttgir:1446:13
	v_div_fmas_f32 v67, 0, 0, v67
	.loc	1 1514 13                       ; 7.ttgir:1514:13
	s_cselect_b64 s[6:7], -1, 0
	.loc	1 1446 13                       ; 7.ttgir:1446:13
	v_div_fixup_f32 v80, v67, v79, 1.0
	.loc	1 1516 5                        ; 7.ttgir:1516:5
	s_or_b64 s[4:5], s[4:5], s[6:7]
	.loc	1 1496 13                       ; 7.ttgir:1496:13
	v_fma_mixlo_f16 v78, v53, v80, 0
	.loc	1 1497 13                       ; 7.ttgir:1497:13
	v_fma_mixlo_f16 v69, v54, v80, 0
	.loc	1 1441 13                       ; 7.ttgir:1441:13
	v_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]
	.loc	1 1496 13                       ; 7.ttgir:1496:13
	v_fma_mixlo_f16 v75, v50, v80, 0
	v_fma_mixlo_f16 v76, v51, v80, 0
	v_fma_mixlo_f16 v77, v52, v80, 0
	.loc	1 1497 13                       ; 7.ttgir:1497:13
	v_fma_mixlo_f16 v71, v55, v80, 0
	v_fma_mixlo_f16 v73, v56, v80, 0
	v_fma_mixlo_f16 v74, v57, v80, 0
	.loc	1 1498 13                       ; 7.ttgir:1498:13
	v_fma_mixlo_f16 v67, v58, v80, 0
	v_fma_mixlo_f16 v68, v59, v80, 0
	v_fma_mixlo_f16 v70, v60, v80, 0
	v_fma_mixlo_f16 v72, v61, v80, 0
	.loc	1 1499 13                       ; 7.ttgir:1499:13
	v_fma_mixlo_f16 v60, v62, v80, 0
	v_fma_mixlo_f16 v62, v63, v80, 0
	v_fma_mixlo_f16 v64, v64, v80, 0
	v_fma_mixlo_f16 v65, v65, v80, 0
	.loc	1 1500 13                       ; 7.ttgir:1500:13
	v_fma_mixlo_f16 v56, v34, v80, 0
	v_fma_mixlo_f16 v58, v35, v80, 0
	v_fma_mixlo_f16 v61, v36, v80, 0
	v_fma_mixlo_f16 v63, v37, v80, 0
	.loc	1 1501 13                       ; 7.ttgir:1501:13
	v_fma_mixlo_f16 v52, v38, v80, 0
	v_fma_mixlo_f16 v54, v39, v80, 0
	v_fma_mixlo_f16 v57, v40, v80, 0
	v_fma_mixlo_f16 v59, v41, v80, 0
	.loc	1 1502 13                       ; 7.ttgir:1502:13
	v_fma_mixlo_f16 v50, v42, v80, 0
	v_fma_mixlo_f16 v51, v43, v80, 0
	v_fma_mixlo_f16 v53, v44, v80, 0
	v_fma_mixlo_f16 v55, v45, v80, 0
	.loc	1 1503 13                       ; 7.ttgir:1503:13
	v_fma_mixlo_f16 v44, v46, v80, 0
	v_fma_mixlo_f16 v46, v47, v80, 0
	v_fma_mixlo_f16 v48, v48, v80, 0
	v_fma_mixlo_f16 v49, v49, v80, 0
	.loc	1 1504 13                       ; 7.ttgir:1504:13
	v_fma_mixlo_f16 v40, v18, v80, 0
	v_fma_mixlo_f16 v42, v19, v80, 0
	v_fma_mixlo_f16 v45, v20, v80, 0
	v_fma_mixlo_f16 v47, v21, v80, 0
	.loc	1 1505 13                       ; 7.ttgir:1505:13
	v_fma_mixlo_f16 v36, v22, v80, 0
	v_fma_mixlo_f16 v38, v23, v80, 0
	v_fma_mixlo_f16 v41, v24, v80, 0
	v_fma_mixlo_f16 v43, v25, v80, 0
	.loc	1 1506 13                       ; 7.ttgir:1506:13
	v_fma_mixlo_f16 v34, v26, v80, 0
	v_fma_mixlo_f16 v35, v27, v80, 0
	v_fma_mixlo_f16 v37, v28, v80, 0
	v_fma_mixlo_f16 v39, v29, v80, 0
	.loc	1 1507 13                       ; 7.ttgir:1507:13
	v_fma_mixlo_f16 v24, v30, v80, 0
	v_fma_mixlo_f16 v26, v31, v80, 0
	v_fma_mixlo_f16 v28, v32, v80, 0
	v_fma_mixlo_f16 v29, v33, v80, 0
	.loc	1 1508 13                       ; 7.ttgir:1508:13
	v_fma_mixlo_f16 v20, v2, v80, 0
	v_fma_mixlo_f16 v22, v3, v80, 0
	v_fma_mixlo_f16 v25, v4, v80, 0
	v_fma_mixlo_f16 v27, v5, v80, 0
	.loc	1 1509 13                       ; 7.ttgir:1509:13
	v_fma_mixlo_f16 v18, v6, v80, 0
	v_fma_mixlo_f16 v19, v7, v80, 0
	v_fma_mixlo_f16 v21, v8, v80, 0
	v_fma_mixlo_f16 v23, v9, v80, 0
	.loc	1 1510 13                       ; 7.ttgir:1510:13
	v_fma_mixlo_f16 v4, v10, v80, 0
	v_fma_mixlo_f16 v6, v11, v80, 0
	v_fma_mixlo_f16 v8, v12, v80, 0
	v_fma_mixlo_f16 v9, v13, v80, 0
	.loc	1 1511 13                       ; 7.ttgir:1511:13
	v_fma_mixlo_f16 v2, v14, v80, 0
	v_fma_mixlo_f16 v3, v15, v80, 0
	v_fma_mixlo_f16 v5, v16, v80, 0
	v_fma_mixlo_f16 v7, v17, v80, 0
	.loc	1 1516 5                        ; 7.ttgir:1516:5
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_18
; %bb.17:
	.loc	1 1520 13                       ; 7.ttgir:1520:13
	v_cmp_gt_i32_e32 vcc, s44, v166
	.loc	1 1522 13                       ; 7.ttgir:1522:13
	s_nop 1
	v_cndmask_b32_e64 v75, v75, 0, vcc
	v_cndmask_b32_e64 v76, v76, 0, vcc
	v_cndmask_b32_e64 v77, v77, 0, vcc
	v_cndmask_b32_e64 v78, v78, 0, vcc
	v_cndmask_b32_e64 v69, v69, 0, vcc
	v_cndmask_b32_e64 v71, v71, 0, vcc
	v_cndmask_b32_e64 v73, v73, 0, vcc
	v_cndmask_b32_e64 v74, v74, 0, vcc
	v_cndmask_b32_e64 v67, v67, 0, vcc
	v_cndmask_b32_e64 v68, v68, 0, vcc
	v_cndmask_b32_e64 v70, v70, 0, vcc
	v_cndmask_b32_e64 v72, v72, 0, vcc
	v_cndmask_b32_e64 v60, v60, 0, vcc
	v_cndmask_b32_e64 v62, v62, 0, vcc
	v_cndmask_b32_e64 v64, v64, 0, vcc
	v_cndmask_b32_e64 v65, v65, 0, vcc
	v_cndmask_b32_e64 v56, v56, 0, vcc
	v_cndmask_b32_e64 v58, v58, 0, vcc
	v_cndmask_b32_e64 v61, v61, 0, vcc
	v_cndmask_b32_e64 v63, v63, 0, vcc
	v_cndmask_b32_e64 v52, v52, 0, vcc
	v_cndmask_b32_e64 v54, v54, 0, vcc
	v_cndmask_b32_e64 v57, v57, 0, vcc
	v_cndmask_b32_e64 v59, v59, 0, vcc
	v_cndmask_b32_e64 v50, v50, 0, vcc
	v_cndmask_b32_e64 v51, v51, 0, vcc
	v_cndmask_b32_e64 v53, v53, 0, vcc
	v_cndmask_b32_e64 v55, v55, 0, vcc
	v_cndmask_b32_e64 v44, v44, 0, vcc
	v_cndmask_b32_e64 v46, v46, 0, vcc
	v_cndmask_b32_e64 v48, v48, 0, vcc
	v_cndmask_b32_e64 v49, v49, 0, vcc
	v_cndmask_b32_e64 v40, v40, 0, vcc
	v_cndmask_b32_e64 v42, v42, 0, vcc
	v_cndmask_b32_e64 v45, v45, 0, vcc
	v_cndmask_b32_e64 v47, v47, 0, vcc
	v_cndmask_b32_e64 v36, v36, 0, vcc
	v_cndmask_b32_e64 v38, v38, 0, vcc
	v_cndmask_b32_e64 v41, v41, 0, vcc
	v_cndmask_b32_e64 v43, v43, 0, vcc
	v_cndmask_b32_e64 v34, v34, 0, vcc
	v_cndmask_b32_e64 v35, v35, 0, vcc
	v_cndmask_b32_e64 v37, v37, 0, vcc
	v_cndmask_b32_e64 v39, v39, 0, vcc
	v_cndmask_b32_e64 v24, v24, 0, vcc
	v_cndmask_b32_e64 v26, v26, 0, vcc
	v_cndmask_b32_e64 v28, v28, 0, vcc
	v_cndmask_b32_e64 v29, v29, 0, vcc
	v_cndmask_b32_e64 v20, v20, 0, vcc
	v_cndmask_b32_e64 v22, v22, 0, vcc
	v_cndmask_b32_e64 v25, v25, 0, vcc
	v_cndmask_b32_e64 v27, v27, 0, vcc
	v_cndmask_b32_e64 v18, v18, 0, vcc
	v_cndmask_b32_e64 v19, v19, 0, vcc
	v_cndmask_b32_e64 v21, v21, 0, vcc
	v_cndmask_b32_e64 v23, v23, 0, vcc
	v_cndmask_b32_e64 v4, v4, 0, vcc
	v_cndmask_b32_e64 v6, v6, 0, vcc
	v_cndmask_b32_e64 v8, v8, 0, vcc
	v_cndmask_b32_e64 v9, v9, 0, vcc
	v_cndmask_b32_e64 v2, v2, 0, vcc
	v_cndmask_b32_e64 v3, v3, 0, vcc
	v_cndmask_b32_e64 v5, v5, 0, vcc
	v_cndmask_b32_e64 v7, v7, 0, vcc
.LBB0_18:
	.loc	1 1526 13                       ; 7.ttgir:1526:13
	s_mul_i32 s4, s18, 0x3d640
	.loc	1 1527 13                       ; 7.ttgir:1527:13
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s1, s8, s4
	.loc	1 1528 13                       ; 7.ttgir:1528:13
	s_mul_i32 s4, s17, 0x3d64
	.loc	1 1527 13                       ; 7.ttgir:1527:13
	s_addc_u32 s6, s9, s5
	.loc	1 1529 13                       ; 7.ttgir:1529:13
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s1, s1, s4
	s_addc_u32 s6, s6, s5
	.loc	1 1530 13                       ; 7.ttgir:1530:13
	s_ashr_i32 s53, s52, 31
	s_lshl_b64 s[4:5], s[52:53], 2
	s_add_u32 s4, s1, s4
	s_addc_u32 s1, s6, s5
	.loc	1 1531 13                       ; 7.ttgir:1531:13
	s_sub_i32 s0, s0, s16
	.loc	1 1532 13                       ; 7.ttgir:1532:13
	s_cmp_lt_i32 s0, 1
	.loc	1 103 10                        ; 7.ttgir:103:10
	v_and_b32_e32 v11, 0x7f, v0
	.loc	1 1532 13                       ; 7.ttgir:1532:13
	s_cselect_b64 s[8:9], -1, 0
	.loc	1 1533 5                        ; 7.ttgir:1533:5
	s_and_b64 vcc, exec, s[8:9]
	v_lshl_add_u32 v10, v165, 2, 0
	v_lshlrev_b32_e32 v0, 2, v11
	s_cbranch_vccnz .LBB0_20
; %bb.19:
	.loc	1 0 5 is_stmt 0                 ; 7.ttgir:0:5
	s_mov_b32 s2, 0x800000
	.loc	1 1539 13 is_stmt 1             ; 7.ttgir:1539:13
	v_cmp_gt_f32_e32 vcc, s2, v79
	v_mov_b32_e32 v12, 0x42000000
	.loc	1 1536 13                       ; 7.ttgir:1536:13
	s_sub_i32 s0, 0x80, s0
	.loc	1 1539 13                       ; 7.ttgir:1539:13
	v_cndmask_b32_e64 v13, 0, 32, vcc
	v_ldexp_f32 v13, v79, v13
	v_log_f32_e32 v13, v13
	v_cndmask_b32_e32 v12, 0, v12, vcc
	.loc	1 1538 13                       ; 7.ttgir:1538:13
	v_cmp_gt_i32_e32 vcc, s0, v11
	.loc	1 1539 13                       ; 7.ttgir:1539:13
	v_sub_f32_e32 v11, v13, v12
	.loc	1 1540 13                       ; 7.ttgir:1540:13
	v_add_f32_e32 v11, v66, v11
	.loc	1 1541 5                        ; 7.ttgir:1541:5
	s_barrier
	.loc	1 1542 13                       ; 7.ttgir:1542:13
	ds_write_b32 v10, v11
	v_add_u32_e32 v11, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v11, v11
	.loc	1 1543 5                        ; 7.ttgir:1543:5
	v_bfrev_b32_e32 v12, 1
	s_and_b64 vcc, s[82:83], vcc
	s_and_b32 s5, s1, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e32 v12, v12, v0, vcc
	s_mov_b64 s[2:3], 0
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v11, v12, s[4:7], 0 offen
.LBB0_20:                               ; %Flow
	.loc	1 1533 5                        ; 7.ttgir:1533:5
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_22
; %bb.21:
	.loc	1 0 5 is_stmt 0                 ; 7.ttgir:0:5
	s_mov_b32 s0, 0x800000
	.loc	1 1547 13 is_stmt 1             ; 7.ttgir:1547:13
	v_cmp_gt_f32_e32 vcc, s0, v79
	v_mov_b32_e32 v11, 0x42000000
	s_nop 0
	v_cndmask_b32_e64 v12, 0, 32, vcc
	v_ldexp_f32 v12, v79, v12
	v_log_f32_e32 v12, v12
	v_cndmask_b32_e32 v11, 0, v11, vcc
	.loc	1 1549 5                        ; 7.ttgir:1549:5
	s_barrier
	.loc	1 1547 13                       ; 7.ttgir:1547:13
	v_sub_f32_e32 v11, v12, v11
	.loc	1 1548 13                       ; 7.ttgir:1548:13
	v_add_f32_e32 v11, v66, v11
	.loc	1 1550 13                       ; 7.ttgir:1550:13
	ds_write_b32 v10, v11
	v_add_u32_e32 v10, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v10, v10
	.loc	1 1551 5                        ; 7.ttgir:1551:5
	v_bfrev_b32_e32 v11, 1
	s_and_b32 s5, s1, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_cndmask_b32_e64 v0, v11, v0, s[82:83]
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v10, v0, s[4:7], 0 offen
.LBB0_22:
	.loc	1 1555 13                       ; 7.ttgir:1555:13
	s_mul_i32 s0, s45, s18
	.loc	1 1556 13                       ; 7.ttgir:1556:13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s10, s0
	.loc	1 1557 13                       ; 7.ttgir:1557:13
	s_mul_i32 s0, s46, s17
	.loc	1 1556 13                       ; 7.ttgir:1556:13
	s_addc_u32 s3, s11, s1
	.loc	1 1558 13                       ; 7.ttgir:1558:13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	.loc	1 1559 13                       ; 7.ttgir:1559:13
	s_mul_i32 s0, s54, s47
	.loc	1 1558 13                       ; 7.ttgir:1558:13
	s_addc_u32 s3, s3, s1
	.loc	1 1560 13                       ; 7.ttgir:1560:13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s2, s0
	.loc	1 1562 13                       ; 7.ttgir:1562:13
	s_mul_i32 s0, s47, s52
	.loc	1 1560 13                       ; 7.ttgir:1560:13
	s_addc_u32 s3, s3, s1
	.loc	1 1565 13                       ; 7.ttgir:1565:13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	.loc	1 1651 5                        ; 7.ttgir:1651:5
	s_and_b32 s2, s47, 0x3fff
	.loc	1 189 11                        ; 7.ttgir:189:11
	v_cmp_gt_i32_e32 vcc, s16, v166
	.loc	1 1564 13                       ; 7.ttgir:1564:13
	v_mul_lo_u32 v12, s47, v165
	.loc	1 1651 5                        ; 7.ttgir:1651:5
	s_bitset1_b32 s2, 14
	s_and_b32 s1, s1, 0xffff
	s_lshl_b32 s2, s2, 16
	s_mov_b32 s4, 0x5040100
	.loc	1 1615 13                       ; 7.ttgir:1615:13
	v_add_lshl_u32 v13, v12, v1, 1
	.loc	1 1651 5                        ; 7.ttgir:1651:5
	v_bfrev_b32_e32 v14, 1
	.loc	1 1650 13                       ; 7.ttgir:1650:13
	s_or_b64 vcc, s[8:9], vcc
	.loc	1 1651 5                        ; 7.ttgir:1651:5
	s_or_b32 s1, s1, s2
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_perm_b32 v11, v78, v77, s4
	v_perm_b32 v10, v76, v75, s4
	v_cndmask_b32_e32 v0, v14, v13, vcc
	buffer_store_dwordx2 v[10:11], v0, s[0:3], 0 offen
	v_add_lshl_u32 v10, v12, v156, 1
	v_perm_b32 v1, v74, v73, s4
	v_perm_b32 v0, v71, v69, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v12, v155, 1
	v_perm_b32 v1, v72, v70, s4
	v_perm_b32 v0, v68, v67, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_lshl_u32 v10, v12, v154, 1
	v_perm_b32 v1, v65, v64, s4
	v_perm_b32 v0, v62, v60, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 64, v13
	v_perm_b32 v1, v63, v61, s4
	v_perm_b32 v0, v58, v56, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0x50, v13
	v_perm_b32 v1, v59, v57, s4
	v_perm_b32 v0, v54, v52, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0x60, v13
	v_perm_b32 v1, v55, v53, s4
	v_perm_b32 v0, v51, v50, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0x70, v13
	v_perm_b32 v1, v49, v48, s4
	v_perm_b32 v0, v46, v44, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0x80, v13
	v_perm_b32 v1, v47, v45, s4
	v_perm_b32 v0, v42, v40, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0x90, v13
	v_perm_b32 v1, v43, v41, s4
	v_perm_b32 v0, v38, v36, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0xa0, v13
	v_perm_b32 v1, v39, v37, s4
	v_perm_b32 v0, v35, v34, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0xb0, v13
	v_perm_b32 v1, v29, v28, s4
	v_perm_b32 v0, v26, v24, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0xc0, v13
	v_perm_b32 v1, v27, v25, s4
	v_perm_b32 v0, v22, v20, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_add_u32_e32 v10, 0xd0, v13
	v_perm_b32 v1, v23, v21, s4
	v_perm_b32 v0, v19, v18, s4
	v_cndmask_b32_e32 v10, v14, v10, vcc
	buffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen
	v_perm_b32 v0, v6, v4, s4
	v_add_u32_e32 v4, 0xe0, v13
	v_perm_b32 v1, v9, v8, s4
	v_cndmask_b32_e32 v4, v14, v4, vcc
	buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen
	v_perm_b32 v0, v3, v2, s4
	v_add_u32_e32 v2, 0xf0, v13
	v_perm_b32 v1, v7, v5, s4
	v_cndmask_b32_e32 v2, v14, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	.loc	1 1652 5                        ; 7.ttgir:1652:5
	s_endpgm
.Ltmp2:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 160
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
		.amdhsa_next_free_vgpr 228
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 228
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
	.set attn_fwd.num_vgpr, 228
	.set attn_fwd.num_agpr, 0
	.set attn_fwd.numbered_sgpr, 100
	.set attn_fwd.private_seg_size, 0
	.set attn_fwd.uses_vcc, 1
	.set attn_fwd.uses_flat_scratch, 0
	.set attn_fwd.has_dyn_sized_stack, 0
	.set attn_fwd.has_recursion, 0
	.set attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14624
; TotalNumSgprs: 106
; NumVgprs: 228
; NumAgprs: 0
; TotalNumVgprs: 228
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 28
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 228
; AccumOffset: 228
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 56
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
	.byte	0                               ; DW_CHILDREN_no
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"7.ttgir"                       ; string offset=7
.Linfo_string2:
	.asciz	"ttgir"                         ; string offset=15
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
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .offset:         128
        .size:           4
        .value_kind:     by_value
      - .offset:         132
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     global_buffer
      - .offset:         144
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         152
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 160
    .max_flat_workgroup_size: 256
    .name:           attn_fwd
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 4
    .symbol:         attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     228
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

new_asm= b'\t.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n\t.amdhsa_code_object_version 5\n\t.text\n\t.globl\tattn_fwd                        ; -- Begin function attn_fwd\n\t.p2align\t8\n\t.type\tattn_fwd,@function\nattn_fwd:                               ; @attn_fwd\n.Lfunc_begin0:\n\t.cfi_sections .debug_frame\n\t.cfi_startproc\n; %bb.23:\n\t.file\t1 "ttgir" "7.ttgir"\n\t.loc\t1 47 0 prologue_end             ; 7.ttgir:47:0\n\ts_load_dwordx2 s[2:3], s[0:1], 0x0\n\ts_load_dwordx8 s[4:11], s[0:1], 0x8\n\ts_load_dwordx4 s[12:15], s[0:1], 0x28\n\ts_waitcnt lgkmcnt(0)\n\ts_branch .LBB0_0\n\t.loc\t1 0 0 is_stmt 0                 ; :0:0\n.Ltmp0:\n\t.p2align\t8\n; %bb.24:\n.LBB0_0:\n.Ltmp1:\n\t.loc\t1 97 10 is_stmt 1               ; 7.ttgir:97:10\n\ts_load_dwordx8 s[40:47], s[0:1], 0x38\n\ts_load_dwordx4 s[20:23], s[0:1], 0x70\n\t.loc\t1 111 11                        ; 7.ttgir:111:11\n\ts_ashr_i32 s19, s18, 31\n\t.loc\t1 100 10                        ; 7.ttgir:100:10\n\ts_lshl_b32 s52, s16, 7\n\t.loc\t1 111 11                        ; 7.ttgir:111:11\n\ts_lshl_b64 s[0:1], s[18:19], 2\n\t.loc\t1 153 11                        ; 7.ttgir:153:11\n\tv_lshlrev_b32_e32 v2, 3, v0\n\t.loc\t1 111 11                        ; 7.ttgir:111:11\n\ts_waitcnt lgkmcnt(0)\n\ts_add_u32 s20, s20, s0\n\ts_addc_u32 s21, s21, s1\n\t.loc\t1 112 11                        ; 7.ttgir:112:11\n\ts_load_dwordx2 s[54:55], s[20:21], 0x0\n\t.loc\t1 102 10                        ; 7.ttgir:102:10\n\tv_lshrrev_b32_e32 v158, 4, v0\n\t.loc\t1 153 11                        ; 7.ttgir:153:11\n\tv_and_b32_e32 v138, 0x78, v2\n\t.loc\t1 102 10                        ; 7.ttgir:102:10\n\tv_or_b32_e32 v159, 16, v158\n\tv_or_b32_e32 v1, s52, v158\n\t.loc\t1 115 11                        ; 7.ttgir:115:11\n\ts_waitcnt lgkmcnt(0)\n\ts_sub_i32 s16, s55, s54\n\t.loc\t1 116 11                        ; 7.ttgir:116:11\n\ts_add_u32 s0, s22, s0\n\ts_addc_u32 s1, s23, s1\n\t.loc\t1 117 11                        ; 7.ttgir:117:11\n\ts_load_dwordx2 s[20:21], s[0:1], 0x0\n\t.loc\t1 107 11                        ; 7.ttgir:107:11\n\tv_or_b32_e32 v4, s52, v159\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_bfrev_b32_e32 v31, 1\n\t.loc\t1 190 11                        ; 7.ttgir:190:11\n\tv_cmp_gt_i32_e32 vcc, s16, v1\n\t.loc\t1 107 11                        ; 7.ttgir:107:11\n\tv_or_b32_e32 v12, 32, v1\n\t.loc\t1 120 11                        ; 7.ttgir:120:11\n\ts_waitcnt lgkmcnt(0)\n\ts_sub_i32 s19, s21, s20\n\t.loc\t1 121 11                        ; 7.ttgir:121:11\n\ts_add_i32 s0, s19, 31\n\t.loc\t1 122 11                        ; 7.ttgir:122:11\n\ts_ashr_i32 s1, s0, 31\n\ts_lshr_b32 s1, s1, 27\n\ts_add_i32 s0, s0, s1\n\t.loc\t1 126 11                        ; 7.ttgir:126:11\n\ts_sub_i32 s1, s52, s16\n\ts_add_i32 s1, s1, s19\n\t.loc\t1 127 11                        ; 7.ttgir:127:11\n\ts_addk_i32 s1, 0x9f\n\t.loc\t1 128 11                        ; 7.ttgir:128:11\n\ts_ashr_i32 s21, s1, 31\n\ts_lshr_b32 s21, s21, 27\n\ts_add_i32 s1, s1, s21\n\t.loc\t1 122 11                        ; 7.ttgir:122:11\n\ts_ashr_i32 s0, s0, 5\n\t.loc\t1 128 11                        ; 7.ttgir:128:11\n\ts_ashr_i32 s1, s1, 5\n\t.loc\t1 129 11                        ; 7.ttgir:129:11\n\ts_min_i32 s72, s0, s1\n\t.loc\t1 131 5                         ; 7.ttgir:131:5\n\ts_and_b32 s0, s19, 31\n\ts_sub_i32 s1, 32, s19\n\t.loc\t1 130 11                        ; 7.ttgir:130:11\n\ts_cmp_lt_i32 s19, 32\n\t.loc\t1 131 5                         ; 7.ttgir:131:5\n\ts_cselect_b32 s67, s1, s0\n\t.loc\t1 139 11                        ; 7.ttgir:139:11\n\ts_mul_i32 s0, s12, s18\n\t.loc\t1 140 11                        ; 7.ttgir:140:11\n\ts_ashr_i32 s1, s0, 31\n\ts_lshl_b64 s[0:1], s[0:1], 1\n\ts_add_u32 s2, s2, s0\n\t.loc\t1 141 11                        ; 7.ttgir:141:11\n\ts_mul_i32 s0, s13, s17\n\t.loc\t1 140 11                        ; 7.ttgir:140:11\n\ts_addc_u32 s3, s3, s1\n\t.loc\t1 142 11                        ; 7.ttgir:142:11\n\ts_ashr_i32 s1, s0, 31\n\ts_lshl_b64 s[0:1], s[0:1], 1\n\ts_add_u32 s2, s2, s0\n\t.loc\t1 143 11                        ; 7.ttgir:143:11\n\ts_mul_i32 s0, s54, s14\n\t.loc\t1 142 11                        ; 7.ttgir:142:11\n\ts_addc_u32 s3, s3, s1\n\t.loc\t1 144 11                        ; 7.ttgir:144:11\n\ts_ashr_i32 s1, s0, 31\n\ts_lshl_b64 s[0:1], s[0:1], 1\n\ts_add_u32 s2, s2, s0\n\t.loc\t1 148 11                        ; 7.ttgir:148:11\n\ts_mul_i32 s0, s14, s52\n\t.loc\t1 144 11                        ; 7.ttgir:144:11\n\ts_addc_u32 s3, s3, s1\n\t.loc\t1 151 11                        ; 7.ttgir:151:11\n\ts_ashr_i32 s1, s0, 31\n\t.loc\t1 150 11                        ; 7.ttgir:150:11\n\ts_lshl_b32 s12, s14, 4\n\t.loc\t1 151 11                        ; 7.ttgir:151:11\n\ts_lshl_b64 s[0:1], s[0:1], 1\n\ts_add_u32 s0, s2, s0\n\ts_addc_u32 s1, s3, s1\n\t.loc\t1 158 11                        ; 7.ttgir:158:11\n\tv_mad_u64_u32 v[2:3], s[2:3], s14, v158, v[138:139]\n\t.loc\t1 159 11                        ; 7.ttgir:159:11\n\ts_mul_i32 s2, s15, s18\n\t.loc\t1 160 11                        ; 7.ttgir:160:11\n\ts_ashr_i32 s3, s2, 31\n\ts_lshl_b64 s[34:35], s[2:3], 1\n\ts_add_u32 s13, s4, s34\n\t.loc\t1 161 11                        ; 7.ttgir:161:11\n\ts_mul_i32 s2, s40, s17\n\t.loc\t1 160 11                        ; 7.ttgir:160:11\n\ts_addc_u32 s15, s5, s35\n\t.loc\t1 162 11                        ; 7.ttgir:162:11\n\ts_ashr_i32 s3, s2, 31\n\ts_lshl_b64 s[56:57], s[2:3], 1\n\ts_add_u32 s13, s13, s56\n\t.loc\t1 163 11                        ; 7.ttgir:163:11\n\ts_mul_i32 s2, s20, s41\n\t.loc\t1 162 11                        ; 7.ttgir:162:11\n\ts_addc_u32 s15, s15, s57\n\t.loc\t1 164 11                        ; 7.ttgir:164:11\n\ts_ashr_i32 s3, s2, 31\n\ts_lshl_b64 s[58:59], s[2:3], 1\n\ts_add_u32 s28, s13, s58\n\t.loc\t1 174 11                        ; 7.ttgir:174:11\n\ts_mul_i32 s2, s42, s18\n\t.loc\t1 164 11                        ; 7.ttgir:164:11\n\ts_addc_u32 s40, s15, s59\n\t.loc\t1 175 11                        ; 7.ttgir:175:11\n\ts_ashr_i32 s3, s2, 31\n\t.loc\t1 171 11                        ; 7.ttgir:171:11\n\ts_lshl_b32 s13, s41, 4\n\t.loc\t1 175 11                        ; 7.ttgir:175:11\n\ts_lshl_b64 s[60:61], s[2:3], 1\n\ts_add_u32 s15, s6, s60\n\t.loc\t1 176 11                        ; 7.ttgir:176:11\n\ts_mul_i32 s2, s43, s17\n\t.loc\t1 175 11                        ; 7.ttgir:175:11\n\ts_addc_u32 s21, s7, s61\n\t.loc\t1 177 11                        ; 7.ttgir:177:11\n\ts_ashr_i32 s3, s2, 31\n\ts_lshl_b64 s[42:43], s[2:3], 1\n\ts_add_u32 s15, s15, s42\n\t.loc\t1 178 11                        ; 7.ttgir:178:11\n\ts_mul_i32 s2, s20, s44\n\t.loc\t1 177 11                        ; 7.ttgir:177:11\n\ts_addc_u32 s21, s21, s43\n\t.loc\t1 179 11                        ; 7.ttgir:179:11\n\ts_ashr_i32 s3, s2, 31\n\ts_lshl_b64 s[62:63], s[2:3], 1\n\ts_add_u32 s36, s15, s62\n\ts_addc_u32 s66, s21, s63\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\ts_and_b32 s2, s14, 0x3fff\n\ts_bitset1_b32 s2, 14\n\t.loc\t1 158 11                        ; 7.ttgir:158:11\n\tv_add_u32_e32 v3, s12, v2\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\ts_and_b32 s1, s1, 0xffff\n\ts_lshl_b32 s2, s2, 16\n\tv_lshlrev_b32_e32 v2, 1, v2\n\t.loc\t1 107 11                        ; 7.ttgir:107:11\n\tv_or_b32_e32 v13, 48, v1\n\tv_or_b32_e32 v20, 64, v1\n\tv_or_b32_e32 v21, 0x50, v1\n\tv_or_b32_e32 v28, 0x60, v1\n\tv_or_b32_e32 v29, 0x70, v1\n\t.loc\t1 158 11                        ; 7.ttgir:158:11\n\tv_add_u32_e32 v14, s12, v3\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\ts_or_b32 s1, s1, s2\n\ts_mov_b32 s3, 0x27000\n\ts_mov_b32 s2, 0x7ffffffe\n\tv_cndmask_b32_e32 v1, v31, v2, vcc\n\tv_lshlrev_b32_e32 v2, 1, v3\n\t.loc\t1 190 11                        ; 7.ttgir:190:11\n\tv_cmp_gt_i32_e32 vcc, s16, v4\n\t.loc\t1 158 11                        ; 7.ttgir:158:11\n\tv_add_u32_e32 v15, s12, v14\n\tv_add_u32_e32 v22, s12, v15\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_cndmask_b32_e32 v2, v31, v2, vcc\n\tbuffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen\n\tbuffer_load_dwordx4 v[8:11], v2, s[0:3], 0 offen\n\tv_lshlrev_b32_e32 v1, 1, v14\n\t.loc\t1 190 11                        ; 7.ttgir:190:11\n\tv_cmp_gt_i32_e32 vcc, s16, v12\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_lshlrev_b32_e32 v2, 1, v15\n\t.loc\t1 158 11                        ; 7.ttgir:158:11\n\tv_add_u32_e32 v23, s12, v22\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_cndmask_b32_e32 v1, v31, v1, vcc\n\t.loc\t1 190 11                        ; 7.ttgir:190:11\n\tv_cmp_gt_i32_e32 vcc, s16, v13\n\t.loc\t1 158 11                        ; 7.ttgir:158:11\n\tv_add_u32_e32 v30, s12, v23\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_and_b32_e32 v38, 0x80, v0\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_cndmask_b32_e32 v2, v31, v2, vcc\n\tbuffer_load_dwordx4 v[12:15], v1, s[0:3], 0 offen\n\tbuffer_load_dwordx4 v[16:19], v2, s[0:3], 0 offen\n\tv_lshlrev_b32_e32 v1, 1, v22\n\t.loc\t1 190 11                        ; 7.ttgir:190:11\n\tv_cmp_gt_i32_e32 vcc, s16, v20\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_lshlrev_b32_e32 v2, 1, v23\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_lshrrev_b32_e32 v152, 1, v38\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_cndmask_b32_e32 v1, v31, v1, vcc\n\t.loc\t1 190 11                        ; 7.ttgir:190:11\n\tv_cmp_gt_i32_e32 vcc, s16, v21\n\t.loc\t1 220 12                        ; 7.ttgir:220:12\n\ts_mov_b32 s30, s2\n\ts_mov_b32 s31, s3\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_cndmask_b32_e32 v2, v31, v2, vcc\n\tbuffer_load_dwordx4 v[20:23], v1, s[0:3], 0 offen\n\tbuffer_load_dwordx4 v[24:27], v2, s[0:3], 0 offen\n\tv_lshlrev_b32_e32 v1, 1, v30\n\t.loc\t1 190 11                        ; 7.ttgir:190:11\n\tv_cmp_gt_i32_e32 vcc, s16, v28\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_add_lshl_u32 v2, v30, s12, 1\n\t.loc\t1 222 12                        ; 7.ttgir:222:12\n\ts_mov_b32 s38, s2\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_cndmask_b32_e32 v1, v31, v1, vcc\n\t.loc\t1 190 11                        ; 7.ttgir:190:11\n\tv_cmp_gt_i32_e32 vcc, s16, v29\n\t.loc\t1 222 12                        ; 7.ttgir:222:12\n\ts_mov_b32 s39, s3\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_and_b32_e32 v153, 31, v0\n\t.loc\t1 194 11                        ; 7.ttgir:194:11\n\tv_cndmask_b32_e32 v2, v31, v2, vcc\n\tbuffer_load_dwordx4 v[28:31], v1, s[0:3], 0 offen\n\tbuffer_load_dwordx4 v[32:35], v2, s[0:3], 0 offen\n\t.loc\t1 173 11                        ; 7.ttgir:173:11\n\tv_mad_u64_u32 v[36:37], s[0:1], s41, v158, v[138:139]\n\t.loc\t1 196 11                        ; 7.ttgir:196:11\n\tv_lshrrev_b32_e32 v37, 1, v0\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_and_b32_e32 v2, 64, v0\n\t.loc\t1 196 11                        ; 7.ttgir:196:11\n\tv_and_b32_e32 v1, 24, v37\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_lshrrev_b32_e32 v151, 1, v2\n\t.loc\t1 196 11                        ; 7.ttgir:196:11\n\tv_xor_b32_e32 v1, v1, v138\n\tv_xor_b32_e32 v1, v1, v151\n\tv_xor_b32_e32 v1, v1, v152\n\t.loc\t1 202 11                        ; 7.ttgir:202:11\n\ts_and_b32 s0, s16, 0x7f\n\t.loc\t1 196 11                        ; 7.ttgir:196:11\n\tv_lshlrev_b32_e32 v1, 1, v1\n\t.loc\t1 203 11                        ; 7.ttgir:203:11\n\ts_or_b32 s0, s67, s0\n\t.loc\t1 196 11                        ; 7.ttgir:196:11\n\tv_lshl_or_b32 v1, v158, 8, v1\n\t.loc\t1 203 11                        ; 7.ttgir:203:11\n\ts_cmp_eq_u32 s0, 0\n\t.loc\t1 196 11                        ; 7.ttgir:196:11\n\tv_add_u32_e32 v39, 0, v1\n\t.loc\t1 206 12                        ; 7.ttgir:206:12\n\ts_cselect_b32 s12, 4, 5\n\t.loc\t1 220 12                        ; 7.ttgir:220:12\n\ts_and_b32 s0, s41, 0x3fff\n\t.loc\t1 196 11                        ; 7.ttgir:196:11\n\ts_waitcnt vmcnt(7)\n\tds_write_b128 v39, v[4:7]\n\ts_waitcnt vmcnt(6)\n\tds_write_b128 v39, v[8:11] offset:4096\n\ts_waitcnt vmcnt(5)\n\tds_write_b128 v39, v[12:15] offset:8192\n\ts_waitcnt vmcnt(4)\n\tds_write_b128 v39, v[16:19] offset:12288\n\ts_waitcnt vmcnt(3)\n\tds_write_b128 v39, v[20:23] offset:16384\n\ts_waitcnt vmcnt(2)\n\tds_write_b128 v39, v[24:27] offset:20480\n\ts_waitcnt vmcnt(1)\n\tds_write_b128 v39, v[28:31] offset:24576\n\ts_waitcnt vmcnt(0)\n\tds_write_b128 v39, v[32:35] offset:28672\n\t.loc\t1 220 12                        ; 7.ttgir:220:12\n\ts_bitset1_b32 s0, 14\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_and_b32_e32 v7, 16, v0\n\t.loc\t1 102 10                        ; 7.ttgir:102:10\n\tv_and_b32_e32 v8, 32, v0\n\t.loc\t1 220 12                        ; 7.ttgir:220:12\n\ts_and_b32 s1, s40, 0xffff\n\ts_lshl_b32 s53, s0, 16\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_lshrrev_b32_e32 v1, 3, v8\n\t.loc\t1 110 11                        ; 7.ttgir:110:11\n\tv_lshrrev_b32_e32 v3, 3, v7\n\t.loc\t1 220 12                        ; 7.ttgir:220:12\n\ts_or_b32 s29, s1, s53\n\tv_lshlrev_b32_e32 v160, 1, v36\n\t.loc\t1 110 11                        ; 7.ttgir:110:11\n\tv_or_b32_e32 v4, v3, v1\n\tv_lshrrev_b32_e32 v6, 3, v2\n\tv_lshrrev_b32_e32 v15, 3, v38\n\t.loc\t1 197 5                         ; 7.ttgir:197:5\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\t.loc\t1 220 12                        ; 7.ttgir:220:12\n\tv_add_lshl_u32 v161, v36, s13, 1\n\tbuffer_load_dwordx4 v[16:19], v160, s[28:31], 0 offen\n\tbuffer_load_dwordx4 v[20:23], v161, s[28:31], 0 offen\n\t.loc\t1 110 11                        ; 7.ttgir:110:11\n\tv_or3_b32 v162, v4, v6, v15\n\t.loc\t1 186 11                        ; 7.ttgir:186:11\n\tv_mad_u64_u32 v[4:5], s[0:1], s44, v162, v[138:139]\n\t.loc\t1 222 12                        ; 7.ttgir:222:12\n\ts_and_b32 s0, s44, 0x3fff\n\ts_bitset1_b32 s0, 14\n\ts_and_b32 s1, s66, 0xffff\n\ts_lshl_b32 s55, s0, 16\n\ts_or_b32 s37, s1, s55\n\tv_lshlrev_b32_e32 v163, 1, v4\n\tv_add_lshl_u32 v164, v4, s44, 1\n\tbuffer_load_dwordx4 v[114:117], v163, s[36:39], 0 offen\n\tbuffer_load_dwordx4 v[118:121], v164, s[36:39], 0 offen\n\t.loc\t1 199 11                        ; 7.ttgir:199:11\n\tv_lshrrev_b32_e32 v5, 2, v0\n\ts_movk_i32 s13, 0x60\n\tv_and_b32_e32 v141, 8, v5\n\tv_and_b32_e32 v4, 15, v0\n\tv_bfe_u32 v24, v0, 5, 1\n\tv_and_or_b32 v5, v37, s13, v153\n\tv_or_b32_e32 v9, 32, v141\n\tv_xor_b32_e32 v25, v24, v4\n\tv_or_b32_e32 v24, 2, v24\n\tv_lshlrev_b32_e32 v139, 3, v4\n\tv_or_b32_e32 v10, 48, v141\n\tv_or_b32_e32 v11, 64, v141\n\tv_lshlrev_b32_e32 v25, 3, v25\n\tv_lshlrev_b32_e32 v5, 7, v5\n\tv_xor_b32_e32 v24, v24, v4\n\tv_xor_b32_e32 v9, v9, v139\n\tv_or_b32_e32 v12, 0x50, v141\n\tv_or_b32_e32 v13, 0x60, v141\n\tv_or_b32_e32 v26, v25, v5\n\tv_lshlrev_b32_e32 v24, 3, v24\n\tv_or_b32_e32 v4, v5, v9\n\tv_xor_b32_e32 v10, v10, v139\n\tv_xor_b32_e32 v11, v11, v139\n\tv_or_b32_e32 v14, 0x70, v141\n\tv_or_b32_e32 v27, v24, v5\n\tv_or_b32_e32 v28, v5, v10\n\tv_or_b32_e32 v29, v5, v11\n\tv_xor_b32_e32 v12, v12, v139\n\tv_xor_b32_e32 v13, v13, v139\n\tv_lshl_add_u32 v26, v26, 1, 0\n\tv_lshl_add_u32 v4, v4, 1, 0\n\tv_or_b32_e32 v30, v5, v12\n\tv_or_b32_e32 v31, v5, v13\n\tv_xor_b32_e32 v14, v14, v139\n\tv_lshl_add_u32 v27, v27, 1, 0\n\tds_read_b128 v[110:113], v26\n\tds_read_b128 v[106:109], v27\n\tv_lshl_add_u32 v26, v28, 1, 0\n\tds_read_b128 v[102:105], v4\n\tds_read_b128 v[98:101], v26\n\tv_lshl_add_u32 v4, v29, 1, 0\n\tv_or_b32_e32 v5, v5, v14\n\tv_lshl_add_u32 v26, v30, 1, 0\n\tds_read_b128 v[94:97], v4\n\tds_read_b128 v[90:93], v26\n\tv_lshl_add_u32 v4, v31, 1, 0\n\tv_lshl_add_u32 v5, v5, 1, 0\n\tds_read_b128 v[86:89], v4\n\tds_read_b128 v[82:85], v5\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_and_b32_e32 v4, 1, v0\n\tv_cmp_eq_u32_e64 s[24:25], 0, v4\n\tv_and_b32_e32 v4, 2, v0\n\tv_cmp_eq_u32_e64 s[26:27], 0, v4\n\tv_and_b32_e32 v4, 4, v0\n\tv_cmp_eq_u32_e64 s[20:21], 0, v4\n\tv_and_b32_e32 v4, 8, v0\n\t.loc\t1 242 14                        ; 7.ttgir:242:14\n\tv_lshlrev_b32_e32 v140, 7, v153\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_cmp_eq_u32_e64 s[22:23], 0, v4\n\t.loc\t1 242 14                        ; 7.ttgir:242:14\n\tv_or_b32_e32 v4, v25, v140\n\t.loc\t1 224 5                         ; 7.ttgir:224:5\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\ts_waitcnt vmcnt(3)\n\t.loc\t1 226 5                         ; 7.ttgir:226:5\n\tds_write_b128 v39, v[16:19]\n\ts_waitcnt vmcnt(2)\n\tds_write_b128 v39, v[20:23] offset:4096\n\t.loc\t1 242 14                        ; 7.ttgir:242:14\n\tv_lshl_add_u32 v177, v4, 1, 0\n\t.loc\t1 243 14                        ; 7.ttgir:243:14\n\tv_or_b32_e32 v4, v24, v140\n\t.loc\t1 207 12                        ; 7.ttgir:207:12\n\ts_min_i32 s1, s12, s72\n\t.loc\t1 243 14                        ; 7.ttgir:243:14\n\tv_lshl_add_u32 v179, v4, 1, 0\n\t.loc\t1 242 14                        ; 7.ttgir:242:14\n\tds_read_b128 v[126:129], v177\n\t.loc\t1 243 14                        ; 7.ttgir:243:14\n\tds_read_b128 v[122:125], v179\n\t.loc\t1 208 12                        ; 7.ttgir:208:12\n\ts_sub_i32 s0, s72, s1\n\t.loc\t1 209 12                        ; 7.ttgir:209:12\n\ts_lshl_b32 s33, s0, 5\n\t.loc\t1 215 12                        ; 7.ttgir:215:12\n\ts_lshl_b32 s30, s41, 5\n\t.loc\t1 211 12                        ; 7.ttgir:211:12\n\ts_or_b32 s2, s33, 31\n\t.loc\t1 233 13                        ; 7.ttgir:233:13\n\ts_ashr_i32 s31, s30, 31\n\t.loc\t1 101 10                        ; 7.ttgir:101:10\n\tv_cmp_eq_u32_e64 s[82:83], 0, v38\n\tv_or3_b32 v5, v6, v1, v15\n\t.loc\t1 297 12                        ; 7.ttgir:297:12\n\ts_cmp_gt_i32 s2, 63\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v156, 8, v1\n\tv_or_b32_e32 v155, 16, v1\n\tv_or_b32_e32 v154, 24, v1\n\tv_or_b32_e32 v165, v9, v140\n\tv_or_b32_e32 v168, v10, v140\n\tv_or_b32_e32 v169, v11, v140\n\tv_or_b32_e32 v170, v12, v140\n\tv_or_b32_e32 v172, v13, v140\n\tv_or_b32_e32 v174, v14, v140\n\tv_lshlrev_b32_e32 v6, 2, v0\n\tv_lshlrev_b32_e32 v4, 5, v153\n\t.loc\t1 298 5                         ; 7.ttgir:298:5\n\ts_cbranch_scc1 .LBB0_2\n; %bb.1:                                ; %.._crit_edge_crit_edge\n\t.loc\t1 658 12                        ; 7.ttgir:658:12\n\tv_or_b32_e32 v135, v9, v140\n\t.loc\t1 899 5                         ; 7.ttgir:899:5\n\tv_mov_b32_e32 v9, 0x440\n\t.loc\t1 662 12                        ; 7.ttgir:662:12\n\tv_or_b32_e32 v133, v11, v140\n\t.loc\t1 899 5                         ; 7.ttgir:899:5\n\tv_cndmask_b32_e64 v148, v9, 0, s[82:83]\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_mov_b32_e32 v9, 0x110\n\tv_mov_b32_e32 v11, 0x204\n\t.loc\t1 660 12                        ; 7.ttgir:660:12\n\tv_or_b32_e32 v134, v10, v140\n\t.loc\t1 666 12                        ; 7.ttgir:666:12\n\tv_or_b32_e32 v131, v13, v140\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_cndmask_b32_e64 v10, v9, 0, s[24:25]\n\tv_cndmask_b32_e64 v11, v11, 0, s[26:27]\n\tv_mov_b32_e32 v13, 0x408\n\t.loc\t1 664 12                        ; 7.ttgir:664:12\n\tv_or_b32_e32 v132, v12, v140\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_or_b32_e32 v12, v10, v11\n\tv_cndmask_b32_e64 v13, v13, 0, s[20:21]\n\tv_mov_b32_e32 v15, 0x810\n\t.loc\t1 668 12                        ; 7.ttgir:668:12\n\tv_or_b32_e32 v130, v14, v140\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_or_b32_e32 v14, v12, v13\n\tv_cndmask_b32_e64 v15, v15, 0, s[22:23]\n\tv_xor_b32_e32 v16, v14, v15\n\tv_or_b32_e32 v14, 32, v14\n\tv_xor_b32_e32 v14, v14, v15\n\tv_or_b32_e32 v14, v14, v3\n\tv_xor_b32_e32 v143, v5, v14\n\tv_or_b32_e32 v14, 0x44, v10\n\tv_xor_b32_e32 v14, v14, v11\n\tv_or_b32_e32 v14, v14, v13\n\tv_xor_b32_e32 v14, v14, v15\n\tv_or_b32_e32 v14, v14, v3\n\tv_xor_b32_e32 v144, v5, v14\n\tv_or_b32_e32 v14, 0x64, v10\n\tv_xor_b32_e32 v14, v14, v11\n\tv_or_b32_e32 v14, v14, v13\n\tv_xor_b32_e32 v14, v14, v15\n\tv_or_b32_e32 v14, v14, v3\n\tv_xor_b32_e32 v145, v5, v14\n\tv_or_b32_e32 v14, 0x88, v12\n\tv_or_b32_e32 v13, v15, v13\n\tv_or_b32_e32 v12, 0xa8, v12\n\tv_xor_b32_e32 v12, v13, v12\n\tv_or_b32_e32 v12, v12, v3\n\tv_xor_b32_e32 v147, v5, v12\n\tv_or_b32_e32 v12, 0xcc, v10\n\tv_or_b32_e32 v11, v13, v11\n\tv_xor_b32_e32 v12, v11, v12\n\tv_or_b32_e32 v10, 0xec, v10\n\tv_or_b32_e32 v12, v12, v3\n\tv_xor_b32_e32 v10, v11, v10\n\tv_xor_b32_e32 v14, v13, v14\n\tv_xor_b32_e32 v149, v5, v12\n\tv_or_b32_e32 v10, v10, v3\n\t.loc\t1 788 12                        ; 7.ttgir:788:12\n\tv_mov_b32_e32 v12, 0x44\n\tv_mov_b32_e32 v13, 0x88\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_xor_b32_e32 v150, v5, v10\n\t.loc\t1 788 12                        ; 7.ttgir:788:12\n\tv_and_b32_e32 v10, 32, v4\n\tv_cndmask_b32_e64 v12, v12, 0, s[26:27]\n\tv_cndmask_b32_e64 v13, v13, 0, s[20:21]\n\tv_or_b32_e32 v11, v1, v10\n\tv_or_b32_e32 v12, v13, v12\n\tv_cndmask_b32_e64 v9, v9, 0, s[22:23]\n\tv_xor_b32_e32 v11, v12, v11\n\tv_or_b32_e32 v167, v11, v9\n\t.loc\t1 796 12                        ; 7.ttgir:796:12\n\tv_or_b32_e32 v11, v156, v10\n\tv_or_b32_e32 v13, v12, v9\n\tv_xor_b32_e32 v171, v13, v11\n\t.loc\t1 798 12                        ; 7.ttgir:798:12\n\tv_xor_b32_e32 v11, v12, v11\n\tv_or_b32_e32 v173, v11, v9\n\t.loc\t1 804 12                        ; 7.ttgir:804:12\n\tv_or_b32_e32 v11, v155, v10\n\t.loc\t1 812 12                        ; 7.ttgir:812:12\n\tv_or_b32_e32 v10, v154, v10\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_or_b32_e32 v16, v16, v3\n\tv_or_b32_e32 v14, v14, v3\n\t.loc\t1 804 12                        ; 7.ttgir:804:12\n\tv_xor_b32_e32 v175, v13, v11\n\t.loc\t1 806 12                        ; 7.ttgir:806:12\n\tv_xor_b32_e32 v11, v12, v11\n\t.loc\t1 812 12                        ; 7.ttgir:812:12\n\tv_xor_b32_e32 v166, v13, v10\n\t.loc\t1 814 12                        ; 7.ttgir:814:12\n\tv_xor_b32_e32 v10, v12, v10\n\t.loc\t1 707 12                        ; 7.ttgir:707:12\n\tv_xor_b32_e32 v157, 0x80, v6\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_xor_b32_e32 v142, v5, v16\n\tv_xor_b32_e32 v146, v5, v14\n\t.loc\t1 806 12                        ; 7.ttgir:806:12\n\tv_xor_b32_e32 v178, v11, v9\n\t.loc\t1 814 12                        ; 7.ttgir:814:12\n\tv_xor_b32_e32 v176, v10, v9\n\ts_mov_b64 s[2:3], 0\n\ts_branch .LBB0_3\n.LBB0_2:\n\t.loc\t1 0 12 is_stmt 0                ; 7.ttgir:0:12\n\ts_mov_b64 s[2:3], -1\n                                        ; implicit-def: $vgpr135\n                                        ; implicit-def: $vgpr134\n                                        ; implicit-def: $vgpr133\n                                        ; implicit-def: $vgpr132\n                                        ; implicit-def: $vgpr131\n                                        ; implicit-def: $vgpr130\n                                        ; implicit-def: $vgpr157\n                                        ; implicit-def: $vgpr148\n                                        ; implicit-def: $vgpr142\n                                        ; implicit-def: $vgpr143\n                                        ; implicit-def: $vgpr144\n                                        ; implicit-def: $vgpr145\n                                        ; implicit-def: $vgpr146\n                                        ; implicit-def: $vgpr147\n                                        ; implicit-def: $vgpr149\n                                        ; implicit-def: $vgpr150\n                                        ; implicit-def: $vgpr167\n                                        ; implicit-def: $vgpr171\n                                        ; implicit-def: $vgpr173\n                                        ; implicit-def: $vgpr175\n                                        ; implicit-def: $vgpr178\n                                        ; implicit-def: $vgpr166\n                                        ; implicit-def: $vgpr176\n.LBB0_3:                                ; %Flow394\n\ts_lshl_b32 s38, s44, 5\n\t.loc\t1 298 5 is_stmt 1               ; 7.ttgir:298:5\n\ts_andn2_b64 vcc, exec, s[2:3]\n\tv_cmp_eq_u32_e64 s[14:15], 0, v7\n\tv_cmp_eq_u32_e64 s[12:13], 0, v8\n\tv_cmp_eq_u32_e64 s[2:3], 0, v2\n\ts_cbranch_vccnz .LBB0_7\n; %bb.4:                                ; %.lr.ph\n\t.loc\t1 212 12                        ; 7.ttgir:212:12\n\ts_and_b32 s29, s0, 0x7ffffff\n\t.loc\t1 233 13                        ; 7.ttgir:233:13\n\ts_lshl_b64 s[64:65], s[30:31], 1\n\ts_add_u32 s48, s28, s64\n\ts_addc_u32 s37, s40, s65\n\t.loc\t1 236 13                        ; 7.ttgir:236:13\n\ts_and_b32 s37, s37, 0xffff\n\ts_or_b32 s49, s37, s53\n\ts_mov_b32 s51, 0x27000\n\ts_mov_b32 s50, 0x7ffffffe\n\t.loc\t1 237 13                        ; 7.ttgir:237:13\n\tbuffer_load_dwordx4 v[130:133], v161, s[48:51], 0 offen\n\t.loc\t1 236 13                        ; 7.ttgir:236:13\n\tbuffer_load_dwordx4 v[134:137], v160, s[48:51], 0 offen\n\tv_mov_b32_e32 v2, 0x110\n\tv_mov_b32_e32 v8, 0x204\n\tv_cndmask_b32_e64 v7, v2, 0, s[24:25]\n\tv_cndmask_b32_e64 v9, v8, 0, s[26:27]\n\tv_mov_b32_e32 v11, 0x408\n\tv_or_b32_e32 v10, v7, v9\n\tv_cndmask_b32_e64 v11, v11, 0, s[20:21]\n\tv_mov_b32_e32 v13, 0x810\n\tv_or_b32_e32 v12, v10, v11\n\tv_cndmask_b32_e64 v13, v13, 0, s[22:23]\n\tv_xor_b32_e32 v14, v12, v13\n\tv_or_b32_e32 v12, 32, v12\n\tv_xor_b32_e32 v12, v12, v13\n\tv_or_b32_e32 v12, v12, v3\n\tv_xor_b32_e32 v143, v5, v12\n\tv_or_b32_e32 v12, 0x44, v7\n\tv_xor_b32_e32 v12, v12, v9\n\tv_or_b32_e32 v12, v12, v11\n\tv_xor_b32_e32 v12, v12, v13\n\tv_or_b32_e32 v12, v12, v3\n\tv_xor_b32_e32 v144, v5, v12\n\tv_or_b32_e32 v12, 0x64, v7\n\tv_xor_b32_e32 v12, v12, v9\n\tv_or_b32_e32 v12, v12, v11\n\tv_xor_b32_e32 v12, v12, v13\n\tv_or_b32_e32 v12, v12, v3\n\tv_xor_b32_e32 v145, v5, v12\n\tv_or_b32_e32 v12, 0x88, v10\n\tv_or_b32_e32 v11, v13, v11\n\tv_or_b32_e32 v10, 0xa8, v10\n\tv_xor_b32_e32 v10, v11, v10\n\tv_or_b32_e32 v10, v10, v3\n\tv_xor_b32_e32 v147, v5, v10\n\tv_or_b32_e32 v10, 0xcc, v7\n\tv_or_b32_e32 v9, v11, v9\n\tv_or_b32_e32 v7, 0xec, v7\n\tv_xor_b32_e32 v12, v11, v12\n\tv_xor_b32_e32 v10, v9, v10\n\tv_xor_b32_e32 v7, v9, v7\n\tv_or_b32_e32 v14, v14, v3\n\tv_or_b32_e32 v12, v12, v3\n\tv_or_b32_e32 v10, v10, v3\n\tv_or_b32_e32 v3, v7, v3\n\tv_xor_b32_e32 v150, v5, v3\n\tv_mov_b32_e32 v3, 0x88\n\tv_xor_b32_e32 v142, v5, v14\n\tv_xor_b32_e32 v146, v5, v12\n\tv_xor_b32_e32 v149, v5, v10\n\tv_xor_b32_e32 v157, 0x80, v6\n\tv_cndmask_b32_e64 v5, v3, 0, s[14:15]\n\tv_cndmask_b32_e64 v6, v2, 0, s[12:13]\n\tv_or_b32_e32 v5, v5, v6\n\tv_mov_b32_e32 v6, 0x220\n\tv_mov_b32_e32 v7, 0x440\n\tv_cndmask_b32_e64 v6, v6, 0, s[2:3]\n\tv_cndmask_b32_e64 v148, v7, 0, s[82:83]\n\tv_or3_b32 v5, v5, v6, v148\n\tv_mov_b32_e32 v6, 0x44\n\tv_xor_b32_e32 v5, v5, v138\n\tv_and_b32_e32 v4, 32, v4\n\tv_cndmask_b32_e64 v6, v6, 0, s[26:27]\n\tv_cndmask_b32_e64 v3, v3, 0, s[20:21]\n\tv_lshl_add_u32 v196, v5, 1, 0\n\tv_or_b32_e32 v5, v1, v4\n\tv_or_b32_e32 v3, v3, v6\n\ts_ashr_i32 s39, s38, 31\n\tv_cndmask_b32_e64 v2, v2, 0, s[22:23]\n\tv_xor_b32_e32 v5, v3, v5\n\tv_cndmask_b32_e64 v7, v8, 0, s[14:15]\n\tv_or_b32_e32 v167, v5, v2\n\t.loc\t1 298 5                         ; 7.ttgir:298:5\n\ts_add_u32 s20, s62, s42\n\tv_xor_b32_e32 v5, v167, v7\n\ts_addc_u32 s21, s63, s43\n\tv_xor_b32_e32 v6, 0x408, v5\n\ts_add_u32 s22, s20, s60\n\tv_lshl_add_u32 v197, v5, 1, 0\n\tv_lshl_add_u32 v198, v6, 1, 0\n\tv_xor_b32_e32 v6, 0x810, v5\n\tv_xor_b32_e32 v5, 0xc18, v5\n\ts_addc_u32 s23, s21, s61\n\ts_lshl_b64 s[20:21], s[38:39], 1\n\tv_lshl_add_u32 v199, v6, 1, 0\n\tv_lshl_add_u32 v200, v5, 1, 0\n\tv_or_b32_e32 v5, v156, v4\n\tv_or_b32_e32 v6, v3, v2\n\ts_add_u32 s22, s22, s20\n\tv_xor_b32_e32 v171, v6, v5\n\tv_xor_b32_e32 v5, v3, v5\n\ts_addc_u32 s23, s23, s21\n\tv_or_b32_e32 v173, v5, v2\n\ts_add_u32 s22, s6, s22\n\tv_xor_b32_e32 v8, v171, v7\n\tv_xor_b32_e32 v5, v173, v7\n\ts_addc_u32 s23, s7, s23\n\ts_lshl_b64 s[24:25], s[30:31], 2\n\tv_lshl_add_u32 v202, v8, 1, 0\n\tv_xor_b32_e32 v8, 0x408, v5\n\ts_add_u32 s24, s24, s58\n\tv_lshl_add_u32 v203, v8, 1, 0\n\tv_xor_b32_e32 v8, 0x810, v5\n\tv_xor_b32_e32 v5, 0xc18, v5\n\ts_addc_u32 s25, s25, s59\n\tv_lshl_add_u32 v205, v5, 1, 0\n\tv_or_b32_e32 v5, v155, v4\n\ts_add_u32 s24, s24, s56\n\tv_xor_b32_e32 v175, v6, v5\n\tv_xor_b32_e32 v5, v3, v5\n\tv_or_b32_e32 v4, v154, v4\n\ts_addc_u32 s25, s25, s57\n\tv_xor_b32_e32 v178, v5, v2\n\tv_xor_b32_e32 v3, v3, v4\n\ts_add_u32 s24, s24, s34\n\tv_lshl_add_u32 v204, v8, 1, 0\n\tv_xor_b32_e32 v8, v175, v7\n\tv_xor_b32_e32 v5, v178, v7\n\tv_xor_b32_e32 v176, v3, v2\n\ts_addc_u32 s25, s25, s35\n\tv_lshl_add_u32 v206, v8, 1, 0\n\tv_xor_b32_e32 v8, 0x408, v5\n\tv_xor_b32_e32 v2, v176, v7\n\ts_add_u32 s24, s4, s24\n\tv_lshl_add_u32 v207, v8, 1, 0\n\tv_xor_b32_e32 v8, 0x810, v5\n\tv_xor_b32_e32 v5, 0xc18, v5\n\tv_xor_b32_e32 v166, v6, v4\n\tv_xor_b32_e32 v3, 0x408, v2\n\ts_addc_u32 s25, s5, s25\n\ts_min_u32 s26, s29, 2\n\ts_not_b32 s27, s29\n\tv_lshl_add_u32 v209, v5, 1, 0\n\tv_xor_b32_e32 v5, v166, v7\n\tv_lshl_add_u32 v211, v3, 1, 0\n\tv_xor_b32_e32 v3, 0x810, v2\n\tv_xor_b32_e32 v2, 0xc18, v2\n\ts_add_i32 s26, s27, s26\n\tv_mov_b32_e32 v50, 0\n\t.loc\t1 222 12                        ; 7.ttgir:222:12\n\ts_waitcnt vmcnt(2)\n\tv_lshrrev_b32_e32 v187, 16, v118\n\tv_lshl_add_u32 v181, v165, 1, 0\n\tv_lshl_add_u32 v182, v168, 1, 0\n\tv_lshl_add_u32 v183, v169, 1, 0\n\tv_lshl_add_u32 v184, v170, 1, 0\n\tv_lshl_add_u32 v185, v172, 1, 0\n\tv_lshl_add_u32 v186, v174, 1, 0\n\tv_lshl_add_u32 v188, v142, 1, 0\n\tv_lshl_add_u32 v189, v143, 1, 0\n\tv_lshl_add_u32 v190, v144, 1, 0\n\tv_lshl_add_u32 v191, v145, 1, 0\n\tv_lshl_add_u32 v192, v146, 1, 0\n\tv_lshl_add_u32 v193, v147, 1, 0\n\tv_lshl_add_u32 v194, v149, 1, 0\n\tv_lshl_add_u32 v195, v150, 1, 0\n\tv_lshl_add_u32 v208, v8, 1, 0\n\tv_lshl_add_u32 v210, v5, 1, 0\n\tv_lshl_add_u32 v212, v3, 1, 0\n\tv_lshl_add_u32 v213, v2, 1, 0\n\tv_mov_b32_e32 v201, 1.0\n\tv_mov_b32_e32 v180, 0xff800000\n\tv_mov_b32_e32 v214, s26\n\ts_mov_b32 s26, 0x5040100\n\ts_mov_b32 s27, 0x7060302\n\ts_mov_b32 s29, 0x3e0293ee\n\tv_mov_b32_e32 v51, v50\n\tv_mov_b32_e32 v52, v50\n\tv_mov_b32_e32 v53, v50\n\tv_mov_b32_e32 v54, v50\n\tv_mov_b32_e32 v55, v50\n\tv_mov_b32_e32 v56, v50\n\tv_mov_b32_e32 v57, v50\n\tv_mov_b32_e32 v58, v50\n\tv_mov_b32_e32 v59, v50\n\tv_mov_b32_e32 v60, v50\n\tv_mov_b32_e32 v61, v50\n\tv_mov_b32_e32 v62, v50\n\tv_mov_b32_e32 v63, v50\n\tv_mov_b32_e32 v64, v50\n\tv_mov_b32_e32 v65, v50\n\tv_mov_b32_e32 v34, v50\n\tv_mov_b32_e32 v35, v50\n\tv_mov_b32_e32 v36, v50\n\tv_mov_b32_e32 v37, v50\n\tv_mov_b32_e32 v38, v50\n\tv_mov_b32_e32 v39, v50\n\tv_mov_b32_e32 v40, v50\n\tv_mov_b32_e32 v41, v50\n\tv_mov_b32_e32 v42, v50\n\tv_mov_b32_e32 v43, v50\n\tv_mov_b32_e32 v44, v50\n\tv_mov_b32_e32 v45, v50\n\tv_mov_b32_e32 v46, v50\n\tv_mov_b32_e32 v47, v50\n\tv_mov_b32_e32 v48, v50\n\tv_mov_b32_e32 v49, v50\n\tv_mov_b32_e32 v18, v50\n\tv_mov_b32_e32 v19, v50\n\tv_mov_b32_e32 v20, v50\n\tv_mov_b32_e32 v21, v50\n\tv_mov_b32_e32 v22, v50\n\tv_mov_b32_e32 v23, v50\n\tv_mov_b32_e32 v24, v50\n\tv_mov_b32_e32 v25, v50\n\tv_mov_b32_e32 v26, v50\n\tv_mov_b32_e32 v27, v50\n\tv_mov_b32_e32 v28, v50\n\tv_mov_b32_e32 v29, v50\n\tv_mov_b32_e32 v30, v50\n\tv_mov_b32_e32 v31, v50\n\tv_mov_b32_e32 v32, v50\n\tv_mov_b32_e32 v33, v50\n\tv_mov_b32_e32 v2, v50\n\tv_mov_b32_e32 v3, v50\n\tv_mov_b32_e32 v4, v50\n\tv_mov_b32_e32 v5, v50\n\tv_mov_b32_e32 v6, v50\n\tv_mov_b32_e32 v7, v50\n\tv_mov_b32_e32 v8, v50\n\tv_mov_b32_e32 v9, v50\n\tv_mov_b32_e32 v10, v50\n\tv_mov_b32_e32 v11, v50\n\tv_mov_b32_e32 v12, v50\n\tv_mov_b32_e32 v13, v50\n\tv_mov_b32_e32 v14, v50\n\tv_mov_b32_e32 v15, v50\n\tv_mov_b32_e32 v16, v50\n\tv_mov_b32_e32 v17, v50\n.LBB0_5:                                ; =>This Inner Loop Header: Depth=1\n\t.loc\t1 320 12                        ; 7.ttgir:320:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[110:111], 0\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[112:113], v[66:81]\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 319 12                        ; 7.ttgir:319:12\n\tds_read_b128 v[126:129], v181\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 322 12                        ; 7.ttgir:322:12\n\tds_read_b128 v[216:219], v182\n\t.loc\t1 324 12                        ; 7.ttgir:324:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[106:107], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[108:109], v[66:81]\n\t.loc\t1 325 12                        ; 7.ttgir:325:12\n\tds_read_b128 v[122:125], v183\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 327 12                        ; 7.ttgir:327:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[102:103], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[104:105], v[66:81]\n\t.loc\t1 328 12                        ; 7.ttgir:328:12\n\tds_read_b128 v[126:129], v184\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 330 12                        ; 7.ttgir:330:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[216:217], v[98:99], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[218:219], v[100:101], v[66:81]\n\t.loc\t1 331 12                        ; 7.ttgir:331:12\n\tds_read_b128 v[216:219], v185\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 333 12                        ; 7.ttgir:333:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[94:95], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[96:97], v[66:81]\n\t.loc\t1 334 12                        ; 7.ttgir:334:12\n\tds_read_b128 v[122:125], v186\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 336 12                        ; 7.ttgir:336:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[90:91], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[92:93], v[66:81]\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 338 12                        ; 7.ttgir:338:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[216:217], v[86:87], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[218:219], v[88:89], v[66:81]\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 340 12                        ; 7.ttgir:340:12\n\ts_waitcnt lgkmcnt(0)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[82:83], v[66:81]\n\tv_mov_b32_e32 v122, v201\n\tv_mov_b32_e32 v123, v180\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[84:85], v[66:81]\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 342 5                         ; 7.ttgir:342:5\n\ts_setprio 0\n\t.loc\t1 354 5                         ; 7.ttgir:354:5\n\ts_waitcnt vmcnt(2)\n\tv_perm_b32 v118, v118, v114, s26\n\tv_alignbit_b32 v114, v187, v114, 16\n\t.loc\t1 347 5                         ; 7.ttgir:347:5\n\ts_barrier\n\t.loc\t1 354 5                         ; 7.ttgir:354:5\n\tds_write_b32 v188, v118 offset:8192\n\tds_write_b32 v189, v114 offset:8192\n\tv_perm_b32 v114, v119, v115, s26\n\tds_write_b32 v190, v114 offset:8192\n\tv_perm_b32 v114, v119, v115, s27\n\tds_write_b32 v191, v114 offset:8192\n\tv_perm_b32 v114, v120, v116, s26\n\tds_write_b32 v192, v114 offset:8192\n\tv_perm_b32 v114, v120, v116, s27\n\tds_write_b32 v193, v114 offset:8192\n\tv_perm_b32 v114, v121, v117, s26\n\tds_write_b32 v194, v114 offset:8192\n\tv_perm_b32 v114, v121, v117, s27\n\t.loc\t1 359 13                        ; 7.ttgir:359:13\n\ts_and_b32 s37, s23, 0xffff\n\t.loc\t1 354 5                         ; 7.ttgir:354:5\n\tds_write_b32 v195, v114 offset:8192\n\t.loc\t1 359 13                        ; 7.ttgir:359:13\n\ts_or_b32 s49, s37, s55\n\ts_mov_b32 s48, s22\n\t; sched_barrier mask(0x0000040F)\n\tbuffer_load_dwordx4 v[118:121], v164, s[48:51], 0 offen\n\tbuffer_load_dwordx4 v[114:117], v163, s[48:51], 0 offen\n\t.loc\t1 379 12                        ; 7.ttgir:379:12\n\tv_fma_f32 v66, v66, s29, 0\n\tv_fma_f32 v67, v67, s29, 0\n\tv_fma_f32 v68, v68, s29, 0\n\tv_fma_f32 v69, v69, s29, 0\n\t.loc\t1 389 15                        ; 7.ttgir:389:15\n\tv_max_f32_e32 v124, v66, v67\n\t.loc\t1 381 12                        ; 7.ttgir:381:12\n\tv_fma_f32 v70, v70, s29, 0\n\tv_fma_f32 v71, v71, s29, 0\n\t.loc\t1 389 15                        ; 7.ttgir:389:15\n\tv_max3_f32 v124, v124, v68, v69\n\t.loc\t1 381 12                        ; 7.ttgir:381:12\n\tv_fma_f32 v72, v72, s29, 0\n\tv_fma_f32 v73, v73, s29, 0\n\t.loc\t1 389 15                        ; 7.ttgir:389:15\n\tv_max3_f32 v124, v124, v70, v71\n\t.loc\t1 383 12                        ; 7.ttgir:383:12\n\tv_fma_f32 v74, v74, s29, 0\n\tv_fma_f32 v75, v75, s29, 0\n\t.loc\t1 389 15                        ; 7.ttgir:389:15\n\tv_max3_f32 v124, v124, v72, v73\n\t.loc\t1 383 12                        ; 7.ttgir:383:12\n\tv_fma_f32 v76, v76, s29, 0\n\tv_fma_f32 v77, v77, s29, 0\n\t.loc\t1 389 15                        ; 7.ttgir:389:15\n\tv_max3_f32 v124, v124, v74, v75\n\t.loc\t1 385 12                        ; 7.ttgir:385:12\n\tv_fma_f32 v78, v78, s29, 0\n\tv_fma_f32 v79, v79, s29, 0\n\t.loc\t1 389 15                        ; 7.ttgir:389:15\n\tv_max3_f32 v124, v124, v76, v77\n\t.loc\t1 385 12                        ; 7.ttgir:385:12\n\tv_fma_f32 v80, v80, s29, 0\n\tv_fma_f32 v81, v81, s29, 0\n\t.loc\t1 389 15                        ; 7.ttgir:389:15\n\tv_max3_f32 v124, v124, v78, v79\n\tv_max3_f32 v124, v124, v80, v81\n\t; sched_barrier mask(0x0000040F)\n\t.loc\t1 387 12                        ; 7.ttgir:387:12\n\tds_bpermute_b32 v125, v157, v124\n\t.loc\t1 392 12                        ; 7.ttgir:392:12\n\ts_waitcnt lgkmcnt(0)\n\tv_max3_f32 v180, v123, v124, v125\n\t.loc\t1 359 13                        ; 7.ttgir:359:13\n\ts_waitcnt vmcnt(1)\n\tv_lshrrev_b32_e32 v187, 16, v118\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 416 14                        ; 7.ttgir:416:14\n\ts_and_b32 s37, s25, 0xffff\n\ts_mov_b32 s48, s24\n\ts_or_b32 s49, s37, s53\n\t.loc\t1 407 5                         ; 7.ttgir:407:5\n\tds_write_b128 v196, v[134:137]\n\t; sched_barrier mask(0x0000040F)\n\t.loc\t1 416 14                        ; 7.ttgir:416:14\n\tbuffer_load_dwordx4 v[134:137], v160, s[48:51], 0 offen\n\t; sched_barrier mask(0x0000040F)\n\t.loc\t1 420 5                         ; 7.ttgir:420:5\n\tds_write_b128 v196, v[130:133] offset:4096\n\t; sched_barrier mask(0x0000040F)\n\t.loc\t1 423 14                        ; 7.ttgir:423:14\n\tbuffer_load_dwordx4 v[130:133], v161, s[48:51], 0 offen\n\t.loc\t1 426 12                        ; 7.ttgir:426:12\n\tv_sub_f32_e32 v66, v66, v180\n\tv_sub_f32_e32 v67, v67, v180\n\tv_sub_f32_e32 v68, v68, v180\n\t.loc\t1 427 12                        ; 7.ttgir:427:12\n\tv_exp_f32_e32 v66, v66\n\tv_exp_f32_e32 v67, v67\n\t.loc\t1 426 12                        ; 7.ttgir:426:12\n\tv_sub_f32_e32 v69, v69, v180\n\t.loc\t1 427 12                        ; 7.ttgir:427:12\n\tv_exp_f32_e32 v68, v68\n\t.loc\t1 429 12                        ; 7.ttgir:429:12\n\tv_sub_f32_e32 v70, v70, v180\n\t.loc\t1 427 12                        ; 7.ttgir:427:12\n\tv_exp_f32_e32 v69, v69\n\t.loc\t1 429 12                        ; 7.ttgir:429:12\n\tv_sub_f32_e32 v71, v71, v180\n\t.loc\t1 430 12                        ; 7.ttgir:430:12\n\tv_exp_f32_e32 v70, v70\n\t.loc\t1 429 12                        ; 7.ttgir:429:12\n\tv_sub_f32_e32 v72, v72, v180\n\t.loc\t1 430 12                        ; 7.ttgir:430:12\n\tv_exp_f32_e32 v71, v71\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v66, v67\n\t.loc\t1 429 12                        ; 7.ttgir:429:12\n\tv_sub_f32_e32 v73, v73, v180\n\t.loc\t1 430 12                        ; 7.ttgir:430:12\n\tv_exp_f32_e32 v72, v72\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v68, v124\n\t.loc\t1 432 12                        ; 7.ttgir:432:12\n\tv_sub_f32_e32 v74, v74, v180\n\t.loc\t1 430 12                        ; 7.ttgir:430:12\n\tv_exp_f32_e32 v73, v73\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v69, v124\n\t.loc\t1 432 12                        ; 7.ttgir:432:12\n\tv_sub_f32_e32 v75, v75, v180\n\t.loc\t1 433 12                        ; 7.ttgir:433:12\n\tv_exp_f32_e32 v74, v74\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v70, v124\n\t.loc\t1 432 12                        ; 7.ttgir:432:12\n\tv_sub_f32_e32 v76, v76, v180\n\t.loc\t1 433 12                        ; 7.ttgir:433:12\n\tv_exp_f32_e32 v75, v75\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v71, v124\n\t.loc\t1 432 12                        ; 7.ttgir:432:12\n\tv_sub_f32_e32 v77, v77, v180\n\t.loc\t1 433 12                        ; 7.ttgir:433:12\n\tv_exp_f32_e32 v76, v76\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v72, v124\n\t.loc\t1 435 12                        ; 7.ttgir:435:12\n\tv_sub_f32_e32 v78, v78, v180\n\t.loc\t1 433 12                        ; 7.ttgir:433:12\n\tv_exp_f32_e32 v77, v77\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v73, v124\n\t.loc\t1 435 12                        ; 7.ttgir:435:12\n\tv_sub_f32_e32 v79, v79, v180\n\t.loc\t1 436 12                        ; 7.ttgir:436:12\n\tv_exp_f32_e32 v78, v78\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v74, v124\n\t.loc\t1 435 12                        ; 7.ttgir:435:12\n\tv_sub_f32_e32 v80, v80, v180\n\t.loc\t1 436 12                        ; 7.ttgir:436:12\n\tv_exp_f32_e32 v79, v79\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v75, v124\n\t.loc\t1 435 12                        ; 7.ttgir:435:12\n\tv_sub_f32_e32 v81, v81, v180\n\t.loc\t1 436 12                        ; 7.ttgir:436:12\n\tv_exp_f32_e32 v80, v80\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v76, v124\n\t.loc\t1 436 12                        ; 7.ttgir:436:12\n\tv_exp_f32_e32 v81, v81\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\tv_add_f32_e32 v124, v77, v124\n\tv_add_f32_e32 v124, v78, v124\n\tv_add_f32_e32 v124, v79, v124\n\tv_add_f32_e32 v124, v80, v124\n\tv_add_f32_e32 v124, v81, v124\n\t; sched_barrier mask(0x0000040F)\n\t.loc\t1 438 12                        ; 7.ttgir:438:12\n\tds_bpermute_b32 v125, v157, v124\n\t.loc\t1 443 12                        ; 7.ttgir:443:12\n\tv_sub_f32_e32 v123, v123, v180\n\t.loc\t1 444 12                        ; 7.ttgir:444:12\n\tv_exp_f32_e32 v123, v123\n\t.loc\t1 440 15                        ; 7.ttgir:440:15\n\ts_waitcnt lgkmcnt(0)\n\tv_add_f32_e32 v201, v124, v125\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 483 12                        ; 7.ttgir:483:12\n\tv_cvt_f16_f32_e32 v124, v66\n\tv_cvt_f16_f32_e32 v125, v67\n\tv_cvt_f16_f32_e32 v126, v68\n\tv_cvt_f16_f32_e32 v127, v69\n\t.loc\t1 484 12                        ; 7.ttgir:484:12\n\tv_cvt_f16_f32_e32 v128, v70\n\tv_cvt_f16_f32_e32 v129, v71\n\t.loc\t1 488 5                         ; 7.ttgir:488:5\n\ts_barrier\n\t.loc\t1 496 12                        ; 7.ttgir:496:12\n\tds_read_b64 v[66:67], v197 offset:8192\n\t.loc\t1 455 12                        ; 7.ttgir:455:12\n\tv_mul_f32_e32 v50, v50, v123\n\tv_mul_f32_e32 v51, v51, v123\n\tv_mul_f32_e32 v52, v52, v123\n\tv_mul_f32_e32 v53, v53, v123\n\t.loc\t1 458 12                        ; 7.ttgir:458:12\n\tv_mul_f32_e32 v54, v54, v123\n\tv_mul_f32_e32 v55, v55, v123\n\tv_mul_f32_e32 v56, v56, v123\n\tv_mul_f32_e32 v57, v57, v123\n\t.loc\t1 461 12                        ; 7.ttgir:461:12\n\tv_mul_f32_e32 v58, v58, v123\n\tv_mul_f32_e32 v59, v59, v123\n\tv_mul_f32_e32 v60, v60, v123\n\tv_mul_f32_e32 v61, v61, v123\n\t.loc\t1 464 12                        ; 7.ttgir:464:12\n\tv_mul_f32_e32 v62, v62, v123\n\tv_mul_f32_e32 v63, v63, v123\n\tv_mul_f32_e32 v64, v64, v123\n\tv_mul_f32_e32 v65, v65, v123\n\t.loc\t1 484 12                        ; 7.ttgir:484:12\n\tv_cvt_f16_f32_e32 v215, v72\n\tv_cvt_f16_f32_e32 v216, v73\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 504 12                        ; 7.ttgir:504:12\n\tds_read_b64 v[68:69], v198 offset:8192\n\t.loc\t1 498 12                        ; 7.ttgir:498:12\n\tv_mul_f32_e32 v34, v34, v123\n\tv_mul_f32_e32 v35, v35, v123\n\tv_mul_f32_e32 v36, v36, v123\n\tv_mul_f32_e32 v37, v37, v123\n\t.loc\t1 501 12                        ; 7.ttgir:501:12\n\tv_mul_f32_e32 v38, v38, v123\n\tv_mul_f32_e32 v39, v39, v123\n\tv_mul_f32_e32 v40, v40, v123\n\tv_mul_f32_e32 v41, v41, v123\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 512 12                        ; 7.ttgir:512:12\n\tds_read_b64 v[70:71], v199 offset:8192\n\t.loc\t1 506 12                        ; 7.ttgir:506:12\n\tv_mul_f32_e32 v42, v42, v123\n\tv_mul_f32_e32 v43, v43, v123\n\tv_mul_f32_e32 v44, v44, v123\n\tv_mul_f32_e32 v45, v45, v123\n\t.loc\t1 509 12                        ; 7.ttgir:509:12\n\tv_mul_f32_e32 v46, v46, v123\n\tv_mul_f32_e32 v47, v47, v123\n\tv_mul_f32_e32 v48, v48, v123\n\tv_mul_f32_e32 v49, v49, v123\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 516 12                        ; 7.ttgir:516:12\n\tv_mul_f32_e32 v18, v18, v123\n\tv_mul_f32_e32 v19, v19, v123\n\tv_mul_f32_e32 v20, v20, v123\n\tv_mul_f32_e32 v21, v21, v123\n\t.loc\t1 519 12                        ; 7.ttgir:519:12\n\tv_mul_f32_e32 v22, v22, v123\n\tv_mul_f32_e32 v23, v23, v123\n\tv_mul_f32_e32 v24, v24, v123\n\tv_mul_f32_e32 v25, v25, v123\n\t.loc\t1 522 12                        ; 7.ttgir:522:12\n\tv_mul_f32_e32 v26, v26, v123\n\tv_mul_f32_e32 v27, v27, v123\n\tv_mul_f32_e32 v28, v28, v123\n\tv_mul_f32_e32 v29, v29, v123\n\t.loc\t1 525 12                        ; 7.ttgir:525:12\n\tv_mul_f32_e32 v30, v30, v123\n\tv_mul_f32_e32 v31, v31, v123\n\tv_mul_f32_e32 v32, v32, v123\n\tv_mul_f32_e32 v33, v33, v123\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 535 12                        ; 7.ttgir:535:12\n\tv_mul_f32_e32 v2, v2, v123\n\tv_mul_f32_e32 v3, v3, v123\n\tv_mul_f32_e32 v4, v4, v123\n\tv_mul_f32_e32 v5, v5, v123\n\t.loc\t1 538 12                        ; 7.ttgir:538:12\n\tv_mul_f32_e32 v6, v6, v123\n\tv_mul_f32_e32 v7, v7, v123\n\tv_mul_f32_e32 v8, v8, v123\n\tv_mul_f32_e32 v9, v9, v123\n\t.loc\t1 541 12                        ; 7.ttgir:541:12\n\tv_mul_f32_e32 v10, v10, v123\n\tv_mul_f32_e32 v11, v11, v123\n\tv_mul_f32_e32 v12, v12, v123\n\tv_mul_f32_e32 v13, v13, v123\n\t.loc\t1 544 12                        ; 7.ttgir:544:12\n\tv_mul_f32_e32 v14, v14, v123\n\tv_mul_f32_e32 v15, v15, v123\n\tv_mul_f32_e32 v16, v16, v123\n\tv_mul_f32_e32 v17, v17, v123\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 553 12                        ; 7.ttgir:553:12\n\tv_fmac_f32_e32 v201, v122, v123\n\t.loc\t1 548 12                        ; 7.ttgir:548:12\n\tv_cvt_f16_f32_e32 v217, v74\n\tv_cvt_f16_f32_e32 v218, v75\n\tv_cvt_f16_f32_e32 v76, v76\n\tv_cvt_f16_f32_e32 v77, v77\n\t.loc\t1 549 12                        ; 7.ttgir:549:12\n\tv_cvt_f16_f32_e32 v78, v78\n\tv_cvt_f16_f32_e32 v79, v79\n\tv_cvt_f16_f32_e32 v80, v80\n\tv_cvt_f16_f32_e32 v81, v81\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 558 5                         ; 7.ttgir:558:5\n\ts_setprio 3\n\t; sched_barrier mask(0x00000000)\n\t.loc\t1 561 12                        ; 7.ttgir:561:12\n\tv_pack_b32_f16 v73, v126, v127\n\tv_pack_b32_f16 v72, v124, v125\n\t.loc\t1 297 12                        ; 7.ttgir:297:12\n\ts_add_u32 s22, s22, s20\n\ts_addc_u32 s23, s23, s21\n\t.loc\t1 561 12                        ; 7.ttgir:561:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[66:67], v[72:73], v[50:65]\n\t.loc\t1 562 12                        ; 7.ttgir:562:12\n\tds_read_b64 v[66:67], v200 offset:8192\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 297 12                        ; 7.ttgir:297:12\n\ts_add_u32 s24, s24, s64\n\tv_add_co_u32_e32 v214, vcc, 1, v214\n\ts_addc_u32 s25, s25, s65\n\t.loc\t1 298 5                         ; 7.ttgir:298:5\n\ts_andn2_b64 vcc, exec, vcc\n\t.loc\t1 564 12                        ; 7.ttgir:564:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[68:69], v[72:73], v[34:49]\n\t.loc\t1 565 12                        ; 7.ttgir:565:12\n\tds_read_b64 v[68:69], v202 offset:8192\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 570 12                        ; 7.ttgir:570:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[72:73], v[2:17]\n\t.loc\t1 573 12                        ; 7.ttgir:573:12\n\tv_pack_b32_f16 v67, v215, v216\n\tv_pack_b32_f16 v66, v128, v129\n\t.loc\t1 567 12                        ; 7.ttgir:567:12\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[70:71], v[72:73], v[18:33]\n\t.loc\t1 568 12                        ; 7.ttgir:568:12\n\tds_read_b64 v[70:71], v203 offset:8192\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 571 12                        ; 7.ttgir:571:12\n\tds_read_b64 v[74:75], v204 offset:8192\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 573 12                        ; 7.ttgir:573:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[66:67], v[50:65]\n\t.loc\t1 574 12                        ; 7.ttgir:574:12\n\tds_read_b64 v[68:69], v205 offset:8192\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 576 12                        ; 7.ttgir:576:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[66:67], v[34:49]\n\t.loc\t1 577 12                        ; 7.ttgir:577:12\n\tds_read_b64 v[70:71], v206 offset:8192\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 579 12                        ; 7.ttgir:579:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[66:67], v[18:33]\n\t.loc\t1 580 12                        ; 7.ttgir:580:12\n\tds_read_b64 v[72:73], v207 offset:8192\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 583 12                        ; 7.ttgir:583:12\n\tds_read_b64 v[74:75], v208 offset:8192\n\t.loc\t1 582 12                        ; 7.ttgir:582:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[66:67], v[2:17]\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 586 12                        ; 7.ttgir:586:12\n\tds_read_b64 v[68:69], v209 offset:8192\n\t.loc\t1 585 12                        ; 7.ttgir:585:12\n\tv_pack_b32_f16 v67, v76, v77\n\tv_pack_b32_f16 v66, v217, v218\n\ts_waitcnt lgkmcnt(3)\n\ts_nop 0\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[70:71], v[66:67], v[50:65]\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 589 12                        ; 7.ttgir:589:12\n\tds_read_b64 v[70:71], v210 offset:8192\n\t.loc\t1 588 12                        ; 7.ttgir:588:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[72:73], v[66:67], v[34:49]\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 591 12                        ; 7.ttgir:591:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[66:67], v[18:33]\n\t.loc\t1 592 12                        ; 7.ttgir:592:12\n\tds_read_b64 v[72:73], v211 offset:8192\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 595 12                        ; 7.ttgir:595:12\n\tds_read_b64 v[74:75], v212 offset:8192\n\t.loc\t1 594 12                        ; 7.ttgir:594:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[66:67], v[2:17]\n\t.loc\t1 596 12                        ; 7.ttgir:596:12\n\tds_read_b64 v[66:67], v213 offset:8192\n\t.loc\t1 599 12                        ; 7.ttgir:599:12\n\tv_pack_b32_f16 v69, v80, v81\n\tv_pack_b32_f16 v68, v78, v79\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 601 13                        ; 7.ttgir:601:13\n\tds_read_b128 v[126:129], v177\n\t.loc\t1 599 12                        ; 7.ttgir:599:12\n\ts_waitcnt lgkmcnt(4)\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[70:71], v[68:69], v[50:65]\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 606 13                        ; 7.ttgir:606:13\n\tds_read_b128 v[122:125], v179\n\t.loc\t1 604 12                        ; 7.ttgir:604:12\n\ts_waitcnt lgkmcnt(4)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[72:73], v[68:69], v[34:49]\n\t; sched_barrier mask(0x00000406)\n\t.loc\t1 609 12                        ; 7.ttgir:609:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[74:75], v[68:69], v[18:33]\n\t.loc\t1 610 12                        ; 7.ttgir:610:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[66:67], v[68:69], v[2:17]\n\t.loc\t1 298 5                         ; 7.ttgir:298:5\n\ts_cbranch_vccnz .LBB0_5\n; %bb.6:                                ; %._crit_edge.loopexit\n\t.loc\t1 0 5 is_stmt 0                 ; 7.ttgir:0:5\n\ts_waitcnt vmcnt(1)\n\tv_mov_b32_e32 v135, v165\n\tv_mov_b32_e32 v134, v168\n\ts_waitcnt vmcnt(0)\n\tv_mov_b32_e32 v133, v169\n\tv_mov_b32_e32 v132, v170\n\tv_mov_b32_e32 v131, v172\n\tv_mov_b32_e32 v130, v174\n\ts_branch .LBB0_8\n.LBB0_7:\n\tv_mov_b32_e32 v17, 0\n\tv_mov_b32_e32 v201, 1.0\n\tv_mov_b32_e32 v180, 0xff800000\n\tv_mov_b32_e32 v16, v17\n\tv_mov_b32_e32 v15, v17\n\tv_mov_b32_e32 v14, v17\n\tv_mov_b32_e32 v13, v17\n\tv_mov_b32_e32 v12, v17\n\tv_mov_b32_e32 v11, v17\n\tv_mov_b32_e32 v10, v17\n\tv_mov_b32_e32 v9, v17\n\tv_mov_b32_e32 v8, v17\n\tv_mov_b32_e32 v7, v17\n\tv_mov_b32_e32 v6, v17\n\tv_mov_b32_e32 v5, v17\n\tv_mov_b32_e32 v4, v17\n\tv_mov_b32_e32 v3, v17\n\tv_mov_b32_e32 v2, v17\n\tv_mov_b32_e32 v33, v17\n\tv_mov_b32_e32 v32, v17\n\tv_mov_b32_e32 v31, v17\n\tv_mov_b32_e32 v30, v17\n\tv_mov_b32_e32 v29, v17\n\tv_mov_b32_e32 v28, v17\n\tv_mov_b32_e32 v27, v17\n\tv_mov_b32_e32 v26, v17\n\tv_mov_b32_e32 v25, v17\n\tv_mov_b32_e32 v24, v17\n\tv_mov_b32_e32 v23, v17\n\tv_mov_b32_e32 v22, v17\n\tv_mov_b32_e32 v21, v17\n\tv_mov_b32_e32 v20, v17\n\tv_mov_b32_e32 v19, v17\n\tv_mov_b32_e32 v18, v17\n\tv_mov_b32_e32 v49, v17\n\tv_mov_b32_e32 v48, v17\n\tv_mov_b32_e32 v47, v17\n\tv_mov_b32_e32 v46, v17\n\tv_mov_b32_e32 v45, v17\n\tv_mov_b32_e32 v44, v17\n\tv_mov_b32_e32 v43, v17\n\tv_mov_b32_e32 v42, v17\n\tv_mov_b32_e32 v41, v17\n\tv_mov_b32_e32 v40, v17\n\tv_mov_b32_e32 v39, v17\n\tv_mov_b32_e32 v38, v17\n\tv_mov_b32_e32 v37, v17\n\tv_mov_b32_e32 v36, v17\n\tv_mov_b32_e32 v35, v17\n\tv_mov_b32_e32 v34, v17\n\tv_mov_b32_e32 v65, v17\n\tv_mov_b32_e32 v64, v17\n\tv_mov_b32_e32 v63, v17\n\tv_mov_b32_e32 v62, v17\n\tv_mov_b32_e32 v61, v17\n\tv_mov_b32_e32 v60, v17\n\tv_mov_b32_e32 v59, v17\n\tv_mov_b32_e32 v58, v17\n\tv_mov_b32_e32 v57, v17\n\tv_mov_b32_e32 v56, v17\n\tv_mov_b32_e32 v55, v17\n\tv_mov_b32_e32 v54, v17\n\tv_mov_b32_e32 v53, v17\n\tv_mov_b32_e32 v52, v17\n\tv_mov_b32_e32 v51, v17\n\tv_mov_b32_e32 v50, v17\n.LBB0_8:                                ; %._crit_edge\n\t.loc\t1 101 10 is_stmt 1              ; 7.ttgir:101:10\n\tv_or3_b32 v165, v151, v153, v152\n\t.loc\t1 110 11                        ; 7.ttgir:110:11\n\tv_or_b32_e32 v189, 1, v162\n\t.loc\t1 651 5                         ; 7.ttgir:651:5\n\ts_setprio 3\n\t.loc\t1 677 12                        ; 7.ttgir:677:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[110:111], 0\n\t.loc\t1 658 12                        ; 7.ttgir:658:12\n\tv_lshl_add_u32 v183, v135, 1, 0\n\t.loc\t1 652 5                         ; 7.ttgir:652:5\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\ts_waitcnt vmcnt(1)\n\t.loc\t1 658 12                        ; 7.ttgir:658:12\n\tds_read_b128 v[114:117], v183\n\t.loc\t1 660 12                        ; 7.ttgir:660:12\n\tv_lshl_add_u32 v184, v134, 1, 0\n\ts_waitcnt vmcnt(0)\n\tds_read_b128 v[118:121], v184\n\t.loc\t1 677 12                        ; 7.ttgir:677:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[112:113], v[66:81]\n\t.loc\t1 662 12                        ; 7.ttgir:662:12\n\tv_lshl_add_u32 v185, v133, 1, 0\n\t.loc\t1 664 12                        ; 7.ttgir:664:12\n\tv_lshl_add_u32 v186, v132, 1, 0\n\t.loc\t1 666 12                        ; 7.ttgir:666:12\n\tv_lshl_add_u32 v187, v131, 1, 0\n\t.loc\t1 668 12                        ; 7.ttgir:668:12\n\tv_lshl_add_u32 v188, v130, 1, 0\n\t.loc\t1 678 12                        ; 7.ttgir:678:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[106:107], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[108:109], v[66:81]\n\t.loc\t1 679 12                        ; 7.ttgir:679:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[102:103], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[104:105], v[66:81]\n\t.loc\t1 662 12                        ; 7.ttgir:662:12\n\tds_read_b128 v[114:117], v185\n\t.loc\t1 680 12                        ; 7.ttgir:680:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[98:99], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[100:101], v[66:81]\n\t.loc\t1 664 12                        ; 7.ttgir:664:12\n\tds_read_b128 v[118:121], v186\n\t.loc\t1 681 12                        ; 7.ttgir:681:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[94:95], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[96:97], v[66:81]\n\t.loc\t1 666 12                        ; 7.ttgir:666:12\n\tds_read_b128 v[114:117], v187\n\t.loc\t1 682 12                        ; 7.ttgir:682:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[90:91], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[92:93], v[66:81]\n\t.loc\t1 668 12                        ; 7.ttgir:668:12\n\tds_read_b128 v[118:121], v188\n\t.loc\t1 683 12                        ; 7.ttgir:683:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[86:87], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[88:89], v[66:81]\n\t.loc\t1 684 12                        ; 7.ttgir:684:12\n\ts_waitcnt lgkmcnt(0)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[82:83], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[84:85], v[66:81]\n\t.loc\t1 685 5                         ; 7.ttgir:685:5\n\ts_setprio 0\n\ts_mov_b32 s20, 0x3e0293ee\n\t.loc\t1 699 12                        ; 7.ttgir:699:12\n\ts_nop 7\n\ts_nop 0\n\tv_fma_f32 v66, v66, s20, 0\n\tv_fma_f32 v67, v67, s20, 0\n\tv_fma_f32 v68, v68, s20, 0\n\tv_fma_f32 v69, v69, s20, 0\n\t.loc\t1 709 15                        ; 7.ttgir:709:15\n\tv_max_f32_e32 v114, v66, v67\n\t.loc\t1 701 12                        ; 7.ttgir:701:12\n\tv_fma_f32 v70, v70, s20, 0\n\tv_fma_f32 v71, v71, s20, 0\n\t.loc\t1 709 15                        ; 7.ttgir:709:15\n\tv_max3_f32 v114, v114, v68, v69\n\t.loc\t1 701 12                        ; 7.ttgir:701:12\n\tv_fma_f32 v72, v72, s20, 0\n\tv_fma_f32 v73, v73, s20, 0\n\t.loc\t1 709 15                        ; 7.ttgir:709:15\n\tv_max3_f32 v114, v114, v70, v71\n\t.loc\t1 703 12                        ; 7.ttgir:703:12\n\tv_fma_f32 v74, v74, s20, 0\n\tv_fma_f32 v75, v75, s20, 0\n\t.loc\t1 709 15                        ; 7.ttgir:709:15\n\tv_max3_f32 v114, v114, v72, v73\n\t.loc\t1 703 12                        ; 7.ttgir:703:12\n\tv_fma_f32 v76, v76, s20, 0\n\tv_fma_f32 v77, v77, s20, 0\n\t.loc\t1 709 15                        ; 7.ttgir:709:15\n\tv_max3_f32 v114, v114, v74, v75\n\t.loc\t1 705 12                        ; 7.ttgir:705:12\n\tv_fma_f32 v78, v78, s20, 0\n\tv_fma_f32 v79, v79, s20, 0\n\t.loc\t1 709 15                        ; 7.ttgir:709:15\n\tv_max3_f32 v114, v114, v76, v77\n\t.loc\t1 705 12                        ; 7.ttgir:705:12\n\tv_fma_f32 v80, v80, s20, 0\n\tv_fma_f32 v81, v81, s20, 0\n\t.loc\t1 709 15                        ; 7.ttgir:709:15\n\tv_max3_f32 v114, v114, v78, v79\n\tv_max3_f32 v114, v114, v80, v81\n\t.loc\t1 707 12                        ; 7.ttgir:707:12\n\tds_bpermute_b32 v115, v157, v114\n\t.loc\t1 712 12                        ; 7.ttgir:712:12\n\ts_waitcnt lgkmcnt(0)\n\tv_max3_f32 v123, v180, v114, v115\n\t.loc\t1 716 12                        ; 7.ttgir:716:12\n\tv_sub_f32_e32 v66, v66, v123\n\tv_sub_f32_e32 v67, v67, v123\n\tv_sub_f32_e32 v68, v68, v123\n\t.loc\t1 723 12                        ; 7.ttgir:723:12\n\tv_exp_f32_e32 v114, v66\n\tv_exp_f32_e32 v67, v67\n\t.loc\t1 716 12                        ; 7.ttgir:716:12\n\tv_sub_f32_e32 v69, v69, v123\n\t.loc\t1 723 12                        ; 7.ttgir:723:12\n\tv_exp_f32_e32 v115, v68\n\t.loc\t1 718 12                        ; 7.ttgir:718:12\n\tv_sub_f32_e32 v70, v70, v123\n\t.loc\t1 723 12                        ; 7.ttgir:723:12\n\tv_exp_f32_e32 v116, v69\n\t.loc\t1 718 12                        ; 7.ttgir:718:12\n\tv_sub_f32_e32 v71, v71, v123\n\t.loc\t1 724 12                        ; 7.ttgir:724:12\n\tv_exp_f32_e32 v117, v70\n\t.loc\t1 718 12                        ; 7.ttgir:718:12\n\tv_sub_f32_e32 v72, v72, v123\n\t.loc\t1 724 12                        ; 7.ttgir:724:12\n\tv_exp_f32_e32 v118, v71\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v114, v67\n\t.loc\t1 718 12                        ; 7.ttgir:718:12\n\tv_sub_f32_e32 v73, v73, v123\n\t.loc\t1 724 12                        ; 7.ttgir:724:12\n\tv_exp_f32_e32 v119, v72\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v115, v66\n\t.loc\t1 720 12                        ; 7.ttgir:720:12\n\tv_sub_f32_e32 v74, v74, v123\n\t.loc\t1 724 12                        ; 7.ttgir:724:12\n\tv_exp_f32_e32 v120, v73\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v116, v66\n\t.loc\t1 720 12                        ; 7.ttgir:720:12\n\tv_sub_f32_e32 v75, v75, v123\n\t.loc\t1 725 12                        ; 7.ttgir:725:12\n\tv_exp_f32_e32 v121, v74\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v117, v66\n\t.loc\t1 720 12                        ; 7.ttgir:720:12\n\tv_sub_f32_e32 v76, v76, v123\n\t.loc\t1 725 12                        ; 7.ttgir:725:12\n\tv_exp_f32_e32 v124, v75\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v118, v66\n\t.loc\t1 720 12                        ; 7.ttgir:720:12\n\tv_sub_f32_e32 v77, v77, v123\n\t.loc\t1 725 12                        ; 7.ttgir:725:12\n\tv_exp_f32_e32 v125, v76\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v119, v66\n\t.loc\t1 722 12                        ; 7.ttgir:722:12\n\tv_sub_f32_e32 v78, v78, v123\n\t.loc\t1 725 12                        ; 7.ttgir:725:12\n\tv_exp_f32_e32 v126, v77\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v120, v66\n\t.loc\t1 722 12                        ; 7.ttgir:722:12\n\tv_sub_f32_e32 v79, v79, v123\n\t.loc\t1 726 12                        ; 7.ttgir:726:12\n\tv_exp_f32_e32 v78, v78\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v121, v66\n\t.loc\t1 722 12                        ; 7.ttgir:722:12\n\tv_sub_f32_e32 v80, v80, v123\n\t.loc\t1 726 12                        ; 7.ttgir:726:12\n\tv_exp_f32_e32 v79, v79\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v124, v66\n\t.loc\t1 722 12                        ; 7.ttgir:722:12\n\tv_sub_f32_e32 v81, v81, v123\n\t.loc\t1 726 12                        ; 7.ttgir:726:12\n\tv_exp_f32_e32 v80, v80\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v125, v66\n\t.loc\t1 726 12                        ; 7.ttgir:726:12\n\tv_exp_f32_e32 v81, v81\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\tv_add_f32_e32 v66, v126, v66\n\tv_add_f32_e32 v66, v78, v66\n\tv_add_f32_e32 v66, v79, v66\n\tv_add_f32_e32 v66, v80, v66\n\tv_add_f32_e32 v68, v81, v66\n\t.loc\t1 728 12                        ; 7.ttgir:728:12\n\tds_bpermute_b32 v69, v157, v68\n\t.loc\t1 733 12                        ; 7.ttgir:733:12\n\tv_sub_f32_e32 v66, v180, v123\n\t.loc\t1 734 12                        ; 7.ttgir:734:12\n\tv_exp_f32_e32 v66, v66\n\t.loc\t1 730 15                        ; 7.ttgir:730:15\n\ts_waitcnt lgkmcnt(0)\n\tv_add_f32_e32 v122, v68, v69\n\t.loc\t1 737 5                         ; 7.ttgir:737:5\n\ts_setprio 2\n\t.loc\t1 788 12                        ; 7.ttgir:788:12\n\tv_mov_b32_e32 v68, 0x204\n\tv_cndmask_b32_e64 v127, v68, 0, s[14:15]\n\tv_xor_b32_e32 v68, v167, v127\n\t.loc\t1 790 12                        ; 7.ttgir:790:12\n\tv_xor_b32_e32 v69, 0x408, v68\n\t.loc\t1 788 12                        ; 7.ttgir:788:12\n\tv_lshl_add_u32 v168, v68, 1, 0\n\t.loc\t1 790 12                        ; 7.ttgir:790:12\n\tv_lshl_add_u32 v167, v69, 1, 0\n\t.loc\t1 792 12                        ; 7.ttgir:792:12\n\tv_xor_b32_e32 v69, 0x810, v68\n\t.loc\t1 794 12                        ; 7.ttgir:794:12\n\tv_xor_b32_e32 v68, 0xc18, v68\n\t.loc\t1 792 12                        ; 7.ttgir:792:12\n\tv_lshl_add_u32 v169, v69, 1, 0\n\t.loc\t1 794 12                        ; 7.ttgir:794:12\n\tv_lshl_add_u32 v170, v68, 1, 0\n\t.loc\t1 788 12                        ; 7.ttgir:788:12\n\tds_read_b64 v[68:69], v168 offset:8192\n\t.loc\t1 821 12                        ; 7.ttgir:821:12\n\tv_cvt_f16_f32_e32 v76, v114\n\tv_cvt_f16_f32_e32 v67, v67\n\tv_cvt_f16_f32_e32 v77, v115\n\tv_cvt_f16_f32_e32 v114, v116\n\t.loc\t1 740 12                        ; 7.ttgir:740:12\n\tv_mul_f32_e32 v50, v50, v66\n\t.loc\t1 833 12                        ; 7.ttgir:833:12\n\tv_pack_b32_f16 v76, v76, v67\n\t.loc\t1 796 12                        ; 7.ttgir:796:12\n\tv_xor_b32_e32 v67, v171, v127\n\t.loc\t1 740 12                        ; 7.ttgir:740:12\n\tv_mul_f32_e32 v51, v51, v66\n\tv_mul_f32_e32 v52, v52, v66\n\tv_mul_f32_e32 v53, v53, v66\n\t.loc\t1 743 12                        ; 7.ttgir:743:12\n\tv_mul_f32_e32 v54, v54, v66\n\tv_mul_f32_e32 v55, v55, v66\n\tv_mul_f32_e32 v56, v56, v66\n\tv_mul_f32_e32 v57, v57, v66\n\t.loc\t1 746 12                        ; 7.ttgir:746:12\n\tv_mul_f32_e32 v58, v58, v66\n\tv_mul_f32_e32 v59, v59, v66\n\tv_mul_f32_e32 v60, v60, v66\n\tv_mul_f32_e32 v61, v61, v66\n\t.loc\t1 749 12                        ; 7.ttgir:749:12\n\tv_mul_f32_e32 v62, v62, v66\n\tv_mul_f32_e32 v63, v63, v66\n\tv_mul_f32_e32 v64, v64, v66\n\tv_mul_f32_e32 v65, v65, v66\n\t.loc\t1 833 12                        ; 7.ttgir:833:12\n\tv_pack_b32_f16 v77, v77, v114\n\t.loc\t1 796 12                        ; 7.ttgir:796:12\n\tv_lshl_add_u32 v172, v67, 1, 0\n\t.loc\t1 798 12                        ; 7.ttgir:798:12\n\tv_xor_b32_e32 v67, v173, v127\n\t.loc\t1 790 12                        ; 7.ttgir:790:12\n\tds_read_b64 v[70:71], v167 offset:8192\n\t.loc\t1 792 12                        ; 7.ttgir:792:12\n\tds_read_b64 v[72:73], v169 offset:8192\n\t.loc\t1 794 12                        ; 7.ttgir:794:12\n\tds_read_b64 v[74:75], v170 offset:8192\n\t.loc\t1 833 12                        ; 7.ttgir:833:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]\n\t.loc\t1 798 12                        ; 7.ttgir:798:12\n\tv_xor_b32_e32 v68, 0x408, v67\n\tv_lshl_add_u32 v171, v68, 1, 0\n\t.loc\t1 800 12                        ; 7.ttgir:800:12\n\tv_xor_b32_e32 v68, 0x810, v67\n\tv_lshl_add_u32 v173, v68, 1, 0\n\t.loc\t1 796 12                        ; 7.ttgir:796:12\n\tds_read_b64 v[68:69], v172 offset:8192\n\t.loc\t1 752 12                        ; 7.ttgir:752:12\n\tv_mul_f32_e32 v34, v34, v66\n\tv_mul_f32_e32 v35, v35, v66\n\tv_mul_f32_e32 v36, v36, v66\n\tv_mul_f32_e32 v37, v37, v66\n\t.loc\t1 755 12                        ; 7.ttgir:755:12\n\tv_mul_f32_e32 v38, v38, v66\n\tv_mul_f32_e32 v39, v39, v66\n\tv_mul_f32_e32 v40, v40, v66\n\tv_mul_f32_e32 v41, v41, v66\n\t.loc\t1 758 12                        ; 7.ttgir:758:12\n\tv_mul_f32_e32 v42, v42, v66\n\tv_mul_f32_e32 v43, v43, v66\n\tv_mul_f32_e32 v44, v44, v66\n\tv_mul_f32_e32 v45, v45, v66\n\t.loc\t1 761 12                        ; 7.ttgir:761:12\n\tv_mul_f32_e32 v46, v46, v66\n\tv_mul_f32_e32 v47, v47, v66\n\tv_mul_f32_e32 v48, v48, v66\n\tv_mul_f32_e32 v49, v49, v66\n\t.loc\t1 764 12                        ; 7.ttgir:764:12\n\tv_mul_f32_e32 v18, v18, v66\n\tv_mul_f32_e32 v19, v19, v66\n\tv_mul_f32_e32 v20, v20, v66\n\tv_mul_f32_e32 v21, v21, v66\n\t.loc\t1 767 12                        ; 7.ttgir:767:12\n\tv_mul_f32_e32 v22, v22, v66\n\tv_mul_f32_e32 v23, v23, v66\n\tv_mul_f32_e32 v24, v24, v66\n\tv_mul_f32_e32 v25, v25, v66\n\t.loc\t1 770 12                        ; 7.ttgir:770:12\n\tv_mul_f32_e32 v26, v26, v66\n\tv_mul_f32_e32 v27, v27, v66\n\tv_mul_f32_e32 v28, v28, v66\n\tv_mul_f32_e32 v29, v29, v66\n\t.loc\t1 773 12                        ; 7.ttgir:773:12\n\tv_mul_f32_e32 v30, v30, v66\n\tv_mul_f32_e32 v31, v31, v66\n\tv_mul_f32_e32 v32, v32, v66\n\tv_mul_f32_e32 v33, v33, v66\n\t.loc\t1 776 12                        ; 7.ttgir:776:12\n\tv_mul_f32_e32 v2, v2, v66\n\tv_mul_f32_e32 v3, v3, v66\n\tv_mul_f32_e32 v4, v4, v66\n\tv_mul_f32_e32 v5, v5, v66\n\t.loc\t1 779 12                        ; 7.ttgir:779:12\n\tv_mul_f32_e32 v6, v6, v66\n\tv_mul_f32_e32 v7, v7, v66\n\tv_mul_f32_e32 v8, v8, v66\n\tv_mul_f32_e32 v9, v9, v66\n\t.loc\t1 782 12                        ; 7.ttgir:782:12\n\tv_mul_f32_e32 v10, v10, v66\n\tv_mul_f32_e32 v11, v11, v66\n\tv_mul_f32_e32 v12, v12, v66\n\tv_mul_f32_e32 v13, v13, v66\n\t.loc\t1 785 12                        ; 7.ttgir:785:12\n\tv_mul_f32_e32 v14, v14, v66\n\tv_mul_f32_e32 v15, v15, v66\n\tv_mul_f32_e32 v16, v16, v66\n\tv_mul_f32_e32 v17, v17, v66\n\t.loc\t1 802 12                        ; 7.ttgir:802:12\n\tv_xor_b32_e32 v67, 0xc18, v67\n\t.loc\t1 834 12                        ; 7.ttgir:834:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]\n\t.loc\t1 802 12                        ; 7.ttgir:802:12\n\tv_lshl_add_u32 v174, v67, 1, 0\n\t.loc\t1 822 12                        ; 7.ttgir:822:12\n\tv_cvt_f16_f32_e32 v67, v117\n\tv_cvt_f16_f32_e32 v114, v118\n\t.loc\t1 857 12                        ; 7.ttgir:857:12\n\ts_mul_i32 s20, s33, s41\n\t.loc\t1 858 12                        ; 7.ttgir:858:12\n\ts_ashr_i32 s21, s20, 31\n\ts_lshl_b64 s[20:21], s[20:21], 1\n\ts_add_u32 s64, s28, s20\n\t.loc\t1 835 12                        ; 7.ttgir:835:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]\n\t.loc\t1 859 12                        ; 7.ttgir:859:12\n\ts_mul_i32 s22, s33, s44\n\t.loc\t1 858 12                        ; 7.ttgir:858:12\n\ts_addc_u32 s25, s40, s21\n\t.loc\t1 860 12                        ; 7.ttgir:860:12\n\ts_ashr_i32 s23, s22, 31\n\ts_lshl_b64 s[22:23], s[22:23], 1\n\ts_add_u32 s68, s36, s22\n\ts_addc_u32 s26, s66, s23\n\t.loc\t1 864 12                        ; 7.ttgir:864:12\n\ts_lshl_b32 s24, s1, 5\n\t.loc\t1 836 12                        ; 7.ttgir:836:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]\n\t.loc\t1 822 12                        ; 7.ttgir:822:12\n\tv_cvt_f16_f32_e32 v76, v119\n\tv_cvt_f16_f32_e32 v77, v120\n\t.loc\t1 798 12                        ; 7.ttgir:798:12\n\tds_read_b64 v[70:71], v171 offset:8192\n\t.loc\t1 800 12                        ; 7.ttgir:800:12\n\tds_read_b64 v[72:73], v173 offset:8192\n\t.loc\t1 802 12                        ; 7.ttgir:802:12\n\tds_read_b64 v[74:75], v174 offset:8192\n\t.loc\t1 865 12                        ; 7.ttgir:865:12\n\ts_or_b32 s27, s24, 31\n\t.loc\t1 871 12                        ; 7.ttgir:871:12\n\ts_cmp_lg_u32 s67, 0\n\t.loc\t1 837 12                        ; 7.ttgir:837:12\n\tv_pack_b32_f16 v77, v76, v77\n\tv_pack_b32_f16 v76, v67, v114\n\t.loc\t1 804 12                        ; 7.ttgir:804:12\n\tv_xor_b32_e32 v67, v175, v127\n\tv_lshl_add_u32 v177, v67, 1, 0\n\t.loc\t1 806 12                        ; 7.ttgir:806:12\n\tv_xor_b32_e32 v67, v178, v127\n\t.loc\t1 837 12                        ; 7.ttgir:837:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]\n\t.loc\t1 806 12                        ; 7.ttgir:806:12\n\tv_xor_b32_e32 v68, 0x408, v67\n\tv_lshl_add_u32 v175, v68, 1, 0\n\t.loc\t1 808 12                        ; 7.ttgir:808:12\n\tv_xor_b32_e32 v68, 0x810, v67\n\tv_lshl_add_u32 v179, v68, 1, 0\n\t.loc\t1 804 12                        ; 7.ttgir:804:12\n\tds_read_b64 v[68:69], v177 offset:8192\n\t.loc\t1 810 12                        ; 7.ttgir:810:12\n\tv_xor_b32_e32 v67, 0xc18, v67\n\tv_lshl_add_u32 v181, v67, 1, 0\n\t.loc\t1 838 12                        ; 7.ttgir:838:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]\n\t.loc\t1 823 12                        ; 7.ttgir:823:12\n\tv_cvt_f16_f32_e32 v67, v121\n\tv_cvt_f16_f32_e32 v114, v124\n\t.loc\t1 871 12                        ; 7.ttgir:871:12\n\ts_cselect_b64 s[48:49], -1, 0\n\t.loc\t1 891 12                        ; 7.ttgir:891:12\n\ts_and_b32 s25, s25, 0xffff\n\ts_or_b32 s65, s25, s53\n\ts_mov_b32 s67, 0x27000\n\ts_mov_b32 s66, 0x7ffffffe\n\t.loc\t1 839 12                        ; 7.ttgir:839:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]\n\t.loc\t1 896 12                        ; 7.ttgir:896:12\n\ts_and_b32 s25, s26, 0xffff\n\ts_or_b32 s69, s25, s55\n\ts_mov_b32 s70, s66\n\ts_mov_b32 s71, s67\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_lshl_add_u32 v193, v142, 1, 0\n\tv_lshl_add_u32 v194, v143, 1, 0\n\tv_lshl_add_u32 v195, v144, 1, 0\n\t.loc\t1 840 12                        ; 7.ttgir:840:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]\n\t.loc\t1 823 12                        ; 7.ttgir:823:12\n\tv_cvt_f16_f32_e32 v76, v125\n\tv_cvt_f16_f32_e32 v77, v126\n\t.loc\t1 806 12                        ; 7.ttgir:806:12\n\tds_read_b64 v[70:71], v175 offset:8192\n\t.loc\t1 808 12                        ; 7.ttgir:808:12\n\tds_read_b64 v[72:73], v179 offset:8192\n\t.loc\t1 810 12                        ; 7.ttgir:810:12\n\tds_read_b64 v[74:75], v181 offset:8192\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_lshl_add_u32 v196, v145, 1, 0\n\tv_lshl_add_u32 v197, v146, 1, 0\n\t.loc\t1 841 12                        ; 7.ttgir:841:12\n\tv_pack_b32_f16 v77, v76, v77\n\tv_pack_b32_f16 v76, v67, v114\n\t.loc\t1 812 12                        ; 7.ttgir:812:12\n\tv_xor_b32_e32 v67, v166, v127\n\tv_lshl_add_u32 v178, v67, 1, 0\n\t.loc\t1 814 12                        ; 7.ttgir:814:12\n\tv_xor_b32_e32 v67, v176, v127\n\t.loc\t1 841 12                        ; 7.ttgir:841:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]\n\t.loc\t1 814 12                        ; 7.ttgir:814:12\n\tv_xor_b32_e32 v68, 0x408, v67\n\tv_lshl_add_u32 v176, v68, 1, 0\n\t.loc\t1 816 12                        ; 7.ttgir:816:12\n\tv_xor_b32_e32 v68, 0x810, v67\n\tv_lshl_add_u32 v180, v68, 1, 0\n\t.loc\t1 812 12                        ; 7.ttgir:812:12\n\tds_read_b64 v[68:69], v178 offset:8192\n\t.loc\t1 818 12                        ; 7.ttgir:818:12\n\tv_xor_b32_e32 v67, 0xc18, v67\n\tv_lshl_add_u32 v182, v67, 1, 0\n\t.loc\t1 843 12                        ; 7.ttgir:843:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]\n\t.loc\t1 824 12                        ; 7.ttgir:824:12\n\tv_cvt_f16_f32_e32 v67, v78\n\tv_cvt_f16_f32_e32 v78, v79\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\tv_lshl_add_u32 v198, v147, 1, 0\n\tv_lshl_add_u32 v199, v149, 1, 0\n\tv_lshl_add_u32 v200, v150, 1, 0\n\t.loc\t1 845 12                        ; 7.ttgir:845:12\n\tv_pack_b32_f16 v118, v67, v78\n\t.loc\t1 878 12                        ; 7.ttgir:878:12\n\tv_or_b32_e32 v67, s33, v158\n\t.loc\t1 842 12                        ; 7.ttgir:842:12\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]\n\t.loc\t1 891 12                        ; 7.ttgir:891:12\n\tv_bfrev_b32_e32 v78, 1\n\t.loc\t1 885 12                        ; 7.ttgir:885:12\n\tv_cmp_gt_i32_e32 vcc, s19, v67\n\t.loc\t1 908 12                        ; 7.ttgir:908:12\n\ts_cmp_gt_i32 s27, 63\n\t.loc\t1 891 12                        ; 7.ttgir:891:12\n\ts_nop 0\n\tv_cndmask_b32_e32 v67, v78, v160, vcc\n\t.loc\t1 844 12                        ; 7.ttgir:844:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]\n\t.loc\t1 824 12                        ; 7.ttgir:824:12\n\tv_cvt_f16_f32_e32 v74, v80\n\tv_cvt_f16_f32_e32 v75, v81\n\t.loc\t1 814 12                        ; 7.ttgir:814:12\n\tds_read_b64 v[70:71], v176 offset:8192\n\t.loc\t1 816 12                        ; 7.ttgir:816:12\n\tds_read_b64 v[72:73], v180 offset:8192\n\t.loc\t1 818 12                        ; 7.ttgir:818:12\n\tds_read_b64 v[80:81], v182 offset:8192\n\t.loc\t1 879 12                        ; 7.ttgir:879:12\n\tv_or_b32_e32 v76, s33, v162\n\tv_or_b32_e32 v77, s33, v189\n\t.loc\t1 845 12                        ; 7.ttgir:845:12\n\tv_pack_b32_f16 v119, v74, v75\n\t.loc\t1 853 5                         ; 7.ttgir:853:5\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\t.loc\t1 845 12                        ; 7.ttgir:845:12\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[118:119], v[50:65]\n\t.loc\t1 878 12                        ; 7.ttgir:878:12\n\tv_or_b32_e32 v68, s33, v159\n\t.loc\t1 885 12                        ; 7.ttgir:885:12\n\tv_cmp_gt_i32_e32 vcc, s19, v68\n\t.loc\t1 847 12                        ; 7.ttgir:847:12\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[118:119], v[18:33]\n\t.loc\t1 892 12                        ; 7.ttgir:892:12\n\ts_nop 0\n\tv_cndmask_b32_e32 v72, v78, v161, vcc\n\t.loc\t1 894 12                        ; 7.ttgir:894:12\n\tv_cmp_gt_i32_e32 vcc, s19, v76\n\t.loc\t1 846 12                        ; 7.ttgir:846:12\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[118:119], v[34:49]\n\t.loc\t1 891 12                        ; 7.ttgir:891:12\n\tbuffer_load_dwordx4 v[68:71], v67, s[64:67], 0 offen\n\t.loc\t1 892 12                        ; 7.ttgir:892:12\n\ts_nop 0\n\tbuffer_load_dwordx4 v[72:75], v72, s[64:67], 0 offen\n\t.loc\t1 896 12                        ; 7.ttgir:896:12\n\tv_cndmask_b32_e32 v67, v78, v163, vcc\n\t.loc\t1 894 12                        ; 7.ttgir:894:12\n\tv_cmp_gt_i32_e32 vcc, s19, v77\n\t.loc\t1 896 12                        ; 7.ttgir:896:12\n\ts_nop 1\n\tv_cndmask_b32_e32 v114, v78, v164, vcc\n\tbuffer_load_dwordx4 v[76:79], v67, s[68:71], 0 offen\n\ts_nop 0\n\tbuffer_load_dwordx4 v[114:117], v114, s[68:71], 0 offen\n\t.loc\t1 848 12                        ; 7.ttgir:848:12\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[80:81], v[118:119], v[2:17]\n\t.loc\t1 899 5                         ; 7.ttgir:899:5\n\tv_mov_b32_e32 v67, 0x88\n\tv_mov_b32_e32 v80, 0x110\n\tv_cndmask_b32_e64 v67, v67, 0, s[14:15]\n\tv_cndmask_b32_e64 v80, v80, 0, s[12:13]\n\tv_or_b32_e32 v67, v67, v80\n\tv_mov_b32_e32 v80, 0x220\n\tv_cndmask_b32_e64 v80, v80, 0, s[2:3]\n\tv_or3_b32 v67, v67, v80, v148\n\tv_xor_b32_e32 v67, v67, v138\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\ts_mov_b32 s2, 0x5040100\n\t.loc\t1 899 5                         ; 7.ttgir:899:5\n\tv_lshl_add_u32 v192, v67, 1, 0\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\ts_mov_b32 s3, 0x7060302\n\t.loc\t1 899 5                         ; 7.ttgir:899:5\n\ts_waitcnt vmcnt(3)\n\tds_write_b128 v192, v[68:71]\n\t.loc\t1 901 5                         ; 7.ttgir:901:5\n\ts_waitcnt vmcnt(2)\n\tds_write_b128 v192, v[72:75] offset:4096\n\tv_or_b32_e32 v68, 16, v141\n\t.loc\t1 904 5                         ; 7.ttgir:904:5\n\ts_waitcnt vmcnt(0)\n\tv_perm_b32 v67, v114, v76, s2\n\tds_write_b32 v193, v67 offset:8192\n\tv_perm_b32 v67, v114, v76, s3\n\tds_write_b32 v194, v67 offset:8192\n\tv_perm_b32 v67, v115, v77, s2\n\tds_write_b32 v195, v67 offset:8192\n\tv_perm_b32 v67, v115, v77, s3\n\tds_write_b32 v196, v67 offset:8192\n\tv_perm_b32 v67, v116, v78, s2\n\tds_write_b32 v197, v67 offset:8192\n\tv_perm_b32 v67, v116, v78, s3\n\tds_write_b32 v198, v67 offset:8192\n\tv_perm_b32 v67, v117, v79, s2\n\tds_write_b32 v199, v67 offset:8192\n\tv_perm_b32 v67, v117, v79, s3\n\tds_write_b32 v200, v67 offset:8192\n\tv_xor_b32_e32 v67, v141, v139\n\tv_or_b32_e32 v191, v67, v140\n\t.loc\t1 909 5                         ; 7.ttgir:909:5\n\ts_cbranch_scc1 .LBB0_10\n; %bb.9:                                ; %._crit_edge.._crit_edge669_crit_edge\n\t.loc\t1 1204 12                       ; 7.ttgir:1204:12\n\tv_xor_b32_e32 v69, v68, v139\n\t.loc\t1 861 12                        ; 7.ttgir:861:12\n\ts_add_i32 s12, s33, 32\n\t.loc\t1 1202 12                       ; 7.ttgir:1202:12\n\tv_or_b32_e32 v67, v67, v140\n\t.loc\t1 1204 12                       ; 7.ttgir:1204:12\n\tv_or_b32_e32 v190, v69, v140\n\ts_mov_b64 s[2:3], 0\n\ts_branch .LBB0_11\n.LBB0_10:\n\t.loc\t1 0 12 is_stmt 0                ; 7.ttgir:0:12\n\ts_mov_b64 s[2:3], -1\n                                        ; implicit-def: $sgpr12\n                                        ; implicit-def: $vgpr67\n                                        ; implicit-def: $vgpr190\n.LBB0_11:                               ; %Flow391\n\tv_or_b32_e32 v166, s52, v165\n\tv_fmac_f32_e32 v122, v201, v66\n\ts_lshl_b32 s92, s72, 5\n\t.loc\t1 909 5 is_stmt 1               ; 7.ttgir:909:5\n\ts_andn2_b64 vcc, exec, s[2:3]\n\t.loc\t1 0 0 is_stmt 0                 ; 7.ttgir:0\n\ts_sub_i32 s44, s16, s19\n\t.loc\t1 909 5                         ; 7.ttgir:909:5\n\ts_cbranch_vccnz .LBB0_15\n; %bb.12:                               ; %.lr.ph668\n\t.loc\t1 866 12 is_stmt 1              ; 7.ttgir:866:12\n\ts_and_b32 s1, s1, 0x7ffffff\n\ts_ashr_i32 s39, s38, 31\n\t.loc\t1 909 5                         ; 7.ttgir:909:5\n\ts_add_i32 s93, s0, 1\n\ts_add_u32 s0, s62, s42\n\ts_addc_u32 s2, s63, s43\n\ts_add_u32 s0, s0, s60\n\ts_addc_u32 s2, s2, s61\n\ts_add_u32 s0, s0, s22\n\ts_addc_u32 s2, s2, s23\n\ts_lshl_b64 s[50:51], s[38:39], 1\n\ts_add_u32 s0, s0, s50\n\ts_addc_u32 s2, s2, s51\n\ts_add_u32 s94, s6, s0\n\ts_addc_u32 s95, s7, s2\n\ts_sub_i32 s96, 0, s24\n\ts_add_i32 s0, s44, s92\n\tv_add_u32_e32 v203, s0, v1\n\ts_add_u32 s0, s58, s56\n\ts_addc_u32 s2, s59, s57\n\ts_add_u32 s0, s0, s34\n\ts_addc_u32 s2, s2, s35\n\ts_add_u32 s0, s0, s20\n                                        ; implicit-def: $vgpr227 : SGPR spill to VGPR lane\n\ts_addc_u32 s2, s2, s21\n\ts_lshl_b64 s[56:57], s[30:31], 1\n\tv_writelane_b32 v227, s82, 0\n\tv_xor_b32_e32 v66, v68, v139\n\ts_add_u32 s0, s0, s56\n\tv_writelane_b32 v227, s83, 1\n\tv_or_b32_e32 v190, v66, v140\n\ts_addc_u32 s2, s2, s57\n\tv_sub_u32_e64 v66, s1, 2 clamp\n\tv_writelane_b32 v227, s10, 2\n\ts_add_u32 s97, s4, s0\n\tv_readfirstlane_b32 s0, v66\n\tv_writelane_b32 v227, s11, 3\n\tv_lshl_add_u32 v201, v191, 1, 0\n\tv_lshl_add_u32 v202, v190, 1, 0\n\tv_add_u32_e32 v204, s92, v1\n\ts_addc_u32 s98, s5, s2\n\ts_add_i32 s99, s0, 1\n\ts_movk_i32 s0, 0xffe0\n\ts_mov_b32 s43, 0x27000\n\ts_mov_b32 s42, 0x7ffffffe\n\tv_bfrev_b32_e32 v205, 1\n\ts_xor_b64 s[58:59], s[48:49], -1\n\tv_mov_b32_e32 v206, 0xff800000\n\ts_mov_b32 s1, 0x5040100\n\ts_mov_b32 s10, 0x7060302\n.LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n\t.loc\t1 0 5 is_stmt 0                 ; 7.ttgir:0:5\n\tv_mov_b32_e32 v207, v122\n\tv_mov_b32_e32 v210, v123\n\t.loc\t1 911 5 is_stmt 1               ; 7.ttgir:911:5\n\ts_setprio 0\n\t.loc\t1 917 12                        ; 7.ttgir:917:12\n\ts_lshl_b32 s33, s93, 5\n\t.loc\t1 921 12                        ; 7.ttgir:921:12\n\tv_or_b32_e32 v66, s33, v158\n\tv_or_b32_e32 v67, s33, v159\n\t.loc\t1 928 12                        ; 7.ttgir:928:12\n\tv_cmp_gt_i32_e32 vcc, s19, v66\n\t.loc\t1 934 12                        ; 7.ttgir:934:12\n\ts_and_b32 s4, s98, 0xffff\n\t.loc\t1 928 12                        ; 7.ttgir:928:12\n\tv_cmp_gt_i32_e64 s[2:3], s19, v67\n\t.loc\t1 934 12                        ; 7.ttgir:934:12\n\ts_or_b32 s41, s4, s53\n\ts_mov_b32 s40, s97\n\tv_cndmask_b32_e32 v66, v205, v160, vcc\n\tbuffer_load_dwordx4 v[114:117], v66, s[40:43], 0 offen\n\t.loc\t1 935 12                        ; 7.ttgir:935:12\n\tv_cndmask_b32_e64 v66, v205, v161, s[2:3]\n\tbuffer_load_dwordx4 v[118:121], v66, s[40:43], 0 offen\n\t.loc\t1 954 12                        ; 7.ttgir:954:12\n\ts_cmp_lg_u32 s96, s0\n\t.loc\t1 959 12                        ; 7.ttgir:959:12\n\tv_add_u32_e32 v66, s96, v204\n\t.loc\t1 954 12                        ; 7.ttgir:954:12\n\ts_cselect_b64 s[60:61], -1, 0\n\t.loc\t1 959 12                        ; 7.ttgir:959:12\n\tv_add_u32_e32 v67, 1, v66\n\tv_add_u32_e32 v68, 2, v66\n\t.loc\t1 962 12                        ; 7.ttgir:962:12\n\tv_add_u32_e32 v69, 3, v66\n\tv_add_u32_e32 v70, 8, v66\n\tv_add_u32_e32 v71, 9, v66\n\tv_add_u32_e32 v72, 10, v66\n\t.loc\t1 965 12                        ; 7.ttgir:965:12\n\tv_add_u32_e32 v73, 11, v66\n\tv_add_u32_e32 v74, 16, v66\n\tv_add_u32_e32 v75, 17, v66\n\tv_add_u32_e32 v76, 18, v66\n\t.loc\t1 968 12                        ; 7.ttgir:968:12\n\tv_add_u32_e32 v77, 19, v66\n\tv_add_u32_e32 v78, 24, v66\n\tv_add_u32_e32 v79, 25, v66\n\tv_add_u32_e32 v80, 26, v66\n\t.loc\t1 970 12                        ; 7.ttgir:970:12\n\tv_add_u32_e32 v81, 27, v66\n\tv_cmp_gt_i32_e32 vcc, s19, v66\n\t.loc\t1 974 12                        ; 7.ttgir:974:12\n\tv_add_u32_e32 v66, s96, v203\n\t.loc\t1 970 12                        ; 7.ttgir:970:12\n\tv_cmp_gt_i32_e64 s[2:3], s19, v67\n\tv_cmp_gt_i32_e64 s[4:5], s19, v68\n\tv_cmp_gt_i32_e64 s[6:7], s19, v69\n\tv_cmp_gt_i32_e64 s[12:13], s19, v70\n\tv_cmp_gt_i32_e64 s[14:15], s19, v71\n\tv_cmp_gt_i32_e64 s[20:21], s19, v72\n\tv_cmp_gt_i32_e64 s[22:23], s19, v73\n\tv_cmp_gt_i32_e64 s[24:25], s19, v74\n\tv_cmp_gt_i32_e64 s[26:27], s19, v75\n\tv_cmp_gt_i32_e64 s[28:29], s19, v76\n\tv_cmp_gt_i32_e64 s[30:31], s19, v77\n\tv_cmp_gt_i32_e64 s[34:35], s19, v78\n\tv_cmp_gt_i32_e64 s[36:37], s19, v79\n\tv_cmp_gt_i32_e64 s[38:39], s19, v80\n\tv_cmp_gt_i32_e64 s[40:41], s19, v81\n\t.loc\t1 972 12                        ; 7.ttgir:972:12\n\ts_or_b64 s[90:91], s[58:59], s[60:61]\n\t.loc\t1 974 12                        ; 7.ttgir:974:12\n\tv_add_u32_e32 v67, 1, v66\n\tv_add_u32_e32 v68, 2, v66\n\tv_add_u32_e32 v69, 3, v66\n\tv_add_u32_e32 v70, 8, v66\n\tv_add_u32_e32 v71, 9, v66\n\tv_add_u32_e32 v72, 10, v66\n\tv_add_u32_e32 v73, 11, v66\n\tv_add_u32_e32 v74, 16, v66\n\tv_add_u32_e32 v75, 17, v66\n\tv_add_u32_e32 v76, 18, v66\n\tv_add_u32_e32 v77, 19, v66\n\tv_add_u32_e32 v78, 24, v66\n\tv_add_u32_e32 v79, 25, v66\n\tv_add_u32_e32 v80, 26, v66\n\t.loc\t1 985 12                        ; 7.ttgir:985:12\n\tv_add_u32_e32 v81, 27, v66\n\t.loc\t1 936 5                         ; 7.ttgir:936:5\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\t.loc\t1 938 12                        ; 7.ttgir:938:12\n\tds_read_b128 v[150:153], v201\n\t.loc\t1 940 12                        ; 7.ttgir:940:12\n\tds_read_b128 v[146:149], v202\n\t.loc\t1 942 12                        ; 7.ttgir:942:12\n\tds_read_b128 v[142:145], v183\n\t.loc\t1 944 12                        ; 7.ttgir:944:12\n\tds_read_b128 v[138:141], v184\n\t.loc\t1 946 12                        ; 7.ttgir:946:12\n\tds_read_b128 v[134:137], v185\n\t.loc\t1 948 12                        ; 7.ttgir:948:12\n\tds_read_b128 v[130:133], v186\n\t.loc\t1 950 12                        ; 7.ttgir:950:12\n\tds_read_b128 v[126:129], v187\n\t.loc\t1 952 12                        ; 7.ttgir:952:12\n\tds_read_b128 v[122:125], v188\n\t.loc\t1 972 12                        ; 7.ttgir:972:12\n\ts_or_b64 s[60:61], s[90:91], vcc\n\ts_or_b64 s[62:63], s[90:91], s[2:3]\n\ts_or_b64 s[64:65], s[90:91], s[4:5]\n\ts_or_b64 s[66:67], s[90:91], s[6:7]\n\ts_or_b64 s[68:69], s[90:91], s[12:13]\n\ts_or_b64 s[70:71], s[90:91], s[14:15]\n\ts_or_b64 s[72:73], s[90:91], s[20:21]\n\ts_or_b64 s[74:75], s[90:91], s[22:23]\n\ts_or_b64 s[76:77], s[90:91], s[24:25]\n\ts_or_b64 s[78:79], s[90:91], s[26:27]\n\ts_or_b64 s[80:81], s[90:91], s[28:29]\n\ts_or_b64 s[82:83], s[90:91], s[30:31]\n\ts_or_b64 s[84:85], s[90:91], s[34:35]\n\ts_or_b64 s[86:87], s[90:91], s[36:37]\n\ts_or_b64 s[88:89], s[90:91], s[38:39]\n\ts_or_b64 s[90:91], s[90:91], s[40:41]\n\t.loc\t1 985 12                        ; 7.ttgir:985:12\n\tv_cmp_ge_i32_e32 vcc, v166, v66\n\tv_cmp_ge_i32_e64 s[2:3], v166, v67\n\tv_cmp_ge_i32_e64 s[4:5], v166, v68\n\tv_cmp_ge_i32_e64 s[6:7], v166, v69\n\tv_cmp_ge_i32_e64 s[12:13], v166, v70\n\tv_cmp_ge_i32_e64 s[14:15], v166, v71\n\tv_cmp_ge_i32_e64 s[20:21], v166, v72\n\tv_cmp_ge_i32_e64 s[22:23], v166, v73\n\tv_cmp_ge_i32_e64 s[24:25], v166, v74\n\tv_cmp_ge_i32_e64 s[26:27], v166, v75\n\tv_cmp_ge_i32_e64 s[28:29], v166, v76\n\tv_cmp_ge_i32_e64 s[30:31], v166, v77\n\tv_cmp_ge_i32_e64 s[34:35], v166, v78\n\tv_cmp_ge_i32_e64 s[36:37], v166, v79\n\tv_cmp_ge_i32_e64 s[38:39], v166, v80\n\tv_cmp_ge_i32_e64 s[40:41], v166, v81\n\t.loc\t1 1008 12                       ; 7.ttgir:1008:12\n\ts_waitcnt lgkmcnt(7)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[150:151], v[110:111], 0\n\t.loc\t1 988 12                        ; 7.ttgir:988:12\n\ts_and_b64 s[60:61], vcc, s[60:61]\n\ts_and_b64 s[2:3], s[2:3], s[62:63]\n\ts_and_b64 s[4:5], s[4:5], s[64:65]\n\ts_and_b64 s[6:7], s[6:7], s[66:67]\n\t.loc\t1 999 12                        ; 7.ttgir:999:12\n\tv_cndmask_b32_e64 v211, v206, 0, s[60:61]\n\tv_cndmask_b32_e64 v212, v206, 0, s[2:3]\n\t.loc\t1 991 12                        ; 7.ttgir:991:12\n\ts_and_b64 s[12:13], s[12:13], s[68:69]\n\t.loc\t1 1008 12                       ; 7.ttgir:1008:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[152:153], v[112:113], v[66:81]\n\t.loc\t1 991 12                        ; 7.ttgir:991:12\n\ts_and_b64 s[14:15], s[14:15], s[70:71]\n\t.loc\t1 999 12                        ; 7.ttgir:999:12\n\tv_cndmask_b32_e64 v213, v206, 0, s[4:5]\n\tv_cndmask_b32_e64 v214, v206, 0, s[6:7]\n\t.loc\t1 991 12                        ; 7.ttgir:991:12\n\ts_and_b64 s[20:21], s[20:21], s[72:73]\n\ts_and_b64 s[22:23], s[22:23], s[74:75]\n\t.loc\t1 999 12                        ; 7.ttgir:999:12\n\tv_cndmask_b32_e64 v215, v206, 0, s[12:13]\n\tv_cndmask_b32_e64 v216, v206, 0, s[14:15]\n\t.loc\t1 1009 12                       ; 7.ttgir:1009:12\n\ts_waitcnt lgkmcnt(6)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[106:107], v[66:81]\n\t.loc\t1 994 12                        ; 7.ttgir:994:12\n\ts_and_b64 s[24:25], s[24:25], s[76:77]\n\ts_and_b64 s[26:27], s[26:27], s[78:79]\n\t.loc\t1 999 12                        ; 7.ttgir:999:12\n\tv_cndmask_b32_e64 v217, v206, 0, s[20:21]\n\tv_cndmask_b32_e64 v218, v206, 0, s[22:23]\n\t.loc\t1 994 12                        ; 7.ttgir:994:12\n\ts_and_b64 s[28:29], s[28:29], s[80:81]\n\ts_and_b64 s[30:31], s[30:31], s[82:83]\n\t.loc\t1 999 12                        ; 7.ttgir:999:12\n\tv_cndmask_b32_e64 v219, v206, 0, s[24:25]\n\t.loc\t1 1009 12                       ; 7.ttgir:1009:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[148:149], v[108:109], v[66:81]\n\t.loc\t1 999 12                        ; 7.ttgir:999:12\n\tv_cndmask_b32_e64 v220, v206, 0, s[26:27]\n\t.loc\t1 997 12                        ; 7.ttgir:997:12\n\ts_and_b64 s[34:35], s[34:35], s[84:85]\n\ts_and_b64 s[36:37], s[36:37], s[86:87]\n\t.loc\t1 999 12                        ; 7.ttgir:999:12\n\tv_cndmask_b32_e64 v221, v206, 0, s[28:29]\n\tv_cndmask_b32_e64 v222, v206, 0, s[30:31]\n\t.loc\t1 997 12                        ; 7.ttgir:997:12\n\ts_and_b64 s[38:39], s[38:39], s[88:89]\n\ts_and_b64 s[40:41], s[40:41], s[90:91]\n\t.loc\t1 1010 12                       ; 7.ttgir:1010:12\n\ts_waitcnt lgkmcnt(5)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[102:103], v[66:81]\n\t.loc\t1 999 12                        ; 7.ttgir:999:12\n\tv_cndmask_b32_e64 v223, v206, 0, s[34:35]\n\tv_cndmask_b32_e64 v224, v206, 0, s[36:37]\n\tv_cndmask_b32_e64 v225, v206, 0, s[38:39]\n\tv_cndmask_b32_e64 v226, v206, 0, s[40:41]\n\t.loc\t1 922 12                        ; 7.ttgir:922:12\n\tv_or_b32_e32 v208, s33, v162\n\tv_or_b32_e32 v209, s33, v189\n\t.loc\t1 1117 12                       ; 7.ttgir:1117:12\n\tv_cmp_gt_i32_e32 vcc, s19, v208\n\t.loc\t1 1010 12                       ; 7.ttgir:1010:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[104:105], v[66:81]\n\t.loc\t1 1117 12                       ; 7.ttgir:1117:12\n\tv_cmp_gt_i32_e64 s[2:3], s19, v209\n\t.loc\t1 1119 12                       ; 7.ttgir:1119:12\n\ts_and_b32 s4, s95, 0xffff\n\ts_or_b32 s41, s4, s55\n\ts_mov_b32 s40, s94\n\t.loc\t1 908 12                        ; 7.ttgir:908:12\n\ts_add_u32 s94, s94, s50\n\ts_addc_u32 s95, s95, s51\n\ts_sub_i32 s0, s0, 32\n\t.loc\t1 1011 12                       ; 7.ttgir:1011:12\n\ts_waitcnt lgkmcnt(4)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[98:99], v[66:81]\n\t.loc\t1 908 12                        ; 7.ttgir:908:12\n\ts_add_u32 s97, s97, s56\n\ts_addc_u32 s98, s98, s57\n\ts_add_i32 s93, s93, 1\n\ts_add_i32 s99, s99, -1\n\tv_add_u32_e32 v203, 32, v203\n\tv_add_u32_e32 v204, 32, v204\n\ts_cmp_lg_u32 s99, 0\n\t.loc\t1 1011 12                       ; 7.ttgir:1011:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[100:101], v[66:81]\n\t.loc\t1 1012 12                       ; 7.ttgir:1012:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[94:95], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[96:97], v[66:81]\n\t.loc\t1 1013 12                       ; 7.ttgir:1013:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[90:91], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[92:93], v[66:81]\n\t.loc\t1 1014 12                       ; 7.ttgir:1014:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[86:87], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[88:89], v[66:81]\n\t.loc\t1 1015 12                       ; 7.ttgir:1015:12\n\ts_waitcnt lgkmcnt(0)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[122:123], v[82:83], v[66:81]\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[84:85], v[66:81]\n\t.loc\t1 1029 12                       ; 7.ttgir:1029:12\n\ts_nop 7\n\ts_nop 2\n\tv_fmac_f32_e32 v211, 0x3e0293ee, v66\n\tv_fmac_f32_e32 v212, 0x3e0293ee, v67\n\tv_fmac_f32_e32 v213, 0x3e0293ee, v68\n\tv_fmac_f32_e32 v214, 0x3e0293ee, v69\n\t.loc\t1 1039 15                       ; 7.ttgir:1039:15\n\tv_max_f32_e32 v66, v211, v212\n\t.loc\t1 1031 12                       ; 7.ttgir:1031:12\n\tv_fmac_f32_e32 v215, 0x3e0293ee, v70\n\tv_fmac_f32_e32 v216, 0x3e0293ee, v71\n\t.loc\t1 1039 15                       ; 7.ttgir:1039:15\n\tv_max3_f32 v66, v66, v213, v214\n\t.loc\t1 1031 12                       ; 7.ttgir:1031:12\n\tv_fmac_f32_e32 v217, 0x3e0293ee, v72\n\tv_fmac_f32_e32 v218, 0x3e0293ee, v73\n\t.loc\t1 1039 15                       ; 7.ttgir:1039:15\n\tv_max3_f32 v66, v66, v215, v216\n\t.loc\t1 1033 12                       ; 7.ttgir:1033:12\n\tv_fmac_f32_e32 v219, 0x3e0293ee, v74\n\tv_fmac_f32_e32 v220, 0x3e0293ee, v75\n\t.loc\t1 1039 15                       ; 7.ttgir:1039:15\n\tv_max3_f32 v66, v66, v217, v218\n\t.loc\t1 1033 12                       ; 7.ttgir:1033:12\n\tv_fmac_f32_e32 v221, 0x3e0293ee, v76\n\tv_fmac_f32_e32 v222, 0x3e0293ee, v77\n\t.loc\t1 1039 15                       ; 7.ttgir:1039:15\n\tv_max3_f32 v66, v66, v219, v220\n\t.loc\t1 1035 12                       ; 7.ttgir:1035:12\n\tv_fmac_f32_e32 v223, 0x3e0293ee, v78\n\tv_fmac_f32_e32 v224, 0x3e0293ee, v79\n\t.loc\t1 1039 15                       ; 7.ttgir:1039:15\n\tv_max3_f32 v66, v66, v221, v222\n\t.loc\t1 1035 12                       ; 7.ttgir:1035:12\n\tv_fmac_f32_e32 v225, 0x3e0293ee, v80\n\tv_fmac_f32_e32 v226, 0x3e0293ee, v81\n\t.loc\t1 1039 15                       ; 7.ttgir:1039:15\n\tv_max3_f32 v66, v66, v223, v224\n\tv_max3_f32 v66, v66, v225, v226\n\t.loc\t1 1037 12                       ; 7.ttgir:1037:12\n\tds_bpermute_b32 v67, v157, v66\n\t.loc\t1 1042 12                       ; 7.ttgir:1042:12\n\ts_waitcnt lgkmcnt(0)\n\tv_max3_f32 v123, v210, v66, v67\n\t.loc\t1 1046 12                       ; 7.ttgir:1046:12\n\tv_sub_f32_e32 v66, v211, v123\n\tv_sub_f32_e32 v67, v212, v123\n\tv_sub_f32_e32 v68, v213, v123\n\t.loc\t1 1053 12                       ; 7.ttgir:1053:12\n\tv_exp_f32_e32 v148, v66\n\tv_exp_f32_e32 v149, v67\n\t.loc\t1 1046 12                       ; 7.ttgir:1046:12\n\tv_sub_f32_e32 v69, v214, v123\n\t.loc\t1 1053 12                       ; 7.ttgir:1053:12\n\tv_exp_f32_e32 v150, v68\n\t.loc\t1 1048 12                       ; 7.ttgir:1048:12\n\tv_sub_f32_e32 v70, v215, v123\n\t.loc\t1 1053 12                       ; 7.ttgir:1053:12\n\tv_exp_f32_e32 v151, v69\n\t.loc\t1 1048 12                       ; 7.ttgir:1048:12\n\tv_sub_f32_e32 v71, v216, v123\n\t.loc\t1 1054 12                       ; 7.ttgir:1054:12\n\tv_exp_f32_e32 v152, v70\n\t.loc\t1 1048 12                       ; 7.ttgir:1048:12\n\tv_sub_f32_e32 v72, v217, v123\n\t.loc\t1 1054 12                       ; 7.ttgir:1054:12\n\tv_exp_f32_e32 v153, v71\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v148, v149\n\t.loc\t1 1048 12                       ; 7.ttgir:1048:12\n\tv_sub_f32_e32 v73, v218, v123\n\t.loc\t1 1054 12                       ; 7.ttgir:1054:12\n\tv_exp_f32_e32 v211, v72\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v150, v66\n\t.loc\t1 1050 12                       ; 7.ttgir:1050:12\n\tv_sub_f32_e32 v74, v219, v123\n\t.loc\t1 1054 12                       ; 7.ttgir:1054:12\n\tv_exp_f32_e32 v212, v73\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v151, v66\n\t.loc\t1 1050 12                       ; 7.ttgir:1050:12\n\tv_sub_f32_e32 v75, v220, v123\n\t.loc\t1 1055 12                       ; 7.ttgir:1055:12\n\tv_exp_f32_e32 v213, v74\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v152, v66\n\t.loc\t1 1050 12                       ; 7.ttgir:1050:12\n\tv_sub_f32_e32 v76, v221, v123\n\t.loc\t1 1055 12                       ; 7.ttgir:1055:12\n\tv_exp_f32_e32 v214, v75\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v153, v66\n\t.loc\t1 1050 12                       ; 7.ttgir:1050:12\n\tv_sub_f32_e32 v77, v222, v123\n\t.loc\t1 1055 12                       ; 7.ttgir:1055:12\n\tv_exp_f32_e32 v215, v76\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v211, v66\n\t.loc\t1 1052 12                       ; 7.ttgir:1052:12\n\tv_sub_f32_e32 v78, v223, v123\n\t.loc\t1 1055 12                       ; 7.ttgir:1055:12\n\tv_exp_f32_e32 v216, v77\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v212, v66\n\t.loc\t1 1052 12                       ; 7.ttgir:1052:12\n\tv_sub_f32_e32 v79, v224, v123\n\t.loc\t1 1056 12                       ; 7.ttgir:1056:12\n\tv_exp_f32_e32 v217, v78\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v213, v66\n\t.loc\t1 1052 12                       ; 7.ttgir:1052:12\n\tv_sub_f32_e32 v80, v225, v123\n\t.loc\t1 1056 12                       ; 7.ttgir:1056:12\n\tv_exp_f32_e32 v218, v79\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v214, v66\n\t.loc\t1 1052 12                       ; 7.ttgir:1052:12\n\tv_sub_f32_e32 v81, v226, v123\n\t.loc\t1 1056 12                       ; 7.ttgir:1056:12\n\tv_exp_f32_e32 v219, v80\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v215, v66\n\t.loc\t1 1056 12                       ; 7.ttgir:1056:12\n\tv_exp_f32_e32 v220, v81\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\tv_add_f32_e32 v66, v216, v66\n\tv_add_f32_e32 v66, v217, v66\n\tv_add_f32_e32 v66, v218, v66\n\tv_add_f32_e32 v66, v219, v66\n\tv_add_f32_e32 v66, v220, v66\n\t.loc\t1 1058 12                       ; 7.ttgir:1058:12\n\tds_bpermute_b32 v67, v157, v66\n\t.loc\t1 1119 12                       ; 7.ttgir:1119:12\n\tv_cndmask_b32_e64 v70, v205, v164, s[2:3]\n\tbuffer_load_dwordx4 v[70:73], v70, s[40:43], 0 offen\n\t.loc\t1 1154 12                       ; 7.ttgir:1154:12\n\tv_cvt_f16_f32_e32 v148, v148\n\t.loc\t1 1155 12                       ; 7.ttgir:1155:12\n\tv_cvt_f16_f32_e32 v208, v212\n\t.loc\t1 1060 15                       ; 7.ttgir:1060:15\n\ts_waitcnt lgkmcnt(0)\n\tv_add_f32_e32 v122, v66, v67\n\t.loc\t1 1063 12                       ; 7.ttgir:1063:12\n\tv_sub_f32_e32 v66, v210, v123\n\t.loc\t1 1064 12                       ; 7.ttgir:1064:12\n\tv_exp_f32_e32 v210, v66\n\t.loc\t1 1119 12                       ; 7.ttgir:1119:12\n\tv_cndmask_b32_e32 v66, v205, v163, vcc\n\tbuffer_load_dwordx4 v[66:69], v66, s[40:43], 0 offen\n\t.loc\t1 1121 12                       ; 7.ttgir:1121:12\n\tds_read_b64 v[74:75], v168 offset:8192\n\t.loc\t1 1123 12                       ; 7.ttgir:1123:12\n\tds_read_b64 v[76:77], v167 offset:8192\n\t.loc\t1 1125 12                       ; 7.ttgir:1125:12\n\tds_read_b64 v[78:79], v169 offset:8192\n\t.loc\t1 1127 12                       ; 7.ttgir:1127:12\n\tds_read_b64 v[80:81], v170 offset:8192\n\t.loc\t1 1129 12                       ; 7.ttgir:1129:12\n\tds_read_b64 v[124:125], v172 offset:8192\n\t.loc\t1 1131 12                       ; 7.ttgir:1131:12\n\tds_read_b64 v[126:127], v171 offset:8192\n\t.loc\t1 1133 12                       ; 7.ttgir:1133:12\n\tds_read_b64 v[128:129], v173 offset:8192\n\t.loc\t1 1135 12                       ; 7.ttgir:1135:12\n\tds_read_b64 v[130:131], v174 offset:8192\n\t.loc\t1 1137 12                       ; 7.ttgir:1137:12\n\tds_read_b64 v[132:133], v177 offset:8192\n\t.loc\t1 1139 12                       ; 7.ttgir:1139:12\n\tds_read_b64 v[134:135], v175 offset:8192\n\t.loc\t1 1141 12                       ; 7.ttgir:1141:12\n\tds_read_b64 v[136:137], v179 offset:8192\n\t.loc\t1 1143 12                       ; 7.ttgir:1143:12\n\tds_read_b64 v[138:139], v181 offset:8192\n\t.loc\t1 1145 12                       ; 7.ttgir:1145:12\n\tds_read_b64 v[140:141], v178 offset:8192\n\t.loc\t1 1147 12                       ; 7.ttgir:1147:12\n\tds_read_b64 v[142:143], v176 offset:8192\n\t.loc\t1 1149 12                       ; 7.ttgir:1149:12\n\tds_read_b64 v[144:145], v180 offset:8192\n\t.loc\t1 1151 12                       ; 7.ttgir:1151:12\n\tds_read_b64 v[146:147], v182 offset:8192\n\t.loc\t1 1153 12                       ; 7.ttgir:1153:12\n\tv_fmac_f32_e32 v122, v207, v210\n\t.loc\t1 1154 12                       ; 7.ttgir:1154:12\n\tv_cvt_f16_f32_e32 v207, v149\n\tv_cvt_f16_f32_e32 v149, v150\n\tv_cvt_f16_f32_e32 v150, v151\n\t.loc\t1 1069 12                       ; 7.ttgir:1069:12\n\tv_mul_f32_e32 v50, v50, v210\n\tv_mul_f32_e32 v51, v51, v210\n\tv_mul_f32_e32 v52, v52, v210\n\tv_mul_f32_e32 v53, v53, v210\n\t.loc\t1 1072 12                       ; 7.ttgir:1072:12\n\tv_mul_f32_e32 v54, v54, v210\n\tv_mul_f32_e32 v55, v55, v210\n\tv_mul_f32_e32 v56, v56, v210\n\tv_mul_f32_e32 v57, v57, v210\n\t.loc\t1 1075 12                       ; 7.ttgir:1075:12\n\tv_mul_f32_e32 v58, v58, v210\n\tv_mul_f32_e32 v59, v59, v210\n\tv_mul_f32_e32 v60, v60, v210\n\tv_mul_f32_e32 v61, v61, v210\n\t.loc\t1 1078 12                       ; 7.ttgir:1078:12\n\tv_mul_f32_e32 v62, v62, v210\n\tv_mul_f32_e32 v63, v63, v210\n\tv_mul_f32_e32 v64, v64, v210\n\tv_mul_f32_e32 v65, v65, v210\n\t.loc\t1 1081 12                       ; 7.ttgir:1081:12\n\tv_mul_f32_e32 v34, v34, v210\n\tv_mul_f32_e32 v35, v35, v210\n\tv_mul_f32_e32 v36, v36, v210\n\tv_mul_f32_e32 v37, v37, v210\n\t.loc\t1 1084 12                       ; 7.ttgir:1084:12\n\tv_mul_f32_e32 v38, v38, v210\n\tv_mul_f32_e32 v39, v39, v210\n\tv_mul_f32_e32 v40, v40, v210\n\tv_mul_f32_e32 v41, v41, v210\n\t.loc\t1 1087 12                       ; 7.ttgir:1087:12\n\tv_mul_f32_e32 v42, v42, v210\n\tv_mul_f32_e32 v43, v43, v210\n\tv_mul_f32_e32 v44, v44, v210\n\tv_mul_f32_e32 v45, v45, v210\n\t.loc\t1 1090 12                       ; 7.ttgir:1090:12\n\tv_mul_f32_e32 v46, v46, v210\n\tv_mul_f32_e32 v47, v47, v210\n\tv_mul_f32_e32 v48, v48, v210\n\tv_mul_f32_e32 v49, v49, v210\n\t.loc\t1 1093 12                       ; 7.ttgir:1093:12\n\tv_mul_f32_e32 v18, v18, v210\n\tv_mul_f32_e32 v19, v19, v210\n\tv_mul_f32_e32 v20, v20, v210\n\tv_mul_f32_e32 v21, v21, v210\n\t.loc\t1 1096 12                       ; 7.ttgir:1096:12\n\tv_mul_f32_e32 v22, v22, v210\n\tv_mul_f32_e32 v23, v23, v210\n\tv_mul_f32_e32 v24, v24, v210\n\tv_mul_f32_e32 v25, v25, v210\n\t.loc\t1 1099 12                       ; 7.ttgir:1099:12\n\tv_mul_f32_e32 v26, v26, v210\n\tv_mul_f32_e32 v27, v27, v210\n\tv_mul_f32_e32 v28, v28, v210\n\tv_mul_f32_e32 v29, v29, v210\n\t.loc\t1 1102 12                       ; 7.ttgir:1102:12\n\tv_mul_f32_e32 v30, v30, v210\n\tv_mul_f32_e32 v31, v31, v210\n\tv_mul_f32_e32 v32, v32, v210\n\tv_mul_f32_e32 v33, v33, v210\n\t.loc\t1 1105 12                       ; 7.ttgir:1105:12\n\tv_mul_f32_e32 v2, v2, v210\n\tv_mul_f32_e32 v3, v3, v210\n\tv_mul_f32_e32 v4, v4, v210\n\tv_mul_f32_e32 v5, v5, v210\n\t.loc\t1 1108 12                       ; 7.ttgir:1108:12\n\tv_mul_f32_e32 v6, v6, v210\n\tv_mul_f32_e32 v7, v7, v210\n\tv_mul_f32_e32 v8, v8, v210\n\tv_mul_f32_e32 v9, v9, v210\n\t.loc\t1 1111 12                       ; 7.ttgir:1111:12\n\tv_mul_f32_e32 v10, v10, v210\n\tv_mul_f32_e32 v11, v11, v210\n\tv_mul_f32_e32 v12, v12, v210\n\tv_mul_f32_e32 v13, v13, v210\n\t.loc\t1 1114 12                       ; 7.ttgir:1114:12\n\tv_mul_f32_e32 v14, v14, v210\n\tv_mul_f32_e32 v15, v15, v210\n\tv_mul_f32_e32 v16, v16, v210\n\tv_mul_f32_e32 v17, v17, v210\n\t.loc\t1 1166 12                       ; 7.ttgir:1166:12\n\tv_pack_b32_f16 v149, v149, v150\n\tv_pack_b32_f16 v148, v148, v207\n\t.loc\t1 1155 12                       ; 7.ttgir:1155:12\n\tv_cvt_f16_f32_e32 v151, v152\n\tv_cvt_f16_f32_e32 v152, v153\n\t.loc\t1 1166 12                       ; 7.ttgir:1166:12\n\ts_waitcnt lgkmcnt(14)\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[74:75], v[148:149], v[50:65]\n\t.loc\t1 1155 12                       ; 7.ttgir:1155:12\n\tv_cvt_f16_f32_e32 v153, v211\n\t.loc\t1 1156 12                       ; 7.ttgir:1156:12\n\tv_cvt_f16_f32_e32 v209, v213\n\t.loc\t1 1170 12                       ; 7.ttgir:1170:12\n\tv_pack_b32_f16 v74, v151, v152\n\t.loc\t1 1156 12                       ; 7.ttgir:1156:12\n\tv_cvt_f16_f32_e32 v210, v214\n\t.loc\t1 1170 12                       ; 7.ttgir:1170:12\n\tv_pack_b32_f16 v75, v153, v208\n\t.loc\t1 1156 12                       ; 7.ttgir:1156:12\n\tv_cvt_f16_f32_e32 v211, v215\n\tv_cvt_f16_f32_e32 v212, v216\n\t.loc\t1 1167 12                       ; 7.ttgir:1167:12\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[76:77], v[148:149], v[34:49]\n\t.loc\t1 1157 12                       ; 7.ttgir:1157:12\n\tv_cvt_f16_f32_e32 v213, v217\n\tv_cvt_f16_f32_e32 v214, v218\n\tv_cvt_f16_f32_e32 v215, v219\n\tv_cvt_f16_f32_e32 v216, v220\n\t.loc\t1 1187 5                        ; 7.ttgir:1187:5\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\t.loc\t1 1168 12                       ; 7.ttgir:1168:12\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[78:79], v[148:149], v[18:33]\n\t.loc\t1 1189 5                        ; 7.ttgir:1189:5\n\ts_waitcnt vmcnt(3)\n\tds_write_b128 v192, v[114:117]\n\t.loc\t1 1191 5                        ; 7.ttgir:1191:5\n\ts_waitcnt vmcnt(2)\n\tds_write_b128 v192, v[118:121] offset:4096\n\t.loc\t1 1169 12                       ; 7.ttgir:1169:12\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[80:81], v[148:149], v[2:17]\n\t.loc\t1 1170 12                       ; 7.ttgir:1170:12\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[124:125], v[74:75], v[50:65]\n\t.loc\t1 1171 12                       ; 7.ttgir:1171:12\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[126:127], v[74:75], v[34:49]\n\t.loc\t1 1172 12                       ; 7.ttgir:1172:12\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[128:129], v[74:75], v[18:33]\n\t.loc\t1 1173 12                       ; 7.ttgir:1173:12\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[130:131], v[74:75], v[2:17]\n\t.loc\t1 1174 12                       ; 7.ttgir:1174:12\n\tv_pack_b32_f16 v75, v211, v212\n\tv_pack_b32_f16 v74, v209, v210\n\ts_nop 1\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[132:133], v[74:75], v[50:65]\n\t.loc\t1 1175 12                       ; 7.ttgir:1175:12\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[134:135], v[74:75], v[34:49]\n\t.loc\t1 1176 12                       ; 7.ttgir:1176:12\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[136:137], v[74:75], v[18:33]\n\t.loc\t1 1177 12                       ; 7.ttgir:1177:12\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[138:139], v[74:75], v[2:17]\n\t.loc\t1 1178 12                       ; 7.ttgir:1178:12\n\tv_pack_b32_f16 v75, v215, v216\n\tv_pack_b32_f16 v74, v213, v214\n\ts_nop 1\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[140:141], v[74:75], v[50:65]\n\t.loc\t1 1179 12                       ; 7.ttgir:1179:12\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[142:143], v[74:75], v[34:49]\n\t.loc\t1 1180 12                       ; 7.ttgir:1180:12\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[144:145], v[74:75], v[18:33]\n\t.loc\t1 1181 12                       ; 7.ttgir:1181:12\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[146:147], v[74:75], v[2:17]\n\t.loc\t1 1194 5                        ; 7.ttgir:1194:5\n\ts_waitcnt vmcnt(0)\n\tv_perm_b32 v74, v70, v66, s1\n\tv_perm_b32 v66, v70, v66, s10\n\tds_write_b32 v193, v74 offset:8192\n\tds_write_b32 v194, v66 offset:8192\n\tv_perm_b32 v66, v71, v67, s1\n\tds_write_b32 v195, v66 offset:8192\n\tv_perm_b32 v66, v71, v67, s10\n\tds_write_b32 v196, v66 offset:8192\n\tv_perm_b32 v66, v72, v68, s1\n\tds_write_b32 v197, v66 offset:8192\n\tv_perm_b32 v66, v72, v68, s10\n\tds_write_b32 v198, v66 offset:8192\n\tv_perm_b32 v66, v73, v69, s1\n\tds_write_b32 v199, v66 offset:8192\n\tv_perm_b32 v66, v73, v69, s10\n\tds_write_b32 v200, v66 offset:8192\n\t.loc\t1 909 5                         ; 7.ttgir:909:5\n\ts_cbranch_scc1 .LBB0_13\n; %bb.14:                               ; %._crit_edge669.loopexit\n\t.loc\t1 0 5 is_stmt 0                 ; 7.ttgir:0:5\n\tv_readlane_b32 s10, v227, 2\n\tv_readlane_b32 s82, v227, 0\n\t.loc\t1 1217 12 is_stmt 1             ; 7.ttgir:1217:12\n\ts_add_i32 s12, s33, 32\n\tv_readlane_b32 s11, v227, 3\n\tv_readlane_b32 s83, v227, 1\n\ts_branch .LBB0_16\n.LBB0_15:\n\t.loc\t1 0 12 is_stmt 0                ; 7.ttgir:0:12\n\tv_mov_b32_e32 v191, v67\n.LBB0_16:                               ; %._crit_edge669\n\t.loc\t1 1202 12 is_stmt 1             ; 7.ttgir:1202:12\n\tv_lshl_add_u32 v66, v191, 1, 0\n\t.loc\t1 1204 12                       ; 7.ttgir:1204:12\n\tv_lshl_add_u32 v67, v190, 1, 0\n\t.loc\t1 1200 5                        ; 7.ttgir:1200:5\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\t.loc\t1 1202 12                       ; 7.ttgir:1202:12\n\tds_read_b128 v[124:127], v66\n\t.loc\t1 1204 12                       ; 7.ttgir:1204:12\n\tds_read_b128 v[128:131], v67\n\t.loc\t1 1206 12                       ; 7.ttgir:1206:12\n\tds_read_b128 v[132:135], v183\n\t.loc\t1 1208 12                       ; 7.ttgir:1208:12\n\tds_read_b128 v[136:139], v184\n\t.loc\t1 1210 12                       ; 7.ttgir:1210:12\n\tds_read_b128 v[140:143], v185\n\t.loc\t1 1212 12                       ; 7.ttgir:1212:12\n\tds_read_b128 v[144:147], v186\n\t.loc\t1 1214 12                       ; 7.ttgir:1214:12\n\tds_read_b128 v[118:121], v187\n\t.loc\t1 1216 12                       ; 7.ttgir:1216:12\n\tds_read_b128 v[114:117], v188\n\t.loc\t1 1272 12                       ; 7.ttgir:1272:12\n\ts_waitcnt lgkmcnt(7)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[124:125], v[110:111], 0\n\t.loc\t1 124 11                        ; 7.ttgir:124:11\n\ts_add_i32 s0, s52, 0x80\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v148, 1, v1\n\t.loc\t1 1218 12                       ; 7.ttgir:1218:12\n\ts_cmp_lg_u32 s12, s92\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v149, 2, v1\n\t.loc\t1 1218 12                       ; 7.ttgir:1218:12\n\ts_cselect_b64 s[42:43], -1, 0\n\t.loc\t1 1219 12                       ; 7.ttgir:1219:12\n\ts_xor_b64 s[48:49], s[48:49], -1\n\t.loc\t1 1223 12                       ; 7.ttgir:1223:12\n\tv_or_b32_e32 v183, s33, v148\n\t.loc\t1 1272 12                       ; 7.ttgir:1272:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[126:127], v[112:113], v[66:81]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\ts_add_i32 s1, s33, s44\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v150, 3, v1\n\t.loc\t1 1223 12                       ; 7.ttgir:1223:12\n\tv_or_b32_e32 v184, s33, v149\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[2:3], s19, v183\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[80:81], s[48:49], s[42:43]\n\t.loc\t1 1223 12                       ; 7.ttgir:1223:12\n\tv_or_b32_e32 v185, s33, v150\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[4:5], s19, v184\n\t.loc\t1 1273 12                       ; 7.ttgir:1273:12\n\ts_waitcnt lgkmcnt(6)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[128:129], v[106:107], v[66:81]\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[56:57], s[80:81], s[2:3]\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v151, 9, v1\n\t.loc\t1 1226 12                       ; 7.ttgir:1226:12\n\tv_or_b32_e32 v186, s33, v156\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[6:7], s19, v185\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[58:59], s[80:81], s[4:5]\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v152, 10, v1\n\t.loc\t1 1226 12                       ; 7.ttgir:1226:12\n\tv_or_b32_e32 v187, s33, v151\n\t.loc\t1 1273 12                       ; 7.ttgir:1273:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[130:131], v[108:109], v[66:81]\n\t.loc\t1 1232 12                       ; 7.ttgir:1232:12\n\tv_or_b32_e32 v190, s33, v154\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[12:13], s19, v186\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[60:61], s[80:81], s[6:7]\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v153, 11, v1\n\t.loc\t1 1226 12                       ; 7.ttgir:1226:12\n\tv_or_b32_e32 v110, s33, v152\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[14:15], s19, v187\n\tv_cmp_gt_i32_e64 s[34:35], s19, v190\n\t.loc\t1 1274 12                       ; 7.ttgir:1274:12\n\ts_waitcnt lgkmcnt(5)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[132:133], v[102:103], v[66:81]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v102, s1, v149\n\tv_add_u32_e32 v103, s1, v150\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[4:5], v166, v102\n\tv_cmp_ge_i32_e64 s[6:7], v166, v103\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[62:63], s[80:81], s[12:13]\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v161, 25, v1\n\t.loc\t1 1226 12                       ; 7.ttgir:1226:12\n\tv_or_b32_e32 v111, s33, v153\n\t.loc\t1 1274 12                       ; 7.ttgir:1274:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[134:135], v[104:105], v[66:81]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v104, s1, v156\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[12:13], v166, v104\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[20:21], s19, v110\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[64:65], s[80:81], s[14:15]\n\ts_or_b64 s[78:79], s[80:81], s[34:35]\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v158, 17, v1\n\t.loc\t1 1223 12                       ; 7.ttgir:1223:12\n\tv_or_b32_e32 v164, s33, v1\n\t.loc\t1 1275 12                       ; 7.ttgir:1275:12\n\ts_waitcnt lgkmcnt(4)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[136:137], v[98:99], v[66:81]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v99, s1, v148\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[2:3], v166, v99\n\t.loc\t1 1252 12                       ; 7.ttgir:1252:12\n\ts_and_b64 s[2:3], s[2:3], s[56:57]\n\t.loc\t1 1229 12                       ; 7.ttgir:1229:12\n\tv_or_b32_e32 v124, s33, v155\n\t.loc\t1 1232 12                       ; 7.ttgir:1232:12\n\tv_or_b32_e32 v112, s33, v161\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[22:23], s19, v111\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[66:67], s[80:81], s[20:21]\n\t.loc\t1 1275 12                       ; 7.ttgir:1275:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[138:139], v[100:101], v[66:81]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v100, s1, v151\n\tv_add_u32_e32 v101, s1, v152\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[14:15], v166, v100\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v105, s1, v153\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[20:21], v166, v101\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v159, 18, v1\n\t.loc\t1 1229 12                       ; 7.ttgir:1229:12\n\tv_or_b32_e32 v125, s33, v158\n\t.loc\t1 1276 12                       ; 7.ttgir:1276:12\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[140:141], v[94:95], v[66:81]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v94, s1, v154\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[34:35], v166, v94\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e32 vcc, s19, v164\n\tv_cmp_gt_i32_e64 s[24:25], s19, v124\n\tv_cmp_gt_i32_e64 s[36:37], s19, v112\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[68:69], s[80:81], s[22:23]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v98, s1, v1\n\t.loc\t1 1276 12                       ; 7.ttgir:1276:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[142:143], v[96:97], v[66:81]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v106, s1, v155\n\tv_add_u32_e32 v95, s1, v161\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[22:23], v166, v105\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v160, 19, v1\n\t.loc\t1 1229 12                       ; 7.ttgir:1229:12\n\tv_or_b32_e32 v188, s33, v159\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[26:27], s19, v125\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[50:51], s[80:81], vcc\n\t.loc\t1 1277 12                       ; 7.ttgir:1277:12\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[144:145], v[90:91], v[66:81]\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_mov_b32_e32 v90, 0xff800000\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[70:71], s[80:81], s[24:25]\n\ts_or_b64 s[48:49], s[80:81], s[36:37]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v107, s1, v158\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e32 vcc, v166, v98\n\tv_cmp_ge_i32_e64 s[24:25], v166, v106\n\tv_cmp_ge_i32_e64 s[36:37], v166, v95\n\t.loc\t1 1277 12                       ; 7.ttgir:1277:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[146:147], v[92:93], v[66:81]\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v92, v90, 0, s[2:3]\n\t.loc\t1 1252 12                       ; 7.ttgir:1252:12\n\ts_and_b64 s[2:3], s[4:5], s[58:59]\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v93, v90, 0, s[2:3]\n\t.loc\t1 1252 12                       ; 7.ttgir:1252:12\n\ts_and_b64 s[2:3], s[6:7], s[60:61]\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v94, v90, 0, s[2:3]\n\t.loc\t1 1255 12                       ; 7.ttgir:1255:12\n\ts_and_b64 s[2:3], s[12:13], s[62:63]\n\t.loc\t1 1229 12                       ; 7.ttgir:1229:12\n\tv_or_b32_e32 v189, s33, v160\n\t.loc\t1 1278 12                       ; 7.ttgir:1278:12\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[118:119], v[86:87], v[66:81]\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v86, v90, 0, s[2:3]\n\t.loc\t1 1255 12                       ; 7.ttgir:1255:12\n\ts_and_b64 s[2:3], s[14:15], s[64:65]\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v87, v90, 0, s[2:3]\n\t.loc\t1 1255 12                       ; 7.ttgir:1255:12\n\ts_and_b64 s[2:3], s[20:21], s[66:67]\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v95, v90, 0, s[2:3]\n\t.loc\t1 1255 12                       ; 7.ttgir:1255:12\n\ts_and_b64 s[2:3], s[22:23], s[68:69]\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[28:29], s19, v188\n\t.loc\t1 1278 12                       ; 7.ttgir:1278:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[120:121], v[88:89], v[66:81]\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[72:73], s[80:81], s[26:27]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v108, s1, v159\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[26:27], v166, v107\n\t.loc\t1 1252 12                       ; 7.ttgir:1252:12\n\ts_and_b64 s[50:51], vcc, s[50:51]\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v96, v90, 0, s[2:3]\n\t.loc\t1 1258 12                       ; 7.ttgir:1258:12\n\ts_and_b64 s[2:3], s[24:25], s[70:71]\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[30:31], s19, v189\n\t.loc\t1 1279 12                       ; 7.ttgir:1279:12\n\ts_waitcnt lgkmcnt(0)\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[114:115], v[82:83], v[66:81]\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[74:75], s[80:81], s[28:29]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v109, s1, v160\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[28:29], v166, v108\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v91, v90, 0, s[50:51]\n\tv_cndmask_b32_e64 v88, v90, 0, s[2:3]\n\t.loc\t1 1258 12                       ; 7.ttgir:1258:12\n\ts_and_b64 s[2:3], s[26:27], s[72:73]\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v162, 26, v1\n\t.loc\t1 1279 12                       ; 7.ttgir:1279:12\n\tv_mfma_f32_32x32x8_f16 v[66:81], v[116:117], v[84:85], v[66:81]\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[76:77], s[80:81], s[30:31]\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[30:31], v166, v109\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v89, v90, 0, s[2:3]\n\t.loc\t1 1258 12                       ; 7.ttgir:1258:12\n\ts_and_b64 s[2:3], s[28:29], s[74:75]\n\t.loc\t1 109 11                        ; 7.ttgir:109:11\n\tv_or_b32_e32 v163, 27, v1\n\t.loc\t1 1232 12                       ; 7.ttgir:1232:12\n\tv_or_b32_e32 v113, s33, v162\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v97, v90, 0, s[2:3]\n\t.loc\t1 1293 12                       ; 7.ttgir:1293:12\n\ts_nop 3\n\tv_fmac_f32_e32 v91, 0x3e0293ee, v66\n\tv_fmac_f32_e32 v92, 0x3e0293ee, v67\n\t.loc\t1 1258 12                       ; 7.ttgir:1258:12\n\ts_and_b64 s[2:3], s[30:31], s[76:77]\n\t.loc\t1 1293 12                       ; 7.ttgir:1293:12\n\tv_fmac_f32_e32 v93, 0x3e0293ee, v68\n\tv_fmac_f32_e32 v94, 0x3e0293ee, v69\n\t.loc\t1 1303 15                       ; 7.ttgir:1303:15\n\tv_max_f32_e32 v66, v91, v92\n\t.loc\t1 1232 12                       ; 7.ttgir:1232:12\n\tv_or_b32_e32 v126, s33, v163\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[38:39], s19, v113\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v110, s1, v162\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v98, v90, 0, s[2:3]\n\t.loc\t1 1261 12                       ; 7.ttgir:1261:12\n\ts_and_b64 s[2:3], s[34:35], s[78:79]\n\t.loc\t1 1295 12                       ; 7.ttgir:1295:12\n\tv_fmac_f32_e32 v86, 0x3e0293ee, v70\n\tv_fmac_f32_e32 v87, 0x3e0293ee, v71\n\t.loc\t1 1303 15                       ; 7.ttgir:1303:15\n\tv_max3_f32 v66, v66, v93, v94\n\t.loc\t1 1234 12                       ; 7.ttgir:1234:12\n\tv_cmp_gt_i32_e64 s[40:41], s19, v126\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[42:43], s[80:81], s[38:39]\n\t.loc\t1 1238 12                       ; 7.ttgir:1238:12\n\tv_add_u32_e32 v111, s1, v163\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[38:39], v166, v110\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v82, v90, 0, s[2:3]\n\t.loc\t1 1261 12                       ; 7.ttgir:1261:12\n\ts_and_b64 s[2:3], s[36:37], s[48:49]\n\t.loc\t1 1295 12                       ; 7.ttgir:1295:12\n\tv_fmac_f32_e32 v95, 0x3e0293ee, v72\n\tv_fmac_f32_e32 v96, 0x3e0293ee, v73\n\t.loc\t1 1303 15                       ; 7.ttgir:1303:15\n\tv_max3_f32 v66, v66, v86, v87\n\t.loc\t1 1236 12                       ; 7.ttgir:1236:12\n\ts_or_b64 s[80:81], s[80:81], s[40:41]\n\t.loc\t1 1249 12                       ; 7.ttgir:1249:12\n\tv_cmp_ge_i32_e64 s[40:41], v166, v111\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v83, v90, 0, s[2:3]\n\t.loc\t1 1261 12                       ; 7.ttgir:1261:12\n\ts_and_b64 s[2:3], s[38:39], s[42:43]\n\t.loc\t1 1297 12                       ; 7.ttgir:1297:12\n\tv_fmac_f32_e32 v88, 0x3e0293ee, v74\n\tv_fmac_f32_e32 v89, 0x3e0293ee, v75\n\t.loc\t1 1303 15                       ; 7.ttgir:1303:15\n\tv_max3_f32 v66, v66, v95, v96\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v99, v90, 0, s[2:3]\n\t.loc\t1 1261 12                       ; 7.ttgir:1261:12\n\ts_and_b64 s[2:3], s[40:41], s[80:81]\n\t.loc\t1 1297 12                       ; 7.ttgir:1297:12\n\tv_fmac_f32_e32 v97, 0x3e0293ee, v76\n\tv_fmac_f32_e32 v98, 0x3e0293ee, v77\n\t.loc\t1 1303 15                       ; 7.ttgir:1303:15\n\tv_max3_f32 v66, v66, v88, v89\n\t.loc\t1 1263 12                       ; 7.ttgir:1263:12\n\tv_cndmask_b32_e64 v90, v90, 0, s[2:3]\n\t.loc\t1 1299 12                       ; 7.ttgir:1299:12\n\tv_fmac_f32_e32 v82, 0x3e0293ee, v78\n\tv_fmac_f32_e32 v83, 0x3e0293ee, v79\n\t.loc\t1 1303 15                       ; 7.ttgir:1303:15\n\tv_max3_f32 v66, v66, v97, v98\n\t.loc\t1 1299 12                       ; 7.ttgir:1299:12\n\tv_fmac_f32_e32 v99, 0x3e0293ee, v80\n\tv_fmac_f32_e32 v90, 0x3e0293ee, v81\n\t.loc\t1 1303 15                       ; 7.ttgir:1303:15\n\tv_max3_f32 v66, v66, v82, v83\n\tv_max3_f32 v66, v66, v99, v90\n\t.loc\t1 1301 12                       ; 7.ttgir:1301:12\n\tds_bpermute_b32 v67, v157, v66\n\t.loc\t1 1513 13                       ; 7.ttgir:1513:13\n\ts_cmp_le_i32 s44, s52\n\ts_mov_b64 s[2:3], -1\n\t.loc\t1 1306 12                       ; 7.ttgir:1306:12\n\ts_waitcnt lgkmcnt(0)\n\tv_max3_f32 v66, v123, v66, v67\n\t.loc\t1 1310 12                       ; 7.ttgir:1310:12\n\tv_sub_f32_e32 v68, v92, v66\n\tv_sub_f32_e32 v67, v91, v66\n\tv_sub_f32_e32 v69, v93, v66\n\tv_sub_f32_e32 v70, v94, v66\n\t.loc\t1 1317 12                       ; 7.ttgir:1317:12\n\tv_exp_f32_e32 v78, v68\n\t.loc\t1 1327 12                       ; 7.ttgir:1327:12\n\tv_sub_f32_e32 v68, v123, v66\n\t.loc\t1 1317 12                       ; 7.ttgir:1317:12\n\tv_exp_f32_e32 v67, v67\n\tv_exp_f32_e32 v79, v69\n\tv_exp_f32_e32 v80, v70\n\t.loc\t1 1328 12                       ; 7.ttgir:1328:12\n\tv_exp_f32_e32 v84, v68\n\t.loc\t1 1381 13                       ; 7.ttgir:1381:13\n\tds_read_b64 v[68:69], v168 offset:8192\n\t.loc\t1 1414 13                       ; 7.ttgir:1414:13\n\tv_cvt_f16_f32_e32 v76, v67\n\tv_cvt_f16_f32_e32 v91, v78\n\tv_cvt_f16_f32_e32 v77, v79\n\tv_cvt_f16_f32_e32 v92, v80\n\t.loc\t1 1312 12                       ; 7.ttgir:1312:12\n\tv_sub_f32_e32 v71, v86, v66\n\tv_sub_f32_e32 v72, v87, v66\n\tv_sub_f32_e32 v73, v95, v66\n\tv_sub_f32_e32 v74, v96, v66\n\t.loc\t1 1318 12                       ; 7.ttgir:1318:12\n\tv_exp_f32_e32 v81, v71\n\tv_exp_f32_e32 v85, v72\n\tv_exp_f32_e32 v86, v73\n\tv_exp_f32_e32 v87, v74\n\t.loc\t1 1383 13                       ; 7.ttgir:1383:13\n\tds_read_b64 v[70:71], v167 offset:8192\n\t.loc\t1 1385 13                       ; 7.ttgir:1385:13\n\tds_read_b64 v[72:73], v169 offset:8192\n\t.loc\t1 1387 13                       ; 7.ttgir:1387:13\n\tds_read_b64 v[74:75], v170 offset:8192\n\t.loc\t1 1333 12                       ; 7.ttgir:1333:12\n\tv_mul_f32_e32 v50, v50, v84\n\tv_mul_f32_e32 v51, v51, v84\n\tv_mul_f32_e32 v52, v52, v84\n\tv_mul_f32_e32 v53, v53, v84\n\t.loc\t1 1336 12                       ; 7.ttgir:1336:12\n\tv_mul_f32_e32 v54, v54, v84\n\tv_mul_f32_e32 v55, v55, v84\n\tv_mul_f32_e32 v56, v56, v84\n\tv_mul_f32_e32 v57, v57, v84\n\t.loc\t1 1339 12                       ; 7.ttgir:1339:12\n\tv_mul_f32_e32 v58, v58, v84\n\tv_mul_f32_e32 v59, v59, v84\n\tv_mul_f32_e32 v60, v60, v84\n\tv_mul_f32_e32 v61, v61, v84\n\t.loc\t1 1342 12                       ; 7.ttgir:1342:12\n\tv_mul_f32_e32 v62, v62, v84\n\tv_mul_f32_e32 v63, v63, v84\n\tv_mul_f32_e32 v64, v64, v84\n\tv_mul_f32_e32 v65, v65, v84\n\t.loc\t1 1426 13                       ; 7.ttgir:1426:13\n\tv_pack_b32_f16 v77, v77, v92\n\tv_pack_b32_f16 v76, v76, v91\n\t.loc\t1 1345 12                       ; 7.ttgir:1345:12\n\tv_mul_f32_e32 v34, v34, v84\n\tv_mul_f32_e32 v35, v35, v84\n\tv_mul_f32_e32 v36, v36, v84\n\tv_mul_f32_e32 v37, v37, v84\n\t.loc\t1 1348 12                       ; 7.ttgir:1348:12\n\tv_mul_f32_e32 v38, v38, v84\n\tv_mul_f32_e32 v39, v39, v84\n\tv_mul_f32_e32 v40, v40, v84\n\tv_mul_f32_e32 v41, v41, v84\n\t.loc\t1 1351 12                       ; 7.ttgir:1351:12\n\tv_mul_f32_e32 v42, v42, v84\n\tv_mul_f32_e32 v43, v43, v84\n\tv_mul_f32_e32 v44, v44, v84\n\tv_mul_f32_e32 v45, v45, v84\n\t.loc\t1 1354 12                       ; 7.ttgir:1354:12\n\tv_mul_f32_e32 v46, v46, v84\n\tv_mul_f32_e32 v47, v47, v84\n\tv_mul_f32_e32 v48, v48, v84\n\tv_mul_f32_e32 v49, v49, v84\n\t.loc\t1 1426 13                       ; 7.ttgir:1426:13\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]\n\t.loc\t1 1389 13                       ; 7.ttgir:1389:13\n\tds_read_b64 v[68:69], v172 offset:8192\n\t.loc\t1 1357 12                       ; 7.ttgir:1357:12\n\tv_mul_f32_e32 v18, v18, v84\n\tv_mul_f32_e32 v19, v19, v84\n\tv_mul_f32_e32 v20, v20, v84\n\tv_mul_f32_e32 v21, v21, v84\n\t.loc\t1 1360 12                       ; 7.ttgir:1360:12\n\tv_mul_f32_e32 v22, v22, v84\n\tv_mul_f32_e32 v23, v23, v84\n\t.loc\t1 1427 13                       ; 7.ttgir:1427:13\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]\n\t.loc\t1 1360 12                       ; 7.ttgir:1360:12\n\tv_mul_f32_e32 v24, v24, v84\n\tv_mul_f32_e32 v25, v25, v84\n\t.loc\t1 1363 12                       ; 7.ttgir:1363:12\n\tv_mul_f32_e32 v26, v26, v84\n\tv_mul_f32_e32 v27, v27, v84\n\tv_mul_f32_e32 v28, v28, v84\n\tv_mul_f32_e32 v29, v29, v84\n\t.loc\t1 1366 13                       ; 7.ttgir:1366:13\n\tv_mul_f32_e32 v30, v30, v84\n\tv_mul_f32_e32 v31, v31, v84\n\tv_mul_f32_e32 v32, v32, v84\n\tv_mul_f32_e32 v33, v33, v84\n\t.loc\t1 1369 13                       ; 7.ttgir:1369:13\n\tv_mul_f32_e32 v2, v2, v84\n\tv_mul_f32_e32 v3, v3, v84\n\tv_mul_f32_e32 v4, v4, v84\n\tv_mul_f32_e32 v5, v5, v84\n\t.loc\t1 1372 13                       ; 7.ttgir:1372:13\n\tv_mul_f32_e32 v6, v6, v84\n\tv_mul_f32_e32 v7, v7, v84\n\tv_mul_f32_e32 v8, v8, v84\n\tv_mul_f32_e32 v9, v9, v84\n\t.loc\t1 1375 13                       ; 7.ttgir:1375:13\n\tv_mul_f32_e32 v10, v10, v84\n\tv_mul_f32_e32 v11, v11, v84\n\tv_mul_f32_e32 v12, v12, v84\n\tv_mul_f32_e32 v13, v13, v84\n\t.loc\t1 1378 13                       ; 7.ttgir:1378:13\n\tv_mul_f32_e32 v14, v14, v84\n\tv_mul_f32_e32 v15, v15, v84\n\tv_mul_f32_e32 v16, v16, v84\n\tv_mul_f32_e32 v17, v17, v84\n\t.loc\t1 1428 13                       ; 7.ttgir:1428:13\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]\n\t.loc\t1 1415 13                       ; 7.ttgir:1415:13\n\tv_cvt_f16_f32_e32 v91, v81\n\tv_cvt_f16_f32_e32 v92, v85\n\t.loc\t1 1324 15                       ; 7.ttgir:1324:15\n\tv_add_f32_e32 v67, v67, v78\n\tv_add_f32_e32 v67, v79, v67\n\tv_add_f32_e32 v67, v80, v67\n\tv_add_f32_e32 v67, v81, v67\n\tv_add_f32_e32 v67, v85, v67\n\t.loc\t1 1429 13                       ; 7.ttgir:1429:13\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]\n\t.loc\t1 1415 13                       ; 7.ttgir:1415:13\n\tv_cvt_f16_f32_e32 v76, v86\n\tv_cvt_f16_f32_e32 v77, v87\n\t.loc\t1 1391 13                       ; 7.ttgir:1391:13\n\tds_read_b64 v[70:71], v171 offset:8192\n\t.loc\t1 1393 13                       ; 7.ttgir:1393:13\n\tds_read_b64 v[72:73], v173 offset:8192\n\t.loc\t1 1395 13                       ; 7.ttgir:1395:13\n\tds_read_b64 v[74:75], v174 offset:8192\n\t.loc\t1 1324 15                       ; 7.ttgir:1324:15\n\tv_add_f32_e32 v67, v86, v67\n\tv_add_f32_e32 v67, v87, v67\n\t.loc\t1 1430 13                       ; 7.ttgir:1430:13\n\tv_pack_b32_f16 v77, v76, v77\n\tv_pack_b32_f16 v76, v91, v92\n\ts_waitcnt lgkmcnt(3)\n\ts_nop 0\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]\n\t.loc\t1 1314 12                       ; 7.ttgir:1314:12\n\tv_sub_f32_e32 v68, v88, v66\n\tv_sub_f32_e32 v69, v89, v66\n\t.loc\t1 1319 12                       ; 7.ttgir:1319:12\n\tv_exp_f32_e32 v88, v68\n\tv_exp_f32_e32 v89, v69\n\t.loc\t1 1397 13                       ; 7.ttgir:1397:13\n\tds_read_b64 v[68:69], v177 offset:8192\n\t.loc\t1 1416 13                       ; 7.ttgir:1416:13\n\tv_cvt_f16_f32_e32 v93, v88\n\t.loc\t1 1431 13                       ; 7.ttgir:1431:13\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]\n\t.loc\t1 1314 12                       ; 7.ttgir:1314:12\n\tv_sub_f32_e32 v70, v97, v66\n\tv_sub_f32_e32 v71, v98, v66\n\t.loc\t1 1319 12                       ; 7.ttgir:1319:12\n\tv_exp_f32_e32 v91, v70\n\tv_exp_f32_e32 v92, v71\n\t.loc\t1 1416 13                       ; 7.ttgir:1416:13\n\tv_cvt_f16_f32_e32 v94, v89\n\t.loc\t1 1324 15                       ; 7.ttgir:1324:15\n\tv_add_f32_e32 v67, v88, v67\n\tv_add_f32_e32 v67, v89, v67\n\t.loc\t1 1432 13                       ; 7.ttgir:1432:13\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]\n\t.loc\t1 1324 15                       ; 7.ttgir:1324:15\n\tv_add_f32_e32 v67, v91, v67\n\tv_add_f32_e32 v67, v92, v67\n\t.loc\t1 1433 13                       ; 7.ttgir:1433:13\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]\n\t.loc\t1 1416 13                       ; 7.ttgir:1416:13\n\tv_cvt_f16_f32_e32 v76, v91\n\tv_cvt_f16_f32_e32 v77, v92\n\t.loc\t1 1399 13                       ; 7.ttgir:1399:13\n\tds_read_b64 v[70:71], v175 offset:8192\n\t.loc\t1 1401 13                       ; 7.ttgir:1401:13\n\tds_read_b64 v[72:73], v179 offset:8192\n\t.loc\t1 1403 13                       ; 7.ttgir:1403:13\n\tds_read_b64 v[74:75], v181 offset:8192\n\t.loc\t1 1434 13                       ; 7.ttgir:1434:13\n\tv_pack_b32_f16 v77, v76, v77\n\tv_pack_b32_f16 v76, v93, v94\n\ts_waitcnt lgkmcnt(3)\n\ts_nop 0\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]\n\t.loc\t1 1316 12                       ; 7.ttgir:1316:12\n\tv_sub_f32_e32 v68, v82, v66\n\tv_sub_f32_e32 v69, v83, v66\n\t.loc\t1 1320 12                       ; 7.ttgir:1320:12\n\tv_exp_f32_e32 v82, v68\n\tv_exp_f32_e32 v83, v69\n\t.loc\t1 1405 13                       ; 7.ttgir:1405:13\n\tds_read_b64 v[68:69], v178 offset:8192\n\t.loc\t1 1417 13                       ; 7.ttgir:1417:13\n\tv_cvt_f16_f32_e32 v78, v82\n\t.loc\t1 1435 13                       ; 7.ttgir:1435:13\n\ts_waitcnt lgkmcnt(3)\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]\n\t.loc\t1 1316 12                       ; 7.ttgir:1316:12\n\tv_sub_f32_e32 v70, v99, v66\n\tv_sub_f32_e32 v71, v90, v66\n\t.loc\t1 1320 12                       ; 7.ttgir:1320:12\n\tv_exp_f32_e32 v90, v70\n\tv_exp_f32_e32 v93, v71\n\t.loc\t1 1417 13                       ; 7.ttgir:1417:13\n\tv_cvt_f16_f32_e32 v94, v83\n\t.loc\t1 1324 15                       ; 7.ttgir:1324:15\n\tv_add_f32_e32 v67, v82, v67\n\tv_add_f32_e32 v67, v83, v67\n\t.loc\t1 1436 13                       ; 7.ttgir:1436:13\n\ts_waitcnt lgkmcnt(2)\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]\n\t.loc\t1 1324 15                       ; 7.ttgir:1324:15\n\tv_add_f32_e32 v67, v90, v67\n\tv_add_f32_e32 v67, v93, v67\n\t.loc\t1 1437 13                       ; 7.ttgir:1437:13\n\ts_waitcnt lgkmcnt(1)\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]\n\t.loc\t1 1417 13                       ; 7.ttgir:1417:13\n\tv_cvt_f16_f32_e32 v76, v90\n\tv_cvt_f16_f32_e32 v77, v93\n\t.loc\t1 1407 13                       ; 7.ttgir:1407:13\n\tds_read_b64 v[70:71], v176 offset:8192\n\t.loc\t1 1409 13                       ; 7.ttgir:1409:13\n\tds_read_b64 v[72:73], v180 offset:8192\n\t.loc\t1 1411 13                       ; 7.ttgir:1411:13\n\tds_read_b64 v[74:75], v182 offset:8192\n\t.loc\t1 1438 13                       ; 7.ttgir:1438:13\n\tv_pack_b32_f16 v77, v76, v77\n\tv_pack_b32_f16 v76, v78, v94\n\ts_waitcnt lgkmcnt(3)\n\ts_nop 0\n\tv_mfma_f32_32x32x8_f16 v[50:65], v[68:69], v[76:77], v[50:65]\n\t.loc\t1 1322 12                       ; 7.ttgir:1322:12\n\tds_bpermute_b32 v68, v157, v67\n\t.loc\t1 1324 15                       ; 7.ttgir:1324:15\n\ts_waitcnt lgkmcnt(0)\n\tv_add_f32_e32 v79, v67, v68\n\t.loc\t1 1413 13                       ; 7.ttgir:1413:13\n\tv_fmac_f32_e32 v79, v122, v84\n\t.loc\t1 1439 13                       ; 7.ttgir:1439:13\n\tv_mfma_f32_32x32x8_f16 v[34:49], v[70:71], v[76:77], v[34:49]\n\t.loc\t1 1446 13                       ; 7.ttgir:1446:13\n\tv_div_scale_f32 v67, s[4:5], v79, v79, 1.0\n\tv_rcp_f32_e32 v67, v67\n\tv_div_scale_f32 v68, vcc, 1.0, v79, 1.0\n\t.loc\t1 1513 13                       ; 7.ttgir:1513:13\n\ts_cselect_b64 s[4:5], -1, 0\n\t.loc\t1 1446 13                       ; 7.ttgir:1446:13\n\tv_mul_f32_e32 v67, v68, v67\n\t.loc\t1 1440 13                       ; 7.ttgir:1440:13\n\tv_mfma_f32_32x32x8_f16 v[18:33], v[72:73], v[76:77], v[18:33]\n\t.loc\t1 1514 13                       ; 7.ttgir:1514:13\n\ts_cmp_ge_i32 s44, s0\n\t.loc\t1 1446 13                       ; 7.ttgir:1446:13\n\tv_div_fmas_f32 v67, 0, 0, v67\n\t.loc\t1 1514 13                       ; 7.ttgir:1514:13\n\ts_cselect_b64 s[6:7], -1, 0\n\t.loc\t1 1446 13                       ; 7.ttgir:1446:13\n\tv_div_fixup_f32 v80, v67, v79, 1.0\n\t.loc\t1 1516 5                        ; 7.ttgir:1516:5\n\ts_or_b64 s[4:5], s[4:5], s[6:7]\n\t.loc\t1 1496 13                       ; 7.ttgir:1496:13\n\tv_fma_mixlo_f16 v78, v53, v80, 0\n\t.loc\t1 1497 13                       ; 7.ttgir:1497:13\n\tv_fma_mixlo_f16 v69, v54, v80, 0\n\t.loc\t1 1441 13                       ; 7.ttgir:1441:13\n\tv_mfma_f32_32x32x8_f16 v[2:17], v[74:75], v[76:77], v[2:17]\n\t.loc\t1 1496 13                       ; 7.ttgir:1496:13\n\tv_fma_mixlo_f16 v75, v50, v80, 0\n\tv_fma_mixlo_f16 v76, v51, v80, 0\n\tv_fma_mixlo_f16 v77, v52, v80, 0\n\t.loc\t1 1497 13                       ; 7.ttgir:1497:13\n\tv_fma_mixlo_f16 v71, v55, v80, 0\n\tv_fma_mixlo_f16 v73, v56, v80, 0\n\tv_fma_mixlo_f16 v74, v57, v80, 0\n\t.loc\t1 1498 13                       ; 7.ttgir:1498:13\n\tv_fma_mixlo_f16 v67, v58, v80, 0\n\tv_fma_mixlo_f16 v68, v59, v80, 0\n\tv_fma_mixlo_f16 v70, v60, v80, 0\n\tv_fma_mixlo_f16 v72, v61, v80, 0\n\t.loc\t1 1499 13                       ; 7.ttgir:1499:13\n\tv_fma_mixlo_f16 v60, v62, v80, 0\n\tv_fma_mixlo_f16 v62, v63, v80, 0\n\tv_fma_mixlo_f16 v64, v64, v80, 0\n\tv_fma_mixlo_f16 v65, v65, v80, 0\n\t.loc\t1 1500 13                       ; 7.ttgir:1500:13\n\tv_fma_mixlo_f16 v56, v34, v80, 0\n\tv_fma_mixlo_f16 v58, v35, v80, 0\n\tv_fma_mixlo_f16 v61, v36, v80, 0\n\tv_fma_mixlo_f16 v63, v37, v80, 0\n\t.loc\t1 1501 13                       ; 7.ttgir:1501:13\n\tv_fma_mixlo_f16 v52, v38, v80, 0\n\tv_fma_mixlo_f16 v54, v39, v80, 0\n\tv_fma_mixlo_f16 v57, v40, v80, 0\n\tv_fma_mixlo_f16 v59, v41, v80, 0\n\t.loc\t1 1502 13                       ; 7.ttgir:1502:13\n\tv_fma_mixlo_f16 v50, v42, v80, 0\n\tv_fma_mixlo_f16 v51, v43, v80, 0\n\tv_fma_mixlo_f16 v53, v44, v80, 0\n\tv_fma_mixlo_f16 v55, v45, v80, 0\n\t.loc\t1 1503 13                       ; 7.ttgir:1503:13\n\tv_fma_mixlo_f16 v44, v46, v80, 0\n\tv_fma_mixlo_f16 v46, v47, v80, 0\n\tv_fma_mixlo_f16 v48, v48, v80, 0\n\tv_fma_mixlo_f16 v49, v49, v80, 0\n\t.loc\t1 1504 13                       ; 7.ttgir:1504:13\n\tv_fma_mixlo_f16 v40, v18, v80, 0\n\tv_fma_mixlo_f16 v42, v19, v80, 0\n\tv_fma_mixlo_f16 v45, v20, v80, 0\n\tv_fma_mixlo_f16 v47, v21, v80, 0\n\t.loc\t1 1505 13                       ; 7.ttgir:1505:13\n\tv_fma_mixlo_f16 v36, v22, v80, 0\n\tv_fma_mixlo_f16 v38, v23, v80, 0\n\tv_fma_mixlo_f16 v41, v24, v80, 0\n\tv_fma_mixlo_f16 v43, v25, v80, 0\n\t.loc\t1 1506 13                       ; 7.ttgir:1506:13\n\tv_fma_mixlo_f16 v34, v26, v80, 0\n\tv_fma_mixlo_f16 v35, v27, v80, 0\n\tv_fma_mixlo_f16 v37, v28, v80, 0\n\tv_fma_mixlo_f16 v39, v29, v80, 0\n\t.loc\t1 1507 13                       ; 7.ttgir:1507:13\n\tv_fma_mixlo_f16 v24, v30, v80, 0\n\tv_fma_mixlo_f16 v26, v31, v80, 0\n\tv_fma_mixlo_f16 v28, v32, v80, 0\n\tv_fma_mixlo_f16 v29, v33, v80, 0\n\t.loc\t1 1508 13                       ; 7.ttgir:1508:13\n\tv_fma_mixlo_f16 v20, v2, v80, 0\n\tv_fma_mixlo_f16 v22, v3, v80, 0\n\tv_fma_mixlo_f16 v25, v4, v80, 0\n\tv_fma_mixlo_f16 v27, v5, v80, 0\n\t.loc\t1 1509 13                       ; 7.ttgir:1509:13\n\tv_fma_mixlo_f16 v18, v6, v80, 0\n\tv_fma_mixlo_f16 v19, v7, v80, 0\n\tv_fma_mixlo_f16 v21, v8, v80, 0\n\tv_fma_mixlo_f16 v23, v9, v80, 0\n\t.loc\t1 1510 13                       ; 7.ttgir:1510:13\n\tv_fma_mixlo_f16 v4, v10, v80, 0\n\tv_fma_mixlo_f16 v6, v11, v80, 0\n\tv_fma_mixlo_f16 v8, v12, v80, 0\n\tv_fma_mixlo_f16 v9, v13, v80, 0\n\t.loc\t1 1511 13                       ; 7.ttgir:1511:13\n\tv_fma_mixlo_f16 v2, v14, v80, 0\n\tv_fma_mixlo_f16 v3, v15, v80, 0\n\tv_fma_mixlo_f16 v5, v16, v80, 0\n\tv_fma_mixlo_f16 v7, v17, v80, 0\n\t.loc\t1 1516 5                        ; 7.ttgir:1516:5\n\ts_and_b64 vcc, exec, s[4:5]\n\ts_cbranch_vccnz .LBB0_18\n; %bb.17:\n\t.loc\t1 1520 13                       ; 7.ttgir:1520:13\n\tv_cmp_gt_i32_e32 vcc, s44, v166\n\t.loc\t1 1522 13                       ; 7.ttgir:1522:13\n\ts_nop 1\n\tv_cndmask_b32_e64 v75, v75, 0, vcc\n\tv_cndmask_b32_e64 v76, v76, 0, vcc\n\tv_cndmask_b32_e64 v77, v77, 0, vcc\n\tv_cndmask_b32_e64 v78, v78, 0, vcc\n\tv_cndmask_b32_e64 v69, v69, 0, vcc\n\tv_cndmask_b32_e64 v71, v71, 0, vcc\n\tv_cndmask_b32_e64 v73, v73, 0, vcc\n\tv_cndmask_b32_e64 v74, v74, 0, vcc\n\tv_cndmask_b32_e64 v67, v67, 0, vcc\n\tv_cndmask_b32_e64 v68, v68, 0, vcc\n\tv_cndmask_b32_e64 v70, v70, 0, vcc\n\tv_cndmask_b32_e64 v72, v72, 0, vcc\n\tv_cndmask_b32_e64 v60, v60, 0, vcc\n\tv_cndmask_b32_e64 v62, v62, 0, vcc\n\tv_cndmask_b32_e64 v64, v64, 0, vcc\n\tv_cndmask_b32_e64 v65, v65, 0, vcc\n\tv_cndmask_b32_e64 v56, v56, 0, vcc\n\tv_cndmask_b32_e64 v58, v58, 0, vcc\n\tv_cndmask_b32_e64 v61, v61, 0, vcc\n\tv_cndmask_b32_e64 v63, v63, 0, vcc\n\tv_cndmask_b32_e64 v52, v52, 0, vcc\n\tv_cndmask_b32_e64 v54, v54, 0, vcc\n\tv_cndmask_b32_e64 v57, v57, 0, vcc\n\tv_cndmask_b32_e64 v59, v59, 0, vcc\n\tv_cndmask_b32_e64 v50, v50, 0, vcc\n\tv_cndmask_b32_e64 v51, v51, 0, vcc\n\tv_cndmask_b32_e64 v53, v53, 0, vcc\n\tv_cndmask_b32_e64 v55, v55, 0, vcc\n\tv_cndmask_b32_e64 v44, v44, 0, vcc\n\tv_cndmask_b32_e64 v46, v46, 0, vcc\n\tv_cndmask_b32_e64 v48, v48, 0, vcc\n\tv_cndmask_b32_e64 v49, v49, 0, vcc\n\tv_cndmask_b32_e64 v40, v40, 0, vcc\n\tv_cndmask_b32_e64 v42, v42, 0, vcc\n\tv_cndmask_b32_e64 v45, v45, 0, vcc\n\tv_cndmask_b32_e64 v47, v47, 0, vcc\n\tv_cndmask_b32_e64 v36, v36, 0, vcc\n\tv_cndmask_b32_e64 v38, v38, 0, vcc\n\tv_cndmask_b32_e64 v41, v41, 0, vcc\n\tv_cndmask_b32_e64 v43, v43, 0, vcc\n\tv_cndmask_b32_e64 v34, v34, 0, vcc\n\tv_cndmask_b32_e64 v35, v35, 0, vcc\n\tv_cndmask_b32_e64 v37, v37, 0, vcc\n\tv_cndmask_b32_e64 v39, v39, 0, vcc\n\tv_cndmask_b32_e64 v24, v24, 0, vcc\n\tv_cndmask_b32_e64 v26, v26, 0, vcc\n\tv_cndmask_b32_e64 v28, v28, 0, vcc\n\tv_cndmask_b32_e64 v29, v29, 0, vcc\n\tv_cndmask_b32_e64 v20, v20, 0, vcc\n\tv_cndmask_b32_e64 v22, v22, 0, vcc\n\tv_cndmask_b32_e64 v25, v25, 0, vcc\n\tv_cndmask_b32_e64 v27, v27, 0, vcc\n\tv_cndmask_b32_e64 v18, v18, 0, vcc\n\tv_cndmask_b32_e64 v19, v19, 0, vcc\n\tv_cndmask_b32_e64 v21, v21, 0, vcc\n\tv_cndmask_b32_e64 v23, v23, 0, vcc\n\tv_cndmask_b32_e64 v4, v4, 0, vcc\n\tv_cndmask_b32_e64 v6, v6, 0, vcc\n\tv_cndmask_b32_e64 v8, v8, 0, vcc\n\tv_cndmask_b32_e64 v9, v9, 0, vcc\n\tv_cndmask_b32_e64 v2, v2, 0, vcc\n\tv_cndmask_b32_e64 v3, v3, 0, vcc\n\tv_cndmask_b32_e64 v5, v5, 0, vcc\n\tv_cndmask_b32_e64 v7, v7, 0, vcc\n.LBB0_18:\n\t.loc\t1 1526 13                       ; 7.ttgir:1526:13\n\ts_mul_i32 s4, s18, 0x3d640\n\t.loc\t1 1527 13                       ; 7.ttgir:1527:13\n\ts_ashr_i32 s5, s4, 31\n\ts_lshl_b64 s[4:5], s[4:5], 2\n\ts_add_u32 s1, s8, s4\n\t.loc\t1 1528 13                       ; 7.ttgir:1528:13\n\ts_mul_i32 s4, s17, 0x3d64\n\t.loc\t1 1527 13                       ; 7.ttgir:1527:13\n\ts_addc_u32 s6, s9, s5\n\t.loc\t1 1529 13                       ; 7.ttgir:1529:13\n\ts_ashr_i32 s5, s4, 31\n\ts_lshl_b64 s[4:5], s[4:5], 2\n\ts_add_u32 s1, s1, s4\n\ts_addc_u32 s6, s6, s5\n\t.loc\t1 1530 13                       ; 7.ttgir:1530:13\n\ts_ashr_i32 s53, s52, 31\n\ts_lshl_b64 s[4:5], s[52:53], 2\n\ts_add_u32 s4, s1, s4\n\ts_addc_u32 s1, s6, s5\n\t.loc\t1 1531 13                       ; 7.ttgir:1531:13\n\ts_sub_i32 s0, s0, s16\n\t.loc\t1 1532 13                       ; 7.ttgir:1532:13\n\ts_cmp_lt_i32 s0, 1\n\t.loc\t1 103 10                        ; 7.ttgir:103:10\n\tv_and_b32_e32 v11, 0x7f, v0\n\t.loc\t1 1532 13                       ; 7.ttgir:1532:13\n\ts_cselect_b64 s[8:9], -1, 0\n\t.loc\t1 1533 5                        ; 7.ttgir:1533:5\n\ts_and_b64 vcc, exec, s[8:9]\n\tv_lshl_add_u32 v10, v165, 2, 0\n\tv_lshlrev_b32_e32 v0, 2, v11\n\ts_cbranch_vccnz .LBB0_20\n; %bb.19:\n\t.loc\t1 0 5 is_stmt 0                 ; 7.ttgir:0:5\n\ts_mov_b32 s2, 0x800000\n\t.loc\t1 1539 13 is_stmt 1             ; 7.ttgir:1539:13\n\tv_cmp_gt_f32_e32 vcc, s2, v79\n\tv_mov_b32_e32 v12, 0x42000000\n\t.loc\t1 1536 13                       ; 7.ttgir:1536:13\n\ts_sub_i32 s0, 0x80, s0\n\t.loc\t1 1539 13                       ; 7.ttgir:1539:13\n\tv_cndmask_b32_e64 v13, 0, 32, vcc\n\tv_ldexp_f32 v13, v79, v13\n\tv_log_f32_e32 v13, v13\n\tv_cndmask_b32_e32 v12, 0, v12, vcc\n\t.loc\t1 1538 13                       ; 7.ttgir:1538:13\n\tv_cmp_gt_i32_e32 vcc, s0, v11\n\t.loc\t1 1539 13                       ; 7.ttgir:1539:13\n\tv_sub_f32_e32 v11, v13, v12\n\t.loc\t1 1540 13                       ; 7.ttgir:1540:13\n\tv_add_f32_e32 v11, v66, v11\n\t.loc\t1 1541 5                        ; 7.ttgir:1541:5\n\ts_barrier\n\t.loc\t1 1542 13                       ; 7.ttgir:1542:13\n\tds_write_b32 v10, v11\n\tv_add_u32_e32 v11, 0, v0\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\tds_read_b32 v11, v11\n\t.loc\t1 1543 5                        ; 7.ttgir:1543:5\n\tv_bfrev_b32_e32 v12, 1\n\ts_and_b64 vcc, s[82:83], vcc\n\ts_and_b32 s5, s1, 0xffff\n\ts_mov_b32 s7, 0x27000\n\ts_mov_b32 s6, 0x7ffffffe\n\tv_cndmask_b32_e32 v12, v12, v0, vcc\n\ts_mov_b64 s[2:3], 0\n\ts_waitcnt lgkmcnt(0)\n\tbuffer_store_dword v11, v12, s[4:7], 0 offen\n.LBB0_20:                               ; %Flow\n\t.loc\t1 1533 5                        ; 7.ttgir:1533:5\n\ts_andn2_b64 vcc, exec, s[2:3]\n\ts_cbranch_vccnz .LBB0_22\n; %bb.21:\n\t.loc\t1 0 5 is_stmt 0                 ; 7.ttgir:0:5\n\ts_mov_b32 s0, 0x800000\n\t.loc\t1 1547 13 is_stmt 1             ; 7.ttgir:1547:13\n\tv_cmp_gt_f32_e32 vcc, s0, v79\n\tv_mov_b32_e32 v11, 0x42000000\n\ts_nop 0\n\tv_cndmask_b32_e64 v12, 0, 32, vcc\n\tv_ldexp_f32 v12, v79, v12\n\tv_log_f32_e32 v12, v12\n\tv_cndmask_b32_e32 v11, 0, v11, vcc\n\t.loc\t1 1549 5                        ; 7.ttgir:1549:5\n\ts_barrier\n\t.loc\t1 1547 13                       ; 7.ttgir:1547:13\n\tv_sub_f32_e32 v11, v12, v11\n\t.loc\t1 1548 13                       ; 7.ttgir:1548:13\n\tv_add_f32_e32 v11, v66, v11\n\t.loc\t1 1550 13                       ; 7.ttgir:1550:13\n\tds_write_b32 v10, v11\n\tv_add_u32_e32 v10, 0, v0\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n\tds_read_b32 v10, v10\n\t.loc\t1 1551 5                        ; 7.ttgir:1551:5\n\tv_bfrev_b32_e32 v11, 1\n\ts_and_b32 s5, s1, 0xffff\n\ts_mov_b32 s7, 0x27000\n\ts_mov_b32 s6, 0x7ffffffe\n\tv_cndmask_b32_e64 v0, v11, v0, s[82:83]\n\ts_waitcnt lgkmcnt(0)\n\tbuffer_store_dword v10, v0, s[4:7], 0 offen\n.LBB0_22:\n\t.loc\t1 1555 13                       ; 7.ttgir:1555:13\n\ts_mul_i32 s0, s45, s18\n\t.loc\t1 1556 13                       ; 7.ttgir:1556:13\n\ts_ashr_i32 s1, s0, 31\n\ts_lshl_b64 s[0:1], s[0:1], 1\n\ts_add_u32 s2, s10, s0\n\t.loc\t1 1557 13                       ; 7.ttgir:1557:13\n\ts_mul_i32 s0, s46, s17\n\t.loc\t1 1556 13                       ; 7.ttgir:1556:13\n\ts_addc_u32 s3, s11, s1\n\t.loc\t1 1558 13                       ; 7.ttgir:1558:13\n\ts_ashr_i32 s1, s0, 31\n\ts_lshl_b64 s[0:1], s[0:1], 1\n\ts_add_u32 s2, s2, s0\n\t.loc\t1 1559 13                       ; 7.ttgir:1559:13\n\ts_mul_i32 s0, s54, s47\n\t.loc\t1 1558 13                       ; 7.ttgir:1558:13\n\ts_addc_u32 s3, s3, s1\n\t.loc\t1 1560 13                       ; 7.ttgir:1560:13\n\ts_ashr_i32 s1, s0, 31\n\ts_lshl_b64 s[0:1], s[0:1], 1\n\ts_add_u32 s2, s2, s0\n\t.loc\t1 1562 13                       ; 7.ttgir:1562:13\n\ts_mul_i32 s0, s47, s52\n\t.loc\t1 1560 13                       ; 7.ttgir:1560:13\n\ts_addc_u32 s3, s3, s1\n\t.loc\t1 1565 13                       ; 7.ttgir:1565:13\n\ts_ashr_i32 s1, s0, 31\n\ts_lshl_b64 s[0:1], s[0:1], 1\n\ts_add_u32 s0, s2, s0\n\ts_addc_u32 s1, s3, s1\n\t.loc\t1 1651 5                        ; 7.ttgir:1651:5\n\ts_and_b32 s2, s47, 0x3fff\n\t.loc\t1 189 11                        ; 7.ttgir:189:11\n\tv_cmp_gt_i32_e32 vcc, s16, v166\n\t.loc\t1 1564 13                       ; 7.ttgir:1564:13\n\tv_mul_lo_u32 v12, s47, v165\n\t.loc\t1 1651 5                        ; 7.ttgir:1651:5\n\ts_bitset1_b32 s2, 14\n\ts_and_b32 s1, s1, 0xffff\n\ts_lshl_b32 s2, s2, 16\n\ts_mov_b32 s4, 0x5040100\n\t.loc\t1 1615 13                       ; 7.ttgir:1615:13\n\tv_add_lshl_u32 v13, v12, v1, 1\n\t.loc\t1 1651 5                        ; 7.ttgir:1651:5\n\tv_bfrev_b32_e32 v14, 1\n\t.loc\t1 1650 13                       ; 7.ttgir:1650:13\n\ts_or_b64 vcc, s[8:9], vcc\n\t.loc\t1 1651 5                        ; 7.ttgir:1651:5\n\ts_or_b32 s1, s1, s2\n\ts_mov_b32 s3, 0x27000\n\ts_mov_b32 s2, 0x7ffffffe\n\tv_perm_b32 v11, v78, v77, s4\n\tv_perm_b32 v10, v76, v75, s4\n\tv_cndmask_b32_e32 v0, v14, v13, vcc\n\tbuffer_store_dwordx2 v[10:11], v0, s[0:3], 0 offen\n\tv_add_lshl_u32 v10, v12, v156, 1\n\tv_perm_b32 v1, v74, v73, s4\n\tv_perm_b32 v0, v71, v69, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_lshl_u32 v10, v12, v155, 1\n\tv_perm_b32 v1, v72, v70, s4\n\tv_perm_b32 v0, v68, v67, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_lshl_u32 v10, v12, v154, 1\n\tv_perm_b32 v1, v65, v64, s4\n\tv_perm_b32 v0, v62, v60, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 64, v13\n\tv_perm_b32 v1, v63, v61, s4\n\tv_perm_b32 v0, v58, v56, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0x50, v13\n\tv_perm_b32 v1, v59, v57, s4\n\tv_perm_b32 v0, v54, v52, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0x60, v13\n\tv_perm_b32 v1, v55, v53, s4\n\tv_perm_b32 v0, v51, v50, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0x70, v13\n\tv_perm_b32 v1, v49, v48, s4\n\tv_perm_b32 v0, v46, v44, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0x80, v13\n\tv_perm_b32 v1, v47, v45, s4\n\tv_perm_b32 v0, v42, v40, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0x90, v13\n\tv_perm_b32 v1, v43, v41, s4\n\tv_perm_b32 v0, v38, v36, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0xa0, v13\n\tv_perm_b32 v1, v39, v37, s4\n\tv_perm_b32 v0, v35, v34, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0xb0, v13\n\tv_perm_b32 v1, v29, v28, s4\n\tv_perm_b32 v0, v26, v24, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0xc0, v13\n\tv_perm_b32 v1, v27, v25, s4\n\tv_perm_b32 v0, v22, v20, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_add_u32_e32 v10, 0xd0, v13\n\tv_perm_b32 v1, v23, v21, s4\n\tv_perm_b32 v0, v19, v18, s4\n\tv_cndmask_b32_e32 v10, v14, v10, vcc\n\tbuffer_store_dwordx2 v[0:1], v10, s[0:3], 0 offen\n\tv_perm_b32 v0, v6, v4, s4\n\tv_add_u32_e32 v4, 0xe0, v13\n\tv_perm_b32 v1, v9, v8, s4\n\tv_cndmask_b32_e32 v4, v14, v4, vcc\n\tbuffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen\n\tv_perm_b32 v0, v3, v2, s4\n\tv_add_u32_e32 v2, 0xf0, v13\n\tv_perm_b32 v1, v7, v5, s4\n\tv_cndmask_b32_e32 v2, v14, v2, vcc\n\tbuffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen\n\t.loc\t1 1652 5                        ; 7.ttgir:1652:5\n\ts_endpgm\n.Ltmp2:\n\t.section\t.rodata,"a",@progbits\n\t.p2align\t6, 0x0\n\t.amdhsa_kernel attn_fwd\n\t\t.amdhsa_group_segment_fixed_size 0\n\t\t.amdhsa_private_segment_fixed_size 0\n\t\t.amdhsa_kernarg_size 160\n\t\t.amdhsa_user_sgpr_count 16\n\t\t.amdhsa_user_sgpr_dispatch_ptr 0\n\t\t.amdhsa_user_sgpr_queue_ptr 0\n\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1\n\t\t.amdhsa_user_sgpr_dispatch_id 0\n\t\t.amdhsa_user_sgpr_kernarg_preload_length 14\n\t\t.amdhsa_user_sgpr_kernarg_preload_offset 0\n\t\t.amdhsa_user_sgpr_private_segment_size 0\n\t\t.amdhsa_uses_dynamic_stack 0\n\t\t.amdhsa_enable_private_segment 0\n\t\t.amdhsa_system_sgpr_workgroup_id_x 1\n\t\t.amdhsa_system_sgpr_workgroup_id_y 1\n\t\t.amdhsa_system_sgpr_workgroup_id_z 1\n\t\t.amdhsa_system_sgpr_workgroup_info 0\n\t\t.amdhsa_system_vgpr_workitem_id 0\n\t\t.amdhsa_next_free_vgpr 228\n\t\t.amdhsa_next_free_sgpr 100\n\t\t.amdhsa_accum_offset 228\n\t\t.amdhsa_reserve_vcc 1\n\t\t.amdhsa_reserve_xnack_mask 1\n\t\t.amdhsa_float_round_mode_32 0\n\t\t.amdhsa_float_round_mode_16_64 0\n\t\t.amdhsa_float_denorm_mode_32 3\n\t\t.amdhsa_float_denorm_mode_16_64 3\n\t\t.amdhsa_dx10_clamp 1\n\t\t.amdhsa_ieee_mode 1\n\t\t.amdhsa_fp16_overflow 0\n\t\t.amdhsa_tg_split 0\n\t\t.amdhsa_exception_fp_ieee_invalid_op 0\n\t\t.amdhsa_exception_fp_denorm_src 0\n\t\t.amdhsa_exception_fp_ieee_div_zero 0\n\t\t.amdhsa_exception_fp_ieee_overflow 0\n\t\t.amdhsa_exception_fp_ieee_underflow 0\n\t\t.amdhsa_exception_fp_ieee_inexact 0\n\t\t.amdhsa_exception_int_div_zero 0\n\t.end_amdhsa_kernel\n\t.text\n.Lfunc_end0:\n\t.size\tattn_fwd, .Lfunc_end0-attn_fwd\n\t.cfi_endproc\n                                        ; -- End function\n\t.set attn_fwd.num_vgpr, 228\n\t.set attn_fwd.num_agpr, 0\n\t.set attn_fwd.numbered_sgpr, 100\n\t.set attn_fwd.private_seg_size, 0\n\t.set attn_fwd.uses_vcc, 1\n\t.set attn_fwd.uses_flat_scratch, 0\n\t.set attn_fwd.has_dyn_sized_stack, 0\n\t.set attn_fwd.has_recursion, 0\n\t.set attn_fwd.has_indirect_call, 0\n\t.section\t.AMDGPU.csdata,"",@progbits\n; Kernel info:\n; codeLenInByte = 14624\n; TotalNumSgprs: 106\n; NumVgprs: 228\n; NumAgprs: 0\n; TotalNumVgprs: 228\n; ScratchSize: 0\n; MemoryBound: 0\n; FloatMode: 240\n; IeeeMode: 1\n; LDSByteSize: 0 bytes/workgroup (compile time only)\n; SGPRBlocks: 13\n; VGPRBlocks: 28\n; NumSGPRsForWavesPerEU: 106\n; NumVGPRsForWavesPerEU: 228\n; AccumOffset: 228\n; Occupancy: 2\n; WaveLimiterHint : 0\n; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0\n; COMPUTE_PGM_RSRC2:USER_SGPR: 16\n; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0\n; COMPUTE_PGM_RSRC2:TGID_X_EN: 1\n; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1\n; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1\n; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0\n; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 56\n; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0\n\t.text\n\t.p2alignl 6, 3212836864\n\t.fill 256, 4, 3212836864\n\t.section\t.AMDGPU.gpr_maximums,"",@progbits\n\t.set amdgpu.max_num_vgpr, 0\n\t.set amdgpu.max_num_agpr, 0\n\t.set amdgpu.max_num_sgpr, 0\n\t.text\n\t.section\t.debug_abbrev,"",@progbits\n\t.byte\t1                               ; Abbreviation Code\n\t.byte\t17                              ; DW_TAG_compile_unit\n\t.byte\t0                               ; DW_CHILDREN_no\n\t.byte\t37                              ; DW_AT_producer\n\t.byte\t14                              ; DW_FORM_strp\n\t.byte\t19                              ; DW_AT_language\n\t.byte\t5                               ; DW_FORM_data2\n\t.byte\t3                               ; DW_AT_name\n\t.byte\t14                              ; DW_FORM_strp\n\t.byte\t16                              ; DW_AT_stmt_list\n\t.byte\t23                              ; DW_FORM_sec_offset\n\t.byte\t27                              ; DW_AT_comp_dir\n\t.byte\t14                              ; DW_FORM_strp\n\t.byte\t17                              ; DW_AT_low_pc\n\t.byte\t1                               ; DW_FORM_addr\n\t.byte\t18                              ; DW_AT_high_pc\n\t.byte\t6                               ; DW_FORM_data4\n\t.byte\t0                               ; EOM(1)\n\t.byte\t0                               ; EOM(2)\n\t.byte\t0                               ; EOM(3)\n\t.section\t.debug_info,"",@progbits\n.Lcu_begin0:\n\t.long\t.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit\n.Ldebug_info_start0:\n\t.short\t4                               ; DWARF version number\n\t.long\t.debug_abbrev                   ; Offset Into Abbrev. Section\n\t.byte\t8                               ; Address Size (in bytes)\n\t.byte\t1                               ; Abbrev [1] 0xb:0x1f DW_TAG_compile_unit\n\t.long\t.Linfo_string0                  ; DW_AT_producer\n\t.short\t2                               ; DW_AT_language\n\t.long\t.Linfo_string1                  ; DW_AT_name\n\t.long\t.Lline_table_start0             ; DW_AT_stmt_list\n\t.long\t.Linfo_string2                  ; DW_AT_comp_dir\n\t.quad\t.Lfunc_begin0                   ; DW_AT_low_pc\n\t.long\t.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc\n.Ldebug_info_end0:\n\t.section\t.debug_str,"MS",@progbits,1\n.Linfo_string0:\n\t.asciz\t"triton"                        ; string offset=0\n.Linfo_string1:\n\t.asciz\t"7.ttgir"                       ; string offset=7\n.Linfo_string2:\n\t.asciz\t"ttgir"                         ; string offset=15\n\t.section\t".note.GNU-stack","",@progbits\n\t.amdgpu_metadata\n---\namdhsa.kernels:\n  - .agpr_count:     0\n    .args:\n      - .address_space:  global\n        .offset:         0\n        .size:           8\n        .value_kind:     global_buffer\n      - .address_space:  global\n        .offset:         8\n        .size:           8\n        .value_kind:     global_buffer\n      - .address_space:  global\n        .offset:         16\n        .size:           8\n        .value_kind:     global_buffer\n      - .address_space:  global\n        .offset:         24\n        .size:           8\n        .value_kind:     global_buffer\n      - .address_space:  global\n        .offset:         32\n        .size:           8\n        .value_kind:     global_buffer\n      - .offset:         40\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         44\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         48\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         52\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         56\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         60\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         64\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         68\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         72\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         76\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         80\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         84\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         88\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         92\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         96\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         100\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         104\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         108\n        .size:           4\n        .value_kind:     by_value\n      - .address_space:  global\n        .offset:         112\n        .size:           8\n        .value_kind:     global_buffer\n      - .address_space:  global\n        .offset:         120\n        .size:           8\n        .value_kind:     global_buffer\n      - .offset:         128\n        .size:           4\n        .value_kind:     by_value\n      - .offset:         132\n        .size:           4\n        .value_kind:     by_value\n      - .address_space:  global\n        .offset:         136\n        .size:           8\n        .value_kind:     global_buffer\n      - .offset:         144\n        .size:           4\n        .value_kind:     by_value\n      - .address_space:  global\n        .offset:         152\n        .size:           8\n        .value_kind:     global_buffer\n    .group_segment_fixed_size: 0\n    .kernarg_segment_align: 8\n    .kernarg_segment_size: 160\n    .max_flat_workgroup_size: 256\n    .name:           attn_fwd\n    .private_segment_fixed_size: 0\n    .sgpr_count:     106\n    .sgpr_spill_count: 4\n    .symbol:         attn_fwd.kd\n    .uses_dynamic_stack: false\n    .vgpr_count:     228\n    .vgpr_spill_count: 0\n    .wavefront_size: 64\namdhsa.target:   amdgcn-amd-amdhsa--gfx942\namdhsa.version:\n  - 1\n  - 2\n...\n\n\t.end_amdgpu_metadata\n\t.section\t.debug_line,"",@progbits\n.Lline_table_start0:\n'
fused-attention-fwd-d128-layoutthd:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K      triton
0    2.0  16.0  16.0  32768.0  32768.0  311.311445
