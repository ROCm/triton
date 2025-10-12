BB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v235, v200
	s_mov_b32 s31, s17
	s_mov_b32 s17, s3
	v_mov_b32_e32 v246, v195
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_f16 v[66:81], v[66:69], v[142:145], 0
	v_add_f32_e32 v82, v241, v214
	v_add_f32_e32 v82, v82, v222
	v_add_f32_e32 v82, v82, v215
	v_add_f32_e32 v82, v82, v217
	v_add_f32_e32 v82, v82, v216
	v_add_f32_e32 v195, v82, v229
	v_mfma_f32_32x32x16_f16 v[82:97], v[178:181], v[142:145], 0
	v_mul_f32_e64 v50, v50, v194
	v_mul_f32_e64 v51, v51, v194
	v_add_f32_e32 v178, v195, v220
	v_mul_f32_e64 v52, v52, v194
	v_mul_f32_e64 v53, v53, v194
	v_add_f32_e32 v178, v178, v219
	v_pk_mul_f32 v[16:17], v[16:17], v[194:195] op_sel_hi:[1,0]
	v_add_f32_e32 v178, v178, v221
	v_mfma_f32_32x32x16_f16 v[82:97], v[170:173], v[138:141], v[82:97]
	v_add_f32_e32 v178, v178, v224
	v_mul_f32_e64 v14, v14, v194
	v_mul_f32_e64 v15, v15, v194
	v_add_f32_e32 v178, v178, v223
	v_add_f32_e32 v178, v178, v242
	v_pk_mul_f32 v[12:13], v[12:13], v[194:195] op_sel_hi:[1,0]
	v_add_f32_e32 v170, v178, v226
	v_mfma_f32_32x32x16_f16 v[82:97], v[166:169], v[134:137], v[82:97]
	v_add_f32_e32 v170, v170, v243
	v_mul_f32_e64 v8, v8, v194
	v_mul_f32_e64 v9, v9, v194
	v_add_f32_e32 v170, v170, v228
	v_mul_f32_e64 v10, v10, v194
	v_mul_f32_e64 v11, v11, v194
	v_add_f32_e32 v170, v170, v227
	v_pk_mul_f32 v[6:7], v[6:7], v[194:195] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[82:97], v[162:165], v[130:133], v[82:97]
	v_mul_f32_e64 v4, v4, v194
	v_mul_f32_e64 v5, v5, v194
	v_cvt_pk_f16_f32 v165, v229, v220
	v_cvt_pk_f16_f32 v164, v217, v216
	v_cvt_pk_f16_f32 v163, v222, v215
	v_add_f32_e32 v170, v170, v230
	v_add_f32_e32 v170, v170, v225
	v_mfma_f32_32x32x16_f16 v[82:97], v[158:161], v[126:129], v[82:97]
	v_add_f32_e32 v166, v170, v244
	v_cvt_pk_f16_f32 v161, v212, v238
	v_add_f32_e32 v166, v166, v232
	v_cvt_pk_f16_f32 v160, v239, v240
	v_add_f32_e32 v166, v166, v233
	v_cvt_pk_f16_f32 v159, v236, v237
	v_mfma_f32_32x32x16_f16 v[82:97], v[154:157], v[122:125], v[82:97]
	v_add_f32_e32 v166, v166, v234
	v_cvt_pk_f16_f32 v157, v234, v245
	v_add_f32_e32 v166, v166, v245
	v_add_f32_e32 v166, v166, v218
	v_cvt_pk_f16_f32 v156, v232, v233
	v_add_f32_e32 v162, v166, v231
	v_mfma_f32_32x32x16_f16 v[82:97], v[150:153], v[118:121], v[82:97]
	v_cvt_pk_f16_f32 v155, v225, v244
	v_cvt_pk_f16_f32 v153, v243, v228
	v_cvt_pk_f16_f32 v152, v242, v226
	v_cvt_pk_f16_f32 v151, v224, v223
	v_cvt_pk_f16_f32 v150, v219, v221
	v_pk_mul_f32 v[2:3], v[2:3], v[194:195] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[186:189], v[138:141], v[66:81]
	v_add_f32_e32 v162, v162, v236
	v_cvt_pk_f16_f32 v154, v227, v230
	v_add_f32_e32 v162, v162, v237
	v_mul_f32_e64 v32, v32, v194
	v_mul_f32_e64 v33, v33, v194
	v_add_f32_e32 v162, v162, v239
	v_pk_mul_f32 v[28:29], v[28:29], v[194:195] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[182:185], v[134:137], v[66:81]
	v_mul_f32_e64 v30, v30, v194
	v_mul_f32_e64 v31, v31, v194
	v_add_f32_e32 v162, v162, v240
	v_add_f32_e32 v162, v162, v212
	v_mul_f32_e64 v26, v26, v194
	v_mul_f32_e64 v27, v27, v194
	v_add_f32_e32 v158, v162, v238
	v_pk_mul_f32 v[24:25], v[24:25], v[194:195] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[174:177], v[130:133], v[66:81]
	v_mul_f32_e64 v22, v22, v194
	v_mul_f32_e64 v23, v23, v194
	v_mov_b32_e32 v162, v158
	v_mul_f32_e64 v20, v20, v194
	v_mul_f32_e64 v21, v21, v194
	v_permlane32_swap_b32_e32 v158, v162
	v_add_f32_e32 v200, v158, v162
	v_pk_mul_f32 v[48:49], v[48:49], v[194:195] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[110:113], v[126:129], v[66:81]
	v_cvt_pk_f16_f32 v158, v218, v231
	v_cvt_pk_f16_f32 v162, v241, v214
	v_fmac_f32_e32 v200, v235, v194
	v_mul_f32_e64 v18, v18, v194
	v_mul_f32_e64 v19, v19, v194
	v_pk_mul_f32 v[46:47], v[46:47], v[194:195] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[194:195] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[106:109], v[122:125], v[66:81]
	v_mul_f32_e64 v40, v40, v194
	v_mul_f32_e64 v41, v41, v194
	v_mul_f32_e64 v42, v42, v194
	v_mul_f32_e64 v43, v43, v194
	v_mul_f32_e64 v36, v36, v194
	v_mul_f32_e64 v37, v37, v194
	v_pk_mul_f32 v[38:39], v[38:39], v[194:195] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[194:195] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[194:195] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_f16 v[66:81], v[102:105], v[118:121], v[66:81]
	v_mul_f32_e64 v60, v60, v194
	v_mul_f32_e64 v61, v61, v194
	v_mul_f32_e64 v62, v62, v194
	v_mul_f32_e64 v63, v63, v194
	v_mul_f32_e64 v56, v56, v194
	v_mul_f32_e64 v57, v57, v194
	v_pk_mul_f32 v[58:59], v[58:59], v[194:195] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[194:195] op_sel_hi:[1,0]
	; iglp_opt mask(0x0000000A)
	v_mfma_f32_32x32x16_f16 v[66:81], v[98:101], v[114:117], v[66:81]
	v_mfma_f32_32x32x16_f16 v[82:97], v[146:149], v[114:117], v[82:97]
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_add_i32 s3, s5, 1
	s_cmp_lt_i32 s3, 2
	s_cselect_b32 s27, s3, 0
	s_lshl_b32 s25, s27, 13
	s_lshl_b32 s3, s27, 14
	s_add_i32 s26, s3, 0
	s_ashr_i32 s3, s25, 5
	s_add_i32 s24, s26, s3
	s_add_i32 s30, s24, 0x87c0
	v_add_u32_e32 v98, s30, v213
	s_nop 0
	v_readfirstlane_b32 s3, v98
	v_add_u32_e32 v98, s30, v205
	s_and_b32 s13, s1, 0xffff
	s_mov_b32 s12, s0
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v98
	buffer_load_dwordx4 v193, s[12:15], 0 offen lds
	s_mov_b32 m0, s3
	s_nop 0
	buffer_load_dwordx4 v206, s[12:15], 0 offen lds
	v_add_u32_e32 v98, s31, v207
	v_add_u32_e32 v194, v98, v208
	v_add_u32_e32 v195, v98, v209
	ds_read_b64_tr_b16 v[98:99], v194
	ds_read_b64_tr_b16 v[102:103], v194 offset:64
	ds_read_b64_tr_b16 v[106:107], v194 offset:128
	ds_read_b64_tr_b16 v[110:111], v194 offset:192
	ds_read_b64_tr_b16 v[100:101], v195 offset:8192
	ds_read_b64_tr_b16 v[104:105], v195 offset:8256
	ds_read_b64_tr_b16 v[108:109], v195 offset:8320
	ds_read_b64_tr_b16 v[112:113], v195 offset:8384
	ds_read_b64_tr_b16 v[146:147], v194 offset:256
	ds_read_b64_tr_b16 v[166:167], v194 offset:320
	ds_read_b64_tr_b16 v[170:171], v194 offset:384
	ds_read_b64_tr_b16 v[174:175], v194 offset:448
	ds_read_b64_tr_b16 v[148:149], v195 offset:8448
	ds_read_b64_tr_b16 v[168:169], v195 offset:8512
	ds_read_b64_tr_b16 v[172:173], v195 offset:8576
	ds_read_b64_tr_b16 v[176:177], v195 offset:8640
	ds_read_b64_tr_b16 v[178:179], v194 offset:512
	ds_read_b64_tr_b16 v[182:183], v194 offset:576
	ds_read_b64_tr_b16 v[186:187], v194 offset:640
	ds_read_b64_tr_b16 v[220:221], v194 offset:704
	ds_read_b64_tr_b16 v[180:181], v195 offset:8704
	ds_read_b64_tr_b16 v[184:185], v195 offset:8768
	ds_read_b64_tr_b16 v[188:189], v195 offset:8832
	ds_read_b64_tr_b16 v[222:223], v195 offset:8896
	ds_read_b64_tr_b16 v[226:227], v194 offset:768
	ds_read_b64_tr_b16 v[230:231], v194 offset:832
	ds_read_b64_tr_b16 v[236:237], v194 offset:896
	ds_read_b64_tr_b16 v[240:241], v194 offset:960
	ds_read_b64_tr_b16 v[228:229], v195 offset:8960
	ds_read_b64_tr_b16 v[232:233], v195 offset:9024
	ds_read_b64_tr_b16 v[238:239], v195 offset:9088
	ds_read_b64_tr_b16 v[242:243], v195 offset:9152
	; sched_barrier mask(0x00000000)
	s_setprio 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_f16 v[50:65], v[98:101], v[162:165], v[50:65]
	v_maximum3_f32 v98, v66, v67, v67
	v_maximum3_f32 v98, v98, v68, v69
	v_maximum3_f32 v98, v98, v70, v71
	v_maximum3_f32 v98, v98, v72, v73
	v_maximum3_f32 v98, v98, v74, v75
	v_maximum3_f32 v98, v98, v76, v77
	v_mfma_f32_32x32x16_f16 v[34:49], v[102:105], v[162:165], v[34:49]
	v_maximum3_f32 v98, v98, v78, v79
	v_maximum3_f32 v98, v98, v80, v81
	v_maximum3_f32 v98, v98, v82, v83
	v_maximum3_f32 v98, v98, v84, v85
	v_maximum3_f32 v98, v98, v86, v87
	v_maximum3_f32 v98, v98, v88, v89
	v_mfma_f32_32x32x16_f16 v[34:49], v[166:169], v[150:153], v[34:49]
	v_maximum3_f32 v98, v98, v90, v91
	v_maximum3_f32 v98, v98, v92, v93
	v_maximum3_f32 v98, v98, v94, v95
	v_maximum3_f32 v98, v98, v96, v97
	v_mov_b32_e32 v99, v98
	s_nop 1
	v_permlane32_swap_b32_e32 v98, v99
	v_mfma_f32_32x32x16_f16 v[34:49], v[182:185], v[154:157], v[34:49]
	v_maximum3_f32 v195, v246, v98, v99
	v_mul_f32_e32 v235, 0x3e0293ee, v195
	v_fma_f32 v67, v67, s22, -v235
	v_fma_f32 v98, v246, s22, -v235
	v_fma_f32 v69, v69, s22, -v235
	v_fma_f32 v70, v70, s22, -v235
	v_mfma_f32_32x32x16_f16 v[34:49], v[230:233], v[158:161], v[34:49]
	v_fma_f32 v71, v71, s22, -v235
	v_fma_f32 v74, v74, s22, -v235
	v_fma_f32 v76, v76, s22, -v235
	v_fma_f32 v83, v83, s22, -v235
	v_fma_f32 v84, v84, s22, -v235
	v_fma_f32 v85, v85, s22, -v235
	v_mfma_f32_32x32x16_f16 v[18:33], v[106:109], v[162:165], v[18:33]
	v_fma_f32 v86, v86, s22, -v235
	v_fma_f32 v87, v87, s22, -v235
	v_fma_f32 v88, v88, s22, -v235
	v_fma_f32 v89, v89, s22, -v235
	v_fma_f32 v90, v90, s22, -v235
	v_fma_f32 v91, v91, s22, -v235
	v_mfma_f32_32x32x16_f16 v[2:17], v[110:113], v[162:165], v[2:17]
	v_fma_f32 v96, v96, s22, -v235
	v_exp_f32_e32 v217, v70
	v_exp_f32_e32 v212, v96
	v_exp_f32_e32 v244, v85
	v_mfma_f32_32x32x16_f16 v[2:17], v[174:177], v[150:153], v[2:17]
	v_exp_f32_e32 v231, v91
	v_exp_f32_e32 v214, v67
	v_exp_f32_e32 v194, v98
	v_mfma_f32_32x32x16_f16 v[2:17], v[220:223], v[154:157], v[2:17]
	v_fma_f32 v68, v68, s22, -v235
	v_fma_f32 v73, v73, s22, -v235
	v_fma_f32 v75, v75, s22, -v235
	v_fma_f32 v77, v77, s22, -v235
	v_exp_f32_e32 v215, v69
	v_mfma_f32_32x32x16_f16 v[2:17], v[240:243], v[158:161], v[2:17]
	v_fma_f32 v95, v95, s22, -v235
	v_fma_f32 v66, v66, s22, -v235
	v_fma_f32 v78, v78, s22, -v235
	v_fma_f32 v80, v80, s22, -v235
	v_exp_f32_e32 v245, v89
	v_mfma_f32_32x32x16_f16 v[50:65], v[146:149], v[150:153], v[50:65]
	v_exp_f32_e32 v243, v80
	v_exp_f32_e32 v234, v88
	v_exp_f32_e32 v233, v87
	v_mfma_f32_32x32x16_f16 v[50:65], v[178:181], v[154:157], v[50:65]
	v_exp_f32_e32 v232, v86
	v_exp_f32_e32 v218, v90
	v_exp_f32_e32 v225, v84
	v_mfma_f32_32x32x16_f16 v[50:65], v[226:229], v[158:161], v[50:65]
	v_fma_f32 v72, v72, s22, -v235
	v_fma_f32 v82, v82, s22, -v235
	v_fma_f32 v79, v79, s22, -v235
	v_fma_f32 v81, v81, s22, -v235
	v_exp_f32_e32 v224, v76
	v_mfma_f32_32x32x16_f16 v[18:33], v[170:173], v[150:153], v[18:33]
	v_exp_f32_e32 v226, v79
	v_exp_f32_e32 v228, v81
	v_exp_f32_e32 v242, v78
	v_mfma_f32_32x32x16_f16 v[18:33], v[186:189], v[154:157], v[18:33]
	v_exp_f32_e32 v241, v66
	v_exp_f32_e32 v227, v82
	v_exp_f32_e32 v221, v75
	v_mfma_f32_32x32x16_f16 v[18:33], v[236:239], v[158:161], v[18:33]
	v_fma_f32 v97, v97, s22, -v235
	v_fma_f32 v92, v92, s22, -v235
	v_fma_f32 v94, v94, s22, -v235
	v_fma_f32 v93, v93, s22, -v235
	v_exp_f32_e32 v216, v71
	v_exp_f32_e32 v219, v74
	v_exp_f32_e32 v220, v73
	v_exp_f32_e32 v223, v77
	v_exp_f32_e32 v230, v83
	v_exp_f32_e32 v222, v68
	v_exp_f32_e32 v229, v72
	v_exp_f32_e32 v236, v92
	v_exp_f32_e32 v237, v93
	v_exp_f32_e32 v239, v94
	v_exp_f32_e32 v240, v95
	v_exp_f32_e32 v238, v97
	; iglp_opt mask(0x0000000A)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_setprio 1
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s28, s6
	s_addc_u32 s31, s29, s7
	s_lshl_b32 s3, s5, 13
	s_lshl_b32 s5, s5, 14
	s_add_i32 s5, s5, 0
	s_ashr_i32 s3, s3, 3
	s_add_i32 s3, s5, s3
	v_add_u32_e32 v66, s3, v201
	s_and_b32 s13, s31, 0xffff
	v_readfirstlane_b32 s5, v66
	v_add_u32_e32 v66, s3, v198
	s_mov_b32 m0, s5
	v_readfirstlane_b32 s5, v66
	buffer_load_dwordx4 v210, s[12:15], 0 offen lds
	s_mov_b32 m0, s5
	v_add_u32_e32 v70, s4, v199
	buffer_load_dwordx4 v211, s[12:15], 0 offen lds
	ds_read_b128 v[66:69], v70
	ds_read_b128 v[186:189], v70 offset:32
	ds_read_b128 v[182:185], v70 offset:64
	ds_read_b128 v[174:177], v70 offset:96
	ds_read_b128 v[110:113], v70 offset:128
	ds_read_b128 v[106:109], v70 offset:160
	ds_read_b128 v[102:105], v70 offset:192
	ds_read_b128 v[98:101], v70 offset:224
	ds_read_b128 v[178:181], v70 offset:256
	ds_read_b128 v[170:173], v70 offset:288
	ds_read_b128 v[166:169], v70 offset:320
	ds_read_b128 v[162:165], v70 offset:352
	ds_read_b128 v[158:161], v70 offset:384
	ds_read_b128 v[154:157], v70 offset:416
	ds_read_b128 v[150:153], v70 offset:448
	ds_read_b128 v[146:149], v70 offset:480
	; sched_barrier mask(0x00000000)
	s_setprio 0
	s_add_u32 s28, s28, s6
	s_addc_u32 s29, s29, s7
	s_add_u32 s0, s0, s34
	s_addc_u32 s1, s1, s35
	s_add_i32 s2, s2, 64
	s_cmpk_lt_u32 s2, 0x3f00
	s_mov_b32 s4, s30
	s_mov_b32 s5, s27
	s_waitcnt lgkmcnt(0)
	s_cbranch_scc1 .LBB0_3
