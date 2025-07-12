	s_barrier
	v_mfma_f32_32x32x8_f16 v[66:81], v[98:99], v[138:139], v[66:81]
	v_add_u32_e32 v98, 0, v157
	s_waitcnt vmcnt(0)
	v_perm_b32 v99, v102, v106, s22
	v_mfma_f32_32x32x8_f16 v[66:81], v[100:101], v[140:141], v[66:81]
	v_perm_b32 v100, v102, v106, s23
	v_perm_b32 v101, v103, v107, s22
	v_perm_b32 v102, v103, v107, s23
	v_perm_b32 v103, v104, v108, s22
	v_perm_b32 v104, v104, v108, s23
	v_perm_b32 v106, v105, v109, s22
	v_perm_b32 v105, v105, v109, s23
	v_mfma_f32_32x32x8_f16 v[82:97], v[110:111], v[142:143], v[82:97]
	ds_write_b32 v98, v99
	ds_write_b32 v166, v100
	ds_write_b32 v167, v101
	ds_write_b32 v168, v102
	ds_write_b32 v169, v103
	ds_write_b32 v170, v104
	ds_write_b32 v171, v106
	ds_write_b32 v172, v105
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[98:101], v210 offset1:8
	ds_read_b64 v[102:103], v180
	ds_read_b64 v[104:105], v179
	v_mfma_f32_32x32x8_f16 v[66:81], v[202:203], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_f16 v[82:97], v[112:113], v[144:145], v[82:97]
	ds_read_b64 v[106:107], v187
	ds_read_b64 v[108:109], v188
	ds_read_b64 v[110:111], v186
	ds_read_b64 v[112:113], v194
	ds_read_b64 v[202:203], v195
	ds_read_b64 v[206:207], v196
	ds_read_b64 v[208:209], v193
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_nop 3
	v_max_f32_e32 v147, v83, v83
	v_mfma_f32_32x32x8_f16 v[66:81], v[204:205], v[144:145], v[66:81]
	v_max_f32_e32 v204, v82, v82
	v_max_f32_e32 v147, v204, v147
	v_max3_f32 v147, v147, v84, v85
	v_max3_f32 v147, v147, v86, v87
	v_max3_f32 v147, v147, v88, v89
	v_max3_f32 v147, v147, v90, v91
	v_max3_f32 v147, v147, v92, v93
	v_max3_f32 v147, v147, v94, v95
	v_max3_f32 v147, v147, v96, v97
	s_nop 1
	v_max3_f32 v147, v147, v66, v67
	v_max3_f32 v147, v147, v68, v69
	v_max3_f32 v147, v147, v70, v71
	v_max3_f32 v147, v147, v72, v73
	v_max3_f32 v147, v147, v74, v75
	v_max3_f32 v147, v147, v76, v77
	v_max3_f32 v147, v147, v78, v79
	v_max3_f32 v147, v147, v80, v81
	ds_bpermute_b32 v204, v154, v147
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v147, v146, v147, v204
	v_pk_mul_f32 v[204:205], v[146:147], s[12:13] op_sel_hi:[1,0]
	s_add_u32 s13, s13, s6
	v_fma_f32 v82, v82, s12, -v205
	v_fma_f32 v83, v83, s12, -v205
	v_fma_f32 v84, v84, s12, -v205
	v_fma_f32 v85, v85, s12, -v205
	v_fma_f32 v146, v86, s12, -v205
	v_sub_f32_e32 v86, v204, v205
	v_exp_f32_e32 v204, v82
	v_exp_f32_e32 v211, v83
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v212, v84
	v_exp_f32_e32 v213, v85
	v_fma_f32 v87, v87, s12, -v205
	v_pk_mul_f32 v[18:19], v[18:19], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[86:87] op_sel_hi:[1,0]
	v_cvt_pkrtz_f16_f32 v82, v204, v211
	v_cvt_pkrtz_f16_f32 v83, v212, v213
	v_pk_mul_f32 v[2:3], v[2:3], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[86:87] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[18:33], v[102:103], v[82:83], v[18:33]
	v_pk_mul_f32 v[6:7], v[6:7], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[86:87] op_sel_hi:[1,0]
	v_fma_f32 v88, v88, s12, -v205
	v_fma_f32 v89, v89, s12, -v205
	v_pk_mul_f32 v[34:35], v[34:35], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[86:87] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[2:17], v[98:99], v[82:83], v[2:17]
	v_exp_f32_e32 v146, v146
	v_pk_mul_f32 v[50:51], v[50:51], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[86:87] op_sel_hi:[1,0]
	v_mfma_f32_32x32x8_f16 v[34:49], v[106:107], v[82:83], v[34:49]
	v_pk_mul_f32 v[62:63], v[62:63], v[86:87] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[86:87] op_sel_hi:[1,0]
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v106, v88
	v_exp_f32_e32 v107, v89
	v_fma_f32 v94, v94, s12, -v205
	v_fma_f32 v95, v95, s12, -v205
	v_mfma_f32_32x32x8_f16 v[50:65], v[112:113], v[82:83], v[50:65]
	v_cvt_pkrtz_f16_f32 v82, v146, v87
	v_cvt_pkrtz_f16_f32 v83, v106, v107
	v_fma_f32 v96, v96, s12, -v205
	v_fma_f32 v97, v97, s12, -v205
	v_fma_f32 v66, v66, s12, -v205
	v_fma_f32 v67, v67, s12, -v205
	v_fma_f32 v68, v68, s12, -v205
	v_mfma_f32_32x32x8_f16 v[18:33], v[100:101], v[82:83], v[18:33]
	ds_read_b64 v[84:85], v173
	ds_read_b64 v[88:89], v174
	ds_read_b64 v[98:99], v175
	ds_read_b64 v[100:101], v176
	v_fma_f32 v69, v69, s12, -v205
	v_exp_f32_e32 v214, v69
	v_fma_f32 v70, v70, s12, -v205
	v_fma_f32 v71, v71, s12, -v205
	v_fma_f32 v72, v72, s12, -v205
	v_fma_f32 v73, v73, s12, -v205
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[2:17], v[84:85], v[82:83], v[2:17]
	v_fma_f32 v84, v90, s12, -v205
	v_fma_f32 v85, v91, s12, -v205
	v_fma_f32 v90, v92, s12, -v205
	v_fma_f32 v91, v93, s12, -v205
	v_exp_f32_e32 v112, v90
	v_exp_f32_e32 v113, v91
	v_exp_f32_e32 v70, v70
	v_mfma_f32_32x32x8_f16 v[34:49], v[108:109], v[82:83], v[34:49]
	v_exp_f32_e32 v108, v84
	v_exp_f32_e32 v109, v85
	v_cvt_pkrtz_f16_f32 v91, v112, v113
	v_exp_f32_e32 v71, v71
	v_exp_f32_e32 v72, v72
	v_cvt_pkrtz_f16_f32 v90, v108, v109
	v_exp_f32_e32 v73, v73
	v_mfma_f32_32x32x8_f16 v[50:65], v[202:203], v[82:83], v[50:65]
	v_exp_f32_e32 v202, v94
	v_exp_f32_e32 v203, v95
	s_addc_u32 s15, s15, s7
	s_add_i32 s14, s14, 64
	s_cmpk_lt_u32 s14, 0x1fc0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 v[2:17], v[88:89], v[90:91], v[2:17]
	ds_read_b64 v[82:83], v181
	ds_read_b64 v[88:89], v182
	ds_read_b64 v[92:93], v183
	ds_read_b64 v[102:103], v184
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[18:33], v[82:83], v[90:91], v[18:33]
	ds_read2st64_b64 v[82:85], v210 offset0:16 offset1:24
	v_exp_f32_e32 v210, v68
	v_mfma_f32_32x32x8_f16 v[50:65], v[206:207], v[90:91], v[50:65]
	v_exp_f32_e32 v206, v96
	v_exp_f32_e32 v207, v97
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[34:49], v[82:83], v[90:91], v[34:49]
	v_cvt_pkrtz_f16_f32 v82, v202, v203
	v_cvt_pkrtz_f16_f32 v83, v206, v207
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[50:65], v[84:85], v[82:83], v[50:65]
	ds_read_b64 v[84:85], v189
	ds_read_b64 v[90:91], v190
	v_mfma_f32_32x32x8_f16 v[18:33], v[88:89], v[82:83], v[18:33]
	ds_read_b64 v[88:89], v191
	ds_read_b64 v[94:95], v192
	v_mfma_f32_32x32x8_f16 v[2:17], v[98:99], v[82:83], v[2:17]
	v_exp_f32_e32 v98, v66
	v_exp_f32_e32 v99, v67
	v_cvt_pkrtz_f16_f32 v67, v210, v214
	v_cvt_pkrtz_f16_f32 v66, v98, v99
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[34:49], v[84:85], v[82:83], v[34:49]
	v_mfma_f32_32x32x8_f16 v[18:33], v[92:93], v[66:67], v[18:33]
	ds_read_b64 v[68:69], v177
	ds_read_b64 v[82:83], v178
	ds_read_b64 v[84:85], v197
	ds_read_b64 v[92:93], v198
	v_mfma_f32_32x32x8_f16 v[2:17], v[100:101], v[66:67], v[2:17]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 v[34:49], v[90:91], v[66:67], v[34:49]
	ds_read_b64 v[90:91], v199
	ds_read_b64 v[96:97], v200
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[50:65], v[84:85], v[66:67], v[50:65]
	v_cvt_pkrtz_f16_f32 v66, v70, v71
	v_cvt_pkrtz_f16_f32 v67, v72, v73
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[2:17], v[68:69], v[66:67], v[2:17]
	v_fma_f32 v68, v74, s12, -v205
	v_fma_f32 v69, v75, s12, -v205
	v_fma_f32 v74, v76, s12, -v205
	v_fma_f32 v75, v77, s12, -v205
	v_exp_f32_e32 v76, v68
	v_exp_f32_e32 v77, v69
	ds_read_b64 v[68:69], v185
	v_mfma_f32_32x32x8_f16 v[18:33], v[102:103], v[66:67], v[18:33]
	v_exp_f32_e32 v74, v74
	v_exp_f32_e32 v75, v75
	v_mfma_f32_32x32x8_f16 v[34:49], v[88:89], v[66:67], v[34:49]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 v[50:65], v[92:93], v[66:67], v[50:65]
	v_cvt_pkrtz_f16_f32 v66, v76, v77
	v_cvt_pkrtz_f16_f32 v67, v74, v75
	s_nop 1
	v_mfma_f32_32x32x8_f16 v[2:17], v[82:83], v[66:67], v[2:17]
	v_add_f32_e32 v83, v204, v211
	v_add_f32_e32 v83, v212, v83
	v_add_f32_e32 v83, v213, v83
	v_mov_b32_e32 v82, v201
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 v[18:33], v[68:69], v[66:67], v[18:33]
	v_fma_f32 v68, v78, s12, -v205
	v_fma_f32 v69, v79, s12, -v205
	v_fma_f32 v78, v80, s12, -v205
	v_exp_f32_e32 v80, v68
	v_fma_f32 v79, v81, s12, -v205
	v_exp_f32_e32 v81, v69
	v_exp_f32_e32 v78, v78
	v_mfma_f32_32x32x8_f16 v[34:49], v[94:95], v[66:67], v[34:49]
	v_exp_f32_e32 v79, v79
	v_cvt_pkrtz_f16_f32 v68, v80, v81
	v_cvt_pkrtz_f16_f32 v69, v78, v79
	v_mfma_f32_32x32x8_f16 v[50:65], v[90:91], v[66:67], v[50:65]
	v_add_f32_e32 v66, v146, v83
	v_add_f32_e32 v66, v87, v66
	v_add_f32_e32 v66, v106, v66
	v_add_f32_e32 v66, v107, v66
	v_add_f32_e32 v66, v108, v66
	v_add_f32_e32 v66, v109, v66
	v_add_f32_e32 v66, v112, v66
	v_add_f32_e32 v66, v113, v66
	v_add_f32_e32 v66, v202, v66
	v_add_f32_e32 v66, v203, v66
	v_add_f32_e32 v66, v206, v66
	v_add_f32_e32 v66, v207, v66
	v_add_f32_e32 v66, v98, v66
	v_add_f32_e32 v66, v99, v66
	v_add_f32_e32 v66, v210, v66
	v_add_f32_e32 v66, v214, v66
	v_add_f32_e32 v66, v70, v66
	v_add_f32_e32 v66, v71, v66
	v_add_f32_e32 v66, v72, v66
	v_add_f32_e32 v66, v73, v66
	v_add_f32_e32 v66, v76, v66
	v_add_f32_e32 v66, v77, v66
	v_add_f32_e32 v66, v74, v66
	v_add_f32_e32 v66, v75, v66
	v_add_f32_e32 v66, v80, v66
	v_add_f32_e32 v66, v81, v66
	v_add_f32_e32 v66, v78, v66
	v_mfma_f32_32x32x8_f16 v[2:17], v[104:105], v[68:69], v[2:17]
	v_add_f32_e32 v66, v79, v66
	ds_bpermute_b32 v67, v154, v66
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v201, v66, v67
	v_mfma_f32_32x32x8_f16 v[18:33], v[110:111], v[68:69], v[18:33]
	v_fmac_f32_e32 v201, v82, v86
	v_mfma_f32_32x32x8_f16 v[34:49], v[208:209], v[68:69], v[34:49]
	v_mfma_f32_32x32x8_f16 v[50:65], v[96:97], v[68:69], v[50:65]
