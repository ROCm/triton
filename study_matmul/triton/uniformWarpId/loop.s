LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s26, s13
	s_mov_b32 s13, s12
	s_mov_b32 s12, s17
	s_mov_b32 s17, s11
	s_add_i32 s11, s30, 1
	s_cmp_lt_i32 s11, 3
	v_add_u32_e32 v141, s12, v135
	s_cselect_b32 s30, s11, 0
	s_add_i32 s26, s26, s25
	v_add3_u32 v141, v141, v134, v136
	v_add_u32_e32 v142, s26, v133
	s_lshl_b32 s11, s30, 13
	ds_read_b128 v[144:147], v141
	ds_read_b128 v[148:151], v141 offset:256
	ds_read_b128 v[152:155], v141 offset:512
	ds_read_b128 v[156:159], v141 offset:768
	v_add3_u32 v141, v142, v135, v136
	ds_read_b128 v[160:163], v141
	ds_read_b128 v[164:167], v141 offset:128
	ds_read_b128 v[168:171], v141 offset:256
	ds_read_b128 v[172:175], v141 offset:384
	ds_read_b128 v[176:179], v141 offset:512
	ds_read_b128 v[180:183], v141 offset:640
	ds_read_b128 v[184:187], v141 offset:768
	ds_read_b128 v[188:191], v141 offset:896
	s_ashr_i32 s12, s11, 5
	s_add_i32 s12, s12, s11
	s_lshl1_add_u32 s12, s12, 0
	s_add_i32 s11, s12, 0xc5e0
	; sched_barrier mask(0x00000000)
	s_add_i32 s26, s12, s19
	s_mov_b32 m0, s26
	s_add_i32 s27, s12, s24
	buffer_load_dwordx4 v140, s[20:23], 0 offen lds
	s_add_i32 m0, s27, 0x2000
	s_nop 0
	buffer_load_dwordx4 v139, s[20:23], 0 offen lds
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x32_bf16 v[124:127], v[144:147], v[160:163], v[124:127]
	s_add_i32 m0, s26, 0xc5e0
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_nop 0
	buffer_load_dwordx4 v138, s[4:7], 0 offen lds
	s_add_i32 m0, s27, 0xe5e0
	v_mfma_f32_16x16x32_bf16 v[120:123], v[148:151], v[160:163], v[120:123]
	buffer_load_dwordx4 v137, s[4:7], 0 offen lds
	; sched_group_barrier mask(0x00000004) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(3) SyncID(0)
	v_mfma_f32_16x16x32_bf16 v[116:119], v[152:155], v[160:163], v[116:119]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_f32_16x16x32_bf16 v[112:115], v[156:159], v[160:163], v[112:115]
	v_mfma_f32_16x16x32_bf16 v[108:111], v[144:147], v[164:167], v[108:111]
	v_mfma_f32_16x16x32_bf16 v[104:107], v[148:151], v[164:167], v[104:107]
	v_mfma_f32_16x16x32_bf16 v[100:103], v[152:155], v[164:167], v[100:103]
	v_mfma_f32_16x16x32_bf16 v[96:99], v[156:159], v[164:167], v[96:99]
	v_mfma_f32_16x16x32_bf16 v[92:95], v[144:147], v[168:171], v[92:95]
	v_mfma_f32_16x16x32_bf16 v[88:91], v[148:151], v[168:171], v[88:91]
	v_mfma_f32_16x16x32_bf16 v[84:87], v[152:155], v[168:171], v[84:87]
	v_mfma_f32_16x16x32_bf16 v[80:83], v[156:159], v[168:171], v[80:83]
	v_mfma_f32_16x16x32_bf16 v[76:79], v[144:147], v[172:175], v[76:79]
	v_mfma_f32_16x16x32_bf16 v[72:75], v[148:151], v[172:175], v[72:75]
	v_mfma_f32_16x16x32_bf16 v[68:71], v[152:155], v[172:175], v[68:71]
	v_mfma_f32_16x16x32_bf16 v[64:67], v[156:159], v[172:175], v[64:67]
	v_mfma_f32_16x16x32_bf16 v[60:63], v[144:147], v[176:179], v[60:63]
	v_mfma_f32_16x16x32_bf16 v[56:59], v[148:151], v[176:179], v[56:59]
	v_mfma_f32_16x16x32_bf16 v[52:55], v[152:155], v[176:179], v[52:55]
	v_mfma_f32_16x16x32_bf16 v[48:51], v[156:159], v[176:179], v[48:51]
	v_mfma_f32_16x16x32_bf16 v[44:47], v[144:147], v[180:183], v[44:47]
	v_mfma_f32_16x16x32_bf16 v[40:43], v[148:151], v[180:183], v[40:43]
	v_mfma_f32_16x16x32_bf16 v[36:39], v[152:155], v[180:183], v[36:39]
	v_mfma_f32_16x16x32_bf16 v[32:35], v[156:159], v[180:183], v[32:35]
	v_mfma_f32_16x16x32_bf16 v[28:31], v[144:147], v[184:187], v[28:31]
	v_mfma_f32_16x16x32_bf16 v[24:27], v[148:151], v[184:187], v[24:27]
	v_mfma_f32_16x16x32_bf16 v[20:23], v[152:155], v[184:187], v[20:23]
	v_mfma_f32_16x16x32_bf16 v[16:19], v[156:159], v[184:187], v[16:19]
	v_mfma_f32_16x16x32_bf16 v[12:15], v[144:147], v[188:191], v[12:15]
	v_mfma_f32_16x16x32_bf16 v[8:11], v[148:151], v[188:191], v[8:11]
	v_mfma_f32_16x16x32_bf16 v[4:7], v[152:155], v[188:191], v[4:7]
	v_mfma_f32_16x16x32_bf16 v[0:3], v[156:159], v[188:191], v[0:3]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s29, s29, -1
	v_add_u32_e32 v137, 64, v137
	v_add_u32_e32 v138, 64, v138
	v_add_u32_e32 v139, 64, v139
	s_cmp_lg_u32 s29, 0
	v_add_u32_e32 v140, 64, v140
	s_cbranch_scc1 .LBB0_5
