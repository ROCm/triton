LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s19, s13
	s_mov_b32 s13, s12
	s_mov_b32 s12, s14
	s_mov_b32 s14, s11
	s_add_i32 s11, s17, 1
	s_cmp_lt_i32 s11, 3
	v_add3_u32 v145, s19, v137, v0
	v_add_u32_e32 v146, s12, v136
	s_cselect_b32 s17, s11, 0
	v_add3_u32 v145, v145, v136, v138
	v_add3_u32 v146, v146, v1, v138
	s_lshl_b32 s11, s17, 13
	ds_read_b128 v[148:151], v145
	ds_read_b128 v[152:155], v145 offset:128
	ds_read_b128 v[156:159], v145 offset:256
	ds_read_b128 v[160:163], v145 offset:384
	ds_read_b128 v[164:167], v145 offset:512
	ds_read_b128 v[168:171], v145 offset:640
	ds_read_b128 v[172:175], v145 offset:768
	ds_read_b128 v[176:179], v145 offset:896
	ds_read_b128 v[180:183], v146
	ds_read_b128 v[184:187], v146 offset:256
	ds_read_b128 v[188:191], v146 offset:512
	ds_read_b128 v[192:195], v146 offset:768
	s_ashr_i32 s12, s11, 5
	s_add_i32 s12, s12, s11
	s_lshl1_add_u32 s12, s12, 0
	s_add_i32 s11, s12, 0xc5e0
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v145, s12, v139
	v_add_u32_e32 v146, s12, v140
	v_readfirstlane_b32 s19, v145
	v_add_u32_e32 v147, 0x2000, v146
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s19, v147
	buffer_load_dwordx4 v144, s[20:23], 0 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v143, s[20:23], 0 offen lds
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(2) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v145, 0xc5e0, v145
	v_mfma_f32_16x16x32_bf16 v[126:129], v[180:183], v[148:151], v[126:129]
	v_readfirstlane_b32 s19, v145
	v_add_u32_e32 v145, 0xe5e0, v146
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s19, v145
	buffer_load_dwordx4 v142, s[4:7], 0 offen lds
	s_mov_b32 m0, s19
	v_mfma_f32_16x16x32_bf16 v[122:125], v[184:187], v[148:151], v[122:125]
	buffer_load_dwordx4 v141, s[4:7], 0 offen lds
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(3) SyncID(0)
	v_mfma_f32_16x16x32_bf16 v[118:121], v[188:191], v[148:151], v[118:121]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[192:195], v[148:151], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[110:113], v[180:183], v[152:155], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[106:109], v[184:187], v[152:155], v[106:109]
	v_mfma_f32_16x16x32_bf16 v[102:105], v[188:191], v[152:155], v[102:105]
	v_mfma_f32_16x16x32_bf16 v[98:101], v[192:195], v[152:155], v[98:101]
	v_mfma_f32_16x16x32_bf16 v[94:97], v[180:183], v[156:159], v[94:97]
	v_mfma_f32_16x16x32_bf16 v[90:93], v[184:187], v[156:159], v[90:93]
	v_mfma_f32_16x16x32_bf16 v[86:89], v[188:191], v[156:159], v[86:89]
	v_mfma_f32_16x16x32_bf16 v[82:85], v[192:195], v[156:159], v[82:85]
	v_mfma_f32_16x16x32_bf16 v[78:81], v[180:183], v[160:163], v[78:81]
	v_mfma_f32_16x16x32_bf16 v[74:77], v[184:187], v[160:163], v[74:77]
	v_mfma_f32_16x16x32_bf16 v[70:73], v[188:191], v[160:163], v[70:73]
	v_mfma_f32_16x16x32_bf16 v[66:69], v[192:195], v[160:163], v[66:69]
	v_mfma_f32_16x16x32_bf16 v[62:65], v[180:183], v[164:167], v[62:65]
	v_mfma_f32_16x16x32_bf16 v[58:61], v[184:187], v[164:167], v[58:61]
	v_mfma_f32_16x16x32_bf16 v[54:57], v[188:191], v[164:167], v[54:57]
	v_mfma_f32_16x16x32_bf16 v[50:53], v[192:195], v[164:167], v[50:53]
	v_mfma_f32_16x16x32_bf16 v[46:49], v[180:183], v[168:171], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[42:45], v[184:187], v[168:171], v[42:45]
	v_mfma_f32_16x16x32_bf16 v[38:41], v[188:191], v[168:171], v[38:41]
	v_mfma_f32_16x16x32_bf16 v[34:37], v[192:195], v[168:171], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[30:33], v[180:183], v[172:175], v[30:33]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[184:187], v[172:175], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[188:191], v[172:175], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[18:21], v[192:195], v[172:175], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[180:183], v[176:179], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[184:187], v[176:179], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[6:9], v[188:191], v[176:179], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[2:5], v[192:195], v[176:179], v[2:5]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s15, s15, -1
	v_add_u32_e32 v141, 64, v141
	v_add_u32_e32 v142, 64, v142
	v_add_u32_e32 v143, 64, v143
	s_cmp_lg_u32 s15, 0
	v_add_u32_e32 v144, 64, v144
	s_cbranch_scc1 .LBB0_5
