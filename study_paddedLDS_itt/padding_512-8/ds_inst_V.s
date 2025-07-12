        ds_write2_b32 v172, v102, v98 offset1:32
	ds_write2_b32 v172, v103, v99 offset0:64 offset1:96
	ds_write2_b32 v166, v106, v100 offset0:128 offset1:160
	ds_write2_b32 v166, v107, v101 offset0:192 offset1:224

	ds_read2_b64 v[100:103], v167 offset1:2
	ds_read2_b64 v[104:107], v176 offset1:2
	ds_read2_b64 v[108:111], v177 offset1:2
	ds_read2_b64 v[172:175], v178 offset1:2
	ds_read2_b64 v[84:87], v167 offset0:4 offset1:6
	ds_read2_b64 v[100:103], v177 offset0:4 offset1:6
	ds_read2_b64 v[104:107], v178 offset0:4 offset1:6
	ds_read2_b64 v[88:91], v176 offset0:4 offset1:6
	ds_read2_b64 v[92:95], v178 offset0:8 offset1:10
	ds_read2_b64 v[84:87], v167 offset0:8 offset1:10
	ds_read2_b64 v[88:91], v177 offset0:8 offset1:10
	ds_read2_b64 v[66:69], v176 offset0:8 offset1:10
	ds_read2_b64 v[66:69], v167 offset0:12 offset1:14
	ds_read2_b64 v[70:73], v176 offset0:12 offset1:14
	ds_read2_b64 v[74:77], v177 offset0:12 offset1:14
	ds_read2_b64 v[84:87], v178 offset0:12 offset1:14
