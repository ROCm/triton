        ds_write_b32 v164, v98
	ds_write_b32 v164, v99 offset:1056
	ds_write_b32 v164, v104 offset:2112
	ds_write_b32 v164, v105 offset:3168
	ds_write_b32 v164, v106 offset:4232
	ds_write_b32 v164, v107 offset:5288
	ds_write_b32 v164, v108 offset:6344
	ds_write_b32 v164, v109 offset:7400

	ds_read2_b64 v[98:101], v165 offset1:2
	ds_read2_b64 v[104:107], v165 offset0:33 offset1:35
	ds_read2_b64 v[108:111], v165 offset0:66 offset1:68
	ds_read2_b64 v[168:171], v165 offset0:99 offset1:101

	ds_read2_b64 v[84:87], v165 offset0:4 offset1:6
	ds_read2_b64 v[98:101], v165 offset0:70 offset1:72
	ds_read2_b64 v[104:107], v165 offset0:103 offset1:105
	ds_read2_b64 v[88:91], v165 offset0:37 offset1:39
	ds_read2_b64 v[92:95], v165 offset0:107 offset1:109
	ds_read2_b64 v[84:87], v165 offset0:8 offset1:10
	ds_read2_b64 v[88:91], v165 offset0:74 offset1:76
	ds_read2_b64 v[66:69], v165 offset0:41 offset1:43
	ds_read2_b64 v[66:69], v165 offset0:12 offset1:14
	ds_read2_b64 v[70:73], v165 offset0:45 offset1:47
	ds_read2_b64 v[74:77], v165 offset0:78 offset1:80
	ds_read2_b64 v[84:87], v165 offset0:111 offset1:113
