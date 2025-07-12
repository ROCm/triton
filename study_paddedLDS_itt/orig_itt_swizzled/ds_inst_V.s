        ;; LWV
        ds_write_b32 v98, v99
	ds_write_b32 v166, v100
	ds_write_b32 v167, v101
	ds_write_b32 v168, v102
	ds_write_b32 v169, v103
	ds_write_b32 v170, v104
	ds_write_b32 v171, v106
	ds_write_b32 v172, v105

        ;; LRV
	ds_read2st64_b64 v[98:101], v210 offset1:8
	ds_read_b64 v[102:103], v180
	ds_read_b64 v[104:105], v179
	ds_read_b64 v[106:107], v187
	ds_read_b64 v[108:109], v188
	ds_read_b64 v[110:111], v186
	ds_read_b64 v[112:113], v194
	ds_read_b64 v[202:203], v195
	ds_read_b64 v[206:207], v196
	ds_read_b64 v[208:209], v193
	ds_read_b64 v[84:85], v173
	ds_read_b64 v[88:89], v174
	ds_read_b64 v[98:99], v175
	ds_read_b64 v[100:101], v176
	ds_read_b64 v[82:83], v181
	ds_read_b64 v[88:89], v182
	ds_read_b64 v[92:93], v183
	ds_read_b64 v[102:103], v184
	ds_read2st64_b64 v[82:85], v210 offset0:16 offset1:24
	ds_read_b64 v[84:85], v189
	ds_read_b64 v[90:91], v190
	ds_read_b64 v[88:89], v191
	ds_read_b64 v[94:95], v192
	ds_read_b64 v[68:69], v177
	ds_read_b64 v[82:83], v178
	ds_read_b64 v[84:85], v197
	ds_read_b64 v[92:93], v198
	ds_read_b64 v[90:91], v199
	ds_read_b64 v[96:97], v200
	ds_read_b64 v[68:69], v185
