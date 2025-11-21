## Visualize TTrace

### Option 1: Use Thread Trace Viewer (TTV)

- Source code: https://github.amd.com/Shader-analysis/TTV
- Internal Prebuilt: \\amd.com\svdc\dxgametraces\TOOLS\ThreadTraceViewer\Latest

Generate a ".mon" file from AM. For option 2, you also need the ".sp3" file.

Example from mxfp fa kernel:

- CAP file: /proj/triton_regr/TRITON/MXFP_FA/11182025/roc_capture_mxfp_fa_gfx1250_e4m3_e4m3_b1_q8192_k8192_qh2_kh2_h128_bm128_bn128_blk_pre_kwidth8_39fd50a.cap
- MON file: /proj/triton_regr/TRITON/MXFP_FA/11182025/roc_capture_mxfp_fa_gfx1250_e4m3_e4m3_b1_q8192_k8192_qh2_kh2_h128_bm128_bn128_blk_pre_kwidth8_39fd50a.mon
- SP3 file: /proj/triton_regr/TRITON/MXFP_FA/11182025/roc_capture_mxfp_fa_gfx1250_e4m3_e4m3_b1_q8192_k8192_qh2_kh2_h128_bm128_bn128_blk_pre_kwidth8_39fd50a.out/shader_0.sp3

### Option 2: Use ROCprof Computer Viewer (RCV)

- Source code: https://github.com/ROCm/rocprof-compute-viewer
- Public prebuilt: https://github.com/ROCm/rocprof-compute-viewer/releases
- Internal prebuilt: https://compute-artifactory.amd.com/artifactory/rocm-generic-local/rocprofiler-packages/att-viewer/

Convert .mon file for RCV:

```sh
# 1) Download:
#    - amconvert.py: \\amd.com\atl\gfxip_sde_users01\msinger\rcv\amconvert.py
#    - amtool: \\amd.com\atl\gfxip_sde_users01\msinger\rcv\amtool
# 2) Rename your .mon file to se0_***.mon like se0_ttrace.mon
# 3) Convert .mon to an intermediate file 0_0_shader_engine_0_0.att
python3 amconvert.py se0_ttrace.mon
# 4) Output RCV files from the intermediate file and sp3 file
./amtool ./out 0_0_shader_engine_0_0.att shader_0.sp3
# 5) Open ./out from RCV
```
