# FAv3 without AsyncCopy on gfx942 (MI300)

## impl0 -- Redesign the pipeline and cluser according to [ticket#902](https://github.com/ROCm/triton-internal/issues/902)

- triton compiler: https://github.com/ROCm/triton/tree/FAv3_gfx942_noAsyncCopy @e63020d7f07
- FA kernel: https://github.com/ROCm/triton/blob/FAv3_gfx942_noAsyncCopy/fa/flash-attention.py

Command to run the kernel
```bash
python fa/flash-attention.py
```

Dumped IR
- original run: `third_party/amd/FAv3_note/impl0/`
- annotated ttgir with `sched.barrier`: `third_party/amd/FAv3_note/impl0/attn_fwd.ttgir.annot`
- isa dump from the annotated ttgir: `third_party/amd/FAv3_note/impl0_annot/`

## impl1 -- set kpack = 2


- triton compiler: https://github.com/ROCm/triton/tree/FAv3_gfx942_noAsyncCopy @505f03044798
- FA kernel: https://github.com/ROCm/triton/blob/FAv3_gfx942_noAsyncCopy/fa/flash-attention.py

Dumped IR
- original run: `third_party/amd/FAv3_note/impl1/`

Instruction analysis is kept in the "4-stage on MI300" tab in the [project excel](https://amdcloud-my.sharepoint.com/:x:/r/personal/lixzhang_amd_com/Documents/8-wave_FAv3_project.xlsx?d=wf597bd4f12b048d1a5f7839e263e965e&csf=1&web=1&e=C2eGj1) with label `impl1`.


