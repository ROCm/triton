# FAv3 without AsyncCopy on gfx942 (MI300)

## impl0 -- Redesign the pipeline and cluser according to [ticket#902](https://github.com/ROCm/triton-internal/issues/902)

- triton compiler: https://github.com/ROCm/triton/tree/FAv3_gfx942_noAsyncCopy @e63020d7f07
- FA kernel: https://github.com/ROCm/triton/blob/FAv3_gfx942_noAsyncCopy/fa/flash-attention.py

Command to run the kernel
```bash
python fa/flash-attention.py
```

Dumped IR: `third_party/amd/FAv3_note/impl0/`


