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


```bash
ara
impl0
impl0_annot
impl1
impl1_BN32
impl1_BN32_annot
impl1_TritonInterleaveAndRematRebase0
impl1_TritonInterleaveAndRematRebase0_scalarized
impl1_TritonInterleaveAndRematRebase0_scalarized_BN32
impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpilogue
impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue
impl1_annot
impl1_gfx950
impl1_gfx950_annot
impl1_rtz
impl1_rtz_annot
impl1_scalerized
```

Base on impl1, we have the following experiments
- `impl1_rtz_annot`: use RTZ when converting f32 to f16. 
  This leads to `v_cvt_pkrtz_f16_f32`, which converts two f32 data at a time
  and pack the two result f16 data in the dst register.
  This saves us a lot of `v_pack` instructions compared to using
  `v_cvt_f16_f32_e32`. 
- `impl1_TritonInterleaveAndRematRebase0`: use customized LLVM branch [TritonInterleaveAndRematRebase0](https://github.com/jrbyrnes/llvm-project/tree/TritonInterleaveAndRematRebase0) and RTZ.
- `impl1_TritonInterleaveAndRematRebase0_scalarized`
  - use customized LLVM branch [TritonInterleaveAndRematRebase0](https://github.com/jrbyrnes/llvm-project/tree/TritonInterleaveAndRematRebase0)
  - use RTZ
  - use `AMDGCN_SCALARIZE_PACKED_FOPS=1` to break `v_pk_mul` into `v_mul`
- `impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue` has the same settings as
  `impl1_TritonInterleaveAndRematRebase0_scalarized` plus
  - insert `sched.barrier` in the prologue and epilogue to separate ops from different clusters
  
### Advanced Register Analyzer `ara.py`

Usage:
```bash
# cd into current dir
python3 ara/ara.py impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue/attn_fwd.s
```

This tool analyzes the register usage of the computer clusters in the main loop for FAv3 kernels.
The register usage is saved in `ara/reg_usage.txt`. The contents of the reg usage for
the above command is shown below
```bash
QK: 32;66:97
PP: 33;1,195:222,246:249
P: 16;114:129
K: 64;80:95,114:129,162:193
V: 64;162:193,196:223,246:249
Q: 32;130:161
ACC: 64;2:65
```
The format is `SYMBOL: N;x:y`, which means
- SYMBOL: the symbol for the tensor.
  - QK: result tensor of 1st dot.
    - These values are produced by mfma instructions in cluster 0.
    - These values are used (or updated online) by `v_max`, `v_fma`, and `v_exp` instructions in cluster 2.
  - PP: result of exp(QK) in 32-bit.
    - These values are produced by `v_exp` instructions in cluster 2.
    - These values are used by `v_add` and `v_cvt_pkrtz` instructions in cluster 0.
  - P:  result of exp(QK) in 16-bit. This is the result of truncation from PP.
    - These values are produced by `v_cvt_pkrtz` instructions in cluster 0
    - These values are used as the 2-th operands of mfma instructions in cluser 2.
  - K:  K tensor.
    - These values are read by `ds_read` instructions in cluster 3.
    - These values are used as the 1-th operands of mfma instructions in cluster 0.
  - V:  V tensor.
    - These values are read by `ds_read` instructions in cluster 1.
    - These values are used as the 1-th operands of mfma instructions in cluster 2.
  - Q:  Q tensor. Loop invariant.
    - These values are read into registers in the prologue.
    - These values are used as the 2-th operands of mfma instructions in cluster 0.
  - ACC: result tensor of the 2nd dot. Loop invariant
    - These values are initialized in registers in th prologue.
    - These values are used and updated as the accumulator (0-th and 3-th operands) of
      mfma instructions in cluster 2.
- N: number of registers used to hold values for this tensor
- x:y: register indices

The `reg_usage.txt` file tells us the register usage of these tensors.
What is more interesting is the overlap of these registers. For example, K tensor and V tensor
have disjoint liveness in registers, which means we can use the exactly same set of registers
for K and V. The LLVM backend compiler may or may not allocate registers as we expect.
The above command will output the overlap of each pair of tensors.
It also outputs the total number of registers, together with the register indices 
for all of these tensors.

We can use this tool to evaluate the LLVM register allocation for different experiments.
Here is the register usage inside the loop for some interesting experiments.

setting | reg usage inside the loop | kernel reg spill
-- | -- | --
`impl1_rtz_annot` | 235 | 70
`impl1_TritonInterleaveAndRematRebase0_scalarized` | 214 | 72
`impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue` | 210 | 43

From the above table, we can see
- The `TritonInterleaveAndRematRebase0` really helps a lot in reducing the register
  usage **inside the loop**.
- The kernel spills a lot. Annotating the pro and epilogue with `sched.barrier`
  to separate the ops from different clusters helps a lot in reducing the spills.

### Investigation of the epilogue

The VectorCombine pass moves some op across `sched.barrier`.
It moves the accUpdate op, i.e. `v_fmul` from cluster 0 to right before the mfma
instructions in cluster 2. Why??


### Investigation of `ds_read` for V tensor

- in-thread-transpose. I tried to use `transV` kernel, in which V tensor is pre-tranposed
  so there is no need to use in-thread-transpose.
  This only reduces register spill from 73 to 63.
- [PR#7355](https://github.com/triton-lang/triton/pull/7355). The new lowering of localLdSt
  improves the xor computation for the addresses of ds instructions.
