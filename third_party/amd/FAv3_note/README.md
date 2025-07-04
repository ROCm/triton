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
- `disable_vector_combine` has the same settings as `impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue` plus `DISABLE_LLVM_OPT="disable-vector-combine"`
- `disable_vector_combine_hack` has the same settings as `disable_vector_combine` plus manual hack
  of llvm ir to restore the order of ops into their designated clusters in the epilogue.
  
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
`disable_vector_combine` | 210 | 48
`disable_vector_combine_hack` | 226 | 68

From the above table, we can see
- The `TritonInterleaveAndRematRebase0` really helps a lot in reducing the register
  usage **inside the loop**.
- The kernel spills a lot. Annotating the pro and epilogue with `sched.barrier`
  to separate the ops from different clusters helps a lot in reducing the spills.
  

### Investigation of the epilogue

The VectorCombine pass moves some op across `sched.barrier`.
It moves the accUpdate op, i.e. `v_fmul` from cluster 0 to right before the mfma
instructions in cluster 2. Why??

Disabling the vectorCombine pass only helps keep `mul` (updateAcc) in its
designated cluster. Some other ops are still pushed into later clusters.
```bash
DISABLE_LLVM_OPT="disable-vector-combine" AMDGCN_SCALARIZE_PACKED_FOPS=1 python fa/flash-attention.py
```

The SLPVectorizerPass moves sub and exp across sched.barrier.
However, disabling this pass causes even more spilling. :(

The epilogue scheduling with vectorCombine and SLP passes is as follows
```
E0
E0-Cluster0(DOT1[2]+VEC2[1])
DOT1[2]: 32 x mfma
VEC2[1]: 32 x add
VEC2[1]: 64 x mul <--- missing
VEC2[1]: 16 x cvtrtz

E0-Cluster1
LRV[1]: 32 x load<4 x half>
LWK[3]: 2  x store<8 x half>

E0-Cluster2(DOT2[1]+VEC1[2])
VEC2[1]: 64 x mul <--- From E0-Cluster0
DOT2[1]: 32 x mfma
VEC1[2]: 32 x max
VEC1[2]: 32 x mul

VEC1[2]: 32 x sub <--- missing
VEC1[2]: 32 x exp <--- missing

E0-Cluster3
LRK[3]: 16 x load<8 x half>
LWV[2]: 8  x store<2 x half>
GRV[3]

///////////////////////////////////////////////////
E1
E1-Cluster0(DOT2[3]+VEC2[2])
DOT2[3]: 32 x mfma

VEC2[2]: 32 x add <--- missing
VEC2[2]: 64 x mul <--- missing
VEC2[2]: 16 x cvtrtz <--- missing

E1-Cluster1(LRV[2])
LRV[2]: 32 x load<4 x half>

E1-Cluster2(DOT2[2]+VEC1[3])
VEC1[3]: 32 x max

VEC1[2]: 32 x sub <--- From E0-Cluster2
VEC1[2]: 32 x exp <--- From E0-Cluster2

VEC2[2]: 32 x add <--- From E1-Cluster0
VEC2[2]: 16 x cvtrtz <--- From E1-Cluster0
VEC2[2]: 64 x mul <--- From E1-Cluster0

VEC1[3]: 32 x mul
VEC1[3]: 32 x sub
VEC1[3]: 32 x exp

E1-Cluster3(LWV[3])
LWV[3]: 8 x store<2 x half>

/////////////////////////////////////////////////////
E2
E2-Cluster0(VEC2[3])
VEC2[3]: 32 x add
VEC2[3]: 16 x cvtrtz

VEC2[3]: 64 x mul <--- missing

E2-Cluster1(LRV[3])
LRV[3]: 32 x load<4 x half>

E2-Cluster2(DOT2[3])
VEC2[3]: 64 x mul <--- From E2-Cluster0
DOT2[3]: 32 x mfma

```

Disabling `vector-combine` pass can restore `mul` to its designated cluster. 
I have to manually restore other ops. The hacked llvm ir is
`/var/lib/jenkins/OAI-triton/third_party/amd/FAv3_note/disable_vector_combine/attn_fwd.llir.hack`.
The result IR dump dir is `disable_vector_combine_hack`.


### Investigation of `ds_read` for V tensor

- in-thread-transpose. I tried to use `transV` kernel, in which V tensor is pre-tranposed
  so there is no need to use in-thread-transpose.
  This only reduces register spill from 73 to 63.
- [PR#7355](https://github.com/triton-lang/triton/pull/7355). The new lowering of localLdSt
  improves the xor computation for the addresses of ds instructions.
  - Cherry picked PR7201 --> 6921 --> 7036 --> 6982 --> 7218 --> 7241 --> 7248 --> 7355
  - new branch: `FAv3_gfx942_noAsyncCopy_newLowerLdSt`
  - This makes things worse. Now the kernel has 113 spills :( 



### Add some basic blocks?

The experiments seem to tell us that
- The backend is not doing well if the basic block has too many instructions.
  This is aligned with the observations when we unroll the loop, in which case
  there are many spills.
- The epilogue contains 3 iterations of the loop. This is similar to unroll
  the loop by a factor of 3.
  Maybe we can put each iteration into a separate basic block?
- sched.barrier seems to be a mark for instruction scheduling.
  It does not mark a region for register allocation.
