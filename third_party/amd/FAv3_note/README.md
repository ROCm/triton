# FAv3 without AsyncCopy on gfx942 (MI300)

Instruction analysis is kept in the "4-stage on MI300" tab in the [project excel](https://amdcloud-my.sharepoint.com/:x:/r/personal/lixzhang_amd_com/Documents/8-wave_FAv3_project.xlsx?d=wf597bd4f12b048d1a5f7839e263e965e&csf=1&web=1&e=C2eGj1) with label `impl1`.

We also developed [ara](https://github.com/ROCm/triton/tree/FAv3_gfx942_noAsyncCopy/third_party/amd/FAv3_note/ara), a tool for advanced register analysis for FAv3 kernels.


## Redesign the pipeline and clusters according to [ticket#902](https://github.com/ROCm/triton-internal/issues/902)

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


## Adjust pingpong FAv3 and use rtz + kpack=2

Compiler change: 
- Adjust the logic in pingpong to insert sched.barrier to separate the clusters.
- commit: https://github.com/ROCm/triton/tree/FAv3_gfx942_noAsyncCopy @140d69ae913

Kernel change:
- kernel: https://github.com/ROCm/triton/blob/FAv3_gfx942_noAsyncCopy/fa/flash-attention.py
- Use rtz to convert f32 to f16.
- Set kpack=2

Command to run the kernel
```bash
AMDGCN_SCALARIZE_PACKED_FOPS=1 python fa/flash-attention.py
```

### Experiment Summary

- `impl1_TritonInterleaveAndRematRebase0_scalarized`
  - use customized LLVM branch [TritonInterleaveAndRematRebase0](https://github.com/jrbyrnes/llvm-project/tree/TritonInterleaveAndRematRebase0)
  - use RTZ when converting f32 to f16. 
    This leads to `v_cvt_pkrtz_f16_f32`, which converts two f32 data at a time
    and pack the two result f16 data in the dst register.
    This saves us a lot of `v_pack` instructions compared to using
  - use `AMDGCN_SCALARIZE_PACKED_FOPS=1` to break `v_pk_mul` into `v_mul`
- `impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue` has the same settings as
  `impl1_TritonInterleaveAndRematRebase0_scalarized` plus
  - insert `sched.barrier` in the prologue and epilogue to separate ops from different clusters
- `disable_vector_combine` has the same settings as 
  `impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue` 
  plus `DISABLE_LLVM_OPT="disable-vector-combine"`
- `disable_vector_combine_hack` has the same settings as `disable_vector_combine` plus manual hack
  of llvm ir to restore the order of ops into their designated clusters in the epilogue.
  

We can use `ara` to evaluate the LLVM register allocation for different experiments.
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
- Instruction scheduling in the epilogue seems to have a huge impact on the reg usage
  inside the loop, as well as the reg spills for the kernel.
  - Annotating the pro and epilogue with `sched.barrier` to separate the ops
    from different clusters (`annotEpiloguePrologue`) can further reduce the
    reg usage inside the loop and reduce spills a lot.
  - Restoring ops into their designated clusters in the epilogue
    (`disable_vector_combine_hack`) makes it worse in terms of reg usage inside
    the loop and total spills.
  

### Investigation of the epilogue

The epilogue scheduling in
`impl1_TritonInterleaveAndRematRebase0_scalarized_annotEpiloguePrologue` is

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
E1-Cluster0(DOT1[3]+VEC2[2])
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
- The `mul` used to update acc is pushed from cluster 0 to cluster 2.
- The `sub` and `exp` ops from E0 cluster 2 are pushed to E1 cluster 2.
- All the ops in VEC2[2] in E1 cluster 0 are pushed to E1 cluster 2.

The problem of moving `sub` and `exp` from E0 cluster 2 to E1 cluster 2 is that
now the epilogue needs more registers than in the loop.
DOT1[3] in E1 cluster 0 needs registers for QK in E1.
However, `sub` and `exp` from E0 cluster 2 need QK from E0 to do the computation.
Therefore, **QK from E0 and E1 are live at the same time** so they cannot share
the same set of registers.

#### Restore the same structure in E0,E1,E2 as inside the loop

The VectorCombine pass moves some op across `sched.barrier`.
It moves the accUpdate op, i.e. `v_fmul` from cluster 0 to right before the mfma
instructions in cluster 2.
We can disable the vector-combine pass by
```bash
DISABLE_LLVM_OPT="disable-vector-combine" AMDGCN_SCALARIZE_PACKED_FOPS=1 python fa/flash-attention.py
```
The result IR dump is saved in `disable_vector_combine`.

Disabling `vector-combine` pass can restore `mul` to its designated cluster.
The SLPVectorizerPass moves `sub` and `exp` across sched.barrier.
However, disabling this pass causes even more spilling. :(

To restore `sub` and `exp`, we have to manually restore other ops. 
The hacked llvm ir is
`/var/lib/jenkins/OAI-triton/third_party/amd/FAv3_note/disable_vector_combine/attn_fwd.llir.hack`.
The result IR dump is saved in `disable_vector_combine_hack`.

However, from the above table we can see restoring the order of ops in clusters
does not help with register pressure of the kernel. 
`disable_vector_combine_hack` even does worse in terms of register allocation
inside the loop.

#### Add some basic blocks to guard the structure of E0,E1,E2

The experiments seem to tell us that
- The backend is not doing well if the basic block has too many instructions.
  This is aligned with the observations when we unroll the loop, in which case
  there are many spills.
- The epilogue contains 3 iterations of the loop. This is similar to unroll
  the loop by a factor of 3.
  Maybe we can put each iteration into a separate basic block?
- sched.barrier seems to be a mark for instruction scheduling.
  It does not mark a region for register allocation.


### Investigation of `ds_read` for V tensor

- in-thread-transpose. I tried to use `transV` kernel, in which V tensor is pre-tranposed
  so there is no need to use in-thread-transpose.
  This only reduces register spill from 73 to 63.
- [PR#7355](https://github.com/triton-lang/triton/pull/7355). The new lowering of localLdSt
  improves the xor computation for the addresses of ds instructions.
  - Cherry picked PR7201 --> 6921 --> 7036 --> 6982 --> 7218 --> 7241 --> 7248 --> 7355
  - new branch: `FAv3_gfx942_noAsyncCopy_newLowerLdSt`
  - This makes things worse. Now the kernel has 113 spills :( 



