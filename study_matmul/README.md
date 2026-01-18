# Study matmul

Base triton compiler commit: 77e7a7b74f0731d0e65fb9 (PR#9092)

command
```bash
TRITON_LLVM_DEBUG_ONLY="tritonamdgpu-prera-llir-schedule" python bench_gemm_a16w16.py --shape 4096 4096 4096
```

## Triton path

orig (81%)

- Triton compiler: base
- aiter: 707b9fca3aa68ae86aa (PR#1715)
  - Need to change the confign in `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16.json`
    to 256x256x32 w/ nS=3
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/triton/orig`


uniformWarpId (69%)

- triton compiler: https://github.com/triton-lang/triton/pull/9073 rebased on top of 77e7a7b74f0731
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/triton/uniformWarpId`

This PR is necessary to remove the `readfirstlane` before `buffer_load_lds`.


## gluon 4 wave

- Branch: `matmul_4waves`
- Commit: `813f6f4fe8`
- 58%

Config:
```json
"any": {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": null,
        "NUM_KSPLIT": 1,
        "kpack": 1
    }
```

On top of the base commit, we need the following optimizations
- PR#9073 Add uniform hint to ttg.warp_id.
  This removes `readfirstlane` before `buffer_load_lds`.
- `95909a0199` Optimize address calculation for paddedShared layout.
  This makes sure `ds_read` uses the same gpr as the addr for padded shared layout.
- PR#8474 Optimize address increments for buffer loads in loops。
  This replaces `v_add` with `s_add` when updating the addr of `buffer_load_lds`.

IR Dump: `/var/lib/jenkins/OAI-triton/study_matmul/triton/4waves_orig`

```
ROCPROF_ATT_LIBRARY_PATH=/var/lib/jenkins/att-decoder-v3-3.0.0-Linux/opt/rocm/lib/ rocprofv3 --att -i att_matmul.json -d ./study_matmul/gluon/v4/att_output -- python study_matmul/gluon/gl_matmul.py
```

```
AMD_INSERT_AMDGCN=/var/lib/jenkins/OAI-triton/study_matmul/asm_tool/new_asm.s
```

```
TRITON_LLVM_DEBUG_ONLY="tritonamdgpu-prera-llir-schedule" python study_matmul/gluon/gl_matmul.py
```

### llir sched v0

- Commit: `3e3966a173`
- gluon kernel version: v5
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v5_pred_llirSchedV0`
- vgpr: 510
- perf: 950 tflops
- 53%

The `v5_pred` version of the kernel implements a 3-stage pipeline and full LDS buffer
prefetch for both A and B.

The llir scheduler does
- Insert 4 mfma after each buffer.load.lds
- Insert 1 mfma after each lds load

However, the backend does not respect the scheduling at llir level

### llir sched v1

- Commit: `558501b132`
- gluon kernel version: v5
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v5_pred_llirSchedV1`
- vgpr: 510
- perf: 1120 tflops
- 77%

Same kernel is used as v0.

This version of the llir scheduler further inserts `sched.barrier` before
each anchor instructions, i.e. `buffer.load.lds` and `lds load`.
This forces the backend to respect the scheduling arranged at llir level.

This version forces mfma and `ds_read` to interleave at the end so
there are `v_mov` instructions to copy data from `ds_read` into the vgprs
expected by mfma.
However, `ds_read` and mfma are NOT using disjoint sets of vgprs and
that's why the kernel is not using more than 512 vgprs.
Note that from the roofline analysis, such 3 stage full LDS buffer prefetch
needs more than 512 vgprs.

Next steps:
1. Unroll the loop to resolve the `v_mov` issue
2. Epilogue takes 12% of the total time.

|          | cycles | portion |
|----------|--------|---------|
| prologue | 6256   | 3%      |
| loop     | 164316 | 85%     |
| epilogue | 21784  | 12%     |


### llir sched v2

We can actually achieve the same thing by disabling `misched` and `post-misched`.
- Commit: `a9ac373e8b`
- gluon kernel version: v5
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v5_pred_llirSchedV2`
- vgpr: 512
- perf: 1120 tflops

### kernel v6

The same llir scheduler happens to work with the v6 version of the kernel, which does
- 3 stage pipeline with partial LDS prefetch
- Set `tiles_per_warp`=[2,2]
- Select a better layout for `buffer_store` to enable `buffer_store_dwordx4`

Result
- gluon kernel version: v6
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v6`
- vgpr: 478
- perf: 1150 tflops
- 74%

### kernel v7

Unroll the loop

next step
- RA. Since `v_cvt` only works with vgprs, we want acc to be in vgprs ideally.
  However, acc needs 256 vgprs, which leaves 0 for anything else requiring vgprs
  such as `ds_read` addr and `buffer_load` addr.

### llir sched v3

Region based llvm ir scheduler.

By enabling `sched.barrier`, the loop can be partitioned into several scheduling regions.
The scheduler only moves instructions within its own region.

This version shows the key idea of the llvm ir scheduler:
- The gluon kernel handles op scheduling at region scope.
- Then the llir scheduler does fine grained interleaving within each region.


- gluon kernel version: v7
  - Note that we also set K as `gl.constexpr` since this cleans up the basic block
    structure in the final assembly code.
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v7`
- vgpr: 512
- perf: 1190 tflops
- 81%

### amdgcn v0

Use the python assembler to further compile the generated amdgcn file from v7.
This version removes `v_accvgpr_` and related `s_nop` instructions inside the loop.

- att:  `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v7/att_output_amdgcnasV0`
- new assembly: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v7/new_asm.s`
- 87%

Next steps
- [ ] `s_nop` and `v_accvgpr` before `buffer_load`
- [ ] The loop starts and ends with a number of `s_xx` instructions
- [ ] The following pattern to compute `ds_read` addr
      ```
      v_accvgpr_read_b32 v1, a174
      v_add_u32_e32 v254, 0, v1
      v_add_u32_e32 v6, 0x18bc0, v254
      ```
- [ ] the first `buffer_load` always takes longer?
- [ ] `s_mov_b32 m0, s25` We should be able to compute m0 directly
- [ ] There are a few salu and valu instructions between `ds_read` and `buffer_load`
      regions. And we see DIDT issue after that. We need more mfma between them.

### llir sched v4

This version puts 2 more mfma after the last anchor, and 2 more mfma between
different kinds of anchors.

- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v7_llirSchedV4`

### llir sched v5

This version inserts `s_waitcnt lgkmcnt(0)` in front the 1st mfma in each region.
This reduces the number of `s_xx` instructions.
And since there are enough mfma instructions from the last `ds_read` and
the 1st mfma, there should not be any extra wait cycles at the `lgkmcnt(0)`.

- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v7_llirSchedV5`


#### Study buffer load

```
s_cmpk_eq_i32 s4, 0x1f00
s_cselect_b64 vcc, -1, 0

v_accvgpr_read_b32 v1, a153
v_cndmask_b32_e32 v1, v1, v0, vcc
```

| tensor | addr | comes from |
|--------|------|------------|
| A (8)  | v1   | a153       |
|        | v1   | a154       |
|        | v1   | a155       |
|        | v1   | a156       |
|        | v1   | a157       |
|        | v1   | a158       |
|        | v1   | a159       |
|        | v1   | a160       |
| B0 (4) | v1   | a161       |
|        | v1   | a162       |
|        | v1   | a163       |
|        | v1   | a164       |
| B1 (4) | v10  | a165       |
|        | v1   | a166       |
|        | v11  | a167       |
|        | v12  | a168       |
| unroll |      |            |
| A (8)  | v0   | a179       |
|        | v0   | a196       |
|        | v0   | a197       |
|        | v0   | a198       |
|        | v0   | a199       |
|        | v0   | a200       |
|        | v0   | a201       |
|        | v0   | a202       |
| B0 (4) | v0   | a203       |
|        | v0   | a204       |
|        | v0   | a205       |
|        | v0   | a206       |
| B1 (4) | v10  | a165       |
|        | v1   | a166       |
|        | v11  | a167       |
|        | v12  | a168       |

### kernel v8

This version changes where to update the base ptr for buffer load

before
```
load A
load B0
load B1

base += BLOCK_K
```

after
```
load A
load B0

base += BLOCK_K
load B1
```

In this way, the `s_add` instructions to update the base won't be scheduled at
the very beginning of the loop.

- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v8`
- 82%

```
ROCPROF_ATT_LIBRARY_PATH=/var/lib/jenkins/att-decoder-v3-3.0.0-Linux/opt/rocm/lib/ rocprofv3 --att -i att_matmul.json -d ./study_matmul/gluon/v8/att_output -- python study_matmul/gluon/gl_matmul.py
```

### amdgcnas v1

This version of the amdgcnas optimized the register allocation for
the `voff` regs of `buffer_load`.


- asm: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v8/v8_amdgcnasv1.s`
- 94%

### amdgcnas v2

Added LICM. This helps hoist instructions to compute the `ds_read` addr.

- asm: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v8/v8_amdgcnasv2.s`
- 97%

|          | cycles | portion |
|----------|--------|---------|
| prologue | 5156   | 3.4%    |
| loop     | 135220 | 90%     |
| epilogue | 9120   | 6%      |

### kernel v9

Add support for xcd remapping logic

Kernel level change, loop code stays the same.


### amdgcnas v3

- `rotate_lgkmcnt`
- add mfma between `waitcnt` and `s_barrier`
- separate non-mfma with neighboring mfma instructions

```
AMD_INSERT_AMDGCN=/var/lib/jenkins/OAI-triton/study_matmul/gluon/experiment/correct/v10.s python study_matmul/gluon/gl_matmul.py
```

### kernel v10 + amdgcn v4

Kernel level change
- slice acc0 and acc1 further into 4 pieces
- Use blocked layout for `buffer_store`

amdgcnas v4
- Reassign registers for `ds_read` based on buffer reuse analysis
- Can also reassign registers for `mfma` if they are using too many vgprs
- Bug fixes and refactor

- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v10_amdgcnasV4`

### kernel v11

Explicitly allocate LDS for c0 and c1

In this case, `lgkmcnt` after `ds_read` and before `buffer_store` are very long.
So this version is worse than v10 :shrug

- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v11_amdgcnasV4`


### kernel v9_1

Replace the linear layout with basic blocked layout for `buffer_store`.


### Summary

The baseline kernel, v5, implements a 3-stage pipeline with full LDS buffer prefetch.
The llir schduler interleaves mfma and `ds_read` and `buffer_load`.
But the backend does not respect such scheduling.

Notes
- `v5_llirSchedV0`: The baseline kernel, v5, implements a 3-stage pipeline with full LDS buffer prefetch.
  The llir schduler interleaves mfma and `ds_read` and `buffer_load`.
  But the backend does not respect such scheduling.
- `v5_llirSchedV1`: diable misched in LLVM so the llir level scheduling is repsected
- `v6`: partial LDS prefetch and better linear layout to enable `buffer_store_dwordx4` by `v_permlane`.
- `v7`: unroll the loop + llirSchedV3, which does region based llvm ir scheduling
- `v7_amdgcnasV0`: introduced the amdgcnas v0, which re-assign registers for mfma chains
  to remove `v_accvgpr_` instructions.
- `v7_llirSchedV4`: llirSchedV4, which puts 2 extra mfma during the anchor op transition region.
  At this region, there are usually valu and salu instructions to compute the addr for the
  anchor ops. Thus we need more mfma to cover them.
- `v8`: update `buffer_load` base addr in the middle of the loop + llirSchedV5, which
  inserts `s_waitcnt lgkmcnt(0)` at the beginning of each cluster to save `s_waitcnt`
  instructions inside the cluster.
- `v8_amdgcnasV0`: apply amdgcnas v0 on top of `v8`.
- `v8_amdgcnasV1`: amdgcnas v1, which optimizes `voff` reg calculation of `buffer_load`.
- `v8_amdgcnasV2`: amdgcnas v2, which implements LICM to hoist valu and salu insts used
  to compute `ds_read` addr out of the loop.
- `v9_amdgcnasV3`: xcd remapping + `group_m` + amdgcnas v3, which contains several peephole opt
  - rotate `lgkmcnt`: so that the loop can start with mfma.
  - add mfma between `waitcnt` and `s_barrier` so that the latency of `s_barrier`
    can be partially hidden.
  - separate non-mfma with neighboring mfma instructions
- `v9_amdgcnasV4`: amdgcnas v4, which can assign registers for mfma and lds chains.
  Also bug fixes and refactor.
- `v9_1_amdgcnasV4`: kernel `v9_1` uses blocked layout for `buffer_store`, which leads to
  LDS traffic in the epilogue
- `v10_amdgcnasV4`: kernel `v10` slices acc0 and acc1 further into 4 pieces
- `v11_amdgcnasV4`: kernel `v11` uses LDS buffer explicitly for layout conversion.


|                   | mfma eff | tflops | prologue  | epilogue    | loop         |
|-------------------|----------|--------|-----------|-------------|--------------|
| `v5_llirSchedV0`  | 52.88%   | 950    | 5228 (2%) | 21912 (8%)  | 240112 (90%) |
| `v5_llirSchedV1`  | 77.28%   | 1120   | 6260 (3%) | 21784 (11%) | 164321 (85%) |
| `v6`              | 73.95%   | 1150   | 5484 (3%) | 8384 (4%)   | 174484 (93%) |
| `v7`              | 81.76%   | 1190   | 5268 (3%) | 9120 (5%)   | 160312 (92%) |
| `v7_amdgcnasV0`   | 86.77%   |        | 5612 (3%) | 9124 (5%)   | 151056 (91%) |
| `v7_llirSchedV4`  | 85.48%   |        | 5360 (3%) | 8068 (5%)   | 153344 (92%) |
| `v8`              | 82%      |        | 5080 (3%) | 9140 (5%)   | 159832 (92%) |
| `v8_amdgcnasV0`   | 86.87%   |        | 5228 (3%) | 8112 (5%)   | 150888 (92%) |
| `v8_amdgcnasV1`   | 94.7%    |        | 5520 (4%) | 8116 (5%)   | 138288 (91%) |
| `v8_amdgcnasV2`   | 96.9%    |        | 5160 (3$) | 9120 (6%)   | 135216 (90%) |
| `v9_amdgcnasV3`   | 98.5%    |        | 4436 (3%) | 8056 (6%)   | 133060 (91%) |
| `v9_amdgcnasV4`   | 98.6%    | 1387   | 4324 (3%) | 8056 (6%)   | 132924 (91%) |
| `v9_1_amdgcnasV4` | 98.64%   | 1428   | 4332 (3%) | 11460 (8%)  | 132880 (89%) |
| `v10_amdgcnasV4`  | 98.21%   | 1425   | 4536 (3%) | 8808 (6%)   | 133456 (91%) |
| `v11_amdgcnasV4`  | 98.33%   | 1395   | 4540 (3%) | 13936 (9%)  | 133300 (88%) |
