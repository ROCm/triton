# Study matmul

Base triton compiler commit: 77e7a7b74f0731d0e65fb9 (PR#9092)

command
```bash
TRITON_LLVM_DEBUG_ONLY="tritonamdgpu-prera-llir-schedule" python bench_gemm_a16w16.py --shape 4096 4096 4096
```

## Triton path

orig

- Triton compiler: base
- aiter: 707b9fca3aa68ae86aa (PR#1715)
  - Need to change the confign in `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16.json`
    to 256x256x32 w/ nS=3
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/triton/orig`


uniformWarpId

- triton compiler: https://github.com/triton-lang/triton/pull/9073 rebased on top of 77e7a7b74f0731
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/triton/uniformWarpId`

This PR is necessary to remove the `readfirstlane` before `buffer_load_lds`.


## 4 wave

- Branch: `matmul_4waves`
- Commit: `813f6f4fe8`

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

### llir sched v0

- Commit: `3e3966a173`
- gluon kernel version: v5
- IR dump: `/var/lib/jenkins/OAI-triton/study_matmul/gluon/v5_pred_llirSchedV0`
- vgpr: 510
- perf: 950 tflops

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
