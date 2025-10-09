# Hack pingpong0 (pp0)

Experiment setup
- Triton compiler branch: https://github.com/AlexAUT/triton/commits/faPipelining/
- commit: acacb5fb605331239a6faf5a62c1be563b4f2ca7

Command
```bash
rm -rf ~/.triton/cache
TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_USE_ASYNC_COPY=1 python fa/flash-attention.py
```

The generated IR is saved in `study_4-stage/orig`

## Enable pingpong0

We hack `study_4-stage/orig/attn_fwd.ttgir` to enable pingpong, the hacked
ttgir is saved as `study_4-stage/hack_pingpong0.ttgir`.

Run the kernel with `hack_pingpong0.ttgir`, the generated IR is saved in
`study_4-stage/hack_pingpong0/IR_dump/`


# Hack pingpong1 (pp1)

Experiment setup
- Triton compiler branch: https://github.com/AlexAUT/triton/commits/faPipelining/
- commit: 9ca58ef7eaa48d830f731b3e2e8db8839d92f633
- Changes from pp0
  - bypass permute for AsyncCopy when load dim is contiguous
  - Fix the extra barrier between ACK[3] and LRV[0]
  - Apply [#781](https://github.com/ROCm/triton/pull/781) to flash-attention.py
    to fold sub into fma
    
Hacked ttgir is saved as `study_4-stage/hack_pp1.ttgir`.
The IR dumps from this hacked ttgir is saved in `study_4-stage/hack_pp1/IR_dump/`.

Since there is barrier between bufferLoadToLocal and `ds_read`, I have to remove
the extra `s_barrier` and the hacked assembly is saved as `study_4-stage/hack_pp1/hack_remove-barrier.amdgcn`.

# Hack pingpong2 (pp2)

- commit: eba533068bed789b6
- changes from p1
  - move localLoad before AsyncCopy to remove the barrier inside the cluster
  - Set vecSize = 8 to enable dwordx4 AsyncCopy
  - Applied [PR6594](https://github.com/triton-lang/triton/pull/6594) to
    replace `ds_bpermute` with `v_permlane32_swap`.
    
Hacked ttgir is saved as `study_4-stage/hack_pp2.ttgir`.
The IR dumps from this hacked ttgir is saved in `study_4-stage/hack_pp2/IR_dump/`.

# Hack pingpong3 (pp3)

`study_4-stage/hack_pp3.ttgir`

Changes from pp2:
- Reverse the priority of compute and memory cluster. Now mem cluster has higher prio than compute.
- remove gpu.barrier after mem cluster

Now I need to hack the assembly
- `hack_asm0.amdgcn`: Add s_barrier after mem cluster ==> 788 tflops
- `hack_asm1.amdgcn`: Manually breakdown some `v_pk_mul` ==> 800 tflops
  - If further remove all leftover `v_pk_mul` ==> 819 tflops
  
## bf16 input

`study_4-stage/hack_pp3_bf16.ttgir`

- Original run ==> 756
- `hack_asm0.amdgcn`: add s_barrier after memory cluster ==> 800 tflops
- `hack_asm1.amdgcn`: Manually breakdown some `v_pk_mul` ==> 830 tflops
  - If further remove all leftover `v_pk_mul` ==> 840 tflops

# Hack pingpong4 (pp4)

unroll the loop, does not work well

# Hack pingpong5 (pp5)

`study_4-stage/hack_pp5.ttgir`

Changes from pp3
- Set vecSize=32 for sharedLayout of V tensor to remove bank conflicts with `ds_read_tr` instructions

Assembly hack
- `hack_asm0.amdgcn`: add s_barrier after memory cluster ==> 820 tflops
- `hack_asm1.amdgcn`: Manually breakdown some `v_pk_mul` ==> 880 tflops
  - If further remove all leftover `v_pk_mul` ==>  900 tflops
- `hack_asm2.amdgcn`: Manually reorder some `mfma` and `exp` ==> 890 tflops
  - If further remove all leftover `v_exp` ==>  915 tflops

# Automatic pingpong

Triton compiler:
- base branch: https://github.com/AlexAUT/triton/commits/faPipelining
- base commit: f7d950bdec23f8dc945ad22c99218c03dfeb7f19
- cherry picked: 50b2057e480447e0cddce0f8245555c92fee73e9 from https://github.com/jungpark-mlir/triton/commits/pp-4s/

Perf
- fp16: 830 tflops
- bf16: 870 tflops

With this version of compiler, there is no need to hack ttgir or assembly.
The dumped IR is saved in `study_4-stage/pp-4s/IR_dump`

# unpack `v_pk_mul` right after llir

Cherry picked https://github.com/triton-lang/triton/pull/6656,
which unpacks `v_pk_mul` at llvmir level.

Perf improves a bit. However, it reduces vgpr count from 256 (8) to 253 (0).
The dumped IR is saved in `study_4-stage/pp-4s_unpack-mul/IR_dump`.
The branch associated with this change is checkout as `test_fav3`.

# loop unrolling

- `hack_unroll0.ttgir`: unroll the loop with a factor of 2 but early return inside
the loop. Done by Alex. Original [gist](https://gist.github.com/AlexAUT/3bc3305fff10f1c989d47b2a1336bf81).
- `hack_unroll1.ttgir`: since we unroll the loop, we can use hardcoded buffer index
for ACK and ACV in the loop

We think it should help with vgprs usages (and it does for a simple matmul kernel), 
but the FA kernel is too complicated for LLVM to make reasonable decisions about reg allocation. 
E.g. we think the two reg sets for LRK and LRV should be the same since they do not overlap. 
But it seems the backend does not distinguish different instruction types. 
Also we think element-wise instructions should just update the reg in place, but some time it uses a new reg. 
That being said, the backend is already struggling to figure out reg allocation to the loop body, 
it makes it worse when the loop is unrolled. 
 
On the other hand, Maks PR reduces the vgpr usage from 8 spills to 0 (253 vgprs in total). 
I checked the kernel and it's just the backend allocates the regs in a different way :shrug

Since we have no spills, there is no need to explore the loop unroll pass, 
which does not work with the current LLVM backend anyway.

# Early wait for LR to finish

We can wait with `lgkmcnt(0)` at the beginning of the compute cluster given that 
`ds_read` should finish before compute cluster begins. This can save a lot of 
`s_waitcnt` instructions inside the compute cluster. Note that `s_waitcnt` cannot
be co-issued (not co-execute) with neither mfma or valu instructions.
Therefore, this can save the lost cycles inside the compute cluster.

The IR change is saved as `study_4-stage/hack_s_waitcnt.ttgir`.
The dumped IR is saved in `study_4-stage/hack_s_waitcnt/IR_dump`.

However, such change also affect how backend schedules mfma and valu instructions.
As a result, worse interleaving is generated.

Then we hack the original amdgcn directly by removing all `s_waitcnt lgkmcnt(x)`
and change the first one to be `lgkmcnt(0)` in the compute cluster.
The hacked assembly code is saved as `hack_s_waitcnt.s`

Perf
- before this change: 850 tflops
- use `hack_s_waitcnt.ttgir`: 830 tflops
- use `hack_s_waitcnt.s`: 860 tflops

# Update faPipelining branch

commit: 9adc728311978a042967466f8a1f3a5981d06749

This branch has lower perf than branch `test_fav3` with command
`TRITON_HIP_SCALARIZE_VECTOR_FOPS=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_USE_ASYNC_COPY=1 python fa/flash-attention.py`

It also has 4 spills.

# Register spill study

Experiment setup
- docker: `compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-npi-mi350:765_ubuntu22.04_py3.10_pytorch_rocm6.4_internal_testing_gfx950_f2faad5`
- branch: https://github.com/ROCm/triton/blob/fa_reg_spill
  - install: `cd triton && pip install -e .`
- FA kernel: https://github.com/ROCm/triton/blob/fa_reg_spill/fa/flash-attention.py

Parameters:
- `assume` vs `no assume`: this refers to the `tl.assume` before the for loop at
this [line](https://github.com/ROCm/triton/blob/d04c4cc1eb586b27176c1a8ae2a66c969f394613/fa/flash-attention.py#L252).
With `tl.assume`, we can get rid of `if` in the epilogue of the first loop.
- `rm-2ndFor`: this refers to removal of the 2nd forLoop in ttgir and run
the kernel with modified ttgir directly by uncommenting [these lines](https://github.com/ROCm/triton/blob/d04c4cc1eb586b27176c1a8ae2a66c969f394613/third_party/amd/backend/compiler.py#L281-L297).

All IR dumps can be found at `triton/python/perf-kernels/study_4-stage/reg_pressure/`.


## Causal = False

command: 
`TRITON_HIP_SCALARIZE_VECTOR_FOPS=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_USE_ASYNC_COPY=1 python fa/flash-attention.py -model llama3-405B`

  | no assume | no assume rm-2ndFor | assume | assume rm-2ndFor
-- | -- | -- | -- | --
reg | 256 (27') | 256 (7') | 256 (21) | 256 (3)
perf | 734 | 780 | 813 | 817

Notes
- perf has unit of tflops. 
- reg has unit of vgpr count (vgpr spills). Prime (') means there are spills
inside the first loop.

## Causal = True

command: 
`TRITON_HIP_SCALARIZE_VECTOR_FOPS=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_USE_ASYNC_COPY=1 python fa/flash-attention.py -model llama3-405B -causal`

  | no assume | no assume rm-2ndFor | assume | assume rm-2ndFor
-- | -- | -- | -- | --
reg | 256 (32') | 256 (18') | 256 (22') | 256 (3)
perf | 515 | 481 | 692 | 764

## Things to investigate

The ultimate goal is to optimize the `causal_assume` case.

### `rm-2ndFor`

- [ ] Why does the 2nd loop increase vgpr usage a lot?

### `causal_assume` vs `nonCausal_assume`

Let's say we have `tl.assume` enabled to remove `if` from the epilogue. We also keep the 2nd loop.
The `causal_assume` and `nonCausal_assume` has almost the same spills (22 vs 21).
- [ ] Why in the `causal_assume` case, the spills happen inside the first loop? 
But in the `nonCausal_assume` case, the spills happen outside of the first loop?

### `assume` vs `no assume`

- [ ] why does `if` in the epilogue increase reg usage? This is true for both causal and non-causal cases.
- [ ] why does `if` in the epilogue introduce spill inside the loop? Check `nonCausal_nonAssume_rm-2ndFor`. 
It has only 7 spills, but some of them are inside the loop, which kills the perf.



# Performance report 7/20/2025

## Reproduce perf numbers with pre-installed llvm and triton
1. docker pull rocmshared/triton-attn-bench:iglp10
2. start a container (this example names the container as `test_triton_fav3`)
   ```bash
   sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME:/data --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864 --name test_triton_fav3 rocmshared/triton-attn-bench:iglp10
   ```
3. check status
  - `cd /var/lib/jenkins/triton && git log -1`
    ```bash
    commit 553023c924bf402b247071fdabf72e0ca71e2840 (HEAD -> shared/triton-gfx950-launch
    , origin/shared/triton-gfx950-launch)
    Author: root <root@asrock-126-009c.aus.dcgpu>
    Date:   Sat Jul 5 00:02:25 2025 +0000
    
        Add iglp(10)
    ```
  - `cd /var/lib/jenkins/llvm-project && git log -1`
     ```bash
     commit 0a49c1fa2cc72927ab31c852f05822bd8df4d95b (HEAD -> TritonInterleaveAndRematRebase0, 
     origin/TritonInterleaveAndRematRebase0)
     Author: Jeffrey Byrnes <Jeffrey.Byrnes@amd.com>
     Date:   Thu May 22 13:34:53 2025 -0700
     
         Auto select good flags
     
         Change-Id: I5aa9f71c403215205fb98d79d866453659a42661
     ```
4. batch benchmark
   ```bash
   cd /var/lib/jenkins/triton && ./bench.sh
   ```
5. Run a single config with causal = 0
   ```bash
   rm -rf ~/.triton/cache/
   TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 AMDGCN_SCALARIZE_PACKED_FOPS=1 python3 fa/flash-attention.py -d 128 -hq 64 -b 1 -sq 16384 -causal 0 -layout "bshd"
   ```
   Output
   ```bash
   fused-attention-fwd-d128-layoutbshd-causal0:
      BATCH    HQ    HK  N_CTX_Q  N_CTX_K      triton      torch
   0    1.0  64.0  64.0  16384.0  16384.0  924.917845  70.473362
   ```
   Triton perf is 924 tflops
   
   The dumped IR and ISA is saved in `python/perf-kernels/study_4-stage/MI350/perf_report_7-20-2025/b1-d128-h64-s16384-causal0_remat`
6. Run single config with causal = 1
   ```bash
   rm -rf ~/.triton/cache/
   TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 AMDGCN_SCALARIZE_PACKED_FOPS=1 python3 fa/flash-attention.py -d 128 -hq 64 -b 1 -sq 16384 -causal 1 -layout "bshd"
   ```
   Output
   ```bash
   fused-attention-fwd-d128-layoutbshd-causal1:
      BATCH    HQ    HK  N_CTX_Q  N_CTX_K      triton      torch
   0    1.0  64.0  64.0  16384.0  16384.0  650.500986  67.068967
   ```
   Triton perf is 650 tflops
   
   The dumped IR and ISA is saved in `python/perf-kernels/study_4-stage/MI350/perf_report_7-20-2025/b1-d128-h64-s16384-causal1_remat`

## Build triton with custom LLVM

Instruction can be found [here](https://github.com/triton-lang/triton/blob/main/README.md#building-with-a-custom-llvm).
The alternative method is tested.

## Examine IR and ISA

IR and ISA dump can be found in the cache: `~/.triton/cache/`.
To quickly locate the dump dir, use `find ~/.triton/cache/* -name attn_fwd.llir`



## FAv3 perf without Remat

The above perf numbers are from triton with a customized LLVM at https://github.com/jrbyrnes/llvm-project/commits/TritonInterleaveAndRematRebase0/

Here is the command to run triton without the above patch
```bash
cd /var/lib/jenkins/llvm-project/build
git checkout e12cbd8339b8956
ninja

cd /var/lib/jenkins/triton
git checkout e347b1418e0
export LLVM_BUILD_DIR=/var/lib/jenkins/llvm-project/build/
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR pip install -e .

rm -rf ~/.triton/cache/
TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 AMDGCN_SCALARIZE_PACKED_FOPS=1 python3 fa/flash-attention.py -d 128 -hq 64 -b 1 -sq 16384 -causal 0 -layout "bshd"
```
Output
```bash
fused-attention-fwd-d128-layoutbshd-causal0:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K      triton      torch
0    1.0  64.0  64.0  16384.0  16384.0  890.469697  70.490166
```
And the kernel has 2 spills.

The dumped IR and ISA is saved in `python/perf-kernels/study_4-stage/MI350/perf_report_7-20-2025/b1-d128-h64-s16384-causal0_e12cbd8339b`

Causal = 1
```bash
rm -rf ~/.triton/cache/
   TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 AMDGCN_SCALARIZE_PACKED_FOPS=1 python3 fa/flash-attention.py -d 128 -hq 64 -b 1 -sq 16384 -causal 1 -layout "bshd"
```
Output
```bash
fused-attention-fwd-d128-layoutbshd-causal1:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K      triton      torch
0    1.0  64.0  64.0  16384.0  16384.0  483.577941  70.490166
```
The kernel has 34 spills.

The dumped IR and ISA is saved in `python/perf-kernels/study_4-stage/MI350/perf_report_7-20-2025/b1-d128-h64-s16384-causal1_e12cbd8339b`

## Bottleneck

Let's focus on the kernel with causal=0 and remat llvm branch, i.e.
`python/perf-kernels/study_4-stage/MI350/perf_report_7-20-2025/b1-d128-h64-s16384-causal0_remat`

mfma efficiency inside the loop: 62%, which means the exposed cycles is about 66% of mfma cycles.
Now we want to have a breakdown of this 66% number.

### 1st cluster: compute DOT1 + VEC2

Without any interference from other waves, the 1st compute cluster takes 632 cycles.
in which mfma takes 512 cycles.
The exposed cycles (632-512 = 120) is about 11.7% of the mfma cycles.

There are 3 sources of the exposed cycles
1. There are 13 `v_mul` instructions not covered by mfma, which takes 52 cycles,
   i.e. 5% of mfma cycles.
2. There are 7 valu/salu instructions before the first mfma,
   which take up 28 cycles, i.e. 2.7% of mfma cycles.
3. There 7 valu/salu instructions from softmax that cannot be covered by mfma.
   And `s_nop` and `v_permlane32_swap` each takes 8 cycles.
   So there are 36 cycles exposed, i.e. 3.5% of mfma cycles.
   
Todo:
1. The exposed `v_mul` can be combined as `v_pk_mul`, 
   which can save 28 cycles, i.e. 2.7% of mfma cycles.
2. Investigate why `s_nop` and `v_permlane32_swap` each takes 8 cycles.
3. Investigate why there are `v_mov` and `s_mov` instructions before the first mfma
   and see if we can remove them.

### 2nd cluster: memory LRV + ACK

Memory cluster itself does not count into mfma efficiency.
However, there are valu/salu instructions in the memory cluster, which can
delay the execution of valu/salu instructions in the compute cluster.

1. There are 17 salu + 1 valu instructions at the beginning of the memory
   cluster, which is overlapped with the leading valu/salu instructions
   in the 1st cluster. 
   This makes the valu/salu instructions take 112, rather than 28 cycles,
   in the compute cluster, which leads to another 8.2% mfma cycles to be
   exposed.
2. There are 13 valu instructions for LRV. These will delay the valu instructions
   in the compute cluster, which leads to another 5.1% mfma cycles to be exposed.

Todo:
1. We need to try paddedShared layout + AsyncCopy to see if fewer instructions
   are needed to calculate the addresses.
   
### 3rd cluster: compute DOT2 + VEC1


1. There are 12 `v_exp` instructions that cannot be covered by mfma, which
   leads to 9.3% mfma cycles exposed.
2. 1 `v_setprio` before the 1st mfma
3. Some valu instructions are taking longer than expected.
   From the averaged cycles, there are 118 extra cycles, which is 11.5%
   exposed mfma cycles.
   Some of these cycles can be explained by the other memory cluster on
   the same SIMD. But some long cyles cannot.
   

Todo
1. There are 5 `s_waitcnt` instructions waiting for LDS.
   We can use one `s_waitcnt lgkmcnt(0)` at the beginning of the compute cluster
   to wait for all LDS instructions to finish.
   This can save 1.5% exposed mfma cycles.
2. Investigate why `s_nop` and `v_permlane32_swap` each takes 8 cycles.
3. Investigate why some valu instructions take too long to execute/issue.

### 4th cluster: memory LRK + ACV


1. There are 12 salu instructions for AsyncCopy instructions.
2. There are 8 `v_add3_u32` instructions to update the addresses of LRK.
   This is about 5% exposed mfma cycles.

Todo
1. PaddedShared layout should resolve both the above two problems


## Ablation Experiment

### Remove valu from memory cluster

The modified assembly code is saved as 
`study_4-stage/MI350/perf_report_7-20-2025/b1-d128-h64-s16384-causal0_remat/attn_fwd.s.rm-valu`

```bash
cd <triton dir> && git pull
AMD_INSERT_AMDGCN=[path to attn_fwd.s.rm-valu]   python3 fa/flash-attention.py -d 128 -hq 64 -b 1 -sq 16384 -causal 0 -layout "bshd"
```
Output
```bash
fused-attention-fwd-d128-layoutbshd-causal0:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K       triton      torch
0    1.0  64.0  64.0  16384.0  16384.0  1127.255706  29.873194
```
This version has 1127 tflops with mfma efficiency = 65%.
So this version must has a much higher freq. Need to confirm with agt.


# Update the compiler with PaddedSharedLayout

Commands:
- default, i.e. swizzle: `TRITON_HIP_USE_ASYNC_COPY=1 AMDGCN_SCALARIZE_PACKED_FOPS=1 python3 fa/flash-attention.py -d 128 -hq 64 -b 1 -sq 16384 -causal 0 -layout "bshd"`
- PaddedSharedLayout: + `TRITON_HIP_USE_PADDED_SHARED_LAYOUT=1`
- disable vector-combine pass (noVC): + `DISABLE_LLVM_OPT="disable-vector-combine"`
- customLLVM: insert iglp10 in the compute cluster
  - LLVM branch: https://github.com/kerbowa/llvm-project/tree/TritonInterleaveAndRematRebase2
- fixBasePtr refers to this commit: https://github.com/ROCm/triton/commit/5d8af4a6e2a70e7bcb17ea7c18367c36ad171bcb. More details can be found in https://github.com/ROCm/triton-internal/issues/1296


|                               | tflops | reg usage | mfma efficiency |
|-------------------------------|--------|-----------|-----------------|
| main swizzle                  | 1020   | 256 (2)   | 58.51%          |
| main swizzle noVC             | 1015   | 249       | 58.02%          |
| main swizzle customLLVM       | 1029   | 256 (1)   | 60.52%          |
| main swizzle customLLVM noVC  | 1038   | 245       | 62.82%          |
| main padded                   | 1023   | 256 (2)   | 58.53%          |
| main padded noVC              | 1021   | 249       | 58.02%          |
| main padded customLLVM        | 1024   | 256 (1)   | 60.51%          |
| main padded customLLVM noVC   | 1036   | 245       | 62.82%          |
| PR8398 padded customLLVM noVC | 1039   | 249       | 62.79%          |
| fixBasePtr                    | 1049   | 256       | 64.08%          |
| hack                          | 1107   |           | 71.7%           |
|                               |        |           |                 |

`AMD_INSERT_AMDGCN=/var/lib/jenkins/AMD-triton/python/perf-kernels/study_4-stage/MI350/MI355/IR/hack.s`

Command to collect trace
```
DISABLE_LLVM_OPT="disable-vector-combine" TRITON_HIP_USE_PADDED_SHARED_LAYOUT=1 TRITON_HIP_USE_ASYNC_COPY=1 AMDGCN_SCALARIZE_PACKED_FOPS=1 ROCPROF_ATT_LIBRARY_PATH=/var/lib/jenkins/att-decoder-v3-3.0.0-Linux/opt/rocm/lib/ rocprofv3 --att -i att.json -d /var/lib/jenkins/AMD-triton/python/perf-kernels/study_4-stage/MI350/MI355/fixBasePtr_padded_customLLVM_noVC -- python3 fa/flash-attention.py -d 128 -hq 64 -b 1 -sq 16384 -causal 0 -layout "bshd"
```

att.json:
```json
{
    "jobs": [
        {
        "kernel_include_regex": "attn_fwd",
        "kernel_exclude_regex": "",
        "kernel_iteration_range": "[10]",
        "advanced_thread_trace": true,
        "att_parse" : "trace",
        "att_target_cu" : 0,
        "att_shader_engine_mask" : "0xF",
        "att_simd_select": "0xF",
        "att_buffer_size": "0x60000000"
    }
    ]
}
```

Command to calculate mfma efficiency
```
./process_json.py MI350/MI355/fixBasePtr_padded_customLLVM_noVC/ui_output_agent_25932_dispatch_30/
```

Note that we have to disable the vector-combine pass otherwise the `mul` instruction used
update the acc will be moved from cluster 0 to cluster 2.



## Room for improvement

While LLVM team is rebasing the patch for mfma and valu interleaving, there are
a few things we can improve
- bank conflicts for K
  - old one
    ```
    #shared1 = #ttg.padded_shared<[512:+8] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [64, 0], [0, 32], [0, 1], [0, 2], [0, 4], [0, 8]], block = []}>
    ```
  - new one
    ```
    #shared1 = #ttg.padded_shared<[512:+8] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 1], [64, 0], [0, 32], [0, 2], [0, 4], [0, 8], [0, 16]], block = []}>
    ```
  - The issue is fixed by [37a9e809315a4](https://github.com/AlexAUT/triton/commit/37a9e809315a45c18e5d0ba5c20f4c416d8aafa2).
    The bank conflicts are gone. But perf is slightly worse compared to the new one above.
- 1st compute cluster
  - It starts with `lgkmcnt(0)`, which always takes 20 cycles?
  - The above `lgkmcnt(0)` is always followed by a 12-cycle NONE?
  - Then the 3rd instruction is mfma. The above two take 32 cycles, is it coincident?
    - If I move the above `lgkmcnt(0)` into the memory cluster, the NONE part takes 16 cycles ??
      This is better than 32 cycles, but why???
  - 2 `s_mov` + 2 x `v_mov` ?
  - `s_nop` is followed by `v_permlane32_swap`?
  - Long stall for some of the early valu instructions
- 4th mem cluster
  - transition from 3rd compute cluster (helps)
    - `v_exp` --> `setprio` --> `s_waitcnt` --> `barrier`.
      We can try to move the 2 `s_xx` instruction after the barrer so that the
      mfma in the compute cluster can start earlier.
    - 6 `v_add` instruction at the end of the loop
      - 216-219 are about `buffer_load` address update.
        Now it's always updating the vgpr address. We should update the sgpr address.
        ==> The pointer canonicalization pass can update the inc onto the base ptr.
      - 206 and 207 are coming from ttgir. There are redundant `addptr` ops for
        k and v addresses: one with linear layout, which is selected for asyncCopy
        for paddedSharedLayout, the other with blocked. It seems the layout progagation
        does not cross the boundary of the loop
- mem cluster
  - We should place lgkmcnt(0) at the end of memory cluster
