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
