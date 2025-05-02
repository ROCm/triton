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
The dumped IR is saved in `study_4-stage/pp-4s_unpack-mul/IR_dump`

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
