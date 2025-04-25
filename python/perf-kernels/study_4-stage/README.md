# Hack pingpong0 (pp0)

Experiment setup
- Triton compiler branch: https://github.com/AlexAUT/triton/commits/faPipelining/
- commit: acacb5fb605331239a6faf5a62c1be563b4f2ca7

Command
```bash
rm -rf ~/.triton/cache
TRITON_HIP_USE_ASYNC_COPY=1 python fa/flash-attention.py
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
- Reverse the priority of compute and memory cluster. Now mem cluster has higher prio than compute.
- remove gpu.barrier after mem cluster

Now I need to hack the assembly
- `hack_asm0.amdgcn`: Add s_barrier after mem cluster ==> 788 tflops
- `hack_asm1.amdgcn`: Manually breakdown some `v_pk_mul` ==> 800 tflops
  - If further remove all leftover `v_pk_mul` ==> 819 tflops
  
## bf16 input

`study_4-stage/hack_pp3_bf16.ttgir`

- Original run ==> 756
- `hack_asm0.amdgcn`: add s_barrier after memory cluster ==> 800
- `hack_asm1.amdgcn`: Manually breakdown some `v_pk_mul` ==> 830 tflops
  - If further remove all leftover `v_pk_mul` ==> 840 tflops

# Hack pingpong4 (pp4)

unroll the loop, does not work well
