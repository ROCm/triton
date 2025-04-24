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
  - Fix teh extra barrier between ACK[3] and LRV[0]
  - Apply [#781](https://github.com/ROCm/triton/pull/781) to flash-attention.py
    to fold sub into fma
    
Hacked ttgir is saved as `study_4-stage/hack_pp1.ttgir`.
The IR dumps from this hacked ttgir is saved in `study_4-stage/hack_pp1/IR_dump/`.

