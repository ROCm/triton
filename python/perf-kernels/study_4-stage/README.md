# FA 4-stage pingpong experiment

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


