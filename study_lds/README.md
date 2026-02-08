# study `ds_read`

```
ROCPROF_ATT_LIBRARY_PATH=/root/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib/ rocprofv3 --att -i att_matmul.json -d ./study_lds/nW1_1024-32/att_output -- python study_lds/play_lds.py 
```


## 1 wave

Experiment design
- Simplified version of FA kernel, i.e. only keep th e1st dot.
- Load A directly into dotOperandLayout outside of the loop.
- Allocate LDS for B. Use B LDS buffer size to control occupancy and make sure
  there is only 1 workgroup per CU.
- B tile size: `BLOCK_K` x `BLOCK_N` = 128 x (`num_warps` x 256)
  - This is to make sure for difference number of warps, there are always
    32 x `ds_read_b128` inside the loop.

Padding parameters

| padding info | bank conflicts | cycles/ds_read | output         |
|--------------|----------------|----------------|----------------|
| 1024:+16     | 2              | 8              | `nW1_1024-16`  |
| 1024:+32     | 2              | 8              | `nW1_1024-32`  |
| 1024:+64     | 2              | 8              | `nW1_1024-64`  |
| 1024:+128    | 4              | 16             | `nW1_1024-128` |
| 1024:+256    | 8              | 32             | `nW1_1024-256` |
| 1024:+4      | not important  | 64             | `nW1_1024-4`   |

## 4 wave

Compared to 1-wave experiment, the 4-wave one
- uses 128x512 as the B tile size. Therefore, there are 16 `ds_read_b128`
  instructions to load dat for B.
  This is due to the limit of the offset field of `ds_read`.
  If we use 128x1024 tile size, we cannot use a single vgpr as the addr,
  since the offset field is not large enough to address data in the whole tensor.
  
Padding parameters

| padding info | bank conflicts | cycles/ds_read | output         |
|--------------|----------------|----------------|----------------|
| 1024:+16     | 2              | 32             | `nW4_1024-16`  |
| 1024:+32     | 2              | 32             | `nW4_1024-32`  |
| 1024:+64     | 2              | 32             | `nW4_1024-64`  |
| 1024:+128    | 4              | 64             | `nW4_1024-128` |
| 1024:+256    | 8              | 128            | `nW4_1024-256` |
| 1024:+4      | not important  | 256            | `nW4_1024-4`   |
