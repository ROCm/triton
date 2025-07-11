# Experiment to enable pingpong scheduling for kpack=2

The key difference between kpack=2 and kpack=1 is that when kpack=2, kWidth=8,
then the kDim for mfma_16x16 instruction is 32.
Therefore, the k dimension for the slice must be >= 32.
For tile size 256x256x64, each tensor can only be sliced into 2 slices
along the k dim.
However, pingpong requires 4 slices along the kdim.
To solve this problem, we try to slice along the M or N dim.

## Set up

- Triton compiler: 575bed88e90
- LLVM: 0ea4fb92648b2aa7 + revert https://github.com/llvm/llvm-project/pull/130776
- config:
  ```yaml
  - {'M': 8192, 'N': 8192, 'K': 8192, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu': 0, 'kpack': 2, 'matrix_instr_nonkdim': 16, 'schedule_hint': 'none'}
  ```

## Experiment steps

working dir: `python/perf-kernels/tools/tune_gemm/study_pingpong/kp2/enable_pingpong/`

1. Run the matmul kernel with the above config normally as
   ```bash
   ./tune_gemm.py --gemm_size_file gemm_config.yaml --compare_wo_tuning
   ```
2. Get the `ttgir` file from cache. Hack it to add cond.barrier before and after the forloop.
The result file is named `hack0.ttgir`.
3. Compile the kernel again with the hacked ttgir using the hack in
`make_ttgir` as shown [here](https://github.com/triton-lang/triton/compare/main...jungpark-mlir:triton:irmod).
4. Get the llir file from the cache. Hack it to implement the pingpong scheduling
inside the loop.
The hacked llvm ir file is named `rename_hack0.llir`.
5. Compile the kenrel again with the haked llir using the hack in `make_llir`
as shown [here](https://github.com/triton-lang/triton/compare/main...jungpark-mlir:triton:irmod).

The dumped IR and assembly code is saved in `IR_dump`.

The att output is saved in `att_output`.

One note about names of variables in llir.
The original llir file contains anonymous variable name, i.e. `%<number>`.
This is inconvenient if we want to reorder the instructions, 
since llvm ir parser expects the number of these variables to be ordered.
Therefore, we need to convert all anonymous variables to named variables.
One simple way to do so is convert `%<number>` into `%i<number>`.
This can be done by tools like `sed`, `grep`, or `awk`.
It can also be done by the `instnamer` pass as
```
opt -S --passes=instnamer input.llir -o output.llir
```

## Comparison between pingpong with kpack=1 and kpack=2

As a reference, the IR dump of pingpong with kpack=1 is saved in
`kp1_pingpong`.

### vgprs

|      | kpack=1  | kpack=2 |
| ---- | ------------- | ------------- |
| vgprs       | 232  | 230  |

### cluster structure

|      | kpack=1  | kpack=2 |
| ---- | ------------- | ------------- |
| cluster 0   | 4x`buffer_load_dwordx4` + 6x`ds_read2st64_b64`  | 4x`buffer_load_dwordx4` + 10x`ds_read_b128`  |
| cluster 1   | 32x`mfma_16x16`  | 32x`mfma_16x16`  |
| cluster 2   | 4x`buffer_load_dwordx4` + 6x`ds_read2st64_b64`  | 4x`buffer_load_dwordx4` + 2x`ds_read_b128`  |
| cluster 3   | 32x`mfma_16x16`  | 32x`mfma_16x16`  |
| cluster 4   | 12x`ds_read2st64_b64`  | 12x`ds_read_b128` |
| cluster 5   | 32x`mfma_16x16`  | 32x`mfma_16x16`  |
| cluster 6   | 10x`ds_write_b64` + 3x`ds_write2st64_b64`  | 8x`ds_write_b128` |
| cluster 7   | 32x`mfma_16x16`  | 32x`mfma_16x16`  |

### slices

Let's use mxn:tensor[off0, off1] to denote a m by n slice from tensor at offset [off0, off1].

|      | kpack=1  | kpack=2 |
| ---- | ------------- | ------------- |
| cluster 1   | A: 256x16:tensorA[0,0], B: 16x256:tensorB[0,0]  | A: 256x32:tensorA[0,0], B: 32x128:tensorB[0,0]  |
| cluster 3   | A: 256x16:tensorA[0,16], B: 16x256:tensorB[16,0]  | A: 256x32:tensorA[0,0], B: 32x128:tensorB[0,32]  |
| cluster 5   | A: 256x16:tensorA[0,32], B: 16x256:tensorB[32,0]  | A: 256x32:tensorA[0,32], B: 32x128:tensorB[32,0]  |
| cluster 7   | A: 256x16:tensorA[0,48], B: 16x256:tensorB[48,0]  | A: 256x32:tensorA[0,32], B: 32x128:tensorB[32,32]  |
