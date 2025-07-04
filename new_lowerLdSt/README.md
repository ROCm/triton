# Study the new LDS address calculation path

Experiment branches

- main branch: `9af26ee264d208d904`
- enable branch:  `enable_st.shared_AMD@fc76b317f5c54ea2`

Notes
- To disable pingpong with kpack=1, use `TRITON_HIP_USE_BLOCK_PINGPONG=0`

Commands:
```bash
cd python/perf-kernels/tools/tune_gemm/
./tune_gemm.py --gemm_size_file gemm_config.yaml --compare_wo_tuning
```

## MI300 results

### `ds_write`


