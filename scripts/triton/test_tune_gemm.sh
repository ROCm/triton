#!/bin/bash

rm -rf ~/.triton/cache/

run_test() {
TRITON_ALWAYS_COMPILE=1 \
  python python/perf-kernels/tools/tune_gemm/tune_gemm.py \
  --gemm_size_file test_config.yaml \
  --iters 1 \
  --compare_wo_tuning
}

for iter in {1..100}
do
  echo "iter: $iter"
  run_test
done

