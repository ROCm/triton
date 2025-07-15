#!/usr/bin/bash

rm -rf ~/.triton/cache/

SIZE=8192

TRITON_ALWAYS_COMPILE=1 \
  python op_tests/op_benchmarks/triton/bench_gemm_a16w16.py --shape $SIZE $SIZE $SIZE

#  python op_tests/op_benchmarks/triton/bench_gemm_a16w16.py --shape 1024 1024 1024
