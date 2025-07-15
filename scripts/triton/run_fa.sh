#!/bin/bash

rm -rf ~/.triton/cache/
rm -rf python/perf-kernels/output/

python3 ./python/perf-kernels/flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd -equal_seqlens

#  --benchmark
#  TRITON_HIP_GLOBAL_PREFETCH=1 \
#  TRITON_HIP_LOCAL_PREFETCH=1 \
