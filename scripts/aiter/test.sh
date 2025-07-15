#!/usr/bin/bash

rm -rf ~/.triton/cache/
test_fp16() {
TRITON_ALWAYS_COMPILE=1 \
  pytest ./op_tests/triton_tests/test_gemm_a16w16.py
}
test_fp16

#  TRITON_HIP_AGGREGATE_LOAD_FACTOR=2 \
# rocprof --stats
# head -n 2 results.stats.csv

################################################################
# Dump & Override Options
#  TRITON_KERNEL_DUMP=1 \
#  TRITON_DUMP_DIR=triton_dump_dir \
#  TRITON_KERNEL_OVERRIDE=1 \
#  TRITON_OVERRIDE_DIR=triton_override_dir \

################################################################
# Compilation Options
#  TRITON_HIP_USE_ASYNC_COPY=1 \
#  TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 \
#  TRITON_HIP_GLOBAL_PREFETCH=1 \
#  TRITON_HIP_LOCAL_PREFETCH=1 \

################################################################
# Problem Sizes
#  python bench_gemm_afp4wfp4.py --model llama3-405B -M 16 --metric bandwidth
#  python bench_gemm_afp4wfp4.py --model all -M 16 --metric bandwidth
#  python bench_gemm_afp4wfp4.py --shape 32 106496 16384 --metric bandwidth
#  python bench_gemm_afp4wfp4.py --shape 32 53248 16384 --metric bandwidth
