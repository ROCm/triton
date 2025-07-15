#!/usr/bin/bash

rm -rf ~/.triton/cache/

run_mxfp4() {
TRITON_ALWAYS_COMPILE=1 \
  TRITON_PRINT_AUTOTUNING=1 \
  AMD_SERIALIZE_KERNEL=3 \
  TRITON_HIP_USE_ASYNC_COPY=1 \
  TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 \
  TRITON_HIP_ASYNC_FAST_SWIZZLE=1 \
  TRITON_HIP_ASYNC_COPY_OVERLAP=0 \
  rocprof --stats python op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py --shape 128 106496 16384 --metric bandwidth

head -n 2 results.stats.csv
}
run_mxfp4

#  rocprof --stats python op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py --shape 128 106496 16384 --metric bandwidth
#  TRITON_HIP_AGGREGATE_LOAD_FACTOR=0 \
# rocprof --stats 
#  TRITON_HIP_AGGREGATE_LOAD_FACTOR=2 \
#  TRITON_HIP_ASYNC_COPY_OVERLAP=0 \
# TRITON_HIP_AGGREGATE_LOAD_FACTOR=2
# AMD_SERIALIZE_KERNEL=0
# TRITON_HIP_USE_ASYNC_COPY=0

#  TRITON_HIP_ASYNC_COPY_OVERLAP=0 \
#  TRITON_HIP_USE_BLOCK_PINGPONG=1 \
#  TRITON_HIP_ASYNC_COPY_OVERLAP=0 \
#  TRITON_HIP_ASYNC_FAST_SWIZZLE=1 \
#  AMDGCN_USE_BUFFER_OPS=1 \
#  TRITON_HIP_USE_BLOCK_PINGPONG=0 \
#  TRITON_HIP_ASYNC_FAST_SWIZZLE=1 \
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
