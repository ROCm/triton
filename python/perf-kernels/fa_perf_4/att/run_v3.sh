#!/usr/bin/bash


rm -rf ~/.triton/cache

HIP_VISIBLE_DEVICES=3 \
  ROCPROF_ATT_LIBRARY_PATH=~/system/att-decoder-v3-3.0.0-Linux/opt/rocm/lib \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_KERNEL_OVERRIDE=1 \
  TRITON_OVERRIDE_DIR=../triton_override_dir \
  FA_CONFIG=../config.yaml \
  rocprofv3 \
  -d fa4asm_check \
  -i att.json \
  --advanced-thread-trace --att-parse trace \
  -- \
  ./att_exec.sh

#  -i att.txt \
# 
#  TRITON_MLIR_INSERT_REFINE_OPS=../ttgir/7.ttgir \
