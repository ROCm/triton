#!/usr/bin/bash

rm -rf ~/.triton/cache

HIP_VISIBLE_DEVICES=3 \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_KERNEL_OVERRIDE=1 \
  TRITON_OVERRIDE_DIR=../triton_override_dir \
  FA_CONFIG=../config.yaml \
  rocprofv2 \
  -d fa4asm_5 \
  -i att.txt \
  --plugin att auto \
  --mode file \
  ./att_exec.sh

# 
#  TRITON_MLIR_INSERT_REFINE_OPS=../ttgir/7.ttgir \
