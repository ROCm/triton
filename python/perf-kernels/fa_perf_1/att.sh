#!/usr/bin/bash

rm -rf triton_cache
rm -rf ~/.triton/cache

HIP_VISIBLE_DEVICES=3 TRITON_MLIR_INSERT_REFINE_OPS=ttgir/0.ttgir \
  TRITON_ALWAYS_COMPILE=1 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config16.yaml \
  rocprofv2 \
  -d fa1_0 \
  -i att.txt \
  --plugin att auto \
  --mode file \
  ./att_exec.sh

# 
