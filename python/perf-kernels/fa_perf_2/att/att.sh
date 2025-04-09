#!/usr/bin/bash

rm -rf triton_cache
rm -rf ~/.triton/cache

HIP_VISIBLE_DEVICES=3 TRITON_MLIR_INSERT_REFINE_OPS=../ttgir/12.ttgir \
  TRITON_ALWAYS_COMPILE=1 TRITON_HIP_STREAM_MAX_DEPTH=0 FA_CONFIG=../config.yaml \
  rocprofv2 \
  -d fa2_12 \
  -i att.txt \
  --plugin att auto \
  --mode file \
  ./att_exec.sh

# 
