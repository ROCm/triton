#!/usr/bin/bash

rm -rf triton_cache
rm -rf ~/.triton/cache

HIP_VISIBLE_DEVICES=3 TRITON_MLIR_INSERT_REFINE_OPS=ttgir_128x32_32/1.ttgir \
  TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config32.yaml \
  rocprofv2 \
  -d fa_128x32_32_novpk \
  -i att.txt \
  --plugin att auto \
  --mode file \
  ./att_exec.sh

# 
