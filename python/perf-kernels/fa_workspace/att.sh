#!/usr/bin/bash

rm -rf triton_cache
rm -rf ~/.triton/cache

TRITON_MLIR_INSERT_REFINE_OPS=ttgir_128x64/4.ttgir TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml rocprofv2 \
  -d att_fa_4_1wps \
  -i att.txt \
  --plugin att auto \
  --mode file \
  ./att_exec.sh
