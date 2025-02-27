#!/bin/bash

TRITON_MLIR_INSERT_REFINE_OPS=ttgir_128x64/4.ttgir \
  TRITON_ALWAYS_COMPILE=1 \
  MLIR_ENABLE_DUMP=0 \
  TRITON_HIP_STREAM_MAX_DEPTH=1 \
  FA_CONFIG=./config.yaml \
  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 1024 -sk 8192 -d 128 --dump-ir amdgcn
