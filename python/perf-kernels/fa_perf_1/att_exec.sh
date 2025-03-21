#!/bin/bash

HIP_VISIBLE_DEVICES=3 \
  TRITON_MLIR_INSERT_REFINE_OPS=ttgir/0.ttgir \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_HIP_STREAM_MAX_DEPTH=1 \
  FA_CONFIG=./config32.yaml \
  python3 ./flash-attention.py \
  -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 \
  -layout thd \
  --dump-ir amdgcn

# HIP_VISIBLE_DEVICES=3 \
