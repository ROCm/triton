#!/bin/bash

python3 .././flash-attention.py \
  -b 2 -hq 16 -hk 16 -sq 32768 -sk 32768 -d 128 \
  -layout thd \
  -causal \
  -equal_seqlens \
  --dump-ir amdgcn

#  -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 \
