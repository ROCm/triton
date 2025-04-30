#!/usr/bin/bash

rm -rf ~/.triton/cache/*

HIP_VISIBLE_DEVICES=5 \
  MLIR_ENABLE_DUMP=0 \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_PRINT_AUTOTUNING=1 \
  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 16384 -sk 16384 -d 128 -causal -layout thd 
#python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd 
#python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 32768 -sk 32768 -d 128 -causal -layout thd 
#python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 65536 -sk 65536 -d 128 -causal -layout thd 

#python3 ./flash-attention.py -b 16 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 64 -causal -layout bhsd -persistent dynamic
# TRITON_PRINT_AUTOTUNING=1
