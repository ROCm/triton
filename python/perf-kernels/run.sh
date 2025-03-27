#!/usr/bin/bash


MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_PRINT_AUTOTUNING=1 python3 ./flash-attention-new.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout bhsd

#python3 ./flash-attention.py -b 16 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 64 -causal -layout bhsd -persistent dynamic
# TRITON_PRINT_AUTOTUNING=1
