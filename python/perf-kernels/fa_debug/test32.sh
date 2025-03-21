#!/usr/bin/bash

#TRITON_PRINT_AUTOTUNING=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 TRITON_HIP_STREAM_MAX_DEPTH=1 pytest flash-attention.py 

HIP_VISIBLE_DEVICES=3 TRITON_ALWAYS_COMPILE=1 FA_CONFIG=config_test_32.yaml pytest flash-attention.py
#TRITON_ALWAYS_COMPILE=1 pytest flash-attention.py
