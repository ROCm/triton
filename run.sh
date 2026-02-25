#!/usr/bin/bash
rm -rf ~/.triton/cache/

#FILE=python/tutorials/03-matrix-multiplication.py
FILE=matmul.py
#FILE=fp8_matmul.py

TRITON_ALWAYS_COMPILE=1 \
  TRITON_PRINT_AUTOTUNING=1 \
  TRITON_HIP_USE_LDS_PREFETCH=1 \
  TRITON_HIP_PREFETCH_INSERT_SCHED_BARRIER=1 \
  TRITON_LLVM_DEBUG_ONLY=tritongpu-prefetch \
  python3 $FILE
