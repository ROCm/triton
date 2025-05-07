# Run auto-tune.
#HIP_VISIBLE_DEVICES=3 TRITON_MLIR_DUMP_REFINE_OPS=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn

# Baseline Upstream Perfomance - no causal
#TRITON_PRINT_AUTOTUNING=1 HIP_VISIBLE_DEVICES=3 TRITON_ALWAYS_COMPILE=1 python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd --dump-ir amdgcn

# Baseline Upstream Perfomance - w/ causal
#TRITON_PRINT_AUTOTUNING=1 HIP_VISIBLE_DEVICES=3 TRITON_ALWAYS_COMPILE=1 python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd --dump-ir amdgcn

# ROCm Performancea - no causal
#TRITON_PRINT_AUTOTUNING=1 HIP_VISIBLE_DEVICES=3 TRITON_MLIR_DUMP_REFINE_OPS=0 TRITON_ALWAYS_COMPILE=1 TRITON_HIP_STREAM_MAX_DEPTH=0 python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd --dump-ir amdgcn

# ROCm Performance - w/ causal
HIP_VISIBLE_DEVICES=3 \
  TRITON_PRINT_AUTOTUNING=1 \
  TRITON_ALWAYS_COMPILE=1 \
  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq  8192 -sk  8192 -d 128 -causal -layout thd -equal_seqlens && \
HIP_VISIBLE_DEVICES=3 \
  TRITON_PRINT_AUTOTUNING=1 \
  TRITON_ALWAYS_COMPILE=1 \
  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 32768 -sk 32768 -d 128 -causal -layout thd -equal_seqlens

# -causal -layout thd

