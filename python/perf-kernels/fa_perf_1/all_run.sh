# Run auto-tune.
#HIP_VISIBLE_DEVICES=3 TRITON_MLIR_DUMP_REFINE_OPS=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn

TRITON_PRINT_AUTOTUNING=1 HIP_VISIBLE_DEVICES=3 TRITON_MLIR_DUMP_REFINE_OPS=0 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd --dump-ir amdgcn

# -causal -layout thd

