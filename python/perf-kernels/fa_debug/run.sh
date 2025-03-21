# Run auto-tune old FA.
#TRITON_PRINT_AUTOTUNING=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn

# Run auto-tune new FA.
#TRITON_PRINT_AUTOTUNING=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn -causal -layout thd

# Run normal.
# TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn

# Dump refined ttgir.
#TRITON_MLIR_DUMP_REFINE_OPS=0 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn
HIP_VISIBLE_DEVICES=3 TRITON_MLIR_DUMP_REFINE_OPS=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd --dump-ir amdgcn

# Run custom ttgir.
#TRITON_MLIR_INSERT_REFINE_OPS=ttgir_128x64/6.ttgir TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn
#TRITON_MLIR_INSERT_REFINE_OPS=ttgir_128x64/4.ttgir TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn

# -causal -layout thd

# debug single
#TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 --dump-ir amdgcn -causal -layout thd
