# Dump custom ttgir.
TRITON_HIP_STREAM_MAX_DEPTH=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_USE_IN_THREAD_TRANSPOSE=0 HIP_VISIBLE_DEVICES=3 TRITON_MLIR_DUMP_REFINE_OPS=1 TRITON_ALWAYS_COMPILE=1 FA_CONFIG=./config32.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd --dump-ir amdgcn

#TRITON_HIP_STREAM_MAX_DEPTH=2 

# Run custom ttgir.
#HIP_VISIBLE_DEVICES=3 TRITON_MLIR_INSERT_REFINE_OPS=ttgir/3.ttgir \
#  TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 FA_CONFIG=./config16.yaml \
#  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd --dump-ir amdgcn

# -causal -layout thd

# TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1
#        stream_max_depth = int(os.getenv("TRITON_HIP_STREAM_MAX_DEPTH", "1"))
#        global_prefetch = int(os.getenv("TRITON_HIP_GLOBAL_PREFETCH", "0"))
#        local_prefetch = int(os.getenv("TRITON_HIP_LOCAL_PREFETCH", "0"))
