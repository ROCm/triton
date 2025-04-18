rm -rf ~/.triton/cache/

# Run.
HIP_VISIBLE_DEVICES=3 \
  TRITON_ALWAYS_COMPILE=1 \
  MLIR_ENABLE_DUMP=1 \
  TRITON_MLIR_DUMP_REFINE_OPS=1 \
  TRITON_MLIR_DUMP_SCHED_HINT_OPS=0 \
  FA_CONFIG=./config.yaml \
  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd -causal --dump-ir amdgcn

#  TRITON_MLIR_INSERT_REFINE_OPS=1 \
#  TRITON_MLIR_INSERT_SCHED_HINT_OPS=1 \

#  TRITON_HIP_STREAM_MAX_DEPTH=0 \
#  TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 \
# Run custom ttgir.
#HIP_VISIBLE_DEVICES=5 TRITON_MLIR_INSERT_REFINE_OPS=ttgir/12.ttgir \
#  TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 FA_CONFIG=./config.yaml \
#  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd -causal --dump-ir amdgcn


