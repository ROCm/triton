rm -rf ~/.triton/cache/

# Run.
HIP_VISIBLE_DEVICES=3 \
  TRITON_ALWAYS_COMPILE=1 \
  MLIR_ENABLE_DUMP=0 \
  FA_CONFIG=./config32.yaml \
  TRITON_MLIR_INSERT_REFINE_OPS=ttgir_32/1.ttgir \
  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd -causal --dump-ir amdgcn &> out.mlir

#  TRITON_MLIR_INSERT_REFINE_OPS=ttgir_32/0.ttgir \
#  TRITON_MLIR_INSERT_SCHED_HINT_OPS=ttgir_32/sched_hint_thorough.ttgir \
#  TRITON_MLIR_DUMP_REFINE_OPS=1 \
#  TRITON_MLIR_INSERT_SCHED_HINT_OPS=ttgir_16/sched_hint_b64.ttgir \
#  TRITON_MLIR_DUMP_REFINE_OPS=1 \
#  TRITON_MLIR_DUMP_SCHED_HINT_OPS=1 \
#  TRITON_MLIR_INSERT_REFINE_OPS=ttgir/0.ttgir \
#  TRITON_MLIR_INSERT_SCHED_HINT_OPS=ttgir/sched_hint.ttgir \

#  TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 \


