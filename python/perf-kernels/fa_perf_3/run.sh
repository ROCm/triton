rm -rf ~/.triton/cache/

# Run.
foo_bar() {
HIP_VISIBLE_DEVICES=4 \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_PRINT_AUTOTUNING=1 \
  MLIR_ENABLE_DUMP=0 \
  FA_CONFIG=./config32.yaml \
  TRITON_MLIR_INSERT_REFINE_OPS=ttgir/19.ttgir \
  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 32768 -sk 32768 -d 128 -layout thd -causal -equal_seqlens --dump-ir amdgcn
}
foo_bar
foo_bar
foo_bar

#  TRITON_KERNEL_DUMP=1 \
#  TRITON_DUMP_DIR=triton_dump_dir \
#  TRITON_KERNEL_OVERRIDE=1 \
#  TRITON_OVERRIDE_DIR=triton_override_dir \

#  TRITON_MLIR_INSERT_REFINE_OPS=ttgir/no_sched_hint.ttgir \
# python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 65536 -sk 65536 -d 128 -layout thd -causal --dump-ir amdgcn
# python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 32768 -sk 32768 -d 128 -layout thd -causal --dump-ir amdgcn
# python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 16384 -sk 16384 -d 128 -causal -layout thd  --dump-ir amdgcn
# python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -layout thd -causal --dump-ir amdgcn

# FA_CONFIG=./config32.yaml \
# TRITON_MLIR_INSERT_REFINE_OPS=ttgir_32/13.ttgir \
#  TRITON_MLIR_INSERT_REFINE_OPS=ttgir_32/0.ttgir \
#  TRITON_MLIR_INSERT_SCHED_HINT_OPS=ttgir_32/sched_hint_thorough.ttgir \
#  TRITON_MLIR_DUMP_REFINE_OPS=1 \
#  TRITON_MLIR_INSERT_SCHED_HINT_OPS=ttgir_16/sched_hint_b64.ttgir \
#  TRITON_MLIR_DUMP_REFINE_OPS=1 \
#  TRITON_MLIR_DUMP_SCHED_HINT_OPS=1 \
#  TRITON_MLIR_INSERT_REFINE_OPS=ttgir/0.ttgir \
#  TRITON_MLIR_INSERT_SCHED_HINT_OPS=ttgir/sched_hint.ttgir \

#  TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1 \


