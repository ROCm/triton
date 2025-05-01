rm -rf ~/.triton/cache/


HIP_VISIBLE_DEVICES=5 \
  TRITON_PRINT_AUTOTUNING=1 \
  MLIR_ENABLE_DUMP=0 \
  FA_CONFIG=./config.yaml \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_KERNEL_OVERRIDE=1 \
  TRITON_OVERRIDE_DIR=triton_override_dir \
  python3 ~/repos/rocm_triton/python/perf-kernels/tools/rocm-triton-prof/rocm-triton-prof.py --kernel attn_fwd --cmd \
  python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 32768 -sk 32768 -d 128 -layout thd -causal --dump-ir amdgcn
