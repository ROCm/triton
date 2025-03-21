# Original Tests
HIP_VISIBLE_DEVICES=3 TRITON_MLIR_DUMP_REFINE_OPS=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config_test_16.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd &> out16.mlir &
HIP_VISIBLE_DEVICES=4 TRITON_MLIR_DUMP_REFINE_OPS=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=0 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config_test_32.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd &> out32.mlir &

#Tests failing even after fixes.
#HIP_VISIBLE_DEVICES=3 TRITON_MLIR_DUMP_REFINE_OPS=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config_test_16.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd &> out16.mlir &
#HIP_VISIBLE_DEVICES=4 TRITON_MLIR_DUMP_REFINE_OPS=1 TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 TRITON_HIP_STREAM_MAX_DEPTH=1 FA_CONFIG=./config_test_32.yaml python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd &> out32.mlir &
