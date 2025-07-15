#!/usr/bin/bash
rm -rf ~/.triton/cache/
SIZE=8192

#python op_tests/op_benchmarks/triton/bench_gemm_a16w16.py --shape $SIZE $SIZE $SIZE

python op_tests/op_benchmarks/triton/bench_extend_attention.py
python op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py
python op_tests/op_benchmarks/triton/bench_moe_mx.py
python op_tests/op_benchmarks/triton/bench_rope.py
python op_tests/op_benchmarks/triton/bench_mha.py
python op_tests/op_benchmarks/triton/bench_pa_decode.py
python op_tests/op_benchmarks/triton/bench_routing.py
python op_tests/op_benchmarks/triton/bench_gemm_a8w8.py
python op_tests/op_benchmarks/triton/bench_moe.py
python op_tests/op_benchmarks/triton/bench_pa_prefill.py
python op_tests/op_benchmarks/triton/bench_topk.py
python op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py
python op_tests/op_benchmarks/triton/bench_moe_align_block_size.py
python op_tests/op_benchmarks/triton/bench_rmsnorm.py
