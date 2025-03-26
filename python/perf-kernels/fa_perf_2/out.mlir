loc("ttgir/0.ttgir":240:36): error: use of value '%131' expects different type than prior uses: 'tensor<64x128xi32, #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>' vs 'tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>'
INFO: running a single config given from a file
Traceback (most recent call last):
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_perf_2/./flash-attention.py", line 2198, in <module>
    sys.exit(main())
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_perf_2/./flash-attention.py", line 2193, in main
    run_benchmark(custom_config, args)
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_perf_2/./flash-attention.py", line 2115, in run_benchmark
    bench_flash_attention.run(save_path=".", print_data=True, show_plots=True)
  File "/home/dtanner/repos/triton/python/triton/testing.py", line 388, in run
    result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))
  File "/home/dtanner/repos/triton/python/triton/testing.py", line 335, in _run
    ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_perf_2/./flash-attention.py", line 2106, in bench_flash_attention
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
  File "/home/dtanner/repos/triton/python/triton/testing.py", line 145, in do_bench
    fn()
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_perf_2/./flash-attention.py", line 2082, in <lambda>
    fn = lambda: attention(q, k, v, o, input_metadata)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/function.py", line 598, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_perf_2/./flash-attention.py", line 1204, in forward
    handle = attn_fwd[grid](q, k, v, metadata.bias, metadata.sm_scale, M, o, *q_strides, *k_strides, *v_strides, *o_strides,
  File "/home/dtanner/repos/triton/python/triton/runtime/jit.py", line 368, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/dtanner/repos/triton/python/triton/runtime/autotuner.py", line 206, in run
    ret = self.fn.run(
  File "/home/dtanner/repos/triton/python/triton/runtime/jit.py", line 608, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
  File "/home/dtanner/repos/triton/python/triton/compiler/compiler.py", line 285, in compile
    next_module = compile_ir(module, metadata)
  File "/home/dtanner/repos/triton/python/triton/backends/amd/compiler.py", line 451, in <lambda>
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
  File "/home/dtanner/repos/triton/python/triton/backends/amd/compiler.py", line 310, in make_llir
    new_mod = ir.parse_mlir_module(insert_module_path, mod.context)
RuntimeError: Parse MLIR file failed.
