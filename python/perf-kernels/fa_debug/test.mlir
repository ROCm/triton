============================= test session starts ==============================
platform linux -- Python 3.10.15, pytest-7.3.2, pluggy-1.5.0
rootdir: /home/dtanner/repos/rocm_triton/python
plugins: hypothesis-5.35.1, xdist-3.3.1, xdoctest-1.1.0, rerunfailures-14.0, flakefinder-1.1.0, cpp-2.3.0, typeguard-4.3.0
collected 586 items

flash-attention.py FF

=================================== FAILURES ===================================
_______________ test_op_fwd[bshd-True-True-4-48-24-1024-1024-64] _______________

Z = 4, HQ = 48, HK = 24, N_CTX_Q = 1024, N_CTX_K = 1024, D_HEAD = 64
causal = True, use_alibi = True, layout = 'bshd', dtype = torch.float16

    @pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
        (4, 48, 24, 1024, 1024, 64),
        (1, 24, 6, 8192, 8192, 64),
        (1, 4, 2, 16384, 16384, 128),
        (2, 16, 4, 1020, 987, 128),
        (2, 16, 4, 15498, 2, 128),
        (2, 16, 2, 7, 16219, 64),
        (4, 48, 12, 1, 1, 64),
        (4, 48, 48, 1, 1, 128),
        (4, 48, 24, 3, 3, 128),
        (4, 48, 48, 1001, 990, 64),
        (1, 8, 8, 8081, 7099, 64),
        (1, 4, 4, 16330, 15989, 128),
        (4, 4, 1, 1024, 1024, 33),
        (4, 4, 2, 65, 1018, 65),
        (4, 4, 4, 128, 128, 65),
        (4, 4, 4, 113, 123, 1),
    ])
    @pytest.mark.parametrize('causal', [True, False])
    @pytest.mark.parametrize('use_alibi', [True, False])
    @pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
    def test_op_fwd(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_alibi, layout, dtype=torch.float16):
        torch.manual_seed(20)
        q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
        if causal:
            input_metadata.need_causal()
    
        if use_alibi:
            # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
            alibi_slopes = torch.tensor([2**(-8 / HQ * i) for i in range(1, HQ + 1)], dtype=torch.float32,
                                        device="cuda").repeat(Z, 1)
            input_metadata.need_alibi(alibi_slopes, Z, HQ)
        else:
            alibi_slopes = None
    
        o = torch.empty_like(q)
    
        # triton implementation
>       tri_out, _, _ = attention(q, k, v, o, input_metadata)

flash-attention.py:1480: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/function.py:598: in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
flash-attention.py:1221: in forward
    handle = attn_fwd[grid](q, k, v, metadata.bias, metadata.sm_scale, M, o, *q_strides, *k_strides, *v_strides, *o_strides,
../../../../triton/python/triton/runtime/jit.py:368: in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
../../../../triton/python/triton/runtime/autotuner.py:206: in run
    ret = self.fn.run(
../../../../triton/python/triton/runtime/jit.py:608: in run
    kernel = self.compile(src, target=target, options=options.__dict__)
../../../../triton/python/triton/compiler/compiler.py:284: in compile
    next_module = compile_ir(module, metadata)
../../../../triton/python/triton/backends/amd/compiler.py:438: in <lambda>
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

src = <triton._C.libtriton.ir.module object at 0x7f90e2cfbc40>
metadata = {'allow_flush_denorm': False, 'allowed_dot_input_precisions': ('ieee', 'tf32'), 'arch': 'gfx942', 'backend_name': 'hip', ...}
options = HIPOptions(num_warps=4, waves_per_eu=2, num_stages=2, num_ctas=1, extern_libs=(('ocml', '/home/dtanner/repos/triton/py... allow_flush_denorm=False, max_num_imprecise_acc_default=0, backend_name='hip', instruction_sched_variant='refine_ops')

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_decompose_unsupported_conversions(pm, options.arch)
        # custom_lds_size is an experimental parameter that defines amount of LDS available
        # for one thread block. Measured in bytes.
        #
        # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
        # LDS size is determined by provided arch name.
        custom_lds_size = 0
        amd.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
    
        passes.ttgpuir.add_allocate_shared_memory(pm)
        amd.passes.ttgpuir.add_membar_analysis(pm)
        amd.passes.ttgpuir.add_refine_amdgpu_ops(pm, options.arch)
    
        # TODO - remove
        passes.common.add_canonicalizer(pm)
>       pm.run(mod)
E       RuntimeError: PassManager::run failed

../../../../triton/python/triton/backends/amd/compiler.py:296: RuntimeError
----------------------------- Captured stderr call -----------------------------
[tritonamdgpu-refine-ops]: refinedShape: 16x32
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %130 = ttg.memdesc_subview %125[%c0_i32_19, %c0_i32_20] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %132 = ttg.memdesc_subview %125[%c16_i32_21, %c0_i32_22] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %134 = ttg.memdesc_subview %125[%c32_i32, %c0_i32_23] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %136 = ttg.memdesc_subview %125[%c48_i32, %c0_i32_24] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %138 = ttg.memdesc_subview %125[%c0_i32_25, %c32_i32_26] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %140 = ttg.memdesc_subview %125[%c16_i32_27, %c32_i32_28] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %142 = ttg.memdesc_subview %125[%c32_i32_29, %c32_i32_30] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %144 = ttg.memdesc_subview %125[%c48_i32_31, %c32_i32_32] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: ConcatOp: %146 = amdgpu.concat %131, %133, %135, %137, %139, %141, %143, %145 [4, 2] : tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-refine-ops]: refinedShape: 8x32
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %179 = ttg.memdesc_subview %177[%c0_i32_33, %c0_i32_34] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %181 = ttg.memdesc_subview %177[%c8_i32, %c0_i32_35] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %183 = ttg.memdesc_subview %177[%c16_i32_36, %c0_i32_37] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %185 = ttg.memdesc_subview %177[%c24_i32, %c0_i32_38] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %187 = ttg.memdesc_subview %177[%c32_i32_39, %c0_i32_40] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %189 = ttg.memdesc_subview %177[%c40_i32, %c0_i32_41] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %191 = ttg.memdesc_subview %177[%c48_i32_42, %c0_i32_43] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %193 = ttg.memdesc_subview %177[%c56_i32, %c0_i32_44] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %195 = ttg.memdesc_subview %177[%c0_i32_45, %c32_i32_46] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %197 = ttg.memdesc_subview %177[%c8_i32_47, %c32_i32_48] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %199 = ttg.memdesc_subview %177[%c16_i32_49, %c32_i32_50] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %201 = ttg.memdesc_subview %177[%c24_i32_51, %c32_i32_52] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %203 = ttg.memdesc_subview %177[%c32_i32_53, %c32_i32_54] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %205 = ttg.memdesc_subview %177[%c40_i32_55, %c32_i32_56] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %207 = ttg.memdesc_subview %177[%c48_i32_57, %c32_i32_58] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %209 = ttg.memdesc_subview %177[%c56_i32_59, %c32_i32_60] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: ConcatOp: %211 = amdgpu.concat %180, %182, %184, %186, %188, %190, %192, %194, %196, %198, %200, %202, %204, %206, %208, %210 [8, 2] : tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
[tritonamdgpu-refine-ops]: totalReps: 1x2x4
[tritonamdgpu-refine-ops]: repsPerDotTile: 1x1x1
[tritonamdgpu-refine-ops]: local_load_a[0][0] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][1] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][2] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][3] extract_slice
[tritonamdgpu-refine-ops]: local_load_b[0][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[0][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[1][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[1][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[2][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[2][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[3][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[3][1] extact_slice
[tritonamdgpu-refine-ops]: dot-tile[0]
[tritonamdgpu-refine-ops]:   dot[0][0][0]
[tritonamdgpu-refine-ops]: dot-tile[1]
[tritonamdgpu-refine-ops]:   dot[0][1][0]
[tritonamdgpu-refine-ops]: dot-tile[2]
[tritonamdgpu-refine-ops]:   dot[0][0][1]
[tritonamdgpu-refine-ops]: dot-tile[3]
[tritonamdgpu-refine-ops]:   dot[0][1][1]
[tritonamdgpu-refine-ops]: dot-tile[4]
[tritonamdgpu-refine-ops]:   dot[0][0][2]
[tritonamdgpu-refine-ops]: dot-tile[5]
[tritonamdgpu-refine-ops]:   dot[0][1][2]
[tritonamdgpu-refine-ops]: dot-tile[6]
[tritonamdgpu-refine-ops]:   dot[0][0][3]
[tritonamdgpu-refine-ops]: dot-tile[7]
[tritonamdgpu-refine-ops]:   dot[0][1][3]
[tritonamdgpu-refine-ops]: totalReps: 1x2x8
[tritonamdgpu-refine-ops]: repsPerDotTile: 1x2x1
[tritonamdgpu-refine-ops]: local_load_a[0][0] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][1] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][2] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][3] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][4] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][5] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][6] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][7] extract_slice
[tritonamdgpu-refine-ops]: local_load_b[0][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[0][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[1][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[1][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[2][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[2][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[3][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[3][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[4][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[4][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[5][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[5][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[6][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[6][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[7][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[7][1] extact_slice
[tritonamdgpu-refine-ops]: dot-tile[0]
[tritonamdgpu-refine-ops]:   dot[0][0][0]
[tritonamdgpu-refine-ops]:   dot[0][1][0]
[tritonamdgpu-refine-ops]: dot-tile[1]
[tritonamdgpu-refine-ops]:   dot[0][0][1]
[tritonamdgpu-refine-ops]:   dot[0][1][1]
[tritonamdgpu-refine-ops]: dot-tile[2]
[tritonamdgpu-refine-ops]:   dot[0][0][2]
[tritonamdgpu-refine-ops]:   dot[0][1][2]
[tritonamdgpu-refine-ops]: dot-tile[3]
[tritonamdgpu-refine-ops]:   dot[0][0][3]
[tritonamdgpu-refine-ops]:   dot[0][1][3]
[tritonamdgpu-refine-ops]: dot-tile[4]
[tritonamdgpu-refine-ops]:   dot[0][0][4]
[tritonamdgpu-refine-ops]:   dot[0][1][4]
[tritonamdgpu-refine-ops]: dot-tile[5]
[tritonamdgpu-refine-ops]:   dot[0][0][5]
[tritonamdgpu-refine-ops]:   dot[0][1][5]
[tritonamdgpu-refine-ops]: dot-tile[6]
[tritonamdgpu-refine-ops]:   dot[0][0][6]
[tritonamdgpu-refine-ops]:   dot[0][1][6]
[tritonamdgpu-refine-ops]: dot-tile[7]
[tritonamdgpu-refine-ops]:   dot[0][0][7]
[tritonamdgpu-refine-ops]:   dot[0][1][7]
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %200 = math.exp2 %199 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
srcShape: 128x64
srcShapePerCtaTile: 128x64
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
numReps: 1x1
[tritonamdgpu-refine-ops]: failed to refine binary op: %200 = math.exp2 %199 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %205 = math.exp2 %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
getRefinedShape: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
inDimNames
register: bases: 1
lane: bases: 32
warp: bases: 4
block: bases: 1
outDimNames
dim0
getRefinedShape(slice/mfma)
llShape: 128
mfmaShape: 128x32
retShape: 128
srcShape: 128
srcShapePerCtaTile: 128
getRefinedShape: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
inDimNames
register: bases: 1
lane: bases: 32
warp: bases: 4
block: bases: 1
outDimNames
dim0
getRefinedShape(slice/mfma)
llShape: 128
mfmaShape: 128x32
retShape: 128
numReps: 1
[tritonamdgpu-refine-ops]: failed to refine binary op: %205 = math.exp2 %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %211 = arith.truncf %200 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
srcShape: 128x64
srcShapePerCtaTile: 128x64
getRefinedShape: tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
numReps: 1x1
[tritonamdgpu-refine-ops]: failed to refine binary op: %211 = arith.truncf %200 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %189 = arith.sitofp %188 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
getRefinedShape: tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
srcShape: 128x64
srcShapePerCtaTile: 128x64
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
numReps: 1x1
[tritonamdgpu-refine-ops]: failed to refine binary op: %189 = arith.sitofp %188 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %187 = arith.subi %107, %186 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
getRefinedShape: tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
srcShape: 128x64
srcShapePerCtaTile: 128x64
getRefinedShape: tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
getRefinedShape: tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
numReps: 1x1
[tritonamdgpu-refine-ops]: failed to refine binary op: %187 = arith.subi %107, %186 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %184 = arith.addi %183, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
getRefinedShape: tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
inDimNames
register: bases: 32
lane: bases: 2
warp: bases: 1
block: bases: 1
outDimNames
dim0
getRefinedShape(slice/mfma)
python: /home/dtanner/.triton/llvm/llvm-2619c2ed-ubuntu-x64/include/llvm/ADT/SmallVector.h:291: reference llvm::SmallVectorTemplateCommon<unsigned int>::operator[](size_type) [T = unsigned int]: Assertion `idx < size()' failed.
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: f32, %arg23: i32, %arg24: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.44269502> : tensor<128x64xf32, #mma>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #mma>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128x64xf32, #mma>
    %cst_2 = arith.constant dense<1024> : tensor<1x64xi32, #blocked>
    %cst_3 = arith.constant dense<1024> : tensor<64x1xi32, #blocked1>
    %cst_4 = arith.constant dense<0.180336878> : tensor<128x64xf32, #mma>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #mma>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #mma>
    %cst_8 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_9 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %c16_i32 = arith.constant 16 : i32
    %cst_10 = arith.constant -1.000000e+00 : f32
    %c63_i32 = arith.constant 63 : i32
    %c1152_i32 = arith.constant 1152 : i32
    %c49152_i32 = arith.constant 49152 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_11 = arith.constant dense<1024> : tensor<128xi32, #blocked2>
    %cst_12 = arith.constant dense<0x7F800000> : tensor<128xf32, #blocked2>
    %cst_13 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked1>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst_16 = arith.constant dense<1024> : tensor<128x1xi32, #blocked1>
    %cst_17 = arith.constant dense<1024> : tensor<128x1xi32, #mma>
    %cst_18 = arith.constant dense<true> : tensor<128x64xi1, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %0, %c128_i32 : i32
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %7 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %8 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %9 = tt.splat %3 : i32 -> tensor<128xi32, #blocked2>
    %10 = arith.addi %7, %4 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %11 = arith.addi %8, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %12 = arith.addi %9, %6 : tensor<128xi32, #blocked2>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = arith.addi %0, %c1_i32 : i32
    %17 = arith.muli %16, %c128_i32 : i32
    %18 = arith.addi %17, %c63_i32 : i32
    %19 = arith.divsi %18, %c64_i32 : i32
    %20 = arith.minsi %19, %c16_i32 : i32
    %21 = arith.cmpi sle, %20, %c0_i32 : i32
    scf.while (%arg27 = %c0_i32) : (i32) -> () {
      %22 = arith.cmpi slt, %arg27, %c1_i32 : i32
      scf.condition(%22)
    } do {
      scf.if %21 {
        %22 = arith.muli %2, %arg14 : i32
        %23 = tt.addptr %arg4, %22 : !tt.ptr<f16>, i32
        %24 = arith.muli %1, %arg15 : i32
        %25 = tt.addptr %23, %24 : !tt.ptr<f16>, i32
        %26 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
        %27 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #blocked1>
        %28 = arith.muli %26, %27 : tensor<128x1xi32, #blocked1>
        %29 = tt.splat %25 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
        %30 = tt.addptr %29, %28 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
        %31 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %32 = tt.expand_dims %31 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
        %33 = tt.broadcast %30 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
        %34 = tt.broadcast %32 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %35 = tt.addptr %33, %34 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %36 = arith.cmpi slt, %26, %cst_16 : tensor<128x1xi32, #blocked1>
        %37 = tt.broadcast %36 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        tt.store %35, %cst_15, %37 : tensor<128x64x!tt.ptr<f16>, #blocked1>
        %38 = arith.muli %2, %c49152_i32 : i32
        %39 = tt.addptr %arg3, %38 : !tt.ptr<f32>, i32
        %40 = arith.muli %1, %c1024_i32 : i32
        %41 = tt.addptr %39, %40 : !tt.ptr<f32>, i32
        %42 = tt.splat %41 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
        %43 = tt.addptr %42, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %44 = arith.cmpi slt, %12, %cst_11 : tensor<128xi32, #blocked2>
        tt.store %43, %cst_12, %44 : tensor<128x!tt.ptr<f32>, #blocked2>
      } else {
        %22 = arith.divsi %1, %c2_i32 : i32
        %23 = arith.muli %2, %arg5 : i32
        %24 = tt.addptr %arg0, %23 : !tt.ptr<f16>, i32
        %25 = arith.muli %1, %arg6 : i32
        %26 = tt.addptr %24, %25 : !tt.ptr<f16>, i32
        %27 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
        %28 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
        %29 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked1>
        %30 = arith.muli %28, %29 : tensor<128x1xi32, #blocked1>
        %31 = tt.splat %26 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
        %32 = tt.addptr %31, %30 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
        %33 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %34 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
        %35 = tt.expand_dims %33 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
        %36 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
        %37 = tt.broadcast %32 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
        %38 = tt.broadcast %34 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
        %39 = tt.broadcast %35 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %40 = tt.addptr %37, %39 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %41 = arith.muli %2, %arg8 : i32
        %42 = tt.addptr %arg1, %41 : !tt.ptr<f16>, i32
        %43 = arith.muli %22, %arg9 : i32
        %44 = tt.addptr %42, %43 : !tt.ptr<f16>, i32
        %45 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %46 = tt.expand_dims %45 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %47 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
        %48 = tt.splat %44 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
        %49 = tt.addptr %48, %46 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
        %50 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #blocked>
        %51 = arith.muli %36, %50 : tensor<1x64xi32, #blocked>
        %52 = tt.broadcast %49 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #blocked>
        %53 = tt.broadcast %51 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
        %54 = tt.addptr %52, %53 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
        %55 = arith.muli %2, %arg11 : i32
        %56 = tt.addptr %arg2, %55 : !tt.ptr<f16>, i32
        %57 = arith.muli %22, %arg12 : i32
        %58 = tt.addptr %56, %57 : !tt.ptr<f16>, i32
        %59 = tt.splat %arg13 : i32 -> tensor<64x1xi32, #blocked1>
        %60 = arith.muli %47, %59 : tensor<64x1xi32, #blocked1>
        %61 = tt.splat %58 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1>
        %62 = tt.addptr %61, %60 : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1>
        %63 = tt.broadcast %62 : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x64x!tt.ptr<f16>, #blocked1>
        %64 = tt.broadcast %35 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
        %65 = tt.addptr %63, %64 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
        %66 = arith.muli %2, %arg21 : i32
        %67 = arith.addi %66, %1 : i32
        %68 = tt.addptr %arg26, %67 : !tt.ptr<f32>, i32
        %69 = tt.load %68 : !tt.ptr<f32>
        %70 = arith.cmpi slt, %27, %cst_17 : tensor<128x1xi32, #mma>
        %71 = arith.cmpi slt, %28, %cst_16 : tensor<128x1xi32, #blocked1>
        %72 = tt.broadcast %70 : tensor<128x1xi1, #mma> -> tensor<128x64xi1, #mma>
        %73 = tt.broadcast %71 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %74 = tt.load %40, %73, %cst_15 : tensor<128x64x!tt.ptr<f16>, #blocked1>
        %75 = ttg.local_alloc %74 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %76 = ttg.local_load %75 : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
        %77 = arith.minsi %20, %c2_i32 : i32
        %78 = arith.subi %20, %77 : i32
        %79 = arith.muli %20, %c64_i32 : i32
        %80 = arith.cmpi sgt, %78, %c0_i32 : i32
        %81:4 = scf.if %80 -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>, i32) {
          %112 = arith.muli %78, %c64_i32 : i32
          %113 = tt.broadcast %27 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
          %114 = arith.mulf %69, %cst_10 : f32
          %115 = tt.splat %114 : f32 -> tensor<128x64xf32, #mma>
          %116 = arith.muli %arg10, %c64_i32 : i32
          %117 = tt.splat %116 : i32 -> tensor<64x64xi32, #blocked>
          %118 = arith.muli %arg13, %c64_i32 : i32
          %119 = tt.splat %118 : i32 -> tensor<64x64xi32, #blocked1>
          %120 = arith.cmpi sgt, %112, %c0_i32 : i32
          %121 = tt.splat %120 : i1 -> tensor<64x64xi1, #blocked>
          %122 = tt.load %54, %121 : tensor<64x64x!tt.ptr<f16>, #blocked>
          %123 = ttg.local_alloc %122 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
          %124 = arith.subi %112, %c64_i32 : i32
          %125:6 = scf.for %arg27 = %c0_i32 to %124 step %c64_i32 iter_args(%arg28 = %cst_5, %arg29 = %cst_8, %arg30 = %cst_9, %arg31 = %54, %arg32 = %65, %arg33 = %123) -> (tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>)  : i32 {
            amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
            %170 = tt.addptr %arg31, %117 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
            %171 = tt.load %170 : tensor<64x64x!tt.ptr<f16>, #blocked>
            %172 = ttg.local_load %arg33 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
            %173 = tt.load %arg32 : tensor<64x64x!tt.ptr<f16>, #blocked1>
            %174 = tt.dot %76, %172, %cst_5, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
            %175 = arith.mulf %174, %cst_4 : tensor<128x64xf32, #mma>
            %176 = arith.addf %175, %cst_5 : tensor<128x64xf32, #mma>
            %177 = tt.splat %arg27 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %178 = arith.addi %177, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %179 = tt.expand_dims %178 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
            %180 = tt.broadcast %179 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
            %181 = arith.subi %113, %180 : tensor<128x64xi32, #mma>
            %182 = math.absi %181 : tensor<128x64xi32, #mma>
            %183 = arith.sitofp %182 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
            %184 = arith.mulf %115, %183 : tensor<128x64xf32, #mma>
            %185 = arith.mulf %184, %cst : tensor<128x64xf32, #mma>
            %186 = arith.addf %176, %185 : tensor<128x64xf32, #mma>
            %187 = "tt.reduce"(%186) <{axis = 1 : i32}> ({
            ^bb0(%arg34: f32, %arg35: f32):
              %208 = arith.maxnumf %arg34, %arg35 : f32
              tt.reduce.return %208 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %188 = arith.maxnumf %arg30, %187 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %189 = tt.expand_dims %188 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %190 = tt.broadcast %189 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %191 = arith.subf %186, %190 : tensor<128x64xf32, #mma>
            %192 = math.exp2 %191 : tensor<128x64xf32, #mma>
            %193 = "tt.reduce"(%192) <{axis = 1 : i32}> ({
            ^bb0(%arg34: f32, %arg35: f32):
              %208 = arith.addf %arg34, %arg35 : f32
              tt.reduce.return %208 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %194 = arith.subf %arg30, %188 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %195 = math.exp2 %194 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %196 = tt.expand_dims %195 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %197 = tt.broadcast %196 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %198 = arith.mulf %arg28, %197 : tensor<128x64xf32, #mma>
            %199 = arith.mulf %arg29, %195 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %200 = arith.addf %199, %193 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %201 = arith.truncf %192 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
            %202 = ttg.convert_layout %201 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
            %203 = ttg.local_alloc %173 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared2, #smem>
            %204 = ttg.local_load %203 : !ttg.memdesc<64x64xf16, #shared2, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
            %205 = tt.dot %202, %204, %198, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
            %206 = tt.addptr %arg32, %119 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
            %207 = ttg.local_alloc %171 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
            scf.yield %205, %200, %188, %170, %206, %207 : tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
          }
          %126 = arith.addi %112, %c63_i32 : i32
          %127 = arith.divsi %126, %c64_i32 : i32
          %128 = arith.subi %127, %c1_i32 : i32
          %129 = arith.maxsi %128, %c0_i32 : i32
          %130 = arith.muli %129, %c64_i32 : i32
          %131 = arith.cmpi sge, %127, %c1_i32 : i32
          %132 = ttg.local_load %125#5 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
          %133 = tt.splat %131 : i1 -> tensor<64x64xi1, #blocked1>
          %134 = tt.load %125#4, %133 : tensor<64x64x!tt.ptr<f16>, #blocked1>
          %135 = scf.if %131 -> (tensor<128x64xf32, #mma>) {
            %170 = tt.dot %76, %132, %cst_5, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
            scf.yield %170 : tensor<128x64xf32, #mma>
          } else {
            scf.yield %cst_5 : tensor<128x64xf32, #mma>
          }
          %136 = arith.mulf %135, %cst_4 : tensor<128x64xf32, #mma>
          %137 = arith.addf %136, %cst_5 : tensor<128x64xf32, #mma>
          %138 = tt.splat %130 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %139 = arith.addi %138, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %140 = tt.expand_dims %139 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
          %141 = tt.broadcast %140 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
          %142 = arith.subi %113, %141 : tensor<128x64xi32, #mma>
          %143 = math.absi %142 : tensor<128x64xi32, #mma>
          %144 = arith.sitofp %143 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
          %145 = arith.mulf %115, %144 : tensor<128x64xf32, #mma>
          %146 = arith.mulf %145, %cst : tensor<128x64xf32, #mma>
          %147 = arith.addf %137, %146 : tensor<128x64xf32, #mma>
          %148 = "tt.reduce"(%147) <{axis = 1 : i32}> ({
          ^bb0(%arg27: f32, %arg28: f32):
            %170 = arith.maxnumf %arg27, %arg28 : f32
            tt.reduce.return %170 : f32
          }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %149 = arith.maxnumf %125#2, %148 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %150 = tt.expand_dims %149 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %151 = tt.broadcast %150 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
          %152 = arith.subf %147, %151 : tensor<128x64xf32, #mma>
          %153 = math.exp2 %152 : tensor<128x64xf32, #mma>
          %154 = "tt.reduce"(%153) <{axis = 1 : i32}> ({
          ^bb0(%arg27: f32, %arg28: f32):
            %170 = arith.addf %arg27, %arg28 : f32
            tt.reduce.return %170 : f32
          }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %155 = arith.subf %125#2, %149 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %156 = math.exp2 %155 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %157 = tt.expand_dims %156 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %158 = tt.broadcast %157 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
          %159 = arith.mulf %125#0, %158 : tensor<128x64xf32, #mma>
          %160 = arith.mulf %125#1, %156 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %161 = arith.addf %160, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %162 = arith.truncf %153 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
          %163 = ttg.convert_layout %162 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
          %164 = ttg.local_alloc %134 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared2, #smem>
          %165 = ttg.local_load %164 : !ttg.memdesc<64x64xf16, #shared2, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
          %166 = scf.if %131 -> (tensor<128x64xf32, #mma>) {
            %170 = tt.dot %163, %165, %159, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
            scf.yield %170 : tensor<128x64xf32, #mma>
          } else {
            scf.yield %159 : tensor<128x64xf32, #mma>
          }
          %167 = arith.select %131, %166, %125#0 : tensor<128x64xf32, #mma>
          %168 = arith.select %131, %161, %125#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %169 = arith.select %131, %149, %125#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          scf.yield %169, %168, %167, %112 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>, i32
        } else {
          scf.yield %cst_9, %cst_8, %cst_5, %c0_i32 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>, i32
        }
        gpu.barrier
        %82 = arith.cmpi sgt, %77, %c0_i32 : i32
        %83:3 = scf.if %82 -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>) {
          %112 = arith.muli %78, %c64_i32 : i32
          %113 = arith.muli %112, %arg10 : i32
          %114 = tt.splat %113 : i32 -> tensor<64x64xi32, #blocked>
          %115 = tt.addptr %54, %114 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
          %116 = arith.muli %112, %arg13 : i32
          %117 = tt.splat %116 : i32 -> tensor<64x64xi32, #blocked1>
          %118 = tt.addptr %65, %117 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
          %119 = tt.broadcast %27 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
          %120 = arith.mulf %69, %cst_10 : f32
          %121 = tt.splat %120 : f32 -> tensor<128x64xf32, #mma>
          %122 = arith.muli %arg10, %c64_i32 : i32
          %123 = tt.splat %122 : i32 -> tensor<64x64xi32, #blocked>
          %124 = arith.muli %arg13, %c64_i32 : i32
          %125 = tt.splat %124 : i32 -> tensor<64x64xi32, #blocked1>
          %126 = arith.cmpi slt, %81#3, %79 : i32
          %127 = tt.splat %81#3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %128 = arith.addi %127, %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %129 = tt.expand_dims %128 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
          %130 = arith.cmpi slt, %129, %cst_2 : tensor<1x64xi32, #blocked>
          %131 = tt.broadcast %130 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %132 = tt.splat %126 : i1 -> tensor<64x64xi1, #blocked>
          %133 = arith.andi %132, %131 : tensor<64x64xi1, #blocked>
          %134 = tt.load %115, %133, %cst_14 : tensor<64x64x!tt.ptr<f16>, #blocked>
          %135 = ttg.local_alloc %134 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
          %136 = arith.subi %79, %c64_i32 : i32
          %137:6 = scf.for %arg27 = %81#3 to %136 step %c64_i32 iter_args(%arg28 = %81#2, %arg29 = %81#1, %arg30 = %81#0, %arg31 = %115, %arg32 = %118, %arg33 = %135) -> (tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>)  : i32 {
            amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
            %192 = tt.addptr %arg31, %123 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
            %193 = arith.addi %arg27, %c64_i32 : i32
            %194 = tt.splat %193 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
            %195 = tt.splat %arg27 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %196 = tt.splat %arg27 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
            %197 = arith.addi %194, %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
            %198 = arith.addi %195, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %199 = arith.addi %196, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
            %200 = tt.expand_dims %197 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
            %201 = tt.expand_dims %198 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
            %202 = arith.cmpi slt, %200, %cst_2 : tensor<1x64xi32, #blocked>
            %203 = tt.broadcast %202 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
            %204 = tt.load %192, %203, %cst_14 : tensor<64x64x!tt.ptr<f16>, #blocked>
            %205 = ttg.local_load %arg33 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
            %206 = tt.expand_dims %199 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
            %207 = arith.cmpi slt, %206, %cst_3 : tensor<64x1xi32, #blocked1>
            %208 = tt.broadcast %207 : tensor<64x1xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
            %209 = tt.load %arg32, %208, %cst_13 : tensor<64x64x!tt.ptr<f16>, #blocked1>
            %210 = tt.broadcast %201 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
            %211 = arith.cmpi sge, %119, %210 : tensor<128x64xi32, #mma>
            %212 = arith.select %211, %cst_5, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
            %213 = tt.dot %76, %205, %cst_5, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
            %214 = arith.mulf %213, %cst_4 : tensor<128x64xf32, #mma>
            %215 = arith.addf %212, %214 : tensor<128x64xf32, #mma>
            %216 = arith.subi %119, %210 : tensor<128x64xi32, #mma>
            %217 = math.absi %216 : tensor<128x64xi32, #mma>
            %218 = arith.sitofp %217 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
            %219 = arith.mulf %121, %218 : tensor<128x64xf32, #mma>
            %220 = arith.mulf %219, %cst : tensor<128x64xf32, #mma>
            %221 = arith.addf %215, %220 : tensor<128x64xf32, #mma>
            %222 = "tt.reduce"(%221) <{axis = 1 : i32}> ({
            ^bb0(%arg34: f32, %arg35: f32):
              %243 = arith.maxnumf %arg34, %arg35 : f32
              tt.reduce.return %243 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %223 = arith.maxnumf %arg30, %222 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %224 = tt.expand_dims %223 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %225 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %226 = arith.subf %221, %225 : tensor<128x64xf32, #mma>
            %227 = math.exp2 %226 : tensor<128x64xf32, #mma>
            %228 = "tt.reduce"(%227) <{axis = 1 : i32}> ({
            ^bb0(%arg34: f32, %arg35: f32):
              %243 = arith.addf %arg34, %arg35 : f32
              tt.reduce.return %243 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %229 = arith.subf %arg30, %223 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %230 = math.exp2 %229 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %231 = tt.expand_dims %230 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %232 = tt.broadcast %231 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %233 = arith.mulf %arg28, %232 : tensor<128x64xf32, #mma>
            %234 = arith.mulf %arg29, %230 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %235 = arith.addf %234, %228 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %236 = arith.truncf %227 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
            %237 = ttg.convert_layout %236 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
            %238 = ttg.local_alloc %209 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared2, #smem>
            %239 = ttg.local_load %238 : !ttg.memdesc<64x64xf16, #shared2, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
            %240 = tt.dot %237, %239, %233, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
            %241 = tt.addptr %arg32, %125 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
            %242 = ttg.local_alloc %204 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
            scf.yield %240, %235, %223, %192, %241, %242 : tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
          }
          %138 = arith.subi %79, %81#3 : i32
          %139 = arith.addi %138, %c63_i32 : i32
          %140 = arith.divsi %139, %c64_i32 : i32
          %141 = arith.subi %140, %c1_i32 : i32
          %142 = arith.maxsi %141, %c0_i32 : i32
          %143 = arith.muli %142, %c64_i32 : i32
          %144 = arith.addi %81#3, %143 : i32
          %145 = arith.cmpi sge, %140, %c1_i32 : i32
          %146 = tt.splat %144 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %147 = tt.splat %144 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %148 = arith.addi %146, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %149 = arith.addi %147, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %150 = tt.expand_dims %148 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
          %151 = ttg.local_load %137#5 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
          %152 = tt.expand_dims %149 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
          %153 = arith.cmpi slt, %152, %cst_3 : tensor<64x1xi32, #blocked1>
          %154 = tt.broadcast %153 : tensor<64x1xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
          %155 = tt.splat %145 : i1 -> tensor<64x64xi1, #blocked1>
          %156 = arith.andi %155, %154 : tensor<64x64xi1, #blocked1>
          %157 = tt.load %137#4, %156, %cst_13 : tensor<64x64x!tt.ptr<f16>, #blocked1>
          %158 = tt.broadcast %150 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
          %159 = arith.cmpi sge, %119, %158 : tensor<128x64xi32, #mma>
          %160 = arith.select %159, %cst_5, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
          %161 = scf.if %145 -> (tensor<128x64xf32, #mma>) {
            %192 = tt.dot %76, %151, %cst_5, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
            scf.yield %192 : tensor<128x64xf32, #mma>
          } else {
            scf.yield %cst_5 : tensor<128x64xf32, #mma>
          }
          %162 = arith.mulf %161, %cst_4 : tensor<128x64xf32, #mma>
          %163 = arith.addf %160, %162 : tensor<128x64xf32, #mma>
          %164 = arith.subi %119, %158 : tensor<128x64xi32, #mma>
          %165 = math.absi %164 : tensor<128x64xi32, #mma>
          %166 = arith.sitofp %165 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
          %167 = arith.mulf %121, %166 : tensor<128x64xf32, #mma>
          %168 = arith.mulf %167, %cst : tensor<128x64xf32, #mma>
          %169 = arith.addf %163, %168 : tensor<128x64xf32, #mma>
          %170 = "tt.reduce"(%169) <{axis = 1 : i32}> ({
          ^bb0(%arg27: f32, %arg28: f32):
            %192 = arith.maxnumf %arg27, %arg28 : f32
            tt.reduce.return %192 : f32
          }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %171 = arith.maxnumf %137#2, %170 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %172 = tt.expand_dims %171 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %173 = tt.broadcast %172 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
          %174 = arith.subf %169, %173 : tensor<128x64xf32, #mma>
          %175 = math.exp2 %174 : tensor<128x64xf32, #mma>
          %176 = "tt.reduce"(%175) <{axis = 1 : i32}> ({
          ^bb0(%arg27: f32, %arg28: f32):
            %192 = arith.addf %arg27, %arg28 : f32
            tt.reduce.return %192 : f32
          }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %177 = arith.subf %137#2, %171 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %178 = math.exp2 %177 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %179 = tt.expand_dims %178 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %180 = tt.broadcast %179 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
          %181 = arith.mulf %137#0, %180 : tensor<128x64xf32, #mma>
          %182 = arith.mulf %137#1, %178 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %183 = arith.addf %182, %176 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %184 = arith.truncf %175 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
          %185 = ttg.convert_layout %184 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
          %186 = ttg.local_alloc %157 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared2, #smem>
          %187 = ttg.local_load %186 : !ttg.memdesc<64x64xf16, #shared2, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
          %188 = scf.if %145 -> (tensor<128x64xf32, #mma>) {
            %192 = tt.dot %185, %187, %181, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
            scf.yield %192 : tensor<128x64xf32, #mma>
          } else {
            scf.yield %181 : tensor<128x64xf32, #mma>
          }
          %189 = arith.select %145, %188, %137#0 : tensor<128x64xf32, #mma>
          %190 = arith.select %145, %183, %137#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %191 = arith.select %145, %171, %137#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          scf.yield %191, %190, %189 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>
        } else {
          scf.yield %81#0, %81#1, %81#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>
        }
        %84 = tt.expand_dims %83#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
        %85 = arith.divf %cst_7, %84 : tensor<128x1xf32, #mma>
        %86 = tt.broadcast %85 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
        %87 = arith.mulf %83#2, %86 : tensor<128x64xf32, #mma>
        %88 = arith.truncf %87 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
        %89 = arith.cmpi slt, %3, %c0_i32 : i32
        %90 = arith.cmpi sgt, %17, %c0_i32 : i32
        %91 = arith.andi %89, %90 : i1
        %92 = scf.if %91 -> (tensor<128x64xf16, #mma>) {
          %112 = arith.cmpi sge, %27, %cst_0 : tensor<128x1xi32, #mma>
          %113 = tt.broadcast %112 : tensor<128x1xi1, #mma> -> tensor<128x64xi1, #mma>
          %114 = arith.select %113, %88, %cst_6 : tensor<128x64xi1, #mma>, tensor<128x64xf16, #mma>
          scf.yield %114 : tensor<128x64xf16, #mma>
        } else {
          scf.yield %88 : tensor<128x64xf16, #mma>
        }
        %93 = arith.muli %2, %c49152_i32 : i32
        %94 = tt.addptr %arg3, %93 : !tt.ptr<f32>, i32
        %95 = arith.muli %1, %c1024_i32 : i32
        %96 = tt.addptr %94, %95 : !tt.ptr<f32>, i32
        %97 = tt.splat %96 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
        %98 = tt.addptr %97, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %99 = arith.subi %17, %c1024_i32 : i32
        %100 = arith.cmpi sgt, %99, %c0_i32 : i32
        scf.if %100 {
          %112 = arith.subi %c1152_i32, %17 : i32
          %113 = tt.splat %112 : i32 -> tensor<128xi32, #blocked2>
          %114 = arith.cmpi slt, %6, %113 : tensor<128xi32, #blocked2>
          %115 = math.log2 %83#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %116 = arith.addf %83#0, %115 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %117 = ttg.convert_layout %116 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
          tt.store %98, %117, %114 : tensor<128x!tt.ptr<f32>, #blocked2>
        } else {
          %112 = math.log2 %83#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %113 = arith.addf %83#0, %112 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %114 = ttg.convert_layout %113 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
          tt.store %98, %114 : tensor<128x!tt.ptr<f32>, #blocked2>
        }
        %101 = arith.muli %2, %arg14 : i32
        %102 = tt.addptr %arg4, %101 : !tt.ptr<f16>, i32
        %103 = arith.muli %1, %arg15 : i32
        %104 = tt.addptr %102, %103 : !tt.ptr<f16>, i32
        %105 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #mma>
        %106 = arith.muli %27, %105 : tensor<128x1xi32, #mma>
        %107 = tt.splat %104 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
        %108 = tt.addptr %107, %106 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
        %109 = tt.broadcast %108 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x64x!tt.ptr<f16>, #mma>
        %110 = tt.addptr %109, %38 : tensor<128x64x!tt.ptr<f16>, #mma>, tensor<128x64xi32, #mma>
        %111 = arith.select %100, %72, %cst_18 : tensor<128x64xi1, #mma>
        tt.store %110, %92, %111 : tensor<128x64x!tt.ptr<f16>, #mma>
      }
      scf.yield %c1_i32 : i32
    }
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(decompose-unsupported-amd-conversions{arch=gfx942}, optimize-amd-lds-usage{lds-limit=0 target-arch=gfx942}, convert-scf-to-cf, convert-index-to-llvm{index-bitwidth=0}, allocate-shared-memory, triton-amdgpu-membar-analysis, triton-amdgpu-refine-ops{arch=gfx942}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true})",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/flash-attention.py:527:0: error: Failures have been detected while processing an MLIR pass pipeline
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/flash-attention.py:527:0: note: Pipeline failed while executing [`TritonAMDGPURefineOps` on 'builtin.module' operation]: reproducer generated at `std::errs, please share the reproducer above with Triton project.`
_______________ test_op_fwd[bshd-True-True-1-24-6-8192-8192-64] ________________

Z = 1, HQ = 24, HK = 6, N_CTX_Q = 8192, N_CTX_K = 8192, D_HEAD = 64
causal = True, use_alibi = True, layout = 'bshd', dtype = torch.float16

    @pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
        (4, 48, 24, 1024, 1024, 64),
        (1, 24, 6, 8192, 8192, 64),
        (1, 4, 2, 16384, 16384, 128),
        (2, 16, 4, 1020, 987, 128),
        (2, 16, 4, 15498, 2, 128),
        (2, 16, 2, 7, 16219, 64),
        (4, 48, 12, 1, 1, 64),
        (4, 48, 48, 1, 1, 128),
        (4, 48, 24, 3, 3, 128),
        (4, 48, 48, 1001, 990, 64),
        (1, 8, 8, 8081, 7099, 64),
        (1, 4, 4, 16330, 15989, 128),
        (4, 4, 1, 1024, 1024, 33),
        (4, 4, 2, 65, 1018, 65),
        (4, 4, 4, 128, 128, 65),
        (4, 4, 4, 113, 123, 1),
    ])
    @pytest.mark.parametrize('causal', [True, False])
    @pytest.mark.parametrize('use_alibi', [True, False])
    @pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
    def test_op_fwd(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_alibi, layout, dtype=torch.float16):
        torch.manual_seed(20)
        q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
        if causal:
            input_metadata.need_causal()
    
        if use_alibi:
            # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
            alibi_slopes = torch.tensor([2**(-8 / HQ * i) for i in range(1, HQ + 1)], dtype=torch.float32,
                                        device="cuda").repeat(Z, 1)
            input_metadata.need_alibi(alibi_slopes, Z, HQ)
        else:
            alibi_slopes = None
    
        o = torch.empty_like(q)
    
        # triton implementation
>       tri_out, _, _ = attention(q, k, v, o, input_metadata)

flash-attention.py:1480: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/function.py:598: in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
flash-attention.py:1221: in forward
    handle = attn_fwd[grid](q, k, v, metadata.bias, metadata.sm_scale, M, o, *q_strides, *k_strides, *v_strides, *o_strides,
../../../../triton/python/triton/runtime/jit.py:368: in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
../../../../triton/python/triton/runtime/autotuner.py:206: in run
    ret = self.fn.run(
../../../../triton/python/triton/runtime/jit.py:608: in run
    kernel = self.compile(src, target=target, options=options.__dict__)
../../../../triton/python/triton/compiler/compiler.py:284: in compile
    next_module = compile_ir(module, metadata)
../../../../triton/python/triton/backends/amd/compiler.py:438: in <lambda>
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

src = <triton._C.libtriton.ir.module object at 0x7f80d3d2f880>
metadata = {'allow_flush_denorm': False, 'allowed_dot_input_precisions': ('ieee', 'tf32'), 'arch': 'gfx942', 'backend_name': 'hip', ...}
options = HIPOptions(num_warps=4, waves_per_eu=2, num_stages=2, num_ctas=1, extern_libs=(('ocml', '/home/dtanner/repos/triton/py... allow_flush_denorm=False, max_num_imprecise_acc_default=0, backend_name='hip', instruction_sched_variant='refine_ops')

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_decompose_unsupported_conversions(pm, options.arch)
        # custom_lds_size is an experimental parameter that defines amount of LDS available
        # for one thread block. Measured in bytes.
        #
        # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
        # LDS size is determined by provided arch name.
        custom_lds_size = 0
        amd.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
    
        passes.ttgpuir.add_allocate_shared_memory(pm)
        amd.passes.ttgpuir.add_membar_analysis(pm)
        amd.passes.ttgpuir.add_refine_amdgpu_ops(pm, options.arch)
    
        # TODO - remove
        passes.common.add_canonicalizer(pm)
>       pm.run(mod)
E       RuntimeError: PassManager::run failed

../../../../triton/python/triton/backends/amd/compiler.py:296: RuntimeError
----------------------------- Captured stderr call -----------------------------
[tritonamdgpu-refine-ops]: refinedShape: 16x32
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %130 = ttg.memdesc_subview %125[%c0_i32_19, %c0_i32_20] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %132 = ttg.memdesc_subview %125[%c16_i32, %c0_i32_21] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %134 = ttg.memdesc_subview %125[%c32_i32, %c0_i32_22] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %136 = ttg.memdesc_subview %125[%c48_i32, %c0_i32_23] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %138 = ttg.memdesc_subview %125[%c0_i32_24, %c32_i32_25] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %140 = ttg.memdesc_subview %125[%c16_i32_26, %c32_i32_27] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %142 = ttg.memdesc_subview %125[%c32_i32_28, %c32_i32_29] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %144 = ttg.memdesc_subview %125[%c48_i32_30, %c32_i32_31] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: ConcatOp: %146 = amdgpu.concat %131, %133, %135, %137, %139, %141, %143, %145 [4, 2] : tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>, tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-refine-ops]: refinedShape: 8x32
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %179 = ttg.memdesc_subview %177[%c0_i32_32, %c0_i32_33] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %181 = ttg.memdesc_subview %177[%c8_i32, %c0_i32_34] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %183 = ttg.memdesc_subview %177[%c16_i32_35, %c0_i32_36] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %185 = ttg.memdesc_subview %177[%c24_i32, %c0_i32_37] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %187 = ttg.memdesc_subview %177[%c32_i32_38, %c0_i32_39] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %189 = ttg.memdesc_subview %177[%c40_i32, %c0_i32_40] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %191 = ttg.memdesc_subview %177[%c48_i32_41, %c0_i32_42] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %193 = ttg.memdesc_subview %177[%c56_i32, %c0_i32_43] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %195 = ttg.memdesc_subview %177[%c0_i32_44, %c32_i32_45] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %197 = ttg.memdesc_subview %177[%c8_i32_46, %c32_i32_47] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %199 = ttg.memdesc_subview %177[%c16_i32_48, %c32_i32_49] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %201 = ttg.memdesc_subview %177[%c24_i32_50, %c32_i32_51] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %203 = ttg.memdesc_subview %177[%c32_i32_52, %c32_i32_53] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %205 = ttg.memdesc_subview %177[%c40_i32_54, %c32_i32_55] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %207 = ttg.memdesc_subview %177[%c48_i32_56, %c32_i32_57] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: RefinedLocalLoadSubvew: %209 = ttg.memdesc_subview %177[%c56_i32_58, %c32_i32_59] : !ttg.memdesc<64x64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory> -> !ttg.memdesc<8x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 64x64>
[tritonamdgpu-refine-ops]: ConcatOp: %211 = amdgpu.concat %180, %182, %184, %186, %188, %190, %192, %194, %196, %198, %200, %202, %204, %206, %208, %210 [8, 2] : tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>, tensor<8x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>, kWidth = 4}>>
[tritonamdgpu-refine-ops]: totalReps: 1x2x4
[tritonamdgpu-refine-ops]: repsPerDotTile: 1x1x1
[tritonamdgpu-refine-ops]: local_load_a[0][0] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][1] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][2] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][3] extract_slice
[tritonamdgpu-refine-ops]: local_load_b[0][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[0][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[1][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[1][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[2][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[2][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[3][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[3][1] extact_slice
[tritonamdgpu-refine-ops]: dot-tile[0]
[tritonamdgpu-refine-ops]:   dot[0][0][0]
[tritonamdgpu-refine-ops]: dot-tile[1]
[tritonamdgpu-refine-ops]:   dot[0][1][0]
[tritonamdgpu-refine-ops]: dot-tile[2]
[tritonamdgpu-refine-ops]:   dot[0][0][1]
[tritonamdgpu-refine-ops]: dot-tile[3]
[tritonamdgpu-refine-ops]:   dot[0][1][1]
[tritonamdgpu-refine-ops]: dot-tile[4]
[tritonamdgpu-refine-ops]:   dot[0][0][2]
[tritonamdgpu-refine-ops]: dot-tile[5]
[tritonamdgpu-refine-ops]:   dot[0][1][2]
[tritonamdgpu-refine-ops]: dot-tile[6]
[tritonamdgpu-refine-ops]:   dot[0][0][3]
[tritonamdgpu-refine-ops]: dot-tile[7]
[tritonamdgpu-refine-ops]:   dot[0][1][3]
[tritonamdgpu-refine-ops]: totalReps: 1x2x8
[tritonamdgpu-refine-ops]: repsPerDotTile: 1x2x1
[tritonamdgpu-refine-ops]: local_load_a[0][0] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][1] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][2] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][3] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][4] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][5] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][6] extract_slice
[tritonamdgpu-refine-ops]: local_load_a[0][7] extract_slice
[tritonamdgpu-refine-ops]: local_load_b[0][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[0][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[1][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[1][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[2][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[2][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[3][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[3][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[4][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[4][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[5][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[5][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[6][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[6][1] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[7][0] extact_slice
[tritonamdgpu-refine-ops]: local_load_b[7][1] extact_slice
[tritonamdgpu-refine-ops]: dot-tile[0]
[tritonamdgpu-refine-ops]:   dot[0][0][0]
[tritonamdgpu-refine-ops]:   dot[0][1][0]
[tritonamdgpu-refine-ops]: dot-tile[1]
[tritonamdgpu-refine-ops]:   dot[0][0][1]
[tritonamdgpu-refine-ops]:   dot[0][1][1]
[tritonamdgpu-refine-ops]: dot-tile[2]
[tritonamdgpu-refine-ops]:   dot[0][0][2]
[tritonamdgpu-refine-ops]:   dot[0][1][2]
[tritonamdgpu-refine-ops]: dot-tile[3]
[tritonamdgpu-refine-ops]:   dot[0][0][3]
[tritonamdgpu-refine-ops]:   dot[0][1][3]
[tritonamdgpu-refine-ops]: dot-tile[4]
[tritonamdgpu-refine-ops]:   dot[0][0][4]
[tritonamdgpu-refine-ops]:   dot[0][1][4]
[tritonamdgpu-refine-ops]: dot-tile[5]
[tritonamdgpu-refine-ops]:   dot[0][0][5]
[tritonamdgpu-refine-ops]:   dot[0][1][5]
[tritonamdgpu-refine-ops]: dot-tile[6]
[tritonamdgpu-refine-ops]:   dot[0][0][6]
[tritonamdgpu-refine-ops]:   dot[0][1][6]
[tritonamdgpu-refine-ops]: dot-tile[7]
[tritonamdgpu-refine-ops]:   dot[0][0][7]
[tritonamdgpu-refine-ops]:   dot[0][1][7]
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %200 = math.exp2 %199 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
srcShape: 128x64
srcShapePerCtaTile: 128x64
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
numReps: 1x1
[tritonamdgpu-refine-ops]: failed to refine binary op: %200 = math.exp2 %199 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %205 = math.exp2 %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
getRefinedShape: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
inDimNames
register: bases: 1
lane: bases: 32
warp: bases: 4
block: bases: 1
outDimNames
dim0
getRefinedShape(slice/mfma)
llShape: 128
mfmaShape: 128x32
retShape: 128
srcShape: 128
srcShapePerCtaTile: 128
getRefinedShape: tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
inDimNames
register: bases: 1
lane: bases: 32
warp: bases: 4
block: bases: 1
outDimNames
dim0
getRefinedShape(slice/mfma)
llShape: 128
mfmaShape: 128x32
retShape: 128
numReps: 1
[tritonamdgpu-refine-ops]: failed to refine binary op: %205 = math.exp2 %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %211 = arith.truncf %200 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
srcShape: 128x64
srcShapePerCtaTile: 128x64
getRefinedShape: tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
numReps: 1x1
[tritonamdgpu-refine-ops]: failed to refine binary op: %211 = arith.truncf %200 : tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf16, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %189 = arith.sitofp %188 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
getRefinedShape: tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
srcShape: 128x64
srcShapePerCtaTile: 128x64
getRefinedShape: tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
numReps: 1x1
[tritonamdgpu-refine-ops]: failed to refine binary op: %189 = arith.sitofp %188 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>> to tensor<128x64xf32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %187 = arith.subi %107, %186 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
getRefinedShape: tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
srcShape: 128x64
srcShapePerCtaTile: 128x64
getRefinedShape: tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
getRefinedShape: tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
inDimNames
register: bases: 1x32
lane: bases: 32x2
warp: bases: 4x1
block: bases: 1x1
outDimNames
dim0
dim1
getRefinedShape(mfma)
llShape: 128x8
mfmaShape: 128x32
retShape: 128x32
numReps: 1x1
[tritonamdgpu-refine-ops]: failed to refine binary op: %187 = arith.subi %107, %186 : tensor<128x64xi32, #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>>
[tritonamdgpu-refine-ops]: 


rewriteElementWiseOp(): %184 = arith.addi %183, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
getRefinedShape: tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>}>>
inDimNames
register: bases: 32
lane: bases: 2
warp: bases: 1
block: bases: 1
outDimNames
dim0
getRefinedShape(slice/mfma)
python: /home/dtanner/.triton/llvm/llvm-2619c2ed-ubuntu-x64/include/llvm/ADT/SmallVector.h:291: reference llvm::SmallVectorTemplateCommon<unsigned int>::operator[](size_type) [T = unsigned int]: Assertion `idx < size()' failed.
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32, %arg22: f32, %arg23: i32, %arg24: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.44269502> : tensor<128x64xf32, #mma>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #mma>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128x64xf32, #mma>
    %cst_2 = arith.constant dense<8192> : tensor<1x64xi32, #blocked>
    %cst_3 = arith.constant dense<8192> : tensor<64x1xi32, #blocked1>
    %cst_4 = arith.constant dense<0.180336878> : tensor<128x64xf32, #mma>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #mma>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #mma>
    %cst_8 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_9 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_10 = arith.constant -1.000000e+00 : f32
    %c63_i32 = arith.constant 63 : i32
    %c8320_i32 = arith.constant 8320 : i32
    %c2_i32 = arith.constant 2 : i32
    %c196608_i32 = arith.constant 196608 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_11 = arith.constant dense<8192> : tensor<128xi32, #blocked2>
    %cst_12 = arith.constant dense<0x7F800000> : tensor<128xf32, #blocked2>
    %cst_13 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked1>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %cst_16 = arith.constant dense<8192> : tensor<128x1xi32, #blocked1>
    %cst_17 = arith.constant dense<8192> : tensor<128x1xi32, #mma>
    %cst_18 = arith.constant dense<true> : tensor<128x64xi1, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %0, %c128_i32 : i32
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %7 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %8 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %9 = tt.splat %3 : i32 -> tensor<128xi32, #blocked2>
    %10 = arith.addi %7, %4 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %11 = arith.addi %8, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %12 = arith.addi %9, %6 : tensor<128xi32, #blocked2>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = arith.addi %0, %c1_i32 : i32
    %17 = arith.muli %16, %c128_i32 : i32
    %18 = arith.addi %17, %c63_i32 : i32
    %19 = arith.divsi %18, %c64_i32 : i32
    %20 = arith.minsi %19, %c128_i32 : i32
    %21 = arith.cmpi sle, %20, %c0_i32 : i32
    scf.while (%arg27 = %c0_i32) : (i32) -> () {
      %22 = arith.cmpi slt, %arg27, %c1_i32 : i32
      scf.condition(%22)
    } do {
      scf.if %21 {
        %22 = arith.muli %2, %arg14 : i32
        %23 = tt.addptr %arg4, %22 : !tt.ptr<f16>, i32
        %24 = arith.muli %1, %arg15 : i32
        %25 = tt.addptr %23, %24 : !tt.ptr<f16>, i32
        %26 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
        %27 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #blocked1>
        %28 = arith.muli %26, %27 : tensor<128x1xi32, #blocked1>
        %29 = tt.splat %25 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
        %30 = tt.addptr %29, %28 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
        %31 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %32 = tt.expand_dims %31 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
        %33 = tt.broadcast %30 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
        %34 = tt.broadcast %32 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %35 = tt.addptr %33, %34 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %36 = arith.cmpi slt, %26, %cst_16 : tensor<128x1xi32, #blocked1>
        %37 = tt.broadcast %36 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        tt.store %35, %cst_15, %37 : tensor<128x64x!tt.ptr<f16>, #blocked1>
        %38 = arith.muli %2, %c196608_i32 : i32
        %39 = tt.addptr %arg3, %38 : !tt.ptr<f32>, i32
        %40 = arith.muli %1, %c8192_i32 : i32
        %41 = tt.addptr %39, %40 : !tt.ptr<f32>, i32
        %42 = tt.splat %41 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
        %43 = tt.addptr %42, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %44 = arith.cmpi slt, %12, %cst_11 : tensor<128xi32, #blocked2>
        tt.store %43, %cst_12, %44 : tensor<128x!tt.ptr<f32>, #blocked2>
      } else {
        %22 = arith.divsi %1, %c4_i32 : i32
        %23 = arith.muli %2, %arg5 : i32
        %24 = tt.addptr %arg0, %23 : !tt.ptr<f16>, i32
        %25 = arith.muli %1, %arg6 : i32
        %26 = tt.addptr %24, %25 : !tt.ptr<f16>, i32
        %27 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
        %28 = tt.expand_dims %11 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
        %29 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked1>
        %30 = arith.muli %28, %29 : tensor<128x1xi32, #blocked1>
        %31 = tt.splat %26 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
        %32 = tt.addptr %31, %30 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
        %33 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %34 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
        %35 = tt.expand_dims %33 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
        %36 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
        %37 = tt.broadcast %32 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
        %38 = tt.broadcast %34 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
        %39 = tt.broadcast %35 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %40 = tt.addptr %37, %39 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %41 = arith.muli %2, %arg8 : i32
        %42 = tt.addptr %arg1, %41 : !tt.ptr<f16>, i32
        %43 = arith.muli %22, %arg9 : i32
        %44 = tt.addptr %42, %43 : !tt.ptr<f16>, i32
        %45 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %46 = tt.expand_dims %45 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %47 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
        %48 = tt.splat %44 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
        %49 = tt.addptr %48, %46 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
        %50 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #blocked>
        %51 = arith.muli %36, %50 : tensor<1x64xi32, #blocked>
        %52 = tt.broadcast %49 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x64x!tt.ptr<f16>, #blocked>
        %53 = tt.broadcast %51 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
        %54 = tt.addptr %52, %53 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
        %55 = arith.muli %2, %arg11 : i32
        %56 = tt.addptr %arg2, %55 : !tt.ptr<f16>, i32
        %57 = arith.muli %22, %arg12 : i32
        %58 = tt.addptr %56, %57 : !tt.ptr<f16>, i32
        %59 = tt.splat %arg13 : i32 -> tensor<64x1xi32, #blocked1>
        %60 = arith.muli %47, %59 : tensor<64x1xi32, #blocked1>
        %61 = tt.splat %58 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1>
        %62 = tt.addptr %61, %60 : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1>
        %63 = tt.broadcast %62 : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x64x!tt.ptr<f16>, #blocked1>
        %64 = tt.broadcast %35 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
        %65 = tt.addptr %63, %64 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
        %66 = arith.muli %2, %arg21 : i32
        %67 = arith.addi %66, %1 : i32
        %68 = tt.addptr %arg26, %67 : !tt.ptr<f32>, i32
        %69 = tt.load %68 : !tt.ptr<f32>
        %70 = arith.cmpi slt, %27, %cst_17 : tensor<128x1xi32, #mma>
        %71 = arith.cmpi slt, %28, %cst_16 : tensor<128x1xi32, #blocked1>
        %72 = tt.broadcast %70 : tensor<128x1xi1, #mma> -> tensor<128x64xi1, #mma>
        %73 = tt.broadcast %71 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %74 = tt.load %40, %73, %cst_15 : tensor<128x64x!tt.ptr<f16>, #blocked1>
        %75 = ttg.local_alloc %74 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %76 = ttg.local_load %75 : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
        %77 = arith.minsi %20, %c2_i32 : i32
        %78 = arith.subi %20, %77 : i32
        %79 = arith.muli %20, %c64_i32 : i32
        %80 = arith.cmpi sgt, %78, %c0_i32 : i32
        %81:4 = scf.if %80 -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>, i32) {
          %112 = arith.muli %78, %c64_i32 : i32
          %113 = tt.broadcast %27 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
          %114 = arith.mulf %69, %cst_10 : f32
          %115 = tt.splat %114 : f32 -> tensor<128x64xf32, #mma>
          %116 = arith.muli %arg10, %c64_i32 : i32
          %117 = tt.splat %116 : i32 -> tensor<64x64xi32, #blocked>
          %118 = arith.muli %arg13, %c64_i32 : i32
          %119 = tt.splat %118 : i32 -> tensor<64x64xi32, #blocked1>
          %120 = arith.cmpi sgt, %112, %c0_i32 : i32
          %121 = tt.splat %120 : i1 -> tensor<64x64xi1, #blocked>
          %122 = tt.load %54, %121 : tensor<64x64x!tt.ptr<f16>, #blocked>
          %123 = ttg.local_alloc %122 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
          %124 = arith.subi %112, %c64_i32 : i32
          %125:6 = scf.for %arg27 = %c0_i32 to %124 step %c64_i32 iter_args(%arg28 = %cst_5, %arg29 = %cst_8, %arg30 = %cst_9, %arg31 = %54, %arg32 = %65, %arg33 = %123) -> (tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>)  : i32 {
            amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
            %170 = tt.addptr %arg31, %117 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
            %171 = tt.load %170 : tensor<64x64x!tt.ptr<f16>, #blocked>
            %172 = ttg.local_load %arg33 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
            %173 = tt.load %arg32 : tensor<64x64x!tt.ptr<f16>, #blocked1>
            %174 = tt.dot %76, %172, %cst_5, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
            %175 = arith.mulf %174, %cst_4 : tensor<128x64xf32, #mma>
            %176 = arith.addf %175, %cst_5 : tensor<128x64xf32, #mma>
            %177 = tt.splat %arg27 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %178 = arith.addi %177, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %179 = tt.expand_dims %178 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
            %180 = tt.broadcast %179 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
            %181 = arith.subi %113, %180 : tensor<128x64xi32, #mma>
            %182 = math.absi %181 : tensor<128x64xi32, #mma>
            %183 = arith.sitofp %182 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
            %184 = arith.mulf %115, %183 : tensor<128x64xf32, #mma>
            %185 = arith.mulf %184, %cst : tensor<128x64xf32, #mma>
            %186 = arith.addf %176, %185 : tensor<128x64xf32, #mma>
            %187 = "tt.reduce"(%186) <{axis = 1 : i32}> ({
            ^bb0(%arg34: f32, %arg35: f32):
              %208 = arith.maxnumf %arg34, %arg35 : f32
              tt.reduce.return %208 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %188 = arith.maxnumf %arg30, %187 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %189 = tt.expand_dims %188 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %190 = tt.broadcast %189 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %191 = arith.subf %186, %190 : tensor<128x64xf32, #mma>
            %192 = math.exp2 %191 : tensor<128x64xf32, #mma>
            %193 = "tt.reduce"(%192) <{axis = 1 : i32}> ({
            ^bb0(%arg34: f32, %arg35: f32):
              %208 = arith.addf %arg34, %arg35 : f32
              tt.reduce.return %208 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %194 = arith.subf %arg30, %188 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %195 = math.exp2 %194 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %196 = tt.expand_dims %195 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %197 = tt.broadcast %196 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %198 = arith.mulf %arg28, %197 : tensor<128x64xf32, #mma>
            %199 = arith.mulf %arg29, %195 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %200 = arith.addf %199, %193 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %201 = arith.truncf %192 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
            %202 = ttg.convert_layout %201 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
            %203 = ttg.local_alloc %173 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared2, #smem>
            %204 = ttg.local_load %203 : !ttg.memdesc<64x64xf16, #shared2, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
            %205 = tt.dot %202, %204, %198, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
            %206 = tt.addptr %arg32, %119 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
            %207 = ttg.local_alloc %171 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
            scf.yield %205, %200, %188, %170, %206, %207 : tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
          }
          %126 = arith.addi %112, %c63_i32 : i32
          %127 = arith.divsi %126, %c64_i32 : i32
          %128 = arith.subi %127, %c1_i32 : i32
          %129 = arith.maxsi %128, %c0_i32 : i32
          %130 = arith.muli %129, %c64_i32 : i32
          %131 = arith.cmpi sge, %127, %c1_i32 : i32
          %132 = ttg.local_load %125#5 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
          %133 = tt.splat %131 : i1 -> tensor<64x64xi1, #blocked1>
          %134 = tt.load %125#4, %133 : tensor<64x64x!tt.ptr<f16>, #blocked1>
          %135 = scf.if %131 -> (tensor<128x64xf32, #mma>) {
            %170 = tt.dot %76, %132, %cst_5, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
            scf.yield %170 : tensor<128x64xf32, #mma>
          } else {
            scf.yield %cst_5 : tensor<128x64xf32, #mma>
          }
          %136 = arith.mulf %135, %cst_4 : tensor<128x64xf32, #mma>
          %137 = arith.addf %136, %cst_5 : tensor<128x64xf32, #mma>
          %138 = tt.splat %130 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %139 = arith.addi %138, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %140 = tt.expand_dims %139 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
          %141 = tt.broadcast %140 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
          %142 = arith.subi %113, %141 : tensor<128x64xi32, #mma>
          %143 = math.absi %142 : tensor<128x64xi32, #mma>
          %144 = arith.sitofp %143 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
          %145 = arith.mulf %115, %144 : tensor<128x64xf32, #mma>
          %146 = arith.mulf %145, %cst : tensor<128x64xf32, #mma>
          %147 = arith.addf %137, %146 : tensor<128x64xf32, #mma>
          %148 = "tt.reduce"(%147) <{axis = 1 : i32}> ({
          ^bb0(%arg27: f32, %arg28: f32):
            %170 = arith.maxnumf %arg27, %arg28 : f32
            tt.reduce.return %170 : f32
          }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %149 = arith.maxnumf %125#2, %148 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %150 = tt.expand_dims %149 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %151 = tt.broadcast %150 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
          %152 = arith.subf %147, %151 : tensor<128x64xf32, #mma>
          %153 = math.exp2 %152 : tensor<128x64xf32, #mma>
          %154 = "tt.reduce"(%153) <{axis = 1 : i32}> ({
          ^bb0(%arg27: f32, %arg28: f32):
            %170 = arith.addf %arg27, %arg28 : f32
            tt.reduce.return %170 : f32
          }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %155 = arith.subf %125#2, %149 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %156 = math.exp2 %155 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %157 = tt.expand_dims %156 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %158 = tt.broadcast %157 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
          %159 = arith.mulf %125#0, %158 : tensor<128x64xf32, #mma>
          %160 = arith.mulf %125#1, %156 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %161 = arith.addf %160, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %162 = arith.truncf %153 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
          %163 = ttg.convert_layout %162 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
          %164 = ttg.local_alloc %134 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared2, #smem>
          %165 = ttg.local_load %164 : !ttg.memdesc<64x64xf16, #shared2, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
          %166 = scf.if %131 -> (tensor<128x64xf32, #mma>) {
            %170 = tt.dot %163, %165, %159, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
            scf.yield %170 : tensor<128x64xf32, #mma>
          } else {
            scf.yield %159 : tensor<128x64xf32, #mma>
          }
          %167 = arith.select %131, %166, %125#0 : tensor<128x64xf32, #mma>
          %168 = arith.select %131, %161, %125#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %169 = arith.select %131, %149, %125#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          scf.yield %169, %168, %167, %112 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>, i32
        } else {
          scf.yield %cst_9, %cst_8, %cst_5, %c0_i32 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>, i32
        }
        gpu.barrier
        %82 = arith.cmpi sgt, %77, %c0_i32 : i32
        %83:3 = scf.if %82 -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>) {
          %112 = arith.muli %78, %c64_i32 : i32
          %113 = arith.muli %112, %arg10 : i32
          %114 = tt.splat %113 : i32 -> tensor<64x64xi32, #blocked>
          %115 = tt.addptr %54, %114 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
          %116 = arith.muli %112, %arg13 : i32
          %117 = tt.splat %116 : i32 -> tensor<64x64xi32, #blocked1>
          %118 = tt.addptr %65, %117 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
          %119 = tt.broadcast %27 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
          %120 = arith.mulf %69, %cst_10 : f32
          %121 = tt.splat %120 : f32 -> tensor<128x64xf32, #mma>
          %122 = arith.muli %arg10, %c64_i32 : i32
          %123 = tt.splat %122 : i32 -> tensor<64x64xi32, #blocked>
          %124 = arith.muli %arg13, %c64_i32 : i32
          %125 = tt.splat %124 : i32 -> tensor<64x64xi32, #blocked1>
          %126 = arith.cmpi slt, %81#3, %79 : i32
          %127 = tt.splat %81#3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %128 = arith.addi %127, %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %129 = tt.expand_dims %128 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
          %130 = arith.cmpi slt, %129, %cst_2 : tensor<1x64xi32, #blocked>
          %131 = tt.broadcast %130 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %132 = tt.splat %126 : i1 -> tensor<64x64xi1, #blocked>
          %133 = arith.andi %132, %131 : tensor<64x64xi1, #blocked>
          %134 = tt.load %115, %133, %cst_14 : tensor<64x64x!tt.ptr<f16>, #blocked>
          %135 = ttg.local_alloc %134 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
          %136 = arith.subi %79, %c64_i32 : i32
          %137:6 = scf.for %arg27 = %81#3 to %136 step %c64_i32 iter_args(%arg28 = %81#2, %arg29 = %81#1, %arg30 = %81#0, %arg31 = %115, %arg32 = %118, %arg33 = %135) -> (tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>)  : i32 {
            amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
            %192 = tt.addptr %arg31, %123 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
            %193 = arith.addi %arg27, %c64_i32 : i32
            %194 = tt.splat %193 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
            %195 = tt.splat %arg27 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %196 = tt.splat %arg27 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
            %197 = arith.addi %194, %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
            %198 = arith.addi %195, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
            %199 = arith.addi %196, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
            %200 = tt.expand_dims %197 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
            %201 = tt.expand_dims %198 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
            %202 = arith.cmpi slt, %200, %cst_2 : tensor<1x64xi32, #blocked>
            %203 = tt.broadcast %202 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
            %204 = tt.load %192, %203, %cst_14 : tensor<64x64x!tt.ptr<f16>, #blocked>
            %205 = ttg.local_load %arg33 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
            %206 = tt.expand_dims %199 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
            %207 = arith.cmpi slt, %206, %cst_3 : tensor<64x1xi32, #blocked1>
            %208 = tt.broadcast %207 : tensor<64x1xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
            %209 = tt.load %arg32, %208, %cst_13 : tensor<64x64x!tt.ptr<f16>, #blocked1>
            %210 = tt.broadcast %201 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
            %211 = arith.cmpi sge, %119, %210 : tensor<128x64xi32, #mma>
            %212 = arith.select %211, %cst_5, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
            %213 = tt.dot %76, %205, %cst_5, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
            %214 = arith.mulf %213, %cst_4 : tensor<128x64xf32, #mma>
            %215 = arith.addf %212, %214 : tensor<128x64xf32, #mma>
            %216 = arith.subi %119, %210 : tensor<128x64xi32, #mma>
            %217 = math.absi %216 : tensor<128x64xi32, #mma>
            %218 = arith.sitofp %217 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
            %219 = arith.mulf %121, %218 : tensor<128x64xf32, #mma>
            %220 = arith.mulf %219, %cst : tensor<128x64xf32, #mma>
            %221 = arith.addf %215, %220 : tensor<128x64xf32, #mma>
            %222 = "tt.reduce"(%221) <{axis = 1 : i32}> ({
            ^bb0(%arg34: f32, %arg35: f32):
              %243 = arith.maxnumf %arg34, %arg35 : f32
              tt.reduce.return %243 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %223 = arith.maxnumf %arg30, %222 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %224 = tt.expand_dims %223 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %225 = tt.broadcast %224 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %226 = arith.subf %221, %225 : tensor<128x64xf32, #mma>
            %227 = math.exp2 %226 : tensor<128x64xf32, #mma>
            %228 = "tt.reduce"(%227) <{axis = 1 : i32}> ({
            ^bb0(%arg34: f32, %arg35: f32):
              %243 = arith.addf %arg34, %arg35 : f32
              tt.reduce.return %243 : f32
            }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %229 = arith.subf %arg30, %223 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %230 = math.exp2 %229 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %231 = tt.expand_dims %230 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
            %232 = tt.broadcast %231 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
            %233 = arith.mulf %arg28, %232 : tensor<128x64xf32, #mma>
            %234 = arith.mulf %arg29, %230 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %235 = arith.addf %234, %228 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
            %236 = arith.truncf %227 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
            %237 = ttg.convert_layout %236 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
            %238 = ttg.local_alloc %209 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared2, #smem>
            %239 = ttg.local_load %238 : !ttg.memdesc<64x64xf16, #shared2, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
            %240 = tt.dot %237, %239, %233, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
            %241 = tt.addptr %arg32, %125 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
            %242 = ttg.local_alloc %204 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
            scf.yield %240, %235, %223, %192, %241, %242 : tensor<128x64xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked1>, !ttg.memdesc<64x64xf16, #shared1, #smem, mutable>
          }
          %138 = arith.subi %79, %81#3 : i32
          %139 = arith.addi %138, %c63_i32 : i32
          %140 = arith.divsi %139, %c64_i32 : i32
          %141 = arith.subi %140, %c1_i32 : i32
          %142 = arith.maxsi %141, %c0_i32 : i32
          %143 = arith.muli %142, %c64_i32 : i32
          %144 = arith.addi %81#3, %143 : i32
          %145 = arith.cmpi sge, %140, %c1_i32 : i32
          %146 = tt.splat %144 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %147 = tt.splat %144 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %148 = arith.addi %146, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
          %149 = arith.addi %147, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %150 = tt.expand_dims %148 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
          %151 = ttg.local_load %137#5 : !ttg.memdesc<64x64xf16, #shared1, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
          %152 = tt.expand_dims %149 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
          %153 = arith.cmpi slt, %152, %cst_3 : tensor<64x1xi32, #blocked1>
          %154 = tt.broadcast %153 : tensor<64x1xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
          %155 = tt.splat %145 : i1 -> tensor<64x64xi1, #blocked1>
          %156 = arith.andi %155, %154 : tensor<64x64xi1, #blocked1>
          %157 = tt.load %137#4, %156, %cst_13 : tensor<64x64x!tt.ptr<f16>, #blocked1>
          %158 = tt.broadcast %150 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
          %159 = arith.cmpi sge, %119, %158 : tensor<128x64xi32, #mma>
          %160 = arith.select %159, %cst_5, %cst_1 : tensor<128x64xi1, #mma>, tensor<128x64xf32, #mma>
          %161 = scf.if %145 -> (tensor<128x64xf32, #mma>) {
            %192 = tt.dot %76, %151, %cst_5, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
            scf.yield %192 : tensor<128x64xf32, #mma>
          } else {
            scf.yield %cst_5 : tensor<128x64xf32, #mma>
          }
          %162 = arith.mulf %161, %cst_4 : tensor<128x64xf32, #mma>
          %163 = arith.addf %160, %162 : tensor<128x64xf32, #mma>
          %164 = arith.subi %119, %158 : tensor<128x64xi32, #mma>
          %165 = math.absi %164 : tensor<128x64xi32, #mma>
          %166 = arith.sitofp %165 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
          %167 = arith.mulf %121, %166 : tensor<128x64xf32, #mma>
          %168 = arith.mulf %167, %cst : tensor<128x64xf32, #mma>
          %169 = arith.addf %163, %168 : tensor<128x64xf32, #mma>
          %170 = "tt.reduce"(%169) <{axis = 1 : i32}> ({
          ^bb0(%arg27: f32, %arg28: f32):
            %192 = arith.maxnumf %arg27, %arg28 : f32
            tt.reduce.return %192 : f32
          }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %171 = arith.maxnumf %137#2, %170 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %172 = tt.expand_dims %171 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %173 = tt.broadcast %172 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
          %174 = arith.subf %169, %173 : tensor<128x64xf32, #mma>
          %175 = math.exp2 %174 : tensor<128x64xf32, #mma>
          %176 = "tt.reduce"(%175) <{axis = 1 : i32}> ({
          ^bb0(%arg27: f32, %arg28: f32):
            %192 = arith.addf %arg27, %arg28 : f32
            tt.reduce.return %192 : f32
          }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %177 = arith.subf %137#2, %171 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %178 = math.exp2 %177 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %179 = tt.expand_dims %178 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
          %180 = tt.broadcast %179 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
          %181 = arith.mulf %137#0, %180 : tensor<128x64xf32, #mma>
          %182 = arith.mulf %137#1, %178 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %183 = arith.addf %182, %176 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %184 = arith.truncf %175 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
          %185 = ttg.convert_layout %184 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
          %186 = ttg.local_alloc %157 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared2, #smem>
          %187 = ttg.local_load %186 : !ttg.memdesc<64x64xf16, #shared2, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
          %188 = scf.if %145 -> (tensor<128x64xf32, #mma>) {
            %192 = tt.dot %185, %187, %181, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
            scf.yield %192 : tensor<128x64xf32, #mma>
          } else {
            scf.yield %181 : tensor<128x64xf32, #mma>
          }
          %189 = arith.select %145, %188, %137#0 : tensor<128x64xf32, #mma>
          %190 = arith.select %145, %183, %137#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %191 = arith.select %145, %171, %137#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          scf.yield %191, %190, %189 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>
        } else {
          scf.yield %81#0, %81#1, %81#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128x64xf32, #mma>
        }
        %84 = tt.expand_dims %83#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
        %85 = arith.divf %cst_7, %84 : tensor<128x1xf32, #mma>
        %86 = tt.broadcast %85 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
        %87 = arith.mulf %83#2, %86 : tensor<128x64xf32, #mma>
        %88 = arith.truncf %87 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
        %89 = arith.cmpi slt, %3, %c0_i32 : i32
        %90 = arith.cmpi sgt, %17, %c0_i32 : i32
        %91 = arith.andi %89, %90 : i1
        %92 = scf.if %91 -> (tensor<128x64xf16, #mma>) {
          %112 = arith.cmpi sge, %27, %cst_0 : tensor<128x1xi32, #mma>
          %113 = tt.broadcast %112 : tensor<128x1xi1, #mma> -> tensor<128x64xi1, #mma>
          %114 = arith.select %113, %88, %cst_6 : tensor<128x64xi1, #mma>, tensor<128x64xf16, #mma>
          scf.yield %114 : tensor<128x64xf16, #mma>
        } else {
          scf.yield %88 : tensor<128x64xf16, #mma>
        }
        %93 = arith.muli %2, %c196608_i32 : i32
        %94 = tt.addptr %arg3, %93 : !tt.ptr<f32>, i32
        %95 = arith.muli %1, %c8192_i32 : i32
        %96 = tt.addptr %94, %95 : !tt.ptr<f32>, i32
        %97 = tt.splat %96 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
        %98 = tt.addptr %97, %12 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %99 = arith.subi %17, %c8192_i32 : i32
        %100 = arith.cmpi sgt, %99, %c0_i32 : i32
        scf.if %100 {
          %112 = arith.subi %c8320_i32, %17 : i32
          %113 = tt.splat %112 : i32 -> tensor<128xi32, #blocked2>
          %114 = arith.cmpi slt, %6, %113 : tensor<128xi32, #blocked2>
          %115 = math.log2 %83#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %116 = arith.addf %83#0, %115 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %117 = ttg.convert_layout %116 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
          tt.store %98, %117, %114 : tensor<128x!tt.ptr<f32>, #blocked2>
        } else {
          %112 = math.log2 %83#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %113 = arith.addf %83#0, %112 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
          %114 = ttg.convert_layout %113 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked2>
          tt.store %98, %114 : tensor<128x!tt.ptr<f32>, #blocked2>
        }
        %101 = arith.muli %2, %arg14 : i32
        %102 = tt.addptr %arg4, %101 : !tt.ptr<f16>, i32
        %103 = arith.muli %1, %arg15 : i32
        %104 = tt.addptr %102, %103 : !tt.ptr<f16>, i32
        %105 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #mma>
        %106 = arith.muli %27, %105 : tensor<128x1xi32, #mma>
        %107 = tt.splat %104 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
        %108 = tt.addptr %107, %106 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
        %109 = tt.broadcast %108 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x64x!tt.ptr<f16>, #mma>
        %110 = tt.addptr %109, %38 : tensor<128x64x!tt.ptr<f16>, #mma>, tensor<128x64xi32, #mma>
        %111 = arith.select %100, %72, %cst_18 : tensor<128x64xi1, #mma>
        tt.store %110, %92, %111 : tensor<128x64x!tt.ptr<f16>, #mma>
      }
      scf.yield %c1_i32 : i32
    }
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(decompose-unsupported-amd-conversions{arch=gfx942}, optimize-amd-lds-usage{lds-limit=0 target-arch=gfx942}, convert-scf-to-cf, convert-index-to-llvm{index-bitwidth=0}, allocate-shared-memory, triton-amdgpu-membar-analysis, triton-amdgpu-refine-ops{arch=gfx942}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true})",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/flash-attention.py:527:0: error: Failures have been detected while processing an MLIR pass pipeline
/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/flash-attention.py:527:0: note: Pipeline failed while executing [`TritonAMDGPURefineOps` on 'builtin.module' operation]: reproducer generated at `std::errs, please share the reproducer above with Triton project.`
=============================== warnings summary ===============================
../../../../triton/python/triton/runtime/autotuner.py:105
  /home/dtanner/repos/triton/python/triton/runtime/autotuner.py:105: DeprecationWarning: warmup, rep, and use_cuda_graph parameters are deprecated. See https://github.com/triton-lang/triton/pull/4496 for details.
    warnings.warn(("warmup, rep, and use_cuda_graph parameters are deprecated. See "

perf-kernels/fa_workspace/flash-attention.py::test_op_fwd[bshd-True-True-1-4-2-16384-16384-128]
  /opt/conda/envs/py_3.10/lib/python3.10/site-packages/_pytest/runner.py:224: PluggyTeardownRaisedWarning: A plugin raised an exception during an old-style hookwrapper teardown.
  Plugin: skipping, Hook: pytest_runtest_makereport
  KeyboardInterrupt: 
  For more information see https://pluggy.readthedocs.io/en/stable/api_reference.html#pluggy.PluggyTeardownRaisedWarning
    report: TestReport = hook.pytest_runtest_makereport(item=item, call=call)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED flash-attention.py::test_op_fwd[bshd-True-True-4-48-24-1024-1024-64] - RuntimeError: PassManager::run failed
FAILED flash-attention.py::test_op_fwd[bshd-True-True-1-24-6-8192-8192-64] - RuntimeError: PassManager::run failed
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! KeyboardInterrupt !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
/opt/conda/envs/py_3.10/lib/python3.10/ast.py:50: KeyboardInterrupt
(to show a full traceback on KeyboardInterrupt use --full-trace)
======================== 2 failed, 2 warnings in 6.25s =========================
