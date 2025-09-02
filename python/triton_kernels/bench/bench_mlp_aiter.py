#!/usr/bin/env python3

import os
from pathlib import Path
import torch
import time
from dataclasses import dataclass

# Import aiter MoE components
import aiter
from aiter import dtypes, ActivationType
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.shuffle import shuffle_weight
from aiter.int4_utils import rearrange_4bit_elements, convert_int8_to_uint32_int4
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")


def shuffle_mxfp4_weight(src: torch.Tensor, NLane: int, gate_up: bool) -> torch.Tensor:
    experts_cnt, N, K_pk = src.shape
    if gate_up:
        N = N // 2
    KPack = 16
    KLane = 64 // NLane #4
    N0 = N // NLane
    K0 = K_pk // (KLane * KPack)
    if (gate_up):
        src_reshaped = src.view(experts_cnt, 2, N0, NLane, K0, KLane, KPack)
        src_reshaped = src_reshaped.permute(0, 2, 1, 4, 5, 3, 6).contiguous()
        interleaved = src_reshaped.view(*src.shape)
    else:
        src_reshaped = src.view(experts_cnt, N0, NLane, K0, KLane, KPack)
        interleaved = src_reshaped.permute(0, 1, 3, 4, 2, 5).contiguous().view(*src.shape)
    return interleaved.contiguous()


def shuffle_mxfp4_scale(src: torch.Tensor, experts_cnt: int, gate_up: bool) -> torch.Tensor:
    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane
    K1 = k_ // K_Pack // K_Lane
    N1 = n_ // N_Lane // N_Pack
    real_k = 32 * k_ * K_Pack * K_Lane
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"
    
    if gate_up:
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    
    return shfl_scale.view(*src.shape).contiguous()


def cktile_moe_stage1(hidden_states, w1, w2, sorted_token_ids, sorted_expert_ids, 
                     num_valid_ids, w1_scale, a1_scale, dtype, topk, block_size=32,
                     Activation=ActivationType.Silu, quant_type=aiter.QuantType.No, 
                     sorted_weights=None):
    token_num = hidden_states.shape[0]
    _, n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2*2

    if w1.dtype is torch.uint32:
        D = D * 8
    out = torch.empty((token_num, topk, D), dtype=dtype)
    
    aiter.moe_cktile2stages_gemm1(
        hidden_states, w1, out, sorted_token_ids, sorted_expert_ids, num_valid_ids,
        topk, sorted_weights, a1_scale, w1_scale, block_size,
    )
    return out


def cktile_moe_stage2(hidden_states, w1, w2, sorted_token_ids, sorted_expert_ids,
                     num_valid_ids, w2_scale, a2_scale, dtype, topk, block_size=32,
                     Activation=ActivationType.Silu, quant_type=aiter.QuantType.No,
                     sorted_weights=None):
    token_num = hidden_states.shape[0]
    D = w2.shape[1]

    out = torch.zeros((token_num, D), dtype=dtype, device=hidden_states.device)
    
    aiter.moe_cktile2stages_gemm2(
        hidden_states, w2, out, sorted_token_ids, sorted_expert_ids, num_valid_ids,
        topk, sorted_weights, a2_scale, w2_scale, block_size,
    )
    return out


@dataclass
class PerfData:
    time: float
    flops: float
    bytes: float
    bitwidth: int
    device_type: str = "cuda"
    device_info: dict = None

    @property
    def tflops(self):
        return self.flops / self.time * 1e-3

    @property
    def tbps(self):
        return self.bytes / self.time * 1e-3

    @property
    def opint(self):
        assert self.bytes > 0
        return self.flops / self.bytes

    @property
    def util(self) -> float:
        return self.tflops / 100.0


def bench_mlp_aiter(batch, dim1, dim2, dim3, n_expts_tot, n_expts_act, x_dtype, w_dtype, 
                   TP=1, EP=1, name="aiter_moe"):
    dev = "cuda"
    topk = 4
    BLOCK_SIZE_M = 64
    quant_type = aiter.QuantType.per_1x32
    
    # 数据类型映射
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16, 
        "fp8": dtypes.fp8,
        "mx4": dtypes.fp4x2
    }
    
    dtype = dtype_map[x_dtype]
    AQDType = dtype_map[x_dtype] 
    WQDType = dtype_map[w_dtype]
    
    # 生成输入数据
    input_data = torch.randn((batch, dim1), dtype=dtype, device=dev)
    w1 = torch.randn((n_expts_tot, dim2 * 2, dim1), dtype=dtype, device=dev)
    w2 = torch.randn((n_expts_tot, dim1, dim2), dtype=dtype, device=dev)
    score = torch.randn((batch, n_expts_tot), dtype=dtype, device=dev)
    

    topk_weights, topk_ids = fused_topk(input_data, score, topk, True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, n_expts_tot, dim1, dtype, BLOCK_SIZE_M
    )
    
    # 权重量化
    torch_quant = aiter.get_torch_quant(quant_type)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)
    
    if WQDType == dtypes.fp4x2:
        w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
    
    # 输入量化
    if quant_type == aiter.QuantType.per_1x32 and (AQDType in [dtypes.bf16, dtypes.fp16]):
        a1_qt = input_data.to(AQDType)
        a1_scale = None
    else:
        a1_qt, a1_scale = torch_quant(input_data, quant_dtype=AQDType)
    
    # 权重预处理
    w1_qt_aiter = w1_qt
    w1_scale_aiter = w1_scale
    w2_qt_aiter = w2_qt
    w2_scale_aiter = w2_scale
    
    if WQDType == torch.int4:
        w1_qt_aiter = rearrange_4bit_elements(convert_int8_to_uint32_int4(
            shuffle_weight(w1_qt_aiter, (16, 16), use_int4=True)))
        w2_qt_aiter = rearrange_4bit_elements(convert_int8_to_uint32_int4(
            shuffle_weight(w2_qt_aiter, (16, 16), use_int4=True)))
    elif (AQDType in [dtypes.bf16, dtypes.fp16]) and WQDType == dtypes.fp4x2:
        w1_qt_aiter = shuffle_mxfp4_weight(w1_qt_aiter, 16, True)
        w1_scale_aiter = shuffle_mxfp4_scale(w1_scale, n_expts_tot, True)
        w2_qt_aiter = shuffle_mxfp4_weight(w2_qt_aiter, 16, False)
        w2_scale_aiter = shuffle_mxfp4_scale(w2_scale, n_expts_tot, False)
    elif WQDType != dtypes.fp4x2:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))
    
    # 基准测试 Stage 1
    torch.cuda.synchronize()
    for _ in range(10):  # 预热
        out1 = cktile_moe_stage1(
            a1_qt, w1_qt_aiter, w2_qt_aiter, sorted_ids, sorted_expert_ids, 
            num_valid_ids, w1_scale_aiter, a1_scale, dtype, topk, BLOCK_SIZE_M,
            ActivationType.Silu, quant_type, sorted_weights
        )
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(100):
        out1 = cktile_moe_stage1(
            a1_qt, w1_qt_aiter, w2_qt_aiter, sorted_ids, sorted_expert_ids, 
            num_valid_ids, w1_scale_aiter, a1_scale, dtype, topk, BLOCK_SIZE_M,
            ActivationType.Silu, quant_type, sorted_weights
        )
    torch.cuda.synchronize()
    stage1_time = (time.perf_counter() - start_time) / 100
    
    # Stage 2 量化
    if quant_type == aiter.QuantType.per_1x32 and (AQDType in [dtypes.bf16, dtypes.fp16]):
        a2_qt = out1
        a2_scale = None
    else:
        a2_qt, a2_scale = torch_quant(out1, quant_dtype=AQDType)
    a2_qt = a2_qt.view(batch, topk, -1)
    
    # 基准测试 Stage 2
    torch.cuda.synchronize()
    for _ in range(10):  # 预热
        out2 = cktile_moe_stage2(
            a2_qt, w1_qt_aiter, w2_qt_aiter, sorted_ids, sorted_expert_ids,
            num_valid_ids, w2_scale_aiter, a2_scale, dtype, topk, BLOCK_SIZE_M,
            ActivationType.Silu, quant_type, sorted_weights
        )
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(100):
        out2 = cktile_moe_stage2(
            a2_qt, w1_qt_aiter, w2_qt_aiter, sorted_ids, sorted_expert_ids,
            num_valid_ids, w2_scale_aiter, a2_scale, dtype, topk, BLOCK_SIZE_M,
            ActivationType.Silu, quant_type, sorted_weights
        )
    torch.cuda.synchronize()
    stage2_time = (time.perf_counter() - start_time) / 100
    
    total_time = stage1_time + stage2_time
    print(f"Timing - Stage1: {stage1_time*1000:.3f}ms, Stage2: {stage2_time*1000:.3f}ms, Total: {total_time*1000:.3f}ms")
    
    # 计算性能指标
    flops = batch * dim1 * dim2 * n_expts_act * topk * 4
    bytes = (batch * dim1 + n_expts_tot * dim2 * dim1 + n_expts_tot * dim1 * dim2) * dtype.itemsize
    
    return PerfData(
        time=total_time,
        flops=flops,
        bytes=bytes,
        bitwidth=dtype.itemsize * 8
    )


def roofline_mlp_aiter(batch_ranges, dim1, dim2, dim3, n_expts_tot, n_expts_act, 
                      x_dtype, w_dtype, TP=1, EP=1, name="aiter_moe", verbose=True):
    """生成 aiter MoE 实现的性能图表"""
    from itertools import chain
    
    batches = list(chain(*[range(*r) for r in batch_ranges]))
    
    perfs = []
    bench_case = f"aiter MoE ({x_dtype}x{w_dtype}, TP={TP}, EP={EP})"
    print(f"Benchmarking {bench_case}...")
    print("=" * 65)
    
    for batch in batches:
        try:
            perf = bench_mlp_aiter(batch, dim1, dim2, dim3, n_expts_tot, n_expts_act, 
                                  x_dtype, w_dtype, TP, EP, name)
            perfs.append(perf)
            if verbose:
                print(f"Batch: {batch}; Kernel Latency (us): {perf.time*1e6:.2f}; "
                      f"Util: {perf.util:.4f}; TFLOPS: {perf.tflops:.2f}; TBPS: {perf.tbps:.2f}")
        except Exception as e:
            print(f"Error at batch {batch}: {e}")
    
    print("=" * 65)
    
    if perfs:
        print(f"Benchmark completed with {len(perfs)} successful runs")


if __name__ == "__main__":
    """主函数 - 运行 aiter MoE 基准测试"""
    # 检查硬件支持
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_gfx() == "gfx950"
    
    # 配置
    batch_ranges_moe = [(8192, 8200, 32)]
    quantized_dtypes = ["bf16", "mx4"] if has_native_mx4 else ["bf16", "fp8"]
    
    print("使用 aiter MoE 实现进行基准测试...")
    print(f"硬件支持 MX4: {has_native_mx4}")
    print(f"数据类型: {quantized_dtypes}")
    
    try:
        roofline_mlp_aiter(
            batch_ranges_moe, 
            dim1=3072,      # model_dim
            dim2=3072,      # inter_dim  
            dim3=3072,      # output_dim
            n_expts_tot=128,  # 总专家数 (减少以加快测试)
            n_expts_act=4,   # 激活专家数
            x_dtype=quantized_dtypes[0], 
            w_dtype=quantized_dtypes[1],
            TP=1, EP=1,
            name="aiter_moe_bench"
        )
        print("\n基准测试完成! 结果保存在 logs/ 目录中。")
    except Exception as e:
        print(f"基准测试失败: {e}")
        import traceback
        traceback.print_exc()
