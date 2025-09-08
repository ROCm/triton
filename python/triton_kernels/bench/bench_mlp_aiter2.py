import os
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import triton.profiler as proton
from triton.profiler import viewer
import torch
import triton_kernels
import triton_kernels.swiglu
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation, matmul_ogs_torch
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import routing
from triton_kernels.target_info import is_hip, get_cdna_version
from triton_kernels.tensor import convert_layout
from triton_kernels.tensor_details.layout import StridedLayout, BlackwellMXScaleLayout, HopperMXScaleLayout, HopperMXValueLayout, GFX950MXScaleLayout
from triton_kernels.tensor import wrap_torch_tensor, FP4
from dataclasses import dataclass

torch.set_default_device("cuda")

if torch.cuda.is_available() and not is_hip():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

import aiter
from aiter.fused_moe import fused_topk, moe_sorting
from aiter import dtypes, QuantType
from aiter.test_common import checkAllclose
BLOCK_SIZE_M = 128
    
def shuffle_mxfp4_weight(src: torch.Tensor, NLane: int, gate_up: bool) -> torch.Tensor:
        """
        src: shape [experts_cnt, N, K_pk], where K_pk = K // 2
        Returns: shuffled tensor of shape [experts_cnt, N0*2, K0, KLane, NLane, KPack]
        """
        # print("gemm shape:", src.shape)
        experts_cnt, N, K_pk = src.shape
        if gate_up:
            N = N // 2
        KPack = 16
        KLane = 64 // NLane #4
        N0 = N // NLane
        K0 = K_pk // (KLane * KPack)
        if (gate_up):
            src_reshaped = src.view(experts_cnt, 2, N0, NLane, K0, KLane, KPack)  # [E,2, N0, NLane ,K0, KLane, KPack]
            src_reshaped = src_reshaped.permute(0, 2, 1, 4, 5, 3, 6).contiguous()  # [E, N0, 2, K0, KLane, NLane, KPack]
            interleaved = src_reshaped.view(*src.shape)
        else:
            src_reshaped = src.view(experts_cnt, N0, NLane, K0, KLane, KPack)
            interleaved = src_reshaped.permute(0, 1, 3, 4, 2, 5).contiguous().view(*src.shape)
        # print("interleaved shape:", interleaved.shape)
        return interleaved.contiguous()
    
def shuffle_mxfp4_scale(src: torch.Tensor, experts_cnt: int, gate_up: bool) -> torch.Tensor:
    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    # MXFP4 constants
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane  # 4

    # Basic dimensions
    K1 = k_ // K_Pack // K_Lane  # k_ // 8
    N1 = n_ // N_Lane // N_Pack        # n_ // 32
    real_k =32 * k_ * K_Pack * K_Lane # 1x32 quant
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"
    # print("src shape", src.shape)
    # Reshape based on moe_kind
    if gate_up:
        # Reshape to: [E, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane]
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        # Reshape to: [E, K1, K_Pack, K_Lane, N1, N_Pack, N_Lane]
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    # print("shf_scale shape:", shfl_scale.shape)
    return shfl_scale.view(*src.shape).contiguous()

def cktile_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w1_scale,
    a1_scale,
    b1,
    dtype,
    topk,
    block_size=32,
    sorted_weights=None,  # [max_num_tokens_padded]
):
    token_num = hidden_states.shape[0]
    _, n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2*2 #bit4 format
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    if w1.dtype is torch.uint32:
        D = D * 8
    out = torch.zeros((token_num, topk, D), dtype=dtype)
    # print("Run cktile_moe_stage1: M=%d, N(N*2)=%d, K=%d, topk=%d, expert=%d"%(token_num, w1.shape[1], hidden_states.shape[1], topk, w1.shape[0]))
    aiter.moe_cktile2stages_gemm1(
        hidden_states,
        w1,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        sorted_weights,
        a1_scale,
        w1_scale,
        b1,
        block_size,
    )
    return out

def cktile_moe_stage2(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w2_scale,
    a2_scale,
    b2,
    dtype,
    topk,
    block_size=32,
    sorted_weights=None,  # [max_num_tokens_padded]
):
    token_num = hidden_states.shape[0]
    D = w2.shape[1]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.zeros(
        (token_num, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    # print("Run cktile_moe_stage2: M=%d, N=%d, K=%d, topk=%d, expert=%d"%(hidden_states.shape[0]*hidden_states.shape[1], w2.shape[1], hidden_states.shape[2], topk, w2.shape[0]))

    aiter.moe_cktile2stages_gemm2(
        hidden_states,
        w2,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        sorted_weights,
        a2_scale,
        w2_scale,
        b2,
        block_size,
    )
    return out

def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), None
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 \
            else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), None
    else:
        assert dtype == "mx4", f"{dtype=}"
        w, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=1)
        w = convert_layout(wrap_torch_tensor(w, dtype=FP4), opt["value_layout"])
        w_scale = convert_layout(wrap_torch_tensor(w_scale), opt["scale_layout"])
        return w, InFlexData(), w_scale

def swiglu(x, alpha: float = 1.702, limit: float = 7.0, interleaved: bool = True):
    if interleaved:
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
    else:
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)
        
@dataclass
class PerfData:
    time: float
    flops: float
    bytes: float
    bitwidth: int
    device_type: str
    device_info: dict

    @property
    def tflops(self):
        return self.flops / self.time * 1e-3

    @property
    def tbps(self):
        return self.bytes / self.time * 1e-3

    @property
    def opint(self):
        # operational intensity
        assert self.bytes > 0
        return self.flops / self.bytes

    @property
    def max_tbps(self):
        return proton.specs.max_bps(self.device_type, self.device_info["arch"], self.device_info["bus_width"],
                                    self.device_info["memory_clock_rate"]) * 1e-12

    @property
    def max_tflops(self):
        return proton.specs.max_flops(self.device_type, self.device_info["arch"], self.bitwidth,
                                      self.device_info["num_sms"], self.device_info["clock_rate"]) * 1e-12

    @property
    def util(self) -> float:
        assert self.bitwidth in (8, 16)
        min_t_flop = self.flops / self.max_tflops * 1e-3
        min_t_bw = self.bytes / self.max_tbps * 1e-3
        return max(min_t_flop, min_t_bw) / self.time


def bench_mlp(batch, dim1, dim2, dim3, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    dev = "cuda"

    # input
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    # w1 = torch.randn((1, dim1, dim2 // TP), device=dev).repeat(n_expts_tot // EP, 1, 1)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim3), device=dev)
    #w2 = torch.randn((n_expts_tot // EP, dim2 // TP, dim3), device=dev)
    # biases
    bg = torch.randn((n_expts_tot, ), device=dev)
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot // EP, dim3), device=dev)
    

    aiter_quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)


    # -- numerics --
    optg = dict()
    opt1 = dict()
    opt2 = dict()
    if w_dtype == "mx4":
        value_layout = StridedLayout
        scale_layout = StridedLayout
        if not is_hip():
            if torch.cuda.get_device_capability()[0] == 9:
                value_layout = HopperMXValueLayout
                scale_layout = HopperMXScaleLayout
            if torch.cuda.get_device_capability()[0] == 10:
                scale_layout = BlackwellMXScaleLayout
        else:
            use_scale_preshuffling = os.environ.get("TRITON_HIP_PRESHUFFLE_SCALES", "0") == "1"
            if use_scale_preshuffling:
                scale_layout = GFX950MXScaleLayout
        opt1 = {"value_layout": value_layout, "scale_layout": scale_layout}
        opt2 = deepcopy(opt1)
        if TP > 1:
            opt2['scale_layout'] = StridedLayout
            
    w1_aiter = w1.transpose(1, 2).contiguous()
    w2_aiter = w2.transpose(1, 2).contiguous()
    
    wg, wg_flex, wg_scale = quantize(wg, "bf16", dev, **optg)
    w1, w1_flex, w1_scale = quantize(w1, w_dtype, dev, **opt1)
    w2, w2_flex, w2_scale = quantize(w2, w_dtype, dev, **opt2)
    
    w1_aiter, w1_scale_aiter = aiter_quant(w1_aiter, quant_dtype=dtypes.fp4x2)
    w2_aiter, w2_scale_aiter = aiter_quant(w2_aiter, quant_dtype=dtypes.fp4x2)
    
    w1_aiter = shuffle_mxfp4_weight(w1.storage.data.transpose(1,2).contiguous(), 16, True)
    w1_scale_aiter = shuffle_mxfp4_scale(w1_scale_aiter, n_expts_tot // EP, True)
    w2_aiter = shuffle_mxfp4_weight(w2.storage.data.transpose(1,2).contiguous(), 16, False)
    w2_scale_aiter = shuffle_mxfp4_scale(w2_scale_aiter, n_expts_tot // EP, False)

    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=wg_flex), weight_scale=wg_scale)
    # act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.702, 7.0), 2)
    pc1 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex), weight_scale=w1_scale)
    pc2 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex), weight_scale=w2_scale)

    # -- benchmark --
    fpath = Path(f"logs/{name}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/profiles/batch-{batch}.hatchet")
    fpath.parent.mkdir(parents=True, exist_ok=True)
    x_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    x = torch.randn((batch, dim1), device=dev)
    xg = x.to(wg.dtype if n_expts_tot > 1 else x_dtype)
    x = x.to(x_dtype)
    # run layer
    # proton.start(str(fpath.with_suffix('')), hook="triton")
    for i in range(1):
        logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
        if n_expts_tot > 1:
            rdata, gather_indx, scatter_indx = routing(logits, n_expts_act, False, simulated_ep=EP)
        else:
            rdata, gather_indx, scatter_indx = None, None, None
        
        topk_weights, topk_ids = fused_topk(x, logits, n_expts_act, True)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, n_expts_tot, dim1, torch.bfloat16, BLOCK_SIZE_M)
        # expert weights not be used in trition, so set to 1.
        sorted_weights = torch.ones(sorted_weights.shape)
        
        ckmoe1_out = cktile_moe_stage1(
                x, w1_aiter, w2_aiter, sorted_ids, sorted_expert_ids, num_valid_ids,
                w1_scale_aiter, None, b1, x.dtype, n_expts_act, BLOCK_SIZE_M, None
            )
        # ck1 = ck1.view(batch, n_expts_act, -1)

        # x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
        # trition moe stage1
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        x = swiglu(x, interleaved=False)
        
        # trition moe1 vs. aiter moe1
        topk_ids_reorder = topk_ids.argsort(dim=1).argsort(dim=1)
        topk_ids_reorder = topk_ids_reorder.unsqueeze(-1).expand(-1, -1, ckmoe1_out.size(2))
        x_aiter = x[gather_indx.dst_indx].reshape(batch, n_expts_act, -1)
        x_aiter = torch.gather(x_aiter, 1, topk_ids_reorder)
        checkAllclose(x_aiter, ckmoe1_out, msg="stage1 triton vs. aiter")
        
        ck2 = cktile_moe_stage2(x_aiter, w1_aiter, w2_aiter, sorted_ids, sorted_expert_ids, num_valid_ids, w2_scale_aiter, None, b2, x.dtype, n_expts_act, BLOCK_SIZE_M, sorted_weights)
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2)
        # triton moe2 vs. aiter moe2
        checkAllclose(x, ck2, msg="stage2 triton vs. aiter")
    # proton.finalize()

    # -- analyze --
    gf, _, _, info = viewer.read(fpath)
    # Now the dataframe only contains leave nodes (i.e., kernels) that perform matmuls

    # Overall perf
    # matmuls = gf.filter("MATCH ('*', c) WHERE c.'name' =~ '.*matmul.*' AND c IS LEAF").dataframe

    # moe1
    matmuls = gf.filter("MATCH ('*', c) WHERE c.'name' =~ '.*matmul.*swiglu*' AND c IS LEAF").dataframe

    # moe2
    # matmuls = gf.filter(f"MATCH ('*', c) WHERE c.'name' =~ '.*matmul.*N = {dim1}*' AND c IS LEAF").dataframe
    bytes = matmuls["bytes"].sum()
    flops = sum(matmuls[[c for c in ["flops8", "flops16"] if c in matmuls.columns]].sum())
    time = matmuls["time (ns)"].sum()
    device_type = matmuls["device_type"].iloc[0]
    device_id = matmuls["device_id"].iloc[0]
    device_info = info[device_type][device_id]
    return PerfData(time=time, flops=flops, bytes=bytes, bitwidth=x.dtype.itemsize * 8, device_type=device_type,
                    device_info=device_info)


def roofline_mlp(batch_ranges, dim1, dim2, dim3, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP=1, EP=1, name="",
                 verbose=True):
    from itertools import chain
    from bisect import bisect_left
    batches = list(chain(*[range(*r) for r in batch_ranges]))
    # collect performance data
    perfs = []
    bench_case = f"{name} ({x_dtype}x{w_dtype}, TP={TP}, EP={EP})"
    print(f"Benchmarking {bench_case}...")
    print("===============================================================")
    for batch in batches:
        bench_mlp(batch, dim1, dim2, dim3, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name)
        # perfs += [bench_mlp(batch, dim1, dim2, dim3, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name)]
        # if verbose:
        #     print(
        #         f"Batch: {batch}; Kernel Latency (us): {perfs[-1].time * 1e-3 * 1e-2}; Util: {perfs[-1].util}; TFLOPS: {perfs[-1].tflops}; TBPS: {perfs[-1].tbps}"
        #     )
    print("===============================================================")
    # machine limits
    # max_tbps = perfs[0].max_tbps
    # max_tflops = perfs[0].max_tflops
    # fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    # ax.set_xlabel("batch size (toks/expt)")
    # ax.set_ylabel("performance  [TFLOP/s]")
    # ax.set_title(f"{bench_case} roofline")
    # # add a tiny margin so points are not flush with the frame
    # xs = [batch * n_expts_act / n_expts_tot for batch in batches]
    # perf = [p.tflops for p in perfs]
    # xmin, xmax = min(xs), max(xs)
    # dx = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    # ax.set_xlim(xmin - dx, xmax + dx)
    # ax.set_ylim(100, max_tflops + 500)
    # # plot roofline
    # opints = [p.opint for p in perfs]
    # knee = bisect_left(opints, max_tflops / max_tbps) - 1
    # x_bw, x_comp = xs[:knee], xs[knee:]
    # x_bw = [x_bw[0], x_comp[0]]
    # y_bw = [opints[0] * max_tbps, max_tflops]
    # y_comp = [max_tflops] * len(x_comp)
    # ax.plot(x_bw, y_bw, "--", label=f"BW-bound  ({max_tbps:.1f} TB/s)")
    # ax.plot(x_comp, y_comp, "--", label=f"Compute-bound  ({max_tflops:.0f} TFLOP/s)")
    # # plot data
    # ax.scatter(xs, perf, marker="+")
    # ax.legend(frameon=False, loc="lower right")
    # ax.grid(True, which="both", ls=":", lw=0.5)
    # fig.tight_layout()
    # fpath = Path(f"logs/{name}/{x_dtype}-{w_dtype}-TP{TP}-EP{EP}/roofline.png")
    # plt.savefig(fpath)


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    batch_ranges_dense = [(1024, 32768, 1024)]
    batch_ranges_moe = [(128, 512, 32), (512, 32000, 128)]
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    # roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *dense_dtypes, TP=1, EP=1, name="dense")
    # roofline_mlp(batch_ranges_dense, 8192, 8192, 1, 1, *quantized_dtypes, TP=1, EP=1, name="dense")
    # roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *dense_dtypes, TP=1, EP=1, name="llama4-maverick")
    # roofline_mlp(batch_ranges_moe, 5120, 8192, 128, 4, *quantized_dtypes, TP=1, EP=1, name="llama4-maverick")

    batch_ranges_moe = [(1, 2, 1), (2, 5, 2), (8, 18, 8), (32, 65, 32), (128, 257, 128), (1024, 4100, 1024),
                        (8192, 8200, 32)]
    batch_ranges_moe = [(8,10,3)]
    # batch_ranges_moe = [(1024, 4100, 1024), (8192, 8200, 32)]
    # batch_ranges_moe = [(8192, 8200, 32)]

    quantized_dtypes = ["bf16", "mx4"]
    roofline_mlp(batch_ranges_moe, 256, 512, 256, 128, 4, *quantized_dtypes, TP=1, EP=1, name="oai")
    # roofline_mlp(batch_ranges_moe, 3072, 6144, 3072, 128, 8, *quantized_dtypes, TP=1, EP=1, name="oai")
    # roofline_mlp(batch_ranges_moe, 5888, 3072, 128, 4, *quantized_dtypes, TP=1, EP=1, name="oai")
