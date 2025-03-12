from utils.extend_attention import extend_attention_fwd

# from MLA_flash import extend_fused_attention_fwd
from utils.extend_attention import extend_fused_attention_fwd

import logging
import time

import triton
import triton.language as tl

import sys
import torch
import pytest

import argparse

dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.int8,
    ]
}

supported_fp8 = [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]

def quantize_tensor(tensor: torch.Tensor, dtype, dim=()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantize_dim = [i for i in range(tensor.dim()) if i not in dim]
    max_vals = tensor.abs().amax(dim=quantize_dim, keepdim=True)
    max_repr_val = dtype_max[dtype]
    # Avoid division by zero
    max_vals[max_vals == 0] = 1e-8

    # Compute scale factors for each channel
    scale: torch.Tensor = max_repr_val / max_vals.to(torch.float32)

    # Quantize the tensor
    tensor = tensor * scale
    if dtype == torch.int8:
        tensor = tensor.round_()
    tensor.clamp_(-max_repr_val, max_repr_val)
    tensor_quantized = tensor.to(dtype)

    scale = scale.squeeze(dim=quantize_dim)

    return tensor_quantized, scale, 1 / scale

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

is_hip_ = is_hip()

def input_helper_fused(B, H, prefix_length, extend_length, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device, fp8=False):
    torch.manual_seed(0)
    q_extend = torch.randn(B * extend_length, H, v_head_dim + qk_rope_head_dim, dtype=dtype, device=device)

    # extend parts
    k_extend = torch.randn(B * (extend_length), 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_extend = k_extend[..., :kv_lora_rank]

    # extend indexing
    qo_indptr = torch.arange(B + 1, device=device) * (extend_length) # 0, extend_length, extend_length*2
    
    # prefix parts
    k_buffer = torch.randn(B * (extend_length), 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_buffer = k_buffer[..., :kv_lora_rank]

    # prefix indexing
    kv_indptr = torch.arange(B + 1, device=device) * prefix_length # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(B*(prefix_length), device=device)

    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length

    w_kc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device)
    w_vc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device)

    # FP8 quantization for gemms
    if fp8:
        q_extend_q, _, q_descale = quantize_tensor(q_extend, torch.float8_e4m3fnuz, dim=())
        q_descale = q_descale.item()
        q_extend = q_extend_q
        
        w_kc_q, _, w_kc_descale = quantize_tensor(w_kc, torch.float8_e4m3fnuz, dim=())
        w_kc_descale = w_kc_descale.item()
        w_kc = w_kc_q

        w_vc_q, _, w_vc_descale = quantize_tensor(w_vc, torch.float8_e4m3fnuz, dim=())
        w_vc_descale = w_vc_descale.item()
        w_vc = w_vc_q
    else:
        q_descale = None
        w_kc_descale = None
        w_vc_descale = None

    return q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc, q_descale, w_kc_descale, w_vc_descale

@pytest.mark.parametrize("B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim", [
    (2, 16, 2048, 512, 512, 64, 128),
    (2, 16, 0, 2048, 512, 64, 128),
])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('ref_attn_impl', ["absorb"])
@pytest.mark.parametrize('fuse_wkc', [False, True])
@pytest.mark.parametrize('fuse_wvc', [False, True])
@pytest.mark.parametrize('fp8', [False])
def test_op_fwd(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, ref_attn_impl, fuse_wkc, fuse_wvc, fp8, sm_scale=1.0, logit_cap=0.0, device="cuda"):
    torch.manual_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    if ref_attn_impl == "normal":
        forward = forward_normal
    else:
        forward = forward_absorb
    
    q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc, q_descale, w_kc_descale, w_vc_descale = input_helper_fused(
                    B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device, fp8=fp8)
   
    # Reference
    # torch.bmm does not support fp8 so scale back to dtype outside
    output_ref = forward(((q_extend.to(torch.float32)) * q_descale).to(dtype) if fp8 else q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                fuse_wkc, fuse_wvc, (w_kc.to(torch.float32)*w_kc_descale).to(dtype) if fp8 else w_kc, (w_vc.to(torch.float32)*w_vc_descale).to(dtype) if fp8 else w_vc, None, None, None, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused=False, fp8=False)

    # Fused
    if fp8:
        if not fuse_wkc:
            w_kc = (w_kc.to(torch.float32)*w_kc_descale).to(dtype)
            q_extend = ((q_extend.to(torch.float32)) * q_descale).to(dtype)
        if not fuse_wvc:
            w_vc = (w_vc.to(torch.float32)*w_vc_descale).to(dtype)

    output_fused = forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                   fuse_wkc, fuse_wvc, w_kc, w_vc, q_descale, w_kc_descale, w_vc_descale, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused=True, fp8=fp8)

    # Compare the outputs
    # Print debug information for mismatches
    max_mismatches = 10
    diff = (output_ref - output_fused).abs()
    mismatches = diff > 1e-2
    if mismatches.any():
        mismatch_indices = torch.nonzero(mismatches)[:max_mismatches]
        print(f"\nFound {mismatches.sum().item()} mismatches, showing first {len(mismatch_indices)}:")
        for idx in mismatch_indices:
            i, h, d = idx.tolist()
            print(f"Position [{i}, {h}, {d}]: ref={output_ref[i, h, d].item():.6f}, fused={output_fused[i, h, d].item():.6f}, diff={diff[i, h, d].item():.6f}")
    torch.testing.assert_close(output_ref, output_fused, rtol=1e-2, atol=1e-2)


def forward_absorb(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                    fuse_wkc, fuse_wvc, w_kc, w_vc, q_descale, w_kc_descale, w_vc_descale, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused, fp8=False):
    
    if not fused: # aka reference
        fuse_wkc = False
        fuse_wvc = False

    if not fuse_wkc: # 1st gemm
        q_input = torch.empty((*q_extend.shape[:-1], kv_lora_rank + qk_rope_head_dim), dtype=q_extend.dtype, device=q_extend.device)
        q_input[..., kv_lora_rank:] = q_extend[..., v_head_dim:]
        q_nope = q_extend[..., :v_head_dim]
        q_nope = torch.bmm(q_nope.transpose(0, 1), w_kc.transpose(1, 2)).to(q_input.dtype)
        q_input[..., :kv_lora_rank] = q_nope.transpose(0, 1)
    else:
        q_input = q_extend

    if fuse_wvc: # Depending on if we fuse the 2nd gemm, we output different head dim
        out = torch.empty( (*q_extend.shape[:-1], v_head_dim), dtype=v_extend.dtype, device=q_extend.device)
    else:
        out = torch.empty( (*q_extend.shape[:-1], kv_lora_rank), dtype=v_extend.dtype, device=q_extend.device)

    if not fused:
        extend_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap)
    else:
        extend_fused_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap,
                                   fuse_w_kc=fuse_wkc, fuse_w_vc=fuse_wvc, w_kc=w_kc, w_vc=w_vc, q_descale=q_descale, w_kc_descale=w_kc_descale, w_vc_descale=w_vc_descale, fp8=fp8)

    if not fuse_wvc: # 2nd gemm
        # w_vc = (w_vc * w_vc_descale).to(out.dtype)
        bmm_output = torch.bmm(out.transpose(0, 1), w_vc).to(out.dtype)
        out = bmm_output.transpose(0, 1)

    return out

def forward_normal(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                    fuse_wkc, fuse_wvc, w_kc, w_vc, q_descale, w_kc_descale, w_vc_descale, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused, fp8=False):
    
    out = torch.empty((*q_extend.shape[:-1], v_head_dim), dtype=v_extend.dtype, device=v_extend.device)

    if not fused:  # aka reference
        fuse_wkc = False
        fuse_wvc = False

    q_input = q_extend
    H = q_input.shape[1]
    
    if not fuse_wkc:  # 1st gemm
        k_extend_c = torch.einsum('zc,hcd->zhd', k_extend[..., :kv_lora_rank].squeeze(), w_kc) 
        k_extend_r = k_extend[..., kv_lora_rank:].repeat(1, H, 1)
        k_extend = torch.cat((k_extend_c, k_extend_r), dim=-1).to(q_input.dtype)
        
        k_buffer_c = torch.einsum('zc,hcd->zhd', k_buffer[..., :kv_lora_rank].squeeze(), w_kc)
        k_buffer_r = k_buffer[..., kv_lora_rank:].repeat(1, H, 1)
        k_buffer = torch.cat((k_buffer_c, k_buffer_r), dim=-1).to(q_input.dtype)

    if not fuse_wvc:  # 2nd gemm
        v_extend = torch.einsum('zc,hcd->zhd', v_extend.squeeze(), w_vc).to(q_input.dtype)
        v_buffer = torch.einsum('zc,hcd->zhd', v_buffer.squeeze(), w_vc).to(q_input.dtype)

    if not fused:
        extend_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap)
    else:
        extend_fused_attention_fwd(q_input, k_extend, v_extend, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend, sm_scale=sm_scale, logit_cap=logit_cap,
                                   fuse_w_kc=fuse_wkc, fuse_w_vc=fuse_wvc, w_kc=w_kc, w_vc=w_vc, q_descale=q_descale, w_kc_descale=w_kc_descale, w_vc_descale=w_vc_descale, fp8=fp8)

    return out
    
def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []
    x_vals_list = [
        (16, 16, 0, 4096, 512, 64, 128, "absorb"),
        (1, 16, 4096, 2048, 512, 64, 128, "absorb"),
        # (2, 16, 4096, 1024, 512, 64, 128, "absorb"), # OOMs if run with other
    ]
    
    if args.B or args.attn_impl != "":
        x_vals_list = [
            (args.B, 16, args.prefix_len, args.extend_len, 512, 64, 128, args.attn_impl),
        ]

    x_names = ["B", "H", "prefix", "extend", "kv_lora_rank", "qk_rope_head_dim", "v_head_dim", "attn_impl"]
    line_vals = ["ref", "fused"]

    if args.ref:
        line_vals = ["ref"]
    if args.fused:
        line_vals = ["fused"]

    plot_name = "MLA-decode-fuse_wkc-{}-fuse_wvc-{}".format(args.fuse_wkc, args.fuse_wvc)

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, attn_impl, sm_scale, logit_cap, device, provider):
        warmup = 25
        rep = 100

        # Prepare inputs
        q_extend, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc, q_descale, w_kc_descale, w_vc_descale = input_helper_fused(
            B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device, fp8=args.fp8 if provider == "fused" else False)

        if attn_impl == "normal":
            forward = forward_normal
        else:
            forward = forward_absorb

        # Define the function to benchmark based on provider
        if "fused" in provider:
            # torch.bmm does not support fp8 so scale back to dtype outside
            if not args.fuse_wkc and args.fp8:
                q_extend = ((q_extend.to(torch.float32)) * q_descale).to(k_extend.dtype)
                w_kc = (w_kc.to(torch.float32)*w_kc_descale).to(k_extend.dtype)
            
            if not args.fuse_wvc and args.fp8:
                w_vc = (w_vc.to(torch.float32)*w_vc_descale).to(v_extend.dtype)
            def fn():
                return forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, 
                                     custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                     args.fuse_wkc, args.fuse_wvc, w_kc, w_vc, q_descale, w_kc_descale, w_vc_descale, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused=True, fp8=args.fp8)
        
        elif "ref" in provider:
            def fn():
                return forward(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, 
                                     custom_mask, mask_indptr, max_len_extend, sm_scale, logit_cap,
                                     False, False, w_kc, w_vc, None, None, None, kv_lora_rank, qk_rope_head_dim, v_head_dim, fused=False, fp8=False)
        
        # warmup
        if args.cuda_graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            
            # Use CUDA graph for benchmarking
            torch.cuda.synchronize()  # Synchronize before capturing
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()  # Capture the function execution
            torch.cuda.synchronize()  # Synchronize after capturing
            func = g.replay
        else:
            func = fn

        # Replay the graph for benchmarking
        ms = triton.testing.do_bench(func, warmup=warmup, rep=rep)
        torch.cuda.empty_cache()
        return ms

    bench_MLA.run(save_path=None, print_data=True, show_plots=False)
    return x_vals_list, x_names, line_vals

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-dtype", default='bf16')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-fused", action="store_true", default=False)
    parser.add_argument("-ref", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("-fuse_wkc", action="store_true", default=False)
    parser.add_argument("-fuse_wvc", action="store_true", default=False)
    parser.add_argument("-attn_impl", type=str, default="")
    parser.add_argument("-cuda_graph", action="store_true", default=False)
    parser.add_argument("-B", type=int, default=0)
    parser.add_argument("-prefix_len", type=int, default=0)
    parser.add_argument("-extend_len", type=int, default=4096)
    parser.add_argument("-fp8", action="store_true", default=False)

    return parser.parse_args()

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

import re
from prettytable import PrettyTable

def parse_vgpr_usage(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Extract VGPR-related information
    vgpr_info = []
    table_lines = []
    in_table = False

    for line in lines:
        # Parse autotuning outputs
        if re.search(r"Autotuning kernel", line):
            vgpr_info.append(line.strip())
        if re.search(r"Triton autotuning for function", line):
            vgpr_info.append(line.strip())

        if re.search(r"\.name:", line):
            vgpr_info.append(line.strip())
        if re.search(r"\.vgpr_count:", line) or re.search(r"\.vgpr_spill_count:", line):
            vgpr_info.append(line.strip())
        # Detect start of table
        if re.match(r"^\s*MLA-decode", line):
            vgpr_info.append(line.strip())
            in_table = True
        elif in_table:
            table_lines.append(line.strip())

    # Print extracted information
    print("\n".join(vgpr_info))

    table = PrettyTable()
    table.field_names = table_lines[0].split()
    [table.add_row(line.split()[1:]) for line in table_lines[1:]]

    print(table)


def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)

import sys
import time
import re
import os
import tempfile

def print_vgpr(args):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        output_file = temp_file.name

        # Redirect stdout and stderr to the temporary file
        sys.stdout = temp_file
        sys.stderr = temp_file
        
        os.environ["AMDGCN_ENABLE_DUMP"] = "1"
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
        run_bench(args)  # Run the benchmark
        
        sys.stdout.flush()
        sys.stderr.flush()

    # Restore stdout and stderr to normal
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    time.sleep(0.5)  # Ensure everything is written before reading

    # Parse and print relevant output
    parse_vgpr_usage(output_file)

    # Remove the temporary file
    os.unlink(output_file)

def main():
    args = parse_args()
    if args.print_vgpr:
        print_vgpr(args)
        return 0
    # run_bench(args)
    test_op_fwd(2, 16, 2048, 2048, 512, 64, 128, torch.bfloat16, "absorb", False, True, True, 1.0, 0.0, "cuda") # sanity check for function body correctness

if __name__ == "__main__":
    main()


