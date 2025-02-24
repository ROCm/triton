from utils.sglang_ref_prefill import extend_attention_fwd, extend_fused_attention_fwd


import logging
import time

import triton
import triton.language as tl

import sys
import torch
import pytest

import argparse

from utils.rotary_embedding import DeepseekScalingRotaryEmbedding

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


is_hip_ = is_hip()



def input_helper(B, H, prefix_length, extend_length, kv_lora_rank, qk_rope_head_dim, dtype, device):
    
    q_extend = torch.randn(B * extend_length, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    # kv_cache = torch.randn(B * (prefix_length + extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)

    # extend parts
    k_extend = torch.randn(B * (extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_extend = k_extend[..., :kv_lora_rank]
    o_extend = torch.empty(B*extend_length, H, kv_lora_rank, dtype=dtype, device=device)

    # extend indexing
    qo_indptr = torch.arange(B + 1, device=device) * (extend_length) # 0, extend_length, extend_length*2
    
    # prefix parts
    k_buffer = torch.randn(B * (extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_buffer = k_buffer[..., :kv_lora_rank]

    # prefix indexing
    kv_indptr = torch.arange(B + 1, device=device) * prefix_length # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(B*(prefix_length), device=device)

    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length

    return q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend


def input_helper_fused(B, H, prefix_length, extend_length, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device):
    
    q_extend = torch.randn(B * extend_length, H, v_head_dim + qk_rope_head_dim, dtype=dtype, device=device)
    # kv_cache = torch.randn(B * (prefix_length + extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)

    # extend parts
    k_extend = torch.randn(B * (extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_extend = k_extend[..., :kv_lora_rank]
    o_extend = torch.empty(B*extend_length, H, v_head_dim, dtype=dtype, device=device)

    # extend indexing
    qo_indptr = torch.arange(B + 1, device=device) * (extend_length) # 0, extend_length, extend_length*2
    
    # prefix parts
    k_buffer = torch.randn(B * (extend_length), kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    v_buffer = k_buffer[..., :kv_lora_rank]

    # prefix indexing
    kv_indptr = torch.arange(B + 1, device=device) * prefix_length # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(B*(prefix_length), device=device)



    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length


    w_kc = torch.randn(H, kv_lora_rank, v_head_dim)
    w_vc = torch.randn(H, kv_lora_rank, v_head_dim)


    return q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc


# def input_helper(B, H, S, kv_lora_rank, rotary_dim, qk_rope_head_dim, num_kv_splits, dtype, device, rope_base=10,
#                  rope_max_seq_len=16324, rope_scaling=1.0, is_neox_style=True):
#     q = torch.randn(B, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
#     kv_cache = torch.randn(B * S, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)

#     # interlancing [batch_start_off, batch_seq_len, batch_start_off, batch_seq_len, ...,]
#     kv_indptr = torch.arange(B + 1, device=device) * S
#     kv_indices = torch.arange(B*S, device=device)

#     attn_logits = torch.empty(B, H, num_kv_splits, kv_lora_rank + 1, dtype=dtype, device=device)

#     rotary_emb = DeepseekScalingRotaryEmbedding(
#         qk_rope_head_dim,
#         rotary_dim,
#         rope_max_seq_len,
#         rope_base,
#         is_neox_style,
#         rope_scaling,
#         q.dtype,
#         device=device,
#     )

#     positions = torch.tensor([S], device=device).unsqueeze(0).repeat(B, 1)  # k positions and q position as last

#     return kv_indptr, kv_indices, q, kv_cache, attn_logits, rotary_emb, positions


# @pytest.mark.parametrize('B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim', [
#     (32, 16, 2048, 512, 128, 64),
# ])
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
# def test_op_fwd(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, dtype, num_kv_splits=32, sm_scale=1.0, logit_cap=0.0,
#                 device="cuda"):
#     torch.manual_seed(0)

#     D = qk_nope_head_dim
#     q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend = input_helper(
#         B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device)

#     # k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)

#     tri_logits = ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
#                                   logit_cap, rotary_emb, positions, device="cuda", persistent=True)

#     # reference
#     ref_logits = ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
#                                   logit_cap, rotary_emb, positions, device="cuda")

#     print("first 10 logits:")
#     print(f"ref: {ref_logits[:,:,-1].flatten()[:]}") # to debug the rope, check last split
#     print(f"tri: {tri_logits[:,:,-1].flatten()[:]}")
#     torch.testing.assert_close(ref_logits, tri_logits, atol=1e-2, rtol=1e-2)
#     print("attn_logits from stage 1 matches with ref")
#     # stage 2 is shared

# def ref_preprocess(kv_cache, kv_lora_rank):
#     latent_cache = kv_cache
#     v_input = latent_cache[..., :kv_lora_rank]
#     v_input = v_input.contiguous().unsqueeze(1)
#     k_input = latent_cache.unsqueeze(1)
#     k_input[..., :kv_lora_rank] = v_input
#     return k_input, v_input

# def ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap, rotary_emb, positions, device="cuda", persistent=False):
#     q_input = q
#     attn_logits = extend_attention_fwd(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen,
#                                             num_kv_splits, sm_scale, logit_cap, persistent=persistent)
#     return attn_logits


def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []
    x_vals_list = [
                    (1, 16, 1024, 1024, 512, 64, 128),
                    ]
    
    if args.B:
        x_vals_list = [
                    (args.B, 16, 1024, 1024, 512, 64, 128),
                    ]

    x_names = ["B", "H", "prefix", "extend", "kv_lora_rank", "qk_rope_head_dim", "v_head_dim"]

    line_vals = ["ref", "fused"]

    if args.ref:
        line_vals = ["ref"]

    if args.fused:
        line_vals = ["fused"]

    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, sm_scale, logit_cap, device,
                  provider):
        warmup = 25
        rep = 100

        

        if "ref" in provider:
            q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend = input_helper(
                    B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, dtype, device)
            fn = lambda: extend_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend)

        if "fused" in provider:
            q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, kv_indptr, kv_indices, qo_indptr, custom_mask, mask_indptr, max_len_extend, w_kc, w_vc = input_helper_fused(
                    B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, dtype, device)
            fn = lambda: extend_fused_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, w_kc, w_vc, qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr, max_len_extend)
        
        
        
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=".", print_data=True, show_plots=False)
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
    parser.add_argument("-B", type=int, default=0)
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
        if re.search(r"\.name:", line):
            vgpr_info.append(line.strip())
        if re.search(r"\.vgpr_count:", line) or re.search(r"\.vgpr_spill_count:", line):
            vgpr_info.append(line.strip())
        # Detect start of table
        if re.match(r"^\s*MLA-decode:", line):
            in_table = True
            # table_lines.append(line.strip())
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
        # os.environ["TRITON_ALWAYS_COMPILE"] = "1"
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
    
    run_bench(args)


if __name__ == "__main__":
    main()


