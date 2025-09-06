import pandas as pd
import argparse
import os
import shutil
from collections import defaultdict
import numpy as np
import math
import triton
import torch

def get_num_sms():
    # Returns the Compute Unit count of the current device
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms

def calculate_mem_bw(batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, block_size, time_us, use_3d):
    Q = seq_q_l * num_heads_q * head_size * 2
    K = V = seq_kv_l * num_heads_k * head_size * 2
    if use_3d == False:
        out = Q
    else:
       
       num_splits = calculate_num_seq_decode_split(batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, block_size)
       main_piece = (num_heads_q * num_splits)
       out = (main_piece * triton.next_power_of_2(head_size) + 2*main_piece) * 4 # float32
    mem = (Q + K + V + out) * batch_size
    return (mem / 1e9) / (time_us * 1e-6)


def calculate_num_seq_decode_split(batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, block_size):
    BLOCK_M = 16
    num_queries_per_kv = num_heads_q // num_heads_k
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    # TODO (cagri): what happens when BLOCK_M < num_queries_per_kv?
    # should we consider that case?
    if BLOCK_Q == 0:
        BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
        BLOCK_Q = BLOCK_M // num_queries_per_kv
    # q.shape[0] = batch_size * seq_q_l
    q_shape_0 = batch_size * seq_q_l
    total_num_q_blocks = q_shape_0 // BLOCK_Q + batch_size
    target_num_prgms = get_num_sms() * 2
    NUM_SEGMENTS = math.ceil(target_num_prgms / (total_num_q_blocks * num_heads_k))
    NUM_SEGMENTS = triton.next_power_of_2(NUM_SEGMENTS) * 2
    NUM_SEGMENTS = min(NUM_SEGMENTS, 256)
    MIN_SEGMENTS = 16 if block_size <= 16 else 8
    NUM_SEGMENTS = max(NUM_SEGMENTS, MIN_SEGMENTS)
    return NUM_SEGMENTS

def calculate_tflops(batch_size, seq_q_l, seq_kv_l, num_heads_q, num_heads_k, head_size, time_us):
    # FLOPs for QK^T (multiply + add), divide by 2 for causal masking
    flops_qk = (2.0 * batch_size * seq_q_l * seq_kv_l * num_heads_q * head_size) // 2

    # FLOPs for A x V (multiply + add), divide by 2 for causal masking
    flops_av = (2.0 * batch_size * seq_q_l * seq_kv_l * num_heads_q * head_size) // 2
    # Total FLOPs
    total_flops = flops_qk + flops_av

    time_s = time_us * 1e-6

    # TFLOPs = total FLOPs / (time in seconds * 1e12)
    tflops = total_flops / (time_s * 1e12)
    return tflops

def match_name(candidate, names):
    for n in names:
        if candidate in n or n in candidate:
            return True, n
    return False, None

parser = argparse.ArgumentParser(description="")
parser.add_argument('--num_heads_q', type=int, default=16, help='')
parser.add_argument('--num_heads_k', type=int, default=2, help='')
parser.add_argument('--head_size', type=int, default=64, help='')
parser.add_argument('--seq_q_l', type=int, default=1, help='')
parser.add_argument('--seq_kv_l', type=int, default=1024, help='')
parser.add_argument('--bs', type=int, default=1, help='')
parser.add_argument('--window_size', type=int, default=0, help='')
parser.add_argument('--block_size', type=int, default=16, help='')
parser.add_argument('--path', type=str, default="res", help='')
parser.add_argument('--repeat', type=int, default=1000, help='')
parser.add_argument('--kernel_names', type=str, nargs='*', help='')

args = parser.parse_args()

print(args.kernel_names)
repeat = args.repeat
path = args.path
kernel_names = args.kernel_names

data = pd.read_csv("results_attn.csv")
kernel_data = defaultdict(list)
name_map = dict()
for i, f_name in enumerate(data['Kernel_Name']):
    match, matched_name = match_name(f_name, kernel_names)
    if match:
        time = data['End_Timestamp'][i] - data['Start_Timestamp'][i]
        kernel_data[f_name].append(time * 10**-3)
        name_map[f_name] = matched_name

res_dict = dict()
for k, vals in kernel_data.items():
    # get only the actual runs, not tuning ones
    vals = vals[-repeat:]
    # remove warmup runs
    warm_cnt = len(vals) // 10
    vals = vals[warm_cnt:]
    vals = np.sort(vals)
    outliers = len(vals) // 5
    vals = vals[outliers:-outliers]
    if len(vals) == 0:
        continue
    results = [np.mean(vals), np.min(vals), np.max(vals), np.std(vals),np.median(vals)]
    mem_BW = calculate_mem_bw(args.bs, args.seq_q_l, args.seq_kv_l, args.num_heads_q, args.num_heads_k, args.head_size, args.block_size, np.mean(vals), "3d" in k)
    tflops = calculate_tflops(args.bs, args.seq_q_l, args.seq_kv_l, args.num_heads_q, args.num_heads_k, args.head_size, np.mean(vals))
    print(f"{k}:{args.bs=},{args.window_size=},{args.seq_q_l=},{args.seq_kv_l=},{args.num_heads_q=},{args.num_heads_k=},{args.head_size=}, {results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]},{mem_BW=}, {tflops=}")
    k = name_map[k]
    file_path = f"{path}/{k}_data.csv"
    if not os.path.exists(file_path):
        with open(file_path, "w") as fptr:
            print("batch_size,prefill_cnt,decode_cnt,window_size,seq_q_len,seq_kv_len,num_heads_q,num_heads_k,head_size,avg,min,max,std,median,BW(GB/s),TFLOPs", file=fptr)
    with open(file_path, "a") as fptr:
        print(f"{args.bs},{args.window_size},{args.seq_q_l},{args.seq_kv_l},{args.num_heads_q},{args.num_heads_k},{args.head_size},{results[0]},{results[1]}, {results[2]}, {results[3]}, {results[4]},{mem_BW}, {tflops}", file=fptr)