from sim.aot import aot_compile
from sim.mi400Simulator import MI400Simulator
from sim.sim_arguments import Arguments
from triton.tools.env import getTritonBasePath

import os

import torch

base_configs = [
    {"N_CTX": 64, "BLOCK_M": 64, "BLOCK_N": 64, "HEAD_DIM": 64, "NUM_WARPS": 4},
    {"N_CTX": 2048, "BLOCK_M": 64, "BLOCK_N": 64, "HEAD_DIM": 128, "NUM_WARPS": 4},
    {"N_CTX": 128, "BLOCK_M": 128, "BLOCK_N": 64, "HEAD_DIM": 64, "NUM_WARPS": 4},
    {"N_CTX": 32, "BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 128, "NUM_WARPS": 2},
    {"N_CTX": 128, "BLOCK_M": 128, "BLOCK_N": 64, "HEAD_DIM": 128, "NUM_WARPS": 4},
    {"N_CTX": 128, "BLOCK_M": 128, "BLOCK_N": 128, "HEAD_DIM": 128, "NUM_WARPS": 4},
]


def generate_configs():
    configs = []
    for dtype in ["float16"]:
        for config in base_configs:
            new_config = config.copy()
            new_config["DTYPE"] = dtype
            configs.append(new_config)
    print(configs)
    return configs


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    m = torch.max(x, axis=1).values[:, None]
    # e_x = x-m
    e_x = torch.exp(x - m)
    return e_x / e_x.sum(axis=1)[:, None]  # only difference


def attn(Q, K, V):
    S = torch.matmul(Q.float(), torch.transpose(K.float(), 0, 1))
    # P = torch.softmax(S.float(), dim=-1)
    P = softmax(S.float())
    # P = torch.softmax(S.float(), dim=1)
    # P = S.float()
    O = torch.matmul(P, V.float()).to(Q.dtype)
    # O = S.half()
    return O


def create_config_id(config):
    config_key_value_pairs = [f"{key}_{value}" for key, value in config.items()]
    return "_".join(config_key_value_pairs)


def create_output_dir(config):
    return os.path.join(getTritonBasePath(), "fa", create_config_id(config))


def test_fa(config):
    print(f"Compiling {create_config_id(config)}")

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLK_SLICE_FACTOR = 2
    HEAD_DIM = config["HEAD_DIM"]
    ACTUAL_HEAD_DIM = HEAD_DIM
    ENABLE_DROPOUT = 0
    IS_VARLEN = 0
    USE_EXP2 = 1
    IS_FP8 = 0
    FP8_MAX = 0
    FP8_RETURN_DESCALE = 0
    DEBUG_TRITON = 0
    DEBUG_TRITON_DETAIL = 0

    NUM_WAPRS = config["NUM_WARPS"]
    DTYPE = config["DTYPE"]
    H = 1
    STAGE = 1
    WAVES_PER_EU = 1

    signatureQ = "i32:16,i32:16,i32:16,i32:16"
    signatureK = "i32:16,i32:16,i32:16,i32:16"
    signatureV = "i32:16,i32:16,i32:16,i32:16"
    signatureO = "i32:16,i32:16,i32:16,i32:16"

    args = Arguments()
    ptype = "fp16"
    if DTYPE == "bfloat16":
        ptype = "bf16"

    args.arch = "gfx1251"
    #args.kernel_name = "flash_attention_kernel"
    #args.path = os.path.join(getTritonBasePath(), "mi400/kernels/flash_attention_kernel.py")
    args.kernel_name = "_bwd_kernel_dkdv_causal"
    args.path = os.path.join(getTritonBasePath(), "mi400/kernels/bwd_prefill_split.py")
    #args.signature = f"*{ptype}:16,*{ptype}:16,*{ptype}:16"
    #args.signature += ",fp32"
    #args.signature += f"*{ptype}:16,*{ptype}:16,*{ptype}:16"
    #args.signature += ",fp32,fp32" #softmax_lse, delta
    #args.signature += f",{signatureQ},{signatureK},{signatureV}"
    #args.signature += f",{signatureQ},{signatureK},{signatureV}"
    #args.signature += f",{signatureQ},{signatureK}"
    #args.signature += ",1,1" #nheads_q, nheads_k
    #args.signature += ",1,1,i32:16"
    #args.signature += f",{BLOCK_M}, {BLOCK_N}, {HEAD_DIM}, {STAGE}"
    args.signature = "*fp16,*fp16,*fp16, fp32, *fp16,*fp16,*fp16, *fp32,*fp32, i32,i32,i32,i32, i32,i32,i32,i32, i32,i32,i32,i32, i32,i32,i32,i32, i32,i32,i32, i32,i32,i32,i32, i32,i32,i32,i32, i32,i32,i32,i32, i32,i32, i32,i32, i32,i32, *fp32,fp32,i32,i32, fp32,fp32,fp32,fp32"
    args.signature += f",{BLOCK_M}, {BLOCK_N}, {BLK_SLICE_FACTOR}, {HEAD_DIM}, {ACTUAL_HEAD_DIM}, {ENABLE_DROPOUT}, {IS_VARLEN}, {USE_EXP2}, {IS_FP8}, {FP8_MAX}, {FP8_RETURN_DESCALE}, {DEBUG_TRITON}, {DEBUG_TRITON_DETAIL}"

    # Store each config in a subfolder
    args.out_path = create_output_dir(config)
    args.num_warps = NUM_WAPRS
    args.num_stages = 3
    args.waves_per_eu = WAVES_PER_EU
    shaderInfo = aot_compile(args)

    # For reproducibility and debuggability
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=30, linewidth=100000)

    sim = MI400Simulator(args.out_path)
    torch_type = getattr(torch, DTYPE)
    q = (torch.rand((N_CTX, HEAD_DIM))).to(torch_type)
    k = (torch.rand((N_CTX, HEAD_DIM))).to(torch_type)
    v = (torch.rand((N_CTX, HEAD_DIM))).to(torch_type)

    # q = (torch.randint(1, 2, (N_CTX, HEAD_DIM))).to(torch.float16)
    # k = (torch.randint(1, 3, (N_CTX, HEAD_DIM))).to(torch.float16)
    # v = (torch.randint(1, 3, (N_CTX, HEAD_DIM))).to(torch.float16)
    # To reproduce num_warps==1 error for large configs
    # for n in range(0, N_CTX):
    #     for m in range(0, HEAD_DIM):
    #         k[(n, m)] = n
    # print(k)

    o = attn(q, k, v)

    addressQ = sim.createInputSurface(q)
    dimsQ = [HEAD_DIM * N_CTX * H, HEAD_DIM * N_CTX, HEAD_DIM]
    addressK = sim.createInputSurface(k)
    dimsK = [HEAD_DIM * N_CTX * H, HEAD_DIM * N_CTX, HEAD_DIM]
    addressV = sim.createInputSurface(v)
    dimsV = [HEAD_DIM * N_CTX * H, HEAD_DIM * N_CTX, HEAD_DIM]
    addressO = sim.createOutputSurface(o)
    dimsO = [HEAD_DIM * N_CTX * H, HEAD_DIM * N_CTX, HEAD_DIM]
    kargs = [addressQ, addressK, addressV, addressO, 1.0] + dimsQ + dimsK + dimsV + dimsO + [N_CTX]
    numBlocks = int(N_CTX / BLOCK_M)
    grid = [numBlocks, 1, 1]
    print(shaderInfo)
    sim.createArgs(kargs, grid)
    sim.launch(args.num_warps, grid, shaderInfo)
    sim.done()


def generate_test_config(configs, variance):
    config_str = ""
    for config in configs:

        config_str += f"""
        {{
                "mi400_fa_test_test_{create_config_id(config)}",
                "{create_output_dir(config)}/flash_attention_kernel.sp3",
                "{create_output_dir(config)}/memory_surface.ini",
                "{create_output_dir(config)}/reg_seq.ini",
                {variance}f, /*Variance*/
        }},
        """
    print(config_str)


if __name__ == "__main__":
    configs = generate_configs()
    for config in configs:
        test_fa(config)
    generate_test_config(configs, 0.01)
