import hip

hip.hip.hipInit(0)

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

from kernels.test_common import allclose_numpy


def generate_configs():
    base_configs = [
        {
            "M": 256,
            "N": 256,
            "K": 64,
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
        },
        # TODO: Add more shapes
    ]
    configs = []
    for config in base_configs:
        new_config = config.copy()
        configs.append(new_config)
    return configs


@gluon.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr,  #
                M, N, K,  #
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,  #
                BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr):

    BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [2, 2])

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K)):
        a = gl.load(a_ptr + offs_a, mask=offs_ak[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = gl.load(b_ptr + offs_b, mask=offs_bk[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        a = gl.convert_layout(a, gl.DotOperandLayout(0, WMMA_LAYOUT, 16))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, WMMA_LAYOUT, 16))

        accumulator = gl.amd.gfx1250.wmma(a, b, accumulator)

        offs_a += BLOCK_SIZE_K * stride_ak
        offs_b += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    gl.store(c_ptr + offs_c, accumulator)


def testGemm(config):
    print(config)
    M = config["M"]
    N = config["N"]
    K = config["K"]
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config["BLOCK_SIZE_K"]

    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1)
    gemm_kernel[grid](
        a_device, b_device, c_device,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        num_warps=4, num_ctas=1)

    c_triton = c_device.cpu().numpy()
    c_numpy = a.to(torch.float32).numpy() @ b.to(torch.float32).numpy()
    assert allclose_numpy(c_triton, c_numpy, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    for config in generate_configs():
        testGemm(config)
