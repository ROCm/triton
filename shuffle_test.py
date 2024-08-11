import triton
import triton.language as tl
import torch

import numpy as np


def to_numpy(x):
    return x.cpu().numpy()


@triton.jit
def kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn, BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
    Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
    Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
    x = tl.load(Xs)
    y = tl.load(Ys)
    z = tl.dot(x, y)
    tl.store(Zs, z)


def permute_weight(x: torch.Tensor) -> torch.Tensor:
    x_ = x.clone()
    x_ = x_.view(x.shape[0], x.shape[1] // 16, 16, x.shape[2] // 32, 4, 8)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(x.shape[0], x.shape[1], x.shape[2])
    return x_


M = 128
N = 128
K = 128

x = torch.zeros((M, K), dtype=torch.float16, device="cuda")
for i in range(M):
    x[i, i] = 1
y = torch.zeros((1, N, K), dtype=torch.float16, device="cuda")
for i in range(K):
    for j in range(N):
        y[0, j, i] = i + j * K

y = permute_weight(y)
z = torch.zeros((M, N), dtype=torch.float32, device="cuda")

kernel[(1, 1, 1)](x, x.stride(0), x.stride(1), y, y.stride(2), y.stride(1), z, z.stride(0), z.stride(1), M, N, K,
                  enable_moe_lds_bypass=True, num_warps=4, matrix_instr_nonkdim=16)

ref = torch.matmul(x, y.permute([0, 2, 1]).reshape(K, N))

np.testing.assert_allclose(to_numpy(ref), to_numpy(z))
