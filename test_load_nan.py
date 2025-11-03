import torch
# from torch._dynamo.testing import rand_strided
import triton.language as tl
import triton
from torch._inductor.runtime.triton_helpers import math as tl_math, libdevice
import os
from collections.abc import Sequence
from typing import Union

def rand_strided(
    size: Sequence[int],
    stride: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    extra_size: int = 0,
) -> torch.Tensor:
    needed_size = extra_size
    if all(s > 0 for s in size):
        # only need to allocate if all sizes are non-zero
        needed_size += (
            sum((shape - 1) * stride for shape, stride in zip(size, stride)) + 1
        )
    if dtype.is_floating_point:
        if dtype.itemsize == 1:
            """
            normal distribution kernel is not implemented for fp8..
            Workaround that by creating a fp16 tensor and then cast.
            """
            buffer = torch.randn(needed_size, dtype=torch.float16, device=device).to(
                dtype=dtype
            )
        else:
            buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)

torch.manual_seed(1337)
#N, M, S = 480, 640, 896
N, M, S = 512, 640, 896
x0 = rand_strided((N, M), (S, 1), device="cuda", dtype=torch.bfloat16)
x1 = torch.randint(10, 131768, (N,), device="cuda", dtype=torch.int64)

#print(f"{x1=}")

CALL_LOG = os.getenv("CALL_LOG", "1") == "1"

def fx_call(x0, x1):
    tmp1 = x0.float()
    tmp3 = x1.float()[:, None]
    tmp5 = tmp3 + 1.0
    tmp7 = tmp5 * 0.0001220703125
    tmp8 = tmp7.floor()
    tmp10 = (tmp8 + 1.0).log()
    out = (tmp1 * (tmp10 * 0.1 + 1.0)).bfloat16()
    return out

@triton.jit
def triton_kernel(in_ptr0, in_ptr1, out_ptr0, tmp, xnumel, XBLOCK: tl.constexpr, CALL_LOG: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 640)
    x1 = xindex // 640
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0+896*x1), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0001220703125
    tmp7 = tmp5 * tmp6
    tmp8 = libdevice.floor(tmp7)
    #tmp9 = tl.maximum(tmp8 + tmp4, 1e-7)
    tmp9 = tmp8 + tmp4
    tmp10 = tl_math.log(tmp9)
    tmp11 = 0.1
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 + tmp4
    tmp14 = tmp1 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp15, xmask)
    #tl.store(tmp+xindex, tmp3)


def triton_call(x0, x1):
    act = torch.empty((N, M), device="cuda", dtype=torch.bfloat16)
    XBLOCK=1024
    gridSize = (M * N + XBLOCK - 1) // XBLOCK
    print(f"{gridSize=}")
    tmp = torch.empty(gridSize*XBLOCK, device="cuda", dtype=torch.float32)
    triton_kernel[(gridSize,)](x0, x1, act, tmp, M * N, XBLOCK=XBLOCK, CALL_LOG=CALL_LOG, num_warps=4)
    return act, tmp


def main() :
    ref = fx_call(x0, x1)
    act, tmp = triton_call(x0, x1)

    #print(tmp)
    tol = 1e-3
    if not torch.allclose(ref, act, equal_nan=True, atol=tol, rtol=tol):
        print(ref)
        print(act)
        print("eager count nan", ref.isnan().sum())
        print("triton count nan", act.isnan().sum())
    else:
        print("PASS")
        print("eager count nan", ref.isnan().sum())
        print("triton count nan", act.isnan().sum())
main()
