from triton.tools.env import getTritonBasePath
from triton.tools.mxfp import MXFP4Tensor, MXFP6Tensor, MXScaleTensor
import triton
import warnings

import os
import torch


def test_fp6():
    data = torch.tensor([1, 2, 3, 4, 5, 6, 0, -3])
    a = MXFP6Tensor(data=data)
    a_f32 = a.to(torch.float32)
    print(f"data_f32 = \n{data}")
    print(f"a_f32 = \n{a_f32}")

    b = a.to_packed_tensor(dim=0)
    print(f"b = {b}")
    c = a.unpack_packed_tensor(b, dim=0)
    print(f"c = \n{c}")
    d = a.to(torch.float32)
    print(f"d = {d}")
