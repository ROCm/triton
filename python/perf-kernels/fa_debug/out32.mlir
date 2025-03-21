/home/dtanner/repos/rocm_triton/python/perf-kernels/fa_workspace/./flash-attention.py:2091: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /var/lib/jenkins/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:517.)
  fn = lambda: torch.nn.functional.scaled_dot_product_attention(
INFO: running a single config given from a file
fused-attention-fwd-d128-layoutthd:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K     triton      torch
0    2.0  16.0  16.0   8192.0   8192.0  71.137712  15.238919
