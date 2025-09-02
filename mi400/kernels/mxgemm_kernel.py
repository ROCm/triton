import triton
import triton.language as tl

# @triton.jit
def getMixedType(fpflag):
    if fpflag == 4:
        return "e2m1"
        # accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)
    elif fpflag == 62:
        return "e2m3"
        # accumulator = tl.dot_scaled(a, scale_a, "e2m3", b, scale_b, "e2m3", accumulator)
    elif fpflag == 63:
        return "e3m2"
        # accumulator = tl.dot_scaled(a, scale_a, "e3m2", b, scale_b, "e3m2", accumulator)
    else:
        return "e5m2"
        # accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e5m2", accumulator)

@triton.jit
def mxgemm_kernel(
        a_ptr, b_ptr, output_ptr,
        a_scale, b_scale,
        M, N, K,
        stride_scale: tl.constexpr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        fpflag_a :tl.constexpr,
        fpflag_b :tl.constexpr,
        SCALE_BLOCK : tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, USE_TDM : tl.constexpr):
    DIV_FACTOR_A: tl.constexpr = 2 if fpflag_a == 4 else 1
    DIV_FACTOR_B: tl.constexpr = 2 if fpflag_b == 4 else 1
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_k_a = tl.arange(0, BLOCK_K // DIV_FACTOR_A)
    offs_k_b = tl.arange(0, BLOCK_K // DIV_FACTOR_B)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_scale_k = tl.arange(0, BLOCK_K // SCALE_BLOCK)
    a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    if USE_TDM:
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr+(pid_m*BLOCK_M)*stride_am,
            shape=(M, K),
            strides=(stride_am, 1),
            block_shape=(BLOCK_M, BLOCK_K//DIV_FACTOR_A)
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr+(pid_n*BLOCK_N)*stride_bn,
            shape=(K, N),
            strides=(stride_bk, 1),
            block_shape=(BLOCK_K//DIV_FACTOR_B, BLOCK_N))
    else:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_a[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k_b[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining_a = K - k * (BLOCK_K // DIV_FACTOR_A)
        k_remaining_b = K - k * (BLOCK_K // DIV_FACTOR_B)
        valid_k_a = offs_k_a < k_remaining_a
        valid_k_b = offs_k_b < k_remaining_b
        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)

        if USE_TDM:
            a = a_desc.load([0, k*(BLOCK_K//DIV_FACTOR_A)])
            b = b_desc.load([k*(BLOCK_K//DIV_FACTOR_B), 0])
        else:
            a = tl.load(a_ptrs, mask=valid_k_a[None, :], other=0.)
            b = tl.load(b_ptrs, mask=valid_k_b[:, None], other=0.)
            a_ptrs += (BLOCK_K // DIV_FACTOR_A) * stride_ak
            b_ptrs += (BLOCK_K // DIV_FACTOR_B) * stride_bk

        if fpflag_a == 4 and fpflag_b == 4:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)
        elif fpflag_a == 8 and fpflag_b == 8:
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e5m2", accumulator)
        elif fpflag_a == 4 and fpflag_b == 8:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e5m2", accumulator)
        elif fpflag_a == 8 and fpflag_b == 4:
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e2m1", accumulator)

        a_scale_ptr += BLOCK_K // SCALE_BLOCK
        b_scale_ptr += BLOCK_K // SCALE_BLOCK

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)