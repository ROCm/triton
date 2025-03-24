import torch
import triton
import triton.language as tl
import sys
import os
import pytest
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # This goes one level up from fused-moe/
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from gemm import matmul, leaky_relu


@triton.jit
def reduce_buffers(
    o_buffers_ptr,
    o_ptr,
    M,
    K,
    N_BUFFERS,
    stride_o_buffers_m,
    stride_o_buffers_k,
    stride_om,
    stride_ok,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Reduce N_BUFFERS output tensors into a single output tensor.
    o_buffers has shape (M, K*N_BUFFERS)
    o has shape (M, K)
    """
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Create offsets for the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create mask for valid indices
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    
    # Initialize output with zeros
    o_output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=o_ptr.type.element_ty)
    
    # Iterate over each buffer and accumulate
    for buffer_idx in range(N_BUFFERS):
        # Calculate offset in the buffer tensor
        buffer_offset = buffer_idx * K
        
        # Calculate pointers to the current buffer
        o_buffer_ptrs = o_buffers_ptr + (offs_m[:, None] * stride_o_buffers_m + 
                                         (offs_k[None, :] + buffer_offset) * stride_o_buffers_k)
        
        # Load and accumulate
        o_buffer_values = tl.load(o_buffer_ptrs, mask=mask, other=0.0)
        o_output += o_buffer_values
    
    # Store the final result
    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, o_output, mask=mask)

# Problems:
# We need the output blocks per persistent workgroup to fit into L2 cache. L2 cache size is 4MB. Required size is BLOCK_SIZE_M * K * bytes_per_output_element (2 for fp16).
# With  BLOCK_SIZE_M = 32, K=4096, bytes_per_output_element=2, we get 256KB. Should be able to fit.
# M / BLOCK_SIZE_M will not saturate occupancy. Solution: We can multiply the number of launched workgroups by splitting the N dimension with N_buffers, and have a workgroup update a correct buffer for a output block.

@triton.jit
def gemm2gemm_persistent_buffered(
    a_ptr,
    b_ptr,
    c_ptr,
    acc_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    stride_ck,
    stride_accm,
    stride_accn,
    stride_om,
    stride_ok,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_WG: tl.constexpr,
    ACTIVATION: tl.constexpr,
    N_BUFFERS: tl.constexpr,
):
    """
    End to end fusion of two consecutive GEMMs.
    acc = A x B.
    out = acc x C
    A has shape (M, K), B has shape (K, N) and C has shape (N, K)

    Buffered version: we split in N dimension and store the output block in N_BUFFERS buffers. We launch a seprarate reduce_buffers kernel to sum the buffers.
    This allows us to have larger launch grid (i.e. more parallelism) for this first kernel: (M / BLOCK_SIZE_M) * N_BUFFERS.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ck > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    pids_per_WG = (num_pid_m * N_BUFFERS) // NUM_WG
    if pid < (num_pid_m * N_BUFFERS) % NUM_WG:
        pids_per_WG += 1

    pids_n_per_buffer = tl.cdiv(num_pid_n, N_BUFFERS)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # persistent kernel looping over programs:
    # one program computes a row of output blocks of out
    for _ in tl.range(0, pids_per_WG, step=1, num_stages=1):
        
        pid_m = pid % num_pid_m
        buffer_offset = pid // num_pid_m


        # gemm1
        # compute and store the row of accumulator blocks (acc = A x B) for the current buffer
        for pid_n_ in tl.range(0, pids_n_per_buffer, 1, num_stages=1):
            pid_n = pid_n_ + buffer_offset * pids_n_per_buffer
            if pid_n < num_pid_n: # check because num_pid_n / N_BUFFERS maybe not an integer # TODO: refactor. 
                # accumulator block offsets
                offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
                offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
                
                acc_ptrs = acc_ptr + (offs_m[:, None] * stride_accm + offs_n[None, :] * stride_accn)
                acc_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

                accumulator = tl.load(acc_ptrs, mask=acc_mask, other=0.0) # BLOCK_SIZE_M x BLOCK_SIZE_N

                acc_dtype = c_ptr.type.element_ty
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
                # Create pointers for first block k of A and B input matrices
                a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
                # compute the block of acc = A x B
                for k1 in range(0, num_pid_k):
                    # Load the next block of A and B, generate a mask by checking the K dimension.
                    # If it is out of bounds, set it to 0.
                    if EVEN_K:
                        a = tl.load(a_ptrs)
                        b = tl.load(b_ptrs)
                    else:
                        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k1 * BLOCK_SIZE_K, other=0.0)
                        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k1 * BLOCK_SIZE_K, other=0.0)
                    accumulator += tl.dot(a, b, out_dtype=acc_dtype)

                    # Advance the ptrs to the next K block.
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_ptrs += BLOCK_SIZE_K * stride_bk

                # Apply activation function, if specified.
                if ACTIVATION == "leaky_relu":
                    accumulator = leaky_relu(accumulaccumulatorator_add)
                
                # only this workgroup will be needing the values so coherence in L2 is enough
                # tl.atomic_add(acc_ptrs, accumulator_add, mask=acc_mask, scope="cta")
                tl.store(acc_ptrs, accumulator, mask=acc_mask)
        
        tl.debug_barrier()
        
        # for this buffer, go over the row of output blocks which share the same row of accumulator blocks (from acc = A x B)
        for k in tl.range(0, num_pid_k, step=1, num_stages=1): 
            # output block accumulator
            o_acc_dtype = o_ptr.type.element_ty
            o_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=o_acc_dtype)
            for pid_n_ in tl.range(0, pids_n_per_buffer, 1, num_stages=1):
                pid_n = pid_n_ + buffer_offset * pids_n_per_buffer
                if pid_n < num_pid_n: # check because num_pid_n / N_BUFFERS maybe not an integer # TODO: refactor. 
                    # accumulator block offsets
                    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
                    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
                    
                    acc_ptrs = acc_ptr + (offs_m[:, None] * stride_accm + offs_n[None, :] * stride_accn)
                    acc_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

                    accumulator = tl.load(acc_ptrs, mask=acc_mask, other=0.0) # BLOCK_SIZE_M x BLOCK_SIZE_N
                
                    c_ptrs = c_ptr + (offs_n[:, None] * stride_cn + offs_k[None, :] * stride_ck)
                    c_ptrs += k * BLOCK_SIZE_K * stride_ck

                    if EVEN_K:
                        c = tl.load(c_ptrs)
                    else:
                        c = tl.load(c_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                    
                    o_accumulator += tl.dot(accumulator, c, out_dtype=o_acc_dtype)

            # store the output accumulator block (to the correct buffer)
            offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_ok = k * BLOCK_SIZE_K + offs_k
            o_ptrs = o_ptr + stride_om * offs_om[:, None] + stride_ok * (offs_ok[None, :] + buffer_offset * K)
            o_mask = (offs_om[:, None] < M) & (offs_ok[None, :] < K)
            tl.store(o_ptrs, o_accumulator, mask=o_mask)
        
        pid += NUM_WG # move to next program
            
        

@triton.jit
def gemm2gemm_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    acc_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    stride_ck,
    stride_accm,
    stride_accn,
    stride_om,
    stride_ok,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_WG: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    End to end fusion of two consecutive GEMMs.
    acc = A x B.
    out = acc x C
    A has shape (M, K), B has shape (K, N) and C has shape (N, K)

    Persistent version: persistent kernel computes 
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ck > 0)

    pid_m = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    pids_per_WG = num_pid_m // NUM_WG
    if pid_m < num_pid_m % NUM_WG:
        pids_per_WG += 1

    o_acc_dtype = o_ptr.type.element_ty
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # persistent kernel looping over programs:
    # one program computes a row of output blocks of out
    for _ in tl.range(0, pids_per_WG, 1, num_stages=1):
        
        # gemm1
        # compute and store the row of accumulator blocks (acc = A x B)
        for pid_n in tl.range(0, num_pid_n, 1, num_stages=1):
            # accumulator block offsets
            offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
            offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
            
            acc_ptrs = acc_ptr + (offs_m[:, None] * stride_accm + offs_n[None, :] * stride_accn)
            acc_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
            acc_dtype = c_ptr.type.element_ty
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            # Create pointers for first block k of A and B input matrices
            a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
            for k1 in range(0, num_pid_k):
                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k1 * BLOCK_SIZE_K, other=0.0)
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k1 * BLOCK_SIZE_K, other=0.0)
                accumulator += tl.dot(a, b, out_dtype=acc_dtype)

                # Advance the ptrs to the next K block.
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            # Apply activation function, if specified.
            if ACTIVATION == "leaky_relu":
                accumulator = leaky_relu(accumulator)

            acc_ptrs = acc_ptr + (offs_m[:, None] * stride_accm + offs_n[None, :] * stride_accn)
            acc_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
            # only this workgroup will be needing these values later so coherence in L2 is enough
            tl.store(acc_ptrs, accumulator, mask=acc_mask)
            # tl.atomic_add(acc_ptrs, accumulator, mask=acc_mask, scope="cta")
        
        tl.debug_barrier()

        # gemm2
        # compute the row of output blocks loading the computed accumulator blocks and multiplying with corresponding column blocks of C (out = acc x C)
        for k in range(0, num_pid_k): 
            # output block accumulator
            o_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=o_acc_dtype)
            for pid_n in tl.range(0, num_pid_n, 1, num_stages=1):
                # accumulator block offsets
                offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
                offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
                
                acc_ptrs = acc_ptr + (offs_m[:, None] * stride_accm + offs_n[None, :] * stride_accn)
                acc_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                
                
                # The idea is that this load would be from L2
                accumulator = tl.load(acc_ptrs, mask=acc_mask, other=0.0) # BLOCK_SIZE_M x BLOCK_SIZE_N
                
                c_ptrs = c_ptr + (offs_n[:, None] * stride_cn + offs_k[None, :] * stride_ck)
                c_ptrs += k * BLOCK_SIZE_K * stride_ck

                if EVEN_K:
                    c = tl.load(c_ptrs)
                else:
                    c = tl.load(c_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

                o_accumulator += tl.dot(accumulator, c, out_dtype=o_acc_dtype)

            # store the output accumulator block
            offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_ok = k * BLOCK_SIZE_K + offs_k
            o_ptrs = o_ptr + stride_om * offs_om[:, None] + stride_ok * offs_ok[None, :]
            o_mask = (offs_om[:, None] < M) & (offs_ok[None, :] < K)
            tl.store(o_ptrs, o_accumulator, mask=o_mask)

        pid_m += NUM_WG # move to next program


@triton.jit
def gemm2gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    stride_ck,
    stride_om,
    stride_ok,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    XCD: tl.constexpr,
):
    """
    End to end fusion of two consecutive GEMMs.
    acc = A x B.
    out = acc x C
    A has shape (M, K), B has shape (K, N) and C has shape (N, K)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ck > 0)

    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # TODO: map all the pids with the same pid_m to same XCD. This is in order for them to share the same L2 cache -> atomic adds can use the L2 cache scope.
    pid_n = pid // XCD % (num_pid_n)
    pid_m = pid % XCD + (pid // (num_pid_n * XCD)) * XCD

    # pid_m = pid // num_pid_n
    # pid_n = pid % num_pid_n

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc_dtype = c_ptr.type.element_ty
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b, out_dtype=acc_dtype)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    
    # Apply activation function, if specified.
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    c_ptrs = c_ptr + (offs_n[:, None] * stride_cn + offs_k[None, :] * stride_ck)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            c = tl.load(c_ptrs)
        else:
            c = tl.load(c_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        o_partial = tl.dot(accumulator, c)
        offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_ok = k * BLOCK_SIZE_K + offs_k
        o_ptrs = o_ptr + stride_om * offs_om[:, None] + stride_ok * offs_ok[None, :]
        o_mask = (offs_om[:, None] < M) & (offs_ok[None, :] < K)
        tl.atomic_add(o_ptrs, o_partial, mask=o_mask, scope="cta")
        # tl.store(o_ptrs, o_partial, mask=o_mask)
        c_ptrs += BLOCK_SIZE_K * stride_ck


# Wrapper for gemm kernel.
def twogemms(a, b, c, o, activation="", persistent=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0] and b.shape[1] == c.shape[0], "Incompatible dimensions!!!"
    assert a.dtype == b.dtype and b.dtype==c.dtype, "Mixed dtype GEMMs are not supported!!!"
    assert a.shape[1] == c.shape[1]
    M, K = a.shape
    K, N = b.shape
    
    NUM_WG = 304  # torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M=128
    BLOCK_SIZE_N=128
    BLOCK_SIZE_K=128
    EVEN_K=K % BLOCK_SIZE_K == 0
    
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)

    args = {"num_stages": 1, "num_warps": 4, "waves_per_eu": 1}

    if persistent:
        if NUM_WG > num_pid_m: # get more parallelism by buffering the output
            
            N_BUFFERS = min(triton.cdiv(NUM_WG, num_pid_m), num_pid_n)
            grid = (NUM_WG,)
            o_buffers = torch.empty_like(o).contiguous().repeat(1, N_BUFFERS)
            acc = torch.zeros(M, N, device="cuda", dtype=o.dtype).contiguous()

            gemm2gemm_persistent_buffered[grid](
                a,
                b,
                c,
                acc,
                o_buffers,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                acc.stride(0),
                acc.stride(1),
                o_buffers.stride(0),
                o_buffers.stride(1),
                NUM_WG=NUM_WG,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                EVEN_K=EVEN_K,
                ACTIVATION=activation,
                N_BUFFERS=N_BUFFERS,
                **args,
            )

            grid_ = (triton.cdiv(M, 128), triton.cdiv(K, 128))
            
            reduce_buffers[grid_](
                o_buffers,
                o,
                M,
                K,
                N_BUFFERS,
                o_buffers.stride(0),
                o_buffers.stride(1),
                o.stride(0),
                o.stride(1),
                BLOCK_SIZE_M=128,
                BLOCK_SIZE_K=128,
            )
        else:
            grid = (NUM_WG,)
            acc = torch.zeros(M, N, device="cuda", dtype=o.dtype).contiguous()
            
            gemm2gemm_persistent[grid](
                a,
                b,
                c,
                acc,
                o,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                acc.stride(0),
                acc.stride(1), 
                o.stride(0),
                o.stride(1),
                NUM_WG=NUM_WG,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                EVEN_K=EVEN_K,
                ACTIVATION=activation,
                **args,
            )
        
    else:
        grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),) # one pid per block of output matrix
        XCD = 8 # group row of intermediate output blocks (from the first GEMM) to be in the same XCD
        gemm2gemm[grid](
            a,
            b,
            c,
            o,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            o.stride(0),
            o.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            EVEN_K=EVEN_K,
            **args,
            ACTIVATION=activation,
            XCD=XCD,
        )

def twomatmuls(a, b, c, o, activation=""):
    o_temp = torch.empty(a.shape[0], b.shape[-1], device="cuda", dtype=c.dtype)
    matmul(a, b, o_temp, 1.0, 1.0, activation=activation)
    matmul(o_temp, c, o, 1.0, 1.0)


def get_x_vals():
    x_vals = [(128*512, 4096, 4096), # non-buffered
              (4096, 4096, 4096) # buffered
              ]
    return x_vals

@pytest.mark.parametrize('M, N, K', get_x_vals())
@pytest.mark.parametrize('persistent', [True])
def test_correctness(M, N, K, persistent, dtype=torch.float32):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.randn(N, K, device="cuda", dtype=dtype)
    o_tri = torch.empty_like(a)

    twogemms(a, b, c, o_tri, activation="", persistent=persistent)
    
    o_ref = torch.empty_like(a)
    twomatmuls(a, b, c, o_ref, activation="")

    torch.testing.assert_close(o_ref, o_tri, atol=2e-2, rtol=2e-2)
    print(f"test_correctness passed for arguments M={M}, N={N}, K={K}, persistent={persistent}, dtype={dtype}.")


def benchmark(args):

    if args.M or args.N or args.K:
        assert args.M and args.N and args.K, "All M, N, K should be provided."
        x_vals = [(args.M, args.N, args.K)]
    else:
        x_vals = get_x_vals()
    
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=x_vals,
            line_arg='provider',
            line_vals=[
                'e2e', 'ref'
            ],
            line_names=[
                'e2e', 'ref'
            ],
            ylabel="ms",
            plot_name="Two layer MLP performance (ms)",
            args={},
        ))
    def bench(M, N, K, provider, dtype=torch.float16):
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)
        c = torch.randn(N, K, device="cuda", dtype=dtype)
        
        o = torch.empty_like(a)

        quantiles = [0.5, 0.2, 0.8]
        if 'e2e' in provider:
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: twogemms(a, b, c, o, activation="", persistent=True), quantiles=quantiles)
        else: # two sequential gemms
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: twomatmuls(a, b, c, o, activation=""), quantiles=quantiles)

        return ms
    bench.run(save_path=None, print_data=True, show_plots=False)

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
        if re.match(r"^\s*Two layer MLP performance", line):
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


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark gemm2gemm",
        allow_abbrev=False,
    )
    parser.add_argument("-M", type=int, default=0)
    parser.add_argument("-N", type=int, default=0)
    parser.add_argument("-K", type=int, default=0)
    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-print_vgpr", action='store_true', default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.print_vgpr:
        print_vgpr(args)
        return 0
    run_bench(args)



if __name__ == "__main__":
    test_correctness(128*512, 4096, 4096, True)
    test_correctness(4096, 4096, 4096, True)

    main()
    
