import triton
import triton.language as tl

from flash import (
    bwd_preprocess as bare_bwd_preprocess,
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq,
)

TRITON_CONFIG_LIST_BWD = []
# for BLOCK_M in [16, 32, 64]:
#     for BLOCK_N in [16, 32, 64]:
for BLOCK_M, BLOCK_N in [(32, 64), (64, 16)]:
    dic = {}
    dic['BLOCK_M'] = BLOCK_M
    dic['BLOCK_N'] = BLOCK_N
    # for waves_per_eu in range(0, 4+1):
    for waves_per_eu in [0, 3]:
        dic['waves_per_eu'] = waves_per_eu
        # for num_stages in [0, 1]:
        for num_stages in [1]:
            # for num_warps in [1,2,4,8]:
            for num_warps in [1,2]:
                cfg = triton.Config(dict(dic), num_stages=num_stages, num_warps=num_warps)
                TRITON_CONFIG_LIST_BWD.append(cfg)

print(TRITON_CONFIG_LIST_BWD)

@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD,
   key=['BLOCK_DMODEL', 'max_seqlen_q', 'max_seqlen_k'],
)
@triton.jit
def tuned_bwd_kernel_dk_dv(
    Q, K, V, B, sm_scale, Out, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,
    max_seqlen_q,
    max_seqlen_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    bare_bwd_kernel_dk_dv(
            Q, K, V, B, sm_scale, Out, DO,
            DK, DV,
            L,
            D,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            stride_bz, stride_bh, stride_bm, stride_bn,
            stride_oz, stride_oh, stride_om, stride_ok,
            stride_dkz, stride_dkh, stride_dkn, stride_dkk,
            stride_dvz, stride_dvh, stride_dvk, stride_dvn,
            cu_seqlens_q,
            cu_seqlens_k,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            dropout_p,
            philox_seed,
            philox_offset_base,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            CAUSAL,
            ENABLE_DROPOUT,
            PADDED_HEAD=PADDED_HEAD,
            BIAS_TYPE=BIAS_TYPE,
            )

@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD,
   key=['BLOCK_DMODEL', 'max_seqlen_q', 'max_seqlen_k'],
)
@triton.jit
def tuned_bwd_kernel_dq(
    Q, K, V, B, sm_scale, Out, DO,
    DQ, DB,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,
    max_seqlen_q,
    max_seqlen_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    bare_bwd_kernel_dq(Q, K, V, B, sm_scale, Out, DO,
        DQ, DB,
        L,
        D,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_bz, stride_bh, stride_bm, stride_bn,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_dqz, stride_dqh, stride_dqm, stride_dqk,
        stride_dbz, stride_dbh, stride_dbm, stride_dbn,
        cu_seqlens_q,
        cu_seqlens_k,
        num_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        head_dim,
        dropout_p,
        philox_seed,
        philox_offset_base,
        BLOCK_M, BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD=PADDED_HEAD,
        BIAS_TYPE=BIAS_TYPE,
        )
