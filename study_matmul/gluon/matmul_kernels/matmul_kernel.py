import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.amd.cdna3 import sched_barrier


@gluon.jit
def get_pids(M, N, BM: gl.constexpr, BN: gl.constexpr, GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr,
             GROUP_SIZE_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        # When GRID_MN cannot divide NUM_XCDS, some xcds will have
        # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
        # We calculate the number of xcds that have pids_per_xcd pids as
        # tall_xcds
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        # Note that we need to consider the following two cases:
        # 1. the currnt pid is on a tall xcd
        # 2. the current pid is on a short xcd
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def get_ab_base_offsets(a_ptr, b_ptr, stride_am, stride_ak,  #
                        stride_bk, stride_bn,  #
                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, pid_m, pid_n,
                        num_warps: gl.constexpr, isLLGR: gl.constexpr = False):
    if isLLGR:
        if num_warps == 8:
            gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [128, 0]], lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0],
                                                                                  [32, 0], [64, 0]],
                warp_bases=[[1, 0], [2, 0], [4, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_K])
            gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 128]], lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16],
                                                                                  [0, 32], [0, 64]],
                warp_bases=[[0, 1], [0, 2], [0, 4]], block_bases=[], shape=[BLOCK_K, BLOCK_N])
        elif num_warps == 4:
            gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]], lane_bases=[[0, 8], [0, 16], [0, 32],
                                                                                          [16, 0], [32, 0], [64, 0]],
                warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_K])
            gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 128]], lane_bases=[[8, 0], [16, 0], [32, 0],
                                                                                          [0, 16], [0, 32], [0, 64]],
                warp_bases=[[0, 1], [0, 2]], block_bases=[], shape=[BLOCK_K, BLOCK_N])
    else:
        gLoadLayoutA: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BLOCK_K, BLOCK_K // 8], [8, 1], [1, 0])
        gLoadLayoutB: gl.constexpr = gl.BlockedLayout([8, 1], [BLOCK_K // 8, 512 // BLOCK_K], [1, 8], [0, 1])

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    return a_base, b_base, a_offsets, b_offsets


@gluon.jit
def store_c(c_ptr, pid_m, pid_n, BLOCK_M, BLOCK_N, c, stride_cm, stride_cn, mfmaLayout):
    if mfmaLayout is None:
        gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BLOCK_N, BLOCK_N // 8], [8, 1], [1, 0])
    else:
        gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    gl.amd.cdna3.buffer_store(stored_value=c, ptr=c_base, offsets=c_offsets)


@gluon.jit
def v0(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)

    a_base, b_base, a_offsets, b_offsets = get_ab_base_offsets(a_ptr, b_ptr, stride_am, stride_ak, stride_bk, stride_bn,
                                                               BLOCK_M, BLOCK_N, BLOCK_K, pid_m, pid_n)

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    warps_per_cta=[2, 4])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    for k in range(0, gl.cdiv(K, BLOCK_K)):
        ga = gl.amd.cdna4.buffer_load(a_base, a_offsets)
        gb = gl.amd.cdna4.buffer_load(b_base, b_offsets)
        a = gl.convert_layout(ga, layout=dotOpLayoutA)
        b = gl.convert_layout(gb, layout=dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    # C: BLOCK_M x BLOCK_N (256x256)
    store_c(c_ptr, pid_m, pid_n, BLOCK_M, BLOCK_N, c, stride_cm, stride_cn)


## Replace buffer_load with AsyncCopy
@gluon.jit
def v1(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)

    a_base, b_base, a_offsets, b_offsets = get_ab_base_offsets(a_ptr, b_ptr, stride_am, stride_ak, stride_bk, stride_bn,
                                                               BLOCK_M, BLOCK_N, BLOCK_K, pid_m, pid_n)

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    warps_per_cta=[2, 4])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    #sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    #sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)
    for k in range(0, iterMax):
        ga = cdna4_async_copy.buffer_load_to_shared(smemA, a_base, a_offsets)
        gb = cdna4_async_copy.buffer_load_to_shared(smemB, b_base, b_offsets)
        cdna4_async_copy.async_wait(0)

        a = cdna4_async_copy.load_shared_relaxed(smemA, dotOpLayoutA)
        b = cdna4_async_copy.load_shared_relaxed(smemB, dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    # C: BLOCK_M x BLOCK_N (256x256)
    store_c(c_ptr, pid_m, pid_n, BLOCK_M, BLOCK_N, c, stride_cm, stride_cn)


## Replace swizzledSharedLayout with paddedSharedLayout
@gluon.jit
def v2(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)

    a_base, b_base, a_offsets, b_offsets = get_ab_base_offsets(a_ptr, b_ptr, stride_am, stride_ak, stride_bk, stride_bn,
                                                               BLOCK_M, BLOCK_N, BLOCK_K, pid_m, pid_n, True)

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    warps_per_cta=[2, 4])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],
                                                         [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]],
                                                        [], [BLOCK_M, BLOCK_K])
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16],
                                                         [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 128]],
                                                        [], [BLOCK_K, BLOCK_N])

    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)
    for k in range(0, iterMax):
        ga = cdna4_async_copy.buffer_load_to_shared(smemA, a_base, a_offsets)
        gb = cdna4_async_copy.buffer_load_to_shared(smemB, b_base, b_offsets)
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(0)

        a = cdna4_async_copy.load_shared_relaxed(smemA, dotOpLayoutA)
        b = cdna4_async_copy.load_shared_relaxed(smemB, dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    # C: BLOCK_M x BLOCK_N (256x256)
    store_c(c_ptr, pid_m, pid_n, BLOCK_M, BLOCK_N, c, stride_cm, stride_cn)


# 4 waves
@gluon.jit
def v3(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)
    num_warps: gl.constexpr = 4

    a_base, b_base, a_offsets, b_offsets = get_ab_base_offsets(a_ptr, b_ptr, stride_am, stride_ak, stride_bk, stride_bn,
                                                               BLOCK_M, BLOCK_N, BLOCK_K, pid_m, pid_n, num_warps, True)

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    warps_per_cta=[2, 2])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],
                                                         [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]],
                                                        [], [BLOCK_M, BLOCK_K])
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16],
                                                         [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 128]],
                                                        [], [BLOCK_K, BLOCK_N])

    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)
    for k in range(0, iterMax):
        ga = cdna4_async_copy.buffer_load_to_shared(smemA, a_base, a_offsets)
        gb = cdna4_async_copy.buffer_load_to_shared(smemB, b_base, b_offsets)
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(0)

        a = cdna4_async_copy.load_shared_relaxed(smemA, dotOpLayoutA)
        b = cdna4_async_copy.load_shared_relaxed(smemB, dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    # C: BLOCK_M x BLOCK_N (256x256)
    store_c(c_ptr, pid_m, pid_n, BLOCK_M, BLOCK_N, c, stride_cm, stride_cn, mfmaLayout)


@gluon.jit
def v4(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    '''
    v4
    2 stage pipeline
    stage0: AC A,B
    stage1: LR A,B + DOT(A,B)
    '''

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)
    num_warps: gl.constexpr = 4

    a_base, b_base, a_offsets, b_offsets = get_ab_base_offsets(a_ptr, b_ptr, stride_am, stride_ak, stride_bk, stride_bn,
                                                               BLOCK_M, BLOCK_N, BLOCK_K, pid_m, pid_n, num_warps, True)

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    warps_per_cta=[2, 2])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],
                                                         [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]],
                                                        [], [BLOCK_M, BLOCK_K])
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16],
                                                         [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 128]],
                                                        [], [BLOCK_K, BLOCK_N])

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N], sharedLayoutB)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ##
    ## AC A0, B0 --> buffer 0
    ##
    ## InLoop
    ##
    ## AC A1, B1 --> buffer 1
    ## async_wait buffer 0
    ## local_load A0, B0 <-- buffer 0
    ## DOT(A0, B0)
    ##
    ## Epilogue
    ##
    ## async_wait buffer 0
    ## local_load A0, B0 <-- buffer 0
    ## DOT(A0, B0)
    ## store(acc)

    ## Prologue
    g_idx = 0
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    cdna4_async_copy.commit_group()
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    for k in range(0, iterMax - 1):
        l_idx = k % 2
        g_idx = 1 - l_idx
        cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
        cdna4_async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
        cdna4_async_copy.commit_group()

        cdna4_async_copy.wait_group(1)

        a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b = cdna4_async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    ## Epilogue
    cdna4_async_copy.wait_group(0)
    l_idx = (iterMax - 1) % 2
    a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b = cdna4_async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

    acc = gl.amd.cdna3.mfma(a, b, acc)

    c = acc.to(tl.float16)

    # C: BLOCK_M x BLOCK_N (256x256)
    store_c(c_ptr, pid_m, pid_n, BLOCK_M, BLOCK_N, c, stride_cm, stride_cn, mfmaLayout)


@gluon.jit
def v5(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    '''
    v5
    3 stage pipeline
    stage0: AC A,B
    stage1: LR A,B
    stage2: DOT(A,B)
    '''

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)
    num_warps: gl.constexpr = 4

    a_base, b_base, a_offsets, b_offsets = get_ab_base_offsets(a_ptr, b_ptr, stride_am, stride_ak, stride_bk, stride_bn,
                                                               BLOCK_M, BLOCK_N, BLOCK_K, pid_m, pid_n, num_warps, True)

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    warps_per_cta=[2, 2])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],
                                                         [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]],
                                                        [], [BLOCK_M, BLOCK_K])
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16],
                                                         [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 128]],
                                                        [], [BLOCK_K, BLOCK_N])

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N], sharedLayoutB)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ##
    ## AC A0, B0 --> buffer 0
    ## AC A1, B1 --> buffer 1
    ## async_wait buffer 0
    ## local_load A0, B0 <-- buffer 0
    ##
    ## InLoop
    ##
    ## DOT(A0, B0)
    ## async_wait buffer 1
    ## local_load A1, B1 <-- buffer 1
    ## AC A2, B2 --> buffer 0
    ##
    ## Epilogue
    ##
    ## DOT(A{n-1}, B{n-1})
    ## store(acc)

    ## Prologue
    g_idx = 0
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    cdna4_async_copy.commit_group()
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    g_idx = 1
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    cdna4_async_copy.commit_group()
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    cdna4_async_copy.wait_group(1)
    l_idx = 0
    a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b = cdna4_async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

    for k in range(0, iterMax - 1):
        g_idx = k % 2
        l_idx = 1 - g_idx
        acc = gl.amd.cdna3.mfma(a, b, acc)

        cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        cdna4_async_copy.wait_group(1)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b = cdna4_async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    ## Epilogue
    ## iterMax - 2
    #l_idx = (iterMax - 1) % 2
    #acc = gl.amd.cdna3.mfma(a, b, acc)

    #cdna4_async_copy.wait_group(0)
    #a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    #b = cdna4_async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

    ## iterMax - 1
    acc = gl.amd.cdna3.mfma(a, b, acc)

    c = acc.to(tl.float16)

    # C: BLOCK_M x BLOCK_N (256x256)
    store_c(c_ptr, pid_m, pid_n, BLOCK_M, BLOCK_N, c, stride_cm, stride_cn, mfmaLayout)


@gluon.jit
def v6(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    '''
    v6
    3 stage pipeline with partial LDS prefetch
    stage0: AC A,B0, B1
    stage1: LR A,B0
    stage2: LR B1, DOT(A,B0), DOT(A, B1)

    This leads to acc0 and acc1 in the epilogue.
    We also set tiles_per_warp=[2,2] and a better layout for buffer_store.
    '''

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)
    num_warps: gl.constexpr = 4

    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]], lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0],
                                                                                  [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_K])
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]], lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32],
                                                                        [0, 64]], warp_bases=[[0, 1], [0, 2]],
        block_bases=[], shape=[BLOCK_K, BLOCK_N // 2])

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b0_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b1_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn + BLOCK_N * stride_bn // 2

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    tiles_per_warp=[2, 2], warps_per_cta=[2, 2])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],
                                                         [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]],
                                                        [], [BLOCK_M, BLOCK_K])
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8]],
        [], [BLOCK_K, BLOCK_N // 2])

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB0 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)
    smemB1 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)

    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ##
    ## AC A[0], B0[0], B1[0] --> buffer 0
    ## AC A[1], B0[1], B1[1] --> buffer 1
    ## async_wait buffer 0
    ## local_load (A+B0)[0] <-- buffer 0
    ##
    ## InLoop
    ##
    ## DOT(A, B0)[0]
    ## local_load B1[0] <-- buffer 0
    ## AC (A+B0)[2] --> buffer 0
    ##
    ## DOT(A, B1)[0]
    ## async_wait buffer 1
    ## local_load (A+B0)[1] <-- buffer 1
    ## AC B1[2] --> buffer 0
    ##
    ## Epilogue
    ##
    ## local_load B1[n-1]
    ## DOT(A{n-1}, B0{n-1})
    ## store(acc0)
    ## DOT(A{n-1}, B1{n-1})
    ## store(acc1)

    ## Prologue
    g_idx = 0
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_offsets)
    cdna4_async_copy.commit_group()

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    g_idx = 1
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_offsets)
    cdna4_async_copy.commit_group()

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    cdna4_async_copy.wait_group(3)
    l_idx = 0
    a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(l_idx), dotOpLayoutB)

    for k in range(0, iterMax - 1):
        g_idx = k % 2
        l_idx = 1 - g_idx

        acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

        cdna4_async_copy.wait_group(2)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(g_idx), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        acc1 = gl.amd.cdna3.mfma(a, b1, acc1)

        cdna4_async_copy.wait_group(2)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(l_idx), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    gStoreLayoutC: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [16, 0], [0, 64], [64, 0], [128, 0]], lane_bases=[[1, 0], [2, 0], [4, 0],
                                                                                             [8, 0], [0, 16], [0, 8]],
        warp_bases=[[0, 32], [32, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_N // 2])

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c0_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c1_base = c0_base + BLOCK_N * stride_cn // 2
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    l_idx = (iterMax - 1) % 2

    acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

    cdna4_async_copy.wait_group(0)
    b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(l_idx), dotOpLayoutB)
    c0 = acc0.to(tl.float16)
    c0 = gl.convert_layout(c0, layout=gStoreLayoutC)

    acc1 = gl.amd.cdna3.mfma(a, b1, acc1)
    gl.amd.cdna3.buffer_store(stored_value=c0, ptr=c0_base, offsets=c_offsets)
    c1 = acc1.to(tl.float16)
    c1 = gl.convert_layout(c1, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c1, ptr=c1_base, offsets=c_offsets)


@gluon.jit
def v7(a_ptr, b_ptr, c_ptr, M, N, K: gl.constexpr, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    '''
    v7
    Unroll the loop
    '''

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)
    num_warps: gl.constexpr = 4

    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]], lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0],
                                                                                  [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_K])
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]], lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32],
                                                                        [0, 64]], warp_bases=[[0, 1], [0, 2]],
        block_bases=[], shape=[BLOCK_K, BLOCK_N // 2])

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b0_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b1_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn + BLOCK_N * stride_bn // 2

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    tiles_per_warp=[2, 2], warps_per_cta=[2, 2])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],
                                                         [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]],
                                                        [], [BLOCK_M, BLOCK_K])
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8]],
        [], [BLOCK_K, BLOCK_N // 2])

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB0 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)
    smemB1 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)

    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax: gl.constexpr = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ##
    ## AC A[0], B0[0], B1[0] --> buffer 0
    ## AC A[1], B0[1], B1[1] --> buffer 1
    ## async_wait buffer 0
    ## local_load (A+B0)[0] <-- buffer 0
    ##
    ## InLoop
    ##
    ## DOT(A, B0)[0]
    ## local_load B1[0] <-- buffer 0
    ## AC (A+B0)[2] --> buffer 0
    ##
    ## DOT(A, B1)[0]
    ## async_wait buffer 1
    ## local_load (A+B0)[1] <-- buffer 1
    ## AC B1[2] --> buffer 0
    ##
    ## Epilogue
    ##
    ## local_load B1[n-1]
    ## DOT(A{n-1}, B0{n-1})
    ## store(acc0)
    ## DOT(A{n-1}, B1{n-1})
    ## store(acc1)

    ## Prologue
    g_idx = 0
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_offsets)
    cdna4_async_copy.commit_group()

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    g_idx = 1
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_offsets)
    cdna4_async_copy.commit_group()

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    cdna4_async_copy.wait_group(3)
    l_idx = 0
    a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(l_idx), dotOpLayoutB)

    for k in range(0, iterMax - 1, 2):
        g_idx = k % 2  ## 0
        l_idx = 1 - g_idx  ## 1

        sched_barrier(0)

        ## DOT(A, B0)[0]
        ## LR B1[0]
        ## AC (A+B0)[2]
        acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

        cdna4_async_copy.wait_group(2)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(0), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemA.index(0), a_base, a_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.buffer_load_to_shared(smemB0.index(0), b_base, b0_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        ## DOT(A, B1)[0]
        ## LR (A+B0)[1]
        ## AC B1[2]
        acc1 = gl.amd.cdna3.mfma(a, b1, acc1)

        cdna4_async_copy.wait_group(2)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(1), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemB1.index(0), b_base, b1_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        sched_barrier(0)

        ## ---------------------------------------------------------------------
        ## Loop unroll separator
        ## ---------------------------------------------------------------------
        ## DOT(A, B0)[1]
        ## LR B1[1]
        ## AC (A+B0)[3]
        acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

        cdna4_async_copy.wait_group(2)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(1), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemA.index(1), a_base, a_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.buffer_load_to_shared(smemB0.index(1), b_base, b0_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        ## DOT(A, B1)[1]
        ## LR (A+B0)[1]
        ## AC B1[3]
        acc1 = gl.amd.cdna3.mfma(a, b1, acc1)

        cdna4_async_copy.wait_group(2)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(0), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemB1.index(1), b_base, b1_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        sched_barrier(0)

    gStoreLayoutC: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [16, 0], [0, 64], [64, 0], [128, 0]], lane_bases=[[1, 0], [2, 0], [4, 0],
                                                                                             [8, 0], [0, 16], [0, 8]],
        warp_bases=[[0, 32], [32, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_N // 2])

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c0_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c1_base = c0_base + BLOCK_N * stride_cn // 2
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    l_idx = (iterMax - 1) % 2

    sched_barrier(0)

    acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

    cdna4_async_copy.wait_group(0)
    b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(l_idx), dotOpLayoutB)
    c0 = acc0.to(tl.float16)
    c0 = gl.convert_layout(c0, layout=gStoreLayoutC)

    sched_barrier(0)

    acc1 = gl.amd.cdna3.mfma(a, b1, acc1)
    gl.amd.cdna3.buffer_store(stored_value=c0, ptr=c0_base, offsets=c_offsets)
    c1 = acc1.to(tl.float16)
    c1 = gl.convert_layout(c1, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c1, ptr=c1_base, offsets=c_offsets)


@gluon.jit
def v8(a_ptr, b_ptr, c_ptr, M, N, K: gl.constexpr, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):
    '''
    v7
    Unroll the loop
    '''

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N)
    num_warps: gl.constexpr = 4

    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]], lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0],
                                                                                  [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_K])
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]], lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32],
                                                                        [0, 64]], warp_bases=[[0, 1], [0, 2]],
        block_bases=[], shape=[BLOCK_K, BLOCK_N // 2])

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b0_init_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    b0_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn + BLOCK_K * stride_bk
    b1_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn + BLOCK_N * stride_bn // 2

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    tiles_per_warp=[2, 2], warps_per_cta=[2, 2])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],
                                                         [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]],
                                                        [], [BLOCK_M, BLOCK_K])
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8]],
        [], [BLOCK_K, BLOCK_N // 2])

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB0 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)
    smemB1 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)

    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax: gl.constexpr = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ##
    ## AC A[0], B0[0], B1[0] --> buffer 0
    ## AC A[1], B0[1], B1[1] --> buffer 1
    ## async_wait buffer 0
    ## local_load (A+B0)[0] <-- buffer 0
    ##
    ## InLoop
    ##
    ## DOT(A, B0)[0]
    ## local_load B1[0] <-- buffer 0
    ## AC (A+B0)[2] --> buffer 0
    ##
    ## DOT(A, B1)[0]
    ## async_wait buffer 1
    ## local_load (A+B0)[1] <-- buffer 1
    ## AC B1[2] --> buffer 0
    ##
    ## Epilogue
    ##
    ## local_load B1[n-1]
    ## DOT(A{n-1}, B0{n-1})
    ## store(acc0)
    ## DOT(A{n-1}, B1{n-1})
    ## store(acc1)

    ## Prologue
    g_idx = 0
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_init_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    g_idx = 1
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    cdna4_async_copy.wait_group(3)
    l_idx = 0
    a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(l_idx), dotOpLayoutB)

    for k in range(0, iterMax - 1, 2):

        sched_barrier(0)

        ## DOT(A, B0)[0]
        ## LR B1[0]
        ## AC (A+B0)[2]
        acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

        cdna4_async_copy.wait_group(2)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(0), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemA.index(0), a_base, a_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.buffer_load_to_shared(smemB0.index(0), b_base, b0_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        ## DOT(A, B1)[0]
        ## LR (A+B0)[1]
        ## AC B1[2]
        acc1 = gl.amd.cdna3.mfma(a, b1, acc1)

        cdna4_async_copy.wait_group(2)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(1), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemB1.index(0), b_base, b1_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        ## ---------------------------------------------------------------------
        ## Loop unroll separator
        ## ---------------------------------------------------------------------
        ## DOT(A, B0)[1]
        ## LR B1[1]
        ## AC (A+B0)[3]
        acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

        cdna4_async_copy.wait_group(2)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(1), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemA.index(1), a_base, a_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.buffer_load_to_shared(smemB0.index(1), b_base, b0_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        ## DOT(A, B1)[1]
        ## LR (A+B0)[1]
        ## AC B1[3]
        acc1 = gl.amd.cdna3.mfma(a, b1, acc1)

        cdna4_async_copy.wait_group(2)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(0), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemB1.index(1), b_base, b1_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

    gStoreLayoutC: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [16, 0], [0, 64], [64, 0], [128, 0]], lane_bases=[[1, 0], [2, 0], [4, 0],
                                                                                             [8, 0], [0, 16], [0, 8]],
        warp_bases=[[0, 32], [32, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_N // 2])

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c0_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c1_base = c0_base + BLOCK_N * stride_cn // 2
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    l_idx = (iterMax - 1) % 2

    sched_barrier(0)

    acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

    cdna4_async_copy.wait_group(0)
    b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(l_idx), dotOpLayoutB)
    c0 = acc0.to(tl.float16)
    c0 = gl.convert_layout(c0, layout=gStoreLayoutC)

    sched_barrier(0)

    acc1 = gl.amd.cdna3.mfma(a, b1, acc1)
    gl.amd.cdna3.buffer_store(stored_value=c0, ptr=c0_base, offsets=c_offsets)
    c1 = acc1.to(tl.float16)
    c1 = gl.convert_layout(c1, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c1, ptr=c1_base, offsets=c_offsets)


@gluon.jit
def v9(a_ptr, b_ptr, c_ptr, M, N, K: gl.constexpr, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr  #
       ):
    '''
    v7
    Unroll the loop
    '''

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)
    num_warps: gl.constexpr = 4

    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]], lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0],
                                                                                  [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_K])
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]], lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32],
                                                                        [0, 64]], warp_bases=[[0, 1], [0, 2]],
        block_bases=[], shape=[BLOCK_K, BLOCK_N // 2])

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b0_init_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    b0_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn + BLOCK_K * stride_bk
    b1_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn + BLOCK_N * stride_bn // 2

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                    tiles_per_warp=[2, 2], warps_per_cta=[2, 2])
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout([[512, 16]],
                                                        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0],
                                                         [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]],
                                                        [], [BLOCK_M, BLOCK_K])
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8]],
        [], [BLOCK_K, BLOCK_N // 2])

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB0 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)
    smemB1 = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)

    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax: gl.constexpr = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ##
    ## AC A[0], B0[0], B1[0] --> buffer 0
    ## AC A[1], B0[1], B1[1] --> buffer 1
    ## async_wait buffer 0
    ## local_load (A+B0)[0] <-- buffer 0
    ##
    ## InLoop
    ##
    ## DOT(A, B0)[0]
    ## local_load B1[0] <-- buffer 0
    ## AC (A+B0)[2] --> buffer 0
    ##
    ## DOT(A, B1)[0]
    ## async_wait buffer 1
    ## local_load (A+B0)[1] <-- buffer 1
    ## AC B1[2] --> buffer 0
    ##
    ## Epilogue
    ##
    ## local_load B1[n-1]
    ## DOT(A{n-1}, B0{n-1})
    ## store(acc0)
    ## DOT(A{n-1}, B1{n-1})
    ## store(acc1)

    ## Prologue
    g_idx = 0
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_init_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    g_idx = 1
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    cdna4_async_copy.wait_group(3)
    l_idx = 0
    a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(l_idx), dotOpLayoutB)

    for k in range(0, iterMax - 1, 2):

        sched_barrier(0)

        ## DOT(A, B0)[0]
        ## LR B1[0]
        ## AC (A+B0)[2]
        acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

        cdna4_async_copy.wait_group(2)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(0), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemA.index(0), a_base, a_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.buffer_load_to_shared(smemB0.index(0), b_base, b0_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        ## DOT(A, B1)[0]
        ## LR (A+B0)[1]
        ## AC B1[2]
        acc1 = gl.amd.cdna3.mfma(a, b1, acc1)

        cdna4_async_copy.wait_group(2)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(1), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemB1.index(0), b_base, b1_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        ## ---------------------------------------------------------------------
        ## Loop unroll separator
        ## ---------------------------------------------------------------------
        ## DOT(A, B0)[1]
        ## LR B1[1]
        ## AC (A+B0)[3]
        acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

        cdna4_async_copy.wait_group(2)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(1), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemA.index(1), a_base, a_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.buffer_load_to_shared(smemB0.index(1), b_base, b0_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        ## DOT(A, B1)[1]
        ## LR (A+B0)[1]
        ## AC B1[3]
        acc1 = gl.amd.cdna3.mfma(a, b1, acc1)

        cdna4_async_copy.wait_group(2)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(0), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(smemB1.index(1), b_base, b1_offsets, mask=(k != (iterMax - 2)))
        cdna4_async_copy.commit_group()

        sched_barrier(0)

    gStoreLayoutC: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [16, 0], [0, 64], [64, 0], [128, 0]], lane_bases=[[1, 0], [2, 0], [4, 0],
                                                                                             [8, 0], [0, 16], [0, 8]],
        warp_bases=[[0, 32], [32, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_N // 2])

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c0_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c1_base = c0_base + BLOCK_N * stride_cn // 2
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    l_idx = (iterMax - 1) % 2

    sched_barrier(0)

    acc0 = gl.amd.cdna3.mfma(a, b0, acc0)

    cdna4_async_copy.wait_group(0)
    b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(l_idx), dotOpLayoutB)
    c0 = acc0.to(tl.float16)
    c0 = gl.convert_layout(c0, layout=gStoreLayoutC)

    sched_barrier(0)

    acc1 = gl.amd.cdna3.mfma(a, b1, acc1)
    gl.amd.cdna3.buffer_store(stored_value=c0, ptr=c0_base, offsets=c_offsets)
    c1 = acc1.to(tl.float16)
    c1 = gl.convert_layout(c1, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c1, ptr=c1_base, offsets=c_offsets)
