# ruff: noqa: E402
"""
Common utilities for GFX1250 GEMM kernels.

This module contains shared functions, classes, and utilities used by both
persistent and StreamK GEMM implementations.
"""

import re
from triton.experimental import gluon
from triton.language.core import _aggregate as aggregate
import triton.experimental.gluon.language as ttgl


def static_profile(kernel):
    amdgcn = kernel.asm['amdgcn']

    sgpr_count = int(re.search(r'\.sgpr_count:\s+(\d+)', amdgcn).group(1))
    sgpr_spill_count = int(re.search(r'\.sgpr_spill_count:\s+(\d+)', amdgcn).group(1))
    vgpr_count = int(re.search(r'\.vgpr_count:\s+(\d+)', amdgcn).group(1))
    vgpr_spill_count = int(re.search(r'\.vgpr_spill_count:\s+(\d+)', amdgcn).group(1))
    scratch_size = int(re.search(r';\s+ScratchSize:\s+(\d+)', amdgcn).group(1))
    code_len_in_byte = int(re.search(r';\s+codeLenInByte\s+=\s+(\d+)', amdgcn).group(1))
    occupancy = int(re.search(r';\s+Occupancy:\s+(\d+)', amdgcn).group(1))

    print(f"- sgpr_count: {sgpr_count}\n"
          f"- sgpr_spill_count: {sgpr_spill_count}\n"
          f"- vgpr_count: {vgpr_count}\n"
          f"- vgpr_spill_count: {vgpr_spill_count}\n"
          f"- scratch_size: {scratch_size}\n"
          f"- code_len_in_byte: {code_len_in_byte}\n"
          f"- occupancy: {occupancy}\n")


@gluon.constexpr_function
def create_shared_layouts(BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                          TRANSPOSE_B: ttgl.constexpr):

    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0])
    if not TRANSPOSE_B:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N],
                                                                                    [1, 0])
    else:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K],
                                                                                    [1, 0])

    return (SHARED_LAYOUT_A, SHARED_LAYOUT_B)


@gluon.jit
def create_tensor_descriptors(a_ptr, b_ptr, off_am, off_bn, stride_am, stride_ak, stride_bn, stride_bk,
                              shared_layout_a: ttgl.constexpr, shared_layout_b: ttgl.constexpr, M: ttgl.constexpr,
                              N: ttgl.constexpr, K: ttgl.constexpr, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                              BLOCK_K: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr + off_am, shape=(M, K),
                                                         strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K),
                                                         layout=shared_layout_a)
    if not TRANSPOSE_B:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + off_bn, shape=(K, N),
                                                             strides=(stride_bk, stride_bn),
                                                             block_shape=(BLOCK_K, BLOCK_N), layout=shared_layout_b)
    else:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + off_bn, shape=(N, K),
                                                             strides=(stride_bn, stride_bk),
                                                             block_shape=(BLOCK_N, BLOCK_K), layout=shared_layout_b)

    return a_desc, b_desc


@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K: ttgl.constexpr,
                NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, pred=1):

    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K], a_buffer.index(producer % NUM_BUFFERS),
                                    pred=pred)
    if not TRANSPOSE_B:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=pred)
    else:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=pred)
    producer += 1
    return producer


@gluon.jit
def issue_wmma(consumer, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr, accumulator,
               wait_producers_cnt, NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):

    ttgl.amd.gfx1250.tdm.async_wait(wait_producers_cnt)

    a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=b_layout)
    else:
        b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=b_layout)

    accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
    consumer += 1
    return consumer, accumulator


@gluon.jit
def lds_subtile_load(consumer, start, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr,
                     NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, SUBTILE_LEN: ttgl.constexpr):

    index = consumer % NUM_BUFFERS
    a = a_buffer.index(index).slice(start, SUBTILE_LEN, 1).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(index).slice(start, SUBTILE_LEN, 0).load(layout=b_layout)
    else:
        b = b_buffer.index(index).slice(start, SUBTILE_LEN, 1).permute([1, 0]).load(layout=b_layout)

    return a, b


@aggregate
class TileScheduler:
    """
    Unified Tile Scheduler - handles both Persistent and StreamK modes.

    Persistent Mode (STREAMK_TILES=0):
    - All tiles are processed as full tiles
    - Each CU processes tiles from [pid_start, pid_end)

    StreamK Mode (STREAMK_TILES>0):
    - Full tiles: processed normally, one complete tile per iteration
    - StreamK tiles: last STREAMK_TILES use K-dimension splitting for load balancing
    """
    pid_start: ttgl.tensor
    pid_end: ttgl.tensor
    num_pid_m: ttgl.tensor
    num_pid_n: ttgl.tensor
    total_tiles: ttgl.tensor
    total_full_tiles: ttgl.tensor
    iters_per_tile: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, pid_start, pid_end, num_pid_m, num_pid_n, total_tiles, total_full_tiles, iters_per_tile):
        self.pid_start = pid_start
        self.pid_end = pid_end
        self.num_pid_m = num_pid_m
        self.num_pid_n = num_pid_n
        self.total_tiles = total_tiles
        self.total_full_tiles = total_full_tiles
        self.iters_per_tile = iters_per_tile

    @gluon.jit
    def initialize(M, N, K, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                   STREAMK_TILES: ttgl.constexpr):
        """Initialize unified scheduler for both Persistent and StreamK modes."""
        kernel_id = ttgl.program_id(axis=0)
        num_kernels = ttgl.num_programs(axis=0)
        num_pid_m = ttgl.cdiv(M, BLOCK_M)
        num_pid_n = ttgl.cdiv(N, BLOCK_N)
        total_tiles = num_pid_m * num_pid_n

        # When STREAMK_TILES=0: persistent mode, all tiles are full tiles
        # When STREAMK_TILES>0: StreamK mode, last STREAMK_TILES are for StreamK
        total_full_tiles = total_tiles - STREAMK_TILES
        iters_per_tile = ttgl.cdiv(K, BLOCK_K)

        pid_per_kernel = ttgl.cdiv(total_full_tiles, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, total_full_tiles)

        return TileScheduler(pid_start, pid_end, num_pid_m, num_pid_n, total_tiles, total_full_tiles, iters_per_tile)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_num_full_tiles(self):
        return self.total_full_tiles

    @gluon.jit
    def get_num_streamk_tiles(self):
        return self.total_tiles - self.total_full_tiles

    @gluon.jit
    def get_iters_per_tile(self):
        return self.iters_per_tile

    @gluon.jit
    def get_linear_tile_coords(self, idx):
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n

    @gluon.jit
    def get_swizzled_tile_coords(self, tile_id, GROUP_SIZE_M: ttgl.constexpr):
        num_pid_in_group = GROUP_SIZE_M * self.num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    @gluon.jit
    def get_streamk_params(self):
        num_sms = ttgl.num_programs(axis=0)
        total_streamk_iters = self.get_num_streamk_tiles() * self.iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
        return total_streamk_iters, streamk_iters_pcu, streamk_remainder_iters

    @gluon.jit
    def get_streamk_iteration_range(self):
        pid = ttgl.program_id(axis=0)
        num_sms = ttgl.num_programs(axis=0)

        # Get StreamK distribution parameters
        total_streamk_iters = self.get_num_streamk_tiles() * self.iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms

        # Calculate this CU's iteration range
        # Offset by total_full_tiles * iters_per_tile to start after full tiles
        base_offset = self.total_full_tiles * self.iters_per_tile
        start_iter = base_offset + pid * streamk_iters_pcu + min(pid, streamk_remainder_iters)
        last_iter = base_offset + (pid + 1) * streamk_iters_pcu + min(pid + 1, streamk_remainder_iters)

        return start_iter, last_iter
