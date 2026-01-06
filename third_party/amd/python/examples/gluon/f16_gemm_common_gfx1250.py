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


@gluon.jit
def chiplet_transform(pid, num_workgroups, num_xcds: ttgl.constexpr):
    """
    Basic chiplet transformation for multi-XCD AMD GPUs.

    Transforms program ID to distribute work evenly across chiplets (XCDs).
    Each XCD gets a contiguous range of work items.
    """
    xcd = pid % num_xcds
    pos_in_xcd = pid // num_xcds
    min_per_xcd = num_workgroups // num_xcds
    extra_sms = num_workgroups % num_xcds
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd


@gluon.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: ttgl.constexpr, chunk_size: ttgl.constexpr):
    """
    Chunked chiplet transformation for improved memory locality.

    Groups work items into chunks of size `chunk_size` per XCD, ensuring
    adjacent work items within a chunk are on the same chiplet for better
    cache utilization and memory bandwidth.
    """
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        # Outside of the contiguous chunked region, leave unchanged
        return pid

    local_pid = pid // num_xcds
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size

    # Calculate new PID
    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@gluon.jit
def remap_xcd_chunked(pid, grid_mn, num_xcds: ttgl.constexpr = 8, chunk_size: ttgl.constexpr = 2):
    """
    XCD remapping with chunked distribution (alternative implementation).

    Similar to chiplet_transform_chunked but with default parameters
    optimized for AMD MI300 series (8 XCDs).
    """
    # Compute current XCD and local PID
    xcd = pid % num_xcds
    # Distribute the modulo pids in round robin
    if pid > (grid_mn // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid
    local_pid = pid // num_xcds
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    # Calculate new PID
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


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
                NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, pred=True):
    # Note: pred parameter is for conditional execution via if-statements, not passed to async_load
    if pred:
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K], a_buffer.index(producer % NUM_BUFFERS))
        if not TRANSPOSE_B:
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn],
                                            b_buffer.index(producer % NUM_BUFFERS))
        else:
            ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K],
                                            b_buffer.index(producer % NUM_BUFFERS))
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
    - Each CU processes tiles in a strided pattern: pid, pid+num_sms, pid+2*num_sms, ...
    - Optionally applies chiplet transformation for multi-XCD GPUs

    StreamK Mode (STREAMK_TILES>0):
    - Full tiles: processed normally using strided persistent pattern
    - StreamK tiles: last STREAMK_TILES use K-dimension splitting for load balancing

    Chiplet Optimization:
    - Use apply_chiplet_transform() to remap PIDs for better memory locality on MI300
    - Recommended for kernels with tile swizzling and smaller block sizes
    """
    num_pid_m: ttgl.tensor
    num_pid_n: ttgl.tensor
    total_tiles: ttgl.tensor
    total_full_tiles: ttgl.tensor
    iters_per_tile: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, num_pid_m, num_pid_n, total_tiles, total_full_tiles, iters_per_tile):
        self.num_pid_m = num_pid_m
        self.num_pid_n = num_pid_n
        self.total_tiles = total_tiles
        self.total_full_tiles = total_full_tiles
        self.iters_per_tile = iters_per_tile

    @gluon.jit
    def initialize(M, N, K, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                   STREAMK_TILES: ttgl.constexpr):
        """Initialize unified scheduler for both Persistent and StreamK modes."""
        num_pid_m = ttgl.cdiv(M, BLOCK_M)
        num_pid_n = ttgl.cdiv(N, BLOCK_N)
        total_tiles = num_pid_m * num_pid_n

        # When STREAMK_TILES=0: persistent mode, all tiles are full tiles
        # When STREAMK_TILES>0: StreamK mode, last STREAMK_TILES are for StreamK
        total_full_tiles = total_tiles - STREAMK_TILES
        iters_per_tile = ttgl.cdiv(K, BLOCK_K)

        return TileScheduler(num_pid_m, num_pid_n, total_tiles, total_full_tiles, iters_per_tile)

    @gluon.jit
    def get_num_tiles(self):
        """Return total number of full tiles for persistent loop."""
        return self.total_full_tiles

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
    def get_pid(self):
        """Return current program ID."""
        return ttgl.program_id(axis=0)

    @gluon.jit
    def get_num_sms(self):
        """Return total number of SMs/CUs available."""
        return ttgl.num_programs(axis=0)

    @gluon.jit
    def get_linear_tile_coords(self, tile_id):
        """Convert global tile ID to (pid_m, pid_n) coordinates using linear ordering."""
        pid_m = tile_id % self.num_pid_m
        pid_n = tile_id // self.num_pid_m
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
        num_sms = self.get_num_sms()
        total_streamk_iters = self.get_num_streamk_tiles() * self.iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
        return total_streamk_iters, streamk_iters_pcu, streamk_remainder_iters

    @gluon.jit
    def get_streamk_iteration_range(self):
        pid = self.get_pid()
        num_sms = self.get_num_sms()

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

    @gluon.jit
    def apply_chiplet_transform(self, pid, num_sms, num_xcds: ttgl.constexpr):
        """
        Apply basic chiplet transformation to a program ID.

        Redistributes work so each XCD gets a contiguous range, improving
        memory locality when tiles are processed in order.

        Args:
            pid: Current program ID
            num_sms: Total number of SMs/CUs
            num_xcds: Number of chiplets (XCDs), typically 8 for MI300

        Returns:
            Transformed PID optimized for multi-chiplet architecture
        """
        return chiplet_transform(pid, num_sms, num_xcds)

    @gluon.jit
    def apply_chiplet_transform_chunked(self, pid, num_sms, num_xcds: ttgl.constexpr, chunk_size: ttgl.constexpr):
        """
        Apply chunked chiplet transformation for improved cache locality.

        Creates small contiguous chunks on each XCD instead of large blocks.
        Recommended for GEMM kernels with tile swizzling.

        Args:
            pid: Current program ID
            num_sms: Total number of SMs/CUs
            num_xcds: Number of chiplets (XCDs), typically 8 for MI300
            chunk_size: Size of contiguous chunks per XCD (recommended: 2-4)

        Returns:
            Transformed PID optimized for memory locality

        Example:
            pid = ttgl.program_id(axis=0)
            num_sms = ttgl.num_programs(axis=0)
            transformed_pid = scheduler.apply_chiplet_transform_chunked(
                pid, num_sms, num_xcds=8, chunk_size=2
            )
            for tile_idx in range(transformed_pid, scheduler.get_num_tiles(), num_sms):
                # Process tiles with improved locality
        """
        return chiplet_transform_chunked(pid, num_sms, num_xcds, chunk_size)
