# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
AMD GPU instruction scheduling intrinsics for Gluon kernels.

These functions emit LLVM AMDGCN scheduling intrinsics through the
TritonAMDGPU dialect -> ROCDL lowering pipeline.

Usage -- pass a live tensor through the barrier to anchor it in the
data-flow graph and prevent the compiler from eliminating it::

    x = gl.load(ptr + offsets)
    x = gl.amd.sched_barrier(x, 0)     # full scheduling fence
    y = x * 2.0
    y = gl.amd.set_prio(y, 1)          # raise wave priority
"""
from __future__ import annotations

from .._core import builtin, _unwrap_if_constexpr

__all__ = ["sched_barrier", "sched_group_barrier", "iglp_opt", "set_prio"]


@builtin
def sched_barrier(tensor, mask=0, _semantic=None):
    """
    Insert an instruction scheduling barrier, passing *tensor* through.

    Lowers to ``llvm.amdgcn.sched.barrier`` via ROCDL.

    Args:
        tensor: A live tensor to pass through (anchors the barrier in the
            data-flow graph).
        mask (int): Bitmask controlling which instruction types are blocked.
            0x0000 = full barrier (nothing may cross).

    Returns:
        The input tensor, unchanged.
    """
    mask = int(_unwrap_if_constexpr(mask))
    _semantic.builder.create_sched_barrier(mask)
    return tensor


@builtin
def sched_group_barrier(tensor, mask=0, count=1, sync_id=0, _semantic=None):
    """
    Insert an instruction-group scheduling barrier, passing *tensor* through.

    Lowers to ``llvm.amdgcn.sched.group.barrier`` via ROCDL.

    Args:
        tensor: A live tensor to pass through.
        mask (int): Bitmask selecting instruction types in this group.
        count (int): Number of instructions in this group.
        sync_id (int): Synchronization group ID (0-based).

    Returns:
        The input tensor, unchanged.
    """
    mask = int(_unwrap_if_constexpr(mask))
    count = int(_unwrap_if_constexpr(count))
    sync_id = int(_unwrap_if_constexpr(sync_id))
    _semantic.builder.create_sched_group_barrier(mask, count, sync_id)
    return tensor


@builtin
def iglp_opt(tensor, value=0, _semantic=None):
    """
    Insert an Instruction Group Level Parallelism optimization hint,
    passing *tensor* through.

    Lowers to ``llvm.amdgcn.iglp.opt`` via ROCDL.

    Args:
        tensor: A live tensor to pass through.
        value (int): IGLP strategy (0 = disabled, 1 = round-robin,
            2 = attention-optimized).

    Returns:
        The input tensor, unchanged.
    """
    value = int(_unwrap_if_constexpr(value))
    _semantic.builder.create_iglp_opt(value)
    return tensor


@builtin
def set_prio(tensor, value, _semantic=None):
    """
    Set wave scheduling priority, passing *tensor* through.

    Lowers to ``s_setprio`` via ROCDL.

    Args:
        tensor: A live tensor to pass through.
        value (int): Priority level (0 = low/normal, 1 = high).

    Returns:
        The input tensor, unchanged.
    """
    value = int(_unwrap_if_constexpr(value))
    _semantic.builder.create_set_prio(value)
    return tensor
