#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl

@triton.jit
def closed_interval_isect(a_s, a_e, b_s, b_e):
    if (b_s > a_e or a_s > b_e) or (a_s > a_e):
        return -114, -514
    o_s = max(a_s, b_s)
    o_e = min(a_e, b_e)
    return o_s, o_e

@triton.jit
def is_closed_interval_empty(a_s, a_e) -> bool:
    return a_s > a_e

@triton.jit
def closed_interval_size(a_s, a_e) -> bool:
    return max(0, a_e - a_s + 1)

@triton.jit
def div_rd(x, y):
    d = x // y
    return (d - 1) if d * y > x else d

'''
Parse IS_CAUSAL/CAUSAL_TYPE/Window_left/Window_right to actual window_left and window_right.
Window_left and Window_right support two special values: 0x80000001i32 and 0x80000002i32
In that case, the output window_left and window_right will be the value for CAUSAL_TYPE = 1 and 2 respectively.
These two special values are for varlen where there is no uniform value for top left/bottom right aligned window.

Must split parse_window away from calculate_intervals
calculate_intervals is designed for row major processing, and to re-use the
code, we transpose inputs for calculate_intervals when handling col major processing in dkdv kernel.

However, window_left and window_right parsing is invariant row/col major
processing, and need to split otherwise dkdv kernel's window_left/right will be incorrect.
'''
@triton.jit
def parse_window(IS_CAUSAL,
                 CAUSAL_TYPE,
                 Window_left,
                 Window_right,
                 seqlen_q,
                 seqlen_k):
    # 0x80000001i32 = -2147483647
    if CAUSAL_TYPE == 1 or Window_left == -2147483647:
        window_left = seqlen_q
    # 0x80000002i32 = -2147483646
    elif CAUSAL_TYPE == 2 or Window_left == -2147483646:
        window_left = seqlen_q
    else:
        window_left = Window_left
    if CAUSAL_TYPE == 1 or Window_right == -2147483647:
        window_right = 0
    elif CAUSAL_TYPE == 2 or Window_right == -2147483646:
        window_right = seqlen_k - seqlen_q
    else:
        window_right = Window_right
    return window_left, window_right


@triton.jit
def calculate_intervals(IS_CAUSAL,
                        CAUSAL_TYPE,
                        window_left,
                        window_right,
                        start_M,
                        seqlen_q,
                        seqlen_k,
                        mask_on_seq_q,
                        BLOCK_M,
                        BLOCK_N,
                        DEBUG=False):
    masked_seq_k_block = seqlen_k // BLOCK_N
    if DEBUG:
        tl.device_print('0 start_M', start_M)
        tl.device_print('0 BLOCK_M', BLOCK_M)
        tl.device_print('0 BLOCK_N', BLOCK_N)
        tl.device_print('0 seqlen_q', seqlen_q)
        tl.device_print('0 seqlen_k', seqlen_k)

    if IS_CAUSAL:
        if DEBUG:
            tl.device_print('0 window_left', window_left)
            tl.device_print('0 window_right', window_right)
            tl.device_print('0 mask_on_seq_q', mask_on_seq_q)

        ### Intervals for Left Masked Blocks (Infinite on N axis)
        # Intersection point (N axis) b/w window left line and M = start_M
        # start_M - window_left = N
        isec_lo = start_M - window_left
        # Intersection point (N axis) b/w window left line and M = min(start_M + BLOCK_M, seqlen_q)
        # M = + window_left + N = min(start_M + BLOCK_M, seqlen_q)
        isec_hi = min(start_M + BLOCK_M, seqlen_q) - window_left
        lsec_lob = div_rd(isec_lo, BLOCK_N)
        lsec_hib = div_rd(isec_hi - 1, BLOCK_N)
        if DEBUG:
            tl.device_print('1 lb_lo isect', isec_lo)
            tl.device_print('1 lb_hi isect', isec_hi)
            tl.device_print('1 Init lb_lo', lsec_lob)
            tl.device_print('1 Init lb_hi', lsec_hib)

        ### Intervals for Right Masked Blocks (Infinite on N axis)
        # Intersection point (N axis) b/w window left line and M = start_M
        # M = N - window_right = start_M
        isec_lo = start_M + window_right
        # Intersection point (N axis) b/w window left line and M = min(start_M + BLOCK_M, seqlen_q)
        # M = N - window_right = min(start_M + BLOCK_M, seqlen_q)
        isec_hi = min(start_M + BLOCK_M, seqlen_q) + window_right
        rsec_lob = div_rd(isec_lo, BLOCK_N)
        # FIXME: we need to solve this
        #  rsec_hib in Z
        #  rsec_hib * BLOCK_N < isec_hi
        #  maxmize rsec_hib
        #  Example: isec_hi = 0, then rsec_hib = -1
        rsec_hib = div_rd(isec_hi - 1, BLOCK_N)
        if DEBUG:
            tl.device_print('2 rb_lo isect', isec_lo)
            tl.device_print('2 rb_hi isect', isec_hi)
            tl.device_print('2 Init rb_lo', rsec_lob)
            tl.device_print('2 Init rb_hi', rsec_hib)

        ### Sanitize intervals: n < seqlen_k && n >= 0
        # Valid Closed Interval WITH the trailing partial mask
        # (intersection of n < seqlen_k and full blocks will handled later)
        vb_lo = 0
        vb_hi = (seqlen_k - 1) // BLOCK_N
        # lb_*: Left masked Blocks index (LOw/HIgh)
        # rb_*: Right masked Blocks index (LOw/HIgh)
        # fb_*: Full Blocks index (LOw/HIgh)
        lb_lo, lb_hi = closed_interval_isect(lsec_lob, lsec_hib, vb_lo, vb_hi)
        rb_lo, rb_hi = closed_interval_isect(rsec_lob, rsec_hib, vb_lo, vb_hi)
        if DEBUG:
            tl.device_print('3 Valid lb_lo', lb_lo)
            tl.device_print('3 Valid lb_hi', lb_hi)
            tl.device_print('3 Valid rb_lo', rb_lo)
            tl.device_print('3 Valid rb_hi', rb_hi)

        # Check if left and right masked blocks overlapped
        # ub_: United maksed Blocks
        ub_lo, ub_hi = closed_interval_isect(lsec_lob, lsec_hib, rsec_lob, rsec_hib)
        ub_empty = is_closed_interval_empty(ub_lo, ub_hi)

        ### Sanitize intervals: m < seqlen_q (m >= 0 is ensured by start_m)
        # All blocks need masking, unify them
        if ub_empty and not mask_on_seq_q:
            '''
            fb_: Full blocks
            Only exists when there is no intersection b/w LB and RB,
            AND
            Seq. Q DOES NOT NEED MASKING
            '''
            fb_lo, fb_hi = closed_interval_isect(lsec_hib+1, rsec_lob-1, vb_lo, vb_hi)
        else:
            '''
            Uniformly masked blocks
            LB and RB overlapps
            OR
            Seq. Q needs padding
            '''
            # Must use the un-sanitized interval as inputs
            lb_lo, lb_hi = closed_interval_isect(lsec_lob, rsec_hib, vb_lo, vb_hi)
            fb_lo, fb_hi =  -3, -4
            rb_lo, rb_hi =  -3, -4

        '''
        n = seqlen_k intersects with full blocks
        Note: this implies RB is excluded by (n < seqlen_k && n >= 0)
        '''
        if fb_lo * BLOCK_N < seqlen_k and seqlen_k < fb_hi * BLOCK_N + BLOCK_N:
            # BLOCK_N = 16
            # seqlen_k  | fb_hi | Comment   |
            # -------------------------------
            # 15        |  -1   |  Empty    |
            # 17        |  0    | [..., 0]  |
            fb_hi = seqlen_k // BLOCK_N - 1
            # n = seqlen_k & FB implies RB is empty
            rb_lo = masked_seq_k_block
            rb_hi = masked_seq_k_block
    else:
        lb_lo, lb_hi = -1, -2
        n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
        if mask_on_seq_q:
            lb_lo, lb_hi = 0, n_blocks - 1
            fb_lo, fb_hi = -1, -2
            rb_lo, rb_hi = -1, -2
            # need_to_handle_mask_on_seq_k = False # Handled by lb_* with [0, n_blocks]
        else:
            fb_lo, fb_hi = 0, seqlen_k // BLOCK_N
            # automatically be empty when mask_on_seq_k == False
            rb_lo, rb_hi = seqlen_k // BLOCK_N + 1, n_blocks - 1
            # need_to_handle_mask_on_seq_k = False # Handled by lb_* with [0, n_blocks]
        if fb_lo * BLOCK_N < seqlen_k and seqlen_k < fb_hi * BLOCK_N + BLOCK_N:
            fb_hi = seqlen_k // BLOCK_N - 1
            rb_lo = masked_seq_k_block
            rb_hi = masked_seq_k_block

    if DEBUG:
        tl.device_print('4 Final lb_lo', lb_lo)
        tl.device_print('4 Final lb_hi', lb_hi)
        tl.device_print('4 Final fb_lo', fb_lo)
        tl.device_print('4 Final fb_hi', fb_hi)
        tl.device_print('4 Final rb_lo', rb_lo)
        tl.device_print('4 Final rb_hi', rb_hi)

    return lb_lo, lb_hi, fb_lo, fb_hi, rb_lo, rb_hi

# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, _in_boundary_first, _in_boundary_second):
    boundary_first = _in_boundary_first
    boundary_second = _in_boundary_second
    """
    # Debugging GPU segfault
    boundary_first = 0
    boundary_second = 0
    mask = (offset_first[:, None] < boundary_first) & \
           (offset_second[None, :] < boundary_second)
    return tl.load(ptrs, mask=mask, other=0.0)
    """
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor

@triton.jit
def mload1d(
        REGS : tl.constexpr,
        i_base,
        i_start,
        i_nums,
):
    offs = tl.arange(0, REGS) + i_start
    i_ptrs = i_base + offs
    # return tl.load(i_base + offs)
    overflow = i_start + REGS - i_nums
    # if overflow <= 0:
    #     return tl.load(i_ptrs)
    i_ptrs_mask = tl.full([REGS], 1, dtype=tl.int1)
    i_ptrs_mask = i_ptrs_mask & (offs < i_nums)
    return tl.load(i_ptrs, mask=i_ptrs_mask, other=0.0)

@triton.jit
def mload2d(
        REG_ROWS : tl.constexpr,
        REG_COLS : tl.constexpr,
        i_base,
        i_start_row,
        i_start_col,
        i_rows,
        i_cols,
        stride_row,
        stride_col,
):
    off_rows = tl.arange(0, REG_ROWS) + i_start_row
    off_cols = tl.arange(0, REG_COLS) + i_start_col
    i_ptrs = i_base + off_rows[:, None] * stride_row + off_cols[None, :] * stride_col
    row_overflow = i_start_row + REG_ROWS - i_rows
    col_overflow = i_start_col + REG_COLS - i_cols
    # if row_overflow <= 0 and col_overflow <= 0:
    # if NOCHECK:
    #     return tl.load(i_ptrs)
    i_ptrs_mask = tl.full([REG_ROWS, REG_COLS], 1, dtype=tl.int1)
    if row_overflow > 0:
        i_ptrs_mask = i_ptrs_mask & (off_rows[:, None] < i_rows)
    if col_overflow > 0:
        i_ptrs_mask = i_ptrs_mask & (off_cols[None, :] < i_cols)
    return tl.load(i_ptrs, mask=i_ptrs_mask, other=0.0)

@triton.jit
def mstore2d(
        registers,
        REG_ROWS : tl.constexpr,
        REG_COLS : tl.constexpr,
        o_base,
        o_start_row,
        o_start_col,
        o_rows,
        o_cols,
        stride_row,
        stride_col,
):
    off_rows = tl.arange(0, REG_ROWS) + o_start_row
    off_cols = tl.arange(0, REG_COLS) + o_start_col
    o_ptrs = o_base + off_rows[:, None] * stride_row + off_cols[None, :] * stride_col
    o_ptrs_mask = tl.full([REG_ROWS, REG_COLS], 1, dtype=tl.int1)
    row_overflow = o_start_row + REG_ROWS - o_rows
    if row_overflow > 0:
        o_ptrs_mask = o_ptrs_mask & (off_rows[:, None] < o_rows)
    col_overflow = o_start_col + REG_COLS - o_cols
    if col_overflow > 0:
        o_ptrs_mask = o_ptrs_mask & (off_cols[None, :] < o_cols)
    tl.store(o_ptrs, registers, mask=o_ptrs_mask)
    # Do not return variables unused by the caller
    # See: https://github.com/triton-lang/triton/issues/5768
    # return o_ptrs, o_ptrs_mask
