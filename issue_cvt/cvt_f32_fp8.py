import torch

import triton
import triton.language as tl
from triton.runtime import driver

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# torch.float8_e4m3fn



@gluon.jit
def cvt_f32_f8_kernel(x_ptr, y_ptr, x_row_stride, x_col_stride, y_row_stride, y_col_stride,
                      M: gl.constexpr, N: gl.constexpr):
    '''
    This kernel assume the input tensor has fp8 elements

    The generate ISA is as follows

        v_mul_lo_u32 v2, s6, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[2:3], 0, v[2:3]
	global_load_dwordx2 v[2:3], v[2:3], off
	s_mov_b32 s0, 0x3020104
	v_mul_lo_u32 v0, s7, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[8:9], v[0:1], 2, s[4:5]
	s_waitcnt vmcnt(0)
	v_perm_b32 v6, v2, v2, s0
	v_cvt_scalef32_pk_f32_fp8 v[4:5], v6, 1.0
	v_cvt_scalef32_pk_f32_fp8 v[6:7], v6, 1.0 op_sel:[1,0,0]
	v_cvt_scalef32_pk_f32_fp8 v[0:1], v3, 1.0
	v_cvt_scalef32_pk_f32_fp8 v[2:3], v3, 1.0 op_sel:[1,0,0]
	global_store_dwordx4 v[8:9], v[4:7], off
	global_store_dwordx4 v[8:9], v[0:3], off offset:16


    Why does it need to rotate the 4 fp8 elements in v2,
    i.e. `v_perm v6, v2, v2, 0x3020104` ?

    Why does buffer_load only load ubyte at a time?
    '''

    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [0, 1])

    col_offsets = gl.arange(0, N, gl.SliceLayout(0, layout))
    row_offsets = gl.arange(0, M, gl.SliceLayout(1, layout))
    x_offsets = row_offsets[:, None] * x_row_stride + col_offsets[None, :] * x_col_stride
    x_ptrs = x_ptr + row_offsets[:, None] * x_row_stride + col_offsets[None, :] * x_col_stride
    y_offsets = row_offsets[:, None] * y_row_stride + col_offsets[None, :] * y_col_stride
    y_ptrs = y_ptr + row_offsets[:, None] * y_row_stride + col_offsets[None, :] * y_col_stride

    #x = gl.load(x_ptrs)
    x = gl.amd.cdna3.buffer_load(ptr=x_ptr, offsets=x_offsets)
    y = x.to(tl.float32)
    #gl.store(y_ptrs, y)
    gl.amd.cdna3.buffer_store(ptr=y_ptr, offsets=y_offsets, stored_value=y)


def test_cvt_f32_f8():

    M, N = 128, 64
    x = torch.randn((N, M), dtype=torch.float32, device=DEVICE).T
    x = x.to(torch.float8_e4m3fn)
    y = torch.randn((N, M), dtype=torch.float32, device=DEVICE).T

    grid = (1, 1)
    num_warps=1

    cvt_f32_f8_kernel[grid](x, y, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                            M, N, num_warps=num_warps)


@gluon.jit
def cvt_f32_f8_packed_kernel(x_ptr, y_ptr, x_row_stride, x_col_stride, y_row_stride, y_col_stride,
                             M: gl.constexpr, N: gl.constexpr):
    '''
    In this kernel, we assume each element of x is an int32 that is a packed 4xfp8.
    We also use LDS to convert the layout of the loaded input values

    The generated ISA is as follows

        s_mov_b64 s[8:9], s[2:3]
	v_lshlrev_b32_e32 v0, 2, v0
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_mul_lo_u32 v1, v0, s6
	buffer_load_dwordx2 v[6:7], v1, s[8:11], 0 offen
	s_mov_b32 s0, s7
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, s10
	s_mov_b32 s7, s11
	v_mul_lo_u32 v8, v0, s0
	s_waitcnt vmcnt(0)
	v_cvt_scalef32_pk_f32_fp8 v[0:1], v6, 1.0
	v_cvt_scalef32_pk_f32_fp8 v[2:3], v6, 1.0 op_sel:[1,0,0]
	v_cvt_scalef32_pk_f32_fp8 v[4:5], v7, 1.0
	v_cvt_scalef32_pk_f32_fp8 v[6:7], v7, 1.0 op_sel:[1,0,0]
	buffer_store_dwordx4 v[0:3], v8, s[4:7], 0 offen
	buffer_store_dwordx4 v[4:7], v8, s[4:7], 0 offen offset:16

    '''

    layout3D: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 64], [1, 1, 1], [1, 0, 2])
    ## Global load layout for x, which must be a slice the above 3d layout
    layout2D: gl.constexpr = gl.SliceLayout(1, layout3D)

    ## shift distance tensor layout, which must be the above 3d layout w/o dim0 and dim2
    layoutOffset: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout3D))

    ## Global store layout for y
    layout_y: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [0, 1])

    x_col_offsets = gl.arange(0, N, gl.SliceLayout(0, layout2D))
    x_row_offsets = gl.arange(0, M, gl.SliceLayout(1, layout2D))
    x_offsets = x_row_offsets[:, None] * x_row_stride + x_col_offsets[None, :] * x_col_stride
    x_ptrs = x_ptr + x_row_offsets[:, None] * x_row_stride + x_col_offsets[None, :] * x_col_stride

    #x = gl.load(x_ptrs)
    x = gl.amd.cdna3.buffer_load(ptr=x_ptr, offsets=x_offsets)
    ## x is M x N of int32 elements
    ## we duplicate each elements 4 times to make it M x 4 x N
    x = x.expand_dims(1).broadcast_to(M, 4, N)
    ## The 4 duplicated values need to shr different bits: 0, 8, 16, 24
    offsets = gl.arange(0, 4, layoutOffset) * 8
    x = x >> offsets[None, :, None]

    ## reshape back to M*4 x N
    x = x.reshape(M*4, N)

    ## reinterpret each element as int8 and bitcast to fp8 and convert to f32
    y = x.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)

    y_col_offsets = gl.arange(0, N, gl.SliceLayout(0, layout_y))
    y_row_offsets = gl.arange(0, M*4, gl.SliceLayout(1, layout_y))
    y_offsets = y_row_offsets[:, None] * y_row_stride + y_col_offsets[None, :] * y_col_stride
    y_ptrs = y_ptr + y_row_offsets[:, None] * y_row_stride + y_col_offsets[None, :] * y_col_stride
    gl.amd.cdna3.buffer_store(ptr=y_ptr, offsets=y_offsets, stored_value=y)


def test_cvt_f32_f8_packed():

    M, N = 32, 64
    x = torch.randn((N, M), dtype=torch.float32, device=DEVICE).T
    x = x.to(torch.int32)
    y = torch.randn((N, M*4), dtype=torch.float32, device=DEVICE).T

    grid = (1, 1)
    num_warps=1

    cvt_f32_f8_packed_kernel[grid](x, y, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                                   M, N, num_warps=num_warps)


@gluon.jit
def cvt_f32_f8_packed_LDS_kernel(x_ptr, y_ptr, x_row_stride, x_col_stride, y_row_stride, y_col_stride,
                             M: gl.constexpr, N: gl.constexpr):
    '''
    In this kernel, we assume each element of x is an int32 that is a packed 4xfp8.
    We also use LDS to convert the layout of the loaded input values

    '''

    layout3D: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 64], [1, 1, 1], [1, 0, 2])
    ## Global load layout for x, which must be a slice the above 3d layout
    layout_x: gl.constexpr = gl.BlockedLayout([4, 1], [8, 8], [1, 1], [0, 1])

    layout2D: gl.constexpr = gl.SliceLayout(1, layout3D)

    ## shift distance tensor layout, which must be the above 3d layout w/o dim0 and dim2
    layoutOffset: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout3D))

    ## Global store layout for y
    layout_y: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [0, 1])

    x_col_offsets = gl.arange(0, N, gl.SliceLayout(0, layout_x))
    x_row_offsets = gl.arange(0, M, gl.SliceLayout(1, layout_x))
    x_offsets = x_row_offsets[:, None] * x_row_stride + x_col_offsets[None, :] * x_col_stride
    x_ptrs = x_ptr + x_row_offsets[:, None] * x_row_stride + x_col_offsets[None, :] * x_col_stride

    #x = gl.load(x_ptrs)
    x = gl.amd.cdna3.buffer_load(ptr=x_ptr, offsets=x_offsets)
    ## x is M x N of int32 elements
    ## We first convert its layout to layout2D
    x = gl.convert_layout(x, layout2D)

    ## we duplicate each elements 4 times to make it M x 4 x N
    x = x.expand_dims(1).broadcast_to(M, 4, N)
    ## The 4 duplicated values need to shr different bits: 0, 8, 16, 24
    offsets = gl.arange(0, 4, layoutOffset) * 8
    x = x >> offsets[None, :, None]

    ## reshape back to M*4 x N
    x = x.reshape(M*4, N)

    ## reinterpret each element as int8 and bitcast to fp8 and convert to f32
    y = x.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)

    y_col_offsets = gl.arange(0, N, gl.SliceLayout(0, layout_y))
    y_row_offsets = gl.arange(0, M*4, gl.SliceLayout(1, layout_y))
    y_ptrs = y_ptr + y_row_offsets[:, None] * y_row_stride + y_col_offsets[None, :] * y_col_stride
    gl.store(y_ptrs, y)


def test_cvt_f32_f8_packed_LDS():

    M, N = 32, 64
    x = torch.randn((N, M), dtype=torch.float32, device=DEVICE).T
    x = x.to(torch.int32)
    y = torch.randn((N, M*4), dtype=torch.float32, device=DEVICE).T

    grid = (1, 1)
    num_warps=1

    cvt_f32_f8_packed_LDS_kernel[grid](x, y, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                                       M, N, num_warps=num_warps)


if __name__ == "__main__":
    test_cvt_f32_f8()
    #test_cvt_f32_f8_packed()
    #test_cvt_f32_f8_packed_LDS()
