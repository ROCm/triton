import triton.language as tl
from triton.language.extra import libdevice
import triton


@triton.jit
def gelu_kernel(yptr, xptr, pi: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    input_ptr = xptr + BLOCK_SIZE * pid
    output_ptr = yptr + BLOCK_SIZE * pid
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(tl.multiple_of(input_ptr + col_offsets, 256))
    y = 0.5 * x * (1 + libdevice.tanh(tl.sqrt(2.0 / pi) * (x + 0.044715 * x * x * x)))
    tl.store(tl.multiple_of(output_ptr + col_offsets, 256), y)
