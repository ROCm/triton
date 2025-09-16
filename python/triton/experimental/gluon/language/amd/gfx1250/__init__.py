from ..._core import builtin
from .._ops import _wmma
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check
from ..._layouts import DotOperandLayout
from .._layouts import AMDWMMALayout

__all__ = ["wmma", "wmma_scaled"]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(3, a, b, acc, _semantic)


@builtin
def wmma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None):
    """
    AMD Scaled WMMA operation.

    ```
    c = a * a_scale @ b * b_scale + acc
    ```

    `a` and `b` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (tensor): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1'.
        b (tensor): The operand B to be multiplied.
        b_scale (tensor): Scale factor for operand B.
        b_format (str): Format of the operand B. Available formats: `e2m1'.
        acc (tensor): Accumulator tensor.
    """

    _check(acc is not None, lambda: "acc is required")
    layout = acc.type.layout
    _check(isinstance(layout, AMDWMMALayout), lambda: "Expected layout to be an instance of AMDWMMALayout")
    _check(
        isinstance(a.type.layout, DotOperandLayout) and a.type.layout.parent == layout,
        lambda: "Expected a's layout to be a DotOperandLayout with parent matching AMDWMMALayout")
    _check(
        isinstance(b.type.layout, DotOperandLayout) and b.type.layout.parent == layout,
        lambda: "Expected b's layout to be a DotOperandLayout with parent matching AMDWMMALayout")

    # TODO: Add more formats
    assert a_format.value in {"e2m1"}, f"Unsupported lhs_format: {a_format.value}"
    assert b_format.value in {"e2m1"}, f"Unsupported rhs_format: {b_format.value}"

    handle = _semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, fast_math=False, lhs_k_pack=True,
                                  rhs_k_pack=True, out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)
