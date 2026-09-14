"""gfx950 workaround for the CDNA4 MFMA operand read-skip erratum.

See ROCM-27743 / DEGGIGX90-5078 / llvm/llvm-project#206825, and
disable_packed_fp32_ops() in third_party/amd/backend/compiler.py.

An MFMA that skips its VGPR read wrongly makes a v_pk_* op on a co-executing
wave skip its own operand read, so a gfx950 kernel containing MFMA must not emit
packed-FP32 opcodes. Compile-only -- no gfx950 hardware required.
"""

import os
import re
from pathlib import Path

import pytest
import triton
from triton.backends.amd.compiler import disable_packed_fp32_ops
from triton.backends.compiler import GPUTarget

TTIR_PATH = str(Path(__file__).parent / "attn_fwd.ttir")
GFX950_TARGET = GPUTarget("hip", "gfx950", 64)
GFX942_TARGET = GPUTarget("hip", "gfx942", 64)

PACKED_FP32 = re.compile(r"v_pk_(?:add|sub|mul|fma)_f32")


@pytest.fixture
def knob_unset():
    """Run with the knob on its default (auto), and restore it afterwards."""
    key = "TRITON_HIP_DISABLE_PACKED_FP32_OPS"
    prev = os.environ.pop(key, None)
    triton.knobs.amd.disable_packed_fp32_ops = triton.knobs.env
    yield
    os.environ.pop(key, None)
    if prev is not None:
        os.environ[key] = prev
    triton.knobs.amd.disable_packed_fp32_ops = triton.knobs.env


def kernel_asm(target):
    amdgcn = triton.compile(TTIR_PATH, target=target).asm["amdgcn"]
    body = re.findall(r"^attn_fwd:(.*); -- End function", amdgcn, flags=re.DOTALL | re.MULTILINE)
    assert len(body) == 1, "couldn't find kernel body in asm"
    return body[0]


def test_gate(knob_unset):
    mfma_ir = "  %0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(...)\n"
    # Auto: gfx950 kernels that actually contain MFMA/WMMA, and nothing else.
    assert disable_packed_fp32_ops("gfx950", mfma_ir)
    assert not disable_packed_fp32_ops("gfx950", "  %0 = fadd <2 x float> %a, %b\n")
    assert not disable_packed_fp32_ops("gfx942", mfma_ir)
    assert not disable_packed_fp32_ops("gfx1250", mfma_ir)


@pytest.mark.parametrize("forced, expected", [(True, True), (False, False)])
def test_gate_knob_overrides_auto(knob_unset, forced, expected):
    triton.knobs.amd.disable_packed_fp32_ops = forced
    # The override wins over both the arch check and the MFMA check.
    assert disable_packed_fp32_ops("gfx950", "") == expected
    assert disable_packed_fp32_ops("gfx942", "") == expected


def test_gfx950_mfma_kernel_has_no_packed_fp32(knob_unset):
    body = kernel_asm(GFX950_TARGET)
    assert "mfma" in body, "expected an MFMA kernel; the workaround would not be gated on"
    found = PACKED_FP32.findall(body)
    assert not found, f"gfx950 MFMA kernel must not emit packed FP32 ops, got {set(found)}"


def test_gfx942_keeps_packed_fp32(knob_unset):
    # The erratum is CDNA4-only; older targets must keep double-rate packed math.
    body = kernel_asm(GFX942_TARGET)
    assert PACKED_FP32.search(body), "gfx942 should be unaffected by the gfx950 workaround"


def test_knob_can_force_workaround_off(knob_unset):
    # Escape hatch, and a check that the workaround is what removes the ops.
    triton.knobs.amd.disable_packed_fp32_ops = False
    body = kernel_asm(GFX950_TARGET)
    assert PACKED_FP32.search(body), "disabling the workaround should restore packed FP32 ops"
