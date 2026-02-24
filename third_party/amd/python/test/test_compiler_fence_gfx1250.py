"""
E2E test verifying that the compiler fence prevents MachineSink from sinking
LDS loads past barriers in tensor atomic RMW lowering on gfx1250.

Background: When buffer atomics are not enabled, AtomicRMWOp lowering creates
a condBr (s_cbranch) to mask which threads execute the atomic. LLVM's
MachineSink can sink preceding LDS loads (from reduce cross-warp communication)
past barriers into the condBr's successor blocks, causing incorrect results.

The compiler fence (inline asm with ~{memory}) has mayStore()=true, which sets
SawStore in MachineSink's bottom-up walk, preventing loads from being sunk.

This test compiles a kernel that exercises the vulnerable pattern:
  2D load -> tl.sum (reduce, uses LDS) -> tl.atomic_add

It checks the AMDGCN assembly to verify that the last ds_load (LDS load
from the reduce) appears BEFORE the s_cbranch (condBr from emitAtomicRMW),
not after it. If MachineSink sinks the load, it would appear after the branch.

When running on a gfx1250 GPU, it also verifies correctness of the result.

See https://github.com/llvm/llvm-project/issues/181708.
"""

import re

import numpy as np
import triton
import triton.language as tl
from triton.compiler import ASTSource, compile as triton_compile
from triton.backends.compiler import GPUTarget

from triton._internal_testing import is_hip_gfx1250


def get_kernel_body(amdgcn, kernel_name):
    """Extract the kernel function body from AMDGCN assembly."""
    pattern = rf"^{re.escape(kernel_name)}:\s*;.*?$(.+?)^\s*\.size\s+{re.escape(kernel_name)}"
    match = re.search(pattern, amdgcn, flags=re.DOTALL | re.MULTILINE)
    assert match, f"couldn't find kernel body for {kernel_name}"
    return match.group(1)


def test_ds_load_not_sunk_past_cbranch():
    """Verify that ds_load from reduce is not sunk past s_cbranch from atomic."""

    shape0, shape1 = 4, 64

    @triton.jit
    def kernel(Z, X, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        x = tl.load(X + off0[:, None] * SHAPE1 + off1[None, :])
        z = tl.sum(x, axis=1)
        tl.atomic_add(Z + off0, z)

    # Assembly check: compile and verify instruction ordering
    target = GPUTarget("hip", "gfx1250", 32)
    src = ASTSource(
        fn=kernel,
        signature={"Z": "*fp64", "X": "*fp64"},
        constexprs={"SHAPE0": shape0, "SHAPE1": shape1},
    )
    compiled = triton_compile(src, target=target)
    amdgcn = compiled.asm["amdgcn"]

    body = get_kernel_body(amdgcn, "kernel")
    lines = body.splitlines()

    # Find the positions of key instructions
    last_ds_load = -1
    first_cbranch_after_last_ds_load = -1
    first_global_atomic = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("ds_load"):
            last_ds_load = i
            first_cbranch_after_last_ds_load = -1  # reset
        elif stripped.startswith("s_cbranch") and last_ds_load >= 0 and first_cbranch_after_last_ds_load < 0:
            first_cbranch_after_last_ds_load = i
        elif stripped.startswith("global_atomic") and first_global_atomic < 0:
            first_global_atomic = i

    assert last_ds_load >= 0, "expected ds_load instructions (from reduce LDS communication)"
    assert first_cbranch_after_last_ds_load >= 0, "expected s_cbranch (from atomic condBr thread masking)"
    assert first_global_atomic >= 0, "expected global_atomic instruction"

    # The critical check: ds_load must come BEFORE the s_cbranch, not after.
    # If MachineSink sinks the load, it would appear in a successor block
    # after the branch.
    assert last_ds_load < first_cbranch_after_last_ds_load, (
        f"ds_load (line {last_ds_load}) was sunk past s_cbranch "
        f"(line {first_cbranch_after_last_ds_load}). "
        f"The compiler fence may not be working.\n"
        f"ds_load line: {lines[last_ds_load].strip()}\n"
        f"s_cbranch line: {lines[first_cbranch_after_last_ds_load].strip()}")

    # Also verify the overall ordering: ds_load < s_cbranch < global_atomic
    assert first_cbranch_after_last_ds_load < first_global_atomic, (
        f"s_cbranch (line {first_cbranch_after_last_ds_load}) should come before "
        f"global_atomic (line {first_global_atomic})")

    # Correctness check: run the kernel on GPU and verify results
    if not is_hip_gfx1250():
        return

    import torch

    x = torch.randn((shape0, shape1), device="cuda", dtype=torch.float64)
    z = torch.zeros((shape0, ), device="cuda", dtype=torch.float64)

    kernel[(1, )](z, x, shape0, shape1)

    z_ref = x.sum(axis=1)
    np.testing.assert_allclose(z.cpu().numpy(), z_ref.cpu().numpy(), rtol=1e-5)
