"""Unit tests for the amdgcnas_gfx12 peephole passes."""
import textwrap

import pytest

from triton.tools.amdgcnas_gfx12 import (
    RegionMarker,
    _encode_msb_byte,
    emit_program,
    find_region_spans,
    hoist_loop_invariant_addrs,
    merge_dscnt_waits,
    overlap_wmma_with_barrier,
    parse_asm,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

def _program(body: str) -> str:
    """Wrap a loop-body snippet in the minimum scaffolding so parse_asm
    treats it as a basic block."""
    return textwrap.dedent(body)


# Production-format region marker (``;; Region N: ...``).
MARKER_0 = "\t;;#ASMSTART\n\t;; Region 0: 4 wmma, 0 GR, 4 LR\n\t;;#ASMEND"
MARKER_1 = "\t;;#ASMSTART\n\t;; Region 1: 4 wmma, 0 GR, 4 LR\n\t;;#ASMEND"


# -------------------------------------------------------------------------
# Region detection
# -------------------------------------------------------------------------

class TestFindRegionSpans:

    def test_single_region(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        spans = find_region_spans(prog)
        assert len(spans) == 1
        assert spans[0].marker.region == 0
        assert spans[0].marker.wmma == 4
        assert spans[0].marker.ds_load == 4  # LR
        assert spans[0].marker.tdm == 0      # GR

    def test_two_regions_cover_full_block(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            f"{MARKER_1}\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        spans = find_region_spans(prog)
        assert len(spans) == 2
        assert spans[0].marker.region == 0
        assert spans[1].marker.region == 1
        # The first span ends where the second begins.
        assert spans[0].end == spans[1].start
        # The second span covers the rest of the block.
        bb = prog.blocks[0]
        assert spans[1].end == len(bb.instructions)


# -------------------------------------------------------------------------
# merge_dscnt_waits
# -------------------------------------------------------------------------

class TestMergeDscntWaits:

    def _build(self, region_body: str) -> str:
        # Wrap the region body in a self-branching loop so the pass'
        # loop-only restriction is satisfied.
        return (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_0}\n"
            f"{region_body}"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )

    def test_last_wait_minus_preceding_loads(self):
        # Region has 2 ds_loads before the last wait.  Last wait is
        # s_wait_dscnt 0x10 (=16).  Consolidated value = 16 - 2 = 14 (0xe).
        body = (
            "\ts_wait_dscnt 0xe\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\tds_load_b128 v[50:53], v100\n"
            "\tds_load_b128 v[54:57], v100 offset:32\n"
            "\ts_wait_dscnt 0x10\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
        )
        prog = parse_asm(self._build(body))
        removed = merge_dscnt_waits(prog)
        assert removed == 2
        out = emit_program(prog)
        # Exactly one wait remains, with value 0xe, inserted right after
        # the marker.
        waits = [ln for ln in out.splitlines() if 's_wait_dscnt' in ln]
        assert len(waits) == 1
        assert '0xe' in waits[0]

    def test_wait_underflow_clamps_to_zero(self):
        # Last wait is 0x2 after 10 ds_loads -> V_new = -8 -> clamp to 0.
        body = ""
        for i in range(10):
            body += f"\tds_load_b128 v[{50+i*4}:{53+i*4}], v100\n"
        body += "\ts_wait_dscnt 0x2\n"
        body += "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
        prog = parse_asm(self._build(body))
        merge_dscnt_waits(prog)
        out = emit_program(prog)
        waits = [ln for ln in out.splitlines() if 's_wait_dscnt' in ln]
        assert len(waits) == 1
        assert '0x0' in waits[0]

    def test_no_waits_no_change(self):
        body = (
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\tds_load_b128 v[50:53], v100\n"
        )
        before = self._build(body)
        prog = parse_asm(before)
        removed = merge_dscnt_waits(prog)
        assert removed == 0
        assert emit_program(prog) == before

    def test_indentation_preserved(self):
        body = (
            "\ts_wait_dscnt 0x5\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
        )
        prog = parse_asm(self._build(body))
        merge_dscnt_waits(prog)
        out = emit_program(prog)
        # The inserted wait should be tab-indented to match surrounding code.
        for line in out.splitlines():
            if 's_wait_dscnt' in line:
                assert line.startswith('\t')
                break

    def test_waits_in_two_regions_handled_independently(self):
        # Region 0: V_last=0x5, 0 ds_loads before it -> new value 0x5.
        # Region 1: V_last=0x8, 2 ds_loads before it -> new value 0x6.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_0}\n"
            "\ts_wait_dscnt 0x5\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            f"{MARKER_1}\n"
            "\tds_load_b128 v[50:53], v100\n"
            "\tds_load_b128 v[54:57], v100 offset:32\n"
            "\ts_wait_dscnt 0x8\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        merge_dscnt_waits(prog)
        out = emit_program(prog)
        waits = [ln.strip() for ln in out.splitlines() if 's_wait_dscnt' in ln]
        assert waits == ['s_wait_dscnt 0x5', 's_wait_dscnt 0x6']

    def test_epilogue_region_untouched(self):
        # A region before the ``s_cbranch`` (loop body) should be merged;
        # a region after it (epilogue, executed once) should be left
        # alone because the loop-only restriction excludes post-branch
        # regions.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_0}\n"
            "\tds_load_b128 v[50:53], v100\n"
            "\ts_wait_dscnt 0xe\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_wait_dscnt 0x10\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            f"{MARKER_1}\n"
            "\ts_wait_dscnt 0x3\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
            "\ts_wait_dscnt 0x4\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        removed = merge_dscnt_waits(prog)
        assert removed == 2
        out = emit_program(prog)
        waits = [ln.strip() for ln in out.splitlines() if 's_wait_dscnt' in ln]
        # Loop region merged to a single wait; epilogue's two waits preserved.
        assert waits.count('s_wait_dscnt 0x3') == 1
        assert waits.count('s_wait_dscnt 0x4') == 1
        # And the merged loop wait is still present.
        assert any(w.startswith('s_wait_dscnt 0x') and w not in
                   ('s_wait_dscnt 0x3', 's_wait_dscnt 0x4') for w in waits)

    def test_no_loop_no_change(self):
        # Program with no self-branching loop -> pass is a no-op.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_0}\n"
            "\tds_load_b128 v[50:53], v100\n"
            "\ts_wait_dscnt 0xe\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_wait_dscnt 0x10\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        before = src
        prog = parse_asm(before)
        removed = merge_dscnt_waits(prog)
        assert removed == 0
        assert emit_program(prog) == before


# -------------------------------------------------------------------------
# overlap_wmma_with_barrier
# -------------------------------------------------------------------------

class TestOverlapWmmaWithBarrier:

    def _program(self, body: str) -> str:
        return (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{body}"
            "\ts_endpgm\n"
        )

    def _opcodes(self, prog):
        bb = prog.blocks[0]
        return [i.opcode for i in bb.instructions if i.opcode]

    def test_simple_signal_wait_wmma(self):
        body = (
            "\ts_barrier_signal -1\n"
            "\ts_barrier_wait -1\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
        )
        prog = parse_asm(self._program(body))
        moved = overlap_wmma_with_barrier(prog)
        assert moved == 1
        assert self._opcodes(prog) == [
            's_barrier_signal', 'v_wmma_f32_16x16x32_f16', 's_barrier_wait',
            's_endpgm',
        ]

    def test_with_delay_alu(self):
        body = (
            "\ts_barrier_signal -1\n"
            "\ts_barrier_wait -1\n"
            "\ts_delay_alu instid0(TRANS32_DEP_1)\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
        )
        prog = parse_asm(self._program(body))
        moved = overlap_wmma_with_barrier(prog)
        assert moved == 1
        # s_delay_alu must remain adjacent to the wmma.
        ops = self._opcodes(prog)
        assert ops == [
            's_barrier_signal', 's_delay_alu', 'v_wmma_f32_16x16x32_f16',
            's_barrier_wait', 's_endpgm',
        ]

    def test_no_following_wmma_no_change(self):
        body = (
            "\ts_barrier_signal -1\n"
            "\ts_barrier_wait -1\n"
            "\tds_load_b128 v[50:53], v100\n"
        )
        before = self._program(body)
        prog = parse_asm(before)
        moved = overlap_wmma_with_barrier(prog)
        assert moved == 0
        assert emit_program(prog) == before

    def test_multiple_pairs_each_hoisted(self):
        body = (
            "\ts_barrier_signal -1\n"
            "\ts_barrier_wait -1\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_barrier_signal -1\n"
            "\ts_barrier_wait -1\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
        )
        prog = parse_asm(self._program(body))
        moved = overlap_wmma_with_barrier(prog)
        assert moved == 2

    def test_only_signal_unchanged(self):
        # Lone s_barrier_signal without a matching wait is left alone.
        body = (
            "\ts_barrier_signal -1\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
        )
        before = self._program(body)
        prog = parse_asm(before)
        moved = overlap_wmma_with_barrier(prog)
        assert moved == 0
        assert emit_program(prog) == before


# -------------------------------------------------------------------------
# hoist_loop_invariant_addrs
# -------------------------------------------------------------------------

class TestEncodeMsbByte:

    def test_all_zero(self):
        assert _encode_msb_byte(dst=0, src0=0, src1=0, src2=0) == 0x00

    def test_dst_only(self):
        # 0x80 → dst=2, others=0
        assert _encode_msb_byte(dst=2, src0=0, src1=0, src2=0) == 0x80

    def test_all_fields(self):
        # bits 1-0=src0, 3-2=src1, 5-4=src2, 7-6=dst
        # dst=1, src0=2, src1=2, src2=1 → 0x5a
        assert _encode_msb_byte(dst=1, src0=2, src1=2, src2=1) == 0x5a


def _wrap_loop(body: str, prologue: str = "", epilogue: str = "") -> str:
    """Wrap an assembly body in a minimal program with an explicit
    preheader (``.Lpre``) and a self-branching loop (``.Lloop``)."""
    return (
        ".text\n"
        ".globl test_kernel\n"
        "test_kernel:\n"
        "; %bb.0:\n"
        ".Lpre:\n"
        f"{prologue}"
        ".Lloop:\n"
        f"{body}"
        "\ts_cbranch_scc1 .Lloop\n"
        f"{epilogue}"
        "\ts_endpgm\n"
    )


class TestHoistLoopInvariantAddrs:

    def test_basic_single_candidate(self):
        # Prologue defines v131.  Loop has one v_add that reads it.
        # (Plus a dummy ds_load that uses the computed address, to make
        # it resemble the v9 pattern.)
        prologue = "\tv_add3_u32 v131, 1, 2, 0\n"
        body = (
            "\tv_add_nc_u32_e32 v132, 0x100, v131\n"
            "\tds_load_b128 v[10:13], v132\n"
        )
        prog = parse_asm(_wrap_loop(body, prologue=prologue))
        n = hoist_loop_invariant_addrs(prog)
        assert n == 1
        out = emit_program(prog)
        # The v_add moved to the preheader.
        pre_idx = out.index(".Lpre:")
        loop_idx = out.index(".Lloop:")
        vadd_idx = out.index("v_add_nc_u32_e32 v132")
        assert pre_idx < vadd_idx < loop_idx

    def test_forward_through_trivial_copy(self):
        # Loop has a trivial copy (v_add ..., 0, ...) that feeds an
        # otherwise-hoistable v_add.  Both should be hoisted (after the
        # copy is forwarded) and the copy removed from the loop.
        prologue = "\tv_add3_u32 v131, 1, 2, 0\n"
        body = (
            "\tv_add_nc_u32_e32 v38, 0, v131\n"     # trivial copy
            "\tv_add_nc_u32_e32 v132, 0x100, v38\n"  # consumes copy
            "\tds_load_b128 v[10:13], v132\n"
        )
        prog = parse_asm(_wrap_loop(body, prologue=prologue))
        n = hoist_loop_invariant_addrs(prog)
        # One address v_add + one trivial copy removed.
        assert n == 2
        out = emit_program(prog)
        # Address v_add now reads the prologue root (v131), not v38.
        pre_idx = out.index(".Lpre:")
        loop_idx = out.index(".Lloop:")
        hoisted_line = [
            ln for ln in out.splitlines()
            if "v_add_nc_u32_e32 v132" in ln
        ][0]
        assert "v131" in hoisted_line
        assert "v38" not in hoisted_line
        # Trivial copy is gone entirely.
        assert "v_add_nc_u32_e32 v38" not in out

    def test_skip_when_source_not_invariant(self):
        # Source of the v_add is defined inside the loop (not a trivial
        # copy, so we can't forward).
        body = (
            "\tv_add_u32 v38, v100, v101\n"           # non-trivial loop def
            "\tv_add_nc_u32_e32 v132, 0x100, v38\n"   # source isn't invariant
            "\tds_load_b128 v[10:13], v132\n"
        )
        prog = parse_asm(_wrap_loop(body))
        n = hoist_loop_invariant_addrs(prog)
        assert n == 0

    def test_hoist_with_rename_when_dst_aliased(self):
        # dst v132 is also written as a ds_load data range (simulates the
        # v9 v646/v512 dual-role case).  Instead of bailing, the pass
        # should allocate a fresh VGPR, rewrite the downstream address
        # consumer to that new register, and hoist the v_add.  The data
        # ds_load that writes v[132:135] must be left untouched.
        prologue = "\tv_add3_u32 v131, 1, 2, 0\n"
        body = (
            "\tds_load_b128 v[132:135], v200\n"       # data write to v132
            "\tv_add_nc_u32_e32 v132, 0x100, v131\n"  # addr def (candidate)
            "\tds_load_b128 v[10:13], v132\n"          # addr use (rename)
            "\tv_wmma_f32_16x16x32_f16 v[200:207], v[10:17], v[132:139], v[200:207]\n"
        )
        prog = parse_asm(_wrap_loop(body, prologue=prologue))
        n = hoist_loop_invariant_addrs(prog)
        assert n == 1
        out = emit_program(prog)
        # v_add is now in the preheader and writes a different VGPR.
        pre = out[out.index(".Lpre:"):out.index(".Lloop:")]
        loop = out[out.index(".Lloop:"):]
        assert "v_add_nc_u32_e32" in pre
        # No v_add remains in the loop body.
        assert "v_add_nc_u32_e32" not in loop
        # The data ds_load that writes v[132:135] is unchanged.
        assert "ds_load_b128 v[132:135], v200" in loop
        # The downstream addr-use ds_load no longer references v132: it
        # was renamed to the fresh register the pass allocated.
        addr_line = [ln for ln in loop.splitlines()
                     if "ds_load_b128 v[10:13]" in ln][0]
        assert ", v132" not in addr_line
        # The wmma's src2 v[132:139] remains as a data consumer
        # (range reads are never renamed).
        assert "v[132:139]" in loop

    def test_rename_stops_at_kill_def(self):
        # If a later instruction redefines the addr VGPR (data write),
        # reads beyond that kill point must NOT be renamed -- they
        # consume the new (data) value, not the hoisted address.
        prologue = "\tv_add3_u32 v131, 1, 2, 0\n"
        body = (
            "\tds_load_b128 v[132:135], v200\n"              # initial data def
            "\tv_add_nc_u32_e32 v132, 0x100, v131\n"         # addr def
            "\tds_load_b128 v[10:13], v132\n"                 # addr use (rename)
            "\tds_load_b128 v[132:135], v201 offset:32\n"     # kill (redef)
            "\tv_add_u32 v50, v132, 0\n"                      # data use (keep)
        )
        prog = parse_asm(_wrap_loop(body, prologue=prologue))
        n = hoist_loop_invariant_addrs(prog)
        assert n == 1
        out = emit_program(prog)
        loop = out[out.index(".Lloop:"):]
        # Data use after the kill still reads v132.
        post_kill_line = [ln for ln in loop.splitlines()
                          if "v_add_u32 v50" in ln][0]
        assert "v132" in post_kill_line

    def test_strips_adjacent_wait_and_delay(self):
        # The s_wait_alu after a hoisted v_add was emitted to drain the
        # v_add's VA_VDST counter; after hoist it would just stall the
        # wave waiting for whatever VALU happens to be pending.  Same
        # for the s_delay_alu before the v_add (a hint about its
        # latency).  Both should be removed alongside the v_add.
        prologue = "\tv_add3_u32 v131, 1, 2, 0\n"
        body = (
            "\ts_delay_alu instid0(VALU_DEP_1)\n"
            "\tv_add_nc_u32_e32 v132, 0x100, v131\n"
            "\ts_wait_alu depctr_va_vdst(0)\n"
            "\tds_load_b128 v[10:13], v132\n"
        )
        prog = parse_asm(_wrap_loop(body, prologue=prologue))
        n = hoist_loop_invariant_addrs(prog)
        # 1 v_add + 1 s_delay_alu + 1 s_wait_alu.
        assert n == 3
        out = emit_program(prog)
        loop = out[out.index(".Lloop:"):]
        assert "v_add_nc_u32_e32" not in loop
        assert "s_delay_alu" not in loop
        assert "s_wait_alu" not in loop

    def test_multiple_candidates_single_msb_setup(self):
        # Two hoistable v_adds with the same MSB requirement should be
        # grouped under a single s_set_vgpr_msb.
        prologue = "\tv_add3_u32 v131, 1, 2, 0\n"
        body = (
            "\tv_add_nc_u32_e32 v132, 0x100, v131\n"
            "\tv_add_nc_u32_e32 v133, 0x200, v131\n"
            "\tds_load_b128 v[10:13], v132\n"
            "\tds_load_b128 v[14:17], v133\n"
        )
        prog = parse_asm(_wrap_loop(body, prologue=prologue))
        n = hoist_loop_invariant_addrs(prog)
        assert n == 2
        out = emit_program(prog)
        # Exactly one s_set_vgpr_msb was emitted by the pass.
        pre_section = out[out.index(".Lpre:"):out.index(".Lloop:")]
        assert pre_section.count("s_set_vgpr_msb") == 1


# -------------------------------------------------------------------------
# Kernel end-to-end regression (fixture-driven)
# -------------------------------------------------------------------------

# Fixture file names for each kernel we exercise end-to-end.  Must be in
# sync with the ``VERSION_MAP`` in ``gfx12-gluon-tutorials/kernels/gemm/
# a16w16/bench.py``.
_KERNEL_FIXTURES = [
    "v9_sliceM.amdgcn",
    "v10_double_local_prefetch.amdgcn",
]


@pytest.fixture(params=_KERNEL_FIXTURES, ids=lambda p: p.split('.')[0])
def kernel_asm(request):
    import glob
    import os
    cache_dir = os.path.expanduser("~/.triton/cache")
    matches = glob.glob(os.path.join(cache_dir, "*", request.param))
    if not matches:
        pytest.skip(f"{request.param} fixture not available")
    with open(matches[0]) as f:
        return request.param, f.read()


class TestKernelFixture:

    def test_region_spans_detected(self, kernel_asm):
        name, text = kernel_asm
        prog = parse_asm(text)
        spans = find_region_spans(prog)
        # Both v9 and v10 emit at least 8 loop regions plus epilogue regions.
        # Assert a lower bound so the test tolerates scheduler tweaks.
        assert len(spans) >= 8, f"{name}: only {len(spans)} regions"

    def test_merge_dscnt_reduces_wait_count(self, kernel_asm):
        name, text = kernel_asm
        prog = parse_asm(text)
        before = sum(1 for i in prog.iter_instructions()
                     if i.opcode == 's_wait_dscnt')
        merge_dscnt_waits(prog)
        after = sum(1 for i in prog.iter_instructions()
                    if i.opcode == 's_wait_dscnt')
        assert after < before, f"{name}: wait count did not decrease"
        # After merging there is at most one wait per region.
        spans = find_region_spans(prog)
        for span in spans:
            waits = [i for i in prog.blocks[0].instructions[span.start:span.end]
                     if i.opcode == 's_wait_dscnt']
            assert len(waits) <= 1, f"{name}: region has {len(waits)} waits"

    def test_barrier_pass_hoists_some(self, kernel_asm):
        name, text = kernel_asm
        prog = parse_asm(text)
        moved = overlap_wmma_with_barrier(prog)
        # The loop body has multiple barrier pairs with wmma following.
        assert moved > 0, f"{name}: barrier pass hoisted nothing"
