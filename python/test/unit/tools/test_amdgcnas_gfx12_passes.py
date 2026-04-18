"""Unit tests for the amdgcnas_gfx12 peephole passes."""
import textwrap

import pytest

from triton.tools.amdgcnas_gfx12 import (
    RegionMarker,
    emit_program,
    find_region_spans,
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
