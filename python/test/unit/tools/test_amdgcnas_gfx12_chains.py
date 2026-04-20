"""Unit tests for Stage 2 of amdgcnas_gfx12: region annotation,
def-use index, and WMMA/DS chain collection."""
import textwrap

import pytest

from triton.tools.amdgcnas_gfx12 import (
    DSChain,
    WMMAChain,
    annotate_regions,
    build_def_use_index,
    collect_ds_chains,
    collect_wmma_chains,
    emit_program,
    find_region_spans,
    parse_asm,
    report_chains,
)


MARKER_R0 = "\t;;#ASMSTART\n\t;; Region 0: 2 wmma, 0 GR, 2 LR\n\t;;#ASMEND"
MARKER_R1 = "\t;;#ASMSTART\n\t;; Region 1: 2 wmma, 0 GR, 2 LR\n\t;;#ASMEND"
MARKER_EPI_R0 = ("\t;;#ASMSTART\n\t;; Epilogue Region 0: 2 wmma, 0 GR, 2 LR\n"
                 "\t;;#ASMEND")


# -------------------------------------------------------------------------
# Region annotation
# -------------------------------------------------------------------------

class TestAnnotateRegions:

    def test_instructions_tagged_with_region_idx(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            f"{MARKER_R1}\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        annotate_regions(prog)
        insts = [i for i in prog.iter_instructions() if i.opcode]
        # Find the two wmmas and confirm their region tags.
        wmmas = [i for i in insts if i.opcode.startswith('v_wmma')]
        assert len(wmmas) == 2
        assert wmmas[0].region_idx == 0
        assert wmmas[1].region_idx == 1

    def test_pre_marker_instructions_have_none(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            "\tv_mov_b32 v0, 0\n"   # no region marker yet
            f"{MARKER_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        annotate_regions(prog)
        insts = [i for i in prog.iter_instructions() if i.opcode]
        mov = next(i for i in insts if i.opcode == 'v_mov_b32')
        wmma = next(i for i in insts if i.opcode.startswith('v_wmma'))
        assert mov.region_idx is None
        assert mov.region_is_epilogue is None
        assert wmma.region_idx == 0
        assert wmma.region_is_epilogue is False

    def test_loop_and_epilogue_regions_distinguished(self):
        # Loop Region 0 and Epilogue Region 0 share the same integer
        # index but must be distinguishable via region_is_epilogue.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            f"{MARKER_EPI_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        annotate_regions(prog)
        wmmas = [i for i in prog.iter_instructions()
                 if i.opcode.startswith('v_wmma')]
        assert len(wmmas) == 2
        assert wmmas[0].region_idx == 0
        assert wmmas[0].region_is_epilogue is False
        assert wmmas[1].region_idx == 0
        assert wmmas[1].region_is_epilogue is True


# -------------------------------------------------------------------------
# Def-use index
# -------------------------------------------------------------------------

class TestDefUseIndex:

    def _parse_single_bb(self, body: str):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{body}"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        return prog.blocks[0]

    def test_def_and_use_recorded(self):
        bb = self._parse_single_bb(
            "\tv_mov_b32 v0, 1\n"
            "\tv_add_f32 v1, v0, v0\n"
        )
        idx = build_def_use_index(bb)
        mov = bb.instructions[1] if not bb.instructions[0].opcode else bb.instructions[0]
        # Find by opcode instead of relying on indices (parse may insert
        # empty/label lines).
        mov = next(i for i in bb.instructions if i.opcode == 'v_mov_b32')
        add = next(i for i in bb.instructions if i.opcode == 'v_add_f32')
        assert mov in idx.defs[('v', 0)]
        assert add in idx.defs[('v', 1)]
        assert add in idx.uses[('v', 0)]

    def test_last_def_before(self):
        bb = self._parse_single_bb(
            "\tv_mov_b32 v0, 1\n"
            "\tv_add_f32 v1, v0, v0\n"
            "\tv_mov_b32 v0, 2\n"
        )
        idx = build_def_use_index(bb)
        mov1 = [i for i in bb.instructions if i.opcode == 'v_mov_b32'][0]
        mov2 = [i for i in bb.instructions if i.opcode == 'v_mov_b32'][1]
        add = next(i for i in bb.instructions if i.opcode == 'v_add_f32')
        # The def of v0 reaching the add should be mov1, not mov2.
        assert idx.last_def_before(('v', 0), add) is mov1
        # Nothing defines v0 before mov1 itself.
        assert idx.last_def_before(('v', 0), mov1) is None


# -------------------------------------------------------------------------
# WMMA chain collection
# -------------------------------------------------------------------------

class TestCollectWMMAChains:

    def test_single_wmma_single_chain(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_wmma_chains(prog)
        assert len(chains) == 1
        assert chains[0].size == 1
        assert chains[0].canonical.ids == list(range(0, 8))

    def test_four_wmmas_one_chain(self):
        # All four WMMAs accumulate into v[0:7].
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[24:31], v[32:39], v[0:7]\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[40:47], v[48:55], v[0:7]\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[56:63], v[64:71], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_wmma_chains(prog)
        assert len(chains) == 1
        assert chains[0].size == 4
        # Each WMMA is linked back to its chain.
        for w in chains[0].wmmas:
            assert w.wmma_chain is chains[0]

    def test_two_distinct_accumulators(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[48:55], v[56:63], v[0:7]\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[64:71], v[72:79], v[24:31]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_wmma_chains(prog)
        assert len(chains) == 2
        sizes = sorted(c.size for c in chains)
        assert sizes == [2, 2]


# -------------------------------------------------------------------------
# DS chain collection
# -------------------------------------------------------------------------

class TestCollectDSChains:

    def test_ds_load_feeds_wmma_src0(self):
        # ds_load writes v[8:15]; wmma reads it as src0 (first src operand).
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            "\tds_load_b128 v[8:11], v100\n"
            "\tds_load_b128 v[12:15], v100 offset:16\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert len(chains) == 1
        chain = chains[0]
        assert len(chain.ds_loads) == 2
        assert chain.operand_idx == 1  # src0 slot
        assert len(chain.consumers) == 1
        assert chain.wmma_chain.canonical.ids == list(range(0, 8))

    def test_ds_load_feeds_wmma_src1(self):
        # Same wmma; ds_load targets the second src (src1) slot.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            "\tds_load_b128 v[16:19], v100\n"
            "\tds_load_b128 v[20:23], v100 offset:16\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert len(chains) == 1
        assert chains[0].operand_idx == 2  # src1 slot

    def test_two_tiles_two_chains(self):
        # Tile A feeds src0; tile B feeds src1.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            "\tds_load_b128 v[8:11], v100\n"         # tile A part 1
            "\tds_load_b128 v[12:15], v100 offset:16\n"  # tile A part 2
            "\tds_load_b128 v[16:19], v101\n"         # tile B part 1
            "\tds_load_b128 v[20:23], v101 offset:16\n"  # tile B part 2
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert len(chains) == 2
        slots = sorted(c.operand_idx for c in chains)
        assert slots == [1, 2]

    def test_unused_ds_load_not_in_any_chain(self):
        # ds_load writes to a register no WMMA consumes (e.g., for prologue
        # address setup).  It should be dropped from the chain list.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            "\tds_load_b128 v[100:103], v50\n"  # unused by any wmma
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert chains == []


# -------------------------------------------------------------------------
# Kernel fixture tests (v9 + v10)
# -------------------------------------------------------------------------

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

    def test_region_annotation_covers_loop(self, kernel_asm):
        name, text = kernel_asm
        prog = parse_asm(text)
        annotate_regions(prog)
        # At least the WMMAs inside the loop should be tagged with a
        # region index (v9/v10 emit 8 loop regions).
        wmmas_in_region = [
            i for i in prog.iter_instructions()
            if i.opcode.startswith('v_wmma') and i.region_idx is not None
        ]
        assert len(wmmas_in_region) > 0, f"{name}: no WMMAs tagged"

    def test_wmma_chains_match_accumulator_count(self, kernel_asm):
        name, text = kernel_asm
        prog = parse_asm(text)
        chains = collect_wmma_chains(prog)
        # v9 and v10 both use 4 accumulator tensors (C_tl, C_bl, C_tr,
        # C_br), each 128 VGPRs wide.  Depending on how the sliced
        # epilogue writes back, a few extra small chains may appear, but
        # we expect at least 4 WMMA chains in the full program.
        assert len(chains) >= 4, (
            f"{name}: only {len(chains)} WMMA chains")

    def test_ds_chains_cover_all_ds_loads(self, kernel_asm):
        # Every ds_load in the kernel should feed some WMMA (either in
        # the same BB or across the prologue→loop boundary).  If this
        # ever fails, a tensor has been missed by the cross-BB consumer
        # search.
        name, text = kernel_asm
        prog = parse_asm(text)
        all_ds = [i for i in prog.iter_instructions()
                  if i.opcode.startswith('ds_load')]
        chains = collect_ds_chains(prog)
        covered = sum(len(c.ds_loads) for c in chains)
        assert covered == len(all_ds), (
            f"{name}: {covered}/{len(all_ds)} ds_loads mapped")

    def test_ds_chains_span_prologue_and_loop(self, kernel_asm):
        # The prologue prefetches first-iteration tiles.  Their ds_loads
        # should be grouped into the same DSChain as the subsequent loop
        # ds_loads that refill the same registers.
        name, text = kernel_asm
        prog = parse_asm(text)
        chains = collect_ds_chains(prog)
        cross_bb = [
            c for c in chains
            if len({l.parent_bb.name for l in c.ds_loads}) > 1
        ]
        assert cross_bb, f"{name}: no DS chain spans prologue and loop"

    def test_report_chains_runs(self, kernel_asm):
        name, text = kernel_asm
        prog = parse_asm(text)
        annotate_regions(prog)
        report = report_chains(prog)
        assert "WMMA chains" in report
        assert "DS chains" in report
