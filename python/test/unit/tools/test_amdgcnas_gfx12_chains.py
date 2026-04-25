"""Unit tests for Stage 2 of amdgcnas_gfx12: region annotation,
def-use index, and WMMA/DS chain collection."""
import textwrap

import pytest

from triton.tools.amdgcnas_gfx12 import (
    BankAssignment,
    DSChain,
    VGPRAllocation,
    WMMAChain,
    allocate_vgprs,
    annotate_regions,
    apply_allocation,
    assign_banks,
    build_def_use_index,
    can_share_data_vgprs,
    collect_ds_chains,
    collect_wmma_chains,
    emit_program,
    find_region_spans,
    hoist_loop_invariant_addrs,
    merge_dscnt_waits,
    overlap_wmma_with_barrier,
    parse_asm,
    report_chains,
)


MARKER_R0 = "\t;;#ASMSTART\n\t;; Region 0: 2 wmma, 0 GR, 2 LR\n\t;;#ASMEND"
MARKER_R1 = "\t;;#ASMSTART\n\t;; Region 1: 2 wmma, 0 GR, 2 LR\n\t;;#ASMEND"
MARKER_R2 = "\t;;#ASMSTART\n\t;; Region 2: 2 wmma, 0 GR, 2 LR\n\t;;#ASMEND"
MARKER_R3 = "\t;;#ASMSTART\n\t;; Region 3: 2 wmma, 0 GR, 2 LR\n\t;;#ASMEND"
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
    # Region markers are required for the new per-region DSChain
    # structure -- ds_loads outside any region are skipped.

    def test_ds_load_feeds_wmma_src0(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tds_load_b128 v[8:11], v100\n"
            "\tds_load_b128 v[12:15], v100 offset:16\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert len(chains) == 1
        chain = chains[0]
        assert chain.loading_region == 0
        assert chain.is_epilogue_region is False
        assert chain.op_idx == 1  # src0 slot
        # Two ds_loads form one contiguous tile (v[8:15]) -> one DSGroup.
        assert len(chain.dsgroups) == 1
        g = chain.dsgroups[0]
        assert len(g.ds_loads) == 2
        assert g.op_idx == 1
        assert g.tile.ids == list(range(8, 16))
        assert len(g.consumer_wmmas) == 1

    def test_ds_load_feeds_wmma_src1(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tds_load_b128 v[16:19], v100\n"
            "\tds_load_b128 v[20:23], v100 offset:16\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert len(chains) == 1
        assert chains[0].op_idx == 2  # src1 slot

    def test_two_tiles_in_one_region_form_one_chain(self):
        # Two tiles (src0 tile A, src1 tile B) load in the same region.
        # They would fail the op_idx sanity check because A feeds src0
        # but B feeds src1 -- they must be in different regions.
        # Test that feeding a single src slot with two separate tiles
        # works: both tiles are in the same DSChain with two DSGroups.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tds_load_b128 v[8:11], v100\n"
            "\tds_load_b128 v[12:15], v100 offset:16\n"
            "\tds_load_b128 v[24:27], v100 offset:32\n"
            "\tds_load_b128 v[28:31], v100 offset:48\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[24:31], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert len(chains) == 1
        c = chains[0]
        # Two tiles -> two DSGroups, both src0.
        assert len(c.dsgroups) == 2
        assert c.op_idx == 1
        assert all(g.op_idx == 1 for g in c.dsgroups)
        assert {g.tile.ids[0] for g in c.dsgroups} == {8, 24}

    def test_unused_ds_load_not_in_any_chain(self):
        # ds_load writes to a register no WMMA consumes -- the group gets
        # dropped, and with no remaining groups the chain is dropped too.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tds_load_b128 v[100:103], v50\n"  # unused by any wmma
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert chains == []

    def test_mixed_slot_raises(self):
        # Two ds_loads form one contiguous tile, but one wmma reads the
        # tile as src0 and another reads it as src1 -- sanity check
        # should fire.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tds_load_b128 v[8:11], v100\n"
            "\tds_load_b128 v[12:15], v100 offset:16\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[16:23], v[8:15], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        with pytest.raises(ValueError, match="mixed"):
            collect_ds_chains(prog)


# -------------------------------------------------------------------------
# Stage 4.2: DSChain lifetime analysis
# -------------------------------------------------------------------------

class TestDSChainLifetime:

    def test_lifetime_fields_populated(self):
        # Two regions, ds_load in R0 with consumer wmma in R1.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tds_load_b128 v[8:11], v100\n"
            "\tds_load_b128 v[12:15], v100 offset:16\n"
            f"{MARKER_R1}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        assert len(chains) == 1
        c = chains[0]
        assert c.loading_pos is not None
        assert c.last_consumer_pos is not None
        # Loading happens before consumer in program order.
        assert c.loading_pos < c.last_consumer_pos

    def test_can_share_disjoint_lifetimes(self):
        # Chain A: load early, consumer shortly after.
        # Chain B: load after A's consumer, consumer later.
        # A and B have disjoint lifetimes -> shareable.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tds_load_b128 v[8:11], v100\n"
            "\tds_load_b128 v[12:15], v100 offset:16\n"
            f"{MARKER_R1}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            f"{MARKER_R2}\n"
            "\tds_load_b128 v[24:27], v200\n"
            "\tds_load_b128 v[28:31], v200 offset:16\n"
            f"{MARKER_R3}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[24:31], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        a = next(c for c in chains if c.loading_region == 0)
        b = next(c for c in chains if c.loading_region == 2)
        assert can_share_data_vgprs(a, b)
        assert can_share_data_vgprs(b, a)  # symmetric

    def test_cannot_share_overlapping_lifetimes(self):
        # Chain A's consumer comes AFTER chain B's load -> overlap.
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tds_load_b128 v[8:11], v100\n"
            "\tds_load_b128 v[12:15], v100 offset:16\n"
            f"{MARKER_R1}\n"
            "\tds_load_b128 v[24:27], v200\n"
            "\tds_load_b128 v[28:31], v200 offset:16\n"
            f"{MARKER_R2}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            f"{MARKER_R3}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[24:31], v[16:23], v[0:7]\n"
            "\ts_cbranch_scc1 .LBB0_0\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        chains = collect_ds_chains(prog)
        a = next(c for c in chains if c.loading_region == 0)
        b = next(c for c in chains if c.loading_region == 1)
        # A loads at R0, consumer in R2. B loads at R1, consumer in R3.
        # A.last (R2) > B.load (R1) AND B.last (R3) > A.load (R0)
        # -> lifetimes overlap, cannot share.
        assert not can_share_data_vgprs(a, b)

    def test_v9_stride4_pairs_can_share(self):
        # Real v9 fixture: each stride-4 sibling pair should be
        # shareable.  L0/L4, L1/L5, L2/L6, L3/L7.
        import glob
        matches = glob.glob('/home/lixzhang/.triton/cache/*/v9_sliceM.amdgcn')
        if not matches:
            pytest.skip('v9 fixture not available')
        with open(matches[0]) as f:
            text = f.read()
        prog = parse_asm(text)
        chains = collect_ds_chains(prog)
        loop_chains_by_region = {
            c.loading_region: c for c in chains
            if not c.is_epilogue_region
        }
        for r in (0, 1, 2, 3):
            a = loop_chains_by_region.get(r)
            b = loop_chains_by_region.get(r + 4)
            assert a is not None and b is not None, f'L{r} or L{r+4} missing'
            assert can_share_data_vgprs(a, b), (
                f'L{r} (load_pos={a.loading_pos}, last={a.last_consumer_pos}) '
                f'and L{r+4} (load_pos={b.loading_pos}, last={b.last_consumer_pos}) '
                f'should share but cannot')


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

    def test_ds_chains_cover_every_region_ds_load(self, kernel_asm):
        # Under the per-region DSChain model, prologue ds_loads (those
        # not enclosed by a region marker) are intentionally skipped --
        # they'll be attached via dependency propagation in Stage 4.
        # But every ds_load inside a region marker should be in some
        # DSChain.
        name, text = kernel_asm
        prog = parse_asm(text)
        annotate_regions(prog)
        region_ds = [i for i in prog.iter_instructions()
                     if i.opcode.startswith('ds_load') and i.region_idx is not None]
        chains = collect_ds_chains(prog)
        covered = sum(len(c.ds_loads) for c in chains)
        assert covered == len(region_ds), (
            f"{name}: {covered}/{len(region_ds)} region-enclosed ds_loads mapped")

    def test_ds_chain_is_per_loading_region(self, kernel_asm):
        # All ds_loads in one DSChain should share loading_region +
        # is_epilogue_region.  This is the defining invariant of the
        # per-region model.
        name, text = kernel_asm
        prog = parse_asm(text)
        chains = collect_ds_chains(prog)
        for c in chains:
            for ld in c.ds_loads:
                assert ld.region_idx == c.loading_region, (
                    f"{name}: ds_load region {ld.region_idx} in chain "
                    f"L{c.loading_region}")
                assert bool(ld.region_is_epilogue) == c.is_epilogue_region

    def test_ds_chain_sanity_checks_hold(self, kernel_asm):
        # Every DSChain: shared addr, shared op_idx across its DSGroups.
        name, text = kernel_asm
        prog = parse_asm(text)
        chains = collect_ds_chains(prog)
        for c in chains:
            addrs = {g.addr_reg for g in c.dsgroups if g.addr_reg is not None}
            slots = {g.op_idx for g in c.dsgroups if g.op_idx is not None}
            assert len(addrs) <= 1, f"{name}: L{c.loading_region} mixed addr"
            assert len(slots) <= 1, f"{name}: L{c.loading_region} mixed op_idx"

    def test_prologue_loads_attached_to_loop_dsgroups(self, kernel_asm):
        # Every prologue ds_load (no region marker) should land in some
        # loop DSGroup's prologue_loads based on tile-match with a
        # steady-state sibling.  Unmatched ones would be a collection
        # bug.
        name, text = kernel_asm
        prog = parse_asm(text)
        chains = collect_ds_chains(prog)
        pre_loads_total = 0
        attached = 0
        for inst in prog.iter_instructions():
            if not inst.opcode.startswith('ds_load'):
                continue
            if inst.region_idx is not None:
                continue
            pre_loads_total += 1
            if inst.ds_chain is not None:
                attached += 1
        assert pre_loads_total > 0, f'{name}: no prologue ds_loads found'
        assert attached == pre_loads_total, (
            f'{name}: only {attached}/{pre_loads_total} prologue ds_loads '
            f'attached to DSGroups')

    def test_ds_chain_consumers_may_span_regions(self, kernel_asm):
        # Unlike my earlier sanity guess, per v9/v10 the consumers of a
        # single region's ds_loads DO span multiple regions (two
        # regions each, typically).  This is the back-edge / pipeline
        # reality.
        name, text = kernel_asm
        prog = parse_asm(text)
        chains = collect_ds_chains(prog)
        multi_region_chains = [
            c for c in chains
            if len({(w.region_is_epilogue, w.region_idx)
                    for w in c.consumer_wmmas
                    if w.region_idx is not None}) > 1
        ]
        assert multi_region_chains, (
            f"{name}: expected some chain's consumers to span regions")

    def test_report_chains_runs(self, kernel_asm):
        name, text = kernel_asm
        prog = parse_asm(text)
        annotate_regions(prog)
        report = report_chains(prog)
        assert "WMMA chains" in report

    def test_bank_assignment_has_no_conflicts(self, kernel_asm):
        # Stage 3's four-bank-per-4-region scheme should apply cleanly
        # to v9 and v10 -- they're the designed targets.
        name, text = kernel_asm
        prog = parse_asm(text)
        result = assign_banks(prog)
        assert result.conflicts == [], (
            f"{name}: Stage 3 reported conflicts: {result.conflicts[:3]}")

    def test_bank_assignment_four_distinct_acc_banks(self, kernel_asm):
        # The 4-region pipeline cycle should place acc_vgprs in 4
        # distinct banks (one per region mod 4).
        name, text = kernel_asm
        prog = parse_asm(text)
        result = assign_banks(prog)
        wcs = collect_wmma_chains(prog)
        assigned = [result.acc_bank(c) for c in wcs
                    if result.acc_bank(c) is not None]
        assert len(set(assigned)) == 4, f"{name}: acc banks = {set(assigned)}"

    def test_bank_assignment_region_msb_cycles_every_four(self, kernel_asm):
        # Region N and Region N+4 share the same WMMA chains (pipeline
        # cycle), so their MSB tuples must be identical.
        name, text = kernel_asm
        prog = parse_asm(text)
        result = assign_banks(prog)
        for r in (0, 1, 2, 3):
            m = result.region_msb.get(r)
            m4 = result.region_msb.get(r + 4)
            if m is None or m4 is None:
                continue
            assert m == m4, (
                f"{name}: L{r} MSB {m} != L{r + 4} MSB {m4}")


# -------------------------------------------------------------------------
# Stage 3 unit tests (synthetic)
# -------------------------------------------------------------------------

class TestAssignBanks:

    def _wrap_loop(self, body: str, prologue: str = "") -> str:
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
            "\ts_endpgm\n"
        )

    def test_no_loop_returns_empty(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        result = assign_banks(prog)
        assert result.wmma_acc_bank == {}
        assert result.region_msb == {}
        assert result.conflicts == []

    def test_two_regions_get_two_banks(self):
        body = (
            f"{MARKER_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            f"{MARKER_R1}\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
        )
        prog = parse_asm(self._wrap_loop(body))
        result = assign_banks(prog)
        wcs = collect_wmma_chains(prog)
        acc_banks = {result.acc_bank(c) for c in wcs if result.acc_bank(c) is not None}
        assert acc_banks == {0, 1}

    def test_chain_spanning_two_regions_keeps_first_bank(self):
        # Same chain (same dst) appears in both regions; should keep
        # Region 0's bank (0), not be reassigned in Region 1.
        body = (
            f"{MARKER_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            f"{MARKER_R1}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[32:39], v[40:47], v[0:7]\n"
        )
        prog = parse_asm(self._wrap_loop(body))
        result = assign_banks(prog)
        wcs = collect_wmma_chains(prog)
        assert len(wcs) == 1
        assert result.acc_bank(wcs[0]) == 0


# -------------------------------------------------------------------------
# Stage 4.3: bank-scoped VGPR allocator
# -------------------------------------------------------------------------

class TestAllocateVgprs:

    def _wrap_loop(self, body: str, prologue: str = "") -> str:
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
            "\ts_endpgm\n"
        )

    def test_no_loop_returns_empty(self):
        src = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            f"{MARKER_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            "\ts_endpgm\n"
        )
        prog = parse_asm(src)
        ba = assign_banks(prog)
        alloc = allocate_vgprs(prog, ba)
        assert alloc.wmma_acc == {}
        assert alloc.budget == 0

    def test_acc_lands_in_right_bank(self):
        # One wmma in each region -> one chain per region with
        # acc_bank = region_idx % 4 (Stage 3's rule).
        body = (
            f"{MARKER_R0}\n"
            "\tv_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[0:7]\n"
            f"{MARKER_R1}\n"
            "\tv_wmma_f32_16x16x32_f16 v[24:31], v[32:39], v[40:47], v[24:31]\n"
        )
        prog = parse_asm(self._wrap_loop(body))
        ba = assign_banks(prog)
        alloc = allocate_vgprs(prog, ba)
        wcs = collect_wmma_chains(prog)
        for c in wcs:
            new = alloc.acc(c)
            assert new is not None
            bank = ba.acc_bank(c)
            # New acc should land in the assigned bank (logical id /256).
            assert new.ids[0] // 256 == bank

    def test_v9_fixture_uses_expected_layout(self):
        import glob
        matches = glob.glob('/home/lixzhang/.triton/cache/*/v9_sliceM.amdgcn')
        if not matches:
            pytest.skip('v9 fixture not available')
        with open(matches[0]) as f:
            text = f.read()
        prog = parse_asm(text)
        ba = assign_banks(prog)
        alloc = allocate_vgprs(prog, ba)
        wcs = collect_wmma_chains(prog)
        dcs = collect_ds_chains(prog)

        # 4 banks of 16 wmma chains x 8 VGPRs = 128 acc VGPRs per bank.
        for bank in range(4):
            in_bank = [c for c in wcs if ba.acc_bank(c) == bank]
            assert len(in_bank) == 16, f'bank {bank}: {len(in_bank)} acc chains'
            for c in in_bank:
                new = alloc.acc(c)
                assert new is not None
                assert new.size == 8
                assert new.ids[0] >= bank * 256
                assert new.ids[-1] < (bank + 1) * 256

        # Stride-4 sibling pairs must share their data tile registers
        # (L0/L4, L1/L5, L2/L6, L3/L7 in v9).
        loop_chains = {c.loading_region: c for c in dcs
                       if not c.is_epilogue_region}
        for r in (0, 1, 2, 3):
            a = loop_chains[r]
            b = loop_chains[r + 4]
            # For each DSGroup of A, find a DSGroup of B with the same
            # canonical-order index and verify they map to the same
            # new Register.  We sort each chain's groups by tile.start
            # to mirror the allocator's canonical mapping.
            a_groups = sorted(a.dsgroups, key=lambda g: g.tile.ids[0])
            b_groups = sorted(b.dsgroups, key=lambda g: g.tile.ids[0])
            assert len(a_groups) == len(b_groups)
            for ga, gb in zip(a_groups, b_groups):
                ra = alloc.data(ga)
                rb = alloc.data(gb)
                assert ra is not None and rb is not None
                assert ra.ids == rb.ids, (
                    f'L{r}/L{r+4} group {ga.tile} <-> {gb.tile}: '
                    f'{ra} != {rb}')

        # Budget = highest used logical VGPR id + 1.  Bank-aligned
        # layout means bank 3's tail dictates the budget; expect on
        # the order of 768 + 192 (acc+data) ~= 960, well under the
        # 1024 hardware ceiling.
        assert alloc.budget < 1024, (
            f'v9 budget {alloc.budget} exceeds 1024 hardware limit')

    def test_v10_fixture_l1_l5_dont_share(self):
        # In v10, L1 (A_top_next) and L5 (A_top) are different tensors
        # in the same data_bank with overlapping lifetimes.  The
        # allocator should give them DIFFERENT VGPR pools.
        import glob
        matches = glob.glob(
            '/home/lixzhang/.triton/cache/*/v10_double_local_prefetch.amdgcn')
        if not matches:
            pytest.skip('v10 fixture not available')
        with open(matches[0]) as f:
            text = f.read()
        prog = parse_asm(text)
        ba = assign_banks(prog)
        alloc = allocate_vgprs(prog, ba)
        dcs = collect_ds_chains(prog)
        loop_chains = {c.loading_region: c for c in dcs
                       if not c.is_epilogue_region}
        l1 = loop_chains[1]
        l5 = loop_chains[5]
        # Same data_bank but different tile pools.
        assert ba.data_bank(l1) == ba.data_bank(l5)
        l1_starts = {alloc.data(g).ids[0] for g in l1.dsgroups
                     if alloc.data(g) is not None}
        l5_starts = {alloc.data(g).ids[0] for g in l5.dsgroups
                     if alloc.data(g) is not None}
        assert l1_starts.isdisjoint(l5_starts), (
            f'v10: L1 and L5 should NOT share VGPRs but overlap at '
            f'{l1_starts & l5_starts}')

    def test_addr_one_per_chain_in_addr_bank(self):
        import glob
        matches = glob.glob('/home/lixzhang/.triton/cache/*/v9_sliceM.amdgcn')
        if not matches:
            pytest.skip('v9 fixture not available')
        with open(matches[0]) as f:
            text = f.read()
        prog = parse_asm(text)
        ba = assign_banks(prog)
        alloc = allocate_vgprs(prog, ba)
        dcs = collect_ds_chains(prog)
        for c in dcs:
            if c.is_epilogue_region:
                continue
            ab = ba.addr_bank(c)
            if ab is None:
                continue
            new = alloc.addr(c)
            assert new is not None
            assert new.size == 1
            assert new.ids[0] // 256 == ab


# -------------------------------------------------------------------------
# apply_allocation (Stage 4.4+5)
# -------------------------------------------------------------------------

class TestApplyAllocation:

    def _run_pipeline(self, fixture_glob):
        import glob
        matches = glob.glob(fixture_glob)
        if not matches:
            pytest.skip(f'fixture not available: {fixture_glob}')
        with open(matches[0]) as f:
            text = f.read()
        prog = parse_asm(text)
        hoist_loop_invariant_addrs(prog)
        merge_dscnt_waits(prog)
        overlap_wmma_with_barrier(prog)
        ba = assign_banks(prog)
        alloc = allocate_vgprs(prog, ba)
        apply_allocation(prog, ba, alloc)
        return prog, ba, alloc

    def _count_acc_init_writes(self, prog):
        writes = set()
        for bb in prog.blocks:
            for inst in bb.instructions:
                if inst.opcode == 'v_mov_b32_e32' and len(inst.operands) >= 2:
                    src1 = inst.operands[1].text
                    if src1 == '0' or src1 == 'v64':
                        writes.add(inst.operands[0].regs[0].ids[0])
                elif inst.opcode == 'v_dual_mov_b32' and inst.dual_issue:
                    for slot in (0, 2):
                        if slot < len(inst.operands):
                            op = inst.operands[slot]
                            if op.regs and op.regs[0].kind == 'v':
                                writes.add(op.regs[0].ids[0])
        return writes

    def test_v9_apply_round_trips(self):
        prog, ba, alloc = self._run_pipeline(
            '/home/lixzhang/.triton/cache/*/v9_sliceM.amdgcn')
        # Re-emit must round-trip through the parser without errors.
        text = emit_program(prog)
        re_prog = parse_asm(text)
        assert len(re_prog.blocks) == len(prog.blocks)

    def test_v9_dual_mov_preserved(self):
        prog, ba, alloc = self._run_pipeline(
            '/home/lixzhang/.triton/cache/*/v9_sliceM.amdgcn')
        # Every v_dual_mov_b32 in the output must keep the :: separator
        # and have 4 parsed operands.
        for bb in prog.blocks:
            for inst in bb.instructions:
                if inst.opcode == 'v_dual_mov_b32':
                    assert inst.dual_issue is True
                    assert len(inst.operands) == 4
                    assert ' :: ' in inst.raw_line

    def test_v9_acc_init_covers_all_chains(self):
        prog, ba, alloc = self._run_pipeline(
            '/home/lixzhang/.triton/cache/*/v9_sliceM.amdgcn')
        writes = self._count_acc_init_writes(prog)
        expected = set()
        for c in collect_wmma_chains(prog):
            new = alloc.acc(c)
            if new is not None:
                expected.update(new.ids)
        # Every allocated acc element should be initialized.
        assert expected.issubset(writes), (
            f'missing inits for {sorted(expected - writes)[:10]}')

    def test_v9_budget_bumped(self):
        prog, ba, alloc = self._run_pipeline(
            '/home/lixzhang/.triton/cache/*/v9_sliceM.amdgcn')
        text = emit_program(prog)
        # The .amdhsa_next_free_vgpr directive should reflect alloc.budget.
        assert f'.amdhsa_next_free_vgpr {alloc.budget}' in text

    def test_v10_apply_round_trips(self):
        prog, ba, alloc = self._run_pipeline(
            '/home/lixzhang/.triton/cache/*/v10_double_local_prefetch.amdgcn')
        text = emit_program(prog)
        re_prog = parse_asm(text)
        assert len(re_prog.blocks) == len(prog.blocks)
