"""Unit tests for ``amdgcnas_gfx12``'s parser and round-trip emitter."""
import textwrap

import pytest

from triton.tools.amdgcnas_gfx12 import (
    BasicBlock,
    Instruction,
    Operand,
    Program,
    Register,
    RegionMarker,
    SubRegionMarker,
    _decode_msb,
    _decode_msb_imm,
    _parse_operand,
    _parse_instruction_line,
    emit_program,
    parse_asm,
)


# -------------------------------------------------------------------------
# Operand-level tests
# -------------------------------------------------------------------------

class TestParseOperand:

    def test_vgpr_range_with_comment(self):
        op = _parse_operand("v[0:7] /*v[512:519]*/")
        assert len(op.regs) == 1
        r = op.regs[0]
        assert r.kind == 'v'
        assert r.ids == list(range(512, 520))
        assert r.raw_ids == list(range(0, 8))
        assert r.msb() == 2
        assert op.logical_text == "/*v[512:519]*/"

    def test_vgpr_range_no_comment(self):
        op = _parse_operand("v[0:7]")
        r = op.regs[0]
        assert r.ids == list(range(0, 8))
        assert r.raw_ids == list(range(0, 8))
        assert r.msb() == 0
        assert op.logical_text is None

    def test_vgpr_single_with_comment(self):
        op = _parse_operand("v194 /*v706*/")
        r = op.regs[0]
        assert r.kind == 'v'
        assert r.ids == [706]
        assert r.raw_ids == [194]
        assert r.msb() == 2
        assert op.logical_text == "/*v706*/"

    def test_vgpr_single_no_comment(self):
        op = _parse_operand("v42")
        r = op.regs[0]
        assert r.ids == [42]
        assert r.msb() == 0

    def test_sgpr_range(self):
        op = _parse_operand("s[16:19]")
        r = op.regs[0]
        assert r.kind == 's'
        assert r.ids == [16, 17, 18, 19]

    def test_sgpr_single(self):
        op = _parse_operand("s4")
        r = op.regs[0]
        assert r.kind == 's' and r.ids == [4]

    def test_literal_no_register(self):
        op = _parse_operand("0x80")
        assert op.regs == []
        assert op.text == "0x80"

    def test_offset_suffix_preserved(self):
        op = _parse_operand("v194 /*v706*/ offset:32")
        assert op.regs[0].ids == [706]
        # The suffix must survive re-emission.
        emitted = op.emit()
        assert "offset:32" in emitted
        assert "/*v706*/" in emitted


# -------------------------------------------------------------------------
# MSB immediate decoding
# -------------------------------------------------------------------------

class TestDecodeMSB:

    def test_imm_dst_only(self):
        # 0x80: dst=2, src0=src1=src2=0
        assert _decode_msb_imm("0x80") == (2, 0, 0, 0)

    def test_imm_all_fields(self):
        # 0x5a: dst=1, src0=2, src1=2, src2=1
        assert _decode_msb_imm("0x5a") == (1, 2, 2, 1)

    def test_imm_with_high_bit(self):
        # 0x8008: dst=0, src0=0, src1=2, src2=0
        assert _decode_msb_imm("0x8008") == (0, 0, 2, 0)

    def test_imm_zero(self):
        assert _decode_msb_imm("0x0") == (0, 0, 0, 0)

    def test_prefers_comment(self):
        # Comment disagrees with imm; we trust the comment.
        assert _decode_msb("0x0", " msbs: dst=1 src0=2 src1=3 src2=0") == (1, 2, 3, 0)


# -------------------------------------------------------------------------
# Instruction-line parsing
# -------------------------------------------------------------------------

class TestParseInstruction:

    def test_wmma(self):
        line = ("\tv_wmma_f32_16x16x32_f16 v[192:199] /*v[448:455]*/, "
                "v[80:87] /*v[592:599]*/, v[56:63] /*v[568:575]*/, "
                "v[192:199] /*v[448:455]*/")
        inst = _parse_instruction_line(line)
        assert inst.opcode == "v_wmma_f32_16x16x32_f16"
        assert len(inst.operands) == 4
        assert inst.dst_reg().ids == list(range(448, 456))
        assert inst.dst_reg().msb() == 1
        srcs = inst.src_regs()
        assert srcs[0].msb() == 2  # src0
        assert srcs[1].msb() == 2  # src1
        assert srcs[2].msb() == 1  # src2 (acc)

    def test_ds_load_b128(self):
        line = "\tds_load_b128 v[120:123] /*v[632:635]*/, v194 /*v706*/"
        inst = _parse_instruction_line(line)
        assert inst.opcode == "ds_load_b128"
        assert inst.dst_reg().ids == list(range(632, 636))
        addr = inst.src_regs()[0]
        assert addr.ids == [706]

    def test_ds_load_with_offset(self):
        line = "\tds_load_b128 v[124:127] /*v[636:639]*/, v194 /*v706*/ offset:32"
        inst = _parse_instruction_line(line)
        assert len(inst.operands) == 2
        assert inst.operands[1].suffix == "offset:32"

    def test_tensor_load_to_lds(self):
        line = "\ttensor_load_to_lds s[16:19], s[0:7]"
        inst = _parse_instruction_line(line)
        assert inst.opcode == "tensor_load_to_lds"
        assert inst.dst_reg().kind == 's'
        assert inst.dst_reg().ids == [16, 17, 18, 19]
        assert inst.src_regs()[0].ids == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_set_vgpr_msb(self):
        line = "\ts_set_vgpr_msb 0x5a                     ;  msbs: dst=1 src0=2 src1=2 src2=1"
        inst = _parse_instruction_line(line)
        assert inst.opcode == "s_set_vgpr_msb"
        assert inst.msb_bits == (1, 2, 2, 1)
        assert inst.trailing_comment is not None

    def test_s_wait_dscnt(self):
        line = "\ts_wait_dscnt 0x6"
        inst = _parse_instruction_line(line)
        assert inst.opcode == "s_wait_dscnt"

    def test_s_wait_tensorcnt(self):
        line = "\ts_wait_tensorcnt 0x6"
        inst = _parse_instruction_line(line)
        assert inst.opcode == "s_wait_tensorcnt"

    def test_s_add_nc_u64(self):
        line = "\ts_add_nc_u64 s[2:3], s[46:47], 0x80"
        inst = _parse_instruction_line(line)
        assert inst.opcode == "s_add_nc_u64"
        assert len(inst.operands) == 3
        assert inst.operands[2].text == "0x80"

    def test_trailing_comment_preserved(self):
        line = "\tv_mov_b32 v0, 0x1 ; initialise acc"
        inst = _parse_instruction_line(line)
        assert inst.trailing_comment == " initialise acc"

    def test_directive_line(self):
        line = "\t.loc\t1 42 9"
        inst = _parse_instruction_line(line)
        assert inst.opcode == ".loc"

    def test_comment_inside_block_comment_not_split(self):
        # The trailing-comment scanner must not treat a ';' inside /*...*/
        # as the start of a comment (there isn't one in the current
        # assembly, but we still want to guard the invariant).
        line = "\tv_mov_b32 v0 /*v256*/, 0x1"
        inst = _parse_instruction_line(line)
        assert inst.trailing_comment is None
        assert inst.dst_reg().ids == [256]


# -------------------------------------------------------------------------
# Inline asm block (scheduler markers)
# -------------------------------------------------------------------------

_ASM_BLOCK_REGION = textwrap.dedent("""\
\t;;#ASMSTART
\t; region 0: wmma=32 ds_load=16 tdm=0
\t;;#ASMEND
""")

_ASM_BLOCK_REGION_PROD = textwrap.dedent("""\
\t;;#ASMSTART
\t;; Region 3: 32 wmma, 1 GR, 16 LR
\t;;#ASMEND
""")

_ASM_BLOCK_REGION_EPILOGUE = textwrap.dedent("""\
\t;;#ASMSTART
\t;; Epilogue Region 3: 32 wmma, 0 GR, 16 LR, 0 LW, 0 CVT
\t;;#ASMEND
""")

_ASM_BLOCK_SUBREGION = textwrap.dedent("""\
\t;;#ASMSTART
\t; sub-region 2: wmma=8 ds_load=4 tdm=1
\t;;#ASMEND
""")


class TestParseAsmBlocks:

    def test_region_marker_parsed(self):
        # Wrap in a minimal program with a basic block.
        text = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            + _ASM_BLOCK_REGION
            + "\ts_endpgm\n"
        )
        prog = parse_asm(text)
        insts = list(prog.iter_instructions())
        markers = [i.region_marker for i in insts if i.region_marker]
        assert len(markers) == 1
        assert markers[0] == RegionMarker(region=0, wmma=32, ds_load=16, tdm=0,
                                          is_epilogue=False)

    def test_region_marker_prod_format(self):
        text = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            + _ASM_BLOCK_REGION_PROD
            + "\ts_endpgm\n"
        )
        prog = parse_asm(text)
        markers = [i.region_marker for i in prog.iter_instructions() if i.region_marker]
        assert len(markers) == 1
        assert markers[0].region == 3
        assert markers[0].wmma == 32
        assert markers[0].is_epilogue is False

    def test_epilogue_region_marker_parsed(self):
        text = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            + _ASM_BLOCK_REGION_EPILOGUE
            + "\ts_endpgm\n"
        )
        prog = parse_asm(text)
        markers = [i.region_marker for i in prog.iter_instructions() if i.region_marker]
        assert len(markers) == 1
        assert markers[0].region == 3
        assert markers[0].is_epilogue is True

    def test_subregion_marker_parsed(self):
        text = (
            "; %bb.0:\n"
            ".LBB0_0:\n"
            + _ASM_BLOCK_SUBREGION
            + "\ts_endpgm\n"
        )
        prog = parse_asm(text)
        markers = [i.subregion_marker for i in prog.iter_instructions()
                   if i.subregion_marker]
        assert len(markers) == 1
        assert markers[0] == SubRegionMarker(sub_region=2, wmma=8, ds_load=4, tdm=1)


# -------------------------------------------------------------------------
# Top-level parse / round-trip
# -------------------------------------------------------------------------

_MINIMAL_PROGRAM = textwrap.dedent("""\
\t.text
\t.globl\ttest_kernel
\t.p2align\t8
\t.type\ttest_kernel,@function
test_kernel:
; %bb.0:
.LBB0_0:
\ts_set_vgpr_msb 0x5a                     ;  msbs: dst=1 src0=2 src1=2 src2=1
\tv_wmma_f32_16x16x32_f16 v[192:199] /*v[448:455]*/, v[80:87] /*v[592:599]*/, v[56:63] /*v[568:575]*/, v[192:199] /*v[448:455]*/
\tds_load_b128 v[120:123] /*v[632:635]*/, v194 /*v706*/
\t;;#ASMSTART
\t; region 0: wmma=32 ds_load=16 tdm=0
\t;;#ASMEND
\ts_endpgm
""")


class TestRoundTrip:

    def test_minimal_program_round_trips(self):
        prog = parse_asm(_MINIMAL_PROGRAM)
        out = emit_program(prog)
        assert out == _MINIMAL_PROGRAM

    def test_block_structure(self):
        prog = parse_asm(_MINIMAL_PROGRAM)
        assert len(prog.blocks) == 1
        bb = prog.blocks[0]
        assert bb.name == ".LBB0_0"
        # Instructions: s_set_vgpr_msb, v_wmma, ds_load, asm block, s_endpgm
        real = [i for i in bb.instructions if i.opcode]
        opcodes = [i.opcode for i in real]
        assert "s_set_vgpr_msb" in opcodes
        assert "v_wmma_f32_16x16x32_f16" in opcodes
        assert "ds_load_b128" in opcodes
        assert "__asm_block__" in opcodes
        assert "s_endpgm" in opcodes

    def test_kernel_fixture_round_trips(self, kernel_asm):
        name, text = kernel_asm
        prog = parse_asm(text)
        out = emit_program(prog)
        assert out == text, f"round-trip mismatch for {name}"


# Fixture file names for each kernel we exercise end-to-end.  Must be in
# sync with the ``VERSION_MAP`` in ``gfx12-gluon-tutorials/kernels/gemm/
# a16w16/bench.py``.
_KERNEL_FIXTURES = [
    "v9_sliceM.amdgcn",
    "v10_double_local_prefetch.amdgcn",
]


def _load_kernel_asm(kernel_file):
    import glob
    import os
    cache_dir = os.path.expanduser("~/.triton/cache")
    if not os.path.isdir(cache_dir):
        return None
    matches = glob.glob(os.path.join(cache_dir, "*", kernel_file))
    if not matches:
        return None
    with open(matches[0]) as f:
        return f.read()


@pytest.fixture(params=_KERNEL_FIXTURES, ids=lambda p: p.split('.')[0])
def kernel_asm(request):
    """Load a cached kernel assembly produced by the matching benchmark.

    Users generate the cache by running e.g.::

        TRITON_ENABLE_LLIR_SCHED=1 python bench.py --version 9 --K 1024 \\
            --dtype fp16

    Useful during development for realistic round-trip tests; individual
    tests skip when the fixture file is absent so this doesn't block CI
    on hosts that don't run the gluon kernels.
    """
    text = _load_kernel_asm(request.param)
    if text is None:
        pytest.skip(f"{request.param} fixture not available")
    return request.param, text
