"""
gfx1250 (MI450) assembly post-processor.

Rewrites VGPR assignments in the generated AMDGCN assembly to minimize
``s_set_vgpr_msb`` switches by placing each logical tensor in the bank
designated by the kernel's sub-region structure.

Stage 1 (this file): parser and round-trip emitter. Subsequent stages
(chain collection, bank allocation, VGPR renaming, MSB regeneration)
will be added incrementally.

Compared to the CDNA3 (mi350) post-processor ``amdgcnas.py``:
  * VGPR operands carry a logical-id comment when the MSB is non-zero:
    ``v[0:7] /*v[512:519]*/`` means raw v[0:7] with DST/SRC MSB=2.
    Single form: ``v194 /*v706*/``.  The comment is the truth.
  * New opcodes: ``ds_load_*``, ``ds_load_tr16_*``, ``tensor_load_to_lds``,
    ``tensor_store_from_lds``, ``s_set_vgpr_msb``, ``s_wait_dscnt``,
    ``s_wait_tensorcnt``, ``v_wmma_*``, ``s_barrier_signal``,
    ``s_barrier_wait``.
  * Scheduler sub-region markers are embedded as inline asm comments:
      ``;;#ASMSTART`` / ``; region N: wmma=X ds_load=Y tdm=Z`` / ``;;#ASMEND``
      ``;;#ASMSTART`` / ``; sub-region N: wmma=X ds_load=Y tdm=Z`` / ``;;#ASMEND``
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional


# -------------------------------------------------------------------------
# Register
# -------------------------------------------------------------------------
#
# For VGPRs, we store the LOGICAL id (true 10-bit VGPR number 0..1023)
# as the ground truth, and remember the original raw id (8-bit field in
# the instruction encoding) so we can round-trip the unchanged assembly.
#
# For SGPRs/other kinds, raw == logical.

@dataclass(eq=False)
class Register:
    kind: str                # 'v', 's', 'a', 'm'
    ids: list[int]           # logical ids, sorted contiguous
    raw_ids: list[int] = field(default_factory=list)  # original 8-bit field

    def __post_init__(self):
        if len(self.ids) > 1:
            expected = list(range(self.ids[0], self.ids[0] + len(self.ids)))
            if self.ids != expected:
                raise ValueError(f"Non-contiguous register ids: {self.ids}")
        if not self.raw_ids:
            self.raw_ids = list(self.ids)

    @property
    def start(self) -> int:
        return self.ids[0]

    @property
    def end(self) -> int:
        return self.ids[-1]

    @property
    def size(self) -> int:
        return len(self.ids)

    def is_single(self) -> bool:
        return self.size == 1

    def overlaps(self, other: "Register") -> bool:
        if self.kind != other.kind:
            return False
        return not (self.end < other.start or other.end < self.start)

    def contains(self, other: "Register") -> bool:
        if self.kind != other.kind:
            return False
        return self.start <= other.start and self.end >= other.end

    def msb(self) -> int:
        """MSB value (0..3) implied by the first logical id for a VGPR."""
        if self.kind != 'v':
            return 0
        return self.ids[0] // 256

    def __hash__(self) -> int:
        return hash((self.kind, tuple(self.ids)))

    def __eq__(self, other) -> bool:
        return isinstance(other, Register) and self.kind == other.kind and self.ids == other.ids

    def __repr__(self) -> str:
        if len(self.ids) == 1:
            return f"{self.kind}{self.ids[0]}"
        return f"{self.kind}[{self.ids[0]}:{self.ids[-1]}]"


# -------------------------------------------------------------------------
# Operand
# -------------------------------------------------------------------------

@dataclass
class Operand:
    """A single instruction operand.

    ``text`` is just the register part as it appeared in the source
    (e.g., ``v134`` or ``v[230:233]``).  ``regs`` is the list of
    registers mentioned in the operand (usually 0 or 1).
    ``logical_text`` is the ``/*v[...]*/`` annotation if present, else
    ``None``.  ``suffix`` is any post-comment modifier such as
    ``offset:32``.

    Emit re-assembles them in the canonical AMDGCN order:
    ``<reg> <logical_text> <suffix>`` so that subsequent re-parses (and
    the assembler) see the same shape as the LLVM-emitted source.
    """
    text: str
    regs: list[Register]
    logical_text: Optional[str] = None
    suffix: Optional[str] = None

    def emit(self) -> str:
        parts: list[str] = [self.text]
        if self.logical_text is not None:
            parts.append(self.logical_text)
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)


# -------------------------------------------------------------------------
# Markers (scheduler inline asm comments)
# -------------------------------------------------------------------------

@dataclass
class RegionMarker:
    region: int
    wmma: int
    ds_load: int
    tdm: int
    # ``;; Region N`` vs ``;; Epilogue Region N``.  The AMDGPU backend
    # restarts region numbering in the epilogue, so the ``region`` int
    # alone doesn't uniquely identify a marker.
    is_epilogue: bool = False


@dataclass
class SubRegionMarker:
    sub_region: int
    wmma: int
    ds_load: int
    tdm: int


# -------------------------------------------------------------------------
# Instruction
# -------------------------------------------------------------------------

@dataclass
class Instruction:
    opcode: str
    operands: list[Operand]
    raw_line: str                     # original source line, whitespace preserved
    trailing_comment: Optional[str] = None  # everything after ';' (excluding the ';')

    parent_bb: Optional["BasicBlock"] = None
    index: int = -1                  # position within parent BB

    # Scheduler markers emitted by the LLIR pass (one Instruction holds the
    # whole ``;;#ASMSTART ... ;;#ASMEND`` block; this field is set when the
    # block carries a region/sub-region marker).
    region_marker: Optional[RegionMarker] = None
    subregion_marker: Optional[SubRegionMarker] = None

    # MSB bits implied by an ``s_set_vgpr_msb`` instruction, decoded into
    # four 2-bit fields {dst, src0, src1, src2}.  None for other opcodes.
    msb_bits: Optional[tuple[int, int, int, int]] = None

    # Stage 2 analysis tags.  Populated by ``annotate_regions`` and chain
    # collectors; None until those passes run.
    region_idx: Optional[int] = None      # enclosing Region marker index
    region_is_epilogue: Optional[bool] = None  # True for "Epilogue Region N"
    sub_region_idx: Optional[int] = None  # enclosing SubRegion marker index
    wmma_chain: Optional["WMMAChain"] = None
    ds_chain: Optional["DSChain"] = None

    # GFX1250 dual-issue: ``v_dual_mov_b32 a, b :: v_dual_mov_b32 c, d``
    # parses as a single instruction with 4 operands; emit re-inserts the
    # ``::`` separator.
    dual_issue: bool = False

    def emit(self) -> str:
        # Preserve original line verbatim when the instruction has no
        # structured form (labels, directives, inline asm blocks).
        if self.opcode.startswith('.') or self.opcode == '__asm_block__':
            return self.raw_line
        if self.dual_issue and len(self.operands) == 4:
            left = ", ".join(op.emit() for op in self.operands[:2])
            right = ", ".join(op.emit() for op in self.operands[2:])
            line = f"{self.opcode} {left} :: {self.opcode} {right}"
        else:
            parts = [self.opcode]
            if self.operands:
                parts.append(", ".join(op.emit() for op in self.operands))
            line = " ".join(parts)
        if self.trailing_comment is not None:
            line = f"{line:<40} ;{self.trailing_comment}"
        return line

    # Convenience getters
    def dst_reg(self) -> Optional[Register]:
        if self.operands and self.operands[0].regs:
            return self.operands[0].regs[0]
        return None

    def src_regs(self) -> list[Register]:
        out: list[Register] = []
        for op in self.operands[1:]:
            out.extend(op.regs)
        return out

    def all_regs(self) -> list[Register]:
        out: list[Register] = []
        for op in self.operands:
            out.extend(op.regs)
        return out


# -------------------------------------------------------------------------
# BasicBlock / Program
# -------------------------------------------------------------------------

@dataclass
class BasicBlock:
    name: str
    label_line: Optional[str] = None   # full label line (may have trailing comment)
    instructions: list[Instruction] = field(default_factory=list)

    def add_inst(self, inst: Instruction) -> None:
        inst.parent_bb = self
        inst.index = len(self.instructions)
        self.instructions.append(inst)

    def emit(self) -> list[str]:
        lines: list[str] = []
        if self.label_line is not None:
            lines.append(self.label_line)
        elif self.name:
            lines.append(f"{self.name}:")
        for inst in self.instructions:
            lines.append(inst.raw_line)
        return lines


@dataclass
class Program:
    header_lines: list[str] = field(default_factory=list)   # everything before the first BB
    blocks: list[BasicBlock] = field(default_factory=list)
    tail_lines: list[str] = field(default_factory=list)     # everything after s_endpgm

    def iter_instructions(self) -> Iterator[Instruction]:
        for bb in self.blocks:
            yield from bb.instructions

    def emit(self) -> str:
        lines: list[str] = list(self.header_lines)
        for bb in self.blocks:
            lines.extend(bb.emit())
        lines.extend(self.tail_lines)
        return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------
# Lexer-level regexes for operands
# -------------------------------------------------------------------------

# VGPR range with logical-id comment:  v[192:199] /*v[448:455]*/
_VGPR_RANGE_COMMENT = re.compile(
    r'v\[(\d+):(\d+)\]\s*/\*v\[(\d+):(\d+)\]\*/'
)
# VGPR range without comment:          v[192:199]
_VGPR_RANGE = re.compile(r'v\[(\d+):(\d+)\]')
# VGPR single with comment:            v194 /*v706*/
_VGPR_SINGLE_COMMENT = re.compile(r'v(\d+)\s*/\*v(\d+)\*/')
# VGPR single:                         v194
_VGPR_SINGLE = re.compile(r'(?<![a-zA-Z0-9_])v(\d+)(?![a-zA-Z0-9_])')
# SGPR range:                          s[16:19]
_SGPR_RANGE = re.compile(r's\[(\d+):(\d+)\]')
# SGPR single:                         s16
_SGPR_SINGLE = re.compile(r'(?<![a-zA-Z0-9_])s(\d+)(?![a-zA-Z0-9_])')
# AGPR range / single (mi450 rarely uses these but parse for completeness)
_AGPR_RANGE = re.compile(r'a\[(\d+):(\d+)\]')
_AGPR_SINGLE = re.compile(r'(?<![a-zA-Z0-9_])a(\d+)(?![a-zA-Z0-9_])')

# Marker comments emitted by the scheduler inline asm.  Two formats in
# the wild:
#   ;; Region 0: 32 wmma, 1 GR, 16 LR            (production)
#   ;; Epilogue Region 0: 32 wmma, 0 GR, 16 LR, 0 LW, 0 CVT  (epilogue)
#   ; region 0: wmma=32 ds_load=16 tdm=0          (legacy)
_REGION_MARKER_PROD = re.compile(
    r';{1,2}\s*(Epilogue\s+)?Region\s+(\d+)\s*:\s*'
    r'(\d+)\s+wmma\s*,\s*(\d+)\s+GR\s*,\s*(\d+)\s+LR',
    re.IGNORECASE,
)
_REGION_MARKER_LEGACY = re.compile(
    r';\s*region\s+(\d+)\s*:\s*wmma=(\d+)\s+ds_load=(\d+)\s+tdm=(\d+)',
    re.IGNORECASE,
)
_SUBREGION_MARKER = re.compile(
    r';\s*sub-region\s+(\d+)\s*:\s*wmma=(\d+)\s+ds_load=(\d+)\s+tdm=(\d+)',
    re.IGNORECASE,
)

# Decode ``s_set_vgpr_msb`` into per-operand bank indices.
#
# The immediate low byte encodes four 2-bit bank fields:
#   bits 1-0: src0, bits 3-2: src1, bits 5-4: src2, bits 7-6: dst
# Bits 8-15 carry additional signals (validity/commit) we don't need here.
#
# The assembler also emits a human-readable decode as a trailing comment:
#   ``s_set_vgpr_msb 0x5a ; msbs: dst=1 src0=2 src1=2 src2=1``
# We prefer that form when present; fall back to the numeric decode.
_MSB_COMMENT = re.compile(
    r'msbs:\s*dst=(\d+)\s+src0=(\d+)\s+src1=(\d+)\s+src2=(\d+)'
)


def _decode_msb_imm(imm_str: str) -> tuple[int, int, int, int]:
    imm = int(imm_str, 0)
    src0 = imm & 0x3
    src1 = (imm >> 2) & 0x3
    src2 = (imm >> 4) & 0x3
    dst = (imm >> 6) & 0x3
    return (dst, src0, src1, src2)


def _decode_msb(imm_str: str, comment: Optional[str]) -> tuple[int, int, int, int]:
    if comment:
        m = _MSB_COMMENT.search(comment)
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return _decode_msb_imm(imm_str)


# -------------------------------------------------------------------------
# Operand parsing
# -------------------------------------------------------------------------

def _parse_operand(text: str) -> Operand:
    """Parse an operand text, extracting registers and preserving the
    logical-id comment (``/*v[...]*/``) if present.

    ``text`` is expected to be stripped of leading/trailing whitespace but
    may contain internal spaces (e.g., between a register and its comment).
    """
    regs: list[Register] = []
    logical_text: Optional[str] = None

    # Try VGPR range with comment first (longest match wins).
    m = _VGPR_RANGE_COMMENT.search(text)
    if m:
        raw_lo, raw_hi = int(m.group(1)), int(m.group(2))
        log_lo, log_hi = int(m.group(3)), int(m.group(4))
        regs.append(Register(
            kind='v',
            ids=list(range(log_lo, log_hi + 1)),
            raw_ids=list(range(raw_lo, raw_hi + 1)),
        ))
        # Just the bracketed register part, e.g. "v[192:199]".
        op_text = text[m.start():m.end(2) + 1]
        logical_text = f"/*v[{log_lo}:{log_hi}]*/"
        suffix = text[m.end():].lstrip() or None
        return Operand(text=op_text, regs=regs,
                       logical_text=logical_text, suffix=suffix)

    m = _VGPR_SINGLE_COMMENT.search(text)
    if m:
        raw = int(m.group(1))
        log = int(m.group(2))
        regs.append(Register(kind='v', ids=[log], raw_ids=[raw]))
        op_text = f"v{raw}"
        logical_text = f"/*v{log}*/"
        suffix = text[m.end():].lstrip() or None
        return Operand(text=op_text, regs=regs,
                       logical_text=logical_text, suffix=suffix)

    # No comment forms.  Always carve text into (register, suffix) so
    # later rewrites that replace ``op.text`` don't lose any trailing
    # ``offset:N`` modifier.
    m = _VGPR_RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        regs.append(Register(kind='v', ids=list(range(lo, hi + 1))))
        op_text = text[m.start():m.end()]
        suffix = text[m.end():].lstrip() or None
        return Operand(text=op_text, regs=regs, suffix=suffix)

    m = _SGPR_RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        regs.append(Register(kind='s', ids=list(range(lo, hi + 1))))
        op_text = text[m.start():m.end()]
        suffix = text[m.end():].lstrip() or None
        return Operand(text=op_text, regs=regs, suffix=suffix)

    m = _AGPR_RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        regs.append(Register(kind='a', ids=list(range(lo, hi + 1))))
        op_text = text[m.start():m.end()]
        suffix = text[m.end():].lstrip() or None
        return Operand(text=op_text, regs=regs, suffix=suffix)

    m = _VGPR_SINGLE.search(text)
    if m:
        regs.append(Register(kind='v', ids=[int(m.group(1))]))
        op_text = text[m.start():m.end()]
        suffix = text[m.end():].lstrip() or None
        return Operand(text=op_text, regs=regs, suffix=suffix)

    m = _SGPR_SINGLE.search(text)
    if m:
        regs.append(Register(kind='s', ids=[int(m.group(1))]))
        op_text = text[m.start():m.end()]
        suffix = text[m.end():].lstrip() or None
        return Operand(text=op_text, regs=regs, suffix=suffix)

    m = _AGPR_SINGLE.search(text)
    if m:
        regs.append(Register(kind='a', ids=[int(m.group(1))]))
        op_text = text[m.start():m.end()]
        suffix = text[m.end():].lstrip() or None
        return Operand(text=op_text, regs=regs, suffix=suffix)

    # Literal or modifier (``0x80``, ``offset:32``, ``m0``, etc.)
    if text == 'm0':
        regs.append(Register(kind='m', ids=[0]))

    return Operand(text=text, regs=regs)


# -------------------------------------------------------------------------
# Instruction parsing
# -------------------------------------------------------------------------

def _split_operands(text: str) -> list[str]:
    """Split operand text at top-level commas, preserving brackets."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            tok = ''.join(buf).strip()
            if tok:
                parts.append(tok)
            buf = []
        else:
            buf.append(ch)
    tok = ''.join(buf).strip()
    if tok:
        parts.append(tok)
    return parts


def _strip_trailing_comment(line: str) -> tuple[str, Optional[str]]:
    """Separate an instruction line from its trailing ``; ...`` comment,
    being careful not to split inside a ``/* ... */`` comment.

    Returns (code_portion, comment_or_None)."""
    depth_block = 0
    for i, ch in enumerate(line):
        if ch == '/' and i + 1 < len(line) and line[i + 1] == '*':
            depth_block += 1
        elif ch == '*' and i + 1 < len(line) and line[i + 1] == '/':
            depth_block -= 1
        elif ch == ';' and depth_block == 0:
            return line[:i].rstrip(), line[i + 1:]
    return line, None


_DUAL_SEP_RE = re.compile(r'\s+::\s+(v_dual_[A-Za-z0-9_]+)\s+')


def _parse_instruction_line(line: str) -> Instruction:
    """Parse a non-empty, non-label instruction line."""
    code, comment = _strip_trailing_comment(line)
    stripped = code.strip()
    # Directives (``.file``, ``.loc``, ``.size`` etc.) are preserved verbatim.
    if stripped.startswith('.'):
        return Instruction(opcode=stripped.split(None, 1)[0], operands=[],
                           raw_line=line, trailing_comment=comment)

    head, _, operand_text = stripped.partition(' ')
    opcode = head.strip()

    # Detect GFX1250 dual-issue: ``op a, b :: op c, d``.  The two halves
    # share the same opcode; we collapse to one Instruction with 4
    # operands and set ``dual_issue=True`` so emit can restore the
    # separator.
    dual_issue = False
    if opcode.startswith('v_dual_'):
        m = _DUAL_SEP_RE.search(operand_text)
        if m and m.group(1) == opcode:
            left_text = operand_text[:m.start()]
            right_text = operand_text[m.end():]
            operands: list[Operand] = []
            for part in _split_operands(left_text):
                operands.append(_parse_operand(part))
            for part in _split_operands(right_text):
                operands.append(_parse_operand(part))
            dual_issue = True
        else:
            operands = [_parse_operand(p)
                        for p in _split_operands(operand_text)]
    else:
        operands = [_parse_operand(p)
                    for p in _split_operands(operand_text)]

    inst = Instruction(opcode=opcode, operands=operands,
                       raw_line=line, trailing_comment=comment,
                       dual_issue=dual_issue)
    if opcode == 's_set_vgpr_msb' and operands and operands[0].text:
        try:
            inst.msb_bits = _decode_msb(operands[0].text, comment)
        except ValueError:
            pass
    return inst


# -------------------------------------------------------------------------
# Inline asm block (scheduler marker) parsing
# -------------------------------------------------------------------------

def _parse_asm_block(lines: list[str]) -> Instruction:
    """Given the lines of a ``;;#ASMSTART ... ;;#ASMEND`` block (inclusive),
    build a single Instruction holding the raw text and decoded marker."""
    raw = "\n".join(lines)
    inst = Instruction(opcode='__asm_block__', operands=[], raw_line=raw)
    # Scan inner lines for a recognized marker.
    for ln in lines:
        m = _REGION_MARKER_PROD.search(ln)
        if m:
            # Production format: (Epilogue?, region, wmma, GR, LR).  GR
            # here includes TDM (global reads); LR is ds_load count.
            inst.region_marker = RegionMarker(
                region=int(m.group(2)),
                wmma=int(m.group(3)),
                ds_load=int(m.group(5)),
                tdm=int(m.group(4)),
                is_epilogue=m.group(1) is not None,
            )
            continue
        m = _REGION_MARKER_LEGACY.search(ln)
        if m:
            inst.region_marker = RegionMarker(
                region=int(m.group(1)),
                wmma=int(m.group(2)),
                ds_load=int(m.group(3)),
                tdm=int(m.group(4)),
            )
            continue
        m = _SUBREGION_MARKER.search(ln)
        if m:
            inst.subregion_marker = SubRegionMarker(
                sub_region=int(m.group(1)),
                wmma=int(m.group(2)),
                ds_load=int(m.group(3)),
                tdm=int(m.group(4)),
            )
    return inst


# -------------------------------------------------------------------------
# Top-level assembly parse
# -------------------------------------------------------------------------

_LABEL_RE = re.compile(r'^\s*(\.?[A-Za-z_][A-Za-z0-9_\.]*):\s*(?:;.*)?$')


def parse_asm(text: str) -> Program:
    """Parse a full AMDGCN assembly text emitted by the LLVM backend for
    gfx1250 into a :class:`Program`.

    Preserves the source verbatim: ``emit_program(parse_asm(text))`` yields
    a byte-identical string for unmodified programs.
    """
    program = Program()
    current_bb: Optional[BasicBlock] = None
    in_program = False
    ended = False

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if ended:
            program.tail_lines.append(line)
            i += 1
            continue

        # Enter a ;;#ASMSTART ... ;;#ASMEND block and take it whole.
        if stripped.startswith(';;#ASMSTART'):
            block_lines = [line]
            i += 1
            while i < len(lines):
                block_lines.append(lines[i])
                if lines[i].strip().startswith(';;#ASMEND'):
                    i += 1
                    break
                i += 1
            if current_bb is None:
                program.header_lines.extend(block_lines)
            else:
                current_bb.add_inst(_parse_asm_block(block_lines))
            continue

        # Label line.
        m = _LABEL_RE.match(line)
        if m and not stripped.startswith(';'):
            name = m.group(1)
            # Only treat as a basic-block label once the ``.text`` section
            # has begun and it's not a directive like ``.text:``.
            if in_program:
                current_bb = BasicBlock(name=name, label_line=line)
                program.blocks.append(current_bb)
                i += 1
                continue

        # Recognize the start of the function body by the first basic block.
        if line.startswith('; %bb.') and current_bb is None:
            in_program = True
            # The basic-block label will follow this line; keep scanning.

        # If we haven't entered any basic block yet, this is header.
        if current_bb is None:
            program.header_lines.append(line)
            if stripped.startswith('; %bb.'):
                in_program = True
            i += 1
            continue

        # Empty line or pure comment - attach as an instruction with empty
        # opcode so round-trip preserves whitespace/comments.
        if not stripped or stripped.startswith(';'):
            inst = Instruction(opcode='', operands=[], raw_line=line,
                               trailing_comment=None)
            current_bb.add_inst(inst)
            i += 1
            continue

        inst = _parse_instruction_line(line)
        current_bb.add_inst(inst)
        if inst.opcode == 's_endpgm':
            ended = True
        i += 1

    return program


# -------------------------------------------------------------------------
# Passes (peephole-style cleanups)
# -------------------------------------------------------------------------

@dataclass
class RegionSpan:
    """A contiguous range of instructions in a BasicBlock owned by one
    ``;; Region N`` marker.  ``start`` points at the inline-asm block
    carrying the marker; ``end`` is the index of the next region's
    marker (or len(bb.instructions) for the last region in the block)."""
    bb: BasicBlock
    start: int            # inclusive (points at the marker instruction)
    end: int              # exclusive
    marker: RegionMarker


def find_region_spans(program: Program) -> list[RegionSpan]:
    """Return all region spans in the program (loop and epilogue).

    A region starts at an inline-asm block carrying a :class:`RegionMarker`
    and ends at the next such marker in the same block or at the end of
    the block.
    """
    spans: list[RegionSpan] = []
    for bb in program.blocks:
        # Find all region-marker positions in this block.
        starts: list[tuple[int, RegionMarker]] = []
        for idx, inst in enumerate(bb.instructions):
            if inst.region_marker is not None:
                starts.append((idx, inst.region_marker))
        for i, (idx, marker) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(bb.instructions)
            spans.append(RegionSpan(bb=bb, start=idx, end=end, marker=marker))
    return spans


def _is_ds_mem_op(inst: Instruction) -> bool:
    """True if this instruction increments the DS (LDS) counter observed
    by ``s_wait_dscnt``.  Covers both loads and stores."""
    op = inst.opcode
    return (
        op.startswith('ds_load') or op.startswith('ds_read') or
        op.startswith('ds_store') or op.startswith('ds_write')
    )


def _wait_count(inst: Instruction) -> Optional[int]:
    """Extract the numeric argument of ``s_wait_dscnt``.  Returns None if
    the argument can't be parsed."""
    if inst.opcode != 's_wait_dscnt':
        return None
    if not inst.operands:
        return None
    try:
        return int(inst.operands[0].text, 0)
    except ValueError:
        return None


def _loop_body_range(program: Program) -> Optional[tuple["BasicBlock", int]]:
    """Return ``(loop_bb, cbranch_idx)`` where instructions at indices
    ``[0, cbranch_idx]`` in ``loop_bb`` form the loop body.  The
    AMDGPU backend emits the loop and any fall-through epilogue into
    the same basic block, so we need the ``s_cbranch`` position to
    separate them.  Returns None if there is no self-loop."""
    loop_bb = _find_self_loop(program)
    if loop_bb is None:
        return None
    for i, inst in enumerate(loop_bb.instructions):
        if not inst.opcode.startswith('s_cbranch'):
            continue
        tgt = inst.operands[-1].text if inst.operands else ""
        if tgt.strip() == loop_bb.name:
            return loop_bb, i
    return None


def merge_dscnt_waits(program: Program) -> int:
    """Consolidate ``s_wait_dscnt`` instructions inside each *loop*
    region into a single wait at the region start.

    Restricted to the loop body on purpose: epilogue regions are
    executed only once, so consolidating their waits doesn't save
    anything, and the arithmetic used to pick the new wait count
    (``V_last - N_before_last``) depends on the steady-state
    ds-in-flight accounting of a pipelined inner loop -- it's not
    meaningful for the one-shot epilogue.

    Rationale for loop regions: the LLIR scheduler emits multiple
    intermediate waits that progressively drain the outstanding
    LDS-load counter.  For each region, the last wait is typically
    the most restrictive, at a point after all of the region's own
    ``ds_load`` instructions have been issued.  Consolidating at the
    region start with a count of ``V_last - N_before_last`` guarantees
    the outstanding counter is at most ``V_last`` at every downstream
    use (any new ds_loads issued in the region only add up to
    ``N_before_last`` to the count).

    Returns the number of wait instructions removed (for reporting).
    """
    loop_range = _loop_body_range(program)
    if loop_range is None:
        return 0
    loop_bb, cbranch_idx = loop_range

    removed = 0
    # Process in reverse so earlier spans' indices remain valid after we
    # mutate later spans in the same block.
    for span in reversed(find_region_spans(program)):
        if span.bb is not loop_bb or span.start >= cbranch_idx:
            continue
        bb = span.bb
        # Collect positions of waits and ds-ops within [start+1, end).
        wait_positions: list[int] = []
        wait_values: list[int] = []
        for idx in range(span.start + 1, span.end):
            inst = bb.instructions[idx]
            v = _wait_count(inst)
            if v is not None:
                wait_positions.append(idx)
                wait_values.append(v)
        if not wait_positions:
            continue

        last_idx = wait_positions[-1]
        v_last = wait_values[-1]
        # Count ds-ops that precede the last wait within this region.
        n_before_last = 0
        for idx in range(span.start + 1, last_idx):
            if _is_ds_mem_op(bb.instructions[idx]):
                n_before_last += 1
        v_new = v_last - n_before_last
        if v_new < 0:
            # Conservative: if arithmetic would underflow, fall back to 0
            # (drain everything), which is always safe.
            v_new = 0

        # Remove existing waits (highest index first to avoid shifting).
        for idx in reversed(wait_positions):
            del bb.instructions[idx]
            removed += 1

        # Insert a single wait immediately after the region marker.
        # Use the indentation from a neighbouring code instruction (the
        # marker itself is a multi-line inline-asm block whose first line
        # isn't necessarily indented).
        leading_ws = '\t'
        for idx in range(span.start + 1, span.end):
            cand = bb.instructions[idx].raw_line
            stripped = cand.lstrip('\t ')
            if stripped and not stripped.startswith(';;#ASM') and not stripped.startswith('.'):
                leading_ws = cand[: len(cand) - len(stripped)]
                break
        new_line = f"{leading_ws}s_wait_dscnt 0x{v_new:x}"
        new_inst = _parse_instruction_line(new_line)
        bb.instructions.insert(span.start + 1, new_inst)
        # Shift the end index for downstream spans in the same block is
        # handled automatically because we recompute spans on next call.

        # Reindex the block after mutation.
        for i, inst in enumerate(bb.instructions):
            inst.index = i
    return removed


def overlap_wmma_with_barrier(program: Program) -> int:
    """Hoist the v_wmma that follows an ``s_barrier_signal/s_barrier_wait``
    pair into the gap between the signal and the wait so the wmma's
    compute overlaps the barrier's sync latency.

    Matches the pattern::

        s_barrier_signal -1
        s_barrier_wait -1
        [optional s_delay_alu / .loc]
        v_wmma_*

    and rewrites it to::

        s_barrier_signal -1
        [optional s_delay_alu / .loc]
        v_wmma_*
        s_barrier_wait -1

    The reorder is safe because the barrier wait is a workgroup-level
    synchronization that doesn't produce register values consumed by the
    wmma; the wmma's operands were defined before the signal.

    Returns the number of wmmas hoisted.
    """
    moved = 0
    for bb in program.blocks:
        i = 0
        while i < len(bb.instructions):
            if bb.instructions[i].opcode != 's_barrier_signal':
                i += 1
                continue
            # Find the matching wait immediately after (skipping no-op
            # directives like .loc — though in practice they follow the
            # wait, not between).
            j = i + 1
            while (j < len(bb.instructions) and
                   bb.instructions[j].opcode in ('', '.loc')):
                j += 1
            if j >= len(bb.instructions) or bb.instructions[j].opcode != 's_barrier_wait':
                i += 1
                continue
            wait_idx = j
            # Scan for the next v_wmma, allowing only "transparent"
            # fillers (directives, whitespace, s_delay_alu) in between.
            k = wait_idx + 1
            transparent_end = k
            while k < len(bb.instructions):
                op = bb.instructions[k].opcode
                if op.startswith('v_wmma'):
                    break
                if op in ('', '.loc', 's_delay_alu'):
                    transparent_end = k + 1
                    k += 1
                    continue
                # Any other instruction blocks the reorder.
                break
            else:
                i = wait_idx + 1
                continue
            if k >= len(bb.instructions) or not bb.instructions[k].opcode.startswith('v_wmma'):
                i = wait_idx + 1
                continue
            wmma_idx = k

            # Move [wait_idx+1 .. wmma_idx] (delay_alu + wmma + any .loc)
            # to between the signal and the wait.
            block = bb.instructions[wait_idx + 1 : wmma_idx + 1]
            wait_inst = bb.instructions[wait_idx]
            del bb.instructions[wait_idx : wmma_idx + 1]
            bb.instructions[i + 1 : i + 1] = block + [wait_inst]
            moved += 1
            # Advance past the inserted region.
            i = i + 1 + len(block) + 1
        # Reindex the block after mutation.
        for idx, inst in enumerate(bb.instructions):
            inst.index = idx
    return moved


# -------------------------------------------------------------------------
# Stage 2: region annotation, def-use, chain collection
# -------------------------------------------------------------------------

def annotate_regions(program: Program) -> None:
    """Tag each instruction in the program with the index of the enclosing
    ``;; Region`` and (if present) ``;; SubRegion`` markers.

    Instructions preceding the first marker in a block receive ``None``
    tags, matching their default state.  Loop-body regions and
    epilogue regions both use the same integer numbering space, so we
    also record ``region_is_epilogue`` from the marker's flag -- the
    pair ``(region_is_epilogue, region_idx)`` uniquely identifies a
    region.
    """
    for bb in program.blocks:
        cur_region: Optional[int] = None
        cur_is_epilogue: Optional[bool] = None
        cur_sub: Optional[int] = None
        for inst in bb.instructions:
            if inst.region_marker is not None:
                cur_region = inst.region_marker.region
                cur_is_epilogue = inst.region_marker.is_epilogue
                cur_sub = None
            if inst.subregion_marker is not None:
                cur_sub = inst.subregion_marker.sub_region
            inst.region_idx = cur_region
            inst.region_is_epilogue = cur_is_epilogue
            inst.sub_region_idx = cur_sub


def _iter_regs(reg_iter) -> Iterator[tuple[str, int]]:
    """Flatten a list of Register into (kind, logical_id) pairs."""
    for reg in reg_iter:
        for rid in reg.ids:
            yield (reg.kind, rid)


@dataclass
class DefUseIndex:
    """Linear per-block def-use map keyed on ``(kind, logical_id)``.

    ``defs`` lists every instruction that writes that register in source
    order; ``uses`` lists every instruction that reads it.  Unmodified
    passes use this to trace ds_load dst → WMMA src links.
    """
    defs: dict[tuple[str, int], list[Instruction]] = field(default_factory=dict)
    uses: dict[tuple[str, int], list[Instruction]] = field(default_factory=dict)

    def add_def(self, reg_id: tuple[str, int], inst: Instruction) -> None:
        self.defs.setdefault(reg_id, []).append(inst)

    def add_use(self, reg_id: tuple[str, int], inst: Instruction) -> None:
        self.uses.setdefault(reg_id, []).append(inst)

    def last_def_before(self, reg_id: tuple[str, int],
                        inst: Instruction) -> Optional[Instruction]:
        """Return the most recent definer of ``reg_id`` strictly before
        ``inst`` within the same basic block, or None."""
        defs = self.defs.get(reg_id, [])
        best: Optional[Instruction] = None
        for d in defs:
            if d.parent_bb is not inst.parent_bb:
                continue
            if d.index < inst.index:
                best = d
            else:
                break
        return best


def _add_insts_to_index(insts: Iterator[Instruction],
                        idx: DefUseIndex) -> None:
    for inst in insts:
        if not inst.opcode or inst.opcode == '__asm_block__':
            continue
        if not inst.operands:
            continue
        dst_op = inst.operands[0]
        if dst_op.regs:
            # Treat the first operand as the def for every instruction
            # that has register operands.  Overbroad for compares (which
            # define scalars), harmless for the VGPR flows we trace.
            for rid in _iter_regs(dst_op.regs):
                idx.add_def(rid, inst)
        for op in inst.operands[1:]:
            for rid in _iter_regs(op.regs):
                idx.add_use(rid, inst)


def build_def_use_index(bb: BasicBlock) -> DefUseIndex:
    """Scan a basic block linearly and index logical VGPR defs and uses."""
    idx = DefUseIndex()
    _add_insts_to_index(iter(bb.instructions), idx)
    return idx


def build_program_def_use_index(program: Program) -> DefUseIndex:
    """Scan the whole program in block order and index logical VGPR defs
    and uses.  Used for tracing cross-BB flows (prologue ds_load → loop
    WMMA, epilogue ds_load → epilogue WMMA)."""
    idx = DefUseIndex()
    _add_insts_to_index(program.iter_instructions(), idx)
    return idx


def _program_order(inst: Instruction, block_order: dict[str, int]) -> tuple[int, int]:
    """A sortable key reflecting program order across blocks."""
    bb_name = inst.parent_bb.name if inst.parent_bb else ""
    return (block_order.get(bb_name, 0), inst.index)


# -------------------------------------------------------------------------
# WMMA chains
# -------------------------------------------------------------------------

@dataclass
class WMMAChain:
    """A group of WMMA instructions that share the same accumulator
    register range.  All WMMAs in the chain have ``dst == src2``, so the
    chain forms a straight-line accumulation sequence (possibly
    interleaved with other WMMAs in the scheduled assembly)."""
    canonical: Register
    wmmas: list[Instruction] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.wmmas)

    def summary(self) -> str:
        return (f"WMMAChain(dst={self.canonical}, size={self.size}, "
                f"bb={self.wmmas[0].parent_bb.name if self.wmmas else '?'})")


def collect_wmma_chains(program: Program) -> list[WMMAChain]:
    """Group WMMA instructions by their accumulator register.

    Returns a list of :class:`WMMAChain` objects, one per distinct dst
    register used by any WMMA in the program.  Instructions are linked
    back via ``Instruction.wmma_chain``.

    Idempotent: if ``Instruction.wmma_chain`` is already populated
    (from a prior call), the existing chains are returned unchanged so
    callers that compare by object identity (dict keys, id(...)) stay
    consistent across calls.
    """
    cached: list[WMMAChain] = []
    seen: set[int] = set()
    for inst in program.iter_instructions():
        c = inst.wmma_chain
        if c is not None and id(c) not in seen:
            cached.append(c)
            seen.add(id(c))
    if cached:
        return cached

    chains: dict[Register, WMMAChain] = {}
    for inst in program.iter_instructions():
        if not inst.opcode.startswith('v_wmma'):
            continue
        dst = inst.dst_reg()
        if dst is None:
            continue
        chain = chains.get(dst)
        if chain is None:
            chain = WMMAChain(canonical=dst)
            chains[dst] = chain
        chain.wmmas.append(inst)
        inst.wmma_chain = chain
    return list(chains.values())


# -------------------------------------------------------------------------
# DS-load chains
# -------------------------------------------------------------------------

def _is_ds_load(inst: Instruction) -> bool:
    return inst.opcode.startswith('ds_load') or inst.opcode.startswith('ds_read')


@dataclass
class DSGroup:
    """A tile-sized group of ds_load instructions that together load one
    operand tile for a wmma.

    For wmma_f32_16x16x32_f16, a single src0/src1 tile is 8 VGPRs loaded
    by 2 ``ds_load_b128`` instructions with contiguous dst ranges.  This
    class keeps those together along with the set of wmma instructions
    that consume the tile (a single tile can feed many wmmas when
    WMMAChains share a matrix operand).

    All ds_loads in a DSGroup share the same ``addr`` register (they
    differ only in the instruction-encoded offset) and feed the same
    wmma operand slot (``op_idx``).

    ``prologue_loads`` holds any pre-loop ds_loads that prefetch the
    same tile for the first iteration -- they're scheduled before the
    loop entry (no region marker), but under the pipelined pattern
    they mirror the group's steady-state loop loads and write to the
    same logical tile VGPRs.  Stage 4 renames them alongside
    ``ds_loads``.
    """
    ds_loads: list[Instruction] = field(default_factory=list)
    consumer_wmmas: list[Instruction] = field(default_factory=list)
    prologue_loads: list[Instruction] = field(default_factory=list)
    # 1 = src0, 2 = src1, 3 = src2 (rare).  ``None`` on construction
    # until the consumer scan fills it in.
    op_idx: Optional[int] = None

    @property
    def addr_reg(self) -> Optional[Register]:
        if not self.ds_loads or len(self.ds_loads[0].operands) < 2:
            return None
        regs = self.ds_loads[0].operands[1].regs
        return regs[0] if regs else None

    @property
    def data_regs(self) -> list[Register]:
        return [ld.dst_reg() for ld in self.ds_loads if ld.dst_reg()]

    @property
    def tile(self) -> Optional[Register]:
        """The combined contiguous VGPR range covered by all data_regs."""
        regs = self.data_regs
        if not regs:
            return None
        lo = min(r.start for r in regs)
        hi = max(r.end for r in regs)
        return Register(kind='v', ids=list(range(lo, hi + 1)))

    @property
    def all_loads(self) -> list[Instruction]:
        """Every ds_load writing this tile: steady-state + prologue."""
        return self.ds_loads + self.prologue_loads

    def summary(self) -> str:
        tile = self.tile
        slot = f'src{self.op_idx - 1}' if self.op_idx is not None else 'src?'
        extra = f' + {len(self.prologue_loads)} pre' if self.prologue_loads else ''
        return (f'DSGroup({len(self.ds_loads)} loads{extra}, tile={tile}, '
                f'{slot}, {len(self.consumer_wmmas)} consumers)')


@dataclass
class DSChain:
    """All ds_load tile-groups loaded in one region of the program.

    Shape: a DSChain has one loading region (identified by the
    ``(is_epilogue_region, loading_region)`` pair), and owns one or
    more :class:`DSGroup` tiles.  Under the LLIR scheduler's design,
    all the DSGroups within one region:

      * share the same ``addr`` register (the region's base pointer);
      * share the same ``op_idx`` (all src0 or all src1);

    (enforced by :func:`collect_ds_chains`).  The consumers of those
    tiles may be wmmas in multiple regions (including the epilogue
    and, via the loop back-edge, earlier loop regions of the next
    iteration).
    """
    loading_region: int
    is_epilogue_region: bool
    dsgroups: list[DSGroup] = field(default_factory=list)
    # Program-order positions used by Stage 4.3 to share data VGPRs
    # between same-bank DSChains with non-overlapping lifetimes.
    # ``loading_pos`` is the position of the chain's first loop
    # ds_load (prologue loads attached to the chain are not counted --
    # they're a Stage-4.4 rewrite concern, not a sharing one).
    # ``last_consumer_pos`` is the latest reaching-def consumer wmma.
    # Both keys are ``(block_order_index, instruction_index)`` tuples
    # produced by :func:`_program_order`.  None if the chain has no
    # ds_loads or consumers.
    loading_pos: Optional[tuple[int, int]] = None
    last_consumer_pos: Optional[tuple[int, int]] = None

    @property
    def addr_reg(self) -> Optional[Register]:
        return self.dsgroups[0].addr_reg if self.dsgroups else None

    @property
    def op_idx(self) -> Optional[int]:
        return self.dsgroups[0].op_idx if self.dsgroups else None

    @property
    def ds_loads(self) -> list[Instruction]:
        return [ld for g in self.dsgroups for ld in g.ds_loads]

    @property
    def consumer_wmmas(self) -> list[Instruction]:
        seen: set[int] = set()
        out: list[Instruction] = []
        for g in self.dsgroups:
            for w in g.consumer_wmmas:
                if id(w) not in seen:
                    seen.add(id(w))
                    out.append(w)
        return out

    def summary(self) -> str:
        tag = 'E' if self.is_epilogue_region else 'L'
        slot = f'src{self.op_idx - 1}' if self.op_idx is not None else 'src?'
        return (f'DSChain({tag}{self.loading_region}, addr={self.addr_reg}, '
                f'{slot}, {len(self.dsgroups)} groups, '
                f'{sum(len(g.ds_loads) for g in self.dsgroups)} loads)')


def can_share_data_vgprs(a: "DSChain", b: "DSChain") -> bool:
    """Return True iff two DSChains' data lifetimes are disjoint -- one's
    last consumer wmma comes strictly before the other's first ds_load.

    Stage 4.3 uses this to share data VGPRs between same-bank DSChains.
    For example in v9, L0 (last consumer at L3) and L4 (loading at L4)
    return True: L0's data is fully consumed by the time L4 overwrites
    the same physical registers.  Stride-4 sibling pairs L0/L4, L1/L5,
    L2/L6, L3/L7 all share by this rule, keeping the loop's data VGPR
    footprint at one set per bank instead of two.

    Returns False if either chain has incomplete lifetime info (no
    ds_loads or no consumers); callers should treat that as
    non-shareable.
    """
    if (a.loading_pos is None or b.loading_pos is None or
            a.last_consumer_pos is None or b.last_consumer_pos is None):
        return False
    return (a.last_consumer_pos < b.loading_pos
            or b.last_consumer_pos < a.loading_pos)


def _find_reaching_consumers(ld: Instruction, pindex: "DefUseIndex",
                             block_order: dict[str, int]
                             ) -> tuple[list[Instruction], set[int]]:
    """For a single ds_load, find every wmma that reads any of its dst
    VGPRs before the next instruction redefines those VGPRs (true
    reaching-def consumers, not raw uses).  Also returns the set of
    operand-slot indices those consumers read the data in."""
    dst = ld.dst_reg()
    if dst is None:
        return [], set()
    load_key = _program_order(ld, block_order)
    consumers: list[Instruction] = []
    consumer_ids: set[int] = set()
    slots: set[int] = set()
    for rid in _iter_regs([dst]):
        next_def_key: Optional[tuple[int, int]] = None
        for d in pindex.defs.get(rid, []):
            dk = _program_order(d, block_order)
            if dk <= load_key:
                continue
            if next_def_key is None or dk < next_def_key:
                next_def_key = dk
        for u in pindex.uses.get(rid, []):
            if not u.opcode.startswith('v_wmma'):
                continue
            uk = _program_order(u, block_order)
            if uk <= load_key:
                continue
            if next_def_key is not None and uk >= next_def_key:
                continue
            # Which operand slot of u reads this rid?
            for slot_idx, op in enumerate(u.operands[1:], start=1):
                if any(uid == rid for uid in _iter_regs(op.regs)):
                    slots.add(slot_idx)
                    break
            if id(u) not in consumer_ids:
                consumer_ids.add(id(u))
                consumers.append(u)
    return consumers, slots


def collect_ds_chains(program: Program) -> list[DSChain]:
    """Build per-loading-region DSChains with two-level DSGroup structure.

    Within each region, ds_loads are sorted by dst start VGPR and then
    grouped into DSGroups whose dsts are contiguous (a tile).  Each
    DSGroup's consumer wmmas are found by reaching-def analysis
    (instructions reading any of the tile's VGPRs before the next
    redefinition), and may live in any later region -- including back-
    edge reads in earlier loop regions of the next iteration, and
    epilogue regions.

    DSGroups with no consumer wmmas (scratch / unused) are dropped.
    ds_loads that have no enclosing region marker (prologue loads) are
    also skipped for now; a future pass will attach them via dependency
    propagation.

    Sanity checks per DSChain (raises on violation): shared addr reg,
    shared op_idx.

    Idempotent: if ``Instruction.ds_chain`` is already populated, the
    existing chains are returned unchanged.
    """
    cached: list[DSChain] = []
    seen: set[int] = set()
    for inst in program.iter_instructions():
        c = inst.ds_chain
        if c is not None and id(c) not in seen:
            cached.append(c)
            seen.add(id(c))
    if cached:
        return cached

    annotate_regions(program)
    if not any(i.wmma_chain for i in program.iter_instructions()
               if i.opcode.startswith('v_wmma')):
        collect_wmma_chains(program)

    pindex = build_program_def_use_index(program)
    block_order = {bb.name: i for i, bb in enumerate(program.blocks)}

    by_region: dict[tuple[bool, int], list[Instruction]] = {}
    for inst in program.iter_instructions():
        if not _is_ds_load(inst):
            continue
        if inst.region_idx is None:
            continue  # prologue ds_loads skipped; handled via propagation later
        key = (bool(inst.region_is_epilogue), inst.region_idx)
        by_region.setdefault(key, []).append(inst)

    chains: list[DSChain] = []
    for (is_epi, region_idx), loads in by_region.items():
        # Build tile-per-wmma-operand DSGroups: one DSGroup per distinct
        # (wmma-operand-range, op_idx) tile read by a consumer.  Each
        # ds_load is assigned to the tile its dst sits inside.  Under the
        # LLIR scheduler's design, a tile is the full 8-VGPR operand of
        # one wmma (filled by 2 ds_load_b128's), shared by multiple
        # consumer wmmas when the scheduler reuses the tile.
        tile_map: dict[tuple[tuple[int, ...], int], DSGroup] = {}
        for ld in loads:
            consumers, slots = _find_reaching_consumers(
                ld, pindex, block_order)
            if not consumers:
                continue
            if len(slots) > 1:
                raise ValueError(
                    f'ds_load at L{region_idx} feeds mixed operand '
                    f'slots {sorted(slots)}')
            slot = next(iter(slots)) if slots else None
            if slot is None:
                continue
            # Pick the tile = consumer's operand[slot] range.  All
            # consumers of this ds_load should agree on the range
            # (they share the tile).
            dst = ld.dst_reg()
            if dst is None:
                continue
            dst_id_set = set(dst.ids)
            tile_ids: Optional[tuple[int, ...]] = None
            for u in consumers:
                if slot >= len(u.operands) or not u.operands[slot].regs:
                    continue
                candidate = tuple(u.operands[slot].regs[0].ids)
                if not dst_id_set.issubset(candidate):
                    continue
                if tile_ids is None:
                    tile_ids = candidate
                elif tile_ids != candidate:
                    raise ValueError(
                        f'ds_load at L{region_idx} dst {dst} has '
                        f'consumers reading inconsistent tile ranges '
                        f'{tile_ids} vs {candidate}')
            if tile_ids is None:
                continue  # no consumer whose operand range contains this dst
            key = (tile_ids, slot)
            g = tile_map.get(key)
            if g is None:
                g = DSGroup(op_idx=slot)
                tile_map[key] = g
            if ld not in g.ds_loads:
                g.ds_loads.append(ld)
            for u in consumers:
                if (slot < len(u.operands)
                        and u.operands[slot].regs
                        and tuple(u.operands[slot].regs[0].ids) == tile_ids
                        and u not in g.consumer_wmmas):
                    g.consumer_wmmas.append(u)

        kept = list(tile_map.values())
        if not kept:
            continue

        # Within a region, DSGroups can use multiple addr registers when
        # the kernel reads from more than one LDS buffer in the same
        # region (e.g., v9's epilogue after the dual-buffer refactor:
        # one tdm.async_store sources its data from buffer A via addr
        # v642 and another from buffer B via addr v806).  Split into one
        # DSChain per (addr_reg, op_idx) pair so each chain still
        # satisfies the "shared addr + shared slot" invariant downstream
        # passes rely on.  Order: keep insertion order from tile_map for
        # determinism.
        from collections import OrderedDict
        by_key: dict[tuple, list[DSGroup]] = OrderedDict()
        for g in kept:
            key = (g.addr_reg, g.op_idx)
            by_key.setdefault(key, []).append(g)

        for (addr_key, slot_key), groups in by_key.items():
            chain = DSChain(
                loading_region=region_idx,
                is_epilogue_region=is_epi,
                dsgroups=groups,
            )
            chain_loads = chain.ds_loads
            if chain_loads:
                chain.loading_pos = min(_program_order(ld, block_order)
                                        for ld in chain_loads)
            chain_consumers = chain.consumer_wmmas
            if chain_consumers:
                chain.last_consumer_pos = max(
                    _program_order(w, block_order)
                    for w in chain_consumers)
            chains.append(chain)
            for g in groups:
                for ld in g.ds_loads:
                    ld.ds_chain = chain

    # Attach prologue ds_loads (no region marker) to the loop DSGroup
    # whose steady-state tile they mirror.  Matching rule: the first
    # reaching-def wmma consumer defines the tile shape (operand range
    # + op_idx); we find the DSGroup with the same (tile, op_idx) and
    # the same addr register.
    prologue_loads = [
        inst for inst in program.iter_instructions()
        if _is_ds_load(inst) and inst.region_idx is None
    ]
    if prologue_loads:
        from collections import defaultdict
        # Build a lookup from (tile_ids, op_idx) -> list of (DSChain, DSGroup).
        tile_index: dict[tuple[tuple[int, ...], int],
                         list[tuple[DSChain, DSGroup]]] = defaultdict(list)
        for c in chains:
            if c.is_epilogue_region:
                continue
            for g in c.dsgroups:
                if g.tile is None or g.op_idx is None:
                    continue
                tile_index[(tuple(g.tile.ids), g.op_idx)].append((c, g))

        for ld in prologue_loads:
            dst = ld.dst_reg()
            if dst is None:
                continue
            addr_op = ld.operands[1] if len(ld.operands) > 1 else None
            addr_reg = (addr_op.regs[0]
                        if addr_op and addr_op.regs else None)
            # Find the first reaching-def wmma consumer of any of dst's rids.
            load_key = _program_order(ld, block_order)
            tile_key: Optional[tuple[tuple[int, ...], int]] = None
            for rid in _iter_regs([dst]):
                # Find next def of rid after this prologue load.
                next_key = None
                for d in pindex.defs.get(rid, []):
                    dk = _program_order(d, block_order)
                    if dk <= load_key:
                        continue
                    if next_key is None or dk < next_key:
                        next_key = dk
                # First wmma consumer in [load_key, next_key).
                for u in pindex.uses.get(rid, []):
                    if not u.opcode.startswith('v_wmma'):
                        continue
                    uk = _program_order(u, block_order)
                    if uk <= load_key:
                        continue
                    if next_key is not None and uk >= next_key:
                        continue
                    for slot_idx, op in enumerate(u.operands[1:], start=1):
                        if any(uid == rid for uid in _iter_regs(op.regs)):
                            if op.regs:
                                tile_key = (tuple(op.regs[0].ids), slot_idx)
                            break
                    if tile_key is not None:
                        break
                if tile_key is not None:
                    break
            if tile_key is None:
                continue
            # Find a candidate (chain, group) with matching tile+op_idx.
            # When multiple loop DSChains load the same tile (e.g.,
            # stride-4 siblings L3 and L7 both load v[512:519] with
            # src1+v642 in v9), pick the one with the largest
            # loading_region: that's the last load of the tile in the
            # pipeline cycle, hence the slot the prologue is
            # pre-iterating via the loop back-edge.  Filter by addr
            # first so pre-loads that use a specific base pointer
            # don't snap to a sibling that uses a different base.
            candidates = tile_index.get(tile_key, [])
            if not candidates:
                continue
            if addr_reg is not None:
                addr_matched = [(c, g) for c, g in candidates
                                if c.addr_reg == addr_reg]
                if addr_matched:
                    candidates = addr_matched
            picked = max(candidates, key=lambda cg: cg[0].loading_region)
            c, g = picked
            if ld not in g.prologue_loads:
                g.prologue_loads.append(ld)
            ld.ds_chain = c

    return chains


# -------------------------------------------------------------------------
# Human-readable diagnostic report
# -------------------------------------------------------------------------

def report_chains(program: Program) -> str:
    """Build a human-readable summary of the chains in the program.

    Intended for ``TRITON_ENABLE_AMDGCN_AS=2`` runs and for debugging the
    upcoming bank-assignment stage.  Format is informational only.
    """
    lines: list[str] = []
    wmma_chains = collect_wmma_chains(program)
    ds_chains = collect_ds_chains(program)

    def _fmt_region(inst):
        if inst.region_idx is None:
            return None
        return f'E{inst.region_idx}' if inst.region_is_epilogue else f'L{inst.region_idx}'

    lines.append(f"WMMA chains: {len(wmma_chains)}")
    for chain in sorted(wmma_chains, key=lambda c: c.canonical.start):
        bbs = {w.parent_bb.name for w in chain.wmmas if w.parent_bb}
        regions = sorted({_fmt_region(w) for w in chain.wmmas
                          if _fmt_region(w) is not None})
        lines.append(f"  {chain.summary()} regions={regions} bbs={sorted(bbs)}")

    lines.append(f"DS chains: {len(ds_chains)}")
    for chain in sorted(ds_chains,
                        key=lambda c: (c.is_epilogue_region, c.loading_region)):
        consumer_regions = sorted({_fmt_region(w) for w in chain.consumer_wmmas
                                   if _fmt_region(w) is not None})
        lines.append(f"  {chain.summary()} consumers={consumer_regions}")

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Stage 3: bank assignment
# -------------------------------------------------------------------------
#
# Assigns a VGPR bank (0..3) to every accumulator chain and every
# ds_load chain based on the pipelined-region structure the LLIR
# scheduler emits.  The invariant the assignment enforces, for every
# loop region, is:
#
#     all wmmas in the region share one MSB state
#     all ds_loads in the region share that same MSB state
#
# which means Stage 4 (renaming) + Stage 5 (MSB regeneration) should
# be able to emit exactly one ``s_set_vgpr_msb`` per region boundary.
#
# Algorithm (no VGPR renaming yet; this pass just decides the target
# banks):
#
#   Phase 2.  wmmaChain.acc_bank = (first_loop_region % 4) when the
#             chain is first visited; later regions that contain the
#             same chain (regions N and N+4 in the 4-region pipeline
#             cycle) preserve that bank.
#
#   Phase 3.  DSChain.data_bank = acc_bank of the region hosting its
#             ds_loads -- so the ds_load's dst sits in the same bank
#             as the region's wmma accumulators.
#
#   Phase 4.  wmmaChain.src0_bank / src1_bank = data_bank of the
#             DSChain feeding that slot (Stage 2 gave us the
#             wmma_chain<->DSChain edge).
#
#   Phase 5.  DSChain.addr_bank = the src0_bank of the wmmaChain that
#             lives in the same region as the ds_load.  Makes the
#             ds_load's addr operand agree with the region's single
#             MSB state.
#
# Prologue / epilogue ds_loads and wmmas are handled implicitly: the
# DSChain is program-wide, so a prologue ds_load that feeds a loop
# wmma is part of the same DSChain and inherits its data_bank;
# similarly, an epilogue wmma that reads a loop accumulator is on the
# same WMMAChain and inherits its acc_bank.


@dataclass
class BankAssignment:
    """Result of :func:`assign_banks`.

    All the per-chain dicts are keyed by ``id(chain)`` so plain object
    identity works without having to make the chain dataclasses
    hashable.  Use the helper accessors for readability.
    """
    wmma_acc_bank: dict[int, int] = field(default_factory=dict)
    wmma_src0_bank: dict[int, int] = field(default_factory=dict)
    wmma_src1_bank: dict[int, int] = field(default_factory=dict)
    ds_data_bank: dict[int, int] = field(default_factory=dict)
    ds_addr_bank: dict[int, int] = field(default_factory=dict)
    # Per-loop-region expected (dst, src0, src1, src2) MSB tuple.
    region_msb: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    # Human-readable messages for inconsistencies the algorithm hit.
    conflicts: list[str] = field(default_factory=list)

    def acc_bank(self, chain: "WMMAChain") -> Optional[int]:
        return self.wmma_acc_bank.get(id(chain))

    def src_bank(self, chain: "WMMAChain", slot: int) -> Optional[int]:
        if slot == 1:
            return self.wmma_src0_bank.get(id(chain))
        if slot == 2:
            return self.wmma_src1_bank.get(id(chain))
        if slot == 3:
            return self.wmma_acc_bank.get(id(chain))
        return None

    def data_bank(self, chain: "DSChain") -> Optional[int]:
        return self.ds_data_bank.get(id(chain))

    def addr_bank(self, chain: "DSChain") -> Optional[int]:
        return self.ds_addr_bank.get(id(chain))


def _loop_region_of(inst: Instruction, loop_bb: BasicBlock,
                    cbranch_idx: int) -> Optional[int]:
    """Return the loop-body region index of ``inst`` (before the
    closing ``s_cbranch``), or None if the instruction is in the
    epilogue, in a different basic block, or not under a region
    marker."""
    if inst.parent_bb is not loop_bb:
        return None
    if inst.region_idx is None or inst.region_is_epilogue:
        return None
    if inst.index > cbranch_idx:
        return None
    return inst.region_idx


def assign_banks(program: Program) -> BankAssignment:
    """Compute per-chain bank assignments as described above.  Returns
    an empty :class:`BankAssignment` if the program has no
    self-branching loop."""
    annotate_regions(program)
    wchains = collect_wmma_chains(program)
    dchains = collect_ds_chains(program)

    result = BankAssignment()

    loop_range = _loop_body_range(program)
    if loop_range is None:
        return result
    loop_bb, cbranch_idx = loop_range

    # Group WMMAChains by the loop regions they appear in.  Visiting
    # regions in order lets the user's "first region wins" rule play
    # out deterministically.
    from collections import defaultdict
    loop_wmmas_per_region: dict[int, list[WMMAChain]] = defaultdict(list)
    for c in wchains:
        seen_regions = set()
        for w in c.wmmas:
            r = _loop_region_of(w, loop_bb, cbranch_idx)
            if r is not None and r not in seen_regions:
                loop_wmmas_per_region[r].append(c)
                seen_regions.add(r)

    # Phase 2: acc_bank per WMMAChain.
    for region_idx in sorted(loop_wmmas_per_region):
        bank = region_idx % 4
        for chain in loop_wmmas_per_region[region_idx]:
            if id(chain) not in result.wmma_acc_bank:
                result.wmma_acc_bank[id(chain)] = bank

    # Verify every loop region's chains agree on acc_bank (i.e., the
    # user's pipelined-chain assumption holds).  Non-fatal: we record
    # the conflict and keep going.
    for region_idx, chains in loop_wmmas_per_region.items():
        banks = {result.wmma_acc_bank[id(c)] for c in chains}
        if len(banks) > 1:
            result.conflicts.append(
                f'L{region_idx}: wmmaChains span acc banks {sorted(banks)}')

    # Phase 3: ds_load data_bank = acc_bank of the DSChain's loading
    # region.  Each DSChain is per-region under the new structure, so
    # the lookup is direct.  Epilogue DSChains don't get banked here --
    # they'll inherit via dependency propagation in Stage 4.
    for dc in dchains:
        if dc.is_epilogue_region:
            continue
        hosts = loop_wmmas_per_region.get(dc.loading_region, [])
        if not hosts:
            continue
        bank = result.wmma_acc_bank.get(id(hosts[0]))
        if bank is not None:
            result.ds_data_bank[id(dc)] = bank

    # Phase 4: per-region src0_bank / src1_bank.  A wmmaChain can span
    # multiple loop regions (e.g., a pipelined accumulator in regions
    # N and N+4), and each of its wmmas reads different tiles loaded
    # by different ds_loads at different scheduled positions.  So
    # src_bank is not a chain-level property -- it's a *region-level*
    # property shared by all wmmas in the region.  We find the
    # reaching (most-recent) ds_load def for each wmma's src0 and
    # src1, and aggregate its bank per region.  A well-scheduled
    # pipelined loop has one reaching-def region per src slot per
    # loop region.
    pindex = build_program_def_use_index(program)
    block_order = {bb.name: i for i, bb in enumerate(program.blocks)}

    def _reaching_load_bank(wmma: Instruction, slot: int) -> Optional[int]:
        if slot >= len(wmma.operands) or not wmma.operands[slot].regs:
            return None
        reg = wmma.operands[slot].regs[0]
        rid = next(iter(_iter_regs([reg])), None)
        if rid is None:
            return None
        wmma_key = _program_order(wmma, block_order)
        best = None
        best_key = None
        for d in pindex.defs.get(rid, []):
            if not _is_ds_load(d):
                continue
            dkey = _program_order(d, block_order)
            if dkey >= wmma_key:
                continue
            if best is None or dkey > best_key:
                best = d
                best_key = dkey
        if best is None:
            return None
        r = _loop_region_of(best, loop_bb, cbranch_idx)
        if r is None:
            return None
        hosts = loop_wmmas_per_region.get(r, [])
        if not hosts:
            return None
        return result.wmma_acc_bank.get(id(hosts[0]))

    # Per-region src0/src1 banks -- used both for the region_msb table
    # and, aggregated, to back-fill WMMAChain-level src banks for
    # chains whose wmmas all agree.
    region_src_banks: dict[int, tuple[set[int], set[int]]] = {
        r: (set(), set()) for r in loop_wmmas_per_region
    }
    # Track per-chain-per-slot banks across regions so we can summarize
    # to BankAssignment.wmma_src0_bank / src1_bank.
    chain_slot_banks: dict[tuple[int, int], set[int]] = defaultdict(set)
    for r, chains in loop_wmmas_per_region.items():
        s0, s1 = region_src_banks[r]
        for c in chains:
            for w in c.wmmas:
                if _loop_region_of(w, loop_bb, cbranch_idx) != r:
                    continue
                b0 = _reaching_load_bank(w, 1)
                b1 = _reaching_load_bank(w, 2)
                if b0 is not None:
                    s0.add(b0)
                    chain_slot_banks[(id(c), 1)].add(b0)
                if b1 is not None:
                    s1.add(b1)
                    chain_slot_banks[(id(c), 2)].add(b1)

    for r, (s0, s1) in region_src_banks.items():
        if len(s0) > 1:
            result.conflicts.append(
                f'L{r}: src0 spans banks {sorted(s0)} (can\'t unify MSB)')
        if len(s1) > 1:
            result.conflicts.append(
                f'L{r}: src1 spans banks {sorted(s1)} (can\'t unify MSB)')

    for (cid, slot), banks in chain_slot_banks.items():
        if len(banks) == 1:
            target = result.wmma_src0_bank if slot == 1 else result.wmma_src1_bank
            target[cid] = banks.pop()

    # Phase 5: ds_load addr_bank = src0_bank of the wmmaChain in the
    # DSChain's loading region.  (src0 because ds_load's operand[1]
    # is encoded in the src0 MSB slot, matching the wmma's src0.)
    for dc in dchains:
        if dc.is_epilogue_region:
            continue
        addr_banks: set[int] = set()
        for c in loop_wmmas_per_region.get(dc.loading_region, []):
            b = result.wmma_src0_bank.get(id(c))
            if b is not None:
                addr_banks.add(b)
        if len(addr_banks) == 1:
            result.ds_addr_bank[id(dc)] = addr_banks.pop()
        elif len(addr_banks) > 1:
            result.conflicts.append(
                f'DSChain L{dc.loading_region} src{(dc.op_idx or 1) - 1}: '
                f'addr_bank ambiguous {sorted(addr_banks)}')

    # Per-region MSB state.  dst and src2 share a bank (acc == dst for
    # the wmma, and ds_load's dst sits in that bank by Phase 3); src0
    # and src1 come from the chain-level assignment so the first-
    # iteration region (whose wmmas' reaching defs are prologue
    # ds_loads with no loop region) inherits the steady-state bank
    # from the chain's other loop regions.
    for region_idx, chains in sorted(loop_wmmas_per_region.items()):
        if not chains:
            continue
        dst = result.wmma_acc_bank.get(id(chains[0]), 0)
        s0 = {result.wmma_src0_bank[id(c)] for c in chains
              if id(c) in result.wmma_src0_bank}
        s1 = {result.wmma_src1_bank[id(c)] for c in chains
              if id(c) in result.wmma_src1_bank}
        src0 = next(iter(s0)) if len(s0) == 1 else 0
        src1 = next(iter(s1)) if len(s1) == 1 else 0
        if len(s0) > 1 or len(s1) > 1:
            result.conflicts.append(
                f'L{region_idx}: chains disagree on src banks '
                f'(src0={sorted(s0)}, src1={sorted(s1)})')
        result.region_msb[region_idx] = (dst, src0, src1, dst)

    return result


# -------------------------------------------------------------------------
# Stage 4.3: bank-scoped VGPR allocator
# -------------------------------------------------------------------------
#
# Pure planner: takes Stage 3's BankAssignment and Stage 4.2's
# DSChain.lifetime info, produces a per-chain mapping from old role
# to new logical-VGPR Register without modifying the program.
# Stage 4.4+5 consumes this mapping to do the actual rewrite.
#
# Layout per bank (each bank covers 256 logical VGPRs):
#
#     [accumulators][shared data tiles][addr regs]
#
# Allocation order:
#
#   1.  WMMAChain.acc -- 8 contiguous VGPRs per chain in acc_bank.
#       Sorted by canonical for determinism.  Banks fill from the
#       bottom of their 256-VGPR range up.
#
#   2.  DSGroup.data -- the chain-by-chain logic the user described:
#       within each data_bank, pack DSChains into "tracks" using
#       interval scheduling.  Two chains can share a track iff their
#       data lifetimes are disjoint (Stage 4.2's
#       can_share_data_vgprs).  Each track gets enough VGPRs for the
#       largest chain in the track (8 VGPRs per DSGroup tile, summed
#       over tiles).  Within a track every chain's DSGroups get
#       mapped to the track's tile slots in canonical order (sorted
#       by original tile start).  Stride-4 sibling pairs in v9 thus
#       share the same 64 data VGPRs; v10's L1/L5 fall in different
#       tracks and consume two 64-VGPR slots in bank 1.
#
#   3.  DSChain.addr -- 1 VGPR per loop DSChain in addr_bank.  Each
#       DSChain gets its own (no sharing); regions whose current addr
#       is already a hoisted v_add will have that v_add re-targeted,
#       and regions sharing v642 will get fresh preheader copies in
#       Stage 4.4+5.
#
# Epilogue DSChains and DSChains without a Stage-3 bank are skipped
# (they inherit via dependency propagation when the rewrite walks
# them).  Prologue ds_loads attached to a DSGroup share that
# DSGroup's data Register automatically.


@dataclass
class VGPRAllocation:
    """Output of :func:`allocate_vgprs`.  All maps are keyed by
    ``id(chain)`` / ``id(group)`` so we don't need the chain
    dataclasses to be hashable."""
    wmma_acc: dict[int, Register] = field(default_factory=dict)
    ds_group_data: dict[int, Register] = field(default_factory=dict)
    ds_chain_addr: dict[int, Register] = field(default_factory=dict)
    # Per-bank highest-allocated logical VGPR id + 1, useful for the
    # descriptor-bump in Stage 4.4+5.
    bank_high_water: dict[int, int] = field(default_factory=dict)
    # ``budget`` = max(bank_high_water.values()), the new
    # ``.amdhsa_next_free_vgpr`` value.
    budget: int = 0

    def acc(self, chain: "WMMAChain") -> Optional[Register]:
        return self.wmma_acc.get(id(chain))

    def data(self, group: "DSGroup") -> Optional[Register]:
        return self.ds_group_data.get(id(group))

    def addr(self, chain: "DSChain") -> Optional[Register]:
        return self.ds_chain_addr.get(id(chain))


def _make_logical_register(start: int, size: int) -> Register:
    """Build a logical-VGPR Register at logical id ``start`` covering
    ``size`` VGPRs.  Computes the raw_id automatically (the same
    register's raw form under the bank-implied MSB)."""
    bank = start // 256
    raw_start = start - bank * 256
    return Register(
        kind='v',
        ids=list(range(start, start + size)),
        raw_ids=list(range(raw_start, raw_start + size)),
    )


def allocate_vgprs(program: Program,
                   ba: BankAssignment) -> VGPRAllocation:
    """Decide a target logical VGPR for every chain role.  No rewrite."""
    wchains = collect_wmma_chains(program)
    dchains = collect_ds_chains(program)

    alloc = VGPRAllocation()
    # Each bank has 256 logical VGPRs at offsets [bank*256, bank*256+256).
    bank_next: dict[int, int] = {b: b * 256 for b in range(4)}

    # Phase 1: WMMAChain accumulators.
    sorted_wchains = sorted(wchains, key=lambda c: (
        ba.acc_bank(c) if ba.acc_bank(c) is not None else 99,
        c.canonical.start,
    ))
    for c in sorted_wchains:
        bank = ba.acc_bank(c)
        if bank is None:
            continue  # epilogue-only chain; not in BankAssignment
        size = c.canonical.size
        start = bank_next[bank]
        if start + size > (bank + 1) * 256:
            raise ValueError(
                f'WMMAChain {c.canonical} (bank {bank}) overflows: '
                f'allocated {start - bank * 256 + size} > 256')
        alloc.wmma_acc[id(c)] = _make_logical_register(start, size)
        bank_next[bank] = start + size

    # Phase 2: DSGroup data.  Pack DSChains into tracks per bank using
    # interval scheduling; each track shares one VGPR pool.
    from collections import defaultdict as _defaultdict
    chains_by_data_bank: dict[int, list["DSChain"]] = _defaultdict(list)
    for c in dchains:
        if c.is_epilogue_region:
            continue
        b = ba.data_bank(c)
        if b is None:
            continue
        chains_by_data_bank[b].append(c)

    for bank, chains_in_bank in chains_by_data_bank.items():
        chains_sorted = sorted(
            chains_in_bank,
            key=lambda c: c.loading_pos if c.loading_pos else (0, 0),
        )
        # Greedy interval-graph coloring -- track[i] = list of chains
        # sharing the same VGPR pool.  Two chains can share iff their
        # lifetimes are disjoint per Stage 4.2.
        tracks: list[list["DSChain"]] = []
        for c in chains_sorted:
            placed = False
            for tr in tracks:
                if all(can_share_data_vgprs(c, other) for other in tr):
                    tr.append(c)
                    placed = True
                    break
            if not placed:
                tracks.append([c])

        # Allocate per-track tile pool.  Each DSGroup is one tile
        # (8 VGPRs in v9/v10).  All chains in a track share the pool;
        # their DSGroups get mapped to consecutive new tiles in their
        # original-tile-id order.
        for tr in tracks:
            max_groups = max(len(c.dsgroups) for c in tr)
            tile_size = max((g.tile.size for c in tr for g in c.dsgroups
                             if g.tile is not None), default=0)
            if tile_size == 0:
                continue
            track_start = bank_next[bank]
            new_tile_starts = [track_start + i * tile_size
                               for i in range(max_groups)]
            track_total = max_groups * tile_size
            if track_start + track_total > (bank + 1) * 256:
                raise ValueError(
                    f'DSChain track in bank {bank} overflows: '
                    f'{track_start - bank * 256 + track_total} > 256')
            bank_next[bank] = track_start + track_total

            for c in tr:
                groups_sorted = sorted(
                    c.dsgroups,
                    key=lambda g: g.tile.ids[0] if g.tile else 0,
                )
                for i, g in enumerate(groups_sorted):
                    new_start = new_tile_starts[i]
                    alloc.ds_group_data[id(g)] = _make_logical_register(
                        new_start, tile_size)

    # Phase 3: DSChain addrs (1 VGPR each in addr_bank).
    sorted_addr_chains = sorted(
        (c for c in dchains
         if not c.is_epilogue_region and ba.addr_bank(c) is not None),
        key=lambda c: (ba.addr_bank(c), c.loading_region),
    )
    for c in sorted_addr_chains:
        bank = ba.addr_bank(c)
        start = bank_next[bank]
        if start + 1 > (bank + 1) * 256:
            raise ValueError(
                f'DSChain L{c.loading_region} addr (bank {bank}) overflows')
        alloc.ds_chain_addr[id(c)] = _make_logical_register(start, 1)
        bank_next[bank] = start + 1

    alloc.bank_high_water = dict(bank_next)
    # Budget = highest used logical VGPR id + 1.  Banks that received
    # no allocations don't contribute (their bank_next still equals
    # bank * 256, the empty-bank starting point).
    used = [v for b, v in bank_next.items() if v > b * 256]
    alloc.budget = max(used) if used else 0
    return alloc


# -------------------------------------------------------------------------
# LICM: hoist loop-invariant address computations out of the loop
# -------------------------------------------------------------------------

def _encode_msb_byte(dst: int, src0: int, src1: int, src2: int) -> int:
    """Encode per-operand MSB values into the low byte of an
    ``s_set_vgpr_msb`` immediate (new-state byte)."""
    return ((src0 & 3) |
            ((src1 & 3) << 2) |
            ((src2 & 3) << 4) |
            ((dst & 3) << 6))


def _find_self_loop(program: Program) -> Optional[BasicBlock]:
    """Return the BasicBlock that ends with an ``s_cbranch`` back to
    itself (the loop body), or None."""
    for bb in program.blocks:
        for inst in bb.instructions:
            if not inst.opcode.startswith('s_cbranch'):
                continue
            # The branch target is in the last operand's text.
            tgt = inst.operands[-1].text if inst.operands else ""
            if tgt.strip() == bb.name:
                return bb
    return None


def _find_preheader(program: Program,
                    loop_bb: BasicBlock) -> Optional[BasicBlock]:
    """The preheader is the BB immediately preceding ``loop_bb`` in
    program order (fall-through entry to the loop)."""
    try:
        idx = program.blocks.index(loop_bb)
    except ValueError:
        return None
    if idx == 0:
        return None
    return program.blocks[idx - 1]


def _is_integer_literal(text: str) -> bool:
    try:
        int(text.strip(), 0)
        return True
    except ValueError:
        return False


def _collect_trivial_copies(
    loop_bb: BasicBlock,
) -> dict[tuple[str, int], tuple[Instruction, Register]]:
    """Find ``v_add_nc_u32_e32 dst, 0, src_vgpr`` instructions in the
    loop -- these add zero and behave as moves.  Returns a map from the
    destination's ``(kind, logical_id)`` to ``(copy_inst, src_reg)``."""
    copies: dict[tuple[str, int], tuple[Instruction, Register]] = {}
    for inst in loop_bb.instructions:
        if inst.opcode != 'v_add_nc_u32_e32':
            continue
        if len(inst.operands) < 3:
            continue
        if inst.operands[1].text.strip() != '0':
            continue
        if not inst.operands[0].regs or not inst.operands[2].regs:
            continue
        dst = inst.operands[0].regs[0]
        src = inst.operands[2].regs[0]
        copies[(dst.kind, dst.ids[0])] = (inst, src)
    return copies


def _collect_used_logical_vgprs(program: Program) -> set[int]:
    """All logical VGPR ids mentioned by any operand in the program."""
    used: set[int] = set()
    for inst in program.iter_instructions():
        for op in inst.operands:
            for r in op.regs:
                if r.kind == 'v':
                    used.update(r.ids)
    return used


_VGPR_BUDGET_RE = re.compile(r'(\.amdhsa_next_free_vgpr\s+)(\d+)')
_VGPR_COUNT_RE = re.compile(r'(\.vgpr_count:\s*)(\d+)')


def _vgpr_budget(program: Program) -> int:
    """The kernel-declared logical VGPR budget (one past the highest
    addressable VGPR).  Read from ``.amdhsa_next_free_vgpr`` in the
    kernel descriptor (which usually lives in ``tail_lines`` after the
    function body).  Defaults to 1024 if the directive isn't present."""
    for line_list in (program.header_lines, program.tail_lines):
        for line in line_list:
            m = _VGPR_BUDGET_RE.search(line)
            if m:
                return int(m.group(2))
    for bb in program.blocks:
        for inst in bb.instructions:
            m = _VGPR_BUDGET_RE.search(inst.raw_line)
            if m:
                return int(m.group(2))
    return 1024


def _set_vgpr_budget(program: Program, new_budget: int) -> None:
    """Update both ``.amdhsa_next_free_vgpr`` and ``.vgpr_count`` lines
    to ``new_budget`` so the kernel descriptor matches the actual
    logical VGPR usage after hoisting."""

    def _patch(line: str) -> str:
        line = _VGPR_BUDGET_RE.sub(rf'\g<1>{new_budget}', line)
        line = _VGPR_COUNT_RE.sub(rf'\g<1>{new_budget}', line)
        return line

    program.header_lines = [_patch(ln) for ln in program.header_lines]
    program.tail_lines = [_patch(ln) for ln in program.tail_lines]
    for bb in program.blocks:
        for inst in bb.instructions:
            patched = _patch(inst.raw_line)
            if patched != inst.raw_line:
                inst.raw_line = patched


def _allocate_unused_vgpr(used: set[int],
                          budget: Optional[int] = None) -> Optional[int]:
    """Find an unused logical VGPR id, mark it used, and return it.

    Search order:
      1. Slots within ``budget`` (top-down, so tile-data ranges near
         the bottom aren't disturbed).
      2. Slots beyond ``budget`` (bottom-up, so any descriptor bump
         stays as small as possible).

    Returns None if every slot in [0, 1024) is already taken.

    Bank choice doesn't matter at this stage: bank assignment is
    handled in a later stage and the current LLVM-emitted layout is
    already sub-optimal.  We just need ANY free VGPR so the v_add can
    be hoisted out of the loop and break the dual-role assignment."""
    upper = budget if budget is not None else 1024
    for v in range(upper - 1, -1, -1):
        if v not in used:
            used.add(v)
            return v
    if budget is not None and budget < 1024:
        for v in range(budget, 1024):
            if v not in used:
                used.add(v)
                return v
    return None


def _rename_single_vgpr(op: Operand, old_logical: int, old_raw: int,
                        new_raw: int, new_logical: int) -> bool:
    """If ``op`` references a single-reg VGPR with PHYSICAL id
    ``old_logical`` (encoded as raw ``old_raw`` plus the implied MSB),
    rewrite it to ``new_raw`` / ``new_logical``.  Updates ``op.text``,
    ``op.logical_text`` and ``op.regs``.  Returns True if a rename
    occurred.

    Matching on logical (physical) id rather than raw is essential:
    different MSB contexts make the same raw id refer to different
    physical registers, and we must only rewrite the consumers that
    actually read the renamed register's physical bank."""
    pattern = re.compile(rf'\bv{old_raw}\b')
    modified = False
    for i, r in enumerate(op.regs):
        if (r.kind == 'v' and len(r.ids) == 1 and
                len(r.raw_ids) == 1 and r.ids[0] == old_logical):
            op.regs[i] = Register(kind='v', ids=[new_logical],
                                  raw_ids=[new_raw])
            modified = True
    if not modified:
        return False
    op.text = pattern.sub(f'v{new_raw}', op.text)
    if op.logical_text is not None:
        op.logical_text = f'/*v{new_logical}*/'
    return True


def _rebuild_raw_line(inst: Instruction) -> None:
    """Regenerate ``inst.raw_line`` from current opcode/operands.  Used
    after an operand has been rewritten in place so the emitter picks
    up the change."""
    if inst.dual_issue and len(inst.operands) == 4:
        left = ", ".join(op.emit() for op in inst.operands[:2])
        right = ", ".join(op.emit() for op in inst.operands[2:])
        line = f"\t{inst.opcode} {left} :: {inst.opcode} {right}"
    else:
        parts = [inst.opcode]
        if inst.operands:
            parts.append(", ".join(op.emit() for op in inst.operands))
        line = "\t" + " ".join(parts)
    if inst.trailing_comment is not None:
        line = f"{line:<40} ;{inst.trailing_comment}"
    inst.raw_line = line


def _adjust_msbs_for_renames(loop_bb: BasicBlock) -> bool:
    """Walk each ``s_set_vgpr_msb`` in the loop and update its src0 (and
    dst) bank fields to match the actual register usage of the
    consumers in its scope.  This catches MSB drift caused by VGPR
    renames where the consumer now reads from a different bank than
    the LLVM-emitted ``s_set_vgpr_msb`` was originally set to.

    For each MSB scope ``[m, next_m)``:
      - Collect required dst/src0 banks from ds_load consumers in that
        scope (operand[0]=dst, operand[1]=addr/src0).
      - If consumers disagree on a field, the LLVM-emitted MSB couldn't
        cover them collectively after rename; we leave the field alone
        on the assumption the next-stage bank assignment will rebuild
        MSBs from scratch.

    Returns True (always — left as boolean for future conflict signaling).
    """
    msb_positions = [(i, inst) for i, inst in enumerate(loop_bb.instructions)
                     if inst.opcode == 's_set_vgpr_msb' and inst.msb_bits]
    for k, (mi, msb_inst) in enumerate(msb_positions):
        scope_end = (msb_positions[k + 1][0]
                     if k + 1 < len(msb_positions)
                     else len(loop_bb.instructions))

        required_dst: set[int] = set()
        required_src0: set[int] = set()
        for j in range(mi + 1, scope_end):
            inst = loop_bb.instructions[j]
            if not _is_ds_load(inst):
                continue
            if inst.operands and inst.operands[0].regs:
                r = inst.operands[0].regs[0]
                if r.kind == 'v':
                    required_dst.add(r.msb())
            if len(inst.operands) >= 2 and inst.operands[1].regs:
                r = inst.operands[1].regs[0]
                if r.kind == 'v' and len(r.ids) == 1:
                    required_src0.add(r.msb())

        d, s0, s1, s2 = msb_inst.msb_bits
        new_d = required_dst.pop() if len(required_dst) == 1 else d
        new_s0 = required_src0.pop() if len(required_src0) == 1 else s0
        if (new_d, new_s0) == (d, s0):
            continue

        new_state = (new_d, new_s0, s1, s2)
        new_low = _encode_msb_byte(*new_state)
        try:
            original_imm = int(msb_inst.operands[0].text, 0)
        except ValueError:
            original_imm = 0
        high_byte = (original_imm >> 8) & 0xff
        new_imm = (high_byte << 8) | new_low
        msb_inst.operands[0].text = f"{new_imm:#x}"
        msb_inst.msb_bits = new_state
        if msb_inst.trailing_comment is not None:
            msb_inst.trailing_comment = (f"  msbs: dst={new_d} src0={new_s0} "
                                         f"src1={s1} src2={s2}")
        _rebuild_raw_line(msb_inst)
    return True


def _rename_live_range(inst: Instruction, loop_bb: BasicBlock,
                       program: Program, old_logical: int,
                       old_raw: int, new_raw: int, new_logical: int) -> None:
    """Rename single-reg READs of physical ``old_logical`` (encoded as
    raw ``old_raw`` + implicit MSB) to ``new_raw`` / ``new_logical``,
    starting just after ``inst`` (the candidate v_add) and continuing
    through the rest of the loop body AND any subsequent basic blocks
    (epilogue).

    Stops as soon as another instruction DEFINES physical
    ``old_logical`` (its operand[0] range includes that physical id) --
    that def ends the v_add's live range and later reads target the
    new def's value.

    Both the rename match and the kill check use the parsed Register's
    *logical* (= physical) ids, not raw ids.  Without that, a wmma
    writing raw v[128:135] with dst MSB=0 (physical v[128:135]) would
    look like a kill of physical v646 just because raw v134 falls in
    the range -- but they're entirely different physical registers.

    The epilogue extension is essential: the hoisted v_add's value
    persists past the loop's back-edge, and the LLVM-emitted epilogue
    typically reads the value the in-loop v_add left in its dst on the
    final iteration."""
    try:
        loop_idx_in_prog = program.blocks.index(loop_bb)
    except ValueError:
        loop_idx_in_prog = -1
    try:
        start_pos = loop_bb.instructions.index(inst)
    except ValueError:
        return

    bbs = [loop_bb]
    if loop_idx_in_prog >= 0:
        bbs.extend(program.blocks[loop_idx_in_prog + 1:])

    for bi, bb in enumerate(bbs):
        first = start_pos + 1 if bi == 0 else 0
        for i in range(first, len(bb.instructions)):
            other = bb.instructions[i]
            if not other.opcode or other.opcode == '__asm_block__':
                continue
            modified = False
            for op in other.operands[1:]:
                if _rename_single_vgpr(op, old_logical, old_raw,
                                       new_raw, new_logical):
                    modified = True
            if modified:
                _rebuild_raw_line(other)
            if other.operands:
                for r in other.operands[0].regs:
                    if (r.kind == 'v' and r.ids and
                            r.ids[0] <= old_logical <= r.ids[-1]):
                        return


def _resolve_to_loop_invariant(
    reg: Register,
    loop_bb: BasicBlock,
    pindex: DefUseIndex,
    copies: dict[tuple[str, int], tuple[Instruction, Register]],
) -> Optional[Register]:
    """Follow trivial copies inside ``loop_bb`` to trace ``reg`` back to
    a definition that lives outside the loop.  Returns the
    outside-loop-defined ``Register`` if reachable, else None."""
    visited: set[tuple[str, int]] = set()
    current = reg
    for _ in range(8):  # guard against pathological chains
        key = (current.kind, current.ids[0])
        if key in visited:
            return None
        visited.add(key)
        defs_all = pindex.defs.get(key, [])
        defs_in = [d for d in defs_all if d.parent_bb is loop_bb]
        if not defs_in:
            return current
        if len(defs_in) != 1:
            return None
        if key not in copies:
            return None
        _, src = copies[key]
        current = src
    return None


def hoist_loop_invariant_addrs(program: Program) -> int:
    """Hoist loop-invariant v_add_nc_u32_e32 address computations out of
    the loop into the preheader.

    Targets the pattern emitted by the AMDGPU backend when
    ``MachineLICM`` declines to hoist due to register pressure::

        LOOP:
            v_add_nc_u32_e32  <copy_dst>, 0, <prologue_vgpr>
            ...
            v_add_nc_u32_e32  <addr_dst>, <const_imm>, <copy_dst>
            ...
            ds_load_b128 ..., <addr_dst>

    After hoisting the 4 address v_adds (and their shared trivial copy)
    into the preheader, the loop body contains only the ds_loads.  The
    hoisted v_adds each write a distinct destination and all read the
    prologue-defined root register, so no intra-group VALU hazards
    remain and no ``s_wait_alu`` / ``s_delay_alu`` is required between
    them.

    Returns the number of instructions hoisted (ds_load addr computations
    plus the trivial copies they needed).
    """
    loop_bb = _find_self_loop(program)
    if loop_bb is None:
        return 0
    preheader_bb = _find_preheader(program, loop_bb)
    if preheader_bb is None:
        return 0

    pindex = build_program_def_use_index(program)
    loop_idx = build_def_use_index(loop_bb)
    copies = _collect_trivial_copies(loop_bb)

    # Gather (inst, resolved_src_reg, needs_rename) for every
    # v_add_nc_u32_e32 with (literal, vreg) source operands where the
    # vreg resolves to a loop-invariant root.
    #
    # ``needs_rename=True`` means the dst is ALSO written by another
    # instruction in the loop (e.g., a ds_load writing a data range
    # that aliases the addr register).  We break that dual role by
    # allocating a fresh VGPR in the same bank for the hoisted v_add
    # and renaming its downstream address-use consumers.
    candidates: list[tuple[Instruction, Register, bool]] = []
    for inst in loop_bb.instructions:
        if inst.opcode != 'v_add_nc_u32_e32':
            continue
        if len(inst.operands) < 3:
            continue
        # Skip trivial copies themselves -- they're handled via the
        # copies map.
        if inst.operands[1].text.strip() == '0':
            continue
        if not _is_integer_literal(inst.operands[1].text):
            continue
        if not inst.operands[0].regs or not inst.operands[2].regs:
            continue
        dst = inst.operands[0].regs[0]
        src = inst.operands[2].regs[0]
        dst_defs = loop_idx.defs.get((dst.kind, dst.ids[0]), [])
        needs_rename = bool([d for d in dst_defs if d is not inst])
        root = _resolve_to_loop_invariant(src, loop_bb, pindex, copies)
        if root is None:
            continue
        candidates.append((inst, root, needs_rename))

    if not candidates:
        return 0

    # Determine, BEFORE renaming, which trivial copies will be dead
    # after hoist (consumers were all candidate v_adds, all going
    # away).  We need this up front so the allocator can reuse those
    # dst slots and the post-rename consumer check doesn't get fooled
    # by the just-renamed reads.
    candidate_inst_ids = {id(inst) for inst, _, _ in candidates}
    removable_copies: set[int] = set()
    for copy_key, (copy_inst, _) in copies.items():
        has_other_consumer = False
        for other in loop_bb.instructions:
            if other is copy_inst or id(other) in candidate_inst_ids:
                continue
            for op in other.operands[1:]:
                for r in op.regs:
                    if (r.kind, r.ids[0]) == copy_key:
                        has_other_consumer = True
                        break
                if has_other_consumer:
                    break
            if has_other_consumer:
                break
        if not has_other_consumer:
            removable_copies.add(id(copy_inst))

    # Allocate fresh VGPRs for each renamed candidate from any bank
    # (bank assignment is a separate, later stage; here we just need
    # any free VGPR so the v_add can be hoisted and the dual-role
    # assignment broken).  Discard slots about to be vacated by
    # removable trivial copies so the allocator can recycle them.
    used_vgprs = _collect_used_logical_vgprs(program)
    for copy_key, (copy_inst, _) in copies.items():
        if id(copy_inst) in removable_copies:
            used_vgprs.discard(copy_key[1])
    original_budget = _vgpr_budget(program)
    kept: list[tuple[Instruction, Register, bool]] = []
    new_dsts: list[tuple[int, int]] = []  # (raw, logical) per kept candidate
    for inst, root, needs_rename in candidates:
        old = inst.operands[0].regs[0]
        if not needs_rename:
            kept.append((inst, root, False))
            new_dsts.append((old.raw_ids[0], old.ids[0]))
            continue
        new_log = _allocate_unused_vgpr(used_vgprs, budget=original_budget)
        if new_log is None:
            continue
        new_raw = new_log - (new_log // 256) * 256
        kept.append((inst, root, True))
        new_dsts.append((new_raw, new_log))
        _rename_live_range(inst, loop_bb, program,
                           old_logical=old.ids[0],
                           old_raw=old.raw_ids[0],
                           new_raw=new_raw, new_logical=new_log)

    if not kept:
        return 0
    candidates = kept

    # If we allocated VGPRs above the kernel's declared budget, bump
    # both ``.amdhsa_next_free_vgpr`` and ``.vgpr_count`` so the
    # runtime allocates enough physical VGPRs to back our new logicals.
    max_logical = max(log for _, log in new_dsts) if new_dsts else 0
    if max_logical >= original_budget:
        _set_vgpr_budget(program, max_logical + 1)

    # Sort candidates so same-(dst_msb, src_msb) groups are adjacent --
    # lets us emit one ``s_set_vgpr_msb`` per group instead of one per
    # v_add.
    order = sorted(range(len(candidates)),
                   key=lambda i: (new_dsts[i][1] // 256,
                                  candidates[i][1].msb()))
    candidates = [candidates[i] for i in order]
    new_dsts = [new_dsts[i] for i in order]

    # Build the raw_line for each hoisted instruction.  The source
    # register is the loop-invariant ``root`` (forwarded through any
    # trivial copy).  Emit a fresh ``s_set_vgpr_msb`` whenever the
    # required (dst, src1) bank pair changes between candidates.
    hoisted_lines: list[str] = []
    last_state: Optional[tuple[int, int, int, int]] = None
    for (inst, root, _), (new_raw, new_log) in zip(candidates, new_dsts):
        dst_msb = new_log // 256
        src_msb = root.msb()
        state = (dst_msb, 0, src_msb, 0)
        if state != last_state:
            new_imm = _encode_msb_byte(*state)
            hoisted_lines.append(f"\ts_set_vgpr_msb {new_imm:#x}")
            last_state = state
        imm_text = inst.operands[1].text.strip()
        hoisted_lines.append(
            f"\tv_add_nc_u32_e32 v{new_raw} /*v{new_log}*/, "
            f"{imm_text}, v{root.raw_ids[0]} /*v{root.ids[0]}*/"
        )

    # Insert the new instructions into the preheader, just before the
    # last ``s_set_vgpr_msb`` (which resets state to all zeros before the
    # loop entry).  If no such instruction exists, append to the end.
    insert_idx = len(preheader_bb.instructions)
    for i in range(len(preheader_bb.instructions) - 1, -1, -1):
        if preheader_bb.instructions[i].opcode == 's_set_vgpr_msb':
            insert_idx = i
            break
    new_insts = [_parse_instruction_line(ln) for ln in hoisted_lines]
    for inst in new_insts:
        inst.parent_bb = preheader_bb
    preheader_bb.instructions[insert_idx:insert_idx] = new_insts

    # Remove the original v_adds from the loop, plus the trivial
    # copies we already determined are dead after hoist.  Use object
    # identity (not hash/equality) since Instruction is a dataclass
    # that isn't frozen; lookups go through ``id()``.
    to_remove_ids: set[int] = {id(inst) for inst, _, _ in candidates}
    to_remove_ids |= removable_copies

    # Also remove the s_delay_alu / s_wait_alu sandwiching each removed
    # v_add.  LLVM emitted them to model the v_add's VALU latency
    # (delay) and to drain its VA_VDST counter (wait); with the v_add
    # gone they have no specific dependency to guard and would just
    # stall the wave waiting for whatever wmma is currently pending.
    insts = loop_bb.instructions
    n = len(insts)

    def _is_skip(i: int) -> bool:
        op = insts[i].opcode
        return not op or op == '.loc' or op == '__asm_block__'

    extra_remove: set[int] = set()
    for idx in range(n):
        if id(insts[idx]) not in to_remove_ids:
            continue
        # s_wait_alu after the removed instruction (skip .loc/empty).
        j = idx + 1
        while j < n and _is_skip(j):
            j += 1
        if j < n and insts[j].opcode == 's_wait_alu':
            extra_remove.add(id(insts[j]))
        # s_delay_alu before the removed instruction.
        j = idx - 1
        while j >= 0 and _is_skip(j):
            j -= 1
        if j >= 0 and insts[j].opcode == 's_delay_alu':
            extra_remove.add(id(insts[j]))
    to_remove_ids |= extra_remove

    loop_bb.instructions = [i for i in loop_bb.instructions
                            if id(i) not in to_remove_ids]

    # Update the LLVM-emitted s_set_vgpr_msb instructions in the loop
    # body AND the epilogue blocks so each one's src0/dst banks match
    # the (possibly renamed) consumers in its scope.  No-op when
    # nothing was renamed.
    loop_idx_in_prog = program.blocks.index(loop_bb)
    for bb in program.blocks[loop_idx_in_prog:]:
        _adjust_msbs_for_renames(bb)

    # Re-index both blocks.
    for i, inst in enumerate(preheader_bb.instructions):
        inst.index = i
    for i, inst in enumerate(loop_bb.instructions):
        inst.index = i

    return len(to_remove_ids)


# -------------------------------------------------------------------------
# Stage 4.4+5: apply allocation (rename) and regenerate MSBs
# -------------------------------------------------------------------------
#
# Atomic pass: walks the program once and rewrites every chain operand
# against the Stage 4.3 ``VGPRAllocation``, materializes preheader
# copy v_adds for shared bases (v642), repoints LICM-hoisted v_adds
# to the new addr_bank, strips all in-loop ``s_set_vgpr_msb``
# instructions, regenerates them everywhere based on actual operand
# banks, adds a final ``s_wait_alu depctr_va_vdst(0)`` before the
# loop entry to drain new VALU writes, and bumps the kernel
# descriptor to ``alloc.budget``.
#
# Phases (in order):
#   A. Build OLD->NEW translation maps from the allocation.
#   B. Repoint existing LICM-hoisted v_adds + emit preheader copy
#      v_adds for DSChains using a shared base.
#   C. Walk every instruction, rewrite operands.
#   D. Strip every existing s_set_vgpr_msb.
#   E. Walk every instruction and emit s_set_vgpr_msb instructions
#      based on actual operand banks (state-tracked).
#   F. Insert s_wait_alu depctr_va_vdst(0) just before the loop entry.
#   G. Bump kernel descriptor.


def _operand_text_for_register(reg: Register) -> tuple[str, Optional[str]]:
    """Return (raw_text, logical_text) for an operand's Register.
    ``logical_text`` is None when the logical id matches the raw id
    (i.e., bank 0, default MSB)."""
    if reg.size == 1:
        raw = f'v{reg.raw_ids[0]}'
        if reg.ids[0] == reg.raw_ids[0]:
            return raw, None
        return raw, f'/*v{reg.ids[0]}*/'
    raw = f'v[{reg.raw_ids[0]}:{reg.raw_ids[-1]}]'
    if reg.ids[0] == reg.raw_ids[0]:
        return raw, None
    return raw, f'/*v[{reg.ids[0]}:{reg.ids[-1]}]*/'


def _replace_operand_register(op: Operand, new_reg: Register) -> None:
    """Replace ``op``'s first VGPR Register with ``new_reg`` in-place.
    Updates ``op.text``, ``op.logical_text``, and ``op.regs[0]`` while
    preserving ``op.suffix`` (e.g., ``offset:32``)."""
    if not op.regs:
        return
    op.regs[0] = new_reg
    raw_text, logical_text = _operand_text_for_register(new_reg)
    op.text = raw_text
    # Preserve presence/absence of logical comment based on whether
    # the new register actually needs one (non-zero MSB).
    op.logical_text = logical_text


def _make_msb_instruction(state: tuple[int, int, int, int],
                          prev_state: tuple[int, int, int, int]) -> Instruction:
    """Build an ``s_set_vgpr_msb`` Instruction with low byte = new
    state and high byte = previous state's encoded byte (matches
    LLVM's gfx1250 emission: bits 8-15 carry the prior MSB context
    for the validity/commit tracker)."""
    new_low = _encode_msb_byte(*state)
    prev_low = _encode_msb_byte(*prev_state)
    imm = (prev_low << 8) | new_low
    return _make_msb_instruction_from_imm(imm)


def _make_msb_instruction_from_imm(imm: int) -> Instruction:
    """Build an ``s_set_vgpr_msb`` Instruction with the exact 16-bit
    immediate.  Decodes the low byte for the trailing comment."""
    low = imm & 0xff
    dst, src0, src1, src2 = _decode_msb_imm(f"{low:#x}")
    line = (f"\ts_set_vgpr_msb {imm:#x}                   "
            f";  msbs: dst={dst} src0={src0} "
            f"src1={src1} src2={src2}")
    return _parse_instruction_line(line)


# =====================================================================
# Faithful port of LLVM's AMDGPULowerVGPREncoding.cpp
# =====================================================================
#
# Slot layout matches LLVM: ``Ops = [src0, src1, src2, vdst]`` where
# each slot is an ``Optional[int]`` -- ``None`` means "no demand"
# (carry-forward from prior context); a value means "this slot must
# be at this bank for the current instruction".
#
# The hardware encoding has src0 in bits 1-0, src1 in 3-2, src2 in
# 5-4, dst in 7-6 (low byte).  High byte is the previous low byte
# (validity/commit tracker).


class _ModeTy:
    """Mirror of LLVM's ModeTy: 4 optional MSB slots."""
    __slots__ = ('ops',)

    def __init__(self, ops: Optional[list] = None):
        # ops[0]=src0, ops[1]=src1, ops[2]=src2, ops[3]=vdst.
        # Each is Optional[int] (None = no demand).
        self.ops = list(ops) if ops is not None else [None, None, None, None]

    def copy(self) -> '_ModeTy':
        return _ModeTy(self.ops)

    def update(self, new: '_ModeTy') -> tuple[bool, bool]:
        """LLVM's update: for each slot where new has a demand, write
        it into self.  Returns ``(updated, rewritten)``:
          updated: True iff any slot changed
          rewritten: True iff a slot that was previously SET is being
                     overwritten with a different value (forces a new
                     MSB emit instead of piggybacking)."""
        updated = False
        rewritten = False
        for i in range(4):
            new_v = new.ops[i]
            if new_v is None:
                continue
            cur_v = self.ops[i]
            cur_or_zero = 0 if cur_v is None else cur_v
            if new_v != cur_or_zero:
                updated = True
                if cur_v is not None:
                    rewritten = True
            self.ops[i] = new_v
        return updated, rewritten

    def is_compatible(self, new: '_ModeTy') -> bool:
        """True iff ``new``'s demands are already satisfied by self."""
        for i in range(4):
            d = new.ops[i]
            if d is None:
                continue
            cur_or_zero = 0 if self.ops[i] is None else self.ops[i]
            if d != cur_or_zero:
                return False
        return True

    def encode(self) -> int:
        """Encoded low byte (8 bits)."""
        v = 0
        for i, o in enumerate(self.ops):
            v |= (0 if o is None else (o & 3)) << (i * 2)
        return v

    def __repr__(self):
        return f"_ModeTy({self.ops})"


# ---- per-instruction-class operand -> slot mapping -----------------
#
# Returns a list of length 4 where each entry is the index into
# ``inst.operands[]`` for that MSB slot (or None if no operand maps
# to that slot for this opcode class).  Slots are
# [src0, src1, src2, vdst] matching LLVM's _ModeTy layout.

def _msb_slot_to_operand_index(inst: Instruction
                               ) -> Optional[list[Optional[int]]]:
    """Return mapping from MSB slot (0=src0, 1=src1, 2=src2, 3=vdst)
    to ``inst.operands`` index, or None if this opcode has no MSB
    encoding.  Mirrors LLVM's ``getVGPRLoweringOperandTables``.
    """
    op = inst.opcode
    if not op:
        return None

    # Dual-issue VOPD (v_dual_mov_b32 a, b :: v_dual_mov_b32 c, d).
    # Parsed as 4 operands [dst1, src1_lane, dst2, src2_lane].  X and
    # Y components must share the same dst and src0 banks; we expose
    # the X component into vdst/src0 slots.  Source operand bank
    # assignment: for a single ``v_dual_mov_b32`` lane, the source is
    # in the src0 slot.
    if inst.dual_issue and len(inst.operands) >= 4:
        # X component: inst.operands[0]=dstX, [1]=src0X.
        # Y: [2]=dstY, [3]=src0Y.  Both lanes share the same MSB
        # context -- we register both via slots 0 and 3.
        return [1, None, None, 0]

    # DS instructions (ds_load, ds_store, ds_read, ds_write):
    # VDSOps = {addr, data0, data1, vdst}.  In assembly textual order:
    #   ds_load_b*  : op[0]=vdst, op[1]=addr [, offset suffix]
    #   ds_store_b* : op[0]=addr, op[1]=data0 [, op[2]=data1 ...]
    if op.startswith('ds_'):
        if op.startswith('ds_load') or op.startswith('ds_read'):
            # vdst at op[0], addr at op[1].
            return [1 if len(inst.operands) >= 2 else None, None, None,
                    0 if inst.operands else None]
        if op.startswith('ds_store') or op.startswith('ds_write'):
            # addr at op[0], data0 at op[1], data1 at op[2] (b96+).
            slots = [None, None, None, None]
            if inst.operands:
                slots[0] = 0
            if len(inst.operands) >= 2:
                slots[1] = 1
            if len(inst.operands) >= 3:
                slots[2] = 2
            return slots
        # Other ds_ ops (ds_swizzle, ds_consume, ds_append, ...):
        # be conservative and skip.
        return None

    # Buffer / typed-buffer instructions:
    #   buffer_load_*  : op[0]=vdst,  op[1]=vaddr, op[2]=srsrc, op[3]=soffset
    #   buffer_store_* : op[0]=vdata, op[1]=vaddr, op[2]=srsrc, op[3]=soffset
    if op.startswith('buffer_') or op.startswith('tbuffer_'):
        if op.startswith('buffer_load') or op.startswith('tbuffer_load'):
            return [1 if len(inst.operands) >= 2 else None, None, None,
                    0 if inst.operands else None]
        if op.startswith('buffer_store') or op.startswith('tbuffer_store'):
            # vdata in vdst slot (the data being stored), vaddr in src0.
            return [1 if len(inst.operands) >= 2 else None, None, None,
                    0 if inst.operands else None]
        return None

    # Flat / global / scratch:
    #   flat_load_*    : op[0]=vdst,  op[1]=vaddr
    #   flat_store_*   : op[0]=vaddr, op[1]=vdata
    if (op.startswith('flat_') or op.startswith('global_')
            or op.startswith('scratch_')):
        if 'load' in op:
            return [1 if len(inst.operands) >= 2 else None, None, None,
                    0 if inst.operands else None]
        if 'store' in op:
            return [0 if inst.operands else None,
                    1 if len(inst.operands) >= 2 else None, None, None]
        if 'atomic' in op:
            # vaddr at op[0], vdata at op[1].  Most atomics also have
            # vdst==op[0] when GLC is set, but treat as src0/src1.
            return [0 if inst.operands else None,
                    1 if len(inst.operands) >= 2 else None, None, None]
        return None

    # Image:
    #   image_*  : op[0]=vdata, op[1..]=vaddr0,vaddr1,vaddr2
    if op.startswith('image_'):
        return [1 if len(inst.operands) >= 2 else None,
                2 if len(inst.operands) >= 3 else None,
                3 if len(inst.operands) >= 4 else None,
                0 if inst.operands else None]

    # Tensor instructions on gfx1250 (tensor_load_to_lds etc.) -- no
    # VGPR operands typically; fall through (no demand) and let the
    # state carry over.
    if op.startswith('tensor_'):
        return None

    # VOP1/VOP2/VOP3/VOP3P/VOPC/DPP -- regular VALU.
    # Order in assembly: vdst, src0 [, src1 [, src2]].
    if op.startswith('v_'):
        slots = [None, None, None, None]
        # vdst at op[0]
        if inst.operands:
            slots[3] = 0
        # src0 at op[1]
        if len(inst.operands) >= 2:
            slots[0] = 1
        # src1 at op[2]
        if len(inst.operands) >= 3:
            slots[1] = 2
        # src2 at op[3]
        if len(inst.operands) >= 4:
            slots[2] = 3
        return slots

    # SALU and others have no VGPR demand.
    return None


def _compute_new_mode(inst: Instruction) -> _ModeTy:
    """Build a fresh _ModeTy for ``inst`` -- only slots whose mapped
    operand is a VGPR get a demand."""
    new = _ModeTy()
    mapping = _msb_slot_to_operand_index(inst)
    if mapping is None:
        return new
    for slot, op_idx in enumerate(mapping):
        if op_idx is None or op_idx >= len(inst.operands):
            continue
        op = inst.operands[op_idx]
        if op.regs and op.regs[0].kind == 'v':
            new.ops[slot] = op.regs[0].msb()
    return new


# Legacy 4-tuple-based helpers, retained as adapters for callers
# elsewhere in the file.

def _msb_demands_for_inst(inst: Instruction,
                          ) -> tuple[Optional[int], Optional[int],
                                     Optional[int], Optional[int]]:
    """Legacy adapter returning ``(dst, src0, src1, src2)`` demands."""
    new = _compute_new_mode(inst)
    return (new.ops[3], new.ops[0], new.ops[1], new.ops[2])


def _required_msb_for_inst(inst: Instruction,
                           current: tuple[int, int, int, int],
                           ) -> tuple[int, int, int, int]:
    """Legacy adapter -- reset-unused emission semantics."""
    del current
    demands = _msb_demands_for_inst(inst)
    return tuple(d if d is not None else 0 for d in demands)


def _instruction_uses_vgpr(inst: Instruction) -> bool:
    """True if any operand of inst is a VGPR.  Used to gate MSB
    regeneration: SALU instructions don't need an MSB context."""
    for op in inst.operands:
        for r in op.regs:
            if r.kind == 'v':
                return True
    return False


def apply_allocation(program: Program,
                     ba: BankAssignment,
                     alloc: VGPRAllocation,
                     verbose: bool = False) -> None:
    """Rewrite ``program`` in place to use the VGPR layout from
    ``alloc``.  Strips existing ``s_set_vgpr_msb`` instructions and
    regenerates them based on actual operand banks.  No-op when
    ``alloc`` is empty (e.g., no self-loop)."""
    if not alloc.wmma_acc:
        return  # nothing to apply

    from collections import defaultdict

    annotate_regions(program)
    pindex = build_program_def_use_index(program)
    block_order = {bb.name: i for i, bb in enumerate(program.blocks)}
    wchains = collect_wmma_chains(program)
    dchains = collect_ds_chains(program)

    loop_range = _loop_body_range(program)
    loop_bb, cbranch_idx = (loop_range if loop_range is not None
                            else (None, None))
    preheader_bb = (_find_preheader(program, loop_bb)
                    if loop_bb is not None else None)

    # ---- Phase A: build maps -------------------------------------

    # acc: per-WMMAChain, OLD canonical's first id -> (chain, new Register).
    acc_chain_by_old_first: dict[int, WMMAChain] = {}
    for c in wchains:
        if alloc.acc(c) is None:
            continue
        acc_chain_by_old_first[c.canonical.ids[0]] = c

    # ds_load -> DSGroup.
    ds_load_to_group: dict[int, DSGroup] = {}
    # Prologue loads need their dst rewritten (to land in the chain's
    # new data-tile slot) but not their addr -- the prologue uses a
    # different base pointer than the steady-state loop body
    # (e.g., v0 vs v807 in v9).
    is_prologue_load: set[int] = set()
    for dc in dchains:
        for g in dc.dsgroups:
            for ld in g.ds_loads:
                ds_load_to_group[id(ld)] = g
            for ld in g.prologue_loads:
                ds_load_to_group[id(ld)] = g
                is_prologue_load.add(id(ld))

    # DSGroup -> DSChain.
    dsgroup_to_chain: dict[int, DSChain] = {}
    for dc in dchains:
        for g in dc.dsgroups:
            dsgroup_to_chain[id(g)] = dc

    # ``DSGroup.tile`` is a property derived from ``ds_loads[*].dst_reg``;
    # once Phase C rewrites the first ds_load's dst, the property
    # changes and the second ds_load can't be relocated correctly.
    # Snapshot the *old* tile of each group here so Phase C uses a
    # stable reference.
    dsgroup_old_tile: dict[int, Register] = {}
    for dc in dchains:
        for g in dc.dsgroups:
            t = g.tile
            if t is not None:
                dsgroup_old_tile[id(g)] = t

    # OLD addr_reg's first id -> list[DSChain] (multiple chains can
    # share the same current addr base, e.g., v642 in v9).
    addr_old_first_to_chains: dict[int, list[DSChain]] = defaultdict(list)
    for dc in dchains:
        if dc.is_epilogue_region:
            continue
        if dc.addr_reg is None or alloc.addr(dc) is None:
            continue
        addr_old_first_to_chains[dc.addr_reg.ids[0]].append(dc)

    # ---- Phase B: addr v_adds ------------------------------------

    # Phase-B-touched instructions that Phase C must skip: the new addr
    # we write here may collide with a WMMAChain canonical's first id,
    # which Phase C's else-branch would otherwise re-rewrite to the new
    # acc start.  E.g., L2's new addr=v448 collides with WMMA chain
    # canonical=v[448:455] (renamed to v[64:71]).
    phase_b_handled: set[int] = set()

    # Step B1: re-target existing LICM-hoisted v_adds.  An LICM v_add
    # has the shape ``v_add_nc_u32_e32 v_X /*log_X*/, imm, v_base``
    # where v_X is the addr_reg of exactly one DSChain.  We change
    # its dst to the chain's allocated new addr.
    if preheader_bb is not None:
        addr_to_dschain: dict[int, DSChain] = {}
        for dc in dchains:
            if dc.is_epilogue_region or dc.addr_reg is None:
                continue
            if alloc.addr(dc) is None:
                continue
            # Multiple DSChains can share an addr; the LICM v_add (if
            # any) produces a UNIQUE addr.  An addr that's only
            # produced once (by a hoisted v_add) is unique to one
            # chain; an addr that's set in the prologue (v642) is not.
            shared = len(addr_old_first_to_chains.get(
                dc.addr_reg.ids[0], [])) > 1
            if not shared:
                addr_to_dschain[dc.addr_reg.ids[0]] = dc

        for inst in preheader_bb.instructions:
            if inst.opcode != 'v_add_nc_u32_e32':
                continue
            if (len(inst.operands) < 3
                    or not inst.operands[0].regs
                    or inst.operands[1].text.strip() == '0'):
                continue
            dst_old = inst.operands[0].regs[0]
            dc = addr_to_dschain.get(dst_old.ids[0])
            if dc is None:
                continue
            new_addr = alloc.addr(dc)
            _replace_operand_register(inst.operands[0], new_addr)
            _rebuild_raw_line(inst)
            phase_b_handled.add(id(inst))

    # Step B2: emit new copy v_adds for DSChains whose current addr
    # is shared (v642-style).  One copy per chain.  Insert in the
    # preheader, just before the closing ``s_set_vgpr_msb 0x4000``
    # reset that LICM puts there (or at the end if no such reset).
    if preheader_bb is not None:
        # Group by old shared addr base.
        for old_first, chains_using_base in addr_old_first_to_chains.items():
            if len(chains_using_base) <= 1:
                continue  # not shared; covered by step B1
            # Pick a "source" DSChain to read its addr_reg (any will
            # do, they're all the same register).
            base_reg = chains_using_base[0].addr_reg
            for dc in chains_using_base:
                new_addr = alloc.addr(dc)
                if new_addr is None:
                    continue
                if (new_addr.ids == base_reg.ids
                        and new_addr.raw_ids == base_reg.raw_ids):
                    continue  # identity rename: copy would be a no-op
                # Build: v_add_nc_u32_e32 <new_addr>, 0, <base_reg>
                # We let _build_v_add_copy_line construct the text.
                new_text, new_logical = _operand_text_for_register(new_addr)
                base_text, base_logical = _operand_text_for_register(base_reg)
                dst_op = (f'{new_text} {new_logical}'
                          if new_logical else new_text)
                base_op = (f'{base_text} {base_logical}'
                           if base_logical else base_text)
                line = (f'\tv_add_nc_u32_e32 {dst_op}, 0, {base_op}')
                copy_inst = _parse_instruction_line(line)
                copy_inst.parent_bb = preheader_bb
                # Find insertion point: before the last s_set_vgpr_msb
                # of the preheader (the loop-entry MSB reset).
                insert_idx = len(preheader_bb.instructions)
                for i in range(len(preheader_bb.instructions) - 1, -1, -1):
                    if (preheader_bb.instructions[i].opcode
                            == 's_set_vgpr_msb'):
                        insert_idx = i
                        break
                preheader_bb.instructions.insert(insert_idx, copy_inst)
                phase_b_handled.add(id(copy_inst))

    # ---- Phase C: walk all instructions and rewrite operands -----

    def _find_dsgroup_via_reaching_def(read_inst: Instruction,
                                       op: Operand
                                       ) -> Optional[DSGroup]:
        """For a wmma's src or a non-chain read of a tile, find the
        DSGroup whose ds_load most recently wrote the operand's data."""
        if not op.regs or op.regs[0].kind != 'v':
            return None
        first_id = op.regs[0].ids[0]
        rid = ('v', first_id)
        read_key = _program_order(read_inst, block_order)
        best = None
        best_key = None
        for d in pindex.defs.get(rid, []):
            if not _is_ds_load(d):
                continue
            dk = _program_order(d, block_order)
            if dk >= read_key:
                continue
            if best is None or dk > best_key:
                best = d
                best_key = dk
        if best is None:
            return None
        return ds_load_to_group.get(id(best))

    # Build an OLD acc-id -> WMMAChain map for non-chain rewrites.
    # An old VGPR id falls inside a chain's canonical range iff that
    # chain "owns" it.
    acc_old_id_to_chain: dict[int, WMMAChain] = {}
    for c in wchains:
        if alloc.acc(c) is None:
            continue
        for old_id in c.canonical.ids:
            acc_old_id_to_chain[old_id] = c

    def _new_register_for_acc_slice(chain: WMMAChain,
                                    old_first: int, size: int
                                    ) -> Optional[Register]:
        """Get the NEW Register for an old [old_first..old_first+size)
        slice of chain's acc."""
        new_acc = alloc.acc(chain)
        if new_acc is None:
            return None
        offset = old_first - chain.canonical.ids[0]
        if offset < 0 or offset + size > new_acc.size:
            return None
        new_start = new_acc.ids[0] + offset
        return _make_logical_register(new_start, size)

    def _new_register_for_data_slice(group: DSGroup,
                                     old_first: int, size: int
                                     ) -> Optional[Register]:
        """Get the NEW Register for an old slice of a DSGroup tile.
        Uses the Phase-A tile snapshot (``dsgroup_old_tile``) since the
        live ``group.tile`` property mutates as Phase C rewrites the
        ds_load destinations."""
        new_data = alloc.data(group)
        if new_data is None:
            return None
        old_tile = dsgroup_old_tile.get(id(group))
        if old_tile is None:
            return None
        offset = old_first - old_tile.ids[0]
        if offset < 0 or offset + size > new_data.size:
            return None
        new_start = new_data.ids[0] + offset
        return _make_logical_register(new_start, size)

    for bb in program.blocks:
        for inst in bb.instructions:
            if not inst.opcode or inst.opcode == '__asm_block__':
                continue
            opcode = inst.opcode

            if opcode.startswith('v_wmma'):
                # dst, src2 -> acc.
                chain = inst.wmma_chain
                if chain is None or alloc.acc(chain) is None:
                    continue
                new_acc = alloc.acc(chain)
                if inst.operands and inst.operands[0].regs:
                    _replace_operand_register(inst.operands[0], new_acc)
                if len(inst.operands) >= 4 and inst.operands[3].regs:
                    _replace_operand_register(inst.operands[3], new_acc)
                # src0, src1 -> data tile.
                for slot in (1, 2):
                    if slot >= len(inst.operands):
                        continue
                    op = inst.operands[slot]
                    if not op.regs or op.regs[0].kind != 'v':
                        continue
                    g = _find_dsgroup_via_reaching_def(inst, op)
                    if g is None:
                        continue
                    new_slice = _new_register_for_data_slice(
                        g, op.regs[0].ids[0], op.regs[0].size)
                    if new_slice is not None:
                        _replace_operand_register(op, new_slice)
                _rebuild_raw_line(inst)

            elif _is_ds_load(inst):
                g = ds_load_to_group.get(id(inst))
                if g is None:
                    continue
                # dst -> data half (or other slice) of group.
                if inst.operands and inst.operands[0].regs:
                    op = inst.operands[0]
                    new_slice = _new_register_for_data_slice(
                        g, op.regs[0].ids[0], op.regs[0].size)
                    if new_slice is not None:
                        _replace_operand_register(op, new_slice)
                # addr -> chain's new addr.  Skip prologue loads:
                # they prefetch via a different base pointer (e.g., v0
                # in v9) that the chain's steady-state addr doesn't
                # alias.
                if id(inst) not in is_prologue_load:
                    dc = dsgroup_to_chain.get(id(g))
                    if (dc is not None and len(inst.operands) >= 2
                            and alloc.addr(dc) is not None):
                        new_addr = alloc.addr(dc)
                        _replace_operand_register(inst.operands[1], new_addr)
                _rebuild_raw_line(inst)

            elif id(inst) in phase_b_handled:
                # Phase B already wrote the final dst (a DSChain new
                # addr) which may collide with a WMMAChain canonical
                # first id.  Don't second-guess it here.
                continue

            else:
                # Non-chain instruction.  Rewrite any operand whose
                # first VGPR id is in our acc-id map (e.g., acc-init
                # v_dual_mov_b32, epilogue v_cvt_pk_f16_f32 reading
                # acc, buffer_store reading output).
                changed = False
                for op in inst.operands:
                    if not op.regs or op.regs[0].kind != 'v':
                        continue
                    old_first = op.regs[0].ids[0]
                    chain = acc_old_id_to_chain.get(old_first)
                    if chain is None:
                        continue
                    new_slice = _new_register_for_acc_slice(
                        chain, old_first, op.regs[0].size)
                    if new_slice is not None:
                        _replace_operand_register(op, new_slice)
                        changed = True
                if changed:
                    _rebuild_raw_line(inst)

    # ---- Phase D: strip all s_set_vgpr_msb -----------------------

    import os as _os_de
    skip_de = _os_de.environ.get('TRITON_AMDGCN_AS_NO_DE') == '1'
    if skip_de:
        return  # Phase D/E disabled; F/G also skipped intentionally
    for bb in program.blocks:
        bb.instructions = [i for i in bb.instructions
                           if i.opcode != 's_set_vgpr_msb']
        for i, inst in enumerate(bb.instructions):
            inst.index = i

    # ---- Phase E: regenerate s_set_vgpr_msb (LLVM-faithful) ------
    # Faithful port of LLVM's ``AMDGPULowerVGPREncoding::run``:
    #   * ``CurrentMode`` is a 4-slot Optional[int] state; only slots
    #     touched by an instruction with a VGPR demand are set.
    #   * On a state transition that *rewrites* a previously-set slot,
    #     emit a brand new ``s_set_vgpr_msb`` instruction with imm =
    #     ``NewMode.encode() | (OldCurrent.encode() << 8)`` and reset
    #     ``CurrentMode = NewMode`` (slots not demanded by NewMode go
    #     back to None).
    #   * Otherwise, *piggyback* by mutating the most recently emitted
    #     MSB's imm to ``CurrentMode.encode() | OldHigh`` (preserves
    #     accumulated demands).
    #   * Reset to all-zero at end of each basic block and before
    #     terminators / branches (LLVM's "non-fall-through BBs start
    #     with all 4 MSBs zero" convention).
    #   * Hoist new MSBs back past ``s_delay_alu``, ``s_wait*``, and
    #     barrier signal/wait instructions (handleCoissue equivalent).

    # Mirror of LLVM's ``isProgramStateInstr`` in handleCoissue:
    # ``isBarrier(Opc) || isWaitcnt(Opc) || Opc == S_DELAY_ALU``.
    # ``isWaitcnt`` covers the dscnt/loadcnt/etc. counter waits but
    # not ``s_wait_alu`` (a depctr instr) or ``s_wait_tensorcnt`` /
    # ``s_wait_event``.  ``isBarrier`` covers s_barrier_*.
    _coissue_skip = (
        's_delay_alu',
        # isWaitcnt opcodes:
        's_waitcnt', 's_waitcnt_vscnt', 's_waitcnt_vmcnt',
        's_waitcnt_expcnt', 's_waitcnt_lgkmcnt',
        's_wait_loadcnt', 's_wait_loadcnt_dscnt',
        's_wait_storecnt', 's_wait_storecnt_dscnt',
        's_wait_samplecnt', 's_wait_bvhcnt', 's_wait_expcnt',
        's_wait_dscnt', 's_wait_kmcnt', 's_wait_idle',
        # isBarrier opcodes (synchronization):
        's_barrier', 's_barrier_signal', 's_barrier_wait',
        's_barrier_leave', 's_barrier_signal_isfirst',
        's_barrier_signal_isfirst_imm', 's_barrier_signal_isfirst_m0',
        's_barrier_signal_imm', 's_barrier_signal_m0',
    )

    def _is_meta(inst: Instruction) -> bool:
        """Pseudo-directives that aren't MIR instructions: ``.loc``,
        ``.file``, etc.  These are debug metadata in MIR and
        ``handleCoissue`` doesn't see them.  ``__asm_block__``
        (INLINEASM in MIR) IS a real instruction and stops the walk."""
        op = inst.opcode
        return op.startswith('.') if op else True

    def _hoist_back(insts: list[Instruction]) -> int:
        """Mirror of ``handleCoissue``: walk back past program-state
        SALUs (delay/wait/barrier) so the new MSB lands before them.
        Meta pseudo-instructions (``.loc``, asm block markers) are
        transparent -- LLVM's MIR-level hoist doesn't see them."""
        i = len(insts)
        while i > 0 and (insts[i - 1].opcode in _coissue_skip
                         or _is_meta(insts[i - 1])):
            i -= 1
        return i

    def _is_terminator_or_call(inst: Instruction) -> bool:
        op = inst.opcode
        if not op:
            return False
        return (op.startswith('s_branch') or op.startswith('s_cbranch')
                or op == 's_setpc_b64' or op == 's_swappc_b64'
                or op == 's_call_b64' or op == 's_endpgm'
                or op == 's_endpgm_saved')

    def _emit_set_mode(new_mode: _ModeTy,
                       new_insts: list[Instruction],
                       current_mode: _ModeTy,
                       most_recent_msb: Optional[Instruction],
                       at_end: bool = False,
                       ) -> tuple[_ModeTy, Optional[Instruction], bool]:
        """Apply LLVM's setMode logic.  ``at_end=True`` mirrors LLVM
        passing ``MBB.instr_end()`` (handleCoissue returns immediately
        for end iterators), so the new MSB is appended without
        hoisting back through program-state instrs.

        Returns ``(current_mode, most_recent_msb, changed)``.
        Mutates ``new_insts`` and the most-recent MSB in-place when
        piggybacking."""
        old_mode_bits = current_mode.encode() << 8
        updated, rewritten = current_mode.update(new_mode)
        if not updated:
            return current_mode, most_recent_msb, False

        if most_recent_msb is not None and not rewritten:
            # Piggyback: rewrite the existing s_set_vgpr_msb's imm.
            try:
                old_imm = int(most_recent_msb.operands[0].text, 0)
            except (ValueError, IndexError):
                old_imm = 0
            keep_high = old_imm & 0xff00
            new_imm = (current_mode.encode() & 0xff) | keep_high
            most_recent_msb.operands[0].text = f'{new_imm:#x}'
            most_recent_msb.operands[0].regs = []
            decoded = _decode_msb_imm(f'{new_imm:#x}')
            most_recent_msb.msb_bits = decoded
            most_recent_msb.trailing_comment = (
                f"  msbs: dst={decoded[0]} src0={decoded[1]} "
                f"src1={decoded[2]} src2={decoded[3]}")
            _rebuild_raw_line(most_recent_msb)
            return current_mode, most_recent_msb, True

        # New emit: imm uses NewMode (not full CurrentMode) so unused
        # slots are 0.  After emission, CurrentMode collapses back to
        # NewMode.
        imm = new_mode.encode() | old_mode_bits
        msb = _make_msb_instruction_from_imm(imm)
        insert_at = len(new_insts) if at_end else _hoist_back(new_insts)
        new_insts.insert(insert_at, msb)
        return new_mode.copy(), msb, True

    for bb in program.blocks:
        new_insts: list[Instruction] = []
        current_mode = _ModeTy()
        most_recent_msb: Optional[Instruction] = None
        for inst in bb.instructions:
            if not inst.opcode or inst.opcode == '__asm_block__':
                new_insts.append(inst)
                continue
            # Reset MSB to (0,0,0,0) before terminators/branches/calls.
            # LLVM exception: s_endpgm/s_endpgm_saved skip the reset
            # (just clear CurrentMode internally) since the kernel is
            # done -- no following instruction can read MSB.
            if _is_terminator_or_call(inst):
                if inst.opcode in ('s_endpgm', 's_endpgm_saved'):
                    current_mode = _ModeTy()
                elif any(s is not None and s != 0 for s in current_mode.ops):
                    reset_mode = _ModeTy([0, 0, 0, 0])
                    current_mode, most_recent_msb, _ = _emit_set_mode(
                        reset_mode, new_insts, current_mode, most_recent_msb)
                most_recent_msb = None
                new_insts.append(inst)
                continue

            mapping = _msb_slot_to_operand_index(inst)
            if mapping is not None:
                new_mode = _compute_new_mode(inst)
                if not current_mode.is_compatible(new_mode):
                    current_mode, most_recent_msb, _ = _emit_set_mode(
                        new_mode, new_insts, current_mode, most_recent_msb)
            new_insts.append(inst)

        # End-of-BB reset (only if state is non-default and the BB
        # falls through to the next block).  ``at_end=True`` matches
        # LLVM passing ``MBB.instr_end()`` so the reset is appended
        # without hoisting back through program-state instrs.
        if any(s is not None and s != 0 for s in current_mode.ops):
            reset_mode = _ModeTy([0, 0, 0, 0])
            current_mode, most_recent_msb, _ = _emit_set_mode(
                reset_mode, new_insts, current_mode, most_recent_msb,
                at_end=True)

        bb.instructions = new_insts
        for i, inst in enumerate(bb.instructions):
            inst.index = i

    # ---- Phase F: drain VALU writes before loop entry ------------

    import os as _os_f
    if (preheader_bb is not None and loop_bb is not None
            and _os_f.environ.get('TRITON_AMDGCN_AS_NO_F') != '1'):
        # Insert s_wait_alu depctr_va_vdst(0) BEFORE the LICM-end
        # MSB reset (the trailing s_set_vgpr_msb 0x...00 that Phase E
        # appended).  Putting the wait after the MSB reset triggers a
        # gfx1250 hazard that crashes the simulator.  LLVM emits
        # s_wait_alu right after the last VALU write; we mirror that
        # by inserting it just before the LICM-end MSB.
        wait_inst = _parse_instruction_line(
            "\ts_wait_alu depctr_va_vdst(0)")
        wait_inst.parent_bb = preheader_bb
        # Find the LICM-end reset MSB (last instruction with low byte 0).
        insert_idx = len(preheader_bb.instructions)
        for j in range(len(preheader_bb.instructions) - 1, -1, -1):
            cand = preheader_bb.instructions[j]
            if cand.opcode != 's_set_vgpr_msb':
                continue
            try:
                if (int(cand.operands[0].text, 0) & 0xff) == 0:
                    insert_idx = j
                    break
            except (ValueError, IndexError):
                pass
        preheader_bb.instructions.insert(insert_idx, wait_inst)
        for i, inst in enumerate(preheader_bb.instructions):
            inst.index = i

    # ---- Phase G: bump kernel descriptor -------------------------

    import os as _os_g
    if _os_g.environ.get('TRITON_AMDGCN_AS_NO_G') == '1':
        return
    if alloc.budget > 0:
        current_budget = _vgpr_budget(program)
        if alloc.budget > current_budget:
            _set_vgpr_budget(program, alloc.budget)


# -------------------------------------------------------------------------
# Round-trip emit
# -------------------------------------------------------------------------

def emit_program(program: Program) -> str:
    return program.emit()


# -------------------------------------------------------------------------
# Top-level entry point
# -------------------------------------------------------------------------

def amdgcnas_gfx12(text: str, verbose: bool = False) -> str:
    """Apply gfx1250-specific assembly post-processing passes and return
    the updated assembly text.

    Runs, in order:
      - ``hoist_loop_invariant_addrs``  hoists loop-invariant v_add
        address computations that MachineLICM declined to hoist.
      - ``merge_dscnt_waits``           consolidates per-region
        ``s_wait_dscnt`` instructions.
      - ``overlap_wmma_with_barrier``   reorders wmma across barrier
        signal/wait pairs to hide sync latency.
      - ``assign_banks`` + ``allocate_vgprs`` + ``apply_allocation``
        bank-aware VGPR reallocation that minimizes
        ``s_set_vgpr_msb`` thrash inside the loop.
    """
    program = parse_asm(text)
    import os as _os
    n_hoisted = n_waits = n_barriers = 0
    if _os.environ.get('TRITON_AMDGCN_AS_NO_PREPASS') != '1':
        n_hoisted = hoist_loop_invariant_addrs(program)
        n_waits = merge_dscnt_waits(program)
        n_barriers = overlap_wmma_with_barrier(program)
    annotate_regions(program)
    ba = assign_banks(program)
    alloc = allocate_vgprs(program, ba)
    # Stage 4.4+5 is opt-in until the MSB-regen pattern matches LLVM's
    # exact emission convention (validity/commit byte semantics).
    if _os.environ.get('TRITON_AMDGCN_AS_APPLY') == '1':
        # DEBUG: identity allocation to test rewrite-only path
        if _os.environ.get('TRITON_AMDGCN_AS_IDENTITY') == '1':
            chains = collect_wmma_chains(program)
            alloc.wmma_acc = {id(c): c.canonical for c in chains}
            alloc.ds_chain_addr = {}
            alloc.ds_group_data = {}
            for dc in collect_ds_chains(program):
                if dc.is_epilogue_region:
                    continue
                if dc.addr_reg is not None:
                    alloc.ds_chain_addr[id(dc)] = dc.addr_reg
                for g in dc.dsgroups:
                    if g.tile is not None:
                        alloc.ds_group_data[id(g)] = g.tile
        apply_allocation(program, ba, alloc, verbose=verbose)
    if verbose:
        print(f"[amdgcnas_gfx12] hoisted {n_hoisted} invariant addr insts, "
              f"merged {n_waits} s_wait_dscnt, "
              f"hoisted {n_barriers} wmma into barrier pairs, "
              f"vgpr budget = {alloc.budget}")
        print(report_chains(program))
    return emit_program(program)
