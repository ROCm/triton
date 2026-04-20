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

    def emit(self) -> str:
        # Preserve original line verbatim when the instruction has no
        # structured form (labels, directives, inline asm blocks).
        if self.opcode.startswith('.') or self.opcode == '__asm_block__':
            return self.raw_line
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

    # No comment forms.
    m = _VGPR_RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        regs.append(Register(kind='v', ids=list(range(lo, hi + 1))))
        return Operand(text=text, regs=regs)

    m = _SGPR_RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        regs.append(Register(kind='s', ids=list(range(lo, hi + 1))))
        return Operand(text=text, regs=regs)

    m = _AGPR_RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        regs.append(Register(kind='a', ids=list(range(lo, hi + 1))))
        return Operand(text=text, regs=regs)

    m = _VGPR_SINGLE.search(text)
    if m:
        regs.append(Register(kind='v', ids=[int(m.group(1))]))
        return Operand(text=text, regs=regs)

    m = _SGPR_SINGLE.search(text)
    if m:
        regs.append(Register(kind='s', ids=[int(m.group(1))]))
        return Operand(text=text, regs=regs)

    m = _AGPR_SINGLE.search(text)
    if m:
        regs.append(Register(kind='a', ids=[int(m.group(1))]))
        return Operand(text=text, regs=regs)

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
    operands: list[Operand] = []
    for part in _split_operands(operand_text):
        operands.append(_parse_operand(part))

    inst = Instruction(opcode=opcode, operands=operands,
                       raw_line=line, trailing_comment=comment)
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
class DSChain:
    """A group of ds_load instructions that together populate a single
    logical tile tensor.

    Two ds_loads belong to the same tile when their destinations feed
    the same operand slot (``src0``, ``src1``, or ``src2``) of the same
    WMMAChain.  ``operand_idx`` records that slot.
    """
    wmma_chain: WMMAChain
    operand_idx: int          # 1 == src0, 2 == src1, 3 == src2 in MLIR order
    ds_loads: list[Instruction] = field(default_factory=list)
    consumers: list[Instruction] = field(default_factory=list)  # WMMA users

    def data_regs(self) -> list[Register]:
        return [d.dst_reg() for d in self.ds_loads if d.dst_reg()]

    def summary(self) -> str:
        regs = self.data_regs()
        span = f"{regs[0]}..{regs[-1]}" if regs else "?"
        return (f"DSChain(#loads={len(self.ds_loads)}, regs={span}, "
                f"→ chain dst={self.wmma_chain.canonical}, "
                f"operand=src{self.operand_idx - 1})")


def collect_ds_chains(program: Program) -> list[DSChain]:
    """For each ds_load in the program, identify the WMMAChain and
    operand slot it feeds, and group loads feeding the same
    (chain, operand) pair into one :class:`DSChain`.

    The consumer search is program-wide so that ds_loads in the
    prologue feeding first-iteration loop WMMAs are correctly paired
    with their loop consumers (and similarly for epilogue chains that
    span blocks).

    ds_loads whose output is never consumed by a WMMA (address setup,
    unused scratch, ...) are omitted from the result.

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

    if not any(i.wmma_chain for i in program.iter_instructions()
               if i.opcode.startswith('v_wmma')):
        collect_wmma_chains(program)

    # Program-wide index + a block-order map so we can compare positions
    # across basic blocks.
    pindex = build_program_def_use_index(program)
    block_order = {bb.name: i for i, bb in enumerate(program.blocks)}

    chains: dict[tuple[int, int], DSChain] = {}
    for inst in program.iter_instructions():
        if not _is_ds_load(inst):
            continue
        dst = inst.dst_reg()
        if dst is None or inst.parent_bb is None:
            continue
        load_key = _program_order(inst, block_order)
        # Locate the first WMMA consumer of any VGPR in dst's range,
        # scanning program-wide so prologue ds_loads are paired with
        # their first-iteration loop WMMA consumer.
        best_user: Optional[Instruction] = None
        best_slot: Optional[int] = None
        best_key: Optional[tuple[int, int]] = None
        for rid in _iter_regs([dst]):
            for user in pindex.uses.get(rid, []):
                if not user.opcode.startswith('v_wmma'):
                    continue
                user_key = _program_order(user, block_order)
                if user_key <= load_key:
                    continue
                # Identify which operand slot the ds_load feeds.
                slot = None
                for slot_idx, op in enumerate(user.operands[1:], start=1):
                    if any(uid == rid for uid in _iter_regs(op.regs)):
                        slot = slot_idx
                        break
                if slot is None:
                    continue
                if best_user is None or user_key < best_key:
                    best_user = user
                    best_slot = slot
                    best_key = user_key
        if best_user is None or best_user.wmma_chain is None:
            continue
        key = (id(best_user.wmma_chain), best_slot)
        chain = chains.get(key)
        if chain is None:
            chain = DSChain(wmma_chain=best_user.wmma_chain,
                            operand_idx=best_slot)
            chains[key] = chain
        chain.ds_loads.append(inst)
        if best_user not in chain.consumers:
            chain.consumers.append(best_user)
        inst.ds_chain = chain
    return list(chains.values())


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
                        key=lambda c: (c.wmma_chain.canonical.start,
                                       c.operand_idx)):
        regions = sorted({_fmt_region(d) for d in chain.ds_loads
                          if _fmt_region(d) is not None})
        lines.append(f"  {chain.summary()} regions={regions}")

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

    # Phase 3: ds_load data_bank = acc_bank of the hosting loop region.
    for dc in dchains:
        banks: set[int] = set()
        for ld in dc.ds_loads:
            r = _loop_region_of(ld, loop_bb, cbranch_idx)
            if r is None:
                continue  # prologue/epilogue ds_loads resolve later
            # Pick any wmmaChain in region r to read its acc_bank --
            # by Phase 2 they all agree (or we reported a conflict).
            chains = loop_wmmas_per_region.get(r, [])
            if not chains:
                continue
            banks.add(result.wmma_acc_bank[id(chains[0])])
        if len(banks) == 1:
            result.ds_data_bank[id(dc)] = banks.pop()
        elif len(banks) > 1:
            result.conflicts.append(
                f'DSChain {dc.wmma_chain.canonical}.src{dc.operand_idx - 1}: '
                f'data_bank ambiguous across regions {sorted(banks)}')

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
    # region hosting the ds_load.  (src0 because ds_load's operand[1]
    # is encoded in the src0 MSB slot, matching the wmma's src0.)
    for dc in dchains:
        addr_banks: set[int] = set()
        for ld in dc.ds_loads:
            r = _loop_region_of(ld, loop_bb, cbranch_idx)
            if r is None:
                continue
            for c in loop_wmmas_per_region.get(r, []):
                b = result.wmma_src0_bank.get(id(c))
                if b is not None:
                    addr_banks.add(b)
        if len(addr_banks) == 1:
            result.ds_addr_bank[id(dc)] = addr_banks.pop()
        elif len(addr_banks) > 1:
            result.conflicts.append(
                f'DSChain {dc.wmma_chain.canonical}.src{dc.operand_idx - 1}: '
                f'addr_bank ambiguous across regions {sorted(addr_banks)}')

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
    """
    program = parse_asm(text)
    n_hoisted = hoist_loop_invariant_addrs(program)
    n_waits = merge_dscnt_waits(program)
    n_barriers = overlap_wmma_with_barrier(program)
    if verbose:
        annotate_regions(program)
        print(f"[amdgcnas_gfx12] hoisted {n_hoisted} invariant addr insts, "
              f"merged {n_waits} s_wait_dscnt, "
              f"hoisted {n_barriers} wmma into barrier pairs")
        print(report_chains(program))
    return emit_program(program)
