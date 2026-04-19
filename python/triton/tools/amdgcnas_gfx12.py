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

    ``text`` is the original operand text as it appeared in the source
    (excluding the logical-id comment).  ``regs`` is the list of registers
    mentioned in the operand (usually 0 or 1).  ``logical_text`` is the
    ``/*v[...]*/`` annotation text if present, else ``None``.
    """
    text: str
    regs: list[Register]
    logical_text: Optional[str] = None

    def emit(self) -> str:
        if self.logical_text is not None:
            return f"{self.text} {self.logical_text}"
        return self.text


# -------------------------------------------------------------------------
# Markers (scheduler inline asm comments)
# -------------------------------------------------------------------------

@dataclass
class RegionMarker:
    region: int
    wmma: int
    ds_load: int
    tdm: int


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
    r';{1,2}\s*(?:Epilogue\s+)?Region\s+(\d+)\s*:\s*'
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
        op_text = text[:m.start()].rstrip() + text[m.start():m.end(2) + 1]
        logical_text = f"/*v[{log_lo}:{log_hi}]*/"
        # Preserve any suffix modifiers (offset:, etc.) after the comment
        suffix = text[m.end():].lstrip()
        if suffix:
            op_text = f"{op_text} {suffix}"
        return Operand(text=op_text, regs=regs, logical_text=logical_text)

    m = _VGPR_SINGLE_COMMENT.search(text)
    if m:
        raw = int(m.group(1))
        log = int(m.group(2))
        regs.append(Register(kind='v', ids=[log], raw_ids=[raw]))
        op_text = text[:m.start()].rstrip() + f"v{raw}"
        logical_text = f"/*v{log}*/"
        suffix = text[m.end():].lstrip()
        if suffix:
            op_text = f"{op_text} {suffix}"
        return Operand(text=op_text, regs=regs, logical_text=logical_text)

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
            # Production format: (region, wmma, GR, LR).  GR here includes
            # TDM (global reads); LR is ds_load count.
            inst.region_marker = RegionMarker(
                region=int(m.group(1)),
                wmma=int(m.group(2)),
                ds_load=int(m.group(4)),
                tdm=int(m.group(3)),
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
    tags, matching their default state.
    """
    for bb in program.blocks:
        cur_region: Optional[int] = None
        cur_sub: Optional[int] = None
        for inst in bb.instructions:
            if inst.region_marker is not None:
                cur_region = inst.region_marker.region
                cur_sub = None
            if inst.subregion_marker is not None:
                cur_sub = inst.subregion_marker.sub_region
            inst.region_idx = cur_region
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
    """
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
    """
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

    lines.append(f"WMMA chains: {len(wmma_chains)}")
    for chain in sorted(wmma_chains, key=lambda c: c.canonical.start):
        bbs = {w.parent_bb.name for w in chain.wmmas if w.parent_bb}
        regions = sorted({w.region_idx for w in chain.wmmas
                          if w.region_idx is not None})
        lines.append(f"  {chain.summary()} regions={regions} bbs={sorted(bbs)}")

    lines.append(f"DS chains: {len(ds_chains)}")
    for chain in sorted(ds_chains,
                        key=lambda c: (c.wmma_chain.canonical.start,
                                       c.operand_idx)):
        regions = sorted({d.region_idx for d in chain.ds_loads
                          if d.region_idx is not None})
        lines.append(f"  {chain.summary()} regions={regions}")

    return "\n".join(lines)


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

    Currently runs the peephole cleanups ``merge_dscnt_waits`` and
    ``overlap_wmma_with_barrier``.  Future stages will add bank-aware
    VGPR renaming.
    """
    program = parse_asm(text)
    n_waits = merge_dscnt_waits(program)
    n_barriers = overlap_wmma_with_barrier(program)
    if verbose:
        annotate_regions(program)
        print(f"[amdgcnas_gfx12] merged {n_waits} s_wait_dscnt, "
              f"hoisted {n_barriers} wmma into barrier pairs")
        print(report_chains(program))
    return emit_program(program)
