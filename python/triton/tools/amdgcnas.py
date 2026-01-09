from collections import defaultdict, Counter, deque
from typing import Set, Tuple, Optional
import re
import logging

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )


class Register:

    def __init__(self, kind, ids):
        self.kind = kind  # 's', 'v', 'a'
        self.ids = ids  # list of ints

        if len(self.ids) > 1:
            expected = list(range(self.ids[0], self.ids[0] + len(self.ids)))
            if self.ids != expected:
                raise ValueError(f"Non-contiguous register ids: {self.ids}")

    @property
    def start(self):
        return min(self.ids)

    @property
    def size(self):
        return len(self.ids)

    @property
    def end(self):
        return max(self.ids)

    def overlaps(self, other):
        if self.kind != other.kind:
            return False
        return not (self.end < other.start or other.end < self.start)

    def contains(self, other):
        """True if self fully covers other"""
        if self.kind != other.kind:
            return False
        return self.start <= other.start and self.end >= other.end

    def is_single(self):
        return self.size == 1

    def __hash__(self):
        return hash((self.kind, tuple(sorted(self.ids))))

    def __eq__(self, other):
        return self.kind == other.kind and self.ids == other.ids

    def emit(self):
        return self.__repr__()

    def __repr__(self):
        if len(self.ids) == 1:
            return f"{self.kind}{next(iter(self.ids))}"
        return f"{self.kind}[{min(self.ids)}:{max(self.ids)}]"


class Instruction:

    def __init__(self, opcode, operands, regs_by_operand, loc, raw_line, bb):
        self.opcode = opcode
        self.operands = operands
        self.regs_by_operand = regs_by_operand  # List[List[Register]]
        self.loc = loc
        self.raw_line = raw_line

        self.defs = set()  # set[(kind, id)]
        self.uses = set()

        self.users = set()  # instructions that read registers produced by this instruction
        self.producers = set()  # instructions that write registers used by this instruction

        self.lds_chain: LDSReadChain = None

        self.parent_bb = bb
        self.index = None  # position inside BB

    def emit(self):
        if not self.operands:
            return self.opcode
        return f"{self.opcode} " + ", ".join(self.operands)

    def get_dst_regs(self) -> Register:
        if (not self.operands) or (not self.regs_by_operand):
            return None
        return parse_register(self.operands[0])

    def get_src_regs(self) -> list[Register]:
        regs = []
        if len(self.regs_by_operand) > 1:
            for op in self.operands[1:]:
                regs.extend(extract_registers(op))
        return regs

    def defines(self, reg) -> bool:
        if self.regs_by_operand:
            return self.regs_by_operand[0].overlaps(reg)
        else:
            return False

    def uses_regs(self, reg) -> bool:
        if len(self.regs_by_operand) > 1:
            return any(r.overlaps(reg) for r in self.regs_by_operand[1:])
        else:
            return False

    def replace_dst(self, reg: Register):
        assert len(self.regs_by_operand) > 0
        self.operands[0] = reg.emit()
        self.update_regs_by_operand()

    def update_regs_by_operand(self):
        self.regs_by_operand = []
        for op in self.operands:
            self.regs_by_operand.extend(extract_registers(op))

    ## Replace old_reg with new_reg in this instruction
    def replace_reg(self, old_reg: Register, new_reg: Register):
        for i, op in enumerate(self.operands):
            if op == old_reg.emit():
                self.operands[i] = new_reg.emit()
                break
        self.update_regs_by_operand()

    ## Replace the uses of this inst.def with reg
    def replace_users_with(self, reg: Register):
        #def_reg = coalesce_regs(self.defs)[0]
        def_reg = self.get_dst_regs()
        for user in self.users:
            user.replace_reg(def_reg, reg)

    # ---------- classification ----------
    def is_memory(self):
        return (self.opcode.startswith("ds_") or self.opcode.startswith("buffer_"))

    def is_control(self):
        return self.opcode.startswith("s_branch") or self.opcode.startswith("s_cbranch")

    def is_pure(self):
        return not (self.is_memory() or self.is_control())

    # MFMA-only helpers
    def is_mfma(self) -> bool:
        return self.opcode.startswith("v_mfma")

    def get_mfma_dst(self) -> Register:
        assert self.is_mfma()
        regs = self.get_dst_regs()
        assert regs
        return regs

    def get_mfma_acc(self) -> Register:
        assert self.is_mfma()
        regs = extract_registers(self.operands[-1])
        assert len(regs) == 1
        return regs[0]

    def replace_mfma_dst(self, reg: Register):
        assert self.is_mfma()
        self.operands[0] = reg.emit()
        self.update_regs_by_operand()

    def replace_mfma_acc(self, reg: Register):
        assert self.is_mfma()
        self.operands[-1] = reg.emit()
        self.update_regs_by_operand()

    def replace_mfma_operand(self, reg: Register, opIdx: int):
        assert self.is_mfma()
        assert (opIdx == 1) or (opIdx == 2)
        self.operands[opIdx] = reg.emit()
        self.update_regs_by_operand()

    def get_op_idx(self, reg: Register):
        assert self.is_mfma()
        op1 = extract_registers(self.operands[1])[0]
        op2 = extract_registers(self.operands[2])[0]
        if op1.overlaps(reg):
            return 1
        elif op2.overlaps(reg):
            return 2
        else:
            return 0


class BasicBlock:

    def __init__(self, name):
        self.name = name
        self.instructions = []
        self.succs = []  # filled later
        self.preds = []

        self.defs = set()
        self.uses = set()

        self.live_in = set()
        self.live_out = set()

        self.free_regs = set()  # all_reg - (defs | uses | live_in)

    def add_inst(self, inst):
        inst.parent_bb = self
        inst.index = len(self.instructions)
        self.instructions.append(inst)

    def instructions_between(self, inst_a, inst_b):
        """
        Return instructions strictly between inst_a and inst_b.
        Order does not matter; handles inst_a before inst_b only.
        """
        idx_a = self.instructions.index(inst_a)
        idx_b = self.instructions.index(inst_b)

        if idx_a >= idx_b:
            return []

        return self.instructions[idx_a + 1:idx_b]

    def instructions_before(self, inst, including=False):
        idx = self.instructions.index(inst)
        if including:
            return self.instructions[:idx + 1]
        else:
            return self.instructions[:idx]

    def next_instruction(self, inst):
        if inst == self.instructions[-1]:
            return None
        idx = self.instructions.index(inst)
        return self.instructions[idx + 1]

    def get_reaching_defs(self, inst, reg, including=False):
        """
        reg: Register (possibly a range)
        Returns: set of Instructions
        """
        needed = set(reg.ids)
        reaching_defs = set()

        for prev in reversed(self.instructions_before(inst, including)):
            dreg = prev.get_dst_regs()
            if not dreg:
                continue
            if dreg.kind != reg.kind:
                continue

            overlap = needed & set(dreg.ids)
            if overlap:
                reaching_defs.add(prev)
                needed -= overlap

            if not needed:
                break

        return reaching_defs

    def emit(self):
        lines = []
        lines.append(f"{self.name}:")
        for inst in self.instructions:
            lines.append(inst.emit())
        return "\n".join(lines)


class Program:

    def __init__(self):
        self.header_lines = []  # before first BB
        self.blocks = []  # parsed basic blocks
        self.tail_lines = []  # after s_endpgm


class set_queue:

    def __init__(self):
        self.worklist = deque()
        self.in_worklist = set()

    def push(self, x):
        if x not in self.in_worklist:
            self.worklist.append(x)
            self.in_worklist.add(x)

    def pop(self):
        x = self.worklist.popleft()
        self.in_worklist.remove(x)
        return x

    def isNotEmpty(self):
        return self.in_worklist


class AccumulatorChain:

    def __init__(self, root_mfma: Instruction):
        self.root = root_mfma
        self.mfmas: list[Instruction] = []
        self.copy_instrs: list[Instruction] = []
        self.nops: list[Instruction] = []
        self.acc_regs: set[Register] = set()
        self.canonical: Register | None = None
        self.entry_acc: Register = None
        self.exit_acc: Register = None
        self.zero_init: set[Instruction] = None
        self.use_acc: set[Instruction] = set()  # Other instructions that uses acc

    def rewrite(self):
        canon = self.canonical
        assert canon is not None

        for mfma in self.mfmas:
            mfma.replace_mfma_dst(canon)
            mfma.replace_mfma_acc(canon)

        for inst in self.copy_instrs:
            inst.mark_dead = True

    def set_canon(self, reg):
        self.canonical = reg

    def get_zero_init(self, prologue: BasicBlock):
        self.zero_init = prologue.get_reaching_defs(prologue.instructions[-1], self.entry_acc, True)
        return self.zero_init


class LDSReadChain:

    def __init__(self, root_ds: Instruction):
        self.ds = root_ds
        self.users: set[Instruction] = []
        self.opIdx: Int = 0
        self.data: Register = None
        self.addr: Register = None
        self.loc: Int = 0
        self.isLiveAcrossBB: bool = False


class BufferLoadLDSChain:

    def __init__(self):
        self.entry_reg = None
        self.buffer_load: list[Instruction] = []
        self.copy_inst: list[Instruction] = []
        self.cndmask: Instruction = None


REG_SINGLE = re.compile(r'([sva])(\d+)')
REG_RANGE = re.compile(r'([sva])\[(\d+):(\d+)\]')
REG_PATTERNS = [
    r's\d+',
    r'v\d+',
    r'a\d+',
    r'm0',
    r's\[\d+:\d+\]',
    r'v\[\d+:\d+\]',
    r'a\[\d+:\d+\]',
]


def count_regs(regs: list[Register]):
    num_regs = Counter()
    for flattened in flatten_regs(regs):
        num_regs[flattened[0]] += 1

    return num_regs['a'], num_regs['v']


def parse_register(text):
    if text == 'm0':
        return Register('m', [0])

    kind = text[0]
    if '[' in text:
        lo, hi = map(int, text[text.find('[') + 1:text.find(']')].split(':'))
        return Register(kind, list(range(lo, hi + 1)))
    else:
        return Register(kind, [int(text[1:])])


def split_operands(text):
    if not text:
        return []

    # First split by commas
    comma_parts = [p.strip() for p in text.split(',')]

    operands = []
    for part in comma_parts:
        # Further split space-separated modifiers
        operands.extend(part.split())

    return operands


def extract_registers(op):
    regs = []
    for pat in REG_PATTERNS:
        for m in re.findall(pat, op):
            regs.append(parse_register(m))
    return regs


def parse_instruction(line, loc, bb):
    line = line.strip()
    if not line or line.startswith(';'):
        return None

    # Remove trailing comments
    line = line.split(';', 1)[0].strip()

    parts = line.split(None, 1)
    opcode = parts[0]

    operand_text = parts[1] if len(parts) > 1 else ""
    operands = split_operands(operand_text)

    regs_by_operand = []
    for op in operands:
        regs_by_operand.extend(extract_registers(op))

    return Instruction(opcode, operands, regs_by_operand, loc, line, bb)


def extract_label(line):
    """
    Extract the label at the start of the line, before the first ':'.
    Returns None if no label is present.
    """
    line = line.strip()
    if ':' not in line:
        return None
    # Split at the first colon
    label = line.split(':', 1)[0]
    return label.strip()


def parse_asm(text):
    program = Program()

    cur_block = None
    cur_loc = None
    files = {}
    in_blocks = False
    ended = False

    for raw in text.splitlines():
        line = raw.rstrip()

        # Program tail (after s_endpgm)
        if ended:
            program.tail_lines.append(line)
            continue

        # Detect program end
        if 's_endpgm' in line:
            if cur_block:
                inst = parse_instruction(line, cur_loc, cur_block)
                if inst:
                    cur_block.add_inst(inst)
            program.tail_lines.append(line)
            ended = True
            continue

        # Before first basic block → header
        if not in_blocks:
            program.header_lines.append(line)
            if line.strip().startswith('; %bb.') or line.strip().startswith('.LBB'):
                in_blocks = True
                program.header_lines.pop()  # this line is a BB label
            else:
                continue

        # file directive
        if line.strip().startswith('.file'):
            _, fid, path, name = line.split(maxsplit=3)
            files[int(fid)] = name.strip('"')
            continue

        # loc directive
        if line.strip().startswith('.loc'):
            parts = line.split()
            fid = int(parts[1])
            line_no = int(parts[2])
            col = int(parts[3])
            cur_loc = (files.get(fid, "unknown"), line_no, col)
            continue

        # ignore debug labels
        if line.strip().startswith('.Ltmp'):
            continue

        # new basic block
        if line.strip().startswith('; %bb.') or line.strip().startswith('.LBB'):
            label = extract_label(line)
            cur_block = BasicBlock(label)
            program.blocks.append(cur_block)
            continue

        # sched barrier comment
        if 'sched_barrier' in line:
            continue

        # instruction
        inst = parse_instruction(line, cur_loc, cur_block)
        if inst:
            cur_block.add_inst(inst)

    return program


def emit_blocks(blocks):
    out = []
    for bb in blocks:
        out.append(f"{bb.name}:")
        for inst in bb.instructions:
            out.append(inst.emit())
    return out


def emit_program(program):
    out = []
    out.extend(program.header_lines)
    out.extend(emit_blocks(program.blocks))
    out.extend(program.tail_lines)
    return "\n".join(out)


############################
## Start def-use utilities
############################

NO_DEF_OPS = {
    's_waitcnt',
    's_nop',
    's_branch',
    's_cbranch_scc0',
    's_cbranch_scc1',
}

CMP_PREFIXES = ('s_cmp_', 'v_cmp_')

## TODO(lixun)
## Only buffer_load lds should be included in ALL_USERS set
ALL_USERS = ('s_cmp_', 'v_cmp_', 'v_permlane', 'buffer_store', 'buffer_load')
ALL_DEFS_USES = ('v_permlane')
COPY_DATA = ('v_accvgpr_read', 'v_accvgpr_write', 'v_accvgpr_mov', 'v_mov')


def flatten_regs(regs):
    """
    regs: Register | Iterable[Register]
    return: set of (kind, id)
    """
    assert isinstance(regs, (Register, list)), type(regs)

    out = set()

    if isinstance(regs, Register):
        regs = [regs]

    for r in regs:
        for rid in r.ids:
            out.add((r.kind, rid))

    return out


def coalesce_regs(flat_regs):
    """
    flat_regs: set of (kind, id)
    returns: list[Register]
    """
    by_kind = defaultdict(list)

    # 1. Group by register kind
    for kind, rid in flat_regs:
        by_kind[kind].append(rid)

    regs = []

    # 2. For each kind, sort and coalesce contiguous ids
    for kind, ids in by_kind.items():
        ids = sorted(ids)

        start = ids[0]
        prev = ids[0]

        for cur in ids[1:]:
            if cur == prev + 1:
                prev = cur
                continue

            # end of contiguous range
            regs.append(Register(kind, list(range(start, prev + 1))))
            start = prev = cur

        # last range
        regs.append(Register(kind, list(range(start, prev + 1))))

    return regs


def compute_inst_def_use(inst):
    inst.defs.clear()
    inst.uses.clear()

    if inst.opcode in NO_DEF_OPS:
        return

    if inst.opcode.startswith(ALL_USERS):
        for reg in inst.regs_by_operand:
            inst.uses |= flatten_regs(reg)
        return

    if inst.opcode.startswith(ALL_DEFS_USES):
        for reg in inst.regs_by_operand:
            inst.defs |= flatten_regs(reg)
            inst.uses |= flatten_regs(reg)
        return

    # Normal case
    if inst.regs_by_operand:
        inst.defs |= flatten_regs(inst.regs_by_operand[0])
        for reg in inst.regs_by_operand[1:]:
            inst.uses |= flatten_regs(reg)


def compute_bb_def_use(bb):
    bb.defs.clear()
    bb.uses.clear()

    for inst in bb.instructions:
        compute_inst_def_use(inst)

        for u in inst.uses:
            if u not in bb.defs:
                bb.uses.add(u)

        bb.defs |= inst.defs


def build_cfg(blocks):
    label_map = {bb.name: bb for bb in blocks}

    for i, bb in enumerate(blocks):
        if not bb.instructions:
            continue

        last = bb.instructions[-1].opcode
        text = bb.instructions[-1].raw_line

        if last == 's_branch':
            tgt = text.split()[-1]
            bb.succs.append(label_map[tgt])

        elif last.startswith('s_cbranch'):
            tgt = text.split()[-1]
            bb.succs.append(label_map[tgt])
            if i + 1 < len(blocks):
                bb.succs.append(blocks[i + 1])

        else:
            if i + 1 < len(blocks):
                bb.succs.append(blocks[i + 1])

    for bb in blocks:
        for s in bb.succs:
            s.preds.append(bb)


def compute_liveness(blocks):
    changed = True
    while changed:
        changed = False

        for bb in reversed(blocks):
            new_out = set()
            for s in bb.succs:
                new_out |= s.live_in

            new_in = bb.uses | (new_out - bb.defs)

            if new_out != bb.live_out or new_in != bb.live_in:
                bb.live_out = new_out
                bb.live_in = new_in
                changed = True


def collapse_ranges(ids):
    """
    ids: sorted list[int]
    return: list of strings, e.g. ['56:59', '64:67', '72']
    """
    if not ids:
        return []

    ranges = []
    start = prev = ids[0]

    for x in ids[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x

    ranges.append((start, prev))

    def fmt(lo, hi):
        return f"{lo}:{hi}" if lo != hi else f"{lo}"

    return [fmt(lo, hi) for lo, hi in ranges]


def build_def_use_chains_linear(blocks):
    """
    Correct reaching-definition-based def-use chains.
    Works for non-SSA assembly.
    """

    current_def = {}  # (kind, id) -> Instruction

    for bb in blocks:
        for inst in bb.instructions:
            inst.users.clear()
            inst.producers.clear()
            # ---------
            # Uses: find producers
            # ---------
            for reg in inst.uses:
                if reg in current_def:
                    prod = current_def[reg]
                    inst.producers.add(prod)
                    prod.users.add(inst)

            # ---------
            # Defs: overwrite current definition
            # ---------
            for reg in inst.defs:
                current_def[reg] = inst


def analyze_blocks(blocks):
    logging.debug("")
    logging.debug("============================================================")
    for bb in blocks:
        logging.debug(f"BasicBlock {bb.name}:")

        # --------------------------------------------------
        # Instruction histogram
        # --------------------------------------------------
        inst_hist = Counter()
        for inst in bb.instructions:
            inst_hist[(inst.opcode, inst.loc[1])] += 1
            inst.index = bb.instructions.index(inst)

        logging.debug("  Instruction histogram:")
        for op, cnt in inst_hist.most_common():
            logging.debug(f"    {op[1]:3d}:{op[0]:<40} {cnt}")

        # --------------------------------------------------
        # Register usage (from defs ∪ uses)
        # --------------------------------------------------
        regs_by_kind = defaultdict(set)

        for inst in bb.instructions:
            for kind, rid in inst.defs | inst.uses:
                regs_by_kind[kind].add(rid)

        def dump_regs(kind, name):
            ids = sorted(regs_by_kind.get(kind, []))
            if not ids:
                logging.debug(f"    {name}: none")
                return

            ranges = collapse_ranges(ids)
            logging.debug(f"    {name}: {len(ids)} regs")
            logging.debug(f"      ranges: {', '.join(ranges)}")

        def dump_reg_set(regset, name):
            by_kind = {}
            for k, r in regset:
                by_kind.setdefault(k, set()).add(r)

            logging.debug(f"  {name}:")
            for kind, ids in by_kind.items():
                ranges = collapse_ranges(sorted(ids))
                logging.debug(f"    {kind}: {', '.join(ranges)}")

        logging.debug("  Register usage:")
        dump_regs('s', "SGPR")
        dump_regs('v', "VGPR")
        dump_regs('a', "AGPR")
        dump_regs('m', "Special")

        # --------------------------------------------------
        # Liveness
        # --------------------------------------------------
        logging.debug("  Liveness:")
        dump_reg_set(bb.live_in, "Live-in")
        dump_reg_set(bb.live_out, "Live-out")

        logging.debug("--------------------------------------------------------")


def analyze_block(bb, mfmaChainsInBB, LDSChains):

    all_regs = set()
    live_in = bb.live_in
    entry_acc = set()
    entry_lds_data = set()

    for i in range(512):
        if i > 255:
            kind = 'a'
            id = i - 256
            all_regs.add((kind, id))
        else:
            kind = 'v'
            id = i
            all_regs.add((kind, id))

    for chain in mfmaChainsInBB:
        entry_acc |= flatten_regs(chain.entry_acc)

    for chain in LDSChains:
        ds_read_inst = chain.ds
        if (not ds_read_inst in bb.instructions) and (next(iter(chain.users), None) in bb.instructions):
            entry_lds_data |= flatten_regs(ds_read_inst.get_dst_regs())

    assert entry_acc.issubset(live_in)
    assert entry_lds_data.issubset(live_in)

    remaining = live_in - entry_acc - entry_lds_data

    live_in_a, live_in_v = count_regs(coalesce_regs(live_in))
    logging.debug(f"live_in: a={live_in_a} v={live_in_v} {coalesce_regs(live_in)}")
    acc_a, acc_v = count_regs(coalesce_regs(entry_acc))
    logging.debug(f"live-in acc: a={acc_a} v={acc_v} {coalesce_regs(entry_acc)}")
    lds_a, lds_v = count_regs(coalesce_regs(entry_lds_data))
    logging.debug(f"live-in lds data: a={lds_a} v={lds_v} {coalesce_regs(entry_lds_data)}")

    re_a, re_v = count_regs(coalesce_regs(remaining))
    logging.debug(f"live-in remaining: a={re_a} v={re_v} {coalesce_regs(remaining)}")

    bb_uses = bb.defs | bb.uses
    if live_in.issubset(bb_uses):
        logging.debug(f"bb_uses contains live_in")
    #bb_uses = bb_uses - live_in
    bb_a, bb_v = count_regs(coalesce_regs(bb_uses))
    logging.debug(f"BB uses: a={bb_a} v={bb_v} {coalesce_regs(bb_uses)}")

    live_through = live_in - bb_uses
    th_a, th_v = count_regs(coalesce_regs(live_through))
    logging.debug(f"live through: a={th_a} v={th_v} {coalesce_regs(live_through)}")

    free_regs = all_regs - bb_uses - live_through
    free_a, free_v = count_regs(coalesce_regs(free_regs))
    logging.debug(f"free: a={free_a} v={free_v} {coalesce_regs(free_regs)}")

    bb.free_regs = free_regs


def regs_overlap(a, b):
    return a.overlaps(b)


def collect_mfma_chains(bb):
    visited = set()
    chains = []

    for inst in reversed(bb.instructions):
        if not inst.is_mfma():
            continue
        if inst in visited:
            continue

        chain = AccumulatorChain(inst)

        ## bwd pass
        worklist = [inst]

        while worklist:
            cur = worklist.pop()
            if cur in visited:
                continue
            visited.add(cur)

            #logging.debug(f"visiting mfma: {cur.emit()}")

            if cur.is_mfma():
                chain.mfmas.append(cur)
                next = bb.next_instruction(cur)
                if next and 'nop' in next.opcode:
                    chain.nops.append(next)
                chain.acc_regs.add(cur.get_mfma_dst())

                acc = cur.get_mfma_acc()
                prods = bb.get_reaching_defs(cur, acc)

                for prod in prods:
                    if prod and prod not in visited:
                        worklist.append(prod)

            elif cur.opcode.startswith(COPY_DATA):
                chain.copy_instrs.append(cur)
                for r in cur.get_src_regs():
                    chain.acc_regs.add(r)

                for r in cur.get_src_regs():
                    prods = bb.get_reaching_defs(cur, r)
                    for prod in prods:
                        if prod and prod not in visited:
                            worklist.append(prod)

        if len(chain.mfmas) >= 2:
            chains.append(chain)
            #logging.debug(f"How many mfma in the chain: {len(chain.mfmas)}")

        ## fwd pass
        worklist = []
        for user in inst.users:
            if user in bb.instructions:
                worklist.append(user)

        while worklist:
            cur = worklist.pop()
            if cur in visited:
                continue
            visited.add(cur)

            assert cur.opcode.startswith(COPY_DATA)
            chain.copy_instrs.append(cur)

            for user in cur.users:
                if user in bb.instructions:
                    worklist.append(user)

        get_entry_exit_acc_reg(bb, chain)

    return chains


def choose_canonical_acc(chain, bb):
    candidate = chain.exit_acc

    logging.debug(f"Need optimize copy inst on the chain!! Use {candidate=} while {chain.entry_acc=}")
    logging.debug(f"  number of nops: {len(chain.nops)}")
    if chain.entry_acc != chain.exit_acc:
        logging.debug(f"    mismatch acc")
        mfma = chain.mfmas[-1]
        reach_defs = bb.get_reaching_defs(mfma, mfma.get_mfma_acc())
        for reaching_def in reach_defs:
            logging.debug(f"    {reaching_def.emit()}")

    for inst in bb.instructions_between(chain.mfmas[-1], chain.mfmas[0]):
        if inst in chain.mfmas or inst in chain.copy_instrs:
            continue
        if inst.defines(candidate):
            chain.use_acc.add(inst)
            logging.debug(f"    oops, {inst.emit()} defines {candidate}")
            if 'ds_read_b128' in inst.opcode:
                for user in inst.users:
                    logging.debug(f"      {user.emit()} {user in bb.instructions}")
            return None
        if inst.uses_regs(candidate):
            logging.debug(f"    oops, {inst.emit()} uses {candidate}")
            return None

    return candidate


def cleanup_bb(bb):
    bb.instructions = [inst for inst in bb.instructions if not getattr(inst, "mark_dead", False)]


def get_entry_exit_acc_reg(bb, chain):
    ## entry acc is the reaching definition of the acc regs of chain.mfmas[-1]
    mfma = chain.mfmas[-1]
    worklist = []
    for reaching_def in bb.get_reaching_defs(mfma, mfma.get_mfma_acc()):
        worklist.append(reaching_def)

    final_reaching_defs = []

    while worklist:
        cur = worklist.pop()
        reaching_defs = bb.get_reaching_defs(cur, cur.get_src_regs()[0])
        if reaching_defs:
            for reaching_def in reaching_defs:
                worklist.append(reaching_def)
        else:
            final_reaching_defs.append(cur)

    entry_acc_regs = set()
    if final_reaching_defs:
        for inst in final_reaching_defs:
            entry_acc_regs |= flatten_regs(inst.get_src_regs())
    else:
        entry_acc_regs |= flatten_regs(mfma.get_mfma_acc())

    chain.entry_acc = coalesce_regs(entry_acc_regs)[0]

    ## exit acc is the last user of the dst reg of chain.mfmas[0]
    mfma = chain.mfmas[0]
    worklist = []
    for user in mfma.users:
        if user in bb.instructions:
            worklist.append(user)

    final_users = []

    while worklist:
        cur = worklist.pop()
        all_in_ep = True
        for user in cur.users:
            if user in bb.instructions:
                worklist.append(user)
                all_in_ep = False
        if all_in_ep:
            final_users.append(cur)

    exit_acc_regs = set()
    if final_users:
        for inst in final_users:
            exit_acc_regs |= flatten_regs(inst.get_dst_regs())
    else:
        exit_acc_regs |= flatten_regs(mfma.get_mfma_dst())

    chain.exit_acc = coalesce_regs(exit_acc_regs)[0]


def optimize_mfma_accumulators(bb, chains):

    logging.debug(f"===================================================================")
    logging.debug(f"Optimizing {len(chains)} mfma chains")

    entry_regs = []
    exit_regs = []
    entry_hist = Counter()
    exit_hist = Counter()
    for chain in chains:
        entry_reg, exit_reg = chain.entry_acc, chain.exit_acc
        entry_hist[entry_reg] += 1
        exit_hist[exit_reg] += 1
        entry_regs.append(entry_reg)
        exit_regs.append(exit_reg)
        #if len(chain.copy_instrs) == 0:
        #    continue
        canon = choose_canonical_acc(chain, bb)
        if canon is None:
            continue

        chain.set_canon(canon)
        chain.rewrite()

    entry_acc_flatten = flatten_regs(entry_regs)
    exit_acc_flatten = flatten_regs(exit_regs)
    entry_acc = coalesce_regs(entry_acc_flatten)
    exit_acc = coalesce_regs(exit_acc_flatten)
    a_num, v_num = count_regs(entry_regs)
    logging.debug(f"{a_num=} {v_num=}: {entry_acc=}")
    #for reg, cnt in entry_hist.most_common():
    #    logging.debug(f"    {reg}: {cnt}")

    logging.debug(f"{len(exit_acc_flatten)}: {exit_acc=}")
    #for reg, cnt in exit_hist.most_common():
    #    logging.debug(f"    {reg}: {cnt}")

    cleanup_bb(bb)

    logging.debug(f"=========================== done ====================================")


def pick_and_remove_contiguous_regs(reg_pool: Set[Tuple[str, int]], x: int,
                                    myKind: str = None) -> Optional[Set[Tuple[str, int]]]:
    if x <= 0:
        raise ValueError("x must be positive")

    # Group register ids by kind
    by_kind = defaultdict(list)
    for kind, rid in reg_pool:
        by_kind[kind].append(rid)

    for kind, ids in by_kind.items():
        if myKind and (kind != myKind):
            continue
        ids = sorted(ids)
        id_set = set(ids)

        # Try every possible contiguous window
        for start in ids:
            end = start + x - 1

            # Alignment constraint
            if x > 1 and start % 2 != 0:
                continue

            # Check contiguous existence
            window = list(range(start, start + x))
            if all(i in id_set for i in window):
                chosen = {(kind, i) for i in window}

                # Remove from original pool (in place)
                reg_pool.difference_update(chosen)

                return chosen

    return None


def clear_optimize_mfma_obstacles(bb, chains):
    logging.debug(f"========== clear obstacles ==========")

    x = 4
    for chain in chains:
        for inst in chain.use_acc:
            logging.debug(f"{inst.emit()}")
            avai_regs = pick_and_remove_contiguous_regs(bb.free_regs, x)
            if avai_regs is None:
                logging.debug(f"Not enough free registers")
                return
            logging.debug(f"free regs: {coalesce_regs(avai_regs)}, remaining: {coalesce_regs(bb.free_regs)}")
            free_reg = coalesce_regs(avai_regs)[0]
            if 'ds_read' in inst.opcode:
                lds_chain = inst.lds_chain
                inst.replace_dst(free_reg)
                opIdx = lds_chain.opIdx
                for user in lds_chain.users:
                    user.replace_mfma_operand(free_reg, opIdx)

    logging.debug(f"========== done clear obstacles ==========")


def collect_ds_chains(blocks):

    logging.debug(f"========== Collecting lds chains ==========")
    chains = []

    for bb in blocks:
        for inst in bb.instructions:
            if 'ds_read_b128' in inst.opcode:
                chain = LDSReadChain(inst)
                chain.users = inst.users
                #logging.debug(f"lds users: {len(chain.users)}  {inst.get_dst_regs()[0]}")
                data_reg = inst.get_dst_regs()
                assert chain.opIdx == 0
                chain.loc = inst.loc[1]
                chain.addr = inst.get_src_regs()[0]
                #logging.debug(f"{inst.emit()}")
                for user in chain.users:
                    assert user.is_mfma()
                    opIdx = user.get_op_idx(data_reg)
                    #logging.debug(f"    {opIdx=}: {user.emit()}")
                    assert opIdx == 1 or opIdx == 2
                    if chain.opIdx != 0 and chain.opIdx != opIdx:
                        logging.debug(f"mismatch opIdx!!!")
                    chain.opIdx = opIdx

                    if not user in bb.instructions:
                        chain.isLiveAcrossBB = True
                inst.lds_chain = chain
                chains.append(chain)

    total_lds_regs = []
    lds_regs_by_loc = {}
    for chain in chains:
        total_lds_regs.append(chain.ds.get_dst_regs())
        #logging.debug(f"{chain.loc=}")
        lds_regs_by_loc.setdefault(chain.loc, list()).append(chain.ds.get_dst_regs())

    #for loc, regs in lds_regs_by_loc.items():
    #    reg_list = coalesce_regs(flatten_regs(regs))
    #    logging.debug(f"{loc=}: {len(regs)} {reg_list}")

    a_num, v_num = count_regs(total_lds_regs)
    logging.debug(f"total LDS regs: {a_num=}  {v_num=} {coalesce_regs(flatten_regs(total_lds_regs))}")

    logging.debug(f"========== Done Collecting lds chains ==========")
    return chains


def collect_buffer_load_chains(bb):
    logging.debug(f"========== Collecting buffer load chains ==========")

    visited = set()
    chains = set()

    for inst in bb.instructions:
        if 'buffer_load' in inst.opcode:
            if inst in visited:
                continue
            visited.add(inst)
            '''
            v_accvgpr_read_b32 v1, a153
            v_cndmask_b32_e32 v1, v1, v0, vcc
            buffer_load_dwordx4 v1, s[0:3], 0 offen lds
            v_accvgpr_write_b32 a179, v1
            v_accvgpr_read_b32 v0, a179
            buffer_load_dwordx4 v0, s[52:55], 0 offen lds
            '''

            chain = BufferLoadLDSChain()
            chain.buffer_load.append(inst)
            voff = inst.regs_by_operand[0]
            #logging.debug(f"{inst.emit()} --> {voff=}")
            cndmask_inst = next(iter(bb.get_reaching_defs(inst, voff)))
            #logging.debug(f"  {cndmask_inst.emit()=}")
            assert 'cndmask' in cndmask_inst.opcode
            chain.cndmask = cndmask_inst

            vreg = cndmask_inst.regs_by_operand[1]
            copy_inst = next(iter(bb.get_reaching_defs(cndmask_inst, vreg)))
            assert 'v_accvgpr' in copy_inst.opcode

            chain.copy_inst.append(copy_inst)
            chain.entry_reg = copy_inst.regs_by_operand[1]
            #logging.debug(f"  {chain.entry_reg=}")
            assert flatten_regs(chain.entry_reg).issubset(bb.live_in)

            for user_inst in bb.instructions_between(inst, bb.instructions[-1]):
                if user_inst.uses_regs(voff) or ('buffer_load' in user_inst.opcode
                                                 and user_inst.regs_by_operand[0] == voff):
                    #logging.debug(f"  Found a user of voff: {user_inst.emit()}")
                    if 'buffer_load' in user_inst.opcode:
                        chain.buffer_load.append(user_inst)
                        visited.add(user_inst)
                        #logging.debug(f"  2nd buffer_load: {user_inst.emit()}")
                    else:
                        #logging.debug(f"  else: {user_inst.emit()}")
                        assert 'v_accvgpr' in user_inst.opcode
                        chain.copy_inst.append(user_inst)
                        copy_2nd = next(iter(user_inst.users))
                        assert 'v_accvgpr' in copy_2nd.opcode
                        chain.copy_inst.append(copy_2nd)
                        buffer_load_2nd = next(iter(copy_2nd.users))
                        assert 'buffer_load' in buffer_load_2nd.opcode
                        chain.buffer_load.append(buffer_load_2nd)
                        visited.add(buffer_load_2nd)
                    break

            chains.add(chain)

    logging.debug(f"========== Done Collecting buffer load chains ==========")

    return chains


def optimize_buffer_load_voff(bb, chains):

    prologue = bb.preds[0]
    for chain in chains:
        ## mark copy inst as dead
        for inst in chain.copy_inst:
            inst.mark_dead = True
        pick_reg = pick_and_remove_contiguous_regs(bb.free_regs, 1, 'v')
        assert pick_reg
        free_reg = coalesce_regs(pick_reg)[0]
        ## rewrite cndmask instruction
        if chain.cndmask:
            chain.cndmask.replace_dst(free_reg)
            chain.cndmask.operands[1] = free_reg.emit()
            chain.cndmask.update_regs_by_operand()
            #logging.debug(f"replaced cndmask dst and op1 with {free_reg}")

        ## rewrite buffer_load voff
        for buffer_inst in chain.buffer_load:
            buffer_inst.replace_dst(free_reg)

        ## copy old voff into new voff
        old_reg = chain.entry_reg
        assert old_reg and old_reg.kind == 'a'
        inst_str = f'v_accvgpr_read_b32 {free_reg.emit()} {old_reg.emit()}'
        #logging.debug(f"writing {inst_str} in the prologue")
        prologue.add_inst(parse_instruction(inst_str, 0, prologue))

    cleanup_bb(bb)


def optimize_buffer_load_m0(bb, chains):

    for chain in chains:
        for buffer_load in chain.buffer_load:
            idx = bb.instructions.index(buffer_load)
            ## pattern 1: s_mov_b32 m0 --> s_nop 0 --> buffer_load --> mfma
            ## pattern 2: s_mov_b32 m0 --> buffer_load --> mfma
            ## swap buffer_load and mfma
            mfma = bb.instructions[idx + 1]
            assert 'mfma' in mfma.opcode
            bb.instructions[idx], bb.instructions[idx + 1] = bb.instructions[idx + 1], bb.instructions[idx]
            ## remove s_nop
            if 'nop' in bb.instructions[idx - 1].opcode:
                bb.instructions.pop(idx - 1)


def print_map(map, loc):
    logging.debug(f"{loc}: ")
    for regs in map[loc]:
        logging.debug(f"{regs} ")
    logging.debug("")


def construct_lds_reg_map():
    '''
    loc  tensor  regs
    754  A       a[0:63]
    755  B0      a[64:95]
    769  B1      a[96:127]
    783  A'      a[128:191]
    784  B0      a[64:95]
    803  B1      a[96:127]
    817  A       a[0:63]
    818  B0      a[64:95]
    846  B1      a[96:127]
    '''

    lds_reg_assignment = {}  # loc --> list[Register]
    regs_A = []
    for i in range(0, 64, 4):
        ids = [i + x for x in range(4)]
        kind = 'a'
        regs_A.append(Register(kind, ids))

    regs_B0 = []
    for i in range(64, 96, 4):
        ids = [i + x for x in range(4)]
        kind = 'a'
        regs_B0.append(Register(kind, ids))

    regs_B1 = []
    for i in range(96, 128, 4):
        ids = [i + x for x in range(4)]
        kind = 'a'
        regs_B1.append(Register(kind, ids))

    regs_A1 = []
    for i in range(128, 192, 4):
        ids = [i + x for x in range(4)]
        kind = 'a'
        regs_A1.append(Register(kind, ids))

    lds_reg_assignment[754] = regs_A
    lds_reg_assignment[755] = regs_B0
    lds_reg_assignment[769] = regs_B1
    lds_reg_assignment[783] = regs_A1
    lds_reg_assignment[784] = regs_B0
    lds_reg_assignment[803] = regs_B1
    lds_reg_assignment[817] = regs_A
    lds_reg_assignment[818] = regs_B0
    lds_reg_assignment[846] = regs_B1

    #print_map(lds_reg_assignment, 754)
    #print_map(lds_reg_assignment, 755)
    #print_map(lds_reg_assignment, 769)
    #print_map(lds_reg_assignment, 783)
    #print_map(lds_reg_assignment, 784)
    #print_map(lds_reg_assignment, 803)
    #print_map(lds_reg_assignment, 817)
    #print_map(lds_reg_assignment, 818)
    #print_map(lds_reg_assignment, 846)

    return lds_reg_assignment


def reassign_lds_regs(chains: list[LDSReadChain]):

    map = construct_lds_reg_map()


def optimize_nops(bb):

    for inst in bb.instructions:
        if 'nop' in inst.opcode:
            idx = bb.instructions.index(inst)
            ## The only allowed nop is the one between set m0 and buffer_load
            if idx == 0:
                inst.mark_dead = True
                continue
            if inst == bb.instructions[-1]:
                inst.mark_dead = True
                continue
            prev = bb.instructions[idx - 1]
            prev_dst_reg = prev.get_dst_regs()
            if prev_dst_reg is None:
                inst.mark_dead = True
                continue
            if prev_dst_reg.kind != 'm':
                inst.mark_dead = True
                continue
            suc = bb.instructions[idx + 1]
            if 'buffer_load' in suc.opcode:
                continue
            else:
                inst.mark_dead = True

    cleanup_bb(bb)


def find_loop_invariants(bb: BasicBlock):
    invariant_regs = set()
    invariant_insts = set()

    # Collect defs inside loop
    defs_in_loop = {}
    for inst in bb.instructions:
        for r in inst.defs:
            defs_in_loop.setdefault(r, set()).add(inst)

    # Step 1: live-in registers
    for inst in bb.instructions:
        for r in inst.uses:
            if r not in defs_in_loop:
                invariant_regs.add(r)

    changed = True
    while changed:
        changed = False
        for inst in bb.instructions:
            if inst in invariant_insts:
                continue
            if not inst.is_pure():
                continue
            if not inst.regs_by_operand:
                continue
            if inst.get_dst_regs() and inst.get_dst_regs().kind == 'm':
                continue

            if all(r in invariant_regs for r in inst.uses):
                invariant_insts.add(inst)
                for r in inst.defs:
                    if r not in invariant_regs:
                        invariant_regs.add(r)
                        changed = True

    return invariant_insts, invariant_regs


def can_hoist(inst, bb, invariant_regs):
    # Must be pure
    if not inst.is_pure():
        return False

    # Each dest reg must:
    # 1. Have exactly one reaching def inside loop (itself)
    reg = inst.get_dst_regs()
    assert reg
    reaching = bb.get_reaching_defs(inst, reg, True)
    if len(reaching) != 1 or inst not in reaching:
        return False

    # 2. No redefinition after inst
    redef = False
    for later in bb.instructions[inst.index + 1:]:
        if later.get_dst_regs():
            if flatten_regs(later.get_dst_regs()) & flatten_regs(reg):
                redef = True
                break
                #return False
    if redef:
        ## Try to use a different reg
        def_reg = coalesce_regs(inst.defs)[0]
        logging.debug(f"  inst redefined, trying to rewrite {def_reg.emit()}")
        kind, ids = def_reg.kind, def_reg.ids
        num = len(ids)
        free_reg = pick_and_remove_contiguous_regs(bb.free_regs, num, kind)
        if not free_reg:
            logging.debug(f"Not enough free regs")
            return False
        free_reg = coalesce_regs(free_reg)[0]
        logging.debug(f"  found free reg: {free_reg.emit()}")
        inst.replace_users_with(free_reg)
        #for user in inst.users:
        #    logging.debug(f"    {user.emit()}")
        inst.replace_reg(def_reg, free_reg)
        logging.debug(f"  new inst: {inst.emit()}")

    return True


def hoist_loop_invariants(bb: BasicBlock):
    invariant_insts, invariant_regs = find_loop_invariants(bb)

    hoistable = []
    for inst in invariant_insts:
        logging.debug(f"{inst.emit()=}")
        if can_hoist(inst, bb, invariant_regs):
            hoistable.append(inst)
            logging.debug(f"  can hoist!!")

    if not hoistable:
        return [], bb.instructions

    hoistable.sort(key=lambda i: i.index)

    new_loop = []
    hoisted_set = set(hoistable)

    for inst in bb.instructions:
        if inst not in hoisted_set:
            new_loop.append(inst)

    return hoistable, new_loop

def remove_debug_info_section(asm_text: str) -> str:
    """
    Remove the .debug_info section from an AMDGPU assembly file.
    """
    lines = asm_text.splitlines(keepends=True)
    output = []

    in_debug_section = False

    for line in lines:
        if line.strip().startswith(".section") and ".debug_info" in line:
            in_debug_section = True
            continue

        if in_debug_section:
            if line.strip().startswith(".Ldebug_info_end"):
                in_debug_section = False
            continue

        output.append(line)

    return "".join(output)

def amdgcn_as(text, verbose = False):

    setup_logging(debug=verbose)

    program = parse_asm(text)

    blocks = program.blocks
    loop = None
    prologue = None

    #########################################################
    ## 1st round
    #########################################################
    for bb in blocks:
        compute_bb_def_use(bb)
        if bb.name == '.LBB0_1':
            loop = bb
        if bb.name == '.LBB0_0':
            prologue = bb

    build_def_use_chains_linear(blocks)

    build_cfg(blocks)
    compute_liveness(blocks)
    #analyze_blocks(blocks)

    LDSChains = collect_ds_chains(blocks)
    reassign_lds_regs(LDSChains)

    mfmaChainsInLoop = collect_mfma_chains(loop)

    logging.debug(f"========== Analyze loop ==========")
    analyze_block(loop, mfmaChainsInLoop, LDSChains)
    logging.debug(f"========== Done Analyze loop ==========")

    optimize_mfma_accumulators(loop, mfmaChainsInLoop)

    clear_optimize_mfma_obstacles(loop, mfmaChainsInLoop)

    #########################################################
    ## 2nd round
    #########################################################
    logging.debug("")
    logging.debug("")
    for bb in blocks:
        compute_bb_def_use(bb)
    build_def_use_chains_linear(blocks)

    compute_liveness(blocks)

    mfmaChainsInLoop = collect_mfma_chains(loop)
    LDSChains = collect_ds_chains(blocks)

    optimize_mfma_accumulators(loop, mfmaChainsInLoop)

    for chain in mfmaChainsInLoop:
        zero_init = chain.get_zero_init(prologue)

    logging.debug(f"========== Analyze loop ==========")
    analyze_block(loop, mfmaChainsInLoop, LDSChains)
    logging.debug(f"========== Done Analyze loop ==========")

    optimize_nops(loop)

    bufferLoadChains = collect_buffer_load_chains(loop)

    logging.debug(f"========== Optimize buffer load voff ==========")
    optimize_buffer_load_voff(loop, bufferLoadChains)
    logging.debug(f"========== Done Optimize buffer load voff ==========")

    for bb in blocks:
        compute_bb_def_use(bb)
    build_def_use_chains_linear(blocks)

    compute_liveness(blocks)

    logging.debug(f"========== Analyze loop ==========")
    analyze_block(loop, mfmaChainsInLoop, LDSChains)
    logging.debug(f"========== Done Analyze loop ==========")

    optimize_buffer_load_m0(loop, bufferLoadChains)

    for bb in blocks:
        compute_bb_def_use(bb)
    build_def_use_chains_linear(blocks)

    compute_liveness(blocks)

    logging.debug(f"========== LICM ==========")
    hoisted, new_loop = hoist_loop_invariants(loop)
    loop.instructions = new_loop

    #logging.debug(f"Hoisting the following before the loop:")
    for inst in hoisted:
        logging.debug(f"{inst.emit()}")
        prologue.add_inst(inst)
    logging.debug(f"========== Done LICM ==========")

    # Suppose emit_program(program) returns a string of the assembly
    emitted_text = emit_program(program)

    emitted_text = remove_debug_info_section(emitted_text)

    setup_logging(debug=False)

    return emitted_text


if __name__ == "__main__":
    with open("/var/lib/jenkins/OAI-triton/study_matmul/gluon/v9/v9.amdgcn") as f:
        #with open("./v8/v8.amdgcn") as f:
        text = f.read()

    emitted_text = amdgcn_as(text)

    # Write to a file
    output_file = "./dump/new_asm.s"  # or whatever you want
    with open(output_file, "w") as f:
        f.write(emitted_text)

    logging.debug(f"Emitted program written to {output_file}")
