from collections import defaultdict, Counter, deque
import re

class Register:
    def __init__(self, kind, ids):
        self.kind = kind    # 's', 'v', 'a'
        self.ids = ids      # set of ints

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
    def __init__(self, opcode, operands, regs_by_operand, loc, raw_line):
        self.opcode = opcode
        self.operands = operands
        self.regs_by_operand = regs_by_operand  # List[List[Register]]
        self.loc = loc
        self.raw_line = raw_line

        self.defs = set()  # filled later
        self.uses = set()

        self.users = set()  # instructions that read registers produced by this instruction
        self.producers = set()  # instructions that write registers used by this instruction

    def emit(self):
        if not self.operands:
            return self.opcode
        return f"{self.opcode} " + ", ".join(self.operands)

    def get_dst_regs(self) -> list[Register]:
        if not self.operands:
            return []
        return [op for op in extract_registers(self.operands[0])]

    def get_src_regs(self) -> list[Register]:
        regs = []
        for op in self.operands[1:]:
            regs.extend(extract_registers(op))
        return regs

    def defines(self, reg) -> bool:
        #return any(r.overlaps(reg) for r in self.defs)
        #print(f"In defines: reg = {reg=}")
        if self.regs_by_operand:
            #print(f"{self.regs_by_operand[0]=}")
            #print(f"{self.regs_by_operand[1]=}")
            return self.regs_by_operand[0].overlaps(reg)
        else:
            return False

    def uses_regs(self, reg) -> bool:
        #return any(r.overlaps(reg) for r in self.uses)
        if len(self.uses) > 0:
            #print(f"{self.regs_by_operand[1]=}")
            return any(r.overlaps(reg) for r in self.uses)
        else:
            return False

    def replace_reg(self, old: Register, new: Register):
        for op in self.operands:
            op.replace_reg(old, new)

        if old in self.defs:
            self.defs.remove(old)
            self.defs.add(new)

        if old in self.uses:
            self.uses.remove(old)
            self.uses.add(new)

    # MFMA-only helpers
    def is_mfma(self) -> bool:
        return self.opcode.startswith("v_mfma")

    def get_mfma_dst(self) -> Register:
        assert self.is_mfma()
        regs = self.get_dst_regs()
        assert len(regs) == 1
        return regs[0]

    def get_mfma_acc(self) -> Register:
        assert self.is_mfma()
        regs = extract_registers(self.operands[-1])
        assert len(regs) == 1
        return regs[0]

    def replace_mfma_dst(self, reg: Register):
        assert self.is_mfma()
        self.operands[0] = reg.emit()
        self.defs = reg

    def replace_mfma_acc(self, reg: Register):
        assert self.is_mfma()
        self.operands[-1] = reg.emit()


class BasicBlock:
    def __init__(self, name):
        self.name = name
        self.instructions = []
        self.succs = []   # filled later
        self.preds = []

        self.defs = set()
        self.uses = set()

        self.live_in = set()
        self.live_out = set()

    def add_inst(self, inst):
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

        return self.instructions[idx_a + 1 : idx_b]

    def instructions_before(self, inst):
        idx = self.instructions.index(inst)
        return self.instructions[:idx]

    def get_reaching_defs(self, inst, reg):
        """
        reg: Register (possibly a range)
        Returns: set of Instructions
        """
        needed = set(reg.ids)
        reaching_defs = set()

        for prev in reversed(self.instructions_before(inst)):
            for dreg in prev.get_dst_regs():
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
        self.header_lines = []     # before first BB
        self.blocks = []           # parsed basic blocks
        self.tail_lines = []       # after s_endpgm

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
        self.acc_regs: set[Register] = set()
        self.canonical: Register | None = None
        self.entry_acc: Register = None
        self.exit_acc: Register = None

REG_SINGLE = re.compile(r'([sva])(\d+)')
REG_RANGE  = re.compile(r'([sva])\[(\d+):(\d+)\]')
REG_PATTERNS = [
    r's\d+',
    r'v\d+',
    r'a\d+',
    r'm0',
    r's\[\d+:\d+\]',
    r'v\[\d+:\d+\]',
    r'a\[\d+:\d+\]',
]

def parse_register(text):
    if text == 'm0':
        return Register('m', [0])

    kind = text[0]
    if '[' in text:
        lo, hi = map(int, text[text.find('[')+1:text.find(']')].split(':'))
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


def parse_instruction(line, loc):
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

    return Instruction(opcode, operands, regs_by_operand, loc, line)

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
                inst = parse_instruction(line, cur_loc)
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
        inst = parse_instruction(line, cur_loc)
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
    's_waitcnt', 's_nop',
    's_branch', 's_cbranch_scc0', 's_cbranch_scc1',
}

CMP_PREFIXES = ('s_cmp_', 'v_cmp_')

## TODO(lixun)
## Only buffer_load lds should be included in ALL_USERS set
ALL_USERS = ('s_cmp_', 'v_cmp_', 'v_permlane', 'buffer_store', 'buffer_load')
ALL_DEFS = ('v_permlane')
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
            inst.uses.add(reg)

    if inst.opcode.startswith(ALL_DEFS):
        for reg in inst.regs_by_operand:
            inst.defs.add(reg)
        return

    # Normal case
    if inst.regs_by_operand:
        inst.defs.add(inst.regs_by_operand[0])
        for reg in inst.regs_by_operand[1:]:
            inst.uses.add(reg)

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
            # ---------
            # Uses: find producers
            # ---------
            #if 'v_permlane' in inst.opcode:
            #    print(inst.emit())
            #    print(f"  defs: {inst.defs}")
            #    print(f"  uses: {inst.uses}")
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
    print("")
    print("============================================================")
    for bb in blocks:
        print(f"BasicBlock {bb.name}:")

        # --------------------------------------------------
        # Instruction histogram
        # --------------------------------------------------
        inst_hist = Counter()
        for inst in bb.instructions:
            inst_hist[(inst.opcode, inst.loc[1])] += 1

        print("  Instruction histogram:")
        for op, cnt in inst_hist.most_common():
            print(f"    {op[1]:3d}:{op[0]:<40} {cnt}")

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
                print(f"    {name}: none")
                return

            ranges = collapse_ranges(ids)
            print(f"    {name}: {len(ids)} regs")
            print(f"      ranges: {', '.join(ranges)}")

        def dump_reg_set(regset, name):
            by_kind = {}
            for k, r in regset:
                by_kind.setdefault(k, set()).add(r)

            print(f"  {name}:")
            for kind, ids in by_kind.items():
                ranges = collapse_ranges(sorted(ids))
                print(f"    {kind}: {', '.join(ranges)}")


        print("  Register usage:")
        dump_regs('s', "SGPR")
        dump_regs('v', "VGPR")
        dump_regs('a', "AGPR")
        dump_regs('m', "Special")

        # --------------------------------------------------
        # Liveness
        # --------------------------------------------------
        print("  Liveness:")
        dump_reg_set(bb.live_in, "Live-in")
        dump_reg_set(bb.live_out, "Live-out")

        print("--------------------------------------------------------")

def adjust_regs(blocks):
    loop = None
    for bb in blocks:
        if bb.name == '.LBB0_1':
            loop = bb

    if loop is None:
        return

    worklist = set_queue()

    for inst in loop.instructions:
        if 'mfma' in inst.opcode:
            if inst.loc[1] == 766:
                acc_new = inst.regs_by_operand[0]
                acc_old = inst.regs_by_operand[3]
                print(f"{acc_new} <-- {acc_old}")
                #print(inst.emit())
                #for prod in inst.producers:
                #    print(f"{prod.opcode} " + ", ".join(prod.operands))

                worklist.push(inst)
                while worklist.isNotEmpty():
                    cur_inst = worklist.pop()
                    print(cur_inst.emit())
                    for user in cur_inst.users:
                        worklist.push(user)

                break

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

            #print(f"visiting mfma: {cur.emit()}")

            if cur.is_mfma():
                chain.mfmas.append(cur)
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
            #print(f"How many mfma in the chain: {len(chain.mfmas)}")

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

    return chains

def choose_canonical_acc(chain, bb):
    candidate = chain.mfmas[-1].get_mfma_acc()

    print(f"Choose canon acc!! {candidate=} between:")
    print(f"    {chain.mfmas[-1].emit()}")
    print(f"    {chain.mfmas[0].emit()}")

    for inst in bb.instructions_between(
        chain.mfmas[-1], chain.mfmas[0]
    ):
        #print(f"{inst.emit()=}   ", end="")
        if inst in chain.mfmas or inst in chain.copy_instrs:
            #print("on the chain, continue")
            continue
        #if inst.defines(candidate) or inst.uses(candidate):
        
        if inst.defines(candidate):
            #print("liveness interleave, return none")
            #print(f"    oops, {inst.emit()} defines {candidate}")
            return None
        if inst.uses_regs(candidate):
            #print("liveness interleave, return none")
            #print(f"    oops, {inst.emit()} uses {candidate}")
            return None

        #print("un-related, continue")

    chain.canonical = candidate
    return candidate

def rewrite_chain(chain):
    canon = chain.canonical
    assert canon is not None

    #print(f"need to rewrite a chain with {canon=}")

    for mfma in chain.mfmas:
        #print(f"before: {mfma.emit()=}  {mfma.operands=}")
        mfma.replace_mfma_dst(canon)
        #print(f"after: {mfma.emit()=}")
        mfma.replace_mfma_acc(canon)
        #print(f"after: {mfma.emit()=}")

    for inst in chain.copy_instrs:
        inst.mark_dead = True

def cleanup_bb(bb):
    bb.instructions = [
        inst for inst in bb.instructions
        if not getattr(inst, "mark_dead", False)
    ]

def get_entry_exit_acc_reg(bb, chain):
    ## entry acc is the reaching definition of the acc regs of chain.mfmas[-1]
    mfma = chain.mfmas[-1]
    worklist = []
    for reaching_def in bb.get_reaching_defs(mfma, mfma.get_mfma_acc()):
        worklist.append(reaching_def)

    final_reaching_defs = []

    while worklist:
        cur = worklist.pop()
        reaching_defs = bb.get_reaching_defs(cur, cur.get_dst_regs()[0])
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

    return coalesce_regs(entry_acc_regs)


    ## exit acc is the last user of the dst reg of chain.mfmas[0]

    

def optimize_mfma_accumulators(bb):
    chains = collect_mfma_chains(bb)

    print(f"how many chains: {len(chains)}")

    entry_regs = []
    for chain in chains:
        entry_reg = get_entry_exit_acc_reg(bb, chain)
        entry_regs.append(entry_reg[0])
        print(f"{entry_reg=}")
        if len(chain.copy_instrs) == 0:
            continue
        if choose_canonical_acc(chain, bb) is None:
            continue

        rewrite_chain(chain)

    entry_acc_flatten = flatten_regs(entry_regs)
    entry_acc = coalesce_regs(entry_acc_flatten)
    print(f"{len(entry_acc_flatten)}: {entry_acc=}")

    cleanup_bb(bb)


if __name__ == "__main__":
    with open("/var/lib/jenkins/OAI-triton/study_matmul/gluon/v7/v7.amdgcn") as f:
        text = f.read()

    program = parse_asm(text)

    blocks = program.blocks
    loop = None
    for bb in blocks:
        compute_bb_def_use(bb)
        if bb.name == '.LBB0_1':
            loop = bb

    build_def_use_chains_linear(blocks)

    build_cfg(blocks)
    compute_liveness(blocks)
    #analyze_blocks(blocks)

    optimize_mfma_accumulators(loop)

    # Suppose emit_program(program) returns a string of the assembly
    emitted_text = emit_program(program)

    # Write to a file
    output_file = "/var/lib/jenkins/OAI-triton/study_matmul/asm_tool/new_asm.s"  # or whatever you want
    with open(output_file, "w") as f:
        f.write(emitted_text)

    print(f"Emitted program written to {output_file}")
