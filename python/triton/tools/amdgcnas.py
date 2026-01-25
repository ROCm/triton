from collections import defaultdict, Counter, deque
from typing import Set, Tuple, Optional
from operator import attrgetter
import re
import logging
import argparse
import bisect

NO_DEF_OPS = {
    's_waitcnt',
    's_nop',
    's_branch',
    's_cbranch_scc0',
    's_cbranch_scc1',
}

CMP_PREFIXES = ('s_cmp', 'v_cmp')

## TODO(lixun)
## Only buffer_load lds should be included in ALL_USERS set
ALL_USERS = ('s_cmp', 'v_cmp', 'v_permlane', 'buffer_store', 'buffer_load')
ALL_DEFS_USES = ('v_permlane')
COPY_DATA = ('v_accvgpr_read', 'v_accvgpr_write', 'v_accvgpr_mov', 'v_mov', 'scratch_load', 'scratch_store')


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def dbg(msg, indent=0):
    logging.debug(" " * indent + msg)


class Register:

    def __init__(self, kind, ids):
        self.kind = kind  # 's', 'v', 'a', 'm', 'l'
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

        self.lds_group: DSReadGroup = None
        self.mfma_chain: AccumulatorChain = None

        self.parent_bb = bb
        self.index = None  # position inside BB

        self.mark_dead: bool = False
        self.is_mfma_copy: bool = False

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

    ## Replace old_reg with new_reg in this instruction's src regs
    def replace_use_reg(self, old_reg: Register, new_reg: Register, repeat=10):
        cnt = 0
        for i, op in enumerate(self.operands):
            if i == 0:
                continue
            if op == old_reg.emit():
                self.operands[i] = new_reg.emit()
                cnt += 1
            if cnt >= repeat:
                break
        self.update_regs_by_operand()

    ## Replace the uses of this inst.def with reg
    def replace_users_with(self, reg: Register, repeat=10):
        #def_reg = coalesce_regs(self.defs)[0]
        def_reg = self.get_dst_regs()
        for user in self.users:
            if user.defs:
                user.replace_use_reg(def_reg, reg, repeat)
            else:
                user.replace_reg(def_reg, reg)

    # ---------- classification ----------
    def is_memory(self):
        return (self.opcode.startswith("ds_") or self.opcode.startswith("buffer_"))

    def is_control(self):
        return self.opcode.startswith("s_branch") or self.opcode.startswith("s_cbranch")

    def is_cmp(self):
        return self.opcode.startswith("s_cmp") or self.opcode.startswith("v_cmp")

    def is_pure(self):
        return not (self.is_memory() or self.is_control() or self.is_cmp())

    # ---------- MFMA-only helpers ----------
    def is_mfma(self) -> bool:
        return self.opcode.startswith("v_mfma")

    def get_mfma_dst(self) -> Register:
        assert self.is_mfma()
        regs = self.get_dst_regs()
        assert regs
        return regs

    def get_mfma_acc(self) -> Register:
        assert self.is_mfma()
        regs = extract_registers(self.operands[3])
        assert len(regs) == 1
        return regs[0]

    def replace_mfma_dst(self, reg: Register):
        assert self.is_mfma()
        self.operands[0] = reg.emit()
        self.update_regs_by_operand()

    def replace_mfma_acc(self, reg: Register):
        assert self.is_mfma()
        self.operands[3] = reg.emit()
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

    # ---------- copy inst ----------
    def is_copy(self):
        return self.opcode.startswith(COPY_DATA)

    def get_copy_src(self) -> Register | str:
        '''
        return the source register of this copy instruction
        src can be
        1. s, a, v
        2. l: this refers to a memory loc with off, e.g.
           l0 means the src of
           scratch_load_dword off v0, off, off
           l4 means the src of
           scratch_load_dword off v0, off, off offset:4
        '''
        assert self.is_copy()
        if self.opcode.startswith("v_"):
            if self.get_src_regs():
                return self.get_src_regs()[0]
            else:
                return self.operands[1]
        if "load" in self.opcode:
            off = self.operands[-1]
            if 'offset' not in off:
                return Register('l', [0])
            else:
                off = off.split(':')[1]
                return Register('l', [int(off)])
        if "store" in self.opcode:
            return parse_register(self.operands[1])

    def get_copy_dst(self) -> Register:
        assert self.is_copy()
        if self.opcode.startswith("v_"):
            return self.get_dst_regs()
        if "load" in self.opcode:
            return self.get_dst_regs()
        if "store" in self.opcode:
            off = self.operands[-1]
            if 'offset' not in off:
                return Register('l', [0])
            else:
                off = off.split(':')[1]
                return Register('l', [int(off)])

    def compute_def_use(self):
        self.defs.clear()
        self.uses.clear()

        if self.opcode in NO_DEF_OPS:
            return

        if 'scratch_load' in self.opcode:
            self.defs |= flatten_regs(self.regs_by_operand[0])
            return

        if 'scratch_store' in self.opcode:
            self.uses |= flatten_regs(self.regs_by_operand[0])
            return

        if self.opcode.startswith(ALL_USERS):
            for reg in self.regs_by_operand:
                self.uses |= flatten_regs(reg)
            return

        if self.opcode.startswith(ALL_DEFS_USES):
            for reg in self.regs_by_operand:
                self.defs |= flatten_regs(reg)
                self.uses |= flatten_regs(reg)
            return

        # Normal case
        if self.regs_by_operand:
            self.defs |= flatten_regs(self.regs_by_operand[0])
            if len(self.regs_by_operand) > 1:
                for reg in self.regs_by_operand[1:]:
                    self.uses |= flatten_regs(reg)


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
        ## memory locations this block read from and write to
        ## set of (kind, idx) objects
        self.read_from = set()
        self.write_to = set()

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

        #dbg(f"Looking for reaching defs for {needed}")

        for prev in reversed(self.instructions_before(inst, including)):
            #dbg(f"Examining {prev.emit()}")
            if not prev.defs:
                #dbg(f"  no defs, pass")
                continue
            dreg = coalesce_regs(prev.defs)[0]
            if not dreg:
                #dbg(f"  no dreg, pass")
                continue
            if dreg.kind != reg.kind:
                #dbg(f"  defines a different kind, pass")
                continue

            overlap = needed & set(dreg.ids)
            if overlap:
                #dbg(f"  defines ovelap {dreg}, adding to reaching_defs")
                reaching_defs.add(prev)
                needed -= overlap
                #dbg(f"  now needed: {needed}")

            if not needed:
                break

        return reaching_defs

    def is_loop(self):
        if len(self.preds) == 0 or len(self.succs) == 0:
            return False
        return (self in self.preds) and (self in self.succs)

    def is_prologue(self):
        if len(self.succs) == 0:
            return False

        for succ in self.succs:
            if succ != self and succ.is_loop():
                return True

        return False

    def is_epilogue(self):
        if len(self.preds) == 0:
            return False

        for pred in self.preds:
            if pred != self and pred.is_loop():
                return True

        return False

    def cleanup_bb(self):
        self.instructions = [inst for inst in self.instructions if not getattr(inst, "mark_dead", False)]

    def swap_inst(self, ida, idb):
        s = len(self.instructions)
        if ida >= s or idb >= s:
            return
        if ida == idb:
            return
        self.instructions[ida], self.instructions[idb] = self.instructions[idb], self.instructions[ida]

    def compute_bb_def_use(self):
        self.defs.clear()
        self.uses.clear()

        for inst in self.instructions:
            if inst.mark_dead:
                continue
            inst.compute_def_use()

            for u in inst.uses:
                if u not in self.defs:
                    self.uses.add(u)

            self.defs |= inst.defs

            #if self.is_epilogue() and inst.is_copy():
            #    print(f"{inst.emit()}")
            #    print(f"{coalesce_regs(self.defs)}")
            #    print(f"{coalesce_regs(self.uses)}")

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
        self.LDSChains = []
        self.mfmaChains = []

    def get_prologue(self):
        for bb in self.blocks:
            if bb.is_prologue():
                return bb

    def get_loop(self):
        for bb in self.blocks:
            if bb.is_loop():
                return bb

    def get_epilogue(self):
        for bb in self.blocks:
            if bb.is_epilogue():
                return bb

    def update_inst_index(self):
        for bb in self.blocks:
            for idx, inst in enumerate(bb.instructions):
                inst.index = idx

    def compute_liveness(self, indent):

        dbg("========== liveness ==========", indent)
        changed = True
        cnt = 0
        for bb in self.blocks:
            bb.live_in = set()
            bb.live_out = set()

        while changed:
            changed = False

            #dbg(f"{cnt=}")

            for bb in reversed(self.blocks):
                new_out = set()
                for s in bb.succs:
                    new_out |= s.live_in

                new_in = bb.uses | (new_out - bb.defs)

                if new_out != bb.live_out or new_in != bb.live_in:
                    bb.live_out = new_out
                    bb.live_in = new_in
                    changed = True

                #dbg(f"{bb.name} defs: {coalesce_regs(bb.defs)}")
                #dbg(f"{bb.name} uses: {coalesce_regs(bb.uses)}")
                #dbg(f"{bb.name} live_in: {coalesce_regs(bb.live_in)}")
            cnt += 1

        dbg("========== liveness done =====", indent)

    def build_cfg(self):
        label_map = {bb.name: bb for bb in self.blocks}

        for i, bb in enumerate(self.blocks):
            bb.preds = []
            bb.succs = []
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
                if i + 1 < len(self.blocks):
                    bb.succs.append(self.blocks[i + 1])

            else:
                if i + 1 < len(self.blocks):
                    bb.succs.append(self.blocks[i + 1])

        for bb in self.blocks:
            for s in bb.succs:
                s.preds.append(bb)

        #logging.debug("========== cfg info ==========")
        #for bb in self.blocks:
        #    msg = "is "
        #    if bb.is_loop():
        #        msg += "loop "
        #    if bb.is_prologue():
        #        msg += "prologue "
        #    if bb.is_epilogue():
        #        msg += "epilogue"
        #    logging.debug(f"{bb.name}: {msg}")
        #    logging.debug("  pred:")
        #    for pred in bb.preds:
        #        logging.debug(f"    {pred.name}")
        #    logging.debug("  succ:")
        #    for succ in bb.succs:
        #        logging.debug(f"    {succ.name}")
        #logging.debug("========== done cfg info ==========")

    def update_free_regs(self, indent):

        dbg("========== update blocks regs ==========", indent)
        for bb in self.blocks:
            bb.compute_bb_def_use()

        self.build_cfg()
        self.compute_liveness(indent + 2)

        all_regs = set()
        for i in range(512):
            if i > 255:
                kind = 'a'
                id = i - 256
                all_regs.add((kind, id))
            else:
                kind = 'v'
                id = i
                all_regs.add((kind, id))

        for bb in self.blocks:
            live_in = bb.live_in
            bb_uses = bb.defs | bb.uses
            live_through = live_in - bb_uses
            free_regs = all_regs - bb_uses - live_through
            bb.free_regs = free_regs
            ## collect memory locations for scratch load and store
            bb.read_from = set()
            bb.write_to = set()
            for inst in bb.instructions:
                if inst.mark_dead:
                    continue
                if 'scratch_load' in inst.opcode:
                    bb.read_from |= flatten_regs(inst.get_copy_src())
                if 'scratch_store' in inst.opcode:
                    bb.write_to |= flatten_regs(inst.get_copy_dst())

        dbg("========== update blocks regs done =====", indent)

    def build_def_use_chains_linear(self, indent):
        """
        Correct reaching-definition-based def-use chains across blocks
        """

        dbg("========== build def-use chains ==========", indent)
        current_def = {}  # (kind, id) -> Instruction

        for bb in self.blocks:
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

        dbg("========== build def-use chains done =====", indent)

    def process_blocks(self, indent):

        dbg("========== process blocks =========", indent)

        self.update_free_regs(indent + 2)

        self.build_def_use_chains_linear(indent + 2)

        dbg("========== process blocks done ====", indent)

    def collect_ds_chains(self, indent):

        dbg("========== collecting lds chains ==========", indent)
        groups = []

        ## collect all the ds groups

        ## A map from (mfma_inst, opIdx) to its corresponding DSReadGroup
        ds_map = defaultdict(DSReadGroup)
        for bb in self.blocks:
            if bb.is_epilogue():
                continue
            for inst in bb.instructions:
                if 'ds_read_b128' in inst.opcode:
                    used_by_mfma = True
                    for user in inst.users:
                        if not user.is_mfma():
                            used_by_mfma = False
                            break
                    if not used_by_mfma:
                        dbg(f"{inst.emit()} has non mfma users !!!", indent)
                        for user in inst.users:
                            dbg(f"{user.raw_line}", indent + 2)
                        continue

                    data_reg = inst.get_dst_regs()
                    firstMfma = min(inst.users, key=lambda i: i.index)
                    assert 'mfma' in firstMfma.opcode

                    opIdx = firstMfma.get_op_idx(data_reg)
                    assert opIdx == 1 or opIdx == 2

                    key = (firstMfma, opIdx)
                    if key in ds_map:
                        dbg(f"{inst.emit()} added into exisitng group", indent)
                        ## the ds_group already exist, add this ds_inst
                        ds_group = ds_map[key]
                        ds_group.insert_ds(inst)
                        ds_group.users |= inst.users
                    else:
                        dbg(f"{inst.emit()} added into new group", indent)
                        ## the ds_group does not exist
                        ds_group = DSReadGroup(inst)
                        ds_group.users = inst.users
                        ds_map[key] = ds_group
                        groups.append(ds_group)

                    ds_group.update_data()
                    ds_group.opIdx = opIdx
                    ds_group.loc = inst.loc[1]
                    ds_group.addr = inst.get_src_regs()[0]
                    if firstMfma not in bb.instructions:
                        ds_group.isLiveAcrossBB = True
                    inst.lds_group = ds_group

        dbg(f"Collected {len(groups)} groups")
        ## construct ds read chains based on all the collect groups
        chains = []

        loc_to_groups = defaultdict(list)  # list of LDSChains of the same loc
        for group in groups:
            loc_to_groups[group.loc].append(group)

        for loc, groups_in_map in loc_to_groups.items():
            regs = set()
            crossLive = None
            chain = DSReadChain(loc)
            for group in groups_in_map:
                chain.ds_groups.append(group)
                regs |= flatten_regs(group.data)
                if crossLive is None:
                    crossLive = group.isLiveAcrossBB
                elif crossLive != group.isLiveAcrossBB:
                    dbg("Error: partial ds_read is live across bb !!", indent)
                    assert False
                else:
                    crossLive = group.isLiveAcrossBB
                chain.isLiveAcrossBB = crossLive

            regs = coalesce_regs(regs)
            chain.regs = regs
            chains.append(chain)

        dbg(f"collected {len(chains)} lds chains", indent)
        dbg("========== collecting lds chains done =====", indent)
        self.LDSChains = chains

    def collect_mfma_chains(self, indent):
        '''
        Collect a chain of insrtuctions that includes
        1. mfma instructions that are supposed to use the same regs for acc
        2. copy instructions that push and pull regs for acc
           copy instructions refer to inst start with COPY_DATA

        The alg goes backward to collect mfma and copy instructions.
        Then it goes forward to collect copy instructions after the last mfma on the chain.
        '''
        bb = self.get_loop()
        dbg("====== collect mfma chains ======", indent)
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

                ## Found mfma
                if cur.is_mfma():
                    chain.mfmas.append(cur)
                    cur.mfma_chain = chain
                    next = bb.next_instruction(cur)
                    if next and 'nop' in next.opcode:
                        chain.nops.append(next)
                        next.mfma_chain = chain
                    chain.acc_regs.add(cur.get_mfma_dst())

                    acc = cur.get_mfma_acc()
                    prods = bb.get_reaching_defs(cur, acc)

                    for prod in prods:
                        if prod and prod not in visited:
                            worklist.append(prod)

                ## Found copy
                elif cur.opcode.startswith(COPY_DATA):
                    chain.copy_instrs.append(cur)
                    cur.mfma_chain = chain
                    for r in cur.get_src_regs():
                        chain.acc_regs.add(r)

                    for r in cur.get_src_regs():
                        prods = bb.get_reaching_defs(cur, r)
                        for prod in prods:
                            if prod and prod not in visited:
                                worklist.append(prod)

                ## Something else?
                else:
                    dbg(f"Not expected to see {cur.emit()} on the chain", indent)
                    assert False

            if len(chain.mfmas) >= 2:
                chains.append(chain)
            else:
                dbg("less than 2 mfma in the chain !!", indent)

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
                cur.mfma_chain = chain

                for user in cur.users:
                    if user in bb.instructions:
                        worklist.append(user)

            get_entry_exit_acc_reg(bb, chain)

        dbg(f"collected {len(chains)} chains", indent)
        entry_acc = set()
        for chain in chains:
            entry_acc |= flatten_regs(chain.entry_acc)
        entry_acc = coalesce_regs(entry_acc)
        num_a, num_v = count_regs(entry_acc)
        dbg(f"entry acc ({num_a} x a, {num_v} x v): {entry_acc}", indent)
        dbg("========== collect mfma chains done =====", indent)

        self.mfmaChains = chains

    def rewrite_mfma_acc(self, min_a, indent):
        logging.debug("========== rewrite mfma acc regs ===========")

        entry_acc = set()
        for chain in self.mfmaChains:
            entry_acc |= flatten_regs(chain.entry_acc)
            if chain.exit_acc != chain.entry_acc:
                logging.debug("mismatch acc !!")
                return
            #inst_chain = {}
            #if len(chain.copy_instrs) > 0:
            #    dbg(f"This chain (acc: {chain.entry_acc}) has {len(chain.copy_instrs)} copy instructions and {len(chain.nops)} nops", indent)
            #    for mfma in reversed(chain.mfmas):
            #        idx = self.get_loop().instructions.index(mfma)
            #        #dbg(f"{idx}: {mfma.emit()}", indent + 2)
            #        inst_chain[idx] = mfma
            #    for copys in chain.copy_instrs:
            #        idx = self.get_loop().instructions.index(copys)
            #        #dbg(f"{idx}: {copys.emit()}", indent + 2)
            #        inst_chain[idx] = copys
            #        copys.is_mfma_copy = True
            #if len(chain.copy_instrs) > 0:
            canon = chain.entry_acc
            chain.set_canon(canon)
            chain.rewrite()

            #inst_chain = dict(sorted(inst_chain.items()))
            #for idx, inst in inst_chain.items():
            #    dbg(f"{idx}: {inst.emit()}", indent + 2)

            #cnt = 0
            #for mfma in chain.mfmas:
            #    if mfma.get_mfma_acc() != chain.entry_acc or mfma.get_mfma_dst() != chain.entry_acc:
            #        cnt += 1

            #if cnt > 0:
            #    dbg(f"This chain (acc: {chain.entry_acc}) uses extra regs", indent)
            #    dbg(f"before rewriting: ", indent + 2)
            #    for mfma in chain.mfmas:
            #        dbg(f"{mfma.emit()}", indent + 4)
            #    canon = chain.entry_acc
            #    chain.set_canon(canon)
            #    chain.rewrite()
            #    dbg(f"after rewriting: ", indent + 2)
            #    for mfma in chain.mfmas:
            #        dbg(f"{mfma.emit()}", indent + 4)

        acc_a, acc_v = count_regs(coalesce_regs(entry_acc))
        logging.debug(f"entry-in acc: a={acc_a} v={acc_v} {coalesce_regs(entry_acc)}")

        self.update_free_regs(indent + 2)

        loop_free_regs = coalesce_regs(self.get_loop().free_regs)
        free_a, free_v = count_regs(loop_free_regs)
        logging.debug(f"free reg of loop ({free_a} x a, {free_v} x v): {loop_free_regs}")

        if acc_a < min_a:
            logging.debug(f"not enough vgpr left: {acc_a=}, {min_a=}")

            for chain in self.mfmaChains:
                my_acc = chain.entry_acc
                if my_acc.kind == 'a':
                    continue
                assert my_acc.kind == 'v'

                cnt = len(my_acc.ids)
                new_acc = pick_and_remove_contiguous_regs(self.get_loop().free_regs, cnt, 'a')
                if not new_acc:
                    break

                ## rewrite this chain again with a agpr
                new_acc = coalesce_regs(new_acc)[0]
                logging.debug(f"rewriting chain {chain.root.emit()} with {new_acc}")
                chain.set_canon(new_acc)
                chain.rewrite()

                ## fix zero init in prologue
                new_acc_list = sorted(flatten_regs(new_acc), key=lambda x: x[1])
                for i, zero_init in enumerate(chain.get_zero_init(self.get_prologue())):
                    new_reg = coalesce_regs([new_acc_list[i]])[0]
                    logging.debug(f"zero init: {zero_init.emit()} to be replaced by {new_reg} ==>")
                    zero_init.replace_dst(new_reg)
                    zero_init.opcode = 'v_accvgpr_write_b32'
                    logging.debug(f"zero init: {zero_init.emit()}")

                ## fix users in epilogue
                my_acc_list = sorted(flatten_regs(my_acc), key=lambda x: x[1])
                new_insts = []
                epi = self.get_epilogue()
                for i, the_old_acc in enumerate(my_acc_list):
                    the_new_acc = new_acc_list[i]
                    assert the_old_acc[0] == 'v' and the_new_acc[0] == 'a'
                    opcode = 'v_accvgpr_read_b32'
                    dsts = coalesce_regs([the_old_acc])
                    srcs = coalesce_regs([the_new_acc])
                    operands = []
                    regs_by_operand = []
                    operands.append(dsts[0].emit())
                    operands.append(srcs[0].emit())
                    regs_by_operand.append(dsts)
                    regs_by_operand.append(srcs)
                    raw_line = f"{opcode} {operands[0]}, {operands[1]}"
                    loc = -1
                    copy_inst = Instruction(opcode, operands, regs_by_operand, loc, raw_line, epi)
                    new_insts.append(copy_inst)
                    logging.debug(f"new copy inst: {copy_inst.emit()}")

                epi.instructions = new_insts + epi.instructions

                entry_acc -= flatten_regs(my_acc)
                entry_acc |= flatten_regs(new_acc)
                chain.entry_acc = new_acc
                chain.exit_acc = new_acc

                acc_a += cnt
                if acc_a >= min_a:
                    break

            if acc_a < min_a:
                logging.debug(f"still not enough vgpr left: {acc_a=}, {min_a=}, but no more free agprs")

            self.update_free_regs(indent + 2)

            loop_free_regs = coalesce_regs(self.get_loop().free_regs)
            free_a, free_v = count_regs(loop_free_regs)
            logging.debug(f"free reg of loop ({free_a} x a, {free_v} x v): {loop_free_regs}")

            acc_a, acc_v = count_regs(coalesce_regs(entry_acc))
            logging.debug(f"entry-in acc: a={acc_a} v={acc_v} {coalesce_regs(entry_acc)}")

        logging.debug("========== rewrite mfma acc regs done ======")

        return entry_acc


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
        '''
        1. Replace the dst and acc register of all mfma's on this chain
        with canonical.
        2. Mark all copy instructions as mark_dead
        '''
        canon = self.canonical
        assert canon is not None

        for mfma in self.mfmas:
            mfma.replace_mfma_dst(canon)
            mfma.replace_mfma_acc(canon)

        for inst in self.copy_instrs:
            inst.mark_dead = True

        #for inst in self.nops:
        #    inst.mark_dead = True

    def set_canon(self, reg):
        self.canonical = reg

    def get_zero_init(self, prologue: BasicBlock):
        self.zero_init = prologue.get_reaching_defs(prologue.instructions[-1], self.entry_acc, True)
        return self.zero_init


class DSReadGroup:
    '''
    One DSReadGroup contians
    1. one or more ds_read instruction as the root --> ds: list[inst]
       The operand of one mfma can take more registers than a single ds_read.
    2. All user mfma instructions of the ds_read --> users: set[Instruction]
    3. which opIdx of the mfma is using the loaded registers of the ds_read --> opIdx: int
    4. registers to hold the loaded data --> data: Register
    5. register to hold the address --> addr: register
    6. location of this ds_read in the source code --> loc: int
    '''

    def __init__(self, root_ds: Instruction):
        self.ds = [root_ds]
        self.users: set[Instruction] = []
        self.opIdx = 0
        self.data: Register = None
        self.addr: Register = None
        self.loc = 0
        self.isLiveAcrossBB: bool = False

    def update_data(self):
        regs = set()
        for ds_inst in self.ds:
            regs |= flatten_regs(ds_inst.get_dst_regs())
        self.data = coalesce_regs(regs)[0]

    def insert_ds(self, ds_inst):
        '''
        Insert `ds_inst` into self.ds while keeping the register order
        We assume the registers are of the same kind
        '''
        keys = [inst.get_dst_regs().start for inst in self.ds]
        pos = bisect.bisect_left(keys, ds_inst.get_dst_regs().start)
        self.ds.insert(pos, ds_inst)


class DSReadChain:
    '''
    A DSReadChain object represents all of the ds_read sharing the same
    location in the source code. The loaded data usually represents a tensor
    or a sub-tensor as one operand of the dot operation.
    It contains
    1. location in the source dode --> loc: int
    2. All the ds_read instructions --> ds_groups: set[DSReadGroup]
    3. If the users of the ds_read are in a different block --> isLiveAcrossBB: bool
    '''

    def __init__(self, loc: int):
        self.loc = loc
        self.ds_groups: list[DSReadGroup] = []
        self.isLiveAcrossBB: bool = False
        self.regs: list[Register] = []

    def update_regs(self):
        regs = []
        for ds_group in self.ds_groups:
            ds_group.update_data()
            regs.append(ds_group.data)
        regs = coalesce_regs(flatten_regs(regs))
        self.regs = regs

    def within_bb(self, bb):
        if self.ds_groups:
            return self.ds_groups[0].ds[0] in bb.instructions
        else:
            return False


REG_SINGLE = re.compile(r'([sva])(\d+)')
REG_RANGE = re.compile(r'([sva])\[(\d+):(\d+)\]')

REG_PATTERNS = [
    r'(?<![A-Za-z0-9_])s\d+(?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])v\d+(?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])a\d+(?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])m0(?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])s\[\d+:\d+\](?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])v\[\d+:\d+\](?![A-Za-z0-9_])',
    r'(?<![A-Za-z0-9_])a\[\d+:\d+\](?![A-Za-z0-9_])',
]

REGEXES = [re.compile(p) for p in REG_PATTERNS]


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

def split_top_level_commas(text: str) -> list[str]:
    parts = []
    cur = []
    depth = 0

    for ch in text:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1

        if ch == ',' and depth == 0:
            part = ''.join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)

    part = ''.join(cur).strip()
    if part:
        parts.append(part)

    return parts

def split_by_whitespace(text: str) -> list[str]:
    tokens = []
    cur = []
    depth = 0

    for ch in text:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1

        if ch.isspace() and depth == 0:
            tok = ''.join(cur).strip()
            if tok:
                tokens.append(tok)
            cur = []
        else:
            cur.append(ch)

    tok = ''.join(cur).strip()
    if tok:
        tokens.append(tok)

    return tokens

def split_operands(text: str) -> list[str]:
    if not text:
        return []

    operands = []
    for part in split_top_level_commas(text):
        operands.extend(split_by_whitespace(part))

    return operands

def extract_registers(op: str):
    regs = []
    for rx in REGEXES:
        for m in rx.findall(op):
            regs.append(parse_register(m))
    return regs

def is_modifier_token(tok: str) -> bool:
    # Things like op_sel_hi:[...], cbsz:1, blgp:1
    return ':' in tok and not any(c.isdigit() for c in tok.split(':', 1)[0])

def parse_instruction(line, loc, bb):
    line = line.strip()
    if not line or line.startswith(';'):
        return None

    # Remove trailing comments
    line = line.split(';', 1)[0].strip()

    parts = line.split(None, 1)
    opcode = parts[0]

    operand_text = parts[1] if len(parts) > 1 else ""
    raw_operands = split_operands(operand_text)

    operands = []
    regs_by_operand = []

    for tok in raw_operands:
        # Skip MFMA modifiers (op_sel, cbsz, blgp, etc.)
        #if is_modifier_token(tok):
        #    continue

        regs = extract_registers(tok)
        operands.append(tok)
        if regs:
            regs_by_operand.append(regs)

    return Instruction(
        opcode=opcode,
        operands=operands,
        regs_by_operand=regs_by_operand,
        loc=loc,
        raw_line=line,
        bb=bb,
    )

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


def print_check(check):
    return "✅" if check else "❌"


def analyze_lds_chains(bb, chains: list[DSReadChain], indent):

    chain_cnt = len(chains)
    dbg(f"========== analyzing {chain_cnt} ds read chains ========== ", indent)

    total_lds_regs = set()
    total_lds_regs_in_loop = set()
    entry_lds_data = set()
    for chain in chains:
        chain.update_regs()

        total_lds_regs |= flatten_regs(chain.regs)
        if chain.ds_groups[0].ds[0] in bb.instructions:
            total_lds_regs_in_loop |= flatten_regs(chain.regs)

        if chain.isLiveAcrossBB:
            entry_lds_data |= flatten_regs(chain.regs)

        cnt = len(flatten_regs(chain.regs))
        check_mark = print_check(chain.isLiveAcrossBB)
        dbg(f"loc-{chain.loc} ({cnt} regs, isLiveAcrossBB: {check_mark}): {chain.regs}", indent)
        dbg(f"This chain has {len(chain.ds_groups)} lds groups")

    total_lds_regs_in_loop = coalesce_regs(total_lds_regs_in_loop)
    a_num, v_num = count_regs(total_lds_regs_in_loop)
    dbg(f"total LDS regs in loop ({a_num} x a, {v_num} x v): {total_lds_regs_in_loop}", indent)

    entry_lds_data = coalesce_regs(entry_lds_data)
    lds_a, lds_v = count_regs(entry_lds_data)
    dbg(f"live-in lds data ({lds_a} x a, {lds_v} x v): {entry_lds_data}", indent)

    for inst in bb.instructions:
        if inst.is_mfma() or "ds_read" in inst.opcode or inst.mark_dead:
            continue
        if inst.uses & total_lds_regs:
            dbg(f"Found user {inst.emit()} of {coalesce_regs(inst.uses & total_lds_regs)}", indent)
        if inst.defs & total_lds_regs:
            dbg(f"Found defer {inst.emit()} of {coalesce_regs(inst.defs & total_lds_regs)}", indent)

    dbg(f"========== analyzing {chain_cnt} ds read chains done ===== ", indent)


def analyze_regs(bb, mfmaChainsInBB, indent):
    dbg(f"========== Analyze regs in block {bb.name} ==========", indent)

    live_in = bb.live_in
    entry_acc = set()

    for chain in mfmaChainsInBB:
        entry_acc |= flatten_regs(chain.entry_acc)

    assert entry_acc.issubset(live_in)

    live_in_a, live_in_v = count_regs(coalesce_regs(live_in))
    dbg(f"live in ({live_in_a} x a, {live_in_v} x v): {coalesce_regs(live_in)}", indent)
    acc_a, acc_v = count_regs(coalesce_regs(entry_acc))
    dbg(f"acc ({acc_a} x a, {acc_v} x v): {coalesce_regs(entry_acc)}", indent)

    bb_uses = bb.defs | bb.uses
    if live_in.issubset(bb_uses):
        dbg("bb_uses contains live_in", indent)
    bb_a, bb_v = count_regs(coalesce_regs(bb_uses))
    dbg(f"BB uses ({bb_a} x a, {bb_v} x v): {coalesce_regs(bb_uses)}", indent)

    live_through = live_in - bb_uses
    th_a, th_v = count_regs(coalesce_regs(live_through))
    dbg(f"live through ({th_a} x a, {th_v} x v): {coalesce_regs(live_through)}", indent)

    free_regs = bb.free_regs
    free_a, free_v = count_regs(coalesce_regs(free_regs))
    dbg(f"free regs ({free_a} x a, {free_v} x v): {coalesce_regs(free_regs)}", indent)

    read_from = coalesce_regs(bb.read_from)
    write_to = coalesce_regs(bb.write_to)
    dbg(f"{read_from=}  {write_to=}", indent)

    dbg(f"========== Analyze regs in block {bb.name} done =====", indent)


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


def fix_acc_users(bb, entry_acc):
    '''
    After rewrite dst and acc regs of all mfma insts and mark copy inst as dead,
    now we need to fix insts that define any reg in entry_acc to use a new free reg.
    We will ignore ds_read instructions since later we will rewrite their data regs
    to avoid using acc regs.
    '''
    logging.debug("========== fix acc users ==========")

    for inst in bb.instructions:
        if inst.mark_dead:
            continue
        if "ds_read" in inst.opcode:
            continue
        if inst.is_mfma():
            continue

        if inst.is_mfma_copy:
            continue

        #dbg(f"Checking {inst.emit()}")
        defines = inst.defs
        conflict_reg = defines & entry_acc
        if conflict_reg:
            logging.debug(f"Found an instruction that defines acc regs: {inst.emit()}")
            x = len(conflict_reg)
            kind = coalesce_regs(conflict_reg)[0].kind
            new_reg = pick_and_remove_contiguous_regs(bb.free_regs, x, kind)
            if new_reg is None:
                logging.debug("no more free regs")
                break
            new_reg = coalesce_regs(new_reg)[0]
            logging.debug(f"  new reg found: {new_reg}")
            dbg("users before replacement", 2)
            for user in inst.users:
                dbg(f"{user.emit()}", 4)
            repeat = 1 if 'ds_read' in inst.opcode else 10
            inst.replace_users_with(new_reg, repeat)
            dbg("users after replacement", 2)
            for user in inst.users:
                dbg(f"{user.emit()}", 4)
            inst.replace_reg(coalesce_regs(conflict_reg)[0], new_reg)
            logging.debug(f"  after replacement: {inst.emit()}")
        #if inst.uses & entry_acc:
        #    logging.debug(f"Someone was using acc reg: {inst.emit()}")

    logging.debug("========== fix acc users done =====")
    logging.debug("")


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


def optimize_buffer_load_m0(bb):
    logging.debug("========== optimize buffer load m0 ==========")

    i = 0
    end = len(bb.instructions)
    while i < end:
        inst = bb.instructions[i]
        if "buffer_load" in inst.opcode:
            idx = i
            ## pattern 1: s_mov_b32 m0 --> s_nop 0 --> buffer_load --> mfma
            ## pattern 2: s_mov_b32 m0 --> buffer_load --> mfma
            ## swap buffer_load and mfma
            mfma = bb.instructions[idx + 1]
            assert mfma.is_mfma()
            bb.instructions[idx], bb.instructions[idx + 1] = bb.instructions[idx + 1], bb.instructions[idx]
            ## remove s_nop
            if 'nop' in bb.instructions[idx - 1].opcode:
                bb.instructions[idx - 1].mark_dead = True

            i += 2
        else:
            i += 1

    logging.debug("========== optimize buffer load m0 done =====")


def reuse_regs(bb):
    '''
    For any inst
    opcode dst, src0, src1, ...
    We can save registers by replacing dst with srcx if
    1. dst and srcx use the same number of regs
    2. inst is the only user of srcx in bb
    '''
    logging.debug("========== try to reuse regs ==========")
    reg_uses = Counter()
    for inst in bb.instructions:
        if inst.mark_dead:
            continue
        for r in inst.uses:
            reg_uses[r] += 1

    cand_inst = defaultdict(Register)
    for inst in bb.instructions:
        if inst.mark_dead:
            continue
        if inst.is_mfma() or 'ds_read' in inst.opcode:
            continue
        if (not inst.defs) or (not inst.uses):
            continue
        if inst.defs & inst.uses:
            continue

        dst_reg = inst.get_dst_regs()
        if dst_reg.kind == 'm':
            continue
        for src_reg in inst.get_src_regs():
            if len(src_reg.ids) != len(dst_reg.ids):
                continue
            used_again = False
            for r in flatten_regs(src_reg):
                if reg_uses[r] != 1:
                    used_again = True
                    break
            if used_again:
                continue
            if dst_reg == src_reg:
                continue
            cand_inst[inst] = src_reg
            break

    for inst, src_reg in cand_inst.items():
        logging.debug(f"Rewrite dst reg of {inst.emit()} with {src_reg}")
        dst_reg = inst.get_dst_regs()
        for user in inst.users:
            if user.mark_dead:
                continue
            logging.debug(f"  rewriting user before: {user.emit()}")
            user.replace_reg(dst_reg, src_reg)
            logging.debug(f"  rewriting user after: {user.emit()}")
        inst.replace_dst(src_reg)

    logging.debug("========== try to reuse regs done =====")


def print_map(map, loc):
    regs = set()
    for reg in map[loc]:
        regs |= flatten_regs(reg)
    logging.debug(f"{loc}: {coalesce_regs(regs)}")


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

    print_map(lds_reg_assignment, 754)
    print_map(lds_reg_assignment, 755)
    print_map(lds_reg_assignment, 769)
    print_map(lds_reg_assignment, 783)
    print_map(lds_reg_assignment, 784)
    print_map(lds_reg_assignment, 803)
    print_map(lds_reg_assignment, 817)
    print_map(lds_reg_assignment, 818)
    print_map(lds_reg_assignment, 846)

    return lds_reg_assignment


def find_reuse(loc, loc_to_interval, loc_to_regs):

    found = None
    for this_loc, interval in sorted(loc_to_interval.items()):
        if this_loc == loc:
            return found
        this_start, this_end = interval
        my_start, my_end = loc_to_interval[loc]
        if my_start > this_end:
            this_regs = loc_to_regs[this_loc]
            my_regs = loc_to_regs[loc]
            if this_regs == my_regs:
                found = this_loc
                new_start = min(this_start, my_start)
                new_end = max(this_end, my_end)
                loc_to_interval[loc] = [new_start, new_end]
                loc_to_interval[this_loc] = [new_start, new_end]
                return found

    return found


def rewrite_lds_group(lds_group, free_reg, indent=0):
    dbg(f"  rewriting the following ds inst with {free_reg}", indent)
    my_regs = flatten_regs(free_reg)
    x = 4
    kind = free_reg.kind
    for ds_inst in lds_group.ds:
        dbg(f"{ds_inst.emit()}", indent + 2)
        my_reg = pick_and_remove_contiguous_regs(my_regs, x, kind)
        if not my_reg:
            dbg(f"Not enough regs", indent + 2)
            assert False
        ## replace ds_read
        ## Note that here we don't want to replace_all_users_with.
        ## We have a strong connection from the ds_read to its users with
        ## opIdx keeping tracking of which operand of mfma is using the result.
        #ds_inst.replace_users_with(free_reg)
        ds_inst.replace_dst(coalesce_regs(my_reg)[0])

    lds_group.data = free_reg
    opIdx = lds_group.opIdx
    for user in lds_group.users:
        user.replace_mfma_operand(free_reg, opIdx)


def rewrite_lds_data(bb, chains, entry_acc):
    logging.debug("========== rewrite lds data ==========")

    loc_to_interval = defaultdict(list)  # [1st_ds_read, index of last user] of all chains of this loc
    loc_to_regs = defaultdict(list)  # track the number of registers needed for each loc
    loc_to_chain = defaultdict(DSReadChain)  # map from loc to DSReadChain

    chains.sort(key=attrgetter("loc"))

    ds_total_regs = set()

    for chain in chains:
        last_user_idx = 0
        first_ds_idx = len(bb.instructions)
        loc = chain.loc
        num_regs = len(flatten_regs(chain.regs))
        loc_to_regs[loc] = num_regs
        loc_to_chain[loc] = chain
        if chain.within_bb(bb) or chain.isLiveAcrossBB:
            ds_total_regs |= flatten_regs(chain.regs)
        for group in chain.ds_groups:
            for user in group.users:
                if user in bb.instructions:
                    last_user_idx = max(last_user_idx, user.index)
            if group.ds[0] in bb.instructions:
                first_ds_idx = min(first_ds_idx, group.ds[0].index)
            else:
                first_ds_idx = 0
        loc_to_interval[loc] = [first_ds_idx, last_user_idx]

    for loc in loc_to_interval.keys():
        start = loc_to_interval[loc][0]
        end = loc_to_interval[loc][1]
        logging.debug(f"{loc=} ({loc_to_regs[loc]} regs): interval = [{start}, {end}]")

    ## Process each loc in order and replace the data reg if
    ## 1. loc is inside the loop, i.e. the chains at loc is not liveAcrossBB
    ## 2. See if the chains at this loc can reuse all the regs from a previous loc.
    ##    If so, just reuse them.
    ## 3. If cannot reuse and some chains uses acc regs, replace them with free regs of bb
    #for loc, chains_in_map in sorted(loc_to_chains.items()):
    ds_reg_used = set()
    ds_reg_free = (bb.free_regs | ds_total_regs - entry_acc).copy()

    for chain in chains:
        loc = chain.loc
        logging.debug(f"resource for chain at {loc=}")
        logging.debug(f"  ds reg used: {coalesce_regs(ds_reg_used)}")
        logging.debug(f"  ds reg free: {coalesce_regs(ds_reg_free)}")
        ## Skip the loc if it's liveAcrossBB
        if chain.isLiveAcrossBB:
            ds_reg_used |= flatten_regs(chain.regs)
            ds_reg_free -= flatten_regs(chain.regs)
            continue
        ## Skip the chain in the epilogue
        if not (chain.within_bb(bb) or chain.isLiveAcrossBB):
            continue
        ## Check if this loc should reuse a previous loc
        reuse_loc = find_reuse(loc, loc_to_interval, loc_to_regs)
        if reuse_loc is None:
            logging.debug(f"{loc=} need new reg")
            old_regs = flatten_regs(chain.regs)
            x = len(flatten_regs(chain.ds_groups[0].data))
            #cnt1 = 0
            for lds_group in chain.ds_groups:
                avai_regs = pick_and_remove_contiguous_regs(ds_reg_free, x, 'a')
                if avai_regs is None:
                    avai_regs = pick_and_remove_contiguous_regs(ds_reg_free, x, 'v')
                    if avai_regs is None:
                        logging.debug("  Not enough free registers")
                        assert False
                ds_reg_used |= avai_regs
                free_reg = coalesce_regs(avai_regs)[0]
                rewrite_lds_group(lds_group, free_reg)
        else:
            logging.debug(f"{loc=} can reuse {reuse_loc}")
            for idx, lds_group in enumerate(chain.ds_groups):
                target_reg = loc_to_chain[reuse_loc].ds_groups[idx].data
                rewrite_lds_group(lds_group, target_reg)
        chain.update_regs()
        #ds_reg_free = ds_reg_free | old_regs - entry_acc

    logging.debug("========== rewrite lds data done =====")


def optimize_nops(bb):
    logging.debug("========== optimize no-ops ==========")

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

    bb.cleanup_bb()

    logging.debug("========== optimize no-ops done =====")


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

    defs_in_loop_set = set()
    for r in defs_in_loop:
        defs_in_loop_set.add(r)

    logging.debug(f"defs_in_loop: {coalesce_regs(defs_in_loop_set)}")
    logging.debug(f"invariant regs: {coalesce_regs(invariant_regs)}")

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
            if inst.mark_dead:
                continue
            if 'cndmask' in inst.opcode:
                ## Here we simply assume cndmask is not loop invariant,
                ## neither is the dst reg
                invariant_regs -= flatten_regs(inst.get_dst_regs())
                continue

            if all(r in invariant_regs for r in inst.uses):
                if 'scratch_load' in inst.opcode:
                    read_from = flatten_regs(inst.get_copy_src())
                    if read_from & bb.write_to:
                        invariant_regs -= read_from
                    else:
                        invariant_insts.add(inst)
                        invariant_regs |= read_from
                        logging.debug(f"Adding {inst.emit()}, which defines {inst.get_dst_regs()}")
                    continue
                invariant_insts.add(inst)
                logging.debug(f"Adding {inst.emit()}, which uses {coalesce_regs(inst.uses)}")
                for r in inst.defs:
                    if r not in invariant_regs:
                        invariant_regs.add(r)
                        changed = True
            else:
                ## This instruction is not loop invariant, so
                ## what it defines needs to be removed from the
                ## invariant_reg set
                for r in inst.defs:
                    invariant_regs.discard(r)

    return invariant_insts, invariant_regs


def can_hoist(inst, bb, invariant_regs):
    # Must be pure
    if not inst.is_pure():
        return False

    if inst.mfma_chain is not None:
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
    for later in bb.instructions:
        if later == inst:
            continue
        if later.get_dst_regs():
            if flatten_regs(later.get_dst_regs()) & flatten_regs(reg):
                redef = True
                break
                #return False
    if redef:
        ## Try to use a different reg
        def_reg = coalesce_regs(inst.defs)[0]
        logging.debug(f"  reg redefined, trying to rewrite {def_reg.emit()}")
        kind, ids = def_reg.kind, def_reg.ids
        num = len(ids)
        free_reg = pick_and_remove_contiguous_regs(bb.free_regs, num, kind)
        if not free_reg:
            logging.debug("Not enough free regs")
            return False
        free_reg = coalesce_regs(free_reg)[0]
        logging.debug(f"  found free reg: {free_reg.emit()}")
        inst.replace_users_with(free_reg)
        inst.replace_reg(def_reg, free_reg)
        logging.debug(f"  new inst: {inst.emit()}")

    return True


def hoist_loop_invariants(bb: BasicBlock):
    invariant_insts, invariant_regs = find_loop_invariants(bb)

    hoistable = []
    for inst in invariant_insts:
        logging.debug(f"loop invariant: {inst.emit()}")
        if can_hoist(inst, bb, invariant_regs):
            hoistable.append(inst)
            if 'scratch_load' in inst.opcode:
                next_inst = bb.next_instruction(inst)
                if 'vmcnt(0)' in next_inst.operands[0]:
                    logging.debug(f"    Also hoist {next_inst.emit()}")
                    hoistable.append(next_inst)
            logging.debug("  can hoist!!")

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


def rewrite_next_free_vgpr(text: str, new_value: int = 512) -> str:
    lines = text.splitlines(keepends=True)
    out = []

    for line in lines:
        stripped = line.lstrip()
        leading_ws = line[:len(line) - len(stripped)]
        if stripped.startswith(".amdhsa_next_free_vgpr"):
            out.append(f"{leading_ws}.amdhsa_next_free_vgpr 512\n")
        elif stripped.startswith(".vgpr_count:"):
            out.append(f"{leading_ws}.vgpr_count: {new_value}\n")
        else:
            out.append(line)

    return "".join(out)


def optimize_copy(bb):
    '''
    x = ...
    U0: users of x
    y = x
    z = y
    U1: users of z

    can be optimized into

    w = ...
    U0: now they use w instead of x
    U1: now they use w instead of z
    '''

    logging.debug("========== optimize copy ==========")

    for inst in bb.instructions:
        if inst.mark_dead:
            continue
        if not inst.is_copy():
            continue

        ## Found y = x ==> inst
        # y = inst.get_copy_dst()
        x = inst.get_copy_src()

        users = inst.users
        if len(users) != 1:
            continue
        assert len(users) == 1
        user = next(iter(users))
        if not user.is_copy():
            continue

        x_def = bb.get_reaching_defs(inst, x)
        if len(x_def) != 1:
            logging.debug(f"Found {len(x_def)} reaching defs of {inst.emit()}")
        assert len(x_def) == 1
        x_def = next(iter(x_def))
        u0 = x_def.users
        ## z = y ==> user
        z = user.get_copy_dst()
        u1 = user.users

        ## remove y = x and z = y
        inst.mark_dead = True
        user.mark_dead = True

        ## Find a free reg for x
        kind = x.kind
        cnt = len(x.ids)
        free_reg = pick_and_remove_contiguous_regs(bb.free_regs, cnt, kind)
        if not free_reg:
            logging.debug(f"Not enough free regs of kind {kind}")
            return

        free_reg = coalesce_regs(free_reg)[0]

        logging.debug(f"x_def before: {x_def.emit()} defines {x}")
        x_def.replace_dst(free_reg)
        logging.debug(f"x_def after: {x_def.emit()} defines {x}")
        ## replace x with free_reg in u0
        logging.debug(f"replace {x} with {free_reg}")
        for u0_user in u0:
            logging.debug(f"  before: {u0_user.emit()}")
            u0_user.replace_reg(x, free_reg)
            logging.debug(f"  after: {u0_user.emit()}")
        ## replace z with free_reg in u1
        logging.debug(f"replace {z} with {free_reg}")
        for u1_user in u1:
            logging.debug(f"  before: {u1_user.emit()}")
            u1_user.replace_reg(z, free_reg)
            logging.debug(f"  after: {u1_user.emit()}")

    logging.debug("========== optimize copy ==========")


def licm(program):
    logging.debug("========== LICM ==========")
    loop = program.get_loop()
    hoisted, new_loop = hoist_loop_invariants(loop)
    loop.instructions = new_loop

    logging.debug("Hoisting the following before the loop:")
    prologue = program.get_prologue()
    for inst in hoisted:
        logging.debug(f"{inst.emit()}")
        prologue.add_inst(inst)
    logging.debug("========== LICM done =====")


def rotate_lgkmcnt(program):
    loop = program.get_loop()
    target = None
    for inst in loop.instructions:
        if 's_waitcnt' in inst.opcode and 'lgkmcnt' in inst.operands[0]:
            target = inst
            break

    if target is None:
        return
    ## 1. remove the target from the loop
    ## 2. add the target at the end of the prologue
    ## 3. add the target before the s_cbranch inst in the loop
    idx = len(loop.instructions) - 1
    loop.instructions.insert(idx, target)
    idx = target.index
    loop.instructions.pop(idx)
    program.get_prologue().add_inst(target)


def separate_waitcnt_and_barrier(loop):
    for inst in loop.instructions:
        if 's_barrier' in inst.opcode:
            ## pattern
            ## mfma --> waitcnt --> barrier
            ## change to
            ## waitcnt --> mfma --> barrier
            idx = loop.instructions.index(inst)
            if idx < 2:
                continue
            maybe_mfma = loop.instructions[idx - 2]
            if 'mfma' not in maybe_mfma.opcode:
                continue
            maybe_waitcnt = loop.instructions[idx - 1]
            if 'waitcnt' not in maybe_waitcnt.opcode:
                continue
            loop.swap_inst(idx - 2, idx - 1)


def schedule_window(window):
    mfmas = [i for i in window if i.is_mfma()]
    nonmfmas = [i for i in window if not i.is_mfma() and not i.is_control()]
    barriers = [i for i in window if i.is_control()]

    if len(nonmfmas) <= 5 and window[0].is_mfma():
        return window[:]  # unchanged

    logging.debug("Found window:")
    for inst in window:
        logging.debug(f"  {inst.emit()}")

    out = []
    mi = ni = 0

    while mi < len(mfmas) or ni < len(nonmfmas):
        if mi < len(mfmas):
            out.append(mfmas[mi])
            mi += 1
        for _ in range(5):
            if ni < len(nonmfmas):
                out.append(nonmfmas[ni])
                ni += 1

        if mi >= len(mfmas):
            out.extend(nonmfmas[ni:])
            break
        if ni >= len(nonmfmas):
            out.extend(mfmas[mi:])
            break

    # Always put one MFMA before barrier if possible
    if barriers:
        if out and not out[-1].is_mfma() and mfmas:
            out.insert(len(out), mfmas[-1])
        out.extend(barriers)

    return out


def optimize_mfma_density(block):
    i = 0
    n = len(block)
    out = []

    while i < n:
        # ---- CASE A: start of block ----
        if i == 0:
            j = i
            while j < n and not block[j].is_mfma():
                j += 1
            if j < n:  # found mfma
                k = j
                while k < n and block[k].is_mfma():
                    k += 1
                window_end = k - 1
                window = block[i:window_end + 1]
                rewritten = schedule_window(window)
                out.extend(rewritten)
                i = window_end + 1
                continue

        # ---- CASE B: non-mfma -> mfma transition ----
        if i > 0 and not block[i - 1].is_mfma() and block[i].is_mfma():
            start = i
            j = i
            while j < n and block[j].is_mfma():
                j += 1
            k = j
            while k < n and not block[k].is_mfma() and not block[k].is_control():
                k += 1
            window_end = k - 1
            window = block[start:window_end + 1]
            rewritten = schedule_window(window)
            out.extend(rewritten)
            i = window_end + 1
            continue

        # ---- Default: copy ----
        out.append(block[i])
        i += 1

    return out


def amdgcn_as(text, verbose=False):

    setup_logging(debug=verbose)

    program = parse_asm(text)
    indent = 0

    program.update_inst_index()
    program.process_blocks(indent)

    program.collect_ds_chains(indent)
    program.collect_mfma_chains(indent)

    loop = program.get_loop()

    mfmaChainsInLoop = program.mfmaChains
    LDSChains = program.LDSChains

    analyze_regs(loop, mfmaChainsInLoop, indent)
    analyze_lds_chains(loop, LDSChains, indent)

    dbg("######################################################", indent)
    dbg("## Step 1", indent)
    dbg("## 1. Rewrite dst and acc regs with entry_acc", indent)
    dbg("## 2. mark all copy inst as dead", indent)
    dbg("######################################################", indent)
    min_a = 64
    entry_acc = program.rewrite_mfma_acc(min_a, 0)

    program.update_free_regs(indent)
    analyze_regs(loop, mfmaChainsInLoop, indent)

    dbg("######################################################", indent)
    dbg("## Step 2", indent)
    dbg("## fix any inst that uses entry_acc", indent)
    dbg("## ignoring ds_read, which will be fixed later", indent)
    dbg("######################################################", indent)
    fix_acc_users(loop, entry_acc)

    program.update_free_regs(indent)
    analyze_regs(loop, mfmaChainsInLoop, indent)
    analyze_lds_chains(loop, LDSChains, indent)

    dbg("######################################################", indent)
    dbg("## Step 3", indent)
    dbg("## licm", indent)
    dbg("######################################################", indent)
    licm(program)

    program.update_free_regs(indent)
    analyze_regs(loop, mfmaChainsInLoop, indent)
    analyze_lds_chains(loop, LDSChains, indent)

    dbg("######################################################", indent)
    dbg("## Step 4", indent)
    dbg("## optimize copy chains", indent)
    dbg("######################################################", indent)
    optimize_copy(loop)

    program.update_free_regs(indent)
    analyze_regs(loop, mfmaChainsInLoop, indent)
    analyze_lds_chains(loop, LDSChains, indent)

    dbg("######################################################", indent)
    dbg("## Step 5", indent)
    dbg("## rewrite ds_read data regs", indent)
    dbg("######################################################", indent)
    rewrite_lds_data(loop, LDSChains, entry_acc)

    program.update_free_regs(indent)
    analyze_regs(loop, mfmaChainsInLoop, indent)
    analyze_lds_chains(loop, LDSChains, indent)

    dbg("######################################################", indent)
    dbg("## Step 6", indent)
    dbg("## optimize nops and insert mfma between m0 and buffer_load", indent)
    dbg("######################################################", indent)
    optimize_nops(loop)
    optimize_buffer_load_m0(loop)

    program.process_blocks(indent)
    program.update_free_regs(indent)
    analyze_regs(loop, mfmaChainsInLoop, indent)

    dbg("######################################################", indent)
    dbg("## Step 7", indent)
    dbg("## reuse regs when possible", indent)
    dbg("######################################################", indent)
    reuse_regs(loop)

    program.update_free_regs(indent)
    analyze_regs(loop, mfmaChainsInLoop, indent)

    loop.cleanup_bb()

    dbg("######################################################", indent)
    dbg("## Step 8", indent)
    dbg("## loophole optimizations", indent)
    dbg("######################################################", indent)
    #rotate_lgkmcnt(program)
    #separate_waitcnt_and_barrier(loop)

    #loop.instructions = optimize_mfma_density(loop.instructions)

    program.process_blocks(indent)
    program.update_free_regs(indent)
    analyze_regs(loop, mfmaChainsInLoop, indent)

    loop.cleanup_bb()
    emitted_text = emit_program(program)
    emitted_text = remove_debug_info_section(emitted_text)
    emitted_text = rewrite_next_free_vgpr(emitted_text, 512)
    setup_logging(debug=False)
    return emitted_text


def main():
    parser = argparse.ArgumentParser(description="AMDGPU assembly optimizer")
    parser.add_argument("input", help="Input AMDGCN assembly file")
    parser.add_argument("output", help="Output AMDGCN assembly file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")

    with open(args.input, "r") as f:
        text = f.read()

    emitted_text = amdgcn_as(text)

    with open(args.output, "w") as f:
        f.write(emitted_text)

    logging.debug("Emitted program written to %s", args.output)


if __name__ == "__main__":
    main()
