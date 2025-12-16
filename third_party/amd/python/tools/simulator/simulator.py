#!/usr/bin/env python3
"""
Instruction Scheduler Simulator
Developed by: Lixun Zhang, Austin Kerbow
Copyright (c) 2025
This script simulates WMMA/VALU/EXP/DS scheduling behavior and annotates
per-instruction delays for analysis.

Usage:
    simulator.py input.s annotated_output.s
"""

import argparse
import re
import sys
from collections import defaultdict

# yapf: disable
# -------------------------------
# Delay reason table
# -------------------------------
DELAY_REASONS = {
    0: "tri-exec",      ## 1
    1: "le V law",      ## 2
    2: "data dep",      ## 4
    3: "le dscnt",      ## 8
    4: "trans op",      ## 16
}

# -------------------------------
# Instruction type mapping
# -------------------------------
INSTR_TYPES = {
    'v_wmma_scale_f32_16x16x128_f8f6f4': 'wmma_scale',
    'v_wmma_f32_16x16x32': 'wmma',
    'v_exp': 'exp',
    'v_cvt_scalef32_pk8': 'cvt_pk8',
    'ds_load': 'ds',
    'tensor_load': 'tdm',
    's_set_vgpr_msb': 'control',
    's_delay_alu': 'control',
    's_wait_alu': 'control',
    's_wait_dscnt': 'ds_wait',
    's_nop': 'internal',
    's_barrier_wait': 'internal',
    's_clause': 'internal',
    's_setprio': 'internal',
    's_wait': 'internal',
    'ld_scale': 'valu'
}

DS_LATENCY = 70  # cycles for ds_load to complete

INSTR_LATENCY = {
    'wmma': 8,
    'wmma_scale': 8,
    'exp': 8,
    'valu': 5,
    'ds': 70,
    'tdm': 1,
    'control': 1,
    'salu': 1,
    'internal': 1,
    'cvt_pk8': 8
}

INSTR_REPEAT = {
    'cvt_pk8': 4,
    'cvt_pk16': 8,
    'cvt_pk32': 16,
}

CONTROL_INSTRUCTIONS = {'s_set_vgpr_msb', 's_delay_alu', 's_wait_alu'}

# regex to capture comment-registers like /*v[10:25]*/ or /*v123*/ or /*s4*/
COMMENT_REG_RE = re.compile(
    r'/\*\s*(v\[[0-9]+:[0-9]+\]|v[0-9]+|s[0-9]+)\s*\*/',
    re.IGNORECASE
)

# regex for registers in the operand itself
REG_TOKEN_RE = re.compile(
    r'\b(v\[[0-9]+:[0-9]+\]|v[0-9]+|s[0-9]+)\b',
    re.IGNORECASE
)
# yapf: enable


# -------------------------------
# Parse instruction type
# -------------------------------
def get_instr_type(opcode):
    for key in INSTR_TYPES:
        if opcode.startswith(key):
            return INSTR_TYPES[key]
    if opcode.startswith('s_'):
        return 'salu'
    if opcode.startswith('v_'):
        return 'valu'
    return 'other'


# -------------------------------
# Instruction class
# -------------------------------
class Instruction:

    def __init__(self, line):
        self.line = line.rstrip('\n')
        self.line = self.line.lstrip()
        self.opcode = line.strip().split()[0]
        self.type = get_instr_type(self.opcode)
        self.issue_cycle = None
        self.co_issued = '-'
        self.wmma_coexec_cycle = '-'
        self.coexec_with_exp = 'n'
        self.delay_reason = set()
        self.deps = []  # registers read
        self.out_regs = []  # registers written
        self.parse_regs()

    def expand_register(self, reg):
        """
        Convert:
        - 'v[10:15]' → ['v10','v11','v12','v13','v14','v15']
        - 'v123'     → ['v123']
        - 's5'       → ['s5']
        """
        reg = reg.lower()
        if reg.startswith("v["):
            # vector register range: v[lo:hi]
            lo, hi = map(int, reg[2:-1].split(":"))
            return [f"v{i}" for i in range(lo, hi + 1)]
        elif reg.startswith("v"):
            return [reg]
        elif reg.startswith("s"):
            return [reg]
        return []

    def parse_regs(self):

        # reset
        self.out_regs = []
        self.deps = []

        line = self.line.strip()

        # If dual-instruction, split but DO NOT create multiple instructions
        sub_insts = [p.strip() for p in line.split("::")] if "::" in line else [line]

        # Helper regex utilities
        def expand_vrange(text):
            m = re.match(r'v\[(\d+):(\d+)\]', text)
            if not m:
                return None
            lo, hi = int(m.group(1)), int(m.group(2))
            return [f"v{i}" for i in range(lo, hi + 1)]

        def extract_comment_regs(text):
            regs = []

            # Range comment /*v[10:20]*/
            for lo, hi in re.findall(r'/\*v\[(\d+):(\d+)\]\*/', text):
                regs.extend([f"v{i}" for i in range(int(lo), int(hi) + 1)])

            # Single register comment /*v123*/
            for reg in re.findall(r'/\*v(\d+)\*/', text):
                regs.append(f"v{reg}")

            return regs

        def parse_one(text, out_regs, deps):
            """Parses register info for a single sub-instruction."""
            parts = text.split(None, 1)
            if len(parts) < 2:
                return

            operands_text = parts[1]
            raw_operands = [p.strip() for p in operands_text.split(",")]

            # ---------- Parse output operand (operand 0) ----------
            first = raw_operands[0]

            # Comment override for output
            comment_override = extract_comment_regs(first)
            if comment_override:
                out_regs.extend(comment_override)
            else:
                # Range in output
                vr = expand_vrange(first)
                if vr:
                    out_regs.extend(vr)
                else:
                    # Single vXXX
                    m = re.match(r'(v\d+)', first)
                    if m:
                        out_regs.append(m.group(1))

            # ---------- Parse input operands ----------
            for op in raw_operands[1:]:

                # Comment override first
                comment_regs = extract_comment_regs(op)
                if comment_regs:
                    deps.extend(comment_regs)
                    continue

                # Range v[xx:yy]
                vr = expand_vrange(op)
                if vr:
                    deps.extend(vr)
                    continue

                # Single vXXX or sXXX tokens
                tokens = op.replace("-", " ").split()
                for tok in tokens:
                    m = re.match(r'(v\d+)', tok)
                    if m:
                        deps.append(m.group(1))
                    m = re.match(r'(s\d+)', tok)
                    if m:
                        deps.append(m.group(1))

        # Parse each sub-instruction and merge results
        for sub in sub_insts:
            parse_one(sub, self.out_regs, self.deps)


# -------------------------------
# Simulator
# -------------------------------
class Simulator:

    def __init__(self, instructions):
        self.instructions = instructions
        self.register_ready = {}  # reg -> (ready_cycle, source_type)
        self.last_wmma = None
        self.last_exp_cycle = -10
        self.ds_load_queue = []  # list of (issue_cycle, ready_cycle) for in-flight ds_loads
        self.ds_wait_stalls = []  # list of (wait_cnt, stall_cycles, issue_cycle) for each s_wait_dscnt
        self.delay_cycles = {k: 0 for k in DELAY_REASONS}
        self.wasted_wmma_slots = 0
        self.current_wmma_used_slots = set()
        self.vgpr_spills = 0
        self.vgpr_reloads = 0

    def prev_instr(self, idx):
        i = idx - 1
        while (self.instructions[i].type == 'other' and i > 0):
            i -= 1
        return self.instructions[i]

    def prev_exec_instr(self, idx):
        i = idx - 1
        while i >= 0:
            if self.instructions[i].type not in ['other', 'control']:
                return self.instructions[i]
            i -= 1
        return None

    def extract_dwords(self, instr: str) -> int:
        # Look for pattern 'x<number>' at the end of the string
        m = re.search(r'x(\d+)$', instr)
        if m:
            return int(m.group(1))
        return 1  # default when no xN suffix

    def count_bank_conflicts_max(self, regs, num_banks):
        bank_counts = defaultdict(int)

        for r in regs:
            if r.startswith("s"):
                continue
            reg_id = int(r[1:])  # strip 'v'
            bank = reg_id % num_banks
            bank_counts[bank] += 1

        # max conflict among all banks
        max_conflict = 0
        for count in bank_counts.values():
            if count > 1:
                max_conflict = max(max_conflict, count - 1)

        return max_conflict

    def simulate(self):
        # yapf: disable
        wm_coexec_slots = {
            "wmma_scale": {
                1: ['ds_wait','control','internal','ds','tdm','salu'],
                2: ['ds_wait','control','internal','ds','tdm','salu'],
                3: ['ds_wait','control','internal','ds','tdm','salu','valu','exp'],
                4: ['ds_wait','control','internal','ds','tdm','salu'],
                5: ['ds_wait','control','internal','ds','tdm','salu'],
                6: ['ds_wait','control','internal','ds','tdm','salu','valu','exp'],
                7: ['ds_wait','control','internal','ds','tdm','salu','valu','exp']
            },
            "wmma": {
                1: ['ds_wait','control','internal','ds','tdm','salu'],
                2: ['ds_wait','control','internal','ds','tdm','salu','valu','exp'],
                3: ['ds_wait','control','internal','ds','tdm','salu','valu','exp'],
                4: ['ds_wait','control','internal','ds','tdm','salu'],
                5: ['ds_wait','control','internal','ds','tdm','salu'],
                6: ['ds_wait','control','internal','ds','tdm','salu','valu','exp'],
                7: ['ds_wait','control','internal','ds','tdm','salu','valu','exp']
            }
        }
        num_v_blocks = {
            "wmma_scale": 2,
            "wmma": 1
        }
        # yapf: enable
        wm_cycles = 8
        cycle = 0

        for idx, instr in enumerate(self.instructions):
            # -----------------------
            # Track VGPR spills/reloads (before type check)
            # -----------------------
            if 'scratch_store' in instr.opcode:
                self.vgpr_spills += self.extract_dwords(instr.opcode)
            elif 'scratch_load' in instr.opcode:
                self.vgpr_reloads += self.extract_dwords(instr.opcode)

            # ------------------------
            # skip non instructions
            # ------------------------
            if instr.type == 'other':
                continue

            # -----------------------
            # 1. Determine earliest cycle due to dependencies
            # -----------------------
            max_dep = 0
            caused_by_ds = False
            caused_by_wmma = False
            for dep in instr.deps:
                if dep in self.register_ready:
                    ready_cycle, source_type = self.register_ready[dep]
                    if ready_cycle > max_dep:
                        max_dep = ready_cycle
                        caused_by_ds = (source_type == 'ds')
                        caused_by_wmma = (source_type == 'wmma' or source_type == 'wmma_scale')
                    elif ready_cycle == max_dep and source_type == 'ds':
                        caused_by_ds = True

            if max_dep > cycle:
                reason = 3 if caused_by_ds else 2
                if not caused_by_wmma:
                    instr.delay_reason.add(reason)
                    self.delay_cycles[reason] += (max_dep - cycle)
            cycle = max(cycle, max_dep)

            # -----------------------
            # 2. EXP consecutive cycle check
            # -----------------------
            if instr.type == 'exp':
                if cycle <= self.last_exp_cycle:
                    stall = self.last_exp_cycle + 1 - cycle
                    cycle = self.last_exp_cycle + 1
                    instr.delay_reason.add(4)
                    self.delay_cycles[4] += stall
                self.last_exp_cycle = cycle + 1  # EXP takes 2 cycles

            # -----------------------
            # 3. WMMA co-execution
            # -----------------------
            if self.last_wmma and instr.type != 'wmma' and instr.type != 'wmma_scale':
                wm_issue = self.last_wmma.issue_cycle
                if cycle < wm_issue + wm_cycles:
                    slot_found = False
                    for slot in range(1, wm_cycles):
                        if instr.type in wm_coexec_slots[self.last_wmma.type][slot] and cycle <= wm_issue + slot:
                            ## Found a potential coExec slot at wm_issue + slot
                            ## Need to check tri-exec
                            prev = self.prev_instr(idx)
                            prev_slot = prev.wmma_coexec_cycle
                            if instr.type == 'valu' and prev.type == 'exp' and prev_slot == (slot - 1):
                                cycle += 1
                                instr.delay_reason.add(0)
                            else:
                                instr.wmma_coexec_cycle = slot
                                self.current_wmma_used_slots.add(slot)
                                cycle = wm_issue + slot
                                slot_found = True
                                break
                    if slot_found is False:
                        cycle = wm_issue + wm_cycles

            # ----------------------
            # 4. Re-visit exp again
            # ----------------------
            if instr.type == 'exp':
                self.last_exp_cycle = cycle + 1  # EXP takes 2 cycles
                ## If this exp is co-executing with wmma, it's not
                ## delayed by exp.
                ## Case 1:
                ##   cycle 3: exp
                ##   cycle 6: exp <-- this exp takes the next coExec slot of wmma, good
                ## Case 2:
                ##   cycle 6: exp
                ##   cycle 10: exp <-- this is delayed by exp + V rule.
                ##                     But it's no longer co-executing with wmma
                if instr.wmma_coexec_cycle != '-':
                    instr.delay_reason.discard(4)

            # -----------------------
            # 5. Control co-issue
            # -----------------------
            if instr.type == 'control' and idx > 0:
                prev = self.prev_instr(idx)
                if prev.type == 'other':
                    cycle = 0
                    instr.co_issued = 'n'
                elif prev.type != 'control' and prev.type != 'ds_wait' and prev.type != 'internal' and prev.type != 'ds':
                    instr.co_issued = 'y'
                    cycle = prev.issue_cycle
                    instr.wmma_coexec_cycle = prev.wmma_coexec_cycle
                else:
                    instr.co_issued = 'n'

            # -----------------------
            # 6. s_wait_dscnt handling
            # -----------------------
            if instr.type == 'ds_wait':
                # Parse the wait count from the instruction: s_wait_dscnt 0xN
                match = re.search(r's_wait_dscnt\s+0x([0-9a-fA-F]+)', instr.line)
                if match:
                    wait_cnt = int(match.group(1), 16)
                    # Remove completed ds_loads from queue
                    self.ds_load_queue = [(issue, ready) for issue, ready in self.ds_load_queue if ready > cycle]
                    # Wait until only wait_cnt loads remain in-flight
                    num_to_complete = len(self.ds_load_queue) - wait_cnt
                    stall_cycles = 0
                    if num_to_complete > 0:
                        # Sort by ready time and wait for the oldest ones to complete
                        sorted_queue = sorted(self.ds_load_queue, key=lambda x: x[1])
                        wait_until = sorted_queue[num_to_complete - 1][1]
                        if wait_until > cycle:
                            stall_cycles = wait_until - cycle
                            instr.delay_reason.add(3)
                            cycle = wait_until
                            self.delay_cycles[3] += stall_cycles
                        # Remove completed loads
                        self.ds_load_queue = [(issue, ready) for issue, ready in self.ds_load_queue if ready > cycle]
                    self.ds_wait_stalls.append((wait_cnt, stall_cycles, cycle))

            # ----------------------
            # 7. Co-execute with exp
            # ---------------------
            if cycle == self.last_exp_cycle:
                ## We already checked back to back exp and tri-exec of wmma+exp+valu
                ## Now we only need to rule out cvt_pk8
                ## wmma+exp+valu cannot co-exec together
                if instr.type == 'cvt_pk8':
                    cycle += 1
                    instr.delay_reason.add(4)
                else:
                    instr.coexec_with_exp = 'y'

            # -----------------------
            # 8. The V rule
            # -----------------------
            if (instr.type == 'valu' or instr.type == 'exp') and self.last_wmma:
                diff = cycle - self.last_wmma.issue_cycle
                v_blocks = num_v_blocks[self.last_wmma.type]
                if diff >= wm_cycles and diff < wm_cycles + v_blocks:
                    stall = (self.last_wmma.issue_cycle + wm_cycles + v_blocks) - cycle
                    cycle = self.last_wmma.issue_cycle + wm_cycles + v_blocks
                    instr.delay_reason.add(1)
                    self.delay_cycles[1] += stall
                    if instr.type == 'exp':
                        self.last_exp_cycle = cycle + 1

            # -----------------------
            # 9. Update last WMMA
            # -----------------------
            if instr.type == 'wmma' or instr.type == 'wmma_scale':
                instr.wmma_coexec_cycle = 0
                if self.last_wmma:
                    self.wasted_wmma_slots += (7 - len(self.current_wmma_used_slots))
                    self.current_wmma_used_slots = set()
                    cycle = max(cycle, self.last_wmma.issue_cycle + wm_cycles)
                self.last_wmma = instr

            # -----------------------
            # 10. Record issue cycle
            # -----------------------
            instr.issue_cycle = cycle

            # -----------------------
            # 11. Update register ready times
            # -----------------------
            reg_bank_conflicts = self.count_bank_conflicts_max(instr.deps, 8)
            for reg in instr.out_regs:
                latency = INSTR_LATENCY[instr.type]
                if instr.type != 'wmma' and instr.type != 'wmma_scale':
                    latency += reg_bank_conflicts
                self.register_ready[reg] = (instr.issue_cycle + latency, instr.type)

            # -----------------------
            # 12. Track ds_load in queue
            # -----------------------
            if instr.type == 'ds':
                ready_cycle = instr.issue_cycle + DS_LATENCY
                self.ds_load_queue.append((instr.issue_cycle, ready_cycle))

            # -----------------------
            # 13. Next cycle
            # -----------------------
            if 'cvt' in instr.type:
                cycle += INSTR_REPEAT[instr.type]
            else:
                cycle += 1

        if self.last_wmma:
            self.wasted_wmma_slots += (7 - len(self.current_wmma_used_slots))

    # -------------------------------
    # Output annotations
    # -------------------------------
    def output_annotations(self, out_file=None):
        # Get summary lines first
        summary_lines = self.analyze()

        out = open(out_file, 'w') if out_file else sys.stdout
        try:
            # Write summary at top as comments
            for line in summary_lines:
                out.write(f"; {line}\n")
            out.write(";\n")

            # Write annotated instructions
            for instr in self.instructions:
                if '//' in instr.line:
                    continue
                elif instr.type == 'other':
                    out.write(f"{instr.line}\n")
                else:
                    issue = f'{instr.issue_cycle:04d}'
                    co = instr.co_issued
                    wmma = f'{instr.wmma_coexec_cycle}' if instr.wmma_coexec_cycle != '-' else '-'
                    coexp = instr.coexec_with_exp
                    delay_val = sum(2**r for r in instr.delay_reason) if instr.delay_reason else 0
                    delay = f'{delay_val:2d}' if delay_val else '--'
                    out.write(f"{issue}:{co}:{wmma}:{coexp}:{delay}   {instr.line}\n")
        finally:
            if out_file:
                out.close()

    def num_control_coissued(self):
        cnt = 0
        for idx, instr in enumerate(self.instructions):
            if instr.type == 'control' and instr.co_issued == 'y':
                cnt += 1
        return cnt

    def get_wmma_type(self):
        wmma_type = 'other'
        for idx, instr in enumerate(self.instructions):
            if instr.type == 'wmma' or instr.type == 'wmma_scale':
                wmma_type = instr.type
                break
        return wmma_type

    def analyze(self):
        ## instr histogram
        INSTR_CNT = {
            'wmma': 0, 'exp': 0, 'valu': 0, 'ds': 0, 'ds_wait': 0, 'tdm': 0, 'control': 0, 'salu': 0, 'internal': 0,
            'other': 0, 'cvt_pk8': 0, 'wmma_scale': 0
        }
        COEXEC_WMMA = {
            'wmma': 0, 'exp': 0, 'valu': 0, 'ds': 0, 'ds_wait': 0, 'tdm': 0, 'control': 0, 'salu': 0, 'internal': 0,
            'other': 0, 'cvt_pk8': 0, 'wmma_scale': 0
        }
        COEXEC_EXP = {
            'wmma': 0, 'exp': 0, 'valu': 0, 'ds': 0, 'ds_wait': 0, 'tdm': 0, 'control': 0, 'salu': 0, 'internal': 0,
            'other': 0, 'cvt_pk8': 0, 'wmma_scale': 0
        }
        DELAY_CNT = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        vnop_cnt = 0

        wmma_type = self.get_wmma_type()
        if wmma_type == 'other':
            wmma_type = 'wmma'

        for idx, instr in enumerate(self.instructions):
            INSTR_CNT[get_instr_type(instr.opcode)] += 1
            if instr.wmma_coexec_cycle != '-':
                COEXEC_WMMA[get_instr_type(instr.opcode)] += 1
            if instr.coexec_with_exp == 'y':
                COEXEC_EXP[get_instr_type(instr.opcode)] += 1
            if instr.delay_reason:
                for reason in instr.delay_reason:
                    DELAY_CNT[reason] += 1
            if instr.opcode.startswith('v_nop'):
                vnop_cnt += 1

        ## Metric 1: is control always co-issued with its previous instruction?
        control_coissued = self.num_control_coissued()

        # Find the last instruction that has an issue_cycle
        total_cycle = 0
        for instr in reversed(self.instructions):
            if instr.issue_cycle is not None:
                total_cycle = instr.issue_cycle
                break

        total_ds_stall = sum(stall for _, stall, _ in self.ds_wait_stalls)

        # Build output lines
        lines = []
        lines.append("======================= Overall ==========================")
        lines.append(f"total cycle: {total_cycle}")
        if total_cycle > 0:
            lines.append(f"wmma eff (#wmma * 8 / total_cycle): {8 * INSTR_CNT[wmma_type] / total_cycle * 100:.1f}%")
        else:
            lines.append("wmma eff (#wmma * 8 / total_cycle): N/A")
        lines.append("======================= Co-Exec with wmma ================")
        lines.append(f"valu    ( / total #valu): {COEXEC_WMMA['valu']:3d} / {INSTR_CNT['valu']}")
        lines.append(f"cvt_pk8 ( / total #cvt):  {COEXEC_WMMA['cvt_pk8']:3d} / {INSTR_CNT['cvt_pk8']}")
        lines.append(f"exp     ( / total #exp):  {COEXEC_WMMA['exp']:3d} / {INSTR_CNT['exp']}")
        lines.append(f"salu    ( / total #salu): {COEXEC_WMMA['salu']:3d} / {INSTR_CNT['salu']}")
        lines.append(f"ds      ( / total #ds):   {COEXEC_WMMA['ds']:3d} / {INSTR_CNT['ds']}")
        lines.append("======================= Co-Exec with exp ================")
        lines.append(f"wmma    ( / total #wmma): {COEXEC_EXP[wmma_type]:3d} / {INSTR_CNT[wmma_type]}")
        lines.append(f"valu    ( / total #valu): {COEXEC_EXP['valu']:3d} / {INSTR_CNT['valu']}")
        lines.append(f"cvt_pk8 ( / total #cvt):  {COEXEC_EXP['cvt_pk8']:3d} / {INSTR_CNT['cvt_pk8']}")
        lines.append(f"salu    ( / total #salu): {COEXEC_EXP['salu']:3d} / {INSTR_CNT['salu']}")
        lines.append(f"ds      ( / total #ds):   {COEXEC_EXP['ds']:3d} / {INSTR_CNT['ds']}")
        lines.append("======================= Co-Issue =========================")
        lines.append(f"co-issued / total #control: {control_coissued} / {INSTR_CNT['control']}")
        lines.append("===================== s_wait_dscnt =======================")
        lines.append(f"#s_wait_dscnt: {INSTR_CNT['ds_wait']}")
        lines.append(f"total ds stall cycles: {total_ds_stall}")
        for wait_cnt, stall, issue_cycle in self.ds_wait_stalls:
            lines.append(f"  s_wait_dscnt 0x{wait_cnt:x} @ cycle {issue_cycle}: stalled {stall} cycles")
        lines.append("======================== Delays ==========================")
        lines.append(f"{'Reason':<14} {'Count':>6} {'Cycles':>8}")
        lines.append(f"{'-'*14} {'-'*6} {'-'*8}")
        lines.append(f"{'le V law':<14} {DELAY_CNT[1]:>6} {self.delay_cycles[1]:>8}")
        lines.append(f"{'data dep':<14} {DELAY_CNT[2]:>6} {self.delay_cycles[2]:>8}")
        lines.append(f"{'ds latency':<14} {DELAY_CNT[4]:>6} {self.delay_cycles[3]:>8}")
        lines.append(f"{'exp stall':<14} {DELAY_CNT[5]:>6} {self.delay_cycles[4]:>8}")
        lines.append(f"{'wasted wmma':<14} {'-':>6} {self.wasted_wmma_slots:>8}")
        lines.append(f"#v_nop: {vnop_cnt}")
        lines.append("======================= VGPR Spills =======================")
        lines.append(f"VGPR Spills:   {self.vgpr_spills} stores ({self.vgpr_spills * 4} bytes)")
        lines.append(f"VGPR Reloads:  {self.vgpr_reloads} loads ({self.vgpr_reloads * 4} bytes)")

        total_coexec_stalls = self.delay_cycles[1] + self.delay_cycles[4]
        total_ds_stalls = self.delay_cycles[3]
        total_data_dep_stalls = self.delay_cycles[2]
        total_stalls = sum(self.delay_cycles.values()) + self.wasted_wmma_slots

        lines.append(f"{'-'*30}")
        lines.append(f"Total Co-exec Stalls:    {total_coexec_stalls:>8} cycles")
        lines.append(f"Total DS Stalls:         {total_ds_stalls:>8} cycles")
        lines.append(f"Total Data Dep Stalls:   {total_data_dep_stalls:>8} cycles")
        lines.append(f"Total Wasted WMMA Slots: {self.wasted_wmma_slots:>8} cycles")
        lines.append(f"{'-'*30}")
        lines.append(f"Total Estimated Stalls:  {total_stalls:>8} cycles")
        lines.append("==========================================================")

        # Print to stdout
        for line in lines:
            print(line)

        return lines


# -------------------------------
# Main
# -------------------------------


def insert_ld_scale(instr_lines):
    """
    Insert a synthetic 'ld_scale' instruction before each v_wmma_scale.
    """
    new_lines = []

    for line in instr_lines:
        stripped = line.lstrip()

        if stripped.startswith('v_wmma_scale'):
            # synthetic instruction
            new_lines.append('ld_scale\n')

        new_lines.append(line)

    return new_lines


def main():
    parser = argparse.ArgumentParser(description="Simulate instruction cycles with annotations")
    parser.add_argument('input_file', help='Input assembly file path')
    parser.add_argument('output_file', nargs='?', default=None, help='Output annotated file path (stdout if omitted)')
    parser.add_argument('--loop-only', action='store_true', help='Only analyze the loop body (.LBB0_1 to s_cbranch)')
    args = parser.parse_args()

    # Read instructions
    with open(args.input_file, 'r') as f:
        instr_lines = f.readlines()

    # If --loop-only, extract just the loop body
    if args.loop_only:
        loop_start = None
        loop_end = None
        for i, line in enumerate(instr_lines):
            if '.LBB0_1:' in line:
                loop_start = i + 1  # Start after the label
            if loop_start is not None and 's_cbranch' in line and '.LBB0_1' in line:
                loop_end = i + 1  # Include the branch instruction
                break
        if loop_start is not None and loop_end is not None:
            print(f"Extracting loop body: lines {loop_start+1} to {loop_end}")
            instr_lines = instr_lines[loop_start:loop_end]
        else:
            print("Warning: Could not find loop markers (.LBB0_1), processing entire file")

    instr_lines = insert_ld_scale(instr_lines)
    instructions = [Instruction(line) for line in instr_lines if line.strip()]

    # Simulate
    sim = Simulator(instructions)
    sim.simulate()
    sim.output_annotations(args.output_file)

    if args.output_file:
        print(f"Annotated file saved to {args.output_file}")


if __name__ == "__main__":
    main()
