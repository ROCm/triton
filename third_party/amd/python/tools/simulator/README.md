# User guide of the Simulator

## Usage

Extract a basic block from the generated ISA of the kernel, and save it into `bb.s`.
Run the simulator and provide an output filename as:
```bash
simulator.py bb.s annot_bb.s
```

In the output file, `annot_bb.s`, each instruction is annotated with a compact
string representing cycle and delay related information.
It also contains various metrics of the given basic block.

## **✔ Instruction Annotation**

Fixed-width annotation prefix: IIII:C:W:E:D   <instruction>
  * IIII = 4-digit issue cycle (zero-padded)
  * C    = y/n/- (co-issued with previous)
  * W    = wmma co-exec slot (0..7 or '-')
  * E    = y/n (co-executed with exp)
  * D    = delay id (integer as string, '-' if none)

### Delay ids

Each instruction may be delayed for one or more reasons.
The simulator encodes all delay causes into a single integer using a bitmask.

| Bit | Value | Meaning    |
|-----|-------|------------|
| 0   | `1`   | `tri-exec` |
| 1   | `2`   | `le V law` |
| 2   | `4`   | `data dep` |
| 3   | `8`   | `ld_scale` |
| 4   | `16`  | `le dscnt` |
| 5   | `32`  | `trans op` |

- Each delay reason corresponds to a bit position.
- When an instruction is delayed for multiple reasons, the simulator sets multiple bits.
- The final delay value is the sum of all contributing bit values.

### Delay reasons

- "tri-exec": These 3 instructions cannot co-execute at the same cycle.
  E.g. we cannot have the following execution flow
  ```asm
  0: wmma
  6: exp
  7: valu
  ```
  Cycle 7 is co-executing wmma+exp+valu, which is not allowed by the hw.
  Therefore, the valu has to be delayed by 1 cycle.
- "le V law". The cycle 8 and 9 of a wmma cannot issue valu.
  This is not documented but observed by Niels.
  In the above example, valu needs to be delayed from cycle 7. But it cannot be
  issued at cycle 8 or 9 due to the V law.
  Then it is issued at cycle 10.
  Note that in this case, the valu is delayed by both "wmma+exp+valu" and "le V law".
  Therefore, its delay value is 3.
- "data dep": RAW data dependency.
- "ld scale": For scaled wmma instruction, it breaks into `ld_scale` and `wmma`
  at hw execution.
  The `ld_scale` acts like a valu except that it is not affected by "le V law".
  Therefore, we cannot have the following flow
  ```asm
  0: wmma
  7: valu
  8: wmma_scaled
  ```
  The `wmma_scaled` instruction need to do `ld_scale` first, which happens at cycle 8.
  Then the `wmma_scaled` is issued at cycle 9.
  It's better to schedule a non-valu instruction between 2 consecutive wmma instructions.
- "le dscnt": delay due to LDS data latency.
  Note that the data latency is set to 70, which is just an estimate.
  In the optimal schedule, we should have enough instructions to hide the LDS latency.
  The model of ds latency is just a reference.
- "trans op": trans ops, such as exp, log, takes 2 cycles to finish. The 2nd cycle can
  co-exec a valu, salu, mem, ds, but cannot co-exec wmma or trans.

## **✔ Dependency Tracking**

* Identifies all vector/scalar read and write dependencies.
* Correctly handles cases like:

  * Instructions using their own output as an input
  * Comment-mapped register renumbering
  * Dual-instruction merged dependencies

## **✔ Cycle Simulation**

* Computes issue cycle for each instruction based on:

  * Register hazards (RAW, WAW, WAR)
  * Functional unit availability (WMMA, VALU, LDS, SALU, EXP)
  * Custom FU slot rules (e.g., LDS taking WMMA slot 4)
* Tracks completion cycles and pipeline latency.

## **✔ Metrics**

Automatically collects:

* Instruction counts by category
* Total cycles
* Efficiency metrics (e.g., `wmma eff`)

An example of metric output is as follows
```bash
; ======================= Overall ==========================
; total cycle: 1633
; wmma eff (#wmma * 8 / total_cycle): 31.4%
; ======================= Co-Exec with wmma ================
; valu ( / total #valu):  66 / 641
; exp  ( / total #exp):   77 / 260
; salu ( / total #salu):  49 / 76
; ds   ( / total #ds):    12 / 96
; ======================= Co-Exec with exp ================
; wmma ( / total #wmma):   0 / 64
; valu ( / total #valu): 139 / 641
; salu ( / total #salu):  16 / 76
; ds   ( / total #ds):    40 / 96
; ======================= Co-Issue =========================
; co-issued / total #control: 721 / 734
; ===================== s_wait_dscnt =======================
; #s_wait_dscnt: 9
; total ds stall cycles: 122
;   s_wait_dscnt 0x0 @ cycle 178: stalled 0 cycles
;   s_wait_dscnt 0x13 @ cycle 516: stalled 20 cycles
;   s_wait_dscnt 0xb @ cycle 532: stalled 7 cycles
;   s_wait_dscnt 0x2 @ cycle 565: stalled 0 cycles
;   s_wait_dscnt 0x0 @ cycle 578: stalled 0 cycles
;   s_wait_dscnt 0x15 @ cycle 897: stalled 20 cycles
;   s_wait_dscnt 0xe @ cycle 916: stalled 0 cycles
;   s_wait_dscnt 0x0 @ cycle 939: stalled 13 cycles
;   s_wait_dscnt 0x0 @ cycle 1268: stalled 62 cycles
; ======================== Delays ==========================
; Reason          Count   Cycles
; -------------- ------ --------
; le V law           46       91
; data dep           12       23
; ld_scale            6        6
; ds latency          5      122
; exp stall          25       25
; wasted wmma         -      208
; #v_nop: 43
; ======================= VGPR Spills =======================
; VGPR Spills:   0 stores (0 bytes)
; VGPR Reloads:  0 loads (0 bytes)
; ------------------------------
; Total Co-exec Stalls:         122 cycles
; Total DS Stalls:              122 cycles
; Total Data Dep Stalls:         23 cycles
; Total Wasted WMMA Slots:      208 cycles
; ------------------------------
; Total Estimated Stalls:       475 cycles
; ==========================================================
```
