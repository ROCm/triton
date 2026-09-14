# gfx950 (CDNA4): don't put packed-FP32 VALU ops in an MFMA kernel

**Audience.** Agents writing GPU kernels for gfx950 / MI350X at a level where you control
the emitted instructions — MLIR, LLVM IR, FlyDSL, or hand-written AMDGCN. If you only use
Triton, this is already handled for you (see §8); read §2 and §6 anyway so you can
recognise the failure if it shows up somewhere else.

**The one rule.**

> On gfx950, a kernel that issues `v_mfma_*` must not also issue `v_pk_*_f32`
> (`v_pk_add_f32`, `v_pk_mul_f32`, `v_pk_fma_f32`). The combination silently produces
> wrong results, nondeterministically, at high occupancy.

You do not have to write the packed ops yourself for this to bite — LLVM forms them from
ordinary scalar FP32 code. **Check the disassembly, not your source** (§6).

---

## 1. Why you are reading this and not just fixing a normal bug

This defect has an unusually deceptive signature. It burned one full investigation cycle
already, which chased the wrong participant for days:

- It is **silent**. No fault, no NaN, no assert. Just wrong floating-point values.
- It is **nondeterministic**. Same binary, same inputs, same device: some runs correct,
  some wrong, different elements each time.
- It is **occupancy-gated**. Below ~full CU occupancy it does not reproduce *at all*. Your
  unit test at grid=64 will pass forever.
- It **looks like your kernel's fault**. The corrupted values are stale register contents,
  so they look like an indexing, masking or synchronisation bug in your code.
- It **moves when you touch anything**. Because the trigger depends on instruction
  scheduling, unrelated edits make it appear and disappear. This is what makes it so easy
  to "fix" by accident and conclude something false about the cause.

If you are staring at a gfx950 kernel with those five properties, stop debugging your
indexing and check §6 first.

---

## 2. The defect

gfx950 has an MFMA operand-reuse optimisation: when an MFMA's srcA/srcB is already resident
in the XDL buffer, the hardware skips re-reading it from the VGPRs. (It is the same
hardware capability that MI450 exposes explicitly as `.reuse` flags; on MI350 it is
implicit.)

The read-suppression mask is applied to the VALU register-read path using the wrong
signal. The consequence:

> An MFMA on **wave A** that skips its operand read causes a co-executing `v_pk_*` on
> **wave B** to skip *its* operand read. Wave B's packed op then computes on whatever was
> left in the register from before.

Two properties follow, and both matter for how you write kernels:

1. **The aggressor and the victim are on different waves.** It is not a data dependence
   you can see by reading one instruction stream. Wave B's packed op can be arbitrarily
   far from any MFMA *in its own code* and still be corrupted.
2. **The victim is the packed op, not the MFMA.** The MFMA's own result is fine. Inputs to
   the corrupted op are correct in memory and in the register file — the ticket dumped
   srcA/srcB/srcC as bit-identical across wavefronts while the result differed.

Tickets: `llvm/llvm-project#206825` (public mirror), ROCM-27743, DEGGIGX90-5078. Status as
of 2026-09-14: **unresolved, blocked on hardware**. No silicon or LLVM fix exists.

---

## 3. All three conditions must hold

| # | condition | notes |
|---|---|---|
| 1 | target is **gfx950 / CDNA4** | not gfx942, not gfx90a, not RDNA |
| 2 | `v_mfma_*` and `v_pk_*_f32` are **both resident on the same CU** | normally two waves of the same kernel; two concurrent kernels sharing a CU also qualifies |
| 3 | **high occupancy** | onset near full CU occupancy |

On condition 3, the measured threshold on a 256-CU MI350X: stable up to 1024 workgroups,
races reliably from 2048 (= 256 CU x 8 WG x 4 waves = 32 waves/CU). ROCM-27743 saw first
failures around grid=1200 with 20000 runs. **Treat "it didn't reproduce" as meaningless
unless you tested at full occupancy.**

Condition 3 is why lowering `waves_per_eu` appears to fix kernels. It does not fix
anything — it hides the bug by keeping you under the threshold, and it costs you occupancy.
Do not ship that as a fix.

---

## 4. Which instructions are the victim

**Confirmed victim class** — named in the tickets, and removing them empirically eliminated
the corruption:

```
v_pk_add_f32    v_pk_mul_f32    v_pk_fma_f32
```

These are the CDNA "double-rate FP32" ops. Each reads a **64-bit VGPR pair** (even/odd)
per operand. They come from the `packed-fp32-ops` subtarget feature (gfx90a and later).

**Suspect, not tested** — other VALU instructions that also read 64-bit VGPR pairs. The
erratum is in odd/even register-pair read selection, so these are plausibly in the same
class, but nothing in the tickets exercises them and nothing here tested them:

```
v_pk_mov_b32
v_add_f64, v_mul_f64, v_fma_f64, ... (FP64 VALU)
```

If your kernel is FP64-heavy and MFMA-heavy on gfx950 and you see the §1 signature, assume
you may have found the same bug with a different victim, and say so rather than assuming
you are safe.

**Believed unaffected, by reasoning rather than measurement** — packed ops on *16-bit*
types:

```
v_pk_add_f16, v_pk_mul_f16, v_pk_fma_f16, v_pk_*_bf16, v_pk_*_i16
```

These pack two values into a **single 32-bit VGPR**, so there is no register pair and no
odd/even selection to get wrong. This is an inference from the mechanism, not a tested
result — the kernel that reproduced the bug contained no 16-bit packed ops, so it provides
no evidence either way. Do not rely on this for something safety-critical without testing.

---

## 5. Patterns to avoid, by level

The thing to understand before the specifics: **you rarely write `v_pk_*_f32` explicitly.**
The compiler creates it. Ordinary scalar FP32 arithmetic gets vectorised into `<2 x float>`
binops by LLVM's VectorCombine / SLP, and those select to packed ops. So the rule cannot be
enforced by reading your own source.

### 5.1 LLVM IR

Avoid, in any function that also contains `@llvm.amdgcn.mfma.*` / `@llvm.amdgcn.wmma.*`:

```llvm
%v = fadd <2 x float> %a, %b        ; -> v_pk_add_f32
%v = fmul <2 x float> %a, %b        ; -> v_pk_mul_f32
%v = fsub <2 x float> %a, %b
%v = call <2 x float> @llvm.fmuladd.v2f32(...)   ; -> v_pk_fma_f32
```

Write them as scalar `float` operations with `extractelement` / `insertelement` around
them, or just disable the feature (§7). Note that writing scalar IR is **not sufficient on
its own** — the optimiser can re-vectorise it. Scalarising must happen *after* the
vectorising passes, which is exactly why Triton's own `ScalarizePackedFOps` pass runs at
the end of the pipeline rather than in the frontend.

### 5.2 MLIR

Same thing one level up. Avoid, in kernels that lower to MFMA:

```mlir
arith.addf %a, %b : vector<2xf32>
arith.mulf %a, %b : vector<2xf32>
math.fma   %a, %b, %c : vector<2xf32>
```

and anything that produces them: `vector.fma` on `vector<2xf32>`, elementwise epilogues on
a 2-element-contiguous FP32 layout, `amdgpu.*` ops with `vector<2xf32>` operands. The FP32
accumulator epilogue after a `tt.dot`-equivalent is the classic source — that is precisely
where the reproducing kernel's packed ops came from.

Do **not** rely on keeping MLIR-level vectors at width 1: the LLVM backend re-forms them.

### 5.3 FlyDSL

FlyDSL drives the ROCDL pipeline with an options dict in
`flydsl/compiler/backends/rocm.py` (`_pipeline_parts`), which contains a `features` entry
that is empty by default:

```python
rocdl_opts = {
    "O": 2, "abi": 600, "chip": chip,
    ...
    "features": "",          # <- the hook; set to "-packed-fp32-ops"
    ...
}
```

That is the direct analogue of Triton's target-feature string and is where the workaround
belongs. **Untested here** — FlyDSL was not installed in this sandbox, so this is located
by reading the wheel, not verified by running it. Confirm with the §6 disassembly check
before trusting it.

There is also `flydsl.compiler.llvm_options.llvm_options({...})`, a scoped `cl::opt`
setter. That controls LLVM command-line options, *not* subtarget features — it is the wrong
mechanism for this, though it is useful for the scheduling experiments in §9.

### 5.4 Hand-written AMDGCN

If you are writing assembly you have direct control: just don't emit `v_pk_*_f32` in a
kernel with `v_mfma_*`, and use two `v_add_f32` / `v_mul_f32` instead. If you have inherited
asm you cannot restructure, use the `v_nop` mitigation in §7.

---

## 6. How to check a kernel — do this, don't reason about it

### 6.1 Static check (seconds, no GPU)

Disassemble and count. Any nonzero result in an MFMA kernel is a bug:

```bash
llvm-objdump -d --mcpu=gfx950 kernel.hsaco | grep -cE 'v_pk_(add|sub|mul|fma)_f32'
# or straight from the .s
grep -cE 'v_pk_(add|sub|mul|fma)_f32' kernel.s
```

Sanity-check that you are looking at an MFMA kernel at all:

```bash
grep -c 'v_mfma' kernel.s
```

Put this in CI for gfx950 kernels. It is cheap and it is the only check that does not
depend on getting the occupancy right.

### 6.2 Dynamic check (minutes, needs a gfx950)

Use **self-consistency, not a golden reference**. A kernel with no atomics, no split-k and
a fixed grid is bit-reproducible, so you need no reference implementation and no tolerance
threshold:

1. Build inputs once. Do not regenerate them between runs.
2. Launch the same kernel N times (N >= 32) on those byte-identical inputs.
3. Compare every result against run 0 with `==`, not `allclose`.
4. **Any** run that differs by any amount is the bug.

Critically, sweep occupancy — the whole point is that low grids prove nothing:

```
grid:  128, 256, 512, 1024, 2048, 3072, 4096, 6144     (on a 256-CU part)
```

A clean result at 128–1024 with failures at 2048+ is this defect's fingerprint. A working
implementation of exactly this is in `repro_gfx950_ds_read_tr_race.py` in this directory.

Measured example, one flash-attention backward kernel, 32 runs per row:

| workgroups | packed FP32 present | packed FP32 removed |
|---|---|---|
| 128 – 1024 | 0/31 differing | 0/31 |
| 2048 | **15/15 differing** | 0/31 |
| 3072 / 4096 / 6144 | **13–15/15 differing** | 0/31 |

---

## 7. If you cannot avoid packed FP32

In rough order of preference.

**1. Drop the subtarget feature.** Cleanest, and it is what Triton now does. Removes the
opcode class entirely:

```bash
llc -mcpu=gfx950 -mattr=-packed-fp32-ops ...
```

In MLIR/FlyDSL, set the ROCDL target `features` string to `-packed-fp32-ops` (§5.3). In a
`#rocdl.target` attribute, the `features` field is the same knob.

Cost: you lose double-rate FP32 VALU throughput. On the kernel measured here that cost was
**not detectable** (0.0529 vs 0.0538 ms median) — packed ops adjacent to MFMA cannot
dual-issue with it anyway, so near MFMA the packed form buys little. Measure before
assuming it is expensive.

**2. Scalarise after vectorisation.** If you need packed FP32 elsewhere in the same module,
rewrite `<2 x float>` binops to scalar pairs in a late pass, after VectorCombine/SLP.
Triton's `ScalarizePackedFOps`
(`third_party/amd/lib/TritonAMDGPUToLLVM/ScalarizePackedFOps.cpp`) is a ~120-line worked
example you can copy. Note its limitation: it only rewrites basic blocks that *contain* an
MFMA, which leaves packed ops in MFMA-free blocks of the same kernel — and those waves
still co-execute with other waves' MFMAs (condition 2 in §3 is about the CU, not the basic
block). It fixed the measured kernel, but it is the weaker guarantee.

**3. `v_nop` before every `v_mfma`.** ROCM-27743's own validated mitigation: one **vector**
op immediately preceding the MFMA. Use only if you are emitting asm directly and cannot do
(1). It is fragile — any later scheduling pass can separate the pair.

**4. Reduce occupancy.** Works, is not a fix, costs performance. See §3.

---

## 8. If you are using Triton

Handled automatically as of this branch: `disable_packed_fp32_ops()` in
`third_party/amd/backend/compiler.py` appends `-packed-fp32-ops` for gfx950 kernels whose
LLVM IR contains an MFMA/WMMA intrinsic. Kernels with no MFMA keep double-rate packed math.

```bash
TRITON_HIP_DISABLE_PACKED_FP32_OPS=1   # force on everywhere (covers the §3 cross-kernel case)
TRITON_HIP_DISABLE_PACKED_FP32_OPS=0   # force off (A/B testing only)
```

Residual exposure, deliberately left open: a Triton kernel with no MFMA of its own can
still be victimised by a *concurrent* MFMA kernel sharing the CU. Set the knob to `1` if
you run mixed workloads and cannot tolerate that.

---

## 9. Traps — things that look like they work and don't

- **Do not chase the LDS read lowering.** The original investigation concluded the
  contiguous `ds_read_b64` / `ds_read_b64_tr_b16` lowering was the culprit, because
  switching to strided `ds_read2_b32` made the race disappear. It is a bystander. Removing
  the packed FP32 ops fixes the corruption with the LDS lowering left *completely
  untouched* (48 `ds_read_b64_tr`, 0 `ds_read2`, unchanged). Changing the LDS access
  pattern only reshuffles the schedule until the trigger stops firing; it costs
  vectorisation and buys a coin flip. PR #11169 is this workaround, and it is why gfx950
  low-vector swizzle reorder is gated to <=32 banks — treat that as historical, not as a
  model to copy.
- **`s_nop` does not work.** Confirmed in ROCM-27743 for `s_nop 0` through `s_nop 14`. The
  mitigation must be a **VALU** instruction.
- **Extra `s_waitcnt` does not work.** Neither does an immediate `lgkmcnt(0)`. This is not
  a memory-ordering problem and no barrier will fix it.
- **Changing the read mnemonic alone does not work.** ROCM-27743 tested substituting
  `ds_read2_b32` for `ds_read_b64` by editing the asm only: still races.
- **Do not propose the chicken bit.** `SQ_CONFIG1.DISABLE_SP_MFMA_SRCAB_VGPR_READ_SKIP`
  turns the read-skip off globally, per-XCD, via a debug register. It confirms attribution
  and is useful for nothing else — it disables a hardware power/performance feature and end
  users cannot be required to set it. It gets rediscovered in the Jira and re-proposed
  roughly once per investigation.
- **Do not trust a low observed failure rate.** It is a race. 2 of 10 runs came back
  completely clean on a configuration that fails ~90% of the time.
- **Do not trust accuracy-driven autotuning on affected kernels.** If your tuner measures
  correctness to pick a variant, this bug randomises that measurement: a good kernel gets
  recorded as inaccurate and a racy one as accurate, at random. Any tuning database
  populated on gfx950 while packed FP32 was live should be regenerated, not just reused.

---

## 10. Provenance — what is measured and what is inferred

Stated plainly so you know how much weight each claim carries.

**Measured on an MI350X (gfx950, 256 CU) in this sandbox:**

- Baseline reproduces at 15/15 runs at >=2048 workgroups, 0/15 below 1024.
- Removing `v_pk_*_f32` (via `-packed-fp32-ops`) gives 0/31 at every occupancy point,
  with the LDS lowering, MFMA count and `ds_read_b64_tr` count all unchanged.
- Enabling `ScalarizePackedFOps` instead also gives 0/31, leaving 5 `v_pk_mul_f32` in
  MFMA-free blocks.
- Perf cost on that kernel: none detectable, slightly favourable.
- Numerics unaffected: `v_pk_add_f32` and two `v_add_f32` are bit-identical per component.

**From the tickets, not re-verified here:** the RTL mechanism; the `v_nop` mitigation; the
`s_nop` negative result; the grid=1200 first-failure point; the chicken-bit result.

**Inferred, not tested:** that 16-bit packed ops are safe (§4); that FP64 VALU may be
exposed (§4); the FlyDSL hook location (§5.3).

**Honest caveat on the central claim:** dropping a subtarget feature changes instruction
selection *and* therefore scheduling, so "removing the victim" is not a perfectly isolated
single-variable experiment. Two things argue against it being just another reschedule: the
mechanism matches the RTL description exactly, and 18 of 32 MFMAs in the fixed binary are
still immediately preceded by a scalar op — "unprotected" in the `v_nop` sense — yet
nothing races in 31 runs at four occupancy points.

---

## 11. When to delete this document

When the hardware or LLVM fix lands and ROCM-27743 / DEGGIGX90-5078 close. At that point
`-packed-fp32-ops` should be reverted everywhere it was added — it is a real (if small)
performance give-up, and the whole point of keeping it a one-line target-feature change
rather than a codegen restructuring is that reverting is trivial. Grep for
`disable_packed_fp32_ops` and `packed-fp32-ops`.

Related, in this directory: `TRITON-HANDOFF-gfx950-ds_read_tr-race.md` — the full
investigation log, including the false trail and the evidence that settled it.
