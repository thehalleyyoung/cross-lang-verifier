# CAPABILITIES — what this oracle soundly decides (and what it does not)

_Per `100_STEPS.md` Step 4. This is an honest, calibrated statement of scope. It
is intended to be the paper's threats-to-validity section in miniature. When in
doubt, the tool **abstains loudly** rather than claiming equivalence._

## Soundness direction

The oracle is **sound for divergence**: when it reports a *confirmed* divergence,
that divergence is real — it is exhibited by a **fully-defined input** on which a
real source build and a real target build observably differ (or on which the
source executes undefined behavior that a real optimizer exploits, while the
target is defined). It is **not** a sound equivalence prover: "no divergence
found" means only that none was found within the current search bound and the
supported divergence classes.

## Current supported surface (this checkout)

| Axis | Supported now | Status |
|---|---|---|
| Language pair | **C → Rust** (the anchor) | flagship, validated end-to-end |
| Divergence classes (catalogue) | 12 classes enumerated in `src/ub_oracle/catalogue.py` | catalogue is data-complete; see oracle status below |
| Divergence **oracles** (executable) | **signed integer overflow** (`add`, `sub`; widths 32, 64); **shift-out-of-range**; **division/remainder by zero**; **`INT_MIN / -1`**; **out-of-bounds array access**; **strict-aliasing violation**; **floating-point contraction (FMA fusion)** | each finds a Z3 witness **and** confirms it against real clang+UBSan and rustc |
| Multi-oracle entry point | `verify_unit` runs every applicable oracle under a **sound-for-divergence** policy with **loud abstention** (`DIVERGENT` / `CANDIDATE` / `NO_DIVERGENCE_FOUND` / `UNKNOWN` / `NOT_COVERED`); oracles are **gated by declared language pair** so an unsupported pair is honestly `NOT_COVERED`, never silently treated as the anchor | `src/ub_oracle/verify.py` |
| Per-class precision/recall | labelled benchmark; **P = R = 1.0** symbolically and after real-compiler confirmation | `src/ub_oracle/metrics.py` |
| Ground-truth confirmation | real `clang -O0 / -O2 / -fsanitize=undefined` + real `rustc -O`, in three modes (`exploited`, `trap_vs_defined`, `optimizer_exploited`) | `src/ub_oracle/reexec.py` |
| Honest aggregate reporting | `aggregate_reports` emits **decided / abstained / unknown** fractions broken down by language pair and divergence class, with candidate-vs-not-covered sub-buckets and an explicit "not a proof of equivalence" disclaimer | `src/ub_oracle/report.py` |
| SARIF 2.1.0 output | `to_sarif` renders confirmed `DIVERGENT` findings at `error` level and unconfirmed `CANDIDATE` witnesses at `warning` level, with catalogue-derived rules and partial fingerprints; physical locations are emitted only when a unit declares one (never fabricated) | `src/ub_oracle/report.py` |
| Command-line verifier | `cross-lang-verify` (a.k.a. `python -m ub_oracle`): manifest-driven, pair-aware CLI with colored verdicts, the abstention summary, optional `--sarif`, and a `--fail-on` CI gate (exit 1 on confirmed divergence by default) | `src/ub_oracle/cli.py` |
| Differential-testing baseline | exact + seeded-empirical fuzzing-gap measurement | `src/ub_oracle/diff_testing.py` |

### What the executable oracles soundly establish

The **signed-overflow** oracle, for a unit `f(x) = x <op> c` over a C signed
integer type:

1. uses Z3 to find an input `x` for which `x <op> c` signed-overflows (UB), and
2. confirms, by compiling and running real binaries, that on that `x`:
   - the UBSan build **traps** (the UB is genuinely reachable),
   - the `-O0` and `-O2` builds **disagree** (the UB is *consequential*: the
     optimizer exploits it), and
   - the Rust translation produces a single **defined** value.

The **integer-model** oracles (shift-out-of-range, division/remainder by zero,
`INT_MIN / -1`) instead witness a **definedness divergence**: Z3 finds a defined
input on which

   - the UBSan build **traps** (C is undefined), while
   - the Rust translation is **defined and deterministic** — either a value or a
     clean, repeatable panic (confirmed by running Rust twice via the
     `trap_vs_defined` harness mode).

A divergence is reported as **confirmed** only when these conditions hold.

The **strict-aliasing** oracle uses a third mode, `optimizer_exploited`: no
sanitizer can trap a strict-aliasing violation, so the evidence is that the
**same C source produces different output at `-O0` vs `-O2 -fstrict-aliasing`** —
two builds of one deterministic program disagreeing proves the C result is
under-determined — while the Rust translation is a single defined value.

The **floating-point contraction** oracle (`fp_contraction`) reuses the
`optimizer_exploited` mode with a contraction-sensitive flag pair: C's `a*b + c`
may be fused into a single-rounding FMA at the implementation's discretion (C17
6.5p8 / `FP_CONTRACT`, *unspecified*), so `-ffp-contract=off` (two roundings)
and `-ffp-contract=fast` (one rounding) **disagree on the same source and input**,
while Rust's `a*b + c` always rounds twice (a single deterministic value). Z3's
floating-point theory finds a heavily-cancelling witness `(a,b,c)` where the
fused and unfused results actually differ in printed output.

## What it does NOT do yet (abstains / out of scope here)

- **Remaining divergence classes** (uninitialized reads, use-after-free,
  null-deref, eval-order, ...) are **catalogued but not yet executable oracles**.
  The framework (`src/ub_oracle/plugin.py`) is designed for them to be added as
  plugins; until then the tool **abstains loudly** (`NOT_COVERED`) rather than
  guessing.
- **Other language pairs** (C→Go, Python→Rust, ...) are part of the roadmap
  (Step 37) but not implemented here.
- **General whole-program equivalence**: no claim. The oracle reasons about the
  specific divergence classes it implements, on the units it is given.
- **Loops/recursion of unbounded depth, pointers/heap, concurrency**: not modeled
  by the current oracles.
- **Soundness for equivalence**: never claimed.

## Reproducibility

- `make reproduce` regenerates `experiments/ub_divergence/results.json`
  deterministically (no toolchain, seeded randomness).
- `make reproduce-check` asserts byte-identical regeneration.
- `make reproduce-confirm` additionally writes `confirmations.json` from real
  compiler runs (environment-dependent; not part of the byte-identical check).
- `make guard` enforces that no simulated numbers enter trusted artifacts.
