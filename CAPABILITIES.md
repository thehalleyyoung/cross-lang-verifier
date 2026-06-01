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
| Divergence classes (catalogue) | 11 classes enumerated in `src/ub_oracle/catalogue.py` | catalogue is data-complete; see oracle status below |
| Divergence **oracles** (executable) | **signed integer overflow** (`add`, `sub`; widths 32, 64); **shift-out-of-range**; **division/remainder by zero**; **`INT_MIN / -1`**; **out-of-bounds array access**; **strict-aliasing violation** | each finds a Z3 witness **and** confirms it against real clang+UBSan and rustc |
| Multi-oracle entry point | `verify_unit` runs every applicable oracle under a **sound-for-divergence** policy with **loud abstention** (`DIVERGENT` / `CANDIDATE` / `NO_DIVERGENCE_FOUND` / `UNKNOWN` / `NOT_COVERED`) | `src/ub_oracle/verify.py` |
| Per-class precision/recall | labelled benchmark; **P = R = 1.0** symbolically and after real-compiler confirmation | `src/ub_oracle/metrics.py` |
| Ground-truth confirmation | real `clang -O0 / -O2 / -fsanitize=undefined` + real `rustc -O`, in three modes (`exploited`, `trap_vs_defined`, `optimizer_exploited`) | `src/ub_oracle/reexec.py` |
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
