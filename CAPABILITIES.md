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
| Divergence **oracles** (executable) | **signed integer overflow** (`add`, `sub`; widths 32, 64) | finds a Z3 witness **and** confirms it against real clang+UBSan and rustc |
| Ground-truth confirmation | real `clang -O0 / -O2 / -fsanitize=undefined` + real `rustc -O` | `src/ub_oracle/reexec.py` |
| Differential-testing baseline | exact + seeded-empirical fuzzing-gap measurement | `src/ub_oracle/diff_testing.py` |

### What the signed-overflow oracle soundly establishes

For a unit `f(x) = x <op> c` over a C signed integer type, the oracle:

1. uses Z3 to find an input `x` for which `x <op> c` signed-overflows (UB), and
2. confirms, by compiling and running real binaries, that on that `x`:
   - the UBSan build **traps** (the UB is genuinely reachable),
   - the `-O0` and `-O2` builds **disagree** (the UB is *consequential*: the
     optimizer exploits it), and
   - the Rust translation produces a single **defined** value.

A divergence is reported as **confirmed** only when all three hold.

## What it does NOT do yet (abstains / out of scope here)

- **Other divergence classes** (shift-out-of-range, division-by-zero, OOB,
  aliasing, use-after-free, eval-order, ...) are **catalogued but not yet
  executable oracles**. The framework (`src/ub_oracle/plugin.py`) is designed for
  them to be added as plugins; until then the tool does not decide them.
- **Other language pairs** (C→Go, Python→Rust, ...) are part of the roadmap
  (Step 37) but not implemented here.
- **General whole-program equivalence**: no claim. The oracle reasons about the
  specific divergence classes it implements, on the units it is given.
- **Loops/recursion of unbounded depth, pointers/heap, concurrency**: not modeled
  by the signed-overflow oracle.
- **Soundness for equivalence**: never claimed.

## Reproducibility

- `make reproduce` regenerates `experiments/ub_divergence/results.json`
  deterministically (no toolchain, seeded randomness).
- `make reproduce-check` asserts byte-identical regeneration.
- `make reproduce-confirm` additionally writes `confirmations.json` from real
  compiler runs (environment-dependent; not part of the byte-identical check).
- `make guard` enforces that no simulated numbers enter trusted artifacts.
