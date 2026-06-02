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
| Language pair | **C → Rust** (the anchor), **C → Go**, and **C → Swift** | all three validated end-to-end against real compilers; targets are added as **data-only semantic packs** (`src/ub_oracle/target_semantics.py`), not new harness/oracle code |
| Divergence classes (catalogue) | 12 classes enumerated in `src/ub_oracle/catalogue.py` | catalogue is data-complete; see oracle status below |
| Divergence **oracles** (executable) | **signed integer overflow** (`add`, `sub`; widths 32, 64); **shift-out-of-range**; **division/remainder by zero**; **`INT_MIN / -1`**; **out-of-bounds array access**; **strict-aliasing violation**; **floating-point contraction (FMA fusion)**; the five argv-driven integer/memory classes are also implemented for the **C → Go** and **C → Swift** pairs | each finds a Z3 witness **and** confirms it against real clang+UBSan and the target compiler (rustc / go / swiftc) |
| Multi-oracle entry point | `verify_unit` runs every applicable oracle under a **sound-for-divergence** policy with **loud abstention** (`DIVERGENT` / `CANDIDATE` / `NO_DIVERGENCE_FOUND` / `UNKNOWN` / `NOT_COVERED`); oracles are **gated by declared language pair** so an unsupported pair is honestly `NOT_COVERED`, never silently treated as the anchor | `src/ub_oracle/verify.py` |
| Per-class precision/recall | labelled benchmark; **P = R = 1.0** symbolically and after real-compiler confirmation | `src/ub_oracle/metrics.py` |
| Ground-truth confirmation | real `clang -O0 / -O2 / -fsanitize=undefined` + the target compiler (`rustc -O`, `go build`, or `swiftc -O`), in three modes (`exploited`, `trap_vs_defined`, `optimizer_exploited`); each target language carries its own definedness predicate as a data pack (Rust rc∈{0,101}, Go rc∈{0,2}, Swift rc∈{0,-5}) | `src/ub_oracle/reexec.py`, `src/ub_oracle/target_semantics.py` |
| Cross-pair regression matrix | a deterministic, byte-reproducible artifact (and `make matrix` / `matrix-check` targets, wired into CI) running **every** oracle across **every** `(source, target)` pair — the living evidence of generality; an optional `--confirm` pass re-runs each cell against the real toolchains (17/17 cells confirmed locally against clang+UBSan + rustc + go + swiftc) | `src/ub_oracle/regression_matrix.py`, `experiments/cross_pair_matrix/` |
| Honest aggregate reporting | `aggregate_reports` emits **decided / abstained / unknown** fractions broken down by language pair and divergence class, with candidate-vs-not-covered sub-buckets and an explicit "not a proof of equivalence" disclaimer | `src/ub_oracle/report.py` |
| SARIF 2.1.0 output | `to_sarif` renders confirmed `DIVERGENT` findings at `error` level and unconfirmed `CANDIDATE` witnesses at `warning` level, with catalogue-derived rules and partial fingerprints; physical locations are emitted only when a unit declares one (never fabricated) | `src/ub_oracle/report.py` |
| Command-line verifier | `cross-lang-verify` (a.k.a. `python -m ub_oracle`): manifest-driven, pair-aware CLI with colored verdicts, the abstention summary, optional `--sarif`, and a `--fail-on` CI gate (exit 1 on confirmed divergence by default) | `src/ub_oracle/cli.py` |
| Triage UX | `--triage` ranks every unit into severity-ordered priority tiers — **confirmed-divergence** (critical → moderate → minor, by catalogue severity) **>** symbolic **candidate** **>** **unknown** (solver abstained) **>** **not-covered** **>** **no-divergence** (informational) — so the highest-consequence, most-certain findings float to the top of a big migration; the ordering is deterministic and screenshot-stable | `src/ub_oracle/triage.py` |
| Config + suppression / baseline files | `--write-baseline` captures every current finding (pinned by its stable SARIF fingerprint) into a linter-style JSON baseline; `--suppress` then keeps those known-accepted divergences **visible but non-blocking**, so a team can adopt the checker on a large existing migration and have CI fail only on *new* divergences. Rules match by class / pair / unit-glob / fingerprint, require a `reason`, support `expires:` dates, and the CLI loudly warns about expired, unused, or overly-broad (constraint-free) rules | `src/ub_oracle/suppress.py` |
| Counterexample minimization | a **faithful** witness minimizer (`minimize_counterexample`) shrinks each confirmed witness to a locally-minimal canonical form by greedy 1-minimization, accepting a reduction only when real clang+UBSan + the target compiler still confirm it **and** the simplified witness triggers the *same* UBSan diagnostic category — so `INT_MIN/-1` never collapses into a plain division-by-zero and a too-large shift never becomes a negative shift; the `cex-quality` study (deterministic byte-reproducible `baseline.json`, gitignored real-compiler `minimized.json`, wired into CI) minimizes every anchor witness (e.g. signed overflow `1073741824 → 1`) with certified local minimality | `src/ub_oracle/minimizer.py`, `experiments/cex_quality/` |
| Differential-testing baseline | exact + seeded-empirical fuzzing-gap measurement | `src/ub_oracle/diff_testing.py` |
| Head-to-head vs differential testing | equal-budget fuzzing **+ UBSan** on the *same* real binaries vs. the oracle; tabulates the **false-negative gap** (sparse-UB divergences the oracle confirms but an equal-budget fuzzer misses) while showing dense-UB parity (the fuzzer finds out-of-range shifts immediately) | `src/ub_oracle/headtohead.py` |
| Ablation study | disabling each class-oracle in turn over the labelled set; reports the per-class recall drop, the positives uniquely missed, and a **zero cross-class leak** check (each oracle covers only its own class) | `src/ub_oracle/ablation.py` |
| Adoption: GitHub Action | "Translation Equivalence Guard" composite action + example workflow: runs the CLI on a manifest, emits SARIF, and uploads to code scanning | `.github/actions/translation-equivalence-guard/action.yml` |

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
- **Further language pairs** (Python→Rust, C++→Rust, ...) are part of the
  roadmap; **C → Go** is implemented and validated end-to-end (Step 37) as the
  generality proof that a new target only needs target-source emission plus a
  definedness predicate — the witness search and verifier are reused unchanged.
- **General whole-program equivalence**: no claim. The oracle reasons about the
  specific divergence classes it implements, on the units it is given.
- **Loops/recursion of unbounded depth, pointers/heap, concurrency**: not modeled
  by the current oracles.
- **Soundness for equivalence**: never claimed.

## Installation

The flagship verifier is a real, pip-installable distribution
(`cross-lang-verifier`) whose only runtime dependency is `z3-solver`; the
ground-truth confirmation step additionally discovers real compilers
(`clang`/`rustc`/`go`/`swiftc`) on `PATH` at run time.

```sh
pip install .                 # from a checkout (PEP 621 / pyproject.toml)
# or, from a built wheel:
pip wheel . --no-deps -w dist && pip install dist/cross_lang_verifier-*.whl
```

This installs the self-contained `ub_oracle` top-level package and two console
scripts — `cross-lang-verify` and the alias `cross-lang-verifier` — both bound
to `ub_oracle.cli:main`:

```sh
cross-lang-verify --units examples/units_manifest.json            # ground-truth
cross-lang-verify --units examples/units_manifest.json --no-confirm  # symbolic
```

`make package-check` is the reproducible proof: it builds the wheel, installs it
into a **fresh** virtualenv (with `PYTHONPATH` unset and run from a neutral
directory so nothing leaks in from the source tree), imports the installed
top-level package, and runs the console script end-to-end — confirming real
divergences against actual compilers and asserting the CI gate exit code.

## Reproducibility

- `make reproduce` regenerates `experiments/ub_divergence/results.json`
  deterministically (no toolchain, seeded randomness).
- `make reproduce-check` asserts byte-identical regeneration.
- `make reproduce-confirm` additionally writes `confirmations.json` from real
  compiler runs (environment-dependent; not part of the byte-identical check).
- `make guard` enforces that no simulated numbers enter trusted artifacts.
