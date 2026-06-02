# SemRec / XLEV

`cross-lang-verifier` ships two Python CLIs for **C↔Rust source-level
equivalence analysis**:

- `semrec` for bounded, focused equivalence checking and benchmark runs
- `xlev` for project discovery, verification, fuzzing, and analysis workflows

The strongest verified claim for this repository is that it contains a real
SMT-backed analysis stack for comparing C and Rust functions, with explicit
front ends for source strings, source files, benchmark suites, and project
discovery. Any broader benchmark numbers should be treated cautiously unless you
rerun them locally.

## Install

```bash
python3 -m pip install -e .
```

## CLI entry points

```bash
python3 -m src.semrec_cli --help
python3 -m src.cli.main --help
```

The module entry points above correspond to:

- `semrec`: `verify`, `cegar`, `bench`
- `xlev`: `verify`, `discover`, `fuzz`, `analyze`, `benchmark`

## Correct usage examples

```bash
# Inline C / Rust comparison
python3 -m src.semrec_cli verify \
  --c-code 'int f(int x){return x+1;}' \
  --rs-code 'pub fn f(x:i32)->i32{x.wrapping_add(1)}' \
  --format text

# File-based project-level verification
python3 -m src.cli.main verify \
  --c-source examples/sample_project/c_impl/math_utils.c \
  --rust-source examples/sample_project/src/lib.rs \
  --format text

# Auto-discover candidate pairs in a mixed project
python3 -m src.cli.main discover \
  --cargo-dir examples/sample_project \
  --c-dir examples/sample_project/c_impl
```

## What the repository demonstrably contains

- SMT-backed equivalence checking in `src/`
- source-string and source-file verification flows
- project discovery and benchmark runners in `src/cli/`
- a faithful counterexample minimizer (`src/ub_oracle/minimizer.py`) that shrinks
  each confirmed witness to a locally-minimal canonical form while preserving the
  exact UBSan diagnostic category — see the `make cex-quality` study
- a severity-ranked triage view and linter-style baseline/suppression files
  (`cross-lang-verify --triage` / `--write-baseline` / `--suppress`) so teams can
  adopt the checker on a large existing migration and fail CI only on *new*
  divergences
- an incremental cache (`--cache`) that re-verifies only changed units — keyed by
  unit content **and** the real toolchain version, so a compiler upgrade
  invalidates stale verdicts (~25× speedup at full reuse on the sample manifest)
- a self-contained, offline migration-risk HTML dashboard (`--dashboard`)
- a performance/scalability study (`make perf`) that times the *real* symbolic
  searches across every class and language pair and characterises the SMT
  backbone's growth out to 512-bit widths — the deterministic grid
  (`experiments/perf_curves/grid.json`) is checked for byte-identical
  regeneration in CI (`make perf-check`)
- a documented plugin **SDK** ([`docs/SDK.md`](docs/SDK.md)) with a worked
  *external* oracle ([`examples/plugins/float_cast_overflow_oracle.py`](examples/plugins/float_cast_overflow_oracle.py))
  proving a third party can add a brand-new divergence class — confirmed against
  real `clang`+`rustc` — without forking the engine
- an internal **red-team** (`make redteam`) that, for every oracle on every
  supported language pair, throws a battery of semantics-preserving adversarial
  mutations of a genuinely-divergent unit at the verifier and proves it never
  falsely returns "no divergence" — 63/63 cases confirmed divergent against real
  compilers, **zero soundness breaches** (the byte-reproducible adversarial grid
  is CI-checked via `make redteam-check`)
- a **branch-coverage ratchet** over the toolchain-independent brain of the tool
  (`make coverage` / `make coverage-check`): 20 curated core modules — the IR,
  oracle SPI, decision layer, symbolic searches, and finding pipeline — held at
  **91 %+ mean branch coverage** with a committed floor in `coverage_floor.json`
  that CI never lets regress
- an **interval-domain abstract-interpretation pre-pass** (`abstract_interp.py`)
  that, before any SMT call, *proves* a divergence class's undefined-behavior
  region unreachable under a unit's declared operating range and discharges it
  without invoking the solver — a sound accelerator (it only ever discharges
  no-divergence, never asserts one) that also lets a unit declare a safe range
  so it isn't flagged for a divergence that range forbids; the same range is
  honored by the oracles' searches so the fast path and the SMT path always agree
- a **frozen shared-IR contract** (`ir.py`, spec in `docs/IR.md`): the single
  language-pair-agnostic translation-unit shape every frontend lowers into and
  every oracle consumes, plus a validator that **rejects ill-formed lowerings**
  at the manifest boundary (the CLI exits non-zero on a malformed unit) while
  still accepting well-formed-but-uncovered units as `NOT_COVERED`
- benchmark assets and sample projects under `examples/`

## Best way to use this checkout

Treat it as a **research-grade verifier and exploration harness** for cross-
language equivalence. The implementation is substantial and the CLI surface is
real; the responsible story is about executable analysis workflows, not about
unrerun leaderboard-style claims.
