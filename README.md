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
- benchmark assets and sample projects under `examples/`

## Best way to use this checkout

Treat it as a **research-grade verifier and exploration harness** for cross-
language equivalence. The implementation is substantial and the CLI surface is
real; the responsible story is about executable analysis workflows, not about
unrerun leaderboard-style claims.
