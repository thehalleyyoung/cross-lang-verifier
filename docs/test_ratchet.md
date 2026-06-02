# True-green test ratchet

A divergence oracle only earns trust if its **own test suite is honest**. The
ratchet (`scripts/test_ratchet.py`, decision core
`src/ub_oracle/test_ratchet_core.py`) gates that honesty and is enforced in CI
(`make green-check`).

## What it enforces

A run is accepted only when it is **true-green** *and* **non-regressed** against
the committed floor in `tests/green_baseline.json`:

| Condition | Why it fails the gate |
| --- | --- |
| `failed > 0` or `error > 0` | the suite is red |
| `xpassed > 0` | an `xfail` unexpectedly passed — a stale lie; fix the test or drop the marker |
| `passed < floor.passed` | a passing test was lost (deleted or `@skip`-ped to hide red) |
| `skipped > floor.skipped` | a test was *silently* skipped — invisible rot |

## Profiles

* **`--fast` (default, CI):** every test file **except** the heavyweight
  real-compiler driver `tests/test_ub_oracle.py`. Toolchain-independent,
  deterministic, runs in seconds. (The fast unit portion of `ub_oracle` is
  separately ratcheted for coverage by `scripts/coverage_gate.py`; the full
  real-compiler suite runs as the `test-ub` CI job.)
* **`--full`:** the entire suite including the live `clang`/`rustc`/`go`
  confirmation tests — for local end-to-end verification.

## Usage

```console
$ python scripts/test_ratchet.py            # enforce the fast floor
$ python scripts/test_ratchet.py --report   # current vs. baseline
$ python scripts/test_ratchet.py --update   # ratchet the floor up to a green run
```

`--update` refuses to record a non-green run as a baseline, so the floor only
ever moves toward *more* green.

## Soundness

The decision core (`violations` / `enforce_counts`) is pure and is
machine-checked by `traceability._thm_true_green_ratchet` (claim
`C56-true-green-ratchet`) on synthetic count vectors: it must accept a green,
non-regressed run and reject every red or regressed one.
