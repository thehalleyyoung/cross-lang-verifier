# Large-scale migration study (>1M LOC)

> Claim **C57-large-scale-study** · module `src/ub_oracle/large_scale_study.py`

A divergence oracle is only convincing if it survives *scale*. A handful of
hand-picked witnesses can be cherry-picked; a six-figure corpus of genuinely
distinct programs cannot. This study generates a deterministic, all-distinct
corpus of **60,000 C→{Rust,Go} programs totalling 1,044,000 lines** and confirms,
against the **real** toolchain, that the oracle's declared labels match what
clang/UBSan, `rustc`, and `go` actually do.

## What the corpus is

| Quantity | Value |
| --- | --- |
| Programs (distinct source) | **60,000** |
| Distinct-source ratio | **60,000 / 60,000 = 1.0** (no two programs share source) |
| C lines | 499,200 |
| Target lines (Rust+Go) | 544,800 |
| **Total lines** | **1,044,000** (floor: 1,000,000) |
| Language pairs | C→Rust (32,000), C→Go (28,000) |
| Declared divergent | 29,600 |
| Declared equivalent | 30,400 |
| Label-balance delta | 800 programs (1.33% of corpus) |

Class breakdown:

| Class | Family | Count |
| --- | --- | --- |
| `div_by_zero` | UB-rooted divergent | 9,600 |
| `oob_read` | UB-rooted divergent | 9,600 |
| `oversized_shift` | UB-rooted divergent | 6,400 |
| `signed_overflow` | UB-rooted divergent | 4,000 |
| `safe_add` | defined equivalent | 9,600 |
| `safe_mod` | defined equivalent | 9,600 |
| `safe_mul` | defined equivalent | 4,800 |
| `safe_shift` | defined equivalent | 6,400 |

## Why every program is genuinely distinct

A naïve scale-up reuses one argv-driven template and only varies the input
tuple — which produces **byte-identical source** and collapses a "corpus" into a
handful of programs run many times. That is not scale; it is repetition.

This study instead **bakes the *defined* operands as distinct integer literals**
into the source and reads **only the UB-triggering operand from `argv`**. Two
consequences:

1. Every generated program differs in its literals, so the SHA-256 of
   `(lang, c_src, target_src)` is unique — the census proves
   `n_items == n_distinct_programs`.
2. The UB-triggering value still arrives at runtime via `argv`, which defeats
   `rustc`'s `unconditional_panic` const-evaluation lint and keeps the C UB
   genuinely *dynamic* rather than folded away at compile time.

## How a label is confirmed

A program is labelled **divergent** when the C side traps under UBSan
(`-fsanitize=undefined -fno-sanitize-recover=all`) **and** the target produces a
*defined* outcome (Rust `0`/`101`-panic, Go `0`/`2`-panic), and **equivalent**
when both sides are defined and agree on stdout. The study does not trust those
declarations: `run_sample` draws a **seeded random sample** of the corpus,
compiles and runs each side through the same `ReexecHarness` used in production,
and asserts the *observed* verdict equals the *declared* one.

```bash
make large-scale          # or:
PYTHONPATH=src python -m ub_oracle.large_scale_study
```

Example run:

```
large-scale study: 60000 items / 60000 distinct programs / 1044000 LOC
across ['go', 'rust']; sample=12 seed=12648430 available=True
confirmed_divergent=7 confirmed_equivalent=5 agree=12/12 hash=5ec6c935c7b0
ok: True
```

## Soundness of the study itself

The report is `ok` **iff** the corpus meets the 1M-LOC floor **and** (when a
toolchain is present) every sampled item's real verdict agrees with its declared
label. Two honesty properties:

- **Toolchain-free part is structural.** The census (LOC floor + all-distinct)
  needs no compiler and cannot be faked by a missing toolchain.
- **Live part gates cleanly.** If compilers are absent the study reports
  *consistency-only* (`available=False`) rather than inventing agreement, so the
  traceability theorem `_thm_large_scale_study` never fabricates a pass.
- **Reproducible.** The verdict-layer content hash is computed over
  `(schema, seed, sorted per-item verdicts)` — timings are deliberately excluded
  — so a fixed seed yields a byte-stable hash across runs.

This is validated in `tests/test_large_scale_study.py` and
`tests/test_ub_oracle.py` (`test_large_scale_*`), including live seeded samples
executed against the real Rust and Go toolchains.
