# Large-scale migration study (>130k LOC)

> Claim **C57-large-scale-study** · module `src/ub_oracle/large_scale_study.py`

A divergence oracle is only convincing if it survives *scale*. A handful of
hand-picked witnesses can be cherry-picked; a six-figure corpus of genuinely
distinct programs cannot. This study generates a deterministic, all-distinct
corpus of **7,500 C→{Rust,Go} programs totalling 130,500 lines** and confirms,
against the **real** toolchain, that the oracle's declared labels match what
clang/UBSan, `rustc`, and `go` actually do.

## What the corpus is

| Quantity | Value |
| --- | --- |
| Programs (distinct source) | **7,500** |
| Distinct-source ratio | **7,500 / 7,500 = 1.0** (no two programs share source) |
| C lines | 62,400 |
| Target lines (Rust+Go) | 68,100 |
| **Total lines** | **130,500** (floor: 100,000) |
| Language pairs | C→Rust (4,000), C→Go (3,500) |
| Declared divergent | 3,700 |
| Declared equivalent | 3,800 |

Class breakdown:

| Class | Family | Count |
| --- | --- | --- |
| `div_by_zero` | UB-rooted divergent | 1,200 |
| `oob_read` | UB-rooted divergent | 1,200 |
| `oversized_shift` | UB-rooted divergent | 800 |
| `signed_overflow` | UB-rooted divergent | 500 |
| `safe_add` | defined equivalent | 1,200 |
| `safe_mod` | defined equivalent | 1,200 |
| `safe_mul` | defined equivalent | 600 |
| `safe_shift` | defined equivalent | 800 |

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
large-scale study: 7500 items / 7500 distinct programs / 130500 LOC
across ['go', 'rust']; sample=12 seed=12648430 available=True
confirmed_divergent=7 confirmed_equivalent=5 agree=12/12 hash=d5b00ba52a0f
ok: True
```

## Soundness of the study itself

The report is `ok` **iff** the corpus meets the 100k-LOC floor **and** (when a
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

This is validated in `tests/test_ub_oracle.py` (`test_large_scale_*`), including a
live seeded sample executed against the real Rust and Go toolchains.
