# Negative true-equivalence corpus

Step 164 adds an all-negative C->Rust/C->Go corpus whose job is to bound
spurious divergence reports, not to claim arbitrary-program equivalence.

`src/ub_oracle/negative_corpus.py` deterministically generates 1,000 distinct
ports across five safe integer families: bounded signed add/sub, non-zero
division/remainder, and in-range shifts.  Every item carries a verifier unit with
explicit operating ranges.  The focused gate proves:

- all 1,000 ports are distinct and declared `equivalent`;
- every item is covered by registered oracles, with zero `NOT_COVERED` cases;
- every applicable class is range-pruned before solver search;
- zero items produce a `DIVERGENT` or `CANDIDATE` flag;
- 3,000 bounded proof inputs are defined and observably equal; and
- a seeded sample is labeled `equivalent` by real `clang`/UBSan plus `rustc`/`go`.

Run it with:

```bash
make negative-corpus-check
```

The byte-stable manifest lives at
`experiments/negative_corpus/negative_corpus.json`; `check_results()` regenerates
it and fails if any source hash, range, coverage property, or false-positive
count drifts.
