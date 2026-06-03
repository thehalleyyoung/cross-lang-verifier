# Completeness characterization

This tool's headline guarantee is **soundness for divergence**: a `DIVERGENT`
verdict is only ever returned when a divergence has been *confirmed by
ground-truth re-execution* of real compiled programs. That guarantee is
necessarily one-sided — outside a covered class, or where a solver abstains, we
say nothing and never claim equivalence.

For the **integer divergence classes**, however, the symbolic search is not just
sound but **complete on a precisely-stated fragment**. This document states that
fragment per class and explains how the claim is *executed* (not merely
asserted) by `src/ub_oracle/completeness.py`.

## What "complete on a fragment" means here

For a divergence class `C`, an oracle is **complete on fragment `F`** when: for
every translation unit `u ∈ F`, the oracle reports a `DIVERGENT` witness **iff**
some input in `u`'s declared operating range triggers `C`'s C-undefined-behavior
condition. Equivalently, on `F` the oracle has **no false negatives** (and, being
sound, no false positives) — its verdict is exactly the ground truth.

We make this checkable by computing ground truth **by brute-force enumeration**:
for each unit we enumerate every input in its (finite) declared range and apply
the concrete C-UB predicate directly (no solver, no formula). The completeness
check then asserts the oracle's Z3 search agrees at every point, and that any
witness it returns is in range and genuinely triggers the UB. This two-sided
agreement, `∃-witness ⇔ oracle-finds-witness`, *is* the evidence.

## The characterized fragments

| class | fragment `F` | complete iff a witness is reported when… |
| --- | --- | --- |
| `signed_overflow` | `f(x) = x {+,-} c`, signed width ∈ {32, 64}, `x ∈ [lo, hi]` | some `x ∈ [lo,hi]` makes the exact-integer result leave `[INT_MIN, INT_MAX]` |
| `shift_oob` | `f(x,s) = x << s`, signed width ∈ {32, 64}, `s ∈ [lo, hi]` with `lo ≥ 0` | the interval contains an `s ≥ width` |
| `div_by_zero` | `f(a,b) = a {/,%} b`, signed width ∈ {32, 64}, `b ∈ [lo, hi]` | `0 ∈ [lo, hi]` |
| `intmin_div_neg1` | `f(a,b) = a {/,%} b`, signed width ∈ {32, 64}, `a ∈ […], b ∈ […]` | `INT_MIN ∈` a's interval **and** `-1 ∈` b's interval |

The same operating-range fields (`x_range` / `a_range` / `b_range` /
`shift_range`) feed both the abstract-interpretation pre-pass and the oracles'
SMT searches, so the fast path and the complete path are consistent (see
`docs/IR.md` and the pre-pass).

## Honest limits

* **Bounded ranges.** Completeness is asserted over the *finite* operating ranges
  of the fragment. The grid windows are positioned around the interesting
  boundaries (`INT_MAX`, `0`, `INT_MIN`, the width) so they exercise both sides
  of every boundary, but the claim is about those ranges, not all of `2^64`.
* **Non-negative shifts only.** Negative shift amounts are *also* C-UB but are
  **outside** the shift fragment: the witness search and re-execution model the
  unsigned shift-amount regime (the emitted target harness takes the amount as an
  unsigned value). This is stated in the fragment description, not hidden.
* **Out-of-fragment classes.** The memory-shape classes (`array_oob`,
  `strict_aliasing`) and the floating-point class (`fp_contraction`) are
  **explicitly not** claimed complete here — their searches remain sound but are
  not characterized to completeness. They are listed in
  `completeness.OUT_OF_FRAGMENT`.

## Reproducing the claim

```python
from src.ub_oracle import completeness as comp

for r in comp.check_all_completeness():          # C->Rust anchor
    assert r.complete, (r.divergence_class, r.mismatches, r.bad_witnesses)

for pair, results in comp.check_pair_completeness().items():  # every pair
    for r in results:
        assert r.complete
```

The test suite runs exactly this (`test_oracle_is_complete_on_its_characterized_fragment`,
`test_completeness_holds_across_every_registered_pair`) and, when a real
toolchain/runtime is present, additionally confirms one diverging unit per class
end-to-end against clang+UBSan and the target compiler/runtime
(`test_completeness_witnesses_confirm_against_real_compilers`). The shared
integer search transfers to every registered pair that implements these classes,
including C→Rust/Go/Swift/OCaml/Zig/WebAssembly.

## Per-pair soundness statements for the newer pairs

`src/ub_oracle/pair_soundness.py` makes the step-116–121 pair claims executable.
For C→Zig, C→C++20, Rust→C, C→WebAssembly, Go→Rust, and C→OCaml it checks that
the registered oracle exists, uses the expected confirmation mode, has a target
semantics pack where applicable, finds the positive witness, rejects the negative
control, and live-confirms the witness when the host has the real compiler or
runtime. `src/ub_oracle/generalization.py::run_generalization_v2()` then replays
those registered pairs as a cross-pair breadth study: every available positive
case must confirm and every safe control must remain unflagged.

```python
from src.ub_oracle import generalization, pair_soundness

g = generalization.confirm_generalization_v2()
assert g.ok or not g.available       # unavailable means no new-pair toolchains
assert pair_soundness.confirm_pair_soundness().ok
```
