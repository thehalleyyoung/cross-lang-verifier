# Mechanized Soundness (scoped)

The relational/product-program decision procedure at the heart of the tool
(see [`PRODUCT_PROGRAM.md`](PRODUCT_PROGRAM.md) and
`src/ub_oracle/product_program.py`) is not only tested against real compilers —
its **core soundness argument is machine-checked by the Lean 4 kernel** and its
central theorem is independently cross-checked in Coq.

The development is `formal/ProductSoundness.lean`. It is self-contained (Lean 4
core only, no Mathlib), so

```
lean formal/ProductSoundness.lean      # exit 0  ==  kernel accepted
```

type-checks every theorem with the kernel. `src/ub_oracle/mechanized_soundness.py`
runs exactly this and reports whether the kernel accepted the proof.

The independent cross-check is `formal/CoreSoundness.v`. It deliberately does
not import the Lean development; it re-proves the recorded-observable soundness,
relative completeness, pack-parametric Rust instance, and product-witness
preservation lemmas in Coq:

```
coqc formal/CoreSoundness.v        # exit 0  ==  Coq kernel accepted
```

When Coq is not installed the driver reports source-contract-only status: the
file and required theorem names are present, but `fully_checked` is false.

Step 129 adds an extracted checker artifact without moving the proof: Lake builds
`formal/VerifiedChecker.lean`, which imports `ProductSoundness`, calls the proven
`productViolated` definition directly, and has compile-time theorem guards tied to
`oracle_sound`.

Step 131 adds `formal/CompletenessBoundary.lean`, also built through Lake because
it imports `ProductSoundness`. It mechanizes the public boundary between classes
that are complete on their declared finite fragment and classes that remain
sound-but-may-abstain; it does not replace the executable brute-force completeness
evidence in `src/ub_oracle/completeness.py`.

```
make verified-check
cross-lang-verify --units examples/readme_demo_units.json --verified-check --fail-on unknown
```

The checker validates the **final source-UB positive-claim inference** from
trusted re-execution facts (`ubReached`, `tgtDefined`, `consequence`) to a
UB-rooted divergence. It does not replace compiler re-execution and it is not
applied to safe→safe defined-divergence claims; it prevents the last UB-rooted
verdict step from drifting away from the Lean theorem.

## What is proven

The proof works over the **recorded-observable abstraction** the oracle actually
decides over: for one concrete input, the triple

* `ubReached` — the C premise `P` (a UBSan trap / exploited miscompile was seen),
* `tgtDefined` — `T` (the target ran with a defined outcome),
* `consequence` — `C` (the observable behaviours differ),

with the relational assertion `R = ¬(P ∧ T ∧ C)` and `productViolated = ¬R`.

| Theorem | Statement |
| --- | --- |
| `oracle_sound` | a reported divergence is a genuine UB-rooted divergence (no false alarms). |
| `oracle_complete_rel` | every genuine UB-rooted divergence in the abstraction is reported. |
| `oracle_decides` | the oracle reports a divergence **iff** one exists (decision-procedure correctness). |
| `equivalence_never_reported` | observationally-equivalent pairs are never flagged. |
| `report_implies_ub` | every counterexample is rooted in source-level UB, not a target quirk. |
| `pack_oracle_sound` | soundness is **language-pair-parametric** (holds for any target-semantics pack). |
| `rust_oracle_sound` | the concrete **C → Rust** instantiation (`RustPack`, defined codes `{0, 101}`). |
| `product_program_preserves_divergence_witness` | the end-to-end product-program construction emits a counterexample only by copying the source/target/input payload unchanged and carrying the observation derived from raw run facts; that observation is a genuine UB-rooted divergence. |
| `product_program_emits_witness_iff_product_violated` | witness emission is exactly product-assertion violation. |
| `product_program_witness_iff_divergence` | witness emission is equivalent to a UB-rooted divergence in the recorded-observable abstraction. |
| `strict_aliasing_oracle_sound` | a strict-aliasing report confirmed by optimizer exploitation is a genuine UB-rooted divergence. |
| `strict_aliasing_report_implies_type_pun` | the strict-aliasing report carries the generated incompatible-access/same-storage type-pun shape. |
| `strict_aliasing_report_implies_optimizer_exploited` | the report carries the exact `optimizer_exploited` confirmation signal: two clean C builds disagree while the target is defined and deterministic. |
| `pointer_provenance_oracle_sound` | a pointer-provenance report confirmed by `trap_vs_defined` is a genuine UB-rooted divergence. |
| `pointer_provenance_report_implies_out_of_provenance` | the report carries the generated source shape: valid integer input, out-of-provenance pointer arithmetic. |
| `pointer_provenance_report_implies_checked_target` | the report carries the safe target shape: checked index, defined result, deterministic run. |
| `pointer_provenance_report_implies_trap_vs_defined` | the report carries the exact `trap_vs_defined` signal: source UBSan trap plus defined target outcome. |
| `mechanized_completeness_boundary` | every published completeness-boundary class is classified as exactly one of complete-on-declared-fragment or sound-but-may-abstain. |
| `completeness_boundary_total` / `completeness_boundary_disjoint` | the completeness-boundary partition is exhaustive and non-overlapping. |
| `complete_fragment_decides_recorded_observation` | in-fragment classes reuse the recorded-observable decision theorem (`oracle_decides`); concrete finite-range completeness is still checked by `completeness.py`. |

The file also contains fully-evaluated `example`s: the canonical div-by-zero
witness (C traps, Rust panics with `101`, behaviours differ ⇒ reported and
certified UB-rooted), a safe-input witness (agree ⇒ silent), strict-aliasing and
pointer-provenance positive/negative examples, and an end-to-end product-run
example where raw run facts emit a preserved counterexample.

For the end-to-end product-program theorem, the Lean model deliberately stores
only raw run facts in `ProductRun`: source/target/input payload, target semantics
pack, whether source UB was reached, the target return code, and whether
behaviours differ. The `Observation` is derived through `observe`, then copied
into the emitted `ProductCounterexample`. The preservation theorem proves that
any emitted counterexample carries the payload unchanged and the exact derived
observation, and that this observation satisfies `isUBDivergence`.

For strict aliasing, the mechanized class-specific lemma mirrors the real
`confirm_optimizer_exploited` harness rather than pretending a sanitizer exists:
if the same generated C type-pun source has two clean standard-conforming builds
with different outputs, and the target has one defined deterministic output, then
at least one C build must differ from that target output. That pigeonhole step is
what lets the generic product-program theorem certify the report as UB-rooted.

For pointer provenance, the mechanized class-specific lemma mirrors the real
`trap_vs_defined` harness: the generated C witness has valid data input but
forms an out-of-provenance pointer, UBSan traps on that concrete run, and the
Rust/Go-style target keeps a checked index and returns deterministically. In the
recorded-observable abstraction, the trap versus the defined target outcome is
the behavioural difference consumed by the product-program theorem.

## Scope (honest)

This is a *scoped* mechanization: we formalize the decision procedure over the
recorded-observable abstraction, not a full denotational C semantics. "End to
end" here means raw run facts → product construction → emitted counterexample;
the source and target payload strings are opaque provenance carried unchanged,
not parsed or given denotational semantics in Lean. Within that abstraction the
theorems are exactly the guarantees the Python oracle relies on, and the
soundness theorem is pack-independent — instantiating any target pack (Rust
shown; Go/Swift identical) cannot weaken it. The class-specific extensions
mechanize the evidence consumed by the implementation: strict aliasing treats
the generated `int*`/`long*` same-storage type pun as a construction invariant
and pointer provenance treats the generated out-of-provenance pointer arithmetic
plus UBSan trap as the source-side confirmation. They are not full formal C11
aliasing/provenance semantics. Extending the abstraction toward a full
operational C semantics is future work (step 75's "even partial mechanization is
a strong differentiator"). The completeness-boundary module is similarly scoped:
Lean proves the class partition and its connection to the recorded-observable
decision theorem, while the finite integer-fragment no-false-negative evidence is
the executable brute-force/Z3 agreement check documented in
[`COMPLETENESS.md`](COMPLETENESS.md).

## Confirmation hook

```python
from ub_oracle.mechanized_soundness import confirm_mechanized_soundness
rep = confirm_mechanized_soundness()
assert rep.ok
assert rep.fully_checked          # True when the Lean kernel actually ran
print(rep.theorems_present)       # all required theorems, including class extensions

from ub_oracle.mechanized_soundness import confirm_coq_crosscheck
coq = confirm_coq_crosscheck()
assert coq.ok
print(coq.fully_checked)          # True only when coqc accepted CoreSoundness.v

from ub_oracle.mechanized_soundness import confirm_mechanized_completeness_boundary
boundary = confirm_mechanized_completeness_boundary()
assert boundary.ok
print(boundary.fully_checked)     # True only when Lake/Lean accepted the module
```

```python
from ub_oracle.mechanized_soundness import build_verified_checker, run_verified_checker
build = build_verified_checker()
assert build.ok
assert run_verified_checker("divergent", True, True, True, build=False).ok
assert not run_verified_checker("divergent", False, True, True, build=False).accepted
```
