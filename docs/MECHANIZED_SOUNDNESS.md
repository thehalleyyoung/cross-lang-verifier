# Mechanized Soundness (scoped)

The relational/product-program decision procedure at the heart of the tool
(see [`PRODUCT_PROGRAM.md`](PRODUCT_PROGRAM.md) and
`src/ub_oracle/product_program.py`) is not only tested against real compilers —
its **core soundness argument is machine-checked by the Lean 4 kernel**.

The development is `formal/ProductSoundness.lean`. It is self-contained (Lean 4
core only, no Mathlib), so

```
lean formal/ProductSoundness.lean      # exit 0  ==  kernel accepted
```

type-checks every theorem with the kernel. `src/ub_oracle/mechanized_soundness.py`
runs exactly this and reports whether the kernel accepted the proof.

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
| `strict_aliasing_oracle_sound` | a strict-aliasing report confirmed by optimizer exploitation is a genuine UB-rooted divergence. |
| `strict_aliasing_report_implies_type_pun` | the strict-aliasing report carries the generated incompatible-access/same-storage type-pun shape. |
| `strict_aliasing_report_implies_optimizer_exploited` | the report carries the exact `optimizer_exploited` confirmation signal: two clean C builds disagree while the target is defined and deterministic. |

The file also contains two fully-evaluated `example`s — the canonical
div-by-zero witness (C traps, Rust panics with `101`, behaviours differ ⇒
reported and certified UB-rooted) and a safe-input witness (agree ⇒ silent),
plus strict-aliasing positive/negative examples.

For strict aliasing, the mechanized class-specific lemma mirrors the real
`confirm_optimizer_exploited` harness rather than pretending a sanitizer exists:
if the same generated C type-pun source has two clean standard-conforming builds
with different outputs, and the target has one defined deterministic output, then
at least one C build must differ from that target output. That pigeonhole step is
what lets the generic product-program theorem certify the report as UB-rooted.

## Scope (honest)

This is a *scoped* mechanization: we formalize the decision procedure over the
recorded-observable abstraction, not a full denotational C semantics. Within
that abstraction the theorems are exactly the guarantees the Python oracle
relies on, and the soundness theorem is pack-independent — instantiating any
target pack (Rust shown; Go/Swift identical) cannot weaken it. The strict-aliasing
extension mechanizes the optimizer-exploitation evidence used by the implementation
and treats the generated `int*`/`long*` same-storage type pun as a construction
invariant, not as a full formal C11 aliasing semantics. Extending the abstraction
toward a full operational C semantics is future work (step 75's "even partial
mechanization is a strong differentiator").

## Confirmation hook

```python
from ub_oracle.mechanized_soundness import confirm_mechanized_soundness
rep = confirm_mechanized_soundness()
assert rep.ok
assert rep.fully_checked          # True when the Lean kernel actually ran
print(rep.theorems_present)       # all required theorems, including strict aliasing
```
