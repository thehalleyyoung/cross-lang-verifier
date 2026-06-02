# Cross-language translation validation

This page positions the core of `cross-lang-verifier` in the **translation
validation** tradition and connects it, formally, to that literature. The
companion code is
[`src/ub_oracle/translation_validation.py`](../src/ub_oracle/translation_validation.py),
and the test suite proves the validator below produces **re-executable**
witnesses on real compiled programs.

## What translation validation is

*Translation validation* (Pnueli, Siegel & Singerman, *TACAS '98*) replaces the
unattainable goal of *verifying a translator once and for all* with the tractable
goal of *validating each translation instance*: given a source program `P_S` and
the producer's output `P_T`, a separate **validator** either certifies that `P_T`
refines `P_S` or returns a **counterexample**. Later strands — Necula's
credible-compilation / witness-producing validators (*PLDI '00*), Rinard &
Marinov's correctness checking, and the SMT-backed validators used for LLVM
(Alive/Alive2) — share the shape: an *external*, *per-instance*, *witness-
producing* check that does not trust the producer.

Classically the source and target are the **same** IR at two pipeline stages
(e.g. LLVM-before vs LLVM-after). Our setting is the natural **cross-language**
generalization: `P_S` is a C unit and `P_T` is its translation into a *different*
language (`Rust`, `Go`, `Swift`) produced by a transpiler or a human/LLM port.

## Our validator and its validity relation

A transpiler or porter claims `P_T` is a faithful translation of `P_S`. We make
the **validity relation** precise (it is exactly the relational assertion `R_m`
of [`PRODUCT_PROGRAM.md`](PRODUCT_PROGRAM.md)):

> `P_T` is a *valid* translation of `P_S` **at input `i`** iff `R_m(src(i),
> tgt(i))` holds — i.e. the pair does **not** diverge modulo source
> undefinedness at `i`.

The validator `V(P_S, P_T, I, T)` probes a set of inputs `I` and returns:

* **`REFUTED(w)`** — a **counterexample witness** `w` at some `i ∈ I`: a fully
  self-contained record `(c_src, target_src, input i, target T, product
  observable, reason)` on which `R_m` is violated; or
* **`NOT_REFUTED`** — no probed input violated `R_m`. This is a **one-sided**
  result: like every translation validator over an input subset, it certifies
  *the probed inputs*, never global equivalence. We never report "equivalent".

This one-sidedness is the honest reading of translation validation over a
bounded input set, and it matches the tool's global soundness direction (sound
*for divergence*; see [`SEMANTICS.md`](SEMANTICS.md)).

## The witness is the contribution

What makes a translation validator credible is that its counterexample is an
*independently checkable artifact*, not an internal solver state. Our witness is
**re-executable**: it carries the two source texts and the concrete input, and
its `replay()` recompiles `P_S` (under `-O0`, `-O2` and UBSan) and `P_T` **from
scratch** and re-derives the product observable. The validator's **soundness
theorem** is therefore *operational and checkable*:

> **Theorem (witness soundness).** If `V` returns `REFUTED(w)`, then replaying
> `w` against fresh compilations reproduces a violation of `R_m` — i.e. `w`
> witnesses a genuine divergence, reproducible by any third party with the same
> toolchain.

> **Theorem (witness determinism).** Replaying the same witness twice yields the
> identical product observable (the witness is a stable, diffable artifact).

Both are discharged on real code by `confirm_translation_validation`: over real
divergent corpus items the validator must `REFUTE` and the witness must replay to
the same violation; over equivalent items it must return `NOT_REFUTED`; and a
re-replay must reproduce the byte-identical observable.

## Why the framing earns its keep

* It gives the tool a **recognized theoretical home** — reviewers can place it
  immediately, and the soundness obligation is the standard one (a witness that
  refutes the producer's claim), discharged operationally rather than assumed.
* It is **producer-agnostic**: c2rust output, a human port, or an LLM
  translation are all just `P_T` to validate — the validator and witness are
  unchanged.
* It is **target-parameterized**: a new target only supplies a
  `target_semantics.TargetPack`; the validity relation `R_m`, the witness format,
  and the replay procedure are unchanged.

## References (for the paper's related-work)

* A. Pnueli, M. Siegel, E. Singerman. *Translation Validation*. TACAS 1998.
* G. Necula. *Translation Validation for an Optimizing Compiler*. PLDI 2000.
* N. P. Lopes, J. Lee, C.-K. Hur, Z. Liu, J. Regehr. *Alive2: Bounded
  Translation Validation for LLVM*. PLDI 2021.
