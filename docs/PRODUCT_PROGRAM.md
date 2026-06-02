# Relational product-program construction and soundness

This page gives the *relational* account of what the oracle witnesses: a
**product program** `P_S × P_T` whose runs are pairs of a source run and a
target run on the **same** input, together with a **relational assertion** `R`
whose violation is *exactly* a cross-language divergence. The construction is
**parameterized over the source and target semantics** — C is the motivating
source and `{Rust, Go, Swift}` the motivating targets, but nothing below depends
on that choice beyond the two semantic packs it is instantiated with.

The same construction is given as executable code in
[`src/ub_oracle/product_program.py`](../src/ub_oracle/product_program.py), and
the test suite proves the inference rules below decide identically to the real
re-execution harness on real compiled programs.

## Setting

Fix a language pair `(S, T)`. A *semantics pack* for a language `L` supplies a
small-step relation `⟶_L` over machine configurations and a predicate
`defined_L(c)` marking configurations whose behaviour the language standard
fixes. For the source we additionally distinguish a predicate `ub_L(c)` marking
configurations that exhibit **undefined behaviour** (for `L = C`, witnessed
operationally by a `-fsanitize=undefined` trap). For the target, `det_T`
(determinism) is recorded by re-execution.

Write `P_S ⇓ o` for "source program `P_S` evaluates on the fixed input `i` to
observable `o`", where an observable is either a value `val(v)` or a trap
`trap`. We summarise the source side by the triple

```
src(i) = ( o0, o2, san )      o0, o2 ∈ {val(v), trap},  san ∈ {⊤, ⊥}
```

(`o0`, `o2` are the observables of the `-O0` and `-O2` builds; `san = ⊤` iff the
sanitizer build traps, i.e. `ub_C` is reached on `i`) and the target side by

```
tgt(i) = ( defined, det )      defined, det ∈ {⊤, ⊥}.
```

## The product transition system

The product program `P_S × P_T` is the synchronous composition that, on input
`i`, runs `P_S` and `P_T` independently to completion and pairs their
observables. Its transition relation is the standard relational product:

```
            c_S ⟶_S c_S'                         c_T ⟶_T c_T'
  ───────────────────────────── (Step-L)   ───────────────────────────── (Step-R)
   ⟨c_S, c_T⟩ ⟶× ⟨c_S', c_T⟩                ⟨c_S, c_T⟩ ⟶× ⟨c_S, c_T'⟩

                 c_S, c_T both terminal
  ──────────────────────────────────────────────────── (Join)
   ⟨c_S, c_T⟩ ⟶× ⟨done( src(i) ), done( tgt(i) )⟩
```

Because the two sides share no state, the order of `Step-L` / `Step-R` is
irrelevant; the product is confluent up to the final `done` pair, so the product
observable `(src(i), tgt(i))` is well defined.

## The relational assertion `R`

The product carries a single relational post-assertion `R`, parameterized by the
consequence **mode** `m ∈ {exploited, trap_vs_defined}`:

```
R_m( src(i), tgt(i) )  :≡  ¬( P(src) ∧ T(tgt) ∧ C_m(src, tgt) )
```

with the three clauses

* **(P) premise** — `P(src) :≡ (san = ⊤)`  — the source actually reaches UB on
  `i` (witnessed, not feared);
* **(T) target-defined** — `T(tgt) :≡ defined ∧ (det ∨ m = exploited)`;
* **(C) consequence**, by mode —
  * `C_exploited(src,·) :≡ o0 = val(v0) ∧ o2 = val(v2) ∧ v0 ≠ v2` (the optimizer
    turned the UB into an observable value change), or
  * `C_trap_vs_defined(src, tgt) :≡ (san = ⊤) ∧ defined ∧ det` (the consequence
    *is* the definedness gap: source undefined, target defined & deterministic).

The pair **diverges at `i`** iff `R_m` is **violated**:

```
Diverge_m(i)  :≡  ¬ R_m( src(i), tgt(i) )  =  P(src) ∧ T(tgt) ∧ C_m(src, tgt).
```

This is definitionally the same predicate as the operational divergence
definition in [`SEMANTICS.md`](SEMANTICS.md); the product framing only repackages
it as a relational assertion over the composed run, which is the object the
translation-validation literature reasons about.

## Soundness (and relative completeness)

> **Theorem (product soundness for divergence).** For every input `i` in the
> unit's declared domain and every mode `m`, if the product assertion `R_m` is
> violated on the product observable `(src(i), tgt(i))`, then `(P_S, P_T)`
> diverges at `i` under the operational definition of `SEMANTICS.md`.
>
> Conversely (relative completeness w.r.t. the recorded observables), if the
> pair diverges at `i` under that definition, the product assertion `R_m` is
> violated.

*Proof.* Immediate from the definitions: `Diverge_m(i)` is literally
`¬R_m(src(i), tgt(i))`, and both `¬R_m` and the operational predicate `judge(·)`
of `SEMANTICS.md` expand to the **same** conjunction `P ∧ T ∧ C_m` over the same
recorded observables. The two are therefore the same Boolean function of
`(o0, o2, san, defined, det, m)`. ∎

Soundness is *relative to* the recorded observables being faithful to the real
executions; that faithfulness is not assumed but **measured**: the executable
construction builds `src(i)`/`tgt(i)` from real `clang`/UBSan + `rustc`/`go`
runs, and `confirm_product_program` checks, on real divergent and equivalent
corpus items across packs, that

```
product_violated( i )  ==  is_divergence( obs(i) )  ==  harness.confirmed( i ).
```

So the relational product is provably the *same decision* as the operational
oracle on real code, parameterized over the target semantics pack.

## Why this framing earns its keep

* It connects the tool to **cross-language translation validation**: the oracle
  is a translation validator whose validity assertion is `R_m`, and a violation
  is a concrete counterexample to the translation's correctness *modulo source
  undefinedness*.
* The construction is **pack-parameterized**: instantiating a new target only
  supplies `defined_T` / `det_T` (a `target_semantics.TargetPack`); the product
  rules, the assertion `R_m`, and the soundness theorem are unchanged.
* It makes the **completeness fragment** explicit: within the recorded-observable
  abstraction the product is sound *and* complete, so every honest gap is
  attributable to the abstraction (which `COMPLETENESS.md` characterizes), not to
  the relational reasoning.
