# Divergence semantics

This page states, in prose, the property the tool *witnesses*. The same
definition is given as executable code in `src/ub_oracle/semantics.py`, and the
test suite proves the two are the same decision on real compiled programs.

## The object of study

Fix a language pair `(S, T)` — here `S = C` and `T ∈ {Rust, Go, Swift}` — and a
translation pair `(P_S, P_T)` of a single unit. We study one concrete input `i`
drawn from the unit's declared operating domain `D` (the `*_range` fields of the
shared IR; see [`IR.md`](IR.md)).

A conforming C implementation is licensed to do *anything* on an input that
triggers undefined behavior. We approximate the set of behaviors a conforming
implementation may exhibit by three measurements of `P_S` on `i`:

* `o0` — observable of `P_S` built `-O0` (UB usually "benign");
* `o2` — observable of `P_S` built `-O2` (the optimizer may exploit UB);
* `san` — whether a `-fsanitize=undefined` build **traps** on `i` (a *witness*
  that `i` actually reaches UB, not just that we feared it might).

The target side is recorded as: is the target outcome **defined** (a normal
value, or a clean trap/panic the target language *guarantees*), and is it
**deterministic** across repeated runs.

## Definition: divergence modulo source-undefinedness

The pair **diverges at `i`** iff all three clauses hold:

* **(P) Premise — source undefinedness is reached.** The sanitizer build traps
  on `i`. This is what makes a verdict *rooted in UB*: we never report a
  divergence that is merely a value mismatch on a defined input.
* **(T) Target is defined.** The target outcome is defined (and, in the
  `trap_vs_defined` mode, deterministic).
* **(C) Consequence**, parameterized by *mode*:
  * `exploited` — `o0` and `o2` are both defined values that **disagree**: the
    optimizer demonstrably turned the UB into an observable change.
  * `trap_vs_defined` — the consequence *is* the definedness gap: C is undefined
    here while T is defined. Here (C) reduces to (P) ∧ (T) (plus target
    determinism). Most crashing UB classes (division by zero, out-of-range
    shift, `INT_MIN / -1`) live here.

## Why it is one-sided (and why that is the point)

Divergence is a *witnessed failure* of "observational equivalence on defined
inputs". Equivalence would require that for **every** defined `i` the single
defined target observable coincide with a single stable source observable.
Proving that universally is not the tool's claim. Instead the tool only ever
**exhibits a witness** of divergence, and only when (P) holds — so every
positive verdict is backed by a concrete, re-executable input that provably
reaches source UB. Outside a covered class, or when a solver abstains, the tool
says nothing; it never claims equivalence.

## The math and the implementation are the same object

`semantics.is_divergence(obs)` is the boolean form of the definition above.
`semantics.observation_from_reexec(result)` lifts a real
[`ReexecHarness`](../src/ub_oracle/reexec.py) run into an `Observation`, reusing
the harness's own measurements (`ub_reachable`, the O0/O2 stdout, `rust_defined`)
verbatim. The theorem

```
is_divergence(observation_from_reexec(result)) == result.confirmed
```

is checked by `semantics.coincides_with_harness` and exercised by the test suite
— including, when a toolchain is present, on programs compiled and run for real
(`test_semantics_predicate_coincides_with_harness_on_real_programs`). So the
operational `confirmed` flag the harness sets is *exactly* the formal predicate,
not an informal approximation of it.

## Relationship to completeness (Step 81)

[`COMPLETENESS.md`](COMPLETENESS.md) is the *search-side* companion to this
*confirmation-side* definition. Completeness says the symbolic search finds a
candidate witness whenever one exists on a class's bounded fragment; the
semantics here say what it means for a candidate, once re-executed, to *count*
as a confirmed divergence. Together: the search misses no fragment witness, and
every reported witness satisfies the formal divergence definition against real
compilers.
