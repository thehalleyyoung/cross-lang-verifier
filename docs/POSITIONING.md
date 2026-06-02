# Positioning: what this tool does that adjacent verifiers do not

This page situates `cross-lang-verifier` against the closest existing tools and
states precisely the gap it fills. The short version: existing verifiers reason
about **one** program (or two versions of the *same* language); this tool reasons
about a **C source** and its **translation into a different language** and asks a
question none of them ask — *does the translation diverge specifically because
the C source relied on undefined behavior?* Every claim below is backed by a
capability listed in [`CAPABILITIES.md`](../CAPABILITIES.md) and mapped to code
in [`TRACEABILITY.md`](TRACEABILITY.md).

## The question we answer

Given a C unit `P_C` and a candidate translation `P_T` (T ∈ {Rust, Go, Swift}),
exhibit a concrete input on which:

* `P_C` reaches **undefined behavior** (witnessed by a trapping sanitizer build),
* the optimizer's freedom over that UB makes `P_C`'s observable behavior
  **unstable** (an `-O0`/`-O2` value flip) **or** `P_C` traps while `P_T` is
  defined, and
* `P_T` is **defined** (and deterministic).

This is *divergence modulo source-undefinedness* (see
[`SEMANTICS.md`](SEMANTICS.md)). It is a one-sided witness property — we never
claim equivalence — and the witness is always re-executable against real
compilers.

## Adjacent tool families and the specific gap

### 1. Bounded model checkers for C (CBMC, ESBMC, …)

CBMC-style tools symbolically execute a **single C program** and check assertions
or built-in properties (including many UB classes) up to a bound. They are
excellent at *finding* UB in C. They do **not** model a *second program in
another language*, and so cannot express the proposition "this C UB causes the
Rust/Go/Swift translation to observably differ." They answer "is this C program
safe?"; we answer "is this *translation* faithful *despite* the C program's
UB-sensitivity?"

What we reuse from this lineage: SMT-backed witness search (Z3). What we add: a
*paired* target program with its own language semantics, and a re-execution
oracle that confirms the cross-language observable gap.

### 2. Single-language equivalence / translation-validation checkers

Translation validators and equivalence checkers (e.g. compiler IR equivalence,
peephole-superoptimizer validators, same-language refactoring checkers) verify
that two programs **in the same language / IR** agree. Cross-language migration
breaks their core assumption: there is no shared IR, the type and trap semantics
differ, and "equivalent" is not even well-defined on inputs where the *source*
is undefined. Our contribution is precisely to make the goal well-posed on those
inputs (by rooting divergence in witnessed source UB) and to operationalize it
across a *frozen, pair-agnostic IR* that every frontend lowers into (see
[`IR.md`](IR.md)) while the target half is supplied by data-driven semantics
packs.

### 3. Target-language verifiers (Kani, Prusti, Miri, `go vet`/race, …)

These reason about the **target** in isolation: Kani/Miri check Rust for panics
or UB, Prusti proves Rust contracts, Go's tooling finds races and mistakes. They
can tell you the *target* is well-defined — which is exactly clause (T) of our
definition — but they have no notion of the **C source** the code was translated
from, and therefore cannot detect that a *defined* target silently disagrees with
what the original (UB-bearing) C did under some compiler. A translation can pass
every target-side check and still be an incorrect port; that is the failure mode
we surface.

What we reuse conceptually: "target is defined" is established the way these
tools would endorse (definedness + determinism of the target outcome). What we
add: the cross-language comparison against the C source's UB-induced instability.

### 4. Differential testing / fuzzing of translations

Naively fuzzing both programs and diffing outputs produces **false alarms**: on
inputs where C is undefined, *any* difference is "expected" and not a real bug,
while on inputs where C is defined a difference is a genuine port bug. Undirected
diffing cannot tell these apart. Our oracle's UB-premise (clause (P)) is exactly
the filter that separates "difference licensed by UB" from "difference that is a
real divergence", and our SMT search is *directed* at the boundary inputs that
trigger UB rather than hoping a fuzzer stumbles onto them. (The repository still
includes differential-fuzz utilities, but as a cross-check, not the oracle.)

## Summary table

| Tool family | Programs reasoned about | Notion of "wrong" | Handles cross-language UB-rooted divergence? |
| --- | --- | --- | --- |
| BMC for C (CBMC/ESBMC) | one C program | assertion / built-in UB violation | No — no second-language program |
| Equivalence / translation validation | two same-language / same-IR programs | observational inequivalence | No — assumes a shared language/IR; UB inputs ill-defined |
| Target verifiers (Kani/Prusti/Miri) | one target program | target-side UB / contract / panic | No — no source program to compare against |
| Differential fuzzing | two programs, undirected | any output difference | Partially — but cannot separate UB-licensed vs real differences, and is undirected |
| **This tool** | **C source + its translation (Rust/Go/Swift)** | **defined target diverges from UB-unstable C** | **Yes — the design goal** |

## The gap, in one sentence

No existing verifier takes a C program *and* its translation into another
language and isolates the inputs where the translation's defined behavior
diverges from the C source *because* that C source depended on undefined
behavior — which is the single question this tool is built to answer soundly,
with re-executable witnesses, and (on the integer fragment) completely.
