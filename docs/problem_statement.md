# Cross-Language Equivalence Verification via Semantic Reconciliation

## Title

**Semantic Reconciliation for C↔Rust Equivalence Verification: Proving Migration Correctness Through Source-Level Semantic Analysis**

## Problem Statement

Every C-to-Rust migration terminates with the same unanswerable question: *did we preserve the original behavior?* The state of practice is to run the test suite and hope, but test suites encode a vanishing fraction of actual program semantics. Migration bugs escape to production because the source and target disagree on edge cases no test covers: a signed integer overflow that is undefined behavior in C but wraps in Rust, a division by zero that is UB in C but panics in Rust.

**The LLVM IR erasure problem.** The naive approach—compile both programs to LLVM IR and apply existing equivalence checkers like LLREVE or Alive2—fails because LLVM IR erases the semantic distinctions that cause cross-language divergence. When Clang compiles `x + y` for signed integers, it emits `add nsw` (no signed wrap). When rustc compiles the same operation in release mode, it also emits `add nsw`. At the LLVM IR level, both look identical, yet their source-language semantics differ: C says overflow is undefined behavior, while Rust defines it as wrapping mod 2³². A tool operating on LLVM IR cannot distinguish these cases.

**Concrete demonstration.** Consider `int sum = a + b` where `a = INT_MAX, b = 1`. In C, this is undefined behavior. In Rust with `wrapping_add`, this wraps to `INT_MIN`. Both compile to the same LLVM IR. SemRec detects this divergence at the source level by encoding the different overflow semantics into Z3 bitvector constraints.

## What SemRec Actually Is

SemRec is a source-level semantic equivalence verifier for **single C↔Rust function pairs**. Given a C function and its Rust translation, it produces one of three verdicts:

- **Equivalent**: functions agree on all inputs where the C function has defined behavior (UNSAT from Z3).
- **Divergent**: with a concrete counterexample extracted from the Z3 model.
- **Unknown**: Z3 times out or hits unsupported theory.

### Architecture

The pipeline has six stages, all implemented in Python:

1. **CParser** — hand-written recursive descent parser for C
2. **RustParser** — hand-written recursive descent parser for Rust
3. **CIRLowering / RustIRLowering** — lower ASTs to a shared typed SSA IR
4. **FunctionAligner** — structurally aligns two IR functions
5. **ProductBuilder** — builds a product program with σ-bridge coercion points encoding per-language semantics (C11 UB vs. Rust wrapping/panicking)
6. **SMTEncoder → Z3** — encodes the product program as QF_BV constraints and checks with Z3

### Actual Scope

- **Single functions** of ~50 lines or fewer
- **Integer arithmetic**: signed/unsigned, 8–64 bit, overflow, division, shifts, casts
- **Control flow**: if/else, ternary, bounded loops (unrolled up to K=32)
- **No support for**: structs, pointers, heap, interprocedural analysis, concurrency, function pointers, strings

### Implementation

The implementation is ~49K lines of Python across 135 source files. The core pipeline (parsers, IR lowering, product builder, SMT encoder, solver) is the load-bearing subset; many other source files exist for auxiliary features that are not part of the verified pipeline.

## Value Proposition

**The missing safety layer for AI-assisted C-to-Rust migration.** GPT-4 and other LLMs are increasingly used for code migration, but they routinely introduce subtle behavioral divergences that pass human review. SemRec provides automated verification: an engineer generates a Rust translation via LLM, runs SemRec, and receives either a formal equivalence proof or a concrete counterexample.

**CEGAR LLM-in-the-loop.** The most compelling use case is the CEGAR loop: SemRec finds a divergence, feeds the counterexample back to the LLM, and the LLM repairs its translation. On 20 test functions, this achieves 45% fully-verified success rate (9/20) with an average of 3.2 iterations, finding 17 bugs total.

**What this tool does NOT claim.** We do not claim machine-checked proofs (our frontends are unverified). Equivalence verdicts are sound modulo frontend correctness and loop bound. We do not handle structurally dissimilar translations. We target the specific class of structurally similar translations produced by transpilers and LLMs, focused on integer arithmetic functions.

## Experiment Results

### Core Benchmark (32 pairs)
- 16 equivalent, 16 divergent, 0 unknown
- 100% pipeline success rate
- 39 SMT queries in ~3.1 seconds
- 13/16 divergences are IR-invisible

### CEGAR LLM-in-the-Loop (20 functions)
- 9/20 verified correct after repair (45%)
- 17 bugs found, avg 3.2 iterations
- Dominant bug types: signed_overflow (6), INT_MIN/−1 (6)

### Baseline Comparison (10 pairs)
- SemRec: 7/10 correct (70%), 6 divergences found
- Diff testing (10K + boundary): 10/10 correct (100%), 7 divergences found
- SemRec finds 1 divergence (saturating_add) diff testing misses
- Diff testing finds 2 divergences SemRec misses

### Ablation (10 pairs)
- Full SemRec: 80% accuracy (8/10)
- No σ-bridge: 40% accuracy (4/10) — σ-bridge halves the error rate
- Random testing: 80% accuracy (8/10)

## Slug

`cross-language-equivalence-verifier`
