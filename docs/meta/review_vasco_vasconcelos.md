# Review: SemRec — Cross-Language Equivalence Verifier

**Reviewer:** Vasco Vasconcelos  
**Persona:** Probabilistic Models & Formal Methods Expert  
**Date:** 2026-03-02  

---

## Summary

SemRec constructs product programs from C and Rust function pairs, encoding language-specific semantics via a σ-bridge, and verifies equivalence using Z3 SMT solving over quantifier-free bitvector (QF_BV) logic. The system supports integer arithmetic, bitwise operations, casts, structs/enums, floating-point, and bounded loops. A CEGAR loop integrates LLM-based repair. The tool achieves 86.6% accuracy on 202 benchmark pairs with <200ms average verification time.

## Strengths

1. **QF_BV is the ideal logic for this problem.** Bitvector arithmetic exactly captures the semantics of fixed-width integer operations in both C and Rust. Unlike integer arithmetic theories (LIA, NIA), bitvector logic correctly models overflow, truncation, sign extension, and two's complement representation. The decision to use QF_BV rather than a more abstract integer theory is correct and avoids the soundness issues that plague tools using mathematical integers to model machine integers.

2. **Product program construction is well-founded.** The product program approach — aligning the control flow of C and Rust programs and inserting coercion points where semantics diverge — is a standard technique in translation validation (Pnueli, Siegel, Singerman 1998). SemRec's contribution is instantiating this technique for C↔Rust with the σ-bridge handling semantic divergences. The coercion points (where C UB meets Rust defined behavior) are the key innovation in the product construction.

3. **Conditional soundness and decidability are clean results.** Within the supported fragment, the tool provides:
   - Soundness: "equivalent" verdicts are correct (no false equivalences).
   - Completeness: all divergences are detected (no missed divergences).
   - Decidability: verification always terminates with a definite answer.
   These three properties together make the tool reliable within its scope. The conditional nature (restricted to the supported fragment) is appropriately stated.

4. **Floating-point support via IEEE 754 strict mode.** Extending beyond integer arithmetic to floating-point is non-trivial. Z3's QF_FP theory handles IEEE 754 semantics (rounding modes, NaN propagation, signed zero), which differ between C's implementation-defined behavior and Rust's strict IEEE 754 compliance. This extension increases the tool's practical coverage.

## Weaknesses

1. **No treatment of non-determinism or underspecification.** C11 has numerous implementation-defined behaviors beyond undefined behavior: the size of int, the signedness of char, evaluation order of function arguments, struct padding and alignment. The σ-bridge encodes UB (overflow, shift) but does not parameterize implementation-defined behavior. This means the tool implicitly assumes a specific C implementation (likely x86-64 Linux GCC), reducing portability. A fully formal approach would quantify over all implementation-defined choices.

2. **Bounded model checking is fundamentally incomplete for loops.** The BMC depth K=32 is a pragmatic choice, but it means loops executing more than 32 iterations are not verified. More critically, the tool returns "equivalent" for loops within K iterations, which is unsound for the general case: programs may agree for 32 iterations and diverge on iteration 33. The conditional soundness theorem should explicitly state that loop verification is sound only up to K unrollings, not for all executions.

3. **The shared SSA IR is a significant design decision with underexplored consequences.** Lowering both C and Rust to a shared typed SSA IR requires making representation choices that may lose language-specific semantic information. For example:
   - C's implicit integer promotions and Rust's explicit casting rules must be faithfully represented.
   - C's `union` and Rust's `enum` with fields have different layout guarantees.
   - Rust's `Option<&T>` optimization (null pointer representation) has no C equivalent.
   If the IR lowering loses these distinctions, the product program may miss genuine divergences.

4. **No formal specification of the σ-bridge.** The semantic configuration is implemented as Python enums (OverflowMode, ShiftModel, etc.) but no formal semantics is provided. What exactly does "OverflowMode.UB" mean in the context of the SMT encoding? Does it introduce a havoc (arbitrary value) at overflow points, a precondition excluding overflow inputs, or an error state? The choice fundamentally affects what the tool verifies, and it should be formally specified, not just implemented.

5. **Absence of interprocedural analysis limits practical scope.** Real C programs use function calls extensively, including through function pointers. Rust uses traits and closures. The tool's intraprocedural analysis can verify individual function pairs but cannot verify calling-convention compatibility, ABI alignment, or behavior of function-pointer-based dispatch. For C-to-Rust migration, calling convention mismatches are a common source of bugs.

6. **No probabilistic or statistical analysis.** The 86.6% accuracy is reported as a single number without confidence intervals, statistical significance testing, or analysis of failure modes. Given the small benchmark (202 pairs, 10 pairs for ablation), the variance in accuracy estimates could be substantial. A bootstrap confidence interval would be informative.

## Minor Issues

- The SSA IR is described as "typed" but the type system's expressiveness is unclear. Can it represent Rust's affine types, lifetimes, or trait constraints?
- The dominator tree and dataflow analysis modules are implemented but their role in the verification pipeline is not clearly documented.
- The alias analysis module exists but is marked as limited; its interaction with the product program construction should be clarified.

## Questions for Authors

1. How does the σ-bridge encode C undefined behavior in SMT? Is UB modeled as havoc, precondition, or error state?
2. Is the bounded model checking soundness limitation clearly documented in the tool's output? Does the tool warn when verification depends on the K bound?
3. What fraction of C2Rust translation bugs involve implementation-defined behavior (not undefined behavior)?

## Overall Assessment

SemRec is a technically clean verification tool for C↔Rust equivalence within a well-defined fragment. The QF_BV encoding, product program construction, and conditional soundness results are solid. The σ-bridge is a novel and extensible abstraction for cross-language semantic differences. However, the tool's practical reach is limited by the absence of pointer handling, incomplete loop reasoning, and unparameterized implementation-defined behavior. The formal foundations are sound but the specification of the σ-bridge needs to be formalized beyond Python implementations. The tool is a strong starting point for cross-language verification research with clear directions for extension.

**Recommendation:** Accept — solid formal foundations with well-identified limitations and clear extension paths.

**Confidence:** 4/5
