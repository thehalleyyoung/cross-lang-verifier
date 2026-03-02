# Review: SemRec — Cross-Language Equivalence Verifier

**Reviewer:** Cesar Sanchez  
**Persona:** Formal Verification & AI Researcher  
**Date:** 2026-03-02  

---

## Summary

SemRec is a formal verification oracle for C↔Rust translation correctness that operates at the source level using Z3 SMT solving. Given a C function and its Rust translation, it returns a structured verdict (equivalent/divergent/unknown) with counterexamples and repair hints. The key innovation is the σ-bridge — a parameterized semantic configuration that encodes language-specific semantics (C11 undefined behavior vs. Rust's defined behavior for overflow, shifts, etc.) at the IR level. The system achieves 86.6% classification accuracy on 202 benchmark pairs.

## Strengths

1. **The σ-bridge is the right abstraction.** Encoding language-specific semantics as a parameterized configuration (OverflowMode, ShiftModel, ErrorModel) rather than hardcoding them into the verifier is a clean design decision. It separates the "what differs between C and Rust" question from the "how to verify equivalence" question. The ablation showing accuracy drops from 80% to 40% without the σ-bridge confirms it is load-bearing, not decorative. This design could be extended to other language pairs (C↔Go, C↔Zig) by adding new semantic configurations.

2. **Source-level UB detection fills a genuine gap.** Differential testing (compile both, compare outputs) cannot detect UB-dependent divergences because compilers erase UB. SemRec's source-level SMT analysis identifies inputs that trigger UB in C (signed overflow, shift-by-width, division-by-zero) and compares them against Rust's defined behavior for the same inputs. This is the correct approach for verifying C-to-Rust translations where UB elimination is a primary goal.

3. **QF_BV encoding is decidable and complete within scope.** The choice of quantifier-free bitvector logic for SMT encoding gives decidable and complete verification for the supported fragment (integer arithmetic, bitwise operations, bounded control flow). No approximation or abstraction is needed within this fragment. This is a strength over approaches that use abstract interpretation or bounded model checking with heuristic bounds.

4. **Conditional soundness theorem is well-stated.** The theorem clearly states: "within the supported fragment (no pointers, bounded loops), if the oracle returns 'equivalent,' the programs are semantically equivalent; if it returns 'divergent,' the counterexample demonstrates a genuine divergence." The conditional nature is honest and appropriate — the tool makes no claims about the unsupported fragment.

## Weaknesses

1. **No pointer/memory handling is a critical limitation.** Pointers and heap allocation are ubiquitous in C code. The tool scores 0% on memory-related benchmarks. For C-to-Rust translation verification, pointer semantics (aliasing, ownership, lifetime) are precisely where the most subtle bugs occur. C2Rust translations frequently introduce raw pointers that must be refactored to safe Rust references — this is the core correctness challenge, and SemRec cannot address it. A tool that handles arithmetic but not pointers addresses the easy part of the problem while ignoring the hard part.

2. **CEGAR loop fails precisely where it matters.** The CEGAR integration (translate → verify → feedback → repeat) achieves 0% convergence on functions with undefined behavior — the core use case. This result suggests that LLMs cannot reliably repair UB-related semantic divergences from counterexample feedback alone. The CEGAR design is sound in principle (counterexample-guided repair is a well-established paradigm), but the LLM repair oracle is too weak for this problem class. The paper should analyze why LLMs fail on UB repairs and what additional guidance would be needed.

3. **Benchmark dataset is synthetic and small.** All 352 benchmark pairs are author-curated, not extracted from real C2Rust translations. The evaluation tells us how the tool performs on hand-crafted examples, not on actual translation outputs. Real C2Rust output contains complex pointer arithmetic, macro expansions, goto statements, and platform-specific constructs that the benchmark does not cover. Without evaluation on real C2Rust output, the 86.6% accuracy is an upper bound on practical performance.

4. **Hand-written parsers are brittle.** Recursive descent parsers for C and Rust cannot handle the full language grammars. C's declaration syntax, preprocessor directives, and platform-specific extensions make robust parsing extremely difficult. Rust's lifetime annotations, trait bounds, and macro hygiene add further complexity. Using established front-ends (Clang AST for C, rustc HIR for Rust) would be more robust, though it would require different IR lowering.

5. **Bounded model checking at K=32 is restrictive.** Loop-heavy code (common in data processing, string operations, array manipulation) requires deeper unrolling or full loop invariant reasoning. The 25% accuracy on loop benchmarks reflects this limitation. For C-to-Rust translations of systems code, loops are the norm, not the exception.

6. **Confidence scoring is uncalibrated.** The OracleResult.confidence field is set to 1.0 for "definitive" results and lower values for timeouts. No calibration analysis is provided showing that the confidence score predicts actual correctness probability. An uncalibrated confidence score can mislead users into trusting incorrect results.

## Minor Issues

- The ablation study uses only 10 pairs — too small for statistical significance.
- No discussion of how the tool handles C implementation-defined behavior (int width, char signedness, struct padding), which differs across platforms.
- The repair hint mechanism in CEGAR is underdocumented — how are hints generated from Z3 counterexamples?

## Questions for Authors

1. What is the plan for pointer/memory handling? Is there a fundamental barrier to extending the σ-bridge to cover pointer semantics (ownership, aliasing)?
2. Why does the CEGAR loop fail on UB functions? Can you analyze the LLM repair patterns to identify what guidance would enable successful repair?
3. Have you evaluated the tool on actual C2Rust output? If not, what obstacles prevent such an evaluation?

## Overall Assessment

SemRec makes a solid contribution with the σ-bridge abstraction and source-level UB detection for C↔Rust equivalence verification. Within its supported fragment (arithmetic, control flow, bounded loops), the tool provides decidable and complete verification with honest conditional soundness guarantees. However, the absence of pointer handling, the CEGAR failure on UB functions, and the synthetic benchmark significantly limit practical impact. The tool addresses the easy half of C-to-Rust verification (arithmetic semantics) while leaving the hard half (memory safety, ownership) untouched. This is a strong foundation for future work, not a deployable verification tool.

**Recommendation:** Weak Accept — clean design and honest evaluation, but scope limitations reduce practical impact significantly.

**Confidence:** 5/5
