# Review: SemRec — Cross-Language Equivalence Verifier

**Reviewer:** Roderick Bloem  
**Persona:** Machine Learning, Specification & Safety Researcher  
**Date:** 2026-03-02  

---

## Summary

SemRec verifies semantic equivalence between C and Rust functions using a σ-bridge that parameterizes language-specific semantics and an SMT-based product program verification pipeline. It includes a CEGAR loop that uses LLM-generated repairs guided by formal counterexamples. The tool targets the C-to-Rust migration correctness problem, with a focus on detecting undefined-behavior-related divergences that testing cannot catch.

## Strengths

1. **Directly addresses the C-to-Rust migration wave.** Government mandates (NSA, CISA) and industry initiatives (Android, Linux kernel) are driving massive C-to-Rust migration. Automated translation tools (C2Rust) produce syntactically correct but semantically fragile Rust code. SemRec fills the verification gap between translation and deployment. The timing and problem selection are excellent.

2. **CEGAR with LLM repair is a promising architecture.** The counterexample-guided repair loop — verify → find counterexample → generate repair hint → LLM produces new translation → re-verify — is the right architecture for human-in-the-loop or LLM-in-the-loop translation refinement. Even though current LLM repair success is low (16.7% overall, 0% for UB), the architecture is sound and will improve as LLMs improve.

3. **Specification-free verification.** The σ-bridge encodes language semantics, not program specifications. Users do not need to write specifications — the tool automatically derives verification conditions from the semantic differences between C and Rust. This is crucial for adoption: C-to-Rust migration teams are not formal methods experts.

4. **Structured verdict format.** The (verdict, counterexample, repair_hint) triple provides actionable output. A developer receiving "divergent on input x=2147483647 due to signed overflow; C wraps to -2147483648, Rust panics" can immediately understand and fix the issue. This is more useful than a bare "not equivalent" verdict.

## Weaknesses

1. **No safety-critical evaluation.** For a tool targeting migration correctness, evaluation on safety-critical codebases (automotive, avionics, medical device firmware) is essential. These are the domains where C-to-Rust migration is most motivated by safety concerns, and where verification failures have the highest consequences. The synthetic benchmark does not establish that the tool handles the code patterns found in safety-critical systems.

2. **CEGAR LLM repair is unreliable.** 0% convergence on UB functions means the CEGAR loop fails precisely on the motivating use case. From a safety perspective, an unreliable repair mechanism is worse than no repair mechanism: users may trust that the CEGAR loop will eventually produce a correct translation and stop manually reviewing. The tool should clearly warn when CEGAR fails to converge and require human review.

3. **Error handling verification is weak.** 50% accuracy on error handling benchmarks reveals that the tool struggles with C errno/Rust Result translation patterns. Error handling is one of the most safety-critical aspects of C-to-Rust migration: incorrect error propagation can cause silent data corruption, resource leaks, or safety violations. The σ-bridge's ErrorModel needs significant development.

4. **No integration with C2Rust pipeline.** The tool operates on standalone function pairs, not on C2Rust translation outputs. In practice, C2Rust produces entire crate translations with inter-function dependencies, global state, and extern blocks. SemRec's function-level verification cannot capture interprocedural correctness issues. An integration with C2Rust that verifies translation units (not just functions) would dramatically increase practical value.

5. **Scalability is uncharacterized.** All benchmarks are 10-50 LOC functions. Real C functions in safety-critical systems can be hundreds or thousands of LOC with deep call chains. The SMT encoding complexity scales with program size, and Z3 performance can degrade dramatically on larger encodings. No scalability evaluation is provided.

6. **No incremental verification.** If a developer makes a small change to the Rust translation (fixing a bug found by SemRec), the entire verification must be re-run. Incremental verification (re-check only affected code paths) would make the tool practical in iterative development workflows.

## Minor Issues

- The tool does not handle C preprocessor directives. Real C code relies heavily on #ifdef guards, platform-specific macros, and conditional compilation. A preprocessing step is needed.
- No support for C varargs or Rust closures, which are common in real code.
- The CLI interface (verify, cegar, bench) is minimal. An IDE integration (VS Code extension) would improve developer experience.

## Questions for Authors

1. Have you evaluated SemRec on any real C2Rust translation outputs? What obstacles prevent integration with the C2Rust pipeline?
2. What is the SMT solving time for functions of 100, 500, and 1000 LOC? Where does Z3 timeout?
3. How would you handle C functions with pointers that are translated to Rust functions with references/borrows?

## Overall Assessment

SemRec addresses the right problem at the right time: C-to-Rust migration correctness verification. The σ-bridge is a clean, extensible design, and the structured verdict format provides actionable developer feedback. However, the tool's practical impact is limited by scope (no pointers, small functions only), CEGAR reliability (0% on UB), and lack of real-world evaluation. For ML/specification/safety applications, the error handling weakness (50% accuracy) and absence of safety-critical benchmarks are the most concerning gaps. The tool is a promising prototype that needs significant engineering investment to become practically useful.

**Recommendation:** Weak Accept — right problem, clean design, but significant scope and evaluation gaps for practical deployment.

**Confidence:** 4/5
