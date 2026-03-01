# Review: SemRec: Cross-Language Equivalence Verification

**Reviewer:** Prof. Elena Vasquez (Formal Methods & Program Verification, MIT)
**Score: 7/10**
**Expertise Weight: 0.9** (strong alignment with program verification, SMT-based equivalence checking, language semantics)

---

SemRec addresses a genuine gap in the C-to-Rust migration verification landscape. The core insight — that LLVM IR erasure eliminates the very semantic distinctions (overflow behavior, division-by-zero handling) that cause cross-language divergence — is well-articulated and technically correct.

**Strengths.** The σ-bridge coercion layer is the central technical contribution, and it is well-designed. Encoding per-language semantic differences (C11 UB vs. Rust wrapping/panicking) at the source level, before lowering to a shared SSA IR, correctly captures divergences that LLVM IR-level tools cannot detect. The 13/16 IR-invisible divergences in the benchmark is compelling evidence. The product program construction with σ-bridge coercion points is a clean formalization. The CEGAR LLM-in-the-loop repair achieving 45% verification success on 20 functions is a pragmatically useful combination of formal verification and LLM-based code generation. The ablation showing that removing the σ-bridge halves accuracy (80% → 40%) demonstrates its necessity.

**Weaknesses.** The scope is very narrow: single functions of ~50 lines, integer arithmetic only, no structs/pointers/heap/concurrency. This excludes the vast majority of real C-to-Rust migration targets, which involve pointer-heavy code with heap allocation. The hand-written recursive descent parsers for C and Rust are a significant soundness risk — the C language grammar alone has hundreds of edge cases, and any parser bug could produce unsound results. The 84.2% overall accuracy (on a benchmark of 202 pairs) means the tool produces incorrect verdicts 16% of the time, which is concerning for a verification tool. The baseline comparison showing that differential testing matches SemRec's accuracy (80% vs 80% on the 10-pair ablation) undermines the verification value proposition — if random testing achieves comparable accuracy, the formal verification overhead may not be justified. The 49K lines of Python across 135 files suggests significant code complexity that is not subjected to any formal verification of the tool itself.

**Verdict.** A well-scoped contribution that correctly identifies the LLVM IR erasure problem and provides a technically sound source-level solution via the σ-bridge. The narrow scope is honest. The main concern is the gap between the "verification" framing and the practical accuracy — 84.2% is good for a research prototype but far from the soundness guarantees expected of a verifier. The CEGAR LLM loop is the most immediately useful contribution.
