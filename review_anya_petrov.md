# Review: SemRec: Cross-Language Equivalence Verification

**Reviewer:** Prof. Anya Petrov (Applied Mathematics & Computational Science, ETH Zürich)
**Score: 5/10**
**Expertise Weight: 0.3** (limited alignment; this is primarily a PL/verification project with minimal mathematical content)

---

From a computational mathematics perspective, this project has limited mathematical depth. The core techniques — SMT encoding of bitvector constraints, product program construction — are standard in the program verification literature.

**Strengths.** The encoding of language-specific integer overflow semantics as bitvector constraints is technically correct and practically useful. The Z3 query performance (39 queries in ~3.1 seconds) demonstrates that the SMT encoding is efficient for the target problem size.

**Weaknesses.** There is no significant mathematical novelty. The σ-bridge coercion layer is a software engineering contribution, not a mathematical one. The lack of support for floating-point arithmetic, which has rich mathematical structure (IEEE 754 rounding modes, NaN propagation, denormalized numbers), limits the mathematical interest of the encoding problem. The bounded loop unrolling (K=32) is a standard technique with no novel analysis of completeness or termination.

**Verdict.** A solid engineering contribution to cross-language verification, but with minimal mathematical depth. The contribution is primarily in PL/verification methodology, not in computational mathematics. I defer to the formal methods reviewer for a more informed assessment.
