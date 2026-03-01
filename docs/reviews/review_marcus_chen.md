# Review: SemRec: Cross-Language Equivalence Verification

**Reviewer:** Prof. Marcus Chen (Machine Learning & Statistical Learning Theory, Stanford)
**Score: 7/10**
**Expertise Weight: 0.5** (moderate alignment via the CEGAR LLM-in-the-loop repair component)

---

My review focuses on the CEGAR LLM-in-the-loop component, which is the primary intersection of this work with ML/AI.

**Strengths.** The CEGAR loop — SemRec finds a divergence, feeds the counterexample back to the LLM, the LLM repairs its translation — is a compelling demonstration of human-AI collaboration paradigms. The 45% fully-verified success rate (9/20) with average 3.2 iterations is a realistic and honest result. The finding that signed_overflow (6 bugs) and INT_MIN/−1 (6 bugs) are the dominant bug types provides actionable feedback for LLM training: these are the systematic failure modes that LLM-based translators need to address. The counterexample-guided feedback is more informative than simple pass/fail, and the 17 bugs found demonstrates that the loop genuinely improves translation quality.

**Weaknesses.** The LLM component is treated as a black box — the paper does not investigate why the LLM makes specific errors or how to improve its performance beyond simple counterexample feedback. More structured feedback (e.g., identifying the specific semantic rule violated) could improve the repair success rate. The 45% success rate means 55% of functions remain unverified after the CEGAR loop, with no analysis of why these cases fail — is it LLM inability to understand the feedback, Z3 timeout, or fundamental semantic gaps in the translation? The comparison against SemRec-free differential testing achieving comparable accuracy (80% vs 80%) is concerning: it suggests that for the benchmarks tested, the formal verification component adds little value over simpler testing approaches. The benchmark is small (20 functions for CEGAR, 32 for core) and may not represent the distribution of real migration targets.

**Verdict.** The CEGAR LLM-in-the-loop is a useful contribution to the growing field of LLM-augmented formal methods. The honest reporting of success rates and failure analysis is commendable. The tool would be stronger with more structured LLM feedback and a larger, more diverse benchmark.
