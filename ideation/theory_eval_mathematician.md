# Theory Evaluation: Mathematician's Verdict

## Proposal: proposal_00 — "Semantic Reconciliation for C↔Rust Equivalence Verification"

**Evaluation Method:** Claude Code Agent Teams — three independent expert evaluators (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial debate, synthesis, and independent verification signoff.

---

## THE MATHEMATICIAN'S QUESTION: Is the Math Load-Bearing?

**Verdict: ~30% load-bearing, ~70% ornamental.**

The proposal claims four mathematical contributions: (1) a semantic configuration space, (2) an integer reconciliation algebra with overflow alignment contracts, (3) product program construction with coercion insertion, and (4) three-output soundness. Evaluated individually:

### What is genuinely load-bearing:

1. **Angelic nondeterminism for cross-language UB** — The choice to existentially quantify over C's undefined behavior outcomes when comparing against Rust's defined behavior is a *necessary modeling decision* that determines whether the tool produces useful results. The demonic alternative (∀-quantify) trivializes every function with reachable UB. This is borrowed from Alive2's single-language setting but correctly adapted to the cross-language case. Not novel math, but a correctness-critical architectural choice. Without it, the tool is useless.

2. **Directed ε-approximation for floats** — The observation that C's `FLT_EVAL_METHOD` extended precision makes the float tolerance relation asymmetric and non-transitive is one paragraph of standard numerical analysis, but it is *necessary*. The error accumulation bound (ε₁ + ε₂ + ε₁ε₂) at coercion points prevents silent precision-related false equivalences. Remove this and the float comparison is wrong.

3. **Trust boundary specification (Assumptions A1–A6)** — The explicit enumeration of which assumptions each verdict depends on, and the key insight that Divergent verdicts are independently validated by concrete execution (so they bypass A1–A3) while Equivalent verdicts require all assumptions, is not a theorem but a *contract*. It is load-bearing because it tells users exactly what to trust.

### What is ornamental:

1. **"Integer reconciliation algebra"** — This is standard bitvector theory (SMT-LIB QF_BV semantics) with new names. The `embed: BV(w,s) → ℤ` and `trunc: ℤ → BV(w,s)` maps are textbook. The roundtrip lemma (`trunc(embed(v)) = v`) is trivially true by construction. The overflow alignment contracts are conditional equality assertions — a 4×4 case analysis over overflow modes, not an algebraic structure. A competent engineer could write the SMT encodings in an afternoon from the English description alone. Calling this an "algebra" invites the objection "where are the algebraic properties?" — and there are none (no closure, no associativity, no identity element). **Strip the algebraic language. Call it what it is: a catalog of SMT encoding patterns indexed by overflow mode pairs.**

2. **Product program soundness "theorem"** — Product programs for relational verification are well-established (Barthe et al. 2011, Zaks-Pnueli). The "soundness theorem" here states that if all coercion assertions hold, then the functions agree on well-defined inputs. This is true by construction of the product program — it's the *definition* of what the assertions check, not a non-obvious consequence. Stating it as a theorem implies there is a surprising proof; there is not.

3. **"Semantic configuration space" as a mathematical object** — The triple σ = (ω, φ, ε) is a design pattern (strategy pattern for parameterized semantics), not a formal structure with mathematical properties worth studying. It becomes `enum SemConfig` in the implementation. The formalization adds notation without insight.

4. **The "LLVM IR erasure thesis" as a novel claim** — The observation that LLVM IR loses source-level semantic information is well-documented (Davis & Chisnall 2017, K-LLVM, Vellvm motivations, Alive2 documentation). The Skeptic's challenge is dispositive: the `nsw` flag in LLVM IR *is* the semantic distinction for overflow — it is not "erased" but "insufficiently leveraged by existing tools." The *empirical quantification* on C2Rust output is new; the observation is not.

### Bottom line on math:
This proposal does not require genuinely NEW mathematics. It correctly applies KNOWN mathematical techniques (bitvector theory, product programs, bounded model checking, angelic nondeterminism) to a NEW and practically important domain. The math is the reason the artifact is *correct*, but it is not the reason the artifact is *hard to build*. The difficulty is in engineering: two faithful language frontends, a parameterized IR, correct SMT encodings for every divergence class, and system integration. The math serves the engineering; it does not drive it.

**A deep mathematician's assessment:** The math is necessary but not sufficient, correct but not novel, load-bearing in spots but ornamental in aggregate. This is an engineering contribution dressed in mathematical clothing. The clothing should be tailored honestly — present the modeling decisions (angelic UB, directed ε) as what they are (sound applied choices), and present the encodings as what they are (a carefully worked-out catalog). Do not call the catalog an algebra.

---

## SCORES

| Criterion | Score | Evidence |
|-----------|-------|----------|
| **1. Extreme Value** | **8/10** | Real DoD mandate for C→Rust migration. No existing tool for source-level semantic verification across languages. C2Rust produces unsafe Rust with no behavioral guarantees. The divergence taxonomy alone is independently valuable. Minor deduction: C2Rust cross-checking exists as a weak dynamic baseline (not acknowledged). |
| **2. Genuine Software Difficulty** | **7/10** | Two language frontends (libclang + syn), custom parameterized SSA IR, product program construction, bounded symbolic execution + SMT encoding, differential fuzzer integration. ~52K LoC is honest for the described system. Individual components are well-understood; difficulty is in integration and semantic fidelity. Frontend correctness is the hardest unsolved problem. Deduction: scoping to C2Rust output avoids the hardest C-standard problems. |
| **3. Best-Paper Potential** | **6/10** | The empirical contribution (exhaustive divergence taxonomy on libsodium/zlib + demonstration of false equivalences in LLREVE/Alive2) is the strongest angle. The formal contributions are incremental — no new theorem a PLDI reviewer hasn't seen. Target OOPSLA/ISSTA, not PLDI. Best paper requires perfect execution and genuinely surprising divergence statistics. Path exists but is narrow. |
| **4. Laptop-CPU Feasibility** | **8/10** | Modular function-level analysis. Z3 on QF_BV resolves most queries in <2s. 10s SMT timeouts prevent pathological cases. Fuzzing fallback at thousands of exec/sec. 2–4 hour estimate for 400+ function pairs is plausible (may stretch to 8–12h with FP-heavy functions). No GPU, no human annotation, no cloud required. |
| **5. Feasibility** | **5/10** | Phase 1 (differential fuzzer, 8–12K LoC, 4 months): 85% probability. Phase 2 (symbolic verification, 12–18K LoC, months 5–12): 50% probability. Full 52K LoC system: 18–24 months realistic, not 10. Comparable tools (KLEE, Alive2, LLREVE) took 2–4 years. Phased strategy is smart risk management — even partial delivery is publishable. Team size unspecified, which is a critical gap. |

**Composite: 6.8/10**

---

## FATAL FLAWS

1. **Ornamental math framing (MEDIUM):** The "integer reconciliation algebra" is standard BV theory renamed. Calling it an algebra invites rejection at formal venues. All three evaluators agree. **Fix: rename to "encoding catalog" or "verification obligation table."**

2. **IR erasure thesis overstated (MEDIUM):** The `nsw` flag counterexample shows "erased" is too strong; "insufficiently preserved for cross-language equivalence" is defensible. The observation is documented in prior work. **Fix: reframe as "first empirical quantification" not "novel thesis."**

3. **Frontend correctness unfalsifiable (MEDIUM-HIGH):** The entire soundness argument rests on Assumption A1 (unproven, unprovable without verified frontend). Silent misencoding produces wrong verdicts — the worst failure mode for a verification tool. **Fix: build differential test corpus in month 1; invest heavily in continuous frontend validation.**

4. **Timeline unrealistic as stated (MEDIUM):** 52K LoC in 10 months requires 3+ experienced developers. Solo PhD student gets Phase 1 only. **Fix: state team size; commit to Phase 1 as deliverable; frame Phase 2 as stretch goal with 18-month timeline.**

5. **No flaw is actually fatal.** All are fixable with reframing and honest scoping. The technical approach is sound.

---

## VERDICT: **CONTINUE**

### Conditions (binding):

1. **Strip the mathematical costume.** The proposal is a good engineering project wearing an ill-fitting mathematical disguise. Drop "algebra," drop "novel thesis." Lead with the empirical contributions: divergence taxonomy, IR erasure demonstration, tool evaluation on production libraries.

2. **Commit to Phase 1 as the deliverable.** The semantically-aware differential fuzzer (8–12K LoC, 4 months) is the high-probability, independently publishable deliverable. Phase 2 (symbolic verification) is the stretch goal. Do not let Phase 2 ambition jeopardize Phase 1.

3. **Build frontend test corpus in month 1.** Frontend correctness is the Achilles heel. Differential testing of both frontends against compiler output on a curated corpus of integer promotion edge cases, signed/unsigned conversions, and platform-dependent types must begin immediately.

4. **Target OOPSLA or ISSTA, not PLDI.** The contribution is a system + empirical study, not novel theory. OOPSLA values end-to-end systems with strong evaluation. ISSTA values testing tools with surprising findings. PLDI wants theorems this proposal doesn't have.

5. **Benchmark against differential testing baseline.** If running both programs on boundary values from the divergence table catches 80% of what the symbolic engine catches, the symbolic engine's marginal value must be demonstrated empirically, not assumed.

6. **State team size and realistic timeline.** Solo: Phase 1 in 10 months. Team of 3: full system in 15–18 months.

---

## TEAM PROCESS RECORD

| Role | Agent | Key Finding |
|------|-------|-------------|
| Independent Auditor | Evidence-based scoring | Value 8, Difficulty 7, Best-Paper 6, Laptop 8, Feasibility 5. IR erasure is well-known, not novel. Frontend correctness is unfalsifiable. |
| Fail-Fast Skeptic | Aggressive rejection | 10 claims tested: 1 SUPPORTED, 5 WEAK, 4 UNSUPPORTED. Math is ornamental — 3-line pseudocode captures the core. No fatal flaw found despite aggressive search. |
| Scavenging Synthesizer | Value salvage | Taxonomy, IR-erasure demo, and fuzzer are undersold. "Semantic gap finder" framing is strictly better. Angelic UB is load-bearing. "Algebra" is not. |
| Adversarial Debate | Challenge-response | 4 disagreements resolved. Consensus: math is ~30% load-bearing. Best-paper path exists but is narrow. Phase 1 is the real deliverable. |
| Independent Verifier | Signoff | APPROVED. Internally consistent, evidence-based, fair, no material omissions. |

**Verification signoff: APPROVED.**
