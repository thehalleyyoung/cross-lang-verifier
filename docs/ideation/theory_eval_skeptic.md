# Skeptic Verification: Proposal 00 — Semantic Reconciliation for C↔Rust Equivalence Verification

## Evaluation Process

**Team composition:** Three independent expert reviewers (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) produced independent evaluations, followed by an adversarial cross-critique round with forced disagreement resolution, and final independent verifier signoff.

**Process:** Independent proposals → adversarial critiques → synthesis of strongest elements → verification signoff.

---

## Proposal Summary

A ~52-55K LoC tool for verifying C-to-Rust migration correctness through source-level semantic analysis. Two phases: (1) differential fuzzing with semantic awareness (~8-12K LoC), (2) bounded symbolic verification via product programs + SMT/Z3 (~25-35K LoC). Core thesis: LLVM IR erases source-level semantic distinctions (overflow, float precision, error handling) that cause cross-language divergence, making IR-level tools (LLREVE, Alive2, UC-KLEE) fundamentally blind to migration bugs.

---

## Three-Pillar Evaluation

### Pillar 1: Does this deliver extreme and obvious value?

**Score: 7/10**

The problem is real and timely — DoD mandates, CISA advisories, and LLM-generated migrations create genuine demand. However, the proposal's central claim that "no tool exists to surface exact inputs where a migration diverges" is **factually false**:

- **RustAssure** (arXiv 2510.07604, 2025): Differential symbolic testing for LLM-transpiled C→Rust code. Uses KLEE on LLVM IR. Achieves 89.8% compilable Rust, 69.9% semantic equivalence across 5 real projects.
- **SMACK libm** (TACAS 2025): Cross-language C↔Rust equivalence checking of a libm port via LLVM IR → Boogie. Published with open artifacts.
- **Kani** (Amazon): Bounded model checker for Rust at MIR level.
- **CBMC + Kani composition**: Running CBMC on C side and Kani on Rust side with matching harnesses covers many divergence classes.

The value proposition survives only if the source-level vs IR-level distinction provides genuine incremental divergence detection — which is **asserted but not empirically demonstrated**. The proposal must be repositioned from "first tool for this problem" to "first source-level tool that catches IR-invisible divergences."

*Auditor: 6, Skeptic: ~5, Synthesizer: 8 → Consensus: 7*

### Pillar 2: Is this genuinely difficult as a software artifact?

**Score: 5/10 (MVP Phase 1) | 8/10 (full system)**

The full 52K LoC system — two language frontends, custom SSA IR, product programs, symbolic execution, differential fuzzing — is genuinely difficult. Each component has irreducible complexity. The C frontend (even scoped to C2Rust output) requires faithful encoding of C integer promotion rules via libclang. The product program alignment across syntactically different but semantically similar code is a hard heuristic problem.

However, the MVP (Phase 1 semantic-aware fuzzer, ~10K LoC) is a tractable engineering project: parse signatures via libclang/syn, generate FFI harness, fuzz with boundary-value seeds, report divergences. This is hard but well-understood fuzzing engineering, not novel systems research.

The difficulty score depends entirely on which version is being evaluated. The full system is 3-4 PhD theses; the MVP is one strong engineering project.

*Auditor: 7, Skeptic: 7, Synthesizer: 9(full)/5(MVP) → Consensus: 5(MVP)/8(full)*

### Pillar 3: Does this have real best-paper potential?

**Score: 6/10**

The competitive landscape has eroded first-mover advantage. RustAssure and SMACK (TACAS 2025) occupy adjacent territory. The remaining novel contributions are:

1. **The IR erasure thesis as empirical result** — demonstrating concrete function pairs where IR-level checkers report "equivalent" and source-level analysis finds divergences. This is well-known *in principle* among compiler researchers (lowering erases source semantics — that's the point of IRs), but **never empirically measured for cross-language equivalence checking**. If demonstrated with striking data, this is a genuine negative result.

2. **The divergence taxonomy** — first systematic characterization of semantic gaps in C2Rust output and LLM translations across production libraries. "37% of libsodium functions contain signed overflow divergence sites" would be a landmark empirical finding.

3. **The semantic divergence table** — a portable, reusable specification of where C and Rust semantics differ, with per-operation SMT encodings. Independently useful for anyone building C↔Rust tooling.

Best-paper at **FSE/ISSTA**: plausible (6-7/10) if the taxonomy is striking. Best-paper at **PLDI/OOPSLA**: unlikely without the full symbolic verification system working on real code.

The "integer reconciliation algebra" is **not** a genuine mathematical contribution. The proposal's own internal review correctly identified it as a case table dressed in algebraic terminology (no closure, no identity, no associativity). The embed/trunc maps are standard BV↔ℤ interpretation functions implemented in every SMT solver. Calling this an "algebra" in a paper submission would invite justified skepticism.

*Auditor: 5, Skeptic: ~4, Synthesizer: 8 → Consensus: 6*

---

## Constraint Evaluation

### Laptop-CPU Feasibility & No-Humans: 8/10

The fuzzer runs trivially on a laptop CPU. Function-level modular analysis with independent tasks is parallelizable. No GPU dependency. No human annotation needed for C2Rust subset (explicit types, no generics).

Caveats: QF_FP queries will NOT resolve in 10-second timeouts for non-trivial float functions (Z3's FP theory is dramatically slower than BV). Pointer-heavy code with QF_ABV will similarly timeout. The "2-4 hours for 400+ pairs" estimate is optimistic — expect 6-12 hours with many timeouts routed to fuzzing. But these are full-system concerns; the MVP fuzzer is fine.

### Feasibility: 6/10 (MVP) | 3/10 (full system)

**Full system (52-55K LoC, 10-12 months): INFEASIBLE.** All three reviewers agree. For context: KLEE is ~50K LoC built by a Stanford group over 2+ years; Alive2 is ~30K LoC over 2+ years. The proposal requires 5-7K LoC/month of research-quality compiler infrastructure — approximately 250 verified lines/day. Most research teams produce 1-2K LoC/month.

**MVP Phase 1 (~10K LoC, 4 months): FEASIBLE.** Differential fuzzing is well-understood. libclang and syn are mature. FFI harness generation is tractable. Coverage-guided fuzzing with semantic seeds is a natural extension of existing tools.

---

## Fatal Flaw Analysis

| # | Flaw | Severity | Mitigable? |
|---|------|----------|------------|
| **F1** | **C2Rust preserves C's UB semantics via `unsafe` Rust.** The claimed divergences (C UB vs Rust wrap) may not exist in C2Rust output — C2Rust's whole purpose is behavioral preservation. The primary evaluation target may yield zero divergences. | **HIGH** | Yes: pivot to LLM translations as primary evaluation target |
| **F2** | **"No tool exists" claim is false.** RustAssure, SMACK (TACAS 2025) directly contradict this. Reviewers will flag immediately. | **HIGH** | Yes: rewrite related work, position as source-level improvement |
| **F3** | **52K LoC full system is infeasible in 12 months.** | **HIGH** | Yes: already mitigated by phasing to MVP |
| **F4** | **syn cannot handle idiomatic Rust** (traits, generics, closures). LLM translation support is overpromised for non-trivial translations. | **MEDIUM** | Yes: scope to unsafe Rust only |
| **F5** | **"Integer reconciliation algebra" overclaims.** It's a lookup table, not an algebra. | **MEDIUM** | Yes: drop algebraic framing |
| **F6** | **Evaluation circularity.** Tool is scoped to structurally similar translations; C2Rust output is structurally similar by design; so evaluation proves tool works on what it was designed for. | **MEDIUM** | Partially: LLM translations break circularity but may have high structural-mismatch rejection rate |

**No single fatal flaw warrants immediate abandonment**, but F1 and F2 together are near-fatal if unaddressed. The proposal's premise depends on divergences existing in the evaluation target (F1) and being novel over prior art (F2).

---

## Scores Summary

| Dimension | Score | Self-Assessed | Delta |
|-----------|:-----:|:------------:|:-----:|
| Extreme Value | **7** | 9 | **-2** |
| Genuine Software Difficulty | **5** (MVP) | 8 | **-3** |
| Best-Paper Potential | **6** | 9 | **-3** |
| Laptop-CPU & No-Humans | **8** | ✓ | — |
| Feasibility | **6** (MVP) / **3** (full) | 6 | **0 / -3** |

**Composite (MVP-scoped): 6.4/10**

---

## VERDICT: **CONTINUE** — Phase 1 Only, with Mandatory Gates

### What Continues

The MVP — a semantic-aware differential fuzzer (~10K LoC) producing a divergence taxonomy — is the right-sized project. It delivers:

1. **The IR erasure thesis demonstrated empirically** (concrete function pairs where IR-level tools miss divergences source-level fuzzing catches)
2. **First systematic divergence taxonomy** across C2Rust and LLM translations of production libraries
3. **An independently useful tool** for validating C→Rust migrations
4. **Target venue: FSE or ISSTA** (not PLDI/OOPSLA without Phase 2)

### What Dies

- Phase 2 (symbolic verification, product programs, SMT): **deferred indefinitely** until Paper 1 accepted
- The 52K LoC framing: **dead**
- "Integer reconciliation algebra" branding: **dead** — call it a semantic divergence table
- "No tool exists" claim: **dead** — reframe as "existing IR-level tools miss source-semantic divergences"
- PLDI/OOPSLA venue targeting: **dead for Paper 1** — aim FSE/ISSTA

### Mandatory Gates

| Gate | Deadline | Criteria | If Fail |
|------|----------|----------|---------|
| **G1: Premise validation** | Week 2 | Demonstrate ≥1 real semantic divergence in LLM-generated Rust translation that existing differential fuzzing (without semantic seeds) misses | **ABANDON** |
| **G2: C2Rust divergence check** | Month 2 | Run C2Rust on tiny-AES-c or similar; inspect 20 pairs for actual divergences. If <5% divergence rate, pivot evaluation to LLM translations as primary | Pivot evaluation strategy |
| **G3: Prior art differentiation** | Month 3 | Compare against RustAssure/SMACK on 10 pairs. If they catch ≥80% of same divergences, source-level contribution is incremental | Pivot to taxonomy-only paper |
| **G4: MVP delivery** | Month 4 | Deliver semantic-aware fuzzer finding ≥10 real divergences in real code | **ABANDON Phase 2** |

### Risk-Adjusted Recommendation

The proposal has a genuine golden core: the problem is real, the IR erasure thesis is demonstrable, and the divergence taxonomy has lasting value. But the full system is 3x overscoped and the novelty claims are eroded by recent prior art (RustAssure, SMACK). The right move is to build the minimum tool that demonstrates the thesis and produces the taxonomy, publish at FSE/ISSTA, and then decide whether the symbolic verification layer justifies a second paper.

**Probability of publishable outcome (Phase 1 MVP): ~70%**
**Probability of best-paper (Phase 1 at FSE/ISSTA): ~20%**
**Probability of completing full 52K LoC system: ~15%**

---

## Reviewer Signoff

| Role | Verdict | Conditions |
|------|---------|------------|
| Independent Auditor | CONTINUE | Fix prior art claims, extend timeline to 18mo for full system |
| Fail-Fast Skeptic | CONTINUE (reluctant) | Phase 1 only; prove Alive2/RustAssure miss within 4 weeks |
| Scavenging Synthesizer | CONTINUE | MVP first, then decide on Phase 2 |
| Independent Verifier | **APPROVED** | All mandatory gates enforced; pivot to LLM translations as primary evaluation target |

**Final verdict: CONTINUE (Phase 1 MVP only, with gates)**
