# Theory Gate Verification Report

**Stage:** Verification
**Date:** 2026-02-22
**Method:** Claude Code Agent Teams — Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer with adversarial cross-critique and forced disagreement resolution.

---

## Executive Summary

One proposal evaluated. Three independent experts produced evaluations, followed by adversarial cross-critique and synthesis. The Fail-Fast Skeptic recommended ABANDON; the Independent Auditor and Scavenging Synthesizer recommended CONTINUE with radical rescoping. After adversarial debate, the synthesis resolved in favor of **CONTINUE — radically rescoped to MVP only**.

---

## Proposal: proposal_00 — Semantic Reconciliation for C↔Rust Equivalence Verification

### Description
A tool for verifying C-to-Rust migration correctness through source-level semantic analysis. Originally scoped as a ~52K LoC two-phase system: (1) differential fuzzing with semantic awareness, (2) bounded symbolic verification via product programs + SMT/Z3.

### Team Scores

| Dimension | Auditor | Skeptic | Synthesizer | Prior Evals (range) | **Consensus** |
|-----------|:-------:|:-------:|:-----------:|:-------------------:|:-------------:|
| Extreme Value | 6 | 4 | 7 | 5–8 | **6** |
| Genuine Difficulty | 7 | 8 | 5 | 5–8 | **5** (MVP) |
| Best-Paper Potential | 4 | 3 | 5 | 5–6 | **4** |
| Laptop-CPU Feasibility | 8 | — | — | 8–9 | **8** |
| Overall Feasibility | 3 | ~2 | ~6 | 3–6 | **5** (MVP) |

**Composite Score: 5 / 10**

### Key Disagreements Resolved

**1. Is the motivating example fatal? (Skeptic vs. Auditor/Synthesizer)**

The Skeptic identified that the paper's own LLVM IR listing shows `add nsw` (C) vs `add` (Rust) — these are *different* instructions, contradicting the "IR erases differences" thesis. **The Skeptic wins this specific argument.** The `nsw` motivating example is self-refuting and must be replaced. However, the underlying problem survives: some semantic gaps (UTF-8 validity, allocation failure, errno vs Result) ARE genuinely invisible at IR level. The thesis must be weakened from "IR erases everything" to "IR is partially blind to cross-language semantic gaps."

**2. Will C2Rust yield zero findings? (Skeptic vs. Synthesizer)**

C2Rust deliberately preserves C semantics via `unsafe` + `wrapping_add`. The Skeptic argues this means zero divergences. The Synthesizer counters: (a) C2Rust doesn't handle all edge cases (setjmp/longjmp, va_args, inline assembly), and (b) LLM translations are the primary goldmine. **Resolution:** Pivot evaluation emphasis to LLM translations (60%) with C2Rust as secondary (40%). The Skeptic is partially right — C2Rust alone is insufficient.

**3. Is 0 LoC a project-killer? (All agree: yes, unless immediate pivot)**

128KB of theory with zero implementation is a red flag all three experts flag independently. The ratio of specification-to-code is infinite. **Resolution:** No more theory writing. Immediate pivot to code. The 52K LoC scope is dead.

**4. CONTINUE vs ABANDON? (Skeptic vs. Auditor/Synthesizer)**

The Skeptic's ABANDON recommendation rests on three pillars: (a) self-refuting example, (b) C2Rust null findings, (c) competitive window closed (RustAssure, SMACK). The Auditor and Synthesizer counter: (a) is survivable with reframing, (b) is mitigated by LLM pivot, (c) the empirical taxonomy is genuinely unclaimed territory. **Resolution: CONTINUE, but the Skeptic's concerns impose mandatory kill gates.**

### Fatal Flaws Identified

| # | Flaw | Severity | Status |
|---|------|----------|--------|
| F1 | Motivating example self-refutes (`nsw` vs non-`nsw` IS visible at IR) | HIGH | Must replace example, weaken thesis |
| F2 | "No tool exists" claim is false (RustAssure, SMACK TACAS 2025) | HIGH | Must rewrite related work |
| F3 | 52K LoC infeasible (0 LoC implemented, 128KB theory) | HIGH | Rescope to ~6K LoC MVP |
| F4 | "Integer reconciliation algebra" is a lookup table | MEDIUM | Drop algebraic framing |
| F5 | C2Rust may yield zero divergences | MEDIUM | Pivot to LLM translations as primary |
| F6 | Evaluation circularity (tool scoped to what C2Rust produces) | MEDIUM | LLM translations break circularity |

**No single flaw is fatal. The aggregate of flaws requires radical rescoping but not abandonment.**

### What Survives (MVP Scope)

**Title:** "Mind the Gap: Empirical Characterization of Semantic Divergences in Automated C-to-Rust Translation"

**Deliverable (~6K LoC, 4 months):**
- Semantic-aware differential fuzzer (AFL++ wrapper + harness generation) — ~3K LoC
- Divergence detector + classifier — ~2K LoC
- Taxonomy database + reporting — ~1K LoC

**What Dies:**
- ❌ 52K LoC full pipeline
- ❌ Product program construction
- ❌ Custom typed SSA IR
- ❌ Bounded symbolic execution engine
- ❌ Standard library stubs (8K LoC)
- ❌ "Integer reconciliation algebra" as primary contribution
- ❌ "LLVM IR erases everything" thesis (replace with nuanced version)
- ❌ PLDI/OOPSLA venue targeting

### Mandatory Gates

| Gate | Deadline | Criterion | If Fail |
|------|----------|-----------|---------|
| G1: First blood | Week 2 | ≥1 real divergence in C2Rust OR LLM translation that plain AFL++ misses | **ABANDON** |
| G2: Alive2 triage | Week 2 | Run Alive2 on 10 pairs. If Alive2 catches ≥4/5 of same divergences | **ABANDON** |
| G3: Taxonomy breadth | Week 6 | ≥3 distinct divergence categories across ≥2 libraries | **ABANDON** |
| G4: Paper draft | Week 10 | Submittable draft with evaluation tables | **ABANDON** |

### Target Venue
- **Primary:** ISSTA 2027 or ASE 2027
- **Fallback:** MSR 2027, ICSE-SEIP 2027
- **Not:** PLDI, POPL, CAV

### Risk Assessment
- Probability of publishable outcome (rescoped MVP): **~65%**
- Probability of best-paper (at ISSTA/ASE): **~10%**
- Probability of completing full 52K LoC system: **~10%**

---

## Verdict: CONTINUE (Phase 1 MVP only, with mandatory kill gates)

### Conditions (binding)
1. **Immediate pivot from theory to code.** Zero additional theory writing until a running fuzzer prototype exists.
2. **Replace the self-refuting motivating example.** Use UTF-8 validity, allocation failure, or errno/Result — not integer overflow.
3. **Rewrite all "no tool exists" claims.** Position as "first systematic empirical characterization" not "first tool."
4. **Drop "algebra" framing.** Call it a semantic divergence catalog.
5. **Enforce all four gates with hard deadlines.** Gate failures → ABANDON.
6. **State team size and realistic timeline.** Solo: 6-8 months. Team of 2: 4 months.

---

## Team Signoff

| Role | Verdict | Key Condition |
|------|---------|---------------|
| Independent Auditor | CONTINUE | Fix prior art claims; 0 LoC is a health emergency |
| Fail-Fast Skeptic | ABANDON → CONTINUE (reluctant) | Only if G1+G2 pass within 2 weeks |
| Scavenging Synthesizer | CONTINUE | Empirical taxonomy paper, not verifier paper |
| **Team Lead (Synthesis)** | **CONTINUE** | **Radical rescope to MVP; all gates enforced** |

---

## Rankings

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 5,
      "verdict": "CONTINUE",
      "reason": "Real problem with a viable MVP path (semantic-aware fuzzer + divergence taxonomy, ~6K LoC). Core thesis overstated but survivable with reframing. 52K LoC full system is dead — rescope to empirical taxonomy paper targeting ISSTA/ASE. Three fatal-flaw-level concerns (self-refuting example, false novelty claims, zero implementation) are all fixable with radical rescoping. Mandatory kill gates at weeks 2, 6, and 10 enforce discipline. Probability of publishable outcome: ~65%.",
      "scavenge_from": []
    }
  ]
}
```
