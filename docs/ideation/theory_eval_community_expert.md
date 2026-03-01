# Community Expert Evaluation: Proposal proposal_00

## Semantic Reconciliation for C↔Rust Equivalence Verification

**Evaluation method:** Claude Code Agent Teams — three independent expert roles (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique and synthesis.

---

## Team Process Summary

Three expert agents independently evaluated the proposal, then challenged each other's findings in an adversarial cross-critique round. The Synthesizer's probability-weighted portfolio analysis emerged as the most rigorous framing, correcting the Auditor's optimistic scores and the Skeptic's misleading "0 fatal flaws" framing.

**Key disagreements resolved:**
- Auditor scored Value 7/10 assuming the full pipeline ships; Synthesizer demonstrated the probability-weighted expected value is ~5/10
- Skeptic found 0 fatal flaws but 5 serious concerns; Auditor correctly noted these compose multiplicatively to ~15% full-pipeline success
- All three agreed the fuzzer + divergence taxonomy is the most probable deliverable and should be treated as the primary target

---

## Pillar Scores

### 1) Extreme Value: 5/10

The C-to-Rust migration problem is real and timely (DoD mandate, CISA advisories, Google memory safety data). No tool currently verifies C↔Rust semantic equivalence at the source level — a genuine gap. However, this is a niche problem with moderate community excitement. Academic PL researchers see it as "someone should work on this" rather than "the most important open problem." The most probable deliverable (differential fuzzer + divergence taxonomy, ~55-80% probability) is a solid ISSTA/ASE contribution but not exceptional. The full verification pipeline (Value 7-8 if delivered) has only ~15% delivery probability.

**Probability-weighted calculation:** 15% × 8 + 40% × 5 + 35% × 4 + 10% × 0 ≈ 4.6, rounded to **5**.

The strongest framing is "verifying AI-generated code translations" (capturing LLM zeitgeist), not pure C↔Rust migration verification.

### 2) Genuine Software Difficulty: 8/10

All three experts agree: this is a genuinely hard artifact. Two language frontends (even scoped to C2Rust output), a novel parameterized IR with 40 instruction types, product program construction with structural alignment, bounded symbolic execution with Z3, and differential fuzzing fallback — this is irreducible complexity. The C frontend alone (faithfully encoding C's integer promotion rules, pointer arithmetic, and UB semantics via libclang) is a substantial engineering challenge. The 52K LoC estimate is honest but likely underestimates by 30-40%. Comparable tools (KLEE ~55K LoC/3 years, Alive2 ~25K LoC/2 years, Prusti ~50K LoC/3 years) took larger teams longer.

### 3) Best-Paper Potential: 5/10

The LLVM-IR-erasure thesis is crisp and demonstrable but may not surprise the FM community — "IR erases source semantics" is their working assumption. Product program construction with coercion insertion is a well-studied technique (Barthe et al. POPL 2011) applied to a new domain — incremental novelty. The integer reconciliation algebra is standard bitvector theory with a mode-indexed lookup table, not genuinely novel math (the proposal is self-aware about this).

**What could push this higher:** If the evaluation reveals practically significant divergences in C2Rust output — cases where the Rust version produces *incorrect output* on specific inputs that escape testing — the paper becomes compelling. If most divergences are specification-level pedantry (C says "UB" but both compile identically), the paper loses impact.

**Probability-weighted:** Full pipeline best-paper potential is 6/10 at ~15% probability; fuzzer-only paper is 3/10 at ~55% probability. Expected best-paper value ≈ **5/10**.

### 4) Laptop-CPU Feasibility & No-Humans: 9/10

No dispute from any expert. Z3 is CPU-native; QF_BV queries resolve in <2 seconds median. Differential fuzzing runs at thousands of executions/second on a single core. Function-level modularity enables trivial parallelism (400 independent tasks across 8 cores). No GPU, no cloud, no human annotation required. Estimated full evaluation runtime 2-4 hours on an 8-core laptop. QF_FP (float) queries are 10-100× slower but affect a minority of functions in the target benchmarks (libsodium/zlib are integer-heavy).

### 5) Feasibility: 4/10

The full pipeline (52K LoC, 10-12 months) is historically aggressive for verification infrastructure. The Skeptic's comparison to KLEE (55K LoC, 3+ years, team of 3-4), Alive2 (25K LoC, 2+ years, team of 2-3), and Prusti (50K LoC, 3+ years, team of 3-4) suggests this is an 18-24 month project for a team, or a 2-3 year project for a solo developer. Five serious concerns compose multiplicatively:

1. C frontend LoC underestimate (50-80% more than claimed)
2. Timeline optimistic by ~50% (Phase 2 critical path is serial)
3. Unknown verdict rate likely 25-40% (weakening verifier framing)
4. Unique-bug-finding claim unvalidated (baseline experiment needed)
5. Product program alignment fragile on real C2Rust output

Phase 1 (differential fuzzer, 4 months) is feasible and independently valuable. Phase 2 (symbolic verification, 6 months allocated but ~12 months needed) is a coin flip.

**Feasibility of most-probable deliverable (fuzzer + taxonomy): 7/10.**

### 6) Fatal Flaws

**No single fatal flaw identified by any expert.** However, the aggregate of five serious concerns yields a ~15% probability of full pipeline delivery. The most important risk:

- **The erasure thesis, while technically correct (~70%), may be oversold.** Clang emits `add nsw` which does encode UB information; Alive2 models poison semantics. The real gap is "wrong relation (refinement ≠ behavioral equivalence) + partial erasure," not "total erasure." Reframing is straightforward but essential.

- **Most C2Rust divergences may be trivial/unexploitable.** If C2Rust output is primarily consumed as a starting point for incremental rewriting, divergences in the transpiler output are known limitations, not bugs. The paper needs concrete examples where divergences cause *real behavioral differences* on concrete inputs.

- **The full pipeline has a ~15% success probability.** This is not fatal because the fallback (fuzzer + taxonomy) has ~80% success probability and is independently publishable. But it means the project as described in the proposal is more likely to partially succeed than fully succeed.

---

## VERDICT: CONTINUE

**With conditions:**

1. **Invert the risk profile.** Treat the differential fuzzer + divergence taxonomy as the primary deliverable (ISSTA/ASE target), not a stepping stone. Symbolic verification is a stretch goal for a follow-up paper.

2. **Run the kill-criterion experiment immediately.** Build a minimal differential fuzzer prototype and test it on 10 C2Rust function pairs from libsodium within 4 weeks. If it doesn't find real behavioral divergences (not just specification-level differences), abandon.

3. **Run the Alive2 baseline experiment early.** Before investing 10 months, set up Alive2 on C2Rust function pairs and verify the erasure thesis holds empirically. If Alive2 catches >50% of divergences, reframe the contribution.

4. **Budget 18 months, not 12.** The full system is a multi-year effort being compressed. Phase 1 paper at month 6-8, full system paper at month 16-20.

5. **Frame around LLM translation verification.** "Automated verification of AI-generated code translations" captures current community excitement better than "C↔Rust migration verification."

6. **Add a "simple linter" baseline.** Show that a 500-line pattern-matching linter is insufficient — either too many false positives or misses path-sensitive divergences. This preempts the reviewer objection "why not just grep for `wrapping_add`?"

---

## Scoring Summary

| Dimension | Score |
|-----------|-------|
| Extreme Value | **5/10** |
| Genuine Software Difficulty | **8/10** |
| Best-Paper Potential | **5/10** |
| Laptop-CPU Feasibility & No-Humans | **9/10** |
| Feasibility | **4/10** |
| Fatal Flaws | None individually fatal; aggregate risk high |
| **VERDICT** | **CONTINUE** (with risk inversion) |

---

## Most-Probable Deliverable Scores (Fuzzer + Taxonomy)

| Dimension | Score |
|-----------|-------|
| Value | 5/10 |
| Difficulty | 5/10 |
| Best-Paper Potential | 3/10 |
| Laptop-CPU Feasibility | 9/10 |
| Feasibility | 7/10 |

---

## Expert Consensus Notes

**Independent Auditor:** "The proposal is well-crafted and self-aware. The phased architecture is the strongest feature — it fails gracefully. But the full pipeline is a 2-3 person, 18-month project being pitched as a solo 10-month project. Target FSE/ISSTA first."

**Fail-Fast Skeptic:** "Zero fatal flaws, five serious concerns. The proposers are unusually honest about their risks, which is itself a good sign. But honesty about risk doesn't reduce risk. Run the Alive2 baseline experiment before committing. Lead with the taxonomy, not the verifier."

**Scavenging Synthesizer:** "The project's floor is a publishable empirical study (near-certain). Its likely outcome is a strong fuzzing/testing paper (80%). Its ceiling is a flagship PLDI/OOPSLA paper (15%). The risk profile should be inverted: build the fuzzer as the primary deliverable, not a stepping stone. This transforms a 15% moonshot into an 80% solid bet with upside potential."

**Cross-critique winner:** Synthesizer — probability-weighted portfolio analysis was the most rigorous framing. The project is worth starting, but only with realistic expectations about the most probable outcome.
