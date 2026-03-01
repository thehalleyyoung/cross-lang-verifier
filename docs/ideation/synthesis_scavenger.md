# Scavenging Synthesizer: Final Verdict

## Proposal: "Semantic Reconciliation for C↔Rust Equivalence Verification"

---

## 1. MINIMUM VIABLE CONTRIBUTION (Independently Publishable)

**The Divergence Taxonomy Paper.** Not the verifier. Not the fuzzer. The *taxonomy*.

The strongest version of this project is an empirical study titled something like:

> **"Where C and Rust Disagree: A Systematic Taxonomy of Semantic Divergences in Automated C-to-Rust Translations"**

The MVP contribution is: run C2Rust and 2-3 LLMs on libsodium + zlib, build a lightweight differential testing harness (~3-5K LoC, not 52K), and produce the first exhaustive characterization of where translations actually diverge on real inputs. Classify every divergence by root cause (signed overflow, float precision, error handling, integer promotion, etc.), severity (silent wrong answer vs. crash vs. specification-level pedantry), and frequency.

This is publishable at **ISSTA, ASE, or ICSE-SEIP** with zero symbolic verification, zero product programs, zero SMT. The taxonomy is the golden core that every evaluator identified but none elevated to primary deliverable status.

**Why this is independently publishable:**
- No one has done exhaustive divergence characterization on production C2Rust output
- The LLM translation angle is timely and underexplored
- Concrete data ("43% of libsodium functions contain at least one signed-overflow divergence site") is citable infrastructure for the entire C↔Rust migration community
- It validates or invalidates the premise of every future tool in this space

---

## 2. ELEMENTS TO SALVAGE FOR THE MVP

### SALVAGE (load-bearing, keep):

| Element | Why | Approximate LoC |
|---------|-----|-----------------|
| **Semantic divergence table** | The actual intellectual contribution. A complete, validated enumeration of C↔Rust semantic gaps with per-operation test strategies. This IS the paper. | ~500 (data, not code) |
| **Angelic UB modeling decision** | Necessary for correct divergence classification. Without it, every C UB site is a "divergence" and the taxonomy is noise. The angelic framing ("does ANY conforming C behavior agree with Rust?") is what separates meaningful from trivial divergences. | ~200 lines of logic |
| **Directed ε-approximation for floats** | The asymmetric tolerance with error accumulation is genuinely needed for float-heavy functions. Without it, float divergences are all false positives or all false negatives. One paragraph of math, but load-bearing. | ~300 lines |
| **FFI harness generation** | Parse C (libclang) + Rust (syn) signatures, generate shared-input test harness, run both, compare outputs. This is the minimum viable tool. | ~2-3K LoC |
| **Boundary-value seed generation** | Seeds derived from the divergence table: INT_MAX±1, type-width boundaries, denormalized floats, division-by-zero, null pointers. This is what makes the fuzzer "semantic-aware" vs. random. | ~500 LoC |
| **Coverage-guided differential fuzzing** | Wrap libFuzzer/AFL++ for completeness beyond boundary seeds. Well-understood, low-risk. | ~1-2K LoC |
| **Concrete witness validation** | Compile both, execute on witness, confirm divergence is real. This is "belts and suspenders" and eliminates false positives from the taxonomy. | ~500 LoC |
| **The IR erasure demonstration** | 3-5 concrete examples where Alive2/LLREVE report "equivalent" and you show a divergent input. This is a figure in the paper, not a system component. Requires ~1 day of manual construction. | ~0 (manual examples) |

### SALVAGE PARTIALLY (useful scaffolding, defer full build):

| Element | What to Keep | What to Defer |
|---------|-------------|---------------|
| **C frontend** | Signature extraction + type info via libclang (~1K LoC) | Full AST-to-IR lowering (~5K LoC) |
| **Rust frontend** | Signature extraction via syn (~800 LoC) | Full AST-to-IR lowering (~5K LoC) |
| **Shared IR** | Type system definitions for the divergence table (~500 LoC) | Full 40-instruction SSA IR (~12K LoC) |
| **Three-output verdicts** | Divergent (concrete witness) + Unknown (coverage stats). Drop "Equivalent" — that requires the symbolic engine. | Bounded equivalence proofs |

**Total salvaged MVP: ~6-8K LoC.** This is a 3-4 month project for one person.

---

## 3. KILL IMMEDIATELY

| Element | Why Kill | Savings |
|---------|----------|---------|
| **Full SSA IR with 40 instruction types** | You don't need a custom IR to fuzz. Parse signatures, generate harness, fuzz. The IR is Phase 2 infrastructure with no Phase 1 payoff. | ~12K LoC |
| **Product program construction** | Requires the full IR, structural alignment, coercion insertion. This is the hardest 5K LoC in the system and provides zero value without the symbolic engine. | ~5K LoC |
| **Bounded symbolic execution engine** | KLEE-scale infrastructure. Multiple PhD theses. Zero probability of working correctly in 4 months. | ~8K LoC |
| **Standard library stubs** | 50 function models is a project unto itself. The fuzzer doesn't need them — it calls the real functions. | ~8K LoC |
| **"Integer reconciliation algebra" framing** | All three evaluators agree: it's a lookup table dressed in algebraic language. Kill the language, keep the table. No reviewer will accept "algebra" for a finite case analysis with no algebraic properties. | 0 LoC (framing only) |
| **"Novel thesis" framing for IR erasure** | It's not novel — it's well-known. Kill "we prove the erasure thesis." Replace with "we quantify the erasure gap empirically for the first time." | 0 LoC (framing only) |
| **QF_FP SMT encoding** | Z3's float solver is 10-100× slower than BV. For the MVP, fuzz floats — don't symbolically verify them. Defer to Phase 2 or never. | ~2K LoC |
| **PLDI/OOPSLA venue targeting** | The MVP is not a PLDI paper. Pretending otherwise distorts the project's shape. Kill the aspiration for Paper 1. | 0 LoC (expectations) |

**Total killed: ~35K LoC, 6+ months of work, and several delusions.**

---

## 4. BEST POSSIBLE FRAMING FOR MAXIMUM IMPACT

### The Winning Title:

> **"Mind the Gap: Empirical Characterization of Semantic Divergences in Automated C-to-Rust Translation"**

### The Winning Pitch (3 sentences):

Automated C-to-Rust translation tools (C2Rust, GPT-4, Claude) produce code that compiles and passes tests but silently disagrees with the original on edge-case inputs where C and Rust have different semantics. We present the first systematic empirical study of these divergences, analyzing all exportable functions in C2Rust translations of libsodium and zlib plus 200+ LLM-generated translations, producing a taxonomy of 6 divergence classes with frequency, severity, and concrete triggering inputs. Our semantically-aware differential testing tool, guided by a comprehensive catalog of C↔Rust semantic gaps, finds divergences that random fuzzing misses and that IR-level equivalence checkers cannot detect by construction.

### Why This Framing Wins:

1. **Empirical-first.** The taxonomy is the contribution. The tool is the method. This is how ISSTA/FSE papers are structured.

2. **LLM angle captures zeitgeist.** "Verifying AI-generated code" is the hottest topic in SE. C↔Rust migration alone is niche; LLM translation verification is mainstream.

3. **Negative result is powerful.** "IR-level checkers miss 40% of real divergences" (or whatever the number is) is a citable finding that reshapes how the community thinks about translation verification.

4. **Concrete data > abstract theorems.** "37% of libsodium functions contain signed-overflow divergence sites" lands harder than "Theorem 2.1: Product program soundness."

5. **Tool is a means, not the end.** Reviewers evaluate whether the findings are interesting, not whether the tool is a contribution to verification theory. Lower bar, higher probability of acceptance.

### Venue Strategy:

| Priority | Venue | Fit | Why |
|----------|-------|-----|-----|
| 1 | **ISSTA 2025/2026** | ★★★★★ | Testing tool + empirical study. Core audience. |
| 2 | **FSE 2025/2026** | ★★★★☆ | SE tool + taxonomy. Broader audience. |
| 3 | **ASE 2025** | ★★★★☆ | Automated SE. Good fallback. |
| 4 | **ICSE-SEIP** | ★★★☆☆ | Industry practice track. If taxonomy is the main result. |
| 5 | **OOPSLA** (only with Phase 2) | ★★☆☆☆ | Requires symbolic verification working. Stretch. |

---

## 5. SCORES: SALVAGED VERSION

| Dimension | Score | Justification |
|-----------|:-----:|---------------|
| **Value** | **7/10** | Real problem, timely (DoD + LLM), first systematic taxonomy, immediately useful data for the migration community. Not 8+ because the niche audience limits citation impact. |
| **Difficulty** | **5/10** | ~6-8K LoC of well-understood fuzzing engineering + careful empirical methodology. Hard enough to be a real project, not so hard it fails. The intellectual difficulty is in the divergence table completeness and the angelic-UB classification, not in the code. |
| **Best-Paper** | **5/10** | Plausible at ISSTA if the numbers are striking ("IR-level tools miss X% of divergences", "LLM translations have 3× more divergences than C2Rust"). Requires genuinely surprising data — which we can't predict in advance. The framing is strong but the ceiling is capped by the absence of formal verification. |

**Composite: 5.7/10 — but with ~75% delivery probability, the risk-adjusted value is ~4.3. Compare to the full system at 6.5/10 composite but ~15% delivery, risk-adjusted ~1.0. The salvaged version is 4× better in expected value.**

---

## 6. VERDICT: **CONTINUE** — Exact Scope Below

### The Scope I'd Recommend:

**Phase 0 (Weeks 1-2): Kill Criterion**
- Manually construct 5 C↔Rust function pairs with known semantic divergences (signed overflow, float cast, division by zero, array bounds, integer promotion).
- Run Alive2 and/or LLREVE on the LLVM IR. Confirm they report "equivalent" or "unknown" on ≥3/5.
- If IR-level tools catch ≥4/5, the source-level thesis is dead. **ABANDON.**
- Run C2Rust on tiny-AES-c (small, integer-heavy). Check if divergences actually exist.
- If C2Rust output produces zero divergences on 20 function pairs, pivot to LLM-only evaluation.

**Phase 1 (Months 1-4): Build the Taxonomy Tool**
- FFI harness generator from libclang + syn signatures (~2K LoC)
- Semantic-aware seed generator from divergence table (~500 LoC)
- Coverage-guided differential fuzzing wrapper (~1.5K LoC)
- Witness validation + minimization (~500 LoC)
- Divergence classifier (overflow/float/error/promotion/bounds/other) (~500 LoC)
- CLI + JSON output (~500 LoC)
- Total: ~5-6K LoC

**Phase 2 (Months 3-5, overlapping): Run the Evaluation**
- C2Rust on libsodium (all ~300 exportable functions)
- C2Rust on zlib (all ~100 exportable functions)
- GPT-4 translations of 100 algorithmic functions
- Claude translations of 100 algorithmic functions
- Baseline: random fuzzing (no semantic seeds) on same targets
- Baseline: Alive2 on LLVM IR of same targets
- Baseline: "just grep for `wrapping_add`" linter (preempts reviewer objection)
- Produce the taxonomy with frequency/severity/trigger data

**Phase 3 (Month 5-6): Write the Paper**
- Lead with taxonomy results (Section 1 + 2)
- Tool as methodology (Section 3)
- IR erasure demonstration as negative result (Section 4)
- Related work (RustAssure, SMACK, Kani, Alive2 — be generous, position honestly)
- Submit to ISSTA

**Phase 4 (Months 7+, conditional): Symbolic Verification**
- Only if Paper 1 is accepted or receives encouraging reviews
- Build the shared IR, product programs, SMT encoding
- This becomes Paper 2 at OOPSLA/PLDI
- Budget 12-18 additional months

### What Success Looks Like:

- **Minimum success (70% probability):** Taxonomy paper with 50+ divergences found across libsodium + zlib + LLM translations. Published at ISSTA/ASE. Cited by future C↔Rust migration work.
- **Good success (40% probability):** Taxonomy paper with genuinely surprising statistics. Demonstrates IR erasure gap empirically. Published at FSE/ISSTA with strong reviews. Tool released as open-source artifact.
- **Great success (15% probability):** All of above, plus Phase 2 delivers working symbolic verification. Second paper at OOPSLA. Two publications from one project.

### What Failure Looks Like:

- **Soft failure (20% probability):** Divergences exist but are all trivial (specification pedantry, never triggered in practice). Paper is publishable but unexciting. Workshop-level.
- **Hard failure (10% probability):** C2Rust output genuinely preserves semantics (few/no divergences), LLM translations are too structurally different to analyze, and IR-level tools catch everything source-level tools catch. Premise is wrong. **ABANDON.**

---

## Summary: The Golden Core

The 52K LoC verifier is a castle in the sky. The golden core buried inside it is:

1. **The divergence table** — a complete, validated catalog of where C and Rust semantics diverge, with per-operation test strategies
2. **The taxonomy** — first empirical characterization of these divergences in real transpiler/LLM output
3. **The IR erasure demonstration** — concrete evidence that IR-level tools miss source-semantic divergences
4. **The angelic UB modeling decision** — the one piece of genuine formalism that makes the tool correct

Everything else is either scaffolding for these four things, or premature optimization for a verification system that probably shouldn't be built yet.

Build the minimum tool that produces the maximum data. The data is the contribution.
