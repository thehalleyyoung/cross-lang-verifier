# Final Approach: Scoped Source-Level Semantic Verification with Fuzzing Foundation

## Synthesis Rationale

After adversarial debate across five expert roles, the winning approach takes Approach A's architecture and core insight (source-level semantic analysis, product programs, SMT verification) but incorporates three critical adjustments from the critique:

1. **Scope the C frontend ruthlessly** to C2Rust output (a predictable, well-defined C subset), eliminating the fractal C-standard-compliance tarpit identified by the Difficulty Assessor.
2. **Build the differential fuzzer first** (Approach C's strength) as an independently useful milestone and fallback, so the project delivers value even if symbolic verification takes longer than expected.
3. **Strip ornamental math to the bone** (per the Math Skeptic): the contribution is a semantic divergence table + per-divergence SMT encoding, not an "algebra." The product program construction is real engineering, honestly described.

Approach B is eliminated: it is strictly dominated by A (same goal, worse execution due to the annotation survival problem) and by C (both approximate, C is simpler).

---

## The Approach

### Architecture: Two-Phase Verification Pipeline

**Phase 1 (Months 1-4): Differential Fuzzing with Semantic Awareness**

Build a type-aware differential fuzzing engine that:
1. Parses C (via libclang) and Rust (via syn) to extract function signatures, type information, and semantic annotations (overflow modes, error handling patterns).
2. Generates an FFI test harness that calls both functions with identical inputs.
3. Runs coverage-guided differential fuzzing (libFuzzer/AFL++) with semantically-informed seed generation: boundary values at integer overflow points, type width boundaries (127/128 for i8, 32767/32768 for i16, etc.), and float precision edges.
4. Reports concrete divergences with minimal reproducing inputs.

This phase produces an independently useful tool (~8-12K LoC) that finds real bugs immediately, validating the problem's importance.

**Phase 2 (Months 4-10): Bounded Symbolic Verification**

Layer bounded symbolic verification on top of the fuzzing foundation:
1. Lower both programs into a shared typed SSA IR parameterized by semantic configuration σ = (overflow_mode, float_model, error_model).
2. Construct product programs with coercion point insertion at semantic divergence sites.
3. Discharge verification conditions via Z3 (QF_BV for integers, QF_FP for floats, QF_ABV for memory), with 10-second per-query timeouts.
4. Route timed-out paths to Phase 1's fuzzing engine, seeded by the symbolic engine's partial path exploration.

This phase adds bounded equivalence proofs (~25-35K LoC additional), upgrading the tool from "bug finder" to "verifier."

### Scoping Decisions

**C subset: C2Rust output only.** C2Rust emits a predictable subset of C: no VLAs, no computed gotos, no `_Generic`, no bitfields (mostly), predictable integer types, explicit casts. This reduces the C frontend from "implement the C11 standard" to "handle C2Rust's specific output patterns." The frontend targets this subset explicitly and rejects non-C2Rust C with a clear error.

**Rust subset: unsafe Rust with explicit types.** C2Rust output is saturated with `unsafe` blocks and explicit types (no trait resolution, no generics, no iterators, no closures). This is the easiest Rust to analyze. LLM translations may use more idiomatic Rust, handled on a best-effort basis with "structural mismatch" verdicts for unsupported patterns.

**Structural similarity required.** Function boundaries, parameter types, and control flow structure must be preserved between source and target. This covers C2Rust output and most LLM translations but explicitly excludes refactored/restructured translations. Structurally dissimilar pairs are rejected with a clear verdict.

### Core Technical Components

#### 1. Semantic Divergence Table (Load-Bearing)

A complete, empirically validated enumeration of where C and Rust semantics differ:

| Operation | C Semantics | Rust Semantics | Divergence Type |
|-----------|-------------|----------------|-----------------|
| Signed integer overflow | Undefined behavior | Wrap (release) / Panic (debug) | Critical |
| Integer promotion | Implicit promotion to `int` | No auto-promotion | Moderate |
| Right-shift of negatives | Implementation-defined | Arithmetic shift (defined) | Moderate |
| Division by zero | Undefined behavior | Panic | Critical |
| Float-to-int out of range | Undefined behavior | Saturating cast (since 1.45) | Critical |
| Null pointer dereference | Undefined behavior | Compile error (safe) / UB (unsafe) | Critical |
| Array out-of-bounds | Undefined behavior | Panic (safe) / UB (unsafe) | Critical |
| Unsigned wrapping | Defined (wraps) | Defined (wraps) | None |
| Pointer arithmetic | Arbitrary | Constrained by provenance | Moderate |

Each row has a corresponding SMT encoding function and a fuzzing seed generation strategy. This table is validated against compiler behavior on a corpus of ~200 edge-case functions.

#### 2. Shared Typed SSA IR (~12K LoC)

~40 instruction types. Type system: fixed-width integers (i8–i128), IEEE 754 floats, pointers with provenance tags, tagged unions. The IR's operational semantics are parameterized by σ, so the same IR program is interpreted differently under C vs. Rust semantics. Key design principle: every semantic divergence from the table above is represented as an explicit IR operation (e.g., `IntAdd(a, b, overflow_mode)`) rather than erased.

#### 3. Language Frontends (~10-12K LoC)

**C frontend (libclang → IR):** Scoped to C2Rust output patterns. Uses libclang for parsing and type resolution. Extracts actual integer promotion decisions from libclang's type APIs rather than reimplementing from the C11 standard. Differential-tested against Clang/GCC on ~500 edge cases.

**Rust frontend (syn → IR):** Scoped to unsafe C2Rust-style Rust. Uses syn for parsing with type extraction from explicit annotations. No trait resolution or monomorphization needed (C2Rust output doesn't use generics). Preserves overflow mode annotations (wrapping vs. checked vs. panicking).

#### 4. Product Program Construction (~5K LoC)

Structural alignment via greedy LCS on basic block signatures. Coercion point insertion at every operation in the semantic divergence table. Structurally dissimilar pairs rejected with explicit "structural mismatch" verdict including a diff showing where alignment failed.

#### 5. Bounded Symbolic Execution + SMT (~8K LoC)

Loop unrolling to k=32. Path exploration with DFS and state merging at join points. Z3 encoding: QF_BV for integers, QF_FP for floats, QF_ABV for memory. Per-query timeout: 10 seconds. Timed-out paths routed to fuzzing.

#### 6. Differential Fuzzing Engine (~5K LoC)

Coverage-guided differential fuzzing wrapping libFuzzer/AFL++. Seeded by: (a) semantic divergence table boundary values, (b) symbolic engine's partial path exploration, (c) standard fuzzing heuristics. Input minimization for counterexamples.

#### 7. Standard Library Stubs (~6K LoC)

Models for ~30 high-frequency C↔Rust stdlib function pairs: malloc/free ↔ Box/Vec, strlen/strcmp ↔ str methods, memcpy/memset ↔ slice operations. Scoped to functions that actually appear in C2Rust output of libsodium and zlib.

#### 8. CLI + Evaluation (~4K LoC)

CLI accepting two source files, producing structured JSON verdict: equivalence determination, concrete counterexample, or "unknown" with coverage report. Evaluation framework with benchmark management and baseline comparison.

### Total Estimated LoC

| Component | LoC | Risk |
|-----------|-----|------|
| Semantic divergence table + encoding | 1K | Low |
| Shared typed SSA IR | 12K | Medium |
| C frontend (C2Rust subset) | 5-6K | Medium (scoped) |
| Rust frontend (unsafe subset) | 5-6K | Medium (scoped) |
| Product program construction | 5K | Medium |
| Symbolic execution + SMT | 8K | High |
| Differential fuzzing engine | 5K | Low |
| Standard library stubs | 6K | Low (tedious) |
| CLI + evaluation | 4K | Low |
| **Total** | **~52-55K** | |

Realistic estimate accounting for iteration: **60-70K LoC.** The LoC is similar to Approach A, but the scoping to C2Rust output and the phased development strategy reduce risk dramatically.

---

## Why This Is Genuinely Difficult

1. **Even scoped C frontend fidelity.** C2Rust output is better-behaved than arbitrary C, but still contains implicit promotions, pointer arithmetic, and struct operations that must be faithfully encoded. Each edge case in the C-to-IR lowering is a potential silent false result.

2. **Product program alignment.** C2Rust introduces extra let-bindings, temporaries, explicit casts, and `unsafe` blocks that don't exist in the C source. The alignment heuristic must normalize these differences while preserving semantically significant ones.

3. **SMT scalability for pointer-heavy code.** Pointer provenance (C's arithmetic vs. Rust's borrowing) requires array theory. The 10-second timeout means the tool will route most pointer-heavy functions to fuzzing, honestly reporting the coverage limitation.

4. **Irreducible system complexity.** Each component exists because removing it degrades the tool: without the IR, you can't compare semantics; without frontends, you can't populate the IR; without product programs, you can't relate two programs; without SMT, you can't prove equivalence; without fuzzing, intractable paths get no verdict.

---

## Load-Bearing Math (Stripped to Essentials)

### 1. Semantic Divergence Enumeration and Per-Operation Encoding

A complete table (not an algebra) of where C and Rust semantics differ, with per-operation SMT encoding. For each binary operation ⊕ and each (c_mode, rust_mode) pair:

```
encode(⊕, σ_C, σ_Rust, x, y):
  let result_c = eval(⊕, σ_C, x, y)
  let result_rust = eval(⊕, σ_Rust, x, y)
  let well_defined_c = ¬triggers_ub(⊕, σ_C, x, y)
  emit:
    if well_defined_c:
      assert(result_c == result_rust)
    else:
      report_divergence(⊕, x, y, "C UB vs Rust defined")
```

This is ~50 lines of Z3 encoding per divergence class, covering signed overflow (4 mode pairs), unsigned operations (identical—no encoding needed), float precision (2 models), division by zero (2 modes), and array bounds (2 modes). **This is the formal core of the tool.** Everything else is engineering to reach this point.

### 2. Product Program Soundness (Construction-Level)

The product program P× for functions f_C, f_Rust shares symbolic inputs and inserts coercion points per the divergence table. The soundness argument: if all coercion assertions in P× hold for all inputs within loop bound k, then f_C and f_Rust agree on all well-defined inputs within that bound. This is true by construction (coercion assertions directly encode the equivalence condition) and does not require a separate proof.

**Completeness gap (honestly stated):** The coercion insertion is complete only with respect to the enumerated divergence table. If the table misses a divergence class (e.g., a platform-specific ABI difference), the tool will miss it silently. The table is validated empirically, not proved complete.

### 3. Three-Output Specification (Not a Theorem)

- **Equivalent (bounded):** All paths within loop bound k verified by SMT. Sound modulo: frontend correctness, loop bound, Z3 correctness.
- **Divergent:** Concrete counterexample. True positive by construction (both programs are executed on the witness).
- **Unknown:** Some paths timed out or exceeded loop bound. Fuzzing explored N inputs with K% branch coverage. Honestly quantified.

---

## Best-Paper Argument

A program committee would select this paper for four reasons:

1. **The LLVM-IR-erasure thesis is a crisp, demonstrable negative result.** We show concrete function pairs where LLREVE (IR-level) reports "equivalent" and our tool reports "divergent" with a concrete witness. This is surprising to the PL community and changes how people think about cross-language verification.

2. **The divergence taxonomy is independently valuable.** Running the tool exhaustively on C2Rust(libsodium) and C2Rust(zlib) produces the first systematic characterization of semantic divergences in production transpiler output. This taxonomy (N% signed overflow, M% float precision, K% error handling) is useful regardless of the tool.

3. **The tool is end-to-end and evaluated on real code.** Not cherry-picked benchmarks—ALL exportable functions from two production libraries. Plus LLM translation benchmarks. With baselines (LLREVE, fuzzing-only, LLM review).

4. **Urgent practical demand.** DoD mandate, CISA advisories, Google's memory safety data. The paper solves a problem people are spending millions of dollars on today.

**Target venues:** PLDI or OOPSLA (primary), CAV (secondary). The paper leads with the empirical results and the erasure thesis, presents the technical approach as the mechanism for obtaining them, and frames the math as the engineering foundation rather than the primary contribution.

---

## Hardest Technical Challenge

**The C frontend, even scoped to C2Rust output.** C2Rust emits code that is syntactically valid C but uses C's full integer promotion and conversion rules. A single missed implicit conversion (e.g., `unsigned short + int` promoting to `int` vs. `unsigned int` depending on platform widths) produces a silent false equivalence report.

**Mitigation strategy (multi-layered):**
1. **Use libclang's semantic APIs, not syntactic pattern matching.** Extract the compiler's actual type decisions for each expression via `clang_getCursorType()`, `clang_getCanonicalType()`, and `clang_Type_getSizeOf()`.
2. **Build a C2Rust output corpus.** Run C2Rust on libsodium, zlib, and 3 other libraries. Catalog the exact C patterns emitted. Scope the frontend to handle only these patterns.
3. **Differential testing against the compiler.** For each frontend test case, compile with Clang at `-O0 -g`, extract operation widths from DWARF debug info, and verify the IR matches.
4. **Fail loudly on unrecognized patterns.** If the frontend encounters a C construct it doesn't handle, it emits an "unsupported construct" error rather than silently misencoding.

---

## Evaluation Plan

### Benchmarks
- **C2Rust(libsodium):** ~300 exportable functions, exhaustive analysis
- **C2Rust(zlib):** ~100 exportable functions, exhaustive analysis
- **LLM translations:** GPT-4 and Claude translations of 100+ algorithmic functions from C to Rust
- **Synthetic stress tests:** Parameterized function pairs with controlled semantic gaps for ablation

### Metrics
- Equivalence rate within 60-second budget per function
- Divergences found (validated by execution)
- Unknown rate with branch coverage statistics
- Time-to-verdict (wall-clock seconds)
- False positive rate (should be zero by construction)

### Baselines
- **LLREVE on LLVM IR:** Proves the LLVM-IR-erasure thesis empirically
- **Differential fuzzing alone (AFL++):** Measures incremental value of symbolic verification
- **LLM review (GPT-4):** Demonstrates LLMs cannot replace formal analysis

### Ablation Studies
- Without semantic bridge (disable coercion insertion)
- Without fuzzing fallback (disable differential fuzzing)
- Per-divergence-class contribution (disable integer/float/error reconciliation individually)

### Divergence Taxonomy
Classify each divergence by root cause: signed overflow, unsigned width mismatch, float precision, error handling, memory safety, other. This taxonomy is an independently valuable empirical contribution.

---

## Timeline

| Phase | Months | Deliverable |
|-------|--------|-------------|
| Phase 1: Differential fuzzer | 1-4 | Working bug-finder with semantic seed generation |
| Phase 2a: IR + frontends | 3-7 | Shared IR with C2Rust-scoped frontends |
| Phase 2b: Product programs + SMT | 6-10 | Bounded verification pipeline |
| Phase 3: Evaluation | 9-12 | Full benchmark suite, paper draft |

**Risk mitigation:** Phase 1 produces a useful tool regardless of Phase 2's outcome. If symbolic verification proves infeasible on the target timeline, the paper can lead with the fuzzing tool + divergence taxonomy (FSE/ISSTA-caliber) while continuing Phase 2 for a follow-up paper.

---

## Feasibility Assessment

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 9 | Urgent need, no existing solution, DoD mandate |
| **Difficulty** | 8 | Scoped C frontend reduces from 9; phased strategy manages risk |
| **Potential** | 9 | LLVM-IR-erasure thesis + divergence taxonomy + working tool |
| **Feasibility** | 6 | Phased approach adds 1 point over Approach A; scoped C subset adds another |
