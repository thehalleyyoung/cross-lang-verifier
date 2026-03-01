# Three Competing Approaches: C↔Rust Equivalence Verification

## Approach A: Source-Level Semantic Product Programs

### Core Idea

Build a custom typed SSA intermediate representation from source code (via `libclang` for C, `syn` for Rust) that preserves the exact semantic distinctions LLVM IR erases. The IR is parameterized by a *semantic configuration* σ = (overflow_mode, float_model, error_model) so the same instruction set interprets differently under C vs. Rust semantics. Given two IR programs, construct a *product program* that shares symbolic inputs and interleaves execution of both, inserting *semantic coercion points* at every operation where C and Rust semantics diverge (signed arithmetic, float precision, error handling). The product program's verification conditions are discharged via bounded symbolic execution (loops unrolled to k=32) targeting Z3's QF_BV/QF_FP theories, with paths that timeout routed to coverage-guided differential fuzzing seeded by the symbolic engine's partial exploration.

The key refinement: the product program is not naively the cross-product of both CFGs (which explodes). A *structural alignment pass* identifies matching control-flow skeletons via a greedy longest-common-subsequence over basic block signatures (instruction opcode sequences, type profiles). Coercion points are inserted only at aligned divergence sites, and unaligned tails are flagged as structural mismatches. This reduces the product program to ≤2x the larger input function for structurally similar translations (C2Rust, LLM output).

### 1. Extreme Value Delivered

**Who needs it:** Defense contractors and critical-infrastructure teams executing DoD-mandated C-to-Rust migrations. Today, a team migrating libsodium or OpenSSL has exactly two options: (1) run the test suite and hope, or (2) hire formal verification consultants at $500K+ per library. This tool gives them a third option: feed in C source + C2Rust output, get back either "equivalent on all inputs within loop bound 32" or "diverges on input a=2147483647, b=1" in under 60 seconds per function.

**Secondary audience:** AI-assisted migration users. Every engineer using GPT-4/Claude to translate C→Rust today has no way to validate the output beyond manual review. This tool is the missing CI gate.

### 2. Why Genuinely Difficult

1. **Frontend fidelity is the hardest unsolved problem.** Building a C frontend that faithfully encodes integer promotion rules (§6.3.1.1), usual arithmetic conversions, implicit conversions, and platform-dependent widths into the IR is ~5-8K LoC of extremely subtle code. One wrong promotion rule produces silent false equivalences. libclang gives you the AST, but the promotion/conversion semantics must be manually reconstructed from the C11 standard.

2. **Structural alignment is heuristic.** C2Rust output inserts extra `let` bindings, temporaries, and explicit casts that don't exist in the C source. The alignment pass must recognize that `let tmp = a as i64; tmp + b as i64` aligns with `a + b` (with implicit promotion). This requires normalization that is fragile and hard to make complete.

3. **SMT encoding of floating-point semantics.** Z3's QF_FP theory is sound but extremely slow. A single floating-point multiplication comparison can take 30+ seconds. The system must decide when to use QF_FP vs. QF_BV vs. real-valued approximations, and these decisions interact with soundness.

4. **The ~52K LoC budget is real** (realistic estimate: 65-75K). The custom IR, two frontends, product program construction, symbolic execution engine, SMT encoding, fuzzing fallback, and CLI are all load-bearing.

### 3. Load-Bearing Math Required

**Integer Reconciliation Case Table.** A complete enumeration of semantic divergence points between C and Rust, with per-operation SMT encoding:
- For each binary operation ⊕ and each (c_overflow_mode, rust_overflow_mode) pair, define the verification obligation: "either the operation is well-defined under both configurations and results agree, or a semantic gap is reported with a concrete witness."
- Four overflow modes: C UB, Rust wrapping, Rust panicking, Rust saturating.
- This is load-bearing: without it, the product program doesn't know what to assert at each divergence site.

**Product Program Construction.** An algorithm for constructing a relational product program from two IR programs under different semantic configurations, with automatic insertion of coercion points. Correctness: if the product program's VCs are valid, the two functions agree on all well-defined inputs within loop bound k.

### 4. Best-Paper Potential

**Venue:** PLDI, OOPSLA, or CAV. **Why accepted:** (1) The LLVM-IR-erasure thesis is a crisp, demonstrable negative result—functions where LLREVE reports "equivalent" and this tool reports "divergent" with a concrete witness. (2) The integer reconciliation encoding is a genuine formal contribution that advances relational verification. (3) The evaluation is exhaustive (ALL functions in C2Rust(libsodium), not cherry-picked) and produces an independently valuable divergence taxonomy. (4) Urgent real-world demand (DoD mandate).

### 5. Hardest Technical Challenge

**C frontend fidelity—specifically, faithfully encoding C's integer promotion and usual arithmetic conversion rules.** Every week will reveal another C semantic corner case (implicit conversions in ternary expressions, variadic functions, compound assignment with side effects) that requires IR changes cascading into product program construction and SMT encoding.

**Mitigation:** (1) Scope to the C subset emitted by C2Rust (predictable, well-defined subset). (2) Use libclang's type resolution APIs to extract the compiler's actual promotion decisions rather than reimplementing from the standard. (3) Differential-test the frontend against GCC/Clang debug info on ~500 edge cases.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 9 | Fills an urgent, well-funded gap with no existing solution |
| **Difficulty** | 9 | ~52-75K LoC, two language frontends, custom IR, novel product programs |
| **Potential** | 9 | LLVM-IR-erasure thesis + divergence taxonomy = strong PLDI/CAV submission |
| **Feasibility** | 5 | Extremely implementation-heavy; frontend fidelity risk; 12-18 months realistic |

---

## Approach B: Augmented IR with Semantic Annotations

### Core Idea

Instead of building entirely custom frontends, *augment* the existing LLVM IR compilation pipeline with semantic metadata annotations that survive lowering. (1) Custom Clang and rustc plugins annotate every semantically significant operation with metadata recording source-language semantics. For example, when Clang emits `add nsw i32 %a, %b`, the pass attaches metadata `!c_overflow_ub {semantics: "UB_on_overflow"}`. When rustc emits the same instruction, the pass attaches `!rust_overflow {semantics: "wrap_release" | "panic_debug"}`. (2) A *modified equivalence checker* on the annotated IR reads these annotations and generates qualified assertions (same coercion logic as Approach A, but expressed as LLVM metadata). (3) KLEE (with modifications) symbolically executes the annotated product program.

The key insight: LLVM IR doesn't *inherently* erase semantics—it just doesn't *record* them. By instrumenting the pipeline before optimization, we get LLVM's mature infrastructure plus source-level semantic distinctions.

### 1. Extreme Value Delivered

Same audience as Approach A, but with lower adoption barrier. Instead of trusting a custom C frontend, users trust Clang and rustc (which they already use) plus a thin annotation layer. The tool integrates into existing build systems: `clang -fplugin=semantic-annotator -emit-llvm` produces annotated IR directly.

### 2. Why Genuinely Difficult

1. **Annotation survival through optimization (Difficulty: 10/10).** LLVM optimization passes aggressively transform IR. Custom metadata is routinely dropped by DCE, GVN, LICM, loop unrolling, inlining, SROA. LLVM's `!metadata` mechanism is explicitly documented as "may be dropped by optimizations." You'd need to either modify 20+ LLVM passes (forking LLVM), run on unoptimized IR (negating benefits), or accept lossy annotations.

2. **Cross-compiler annotation consistency.** Clang and rustc use different LLVM versions (rustc often lags 1-2 releases), different optimization pipelines, and different IR generation strategies.

3. **KLEE modifications.** KLEE is designed for single-program symbolic execution, not product programs. Adapting for relational execution requires ~8K LoC of modifications.

### 3. Load-Bearing Math Required

**Annotation Preservation (Aspired but Likely Infeasible).** A formal argument that semantic annotations survive the curated subset of LLVM optimization passes permitted by the tool. This requires formalizing each permitted pass's effect on annotations—tractable only for a small whitelist (mem2reg, simplifycfg, restricted instcombine).

**Qualified Relational Assertions.** Same as Approach A's coercion logic but expressed as LLVM metadata queries.

### 4. Best-Paper Potential

**Venue:** ISSTA, ASE, or ICSE. The annotation-augmented IR is a practically deployable contribution that integrates with existing build systems. The paper demonstrates that the LLVM-IR-erasure problem can be partially solved without abandoning LLVM IR. Weakness: theoretical contribution is thin—a PLDI/CAV reviewer would view this as engineering.

### 5. Hardest Technical Challenge

**Annotation survival through LLVM optimization passes.** The fundamental tension: KLEE performs poorly on `-O0` IR, but optimized IR loses annotations.

**Mitigation:** Run at `-O1` with a curated pass pipeline. Allow mem2reg, simplifycfg, restricted instcombine. Disallow loop unrolling, vectorization, aggressive inlining. Implement a validation pass that flags instructions missing annotations.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8 | Same problem, better deployment story, slightly less precise |
| **Difficulty** | 7 | Reuses LLVM/KLEE infrastructure; plugin development is hard but bounded |
| **Potential** | 5 | Solid SE venue paper; lacks theoretical depth for top PL venues |
| **Feasibility** | 4 | Annotation survival is an unsolved research problem; silent failures are the worst failure mode |

---

## Approach C: Abstract Interpretation + Differential Testing

### Core Idea

Abandon symbolic execution and SMT solving. Use *abstract interpretation* to compute conservative semantic summaries of both programs, then compare summaries to detect guaranteed divergences or prove equivalence on specific dimensions. Where summaries are inconclusive, fall back to *smart differential fuzzing* seeded by abstract domain boundaries.

Concretely: (1) Parse both functions into lightweight representations (libclang + syn). (2) Run interval analysis on all integer variables, computing [lo, hi] ranges at each program point. (3) At each arithmetic operation, check whether the interval intersects the overflow boundary. If C's interval for `a + b` includes values where `a + b > INT_MAX`, and Rust wraps on overflow, the analysis proves a potential divergence exists and emits the boundary as a fuzzing seed. (4) Differential fuzzing targets the exact regions where divergences are mathematically possible.

No SMT solver, no product programs, no symbolic execution engine. The trade-off is precision: abstract interpretation over-approximates, so some "potential divergences" are false alarms that fuzzing must confirm or refute.

### 1. Extreme Value Delivered

**Who needs it:** Teams needing *fast triage*, not formal proof. A migration team porting a 10,000-function codebase needs to know which functions are *likely divergent* so they can prioritize manual review. This tool runs on the entire codebase in minutes, producing a ranked risk list.

**CI/CD integration.** Sub-second per function for abstract analysis, seconds for targeted fuzzing. This is a linter, not a theorem prover.

### 2. Why Genuinely Difficult

1. **Abstract interpretation precision on real code.** Interval analysis is fast but imprecise. For `int f(int x) { return x * x + x; }`, intervals massively over-approximate the range, triggering false overflow warnings. More precise domains (octagons, polyhedra) are slower and harder to implement.

2. **No equivalence proofs.** Abstract interpretation can prove *absence of overflow* but cannot prove *functional equivalence*. The tool says "no divergence risk detected" but not "provably equivalent." This is a fundamental limitation.

3. **Fuzzing may not confirm potential divergences.** For functions with complex input dependencies, the fuzzer may not find satisfying inputs within its budget.

4. **The "80% just fuzzing" risk.** After building abstract interpretation, you may discover it produces conclusive results on only 10-20% of real function pairs. The remaining 80% trigger fuzzing fallback, making the abstract interpretation infrastructure dead weight.

### 3. Load-Bearing Math Required

**Custom Abstract Domain D = Interval × OverflowFlag × PrecisionFlag.** A reduced product of three domains with abstract transfer functions for all operations. This is the domain definition, Galois connection (standard), widening operator, and divergence detection rules. The domain is genuinely load-bearing: without it, you're just running interval analysis without semantic awareness.

**Summary Comparison Soundness.** If abstract summaries of two functions disagree, a concrete divergence exists in the concretization of the gap. This justifies using abstract interpretation for divergence detection (but not equivalence proof).

### 4. Best-Paper Potential

**Venue:** FSE, ISSTA, or ICSE. Core argument: "you don't need SMT solvers to find 80% of migration bugs." Abstract-interpretation-seeded fuzzing is a novel combination. If the tool finds ≥70% of divergences that Approach A finds in 1% of the time, the speed/precision trade-off is compelling. Weakness: cannot prove equivalence, which a PLDI/CAV reviewer will note.

### 5. Hardest Technical Challenge

**Precision of abstract interpretation on real-world functions with complex control flow.** Interval analysis on `for (int i = 0; i < n; i++) sum += arr[i]` will widen `sum` to [INT_MIN, INT_MAX] after a few iterations, producing MayOverflow for every summation loop.

**Mitigation:** (1) Relational abstract domains (octagons) via the Apron library. (2) Trace partitioning at conditional branches. (3) Parametric widening that keeps loop bounds symbolic. (4) Accept that some functions will produce false positives and track the false alarm rate as a metric.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 7 | Fast triage is valuable but "potential divergence" < "proven divergence" |
| **Difficulty** | 6 | Simpler architecture; abstract interpretation + fuzzing are well-understood |
| **Potential** | 6 | Good SE venue paper; lacks the "surprising result" of Approach A |
| **Feasibility** | 8 | ~20-35K LoC, no SMT dependency, fastest path to working prototype |

---

## Comparative Summary

| Dimension | A: Product Programs | B: Augmented IR | C: AbsInt + Fuzzing |
|-----------|:---:|:---:|:---:|
| **Value** | 9 | 8 | 7 |
| **Difficulty** | 9 | 7 | 6 |
| **Potential** | 9 | 5 | 6 |
| **Feasibility** | 5 | 4 | 8 |
| **LoC Estimate** | 52-75K | 40-55K | 20-35K |
| **Can prove equivalence?** | Yes (bounded) | Yes (bounded) | No |
| **Can find divergences?** | Yes (concrete) | Yes (concrete) | Yes (after fuzzing) |
| **Time per function** | 1–60s | 5–120s | <2s |
| **SMT solver required?** | Yes (Z3) | Yes (via KLEE) | No |
| **Best venue** | PLDI / CAV | ISSTA / ASE | FSE / ISSTA |
| **Primary risk** | Frontend fidelity | Annotation survival | Precision of abstract domains |
