# SemRec: Verification Oracle for C↔Rust Translation

**SemRec catches semantic bugs in C→Rust translations that testing cannot find.**

Given a C function and its Rust translation, SemRec returns a **(verdict, counterexample, repair hint)** triple using Z3-backed formal verification. It detects UB-related divergences—like signed overflow differences—that are invisible to differential testing because C undefined behavior is erased at the IR level.

```
$ semrec verify --c-code 'int f(int x) { return x + 1; }' \
                --rs-code 'pub fn f(x: i32) -> i32 { x + 2 }' --format text
Verdict: divergent
Counterexample: Divergence: output_mismatch
  Inputs: {'input_0': '0'}
  Class: other
Repair: Semantic divergence detected: output_mismatch
Time: 151.2ms
```

## Quick Start

```bash
git clone <repo-url> && cd cross-language-equivalence-verifier
pip install z3-solver

# Verify a single pair
python3 -c "
from src.semrec_cli import main
main(['verify', '--c-code', 'int f(int x){return x+1;}',
      '--rs-code', 'pub fn f(x:i32)->i32{x.wrapping_add(1)}', '--format', 'text'])
"
# → Verdict: divergent
#   Counterexample: input_0=2147483647 (C has UB on overflow, Rust wraps)

# CEGAR loop: auto-translate C→Rust with LLM + verification (requires OPENAI_API_KEY)
python3 -c "
from src.semrec_cli import main
main(['cegar', '--source-code', 'int max2(int a, int b){return a>b?a:b;}'])
"
# → ✓ CONVERGED after 1 iteration
```

## Why SemRec?

Differential testing compiles both C and Rust, then compares outputs on random inputs. But C undefined behavior (signed overflow, shift-by-width, division by zero) is **erased by the compiler**—both binaries produce identical outputs on test inputs, hiding real semantic divergences.

SemRec operates at the **source level**, encoding the C11 and Rust semantics into separate Z3 bitvector formulas via a **σ-bridge** (semantic configuration layer). It finds divergences that testing structurally cannot reach.

In our ablation study, removing the σ-bridge drops accuracy from 80% to 40% on 10 pairs, confirming that UB-aware source-level reasoning is the key differentiator.

## Benchmark Results

| Metric | Value |
|--------|-------|
| Benchmark pairs | **352** across 14 categories (202 original + 150 expanded) |
| Classification accuracy (202 pairs) | **86.6%** (175/202) |
| Expanded categories | struct, enum, float, C2Rust, iterator, cast, compound, control_flow |
| CEGAR convergence (30 functions) | **16.7%** overall; **0%** for UB functions |
| Avg verification time | **< 200ms** per pair |

**Category breakdown** (original 202 pairs, best → worst):

| Category | Pairs | Correct | Accuracy |
|----------|-------|---------|----------|
| Shift | 17 | 17 | 100% |
| Cast | 13 | 13 | 100% |
| Real patterns | 26 | 26 | 100% |
| Compound | 12 | 12 | 100% |
| Control flow | 11 | 11 | 100% |
| Arithmetic | 49 | 46 | 93.9% |
| Division | 21 | 19 | 90.5% |
| Bitwise | 29 | 25 | 86.2% |
| Error handling | 8 | 4 | 50% |
| Loops (BMC K=32) | 8 | 2 | 25% |
| Memory | 6 | 0 | 0% |
| String | 2 | 0 | 0% |

Memory and string categories are outside the supported fragment and return `unknown`.
Loop analysis is **bounded model checking** at depth K=32 (not full verification).

## Architecture

```
C source ──→ CParser ──→ SSA IR ──┐
                                   ├─→ ProductBuilder ──→ SMT Encoder ──→ Z3 ──→ Verdict
Rust source → RustParser → SSA IR ┘           ↑
                                          σ-bridge
                                     (SemanticConfig)
                                   σ_C: overflow=UB, shift=UB
                                   σ_R: overflow=wrap, shift=mask
```

**Key components:**
- **σ-bridge** (`src/semantics/`): Encodes the C11 vs Rust semantic gap. C signed overflow is UB; Rust wraps. C shift ≥ width is UB; Rust masks. Parameterized via `SemanticConfig.c11()` / `SemanticConfig.rust_release()`.
- **Product program** (`src/product_program/`): Aligns C and Rust IR into a single program, adds coercion points where semantics diverge.
- **SMT encoder** (`src/smt/`): Lowers the product program to QF_BV (quantifier-free bitvectors). Decidable and complete within the supported fragment.
- **CEGAR engine** (`src/cegar_engine.py`): Iteratively calls LLM to translate, verifies with oracle, feeds counterexamples back as repair hints.

## CLI Reference

### `semrec verify` — Verify a C/Rust pair

```
--c-file PATH       C source file
--c-code STRING     Inline C code
--rs-file PATH      Rust source file
--rs-code STRING    Inline Rust code
--function, -f      Function name (auto-inferred if omitted)
--timeout MS        Z3 timeout in ms (default: 10000)
--format json|text  Output format (default: json)
--output, -o PATH   Write result to file
```

### `semrec cegar` — CEGAR translation loop

```
--source-file PATH  C source file
--source-code STR   Inline C code
--model MODEL       LLM model (default: gpt-4.1-nano)
--max-iter N        Max CEGAR iterations (default: 5)
--timeout MS        Z3 timeout in ms (default: 10000)
--function, -f      Function name
--output, -o PATH   Write result to file
```

Requires `OPENAI_API_KEY` environment variable. Supported models: `gpt-5-chat-latest`, `gpt-4.1-nano`.

### `semrec bench` — Run benchmark suite

```
--output, -o PATH   Write results to file
--pairs N           Number of pairs (0 = all 202)
--category NAME     Filter by category
--cegar             Run CEGAR evaluation
--model MODEL       LLM model for CEGAR (default: gpt-4.1-nano)
--max-iter N        Max CEGAR iterations (default: 5)
```

### Global flags

```
--version           Show version (0.2.0)
-v, --verbose       Increase verbosity
-q, --quiet         Suppress output
```

## Scope and Limitations

| Feature | Status |
|---------|--------|
| Integer arithmetic (i8–i64, u8–u64) | ✅ Supported |
| Bitwise operations | ✅ Supported |
| Control flow (if/else, ternary, switch/match) | ✅ Supported |
| Type casts (widening, narrowing, sign change) | ✅ Supported |
| Comparisons | ✅ Supported |
| **Struct types** (field access, construction, nested) | ✅ Supported |
| **Enum/tagged union types** (discriminant, variants) | ✅ Supported |
| **Floating-point** (IEEE 754 f32/f64) | ✅ Supported |
| Bounded loops (BMC at K=32) | ⚠ Bounded model checking |
| Pointers, heap allocation | ❌ Outside fragment |
| Strings (pointer-based) | ❌ Outside fragment |
| Interprocedural analysis | ❌ Outside fragment |
| Concurrency | ❌ Outside fragment |

Hand-written recursive descent parsers for C and Rust are a known limitation. Loop analysis is bounded model checking (BMC) at depth K=32—loops exceeding K iterations are not fully verified.

## Repository Structure

```
src/
  oracle/            # VerificationOracle API
  cegar_engine.py    # CEGAR loop engine
  semrec_cli.py      # CLI entry point
  smt/               # SMT encoder, solver, decoder
  semantics/         # σ-bridge (SemanticConfig)
  product_program/   # Product program construction + alignment
  frontend_c/        # C parser and IR lowering
  frontend_rust/     # Rust parser and IR lowering
  ir/                # Shared typed SSA IR
benchmarks/
  pairs/             # 352 benchmark pairs (52 core + 150 scaled + 150 expanded)
    benchmark_pairs.py           # Original 52 core pairs
    expanded_benchmark_pairs.py  # 150 expanded pairs (struct, enum, float, C2Rust, etc.)
tests/               # Unit tests
examples/            # Example scripts
experiments/
  results/             # All experiment results (JSON)
    full_benchmark_v2.json    # 202-pair benchmark (86.6% accuracy)
    ablation_results.json     # σ-bridge ablation study
    cegar_final.json          # CEGAR convergence results
    scaled_cegar_results.json # Scaled CEGAR on 215 functions
docs/
  architecture.md    # Detailed architecture and module map
  reviews/           # Peer review documents
  research/          # Research process artifacts
theory/
  paper.tex            # Paper with proofs (pdflatex-compilable)
```

## Requirements

```
python >= 3.9
z3-solver >= 4.12
openai >= 1.0        # only needed for CEGAR experiments
```

## Paper

```bash
cd theory && pdflatex paper.tex
```

Includes: Conditional Soundness theorem, Counterexample Correctness theorem, Decidability proposition, UB Divergence Irreparability lemma.
