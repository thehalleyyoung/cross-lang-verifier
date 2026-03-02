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
pip install z3-solver tree-sitter tree-sitter-c tree-sitter-rust

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

SemRec operates at the **source level**, encoding the C11 and Rust semantics into separate Z3 bitvector/array formulas via a **σ-bridge** (semantic configuration layer). It finds divergences that testing structurally cannot reach.

## Benchmark Results

| Metric | Value |
|--------|-------|
| Full benchmark (511 pairs) | **35.4%** accuracy (↑ from 15.6%) |
| Core pairs (52 pairs) | **48.1%** accuracy |
| Combined pairs (212 pairs) | **27.4%** accuracy |
| K-sensitivity (K∈{8,16,32,64,128}) | Saturates at **K=32** (63.2%) |
| σ-bridge divergence coverage | **17/26** classes (65.4%) |
| Formal theorems | **8** (4 lemmas + 4 theorems, pen-and-paper proofs) |
| Avg verification time | **138ms** mean |
| Test suite | **847** tests passing |

**Pipeline coverage is the primary bottleneck.** Accuracy on the core fragment (arithmetic, division, shift, cast) exceeds 80%. The gap on larger benchmarks is due to IR lowering failures on struct-heavy, loop-heavy, or pointer-heavy code.

## Architecture

```
C source ──→ TreeSitterCParser ──→ SSA IR ──┐
              (fallback: CParser)            ├→ ProductBuilder → SMT Encoder → Z3 → Verdict
Rust source → TreeSitterRustParser → SSA IR ┘         ↑              ↑
              (fallback: RustParser)              σ-bridge     Points-to +
                                             (SemanticConfig)  Ownership
                                           σ_C: overflow=UB    Axioms
                                           σ_R: overflow=wrap
```

**Key components:**
- **Tree-sitter parsers** (`src/frontend_c/tree_sitter_parser.py`, `src/frontend_rust/tree_sitter_parser.py`): Primary parsing via tree-sitter grammars with hand-written fallback.
- **σ-bridge** (`src/semantics/`): Encodes the C11 vs Rust semantic gap. 17 divergence classes handled.
- **Product program** (`src/product_program/`): Aligns C and Rust IR with σ-bridge coercions. Formal soundness proof with 8 theorems in `src/product_program/soundness.py`.
- **SMT encoder** (`src/smt/`): Lowers to QF_BV/QF_ABV. Includes enhanced memory model with points-to analysis and TBAA.
- **CEGAR engine** (`src/cegar_engine.py`): LLM translation + verification loop with UB-aware hints.

## Scope and Limitations

| Feature | Status |
|---------|--------|
| Integer arithmetic (i8–i64, u8–u64) | ✅ Supported |
| Wrapping/checked/saturating arithmetic | ✅ Supported |
| Bitwise operations | ✅ Supported |
| Control flow (if/else, switch/match) | ✅ Supported |
| Type casts (widening, narrowing, sign) | ✅ Supported |
| Floating-point (IEEE 754) | ✅ Supported |
| Unsafe Rust (raw pointers, deref) | ✅ Basic support |
| Pointer/memory (alloca, load, store, GEP) | ✅ QF_ABV |
| Points-to analysis + ownership axioms | ✅ Supported |
| Bounded loops (BMC at K=32) | ⚠ BMC only (K=32 empirically justified) |
| Struct/enum field access | ⚠ Partial (simple cases only) |
| Interprocedural analysis | ❌ Not supported |
| Generics / trait dispatch | ❌ Not supported |
| Concurrency | ❌ Not supported |

## Requirements

```
python >= 3.9
z3-solver >= 4.12
tree-sitter >= 0.22
tree-sitter-c >= 0.21
tree-sitter-rust >= 0.21
openai >= 1.0        # only needed for CEGAR experiments
```

## Paper

```bash
cd theory && pdflatex tool_paper.tex && pdflatex tool_paper.tex  # 18 pages
```
