# Cross-Language Equivalence Verifier (XLEV)

A verification tool that checks whether a C function and its Rust translation
(e.g., produced by C2Rust) are semantically equivalent, with precise handling
of the semantic differences between C and Rust.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         CLI (cli/)                          │
│  main.py → config.py → pipeline.py → reporter.py           │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                  Verification Pipeline                       │
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌───────────┐  │
│  │Parse C  │──▶│ Lower   │──▶│ Analyze  │──▶│ Normalize │  │
│  │Parse Rust│  │ to IR   │   │ (CFG,Dom)│   │           │  │
│  └─────────┘   └─────────┘   └──────────┘   └─────┬─────┘  │
│                                                     │        │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌─────▼─────┐  │
│  │ Report  │◀──│  Fuzz   │◀──│   SMT    │◀──│ Product   │  │
│  │         │   │(timed-  │   │  Check   │   │ Program + │  │
│  │         │   │ out)    │   │          │   │ SymbExec  │  │
│  └─────────┘   └─────────┘   └──────────┘   └───────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Module Map

| Module | Purpose |
|--------|---------|
| `src/ir/` | Shared intermediate representation (types, instructions, builder, validator) |
| `src/frontend_c/` | C lexer, parser, AST, type resolver, IR lowering |
| `src/frontend_rust/` | Rust lexer, parser, AST, type resolver, IR lowering |
| `src/semantics/` | Semantic configuration, evaluator, divergence tables |
| `src/type_system/` | Type checker, promotion rules, coercion chains |
| `src/analysis/` | CFG, dominator trees, loop detection, dataflow, alias analysis |
| `src/product_program/` | Function alignment, coercion insertion, product construction |
| `src/symbolic_exec/` | Symbolic executor, path manager, symbolic memory |
| `src/smt/` | SMT encoder, solver interface, model decoder |
| `src/fuzzer/` | Fuzz engine, seed generation, coverage, minimization |
| `src/stdlib_models/` | Models for C/Rust standard library functions |
| `src/cli/` | CLI, configuration, pipeline orchestration, reporting |
| `src/utils/` | Source locations, diagnostics, graph utils, bit manipulation |

## Installation

```bash
# Clone and install
cd implementation/
pip install -e .

# Or just add to PYTHONPATH
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

### Dependencies

- **Python 3.9+**
- **pytest** (for tests)
- **z3-solver** (optional, for SMT solving)
- **PyYAML** (optional, for YAML config files)

```bash
pip install pytest z3-solver pyyaml
```

## Quickstart

### Command Line

```bash
# Verify equivalence of C and Rust functions
python -m src.cli.main verify --c-source add.c --rust-source add.rs --function add

# Run with fast profile (reduced bounds)
python -m src.cli.main verify -c lib.c -r lib.rs -f compute --profile fast

# Differential fuzzing only
python -m src.cli.main fuzz -c func.c -r func.rs -f process --iterations 10000

# Run analysis passes
python -m src.cli.main analyze -c lib.c -r lib.rs

# Benchmark suite
python -m src.cli.main benchmark --suite benchmarks/ --timeout 120
```

### Python API

```python
from src.cli.config import VerifyConfig
from src.cli.pipeline import VerificationPipeline

config = VerifyConfig.fast()
pipeline = VerificationPipeline(config)

report = pipeline.verify(
    c_source="int add(int a, int b) { return a + b; }",
    rust_source="pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }",
    c_function="add",
    rust_function="add",
)

print(report.format_terminal())
print(f"Verdict: {report.verdict.kind.value}")
```

### Configuration Profiles

| Profile | Loop Bound | Timeout | SMT Timeout | Fuzz Iters |
|---------|-----------|---------|-------------|------------|
| default | 100 | 300s | 120s | 10,000 |
| fast | 10 | 30s | 10s | 1,000 |
| thorough | 500 | 600s | 300s | 100,000 |
| fuzz-only | — | — | — | 50,000 |

```python
from src.cli.config import VerifyConfig, get_profile

config = get_profile("thorough")
config.loops.max_unroll = 20  # customize
```

### Config File (`.xlev.json`)

```json
{
    "output_format": "json",
    "timeouts": {
        "total_timeout": 120,
        "smt_timeout": 60
    },
    "loops": {
        "max_unroll": 15,
        "default_bound": 50
    },
    "fuzzer": {
        "enabled": true,
        "seed_count": 2000,
        "max_iterations": 20000
    }
}
```

## Mathematical Foundation

The verifier is based on the theory of **product programs** for relational
verification. Given a C function *f_C* and a Rust function *f_R*, we:

1. **Parse and lower** both to a shared IR that preserves semantic information
   (overflow behavior, signedness, pointer provenance).

2. **Align** the two functions using a cost-based block alignment algorithm
   that identifies corresponding code regions.

3. **Construct a product program** *P(f_C, f_R)* that interleaves both
   executions and inserts **coercion points** where C and Rust semantics diverge.

4. **Symbolically execute** the product program, tracking path conditions
   and divergence conditions.

5. **Check via SMT** whether any divergence condition is satisfiable. A SAT
   result produces a **counterexample** (concrete inputs where behavior differs).

### Divergence Categories

| Category | C Behavior | Rust Behavior |
|----------|-----------|---------------|
| `integer_overflow` | Undefined (signed), wraps (unsigned) | Panics (debug), wraps (release) |
| `signed_unsigned_mismatch` | Implicit conversion | Explicit `as` cast required |
| `shift_overflow` | UB if shift ≥ width | Masks to width - 1 |
| `division_by_zero` | UB | Panics |
| `array_bounds` | No checking | Panics on out-of-bounds |
| `null_handling` | Null dereference is UB | Option/Result types |
| `pointer_semantics` | Flat address space | Provenance-based model |
| `float_precision` | Implementation-defined | IEEE 754 strict |
| `cast_truncation` | Implementation-defined | Saturating by default |

### Semantic Configurations

The evaluator is parameterized by a `SemanticConfig` that captures the
language-specific semantics:

```python
from src.semantics import SemanticConfig

c_config = SemanticConfig.c11()           # Standard C11 semantics
c_opt = SemanticConfig.c11_optimized()    # C11 with -O2 assumptions
rust_dbg = SemanticConfig.rust_debug()    # Rust debug mode
rust_rel = SemanticConfig.rust_release()  # Rust release mode
```

## Evaluation Methodology

### Verdict Levels

- **Equivalent** (confidence 100%): All paths verified by SMT solver.
- **Divergent** (confidence 90–100%): Counterexample found and minimized.
- **Unknown** (confidence 0–90%): Partial verification or timeout.

### Coverage Metrics

- **Path coverage**: Fraction of symbolic paths explored.
- **Block coverage**: Fraction of basic blocks reached (per language).
- **Instruction coverage**: Fraction of IR instructions executed.

## Running Tests

```bash
cd implementation/
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_ir_types.py -v
python -m pytest tests/test_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Examples

```bash
# Simple arithmetic verification
python examples/simple_verify.py

# Integer overflow divergence demo
python examples/overflow_demo.py

# C2Rust translation verification
python examples/c2rust_verify.py

# Differential fuzzing demo
python examples/fuzz_demo.py
```

## Project Structure

```
implementation/
├── src/
│   ├── ir/                  # Intermediate representation
│   ├── frontend_c/          # C frontend
│   ├── frontend_rust/       # Rust frontend
│   ├── semantics/           # Semantic evaluation
│   ├── type_system/         # Type checking & promotion
│   ├── analysis/            # Program analysis
│   ├── product_program/     # Product program construction
│   ├── symbolic_exec/       # Symbolic execution engine
│   ├── smt/                 # SMT solver interface
│   ├── fuzzer/              # Differential fuzzer
│   ├── stdlib_models/       # Standard library models
│   ├── cli/                 # Command-line interface
│   └── utils/               # Utility modules
├── tests/                   # Test suite (pytest)
├── examples/                # Example scripts
├── benchmarks/              # Benchmark suite
└── README.md
```

## License

Research prototype. See repository root for license information.
