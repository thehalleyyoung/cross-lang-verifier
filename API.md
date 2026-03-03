# SemRec API Reference

Complete reference for the SemRec verification oracle, pipeline, and CLI. Documents only features implemented in `src/`.

---

## Table of Contents

- [VerificationOracle](#verificationoracle)
- [VerificationPipeline](#verificationpipeline)
- [PipelineBuilder](#pipelinebuilder)
- [CEGAR Engine](#cegar-engine)
- [CLI Commands](#cli-commands)
  - [semrec verify](#semrec-verify)
  - [semrec fuzz](#semrec-fuzz)
  - [semrec analyze](#semrec-analyze)
  - [semrec benchmark](#semrec-benchmark)
  - [semrec cegar](#semrec-cegar)
  - [semrec bench (legacy)](#semrec-bench-legacy)
- [Configuration](#configuration)
- [Data Classes](#data-classes)
- [Output Formats](#output-formats)
- [Error Handling](#error-handling)
- [Tree-Sitter Parsers](#tree-sitter-parsers)
- [SMT Encoder — Struct/Enum Support](#smt-encoder--structenum-support)
- [Enhanced Memory Model](#enhanced-memory-model)
- [Benchmarks](#benchmarks)

---

## VerificationOracle

**Module:** `src/oracle/oracle.py`

The core verification API. Takes C and Rust source strings and returns a structured `(verdict, counterexample, repair_hint)` triple.

### Constructor

```python
VerificationOracle(timeout_ms=10000, func_name="", cegar_mode=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout_ms` | `int` | `10000` | Z3 solver timeout in milliseconds |
| `func_name` | `str` | `""` | Default function name for verification |
| `cegar_mode` | `bool` | `False` | When `True`, UB-only divergences return `"conditionally_equivalent"` instead of `"divergent"` |

### `oracle.verify(c_code, rust_code, func_name=None) → OracleResult`

Verify semantic equivalence of a C/Rust function pair.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_code` | `str` | *(required)* | Complete C function source |
| `rust_code` | `str` | *(required)* | Complete Rust function source |
| `func_name` | `str` | `None` | Function name (auto-inferred from source if omitted) |

**Returns:** [`OracleResult`](#oracleresult)

```python
from src.oracle.oracle import VerificationOracle

oracle = VerificationOracle(timeout_ms=5000)
result = oracle.verify(
    "int f(int x) { return x + 1; }",
    "pub fn f(x: i32) -> i32 { x.wrapping_add(1) }"
)
print(result.verdict)                         # "divergent"
print(result.counterexample.inputs)           # {"input_0": "2147483647"}
print(result.counterexample.reason)           # "c_undefined_behavior..."
print(result.counterexample.divergence_class) # "undefined_behavior"
print(result.repair_hint.suggested_fix)       # "Review C semantics..."
print(result.time_ms)                         # 12.3
```

### `oracle.verify_batch(pairs) → List[OracleResult]`

Verify multiple function pairs in sequence.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pairs` | `List[Tuple[str, str, str]]` | List of `(func_name, c_code, rust_code)` triples |

```python
results = oracle.verify_batch([
    ("add", "int add(int a, int b) { return a + b; }",
            "pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }"),
    ("max", "int max(int a, int b) { return a > b ? a : b; }",
            "pub fn max(a: i32, b: i32) -> i32 { a.max(b) }"),
])
for r in results:
    print(f"{r.func_name}: {r.verdict} ({r.time_ms:.1f}ms)")
```

---

## VerificationPipeline

**Module:** `src/cli/pipeline.py`

The full verification pipeline with parsing, IR lowering, analysis, SMT encoding, fuzzing, and reporting. Use this when you need more control than `VerificationOracle` provides.

### Constructor

```python
VerificationPipeline(config: Optional[VerifyConfig] = None)
```

### `pipeline.verify(c_source, rust_source, c_function=None, rust_function=None) → VerificationReport`

Run the complete verification pipeline on a source pair.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_source` | `str` | *(required)* | C source code |
| `rust_source` | `str` | *(required)* | Rust source code |
| `c_function` | `str` | `None` | C function name (auto-inferred if omitted) |
| `rust_function` | `str` | `None` | Rust function name (auto-inferred if omitted) |

**Returns:** [`VerificationReport`](#verificationreport)

```python
from src.cli.pipeline import VerificationPipeline

pipeline = VerificationPipeline()
report = pipeline.verify(
    'int add(int a, int b) { return a + b; }',
    'pub fn add(a: i32, b: i32) -> i32 { a + b }',
    c_function="add",
    rust_function="add"
)
print(report.verdict)
print(report.counterexamples)
print(report.timing)
```

### `pipeline.verify_from_files(c_path, rust_path, c_function=None, rust_function=None) → VerificationReport`

Verify from file paths instead of source strings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_path` | `str` | *(required)* | Path to C source file |
| `rust_path` | `str` | *(required)* | Path to Rust source file |
| `c_function` | `str` | `None` | C function name |
| `rust_function` | `str` | `None` | Rust function name |

### `pipeline.fuzz_only(c_source, rust_source, c_function=None, rust_function=None) → VerificationReport`

Run only the differential fuzzing phase (no SMT verification).

### `pipeline.analyze_only(c_source, rust_source) → Dict[str, Any]`

Run only the static analysis passes (alias analysis, points-to, etc.) without verification.

### `pipeline.set_progress_callback(callback) → None`

Register a callback for pipeline progress updates.

| Parameter | Type | Description |
|-----------|------|-------------|
| `callback` | `ProgressCallback` | `fn(phase_name: str, status: str)` |

### Pipeline Phases

The pipeline executes these phases in order:

| Phase | Description |
|-------|-------------|
| `parse_c` | Parse C source via TreeSitterCParser (fallback: CParser) |
| `parse_rust` | Parse Rust source via TreeSitterRustParser (fallback: RustParser) |
| `lower_c` | Lower C AST to SSA IR |
| `lower_rust` | Lower Rust AST to SSA IR |
| `validate_ir` | Validate IR well-formedness |
| `analysis` | Run static analysis (alias, points-to, TBAA) |
| `normalize` | Normalize IR for alignment |
| `align` | Align C and Rust IR into product program |
| `product` | Build product program with σ-bridge coercions |
| `symbolic` | Symbolic execution for path enumeration |
| `smt` | SMT encoding and Z3 solving |
| `fuzz` | Differential fuzzing for additional coverage |
| `report` | Build final verification report |

---

## PipelineBuilder

**Module:** `src/cli/pipeline.py`

Fluent builder for constructing a configured `VerificationPipeline`.

```python
from src.cli.pipeline import PipelineBuilder

pipeline = (PipelineBuilder()
    .with_timeout(30.0)
    .with_loop_bound(64)
    .with_progress(lambda phase, status: print(f"{phase}: {status}"))
    .build())

report = pipeline.verify(c_source, rust_source)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `with_config(config)` | `PipelineBuilder` | Set a `VerifyConfig` object |
| `with_profile(profile)` | `PipelineBuilder` | Use a named configuration profile |
| `skip_phase(phase)` | `PipelineBuilder` | Skip a specific pipeline phase |
| `with_progress(callback)` | `PipelineBuilder` | Register a progress callback |
| `with_timeout(total)` | `PipelineBuilder` | Set total pipeline timeout in seconds |
| `with_loop_bound(bound)` | `PipelineBuilder` | Set loop unrolling bound K |
| `build()` | `VerificationPipeline` | Build the configured pipeline |

---

## CEGAR Engine

**Module:** `src/cegar_engine.py`

Iterative LLM translation with verification feedback. Translates C→Rust using an LLM, verifies with the oracle, feeds counterexamples back as repair hints, and repeats until convergence or max iterations.

### Constructor

```python
CEGAREngine(model="gpt-4.1-nano", max_iterations=5, timeout_ms=10000, api_key=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4.1-nano"` | LLM model for translation |
| `max_iterations` | `int` | `5` | Maximum CEGAR iterations |
| `timeout_ms` | `int` | `10000` | Z3 solver timeout in ms |
| `api_key` | `str` | `None` | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |

### `engine.run(c_code, func_name="func") → CEGARResult`

Run the full CEGAR loop on a single function.

```python
from src.cegar_engine import CEGAREngine

engine = CEGAREngine(model="gpt-4.1-nano", max_iterations=5)
result = engine.run("int max2(int a, int b) { return a > b ? a : b; }", "max2")
print(result.converged)        # True
print(result.total_iterations) # 1
print(result.final_verdict)    # "equivalent"
```

### `engine.run_batch(pairs, progress_callback=None) → List[CEGARResult]`

Run CEGAR on a batch of functions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pairs` | `List[Tuple[str, str]]` | List of `(func_name, c_code)` pairs |
| `progress_callback` | `callable` | Optional `fn(i, total, result)` callback |

---

## CLI Commands

SemRec provides two CLI entry points: the newer `src/cli/main.py` and the legacy `src/semrec_cli.py`. Both are accessible via the `semrec` command.

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0` | Equivalent |
| `2` | Divergent |
| `3` | Unknown / error |

### Global Flags

```
--version            Show version (semrec 0.2.0)
-v, --verbose        Increase verbosity (can repeat: -vv)
-q, --quiet          Suppress all output
```

### `semrec verify`

Verify equivalence of a C/Rust function pair.

```bash
semrec verify [OPTIONS] [C_FILE] [RUST_FILE]
```

| Option | Description |
|--------|-------------|
| `--c-file PATH` | Path to C source file |
| `--c-code STRING` | Inline C code (alternative to file) |
| `--rs-file PATH` | Path to Rust source file |
| `--rs-code STRING` | Inline Rust code (alternative to file) |
| `--function, -f NAME` | Function name (auto-inferred if omitted) |
| `--c-function NAME` | C function name (if different from Rust) |
| `--rust-function NAME` | Rust function name (if different from C) |
| `--timeout MS` | Z3 solver timeout in ms (default: 10000) |
| `--smt-timeout MS` | Separate SMT-specific timeout |
| `--loop-bound K` | Loop unrolling bound (default: 32) |
| `--format json\|text\|html` | Output format (default: json) |
| `--output, -o PATH` | Write result to file |
| `--report-dir PATH` | Directory for HTML report artifacts |
| `--no-fuzz` | Skip differential fuzzing phase |

**Examples:**

```bash
# Verify from files
semrec verify --c-file add.c --rs-file add.rs

# Verify inline code
semrec verify --c-code 'int f(int x) { return x + 1; }' \
              --rs-code 'pub fn f(x: i32) -> i32 { x.wrapping_add(1) }'

# With custom timeout and loop bound
semrec verify --c-file sort.c --rs-file sort.rs --timeout 30000 --loop-bound 64

# Text output
semrec verify --c-file f.c --rs-file f.rs --format text
```

**Text output:**

```
Verdict: divergent
Counterexample: Divergence: output_mismatch
  Inputs: {'input_0': '0'}
  Class: other
Repair: Semantic divergence detected: output_mismatch
Time: 151.2ms
```

**JSON output:**

```json
{
  "verdict": "divergent",
  "counterexample": {
    "inputs": {"input_0": "0"},
    "reason": "output_mismatch",
    "divergence_class": "other"
  },
  "repair_hint": {
    "description": "Semantic divergence detected",
    "fix_category": "review"
  },
  "time_ms": 151.2,
  "func_name": "f"
}
```

### `semrec fuzz`

Run differential fuzzing only (no formal verification).

```bash
semrec fuzz [OPTIONS] C_FILE RUST_FILE
```

| Option | Description |
|--------|-------------|
| `--function, -f NAME` | Function name |
| `--iterations N` | Number of fuzzing iterations |
| `--seed-count N` | Number of seed inputs |
| `--timeout MS` | Fuzzing timeout in ms |
| `--coverage-target FLOAT` | Stop when coverage target reached (0.0–1.0) |
| `--output, -o PATH` | Write results to file |

**Example:**

```bash
semrec fuzz --function add --iterations 10000 --seed-count 100 add.c add.rs
```

### `semrec analyze`

Run static analysis passes on a source pair without verification.

```bash
semrec analyze [OPTIONS] C_FILE RUST_FILE
```

| Option | Description |
|--------|-------------|
| `--output, -o PATH` | Write analysis results to file |
| `--dot` | Output control flow graph in DOT format |
| `--no-alias` | Skip alias analysis |
| `--interprocedural` | Enable interprocedural analysis (experimental) |

**Example:**

```bash
semrec analyze --dot --output analysis.json module.c module.rs
```

### `semrec benchmark`

Run verification on a benchmark suite.

```bash
semrec benchmark [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--suite NAME` | Benchmark suite name *(required)* |
| `--output, -o PATH` | Write results to file |
| `--timeout MS` | Per-pair timeout in ms |
| `--parallel N` | Number of parallel workers |
| `--filter PATTERN` | Filter benchmark pairs by name |

**Example:**

```bash
semrec benchmark --suite core --output results.json --parallel 4
```

### `semrec cegar`

Run CEGAR translation loop (LLM + oracle). Requires `OPENAI_API_KEY` environment variable.

```bash
semrec cegar [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--source-file PATH` | C source file to translate |
| `--source-code STRING` | Inline C code (alternative to file) |
| `--model MODEL` | LLM model (default: gpt-4.1-nano) |
| `--max-iter N` | Max CEGAR iterations (default: 5) |
| `--timeout MS` | Z3 solver timeout in ms (default: 10000) |
| `--function, -f NAME` | Function name |
| `--output, -o PATH` | Write result to file |

**Example:**

```bash
export OPENAI_API_KEY=sk-...
semrec cegar --source-code 'int max2(int a, int b) { return a > b ? a : b; }'
```

### `semrec bench` (legacy)

Legacy benchmark command via `src/semrec_cli.py`.

```bash
semrec bench [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--output, -o PATH` | Write results to file |
| `--pairs N` | Number of pairs (0 = all) |
| `--category NAME` | Filter by benchmark category |
| `--cegar` | Also run CEGAR evaluation |
| `--model MODEL` | LLM model for CEGAR (default: gpt-4.1-nano) |
| `--max-iter N` | Max CEGAR iterations (default: 5) |

### `semrec discover`

Auto-discover matching C/Rust function pairs from build systems.

```bash
semrec discover [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--cargo-dir PATH` | Path to Rust project directory containing Cargo.toml *(required)* |
| `--c-dir PATH` | Path to C source directory *(required)* |
| `--compile-commands PATH` | Path to compile_commands.json (uses `--c-dir` if omitted) |
| `--output, -o PATH` | Write results to file |
| `--format text\|json` | Output format (default: text) |

**Examples:**

```bash
# Discover matching function pairs
semrec discover --cargo-dir my_project/ --c-dir my_project/c_src/

# JSON output with compile_commands.json
semrec discover --cargo-dir my_project/ --c-dir my_project/c_src/ \
                --compile-commands build/compile_commands.json --format json
```

### Project Discovery Flags on `verify`

The `verify` subcommand also accepts `--cargo-dir` and `--compile-commands` to list discovered functions before verification:

| Option | Description |
|--------|-------------|
| `--cargo-dir PATH` | Scan Rust project for `#[no_mangle]` / `extern "C"` FFI functions |
| `--compile-commands PATH` | Read compile_commands.json to discover C source files and functions |

---

## Configuration

**Module:** `src/cli/config.py`

### VerifyConfig

Top-level configuration for the verification pipeline.

```python
@dataclass
class VerifyConfig:
    output_format: OutputFormat = OutputFormat.JSON
    verbosity: Verbosity = Verbosity.NORMAL
    output_file: Optional[str] = None
    report_dir: Optional[str] = None
    c_function: str = ""
    rust_function: str = ""
    match_by_name: bool = True
    loops: LoopConfig = LoopConfig()
    timeouts: TimeoutConfig = TimeoutConfig()
    fuzzer: FuzzerConfig = FuzzerConfig()
    frontend: FrontendConfig = FrontendConfig()
    symbolic: SymbolicExecConfig = SymbolicExecConfig()
    smt: SMTConfig = SMTConfig()
    analysis: AnalysisConfig = AnalysisConfig()
```

### SemanticConfig (σ-bridge)

**Module:** `src/semantics/semantic_config.py`

Encodes language-specific semantics. The σ-bridge creates separate `SemanticConfig` instances for C and Rust, capturing differences like overflow behavior.

```python
from src.semantics.semantic_config import SemanticConfig, OverflowMode

c_config = SemanticConfig.c11_default()     # overflow=UB, shift=UB, etc.
r_config = SemanticConfig.rust_default()    # overflow=Wrap (release), etc.

diff = c_config.diff(r_config)              # ConfigDiff listing all differences
```

**Key enums:**

| Enum | Values | Description |
|------|--------|-------------|
| `OverflowMode` | `Wrap`, `Panic`, `UB`, `Saturate` | Signed overflow behavior |
| `FloatModel` | `IEEE754Strict`, `FastMath`, `StrictFinite` | Float semantics |
| `ShiftModel` | — | Shift-by-width behavior |
| `DivisionModel` | — | Division-by-zero behavior |
| `PointerModel` | — | Pointer arithmetic model |
| `ArrayBoundsModel` | — | Array bounds checking |
| `LayoutModel` | — | Struct layout/padding |
| `FloatToIntModel` | — | Float-to-int conversion |

---

## Data Classes

### OracleResult

**Module:** `src/oracle/oracle.py`

```python
@dataclass
class OracleResult:
    verdict: str                              # "equivalent" | "divergent" |
                                              # "conditionally_equivalent" | "unknown" | "error"
    counterexample: Optional[CounterexampleInfo] = None
    repair_hint: Optional[RepairHint] = None
    time_ms: float = 0.0                      # Wall-clock verification time
    smt_queries: int = 0                      # Number of Z3 queries issued
    pipeline_stages: Dict[str, bool] = {}     # Stage → success
    error_msg: Optional[str] = None           # Error message if verdict == "error"
    func_name: str = ""
    confidence: float = 1.0                   # 1.0 for definitive, 0.5 for unknown
```

### CounterexampleInfo

```python
@dataclass
class CounterexampleInfo:
    inputs: Dict[str, str] = {}               # e.g. {"input_0": "2147483647", "input_1": "-1"}
    c_behavior: Optional[str] = None          # C output or "undefined"
    rust_behavior: Optional[str] = None       # Rust output value
    reason: str = ""                          # e.g. "c_undefined_behavior (assumption 0 violated)"
    divergence_class: str = ""                # "undefined_behavior", "overflow", "division",
                                              # "shift", "cast", "other"
```

### RepairHint

```python
@dataclass
class RepairHint:
    description: str = ""                     # Human-readable description
    suggested_fix: str = ""                   # Suggested code change
    fix_category: str = ""                    # "wrapping_op" | "checked_op" | "guard" |
                                              # "cast" | "review"
    confidence: float = 0.0                   # Hint confidence (0.0–1.0)
```

### VerificationReport

**Module:** `src/cli/reporter.py`

```python
@dataclass
class VerificationReport:
    verdict: EquivalenceVerdict               # Enum: EQUIVALENT, DIVERGENT, UNKNOWN, ERROR
    c_source: str = ""
    rust_source: str = ""
    c_function: str = ""
    rust_function: str = ""
    counterexamples: List[Counterexample] = []
    coverage: CoverageSummary                 # Path and branch coverage info
    divergence_summary: DivergenceSummary     # Summary of detected divergences
    timing: TimingInfo                        # Per-phase timing breakdown
    warnings: List[str] = []
    ir_stats: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    timestamp: str = ""
```

### CEGARResult

```python
@dataclass
class CEGARResult:
    func_name: str                            # Function being translated
    c_code: str                               # Original C source
    converged: bool                           # True if oracle returned "equivalent"
    final_verdict: str                        # "equivalent" | "divergent" | "error" | "unknown"
    iterations: List[CEGARIteration] = []     # Per-iteration details
    total_iterations: int = 0
    total_time_ms: float = 0.0
    bug_class: str = ""                       # e.g. "undefined_behavior"
    llm_repairable: bool = False              # Whether LLM successfully repaired
    repair_iterations: int = 0                # Iterations needed to fix (0 if not repaired)
```

### CEGARIteration

```python
@dataclass
class CEGARIteration:
    iteration: int                            # 0-indexed iteration number
    rust_code: str                            # Rust translation produced by LLM
    verdict: str                              # Oracle verdict for this iteration
    counterexample: Optional[Dict[str, Any]] = None
    repair_hint: Optional[str] = None         # Repair hint fed to LLM
    divergence_class: str = ""
    time_ms: float = 0.0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
```

---

## Output Formats

SemRec supports three output formats:

| Format | Method | Description |
|--------|--------|-------------|
| **JSON** | `report.to_json()` | Machine-readable structured output |
| **Text** | `report.format_terminal()` | Human-readable terminal output |
| **HTML** | `report.format_html()` | Rich report with syntax highlighting |

Select via `--format json|text|html` on the CLI, or `OutputFormat` in `VerifyConfig`.

---

## Error Handling

### Pipeline Errors

The pipeline catches errors at each phase and reports them in `OracleResult.pipeline_stages`:

```python
result.pipeline_stages
# {"c_parse": True, "rust_parse": True, "c_ir": False, ...}
```

If a critical phase fails, the verdict is `"error"` with details in `result.error_msg`.

### Exception Types

| Exception | When |
|-----------|------|
| `RuntimeError` | Parse failure (`"C parsing failed"`), IR lowering failure |
| `ValueError` | Configuration validation errors |
| `ImportError` | Optional dependency not installed (e.g., `tree-sitter`) |

### Fallback Strategies

- **Parser fallback:** If tree-sitter fails, the hand-written parser is used automatically.
- **Verification fallback:** If product program construction fails, the oracle attempts direct SMT comparison or structural verification.
- **Graceful degradation:** Pipeline phases that fail non-critically produce warnings rather than errors.

---

## Tree-Sitter Parsers

### C Parser

**Module:** `src/frontend_c/tree_sitter_parser.py`

```python
from src.frontend_c.tree_sitter_parser import TreeSitterCParser

parser = TreeSitterCParser("int add(int a, int b) { return a + b; }")
ast = parser.parse()  # Returns TranslationUnit
print(ast.declarations[0].name)  # "add"
```

### Rust Parser

**Module:** `src/frontend_rust/tree_sitter_parser.py`

```python
from src.frontend_rust.tree_sitter_parser import TreeSitterRustParser

parser = TreeSitterRustParser("fn add(a: i32, b: i32) -> i32 { a + b }")
ast = parser.parse()  # Returns Crate
print(ast.items[0].name)  # "add"
```

Both parsers fall back to hand-written parsers on conversion errors. The oracle automatically prefers tree-sitter parsers.

---

## SMT Encoder — Struct/Enum Support

**Module:** `src/smt/encoder.py`

### Struct Encoding

Structs are encoded as flat bitvectors with field access via `Extract`/`Concat`.

```python
from src.smt.encoder import SMTEncoder, EncodingContext
from src.ir.types import StructType, StructField, IntType

encoder = SMTEncoder()
ctx = EncodingContext()

st = StructType("Point", (
    StructField("x", IntType(32)),
    StructField("y", IntType(32)),
))

sort = encoder.encode_type(st)  # BitVecSort(64)

import z3
fields = [z3.BitVecVal(10, 32), z3.BitVecVal(20, 32)]
struct_bv = encoder.encode_struct_literal(st, fields, ctx)
```

### Enum Encoding

Enums are encoded as tagged bitvectors: discriminant tag + max-payload.

```python
from src.ir.types import EnumType, VoidType

et = EnumType("Option", (
    ("Some", IntType(32)),
    ("None", VoidType()),
))

some_val = encoder.encode_enum_construct(et, "Some", z3.BitVecVal(42, 32), ctx)
none_val = encoder.encode_enum_construct(et, "None", None, ctx)
tag = encoder.encode_enum_discriminant(some_val, et)
```

### EnumType Properties and Methods

| Property/Method | Type | Description |
|----------------|------|-------------|
| `num_variants` | `int` | Number of variants |
| `variant_names` | `tuple[str, ...]` | Ordered variant names |
| `is_c_like` | `bool` | True if all variants have VoidType payload |
| `tag_width` | `int` | Discriminant bit-width (0=auto) |
| `variant_index(name)` | `int` | Index of variant by name |
| `variant_type(name)` | `IRType` | Payload type of variant |
| `size_bits()` | `int` | Total size including tag + max payload + alignment |
| `to_dict()` | `dict` | Serialization for JSON |

### SMT Instructions for Structs/Enums

| Instruction | SMT Encoding | Description |
|-------------|-------------|-------------|
| `ExtractValueInst` | `z3.Extract(hi, lo, bv)` | Extract struct field by bit offset |
| `InsertValueInst` | `Concat(prefix, new_val, suffix)` | Replace field bits in struct |
| `SwitchInst` | ITE chain over case equality | Path conditions for case targets |

---

## Enhanced Memory Model

**Module:** `src/smt/points_to_analysis.py`

```python
from src.smt.points_to_analysis import EnhancedMemoryModel

model = EnhancedMemoryModel()
constraints = model.analyze(c_ir, rust_ir)
# Returns additional SMT constraints for pointer reasoning
```

**Components:**

| Component | Description |
|-----------|-------------|
| `PointsToAnalysis` | Andersen-style flow-insensitive points-to analysis |
| `OwnershipAxiomEncoder` | Non-aliasing constraints from Rust `&mut` references |
| `TBAAEncoder` | Type-based alias analysis (C strict aliasing rule) |

---

## Benchmarks

**Module:** `benchmarks/`

212 benchmark pairs across 18 categories.

```python
from benchmarks.pairs import COMBINED_BENCHMARKS, get_expanded_by_category

all_pairs = COMBINED_BENCHMARKS  # All 232 pairs

struct_pairs = get_expanded_by_category("struct")       # 20 pairs
enum_pairs = get_expanded_by_category("enum")           # 20 pairs
float_pairs = get_expanded_by_category("float")         # 15 pairs
c2rust_pairs = get_expanded_by_category("c2rust")       # 20 pairs
iter_pairs = get_expanded_by_category("iterator")       # 15 pairs
```

**Categories:** `struct`, `enum`, `float`, `c2rust`, `c2rust_realistic`, `iterator`, `cast`, `compound`, `control_flow`, `memory`, `scaled_memory`

---

## Product Program Soundness

**Module:** `src/product_program/soundness.py`

```python
from src.product_program.soundness import verify_coercion_soundness, format_proof_appendix

results = verify_coercion_soundness()
# Returns list of (coercion_name, status, details) tuples

latex = format_proof_appendix()
# Returns LaTeX string for paper appendix
```

Provides formal verification of σ-bridge coercion correctness via boundary-value testing on 7 coercion specifications.
