# SemRec API Reference

Documents only features that are implemented in `src/`.

---

## Verification Oracle (`src/oracle/oracle.py`)

The primary API. Returns structured (verdict, counterexample, repair_hint) triples.

### `VerificationOracle(timeout_ms=10000, func_name="", cegar_mode=False)`

Create a verification oracle instance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout_ms` | `int` | `10000` | Z3 solver timeout in milliseconds |
| `func_name` | `str` | `""` | Default function name for verification |
| `cegar_mode` | `bool` | `False` | When True, UB-only divergences return `"conditionally_equivalent"` instead of `"divergent"` |

### `oracle.verify(c_code, rust_code, func_name=None) → OracleResult`

Verify semantic equivalence of a C/Rust function pair.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_code` | `str` | *(required)* | Complete C function source |
| `rust_code` | `str` | *(required)* | Complete Rust function source |
| `func_name` | `str` | `None` | Function name (auto-inferred if omitted) |

**Returns:** `OracleResult`

```python
from src.oracle.oracle import VerificationOracle

oracle = VerificationOracle()
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

Batch verification of multiple pairs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pairs` | `List[Tuple[str, str, str]]` | List of `(func_name, c_code, rust_code)` triples |

---

## CEGAR Engine (`src/cegar_engine.py`)

Iterative LLM translation with verification feedback.

### `CEGAREngine(model="gpt-4.1-nano", max_iterations=5, timeout_ms=10000, api_key=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4.1-nano"` | LLM model for translation |
| `max_iterations` | `int` | `5` | Maximum CEGAR iterations |
| `timeout_ms` | `int` | `10000` | Z3 solver timeout in ms |
| `api_key` | `str` | `None` | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |

Supported models: `gpt-5-chat-latest`, `gpt-4.1-nano`.

### `engine.run(c_code, func_name="func") → CEGARResult`

Run the full CEGAR loop: LLM translates C→Rust, oracle verifies, counterexample fed back as repair hint, repeat until convergence or max iterations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_code` | `str` | *(required)* | C function source code |
| `func_name` | `str` | `"func"` | Function name |

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

## Data Classes

### `OracleResult`

```python
@dataclass
class OracleResult:
    verdict: str                              # "equivalent" | "divergent" | "conditionally_equivalent" | "unknown" | "error"
    counterexample: Optional[CounterexampleInfo] = None
    repair_hint: Optional[RepairHint] = None
    time_ms: float = 0.0                      # Wall-clock verification time
    smt_queries: int = 0                      # Number of Z3 queries issued
    pipeline_stages: Dict[str, bool] = {}     # Stage → success (c_parse, rust_parse, c_ir, rust_ir, alignment, product, smt)
    error_msg: Optional[str] = None           # Error message if verdict == "error"
    func_name: str = ""
    confidence: float = 1.0                   # 1.0 for definitive, 0.5 for unknown
```

### `CounterexampleInfo`

```python
@dataclass
class CounterexampleInfo:
    inputs: Dict[str, str] = {}               # e.g. {"input_0": "2147483647", "input_1": "-1"}
    c_behavior: Optional[str] = None          # C output or "undefined"
    rust_behavior: Optional[str] = None       # Rust output value
    reason: str = ""                          # e.g. "c_undefined_behavior (assumption 0 violated)"
    divergence_class: str = ""                # Taxonomy: "undefined_behavior", "overflow", "division", "shift", "cast", "other"
```

### `RepairHint`

```python
@dataclass
class RepairHint:
    description: str = ""                     # Human-readable description
    suggested_fix: str = ""                   # Suggested code change
    fix_category: str = ""                    # "wrapping_op" | "checked_op" | "guard" | "cast" | "review"
    confidence: float = 0.0                   # Hint confidence (0.0–1.0)
```

### `CEGARResult`

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
    bug_class: str = ""                       # Classification of original bug (e.g. "undefined_behavior")
    llm_repairable: bool = False              # Whether LLM successfully repaired
    repair_iterations: int = 0                # Iterations needed to fix (0 if not repaired)
```

### `CEGARIteration`

```python
@dataclass
class CEGARIteration:
    iteration: int                            # 0-indexed iteration number
    rust_code: str                            # Rust translation produced by LLM
    verdict: str                              # Oracle verdict for this iteration
    counterexample: Optional[Dict[str, Any]] = None  # Counterexample if divergent
    repair_hint: Optional[str] = None         # Repair hint fed to LLM
    divergence_class: str = ""                # Bug class for this iteration
    time_ms: float = 0.0                      # Verification time
    llm_prompt_tokens: int = 0                # LLM API usage
    llm_completion_tokens: int = 0
```

---

## CLI Commands (`src/semrec_cli.py`)

Entry point: `main(argv: Optional[List[str]] = None) -> int`

### `semrec verify`

Verify equivalence of a C/Rust function pair.

```
--source LANG        Source language (default: C)
--target LANG        Target language (default: Rust)
--c-file PATH        Path to C source file
--c-code STRING      Inline C code (alternative to --c-file)
--rs-file PATH       Path to Rust source file
--rs-code STRING     Inline Rust code (alternative to --rs-file)
--function, -f NAME  Function name (auto-inferred if omitted)
--timeout MS         Z3 solver timeout in ms (default: 10000)
--format json|text   Output format (default: json)
--output, -o PATH    Write result to file
```

**Text output format:**
```
Verdict: divergent
Counterexample: Divergence: output_mismatch
  Inputs: {'input_0': '0'}
  Class: other
Repair: Semantic divergence detected: output_mismatch
Time: 151.2ms
```

**JSON output format:**
```json
{
  "verdict": "divergent",
  "counterexample": {"inputs": {"input_0": "0"}, "reason": "output_mismatch", "divergence_class": "other"},
  "repair_hint": {"description": "Semantic divergence detected", "fix_category": "review"},
  "time_ms": 151.2,
  "func_name": "f"
}
```

### `semrec cegar`

Run CEGAR translation loop (LLM + oracle).

```
--source-file PATH   C source file to translate
--source-code STR    Inline C code (alternative to --source-file)
--model MODEL        LLM model (default: gpt-4.1-nano)
--max-iter N         Max CEGAR iterations (default: 5)
--timeout MS         Z3 solver timeout in ms (default: 10000)
--function, -f NAME  Function name
--output, -o PATH    Write result to file
```

Requires `OPENAI_API_KEY` environment variable.

### `semrec bench`

Run the benchmark suite.

```
--output, -o PATH    Write results to file
--pairs N            Number of pairs (0 = all)
--category NAME      Filter by benchmark category
--cegar              Also run CEGAR evaluation
--model MODEL        LLM model for CEGAR (default: gpt-4.1-nano)
--max-iter N         Max CEGAR iterations (default: 5)
```

### Global flags

```
--version            Show version (semrec 0.2.0)
-v, --verbose        Increase verbosity (can repeat: -vv)
-q, --quiet          Suppress all output
```


---

## SMT Encoder — Struct/Enum Support (`src/smt/encoder.py`)

### Struct Encoding

Structs are encoded as flat bitvectors with field access via `Extract`/`Concat`.

```python
from src.smt.encoder import SMTEncoder, EncodingContext
from src.ir.types import StructType, StructField, IntType

encoder = SMTEncoder()
ctx = EncodingContext()

# Define a struct type
st = StructType("Point", (
    StructField("x", IntType(32)),
    StructField("y", IntType(32)),
))

# Encode type → BitVecSort(64)
sort = encoder.encode_type(st)

# Construct struct literal from field values
import z3
fields = [z3.BitVecVal(10, 32), z3.BitVecVal(20, 32)]
struct_bv = encoder.encode_struct_literal(st, fields, ctx)
```

### Enum Encoding

Enums are encoded as tagged bitvectors: discriminant tag + max-payload.

```python
from src.ir.types import EnumType, VoidType

# Define Rust-style enum
et = EnumType("Option", (
    ("Some", IntType(32)),
    ("None", VoidType()),
))

# Construct variant
some_val = encoder.encode_enum_construct(et, "Some", z3.BitVecVal(42, 32), ctx)
none_val = encoder.encode_enum_construct(et, "None", None, ctx)

# Extract discriminant
tag = encoder.encode_enum_discriminant(some_val, et)
```

### New Instructions

| Instruction | SMT Encoding | Description |
|-------------|-------------|-------------|
| `ExtractValueInst` | `z3.Extract(hi, lo, bv)` | Extract struct field by bit offset |
| `InsertValueInst` | `Concat(prefix, new_val, suffix)` | Replace field bits in struct |
| `SwitchInst` | ITE chain over case equality | Sets path conditions for case targets |

---

## EnumType (`src/ir/types.py`)

New IR type for tagged unions / Rust enums.

```python
from src.ir.types import EnumType, VoidType, IntType, StructType, StructField

# C-like enum (unit variants)
color = EnumType("Color", (
    ("Red", VoidType()),
    ("Green", VoidType()),
    ("Blue", VoidType()),
))
color.is_c_like          # True
color.variant_index("Red")  # 0

# Rust-style data enum
shape = EnumType("Shape", (
    ("Circle", IntType(32)),                    # radius
    ("Rect", StructType("R", (
        StructField("w", IntType(32)),
        StructField("h", IntType(32)),
    ))),
    ("None", VoidType()),
))
shape.is_c_like             # False
shape.variant_type("Circle") # IntType(32)
shape._effective_tag_width() # 8
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_variants` | `int` | Number of variants |
| `variant_names` | `tuple[str, ...]` | Ordered variant names |
| `is_c_like` | `bool` | True if all variants have VoidType payload |
| `tag_width` | `int` | Discriminant bit-width (0=auto) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `variant_index(name)` | `int` | Index of variant by name |
| `variant_type(name)` | `IRType` | Payload type of variant |
| `size_bits()` | `int` | Total size including tag + max payload + alignment |
| `to_dict()` | `dict` | Serialization for JSON |

---

## Expanded Benchmarks (`benchmarks/pairs/`)

212 benchmark pairs across 18 categories.

```python
from benchmarks.pairs import COMBINED_BENCHMARKS, get_expanded_by_category

# All 232 pairs (original + expanded + memory + C2Rust-realistic)
all_pairs = COMBINED_BENCHMARKS

# Filter by category
struct_pairs = get_expanded_by_category("struct")    # 20 pairs
enum_pairs = get_expanded_by_category("enum")        # 20 pairs
float_pairs = get_expanded_by_category("float")      # 15 pairs
c2rust_pairs = get_expanded_by_category("c2rust")    # 20 pairs
iter_pairs = get_expanded_by_category("iterator")    # 15 pairs
```

Categories: `struct`, `enum`, `float`, `c2rust`, `c2rust_realistic`, `iterator`, `cast`, `compound`, `control_flow`, `memory`, `scaled_memory`

---

## Tree-Sitter Parsers (NEW)

### C Parser (`src/frontend_c/tree_sitter_parser.py`)

```python
from src.frontend_c.tree_sitter_parser import TreeSitterCParser

parser = TreeSitterCParser("int add(int a, int b) { return a + b; }")
ast = parser.parse()  # Returns TranslationUnit
print(ast.declarations[0].name)  # "add"
```

### Rust Parser (`src/frontend_rust/tree_sitter_parser.py`)

```python
from src.frontend_rust.tree_sitter_parser import TreeSitterRustParser

parser = TreeSitterRustParser("fn add(a: i32, b: i32) -> i32 { a + b }")
ast = parser.parse()  # Returns Crate
print(ast.items[0].name)  # "add"
```

Both parsers fall back to the hand-written parsers on conversion errors. The oracle (`src/oracle/oracle.py`) automatically prefers tree-sitter parsers.

---

## Enhanced Memory Model (NEW)

### `src/smt/points_to_analysis.py`

```python
from src.smt.points_to_analysis import EnhancedMemoryModel

model = EnhancedMemoryModel()
constraints = model.analyze(c_ir, rust_ir)
# Returns additional SMT constraints for pointer reasoning
```

Components:
- **`PointsToAnalysis`**: Andersen-style flow-insensitive points-to analysis
- **`OwnershipAxiomEncoder`**: Generates non-aliasing constraints from Rust `&mut` references
- **`TBAAEncoder`**: Type-based alias analysis (C strict aliasing rule)

---

## Product Program Soundness (NEW)

### `src/product_program/soundness.py`

```python
from src.product_program.soundness import verify_coercion_soundness, format_proof_appendix

results = verify_coercion_soundness()
# Returns list of (coercion_name, status, details) tuples

latex = format_proof_appendix()
# Returns LaTeX string for paper appendix
```

Provides formal verification of σ-bridge coercion correctness via boundary-value testing on 7 coercion specifications.
