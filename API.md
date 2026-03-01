# SemRec API Reference

Documents only features that are implemented in `implementation/src/`.

---

## Verification Oracle (`implementation/src/oracle/oracle.py`)

The primary API. Returns structured (verdict, counterexample, repair_hint) triples.

### `VerificationOracle(timeout_ms=10000, func_name="")`

Create a verification oracle instance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout_ms` | `int` | `10000` | Z3 solver timeout in milliseconds |
| `func_name` | `str` | `""` | Default function name for verification |

### `oracle.verify(c_code, rust_code, func_name=None) → OracleResult`

Verify semantic equivalence of a C/Rust function pair.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_code` | `str` | *(required)* | Complete C function source |
| `rust_code` | `str` | *(required)* | Complete Rust function source |
| `func_name` | `str` | `None` | Function name (auto-inferred if omitted) |

**Returns:** `OracleResult`

```python
from implementation.src.oracle.oracle import VerificationOracle

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

## CEGAR Engine (`implementation/src/cegar_engine.py`)

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
from implementation.src.cegar_engine import CEGAREngine

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
    verdict: str                              # "equivalent" | "divergent" | "unknown" | "error"
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

## CLI Commands (`implementation/src/semrec_cli.py`)

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
