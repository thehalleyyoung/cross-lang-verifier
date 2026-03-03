# SemRec — Cross-Language Equivalence Verifier

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Z3](https://img.shields.io/badge/solver-Z3%20≥4.12-orange?logo=microsoftazure&logoColor=white)
![Version](https://img.shields.io/badge/version-0.2.0-brightgreen)

SemRec catches **IR-invisible** undefined-behavior divergences in C→Rust
translations via source-level semantic analysis.  When C code is migrated to
Rust, UB divergences — signed overflow, shift UB, division edge cases,
out-of-range casts — produce identical LLVM IR and identical runtime outputs on
x86, so they pass every differential test suite.  SemRec encodes C11 and Rust UB
semantics *separately* through a **σ-bridge** and solves for divergence with Z3,
returning formal counterexamples and repair hints for every bug it finds.

The project ships two CLIs:

| CLI | Entry point | Best for |
|-----|-------------|----------|
| **`semrec`** | `python3 -m src.semrec_cli` | Quick verify / CEGAR / benchmarks |
| **`xlev`** | `python3 -m src.cli.main` | Project-level discovery, fuzz, analyze |

---

## Table of Contents

1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quickstart — Verify from the CLI](#quickstart--verify-from-the-cli)
4. [Detecting UB Divergences (Tested Examples)](#detecting-ub-divergences-tested-examples)
5. [Project-Level Discovery](#project-level-discovery)
6. [Benchmark Suite](#benchmark-suite)
7. [Python API](#python-api)
8. [Full CLI Reference — `semrec`](#full-cli-reference--semrec)
9. [Full CLI Reference — `xlev`](#full-cli-reference--xlev)
10. [Architecture Overview](#architecture-overview)
11. [UB Classes Detected](#ub-classes-detected)
12. [CEGAR Translation Loop](#cegar-translation-loop)
13. [FAQ / Troubleshooting](#faq--troubleshooting)
14. [Citation](#citation)
15. [License](#license)

---

## Key Features

- **6 UB classes detected** — signed overflow, shift UB, division UB,
  float→int out-of-bounds, integer promotion, and negation overflow.
- **σ-bridge encoding** — C11 semantics (overflow = UB) and Rust semantics
  (overflow = wrap/saturate) are encoded independently as SMT assumptions,
  enabling divergence queries invisible to IR-level tools.
- **Z3 counterexamples** — every divergent verdict includes the exact input
  values that trigger the semantic split (e.g., `a = 2147483647, b = 1`).
- **Repair hints** — actionable suggestions for fixing each divergence.
- **Tree-sitter parsing with fallback** — primary parsing via tree-sitter
  grammars for C and Rust; built-in regex-based parsers as fallback.
- **Project-level scanning** — auto-discover matching C/Rust FFI function
  pairs from Cargo projects and `compile_commands.json`.
- **Differential fuzzing** — optional coverage-guided fuzzing phase.
- **CEGAR translation loop** — LLM-powered translate → verify → repair cycle
  for automated C→Rust migration (requires `OPENAI_API_KEY`).
- **52 benchmark pairs across 8 categories** — arithmetic, division, shifts,
  loops, error handling, bitwise, strings, and memory.
- **JSON / text output** — flexible reporting for CI integration or human review.

---

## Installation

```bash
git clone <repo-url>
cd cross-lang-verifier
pip install -e .
```

### Dependencies

| Package | Version | Required |
|---------|---------|:--------:|
| `z3-solver` | ≥ 4.12 | ✅ |
| `tree-sitter` | ≥ 0.22 | recommended |
| `tree-sitter-c` | ≥ 0.21 | recommended |
| `tree-sitter-rust` | ≥ 0.21 | recommended |
| `openai` | ≥ 1.0 | optional (CEGAR mode) |

All required dependencies are installed automatically via `pip install -e .`.

```bash
# Optional: tree-sitter grammars for higher accuracy parsing
pip install tree-sitter>=0.22 tree-sitter-c>=0.21 tree-sitter-rust>=0.21

# Optional: CEGAR mode
pip install openai>=1.0
```

### Verify the installation

```bash
$ python3 -m src.semrec_cli --version
semrec 0.2.0
```

---

## Quickstart — Verify from the CLI

### Check equivalence of two functions (equivalent pair)

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int max(int a, int b){return (a > b) ? a : b;}' \
    --rs-code 'pub fn max(a:i32, b:i32)->i32{if a > b { a } else { b }}' \
    --format text
Verdict: equivalent
Time: 3594.5ms
```

No UB is involved (no arithmetic, no shifts), so the functions are equivalent
on all inputs.

### Detect a signed-overflow divergence

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int f(int x){return x+1;}' \
    --rs-code 'pub fn f(x:i32)->i32{x.wrapping_add(1)}' \
    --format text
Verdict: divergent
Counterexample: Divergence: c_undefined_behavior (assumption 0 violated)
  Inputs: {'input_0': '2147483647'}
  Class: undefined_behavior
Repair: C undefined behavior detected. The C code relies on UB that compilers handle as two's complement.
Time: 1441.0ms
```

Z3 finds that at `x = 2147483647` (INT_MAX), C's `x + 1` is UB, while Rust's
`wrapping_add` wraps to `INT_MIN`.

### Detect a simple output mismatch

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int f(int x){return x+1;}' \
    --rs-code 'pub fn f(x:i32)->i32{x+2}' \
    --format text
Verdict: divergent
Counterexample: Divergence: output_mismatch
  Inputs: {'input_0': '0'}
  Class: other
Repair: Output differs on specific inputs — likely unsigned overflow semantics.
Time: 551.6ms
```

---

## Detecting UB Divergences (Tested Examples)

### 1. Signed overflow (`+` vs `wrapping_add`)

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int add(int a, int b) { return a + b; }' \
    --rs-code 'pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }' \
    --format text
Verdict: divergent
Counterexample: Divergence: c_undefined_behavior (assumption 0 violated)
  Inputs: {'input_0': '-1073741929', 'input_1': '-2147480372'}
  Class: undefined_behavior
Repair: C undefined behavior detected. The C code relies on UB that compilers handle as two's complement.
Time: 4082.8ms
```

### 2. Shift UB (`<<` vs `wrapping_shl`)

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int shl(int x, int n) { return x << n; }' \
    --rs-code 'pub fn shl(x: i32, n: i32) -> i32 { x.wrapping_shl(n as u32) }' \
    --format text
Verdict: divergent
Counterexample: Divergence: c_undefined_behavior (assumption 0 violated)
  Inputs: {'input_0': '0', 'input_1': '32'}
  Class: undefined_behavior
Repair: C undefined behavior detected. The C code relies on UB that compilers handle as two's complement.
Time: 2253.0ms
```

Shifting by the bit-width (32) is UB in C11 but wraps in Rust.

### 3. Negation overflow (`-INT_MIN`)

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int negate(int x) { return -x; }' \
    --rs-code 'pub fn negate(x: i32) -> i32 { x.wrapping_neg() }' \
    --format text
Verdict: divergent
Counterexample: Divergence: c_undefined_behavior (assumption 0 violated)
  Inputs: {'input_0': '-2147483648'}
  Class: undefined_behavior
Repair: C undefined behavior detected. The C code relies on UB that compilers handle as two's complement.
Time: 4772.7ms
```

`-INT_MIN` is UB in C because INT_MIN has no positive counterpart in two's
complement.  Rust's `wrapping_neg()` wraps it back to `INT_MIN`.

### 4. Bitwise AND — equivalent (no UB)

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int band(int a, int b) { return a & b; }' \
    --rs-code 'pub fn band(a: i32, b: i32) -> i32 { a & b }' \
    --format text
Verdict: equivalent
Time: 4008.6ms
```

Bitwise operations have identical semantics in C and Rust — no UB is possible.

### 5. Division with zero guard

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int safe_div(int a, int b){ return b == 0 ? 0 : a / b; }' \
    --rs-code 'pub fn safe_div(a: i32, b: i32) -> i32 { if b == 0 { 0 } else { a.wrapping_div(b) } }' \
    --format text
Verdict: divergent
Counterexample: Divergence: c_undefined_behavior (assumption 0 violated)
  Inputs: {'input_0': '0', 'input_1': '0'}
  Class: undefined_behavior
Repair: C undefined behavior detected. The C code relies on UB that compilers handle as two's complement.
Time: 2650.2ms
```

### 6. Multi-function file verification

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-file examples/sample_project/c_impl/math_utils.c \
    --rs-file examples/sample_project/src/lib.rs \
    --format text
Verdict: likely_equivalent
Time: 6072.4ms
```

When verifying files with multiple functions (`add`, `multiply`, `clamp`),
the oracle analyzes the first matched function and returns a combined verdict.

---

## Project-Level Discovery

The `xlev discover` command scans a Cargo project and a C directory to find
matching FFI function pairs by name.

<!-- tested: actual output captured -->
```bash
$ python3 -m src.cli.main discover \
    --cargo-dir examples/sample_project \
    --c-dir examples/sample_project/c_impl
Rust FFI functions (3):
  add  (examples/sample_project/src/lib.rs:3)
  multiply  (examples/sample_project/src/lib.rs:8)
  clamp  (examples/sample_project/src/lib.rs:13)

C functions (3):
  add  (examples/sample_project/c_impl/math_utils.c:1)
  multiply  (examples/sample_project/c_impl/math_utils.c:5)
  clamp  (examples/sample_project/c_impl/math_utils.c:9)

Matched pairs (3):
  add: examples/sample_project/c_impl/math_utils.c <-> examples/sample_project/src/lib.rs
  clamp: examples/sample_project/c_impl/math_utils.c <-> examples/sample_project/src/lib.rs
  multiply: examples/sample_project/c_impl/math_utils.c <-> examples/sample_project/src/lib.rs
```

The sample project (`examples/sample_project/`) ships with a `compile_commands.json`
for CMake/Clang integration:

```json
[
  {
    "directory": ".",
    "command": "cc -c -o math_utils.o c_impl/math_utils.c",
    "file": "c_impl/math_utils.c"
  }
]
```

---

## Benchmark Suite

SemRec includes 52 curated C/Rust benchmark pairs across 8 categories:

| Category | Pairs | Description |
|----------|:-----:|-------------|
| arithmetic | 12 | Signed overflow, wrapping, checked, saturating |
| division | 5 | Division by zero, `INT_MIN / -1` |
| shift | 3 | Over-shift, negative shift amounts |
| loops | 8 | Bounded loops with accumulation |
| error_handling | 8 | errno vs Result, NULL vs Option |
| bitwise | 8 | AND, OR, XOR, complement |
| string | 2 | String handling patterns |
| memory | 6 | Allocation, deallocation patterns |

### Run the benchmark suite

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli bench --pairs 5
Running 5 benchmark pairs...

==================================================
Benchmark: 5/5 correct (100.0%)
  divergent: 5
==================================================
```

The JSON output includes per-pair details:

<!-- tested: actual output captured (excerpt) -->
```json
{
  "benchmark_results": [
    {
      "name": "add_wrapping",
      "category": "arithmetic",
      "expected": "divergent",
      "actual": "divergent",
      "correct": true,
      "time_ms": 3576.89,
      "divergence_class": "undefined_behavior"
    },
    {
      "name": "mul_overflow",
      "category": "arithmetic",
      "expected": "divergent",
      "actual": "divergent",
      "correct": true,
      "time_ms": 8501.39,
      "divergence_class": "undefined_behavior"
    }
  ],
  "summary": {
    "total": 5,
    "correct": 5,
    "accuracy": 100.0,
    "by_verdict": {
      "equivalent": 0,
      "divergent": 5,
      "unknown": 0,
      "error": 0
    }
  }
}
```

### Filter by category

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli bench --pairs 3 --category arithmetic
Running 3 benchmark pairs...

==================================================
Benchmark: 3/3 correct (100.0%)
  divergent: 3
==================================================
```

### Save results to file

```bash
$ python3 -m src.semrec_cli bench --pairs 10 --output results.json
```

---

## Python API

### `VerificationOracle` — the core interface

```python
import sys; sys.path.insert(0, '.')
from src.oracle.oracle import VerificationOracle

oracle = VerificationOracle()
```

#### Verify a divergent pair

<!-- tested: actual output captured -->
```python
result = oracle.verify(
    'int add(int a, int b) { return a + b; }',
    'pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }',
    'add'
)
print(result.verdict)       # 'divergent'
print(result.smt_queries)   # 2
print(result.counterexample.format_human())
# Divergence: c_undefined_behavior (assumption 0 violated)
#   Inputs: {'input_0': '-1073741929', 'input_1': '-2147480372'}
#   Class: undefined_behavior
print(result.repair_hint.description)
# C undefined behavior detected. The C code relies on UB that
# compilers handle as two's complement.
print(result.repair_hint.suggested_fix[:80])
# Use wrapping arithmetic for ALL signed operations. C UB on overflow means ...
```

#### Verify an equivalent pair

<!-- tested: actual output captured -->
```python
result = oracle.verify(
    'int clamp(int v, int lo, int hi){ if(v<lo) return lo; if(v>hi) return hi; return v; }',
    'pub fn clamp(v:i32, lo:i32, hi:i32)->i32{ if v<lo { lo } else if v>hi { hi } else { v } }',
    'clamp'
)
print(result.verdict)           # 'likely_equivalent'
print(result.counterexample)    # None
```

### `src.api` — high-level convenience functions

```python
from src.api import verify_equivalence, verify_files, batch_verify, quick_check
```

#### `verify_equivalence()`

<!-- tested: actual output captured -->
```python
result = verify_equivalence(
    'int add(int a, int b){ return a+b; }',
    'pub fn add(a:i32, b:i32)->i32{ a.wrapping_add(b) }'
)
print(result.equivalent)    # False
print(result.confidence)    # 1.0
print(result.method)        # 'smt'
print(len(result.divergences))  # 1
```

#### `verify_files()`

<!-- tested: actual output captured -->
```python
result = verify_files(
    'examples/sample_project/c_impl/math_utils.c',
    'examples/sample_project/src/lib.rs'
)
print(result.equivalent)        # False
print(len(result.divergences))  # 1
```

#### `batch_verify()`

<!-- tested: actual output captured -->
```python
results = batch_verify([
    ('int f(int x){return x+1;}', 'pub fn f(x:i32)->i32{x.wrapping_add(1)}'),
    ('int g(int x){return x*2;}', 'pub fn g(x:i32)->i32{x.wrapping_mul(2)}'),
])
for i, r in enumerate(results):
    print(f'Pair {i}: equivalent={r.equivalent}, divergences={len(r.divergences)}')
# Pair 0: equivalent=False, divergences=1
# Pair 1: equivalent=False, divergences=1
```

#### `quick_check()`

```python
print(quick_check(
    'int f(int x){return x+1;}',
    'pub fn f(x:i32)->i32{x.wrapping_add(1)}'
))  # False
```

#### Divergence categories

```python
from src.api import DIVERGENCE_CATEGORIES
print(DIVERGENCE_CATEGORIES)
# ['integer_overflow', 'division_by_zero', 'shift_semantics',
#  'negation_overflow', 'float_precision', 'unsigned_wrap',
#  'pointer_arithmetic', 'array_bounds', 'null_dereference',
#  'type_promotion']
```

---

## Full CLI Reference — `semrec`

The `semrec` CLI is the primary quick-access interface.

<!-- tested: actual output captured -->
```
$ python3 -m src.semrec_cli --help
usage: semrec [-h] [--version] [-v] [-q] {verify,cegar,bench} ...

SemRec: Verification Oracle for Cross-Language Equivalence

positional arguments:
  {verify,cegar,bench}  Available commands
    verify              Verify equivalence of C and Rust functions
    cegar               CEGAR loop: translate C→Rust with LLM + verification
    bench               Run benchmark suite

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v, --verbose
  -q, --quiet

Examples:
  semrec verify --c-file add.c --rs-file add.rs
  semrec verify --c-code 'int f(int x){return x+1;}' --rs-code 'pub fn f(x:i32)->i32{x.wrapping_add(1)}'
  semrec cegar --source-file overflow.c --model gpt-4.1-nano
  semrec bench --output results.json --pairs 200
```

### `semrec verify`

<!-- tested: actual output captured -->
```
$ python3 -m src.semrec_cli verify --help
usage: semrec verify [-h] [--source SOURCE] [--target TARGET] [--c-file C_FILE]
                     [--rs-file RS_FILE] [--c-code C_CODE] [--rs-code RS_CODE]
                     [--function FUNCTION] [--timeout TIMEOUT] [--output OUTPUT]
                     [--format {json,text}] [positional ...]
```

| Flag | Description |
|------|-------------|
| `--c-file` | Path to C source file |
| `--rs-file` | Path to Rust source file |
| `--c-code` | Inline C code string |
| `--rs-code` | Inline Rust code string |
| `--function`, `-f` | Function name to verify |
| `--timeout` | SMT timeout in milliseconds (default: 10000) |
| `--format` | `json` (default) or `text` |
| `--output`, `-o` | Write JSON to file instead of stdout |
| positional | C and Rust files as positional args |

#### JSON output example

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-code 'int max(int a, int b){return (a > b) ? a : b;}' \
    --rs-code 'pub fn max(a:i32, b:i32)->i32{if a > b { a } else { b }}'
```

```json
{
  "verdict": "equivalent",
  "time_ms": 2201.1,
  "smt_queries": 1,
  "func_name": "max",
  "confidence": 0.95,
  "pipeline_stages": {
    "c_parse": true,
    "c_parser_backend": "tree-sitter",
    "rust_parse": true,
    "rust_parser_backend": "tree-sitter",
    "c_ir": true,
    "rust_ir": true,
    "alignment": true,
    "product": true,
    "smt": true,
    "structural": true,
    "structural_verdict": {
      "similarity": 0.85,
      "similar": true,
      "opcode_similarity": 1.0,
      "instruction_similarity": 1.0,
      "return_compatible": false,
      "arg_count_match": true,
      "c_blocks": 4,
      "r_blocks": 4
    },
    "enhanced_memory": true,
    "memory_stats": {
      "ownership_axioms": 0,
      "tbaa_axioms": 0,
      "alias_pairs": 0,
      "non_alias_pairs": 0,
      "c_pts_vars": 0,
      "r_pts_vars": 0
    }
  }
}
```

#### Write output to file

<!-- tested: actual output captured -->
```bash
$ python3 -m src.semrec_cli verify \
    --c-file examples/sample_project/c_impl/math_utils.c \
    --rs-file examples/sample_project/src/lib.rs \
    -o result.json
Result written to result.json
```

### `semrec cegar`

<!-- tested: help output captured -->
```
$ python3 -m src.semrec_cli cegar --help
usage: semrec cegar [-h] [--source-file SOURCE_FILE] [--source-code SOURCE_CODE]
                    [--model MODEL] [--max-iter MAX_ITER] [--timeout TIMEOUT]
                    [--output OUTPUT] [--function FUNCTION]
```

| Flag | Description |
|------|-------------|
| `--source-file` | C source file to translate |
| `--source-code` | Inline C code to translate |
| `--model` | LLM model (default: `gpt-4.1-nano`) |
| `--max-iter` | Max CEGAR iterations (default: 5) |
| `--timeout` | SMT timeout in ms (default: 10000) |
| `--output`, `-o` | Output file (JSON) |
| `--function`, `-f` | Function name |

### `semrec bench`

<!-- tested: help output captured -->
```
$ python3 -m src.semrec_cli bench --help
usage: semrec bench [-h] [--output OUTPUT] [--pairs PAIRS] [--category CATEGORY]
                    [--cegar] [--model MODEL] [--max-iter MAX_ITER]
```

| Flag | Description |
|------|-------------|
| `--pairs` | Number of pairs to run (0 = all 52) |
| `--category` | Filter by category (e.g., `arithmetic`, `shift`) |
| `--cegar` | Also run CEGAR evaluation on divergent pairs |
| `--model` | LLM model for CEGAR (default: `gpt-4.1-nano`) |
| `--output`, `-o` | Output file (JSON) |

---

## Full CLI Reference — `xlev`

The `xlev` CLI provides project-level commands (discover, fuzz, analyze, benchmark).

<!-- tested: actual output captured -->
```
$ python3 -m src.cli.main --help
usage: xlev [-h] [--version] [-v] [-q] [--config CONFIG]
            [--profile {default,fast,thorough,fuzz-only}]
            {verify,discover,fuzz,analyze,benchmark} ...

Cross-Language Equivalence Verifier: verify equivalence of C and Rust function implementations

positional arguments:
  {verify,discover,fuzz,analyze,benchmark}
    verify              Verify equivalence of C and Rust functions
    discover            Auto-discover matching C/Rust function pairs from build systems
    fuzz                Run differential fuzzing only
    analyze             Run analysis passes on source pair
    benchmark           Run verification on a benchmark suite

options:
  --version             show program's version number and exit
  -v, --verbose         Increase verbosity (-v, -vv, -vvv)
  -q, --quiet           Suppress non-essential output
  --config CONFIG       Path to config file (JSON or YAML)
  --profile {default,fast,thorough,fuzz-only}
                        Use a named config profile
```

### `xlev discover`

| Flag | Description |
|------|-------------|
| `--cargo-dir` | Path to Rust project with `Cargo.toml` (required) |
| `--c-dir` | Path to C source directory (required) |
| `--compile-commands` | Path to `compile_commands.json` (optional) |
| `--output`, `-o` | Output file path |
| `--format` | `text` (default) or `json` |

### `xlev verify`

| Flag | Description |
|------|-------------|
| `--c-source`, `-c` | Path to C source file (required) |
| `--rust-source`, `-r` | Path to Rust source file (required) |
| `--function`, `-f` | Function name to verify (both languages) |
| `--c-function` | C function name (overrides `--function`) |
| `--rust-function` | Rust function name (overrides `--function`) |
| `--timeout` | Total timeout in seconds |
| `--smt-timeout` | SMT solver timeout in seconds |
| `--loop-bound` | Loop unrolling bound (default: 32) |
| `--format` | `json` \| `text` \| `html` (default: `json`) |
| `--output`, `-o` | Output file path |
| `--report-dir` | Directory for HTML reports |
| `--no-fuzz` | Disable the fuzzing phase |
| `--cargo-dir` | Path to Rust project to scan for FFI functions |
| `--compile-commands` | Path to `compile_commands.json` for C discovery |

### `xlev fuzz`

| Flag | Description |
|------|-------------|
| `--c-source`, `-c` | Path to C source file (required) |
| `--rust-source`, `-r` | Path to Rust source file (required) |
| `--function`, `-f` | Function name to fuzz |
| `--iterations`, `-n` | Maximum fuzzing iterations |
| `--seed-count` | Number of initial seeds |
| `--timeout` | Fuzzing timeout in seconds |
| `--output`, `-o` | Output file for results |
| `--coverage-target` | Target coverage ratio (0.0–1.0) |

### `xlev analyze`

| Flag | Description |
|------|-------------|
| `--c-source`, `-c` | Path to C source file (required) |
| `--rust-source`, `-r` | Path to Rust source file (required) |
| `--output`, `-o` | Output file for analysis results |
| `--dot` | Output CFG in DOT format to file |
| `--no-alias` | Skip alias analysis |
| `--interprocedural` | Run interprocedural analysis |

### `xlev benchmark`

| Flag | Description |
|------|-------------|
| `--suite`, `-s` | Path to benchmark suite directory (required) |
| `--output`, `-o` | Output file for results |
| `--timeout` | Per-benchmark timeout in seconds |
| `--parallel` | Number of parallel workers (default: 1) |
| `--filter` | Filter benchmarks by name pattern |

### Exit Codes

| Code | Meaning |
|:----:|---------|
| `0` | Equivalent (or command completed successfully) |
| `1` | Error / likely_equivalent for `semrec` |
| `2` | Divergent — UB divergence detected |
| `3` | Unknown — solver timeout |

---

## Architecture Overview

```
                          ┌─────────────────────────────────────────────────────┐
                          │                  SemRec Pipeline                    │
                          │                                                     │
  C source ──┐            │  ┌────────────┐   ┌────────────┐   ┌────────────┐  │
             ├── parse ──→│  │ SSA IR (C)  │──→│  Normalize │──→│            │  │
  .c file ───┘            │  └────────────┘   └────────────┘   │  Product   │  │
                          │        ↑                            │  Builder   │  │
                          │  TreeSitterCParser                  │            │  │
                          │  (fallback: CParser)                │     ↓      │  │
                          │                                     │ ┌────────┐ │  │
                          │  ┌────────────┐   ┌────────────┐   │ │  SMT   │ │  │
                          │  │ SSA IR (R)  │──→│  Normalize │──→│ │Encoder │ │  │
  Rust source ──┐         │  └────────────┘   └────────────┘   │ └───┬────┘ │  │
                ├─ parse →│        ↑                            │     │      │  │
  .rs file ─────┘         │  TreeSitterRustParser               └─────┼──────┘  │
                          │  (fallback: RustParser)                   │         │
                          │                                           ▼         │
                          │  ┌──────────────┐  ┌───────┐  ┌──────────────────┐ │
                          │  │   σ-bridge    │─→│  Z3   │─→│     Verdict      │ │
                          │  │ σ_C: UB       │  │solver │  │ equivalent /     │ │
                          │  │ σ_R: wrap     │  │       │  │ divergent +      │ │
                          │  │ Points-to     │  └───────┘  │ counterexample + │ │
                          │  │ Ownership     │             │ repair hint      │ │
                          │  └──────────────┘              └──────────────────┘ │
                          └─────────────────────────────────────────────────────┘
```

### Pipeline Phases

| # | Phase | Description |
|---|-------|-------------|
| 1 | `parse_c` | Parse C source via tree-sitter (fallback: regex parser) |
| 2 | `parse_rust` | Parse Rust source via tree-sitter (fallback: regex parser) |
| 3 | `lower_c` | Lower C AST to typed SSA IR |
| 4 | `lower_rust` | Lower Rust AST to typed SSA IR |
| 5 | `validate_ir` | Validate well-formedness of both IRs |
| 6 | `analysis` | Run alias analysis, points-to, ownership inference |
| 7 | `normalize` | Canonicalize IR (constant folding, dead code elimination) |
| 8 | `align` | Align C and Rust functions by name and signature |
| 9 | `product` | Build product program combining both IRs |
| 10 | `symbolic` | Symbolic execution over the product program |
| 11 | `smt` | Encode to QF_BV and solve with Z3 |
| 12 | `fuzz` | Differential fuzzing (optional, skippable) |
| 13 | `report` | Generate verdict, counterexample, and repair hint |

### Module Map

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
| `src/oracle/` | Verification oracle — composable API for LLM pipelines |
| `src/cli/` | CLI entry points, configuration, pipeline orchestration |
| `src/utils/` | Source locations, diagnostics, graph utils, bit manipulation |

---

## UB Classes Detected

SemRec's σ-bridge generates SMT constraints for 6 categories of C→Rust
undefined-behavior divergence:

| # | UB Class | C11 Semantics | Rust Semantics | SMT Encoding |
|---|----------|---------------|----------------|--------------|
| 1 | **Signed overflow** | UB — compiler may exploit `nsw` | `wrapping_add/sub/mul` | `Not(overflow_cond)` assumption |
| 2 | **Shift UB** | UB if shift ≥ bit-width or negative | `wrapping_shl/shr` — masks to 5 bits | `ULT(shift, width)` assumption |
| 3 | **Division UB** | UB if divisor = 0 or `INT_MIN / -1` | `wrapping_div` wraps | `Not(div_zero ∨ min_neg1)` assumption |
| 4 | **Float→int OOB** | UB if float outside integer range | Saturating cast (`as i32`) | Direct range-check encoding |
| 5 | **Integer promotion** | Implicit widening before arithmetic | Explicit `as i32` required | Type-width-aware encoding |
| 6 | **Negation overflow** | UB for `-INT_MIN` | `wrapping_neg()` | Overflow check assumption |

**How the σ-bridge works:** For each UB class, the C11 side is encoded as an
*assumption* (the compiler is allowed to assume the UB does not occur).  The Rust
side is encoded as a *concrete bitvector operation* (wrapping, saturating, or
panicking).  Z3 then checks: ∃ inputs where the assumption holds **and** the
outputs differ — or where the assumption is violable.  SAT = divergent with
counterexample.  UNSAT = equivalent.

---

## CEGAR Translation Loop

CEGAR (counterexample-guided abstraction refinement) mode uses an LLM to
translate C→Rust, verifies the result with SemRec, feeds counterexamples back
to the LLM, and iterates until convergence or the iteration limit.

**Requires:** `OPENAI_API_KEY` environment variable.

### CLI usage

```bash
export OPENAI_API_KEY="sk-..."

python3 -m src.semrec_cli cegar \
    --source-code 'int add(int a, int b) { return a + b; }' \
    --function add \
    --model gpt-4.1-nano \
    --max-iter 5
```

### Python API

```python
from src.cegar_engine import CEGAREngine

engine = CEGAREngine(model="gpt-4.1-nano", max_iterations=5, timeout_ms=10000)

result = engine.run(
    "int safe_div(int a, int b) { return b == 0 ? 0 : a / b; }",
    func_name="safe_div"
)

print(result.converged)          # True if LLM produced equivalent Rust
print(result.final_verdict)      # "equivalent" | "divergent"
print(result.total_iterations)   # number of refine iterations
print(result.total_time_ms)      # total wall-clock time

if result.converged:
    print(result.iterations[-1].rust_code)  # final Rust translation
```

### Batch CEGAR

```python
results = engine.run_batch([
    ("checksum", c_checksum_code),
    ("hash", c_hash_code),
])
```

---

## Supported Input Formats

| Format | Extension / Marker | How to Use |
|--------|-------------------|------------|
| C source file | `.c` | `--c-file` (semrec) or `--c-source` (xlev) |
| Rust source file | `.rs` | `--rs-file` (semrec) or `--rust-source` (xlev) |
| Inline C code | — | `--c-code 'int f(int x){return x+1;}'` |
| Inline Rust code | — | `--rs-code 'pub fn f(x:i32)->i32{x+1}'` |
| Cargo project | `Cargo.toml` | `--cargo-dir path/to/project/` |
| CMake / Clang | `compile_commands.json` | `--compile-commands path/to/cc.json` |
| Benchmark suite | directory | `--suite path/to/benchmarks/` |

---

## FAQ / Troubleshooting

### Why does differential testing miss these bugs?

On x86, C signed overflow **wraps** identically to Rust's `wrapping_add`.  The
C compiler's `nsw`/`nuw` flags are LLVM metadata, not runtime behavior.
Differential testing compares outputs, and the outputs are identical for every
concrete input — the divergence is purely *semantic* (one is defined, the other
is UB).  SemRec catches these because the σ-bridge encodes the *specification*
difference, not the runtime behavior.

### What if tree-sitter isn't installed?

SemRec falls back to built-in regex-based parsers (`CParser`, `RustParser`) that
handle the core arithmetic/shift/division/cast fragment.  Tree-sitter provides
higher accuracy on complex syntax (nested macros, attributes, etc.), so it is
recommended for production use.  Install it with:

```bash
pip install tree-sitter>=0.22 tree-sitter-c>=0.21 tree-sitter-rust>=0.21
```

### How do I increase the loop bound?

The default loop unrolling bound is 32 iterations (bounded model checking).  To
increase it:

```bash
# xlev CLI
python3 -m src.cli.main verify -c file.c -r file.rs --loop-bound 128

# Python API
from src.cli.pipeline import PipelineBuilder
pipeline = PipelineBuilder().with_loop_bound(128).build()
```

Higher bounds increase verification time exponentially.  For most scalar UB
patterns, the default of 32 is sufficient.

### Can I verify struct-heavy code?

Struct and enum field access is partially supported.  SemRec works best on
scalar arithmetic, shift, division, and cast operations.  For struct-heavy code,
consider isolating the arithmetic core into standalone functions and verifying
those.

### What does `likely_equivalent` mean?

A `likely_equivalent` verdict means the SMT solver found no divergence but the
pipeline could not prove full equivalence with high confidence.  This typically
happens when multiple functions are present in the source files and structural
similarity is below the threshold.  Try specifying `--function` to target a
specific function.

### What does `unknown` verdict mean?

An `unknown` verdict means the solver could not determine equivalence or
divergence within the timeout.  Try increasing `--timeout` or simplifying
the function under test.

---

## Citation

```bibtex
@misc{semrec2025,
  title   = {{SemRec}: A Verification Oracle for {C}$\to${Rust} Translation},
  author  = {Young, Halley},
  year    = {2025},
  note    = {Tool paper, 18 pages. Available at \texttt{theory/tool\_paper.tex}}
}
```

---

## License

MIT — see [LICENSE](../LICENSE) for details.
