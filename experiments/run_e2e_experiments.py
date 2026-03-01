#!/usr/bin/env python3
"""End-to-end verification experiments using the real pipeline (parse→IR→product→SMT).

Each benchmark pair is run in a fresh subprocess to avoid import caching issues.
Results are saved to experiments/results/e2e_verification_results.json and
experiments/results/e2e_summary.json.
"""

import json
import os
import subprocess
import sys
import textwrap
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SUBPROCESS_TIMEOUT = 30  # seconds

# ── Benchmark pairs ──────────────────────────────────────────────────────────

PAIRS = [
    # Overflow pairs (should be divergent_on_overflow)
    ("add_overflow_signed", "overflow",
     "int add(int a, int b) { return a + b; }",
     "pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }",
     "divergent_on_overflow", "OVF",
     "C: UB on signed overflow; Rust wrapping_add wraps"),

    ("mul_overflow", "overflow",
     "int mul(int a, int b) { return a * b; }",
     "pub fn mul(a: i32, b: i32) -> i32 { a.wrapping_mul(b) }",
     "divergent_on_overflow", "OVF",
     "Multiplication overflow"),

    ("sub_overflow", "overflow",
     "int sub(int a, int b) { return a - b; }",
     "pub fn sub(a: i32, b: i32) -> i32 { a.wrapping_sub(b) }",
     "divergent_on_overflow", "OVF",
     "Subtraction overflow"),

    ("negate_min", "overflow",
     "int neg(int a) { return -a; }",
     "pub fn neg(a: i32) -> i32 { a.wrapping_neg() }",
     "divergent_on_overflow", "OVF",
     "Negating INT_MIN"),

    ("shift_overflow", "overflow",
     "int shl(int a, int b) { return a << b; }",
     "pub fn shl(a: i32, b: i32) -> i32 { a.wrapping_shl(b as u32) }",
     "divergent_on_overflow", "OVF",
     "Shift by >= width"),

    # Equivalent pairs
    ("bitwise_and", "bitwise",
     "unsigned int mask(unsigned int x, unsigned int m) { return x & m; }",
     "pub fn mask(x: u32, m: u32) -> u32 { x & m }",
     "equivalent", None,
     "Bitwise AND, always equivalent"),

    ("bitwise_or", "bitwise",
     "unsigned int combine(unsigned int a, unsigned int b) { return a | b; }",
     "pub fn combine(a: u32, b: u32) -> u32 { a | b }",
     "equivalent", None,
     "Bitwise OR, always equivalent"),

    ("unsigned_add", "arithmetic",
     "unsigned int uadd(unsigned int a, unsigned int b) { return a + b; }",
     "pub fn uadd(a: u32, b: u32) -> u32 { a.wrapping_add(b) }",
     "equivalent", None,
     "Unsigned addition wraps in both"),

    ("int_to_uint_cast", "cast",
     "unsigned int to_uint(int x) { return (unsigned int)x; }",
     "pub fn to_uint(x: i32) -> u32 { x as u32 }",
     "equivalent", None,
     "Same bit pattern"),

    ("widening_cast", "cast",
     "long widen(int x) { return (long)x; }",
     "pub fn widen(x: i32) -> i64 { x as i64 }",
     "equivalent", None,
     "Sign-extending widening"),

    ("truncating_cast", "cast",
     "int trunc(long x) { return (int)x; }",
     "pub fn trunc(x: i64) -> i32 { x as i32 }",
     "equivalent", None,
     "Truncating cast"),

    # Division
    ("div_by_zero", "error_handling",
     "int divide(int a, int b) { return a / b; }",
     "pub fn divide(a: i32, b: i32) -> i32 { a / b }",
     "divergent", "ERR",
     "Division by zero: C UB, Rust panics"),
]

# ── Subprocess worker script ─────────────────────────────────────────────────

WORKER_SCRIPT = textwrap.dedent(r'''
import sys, json, time
sys.path.insert(0, 'implementation')
from src.smt.encoder import SMTEncoder
from src.frontend_c.parser import CParser
from src.frontend_c.ir_lowering import CIRLowering
from src.frontend_rust.parser import RustParser
from src.frontend_rust.ir_lowering import RustIRLowering
from src.frontend_rust.type_resolver import RustTypeResolver
from src.product_program.product import ProductBuilder
from src.product_program.alignment import FunctionAligner
from src.semantics.semantic_config import SemanticConfig
import z3

def verify_pair(c_code, r_code, name):
    stages = {}
    start = time.time()

    # Stage 1: Parse
    c_ast = CParser(c_code, f'{name}.c').parse()
    r_ast = RustParser(r_code, f'{name}.rs').parse()
    stages['parse'] = True

    # Stage 2: IR lowering
    c_mod = CIRLowering().lower(c_ast)
    r_mod = RustIRLowering(RustTypeResolver()).lower(r_ast)
    stages['ir_lowering'] = True

    # Stage 3: Product program
    c_func = list(c_mod.functions.values())[0]
    r_func = list(r_mod.functions.values())[0]
    aligner = FunctionAligner()
    alignment = aligner.align(c_func, r_func)
    builder = ProductBuilder(c_config=SemanticConfig.c11(), rust_config=SemanticConfig.rust_release())
    product = builder.build(c_func, r_func)
    stages['product'] = True
    stages['alignment_score'] = alignment.structural_similarity
    stages['coercion_points'] = product.num_coercion_points

    # Stage 4: SMT encoding via sigma-bridge
    encoder = SMTEncoder()
    result = encoder.encode_sigma_equivalence(c_func, r_func)
    stages['smt_encoding'] = True

    # Stage 5: Z3 solving
    n_queries = 0

    # Query 1: Non-equivalence under well-definedness
    s = z3.Solver()
    s.set('timeout', 10000)
    for a in result['c_assumptions']:
        s.add(a)
    s.add(result['equiv_query'])
    r1 = s.check()
    n_queries += 1

    cex = None
    if r1 == z3.sat:
        m = s.model()
        cex = {n: str(m.evaluate(v, model_completion=True)) for n, v in result['shared_vars'].items()}

    # Query 2: Can C UB occur (overflow)?
    s2 = z3.Solver()
    s2.set('timeout', 10000)
    s2.add(result['overflow_query'])
    r2 = s2.check()
    n_queries += 1

    if r2 == z3.sat and cex is None:
        m = s2.model()
        cex = {n: str(m.evaluate(v, model_completion=True)) for n, v in result['shared_vars'].items()}

    elapsed_ms = (time.time() - start) * 1000

    if r1 == z3.sat:
        verdict = 'divergent'
    elif r2 == z3.sat:
        verdict = 'divergent_on_overflow'
    elif str(r1) == 'unknown':
        verdict = 'unknown'
    else:
        verdict = 'equivalent'

    return {
        'verdict': verdict,
        'counterexample': cex,
        'stages': stages,
        'smt_queries': n_queries,
        'time_ms': elapsed_ms,
        'pipeline': 'e2e',
        'c_ret': str(result.get('c_ret')),
        'r_ret': str(result.get('r_ret')),
    }

if __name__ == '__main__':
    payload = json.loads(sys.argv[1])
    try:
        out = verify_pair(payload['c_code'], payload['r_code'], payload['name'])
        print(json.dumps(out))
    except Exception as exc:
        print(json.dumps({'error': str(exc), 'verdict': 'error', 'pipeline': 'e2e'}))
''')

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_one(name: str, c_code: str, r_code: str) -> dict:
    """Run verification for a single pair in a subprocess."""
    payload = json.dumps({"name": name, "c_code": c_code, "r_code": r_code})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, payload],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT,
            cwd=ROOT,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip().splitlines()
            return {
                "verdict": "error",
                "error": stderr[-1] if stderr else "unknown subprocess error",
                "pipeline": "e2e",
            }
        return json.loads(proc.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"verdict": "timeout", "error": "subprocess timed out", "pipeline": "e2e"}
    except (json.JSONDecodeError, Exception) as exc:
        return {"verdict": "error", "error": str(exc), "pipeline": "e2e"}


def _verdict_symbol(verdict: str) -> str:
    return {"equivalent": "✓", "divergent": "✗", "divergent_on_overflow": "⚠",
            "unknown": "?", "error": "E", "timeout": "T"}.get(verdict, "?")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    correct = 0
    total = len(PAIRS)

    stage_counts = {"parse": 0, "ir_lowering": 0, "product": 0, "smt_encoding": 0}

    header = f"{'#':>2}  {'Name':<25} {'Cat':<12} {'Expected':<22} {'Got':<22} {'Match':>5}  {'Time':>8}  Notes"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for idx, (name, category, c_code, r_code, expected, tag, note) in enumerate(PAIRS, 1):
        result = _run_one(name, c_code, r_code)
        result["name"] = name
        result["category"] = category
        result["expected"] = expected
        result["tag"] = tag
        result["note"] = note
        result["c_code"] = c_code
        result["r_code"] = r_code

        # Track stage successes
        stages = result.get("stages", {})
        for stage in stage_counts:
            if stages.get(stage):
                stage_counts[stage] += 1

        match = result["verdict"] == expected
        if match:
            correct += 1
        sym = _verdict_symbol(result["verdict"])
        time_str = f"{result.get('time_ms', 0):.0f}ms" if "time_ms" in result else "n/a"
        match_str = "  ✓" if match else "  ✗"
        extra = ""
        if result.get("counterexample"):
            cex_short = ", ".join(f"{k}={v}" for k, v in list(result["counterexample"].items())[:3])
            extra = f"  cex: {cex_short}"
        if result.get("error"):
            extra = f"  err: {result['error'][:60]}"

        print(f"{idx:>2}  {name:<25} {category:<12} {expected:<22} {sym} {result['verdict']:<20} {match_str}  {time_str:>8}{extra}")
        results.append(result)

    print(sep)
    print(f"\nResults: {correct}/{total} correct ({correct/total*100:.0f}%)")

    # Stage success rates
    print("\nPipeline stage success rates:")
    for stage, count in stage_counts.items():
        print(f"  {stage:<20} {count}/{total} ({count/total*100:.0f}%)")

    # Category breakdown
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, {"correct": 0, "total": 0})
        cats[c]["total"] += 1
        if r["verdict"] == r["expected"]:
            cats[c]["correct"] += 1
    print("\nCategory breakdown:")
    for c, v in sorted(cats.items()):
        print(f"  {c:<20} {v['correct']}/{v['total']}")

    # Save detailed results
    detail_path = os.path.join(RESULTS_DIR, "e2e_verification_results.json")
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {detail_path}")

    # Save summary
    summary = {
        "total_pairs": total,
        "correct": correct,
        "accuracy_pct": round(correct / total * 100, 1),
        "stage_success": {s: {"count": c, "pct": round(c / total * 100, 1)} for s, c in stage_counts.items()},
        "category_breakdown": cats,
        "verdicts": {},
    }
    for r in results:
        v = r["verdict"]
        summary["verdicts"][v] = summary["verdicts"].get(v, 0) + 1

    summary_path = os.path.join(RESULTS_DIR, "e2e_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
