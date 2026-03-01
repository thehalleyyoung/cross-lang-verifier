#!/usr/bin/env python3
"""
Property-Based Differential Testing Baseline for SemRec.

This implements a differential testing baseline using property-based testing
(hypothesis-style) to compare against SemRec's REAL pipeline verification.
The baseline:
  1. Generates random inputs including boundary values
  2. Simulates C and Rust semantics in Python
  3. Compares outputs
  4. Reports divergences found

SemRec column uses the real pipeline (run_pipeline) which goes through:
  CParser → CIRLowering → RustParser → RustIRLowering → ProductBuilder → Z3 SMT

This addresses the reviewer critique: "compare against the simplest possible
baseline: differential testing with property-based testing frameworks."
"""

import json
import os
import sys
import time
import random
from dataclasses import asdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementation"))

from pipeline_verify import run_pipeline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
UINT32_MAX = 2**32 - 1

# Boundary values for i32 (signed 32-bit)
BOUNDARY_I32 = [
    INT32_MIN, INT32_MIN + 1, INT32_MIN + 2,
    -2, -1, 0, 1, 2,
    INT32_MAX - 2, INT32_MAX - 1, INT32_MAX,
]

# Boundary values for u32 (unsigned 32-bit)
BOUNDARY_U32 = [0, 1, 2, UINT32_MAX - 2, UINT32_MAX - 1, UINT32_MAX]


# Same benchmark set as the CEGAR experiment
BENCHMARK_PAIRS = [
    {
        "name": "safe_add",
        "c_code": "int safe_add(int a, int b) { return a + b; }",
        "rust_code": "pub fn safe_add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }",
        "params": [("a", "i32"), ("b", "i32")],
        "expected": "divergent",
    },
    {
        "name": "negate",
        "c_code": "int negate(int x) { return -x; }",
        "rust_code": "pub fn negate(x: i32) -> i32 { x.wrapping_neg() }",
        "params": [("x", "i32")],
        "expected": "divergent",
    },
    {
        "name": "abs_diff",
        "c_code": """int abs_diff(int a, int b) {
    int diff = a - b;
    return diff < 0 ? -diff : diff;
}""",
        "rust_code": """pub fn abs_diff(a: i32, b: i32) -> i32 {
    let diff = a.wrapping_sub(b);
    if diff < 0 { diff.wrapping_neg() } else { diff }
}""",
        "params": [("a", "i32"), ("b", "i32")],
        "expected": "divergent",
    },
    {
        "name": "average",
        "c_code": "int average(int a, int b) { return (a + b) / 2; }",
        "rust_code": "pub fn average(a: i32, b: i32) -> i32 { a.wrapping_add(b) / 2 }",
        "params": [("a", "i32"), ("b", "i32")],
        "expected": "divergent",
    },
    {
        "name": "clamp",
        "c_code": """int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}""",
        "rust_code": """pub fn clamp(val: i32, lo: i32, hi: i32) -> i32 {
    if val < lo { lo } else if val > hi { hi } else { val }
}""",
        "params": [("val", "i32"), ("lo", "i32"), ("hi", "i32")],
        "expected": "equivalent",
    },
    {
        "name": "rotate_right",
        "c_code": """unsigned int rotate_right(unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}""",
        "rust_code": "pub fn rotate_right(x: u32, n: u32) -> u32 { x.rotate_right(n) }",
        "params": [("x", "u32"), ("n", "u32")],
        "expected": "divergent",
    },
    {
        "name": "sign",
        "c_code": """int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}""",
        "rust_code": """pub fn sign(x: i32) -> i32 {
    if x > 0 { 1 } else if x < 0 { -1 } else { 0 }
}""",
        "params": [("x", "i32")],
        "expected": "equivalent",
    },
    {
        "name": "safe_divide",
        "c_code": """int safe_divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}""",
        "rust_code": """pub fn safe_divide(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a / b }
}""",
        "params": [("a", "i32"), ("b", "i32")],
        "expected": "divergent",  # INT_MIN / -1
    },
    {
        "name": "midpoint",
        "c_code": "int midpoint(int a, int b) { return a + (b - a) / 2; }",
        "rust_code": "pub fn midpoint(a: i32, b: i32) -> i32 { a.wrapping_add((b.wrapping_sub(a)) / 2) }",
        "params": [("a", "i32"), ("b", "i32")],
        "expected": "divergent",
    },
    {
        "name": "saturating_add",
        "c_code": """int saturating_add(int a, int b) {
    long long sum = (long long)a + (long long)b;
    if (sum > 2147483647) return 2147483647;
    if (sum < -2147483648LL) return -2147483648;
    return (int)sum;
}""",
        "rust_code": "pub fn saturating_add(a: i32, b: i32) -> i32 { a.saturating_add(b) }",
        "params": [("a", "i32"), ("b", "i32")],
        "expected": "equivalent",
    },
]


def generate_test_inputs(params: List[Tuple[str, str]], n_random: int = 10000,
                          include_boundaries: bool = True) -> List[Dict[str, int]]:
    """Generate test inputs combining boundary values and random values."""
    inputs = []
    
    if include_boundaries:
        # Generate boundary combinations
        boundary_vals = []
        for pname, ptype in params:
            if ptype in ("i32", "int"):
                boundary_vals.append(BOUNDARY_I32)
            else:
                boundary_vals.append(BOUNDARY_U32)
        
        # Cross-product of boundaries (limited)
        if len(params) == 1:
            for v in boundary_vals[0]:
                inputs.append({params[0][0]: v})
        elif len(params) == 2:
            for v1 in boundary_vals[0]:
                for v2 in boundary_vals[1]:
                    inputs.append({params[0][0]: v1, params[1][0]: v2})
        elif len(params) == 3:
            for v1 in boundary_vals[0][:5]:
                for v2 in boundary_vals[1][:5]:
                    for v3 in boundary_vals[2][:5]:
                        inputs.append({params[0][0]: v1, params[1][0]: v2, params[2][0]: v3})
    
    # Add random values
    for _ in range(n_random):
        inp = {}
        for pname, ptype in params:
            if ptype in ("i32", "int"):
                inp[pname] = random.randint(INT32_MIN, INT32_MAX)
            else:
                inp[pname] = random.randint(0, UINT32_MAX)
        inputs.append(inp)
    
    return inputs


def differential_test_pair(func_name: str, c_code: str, rust_code: str,
                           params: List[Tuple[str, str]],
                           n_random: int = 10000) -> Dict[str, Any]:
    """Run differential testing: generate inputs and simulate both programs.
    
    Since we can't easily compile and run C/Rust in CI, we simulate the
    semantics using Python with proper overflow behavior modeling.
    """
    start = time.time()
    
    inputs = generate_test_inputs(params, n_random=n_random)
    divergences_found = []
    total_tested = 0
    
    for inp in inputs:
        total_tested += 1
        c_result = simulate_c(c_code, func_name, inp)
        rust_result = simulate_rust(rust_code, func_name, inp)
        
        if c_result is not None and rust_result is not None:
            if c_result != rust_result:
                divergences_found.append({
                    "input": inp,
                    "c_result": c_result,
                    "rust_result": rust_result,
                })
        elif c_result == "UB":
            # C has UB, Rust may produce a value — this is a divergence
            if rust_result is not None:
                divergences_found.append({
                    "input": inp,
                    "c_result": "UB",
                    "rust_result": rust_result,
                })
    
    elapsed = (time.time() - start) * 1000
    return {
        "verdict": "divergent" if divergences_found else "equivalent",
        "divergences_found": len(divergences_found),
        "first_divergence": divergences_found[0] if divergences_found else None,
        "total_tested": total_tested,
        "time_ms": elapsed,
    }


def wrap_i32(val: int) -> int:
    """Simulate i32 wrapping arithmetic."""
    val = val & 0xFFFFFFFF
    if val >= 0x80000000:
        val -= 0x100000000
    return val


def has_signed_overflow_add(a: int, b: int) -> bool:
    result = a + b
    return result > INT32_MAX or result < INT32_MIN


def has_signed_overflow_sub(a: int, b: int) -> bool:
    result = a - b
    return result > INT32_MAX or result < INT32_MIN


def has_signed_overflow_mul(a: int, b: int) -> bool:
    result = a * b
    return result > INT32_MAX or result < INT32_MIN


def simulate_c(c_code: str, func_name: str, inputs: Dict[str, int]) -> Any:
    """Simulate C function with UB detection."""
    try:
        vals = list(inputs.values())
        keys = list(inputs.keys())
        
        if func_name == "safe_add":
            a, b = vals[0], vals[1]
            if has_signed_overflow_add(a, b):
                return "UB"
            return a + b
        elif func_name == "negate":
            x = vals[0]
            if x == INT32_MIN:
                return "UB"
            return -x
        elif func_name == "abs_diff":
            a, b = vals[0], vals[1]
            if has_signed_overflow_sub(a, b):
                return "UB"
            diff = a - b
            if diff < 0:
                if diff == INT32_MIN:
                    return "UB"
                return -diff
            return diff
        elif func_name == "average":
            a, b = vals[0], vals[1]
            if has_signed_overflow_add(a, b):
                return "UB"
            return (a + b) // 2
        elif func_name == "clamp":
            val, lo, hi = vals[0], vals[1], vals[2]
            if val < lo:
                return lo
            if val > hi:
                return hi
            return val
        elif func_name == "rotate_right":
            x, n = vals[0], vals[1]
            if n >= 32 or n < 0:
                return "UB"
            if n == 0:
                return x  # 32 - 0 = 32 which is UB for shift
            return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF
        elif func_name == "sign":
            x = vals[0]
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0
        elif func_name == "safe_divide":
            a, b = vals[0], vals[1]
            if b == 0:
                return 0
            if a == INT32_MIN and b == -1:
                return "UB"
            # C truncation toward zero
            if (a < 0) != (b < 0) and a % b != 0:
                return -(abs(a) // abs(b))
            return abs(a) // abs(b) if a >= 0 else -(abs(a) // abs(b))
        elif func_name == "midpoint":
            a, b = vals[0], vals[1]
            if has_signed_overflow_sub(b, a):
                return "UB"
            diff = b - a
            return a + diff // 2
        elif func_name == "saturating_add":
            a, b = vals[0], vals[1]
            s = a + b
            if s > INT32_MAX:
                return INT32_MAX
            if s < INT32_MIN:
                return INT32_MIN
            return s
        return None
    except Exception:
        return None


def simulate_rust(rust_code: str, func_name: str, inputs: Dict[str, int]) -> Any:
    """Simulate Rust function with wrapping semantics."""
    try:
        vals = list(inputs.values())
        
        if func_name == "safe_add":
            return wrap_i32(vals[0] + vals[1])
        elif func_name == "negate":
            return wrap_i32(-vals[0])
        elif func_name == "abs_diff":
            diff = wrap_i32(vals[0] - vals[1])
            if diff < 0:
                return wrap_i32(-diff)
            return diff
        elif func_name == "average":
            s = wrap_i32(vals[0] + vals[1])
            # Python // rounds toward negative infinity, C/Rust truncate toward zero
            if s < 0 and s % 2 != 0:
                return s // 2 + 1
            return s // 2
        elif func_name == "clamp":
            val, lo, hi = vals[0], vals[1], vals[2]
            if val < lo:
                return lo
            if val > hi:
                return hi
            return val
        elif func_name == "rotate_right":
            x, n = vals[0] & 0xFFFFFFFF, vals[1] & 0x1F
            return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF if n > 0 else x
        elif func_name == "sign":
            x = vals[0]
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0
        elif func_name == "safe_divide":
            a, b = vals[0], vals[1]
            if b == 0:
                return 0
            # Rust panics on INT_MIN / -1, but for comparison we note the divergence
            if a == INT32_MIN and b == -1:
                return "PANIC"
            if (a < 0) != (b < 0) and a % b != 0:
                return -(abs(a) // abs(b))
            return abs(a) // abs(b) if a >= 0 else -(abs(a) // abs(b))
        elif func_name == "midpoint":
            a, b = vals[0], vals[1]
            diff = wrap_i32(b - a)
            return wrap_i32(a + diff // 2)
        elif func_name == "saturating_add":
            a, b = vals[0], vals[1]
            s = a + b
            if s > INT32_MAX:
                return INT32_MAX
            if s < INT32_MIN:
                return INT32_MIN
            return s
        return None
    except Exception:
        return None


def run_experiment():
    """Run SemRec pipeline vs differential testing baseline experiment."""
    print("=" * 70)
    print("SemRec Pipeline vs Differential Testing Baseline")
    print("=" * 70)
    print(f"Benchmarks: {len(BENCHMARK_PAIRS)} function pairs")
    print(f"SemRec: real pipeline (CParser → IR → Product → Z3)")
    print(f"Baseline: differential testing (10K random + boundary inputs)")
    print()
    
    results = []
    
    for i, pair in enumerate(BENCHMARK_PAIRS):
        name = pair["name"]
        print(f"\n[{i+1}/{len(BENCHMARK_PAIRS)}] {name}")
        
        # --- SemRec column: real pipeline verification ---
        pipeline_result = run_pipeline(name, pair["c_code"], pair["rust_code"])
        semrec = {
            "verdict": pipeline_result.verdict,
            "time_ms": pipeline_result.time_ms,
            "counterexample": pipeline_result.counterexample,
            "pipeline_stages": pipeline_result.pipeline_stages,
            "pipeline_log": pipeline_result.pipeline_log,
            "smt_queries": pipeline_result.smt_queries,
            "coercion_points": pipeline_result.coercion_points,
            "product_blocks": pipeline_result.product_blocks,
            "alignment_score": pipeline_result.alignment_score,
            "error_msg": pipeline_result.error_msg,
        }
        
        # --- Differential testing column: Python simulation baseline ---
        diff_result = differential_test_pair(
            name, pair["c_code"], pair["rust_code"], pair["params"],
            n_random=10000,
        )
        
        # Determine correctness (treat pipeline_fail/error/unknown as incorrect)
        semrec_verdict = semrec["verdict"]
        semrec_correct = semrec_verdict == pair["expected"]
        diff_correct = diff_result["verdict"] == pair["expected"]
        
        print(f"  SemRec (pipeline):    {semrec_verdict} "
              f"({semrec['smt_queries']} SMT queries, "
              f"{semrec['product_blocks']} product blocks, "
              f"{semrec['time_ms']:.1f}ms)")
        if semrec["error_msg"]:
            print(f"    pipeline note: {semrec['error_msg']}")
        print(f"  Diff testing:         {diff_result['verdict']} "
              f"({diff_result['divergences_found']} divergences in "
              f"{diff_result['total_tested']} tests, "
              f"{diff_result['time_ms']:.1f}ms)")
        
        results.append({
            "name": name,
            "expected": pair["expected"],
            "semrec": semrec,
            "diff_testing": diff_result,
            "semrec_correct": semrec_correct,
            "diff_correct": diff_correct,
        })
    
    # Compute summary
    total = len(results)
    semrec_correct = sum(1 for r in results if r["semrec_correct"])
    diff_correct = sum(1 for r in results if r["diff_correct"])
    semrec_div_found = sum(1 for r in results if r["semrec"]["verdict"] == "divergent")
    diff_div_found = sum(1 for r in results if r["diff_testing"]["verdict"] == "divergent")
    semrec_time = sum(r["semrec"]["time_ms"] for r in results)
    diff_time = sum(r["diff_testing"]["time_ms"] for r in results)
    semrec_pipeline_ok = sum(
        1 for r in results
        if r["semrec"]["verdict"] not in ("pipeline_fail", "error")
    )
    
    # Functions where one method found divergence but the other didn't
    semrec_only = []
    diff_only = []
    for r in results:
        if r["semrec"]["verdict"] == "divergent" and r["diff_testing"]["verdict"] == "equivalent":
            semrec_only.append(r["name"])
        if r["diff_testing"]["verdict"] == "divergent" and r["semrec"]["verdict"] != "divergent":
            diff_only.append(r["name"])
    
    summary = {
        "experiment": "pbt_baseline_comparison",
        "description": "SemRec real pipeline vs differential testing baseline",
        "total_pairs": total,
        "semrec": {
            "correct": semrec_correct,
            "accuracy": semrec_correct / total,
            "divergences_found": semrec_div_found,
            "pipeline_succeeded": semrec_pipeline_ok,
            "total_time_ms": semrec_time,
        },
        "diff_testing": {
            "correct": diff_correct,
            "accuracy": diff_correct / total,
            "divergences_found": diff_div_found,
            "total_time_ms": diff_time,
            "random_samples": 10000,
            "boundary_values": True,
        },
        "semrec_only_found": semrec_only,
        "diff_only_found": diff_only,
        "results": results,
    }
    
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Method':<30} {'Correct':>8} {'Accuracy':>10} {'Divs Found':>12} {'Time':>10}")
    print("-" * 70)
    print(f"{'SemRec (pipeline)':<30} {semrec_correct:>8}/{total} "
          f"{semrec_correct/total:>9.0%} {semrec_div_found:>12} "
          f"{semrec_time:>9.1f}ms")
    print(f"{'Diff Testing (10K + boundary)':<30} {diff_correct:>8}/{total} "
          f"{diff_correct/total:>9.0%} {diff_div_found:>12} "
          f"{diff_time:>9.1f}ms")
    print(f"\nPipeline succeeded on: {semrec_pipeline_ok}/{total} pairs")
    
    if semrec_only:
        print(f"\nSemRec found divergences missed by diff testing: {semrec_only}")
    if diff_only:
        print(f"Diff testing found divergences missed by SemRec: {diff_only}")
    
    outpath = os.path.join(RESULTS_DIR, "pbt_baseline_results.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    
    return summary


if __name__ == "__main__":
    run_experiment()
