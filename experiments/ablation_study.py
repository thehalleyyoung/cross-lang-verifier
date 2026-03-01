#!/usr/bin/env python3
"""
Ablation study for XEquiv: measuring contribution of semantic bridge.

Three configurations:
  A) Full SemRec pipeline via run_pipeline() (σ_C ≠ σ_R with UB detection)
  B) No σ-bridge baseline: direct Z3 with plain BV semantics on both sides
  C) Random testing baseline: differential execution on random inputs

This validates that the semantic bridge (σ parameterization) is responsible
for detecting the IR-invisible divergences.
"""

import json
import os
import sys
import time
import random
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementation"))
import z3

from pipeline_verify import run_pipeline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Benchmark pairs with actual source code ────────────────────────────

PAIRS: List[Dict[str, Any]] = [
    {
        "name": "add_overflow_signed",
        "c_source": "int add(int a, int b) { return a + b; }",
        "rust_source": "pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }",
        "expected_divergent": True,
    },
    {
        "name": "negate_min",
        "c_source": "int neg(int a) { return -a; }",
        "rust_source": "pub fn neg(a: i32) -> i32 { a.wrapping_neg() }",
        "expected_divergent": True,
    },
    {
        "name": "div_by_zero",
        "c_source": "int divide(int a, int b) { if (b == 0) return 0; return a / b; }",
        "rust_source": "pub fn divide(a: i32, b: i32) -> i32 { if b == 0 { 0 } else { a / b } }",
        "expected_divergent": False,
    },
    {
        "name": "max_function",
        "c_source": "int max(int a, int b) { return a > b ? a : b; }",
        "rust_source": "pub fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }",
        "expected_divergent": False,
    },
    {
        "name": "bitwise_and",
        "c_source": "unsigned int mask(unsigned int x, unsigned int m) { return x & m; }",
        "rust_source": "pub fn mask(x: u32, m: u32) -> u32 { x & m }",
        "expected_divergent": False,
    },
    {
        "name": "shift_overflow",
        "c_source": "int shl(int a, int b) { return a << b; }",
        "rust_source": "pub fn shl(a: i32, b: i32) -> i32 { a.wrapping_shl(b as u32) }",
        "expected_divergent": True,
    },
    {
        "name": "mul_overflow",
        "c_source": "int mul(int a, int b) { return a * b; }",
        "rust_source": "pub fn mul(a: i32, b: i32) -> i32 { a.wrapping_mul(b) }",
        "expected_divergent": True,
    },
    {
        "name": "abs_function",
        "c_source": "int abs_val(int x) { return x >= 0 ? x : -x; }",
        "rust_source": "pub fn abs_val(x: i32) -> i32 { if x >= 0 { x } else { x.wrapping_neg() } }",
        "expected_divergent": True,
    },
    {
        "name": "sub_overflow",
        "c_source": "int sub(int a, int b) { return a - b; }",
        "rust_source": "pub fn sub(a: i32, b: i32) -> i32 { a.wrapping_sub(b) }",
        "expected_divergent": True,
    },
    {
        "name": "unsigned_add",
        "c_source": "unsigned int uadd(unsigned int a, unsigned int b) { return a + b; }",
        "rust_source": "pub fn uadd(a: u32, b: u32) -> u32 { a.wrapping_add(b) }",
        "expected_divergent": False,
    },
]


# ── Python simulation lambdas for Config C (random testing) ───────────

_C_SIM = {
    "add_overflow_signed": lambda a, b: ("ub", None) if (a + b > 2**31 - 1 or a + b < -(2**31)) else ("ok", (a + b) & 0xFFFFFFFF),
    "negate_min":          lambda a, b: ("ub", None) if a == -(2**31) else ("ok", (-a) & 0xFFFFFFFF),
    "div_by_zero":         lambda a, b: ("ok", 0) if b == 0 else ("ok", int(a / b)),
    "max_function":        lambda a, b: ("ok", a if a > b else b),
    "bitwise_and":         lambda a, b: ("ok", a & b),
    "shift_overflow":      lambda a, b: ("ub", None) if b >= 32 or b < 0 else ("ok", (a << b) & 0xFFFFFFFF),
    "mul_overflow":        lambda a, b: ("ub", None) if (a * b > 2**31 - 1 or a * b < -(2**31)) else ("ok", (a * b) & 0xFFFFFFFF),
    "abs_function":        lambda a, b: ("ub", None) if a == -(2**31) else ("ok", abs(a)),
    "sub_overflow":        lambda a, b: ("ub", None) if (a - b > 2**31 - 1 or a - b < -(2**31)) else ("ok", (a - b) & 0xFFFFFFFF),
    "unsigned_add":        lambda a, b: ("ok", (a + b) & 0xFFFFFFFF),
}

_RUST_SIM = {
    "add_overflow_signed": lambda a, b: ("ok", (a + b) & 0xFFFFFFFF),
    "negate_min":          lambda a, b: ("ok", (-a) & 0xFFFFFFFF),
    "div_by_zero":         lambda a, b: ("ok", 0) if b == 0 else ("ok", int(a / b)),
    "max_function":        lambda a, b: ("ok", a if a > b else b),
    "bitwise_and":         lambda a, b: ("ok", a & b),
    "shift_overflow":      lambda a, b: ("ok", (a << (b & 31)) & 0xFFFFFFFF),
    "mul_overflow":        lambda a, b: ("ok", (a * b) & 0xFFFFFFFF),
    "abs_function":        lambda a, b: ("ok", (-a) & 0xFFFFFFFF if a < 0 else a),
    "sub_overflow":        lambda a, b: ("ok", (a - b) & 0xFFFFFFFF),
    "unsigned_add":        lambda a, b: ("ok", (a + b) & 0xFFFFFFFF),
}


# ── Config A: Full SemRec pipeline ─────────────────────────────────────

def run_config_a_pipeline(pair: Dict) -> Dict:
    """Config A: Full SemRec pipeline via run_pipeline() (σ_C ≠ σ_R)."""
    result = run_pipeline(pair["name"], pair["c_source"], pair["rust_source"])
    return {
        "verdict": result.verdict,
        "time_ms": result.time_ms,
        "smt_queries": result.smt_queries,
        "counterexample": result.counterexample,
        "pipeline_stages": result.pipeline_stages,
        "coercion_points": result.coercion_points,
        "product_blocks": result.product_blocks,
        "alignment_score": result.alignment_score,
        "error_msg": result.error_msg,
    }


# ── Config B: No σ-bridge (plain BV on both sides) ────────────────────

def run_config_b_no_bridge(pair: Dict) -> Dict:
    """Config B: No σ-bridge — both sides modelled as plain BV, no UB detection."""
    name = pair["name"]
    start = time.time()

    a = z3.BitVec("a", 32)
    b = z3.BitVec("b", 32)

    s = z3.Solver()
    s.set("timeout", 5000)

    # Both C and Rust treated as identical BV operations (no UB/wrap distinction)
    if name in ("add_overflow_signed", "unsigned_add"):
        c_result = a + b
        r_result = a + b
    elif name == "negate_min":
        c_result = -a
        r_result = -a
    elif name == "div_by_zero":
        # Both sides: if b==0 return 0, else sdiv — identical
        c_result = z3.If(b == 0, z3.BitVecVal(0, 32), a / b)
        r_result = z3.If(b == 0, z3.BitVecVal(0, 32), a / b)
    elif name == "max_function":
        c_result = z3.If(a > b, a, b)
        r_result = z3.If(a > b, a, b)
    elif name == "bitwise_and":
        c_result = a & b
        r_result = a & b
    elif name == "shift_overflow":
        c_result = a << b
        r_result = a << b
    elif name == "mul_overflow":
        c_result = a * b
        r_result = a * b
    elif name == "abs_function":
        c_result = z3.If(a >= 0, a, -a)
        r_result = z3.If(a >= 0, a, -a)
    elif name == "sub_overflow":
        c_result = a - b
        r_result = a - b
    else:
        c_result = a
        r_result = a

    s.add(c_result != r_result)
    result = s.check()
    elapsed = (time.time() - start) * 1000

    if result == z3.sat:
        m = s.model()
        cex = {"a": str(m.evaluate(a)), "b": str(m.evaluate(b))}
        return {"verdict": "divergent", "queries": 1, "time_ms": elapsed, "cex": cex}
    return {"verdict": "equivalent", "queries": 1, "time_ms": elapsed}


# ── Config C: Random testing baseline ──────────────────────────────────

def run_config_c_random_testing(pair: Dict, n_samples: int = 10000) -> Dict:
    """Config C: Random testing baseline — differential execution on random inputs."""
    name = pair["name"]
    c_sim = _C_SIM[name]
    r_sim = _RUST_SIM[name]

    start = time.time()
    rng = random.Random(42)
    divergences_found = 0
    first_cex = None

    for _ in range(n_samples):
        a = rng.randint(-(2**31), 2**31 - 1)
        b = rng.randint(-(2**31), 2**31 - 1)
        c_status, c_val = c_sim(a, b)
        r_status, r_val = r_sim(a, b)

        if c_status != r_status or c_val != r_val:
            divergences_found += 1
            if first_cex is None:
                first_cex = {"a": str(a), "b": str(b)}

    elapsed = (time.time() - start) * 1000
    if divergences_found > 0:
        return {"verdict": "divergent", "queries": n_samples,
                "time_ms": elapsed, "divergences": divergences_found,
                "cex": first_cex}
    return {"verdict": "equivalent", "queries": n_samples, "time_ms": elapsed,
            "divergences": 0}


# ── Main ablation runner ───────────────────────────────────────────────

def run_ablation():
    """Run full ablation study across all three configurations."""
    print("=" * 70)
    print("Ablation Study: Semantic Bridge Contribution")
    print("=" * 70)
    print()
    print(f"{'Pair':<25s} {'Config A (Pipeline)':<20s} {'Config B (No-bridge)':<20s} {'Config C (Random)':<18s}")
    print("-" * 83)

    all_results = []
    config_a_correct = 0
    config_b_correct = 0
    config_c_correct = 0
    total = len(PAIRS)

    for pair in PAIRS:
        name = pair["name"]
        expected_div = pair["expected_divergent"]

        a_result = run_config_a_pipeline(pair)
        b_result = run_config_b_no_bridge(pair)
        c_result = run_config_c_random_testing(pair)

        a_div = "divergent" in a_result["verdict"]
        b_div = "divergent" in b_result["verdict"]
        c_div = "divergent" in c_result["verdict"]

        a_correct = a_div == expected_div
        b_correct = b_div == expected_div
        c_correct = c_div == expected_div

        if a_correct: config_a_correct += 1
        if b_correct: config_b_correct += 1
        if c_correct: config_c_correct += 1

        a_mark = "✓" if a_correct else "✗"
        b_mark = "✓" if b_correct else "✗"
        c_mark = "✓" if c_correct else "✗"

        a_v = a_result["verdict"][:18]
        b_v = b_result["verdict"][:18]
        c_v = c_result["verdict"][:15]

        print(f"  {name:<23s} {a_mark} {a_v:<17s} {b_mark} {b_v:<17s} {c_mark} {c_v}")

        all_results.append({
            "name": name,
            "expected_divergent": expected_div,
            "config_a": a_result,
            "config_b": b_result,
            "config_c": c_result,
            "config_a_correct": a_correct,
            "config_b_correct": b_correct,
            "config_c_correct": c_correct,
        })

    print("-" * 83)
    print(f"  {'Accuracy':<23s}   {config_a_correct}/{total}               {config_b_correct}/{total}               {config_c_correct}/{total}")
    print()

    summary = {
        "config_a_accuracy": config_a_correct / total,
        "config_b_accuracy": config_b_correct / total,
        "config_c_accuracy": config_c_correct / total,
        "total_pairs": total,
        "config_a_correct": config_a_correct,
        "config_b_correct": config_b_correct,
        "config_c_correct": config_c_correct,
        "config_a_name": "Full SemRec pipeline (σ_C ≠ σ_R, UB detection)",
        "config_b_name": "No σ-bridge (plain BV semantics, no UB)",
        "config_c_name": "Random testing (10K samples per pair)",
    }

    print(f"Config A (Full pipeline):   {config_a_correct}/{total} = {config_a_correct/total*100:.0f}% accuracy")
    print(f"Config B (No σ-bridge):     {config_b_correct}/{total} = {config_b_correct/total*100:.0f}% accuracy")
    print(f"Config C (Random testing):  {config_c_correct}/{total} = {config_c_correct/total*100:.0f}% accuracy")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results, summary


if __name__ == "__main__":
    run_ablation()
