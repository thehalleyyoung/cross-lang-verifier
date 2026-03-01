#!/usr/bin/env python3
"""
End-to-end experiments for SemRec: cross-language equivalence verification.

Pipeline stages exercised per benchmark:
  1. Parse C and Rust source via real frontends
  2. Lower to shared typed SSA IR via CIRLowering / RustIRLowering
  3. Align functions via FunctionAligner
  4. Construct product program via ProductBuilder
  5. Verify via Z3 bitvector / FP SMT queries with model extraction
  6. Record verdicts, timing, divergence root causes

All counterexamples are extracted from Z3 solver models, never hardcoded.
All results are saved to experiments/results/ as JSON.
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, List, Any

# Add implementation src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementation"))

from src.ir.types import IntType, FloatType, Signedness, FloatKind, OverflowBehavior
from src.ir.instructions import (
    BinaryOp, BinOpKind, CompareOp, CmpPredicate, CastInst, CastKind,
    Constant, Argument, Value, ReturnInst, BranchInst, SelectInst,
)
from src.ir.basic_block import BasicBlock
from src.ir.function import Function
from src.ir.module import Module
from src.ir.builder import IRBuilder
from src.semantics.semantic_config import SemanticConfig
from src.semantics.eval import SemanticEvaluator, EvalResult
from src.smt.solver import SMTSolver, SolverConfig, SolverStatus
from src.smt.encoder import SMTEncoder, EncodingContext
from src.product_program.product import ProductBuilder
from src.product_program.coercion import CoercionGenerator
from src.product_program.alignment import FunctionAligner
from src.type_system.coercions import CoercionChain
from src.frontend_c.parser import CParser
from src.frontend_c.ir_lowering import CIRLowering
from src.frontend_rust.parser import RustParser
from src.frontend_rust.ir_lowering import RustIRLowering
from src.frontend_rust.type_resolver import RustTypeResolver
import z3


def z3_model_value(model, var):
    """Extract a concrete integer value from a Z3 model, handling unconstrained variables."""
    val = model.evaluate(var, model_completion=True)
    if hasattr(val, 'as_signed_long'):
        return val.as_signed_long()
    if hasattr(val, 'as_long'):
        return val.as_long()
    return str(val)


BENCHMARK_PAIRS = []

def add_pair(name, category, c_source, rust_source, expected_verdict,
             divergence_type=None, description=""):
    BENCHMARK_PAIRS.append({
        "name": name,
        "category": category,
        "c_source": c_source,
        "rust_source": rust_source,
        "expected_verdict": expected_verdict,
        "divergence_type": divergence_type,
        "description": description,
    })

# --- Integer overflow pairs ---

add_pair("add_no_overflow", "arithmetic",
    "int add(int a, int b) { return a + b; }",
    "pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="Addition: equivalent when no overflow, divergent on overflow (C UB vs Rust wrap)")

add_pair("add_overflow_signed", "overflow",
    "int add(int a, int b) { return a + b; }",
    "pub fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="C: UB on signed overflow; Rust wrapping_add: wraps")

add_pair("mul_overflow", "overflow",
    "int mul(int a, int b) { return a * b; }",
    "pub fn mul(a: i32, b: i32) -> i32 { a.wrapping_mul(b) }",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="Multiplication overflow: C UB vs Rust wrap")

add_pair("sub_overflow", "overflow",
    "int sub(int a, int b) { return a - b; }",
    "pub fn sub(a: i32, b: i32) -> i32 { a.wrapping_sub(b) }",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="Subtraction overflow semantics differ")

add_pair("negate_min", "overflow",
    "int neg(int a) { return -a; }",
    "pub fn neg(a: i32) -> i32 { a.wrapping_neg() }",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="Negating INT_MIN: C UB, Rust wraps to INT_MIN")

add_pair("shift_overflow", "overflow",
    "int shl(int a, int b) { return a << b; }",
    "pub fn shl(a: i32, b: i32) -> i32 { a.wrapping_shl(b as u32) }",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="Shift by >= width: C UB, Rust wraps shift amount")

# --- Equivalent pairs ---

add_pair("max_function", "control_flow",
    "int max(int a, int b) { return a > b ? a : b; }",
    "pub fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }",
    "equivalent",
    description="Conditional max, equivalent semantics")

add_pair("abs_function", "control_flow",
    "int abs_val(int x) { return x >= 0 ? x : -x; }",
    "pub fn abs_val(x: i32) -> i32 { if x >= 0 { x } else { x.wrapping_neg() } }",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="abs(INT_MIN): C UB, Rust wraps")

add_pair("bitwise_and", "bitwise",
    "unsigned int mask(unsigned int x, unsigned int m) { return x & m; }",
    "pub fn mask(x: u32, m: u32) -> u32 { x & m }",
    "equivalent",
    description="Bitwise AND, always equivalent")

add_pair("bitwise_or", "bitwise",
    "unsigned int combine(unsigned int a, unsigned int b) { return a | b; }",
    "pub fn combine(a: u32, b: u32) -> u32 { a | b }",
    "equivalent",
    description="Bitwise OR, always equivalent")

add_pair("unsigned_add", "arithmetic",
    "unsigned int uadd(unsigned int a, unsigned int b) { return a + b; }",
    "pub fn uadd(a: u32, b: u32) -> u32 { a.wrapping_add(b) }",
    "equivalent",
    description="Unsigned addition wraps in both languages")

# --- Type coercion pairs ---

add_pair("int_to_uint_cast", "cast",
    "unsigned int to_uint(int x) { return (unsigned int)x; }",
    "pub fn to_uint(x: i32) -> u32 { x as u32 }",
    "equivalent",
    description="Signed to unsigned cast, same bit pattern in both")

add_pair("widening_cast", "cast",
    "long widen(int x) { return (long)x; }",
    "pub fn widen(x: i32) -> i64 { x as i64 }",
    "equivalent",
    description="Sign-extending widening cast, equivalent")

add_pair("truncating_cast", "cast",
    "int trunc(long x) { return (int)x; }",
    "pub fn trunc(x: i64) -> i32 { x as i32 }",
    "equivalent",
    description="Truncating cast, both discard high bits")

# --- Division pairs ---

add_pair("div_by_zero", "error_handling",
    "int divide(int a, int b) { return a / b; }",
    "pub fn divide(a: i32, b: i32) -> i32 { a / b }",
    "divergent",
    divergence_type="ERR",
    description="Division by zero: C UB, Rust panics")

add_pair("int_min_div_neg1", "overflow",
    "int divide(int a, int b) { return a / b; }",
    "pub fn divide(a: i32, b: i32) -> i32 { a.wrapping_div(b) }",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="INT_MIN / -1: C UB, Rust wrapping_div wraps")

# --- Floating point pairs ---

add_pair("float_add", "floating_point",
    "double fadd(double a, double b) { return a + b; }",
    "pub fn fadd(a: f64, b: f64) -> f64 { a + b }",
    "divergent",
    divergence_type="FP",
    description="FP addition: C extended precision can produce different rounding than Rust strict IEEE 754")

add_pair("float_mul_accumulate", "floating_point",
    "double fma_like(double a, double b, double c) { return a * b + c; }",
    "pub fn fma_like(a: f64, b: f64, c: f64) -> f64 { a * b + c }",
    "divergent",
    divergence_type="FP",
    description="FP mul-add: C extended precision intermediate can differ from Rust strict rounding")

# --- Loop pairs ---

add_pair("sum_loop", "loop",
    """int sum(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += i;
    return s;
}""",
    """pub fn sum(n: i32) -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        s = s.wrapping_add(i);
        i = i.wrapping_add(1);
    }
    s
}""",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="Loop accumulation: C += is UB on overflow, Rust wraps")

add_pair("factorial", "loop",
    """int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) result *= i;
    return result;
}""",
    """pub fn factorial(n: i32) -> i32 {
    let mut result: i32 = 1;
    let mut i: i32 = 2;
    while i <= n {
        result = result.wrapping_mul(i);
        i = i.wrapping_add(1);
    }
    result
}""",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="Factorial overflow: C UB, Rust wraps")

# --- String encoding pairs (modeled) ---

add_pair("char_check", "string",
    "int is_ascii(char c) { return c >= 0 && c <= 127; }",
    "pub fn is_ascii(c: u8) -> bool { c <= 127 }",
    "equivalent",
    description="C char is signed, but is_ascii result is equivalent at bitvector level")

# --- Memory safety pairs (modeled via types) ---

add_pair("array_access", "memory",
    "int get(int *arr, int idx) { return arr[idx]; }",
    """pub fn get(arr: &[i32], idx: usize) -> i32 { arr[idx] }""",
    "divergent",
    divergence_type="MEM",
    description="C: no bounds check (UB on out-of-bounds); Rust: panics")

# --- More realistic multi-line function pairs ---

add_pair("safe_add_overflow_check", "overflow",
    """int safe_add(int a, int b) {
    int sum = a + b;
    if (sum < a) return 2147483647;
    return sum;
}""",
    """pub fn safe_add(a: i32, b: i32) -> i32 {
    let sum = a.wrapping_add(b);
    if sum < a { return i32::MAX; }
    sum
}""",
    "divergent_on_overflow",
    divergence_type="OVF",
    description="C compiler may optimize away overflow check due to UB; Rust wrapping_add preserves it")

add_pair("clamp_to_range", "control_flow",
    """int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}""",
    """pub fn clamp(val: i32, lo: i32, hi: i32) -> i32 {
    if val < lo { lo }
    else if val > hi { hi }
    else { val }
}""",
    "equivalent",
    description="Clamp function, equivalent on all well-defined inputs")

add_pair("popcount_naive", "bitwise",
    """unsigned int popcount(unsigned int x) {
    unsigned int count = 0;
    while (x) { count += x & 1; x >>= 1; }
    return count;
}""",
    """pub fn popcount(mut x: u32) -> u32 {
    let mut count: u32 = 0;
    while x != 0 {
        count = count.wrapping_add(x & 1);
        x >>= 1;
    }
    count
}""",
    "equivalent",
    description="Population count via bit scan, equivalent for unsigned")

add_pair("rotate_left", "bitwise",
    """unsigned int rotl(unsigned int x, int n) {
    return (x << n) | (x >> (32 - n));
}""",
    """pub fn rotl(x: u32, n: i32) -> u32 {
    x.rotate_left(n as u32)
}""",
    "divergent",
    divergence_type="OVF",
    description="C: UB if n >= 32 or n <= 0 due to shift; Rust: well-defined rotate")

add_pair("sign_extend_cast", "cast",
    """long long sign_extend(int x) {
    return (long long)x;
}""",
    """pub fn sign_extend(x: i32) -> i64 {
    x as i64
}""",
    "equivalent",
    description="Sign extension cast, equivalent in both languages")

add_pair("checked_mul_overflow", "overflow",
    """int checked_mul(int a, int b, int *overflow) {
    long long result = (long long)a * (long long)b;
    *overflow = (result > 2147483647 || result < -2147483648);
    return (int)result;
}""",
    """pub fn checked_mul(a: i32, b: i32) -> (i32, bool) {
    let (result, overflow) = a.overflowing_mul(b);
    (result, overflow)
}""",
    "equivalent",
    description="Checked multiplication with overflow detection")

add_pair("min3", "control_flow",
    """int min3(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}""",
    """pub fn min3(a: i32, b: i32, c: i32) -> i32 {
    let mut m = a;
    if b < m { m = b; }
    if c < m { m = c; }
    m
}""",
    "equivalent",
    description="Three-way minimum, equivalent semantics")

add_pair("power_of_two_check", "bitwise",
    """int is_power_of_two(unsigned int x) {
    return x != 0 && (x & (x - 1)) == 0;
}""",
    """pub fn is_power_of_two(x: u32) -> bool {
    x != 0 && (x & (x.wrapping_sub(1))) == 0
}""",
    "equivalent",
    description="Power-of-two check, equivalent for unsigned")

add_pair("byte_swap_32", "bitwise",
    """unsigned int bswap32(unsigned int x) {
    return ((x >> 24) & 0xFF) |
           ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) |
           ((x << 24) & 0xFF000000);
}""",
    """pub fn bswap32(x: u32) -> u32 {
    x.swap_bytes()
}""",
    "equivalent",
    description="32-bit byte swap, equivalent")

add_pair("saturating_add", "overflow",
    """int sat_add(int a, int b) {
    long long sum = (long long)a + (long long)b;
    if (sum > 2147483647) return 2147483647;
    if (sum < -2147483648LL) return -2147483648;
    return (int)sum;
}""",
    """pub fn sat_add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}""",
    "equivalent",
    description="Saturating add, equivalent when C uses wide arithmetic")

@dataclass
class VerificationResult:
    name: str
    category: str
    verdict: str  # "equivalent", "divergent", "unknown", "error"
    time_ms: float
    description: str
    expected_verdict: str
    divergence_type: Optional[str] = None
    counterexample: Optional[dict] = None
    error_msg: Optional[str] = None
    c_parsed: bool = False
    rust_parsed: bool = False
    c_ir_lowered: bool = False
    rust_ir_lowered: bool = False
    smt_queries: int = 0
    ir_invisible: bool = False
    alignment_score: float = 0.0
    product_built: bool = False
    coercion_points_count: int = 0
    pipeline_log: list = field(default_factory=list)


def run_smt_verification(pair: dict) -> VerificationResult:
    """Run real SMT-based verification on a C/Rust function pair.
    
    Uses the unified pipeline_verify.run_pipeline for ALL verification —
    no hand-coded per-category Z3 constraints.
    """
    start = time.time()
    name = pair["name"]
    result = VerificationResult(
        name=name, category=pair["category"],
        verdict="unknown", time_ms=0, description=pair["description"],
        expected_verdict=pair["expected_verdict"],
        divergence_type=pair.get("divergence_type"),
    )

    try:
        from pipeline_verify import run_pipeline
        pr = run_pipeline(name, pair["c_source"], pair["rust_source"])

        result.verdict = pr.verdict
        result.counterexample = pr.counterexample
        result.smt_queries = pr.smt_queries
        result.alignment_score = pr.alignment_score
        result.pipeline_log = pr.pipeline_log
        result.c_parsed = pr.pipeline_stages.get("c_parse", False)
        result.rust_parsed = pr.pipeline_stages.get("rust_parse", False)
        result.c_ir_lowered = pr.pipeline_stages.get("c_ir", False)
        result.rust_ir_lowered = pr.pipeline_stages.get("rust_ir", False)
        result.product_built = pr.pipeline_stages.get("product", False)
        result.coercion_points_count = pr.coercion_points

        if pr.error_msg:
            result.error_msg = pr.error_msg

        # Determine if divergence would be IR-invisible
        if "divergent" in result.verdict and pair.get("divergence_type") in ("OVF", "ERR"):
            result.ir_invisible = True

    except Exception as e:
        result.verdict = "error"
        result.error_msg = str(e)
        result.pipeline_log = [f"ERROR: {e}"]

    result.time_ms = (time.time() - start) * 1000
    return result


# ── Main ─────────────────────────────────────────────────────────────────

def run_all_experiments():
    results = []
    summary = {
        "total": 0,
        "equivalent": 0,
        "divergent": 0,
        "unknown": 0,
        "error": 0,
        "c_parse_success": 0,
        "rust_parse_success": 0,
        "c_ir_success": 0,
        "rust_ir_success": 0,
        "product_built_count": 0,
        "ir_invisible_divergences": 0,
        "total_smt_queries": 0,
        "total_time_ms": 0,
        "by_category": {},
        "by_divergence_type": {},
    }

    print(f"Running {len(BENCHMARK_PAIRS)} benchmark pairs...")
    print("=" * 70)

    for i, pair in enumerate(BENCHMARK_PAIRS):
        result = run_smt_verification(pair)
        results.append(asdict(result))

        # Update summary
        summary["total"] += 1
        if "equivalent" in result.verdict:
            summary["equivalent"] += 1
        elif "divergent" in result.verdict:
            summary["divergent"] += 1
        elif result.verdict == "error":
            summary["error"] += 1
        else:
            summary["unknown"] += 1

        if result.c_parsed:
            summary["c_parse_success"] += 1
        if result.rust_parsed:
            summary["rust_parse_success"] += 1
        if result.c_ir_lowered:
            summary["c_ir_success"] += 1
        if result.rust_ir_lowered:
            summary["rust_ir_success"] += 1
        if result.ir_invisible:
            summary["ir_invisible_divergences"] += 1
        if result.product_built:
            summary["product_built_count"] += 1
        summary["total_smt_queries"] += result.smt_queries
        summary["total_time_ms"] += result.time_ms

        cat = result.category
        if cat not in summary["by_category"]:
            summary["by_category"][cat] = {"total": 0, "equiv": 0, "div": 0, "unk": 0, "err": 0}
        summary["by_category"][cat]["total"] += 1
        if "equivalent" in result.verdict:
            summary["by_category"][cat]["equiv"] += 1
        elif "divergent" in result.verdict:
            summary["by_category"][cat]["div"] += 1
        elif result.verdict == "error":
            summary["by_category"][cat]["err"] += 1
        else:
            summary["by_category"][cat]["unk"] += 1

        if result.divergence_type:
            dt = result.divergence_type
            summary["by_divergence_type"][dt] = summary["by_divergence_type"].get(dt, 0) + 1

        status = "✓" if result.verdict != "error" else "✗"
        print(f"  [{i+1:2d}/{len(BENCHMARK_PAIRS)}] {status} {result.name:30s} → {result.verdict:25s} ({result.time_ms:.1f}ms)")

    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total pairs:       {summary['total']}")
    print(f"  Equivalent:        {summary['equivalent']}")
    print(f"  Divergent:         {summary['divergent']}")
    print(f"  Unknown:           {summary['unknown']}")
    print(f"  Errors:            {summary['error']}")
    print(f"  IR-invisible:      {summary['ir_invisible_divergences']}")
    print(f"  C parse success:   {summary['c_parse_success']}")
    print(f"  Rust parse success:{summary['rust_parse_success']}")
    print(f"  C IR success:      {summary['c_ir_success']}")
    print(f"  Rust IR success:   {summary['rust_ir_success']}")
    print(f"  Product programs:  {summary['product_built_count']}")
    print(f"  Total SMT queries: {summary['total_smt_queries']}")
    print(f"  Total time:        {summary['total_time_ms']:.1f}ms")
    print(f"  Median time:       {sorted([r['time_ms'] for r in results])[len(results)//2]:.1f}ms")

    print(f"\n  By divergence type:")
    for dt, count in sorted(summary["by_divergence_type"].items()):
        print(f"    {dt}: {count}")

    print(f"\n  By category:")
    for cat, stats in sorted(summary["by_category"].items()):
        print(f"    {cat}: {stats['total']} total, {stats['equiv']} equiv, {stats['div']} div, {stats['unk']} unk, {stats['err']} err")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "verification_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to experiments/results/")
    return results, summary


if __name__ == "__main__":
    run_all_experiments()
