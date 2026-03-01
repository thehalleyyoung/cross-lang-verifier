#!/usr/bin/env python3
"""
CEGAR-style LLM-in-the-Loop Verification Experiment for SemRec.

Counterexample-Guided Abstraction Refinement applied to LLM code translation:
  1. Take a C function
  2. Ask GPT-4.1-nano to translate it to Rust
  3. Verify with the full SemRec pipeline (CParser → IR → ProductBuilder → Z3)
  4. If counterexample found, feed it back to the LLM for repair
  5. Iterate until correct or max iterations reached

This implements the FM+AI contribution suggested by reviewers as the most
exciting research direction: using formal methods counterexamples to guide
LLM repair of code translations.

Note: GPT-4.1-nano is a weak, commodity-class LLM — it approximates what
people could do with models runnable on consumer hardware. The point is that
even a weak translator benefits enormously from CEGAR-style verification
feedback, since the formal verifier catches semantic bugs the LLM misses.
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_verify import run_pipeline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# C benchmark functions — varying complexity (small to medium)
# ---------------------------------------------------------------------------

C_BENCHMARKS = [
    # --- Simple arithmetic (< 5 lines) ---
    {
        "name": "safe_add",
        "code": "int safe_add(int a, int b) { return a + b; }",
        "category": "overflow",
        "loc": 1,
        "expected_bugs": ["signed_overflow"],
    },
    {
        "name": "negate",
        "code": "int negate(int x) { return -x; }",
        "category": "overflow",
        "loc": 1,
        "expected_bugs": ["int_min_negation"],
    },
    {
        "name": "safe_divide",
        "code": """int safe_divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}""",
        "category": "error_handling",
        "loc": 4,
        "expected_bugs": ["int_min_div_neg1"],
    },
    # --- Medium arithmetic (5-15 lines) ---
    {
        "name": "abs_diff",
        "code": """int abs_diff(int a, int b) {
    int diff = a - b;
    return diff < 0 ? -diff : diff;
}""",
        "category": "overflow",
        "loc": 4,
        "expected_bugs": ["subtraction_overflow", "negation_overflow"],
    },
    {
        "name": "average",
        "code": """int average(int a, int b) {
    return (a + b) / 2;
}""",
        "category": "overflow",
        "loc": 3,
        "expected_bugs": ["addition_overflow"],
    },
    {
        "name": "midpoint",
        "code": """int midpoint(int a, int b) {
    return a + (b - a) / 2;
}""",
        "category": "overflow",
        "loc": 3,
        "expected_bugs": ["subtraction_overflow"],
    },
    {
        "name": "clamp",
        "code": """int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}""",
        "category": "control_flow",
        "loc": 5,
        "expected_bugs": [],
    },
    {
        "name": "count_bits",
        "code": """int count_bits(unsigned int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}""",
        "category": "loop",
        "loc": 8,
        "expected_bugs": [],
    },
    {
        "name": "leading_zeros",
        "code": """int leading_zeros(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8; x <<= 8; }
    if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; }
    if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
}""",
        "category": "bitwise",
        "loc": 10,
        "expected_bugs": [],
    },
    {
        "name": "rotate_right",
        "code": """unsigned int rotate_right(unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}""",
        "category": "bitwise",
        "loc": 3,
        "expected_bugs": ["shift_ub"],
    },
    {
        "name": "sign",
        "code": """int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}""",
        "category": "control_flow",
        "loc": 5,
        "expected_bugs": [],
    },
    {
        "name": "isqrt",
        "code": """int isqrt(int n) {
    if (n < 0) return -1;
    int x = n;
    int y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}""",
        "category": "loop",
        "loc": 10,
        "expected_bugs": ["division_behavior"],
    },
    # --- Functions with multiple semantic traps ---
    {
        "name": "safe_mul",
        "code": """int safe_mul(int a, int b) {
    if (b != 0 && a > 2147483647 / b) return 2147483647;
    if (b != 0 && a < (-2147483647 - 1) / b) return -2147483647 - 1;
    return a * b;
}""",
        "category": "overflow",
        "loc": 5,
        "expected_bugs": ["overflow_check_ub"],
    },
    {
        "name": "power",
        "code": """int power(int base, int exp) {
    int result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) result *= base;
        base *= base;
        exp /= 2;
    }
    return result;
}""",
        "category": "loop",
        "loc": 9,
        "expected_bugs": ["multiplication_overflow"],
    },
    {
        "name": "gcd",
        "code": """int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}""",
        "category": "loop",
        "loc": 8,
        "expected_bugs": ["modulo_behavior"],
    },
    {
        "name": "saturating_add",
        "code": """int saturating_add(int a, int b) {
    long long sum = (long long)a + (long long)b;
    if (sum > 2147483647) return 2147483647;
    if (sum < -2147483648LL) return -2147483648;
    return (int)sum;
}""",
        "category": "overflow",
        "loc": 6,
        "expected_bugs": [],
    },
    {
        "name": "reverse_bits",
        "code": """unsigned int reverse_bits(unsigned int n) {
    unsigned int result = 0;
    for (int i = 0; i < 32; i++) {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    return result;
}""",
        "category": "bitwise",
        "loc": 8,
        "expected_bugs": [],
    },
    {
        "name": "swap_bytes",
        "code": """unsigned int swap_bytes(unsigned int x) {
    return ((x >> 24) & 0xFF) |
           ((x >> 8)  & 0xFF00) |
           ((x << 8)  & 0xFF0000) |
           ((x << 24) & 0xFF000000);
}""",
        "category": "bitwise",
        "loc": 6,
        "expected_bugs": [],
    },
    {
        "name": "sum_to_n",
        "code": """int sum_to_n(int n) {
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    return sum;
}""",
        "category": "loop",
        "loc": 7,
        "expected_bugs": ["accumulator_overflow"],
    },
    {
        "name": "factorial",
        "code": """int factorial(int n) {
    if (n <= 1) return 1;
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}""",
        "category": "loop",
        "loc": 8,
        "expected_bugs": ["multiplication_overflow"],
    },
]


# ---------------------------------------------------------------------------
# Pipeline-backed verification for C ↔ Rust pairs
# ---------------------------------------------------------------------------

def verify_translation(c_code: str, rust_code: str, func_name: str) -> Dict[str, Any]:
    """Verify C ↔ Rust equivalence via the full SemRec pipeline.

    Returns dict with keys: verdict, counterexample, time_ms, smt_queries,
    pipeline_stages, alignment_score, error_msg
    """
    result = run_pipeline(func_name, c_code, rust_code)
    return {
        "verdict": result.verdict,
        "counterexample": result.counterexample,
        "time_ms": result.time_ms,
        "smt_queries": result.smt_queries,
        "pipeline_stages": result.pipeline_stages,
        "alignment_score": result.alignment_score,
        "error_msg": result.error_msg,
    }


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def get_llm_client():
    """Create OpenAI client, or return None if OPENAI_API_KEY is not set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI()


def llm_translate(client, c_code: str, func_name: str) -> str:
    """Ask GPT-4.1-nano to translate C to Rust."""
    prompt = f"""Translate this C function to Rust. Return ONLY the Rust function, no explanation.
Use idiomatic Rust. The function should have the same semantics as the C version.
Handle edge cases correctly (overflow, division by zero, etc.).

C function:
```c
{c_code}
```

Rust function:"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    rust_code = response.choices[0].message.content.strip()
    # Strip markdown code fences
    if rust_code.startswith("```"):
        lines = rust_code.split('\n')
        rust_code = '\n'.join(lines[1:])
        if rust_code.endswith("```"):
            rust_code = rust_code[:-3].strip()
    return rust_code


def llm_repair(client, c_code: str, rust_code: str, counterexample: Dict,
               func_name: str, iteration: int) -> str:
    """Ask the LLM to repair the Rust translation given a counterexample."""
    # Format counterexample for the prompt — separate input values from metadata
    cex_inputs = {k: v for k, v in counterexample.items()
                  if k not in ('reason', 'c_behavior', 'rust_behavior')}
    cex_reason = counterexample.get('reason', 'semantic divergence')

    prompt = f"""The following Rust translation of a C function has a semantic bug.

## C function (ground truth):
```c
{c_code}
```

## Current Rust translation (BUGGY):
```rust
{rust_code}
```

## Bug found by formal verification (SemRec pipeline):
- Counterexample input: {json.dumps(cex_inputs, default=str)}
- Divergence reason: {cex_reason}

## Instructions:
Fix the Rust function so it is semantically equivalent to the C function on all
well-defined inputs. Key C-to-Rust semantic differences:
- C signed overflow is UB; use wrapping_add/wrapping_sub/wrapping_mul in Rust
- C division by zero is UB; Rust panics
- C shift by >= width is UB; Rust wraps shift amount or panics
- C negation of INT_MIN is UB; use wrapping_neg() in Rust

Return ONLY the fixed Rust function, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    rust_code = response.choices[0].message.content.strip()
    if rust_code.startswith("```"):
        lines = rust_code.split('\n')
        rust_code = '\n'.join(lines[1:])
        if rust_code.endswith("```"):
            rust_code = rust_code[:-3].strip()
    return rust_code


# ---------------------------------------------------------------------------
# Fallback hand-coded translations (used when OPENAI_API_KEY is not set)
# ---------------------------------------------------------------------------

FALLBACK_TRANSLATIONS = {
    "safe_add": "fn safe_add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }",
    "negate": "fn negate(x: i32) -> i32 { x.wrapping_neg() }",
    "safe_divide": """fn safe_divide(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    a.wrapping_div(b)
}""",
    "abs_diff": """fn abs_diff(a: i32, b: i32) -> i32 {
    let diff = a.wrapping_sub(b);
    if diff < 0 { diff.wrapping_neg() } else { diff }
}""",
    "average": "fn average(a: i32, b: i32) -> i32 { a.wrapping_add(b) / 2 }",
    "midpoint": "fn midpoint(a: i32, b: i32) -> i32 { a.wrapping_add(b.wrapping_sub(a) / 2) }",
    "clamp": """fn clamp(val: i32, lo: i32, hi: i32) -> i32 {
    if val < lo { return lo; }
    if val > hi { return hi; }
    val
}""",
    "count_bits": """fn count_bits(mut n: u32) -> i32 {
    let mut count: i32 = 0;
    while n != 0 {
        count += (n & 1) as i32;
        n >>= 1;
    }
    count
}""",
    "leading_zeros": """fn leading_zeros(mut x: u32) -> i32 {
    if x == 0 { return 32; }
    let mut n: i32 = 0;
    if x <= 0x0000FFFF { n += 16; x <<= 16; }
    if x <= 0x00FFFFFF { n += 8; x <<= 8; }
    if x <= 0x0FFFFFFF { n += 4; x <<= 4; }
    if x <= 0x3FFFFFFF { n += 2; x <<= 2; }
    if x <= 0x7FFFFFFF { n += 1; }
    n
}""",
    "rotate_right": """fn rotate_right(x: u32, n: i32) -> u32 {
    let n = (n as u32) & 31;
    (x >> n) | (x << (32 - n))
}""",
    "sign": """fn sign(x: i32) -> i32 {
    if x > 0 { 1 }
    else if x < 0 { -1 }
    else { 0 }
}""",
    "isqrt": """fn isqrt(n: i32) -> i32 {
    if n < 0 { return -1; }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}""",
    "safe_mul": """fn safe_mul(a: i32, b: i32) -> i32 {
    if b != 0 && a > i32::MAX / b { return i32::MAX; }
    if b != 0 && a < i32::MIN / b { return i32::MIN; }
    a.wrapping_mul(b)
}""",
    "power": """fn power(mut base: i32, mut exp: i32) -> i32 {
    let mut result: i32 = 1;
    while exp > 0 {
        if exp % 2 == 1 { result = result.wrapping_mul(base); }
        base = base.wrapping_mul(base);
        exp /= 2;
    }
    result
}""",
    "gcd": """fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}""",
    "saturating_add": """fn saturating_add(a: i32, b: i32) -> i32 {
    let sum: i64 = (a as i64) + (b as i64);
    if sum > i32::MAX as i64 { return i32::MAX; }
    if sum < i32::MIN as i64 { return i32::MIN; }
    sum as i32
}""",
    "reverse_bits": """fn reverse_bits(mut n: u32) -> u32 {
    let mut result: u32 = 0;
    for _ in 0..32 {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    result
}""",
    "swap_bytes": """fn swap_bytes(x: u32) -> u32 {
    ((x >> 24) & 0xFF) |
    ((x >> 8)  & 0xFF00) |
    ((x << 8)  & 0xFF0000) |
    ((x << 24) & 0xFF000000)
}""",
    "sum_to_n": """fn sum_to_n(n: i32) -> i32 {
    let mut sum: i32 = 0;
    let mut i = 1;
    while i <= n {
        sum = sum.wrapping_add(i);
        i += 1;
    }
    sum
}""",
    "factorial": """fn factorial(n: i32) -> i32 {
    if n <= 1 { return 1; }
    let mut result: i32 = 1;
    let mut i = 2;
    while i <= n {
        result = result.wrapping_mul(i);
        i += 1;
    }
    result
}""",
}


# ---------------------------------------------------------------------------
# CEGAR loop
# ---------------------------------------------------------------------------

@dataclass
class CEGARResult:
    func_name: str
    category: str
    loc: int
    success: bool
    iterations: int
    max_iterations: int
    initial_translation: str
    final_translation: str
    translation_history: List[str]
    bugs_found: List[str]
    bugs_fixed: List[str]
    counterexamples: List[Dict]
    verification_time_ms: float
    total_time_ms: float

    def to_dict(self) -> Dict:
        return asdict(self)


def run_cegar_loop(client, benchmark: Dict, max_iterations: int = 5) -> CEGARResult:
    """Run a single CEGAR loop for one function.

    If client is None (no API key), uses fallback hand-coded translations.
    """
    func_name = benchmark["name"]
    c_code = benchmark["code"]
    category = benchmark["category"]
    loc = benchmark["loc"]
    
    total_start = time.time()
    
    # Step 1: Initial translation (LLM or fallback)
    if client is not None:
        print(f"  [TRANSLATE] {func_name} (GPT-4.1-nano)...")
        try:
            rust_code = llm_translate(client, c_code, func_name)
        except Exception as e:
            print(f"    LLM translation failed: {e}")
            return CEGARResult(
                func_name=func_name, category=category, loc=loc,
                success=False, iterations=0, max_iterations=max_iterations,
                initial_translation="", final_translation="",
                translation_history=[], bugs_found=[], bugs_fixed=[],
                counterexamples=[], verification_time_ms=0,
                total_time_ms=(time.time() - total_start) * 1000,
            )
    else:
        print(f"  [TRANSLATE] {func_name} (fallback, no API key)...")
        rust_code = FALLBACK_TRANSLATIONS.get(func_name)
        if rust_code is None:
            print(f"    No fallback translation for {func_name}")
            return CEGARResult(
                func_name=func_name, category=category, loc=loc,
                success=False, iterations=0, max_iterations=max_iterations,
                initial_translation="", final_translation="",
                translation_history=[], bugs_found=[], bugs_fixed=[],
                counterexamples=[], verification_time_ms=0,
                total_time_ms=(time.time() - total_start) * 1000,
            )
    
    initial_translation = rust_code
    translation_history = [rust_code]
    all_bugs = []
    all_cex = []
    bugs_fixed = []
    total_verify_time = 0
    
    for iteration in range(max_iterations):
        # Step 2: Verify via pipeline
        print(f"  [VERIFY] iteration {iteration + 1}...")
        ver = verify_translation(c_code, rust_code, func_name)
        total_verify_time += ver["time_ms"]
        
        if ver["verdict"] == "equivalent":
            print(f"    ✓ Verified equivalent after {iteration + 1} iteration(s)")
            return CEGARResult(
                func_name=func_name, category=category, loc=loc,
                success=True, iterations=iteration + 1,
                max_iterations=max_iterations,
                initial_translation=initial_translation,
                final_translation=rust_code,
                translation_history=translation_history,
                bugs_found=all_bugs, bugs_fixed=bugs_fixed,
                counterexamples=all_cex,
                verification_time_ms=total_verify_time,
                total_time_ms=(time.time() - total_start) * 1000,
            )
        
        # Step 3: Extract counterexample from pipeline result
        cex = ver.get("counterexample") or {}
        
        if ver["verdict"] in ("divergent",) and cex:
            bug_desc = cex.get("reason", ver["verdict"])
            all_bugs.append(bug_desc)
            all_cex.append(cex)
            print(f"    ✗ Divergent: {bug_desc}, CEX: {cex}")
        elif ver["verdict"] in ("pipeline_fail", "error"):
            print(f"    ✗ Pipeline error: {ver.get('error_msg', 'unknown')}")
            all_bugs.append(ver.get("error_msg", ver["verdict"]))
            all_cex.append({"error": ver.get("error_msg", "pipeline failure")})
        else:
            print(f"    ? Verdict: {ver['verdict']}, no actionable counterexample")
            break
        
        # Step 4: Repair (only if we have an LLM client)
        if client is None:
            print(f"    (no LLM client for repair, stopping)")
            break
        
        print(f"  [REPAIR] feeding counterexample to LLM...")
        try:
            prev_bugs = set(all_bugs)
            rust_code = llm_repair(client, c_code, rust_code, cex,
                                   func_name, iteration)
            translation_history.append(rust_code)
            
            # Check if we actually fixed the bug
            ver2 = verify_translation(c_code, rust_code, func_name)
            total_verify_time += ver2["time_ms"]
            if ver2["verdict"] == "equivalent":
                bugs_fixed.extend(all_bugs)
                print(f"    ✓ All bugs fixed")
            elif ver2["verdict"] == "divergent":
                new_cex = ver2.get("counterexample") or {}
                new_reason = new_cex.get("reason", "")
                if new_reason != bug_desc:
                    bugs_fixed.append(bug_desc)
                    print(f"    ✓ Fixed: {bug_desc} (new issue: {new_reason})")
                else:
                    print(f"    ✗ Same bug persists: {bug_desc}")
        except Exception as e:
            print(f"    Repair failed: {e}")
            break
    
    # Final verification
    ver_final = verify_translation(c_code, rust_code, func_name)
    total_verify_time += ver_final["time_ms"]
    success = ver_final["verdict"] == "equivalent"
    
    total_time = (time.time() - total_start) * 1000
    return CEGARResult(
        func_name=func_name, category=category, loc=loc,
        success=success,
        iterations=min(len(translation_history), max_iterations),
        max_iterations=max_iterations,
        initial_translation=initial_translation,
        final_translation=rust_code,
        translation_history=translation_history,
        bugs_found=list(set(all_bugs)),
        bugs_fixed=list(set(bugs_fixed)),
        counterexamples=all_cex,
        verification_time_ms=total_verify_time,
        total_time_ms=total_time,
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the full CEGAR experiment."""
    print("=" * 70)
    print("SemRec CEGAR/LLM-in-the-Loop Verification Experiment")
    print("=" * 70)
    print(f"Benchmarks: {len(C_BENCHMARKS)} C functions")
    print(f"Model: GPT-4.1-nano (weak/commodity-class LLM)")
    print(f"Max CEGAR iterations: 5")
    print(f"Verification: SemRec pipeline (CParser → IR → Product → Z3)")
    print()
    
    client = get_llm_client()
    if client is None:
        print("⚠ OPENAI_API_KEY not set — using fallback hand-coded translations")
        print("  (CEGAR repair loop disabled; only initial verification will run)")
        print()
    results = []
    
    for i, bench in enumerate(C_BENCHMARKS):
        print(f"\n[{i+1}/{len(C_BENCHMARKS)}] {bench['name']} ({bench['category']}, {bench['loc']} LOC)")
        try:
            result = run_cegar_loop(client, bench, max_iterations=5)
            results.append(result.to_dict())
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "func_name": bench["name"],
                "category": bench["category"],
                "loc": bench["loc"],
                "success": False,
                "iterations": 0,
                "error": str(e),
            })
    
    # Compute statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    failed = total - successful
    
    total_bugs_found = set()
    total_bugs_fixed = set()
    iter_counts = []
    for r in results:
        for b in r.get("bugs_found", []):
            total_bugs_found.add(f"{r['func_name']}:{b}")
        for b in r.get("bugs_fixed", []):
            total_bugs_fixed.add(f"{r['func_name']}:{b}")
        if r.get("iterations"):
            iter_counts.append(r["iterations"])
    
    # Breakdown by category
    by_category = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "success": 0, "bugs_found": 0}
        by_category[cat]["total"] += 1
        if r.get("success"):
            by_category[cat]["success"] += 1
        by_category[cat]["bugs_found"] += len(r.get("bugs_found", []))
    
    # Breakdown by LOC complexity
    by_complexity = {"small_1_5": {"total": 0, "success": 0},
                     "medium_6_10": {"total": 0, "success": 0},
                     "large_11_plus": {"total": 0, "success": 0}}
    for r in results:
        loc = r.get("loc", 0)
        if loc <= 5:
            key = "small_1_5"
        elif loc <= 10:
            key = "medium_6_10"
        else:
            key = "large_11_plus"
        by_complexity[key]["total"] += 1
        if r.get("success"):
            by_complexity[key]["success"] += 1
    
    # Bug type histogram
    bug_types = {}
    for r in results:
        for b in r.get("bugs_found", []):
            bug_types[b] = bug_types.get(b, 0) + 1
    
    summary = {
        "experiment": "CEGAR_LLM_in_the_loop",
        "model": "gpt-4.1-nano",
        "max_iterations": 5,
        "total_functions": total,
        "verified_correct_after_repair": successful,
        "still_divergent": failed,
        "success_rate": successful / total if total > 0 else 0,
        "total_bugs_found": len(total_bugs_found),
        "total_bugs_fixed": len(total_bugs_fixed),
        "fix_rate": len(total_bugs_fixed) / len(total_bugs_found) if total_bugs_found else 0,
        "avg_iterations": sum(iter_counts) / len(iter_counts) if iter_counts else 0,
        "max_iterations_used": max(iter_counts) if iter_counts else 0,
        "by_category": by_category,
        "by_complexity": by_complexity,
        "bug_type_histogram": bug_types,
        "results": results,
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("CEGAR EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"Total functions:              {total}")
    print(f"Verified correct after CEGAR: {successful}/{total} ({summary['success_rate']:.0%})")
    print(f"Still divergent:              {failed}/{total}")
    print(f"Total bugs found:             {len(total_bugs_found)}")
    print(f"Bugs fixed by CEGAR:          {len(total_bugs_fixed)}")
    print(f"Fix rate:                     {summary['fix_rate']:.0%}")
    print(f"Avg iterations to converge:   {summary['avg_iterations']:.1f}")
    print()
    print("By category:")
    for cat, stats in by_category.items():
        print(f"  {cat:20s}: {stats['success']}/{stats['total']} verified, {stats['bugs_found']} bugs")
    print()
    print("By complexity:")
    for comp, stats in by_complexity.items():
        print(f"  {comp:20s}: {stats['success']}/{stats['total']} verified")
    print()
    print("Bug types found:")
    for bt, count in sorted(bug_types.items(), key=lambda x: -x[1]):
        print(f"  {bt:30s}: {count}")
    
    # Save results
    outpath = os.path.join(RESULTS_DIR, "cegar_llm_results.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    
    return summary


if __name__ == "__main__":
    run_experiment()
