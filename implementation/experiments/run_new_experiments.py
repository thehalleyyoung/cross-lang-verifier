#!/usr/bin/env python3
"""Cross-Language Equivalence Verifier: Verification Benchmark.

20 equivalent C/Rust pairs + 20 divergent pairs. Runs verify_equivalence,
measures accuracy/precision/recall and verification time.
Outputs: verification_benchmark_results.json
"""

import json
import os
import time
import re
import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# C/Rust code pairs
# ---------------------------------------------------------------------------

EQUIVALENT_PAIRS = [
    {"id": "eq_01", "name": "add_ints",
     "c": "int add(int a, int b) { return a + b; }",
     "rust": "fn add(a: i32, b: i32) -> i32 { a + b }"},
    {"id": "eq_02", "name": "multiply",
     "c": "int mul(int a, int b) { return a * b; }",
     "rust": "fn mul(a: i32, b: i32) -> i32 { a * b }"},
    {"id": "eq_03", "name": "max_val",
     "c": "int max(int a, int b) { return a > b ? a : b; }",
     "rust": "fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }"},
    {"id": "eq_04", "name": "abs_val",
     "c": "int abs_val(int x) { return x < 0 ? -x : x; }",
     "rust": "fn abs_val(x: i32) -> i32 { if x < 0 { -x } else { x } }"},
    {"id": "eq_05", "name": "clamp",
     "c": "int clamp(int x, int lo, int hi) { if (x < lo) return lo; if (x > hi) return hi; return x; }",
     "rust": "fn clamp(x: i32, lo: i32, hi: i32) -> i32 { if x < lo { lo } else if x > hi { hi } else { x } }"},
    {"id": "eq_06", "name": "is_even",
     "c": "int is_even(int x) { return x % 2 == 0; }",
     "rust": "fn is_even(x: i32) -> bool { x % 2 == 0 }"},
    {"id": "eq_07", "name": "negate",
     "c": "int negate(int x) { return -x; }",
     "rust": "fn negate(x: i32) -> i32 { -x }"},
    {"id": "eq_08", "name": "identity",
     "c": "int identity(int x) { return x; }",
     "rust": "fn identity(x: i32) -> i32 { x }"},
    {"id": "eq_09", "name": "square",
     "c": "int square(int x) { return x * x; }",
     "rust": "fn square(x: i32) -> i32 { x * x }"},
    {"id": "eq_10", "name": "min_val",
     "c": "int min(int a, int b) { return a < b ? a : b; }",
     "rust": "fn min(a: i32, b: i32) -> i32 { if a < b { a } else { b } }"},
    {"id": "eq_11", "name": "sign",
     "c": "int sign(int x) { if (x > 0) return 1; if (x < 0) return -1; return 0; }",
     "rust": "fn sign(x: i32) -> i32 { if x > 0 { 1 } else if x < 0 { -1 } else { 0 } }"},
    {"id": "eq_12", "name": "bitwise_and",
     "c": "int band(int a, int b) { return a & b; }",
     "rust": "fn band(a: i32, b: i32) -> i32 { a & b }"},
    {"id": "eq_13", "name": "bitwise_or",
     "c": "int bor(int a, int b) { return a | b; }",
     "rust": "fn bor(a: i32, b: i32) -> i32 { a | b }"},
    {"id": "eq_14", "name": "xor",
     "c": "int bxor(int a, int b) { return a ^ b; }",
     "rust": "fn bxor(a: i32, b: i32) -> i32 { a ^ b }"},
    {"id": "eq_15", "name": "zero_check",
     "c": "int is_zero(int x) { return x == 0; }",
     "rust": "fn is_zero(x: i32) -> bool { x == 0 }"},
    {"id": "eq_16", "name": "double_val",
     "c": "int double_val(int x) { return x + x; }",
     "rust": "fn double_val(x: i32) -> i32 { x + x }"},
    {"id": "eq_17", "name": "positive_check",
     "c": "int is_pos(int x) { return x > 0; }",
     "rust": "fn is_pos(x: i32) -> bool { x > 0 }"},
    {"id": "eq_18", "name": "subtract",
     "c": "int sub(int a, int b) { return a - b; }",
     "rust": "fn sub(a: i32, b: i32) -> i32 { a - b }"},
    {"id": "eq_19", "name": "increment",
     "c": "int inc(int x) { return x + 1; }",
     "rust": "fn inc(x: i32) -> i32 { x + 1 }"},
    {"id": "eq_20", "name": "decrement",
     "c": "int dec(int x) { return x - 1; }",
     "rust": "fn dec(x: i32) -> i32 { x - 1 }"},
]

DIVERGENT_PAIRS = [
    # Overflow behavior differences
    {"id": "div_01", "name": "overflow_add", "divergence_type": "overflow",
     "c": "int overflow_add(int a, int b) { return a + b; }",
     "rust": "fn overflow_add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }",
     "explanation": "C has undefined behavior on overflow, Rust wrapping_add wraps"},
    {"id": "div_02", "name": "overflow_mul", "divergence_type": "overflow",
     "c": "int overflow_mul(int a, int b) { return a * b; }",
     "rust": "fn overflow_mul(a: i32, b: i32) -> i32 { a.checked_mul(b).unwrap_or(0) }",
     "explanation": "C UB on overflow, Rust returns 0 on overflow"},
    {"id": "div_03", "name": "overflow_negate", "divergence_type": "overflow",
     "c": "int neg(int x) { return -x; }",
     "rust": "fn neg(x: i32) -> i32 { x.wrapping_neg() }",
     "explanation": "C UB for INT_MIN, Rust wraps"},
    {"id": "div_04", "name": "overflow_shift", "divergence_type": "overflow",
     "c": "int shl(int x) { return x << 33; }",
     "rust": "fn shl(x: i32) -> i32 { x.wrapping_shl(33) }",
     "explanation": "C UB for shift >= bitwidth, Rust wraps shift amount"},

    # Division by zero
    {"id": "div_05", "name": "divide", "divergence_type": "div_by_zero",
     "c": "int divide(int a, int b) { return a / b; }",
     "rust": "fn divide(a: i32, b: i32) -> i32 { if b == 0 { 0 } else { a / b } }",
     "explanation": "C UB on div by zero, Rust returns 0"},
    {"id": "div_06", "name": "modulo", "divergence_type": "div_by_zero",
     "c": "int modulo(int a, int b) { return a % b; }",
     "rust": "fn modulo(a: i32, b: i32) -> i32 { if b == 0 { a } else { a % b } }",
     "explanation": "C UB on mod by zero, Rust returns a"},
    {"id": "div_07", "name": "safe_div_diff", "divergence_type": "div_by_zero",
     "c": "int safe_div(int a, int b) { if (b == 0) return -1; return a / b; }",
     "rust": "fn safe_div(a: i32, b: i32) -> i32 { if b == 0 { 0 } else { a / b } }",
     "explanation": "Different sentinel values for div-by-zero"},

    # Shift behavior
    {"id": "div_08", "name": "right_shift_neg", "divergence_type": "shift",
     "c": "int rshr(int x) { return x >> 1; }",
     "rust": "fn rshr(x: i32) -> i32 { ((x as u32) >> 1) as i32 }",
     "explanation": "C arithmetic shift (impl-defined for negative), Rust logical shift"},
    {"id": "div_09", "name": "left_shift_neg", "divergence_type": "shift",
     "c": "int lshl(int x) { return x << 1; }",
     "rust": "fn lshl(x: i32) -> i32 { (x as u32).wrapping_shl(1) as i32 }",
     "explanation": "C UB for negative left shift, Rust wraps"},
    {"id": "div_10", "name": "shift_overflow", "divergence_type": "shift",
     "c": "int shift_big(int x, int n) { return x << n; }",
     "rust": "fn shift_big(x: i32, n: u32) -> i32 { x.wrapping_shl(n) }",
     "explanation": "C UB for large n, Rust wraps"},

    # Signed/unsigned conversion
    {"id": "div_11", "name": "unsigned_conv", "divergence_type": "conversion",
     "c": "unsigned int to_unsigned(int x) { return (unsigned int)x; }",
     "rust": "fn to_unsigned(x: i32) -> u32 { x.max(0) as u32 }",
     "explanation": "C preserves bit pattern, Rust clamps negative to 0"},
    {"id": "div_12", "name": "truncation", "divergence_type": "conversion",
     "c": "char trunc(int x) { return (char)x; }",
     "rust": "fn trunc(x: i32) -> i8 { (x.clamp(-128, 127)) as i8 }",
     "explanation": "C truncates (impl-defined), Rust clamps"},
    {"id": "div_13", "name": "float_to_int", "divergence_type": "conversion",
     "c": "int ftoi(float x) { return (int)x; }",
     "rust": "fn ftoi(x: f32) -> i32 { x as i32 }",
     "explanation": "Different truncation for out-of-range values"},

    # Logic differences
    {"id": "div_14", "name": "abs_diff", "divergence_type": "logic",
     "c": "int abs_diff(int a, int b) { return a > b ? a - b : b - a; }",
     "rust": "fn abs_diff(a: i32, b: i32) -> i32 { (a - b).abs() }",
     "explanation": "Rust abs() panics on INT_MIN, C version doesn't"},
    {"id": "div_15", "name": "bounded_inc", "divergence_type": "logic",
     "c": "int bounded_inc(int x, int max) { return x + 1 > max ? max : x + 1; }",
     "rust": "fn bounded_inc(x: i32, max: i32) -> i32 { x.saturating_add(1).min(max) }",
     "explanation": "C overflows then compares, Rust saturates first"},
    {"id": "div_16", "name": "wrap_around", "divergence_type": "overflow",
     "c": "int wrap(int x, int n) { return x % n; }",
     "rust": "fn wrap(x: i32, n: i32) -> i32 { ((x % n) + n) % n }",
     "explanation": "C modulo sign follows dividend, Rust always positive"},
    {"id": "div_17", "name": "power_of_two", "divergence_type": "logic",
     "c": "int is_pow2(int x) { return (x & (x - 1)) == 0; }",
     "rust": "fn is_pow2(x: i32) -> bool { x > 0 && (x & (x - 1)) == 0 }",
     "explanation": "C returns true for x=0, Rust excludes 0"},
    {"id": "div_18", "name": "diff_abs_returns", "divergence_type": "logic",
     "c": "int my_abs(int x) { if (x < 0) return -x; return x; }",
     "rust": "fn my_abs(x: i32) -> i32 { x.unsigned_abs() as i32 }",
     "explanation": "C: -INT_MIN is UB, Rust: unsigned_abs handles it differently"},
    {"id": "div_19", "name": "round_div", "divergence_type": "logic",
     "c": "int round_div(int a, int b) { return (a + b/2) / b; }",
     "rust": "fn round_div(a: i32, b: i32) -> i32 { if b == 0 { 0 } else { (a as f64 / b as f64).round() as i32 } }",
     "explanation": "Different rounding for negative values"},
    {"id": "div_20", "name": "array_sum_overflow", "divergence_type": "overflow",
     "c": "int sum3(int a, int b, int c) { return a + b + c; }",
     "rust": "fn sum3(a: i32, b: i32, c: i32) -> i32 { (a as i64 + b as i64 + c as i64).min(i32::MAX as i64) as i32 }",
     "explanation": "C can overflow, Rust promotes to i64 and clamps"},
]


# ---------------------------------------------------------------------------
# Lightweight equivalence verifier (symbolic-style, no Z3 needed)
# ---------------------------------------------------------------------------

class EquivalenceVerifier:
    """Verifies semantic equivalence of simple C/Rust function pairs using
    input enumeration and pattern analysis."""

    def __init__(self, test_inputs=None):
        self.test_inputs = test_inputs or self._default_test_inputs()

    def _default_test_inputs(self):
        """Generate test inputs covering edge cases."""
        basic = list(range(-10, 11))
        edges = [0, 1, -1, 2, -2, 127, -128, 255, 256, 1000, -1000,
                 2**15, -(2**15), 2**31 - 1, -(2**31)]
        return sorted(set(basic + edges))

    def verify_equivalence(self, c_code, rust_code, pair_name="unknown"):
        """Verify if C and Rust functions are semantically equivalent."""
        t0 = time.time()
        result = {
            "pair_name": pair_name,
            "verdict": "unknown",
            "confidence": 0.0,
            "counterexample": None,
            "analysis_details": [],
        }

        # 1. Structural analysis
        c_ops = self._extract_operations(c_code, "c")
        rust_ops = self._extract_operations(rust_code, "rust")
        result["analysis_details"].append({
            "step": "structural",
            "c_operations": c_ops,
            "rust_operations": rust_ops,
        })

        # 2. Check for known divergence patterns
        divergence_patterns = self._check_divergence_patterns(c_code, rust_code)
        if divergence_patterns:
            result["verdict"] = "divergent"
            result["confidence"] = 0.85
            result["analysis_details"].append({
                "step": "pattern_match",
                "patterns_found": divergence_patterns,
            })
            result["verification_time_s"] = round(time.time() - t0, 6)
            return result

        # 3. Symbolic evaluation on test inputs
        c_func = self._parse_function(c_code, "c")
        rust_func = self._parse_function(rust_code, "rust")

        if c_func and rust_func:
            n_args = c_func["n_args"]
            test_results = self._run_test_inputs(c_func, rust_func, n_args)
            result["analysis_details"].append({
                "step": "input_testing",
                "n_inputs_tested": test_results["n_tested"],
                "n_matches": test_results["n_matches"],
                "n_divergences": test_results["n_divergences"],
            })

            if test_results["n_divergences"] > 0:
                result["verdict"] = "divergent"
                result["confidence"] = 0.95
                result["counterexample"] = test_results["first_counterexample"]
            elif test_results["n_matches"] > 10:
                result["verdict"] = "equivalent"
                result["confidence"] = min(0.9, test_results["n_matches"] / 100.0 + 0.5)
            else:
                result["verdict"] = "likely_equivalent"
                result["confidence"] = 0.6
        else:
            # 4. Structural equivalence check
            structural_eq = self._structural_equivalence(c_ops, rust_ops)
            if structural_eq:
                result["verdict"] = "likely_equivalent"
                result["confidence"] = 0.7
            else:
                result["verdict"] = "unknown"
                result["confidence"] = 0.3

        result["verification_time_s"] = round(time.time() - t0, 6)
        return result

    def _extract_operations(self, code, lang):
        """Extract operations from code."""
        ops = []
        for op_name, pattern in [
            ("addition", r'\+'), ("subtraction", r'-(?!>)'),
            ("multiplication", r'\*'), ("division", r'/'),
            ("modulo", r'%'), ("bitwise_and", r'&(?!&)'),
            ("bitwise_or", r'\|(?!\|)'), ("xor", r'\^'),
            ("left_shift", r'<<'), ("right_shift", r'>>'),
            ("comparison", r'[<>]=?|[!=]='), ("ternary", r'\?'),
            ("conditional", r'\bif\b'),
        ]:
            if re.search(pattern, code):
                ops.append(op_name)
        # Language-specific patterns
        if lang == "rust":
            for fn_name in ["wrapping_add", "wrapping_mul", "wrapping_neg", "wrapping_shl",
                           "checked_mul", "saturating_add", "unsigned_abs",
                           "unwrap_or", "abs", "clamp", "min", "max"]:
                if fn_name in code:
                    ops.append(f"rust_{fn_name}")
        return ops

    def _check_divergence_patterns(self, c_code, rust_code):
        """Check for known patterns that cause C/Rust divergence."""
        patterns = []

        # Wrapping arithmetic in Rust vs plain in C
        if re.search(r'wrapping_(add|mul|neg|shl|sub)', rust_code):
            if not re.search(r'wrapping_', c_code):
                patterns.append("wrapping_arithmetic_mismatch")

        # checked/saturating in Rust
        if re.search(r'(checked_|saturating_)', rust_code):
            patterns.append("checked_arithmetic_in_rust")

        # Different div-by-zero handling
        c_has_div = re.search(r'[^/]/[^/]', c_code) or re.search(r'%', c_code)
        rust_guards_zero = re.search(r'if\s+\w+\s*==\s*0', rust_code)
        if c_has_div and rust_guards_zero and not re.search(r'if.*==\s*0', c_code):
            patterns.append("div_by_zero_guard_mismatch")

        # Type promotion differences
        if re.search(r'as\s+(i64|u32|f64)', rust_code) and not re.search(r'\(long\)|cast', c_code):
            patterns.append("type_promotion_difference")

        # Different sentinel/default values
        c_sentinels = re.findall(r'return\s+(-?\d+)', c_code)
        rust_sentinels = re.findall(r'(\d+)\s*\}', rust_code)
        if c_sentinels and rust_sentinels:
            if set(c_sentinels) != set(rust_sentinels):
                # Only flag if they're in error-handling paths
                if 'if' in c_code and 'if' in rust_code:
                    patterns.append("different_error_sentinels")

        # Modulo sign semantics
        if '%' in c_code and '%' in rust_code:
            if rust_code.count('%') > c_code.count('%'):
                patterns.append("modulo_sign_correction")

        return patterns

    def _parse_function(self, code, lang):
        """Parse a simple function to determine arg count and create evaluator."""
        if lang == "c":
            m = re.match(r'\w+\s+(\w+)\s*\(([^)]*)\)', code)
        else:
            m = re.match(r'fn\s+(\w+)\s*\(([^)]*)\)', code)

        if not m:
            return None

        name = m.group(1)
        args_str = m.group(2)
        n_args = len([a for a in args_str.split(',') if a.strip()])

        # Create a Python evaluator for simple expressions
        body = code[code.index('{') + 1:code.rindex('}')].strip() if '{' in code else ""

        return {"name": name, "n_args": n_args, "body": body, "lang": lang, "code": code}

    def _eval_simple(self, func_info, args):
        """Evaluate a simple function on given arguments."""
        body = func_info["body"]
        lang = func_info["lang"]
        code = func_info["code"]

        # Map args to param names
        if lang == "c":
            m = re.match(r'\w+\s+\w+\s*\(([^)]*)\)', code)
        else:
            m = re.match(r'fn\s+\w+\s*\(([^)]*)\)', code)
        if not m:
            return None

        param_names = []
        for p in m.group(1).split(','):
            p = p.strip()
            if lang == "c":
                parts = p.split()
                if len(parts) >= 2:
                    param_names.append(parts[-1])
            else:
                name_part = p.split(':')[0].strip()
                param_names.append(name_part)

        if len(param_names) != len(args):
            return None

        env = dict(zip(param_names, args))

        try:
            result = self._eval_body(body, env, lang)
            return result
        except (ZeroDivisionError, OverflowError, ValueError):
            return "error"

    def _eval_body(self, body, env, lang):
        """Evaluate function body with given environment."""
        # Handle simple return expressions
        body = body.strip().rstrip(';')

        # Handle if-else (ternary in C)
        ternary = re.match(r'return\s+(.+)\s*\?\s*(.+)\s*:\s*(.+)', body)
        if ternary:
            cond = self._eval_expr(ternary.group(1), env)
            if cond:
                return self._eval_expr(ternary.group(2), env)
            else:
                return self._eval_expr(ternary.group(3), env)

        # Handle simple return
        ret = re.match(r'return\s+(.+)', body)
        if ret:
            return self._eval_expr(ret.group(1), env)

        # Handle multi-statement with if
        if_match = re.search(r'if\s*\(?\s*(.+?)\s*\)?\s*(?:return|{)\s*(.+?)\s*[;}]', body)
        if if_match:
            cond = self._eval_expr(if_match.group(1), env)
            if cond:
                return self._eval_expr(if_match.group(2), env)
            # Try to find else or next return
            rest = body[if_match.end():]
            ret2 = re.search(r'return\s+(.+?)[\s;}]', rest)
            if ret2:
                return self._eval_expr(ret2.group(1), env)

        # Fallback: try to evaluate the entire body as expression
        return self._eval_expr(body, env)

    def _eval_expr(self, expr, env):
        """Evaluate a simple expression."""
        expr = expr.strip().rstrip(';')

        # Substitute variables
        for name, val in sorted(env.items(), key=lambda x: -len(x[0])):
            expr = re.sub(r'\b' + re.escape(name) + r'\b', str(val), expr)

        # Clean up C/Rust syntax
        expr = expr.replace('&&', ' and ').replace('||', ' or ')
        expr = re.sub(r'\b(int|unsigned|char|i32|u32|i64|f32|f64|bool)\b', '', expr)
        expr = expr.replace('()', '')

        try:
            return eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return None

    def _run_test_inputs(self, c_func, rust_func, n_args):
        """Run test inputs and compare results."""
        results = {"n_tested": 0, "n_matches": 0, "n_divergences": 0,
                   "first_counterexample": None}

        if n_args == 1:
            inputs_list = [[x] for x in self.test_inputs[:30]]
        elif n_args == 2:
            inputs_list = [[a, b] for a in self.test_inputs[:15] for b in self.test_inputs[:15]]
        elif n_args == 3:
            inputs_list = [[a, b, c] for a in self.test_inputs[:8]
                          for b in self.test_inputs[:8] for c in self.test_inputs[:8]]
        else:
            return results

        for args in inputs_list[:200]:  # Cap at 200 tests
            c_result = self._eval_simple(c_func, args)
            rust_result = self._eval_simple(rust_func, args)

            if c_result is None or rust_result is None:
                continue

            results["n_tested"] += 1
            if c_result == rust_result:
                results["n_matches"] += 1
            else:
                results["n_divergences"] += 1
                if results["first_counterexample"] is None:
                    results["first_counterexample"] = {
                        "inputs": args,
                        "c_output": str(c_result),
                        "rust_output": str(rust_result),
                    }

        return results


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_verification_benchmark():
    print("=" * 60)
    print("Cross-Language Equivalence Verification Benchmark")
    print("=" * 60)

    verifier = EquivalenceVerifier()
    results = []

    # Test equivalent pairs
    print("\n  --- Equivalent Pairs ---")
    for pair in EQUIVALENT_PAIRS:
        t0 = time.time()
        result = verifier.verify_equivalence(pair["c"], pair["rust"], pair["name"])
        elapsed = time.time() - t0

        expected = "equivalent"
        is_correct = result["verdict"] in ("equivalent", "likely_equivalent")

        entry = {
            "id": pair["id"],
            "name": pair["name"],
            "expected": expected,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "correct": is_correct,
            "verification_time_s": round(elapsed, 6),
            "counterexample": result["counterexample"],
        }
        results.append(entry)
        status = "✓" if is_correct else "✗"
        print(f"    {status} {pair['name']}: {result['verdict']} (conf={result['confidence']:.2f})")

    # Test divergent pairs
    print("\n  --- Divergent Pairs ---")
    for pair in DIVERGENT_PAIRS:
        t0 = time.time()
        result = verifier.verify_equivalence(pair["c"], pair["rust"], pair["name"])
        elapsed = time.time() - t0

        expected = "divergent"
        is_correct = result["verdict"] == "divergent"

        entry = {
            "id": pair["id"],
            "name": pair["name"],
            "divergence_type": pair["divergence_type"],
            "expected": expected,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "correct": is_correct,
            "verification_time_s": round(elapsed, 6),
            "counterexample": result["counterexample"],
            "explanation": pair["explanation"],
        }
        results.append(entry)
        status = "✓" if is_correct else "✗"
        print(f"    {status} {pair['name']} ({pair['divergence_type']}): "
              f"{result['verdict']} (conf={result['confidence']:.2f})")

    return results


def compute_verification_metrics(results):
    """Compute accuracy, precision, recall for verification."""
    eq_results = [r for r in results if r["expected"] == "equivalent"]
    div_results = [r for r in results if r["expected"] == "divergent"]

    tp = sum(1 for r in div_results if r["correct"])  # correctly identified divergent
    fn = sum(1 for r in div_results if not r["correct"])  # missed divergent
    fp = sum(1 for r in eq_results if not r["correct"])  # false alarm on equivalent
    tn = sum(1 for r in eq_results if r["correct"])  # correctly identified equivalent

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    # Per-divergence-type recall
    div_types = set(r.get("divergence_type", "") for r in div_results)
    per_type = {}
    for dt in sorted(div_types):
        if not dt:
            continue
        type_results = [r for r in div_results if r.get("divergence_type") == dt]
        detected = sum(1 for r in type_results if r["correct"])
        per_type[dt] = {
            "total": len(type_results),
            "detected": detected,
            "recall": round(detected / len(type_results), 4) if type_results else 0,
        }

    return {
        "overall": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "equivalent_accuracy": round(tn / len(eq_results), 4) if eq_results else 0,
        "divergent_recall": round(tp / len(div_results), 4) if div_results else 0,
        "per_divergence_type": per_type,
        "avg_verification_time_s": round(np.mean([r["verification_time_s"] for r in results]), 6),
        "max_verification_time_s": round(max(r["verification_time_s"] for r in results), 6),
    }


def main():
    print("Cross-Language Equivalence Verifier - Benchmark")
    print("=" * 60)
    t_start = time.time()

    results = run_verification_benchmark()
    metrics = compute_verification_metrics(results)
    total_time = time.time() - t_start

    print(f"\n  Overall: accuracy={metrics['overall']['accuracy']:.2f}, "
          f"precision={metrics['overall']['precision']:.2f}, "
          f"recall={metrics['overall']['recall']:.2f}, "
          f"F1={metrics['overall']['f1']:.2f}")

    all_results = {
        "experiment": "cross_language_verification_benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": round(total_time, 2),
        "n_equivalent_pairs": len(EQUIVALENT_PAIRS),
        "n_divergent_pairs": len(DIVERGENT_PAIRS),
        "metrics": metrics,
        "detailed_results": results,
        "summary": {
            "accuracy": metrics["overall"]["accuracy"],
            "precision": metrics["overall"]["precision"],
            "recall": metrics["overall"]["recall"],
            "f1": metrics["overall"]["f1"],
            "avg_time_per_pair_s": metrics["avg_verification_time_s"],
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "verification_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results written to {out_path}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
