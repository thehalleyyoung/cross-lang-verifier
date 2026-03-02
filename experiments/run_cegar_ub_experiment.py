#!/usr/bin/env python3
"""
CEGAR experiment with improved hints for UB functions.

Tests the improved CEGAR engine with better prompts and UB guidance.
Uses gpt-4.1-nano as specified.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cegar_engine import CEGAREngine, analyze_cegar_results

# UB-heavy benchmark functions for CEGAR
CEGAR_UB_FUNCTIONS = [
    ("add_overflow", """
int add_overflow(int a, int b) {
    return a + b;
}
"""),
    ("mul_overflow", """
int mul_overflow(int a, int b) {
    return a * b;
}
"""),
    ("negate", """
int negate(int x) {
    return -x;
}
"""),
    ("abs_val", """
int abs_val(int x) {
    if (x < 0) return -x;
    return x;
}
"""),
    ("div_safe", """
int div_safe(int a, int b) {
    return a / b;
}
"""),
    ("mod_safe", """
int mod_safe(int a, int b) {
    return a % b;
}
"""),
    ("shift_left", """
int shift_left(int x, int n) {
    return x << n;
}
"""),
    ("shift_right", """
int shift_right(int x, int n) {
    return x >> n;
}
"""),
    ("sum_to_n", """
int sum_to_n(int n) {
    int sum = 0;
    int i;
    for (i = 0; i < n && i < 100; i++) {
        sum += i;
    }
    return sum;
}
"""),
    ("factorial", """
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""),
    ("power", """
int power(int base, int exp) {
    int result = 1;
    int i;
    for (i = 0; i < exp && i < 30; i++) {
        result *= base;
    }
    return result;
}
"""),
    ("average", """
int average(int a, int b) {
    return (a + b) / 2;
}
"""),
    ("clamp", """
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
"""),
    ("bit_count", """
int bit_count(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}
"""),
    ("gcd", """
int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}
"""),
    ("max3", """
int max3(int a, int b, int c) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}
"""),
    ("sign", """
int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
"""),
    ("hash_combine", """
int hash_combine(int seed, int value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}
"""),
    ("subtract_clamp", """
int subtract_clamp(int a, int b) {
    int result = a - b;
    if (result < 0) result = 0;
    return result;
}
"""),
    ("rotate_left", """
unsigned int rotate_left(unsigned int x, int n) {
    n = n & 31;
    return (x << n) | (x >> (32 - n));
}
"""),
]


def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("CEGAR Experiment with Improved UB Hints")
    print(f"Model: gpt-4.1-nano, Max iterations: 5")
    print("=" * 60)

    engine = CEGAREngine(
        model="gpt-4.1-nano",
        max_iterations=5,
        timeout_ms=15000,
    )

    results = []
    for i, (name, c_code) in enumerate(CEGAR_UB_FUNCTIONS):
        print(f"\n[{i+1}/{len(CEGAR_UB_FUNCTIONS)}] {name}...", end=" ", flush=True)
        try:
            result = engine.run(c_code, name)
            results.append(result)
            status = "✓ CONVERGED" if result.converged else f"✗ {result.final_verdict}"
            print(f"{status} ({result.total_iterations} iters, {result.total_time_ms:.0f}ms)")
        except Exception as e:
            print(f"ERROR: {e}")
            from src.cegar_engine import CEGARResult
            results.append(CEGARResult(
                func_name=name, c_code=c_code, converged=False,
                final_verdict="error",
            ))

    # Analyze
    analysis = analyze_cegar_results(results)

    print("\n" + "=" * 60)
    print(f"CEGAR Results Summary")
    print(f"  Total: {analysis['total_pairs']}")
    print(f"  Converged: {analysis['converged']} ({analysis['convergence_rate']}%)")
    print(f"  Equivalent on first try: {analysis['equivalent_on_first_try']}")
    print(f"  Remained divergent: {analysis['remained_divergent']}")
    print(f"  Errors: {analysis['errors']}")
    print(f"  Avg iterations: {analysis['avg_iterations']}")
    print(f"  Avg time: {analysis['avg_time_ms']:.0f}ms")

    # Save
    output = {
        "analysis": analysis,
        "individual_results": [r.to_dict() for r in results],
    }
    with open(os.path.join(results_dir, "cegar_ub_improved.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to experiments/results/cegar_ub_improved.json")


if __name__ == "__main__":
    main()
