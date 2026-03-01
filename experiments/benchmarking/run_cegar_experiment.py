#!/usr/bin/env python3
"""
Run CEGAR evaluation experiments.

Tests the CEGAR loop on C functions of varying complexity and divergence types.
Measures: convergence rate, iteration count, bug classification, LLM repairability.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cegar_engine import CEGAREngine, analyze_cegar_results, CEGARResult

# C functions to translate — mix of easy and hard
CEGAR_TEST_FUNCTIONS = [
    # Easy: simple arithmetic (LLM should get right or fix quickly)
    ("abs_val", "int abs_val(int x) { return x < 0 ? -x : x; }"),
    ("max2", "int max2(int a, int b) { return a > b ? a : b; }"),
    ("min2", "int min2(int a, int b) { return a < b ? a : b; }"),
    ("clamp", "int clamp(int x, int lo, int hi) { if (x < lo) return lo; if (x > hi) return hi; return x; }"),
    ("sign", "int sign(int x) { return (x > 0) - (x < 0); }"),
    ("is_positive", "int is_positive(int x) { return x > 0; }"),
    ("is_even", "int is_even(int x) { return (x & 1) == 0; }"),
    
    # Medium: overflow-prone (LLM needs wrapping ops)
    ("add_ints", "int add_ints(int a, int b) { return a + b; }"),
    ("sub_ints", "int sub_ints(int a, int b) { return a - b; }"),
    ("mul_ints", "int mul_ints(int a, int b) { return a * b; }"),
    ("negate", "int negate(int x) { return -x; }"),
    ("double_val", "int double_val(int x) { return x + x; }"),
    ("square", "int square(int x) { return x * x; }"),
    ("inc", "int inc(int x) { return x + 1; }"),
    ("dec", "int dec(int x) { return x - 1; }"),
    
    # Division edge cases
    ("safe_div", "int safe_div(int a, int b) { return b != 0 ? a / b : 0; }"),
    ("safe_mod", "int safe_mod(int a, int b) { return b != 0 ? a % b : 0; }"),
    ("div_ints", "int div_ints(int a, int b) { return a / b; }"),
    
    # Shift operations
    ("shl", "int shl(int x, int n) { return x << n; }"),
    ("shr", "int shr(int x, int n) { return x >> n; }"),
    ("bit_set", "int bit_set(int x, int n) { return x | (1 << (n & 31)); }"),
    
    # Bitwise (should be straightforward)
    ("bit_and", "int bit_and(int a, int b) { return a & b; }"),
    ("bit_or", "int bit_or(int a, int b) { return a | b; }"),
    ("bit_xor", "int bit_xor(int a, int b) { return a ^ b; }"),
    ("bit_not", "int bit_not(int x) { return ~x; }"),
    
    # Multi-operation
    ("poly2", "int poly2(int x) { return 2*x*x + 3*x + 1; }"),
    ("dist_sq", "int dist_sq(int x1, int y1, int x2, int y2) { return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1); }"),
    ("midpoint", "int midpoint(int a, int b) { return (a + b) / 2; }"),
    ("avg3", "int avg3(int a, int b, int c) { return (a + b + c) / 3; }"),
    ("dot2d", "int dot2d(int ax, int ay, int bx, int by) { return ax*bx + ay*by; }"),
    
    # Control flow
    ("classify", "int classify(int x) { if (x > 100) return 3; if (x > 10) return 2; if (x > 0) return 1; return 0; }"),
    ("fizzbuzz_val", "int fizzbuzz(int x) { if (x % 15 == 0) return 3; if (x % 5 == 0) return 2; if (x % 3 == 0) return 1; return 0; }"),
    
    # Real-world patterns
    ("align_up", "unsigned int align_up(unsigned int x, unsigned int a) { return (x + a - 1) & ~(a - 1); }"),
    ("djb2_step", "unsigned int djb2_step(unsigned int h, unsigned int c) { return h * 33 + c; }"),
    ("fnv1a_step", "unsigned int fnv1a_step(unsigned int h, unsigned char b) { return (h ^ b) * 16777619u; }"),
    ("pack_rgb", "unsigned int pack_rgb(unsigned int r, unsigned int g, unsigned int b) { return ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF); }"),
    ("clamp_u8", "int clamp_u8(int x) { if (x < 0) return 0; if (x > 255) return 255; return x; }"),
    ("sat_sub", "unsigned int sat_sub(unsigned int a, unsigned int b) { return a > b ? a - b : 0; }"),
    ("map_err", "int map_err(int code) { if (code == 0) return 0; if (code == -1) return 1; if (code == -2) return 2; return 255; }"),
    ("validate", "int validate(int x) { if (x < 0) return -1; if (x > 100) return -2; if (x == 42) return 1; return 0; }"),
]


def main():
    print(f"CEGAR Evaluation Experiment", file=sys.stderr)
    print(f"Functions: {len(CEGAR_TEST_FUNCTIONS)}", file=sys.stderr)
    print(f"Model: gpt-4.1-nano", file=sys.stderr)
    print(f"Max iterations: 5", file=sys.stderr)
    print(f"=" * 60, file=sys.stderr)

    engine = CEGAREngine(
        model="gpt-4.1-nano",
        max_iterations=5,
        timeout_ms=10000,
    )

    results = []
    for idx, (name, c_code) in enumerate(CEGAR_TEST_FUNCTIONS):
        print(f"[{idx+1}/{len(CEGAR_TEST_FUNCTIONS)}] {name}...", end=" ", file=sys.stderr, flush=True)
        try:
            result = engine.run(c_code, name)
            status = "✓" if result.converged else "✗"
            print(f"{status} ({result.total_iterations} iters, {result.total_time_ms:.0f}ms)",
                  file=sys.stderr)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            results.append(CEGARResult(
                func_name=name, c_code=c_code, converged=False,
                final_verdict="error", bug_class="error",
            ))

    # Analysis
    analysis = analyze_cegar_results(results)

    # Detailed results
    detailed = []
    for r in results:
        detailed.append({
            "func_name": r.func_name,
            "converged": r.converged,
            "final_verdict": r.final_verdict,
            "total_iterations": r.total_iterations,
            "total_time_ms": round(r.total_time_ms, 1),
            "bug_class": r.bug_class,
            "llm_repairable": r.llm_repairable,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "verdict": it.verdict,
                    "divergence_class": it.divergence_class,
                    "rust_code": it.rust_code[:200],
                }
                for it in r.iterations
            ],
        })

    output = {
        "experiment": "CEGAR Evaluation Framework",
        "model": "gpt-4.1-nano",
        "max_iterations": 5,
        "num_functions": len(CEGAR_TEST_FUNCTIONS),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis": analysis,
        "detailed_results": detailed,
    }

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"RESULTS SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Total functions: {analysis['total_pairs']}", file=sys.stderr)
    print(f"Equivalent on first try: {analysis['equivalent_on_first_try']}", file=sys.stderr)
    print(f"Converged (CEGAR fixed): {analysis['converged']}", file=sys.stderr)
    print(f"Convergence rate: {analysis['convergence_rate']}%", file=sys.stderr)
    print(f"Remained divergent: {analysis['remained_divergent']}", file=sys.stderr)
    print(f"Errors: {analysis['errors']}", file=sys.stderr)
    print(f"Avg iterations: {analysis['avg_iterations']}", file=sys.stderr)
    print(f"Avg time: {analysis['avg_time_ms']:.0f}ms", file=sys.stderr)
    print(f"\nBug classification: {analysis['bug_classification']}", file=sys.stderr)
    print(f"Convergence curve: {analysis['convergence_curve']}", file=sys.stderr)
    print(f"Repair by class: {json.dumps(analysis['repair_by_class'], indent=2)}", file=sys.stderr)

    # Write output
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/cegar_experiment.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults written to experiments/results/cegar_experiment.json", file=sys.stderr)

    # Also write JSON to stdout
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
