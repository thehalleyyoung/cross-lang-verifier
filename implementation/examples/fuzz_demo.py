#!/usr/bin/env python3
"""Example: Run differential fuzzing on a function pair.

Demonstrates the fuzzer subsystem by running differential fuzzing
to find inputs where C and Rust implementations diverge.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.cli.config import VerifyConfig
from src.cli.pipeline import VerificationPipeline
from src.cli.reporter import VerdictKind, DivergenceCategory


# ---------------------------------------------------------------------------
# C source with potential divergence
# ---------------------------------------------------------------------------

C_SOURCE = """
/* Division that may behave differently at boundaries */
int safe_div(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

/* Modulo with sign difference potential */
int modulo(int a, int b) {
    if (b == 0) return 0;
    return a % b;
}

/* Shift that may have UB in C */
int shift_left(int x, int amount) {
    if (amount < 0 || amount >= 32) return 0;
    return x << amount;
}

/* Array bounds — C has no checking */
int array_get(int *arr, int idx, int len) {
    if (idx < 0 || idx >= len) return -1;
    return arr[idx];
}
"""

# ---------------------------------------------------------------------------
# Rust source
# ---------------------------------------------------------------------------

RUST_SOURCE = """
pub fn safe_div(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    a.wrapping_div(b)
}

pub fn modulo(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    a.wrapping_rem(b)
}

pub fn shift_left(x: i32, amount: i32) -> i32 {
    if amount < 0 || amount >= 32 { return 0; }
    x.wrapping_shl(amount as u32)
}

pub fn array_get(arr: &[i32], idx: i32) -> i32 {
    if idx < 0 || idx as usize >= arr.len() { return -1; }
    arr[idx as usize]
}
"""


def run_fuzz_demo():
    """Run differential fuzzing demo."""
    print("=" * 60)
    print("Differential Fuzzing Demo")
    print("=" * 60)
    print()

    # Configure for fuzzing
    config = VerifyConfig.fuzz_only()
    config.fuzzer.seed_count = 500
    config.fuzzer.max_iterations = 5000
    config.fuzzer.use_boundary_values = True
    config.fuzzer.minimize_inputs = True

    functions = [
        "safe_div",
        "modulo",
        "shift_left",
    ]

    all_results = []

    for func_name in functions:
        print(f"Fuzzing: {func_name}")
        print(f"  Seeds: {config.fuzzer.seed_count}")
        print(f"  Max iterations: {config.fuzzer.max_iterations}")
        print(f"  Running...", end=" ", flush=True)

        t0 = time.time()
        try:
            pipeline = VerificationPipeline(config)
            report = pipeline.fuzz_only(C_SOURCE, RUST_SOURCE, func_name, func_name)
            elapsed = time.time() - t0

            symbol = report.verdict.kind.symbol
            print(f"{symbol} {report.verdict.kind.value} ({elapsed:.2f}s)")

            if report.counterexamples:
                print(f"  Found {len(report.counterexamples)} divergence(s):")
                for i, ce in enumerate(report.counterexamples[:5]):
                    print(f"    [{i+1}] {ce.category.value}: {ce.description}")
                    for inp in ce.inputs:
                        print(f"        {inp.name} = {inp.c_value}")
                    print(f"        C output: {ce.c_output}")
                    print(f"        Rust output: {ce.rust_output}")
            else:
                print(f"  No divergences found in {config.fuzzer.max_iterations} iterations")

            # Coverage stats
            cov = report.coverage
            if cov.total_paths > 0:
                print(f"  Coverage: {cov.path_coverage:.1%} paths explored")

            all_results.append({
                "function": func_name,
                "verdict": report.verdict.kind.value,
                "counterexamples": len(report.counterexamples),
                "time_seconds": round(elapsed, 2),
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.2f}s): {e}")
            all_results.append({
                "function": func_name,
                "verdict": "error",
                "error": str(e),
                "time_seconds": round(elapsed, 2),
            })

        print()

    # Summary
    print("=" * 60)
    print("Fuzzing Summary:")
    total_divergences = sum(r.get("counterexamples", 0) for r in all_results)
    total_time = sum(r.get("time_seconds", 0) for r in all_results)
    print(f"  Functions fuzzed: {len(functions)}")
    print(f"  Total divergences found: {total_divergences}")
    print(f"  Total time: {total_time:.2f}s")
    print("=" * 60)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "fuzz_demo_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return 0


def main():
    return run_fuzz_demo()


if __name__ == "__main__":
    sys.exit(main())
