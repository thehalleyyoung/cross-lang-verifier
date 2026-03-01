#!/usr/bin/env python3
"""Example: Verify equivalence of simple arithmetic functions.

Demonstrates the basic verification workflow: parse C and Rust sources,
run the pipeline, and inspect the verdict.
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cli.config import VerifyConfig
from src.cli.pipeline import VerificationPipeline
from src.cli.reporter import VerdictKind


# ---------------------------------------------------------------------------
# Example C source: simple arithmetic
# ---------------------------------------------------------------------------

C_SOURCE = """
int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int negate(int x) {
    return -x;
}

int abs_val(int x) {
    if (x < 0) return -x;
    return x;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
"""

# ---------------------------------------------------------------------------
# Equivalent Rust source (using wrapping arithmetic)
# ---------------------------------------------------------------------------

RUST_SOURCE = """
pub fn add(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

pub fn multiply(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

pub fn negate(x: i32) -> i32 {
    x.wrapping_neg()
}

pub fn abs_val(x: i32) -> i32 {
    if x < 0 { x.wrapping_neg() } else { x }
}

pub fn max(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

pub fn min(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

pub fn clamp(x: i32, lo: i32, hi: i32) -> i32 {
    if x < lo { lo }
    else if x > hi { hi }
    else { x }
}
"""


def verify_pair(c_source: str, rust_source: str, func_name: str,
                config: VerifyConfig = None) -> dict:
    """Verify a single function pair and return the result."""
    if config is None:
        config = VerifyConfig.fast()

    pipeline = VerificationPipeline(config)
    report = pipeline.verify(c_source, rust_source, func_name, func_name)

    return {
        "function": func_name,
        "verdict": report.verdict.kind.value,
        "confidence": report.verdict.confidence,
        "reason": report.verdict.reason,
        "counterexamples": len(report.counterexamples),
        "warnings": report.warnings,
    }


def main():
    """Run verification on all function pairs."""
    print("=" * 60)
    print("Simple Arithmetic Equivalence Verification")
    print("=" * 60)
    print()

    functions = ["add", "subtract", "multiply", "negate",
                 "abs_val", "max", "min", "clamp"]

    config = VerifyConfig.fast()
    results = []

    for func_name in functions:
        print(f"Verifying: {func_name}...", end=" ", flush=True)
        try:
            result = verify_pair(C_SOURCE, RUST_SOURCE, func_name, config)
            symbol = {"equivalent": "✓", "divergent": "✗", "unknown": "?"}
            s = symbol.get(result["verdict"], "?")
            print(f"{s} {result['verdict']} (confidence: {result['confidence']:.0%})")
            if result["reason"]:
                print(f"  Reason: {result['reason']}")
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"function": func_name, "verdict": "error", "error": str(e)})

    # Summary
    print()
    print("=" * 60)
    verdicts = [r["verdict"] for r in results]
    print(f"Results: {verdicts.count('equivalent')} equivalent, "
          f"{verdicts.count('divergent')} divergent, "
          f"{verdicts.count('unknown')} unknown, "
          f"{verdicts.count('error')} errors")
    print("=" * 60)

    # Write JSON results
    output_path = os.path.join(os.path.dirname(__file__), "simple_verify_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results: {output_path}")

    return 0 if all(r["verdict"] != "error" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
