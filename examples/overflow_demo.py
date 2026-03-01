#!/usr/bin/env python3
"""Example: Demonstrate integer overflow divergence detection.

Shows how C and Rust handle integer overflow differently:
- C: signed overflow is undefined behavior (often wraps in practice)
- Rust debug: signed overflow panics
- Rust release: signed overflow wraps (like C)

The verifier detects these semantic divergences.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cli.config import VerifyConfig
from src.cli.pipeline import VerificationPipeline
from src.cli.reporter import VerdictKind, DivergenceCategory


# ---------------------------------------------------------------------------
# C source: relies on wrapping overflow behavior
# ---------------------------------------------------------------------------

C_OVERFLOW = """
/* In C, signed integer overflow is technically UB, but most compilers
   produce wrapping behavior. C2Rust translations often rely on this. */

int add_may_overflow(int a, int b) {
    return a + b;  /* UB if overflow */
}

int mul_may_overflow(int a, int b) {
    return a * b;  /* UB if overflow */
}

int negate_may_overflow(int x) {
    return -x;  /* UB if x == INT_MIN */
}

int shift_may_overflow(int x, int shift) {
    return x << shift;  /* UB if shift >= 32 or shift < 0 */
}

unsigned int unsigned_add(unsigned int a, unsigned int b) {
    return a + b;  /* Well-defined: wraps modulo 2^32 */
}

int accumulate(int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];  /* May overflow */
    }
    return sum;
}
"""

# ---------------------------------------------------------------------------
# Rust source: explicit overflow handling
# ---------------------------------------------------------------------------

RUST_OVERFLOW = """
/// Rust uses wrapping_add to match C's wrapping behavior.
/// In debug mode, regular + would panic on overflow.
pub fn add_may_overflow(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

pub fn mul_may_overflow(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

pub fn negate_may_overflow(x: i32) -> i32 {
    x.wrapping_neg()
}

pub fn shift_may_overflow(x: i32, shift: i32) -> i32 {
    // Rust masks shift amount to width - 1
    x.wrapping_shl(shift as u32)
}

pub fn unsigned_add(a: u32, b: u32) -> u32 {
    a.wrapping_add(b)
}

pub fn accumulate(arr: &[i32]) -> i32 {
    let mut sum: i32 = 0;
    for &x in arr.iter() {
        sum = sum.wrapping_add(x);
    }
    sum
}
"""

# ---------------------------------------------------------------------------
# Rust source without wrapping: will diverge on overflow inputs
# ---------------------------------------------------------------------------

RUST_CHECKED = """
/// Using checked arithmetic -- diverges from C on overflow.
pub fn add_may_overflow(a: i32, b: i32) -> i32 {
    a + b  // Panics in debug mode on overflow!
}

pub fn mul_may_overflow(a: i32, b: i32) -> i32 {
    a * b  // Panics in debug mode on overflow!
}

pub fn negate_may_overflow(x: i32) -> i32 {
    -x  // Panics if x == i32::MIN in debug mode
}

pub fn shift_may_overflow(x: i32, shift: i32) -> i32 {
    x << shift  // Panics if shift >= 32
}

pub fn unsigned_add(a: u32, b: u32) -> u32 {
    a + b  // Panics in debug mode on overflow!
}

pub fn accumulate(arr: &[i32]) -> i32 {
    let mut sum: i32 = 0;
    for &x in arr.iter() {
        sum += x;  // Panics on overflow in debug
    }
    sum
}
"""


def run_overflow_demo():
    """Run the overflow divergence detection demo."""
    print("=" * 70)
    print("Integer Overflow Divergence Detection Demo")
    print("=" * 70)
    print()

    config = VerifyConfig.fast()
    pipeline = VerificationPipeline(config)

    # Part 1: C vs wrapping Rust (should be equivalent)
    print("Part 1: C vs Rust (wrapping arithmetic)")
    print("-" * 50)

    functions = ["add_may_overflow", "mul_may_overflow",
                 "negate_may_overflow", "unsigned_add"]

    for func in functions:
        print(f"  {func}: ", end="", flush=True)
        try:
            report = pipeline.verify(C_OVERFLOW, RUST_OVERFLOW, func, func)
            symbol = report.verdict.kind.symbol
            print(f"{symbol} {report.verdict.kind.value}")
        except Exception as e:
            print(f"Error: {e}")

    print()

    # Part 2: C vs checked Rust (should diverge on overflow inputs)
    print("Part 2: C vs Rust (checked arithmetic — expects divergence)")
    print("-" * 50)

    config2 = VerifyConfig.fast()
    pipeline2 = VerificationPipeline(config2)

    for func in functions:
        print(f"  {func}: ", end="", flush=True)
        try:
            report = pipeline2.verify(C_OVERFLOW, RUST_CHECKED, func, func)
            symbol = report.verdict.kind.symbol
            print(f"{symbol} {report.verdict.kind.value}")
            if report.counterexamples:
                ce = report.counterexamples[0]
                print(f"    Counterexample: {ce.description or 'overflow input found'}")
                for inp in ce.inputs:
                    print(f"      {inp.name} = {inp.c_value}")
        except Exception as e:
            print(f"Error: {e}")

    print()

    # Summary of overflow categories
    print("Overflow Divergence Categories:")
    print("-" * 50)
    for cat in [DivergenceCategory.INTEGER_OVERFLOW,
                DivergenceCategory.SHIFT_OVERFLOW,
                DivergenceCategory.SIGNED_UNSIGNED_MISMATCH]:
        print(f"  {cat.value}: {cat.description}")

    print()
    print("Key insight: C2Rust translations using wrapping_* methods preserve")
    print("C semantics. Direct arithmetic in Rust debug mode diverges on overflow.")
    print("=" * 70)


def main():
    run_overflow_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
