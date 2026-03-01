#!/usr/bin/env python3
"""Example: Verify a C2Rust translation of a small library function.

Simulates the workflow of verifying that a C function and its C2Rust
translation are semantically equivalent.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.cli.config import VerifyConfig
from src.cli.pipeline import VerificationPipeline, PipelinePhase, PipelineStatus
from src.cli.reporter import VerdictKind, ReportWriter


# ---------------------------------------------------------------------------
# Original C library: string utilities
# ---------------------------------------------------------------------------

C_LIBRARY = """
#include <stddef.h>

/* Count occurrences of character c in string s */
int count_char(const char *s, char c) {
    int count = 0;
    while (*s) {
        if (*s == c) {
            count++;
        }
        s++;
    }
    return count;
}

/* Safe bounded string copy */
int safe_strcpy(char *dest, const char *src, int dest_size) {
    if (dest_size <= 0) return -1;
    int i = 0;
    while (i < dest_size - 1 && src[i] != '\\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\\0';
    return i;
}

/* Convert lowercase ASCII to uppercase */
char to_upper(char c) {
    if (c >= 'a' && c <= 'z') {
        return c - ('a' - 'A');
    }
    return c;
}

/* Integer absolute value with overflow check */
int safe_abs(int x) {
    if (x < 0) {
        if (x == (-2147483647 - 1)) {  /* INT_MIN */
            return 2147483647;  /* Saturate instead of UB */
        }
        return -x;
    }
    return x;
}

/* Simple hash function */
unsigned int simple_hash(const char *s) {
    unsigned int hash = 5381;
    int c;
    while ((c = *s++)) {
        hash = hash * 33 + c;
    }
    return hash;
}
"""

# ---------------------------------------------------------------------------
# C2Rust translated Rust code
# ---------------------------------------------------------------------------

RUST_TRANSLATION = """
/// C2Rust translation of count_char
pub fn count_char(s: *const u8, c: u8) -> i32 {
    unsafe {
        let mut count: i32 = 0;
        let mut p = s;
        while *p != 0 {
            if *p == c {
                count = count.wrapping_add(1);
            }
            p = p.offset(1);
        }
        count
    }
}

/// C2Rust translation of safe_strcpy
pub fn safe_strcpy(dest: *mut u8, src: *const u8, dest_size: i32) -> i32 {
    unsafe {
        if dest_size <= 0 {
            return -1;
        }
        let mut i: i32 = 0;
        while i < dest_size.wrapping_sub(1) && *src.offset(i as isize) != 0 {
            *dest.offset(i as isize) = *src.offset(i as isize);
            i = i.wrapping_add(1);
        }
        *dest.offset(i as isize) = 0;
        i
    }
}

/// C2Rust translation of to_upper
pub fn to_upper(c: u8) -> u8 {
    if c >= b'a' && c <= b'z' {
        c.wrapping_sub(b'a'.wrapping_sub(b'A'))
    } else {
        c
    }
}

/// C2Rust translation of safe_abs
pub fn safe_abs(x: i32) -> i32 {
    if x < 0 {
        if x == i32::MIN {
            return i32::MAX;  // Saturate
        }
        return x.wrapping_neg();
    }
    x
}

/// C2Rust translation of simple_hash
pub fn simple_hash(s: *const u8) -> u32 {
    unsafe {
        let mut hash: u32 = 5381;
        let mut p = s;
        while *p != 0 {
            let c = *p as u32;
            hash = hash.wrapping_mul(33).wrapping_add(c);
            p = p.offset(1);
        }
        hash
    }
}
"""


def verify_c2rust_translation():
    """Verify each function in the C2Rust translation."""
    print("=" * 70)
    print("C2Rust Translation Verification")
    print("=" * 70)
    print()
    print("Verifying that C2Rust-generated Rust code is semantically")
    print("equivalent to the original C library functions.")
    print()

    config = VerifyConfig.fast()

    phases_log = []

    def progress_callback(phase, status, msg):
        phases_log.append((phase.value, status.value))

    functions = [
        ("count_char", "Counts character occurrences in a string"),
        ("safe_strcpy", "Bounded string copy"),
        ("to_upper", "ASCII lowercase to uppercase"),
        ("safe_abs", "Integer absolute value with overflow handling"),
        ("simple_hash", "DJB2 hash function"),
    ]

    results = []

    for func_name, description in functions:
        print(f"  [{func_name}] {description}")
        print(f"  Verifying...", end=" ", flush=True)

        try:
            pipeline = VerificationPipeline(config)
            pipeline.set_progress_callback(progress_callback)
            report = pipeline.verify(C_LIBRARY, RUST_TRANSLATION, func_name, func_name)

            symbol = report.verdict.kind.symbol
            verdict = report.verdict.kind.value
            print(f"{symbol} {verdict}")

            if report.verdict.reason:
                print(f"    Reason: {report.verdict.reason}")

            if report.counterexamples:
                for i, ce in enumerate(report.counterexamples[:3]):
                    print(f"    Counterexample #{i+1}: {ce.category.value}")
                    for inp in ce.inputs:
                        print(f"      {inp.name} = {inp.c_value}")

            if report.warnings:
                for w in report.warnings[:2]:
                    print(f"    Warning: {w}")

            results.append({
                "function": func_name,
                "description": description,
                "verdict": verdict,
                "confidence": report.verdict.confidence,
                "counterexamples": len(report.counterexamples),
                "time": report.timing.total_seconds,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "function": func_name,
                "verdict": "error",
                "error": str(e),
            })

        print()

    # Summary
    print("=" * 70)
    print("Summary:")
    equivalent = sum(1 for r in results if r["verdict"] == "equivalent")
    divergent = sum(1 for r in results if r["verdict"] == "divergent")
    unknown = sum(1 for r in results if r["verdict"] == "unknown")
    errors = sum(1 for r in results if r["verdict"] == "error")
    print(f"  {equivalent} equivalent, {divergent} divergent, "
          f"{unknown} unknown, {errors} errors")
    total_time = sum(r.get("time", 0) for r in results)
    print(f"  Total verification time: {total_time:.2f}s")
    print("=" * 70)

    # Write detailed results
    output_path = os.path.join(os.path.dirname(__file__), "c2rust_verify_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results: {output_path}")

    return 0


def main():
    return verify_c2rust_translation()


if __name__ == "__main__":
    sys.exit(main())
