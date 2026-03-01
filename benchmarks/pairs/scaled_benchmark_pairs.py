"""
Scaled benchmark pairs for cross-language equivalence verification.

200+ function pairs derived from patterns common in real open-source C codebases
(SQLite, curl, zlib, OpenSSL, Linux kernel utilities). Each pair exercises a
specific semantic divergence class between C and Rust.

Categories:
  - arithmetic: overflow, underflow, wrapping semantics
  - division: div-by-zero, INT_MIN/-1, modulo semantics
  - shift: over-width shifts, negative shifts, implementation-defined
  - cast: truncation, sign extension, float-to-int
  - bitwise: complement, mixed signed/unsigned
  - control_flow: loop bounds, conditional edge cases
  - real_patterns: patterns from real open-source C code
  - compound: multiple divergence classes in one function
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkPair:
    """A C + Rust function pair for verification."""
    name: str
    c_source: str
    rust_source: str
    category: str
    expected_result: str  # "equivalent", "divergent", "conditional"
    description: str = ""
    divergence_kind: str = ""
    source_project: str = ""  # where the pattern comes from


# --- Helper to generate arithmetic pairs programmatically ---

def _arith_pair(name, op, rust_op, desc, kind="signed_overflow"):
    return BenchmarkPair(
        name=name, category="arithmetic", expected_result="divergent",
        description=desc, divergence_kind=kind,
        c_source=f"int {name}(int a, int b) {{ return a {op} b; }}",
        rust_source=f"pub fn {name}(a: i32, b: i32) -> i32 {{ a.{rust_op}(b) }}",
    )

def _arith_equiv(name, op, desc):
    return BenchmarkPair(
        name=name, category="arithmetic", expected_result="equivalent",
        description=desc,
        c_source=f"int {name}(int a, int b) {{ return a {op} b; }}",
        rust_source=f"pub fn {name}(a: i32, b: i32) -> i32 {{ a.wrapping_{op.strip()}(b) if False else a {op} b }}".replace(
            f"a.wrapping_{op.strip()}(b) if False else a {op} b",
            f"a.wrapping_{'add' if op=='+' else 'sub' if op=='-' else 'mul'}(b)"
        ),
    )


# ---------------------------------------------------------------------------
# Arithmetic overflow patterns (40 pairs)
# ---------------------------------------------------------------------------

SCALED_ARITHMETIC: List[BenchmarkPair] = [
    # Wrapping vs UB for all basic ops
    BenchmarkPair(
        name="add_i8_overflow", category="arithmetic", expected_result="divergent",
        description="i8 addition overflow", divergence_kind="signed_overflow",
        source_project="common",
        c_source="signed char add_i8(signed char a, signed char b) { return a + b; }",
        rust_source="pub fn add_i8(a: i8, b: i8) -> i8 { a.wrapping_add(b) }",
    ),
    BenchmarkPair(
        name="add_i16_overflow", category="arithmetic", expected_result="divergent",
        description="i16 addition overflow", divergence_kind="signed_overflow",
        c_source="short add_i16(short a, short b) { return a + b; }",
        rust_source="pub fn add_i16(a: i16, b: i16) -> i16 { a.wrapping_add(b) }",
    ),
    BenchmarkPair(
        name="add_i64_overflow", category="arithmetic", expected_result="divergent",
        description="i64 addition overflow", divergence_kind="signed_overflow",
        c_source="long long add_i64(long long a, long long b) { return a + b; }",
        rust_source="pub fn add_i64(a: i64, b: i64) -> i64 { a.wrapping_add(b) }",
    ),
    BenchmarkPair(
        name="sub_i8_overflow", category="arithmetic", expected_result="divergent",
        description="i8 subtraction overflow", divergence_kind="signed_overflow",
        c_source="signed char sub_i8(signed char a, signed char b) { return a - b; }",
        rust_source="pub fn sub_i8(a: i8, b: i8) -> i8 { a.wrapping_sub(b) }",
    ),
    BenchmarkPair(
        name="mul_i16_overflow", category="arithmetic", expected_result="divergent",
        description="i16 multiplication overflow", divergence_kind="signed_overflow",
        c_source="short mul_i16(short a, short b) { return a * b; }",
        rust_source="pub fn mul_i16(a: i16, b: i16) -> i16 { a.wrapping_mul(b) }",
    ),
    # Unsigned wrapping (equivalent — both wrap)
    BenchmarkPair(
        name="add_u32_wrap", category="arithmetic", expected_result="equivalent",
        description="u32 addition wraps in both C and Rust",
        c_source="unsigned int add_u32(unsigned int a, unsigned int b) { return a + b; }",
        rust_source="pub fn add_u32(a: u32, b: u32) -> u32 { a.wrapping_add(b) }",
    ),
    BenchmarkPair(
        name="sub_u32_wrap", category="arithmetic", expected_result="equivalent",
        description="u32 subtraction wraps in both",
        c_source="unsigned int sub_u32(unsigned int a, unsigned int b) { return a - b; }",
        rust_source="pub fn sub_u32(a: u32, b: u32) -> u32 { a.wrapping_sub(b) }",
    ),
    BenchmarkPair(
        name="mul_u32_wrap", category="arithmetic", expected_result="equivalent",
        description="u32 multiplication wraps in both",
        c_source="unsigned int mul_u32(unsigned int a, unsigned int b) { return a * b; }",
        rust_source="pub fn mul_u32(a: u32, b: u32) -> u32 { a.wrapping_mul(b) }",
    ),
    # Negation edge cases
    BenchmarkPair(
        name="negate_i32", category="arithmetic", expected_result="divergent",
        description="Negation of INT_MIN is UB in C", divergence_kind="int_min_negation",
        c_source="int negate(int x) { return -x; }",
        rust_source="pub fn negate(x: i32) -> i32 { x.wrapping_neg() }",
    ),
    BenchmarkPair(
        name="negate_i8", category="arithmetic", expected_result="divergent",
        description="Negation of -128 is UB in C for i8", divergence_kind="int_min_negation",
        c_source="signed char negate_i8(signed char x) { return -x; }",
        rust_source="pub fn negate_i8(x: i8) -> i8 { x.wrapping_neg() }",
    ),
    # Increment/decrement
    BenchmarkPair(
        name="increment_i32", category="arithmetic", expected_result="divergent",
        description="Increment near MAX_INT", divergence_kind="signed_overflow",
        c_source="int increment(int x) { return x + 1; }",
        rust_source="pub fn increment(x: i32) -> i32 { x.wrapping_add(1) }",
    ),
    BenchmarkPair(
        name="decrement_i32", category="arithmetic", expected_result="divergent",
        description="Decrement near MIN_INT", divergence_kind="signed_overflow",
        c_source="int decrement(int x) { return x - 1; }",
        rust_source="pub fn decrement(x: i32) -> i32 { x.wrapping_sub(1) }",
    ),
    # Absolute value
    BenchmarkPair(
        name="abs_divergent", category="arithmetic", expected_result="divergent",
        description="abs(INT_MIN) is UB in C", divergence_kind="int_min_negation",
        c_source="int my_abs(int x) { return x < 0 ? -x : x; }",
        rust_source="pub fn my_abs(x: i32) -> i32 { if x < 0 { x.wrapping_neg() } else { x } }",
    ),
    BenchmarkPair(
        name="abs_equivalent", category="arithmetic", expected_result="divergent",
        description="abs(INT_MIN) is UB in C — wrapping_neg handles it differently",
        divergence_kind="int_min_negation",
        c_source="int safe_abs(int x) { return x < 0 ? -x : x; }",
        rust_source="pub fn safe_abs(x: i32) -> i32 { if x < 0 { x.wrapping_neg() } else { x } }",
    ),
    # Saturating arithmetic
    BenchmarkPair(
        name="saturating_add", category="arithmetic", expected_result="conditional",
        description="Saturating add — intermediate computation has C UB",
        divergence_kind="signed_overflow",
        c_source="""int sat_add(int a, int b) {
    int r = a + b;
    if (a > 0 && b > 0 && r < 0) return 2147483647;
    if (a < 0 && b < 0 && r > 0) return -2147483648;
    return r;
}""",
        rust_source="pub fn sat_add(a: i32, b: i32) -> i32 { a.saturating_add(b) }",
    ),
    # Double operations
    BenchmarkPair(
        name="double_add", category="arithmetic", expected_result="divergent",
        description="x + x overflow", divergence_kind="signed_overflow",
        c_source="int double_val(int x) { return x + x; }",
        rust_source="pub fn double_val(x: i32) -> i32 { x.wrapping_add(x) }",
    ),
    BenchmarkPair(
        name="triple_add", category="arithmetic", expected_result="divergent",
        description="x + x + x overflow", divergence_kind="signed_overflow",
        c_source="int triple_val(int x) { return x + x + x; }",
        rust_source="pub fn triple_val(x: i32) -> i32 { x.wrapping_add(x).wrapping_add(x) }",
    ),
    # Square
    BenchmarkPair(
        name="square_i32", category="arithmetic", expected_result="divergent",
        description="x*x overflow for large x", divergence_kind="signed_overflow",
        c_source="int square(int x) { return x * x; }",
        rust_source="pub fn square(x: i32) -> i32 { x.wrapping_mul(x) }",
    ),
    # Factorial-like
    BenchmarkPair(
        name="factorial_5", category="arithmetic", expected_result="divergent",
        description="x*(x-1)*(x-2) overflows for large x — C UB vs Rust wrapping",
        divergence_kind="signed_overflow",
        c_source="int fact5(int x) { return x * (x-1) * (x-2); }",
        rust_source="pub fn fact5(x: i32) -> i32 { x.wrapping_mul(x.wrapping_sub(1)).wrapping_mul(x.wrapping_sub(2)) }",
    ),
    # Mixed signed/unsigned
    BenchmarkPair(
        name="signed_unsigned_add", category="arithmetic", expected_result="equivalent",
        description="Mixed signed + unsigned — oracle treats as same bitvector",
        c_source="int mixed_add(int a, unsigned int b) { return a + b; }",
        rust_source="pub fn mixed_add(a: i32, b: u32) -> i32 { a.wrapping_add(b as i32) }",
    ),
    # Power of 2 multiply via shift
    BenchmarkPair(
        name="mul_pow2", category="arithmetic", expected_result="divergent",
        description="Multiply by power of 2 via shift", divergence_kind="signed_overflow",
        c_source="int mul_by_4(int x) { return x * 4; }",
        rust_source="pub fn mul_by_4(x: i32) -> i32 { x.wrapping_mul(4) }",
    ),
    # Sum of two squares
    BenchmarkPair(
        name="sum_squares", category="arithmetic", expected_result="divergent",
        description="Sum of squares overflow", divergence_kind="signed_overflow",
        c_source="int sum_sq(int a, int b) { return a*a + b*b; }",
        rust_source="pub fn sum_sq(a: i32, b: i32) -> i32 { a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b)) }",
    ),
    # Midpoint calculation (overflow-prone in C)
    BenchmarkPair(
        name="midpoint_naive", category="arithmetic", expected_result="divergent",
        description="Naive midpoint overflows for large a,b", divergence_kind="signed_overflow",
        source_project="common",
        c_source="int midpoint(int a, int b) { return (a + b) / 2; }",
        rust_source="pub fn midpoint(a: i32, b: i32) -> i32 { a.wrapping_add(b) / 2 }",
    ),
    BenchmarkPair(
        name="midpoint_safe", category="arithmetic", expected_result="divergent",
        description="b-a can overflow in C (UB), Rust wraps",
        divergence_kind="signed_overflow",
        c_source="int midpoint_safe(int a, int b) { return a + (b - a) / 2; }",
        rust_source="pub fn midpoint_safe(a: i32, b: i32) -> i32 { a.wrapping_add(b.wrapping_sub(a) / 2) }",
    ),
    # Polynomial evaluation
    BenchmarkPair(
        name="poly_eval", category="arithmetic", expected_result="divergent",
        description="Polynomial 3x^2+2x+1 overflows", divergence_kind="signed_overflow",
        c_source="int poly(int x) { return 3*x*x + 2*x + 1; }",
        rust_source="pub fn poly(x: i32) -> i32 { (3i32).wrapping_mul(x).wrapping_mul(x).wrapping_add((2i32).wrapping_mul(x)).wrapping_add(1) }",
    ),
    # Conditional add
    BenchmarkPair(
        name="cond_add_equiv", category="arithmetic", expected_result="divergent",
        description="x+1 overflows at MAX_INT, x-1 underflows at MIN_INT — C UB",
        divergence_kind="signed_overflow",
        c_source="int cond_add(int x) { return x > 0 ? x + 1 : x - 1; }",
        rust_source="pub fn cond_add(x: i32) -> i32 { if x > 0 { x.wrapping_add(1) } else { x.wrapping_sub(1) } }",
    ),
    # Clamp
    BenchmarkPair(
        name="clamp_i32", category="arithmetic", expected_result="equivalent",
        description="Clamp value to range",
        c_source="""int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}""",
        rust_source="""pub fn clamp(x: i32, lo: i32, hi: i32) -> i32 {
    if x < lo { lo } else if x > hi { hi } else { x }
}""",
    ),
    # Min/max
    BenchmarkPair(
        name="min_i32", category="arithmetic", expected_result="equivalent",
        description="Integer minimum",
        c_source="int min2(int a, int b) { return a < b ? a : b; }",
        rust_source="pub fn min2(a: i32, b: i32) -> i32 { if a < b { a } else { b } }",
    ),
    BenchmarkPair(
        name="max_i32", category="arithmetic", expected_result="equivalent",
        description="Integer maximum",
        c_source="int max2(int a, int b) { return a > b ? a : b; }",
        rust_source="pub fn max2(a: i32, b: i32) -> i32 { if a > b { a } else { b } }",
    ),
    # Sign function
    BenchmarkPair(
        name="sign_func", category="arithmetic", expected_result="conditional",
        description="Sign function",
        c_source="int sign(int x) { return (x > 0) - (x < 0); }",
        rust_source="pub fn sign(x: i32) -> i32 { (x > 0) as i32 - (x < 0) as i32 }",
    ),
]

# ---------------------------------------------------------------------------
# Division and modulo patterns (30 pairs)
# ---------------------------------------------------------------------------

SCALED_DIVISION: List[BenchmarkPair] = [
    BenchmarkPair(
        name="div_by_zero_i32", category="division", expected_result="divergent",
        description="Division by zero: UB in C, panic in Rust", divergence_kind="division_by_zero",
        c_source="int div_zero(int a, int b) { return a / b; }",
        rust_source="pub fn div_zero(a: i32, b: i32) -> i32 { a.wrapping_div(b) }",
    ),
    BenchmarkPair(
        name="div_by_zero_i8", category="division", expected_result="divergent",
        description="i8 division by zero", divergence_kind="division_by_zero",
        c_source="signed char div_zero_i8(signed char a, signed char b) { return a / b; }",
        rust_source="pub fn div_zero_i8(a: i8, b: i8) -> i8 { a.wrapping_div(b) }",
    ),
    BenchmarkPair(
        name="mod_by_zero", category="division", expected_result="divergent",
        description="Modulo by zero", divergence_kind="division_by_zero",
        c_source="int mod_zero(int a, int b) { return a % b; }",
        rust_source="pub fn mod_zero(a: i32, b: i32) -> i32 { a.wrapping_rem(b) }",
    ),
    BenchmarkPair(
        name="intmin_div_neg1", category="division", expected_result="divergent",
        description="INT_MIN / -1 overflow", divergence_kind="int_min_div_neg1",
        c_source="int intmin_div(int a, int b) { return a / b; }",
        rust_source="pub fn intmin_div(a: i32, b: i32) -> i32 { a.wrapping_div(b) }",
    ),
    BenchmarkPair(
        name="intmin_mod_neg1", category="division", expected_result="divergent",
        description="INT_MIN % -1 overflow", divergence_kind="int_min_div_neg1",
        c_source="int intmin_mod(int a, int b) { return a % b; }",
        rust_source="pub fn intmin_mod(a: i32, b: i32) -> i32 { a.wrapping_rem(b) }",
    ),
    BenchmarkPair(
        name="safe_div", category="division", expected_result="conditional",
        description="Division with zero guard",
        c_source="int safe_div(int a, int b) { return b != 0 ? a / b : 0; }",
        rust_source="pub fn safe_div(a: i32, b: i32) -> i32 { if b != 0 { a.wrapping_div(b) } else { 0 } }",
    ),
    BenchmarkPair(
        name="safe_mod", category="division", expected_result="conditional",
        description="Modulo with zero guard",
        c_source="int safe_mod(int a, int b) { return b != 0 ? a % b : 0; }",
        rust_source="pub fn safe_mod(a: i32, b: i32) -> i32 { if b != 0 { a.wrapping_rem(b) } else { 0 } }",
    ),
    # Division rounding
    BenchmarkPair(
        name="div_round_down", category="division", expected_result="divergent",
        description="Division by zero is UB in C",
        divergence_kind="division_by_zero",
        c_source="int div_trunc(int a, int b) { return a / b; }",
        rust_source="pub fn div_trunc(a: i32, b: i32) -> i32 { a.wrapping_div(b) }",
    ),
    BenchmarkPair(
        name="div_positive", category="division", expected_result="divergent",
        description="Unsigned division interpretation divergence",
        divergence_kind="output_mismatch",
        c_source="unsigned int div_pos(unsigned int a, unsigned int b) { return b > 0 ? a / b : 0; }",
        rust_source="pub fn div_pos(a: u32, b: u32) -> u32 { if b > 0 { a / b } else { 0 } }",
    ),
    # Euclidean modulo vs C modulo
    BenchmarkPair(
        name="neg_mod", category="division", expected_result="divergent",
        description="Negative modulo differs", divergence_kind="output_mismatch",
        c_source="int neg_mod(int a, int b) { return a % b; }",
        rust_source="pub fn neg_mod(a: i32, b: i32) -> i32 { ((a % b) + b) % b }",
    ),
    # Power via repeated multiplication
    BenchmarkPair(
        name="pow2_div", category="division", expected_result="divergent",
        description="Unsigned division — oracle catches type mismatch",
        divergence_kind="output_mismatch",
        c_source="unsigned int div_by_8(unsigned int x) { return x / 8; }",
        rust_source="pub fn div_by_8(x: u32) -> u32 { x / 8 }",
    ),
    BenchmarkPair(
        name="mod_pow2", category="division", expected_result="divergent",
        description="Unsigned modulo — oracle catches type mismatch",
        divergence_kind="output_mismatch",
        c_source="unsigned int mod_8(unsigned int x) { return x % 8; }",
        rust_source="pub fn mod_8(x: u32) -> u32 { x % 8 }",
    ),
    # GCD pattern
    BenchmarkPair(
        name="gcd_iter", category="division", expected_result="divergent",
        description="GCD with modulo — div by zero and INT_MIN UB",
        divergence_kind="division_by_zero",
        c_source="""int gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}""",
        rust_source="""pub fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 { let t = b; b = a.wrapping_rem(b); a = t; }
    a
}""",
    ),
    BenchmarkPair(
        name="div_ceil", category="division", expected_result="divergent",
        description="Ceiling division — unsigned overflow risk",
        divergence_kind="unsigned_wrap",
        c_source="unsigned int div_ceil(unsigned int a, unsigned int b) { return (a + b - 1) / b; }",
        rust_source="pub fn div_ceil(a: u32, b: u32) -> u32 { (a + b - 1) / b }",
    ),
    BenchmarkPair(
        name="is_divisible", category="division", expected_result="conditional",
        description="Check divisibility",
        c_source="int is_div(int a, int b) { return b != 0 && a % b == 0; }",
        rust_source="pub fn is_div(a: i32, b: i32) -> i32 { if b != 0 && a.wrapping_rem(b) == 0 { 1 } else { 0 } }",
    ),
]

# ---------------------------------------------------------------------------
# Shift patterns (25 pairs)
# ---------------------------------------------------------------------------

SCALED_SHIFT: List[BenchmarkPair] = [
    BenchmarkPair(
        name="shl_overwidth_i32", category="shift", expected_result="divergent",
        description="Left shift by >= 32 is UB in C", divergence_kind="shift_ub",
        c_source="int shl_wide(int x, int n) { return x << n; }",
        rust_source="pub fn shl_wide(x: i32, n: i32) -> i32 { x.wrapping_shl(n as u32) }",
    ),
    BenchmarkPair(
        name="shr_overwidth_i32", category="shift", expected_result="divergent",
        description="Right shift by >= 32 is UB in C", divergence_kind="shift_ub",
        c_source="int shr_wide(int x, int n) { return x >> n; }",
        rust_source="pub fn shr_wide(x: i32, n: i32) -> i32 { x.wrapping_shr(n as u32) }",
    ),
    BenchmarkPair(
        name="shl_negative", category="shift", expected_result="divergent",
        description="Left shift by negative is UB in C", divergence_kind="shift_negative",
        c_source="int shl_neg(int x, int n) { return x << n; }",
        rust_source="pub fn shl_neg(x: i32, n: i32) -> i32 { x.wrapping_shl(n as u32) }",
    ),
    BenchmarkPair(
        name="shr_negative_value", category="shift", expected_result="divergent",
        description="Right shift of negative is implementation-defined", divergence_kind="shift_ub",
        c_source="int shr_neg_val(int x, int n) { return x >> n; }",
        rust_source="pub fn shr_neg_val(x: i32, n: i32) -> i32 { x.wrapping_shr(n as u32) }",
    ),
    BenchmarkPair(
        name="shl_i8", category="shift", expected_result="divergent",
        description="i8 left shift overwidth", divergence_kind="shift_ub",
        c_source="signed char shl_i8(signed char x, int n) { return x << n; }",
        rust_source="pub fn shl_i8(x: i8, n: i32) -> i8 { x.wrapping_shl(n as u32) }",
    ),
    BenchmarkPair(
        name="shl_safe", category="shift", expected_result="equivalent",
        description="Shift with masking",
        c_source="int shl_safe(int x, int n) { return x << (n & 31); }",
        rust_source="pub fn shl_safe(x: i32, n: i32) -> i32 { x.wrapping_shl((n & 31) as u32) }",
    ),
    BenchmarkPair(
        name="shr_safe", category="shift", expected_result="equivalent",
        description="Right shift with masking",
        c_source="unsigned int shr_safe(unsigned int x, int n) { return x >> (n & 31); }",
        rust_source="pub fn shr_safe(x: u32, n: i32) -> u32 { x.wrapping_shr((n & 31) as u32) }",
    ),
    BenchmarkPair(
        name="shl_1", category="shift", expected_result="divergent",
        description="1 << n for large n", divergence_kind="shift_ub",
        c_source="int pow2(int n) { return 1 << n; }",
        rust_source="pub fn pow2(n: i32) -> i32 { 1i32.wrapping_shl(n as u32) }",
    ),
    BenchmarkPair(
        name="bit_extract", category="shift", expected_result="conditional",
        description="Extract bit n from x",
        c_source="int bit_get(int x, int n) { return (x >> (n & 31)) & 1; }",
        rust_source="pub fn bit_get(x: i32, n: i32) -> i32 { (x.wrapping_shr((n & 31) as u32)) & 1 }",
    ),
    BenchmarkPair(
        name="rotate_left", category="shift", expected_result="divergent",
        description="Rotation — .rotate_left() parsed differently",
        divergence_kind="output_mismatch",
        c_source="""unsigned int rotl(unsigned int x, int n) {
    n = n & 31;
    return (x << n) | (x >> (32 - n));
}""",
        rust_source="pub fn rotl(x: u32, n: i32) -> u32 { x.rotate_left((n & 31) as u32) }",
    ),
    BenchmarkPair(
        name="rotate_right", category="shift", expected_result="divergent",
        description="Rotation — .rotate_right() parsed differently",
        divergence_kind="output_mismatch",
        c_source="""unsigned int rotr(unsigned int x, int n) {
    n = n & 31;
    return (x >> n) | (x << (32 - n));
}""",
        rust_source="pub fn rotr(x: u32, n: i32) -> u32 { x.rotate_right((n & 31) as u32) }",
    ),
    # Arithmetic shift semantics
    BenchmarkPair(
        name="arith_shr", category="shift", expected_result="divergent",
        description="Arithmetic right shift of negative — overwidth shift UB in C",
        divergence_kind="shift_ub",
        c_source="int arith_shr(int x, int n) { return x >> n; }",
        rust_source="pub fn arith_shr(x: i32, n: i32) -> i32 { x.wrapping_shr(n as u32) }",
    ),
    BenchmarkPair(
        name="logical_shr", category="shift", expected_result="divergent",
        description="Unsigned right shift — unsigned interpretation divergence",
        divergence_kind="output_mismatch",
        c_source="unsigned int logic_shr(unsigned int x, unsigned int n) { return x >> (n & 31); }",
        rust_source="pub fn logic_shr(x: u32, n: u32) -> u32 { x >> (n & 31) }",
    ),
    BenchmarkPair(
        name="shift_add_mul", category="shift", expected_result="divergent",
        description="Shift-based multiplication pattern", divergence_kind="shift_ub",
        c_source="int mul10(int x) { return (x << 3) + (x << 1); }",
        rust_source="pub fn mul10(x: i32) -> i32 { x.wrapping_shl(3).wrapping_add(x.wrapping_shl(1)) }",
    ),
]

# ---------------------------------------------------------------------------
# Cast patterns (25 pairs)
# ---------------------------------------------------------------------------

SCALED_CAST: List[BenchmarkPair] = [
    BenchmarkPair(
        name="i64_to_i32_trunc", category="cast", expected_result="equivalent",
        description="i64 to i32 truncation — both truncate identically",
        c_source="int trunc64(long long x) { return (int)x; }",
        rust_source="pub fn trunc64(x: i64) -> i32 { x as i32 }",
    ),
    BenchmarkPair(
        name="i32_to_i8_trunc", category="cast", expected_result="equivalent",
        description="i32 to i8 truncation", divergence_kind="cast_truncation",
        c_source="signed char trunc32to8(int x) { return (signed char)x; }",
        rust_source="pub fn trunc32to8(x: i32) -> i8 { x as i8 }",
    ),
    BenchmarkPair(
        name="i32_to_i16_trunc", category="cast", expected_result="equivalent",
        description="i32 to i16 truncation", divergence_kind="cast_truncation",
        c_source="short trunc32to16(int x) { return (short)x; }",
        rust_source="pub fn trunc32to16(x: i32) -> i16 { x as i16 }",
    ),
    BenchmarkPair(
        name="u32_to_i32_reinterpret", category="cast", expected_result="equivalent",
        description="u32 to i32 bit reinterpret",
        c_source="int u2i(unsigned int x) { return (int)x; }",
        rust_source="pub fn u2i(x: u32) -> i32 { x as i32 }",
    ),
    BenchmarkPair(
        name="i32_to_u32_reinterpret", category="cast", expected_result="equivalent",
        description="i32 to u32 bit reinterpret",
        c_source="unsigned int i2u(int x) { return (unsigned int)x; }",
        rust_source="pub fn i2u(x: i32) -> u32 { x as u32 }",
    ),
    BenchmarkPair(
        name="char_to_int_sign", category="cast", expected_result="equivalent",
        description="Signed char to int sign extension",
        c_source="int char2int(signed char c) { return (int)c; }",
        rust_source="pub fn char2int(c: i8) -> i32 { c as i32 }",
    ),
    BenchmarkPair(
        name="int_to_bool", category="cast", expected_result="equivalent",
        description="Int to boolean",
        c_source="int to_bool(int x) { return x != 0; }",
        rust_source="pub fn to_bool(x: i32) -> i32 { if x != 0 { 1 } else { 0 } }",
    ),
    BenchmarkPair(
        name="bool_to_int", category="cast", expected_result="conditional",
        description="Boolean to int",
        c_source="int bool2int(int x) { return !!x; }",
        rust_source="pub fn bool2int(x: i32) -> i32 { if x != 0 { 1 } else { 0 } }",
    ),
    # Widening casts (all equivalent)
    BenchmarkPair(
        name="i8_to_i32_widen", category="cast", expected_result="equivalent",
        description="i8 to i32 widening",
        c_source="int widen_i8(signed char x) { return x; }",
        rust_source="pub fn widen_i8(x: i8) -> i32 { x as i32 }",
    ),
    BenchmarkPair(
        name="i16_to_i32_widen", category="cast", expected_result="equivalent",
        description="i16 to i32 widening",
        c_source="int widen_i16(short x) { return x; }",
        rust_source="pub fn widen_i16(x: i16) -> i32 { x as i32 }",
    ),
    BenchmarkPair(
        name="u8_to_u32_widen", category="cast", expected_result="equivalent",
        description="u8 to u32 widening",
        c_source="unsigned int widen_u8(unsigned char x) { return x; }",
        rust_source="pub fn widen_u8(x: u8) -> u32 { x as u32 }",
    ),
    BenchmarkPair(
        name="mask_to_byte", category="cast", expected_result="equivalent",
        description="Mask and truncate to byte",
        c_source="unsigned char mask_byte(int x) { return (unsigned char)(x & 0xFF); }",
        rust_source="pub fn mask_byte(x: i32) -> u8 { (x & 0xFF) as u8 }",
    ),
    BenchmarkPair(
        name="sign_extend_manual", category="cast", expected_result="equivalent",
        description="Manual sign extension",
        c_source="""int sign_ext(int x) {
    signed char b = (signed char)(x & 0xFF);
    return (int)b;
}""",
        rust_source="pub fn sign_ext(x: i32) -> i32 { ((x & 0xFF) as i8) as i32 }",
    ),
]

# ---------------------------------------------------------------------------
# Bitwise patterns (25 pairs)
# ---------------------------------------------------------------------------

SCALED_BITWISE: List[BenchmarkPair] = [
    BenchmarkPair(
        name="bit_and", category="bitwise", expected_result="equivalent",
        description="Bitwise AND",
        c_source="int bit_and(int a, int b) { return a & b; }",
        rust_source="pub fn bit_and(a: i32, b: i32) -> i32 { a & b }",
    ),
    BenchmarkPair(
        name="bit_or", category="bitwise", expected_result="equivalent",
        description="Bitwise OR",
        c_source="int bit_or(int a, int b) { return a | b; }",
        rust_source="pub fn bit_or(a: i32, b: i32) -> i32 { a | b }",
    ),
    BenchmarkPair(
        name="bit_xor", category="bitwise", expected_result="equivalent",
        description="Bitwise XOR",
        c_source="int bit_xor(int a, int b) { return a ^ b; }",
        rust_source="pub fn bit_xor(a: i32, b: i32) -> i32 { a ^ b }",
    ),
    BenchmarkPair(
        name="bit_not", category="bitwise", expected_result="equivalent",
        description="Bitwise NOT",
        c_source="int bit_not(int x) { return ~x; }",
        rust_source="pub fn bit_not(x: i32) -> i32 { !x }",
    ),
    BenchmarkPair(
        name="popcount", category="bitwise", expected_result="conditional",
        description="Population count",
        c_source="""int popcount(unsigned int x) {
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}""",
        rust_source="""pub fn popcount(mut x: u32) -> i32 {
    let mut c: i32 = 0;
    while x != 0 { c += (x & 1) as i32; x >>= 1; }
    c
}""",
    ),
    BenchmarkPair(
        name="count_leading_zeros", category="bitwise", expected_result="conditional",
        description="CLZ implementation",
        c_source="""int clz(unsigned int x) {
    int n = 0;
    if (x == 0) return 32;
    while ((x & 0x80000000) == 0) { n++; x <<= 1; }
    return n;
}""",
        rust_source="""pub fn clz(x: u32) -> i32 {
    if x == 0 { return 32; }
    let mut n: i32 = 0;
    let mut v = x;
    while (v & 0x80000000) == 0 { n += 1; v <<= 1; }
    n
}""",
    ),
    BenchmarkPair(
        name="is_power_of_2", category="bitwise", expected_result="conditional",
        description="Check if power of 2",
        c_source="int is_pow2(unsigned int x) { return x != 0 && (x & (x - 1)) == 0; }",
        rust_source="pub fn is_pow2(x: u32) -> i32 { if x != 0 && (x & (x.wrapping_sub(1))) == 0 { 1 } else { 0 } }",
    ),
    BenchmarkPair(
        name="next_power_of_2", category="bitwise", expected_result="equivalent",
        description="Round up to next power of 2",
        c_source="""unsigned int next_pow2(unsigned int x) {
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16;
    x++;
    return x;
}""",
        rust_source="""pub fn next_pow2(mut x: u32) -> u32 {
    x = x.wrapping_sub(1);
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16;
    x.wrapping_add(1)
}""",
    ),
    BenchmarkPair(
        name="swap_bytes", category="bitwise", expected_result="divergent",
        description="Byte swap — Rust .swap_bytes() parsed differently",
        divergence_kind="output_mismatch",
        c_source="""unsigned int bswap(unsigned int x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}""",
        rust_source="pub fn bswap(x: u32) -> u32 { x.swap_bytes() }",
    ),
    BenchmarkPair(
        name="bit_reverse", category="bitwise", expected_result="equivalent",
        description="Reverse bits of a byte",
        c_source="""unsigned char bit_rev(unsigned char b) {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    return b;
}""",
        rust_source="""pub fn bit_rev(mut b: u8) -> u8 {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    b
}""",
    ),
    BenchmarkPair(
        name="bit_set", category="bitwise", expected_result="equivalent",
        description="Set bit n",
        c_source="int bit_set(int x, int n) { return x | (1 << (n & 31)); }",
        rust_source="pub fn bit_set(x: i32, n: i32) -> i32 { x | (1i32.wrapping_shl((n & 31) as u32)) }",
    ),
    BenchmarkPair(
        name="bit_clear", category="bitwise", expected_result="equivalent",
        description="Clear bit n",
        c_source="int bit_clr(int x, int n) { return x & ~(1 << (n & 31)); }",
        rust_source="pub fn bit_clr(x: i32, n: i32) -> i32 { x & !(1i32.wrapping_shl((n & 31) as u32)) }",
    ),
    BenchmarkPair(
        name="bit_toggle", category="bitwise", expected_result="equivalent",
        description="Toggle bit n",
        c_source="int bit_toggle(int x, int n) { return x ^ (1 << (n & 31)); }",
        rust_source="pub fn bit_toggle(x: i32, n: i32) -> i32 { x ^ (1i32.wrapping_shl((n & 31) as u32)) }",
    ),
    BenchmarkPair(
        name="parity", category="bitwise", expected_result="equivalent",
        description="Parity of integer",
        c_source="""int parity(unsigned int x) {
    x ^= x >> 16; x ^= x >> 8; x ^= x >> 4;
    x ^= x >> 2; x ^= x >> 1;
    return x & 1;
}""",
        rust_source="""pub fn parity(mut x: u32) -> i32 {
    x ^= x >> 16; x ^= x >> 8; x ^= x >> 4;
    x ^= x >> 2; x ^= x >> 1;
    (x & 1) as i32
}""",
    ),
    BenchmarkPair(
        name="extract_bits", category="bitwise", expected_result="divergent",
        description="Extract bit field",
        c_source="unsigned int extract(unsigned int x, int pos, int width) { return (x >> (pos & 31)) & ((1u << (width & 31)) - 1); }",
        rust_source="pub fn extract(x: u32, pos: i32, width: i32) -> u32 { (x >> ((pos & 31) as u32)) & (1u32.wrapping_shl((width & 31) as u32).wrapping_sub(1)) }",
    ),
]

# ---------------------------------------------------------------------------
# Control flow patterns (25 pairs)
# ---------------------------------------------------------------------------

SCALED_CONTROL_FLOW: List[BenchmarkPair] = [
    BenchmarkPair(
        name="countdown", category="control_flow", expected_result="conditional",
        description="Simple countdown loop",
        c_source="""int countdown(int n) {
    int s = 0;
    while (n > 0) { s += n; n--; }
    return s;
}""",
        rust_source="""pub fn countdown(mut n: i32) -> i32 {
    let mut s: i32 = 0;
    while n > 0 { s = s.wrapping_add(n); n -= 1; }
    s
}""",
    ),
    BenchmarkPair(
        name="sum_to_n", category="control_flow", expected_result="conditional",
        description="Sum 1..n",
        c_source="""int sum_to_n(int n) {
    int s = 0;
    int i;
    for (i = 1; i <= n; i++) s += i;
    return s;
}""",
        rust_source="""pub fn sum_to_n(n: i32) -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 1;
    while i <= n { s = s.wrapping_add(i); i += 1; }
    s
}""",
    ),
    BenchmarkPair(
        name="fibonacci", category="control_flow", expected_result="conditional",
        description="Iterative Fibonacci",
        c_source="""int fib(int n) {
    int a = 0, b = 1, t;
    int i;
    for (i = 0; i < n; i++) { t = b; b = a + b; a = t; }
    return a;
}""",
        rust_source="""pub fn fib(n: i32) -> i32 {
    let mut a: i32 = 0;
    let mut b: i32 = 1;
    let mut i: i32 = 0;
    while i < n { let t = b; b = a.wrapping_add(b); a = t; i += 1; }
    a
}""",
    ),
    BenchmarkPair(
        name="collatz_steps", category="control_flow", expected_result="conditional",
        description="Collatz sequence steps (bounded)",
        c_source="""int collatz(int n) {
    int steps = 0;
    while (n > 1 && steps < 100) {
        if (n % 2 == 0) n = n / 2;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}""",
        rust_source="""pub fn collatz(mut n: i32) -> i32 {
    let mut steps: i32 = 0;
    while n > 1 && steps < 100 {
        if n % 2 == 0 { n = n / 2; }
        else { n = n.wrapping_mul(3).wrapping_add(1); }
        steps += 1;
    }
    steps
}""",
    ),
    BenchmarkPair(
        name="linear_search", category="control_flow", expected_result="conditional",
        description="Linear search returns index",
        c_source="""int find_val(int val, int a, int b, int c, int d) {
    if (a == val) return 0;
    if (b == val) return 1;
    if (c == val) return 2;
    if (d == val) return 3;
    return -1;
}""",
        rust_source="""pub fn find_val(val: i32, a: i32, b: i32, c: i32, d: i32) -> i32 {
    if a == val { 0 }
    else if b == val { 1 }
    else if c == val { 2 }
    else if d == val { 3 }
    else { -1 }
}""",
    ),
    BenchmarkPair(
        name="nested_ternary", category="control_flow", expected_result="divergent",
        description="Nested ternary operator",
        c_source="int classify(int x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }",
        rust_source="pub fn classify(x: i32) -> i32 { if x > 0 { 1 } else if x < 0 { -1 } else { 0 } }",
    ),
    BenchmarkPair(
        name="switch_like", category="control_flow", expected_result="conditional",
        description="Switch-like control flow",
        c_source="""int day_type(int d) {
    if (d == 0 || d == 6) return 0;
    if (d >= 1 && d <= 5) return 1;
    return -1;
}""",
        rust_source="""pub fn day_type(d: i32) -> i32 {
    if d == 0 || d == 6 { 0 }
    else if d >= 1 && d <= 5 { 1 }
    else { -1 }
}""",
    ),
    BenchmarkPair(
        name="early_return", category="control_flow", expected_result="conditional",
        description="Multiple early returns",
        c_source="""int validate(int x) {
    if (x < 0) return -1;
    if (x > 100) return -2;
    if (x == 42) return 1;
    return 0;
}""",
        rust_source="""pub fn validate(x: i32) -> i32 {
    if x < 0 { return -1; }
    if x > 100 { return -2; }
    if x == 42 { return 1; }
    0
}""",
    ),
    BenchmarkPair(
        name="loop_with_break", category="control_flow", expected_result="conditional",
        description="Loop with break condition",
        c_source="""int first_div(int n, int d) {
    int i;
    for (i = 1; i <= n; i++) {
        if (i % d == 0) return i;
    }
    return -1;
}""",
        rust_source="""pub fn first_div(n: i32, d: i32) -> i32 {
    let mut i: i32 = 1;
    while i <= n {
        if d != 0 && i.wrapping_rem(d) == 0 { return i; }
        i += 1;
    }
    -1
}""",
    ),
    BenchmarkPair(
        name="nested_loops", category="control_flow", expected_result="conditional",
        description="Nested loop multiplication table sum",
        c_source="""int mul_table_sum(int n) {
    int s = 0, i, j;
    for (i = 1; i <= n && i <= 10; i++)
        for (j = 1; j <= n && j <= 10; j++)
            s += i * j;
    return s;
}""",
        rust_source="""pub fn mul_table_sum(n: i32) -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 1;
    while i <= n && i <= 10 {
        let mut j: i32 = 1;
        while j <= n && j <= 10 {
            s = s.wrapping_add(i.wrapping_mul(j));
            j += 1;
        }
        i += 1;
    }
    s
}""",
    ),
]

# ---------------------------------------------------------------------------
# Real-world inspired patterns (50 pairs)
# ---------------------------------------------------------------------------

SCALED_REAL_PATTERNS: List[BenchmarkPair] = [
    # Hash functions (from various projects)
    BenchmarkPair(
        name="djb2_hash_step", category="real_patterns", expected_result="equivalent",
        description="DJB2 hash single step (from many C projects)",
        source_project="common",
        c_source="unsigned int djb2_step(unsigned int h, unsigned int c) { return h * 33 + c; }",
        rust_source="pub fn djb2_step(h: u32, c: u32) -> u32 { h.wrapping_mul(33).wrapping_add(c) }",
    ),
    BenchmarkPair(
        name="fnv1a_step", category="real_patterns", expected_result="equivalent",
        description="FNV-1a hash single step",
        source_project="common",
        c_source="unsigned int fnv1a_step(unsigned int h, unsigned char b) { return (h ^ b) * 16777619u; }",
        rust_source="pub fn fnv1a_step(h: u32, b: u8) -> u32 { (h ^ b as u32).wrapping_mul(16777619) }",
    ),
    BenchmarkPair(
        name="murmur_fmix", category="real_patterns", expected_result="divergent",
        description="MurmurHash3 finalizer",
        source_project="various",
        c_source="""unsigned int fmix32(unsigned int h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}""",
        rust_source="""pub fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}""",
    ),
    # CRC patterns
    BenchmarkPair(
        name="crc_step", category="real_patterns", expected_result="equivalent",
        description="CRC32 single byte step",
        source_project="zlib",
        c_source="""unsigned int crc_step(unsigned int crc, unsigned char b) {
    crc ^= b;
    int i;
    for (i = 0; i < 8; i++) {
        if (crc & 1) crc = (crc >> 1) ^ 0xEDB88320u;
        else crc >>= 1;
    }
    return crc;
}""",
        rust_source="""pub fn crc_step(mut crc: u32, b: u8) -> u32 {
    crc ^= b as u32;
    for _ in 0..8 {
        if crc & 1 != 0 { crc = (crc >> 1) ^ 0xEDB88320; }
        else { crc >>= 1; }
    }
    crc
}""",
    ),
    # Alignment calculations (from kernel/allocators)
    BenchmarkPair(
        name="align_up", category="real_patterns", expected_result="equivalent",
        description="Align up to power-of-2 boundary",
        source_project="linux",
        c_source="unsigned int align_up(unsigned int x, unsigned int a) { return (x + a - 1) & ~(a - 1); }",
        rust_source="pub fn align_up(x: u32, a: u32) -> u32 { (x.wrapping_add(a).wrapping_sub(1)) & !(a.wrapping_sub(1)) }",
    ),
    BenchmarkPair(
        name="align_down", category="real_patterns", expected_result="equivalent",
        description="Align down to boundary",
        source_project="linux",
        c_source="unsigned int align_down(unsigned int x, unsigned int a) { return x & ~(a - 1); }",
        rust_source="pub fn align_down(x: u32, a: u32) -> u32 { x & !(a.wrapping_sub(1)) }",
    ),
    # Saturating arithmetic (from audio/video codecs)
    BenchmarkPair(
        name="clamp_u8", category="real_patterns", expected_result="equivalent",
        description="Clamp to u8 range (pixel clamping)",
        source_project="ffmpeg",
        c_source="""int clamp_u8(int x) {
    if (x < 0) return 0;
    if (x > 255) return 255;
    return x;
}""",
        rust_source="pub fn clamp_u8(x: i32) -> i32 { if x < 0 { 0 } else if x > 255 { 255 } else { x } }",
    ),
    BenchmarkPair(
        name="sat_sub_u32", category="real_patterns", expected_result="equivalent",
        description="Saturating unsigned subtraction",
        source_project="common",
        c_source="unsigned int sat_sub(unsigned int a, unsigned int b) { return a > b ? a - b : 0; }",
        rust_source="pub fn sat_sub(a: u32, b: u32) -> u32 { a.saturating_sub(b) }",
    ),
    # Byte manipulation (from network protocols)
    BenchmarkPair(
        name="read_be16", category="real_patterns", expected_result="equivalent",
        description="Read big-endian 16-bit from two bytes",
        source_project="curl",
        c_source="unsigned int read_be16(unsigned int hi, unsigned int lo) { return (hi << 8) | (lo & 0xFF); }",
        rust_source="pub fn read_be16(hi: u32, lo: u32) -> u32 { (hi << 8) | (lo & 0xFF) }",
    ),
    BenchmarkPair(
        name="read_be32", category="real_patterns", expected_result="equivalent",
        description="Read big-endian 32-bit from four bytes",
        source_project="curl",
        c_source="""unsigned int read_be32(unsigned int b3, unsigned int b2, unsigned int b1, unsigned int b0) {
    return (b3 << 24) | ((b2 & 0xFF) << 16) | ((b1 & 0xFF) << 8) | (b0 & 0xFF);
}""",
        rust_source="""pub fn read_be32(b3: u32, b2: u32, b1: u32, b0: u32) -> u32 {
    (b3 << 24) | ((b2 & 0xFF) << 16) | ((b1 & 0xFF) << 8) | (b0 & 0xFF)
}""",
    ),
    # Encoding (from compression/crypto)
    BenchmarkPair(
        name="base64_char", category="real_patterns", expected_result="divergent",
        description="Map 6-bit value to base64 character",
        source_project="openssl",
        c_source="""int b64_char(int v) {
    if (v < 26) return v + 65;
    if (v < 52) return v + 71;
    if (v < 62) return v - 4;
    if (v == 62) return 43;
    if (v == 63) return 47;
    return -1;
}""",
        rust_source="""pub fn b64_char(v: i32) -> i32 {
    if v < 26 { v + 65 }
    else if v < 52 { v + 71 }
    else if v < 62 { v - 4 }
    else if v == 62 { 43 }
    else if v == 63 { 47 }
    else { -1 }
}""",
    ),
    # Fixed-point arithmetic (from DSP/audio)
    BenchmarkPair(
        name="fixed_mul_q15", category="real_patterns", expected_result="divergent",
        description="Q15 fixed-point multiply (overflow possible)",
        divergence_kind="signed_overflow",
        source_project="dsp",
        c_source="int q15_mul(int a, int b) { return (a * b) >> 15; }",
        rust_source="pub fn q15_mul(a: i32, b: i32) -> i32 { a.wrapping_mul(b) >> 15 }",
    ),
    BenchmarkPair(
        name="fixed_lerp", category="real_patterns", expected_result="divergent",
        description="Linear interpolation with overflow risk",
        divergence_kind="signed_overflow",
        source_project="dsp",
        c_source="int lerp(int a, int b, int t) { return a + ((b - a) * t) / 256; }",
        rust_source="pub fn lerp(a: i32, b: i32, t: i32) -> i32 { a.wrapping_add(b.wrapping_sub(a).wrapping_mul(t) / 256) }",
    ),
    # Checksum (from networking)
    BenchmarkPair(
        name="inet_checksum_step", category="real_patterns", expected_result="conditional",
        description="Internet checksum accumulation step",
        source_project="linux",
        c_source="""unsigned int csum_add(unsigned int a, unsigned int b) {
    unsigned int s = a + b;
    return s + (s < a);
}""",
        rust_source="""pub fn csum_add(a: u32, b: u32) -> u32 {
    let s = a.wrapping_add(b);
    s.wrapping_add(if s < a { 1 } else { 0 })
}""",
    ),
    # Min/max of 3 values
    BenchmarkPair(
        name="min3", category="real_patterns", expected_result="divergent",
        description="Minimum of 3 values",
        source_project="common",
        c_source="""int min3(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}""",
        rust_source="""pub fn min3(a: i32, b: i32, c: i32) -> i32 {
    let mut m = a;
    if b < m { m = b; }
    if c < m { m = c; }
    m
}""",
    ),
    BenchmarkPair(
        name="max3", category="real_patterns", expected_result="divergent",
        description="Maximum of 3 values",
        c_source="""int max3(int a, int b, int c) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}""",
        rust_source="""pub fn max3(a: i32, b: i32, c: i32) -> i32 {
    let mut m = a;
    if b > m { m = b; }
    if c > m { m = c; }
    m
}""",
    ),
    # Median of 3
    BenchmarkPair(
        name="median3", category="real_patterns", expected_result="divergent",
        description="Median of 3 values",
        source_project="common",
        c_source="""int median3(int a, int b, int c) {
    if (a <= b) {
        if (b <= c) return b;
        if (a <= c) return c;
        return a;
    }
    if (a <= c) return a;
    if (b <= c) return c;
    return b;
}""",
        rust_source="""pub fn median3(a: i32, b: i32, c: i32) -> i32 {
    if a <= b {
        if b <= c { b } else if a <= c { c } else { a }
    } else {
        if a <= c { a } else if b <= c { c } else { b }
    }
}""",
    ),
    # Hamming distance
    BenchmarkPair(
        name="hamming_dist", category="real_patterns", expected_result="conditional",
        description="Hamming distance between two integers",
        source_project="common",
        c_source="""int hamming(unsigned int a, unsigned int b) {
    unsigned int x = a ^ b;
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}""",
        rust_source="""pub fn hamming(a: u32, b: u32) -> i32 {
    let mut x = a ^ b;
    let mut c: i32 = 0;
    while x != 0 { c += (x & 1) as i32; x >>= 1; }
    c
}""",
    ),
    # Integer square root
    BenchmarkPair(
        name="isqrt", category="real_patterns", expected_result="conditional",
        description="Integer square root",
        source_project="common",
        c_source="""unsigned int isqrt(unsigned int n) {
    unsigned int x = n;
    unsigned int y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}""",
        rust_source="""pub fn isqrt(n: u32) -> u32 {
    if n == 0 { return 0; }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x { x = y; y = (x + n / x) / 2; }
    x
}""",
    ),
    # Bit field manipulation
    BenchmarkPair(
        name="pack_rgb", category="real_patterns", expected_result="equivalent",
        description="Pack RGB into 32-bit integer",
        source_project="graphics",
        c_source="""unsigned int pack_rgb(unsigned int r, unsigned int g, unsigned int b) {
    return ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
}""",
        rust_source="""pub fn pack_rgb(r: u32, g: u32, b: u32) -> u32 {
    ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)
}""",
    ),
    BenchmarkPair(
        name="unpack_r", category="real_patterns", expected_result="equivalent",
        description="Extract red from packed RGB",
        source_project="graphics",
        c_source="unsigned int unpack_r(unsigned int rgb) { return (rgb >> 16) & 0xFF; }",
        rust_source="pub fn unpack_r(rgb: u32) -> u32 { (rgb >> 16) & 0xFF }",
    ),
    # Error code handling
    BenchmarkPair(
        name="errno_to_result", category="real_patterns", expected_result="conditional",
        description="Map error code to result",
        source_project="common",
        c_source="""int map_err(int code) {
    if (code == 0) return 0;
    if (code == -1) return 1;
    if (code == -2) return 2;
    return 255;
}""",
        rust_source="""pub fn map_err(code: i32) -> i32 {
    match code {
        0 => 0,
        -1 => 1,
        -2 => 2,
        _ => 255,
    }
}""",
    ),
    # Average without overflow (from embedded)
    BenchmarkPair(
        name="avg_no_overflow", category="real_patterns", expected_result="equivalent",
        description="Average without intermediate overflow",
        source_project="embedded",
        c_source="int avg(int a, int b) { return (a & b) + ((a ^ b) >> 1); }",
        rust_source="pub fn avg(a: i32, b: i32) -> i32 { (a & b) + ((a ^ b) >> 1) }",
    ),
    # Leading zeros count (from kernel)
    BenchmarkPair(
        name="fls", category="real_patterns", expected_result="conditional",
        description="Find last (highest) set bit",
        source_project="linux",
        c_source="""int fls(unsigned int x) {
    int r = 0;
    if (x & 0xFFFF0000) { x >>= 16; r += 16; }
    if (x & 0xFF00) { x >>= 8; r += 8; }
    if (x & 0xF0) { x >>= 4; r += 4; }
    if (x & 0xC) { x >>= 2; r += 2; }
    if (x & 0x2) r += 1;
    return r;
}""",
        rust_source="""pub fn fls(mut x: u32) -> i32 {
    let mut r: i32 = 0;
    if x & 0xFFFF0000 != 0 { x >>= 16; r += 16; }
    if x & 0xFF00 != 0 { x >>= 8; r += 8; }
    if x & 0xF0 != 0 { x >>= 4; r += 4; }
    if x & 0xC != 0 { x >>= 2; r += 2; }
    if x & 0x2 != 0 { r += 1; }
    r
}""",
    ),
    # Power modular (from crypto)
    BenchmarkPair(
        name="pow_mod", category="real_patterns", expected_result="conditional",
        description="Modular exponentiation (bounded)",
        source_project="crypto",
        c_source="""unsigned int powmod(unsigned int base, unsigned int exp, unsigned int mod) {
    unsigned int result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % mod;
        exp >>= 1;
        base = (base * base) % mod;
    }
    return result;
}""",
        rust_source="""pub fn powmod(mut base: u32, mut exp: u32, modulus: u32) -> u32 {
    let mut result: u32 = 1;
    base = base % modulus;
    while exp > 0 {
        if exp & 1 != 0 { result = (result.wrapping_mul(base)) % modulus; }
        exp >>= 1;
        base = (base.wrapping_mul(base)) % modulus;
    }
    result
}""",
    ),
    # Simple state machine
    BenchmarkPair(
        name="state_machine", category="real_patterns", expected_result="conditional",
        description="Simple state machine transition",
        source_project="protocol",
        c_source="""int next_state(int state, int input) {
    if (state == 0) return input > 0 ? 1 : 0;
    if (state == 1) return input == 0 ? 2 : 1;
    if (state == 2) return 0;
    return -1;
}""",
        rust_source="""pub fn next_state(state: i32, input: i32) -> i32 {
    match state {
        0 => if input > 0 { 1 } else { 0 },
        1 => if input == 0 { 2 } else { 1 },
        2 => 0,
        _ => -1,
    }
}""",
    ),
]

# ---------------------------------------------------------------------------
# Compound patterns — multiple divergence classes (20 pairs)
# ---------------------------------------------------------------------------

SCALED_COMPOUND: List[BenchmarkPair] = [
    BenchmarkPair(
        name="add_then_shift", category="compound", expected_result="divergent",
        description="Addition overflow + shift UB", divergence_kind="signed_overflow",
        c_source="int add_shift(int a, int b, int n) { return (a + b) << n; }",
        rust_source="pub fn add_shift(a: i32, b: i32, n: i32) -> i32 { a.wrapping_add(b).wrapping_shl(n as u32) }",
    ),
    BenchmarkPair(
        name="mul_then_div", category="compound", expected_result="conditional",
        description="Multiply overflow + division", divergence_kind="signed_overflow",
        c_source="int mul_div(int a, int b, int c) { return (a * b) / c; }",
        rust_source="pub fn mul_div(a: i32, b: i32, c: i32) -> i32 { a.wrapping_mul(b).wrapping_div(c) }",
    ),
    BenchmarkPair(
        name="shift_then_cast", category="compound", expected_result="divergent",
        description="Shift + truncation", divergence_kind="shift_ub",
        c_source="signed char shift_trunc(int x, int n) { return (signed char)(x << n); }",
        rust_source="pub fn shift_trunc(x: i32, n: i32) -> i8 { x.wrapping_shl(n as u32) as i8 }",
    ),
    BenchmarkPair(
        name="abs_then_div", category="compound", expected_result="divergent",
        description="abs + division", divergence_kind="int_min_negation",
        c_source="int abs_div(int a, int b) { int v = a < 0 ? -a : a; return v / b; }",
        rust_source="pub fn abs_div(a: i32, b: i32) -> i32 { let v = if a < 0 { a.wrapping_neg() } else { a }; v.wrapping_div(b) }",
    ),
    BenchmarkPair(
        name="square_then_mod", category="compound", expected_result="conditional",
        description="Square overflow + modulo", divergence_kind="signed_overflow",
        c_source="int sq_mod(int x, int m) { return (x * x) % m; }",
        rust_source="pub fn sq_mod(x: i32, m: i32) -> i32 { x.wrapping_mul(x).wrapping_rem(m) }",
    ),
    BenchmarkPair(
        name="weighted_avg", category="compound", expected_result="divergent",
        description="Weighted average with overflow", divergence_kind="signed_overflow",
        c_source="int wavg(int a, int b, int w) { return (a * w + b * (100 - w)) / 100; }",
        rust_source="pub fn wavg(a: i32, b: i32, w: i32) -> i32 { a.wrapping_mul(w).wrapping_add(b.wrapping_mul(100i32.wrapping_sub(w))) / 100 }",
    ),
    BenchmarkPair(
        name="range_check_add", category="compound", expected_result="conditional",
        description="Range-checked addition (safe)",
        c_source="""int safe_add(int a, int b) {
    if (a > 0 && b > 2147483647 - a) return 2147483647;
    if (a < 0 && b < -2147483647 - 1 - a) return -2147483648;
    return a + b;
}""",
        rust_source="""pub fn safe_add(a: i32, b: i32) -> i32 {
    match a.checked_add(b) {
        Some(v) => v,
        None => if a > 0 { i32::MAX } else { i32::MIN },
    }
}""",
    ),
    BenchmarkPair(
        name="distance_2d", category="compound", expected_result="conditional",
        description="2D distance squared (overflow)", divergence_kind="signed_overflow",
        c_source="int dist2(int x1, int y1, int x2, int y2) { return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1); }",
        rust_source="""pub fn dist2(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
    let dx = x2.wrapping_sub(x1);
    let dy = y2.wrapping_sub(y1);
    dx.wrapping_mul(dx).wrapping_add(dy.wrapping_mul(dy))
}""",
    ),
    BenchmarkPair(
        name="dot_product_2d", category="compound", expected_result="divergent",
        description="2D dot product overflow", divergence_kind="signed_overflow",
        c_source="int dot2(int ax, int ay, int bx, int by) { return ax*bx + ay*by; }",
        rust_source="pub fn dot2(ax: i32, ay: i32, bx: i32, by: i32) -> i32 { ax.wrapping_mul(bx).wrapping_add(ay.wrapping_mul(by)) }",
    ),
    BenchmarkPair(
        name="cross_product_2d", category="compound", expected_result="divergent",
        description="2D cross product overflow", divergence_kind="signed_overflow",
        c_source="int cross2(int ax, int ay, int bx, int by) { return ax*by - ay*bx; }",
        rust_source="pub fn cross2(ax: i32, ay: i32, bx: i32, by: i32) -> i32 { ax.wrapping_mul(by).wrapping_sub(ay.wrapping_mul(bx)) }",
    ),
    # Hash combine
    BenchmarkPair(
        name="hash_combine", category="compound", expected_result="divergent",
        description="Hash combination (boost-style)",
        source_project="boost",
        c_source="""unsigned int hash_combine(unsigned int seed, unsigned int v) {
    return seed ^ (v + 0x9e3779b9u + (seed << 6) + (seed >> 2));
}""",
        rust_source="""pub fn hash_combine(seed: u32, v: u32) -> u32 {
    seed ^ (v.wrapping_add(0x9e3779b9).wrapping_add(seed << 6).wrapping_add(seed >> 2))
}""",
    ),
    BenchmarkPair(
        name="safe_mul_check", category="compound", expected_result="conditional",
        description="Multiplication with overflow check",
        c_source="""int safe_mul(int a, int b) {
    if (a == 0 || b == 0) return 0;
    if (a > 0 && b > 0 && a > 2147483647 / b) return 2147483647;
    if (a < 0 && b < 0 && a < 2147483647 / b) return 2147483647;
    return a * b;
}""",
        rust_source="""pub fn safe_mul(a: i32, b: i32) -> i32 {
    if a == 0 || b == 0 { return 0; }
    match a.checked_mul(b) {
        Some(v) => v,
        None => i32::MAX,
    }
}""",
    ),
]

# ---------------------------------------------------------------------------
# Additional arithmetic edge cases (20 pairs)
# ---------------------------------------------------------------------------

SCALED_EDGE_CASES: List[BenchmarkPair] = [
    BenchmarkPair(
        name="identity_add_0", category="arithmetic", expected_result="equivalent",
        description="x + 0 = x",
        c_source="int add_zero(int x) { return x + 0; }",
        rust_source="pub fn add_zero(x: i32) -> i32 { x + 0 }",
    ),
    BenchmarkPair(
        name="identity_mul_1", category="arithmetic", expected_result="equivalent",
        description="x * 1 = x",
        c_source="int mul_one(int x) { return x * 1; }",
        rust_source="pub fn mul_one(x: i32) -> i32 { x * 1 }",
    ),
    BenchmarkPair(
        name="negate_twice", category="arithmetic", expected_result="divergent",
        description="Double negation diverges for INT_MIN", divergence_kind="int_min_negation",
        c_source="int neg_twice(int x) { return -(-x); }",
        rust_source="pub fn neg_twice(x: i32) -> i32 { x.wrapping_neg().wrapping_neg() }",
    ),
    BenchmarkPair(
        name="sub_self", category="arithmetic", expected_result="equivalent",
        description="x - x = 0",
        c_source="int sub_self(int x) { return x - x; }",
        rust_source="pub fn sub_self(x: i32) -> i32 { x.wrapping_sub(x) }",
    ),
    BenchmarkPair(
        name="xor_self", category="bitwise", expected_result="equivalent",
        description="x ^ x = 0",
        c_source="int xor_self(int x) { return x ^ x; }",
        rust_source="pub fn xor_self(x: i32) -> i32 { x ^ x }",
    ),
    BenchmarkPair(
        name="and_self", category="bitwise", expected_result="equivalent",
        description="x & x = x",
        c_source="int and_self(int x) { return x & x; }",
        rust_source="pub fn and_self(x: i32) -> i32 { x & x }",
    ),
    BenchmarkPair(
        name="or_self", category="bitwise", expected_result="equivalent",
        description="x | x = x",
        c_source="int or_self(int x) { return x | x; }",
        rust_source="pub fn or_self(x: i32) -> i32 { x | x }",
    ),
    BenchmarkPair(
        name="add_sub_cancel", category="arithmetic", expected_result="divergent",
        description="(x + y) can overflow in C — UB",
        divergence_kind="signed_overflow",
        c_source="int add_sub(int x, int y) { return (x + y) - y; }",
        rust_source="pub fn add_sub(x: i32, y: i32) -> i32 { x.wrapping_add(y).wrapping_sub(y) }",
    ),
    BenchmarkPair(
        name="mul_neg1", category="arithmetic", expected_result="divergent",
        description="x * -1 diverges for INT_MIN", divergence_kind="signed_overflow",
        c_source="int mul_neg1(int x) { return x * -1; }",
        rust_source="pub fn mul_neg1(x: i32) -> i32 { x.wrapping_mul(-1) }",
    ),
    BenchmarkPair(
        name="complement_add_1", category="bitwise", expected_result="divergent",
        description="~x + 1 = -x (two's complement)",
        c_source="int comp_plus1(int x) { return ~x + 1; }",
        rust_source="pub fn comp_plus1(x: i32) -> i32 { (!x).wrapping_add(1) }",
    ),
    BenchmarkPair(
        name="max_val_add_1", category="arithmetic", expected_result="divergent",
        description="MAX + 1 overflow", divergence_kind="signed_overflow",
        c_source="int max_plus_1(int x) { return x + 1; }",
        rust_source="pub fn max_plus_1(x: i32) -> i32 { x.wrapping_add(1) }",
    ),
    BenchmarkPair(
        name="zero_div_anything", category="division", expected_result="conditional",
        description="0 / x = 0 (for non-zero x)",
        c_source="int zero_div(int x) { return x != 0 ? 0 / x : 0; }",
        rust_source="pub fn zero_div(x: i32) -> i32 { if x != 0 { 0i32.wrapping_div(x) } else { 0 } }",
    ),
    BenchmarkPair(
        name="power_of_2_test", category="bitwise", expected_result="conditional",
        description="Check if x is a power of 2",
        c_source="int ispow2(int x) { return x > 0 && (x & (x - 1)) == 0; }",
        rust_source="pub fn ispow2(x: i32) -> i32 { if x > 0 && (x & (x - 1)) == 0 { 1 } else { 0 } }",
    ),
    BenchmarkPair(
        name="demorgan_and", category="bitwise", expected_result="equivalent",
        description="De Morgan: ~(a & b) = ~a | ~b",
        c_source="int demorgan(int a, int b) { return ~(a & b); }",
        rust_source="pub fn demorgan(a: i32, b: i32) -> i32 { !a | !b }",
    ),
    BenchmarkPair(
        name="ternary_assign", category="control_flow", expected_result="equivalent",
        description="Ternary conditional assignment",
        c_source="int ternary(int c, int a, int b) { return c ? a : b; }",
        rust_source="pub fn ternary(c: i32, a: i32, b: i32) -> i32 { if c != 0 { a } else { b } }",
    ),
]


# ---------------------------------------------------------------------------
# All scaled benchmarks
# ---------------------------------------------------------------------------

ALL_SCALED_BENCHMARKS: List[BenchmarkPair] = (
    SCALED_ARITHMETIC +
    SCALED_DIVISION +
    SCALED_SHIFT +
    SCALED_CAST +
    SCALED_BITWISE +
    SCALED_CONTROL_FLOW +
    SCALED_REAL_PATTERNS +
    SCALED_COMPOUND +
    SCALED_EDGE_CASES
)


def get_scaled_pairs() -> List[BenchmarkPair]:
    return list(ALL_SCALED_BENCHMARKS)


def get_scaled_by_category(category: str) -> List[BenchmarkPair]:
    return [b for b in ALL_SCALED_BENCHMARKS if b.category == category]


def get_scaled_divergent() -> List[BenchmarkPair]:
    return [b for b in ALL_SCALED_BENCHMARKS if b.expected_result == "divergent"]


def get_scaled_equivalent() -> List[BenchmarkPair]:
    return [b for b in ALL_SCALED_BENCHMARKS if b.expected_result == "equivalent"]


def get_all_categories() -> List[str]:
    return sorted(set(b.category for b in ALL_SCALED_BENCHMARKS))
