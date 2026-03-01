"""
Comprehensive benchmark pairs for cross-language equivalence verification.

52 function pairs covering:
- Arithmetic with overflow semantics
- Loop termination
- Struct/enum alignment
- String handling
- Error propagation (errno vs Result)
- Memory allocation patterns
- Bitwise operations
- Division edge cases
- Shift semantics
- Float-to-int conversion
- Negation edge cases
- Signed/unsigned mixing
- Control flow divergence
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
    divergence_kind: str = ""  # if divergent, what kind


# ---------------------------------------------------------------------------
# Category 1: Arithmetic overflow (12 pairs)
# ---------------------------------------------------------------------------

ARITHMETIC_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="add_wrapping",
        category="arithmetic",
        expected_result="divergent",
        description="C int addition has UB on overflow, Rust wrapping_add wraps",
        divergence_kind="signed_overflow",
        c_source="""
int add_wrapping(int a, int b) {
    return a + b;
}
""",
        rust_source="""
pub fn add_wrapping(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}
""",
    ),
    BenchmarkPair(
        name="add_checked_divergent",
        category="arithmetic",
        expected_result="divergent",
        description="C wraps on overflow, Rust debug panics",
        divergence_kind="signed_overflow",
        c_source="""
int add_checked(int a, int b) {
    return a + b;
}
""",
        rust_source="""
pub fn add_checked(a: i32, b: i32) -> i32 {
    a + b
}
""",
    ),
    BenchmarkPair(
        name="mul_overflow",
        category="arithmetic",
        expected_result="divergent",
        description="Multiplication overflow: C=UB, Rust=wrap",
        divergence_kind="signed_overflow",
        c_source="""
int mul_overflow(int a, int b) {
    return a * b;
}
""",
        rust_source="""
pub fn mul_overflow(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}
""",
    ),
    BenchmarkPair(
        name="sub_overflow",
        category="arithmetic",
        expected_result="divergent",
        description="Subtraction overflow divergence",
        divergence_kind="signed_overflow",
        c_source="""
int sub_overflow(int a, int b) {
    return a - b;
}
""",
        rust_source="""
pub fn sub_overflow(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}
""",
    ),
    BenchmarkPair(
        name="saturating_add",
        category="arithmetic",
        expected_result="divergent",
        description="C wraps, Rust saturates",
        divergence_kind="saturation",
        c_source="""
int saturating_add(int a, int b) {
    return a + b;
}
""",
        rust_source="""
pub fn saturating_add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}
""",
    ),
    BenchmarkPair(
        name="negate_min",
        category="arithmetic",
        expected_result="divergent",
        description="Negation of INT_MIN: C=UB, Rust=wraps to INT_MIN",
        divergence_kind="negation_overflow",
        c_source="""
int negate(int x) {
    return -x;
}
""",
        rust_source="""
pub fn negate(x: i32) -> i32 {
    x.wrapping_neg()
}
""",
    ),
    BenchmarkPair(
        name="abs_function",
        category="arithmetic",
        expected_result="divergent",
        description="abs(INT_MIN): C=UB, Rust=wrapping gives INT_MIN",
        divergence_kind="negation_overflow",
        c_source="""
int abs_val(int x) {
    if (x < 0) return -x;
    return x;
}
""",
        rust_source="""
pub fn abs_val(x: i32) -> i32 {
    if x < 0 { x.wrapping_neg() } else { x }
}
""",
    ),
    BenchmarkPair(
        name="unsigned_add",
        category="arithmetic",
        expected_result="equivalent",
        description="Unsigned addition: both wrap identically",
        c_source="""
unsigned int unsigned_add(unsigned int a, unsigned int b) {
    return a + b;
}
""",
        rust_source="""
pub fn unsigned_add(a: u32, b: u32) -> u32 {
    a.wrapping_add(b)
}
""",
    ),
    BenchmarkPair(
        name="mixed_sign_compare",
        category="arithmetic",
        expected_result="divergent",
        description="Comparing signed and unsigned: C promotes, Rust explicit",
        divergence_kind="signedness",
        c_source="""
int mixed_compare(int a, unsigned int b) {
    return a < b;
}
""",
        rust_source="""
pub fn mixed_compare(a: i32, b: u32) -> i32 {
    if (a as u32) < b { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="increment_loop",
        category="arithmetic",
        expected_result="divergent",
        description="Sum loop: C signed add is UB on overflow, Rust wrapping_add wraps",
        divergence_kind="overflow",
        c_source="""
int sum_to_n(int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum = sum + i;
    }
    return sum;
}
""",
        rust_source="""
pub fn sum_to_n(n: i32) -> i32 {
    let mut sum: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        sum = sum.wrapping_add(i);
        i = i.wrapping_add(1);
    }
    sum
}
""",
    ),
    BenchmarkPair(
        name="checked_add_result",
        category="arithmetic",
        expected_result="divergent",
        description="Checked add returns Option in Rust vs raw in C",
        divergence_kind="error_handling",
        c_source="""
int checked_add(int a, int b, int* overflow) {
    int result = a + b;
    *overflow = 0;
    if (a > 0 && b > 0 && result < 0) *overflow = 1;
    if (a < 0 && b < 0 && result > 0) *overflow = 1;
    return result;
}
""",
        rust_source="""
pub fn checked_add(a: i32, b: i32) -> i32 {
    match a.checked_add(b) {
        Some(r) => r,
        None => 0,
    }
}
""",
    ),
    BenchmarkPair(
        name="widening_mul",
        category="arithmetic",
        expected_result="equivalent",
        description="Widening multiply to avoid overflow",
        c_source="""
long long widening_mul(int a, int b) {
    return (long long)a * (long long)b;
}
""",
        rust_source="""
pub fn widening_mul(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category 2: Division and shift edge cases (8 pairs)
# ---------------------------------------------------------------------------

DIVISION_SHIFT_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="div_by_zero",
        category="division",
        expected_result="divergent",
        description="Division by zero: C=UB, Rust=panic",
        divergence_kind="division_by_zero",
        c_source="""
int safe_div(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}
""",
        rust_source="""
pub fn safe_div(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    a / b
}
""",
    ),
    BenchmarkPair(
        name="div_int_min",
        category="division",
        expected_result="divergent",
        description="INT_MIN / -1: C=UB, Rust=panic",
        divergence_kind="division_overflow",
        c_source="""
int div_safe(int a, int b) {
    return a / b;
}
""",
        rust_source="""
pub fn div_safe(a: i32, b: i32) -> i32 {
    a.wrapping_div(b)
}
""",
    ),
    BenchmarkPair(
        name="modulo_negative",
        category="division",
        expected_result="divergent",
        description="Modulo with negative operands: C UB on INT_MIN % -1, Rust wraps",
        divergence_kind="overflow",
        c_source="""
int modulo(int a, int b) {
    if (b == 0) return 0;
    return a % b;
}
""",
        rust_source="""
pub fn modulo(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    a.wrapping_rem(b)
}
""",
    ),
    BenchmarkPair(
        name="shift_left_overflow",
        category="shift",
        expected_result="divergent",
        description="Left shift >= bit width: C=UB, Rust=mask",
        divergence_kind="shift_overflow",
        c_source="""
int shift_left(int x, int n) {
    return x << n;
}
""",
        rust_source="""
pub fn shift_left(x: i32, n: i32) -> i32 {
    x.wrapping_shl(n as u32)
}
""",
    ),
    BenchmarkPair(
        name="shift_right_arithmetic",
        category="shift",
        expected_result="divergent",
        description="Right shift of negative: C=impl-defined, Rust=arithmetic",
        divergence_kind="shift_semantics",
        c_source="""
int asr(int x, int n) {
    return x >> n;
}
""",
        rust_source="""
pub fn asr(x: i32, n: i32) -> i32 {
    x.wrapping_shr(n as u32)
}
""",
    ),
    BenchmarkPair(
        name="shift_negative_amount",
        category="shift",
        expected_result="divergent",
        description="Shift by negative amount: C=UB, Rust=masks",
        divergence_kind="shift_overflow",
        c_source="""
int shift_neg(int x, int n) {
    if (n < 0) return 0;
    return x << n;
}
""",
        rust_source="""
pub fn shift_neg(x: i32, n: i32) -> i32 {
    if n < 0 { return 0; }
    x.wrapping_shl(n as u32)
}
""",
    ),
    BenchmarkPair(
        name="unsigned_div",
        category="division",
        expected_result="equivalent",
        description="Unsigned division (well-defined in both)",
        c_source="""
unsigned int udiv(unsigned int a, unsigned int b) {
    if (b == 0) return 0;
    return a / b;
}
""",
        rust_source="""
pub fn udiv(a: u32, b: u32) -> u32 {
    if b == 0 { return 0; }
    a / b
}
""",
    ),
    BenchmarkPair(
        name="power_of_two_div",
        category="division",
        expected_result="equivalent",
        description="Division by power of 2 (both use shift)",
        c_source="""
int div_by_4(int x) {
    return x / 4;
}
""",
        rust_source="""
pub fn div_by_4(x: i32) -> i32 {
    x / 4
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category 3: Loops with different termination (8 pairs)
# ---------------------------------------------------------------------------

LOOP_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="count_bits",
        category="loops",
        expected_result="equivalent",
        description="Population count loop",
        c_source="""
int count_bits(unsigned int x) {
    int count = 0;
    while (x != 0) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}
""",
        rust_source="""
pub fn count_bits(mut x: u32) -> i32 {
    let mut count: i32 = 0;
    while x != 0 {
        count = count.wrapping_add((x & 1) as i32);
        x >>= 1;
    }
    count
}
""",
    ),
    BenchmarkPair(
        name="gcd_loop",
        category="loops",
        expected_result="divergent",
        description="GCD: C signed negate/mod is UB on INT_MIN, Rust wraps",
        divergence_kind="overflow",
        c_source="""
int gcd(int a, int b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}
""",
        rust_source="""
pub fn gcd(mut a: i32, mut b: i32) -> i32 {
    if a < 0 { a = a.wrapping_neg(); }
    if b < 0 { b = b.wrapping_neg(); }
    while b != 0 {
        let t = b;
        b = a.wrapping_rem(b);
        a = t;
    }
    a
}
""",
    ),
    BenchmarkPair(
        name="fibonacci",
        category="loops",
        expected_result="divergent",
        description="Fibonacci: overflows differently in C vs Rust",
        divergence_kind="signed_overflow",
        c_source="""
int fibonacci(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int t = a + b;
        a = b;
        b = t;
    }
    return b;
}
""",
        rust_source="""
pub fn fibonacci(n: i32) -> i32 {
    if n <= 1 { return n; }
    let mut a: i32 = 0;
    let mut b: i32 = 1;
    let mut i: i32 = 2;
    while i <= n {
        let t = a.wrapping_add(b);
        a = b;
        b = t;
        i = i.wrapping_add(1);
    }
    b
}
""",
    ),
    BenchmarkPair(
        name="binary_search",
        category="loops",
        expected_result="divergent",
        description="Binary search: midpoint overflow (a+b)/2",
        divergence_kind="signed_overflow",
        c_source="""
int midpoint(int lo, int hi) {
    return (lo + hi) / 2;
}
""",
        rust_source="""
pub fn midpoint(lo: i32, hi: i32) -> i32 {
    lo.wrapping_add(hi.wrapping_sub(lo) / 2)
}
""",
    ),
    BenchmarkPair(
        name="reverse_bits_loop",
        category="loops",
        expected_result="equivalent",
        description="Bit reversal loop (unsigned, no overflow issues)",
        c_source="""
unsigned int reverse_bits(unsigned int x) {
    unsigned int result = 0;
    for (int i = 0; i < 32; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}
""",
        rust_source="""
pub fn reverse_bits(mut x: u32) -> u32 {
    let mut result: u32 = 0;
    let mut i: i32 = 0;
    while i < 32 {
        result = (result << 1) | (x & 1);
        x >>= 1;
        i += 1;
    }
    result
}
""",
    ),
    BenchmarkPair(
        name="collatz_steps",
        category="loops",
        expected_result="divergent",
        description="Collatz sequence: 3n+1 can overflow",
        divergence_kind="signed_overflow",
        c_source="""
int collatz_steps(int n) {
    int steps = 0;
    while (n != 1 && n > 0) {
        if (n % 2 == 0) n = n / 2;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}
""",
        rust_source="""
pub fn collatz_steps(mut n: i32) -> i32 {
    let mut steps: i32 = 0;
    while n != 1 && n > 0 {
        if n % 2 == 0 { n = n / 2; }
        else { n = n.wrapping_mul(3).wrapping_add(1); }
        steps = steps.wrapping_add(1);
    }
    steps
}
""",
    ),
    BenchmarkPair(
        name="digit_sum",
        category="loops",
        expected_result="divergent",
        description="Digit sum: C signed negate/div is UB on INT_MIN, Rust wraps",
        divergence_kind="overflow",
        c_source="""
int digit_sum(int n) {
    if (n < 0) n = -n;
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}
""",
        rust_source="""
pub fn digit_sum(mut n: i32) -> i32 {
    if n < 0 { n = n.wrapping_neg(); }
    let mut sum: i32 = 0;
    while n > 0 {
        sum = sum.wrapping_add(n.wrapping_rem(10));
        n = n / 10;
    }
    sum
}
""",
    ),
    BenchmarkPair(
        name="count_trailing_zeros",
        category="loops",
        expected_result="equivalent",
        description="Count trailing zeros",
        c_source="""
int ctz(unsigned int x) {
    if (x == 0) return 32;
    int count = 0;
    while ((x & 1) == 0) {
        count++;
        x >>= 1;
    }
    return count;
}
""",
        rust_source="""
pub fn ctz(x: u32) -> i32 {
    if x == 0 { return 32; }
    let mut count: i32 = 0;
    let mut v = x;
    while (v & 1) == 0 {
        count += 1;
        v >>= 1;
    }
    count
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category 4: Error handling (errno vs Result) (8 pairs)
# ---------------------------------------------------------------------------

ERROR_HANDLING_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="safe_divide_errno",
        category="error_handling",
        expected_result="divergent",
        description="C returns sentinel + errno, Rust returns Result-like",
        divergence_kind="error_model",
        c_source="""
int safe_divide(int a, int b) {
    if (b == 0) return -1;
    if (a == -2147483648 && b == -1) return -1;
    return a / b;
}
""",
        rust_source="""
pub fn safe_divide(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    if a == i32::MIN && b == -1 { return i32::MIN; }
    a / b
}
""",
    ),
    BenchmarkPair(
        name="sqrt_int",
        category="error_handling",
        expected_result="divergent",
        description="Isqrt: C signed add/div overflows (UB) for large n, Rust wraps",
        divergence_kind="overflow",
        c_source="""
int isqrt(int n) {
    if (n < 0) return -1;
    if (n == 0) return 0;
    int x = n;
    int y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}
""",
        rust_source="""
pub fn isqrt(n: i32) -> i32 {
    if n < 0 { return -1; }
    if n == 0 { return 0; }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}
""",
    ),
    BenchmarkPair(
        name="array_index_safe",
        category="error_handling",
        expected_result="divergent",
        description="Array bounds: C no check, Rust panics",
        divergence_kind="bounds_check",
        c_source="""
int get_element(int* arr, int len, int idx) {
    if (idx < 0 || idx >= len) return -1;
    return arr[idx];
}
""",
        rust_source="""
pub fn get_element(arr: &[i32], idx: i32) -> i32 {
    if idx < 0 || idx as usize >= arr.len() { return -1; }
    arr[idx as usize]
}
""",
    ),
    BenchmarkPair(
        name="clamp_value",
        category="error_handling",
        expected_result="equivalent",
        description="Clamping (same behavior, no overflow)",
        c_source="""
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
""",
        rust_source="""
pub fn clamp(x: i32, lo: i32, hi: i32) -> i32 {
    if x < lo { lo }
    else if x > hi { hi }
    else { x }
}
""",
    ),
    BenchmarkPair(
        name="null_vs_option",
        category="error_handling",
        expected_result="divergent",
        description="C uses NULL pointer, Rust uses Option",
        divergence_kind="null_handling",
        c_source="""
int deref_or_default(int* ptr, int default_val) {
    if (ptr == 0) return default_val;
    return *ptr;
}
""",
        rust_source="""
pub fn deref_or_default(val: Option<i32>, default_val: i32) -> i32 {
    match val {
        Some(v) => v,
        None => default_val,
    }
}
""",
    ),
    BenchmarkPair(
        name="find_char",
        category="error_handling",
        expected_result="equivalent",
        description="Find character in bounded array",
        c_source="""
int find_char(int* arr, int len, int target) {
    for (int i = 0; i < len; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}
""",
        rust_source="""
pub fn find_char(arr: &[i32], target: i32) -> i32 {
    let mut i: i32 = 0;
    let len = arr.len() as i32;
    while i < len {
        if arr[i as usize] == target { return i; }
        i += 1;
    }
    -1
}
""",
    ),
    BenchmarkPair(
        name="max_of_three",
        category="error_handling",
        expected_result="equivalent",
        description="Maximum of three values (no overflow)",
        c_source="""
int max3(int a, int b, int c) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}
""",
        rust_source="""
pub fn max3(a: i32, b: i32, c: i32) -> i32 {
    let mut m = a;
    if b > m { m = b; }
    if c > m { m = c; }
    m
}
""",
    ),
    BenchmarkPair(
        name="sign_function",
        category="error_handling",
        expected_result="equivalent",
        description="Sign function (-1, 0, 1)",
        c_source="""
int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
""",
        rust_source="""
pub fn sign(x: i32) -> i32 {
    if x > 0 { 1 }
    else if x < 0 { -1 }
    else { 0 }
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category 5: Bitwise operations (8 pairs)
# ---------------------------------------------------------------------------

BITWISE_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="is_power_of_two",
        category="bitwise",
        expected_result="equivalent",
        description="Check if x is a power of 2",
        c_source="""
int is_power_of_two(unsigned int x) {
    return x != 0 && (x & (x - 1)) == 0;
}
""",
        rust_source="""
pub fn is_power_of_two(x: u32) -> i32 {
    if x != 0 && (x & (x.wrapping_sub(1))) == 0 { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="rotate_right",
        category="bitwise",
        expected_result="divergent",
        description="Rotate: C shift by 32 is UB when n=0, Rust masks shift amount",
        divergence_kind="shift_overflow",
        c_source="""
unsigned int rotate_right(unsigned int x, int n) {
    n = n & 31;
    return (x >> n) | (x << (32 - n));
}
""",
        rust_source="""
pub fn rotate_right(x: u32, n: i32) -> u32 {
    let n = (n & 31) as u32;
    (x >> n) | (x << (32 - n))
}
""",
    ),
    BenchmarkPair(
        name="swap_bytes",
        category="bitwise",
        expected_result="equivalent",
        description="Byte swap (endianness conversion)",
        c_source="""
unsigned int swap_bytes(unsigned int x) {
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8)  |
           ((x & 0x0000FF00) << 8)  |
           ((x & 0x000000FF) << 24);
}
""",
        rust_source="""
pub fn swap_bytes(x: u32) -> u32 {
    ((x & 0xFF000000) >> 24) |
    ((x & 0x00FF0000) >> 8)  |
    ((x & 0x0000FF00) << 8)  |
    ((x & 0x000000FF) << 24)
}
""",
    ),
    BenchmarkPair(
        name="bit_extract",
        category="bitwise",
        expected_result="divergent",
        description="Bit extract: C shift by 32 is UB when hi-lo=32, Rust wraps",
        divergence_kind="shift_overflow",
        c_source="""
unsigned int extract_bits(unsigned int x, int lo, int hi) {
    if (lo < 0 || hi > 32 || lo >= hi) return 0;
    unsigned int mask = ((1u << (hi - lo)) - 1) << lo;
    return (x & mask) >> lo;
}
""",
        rust_source="""
pub fn extract_bits(x: u32, lo: i32, hi: i32) -> u32 {
    if lo < 0 || hi > 32 || lo >= hi { return 0; }
    let width = (hi - lo) as u32;
    let mask = ((1u32 << width) - 1) << (lo as u32);
    (x & mask) >> (lo as u32)
}
""",
    ),
    BenchmarkPair(
        name="leading_zeros",
        category="bitwise",
        expected_result="equivalent",
        description="Count leading zeros",
        c_source="""
int leading_zeros(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8;  x <<= 8; }
    if (x <= 0x0FFFFFFF) { n += 4;  x <<= 4; }
    if (x <= 0x3FFFFFFF) { n += 2;  x <<= 2; }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
}
""",
        rust_source="""
pub fn leading_zeros(x: u32) -> i32 {
    if x == 0 { return 32; }
    let mut n: i32 = 0;
    let mut v = x;
    if v <= 0x0000FFFF { n += 16; v <<= 16; }
    if v <= 0x00FFFFFF { n += 8;  v <<= 8; }
    if v <= 0x0FFFFFFF { n += 4;  v <<= 4; }
    if v <= 0x3FFFFFFF { n += 2;  v <<= 2; }
    if v <= 0x7FFFFFFF { n += 1; }
    n
}
""",
    ),
    BenchmarkPair(
        name="parity",
        category="bitwise",
        expected_result="equivalent",
        description="Compute parity (XOR fold)",
        c_source="""
int parity(unsigned int x) {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1;
}
""",
        rust_source="""
pub fn parity(mut x: u32) -> i32 {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    (x & 1) as i32
}
""",
    ),
    BenchmarkPair(
        name="interleave_bits",
        category="bitwise",
        expected_result="equivalent",
        description="Interleave bits of two 16-bit values",
        c_source="""
unsigned int interleave(unsigned int x, unsigned int y) {
    x = x & 0xFFFF;
    y = y & 0xFFFF;
    unsigned int result = 0;
    for (int i = 0; i < 16; i++) {
        result |= ((x >> i) & 1) << (2 * i);
        result |= ((y >> i) & 1) << (2 * i + 1);
    }
    return result;
}
""",
        rust_source="""
pub fn interleave(x: u32, y: u32) -> u32 {
    let x = x & 0xFFFF;
    let y = y & 0xFFFF;
    let mut result: u32 = 0;
    let mut i: i32 = 0;
    while i < 16 {
        result |= ((x >> (i as u32)) & 1) << (2 * i as u32);
        result |= ((y >> (i as u32)) & 1) << (2 * i as u32 + 1);
        i += 1;
    }
    result
}
""",
    ),
    BenchmarkPair(
        name="bit_reverse_byte",
        category="bitwise",
        expected_result="equivalent",
        description="Reverse bits within a byte",
        c_source="""
unsigned char reverse_byte(unsigned char b) {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    return b;
}
""",
        rust_source="""
pub fn reverse_byte(mut b: u8) -> u8 {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    b
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category 6: String and memory patterns (8 pairs)
# ---------------------------------------------------------------------------

STRING_MEMORY_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="hash_djb2",
        category="string",
        expected_result="equivalent",
        description="DJB2 hash function (wrapping arithmetic)",
        c_source="""
unsigned int hash_djb2(int* data, int len) {
    unsigned int hash = 5381;
    for (int i = 0; i < len; i++) {
        hash = hash * 33 + (unsigned int)data[i];
    }
    return hash;
}
""",
        rust_source="""
pub fn hash_djb2(data: &[i32]) -> u32 {
    let mut hash: u32 = 5381;
    for &d in data.iter() {
        hash = hash.wrapping_mul(33).wrapping_add(d as u32);
    }
    hash
}
""",
    ),
    BenchmarkPair(
        name="array_sum",
        category="string",
        expected_result="divergent",
        description="Array sum: overflow behavior differs",
        divergence_kind="signed_overflow",
        c_source="""
int array_sum(int* arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}
""",
        rust_source="""
pub fn array_sum(arr: &[i32]) -> i32 {
    let mut sum: i32 = 0;
    for &x in arr.iter() {
        sum = sum.wrapping_add(x);
    }
    sum
}
""",
    ),
    BenchmarkPair(
        name="memset_pattern",
        category="memory",
        expected_result="equivalent",
        description="Fill array with value",
        c_source="""
void fill_array(int* arr, int len, int val) {
    for (int i = 0; i < len; i++) {
        arr[i] = val;
    }
}
""",
        rust_source="""
pub fn fill_array(arr: &mut [i32], val: i32) {
    for x in arr.iter_mut() {
        *x = val;
    }
}
""",
    ),
    BenchmarkPair(
        name="array_max",
        category="memory",
        expected_result="equivalent",
        description="Find maximum in array",
        c_source="""
int array_max(int* arr, int len) {
    if (len <= 0) return -2147483648;
    int m = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] > m) m = arr[i];
    }
    return m;
}
""",
        rust_source="""
pub fn array_max(arr: &[i32]) -> i32 {
    if arr.is_empty() { return i32::MIN; }
    let mut m = arr[0];
    let mut i: usize = 1;
    while i < arr.len() {
        if arr[i] > m { m = arr[i]; }
        i += 1;
    }
    m
}
""",
    ),
    BenchmarkPair(
        name="copy_array",
        category="memory",
        expected_result="equivalent",
        description="Copy array elements",
        c_source="""
void copy_array(int* dst, int* src, int len) {
    for (int i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}
""",
        rust_source="""
pub fn copy_array(dst: &mut [i32], src: &[i32]) {
    let len = if dst.len() < src.len() { dst.len() } else { src.len() };
    let mut i: usize = 0;
    while i < len {
        dst[i] = src[i];
        i += 1;
    }
}
""",
    ),
    BenchmarkPair(
        name="dot_product",
        category="memory",
        expected_result="divergent",
        description="Dot product (accumulation overflow)",
        divergence_kind="signed_overflow",
        c_source="""
int dot_product(int* a, int* b, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
""",
        rust_source="""
pub fn dot_product(a: &[i32], b: &[i32]) -> i32 {
    let mut sum: i32 = 0;
    let len = if a.len() < b.len() { a.len() } else { b.len() };
    let mut i: usize = 0;
    while i < len {
        sum = sum.wrapping_add(a[i].wrapping_mul(b[i]));
        i += 1;
    }
    sum
}
""",
    ),
    BenchmarkPair(
        name="insertion_sort",
        category="memory",
        expected_result="equivalent",
        description="Insertion sort (no overflow in comparisons)",
        c_source="""
void insertion_sort(int* arr, int len) {
    for (int i = 1; i < len; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}
""",
        rust_source="""
pub fn insertion_sort(arr: &mut [i32]) {
    let len = arr.len();
    let mut i: usize = 1;
    while i < len {
        let key = arr[i];
        let mut j = i as i32 - 1;
        while j >= 0 && arr[j as usize] > key {
            arr[(j + 1) as usize] = arr[j as usize];
            j -= 1;
        }
        arr[(j + 1) as usize] = key;
        i += 1;
    }
}
""",
    ),
    BenchmarkPair(
        name="linear_search",
        category="memory",
        expected_result="equivalent",
        description="Linear search returning index",
        c_source="""
int linear_search(int* arr, int len, int target) {
    for (int i = 0; i < len; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}
""",
        rust_source="""
pub fn linear_search(arr: &[i32], target: i32) -> i32 {
    let mut i: i32 = 0;
    let len = arr.len() as i32;
    while i < len {
        if arr[i as usize] == target { return i; }
        i += 1;
    }
    -1
}
""",
    ),
]


# ---------------------------------------------------------------------------
# All benchmarks
# ---------------------------------------------------------------------------

ALL_BENCHMARKS: List[BenchmarkPair] = (
    ARITHMETIC_PAIRS +
    DIVISION_SHIFT_PAIRS +
    LOOP_PAIRS +
    ERROR_HANDLING_PAIRS +
    BITWISE_PAIRS +
    STRING_MEMORY_PAIRS
)


def get_benchmarks_by_category(category: str) -> List[BenchmarkPair]:
    return [b for b in ALL_BENCHMARKS if b.category == category]


def get_equivalent_benchmarks() -> List[BenchmarkPair]:
    return [b for b in ALL_BENCHMARKS if b.expected_result == "equivalent"]


def get_divergent_benchmarks() -> List[BenchmarkPair]:
    return [b for b in ALL_BENCHMARKS if b.expected_result == "divergent"]


def get_all_categories() -> List[str]:
    return sorted(set(b.category for b in ALL_BENCHMARKS))


def get_all_pairs() -> List[BenchmarkPair]:
    return list(ALL_BENCHMARKS)
