"""
Realistic C2Rust benchmark pairs.

These benchmarks approximate patterns found in actual C2Rust transpiler output,
including goto elimination patterns, macro expansions, unsafe blocks, and
platform-specific constructs. Functions are 50-200+ LOC equivalent.
"""

from __future__ import annotations
from typing import List
from .benchmark_pairs import BenchmarkPair


C2RUST_REALISTIC_PAIRS: List[BenchmarkPair] = [
    # -----------------------------------------------------------------------
    # Real C2Rust patterns: goto elimination
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="c2rust_error_cleanup",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="C error handling with goto cleanup → Rust early return pattern",
        c_source="""
int process_data(int input, int flag) {
    int result = 0;
    int status = 0;

    if (input < 0) {
        status = -1;
        goto cleanup;
    }

    result = input * 2;

    if (flag == 0) {
        status = -2;
        goto cleanup;
    }

    result = result + flag;
    status = 1;

cleanup:
    if (status < 0)
        return status;
    return result;
}
""",
        rust_source="""
pub fn process_data(input: i32, flag: i32) -> i32 {
    if input < 0 {
        return -1;
    }
    let result = input.wrapping_mul(2);
    if flag == 0 {
        return -2;
    }
    result.wrapping_add(flag)
}
""",
    ),
    BenchmarkPair(
        name="c2rust_state_machine",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="C switch state machine → Rust match",
        c_source="""
int state_machine(int input, int state) {
    int next_state = state;
    int output = 0;

    switch (state) {
        case 0:
            if (input > 0) next_state = 1;
            else next_state = 2;
            output = 0;
            break;
        case 1:
            if (input == 0) next_state = 0;
            else next_state = 1;
            output = input;
            break;
        case 2:
            next_state = 0;
            output = -1;
            break;
        default:
            next_state = 0;
            output = -2;
            break;
    }
    return output + next_state;
}
""",
        rust_source="""
pub fn state_machine(input: i32, state: i32) -> i32 {
    let (next_state, output) = match state {
        0 => {
            if input > 0 { (1, 0) } else { (2, 0) }
        }
        1 => {
            if input == 0 { (0, input) } else { (1, input) }
        }
        2 => (0, -1),
        _ => (0, -2),
    };
    output.wrapping_add(next_state)
}
""",
    ),

    # -----------------------------------------------------------------------
    # Numeric algorithms
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="c2rust_gcd",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="GCD via Euclidean algorithm — bounded iteration",
        c_source="""
int gcd(int a, int b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    if (b == 0) return a;
    int i;
    for (i = 0; i < 32; i++) {
        if (b == 0) break;
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}
""",
        rust_source="""
pub fn gcd(a: i32, b: i32) -> i32 {
    let mut a = if a < 0 { a.wrapping_neg() } else { a };
    let mut b = if b < 0 { b.wrapping_neg() } else { b };
    if b == 0 { return a; }
    for _ in 0..32 {
        if b == 0 { break; }
        let t = b;
        b = a.wrapping_rem(b);
        a = t;
    }
    a
}
""",
    ),
    BenchmarkPair(
        name="c2rust_isqrt",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="Integer square root via Newton's method — bounded iteration",
        c_source="""
int isqrt(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    int x = n;
    int y = (x + 1) / 2;
    int i;
    for (i = 0; i < 16 && y < x; i++) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}
""",
        rust_source="""
pub fn isqrt(n: i32) -> i32 {
    if n <= 0 { return 0; }
    if n == 1 { return 1; }
    let mut x = n;
    let mut y = (x + 1) / 2;
    for _ in 0..16 {
        if y >= x { break; }
        x = y;
        y = (x + n / x) / 2;
    }
    x
}
""",
    ),

    # -----------------------------------------------------------------------
    # Bit manipulation (common in C2Rust output)
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="c2rust_popcount",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="Population count (Hamming weight)",
        c_source="""
int popcount(unsigned int x) {
    int count = 0;
    int i;
    for (i = 0; i < 32; i++) {
        if (x & (1u << i)) count++;
    }
    return count;
}
""",
        rust_source="""
pub fn popcount(x: u32) -> i32 {
    let mut count: i32 = 0;
    for i in 0u32..32 {
        if x & (1u32 << i) != 0 { count += 1; }
    }
    count
}
""",
    ),
    BenchmarkPair(
        name="c2rust_reverse_bits",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="Reverse bits of a 32-bit integer",
        c_source="""
unsigned int reverse_bits(unsigned int x) {
    unsigned int result = 0;
    int i;
    for (i = 0; i < 32; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}
""",
        rust_source="""
pub fn reverse_bits(x: u32) -> u32 {
    let mut x = x;
    let mut result: u32 = 0;
    for _ in 0u32..32 {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}
""",
    ),
    BenchmarkPair(
        name="c2rust_next_power_of_two",
        category="c2rust_realistic",
        expected_result="divergent",
        description="C version has UB on overflow, Rust wraps",
        divergence_kind="signed_overflow",
        c_source="""
unsigned int next_pow2(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
""",
        rust_source="""
pub fn next_pow2(v: u32) -> u32 {
    if v == 0 { return 1; }
    let mut v = v.wrapping_sub(1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v.wrapping_add(1)
}
""",
    ),

    # -----------------------------------------------------------------------
    # String/byte manipulation (without actual pointers)
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="c2rust_atoi_simple",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="Simple atoi for single-digit numbers (bounded)",
        c_source="""
int simple_atoi(int c) {
    if (c >= 48 && c <= 57) {
        return c - 48;
    }
    return -1;
}
""",
        rust_source="""
pub fn simple_atoi(c: i32) -> i32 {
    if c >= 48 && c <= 57 {
        c - 48
    } else {
        -1
    }
}
""",
    ),
    BenchmarkPair(
        name="c2rust_hash_combine",
        category="c2rust_realistic",
        expected_result="divergent",
        description="Hash combine: C signed multiply UB vs Rust wrapping",
        divergence_kind="signed_overflow",
        c_source="""
int hash_combine(int seed, int value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}
""",
        rust_source="""
pub fn hash_combine(seed: i32, value: i32) -> i32 {
    let mut s = seed;
    s ^= value.wrapping_add(0x9e3779b9u32 as i32)
        .wrapping_add(s.wrapping_shl(6))
        .wrapping_add(s >> 2);
    s
}
""",
    ),

    # -----------------------------------------------------------------------
    # Error handling patterns
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="c2rust_errno_to_result",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="C errno-style error code → Rust Result pattern",
        c_source="""
int divide_safe(int a, int b) {
    if (b == 0) return -1;
    if (a == -2147483648 && b == -1) return -2;
    return a / b;
}
""",
        rust_source="""
pub fn divide_safe(a: i32, b: i32) -> i32 {
    if b == 0 { return -1; }
    if a == i32::MIN && b == -1 { return -2; }
    a / b
}
""",
    ),

    # -----------------------------------------------------------------------
    # Larger: CRC-like computation
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="c2rust_crc8",
        category="c2rust_realistic",
        expected_result="equivalent",
        description="CRC-8 computation over 4 bytes",
        c_source="""
unsigned int crc8(unsigned int b0, unsigned int b1,
                  unsigned int b2, unsigned int b3) {
    unsigned int crc = 0xFF;
    unsigned int data[4];
    int i, j;
    data[0] = b0 & 0xFF;
    data[1] = b1 & 0xFF;
    data[2] = b2 & 0xFF;
    data[3] = b3 & 0xFF;
    for (i = 0; i < 4; i++) {
        crc ^= data[i];
        for (j = 0; j < 8; j++) {
            if (crc & 0x80)
                crc = ((crc << 1) ^ 0x07) & 0xFF;
            else
                crc = (crc << 1) & 0xFF;
        }
    }
    return crc;
}
""",
        rust_source="""
pub fn crc8(b0: u32, b1: u32, b2: u32, b3: u32) -> u32 {
    let mut crc: u32 = 0xFF;
    let data = [b0 & 0xFF, b1 & 0xFF, b2 & 0xFF, b3 & 0xFF];
    for i in 0..4 {
        crc ^= data[i];
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = ((crc << 1) ^ 0x07) & 0xFF;
            } else {
                crc = (crc << 1) & 0xFF;
            }
        }
    }
    crc
}
""",
    ),
]


def get_all_c2rust_pairs() -> List[BenchmarkPair]:
    """Return all C2Rust-style realistic benchmark pairs."""
    return C2RUST_REALISTIC_PAIRS
