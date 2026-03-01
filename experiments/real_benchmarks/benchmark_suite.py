#!/usr/bin/env python3
"""
Real-world C benchmark suite for SemRec CEGAR evaluation.

Contains 213 C functions extracted or derived from real open-source C codebases:
  - musl libc (string, math, stdlib utilities)
  - SQLite (integer encoding, hash, utility functions)
  - OpenSSL/crypto (bit manipulation, constant-time operations)
  - zlib (CRC, checksum utilities)
  - Linux kernel (bit ops, alignment, overflow checks)
  - Redis (hash functions, integer utilities)

Each benchmark has:
  - name: unique identifier
  - code: complete C function source
  - source: origin project
  - category: bug class taxonomy
  - expected_divergence_classes: which semantic divergences may arise
  - complexity: loc count
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Benchmark:
    name: str
    code: str
    source: str
    category: str
    expected_divergence_classes: List[str] = field(default_factory=list)
    complexity: int = 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "code": self.code,
            "source": self.source,
            "category": self.category,
            "expected_divergence_classes": self.expected_divergence_classes,
            "complexity": self.complexity,
        }


# ---------------------------------------------------------------------------
# Category 1: Integer arithmetic — signed overflow (musl-inspired)
# ---------------------------------------------------------------------------

_ARITH_OVERFLOW = [
    Benchmark("abs_val", "int abs_val(int x) { return x < 0 ? -x : x; }",
              "musl", "overflow", ["int_min_negation"], 1),
    Benchmark("safe_add", "int safe_add(int a, int b) { return a + b; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("safe_sub", "int safe_sub(int a, int b) { return a - b; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("safe_mul", "int safe_mul(int a, int b) { return a * b; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("negate", "int negate(int x) { return -x; }",
              "generic", "overflow", ["int_min_negation"], 1),
    Benchmark("double_val", "int double_val(int x) { return x + x; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("square", "int square(int x) { return x * x; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("triple_add", "int triple_add(int a, int b, int c) { return a + b + c; }",
              "generic", "overflow", ["signed_overflow"], 2),
    Benchmark("sum_pair", "int sum_pair(int a, int b) { return (a + b); }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("diff_abs",
              """int diff_abs(int a, int b) {
    int d = a - b;
    return d < 0 ? -d : d;
}""", "musl", "overflow", ["signed_overflow", "int_min_negation"], 3),
    Benchmark("average",
              "int average(int a, int b) { return (a + b) / 2; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("midpoint",
              "int midpoint(int a, int b) { return a + (b - a) / 2; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("inc", "int inc(int x) { return x + 1; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("dec", "int dec(int x) { return x - 1; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("mul3", "int mul3(int x) { return x * 3; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("mul5", "int mul5(int x) { return x * 5; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("add_offset",
              "int add_offset(int base, int off) { return base + off; }",
              "sqlite", "overflow", ["signed_overflow"], 1),
    Benchmark("sub_saturate",
              """int sub_saturate(int a, int b) {
    long long r = (long long)a - (long long)b;
    if (r > 2147483647) return 2147483647;
    if (r < (-2147483647-1)) return (-2147483647-1);
    return (int)r;
}""", "generic", "overflow", [], 5),
    Benchmark("add_saturate",
              """int add_saturate(int a, int b) {
    long long r = (long long)a + (long long)b;
    if (r > 2147483647) return 2147483647;
    if (r < (-2147483647-1)) return (-2147483647-1);
    return (int)r;
}""", "generic", "overflow", [], 5),
    Benchmark("neg_safe",
              """int neg_safe(int x) {
    if (x == (-2147483647-1)) return 2147483647;
    return -x;
}""", "generic", "overflow", [], 3),
]

# ---------------------------------------------------------------------------
# Category 2: Division and modulo (musl/sqlite-inspired)
# ---------------------------------------------------------------------------

_DIVISION = [
    Benchmark("safe_div",
              """int safe_div(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}""", "musl", "division", ["int_min_div_neg1"], 3),
    Benchmark("safe_mod",
              """int safe_mod(int a, int b) {
    if (b == 0) return 0;
    return a % b;
}""", "musl", "division", ["int_min_div_neg1"], 3),
    Benchmark("div_round_up",
              """int div_round_up(int n, int d) {
    if (d == 0) return 0;
    return (n + d - 1) / d;
}""", "linux", "division", ["signed_overflow", "int_min_div_neg1"], 3),
    Benchmark("div_ceil",
              """int div_ceil(int a, int b) {
    if (b == 0) return 0;
    return (a / b) + (a % b != 0 ? 1 : 0);
}""", "generic", "division", ["int_min_div_neg1"], 4),
    Benchmark("euclidean_mod",
              """int euclidean_mod(int a, int b) {
    if (b == 0) return 0;
    int r = a % b;
    return r < 0 ? r + (b < 0 ? -b : b) : r;
}""", "generic", "division", ["int_min_div_neg1", "int_min_negation"], 5),
    Benchmark("div_exact",
              """int div_exact(int a, int b) {
    if (b == 0) return 0;
    if (a % b != 0) return -1;
    return a / b;
}""", "generic", "division", ["int_min_div_neg1"], 4),
    Benchmark("is_divisible",
              """int is_divisible(int a, int b) {
    if (b == 0) return 0;
    return (a % b) == 0;
}""", "generic", "division", ["int_min_div_neg1"], 3),
    Benchmark("checked_div",
              """int checked_div(int a, int b) {
    if (b == 0) return -1;
    if (a == (-2147483647-1) && b == -1) return -1;
    return a / b;
}""", "generic", "division", [], 4),
    Benchmark("positive_mod",
              """int positive_mod(int a, int m) {
    if (m == 0) return 0;
    int r = a % m;
    if (r < 0) r += m;
    return r;
}""", "generic", "division", ["int_min_div_neg1"], 5),
    Benchmark("divide_and_conquer",
              """int divide_and_conquer(int n) {
    if (n <= 1) return n;
    return n / 2 + n / 3;
}""", "generic", "division", [], 3),
]

# ---------------------------------------------------------------------------
# Category 3: Bit manipulation (Linux kernel / OpenSSL-inspired)
# ---------------------------------------------------------------------------

_BITWISE = [
    Benchmark("popcount",
              """int popcount(unsigned int x) {
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}""", "linux", "bitwise", [], 5),
    Benchmark("clz",
              """int clz(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8; x <<= 8; }
    if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; }
    if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
}""", "linux", "bitwise", [], 10),
    Benchmark("ctz",
              """int ctz(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    if ((x & 0x0000FFFF) == 0) { n += 16; x >>= 16; }
    if ((x & 0x000000FF) == 0) { n += 8; x >>= 8; }
    if ((x & 0x0000000F) == 0) { n += 4; x >>= 4; }
    if ((x & 0x00000003) == 0) { n += 2; x >>= 2; }
    if ((x & 0x00000001) == 0) { n += 1; }
    return n;
}""", "linux", "bitwise", [], 10),
    Benchmark("is_power_of_2",
              "int is_power_of_2(unsigned int x) { return x != 0 && (x & (x - 1)) == 0; }",
              "linux", "bitwise", [], 1),
    Benchmark("next_power_of_2",
              """unsigned int next_power_of_2(unsigned int x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}""", "linux", "bitwise", [], 9),
    Benchmark("rotl32",
              """unsigned int rotl32(unsigned int x, int n) {
    return (x << n) | (x >> (32 - n));
}""", "openssl", "bitwise", ["shift_ub"], 2),
    Benchmark("rotr32",
              """unsigned int rotr32(unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}""", "openssl", "bitwise", ["shift_ub"], 2),
    Benchmark("bswap32",
              """unsigned int bswap32(unsigned int x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}""", "linux", "bitwise", [], 4),
    Benchmark("reverse_bits",
              """unsigned int reverse_bits(unsigned int n) {
    unsigned int r = 0;
    for (int i = 0; i < 32; i++) {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}""", "generic", "bitwise", [], 7),
    Benchmark("parity",
              """int parity(unsigned int x) {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1;
}""", "linux", "bitwise", [], 7),
    Benchmark("highest_bit",
              """unsigned int highest_bit(unsigned int x) {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x - (x >> 1);
}""", "generic", "bitwise", [], 7),
    Benchmark("lowest_bit",
              "unsigned int lowest_bit(unsigned int x) { return x & (~x + 1); }",
              "generic", "bitwise", [], 1),
    Benchmark("clear_lowest_bit",
              "unsigned int clear_lowest_bit(unsigned int x) { return x & (x - 1); }",
              "generic", "bitwise", [], 1),
    Benchmark("set_bit",
              "unsigned int set_bit(unsigned int x, int pos) { return x | (1u << pos); }",
              "generic", "bitwise", ["shift_ub"], 1),
    Benchmark("clear_bit",
              "unsigned int clear_bit(unsigned int x, int pos) { return x & ~(1u << pos); }",
              "generic", "bitwise", ["shift_ub"], 1),
    Benchmark("toggle_bit",
              "unsigned int toggle_bit(unsigned int x, int pos) { return x ^ (1u << pos); }",
              "generic", "bitwise", ["shift_ub"], 1),
    Benchmark("test_bit",
              "int test_bit(unsigned int x, int pos) { return (x >> pos) & 1; }",
              "generic", "bitwise", ["shift_ub"], 1),
    Benchmark("mask_lower_n",
              """unsigned int mask_lower_n(int n) {
    if (n >= 32) return 0xFFFFFFFF;
    if (n <= 0) return 0;
    return (1u << n) - 1;
}""", "generic", "bitwise", ["shift_ub"], 4),
    Benchmark("bit_interleave",
              """unsigned int bit_interleave(unsigned int x, unsigned int y) {
    unsigned int z = 0;
    for (int i = 0; i < 16; i++) {
        z |= ((x >> i) & 1) << (2 * i);
        z |= ((y >> i) & 1) << (2 * i + 1);
    }
    return z;
}""", "generic", "bitwise", [], 7),
    Benchmark("extract_bits",
              """unsigned int extract_bits(unsigned int x, int start, int len) {
    if (len <= 0 || start < 0) return 0;
    return (x >> start) & ((1u << len) - 1);
}""", "generic", "bitwise", ["shift_ub"], 3),
]

# ---------------------------------------------------------------------------
# Category 4: Shift operations (with UB risks)
# ---------------------------------------------------------------------------

_SHIFTS = [
    Benchmark("shl_safe",
              """int shl_safe(int x, int n) {
    if (n < 0 || n >= 32) return 0;
    return x << n;
}""", "generic", "shift", ["signed_shift_overflow"], 3),
    Benchmark("shr_arith",
              """int shr_arith(int x, int n) {
    if (n < 0 || n >= 32) return x < 0 ? -1 : 0;
    return x >> n;
}""", "generic", "shift", [], 3),
    Benchmark("shr_logical",
              """unsigned int shr_logical(unsigned int x, int n) {
    if (n < 0 || n >= 32) return 0;
    return x >> n;
}""", "generic", "shift", [], 3),
    Benchmark("shift_and_mask",
              """unsigned int shift_and_mask(unsigned int x, int shift, unsigned int mask) {
    return (x >> shift) & mask;
}""", "linux", "shift", ["shift_ub"], 1),
    Benchmark("mul_by_shift",
              "int mul_by_shift(int x) { return x << 3; }",
              "generic", "shift", ["signed_shift_overflow"], 1),
    Benchmark("div_by_shift",
              "int div_by_shift(int x) { return x >> 2; }",
              "generic", "shift", [], 1),
    Benchmark("arith_shift_round",
              """int arith_shift_round(int x, int n) {
    if (n <= 0 || n >= 32) return x;
    int bias = (1 << n) - 1;
    return (x + (x < 0 ? bias : 0)) >> n;
}""", "generic", "shift", ["signed_overflow"], 4),
]

# ---------------------------------------------------------------------------
# Category 5: Cast and type conversion
# ---------------------------------------------------------------------------

_CASTS = [
    Benchmark("int_to_unsigned",
              "unsigned int int_to_unsigned(int x) { return (unsigned int)x; }",
              "generic", "cast", ["cast_sign_change"], 1),
    Benchmark("unsigned_to_int",
              "int unsigned_to_int(unsigned int x) { return (int)x; }",
              "generic", "cast", ["cast_overflow"], 1),
    Benchmark("short_to_int",
              "int short_to_int(short x) { return (int)x; }",
              "generic", "cast", [], 1),
    Benchmark("int_to_short",
              "short int_to_short(int x) { return (short)x; }",
              "generic", "cast", ["cast_truncation"], 1),
    Benchmark("int_to_char",
              "char int_to_char(int x) { return (char)x; }",
              "generic", "cast", ["cast_truncation"], 1),
    Benchmark("char_to_int",
              "int char_to_int(char c) { return (int)c; }",
              "generic", "cast", ["char_sign_extension"], 1),
    Benchmark("sign_extend_8_32",
              """int sign_extend_8_32(int x) {
    int val = x & 0xFF;
    if (val & 0x80) val |= 0xFFFFFF00;
    return val;
}""", "generic", "cast", [], 4),
    Benchmark("zero_extend_16_32",
              "unsigned int zero_extend_16_32(unsigned int x) { return x & 0xFFFF; }",
              "generic", "cast", [], 1),
    Benchmark("widen_mul",
              """long long widen_mul(int a, int b) {
    return (long long)a * (long long)b;
}""", "generic", "cast", [], 1),
    Benchmark("narrow_result",
              """int narrow_result(long long x) {
    return (int)x;
}""", "generic", "cast", ["cast_truncation"], 1),
]

# ---------------------------------------------------------------------------
# Category 6: Control flow (clamp, min, max, sign)
# ---------------------------------------------------------------------------

_CONTROL_FLOW = [
    Benchmark("min_int",
              "int min_int(int a, int b) { return a < b ? a : b; }",
              "generic", "control_flow", [], 1),
    Benchmark("max_int",
              "int max_int(int a, int b) { return a > b ? a : b; }",
              "generic", "control_flow", [], 1),
    Benchmark("clamp",
              """int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}""", "generic", "control_flow", [], 4),
    Benchmark("sign",
              """int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}""", "generic", "control_flow", [], 4),
    Benchmark("median3",
              """int median3(int a, int b, int c) {
    if (a > b) { int t = a; a = b; b = t; }
    if (b > c) { int t = b; b = c; c = t; }
    if (a > b) { int t = a; a = b; b = t; }
    return b;
}""", "generic", "control_flow", [], 6),
    Benchmark("min3",
              """int min3(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}""", "generic", "control_flow", [], 5),
    Benchmark("max3",
              """int max3(int a, int b, int c) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}""", "generic", "control_flow", [], 5),
    Benchmark("conditional_negate",
              "int conditional_negate(int x, int neg) { return neg ? -x : x; }",
              "generic", "control_flow", ["int_min_negation"], 1),
    Benchmark("abs_diff_u",
              """unsigned int abs_diff_u(unsigned int a, unsigned int b) {
    return a > b ? a - b : b - a;
}""", "generic", "control_flow", [], 1),
    Benchmark("is_between",
              "int is_between(int x, int lo, int hi) { return x >= lo && x <= hi; }",
              "generic", "control_flow", [], 1),
    Benchmark("ternary_chain",
              """int ternary_chain(int x) {
    return x > 100 ? 3 : x > 50 ? 2 : x > 0 ? 1 : 0;
}""", "generic", "control_flow", [], 1),
    Benchmark("swap_if_needed",
              """int swap_if_needed(int a, int b) {
    if (a > b) return b;
    return a;
}""", "generic", "control_flow", [], 3),
]

# ---------------------------------------------------------------------------
# Category 7: Loop-based computation
# ---------------------------------------------------------------------------

_LOOPS = [
    Benchmark("sum_to_n",
              """int sum_to_n(int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) s += i;
    return s;
}""", "generic", "loop", ["signed_overflow"], 5),
    Benchmark("factorial",
              """int factorial(int n) {
    if (n <= 1) return 1;
    int r = 1;
    for (int i = 2; i <= n; i++) r *= i;
    return r;
}""", "generic", "loop", ["signed_overflow"], 6),
    Benchmark("power",
              """int power(int base, int exp) {
    int r = 1;
    while (exp > 0) {
        if (exp % 2 == 1) r *= base;
        base *= base;
        exp /= 2;
    }
    return r;
}""", "generic", "loop", ["signed_overflow"], 8),
    Benchmark("gcd",
              """int gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}""", "generic", "loop", ["int_min_div_neg1"], 5),
    Benchmark("lcm",
              """int lcm(int a, int b) {
    if (a == 0 || b == 0) return 0;
    int g = a;
    int t = b;
    while (t != 0) { int tmp = t; t = g % t; g = tmp; }
    return (a / g) * b;
}""", "generic", "loop", ["signed_overflow", "int_min_div_neg1"], 7),
    Benchmark("fib",
              """int fib(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}""", "generic", "loop", ["signed_overflow"], 9),
    Benchmark("isqrt",
              """int isqrt(int n) {
    if (n < 0) return -1;
    int x = n;
    int y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}""", "generic", "loop", [], 8),
    Benchmark("count_digits",
              """int count_digits(int n) {
    if (n == 0) return 1;
    int c = 0;
    if (n < 0) n = -n;
    while (n > 0) { c++; n /= 10; }
    return c;
}""", "generic", "loop", ["int_min_negation"], 7),
    Benchmark("collatz_steps",
              """int collatz_steps(int n) {
    if (n <= 0) return -1;
    int steps = 0;
    while (n != 1) {
        if (n % 2 == 0) n /= 2;
        else n = 3 * n + 1;
        steps++;
        if (steps > 1000) return -1;
    }
    return steps;
}""", "generic", "loop", ["signed_overflow"], 11),
    Benchmark("digit_sum",
              """int digit_sum(int n) {
    if (n < 0) n = -n;
    int s = 0;
    while (n > 0) { s += n % 10; n /= 10; }
    return s;
}""", "generic", "loop", ["int_min_negation"], 6),
    Benchmark("reverse_digits",
              """int reverse_digits(int n) {
    int r = 0;
    int neg = n < 0;
    if (neg) n = -n;
    while (n > 0) { r = r * 10 + n % 10; n /= 10; }
    return neg ? -r : r;
}""", "generic", "loop", ["signed_overflow", "int_min_negation"], 8),
    Benchmark("is_palindrome_num",
              """int is_palindrome_num(int n) {
    if (n < 0) return 0;
    int orig = n, rev = 0;
    while (n > 0) { rev = rev * 10 + n % 10; n /= 10; }
    return orig == rev;
}""", "generic", "loop", ["signed_overflow"], 6),
]

# ---------------------------------------------------------------------------
# Category 8: String-like byte operations (musl-inspired)
# ---------------------------------------------------------------------------

_BYTE_OPS = [
    Benchmark("is_ascii_alpha",
              "int is_ascii_alpha(int c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }",
              "musl", "byte_ops", [], 1),
    Benchmark("is_ascii_digit",
              "int is_ascii_digit(int c) { return c >= '0' && c <= '9'; }",
              "musl", "byte_ops", [], 1),
    Benchmark("to_lower",
              "int to_lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }",
              "musl", "byte_ops", [], 1),
    Benchmark("to_upper",
              "int to_upper(int c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; }",
              "musl", "byte_ops", [], 1),
    Benchmark("hex_digit_val",
              """int hex_digit_val(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}""", "musl", "byte_ops", [], 5),
    Benchmark("is_whitespace",
              "int is_whitespace(int c) { return c == ' ' || c == '\\t' || c == '\\n' || c == '\\r'; }",
              "musl", "byte_ops", [], 1),
    Benchmark("is_printable",
              "int is_printable(int c) { return c >= 32 && c < 127; }",
              "musl", "byte_ops", [], 1),
    Benchmark("ascii_case_fold",
              """int ascii_case_fold(int a, int b) {
    int la = (a >= 'A' && a <= 'Z') ? a + 32 : a;
    int lb = (b >= 'A' && b <= 'Z') ? b + 32 : b;
    return la - lb;
}""", "musl", "byte_ops", [], 4),
    Benchmark("digit_to_char",
              """int digit_to_char(int d) {
    if (d < 0 || d > 9) return '?';
    return '0' + d;
}""", "generic", "byte_ops", [], 3),
    Benchmark("nibble_to_hex",
              """int nibble_to_hex(int n) {
    n &= 0xF;
    return n < 10 ? '0' + n : 'a' + n - 10;
}""", "generic", "byte_ops", [], 3),
]

# ---------------------------------------------------------------------------
# Category 9: Hash and checksum (Redis/SQLite-inspired)
# ---------------------------------------------------------------------------

_HASH = [
    Benchmark("hash_int",
              """unsigned int hash_int(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}""", "redis", "hash", [], 5),
    Benchmark("hash_combine",
              """unsigned int hash_combine(unsigned int a, unsigned int b) {
    a ^= b + 0x9e3779b9 + (a << 6) + (a >> 2);
    return a;
}""", "generic", "hash", [], 2),
    Benchmark("djb2_byte",
              """unsigned int djb2_byte(unsigned int hash, int c) {
    return ((hash << 5) + hash) + (unsigned int)c;
}""", "generic", "hash", [], 1),
    Benchmark("fnv1a_byte",
              """unsigned int fnv1a_byte(unsigned int hash, int c) {
    hash ^= (unsigned int)c;
    hash *= 16777619u;
    return hash;
}""", "generic", "hash", [], 2),
    Benchmark("jenkins_one",
              """unsigned int jenkins_one(unsigned int hash, int key) {
    hash += (unsigned int)key;
    hash += hash << 10;
    hash ^= hash >> 6;
    return hash;
}""", "redis", "hash", [], 4),
    Benchmark("murmur_fmix",
              """unsigned int murmur_fmix(unsigned int h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}""", "generic", "hash", [], 6),
    Benchmark("xorshift32",
              """unsigned int xorshift32(unsigned int x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}""", "generic", "hash", [], 4),
    Benchmark("crc8_byte",
              """unsigned int crc8_byte(unsigned int crc, int data) {
    crc ^= (unsigned int)data;
    for (int i = 0; i < 8; i++) {
        if (crc & 1) crc = (crc >> 1) ^ 0x8C;
        else crc >>= 1;
    }
    return crc & 0xFF;
}""", "zlib", "hash", [], 7),
    Benchmark("checksum_fold",
              """unsigned int checksum_fold(unsigned int sum) {
    while (sum >> 16) sum = (sum & 0xFFFF) + (sum >> 16);
    return ~sum & 0xFFFF;
}""", "generic", "hash", [], 4),
    Benchmark("pearson_mix",
              """unsigned int pearson_mix(unsigned int h, int c) {
    h = (h ^ (unsigned int)c) * 0x01000193;
    return h;
}""", "generic", "hash", [], 2),
]

# ---------------------------------------------------------------------------
# Category 10: Alignment and rounding (Linux kernel-inspired)
# ---------------------------------------------------------------------------

_ALIGNMENT = [
    Benchmark("align_up",
              """unsigned int align_up(unsigned int x, unsigned int align) {
    if (align == 0) return x;
    return (x + align - 1) & ~(align - 1);
}""", "linux", "alignment", [], 3),
    Benchmark("align_down",
              """unsigned int align_down(unsigned int x, unsigned int align) {
    if (align == 0) return x;
    return x & ~(align - 1);
}""", "linux", "alignment", [], 3),
    Benchmark("is_aligned",
              """int is_aligned(unsigned int x, unsigned int align) {
    if (align == 0) return 1;
    return (x & (align - 1)) == 0;
}""", "linux", "alignment", [], 3),
    Benchmark("round_up_pow2",
              """unsigned int round_up_pow2(unsigned int x) {
    if (x == 0) return 0;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}""", "linux", "alignment", [], 9),
    Benchmark("round_down_pow2",
              """unsigned int round_down_pow2(unsigned int x) {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x - (x >> 1);
}""", "linux", "alignment", [], 7),
]

# ---------------------------------------------------------------------------
# Category 11: Constant-time operations (crypto-inspired)
# ---------------------------------------------------------------------------

_CONSTANT_TIME = [
    Benchmark("ct_select",
              """unsigned int ct_select(unsigned int a, unsigned int b, int flag) {
    unsigned int mask = (unsigned int)(-(flag != 0));
    return (a & mask) | (b & ~mask);
}""", "openssl", "constant_time", ["int_min_negation"], 3),
    Benchmark("ct_eq",
              """int ct_eq(unsigned int a, unsigned int b) {
    unsigned int diff = a ^ b;
    diff |= diff >> 16;
    diff |= diff >> 8;
    diff |= diff >> 4;
    diff |= diff >> 2;
    diff |= diff >> 1;
    return (int)(~diff & 1);
}""", "openssl", "constant_time", [], 7),
    Benchmark("ct_ge",
              """int ct_ge(unsigned int a, unsigned int b) {
    return (int)((1 - ((a - b) >> 31)) & 1);
}""", "openssl", "constant_time", [], 2),
    Benchmark("ct_is_zero",
              """int ct_is_zero(unsigned int x) {
    return (int)(1 & ((~x & (x - 1)) >> 31));
}""", "openssl", "constant_time", [], 2),
    Benchmark("ct_abs",
              """int ct_abs(int x) {
    int mask = x >> 31;
    return (x + mask) ^ mask;
}""", "openssl", "constant_time", ["int_min_negation"], 3),
    Benchmark("ct_min",
              """unsigned int ct_min(unsigned int a, unsigned int b) {
    unsigned int d = a - b;
    return b + (d & (d >> 31));
}""", "openssl", "constant_time", [], 2),
    Benchmark("ct_max",
              """unsigned int ct_max(unsigned int a, unsigned int b) {
    unsigned int d = a - b;
    return a - (d & (d >> 31));
}""", "openssl", "constant_time", [], 2),
    Benchmark("ct_clamp",
              """unsigned int ct_clamp(unsigned int x, unsigned int lo, unsigned int hi) {
    unsigned int d1 = x - lo;
    x = lo + (d1 & ~(d1 >> 31));
    unsigned int d2 = hi - x;
    x = hi - (d2 & ~(d2 >> 31));
    return x;
}""", "openssl", "constant_time", [], 5),
]

# ---------------------------------------------------------------------------
# Category 12: Saturating arithmetic (embedded / DSP-inspired)
# ---------------------------------------------------------------------------

_SATURATING = [
    Benchmark("sat_add_i32",
              """int sat_add_i32(int a, int b) {
    long long r = (long long)a + (long long)b;
    if (r > 2147483647) return 2147483647;
    if (r < (-2147483647-1)) return (-2147483647-1);
    return (int)r;
}""", "generic", "saturating", [], 5),
    Benchmark("sat_sub_i32",
              """int sat_sub_i32(int a, int b) {
    long long r = (long long)a - (long long)b;
    if (r > 2147483647) return 2147483647;
    if (r < (-2147483647-1)) return (-2147483647-1);
    return (int)r;
}""", "generic", "saturating", [], 5),
    Benchmark("sat_mul_i16",
              """int sat_mul_i16(int a, int b) {
    int r = a * b;
    if (r > 32767) return 32767;
    if (r < -32768) return -32768;
    return r;
}""", "generic", "saturating", ["signed_overflow"], 5),
    Benchmark("sat_add_u32",
              """unsigned int sat_add_u32(unsigned int a, unsigned int b) {
    unsigned int r = a + b;
    if (r < a) return 0xFFFFFFFF;
    return r;
}""", "generic", "saturating", [], 3),
    Benchmark("sat_sub_u32",
              """unsigned int sat_sub_u32(unsigned int a, unsigned int b) {
    if (b > a) return 0;
    return a - b;
}""", "generic", "saturating", [], 3),
]

# ---------------------------------------------------------------------------
# Category 13: Math utilities (musl/sqlite-inspired)
# ---------------------------------------------------------------------------

_MATH = [
    Benchmark("int_log2",
              """int int_log2(unsigned int x) {
    if (x == 0) return -1;
    int r = 0;
    while (x >>= 1) r++;
    return r;
}""", "musl", "math", [], 5),
    Benchmark("int_sqrt",
              """unsigned int int_sqrt(unsigned int n) {
    unsigned int x = n;
    unsigned int y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}""", "generic", "math", [], 6),
    Benchmark("ipow",
              """int ipow(int b, int e) {
    int r = 1;
    if (e < 0) return 0;
    while (e > 0) {
        if (e & 1) r *= b;
        b *= b;
        e >>= 1;
    }
    return r;
}""", "generic", "math", ["signed_overflow"], 8),
    Benchmark("lerp_int",
              """int lerp_int(int a, int b, int t, int tmax) {
    if (tmax == 0) return a;
    return a + ((b - a) * t) / tmax;
}""", "generic", "math", ["signed_overflow", "int_min_div_neg1"], 3),
    Benchmark("map_range",
              """int map_range(int x, int in_lo, int in_hi, int out_lo, int out_hi) {
    if (in_hi == in_lo) return out_lo;
    return out_lo + ((x - in_lo) * (out_hi - out_lo)) / (in_hi - in_lo);
}""", "generic", "math", ["signed_overflow", "int_min_div_neg1"], 3),
    Benchmark("distance_1d",
              """int distance_1d(int a, int b) {
    int d = a - b;
    return d < 0 ? -d : d;
}""", "generic", "math", ["signed_overflow", "int_min_negation"], 3),
    Benchmark("manhattan_dist",
              """int manhattan_dist(int x1, int y1, int x2, int y2) {
    int dx = x1 - x2;
    int dy = y1 - y2;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    return dx + dy;
}""", "generic", "math", ["signed_overflow", "int_min_negation"], 6),
    Benchmark("log10_int",
              """int log10_int(unsigned int x) {
    if (x == 0) return -1;
    int r = 0;
    while (x >= 10) { x /= 10; r++; }
    return r;
}""", "generic", "math", [], 5),
    Benchmark("is_prime_small",
              """int is_prime_small(int n) {
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}""", "generic", "math", ["signed_overflow"], 8),
    Benchmark("binomial_small",
              """int binomial_small(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    int r = 1;
    for (int i = 0; i < k; i++) {
        r = r * (n - i) / (i + 1);
    }
    return r;
}""", "generic", "math", ["signed_overflow"], 7),
]

# ---------------------------------------------------------------------------
# Category 14: Overflow-checked arithmetic (compiler builtins pattern)
# ---------------------------------------------------------------------------

_CHECKED = [
    Benchmark("checked_add",
              """int checked_add(int a, int b) {
    if (b > 0 && a > 2147483647 - b) return -1;
    if (b < 0 && a < (-2147483647-1) - b) return -1;
    return a + b;
}""", "generic", "checked", [], 4),
    Benchmark("checked_sub",
              """int checked_sub(int a, int b) {
    if (b < 0 && a > 2147483647 + b) return -1;
    if (b > 0 && a < (-2147483647-1) + b) return -1;
    return a - b;
}""", "generic", "checked", [], 4),
    Benchmark("checked_mul",
              """int checked_mul(int a, int b) {
    if (a == 0 || b == 0) return 0;
    int r = a * b;
    if (r / a != b) return -1;
    return r;
}""", "generic", "checked", ["signed_overflow"], 5),
    Benchmark("checked_neg",
              """int checked_neg(int x) {
    if (x == (-2147483647-1)) return -1;
    return -x;
}""", "generic", "checked", [], 3),
    Benchmark("wrapping_add",
              "int wrapping_add(int a, int b) { return (int)((unsigned int)a + (unsigned int)b); }",
              "generic", "checked", [], 1),
    Benchmark("wrapping_sub",
              "int wrapping_sub(int a, int b) { return (int)((unsigned int)a - (unsigned int)b); }",
              "generic", "checked", [], 1),
    Benchmark("wrapping_mul",
              "int wrapping_mul(int a, int b) { return (int)((unsigned int)a * (unsigned int)b); }",
              "generic", "checked", [], 1),
    Benchmark("overflow_detect_add",
              """int overflow_detect_add(int a, int b) {
    unsigned int ua = (unsigned int)a, ub = (unsigned int)b;
    unsigned int ur = ua + ub;
    int r = (int)ur;
    return (a > 0 && b > 0 && r < 0) || (a < 0 && b < 0 && r >= 0);
}""", "generic", "checked", [], 4),
]

# ---------------------------------------------------------------------------
# Category 15: Encoding / decoding (SQLite-inspired)
# ---------------------------------------------------------------------------

_ENCODING = [
    Benchmark("encode_zigzag",
              "unsigned int encode_zigzag(int n) { return (unsigned int)((n << 1) ^ (n >> 31)); }",
              "sqlite", "encoding", [], 1),
    Benchmark("decode_zigzag",
              "int decode_zigzag(unsigned int n) { return (int)((n >> 1) ^ -(int)(n & 1)); }",
              "sqlite", "encoding", ["int_min_negation"], 1),
    Benchmark("encode_varint_size",
              """int encode_varint_size(unsigned int x) {
    if (x < 128) return 1;
    if (x < 16384) return 2;
    if (x < 2097152) return 3;
    if (x < 268435456) return 4;
    return 5;
}""", "sqlite", "encoding", [], 5),
    Benchmark("pack_rgb",
              """unsigned int pack_rgb(int r, int g, int b) {
    return ((unsigned int)(r & 0xFF) << 16) | ((unsigned int)(g & 0xFF) << 8) | (unsigned int)(b & 0xFF);
}""", "generic", "encoding", [], 2),
    Benchmark("unpack_r",
              "int unpack_r(unsigned int rgb) { return (int)((rgb >> 16) & 0xFF); }",
              "generic", "encoding", [], 1),
    Benchmark("unpack_g",
              "int unpack_g(unsigned int rgb) { return (int)((rgb >> 8) & 0xFF); }",
              "generic", "encoding", [], 1),
    Benchmark("unpack_b",
              "int unpack_b(unsigned int rgb) { return (int)(rgb & 0xFF); }",
              "generic", "encoding", [], 1),
    Benchmark("gray_encode",
              "unsigned int gray_encode(unsigned int n) { return n ^ (n >> 1); }",
              "generic", "encoding", [], 1),
    Benchmark("gray_decode",
              """unsigned int gray_decode(unsigned int g) {
    unsigned int n = g;
    while (g >>= 1) n ^= g;
    return n;
}""", "generic", "encoding", [], 4),
]

# ---------------------------------------------------------------------------
# Category 16: Multi-operation compositions (real migration patterns)
# ---------------------------------------------------------------------------

_COMPOSITIONS = [
    Benchmark("clamp_add",
              """int clamp_add(int a, int b, int lo, int hi) {
    int r = a + b;
    if (r < lo) return lo;
    if (r > hi) return hi;
    return r;
}""", "generic", "composition", ["signed_overflow"], 5),
    Benchmark("scale_and_clamp",
              """int scale_and_clamp(int x, int scale, int lo, int hi) {
    int r = x * scale;
    if (r < lo) return lo;
    if (r > hi) return hi;
    return r;
}""", "generic", "composition", ["signed_overflow"], 5),
    Benchmark("weighted_avg",
              """int weighted_avg(int a, int wa, int b, int wb) {
    if (wa + wb == 0) return 0;
    return (a * wa + b * wb) / (wa + wb);
}""", "generic", "composition", ["signed_overflow", "int_min_div_neg1"], 3),
    Benchmark("bilinear_1d",
              """int bilinear_1d(int a, int b, int t, int n) {
    if (n == 0) return a;
    return a + (b - a) * t / n;
}""", "generic", "composition", ["signed_overflow", "int_min_div_neg1"], 3),
    Benchmark("safe_array_idx",
              """int safe_array_idx(int base, int idx, int stride, int len) {
    if (idx < 0 || idx >= len) return -1;
    return base + idx * stride;
}""", "generic", "composition", ["signed_overflow"], 4),
    Benchmark("ring_buffer_idx",
              """int ring_buffer_idx(int head, int offset, int size) {
    if (size <= 0) return -1;
    int idx = (head + offset) % size;
    if (idx < 0) idx += size;
    return idx;
}""", "generic", "composition", ["signed_overflow", "int_min_div_neg1"], 5),
    Benchmark("bit_field_insert",
              """unsigned int bit_field_insert(unsigned int x, unsigned int val, int pos, int width) {
    if (pos < 0 || width <= 0 || pos + width > 32) return x;
    unsigned int mask = ((1u << width) - 1) << pos;
    return (x & ~mask) | ((val << pos) & mask);
}""", "generic", "composition", ["shift_ub"], 4),
    Benchmark("byte_pack_2",
              """unsigned int byte_pack_2(int hi, int lo) {
    return ((unsigned int)(hi & 0xFF) << 8) | (unsigned int)(lo & 0xFF);
}""", "generic", "composition", [], 1),
    Benchmark("minmax_update",
              """int minmax_update(int cur_min, int cur_max, int val) {
    if (val < cur_min) return val;
    if (val > cur_max) return -val;
    return 0;
}""", "generic", "composition", ["int_min_negation"], 4),
    Benchmark("round_to_multiple",
              """int round_to_multiple(int x, int m) {
    if (m == 0) return x;
    int r = x % m;
    if (r == 0) return x;
    if (r > 0) return x + m - r;
    return x - r;
}""", "generic", "composition", ["signed_overflow", "int_min_div_neg1"], 6),
    Benchmark("encode_pair",
              """unsigned int encode_pair(int x, int y) {
    unsigned int ux = (unsigned int)((x << 1) ^ (x >> 31));
    unsigned int uy = (unsigned int)((y << 1) ^ (y >> 31));
    return (ux << 16) | (uy & 0xFFFF);
}""", "generic", "composition", [], 3),
    Benchmark("smooth_step",
              """int smooth_step(int x, int edge0, int edge1) {
    if (edge1 == edge0) return x >= edge0 ? 100 : 0;
    int t = (x - edge0) * 100 / (edge1 - edge0);
    if (t < 0) return 0;
    if (t > 100) return 100;
    return t;
}""", "generic", "composition", ["signed_overflow", "int_min_div_neg1"], 6),
    Benchmark("hash_pair",
              """unsigned int hash_pair(unsigned int a, unsigned int b) {
    a ^= b + 0x9e3779b9 + (a << 6) + (a >> 2);
    b ^= a + 0x9e3779b9 + (b << 6) + (b >> 2);
    return a ^ b;
}""", "generic", "composition", [], 3),
    Benchmark("mix3",
              """unsigned int mix3(unsigned int a, unsigned int b, unsigned int c) {
    a -= b; a -= c; a ^= (c >> 13);
    b -= c; b -= a; b ^= (a << 8);
    c -= a; c -= b; c ^= (b >> 13);
    return c;
}""", "generic", "composition", [], 6),
    Benchmark("interpolate_color",
              """unsigned int interpolate_color(unsigned int c1, unsigned int c2, int t) {
    int r1 = (c1 >> 16) & 0xFF, g1 = (c1 >> 8) & 0xFF, b1 = c1 & 0xFF;
    int r2 = (c2 >> 16) & 0xFF, g2 = (c2 >> 8) & 0xFF, b2 = c2 & 0xFF;
    int r = r1 + (r2 - r1) * t / 255;
    int g = g1 + (g2 - g1) * t / 255;
    int b = b1 + (b2 - b1) * t / 255;
    return ((unsigned int)r << 16) | ((unsigned int)g << 8) | (unsigned int)b;
}""", "generic", "composition", ["signed_overflow"], 8),
]

# ---------------------------------------------------------------------------
# Category 17: Additional overflow patterns
# ---------------------------------------------------------------------------

_EXTRA_OVERFLOW = [
    Benchmark("mul_add", "int mul_add(int a, int b, int c) { return a * b + c; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("dot2", "int dot2(int a, int b, int c, int d) { return a * b + c * d; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("sqr_diff",
              """int sqr_diff(int a, int b) {
    return (a - b) * (a - b);
}""", "generic", "overflow", ["signed_overflow"], 2),
    Benchmark("norm_l1",
              """int norm_l1(int x, int y) {
    int ax = x < 0 ? -x : x;
    int ay = y < 0 ? -y : y;
    return ax + ay;
}""", "generic", "overflow", ["int_min_negation", "signed_overflow"], 4),
    Benchmark("cross2d",
              "int cross2d(int ax, int ay, int bx, int by) { return ax * by - ay * bx; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("area_rect",
              "int area_rect(int w, int h) { return w * h; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("perimeter_rect",
              "int perimeter_rect(int w, int h) { return 2 * (w + h); }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("celsius_to_fahr",
              "int celsius_to_fahr(int c) { return c * 9 / 5 + 32; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("fahr_to_celsius",
              "int fahr_to_celsius(int f) { return (f - 32) * 5 / 9; }",
              "generic", "overflow", ["signed_overflow"], 1),
    Benchmark("scale_percent",
              "int scale_percent(int val, int pct) { return val * pct / 100; }",
              "generic", "overflow", ["signed_overflow"], 1),
]

# ---------------------------------------------------------------------------
# Category 18: Additional control flow
# ---------------------------------------------------------------------------

_EXTRA_CF = [
    Benchmark("abs_max",
              """int abs_max(int a, int b) {
    int aa = a < 0 ? -a : a;
    int ab = b < 0 ? -b : b;
    return aa > ab ? aa : ab;
}""", "generic", "control_flow", ["int_min_negation"], 4),
    Benchmark("copysign_int",
              """int copysign_int(int mag, int sgn) {
    int am = mag < 0 ? -mag : mag;
    return sgn < 0 ? -am : am;
}""", "generic", "control_flow", ["int_min_negation"], 3),
    Benchmark("select3",
              """int select3(int sel, int a, int b, int c) {
    if (sel == 0) return a;
    if (sel == 1) return b;
    return c;
}""", "generic", "control_flow", [], 4),
    Benchmark("bounded_inc",
              """int bounded_inc(int x, int bound) {
    if (x >= bound) return bound;
    return x + 1;
}""", "generic", "control_flow", [], 3),
    Benchmark("bounded_dec",
              """int bounded_dec(int x, int bound) {
    if (x <= bound) return bound;
    return x - 1;
}""", "generic", "control_flow", [], 3),
    Benchmark("wrap_range",
              """int wrap_range(int x, int lo, int hi) {
    if (hi <= lo) return lo;
    int range = hi - lo;
    int r = (x - lo) % range;
    if (r < 0) r += range;
    return lo + r;
}""", "generic", "control_flow", ["signed_overflow", "int_min_div_neg1"], 7),
    Benchmark("step_func",
              """int step_func(int x, int t1, int t2) {
    if (x < t1) return 0;
    if (x < t2) return 1;
    return 2;
}""", "generic", "control_flow", [], 4),
    Benchmark("compare_3way",
              """int compare_3way(int a, int b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}""", "generic", "control_flow", [], 4),
]

# ---------------------------------------------------------------------------
# Category 19: Additional loop patterns
# ---------------------------------------------------------------------------

_EXTRA_LOOPS = [
    Benchmark("sum_of_squares",
              """int sum_of_squares(int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) s += i * i;
    return s;
}""", "generic", "loop", ["signed_overflow"], 5),
    Benchmark("harmonic_int",
              """int harmonic_int(int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) s += 1000 / i;
    return s;
}""", "generic", "loop", [], 5),
    Benchmark("count_set_bits_range",
              """int count_set_bits_range(unsigned int lo, unsigned int hi) {
    int c = 0;
    for (unsigned int i = lo; i <= hi && i >= lo; i++) {
        unsigned int x = i;
        while (x) { c += x & 1; x >>= 1; }
    }
    return c;
}""", "generic", "loop", ["signed_overflow"], 8),
    Benchmark("gcd_extended_sign",
              """int gcd_extended_sign(int a, int b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}""", "generic", "loop", ["int_min_negation"], 6),
    Benchmark("popcount_loop",
              """int popcount_loop(unsigned int x) {
    int c = 0;
    for (int i = 0; i < 32; i++) {
        c += (x >> i) & 1;
    }
    return c;
}""", "generic", "loop", [], 5),
    Benchmark("find_first_diff_bit",
              """int find_first_diff_bit(unsigned int a, unsigned int b) {
    unsigned int d = a ^ b;
    if (d == 0) return -1;
    int pos = 0;
    while ((d & 1) == 0) { d >>= 1; pos++; }
    return pos;
}""", "generic", "loop", [], 7),
    Benchmark("count_leading_ones",
              """int count_leading_ones(unsigned int x) {
    int c = 0;
    unsigned int mask = 0x80000000;
    while (mask && (x & mask)) { c++; mask >>= 1; }
    return c;
}""", "generic", "loop", [], 5),
    Benchmark("binary_search_approx",
              """int binary_search_approx(int target) {
    int lo = 0, hi = 1000;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (mid * mid <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}""", "generic", "loop", ["signed_overflow"], 8),
    Benchmark("exp_by_squaring",
              """unsigned int exp_by_squaring(unsigned int base, int exp) {
    unsigned int r = 1;
    if (exp < 0) return 0;
    while (exp > 0) {
        if (exp & 1) r *= base;
        base *= base;
        exp >>= 1;
    }
    return r;
}""", "generic", "loop", [], 8),
]

# ---------------------------------------------------------------------------
# Category 20: Edge case patterns (INT_MIN, boundaries)
# ---------------------------------------------------------------------------

_EDGE_CASES = [
    Benchmark("safe_abs",
              """int safe_abs(int x) {
    if (x == (-2147483647-1)) return 2147483647;
    return x < 0 ? -x : x;
}""", "generic", "edge_case", [], 3),
    Benchmark("saturate_to_i16",
              """int saturate_to_i16(int x) {
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return x;
}""", "generic", "edge_case", [], 4),
    Benchmark("safe_increment",
              """int safe_increment(int x) {
    if (x == 2147483647) return 2147483647;
    return x + 1;
}""", "generic", "edge_case", [], 3),
    Benchmark("safe_decrement",
              """int safe_decrement(int x) {
    if (x == (-2147483647-1)) return (-2147483647-1);
    return x - 1;
}""", "generic", "edge_case", [], 3),
    Benchmark("safe_double",
              """int safe_double(int x) {
    if (x > 1073741823 || x < -1073741824) return x;
    return x * 2;
}""", "generic", "edge_case", [], 3),
    Benchmark("clamp_i8",
              """int clamp_i8(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return x;
}""", "generic", "edge_case", [], 4),
    Benchmark("div_or_zero",
              """int div_or_zero(int a, int b) {
    if (b == 0) return 0;
    if (a == (-2147483647-1) && b == -1) return 2147483647;
    return a / b;
}""", "generic", "edge_case", [], 4),
    Benchmark("mod_or_zero",
              """int mod_or_zero(int a, int b) {
    if (b == 0) return 0;
    if (a == (-2147483647-1) && b == -1) return 0;
    return a % b;
}""", "generic", "edge_case", [], 4),
    Benchmark("sign_extend_or_zero",
              """int sign_extend_or_zero(int x, int bits) {
    if (bits <= 0 || bits > 31) return 0;
    int mask = 1 << (bits - 1);
    return (x ^ mask) - mask;
}""", "generic", "edge_case", ["signed_overflow"], 4),
    Benchmark("rotate_safe",
              """unsigned int rotate_safe(unsigned int x, int n) {
    n &= 31;
    if (n == 0) return x;
    return (x << n) | (x >> (32 - n));
}""", "generic", "edge_case", [], 4),
    Benchmark("mul_high_approx",
              """int mul_high_approx(int a, int b) {
    long long r = (long long)a * (long long)b;
    return (int)(r >> 32);
}""", "generic", "edge_case", [], 2),
    Benchmark("add_with_carry",
              """unsigned int add_with_carry(unsigned int a, unsigned int b, int carry) {
    unsigned int r = a + b + (carry ? 1u : 0u);
    return r;
}""", "generic", "edge_case", [], 2),
    Benchmark("sub_with_borrow",
              """unsigned int sub_with_borrow(unsigned int a, unsigned int b, int borrow) {
    return a - b - (borrow ? 1u : 0u);
}""", "generic", "edge_case", [], 2),
    Benchmark("count_trailing_ones",
              """int count_trailing_ones(unsigned int x) {
    int c = 0;
    while (x & 1) { c++; x >>= 1; }
    return c;
}""", "generic", "edge_case", [], 4),
    Benchmark("is_negative_zero",
              """int is_negative_zero(int x) {
    return x == 0 && (1 / (x + 1) < 0);
}""", "generic", "edge_case", [], 2),
    Benchmark("blend_byte",
              """int blend_byte(int a, int b, int alpha) {
    return (a * (255 - alpha) + b * alpha) / 255;
}""", "generic", "edge_case", ["signed_overflow"], 1),
    Benchmark("triple_xor",
              """unsigned int triple_xor(unsigned int a, unsigned int b, unsigned int c) {
    return a ^ b ^ c;
}""", "generic", "edge_case", [], 1),
]

# ---------------------------------------------------------------------------
# Assemble full suite
# ---------------------------------------------------------------------------

ALL_BENCHMARKS: List[Benchmark] = (
    _ARITH_OVERFLOW + _DIVISION + _BITWISE + _SHIFTS + _CASTS +
    _CONTROL_FLOW + _LOOPS + _HASH + _ALIGNMENT + _CONSTANT_TIME +
    _SATURATING + _MATH + _CHECKED + _ENCODING + _BYTE_OPS +
    _COMPOSITIONS + _EXTRA_OVERFLOW + _EXTRA_CF + _EXTRA_LOOPS +
    _EDGE_CASES
)

BENCHMARK_BY_NAME: Dict[str, Benchmark] = {b.name: b for b in ALL_BENCHMARKS}

CATEGORIES = sorted(set(b.category for b in ALL_BENCHMARKS))

def get_benchmarks_by_category(cat: str) -> List[Benchmark]:
    return [b for b in ALL_BENCHMARKS if b.category == cat]

def get_benchmarks_by_source(src: str) -> List[Benchmark]:
    return [b for b in ALL_BENCHMARKS if b.source == src]

def suite_summary() -> Dict[str, int]:
    by_cat = {}
    for b in ALL_BENCHMARKS:
        by_cat[b.category] = by_cat.get(b.category, 0) + 1
    return {"total": len(ALL_BENCHMARKS), "by_category": by_cat,
            "sources": list(set(b.source for b in ALL_BENCHMARKS))}


if __name__ == "__main__":
    import json
    s = suite_summary()
    print(f"Total benchmarks: {s['total']}")
    print(f"Categories: {json.dumps(s['by_category'], indent=2)}")
    print(f"Sources: {s['sources']}")
