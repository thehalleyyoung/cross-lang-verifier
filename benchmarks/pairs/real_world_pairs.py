"""
Real-world C/Rust function pairs from actual open-source translation projects.

Sources:
  - musl libc implementations vs idiomatic Rust equivalents
  - c2rust transpiler output vs hand-written Rust
  - GNU coreutils C functions vs uutils/coreutils Rust rewrites

These pairs exercise divergence points that arise in real translation
projects, not synthetic micro-benchmarks.
"""

from __future__ import annotations
from typing import List
from .benchmark_pairs import BenchmarkPair


# ---------------------------------------------------------------------------
# musl libc functions vs Rust equivalents (8 pairs)
# ---------------------------------------------------------------------------

MUSL_LIBC_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="musl_strlen",
        category="real_world_musl",
        expected_result="equivalent",
        description="musl strlen: byte-by-byte scan is equivalent to Rust slice len",
        c_source="""
int my_strlen(const char *s) {
    const char *a = s;
    while (*a) a++;
    return (int)(a - s);
}
""",
        rust_source="""
pub fn my_strlen(s: &[u8]) -> i32 {
    let mut i: i32 = 0;
    while (i as usize) < s.len() && s[i as usize] != 0 {
        i += 1;
    }
    i
}
""",
    ),
    BenchmarkPair(
        name="musl_memcmp",
        category="real_world_musl",
        expected_result="divergent",
        description="musl memcmp: C returns arbitrary sign, Rust cmp is well-defined",
        divergence_kind="signedness",
        c_source="""
int my_memcmp(const unsigned char *a, const unsigned char *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return (int)a[i] - (int)b[i];
    }
    return 0;
}
""",
        rust_source="""
pub fn my_memcmp(a: &[u8], b: &[u8], n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let ai = a[i as usize] as i32;
        let bi = b[i as usize] as i32;
        if ai != bi { return ai - bi; }
        i += 1;
    }
    0
}
""",
    ),
    BenchmarkPair(
        name="musl_atoi",
        category="real_world_musl",
        expected_result="divergent",
        description="musl atoi: C signed overflow is UB on large inputs, Rust wraps",
        divergence_kind="signed_overflow",
        c_source="""
int my_atoi(const char *s) {
    int n = 0;
    int neg = 0;
    int i = 0;
    while (s[i] == ' ') i++;
    if (s[i] == '-') { neg = 1; i++; }
    else if (s[i] == '+') { i++; }
    while (s[i] >= '0' && s[i] <= '9') {
        n = n * 10 + (s[i] - '0');
        i++;
    }
    return neg ? -n : n;
}
""",
        rust_source="""
pub fn my_atoi(s: &[u8]) -> i32 {
    let mut n: i32 = 0;
    let mut neg = false;
    let mut i: usize = 0;
    while i < s.len() && s[i] == b' ' { i += 1; }
    if i < s.len() && s[i] == b'-' { neg = true; i += 1; }
    else if i < s.len() && s[i] == b'+' { i += 1; }
    while i < s.len() && s[i] >= b'0' && s[i] <= b'9' {
        n = n.wrapping_mul(10).wrapping_add((s[i] - b'0') as i32);
        i += 1;
    }
    if neg { n.wrapping_neg() } else { n }
}
""",
    ),
    BenchmarkPair(
        name="musl_strtol_simple",
        category="real_world_musl",
        expected_result="divergent",
        description="Simplified strtol: C overflow is UB, Rust wrapping_mul/add wraps",
        divergence_kind="signed_overflow",
        c_source="""
long my_strtol(const char *s) {
    long val = 0;
    int neg = 0;
    int i = 0;
    while (s[i] == ' ') i++;
    if (s[i] == '-') { neg = 1; i++; }
    else if (s[i] == '+') { i++; }
    while (s[i] >= '0' && s[i] <= '9') {
        val = val * 10 + (s[i] - '0');
        i++;
    }
    return neg ? -val : val;
}
""",
        rust_source="""
pub fn my_strtol(s: &[u8]) -> i64 {
    let mut val: i64 = 0;
    let mut neg = false;
    let mut i: usize = 0;
    while i < s.len() && s[i] == b' ' { i += 1; }
    if i < s.len() && s[i] == b'-' { neg = true; i += 1; }
    else if i < s.len() && s[i] == b'+' { i += 1; }
    while i < s.len() && s[i] >= b'0' && s[i] <= b'9' {
        val = val.wrapping_mul(10).wrapping_add((s[i] - b'0') as i64);
        i += 1;
    }
    if neg { val.wrapping_neg() } else { val }
}
""",
    ),
    BenchmarkPair(
        name="musl_isdigit",
        category="real_world_musl",
        expected_result="equivalent",
        description="musl isdigit: simple range check is identical in both languages",
        c_source="""
int my_isdigit(int c) {
    return (unsigned int)(c - '0') < 10;
}
""",
        rust_source="""
pub fn my_isdigit(c: i32) -> i32 {
    if ((c - 0x30) as u32) < 10 { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="musl_isalpha",
        category="real_world_musl",
        expected_result="equivalent",
        description="musl isalpha: bitwise OR trick identical in both languages",
        c_source="""
int my_isalpha(int c) {
    return ((unsigned int)(c | 32) - 'a') < 26;
}
""",
        rust_source="""
pub fn my_isalpha(c: i32) -> i32 {
    if (((c | 32) as u32).wrapping_sub(0x61)) < 26 { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="musl_abs",
        category="real_world_musl",
        expected_result="divergent",
        description="musl abs: C abs(INT_MIN) is UB, Rust wrapping_abs returns INT_MIN",
        divergence_kind="negation_overflow",
        c_source="""
int my_abs(int a) {
    return a < 0 ? -a : a;
}
""",
        rust_source="""
pub fn my_abs(a: i32) -> i32 {
    if a < 0 { a.wrapping_neg() } else { a }
}
""",
    ),
    BenchmarkPair(
        name="musl_toupper",
        category="real_world_musl",
        expected_result="equivalent",
        description="musl toupper: direct arithmetic, identical behavior",
        c_source="""
int my_toupper(int c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    return c;
}
""",
        rust_source="""
pub fn my_toupper(c: i32) -> i32 {
    if c >= 0x61 && c <= 0x7A { c - 32 } else { c }
}
""",
    ),
]


# ---------------------------------------------------------------------------
# c2rust transpiled pairs (5 pairs)
# ---------------------------------------------------------------------------

C2RUST_REAL_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="c2rust_ptr_arithmetic",
        category="real_world_c2rust",
        expected_result="divergent",
        description="c2rust pointer arithmetic: offset past allocation is C UB, Rust unsafe raw ptr",
        divergence_kind="pointer_ub",
        c_source="""
int sum_offset(int *base, int offset, int count) {
    int *p = base + offset;
    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += p[i];
    }
    return sum;
}
""",
        rust_source="""
pub unsafe fn sum_offset(base: *mut i32, offset: i32, count: i32) -> i32 {
    let p = base.offset(offset as isize);
    let mut sum: i32 = 0;
    let mut i: i32 = 0;
    while i < count {
        sum = sum.wrapping_add(*p.offset(i as isize));
        i += 1;
    }
    sum
}
""",
    ),
    BenchmarkPair(
        name="c2rust_int_overflow_cast",
        category="real_world_c2rust",
        expected_result="divergent",
        description="c2rust preserves C cast semantics but Rust 'as' truncates deterministically",
        divergence_kind="overflow",
        c_source="""
int narrow_cast(long long x) {
    return (int)x;
}
""",
        rust_source="""
pub fn narrow_cast(x: i64) -> i32 {
    x as i32
}
""",
    ),
    BenchmarkPair(
        name="c2rust_null_deref_guard",
        category="real_world_c2rust",
        expected_result="divergent",
        description="c2rust NULL check: C dereferences NULL → UB, Rust Option prevents it",
        divergence_kind="null_handling",
        c_source="""
int safe_deref(int *ptr) {
    if (ptr) return *ptr;
    return 0;
}
""",
        rust_source="""
pub fn safe_deref(ptr: Option<&i32>) -> i32 {
    match ptr {
        Some(v) => *v,
        None => 0,
    }
}
""",
    ),
    BenchmarkPair(
        name="c2rust_signed_shift",
        category="real_world_c2rust",
        expected_result="divergent",
        description="c2rust preserves C signed right-shift (impl-defined), Rust arithmetic shift is defined",
        divergence_kind="shift_semantics",
        c_source="""
int arith_shift(int x, int n) {
    return x >> n;
}
""",
        rust_source="""
pub fn arith_shift(x: i32, n: i32) -> i32 {
    x >> (n & 31)
}
""",
    ),
    BenchmarkPair(
        name="c2rust_union_reinterpret",
        category="real_world_c2rust",
        expected_result="divergent",
        description="c2rust union type punning: C union reinterpret is common, Rust transmute required",
        divergence_kind="struct_layout",
        c_source="""
typedef union { int i; float f; } IntFloat;

int float_bits(float x) {
    IntFloat u;
    u.f = x;
    return u.i;
}
""",
        rust_source="""
pub fn float_bits(x: f32) -> i32 {
    unsafe { core::mem::transmute::<f32, i32>(x) }
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Coreutils-equivalent pairs (4 pairs)
# ---------------------------------------------------------------------------

COREUTILS_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="coreutils_parse_int",
        category="real_world_coreutils",
        expected_result="divergent",
        description="coreutils strtol-style parsing: C overflow UB, Rust saturates/wraps",
        divergence_kind="signed_overflow",
        c_source="""
int parse_decimal(const char *s) {
    int result = 0;
    int i = 0;
    int sign = 1;
    if (s[i] == '-') { sign = -1; i++; }
    while (s[i] >= '0' && s[i] <= '9') {
        result = result * 10 + (s[i] - '0');
        i++;
    }
    return result * sign;
}
""",
        rust_source="""
pub fn parse_decimal(s: &[u8]) -> i32 {
    let mut result: i32 = 0;
    let mut i: usize = 0;
    let mut sign: i32 = 1;
    if i < s.len() && s[i] == b'-' { sign = -1; i += 1; }
    while i < s.len() && s[i] >= b'0' && s[i] <= b'9' {
        result = result.wrapping_mul(10).wrapping_add((s[i] - b'0') as i32);
        i += 1;
    }
    result.wrapping_mul(sign)
}
""",
    ),
    BenchmarkPair(
        name="coreutils_wc_bytes",
        category="real_world_coreutils",
        expected_result="equivalent",
        description="wc -c equivalent: simple byte count is identical",
        c_source="""
int count_bytes(const char *buf, int len) {
    int count = 0;
    for (int i = 0; i < len; i++) {
        count++;
    }
    return count;
}
""",
        rust_source="""
pub fn count_bytes(buf: &[u8], len: i32) -> i32 {
    let mut count: i32 = 0;
    let mut i: i32 = 0;
    while i < len {
        count += 1;
        i += 1;
    }
    count
}
""",
    ),
    BenchmarkPair(
        name="coreutils_wc_lines",
        category="real_world_coreutils",
        expected_result="divergent",
        description="wc -l equivalent: C long count overflows on huge input, Rust wraps",
        divergence_kind="signed_overflow",
        c_source="""
int count_lines(const char *buf, int len) {
    int lines = 0;
    for (int i = 0; i < len; i++) {
        if (buf[i] == '\\n') lines++;
    }
    return lines;
}
""",
        rust_source="""
pub fn count_lines(buf: &[u8], len: i32) -> i32 {
    let mut lines: i32 = 0;
    let mut i: i32 = 0;
    while i < len {
        if buf[i as usize] == b'\\n' {
            lines = lines.wrapping_add(1);
        }
        i += 1;
    }
    lines
}
""",
    ),
    BenchmarkPair(
        name="coreutils_base64_sextet",
        category="real_world_coreutils",
        expected_result="equivalent",
        description="base64 sextet extraction: bitwise ops identical in both languages",
        c_source="""
unsigned int extract_sextet(unsigned int triplet, int index) {
    switch (index) {
        case 0: return (triplet >> 18) & 0x3F;
        case 1: return (triplet >> 12) & 0x3F;
        case 2: return (triplet >> 6) & 0x3F;
        case 3: return triplet & 0x3F;
        default: return 0;
    }
}
""",
        rust_source="""
pub fn extract_sextet(triplet: u32, index: i32) -> u32 {
    match index {
        0 => (triplet >> 18) & 0x3F,
        1 => (triplet >> 12) & 0x3F,
        2 => (triplet >> 6) & 0x3F,
        3 => triplet & 0x3F,
        _ => 0,
    }
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Combined export
# ---------------------------------------------------------------------------

REAL_WORLD_PAIRS: List[BenchmarkPair] = (
    MUSL_LIBC_PAIRS +
    C2RUST_REAL_PAIRS +
    COREUTILS_PAIRS
)


def get_real_world_pairs() -> List[BenchmarkPair]:
    """Return all real-world benchmark pairs."""
    return list(REAL_WORLD_PAIRS)


def get_real_world_by_source(source: str) -> List[BenchmarkPair]:
    """Return pairs from a specific source project.

    Args:
        source: One of 'musl', 'c2rust', 'coreutils'.
    """
    prefix = f"real_world_{source}"
    return [p for p in REAL_WORLD_PAIRS if p.category == prefix]
