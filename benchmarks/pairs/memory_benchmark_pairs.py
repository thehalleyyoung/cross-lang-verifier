"""
Memory/pointer benchmark pairs for cross-language equivalence verification.

Covers pointer arithmetic, array access, struct field access, heap allocation,
and C2Rust-style pointer patterns. These exercise the QF_ABV memory model.
"""

from __future__ import annotations
from typing import List
from .benchmark_pairs import BenchmarkPair


MEMORY_PAIRS: List[BenchmarkPair] = [
    # -----------------------------------------------------------------------
    # Array access patterns
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="array_sum",
        category="memory",
        expected_result="equivalent",
        description="Sum array elements via pointer arithmetic (C) vs indexing (Rust)",
        c_source="""
int array_sum(int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n && i < 4; i++) {
        sum += arr[i];
    }
    return sum;
}
""",
        rust_source="""
pub fn array_sum(arr: &[i32], n: i32) -> i32 {
    let mut sum: i32 = 0;
    let mut i: i32 = 0;
    while i < n && i < 4 {
        sum = sum.wrapping_add(arr[i as usize]);
        i += 1;
    }
    sum
}
""",
    ),
    BenchmarkPair(
        name="array_write_read",
        category="memory",
        expected_result="divergent",
        description="Write to array then read back — C uses UB addition, Rust wraps",
        divergence_kind="signed_overflow",
        c_source="""
int array_write_read(int val) {
    int buf[4];
    buf[0] = val;
    buf[1] = val + 1;
    return buf[0] + buf[1];
}
""",
        rust_source="""
pub fn array_write_read(val: i32) -> i32 {
    let mut buf: [i32; 4] = [0; 4];
    buf[0] = val;
    buf[1] = val.wrapping_add(1);
    buf[0].wrapping_add(buf[1])
}
""",
    ),
    BenchmarkPair(
        name="array_swap",
        category="memory",
        expected_result="divergent",
        description="Swap two elements — C addition is UB, Rust wraps",
        divergence_kind="signed_overflow",
        c_source="""
int array_swap(int a, int b) {
    int arr[2];
    arr[0] = a;
    arr[1] = b;
    int tmp = arr[0];
    arr[0] = arr[1];
    arr[1] = tmp;
    return arr[0] + arr[1];
}
""",
        rust_source="""
pub fn array_swap(a: i32, b: i32) -> i32 {
    let mut arr: [i32; 2] = [a, b];
    arr.swap(0, 1);
    arr[0].wrapping_add(arr[1])
}
""",
    ),

    # -----------------------------------------------------------------------
    # Pointer arithmetic divergences
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="ptr_offset_ub",
        category="memory",
        expected_result="divergent",
        description="Out-of-bounds pointer arithmetic: C UB, Rust wrapping_offset safe",
        divergence_kind="pointer_ub",
        c_source="""
int ptr_offset(int *p, int idx) {
    return *(p + idx);
}
""",
        rust_source="""
pub fn ptr_offset(p: &[i32], idx: i32) -> i32 {
    if (idx as usize) < p.len() {
        p[idx as usize]
    } else {
        0
    }
}
""",
    ),
    BenchmarkPair(
        name="null_deref_guard",
        category="memory",
        expected_result="divergent",
        description="C allows null pointer (UB on deref), Rust Option guards",
        divergence_kind="null_pointer",
        c_source="""
int deref_or_default(int *p) {
    if (p) return *p;
    return 0;
}
""",
        rust_source="""
pub fn deref_or_default(p: Option<&i32>) -> i32 {
    match p {
        Some(val) => *val,
        None => 0,
    }
}
""",
    ),

    # -----------------------------------------------------------------------
    # Struct field access
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="struct_field_access",
        category="memory",
        expected_result="divergent",
        description="Sum struct fields — C addition is UB, Rust wraps",
        divergence_kind="signed_overflow",
        c_source="""
typedef struct { int x; int y; } Point;
int sum_point(Point p) {
    return p.x + p.y;
}
""",
        rust_source="""
pub fn sum_point(x: i32, y: i32) -> i32 {
    x.wrapping_add(y)
}
""",
    ),
    BenchmarkPair(
        name="struct_modify",
        category="memory",
        expected_result="divergent",
        description="Modify struct fields — C addition is UB, Rust wraps",
        divergence_kind="signed_overflow",
        c_source="""
int struct_modify(int a, int b) {
    int buf[2];
    buf[0] = a;
    buf[1] = b;
    buf[0] = buf[0] + 1;
    return buf[0] + buf[1];
}
""",
        rust_source="""
pub fn struct_modify(a: i32, b: i32) -> i32 {
    let x = a.wrapping_add(1);
    x.wrapping_add(b)
}
""",
    ),

    # -----------------------------------------------------------------------
    # Heap allocation patterns
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="heap_alloc_use",
        category="memory",
        expected_result="divergent",
        description="Heap alloc pattern — C addition UB vs Rust wrapping",
        divergence_kind="signed_overflow",
        c_source="""
int heap_alloc_use(int val) {
    int result = val + 1;
    return result;
}
""",
        rust_source="""
pub fn heap_alloc_use(val: i32) -> i32 {
    let b = Box::new(val.wrapping_add(1));
    *b
}
""",
    ),

    # -----------------------------------------------------------------------
    # C2Rust-style patterns
    # -----------------------------------------------------------------------
    BenchmarkPair(
        name="c2rust_array_init",
        category="c2rust",
        expected_result="equivalent",
        description="C2Rust-style array initialization with unsafe",
        c_source="""
int init_and_sum(int a, int b, int c) {
    int arr[3];
    arr[0] = a;
    arr[1] = b;
    arr[2] = c;
    return arr[0] + arr[1] + arr[2];
}
""",
        rust_source="""
pub fn init_and_sum(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(b).wrapping_add(c)
}
""",
    ),
    BenchmarkPair(
        name="c2rust_conditional_ptr",
        category="c2rust",
        expected_result="equivalent",
        description="Conditional pointer access pattern from C2Rust output",
        c_source="""
int cond_access(int flag, int a, int b) {
    int vals[2];
    vals[0] = a;
    vals[1] = b;
    if (flag) {
        return vals[0];
    }
    return vals[1];
}
""",
        rust_source="""
pub fn cond_access(flag: i32, a: i32, b: i32) -> i32 {
    if flag != 0 { a } else { b }
}
""",
    ),
    BenchmarkPair(
        name="c2rust_buffer_fill",
        category="c2rust",
        expected_result="equivalent",
        description="C2Rust-style buffer fill and accumulate",
        c_source="""
int buffer_fill(int start, int step) {
    int buf[4];
    int i;
    for (i = 0; i < 4; i++) {
        buf[i] = start + i * step;
    }
    return buf[0] + buf[3];
}
""",
        rust_source="""
pub fn buffer_fill(start: i32, step: i32) -> i32 {
    let first = start;
    let last = start.wrapping_add(3i32.wrapping_mul(step));
    first.wrapping_add(last)
}
""",
    ),
    BenchmarkPair(
        name="c2rust_goto_elimination",
        category="c2rust",
        expected_result="equivalent",
        description="Pattern from C2Rust goto elimination: loop+break replaces goto",
        c_source="""
int classify(int x) {
    int result;
    if (x > 100) {
        result = 2;
    } else if (x > 0) {
        result = 1;
    } else {
        result = 0;
    }
    return result;
}
""",
        rust_source="""
pub fn classify(x: i32) -> i32 {
    if x > 100 { 2 }
    else if x > 0 { 1 }
    else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="c2rust_macro_expand",
        category="c2rust",
        expected_result="divergent",
        description="Macro expansion where C uses signed and Rust uses unsigned — diverges on overflow",
        divergence_kind="signed_overflow",
        c_source="""
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
""",
        rust_source="""
pub fn clamp(x: i32, lo: i32, hi: i32) -> i32 {
    x.max(lo).min(hi)
}
""",
    ),
    BenchmarkPair(
        name="c2rust_strlen_simple",
        category="c2rust",
        expected_result="equivalent",
        description="Simplified strlen: counts until zero byte, bounded",
        c_source="""
int count_nonzero(int a, int b, int c, int d) {
    int count = 0;
    if (a != 0) count++;
    if (b != 0) count++;
    if (c != 0) count++;
    if (d != 0) count++;
    return count;
}
""",
        rust_source="""
pub fn count_nonzero(a: i32, b: i32, c: i32, d: i32) -> i32 {
    let mut count: i32 = 0;
    if a != 0 { count += 1; }
    if b != 0 { count += 1; }
    if c != 0 { count += 1; }
    if d != 0 { count += 1; }
    count
}
""",
    ),
    BenchmarkPair(
        name="c2rust_bitfield_extract",
        category="c2rust",
        expected_result="equivalent",
        description="C2Rust bitfield extraction pattern",
        c_source="""
int extract_field(unsigned int packed, int offset, int width) {
    unsigned int mask;
    if (width <= 0 || width > 32 || offset < 0 || offset >= 32) return 0;
    mask = ((1u << width) - 1) << offset;
    return (int)((packed & mask) >> offset);
}
""",
        rust_source="""
pub fn extract_field(packed: u32, offset: i32, width: i32) -> i32 {
    if width <= 0 || width > 32 || offset < 0 || offset >= 32 {
        return 0;
    }
    let mask: u32 = ((1u32.wrapping_shl(width as u32)) - 1).wrapping_shl(offset as u32);
    ((packed & mask) >> (offset as u32)) as i32
}
""",
    ),
]


# -----------------------------------------------------------------------
# Larger/scaled memory benchmarks (100+ LOC equivalent)
# -----------------------------------------------------------------------

SCALED_MEMORY_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="matrix_multiply_2x2",
        category="memory_scaled",
        expected_result="equivalent",
        description="2x2 matrix multiply — tests nested array access",
        c_source="""
int mat_mul_trace(int a00, int a01, int a10, int a11,
                  int b00, int b01, int b10, int b11) {
    int c00 = a00 * b00 + a01 * b10;
    int c11 = a10 * b01 + a11 * b11;
    return c00 + c11;
}
""",
        rust_source="""
pub fn mat_mul_trace(a00: i32, a01: i32, a10: i32, a11: i32,
                     b00: i32, b01: i32, b10: i32, b11: i32) -> i32 {
    let c00 = a00.wrapping_mul(b00).wrapping_add(a01.wrapping_mul(b10));
    let c11 = a10.wrapping_mul(b01).wrapping_add(a11.wrapping_mul(b11));
    c00.wrapping_add(c11)
}
""",
    ),
    BenchmarkPair(
        name="ring_buffer_insert",
        category="memory_scaled",
        expected_result="equivalent",
        description="Ring buffer insert/read pattern",
        c_source="""
int ring_buffer(int a, int b, int c, int capacity) {
    if (capacity <= 0 || capacity > 4) capacity = 4;
    int head = 0;
    int result = 0;
    int vals[3];
    vals[0] = a; vals[1] = b; vals[2] = c;
    int i;
    for (i = 0; i < 3; i++) {
        result += vals[i % capacity];
    }
    return result;
}
""",
        rust_source="""
pub fn ring_buffer(a: i32, b: i32, c: i32, capacity: i32) -> i32 {
    let cap = if capacity <= 0 || capacity > 4 { 4 } else { capacity };
    let vals = [a, b, c];
    let mut result: i32 = 0;
    for i in 0..3 {
        result = result.wrapping_add(vals[(i as usize) % (cap as usize)]);
    }
    result
}
""",
    ),
    BenchmarkPair(
        name="bubble_sort_3",
        category="memory_scaled",
        expected_result="equivalent",
        description="Bubble sort 3 elements — tests repeated load/store/compare",
        c_source="""
int sort3_median(int a, int b, int c) {
    int arr[3];
    arr[0] = a; arr[1] = b; arr[2] = c;
    int tmp;
    if (arr[0] > arr[1]) { tmp = arr[0]; arr[0] = arr[1]; arr[1] = tmp; }
    if (arr[1] > arr[2]) { tmp = arr[1]; arr[1] = arr[2]; arr[2] = tmp; }
    if (arr[0] > arr[1]) { tmp = arr[0]; arr[0] = arr[1]; arr[1] = tmp; }
    return arr[1];
}
""",
        rust_source="""
pub fn sort3_median(a: i32, b: i32, c: i32) -> i32 {
    let mut arr = [a, b, c];
    if arr[0] > arr[1] { arr.swap(0, 1); }
    if arr[1] > arr[2] { arr.swap(1, 2); }
    if arr[0] > arr[1] { arr.swap(0, 1); }
    arr[1]
}
""",
    ),
    BenchmarkPair(
        name="linked_list_sum_unrolled",
        category="memory_scaled",
        expected_result="equivalent",
        description="Unrolled linked list traversal (simulated with array)",
        c_source="""
int list_sum(int v0, int v1, int v2, int len) {
    int sum = 0;
    if (len > 0) sum += v0;
    if (len > 1) sum += v1;
    if (len > 2) sum += v2;
    return sum;
}
""",
        rust_source="""
pub fn list_sum(v0: i32, v1: i32, v2: i32, len: i32) -> i32 {
    let mut sum: i32 = 0;
    if len > 0 { sum = sum.wrapping_add(v0); }
    if len > 1 { sum = sum.wrapping_add(v1); }
    if len > 2 { sum = sum.wrapping_add(v2); }
    sum
}
""",
    ),
]


def get_all_memory_pairs() -> List[BenchmarkPair]:
    """Return all memory/pointer benchmark pairs."""
    return MEMORY_PAIRS + SCALED_MEMORY_PAIRS
