"""
Expanded benchmark pairs for cross-language equivalence verification.

Adds 150+ function pairs across new categories:
- Struct operations (field access, construction, nested structs)
- Enum/tagged union patterns (match vs switch, Result types)
- Floating-point semantics (NaN handling, rounding, precision)
- Real-world C2Rust patterns (transpiler output)
- Iterator transforms (for-loop vs iterator chains)
- Error handling patterns (errno vs Result, NULL vs Option)
- Type casting edge cases
- Compound expressions
"""

from __future__ import annotations
from typing import List
from .benchmark_pairs import BenchmarkPair


# ---------------------------------------------------------------------------
# Category: Struct operations (20 pairs)
# ---------------------------------------------------------------------------

STRUCT_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="struct_field_access",
        category="struct",
        expected_result="equivalent",
        description="Simple struct field access — identical semantics",
        c_source="""
struct Point { int x; int y; };
int get_x(struct Point p) { return p.x; }
""",
        rust_source="""
pub struct Point { pub x: i32, pub y: i32 }
pub fn get_x(p: Point) -> i32 { p.x }
""",
    ),
    BenchmarkPair(
        name="struct_sum_fields",
        category="struct",
        expected_result="equivalent",
        description="Sum of struct fields — identical semantics",
        c_source="""
struct Vec2 { int x; int y; };
int sum_fields(struct Vec2 v) { return v.x + v.y; }
""",
        rust_source="""
pub struct Vec2 { pub x: i32, pub y: i32 }
pub fn sum_fields(v: Vec2) -> i32 { v.x + v.y }
""",
    ),
    BenchmarkPair(
        name="struct_field_overflow",
        category="struct",
        expected_result="divergent",
        description="Struct field addition with overflow — C UB vs Rust wrapping",
        divergence_kind="signed_overflow",
        c_source="""
struct Pair { int a; int b; };
int sum_pair(struct Pair p) { return p.a + p.b; }
""",
        rust_source="""
pub struct Pair { pub a: i32, pub b: i32 }
pub fn sum_pair(p: Pair) -> i32 { p.a.wrapping_add(p.b) }
""",
    ),
    BenchmarkPair(
        name="struct_nested_access",
        category="struct",
        expected_result="equivalent",
        description="Nested struct field access",
        c_source="""
struct Inner { int val; };
struct Outer { struct Inner inner; int extra; };
int get_inner_val(struct Outer o) { return o.inner.val; }
""",
        rust_source="""
pub struct Inner { pub val: i32 }
pub struct Outer { pub inner: Inner, pub extra: i32 }
pub fn get_inner_val(o: Outer) -> i32 { o.inner.val }
""",
    ),
    BenchmarkPair(
        name="struct_construction_equiv",
        category="struct",
        expected_result="equivalent",
        description="Struct construction and immediate field access",
        c_source="""
struct Rect { int w; int h; };
int area(int w, int h) {
    struct Rect r;
    r.w = w;
    r.h = h;
    return r.w * r.h;
}
""",
        rust_source="""
pub struct Rect { pub w: i32, pub h: i32 }
pub fn area(w: i32, h: i32) -> i32 {
    let r = Rect { w, h };
    r.w * r.h
}
""",
    ),
    BenchmarkPair(
        name="struct_mul_overflow",
        category="struct",
        expected_result="divergent",
        description="Struct field multiplication — overflow divergence",
        divergence_kind="signed_overflow",
        c_source="""
struct Dims { int w; int h; };
int area(struct Dims d) { return d.w * d.h; }
""",
        rust_source="""
pub struct Dims { pub w: i32, pub h: i32 }
pub fn area(d: Dims) -> i32 { d.w.wrapping_mul(d.h) }
""",
    ),
    BenchmarkPair(
        name="struct_conditional_field",
        category="struct",
        expected_result="equivalent",
        description="Conditional based on struct field",
        c_source="""
struct Flags { int active; int count; };
int check(struct Flags f) {
    if (f.active) return f.count;
    return 0;
}
""",
        rust_source="""
pub struct Flags { pub active: i32, pub count: i32 }
pub fn check(f: Flags) -> i32 {
    if f.active != 0 { f.count } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="struct_swap_fields",
        category="struct",
        expected_result="equivalent",
        description="Return struct with swapped fields — equivalent field ops",
        c_source="""
struct Pair { int a; int b; };
int swap_sum(struct Pair p) {
    int tmp = p.a;
    p.a = p.b;
    p.b = tmp;
    return p.a + p.b;
}
""",
        rust_source="""
pub struct Pair { pub a: i32, pub b: i32 }
pub fn swap_sum(mut p: Pair) -> i32 {
    let tmp = p.a;
    p.a = p.b;
    p.b = tmp;
    p.a + p.b
}
""",
    ),
    BenchmarkPair(
        name="struct_packed_vs_aligned",
        category="struct",
        expected_result="equivalent",
        description="Packed struct field access — same value semantics",
        c_source="""
struct __attribute__((packed)) Compact { char a; int b; };
int get_b(struct Compact c) { return c.b; }
""",
        rust_source="""
#[repr(packed)]
pub struct Compact { pub a: i8, pub b: i32 }
pub fn get_b(c: Compact) -> i32 { unsafe { c.b } }
""",
    ),
    BenchmarkPair(
        name="struct_max_field",
        category="struct",
        expected_result="equivalent",
        description="Return max of struct fields",
        c_source="""
struct Trio { int a; int b; int c; };
int max_field(struct Trio t) {
    int m = t.a;
    if (t.b > m) m = t.b;
    if (t.c > m) m = t.c;
    return m;
}
""",
        rust_source="""
pub struct Trio { pub a: i32, pub b: i32, pub c: i32 }
pub fn max_field(t: Trio) -> i32 {
    let mut m = t.a;
    if t.b > m { m = t.b; }
    if t.c > m { m = t.c; }
    m
}
""",
    ),
    BenchmarkPair(
        name="struct_dot_product",
        category="struct",
        expected_result="divergent",
        description="Dot product of 2D vectors — overflow divergence",
        divergence_kind="signed_overflow",
        c_source="""
struct Vec2 { int x; int y; };
int dot(struct Vec2 a, struct Vec2 b) { return a.x * b.x + a.y * b.y; }
""",
        rust_source="""
pub struct Vec2 { pub x: i32, pub y: i32 }
pub fn dot(a: Vec2, b: Vec2) -> i32 {
    a.x.wrapping_mul(b.x).wrapping_add(a.y.wrapping_mul(b.y))
}
""",
    ),
    BenchmarkPair(
        name="struct_equality_check",
        category="struct",
        expected_result="equivalent",
        description="Check if two structs are equal field-by-field",
        c_source="""
struct Point { int x; int y; };
int equal(struct Point a, struct Point b) {
    return a.x == b.x && a.y == b.y;
}
""",
        rust_source="""
pub struct Point { pub x: i32, pub y: i32 }
pub fn equal(a: Point, b: Point) -> i32 {
    if a.x == b.x && a.y == b.y { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="struct_distance_sq",
        category="struct",
        expected_result="divergent",
        description="Squared distance — mul/add overflow",
        divergence_kind="signed_overflow",
        c_source="""
struct Point { int x; int y; };
int dist_sq(struct Point a, struct Point b) {
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return dx * dx + dy * dy;
}
""",
        rust_source="""
pub struct Point { pub x: i32, pub y: i32 }
pub fn dist_sq(a: Point, b: Point) -> i32 {
    let dx = a.x.wrapping_sub(b.x);
    let dy = a.y.wrapping_sub(b.y);
    dx.wrapping_mul(dx).wrapping_add(dy.wrapping_mul(dy))
}
""",
    ),
    BenchmarkPair(
        name="struct_midpoint",
        category="struct",
        expected_result="equivalent",
        description="Midpoint of two points — equivalent with bit tricks",
        c_source="""
struct Point { int x; int y; };
int midpoint_x(struct Point a, struct Point b) {
    return (a.x >> 1) + (b.x >> 1) + (a.x & b.x & 1);
}
""",
        rust_source="""
pub struct Point { pub x: i32, pub y: i32 }
pub fn midpoint_x(a: Point, b: Point) -> i32 {
    (a.x >> 1) + (b.x >> 1) + (a.x & b.x & 1)
}
""",
    ),
    BenchmarkPair(
        name="struct_scale",
        category="struct",
        expected_result="divergent",
        description="Scale struct fields by factor — overflow divergence",
        divergence_kind="signed_overflow",
        c_source="""
struct Vec2 { int x; int y; };
int scaled_sum(struct Vec2 v, int k) { return v.x * k + v.y * k; }
""",
        rust_source="""
pub struct Vec2 { pub x: i32, pub y: i32 }
pub fn scaled_sum(v: Vec2, k: i32) -> i32 {
    v.x.wrapping_mul(k).wrapping_add(v.y.wrapping_mul(k))
}
""",
    ),
    BenchmarkPair(
        name="struct_manhattan_dist",
        category="struct",
        expected_result="equivalent",
        description="Manhattan distance — absolute value of differences",
        c_source="""
struct Point { int x; int y; };
int abs_val(int v) { return v < 0 ? -v : v; }
int manhattan(struct Point a, struct Point b) {
    return abs_val(a.x - b.x) + abs_val(a.y - b.y);
}
""",
        rust_source="""
pub struct Point { pub x: i32, pub y: i32 }
pub fn manhattan(a: Point, b: Point) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}
""",
    ),
    BenchmarkPair(
        name="struct_clamp_fields",
        category="struct",
        expected_result="equivalent",
        description="Clamp struct fields to range",
        c_source="""
struct Range { int lo; int hi; };
int clamp(struct Range r, int val) {
    if (val < r.lo) return r.lo;
    if (val > r.hi) return r.hi;
    return val;
}
""",
        rust_source="""
pub struct Range { pub lo: i32, pub hi: i32 }
pub fn clamp(r: Range, val: i32) -> i32 {
    if val < r.lo { r.lo } else if val > r.hi { r.hi } else { val }
}
""",
    ),
    BenchmarkPair(
        name="struct_bit_field_emulation",
        category="struct",
        expected_result="equivalent",
        description="Bit field access emulated with masks",
        c_source="""
struct Flags { unsigned int packed; };
int get_bit(struct Flags f, int n) {
    return (f.packed >> n) & 1;
}
""",
        rust_source="""
pub struct Flags { pub packed: u32 }
pub fn get_bit(f: Flags, n: i32) -> i32 {
    ((f.packed >> (n as u32)) & 1) as i32
}
""",
    ),
    BenchmarkPair(
        name="struct_weighted_avg",
        category="struct",
        expected_result="divergent",
        description="Weighted average — division edge case",
        divergence_kind="division_by_zero",
        c_source="""
struct WeightedVal { int value; int weight; };
int weighted_avg(struct WeightedVal a, struct WeightedVal b) {
    return (a.value * a.weight + b.value * b.weight) / (a.weight + b.weight);
}
""",
        rust_source="""
pub struct WeightedVal { pub value: i32, pub weight: i32 }
pub fn weighted_avg(a: WeightedVal, b: WeightedVal) -> i32 {
    let total_w = a.weight + b.weight;
    if total_w == 0 { return 0; }
    (a.value * a.weight + b.value * b.weight) / total_w
}
""",
    ),
    BenchmarkPair(
        name="struct_color_blend",
        category="struct",
        expected_result="equivalent",
        description="Color channel blending with unsigned arithmetic",
        c_source="""
struct Color { unsigned char r; unsigned char g; unsigned char b; };
unsigned char blend(unsigned char a, unsigned char b) {
    return (unsigned char)(((unsigned int)a + (unsigned int)b) / 2);
}
int blend_r(struct Color c1, struct Color c2) {
    return blend(c1.r, c2.r);
}
""",
        rust_source="""
pub struct Color { pub r: u8, pub g: u8, pub b: u8 }
pub fn blend_r(c1: Color, c2: Color) -> i32 {
    (((c1.r as u32) + (c2.r as u32)) / 2) as i32
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category: Enum/Tagged Union patterns (20 pairs)
# ---------------------------------------------------------------------------

ENUM_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="enum_c_like_equiv",
        category="enum",
        expected_result="equivalent",
        description="C-style enum vs Rust enum — identical discriminant values",
        c_source="""
enum Color { RED = 0, GREEN = 1, BLUE = 2 };
int color_to_int(enum Color c) { return (int)c; }
""",
        rust_source="""
#[repr(i32)]
pub enum Color { Red = 0, Green = 1, Blue = 2 }
pub fn color_to_int(c: Color) -> i32 { c as i32 }
""",
    ),
    BenchmarkPair(
        name="enum_switch_vs_match",
        category="enum",
        expected_result="equivalent",
        description="C switch on enum vs Rust match — same semantics",
        c_source="""
enum Shape { CIRCLE = 0, SQUARE = 1, TRIANGLE = 2 };
int sides(enum Shape s) {
    switch(s) {
        case CIRCLE: return 0;
        case SQUARE: return 4;
        case TRIANGLE: return 3;
        default: return -1;
    }
}
""",
        rust_source="""
#[repr(i32)]
pub enum Shape { Circle = 0, Square = 1, Triangle = 2 }
pub fn sides(s: Shape) -> i32 {
    match s {
        Shape::Circle => 0,
        Shape::Square => 4,
        Shape::Triangle => 3,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_fallthrough_divergence",
        category="enum",
        expected_result="divergent",
        description="C switch fallthrough vs Rust match — divergence on fallthrough",
        divergence_kind="control_flow",
        c_source="""
enum Level { LOW = 0, MED = 1, HIGH = 2 };
int threshold(enum Level l) {
    switch(l) {
        case LOW:
        case MED: return 10;
        case HIGH: return 20;
        default: return 0;
    }
}
""",
        rust_source="""
#[repr(i32)]
pub enum Level { Low = 0, Med = 1, High = 2 }
pub fn threshold(l: Level) -> i32 {
    match l {
        Level::Low => 5,
        Level::Med => 10,
        Level::High => 20,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_tagged_union_option",
        category="enum",
        expected_result="equivalent",
        description="C nullable int vs Rust Option<i32>",
        c_source="""
struct OptInt { int has_val; int val; };
int unwrap_or(struct OptInt o, int def) {
    if (o.has_val) return o.val;
    return def;
}
""",
        rust_source="""
pub fn unwrap_or(o: Option<i32>, def: i32) -> i32 {
    match o {
        Some(v) => v,
        None => def,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_result_ok_path",
        category="enum",
        expected_result="equivalent",
        description="C error code vs Rust Result — OK path",
        c_source="""
int safe_div(int a, int b, int *result) {
    if (b == 0) return -1;
    *result = a / b;
    return 0;
}
""",
        rust_source="""
pub fn safe_div(a: i32, b: i32) -> Result<i32, i32> {
    if b == 0 { Err(-1) } else { Ok(a / b) }
}
""",
    ),
    BenchmarkPair(
        name="enum_exhaustiveness",
        category="enum",
        expected_result="divergent",
        description="C missing default case vs Rust exhaustive match",
        divergence_kind="undefined_behavior",
        c_source="""
enum Dir { NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3 };
int delta_x(enum Dir d) {
    switch(d) {
        case EAST: return 1;
        case WEST: return -1;
    }
    return 0;
}
""",
        rust_source="""
#[repr(i32)]
pub enum Dir { North = 0, South = 1, East = 2, West = 3 }
pub fn delta_x(d: Dir) -> i32 {
    match d {
        Dir::East => 1,
        Dir::West => -1,
        _ => 0,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_arithmetic_on_discriminant",
        category="enum",
        expected_result="equivalent",
        description="Arithmetic on enum discriminant values",
        c_source="""
enum Priority { LOW = 1, MED = 2, HIGH = 3 };
int priority_double(enum Priority p) { return (int)p * 2; }
""",
        rust_source="""
#[repr(i32)]
pub enum Priority { Low = 1, Med = 2, High = 3 }
pub fn priority_double(p: Priority) -> i32 { (p as i32) * 2 }
""",
    ),
    BenchmarkPair(
        name="enum_comparison",
        category="enum",
        expected_result="equivalent",
        description="Compare enum values as integers",
        c_source="""
enum Size { SMALL = 0, MEDIUM = 1, LARGE = 2 };
int is_large_or_bigger(enum Size s) { return s >= LARGE; }
""",
        rust_source="""
#[repr(i32)]
pub enum Size { Small = 0, Medium = 1, Large = 2 }
pub fn is_large_or_bigger(s: Size) -> i32 {
    if (s as i32) >= 2 { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="enum_bitflags",
        category="enum",
        expected_result="equivalent",
        description="Enum used as bitflags",
        c_source="""
enum Flags { READ = 1, WRITE = 2, EXEC = 4 };
int has_read_write(int flags) { return (flags & (READ | WRITE)) == (READ | WRITE); }
""",
        rust_source="""
const READ: i32 = 1;
const WRITE: i32 = 2;
const EXEC: i32 = 4;
pub fn has_read_write(flags: i32) -> i32 {
    if (flags & (READ | WRITE)) == (READ | WRITE) { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="enum_next_variant",
        category="enum",
        expected_result="equivalent",
        description="Cycle through enum variants",
        c_source="""
enum Day { MON=0, TUE=1, WED=2, THU=3, FRI=4, SAT=5, SUN=6 };
int next_day(enum Day d) { return ((int)d + 1) % 7; }
""",
        rust_source="""
pub fn next_day(d: i32) -> i32 { (d + 1) % 7 }
""",
    ),
    BenchmarkPair(
        name="enum_option_map",
        category="enum",
        expected_result="equivalent",
        description="C nullable transform vs Rust Option::map",
        c_source="""
struct OptInt { int has_val; int val; };
struct OptInt double_opt(struct OptInt o) {
    struct OptInt result;
    if (o.has_val) { result.has_val = 1; result.val = o.val * 2; }
    else { result.has_val = 0; result.val = 0; }
    return result;
}
int extract(struct OptInt o) { return o.has_val ? o.val : -1; }
int double_or_neg(struct OptInt o) { return extract(double_opt(o)); }
""",
        rust_source="""
pub fn double_or_neg(o: Option<i32>) -> i32 {
    o.map(|v| v * 2).unwrap_or(-1)
}
""",
    ),
    BenchmarkPair(
        name="enum_result_chain",
        category="enum",
        expected_result="equivalent",
        description="Chained error handling — C error codes vs Rust Result",
        c_source="""
int step1(int x) { if (x < 0) return -1; return x + 1; }
int step2(int x) { if (x > 100) return -2; return x * 2; }
int chain(int x) {
    int r = step1(x);
    if (r < 0) return r;
    return step2(r);
}
""",
        rust_source="""
fn step1(x: i32) -> Result<i32, i32> { if x < 0 { Err(-1) } else { Ok(x + 1) } }
fn step2(x: i32) -> Result<i32, i32> { if x > 100 { Err(-2) } else { Ok(x * 2) } }
pub fn chain(x: i32) -> i32 {
    match step1(x).and_then(step2) {
        Ok(v) => v,
        Err(e) => e,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_nested_match",
        category="enum",
        expected_result="equivalent",
        description="Nested switch/match on enums",
        c_source="""
enum Op { ADD_OP = 0, SUB_OP = 1 };
enum Sign { POS = 0, NEG = 1 };
int apply(enum Op op, enum Sign s, int val) {
    int v = (s == NEG) ? -val : val;
    return (op == ADD_OP) ? v : -v;
}
""",
        rust_source="""
pub fn apply(op: i32, s: i32, val: i32) -> i32 {
    let v = if s == 1 { -val } else { val };
    if op == 0 { v } else { -v }
}
""",
    ),
    BenchmarkPair(
        name="enum_state_machine",
        category="enum",
        expected_result="equivalent",
        description="Simple state machine — switch vs match",
        c_source="""
enum State { IDLE=0, RUN=1, DONE=2 };
int advance(int state) {
    switch(state) {
        case 0: return 1;
        case 1: return 2;
        case 2: return 2;
        default: return 0;
    }
}
""",
        rust_source="""
pub fn advance(state: i32) -> i32 {
    match state {
        0 => 1,
        1 => 2,
        2 => 2,
        _ => 0,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_discriminant_bounds",
        category="enum",
        expected_result="divergent",
        description="Out-of-range discriminant — C allows, Rust UB",
        divergence_kind="undefined_behavior",
        c_source="""
enum Small { A = 0, B = 1 };
int classify(int v) {
    enum Small s = (enum Small)v;
    return (int)s;
}
""",
        rust_source="""
pub fn classify(v: i32) -> i32 {
    if v == 0 || v == 1 { v } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="enum_option_filter",
        category="enum",
        expected_result="equivalent",
        description="Filter on optional value",
        c_source="""
int filter_positive(int has_val, int val) {
    if (!has_val) return -1;
    if (val > 0) return val;
    return -1;
}
""",
        rust_source="""
pub fn filter_positive(opt: Option<i32>) -> i32 {
    match opt.filter(|&v| v > 0) {
        Some(v) => v,
        None => -1,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_ternary_vs_match",
        category="enum",
        expected_result="equivalent",
        description="C ternary chain vs Rust match on value",
        c_source="""
int classify_temp(int t) {
    return t < 0 ? 0 : (t < 20 ? 1 : (t < 35 ? 2 : 3));
}
""",
        rust_source="""
pub fn classify_temp(t: i32) -> i32 {
    match t {
        t if t < 0 => 0,
        t if t < 20 => 1,
        t if t < 35 => 2,
        _ => 3,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_error_code_mapping",
        category="enum",
        expected_result="equivalent",
        description="Map error codes between C and Rust conventions",
        c_source="""
int error_to_exit(int err) {
    switch(err) {
        case 0: return 0;
        case -1: return 1;
        case -2: return 2;
        default: return 127;
    }
}
""",
        rust_source="""
pub fn error_to_exit(err: i32) -> i32 {
    match err {
        0 => 0,
        -1 => 1,
        -2 => 2,
        _ => 127,
    }
}
""",
    ),
    BenchmarkPair(
        name="enum_bool_result",
        category="enum",
        expected_result="equivalent",
        description="Boolean result — C int vs Rust bool conversion",
        c_source="""
int is_even(int n) { return n % 2 == 0; }
""",
        rust_source="""
pub fn is_even(n: i32) -> i32 { if n % 2 == 0 { 1 } else { 0 } }
""",
    ),
    BenchmarkPair(
        name="enum_option_unwrap_panic",
        category="enum",
        expected_result="divergent",
        description="C null check vs Rust unwrap panic",
        divergence_kind="error_model",
        c_source="""
int get_or_zero(int has_val, int val) {
    if (!has_val) return 0;
    return val;
}
""",
        rust_source="""
pub fn get_or_zero(opt: Option<i32>) -> i32 {
    opt.unwrap()
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category: Floating-point semantics (15 pairs)
# ---------------------------------------------------------------------------

FLOAT_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="float_add_equiv",
        category="float",
        expected_result="equivalent",
        description="Simple float addition — identical IEEE 754 semantics",
        c_source="""
float fadd(float a, float b) { return a + b; }
""",
        rust_source="""
pub fn fadd(a: f32, b: f32) -> f32 { a + b }
""",
    ),
    BenchmarkPair(
        name="float_nan_comparison",
        category="float",
        expected_result="equivalent",
        description="NaN comparison — both languages follow IEEE 754",
        c_source="""
int is_nan(double x) { return x != x; }
""",
        rust_source="""
pub fn is_nan(x: f64) -> i32 { if x.is_nan() { 1 } else { 0 } }
""",
    ),
    BenchmarkPair(
        name="float_inf_arithmetic",
        category="float",
        expected_result="equivalent",
        description="Infinity arithmetic",
        c_source="""
#include <math.h>
int inf_check(double x) { return isinf(x + 1.0); }
""",
        rust_source="""
pub fn inf_check(x: f64) -> i32 { if (x + 1.0).is_infinite() { 1 } else { 0 } }
""",
    ),
    BenchmarkPair(
        name="float_to_int_truncation",
        category="float",
        expected_result="divergent",
        description="Float-to-int conversion — C truncation vs Rust saturating",
        divergence_kind="float_conversion",
        c_source="""
int ftoi(float x) { return (int)x; }
""",
        rust_source="""
pub fn ftoi(x: f32) -> i32 { x as i32 }
""",
    ),
    BenchmarkPair(
        name="float_division_by_zero",
        category="float",
        expected_result="equivalent",
        description="Float division by zero — both produce infinity",
        c_source="""
double fdivz(double x) { return x / 0.0; }
""",
        rust_source="""
pub fn fdivz(x: f64) -> f64 { x / 0.0 }
""",
    ),
    BenchmarkPair(
        name="float_fma_precision",
        category="float",
        expected_result="conditional",
        description="FMA vs separate mul+add — precision difference",
        divergence_kind="precision",
        c_source="""
double fma_manual(double a, double b, double c) { return a * b + c; }
""",
        rust_source="""
pub fn fma_manual(a: f64, b: f64, c: f64) -> f64 { a.mul_add(b, c) }
""",
    ),
    BenchmarkPair(
        name="float_abs_equiv",
        category="float",
        expected_result="equivalent",
        description="Floating-point absolute value",
        c_source="""
double fabs_manual(double x) { return x < 0.0 ? -x : x; }
""",
        rust_source="""
pub fn fabs_manual(x: f64) -> f64 { x.abs() }
""",
    ),
    BenchmarkPair(
        name="float_rounding_mode",
        category="float",
        expected_result="equivalent",
        description="Default rounding — both use round-to-nearest-even",
        c_source="""
float round_trip(float x) { return (float)(double)x; }
""",
        rust_source="""
pub fn round_trip(x: f32) -> f32 { (x as f64) as f32 }
""",
    ),
    BenchmarkPair(
        name="float_neg_zero",
        category="float",
        expected_result="equivalent",
        description="Negative zero behavior",
        c_source="""
int is_neg_zero(double x) { return x == 0.0 && (1.0/x) < 0.0; }
""",
        rust_source="""
pub fn is_neg_zero(x: f64) -> i32 {
    if x == 0.0 && x.is_sign_negative() { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="float_lerp",
        category="float",
        expected_result="equivalent",
        description="Linear interpolation",
        c_source="""
double lerp(double a, double b, double t) { return a + (b - a) * t; }
""",
        rust_source="""
pub fn lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }
""",
    ),
    BenchmarkPair(
        name="float_compare_epsilon",
        category="float",
        expected_result="equivalent",
        description="Approximate float comparison with epsilon",
        c_source="""
int approx_eq(double a, double b) {
    double diff = a - b;
    if (diff < 0) diff = -diff;
    return diff < 1e-9;
}
""",
        rust_source="""
pub fn approx_eq(a: f64, b: f64) -> i32 {
    if (a - b).abs() < 1e-9 { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="float_sum_associativity",
        category="float",
        expected_result="conditional",
        description="Float sum order matters — non-associativity",
        divergence_kind="precision",
        c_source="""
double sum3(double a, double b, double c) { return (a + b) + c; }
""",
        rust_source="""
pub fn sum3(a: f64, b: f64, c: f64) -> f64 { a + (b + c) }
""",
    ),
    BenchmarkPair(
        name="float_clamp",
        category="float",
        expected_result="equivalent",
        description="Float clamp to range",
        c_source="""
double fclamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
""",
        rust_source="""
pub fn fclamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.clamp(lo, hi)
}
""",
    ),
    BenchmarkPair(
        name="float_int_conv_roundtrip",
        category="float",
        expected_result="equivalent",
        description="Int-to-float-to-int roundtrip for small values",
        c_source="""
int roundtrip(int x) { return (int)(float)x; }
""",
        rust_source="""
pub fn roundtrip(x: i32) -> i32 { (x as f32) as i32 }
""",
    ),
    BenchmarkPair(
        name="float_sign_bit",
        category="float",
        expected_result="equivalent",
        description="Extract sign bit from float",
        c_source="""
int sign_bit(double x) { return x < 0.0 ? 1 : 0; }
""",
        rust_source="""
pub fn sign_bit(x: f64) -> i32 { if x < 0.0 { 1 } else { 0 } }
""",
    ),
]


# ---------------------------------------------------------------------------
# Category: Real-world C2Rust patterns (20 pairs)
# ---------------------------------------------------------------------------

C2RUST_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="c2rust_strlen_loop",
        category="c2rust",
        expected_result="equivalent",
        description="String length via while loop — common transpiler pattern",
        c_source="""
int my_strlen(const char *s) {
    int len = 0;
    while (s[len] != '\\0') len++;
    return len;
}
""",
        rust_source="""
pub fn my_strlen(s: &[u8]) -> i32 {
    let mut len: i32 = 0;
    while (len as usize) < s.len() && s[len as usize] != 0 {
        len += 1;
    }
    len
}
""",
    ),
    BenchmarkPair(
        name="c2rust_array_sum",
        category="c2rust",
        expected_result="divergent",
        description="Array sum — overflow on large arrays",
        divergence_kind="signed_overflow",
        c_source="""
int array_sum(const int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];
    return sum;
}
""",
        rust_source="""
pub fn array_sum(arr: &[i32]) -> i32 {
    arr.iter().fold(0i32, |acc, &x| acc.wrapping_add(x))
}
""",
    ),
    BenchmarkPair(
        name="c2rust_abs_val",
        category="c2rust",
        expected_result="divergent",
        description="Absolute value — INT_MIN edge case",
        divergence_kind="signed_overflow",
        c_source="""
int my_abs(int x) { return x < 0 ? -x : x; }
""",
        rust_source="""
pub fn my_abs(x: i32) -> i32 { x.wrapping_abs() }
""",
    ),
    BenchmarkPair(
        name="c2rust_min_max",
        category="c2rust",
        expected_result="equivalent",
        description="Min/max via ternary vs std::cmp",
        c_source="""
int my_max(int a, int b) { return a > b ? a : b; }
""",
        rust_source="""
pub fn my_max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }
""",
    ),
    BenchmarkPair(
        name="c2rust_swap_xor",
        category="c2rust",
        expected_result="equivalent",
        description="XOR swap — equivalent bit operations",
        c_source="""
int xor_swap_sum(int a, int b) {
    a ^= b; b ^= a; a ^= b;
    return a + b;
}
""",
        rust_source="""
pub fn xor_swap_sum(mut a: i32, mut b: i32) -> i32 {
    a ^= b; b ^= a; a ^= b;
    a + b
}
""",
    ),
    BenchmarkPair(
        name="c2rust_binary_search",
        category="c2rust",
        expected_result="equivalent",
        description="Binary search — common algorithm translation",
        c_source="""
int binary_search(int val, int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (mid == val) return mid;
        if (mid < val) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
""",
        rust_source="""
pub fn binary_search(val: i32, n: i32) -> i32 {
    let mut lo = 0i32;
    let mut hi = n - 1;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        if mid == val { return mid; }
        if mid < val { lo = mid + 1; } else { hi = mid - 1; }
    }
    -1
}
""",
    ),
    BenchmarkPair(
        name="c2rust_fibonacci",
        category="c2rust",
        expected_result="divergent",
        description="Fibonacci — overflow for large n",
        divergence_kind="signed_overflow",
        c_source="""
int fib(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) { int t = a + b; a = b; b = t; }
    return b;
}
""",
        rust_source="""
pub fn fib(n: i32) -> i32 {
    if n <= 1 { return n; }
    let (mut a, mut b) = (0i32, 1i32);
    for _ in 2..=n { let t = a.wrapping_add(b); a = b; b = t; }
    b
}
""",
    ),
    BenchmarkPair(
        name="c2rust_gcd",
        category="c2rust",
        expected_result="equivalent",
        description="GCD via Euclidean algorithm",
        c_source="""
int gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}
""",
        rust_source="""
pub fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}
""",
    ),
    BenchmarkPair(
        name="c2rust_popcount",
        category="c2rust",
        expected_result="equivalent",
        description="Population count — bit counting",
        c_source="""
int popcount(unsigned int x) {
    int count = 0;
    while (x) { count += x & 1; x >>= 1; }
    return count;
}
""",
        rust_source="""
pub fn popcount(mut x: u32) -> i32 {
    let mut count = 0i32;
    while x != 0 { count += (x & 1) as i32; x >>= 1; }
    count
}
""",
    ),
    BenchmarkPair(
        name="c2rust_reverse_bits",
        category="c2rust",
        expected_result="equivalent",
        description="Reverse bits of a 32-bit integer",
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
    for _ in 0..32 {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}
""",
    ),
    BenchmarkPair(
        name="c2rust_count_digits",
        category="c2rust",
        expected_result="equivalent",
        description="Count decimal digits",
        c_source="""
int count_digits(int n) {
    if (n == 0) return 1;
    int count = 0;
    if (n < 0) n = -n;
    while (n > 0) { count++; n /= 10; }
    return count;
}
""",
        rust_source="""
pub fn count_digits(mut n: i32) -> i32 {
    if n == 0 { return 1; }
    let mut count = 0;
    if n < 0 { n = -n; }
    while n > 0 { count += 1; n /= 10; }
    count
}
""",
    ),
    BenchmarkPair(
        name="c2rust_power_iterative",
        category="c2rust",
        expected_result="divergent",
        description="Integer power — overflow for large exponents",
        divergence_kind="signed_overflow",
        c_source="""
int power(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; i++) result *= base;
    return result;
}
""",
        rust_source="""
pub fn power(base: i32, exp: i32) -> i32 {
    let mut result = 1i32;
    for _ in 0..exp { result = result.wrapping_mul(base); }
    result
}
""",
    ),
    BenchmarkPair(
        name="c2rust_is_palindrome_num",
        category="c2rust",
        expected_result="equivalent",
        description="Check if number is a palindrome",
        c_source="""
int is_palindrome(int n) {
    if (n < 0) return 0;
    int rev = 0, orig = n;
    while (n > 0) { rev = rev * 10 + n % 10; n /= 10; }
    return rev == orig;
}
""",
        rust_source="""
pub fn is_palindrome(n: i32) -> i32 {
    if n < 0 { return 0; }
    let mut rev = 0i32;
    let orig = n;
    let mut n = n;
    while n > 0 { rev = rev * 10 + n % 10; n /= 10; }
    if rev == orig { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="c2rust_array_reverse",
        category="c2rust",
        expected_result="equivalent",
        description="Array element sum after reverse — order doesn't matter",
        c_source="""
int sum_reversed(int a, int b, int c) {
    return c + b + a;
}
""",
        rust_source="""
pub fn sum_reversed(a: i32, b: i32, c: i32) -> i32 {
    [a, b, c].iter().rev().sum::<i32>()
}
""",
    ),
    BenchmarkPair(
        name="c2rust_clz",
        category="c2rust",
        expected_result="equivalent",
        description="Count leading zeros",
        c_source="""
int clz(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8; x <<= 8; }
    if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; }
    if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
}
""",
        rust_source="""
pub fn clz(x: u32) -> i32 {
    if x == 0 { 32 } else { x.leading_zeros() as i32 }
}
""",
    ),
    BenchmarkPair(
        name="c2rust_byte_swap",
        category="c2rust",
        expected_result="equivalent",
        description="Byte swap of 32-bit integer",
        c_source="""
unsigned int bswap32(unsigned int x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}
""",
        rust_source="""
pub fn bswap32(x: u32) -> u32 { x.swap_bytes() }
""",
    ),
    BenchmarkPair(
        name="c2rust_saturating_add",
        category="c2rust",
        expected_result="equivalent",
        description="Saturating addition",
        c_source="""
int sat_add(int a, int b) {
    int r = a + b;
    if (a > 0 && b > 0 && r < 0) return 2147483647;
    if (a < 0 && b < 0 && r > 0) return -2147483648;
    return r;
}
""",
        rust_source="""
pub fn sat_add(a: i32, b: i32) -> i32 { a.saturating_add(b) }
""",
    ),
    BenchmarkPair(
        name="c2rust_checked_mul",
        category="c2rust",
        expected_result="equivalent",
        description="Checked multiplication returning -1 on overflow",
        c_source="""
int checked_mul(int a, int b) {
    long long r = (long long)a * (long long)b;
    if (r > 2147483647 || r < -2147483648) return -1;
    return (int)r;
}
""",
        rust_source="""
pub fn checked_mul(a: i32, b: i32) -> i32 {
    match a.checked_mul(b) {
        Some(v) => v,
        None => -1,
    }
}
""",
    ),
    BenchmarkPair(
        name="c2rust_rotate_left",
        category="c2rust",
        expected_result="equivalent",
        description="32-bit left rotation",
        c_source="""
unsigned int rotl(unsigned int x, int n) {
    n &= 31;
    return (x << n) | (x >> (32 - n));
}
""",
        rust_source="""
pub fn rotl(x: u32, n: i32) -> u32 { x.rotate_left(n as u32) }
""",
    ),
    BenchmarkPair(
        name="c2rust_safe_array_access",
        category="c2rust",
        expected_result="divergent",
        description="Array bounds — C no check vs Rust panic",
        divergence_kind="bounds_check",
        c_source="""
int get_elem(int idx) { return idx * 2; }
""",
        rust_source="""
pub fn get_elem(idx: i32) -> i32 {
    let arr = [0, 2, 4, 6, 8];
    if (idx as usize) < arr.len() { arr[idx as usize] } else { -1 }
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category: Iterator transforms (15 pairs)
# ---------------------------------------------------------------------------

ITERATOR_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="iter_sum",
        category="iterator",
        expected_result="equivalent",
        description="Sum via for-loop vs iterator sum",
        c_source="""
int sum_n(int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) s += i;
    return s;
}
""",
        rust_source="""
pub fn sum_n(n: i32) -> i32 { (1..=n).sum() }
""",
    ),
    BenchmarkPair(
        name="iter_count_positive",
        category="iterator",
        expected_result="equivalent",
        description="Count positive values in range",
        c_source="""
int count_pos(int lo, int hi) {
    int c = 0;
    for (int i = lo; i < hi; i++) if (i > 0) c++;
    return c;
}
""",
        rust_source="""
pub fn count_pos(lo: i32, hi: i32) -> i32 {
    (lo..hi).filter(|&x| x > 0).count() as i32
}
""",
    ),
    BenchmarkPair(
        name="iter_find_first",
        category="iterator",
        expected_result="equivalent",
        description="Find first value matching predicate",
        c_source="""
int find_first_gt(int start, int end, int threshold) {
    for (int i = start; i < end; i++) {
        if (i > threshold) return i;
    }
    return -1;
}
""",
        rust_source="""
pub fn find_first_gt(start: i32, end: i32, threshold: i32) -> i32 {
    (start..end).find(|&x| x > threshold).unwrap_or(-1)
}
""",
    ),
    BenchmarkPair(
        name="iter_any_negative",
        category="iterator",
        expected_result="equivalent",
        description="Check if any value in range is negative",
        c_source="""
int any_negative(int lo, int hi) {
    for (int i = lo; i < hi; i++) if (i < 0) return 1;
    return 0;
}
""",
        rust_source="""
pub fn any_negative(lo: i32, hi: i32) -> i32 {
    if (lo..hi).any(|x| x < 0) { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="iter_all_positive",
        category="iterator",
        expected_result="equivalent",
        description="Check if all values in range are positive",
        c_source="""
int all_positive(int lo, int hi) {
    for (int i = lo; i < hi; i++) if (i <= 0) return 0;
    return 1;
}
""",
        rust_source="""
pub fn all_positive(lo: i32, hi: i32) -> i32 {
    if (lo..hi).all(|x| x > 0) { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="iter_product",
        category="iterator",
        expected_result="divergent",
        description="Product of range — overflow",
        divergence_kind="signed_overflow",
        c_source="""
int product(int n) {
    int p = 1;
    for (int i = 1; i <= n; i++) p *= i;
    return p;
}
""",
        rust_source="""
pub fn product(n: i32) -> i32 {
    (1..=n).fold(1i32, |acc, x| acc.wrapping_mul(x))
}
""",
    ),
    BenchmarkPair(
        name="iter_min_value",
        category="iterator",
        expected_result="equivalent",
        description="Find minimum in range",
        c_source="""
int min_in_range(int lo, int hi) {
    if (lo >= hi) return 0;
    return lo;
}
""",
        rust_source="""
pub fn min_in_range(lo: i32, hi: i32) -> i32 {
    (lo..hi).min().unwrap_or(0)
}
""",
    ),
    BenchmarkPair(
        name="iter_sum_of_squares",
        category="iterator",
        expected_result="divergent",
        description="Sum of squares — overflow",
        divergence_kind="signed_overflow",
        c_source="""
int sum_sq(int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) s += i * i;
    return s;
}
""",
        rust_source="""
pub fn sum_sq(n: i32) -> i32 {
    (1..=n).map(|x| x.wrapping_mul(x)).fold(0i32, |a, b| a.wrapping_add(b))
}
""",
    ),
    BenchmarkPair(
        name="iter_fold_xor",
        category="iterator",
        expected_result="equivalent",
        description="XOR fold over range",
        c_source="""
int xor_range(int n) {
    int x = 0;
    for (int i = 0; i < n; i++) x ^= i;
    return x;
}
""",
        rust_source="""
pub fn xor_range(n: i32) -> i32 { (0..n).fold(0, |acc, x| acc ^ x) }
""",
    ),
    BenchmarkPair(
        name="iter_count_divisible",
        category="iterator",
        expected_result="equivalent",
        description="Count values divisible by k",
        c_source="""
int count_div(int n, int k) {
    if (k == 0) return 0;
    int c = 0;
    for (int i = 1; i <= n; i++) if (i % k == 0) c++;
    return c;
}
""",
        rust_source="""
pub fn count_div(n: i32, k: i32) -> i32 {
    if k == 0 { return 0; }
    (1..=n).filter(|&x| x % k == 0).count() as i32
}
""",
    ),
    BenchmarkPair(
        name="iter_max_consecutive",
        category="iterator",
        expected_result="equivalent",
        description="Max of consecutive differences",
        c_source="""
int max_pair_sum(int n) {
    int mx = 0;
    for (int i = 0; i < n - 1; i++) {
        int s = i + (i + 1);
        if (s > mx) mx = s;
    }
    return mx;
}
""",
        rust_source="""
pub fn max_pair_sum(n: i32) -> i32 {
    if n < 2 { return 0; }
    (0..n-1).map(|i| i + (i + 1)).max().unwrap_or(0)
}
""",
    ),
    BenchmarkPair(
        name="iter_enumerate_sum",
        category="iterator",
        expected_result="equivalent",
        description="Sum of indices times values",
        c_source="""
int indexed_sum(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += i * i;
    return s;
}
""",
        rust_source="""
pub fn indexed_sum(n: i32) -> i32 {
    (0..n).enumerate().map(|(i, _)| (i as i32) * (i as i32)).sum()
}
""",
    ),
    BenchmarkPair(
        name="iter_zip_sum",
        category="iterator",
        expected_result="equivalent",
        description="Sum of pairwise products of two ranges",
        c_source="""
int zip_product_sum(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += i * (n - 1 - i);
    return s;
}
""",
        rust_source="""
pub fn zip_product_sum(n: i32) -> i32 {
    (0..n).zip((0..n).rev()).map(|(a, b)| a * b).sum()
}
""",
    ),
    BenchmarkPair(
        name="iter_take_while_sum",
        category="iterator",
        expected_result="equivalent",
        description="Sum while condition holds",
        c_source="""
int sum_while_pos(int start, int n) {
    int s = 0;
    for (int i = start; i < n && i >= 0; i++) s += i;
    return s;
}
""",
        rust_source="""
pub fn sum_while_pos(start: i32, n: i32) -> i32 {
    (start..n).take_while(|&x| x >= 0).sum()
}
""",
    ),
    BenchmarkPair(
        name="iter_skip_take",
        category="iterator",
        expected_result="equivalent",
        description="Sum of subrange",
        c_source="""
int sum_subrange(int lo, int hi, int skip, int take) {
    int s = 0, skipped = 0, taken = 0;
    for (int i = lo; i < hi && taken < take; i++) {
        if (skipped < skip) { skipped++; continue; }
        s += i; taken++;
    }
    return s;
}
""",
        rust_source="""
pub fn sum_subrange(lo: i32, hi: i32, skip: i32, take: i32) -> i32 {
    (lo..hi).skip(skip as usize).take(take as usize).sum()
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category: Type casting edge cases (15 pairs)
# ---------------------------------------------------------------------------

CAST_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="cast_signed_to_unsigned",
        category="cast",
        expected_result="equivalent",
        description="Signed to unsigned cast — identical bit pattern",
        c_source="""
unsigned int s2u(int x) { return (unsigned int)x; }
int cast_test(int x) { return (int)s2u(x); }
""",
        rust_source="""
pub fn cast_test(x: i32) -> i32 { (x as u32) as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_truncation",
        category="cast",
        expected_result="equivalent",
        description="Integer truncation — preserve low bits",
        c_source="""
int trunc_to_byte(int x) { return (signed char)x; }
""",
        rust_source="""
pub fn trunc_to_byte(x: i32) -> i32 { (x as i8) as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_zero_extend",
        category="cast",
        expected_result="equivalent",
        description="Zero extension from u8 to u32",
        c_source="""
unsigned int zext(unsigned char x) { return (unsigned int)x; }
int zext_test(int x) { return (int)zext((unsigned char)x); }
""",
        rust_source="""
pub fn zext_test(x: i32) -> i32 { ((x as u8) as u32) as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_sign_extend",
        category="cast",
        expected_result="equivalent",
        description="Sign extension from i8 to i32",
        c_source="""
int sext(signed char x) { return (int)x; }
int sext_test(int x) { return sext((signed char)x); }
""",
        rust_source="""
pub fn sext_test(x: i32) -> i32 { (x as i8) as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_widening_mul",
        category="cast",
        expected_result="equivalent",
        description="Widening multiplication via cast",
        c_source="""
long long wmul(int a, int b) { return (long long)a * (long long)b; }
int wmul_lo(int a, int b) { return (int)wmul(a, b); }
""",
        rust_source="""
pub fn wmul_lo(a: i32, b: i32) -> i32 {
    ((a as i64) * (b as i64)) as i32
}
""",
    ),
    BenchmarkPair(
        name="cast_bool_to_int",
        category="cast",
        expected_result="equivalent",
        description="Boolean to integer conversion",
        c_source="""
int bool_to_int(int x) { return x != 0; }
""",
        rust_source="""
pub fn bool_to_int(x: i32) -> i32 { if x != 0 { 1 } else { 0 } }
""",
    ),
    BenchmarkPair(
        name="cast_int_to_bool_to_int",
        category="cast",
        expected_result="equivalent",
        description="Int→bool→int roundtrip",
        c_source="""
int normalize_bool(int x) { return !!x; }
""",
        rust_source="""
pub fn normalize_bool(x: i32) -> i32 { if x != 0 { 1 } else { 0 } }
""",
    ),
    BenchmarkPair(
        name="cast_narrowing_unsigned",
        category="cast",
        expected_result="equivalent",
        description="Narrowing cast preserves low bits",
        c_source="""
unsigned char narrow(unsigned int x) { return (unsigned char)x; }
int narrow_test(int x) { return (int)narrow((unsigned int)x); }
""",
        rust_source="""
pub fn narrow_test(x: i32) -> i32 { ((x as u32) as u8) as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_char_to_int_signed",
        category="cast",
        expected_result="equivalent",
        description="Char to int with sign extension",
        c_source="""
int char_val(char c) { return (int)c; }
""",
        rust_source="""
pub fn char_val(c: i8) -> i32 { c as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_double_to_float",
        category="cast",
        expected_result="equivalent",
        description="Double to float narrowing",
        c_source="""
float d2f(double x) { return (float)x; }
int d2f_bits(double x) { float f = d2f(x); return *(int*)&f; }
""",
        rust_source="""
pub fn d2f_bits(x: f64) -> i32 { (x as f32).to_bits() as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_size_t_to_int",
        category="cast",
        expected_result="conditional",
        description="size_t to int — truncation on 64-bit",
        divergence_kind="truncation",
        c_source="""
int size_to_int(unsigned long s) { return (int)s; }
""",
        rust_source="""
pub fn size_to_int(s: u64) -> i32 { s as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_enum_to_int",
        category="cast",
        expected_result="equivalent",
        description="Enum variant to integer",
        c_source="""
enum Dir { N=0, S=1, E=2, W=3 };
int dir_val(enum Dir d) { return (int)d; }
""",
        rust_source="""
pub fn dir_val(d: i32) -> i32 { d }
""",
    ),
    BenchmarkPair(
        name="cast_pointer_to_int_size",
        category="cast",
        expected_result="equivalent",
        description="Pointer-sized integer operations",
        c_source="""
int ptr_arith(long p, int offset) { return (int)(p + offset); }
""",
        rust_source="""
pub fn ptr_arith(p: i64, offset: i32) -> i32 { (p + offset as i64) as i32 }
""",
    ),
    BenchmarkPair(
        name="cast_chain",
        category="cast",
        expected_result="equivalent",
        description="Chain of casts — final result identical",
        c_source="""
int cast_chain(int x) {
    return (int)(short)(char)(long long)(unsigned int)x;
}
""",
        rust_source="""
pub fn cast_chain(x: i32) -> i32 {
    ((((x as u32) as i64) as i8) as i16) as i32
}
""",
    ),
    BenchmarkPair(
        name="cast_conditional_widen",
        category="cast",
        expected_result="equivalent",
        description="Conditional widening for safe arithmetic",
        c_source="""
int safe_add(int a, int b) {
    long long r = (long long)a + (long long)b;
    if (r > 2147483647) return 2147483647;
    if (r < -2147483648LL) return -2147483647 - 1;
    return (int)r;
}
""",
        rust_source="""
pub fn safe_add(a: i32, b: i32) -> i32 {
    let r = (a as i64) + (b as i64);
    if r > i32::MAX as i64 { i32::MAX }
    else if r < i32::MIN as i64 { i32::MIN }
    else { r as i32 }
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category: Compound expressions (15 pairs)
# ---------------------------------------------------------------------------

COMPOUND_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="compound_ternary_chain",
        category="compound",
        expected_result="equivalent",
        description="Chained ternary operators",
        c_source="""
int classify(int x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
""",
        rust_source="""
pub fn classify(x: i32) -> i32 { x.signum() }
""",
    ),
    BenchmarkPair(
        name="compound_short_circuit_and",
        category="compound",
        expected_result="equivalent",
        description="Short-circuit AND evaluation",
        c_source="""
int safe_check(int x, int y) { return x != 0 && y / x > 5; }
""",
        rust_source="""
pub fn safe_check(x: i32, y: i32) -> i32 {
    if x != 0 && y / x > 5 { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="compound_comma_operator",
        category="compound",
        expected_result="equivalent",
        description="C comma operator sequence",
        c_source="""
int comma_seq(int x) { return (x += 1, x *= 2, x - 3); }
""",
        rust_source="""
pub fn comma_seq(x: i32) -> i32 { let x = x + 1; let x = x * 2; x - 3 }
""",
    ),
    BenchmarkPair(
        name="compound_nested_conditional",
        category="compound",
        expected_result="equivalent",
        description="Deeply nested conditionals",
        c_source="""
int deep_cond(int a, int b, int c) {
    if (a > 0) {
        if (b > 0) { return a + b; }
        else { return a - b; }
    } else {
        if (c > 0) { return c; }
        else { return 0; }
    }
}
""",
        rust_source="""
pub fn deep_cond(a: i32, b: i32, c: i32) -> i32 {
    if a > 0 {
        if b > 0 { a + b } else { a - b }
    } else {
        if c > 0 { c } else { 0 }
    }
}
""",
    ),
    BenchmarkPair(
        name="compound_bitwise_extract",
        category="compound",
        expected_result="equivalent",
        description="Extract byte from 32-bit integer",
        c_source="""
int get_byte(unsigned int x, int n) { return (x >> (n * 8)) & 0xFF; }
""",
        rust_source="""
pub fn get_byte(x: u32, n: i32) -> i32 { ((x >> (n * 8)) & 0xFF) as i32 }
""",
    ),
    BenchmarkPair(
        name="compound_arithmetic_identity",
        category="compound",
        expected_result="equivalent",
        description="Arithmetic identity: a*b + a*c = a*(b+c)",
        c_source="""
int distribute(int a, int b, int c) { return a * b + a * c; }
""",
        rust_source="""
pub fn distribute(a: i32, b: i32, c: i32) -> i32 { a * (b + c) }
""",
    ),
    BenchmarkPair(
        name="compound_demorgan",
        category="compound",
        expected_result="equivalent",
        description="De Morgan's law equivalence",
        c_source="""
int demorgan(int a, int b) { return !(a && b); }
""",
        rust_source="""
pub fn demorgan(a: i32, b: i32) -> i32 {
    if a == 0 || b == 0 { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="compound_mixed_ops",
        category="compound",
        expected_result="divergent",
        description="Mixed arithmetic — overflow possible",
        divergence_kind="signed_overflow",
        c_source="""
int mixed(int a, int b) { return (a + b) * (a - b); }
""",
        rust_source="""
pub fn mixed(a: i32, b: i32) -> i32 {
    a.wrapping_add(b).wrapping_mul(a.wrapping_sub(b))
}
""",
    ),
    BenchmarkPair(
        name="compound_predicate_combine",
        category="compound",
        expected_result="equivalent",
        description="Combined predicates",
        c_source="""
int in_range(int x, int lo, int hi) { return x >= lo && x <= hi; }
""",
        rust_source="""
pub fn in_range(x: i32, lo: i32, hi: i32) -> i32 {
    if (lo..=hi).contains(&x) { 1 } else { 0 }
}
""",
    ),
    BenchmarkPair(
        name="compound_abs_diff",
        category="compound",
        expected_result="equivalent",
        description="Absolute difference via branchless trick",
        c_source="""
unsigned int abs_diff(unsigned int a, unsigned int b) {
    return a > b ? a - b : b - a;
}
int abs_diff_test(int a, int b) { return (int)abs_diff((unsigned int)a, (unsigned int)b); }
""",
        rust_source="""
pub fn abs_diff_test(a: i32, b: i32) -> i32 {
    ((a as u32).wrapping_sub(b as u32).max((b as u32).wrapping_sub(a as u32))) as i32
}
""",
    ),
    BenchmarkPair(
        name="compound_min3",
        category="compound",
        expected_result="equivalent",
        description="Minimum of three values",
        c_source="""
int min3(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}
""",
        rust_source="""
pub fn min3(a: i32, b: i32, c: i32) -> i32 { a.min(b).min(c) }
""",
    ),
    BenchmarkPair(
        name="compound_median3",
        category="compound",
        expected_result="equivalent",
        description="Median of three values",
        c_source="""
int median3(int a, int b, int c) {
    if ((a >= b && a <= c) || (a <= b && a >= c)) return a;
    if ((b >= a && b <= c) || (b <= a && b >= c)) return b;
    return c;
}
""",
        rust_source="""
pub fn median3(a: i32, b: i32, c: i32) -> i32 {
    let mut arr = [a, b, c];
    arr.sort();
    arr[1]
}
""",
    ),
    BenchmarkPair(
        name="compound_polynomial_eval",
        category="compound",
        expected_result="divergent",
        description="Horner's method polynomial — overflow",
        divergence_kind="signed_overflow",
        c_source="""
int poly(int x) { return 3*x*x + 2*x + 1; }
""",
        rust_source="""
pub fn poly(x: i32) -> i32 {
    3i32.wrapping_mul(x).wrapping_mul(x).wrapping_add(2i32.wrapping_mul(x)).wrapping_add(1)
}
""",
    ),
    BenchmarkPair(
        name="compound_collatz_step",
        category="compound",
        expected_result="equivalent",
        description="Single Collatz step",
        c_source="""
int collatz_step(int n) {
    if (n % 2 == 0) return n / 2;
    return 3 * n + 1;
}
""",
        rust_source="""
pub fn collatz_step(n: i32) -> i32 {
    if n % 2 == 0 { n / 2 } else { 3 * n + 1 }
}
""",
    ),
    BenchmarkPair(
        name="compound_digit_sum",
        category="compound",
        expected_result="equivalent",
        description="Sum of digits",
        c_source="""
int digit_sum(int n) {
    int s = 0;
    if (n < 0) n = -n;
    while (n > 0) { s += n % 10; n /= 10; }
    return s;
}
""",
        rust_source="""
pub fn digit_sum(mut n: i32) -> i32 {
    let mut s = 0;
    if n < 0 { n = -n; }
    while n > 0 { s += n % 10; n /= 10; }
    s
}
""",
    ),
]


# ---------------------------------------------------------------------------
# Category: Control flow patterns (10 pairs)
# ---------------------------------------------------------------------------

CONTROL_FLOW_PAIRS: List[BenchmarkPair] = [
    BenchmarkPair(
        name="cf_early_return",
        category="control_flow",
        expected_result="equivalent",
        description="Early return vs single return point",
        c_source="""
int classify_sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
""",
        rust_source="""
pub fn classify_sign(x: i32) -> i32 {
    match x.cmp(&0) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}
""",
    ),
    BenchmarkPair(
        name="cf_loop_vs_recursion",
        category="control_flow",
        expected_result="equivalent",
        description="Iterative vs recursive (tail-call style)",
        c_source="""
int sum_loop(int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) s += i;
    return s;
}
""",
        rust_source="""
pub fn sum_loop(n: i32) -> i32 {
    let mut s = 0;
    let mut i = 1;
    while i <= n { s += i; i += 1; }
    s
}
""",
    ),
    BenchmarkPair(
        name="cf_goto_vs_loop",
        category="control_flow",
        expected_result="equivalent",
        description="C goto pattern vs Rust loop/break",
        c_source="""
int find_zero(int a, int b, int c) {
    if (a == 0) goto found;
    if (b == 0) goto found;
    if (c == 0) goto found;
    return -1;
found:
    return 1;
}
""",
        rust_source="""
pub fn find_zero(a: i32, b: i32, c: i32) -> i32 {
    if a == 0 || b == 0 || c == 0 { 1 } else { -1 }
}
""",
    ),
    BenchmarkPair(
        name="cf_do_while_vs_loop",
        category="control_flow",
        expected_result="equivalent",
        description="C do-while vs Rust loop-break",
        c_source="""
int count_halvings(int n) {
    int c = 0;
    do { n /= 2; c++; } while (n > 0);
    return c;
}
""",
        rust_source="""
pub fn count_halvings(mut n: i32) -> i32 {
    let mut c = 0;
    loop { n /= 2; c += 1; if n <= 0 { break; } }
    c
}
""",
    ),
    BenchmarkPair(
        name="cf_nested_break",
        category="control_flow",
        expected_result="equivalent",
        description="Nested loop break — C break vs Rust labeled break",
        c_source="""
int nested_find(int target) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (i * 10 + j == target) return i * 10 + j;
        }
    }
    return -1;
}
""",
        rust_source="""
pub fn nested_find(target: i32) -> i32 {
    'outer: for i in 0..10 {
        for j in 0..10 {
            if i * 10 + j == target { return i * 10 + j; }
        }
    }
    -1
}
""",
    ),
    BenchmarkPair(
        name="cf_switch_fallthrough_sum",
        category="control_flow",
        expected_result="divergent",
        description="Switch fallthrough accumulation",
        divergence_kind="control_flow",
        c_source="""
int accum(int level) {
    int total = 0;
    switch(level) {
        case 3: total += 100;
        case 2: total += 10;
        case 1: total += 1;
    }
    return total;
}
""",
        rust_source="""
pub fn accum(level: i32) -> i32 {
    match level {
        3 => 100,
        2 => 10,
        1 => 1,
        _ => 0,
    }
}
""",
    ),
    BenchmarkPair(
        name="cf_for_vs_while",
        category="control_flow",
        expected_result="equivalent",
        description="For loop vs while loop — same semantics",
        c_source="""
int sum_range(int lo, int hi) {
    int s = 0;
    for (int i = lo; i < hi; i++) s += i;
    return s;
}
""",
        rust_source="""
pub fn sum_range(lo: i32, hi: i32) -> i32 {
    let mut s = 0;
    let mut i = lo;
    while i < hi { s += i; i += 1; }
    s
}
""",
    ),
    BenchmarkPair(
        name="cf_continue_vs_if",
        category="control_flow",
        expected_result="equivalent",
        description="Continue statement vs guarded body",
        c_source="""
int sum_odd(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) continue;
        s += i;
    }
    return s;
}
""",
        rust_source="""
pub fn sum_odd(n: i32) -> i32 {
    let mut s = 0;
    for i in 0..n {
        if i % 2 != 0 { s += i; }
    }
    s
}
""",
    ),
    BenchmarkPair(
        name="cf_ternary_vs_if",
        category="control_flow",
        expected_result="equivalent",
        description="Ternary operator vs if-else expression",
        c_source="""
int abs_ternary(int x) { return x >= 0 ? x : -x; }
""",
        rust_source="""
pub fn abs_ternary(x: i32) -> i32 { if x >= 0 { x } else { -x } }
""",
    ),
    BenchmarkPair(
        name="cf_multiple_returns",
        category="control_flow",
        expected_result="equivalent",
        description="Multiple return paths with different conditions",
        c_source="""
int grade(int score) {
    if (score >= 90) return 4;
    if (score >= 80) return 3;
    if (score >= 70) return 2;
    if (score >= 60) return 1;
    return 0;
}
""",
        rust_source="""
pub fn grade(score: i32) -> i32 {
    match score {
        s if s >= 90 => 4,
        s if s >= 80 => 3,
        s if s >= 70 => 2,
        s if s >= 60 => 1,
        _ => 0,
    }
}
""",
    ),
]


# ---------------------------------------------------------------------------
# All expanded benchmarks
# ---------------------------------------------------------------------------

EXPANDED_BENCHMARKS: List[BenchmarkPair] = (
    STRUCT_PAIRS +
    ENUM_PAIRS +
    FLOAT_PAIRS +
    C2RUST_PAIRS +
    ITERATOR_PAIRS +
    CAST_PAIRS +
    COMPOUND_PAIRS +
    CONTROL_FLOW_PAIRS
)


def get_expanded_benchmarks() -> List[BenchmarkPair]:
    """Return all expanded benchmark pairs."""
    return list(EXPANDED_BENCHMARKS)


def get_expanded_by_category(category: str) -> List[BenchmarkPair]:
    """Return expanded benchmarks filtered by category."""
    return [b for b in EXPANDED_BENCHMARKS if b.category == category]


def get_expanded_categories() -> List[str]:
    """Return sorted list of expanded benchmark categories."""
    return sorted(set(b.category for b in EXPANDED_BENCHMARKS))
