"""
CEGAR Benchmark Pairs: C functions with deliberately-buggy Rust translations.

Each pair consists of a valid C11 function and a syntactically-valid but
semantically-incorrect Rust translation. The CEGAR loop must identify the
divergence and repair the Rust code.

Categories:
  1. signed_overflow   (15 pairs)
  2. unsigned_wrap      (10 pairs)
  3. division           (12 pairs)
  4. shift              (12 pairs)
  5. cast               (15 pairs)
  6. pointer_arithmetic (12 pairs)
  7. struct_access      (10 pairs)
  8. bitwise            (10 pairs)
  9. control_flow       (12 pairs)
 10. mixed              (12 pairs)

Total: 120 pairs
"""

CEGAR_BENCHMARK_PAIRS = [

    # =========================================================================
    # CATEGORY 1: signed_overflow (15 pairs)
    # =========================================================================

    {
        "name": "add_i32",
        "category": "signed_overflow",
        "c_code": "int add_i32(int a, int b) { return a + b; }",
        "buggy_rust": "pub fn add_i32(a: i32, b: i32) -> i32 { a + b }",
        "bug_class": "signed_overflow",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Simple signed addition; C has UB on overflow, Rust panics in debug mode instead of wrapping.",
    },
    {
        "name": "sub_i32",
        "category": "signed_overflow",
        "c_code": "int sub_i32(int a, int b) { return a - b; }",
        "buggy_rust": "pub fn sub_i32(a: i32, b: i32) -> i32 { a - b }",
        "bug_class": "signed_overflow",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Simple signed subtraction; C has UB on overflow, Rust panics.",
    },
    {
        "name": "mul_i32",
        "category": "signed_overflow",
        "c_code": "int mul_i32(int a, int b) { return a * b; }",
        "buggy_rust": "pub fn mul_i32(a: i32, b: i32) -> i32 { a * b }",
        "bug_class": "signed_overflow",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Signed multiplication; both sides have overflow risk.",
    },
    {
        "name": "negate_i32",
        "category": "signed_overflow",
        "c_code": "int negate_i32(int x) { return -x; }",
        "buggy_rust": "pub fn negate_i32(x: i32) -> i32 { -x }",
        "bug_class": "signed_overflow",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Negation of INT_MIN is UB in C; Rust panics in debug.",
    },
    {
        "name": "abs_val",
        "category": "signed_overflow",
        "c_code": (
            "int abs_val(int x) {\n"
            "    return x < 0 ? -x : x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn abs_val(x: i32) -> i32 {\n"
            "    if x < 0 { -x } else { x }\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Absolute value; negating INT_MIN overflows.",
    },
    {
        "name": "sum_pair",
        "category": "signed_overflow",
        "c_code": (
            "int sum_pair(int a, int b) {\n"
            "    int s = a + b;\n"
            "    return s * 2;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn sum_pair(a: i32, b: i32) -> i32 {\n"
            "    let s = a + b;\n"
            "    s * 2\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Double overflow: addition then multiply by 2.",
    },
    {
        "name": "triple_add",
        "category": "signed_overflow",
        "c_code": (
            "int triple_add(int a, int b, int c) {\n"
            "    return a + b + c;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn triple_add(a: i32, b: i32, c: i32) -> i32 {\n"
            "    a + b + c\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Three-operand signed addition; intermediate overflow possible.",
    },
    {
        "name": "square_i32",
        "category": "signed_overflow",
        "c_code": "int square_i32(int x) { return x * x; }",
        "buggy_rust": "pub fn square_i32(x: i32) -> i32 { x * x }",
        "bug_class": "signed_overflow",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Squaring a value; overflow on large magnitudes.",
    },
    {
        "name": "factorial_iter",
        "category": "signed_overflow",
        "c_code": (
            "int factorial_iter(int n) {\n"
            "    int result = 1;\n"
            "    for (int i = 2; i <= n; i++) {\n"
            "        result *= i;\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn factorial_iter(n: i32) -> i32 {\n"
            "    let mut result: i32 = 1;\n"
            "    for i in 2..=n {\n"
            "        result *= i;\n"
            "    }\n"
            "    result\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Iterative factorial; overflows quickly, needs wrapping_mul.",
    },
    {
        "name": "fibonacci",
        "category": "signed_overflow",
        "c_code": (
            "int fibonacci(int n) {\n"
            "    int a = 0, b = 1;\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        int t = a + b;\n"
            "        a = b;\n"
            "        b = t;\n"
            "    }\n"
            "    return a;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn fibonacci(n: i32) -> i32 {\n"
            "    let mut a: i32 = 0;\n"
            "    let mut b: i32 = 1;\n"
            "    for _ in 0..n {\n"
            "        let t = a + b;\n"
            "        a = b;\n"
            "        b = t;\n"
            "    }\n"
            "    a\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Fibonacci; overflows for large n, needs wrapping_add.",
    },
    {
        "name": "power_i32",
        "category": "signed_overflow",
        "c_code": (
            "int power_i32(int base, int exp) {\n"
            "    int result = 1;\n"
            "    for (int i = 0; i < exp; i++) {\n"
            "        result *= base;\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn power_i32(base: i32, exp: i32) -> i32 {\n"
            "    let mut result: i32 = 1;\n"
            "    for _ in 0..exp {\n"
            "        result *= base;\n"
            "    }\n"
            "    result\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Integer exponentiation; overflow accumulates in the loop.",
    },
    {
        "name": "accumulate",
        "category": "signed_overflow",
        "c_code": (
            "int accumulate(int start, int step, int count) {\n"
            "    int acc = start;\n"
            "    for (int i = 0; i < count; i++) {\n"
            "        acc += step;\n"
            "    }\n"
            "    return acc;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn accumulate(start: i32, step: i32, count: i32) -> i32 {\n"
            "    let mut acc = start;\n"
            "    for _ in 0..count {\n"
            "        acc += step;\n"
            "    }\n"
            "    acc\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Running accumulation; repeated addition can overflow.",
    },
    {
        "name": "diff_squared",
        "category": "signed_overflow",
        "c_code": (
            "int diff_squared(int a, int b) {\n"
            "    int d = a - b;\n"
            "    return d * d;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn diff_squared(a: i32, b: i32) -> i32 {\n"
            "    let d = a - b;\n"
            "    d * d\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Squared difference; subtraction and multiplication can each overflow.",
    },
    {
        "name": "sum_of_squares",
        "category": "signed_overflow",
        "c_code": (
            "int sum_of_squares(int a, int b, int c) {\n"
            "    return a * a + b * b + c * c;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn sum_of_squares(a: i32, b: i32, c: i32) -> i32 {\n"
            "    a * a + b * b + c * c\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Sum of three squares; multiple overflow points.",
    },
    {
        "name": "lerp_int",
        "category": "signed_overflow",
        "c_code": (
            "int lerp_int(int a, int b, int t) {\n"
            "    return a + (b - a) * t / 100;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn lerp_int(a: i32, b: i32, t: i32) -> i32 {\n"
            "    a + (b - a) * t / 100\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "hard",
        "has_ub": True,
        "description": "Integer linear interpolation; intermediate overflow in (b-a)*t.",
    },

    # =========================================================================
    # CATEGORY 2: unsigned_wrap (10 pairs)
    # =========================================================================

    {
        "name": "unsigned_sub",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int unsigned_sub(unsigned int a, unsigned int b) {\n"
            "    return a - b;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_sub(a: u32, b: u32) -> u32 {\n"
            "    a - b\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Unsigned subtraction wraps in C; Rust panics in debug mode.",
    },
    {
        "name": "unsigned_decrement",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int unsigned_decrement(unsigned int x) {\n"
            "    return x - 1;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_decrement(x: u32) -> u32 {\n"
            "    x - 1\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Decrementing 0u wraps to UINT_MAX in C; Rust panics.",
    },
    {
        "name": "unsigned_distance",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int unsigned_distance(unsigned int a, unsigned int b) {\n"
            "    return a > b ? a - b : b - a;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_distance(a: u32, b: u32) -> u32 {\n"
            "    if a > b { a - b } else { b - a }\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Unsigned distance; guarded but Rust still uses checked sub.",
    },
    {
        "name": "ring_counter",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int ring_counter(unsigned int val, unsigned int limit) {\n"
            "    return (val + 1) % limit;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn ring_counter(val: u32, limit: u32) -> u32 {\n"
            "    (val + 1) % limit\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Ring counter with modular increment; val+1 can wrap at UINT_MAX.",
    },
    {
        "name": "unsigned_mul_wrap",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int unsigned_mul_wrap(unsigned int a, unsigned int b) {\n"
            "    return a * b;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_mul_wrap(a: u32, b: u32) -> u32 {\n"
            "    a * b\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Unsigned multiplication wraps in C; Rust panics on overflow.",
    },
    {
        "name": "countdown_loop",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int countdown_loop(unsigned int n) {\n"
            "    unsigned int sum = 0;\n"
            "    for (unsigned int i = n; i != 0; i--) {\n"
            "        sum += i;\n"
            "    }\n"
            "    return sum;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn countdown_loop(n: u32) -> u32 {\n"
            "    let mut sum: u32 = 0;\n"
            "    let mut i = n;\n"
            "    while i != 0 {\n"
            "        sum += i;\n"
            "        i -= 1;\n"
            "    }\n"
            "    sum\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Countdown adding; sum can wrap, Rust panics.",
    },
    {
        "name": "unsigned_hash_combine",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int unsigned_hash_combine(unsigned int h1, unsigned int h2) {\n"
            "    return h1 * 31 + h2;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_hash_combine(h1: u32, h2: u32) -> u32 {\n"
            "    h1 * 31 + h2\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Hash combine relies on unsigned wrapping; Rust panics.",
    },
    {
        "name": "unsigned_neg_offset",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int unsigned_neg_offset(unsigned int base, unsigned int offset) {\n"
            "    return base + (0u - offset);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_neg_offset(base: u32, offset: u32) -> u32 {\n"
            "    base + (0u32 - offset)\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "hard",
        "has_ub": False,
        "description": "Uses unsigned negation trick (0 - x); Rust panics when offset > 0.",
    },
    {
        "name": "unsigned_avg_floor",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int unsigned_avg_floor(unsigned int a, unsigned int b) {\n"
            "    return (a + b) / 2;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_avg_floor(a: u32, b: u32) -> u32 {\n"
            "    (a + b) / 2\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Average via (a+b)/2; sum wraps in C, Rust panics.",
    },
    {
        "name": "unsigned_pow_mod",
        "category": "unsigned_wrap",
        "c_code": (
            "unsigned int unsigned_pow_mod(unsigned int base, unsigned int exp, unsigned int mod) {\n"
            "    unsigned int result = 1;\n"
            "    base = base % mod;\n"
            "    while (exp > 0) {\n"
            "        if (exp % 2 == 1)\n"
            "            result = result * base % mod;\n"
            "        exp = exp / 2;\n"
            "        base = base * base % mod;\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_pow_mod(mut base: u32, mut exp: u32, modulus: u32) -> u32 {\n"
            "    let mut result: u32 = 1;\n"
            "    base = base % modulus;\n"
            "    while exp > 0 {\n"
            "        if exp % 2 == 1 {\n"
            "            result = result * base % modulus;\n"
            "        }\n"
            "        exp = exp / 2;\n"
            "        base = base * base % modulus;\n"
            "    }\n"
            "    result\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "hard",
        "has_ub": False,
        "description": "Modular exponentiation; intermediate products wrap in C, Rust panics.",
    },

    # =========================================================================
    # CATEGORY 3: division (12 pairs)
    # =========================================================================

    {
        "name": "div_simple",
        "category": "division",
        "c_code": "int div_simple(int a, int b) { return a / b; }",
        "buggy_rust": "pub fn div_simple(a: i32, b: i32) -> i32 { a / b }",
        "bug_class": "division_by_zero",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Division; UB when b==0 in C, Rust panics.",
    },
    {
        "name": "mod_simple",
        "category": "division",
        "c_code": "int mod_simple(int a, int b) { return a % b; }",
        "buggy_rust": "pub fn mod_simple(a: i32, b: i32) -> i32 { a % b }",
        "bug_class": "division_by_zero",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Modulo; UB when b==0 in C, Rust panics.",
    },
    {
        "name": "int_min_div_neg1",
        "category": "division",
        "c_code": "int int_min_div_neg1(int a, int b) { return a / b; }",
        "buggy_rust": "pub fn int_min_div_neg1(a: i32, b: i32) -> i32 { a / b }",
        "bug_class": "division_by_zero",
        "difficulty": "medium",
        "has_ub": True,
        "description": "INT_MIN / -1 is UB in C (result unrepresentable); Rust panics.",
    },
    {
        "name": "safe_div_guarded",
        "category": "division",
        "c_code": (
            "int safe_div_guarded(int a, int b) {\n"
            "    if (b == 0) return 0;\n"
            "    return a / b;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn safe_div_guarded(a: i32, b: i32) -> i32 {\n"
            "    if b == 0 { return 0; }\n"
            "    a / b\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Guards zero but not INT_MIN/-1; Rust panics on the latter.",
    },
    {
        "name": "euclidean_gcd",
        "category": "division",
        "c_code": (
            "int euclidean_gcd(int a, int b) {\n"
            "    while (b != 0) {\n"
            "        int t = b;\n"
            "        b = a % b;\n"
            "        a = t;\n"
            "    }\n"
            "    return a;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn euclidean_gcd(mut a: i32, mut b: i32) -> i32 {\n"
            "    while b != 0 {\n"
            "        let t = b;\n"
            "        b = a % b;\n"
            "        a = t;\n"
            "    }\n"
            "    a\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "medium",
        "has_ub": True,
        "description": "GCD via Euclidean algorithm; INT_MIN % -1 is UB in C.",
    },
    {
        "name": "div_round_up",
        "category": "division",
        "c_code": (
            "int div_round_up(int n, int d) {\n"
            "    return (n + d - 1) / d;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn div_round_up(n: i32, d: i32) -> i32 {\n"
            "    (n + d - 1) / d\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Ceiling division; numerator can overflow, denominator can be zero.",
    },
    {
        "name": "unsigned_div",
        "category": "division",
        "c_code": (
            "unsigned int unsigned_div(unsigned int a, unsigned int b) {\n"
            "    return a / b;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_div(a: u32, b: u32) -> u32 {\n"
            "    a / b\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Unsigned division; UB on b==0 in C.",
    },
    {
        "name": "div_chain",
        "category": "division",
        "c_code": (
            "int div_chain(int a, int b, int c) {\n"
            "    return a / b / c;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn div_chain(a: i32, b: i32, c: i32) -> i32 {\n"
            "    a / b / c\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Chained division; both b and c can be zero.",
    },
    {
        "name": "mod_pos",
        "category": "division",
        "c_code": (
            "int mod_pos(int a, int m) {\n"
            "    int r = a % m;\n"
            "    return r < 0 ? r + m : r;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn mod_pos(a: i32, m: i32) -> i32 {\n"
            "    let r = a % m;\n"
            "    if r < 0 { r + m } else { r }\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Positive modulo; m==0 is UB in C, addition can overflow.",
    },
    {
        "name": "div_by_power2",
        "category": "division",
        "c_code": (
            "int div_by_power2(int x, int p) {\n"
            "    return x / (1 << p);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn div_by_power2(x: i32, p: i32) -> i32 {\n"
            "    x / (1 << p)\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "hard",
        "has_ub": True,
        "description": "Division by power of 2; shift can produce 0 or UB if p >= 31.",
    },
    {
        "name": "ratio_percent",
        "category": "division",
        "c_code": (
            "int ratio_percent(int num, int den) {\n"
            "    return num * 100 / den;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn ratio_percent(num: i32, den: i32) -> i32 {\n"
            "    num * 100 / den\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Percentage ratio; num*100 can overflow and den can be zero.",
    },
    {
        "name": "modular_inverse_attempt",
        "category": "division",
        "c_code": (
            "int modular_inverse_attempt(int a, int m) {\n"
            "    for (int x = 1; x < m; x++) {\n"
            "        if ((a * x) % m == 1) return x;\n"
            "    }\n"
            "    return -1;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn modular_inverse_attempt(a: i32, m: i32) -> i32 {\n"
            "    for x in 1..m {\n"
            "        if (a * x) % m == 1 { return x; }\n"
            "    }\n"
            "    -1\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "hard",
        "has_ub": True,
        "description": "Brute-force modular inverse; a*x can overflow, m==0 is UB.",
    },

    # =========================================================================
    # CATEGORY 4: shift (12 pairs)
    # =========================================================================

    {
        "name": "shl_basic",
        "category": "shift",
        "c_code": "int shl_basic(int x, int n) { return x << n; }",
        "buggy_rust": "pub fn shl_basic(x: i32, n: i32) -> i32 { x << n }",
        "bug_class": "shift_ub",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Left shift; UB when n >= 32 or n < 0 in C, Rust panics.",
    },
    {
        "name": "shr_basic",
        "category": "shift",
        "c_code": "int shr_basic(int x, int n) { return x >> n; }",
        "buggy_rust": "pub fn shr_basic(x: i32, n: i32) -> i32 { x >> n }",
        "bug_class": "shift_ub",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Right shift; UB when n >= 32 or n < 0 in C.",
    },
    {
        "name": "shl_unsigned",
        "category": "shift",
        "c_code": (
            "unsigned int shl_unsigned(unsigned int x, unsigned int n) {\n"
            "    return x << n;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn shl_unsigned(x: u32, n: u32) -> u32 {\n"
            "    x << n\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Unsigned left shift; UB when n >= 32 in C.",
    },
    {
        "name": "shift_mask",
        "category": "shift",
        "c_code": (
            "unsigned int shift_mask(unsigned int x, int bit) {\n"
            "    return x & (1u << bit);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn shift_mask(x: u32, bit: i32) -> u32 {\n"
            "    x & (1u32 << bit as u32)\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Bit test via mask; bit >= 32 is UB in C, negative bit is UB.",
    },
    {
        "name": "extract_byte",
        "category": "shift",
        "c_code": (
            "unsigned int extract_byte(unsigned int val, int byte_idx) {\n"
            "    return (val >> (byte_idx * 8)) & 0xFF;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn extract_byte(val: u32, byte_idx: i32) -> u32 {\n"
            "    (val >> (byte_idx * 8) as u32) & 0xFF\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Byte extraction; byte_idx > 3 causes shift >= 32 which is UB.",
    },
    {
        "name": "rotate_left",
        "category": "shift",
        "c_code": (
            "unsigned int rotate_left(unsigned int x, int n) {\n"
            "    return (x << n) | (x >> (32 - n));\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn rotate_left(x: u32, n: i32) -> u32 {\n"
            "    (x << n as u32) | (x >> (32 - n) as u32)\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Rotate left; both shifts are UB if n == 0 or n >= 32.",
    },
    {
        "name": "rotate_right",
        "category": "shift",
        "c_code": (
            "unsigned int rotate_right(unsigned int x, int n) {\n"
            "    return (x >> n) | (x << (32 - n));\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn rotate_right(x: u32, n: i32) -> u32 {\n"
            "    (x >> n as u32) | (x << (32 - n) as u32)\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Rotate right; shift amounts can be 0 or 32 causing UB.",
    },
    {
        "name": "set_bit",
        "category": "shift",
        "c_code": (
            "unsigned int set_bit(unsigned int x, int pos) {\n"
            "    return x | (1u << pos);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn set_bit(x: u32, pos: i32) -> u32 {\n"
            "    x | (1u32 << pos as u32)\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Set a single bit; pos >= 32 is UB in C.",
    },
    {
        "name": "clear_bit",
        "category": "shift",
        "c_code": (
            "unsigned int clear_bit(unsigned int x, int pos) {\n"
            "    return x & ~(1u << pos);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn clear_bit(x: u32, pos: i32) -> u32 {\n"
            "    x & !(1u32 << pos as u32)\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "easy",
        "has_ub": True,
        "description": "Clear a single bit; pos >= 32 causes UB shift.",
    },
    {
        "name": "bit_range_mask",
        "category": "shift",
        "c_code": (
            "unsigned int bit_range_mask(int lo, int hi) {\n"
            "    return ((1u << (hi - lo + 1)) - 1) << lo;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn bit_range_mask(lo: i32, hi: i32) -> u32 {\n"
            "    ((1u32 << (hi - lo + 1) as u32) - 1) << lo as u32\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "hard",
        "has_ub": True,
        "description": "Contiguous bit mask; multiple UB-inducing shift amounts possible.",
    },
    {
        "name": "variable_shift_add",
        "category": "shift",
        "c_code": (
            "int variable_shift_add(int a, int b, int shift) {\n"
            "    return a + (b << shift);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn variable_shift_add(a: i32, b: i32, shift: i32) -> i32 {\n"
            "    a + (b << shift)\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Combined shift and add; shift UB plus overflow.",
    },
    {
        "name": "sign_extend_shift",
        "category": "shift",
        "c_code": (
            "int sign_extend_shift(int val, int bits) {\n"
            "    int shift = 32 - bits;\n"
            "    return (val << shift) >> shift;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn sign_extend_shift(val: i32, bits: i32) -> i32 {\n"
            "    let shift = 32 - bits;\n"
            "    (val << shift) >> shift\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "hard",
        "has_ub": True,
        "description": "Sign extension via shift pair; UB if bits <= 0 or bits > 32.",
    },

    # =========================================================================
    # CATEGORY 5: cast (15 pairs)
    # =========================================================================

    {
        "name": "i32_to_i8",
        "category": "cast",
        "c_code": (
            "signed char i32_to_i8(int x) {\n"
            "    return (signed char)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn i32_to_i8(x: i32) -> i8 {\n"
            "    x as i8\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Narrowing cast i32->i8; C truncates, Rust 'as' also truncates but semantics differ for negative values in some contexts.",
    },
    {
        "name": "i32_to_u8",
        "category": "cast",
        "c_code": (
            "unsigned char i32_to_u8(int x) {\n"
            "    return (unsigned char)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn i32_to_u8(x: i32) -> u8 {\n"
            "    x as u8\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Narrowing signed-to-unsigned cast; C converts via modular arithmetic.",
    },
    {
        "name": "u32_to_i32",
        "category": "cast",
        "c_code": (
            "int u32_to_i32(unsigned int x) {\n"
            "    return (int)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn u32_to_i32(x: u32) -> i32 {\n"
            "    x as i32\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "easy",
        "has_ub": False,
        "description": "u32 to i32; values > INT_MAX get implementation-defined behavior in C.",
    },
    {
        "name": "i64_to_i32",
        "category": "cast",
        "c_code": (
            "int i64_to_i32(long long x) {\n"
            "    return (int)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn i64_to_i32(x: i64) -> i32 {\n"
            "    x as i32\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Narrowing 64->32; truncates upper bits.",
    },
    {
        "name": "i8_to_u32",
        "category": "cast",
        "c_code": (
            "unsigned int i8_to_u32(signed char x) {\n"
            "    return (unsigned int)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn i8_to_u32(x: i8) -> u32 {\n"
            "    x as u32\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Sign-extends i8 to i32 then converts to u32; Rust 'as' chain differs.",
    },
    {
        "name": "u16_to_i8",
        "category": "cast",
        "c_code": (
            "signed char u16_to_i8(unsigned short x) {\n"
            "    return (signed char)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn u16_to_i8(x: u16) -> i8 {\n"
            "    x as i8\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Double narrowing and sign change u16->i8.",
    },
    {
        "name": "double_cast_widen",
        "category": "cast",
        "c_code": (
            "long long double_cast_widen(signed char x) {\n"
            "    return (long long)(unsigned char)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn double_cast_widen(x: i8) -> i64 {\n"
            "    (x as u8) as i64\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Cast i8 -> u8 -> i64; sign loss then widen. Negative values differ.",
    },
    {
        "name": "cast_mul_widen",
        "category": "cast",
        "c_code": (
            "long long cast_mul_widen(int a, int b) {\n"
            "    return (long long)a * b;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn cast_mul_widen(a: i32, b: i32) -> i64 {\n"
            "    (a as i64) * (b as i64)\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Widening multiply; C implicitly widens, Rust needs both operands cast.",
    },
    {
        "name": "truncate_and_add",
        "category": "cast",
        "c_code": (
            "unsigned char truncate_and_add(int a, int b) {\n"
            "    return (unsigned char)(a + b);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn truncate_and_add(a: i32, b: i32) -> u8 {\n"
            "    (a + b) as u8\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Add then truncate; a+b can overflow in C (UB), then truncates.",
    },
    {
        "name": "sign_preserving_cast",
        "category": "cast",
        "c_code": (
            "int sign_preserving_cast(unsigned short x) {\n"
            "    return (int)(short)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn sign_preserving_cast(x: u16) -> i32 {\n"
            "    (x as i16) as i32\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "medium",
        "has_ub": False,
        "description": "u16->i16->i32; reinterprets bit pattern as signed then widens.",
    },
    {
        "name": "bool_from_int",
        "category": "cast",
        "c_code": (
            "int bool_from_int(int x) {\n"
            "    return !!x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn bool_from_int(x: i32) -> i32 {\n"
            "    if x != 0 { 1 } else { 0 }\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Boolean conversion; C !! idiom vs explicit branch in Rust.",
    },
    {
        "name": "unsigned_saturate_u8",
        "category": "cast",
        "c_code": (
            "unsigned char unsigned_saturate_u8(unsigned int x) {\n"
            "    return x > 255 ? 255 : (unsigned char)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn unsigned_saturate_u8(x: u32) -> u8 {\n"
            "    if x > 255 { 255u8 } else { x as u8 }\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Saturating cast to u8; structurally identical, reference pair.",
    },
    {
        "name": "clamp_to_i16",
        "category": "cast",
        "c_code": (
            "short clamp_to_i16(int x) {\n"
            "    if (x > 32767) return 32767;\n"
            "    if (x < -32768) return -32768;\n"
            "    return (short)x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn clamp_to_i16(x: i32) -> i16 {\n"
            "    if x > 32767 { return 32767; }\n"
            "    if x < -32768 { return -32768; }\n"
            "    x as i16\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Clamped narrowing cast; guarded, so semantics match.",
    },
    {
        "name": "widen_then_narrow",
        "category": "cast",
        "c_code": (
            "unsigned char widen_then_narrow(unsigned char a, unsigned char b) {\n"
            "    unsigned int wide = (unsigned int)a + (unsigned int)b;\n"
            "    return (unsigned char)wide;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn widen_then_narrow(a: u8, b: u8) -> u8 {\n"
            "    let wide: u32 = a as u32 + b as u32;\n"
            "    wide as u8\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Widen to add, then narrow; result truncated but no UB.",
    },
    {
        "name": "mixed_sign_compare",
        "category": "cast",
        "c_code": (
            "int mixed_sign_compare(int a, unsigned int b) {\n"
            "    return a < (int)b ? a : (int)b;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn mixed_sign_compare(a: i32, b: u32) -> i32 {\n"
            "    if a < b as i32 { a } else { b as i32 }\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "hard",
        "has_ub": False,
        "description": "Mixed sign comparison; C implicit conversion rules vs Rust explicit casts.",
    },

    # =========================================================================
    # CATEGORY 6: pointer_arithmetic (12 pairs)
    # =========================================================================

    {
        "name": "array_sum",
        "category": "pointer_arithmetic",
        "c_code": (
            "int array_sum(int *arr, int n) {\n"
            "    int sum = 0;\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        sum += arr[i];\n"
            "    }\n"
            "    return sum;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_sum(arr: &[i32], n: i32) -> i32 {\n"
            "    let mut sum: i32 = 0;\n"
            "    for i in 0..n as usize {\n"
            "        sum += arr[i + 1];\n"
            "    }\n"
            "    sum\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Array sum with off-by-one: Rust indexes arr[i+1] instead of arr[i].",
    },
    {
        "name": "array_max",
        "category": "pointer_arithmetic",
        "c_code": (
            "int array_max(int *arr, int n) {\n"
            "    int mx = arr[0];\n"
            "    for (int i = 1; i < n; i++) {\n"
            "        if (arr[i] > mx) mx = arr[i];\n"
            "    }\n"
            "    return mx;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_max(arr: &[i32], n: i32) -> i32 {\n"
            "    let mut mx = arr[0];\n"
            "    for i in 0..n as usize {\n"
            "        if arr[i] > mx { mx = arr[i]; }\n"
            "    }\n"
            "    mx\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Array max; buggy Rust starts loop at 0 instead of 1 (still works but differs from C's loop range intent).",
    },
    {
        "name": "array_index_last",
        "category": "pointer_arithmetic",
        "c_code": (
            "int array_index_last(int *arr, int n) {\n"
            "    return arr[n - 1];\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_index_last(arr: &[i32], n: i32) -> i32 {\n"
            "    arr[n as usize]\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Access last element; Rust uses arr[n] instead of arr[n-1], off-by-one.",
    },
    {
        "name": "ptr_offset_read",
        "category": "pointer_arithmetic",
        "c_code": (
            "int ptr_offset_read(int *base, int offset) {\n"
            "    return *(base + offset);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn ptr_offset_read(base: &[i32], offset: i32) -> i32 {\n"
            "    base[(offset + 1) as usize]\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Pointer offset read; buggy Rust adds extra +1 to offset.",
    },
    {
        "name": "array_dot_product",
        "category": "pointer_arithmetic",
        "c_code": (
            "int array_dot_product(int *a, int *b, int n) {\n"
            "    int sum = 0;\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        sum += a[i] * b[i];\n"
            "    }\n"
            "    return sum;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_dot_product(a: &[i32], b: &[i32], n: i32) -> i32 {\n"
            "    let mut sum: i32 = 0;\n"
            "    for i in 0..n as usize {\n"
            "        sum += a[i] * b[n as usize - 1 - i];\n"
            "    }\n"
            "    sum\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Dot product; buggy Rust reverses index of b, computing dot with reversed b.",
    },
    {
        "name": "array_swap",
        "category": "pointer_arithmetic",
        "c_code": (
            "void array_swap(int *arr, int i, int j) {\n"
            "    int tmp = arr[i];\n"
            "    arr[i] = arr[j];\n"
            "    arr[j] = tmp;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_swap(arr: &mut [i32], i: usize, j: usize) {\n"
            "    let tmp = arr[i];\n"
            "    arr[i] = arr[j];\n"
            "    arr[j] = arr[i];\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Swap two elements; buggy Rust writes arr[i] to arr[j] instead of tmp.",
    },
    {
        "name": "array_reverse_sum",
        "category": "pointer_arithmetic",
        "c_code": (
            "int array_reverse_sum(int *arr, int n) {\n"
            "    int sum = 0;\n"
            "    for (int i = n - 1; i >= 0; i--) {\n"
            "        sum += arr[i];\n"
            "    }\n"
            "    return sum;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_reverse_sum(arr: &[i32], n: i32) -> i32 {\n"
            "    let mut sum: i32 = 0;\n"
            "    let mut i = n as usize;\n"
            "    while i > 0 {\n"
            "        sum += arr[i];\n"
            "        i -= 1;\n"
            "    }\n"
            "    sum\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Reverse iteration sum; buggy Rust uses arr[i] before decrement, accessing arr[n] out of bounds.",
    },
    {
        "name": "stride_access",
        "category": "pointer_arithmetic",
        "c_code": (
            "int stride_access(int *arr, int idx, int stride) {\n"
            "    return arr[idx * stride];\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn stride_access(arr: &[i32], idx: i32, stride: i32) -> i32 {\n"
            "    arr[(idx * stride + stride) as usize]\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Strided access; buggy Rust adds extra stride to the index.",
    },
    {
        "name": "array_prefix_sum",
        "category": "pointer_arithmetic",
        "c_code": (
            "void array_prefix_sum(int *out, int *in, int n) {\n"
            "    out[0] = in[0];\n"
            "    for (int i = 1; i < n; i++) {\n"
            "        out[i] = out[i-1] + in[i];\n"
            "    }\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_prefix_sum(out: &mut [i32], inp: &[i32], n: i32) {\n"
            "    out[0] = inp[0];\n"
            "    for i in 1..n as usize {\n"
            "        out[i] = out[i - 1] + inp[i - 1];\n"
            "    }\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Prefix sum; buggy Rust reads inp[i-1] instead of inp[i].",
    },
    {
        "name": "linear_search",
        "category": "pointer_arithmetic",
        "c_code": (
            "int linear_search(int *arr, int n, int target) {\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        if (arr[i] == target) return i;\n"
            "    }\n"
            "    return -1;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn linear_search(arr: &[i32], n: i32, target: i32) -> i32 {\n"
            "    for i in 0..n as usize {\n"
            "        if arr[i] == target { return (i + 1) as i32; }\n"
            "    }\n"
            "    -1\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Linear search; buggy Rust returns 1-based index instead of 0-based.",
    },
    {
        "name": "copy_elements",
        "category": "pointer_arithmetic",
        "c_code": (
            "void copy_elements(int *dst, int *src, int n) {\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        dst[i] = src[i];\n"
            "    }\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn copy_elements(dst: &mut [i32], src: &[i32], n: i32) {\n"
            "    for i in 0..n as usize {\n"
            "        dst[i] = src[n as usize - 1 - i];\n"
            "    }\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Copy array; buggy Rust copies in reverse order.",
    },
    {
        "name": "array_count_eq",
        "category": "pointer_arithmetic",
        "c_code": (
            "int array_count_eq(int *arr, int n, int val) {\n"
            "    int count = 0;\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        if (arr[i] == val) count++;\n"
            "    }\n"
            "    return count;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_count_eq(arr: &[i32], n: i32, val: i32) -> i32 {\n"
            "    let mut count: i32 = 0;\n"
            "    for i in 0..n as usize {\n"
            "        if arr[i] != val { count += 1; }\n"
            "    }\n"
            "    count\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Count elements equal to val; buggy Rust uses != instead of ==.",
    },

    # =========================================================================
    # CATEGORY 7: struct_access (10 pairs)
    # =========================================================================

    {
        "name": "point_add",
        "category": "struct_access",
        "c_code": (
            "typedef struct { int x; int y; } Point;\n"
            "int point_add(int px, int py, int qx, int qy) {\n"
            "    return (px + qx) + (py + qy);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn point_add(px: i32, py: i32, qx: i32, qy: i32) -> i32 {\n"
            "    (px + qy) + (py + qx)\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Add two points; buggy Rust swaps qx and qy field access.",
    },
    {
        "name": "rect_area",
        "category": "struct_access",
        "c_code": (
            "int rect_area(int x1, int y1, int x2, int y2) {\n"
            "    int w = x2 - x1;\n"
            "    int h = y2 - y1;\n"
            "    return w * h;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn rect_area(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {\n"
            "    let w = x2 - x1;\n"
            "    let h = y2 - x1;\n"
            "    w * h\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Rectangle area; buggy Rust computes h as y2-x1 instead of y2-y1.",
    },
    {
        "name": "vec3_dot",
        "category": "struct_access",
        "c_code": (
            "int vec3_dot(int ax, int ay, int az, int bx, int by, int bz) {\n"
            "    return ax * bx + ay * by + az * bz;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn vec3_dot(ax: i32, ay: i32, az: i32, bx: i32, by: i32, bz: i32) -> i32 {\n"
            "    ax * bx + ay * bz + az * by\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "3D dot product; buggy Rust swaps by and bz.",
    },
    {
        "name": "complex_mul_real",
        "category": "struct_access",
        "c_code": (
            "int complex_mul_real(int ar, int ai, int br, int bi) {\n"
            "    return ar * br - ai * bi;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn complex_mul_real(ar: i32, ai: i32, br: i32, bi: i32) -> i32 {\n"
            "    ar * br + ai * bi\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Real part of complex multiply; buggy Rust uses + instead of -.",
    },
    {
        "name": "complex_mul_imag",
        "category": "struct_access",
        "c_code": (
            "int complex_mul_imag(int ar, int ai, int br, int bi) {\n"
            "    return ar * bi + ai * br;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn complex_mul_imag(ar: i32, ai: i32, br: i32, bi: i32) -> i32 {\n"
            "    ar * bi - ai * br\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Imaginary part of complex multiply; buggy Rust uses - instead of +.",
    },
    {
        "name": "rgb_to_gray",
        "category": "struct_access",
        "c_code": (
            "int rgb_to_gray(int r, int g, int b) {\n"
            "    return (r * 77 + g * 150 + b * 29) >> 8;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn rgb_to_gray(r: i32, g: i32, b: i32) -> i32 {\n"
            "    (r * 77 + g * 29 + b * 150) >> 8\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "RGB to grayscale; buggy Rust swaps green and blue coefficients.",
    },
    {
        "name": "weighted_sum_3",
        "category": "struct_access",
        "c_code": (
            "int weighted_sum_3(int a, int b, int c, int wa, int wb, int wc) {\n"
            "    return a * wa + b * wb + c * wc;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn weighted_sum_3(a: i32, b: i32, c: i32, wa: i32, wb: i32, wc: i32) -> i32 {\n"
            "    a * wa + b * wc + c * wb\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Weighted sum of 3 fields; buggy Rust swaps wb and wc.",
    },
    {
        "name": "line_length_sq",
        "category": "struct_access",
        "c_code": (
            "int line_length_sq(int x1, int y1, int x2, int y2) {\n"
            "    int dx = x2 - x1;\n"
            "    int dy = y2 - y1;\n"
            "    return dx * dx + dy * dy;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn line_length_sq(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {\n"
            "    let dx = x2 - x1;\n"
            "    let dy = y2 - y1;\n"
            "    dx * dx + dy * dx\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Squared distance; buggy Rust computes dy*dx instead of dy*dy.",
    },
    {
        "name": "matrix_2x2_det",
        "category": "struct_access",
        "c_code": (
            "int matrix_2x2_det(int a, int b, int c, int d) {\n"
            "    return a * d - b * c;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn matrix_2x2_det(a: i32, b: i32, c: i32, d: i32) -> i32 {\n"
            "    a * d - c * d\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "2x2 matrix determinant; buggy Rust computes c*d instead of b*c.",
    },
    {
        "name": "cross_product_2d",
        "category": "struct_access",
        "c_code": (
            "int cross_product_2d(int ax, int ay, int bx, int by) {\n"
            "    return ax * by - ay * bx;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn cross_product_2d(ax: i32, ay: i32, bx: i32, by: i32) -> i32 {\n"
            "    ax * bx - ay * by\n"
            "}"
        ),
        "bug_class": "struct_field_swap",
        "difficulty": "medium",
        "has_ub": False,
        "description": "2D cross product; buggy Rust uses ax*bx-ay*by instead of ax*by-ay*bx.",
    },

    # =========================================================================
    # CATEGORY 8: bitwise (10 pairs)
    # =========================================================================

    {
        "name": "popcount32",
        "category": "bitwise",
        "c_code": (
            "int popcount32(unsigned int x) {\n"
            "    int count = 0;\n"
            "    while (x) {\n"
            "        count += x & 1;\n"
            "        x >>= 1;\n"
            "    }\n"
            "    return count;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn popcount32(mut x: u32) -> i32 {\n"
            "    let mut count: i32 = 0;\n"
            "    while x != 0 {\n"
            "        count += (x & 1) as i32;\n"
            "        x >>= 2;\n"
            "    }\n"
            "    count\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Population count; buggy Rust shifts by 2 instead of 1, missing bits.",
    },
    {
        "name": "leading_zeros",
        "category": "bitwise",
        "c_code": (
            "int leading_zeros(unsigned int x) {\n"
            "    if (x == 0) return 32;\n"
            "    int n = 0;\n"
            "    while (!(x & 0x80000000u)) {\n"
            "        n++;\n"
            "        x <<= 1;\n"
            "    }\n"
            "    return n;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn leading_zeros(mut x: u32) -> i32 {\n"
            "    if x == 0 { return 32; }\n"
            "    let mut n: i32 = 0;\n"
            "    while (x & 0x80000000) == 0 {\n"
            "        n += 1;\n"
            "        x <<= 1;\n"
            "    }\n"
            "    n + 1\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Count leading zeros; buggy Rust adds 1 to final result (off by one).",
    },
    {
        "name": "trailing_zeros",
        "category": "bitwise",
        "c_code": (
            "int trailing_zeros(unsigned int x) {\n"
            "    if (x == 0) return 32;\n"
            "    int n = 0;\n"
            "    while ((x & 1) == 0) {\n"
            "        n++;\n"
            "        x >>= 1;\n"
            "    }\n"
            "    return n;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn trailing_zeros(mut x: u32) -> i32 {\n"
            "    if x == 0 { return 31; }\n"
            "    let mut n: i32 = 0;\n"
            "    while (x & 1) == 0 {\n"
            "        n += 1;\n"
            "        x >>= 1;\n"
            "    }\n"
            "    n\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Count trailing zeros; buggy Rust returns 31 for x==0 instead of 32.",
    },
    {
        "name": "is_power_of_two",
        "category": "bitwise",
        "c_code": (
            "int is_power_of_two(unsigned int x) {\n"
            "    return x != 0 && (x & (x - 1)) == 0;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn is_power_of_two(x: u32) -> i32 {\n"
            "    if x != 0 && (x & (x - 1)) == 0 { 1 } else { 0 }\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Power of two check; Rust x-1 panics on x==0 in debug (but guarded).",
    },
    {
        "name": "reverse_bits_8",
        "category": "bitwise",
        "c_code": (
            "unsigned char reverse_bits_8(unsigned char x) {\n"
            "    unsigned char result = 0;\n"
            "    for (int i = 0; i < 8; i++) {\n"
            "        result = (result << 1) | (x & 1);\n"
            "        x >>= 1;\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn reverse_bits_8(mut x: u8) -> u8 {\n"
            "    let mut result: u8 = 0;\n"
            "    for _ in 0..7 {\n"
            "        result = (result << 1) | (x & 1);\n"
            "        x >>= 1;\n"
            "    }\n"
            "    result\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Reverse 8 bits; buggy Rust iterates 7 times instead of 8.",
    },
    {
        "name": "swap_nibbles",
        "category": "bitwise",
        "c_code": (
            "unsigned char swap_nibbles(unsigned char x) {\n"
            "    return (x >> 4) | (x << 4);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn swap_nibbles(x: u8) -> u8 {\n"
            "    (x >> 4) | (x << 3)\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Swap high/low nibbles; buggy Rust shifts left by 3 instead of 4.",
    },
    {
        "name": "parity_bit",
        "category": "bitwise",
        "c_code": (
            "int parity_bit(unsigned int x) {\n"
            "    int p = 0;\n"
            "    while (x) {\n"
            "        p ^= 1;\n"
            "        x &= x - 1;\n"
            "    }\n"
            "    return p;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn parity_bit(mut x: u32) -> i32 {\n"
            "    let mut p: i32 = 0;\n"
            "    while x != 0 {\n"
            "        p ^= 1;\n"
            "        x &= x.wrapping_add(1);\n"
            "    }\n"
            "    p\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Compute parity; buggy Rust uses x+1 instead of x-1, wrong Kernighan trick.",
    },
    {
        "name": "lowest_set_bit",
        "category": "bitwise",
        "c_code": (
            "unsigned int lowest_set_bit(unsigned int x) {\n"
            "    return x & (~x + 1);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn lowest_set_bit(x: u32) -> u32 {\n"
            "    x & (!x + 1)\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Isolate lowest set bit; Rust '!' is bitwise NOT on integers (same as ~) so this actually works, but Rust checked add on !x+1 may panic when x==0.",
    },
    {
        "name": "merge_bits",
        "category": "bitwise",
        "c_code": (
            "unsigned int merge_bits(unsigned int a, unsigned int b, unsigned int mask) {\n"
            "    return (a & mask) | (b & ~mask);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn merge_bits(a: u32, b: u32, mask: u32) -> u32 {\n"
            "    (a & mask) | (b & mask)\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Merge bits using mask; buggy Rust uses mask instead of ~mask for b.",
    },
    {
        "name": "next_power_of_two",
        "category": "bitwise",
        "c_code": (
            "unsigned int next_power_of_two(unsigned int x) {\n"
            "    x--;\n"
            "    x |= x >> 1;\n"
            "    x |= x >> 2;\n"
            "    x |= x >> 4;\n"
            "    x |= x >> 8;\n"
            "    x |= x >> 16;\n"
            "    x++;\n"
            "    return x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn next_power_of_two(mut x: u32) -> u32 {\n"
            "    x -= 1;\n"
            "    x |= x >> 1;\n"
            "    x |= x >> 2;\n"
            "    x |= x >> 4;\n"
            "    x |= x >> 8;\n"
            "    x += 1;\n"
            "    x\n"
            "}"
        ),
        "bug_class": "bitwise",
        "difficulty": "hard",
        "has_ub": False,
        "description": "Round up to next power of 2; buggy Rust omits x |= x >> 16 step.",
    },

    # =========================================================================
    # CATEGORY 9: control_flow (12 pairs)
    # =========================================================================

    {
        "name": "clamp",
        "category": "control_flow",
        "c_code": (
            "int clamp(int x, int lo, int hi) {\n"
            "    if (x < lo) return lo;\n"
            "    if (x > hi) return hi;\n"
            "    return x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn clamp(x: i32, lo: i32, hi: i32) -> i32 {\n"
            "    if x < lo { return hi; }\n"
            "    if x > hi { return lo; }\n"
            "    x\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Clamp value; buggy Rust swaps return values lo/hi.",
    },
    {
        "name": "sign_func",
        "category": "control_flow",
        "c_code": (
            "int sign_func(int x) {\n"
            "    if (x > 0) return 1;\n"
            "    if (x < 0) return -1;\n"
            "    return 0;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn sign_func(x: i32) -> i32 {\n"
            "    if x > 0 { return 1; }\n"
            "    if x <= 0 { return -1; }\n"
            "    0\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Sign function; buggy Rust uses <= instead of <, returning -1 for 0.",
    },
    {
        "name": "max3",
        "category": "control_flow",
        "c_code": (
            "int max3(int a, int b, int c) {\n"
            "    int m = a;\n"
            "    if (b > m) m = b;\n"
            "    if (c > m) m = c;\n"
            "    return m;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn max3(a: i32, b: i32, c: i32) -> i32 {\n"
            "    let mut m = a;\n"
            "    if b > m { m = b; }\n"
            "    if c > a { m = c; }\n"
            "    m\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Maximum of three; buggy Rust compares c > a instead of c > m.",
    },
    {
        "name": "min3",
        "category": "control_flow",
        "c_code": (
            "int min3(int a, int b, int c) {\n"
            "    int m = a;\n"
            "    if (b < m) m = b;\n"
            "    if (c < m) m = c;\n"
            "    return m;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn min3(a: i32, b: i32, c: i32) -> i32 {\n"
            "    let mut m = a;\n"
            "    if b < m { m = b; }\n"
            "    if c < m { m = b; }\n"
            "    m\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Minimum of three; buggy Rust assigns b instead of c in second branch.",
    },
    {
        "name": "nested_ternary",
        "category": "control_flow",
        "c_code": (
            "int nested_ternary(int x) {\n"
            "    return x > 100 ? 3 : (x > 50 ? 2 : (x > 0 ? 1 : 0));\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn nested_ternary(x: i32) -> i32 {\n"
            "    if x > 100 { 3 } else if x > 50 { 2 } else if x >= 0 { 1 } else { 0 }\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Nested ternary; buggy Rust uses >= 0 instead of > 0, returning 1 for x==0.",
    },
    {
        "name": "fizzbuzz_val",
        "category": "control_flow",
        "c_code": (
            "int fizzbuzz_val(int n) {\n"
            "    if (n % 15 == 0) return 15;\n"
            "    if (n % 3 == 0) return 3;\n"
            "    if (n % 5 == 0) return 5;\n"
            "    return n;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn fizzbuzz_val(n: i32) -> i32 {\n"
            "    if n % 15 == 0 { return 15; }\n"
            "    if n % 5 == 0 { return 5; }\n"
            "    if n % 3 == 0 { return 3; }\n"
            "    n\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "easy",
        "has_ub": False,
        "description": "FizzBuzz; buggy Rust swaps order of mod-3 and mod-5 checks (still correct since 15 is checked first, but reorder can matter with side effects in general).",
    },
    {
        "name": "collatz_steps",
        "category": "control_flow",
        "c_code": (
            "int collatz_steps(int n) {\n"
            "    int steps = 0;\n"
            "    while (n != 1) {\n"
            "        if (n % 2 == 0) n = n / 2;\n"
            "        else n = 3 * n + 1;\n"
            "        steps++;\n"
            "    }\n"
            "    return steps;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn collatz_steps(mut n: i32) -> i32 {\n"
            "    let mut steps: i32 = 0;\n"
            "    while n != 1 {\n"
            "        if n % 2 == 0 { n = n / 2; }\n"
            "        else { n = 3 * n - 1; }\n"
            "        steps += 1;\n"
            "    }\n"
            "    steps\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Collatz sequence; buggy Rust uses 3*n-1 instead of 3*n+1.",
    },
    {
        "name": "digit_sum",
        "category": "control_flow",
        "c_code": (
            "int digit_sum(int n) {\n"
            "    if (n < 0) n = -n;\n"
            "    int sum = 0;\n"
            "    while (n > 0) {\n"
            "        sum += n % 10;\n"
            "        n /= 10;\n"
            "    }\n"
            "    return sum;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn digit_sum(mut n: i32) -> i32 {\n"
            "    if n < 0 { n = -n; }\n"
            "    let mut sum: i32 = 0;\n"
            "    while n > 0 {\n"
            "        sum += n / 10;\n"
            "        n %= 10;\n"
            "    }\n"
            "    sum\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Sum of digits; buggy Rust swaps / and % operations.",
    },
    {
        "name": "count_down_to_zero",
        "category": "control_flow",
        "c_code": (
            "int count_down_to_zero(int n) {\n"
            "    int count = 0;\n"
            "    while (n > 0) {\n"
            "        n--;\n"
            "        count++;\n"
            "    }\n"
            "    return count;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn count_down_to_zero(mut n: i32) -> i32 {\n"
            "    let mut count: i32 = 0;\n"
            "    while n >= 0 {\n"
            "        n -= 1;\n"
            "        count += 1;\n"
            "    }\n"
            "    count\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Count down; buggy Rust uses >= 0 instead of > 0, extra iteration.",
    },
    {
        "name": "triangular_number",
        "category": "control_flow",
        "c_code": (
            "int triangular_number(int n) {\n"
            "    return n * (n + 1) / 2;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn triangular_number(n: i32) -> i32 {\n"
            "    n * (n - 1) / 2\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "easy",
        "has_ub": False,
        "description": "n-th triangular number; buggy Rust uses n-1 instead of n+1.",
    },
    {
        "name": "binary_search_bound",
        "category": "control_flow",
        "c_code": (
            "int binary_search_bound(int lo, int hi, int target) {\n"
            "    while (lo < hi) {\n"
            "        int mid = lo + (hi - lo) / 2;\n"
            "        if (mid < target) lo = mid + 1;\n"
            "        else hi = mid;\n"
            "    }\n"
            "    return lo;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn binary_search_bound(mut lo: i32, mut hi: i32, target: i32) -> i32 {\n"
            "    while lo < hi {\n"
            "        let mid = lo + (hi - lo) / 2;\n"
            "        if mid < target { lo = mid; }\n"
            "        else { hi = mid; }\n"
            "    }\n"
            "    lo\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "hard",
        "has_ub": False,
        "description": "Binary search lower bound; buggy Rust uses lo=mid instead of mid+1, causing infinite loop on some inputs.",
    },
    {
        "name": "classify_char",
        "category": "control_flow",
        "c_code": (
            "int classify_char(int c) {\n"
            "    if (c >= '0' && c <= '9') return 1;\n"
            "    if (c >= 'a' && c <= 'z') return 2;\n"
            "    if (c >= 'A' && c <= 'Z') return 3;\n"
            "    return 0;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn classify_char(c: i32) -> i32 {\n"
            "    if c >= 0x30 && c <= 0x39 { return 1; }\n"
            "    if c >= 0x41 && c <= 0x5A { return 2; }\n"
            "    if c >= 0x61 && c <= 0x7A { return 3; }\n"
            "    0\n"
            "}"
        ),
        "bug_class": "control_flow",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Character classification; buggy Rust swaps return codes for upper/lower case.",
    },

    # =========================================================================
    # CATEGORY 10: mixed (12 pairs)
    # =========================================================================

    {
        "name": "scale_and_clamp",
        "category": "mixed",
        "c_code": (
            "int scale_and_clamp(int x, int factor, int max_val) {\n"
            "    int scaled = x * factor;\n"
            "    if (scaled > max_val) return max_val;\n"
            "    if (scaled < 0) return 0;\n"
            "    return scaled;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn scale_and_clamp(x: i32, factor: i32, max_val: i32) -> i32 {\n"
            "    let scaled = x * factor;\n"
            "    if scaled > max_val { return max_val; }\n"
            "    if scaled < 0 { return 0; }\n"
            "    scaled\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Multiply then clamp; x*factor overflows in C (UB), Rust panics.",
    },
    {
        "name": "safe_average",
        "category": "mixed",
        "c_code": (
            "int safe_average(int a, int b) {\n"
            "    return a / 2 + b / 2 + (a % 2 + b % 2) / 2;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn safe_average(a: i32, b: i32) -> i32 {\n"
            "    a / 2 + b / 2 + (a % 2 + b % 2) / 2\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "hard",
        "has_ub": False,
        "description": "Overflow-safe average; no UB in C, but Rust semantics differ for negative modulo edge cases.",
    },
    {
        "name": "shift_and_mask",
        "category": "mixed",
        "c_code": (
            "int shift_and_mask(int val, int shift_amt) {\n"
            "    return (val >> shift_amt) & 0xFF;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn shift_and_mask(val: i32, shift_amt: i32) -> i32 {\n"
            "    (val >> shift_amt) & 0xFF\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Shift then mask; shift_amt >= 32 or < 0 is UB in C, Rust panics.",
    },
    {
        "name": "mul_then_shift",
        "category": "mixed",
        "c_code": (
            "int mul_then_shift(int a, int b, int s) {\n"
            "    return (a * b) >> s;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn mul_then_shift(a: i32, b: i32, s: i32) -> i32 {\n"
            "    (a * b) >> s\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "hard",
        "has_ub": True,
        "description": "Multiply-then-shift; a*b overflows (UB), and s can cause shift UB.",
    },
    {
        "name": "cast_shift_combine",
        "category": "mixed",
        "c_code": (
            "unsigned int cast_shift_combine(unsigned short hi, unsigned short lo) {\n"
            "    return ((unsigned int)hi << 16) | (unsigned int)lo;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn cast_shift_combine(hi: u16, lo: u16) -> u32 {\n"
            "    ((hi as u32) << 16) | (lo as u32)\n"
            "}"
        ),
        "bug_class": "cast_truncation",
        "difficulty": "easy",
        "has_ub": False,
        "description": "Combine two u16 into u32; reference pair that should match semantics.",
    },
    {
        "name": "array_weighted_sum",
        "category": "mixed",
        "c_code": (
            "int array_weighted_sum(int *arr, int *weights, int n) {\n"
            "    int sum = 0;\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        sum += arr[i] * weights[i];\n"
            "    }\n"
            "    return sum;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn array_weighted_sum(arr: &[i32], weights: &[i32], n: i32) -> i32 {\n"
            "    let mut sum: i32 = 0;\n"
            "    for i in 0..n as usize {\n"
            "        sum += arr[i] + weights[i];\n"
            "    }\n"
            "    sum\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "medium",
        "has_ub": False,
        "description": "Weighted array sum; buggy Rust uses + instead of * for combining.",
    },
    {
        "name": "overflow_then_cast",
        "category": "mixed",
        "c_code": (
            "unsigned char overflow_then_cast(int a, int b) {\n"
            "    return (unsigned char)(a + b);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn overflow_then_cast(a: i32, b: i32) -> u8 {\n"
            "    (a + b) as u8\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Add overflows then cast to u8; signed overflow UB in C before truncation.",
    },
    {
        "name": "div_with_cast",
        "category": "mixed",
        "c_code": (
            "int div_with_cast(long long a, int b) {\n"
            "    return (int)(a / (long long)b);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn div_with_cast(a: i64, b: i32) -> i32 {\n"
            "    (a / b as i64) as i32\n"
            "}"
        ),
        "bug_class": "division_by_zero",
        "difficulty": "medium",
        "has_ub": True,
        "description": "64-bit division then narrow; b==0 is UB, and result can truncate.",
    },
    {
        "name": "bitfield_insert",
        "category": "mixed",
        "c_code": (
            "unsigned int bitfield_insert(unsigned int base, unsigned int val, int pos, int width) {\n"
            "    unsigned int mask = ((1u << width) - 1) << pos;\n"
            "    return (base & ~mask) | ((val << pos) & mask);\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn bitfield_insert(base: u32, val: u32, pos: i32, width: i32) -> u32 {\n"
            "    let mask: u32 = ((1u32 << width as u32) - 1) << pos as u32;\n"
            "    (base & !mask) | ((val << pos as u32) & mask)\n"
            "}"
        ),
        "bug_class": "shift_ub",
        "difficulty": "hard",
        "has_ub": True,
        "description": "Insert bitfield; shift by width or pos >= 32 is UB in C, Rust panics.",
    },
    {
        "name": "pointer_offset_cast",
        "category": "mixed",
        "c_code": (
            "int pointer_offset_cast(int *arr, int idx) {\n"
            "    unsigned char *p = (unsigned char *)arr;\n"
            "    return *(int *)(p + idx * sizeof(int));\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn pointer_offset_cast(arr: &[i32], idx: i32) -> i32 {\n"
            "    arr[(idx * 4) as usize]\n"
            "}"
        ),
        "bug_class": "pointer_arithmetic",
        "difficulty": "hard",
        "has_ub": False,
        "description": "Byte-level pointer arithmetic; buggy Rust treats i32 slice as byte-indexed, index is 4x too large.",
    },
    {
        "name": "conditional_overflow",
        "category": "mixed",
        "c_code": (
            "int conditional_overflow(int x, int y, int mode) {\n"
            "    if (mode == 0) return x + y;\n"
            "    if (mode == 1) return x - y;\n"
            "    if (mode == 2) return x * y;\n"
            "    return x;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn conditional_overflow(x: i32, y: i32, mode: i32) -> i32 {\n"
            "    if mode == 0 { return x + y; }\n"
            "    if mode == 1 { return x - y; }\n"
            "    if mode == 2 { return x * y; }\n"
            "    x\n"
            "}"
        ),
        "bug_class": "signed_overflow",
        "difficulty": "medium",
        "has_ub": True,
        "description": "Mode-dependent arithmetic; all three operations have signed overflow UB in C.",
    },
    {
        "name": "hash_combine_full",
        "category": "mixed",
        "c_code": (
            "unsigned int hash_combine_full(unsigned int seed, unsigned int val) {\n"
            "    seed ^= val + 0x9e3779b9u + (seed << 6) + (seed >> 2);\n"
            "    return seed;\n"
            "}"
        ),
        "buggy_rust": (
            "pub fn hash_combine_full(mut seed: u32, val: u32) -> u32 {\n"
            "    seed ^= val + 0x9e3779b9u32 + (seed << 6) + (seed >> 2);\n"
            "    seed\n"
            "}"
        ),
        "bug_class": "unsigned_wrap",
        "difficulty": "hard",
        "has_ub": False,
        "description": "Boost hash_combine; relies on unsigned wrapping in additions, Rust panics on overflow.",
    },
]


def get_by_category(category: str):
    """Return all benchmark pairs for a given category."""
    return [p for p in CEGAR_BENCHMARK_PAIRS if p["category"] == category]


def get_by_difficulty(difficulty: str):
    """Return all benchmark pairs for a given difficulty level."""
    return [p for p in CEGAR_BENCHMARK_PAIRS if p["difficulty"] == difficulty]


def get_by_bug_class(bug_class: str):
    """Return all benchmark pairs for a given bug class."""
    return [p for p in CEGAR_BENCHMARK_PAIRS if p["bug_class"] == bug_class]


def get_by_name(name: str):
    """Return a single benchmark pair by function name."""
    for p in CEGAR_BENCHMARK_PAIRS:
        if p["name"] == name:
            return p
    return None


def summary():
    """Print a summary of the benchmark suite."""
    from collections import Counter
    cats = Counter(p["category"] for p in CEGAR_BENCHMARK_PAIRS)
    bugs = Counter(p["bug_class"] for p in CEGAR_BENCHMARK_PAIRS)
    diffs = Counter(p["difficulty"] for p in CEGAR_BENCHMARK_PAIRS)
    ub_count = sum(1 for p in CEGAR_BENCHMARK_PAIRS if p["has_ub"])
    print(f"Total pairs: {len(CEGAR_BENCHMARK_PAIRS)}")
    print(f"\nBy category:")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")
    print(f"\nBy bug class:")
    for bug, cnt in sorted(bugs.items()):
        print(f"  {bug}: {cnt}")
    print(f"\nBy difficulty:")
    for d, cnt in sorted(diffs.items()):
        print(f"  {d}: {cnt}")
    print(f"\nPairs with UB in C code: {ub_count}")
    print(f"Pairs without UB: {len(CEGAR_BENCHMARK_PAIRS) - ub_count}")


if __name__ == "__main__":
    summary()
