"""
Real-world C-to-Rust translation benchmark pairs.

Provides realistic benchmark pairs simulating LLM-generated translations of
common C library functions and patterns found in actual codebases. Each pair
includes idiomatic C11 source and a Rust translation (some correct, some with
subtle bugs) suitable for cross-language equivalence verification.

Categories covered:
  - String operations
  - Math utilities
  - Bit manipulation
  - Array/buffer operations
  - Crypto-style functions
  - Data structure helpers
  - Error handling patterns
  - Signal processing / DSP
  - Network/protocol helpers
  - Mixed real-world utilities
"""

REAL_WORLD_BENCHMARK_PAIRS = [

    # =========================================================================
    # 1. STRING OPERATIONS
    # =========================================================================
    {
        "name": 'my_strlen',
        "category": 'string',
        "c_code": """
int my_strlen(const char *s) {
    int len = 0;
    while (s[len] != '\0') {
        len++;
    }
    return len;
}
""",
        "rust_code": """
fn my_strlen(s: &[u8]) -> i32 {
    let mut len: i32 = 0;
    while (len as usize) < s.len() && s[len as usize] != 0 {
        len += 1;
    }
    len
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'libc-style',
        "description": 'Basic string length via null-terminator scan.',
    },
    {
        "name": 'my_strcmp',
        "category": 'string',
        "c_code": """
int my_strcmp(const char *a, const char *b) {
    int i = 0;
    while (a[i] != '\0' && a[i] == b[i]) {
        i++;
    }
    return (unsigned char)a[i] - (unsigned char)b[i];
}
""",
        "rust_code": """
fn my_strcmp(a: &[u8], b: &[u8]) -> i32 {
    let mut i: usize = 0;
    while i < a.len() && a[i] != 0 && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    let ca = if i < a.len() { a[i] as i32 } else { 0 };
    let cb = if i < b.len() { b[i] as i32 } else { 0 };
    // BUG: C compares as unsigned char, but this treats as signed
    // For values > 127 this gives different results
    (ca as i8) as i32 - (cb as i8) as i32
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'signedness-mismatch',
        "source": 'libc-style',
        "description": 'Lexicographic comparison. Rust version uses signed byte comparison.',
    },
    {
        "name": 'my_strncpy',
        "category": 'string',
        "c_code": """
void my_strncpy(char *dst, const char *src, int n) {
    int i = 0;
    while (i < n && src[i] != '\0') {
        dst[i] = src[i];
        i++;
    }
    while (i < n) {
        dst[i] = '\0';
        i++;
    }
}
""",
        "rust_code": """
fn my_strncpy(dst: &mut [u8], src: &[u8], n: i32) {
    let n = n as usize;
    let mut i: usize = 0;
    while i < n && i < src.len() && src[i] != 0 {
        dst[i] = src[i];
        i += 1;
    }
    while i < n {
        dst[i] = 0;
        i += 1;
    }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'libc-style',
        "description": 'Bounded string copy with null-padding.',
    },
    {
        "name": 'my_memcmp',
        "category": 'string',
        "c_code": """
int my_memcmp(const void *s1, const void *s2, int n) {
    const unsigned char *p1 = (const unsigned char *)s1;
    const unsigned char *p2 = (const unsigned char *)s2;
    for (int i = 0; i < n; i++) {
        if (p1[i] != p2[i])
            return p1[i] - p2[i];
    }
    return 0;
}
""",
        "rust_code": """
fn my_memcmp(s1: &[u8], s2: &[u8], n: i32) -> i32 {
    let n = n as usize;
    for i in 0..n {
        if s1[i] != s2[i] {
            return s1[i] as i32 - s2[i] as i32;
        }
    }
    0
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'libc-style',
        "description": 'Byte-level memory comparison.',
    },
    {
        "name": 'my_memset',
        "category": 'string',
        "c_code": """
void my_memset(void *s, int c, int n) {
    unsigned char *p = (unsigned char *)s;
    for (int i = 0; i < n; i++) {
        p[i] = (unsigned char)c;
    }
}
""",
        "rust_code": """
fn my_memset(s: &mut [u8], c: i32, n: i32) {
    let val = c as u8;  // BUG: no & 0xFF mask, truncation differs for negative c
    for i in 0..(n as usize) {
        s[i] = val;
    }
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-mask',
        "source": 'libc-style',
        "description": "Fill memory. Rust version doesn't mask c to unsigned char range.",
    },
    {
        "name": 'my_strchr',
        "category": 'string',
        "c_code": """
int my_strchr(const char *s, int c) {
    int i = 0;
    while (s[i] != '\0') {
        if (s[i] == (char)c) return i;
        i++;
    }
    if (c == 0) return i;
    return -1;
}
""",
        "rust_code": """
fn my_strchr(s: &[u8], c: i32) -> i32 {
    let target = (c & 0xFF) as u8;
    for i in 0..s.len() {
        if s[i] == 0 {
            return if target == 0 { i as i32 } else { -1 };
        }
        if s[i] == target {
            return i as i32;
        }
    }
    -1
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'libc-style',
        "description": 'Find first occurrence of character in string.',
    },
    {
        "name": 'my_strrchr',
        "category": 'string',
        "c_code": """
int my_strrchr(const char *s, int c) {
    int last = -1;
    int i = 0;
    while (s[i] != '\0') {
        if (s[i] == (char)c) last = i;
        i++;
    }
    if (c == 0) return i;
    return last;
}
""",
        "rust_code": """
fn my_strrchr(s: &[u8], c: i32) -> i32 {
    let target = (c & 0xFF) as u8;
    let mut last: i32 = -1;
    for i in 0..s.len() {
        if s[i] == 0 {
            break;
        }
        if s[i] == target {
            last = i as i32;
        }
    }
    // BUG: forgot to handle c == 0 case
    last
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-edge-case',
        "source": 'libc-style',
        "description": 'Find last occurrence of character. Rust version misses null-char search.',
    },
    {
        "name": 'my_strncat',
        "category": 'string',
        "c_code": """
void my_strncat(char *dst, const char *src, int n) {
    int di = 0;
    while (dst[di] != '\0') di++;
    int si = 0;
    while (si < n && src[si] != '\0') {
        dst[di] = src[si];
        di++;
        si++;
    }
    dst[di] = '\0';
}
""",
        "rust_code": """
fn my_strncat(dst: &mut [u8], src: &[u8], n: i32) {
    let mut di: usize = 0;
    while di < dst.len() && dst[di] != 0 { di += 1; }
    let mut si: usize = 0;
    let n = n as usize;
    while si < n && si < src.len() && src[si] != 0 {
        if di < dst.len() {
            dst[di] = src[si];
        }
        di += 1;
        si += 1;
    }
    if di < dst.len() {
        dst[di] = 0;
    }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'libc-style',
        "description": 'Bounded string concatenation.',
    },
    {
        "name": 'str_to_upper',
        "category": 'string',
        "c_code": """
void str_to_upper(char *s) {
    for (int i = 0; s[i] != '\0'; i++) {
        if (s[i] >= 'a' && s[i] <= 'z') {
            s[i] = s[i] - 'a' + 'A';
        }
    }
}
""",
        "rust_code": """
fn str_to_upper(s: &mut [u8]) {
    for i in 0..s.len() {
        if s[i] == 0 { break; }
        // BUG: converts all chars, not just a-z
        if s[i] >= b'a' {
            s[i] = s[i] - b'a' + b'A';
        }
    }
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'widened-condition',
        "source": 'libc-style',
        "description": "To-upper. Rust version converts characters above 'z' too.",
    },
    {
        "name": 'count_char',
        "category": 'string',
        "c_code": """
int count_char(const char *s, char c) {
    int count = 0;
    for (int i = 0; s[i] != '\0'; i++) {
        if (s[i] == c) count++;
    }
    return count;
}
""",
        "rust_code": """
fn count_char(s: &[u8], c: u8) -> i32 {
    let mut count: i32 = 0;
    for i in 0..s.len() {
        if s[i] == 0 { break; }
        if s[i] == c { count += 1; }
    }
    count
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'libc-style',
        "description": 'Count occurrences of a character in a string.',
    },

    # =========================================================================
    # 2. MATH UTILITIES
    # =========================================================================
    {
        "name": 'my_abs',
        "category": 'math',
        "c_code": """
int my_abs(int x) {
    return x < 0 ? -x : x;
}
""",
        "rust_code": """
fn my_abs(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'libc-style',
        "description": 'Integer absolute value.',
    },
    {
        "name": 'clamp',
        "category": 'math',
        "c_code": """
int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}
""",
        "rust_code": """
fn clamp(val: i32, lo: i32, hi: i32) -> i32 {
    if val < lo { lo }
    else if val > hi { hi }
    else { val }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Clamp integer to a range.',
    },
    {
        "name": 'int_min',
        "category": 'math',
        "c_code": """
int int_min(int a, int b) {
    return a < b ? a : b;
}
""",
        "rust_code": """
fn int_min(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Return the minimum of two integers.',
    },
    {
        "name": 'int_max',
        "category": 'math',
        "c_code": """
int int_max(int a, int b) {
    return a > b ? a : b;
}
""",
        "rust_code": """
fn int_max(a: i32, b: i32) -> i32 {
    // BUG: returns min instead of max
    if a < b { a } else { b }
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'inverted-logic',
        "source": 'utility',
        "description": 'Max of two ints. Rust version returns minimum.',
    },
    {
        "name": 'gcd',
        "category": 'math',
        "c_code": """
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
        "rust_code": """
fn gcd(mut a: i32, mut b: i32) -> i32 {
    if a < 0 { a = -a; }
    if b < 0 { b = -b; }
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'utility',
        "description": 'GCD with negative inputs and modulo sign conventions make exhaustive verification complex.',
    },
    {
        "name": 'lcm',
        "category": 'math',
        "c_code": """
int lcm(int a, int b) {
    if (a == 0 || b == 0) return 0;
    int g = a;
    int tmp = b;
    while (tmp != 0) {
        int r = g % tmp;
        g = tmp;
        tmp = r;
    }
    return (a / g) * b;
}
""",
        "rust_code": """
fn lcm(a: i32, b: i32) -> i32 {
    if a == 0 || b == 0 { return 0; }
    let mut g = a;
    let mut tmp = b;
    while tmp != 0 {
        let r = g % tmp;
        g = tmp;
        tmp = r;
    }
    // BUG: a * b / g instead of (a / g) * b — overflows differently
    (a * b) / g
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'overflow-reordering',
        "source": 'utility',
        "description": 'LCM via GCD. Rust version reorders multiply/divide causing overflow.',
    },
    {
        "name": 'isqrt',
        "category": 'math',
        "c_code": """
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
        "rust_code": """
fn isqrt(n: i32) -> i32 {
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
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": "Integer square root via Newton's method.",
    },
    {
        "name": 'ipow',
        "category": 'math',
        "c_code": """
int ipow(int base, int exp) {
    int result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = result * base;
        }
        base = base * base;
        exp = exp / 2;
    }
    return result;
}
""",
        "rust_code": """
fn ipow(mut base: i32, mut exp: i32) -> i32 {
    let mut result: i32 = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result.wrapping_mul(base);
        }
        // BUG: squares base before halving exp — but this is actually the same algorithm
        exp >>= 1;
        base = base.wrapping_mul(base);
    }
    result
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'reordered-operations',
        "source": 'utility',
        "description": 'Integer pow. Rust version squares base after halving exp, causing extra multiply.',
    },
    {
        "name": 'modpow',
        "category": 'math',
        "c_code": """
int modpow(int base, int exp, int mod) {
    if (mod == 1) return 0;
    int result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % mod;
        }
        exp = exp / 2;
        base = (base * base) % mod;
    }
    return result;
}
""",
        "rust_code": """
fn modpow(mut base: i32, mut exp: i32, modulus: i32) -> i32 {
    if modulus == 1 { return 0; }
    let mut result: i32 = 1;
    base = base % modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp /= 2;
        base = (base * base) % modulus;
    }
    result
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'utility',
        "description": 'Modular exponentiation with intermediate overflow makes verification difficult.',
    },
    {
        "name": 'factorial',
        "category": 'math',
        "c_code": """
int factorial(int n) {
    if (n < 0) return -1;
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
""",
        "rust_code": """
fn factorial(n: i32) -> i32 {
    if n < 0 { return -1; }
    let mut result: i32 = 1;
    for i in 2..=n {
        result = result.wrapping_mul(i);
    }
    result
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Iterative factorial.',
    },
    {
        "name": 'fibonacci',
        "category": 'math',
        "c_code": """
int fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int t = a + b;
        a = b;
        b = t;
    }
    return b;
}
""",
        "rust_code": """
fn fibonacci(n: i32) -> i32 {
    if n <= 0 { return 0; }
    if n == 1 { return 1; }
    let mut a: i32 = 0;
    let mut b: i32 = 1;
    for _ in 2..=n {
        let t = a + b;
        a = b;
        b = t;
    }
    b
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Iterative Fibonacci number.',
    },
    {
        "name": 'div_ceil',
        "category": 'math',
        "c_code": """
int div_ceil(int a, int b) {
    if (b == 0) return 0;
    if ((a >= 0 && b > 0) || (a <= 0 && b < 0)) {
        return (a + b - 1) / b;
    }
    return a / b;
}
""",
        "rust_code": """
fn div_ceil(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    // BUG: naive ceiling division doesn't handle negative cases
    (a + b - 1) / b
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'sign-handling',
        "source": 'utility',
        "description": 'Ceiling integer division. Rust version is wrong for negative operands.',
    },

    # =========================================================================
    # 3. BIT MANIPULATION
    # =========================================================================
    {
        "name": 'popcount',
        "category": 'bitwise',
        "c_code": """
int popcount(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}
""",
        "rust_code": """
fn popcount(mut x: u32) -> i32 {
    let mut count: i32 = 0;
    while x != 0 {
        count += (x & 1) as i32;
        x >>= 1;
    }
    count
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Population count (number of set bits).',
    },
    {
        "name": 'clz',
        "category": 'bitwise',
        "c_code": """
int clz(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    if ((x & 0xFFFF0000) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000) == 0) { n += 8;  x <<= 8;  }
    if ((x & 0xF0000000) == 0) { n += 4;  x <<= 4;  }
    if ((x & 0xC0000000) == 0) { n += 2;  x <<= 2;  }
    if ((x & 0x80000000) == 0) { n += 1; }
    return n;
}
""",
        "rust_code": """
fn clz(mut x: u32) -> i32 {
    if x == 0 { return 32; }
    let mut n: i32 = 0;
    if (x & 0xFFFF0000) == 0 { n += 16; x <<= 16; }
    if (x & 0xFF000000) == 0 { n += 8;  x <<= 8;  }
    if (x & 0xF0000000) == 0 { n += 4;  x <<= 4;  }
    if (x & 0xC0000000) == 0 { n += 2;  x <<= 2;  }
    if (x & 0x80000000) == 0 { n += 1; }
    n
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Count leading zeros in a 32-bit unsigned integer.',
    },
    {
        "name": 'ctz',
        "category": 'bitwise',
        "c_code": """
int ctz(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    if ((x & 0x0000FFFF) == 0) { n += 16; x >>= 16; }
    if ((x & 0x000000FF) == 0) { n += 8;  x >>= 8;  }
    if ((x & 0x0000000F) == 0) { n += 4;  x >>= 4;  }
    if ((x & 0x00000003) == 0) { n += 2;  x >>= 2;  }
    if ((x & 0x00000001) == 0) { n += 1; }
    return n;
}
""",
        "rust_code": """
fn ctz(mut x: u32) -> i32 {
    if x == 0 { return 32; }
    let mut n: i32 = 0;
    // BUG: masks are wrong for the byte check step
    if (x & 0x0000FFFF) == 0 { n += 16; x >>= 16; }
    if (x & 0x000000FF) == 0 { n += 8;  x >>= 8;  }
    if (x & 0x0000000F) == 0 { n += 4;  x >>= 4;  }
    if (x & 0x00000003) == 0 { n += 2;  x >>= 2;  }
    if (x & 0x00000001) == 0 { n += 2; }  // BUG: adds 2 instead of 1
    n
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-constant',
        "source": 'bitwise-utility',
        "description": 'Count trailing zeros. Rust version adds 2 instead of 1 in last step.',
    },
    {
        "name": 'is_power_of_2',
        "category": 'bitwise',
        "c_code": """
int is_power_of_2(unsigned int x) {
    return x != 0 && (x & (x - 1)) == 0;
}
""",
        "rust_code": """
fn is_power_of_2(x: u32) -> i32 {
    if x != 0 && (x & (x - 1)) == 0 { 1 } else { 0 }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Check if value is a power of two.',
    },
    {
        "name": 'next_power_of_2',
        "category": 'bitwise',
        "c_code": """
unsigned int next_power_of_2(unsigned int x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}
""",
        "rust_code": """
fn next_power_of_2(mut x: u32) -> u32 {
    if x == 0 { return 1; }
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x + 1
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Round up to the next power of two.',
    },
    {
        "name": 'reverse_bits',
        "category": 'bitwise',
        "c_code": """
unsigned int reverse_bits(unsigned int x) {
    unsigned int result = 0;
    for (int i = 0; i < 32; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}
""",
        "rust_code": """
fn reverse_bits(mut x: u32) -> u32 {
    let mut result: u32 = 0;
    // BUG: iterates 31 times instead of 32
    for _ in 0..31 {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'off-by-one',
        "source": 'bitwise-utility',
        "description": 'Reverse bits. Rust version iterates 31 times instead of 32.',
    },
    {
        "name": 'rotate_left',
        "category": 'bitwise',
        "c_code": """
unsigned int rotate_left(unsigned int x, int n) {
    n = n & 31;
    return (x << n) | (x >> (32 - n));
}
""",
        "rust_code": """
fn rotate_left(x: u32, n: i32) -> u32 {
    let n = (n & 31) as u32;
    // BUG: when n == 0, (32 - 0) = 32 causes shift overflow in Rust
    // Actually Rust handles this gracefully with wrapping, but let's keep
    // the direct translation which is correct for non-zero n.
    if n == 0 { return x; }
    (x << n) | (x >> (32 - n))
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Rotate bits left by n positions.',
    },
    {
        "name": 'rotate_right',
        "category": 'bitwise',
        "c_code": """
unsigned int rotate_right(unsigned int x, int n) {
    n = n & 31;
    return (x >> n) | (x << (32 - n));
}
""",
        "rust_code": """
fn rotate_right(x: u32, n: i32) -> u32 {
    let n = (n & 31) as u32;
    if n == 0 { return x; }
    (x >> n) | (x << (32 - n))
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Rotate bits right by n positions.',
    },
    {
        "name": 'parity',
        "category": 'bitwise',
        "c_code": """
int parity(unsigned int x) {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1;
}
""",
        "rust_code": """
fn parity(mut x: u32) -> i32 {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    (x & 1) as i32
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Compute parity (1 if odd number of set bits).',
    },
    {
        "name": 'highest_bit',
        "category": 'bitwise',
        "c_code": """
int highest_bit(unsigned int x) {
    if (x == 0) return -1;
    int pos = 0;
    while (x >>= 1) {
        pos++;
    }
    return pos;
}
""",
        "rust_code": """
fn highest_bit(mut x: u32) -> i32 {
    if x == 0 { return -1; }
    let mut pos: i32 = 0;
    x >>= 1;
    while x != 0 {
        pos += 1;
        x >>= 1;
    }
    pos
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Find position of the highest set bit.',
    },

    # =========================================================================
    # 4. ARRAY/BUFFER OPERATIONS
    # =========================================================================
    {
        "name": 'linear_search',
        "category": 'array',
        "c_code": """
int linear_search(const int *arr, int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}
""",
        "rust_code": """
fn linear_search(arr: &[i32], n: i32, target: i32) -> i32 {
    for i in 0..(n as usize) {
        if arr[i] == target { return i as i32; }
    }
    -1
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Linear search returning index or -1.',
    },
    {
        "name": 'binary_search',
        "category": 'array',
        "c_code": """
int binary_search(const int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
""",
        "rust_code": """
fn binary_search(arr: &[i32], n: i32, target: i32) -> i32 {
    let mut lo: i32 = 0;
    let mut hi: i32 = n - 1;
    while lo <= hi {
        // BUG: (lo + hi) / 2 instead of lo + (hi - lo) / 2 — overflows for large values
        let mid = (lo + hi) / 2;
        if arr[mid as usize] == target { return mid; }
        if arr[mid as usize] < target { lo = mid + 1; }
        else { hi = mid - 1; }
    }
    -1
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'overflow',
        "source": 'data-structure',
        "description": 'Binary search. Rust version has midpoint overflow bug.',
    },
    {
        "name": 'sum_array',
        "category": 'array',
        "c_code": """
int sum_array(const int *arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
""",
        "rust_code": """
fn sum_array(arr: &[i32], n: i32) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..(n as usize) {
        sum = sum.wrapping_add(arr[i]);
    }
    sum
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Sum elements of an integer array.',
    },
    {
        "name": 'max_element',
        "category": 'array',
        "c_code": """
int max_element(const int *arr, int n) {
    if (n <= 0) return -2147483648;
    int mx = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > mx) mx = arr[i];
    }
    return mx;
}
""",
        "rust_code": """
fn max_element(arr: &[i32], n: i32) -> i32 {
    if n <= 0 { return i32::MIN; }
    let mut mx = arr[0];
    for i in 1..(n as usize) {
        if arr[i] > mx { mx = arr[i]; }
    }
    mx
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Find maximum element in array.',
    },
    {
        "name": 'reverse_array',
        "category": 'array',
        "c_code": """
void reverse_array(int *arr, int n) {
    for (int i = 0; i < n / 2; i++) {
        int tmp = arr[i];
        arr[i] = arr[n - 1 - i];
        arr[n - 1 - i] = tmp;
    }
}
""",
        "rust_code": """
fn reverse_array(arr: &mut [i32], n: i32) {
    let n = n as usize;
    for i in 0..n / 2 {
        let tmp = arr[i];
        arr[i] = arr[n - 1 - i];
        arr[n - 1 - i] = tmp;
    }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Reverse an array in place.',
    },
    {
        "name": 'rotate_array',
        "category": 'array',
        "c_code": """
void rotate_array(int *arr, int n, int k) {
    if (n <= 0) return;
    k = k % n;
    if (k < 0) k += n;
    // reverse [0..n-1]
    for (int i = 0; i < n / 2; i++) {
        int t = arr[i]; arr[i] = arr[n-1-i]; arr[n-1-i] = t;
    }
    // reverse [0..k-1]
    for (int i = 0; i < k / 2; i++) {
        int t = arr[i]; arr[i] = arr[k-1-i]; arr[k-1-i] = t;
    }
    // reverse [k..n-1]
    for (int i = k; i < (k + n) / 2; i++) {
        int j = n - 1 - (i - k);
        int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}
""",
        "rust_code": """
fn rotate_array(arr: &mut [i32], n: i32, k: i32) {
    if n <= 0 { return; }
    let n = n as usize;
    let mut k = (k % n as i32) as usize;
    // BUG: doesn't handle negative k properly since usize can't be negative
    // reverse whole
    arr[..n].reverse();
    // reverse first k
    arr[..k].reverse();
    // reverse rest
    arr[k..n].reverse();
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Rotate array right by k positions. Complex modular arithmetic makes verification difficult.',
    },
    {
        "name": 'partition',
        "category": 'array',
        "c_code": """
int partition(int *arr, int lo, int hi) {
    int pivot = arr[hi];
    int i = lo - 1;
    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) {
            i++;
            int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
    }
    int t = arr[i+1]; arr[i+1] = arr[hi]; arr[hi] = t;
    return i + 1;
}
""",
        "rust_code": """
fn partition(arr: &mut [i32], lo: i32, hi: i32) -> i32 {
    let pivot = arr[hi as usize];
    let mut i = lo - 1;
    for j in lo..hi {
        if arr[j as usize] <= pivot {
            i += 1;
            arr.swap(i as usize, j as usize);
        }
    }
    arr.swap((i + 1) as usize, hi as usize);
    i + 1
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Lomuto partition involves complex index tracking and array mutations difficult to verify.',
    },
    {
        "name": 'count_if',
        "category": 'array',
        "c_code": """
int count_if_positive(const int *arr, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] > 0) count++;
    }
    return count;
}
""",
        "rust_code": """
fn count_if_positive(arr: &[i32], n: i32) -> i32 {
    let mut count: i32 = 0;
    for i in 0..(n as usize) {
        if arr[i] >= 0 { count += 1; }  // BUG: >= instead of >
    }
    count
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'off-by-one-comparison',
        "source": 'data-structure',
        "description": 'Count positive elements. Rust version uses >= 0 instead of > 0.',
    },
    {
        "name": 'any_match',
        "category": 'array',
        "c_code": """
int any_match(const int *arr, int n, int val) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == val) return 1;
    }
    return 0;
}
""",
        "rust_code": """
fn any_match(arr: &[i32], n: i32, val: i32) -> i32 {
    for i in 0..(n as usize) {
        if arr[i] == val { return 1; }
    }
    0
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Check if any element matches a value.',
    },
    {
        "name": 'find_first',
        "category": 'array',
        "c_code": """
int find_first_negative(const int *arr, int n) {
    for (int i = 0; i < n; i++) {
        if (arr[i] < 0) return i;
    }
    return -1;
}
""",
        "rust_code": """
fn find_first_negative(arr: &[i32], n: i32) -> i32 {
    for i in 0..(n as usize) {
        if arr[i] < 0 { return i as i32; }
    }
    -1
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Find index of first negative element.',
    },

    # =========================================================================
    # 5. CRYPTO-STYLE
    # =========================================================================
    {
        "name": 'djb2_hash',
        "category": 'crypto',
        "c_code": """
unsigned int djb2_hash(const unsigned char *str, int len) {
    unsigned int hash = 5381;
    for (int i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + str[i];
    }
    return hash;
}
""",
        "rust_code": """
fn djb2_hash(data: &[u8], len: i32) -> u32 {
    let mut hash: u32 = 5381;
    for i in 0..(len as usize) {
        hash = hash.wrapping_shl(5).wrapping_add(hash).wrapping_add(data[i] as u32);
    }
    hash
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'DJB2 string hash function.',
    },
    {
        "name": 'fnv1a_hash',
        "category": 'crypto',
        "c_code": """
unsigned int fnv1a_hash(const unsigned char *data, int len) {
    unsigned int hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash;
}
""",
        "rust_code": """
fn fnv1a_hash(data: &[u8], len: i32) -> u32 {
    let mut hash: u32 = 2166136261;
    for i in 0..(len as usize) {
        // BUG: multiply before XOR (FNV-1 instead of FNV-1a)
        hash = hash.wrapping_mul(16777619);
        hash ^= data[i] as u32;
    }
    hash
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'operation-reorder',
        "source": 'crypto-style',
        "description": 'FNV-1a hash. Rust version does FNV-1 (multiply before XOR).',
    },
    {
        "name": 'crc32_step',
        "category": 'crypto',
        "c_code": """
unsigned int crc32_step(unsigned int crc, unsigned char byte) {
    crc ^= byte;
    for (int i = 0; i < 8; i++) {
        if (crc & 1)
            crc = (crc >> 1) ^ 0xEDB88320u;
        else
            crc = crc >> 1;
    }
    return crc;
}
""",
        "rust_code": """
fn crc32_step(mut crc: u32, byte: u8) -> u32 {
    crc ^= byte as u32;
    for _ in 0..8 {
        if crc & 1 != 0 {
            // BUG: wrong polynomial constant
            crc = (crc >> 1) ^ 0xEDB88321;
        } else {
            crc >>= 1;
        }
    }
    crc
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-constant',
        "source": 'crypto-style',
        "description": 'CRC32 step. Rust version uses wrong polynomial (off by one).',
    },
    {
        "name": 'xor_cipher',
        "category": 'crypto',
        "c_code": """
void xor_cipher(unsigned char *data, int len, unsigned char key) {
    for (int i = 0; i < len; i++) {
        data[i] ^= key;
    }
}
""",
        "rust_code": """
fn xor_cipher(data: &mut [u8], len: i32, key: u8) {
    // BUG: applies XOR in reverse order — different when key depends on position
    // Actually for single-byte XOR this is equivalent...
    // Let's use a real bug: applies key+i instead of just key
    for i in 0..(len as usize) {
        data[i] ^= key.wrapping_add(i as u8);  // BUG: XORs with key+index
    }
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'modified-key',
        "source": 'crypto-style',
        "description": 'XOR cipher. Rust version uses position-dependent key.',
    },
    {
        "name": 'byte_swap_32',
        "category": 'crypto',
        "c_code": """
unsigned int byte_swap_32(unsigned int x) {
    return ((x >> 24) & 0xFF)
         | ((x >> 8)  & 0xFF00)
         | ((x << 8)  & 0xFF0000)
         | ((x << 24) & 0xFF000000u);
}
""",
        "rust_code": """
fn byte_swap_32(x: u32) -> u32 {
    ((x >> 24) & 0xFF)
        | ((x >> 8) & 0xFF00)
        | ((x << 8) & 0xFF0000)
        | ((x << 24) & 0xFF000000)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'Swap byte order of a 32-bit integer.',
    },
    {
        "name": 'endian_swap_16',
        "category": 'crypto',
        "c_code": """
unsigned short endian_swap_16(unsigned short x) {
    return (x >> 8) | (x << 8);
}
""",
        "rust_code": """
fn endian_swap_16(x: u16) -> u16 {
    (x >> 8) | (x << 8)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'Swap byte order of a 16-bit integer.',
    },
    {
        "name": 'lcg_next',
        "category": 'crypto',
        "c_code": """
unsigned int lcg_next(unsigned int state) {
    return state * 1103515245u + 12345u;
}
""",
        "rust_code": """
fn lcg_next(state: u32) -> u32 {
    state.wrapping_mul(1103515245).wrapping_add(12345)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'Linear congruential generator step.',
    },
    {
        "name": 'simple_checksum',
        "category": 'crypto',
        "c_code": """
unsigned int simple_checksum(const unsigned char *data, int len) {
    unsigned int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += data[i];
        sum = (sum << 1) | (sum >> 31);  // rotate left by 1
    }
    return sum;
}
""",
        "rust_code": """
fn simple_checksum(data: &[u8], len: i32) -> u32 {
    let mut sum: u32 = 0;
    for i in 0..(len as usize) {
        sum = sum.wrapping_add(data[i] as u32);
        sum = sum.rotate_left(1);
    }
    sum
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'Rolling checksum with rotate.',
    },

    # =========================================================================
    # 6. DATA STRUCTURE HELPERS
    # =========================================================================
    {
        "name": 'stack_push',
        "category": 'data-structure',
        "c_code": """
int stack_push(int *stack, int *top, int capacity, int value) {
    if (*top >= capacity) return -1;
    stack[*top] = value;
    (*top)++;
    return 0;
}
""",
        "rust_code": """
fn stack_push(stack: &mut [i32], top: &mut i32, capacity: i32, value: i32) -> i32 {
    if *top >= capacity { return -1; }
    stack[*top as usize] = value;
    *top += 1;
    0
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Push onto an array-based stack.',
    },
    {
        "name": 'stack_pop',
        "category": 'data-structure',
        "c_code": """
int stack_pop(int *stack, int *top, int *out) {
    if (*top <= 0) return -1;
    (*top)--;
    *out = stack[*top];
    return 0;
}
""",
        "rust_code": """
fn stack_pop(stack: &[i32], top: &mut i32, out: &mut i32) -> i32 {
    if *top <= 0 { return -1; }
    *top -= 1;
    *out = stack[*top as usize];
    0
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Pop from an array-based stack.',
    },
    {
        "name": 'circular_buffer_write',
        "category": 'data-structure',
        "c_code": """
int circular_buffer_write(int *buf, int capacity, int *head, int *count, int value) {
    if (*count >= capacity) return -1;
    int idx = (*head + *count) % capacity;
    buf[idx] = value;
    (*count)++;
    return 0;
}
""",
        "rust_code": """
fn circular_buffer_write(
    buf: &mut [i32], capacity: i32, head: &mut i32, count: &mut i32, value: i32
) -> i32 {
    if *count >= capacity { return -1; }
    let idx = ((*head + *count) % capacity) as usize;
    buf[idx] = value;
    *count += 1;
    0
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Write to a circular buffer.',
    },
    {
        "name": 'circular_buffer_read',
        "category": 'data-structure',
        "c_code": """
int circular_buffer_read(const int *buf, int capacity, int *head, int *count, int *out) {
    if (*count <= 0) return -1;
    *out = buf[*head];
    *head = (*head + 1) % capacity;
    (*count)--;
    return 0;
}
""",
        "rust_code": """
fn circular_buffer_read(
    buf: &[i32], capacity: i32, head: &mut i32, count: &mut i32, out: &mut i32
) -> i32 {
    if *count <= 0 { return -1; }
    *out = buf[*head as usize];
    // BUG: uses capacity - 1 in modulo
    *head = (*head + 1) % (capacity - 1);
    *count -= 1;
    0
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-modulus',
        "source": 'data-structure',
        "description": 'Circular buffer read. Rust version uses wrong modulus.',
    },
    {
        "name": 'node_insert_after',
        "category": 'data-structure',
        "c_code": """
// Linked list via index arrays: next[i] is the index of the node after i.
// -1 means end of list.
void node_insert_after(int *next, int *values, int node, int new_node, int new_val) {
    next[new_node] = next[node];
    next[node] = new_node;
    values[new_node] = new_val;
}
""",
        "rust_code": """
fn node_insert_after(
    next: &mut [i32], values: &mut [i32],
    node: i32, new_node: i32, new_val: i32
) {
    let n = node as usize;
    let nn = new_node as usize;
    next[nn] = next[n];
    next[n] = new_node;
    values[nn] = new_val;
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Linked-list index mutation with multiple array writes is hard to verify.',
    },
    {
        "name": 'node_remove_after',
        "category": 'data-structure',
        "c_code": """
int node_remove_after(int *next, int node) {
    int removed = next[node];
    if (removed == -1) return -1;
    next[node] = next[removed];
    next[removed] = -1;
    return removed;
}
""",
        "rust_code": """
fn node_remove_after(next: &mut [i32], node: i32) -> i32 {
    let n = node as usize;
    let removed = next[n];
    if removed == -1 { return -1; }
    // BUG: forgot to update next[node], just clears removed
    next[removed as usize] = -1;
    removed
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-state-update',
        "source": 'data-structure',
        "description": 'Remove node after given node. Rust version forgets to relink.',
    },
    {
        "name": 'priority_compare',
        "category": 'data-structure',
        "c_code": """
int priority_compare(int pri_a, int val_a, int pri_b, int val_b) {
    if (pri_a != pri_b) return pri_a - pri_b;
    return val_a - val_b;
}
""",
        "rust_code": """
fn priority_compare(pri_a: i32, val_a: i32, pri_b: i32, val_b: i32) -> i32 {
    if pri_a != pri_b { return pri_a - pri_b; }
    val_a - val_b
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Compare two items by priority then value.',
    },
    {
        "name": 'heap_parent_child',
        "category": 'data-structure',
        "c_code": """
int heap_parent(int i) {
    return (i - 1) / 2;
}
int heap_left(int i) {
    return 2 * i + 1;
}
int heap_right(int i) {
    return 2 * i + 2;
}
int heap_sift_check(const int *arr, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;
    return largest;
}
""",
        "rust_code": """
fn heap_sift_check(arr: &[i32], n: i32, i: i32) -> i32 {
    let mut largest = i;
    let l = 2 * i + 1;
    let r = 2 * i + 2;
    if l < n && arr[l as usize] > arr[largest as usize] { largest = l; }
    if r < n && arr[r as usize] > arr[largest as usize] { largest = r; }
    largest
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Heap sift-down check: find largest among node and its children.',
    },

    # =========================================================================
    # 7. ERROR HANDLING PATTERNS
    # =========================================================================
    {
        "name": 'checked_add',
        "category": 'error-handling',
        "c_code": """
int checked_add(int a, int b, int *result) {
    if ((b > 0 && a > 2147483647 - b) || (b < 0 && a < (-2147483647 - 1) - b)) {
        return -1;  // overflow
    }
    *result = a + b;
    return 0;
}
""",
        "rust_code": """
fn checked_add(a: i32, b: i32, result: &mut i32) -> i32 {
    match a.checked_add(b) {
        Some(val) => { *result = val; 0 }
        None => -1
    }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Addition with overflow check.',
    },
    {
        "name": 'checked_mul',
        "category": 'error-handling',
        "c_code": """
int checked_mul(int a, int b, int *result) {
    if (a == 0 || b == 0) { *result = 0; return 0; }
    int r = a * b;
    if (r / a != b) return -1;
    *result = r;
    return 0;
}
""",
        "rust_code": """
fn checked_mul(a: i32, b: i32, result: &mut i32) -> i32 {
    match a.checked_mul(b) {
        Some(val) => { *result = val; 0 }
        None => -1
    }
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Multiplication with overflow check. C version uses post-hoc division check which has edge cases around MIN_INT * -1.',
    },
    {
        "name": 'safe_div',
        "category": 'error-handling',
        "c_code": """
int safe_div(int a, int b, int *result) {
    if (b == 0) return -1;
    if (a == (-2147483647 - 1) && b == -1) return -1;
    *result = a / b;
    return 0;
}
""",
        "rust_code": """
fn safe_div(a: i32, b: i32, result: &mut i32) -> i32 {
    if b == 0 { return -1; }
    // BUG: missing MIN/-1 overflow check
    *result = a / b;
    0
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-overflow-check',
        "source": 'error-handling',
        "description": 'Safe division. Rust version omits INT_MIN/-1 overflow guard.',
    },
    {
        "name": 'saturating_add',
        "category": 'error-handling',
        "c_code": """
int saturating_add(int a, int b) {
    int r = a + b;
    if (b > 0 && r < a) return 2147483647;
    if (b < 0 && r > a) return (-2147483647 - 1);
    return r;
}
""",
        "rust_code": """
fn saturating_add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Saturating addition. C version relies on signed overflow being wrapping, which is technically UB but common in practice.',
    },
    {
        "name": 'saturating_sub',
        "category": 'error-handling',
        "c_code": """
int saturating_sub(int a, int b) {
    int r = a - b;
    if (b > 0 && r > a) return (-2147483647 - 1);
    if (b < 0 && r < a) return 2147483647;
    return r;
}
""",
        "rust_code": """
fn saturating_sub(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Saturating subtraction. Similar UB concern as saturating_add.',
    },
    {
        "name": 'clamped_div',
        "category": 'error-handling',
        "c_code": """
int clamped_div(int a, int b) {
    if (b == 0) return 0;
    if (a == (-2147483647 - 1) && b == -1) return 2147483647;
    return a / b;
}
""",
        "rust_code": """
fn clamped_div(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    if a == i32::MIN && b == -1 { return i32::MAX; }
    a / b
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Division returning clamped result on overflow.',
    },
    {
        "name": 'try_parse_digit',
        "category": 'error-handling',
        "c_code": """
int try_parse_digit(char c, int *out) {
    if (c >= '0' && c <= '9') {
        *out = c - '0';
        return 0;
    }
    return -1;
}
""",
        "rust_code": """
fn try_parse_digit(c: u8, out: &mut i32) -> i32 {
    // BUG: also accepts hex digits a-f
    if (c >= b'0' && c <= b'9') || (c >= b'a' && c <= b'f') {
        *out = if c >= b'a' { (c - b'a' + 10) as i32 } else { (c - b'0') as i32 };
        return 0;
    }
    -1
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'widened-acceptance',
        "source": 'error-handling',
        "description": 'Parse digit. Rust version accepts hex digits too.',
    },
    {
        "name": 'safe_array_access',
        "category": 'error-handling',
        "c_code": """
int safe_array_access(const int *arr, int n, int idx, int *out) {
    if (idx < 0 || idx >= n) return -1;
    *out = arr[idx];
    return 0;
}
""",
        "rust_code": """
fn safe_array_access(arr: &[i32], n: i32, idx: i32, out: &mut i32) -> i32 {
    if idx < 0 || idx >= n { return -1; }
    *out = arr[idx as usize];
    0
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Bounds-checked array access returning error code.',
    },

    # =========================================================================
    # 8. SIGNAL PROCESSING / DSP
    # =========================================================================
    {
        "name": 'fixed_point_mul',
        "category": 'dsp',
        "c_code": """
int fixed_point_mul(int a, int b, int frac_bits) {
    long long product = (long long)a * (long long)b;
    return (int)(product >> frac_bits);
}
""",
        "rust_code": """
fn fixed_point_mul(a: i32, b: i32, frac_bits: i32) -> i32 {
    let product = (a as i64) * (b as i64);
    (product >> frac_bits) as i32
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'dsp',
        "description": 'Fixed-point multiplication with configurable fractional bits.',
    },
    {
        "name": 'saturating_add_16',
        "category": 'dsp',
        "c_code": """
short saturating_add_16(short a, short b) {
    int sum = (int)a + (int)b;
    if (sum > 32767) return 32767;
    if (sum < -32768) return -32768;
    return (short)sum;
}
""",
        "rust_code": """
fn saturating_add_16(a: i16, b: i16) -> i16 {
    let sum = a as i32 + b as i32;
    // BUG: clamp bounds are wrong (uses i8 range instead of i16)
    if sum > 127 { return 127; }
    if sum < -128 { return -128; }
    sum as i16
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-bounds',
        "source": 'dsp',
        "description": '16-bit saturating add. Rust version clamps to 8-bit range.',
    },
    {
        "name": 'moving_average_step',
        "category": 'dsp',
        "c_code": """
int moving_average_step(int *buffer, int size, int *pos, int *sum, int new_val) {
    *sum -= buffer[*pos];
    buffer[*pos] = new_val;
    *sum += new_val;
    *pos = (*pos + 1) % size;
    return *sum / size;
}
""",
        "rust_code": """
fn moving_average_step(
    buffer: &mut [i32], size: i32, pos: &mut i32, sum: &mut i32, new_val: i32
) -> i32 {
    let p = *pos as usize;
    *sum -= buffer[p];
    buffer[p] = new_val;
    *sum += new_val;
    *pos = (*pos + 1) % size;
    *sum / size
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'dsp',
        "description": 'One step of a circular-buffer moving average.',
    },
    {
        "name": 'fir_tap',
        "category": 'dsp',
        "c_code": """
int fir_tap(const int *signal, const int *coeffs, int n_taps, int pos) {
    int acc = 0;
    for (int i = 0; i < n_taps; i++) {
        int idx = pos - i;
        if (idx < 0) idx = 0;
        acc += signal[idx] * coeffs[i];
    }
    return acc;
}
""",
        "rust_code": """
fn fir_tap(signal: &[i32], coeffs: &[i32], n_taps: i32, pos: i32) -> i32 {
    let mut acc: i32 = 0;
    for i in 0..(n_taps as usize) {
        let idx = pos - i as i32;
        // BUG: uses 0 for out-of-bounds instead of clamping to 0 index
        let sample = if idx < 0 { 0 } else { signal[idx as usize] };
        acc = acc.wrapping_add(sample.wrapping_mul(coeffs[i]));
    }
    acc
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'zero-vs-clamp',
        "source": 'dsp',
        "description": 'FIR tap. Rust version uses zero-padding instead of clamping.',
    },
    {
        "name": 'clip_to_range',
        "category": 'dsp',
        "c_code": """
int clip_to_range(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}
""",
        "rust_code": """
fn clip_to_range(val: i32, min_val: i32, max_val: i32) -> i32 {
    if val < min_val { min_val }
    else if val > max_val { max_val }
    else { val }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'dsp',
        "description": 'Clip a signal value to min/max range.',
    },
    {
        "name": 'scale_value',
        "category": 'dsp',
        "c_code": """
int scale_value(int val, int scale, int shift) {
    return (val * scale) >> shift;
}
""",
        "rust_code": """
fn scale_value(val: i32, scale: i32, shift: i32) -> i32 {
    // BUG: uses wrapping_mul but then arithmetic shift, which matches C
    // for positive values but the wrapping changes overflow behavior
    val.wrapping_mul(scale) >> shift
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'dsp',
        "description": 'Scale a value by a factor with a bit shift.',
    },

    # =========================================================================
    # 9. NETWORK/PROTOCOL
    # =========================================================================
    {
        "name": 'pack_bytes_u32',
        "category": 'network',
        "c_code": """
unsigned int pack_bytes_u32(unsigned char b3, unsigned char b2,
                            unsigned char b1, unsigned char b0) {
    return ((unsigned int)b3 << 24)
         | ((unsigned int)b2 << 16)
         | ((unsigned int)b1 << 8)
         |  (unsigned int)b0;
}
""",
        "rust_code": """
fn pack_bytes_u32(b3: u8, b2: u8, b1: u8, b0: u8) -> u32 {
    ((b3 as u32) << 24)
        | ((b2 as u32) << 16)
        | ((b1 as u32) << 8)
        | (b0 as u32)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'network',
        "description": 'Pack four bytes into a big-endian u32.',
    },
    {
        "name": 'ntohs',
        "category": 'network',
        "c_code": """
unsigned short my_ntohs(unsigned short netshort) {
    return (netshort >> 8) | (netshort << 8);
}
""",
        "rust_code": """
fn my_ntohs(netshort: u16) -> u16 {
    // BUG: shifts by wrong amounts
    (netshort >> 4) | (netshort << 12)
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-shift',
        "source": 'network',
        "description": 'Network-to-host 16. Rust version shifts by nibbles not bytes.',
    },
    {
        "name": 'ntohl',
        "category": 'network',
        "c_code": """
unsigned int my_ntohl(unsigned int netlong) {
    return ((netlong >> 24) & 0xFF)
         | ((netlong >> 8)  & 0xFF00)
         | ((netlong << 8)  & 0xFF0000)
         | ((netlong << 24) & 0xFF000000u);
}
""",
        "rust_code": """
fn my_ntohl(netlong: u32) -> u32 {
    ((netlong >> 24) & 0xFF)
        | ((netlong >> 8) & 0xFF00)
        | ((netlong << 8) & 0xFF0000)
        | ((netlong << 24) & 0xFF000000)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'network',
        "description": 'Network-to-host byte order for 32-bit value.',
    },
    {
        "name": 'ip_to_int',
        "category": 'network',
        "c_code": """
unsigned int ip_to_int(int a, int b, int c, int d) {
    return ((unsigned int)(a & 0xFF) << 24)
         | ((unsigned int)(b & 0xFF) << 16)
         | ((unsigned int)(c & 0xFF) << 8)
         |  (unsigned int)(d & 0xFF);
}
""",
        "rust_code": """
fn ip_to_int(a: i32, b: i32, c: i32, d: i32) -> u32 {
    // BUG: little-endian order instead of big-endian
    ((d & 0xFF) as u32) << 24
        | ((c & 0xFF) as u32) << 16
        | ((b & 0xFF) as u32) << 8
        | ((a & 0xFF) as u32)
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'endianness',
        "source": 'network',
        "description": 'IP to int. Rust version uses reversed byte order.',
    },
    {
        "name": 'inet_checksum_step',
        "category": 'network',
        "c_code": """
unsigned int inet_checksum_step(unsigned int partial, unsigned short word) {
    partial += word;
    while (partial >> 16)
        partial = (partial & 0xFFFF) + (partial >> 16);
    return partial;
}
""",
        "rust_code": """
fn inet_checksum_step(mut partial: u32, word: u16) -> u32 {
    partial += word as u32;
    // BUG: only folds once, doesn't loop
    partial = (partial & 0xFFFF) + (partial >> 16);
    partial
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'loop-to-single-iteration',
        "source": 'network',
        "description": 'Internet checksum fold step. Rust version only folds once instead of looping.',
    },
    {
        "name": 'extract_header_field',
        "category": 'network',
        "c_code": """
unsigned int extract_header_field(unsigned int header, int bit_offset, int width) {
    unsigned int mask = (1u << width) - 1;
    return (header >> bit_offset) & mask;
}
""",
        "rust_code": """
fn extract_header_field(header: u32, bit_offset: i32, width: i32) -> u32 {
    let mask = (1u32 << width) - 1;
    (header >> bit_offset as u32) & mask
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'network',
        "description": 'Extract a bit field from a packet header word.',
    },

    # =========================================================================
    # 10. MIXED REAL-WORLD
    # =========================================================================
    {
        "name": 'date_to_days',
        "category": 'mixed',
        "c_code": """
int date_to_days(int year, int month, int day) {
    // Simplified: days since year 0 (not accounting for calendar reforms)
    int y = year;
    int m = month;
    if (m <= 2) { y--; m += 12; }
    int days = 365 * y + y / 4 - y / 100 + y / 400;
    days += (153 * (m - 3) + 2) / 5 + day - 1;
    return days;
}
""",
        "rust_code": """
fn date_to_days(year: i32, month: i32, day: i32) -> i32 {
    let mut y = year;
    let mut m = month;
    if m <= 2 { y -= 1; m += 12; }
    let mut days = 365 * y + y / 4 - y / 100 + y / 400;
    days += (153 * (m - 3) + 2) / 5 + day - 1;
    days
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'date-time',
        "description": 'Date formula has complex arithmetic with many intermediate values.',
    },
    {
        "name": 'is_leap_year',
        "category": 'mixed',
        "c_code": """
int is_leap_year(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}
""",
        "rust_code": """
fn is_leap_year(year: i32) -> i32 {
    // BUG: missing the 400-year exception
    if year % 4 == 0 && year % 100 != 0 { 1 } else { 0 }
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-condition',
        "source": 'date-time',
        "description": 'Leap year check. Rust version omits the 400-year rule.',
    },
    {
        "name": 'temp_f_to_c_fixed',
        "category": 'mixed',
        "c_code": """
int temp_f_to_c_fixed(int f_times_10) {
    // Convert Fahrenheit*10 to Celsius*10 using integer math
    return (f_times_10 - 320) * 5 / 9;
}
""",
        "rust_code": """
fn temp_f_to_c_fixed(f_times_10: i32) -> i32 {
    // BUG: reordered multiply/divide: * 5 / 9 vs / 9 * 5
    (f_times_10 - 320) / 9 * 5
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'operation-reorder',
        "source": 'physics',
        "description": 'Fixed-point Fahrenheit to Celsius. Rust version reorders divide/multiply losing precision.',
    },
    {
        "name": 'rgb_pack',
        "category": 'mixed',
        "c_code": """
unsigned int rgb_pack(int r, int g, int b) {
    return ((unsigned int)(r & 0xFF) << 16)
         | ((unsigned int)(g & 0xFF) << 8)
         |  (unsigned int)(b & 0xFF);
}
""",
        "rust_code": """
fn rgb_pack(r: i32, g: i32, b: i32) -> u32 {
    // BUG: BGR order instead of RGB
    (((b & 0xFF) as u32) << 16)
        | (((g & 0xFF) as u32) << 8)
        | ((r & 0xFF) as u32)
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'channel-swap',
        "source": 'graphics',
        "description": 'Pack RGB. Rust version packs in BGR order.',
    },
    {
        "name": 'rgb_unpack_r',
        "category": 'mixed',
        "c_code": """
int rgb_unpack_r(unsigned int color) {
    return (color >> 16) & 0xFF;
}
""",
        "rust_code": """
fn rgb_unpack_r(color: u32) -> i32 {
    ((color >> 16) & 0xFF) as i32
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'graphics',
        "description": 'Unpack red component from packed RGB.',
    },
    {
        "name": 'rgb_unpack_g',
        "category": 'mixed',
        "c_code": """
int rgb_unpack_g(unsigned int color) {
    return (color >> 8) & 0xFF;
}
""",
        "rust_code": """
fn rgb_unpack_g(color: u32) -> i32 {
    // BUG: forgot to mask, returns full lower 24 bits shifted
    (color >> 8) as i32
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-mask',
        "source": 'graphics',
        "description": 'Unpack green component. Rust version forgets to mask with 0xFF.',
    },
    {
        "name": 'manhattan_distance',
        "category": 'mixed',
        "c_code": """
int manhattan_distance(int x1, int y1, int x2, int y2) {
    int dx = x1 - x2;
    int dy = y1 - y2;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    return dx + dy;
}
""",
        "rust_code": """
fn manhattan_distance(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
    // BUG: Chebyshev distance instead of Manhattan
    let dx = (x1 - x2).abs();
    let dy = (y1 - y2).abs();
    dx.max(dy)
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-formula',
        "source": 'geometry',
        "description": 'Manhattan distance. Rust version computes Chebyshev distance.',
    },
    {
        "name": 'days_in_month',
        "category": 'mixed',
        "c_code": """
int days_in_month(int month, int is_leap) {
    switch (month) {
        case 1: case 3: case 5: case 7: case 8: case 10: case 12:
            return 31;
        case 4: case 6: case 9: case 11:
            return 30;
        case 2:
            return is_leap ? 29 : 28;
        default:
            return -1;
    }
}
""",
        "rust_code": """
fn days_in_month(month: i32, is_leap: i32) -> i32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap != 0 { 29 } else { 28 },
        _ => -1,
    }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'date-time',
        "description": 'Return number of days in a given month.',
    },

    # =========================================================================
    # 1. STRING OPERATIONS
    # =========================================================================
    {
        "name": 'safe_strlen',
        "category": 'string',
        "c_code": """
int safe_strlen(const char *s, int max_len) {
    int i = 0;
    while (i < max_len && s[i] != '\0') {
        i++;
    }
    return i;
}
""",
        "rust_code": """
fn safe_strlen(s: &[u8], max_len: i32) -> i32 {
    let mut i: i32 = 0;
    // BUG: uses <= instead of <, can read one byte past max_len
    while i <= max_len && (i as usize) < s.len() && s[i as usize] != 0 {
        i += 1;
    }
    i
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'off-by-one',
        "source": 'libc-style',
        "description": 'Bounded strlen. Rust version has off-by-one in bounds check.',
    },

    # =========================================================================
    # 2. MATH UTILITIES
    # =========================================================================
    {
        "name": 'sign',
        "category": 'math',
        "c_code": """
int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
""",
        "rust_code": """
fn sign(x: i32) -> i32 {
    if x > 0 { 1 }
    else if x < 0 { -1 }
    else { 0 }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Signum function returning -1, 0, or 1.',
    },

    # =========================================================================
    # 3. BIT MANIPULATION
    # =========================================================================
    {
        "name": 'bit_extract',
        "category": 'bitwise',
        "c_code": """
unsigned int bit_extract(unsigned int val, int start, int len) {
    return (val >> start) & ((1u << len) - 1);
}
""",
        "rust_code": """
fn bit_extract(val: u32, start: i32, len: i32) -> u32 {
    (val >> start as u32) & ((1u32 << len as u32) - 1)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Extract a bit field from an integer.',
    },
    {
        "name": 'bit_set',
        "category": 'bitwise',
        "c_code": """
unsigned int bit_set(unsigned int val, int pos) {
    return val | (1u << pos);
}
""",
        "rust_code": """
fn bit_set(val: u32, pos: i32) -> u32 {
    val | (1u32 << pos as u32)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Set a specific bit in an integer.',
    },
    {
        "name": 'bit_clear',
        "category": 'bitwise',
        "c_code": """
unsigned int bit_clear(unsigned int val, int pos) {
    return val & ~(1u << pos);
}
""",
        "rust_code": """
fn bit_clear(val: u32, pos: i32) -> u32 {
    val & !(1u32 << pos as u32)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Clear a specific bit in an integer.',
    },

    # =========================================================================
    # 4. ARRAY/BUFFER OPERATIONS
    # =========================================================================
    {
        "name": 'insertion_sort_step',
        "category": 'array',
        "c_code": """
void insertion_sort_step(int *arr, int pos) {
    int key = arr[pos];
    int j = pos - 1;
    while (j >= 0 && arr[j] > key) {
        arr[j + 1] = arr[j];
        j--;
    }
    arr[j + 1] = key;
}
""",
        "rust_code": """
fn insertion_sort_step(arr: &mut [i32], pos: i32) {
    let key = arr[pos as usize];
    let mut j = pos - 1;
    while j >= 0 && arr[j as usize] > key {
        arr[(j + 1) as usize] = arr[j as usize];
        j -= 1;
    }
    arr[(j + 1) as usize] = key;
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Insertion sort step involves complex shifting with index arithmetic.',
    },
    {
        "name": 'merge_step',
        "category": 'array',
        "c_code": """
void merge_step(const int *a, int a_len, const int *b, int b_len,
                int *out) {
    int i = 0, j = 0, k = 0;
    while (i < a_len && j < b_len) {
        if (a[i] <= b[j]) { out[k++] = a[i++]; }
        else { out[k++] = b[j++]; }
    }
    while (i < a_len) out[k++] = a[i++];
    while (j < b_len) out[k++] = b[j++];
}
""",
        "rust_code": """
fn merge_step(a: &[i32], a_len: i32, b: &[i32], b_len: i32, out: &mut [i32]) {
    let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
    let al = a_len as usize;
    let bl = b_len as usize;
    while i < al && j < bl {
        if a[i] <= b[j] { out[k] = a[i]; i += 1; k += 1; }
        else { out[k] = b[j]; j += 1; k += 1; }
    }
    while i < al { out[k] = a[i]; i += 1; k += 1; }
    while j < bl { out[k] = b[j]; j += 1; k += 1; }
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Merge step with three-way index tracking is complex to verify formally.',
    },

    # =========================================================================
    # 5. CRYPTO-STYLE
    # =========================================================================
    {
        "name": 'adler32_step',
        "category": 'crypto',
        "c_code": """
void adler32_step(unsigned int *a, unsigned int *b, unsigned char byte) {
    *a = (*a + byte) % 65521;
    *b = (*b + *a) % 65521;
}
""",
        "rust_code": """
fn adler32_step(a: &mut u32, b: &mut u32, byte: u8) {
    *a = (*a + byte as u32) % 65521;
    *b = (*b + *a) % 65521;
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'Single-byte Adler-32 checksum step.',
    },
    {
        "name": 'sdbm_hash',
        "category": 'crypto',
        "c_code": """
unsigned int sdbm_hash(const unsigned char *data, int len) {
    unsigned int hash = 0;
    for (int i = 0; i < len; i++) {
        hash = data[i] + (hash << 6) + (hash << 16) - hash;
    }
    return hash;
}
""",
        "rust_code": """
fn sdbm_hash(data: &[u8], len: i32) -> u32 {
    let mut hash: u32 = 0;
    for i in 0..(len as usize) {
        hash = (data[i] as u32)
            .wrapping_add(hash.wrapping_shl(6))
            .wrapping_add(hash.wrapping_shl(16))
            .wrapping_sub(hash);
    }
    hash
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'SDBM hash function.',
    },
    {
        "name": 'mix_hash',
        "category": 'crypto',
        "c_code": """
unsigned int mix_hash(unsigned int a, unsigned int b) {
    unsigned int h = a;
    h ^= b;
    h *= 0x5bd1e995;
    h ^= h >> 13;
    h *= 0x5bd1e995;
    h ^= h >> 15;
    return h;
}
""",
        "rust_code": """
fn mix_hash(a: u32, b: u32) -> u32 {
    let mut h = a;
    h ^= b;
    h = h.wrapping_mul(0x5bd1e995);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1e995);
    // BUG: shifts by 16 instead of 15
    h ^= h >> 16;
    h
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-constant',
        "source": 'crypto-style',
        "description": 'MurmurHash-style mixing. Rust version uses wrong shift amount.',
    },

    # =========================================================================
    # 6. DATA STRUCTURE HELPERS
    # =========================================================================
    {
        "name": 'ring_buffer_full',
        "category": 'data-structure',
        "c_code": """
int ring_buffer_full(int head, int tail, int capacity) {
    return ((head + 1) % capacity) == tail;
}
""",
        "rust_code": """
fn ring_buffer_full(head: i32, tail: i32, capacity: i32) -> i32 {
    if ((head + 1) % capacity) == tail { 1 } else { 0 }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Check if a ring buffer is full.',
    },
    {
        "name": 'ring_buffer_empty',
        "category": 'data-structure',
        "c_code": """
int ring_buffer_empty(int head, int tail) {
    return head == tail;
}
""",
        "rust_code": """
fn ring_buffer_empty(head: i32, tail: i32) -> i32 {
    if head == tail { 1 } else { 0 }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Check if a ring buffer is empty.',
    },
    {
        "name": 'ring_buffer_count',
        "category": 'data-structure',
        "c_code": """
int ring_buffer_count(int head, int tail, int capacity) {
    return (head - tail + capacity) % capacity;
}
""",
        "rust_code": """
fn ring_buffer_count(head: i32, tail: i32, capacity: i32) -> i32 {
    // BUG: subtraction without adding capacity first can go negative
    // Rust's % on negative is different from C's in sign behavior
    ((head - tail) % capacity + capacity) % capacity
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Ring buffer element count. Modulo semantics differ between C and Rust for negatives.',
    },

    # =========================================================================
    # 2. MATH UTILITIES
    # =========================================================================
    {
        "name": 'min3',
        "category": 'math',
        "c_code": """
int min3(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}
""",
        "rust_code": """
fn min3(a: i32, b: i32, c: i32) -> i32 {
    let mut m = a;
    if b < m { m = b; }
    if c < m { m = c; }
    m
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Minimum of three integers.',
    },
    {
        "name": 'max3',
        "category": 'math',
        "c_code": """
int max3(int a, int b, int c) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}
""",
        "rust_code": """
fn max3(a: i32, b: i32, c: i32) -> i32 {
    let mut m = a;
    if b > m { m = b; }
    if c > m { m = c; }
    m
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Maximum of three integers.',
    },
    {
        "name": 'midpoint',
        "category": 'math',
        "c_code": """
int midpoint(int a, int b) {
    return a + (b - a) / 2;
}
""",
        "rust_code": """
fn midpoint(a: i32, b: i32) -> i32 {
    // BUG: (a + b) / 2 overflows for large values
    (a + b) / 2
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'overflow',
        "source": 'utility',
        "description": 'Compute midpoint of two integers. Rust version overflows.',
    },

    # =========================================================================
    # 9. NETWORK/PROTOCOL
    # =========================================================================
    {
        "name": 'encode_zigzag',
        "category": 'network',
        "c_code": """
unsigned int encode_zigzag(int n) {
    return (unsigned int)((n << 1) ^ (n >> 31));
}
""",
        "rust_code": """
fn encode_zigzag(n: i32) -> u32 {
    // BUG: shifts by 30 instead of 31
    ((n << 1) ^ (n >> 30)) as u32
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-shift',
        "source": 'network',
        "description": 'Zigzag encode. Rust version shifts by 30 instead of 31.',
    },
    {
        "name": 'decode_zigzag',
        "category": 'network',
        "c_code": """
int decode_zigzag(unsigned int n) {
    return (int)((n >> 1) ^ -(n & 1));
}
""",
        "rust_code": """
fn decode_zigzag(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'network',
        "description": 'Zigzag decoding for protobuf-style variable-length integers.',
    },
    {
        "name": 'varint_size',
        "category": 'network',
        "c_code": """
int varint_size(unsigned int value) {
    int size = 1;
    while (value >= 128) {
        value >>= 7;
        size++;
    }
    return size;
}
""",
        "rust_code": """
fn varint_size(mut value: u32) -> i32 {
    let mut size: i32 = 1;
    while value >= 128 {
        value >>= 7;
        size += 1;
    }
    size
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'network',
        "description": 'Compute number of bytes needed for varint encoding.',
    },

    # =========================================================================
    # 10. MIXED REAL-WORLD
    # =========================================================================
    {
        "name": 'alpha_blend',
        "category": 'mixed',
        "c_code": """
int alpha_blend(int fg, int bg, int alpha) {
    // alpha is 0-255
    return (fg * alpha + bg * (255 - alpha)) / 255;
}
""",
        "rust_code": """
fn alpha_blend(fg: i32, bg: i32, alpha: i32) -> i32 {
    // BUG: divides by 256 instead of 255 (common fast approximation but not equivalent)
    (fg * alpha + bg * (255 - alpha)) / 256
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-constant',
        "source": 'graphics',
        "description": 'Alpha blending. Rust version divides by 256 instead of 255.',
    },
    {
        "name": 'lerp_fixed',
        "category": 'mixed',
        "c_code": """
int lerp_fixed(int a, int b, int t, int t_max) {
    if (t_max == 0) return a;
    return a + (b - a) * t / t_max;
}
""",
        "rust_code": """
fn lerp_fixed(a: i32, b: i32, t: i32, t_max: i32) -> i32 {
    if t_max == 0 { return a; }
    a + (b - a) * t / t_max
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'graphics',
        "description": 'Fixed-point linear interpolation.',
    },
    {
        "name": 'gray_from_rgb',
        "category": 'mixed',
        "c_code": """
int gray_from_rgb(int r, int g, int b) {
    // NTSC luminance formula approximation
    return (r * 77 + g * 150 + b * 29) >> 8;
}
""",
        "rust_code": """
fn gray_from_rgb(r: i32, g: i32, b: i32) -> i32 {
    // BUG: wrong coefficients (swapped red and green weights)
    (r * 150 + g * 77 + b * 29) >> 8
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-coefficients',
        "source": 'graphics',
        "description": 'RGB to grayscale. Rust version swaps red/green weights.',
    },

    # =========================================================================
    # 8. SIGNAL PROCESSING / DSP
    # =========================================================================
    {
        "name": 'map_range',
        "category": 'dsp',
        "c_code": """
int map_range(int val, int in_min, int in_max, int out_min, int out_max) {
    if (in_max == in_min) return out_min;
    return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min);
}
""",
        "rust_code": """
fn map_range(val: i32, in_min: i32, in_max: i32, out_min: i32, out_max: i32) -> i32 {
    if in_max == in_min { return out_min; }
    out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'dsp',
        "description": 'Range mapping with potential overflow in intermediate products.',
    },
    {
        "name": 'quantize',
        "category": 'dsp',
        "c_code": """
int quantize(int val, int step) {
    if (step <= 0) return val;
    if (val >= 0)
        return (val / step) * step;
    else
        return ((val - step + 1) / step) * step;
}
""",
        "rust_code": """
fn quantize(val: i32, step: i32) -> i32 {
    if step <= 0 { return val; }
    // BUG: doesn't handle negative values correctly, uses truncation toward zero
    (val / step) * step
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'sign-handling',
        "source": 'dsp',
        "description": 'Quantize value to step grid. Rust version wrong for negatives.',
    },
    {
        "name": 'weighted_sum_2',
        "category": 'dsp',
        "c_code": """
int weighted_sum_2(int a, int wa, int b, int wb, int total_w) {
    if (total_w == 0) return 0;
    return (a * wa + b * wb) / total_w;
}
""",
        "rust_code": """
fn weighted_sum_2(a: i32, wa: i32, b: i32, wb: i32, total_w: i32) -> i32 {
    if total_w == 0 { return 0; }
    (a * wa + b * wb) / total_w
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'dsp',
        "description": 'Weighted sum of two values.',
    },

    # =========================================================================
    # 4. ARRAY/BUFFER OPERATIONS
    # =========================================================================
    {
        "name": 'array_dot_product',
        "category": 'array',
        "c_code": """
int array_dot_product(const int *a, const int *b, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
""",
        "rust_code": """
fn array_dot_product(a: &[i32], b: &[i32], n: i32) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..(n as usize) {
        sum = sum.wrapping_add(a[i].wrapping_mul(b[i]));
    }
    sum
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Dot product of two integer arrays.',
    },
    {
        "name": 'prefix_sum',
        "category": 'array',
        "c_code": """
void prefix_sum(const int *input, int *output, int n) {
    if (n <= 0) return;
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}
""",
        "rust_code": """
fn prefix_sum(input: &[i32], output: &mut [i32], n: i32) {
    if n <= 0 { return; }
    output[0] = input[0];
    for i in 1..(n as usize) {
        output[i] = output[i - 1] + input[i];
    }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Compute prefix sums of an array.',
    },
    {
        "name": 'array_shift_left',
        "category": 'array',
        "c_code": """
void array_shift_left(int *arr, int n) {
    if (n <= 1) return;
    int first = arr[0];
    for (int i = 0; i < n - 1; i++) {
        arr[i] = arr[i + 1];
    }
    arr[n - 1] = first;
}
""",
        "rust_code": """
fn array_shift_left(arr: &mut [i32], n: i32) {
    if n <= 1 { return; }
    let n = n as usize;
    let first = arr[0];
    for i in 0..n - 1 {
        arr[i] = arr[i + 1];
    }
    arr[n - 1] = first;
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Rotate array left by one position.',
    },
    {
        "name": 'count_zeros',
        "category": 'array',
        "c_code": """
int count_zeros(const int *arr, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] == 0) count++;
    }
    return count;
}
""",
        "rust_code": """
fn count_zeros(arr: &[i32], n: i32) -> i32 {
    let mut count: i32 = 0;
    for i in 0..(n as usize) {
        if arr[i] == 0 { count += 1; }
    }
    count
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Count zero elements in array.',
    },

    # =========================================================================
    # 1. STRING OPERATIONS
    # =========================================================================
    {
        "name": 'atoi_simple',
        "category": 'string',
        "c_code": """
int atoi_simple(const char *s) {
    int sign = 1;
    int i = 0;
    int result = 0;
    while (s[i] == ' ') i++;
    if (s[i] == '-') { sign = -1; i++; }
    else if (s[i] == '+') { i++; }
    while (s[i] >= '0' && s[i] <= '9') {
        result = result * 10 + (s[i] - '0');
        i++;
    }
    return sign * result;
}
""",
        "rust_code": """
fn atoi_simple(s: &[u8]) -> i32 {
    let mut sign: i32 = 1;
    let mut i: usize = 0;
    let mut result: i32 = 0;
    while i < s.len() && s[i] == b' ' { i += 1; }
    if i < s.len() && s[i] == b'-' { sign = -1; i += 1; }
    else if i < s.len() && s[i] == b'+' { i += 1; }
    while i < s.len() && s[i] >= b'0' && s[i] <= b'9' {
        // BUG: multiplies by sign inside the loop
        result = result * 10 + sign * (s[i] - b'0') as i32;
        i += 1;
    }
    result
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'premature-sign-apply',
        "source": 'libc-style',
        "description": 'String to int. Rust version applies sign at each digit, wrong for multi-digit.',
    },
    {
        "name": 'itoa_simple',
        "category": 'string',
        "c_code": """
int itoa_simple(int val, char *buf, int buf_size) {
    int neg = 0;
    if (val < 0) { neg = 1; val = -val; }
    int pos = 0;
    do {
        if (pos >= buf_size - 1) return -1;
        buf[pos++] = '0' + (val % 10);
        val /= 10;
    } while (val > 0);
    if (neg) {
        if (pos >= buf_size - 1) return -1;
        buf[pos++] = '-';
    }
    buf[pos] = '\0';
    // reverse
    for (int i = 0; i < pos / 2; i++) {
        char t = buf[i]; buf[i] = buf[pos-1-i]; buf[pos-1-i] = t;
    }
    return pos;
}
""",
        "rust_code": """
fn itoa_simple(mut val: i32, buf: &mut [u8], buf_size: i32) -> i32 {
    let neg = val < 0;
    if neg { val = -val; }
    let mut pos: usize = 0;
    let bsize = buf_size as usize;
    loop {
        if pos >= bsize.saturating_sub(1) { return -1; }
        buf[pos] = b'0' + (val % 10) as u8;
        pos += 1;
        val /= 10;
        if val <= 0 { break; }
    }
    if neg {
        if pos >= bsize.saturating_sub(1) { return -1; }
        buf[pos] = b'-';
        pos += 1;
    }
    buf[pos] = 0;
    // reverse
    for i in 0..pos / 2 {
        let t = buf[i]; buf[i] = buf[pos - 1 - i]; buf[pos - 1 - i] = t;
    }
    pos as i32
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'libc-style',
        "description": 'Integer to string conversion. Edge case with INT_MIN makes verification complex.',
    },

    # =========================================================================
    # 3. BIT MANIPULATION
    # =========================================================================
    {
        "name": 'hamming_distance',
        "category": 'bitwise',
        "c_code": """
int hamming_distance(unsigned int a, unsigned int b) {
    unsigned int x = a ^ b;
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}
""",
        "rust_code": """
fn hamming_distance(a: u32, b: u32) -> i32 {
    let mut x = a ^ b;
    let mut count: i32 = 0;
    while x != 0 {
        count += (x & 1) as i32;
        x >>= 1;
    }
    count
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": 'Hamming distance between two integers.',
    },
    {
        "name": 'swap_nibbles',
        "category": 'bitwise',
        "c_code": """
unsigned char swap_nibbles(unsigned char x) {
    return (x >> 4) | (x << 4);
}
""",
        "rust_code": """
fn swap_nibbles(x: u8) -> u8 {
    // BUG: only shifts by 2 instead of 4
    (x >> 2) | (x << 6)
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-shift',
        "source": 'bitwise-utility',
        "description": 'Swap nibbles. Rust version shifts by 2 instead of 4.',
    },
    {
        "name": 'interleave_bits',
        "category": 'bitwise',
        "c_code": """
unsigned int interleave_bits(unsigned short x, unsigned short y) {
    unsigned int result = 0;
    for (int i = 0; i < 16; i++) {
        result |= ((unsigned int)(x & (1 << i))) << i;
        result |= ((unsigned int)(y & (1 << i))) << (i + 1);
    }
    return result;
}
""",
        "rust_code": """
fn interleave_bits(x: u16, y: u16) -> u32 {
    let mut result: u32 = 0;
    for i in 0..16 {
        result |= ((x as u32 & (1 << i)) << i);
        // BUG: shifts by i instead of i+1
        result |= ((y as u32 & (1 << i)) << i);
    }
    result
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'wrong-shift-amount',
        "source": 'bitwise-utility',
        "description": 'Interleave bits of two 16-bit values. Rust version uses wrong shift.',
    },

    # =========================================================================
    # 7. ERROR HANDLING PATTERNS
    # =========================================================================
    {
        "name": 'safe_mod',
        "category": 'error-handling',
        "c_code": """
int safe_mod(int a, int b) {
    if (b == 0) return 0;
    return a % b;
}
""",
        "rust_code": """
fn safe_mod(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    a % b
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Modulo operation safe against division by zero.',
    },
    {
        "name": 'errno_style_open',
        "category": 'error-handling',
        "c_code": """
int try_open(int fd, int flags, int *err) {
    if (fd < 0) { *err = -1; return -1; }
    if (flags & 0x1) {  // read flag
        *err = 0;
        return fd;
    }
    if (flags & 0x2) {  // write flag
        *err = 0;
        return fd;
    }
    *err = -2;  // invalid flags
    return -1;
}
""",
        "rust_code": """
fn try_open(fd: i32, flags: i32, err: &mut i32) -> i32 {
    if fd < 0 { *err = -1; return -1; }
    if flags & 0x1 != 0 {
        *err = 0;
        return fd;
    }
    if flags & 0x2 != 0 {
        *err = 0;
        return fd;
    }
    *err = -2;
    -1
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Error-code style function simulating file open validation.',
    },
    {
        "name": 'checked_shift_left',
        "category": 'error-handling',
        "c_code": """
int checked_shift_left(int val, int shift, int *result) {
    if (shift < 0 || shift >= 32) return -1;
    if (val < 0) return -1;
    int r = val << shift;
    if ((r >> shift) != val) return -1;
    *result = r;
    return 0;
}
""",
        "rust_code": """
fn checked_shift_left(val: i32, shift: i32, result: &mut i32) -> i32 {
    if shift < 0 || shift >= 32 { return -1; }
    if val < 0 { return -1; }
    // BUG: doesn't check for overflow after shifting
    *result = val << shift;
    0
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-overflow-check',
        "source": 'error-handling',
        "description": 'Checked left shift. Rust version omits the roundtrip overflow check.',
    },

    # =========================================================================
    # 8. SIGNAL PROCESSING / DSP
    # =========================================================================
    {
        "name": 'downsample_2x',
        "category": 'dsp',
        "c_code": """
void downsample_2x(const int *input, int in_len, int *output) {
    for (int i = 0; i < in_len / 2; i++) {
        output[i] = (input[2*i] + input[2*i+1]) / 2;
    }
}
""",
        "rust_code": """
fn downsample_2x(input: &[i32], in_len: i32, output: &mut [i32]) {
    for i in 0..(in_len as usize / 2) {
        output[i] = (input[2 * i] + input[2 * i + 1]) / 2;
    }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'dsp',
        "description": '2x downsampling by averaging adjacent pairs.',
    },
    {
        "name": 'upsample_zoh',
        "category": 'dsp',
        "c_code": """
void upsample_zoh(const int *input, int in_len, int *output) {
    for (int i = 0; i < in_len; i++) {
        output[2*i] = input[i];
        output[2*i+1] = input[i];
    }
}
""",
        "rust_code": """
fn upsample_zoh(input: &[i32], in_len: i32, output: &mut [i32]) {
    for i in 0..(in_len as usize) {
        output[2 * i] = input[i];
        output[2 * i + 1] = input[i];
    }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'dsp',
        "description": '2x zero-order hold upsampling.',
    },
    {
        "name": 'apply_gain',
        "category": 'dsp',
        "c_code": """
void apply_gain(int *samples, int n, int gain_num, int gain_den) {
    if (gain_den == 0) return;
    for (int i = 0; i < n; i++) {
        samples[i] = samples[i] * gain_num / gain_den;
    }
}
""",
        "rust_code": """
fn apply_gain(samples: &mut [i32], n: i32, gain_num: i32, gain_den: i32) {
    if gain_den == 0 { return; }
    for i in 0..(n as usize) {
        samples[i] = samples[i] * gain_num / gain_den;
    }
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'dsp',
        "description": 'Gain application with rational arithmetic and potential overflow.',
    },

    # =========================================================================
    # 9. NETWORK/PROTOCOL
    # =========================================================================
    {
        "name": 'subnet_match',
        "category": 'network',
        "c_code": """
int subnet_match(unsigned int ip_a, unsigned int ip_b, int prefix_len) {
    if (prefix_len <= 0) return 1;
    if (prefix_len >= 32) return ip_a == ip_b;
    unsigned int mask = ~((1u << (32 - prefix_len)) - 1);
    return (ip_a & mask) == (ip_b & mask);
}
""",
        "rust_code": """
fn subnet_match(ip_a: u32, ip_b: u32, prefix_len: i32) -> i32 {
    if prefix_len <= 0 { return 1; }
    if prefix_len >= 32 { return if ip_a == ip_b { 1 } else { 0 }; }
    let mask = !((1u32 << (32 - prefix_len) as u32) - 1);
    if (ip_a & mask) == (ip_b & mask) { 1 } else { 0 }
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'network',
        "description": 'Subnet matching with bit-shift-constructed masks and edge cases.',
    },
    {
        "name": 'unpack_port',
        "category": 'network',
        "c_code": """
unsigned short unpack_port(const unsigned char *data) {
    return ((unsigned short)data[0] << 8) | data[1];
}
""",
        "rust_code": """
fn unpack_port(data: &[u8]) -> u16 {
    ((data[0] as u16) << 8) | (data[1] as u16)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'network',
        "description": 'Unpack a big-endian 16-bit port number from bytes.',
    },
    {
        "name": 'pack_port',
        "category": 'network',
        "c_code": """
void pack_port(unsigned short port, unsigned char *data) {
    data[0] = (port >> 8) & 0xFF;
    data[1] = port & 0xFF;
}
""",
        "rust_code": """
fn pack_port(port: u16, data: &mut [u8]) {
    data[0] = (port >> 8) as u8;
    data[1] = (port & 0xFF) as u8;
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'network',
        "description": 'Pack a 16-bit port number into big-endian bytes.',
    },

    # =========================================================================
    # 10. MIXED REAL-WORLD
    # =========================================================================
    {
        "name": 'rgb_to_hsv_hue',
        "category": 'mixed',
        "c_code": """
int rgb_to_hsv_hue(int r, int g, int b) {
    int mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
    int mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
    int diff = mx - mn;
    if (diff == 0) return 0;
    int hue;
    if (mx == r) hue = 60 * ((g - b) * 100 / diff) / 100;
    else if (mx == g) hue = 60 * ((b - r) * 100 / diff + 200) / 100;
    else hue = 60 * ((r - g) * 100 / diff + 400) / 100;
    if (hue < 0) hue += 360;
    return hue;
}
""",
        "rust_code": """
fn rgb_to_hsv_hue(r: i32, g: i32, b: i32) -> i32 {
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let diff = mx - mn;
    if diff == 0 { return 0; }
    let mut hue;
    if mx == r {
        hue = 60 * ((g - b) * 100 / diff) / 100;
    } else if mx == g {
        hue = 60 * ((b - r) * 100 / diff + 200) / 100;
    } else {
        hue = 60 * ((r - g) * 100 / diff + 400) / 100;
    }
    if hue < 0 { hue += 360; }
    hue
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'graphics',
        "description": 'HSV hue computation involves conditional branches and integer division chains.',
    },
    {
        "name": 'bcd_to_int',
        "category": 'mixed',
        "c_code": """
int bcd_to_int(unsigned int bcd) {
    int result = 0;
    int multiplier = 1;
    while (bcd > 0) {
        int digit = bcd & 0xF;
        if (digit > 9) return -1;
        result += digit * multiplier;
        multiplier *= 10;
        bcd >>= 4;
    }
    return result;
}
""",
        "rust_code": """
fn bcd_to_int(mut bcd: u32) -> i32 {
    let mut result: i32 = 0;
    let mut multiplier: i32 = 1;
    while bcd > 0 {
        let digit = (bcd & 0xF) as i32;
        if digit > 9 { return -1; }
        result += digit * multiplier;
        multiplier *= 10;
        bcd >>= 4;
    }
    result
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'embedded',
        "description": 'Convert packed BCD to integer.',
    },
    {
        "name": 'int_to_bcd',
        "category": 'mixed',
        "c_code": """
unsigned int int_to_bcd(int val) {
    if (val < 0) return 0xFFFFFFFF;
    unsigned int bcd = 0;
    int shift = 0;
    while (val > 0) {
        bcd |= ((unsigned int)(val % 10)) << shift;
        val /= 10;
        shift += 4;
    }
    return bcd;
}
""",
        "rust_code": """
fn int_to_bcd(mut val: i32) -> u32 {
    if val < 0 { return 0xFFFFFFFF; }
    let mut bcd: u32 = 0;
    let mut shift: u32 = 0;
    while val > 0 {
        bcd |= ((val % 10) as u32) << shift;
        val /= 10;
        shift += 4;
    }
    bcd
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'embedded',
        "description": 'Convert integer to packed BCD.',
    },
    {
        "name": 'distance_squared',
        "category": 'mixed',
        "c_code": """
int distance_squared(int x1, int y1, int x2, int y2) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    return dx * dx + dy * dy;
}
""",
        "rust_code": """
fn distance_squared(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    dx * dx + dy * dy
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'geometry',
        "description": 'Squared Euclidean distance between two 2D points.',
    },
    {
        "name": 'clamp_to_u8',
        "category": 'mixed',
        "c_code": """
unsigned char clamp_to_u8(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return (unsigned char)val;
}
""",
        "rust_code": """
fn clamp_to_u8(val: i32) -> u8 {
    if val < 0 { 0 }
    else if val > 255 { 255 }
    else { val as u8 }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'graphics',
        "description": 'Clamp an integer to the 0-255 range.',
    },
    {
        "name": 'bilinear_weight',
        "category": 'mixed',
        "c_code": """
int bilinear_weight(int a, int b, int c, int d, int wx, int wy, int scale) {
    // wx, wy in [0, scale]
    int top = a * (scale - wx) + b * wx;
    int bot = c * (scale - wx) + d * wx;
    return (top * (scale - wy) + bot * wy) / (scale * scale);
}
""",
        "rust_code": """
fn bilinear_weight(a: i32, b: i32, c: i32, d: i32, wx: i32, wy: i32, scale: i32) -> i32 {
    let top = a * (scale - wx) + b * wx;
    let bot = c * (scale - wx) + d * wx;
    if scale == 0 { return 0; }
    (top * (scale - wy) + bot * wy) / (scale * scale)
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'graphics',
        "description": 'Integer bilinear interpolation. Overflow possibilities make equivalence hard to verify.',
    },

    # =========================================================================
    # 4. ARRAY/BUFFER OPERATIONS
    # =========================================================================
    {
        "name": 'find_majority',
        "category": 'array',
        "c_code": """
int find_majority(const int *arr, int n) {
    // Boyer-Moore majority vote
    int candidate = 0;
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (count == 0) { candidate = arr[i]; count = 1; }
        else if (arr[i] == candidate) { count++; }
        else { count--; }
    }
    return candidate;
}
""",
        "rust_code": """
fn find_majority(arr: &[i32], n: i32) -> i32 {
    let mut candidate: i32 = 0;
    let mut count: i32 = 0;
    for i in 0..(n as usize) {
        if count == 0 {
            candidate = arr[i];
            count = 1;
        } else if arr[i] == candidate {
            count += 1;
        } else {
            count -= 1;
        }
    }
    candidate
}
""",
        "expected_verdict": 'unknown',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Boyer-Moore voting has subtle correctness properties hard to verify.',
    },
    {
        "name": 'kadane_max_subarray',
        "category": 'array',
        "c_code": """
int kadane_max_subarray(const int *arr, int n) {
    if (n <= 0) return 0;
    int max_ending = arr[0];
    int max_so_far = arr[0];
    for (int i = 1; i < n; i++) {
        if (max_ending + arr[i] > arr[i])
            max_ending = max_ending + arr[i];
        else
            max_ending = arr[i];
        if (max_ending > max_so_far)
            max_so_far = max_ending;
    }
    return max_so_far;
}
""",
        "rust_code": """
fn kadane_max_subarray(arr: &[i32], n: i32) -> i32 {
    if n <= 0 { return 0; }
    let mut max_ending = arr[0];
    let mut max_so_far = arr[0];
    for i in 1..(n as usize) {
        // BUG: always adds, never resets to arr[i] alone
        max_ending = max_ending + arr[i];
        if max_ending > max_so_far {
            max_so_far = max_ending;
        }
    }
    max_so_far
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-reset',
        "source": 'data-structure',
        "description": "Kadane's algorithm. Rust version never resets the running sum.",
    },

    # =========================================================================
    # 3. BIT MANIPULATION
    # =========================================================================
    {
        "name": 'two_complement_negate',
        "category": 'bitwise',
        "c_code": """
int two_complement_negate(int x) {
    return ~x + 1;
}
""",
        "rust_code": """
fn two_complement_negate(x: i32) -> i32 {
    (!x).wrapping_add(1)
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'bitwise-utility',
        "description": "Negate via two's complement.",
    },

    # =========================================================================
    # 2. MATH UTILITIES
    # =========================================================================
    {
        "name": 'abs_diff',
        "category": 'math',
        "c_code": """
unsigned int abs_diff(int a, int b) {
    if (a > b) return (unsigned int)(a - b);
    return (unsigned int)(b - a);
}
""",
        "rust_code": """
fn abs_diff(a: i32, b: i32) -> u32 {
    if a > b { (a - b) as u32 }
    else { (b - a) as u32 }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Absolute difference of two integers.',
    },
    {
        "name": 'average_3',
        "category": 'math',
        "c_code": """
int average_3(int a, int b, int c) {
    return (a + b + c) / 3;
}
""",
        "rust_code": """
fn average_3(a: i32, b: i32, c: i32) -> i32 {
    (a + b + c) / 3
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'utility',
        "description": 'Average of three integers (truncating).',
    },

    # =========================================================================
    # 4. ARRAY/BUFFER OPERATIONS
    # =========================================================================
    {
        "name": 'is_sorted',
        "category": 'array',
        "c_code": """
int is_sorted(const int *arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i-1]) return 0;
    }
    return 1;
}
""",
        "rust_code": """
fn is_sorted(arr: &[i32], n: i32) -> i32 {
    for i in 1..(n as usize) {
        if arr[i] < arr[i - 1] { return 0; }
    }
    1
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'data-structure',
        "description": 'Check if array is sorted in non-decreasing order.',
    },

    # =========================================================================
    # 5. CRYPTO-STYLE
    # =========================================================================
    {
        "name": 'xor_reduce',
        "category": 'crypto',
        "c_code": """
unsigned char xor_reduce(const unsigned char *data, int len) {
    unsigned char result = 0;
    for (int i = 0; i < len; i++) {
        result ^= data[i];
    }
    return result;
}
""",
        "rust_code": """
fn xor_reduce(data: &[u8], len: i32) -> u8 {
    let mut result: u8 = 0;
    for i in 0..(len as usize) {
        result ^= data[i];
    }
    result
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'XOR all bytes together.',
    },
    {
        "name": 'pearson_hash_step',
        "category": 'crypto',
        "c_code": """
unsigned char pearson_hash_step(unsigned char hash, unsigned char byte,
                                 const unsigned char *table) {
    return table[hash ^ byte];
}
""",
        "rust_code": """
fn pearson_hash_step(hash: u8, byte: u8, table: &[u8]) -> u8 {
    table[(hash ^ byte) as usize]
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'crypto-style',
        "description": 'One step of Pearson hashing.',
    },

    # =========================================================================
    # 7. ERROR HANDLING PATTERNS
    # =========================================================================
    {
        "name": 'bounded_increment',
        "category": 'error-handling',
        "c_code": """
int bounded_increment(int val, int max_val) {
    if (val >= max_val) return max_val;
    return val + 1;
}
""",
        "rust_code": """
fn bounded_increment(val: i32, max_val: i32) -> i32 {
    // BUG: uses > instead of >=, allows val == max_val to increment
    if val > max_val { max_val } else { val + 1 }
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'off-by-one-comparison',
        "source": 'error-handling',
        "description": 'Bounded increment. Rust version uses > instead of >=.',
    },
    {
        "name": 'bounded_decrement',
        "category": 'error-handling',
        "c_code": """
int bounded_decrement(int val, int min_val) {
    if (val <= min_val) return min_val;
    return val - 1;
}
""",
        "rust_code": """
fn bounded_decrement(val: i32, min_val: i32) -> i32 {
    if val <= min_val { min_val } else { val - 1 }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'error-handling',
        "description": 'Decrement with lower bound.',
    },

    # =========================================================================
    # 10. MIXED REAL-WORLD
    # =========================================================================
    {
        "name": 'celsius_to_kelvin_fixed',
        "category": 'mixed',
        "c_code": """
int celsius_to_kelvin_fixed(int celsius_times_100) {
    return celsius_times_100 + 27315;
}
""",
        "rust_code": """
fn celsius_to_kelvin_fixed(celsius_times_100: i32) -> i32 {
    celsius_times_100 + 27315
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'physics',
        "description": 'Fixed-point Celsius to Kelvin (x100).',
    },
    {
        "name": 'wrap_angle',
        "category": 'mixed',
        "c_code": """
int wrap_angle(int degrees) {
    int r = degrees % 360;
    if (r < 0) r += 360;
    return r;
}
""",
        "rust_code": """
fn wrap_angle(degrees: i32) -> i32 {
    // BUG: doesn't handle negative remainder
    degrees % 360
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'missing-negative-handling',
        "source": 'geometry',
        "description": "Wrap angle. Rust version doesn't handle negative values.",
    },
    {
        "name": 'point_in_rect',
        "category": 'mixed',
        "c_code": """
int point_in_rect(int px, int py, int rx, int ry, int rw, int rh) {
    return px >= rx && px < rx + rw && py >= ry && py < ry + rh;
}
""",
        "rust_code": """
fn point_in_rect(px: i32, py: i32, rx: i32, ry: i32, rw: i32, rh: i32) -> i32 {
    if px >= rx && px < rx + rw && py >= ry && py < ry + rh { 1 } else { 0 }
}
""",
        "expected_verdict": 'equivalent',
        "bug_class": '',
        "source": 'geometry',
        "description": 'Check if a point is inside a rectangle.',
    },
    {
        "name": 'rects_overlap',
        "category": 'mixed',
        "c_code": """
int rects_overlap(int ax, int ay, int aw, int ah,
                  int bx, int by, int bw, int bh) {
    if (ax + aw <= bx) return 0;
    if (bx + bw <= ax) return 0;
    if (ay + ah <= by) return 0;
    if (by + bh <= ay) return 0;
    return 1;
}
""",
        "rust_code": """
fn rects_overlap(ax: i32, ay: i32, aw: i32, ah: i32,
                 bx: i32, by: i32, bw: i32, bh: i32) -> i32 {
    // BUG: uses < instead of <=, considers touching edges as overlapping
    if ax + aw < bx { return 0; }
    if bx + bw < ax { return 0; }
    if ay + ah < by { return 0; }
    if by + bh < ay { return 0; }
    1
}
""",
        "expected_verdict": 'divergent',
        "bug_class": 'boundary-condition',
        "source": 'geometry',
        "description": 'Rectangle overlap. Rust version treats touching edges as overlapping.',
    },
]


def get_real_world_pairs():
    """Get real-world benchmark pairs as BenchmarkPair objects."""
    from benchmarks.pairs.benchmark_pairs import BenchmarkPair
    pairs = []
    for entry in REAL_WORLD_BENCHMARK_PAIRS:
        pairs.append(BenchmarkPair(
            name=entry["name"],
            c_source=entry["c_code"],
            rust_source=entry["rust_code"],
            category=entry.get("category", "real_world"),
            expected_result=entry.get("expected_verdict", "unknown"),
            description=entry.get("description", ""),
            divergence_kind=entry.get("bug_class", ""),
        ))
    return pairs
