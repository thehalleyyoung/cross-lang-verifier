/*
 * Integer overflow divergence — inspired by CVE-2014-1266 (Apple goto fail)
 * and CVE-2021-21300 (Git integer overflow in path length).
 *
 * In C, signed integer overflow is UB per C11 §6.5/5.
 * Compilers may optimize assuming it never happens (e.g. elide overflow checks).
 * In Rust, signed overflow panics in debug, wraps in release (two's complement).
 */
#include <stdint.h>
#include <stdlib.h>

/* CVE-style: allocate buffer for n items of size s.
   Overflow in n*s can produce a small allocation, then OOB write. */
void *alloc_items(int32_t n, int32_t item_size) {
    /* A real CVE pattern: if n*item_size overflows, the malloc gets
       a tiny buffer but the caller writes n*item_size bytes. */
    int32_t total = n * item_size;  /* UB if overflow in C */
    if (total < 0) return NULL;     /* compiler may remove: UB means no overflow */
    return malloc((size_t)total);
}

/* Midpoint computation — the classic binary search bug (JDK-5045582).
   In C: (low + high) overflows for large values → UB.
   Fix in Rust is trivial with checked arithmetic. */
int32_t midpoint(int32_t low, int32_t high) {
    return (low + high) / 2;  /* UB when low + high > INT32_MAX */
}

/* Absolute value of INT32_MIN is UB: -(-2147483648) overflows. */
int32_t safe_abs(int32_t x) {
    if (x < 0) return -x;  /* UB when x == INT32_MIN */
    return x;
}

/* Counter increment: wrapping on overflow is the Rust default. */
int32_t increment_counter(int32_t counter, int32_t delta) {
    return counter + delta;  /* UB on overflow in C */
}
