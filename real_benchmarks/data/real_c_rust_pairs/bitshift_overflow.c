/*
 * Bitshift overflow divergence — inspired by CVE-2015-0235 (GHOST glibc)
 * and various kernel bugs involving shift UB.
 *
 * C11 §6.5.7/3-4: shift by >= width is UB.
 * Rust: panics in debug, wraps (shift % width) in release.
 */
#include <stdint.h>

/* Shift by type width — UB in C, defined in Rust. */
uint32_t shift_by_32(uint32_t x) {
    return x << 32;  /* UB: shifting 32-bit value by 32 */
}

/* Shift by variable amount, potentially >= width. */
int32_t dynamic_shift(int32_t value, int32_t amount) {
    if (amount < 0)
        return 0;  /* negative shift is also UB in C */
    return value << amount;  /* UB if amount >= 32 */
}

/* Negative left-shift operand: UB for signed types in C. */
int32_t signed_left_shift(int32_t x, int shift) {
    return x << shift;  /* UB if x < 0 (C11 §6.5.7/4) */
}

/* Right shift of negative value: implementation-defined in C. */
int32_t arithmetic_right_shift(int32_t x, int shift) {
    return x >> shift;  /* impl-defined if x < 0: arithmetic or logical? */
}

/* Bit extraction pattern (common in crypto/protocol code). */
uint32_t extract_bits(uint32_t word, int start, int count) {
    return (word >> start) & ((1u << count) - 1);  /* UB if start >= 32 or count >= 32 */
}
