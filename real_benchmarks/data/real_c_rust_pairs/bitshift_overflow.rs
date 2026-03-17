/*
 * Rust translation of bitshift patterns.
 * Rust: shift >= width panics in debug, wraps in release.
 * Negative shift is impossible (shift amount is unsigned or checked).
 */

/// Shift by 32: Rust wraps — 32 % 32 = 0, so result is x.
pub fn shift_by_32(x: u32) -> u32 {
    x.wrapping_shl(32)  // 32 % 32 = 0 → returns x
}

/// Dynamic shift: Rust wrapping_shl takes u32, amount is mod 32.
pub fn dynamic_shift(value: i32, amount: i32) -> i32 {
    if amount < 0 {
        return 0;
    }
    value.wrapping_shl(amount as u32)  // amount mod 32
}

/// Signed left shift: Rust wrapping_shl is defined for all values.
pub fn signed_left_shift(x: i32, shift: u32) -> i32 {
    x.wrapping_shl(shift)  // no UB, wraps
}

/// Right shift: Rust arithmetic right shift for signed types (always).
pub fn arithmetic_right_shift(x: i32, shift: u32) -> i32 {
    x.wrapping_shr(shift)  // always arithmetic for i32
}

/// Bit extraction: Rust version with wrapping shifts.
pub fn extract_bits(word: u32, start: u32, count: u32) -> u32 {
    (word.wrapping_shr(start)) & (1u32.wrapping_shl(count).wrapping_sub(1))
}
