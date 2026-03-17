/*
 * Rust translation of aliasing violation patterns.
 * Rust's type system and borrow checker prevent aliasing violations.
 * Raw pointer casts require unsafe and are checked by Miri.
 */

/// Int to float reinterpretation: Rust uses f32::from_bits (safe).
pub fn int_to_float_aliased(x: i32) -> f32 {
    f32::from_bits(x as u32)  // safe, defined behavior, no aliasing issue
}

/// Float to int reinterpretation: Rust uses f32::to_bits (safe).
pub fn float_to_int_aliased(x: f32) -> i32 {
    x.to_bits() as i32  // safe, no pointer aliasing
}

/// Safe version: identical to aliased version in Rust — both are safe.
pub fn int_to_float_safe(x: i32) -> f32 {
    f32::from_bits(x as u32)
}

/// Swap: Rust's borrow checker prevents mutable aliasing at compile time.
/// &mut a and &mut b are guaranteed not to overlap.
pub fn swap_values(a: &mut i32, b: &mut f32) {
    let tmp_a = *a;
    let tmp_b = *b;
    *b = tmp_a as f32;
    *a = tmp_b as i32;
}

/// Byte-level access: Rust uses to_ne_bytes() for safe byte access.
pub fn read_through_char(p: &i32) -> i32 {
    let bytes = p.to_ne_bytes();
    let mut sum: i32 = 0;
    for &b in bytes.iter() {
        sum = sum.wrapping_add(b as i32);
    }
    sum
}
