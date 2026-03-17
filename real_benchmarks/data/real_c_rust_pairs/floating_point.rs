/*
 * Rust translation of floating-point to integer conversion patterns.
 * Since Rust 1.45, `as` casts saturate (no UB):
 *   - f32::INFINITY as i32 → i32::MAX
 *   - f32::NEG_INFINITY as i32 → i32::MIN
 *   - f32::NAN as i32 → 0
 */

/// Float to int: Rust saturates instead of UB.
/// f32::MAX as i32 → i32::MAX, NaN → 0.
pub fn float_to_int(x: f32) -> i32 {
    x as i32  // saturating cast since Rust 1.45
}

/// Double to int: same saturating semantics.
pub fn double_to_int(x: f64) -> i32 {
    x as i32  // saturates at i32::MIN/MAX, NaN → 0
}

/// Float to unsigned: negative values saturate to 0 in Rust.
pub fn float_to_uint(x: f32) -> u32 {
    x as u32  // negative → 0, too large → u32::MAX, NaN → 0
}

/// Audio sample conversion: Rust saturating cast handles NaN/infinity.
pub fn audio_float_to_int16(sample: f32) -> i16 {
    let scaled = sample * 32767.0f32;
    let clamped = scaled.clamp(-32768.0, 32767.0);
    // NaN.clamp() returns NaN in Rust, but NaN as i16 → 0
    clamped as i16
}

/// Truncation: Rust `as` truncates toward zero like C, but saturates on overflow.
pub fn trunc_to_int(x: f64) -> i32 {
    x as i32  // saturating, not UB
}
