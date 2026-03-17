/*
 * Rust translation of integer overflow patterns.
 * Key divergence: signed overflow panics in debug, wraps in release.
 * Rust provides i32::checked_mul, wrapping_add, saturating_add etc.
 */

/// CVE-style allocation size computation.
/// In Rust, i32 multiplication wraps (release) or panics (debug).
/// Either way, behavior is *defined* — no silent UB.
pub fn alloc_items(n: i32, item_size: i32) -> Option<Vec<u8>> {
    let total = n.wrapping_mul(item_size); // wrapping: defined behavior
    if total < 0 {
        return None;
    }
    Some(vec![0u8; total as usize])
}

/// Midpoint computation — Rust version.
/// Using wrapping arithmetic: (low + high) wraps instead of UB.
pub fn midpoint(low: i32, high: i32) -> i32 {
    (low.wrapping_add(high)) / 2
}

/// Absolute value: i32::MIN.wrapping_neg() == i32::MIN (wraps back).
/// In Rust, this is defined wrapping behavior, not UB.
pub fn safe_abs(x: i32) -> i32 {
    if x < 0 { x.wrapping_neg() } else { x }
}

/// Counter increment with wrapping semantics (defined in Rust).
pub fn increment_counter(counter: i32, delta: i32) -> i32 {
    counter.wrapping_add(delta)
}
