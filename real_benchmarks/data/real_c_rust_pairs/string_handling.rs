/*
 * Rust translation of C string patterns.
 * Rust's String/&str: length-prefixed, UTF-8 validated, bounds-checked.
 * No null terminator, no buffer overflow from string ops.
 */

/// String copy: Rust clone is safe, bounds-checked.
pub fn copy_string(src: &str) -> String {
    src.to_string()  // no buffer overflow possible
}

/// String concat: Rust's String::push_str handles allocation.
/// Returns Err if would exceed buf_size (explicit check, no UB).
pub fn concat_in_buffer(buf: &mut String, buf_size: usize, suffix: &str) -> Result<(), i32> {
    let total = buf.len().checked_add(suffix.len())
        .and_then(|v| v.checked_add(1));
    match total {
        Some(t) if t > buf_size => Err(-1),
        None => Err(-1),  // overflow detected, not UB
        _ => {
            buf.push_str(suffix);
            Ok(())
        }
    }
}

/// Read length-prefixed field — Rust uses slice bounds checking.
pub fn read_length_prefixed(data: &[u8]) -> i32 {
    if data.len() < 4 { return -1; }
    let len = i32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
    if len < 0 || (4 + len as usize) > data.len() { return -1; }
    let mut sum: i32 = 0;
    for i in 0..len as usize {
        sum = sum.wrapping_add(data[4 + i] as i32);
    }
    sum
}

/// Rust strings can contain arbitrary bytes via Vec<u8>, or use &str (UTF-8).
/// String::len() returns byte count, not stopping at embedded zeros.
pub fn string_length_with_null_bytes(s: &[u8]) -> i32 {
    s.len() as i32  // counts all bytes including \0
}
