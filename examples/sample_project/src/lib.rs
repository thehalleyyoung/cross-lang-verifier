//! Small Rust FFI library with math utility functions.

#[no_mangle]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

#[no_mangle]
pub extern "C" fn multiply(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

#[no_mangle]
pub extern "C" fn clamp(value: i32, lo: i32, hi: i32) -> i32 {
    if value < lo {
        lo
    } else if value > hi {
        hi
    } else {
        value
    }
}
