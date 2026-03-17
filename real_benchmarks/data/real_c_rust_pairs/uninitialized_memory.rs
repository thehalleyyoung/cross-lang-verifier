/*
 * Rust translation of uninitialized memory patterns.
 * Rust's ownership system requires all variables be initialized before use.
 * The compiler refuses to compile if any path leaves a variable uninitialized.
 */

/// Parse header: Rust requires explicit initialization on all paths.
pub fn parse_header(buf: &[u8], len: i32) -> i32 {
    let version: i32;
    let flags: i32;
    if len >= 2 {
        version = buf[0] as i32;
        flags = buf[1] as i32;
    } else {
        version = 0;  // Rust forces initialization on else path
        flags = 0;
    }
    version + flags
}

/// Partial struct init: Rust requires all fields be initialized.
pub struct Config {
    pub mode: i32,
    pub timeout: i32,
    pub retries: i32,
    pub verbose: i32,
}

pub fn get_total_config(has_custom: bool) -> i32 {
    let cfg = if has_custom {
        Config { mode: 1, timeout: 30, retries: 3, verbose: 0 }
    } else {
        Config { mode: 1, timeout: 0, retries: 0, verbose: 0 }  // must init all
    };
    cfg.mode + cfg.timeout + cfg.retries + cfg.verbose
}

/// Stack array: Rust arrays must be initialized.
pub fn sum_uninitialized_array(n: i32) -> i32 {
    let arr = [0i32; 64];  // must initialize
    let mut sum = 0i32;
    let limit = n.min(64) as usize;
    for i in 0..limit {
        sum = sum.wrapping_add(arr[i]);
    }
    sum
}

/// Conditional init: Rust compiler rejects if any path leaves uninit.
pub fn conditional_init(selector: i32, a: i32, b: i32) -> i32 {
    let result: i32;
    if selector > 0 {
        result = a;
    } else if selector < 0 {
        result = b;
    } else {
        result = 0;  // Rust forces a default
    }
    result
}
