/*
 * Rust translation of variadic function patterns.
 * Rust has no variadic functions in safe code.
 * Slices replace va_list — fully type-safe.
 */

/// Sum: Rust uses a slice — type-safe, bounds-checked.
pub fn sum_variadic(values: &[i32]) -> i32 {
    let mut total: i32 = 0;
    for &v in values {
        total = total.wrapping_add(v);
    }
    total
}

/// No type mismatch possible: Rust enforces types at compile time.
/// There is no equivalent of "read int from float vararg" in safe Rust.
pub fn read_as_int_from_varargs(val: i32) -> i32 {
    val  // type is known at compile time, no UB possible
}

/// Max of slice: type-safe, no varargs.
pub fn max_variadic(values: &[i32]) -> i32 {
    let mut max_val = i32::MIN;
    for &v in values {
        if v > max_val {
            max_val = v;
        }
    }
    max_val
}
