/*
 * Rust translation of buffer access patterns.
 * Rust slices are bounds-checked — OOB panics instead of silent UB.
 */

/// Heartbleed-style: Rust panics if offset+i >= buf.len().
/// No silent read-past-buffer.
pub fn read_field(buf: &[u8], _buf_len: i32, offset: i32, len: i32) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..len {
        let idx = (offset + i) as usize;
        sum = sum.wrapping_add(buf[idx] as i32);  // panics if OOB
    }
    sum
}

/// Off-by-one: Rust panics on OOB write, catching the bug.
pub fn copy_with_null(dst: &mut [u8], src: &[u8], n: usize) {
    for i in 0..=n {  // same bug, but Rust panics instead of silent corruption
        dst[i] = src[i];
    }
    dst[n] = 0;
}

/// Lookup table: Rust bounds-checks the index.
pub fn lookup_table(index: usize) -> i32 {
    let table: [i32; 16] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
    table[index]  // panics if index >= 16
}

/// Signed index: Rust uses usize (unsigned) for indexing.
/// Negative index would require explicit cast, preventing silent OOB.
pub fn access_with_signed_index(arr: &[i32], idx: i32) -> i32 {
    arr[idx as usize]  // panics if idx < 0 after cast (wraps to huge usize)
}
