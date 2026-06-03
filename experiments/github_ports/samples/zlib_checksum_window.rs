#[no_mangle]
pub extern "C" fn zlib_checksum_window(sum: i32) -> i32 {
    sum.wrapping_add(65521)
}
