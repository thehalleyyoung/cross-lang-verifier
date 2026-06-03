#[no_mangle]
pub extern "C" fn coreutils_size_accumulate(bytes: i32) -> i32 {
    bytes.wrapping_add(4096)
}
