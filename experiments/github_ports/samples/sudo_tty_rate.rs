#[no_mangle]
pub extern "C" fn sudo_tty_rate(bytes: i32, elapsed: i32) -> i32 {
    bytes / elapsed
}
