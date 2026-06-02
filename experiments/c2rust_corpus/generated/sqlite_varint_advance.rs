#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_assignments,
    unused_mut
)]
#[no_mangle]
pub unsafe extern "C" fn sqlite_varint_advance(mut x: ::core::ffi::c_int) -> ::core::ffi::c_int {
    return x + 7 as ::core::ffi::c_int;
}
