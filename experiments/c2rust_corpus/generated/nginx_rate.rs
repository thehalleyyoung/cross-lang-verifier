#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_assignments,
    unused_mut
)]
#[no_mangle]
pub unsafe extern "C" fn nginx_rate(
    mut bytes: ::core::ffi::c_int,
    mut seconds: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return bytes / seconds;
}
