#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_assignments,
    unused_mut
)]
#[no_mangle]
pub unsafe extern "C" fn curl_remainder(
    mut bytes: ::core::ffi::c_int,
    mut chunk: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return bytes % chunk;
}
