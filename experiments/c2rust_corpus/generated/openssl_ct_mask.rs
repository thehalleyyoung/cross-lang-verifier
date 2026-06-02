#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_assignments,
    unused_mut
)]
#[no_mangle]
pub unsafe extern "C" fn openssl_ct_mask(
    mut v: ::core::ffi::c_int,
    mut bits: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return v << bits;
}
