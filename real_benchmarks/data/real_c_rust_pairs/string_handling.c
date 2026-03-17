/*
 * C string handling divergence — inspired by CVE-2019-14287 (sudo string parsing)
 * and countless buffer overflows from strcpy/strcat without bounds.
 *
 * C strings are null-terminated char arrays with manual length management.
 * Rust String/&str are length-prefixed, UTF-8 validated, bounds-checked.
 */
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* Unsafe string copy without bounds check (strcpy pattern). */
void copy_string(char *dst, const char *src) {
    while (*src) {
        *dst++ = *src++;  /* no bounds check — overwrites past dst if src too long */
    }
    *dst = '\0';
}

/* String concatenation with fixed buffer — classic overflow pattern. */
int concat_in_buffer(char *buf, int buf_size, const char *suffix) {
    int cur_len = (int)strlen(buf);
    int suf_len = (int)strlen(suffix);
    /* Integer overflow in length check (CVE-style). */
    if (cur_len + suf_len + 1 > buf_size)  /* overflow if cur_len + suf_len wraps */
        return -1;
    strcat(buf, suffix);
    return 0;
}

/* Read a length-prefixed field from a binary buffer. */
int32_t read_length_prefixed(const uint8_t *data, int data_len) {
    if (data_len < 4) return -1;
    int32_t len;
    memcpy(&len, data, 4);  /* endianness-dependent — UB territory */
    if (len < 0 || 4 + len > data_len) return -1;
    int32_t sum = 0;
    for (int i = 0; i < len; i++) {
        sum += data[4 + i];
    }
    return sum;
}

/* Embedded null bytes: C sees shorter string, Rust preserves full content. */
int string_length_with_null_bytes(const char *s) {
    return (int)strlen(s);  /* stops at first \0 */
}
