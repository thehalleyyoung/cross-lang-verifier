/*
 * Buffer/array out-of-bounds access — inspired by CVE-2021-3156 (sudo heap overflow)
 * and CVE-2014-0160 (Heartbleed: read past buffer).
 *
 * In C, out-of-bounds access is UB. No runtime check by default.
 * In Rust, array indexing panics on OOB (bounds-checked).
 */
#include <stdint.h>
#include <string.h>

/* Heartbleed-style: read 'len' bytes from buf of size 'buf_len'.
   If len > buf_len, reads past the buffer — UB and info leak. */
int read_field(const uint8_t *buf, int buf_len, int offset, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += buf[offset + i];  /* UB when offset+i >= buf_len */
    }
    return sum;
}

/* Off-by-one: classic fence-post error. */
void copy_with_null(char *dst, const char *src, int n) {
    int i;
    for (i = 0; i <= n; i++) {  /* bug: should be i < n */
        dst[i] = src[i];  /* writes one past dst[n-1] */
    }
    dst[n] = '\0';
}

/* Stack buffer overflow via unchecked index. */
int lookup_table(int index) {
    int table[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    return table[index];  /* UB if index < 0 or index >= 16 */
}

/* Negative index (signed int used as index). */
int access_with_signed_index(int *arr, int idx) {
    return arr[idx];  /* UB if idx < 0 and arr not offset appropriately */
}
