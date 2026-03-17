/*
 * Strict aliasing violation — inspired by real-world type punning bugs.
 * CVE-2008-1447 (DNS cache poisoning exploited via pointer aliasing in bind).
 *
 * C11 §6.5/7 (strict aliasing rule): an object shall have its stored value
 * accessed only by an lvalue of a compatible type. Accessing int through
 * float pointer (or vice versa) is UB.
 *
 * Rust: raw pointer casts require unsafe, and Miri catches violations.
 */
#include <stdint.h>
#include <string.h>

/* Classic strict aliasing violation: int → float through pointer cast. */
float int_to_float_aliased(int32_t x) {
    float *fp = (float *)&x;  /* violates strict aliasing */
    return *fp;  /* UB: reading int object through float lvalue */
}

/* Reverse: float → int through pointer cast. */
int32_t float_to_int_aliased(float x) {
    int32_t *ip = (int32_t *)&x;  /* violates strict aliasing */
    return *ip;  /* UB: same issue */
}

/* Safe version using memcpy (not UB — type-punning via memcpy is legal). */
float int_to_float_safe(int32_t x) {
    float result;
    memcpy(&result, &x, sizeof(result));  /* legal: memcpy is special */
    return result;
}

/* TBAA (Type-Based Alias Analysis) exploit: compiler assumes no aliasing. */
void swap_values(int *restrict a, float *restrict b) {
    /* Compiler assumes a and b don't alias, but if called with overlapping
       memory, the optimization may produce wrong results. */
    int tmp_a = *a;
    float tmp_b = *b;
    *(float *)a = tmp_b;  /* UB: writes float to int object */
    *(int *)b = tmp_a;    /* UB: writes int to float object */
}

/* char* is allowed to alias anything in C (the char exception). */
int read_through_char(int *p) {
    char *cp = (char *)p;  /* legal: char can alias any type */
    int sum = 0;
    for (int i = 0; i < (int)sizeof(int); i++) {
        sum += cp[i];
    }
    return sum;
}
