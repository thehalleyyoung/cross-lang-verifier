/*
 * Floating-point to integer overflow — inspired by CVE-2020-36385 and
 * real-world numeric conversion bugs in audio/video codecs.
 *
 * C11 §6.3.1.4/1: when a finite value of real floating type is converted
 * to an integer type other than _Bool, if the value cannot be represented,
 * the behavior is undefined.
 *
 * Rust: f32::to_int_unchecked is unsafe; normal `as` saturates since Rust 1.45.
 */
#include <stdint.h>
#include <math.h>

/* Float-to-int overflow: UB in C if value doesn't fit. */
int32_t float_to_int(float x) {
    return (int32_t)x;  /* UB if x > INT32_MAX or x < INT32_MIN or x is NaN */
}

/* Double precision: same issue with larger range. */
int32_t double_to_int(double x) {
    return (int32_t)x;  /* UB if out of range */
}

/* Unsigned conversion: large negative float → unsigned is UB. */
uint32_t float_to_uint(float x) {
    return (uint32_t)x;  /* UB if x < 0 or x > UINT32_MAX */
}

/* Audio sample conversion: float [-1.0, 1.0] → int16.
   Real bug: denormals/infinity produce UB in the cast. */
int16_t audio_float_to_int16(float sample) {
    float scaled = sample * 32767.0f;
    if (scaled > 32767.0f) scaled = 32767.0f;
    if (scaled < -32768.0f) scaled = -32768.0f;
    /* Even with clamping, NaN passes through both checks (NaN comparisons are false). */
    return (int16_t)scaled;  /* UB if scaled is NaN */
}

/* Truncation vs rounding difference. */
int32_t trunc_to_int(double x) {
    return (int32_t)x;  /* C truncates toward zero, but UB on overflow */
}
