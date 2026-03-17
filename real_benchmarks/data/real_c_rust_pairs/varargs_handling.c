/*
 * Variadic function divergence — inspired by CVE-2013-2028 (nginx format string)
 * and countless format string vulnerabilities.
 *
 * C variadic functions (stdarg.h) have no type safety at the call site.
 * Wrong format specifiers → UB (reads wrong type from va_list).
 * Rust has no variadic functions in safe code; macros handle formatting.
 */
#include <stdarg.h>
#include <stdint.h>

/* Sum n integers via va_list. */
int sum_variadic(int count, ...) {
    va_list args;
    va_start(args, count);
    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);  /* UB if caller passes fewer than count args */
    }
    va_end(args);
    return total;
}

/* Type mismatch in va_arg: caller passes float, callee reads int.
   This is a real class of printf format string bugs. */
int read_as_int_from_varargs(int dummy, ...) {
    va_list args;
    va_start(args, dummy);
    int val = va_arg(args, int);  /* UB if caller passed a double/float */
    va_end(args);
    return val;
}

/* Max of n int32_t values. */
int32_t max_variadic(int count, ...) {
    va_list args;
    va_start(args, count);
    int32_t max_val = INT32_MIN;
    for (int i = 0; i < count; i++) {
        int32_t v = va_arg(args, int32_t);
        if (v > max_val) max_val = v;
    }
    va_end(args);
    return max_val;
}
