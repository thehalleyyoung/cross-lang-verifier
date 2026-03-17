/*
 * Uninitialized memory divergence — inspired by CVE-2019-15166 (tcpdump)
 * and CVE-2017-12172 (PostgreSQL uninit read).
 *
 * C11 §6.3.2.1/2: reading an uninitialized variable of automatic storage
 * duration that could have been register-qualified is UB.
 * Rust: all variables must be initialized before use (compiler-enforced).
 */
#include <stdint.h>
#include <string.h>

/* Classic: uninitialized variable on error path. */
int parse_header(const uint8_t *buf, int len) {
    int version;
    int flags;
    if (len >= 2) {
        version = buf[0];
        flags = buf[1];
    }
    /* If len < 2, version and flags are uninitialized — UB to read them */
    return version + flags;
}

/* Partial struct initialization (common real-world bug). */
struct Config {
    int mode;
    int timeout;
    int retries;
    int verbose;
};

int get_total_config(int has_custom) {
    struct Config cfg;
    cfg.mode = 1;
    if (has_custom) {
        cfg.timeout = 30;
        cfg.retries = 3;
        cfg.verbose = 0;
    }
    /* If !has_custom, timeout/retries/verbose are uninit — UB */
    return cfg.mode + cfg.timeout + cfg.retries + cfg.verbose;
}

/* Stack buffer: not zeroed by default in C. */
int sum_uninitialized_array(int n) {
    int arr[64];  /* uninitialized stack array */
    /* Forgot to initialize — reading arr[i] is UB */
    int sum = 0;
    for (int i = 0; i < n && i < 64; i++) {
        sum += arr[i];
    }
    return sum;
}

/* Conditional initialization: optimizer may assume init. */
int conditional_init(int selector, int a, int b) {
    int result;
    if (selector > 0)
        result = a;
    else if (selector < 0)
        result = b;
    /* if selector == 0, result is uninitialized — UB */
    return result;
}
