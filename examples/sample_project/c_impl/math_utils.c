/* Matching C implementations of math utility functions. */

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int clamp(int value, int lo, int hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}
