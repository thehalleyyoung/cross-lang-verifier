/*
 * Union type punning divergence — inspired by real networking/serialization code.
 *
 * In C, type-punning through unions is explicitly legal (C11 §6.5.2.3/3, footnote 95).
 * In Rust, unions require unsafe and reading wrong variant is UB per Rust spec.
 * This is a fundamental semantic divergence: legal C → UB Rust.
 */
#include <stdint.h>
#include <string.h>

/* Type punning: reinterpret float bits as integer (legal in C). */
union FloatBits {
    float f;
    uint32_t u;
};

uint32_t float_to_bits(float value) {
    union FloatBits fb;
    fb.f = value;
    return fb.u;  /* Legal in C11 — reads different member than last written */
}

float bits_to_float(uint32_t bits) {
    union FloatBits fb;
    fb.u = bits;
    return fb.f;  /* Legal type pun in C */
}

/* Network byte-order conversion using union (common real-world pattern). */
union NetWord {
    uint32_t word;
    uint8_t bytes[4];
};

uint32_t ntohl_manual(uint32_t net_order) {
    union NetWord nw;
    nw.word = net_order;
    /* Read individual bytes — legal union access in C */
    return ((uint32_t)nw.bytes[0] << 24) |
           ((uint32_t)nw.bytes[1] << 16) |
           ((uint32_t)nw.bytes[2] << 8)  |
           ((uint32_t)nw.bytes[3]);
}

/* Tagged union pattern (discriminated union). */
struct Variant {
    int tag;  /* 0 = int, 1 = float */
    union {
        int32_t i;
        float f;
    } data;
};

int32_t extract_int(struct Variant *v) {
    if (v->tag == 0)
        return v->data.i;
    /* Reading .i when .f was last written — still legal in C */
    return v->data.i;
}
