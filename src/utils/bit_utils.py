"""Bit manipulation utilities for cross-language equivalence verification.

Provides sign/zero extension, overflow detection, two's complement
conversion, bit pattern matching, and IEEE 754 float manipulation.
"""

from __future__ import annotations

import math
import struct
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def mask(width: int) -> int:
    """Return a bitmask of the given width: (1 << width) - 1."""
    if width <= 0:
        return 0
    return (1 << width) - 1


def extract_bits(value: int, lo: int, hi: int) -> int:
    """Extract bits [lo, hi) from value."""
    if hi <= lo:
        return 0
    return (value >> lo) & mask(hi - lo)


def set_bit(value: int, bit: int) -> int:
    return value | (1 << bit)


def clear_bit(value: int, bit: int) -> int:
    return value & ~(1 << bit)


def test_bit(value: int, bit: int) -> bool:
    return bool((value >> bit) & 1)


def popcount(value: int) -> int:
    """Count the number of set bits."""
    return bin(value & mask(64)).count("1") if value >= 0 else bin(value & mask(64)).count("1")


def leading_zeros(value: int, width: int) -> int:
    """Count leading zeros in a width-bit value."""
    if value == 0:
        return width
    count = 0
    for i in range(width - 1, -1, -1):
        if not test_bit(value, i):
            count += 1
        else:
            break
    return count


def trailing_zeros(value: int, width: int) -> int:
    """Count trailing zeros in a width-bit value."""
    if value == 0:
        return width
    count = 0
    for i in range(width):
        if not test_bit(value, i):
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Sign / zero extension
# ---------------------------------------------------------------------------

def sign_extend(value: int, from_width: int, to_width: int) -> int:
    """Sign-extend a value from from_width bits to to_width bits."""
    if from_width <= 0 or to_width <= 0:
        raise ValueError(f"Widths must be positive: from={from_width}, to={to_width}")
    if to_width < from_width:
        raise ValueError(f"Cannot sign-extend to smaller width: {from_width} -> {to_width}")
    if to_width == from_width:
        return value & mask(from_width)

    value = value & mask(from_width)
    sign_bit = test_bit(value, from_width - 1)
    if sign_bit:
        extension = mask(to_width) ^ mask(from_width)
        return value | extension
    return value


def zero_extend(value: int, from_width: int, to_width: int) -> int:
    """Zero-extend a value from from_width bits to to_width bits."""
    if from_width <= 0 or to_width <= 0:
        raise ValueError(f"Widths must be positive: from={from_width}, to={to_width}")
    if to_width < from_width:
        raise ValueError(f"Cannot zero-extend to smaller width: {from_width} -> {to_width}")
    return value & mask(from_width)


def truncate(value: int, to_width: int) -> int:
    """Truncate value to to_width bits."""
    return value & mask(to_width)


# ---------------------------------------------------------------------------
# Two's complement conversion
# ---------------------------------------------------------------------------

def twos_complement(value: int, width: int) -> int:
    """Convert a Python integer to its two's complement representation in `width` bits.

    Always returns an unsigned value in [0, 2^width).
    """
    return value & mask(width)


def twos_complement_inverse(value: int, width: int) -> int:
    """Convert a two's complement unsigned value to a signed Python integer."""
    value = value & mask(width)
    if test_bit(value, width - 1):
        return value - (1 << width)
    return value


def unsigned_to_signed(value: int, width: int) -> int:
    """Interpret an unsigned value as signed in the given width."""
    return twos_complement_inverse(value, width)


def signed_to_unsigned(value: int, width: int) -> int:
    """Convert a signed value to unsigned in the given width."""
    return twos_complement(value, width)


# ---------------------------------------------------------------------------
# Overflow detection
# ---------------------------------------------------------------------------

def _signed_range(width: int) -> Tuple[int, int]:
    """Return (min, max) for signed integers of given width."""
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def _unsigned_range(width: int) -> Tuple[int, int]:
    """Return (min, max) for unsigned integers of given width."""
    return 0, (1 << width) - 1


def detect_overflow_add(a: int, b: int, width: int, signed: bool = True) -> bool:
    """Detect if a + b overflows in the given width."""
    result = a + b
    if signed:
        lo, hi = _signed_range(width)
        sa = twos_complement_inverse(twos_complement(a, width), width)
        sb = twos_complement_inverse(twos_complement(b, width), width)
        sr = sa + sb
        return sr < lo or sr > hi
    else:
        return result < 0 or result >= (1 << width)


def detect_overflow_sub(a: int, b: int, width: int, signed: bool = True) -> bool:
    """Detect if a - b overflows in the given width."""
    if signed:
        lo, hi = _signed_range(width)
        sa = twos_complement_inverse(twos_complement(a, width), width)
        sb = twos_complement_inverse(twos_complement(b, width), width)
        sr = sa - sb
        return sr < lo or sr > hi
    else:
        ua = twos_complement(a, width)
        ub = twos_complement(b, width)
        return ua < ub


def detect_overflow_mul(a: int, b: int, width: int, signed: bool = True) -> bool:
    """Detect if a * b overflows in the given width."""
    if signed:
        lo, hi = _signed_range(width)
        sa = twos_complement_inverse(twos_complement(a, width), width)
        sb = twos_complement_inverse(twos_complement(b, width), width)
        sr = sa * sb
        return sr < lo or sr > hi
    else:
        ua = twos_complement(a, width)
        ub = twos_complement(b, width)
        return ua * ub >= (1 << width)


def detect_overflow_shl(a: int, shift: int, width: int, signed: bool = True) -> bool:
    """Detect if a << shift overflows in the given width."""
    if shift < 0 or shift >= width:
        return True
    return detect_overflow_mul(a, 1 << shift, width, signed)


def detect_overflow_neg(a: int, width: int) -> bool:
    """Detect if negation overflows (only signed MIN)."""
    lo, _ = _signed_range(width)
    sa = twos_complement_inverse(twos_complement(a, width), width)
    return sa == lo


def wrapping_add(a: int, b: int, width: int) -> int:
    """Wrapping addition in the given width."""
    return (a + b) & mask(width)


def wrapping_sub(a: int, b: int, width: int) -> int:
    """Wrapping subtraction in the given width."""
    return (a - b) & mask(width)


def wrapping_mul(a: int, b: int, width: int) -> int:
    """Wrapping multiplication in the given width."""
    return (a * b) & mask(width)


def saturating_add(a: int, b: int, width: int, signed: bool = True) -> int:
    """Saturating addition in the given width."""
    if signed:
        lo, hi = _signed_range(width)
        sa = twos_complement_inverse(twos_complement(a, width), width)
        sb = twos_complement_inverse(twos_complement(b, width), width)
        sr = sa + sb
        sr = max(lo, min(hi, sr))
        return twos_complement(sr, width)
    else:
        ua = twos_complement(a, width)
        ub = twos_complement(b, width)
        result = ua + ub
        _, umax = _unsigned_range(width)
        return min(result, umax)


def saturating_sub(a: int, b: int, width: int, signed: bool = True) -> int:
    """Saturating subtraction in the given width."""
    if signed:
        lo, hi = _signed_range(width)
        sa = twos_complement_inverse(twos_complement(a, width), width)
        sb = twos_complement_inverse(twos_complement(b, width), width)
        sr = sa - sb
        sr = max(lo, min(hi, sr))
        return twos_complement(sr, width)
    else:
        ua = twos_complement(a, width)
        ub = twos_complement(b, width)
        return max(0, ua - ub)


# ---------------------------------------------------------------------------
# IEEE 754 float bit manipulation
# ---------------------------------------------------------------------------

def float32_to_bits(value: float) -> int:
    """Convert a float32 value to its IEEE 754 bit representation."""
    data = struct.pack(">f", value)
    return struct.unpack(">I", data)[0]


def bits_to_float32(bits: int) -> float:
    """Convert IEEE 754 bits to a float32 value."""
    data = struct.pack(">I", bits & 0xFFFFFFFF)
    return struct.unpack(">f", data)[0]


def float64_to_bits(value: float) -> int:
    """Convert a float64 value to its IEEE 754 bit representation."""
    data = struct.pack(">d", value)
    return struct.unpack(">Q", data)[0]


def bits_to_float64(bits: int) -> float:
    """Convert IEEE 754 bits to a float64 value."""
    data = struct.pack(">Q", bits & 0xFFFFFFFFFFFFFFFF)
    return struct.unpack(">d", data)[0]


def float_sign(bits: int, double: bool = False) -> int:
    """Extract sign bit from IEEE 754 representation."""
    if double:
        return (bits >> 63) & 1
    return (bits >> 31) & 1


def float_exponent(bits: int, double: bool = False) -> int:
    """Extract biased exponent from IEEE 754 representation."""
    if double:
        return (bits >> 52) & 0x7FF
    return (bits >> 23) & 0xFF


def float_mantissa(bits: int, double: bool = False) -> int:
    """Extract mantissa from IEEE 754 representation."""
    if double:
        return bits & ((1 << 52) - 1)
    return bits & ((1 << 23) - 1)


def is_nan_bits(bits: int, double: bool = False) -> bool:
    """Check if IEEE 754 bits represent NaN."""
    exp = float_exponent(bits, double)
    mant = float_mantissa(bits, double)
    max_exp = 0x7FF if double else 0xFF
    return exp == max_exp and mant != 0


def is_inf_bits(bits: int, double: bool = False) -> bool:
    """Check if IEEE 754 bits represent infinity."""
    exp = float_exponent(bits, double)
    mant = float_mantissa(bits, double)
    max_exp = 0x7FF if double else 0xFF
    return exp == max_exp and mant == 0


def is_denormal_bits(bits: int, double: bool = False) -> bool:
    """Check if IEEE 754 bits represent a denormalized number."""
    exp = float_exponent(bits, double)
    mant = float_mantissa(bits, double)
    return exp == 0 and mant != 0


def is_zero_bits(bits: int, double: bool = False) -> bool:
    """Check if IEEE 754 bits represent zero (+0 or -0)."""
    exp = float_exponent(bits, double)
    mant = float_mantissa(bits, double)
    return exp == 0 and mant == 0


def float_classify(bits: int, double: bool = False) -> str:
    """Classify IEEE 754 bits: 'nan', 'inf', 'denormal', 'zero', 'normal'."""
    if is_nan_bits(bits, double):
        return "nan"
    if is_inf_bits(bits, double):
        return "inf"
    if is_denormal_bits(bits, double):
        return "denormal"
    if is_zero_bits(bits, double):
        return "zero"
    return "normal"


def next_float32(value: float) -> float:
    """Return next representable float32 after value."""
    bits = float32_to_bits(value)
    if is_nan_bits(bits, False):
        return value
    if value >= 0:
        bits += 1
    else:
        bits -= 1
    return bits_to_float32(bits)


def prev_float32(value: float) -> float:
    """Return previous representable float32 before value."""
    bits = float32_to_bits(value)
    if is_nan_bits(bits, False):
        return value
    if value > 0:
        bits -= 1
    else:
        bits += 1
    return bits_to_float32(bits)


def ulp_distance(a: float, b: float) -> int:
    """Compute ULP (unit in the last place) distance between two floats."""
    if math.isnan(a) or math.isnan(b):
        return -1
    bits_a = float64_to_bits(a)
    bits_b = float64_to_bits(b)
    sa = twos_complement_inverse(bits_a, 64) if test_bit(bits_a, 63) else bits_a
    sb = twos_complement_inverse(bits_b, 64) if test_bit(bits_b, 63) else bits_b
    return abs(sa - sb)


# ---------------------------------------------------------------------------
# Bit pattern matching
# ---------------------------------------------------------------------------

def matches_pattern(value: int, pattern: str, width: int) -> bool:
    """Match value against a bit pattern string ('0', '1', 'x' for don't care).

    Pattern is MSB-first. Must have exactly `width` characters.
    """
    if len(pattern) != width:
        raise ValueError(f"Pattern length {len(pattern)} != width {width}")
    for i, ch in enumerate(pattern):
        bit_pos = width - 1 - i
        bit_val = test_bit(value, bit_pos)
        if ch == "1" and not bit_val:
            return False
        if ch == "0" and bit_val:
            return False
    return True


def rotate_left(value: int, amount: int, width: int) -> int:
    """Rotate value left by amount bits within width."""
    amount = amount % width
    value = value & mask(width)
    return ((value << amount) | (value >> (width - amount))) & mask(width)


def rotate_right(value: int, amount: int, width: int) -> int:
    """Rotate value right by amount bits within width."""
    amount = amount % width
    value = value & mask(width)
    return ((value >> amount) | (value << (width - amount))) & mask(width)


def byte_swap(value: int, width: int) -> int:
    """Byte-swap a value of the given bit width (must be multiple of 8)."""
    if width % 8 != 0:
        raise ValueError(f"Width must be multiple of 8 for byte swap, got {width}")
    num_bytes = width // 8
    result = 0
    for i in range(num_bytes):
        byte = (value >> (i * 8)) & 0xFF
        result |= byte << ((num_bytes - 1 - i) * 8)
    return result
