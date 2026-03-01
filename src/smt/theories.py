"""
Theory-specific encoding helpers.

Provides helper functions for bitvector theory (sign extension, overflow
detection, rotation), floating-point theory (NaN handling, rounding modes,
special values), and array theory (symbolic array operations).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import z3


# ---------------------------------------------------------------------------
# Bitvector theory helpers
# ---------------------------------------------------------------------------

class BitvectorTheory:
    """Helpers for QF_BV theory encoding."""

    @staticmethod
    def sign_extend(expr: z3.BitVecRef, target_width: int) -> z3.BitVecRef:
        """Sign-extend a bitvector to target width."""
        current = expr.size()
        if current >= target_width:
            return expr
        return z3.SignExt(target_width - current, expr)

    @staticmethod
    def zero_extend(expr: z3.BitVecRef, target_width: int) -> z3.BitVecRef:
        """Zero-extend a bitvector to target width."""
        current = expr.size()
        if current >= target_width:
            return expr
        return z3.ZeroExt(target_width - current, expr)

    @staticmethod
    def truncate(expr: z3.BitVecRef, target_width: int) -> z3.BitVecRef:
        """Truncate a bitvector to target width."""
        if expr.size() <= target_width:
            return expr
        return z3.Extract(target_width - 1, 0, expr)

    @staticmethod
    def resize(expr: z3.BitVecRef, target_width: int, signed: bool = True) -> z3.BitVecRef:
        """Resize a bitvector to target width (extend or truncate)."""
        current = expr.size()
        if current == target_width:
            return expr
        if current < target_width:
            if signed:
                return z3.SignExt(target_width - current, expr)
            return z3.ZeroExt(target_width - current, expr)
        return z3.Extract(target_width - 1, 0, expr)

    # -- Overflow detection --

    @staticmethod
    def signed_add_overflow(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
        """Detect signed addition overflow."""
        w = a.size()
        zero = z3.BitVecVal(0, w)
        result = a + b
        pos_overflow = z3.And(a > zero, b > zero, result < zero)
        neg_overflow = z3.And(a < zero, b < zero, result > zero)
        return z3.Or(pos_overflow, neg_overflow)

    @staticmethod
    def signed_sub_overflow(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
        """Detect signed subtraction overflow."""
        w = a.size()
        zero = z3.BitVecVal(0, w)
        result = a - b
        pos_overflow = z3.And(a > zero, b < zero, result < zero)
        neg_overflow = z3.And(a < zero, b > zero, result > zero)
        return z3.Or(pos_overflow, neg_overflow)

    @staticmethod
    def signed_mul_overflow(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
        """Detect signed multiplication overflow."""
        w = a.size()
        ext_a = z3.SignExt(w, a)
        ext_b = z3.SignExt(w, b)
        full = ext_a * ext_b
        truncated = z3.Extract(w - 1, 0, full)
        sign_ext_back = z3.SignExt(w, truncated)
        return full != sign_ext_back

    @staticmethod
    def unsigned_add_overflow(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
        """Detect unsigned addition overflow (carry)."""
        result = a + b
        return z3.ULT(result, a)

    @staticmethod
    def unsigned_sub_overflow(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
        """Detect unsigned subtraction underflow (borrow)."""
        return z3.ULT(a, b)

    @staticmethod
    def unsigned_mul_overflow(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
        """Detect unsigned multiplication overflow."""
        w = a.size()
        ext_a = z3.ZeroExt(w, a)
        ext_b = z3.ZeroExt(w, b)
        full = ext_a * ext_b
        high = z3.Extract(2 * w - 1, w, full)
        return high != z3.BitVecVal(0, w)

    @staticmethod
    def division_by_zero(divisor: z3.BitVecRef) -> z3.BoolRef:
        """Check if divisor is zero."""
        return divisor == z3.BitVecVal(0, divisor.size())

    @staticmethod
    def signed_div_overflow(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
        """Detect INT_MIN / -1 overflow."""
        w = a.size()
        min_val = z3.BitVecVal(1 << (w - 1), w)
        neg_one = z3.BitVecVal((1 << w) - 1, w)
        return z3.And(a == min_val, b == neg_one)

    # -- Shift operations --

    @staticmethod
    def overshift(shift_amount: z3.BitVecRef) -> z3.BoolRef:
        """Check if shift amount >= bit width."""
        w = shift_amount.size()
        return z3.UGE(shift_amount, z3.BitVecVal(w, w))

    @staticmethod
    def masked_shift_left(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        """Shift left with shift amount masked to valid range (Rust behavior)."""
        w = a.size()
        mask = z3.BitVecVal(w - 1, w)
        masked_b = b & mask
        return a << masked_b

    @staticmethod
    def masked_shift_right_logical(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        """Logical shift right with masked shift amount."""
        w = a.size()
        mask = z3.BitVecVal(w - 1, w)
        return z3.LShR(a, b & mask)

    @staticmethod
    def masked_shift_right_arithmetic(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        """Arithmetic shift right with masked shift amount."""
        w = a.size()
        mask = z3.BitVecVal(w - 1, w)
        return a >> (b & mask)

    # -- Rotation --

    @staticmethod
    def rotate_left(a: z3.BitVecRef, amount: z3.BitVecRef) -> z3.BitVecRef:
        """Rotate left."""
        return z3.RotateLeft(a, amount)

    @staticmethod
    def rotate_right(a: z3.BitVecRef, amount: z3.BitVecRef) -> z3.BitVecRef:
        """Rotate right."""
        return z3.RotateRight(a, amount)

    # -- Bit manipulation --

    @staticmethod
    def count_leading_zeros(x: z3.BitVecRef) -> z3.BitVecRef:
        """Count leading zeros (symbolic)."""
        w = x.size()
        result = z3.BitVecVal(w, w)
        for i in range(w - 1, -1, -1):
            bit_set = z3.Extract(i, i, x) == z3.BitVecVal(1, 1)
            result = z3.If(bit_set, z3.BitVecVal(w - 1 - i, w), result)
        return result

    @staticmethod
    def count_trailing_zeros(x: z3.BitVecRef) -> z3.BitVecRef:
        """Count trailing zeros (symbolic)."""
        w = x.size()
        result = z3.BitVecVal(w, w)
        for i in range(w):
            bit_set = z3.Extract(i, i, x) == z3.BitVecVal(1, 1)
            result = z3.If(bit_set, z3.BitVecVal(i, w), result)
        return result

    @staticmethod
    def popcount(x: z3.BitVecRef) -> z3.BitVecRef:
        """Population count (number of set bits)."""
        w = x.size()
        result = z3.BitVecVal(0, w)
        for i in range(w):
            bit = z3.ZeroExt(w - 1, z3.Extract(i, i, x))
            result = result + bit
        return result

    @staticmethod
    def byte_swap(x: z3.BitVecRef) -> z3.BitVecRef:
        """Byte-swap (endianness conversion)."""
        w = x.size()
        if w < 16:
            return x
        num_bytes = w // 8
        bytes_list = []
        for i in range(num_bytes):
            byte_val = z3.Extract((i + 1) * 8 - 1, i * 8, x)
            bytes_list.append(byte_val)
        bytes_list.reverse()
        result = bytes_list[0]
        for b in bytes_list[1:]:
            result = z3.Concat(result, b)
        return result

    # -- Saturating arithmetic --

    @staticmethod
    def saturating_add_signed(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        """Signed saturating addition."""
        w = a.size()
        max_val = z3.BitVecVal((1 << (w - 1)) - 1, w)
        min_val = z3.BitVecVal(-(1 << (w - 1)), w)
        result = a + b
        overflow = BitvectorTheory.signed_add_overflow(a, b)
        zero = z3.BitVecVal(0, w)
        return z3.If(
            overflow,
            z3.If(a > zero, max_val, min_val),
            result,
        )

    @staticmethod
    def saturating_sub_signed(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        """Signed saturating subtraction."""
        w = a.size()
        max_val = z3.BitVecVal((1 << (w - 1)) - 1, w)
        min_val = z3.BitVecVal(-(1 << (w - 1)), w)
        result = a - b
        overflow = BitvectorTheory.signed_sub_overflow(a, b)
        zero = z3.BitVecVal(0, w)
        return z3.If(
            overflow,
            z3.If(a > zero, max_val, min_val),
            result,
        )

    @staticmethod
    def wrapping_add(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        """Wrapping addition (standard bitvector add)."""
        return a + b

    @staticmethod
    def wrapping_sub(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        """Wrapping subtraction."""
        return a - b

    @staticmethod
    def wrapping_mul(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
        """Wrapping multiplication."""
        return a * b

    # -- Constants --

    @staticmethod
    def signed_max(width: int) -> z3.BitVecVal:
        return z3.BitVecVal((1 << (width - 1)) - 1, width)

    @staticmethod
    def signed_min(width: int) -> z3.BitVecVal:
        return z3.BitVecVal(1 << (width - 1), width)

    @staticmethod
    def unsigned_max(width: int) -> z3.BitVecVal:
        return z3.BitVecVal((1 << width) - 1, width)

    @staticmethod
    def zero(width: int) -> z3.BitVecVal:
        return z3.BitVecVal(0, width)

    @staticmethod
    def one(width: int) -> z3.BitVecVal:
        return z3.BitVecVal(1, width)


# ---------------------------------------------------------------------------
# Floating-point theory helpers
# ---------------------------------------------------------------------------

class FloatingPointTheory:
    """Helpers for QF_FP theory encoding."""

    # -- Sort creation --

    @staticmethod
    def float32_sort() -> z3.FPSortRef:
        return z3.FPSort(8, 24)

    @staticmethod
    def float64_sort() -> z3.FPSortRef:
        return z3.FPSort(11, 53)

    @staticmethod
    def float16_sort() -> z3.FPSortRef:
        return z3.FPSort(5, 11)

    @staticmethod
    def float128_sort() -> z3.FPSortRef:
        return z3.FPSort(15, 113)

    # -- Rounding modes --

    @staticmethod
    def round_nearest_even() -> z3.FPRMRef:
        return z3.RNE()

    @staticmethod
    def round_nearest_away() -> z3.FPRMRef:
        return z3.RNA()

    @staticmethod
    def round_toward_positive() -> z3.FPRMRef:
        return z3.RTP()

    @staticmethod
    def round_toward_negative() -> z3.FPRMRef:
        return z3.RTN()

    @staticmethod
    def round_toward_zero() -> z3.FPRMRef:
        return z3.RTZ()

    # -- Special value detection --

    @staticmethod
    def is_nan(x: z3.FPRef) -> z3.BoolRef:
        return z3.fpIsNaN(x)

    @staticmethod
    def is_inf(x: z3.FPRef) -> z3.BoolRef:
        return z3.fpIsInf(x)

    @staticmethod
    def is_zero(x: z3.FPRef) -> z3.BoolRef:
        return z3.fpIsZero(x)

    @staticmethod
    def is_normal(x: z3.FPRef) -> z3.BoolRef:
        return z3.fpIsNormal(x)

    @staticmethod
    def is_subnormal(x: z3.FPRef) -> z3.BoolRef:
        return z3.fpIsSubnormal(x)

    @staticmethod
    def is_negative(x: z3.FPRef) -> z3.BoolRef:
        return z3.fpIsNegative(x)

    @staticmethod
    def is_positive(x: z3.FPRef) -> z3.BoolRef:
        return z3.fpIsPositive(x)

    @staticmethod
    def is_finite(x: z3.FPRef) -> z3.BoolRef:
        """Check if value is finite (not NaN, not Inf)."""
        return z3.And(z3.Not(z3.fpIsNaN(x)), z3.Not(z3.fpIsInf(x)))

    # -- Special value creation --

    @staticmethod
    def nan(sort: z3.FPSortRef) -> z3.FPRef:
        return z3.fpNaN(sort)

    @staticmethod
    def positive_inf(sort: z3.FPSortRef) -> z3.FPRef:
        return z3.fpPlusInfinity(sort)

    @staticmethod
    def negative_inf(sort: z3.FPSortRef) -> z3.FPRef:
        return z3.fpMinusInfinity(sort)

    @staticmethod
    def positive_zero(sort: z3.FPSortRef) -> z3.FPRef:
        return z3.fpPlusZero(sort)

    @staticmethod
    def negative_zero(sort: z3.FPSortRef) -> z3.FPRef:
        return z3.fpMinusZero(sort)

    # -- Arithmetic with rounding mode --

    @staticmethod
    def add(rm: z3.FPRMRef, a: z3.FPRef, b: z3.FPRef) -> z3.FPRef:
        return z3.fpAdd(rm, a, b)

    @staticmethod
    def sub(rm: z3.FPRMRef, a: z3.FPRef, b: z3.FPRef) -> z3.FPRef:
        return z3.fpSub(rm, a, b)

    @staticmethod
    def mul(rm: z3.FPRMRef, a: z3.FPRef, b: z3.FPRef) -> z3.FPRef:
        return z3.fpMul(rm, a, b)

    @staticmethod
    def div(rm: z3.FPRMRef, a: z3.FPRef, b: z3.FPRef) -> z3.FPRef:
        return z3.fpDiv(rm, a, b)

    @staticmethod
    def fma(rm: z3.FPRMRef, a: z3.FPRef, b: z3.FPRef, c: z3.FPRef) -> z3.FPRef:
        """Fused multiply-add: a * b + c."""
        return z3.fpFMA(rm, a, b, c)

    @staticmethod
    def neg(x: z3.FPRef) -> z3.FPRef:
        return z3.fpNeg(x)

    @staticmethod
    def abs(x: z3.FPRef) -> z3.FPRef:
        return z3.fpAbs(x)

    @staticmethod
    def sqrt(rm: z3.FPRMRef, x: z3.FPRef) -> z3.FPRef:
        return z3.fpSqrt(rm, x)

    @staticmethod
    def rem(a: z3.FPRef, b: z3.FPRef) -> z3.FPRef:
        return z3.fpRem(a, b)

    @staticmethod
    def min(a: z3.FPRef, b: z3.FPRef) -> z3.FPRef:
        return z3.fpMin(a, b)

    @staticmethod
    def max(a: z3.FPRef, b: z3.FPRef) -> z3.FPRef:
        return z3.fpMax(a, b)

    # -- Conversions --

    @staticmethod
    def to_bv_signed(rm: z3.FPRMRef, x: z3.FPRef, width: int) -> z3.BitVecRef:
        return z3.fpToSBV(rm, x, z3.BitVecSort(width))

    @staticmethod
    def to_bv_unsigned(rm: z3.FPRMRef, x: z3.FPRef, width: int) -> z3.BitVecRef:
        return z3.fpToUBV(rm, x, z3.BitVecSort(width))

    @staticmethod
    def from_bv_signed(rm: z3.FPRMRef, x: z3.BitVecRef, sort: z3.FPSortRef) -> z3.FPRef:
        return z3.fpSignedToFP(rm, x, sort)

    @staticmethod
    def from_bv_unsigned(rm: z3.FPRMRef, x: z3.BitVecRef, sort: z3.FPSortRef) -> z3.FPRef:
        return z3.fpToFP(rm, x, sort)

    @staticmethod
    def fp_to_fp(rm: z3.FPRMRef, x: z3.FPRef, sort: z3.FPSortRef) -> z3.FPRef:
        """Convert between floating-point sorts."""
        return z3.fpFPToFP(rm, x, sort)

    @staticmethod
    def to_ieee_bv(x: z3.FPRef) -> z3.BitVecRef:
        """Convert to IEEE 754 bit representation."""
        return z3.fpToIEEEBV(x)

    @staticmethod
    def from_ieee_bv(bv: z3.BitVecRef, sort: z3.FPSortRef) -> z3.FPRef:
        """Convert from IEEE 754 bit representation."""
        return z3.fpBVToFP(bv, sort)

    # -- NaN-aware comparison --

    @staticmethod
    def eq_no_nan(a: z3.FPRef, b: z3.FPRef) -> z3.BoolRef:
        """Equality that treats NaN as unequal to everything (IEEE standard)."""
        return z3.fpEQ(a, b)

    @staticmethod
    def eq_bitwise(a: z3.FPRef, b: z3.FPRef) -> z3.BoolRef:
        """Bitwise equality (NaN == NaN if same bits)."""
        return z3.fpToIEEEBV(a) == z3.fpToIEEEBV(b)

    # -- Precision analysis --

    @staticmethod
    def ulp_difference(a: z3.FPRef, b: z3.FPRef, sort: z3.FPSortRef) -> z3.BitVecRef:
        """Compute ULP (unit of least precision) difference between two floats."""
        a_bv = z3.fpToIEEEBV(a)
        b_bv = z3.fpToIEEEBV(b)
        w = a_bv.size()
        diff = z3.If(z3.UGE(a_bv, b_bv), a_bv - b_bv, b_bv - a_bv)
        return diff


# ---------------------------------------------------------------------------
# Array theory helpers
# ---------------------------------------------------------------------------

class ArrayTheory:
    """Helpers for array theory (QF_ABV) encoding."""

    @staticmethod
    def create_array(
        name: str,
        index_sort: z3.SortRef,
        value_sort: z3.SortRef,
    ) -> z3.ArrayRef:
        """Create a fresh symbolic array."""
        return z3.Array(name, index_sort, value_sort)

    @staticmethod
    def byte_array(name: str, address_width: int = 64) -> z3.ArrayRef:
        """Create a byte-addressable memory array."""
        return z3.Array(
            name,
            z3.BitVecSort(address_width),
            z3.BitVecSort(8),
        )

    @staticmethod
    def typed_array(name: str, elem_width: int, index_width: int = 64) -> z3.ArrayRef:
        """Create a typed array (e.g., int32 array)."""
        return z3.Array(
            name,
            z3.BitVecSort(index_width),
            z3.BitVecSort(elem_width),
        )

    @staticmethod
    def select(array: z3.ArrayRef, index: z3.ExprRef) -> z3.ExprRef:
        """Read from array."""
        return z3.Select(array, index)

    @staticmethod
    def store(array: z3.ArrayRef, index: z3.ExprRef, value: z3.ExprRef) -> z3.ArrayRef:
        """Write to array, returning new array."""
        return z3.Store(array, index, value)

    @staticmethod
    def const_array(value: z3.ExprRef, index_sort: z3.SortRef) -> z3.ArrayRef:
        """Create a constant array (all elements equal to value)."""
        return z3.K(index_sort, value)

    @staticmethod
    def store_multi(
        array: z3.ArrayRef,
        base_index: z3.BitVecRef,
        value: z3.BitVecRef,
        num_elements: int,
    ) -> z3.ArrayRef:
        """Store a multi-byte value starting at base_index (little-endian)."""
        elem_width = 8  # Byte array
        for i in range(num_elements):
            byte_val = z3.Extract(
                min((i + 1) * elem_width - 1, value.size() - 1),
                i * elem_width,
                value,
            )
            offset = z3.BitVecVal(i, base_index.size())
            array = z3.Store(array, base_index + offset, byte_val)
        return array

    @staticmethod
    def load_multi(
        array: z3.ArrayRef,
        base_index: z3.BitVecRef,
        num_bytes: int,
    ) -> z3.BitVecRef:
        """Load a multi-byte value from base_index (little-endian)."""
        bytes_loaded = []
        for i in range(num_bytes):
            offset = z3.BitVecVal(i, base_index.size())
            byte_val = z3.Select(array, base_index + offset)
            bytes_loaded.append(byte_val)

        if len(bytes_loaded) == 1:
            return bytes_loaded[0]

        result = bytes_loaded[-1]
        for i in range(len(bytes_loaded) - 2, -1, -1):
            result = z3.Concat(result, bytes_loaded[i])
        return result

    @staticmethod
    def array_equality(
        a: z3.ArrayRef,
        b: z3.ArrayRef,
        base: z3.BitVecRef,
        length: int,
    ) -> z3.BoolRef:
        """Check if two arrays are equal over a range."""
        if length == 0:
            return z3.BoolVal(True)
        conditions = []
        for i in range(length):
            offset = z3.BitVecVal(i, base.size())
            idx = base + offset
            conditions.append(z3.Select(a, idx) == z3.Select(b, idx))
        return z3.And(*conditions)

    @staticmethod
    def array_fill(
        array: z3.ArrayRef,
        base: z3.BitVecRef,
        value: z3.ExprRef,
        length: int,
    ) -> z3.ArrayRef:
        """Fill array range with a value."""
        for i in range(length):
            offset = z3.BitVecVal(i, base.size())
            array = z3.Store(array, base + offset, value)
        return array

    @staticmethod
    def array_copy(
        dst: z3.ArrayRef,
        dst_base: z3.BitVecRef,
        src: z3.ArrayRef,
        src_base: z3.BitVecRef,
        length: int,
    ) -> z3.ArrayRef:
        """Copy array range from src to dst."""
        for i in range(length):
            offset = z3.BitVecVal(i, dst_base.size())
            val = z3.Select(src, src_base + offset)
            dst = z3.Store(dst, dst_base + offset, val)
        return dst
