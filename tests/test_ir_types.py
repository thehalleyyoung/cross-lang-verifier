"""Tests for IR types: creation, compatibility, promotion, size/alignment, serialization."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ir.types import (
    IRType, VoidType, IntType, FloatType, PointerType, ArrayType,
    StructType, StructField, FunctionType, Signedness, FloatKind,
    check_compatibility, type_join, type_meet, are_layout_compatible,
    type_from_dict,
)


class TestIntType:
    def test_create_signed(self):
        t = IntType(32, Signedness.SIGNED)
        assert t.size_bits() == 32
        assert t.is_signed
        assert not t.is_unsigned
        assert t.is_integer()

    def test_create_unsigned(self):
        t = IntType(32, Signedness.UNSIGNED)
        assert t.is_unsigned
        assert not t.is_signed

    def test_bool_type(self):
        t = IntType(1, Signedness.UNSIGNED)
        assert t.is_bool

    def test_various_widths(self):
        for w in [1, 8, 16, 32, 64, 128]:
            t = IntType(w, Signedness.SIGNED)
            assert t.size_bits() == w

    def test_max_min_signed(self):
        t = IntType(8, Signedness.SIGNED)
        assert t.max_value == 127
        assert t.min_value == -128

    def test_max_min_unsigned(self):
        t = IntType(8, Signedness.UNSIGNED)
        assert t.max_value == 255
        assert t.min_value == 0

    def test_mask(self):
        t = IntType(8, Signedness.UNSIGNED)
        assert t.mask() == 0xFF

    def test_truncate(self):
        t = IntType(8, Signedness.UNSIGNED)
        assert t.truncate(256) == 0
        assert t.truncate(255) == 255
        assert t.truncate(300) == 44

    def test_contains(self):
        t = IntType(8, Signedness.SIGNED)
        assert t.contains(0)
        assert t.contains(127)
        assert t.contains(-128)
        assert not t.contains(128)
        assert not t.contains(-129)

    def test_to_unsigned(self):
        t = IntType(32, Signedness.SIGNED)
        u = t.to_unsigned()
        assert u.is_unsigned
        assert u.size_bits() == 32

    def test_to_signed(self):
        t = IntType(32, Signedness.UNSIGNED)
        s = t.to_signed()
        assert s.is_signed
        assert s.size_bits() == 32

    def test_widen(self):
        t = IntType(16, Signedness.SIGNED)
        w = t.widen(32)
        assert w.size_bits() == 32
        assert w.is_signed

    def test_serialization(self):
        t = IntType(32, Signedness.SIGNED)
        d = t.to_dict()
        assert d["kind"] == "int" or "width" in d
        t2 = type_from_dict(d)
        assert isinstance(t2, IntType)
        assert t2.size_bits() == 32

    def test_is_predicates(self):
        t = IntType(32, Signedness.SIGNED)
        assert t.is_integer()
        assert t.is_numeric()
        assert t.is_scalar()
        assert not t.is_float()
        assert not t.is_pointer()
        assert not t.is_void()
        assert not t.is_aggregate()
        assert t.is_sized()


class TestFloatType:
    def test_f32(self):
        t = FloatType(FloatKind.F32)
        assert t.size_bits() == 32
        assert t.is_float()
        assert t.is_numeric()
        assert not t.is_integer()

    def test_f64(self):
        t = FloatType(FloatKind.F64)
        assert t.size_bits() == 64
        assert t.is_float()

    def test_serialization(self):
        t = FloatType(FloatKind.F64)
        d = t.to_dict()
        t2 = type_from_dict(d)
        assert isinstance(t2, FloatType)
        assert t2.size_bits() == 64


class TestVoidType:
    def test_void(self):
        t = VoidType()
        assert t.is_void()
        assert t.size_bits() == 0
        assert not t.is_sized()
        assert not t.is_integer()

    def test_serialization(self):
        t = VoidType()
        d = t.to_dict()
        t2 = type_from_dict(d)
        assert isinstance(t2, VoidType)


class TestPointerType:
    def test_create(self):
        pointee = IntType(32, Signedness.SIGNED)
        p = PointerType(pointee)
        assert p.is_pointer()
        assert p.is_scalar()
        assert not p.is_integer()

    def test_nested(self):
        i32 = IntType(32, Signedness.SIGNED)
        p1 = PointerType(i32)
        p2 = PointerType(p1)
        assert p2.is_pointer()

    def test_serialization(self):
        i32 = IntType(32, Signedness.SIGNED)
        p = PointerType(i32)
        d = p.to_dict()
        p2 = type_from_dict(d)
        assert isinstance(p2, PointerType)


class TestArrayType:
    def test_create(self):
        elem = IntType(32, Signedness.SIGNED)
        arr = ArrayType(elem, 10)
        assert arr.is_aggregate()
        assert arr.size_bits() == 320

    def test_serialization(self):
        elem = IntType(8, Signedness.UNSIGNED)
        arr = ArrayType(elem, 4)
        d = arr.to_dict()
        arr2 = type_from_dict(d)
        assert isinstance(arr2, ArrayType)


class TestStructType:
    def test_create(self):
        i32 = IntType(32, Signedness.SIGNED)
        f64 = FloatType(FloatKind.F64)
        fields = [
            StructField("x", i32, 0),
            StructField("y", f64, 32),
        ]
        s = StructType("point", fields, packed=False)
        assert s.is_aggregate()
        assert not s.is_scalar()

    def test_serialization(self):
        i32 = IntType(32, Signedness.SIGNED)
        fields = [StructField("a", i32, 0)]
        s = StructType("simple", fields, packed=True)
        d = s.to_dict()
        s2 = type_from_dict(d)
        assert isinstance(s2, StructType)


class TestFunctionType:
    def test_create(self):
        i32 = IntType(32, Signedness.SIGNED)
        ft = FunctionType(return_type=i32, param_types=[i32, i32])
        assert ft.is_function()
        assert not ft.is_variadic

    def test_variadic(self):
        i32 = IntType(32, Signedness.SIGNED)
        ft = FunctionType(return_type=i32, param_types=[i32], is_variadic=True)
        assert ft.is_variadic

    def test_serialization(self):
        i32 = IntType(32, Signedness.SIGNED)
        ft = FunctionType(return_type=i32, param_types=[i32])
        d = ft.to_dict()
        ft2 = type_from_dict(d)
        assert isinstance(ft2, FunctionType)


class TestTypeCompatibility:
    def test_same_type(self):
        t = IntType(32, Signedness.SIGNED)
        result = check_compatibility(t, t)
        assert result is not None

    def test_signed_unsigned(self):
        s = IntType(32, Signedness.SIGNED)
        u = IntType(32, Signedness.UNSIGNED)
        result = check_compatibility(s, u)
        assert result is not None

    def test_different_width(self):
        i16 = IntType(16, Signedness.SIGNED)
        i32 = IntType(32, Signedness.SIGNED)
        result = check_compatibility(i16, i32)
        assert result is not None


class TestTypeJoinMeet:
    def test_join_same(self):
        t = IntType(32, Signedness.SIGNED)
        result = type_join(t, t)
        assert isinstance(result, IntType)
        assert result.size_bits() == 32

    def test_join_different_width(self):
        i16 = IntType(16, Signedness.SIGNED)
        i32 = IntType(32, Signedness.SIGNED)
        result = type_join(i16, i32)
        assert isinstance(result, IntType)
        assert result.size_bits() >= 32

    def test_meet_same(self):
        t = IntType(32, Signedness.SIGNED)
        result = type_meet(t, t)
        assert isinstance(result, IntType)

    def test_layout_compatible(self):
        s = IntType(32, Signedness.SIGNED)
        u = IntType(32, Signedness.UNSIGNED)
        assert are_layout_compatible(s, u)

    def test_layout_incompatible_size(self):
        i8 = IntType(8, Signedness.SIGNED)
        i32 = IntType(32, Signedness.SIGNED)
        assert not are_layout_compatible(i8, i32)


class TestAlignBits:
    def test_int_alignment(self):
        for w in [8, 16, 32, 64]:
            t = IntType(w, Signedness.SIGNED)
            assert t.align_bits() >= 8
            assert t.align_bits() <= t.size_bits()

    def test_float_alignment(self):
        f32 = FloatType(FloatKind.F32)
        assert f32.align_bits() == 32
        f64 = FloatType(FloatKind.F64)
        assert f64.align_bits() == 64
