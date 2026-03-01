"""Tests for IR builder: build simple functions, verify insertion, SSA numbering."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ir.types import IntType, FloatType, VoidType, PointerType, Signedness, FloatKind, FunctionType
from src.ir.instructions import BinOpKind, CmpPredicate, CastKind, Constant
from src.ir.basic_block import BasicBlock
from src.ir.function import Function
from src.ir.module import Module
from src.ir.builder import IRBuilder


def make_module(name="test"):
    return Module(name=name, source_filename="test.c",
                  target_triple="x86_64-unknown-linux-gnu",
                  data_layout="", language="C")


def make_i32():
    return IntType(32, Signedness.SIGNED)


def make_u32():
    return IntType(32, Signedness.UNSIGNED)


class TestBuilderBasic:
    def test_create_builder(self):
        builder = IRBuilder()
        assert builder is not None

    def test_position_at_end(self):
        bb = BasicBlock("entry")
        builder = IRBuilder()
        builder.position_at_end(bb)
        # Builder should be ready to insert

    def test_const_int(self):
        builder = IRBuilder()
        c = builder.const_int(42, IntType(32, Signedness.SIGNED))
        assert c.value == 42

    def test_const_float(self):
        builder = IRBuilder()
        c = builder.const_float(3.14, FloatType(FloatKind.F64))
        assert c.value == pytest.approx(3.14)

    def test_const_bool(self):
        builder = IRBuilder()
        t = builder.const_bool(True)
        assert t.value == True

    def test_const_null(self):
        builder = IRBuilder()
        i32 = make_i32()
        n = builder.const_null(PointerType(i32))
        assert n.is_null


class TestBuilderArithmetic:
    def test_add(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32, i32])
        func = mod.create_function("add", ft)
        bb = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(bb)
        a, b = func.arguments
        result = builder.add(a, b, name="sum")
        builder.ret(result)

        assert len(list(bb.instructions)) >= 2

    def test_sub(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32, i32])
        func = mod.create_function("sub", ft)
        bb = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(bb)
        a, b = func.arguments
        result = builder.sub(a, b, name="diff")
        builder.ret(result)
        assert bb.has_terminator

    def test_mul(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32, i32])
        func = mod.create_function("mul", ft)
        bb = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(bb)
        a, b = func.arguments
        result = builder.mul(a, b, name="product")
        builder.ret(result)
        assert bb.has_terminator

    def test_float_arithmetic(self):
        mod = make_module()
        f64 = FloatType(FloatKind.F64)
        ft = FunctionType(f64, [f64, f64])
        func = mod.create_function("fadd", ft)
        bb = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(bb)
        a, b = func.arguments
        result = builder.fadd(a, b, name="fsum")
        builder.ret(result)
        assert bb.has_terminator


class TestBuilderComparisons:
    def test_icmp_eq(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(IntType(1, Signedness.UNSIGNED), [i32, i32])
        func = mod.create_function("eq", ft)
        bb = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(bb)
        a, b = func.arguments
        result = builder.icmp_eq(a, b, name="eq")
        builder.ret(result)

    def test_icmp_all_predicates(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(IntType(1, Signedness.UNSIGNED), [i32, i32])
        func = mod.create_function("cmp", ft)
        bb = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(bb)
        a, b = func.arguments

        cmps = [
            builder.icmp_eq, builder.icmp_ne,
            builder.icmp_slt, builder.icmp_sle,
            builder.icmp_sgt, builder.icmp_sge,
            builder.icmp_ult, builder.icmp_ule,
            builder.icmp_ugt, builder.icmp_uge,
        ]
        for i, cmp_fn in enumerate(cmps):
            cmp_fn(a, b, name=f"cmp{i}")


class TestBuilderControlFlow:
    def test_unconditional_branch(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32])
        func = mod.create_function("f", ft)
        entry = func.create_block("entry")
        target = func.create_block("target")

        builder = IRBuilder()
        builder.position_at_end(entry)
        builder.br(target)

        builder.position_at_end(target)
        builder.ret(func.arguments[0])

    def test_conditional_branch(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32, i32])
        func = mod.create_function("max", ft)
        entry = func.create_block("entry")
        then_bb = func.create_block("then")
        else_bb = func.create_block("else")

        builder = IRBuilder()
        builder.position_at_end(entry)
        a, b = func.arguments
        cond = builder.icmp_sgt(a, b, name="cond")
        builder.cond_br(cond, then_bb, else_bb)

        builder.position_at_end(then_bb)
        builder.ret(a)

        builder.position_at_end(else_bb)
        builder.ret(b)

    def test_phi_node(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32, i32])
        func = mod.create_function("max", ft)
        entry = func.create_block("entry")
        then_bb = func.create_block("then")
        else_bb = func.create_block("else")
        merge = func.create_block("merge")

        builder = IRBuilder()
        builder.position_at_end(entry)
        a, b = func.arguments
        cond = builder.icmp_sgt(a, b, name="cond")
        builder.cond_br(cond, then_bb, else_bb)

        builder.position_at_end(then_bb)
        builder.br(merge)

        builder.position_at_end(else_bb)
        builder.br(merge)

        builder.position_at_end(merge)
        phi = builder.phi(i32, name="result")
        phi.add_incoming(a, then_bb)
        phi.add_incoming(b, else_bb)
        builder.ret(phi)

    def test_select(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32, i32])
        func = mod.create_function("max", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        a, b = func.arguments
        cond = builder.icmp_sgt(a, b)
        result = builder.select(cond, a, b, name="max")
        builder.ret(result)


class TestBuilderMemory:
    def test_alloca_load_store(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32])
        func = mod.create_function("f", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        ptr = builder.alloca(i32, name="ptr")
        builder.store(func.arguments[0], ptr)
        val = builder.load(ptr, i32, name="val")
        builder.ret(val)

    def test_gep(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [PointerType(i32), i32])
        func = mod.create_function("idx", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        base, idx = func.arguments
        ptr = builder.gep(i32, base, [idx], name="ptr")
        val = builder.load(ptr, i32, name="val")
        builder.ret(val)


class TestBuilderCasts:
    def test_trunc(self):
        mod = make_module()
        i32 = make_i32()
        i16 = IntType(16, Signedness.SIGNED)
        ft = FunctionType(i16, [i32])
        func = mod.create_function("trunc", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        result = builder.trunc(func.arguments[0], i16, name="t")
        builder.ret(result)

    def test_sext(self):
        mod = make_module()
        i16 = IntType(16, Signedness.SIGNED)
        i32 = make_i32()
        ft = FunctionType(i32, [i16])
        func = mod.create_function("sext", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        result = builder.sext(func.arguments[0], i32, name="ext")
        builder.ret(result)

    def test_zext(self):
        mod = make_module()
        i16 = IntType(16, Signedness.UNSIGNED)
        i32 = make_u32()
        ft = FunctionType(i32, [i16])
        func = mod.create_function("zext", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        result = builder.zext(func.arguments[0], i32, name="ext")
        builder.ret(result)


class TestBuilderPatterns:
    def test_build_increment(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32])
        func = mod.create_function("inc", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        result = builder.build_increment(func.arguments[0], name="inc")
        builder.ret(result)

    def test_build_decrement(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32])
        func = mod.create_function("dec", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        result = builder.build_decrement(func.arguments[0], name="dec")
        builder.ret(result)

    def test_ret_void(self):
        mod = make_module()
        ft = FunctionType(VoidType(), [])
        func = mod.create_function("noop", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        builder.ret_void()
        assert bb_has_terminator(entry)


def bb_has_terminator(bb):
    """Helper to check if a basic block has a terminator."""
    return bb.has_terminator
