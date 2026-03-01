"""Tests for IR instructions: creation, validation, operand types, visitor, cloning."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ir.types import IntType, FloatType, VoidType, PointerType, Signedness, FloatKind, FunctionType
from src.ir.instructions import (
    Value, Constant, Argument, BinaryOp, UnaryOp, CompareOp,
    CastInst, LoadInst, StoreInst, AllocaInst, CallInst,
    ReturnInst, BranchInst, PhiInst, SelectInst,
    GetElementPtrInst, MemcpyInst, MemsetInst,
    BinOpKind, UnaryOpKind, CmpPredicate, CastKind,
    InstructionVisitor, InstructionMetadata, SourceLocation,
)
from src.ir.basic_block import BasicBlock
from src.ir.function import Function


class TestValue:
    def test_create(self):
        i32 = IntType(32, Signedness.SIGNED)
        v = Value(i32, "x")
        assert v.name == "x"
        assert v.type is i32

    def test_users_tracking(self):
        i32 = IntType(32, Signedness.SIGNED)
        v = Value(i32, "x")
        assert not v.has_users()


class TestConstant:
    def test_int_const(self):
        c = Constant.int_const(42, IntType(32, Signedness.SIGNED))
        assert c.value == 42
        assert c.type.size_bits() == 32

    def test_float_const(self):
        c = Constant.float_const(3.14, FloatType(FloatKind.F32))
        assert c.value == pytest.approx(3.14, rel=1e-5)

    def test_null_ptr(self):
        i32 = IntType(32, Signedness.SIGNED)
        c = Constant.null_ptr(i32)
        assert c.is_null

    def test_bool_const(self):
        t = Constant.bool_const(True)
        assert t.value == True
        f = Constant.bool_const(False)
        assert f.value == False

    def test_undef(self):
        i32 = IntType(32, Signedness.SIGNED)
        u = Constant.undef(i32)
        assert u.is_undef

    def test_is_zero(self):
        c = Constant.int_const(0, IntType(32, Signedness.SIGNED))
        assert c.is_zero
        c2 = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        assert not c2.is_zero


class TestArgument:
    def test_create(self):
        i32 = IntType(32, Signedness.SIGNED)
        arg = Argument(i32, 0, "a")
        assert arg.name == "a"
        assert arg.index == 0


class TestBinaryOp:
    def test_add(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        inst = BinaryOp(BinOpKind.ADD, a, b, i32, name="sum")
        assert inst.opcode_name() == "add" or "ADD" in inst.opcode_name().upper()
        assert inst.num_operands == 2
        assert inst.get_operand(0) is a
        assert inst.get_operand(1) is b

    def test_sub(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(5, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(3, IntType(32, Signedness.SIGNED))
        inst = BinaryOp(BinOpKind.SUB, a, b, i32, name="diff")
        assert inst.name == "diff"

    def test_mul(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(3, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(4, IntType(32, Signedness.SIGNED))
        inst = BinaryOp(BinOpKind.MUL, a, b, i32)
        assert inst.num_operands == 2

    def test_div_ops(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(10, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(3, IntType(32, Signedness.SIGNED))
        for op in [BinOpKind.SDIV, BinOpKind.UDIV, BinOpKind.SREM, BinOpKind.UREM]:
            inst = BinaryOp(op, a, b, i32)
            assert inst.num_operands == 2

    def test_float_ops(self):
        f64 = FloatType(FloatKind.F64)
        a = Constant.float_const(1.0, FloatType(FloatKind.F64))
        b = Constant.float_const(2.0, FloatType(FloatKind.F64))
        for op in [BinOpKind.FADD, BinOpKind.FSUB, BinOpKind.FMUL, BinOpKind.FDIV]:
            inst = BinaryOp(op, a, b, f64)
            assert inst.num_operands == 2

    def test_bitwise_ops(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(0xFF, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(0x0F, IntType(32, Signedness.SIGNED))
        for op in [BinOpKind.AND, BinOpKind.OR, BinOpKind.XOR,
                    BinOpKind.SHL, BinOpKind.LSHR, BinOpKind.ASHR]:
            inst = BinaryOp(op, a, b, i32)
            assert inst.num_operands == 2

    def test_set_operand(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        c = Constant.int_const(3, IntType(32, Signedness.SIGNED))
        inst = BinaryOp(BinOpKind.ADD, a, b, i32)
        inst.set_operand(1, c)
        assert inst.get_operand(1) is c

    def test_clone(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        inst = BinaryOp(BinOpKind.ADD, a, b, i32, name="sum")
        cloned = inst.clone()
        assert cloned is not inst
        assert cloned.op == inst.op


class TestUnaryOp:
    def test_neg(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(5, IntType(32, Signedness.SIGNED))
        inst = UnaryOp(UnaryOpKind.NEG, a, i32, name="neg")
        assert inst.num_operands == 1

    def test_not(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(0xFF, IntType(32, Signedness.SIGNED))
        inst = UnaryOp(UnaryOpKind.BITWISE_NOT, a, i32)
        assert inst.num_operands == 1


class TestCompareOp:
    def test_integer_comparison(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        for pred in [CmpPredicate.EQ, CmpPredicate.NE, CmpPredicate.SLT,
                      CmpPredicate.SLE, CmpPredicate.SGT, CmpPredicate.SGE]:
            inst = CompareOp(pred, a, b, name="cmp")
            assert inst.num_operands == 2

    def test_unsigned_comparison(self):
        i32 = IntType(32, Signedness.UNSIGNED)
        a = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        for pred in [CmpPredicate.ULT, CmpPredicate.ULE,
                      CmpPredicate.UGT, CmpPredicate.UGE]:
            inst = CompareOp(pred, a, b)
            assert inst.num_operands == 2


class TestCastInst:
    def test_trunc(self):
        i32 = IntType(32, Signedness.SIGNED)
        i16 = IntType(16, Signedness.SIGNED)
        v = Constant.int_const(1000, IntType(32, Signedness.SIGNED))
        inst = CastInst(CastKind.TRUNC, v, i16, name="trunc")
        assert inst.num_operands == 1

    def test_zext(self):
        i16 = IntType(16, Signedness.UNSIGNED)
        i32 = IntType(32, Signedness.UNSIGNED)
        v = Constant.int_const(16, 100)
        inst = CastInst(CastKind.ZEXT, v, i32, name="zext")
        assert inst.num_operands == 1

    def test_sext(self):
        i16 = IntType(16, Signedness.SIGNED)
        i32 = IntType(32, Signedness.SIGNED)
        v = Constant.int_const(16, -10)
        inst = CastInst(CastKind.SEXT, v, i32, name="sext")
        assert inst.num_operands == 1

    def test_bitcast(self):
        i32 = IntType(32, Signedness.SIGNED)
        f32 = FloatType(FloatKind.F32)
        v = Constant.int_const(0, IntType(32, Signedness.SIGNED))
        inst = CastInst(CastKind.BITCAST, v, f32, name="bc")
        assert inst.num_operands == 1


class TestMemoryInstructions:
    def test_alloca(self):
        i32 = IntType(32, Signedness.SIGNED)
        inst = AllocaInst(i32, num_elements=1, name="ptr")
        assert inst.num_operands >= 0

    def test_load(self):
        i32 = IntType(32, Signedness.SIGNED)
        ptr = Constant.null_ptr(i32)
        inst = LoadInst(ptr, i32, name="val")
        assert inst.num_operands >= 1

    def test_store(self):
        i32 = IntType(32, Signedness.SIGNED)
        val = Constant.int_const(42, IntType(32, Signedness.SIGNED))
        ptr = Constant.null_ptr(i32)
        inst = StoreInst(val, ptr)
        assert inst.num_operands >= 2


class TestControlFlow:
    def test_return_value(self):
        i32 = IntType(32, Signedness.SIGNED)
        val = Constant.int_const(0, IntType(32, Signedness.SIGNED))
        inst = ReturnInst(val)
        assert inst.num_operands >= 1

    def test_return_void(self):
        inst = ReturnInst(None)
        assert inst.num_operands >= 0

    def test_branch(self):
        bb1 = BasicBlock("target")
        inst = BranchInst(bb1)
        # Unconditional branch
        assert inst is not None

    def test_conditional_branch(self):
        bb1 = BasicBlock("true_bb")
        bb2 = BasicBlock("false_bb")
        cond = Constant.bool_const(True)
        inst = BranchInst(bb1, condition=cond, false_target=bb2)
        assert inst is not None


class TestPhiInst:
    def test_create(self):
        i32 = IntType(32, Signedness.SIGNED)
        bb1 = BasicBlock("bb1")
        bb2 = BasicBlock("bb2")
        v1 = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        v2 = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        phi = PhiInst(i32, [(v1, bb1), (v2, bb2)], name="phi")
        assert phi.name == "phi"


class TestSelectInst:
    def test_create(self):
        i32 = IntType(32, Signedness.SIGNED)
        cond = Constant.bool_const(True)
        a = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        inst = SelectInst(cond, a, b, name="sel")
        assert inst.name == "sel"


class TestCallInst:
    def test_create(self):
        i32 = IntType(32, Signedness.SIGNED)
        ft = FunctionType(i32, [i32, i32])
        callee = Value(ft, "add_func")
        a = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        inst = CallInst(callee, [a, b], i32, callee_name="add_func", name="result")
        assert inst.callee_name == "add_func"


class TestInstructionMetadata:
    def test_source_location(self):
        loc = SourceLocation("test.c", 10, 5, "C")
        assert loc.file == "test.c"
        assert loc.line == 10
        assert loc.column == 5

    def test_metadata(self):
        loc = SourceLocation("test.c", 1, 1, "C")
        meta = InstructionMetadata(source_loc=loc, comment="test instruction")
        assert meta.comment == "test instruction"


class TestInstructionVisitor:
    def test_visitor_pattern(self):
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(1, IntType(32, Signedness.SIGNED))
        b = Constant.int_const(2, IntType(32, Signedness.SIGNED))
        inst = BinaryOp(BinOpKind.ADD, a, b, i32, name="sum")

        visited = []

        class TestVisitor(InstructionVisitor):
            def visit_binary_op(self, inst):
                visited.append(("binary", inst.name))
                return None

        visitor = TestVisitor()
        inst.accept(visitor)
        assert len(visited) == 1
        assert visited[0] == ("binary", "sum")


class TestGEP:
    def test_create(self):
        i32 = IntType(32, Signedness.SIGNED)
        arr_ty = IntType(32, Signedness.SIGNED)
        base = Constant.null_ptr(arr_ty)
        idx = Constant.int_const(0, IntType(32, Signedness.SIGNED))
        ptr_ty = PointerType(i32)
        gep = GetElementPtrInst(arr_ty, base, [idx], inbounds=True,
                                result_type=ptr_ty, name="gep")
        assert gep.name == "gep"
