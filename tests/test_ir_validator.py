"""Tests for IR validator: valid programs pass, invalid programs caught."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ir.types import IntType, VoidType, FloatType, Signedness, FloatKind, FunctionType
from src.ir.instructions import Constant
from src.ir.basic_block import BasicBlock
from src.ir.function import Function
from src.ir.module import Module
from src.ir.builder import IRBuilder
from src.ir.validator import IRValidator, ValidationResult, Severity


def make_module():
    return Module(name="test", source_filename="test.c",
                  target_triple="x86_64-unknown-linux-gnu",
                  data_layout="", language="C")


def make_i32():
    return IntType(32, Signedness.SIGNED)


class TestValidPrograms:
    def test_simple_add(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32, i32])
        func = mod.create_function("add", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        a, b = func.arguments
        result = builder.add(a, b, name="sum")
        builder.ret(result)

        validator = IRValidator(strict=True)
        result = validator.validate_function(func)
        assert result.is_valid

    def test_void_function(self):
        mod = make_module()
        ft = FunctionType(VoidType(), [])
        func = mod.create_function("noop", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        builder.ret_void()

        validator = IRValidator(strict=True)
        result = validator.validate_function(func)
        assert result.is_valid

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
        cond = builder.icmp_sgt(a, b)
        builder.cond_br(cond, then_bb, else_bb)

        builder.position_at_end(then_bb)
        builder.ret(a)

        builder.position_at_end(else_bb)
        builder.ret(b)

        validator = IRValidator(strict=True)
        result = validator.validate_function(func)
        assert result.is_valid

    def test_with_alloca(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32])
        func = mod.create_function("f", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        ptr = builder.alloca(i32, name="tmp")
        builder.store(func.arguments[0], ptr)
        val = builder.load(ptr, i32, name="val")
        builder.ret(val)

        validator = IRValidator(strict=True)
        result = validator.validate_function(func)
        assert result.is_valid

    def test_module_validation(self, sample_ir_module):
        validator = IRValidator(strict=True)
        result = validator.validate_module(sample_ir_module)
        assert result.is_valid


class TestInvalidPrograms:
    def test_no_terminator(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [i32])
        func = mod.create_function("f", ft)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        builder.add(func.arguments[0], Constant.int_const(32, 1))
        # No terminator!

        validator = IRValidator(strict=True)
        result = validator.validate_function(func)
        assert not result.is_valid
        assert result.num_errors > 0

    def test_empty_function(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [])
        func = mod.create_function("empty", ft)
        # No basic blocks at all

        validator = IRValidator(strict=True)
        result = validator.validate_function(func)
        assert not result.is_valid

    def test_no_entry_block(self):
        mod = make_module()
        i32 = make_i32()
        ft = FunctionType(i32, [])
        func = Function("f", ft)
        # Function with no blocks

        validator = IRValidator(strict=True)
        result = validator.validate_function(func)
        assert not result.is_valid


class TestValidationResult:
    def test_empty_result(self):
        result = ValidationResult([])
        assert result.is_valid
        assert result.num_errors == 0
        assert result.num_warnings == 0

    def test_result_with_errors(self):
        from src.ir.validator import ValidationMessage
        msgs = [
            ValidationMessage(Severity.ERROR, "bad thing", "func:bb1"),
            ValidationMessage(Severity.WARNING, "suspicious", "func:bb2"),
        ]
        result = ValidationResult(msgs)
        assert not result.is_valid
        assert result.num_errors == 1
        assert result.num_warnings == 1

    def test_merge(self):
        r1 = ValidationResult([])
        from src.ir.validator import ValidationMessage
        r2 = ValidationResult([
            ValidationMessage(Severity.ERROR, "error", "loc"),
        ])
        r1.merge(r2)
        assert not r1.is_valid
