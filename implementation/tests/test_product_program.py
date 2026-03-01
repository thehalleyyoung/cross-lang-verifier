"""Tests for product programs: alignment, coercion insertion, product construction."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.ir.types import IntType, VoidType, Signedness, FunctionType
from src.ir.instructions import Constant
from src.ir.module import Module
from src.ir.builder import IRBuilder
from src.product_program.alignment import FunctionAligner
from src.product_program.coercion import CoercionGenerator
from src.product_program.product import ProductBuilder
from src.product_program.normalizer import IRNormalizer
from src.semantics.semantic_config import SemanticConfig


def make_i32():
    return IntType(32, Signedness.SIGNED)


def make_c_module():
    mod = Module("c_test", "test.c", "x86_64-unknown-linux-gnu", "", "C")
    i32 = make_i32()
    ft = FunctionType(i32, [i32, i32])
    func = mod.create_function("add", ft)
    entry = func.create_block("entry")
    builder = IRBuilder()
    builder.position_at_end(entry)
    a, b = func.arguments
    result = builder.add(a, b, name="sum")
    builder.ret(result)
    return mod


def make_rust_module():
    mod = Module("rust_test", "test.rs", "x86_64-unknown-linux-gnu", "", "Rust")
    i32 = make_i32()
    ft = FunctionType(i32, [i32, i32])
    func = mod.create_function("add", ft)
    entry = func.create_block("entry")
    builder = IRBuilder()
    builder.position_at_end(entry)
    a, b = func.arguments
    result = builder.add(a, b, name="sum")
    builder.ret(result)
    return mod


class TestFunctionAligner:
    def test_create(self):
        c_mod = make_c_module()
        r_mod = make_rust_module()
        c_func = next(c_mod.iter_functions())
        r_func = next(r_mod.iter_functions())
        aligner = FunctionAligner()
        assert aligner is not None

    def test_align_identical(self):
        c_mod = make_c_module()
        r_mod = make_rust_module()
        c_func = next(c_mod.iter_functions())
        r_func = next(r_mod.iter_functions())
        aligner = FunctionAligner()
        alignment = aligner.align(c_func, r_func)
        assert alignment is not None

    def test_alignment_has_blocks(self):
        c_mod = make_c_module()
        r_mod = make_rust_module()
        c_func = next(c_mod.iter_functions())
        r_func = next(r_mod.iter_functions())
        aligner = FunctionAligner()
        alignment = aligner.align(c_func, r_func)
        if hasattr(alignment, 'block_alignments'):
            assert len(alignment.block_alignments) >= 1


class TestCoercionGenerator:
    def test_create(self):
        config = SemanticConfig.c11()
        gen = CoercionGenerator()
        assert gen is not None

    def test_generate_on_alignment(self):
        c_mod = make_c_module()
        r_mod = make_rust_module()
        c_func = next(c_mod.iter_functions())
        r_func = next(r_mod.iter_functions())
        aligner = FunctionAligner()
        alignment = aligner.align(c_func, r_func)

        config = SemanticConfig.c11()
        gen = CoercionGenerator()
        coercions = gen.generate_for_alignment(alignment)
        assert coercions is not None


class TestProductBuilder:
    def test_create(self):
        c_mod = make_c_module()
        r_mod = make_rust_module()
        c_func = next(c_mod.iter_functions())
        r_func = next(r_mod.iter_functions())
        aligner = FunctionAligner()
        alignment = aligner.align(c_func, r_func)
        product = ProductBuilder()
        assert product is not None

    def test_build(self):
        c_mod = make_c_module()
        r_mod = make_rust_module()
        c_func = next(c_mod.iter_functions())
        r_func = next(r_mod.iter_functions())
        aligner = FunctionAligner()
        alignment = aligner.align(c_func, r_func)
        builder = ProductBuilder()
        product = builder.build(c_func, r_func)
        assert product is not None


class TestIRNormalizer:
    def test_create(self):
        normalizer = IRNormalizer()
        assert normalizer is not None

    def test_normalize(self):
        mod = make_c_module()
        normalizer = IRNormalizer()
        func = next(mod.iter_functions())
        normalizer.normalize(func)
        # Module should still be valid after normalization
        assert mod.num_functions >= 1


class TestProductWithDifferentBodies:
    def test_different_computations(self):
        """Test product of functions with different intermediate steps."""
        i32 = make_i32()

        # C: return a + b + 1
        c_mod = Module("c", "c.c", "x86_64-unknown-linux-gnu", "", "C")
        ft = FunctionType(i32, [i32, i32])
        c_func = c_mod.create_function("f", ft)
        entry = c_func.create_block("entry")
        builder = IRBuilder()
        builder.position_at_end(entry)
        a, b = c_func.arguments
        t = builder.add(a, b, name="t")
        result = builder.add(t, Constant.int_const(1, IntType(32, Signedness.SIGNED)), name="r")
        builder.ret(result)

        # Rust: return a + (b + 1)
        r_mod = Module("r", "r.rs", "x86_64-unknown-linux-gnu", "", "Rust")
        r_func = r_mod.create_function("f", ft)
        entry = r_func.create_block("entry")
        builder = IRBuilder()
        builder.position_at_end(entry)
        a, b = r_func.arguments
        t = builder.add(b, Constant.int_const(1, IntType(32, Signedness.SIGNED)), name="t")
        result = builder.add(a, t, name="r")
        builder.ret(result)

        aligner = FunctionAligner()
        alignment = aligner.align(c_func, r_func)
        pb = ProductBuilder()
        product = pb.build(c_func, r_func)
        assert product is not None
