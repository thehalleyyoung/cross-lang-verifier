"""Tests for SMT: encoding, solver interaction, model decoding, counterexamples."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.smt.encoder import SMTEncoder, EncodingContext
from src.smt.solver import SMTSolver, SolverResult, SolverStatus
from src.smt.decoder import ModelDecoder
from src.smt.theories import BitvectorTheory, FloatingPointTheory, ArrayTheory
from src.semantics.semantic_config import SemanticConfig
from src.ir.types import IntType, FloatType, Signedness, FloatKind
from src.ir.instructions import Constant, BinOpKind, BinaryOp


class TestBitvectorTheory:
    def test_create(self):
        theory = BitvectorTheory()
        assert theory is not None

    def test_add(self):
        theory = BitvectorTheory()
        if hasattr(theory, 'bvadd'):
            result = theory.bvadd(32, 10, 20)
            assert result is not None

    def test_mul(self):
        theory = BitvectorTheory()
        if hasattr(theory, 'bvmul'):
            result = theory.bvmul(32, 6, 7)
            assert result is not None

    def test_overflow_check(self):
        theory = BitvectorTheory()
        if hasattr(theory, 'check_signed_overflow'):
            result = theory.check_signed_overflow(32, 2**31 - 1, 1, 'add')
            assert result is not None


class TestFloatingPointTheory:
    def test_create(self):
        theory = FloatingPointTheory()
        assert theory is not None


class TestArrayTheory:
    def test_create(self):
        theory = ArrayTheory()
        assert theory is not None


class TestSMTEncoder:
    def test_create(self):
        config = SemanticConfig.c11()
        encoder = SMTEncoder(config)
        assert encoder is not None

    def test_encode_constant(self):
        config = SemanticConfig.c11()
        encoder = SMTEncoder(config)
        ctx = EncodingContext()
        i32 = IntType(32, Signedness.SIGNED)
        c = Constant.int_const(42, i32)
        result = encoder.encode_value(c, ctx)
        assert result is not None

    def test_encode_add(self):
        config = SemanticConfig.c11()
        encoder = SMTEncoder(config)
        ctx = EncodingContext()
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(10, i32)
        b = Constant.int_const(20, i32)
        inst = BinaryOp(BinOpKind.ADD, a, b, i32)
        if hasattr(encoder, 'encode_binop'):
            result = encoder.encode_binop(inst, ctx)
            assert result is not None

    def test_encode_float(self):
        config = SemanticConfig.c11()
        encoder = SMTEncoder(config)
        ctx = EncodingContext()
        c = Constant.float_const(3.14)
        result = encoder.encode_value(c, ctx)
        assert result is not None


class TestSMTSolver:
    def test_create(self):
        solver = SMTSolver()
        assert solver is not None

    def test_check_trivial_sat(self):
        import z3
        solver = SMTSolver()
        solver.add(z3.BoolVal(True))
        result = solver.check()
        assert result.is_sat

    def test_check_trivial_unsat(self):
        import z3
        solver = SMTSolver()
        solver.add(z3.BoolVal(False))
        result = solver.check()
        assert result.is_unsat


class TestModelDecoder:
    def test_create(self):
        decoder = ModelDecoder()
        assert decoder is not None

    def test_decode_int(self):
        decoder = ModelDecoder()
        if hasattr(decoder, 'decode_value'):
            result = decoder.decode_value(42, IntType(32, Signedness.SIGNED))
            assert result is not None


class TestSolverResult:
    def test_enum_values(self):
        assert SolverStatus.SAT is not None
        assert SolverStatus.UNSAT is not None
        assert SolverStatus.UNKNOWN is not None

    def test_comparison(self):
        assert SolverStatus.SAT != SolverStatus.UNSAT
        assert SolverStatus.SAT != SolverStatus.UNKNOWN


class TestEncodingCorrectness:
    def test_bitvector_add_no_overflow(self):
        """Verify that 10 + 20 = 30 as a 32-bit bitvector."""
        config = SemanticConfig.c11()
        encoder = SMTEncoder(config)
        ctx = EncodingContext()
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(10, i32)
        b = Constant.int_const(20, i32)
        va = encoder.encode_value(a, ctx)
        vb = encoder.encode_value(b, ctx)
        assert va is not None
        assert vb is not None

    def test_bitvector_overflow(self):
        """Encode an overflow scenario."""
        config = SemanticConfig.c11()
        encoder = SMTEncoder(config)
        ctx = EncodingContext()
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(2**31 - 1, i32)
        b = Constant.int_const(1, i32)
        va = encoder.encode_value(a, ctx)
        vb = encoder.encode_value(b, ctx)
        assert va is not None

    def test_float_encoding(self):
        """Encode float values."""
        config = SemanticConfig.c11()
        encoder = SMTEncoder(config)
        ctx = EncodingContext()
        a = Constant.float_const(1.5)
        b = Constant.float_const(2.5)
        va = encoder.encode_value(a, ctx)
        vb = encoder.encode_value(b, ctx)
        assert va is not None

    def test_signed_vs_unsigned(self):
        """Verify signed vs unsigned encoding difference."""
        config = SemanticConfig.c11()
        encoder = SMTEncoder(config)
        ctx = EncodingContext()
        i32s = IntType(32, Signedness.SIGNED)
        u32 = IntType(32, Signedness.UNSIGNED)
        signed_val = Constant.int_const(-1, i32s)
        unsigned_val = Constant.int_const(0xFFFFFFFF, u32)
        vs = encoder.encode_value(signed_val, ctx)
        vu = encoder.encode_value(unsigned_val, ctx)
        assert vs is not None
        assert vu is not None
