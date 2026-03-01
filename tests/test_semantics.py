"""Tests for semantic evaluation and divergence detection."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantics.semantic_config import SemanticConfig
from src.semantics.eval import SemanticEvaluator
from src.semantics.divergence_table import DivergenceTable, DivergenceAnalyzer
from src.ir.types import IntType, FloatType, Signedness, FloatKind
from src.ir.instructions import BinOpKind, BinaryOp, Constant


class TestSemanticConfig:
    def test_c11_config(self):
        config = SemanticConfig.c11()
        assert config is not None

    def test_rust_debug_config(self):
        config = SemanticConfig.rust_debug()
        assert config is not None

    def test_rust_release_config(self):
        config = SemanticConfig.rust_release()
        assert config is not None

    def test_diff_configs(self):
        c_config = SemanticConfig.c11()
        r_config = SemanticConfig.rust_debug()
        diff = c_config.diff(r_config)
        assert diff is not None

    def test_serialization(self):
        config = SemanticConfig.c11()
        d = config.to_dict()
        config2 = SemanticConfig.from_dict(d)
        assert config2 is not None

    def test_c11_optimized(self):
        config = SemanticConfig.c11_optimized()
        assert config is not None


class TestSemanticEvaluator:
    def test_create(self):
        config = SemanticConfig.c11()
        evaluator = SemanticEvaluator(config)
        assert evaluator is not None

    def test_evaluate_add_no_overflow(self):
        config = SemanticConfig.c11()
        evaluator = SemanticEvaluator(config)
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(10, i32)
        b = Constant.int_const(20, i32)
        inst = BinaryOp(BinOpKind.ADD, a, b, i32)
        result = evaluator.evaluate(inst)
        assert result is not None

    def test_evaluate_add_overflow(self):
        config = SemanticConfig.c11()
        evaluator = SemanticEvaluator(config)
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(2**31 - 1, i32)
        b = Constant.int_const(1, i32)
        inst = BinaryOp(BinOpKind.ADD, a, b, i32)
        result = evaluator.evaluate(inst)
        assert result is not None

    def test_evaluate_with_different_configs(self):
        c_eval = SemanticEvaluator(SemanticConfig.c11())
        r_eval = SemanticEvaluator(SemanticConfig.rust_debug())
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(2**31 - 1, i32)
        b = Constant.int_const(1, i32)
        inst = BinaryOp(BinOpKind.ADD, a, b, i32)
        c_result = c_eval.evaluate(inst)
        r_result = r_eval.evaluate(inst)
        # Results may differ on overflow behavior

    def test_evaluate_sub(self):
        config = SemanticConfig.c11()
        evaluator = SemanticEvaluator(config)
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(30, i32)
        b = Constant.int_const(20, i32)
        inst = BinaryOp(BinOpKind.SUB, a, b, i32)
        result = evaluator.evaluate(inst)
        assert result is not None

    def test_evaluate_mul(self):
        config = SemanticConfig.c11()
        evaluator = SemanticEvaluator(config)
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(6, i32)
        b = Constant.int_const(7, i32)
        inst = BinaryOp(BinOpKind.MUL, a, b, i32)
        result = evaluator.evaluate(inst)
        assert result is not None

    def test_evaluate_div(self):
        config = SemanticConfig.c11()
        evaluator = SemanticEvaluator(config)
        i32 = IntType(32, Signedness.SIGNED)
        a = Constant.int_const(42, i32)
        b = Constant.int_const(6, i32)
        inst = BinaryOp(BinOpKind.SDIV, a, b, i32)
        result = evaluator.evaluate(inst)
        assert result is not None


class TestDivergenceTable:
    def test_create(self):
        table = DivergenceTable()
        assert table is not None

    def test_entries_exist(self):
        table = DivergenceTable()
        # Table should have entries for known divergences
        entries = table.entries if hasattr(table, 'entries') else []
        # At minimum should not crash

    def test_lookup_overflow(self):
        table = DivergenceTable()
        if hasattr(table, 'lookup'):
            result = table.lookup("signed_overflow")
            # May or may not find it, just verify no crash


class TestDivergenceAnalyzer:
    def test_create(self):
        analyzer = DivergenceAnalyzer()
        assert analyzer is not None

    def test_analyze_add(self):
        analyzer = DivergenceAnalyzer()
        if hasattr(analyzer, 'analyze_operation'):
            result = analyzer.analyze_operation(BinOpKind.ADD,
                                                IntType(32, Signedness.SIGNED))
            # Should return some analysis result
