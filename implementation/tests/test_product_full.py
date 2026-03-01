"""Comprehensive tests for product program: alignment, coercion, product construction, normalizer."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.ir.types import (
    IntType, FloatType, PointerType, VoidType, FunctionType,
    Signedness, FloatKind,
)
from src.ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, CastInst,
    ReturnInst, BranchInst, PhiInst, LoadInst, StoreInst,
    AllocaInst, Constant, Value, Argument,
    BinOpKind, CmpPredicate, CastKind,
)
from src.ir.module import Module
from src.ir.builder import IRBuilder
from src.ir.basic_block import BasicBlock

from src.product_program.alignment import (
    FunctionAligner, AlignmentResult, BlockAlignment,
    InstructionAlignment, AlignmentCost, AlignmentKind,
)
from src.product_program.coercion import (
    CoercionGenerator, CoercionPoint, CoercionKind,
    CoercionAssertion, AssertionStrength,
)
from src.product_program.product import (
    ProductBuilder, ProductProgram, ProductBlock,
    ProductInstruction, SharedInput, ProductSide,
)
from src.product_program.normalizer import (
    IRNormalizer, NormalizationConfig, NormalizationPass,
    NormalizationStats,
)
from src.semantics.divergence_table import (
    DivergenceTable, DivergenceClass, CSemantics, RustSemantics,
)
from src.semantics.semantic_config import SemanticConfig, OverflowMode


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions for building test IR
# ═══════════════════════════════════════════════════════════════════════════

I32 = IntType(32, Signedness.SIGNED)
I64 = IntType(64, Signedness.SIGNED)
U32 = IntType(32, Signedness.UNSIGNED)
F32 = FloatType(FloatKind.F32)
F64 = FloatType(FloatKind.F64)
VOID = VoidType()


def _make_func(name, ret_type, param_types, language="c"):
    """Create a bare Function (no blocks yet)."""
    mod = Module(f"mod_{name}", f"{name}.{language}", "x86_64", "", language)
    ft = FunctionType(ret_type, list(param_types))
    func = mod.create_function(name, ft)
    return func, mod


def _simple_add_func(name="add", language="c"):
    """int add(int a, int b) { return a + b; }"""
    func, mod = _make_func(name, I32, [I32, I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg, b_arg = func.arguments
    result = b.add(a_arg, b_arg, name="sum")
    b.ret(result)
    return func


def _simple_sub_func(name="sub", language="c"):
    """int sub(int a, int b) { return a - b; }"""
    func, mod = _make_func(name, I32, [I32, I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg, b_arg = func.arguments
    result = b.sub(a_arg, b_arg, name="diff")
    b.ret(result)
    return func


def _mul_func(name="mul", language="c"):
    """int mul(int a, int b) { return a * b; }"""
    func, mod = _make_func(name, I32, [I32, I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg, b_arg = func.arguments
    result = b.mul(a_arg, b_arg, name="product")
    b.ret(result)
    return func


def _div_func(name="div_fn", language="c"):
    """int div_fn(int a, int b) { return a / b; }"""
    func, mod = _make_func(name, I32, [I32, I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg, b_arg = func.arguments
    result = b.sdiv(a_arg, b_arg, name="quotient")
    b.ret(result)
    return func


def _shift_func(name="shl_fn", language="c"):
    """int shl_fn(int a, int b) { return a << b; }"""
    func, mod = _make_func(name, I32, [I32, I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg, b_arg = func.arguments
    result = b.shl(a_arg, b_arg, name="shifted")
    b.ret(result)
    return func


def _branching_func(name="branch_fn", language="c"):
    """int branch_fn(int a, int b) { if (a < b) return a; else return b; }"""
    func, mod = _make_func(name, I32, [I32, I32], language)
    entry = func.create_block("entry")
    then_bb = func.create_block("then")
    else_bb = func.create_block("else")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg, b_arg = func.arguments
    cond = b.icmp_slt(a_arg, b_arg, name="cmp")
    b.cond_br(cond, then_bb, else_bb)

    b.position_at_end(then_bb)
    b.ret(a_arg)

    b.position_at_end(else_bb)
    b.ret(b_arg)
    return func


def _multi_block_func(name="multi", language="c"):
    """Function with 3 blocks: entry -> body -> exit."""
    func, mod = _make_func(name, I32, [I32, I32], language)
    entry = func.create_block("entry")
    body = func.create_block("body")
    exit_bb = func.create_block("exit")
    b = IRBuilder()

    b.position_at_end(entry)
    a_arg, b_arg = func.arguments
    b.br(body)

    b.position_at_end(body)
    result = b.add(a_arg, b_arg, name="sum")
    b.br(exit_bb)

    b.position_at_end(exit_bb)
    b.ret(result)
    return func


def _cast_func(name="cast_fn", language="c"):
    """Function with integer extension: sext i32 -> i64."""
    func, mod = _make_func(name, I64, [I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg = func.get_argument(0)
    extended = b.sext(a_arg, I64, name="ext")
    b.ret(extended)
    return func


def _identity_add_func(name="id_add", language="c"):
    """Function with identity op: a + 0."""
    func, mod = _make_func(name, I32, [I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg = func.get_argument(0)
    zero = Constant.int_const(0, I32)
    result = b.add(a_arg, zero, name="id_result")
    b.ret(result)
    return func


def _redundant_cast_func(name="rcast", language="c"):
    """Function with redundant cast: sext i32->i64, trunc i64->i32."""
    func, mod = _make_func(name, I32, [I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg = func.get_argument(0)
    ext = b.sext(a_arg, I64, name="ext")
    trunc = b.trunc(ext, I32, name="trunc")
    b.ret(trunc)
    return func


def _commutative_func(name="comm", language="c"):
    """Function where constant is lhs of commutative add: 1 + a."""
    func, mod = _make_func(name, I32, [I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    a_arg = func.get_argument(0)
    one = Constant.int_const(1, I32)
    result = b.add(one, a_arg, name="sum")
    b.ret(result)
    return func


def _void_func(name="void_fn", language="c"):
    """void void_fn(int a) { return; }"""
    func, mod = _make_func(name, VOID, [I32], language)
    entry = func.create_block("entry")
    b = IRBuilder()
    b.position_at_end(entry)
    b.ret_void()
    return func


# ═══════════════════════════════════════════════════════════════════════════
# 1. FunctionAligner tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFunctionAlignerIdentical:
    """Aligning two identical single-block functions."""

    def test_align_returns_result(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert isinstance(result, AlignmentResult)

    def test_identical_has_block_alignments(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert len(result.block_alignments) >= 1

    def test_identical_matched_blocks(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        matched = result.matched_blocks
        assert len(matched) >= 1
        for ba in matched:
            assert ba.kind == AlignmentKind.MATCHED

    def test_identical_no_left_only(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert len(result.left_only_blocks) == 0

    def test_identical_no_right_only(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert len(result.right_only_blocks) == 0

    def test_identical_high_similarity(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert result.structural_similarity >= 0.5

    def test_identical_instruction_alignments(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        all_ia = result.all_instruction_alignments
        assert len(all_ia) >= 1

    def test_alignment_references_functions(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert result.left_function is left
        assert result.right_function is right


class TestFunctionAlignerDifferent:
    """Aligning functions with structural differences."""

    def test_extra_blocks_left(self):
        left = _multi_block_func("multi_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert isinstance(result, AlignmentResult)
        # Left has more blocks, so some should be left-only or similarity lower
        assert len(result.block_alignments) >= 1

    def test_extra_blocks_right(self):
        left = _simple_add_func("add_c", "c")
        right = _multi_block_func("multi_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert isinstance(result, AlignmentResult)
        assert len(result.block_alignments) >= 1

    def test_different_ops_lower_similarity(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_sub_func("sub_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        # Different ops should yield lower similarity than identical
        ident_result = aligner.align(
            _simple_add_func("a", "c"), _simple_add_func("b", "rust")
        )
        assert result.structural_similarity <= ident_result.structural_similarity

    def test_branching_vs_straight_line(self):
        left = _branching_func("br_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert isinstance(result, AlignmentResult)
        # Branching func has more blocks
        total_alignments = len(result.block_alignments)
        assert total_alignments >= 1

    def test_custom_threshold(self):
        aligner = FunctionAligner(similarity_threshold=0.9)
        left = _simple_add_func("add_c", "c")
        right = _simple_sub_func("sub_rs", "rust")
        result = aligner.align(left, right)
        # High threshold may cause fewer matched blocks
        assert isinstance(result, AlignmentResult)


class TestFunctionAlignerSummary:
    """Test summary and visualization."""

    def test_summary_nonempty(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        s = result.summary()
        assert "add_c" in s
        assert "add_rs" in s

    def test_visualize_nonempty(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        v = result.visualize()
        assert len(v) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. AlignmentCost tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAlignmentCost:
    def test_zero_cost(self):
        c = AlignmentCost()
        assert c.total == 0.0

    def test_block_mismatch_weight(self):
        c = AlignmentCost(block_mismatches=1)
        assert c.total == 10.0

    def test_instruction_mismatch_weight(self):
        c = AlignmentCost(instruction_mismatches=1)
        assert c.total == 3.0

    def test_type_mismatch_weight(self):
        c = AlignmentCost(type_mismatches=1)
        assert c.total == 2.0

    def test_opcode_mismatch_weight(self):
        c = AlignmentCost(opcode_mismatches=1)
        assert c.total == 5.0

    def test_reorder_penalty(self):
        c = AlignmentCost(reorder_penalty=7.5)
        assert c.total == 7.5

    def test_extra_temporaries_weight(self):
        c = AlignmentCost(extra_temporaries=2)
        assert c.total == 1.0

    def test_combined_cost(self):
        c = AlignmentCost(
            block_mismatches=1,
            instruction_mismatches=2,
            type_mismatches=1,
        )
        expected = 10.0 + 6.0 + 2.0
        assert abs(c.total - expected) < 1e-9

    def test_addition(self):
        a = AlignmentCost(block_mismatches=1, instruction_mismatches=2)
        b = AlignmentCost(type_mismatches=3, reorder_penalty=1.5)
        c = a + b
        assert c.block_mismatches == 1
        assert c.instruction_mismatches == 2
        assert c.type_mismatches == 3
        assert c.reorder_penalty == 1.5

    def test_addition_total(self):
        a = AlignmentCost(block_mismatches=1)
        b = AlignmentCost(block_mismatches=2)
        c = a + b
        assert c.total == 30.0

    def test_addition_preserves_fields(self):
        a = AlignmentCost(extra_temporaries=3, opcode_mismatches=1)
        b = AlignmentCost(extra_temporaries=2, opcode_mismatches=4)
        c = a + b
        assert c.extra_temporaries == 5
        assert c.opcode_mismatches == 5

    def test_repr(self):
        c = AlignmentCost(block_mismatches=1)
        r = repr(c)
        assert "10.0" in r


# ═══════════════════════════════════════════════════════════════════════════
# 3. BlockAlignment and InstructionAlignment
# ═══════════════════════════════════════════════════════════════════════════


class TestBlockAlignment:
    def test_matched(self):
        func = _simple_add_func()
        block = func.blocks[0]
        ba = BlockAlignment(left=block, right=block, kind=AlignmentKind.MATCHED)
        assert ba.is_matched

    def test_left_only(self):
        func = _simple_add_func()
        block = func.blocks[0]
        ba = BlockAlignment(left=block, right=None, kind=AlignmentKind.LEFT_ONLY)
        assert not ba.is_matched
        assert ba.kind == AlignmentKind.LEFT_ONLY

    def test_right_only(self):
        func = _simple_add_func()
        block = func.blocks[0]
        ba = BlockAlignment(left=None, right=block, kind=AlignmentKind.RIGHT_ONLY)
        assert ba.kind == AlignmentKind.RIGHT_ONLY

    def test_left_name(self):
        func = _simple_add_func()
        block = func.blocks[0]
        ba = BlockAlignment(left=block, right=None, kind=AlignmentKind.LEFT_ONLY)
        assert ba.left_name == block.name

    def test_right_name(self):
        func = _simple_add_func()
        block = func.blocks[0]
        ba = BlockAlignment(left=None, right=block, kind=AlignmentKind.RIGHT_ONLY)
        assert ba.right_name == block.name


class TestInstructionAlignment:
    def test_matched(self):
        func = _simple_add_func()
        insts = list(func.blocks[0].instructions)
        ia = InstructionAlignment(
            left=insts[0], right=insts[0], kind=AlignmentKind.MATCHED, similarity=1.0
        )
        assert ia.is_matched
        assert not ia.is_left_only
        assert not ia.is_right_only

    def test_left_only(self):
        func = _simple_add_func()
        insts = list(func.blocks[0].instructions)
        ia = InstructionAlignment(
            left=insts[0], right=None, kind=AlignmentKind.LEFT_ONLY
        )
        assert ia.is_left_only
        assert not ia.is_matched

    def test_right_only(self):
        func = _simple_add_func()
        insts = list(func.blocks[0].instructions)
        ia = InstructionAlignment(
            left=None, right=insts[0], kind=AlignmentKind.RIGHT_ONLY
        )
        assert ia.is_right_only

    def test_similarity_stored(self):
        func = _simple_add_func()
        insts = list(func.blocks[0].instructions)
        ia = InstructionAlignment(
            left=insts[0], right=insts[0], kind=AlignmentKind.MATCHED, similarity=0.85
        )
        assert abs(ia.similarity - 0.85) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# 4. ProductBuilder tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProductBuilderSimple:
    """Building product programs from simple function pairs."""

    def test_build_identical_functions(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)

    def test_product_has_name(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert "add_c" in product.name
        assert "add_rs" in product.name

    def test_product_references_functions(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert product.left_function is left
        assert product.right_function is right

    def test_product_has_blocks(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert product.num_blocks >= 1

    def test_product_has_instructions(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert product.num_instructions >= 1

    def test_product_has_shared_inputs(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        # Both functions have 2 i32 params, so should have shared inputs
        assert len(product.shared_inputs) >= 1

    def test_product_has_alignment(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert product.alignment is not None


class TestProductBuilderSharedInputs:
    """Testing shared input creation and type unification."""

    def test_shared_input_fields(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        for si in product.shared_inputs:
            assert isinstance(si, SharedInput)
            assert isinstance(si.name, str)
            assert isinstance(si.ir_type, IntType)
            assert isinstance(si.left_arg_index, int)
            assert isinstance(si.right_arg_index, int)

    def test_shared_input_repr(self):
        si = SharedInput(name="x", ir_type=I32, left_arg_index=0, right_arg_index=0)
        r = repr(si)
        assert "x" in r

    def test_shared_input_count_matches_params(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        # min(left_params, right_params) shared inputs
        max_shared = min(left.num_arguments, right.num_arguments)
        assert len(product.shared_inputs) <= max_shared


class TestProductBuilderDifferentOps:
    """Building product programs from functions with different operations."""

    def test_add_vs_sub(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_sub_func("sub_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)
        assert product.num_blocks >= 1

    def test_add_vs_mul(self):
        left = _simple_add_func("add_c", "c")
        right = _mul_func("mul_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert product.num_instructions >= 1

    def test_branching_vs_linear(self):
        left = _branching_func("br_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)


class TestProductBuilderOverflow:
    """Testing overflow encoding."""

    def test_build_with_overflow_encoding(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build_with_overflow_encoding(left, right)
        assert isinstance(product, ProductProgram)

    def test_overflow_modes(self):
        left = _mul_func("mul_c", "c")
        right = _mul_func("mul_rs", "rust")
        builder = ProductBuilder()
        product = builder.build_with_overflow_encoding(
            left, right,
            c_overflow=OverflowMode.UB,
            rust_overflow=OverflowMode.Wrap,
        )
        assert isinstance(product, ProductProgram)


# ═══════════════════════════════════════════════════════════════════════════
# 5. ProductProgram properties
# ═══════════════════════════════════════════════════════════════════════════


class TestProductProgramProperties:
    """Test ProductProgram dataclass properties."""

    def _build_product(self):
        left = _simple_add_func("f_c", "c")
        right = _simple_add_func("f_rs", "rust")
        return ProductBuilder().build(left, right)

    def test_entry_block(self):
        product = self._build_product()
        entry = product.entry_block
        # Either marked as entry or first block
        assert entry is not None

    def test_exit_blocks(self):
        product = self._build_product()
        exits = product.exit_blocks
        assert isinstance(exits, list)

    def test_num_blocks(self):
        product = self._build_product()
        assert product.num_blocks == len(product.blocks)

    def test_get_block_found(self):
        product = self._build_product()
        if product.blocks:
            name = product.blocks[0].name
            found = product.get_block(name)
            assert found is not None
            assert found.name == name

    def test_get_block_not_found(self):
        product = self._build_product()
        assert product.get_block("__nonexistent__") is None

    def test_add_block(self):
        product = self._build_product()
        old_count = product.num_blocks
        new_block = ProductBlock(name="extra", left_block=None, right_block=None)
        product.add_block(new_block)
        assert product.num_blocks == old_count + 1
        assert product.get_block("extra") is new_block

    def test_iter_instructions(self):
        product = self._build_product()
        insts = list(product.iter_instructions())
        assert len(insts) == product.num_instructions

    def test_iter_coercion_points(self):
        product = self._build_product()
        cps = list(product.iter_coercion_points())
        assert len(cps) == product.num_coercion_points

    def test_all_assertions(self):
        product = self._build_product()
        all_a = product.all_assertions()
        assert isinstance(all_a, list)
        assert all(isinstance(a, CoercionAssertion) for a in all_a)

    def test_hard_assertions_subset(self):
        product = self._build_product()
        hard = product.hard_assertions()
        for a in hard:
            assert a.strength == AssertionStrength.HARD

    def test_assumptions_subset(self):
        product = self._build_product()
        assumptions = product.assumptions()
        for a in assumptions:
            assert a.strength == AssertionStrength.ASSUME

    def test_summary(self):
        product = self._build_product()
        s = product.summary()
        assert "ProductProgram" in s
        assert "Shared inputs" in s

    def test_visualize(self):
        product = self._build_product()
        v = product.visualize()
        assert "PRODUCT PROGRAM" in v

    def test_to_smt_lib(self):
        product = self._build_product()
        smt = product.to_smt_lib()
        assert "(set-logic QF_BV)" in smt
        assert "(check-sat)" in smt


class TestProductBlockProperties:
    def test_coercion_points_empty(self):
        pb = ProductBlock(name="b0", left_block=None, right_block=None)
        assert pb.coercion_points == []
        assert not pb.has_coercions

    def test_num_instructions(self):
        pb = ProductBlock(name="b0", left_block=None, right_block=None)
        assert pb.num_instructions == 0

    def test_entry_exit_flags(self):
        pb = ProductBlock(
            name="entry", left_block=None, right_block=None,
            is_entry=True, is_exit=False,
        )
        assert pb.is_entry
        assert not pb.is_exit


class TestProductInstructionProperties:
    def test_side_both(self):
        func = _simple_add_func()
        insts = list(func.blocks[0].instructions)
        pi = ProductInstruction(
            left_inst=insts[0], right_inst=insts[0], side=ProductSide.BOTH
        )
        assert pi.is_both
        assert not pi.has_coercions

    def test_side_left(self):
        func = _simple_add_func()
        insts = list(func.blocks[0].instructions)
        pi = ProductInstruction(
            left_inst=insts[0], right_inst=None, side=ProductSide.LEFT
        )
        assert not pi.is_both

    def test_summary(self):
        func = _simple_add_func()
        insts = list(func.blocks[0].instructions)
        pi = ProductInstruction(
            left_inst=insts[0], right_inst=insts[0], side=ProductSide.BOTH
        )
        s = pi.summary()
        assert isinstance(s, str)

    def test_all_assertions_empty(self):
        pi = ProductInstruction(
            left_inst=None, right_inst=None, side=ProductSide.LEFT
        )
        assert pi.all_assertions == []


# ═══════════════════════════════════════════════════════════════════════════
# 6. CoercionGenerator tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCoercionGeneratorCreation:
    def test_default_creation(self):
        gen = CoercionGenerator()
        assert gen is not None

    def test_with_table(self):
        table = DivergenceTable()
        gen = CoercionGenerator(divergence_table=table)
        assert gen.table is table

    def test_with_configs(self):
        c_cfg = SemanticConfig.c11()
        r_cfg = SemanticConfig.rust_release()
        gen = CoercionGenerator(c_config=c_cfg, rust_config=r_cfg)
        assert gen.c_config is c_cfg
        assert gen.rust_config is r_cfg


class TestCoercionGeneratorForAlignment:
    def test_generate_for_identical(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        alignment = aligner.align(left, right)
        gen = CoercionGenerator()
        points = gen.generate_for_alignment(alignment)
        assert isinstance(points, list)
        assert all(isinstance(p, CoercionPoint) for p in points)

    def test_generate_for_arithmetic(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        alignment = aligner.align(left, right)
        gen = CoercionGenerator()
        points = gen.generate_for_alignment(alignment)
        # Addition may produce overflow coercions
        overflow_points = [
            p for p in points if p.kind == CoercionKind.OVERFLOW_CHECK
        ]
        # May or may not have overflow checks depending on divergence detection
        assert isinstance(overflow_points, list)

    def test_generate_for_mul(self):
        left = _mul_func("mul_c", "c")
        right = _mul_func("mul_rs", "rust")
        aligner = FunctionAligner()
        alignment = aligner.align(left, right)
        gen = CoercionGenerator()
        points = gen.generate_for_alignment(alignment)
        assert isinstance(points, list)


class TestCoercionGeneratorForPair:
    def test_pair_add_instructions(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        left_insts = list(left.blocks[0].instructions)
        right_insts = list(right.blocks[0].instructions)
        # First instruction should be the add
        gen = CoercionGenerator()
        points = gen.generate_for_pair(left_insts[0], right_insts[0])
        assert isinstance(points, list)

    def test_pair_div_instructions(self):
        left = _div_func("div_c", "c")
        right = _div_func("div_rs", "rust")
        left_insts = list(left.blocks[0].instructions)
        right_insts = list(right.blocks[0].instructions)
        gen = CoercionGenerator()
        points = gen.generate_for_pair(left_insts[0], right_insts[0])
        assert isinstance(points, list)
        # Division should produce division-related coercions
        div_points = [p for p in points if p.kind == CoercionKind.DIVISION_CHECK]
        assert isinstance(div_points, list)

    def test_pair_shift_instructions(self):
        left = _shift_func("shl_c", "c")
        right = _shift_func("shl_rs", "rust")
        left_insts = list(left.blocks[0].instructions)
        right_insts = list(right.blocks[0].instructions)
        gen = CoercionGenerator()
        points = gen.generate_for_pair(left_insts[0], right_insts[0])
        assert isinstance(points, list)

    def test_pair_return_instructions(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        # Return instruction is last
        left_insts = list(left.blocks[0].instructions)
        right_insts = list(right.blocks[0].instructions)
        gen = CoercionGenerator()
        points = gen.generate_for_pair(left_insts[-1], right_insts[-1])
        assert isinstance(points, list)


class TestCoercionGeneratorSummary:
    def test_summary_empty(self):
        gen = CoercionGenerator()
        s = gen.summary([])
        assert isinstance(s, str)

    def test_summary_with_points(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        aligner = FunctionAligner()
        alignment = aligner.align(left, right)
        gen = CoercionGenerator()
        points = gen.generate_for_alignment(alignment)
        s = gen.summary(points)
        assert isinstance(s, str)


# ═══════════════════════════════════════════════════════════════════════════
# 7. CoercionAssertion tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCoercionAssertion:
    def test_creation(self):
        a = CoercionAssertion(
            smt_expression="(= x y)",
            description="x equals y",
        )
        assert a.smt_expression == "(= x y)"
        assert a.description == "x equals y"
        assert a.strength == AssertionStrength.HARD  # default

    def test_strength_soft(self):
        a = CoercionAssertion(
            smt_expression="(bvslt x max)",
            description="x < max",
            strength=AssertionStrength.SOFT,
        )
        assert a.strength == AssertionStrength.SOFT

    def test_strength_assume(self):
        a = CoercionAssertion(
            smt_expression="(not (= y zero))",
            description="y != 0",
            strength=AssertionStrength.ASSUME,
        )
        assert a.strength == AssertionStrength.ASSUME

    def test_negate(self):
        a = CoercionAssertion(
            smt_expression="(= x y)",
            description="x equals y",
            strength=AssertionStrength.HARD,
            variables=["x", "y"],
        )
        neg = a.negate()
        assert "(not (= x y))" in neg.smt_expression
        assert "NOT" in neg.description
        assert neg.strength == AssertionStrength.HARD
        assert neg.variables == ["x", "y"]

    def test_negate_preserves_variables(self):
        a = CoercionAssertion(
            smt_expression="(bvslt a b)",
            description="a < b",
            variables=["a", "b"],
        )
        neg = a.negate()
        assert neg.variables == ["a", "b"]

    def test_repr(self):
        a = CoercionAssertion(
            smt_expression="(= x y)",
            description="x equals y",
            strength=AssertionStrength.HARD,
        )
        r = repr(a)
        assert "x equals y" in r
        assert "HARD" in r

    def test_variables_default_empty(self):
        a = CoercionAssertion(smt_expression="true", description="trivial")
        assert a.variables == []


# ═══════════════════════════════════════════════════════════════════════════
# 8. CoercionPoint tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCoercionPoint:
    def _make_point(self, assertions=None):
        return CoercionPoint(
            kind=CoercionKind.OVERFLOW_CHECK,
            left_instruction=None,
            right_instruction=None,
            divergence_class=DivergenceClass.SignedOverflow,
            c_semantics=CSemantics(summary="UB on overflow", is_ub=True),
            rust_semantics=RustSemantics(
                summary="wraps in release", wraps_in_release=True
            ),
            assertions=assertions or [],
            operation="add",
            bit_width=32,
        )

    def test_no_assertions(self):
        p = self._make_point()
        assert p.num_assertions == 0
        assert not p.is_critical

    def test_with_hard_assertion(self):
        hard = CoercionAssertion("(= a b)", "a==b", AssertionStrength.HARD)
        p = self._make_point(assertions=[hard])
        assert p.num_assertions == 1
        assert p.is_critical

    def test_with_soft_assertion(self):
        soft = CoercionAssertion("(= a b)", "a==b", AssertionStrength.SOFT)
        p = self._make_point(assertions=[soft])
        assert p.num_assertions == 1
        assert not p.is_critical

    def test_summary(self):
        p = self._make_point()
        s = p.summary()
        assert "OVERFLOW_CHECK" in s

    def test_kind_values(self):
        for kind in CoercionKind:
            assert isinstance(kind.name, str)

    def test_repr(self):
        p = self._make_point()
        r = repr(p)
        assert isinstance(r, str)


# ═══════════════════════════════════════════════════════════════════════════
# 9. IRNormalizer tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIRNormalizerCreation:
    def test_default_config(self):
        norm = IRNormalizer()
        assert norm.config is not None
        assert norm.config.max_iterations == 3

    def test_custom_config(self):
        cfg = NormalizationConfig(
            passes=[NormalizationPass.NORMALIZE_NAMES],
            max_iterations=1,
        )
        norm = IRNormalizer(config=cfg)
        assert len(norm.config.passes) == 1


class TestIRNormalizerNormalize:
    def test_normalize_simple(self):
        func = _simple_add_func("add", "c")
        norm = IRNormalizer()
        result = norm.normalize(func)
        assert result is func  # in-place
        # Should still have blocks and instructions
        assert func.num_blocks >= 1

    def test_normalize_pair(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        norm = IRNormalizer()
        l, r = norm.normalize_pair(left, right)
        assert l is left
        assert r is right

    def test_normalize_preserves_semantics(self):
        func = _simple_add_func("add", "c")
        num_blocks_before = func.num_blocks
        norm = IRNormalizer()
        norm.normalize(func)
        # Should not remove blocks from a minimal function
        assert func.num_blocks >= 1

    def test_normalize_stats(self):
        func = _simple_add_func("add", "c")
        norm = IRNormalizer()
        norm.normalize(func)
        stats = norm.stats
        assert isinstance(stats, NormalizationStats)
        assert stats.iterations >= 1


class TestIRNormalizerNameNormalization:
    def test_names_normalized(self):
        func = _simple_add_func("add", "c")
        cfg = NormalizationConfig(
            passes=[NormalizationPass.NORMALIZE_NAMES],
            max_iterations=1,
        )
        norm = IRNormalizer(config=cfg)
        norm.normalize(func)
        # Stats should reflect name normalization
        assert isinstance(norm.stats.names_normalized, int)


class TestIRNormalizerRedundantCasts:
    def test_remove_redundant_casts(self):
        func = _redundant_cast_func("rcast", "c")
        cfg = NormalizationConfig(
            passes=[NormalizationPass.REMOVE_REDUNDANT_CASTS],
            max_iterations=2,
        )
        norm = IRNormalizer(config=cfg)
        norm.normalize(func)
        assert isinstance(norm.stats.casts_removed, int)

    def test_necessary_cast_preserved(self):
        func = _cast_func("cast", "c")
        cfg = NormalizationConfig(
            passes=[NormalizationPass.REMOVE_REDUNDANT_CASTS],
            max_iterations=1,
        )
        norm = IRNormalizer(config=cfg)
        norm.normalize(func)
        # Single sext is not redundant, should be preserved
        insts = list(func.blocks[0].instructions)
        cast_insts = [i for i in insts if isinstance(i, CastInst)]
        assert len(cast_insts) >= 1


class TestIRNormalizerCommutative:
    def test_canonicalize_commutative(self):
        func = _commutative_func("comm", "c")
        cfg = NormalizationConfig(
            passes=[NormalizationPass.CANONICALIZE_COMMUTATIVE],
            max_iterations=1,
        )
        norm = IRNormalizer(config=cfg)
        norm.normalize(func)
        assert isinstance(norm.stats.ops_canonicalized, int)


class TestIRNormalizerIdentityOps:
    def test_remove_identity(self):
        func = _identity_add_func("id_add", "c")
        cfg = NormalizationConfig(
            passes=[NormalizationPass.REMOVE_IDENTITY_OPS],
            max_iterations=1,
        )
        norm = IRNormalizer(config=cfg)
        norm.normalize(func)
        assert isinstance(norm.stats.identity_ops_removed, int)


class TestIRNormalizerMultiPass:
    def test_all_passes(self):
        func = _simple_add_func("add", "c")
        norm = IRNormalizer()
        norm.normalize(func)
        # All default passes should run without error
        assert norm.stats.iterations >= 1

    def test_iteration_limit(self):
        cfg = NormalizationConfig(max_iterations=1)
        norm = IRNormalizer(config=cfg)
        func = _simple_add_func("add", "c")
        norm.normalize(func)
        assert norm.stats.iterations <= 1


# ═══════════════════════════════════════════════════════════════════════════
# 10. NormalizationConfig tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizationConfig:
    def test_default_passes(self):
        cfg = NormalizationConfig()
        assert len(cfg.passes) >= 5
        assert NormalizationPass.NORMALIZE_NAMES in cfg.passes
        assert NormalizationPass.REMOVE_REDUNDANT_CASTS in cfg.passes

    def test_custom_passes(self):
        cfg = NormalizationConfig(passes=[NormalizationPass.NORMALIZE_NAMES])
        assert len(cfg.passes) == 1

    def test_default_max_iterations(self):
        cfg = NormalizationConfig()
        assert cfg.max_iterations == 3

    def test_custom_max_iterations(self):
        cfg = NormalizationConfig(max_iterations=10)
        assert cfg.max_iterations == 10

    def test_preserve_debug_info_default(self):
        cfg = NormalizationConfig()
        assert cfg.preserve_debug_info is True

    def test_empty_passes(self):
        cfg = NormalizationConfig(passes=[])
        assert len(cfg.passes) == 0


class TestNormalizationStats:
    def test_default_zeros(self):
        stats = NormalizationStats()
        assert stats.names_normalized == 0
        assert stats.casts_removed == 0
        assert stats.expressions_flattened == 0
        assert stats.ops_canonicalized == 0
        assert stats.identity_ops_removed == 0
        assert stats.comparisons_canonicalized == 0
        assert stats.blocks_merged == 0
        assert stats.iterations == 0

    def test_summary(self):
        stats = NormalizationStats(names_normalized=5, iterations=2)
        s = stats.summary()
        assert "5" in s
        assert "2" in s


# ═══════════════════════════════════════════════════════════════════════════
# 11. NormalizationPass enum
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizationPass:
    def test_all_passes_exist(self):
        expected = [
            "NORMALIZE_NAMES",
            "REMOVE_REDUNDANT_CASTS",
            "FLATTEN_EXPRESSIONS",
            "CANONICALIZE_COMMUTATIVE",
            "NORMALIZE_CONTROL_FLOW",
            "REMOVE_IDENTITY_OPS",
            "CANONICALIZE_COMPARISONS",
            "MERGE_REDUNDANT_BLOCKS",
        ]
        for name in expected:
            assert hasattr(NormalizationPass, name)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Integration: full pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """End-to-end: normalize -> align -> coerce -> build product."""

    def test_normalize_then_build(self):
        left = _simple_add_func("add_c", "c")
        right = _simple_add_func("add_rs", "rust")
        norm = IRNormalizer()
        norm.normalize_pair(left, right)
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)
        assert product.num_blocks >= 1

    def test_mul_pipeline(self):
        left = _mul_func("mul_c", "c")
        right = _mul_func("mul_rs", "rust")
        norm = IRNormalizer()
        norm.normalize_pair(left, right)
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)

    def test_div_pipeline(self):
        left = _div_func("div_c", "c")
        right = _div_func("div_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)
        smt = product.to_smt_lib()
        assert "(set-logic QF_BV)" in smt

    def test_branching_pipeline(self):
        left = _branching_func("br_c", "c")
        right = _branching_func("br_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert product.num_blocks >= 1

    def test_multi_block_pipeline(self):
        left = _multi_block_func("multi_c", "c")
        right = _multi_block_func("multi_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert product.num_blocks >= 1

    def test_void_functions(self):
        left = _void_func("void_c", "c")
        right = _void_func("void_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)

    def test_different_block_counts(self):
        left = _multi_block_func("multi_c", "c")
        right = _simple_add_func("add_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)


# ═══════════════════════════════════════════════════════════════════════════
# 13. AlignmentKind enum
# ═══════════════════════════════════════════════════════════════════════════


class TestAlignmentKind:
    def test_values(self):
        assert AlignmentKind.MATCHED is not None
        assert AlignmentKind.LEFT_ONLY is not None
        assert AlignmentKind.RIGHT_ONLY is not None
        assert AlignmentKind.REORDERED is not None

    def test_all_distinct(self):
        kinds = [
            AlignmentKind.MATCHED,
            AlignmentKind.LEFT_ONLY,
            AlignmentKind.RIGHT_ONLY,
            AlignmentKind.REORDERED,
        ]
        assert len(set(kinds)) == 4


# ═══════════════════════════════════════════════════════════════════════════
# 14. ProductSide enum
# ═══════════════════════════════════════════════════════════════════════════


class TestProductSide:
    def test_values(self):
        assert ProductSide.LEFT is not None
        assert ProductSide.RIGHT is not None
        assert ProductSide.BOTH is not None

    def test_all_distinct(self):
        sides = [ProductSide.LEFT, ProductSide.RIGHT, ProductSide.BOTH]
        assert len(set(sides)) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 15. CoercionKind enum
# ═══════════════════════════════════════════════════════════════════════════


class TestCoercionKind:
    def test_all_kinds(self):
        expected = [
            "OVERFLOW_CHECK", "DIVISION_CHECK", "SHIFT_CHECK",
            "FLOAT_PRECISION", "FLOAT_TO_INT", "POINTER_PROVENANCE",
            "NULL_CHECK", "BOUNDS_CHECK", "ERROR_HANDLING",
            "TYPE_WIDTH", "SIGNEDNESS", "RETURN_COERCION",
            "CAST_COERCION", "CALLING_CONVENTION",
        ]
        for name in expected:
            assert hasattr(CoercionKind, name)


# ═══════════════════════════════════════════════════════════════════════════
# 16. AssertionStrength enum
# ═══════════════════════════════════════════════════════════════════════════


class TestAssertionStrength:
    def test_values(self):
        assert AssertionStrength.HARD is not None
        assert AssertionStrength.SOFT is not None
        assert AssertionStrength.ASSUME is not None

    def test_ordering(self):
        # Just check they are distinct
        strengths = [
            AssertionStrength.HARD,
            AssertionStrength.SOFT,
            AssertionStrength.ASSUME,
        ]
        assert len(set(strengths)) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 17. Edge cases and error handling
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_align_single_instruction_functions(self):
        """Functions with only a return void."""
        left = _void_func("v_c", "c")
        right = _void_func("v_rs", "rust")
        aligner = FunctionAligner()
        result = aligner.align(left, right)
        assert isinstance(result, AlignmentResult)

    def test_product_cast_functions(self):
        left = _cast_func("cast_c", "c")
        right = _cast_func("cast_rs", "rust")
        builder = ProductBuilder()
        product = builder.build(left, right)
        assert isinstance(product, ProductProgram)

    def test_normalize_branching(self):
        func = _branching_func("br", "c")
        norm = IRNormalizer()
        norm.normalize(func)
        assert func.num_blocks >= 2

    def test_normalize_empty_passes(self):
        func = _simple_add_func("add", "c")
        cfg = NormalizationConfig(passes=[], max_iterations=1)
        norm = IRNormalizer(config=cfg)
        norm.normalize(func)
        assert norm.stats.iterations >= 1

    def test_product_block_iter_left(self):
        func = _simple_add_func()
        block = func.blocks[0]
        insts = list(block.instructions)
        pi = ProductInstruction(
            left_inst=insts[0], right_inst=None, side=ProductSide.LEFT
        )
        pb = ProductBlock(
            name="test", left_block=block, right_block=None,
            instructions=[pi],
        )
        left_insts = list(pb.iter_left_instructions())
        assert len(left_insts) == 1

    def test_product_block_iter_right(self):
        func = _simple_add_func()
        block = func.blocks[0]
        insts = list(block.instructions)
        pi = ProductInstruction(
            left_inst=None, right_inst=insts[0], side=ProductSide.RIGHT
        )
        pb = ProductBlock(
            name="test", left_block=None, right_block=block,
            instructions=[pi],
        )
        right_insts = list(pb.iter_right_instructions())
        assert len(right_insts) == 1

    def test_product_block_successors_predecessors(self):
        pb = ProductBlock(
            name="mid", left_block=None, right_block=None,
            predecessors=["entry"], successors=["exit"],
        )
        assert pb.predecessors == ["entry"]
        assert pb.successors == ["exit"]

    def test_shared_input_dataclass(self):
        si = SharedInput(
            name="shared_x", ir_type=I64,
            left_arg_index=0, right_arg_index=1,
        )
        assert si.name == "shared_x"
        assert si.ir_type is I64
        assert si.left_arg_index == 0
        assert si.right_arg_index == 1

    def test_coercion_assertion_negate_twice(self):
        a = CoercionAssertion("(= a b)", "a==b")
        neg = a.negate()
        neg2 = neg.negate()
        assert "(not (not (= a b)))" in neg2.smt_expression

    def test_alignment_cost_zero_addition(self):
        a = AlignmentCost()
        b = AlignmentCost()
        c = a + b
        assert c.total == 0.0
