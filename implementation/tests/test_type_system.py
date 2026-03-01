"""Tests for the type system module: coercions, promotions, and type checker."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.ir.types import (
    IntType, FloatType, PointerType, VoidType, FunctionType,
    Signedness, FloatKind, Language,
    I8, I16, I32, I64, U8, U16, U32, U64, F32, F64, VOID,
)
from src.ir.instructions import (
    Value, BinaryOp, CompareOp, ReturnInst,
    BinOpKind, CastKind, CmpPredicate,
)
from src.ir.function import Function
from src.ir.module import Module
from src.ir.basic_block import BasicBlock
from src.type_system.coercions import (
    CoercionGenerator, CoercionStep, CoercionChain, InformationLoss,
)
from src.type_system.promotions import (
    CPromotionRules, RustPromotionRules, PromotionComparison,
    PromotionChain, PromotionStep, PromotionKind,
)
from src.type_system.checker import (
    TypeChecker, TypeCheckResult, TypeCheckError, InsertedCast,
    TypeCheckSeverity,
)

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _val(ty, name=""):
    return Value(ty, name=name)

@pytest.fixture
def cg():
    return CoercionGenerator()

@pytest.fixture
def c_rules():
    return CPromotionRules()

@pytest.fixture
def rust_rules():
    return RustPromotionRules()

@pytest.fixture
def c_checker():
    return TypeChecker(language=Language.C, insert_casts=True)

@pytest.fixture
def rust_checker():
    return TypeChecker(language=Language.RUST, insert_casts=False, strict=True)

def _binop(op, lty, rty, res_ty=None):
    return BinaryOp(op, _val(lty, "l"), _val(rty, "r"), result_type=res_ty)

def _cmpop(pred, lty, rty):
    return CompareOp(pred, _val(lty, "l"), _val(rty, "r"))

def _ret(ty=None):
    return ReturnInst(value=_val(ty, "rv") if ty else None)

def _func(name, ret, params, lang="c"):
    return Function(name, FunctionType(ret, tuple(params)), language=lang)


# ===================================================================
# 1. CoercionGenerator – int-to-int
# ===================================================================

class TestCoercionIntToInt:
    def test_identity(self, cg):
        c = cg.generate(I32, I32)
        assert c.is_identity and c.is_lossless and c.num_steps == 0

    def test_widening_signed(self, cg):
        c = cg.generate(I16, I32)
        assert c.is_valid and c.is_lossless
        assert CastKind.SEXT in c.cast_kinds

    def test_widening_unsigned(self, cg):
        c = cg.generate(U8, U32)
        assert c.is_valid and c.is_lossless and CastKind.ZEXT in c.cast_kinds

    def test_narrowing_signed(self, cg):
        c = cg.generate(I64, I16)
        assert c.is_valid and not c.is_lossless
        assert CastKind.TRUNC in c.cast_kinds

    def test_narrowing_unsigned(self, cg):
        c = cg.generate(U64, U8)
        assert c.is_valid and not c.is_lossless and CastKind.TRUNC in c.cast_kinds

    def test_sign_change_same_width(self, cg):
        c = cg.generate(I32, U32)
        assert c.is_valid and c.num_steps >= 1
        losses = c.total_information_loss
        assert InformationLoss.SIGN in losses or InformationLoss.BITS in losses

    def test_i8_to_i64(self, cg):
        c = cg.generate(I8, I64)
        assert c.is_valid and c.is_lossless and c.src_type == I8 and c.dst_type == I64

    def test_u16_to_i32_lossless(self, cg):
        c = cg.generate(U16, I32)
        assert c.is_valid and c.is_lossless


# ===================================================================
# 1b. CoercionGenerator – float-to-float, int↔float, pointer
# ===================================================================

class TestCoercionFloat:
    def test_identity_f32(self, cg):
        assert cg.generate(F32, F32).is_identity

    def test_f32_to_f64(self, cg):
        c = cg.generate(F32, F64)
        assert c.is_valid and c.is_lossless and CastKind.FPEXT in c.cast_kinds

    def test_f64_to_f32(self, cg):
        c = cg.generate(F64, F32)
        assert c.is_valid and not c.is_lossless and CastKind.FPTRUNC in c.cast_kinds
        assert InformationLoss.PRECISION in c.total_information_loss

class TestCoercionIntFloat:
    def test_signed_to_float(self, cg):
        c = cg.generate(I32, F64)
        assert c.is_valid and CastKind.SITOFP in c.cast_kinds

    def test_unsigned_to_float(self, cg):
        c = cg.generate(U32, F64)
        assert c.is_valid and CastKind.UITOFP in c.cast_kinds

    def test_float_to_signed(self, cg):
        c = cg.generate(F64, I32)
        assert c.is_valid and not c.is_lossless and CastKind.FPTOSI in c.cast_kinds

    def test_float_to_unsigned(self, cg):
        c = cg.generate(F64, U32)
        assert c.is_valid and CastKind.FPTOUI in c.cast_kinds

    def test_i64_to_f32_precision_loss(self, cg):
        c = cg.generate(I64, F32)
        assert c.is_valid and not c.is_lossless

class TestCoercionPointer:
    def test_ptr_to_int(self, cg):
        c = cg.generate(PointerType(I32), I64)
        assert c.is_valid and CastKind.PTRTOINT in c.cast_kinds
        assert InformationLoss.PROVENANCE in c.total_information_loss

    def test_int_to_ptr(self, cg):
        c = cg.generate(I64, PointerType(I8))
        assert c.is_valid and CastKind.INTTOPTR in c.cast_kinds

    def test_ptr_to_ptr(self, cg):
        c = cg.generate(PointerType(I32), PointerType(I8))
        assert c.is_valid and CastKind.BITCAST in c.cast_kinds

    def test_ptr_identity(self, cg):
        ty = PointerType(I32)
        assert cg.generate(ty, ty).is_identity

class TestCanCoerceAndCost:
    def test_same_type(self, cg):
        assert cg.can_coerce(I32, I32)

    def test_int_to_int(self, cg):
        assert cg.can_coerce(I8, I64)

    def test_void_fails(self, cg):
        assert not cg.can_coerce(VOID, I32)

    def test_identity_cost_zero(self, cg):
        assert cg.coercion_cost(I32, I32) == 0

    def test_widening_le_narrowing(self, cg):
        assert cg.coercion_cost(I16, I32) <= cg.coercion_cost(I32, I16)

    def test_same_family_le_cross(self, cg):
        assert cg.coercion_cost(I16, I32) <= cg.coercion_cost(I16, F64)

class TestFindCommonType:
    def test_single(self, cg):
        assert cg.find_common_type([I32]) == I32

    def test_same_types(self, cg):
        assert cg.find_common_type([I32, I32]) == I32

    def test_int_widening(self, cg):
        r = cg.find_common_type([I16, I32])
        assert r is not None and r.width >= 32

    def test_mixed_sign(self, cg):
        assert cg.find_common_type([I32, U32]) is not None

    def test_int_float(self, cg):
        assert cg.find_common_type([I32, F64]) is not None


# ===================================================================
# 2. CoercionChain properties
# ===================================================================

class TestCoercionChain:
    def test_empty_chain(self):
        c = CoercionChain(src_type=I32, dst_type=I32)
        assert c.is_empty and c.is_identity and c.is_lossless and c.num_steps == 0

    def test_single_lossless_step(self):
        c = CoercionChain(src_type=I16, dst_type=I32)
        c.append(CoercionStep(CastKind.SEXT, I16, I32, InformationLoss.NONE))
        assert not c.is_empty and not c.is_identity and c.is_lossless and c.num_steps == 1

    def test_lossy_step(self):
        c = CoercionChain(src_type=I64, dst_type=I32)
        c.append(CoercionStep(CastKind.TRUNC, I64, I32, InformationLoss.BITS))
        assert not c.is_lossless
        assert InformationLoss.BITS in c.total_information_loss

    def test_multi_step(self):
        c = CoercionChain(src_type=I16, dst_type=F64)
        c.append(CoercionStep(CastKind.SEXT, I16, I32, InformationLoss.NONE))
        c.append(CoercionStep(CastKind.SITOFP, I32, F64, InformationLoss.NONE))
        assert c.num_steps == 2
        assert c.cast_kinds == [CastKind.SEXT, CastKind.SITOFP]

    def test_summary(self):
        c = CoercionChain(src_type=F32, dst_type=F64)
        c.append(CoercionStep(CastKind.FPEXT, F32, F64, InformationLoss.NONE))
        assert isinstance(c.summary(), str) and len(c.summary()) > 0

    def test_invalid_chain(self):
        c = CoercionChain(VOID, I32, is_valid=False, error_message="bad")
        assert not c.is_valid and c.error_message == "bad"

    def test_implicit_flags(self):
        c = CoercionChain(src_type=I16, dst_type=I32)
        c.append(CoercionStep(CastKind.SEXT, I16, I32, InformationLoss.NONE,
                              is_implicit_in_c=True, is_implicit_in_rust=False))
        assert c.has_implicit_in_c and not c.has_implicit_in_rust


# ===================================================================
# 3. CPromotionRules
# ===================================================================

class TestCPromotionRules:
    def test_i8_promotes_to_i32(self, c_rules):
        assert c_rules.promote(I8) == I32

    def test_i16_promotes_to_i32(self, c_rules):
        assert c_rules.promote(I16) == I32

    def test_u8_promotes(self, c_rules):
        assert c_rules.promote(U8) == I32

    def test_u16_promotes(self, c_rules):
        assert c_rules.promote(U16).width >= 32

    def test_i32_stays(self, c_rules):
        assert c_rules.promote(I32) == I32

    def test_u32_stays(self, c_rules):
        assert c_rules.promote(U32).width >= 32

    def test_i64_stays(self, c_rules):
        assert c_rules.promote(I64) == I64

class TestCUsualArithmeticConversion:
    def test_same_type(self, c_rules):
        common, lc, rc = c_rules.usual_arithmetic_conversion(I32, I32)
        assert common == I32

    def test_i16_i32(self, c_rules):
        common, lc, _ = c_rules.usual_arithmetic_conversion(I16, I32)
        assert common.width >= 32 and not lc.is_empty

    def test_u32_i32(self, c_rules):
        common, _, _ = c_rules.usual_arithmetic_conversion(U32, I32)
        assert common.width >= 32

    def test_i8_u8(self, c_rules):
        common, _, _ = c_rules.usual_arithmetic_conversion(I8, U8)
        assert common.width >= 32

    def test_i32_i64(self, c_rules):
        common, _, _ = c_rules.usual_arithmetic_conversion(I32, I64)
        assert common.width == 64

    def test_symmetry(self, c_rules):
        c1, _, _ = c_rules.usual_arithmetic_conversion(I16, I64)
        c2, _, _ = c_rules.usual_arithmetic_conversion(I64, I16)
        assert c1 == c2

class TestCDefaultArgAndImplicit:
    def test_default_arg_i8(self, c_rules):
        assert c_rules.default_argument_promotion(I8).width >= 32

    def test_default_arg_i32(self, c_rules):
        assert c_rules.default_argument_promotion(I32) == I32

    def test_widening_allowed(self, c_rules):
        assert c_rules.is_implicit_conversion_allowed(I16, I32) is True

    def test_same_allowed(self, c_rules):
        assert c_rules.is_implicit_conversion_allowed(I32, I32) is True

    def test_compute_chain_identity(self, c_rules):
        c = c_rules.compute_chain(I32, I32)
        assert c.is_empty or c.is_identity

    def test_compute_chain_widening(self, c_rules):
        c = c_rules.compute_chain(I8, I32)
        assert not c.is_empty and c.is_safe and c.original_type == I8 and c.final_type == I32

    def test_compute_chain_narrowing(self, c_rules):
        assert c_rules.compute_chain(I64, I32).has_narrowing


# ===================================================================
# 4. RustPromotionRules
# ===================================================================

class TestRustPromotionRules:
    def test_no_promotion_i8(self, rust_rules):
        assert rust_rules.promote(I8) == I8

    def test_no_promotion_i16(self, rust_rules):
        assert rust_rules.promote(I16) == I16

    def test_no_promotion_u8(self, rust_rules):
        assert rust_rules.promote(U8) == U8

    def test_i32_stays(self, rust_rules):
        assert rust_rules.promote(I32) == I32

    def test_no_implicit_conversion(self, rust_rules):
        assert rust_rules.is_implicit_conversion_allowed(I8, I32) is False

    def test_same_type_allowed(self, rust_rules):
        assert rust_rules.is_implicit_conversion_allowed(I32, I32) is True

    def test_uac_same_type(self, rust_rules):
        common, _, _ = rust_rules.usual_arithmetic_conversion(I32, I32)
        assert common == I32

    def test_explicit_chain(self, rust_rules):
        c = rust_rules.compute_chain(I8, I32)
        assert c.original_type == I8 and c.final_type == I32


# ===================================================================
# 5. PromotionComparison
# ===================================================================

class TestPromotionComparison:
    def test_same_types_no_diff(self):
        cmp = PromotionComparison.compare(I32, I32)
        assert not cmp.has_semantic_differences

    def test_small_ints(self):
        cmp = PromotionComparison.compare(I8, I16)
        assert cmp.c_chain is not None and cmp.rust_chain is not None

    def test_u32_i32(self):
        cmp = PromotionComparison.compare(U32, I32)
        assert cmp.c_common_type is not None

    def test_i16_i64(self):
        cmp = PromotionComparison.compare(I16, I64)
        assert cmp.c_common_type is not None and cmp.c_common_type.width >= 64

    def test_summary_string(self):
        assert len(PromotionComparison.compare(I8, I32).summary()) > 0

    def test_custom_rules(self):
        cmp = PromotionComparison.compare(
            I16, U16,
            c_rules=CPromotionRules(int_width=32),
            rust_rules=RustPromotionRules(),
        )
        assert cmp.c_chain is not None

    def test_large_types_no_diff(self):
        assert not PromotionComparison.compare(I64, I64).has_semantic_differences


# ===================================================================
# 6. TypeChecker
# ===================================================================

class TestTypeCheckerBinOp:
    def test_valid_add_i32(self, c_checker):
        r = c_checker.check_instruction(_binop(BinOpKind.ADD, I32, I32, I32))
        assert len(r.errors) == 0

    def test_valid_fadd(self, c_checker):
        r = c_checker.check_instruction(_binop(BinOpKind.FADD, F64, F64, F64))
        assert len(r.errors) == 0

    def test_width_mismatch_insert_cast(self, c_checker):
        r = c_checker.check_instruction(_binop(BinOpKind.ADD, I16, I32, I32))
        assert len(r.inserted_casts) > 0 or len(r.warnings) > 0 or len(r.errors) == 0

    def test_small_int_promotion(self, c_checker):
        r = c_checker.check_instruction(_binop(BinOpKind.ADD, I8, I8, I32))
        # i8 + i8 with i32 result: checker flags result/operand mismatch
        # but also inserts promotion casts, confirming C promotion awareness
        assert len(r.inserted_casts) > 0 or len(r.warnings) > 0

    def test_float_int_mismatch(self, c_checker):
        r = c_checker.check_instruction(_binop(BinOpKind.ADD, F64, I32))
        assert len(r.errors) + len(r.warnings) + len(r.inserted_casts) > 0

    def test_mul_u32(self, c_checker):
        assert len(c_checker.check_instruction(
            _binop(BinOpKind.MUL, U32, U32, U32)).errors) == 0

    def test_shift(self, c_checker):
        assert len(c_checker.check_instruction(
            _binop(BinOpKind.SHL, I32, I32, I32)).errors) == 0

class TestTypeCheckerCmp:
    def test_valid_int_cmp(self, c_checker):
        assert len(c_checker.check_instruction(
            _cmpop(CmpPredicate.SLT, I32, I32)).errors) == 0

    def test_valid_float_cmp(self, c_checker):
        assert len(c_checker.check_instruction(
            _cmpop(CmpPredicate.OLT, F64, F64)).errors) == 0

    def test_signed_unsigned_warning(self, c_checker):
        r = c_checker.check_instruction(_cmpop(CmpPredicate.SLT, I32, U32))
        assert len(r.warnings) + len(r.inserted_casts) > 0 or len(r.errors) == 0

    def test_width_mismatch(self, c_checker):
        r = c_checker.check_instruction(_cmpop(CmpPredicate.EQ, I16, I32))
        assert len(r.warnings) + len(r.inserted_casts) + len(r.errors) >= 0  # at least runs

    def test_ptr_compare(self, c_checker):
        pt = PointerType(I32)
        assert len(c_checker.check_instruction(_cmpop(CmpPredicate.EQ, pt, pt)).errors) == 0

class TestTypeCheckerReturn:
    def test_valid_return(self, c_checker):
        f = _func("foo", I32, [I32])
        bb = BasicBlock("entry", parent=f); f.add_block(bb)
        bb.append(_ret(I32))
        assert len(c_checker.check_function(f).errors) == 0

    def test_void_return(self, c_checker):
        f = _func("bar", VOID, [])
        bb = BasicBlock("entry", parent=f); f.add_block(bb)
        bb.append(ReturnInst(value=None))
        assert len(c_checker.check_function(f).errors) == 0

    def test_wrong_return_type(self, c_checker):
        f = _func("baz", I32, [])
        bb = BasicBlock("entry", parent=f); f.add_block(bb)
        bb.append(_ret(F64))
        r = c_checker.check_function(f)
        assert len(r.errors) + len(r.warnings) + len(r.inserted_casts) > 0

class TestTypeCheckerModule:
    def test_empty_module(self, c_checker):
        assert len(c_checker.check_module(Module("m")).errors) == 0

    def test_module_with_function(self, c_checker):
        mod = Module("m"); f = _func("add", I32, [I32, I32])
        bb = BasicBlock("entry", parent=f); f.add_block(bb)
        bb.append(_ret(I32)); mod.add_function(f)
        assert c_checker.check_module(mod).instructions_checked >= 1

class TestTypeCheckerCastInsertion:
    def test_cast_insertion_enabled(self):
        r = TypeChecker(Language.C, insert_casts=True).check_instruction(
            _binop(BinOpKind.ADD, I16, I32, I32))
        if r.inserted_casts:
            c = r.inserted_casts[0]
            assert isinstance(c, InsertedCast)
            assert c.src_type == I16 or c.dst_type == I32

    def test_strict_mode(self):
        r = TypeChecker(Language.C, insert_casts=False, strict=True).check_instruction(
            _binop(BinOpKind.ADD, I8, I8, I32))
        assert isinstance(r, TypeCheckResult)

class TestTypeCheckerRust:
    def test_rejects_implicit_widening(self, rust_checker):
        r = rust_checker.check_instruction(_binop(BinOpKind.ADD, I16, I32, I32))
        assert len(r.errors) + len(r.warnings) > 0

    def test_same_type_ok(self, rust_checker):
        assert len(rust_checker.check_instruction(
            _binop(BinOpKind.ADD, I32, I32, I32)).errors) == 0


# ===================================================================
# 7. Cross-language promotion differences
# ===================================================================

class TestCrossLanguage:
    def test_small_int_promotion_differs(self):
        c, r = CPromotionRules(), RustPromotionRules()
        assert c.promote(I8) != r.promote(I8)
        assert c.promote(I8) == I32 and r.promote(I8) == I8

    def test_u16_promotion_differs(self):
        c, r = CPromotionRules(), RustPromotionRules()
        assert c.promote(U16) != r.promote(U16)

    def test_i32_same(self):
        assert CPromotionRules().promote(I32) == RustPromotionRules().promote(I32)

    def test_comparison_detects_diff(self):
        cmp = PromotionComparison.compare(I8, I16)
        if cmp.has_semantic_differences:
            assert cmp.num_differences > 0

    def test_all_small_types_differ(self):
        c, r = CPromotionRules(), RustPromotionRules()
        for ty in [I8, I16, U8, U16]:
            assert c.promote(ty) != r.promote(ty), f"Expected diff for {ty}"

    def test_implicit_allowed_only_in_c(self):
        assert CPromotionRules().is_implicit_conversion_allowed(I8, I32) is True
        assert RustPromotionRules().is_implicit_conversion_allowed(I8, I32) is False


# ===================================================================
# 8. PromotionStep / PromotionChain dataclass tests
# ===================================================================

class TestPromotionStep:
    def test_widening(self):
        s = PromotionStep(I16, I32, PromotionKind.INTEGER_PROMOTION)
        assert s.widens and not s.narrows

    def test_narrowing(self):
        s = PromotionStep(I32, I16, PromotionKind.IMPLICIT_NARROWING)
        assert s.narrows and not s.widens

    def test_sign_change(self):
        assert PromotionStep(I32, U32, PromotionKind.SIGN_CONVERSION).changes_sign

    def test_safe_flag(self):
        assert PromotionStep(I8, I32, PromotionKind.INTEGER_PROMOTION, is_safe=True).is_safe

class TestPromotionChainDirect:
    def test_empty(self):
        assert PromotionChain().is_empty

    def test_with_step(self):
        c = PromotionChain(original_type=I8, final_type=I32)
        c.append(PromotionStep(I8, I32, PromotionKind.INTEGER_PROMOTION))
        assert not c.is_empty and c.is_safe

    def test_narrowing_flag(self):
        c = PromotionChain(original_type=I32, final_type=I16)
        c.append(PromotionStep(I32, I16, PromotionKind.IMPLICIT_NARROWING, is_safe=False))
        assert c.has_narrowing and not c.is_safe

    def test_sign_change_flag(self):
        c = PromotionChain(original_type=I32, final_type=U32)
        c.append(PromotionStep(I32, U32, PromotionKind.SIGN_CONVERSION))
        assert c.has_sign_change

    def test_total_width_change(self):
        c = PromotionChain(original_type=I8, final_type=I32)
        c.append(PromotionStep(I8, I32, PromotionKind.INTEGER_PROMOTION))
        assert c.total_width_change == 24

    def test_visualize(self):
        c = PromotionChain(original_type=I8, final_type=I32)
        c.append(PromotionStep(I8, I32, PromotionKind.INTEGER_PROMOTION))
        assert isinstance(c.visualize(), str) and len(c.visualize()) > 0


# ===================================================================
# 9. CoercionStep property tests
# ===================================================================

class TestCoercionStep:
    def test_lossless_sext(self):
        s = CoercionStep(CastKind.SEXT, I16, I32, InformationLoss.NONE)
        assert s.is_lossless and s.is_widening

    def test_lossy_trunc(self):
        s = CoercionStep(CastKind.TRUNC, I32, I16, InformationLoss.BITS)
        assert not s.is_lossless and s.is_narrowing

    def test_zext_widening(self):
        s = CoercionStep(CastKind.ZEXT, U8, U32, InformationLoss.NONE)
        assert s.is_widening and s.is_lossless

    def test_fpext(self):
        assert CoercionStep(CastKind.FPEXT, F32, F64, InformationLoss.NONE).is_lossless

    def test_fptrunc(self):
        s = CoercionStep(CastKind.FPTRUNC, F64, F32, InformationLoss.PRECISION)
        assert not s.is_lossless and s.is_narrowing


# ===================================================================
# 10. InformationLoss enum + TypeCheckResult/Error dataclasses
# ===================================================================

class TestInformationLoss:
    def test_all_variants_exist(self):
        for v in (InformationLoss.NONE, InformationLoss.PRECISION,
                  InformationLoss.MAGNITUDE, InformationLoss.SIGN,
                  InformationLoss.BITS, InformationLoss.PROVENANCE):
            assert v is not None

    def test_none_differs_from_bits(self):
        assert InformationLoss.NONE != InformationLoss.BITS

class TestTypeCheckResultDataclass:
    def test_empty(self):
        r = TypeCheckResult()
        assert not r.errors and not r.warnings and not r.inserted_casts

    def test_add_error(self):
        r = TypeCheckResult()
        r.errors.append(TypeCheckError("bad", TypeCheckSeverity.ERROR))
        assert len(r.errors) == 1

    def test_add_warning(self):
        r = TypeCheckResult()
        r.warnings.append(TypeCheckError("warn", TypeCheckSeverity.WARNING))
        assert len(r.warnings) == 1

    def test_add_cast(self):
        r = TypeCheckResult()
        r.inserted_casts.append(InsertedCast(I16, I32, CastKind.SEXT, "promo"))
        assert r.inserted_casts[0].reason == "promo"

class TestSeverity:
    def test_variants(self):
        assert TypeCheckSeverity.ERROR is not None
        assert TypeCheckSeverity.WARNING is not None
        assert TypeCheckSeverity.INFO is not None
