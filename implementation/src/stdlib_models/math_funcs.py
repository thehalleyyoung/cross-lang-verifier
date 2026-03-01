"""
Math function models: C math.h ↔ Rust f64/f32 methods.

Models sin/cos/tan/exp/log ↔ f64 methods, fabs ↔ f64::abs,
pow ↔ f64::powf, sqrt ↔ f64::sqrt, floor/ceil/round ↔ f64 methods.
Tracks precision differences, NaN handling, domain errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple

import z3

from .memory import DivergenceLevel, FunctionEquivalence, ModelResult


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

class MathDomain(Enum):
    """Domain of a math function argument."""
    ALL_REALS = auto()
    NON_NEGATIVE = auto()
    POSITIVE = auto()
    UNIT_INTERVAL = auto()       # [-1, 1]
    NON_ZERO = auto()
    FINITE = auto()
    ANY = auto()


@dataclass
class DomainSpec:
    """Specification of valid domain for a math function."""
    domain: MathDomain
    description: str
    c_error_behavior: str       # What C does outside domain (errno, NaN, etc.)
    rust_error_behavior: str    # What Rust does outside domain

    def z3_constraint(self, x: z3.FPRef, is_f32: bool = False) -> z3.BoolRef:
        """Generate z3 constraint for domain membership."""
        sort = z3.Float32() if is_f32 else z3.Float64()
        zero = z3.fpToFP(z3.RNE(), z3.RealVal(0), sort)
        one = z3.fpToFP(z3.RNE(), z3.RealVal(1), sort)
        neg_one = z3.fpToFP(z3.RNE(), z3.RealVal(-1), sort)

        if self.domain == MathDomain.ALL_REALS:
            return z3.Not(z3.fpIsNaN(x))
        elif self.domain == MathDomain.NON_NEGATIVE:
            return z3.And(z3.Not(z3.fpIsNaN(x)), z3.fpGEQ(x, zero))
        elif self.domain == MathDomain.POSITIVE:
            return z3.And(z3.Not(z3.fpIsNaN(x)), z3.fpGT(x, zero))
        elif self.domain == MathDomain.UNIT_INTERVAL:
            return z3.And(z3.Not(z3.fpIsNaN(x)), z3.fpGEQ(x, neg_one), z3.fpLEQ(x, one))
        elif self.domain == MathDomain.NON_ZERO:
            return z3.And(z3.Not(z3.fpIsNaN(x)), z3.Not(z3.fpIsZero(x)))
        elif self.domain == MathDomain.FINITE:
            return z3.And(z3.Not(z3.fpIsNaN(x)), z3.Not(z3.fpIsInf(x)))
        else:
            return z3.BoolVal(True)


# ---------------------------------------------------------------------------
# fabs model
# ---------------------------------------------------------------------------

class FabsModel:
    """
    Model for fabs/fabsf ↔ f64::abs / f32::abs.
    
    Semantically identical for normal values.
    NaN: Both return NaN. -0.0: Both return +0.0.
    """

    equivalence = FunctionEquivalence(
        c_function="fabs / fabsf",
        rust_equivalent="f64::abs / f32::abs",
        divergence_level=DivergenceLevel.NONE,
        preconditions=[],
        divergence_points=[
            "Exact semantics match: IEEE 754 absolute value",
            "NaN sign bit may differ (implementation-defined in C, unspecified in Rust)",
        ],
    )

    domain = DomainSpec(
        MathDomain.ANY, "all floats", "returns |x|", "returns |x|"
    )

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        result.return_value = z3.fpAbs(x)
        return result


# ---------------------------------------------------------------------------
# sqrt model
# ---------------------------------------------------------------------------

class SqrtModel:
    """
    Model for sqrt/sqrtf ↔ f64::sqrt / f32::sqrt.
    
    C: Negative input sets errno and returns NaN (or implementation-defined).
    Rust: Negative input returns NaN.
    """

    equivalence = FunctionEquivalence(
        c_function="sqrt / sqrtf",
        rust_equivalent="f64::sqrt / f32::sqrt",
        divergence_level=DivergenceLevel.LOW,
        preconditions=["x >= 0 for valid result"],
        divergence_points=[
            "Domain error: C sets errno, Rust returns NaN silently",
            "Both return NaN for negative inputs",
            "Errno: C may set errno=EDOM, Rust has no errno",
        ],
    )

    domain = DomainSpec(
        MathDomain.NON_NEGATIVE,
        "x >= 0",
        "sets errno=EDOM, returns NaN",
        "returns NaN",
    )

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()
        zero = z3.fpToFP(z3.RNE(), z3.RealVal(0), sort)

        is_negative = z3.fpLT(x, zero)
        nan = z3.fpNaN(sort)
        sqrt_val = z3.fpSqrt(z3.RNE(), x)

        result.return_value = z3.If(is_negative, nan, sqrt_val)

        # Divergence: errno side-effect in C
        result.divergence_condition = is_negative
        return result


# ---------------------------------------------------------------------------
# pow model
# ---------------------------------------------------------------------------

class PowModel:
    """
    Model for pow/powf ↔ f64::powf / f32::powf / f64::powi.
    
    Many edge cases: 0^0, negative base with fractional exponent, etc.
    """

    equivalence = FunctionEquivalence(
        c_function="pow / powf",
        rust_equivalent="f64::powf / f32::powf / f64::powi",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=[],
        divergence_points=[
            "0^0: C returns 1.0, Rust returns 1.0 (both IEEE)",
            "Negative^frac: C sets errno, returns NaN; Rust returns NaN",
            "Overflow: C returns ±HUGE_VAL (sets errno), Rust returns ±inf",
            "errno: C sets errno on domain/range errors, Rust doesn't",
            "Precision: Implementation-specific ULP differences possible",
        ],
    )

    domain = DomainSpec(
        MathDomain.ALL_REALS,
        "all reals (some combos produce NaN)",
        "sets errno on error",
        "returns NaN/inf",
    )

    @staticmethod
    def apply(
        base: z3.FPRef,
        exponent: z3.FPRef,
        is_f32: bool = False,
    ) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()
        zero = z3.fpToFP(z3.RNE(), z3.RealVal(0), sort)
        one = z3.fpToFP(z3.RNE(), z3.RealVal(1), sort)
        nan = z3.fpNaN(sort)

        ret = z3.FP("pow_result", sort)

        # 0^0 = 1.0
        both_zero = z3.And(z3.fpIsZero(base), z3.fpIsZero(exponent))
        result.constraints.append(z3.Implies(both_zero, ret == one))

        # NaN input -> NaN output
        nan_input = z3.Or(z3.fpIsNaN(base), z3.fpIsNaN(exponent))
        result.constraints.append(z3.Implies(nan_input, z3.fpIsNaN(ret)))

        # Negative base with non-integer exponent -> NaN
        neg_base = z3.fpLT(base, zero)
        result.divergence_condition = neg_base

        result.return_value = ret
        return result


# ---------------------------------------------------------------------------
# Trigonometric models
# ---------------------------------------------------------------------------

class SinModel:
    """Model for sin/sinf ↔ f64::sin / f32::sin."""

    equivalence = FunctionEquivalence(
        c_function="sin / sinf",
        rust_equivalent="f64::sin / f32::sin",
        divergence_level=DivergenceLevel.LOW,
        preconditions=[],
        divergence_points=[
            "Precision: 1-2 ULP difference possible across implementations",
            "Large inputs: Range reduction may differ between libm implementations",
            "NaN/Inf: Both return NaN, but errno handling differs",
        ],
    )

    domain = DomainSpec(MathDomain.FINITE, "finite reals", "returns NaN for inf/NaN", "returns NaN")

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()
        nan = z3.fpNaN(sort)
        ret = z3.FP("sin_result", sort)

        is_special = z3.Or(z3.fpIsNaN(x), z3.fpIsInf(x))
        result.constraints.append(z3.Implies(is_special, z3.fpIsNaN(ret)))

        # Bound: |sin(x)| <= 1
        one = z3.fpToFP(z3.RNE(), z3.RealVal(1), sort)
        neg_one = z3.fpToFP(z3.RNE(), z3.RealVal(-1), sort)
        result.constraints.append(z3.Implies(
            z3.Not(is_special),
            z3.And(z3.fpGEQ(ret, neg_one), z3.fpLEQ(ret, one)),
        ))

        # sin(0) = 0
        result.constraints.append(z3.Implies(z3.fpIsZero(x), z3.fpIsZero(ret)))

        result.return_value = ret
        return result


class CosModel:
    """Model for cos/cosf ↔ f64::cos / f32::cos."""

    equivalence = FunctionEquivalence(
        c_function="cos / cosf",
        rust_equivalent="f64::cos / f32::cos",
        divergence_level=DivergenceLevel.LOW,
        preconditions=[],
        divergence_points=[
            "Precision: 1-2 ULP difference possible",
            "Large inputs: Range reduction implementation varies",
        ],
    )

    domain = DomainSpec(MathDomain.FINITE, "finite reals", "returns NaN for inf/NaN", "returns NaN")

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()
        ret = z3.FP("cos_result", sort)

        is_special = z3.Or(z3.fpIsNaN(x), z3.fpIsInf(x))
        result.constraints.append(z3.Implies(is_special, z3.fpIsNaN(ret)))

        one = z3.fpToFP(z3.RNE(), z3.RealVal(1), sort)
        neg_one = z3.fpToFP(z3.RNE(), z3.RealVal(-1), sort)
        result.constraints.append(z3.Implies(
            z3.Not(is_special),
            z3.And(z3.fpGEQ(ret, neg_one), z3.fpLEQ(ret, one)),
        ))

        # cos(0) = 1
        result.constraints.append(z3.Implies(z3.fpIsZero(x), ret == one))

        result.return_value = ret
        return result


class TanModel:
    """Model for tan/tanf ↔ f64::tan / f32::tan."""

    equivalence = FunctionEquivalence(
        c_function="tan / tanf",
        rust_equivalent="f64::tan / f32::tan",
        divergence_level=DivergenceLevel.LOW,
        preconditions=[],
        divergence_points=[
            "Near π/2: precision can diverge significantly",
            "Precision: ULP differences in general",
        ],
    )

    domain = DomainSpec(MathDomain.FINITE, "finite reals", "returns NaN for inf/NaN", "returns NaN")

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()
        ret = z3.FP("tan_result", sort)

        is_special = z3.Or(z3.fpIsNaN(x), z3.fpIsInf(x))
        result.constraints.append(z3.Implies(is_special, z3.fpIsNaN(ret)))

        # tan(0) = 0
        result.constraints.append(z3.Implies(z3.fpIsZero(x), z3.fpIsZero(ret)))

        result.return_value = ret
        return result


# ---------------------------------------------------------------------------
# exp / log models
# ---------------------------------------------------------------------------

class ExpModel:
    """Model for exp/expf ↔ f64::exp / f32::exp."""

    equivalence = FunctionEquivalence(
        c_function="exp / expf / exp2 / exp2f",
        rust_equivalent="f64::exp / f32::exp / f64::exp2",
        divergence_level=DivergenceLevel.LOW,
        preconditions=[],
        divergence_points=[
            "Overflow: C sets errno=ERANGE, returns HUGE_VAL; Rust returns inf",
            "Underflow: C may set errno, returns 0; Rust returns 0",
            "Precision: 1-2 ULP differences between libm implementations",
        ],
    )

    domain = DomainSpec(MathDomain.FINITE, "finite reals", "errno on overflow/underflow", "returns inf/0")

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()
        zero = z3.fpToFP(z3.RNE(), z3.RealVal(0), sort)
        one = z3.fpToFP(z3.RNE(), z3.RealVal(1), sort)
        ret = z3.FP("exp_result", sort)

        is_special = z3.Or(z3.fpIsNaN(x), z3.fpIsInf(x))
        result.constraints.append(z3.Implies(z3.fpIsNaN(x), z3.fpIsNaN(ret)))

        # exp(0) = 1
        result.constraints.append(z3.Implies(z3.fpIsZero(x), ret == one))

        # exp(x) > 0 for all finite x
        result.constraints.append(z3.Implies(
            z3.Not(is_special), z3.fpGT(ret, zero)
        ))

        # exp(-inf) = 0, exp(+inf) = +inf
        neg_inf = z3.fpMinusInfinity(sort)
        pos_inf = z3.fpPlusInfinity(sort)
        result.constraints.append(z3.Implies(x == neg_inf, z3.fpIsZero(ret)))
        result.constraints.append(z3.Implies(x == pos_inf, z3.fpIsInf(ret)))

        result.return_value = ret
        return result


class LogModel:
    """Model for log/logf/log2/log10 ↔ f64::ln / f64::log2 / f64::log10."""

    equivalence = FunctionEquivalence(
        c_function="log / logf / log2 / log10",
        rust_equivalent="f64::ln / f64::log2 / f64::log10",
        divergence_level=DivergenceLevel.LOW,
        preconditions=["x > 0 for valid result"],
        divergence_points=[
            "Domain error: C sets errno=EDOM for x<0, returns NaN; Rust returns NaN",
            "Pole error: C sets errno=ERANGE for x=0, returns -HUGE_VAL; Rust returns -inf",
            "Precision: ULP differences between implementations",
        ],
    )

    domain = DomainSpec(
        MathDomain.POSITIVE,
        "x > 0",
        "errno=EDOM/ERANGE",
        "returns NaN/-inf",
    )

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()
        zero = z3.fpToFP(z3.RNE(), z3.RealVal(0), sort)
        one = z3.fpToFP(z3.RNE(), z3.RealVal(1), sort)
        nan = z3.fpNaN(sort)
        neg_inf = z3.fpMinusInfinity(sort)
        ret = z3.FP("log_result", sort)

        # log(NaN) = NaN
        result.constraints.append(z3.Implies(z3.fpIsNaN(x), z3.fpIsNaN(ret)))

        # log(x < 0) = NaN (domain error)
        result.constraints.append(z3.Implies(z3.fpLT(x, zero), z3.fpIsNaN(ret)))

        # log(0) = -inf (pole error)
        result.constraints.append(z3.Implies(z3.fpIsZero(x), ret == neg_inf))

        # log(1) = 0
        result.constraints.append(z3.Implies(x == one, z3.fpIsZero(ret)))

        # log(+inf) = +inf
        pos_inf = z3.fpPlusInfinity(sort)
        result.constraints.append(z3.Implies(x == pos_inf, z3.fpIsInf(ret)))

        result.divergence_condition = z3.Or(z3.fpLT(x, zero), z3.fpIsZero(x))

        result.return_value = ret
        return result


# ---------------------------------------------------------------------------
# floor / ceil / round models
# ---------------------------------------------------------------------------

class FloorModel:
    """Model for floor/floorf ↔ f64::floor / f32::floor."""

    equivalence = FunctionEquivalence(
        c_function="floor / floorf",
        rust_equivalent="f64::floor / f32::floor",
        divergence_level=DivergenceLevel.NONE,
        preconditions=[],
        divergence_points=[
            "Semantically identical: IEEE 754 roundToIntegralTowardNegative",
        ],
    )

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        result.return_value = z3.fpRoundToIntegral(z3.RTN(), x)
        return result


class CeilModel:
    """Model for ceil/ceilf ↔ f64::ceil / f32::ceil."""

    equivalence = FunctionEquivalence(
        c_function="ceil / ceilf",
        rust_equivalent="f64::ceil / f32::ceil",
        divergence_level=DivergenceLevel.NONE,
        preconditions=[],
        divergence_points=[
            "Semantically identical: IEEE 754 roundToIntegralTowardPositive",
        ],
    )

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        result.return_value = z3.fpRoundToIntegral(z3.RTP(), x)
        return result


class RoundModel:
    """
    Model for round/roundf ↔ f64::round / f32::round.
    
    C: round rounds halfway away from zero.
    Rust: f64::round rounds halfway away from zero (same).
    """

    equivalence = FunctionEquivalence(
        c_function="round / roundf / nearbyint",
        rust_equivalent="f64::round / f32::round",
        divergence_level=DivergenceLevel.NONE,
        preconditions=[],
        divergence_points=[
            "Semantically identical for round (ties away from zero)",
            "nearbyint: C uses current rounding mode, Rust::round always ties-away",
        ],
    )

    @staticmethod
    def apply(x: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        result.return_value = z3.fpRoundToIntegral(z3.RNA(), x)
        return result


# ---------------------------------------------------------------------------
# fmod / remainder
# ---------------------------------------------------------------------------

class FmodModel:
    """Model for fmod ↔ f64::rem_euclid / % operator."""

    equivalence = FunctionEquivalence(
        c_function="fmod / fmodf / remainder",
        rust_equivalent="f64 % operator / f64::rem_euclid",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=["y != 0"],
        divergence_points=[
            "Sign: fmod result has sign of dividend, rem_euclid is always non-negative",
            "Division by zero: C is UB (may return NaN), Rust returns NaN",
            "fmod vs remainder: C remainder rounds to nearest, fmod truncates",
        ],
    )

    @staticmethod
    def apply(x: z3.FPRef, y: z3.FPRef, is_f32: bool = False) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()
        nan = z3.fpNaN(sort)

        y_zero = z3.fpIsZero(y)
        nan_input = z3.Or(z3.fpIsNaN(x), z3.fpIsNaN(y))
        x_inf = z3.fpIsInf(x)

        ret = z3.FP("fmod_result", sort)

        # Special cases: NaN input, inf % y, x % 0
        result.constraints.append(z3.Implies(
            z3.Or(nan_input, x_inf, y_zero),
            z3.fpIsNaN(ret),
        ))

        result.return_value = ret
        result.divergence_condition = y_zero
        return result


# ---------------------------------------------------------------------------
# Float-to-int conversion
# ---------------------------------------------------------------------------

class FloatToIntModel:
    """
    Model for C casts (int)f ↔ Rust `as` / saturating casts.
    
    This is a major divergence area.
    """

    equivalence = FunctionEquivalence(
        c_function="(int)f / lround / trunc casts",
        rust_equivalent="f64 as i32 / i32::try_from",
        divergence_level=DivergenceLevel.HIGH,
        preconditions=["Float value is within integer range"],
        divergence_points=[
            "Out of range: C is UB, Rust saturates (as) or returns Err (try_from)",
            "NaN: C is UB, Rust saturates to 0 (as) or Err",
            "Infinity: C is UB, Rust saturates to MAX/MIN",
            "Truncation vs rounding: (int)f truncates, lround rounds",
        ],
    )

    @staticmethod
    def apply(
        x: z3.FPRef,
        target_width: int = 32,
        signed: bool = True,
        is_f32: bool = False,
    ) -> ModelResult:
        result = ModelResult()
        sort = z3.Float32() if is_f32 else z3.Float64()

        if signed:
            max_val = (1 << (target_width - 1)) - 1
            min_val = -(1 << (target_width - 1))
        else:
            max_val = (1 << target_width) - 1
            min_val = 0

        max_fp = z3.fpToFP(z3.RNE(), z3.RealVal(max_val), sort)
        min_fp = z3.fpToFP(z3.RNE(), z3.RealVal(min_val), sort)

        in_range = z3.And(
            z3.Not(z3.fpIsNaN(x)),
            z3.fpLEQ(x, max_fp),
            z3.fpGEQ(x, min_fp),
        )

        # C result: UB when out of range (model as unconstrained)
        c_result = z3.BitVec("float_to_int_c", target_width)

        # Rust result: saturating
        rust_result = z3.BitVec("float_to_int_rust", target_width)
        result.constraints.append(z3.Implies(
            z3.fpIsNaN(x),
            rust_result == z3.BitVecVal(0, target_width),
        ))
        result.constraints.append(z3.Implies(
            z3.And(z3.Not(z3.fpIsNaN(x)), z3.fpGT(x, max_fp)),
            rust_result == z3.BitVecVal(max_val, target_width),
        ))
        result.constraints.append(z3.Implies(
            z3.And(z3.Not(z3.fpIsNaN(x)), z3.fpLT(x, min_fp)),
            rust_result == z3.BitVecVal(min_val & ((1 << target_width) - 1), target_width),
        ))

        # When in range, both agree
        result.constraints.append(z3.Implies(in_range, c_result == rust_result))

        result.divergence_condition = z3.Not(in_range)
        result.return_value = c_result
        return result


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class MathFunctionModels:
    """Registry of all math function models."""

    models = {
        "fabs": FabsModel,
        "fabsf": FabsModel,
        "sqrt": SqrtModel,
        "sqrtf": SqrtModel,
        "pow": PowModel,
        "powf": PowModel,
        "sin": SinModel,
        "sinf": SinModel,
        "cos": CosModel,
        "cosf": CosModel,
        "tan": TanModel,
        "tanf": TanModel,
        "exp": ExpModel,
        "expf": ExpModel,
        "exp2": ExpModel,
        "exp2f": ExpModel,
        "log": LogModel,
        "logf": LogModel,
        "log2": LogModel,
        "log10": LogModel,
        "floor": FloorModel,
        "floorf": FloorModel,
        "ceil": CeilModel,
        "ceilf": CeilModel,
        "round": RoundModel,
        "roundf": RoundModel,
        "fmod": FmodModel,
        "fmodf": FmodModel,
        "remainder": FmodModel,
    }

    @classmethod
    def get_model(cls, func_name: str) -> Optional[type]:
        return cls.models.get(func_name)

    @classmethod
    def get_equivalence(cls, func_name: str) -> Optional[FunctionEquivalence]:
        model = cls.get_model(func_name)
        if model and hasattr(model, 'equivalence'):
            return model.equivalence
        return None

    @classmethod
    def all_equivalences(cls) -> List[FunctionEquivalence]:
        seen = set()
        result = []
        for model_cls in cls.models.values():
            if id(model_cls) not in seen and hasattr(model_cls, 'equivalence'):
                seen.add(id(model_cls))
                result.append(model_cls.equivalence)
        return result

    @classmethod
    def get_domain(cls, func_name: str) -> Optional[DomainSpec]:
        model = cls.get_model(func_name)
        if model and hasattr(model, 'domain'):
            return model.domain
        return None

    @classmethod
    def summary(cls) -> str:
        lines = ["Math Function Models:"]
        for eq in cls.all_equivalences():
            lines.append(f"  {eq.summary()}")
            for dp in eq.divergence_points:
                lines.append(f"    ⚠ {dp}")
        return "\n".join(lines)
