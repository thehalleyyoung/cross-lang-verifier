"""
Semantic evaluator for the Cross-Language Equivalence Verifier IR.

Given an IR instruction and a SemanticConfig, computes the concrete (or
abstract / interval) result.  Tracks overflow flags, undefined-behavior
triggers, panics, and traps.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto, Flag
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..ir.types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    ArrayType,
    StructType,
    VoidType,
    Signedness,
    FloatKind,
)
from ..ir.instructions import (
    Instruction,
    BinaryOp,
    UnaryOp,
    CompareOp,
    CastInst,
    LoadInst,
    StoreInst,
    AllocaInst,
    GetElementPtrInst,
    CallInst,
    ReturnInst,
    SelectInst,
    PhiInst,
    BinOpKind,
    UnaryOpKind,
    CastKind,
    CmpPredicate,
    Value,
    Constant,
)
from .semantic_config import (
    SemanticConfig,
    OverflowMode,
    FloatModel,
    ShiftModel,
    DivisionModel,
    FloatToIntModel,
)


# ── Evaluation flags ─────────────────────────────────────────────────────

class EvalFlags(Flag):
    """Flags set during evaluation."""
    NONE = 0
    OVERFLOW = auto()          # Integer overflow occurred
    UNDERFLOW = auto()         # Integer underflow occurred
    UB_TRIGGERED = auto()      # Undefined behavior was triggered
    PANIC = auto()             # A panic would be raised
    TRAP = auto()              # Hardware trap
    DIVISION_BY_ZERO = auto()  # Division by zero
    SHIFT_OVERFLOW = auto()    # Shift amount out of range
    FLOAT_INEXACT = auto()     # Floating-point precision loss
    FLOAT_INVALID = auto()     # Invalid float operation (e.g. 0/0)
    FLOAT_OOB_CAST = auto()   # Float-to-int out of range
    NULL_DEREF = auto()        # Null pointer dereference
    NARROWING = auto()         # Narrowing conversion lost bits
    SATURATED = auto()         # Value was saturated to range

    @property
    def is_clean(self) -> bool:
        return self is EvalFlags.NONE

    @property
    def has_ub(self) -> bool:
        return bool(self & EvalFlags.UB_TRIGGERED)

    @property
    def has_panic(self) -> bool:
        return bool(self & EvalFlags.PANIC)


# ── Evaluation result ────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Result of evaluating a single instruction."""
    value: Any = None                 # The computed value (int, float, bool, None)
    flags: EvalFlags = EvalFlags.NONE
    result_type: IRType | None = None
    is_bottom: bool = False           # True if UB/panic makes result meaningless
    description: str = ""

    @property
    def is_defined(self) -> bool:
        return not self.is_bottom and not self.flags.has_ub

    def __repr__(self) -> str:
        if self.is_bottom:
            return "EvalResult(⊥)"
        flag_str = ""
        if self.flags is not EvalFlags.NONE:
            flag_str = f", flags={self.flags}"
        return f"EvalResult({self.value}{flag_str})"

    @staticmethod
    def bottom(reason: str = "", flags: EvalFlags = EvalFlags.NONE) -> "EvalResult":
        return EvalResult(value=None, flags=flags, is_bottom=True, description=reason)

    @staticmethod
    def ok(value: Any, result_type: IRType | None = None) -> "EvalResult":
        return EvalResult(value=value, result_type=result_type)


# ── Interval (for abstract evaluation) ───────────────────────────────────

@dataclass
class Interval:
    """A closed interval [lo, hi] for abstract integer evaluation."""
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            self.lo, self.hi = self.hi, self.lo

    @staticmethod
    def point(v: int) -> "Interval":
        return Interval(v, v)

    @staticmethod
    def full_range(width: int, signed: bool) -> "Interval":
        if signed:
            return Interval(-(1 << (width - 1)), (1 << (width - 1)) - 1)
        return Interval(0, (1 << width) - 1)

    @property
    def is_point(self) -> bool:
        return self.lo == self.hi

    @property
    def span(self) -> int:
        return self.hi - self.lo

    def contains(self, v: int) -> bool:
        return self.lo <= v <= self.hi

    def overlaps(self, other: "Interval") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    def join(self, other: "Interval") -> "Interval":
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def meet(self, other: "Interval") -> "Interval | None":
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)

    def add(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def sub(self, other: "Interval") -> "Interval":
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def mul(self, other: "Interval") -> "Interval":
        products = [
            self.lo * other.lo, self.lo * other.hi,
            self.hi * other.lo, self.hi * other.hi,
        ]
        return Interval(min(products), max(products))

    def negate(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def fits_in(self, width: int, signed: bool) -> bool:
        if signed:
            return self.lo >= -(1 << (width - 1)) and self.hi <= (1 << (width - 1)) - 1
        return self.lo >= 0 and self.hi <= (1 << width) - 1

    def __repr__(self) -> str:
        return f"[{self.lo}, {self.hi}]"


# ── Semantic evaluator ───────────────────────────────────────────────────

class SemanticEvaluator:
    """Evaluates IR instructions under a given SemanticConfig.

    Supports:
    - Concrete evaluation (fixed integer/float inputs)
    - Abstract (interval) evaluation for range analysis
    """

    def __init__(self, config: SemanticConfig) -> None:
        self.config = config

    # ── Public API ───────────────────────────────────────────────────────

    def evaluate(
        self,
        inst: Instruction,
        operand_values: dict[int, Any] | None = None,
    ) -> EvalResult:
        """Evaluate *inst* with optional concrete operand values.

        *operand_values* maps Value.id → concrete value (int or float).
        """
        vals = operand_values or {}

        if isinstance(inst, BinaryOp):
            return self._eval_binop(inst, vals)
        if isinstance(inst, UnaryOp):
            return self._eval_unop(inst, vals)
        if isinstance(inst, CompareOp):
            return self._eval_cmp(inst, vals)
        if isinstance(inst, CastInst):
            return self._eval_cast(inst, vals)
        if isinstance(inst, SelectInst):
            return self._eval_select(inst, vals)
        if isinstance(inst, (LoadInst, StoreInst)):
            return self._eval_memory(inst, vals)
        if isinstance(inst, CallInst):
            return EvalResult(value=None, description="Call not concretely evaluable")
        if isinstance(inst, ReturnInst):
            return self._eval_return(inst, vals)
        if isinstance(inst, PhiInst):
            return EvalResult(value=None, description="Phi requires control flow context")
        return EvalResult(value=None, description=f"Unsupported instruction: {inst.opcode_name()}")

    def evaluate_abstract(
        self,
        inst: Instruction,
        operand_intervals: dict[int, Interval] | None = None,
    ) -> Tuple[Interval | None, EvalFlags]:
        """Abstract evaluation using intervals."""
        intervals = operand_intervals or {}

        if isinstance(inst, BinaryOp):
            return self._eval_binop_abstract(inst, intervals)
        if isinstance(inst, UnaryOp):
            return self._eval_unop_abstract(inst, intervals)
        if isinstance(inst, CastInst):
            return self._eval_cast_abstract(inst, intervals)

        return None, EvalFlags.NONE

    # ── Binary operation evaluation ──────────────────────────────────────

    def _eval_binop(self, inst: BinaryOp, vals: dict[int, Any]) -> EvalResult:
        lhs_val = self._resolve_value(inst.lhs, vals)
        rhs_val = self._resolve_value(inst.rhs, vals)
        if lhs_val is None or rhs_val is None:
            return EvalResult(value=None, description="Missing operand value")

        if inst.is_floating():
            return self._eval_float_binop(inst.op, lhs_val, rhs_val, inst.type)

        if not isinstance(inst.lhs.type, IntType):
            return EvalResult(value=None, description="Non-integer binary op")

        int_type = inst.lhs.type
        return self._eval_int_binop(inst.op, lhs_val, rhs_val, int_type)

    def _eval_int_binop(
        self, op: BinOpKind, lhs: int, rhs: int, int_type: IntType,
    ) -> EvalResult:
        width = int_type.width
        signed = int_type.is_signed
        flags = EvalFlags.NONE

        # Division / remainder: check for zero and overflow
        if op in (BinOpKind.SDIV, BinOpKind.UDIV, BinOpKind.SREM, BinOpKind.UREM):
            if rhs == 0:
                if self.config.division_model is DivisionModel.UB:
                    return EvalResult.bottom("Division by zero (UB)",
                                             EvalFlags.DIVISION_BY_ZERO | EvalFlags.UB_TRIGGERED)
                if self.config.division_model is DivisionModel.Panic:
                    return EvalResult.bottom("Division by zero (panic)",
                                             EvalFlags.DIVISION_BY_ZERO | EvalFlags.PANIC)
                return EvalResult.bottom("Division by zero (trap)",
                                         EvalFlags.DIVISION_BY_ZERO | EvalFlags.TRAP)

            # Signed division overflow: INT_MIN / -1
            if signed and op in (BinOpKind.SDIV, BinOpKind.SREM):
                min_val = int_type.min_value
                if lhs == min_val and rhs == -1:
                    mode = self.config.get_overflow_mode(True)
                    if mode is OverflowMode.UB:
                        return EvalResult.bottom("Signed division overflow (UB)",
                                                 EvalFlags.OVERFLOW | EvalFlags.UB_TRIGGERED)
                    if mode is OverflowMode.Panic:
                        return EvalResult.bottom("Signed division overflow (panic)",
                                                 EvalFlags.OVERFLOW | EvalFlags.PANIC)
                    flags |= EvalFlags.OVERFLOW

        # Shift operations: check shift amount
        if op in (BinOpKind.SHL, BinOpKind.LSHR, BinOpKind.ASHR):
            if rhs < 0 or rhs >= width:
                if self.config.shift_model is ShiftModel.UB_on_overshift:
                    return EvalResult.bottom("Shift amount out of range (UB)",
                                             EvalFlags.SHIFT_OVERFLOW | EvalFlags.UB_TRIGGERED)
                if self.config.shift_model is ShiftModel.Panic_on_overshift:
                    return EvalResult.bottom("Shift amount out of range (panic)",
                                             EvalFlags.SHIFT_OVERFLOW | EvalFlags.PANIC)
                # Mask mode
                rhs = rhs & (width - 1)
                flags |= EvalFlags.SHIFT_OVERFLOW

        # Compute the raw result
        raw: int
        if op is BinOpKind.ADD:
            raw = lhs + rhs
        elif op is BinOpKind.SUB:
            raw = lhs - rhs
        elif op is BinOpKind.MUL:
            raw = lhs * rhs
        elif op is BinOpKind.SDIV:
            # Truncate toward zero
            if (lhs < 0) != (rhs < 0) and lhs % rhs != 0:
                raw = -(abs(lhs) // abs(rhs))
            else:
                raw = abs(lhs) // abs(rhs)
                if lhs < 0 and rhs < 0:
                    pass
                elif lhs < 0 or rhs < 0:
                    raw = -raw
        elif op is BinOpKind.UDIV:
            mask = (1 << width) - 1
            raw = (lhs & mask) // (rhs & mask)
        elif op is BinOpKind.SREM:
            if rhs != 0:
                raw = lhs - (lhs // rhs) * rhs
                # C truncation toward zero semantics
                if raw != 0 and (lhs < 0) != (raw < 0):
                    raw = lhs % rhs
                    if raw != 0 and (lhs < 0):
                        raw = -(abs(lhs) % abs(rhs))
                    elif raw != 0:
                        raw = abs(lhs) % abs(rhs)
                raw = int(math.fmod(lhs, rhs))
            else:
                raw = 0
        elif op is BinOpKind.UREM:
            mask = (1 << width) - 1
            raw = (lhs & mask) % (rhs & mask)
        elif op is BinOpKind.SHL:
            raw = lhs << rhs
        elif op is BinOpKind.LSHR:
            mask = (1 << width) - 1
            raw = (lhs & mask) >> rhs
        elif op is BinOpKind.ASHR:
            # Arithmetic shift right preserves sign
            if lhs < 0:
                # Sign-extend then shift
                raw = lhs >> rhs
            else:
                raw = lhs >> rhs
        elif op is BinOpKind.AND:
            raw = lhs & rhs
        elif op is BinOpKind.OR:
            raw = lhs | rhs
        elif op is BinOpKind.XOR:
            raw = lhs ^ rhs
        else:
            return EvalResult(value=None, description=f"Unknown int binop: {op}")

        # Check for overflow (for add/sub/mul)
        if op in (BinOpKind.ADD, BinOpKind.SUB, BinOpKind.MUL):
            max_val = int_type.max_value
            min_val = int_type.min_value
            overflowed = raw > max_val or raw < min_val

            if overflowed:
                mode = self.config.get_overflow_mode(signed)
                if mode is OverflowMode.UB:
                    return EvalResult.bottom(
                        f"Integer overflow (UB): {lhs} {op.value} {rhs} = {raw}",
                        EvalFlags.OVERFLOW | EvalFlags.UB_TRIGGERED,
                    )
                if mode is OverflowMode.Panic:
                    return EvalResult.bottom(
                        f"Integer overflow (panic): {lhs} {op.value} {rhs} = {raw}",
                        EvalFlags.OVERFLOW | EvalFlags.PANIC,
                    )
                if mode is OverflowMode.Saturate:
                    raw = max(min_val, min(raw, max_val))
                    flags |= EvalFlags.OVERFLOW | EvalFlags.SATURATED
                else:
                    # Wrap
                    raw = int_type.truncate(raw)
                    flags |= EvalFlags.OVERFLOW

        # Truncate to bit width
        result = int_type.truncate(raw)
        return EvalResult(value=result, flags=flags, result_type=int_type)

    def _eval_float_binop(
        self, op: BinOpKind, lhs: float, rhs: float, result_type: IRType,
    ) -> EvalResult:
        flags = EvalFlags.NONE

        if op is BinOpKind.FADD:
            raw = lhs + rhs
        elif op is BinOpKind.FSUB:
            raw = lhs - rhs
        elif op is BinOpKind.FMUL:
            raw = lhs * rhs
        elif op is BinOpKind.FDIV:
            if rhs == 0.0:
                if lhs == 0.0:
                    raw = float('nan')
                    flags |= EvalFlags.FLOAT_INVALID
                else:
                    raw = math.copysign(float('inf'), lhs * rhs) if rhs == 0.0 else lhs / rhs
                    # Actually handle +/- inf
                    if rhs == 0.0:
                        raw = float('inf') if lhs > 0 else float('-inf')
            else:
                raw = lhs / rhs
        elif op is BinOpKind.FREM:
            if rhs == 0.0:
                raw = float('nan')
                flags |= EvalFlags.FLOAT_INVALID
            else:
                raw = math.fmod(lhs, rhs)
        else:
            return EvalResult(value=None, description=f"Unknown float binop: {op}")

        if math.isnan(raw) and not (math.isnan(lhs) or math.isnan(rhs)):
            flags |= EvalFlags.FLOAT_INVALID
        if math.isinf(raw) and not (math.isinf(lhs) or math.isinf(rhs)):
            flags |= EvalFlags.OVERFLOW

        # Clamp to f32 if needed
        if isinstance(result_type, FloatType) and result_type.kind is FloatKind.F32:
            raw = self._clamp_f32(raw)

        return EvalResult(value=raw, flags=flags, result_type=result_type)

    # ── Unary operation evaluation ───────────────────────────────────────

    def _eval_unop(self, inst: UnaryOp, vals: dict[int, Any]) -> EvalResult:
        operand = inst.operands[0]
        val = self._resolve_value(operand, vals)
        if val is None:
            return EvalResult(value=None, description="Missing operand")

        if inst.op is UnaryOpKind.NEG:
            if isinstance(operand.type, IntType):
                int_type = operand.type
                raw = -val
                # Check for overflow (negating INT_MIN)
                if val == int_type.min_value and int_type.is_signed:
                    mode = self.config.get_overflow_mode(True)
                    if mode is OverflowMode.UB:
                        return EvalResult.bottom("Negation overflow (UB)",
                                                 EvalFlags.OVERFLOW | EvalFlags.UB_TRIGGERED)
                    if mode is OverflowMode.Panic:
                        return EvalResult.bottom("Negation overflow (panic)",
                                                 EvalFlags.OVERFLOW | EvalFlags.PANIC)
                result = int_type.truncate(raw)
                return EvalResult(value=result, result_type=int_type)
            return EvalResult(value=-val, result_type=operand.type)

        if inst.op is UnaryOpKind.FNEG:
            return EvalResult(value=-val, result_type=operand.type)

        if inst.op is UnaryOpKind.NOT:
            return EvalResult(value=int(not val), result_type=operand.type)

        if inst.op is UnaryOpKind.BITWISE_NOT:
            if isinstance(operand.type, IntType):
                mask = operand.type.mask
                return EvalResult(value=(~val) & mask, result_type=operand.type)
            return EvalResult(value=~val, result_type=operand.type)

        return EvalResult(value=None, description=f"Unknown unop: {inst.op}")

    # ── Comparison evaluation ────────────────────────────────────────────

    def _eval_cmp(self, inst: CompareOp, vals: dict[int, Any]) -> EvalResult:
        lhs_val = self._resolve_value(inst.lhs, vals)
        rhs_val = self._resolve_value(inst.rhs, vals)
        if lhs_val is None or rhs_val is None:
            return EvalResult(value=None, description="Missing operand")

        pred = inst.predicate
        result: bool

        # Float comparisons with NaN handling
        if pred.is_float():
            lhs_nan = isinstance(lhs_val, float) and math.isnan(lhs_val)
            rhs_nan = isinstance(rhs_val, float) and math.isnan(rhs_val)
            any_nan = lhs_nan or rhs_nan

            if pred is CmpPredicate.ORD:
                result = not any_nan
            elif pred is CmpPredicate.UNO:
                result = any_nan
            elif pred is CmpPredicate.OEQ:
                result = not any_nan and lhs_val == rhs_val
            elif pred is CmpPredicate.ONE:
                result = not any_nan and lhs_val != rhs_val
            elif pred is CmpPredicate.OLT:
                result = not any_nan and lhs_val < rhs_val
            elif pred is CmpPredicate.OLE:
                result = not any_nan and lhs_val <= rhs_val
            elif pred is CmpPredicate.OGT:
                result = not any_nan and lhs_val > rhs_val
            elif pred is CmpPredicate.OGE:
                result = not any_nan and lhs_val >= rhs_val
            elif pred is CmpPredicate.UEQ:
                result = any_nan or lhs_val == rhs_val
            elif pred is CmpPredicate.UNE:
                result = any_nan or lhs_val != rhs_val
            else:
                result = False
        else:
            # Integer comparisons
            if pred.is_unsigned() and isinstance(inst.lhs.type, IntType):
                width = inst.lhs.type.width
                mask = (1 << width) - 1
                lhs_val = lhs_val & mask
                rhs_val = rhs_val & mask

            if pred in (CmpPredicate.EQ, CmpPredicate.OEQ):
                result = lhs_val == rhs_val
            elif pred in (CmpPredicate.NE, CmpPredicate.ONE):
                result = lhs_val != rhs_val
            elif pred in (CmpPredicate.SLT, CmpPredicate.OLT):
                result = lhs_val < rhs_val
            elif pred in (CmpPredicate.SLE, CmpPredicate.OLE):
                result = lhs_val <= rhs_val
            elif pred in (CmpPredicate.SGT, CmpPredicate.OGT):
                result = lhs_val > rhs_val
            elif pred in (CmpPredicate.SGE, CmpPredicate.OGE):
                result = lhs_val >= rhs_val
            elif pred is CmpPredicate.ULT:
                result = lhs_val < rhs_val
            elif pred is CmpPredicate.ULE:
                result = lhs_val <= rhs_val
            elif pred is CmpPredicate.UGT:
                result = lhs_val > rhs_val
            elif pred is CmpPredicate.UGE:
                result = lhs_val >= rhs_val
            else:
                result = False

        return EvalResult(value=int(result), result_type=inst.type)

    # ── Cast evaluation ──────────────────────────────────────────────────

    def _eval_cast(self, inst: CastInst, vals: dict[int, Any]) -> EvalResult:
        val = self._resolve_value(inst.operand, vals)
        if val is None:
            return EvalResult(value=None, description="Missing operand")

        src_type = inst.src_type
        dst_type = inst.dest_type
        kind = inst.cast_kind
        flags = EvalFlags.NONE

        if kind is CastKind.TRUNC:
            if isinstance(dst_type, IntType):
                result = dst_type.truncate(val)
                if result != val:
                    flags |= EvalFlags.NARROWING
                return EvalResult(value=result, flags=flags, result_type=dst_type)

        if kind is CastKind.ZEXT:
            if isinstance(src_type, IntType):
                mask = src_type.mask
                result = val & mask
                return EvalResult(value=result, result_type=dst_type)

        if kind is CastKind.SEXT:
            if isinstance(src_type, IntType) and isinstance(dst_type, IntType):
                mask = src_type.mask
                val_masked = val & mask
                if val_masked & (1 << (src_type.width - 1)):
                    # Sign bit is set; extend
                    result = val_masked - (1 << src_type.width)
                else:
                    result = val_masked
                return EvalResult(value=result, result_type=dst_type)

        if kind is CastKind.FPTRUNC:
            result = self._clamp_f32(float(val))
            flags |= EvalFlags.FLOAT_INEXACT
            return EvalResult(value=result, flags=flags, result_type=dst_type)

        if kind is CastKind.FPEXT:
            return EvalResult(value=float(val), result_type=dst_type)

        if kind in (CastKind.FPTOSI, CastKind.FPTOUI):
            if isinstance(dst_type, IntType):
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    if self.config.float_to_int is FloatToIntModel.UB:
                        return EvalResult.bottom(
                            "Float-to-int: NaN/Inf (UB)",
                            EvalFlags.FLOAT_OOB_CAST | EvalFlags.UB_TRIGGERED,
                        )
                    # Saturate: NaN → 0, Inf → max/min
                    if math.isnan(fval):
                        return EvalResult(value=0, flags=EvalFlags.FLOAT_OOB_CAST | EvalFlags.SATURATED,
                                          result_type=dst_type)
                    if fval > 0:
                        return EvalResult(value=dst_type.max_value,
                                          flags=EvalFlags.FLOAT_OOB_CAST | EvalFlags.SATURATED,
                                          result_type=dst_type)
                    return EvalResult(value=dst_type.min_value,
                                      flags=EvalFlags.FLOAT_OOB_CAST | EvalFlags.SATURATED,
                                      result_type=dst_type)

                int_val = int(fval)
                max_v = dst_type.max_value
                min_v = dst_type.min_value
                if int_val > max_v or int_val < min_v:
                    if self.config.float_to_int is FloatToIntModel.UB:
                        return EvalResult.bottom(
                            f"Float-to-int out of range: {fval} (UB)",
                            EvalFlags.FLOAT_OOB_CAST | EvalFlags.UB_TRIGGERED,
                        )
                    # Saturate
                    int_val = max(min_v, min(int_val, max_v))
                    flags |= EvalFlags.FLOAT_OOB_CAST | EvalFlags.SATURATED

                return EvalResult(value=int_val, flags=flags, result_type=dst_type)

        if kind in (CastKind.SITOFP, CastKind.UITOFP):
            fval = float(val)
            if isinstance(dst_type, FloatType) and dst_type.kind is FloatKind.F32:
                fval = self._clamp_f32(fval)
            return EvalResult(value=fval, result_type=dst_type)

        if kind is CastKind.BITCAST:
            return EvalResult(value=val, result_type=dst_type,
                              description="Bitcast: raw bits reinterpreted")

        if kind is CastKind.PTRTOINT:
            return EvalResult(value=int(val), result_type=dst_type)

        if kind is CastKind.INTTOPTR:
            return EvalResult(value=val, result_type=dst_type)

        return EvalResult(value=None, description=f"Unknown cast kind: {kind}")

    # ── Select evaluation ────────────────────────────────────────────────

    def _eval_select(self, inst: SelectInst, vals: dict[int, Any]) -> EvalResult:
        cond = self._resolve_value(inst.condition, vals)
        tv = self._resolve_value(inst.true_value, vals)
        fv = self._resolve_value(inst.false_value, vals)
        if cond is None:
            return EvalResult(value=None, description="Missing condition")
        result = tv if cond else fv
        return EvalResult(value=result, result_type=inst.type)

    # ── Memory evaluation ────────────────────────────────────────────────

    def _eval_memory(self, inst: Instruction, vals: dict[int, Any]) -> EvalResult:
        if isinstance(inst, LoadInst):
            ptr_val = self._resolve_value(inst.address, vals)
            if ptr_val is not None and ptr_val == 0:
                return EvalResult.bottom("Null pointer dereference",
                                         EvalFlags.NULL_DEREF | EvalFlags.UB_TRIGGERED)
            return EvalResult(value=None, description="Load requires memory model")

        if isinstance(inst, StoreInst):
            ptr_val = self._resolve_value(inst.address, vals)
            if ptr_val is not None and ptr_val == 0:
                return EvalResult.bottom("Null pointer store",
                                         EvalFlags.NULL_DEREF | EvalFlags.UB_TRIGGERED)
            return EvalResult(value=None, result_type=VoidType(),
                              description="Store requires memory model")

        return EvalResult(value=None, description="Unknown memory operation")

    # ── Return evaluation ────────────────────────────────────────────────

    def _eval_return(self, inst: ReturnInst, vals: dict[int, Any]) -> EvalResult:
        if inst.is_void_return:
            return EvalResult(value=None, result_type=VoidType())
        ret_val = inst.return_value
        if ret_val is not None:
            v = self._resolve_value(ret_val, vals)
            return EvalResult(value=v, result_type=ret_val.type)
        return EvalResult(value=None, result_type=VoidType())

    # ── Abstract (interval) evaluation ───────────────────────────────────

    def _eval_binop_abstract(
        self, inst: BinaryOp, intervals: dict[int, Interval],
    ) -> Tuple[Interval | None, EvalFlags]:
        lhs_iv = self._resolve_interval(inst.lhs, intervals)
        rhs_iv = self._resolve_interval(inst.rhs, intervals)
        if lhs_iv is None or rhs_iv is None:
            return None, EvalFlags.NONE

        flags = EvalFlags.NONE
        op = inst.op

        if op is BinOpKind.ADD:
            result_iv = lhs_iv.add(rhs_iv)
        elif op is BinOpKind.SUB:
            result_iv = lhs_iv.sub(rhs_iv)
        elif op is BinOpKind.MUL:
            result_iv = lhs_iv.mul(rhs_iv)
        elif op in (BinOpKind.SDIV, BinOpKind.UDIV):
            if rhs_iv.contains(0):
                flags |= EvalFlags.DIVISION_BY_ZERO
                return None, flags
            # Conservative: compute bounds
            corners = []
            for l in (lhs_iv.lo, lhs_iv.hi):
                for r in (rhs_iv.lo, rhs_iv.hi):
                    if r != 0:
                        corners.append(int(l / r))
            if not corners:
                return None, flags
            result_iv = Interval(min(corners), max(corners))
        elif op in (BinOpKind.AND,):
            # Very conservative
            lo = 0
            hi = min(lhs_iv.hi, rhs_iv.hi) if lhs_iv.hi >= 0 and rhs_iv.hi >= 0 else max(lhs_iv.hi, rhs_iv.hi)
            result_iv = Interval(lo, max(lo, hi))
        elif op in (BinOpKind.OR,):
            result_iv = Interval(
                min(lhs_iv.lo, rhs_iv.lo),
                max(lhs_iv.hi, rhs_iv.hi),
            )
        elif op in (BinOpKind.SHL,):
            if rhs_iv.lo < 0:
                flags |= EvalFlags.SHIFT_OVERFLOW
                return None, flags
            if isinstance(inst.lhs.type, IntType) and rhs_iv.hi >= inst.lhs.type.width:
                flags |= EvalFlags.SHIFT_OVERFLOW
            lo = lhs_iv.lo << max(0, rhs_iv.lo)
            hi = lhs_iv.hi << max(0, rhs_iv.hi)
            result_iv = Interval(min(lo, hi), max(lo, hi))
        else:
            return None, EvalFlags.NONE

        # Check for overflow
        if op in (BinOpKind.ADD, BinOpKind.SUB, BinOpKind.MUL):
            if isinstance(inst.lhs.type, IntType):
                int_type = inst.lhs.type
                if not result_iv.fits_in(int_type.width, int_type.is_signed):
                    flags |= EvalFlags.OVERFLOW

        return result_iv, flags

    def _eval_unop_abstract(
        self, inst: UnaryOp, intervals: dict[int, Interval],
    ) -> Tuple[Interval | None, EvalFlags]:
        operand = inst.operands[0]
        iv = self._resolve_interval(operand, intervals)
        if iv is None:
            return None, EvalFlags.NONE

        if inst.op is UnaryOpKind.NEG:
            result = iv.negate()
            flags = EvalFlags.NONE
            if isinstance(operand.type, IntType):
                int_type = operand.type
                min_val = int_type.min_value
                if iv.contains(min_val) and int_type.is_signed:
                    flags |= EvalFlags.OVERFLOW
            return result, flags

        if inst.op is UnaryOpKind.BITWISE_NOT:
            return Interval(~iv.hi, ~iv.lo), EvalFlags.NONE

        return None, EvalFlags.NONE

    def _eval_cast_abstract(
        self, inst: CastInst, intervals: dict[int, Interval],
    ) -> Tuple[Interval | None, EvalFlags]:
        iv = self._resolve_interval(inst.operand, intervals)
        if iv is None:
            return None, EvalFlags.NONE

        dst = inst.dest_type
        flags = EvalFlags.NONE

        if inst.cast_kind is CastKind.TRUNC:
            if isinstance(dst, IntType):
                max_v = dst.max_value
                min_v = dst.min_value
                if iv.lo < min_v or iv.hi > max_v:
                    flags |= EvalFlags.NARROWING
                    return Interval(min_v, max_v), flags
                return Interval(max(iv.lo, min_v), min(iv.hi, max_v)), flags

        if inst.cast_kind in (CastKind.ZEXT, CastKind.SEXT):
            return iv, flags

        return None, flags

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_value(self, value: Value, vals: dict[int, Any]) -> Any | None:
        if value.id in vals:
            return vals[value.id]
        if isinstance(value, Constant):
            return value.value
        return None

    def _resolve_interval(
        self, value: Value, intervals: dict[int, Interval],
    ) -> Interval | None:
        if value.id in intervals:
            return intervals[value.id]
        if isinstance(value, Constant) and isinstance(value.value, int):
            return Interval.point(value.value)
        return None

    @staticmethod
    def _clamp_f32(val: float) -> float:
        """Clamp a float value to f32 precision."""
        if math.isnan(val) or math.isinf(val):
            return val
        try:
            packed = struct.pack('f', val)
            return struct.unpack('f', packed)[0]
        except (OverflowError, struct.error):
            return float('inf') if val > 0 else float('-inf')


# ── Convenience ──────────────────────────────────────────────────────────

def eval_under_c11(inst: Instruction, vals: dict[int, Any] | None = None) -> EvalResult:
    """Quick-evaluate an instruction under C11 semantics."""
    return SemanticEvaluator(SemanticConfig.c11()).evaluate(inst, vals)


def eval_under_rust(
    inst: Instruction,
    vals: dict[int, Any] | None = None,
    debug: bool = False,
) -> EvalResult:
    """Quick-evaluate under Rust semantics."""
    cfg = SemanticConfig.rust_debug() if debug else SemanticConfig.rust_release()
    return SemanticEvaluator(cfg).evaluate(inst, vals)


def compare_eval(
    inst: Instruction,
    vals: dict[int, Any],
    c_config: SemanticConfig | None = None,
    rust_config: SemanticConfig | None = None,
) -> Tuple[EvalResult, EvalResult]:
    """Evaluate under both C and Rust configs and return both results."""
    c_cfg = c_config or SemanticConfig.c11()
    r_cfg = rust_config or SemanticConfig.rust_release()
    c_result = SemanticEvaluator(c_cfg).evaluate(inst, vals)
    r_result = SemanticEvaluator(r_cfg).evaluate(inst, vals)
    return c_result, r_result
