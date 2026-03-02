"""
SMT encoding of IR types and operations into Z3.

Encodes integer operations using QF_BV, floating-point using QF_FP,
and memory using QF_ABV. Handles overflow modes, pointer operations,
and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Any, Set

import z3

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    ArrayType, StructType, EnumType, FunctionType, UnionType,
    Signedness, FloatKind, OverflowBehavior, Language,
)
from ..ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, CastInst,
    LoadInst, StoreInst, CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst, AllocaInst, GetElementPtrInst,
    ExtractValueInst, InsertValueInst, SwitchInst,
    Value, Constant, Argument, BinOpKind, CmpPredicate, CastKind,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from ..semantics.semantic_config import SemanticConfig, OverflowMode


# ---------------------------------------------------------------------------
# Encoding context
# ---------------------------------------------------------------------------

@dataclass
class EncodingContext:
    """Context for SMT encoding, tracking declarations and assertions."""
    declarations: Dict[str, z3.ExprRef] = field(default_factory=dict)
    assertions: List[z3.BoolRef] = field(default_factory=list)
    assumptions: List[z3.BoolRef] = field(default_factory=list)
    pointer_width: int = 64
    _fresh_counter: int = 0
    _alloca_values: Dict[str, z3.ExprRef] = field(default_factory=dict)

    def declare(self, name: str, sort: z3.SortRef) -> z3.ExprRef:
        """Declare a new symbolic constant."""
        if name in self.declarations:
            return self.declarations[name]
        if sort == z3.BoolSort():
            c = z3.Bool(name)
        elif z3.is_bv_sort(sort):
            c = z3.BitVec(name, sort.size())
        elif isinstance(sort, z3.FPSortRef):
            c = z3.FP(name, sort)
        else:
            c = z3.Const(name, sort)
        self.declarations[name] = c
        return c

    def declare_bv(self, name: str, width: int) -> z3.BitVecRef:
        return self.declare(name, z3.BitVecSort(width))

    def declare_fp(self, name: str, ebits: int = 11, sbits: int = 53) -> z3.FPRef:
        return self.declare(name, z3.FPSort(ebits, sbits))

    def declare_bool(self, name: str) -> z3.BoolRef:
        return self.declare(name, z3.BoolSort())

    def declare_array(self, name: str, idx_sort: z3.SortRef, val_sort: z3.SortRef) -> z3.ArrayRef:
        c = z3.Array(name, idx_sort, val_sort)
        self.declarations[name] = c
        return c

    def assert_hard(self, expr: z3.BoolRef) -> None:
        self.assertions.append(expr)

    def assert_assume(self, expr: z3.BoolRef) -> None:
        self.assumptions.append(expr)

    def fresh(self, prefix: str = "t") -> str:
        self._fresh_counter += 1
        return f"{prefix}_{self._fresh_counter}"

    def get(self, name: str) -> Optional[z3.ExprRef]:
        return self.declarations.get(name)


# ---------------------------------------------------------------------------
# SMT Encoder
# ---------------------------------------------------------------------------

class SMTEncoder:
    """
    Encodes IR types and operations into Z3 expressions.
    
    Uses:
    - QF_BV for integer operations (bitvectors of correct width)
    - QF_FP for floats (IEEE 754 with proper rounding modes)
    - QF_ABV for memory (arrays of bitvectors)
    """

    def __init__(
        self,
        config: Optional[SemanticConfig] = None,
        pointer_width: int = 64,
        loop_unroll_bound: int = 32,
    ):
        self.config = config
        self.pointer_width = pointer_width
        self.loop_unroll_bound = loop_unroll_bound

    # -- Type encoding --

    def encode_type(self, ty: IRType) -> z3.SortRef:
        """Encode an IR type as a Z3 sort."""
        tname = type(ty).__name__
        if isinstance(ty, IntType) or tname == "IntType":
            return z3.BitVecSort(getattr(ty, 'width', 32))
        if isinstance(ty, FloatType) or tname == "FloatType":
            if getattr(ty, 'kind', None) == FloatKind.F32:
                return z3.FPSort(8, 24)
            return z3.FPSort(11, 53)
        if isinstance(ty, PointerType) or tname == "PointerType":
            return z3.BitVecSort(self.pointer_width)
        if isinstance(ty, VoidType) or tname == "VoidType":
            return z3.BoolSort()
        if isinstance(ty, ArrayType) or tname == "ArrayType":
            elem_sort = self.encode_type(ty.element)
            return z3.ArraySort(z3.BitVecSort(64), elem_sort)
        if isinstance(ty, StructType) or tname == "StructType":
            # Encode struct as flat bitvector of total size
            total_bits = ty.size_bits(self.pointer_width)
            return z3.BitVecSort(max(total_bits, 8))
        if isinstance(ty, EnumType) or tname == "EnumType":
            # Encode enum as flat bitvector: tag + max payload
            total_bits = ty.size_bits(self.pointer_width)
            return z3.BitVecSort(max(total_bits, 8))
        if isinstance(ty, UnionType) or tname == "UnionType":
            total_bits = ty.size_bits(self.pointer_width)
            return z3.BitVecSort(max(total_bits, 8))
        # Default
        return z3.BitVecSort(32)

    def encode_constant(self, const: Constant, ctx: EncodingContext) -> z3.ExprRef:
        """Encode a constant value."""
        ty = const.type
        val = const.value
        tname = type(ty).__name__

        if isinstance(ty, IntType) or tname == "IntType":
            width = getattr(ty, 'width', 32)
            if val is None:
                return z3.BitVecVal(0, width)
            if isinstance(val, bool):
                val = 1 if val else 0
            return z3.BitVecVal(int(val) & ((1 << width) - 1), width)
        if isinstance(ty, FloatType) or tname == "FloatType":
            sort = self.encode_type(ty)
            return z3.FPVal(float(val), sort)
        if isinstance(ty, PointerType) or tname == "PointerType":
            return z3.BitVecVal(int(val or 0) & ((1 << self.pointer_width) - 1), self.pointer_width)
        return z3.BitVecVal(0, 32)

    def encode_value(self, val: Value, ctx: EncodingContext) -> z3.ExprRef:
        """Encode an IR value, looking up or creating a Z3 expression."""
        # Handle Constants (including cross-module import mismatches)
        if isinstance(val, Constant) or type(val).__name__ == "Constant":
            return self.encode_constant(val, ctx)

        name = val.name or f"v_{val.id}"
        existing = ctx.get(name)
        if existing is not None:
            return existing

        sort = self.encode_type(val.type) if val.type else z3.BitVecSort(32)
        return ctx.declare(name, sort)

    # -- Instruction encoding --

    def encode_instruction(
        self,
        inst: Instruction,
        ctx: EncodingContext,
    ) -> Optional[z3.ExprRef]:
        """Encode an instruction and return its result expression.

        Uses both isinstance checks and type-name fallbacks to handle
        classes imported from different module paths (e.g. ``ir.instructions``
        vs ``src.ir.instructions``).
        """
        tname = type(inst).__name__
        if isinstance(inst, BinaryOp) or tname == "BinaryOp":
            return self.encode_binop(inst, ctx)
        if isinstance(inst, UnaryOp) or tname == "UnaryOp":
            return self.encode_unaryop(inst, ctx)
        if isinstance(inst, CompareOp) or tname == "CompareOp":
            return self.encode_compare(inst, ctx)
        if isinstance(inst, CastInst) or tname == "CastInst":
            return self.encode_cast(inst, ctx)
        if isinstance(inst, SelectInst) or tname == "SelectInst":
            return self.encode_select(inst, ctx)
        if isinstance(inst, PhiInst) or tname == "PhiInst":
            return self.encode_phi(inst, ctx)
        if isinstance(inst, LoadInst) or tname == "LoadInst":
            return self.encode_load(inst, ctx)
        if isinstance(inst, StoreInst) or tname == "StoreInst":
            self.encode_store(inst, ctx)
            return None
        if isinstance(inst, CallInst) or tname == "CallInst":
            return self.encode_call(inst, ctx)
        if isinstance(inst, ReturnInst) or tname == "ReturnInst":
            return self.encode_return(inst, ctx)
        if isinstance(inst, ExtractValueInst) or tname == "ExtractValueInst":
            return self.encode_extract_value(inst, ctx)
        if isinstance(inst, InsertValueInst) or tname == "InsertValueInst":
            return self.encode_insert_value(inst, ctx)
        if isinstance(inst, SwitchInst) or tname == "SwitchInst":
            return self.encode_switch(inst, ctx)
        # Unknown instruction: create unconstrained
        if inst.name and inst.type:
            sort = self.encode_type(inst.type)
            return ctx.declare(inst.name, sort)
        return None

    # -- Binary operations --

    def encode_binop(self, inst: BinaryOp, ctx: EncodingContext) -> z3.ExprRef:
        """Encode a binary operation with overflow mode handling."""
        lhs = self.encode_value(inst.lhs, ctx)
        rhs = self.encode_value(inst.rhs, ctx)

        # Coerce to same width
        if z3.is_bv(lhs) and z3.is_bv(rhs):
            lhs, rhs = self._coerce_bv(lhs, rhs)

        op = inst.op.name
        result = self._encode_binop_core(op, lhs, rhs, inst.type, ctx)

        # Determine overflow mode
        overflow_behavior = self._get_overflow_behavior(inst)

        # Add overflow checks based on mode
        if overflow_behavior == OverflowBehavior.WRAP:
            pass  # Default BV semantics already wrap
        elif overflow_behavior == OverflowBehavior.UNDEFINED:
            # Add overflow as unconstrained behavior
            if op in ("ADD", "SUB", "MUL") and z3.is_bv(lhs):
                overflow_cond = self._signed_overflow_check(op, lhs, rhs)
                if overflow_cond is not None:
                    ctx.assert_assume(z3.Not(overflow_cond))
            # Division by zero and INT_MIN/-1 are UB in C
            if op in ("SDIV", "SREM") and z3.is_bv(rhs):
                width = rhs.size()
                ctx.assert_assume(rhs != z3.BitVecVal(0, width))
                min_val = z3.BitVecVal(-(1 << (width - 1)), width)
                neg_one = z3.BitVecVal(-1, width)
                ctx.assert_assume(z3.Not(z3.And(lhs == min_val, rhs == neg_one)))
            if op in ("UDIV", "UREM") and z3.is_bv(rhs):
                width = rhs.size()
                ctx.assert_assume(rhs != z3.BitVecVal(0, width))
            # Shift by >= width is UB in C
            if op in ("SHL", "LSHR", "ASHR") and z3.is_bv(rhs):
                width = rhs.size()
                ctx.assert_assume(z3.ULT(rhs, z3.BitVecVal(width, width)))
        elif overflow_behavior == OverflowBehavior.TRAP:
            if op in ("ADD", "SUB", "MUL") and z3.is_bv(lhs):
                overflow_cond = self._signed_overflow_check(op, lhs, rhs)
                if overflow_cond is not None:
                    ctx.assert_assume(z3.Not(overflow_cond))
        elif overflow_behavior == OverflowBehavior.SATURATE:
            if op in ("ADD", "SUB", "MUL") and z3.is_bv(lhs):
                width = lhs.size()
                # Check if unsigned type
                inst_type = inst.type
                is_unsigned = False
                if inst_type:
                    tname = type(inst_type).__name__
                    if tname == "IntType" or isinstance(inst_type, IntType):
                        is_unsigned = not getattr(inst_type, 'is_signed', True)

                if is_unsigned:
                    # Unsigned saturating: clamp to MAX
                    max_val = z3.BitVecVal((1 << width) - 1, width)
                    if op == "ADD":
                        # Overflow when a + b < a (unsigned wrap)
                        overflow_cond = z3.ULT(result, lhs)
                        result = z3.If(overflow_cond, max_val, result)
                    elif op == "SUB":
                        # Underflow when a < b
                        overflow_cond = z3.ULT(lhs, rhs)
                        result = z3.If(overflow_cond, z3.BitVecVal(0, width), result)
                else:
                    # Signed saturating
                    overflow_cond = self._signed_overflow_check(op, lhs, rhs)
                    if overflow_cond is not None:
                        max_val = z3.BitVecVal((1 << (width - 1)) - 1, width)
                        min_val = z3.BitVecVal(-(1 << (width - 1)), width)
                        result = z3.If(
                            overflow_cond,
                            z3.If(lhs > z3.BitVecVal(0, width), max_val, min_val),
                            result,
                        )
        elif overflow_behavior == OverflowBehavior.CHECKED:
            # Checked operations: encode (result, overflow_flag) as a
            # wider bitvector where the top bit is the overflow flag.
            if op in ("ADD", "SUB", "MUL") and z3.is_bv(lhs):
                width = lhs.size()
                overflow_cond = self._signed_overflow_check(op, lhs, rhs)
                if overflow_cond is not None:
                    flag_bv = z3.If(overflow_cond,
                                    z3.BitVecVal(1, width),
                                    z3.BitVecVal(0, width))
                    # Store the overflow flag so callers can extract it
                    if inst.name:
                        ctx.declarations[f"{inst.name}_overflow"] = overflow_cond
                        ctx.declarations[f"{inst.name}_flag"] = flag_bv

        # Store result
        if inst.name:
            ctx.declarations[inst.name] = result

        return result

    def _encode_binop_core(
        self, op: str, lhs: z3.ExprRef, rhs: z3.ExprRef,
        result_type: Optional[IRType], ctx: EncodingContext,
    ) -> z3.ExprRef:
        """Core binary operation encoding."""
        if op == "ADD":
            return lhs + rhs
        elif op == "SUB":
            return lhs - rhs
        elif op == "MUL":
            return lhs * rhs
        elif op == "SDIV":
            return lhs / rhs
        elif op == "UDIV":
            return z3.UDiv(lhs, rhs)
        elif op == "SREM":
            return z3.SRem(lhs, rhs)
        elif op == "UREM":
            return z3.URem(lhs, rhs)
        elif op == "SHL":
            return lhs << rhs
        elif op == "LSHR":
            return z3.LShR(lhs, rhs)
        elif op == "ASHR":
            return lhs >> rhs
        elif op == "AND":
            return lhs & rhs
        elif op == "OR":
            return lhs | rhs
        elif op == "XOR":
            return lhs ^ rhs
        elif op == "FADD":
            return self._fp_op("add", lhs, rhs, result_type)
        elif op == "FSUB":
            return self._fp_op("sub", lhs, rhs, result_type)
        elif op == "FMUL":
            return self._fp_op("mul", lhs, rhs, result_type)
        elif op == "FDIV":
            return self._fp_op("div", lhs, rhs, result_type)
        elif op == "FREM":
            return self._fp_op("rem", lhs, rhs, result_type)
        else:
            # Unknown: return lhs as fallback
            return lhs

    def _fp_op(
        self, op: str, lhs: z3.ExprRef, rhs: z3.ExprRef,
        result_type: Optional[IRType],
    ) -> z3.ExprRef:
        """Floating-point binary operation."""
        lhs_fp = self._ensure_fp(lhs, result_type)
        rhs_fp = self._ensure_fp(rhs, result_type)
        rm = z3.RNE()

        if op == "add":
            return z3.fpAdd(rm, lhs_fp, rhs_fp)
        elif op == "sub":
            return z3.fpSub(rm, lhs_fp, rhs_fp)
        elif op == "mul":
            return z3.fpMul(rm, lhs_fp, rhs_fp)
        elif op == "div":
            return z3.fpDiv(rm, lhs_fp, rhs_fp)
        elif op == "rem":
            return z3.fpRem(lhs_fp, rhs_fp)
        return lhs_fp

    def _ensure_fp(self, expr: z3.ExprRef, ty: Optional[IRType]) -> z3.FPRef:
        if z3.is_fp(expr):
            return expr
        sort = z3.FPSort(11, 53)
        if isinstance(ty, FloatType) and ty.kind == FloatKind.F32:
            sort = z3.FPSort(8, 24)
        if z3.is_bv(expr):
            return z3.fpBVToFP(expr, sort)
        return z3.FPVal(0.0, sort)

    def _signed_overflow_check(
        self, op: str, lhs: z3.BitVecRef, rhs: z3.BitVecRef,
    ) -> Optional[z3.BoolRef]:
        """Generate signed overflow check condition."""
        width = lhs.size()
        zero = z3.BitVecVal(0, width)

        if op == "ADD":
            result = lhs + rhs
            pos_overflow = z3.And(lhs > zero, rhs > zero, result < zero)
            neg_overflow = z3.And(lhs < zero, rhs < zero, result > zero)
            return z3.Or(pos_overflow, neg_overflow)
        elif op == "SUB":
            result = lhs - rhs
            pos_overflow = z3.And(lhs > zero, rhs < zero, result < zero)
            neg_overflow = z3.And(lhs < zero, rhs > zero, result > zero)
            return z3.Or(pos_overflow, neg_overflow)
        elif op == "MUL":
            ext_lhs = z3.SignExt(width, lhs)
            ext_rhs = z3.SignExt(width, rhs)
            full_product = ext_lhs * ext_rhs
            truncated = z3.Extract(width - 1, 0, full_product)
            sign_ext_back = z3.SignExt(width, truncated)
            return full_product != sign_ext_back
        return None

    def _get_overflow_behavior(self, inst) -> OverflowBehavior:
        """Determine overflow behavior from instruction metadata.

        Instruction metadata takes priority over config defaults, since
        methods like wrapping_add/saturating_add explicitly set overflow mode.
        """
        # Check instruction metadata first (explicit method annotations)
        if inst.metadata and inst.metadata.overflow:
            overflow_val = inst.metadata.overflow
            if isinstance(overflow_val, OverflowBehavior):
                return overflow_val
            # Handle cross-module enum by name
            oval_name = getattr(overflow_val, 'name', None) or str(overflow_val)
            try:
                return OverflowBehavior[oval_name.upper().replace("OVERFLOWBEHAVIOR.", "")]
            except (KeyError, AttributeError):
                pass

        # Fall back to config-based defaults
        if self.config is not None:
            from ..semantics.semantic_config import OverflowMode
            overflow_mode = getattr(self.config, 'signed_overflow', None)
            if overflow_mode is not None:
                oval_name = getattr(overflow_mode, 'name', str(overflow_mode))
                if oval_name == 'UB':
                    if inst.type and type(inst.type).__name__ == "IntType":
                        is_signed = getattr(inst.type, 'is_signed', False)
                        if is_signed:
                            return OverflowBehavior.UNDEFINED
                elif oval_name == 'Wrap':
                    if inst.type and type(inst.type).__name__ == "IntType":
                        return OverflowBehavior.WRAP
                pass

        # Default based on type
        if inst.type and (isinstance(inst.type, IntType)
                          or type(inst.type).__name__ == "IntType"):
            signed = getattr(inst.type, 'signed', None)
            if signed is None:
                signed = getattr(inst.type, 'is_signed', False)
            if signed:
                return OverflowBehavior.UNDEFINED
            return OverflowBehavior.WRAP
        return OverflowBehavior.WRAP

    # -- Unary operations --

    def encode_unaryop(self, inst: UnaryOp, ctx: EncodingContext) -> z3.ExprRef:
        operand = self.encode_value(inst._operands[0], ctx)
        op = inst.op.name

        if op == "NEG":
            result = -operand
        elif op == "NOT":
            result = z3.Not(operand) if z3.is_bool(operand) else ~operand
        elif op == "BITWISE_NOT":
            result = ~operand
        elif op == "FNEG":
            result = z3.fpNeg(operand) if z3.is_fp(operand) else -operand
        else:
            result = operand

        # Add overflow assumptions for unary negation
        if op == "NEG" and z3.is_bv(operand):
            overflow_behavior = self._get_overflow_behavior(inst)
            if overflow_behavior == OverflowBehavior.UNDEFINED:
                width = operand.size()
                int_min = z3.BitVecVal(-(1 << (width - 1)), width)
                # NEG of INT_MIN overflows
                ctx.assert_assume(operand != int_min)

        if inst.name:
            ctx.declarations[inst.name] = result
        return result

    # -- Comparison --

    def encode_compare(self, inst: CompareOp, ctx: EncodingContext) -> z3.BoolRef:
        lhs = self.encode_value(inst.lhs, ctx)
        rhs = self.encode_value(inst.rhs, ctx)

        if z3.is_bv(lhs) and z3.is_bv(rhs):
            lhs, rhs = self._coerce_bv(lhs, rhs)

        pred = inst.predicate.name
        result = self._encode_predicate(pred, lhs, rhs)

        if inst.name:
            ctx.declarations[inst.name] = result
        return result

    def _encode_predicate(
        self, pred: str, lhs: z3.ExprRef, rhs: z3.ExprRef,
    ) -> z3.BoolRef:
        if pred in ("EQ", "OEQ", "UEQ"):
            return lhs == rhs
        if pred in ("NE", "ONE", "UNE"):
            return lhs != rhs
        if pred in ("SLT", "OLT"):
            return lhs < rhs if z3.is_bv(lhs) else z3.fpLT(lhs, rhs)
        if pred in ("SLE", "OLE"):
            return lhs <= rhs if z3.is_bv(lhs) else z3.fpLEQ(lhs, rhs)
        if pred in ("SGT", "OGT"):
            return lhs > rhs if z3.is_bv(lhs) else z3.fpGT(lhs, rhs)
        if pred in ("SGE", "OGE"):
            return lhs >= rhs if z3.is_bv(lhs) else z3.fpGEQ(lhs, rhs)
        if pred == "ULT":
            return z3.ULT(lhs, rhs)
        if pred == "ULE":
            return z3.ULE(lhs, rhs)
        if pred == "UGT":
            return z3.UGT(lhs, rhs)
        if pred == "UGE":
            return z3.UGE(lhs, rhs)
        if pred == "ORD":
            return z3.And(z3.Not(z3.fpIsNaN(lhs)), z3.Not(z3.fpIsNaN(rhs)))
        if pred == "UNO":
            return z3.Or(z3.fpIsNaN(lhs), z3.fpIsNaN(rhs))
        return lhs == rhs

    # -- Cast --

    def encode_cast(self, inst: CastInst, ctx: EncodingContext) -> z3.ExprRef:
        operand = self.encode_value(inst._operands[0], ctx)
        dst_type = inst.type
        kind = inst.cast_kind.name

        if kind == "TRUNC":
            w = dst_type.width if isinstance(dst_type, IntType) else 32
            result = z3.Extract(w - 1, 0, operand) if z3.is_bv(operand) and operand.size() > w else operand
        elif kind == "ZEXT":
            w = dst_type.width if isinstance(dst_type, IntType) else 64
            if z3.is_bv(operand):
                ext = w - operand.size()
                result = z3.ZeroExt(max(ext, 0), operand) if ext > 0 else operand
            else:
                result = operand
        elif kind == "SEXT":
            w = dst_type.width if isinstance(dst_type, IntType) else 64
            if z3.is_bv(operand):
                ext = w - operand.size()
                result = z3.SignExt(max(ext, 0), operand) if ext > 0 else operand
            else:
                result = operand
        elif kind == "FPTRUNC":
            sort = z3.FPSort(8, 24)
            result = z3.fpFPToFP(z3.RNE(), operand, sort) if z3.is_fp(operand) else z3.FPVal(0.0, sort)
        elif kind == "FPEXT":
            sort = z3.FPSort(11, 53)
            result = z3.fpFPToFP(z3.RNE(), operand, sort) if z3.is_fp(operand) else z3.FPVal(0.0, sort)
        elif kind == "FPTOSI":
            w = dst_type.width if isinstance(dst_type, IntType) else 32
            result = z3.fpToSBV(z3.RTZ(), operand, z3.BitVecSort(w)) if z3.is_fp(operand) else z3.BitVecVal(0, w)
        elif kind == "FPTOUI":
            w = dst_type.width if isinstance(dst_type, IntType) else 32
            result = z3.fpToUBV(z3.RTZ(), operand, z3.BitVecSort(w)) if z3.is_fp(operand) else z3.BitVecVal(0, w)
        elif kind == "SITOFP":
            sort = z3.FPSort(11, 53)
            if isinstance(dst_type, FloatType) and dst_type.kind == FloatKind.F32:
                sort = z3.FPSort(8, 24)
            result = z3.fpSignedToFP(z3.RNE(), operand, sort) if z3.is_bv(operand) else z3.FPVal(0.0, sort)
        elif kind == "UITOFP":
            sort = z3.FPSort(11, 53)
            if isinstance(dst_type, FloatType) and dst_type.kind == FloatKind.F32:
                sort = z3.FPSort(8, 24)
            result = z3.fpToFP(z3.RNE(), operand, sort) if z3.is_bv(operand) else z3.FPVal(0.0, sort)
        elif kind == "BITCAST":
            result = operand
        elif kind == "PTRTOINT":
            w = dst_type.width if isinstance(dst_type, IntType) else self.pointer_width
            if z3.is_bv(operand):
                if operand.size() > w:
                    result = z3.Extract(w - 1, 0, operand)
                elif operand.size() < w:
                    result = z3.ZeroExt(w - operand.size(), operand)
                else:
                    result = operand
            else:
                result = z3.BitVecVal(0, w)
        elif kind == "INTTOPTR":
            if z3.is_bv(operand):
                if operand.size() > self.pointer_width:
                    result = z3.Extract(self.pointer_width - 1, 0, operand)
                elif operand.size() < self.pointer_width:
                    result = z3.ZeroExt(self.pointer_width - operand.size(), operand)
                else:
                    result = operand
            else:
                result = z3.BitVecVal(0, self.pointer_width)
        else:
            result = operand

        if inst.name:
            ctx.declarations[inst.name] = result
        return result

    # -- Select --

    def encode_select(self, inst: SelectInst, ctx: EncodingContext) -> z3.ExprRef:
        cond = self.encode_value(inst._operands[0], ctx)
        true_val = self.encode_value(inst._operands[1], ctx)
        false_val = self.encode_value(inst._operands[2], ctx)

        if z3.is_bv(cond):
            cond = cond != z3.BitVecVal(0, cond.size())

        if z3.is_bv(true_val) and z3.is_bv(false_val):
            true_val, false_val = self._coerce_bv(true_val, false_val)

        result = z3.If(cond, true_val, false_val)
        if inst.name:
            ctx.declarations[inst.name] = result
        return result

    # -- Phi --

    def encode_phi(self, inst: PhiInst, ctx: EncodingContext) -> z3.ExprRef:
        """Encode phi node using ITE chain over predecessor conditions.

        Uses branch conditions recorded during function encoding to determine
        which incoming value is selected.
        """
        tname = type(inst.type).__name__ if inst.type else "None"
        # If type is void, infer from incoming values; default to BV32
        if inst.type and tname != "VoidType" and not isinstance(inst.type, VoidType):
            sort = self.encode_type(inst.type)
        else:
            sort = z3.BitVecSort(32)
        name = inst.name or ctx.fresh("phi")

        incoming = getattr(inst, '_incoming', None)
        if not incoming or len(incoming) == 0:
            return ctx.declare(name, sort)

        # Build ITE chain: if came from pred0 then val0, elif pred1 then val1 ...
        result: Optional[z3.ExprRef] = None
        for val, block in reversed(incoming):
            encoded_val = self.encode_value(val, ctx)
            # Coerce sort mismatches between incoming values
            if result is not None:
                if z3.is_bv(encoded_val) and z3.is_bv(result):
                    if encoded_val.size() != result.size():
                        target_size = max(encoded_val.size(), result.size())
                        if encoded_val.size() < target_size:
                            encoded_val = z3.SignExt(target_size - encoded_val.size(), encoded_val)
                        if result.size() < target_size:
                            result = z3.SignExt(target_size - result.size(), result)
                elif z3.is_bool(encoded_val) and z3.is_bv(result):
                    encoded_val = z3.If(encoded_val, z3.BitVecVal(1, result.size()), z3.BitVecVal(0, result.size()))
                elif z3.is_bv(encoded_val) and z3.is_bool(result):
                    result = z3.If(result, z3.BitVecVal(1, encoded_val.size()), z3.BitVecVal(0, encoded_val.size()))

            if result is None:
                result = encoded_val
            else:
                # Look up recorded branch condition for this incoming block
                block_name = block.name if hasattr(block, 'name') else None
                pred_cond = ctx.get(f"_branch_cond_{block_name}") if block_name else None

                if pred_cond is not None and z3.is_bool(pred_cond):
                    result = z3.If(pred_cond, encoded_val, result)
                else:
                    # Fallback: unconstrained guard
                    guard = ctx.declare_bool(ctx.fresh(f"phi_guard_{block_name}" if block_name else "phi_guard"))
                    result = z3.If(guard, encoded_val, result)

        if result is not None:
            ctx.declarations[name] = result
            return result
        return ctx.declare(name, sort)

    # -- Memory --

    def encode_load(self, inst: LoadInst, ctx: EncodingContext) -> z3.ExprRef:
        """Encode load, using alloca tracking or Z3 array theory."""
        sort = self.encode_type(inst.type) if inst.type else z3.BitVecSort(32)
        name = inst.name or ctx.fresh("load")

        # Check alloca tracking first (handles the common alloca→store→load
        # pattern without needing a full memory model).
        addr_val = getattr(inst, 'address', None) or (inst._operands[0] if inst._operands else None)
        addr_name = getattr(addr_val, 'name', None) if addr_val else None
        if addr_name and addr_name in ctx._alloca_values:
            result = ctx._alloca_values[addr_name]
            ctx.declarations[name] = result
            return result

        # If we have a memory array and the address is known, model the load
        addr = self.encode_value(inst.address, ctx) if hasattr(inst, 'address') and inst.address else None
        mem = ctx.get("_memory")
        if mem is not None and addr is not None and z3.is_bv(addr):
            result = z3.Select(mem, addr)
            ctx.declarations[name] = result
            return result

        # Fallback: fresh symbolic value
        result = ctx.declare(name, sort)
        return result

    def encode_store(self, inst: StoreInst, ctx: EncodingContext) -> None:
        """Encode store, updating alloca tracking or Z3 array."""
        val = self.encode_value(inst.value, ctx)
        addr_val = getattr(inst, 'address', None) or (inst._operands[1] if len(inst._operands) > 1 else None)
        addr_name = getattr(addr_val, 'name', None) if addr_val else None

        # Track value stored to alloca
        if addr_name:
            ctx._alloca_values[addr_name] = val

        addr = self.encode_value(addr_val, ctx) if addr_val else None

        # Update memory array if present
        mem = ctx.get("_memory")
        if mem is not None and addr is not None and z3.is_bv(addr):
            new_mem = z3.Store(mem, addr, val)
            ctx.declarations["_memory"] = new_mem

    # -- Call --

    def encode_call(self, inst: CallInst, ctx: EncodingContext) -> z3.ExprRef:
        # Calls produce unconstrained return values
        if inst.type and not isinstance(inst.type, VoidType):
            sort = self.encode_type(inst.type)
            name = inst.name or ctx.fresh(f"call_{inst.callee_name}")
            result = ctx.declare(name, sort)
            return result
        return z3.BoolVal(True)

    # -- Struct/Enum aggregate operations --

    def encode_extract_value(self, inst: ExtractValueInst, ctx: EncodingContext) -> z3.ExprRef:
        """Encode extractvalue: extract a field from a struct/enum bitvector.

        For structs, fields are packed into a single bitvector.  Extracting
        field *i* corresponds to ``Extract(hi, lo, aggregate_bv)`` where
        ``lo = field_offset(i)`` and ``hi = lo + field_size - 1``.
        """
        aggregate = self.encode_value(inst.aggregate, ctx)
        result_sort = self.encode_type(inst.type) if inst.type else z3.BitVecSort(32)

        agg_ty = inst.aggregate.type
        if not z3.is_bv(aggregate):
            name = inst.name or ctx.fresh("ev")
            result = ctx.declare(name, result_sort)
            return result

        # Walk the index chain to find the offset and width
        current_ty = agg_ty
        bit_offset = 0
        for idx in inst.indices:
            tname = type(current_ty).__name__
            if isinstance(current_ty, StructType) or tname == "StructType":
                bit_offset += current_ty.field_offset(idx, self.pointer_width)
                current_ty = current_ty.fields[idx].type
            elif isinstance(current_ty, EnumType) or tname == "EnumType":
                # For enum, index 0 = tag, index 1 = payload
                if idx == 0:
                    tw = current_ty._effective_tag_width()
                    current_ty = IntType(tw)
                else:
                    tw = current_ty._effective_tag_width()
                    payload_align = max(
                        (t.align_bits(self.pointer_width) for _, t in current_ty.variants if t.is_sized()),
                        default=8,
                    )
                    from ..ir.types import _align_up
                    bit_offset += _align_up(tw, payload_align)
                    # Payload type depends on context; use result type
                    current_ty = inst.type
            elif isinstance(current_ty, ArrayType) or tname == "ArrayType":
                stride = current_ty.element.size_bits(self.pointer_width)
                align = current_ty.element.align_bits(self.pointer_width)
                from ..ir.types import _align_up
                stride = _align_up(stride, align)
                bit_offset += stride * idx
                current_ty = current_ty.element
            else:
                break

        field_width = current_ty.size_bits(self.pointer_width) if hasattr(current_ty, 'size_bits') else 32
        agg_width = aggregate.size()

        if bit_offset + field_width > agg_width:
            name = inst.name or ctx.fresh("ev")
            result = ctx.declare(name, result_sort)
            return result

        lo = bit_offset
        hi = bit_offset + field_width - 1
        result = z3.Extract(hi, lo, aggregate)

        if inst.name:
            ctx.declarations[inst.name] = result
        return result

    def encode_insert_value(self, inst: InsertValueInst, ctx: EncodingContext) -> z3.ExprRef:
        """Encode insertvalue: insert a value into a struct/enum bitvector.

        Replaces the bits at the field's position with the new value while
        preserving all other bits.
        """
        aggregate = self.encode_value(inst.aggregate, ctx)
        new_val = self.encode_value(inst.inserted_value, ctx)

        agg_ty = inst.aggregate.type
        if not z3.is_bv(aggregate):
            name = inst.name or ctx.fresh("iv")
            sort = self.encode_type(inst.type) if inst.type else z3.BitVecSort(32)
            result = ctx.declare(name, sort)
            return result

        current_ty = agg_ty
        bit_offset = 0
        for idx in inst.indices:
            tname = type(current_ty).__name__
            if isinstance(current_ty, StructType) or tname == "StructType":
                bit_offset += current_ty.field_offset(idx, self.pointer_width)
                current_ty = current_ty.fields[idx].type
            elif isinstance(current_ty, EnumType) or tname == "EnumType":
                if idx == 0:
                    tw = current_ty._effective_tag_width()
                    current_ty = IntType(tw)
                else:
                    tw = current_ty._effective_tag_width()
                    payload_align = max(
                        (t.align_bits(self.pointer_width) for _, t in current_ty.variants if t.is_sized()),
                        default=8,
                    )
                    from ..ir.types import _align_up
                    bit_offset += _align_up(tw, payload_align)
                    current_ty = inst.inserted_value.type if inst.inserted_value.type else IntType(32)
            elif isinstance(current_ty, ArrayType) or tname == "ArrayType":
                stride = current_ty.element.size_bits(self.pointer_width)
                align = current_ty.element.align_bits(self.pointer_width)
                from ..ir.types import _align_up
                stride = _align_up(stride, align)
                bit_offset += stride * idx
                current_ty = current_ty.element
            else:
                break

        field_width = current_ty.size_bits(self.pointer_width) if hasattr(current_ty, 'size_bits') else 32
        agg_width = aggregate.size()

        if bit_offset + field_width > agg_width:
            name = inst.name or ctx.fresh("iv")
            sort = self.encode_type(inst.type) if inst.type else z3.BitVecSort(32)
            result = ctx.declare(name, sort)
            return result

        # Coerce new_val to field width
        if z3.is_bv(new_val):
            if new_val.size() < field_width:
                new_val = z3.ZeroExt(field_width - new_val.size(), new_val)
            elif new_val.size() > field_width:
                new_val = z3.Extract(field_width - 1, 0, new_val)
        else:
            new_val = z3.BitVecVal(0, field_width)

        # Build result: prefix | new_val | suffix
        lo = bit_offset
        hi = bit_offset + field_width - 1

        parts = []
        if hi + 1 < agg_width:
            parts.append(z3.Extract(agg_width - 1, hi + 1, aggregate))
        parts.append(new_val)
        if lo > 0:
            parts.append(z3.Extract(lo - 1, 0, aggregate))

        if len(parts) == 1:
            result = parts[0]
        else:
            result = z3.Concat(*parts)

        if inst.name:
            ctx.declarations[inst.name] = result
        return result

    def encode_switch(self, inst: SwitchInst, ctx: EncodingContext) -> None:
        """Encode a switch instruction as branch conditions for each case.

        Sets path conditions for case targets and the default target,
        mirroring how encode_function handles BranchInst conditions.
        """
        cond = self.encode_value(inst.condition, ctx)
        if not z3.is_bv(cond):
            return None

        case_conds = []
        for case_val, target in inst.cases:
            case_bv = self.encode_constant(case_val, ctx) if isinstance(case_val, Constant) else self.encode_value(case_val, ctx)
            if z3.is_bv(case_bv):
                cond_bv, case_bv = self._coerce_bv(cond, case_bv)
            else:
                cond_bv = cond
            eq_cond = cond_bv == case_bv
            case_conds.append(eq_cond)

            if hasattr(target, 'name') and target.name:
                ctx.declarations[f"_branch_cond_{target.name}"] = eq_cond
                ctx.declarations[f"_path_cond_{target.name}"] = eq_cond

        # Default target gets negation of all case conditions
        if case_conds:
            default_cond = z3.And(*[z3.Not(c) for c in case_conds])
        else:
            default_cond = z3.BoolVal(True)

        default_target = inst.default_target
        if hasattr(default_target, 'name') and default_target.name:
            ctx.declarations[f"_branch_cond_{default_target.name}"] = default_cond
            ctx.declarations[f"_path_cond_{default_target.name}"] = default_cond

        return None

    def encode_struct_literal(
        self,
        struct_ty: StructType,
        field_values: List[z3.ExprRef],
        ctx: EncodingContext,
    ) -> z3.BitVecRef:
        """Construct a struct bitvector from individual field values.

        Fields are laid out low-to-high: field 0 occupies the lowest bits.
        Padding bits between fields are set to zero.
        """
        total_bits = struct_ty.size_bits(self.pointer_width)
        if total_bits == 0:
            return z3.BitVecVal(0, 8)

        result = z3.BitVecVal(0, total_bits)
        offsets = struct_ty._compute_layout(self.pointer_width)

        for i, fld in enumerate(struct_ty.fields):
            if i >= len(field_values):
                break
            fld_width = fld.type.size_bits(self.pointer_width)
            val = field_values[i]

            if z3.is_bv(val):
                if val.size() < fld_width:
                    val = z3.ZeroExt(fld_width - val.size(), val)
                elif val.size() > fld_width:
                    val = z3.Extract(fld_width - 1, 0, val)
            else:
                val = z3.BitVecVal(0, fld_width)

            # Shift field into position and OR into result
            if fld_width < total_bits:
                padded = z3.ZeroExt(total_bits - fld_width, val)
            else:
                padded = val
            if offsets[i] > 0:
                padded = padded << offsets[i]
            result = result | padded

        return result

    def encode_enum_construct(
        self,
        enum_ty: EnumType,
        variant_name: str,
        payload: Optional[z3.ExprRef],
        ctx: EncodingContext,
    ) -> z3.BitVecRef:
        """Construct an enum bitvector with the given variant tag and payload."""
        total_bits = enum_ty.size_bits(self.pointer_width)
        if total_bits == 0:
            return z3.BitVecVal(0, 8)

        tw = enum_ty._effective_tag_width()
        tag_val = enum_ty.variant_index(variant_name)
        result = z3.BitVecVal(0, total_bits)

        # Set tag
        if tw > 0:
            tag_bv = z3.BitVecVal(tag_val, tw)
            if tw < total_bits:
                tag_padded = z3.ZeroExt(total_bits - tw, tag_bv)
            else:
                tag_padded = tag_bv
            result = result | tag_padded

        # Set payload
        if payload is not None and z3.is_bv(payload):
            from ..ir.types import _align_up
            payload_align = max(
                (t.align_bits(self.pointer_width) for _, t in enum_ty.variants if t.is_sized()),
                default=8,
            )
            payload_offset = _align_up(tw, payload_align)
            payload_width = payload.size()
            if payload_width + payload_offset <= total_bits:
                if payload_width < total_bits:
                    payload_padded = z3.ZeroExt(total_bits - payload_width, payload)
                else:
                    payload_padded = payload
                if payload_offset > 0:
                    payload_padded = payload_padded << payload_offset
                result = result | payload_padded

        return result

    def encode_enum_discriminant(
        self,
        enum_bv: z3.BitVecRef,
        enum_ty: EnumType,
    ) -> z3.BitVecRef:
        """Extract the discriminant tag from an enum bitvector."""
        tw = enum_ty._effective_tag_width()
        if tw == 0:
            return z3.BitVecVal(0, 8)
        return z3.Extract(tw - 1, 0, enum_bv)

    # -- Return --

    def encode_return(self, inst: ReturnInst, ctx: EncodingContext) -> Optional[z3.ExprRef]:
        if inst.return_value is not None:
            return self.encode_value(inst.return_value, ctx)
        return None

    # -- Function encoding --

    def encode_function(
        self,
        func: Function,
        ctx: Optional[EncodingContext] = None,
        prefix: str = "",
    ) -> Tuple[EncodingContext, Optional[z3.ExprRef]]:
        """Encode an entire function, returning context and return expression.

        When ``loop_unroll_bound`` > 0 (default 32), blocks with back-edges
        are unrolled K times.  Each unrolled iteration gets fresh SSA names,
        and phi nodes are threaded: the phi's incoming value from the
        back-edge predecessor in iteration *i* is wired to the definition
        produced in iteration *i-1*.  This is standard bounded model
        checking (BMC) loop unrolling.
        """
        if ctx is None:
            ctx = EncodingContext(pointer_width=self.pointer_width)

        # Encode arguments
        for arg in func.arguments:
            name = f"{prefix}{arg.name or f'arg_{arg.index}'}"
            sort = self.encode_type(arg.type) if arg.type else z3.BitVecSort(32)
            ctx.declare(name, sort)

        # Detect back-edges for loop unrolling
        block_list = list(func.blocks)
        block_indices = {id(b): i for i, b in enumerate(block_list)}

        # Identify loop headers (blocks with a predecessor whose index >= own)
        # Only consider it a back-edge if the target has a phi node (loop header trait)
        loop_headers: Set[int] = set()  # block indices
        back_edge_preds: Dict[int, List[int]] = {}  # header_idx -> [pred block indices]
        for idx, block in enumerate(block_list):
            for succ in block.successors:
                succ_idx = block_indices.get(id(succ))
                if succ_idx is not None and succ_idx <= idx:
                    # Only treat as loop back-edge if target has a phi node
                    target_block = block_list[succ_idx]
                    has_phi = any(
                        isinstance(inst, PhiInst) or type(inst).__name__ == "PhiInst"
                        for inst in target_block.instructions
                    )
                    if has_phi:
                        loop_headers.add(succ_idx)
                        back_edge_preds.setdefault(succ_idx, []).append(idx)

        # Also detect by name heuristic
        for idx, block in enumerate(block_list):
            if 'loop' in (block.name or '').lower():
                loop_headers.add(idx)

        # Find all blocks that belong to the loop body for each header
        def _loop_body(header_idx: int) -> Set[int]:
            """Collect block indices in the natural loop."""
            body = {header_idx}
            if header_idx not in back_edge_preds:
                return body
            worklist = list(back_edge_preds[header_idx])
            for pred_idx in worklist:
                if pred_idx not in body:
                    body.add(pred_idx)
                    pred_block = block_list[pred_idx]
                    for p in pred_block.predecessors:
                        pi = block_indices.get(id(p))
                        if pi is not None and pi not in body:
                            worklist.append(pi)
            return body

        loop_bodies: Dict[int, Set[int]] = {}
        for h in loop_headers:
            loop_bodies[h] = _loop_body(h)

        # Which blocks are in any loop?
        in_loop: Set[int] = set()
        for body in loop_bodies.values():
            in_loop |= body

        # Encode blocks
        return_expr: Optional[z3.ExprRef] = None
        K = max(self.loop_unroll_bound, 0)

        # Track which loop headers we've already unrolled
        unrolled_headers: Set[int] = set()

        for idx, block in enumerate(block_list):
            # If this block is part of a loop and its header hasn't been
            # unrolled yet, unroll the entire loop now.
            if idx in loop_headers and K > 0 and idx not in unrolled_headers:
                unrolled_headers.add(idx)
                body_indices = sorted(loop_bodies.get(idx, {idx}))
                body_blocks = [block_list[bi] for bi in body_indices]

                # prev_defs maps original_inst_name -> z3 expr from prior iteration
                prev_defs: Dict[str, z3.ExprRef] = {}

                for iteration in range(K):
                    iter_tag = f"{prefix}K{iteration}_"
                    for b in body_blocks:
                        for inst in b.instructions:
                            orig_name = inst.name
                            if inst.name:
                                inst.name = f"{iter_tag}{inst.name}"

                            # Thread phi nodes across iterations
                            if (isinstance(inst, PhiInst) or type(inst).__name__ == "PhiInst") and iteration > 0:
                                # Wire phi incoming from back-edge to prev iteration's def
                                incoming = getattr(inst, '_incoming', None)
                                if incoming:
                                    for val, pred_block in incoming:
                                        pred_idx = block_indices.get(id(pred_block))
                                        if pred_idx is not None and pred_idx in in_loop:
                                            # This incoming edge is the back-edge
                                            vname = val.name or f"v_{val.id}"
                                            if vname in prev_defs:
                                                ctx.declarations[inst.name] = prev_defs[vname]

                            result = self.encode_instruction(inst, ctx)
                            if (isinstance(inst, ReturnInst) or type(inst).__name__ == "ReturnInst") and result is not None:
                                return_expr = result

                            # Save definitions for next iteration's phi threading
                            if inst.name and result is not None:
                                # Map the un-prefixed name to this iteration's result
                                if orig_name:
                                    prev_defs[orig_name] = result

                            # Restore original name
                            if orig_name is not None:
                                inst.name = orig_name

                continue

            # Skip loop-body blocks that were already unrolled
            if idx in in_loop and any(h in unrolled_headers for h in loop_headers if idx in loop_bodies.get(h, set())):
                continue

            # Non-loop block: encode normally
            current_block_name = block.name if hasattr(block, 'name') else None
            # Get path condition for this block
            current_path_cond = ctx.get(f"_path_cond_{current_block_name}") if current_block_name else None
            for inst in block.instructions:
                tname = type(inst).__name__
                # Record conditional branch conditions for phi and multi-return encoding
                if tname == "BranchInst" or isinstance(inst, BranchInst):
                    cond = getattr(inst, 'condition', None)
                    tt = getattr(inst, 'true_target', None)
                    ft = getattr(inst, 'false_target', None)
                    if cond is not None:
                        cond_name = cond.name if hasattr(cond, 'name') and cond.name else None
                        cond_z3 = ctx.get(f"{prefix}{cond_name}" if prefix and cond_name else cond_name) if cond_name else None
                        if cond_z3 is not None:
                            # Coerce bitvector conditions to Bool for path logic
                            if z3.is_bv(cond_z3):
                                cond_z3_bool = cond_z3 != z3.BitVecVal(0, cond_z3.size())
                            else:
                                cond_z3_bool = cond_z3
                            if tt is not None and hasattr(tt, 'name'):
                                ctx.declarations[f"_branch_cond_{tt.name}"] = cond_z3_bool
                                # Path condition = parent path AND branch condition
                                if current_path_cond is not None and z3.is_bool(current_path_cond) and z3.is_bool(cond_z3_bool):
                                    ctx.declarations[f"_path_cond_{tt.name}"] = z3.And(current_path_cond, cond_z3_bool)
                                else:
                                    ctx.declarations[f"_path_cond_{tt.name}"] = cond_z3_bool
                            if ft is not None and hasattr(ft, 'name'):
                                neg = z3.Not(cond_z3_bool) if z3.is_bool(cond_z3_bool) else None
                                if neg is not None:
                                    ctx.declarations[f"_branch_cond_{ft.name}"] = neg
                                    if current_path_cond is not None and z3.is_bool(current_path_cond):
                                        ctx.declarations[f"_path_cond_{ft.name}"] = z3.And(current_path_cond, neg)
                                    else:
                                        ctx.declarations[f"_path_cond_{ft.name}"] = neg
                    else:
                        # Unconditional branch: propagate path condition
                        if tt is not None and hasattr(tt, 'name') and current_path_cond is not None:
                            existing = ctx.get(f"_path_cond_{tt.name}")
                            if existing is None:
                                ctx.declarations[f"_path_cond_{tt.name}"] = current_path_cond
                    continue
                if inst.name:
                    inst.name = f"{prefix}{inst.name}" if prefix else inst.name
                result = self.encode_instruction(inst, ctx)
                if (isinstance(inst, ReturnInst) or tname == "ReturnInst") and result is not None:
                    # Handle multiple returns: combine with path condition
                    path_cond = ctx.get(f"_path_cond_{current_block_name}") if current_block_name else None
                    if return_expr is not None and path_cond is not None and z3.is_bool(path_cond):
                        # Coerce return values to matching sorts
                        r_val, re_val = result, return_expr
                        if z3.is_bv(r_val) and z3.is_bv(re_val) and r_val.size() != re_val.size():
                            target = max(r_val.size(), re_val.size())
                            if r_val.size() < target:
                                r_val = z3.SignExt(target - r_val.size(), r_val)
                            if re_val.size() < target:
                                re_val = z3.SignExt(target - re_val.size(), re_val)
                        return_expr = z3.If(path_cond, r_val, re_val)
                    elif return_expr is None:
                        return_expr = result
                    # else: already have a return and no path condition — skip
                if prefix and inst.name and inst.name.startswith(prefix):
                    inst.name = inst.name[len(prefix):]

        return ctx, return_expr

    # -- Product program encoding --

    def encode_equivalence_check(
        self,
        left_func: Function,
        right_func: Function,
        shared_inputs: Optional[List[str]] = None,
    ) -> Tuple[EncodingContext, z3.BoolRef]:
        """
        Encode an equivalence check between two functions.
        
        Returns (context, equivalence_condition).
        The equivalence_condition is True iff both functions produce
        the same output for all inputs satisfying the assumptions.
        """
        ctx = EncodingContext(pointer_width=self.pointer_width)

        # Create shared inputs
        left_args = list(left_func.arguments)
        right_args = list(right_func.arguments)
        min_args = min(len(left_args), len(right_args))

        for i in range(min_args):
            la = left_args[i]
            ra = right_args[i]
            name = shared_inputs[i] if shared_inputs and i < len(shared_inputs) else f"input_{i}"
            sort = self.encode_type(la.type) if la.type else z3.BitVecSort(32)
            shared = ctx.declare(name, sort)

            # Alias for both function's arguments
            ctx.declarations[f"c_{la.name or f'arg_{i}'}"] = shared
            ctx.declarations[f"rust_{ra.name or f'arg_{i}'}"] = shared

        # Encode both functions
        _, left_ret = self.encode_function(left_func, ctx, prefix="c_")
        _, right_ret = self.encode_function(right_func, ctx, prefix="rust_")

        # Equivalence condition
        if left_ret is not None and right_ret is not None:
            equiv_cond = left_ret == right_ret
        else:
            equiv_cond = z3.BoolVal(True)

        return ctx, equiv_cond

    def encode_sigma_equivalence(
        self,
        c_func: Function,
        r_func: Function,
        c_config: Optional[SemanticConfig] = None,
        r_config: Optional[SemanticConfig] = None,
    ) -> Dict[str, Any]:
        """Encode a σ-bridge equivalence check between C and Rust functions.

        Uses separate encoding contexts for each function with distinct
        ``SemanticConfig`` instances (the σ-bridge), then compares
        return values over shared symbolic inputs.

        Returns a dict with keys:
          - ``shared_vars``: mapping of shared BV variables by name
          - ``c_ret``, ``r_ret``: Z3 return expressions
          - ``c_assumptions``: well-definedness constraints from C side
          - ``equiv_query``: Z3 formula asserting non-equivalence
          - ``overflow_query``: Z3 formula asserting overflow possible
        """
        c_cfg = c_config or SemanticConfig.c11()
        r_cfg = r_config or SemanticConfig.rust_release()

        c_encoder = SMTEncoder(config=c_cfg, pointer_width=self.pointer_width,
                               loop_unroll_bound=self.loop_unroll_bound)
        r_encoder = SMTEncoder(config=r_cfg, pointer_width=self.pointer_width,
                               loop_unroll_bound=self.loop_unroll_bound)

        c_args = list(c_func.arguments)
        r_args = list(r_func.arguments)
        n = min(len(c_args), len(r_args))

        # Create shared symbolic inputs
        shared: Dict[str, z3.BitVecRef] = {}
        for i in range(n):
            la = c_args[i]
            sort_ty = la.type if la.type else None
            width = sort_ty.width if sort_ty and hasattr(sort_ty, 'width') else 32
            name = la.name or f"arg_{i}"
            bv = z3.BitVec(name, width)
            shared[name] = bv

        # Encode C function
        c_ctx = EncodingContext(pointer_width=self.pointer_width)
        for i, (name, bv) in enumerate(shared.items()):
            c_ctx.declarations[name] = bv
            # Seed the alloca tracking: if the function allocas for this
            # argument, pre-populate so store→load resolves correctly.
            c_ctx._alloca_values[name] = bv
            arg_key = f"arg_{i}"
            c_ctx.declarations[arg_key] = bv
            c_ctx._alloca_values[arg_key] = bv
        _, c_ret = c_encoder.encode_function(c_func, c_ctx, prefix="")
        c_assumptions = list(c_ctx.assumptions)

        # Encode Rust function
        r_ctx = EncodingContext(pointer_width=self.pointer_width)
        r_arg_names = [a.name or f"arg_{i}" for i, a in enumerate(r_args)]
        for i, (c_name, bv) in enumerate(shared.items()):
            r_name = r_arg_names[i] if i < len(r_arg_names) else c_name
            r_ctx.declarations[r_name] = bv
            r_ctx._alloca_values[r_name] = bv
            arg_key = f"arg_{i}"
            r_ctx.declarations[arg_key] = bv
            r_ctx._alloca_values[arg_key] = bv
        _, r_ret = r_encoder.encode_function(r_func, r_ctx, prefix="")

        # Build queries
        if c_ret is not None and r_ret is not None:
            # Coerce to same width
            if z3.is_bv(c_ret) and z3.is_bv(r_ret):
                c_ret, r_ret = self._coerce_bv(c_ret, r_ret)
            non_equiv = c_ret != r_ret
        else:
            non_equiv = z3.BoolVal(False)

        # Overflow query: is there an input where C has UB?
        overflow_conds = []
        for a in c_assumptions:
            overflow_conds.append(z3.Not(a))

        return {
            "shared_vars": shared,
            "c_ret": c_ret,
            "r_ret": r_ret,
            "c_assumptions": c_assumptions,
            "r_assumptions": list(r_ctx.assumptions),
            "equiv_query": non_equiv,
            "overflow_query": z3.Or(*overflow_conds) if overflow_conds else z3.BoolVal(False),
            "c_ctx": c_ctx,
            "r_ctx": r_ctx,
        }

    # -- Helpers --

    def initialize_memory(self, ctx: EncodingContext) -> z3.ArrayRef:
        """Initialize the memory model as a Z3 array (addr → byte).
        
        Uses QF_ABV (arrays of bitvectors) for struct/pointer reasoning.
        """
        idx_sort = z3.BitVecSort(self.pointer_width)
        val_sort = z3.BitVecSort(8)  # byte-addressable
        mem = ctx.declare_array("_memory", idx_sort, val_sort)
        return mem

    def _coerce_bv(
        self, a: z3.BitVecRef, b: z3.BitVecRef,
    ) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        if a.size() == b.size():
            return a, b
        if a.size() < b.size():
            a = z3.SignExt(b.size() - a.size(), a)
        else:
            b = z3.SignExt(a.size() - b.size(), b)
        return a, b
