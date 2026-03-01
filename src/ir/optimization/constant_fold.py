"""
Constant folding and propagation for the Cross-Language Equivalence Verifier.

Provides:
- ConstantFolder: evaluate constant expressions at compile time
- ConstantPropagation: propagate constants through the program
- ConditionalConstantPropagation: propagate through conditional branches
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from ...ir.function import Function
from ...ir.module import Module
from ...ir.basic_block import BasicBlock
from ...ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, BinOpKind, UnaryOp, UnaryOpKind,
    CompareOp, CmpPredicate,
    CastInst, CastKind,
    SelectInst, PhiInst,
    BranchInst, ReturnInst,
    LoadInst, StoreInst, AllocaInst,
)
from ...ir.types import IRType, IntType, FloatType, Signedness, FloatKind
from .pass_manager import FunctionPass, PassResult, AnalysisManager

logger = logging.getLogger(__name__)


# ─── Constant Evaluation ───────────────────────────────────────────────

def _get_int_value(c: Constant) -> Optional[int]:
    """Extract integer value from a Constant."""
    if hasattr(c, 'value') and isinstance(c.value, (int, bool)):
        return int(c.value)
    return None


def _get_float_value(c: Constant) -> Optional[float]:
    """Extract float value from a Constant."""
    if hasattr(c, 'value') and isinstance(c.value, float):
        return c.value
    return None


def _make_int_constant(value: int, typ: IRType) -> Constant:
    """Create a Constant with an integer value, wrapping to bit width."""
    if isinstance(typ, IntType):
        width = typ.width
        if typ.signedness == Signedness.UNSIGNED:
            value = value & ((1 << width) - 1)
        else:
            mask = (1 << width) - 1
            value = value & mask
            if value >= (1 << (width - 1)):
                value -= (1 << width)
    return Constant(value=value, ir_type=typ)


def _make_float_constant(value: float, typ: IRType) -> Constant:
    """Create a Constant with a float value."""
    if isinstance(typ, FloatType) and typ.kind == FloatKind.F32:
        value = struct.unpack('f', struct.pack('f', value))[0]
    return Constant(value=value, ir_type=typ)


def _make_bool_constant(value: bool, typ: IRType) -> Constant:
    """Create a boolean constant (i1)."""
    return Constant(value=1 if value else 0, ir_type=IntType(width=1, signedness=Signedness.UNSIGNED))


class ArithmeticEvaluator:
    """Evaluate binary arithmetic operations on constants."""

    @staticmethod
    def evaluate_int_binop(op: BinOpKind, left: int, right: int, typ: IntType) -> Optional[int]:
        """Evaluate integer binary operation, returning None if undefined."""
        width = typ.width
        mask = (1 << width) - 1
        signed = typ.signedness == Signedness.SIGNED

        def to_signed(v: int) -> int:
            v = v & mask
            if v >= (1 << (width - 1)):
                return v - (1 << width)
            return v

        def to_unsigned(v: int) -> int:
            return v & mask

        l = to_signed(left) if signed else to_unsigned(left)
        r = to_signed(right) if signed else to_unsigned(right)

        if op == BinOpKind.ADD:
            return (l + r) & mask
        elif op == BinOpKind.SUB:
            return (l - r) & mask
        elif op == BinOpKind.MUL:
            return (l * r) & mask
        elif op == BinOpKind.SDIV:
            if r == 0:
                return None
            if signed and l == -(1 << (width - 1)) and r == -1:
                return None  # overflow
            result = int(l / r) if (l ^ r) >= 0 else -int(abs(l) / abs(r))
            return result & mask
        elif op == BinOpKind.UDIV:
            ul, ur = to_unsigned(left), to_unsigned(right)
            if ur == 0:
                return None
            return (ul // ur) & mask
        elif op == BinOpKind.SREM:
            if r == 0:
                return None
            result = l - int(l / r) * r if r != 0 else None
            return result & mask if result is not None else None
        elif op == BinOpKind.UREM:
            ul, ur = to_unsigned(left), to_unsigned(right)
            if ur == 0:
                return None
            return (ul % ur) & mask
        elif op == BinOpKind.SHL:
            shift = to_unsigned(right)
            if shift >= width:
                return 0
            return (to_unsigned(left) << shift) & mask
        elif op == BinOpKind.LSHR:
            shift = to_unsigned(right)
            if shift >= width:
                return 0
            return (to_unsigned(left) >> shift) & mask
        elif op == BinOpKind.ASHR:
            shift = to_unsigned(right)
            if shift >= width:
                return mask if l < 0 else 0
            return (l >> shift) & mask
        elif op == BinOpKind.AND:
            return (to_unsigned(left) & to_unsigned(right)) & mask
        elif op == BinOpKind.OR:
            return (to_unsigned(left) | to_unsigned(right)) & mask
        elif op == BinOpKind.XOR:
            return (to_unsigned(left) ^ to_unsigned(right)) & mask
        return None

    @staticmethod
    def evaluate_float_binop(op: BinOpKind, left: float, right: float,
                              typ: FloatType) -> Optional[float]:
        """Evaluate floating-point binary operation."""
        try:
            if op == BinOpKind.FADD:
                result = left + right
            elif op == BinOpKind.FSUB:
                result = left - right
            elif op == BinOpKind.FMUL:
                result = left * right
            elif op == BinOpKind.FDIV:
                if right == 0.0:
                    if left == 0.0:
                        return float('nan')
                    return float('inf') if left > 0 else float('-inf')
                result = left / right
            elif op == BinOpKind.FREM:
                if right == 0.0:
                    return float('nan')
                result = math.fmod(left, right)
            else:
                return None

            if typ.kind == FloatKind.F32:
                result = struct.unpack('f', struct.pack('f', result))[0]
            return result
        except (OverflowError, ValueError):
            return None


class ComparisonEvaluator:
    """Evaluate comparison operations on constants."""

    @staticmethod
    def evaluate_int_cmp(pred: CmpPredicate, left: int, right: int,
                          typ: IntType) -> Optional[bool]:
        width = typ.width
        mask = (1 << width) - 1

        def to_signed(v: int) -> int:
            v = v & mask
            if v >= (1 << (width - 1)):
                return v - (1 << width)
            return v

        def to_unsigned(v: int) -> int:
            return v & mask

        sl, sr = to_signed(left), to_signed(right)
        ul, ur = to_unsigned(left), to_unsigned(right)

        if pred == CmpPredicate.EQ:
            return ul == ur
        elif pred == CmpPredicate.NE:
            return ul != ur
        elif pred == CmpPredicate.SLT:
            return sl < sr
        elif pred == CmpPredicate.SLE:
            return sl <= sr
        elif pred == CmpPredicate.SGT:
            return sl > sr
        elif pred == CmpPredicate.SGE:
            return sl >= sr
        elif pred == CmpPredicate.ULT:
            return ul < ur
        elif pred == CmpPredicate.ULE:
            return ul <= ur
        elif pred == CmpPredicate.UGT:
            return ul > ur
        elif pred == CmpPredicate.UGE:
            return ul >= ur
        return None

    @staticmethod
    def evaluate_float_cmp(pred: CmpPredicate, left: float, right: float) -> Optional[bool]:
        l_nan = math.isnan(left)
        r_nan = math.isnan(right)

        if pred == CmpPredicate.OEQ:
            return not l_nan and not r_nan and left == right
        elif pred == CmpPredicate.ONE:
            return not l_nan and not r_nan and left != right
        elif pred == CmpPredicate.OLT:
            return not l_nan and not r_nan and left < right
        elif pred == CmpPredicate.OLE:
            return not l_nan and not r_nan and left <= right
        elif pred == CmpPredicate.OGT:
            return not l_nan and not r_nan and left > right
        elif pred == CmpPredicate.OGE:
            return not l_nan and not r_nan and left >= right
        elif pred == CmpPredicate.ORD:
            return not l_nan and not r_nan
        elif pred == CmpPredicate.UNO:
            return l_nan or r_nan
        return None


class CastEvaluator:
    """Evaluate cast operations on constants."""

    @staticmethod
    def evaluate(kind: CastKind, value: Any, src_type: IRType, dst_type: IRType) -> Optional[Any]:
        if kind == CastKind.ZEXT:
            if isinstance(value, int) and isinstance(src_type, IntType):
                mask = (1 << src_type.width) - 1
                return value & mask
            return None

        elif kind == CastKind.SEXT:
            if isinstance(value, int) and isinstance(src_type, IntType) and isinstance(dst_type, IntType):
                mask = (1 << src_type.width) - 1
                v = value & mask
                if v >= (1 << (src_type.width - 1)):
                    v -= (1 << src_type.width)
                dst_mask = (1 << dst_type.width) - 1
                return v & dst_mask
            return None

        elif kind == CastKind.TRUNC:
            if isinstance(value, int) and isinstance(dst_type, IntType):
                mask = (1 << dst_type.width) - 1
                return value & mask
            return None

        elif kind == CastKind.SITOFP:
            if isinstance(value, int) and isinstance(src_type, IntType):
                mask = (1 << src_type.width) - 1
                v = value & mask
                if v >= (1 << (src_type.width - 1)):
                    v -= (1 << src_type.width)
                return float(v)
            return None

        elif kind == CastKind.UITOFP:
            if isinstance(value, int) and isinstance(src_type, IntType):
                mask = (1 << src_type.width) - 1
                return float(value & mask)
            return None

        elif kind == CastKind.FPTOSI:
            if isinstance(value, float) and isinstance(dst_type, IntType):
                if math.isnan(value) or math.isinf(value):
                    return None
                iv = int(value)
                mask = (1 << dst_type.width) - 1
                return iv & mask
            return None

        elif kind == CastKind.FPTOUI:
            if isinstance(value, float) and isinstance(dst_type, IntType):
                if math.isnan(value) or math.isinf(value) or value < 0:
                    return None
                iv = int(value)
                mask = (1 << dst_type.width) - 1
                return iv & mask
            return None

        elif kind == CastKind.BITCAST:
            return value  # No change in bit pattern

        return None


# ─── Constant Folder Pass ──────────────────────────────────────────────

class ConstantFolder(FunctionPass):
    """Fold constant expressions at compile time.

    Evaluates BinaryOp, UnaryOp, CompareOp, CastInst, and SelectInst
    where all operands are constants.
    """

    _name = "constant_fold"
    _description = "Evaluate constant expressions at compile time"
    _invalidated_analyses = ["cfg", "domtree"]

    def __init__(self) -> None:
        super().__init__()
        self._arith = ArithmeticEvaluator()
        self._cmp = ComparisonEvaluator()
        self._cast = CastEvaluator()

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        changed = False
        worklist: List[Instruction] = []

        for block in function.blocks:
            for inst in block.instructions:
                worklist.append(inst)

        while worklist:
            inst = worklist.pop()
            result = self._try_fold(inst)
            if result is not None:
                self._replace_with_constant(inst, result, worklist)
                changed = True

        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _try_fold(self, inst: Instruction) -> Optional[Constant]:
        if isinstance(inst, BinaryOp):
            return self._fold_binop(inst)
        elif isinstance(inst, UnaryOp):
            return self._fold_unaryop(inst)
        elif isinstance(inst, CompareOp):
            return self._fold_cmp(inst)
        elif isinstance(inst, CastInst):
            return self._fold_cast(inst)
        elif isinstance(inst, SelectInst):
            return self._fold_select(inst)
        return None

    def _fold_binop(self, inst: BinaryOp) -> Optional[Constant]:
        if not isinstance(inst.left, Constant) or not isinstance(inst.right, Constant):
            return None

        lv = _get_int_value(inst.left)
        rv = _get_int_value(inst.right)

        if lv is not None and rv is not None and isinstance(inst.ir_type, IntType):
            result = self._arith.evaluate_int_binop(inst.op, lv, rv, inst.ir_type)
            if result is not None:
                return _make_int_constant(result, inst.ir_type)

        lf = _get_float_value(inst.left)
        rf = _get_float_value(inst.right)

        if lf is not None and rf is not None and isinstance(inst.ir_type, FloatType):
            result = self._arith.evaluate_float_binop(inst.op, lf, rf, inst.ir_type)
            if result is not None:
                return _make_float_constant(result, inst.ir_type)

        return None

    def _fold_unaryop(self, inst: UnaryOp) -> Optional[Constant]:
        if not isinstance(inst.operand, Constant):
            return None

        iv = _get_int_value(inst.operand)
        if iv is not None and isinstance(inst.ir_type, IntType):
            if inst.op == UnaryOpKind.NEG:
                return _make_int_constant(-iv, inst.ir_type)
            elif inst.op == UnaryOpKind.NOT:
                width = inst.ir_type.width
                mask = (1 << width) - 1
                return _make_int_constant(iv ^ mask, inst.ir_type)

        fv = _get_float_value(inst.operand)
        if fv is not None and isinstance(inst.ir_type, FloatType):
            if inst.op == UnaryOpKind.NEG:
                return _make_float_constant(-fv, inst.ir_type)

        return None

    def _fold_cmp(self, inst: CompareOp) -> Optional[Constant]:
        if not isinstance(inst.left, Constant) or not isinstance(inst.right, Constant):
            return None

        lv = _get_int_value(inst.left)
        rv = _get_int_value(inst.right)

        if lv is not None and rv is not None:
            src_type = inst.left.ir_type
            if isinstance(src_type, IntType):
                result = self._cmp.evaluate_int_cmp(inst.predicate, lv, rv, src_type)
                if result is not None:
                    return _make_bool_constant(result, inst.ir_type)

        lf = _get_float_value(inst.left)
        rf = _get_float_value(inst.right)

        if lf is not None and rf is not None:
            result = self._cmp.evaluate_float_cmp(inst.predicate, lf, rf)
            if result is not None:
                return _make_bool_constant(result, inst.ir_type)

        return None

    def _fold_cast(self, inst: CastInst) -> Optional[Constant]:
        if not isinstance(inst.operand, Constant):
            return None

        value = inst.operand.value if hasattr(inst.operand, 'value') else None
        if value is None:
            return None

        result = self._cast.evaluate(inst.cast_kind, value, inst.operand.ir_type, inst.ir_type)
        if result is not None:
            if isinstance(result, float):
                return _make_float_constant(result, inst.ir_type)
            else:
                return _make_int_constant(int(result), inst.ir_type)
        return None

    def _fold_select(self, inst: SelectInst) -> Optional[Constant]:
        if not isinstance(inst.condition, Constant):
            return None

        cv = _get_int_value(inst.condition)
        if cv is not None:
            if cv != 0:
                if isinstance(inst.true_value, Constant):
                    return inst.true_value
            else:
                if isinstance(inst.false_value, Constant):
                    return inst.false_value
        return None

    def _replace_with_constant(self, inst: Instruction, const: Constant,
                                worklist: List[Instruction]) -> None:
        """Replace all uses of inst with const, then remove inst."""
        if hasattr(inst, 'users'):
            for user in list(inst.users):
                self._substitute_operand(user, inst, const)
                if isinstance(user, Instruction):
                    worklist.append(user)

        if inst.parent is not None:
            inst.parent.remove(inst)
            self.stats.instructions_removed += 1
            self.stats.increment("constants_folded")

    def _substitute_operand(self, user: Instruction, old_val: Value, new_val: Value) -> None:
        """Replace old_val with new_val in all operand positions of user."""
        if isinstance(user, BinaryOp):
            if user.left is old_val:
                user.left = new_val
            if user.right is old_val:
                user.right = new_val
        elif isinstance(user, UnaryOp):
            if user.operand is old_val:
                user.operand = new_val
        elif isinstance(user, CompareOp):
            if user.left is old_val:
                user.left = new_val
            if user.right is old_val:
                user.right = new_val
        elif isinstance(user, CastInst):
            if user.operand is old_val:
                user.operand = new_val
        elif isinstance(user, SelectInst):
            if user.condition is old_val:
                user.condition = new_val
            if user.true_value is old_val:
                user.true_value = new_val
            if user.false_value is old_val:
                user.false_value = new_val
        elif isinstance(user, PhiInst):
            new_incoming = []
            for val, block in user.incoming:
                if val is old_val:
                    new_incoming.append((new_val, block))
                else:
                    new_incoming.append((val, block))
            user.incoming = new_incoming
        elif isinstance(user, BranchInst):
            if user.is_conditional and user.condition is old_val:
                user.condition = new_val
        elif isinstance(user, ReturnInst):
            if user.value is old_val:
                user.value = new_val
        elif isinstance(user, StoreInst):
            if user.value is old_val:
                user.value = new_val


# ─── Constant Propagation ──────────────────────────────────────────────

class ConstantPropagation(FunctionPass):
    """Forward propagation of constant values through the program.

    Tracks which SSA values are constants and substitutes them
    at use sites. Handles phi nodes where all incoming values
    are the same constant.
    """

    _name = "constant_prop"
    _description = "Propagate constant values through SSA"
    _invalidated_analyses = ["cfg"]

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        const_map: Dict[int, Constant] = {}
        changed = False

        # Phase 1: identify constant-valued instructions
        for block in function.iter_blocks_rpo():
            for inst in block.instructions:
                const = self._evaluate_to_constant(inst, const_map)
                if const is not None:
                    const_map[inst.id] = const

        # Phase 2: propagate phi nodes iteratively
        phi_changed = True
        iterations = 0
        max_iterations = 100
        while phi_changed and iterations < max_iterations:
            phi_changed = False
            iterations += 1
            for block in function.blocks:
                for phi in block.phi_nodes:
                    if phi.id in const_map:
                        continue
                    const = self._evaluate_phi_constant(phi, const_map)
                    if const is not None:
                        const_map[phi.id] = const
                        phi_changed = True

        # Phase 3: substitute constants
        if not const_map:
            return PassResult.UNCHANGED

        to_remove: List[Tuple[BasicBlock, Instruction]] = []
        for block in function.blocks:
            for inst in block.instructions:
                if inst.id in const_map:
                    const = const_map[inst.id]
                    if hasattr(inst, 'users'):
                        for user in list(inst.users):
                            self._substitute_operand(user, inst, const)
                            changed = True
                    to_remove.append((block, inst))

        for block, inst in reversed(to_remove):
            if not self._has_side_effects(inst):
                block.remove(inst)
                self.stats.instructions_removed += 1

        self.stats.increment("constants_propagated", len(const_map))
        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _evaluate_to_constant(self, inst: Instruction, known: Dict[int, Constant]) -> Optional[Constant]:
        """Check if inst computes a constant given known constants."""
        if isinstance(inst, BinaryOp):
            left = self._resolve(inst.left, known)
            right = self._resolve(inst.right, known)
            if isinstance(left, Constant) and isinstance(right, Constant):
                lv = _get_int_value(left)
                rv = _get_int_value(right)
                if lv is not None and rv is not None and isinstance(inst.ir_type, IntType):
                    result = ArithmeticEvaluator.evaluate_int_binop(inst.op, lv, rv, inst.ir_type)
                    if result is not None:
                        return _make_int_constant(result, inst.ir_type)
        elif isinstance(inst, CastInst):
            operand = self._resolve(inst.operand, known)
            if isinstance(operand, Constant) and hasattr(operand, 'value'):
                result = CastEvaluator.evaluate(inst.cast_kind, operand.value,
                                                 inst.operand.ir_type, inst.ir_type)
                if result is not None:
                    if isinstance(result, float):
                        return _make_float_constant(result, inst.ir_type)
                    return _make_int_constant(int(result), inst.ir_type)
        return None

    def _evaluate_phi_constant(self, phi: PhiInst, known: Dict[int, Constant]) -> Optional[Constant]:
        """Check if all incoming values of a phi are the same constant."""
        if not phi.incoming:
            return None

        unique_val: Optional[int] = None
        unique_const: Optional[Constant] = None

        for val, _ in phi.incoming:
            resolved = self._resolve(val, known)
            if not isinstance(resolved, Constant):
                return None
            iv = _get_int_value(resolved)
            if iv is None:
                fv = _get_float_value(resolved)
                if fv is None:
                    return None
                if unique_val is None:
                    unique_val = hash(fv)
                    unique_const = resolved
                elif hash(fv) != unique_val:
                    return None
            else:
                if unique_val is None:
                    unique_val = iv
                    unique_const = resolved
                elif iv != unique_val:
                    return None

        return unique_const

    def _resolve(self, val: Value, known: Dict[int, Constant]) -> Value:
        if isinstance(val, Constant):
            return val
        if isinstance(val, Instruction) and val.id in known:
            return known[val.id]
        return val

    def _has_side_effects(self, inst: Instruction) -> bool:
        return isinstance(inst, (StoreInst, ReturnInst, BranchInst, CallInst))

    def _substitute_operand(self, user: Instruction, old_val: Value, new_val: Value) -> None:
        if isinstance(user, BinaryOp):
            if user.left is old_val:
                user.left = new_val
            if user.right is old_val:
                user.right = new_val
        elif isinstance(user, UnaryOp):
            if user.operand is old_val:
                user.operand = new_val
        elif isinstance(user, CompareOp):
            if user.left is old_val:
                user.left = new_val
            if user.right is old_val:
                user.right = new_val
        elif isinstance(user, CastInst):
            if user.operand is old_val:
                user.operand = new_val
        elif isinstance(user, SelectInst):
            if user.condition is old_val:
                user.condition = new_val
            if user.true_value is old_val:
                user.true_value = new_val
            if user.false_value is old_val:
                user.false_value = new_val
        elif isinstance(user, PhiInst):
            user.incoming = [
                (new_val if v is old_val else v, b) for v, b in user.incoming
            ]
        elif isinstance(user, BranchInst) and user.is_conditional:
            if user.condition is old_val:
                user.condition = new_val
        elif isinstance(user, ReturnInst):
            if user.value is old_val:
                user.value = new_val
        elif isinstance(user, StoreInst):
            if user.value is old_val:
                user.value = new_val


# ─── Conditional Constant Propagation ──────────────────────────────────

class _CCPLattice(Enum):
    """Lattice for conditional constant propagation."""
    TOP = auto()       # Unknown (not yet visited)
    CONSTANT = auto()  # Known constant value
    BOTTOM = auto()    # Overdefined (multiple values possible)


@dataclass
class _CCPValue:
    """A value in the CCP lattice."""
    state: _CCPLattice
    constant: Optional[Constant] = None

    @staticmethod
    def top() -> "_CCPValue":
        return _CCPValue(state=_CCPLattice.TOP)

    @staticmethod
    def bottom() -> "_CCPValue":
        return _CCPValue(state=_CCPLattice.BOTTOM)

    @staticmethod
    def const(c: Constant) -> "_CCPValue":
        return _CCPValue(state=_CCPLattice.CONSTANT, constant=c)

    def meet(self, other: "_CCPValue") -> "_CCPValue":
        if self.state == _CCPLattice.TOP:
            return other
        if other.state == _CCPLattice.TOP:
            return self
        if self.state == _CCPLattice.BOTTOM or other.state == _CCPLattice.BOTTOM:
            return _CCPValue.bottom()
        # Both constant
        if (self.constant is not None and other.constant is not None and
                hasattr(self.constant, 'value') and hasattr(other.constant, 'value') and
                self.constant.value == other.constant.value):
            return self
        return _CCPValue.bottom()

    @property
    def is_constant(self) -> bool:
        return self.state == _CCPLattice.CONSTANT

    @property
    def is_top(self) -> bool:
        return self.state == _CCPLattice.TOP

    @property
    def is_bottom(self) -> bool:
        return self.state == _CCPLattice.BOTTOM


class ConditionalConstantPropagation(FunctionPass):
    """Conditional constant propagation using CCP lattice.

    Simultaneously propagates constants and determines reachable
    edges, discovering constants that simple constant propagation
    would miss due to unreachable paths.
    """

    _name = "ccp"
    _description = "Conditional constant propagation"
    _invalidated_analyses = ["cfg", "domtree", "loops"]

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        values: Dict[int, _CCPValue] = {}
        executable_edges: Set[Tuple[int, int]] = set()
        executable_blocks: Set[int] = set()

        # Initialize all values to TOP
        for block in function.blocks:
            for inst in block.instructions:
                values[inst.id] = _CCPValue.top()

        # Arguments are BOTTOM (unknown input)
        for arg in function.arguments:
            values[arg.id] = _CCPValue.bottom()

        # Worklists
        ssa_worklist: List[Instruction] = []
        cfg_worklist: List[BasicBlock] = []

        entry = function.entry_block
        if entry is None:
            return PassResult.UNCHANGED
        cfg_worklist.append(entry)

        # Main loop
        iteration = 0
        max_iterations = 10000

        while (ssa_worklist or cfg_worklist) and iteration < max_iterations:
            iteration += 1

            while cfg_worklist:
                block = cfg_worklist.pop()
                block_id = block.id
                if block_id in executable_blocks:
                    for phi in block.phi_nodes:
                        old = values.get(phi.id, _CCPValue.top())
                        new = self._evaluate_phi(phi, values, executable_edges)
                        if new.state != old.state or (new.is_constant and old.is_constant and
                                                       new.constant.value != old.constant.value):
                            values[phi.id] = new
                            if hasattr(phi, 'users'):
                                ssa_worklist.extend(
                                    u for u in phi.users if isinstance(u, Instruction))
                    continue

                executable_blocks.add(block_id)

                for inst in block.instructions:
                    old = values.get(inst.id, _CCPValue.top())
                    new = self._evaluate_instruction(inst, values)
                    values[inst.id] = new
                    if new.state != old.state:
                        if hasattr(inst, 'users'):
                            ssa_worklist.extend(
                                u for u in inst.users if isinstance(u, Instruction))

                self._propagate_control_flow(block, values, executable_edges, cfg_worklist)

            while ssa_worklist:
                inst = ssa_worklist.pop()
                if inst.parent is None or inst.parent.id not in executable_blocks:
                    continue

                old = values.get(inst.id, _CCPValue.top())
                if isinstance(inst, PhiInst):
                    new = self._evaluate_phi(inst, values, executable_edges)
                else:
                    new = self._evaluate_instruction(inst, values)

                if new.state != old.state or (new.is_constant and old.is_constant and
                                               hasattr(new.constant, 'value') and
                                               hasattr(old.constant, 'value') and
                                               new.constant.value != old.constant.value):
                    values[inst.id] = new
                    if hasattr(inst, 'users'):
                        ssa_worklist.extend(
                            u for u in inst.users if isinstance(u, Instruction))

                    if isinstance(inst, (BranchInst, SwitchInst)):
                        self._propagate_control_flow(
                            inst.parent, values, executable_edges, cfg_worklist)

        # Apply results: substitute constants and remove dead branches
        changed = False
        for block in function.blocks:
            for inst in list(block.instructions):
                val = values.get(inst.id)
                if val is not None and val.is_constant and val.constant is not None:
                    if not isinstance(inst, (ReturnInst, BranchInst, SwitchInst,
                                             StoreInst, CallInst, PhiInst)):
                        if hasattr(inst, 'users'):
                            for user in list(inst.users):
                                self._substitute_operand(user, inst, val.constant)
                        block.remove(inst)
                        self.stats.instructions_removed += 1
                        changed = True

        # Remove non-executable blocks
        for block in list(function.blocks):
            if block.id not in executable_blocks and block is not entry:
                for succ in list(block.successors):
                    for phi in succ.phi_nodes:
                        phi.remove_incoming(block)
                    succ.remove_predecessor(block)
                function.remove_block(block)
                self.stats.blocks_removed += 1
                changed = True

        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _evaluate_instruction(self, inst: Instruction, values: Dict[int, _CCPValue]) -> _CCPValue:
        if isinstance(inst, BinaryOp):
            return self._eval_binop(inst, values)
        elif isinstance(inst, CompareOp):
            return self._eval_cmp(inst, values)
        elif isinstance(inst, CastInst):
            return self._eval_cast(inst, values)
        elif isinstance(inst, SelectInst):
            return self._eval_select(inst, values)
        elif isinstance(inst, (LoadInst, CallInst)):
            return _CCPValue.bottom()
        elif isinstance(inst, AllocaInst):
            return _CCPValue.bottom()
        return _CCPValue.bottom()

    def _resolve_value(self, val: Value, values: Dict[int, _CCPValue]) -> _CCPValue:
        if isinstance(val, Constant):
            return _CCPValue.const(val)
        if isinstance(val, Argument):
            return values.get(val.id, _CCPValue.bottom())
        if isinstance(val, Instruction):
            return values.get(val.id, _CCPValue.top())
        return _CCPValue.bottom()

    def _eval_binop(self, inst: BinaryOp, values: Dict[int, _CCPValue]) -> _CCPValue:
        lv = self._resolve_value(inst.left, values)
        rv = self._resolve_value(inst.right, values)
        if lv.is_bottom or rv.is_bottom:
            return _CCPValue.bottom()
        if lv.is_top or rv.is_top:
            return _CCPValue.top()
        li = _get_int_value(lv.constant)
        ri = _get_int_value(rv.constant)
        if li is not None and ri is not None and isinstance(inst.ir_type, IntType):
            result = ArithmeticEvaluator.evaluate_int_binop(inst.op, li, ri, inst.ir_type)
            if result is not None:
                return _CCPValue.const(_make_int_constant(result, inst.ir_type))
        return _CCPValue.bottom()

    def _eval_cmp(self, inst: CompareOp, values: Dict[int, _CCPValue]) -> _CCPValue:
        lv = self._resolve_value(inst.left, values)
        rv = self._resolve_value(inst.right, values)
        if lv.is_bottom or rv.is_bottom:
            return _CCPValue.bottom()
        if lv.is_top or rv.is_top:
            return _CCPValue.top()
        li = _get_int_value(lv.constant)
        ri = _get_int_value(rv.constant)
        if li is not None and ri is not None:
            src_type = inst.left.ir_type if hasattr(inst.left, 'ir_type') else None
            if isinstance(src_type, IntType):
                result = ComparisonEvaluator.evaluate_int_cmp(inst.predicate, li, ri, src_type)
                if result is not None:
                    return _CCPValue.const(_make_bool_constant(result, inst.ir_type))
        return _CCPValue.bottom()

    def _eval_cast(self, inst: CastInst, values: Dict[int, _CCPValue]) -> _CCPValue:
        ov = self._resolve_value(inst.operand, values)
        if ov.is_bottom:
            return _CCPValue.bottom()
        if ov.is_top:
            return _CCPValue.top()
        if ov.constant is not None and hasattr(ov.constant, 'value'):
            result = CastEvaluator.evaluate(inst.cast_kind, ov.constant.value,
                                             inst.operand.ir_type, inst.ir_type)
            if result is not None:
                if isinstance(result, float):
                    return _CCPValue.const(_make_float_constant(result, inst.ir_type))
                return _CCPValue.const(_make_int_constant(int(result), inst.ir_type))
        return _CCPValue.bottom()

    def _eval_select(self, inst: SelectInst, values: Dict[int, _CCPValue]) -> _CCPValue:
        cv = self._resolve_value(inst.condition, values)
        tv = self._resolve_value(inst.true_value, values)
        fv = self._resolve_value(inst.false_value, values)
        if cv.is_constant and cv.constant is not None:
            c = _get_int_value(cv.constant)
            if c is not None:
                return tv if c != 0 else fv
        if cv.is_bottom:
            return tv.meet(fv)
        return _CCPValue.top()

    def _evaluate_phi(self, phi: PhiInst, values: Dict[int, _CCPValue],
                       executable_edges: Set[Tuple[int, int]]) -> _CCPValue:
        result = _CCPValue.top()
        for val, block in phi.incoming:
            edge = (block.id, phi.parent.id) if phi.parent else (block.id, -1)
            if edge not in executable_edges:
                continue
            incoming_val = self._resolve_value(val, values)
            result = result.meet(incoming_val)
            if result.is_bottom:
                break
        return result

    def _propagate_control_flow(self, block: BasicBlock, values: Dict[int, _CCPValue],
                                 executable_edges: Set[Tuple[int, int]],
                                 cfg_worklist: List[BasicBlock]) -> None:
        term = block.terminator
        if term is None:
            return

        if isinstance(term, BranchInst):
            if not term.is_conditional:
                target = term.target if hasattr(term, 'target') else (
                    term.true_block if hasattr(term, 'true_block') else None)
                if target is not None:
                    edge = (block.id, target.id)
                    if edge not in executable_edges:
                        executable_edges.add(edge)
                        cfg_worklist.append(target)
            else:
                cv = self._resolve_value(term.condition, values)
                if cv.is_constant and cv.constant is not None:
                    c = _get_int_value(cv.constant)
                    if c is not None:
                        taken = term.true_block if c != 0 else term.false_block
                        if taken is not None:
                            edge = (block.id, taken.id)
                            if edge not in executable_edges:
                                executable_edges.add(edge)
                                cfg_worklist.append(taken)
                        return
                # Unknown condition: both edges executable
                for target in [term.true_block, term.false_block]:
                    if target is not None:
                        edge = (block.id, target.id)
                        if edge not in executable_edges:
                            executable_edges.add(edge)
                            cfg_worklist.append(target)

        elif isinstance(term, ReturnInst):
            pass  # No successors

        elif isinstance(term, SwitchInst):
            for succ in block.successors:
                edge = (block.id, succ.id)
                if edge not in executable_edges:
                    executable_edges.add(edge)
                    cfg_worklist.append(succ)

    def _substitute_operand(self, user: Instruction, old_val: Value, new_val: Value) -> None:
        if isinstance(user, BinaryOp):
            if user.left is old_val:
                user.left = new_val
            if user.right is old_val:
                user.right = new_val
        elif isinstance(user, UnaryOp):
            if user.operand is old_val:
                user.operand = new_val
        elif isinstance(user, CompareOp):
            if user.left is old_val:
                user.left = new_val
            if user.right is old_val:
                user.right = new_val
        elif isinstance(user, CastInst):
            if user.operand is old_val:
                user.operand = new_val
        elif isinstance(user, SelectInst):
            if user.condition is old_val:
                user.condition = new_val
            if user.true_value is old_val:
                user.true_value = new_val
            if user.false_value is old_val:
                user.false_value = new_val
        elif isinstance(user, PhiInst):
            user.incoming = [
                (new_val if v is old_val else v, b) for v, b in user.incoming
            ]
        elif isinstance(user, BranchInst) and user.is_conditional:
            if user.condition is old_val:
                user.condition = new_val
        elif isinstance(user, ReturnInst):
            if user.value is old_val:
                user.value = new_val
        elif isinstance(user, StoreInst):
            if user.value is old_val:
                user.value = new_val
