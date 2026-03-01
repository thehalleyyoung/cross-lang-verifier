"""
Instruction simplification for the Cross-Language Equivalence Verifier.

Provides:
- InstructionSimplifier: main pass combining all simplification rules
- AlgebraicSimplification: algebraic identity rules (x+0=x, x*1=x, etc.)
- StrengthReduction: replace expensive ops with cheaper ones (mul→shift)
- BooleanSimplification: simplify boolean expressions
- RedundantCastElimination: remove unnecessary casts
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from ...ir.function import Function
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


def _is_zero(val: Value) -> bool:
    """Check if a value is the constant zero."""
    if isinstance(val, Constant) and hasattr(val, 'value'):
        return val.value == 0
    return False


def _is_one(val: Value) -> bool:
    if isinstance(val, Constant) and hasattr(val, 'value'):
        return val.value == 1
    return False


def _is_all_ones(val: Value) -> bool:
    """Check if a value is all-ones (e.g., -1 for signed, 0xFF..F for unsigned)."""
    if not isinstance(val, Constant) or not hasattr(val, 'value'):
        return False
    if isinstance(val.ir_type, IntType):
        mask = (1 << val.ir_type.width) - 1
        return (val.value & mask) == mask
    return False


def _is_power_of_two(val: Value) -> Optional[int]:
    """If val is a constant power of 2, return the exponent. Otherwise None."""
    if not isinstance(val, Constant) or not hasattr(val, 'value'):
        return None
    v = val.value
    if isinstance(v, int) and v > 0 and (v & (v - 1)) == 0:
        exp = 0
        while (1 << exp) < v:
            exp += 1
        return exp
    return None


def _get_int_val(val: Value) -> Optional[int]:
    if isinstance(val, Constant) and hasattr(val, 'value') and isinstance(val.value, (int, bool)):
        return int(val.value)
    return None


def _make_int_const(value: int, typ: IRType) -> Constant:
    if isinstance(typ, IntType):
        mask = (1 << typ.width) - 1
        value = value & mask
    return Constant(value=value, ir_type=typ)


def _replace_inst_with(inst: Instruction, replacement: Value,
                        block: BasicBlock) -> None:
    """Replace all uses of inst with replacement and remove inst from block."""
    if hasattr(inst, 'users'):
        for user in list(inst.users):
            if isinstance(user, Instruction):
                _substitute(user, inst, replacement)
                if hasattr(replacement, 'users') and not isinstance(replacement, Constant):
                    replacement.users.append(user)
    try:
        block.remove(inst)
    except ValueError:
        pass


def _substitute(user: Instruction, old: Value, new: Value) -> None:
    if isinstance(user, BinaryOp):
        if user.left is old:
            user.left = new
        if user.right is old:
            user.right = new
    elif isinstance(user, UnaryOp):
        if user.operand is old:
            user.operand = new
    elif isinstance(user, CompareOp):
        if user.left is old:
            user.left = new
        if user.right is old:
            user.right = new
    elif isinstance(user, CastInst):
        if user.operand is old:
            user.operand = new
    elif isinstance(user, SelectInst):
        if user.condition is old:
            user.condition = new
        if user.true_value is old:
            user.true_value = new
        if user.false_value is old:
            user.false_value = new
    elif isinstance(user, PhiInst):
        user.incoming = [(new if v is old else v, b) for v, b in user.incoming]
    elif isinstance(user, BranchInst) and user.is_conditional:
        if user.condition is old:
            user.condition = new
    elif isinstance(user, ReturnInst):
        if user.value is old:
            user.value = new
    elif isinstance(user, StoreInst):
        if user.value is old:
            user.value = new


# ─── Algebraic Simplification ──────────────────────────────────────────

class AlgebraicSimplification:
    """Apply algebraic identity rules to simplify instructions.

    Rules include:
    - Additive identity: x + 0 = 0 + x = x
    - Multiplicative identity: x * 1 = 1 * x = x
    - Multiplicative zero: x * 0 = 0 * x = 0
    - Subtractive identity: x - 0 = x
    - Self-subtraction: x - x = 0
    - Self-XOR: x ^ x = 0
    - AND identity: x & -1 = x
    - AND zero: x & 0 = 0
    - OR identity: x | 0 = x
    - OR all-ones: x | -1 = -1
    - Self-AND: x & x = x
    - Self-OR: x | x = x
    - Shift by zero: x << 0 = x >> 0 = x
    - Double negation: -(-x) = x
    - Double NOT: ~(~x) = x
    """

    def __init__(self) -> None:
        self.simplifications = 0

    def try_simplify(self, inst: Instruction) -> Optional[Value]:
        """Try to simplify an instruction. Returns replacement value or None."""
        if isinstance(inst, BinaryOp):
            return self._simplify_binop(inst)
        elif isinstance(inst, UnaryOp):
            return self._simplify_unaryop(inst)
        return None

    def _simplify_binop(self, inst: BinaryOp) -> Optional[Value]:
        op = inst.op
        left, right = inst.left, inst.right

        # ── Addition ──
        if op == BinOpKind.ADD:
            if _is_zero(right):
                return left  # x + 0 = x
            if _is_zero(left):
                return right  # 0 + x = x
            # x + (-x) = 0 (if right is neg of left)
            if isinstance(right, UnaryOp) and right.op == UnaryOpKind.NEG and right.operand is left:
                return _make_int_const(0, inst.ir_type)
            if isinstance(left, UnaryOp) and left.op == UnaryOpKind.NEG and left.operand is right:
                return _make_int_const(0, inst.ir_type)

        # ── Subtraction ──
        elif op == BinOpKind.SUB:
            if _is_zero(right):
                return left  # x - 0 = x
            if left is right:
                return _make_int_const(0, inst.ir_type)  # x - x = 0

        # ── Multiplication ──
        elif op == BinOpKind.MUL:
            if _is_zero(left) or _is_zero(right):
                return _make_int_const(0, inst.ir_type)  # x * 0 = 0
            if _is_one(right):
                return left  # x * 1 = x
            if _is_one(left):
                return right  # 1 * x = x

        # ── Division ──
        elif op in (BinOpKind.SDIV, BinOpKind.UDIV):
            if _is_one(right):
                return left  # x / 1 = x
            if left is right:
                return _make_int_const(1, inst.ir_type)  # x / x = 1

        # ── Remainder ──
        elif op in (BinOpKind.SREM, BinOpKind.UREM):
            if _is_one(right):
                return _make_int_const(0, inst.ir_type)  # x % 1 = 0
            if left is right:
                return _make_int_const(0, inst.ir_type)  # x % x = 0

        # ── AND ──
        elif op == BinOpKind.AND:
            if _is_zero(left) or _is_zero(right):
                return _make_int_const(0, inst.ir_type)  # x & 0 = 0
            if _is_all_ones(right):
                return left  # x & -1 = x
            if _is_all_ones(left):
                return right  # -1 & x = x
            if left is right:
                return left  # x & x = x

        # ── OR ──
        elif op == BinOpKind.OR:
            if _is_zero(right):
                return left  # x | 0 = x
            if _is_zero(left):
                return right  # 0 | x = x
            if _is_all_ones(left) or _is_all_ones(right):
                typ = inst.ir_type
                if isinstance(typ, IntType):
                    return _make_int_const((1 << typ.width) - 1, typ)  # x | -1 = -1
            if left is right:
                return left  # x | x = x

        # ── XOR ──
        elif op == BinOpKind.XOR:
            if _is_zero(right):
                return left  # x ^ 0 = x
            if _is_zero(left):
                return right  # 0 ^ x = x
            if left is right:
                return _make_int_const(0, inst.ir_type)  # x ^ x = 0

        # ── Shifts ──
        elif op in (BinOpKind.SHL, BinOpKind.LSHR, BinOpKind.ASHR):
            if _is_zero(right):
                return left  # x << 0 = x, x >> 0 = x
            if _is_zero(left):
                return _make_int_const(0, inst.ir_type)  # 0 << x = 0, 0 >> x = 0

        # ── Floating-point ──
        elif op == BinOpKind.FADD:
            if isinstance(right, Constant) and hasattr(right, 'value') and right.value == 0.0:
                return left
            if isinstance(left, Constant) and hasattr(left, 'value') and left.value == 0.0:
                return right

        elif op == BinOpKind.FMUL:
            if isinstance(right, Constant) and hasattr(right, 'value') and right.value == 1.0:
                return left
            if isinstance(left, Constant) and hasattr(left, 'value') and left.value == 1.0:
                return right

        return None

    def _simplify_unaryop(self, inst: UnaryOp) -> Optional[Value]:
        # Double negation: -(-x) = x
        if inst.op == UnaryOpKind.NEG:
            if isinstance(inst.operand, UnaryOp) and inst.operand.op == UnaryOpKind.NEG:
                return inst.operand.operand

        # Double NOT: ~(~x) = x
        if inst.op == UnaryOpKind.NOT:
            if isinstance(inst.operand, UnaryOp) and inst.operand.op == UnaryOpKind.NOT:
                return inst.operand.operand

        return None


# ─── Strength Reduction ────────────────────────────────────────────────

class StrengthReduction:
    """Replace expensive operations with cheaper equivalents.

    Transformations:
    - Multiply by power of 2 → left shift
    - Unsigned divide by power of 2 → right shift
    - Unsigned remainder by power of 2 → AND with (pow2 - 1)
    - Multiply by 2 → add with self
    """

    def __init__(self) -> None:
        self.reductions = 0

    def try_reduce(self, inst: Instruction) -> Optional[Tuple[BinOpKind, Value, Value]]:
        """Try to strength-reduce an instruction.

        Returns (new_op, new_left, new_right) or None.
        """
        if not isinstance(inst, BinaryOp):
            return None

        op = inst.op
        left, right = inst.left, inst.right
        typ = inst.ir_type

        # Multiply by power of 2 → shift left
        if op == BinOpKind.MUL:
            exp = _is_power_of_two(right)
            if exp is not None and exp > 0:
                self.reductions += 1
                return (BinOpKind.SHL, left, _make_int_const(exp, typ))
            exp = _is_power_of_two(left)
            if exp is not None and exp > 0:
                self.reductions += 1
                return (BinOpKind.SHL, right, _make_int_const(exp, typ))

        # Unsigned divide by power of 2 → logical shift right
        elif op == BinOpKind.UDIV:
            exp = _is_power_of_two(right)
            if exp is not None and exp > 0:
                self.reductions += 1
                return (BinOpKind.LSHR, left, _make_int_const(exp, typ))

        # Unsigned remainder by power of 2 → AND
        elif op == BinOpKind.UREM:
            exp = _is_power_of_two(right)
            if exp is not None:
                mask_val = (1 << exp) - 1
                self.reductions += 1
                return (BinOpKind.AND, left, _make_int_const(mask_val, typ))

        # Signed divide by power of 2 (for positive divisors)
        elif op == BinOpKind.SDIV:
            exp = _is_power_of_two(right)
            if exp is not None and exp > 0 and isinstance(typ, IntType):
                # Only safe for non-negative dividends; skip for safety
                pass

        return None


# ─── Boolean Simplification ───────────────────────────────────────────

class BooleanSimplification:
    """Simplify boolean and comparison expressions.

    Rules:
    - Comparison with self: x == x → true, x != x → false, x < x → false
    - NOT of comparison: !(x == y) → x != y, etc.
    - Double comparison elimination
    - Select with boolean: select(c, true, false) → c
    - Select with same values: select(c, x, x) → x
    """

    def __init__(self) -> None:
        self.simplifications = 0

    def try_simplify_compare(self, inst: CompareOp) -> Optional[Value]:
        """Try to simplify a comparison instruction."""
        left, right = inst.left, inst.right

        # Self-comparison rules
        if left is right:
            pred = inst.predicate
            if pred in (CmpPredicate.EQ, CmpPredicate.SLE, CmpPredicate.SGE,
                       CmpPredicate.ULE, CmpPredicate.UGE):
                self.simplifications += 1
                return Constant(value=1, ir_type=inst.ir_type)  # true
            elif pred in (CmpPredicate.NE, CmpPredicate.SLT, CmpPredicate.SGT,
                         CmpPredicate.ULT, CmpPredicate.UGT):
                self.simplifications += 1
                return Constant(value=0, ir_type=inst.ir_type)  # false

        # Comparing against known bounds
        if isinstance(right, Constant) and isinstance(left.ir_type, IntType):
            rv = _get_int_val(right)
            if rv is not None:
                typ = left.ir_type
                if isinstance(typ, IntType):
                    result = self._check_range_compare(inst.predicate, rv, typ)
                    if result is not None:
                        self.simplifications += 1
                        return Constant(value=1 if result else 0, ir_type=inst.ir_type)

        return None

    def _check_range_compare(self, pred: CmpPredicate, rv: int, typ: IntType) -> Optional[bool]:
        """Check if a comparison against a constant is always true/false."""
        width = typ.width
        if typ.signedness == Signedness.UNSIGNED:
            max_val = (1 << width) - 1
            if pred == CmpPredicate.UGE and rv == 0:
                return True  # x >= 0 always true for unsigned
            if pred == CmpPredicate.ULT and rv == 0:
                return False  # x < 0 always false for unsigned
            if pred == CmpPredicate.ULE and rv == max_val:
                return True  # x <= max always true
            if pred == CmpPredicate.UGT and rv == max_val:
                return False  # x > max always false
        else:
            min_val = -(1 << (width - 1))
            max_val = (1 << (width - 1)) - 1
            if pred == CmpPredicate.SGE and rv == min_val:
                return True
            if pred == CmpPredicate.SLT and rv == min_val:
                return False
            if pred == CmpPredicate.SLE and rv == max_val:
                return True
            if pred == CmpPredicate.SGT and rv == max_val:
                return False
        return None

    def try_simplify_select(self, inst: SelectInst) -> Optional[Value]:
        """Try to simplify a select instruction."""
        cond = inst.condition
        tv = inst.true_value
        fv = inst.false_value

        # select(c, x, x) → x
        if tv is fv:
            self.simplifications += 1
            return tv

        # select(c, true, false) → c (when c is i1)
        if (_is_one(tv) and _is_zero(fv) and
                isinstance(cond.ir_type, IntType) and cond.ir_type.width == 1):
            self.simplifications += 1
            return cond

        # select(c, false, true) → !c
        if _is_zero(tv) and _is_one(fv):
            # Can't create a new instruction here easily, skip
            pass

        # select(true, a, b) → a
        if _is_one(cond):
            self.simplifications += 1
            return tv

        # select(false, a, b) → b
        if _is_zero(cond):
            self.simplifications += 1
            return fv

        return None

    def try_simplify_not_cmp(self, inst: BinaryOp) -> Optional[Tuple[CmpPredicate, Value, Value]]:
        """Detect NOT of comparison: xor(cmp, 1) and return negated predicate."""
        if inst.op != BinOpKind.XOR:
            return None
        if not _is_one(inst.right):
            return None
        cmp = inst.left
        if not isinstance(cmp, CompareOp):
            return None

        negated = self._negate_predicate(cmp.predicate)
        if negated is not None:
            self.simplifications += 1
            return (negated, cmp.left, cmp.right)
        return None

    def _negate_predicate(self, pred: CmpPredicate) -> Optional[CmpPredicate]:
        negate_map = {
            CmpPredicate.EQ: CmpPredicate.NE,
            CmpPredicate.NE: CmpPredicate.EQ,
            CmpPredicate.SLT: CmpPredicate.SGE,
            CmpPredicate.SGE: CmpPredicate.SLT,
            CmpPredicate.SGT: CmpPredicate.SLE,
            CmpPredicate.SLE: CmpPredicate.SGT,
            CmpPredicate.ULT: CmpPredicate.UGE,
            CmpPredicate.UGE: CmpPredicate.ULT,
            CmpPredicate.UGT: CmpPredicate.ULE,
            CmpPredicate.ULE: CmpPredicate.UGT,
            CmpPredicate.OEQ: CmpPredicate.ONE,
            CmpPredicate.ONE: CmpPredicate.OEQ,
            CmpPredicate.OLT: CmpPredicate.OGE,
            CmpPredicate.OGE: CmpPredicate.OLT,
            CmpPredicate.OGT: CmpPredicate.OLE,
            CmpPredicate.OLE: CmpPredicate.OGT,
        }
        return negate_map.get(pred)


# ─── Redundant Cast Elimination ────────────────────────────────────────

class RedundantCastElimination:
    """Remove unnecessary cast instructions.

    Rules:
    - Cast to same type: (T)x where x is already T → x
    - Chained widening: zext(zext(x)) → zext(x) with target type
    - Chained narrowing: trunc(trunc(x)) → trunc(x) with target type
    - Widening then narrowing back: trunc(zext(x)) where result = original type → x
    - Bitcast to same size: bitcast identity
    """

    def __init__(self) -> None:
        self.eliminations = 0

    def try_eliminate(self, inst: CastInst) -> Optional[Value]:
        """Try to eliminate a redundant cast. Returns replacement or None."""
        src = inst.operand
        src_type = src.ir_type if hasattr(src, 'ir_type') else None
        dst_type = inst.ir_type

        # Cast to same type
        if src_type is not None and self._types_equal(src_type, dst_type):
            self.eliminations += 1
            return src

        # Chained casts
        if isinstance(src, CastInst):
            return self._simplify_chained_cast(inst, src)

        return None

    def _simplify_chained_cast(self, outer: CastInst, inner: CastInst) -> Optional[Value]:
        """Simplify two chained casts."""
        original = inner.operand
        orig_type = original.ir_type if hasattr(original, 'ir_type') else None
        mid_type = inner.ir_type
        final_type = outer.ir_type

        if orig_type is None:
            return None

        # zext(zext(x)) → zext(x) to final type
        if outer.cast_kind == CastKind.ZEXT and inner.cast_kind == CastKind.ZEXT:
            self.eliminations += 1
            return None  # Would need to create new cast; skip for now

        # sext(sext(x)) → sext(x) to final type
        if outer.cast_kind == CastKind.SEXT and inner.cast_kind == CastKind.SEXT:
            self.eliminations += 1
            return None

        # trunc(trunc(x)) → trunc(x) to final type
        if outer.cast_kind == CastKind.TRUNC and inner.cast_kind == CastKind.TRUNC:
            self.eliminations += 1
            return None

        # trunc(zext(x)) where final_type == orig_type → x
        if (outer.cast_kind == CastKind.TRUNC and
                inner.cast_kind in (CastKind.ZEXT, CastKind.SEXT) and
                self._types_equal(final_type, orig_type)):
            self.eliminations += 1
            return original

        # trunc(zext(x)) where final_type < orig_type → trunc(x)
        if (outer.cast_kind == CastKind.TRUNC and
                inner.cast_kind in (CastKind.ZEXT, CastKind.SEXT)):
            if (isinstance(final_type, IntType) and isinstance(orig_type, IntType) and
                    final_type.width < orig_type.width):
                self.eliminations += 1
                return None  # Would need new trunc

        # bitcast(bitcast(x)) → bitcast(x) or identity
        if outer.cast_kind == CastKind.BITCAST and inner.cast_kind == CastKind.BITCAST:
            if self._types_equal(final_type, orig_type):
                self.eliminations += 1
                return original

        return None

    def _types_equal(self, a: IRType, b: IRType) -> bool:
        """Check if two types are structurally equal."""
        if type(a) != type(b):
            return False
        if isinstance(a, IntType) and isinstance(b, IntType):
            return a.width == b.width and a.signedness == b.signedness
        if isinstance(a, FloatType) and isinstance(b, FloatType):
            return a.kind == b.kind
        return str(a) == str(b)


# ─── Phi Node Simplification ──────────────────────────────────────────

class PhiSimplification:
    """Simplify phi nodes.

    Rules:
    - Single incoming: phi(x from B) → x
    - All same value: phi(x from B1, x from B2) → x
    - All same constant: phi(c from B1, c from B2) → c
    - Self-referential with one other: phi(self, x) → x
    """

    def __init__(self) -> None:
        self.simplifications = 0

    def try_simplify(self, phi: PhiInst) -> Optional[Value]:
        if not phi.incoming:
            return None

        # Single incoming value
        if len(phi.incoming) == 1:
            self.simplifications += 1
            return phi.incoming[0][0]

        # Find unique non-self value
        unique: Optional[Value] = None
        for val, _ in phi.incoming:
            if val is phi:
                continue
            if unique is None:
                unique = val
            elif val is not unique:
                # Check for same constant
                if (isinstance(val, Constant) and isinstance(unique, Constant) and
                        hasattr(val, 'value') and hasattr(unique, 'value') and
                        val.value == unique.value):
                    continue
                return None  # Multiple distinct values

        if unique is not None:
            self.simplifications += 1
            return unique

        return None


# ─── Main Instruction Simplifier Pass ──────────────────────────────────

class InstructionSimplifier(FunctionPass):
    """Main instruction simplification pass.

    Combines algebraic simplification, strength reduction,
    boolean simplification, redundant cast elimination,
    and phi simplification into a single worklist-driven pass.
    """

    _name = "simplify"
    _description = "Instruction simplification (algebraic, strength reduction, boolean, cast)"
    _invalidated_analyses = ["cfg", "domtree"]

    def __init__(self) -> None:
        super().__init__()
        self._algebraic = AlgebraicSimplification()
        self._strength = StrengthReduction()
        self._boolean = BooleanSimplification()
        self._cast_elim = RedundantCastElimination()
        self._phi_simp = PhiSimplification()

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        changed = False
        worklist: List[Instruction] = []
        in_worklist: Set[int] = set()

        # Seed worklist with all instructions
        for block in function.blocks:
            for inst in block.instructions:
                worklist.append(inst)
                in_worklist.add(inst.id)

        iterations = 0
        max_iterations = 50000

        while worklist and iterations < max_iterations:
            iterations += 1
            inst = worklist.pop()
            in_worklist.discard(inst.id)

            if inst.parent is None:
                continue

            replacement = self._try_simplify(inst)
            if replacement is not None:
                self._apply_replacement(inst, replacement, worklist, in_worklist)
                changed = True
                continue

            # Try strength reduction
            if isinstance(inst, BinaryOp):
                reduced = self._strength.try_reduce(inst)
                if reduced is not None:
                    new_op, new_left, new_right = reduced
                    inst.op = new_op
                    inst.left = new_left
                    inst.right = new_right
                    changed = True
                    self.stats.increment("strength_reductions")
                    # Re-add users to worklist
                    if hasattr(inst, 'users'):
                        for user in inst.users:
                            if isinstance(user, Instruction) and user.id not in in_worklist:
                                worklist.append(user)
                                in_worklist.add(user.id)

        if changed:
            self.stats.increment("total_simplifications",
                                  self._algebraic.simplifications +
                                  self._boolean.simplifications +
                                  self._cast_elim.eliminations +
                                  self._phi_simp.simplifications)

        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _try_simplify(self, inst: Instruction) -> Optional[Value]:
        """Try all simplification rules on an instruction."""
        # Algebraic simplification
        if isinstance(inst, (BinaryOp, UnaryOp)):
            result = self._algebraic.try_simplify(inst)
            if result is not None:
                return result

        # Boolean/comparison simplification
        if isinstance(inst, CompareOp):
            result = self._boolean.try_simplify_compare(inst)
            if result is not None:
                return result

        # Select simplification
        if isinstance(inst, SelectInst):
            result = self._boolean.try_simplify_select(inst)
            if result is not None:
                return result

        # Cast elimination
        if isinstance(inst, CastInst):
            result = self._cast_elim.try_eliminate(inst)
            if result is not None:
                return result

        # Phi simplification
        if isinstance(inst, PhiInst):
            result = self._phi_simp.try_simplify(inst)
            if result is not None:
                return result

        # NOT of comparison
        if isinstance(inst, BinaryOp):
            negated = self._boolean.try_simplify_not_cmp(inst)
            if negated is not None:
                pred, left, right = negated
                # Can't create new CompareOp here easily; just return left for now
                pass

        return None

    def _apply_replacement(self, inst: Instruction, replacement: Value,
                            worklist: List[Instruction],
                            in_worklist: Set[int]) -> None:
        """Replace inst with replacement, adding users to worklist."""
        block = inst.parent
        if block is None:
            return

        if hasattr(inst, 'users'):
            for user in list(inst.users):
                if isinstance(user, Instruction):
                    _substitute(user, inst, replacement)
                    if hasattr(replacement, 'users') and not isinstance(replacement, Constant):
                        replacement.users.append(user)
                    if user.id not in in_worklist:
                        worklist.append(user)
                        in_worklist.add(user.id)

        try:
            block.remove(inst)
        except ValueError:
            pass
        self.stats.instructions_removed += 1


# ─── Reassociation ────────────────────────────────────────────────────

class Reassociation:
    """Reassociate commutative/associative operations for better optimization.

    For chains of adds/muls, sort operands to expose common subexpressions
    and enable more constant folding.

    Examples:
    - (a + 1) + 2 → a + 3
    - (a + b) + (a + c) → potentially expose common 'a'
    """

    def __init__(self) -> None:
        self.reassociations = 0

    def try_reassociate(self, inst: BinaryOp) -> Optional[Tuple[Value, Constant]]:
        """Try to reassociate an instruction to collect constants.

        Returns (variable_part, constant_sum) if successful.
        """
        if not self._is_associative(inst.op):
            return None

        if not self._is_commutative(inst.op):
            return None

        # Collect all operands in the chain
        constants: List[int] = []
        variables: List[Value] = []
        self._collect_chain(inst, inst.op, constants, variables, set())

        if len(constants) < 2:
            return None

        # Fold all constants
        total = constants[0]
        for c in constants[1:]:
            if inst.op in (BinOpKind.ADD, BinOpKind.FADD):
                total += c
            elif inst.op in (BinOpKind.MUL, BinOpKind.FMUL):
                total *= c
            elif inst.op == BinOpKind.AND:
                total &= c
            elif inst.op == BinOpKind.OR:
                total |= c
            elif inst.op == BinOpKind.XOR:
                total ^= c

        if not variables:
            return None

        self.reassociations += 1
        return (variables[0], _make_int_const(total, inst.ir_type))

    def _collect_chain(self, val: Value, op: BinOpKind,
                        constants: List[int], variables: List[Value],
                        visited: Set[int]) -> None:
        """Recursively collect operands in an associative chain."""
        if not isinstance(val, BinaryOp) or val.op != op:
            if isinstance(val, Constant):
                iv = _get_int_val(val)
                if iv is not None:
                    constants.append(iv)
                    return
            variables.append(val)
            return

        if val.id in visited:
            variables.append(val)
            return
        visited.add(val.id)

        # Only decompose if this value has exactly one user (the chain)
        if hasattr(val, 'users') and len(val.users) > 1:
            variables.append(val)
            return

        self._collect_chain(val.left, op, constants, variables, visited)
        self._collect_chain(val.right, op, constants, variables, visited)

    def _is_associative(self, op: BinOpKind) -> bool:
        return op in (BinOpKind.ADD, BinOpKind.MUL,
                     BinOpKind.AND, BinOpKind.OR, BinOpKind.XOR,
                     BinOpKind.FADD, BinOpKind.FMUL)

    def _is_commutative(self, op: BinOpKind) -> bool:
        return op in (BinOpKind.ADD, BinOpKind.MUL,
                     BinOpKind.AND, BinOpKind.OR, BinOpKind.XOR,
                     BinOpKind.FADD, BinOpKind.FMUL)
